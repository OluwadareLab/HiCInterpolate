from src.loss import CombinedLoss, ExponentialDecay
from src.metric import metrics as eval_metric
from src.misc import plots as plot
from src.interpolator import Interpolator, model
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import torch
import torch.distributed as dist
import time
import gc
import pandas as pd
import traceback
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    def __init__(self, cfg, log, train_dl: DataLoader, val_dl: DataLoader, load_snapshot: bool = False, isDistributed: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.train_dl = train_dl
        self.train_steps = len(self.train_dl)
        self.batch_size = train_dl.batch_size
        self.val_dl = val_dl
        self.val_steps = len(self.val_dl)
        self.save_every = self.cfg.training.save_every

        self.model = Interpolator(self.cfg)

        self.isDistributed = isDistributed and dist.is_available() and dist.is_initialized()
        if self.isDistributed:
            self.device = int(os.environ.get('LOCAL_RANK', 0))
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
        else:
            self.device = self.cfg.device
            self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        log_txt = f"[INFO] Total parameters: {total_params / 1e6:.2f} Millions | Trainable parameters: {trainable_params / 1e6:.2f} Millions"
        print(log_txt)
        self.log.info(log_txt)

        self.loss_fn = CombinedLoss(self.cfg)
        # self.optimizer = optim.Adam(self.model.parameters(),
        #                             lr=self.cfg.training.lr)

        # decay_rate = (self.cfg.training.min_lr / self.cfg.training.lr) ** (1 /
        #                                                                    (self.cfg.training.epochs // self.cfg.training.decay_steps))
        # print(f"[DEBUG] Calculated decay_rate: {decay_rate:.4f}")
        # self.log.debug(f"Calculated decay_rate: {decay_rate:.4f}")
        # self.scheduler = ExponentialDecay(optimizer=self.optimizer, decay_steps=self.cfg.training.decay_steps,
        #                                   decay_rate=decay_rate, staircase=self.cfg.training.lr_staircase)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='max', factor=0.5, patience=10)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,              # Peak learning rate baseline
            betas=(0.9, 0.999),    # Standard momentum coefficients
            weight_decay=1e-4,    # Decoupled L2 regularization to prevent overfitting
            eps=1e-8
        )
        total_iterations = len(self.train_dl) * self.cfg.training.epochs
        warmup_iterations = len(self.train_dl) * 10
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=warmup_iterations, eta_min=1e-6)

        self.epochs_run = 0
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_ssim_per_epoch = 0
        self.val_genome_disco_per_epoch = 0
        self.val_hicrep_per_epoch = 0
        self.val_score_per_epoch = 0

        self.state = {'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [
        ], 'val_ssim': [], 'val_genome_disco': [], 'val_hicrep': [], 'best_val': []}
        self.metric_columns = ['epoch', 'lr', 'train_loss', 'val_loss',
                               'val_ssim', 'val_genome_disco', 'val_hicrep', 'best_val']
        self.patience = 40
        self.epochs_no_improve = 0
        self.best_val = -float('inf')
        self.best_model = f'{self.cfg.file.model}'
        self.best_plot_batch = None

        self.snapshot = f'{self.cfg.file.snapshot}'
        if load_snapshot and os.path.exists(self.snapshot):
            self.log.info(f"Loading snapshot...")
            print(f"[INFO] Loading snapshot...")
            self._load_snapshot(self.snapshot)

    def _remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict

    def _load_snapshot(self, snapshot):
        if self.isDistributed:
            loc = f"cuda:{self.device}"
            snapshot = torch.load(snapshot, map_location=loc)
            self.epochs_run = snapshot['epoch']
            self.model.load_state_dict(snapshot['model'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.scheduler.load_state_dict(snapshot['scheduler'])
            self.state = snapshot['state']
            best_val_list = snapshot['state'].get('best_val', [])
            self.best_val = best_val_list[-1] if best_val_list else - \
                float('inf')
        else:
            snapshot = torch.load(snapshot, map_location=self.device)
            self.epochs_run = snapshot['epoch']
            model_state_dict = self._remove_module_prefix(snapshot['model'])
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.scheduler.load_state_dict(snapshot['scheduler'])
            self.state = snapshot['state']
            best_val_list = self.state.get('best_val', [])
            self.best_val = best_val_list[-1] if best_val_list else - \
                float('inf')
        self.log.info(
            f"Resuming training from snapshot at epoch {self.epochs_run}")
        print(
            f"[INFO] Resuming training from snapshot at epoch {self.epochs_run}")

    @staticmethod
    def _safe_average(total, count):
        return total / count if count else 0.0

    @staticmethod
    def _validation_score(ssim, genome_disco, hicrep):
        return 0.4 * ssim + 0.15 * genome_disco + 0.45 * hicrep

    def _update_metrics(self, epoch, train_samples, train_loss, val_samples, val_loss, val_ssim, val_genome_disco, val_hicrep):
        self.train_loss_per_epoch = self._safe_average(
            train_loss, train_samples)
        self.val_loss_per_epoch = self._safe_average(val_loss, val_samples)
        self.val_ssim_per_epoch = self._safe_average(val_ssim, val_samples)
        self.val_genome_disco_per_epoch = self._safe_average(
            val_genome_disco, val_samples)
        self.val_hicrep_per_epoch = self._safe_average(val_hicrep, val_samples)
        self.val_score_per_epoch = self._validation_score(
            self.val_ssim_per_epoch,
            self.val_genome_disco_per_epoch,
            self.val_hicrep_per_epoch,
        )

        self.state['epoch'].append(epoch+1)
        self.state['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.state['train_loss'].append(self.train_loss_per_epoch)
        self.state['val_loss'].append(self.val_loss_per_epoch)
        self.state['val_ssim'].append(self.val_ssim_per_epoch)
        self.state['val_genome_disco'].append(self.val_genome_disco_per_epoch)
        self.state['val_hicrep'].append(self.val_hicrep_per_epoch)

    def _is_main_process(self):
        return (not self.isDistributed) or self.device == 0

    def _get_model_stats(self, epoch: int):
        snapshot = {
            'epoch': epoch+1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'state': self.state
        }
        return snapshot

    def _save_snapshot(self, epoch: int):
        snapshot = self._get_model_stats(epoch)
        torch.save(snapshot, self.snapshot)
        print(f"[DEBUG] Epoch {epoch+1} saved snapshot at {self.snapshot}")

    def _save_best_model(self, epoch: int, save_artifacts: bool = True):
        best_val = self.val_score_per_epoch

        if best_val > self.best_val:
            self.epochs_no_improve = 0
            self.best_val = best_val
            if save_artifacts:
                snapshot = self._get_model_stats(epoch)
                torch.save(snapshot, self.best_model)
                if self.best_plot_batch is not None:
                    plot_file = os.path.join(
                        self.cfg.dir.model_state, f"epoch_{epoch+1}_output.png")
                    plot.draw_hic_map(2, *self.best_plot_batch, plot_file)
                self.log.info(
                    f"Epoch {self.epochs_run+1} saved best model.")
                print(
                    f"[DEBUG] Epoch {self.epochs_run+1} saved best model.")
        elif self.epochs_run > 10:
            self.epochs_no_improve += 1

    def _save_and_draw_metrics(self):
        metrics_df = pd.DataFrame({
            'epoch': self.state["epoch"],  # Usually an int, no rounding needed
            'lr': self.state["lr"],
            'train_loss': self.state["train_loss"],
            'val_loss': self.state["val_loss"],
            'val_ssim': self.state["val_ssim"],
            'val_genome_disco': self.state["val_genome_disco"],
            'val_hicrep': self.state["val_hicrep"],
            'best_val': self.state["best_val"]
        }, columns=self.metric_columns)

        metrics_to_round = ['train_loss', 'val_loss', 'val_ssim',
                            'val_genome_disco', 'val_hicrep', 'best_val']
        metrics_df[metrics_to_round] = metrics_df[metrics_to_round].round(4)
        if 'lr' in metrics_df.columns:
            metrics_df['lr'] = metrics_df['lr'].round(6)

        metrics_df.to_csv(self.cfg.file.val_metrics, index=False)
        plot.draw_metric(self.cfg, self.state)

    def _format_scores(self, max_epochs: int):
        return (
            f"[{(self.epochs_run+1)}/{max_epochs}] LR: {self.optimizer.param_groups[0]['lr']:.6f}; "
            f"Batch Size: {self.batch_size}; Train Loss: {self.train_loss_per_epoch:.4f}; "
            f"Val (Loss: {self.val_loss_per_epoch:.4f}, SSIM: {self.val_ssim_per_epoch:.4f}, "
            f"GenomeDISCO: {self.val_genome_disco_per_epoch:.4f}, HiCRep: {self.val_hicrep_per_epoch:.4f}, "
            f"Score: {self.val_score_per_epoch:.4f});"
        )

    def _run_epoch(self, epoch):
        self.epochs_run = epoch
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_ssim_per_epoch = 0
        self.val_genome_disco_per_epoch = 0
        self.val_hicrep_per_epoch = 0
        self.val_score_per_epoch = 0

        self.model.train()
        if self.isDistributed and hasattr(self.train_dl.sampler, "set_epoch"):
            self.train_dl.sampler.set_epoch(epoch)

        local_train_loss = torch.tensor(0.0, device=self.device)
        local_train_samples = torch.tensor(0.0, device=self.device)

        for step, (x0, y, x1, time_frame) in enumerate(tqdm(self.train_dl)):
            x0 = x0.to(self.device)
            y = y.to(self.device)
            x1 = x1.to(self.device)
            time_frame = time_frame.to(self.device)
            self.optimizer.zero_grad()

            batch_size = y.size(0)
            pred = self.model(x0, x1, time_frame)
            train_loss = self.loss_fn(pred, y, self.epochs_run)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            local_train_loss += train_loss.detach() * batch_size
            local_train_samples += batch_size

            del x0, y, x1, time_frame, pred

        local_val_loss = torch.tensor(0.0, device=self.device)
        local_val_ssim = torch.tensor(0.0, device=self.device)
        local_val_genome_disco = torch.tensor(0.0, device=self.device)
        local_val_hicrep = torch.tensor(0.0, device=self.device)
        local_val_samples = torch.tensor(0.0, device=self.device)
        self.best_plot_batch = None

        with torch.no_grad():
            self.model.eval()
            for _, (x0, y, x1, time_frame) in enumerate(self.val_dl):
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)
                time_frame = time_frame.to(self.device)

                batch_size = y.size(0)
                pred = self.model(x0, x1, time_frame)
                val_loss = self.loss_fn(pred, y, self.epochs_run)

                local_val_loss += val_loss.detach() * batch_size

                ssim_val = eval_metric.get_ssim_gpu(pred, y)
                genome_disco_val = eval_metric.get_genome_disco_gpu(pred, y)
                hicrep_val = eval_metric.get_hicrep_gpu(pred, y)

                local_val_ssim += ssim_val * batch_size
                local_val_genome_disco += genome_disco_val * batch_size
                local_val_hicrep += hicrep_val * batch_size
                local_val_samples += batch_size

                if self.best_plot_batch is None:
                    self.best_plot_batch = (
                        x0[:2].detach().cpu(),
                        y[:2].detach().cpu(),
                        pred[:2].detach().cpu(),
                        x1[:2].detach().cpu(),
                    )

                del x0, y, x1, time_frame, pred

        if self.isDistributed:
            for value in (
                local_train_samples,
                local_val_samples,
                local_train_loss,
                local_val_loss,
                local_val_ssim,
                local_val_genome_disco,
                local_val_hicrep,
            ):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        self._update_metrics(
            self.epochs_run,
            local_train_samples.item(),
            local_train_loss.item(),
            local_val_samples.item(),
            local_val_loss.item(),
            local_val_ssim.item(),
            local_val_genome_disco.item(),
            local_val_hicrep.item(),
        )

    def train(self, max_epochs: int):
        self.log.info(f"==== Training Started ({self.device}) ====")
        print(f"[INFO] ==== Training Started ({self.device}) ====")

        start_time = time.time()
        try:
            for epoch in range(self.epochs_run, max_epochs):
                if self.epochs_no_improve > self.patience:
                    self.log.info(
                        f"No improvement in last {self.patience} epochs! Stopping early.")
                    print(
                        f"No improvement in last {self.patience} epochs! Stopping early.")
                    break

                self._run_epoch(epoch)
                self._save_best_model(
                    epoch, save_artifacts=self._is_main_process())

                if self._is_main_process():
                    self.state["best_val"].append(self.best_val)
                    if (self.epochs_run+1) % self.save_every == 0:
                        self._save_snapshot(epoch)
                    self._save_and_draw_metrics()
                    scores = self._format_scores(max_epochs)

                    self.log.info(f"{scores}")
                    print(f"[INFO] {scores}")

        except Exception as ex:
            print(ex)
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        end_time = time.time()
        self.log.info(
            f"Total time taken: {format((end_time-start_time), '.2f')} seconds")
        print(
            f"[INFO] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        self.log.info(f"==== Training End ====")
        print(f"[INFO] ==== Training End ====")
