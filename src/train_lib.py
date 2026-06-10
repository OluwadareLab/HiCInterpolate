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
import torch.nn.functional as F
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

        decay_params, no_decay_params = self._split_weight_decay_params(self.model)
        self.optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.cfg.training.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.training.lr,
            betas=(0.9, 0.99),
            eps=1e-8
        )
        total_iterations = len(self.train_dl) * self.cfg.training.epochs
        warmup_iterations = len(self.train_dl) * self.cfg.training.warmup_epochs
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_iterations)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_iterations - warmup_iterations),
            eta_min=self.cfg.training.min_lr)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_iterations])

        self.epochs_run = 0
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_sparse_precision_per_epoch = 0
        self.val_sparse_recall_per_epoch = 0
        self.val_sparse_f1_per_epoch = 0
        self.val_pred_density_per_epoch = 0
        self.val_target_density_per_epoch = 0
        self.val_density_error_per_epoch = 0
        self.val_nonzero_mae_per_epoch = 0
        self.val_zero_mae_per_epoch = 0
        self.val_score_per_epoch = 0

        self.state = {'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [],
                      'val_sparse_precision': [], 'val_sparse_recall': [], 'val_sparse_f1': [],
                      'val_pred_density': [], 'val_target_density': [], 'val_density_error': [],
                      'val_nonzero_mae': [], 'val_zero_mae': [],
                      'val_score': [], 'best_val': []}
        self.metric_columns = ['epoch', 'lr', 'train_loss', 'val_loss',
                               'val_sparse_precision', 'val_sparse_recall', 'val_sparse_f1',
                               'val_pred_density', 'val_target_density', 'val_density_error',
                               'val_nonzero_mae', 'val_zero_mae', 'val_score', 'best_val']
        self.patience = 200
        self.epochs_no_improve = 0
        self.best_val = -float('inf')
        self.best_model = f'{self.cfg.file.model}'
        self.best_plot_batch = None

        self.snapshot = f'{self.cfg.file.snapshot}'
        if load_snapshot and os.path.exists(self.snapshot):
            self.log.info(f"Loading snapshot...")
            print(f"[INFO] Loading snapshot...")
            self._load_snapshot(self.snapshot)

    @staticmethod
    def _split_weight_decay_params(model):
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return decay_params, no_decay_params

    def _remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict


    def _ensure_state_keys(self):
        num_epochs = len(self.state.get('epoch', []))
        for key in self.metric_columns:
            if key not in self.state:
                self.state[key] = [-float('inf')] * num_epochs if key == 'best_val' else [0.0] * num_epochs


    def _load_optimizer_scheduler_state(self, snapshot):
        try:
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.scheduler.load_state_dict(snapshot['scheduler'])
        except ValueError as ex:
            msg = f"Optimizer/scheduler state not loaded; using fresh state. Reason: {ex}"
            self.log.info(msg)
            print(f"[INFO] {msg}")

    def _load_snapshot(self, snapshot):
        if self.isDistributed:
            loc = f"cuda:{self.device}"
            snapshot = torch.load(snapshot, map_location=loc)
            self.epochs_run = snapshot['epoch']
            self.model.load_state_dict(snapshot['model'])
            self._load_optimizer_scheduler_state(snapshot)
            self.state = snapshot['state']
            self._ensure_state_keys()
            best_val_list = snapshot['state'].get('best_val', [])
            self.best_val = best_val_list[-1] if best_val_list else - \
                float('inf')
        else:
            snapshot = torch.load(snapshot, map_location=self.device)
            self.epochs_run = snapshot['epoch']
            model_state_dict = self._remove_module_prefix(snapshot['model'])
            self.model.load_state_dict(model_state_dict)
            self._load_optimizer_scheduler_state(snapshot)
            self.state = snapshot['state']
            self._ensure_state_keys()
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
    def _validation_score(sparse_f1, density_error, nonzero_mae, zero_mae):
        return sparse_f1 - density_error - 0.1 * nonzero_mae - 0.1 * zero_mae

    def _update_metrics(self, epoch, train_samples, train_loss, val_samples, val_loss,
                        val_sparse_precision, val_sparse_recall, val_sparse_f1, val_pred_density, val_target_density,
                        val_density_error, val_nonzero_mae, val_zero_mae):
        self.train_loss_per_epoch = self._safe_average(
            train_loss, train_samples)
        self.val_loss_per_epoch = self._safe_average(val_loss, val_samples)
        self.val_sparse_precision_per_epoch = self._safe_average(val_sparse_precision, val_samples)
        self.val_sparse_recall_per_epoch = self._safe_average(val_sparse_recall, val_samples)
        self.val_sparse_f1_per_epoch = self._safe_average(val_sparse_f1, val_samples)
        self.val_pred_density_per_epoch = self._safe_average(val_pred_density, val_samples)
        self.val_target_density_per_epoch = self._safe_average(val_target_density, val_samples)
        self.val_density_error_per_epoch = self._safe_average(val_density_error, val_samples)
        self.val_nonzero_mae_per_epoch = self._safe_average(val_nonzero_mae, val_samples)
        self.val_zero_mae_per_epoch = self._safe_average(val_zero_mae, val_samples)
        self.val_score_per_epoch = self._validation_score(
            self.val_sparse_f1_per_epoch,
            self.val_density_error_per_epoch,
            self.val_nonzero_mae_per_epoch,
            self.val_zero_mae_per_epoch,
        )

        self.state['epoch'].append(epoch+1)
        self.state['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.state['train_loss'].append(self.train_loss_per_epoch)
        self.state['val_loss'].append(self.val_loss_per_epoch)
        self.state['val_sparse_precision'].append(self.val_sparse_precision_per_epoch)
        self.state['val_sparse_recall'].append(self.val_sparse_recall_per_epoch)
        self.state['val_sparse_f1'].append(self.val_sparse_f1_per_epoch)
        self.state['val_pred_density'].append(self.val_pred_density_per_epoch)
        self.state['val_target_density'].append(self.val_target_density_per_epoch)
        self.state['val_density_error'].append(self.val_density_error_per_epoch)
        self.state['val_nonzero_mae'].append(self.val_nonzero_mae_per_epoch)
        self.state['val_zero_mae'].append(self.val_zero_mae_per_epoch)
        self.state['val_score'].append(self.val_score_per_epoch)

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
            'val_sparse_precision': self.state["val_sparse_precision"],
            'val_sparse_recall': self.state["val_sparse_recall"],
            'val_sparse_f1': self.state["val_sparse_f1"],
            'val_pred_density': self.state["val_pred_density"],
            'val_target_density': self.state["val_target_density"],
            'val_density_error': self.state["val_density_error"],
            'val_nonzero_mae': self.state["val_nonzero_mae"],
            'val_zero_mae': self.state["val_zero_mae"],
            'val_score': self.state["val_score"],
            'best_val': self.state["best_val"]
        }, columns=self.metric_columns)

        metrics_to_round = ['train_loss', 'val_loss',
                            'val_sparse_precision', 'val_sparse_recall', 'val_sparse_f1',
                            'val_pred_density', 'val_target_density', 'val_density_error',
                            'val_nonzero_mae', 'val_zero_mae', 'val_score', 'best_val']
        metrics_df[metrics_to_round] = metrics_df[metrics_to_round].round(4)
        if 'lr' in metrics_df.columns:
            metrics_df['lr'] = metrics_df['lr'].round(6)

        metrics_df.to_csv(self.cfg.file.val_metrics, index=False)
        plot.draw_metric(self.cfg, self.state)

    def _format_scores(self, max_epochs: int):
        return (
            f"[{(self.epochs_run+1)}/{max_epochs}] LR: {self.optimizer.param_groups[0]['lr']:.6f}; "
            f"Batch Size: {self.batch_size}; Train Loss: {self.train_loss_per_epoch:.4f}; "
            f"Val (Loss: {self.val_loss_per_epoch:.4f}, SparseF1: {self.val_sparse_f1_per_epoch:.4f}, "
            f"SparseP: {self.val_sparse_precision_per_epoch:.4f}, SparseR: {self.val_sparse_recall_per_epoch:.4f}, "
            f"PredDensity: {self.val_pred_density_per_epoch:.4f}, TargetDensity: {self.val_target_density_per_epoch:.4f}, "
            f"DensityErr: {self.val_density_error_per_epoch:.4f}, "
            f"NZ-MAE: {self.val_nonzero_mae_per_epoch:.4f}, Z-MAE: {self.val_zero_mae_per_epoch:.4f}, "
            f"Score: {self.val_score_per_epoch:.4f});"
        )

    def _run_epoch(self, epoch):
        self.epochs_run = epoch
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_sparse_precision_per_epoch = 0
        self.val_sparse_recall_per_epoch = 0
        self.val_sparse_f1_per_epoch = 0
        self.val_pred_density_per_epoch = 0
        self.val_target_density_per_epoch = 0
        self.val_density_error_per_epoch = 0
        self.val_nonzero_mae_per_epoch = 0
        self.val_zero_mae_per_epoch = 0
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
            outputs = self.model(x0, x1, time_frame, target=y)
            pred = outputs["final"]
            pred_mask_logits = outputs["mask_logits"]
            gt_mask = (y > 0).float()

            train_loss = self.loss_fn(
                pred, y, self.epochs_run, pred_mask=pred_mask_logits, gt_mask=gt_mask,
                diffusion_noise_pred=outputs.get("diffusion_noise_pred"),
                diffusion_noise_target=outputs.get("diffusion_noise_target"),
                diffusion_mask=outputs.get("diffusion_mask"))

            if not torch.isfinite(train_loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.training.grad_clip)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.step()
            self.scheduler.step()

            local_train_loss += train_loss.detach() * batch_size
            local_train_samples += batch_size

            del x0, y, x1, time_frame, pred

        local_val_loss = torch.tensor(0.0, device=self.device)
        local_val_sparse_precision = torch.tensor(0.0, device=self.device)
        local_val_sparse_recall = torch.tensor(0.0, device=self.device)
        local_val_sparse_f1 = torch.tensor(0.0, device=self.device)
        local_val_pred_density = torch.tensor(0.0, device=self.device)
        local_val_target_density = torch.tensor(0.0, device=self.device)
        local_val_density_error = torch.tensor(0.0, device=self.device)
        local_val_nonzero_mae = torch.tensor(0.0, device=self.device)
        local_val_zero_mae = torch.tensor(0.0, device=self.device)
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
                outputs = self.model(x0, x1, time_frame, target=y)

                pred = outputs["final"]
                pred_mask_logits = outputs["mask_logits"]
                gt_mask = (y > 0).float()

                val_loss = self.loss_fn(
                    pred, y, self.epochs_run, pred_mask=pred_mask_logits, gt_mask=gt_mask,
                    diffusion_noise_pred=outputs.get("diffusion_noise_pred"),
                    diffusion_noise_target=outputs.get("diffusion_noise_target"),
                    diffusion_mask=outputs.get("diffusion_mask"))

                local_val_loss += val_loss.detach() * batch_size

                sparse_metrics = eval_metric.get_sparse_support_metrics(pred, y)

                local_val_sparse_precision += sparse_metrics["sparse_precision"] * batch_size
                local_val_sparse_recall += sparse_metrics["sparse_recall"] * batch_size
                local_val_sparse_f1 += sparse_metrics["sparse_f1"] * batch_size
                local_val_pred_density += sparse_metrics["pred_density"] * batch_size
                local_val_target_density += sparse_metrics["target_density"] * batch_size
                local_val_density_error += sparse_metrics["density_error"] * batch_size
                local_val_nonzero_mae += sparse_metrics["nonzero_mae"] * batch_size
                local_val_zero_mae += sparse_metrics["zero_mae"] * batch_size
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
                local_val_sparse_precision,
                local_val_sparse_recall,
                local_val_sparse_f1,
                local_val_pred_density,
                local_val_target_density,
                local_val_density_error,
                local_val_nonzero_mae,
                local_val_zero_mae,
            ):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        self._update_metrics(
            self.epochs_run,
            local_train_samples.item(),
            local_train_loss.item(),
            local_val_samples.item(),
            local_val_loss.item(),
            local_val_sparse_precision.item(),
            local_val_sparse_recall.item(),
            local_val_sparse_f1.item(),
            local_val_pred_density.item(),
            local_val_target_density.item(),
            local_val_density_error.item(),
            local_val_nonzero_mae.item(),
            local_val_zero_mae.item(),
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
