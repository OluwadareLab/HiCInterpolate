from collections import OrderedDict
from src.loss import CombinedLoss, ExponentialDecay
from src.metric import eval_metrics as eval_metric
from src.misc import plots as plot
from src.interpolator import Interpolator
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import pandas as pd
import time
import gc
import traceback
import torch
import torch.distributed as dist
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    def __init__(self, cfg, log, train_dl: DataLoader, val_dl: DataLoader, load_snapshot: bool = False, isDistributed: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.model = Interpolator(self.cfg)

        self.isDistributed = dist.is_available() and dist.is_initialized()
        if isDistributed:
            self.device = int(os.environ['LOCAL_RANK'])
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
        else:
            self.device = self.cfg.device
            self.model = self.model.to(self.device)

        self.loss_fn = CombinedLoss(self.cfg)
        self.optimizer_name = str(
            self.cfg.training.get("optimizer_name", "adamw")).lower()
        self.scheduler_name = str(
            self.cfg.training.get("scheduler_name", "reduce_on_plateau")).lower()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.train_dl = train_dl
        self.train_steps = len(self.train_dl)
        self.batch_size = train_dl.batch_size
        self.val_dl = val_dl
        self.val_steps = len(self.val_dl)
        self.save_every = self.cfg.training.save_every

        self.epochs_run = 0
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_psnr_per_epoch = 0
        self.val_ssim_per_epoch = 0
        self.val_genome_disco_per_epoch = 0
        self.val_hicrep_per_epoch = 0
        self.val_lpips_per_epoch = 0

        self.state = {'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [],
                      'val_psnr': [], 'val_ssim': [], 'val_genome_disco': [], 'val_hicrep': [], 'val_lpips': [], 'best_val': []}
        self.metric_columns = ['epoch', 'lr', 'train_loss',
                               'val_loss', 'val_psnr', 'val_ssim', 'val_genome_disco', 'val_hicrep', 'val_lpips', 'best_val']
        self.patience = 20
        self.epochs_no_improve = 0
        self.best_val = -float('inf')
        self.best_model = f'{self.cfg.file.model}'

        self.snapshot = f'{self.cfg.file.snapshot}'
        if load_snapshot and os.path.exists(self.snapshot):
            self.log.info(f"Loading snapshot...")
            print(f"[INFO] Loading snapshot...")
            self._load_snapshot(self.snapshot)

    def _build_optimizer(self):
        init_lr = float(self.cfg.training.init_lr)
        weight_decay = float(self.cfg.training.get("weight_decay", 1e-4))

        if self.optimizer_name == "adamw":
            return AdamW(self.model.parameters(), lr=init_lr, weight_decay=weight_decay)
        if self.optimizer_name == "adam":
            return Adam(self.model.parameters(), lr=init_lr)
        if self.optimizer_name == "sgd":
            return SGD(self.model.parameters(), lr=init_lr)
        if self.optimizer_name == "rmsprop":
            return RMSprop(self.model.parameters(), lr=init_lr)

        raise ValueError(f"Unsupported optimizer_name: {self.optimizer_name}")

    def _build_scheduler(self):
        if self.scheduler_name == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='max',
                factor=float(self.cfg.training.get("plateau_factor", 0.5)),
                patience=int(self.cfg.training.get("plateau_patience", 8)),
                threshold=float(self.cfg.training.get(
                    "plateau_threshold", 1e-4)),
                cooldown=int(self.cfg.training.get("plateau_cooldown", 2)),
                min_lr=float(self.cfg.training.min_lr),
            )

        if self.scheduler_name == "exponential":
            return ExponentialDecay(
                optimizer=self.optimizer,
                decay_steps=self.cfg.training.decay_steps,
                decay_rate=self.cfg.training.decay_rate,
                staircase=self.cfg.training.lr_staircase)

        raise ValueError(f"Unsupported scheduler_name: {self.scheduler_name}")

    def _structure_lr_score(self):
        score_weights = self.cfg.training.get("lr_metric_weights", {
            "genome_disco": 0.5,
            "hicrep": 0.5,
        })
        metric_values = {
            "genome_disco": self.val_genome_disco_per_epoch,
            "hicrep": self.val_hicrep_per_epoch,
            "ssim": self.val_ssim_per_epoch,
        }

        numerator = 0.0
        denominator = 0.0
        for metric_name, metric_value in metric_values.items():
            weight = float(score_weights.get(metric_name, 0.0))
            numerator += weight * metric_value
            denominator += weight

        return numerator / max(denominator, 1e-8)

    def _step_scheduler(self, epoch: int):
        warmup_epochs = int(self.cfg.training.get("warmup_epochs", 0))
        init_lr = float(self.cfg.training.init_lr)
        min_lr = float(self.cfg.training.min_lr)

        if epoch < warmup_epochs:
            warmup_lr = init_lr * float(epoch + 1) / \
                float(max(1, warmup_epochs))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            return

        if self.scheduler_name == "reduce_on_plateau":
            self.scheduler.step(self._structure_lr_score())
        else:
            self.scheduler.step()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(float(param_group['lr']), min_lr)

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
            model_state = snapshot['model']
            if not all(k.startswith("module.") for k in model_state.keys()):
                model_state = {f"module.{k}": v for k,
                               v in model_state.items()}
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(snapshot['optimizer'])
            try:
                self.scheduler.load_state_dict(snapshot['scheduler'])
            except Exception as ex:
                self.log.warning(
                    f"Scheduler state load failed, using fresh scheduler: {ex}")
                print(
                    f"[WARN] Scheduler state load failed, using fresh scheduler: {ex}")
            self.state = snapshot['state']
            self.best_val = snapshot['state']['best_val'][-1]
        else:
            snapshot = torch.load(snapshot, map_location=self.device)
            self.epochs_run = snapshot['epoch']
            state_dict = self._remove_module_prefix(snapshot['model'])
            self.model.load_state_dict(state_dict)
            state_dict = self._remove_module_prefix(snapshot['optimizer'])
            self.optimizer.load_state_dict(state_dict)
            state_dict = self._remove_module_prefix(snapshot['scheduler'])
            try:
                self.scheduler.load_state_dict(state_dict)
            except Exception as ex:
                self.log.warning(
                    f"Scheduler state load failed, using fresh scheduler: {ex}")
                print(
                    f"[WARN] Scheduler state load failed, using fresh scheduler: {ex}")
            state_dict = self._remove_module_prefix(snapshot['state'])
            self.state = state_dict
            self.best_val = state_dict['best_val'][-1]
        self.log.info(
            f"Resuming training from snapshot at epoch {self.epochs_run}")
        print(
            f"[INFO] Resuming training from snapshot at epoch {self.epochs_run}")

    def _update_metrics(self, epoch, local_train_steps, local_train_loss, local_val_steps, local_val_loss, local_val_psnr, local_val_ssim, local_val_genome_disco, local_val_hicrep, local_val_lpips):

        self.train_loss_per_epoch = local_train_loss / local_train_steps
        self.val_loss_per_epoch = local_val_loss / local_val_steps
        self.val_psnr_per_epoch = local_val_psnr / local_val_steps
        self.val_ssim_per_epoch = local_val_ssim / local_val_steps
        self.val_genome_disco_per_epoch = local_val_genome_disco / local_val_steps
        self.val_hicrep_per_epoch = local_val_hicrep / local_val_steps
        self.val_lpips_per_epoch = local_val_lpips / local_val_steps

        self.state['epoch'].append(epoch+1)
        self.state['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.state['train_loss'].append(self.train_loss_per_epoch)
        self.state['val_loss'].append(self.val_loss_per_epoch)
        self.state['val_psnr'].append(self.val_psnr_per_epoch)
        self.state['val_ssim'].append(self.val_ssim_per_epoch)
        self.state['val_genome_disco'].append(self.val_genome_disco_per_epoch)
        self.state['val_hicrep'].append(self.val_hicrep_per_epoch)
        self.state['val_lpips'].append(self.val_lpips_per_epoch)

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

    def _save_best_model(self, epoch: int):
        default_weights = {
            "genome_disco": 0.45,
            "hicrep": 0.45,
            "ssim": 0.10,
        }
        metric_weights = self.cfg.training.get(
            "best_model_metric_weights", default_weights)
        eps = float(self.cfg.training.get("best_model_metric_norm_eps", 1e-8))

        metric_values = {
            "genome_disco": self.val_genome_disco_per_epoch,
            "hicrep": self.val_hicrep_per_epoch,
            "ssim": self.val_ssim_per_epoch,
        }

        weighted_score = 0.0
        weight_sum = 0.0
        for metric_name, metric_value in metric_values.items():
            series = self.state.get(f"val_{metric_name}", [])
            if len(series) == 0:
                norm_value = 0.0
            else:
                min_v = min(series)
                max_v = max(series)
                denom = max(max_v - min_v, eps)
                norm_value = (metric_value - min_v) / denom

            weight = float(metric_weights.get(metric_name, 0.0))
            weighted_score += weight * norm_value
            weight_sum += weight

        best_val = weighted_score / max(weight_sum, eps)
        if best_val > self.best_val:
            self.epochs_no_improve = 0
            self.best_val = best_val
            snapshot = self._get_model_stats(epoch)
            torch.save(snapshot, self.best_model)
            self.log.debug(
                f"Epoch {self.epochs_run+1} saved best model.")
            print(
                f"[DEBUG] Epoch {self.epochs_run+1} saved best model.")


            try:
                import random
                self.model.eval()
                val_iter = iter(self.val_dl)
                x0, y, x1, time_frame = next(val_iter)
                batch_size = x0.shape[0]
                if batch_size == 1:
                    idxs = [0]
                else:
                    idxs = random.sample(range(batch_size), min(2, batch_size))
                x0 = x0[idxs].to(self.device)
                y = y[idxs].to(self.device)
                x1 = x1[idxs].to(self.device)
                time_frame = time_frame[idxs].to(self.device)
                with torch.no_grad():
                    pred = self.model(x0, x1, time_frame)

                x0_np = x0.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                x1_np = x1.detach().cpu().numpy()

                vis_file = os.path.join(os.path.dirname(
                    self.best_model), f"best_model_visualization_epoch{self.epochs_run+1}.png")
                plot.draw_hic_map(len(idxs), x0_np, y_np, pred_np, x1_np, vis_file)
                print(f"[DEBUG] Saved best model visualization to {vis_file}")
            except Exception as e:
                print(f"[WARN] Could not save best model visualization: {e}")
        elif self.epochs_run > int(self.cfg.training.epochs/4):
            self.epochs_no_improve += 1

    def _save_and_draw_metrics(self):
        metrics_df = pd.DataFrame({
            'epoch': self.state["epoch"],
            'lr': self.state["lr"],
            'train_loss': self.state["train_loss"],
            'val_loss': self.state["val_loss"],
            'val_psnr':  self.state["val_psnr"],
            'val_ssim':  self.state["val_ssim"],
            'val_genome_disco':  self.state["val_genome_disco"],
            'val_hicrep':  self.state["val_hicrep"],
            'val_lpips':  self.state["val_lpips"],
            'best_val': self.state["best_val"]
        }, columns=self.metric_columns)

        metrics_df.to_csv(self.cfg.file.val_metrics, index=False)
        plot.draw_metric(self.cfg, self.state)

    def _run_epoch(self, epoch):
        self.epochs_run = epoch
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_psnr_per_epoch = 0
        self.val_ssim_per_epoch = 0
        self.val_genome_disco_per_epoch = 0
        self.val_hicrep_per_epoch = 0
        self.val_lpips_per_epoch = 0

        self.model.train()
        if self.isDistributed:
            self.train_dl.sampler.set_epoch(epoch)

        local_train_loss = 0.0

        for step, (x0, y, x1, time_frame) in enumerate(tqdm(self.train_dl)):
            x0, y, x1, time_frame = [t.to(self.device)
                                     for t in (x0, y, x1, time_frame)]
            self.optimizer.zero_grad()
            pred = self.model(x0, x1, time_frame)
            train_loss = self.loss_fn(pred, y, self.epochs_run)
            local_train_loss += train_loss.item()
            train_loss.backward()
            self.optimizer.step()

            del x0, y, x1, time_frame

        local_val_loss = 0
        local_val_psnr = 0
        local_val_ssim = 0
        local_val_genome_disco = 0
        local_val_hicrep = 0
        local_val_lpips = 0

        with torch.no_grad():
            self.model.eval()
            for _, (x0, y, x1, time_frame) in enumerate(self.val_dl):
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)
                time_frame = time_frame.to(self.device)
                pred = self.model(x0, x1, time_frame)
                val_loss = self.loss_fn(pred, y, self.epochs_run)
                local_val_loss += val_loss.item()

                psnr_val = eval_metric.get_psnr(pred, y)
                ssim_val = eval_metric.get_ssim(pred, y)
                genome_disco_val = eval_metric.get_genome_disco(pred, y)
                hicrep_val = eval_metric.get_hicrep(pred, y)
                lpips_val = eval_metric.get_lpips(pred, y)

                local_val_psnr += psnr_val.item()
                local_val_ssim += ssim_val.item()
                local_val_genome_disco += genome_disco_val.item()
                local_val_hicrep += hicrep_val.item()
                local_val_lpips += lpips_val.item()

                del x0, y, x1, time_frame

        if self.isDistributed:
            local_train_steps = torch.tensor(
                self.train_steps, device=self.device)
            local_val_steps = torch.tensor(self.val_steps, device=self.device)
            local_train_loss = torch.tensor(
                local_train_loss, device=self.device)
            local_val_loss = torch.tensor(
                local_val_loss, device=self.device)
            local_val_psnr = torch.tensor(
                local_val_psnr, device=self.device)
            local_val_ssim = torch.tensor(
                local_val_ssim, device=self.device)
            local_val_genome_disco = torch.tensor(
                local_val_genome_disco, device=self.device)
            local_val_hicrep = torch.tensor(
                local_val_hicrep, device=self.device)
            local_val_lpips = torch.tensor(
                local_val_lpips, device=self.device)

            dist.all_reduce(local_train_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_ssim, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_genome_disco, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_hicrep, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_lpips, op=dist.ReduceOp.SUM)

            local_train_steps = local_train_steps.item()
            local_val_steps = local_val_steps.item()
            local_train_loss = local_train_loss.item()
            local_val_loss = local_val_loss.item()
            local_val_psnr = local_val_psnr.item()
            local_val_ssim = local_val_ssim.item()
            local_val_genome_disco = local_val_genome_disco.item()
            local_val_hicrep = local_val_hicrep.item()
            local_val_lpips = local_val_lpips.item()
            self._update_metrics(self.epochs_run, local_train_steps, local_train_loss, local_val_steps,
                                 local_val_loss, local_val_psnr, local_val_ssim, local_val_genome_disco, local_val_hicrep, local_val_lpips)
        else:
            self._update_metrics(self.epochs_run, self.train_steps, local_train_loss, self.val_steps,
                                 local_val_loss, local_val_psnr, local_val_ssim, local_val_genome_disco, local_val_hicrep, local_val_lpips)

        self._step_scheduler(epoch)

    def train(self, max_epochs: int):
        self.log.info(f"==== Training Started ({self.device}) ====")
        print(f"[INFO] ==== Training Started ({self.device}) ====")

        start_time = time.time()
        try:

            for epoch in range(self.epochs_run, max_epochs):
                if self.epochs_no_improve > self.patience:
                    self.log.info(f"No improvement in last 20 epoch!")
                    print(f"No improvement in last 20 epoch!")
                    break

                self._run_epoch(epoch)
                if self.isDistributed and self.device == 0:
                    self._save_best_model(epoch)
                    self.state["best_val"].append(self.best_val)
                    if (self.epochs_run+1) % self.save_every == 0:
                        self._save_snapshot(epoch)
                    self._save_and_draw_metrics()
                    scores = f"[{(self.epochs_run+1)}/{max_epochs}] LR: {self.optimizer.param_groups[0]['lr']}; Batch Size: {self.batch_size}; Train Loss: {format(self.train_loss_per_epoch, '.6f')}; Val (Loss: {format(self.val_loss_per_epoch, '.6f')}, PSNR: {format(self.val_psnr_per_epoch, '.4f')}, SSIM: {format(self.val_ssim_per_epoch, '.4f')}, GenomeDISCO: {format(self.val_genome_disco_per_epoch, '.4f')}, HiCRep: {format(self.val_hicrep_per_epoch, '.4f')}, LPIPS: {format(self.val_lpips_per_epoch, '.4f')};"

                    self.log.info(f"{scores}")
                    print(f"[INFO] {scores}")

                elif not self.isDistributed:
                    self._save_best_model(epoch)
                    self.state["best_val"].append(self.best_val)
                    if (self.epochs_run+1) % self.save_every == 0:
                        self._save_snapshot(epoch)
                    self._save_and_draw_metrics()
                    scores = f"[{(self.epochs_run+1)}/{max_epochs}] LR: {self.optimizer.param_groups[0]['lr']}; Batch Size: {self.batch_size}; Train Loss: {format(self.train_loss_per_epoch, '.6f')}; Val (Loss: {format(self.val_loss_per_epoch, '.6f')}, PSNR: {format(self.val_psnr_per_epoch, '.4f')}, SSIM: {format(self.val_ssim_per_epoch, '.4f')}, GenomeDISCO: {format(self.val_genome_disco_per_epoch, '.4f')}, HiCRep: {format(self.val_hicrep_per_epoch, '.4f')}, LPIPS: {format(self.val_lpips_per_epoch, '.4f')};"

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
