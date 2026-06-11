from src.loss import CombinedLoss
from src.metric import metrics as eval_metric
from src.misc import plots as plot
from src.interpolator import Interpolator
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import math
import torch
import torch.distributed as dist
import time
import gc
import pandas as pd
import traceback
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    EVAL_METRICS = ("psnr", "ssim", "scc",
                    "hicrep", "genome_disco", "lpips")

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
            self.cfg.device = f"cuda:{self.device}"
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
        decay_params, no_decay_params = self._split_weight_decay_params(
            self.model)
        self.optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.cfg.training.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.training.lr,
            betas=(0.9, 0.99),
            eps=1e-8
        )
        total_iterations = max(1, len(self.train_dl) * self.cfg.training.epochs)
        warmup_iterations = max(
            0, len(self.train_dl) * self.cfg.training.warmup_epochs)
        if warmup_iterations > 0:
            start_factor = 0.01
            min_lr_factor = self.cfg.training.min_lr / self.cfg.training.lr
            cosine_iterations = max(1, total_iterations - warmup_iterations)

            def lr_lambda(step):
                if step < warmup_iterations:
                    return start_factor + (1.0 - start_factor) * step / warmup_iterations
                progress = min((step - warmup_iterations) /
                               cosine_iterations, 1.0)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_iterations, eta_min=self.cfg.training.min_lr)

        self.epochs_run = 0
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.active_eval_metrics = self._get_active_eval_metrics()
        self.monitor_metric = self._get_monitor_metric()
        self.metric_values_per_epoch = {
            metric: 0.0 for metric in self.active_eval_metrics}

        self.state = {'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [],
                      **{f'val_{metric}': [] for metric in self.active_eval_metrics},
                      'best_val': []}
        self.metric_columns = ['epoch', 'lr', 'train_loss', 'val_loss',
                               *[f'val_{metric}' for metric in self.active_eval_metrics],
                               'best_val']
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
                self.state[key] = [-float('inf')] * \
                    num_epochs if key == 'best_val' else [0.0] * num_epochs

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

    def _get_active_eval_metrics(self):
        cfg_eval = getattr(self.cfg, "evaluation", None)
        metric_cfg = getattr(cfg_eval, "metrics", None)
        active_metrics = []
        for metric in self.EVAL_METRICS:
            enabled = True if metric_cfg is None else bool(
                getattr(metric_cfg, metric, True))
            if enabled:
                active_metrics.append(metric)
        return active_metrics

    def _get_monitor_metric(self):
        cfg_eval = getattr(self.cfg, "evaluation", None)
        monitor = getattr(cfg_eval, "monitor", "ssim")
        if monitor == "loss" or monitor in self.active_eval_metrics:
            return monitor
        return self.active_eval_metrics[0] if self.active_eval_metrics else "loss"

    def _get_monitor_value(self):
        if self.monitor_metric == "loss":
            return -self.val_loss_per_epoch
        value = self.metric_values_per_epoch.get(
            self.monitor_metric, -self.val_loss_per_epoch)
        return -value if self.monitor_metric == "lpips" else value

    def _update_metrics(self, epoch, train_samples, train_loss, val_samples, val_loss, metric_totals):
        self.train_loss_per_epoch = self._safe_average(
            train_loss, train_samples)
        self.val_loss_per_epoch = self._safe_average(val_loss, val_samples)
        self.metric_values_per_epoch = {
            metric: self._safe_average(metric_totals[metric], val_samples)
            for metric in self.active_eval_metrics
        }

        self.state['epoch'].append(epoch+1)
        self.state['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.state['train_loss'].append(self.train_loss_per_epoch)
        self.state['val_loss'].append(self.val_loss_per_epoch)
        for metric, value in self.metric_values_per_epoch.items():
            self.state[f'val_{metric}'].append(value)

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
        best_val = self._get_monitor_value()

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
        elif self.epochs_run > 5:
            self.epochs_no_improve += 1

    def _save_and_draw_metrics(self):
        metrics_df = pd.DataFrame(
            {key: self.state[key] for key in self.metric_columns},
            columns=self.metric_columns)

        metrics_to_round = [column for column in self.metric_columns
                            if column not in ("epoch", "lr")]
        metrics_df[metrics_to_round] = metrics_df[metrics_to_round].round(4)
        if 'lr' in metrics_df.columns:
            metrics_df['lr'] = metrics_df['lr'].round(6)

        metrics_df.to_csv(self.cfg.file.val_metrics, index=False)
        plot.draw_metric(self.cfg, self.state)

    def _format_scores(self, max_epochs: int):
        val_metrics = ", ".join(
            f"{metric.upper()}: {value:.4f}"
            for metric, value in self.metric_values_per_epoch.items())
        return (
            f"[{(self.epochs_run+1)}/{max_epochs}] LR: {self.optimizer.param_groups[0]['lr']:.6f}; "
            f"Batch Size: {self.batch_size}; Train Loss: {self.train_loss_per_epoch:.4f}; "
            f"Val (Loss: {self.val_loss_per_epoch:.4f}"
            f"{', ' + val_metrics if val_metrics else ''}); "
            f"Monitor: {self.monitor_metric};"
        )

    def _run_epoch(self, epoch):
        self.epochs_run = epoch
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.metric_values_per_epoch = {
            metric: 0.0 for metric in self.active_eval_metrics}

        self.model.train()
        if self.isDistributed and hasattr(self.train_dl.sampler, "set_epoch"):
            self.train_dl.sampler.set_epoch(epoch)

        local_train_loss = torch.tensor(0.0, device=self.device)
        local_train_samples = torch.tensor(0.0, device=self.device)

        for step, (x0, y, x1, _) in enumerate(tqdm(self.train_dl)):
            x0 = x0.to(self.device)
            y = y.to(self.device)
            x1 = x1.to(self.device)
            self.optimizer.zero_grad()

            batch_size = y.size(0)
            outputs = self.model(x0, x1)
            pred = outputs["pred"]
            # pred_mask = outputs["mask"]
            # gt_mask = (y > 0).float()

            train_loss = self.loss_fn(
                pred, y, self.epochs_run)

            # if not torch.isfinite(train_loss):
            #     self.optimizer.zero_grad(set_to_none=True)
            #     continue

            train_loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     self.model.parameters(), max_norm=self.cfg.training.grad_clip)
            # if not torch.isfinite(grad_norm):
            #     self.optimizer.zero_grad(set_to_none=True)
            #     continue
            self.optimizer.step()
            self.scheduler.step()

            local_train_loss += train_loss.detach() * batch_size
            local_train_samples += batch_size

            del x0, y, x1, pred

        local_val_loss = torch.tensor(0.0, device=self.device)
        local_metric_totals = {
            metric: torch.tensor(0.0, device=self.device)
            for metric in self.active_eval_metrics
        }
        local_val_samples = torch.tensor(0.0, device=self.device)
        self.best_plot_batch = None

        with torch.no_grad():
            self.model.eval()
            for _, (x0, y, x1, _) in enumerate(self.val_dl):
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)

                batch_size = y.size(0)
                outputs = self.model(x0, x1)

                pred = outputs["pred"]
                # pred_mask = outputs["mask"]
                # gt_mask = (y > 0).float()

                # Compute the validation loss
                val_loss = self.loss_fn(
                    pred, y, self.epochs_run)

                local_val_loss += val_loss.detach() * batch_size

                for metric in self.active_eval_metrics:
                    metric_value = eval_metric.get_metric_gpu(metric, pred, y)
                    local_metric_totals[metric] += metric_value.detach() * batch_size
                local_val_samples += batch_size

                if self.best_plot_batch is None:
                    self.best_plot_batch = (
                        x0[:2].detach().cpu(),
                        y[:2].detach().cpu(),
                        pred[:2].detach().cpu(),
                        x1[:2].detach().cpu(),
                    )

                del x0, y, x1, pred

        if self.isDistributed:
            for value in (
                local_train_samples,
                local_val_samples,
                local_train_loss,
                local_val_loss,
                *local_metric_totals.values(),
            ):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        self._update_metrics(
            self.epochs_run,
            local_train_samples.item(),
            local_train_loss.item(),
            local_val_samples.item(),
            local_val_loss.item(),
            {metric: value.item()
             for metric, value in local_metric_totals.items()},
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
