from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from .loss import ExponentialDecay, CombinedLoss
from .interpolator import Interpolator
from .misc import plots as plot, metrics as metric
from collections import OrderedDict
import torch.distributed as dist
import torch
import traceback
import gc
import time
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    def __init__(self, cfg, log, train_dl: DataLoader, val_dl: DataLoader, load_snapshot: bool = False, isDistributed: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.isDistributed = dist.is_available() and dist.is_initialized()
        if isDistributed:
            self.device = int(os.environ['LOCAL_RANK'])
            self.model = Interpolator(self.cfg).to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
        else:
            self.device = self.cfg.device
            self.model = Interpolator(self.cfg).to(self.device)

        self.loss_fn = CombinedLoss(self.cfg)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.cfg.training.learning_rate)
        self.scheduler = ExponentialDecay(optimizer=self.optimizer, decay_steps=self.cfg.training.decay_steps,
                                          decay_rate=self.cfg.training.decay_rate, staircase=self.cfg.training.lr_staircase)

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
        self.state = {'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [],
                      'val_psnr': [], 'val_ssim': [], 'best_val': []}
        self.metric_columns = ['epoch', 'lr', 'train_loss',
                               'val_loss', 'val_psnr', 'val_ssim', 'best_val']
        self.best_val = -1.0
        self.best_model = f'{self.cfg.file.model}'

        self.snapshot = f'{self.cfg.file.snapshot}'
        if load_snapshot and os.path.exists(self.snapshot):
            self.log.info(f"[{self.device}] loading snapshot...")
            print(f"[INFO][{self.device}] loading snapshot...")
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
            self.best_val = snapshot['state']['best_val'][-1]
        else:
            snapshot = torch.load(snapshot, map_location=self.device)
            self.epochs_run = snapshot['epoch']
            state_dict = self._remove_module_prefix(snapshot['model'])
            self.model.load_state_dict(state_dict)
            state_dict = self._remove_module_prefix(snapshot['optimizer'])
            self.optimizer.load_state_dict(state_dict)
            state_dict = self._remove_module_prefix(snapshot['scheduler'])
            self.scheduler.load_state_dict(state_dict)
            state_dict = self._remove_module_prefix(snapshot['state'])
            self.state = state_dict
            self.best_val = state_dict['best_val'][-1]
        self.log.info(
            f"[{self.device}] resuming training from snapshot at epoch {self.epochs_run}")
        print(
            f"[INFO][{self.device}] resuming training from snapshot at epoch {self.epochs_run}")

    def _run_batch(self, x0, y, x1, time_frame, isValidation=False):
        if isValidation:
            output = self.model(x0, x1, time_frame)
            loss_fn = self.loss_fn(output, y, self.epochs_run)
            loss_val = loss_fn.item()
        else:
            self.optimizer.zero_grad()
            output = self.model(x0, x1, time_frame)
            loss_fn = self.loss_fn(output, y, self.epochs_run)
            loss_val = loss_fn.item()
            loss_fn.backward()
            self.optimizer.step()
        return output, loss_val

    def _update_metrics(self, epoch, local_train_steps, local_train_loss, local_val_steps, local_val_loss, local_val_psnr, local_val_ssim):

        self.train_loss_per_epoch = local_train_loss / local_train_steps
        self.val_loss_per_epoch = local_val_loss / local_val_steps
        self.val_psnr_per_epoch = local_val_psnr / local_val_steps
        self.val_ssim_per_epoch = local_val_ssim / local_val_steps

        self.state['epoch'].append(epoch+1)
        self.state['lr'].append(self.scheduler.get_last_lr()[0])
        self.state['train_loss'].append(self.train_loss_per_epoch)
        self.state['val_loss'].append(self.val_loss_per_epoch)
        self.state['val_psnr'].append(self.val_psnr_per_epoch)
        self.state['val_ssim'].append(self.val_ssim_per_epoch)

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
        print(
            f"[DEBUG][{self.device}] epoch {epoch+1} saved snapshot at {self.snapshot}")

    def _save_best_model(self, epoch: int):
        if self.val_ssim_per_epoch > self.best_val:
            self.best_val = self.val_ssim_per_epoch
            snapshot = self._get_model_stats(epoch)
            torch.save(snapshot, self.best_model)
            self.log.debug(
                f"[{self.device}] epoch {epoch+1} saved best model.")
            print(
                f"[DEBUG][{self.device}] epoch {epoch+1} saved best model.")

    def _save_and_draw_metrics(self):
        metrics_df = pd.DataFrame({
            'epoch': self.state["epoch"],
            'lr': self.state["lr"],
            'train_loss': self.state["train_loss"],
            'val_loss': self.state["val_loss"],
            'val_psnr':  self.state["val_psnr"],
            'val_ssim':  self.state["val_ssim"],
            'best_val': self.state["best_val"]
        }, columns=self.metric_columns)

        metrics_df.to_csv(self.cfg.file.eval_metrics, index=False)
        plot.draw_metric(self.cfg, self.state)

    def _run_epoch(self, epoch):
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_psnr_per_epoch = 0
        self.val_ssim_per_epoch = 0

        self.model.train()
        if self.isDistributed:
            self.train_dl.sampler.set_epoch(epoch)

        local_train_loss = 0
        for _, (x0, y, x1, time_frame) in enumerate(tqdm(self.train_dl)):
            x0 = x0.to(self.device)
            y = y.to(self.device)
            x1 = x1.to(self.device)
            time_frame = time_frame.to(self.device)
            pred, loss = self._run_batch(x0, y, x1, time_frame)
            local_train_loss += loss
            del x0, y, x1, time_frame

        self.scheduler.step()

        local_val_loss = 0
        local_val_psnr = 0
        local_val_ssim = 0
        with torch.no_grad():
            self.model.eval()
            for _, (x0, y, x1, time_frame) in enumerate(self.val_dl):
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)
                time_frame = time_frame.to(self.device)
                pred, loss = self._run_batch(x0, y, x1, time_frame, True)
                local_val_loss += loss

                psnr_val = metric.calculate_psnr(pred, y)
                ssim_val = metric.calculate_ssim(pred, y)
                local_val_psnr += psnr_val.item()
                local_val_ssim += ssim_val.item()

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

            dist.all_reduce(local_train_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_ssim, op=dist.ReduceOp.SUM)

            local_train_steps = local_train_steps.item()
            local_val_steps = local_val_steps.item()
            local_train_loss = local_train_loss.item()
            local_val_loss = local_val_loss.item()
            local_val_psnr = local_val_psnr.item()
            local_val_ssim = local_val_ssim.item()
            self._update_metrics(epoch, local_train_steps, local_train_loss, local_val_steps,
                                 local_val_loss, local_val_psnr, local_val_ssim)
        else:
            self._update_metrics(epoch, self.train_steps, local_train_loss, self.val_steps,
                                 local_val_loss, local_val_psnr, local_val_ssim)

    def train(self, max_epochs: int):
        self.log.info(f"[{self.device}] ==== training started ====")
        print(f"[INFO][{self.device}] ==== training started ====")

        start_time = time.time()
        try:
            for epoch in range(self.epochs_run, max_epochs):
                self._run_epoch(epoch)

                if self.isDistributed and self.device == 0:
                    self._save_best_model(epoch)
                    self.state["best_val"].append(self.best_val)
                    if (epoch+1) % self.save_every == 0:
                        self._save_snapshot(epoch)
                    self._save_and_draw_metrics()

                    self.log.info(f"epoch [{(epoch+1)}/{max_epochs}] learning rate: {self.scheduler.get_last_lr()[0]}; batch size: {self.batch_size}; train loss: {format(self.train_loss_per_epoch, '.6f')}; validation (loss: {format(self.val_loss_per_epoch, '.6f')}, psnr: {format(self.val_psnr_per_epoch, '.4f')}, ssim: {format(self.val_ssim_per_epoch, '.4f')});")

                    print(f"[INFO] epoch [{(epoch+1)}/{max_epochs}] learning rate: {self.scheduler.get_last_lr()[0]}; batch size: {self.batch_size}; train loss: {format(self.train_loss_per_epoch, '.6f')}; validation (loss: {format(self.val_loss_per_epoch, '.6f')}, psnr: {format(self.val_psnr_per_epoch, '.4f')}, ssim: {format(self.val_ssim_per_epoch, '.4f')});")

                elif not self.isDistributed:
                    self._save_best_model(epoch)
                    self.state["best_val"].append(self.best_val)
                    if (epoch+1) % self.save_every == 0:
                        self._save_snapshot(epoch)
                    self._save_and_draw_metrics()

                    self.log.info(f"epoch [{(epoch+1)}/{max_epochs}] learning rate: {self.scheduler.get_last_lr()[0]}; batch size: {self.batch_size}; train loss: {format(self.train_loss_per_epoch, '.6f')}; validation (loss: {format(self.val_loss_per_epoch, '.6f')}, psnr: {format(self.val_psnr_per_epoch, '.4f')}, ssim: {format(self.val_ssim_per_epoch, '.4f')});")

                    print(f"[INFO] epoch [{(epoch+1)}/{max_epochs}] learning rate: {self.scheduler.get_last_lr()[0]}; batch size: {self.batch_size}; train loss: {format(self.train_loss_per_epoch, '.6f')}; validation (loss: {format(self.val_loss_per_epoch, '.6f')}, psnr: {format(self.val_psnr_per_epoch, '.4f')}, ssim: {format(self.val_ssim_per_epoch, '.4f')});")

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
            f"[{self.device}] total time taken: {format((end_time-start_time), '.2f')} seconds")
        print(
            f"[INFO][{self.device}] total time taken: {format((end_time-start_time), '.2f')} seconds")
        self.log.info(f"[{self.device}] ==== training end ====")
        print(f"[INFO][{self.device}] ==== training end ====")
