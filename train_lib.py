import pandas as pd
import time
import gc
import traceback
import config
import sys
import os
import torch.distributed as dist
from plots import *
from metrics import *
from loss.model import CombinedLoss
from interpolator.model import Interpolator
from tqdm import tqdm
from scheduler import ExponentialDecay
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    def __init__(self, train_dl: DataLoader, val_dl: DataLoader, save_every: int = 20, load_snapshot: bool = False, isDistributed: bool = False) -> None:
        self.isDistributed = dist.is_available() and dist.is_initialized()
        if isDistributed:
            self.device = int(os.environ["LOCAL_RANK"])
            self.model = Interpolator().to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
        else:
            self.device = config.DEVICE
            self.model = Interpolator().to(self.device)

        self.loss_fn = CombinedLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = ExponentialDecay(optimizer=self.optimizer, decay_steps=config.DECAY_STEPS,
                                          decay_rate=config.DECAY_RATE, staircase=config.LEARNING_RATE_STAIRCASE)

        self.train_dl = train_dl
        self.train_steps = len(self.train_dl)
        self.batch_size = train_dl.batch_size
        self.val_dl = val_dl
        self.val_steps = len(self.val_dl)
        self.save_every = save_every

        self.epochs_run = 0
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_psnr_per_epoch = 0
        self.val_ssim_per_epoch = 0
        self.state = {"epoch": [], "lr": [], "train_loss": [], "val_loss": [],
                      "val_psnr": [], "val_ssim": [], "best_val": []}
        self.metric_columns = ['epoch', 'lr', 'train_loss',
                               'val_loss', 'val_psnr', 'val_ssim', 'best_val']
        self.best_val = -1.0
        self.best_model_file = f'{config.MODEL_STATE_DIR}/{config.MODEL_NAME}_best_model.pt'

        self.snapshot_file = f'{config.MODEL_STATE_DIR}/{config.MODEL_NAME}_snapshot.pt'
        if load_snapshot and os.path.exists(self.snapshot_file):
            print(f"[INFO][{self.device}] loading snapshot...")
            self._load_snapshot(self.snapshot_file)

    def _load_snapshot(self, snapshot_file):
        if self.isDistributed:
            loc = f"cuda:{self.device}"
            snapshot = torch.load(snapshot_file, map_location=loc)
        else:
            snapshot = torch.load(snapshot_file, map_location=self.device)

        self.epochs_run = snapshot["epoch"]
        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.state = snapshot["state"]
        self.best_val = snapshot["state"]["best_val"][-1]
        print(
            f"[INFO][{self.device}] resuming training from snapshot at epoch {self.epochs_run}")

    def _run_batch(self, x0, x1, x2, time_frame, isValidation=False):
        if isValidation:
            output = self.model(x0, x2, time_frame)
            loss_fn = self.loss_fn(output, x1, self.epochs_run)
            loss_val = loss_fn.item()
        else:
            self.optimizer.zero_grad()
            output = self.model(x0, x2, time_frame)
            loss_fn = self.loss_fn(output, x1, self.epochs_run)
            loss_val = loss_fn.item()
            loss_fn.backward()
            self.optimizer.step()
        return output, loss_val

    def _update_metrics(self, epoch, local_train_steps, local_train_loss, local_val_steps, local_val_loss, local_val_psnr, local_val_ssim):

        self.train_loss_per_epoch = local_train_loss / local_train_steps
        self.val_loss_per_epoch = local_val_loss / local_val_steps
        self.val_psnr_per_epoch = local_val_psnr / local_val_steps
        self.val_ssim_per_epoch = local_val_ssim / local_val_steps

        self.state["epoch"].append(epoch+1)
        self.state["lr"].append(self.scheduler.get_last_lr()[0])
        self.state["train_loss"].append(self.train_loss_per_epoch)
        self.state["val_loss"].append(self.val_loss_per_epoch)
        self.state["val_psnr"].append(self.val_psnr_per_epoch)
        self.state["val_ssim"].append(self.val_ssim_per_epoch)

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
        torch.save(snapshot, self.snapshot_file)
        print(
            f"[DEBUG][{self.device}] epoch {epoch+1} --> saved snapshot at {self.snapshot_file}")

    def _save_best_model(self, epoch: int):
        if self.val_ssim_per_epoch > self.best_val:
            self.best_val = self.val_ssim_per_epoch
            snapshot = self._get_model_stats(epoch)
            torch.save(snapshot, self.best_model_file)
            print(
                f"[DEBUG][{self.device}] epoch {epoch+1} --> saved best model.")

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

        metrics_df.to_csv(config.EVAL_METRICS_FILE, index=False)
        draw_metric(self.state)

    def _run_epoch(self, epoch):
        self.train_loss_per_epoch = 0
        self.val_loss_per_epoch = 0
        self.val_psnr_per_epoch = 0
        self.val_ssim_per_epoch = 0

        self.model.train()
        if self.isDistributed:
            self.train_dl.sampler.set_epoch(epoch)

        local_train_loss = 0
        for idx, (x0, x1, x2, time_frame) in enumerate(tqdm(self.train_dl)):
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            time_frame = time_frame.to(self.device)
            pred, loss = self._run_batch(x0, x1, x2, time_frame)
            local_train_loss += loss

            # del x0, x1, x2, time_frame, pred

        self.scheduler.step()

        local_val_loss = 0
        local_val_psnr = 0
        local_val_ssim = 0
        with torch.no_grad():
            self.model.eval()
            drawn = False
            for _, (x0, x1, x2, time_frame) in enumerate(self.val_dl):
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                time_frame = time_frame.to(self.device)
                pred, loss = self._run_batch(x0, x1, x2, time_frame, True)
                local_val_loss += loss
                if not drawn:
                    draw_real_in_out_images(
                        x0=x0, x1=x1, x2=x2, pred=pred, epoch=epoch)
                    drawn = True

                psnr_val = calculate_psnr(pred, x1)
                ssim_val = calculate_ssim(pred, x1)
                local_val_psnr += psnr_val.item()
                local_val_ssim += ssim_val.item()

                # del x0, x1, x2, time_frame, pred

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
        print(f"[INFO][{self.device}] training the network...")

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
                    print(f"[INFO] epoch [{(epoch+1)}/{max_epochs}] --> learning rate: {self.scheduler.get_last_lr()[0]}; batch size: {self.batch_size}; train loss: {format(self.train_loss_per_epoch, '.6f')}; validation (loss: {format(self.val_loss_per_epoch, '.6f')}, psnr: {format(self.val_psnr_per_epoch, '.4f')}, ssim: {format(self.val_ssim_per_epoch, '.4f')});")

                elif not self.isDistributed:
                    self._save_best_model(epoch)
                    self.state["best_val"].append(self.best_val)
                    if (epoch+1) % self.save_every == 0:
                        self._save_snapshot(epoch)
                    print(f"[INFO] epoch [{(epoch+1)}/{max_epochs}] --> learning rate: {self.scheduler.get_last_lr()[0]}; batch size: {self.batch_size}; train loss: {format(self.train_loss_per_epoch, '.6f')}; validation (loss: {format(self.val_loss_per_epoch, '.6f')}, psnr: {format(self.val_psnr_per_epoch, '.4f')}, ssim: {format(self.val_ssim_per_epoch, '.4f')});")

        except Exception as ex:
            print(ex)
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        end_time = time.time()

        if self.isDistributed and self.device == 0:
            self._save_and_draw_metrics()
        elif not self.isDistributed:
            self._save_and_draw_metrics()

        print(
            f"[INFO][{self.device}] total time taken: {format((end_time-start_time), '.2f')} seconds")
