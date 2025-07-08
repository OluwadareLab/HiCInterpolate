from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from .interpolator import Interpolator
from .misc import plots as plot, metrics as metric
import torch.distributed as dist
import torch
import traceback
import gc
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Tester:
    def __init__(self, cfg, log, model: str, test_dl: DataLoader, isDistributed: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.isDistributed = dist.is_available() and dist.is_initialized()
        if isDistributed:
            self.device = int(os.environ["LOCAL_RANK"])
            self.model = Interpolator(self.cfg).to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
            loc = f"cuda:{self.device}"
            snapshot = torch.load(model, map_location=loc)
            self.model.load_state_dict(snapshot['model'])
        else:
            self.device = self.cfg.device
            self.model = Interpolator(self.cfg).to(self.device)
            snapshot = torch.load(model, map_location=self.device)
            state_dict = self._remove_module_prefix(snapshot['model'])
            self.model.load_state_dict(state_dict)

        

        self.test_dl = test_dl
        self.test_steps = len(self.test_dl)
        self.batch_size = test_dl.batch_size

        self.psnr = 0
        self.ssim = 0

    def _remove_module_prefix(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict

    def _update_metrics(self, local_steps, local_psnr, local_ssim):
        self.psnr = local_psnr / local_steps
        self.ssim = local_ssim / local_steps

    def _run(self):
        local_psnr = 0
        local_ssim = 0
        with torch.no_grad():
            self.model.eval()
            drawn = False
            for _, (x0, y, x1, time_frame) in enumerate(tqdm(self.test_dl)):
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)
                time_frame = time_frame.to(self.device)
                pred = self.model(x0, x1, time_frame)

                psnr_val = metric.calculate_psnr(pred, y)
                ssim_val = metric.calculate_ssim(pred, y)
                local_psnr += psnr_val.item()
                local_ssim += ssim_val.item()

                if not drawn:
                    num_examples = min(
                        self.cfg.eval.num_visualization_samples, len(y))
                    x0_cpu = x0[:num_examples].cpu()
                    y_cpu = y[:num_examples].cpu()
                    pred_cpu = pred[:num_examples].cpu()
                    x1_cpu = x1[:num_examples].cpu()
                    plot.draw_hic_map(num_examples=num_examples, x0=x0_cpu,
                                      y=y_cpu, pred=pred_cpu, x1=x1_cpu, file=self.cfg.file.test_hic_map)
                    drawn = True

                del x0, y, x1, time_frame

        if self.isDistributed:
            local_steps = torch.tensor(
                self.test_steps, device=self.device)
            local_psnr = torch.tensor(
                local_psnr, device=self.device)
            local_ssim = torch.tensor(
                local_ssim, device=self.device)

            dist.all_reduce(local_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_ssim, op=dist.ReduceOp.SUM)

            local_steps = local_steps.item()
            local_psnr = local_psnr.item()
            local_ssim = local_ssim.item()

            self._update_metrics(local_steps, local_psnr, local_ssim)
        else:
            self._update_metrics(self.test_steps, local_psnr, local_ssim)

    def test(self):
        self.log.info(f"[{self.device}] ==== testing started ====")
        print(f"[INFO][{self.device}] ==== testing started ====")

        start_time = time.time()
        try:
            self._run()
            if self.isDistributed and self.device == 0:
                self.log.info(
                    f"psnr: {format(self.psnr, '.4f')}, ssim: {format(self.ssim, '.4f')};")
                print(
                    f"[INFO] psnr: {format(self.psnr, '.4f')}, ssim: {format(self.ssim, '.4f')};")

            elif not self.isDistributed:
                self.log.info(
                    f"psnr: {format(self.psnr, '.4f')}, ssim: {format(self.ssim, '.4f')};")
                print(
                    f"[INFO] psnr: {format(self.psnr, '.4f')}, ssim: {format(self.ssim, '.4f')};")

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
        self.log.info(f"[{self.device}] ==== testing end ====")
        print(f"[INFO][{self.device}] ==== testing end ====")
