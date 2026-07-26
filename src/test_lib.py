from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.interpolator import Interpolator
from src.misc import plots as plot
from src.metric import metrics as eval_metric
import torch.distributed as dist
import torch
import traceback
import gc
import time
import os


class Tester:
    def __init__(self, cfg, log, model: str, test_dl: DataLoader, isDistributed: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.isDistributed = dist.is_available() and dist.is_initialized()
        if isDistributed:
            self.device = int(os.environ["LOCAL_RANK"])
            self.model = Interpolator().to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
            loc = f"cuda:{self.device}"
            snapshot = torch.load(model, map_location=loc)
            self.model.load_state_dict(snapshot['model'])

        else:
            self.device = self.cfg.device
            self.model = Interpolator().to(self.device)
            snapshot = torch.load(model, map_location=self.device)
            state_dict = self._remove_module_prefix(snapshot['model'])
            self.model.load_state_dict(state_dict)

        self.test_dl = test_dl
        self.test_steps = len(self.test_dl)
        self.batch_size = test_dl.batch_size

        self.psnr = 0
        self.ssim = 0
        self.ms_ssim = 0
        self.hicrep = 0

    def _remove_module_prefix(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict

    def _update_metrics(self, local_steps, local_psnr, local_ssim, local_ms_ssim, local_hicrep):
        self.psnr = local_psnr / local_steps
        self.ssim = local_ssim / local_steps
        self.ms_ssim = local_ms_ssim / local_steps
        self.hicrep = local_hicrep / local_steps

    def _run(self):
        local_psnr = 0
        local_ssim = 0
        local_ms_ssim = 0
        local_hicrep = 0

        with torch.no_grad():
            self.model.eval()
            drawn = 0
            for _, batch in enumerate(tqdm(self.test_dl)):
                if batch is None:
                    continue
                x0, y, x1 = batch
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)
                outputs = self.model(x0, x1)
                pred = outputs["final"] if isinstance(
                    outputs, dict) else outputs

                psnr_val = eval_metric.get_psnr_from_tensor(pred, y)
                ssim_val = eval_metric.get_ssim_from_tensor(pred, y)
                ms_ssim_val = eval_metric.get_ms_ssim_from_tensor(pred, y)

                hicrep_val = eval_metric.get_hicrep_from_tensor(
                    pred, y, resol=self.cfg.data.resolution, patch_size=self.cfg.data.patch, h=5)

                local_psnr += psnr_val.item()
                local_ssim += ssim_val.item()
                local_ms_ssim += ms_ssim_val.item()
                local_hicrep += hicrep_val.item()

                if drawn == 0:
                    num_examples = min(
                        self.cfg.file.num_visualization_samples, len(y))
                    x0_cpu = x0[:num_examples]
                    y_cpu = y[:num_examples]
                    pred_cpu = pred[:num_examples]
                    x1_cpu = x1[:num_examples]
                    plot.draw_hic_map(num_examples=num_examples, x0=x0_cpu,
                                      y=y_cpu, pred=pred_cpu, x1=x1_cpu, file=self.cfg.file.test_hic_map)
                drawn += 1

                del x0, y, x1

        if self.isDistributed:
            local_steps = torch.tensor(
                self.test_steps, device=self.device)
            local_psnr = torch.tensor(
                local_psnr, device=self.device)
            local_ssim = torch.tensor(
                local_ssim, device=self.device)
            local_ms_ssim = torch.tensor(
                local_ms_ssim, device=self.device)
            local_hicrep = torch.tensor(
                local_hicrep, device=self.device)

            dist.all_reduce(local_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_ssim, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_ms_ssim, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_hicrep, op=dist.ReduceOp.SUM)

            local_steps = local_steps.item()
            local_psnr = local_psnr.item()
            local_ssim = local_ssim.item()
            local_ms_ssim = local_ms_ssim.item()
            local_hicrep = local_hicrep.item()

            self._update_metrics(local_steps, local_psnr,
                                 local_ssim, local_ms_ssim, local_hicrep)
        else:
            self._update_metrics(self.test_steps, local_psnr,
                                 local_ssim, local_ms_ssim, local_hicrep)

    def test(self):
        self.log.info(f"[{self.device}] ==== Testing Started ====")
        print(f"[INFO][{self.device}] ==== Testing Started ====")
        start_time = time.time()
        try:
            self._run()
            if self.isDistributed and self.device == 0:
                scores = f"PSNR: {format(self.psnr, '.4f')}, SSIM: {format(self.ssim, '.4f')}, MS-SSIM: {format(self.ms_ssim, '.4f')}, HiCRep: {format(self.hicrep, '.4f')};"
                self.log.info(f"{scores}")
                print(f"[INFO] {scores}")

            elif not self.isDistributed:
                scores = f"PSNR: {format(self.psnr, '.4f')}, SSIM: {format(self.ssim, '.4f')}, MS-SSIM: {format(self.ms_ssim, '.4f')}, HiCRep: {format(self.hicrep, '.4f')};"
                self.log.info(f"{scores}")
                print(f"[INFO] {scores}")

            return self.psnr, self.ssim, self.ms_ssim, self.hicrep
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
            f"[{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        print(
            f"[INFO][{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        self.log.info(f"[{self.device}] ==== Testing End ====")
        print(f"[INFO][{self.device}] ==== Testing End ====")
