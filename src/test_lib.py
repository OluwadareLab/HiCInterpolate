from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metric import eval_metrics as eval_metric
from .interpolator import Interpolator
from .misc import plots as plot
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
        self.scc = 0
        self.pcc = 0
        self.genome_disco = 0
        # self.ncc = 0
        self.lpips = 0

    def _remove_module_prefix(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict

    def _update_metrics(self, local_steps, local_psnr, local_ssim, local_scc, local_pcc, local_genome_disco, local_ncc, local_lpips):
        self.psnr = local_psnr / local_steps
        self.ssim = local_ssim / local_steps
        self.scc = local_scc / local_steps
        self.pcc = local_pcc / local_steps
        self.genome_disco = local_genome_disco / local_steps
        self.ncc = local_ncc / local_steps
        self.lpips = local_lpips / local_steps

    def _run(self):
        local_psnr = 0
        local_ssim = 0
        local_scc = 0
        local_pcc = 0
        local_genome_disco = 0
        local_ncc = 0
        local_lpips = 0

        with torch.no_grad():
            self.model.eval()
            drawn = 0
            for _, (x0, y, x1, time_frame) in enumerate(tqdm(self.test_dl)):
                x0 = x0.to(self.device)
                y = y.to(self.device)
                x1 = x1.to(self.device)
                time_frame = time_frame.to(self.device)
                pred = self.model(x0, x1, time_frame)

                psnr_val = eval_metric.get_psnr(pred, y)
                ssim_val = eval_metric.get_ssim(pred, y)
                # scc_val = eval_metric.get_scc(pred, y)
                # pcc_val = eval_metric.get_pcc(pred, y)
                genome_disco_val = eval_metric.get_genome_disco(pred, y)
                # ncc_val = eval_metric.get_ncc(pred, y)
                lpips_val = eval_metric.get_lpips(pred, y)

                local_psnr += psnr_val.item()
                local_ssim += ssim_val.item()
                # local_scc += scc_val.item()
                # local_pcc += pcc_val.item()
                local_genome_disco += genome_disco_val
                # local_ncc += ncc_val.item()
                local_lpips += lpips_val.item()

                if drawn == 10:
                    num_examples = min(
                        self.cfg.file.num_visualization_samples, len(y))
                    x0_cpu = x0[:num_examples]
                    y_cpu = y[:num_examples]
                    pred_cpu = pred[:num_examples]
                    x1_cpu = x1[:num_examples]
                    plot.draw_hic_map(num_examples=num_examples, x0=x0_cpu,
                                      y=y_cpu, pred=pred_cpu, x1=x1_cpu, file=self.cfg.file.test_hic_map)
                drawn += 1

                del x0, y, x1, time_frame

        if self.isDistributed:
            local_steps = torch.tensor(
                self.test_steps, device=self.device)
            local_psnr = torch.tensor(
                local_psnr, device=self.device)
            local_ssim = torch.tensor(
                local_ssim, device=self.device)
            # local_scc = torch.tensor(
            #     local_scc, device=self.device)
            # local_pcc = torch.tensor(
            #     local_pcc, device=self.device)
            local_genome_disco = torch.tensor(
                local_genome_disco, device=self.device)
            # local_ncc = torch.tensor(
            #     local_ncc, device=self.device)
            local_lpips = torch.tensor(
                local_lpips, device=self.device)

            dist.all_reduce(local_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_ssim, op=dist.ReduceOp.SUM)
            # dist.all_reduce(local_scc, op=dist.ReduceOp.SUM)
            # dist.all_reduce(local_pcc, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_genome_disco, op=dist.ReduceOp.SUM)
            # dist.all_reduce(local_ncc, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_lpips, op=dist.ReduceOp.SUM)

            local_steps = local_steps.item()
            local_psnr = local_psnr.item()
            local_ssim = local_ssim.item()
            # local_scc = local_scc.item()
            # local_pcc = local_pcc.item()
            local_genome_disco = local_genome_disco.item()
            # local_ncc = local_ncc.item()
            local_lpips = local_lpips.item()

            self._update_metrics(local_steps, local_psnr, local_ssim,
                                 local_pcc, local_scc, local_genome_disco, local_ncc, local_lpips)
        else:
            self._update_metrics(self.test_steps, local_psnr, local_ssim,
                                 local_pcc, local_scc, local_genome_disco, local_ncc, local_lpips)

    def test(self):
        self.log.info(f"[{self.device}] ==== Testing Started ====")
        print(f"[INFO][{self.device}] ==== Testing Started ====")

        start_time = time.time()
        try:
            self._run()
            if self.isDistributed and self.device == 0:
                scores = f"PSNR: {format(self.psnr, '.4f')}, SSIM: {format(self.ssim, '.4f')}, GenomeDISCO: {format(self.genome_disco, '.4f')}, LPIPS: {format(self.lpips, '.4f')};"
                self.log.info(f"{scores}")
                print(f"[INFO] {scores}")

            elif not self.isDistributed:
                scores = f"PSNR: {format(self.psnr, '.4f')}, SSIM: {format(self.ssim, '.4f')}, GenomeDISCO: {format(self.genome_disco, '.4f')}, LPIPS: {format(self.lpips, '.4f')};"
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
            f"[{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        print(
            f"[INFO][{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        self.log.info(f"[{self.device}] ==== Testing End ====")
        print(f"[INFO][{self.device}] ==== Testing End ====")
