import os
import time
import gc
import traceback
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
from interpolator import Interpolator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

METRIC_KEYS = ("psnr", "ssim", "genome_disco", "hicrep", "lpips")


class MetricAccumulator:
    def __init__(self):
        self.sums = {k: 0.0 for k in METRIC_KEYS}
        self.valid = {k: 0 for k in METRIC_KEYS}

    def update(self, metrics: Dict[str, torch.Tensor]):
        for key in METRIC_KEYS:
            value = metrics[key]
            scalar = value.item() if torch.is_tensor(value) else float(value)
            if scalar == scalar and scalar not in (float("inf"), float("-inf")):
                self.sums[key] += scalar
                self.valid[key] += 1

    def means(self) -> Dict[str, float]:
        return {
            k: (self.sums[k] / self.valid[k] if self.valid[k] > 0 else float("nan"))
            for k in METRIC_KEYS
        }


def get_or_load_model(
    cfg,
    log,
    model_path: str,
    cache: Dict[str, Tuple[torch.nn.Module, torch.device]],
    isDistributed: bool = False,
) -> Tuple[torch.nn.Module, torch.device]:
    if model_path in cache:
        return cache[model_path]
    wrapper = HiCInterpolate(
        cfg=cfg, log=log, model=model_path, dl=None, isDistributed=isDistributed
    )
    cache[model_path] = wrapper._get_model()
    return cache[model_path]


@torch.no_grad()
def evaluate_with_baselines(
    model: torch.nn.Module,
    device,
    dl: DataLoader,
    linear_fn: Callable,
    of_fn: Callable,
    show_progress: bool = True,
) -> Optional[Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]:
    from src.metric.eval_metrics import get_eval_metrics_gpu

    model_acc = MetricAccumulator()
    linear_acc = MetricAccumulator()
    of_acc = MetricAccumulator()

    iterator = tqdm(dl) if show_progress else dl
    model.eval()
    for batch in iterator:
        if batch is None:
            continue
        x1, x2, x3, time_frame = batch
        x1 = x1.to(device)
        x3 = x3.to(device)
        time_frame = time_frame.to(device)
        pred = model(x1, x3, time_frame)
        pred = pred.clamp_min(0.0)
        x2 = x2.to(device)

        model_acc.update(get_eval_metrics_gpu(pred, x2))
        linear_pred = linear_fn(x1, x3).clamp_min(0.0)
        of_pred = of_fn(x1, x3).clamp_min(0.0)
        linear_acc.update(get_eval_metrics_gpu(linear_pred, x2))
        of_acc.update(get_eval_metrics_gpu(of_pred, x2))

    if model_acc.valid["psnr"] == 0:
        return None

    return model_acc.means(), linear_acc.means(), of_acc.means()


def metrics_tuple(
    model_metrics: Dict[str, float],
    linear_metrics: Dict[str, float],
    of_metrics: Dict[str, float],
) -> Tuple[float, ...]:
    return (
        model_metrics["psnr"],
        model_metrics["ssim"],
        model_metrics["genome_disco"],
        model_metrics["hicrep"],
        model_metrics["lpips"],
        linear_metrics["psnr"],
        linear_metrics["ssim"],
        linear_metrics["genome_disco"],
        linear_metrics["hicrep"],
        linear_metrics["lpips"],
        of_metrics["psnr"],
        of_metrics["ssim"],
        of_metrics["genome_disco"],
        of_metrics["hicrep"],
        of_metrics["lpips"],
    )


class HiCInterpolate:
    def __init__(self, cfg, log, model: str, dl: Optional[DataLoader], isDistributed: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.isDistributed = dist.is_available() and dist.is_initialized()
        if isDistributed:
            self.device = int(os.environ["LOCAL_RANK"])
            self.model = Interpolator(self.cfg).to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
            loc = f"cuda:{self.device}"
            snapshot = torch.load(model, map_location=loc)
            self.model.load_state_dict(snapshot["model"])
        else:
            self.device = self.cfg.device
            self.model = Interpolator(self.cfg).to(self.device)
            snapshot = torch.load(model, map_location=self.device)
            state_dict = self._remove_module_prefix(snapshot["model"])
            self.model.load_state_dict(state_dict)

        self.dl = dl
        self.steps = len(self.dl) if dl is not None else 0
        self.batch_size = dl.batch_size if dl is not None else cfg.data.batch_size
        self.pred_list = []

    def _get_model(self):
        return self.model, self.device

    def _remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict

    def _run(self):
        with torch.no_grad():
            self.model.eval()
            for _, (x1, x3, time_frame) in enumerate(tqdm(self.dl)):
                x1 = x1.to(self.device)
                x3 = x3.to(self.device)
                time_frame = time_frame.to(self.device)
                pred = self.model(x1, x3, time_frame)
                self.pred_list.append(pred)
                del x1, x3, time_frame

    def _get_prediction(self):
        return self.pred_list

    def _inference(self):
        self.log.info(f"[{self.device}] ==== Inference Started ====")
        print(f"[INFO][{self.device}] ==== Inference Started ====")
        start_time = time.time()
        try:
            self._run()
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
            f"[{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds"
        )
        print(
            f"[INFO][{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds"
        )
        self.log.info(f"[{self.device}] ==== Inference End ====")
        print(f"[INFO][{self.device}] ==== Inference End ====")
