import os
import time
import gc
import traceback
import torch
import torch.distributed as dist
from interpolator import Interpolator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class HiCInterpolate:
    def __init__(self, cfg, log, model: str, dl: DataLoader, isDistributed: bool = False) -> None:
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

        self.dl = dl
        self.steps = len(self.dl)
        self.batch_size = dl.batch_size
        self.pred_list = []

    def _remove_module_prefix(self, state_dict):
        from collections import OrderedDict
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
            f"[{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        print(
            f"[INFO][{self.device}] Total time taken: {format((end_time-start_time), '.2f')} seconds")
        self.log.info(f"[{self.device}] ==== Inference End ====")
        print(f"[INFO][{self.device}] ==== Inference End ====")
