import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_metric_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")
if "src.metric" not in sys.modules:
    sys.modules["src.metric"] = types.ModuleType("src.metric")

for rel, name in [
    ("src/metric/genome_disco.py", "src.metric.genome_disco"),
    ("src/metric/genome_disco_gpu.py", "src.metric.genome_disco_gpu"),
    ("src/metric/hicrep.py", "src.metric.hicrep"),
    ("src/metric/hicrep_gpu.py", "src.metric.hicrep_gpu"),
    ("src/metric/ent3c.py", "src.metric.ent3c"),
    ("src/metric/metrics.py", "src.metric.metrics"),
]:
    _load_metric_module(rel, name)

get_genome_disco = sys.modules["src.metric.metrics"].get_genome_disco
get_genome_disco_gpu = sys.modules["src.metric.metrics"].get_genome_disco_gpu


def _random_hic_patches(batch_size: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    preds = torch.rand(batch_size, 1, n, n, generator=gen)
    target = torch.rand(batch_size, 1, n, n, generator=gen)
    return preds, target


def _assert_parity(preds: torch.Tensor, target: torch.Tensor, rtol: float = 1e-6, atol: float = 1e-6):
    cpu_score = get_genome_disco(preds, target).item()
    gpu_score = get_genome_disco_gpu(preds, target).item()
    assert cpu_score == pytest.approx(gpu_score, rel=rtol, abs=atol)


def test_genome_disco_gpu_matches_cpu_single_batch():
    preds, target = _random_hic_patches(batch_size=1, n=64, seed=0)
    _assert_parity(preds, target)


def test_genome_disco_gpu_matches_cpu_multi_batch():
    preds, target = _random_hic_patches(batch_size=8, n=64, seed=1)
    _assert_parity(preds, target)


def test_genome_disco_gpu_matches_cpu_per_sample_mean():
    preds, target = _random_hic_patches(batch_size=4, n=64, seed=2)
    per_sample = [
        get_genome_disco(preds[i : i + 1], target[i : i + 1]).item()
        for i in range(preds.size(0))
    ]
    batched = get_genome_disco_gpu(preds, target).item()
    assert batched == pytest.approx(sum(per_sample) / len(per_sample), rel=1e-6, abs=1e-6)


def test_genome_disco_gpu_matches_cpu_zero_row():
    n = 32
    preds = torch.rand(2, 1, n, n)
    target = torch.rand(2, 1, n, n)
    preds[:, :, 0, :] = 0.0
    target[:, :, 5, :] = 0.0
    _assert_parity(preds, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_genome_disco_gpu_matches_cpu_on_cuda():
    preds, target = _random_hic_patches(batch_size=4, n=64, seed=3)
    cpu_score = get_genome_disco(preds, target).item()
    gpu_score = get_genome_disco_gpu(preds.cuda(), target.cuda()).item()
    assert cpu_score == pytest.approx(gpu_score, rel=1e-6, abs=1e-6)
