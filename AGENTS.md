# Repository Guidelines

## Project Structure & Module Organization

HiCInterpolate is a Python 3.9 deep-learning project for Hi-C matrix interpolation. Core model and workflow code lives in `src/`: training and testing in `src/train_lib.py` and `src/test_lib.py`, inference in `src/inference_lib.py`, model components in `src/feature_encoder/`, `src/feature_decoder/`, `src/flow_predictor/`, `src/interpolator/`, losses in `src/loss/`, and metrics in `src/metric/`. Entry points are `hicinterpolate.py` for training/testing and `inference.py` for inference. Configuration files are in `configs/` and `src/inference/`. Data conversion utilities live in `utils/`; evaluation and plotting scripts live in `eval_scripts/`; downstream genomics workflows live in `downstream_analysis/`. The current smoke/regression script is `test_hicinterpolate_diag.py`.

## Build, Test, and Development Commands

- `docker build -t hicinterpolate .`: build the recommended CUDA-ready environment.
- `conda create -n hicinterpolate python=3.9 && conda activate hicinterpolate`: create a local Python environment.
- `pip install -r requirements.txt`: install Python dependencies after installing the matching PyTorch CUDA wheels.
- `torchrun --standalone --nproc_per_node=1 hicinterpolate.py --distributed --train --test --config config`: run training and testing with `configs/config.yaml`.
- `python inference.py --config config`: run inference with `src/inference/config.yaml`.
- `python test_hicinterpolate_diag.py`: run the diagnostic comparison script; update hard-coded paths before use.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python naming: `snake_case` for functions, variables, and files; `PascalCase` for classes and dataclasses. Keep config keys aligned with dataclasses in `configs/config.py` and `src/inference/config.py`. Prefer explicit imports from local packages and keep CUDA/distributed behavior guarded by availability checks, as in `hicinterpolate.py`.

## Testing Guidelines

There is no formal pytest suite yet. Add focused tests or diagnostic scripts when changing metrics, data loading, loss behavior, or model output shapes. Name new tests `test_*.py`, keep sample data small, and document any required external dataset paths or pretrained weights.

## Commit & Pull Request Guidelines

Recent history mixes short status messages with Conventional Commit style, for example `fix(model): preserve sparse interpolation`. Prefer `type(scope): summary` for new commits, such as `fix(inference): handle missing model path`. Pull requests should include purpose, changed configs, commands run, dataset assumptions, GPU/CUDA requirements, and key metric changes. Include plots or screenshots when changing evaluation or visualization output.

## Security & Configuration Tips

Do not commit datasets, pretrained weights, W&B credentials, or local absolute paths. Keep environment-specific paths in YAML configs and verify output directories before long GPU runs.
