# HiCInterpolate

HiCInterpolate is a deep learning framework for **4D spatiotemporal interpolation of Hi-C data**. It predicts high-resolution intermediate chromosomal contact states from two temporal anchor points, preserving critical biological features like Topologically Associating Domains (TADs) and chromatin loops.

## Project Overview

- **Purpose**: Interpolate intermediate Hi-C contact matrices between two time points.
- **Architecture**: A flow-based deep learning model featuring:
  - **Feature Extraction**: Multi-scale convolution branches (1x1, 3x3, 5x5).
  - **Feature Encoder**: U-Net style hierarchical encoding.
  - **Flow Predictor**: Predicts spatiotemporal flow to warp contact features.
  - **Feature Decoder**: Fuses warped features and skip-connections for reconstruction.
  - **Output Projection**: Jointly predicts contact intensity and a probabilistic support mask to handle Hi-C sparsity.
- **Tech Stack**: Python 3.9, PyTorch (DDP supported), PyTorch Geometric (PyG), CUDA, OmegaConf.

## Key Components

### Core Source (`src/`)
- `interpolator/`: Main model assembly.
- `feature_encoder/`, `flow_predictor/`, `feature_decoder/`: Sub-modules of the interpolation architecture.
- `data_loader/`: Custom dataset and loading logic for `.npy` patches.
- `loss/`: Combined loss functions (MSE, SSIM, VGG, Symmetry).
- `metric/`: Biological and computer vision metrics (HiCRep, GenomeDISCO, SCC, PSNR, SSIM, LPIPS).
- `inference_lib.py`, `train_lib.py`: High-level orchestration for training and prediction.

### Configuration (`configs/`)
- `config.yaml`: Main training parameters, model architecture, and loss weights.
- `config.py`: Dataclass definitions for type-safe configuration.

### Utilities (`utils/`)
- `cool_to_square_matrix.py`: Preprocessing utility to convert `.cool` files to `64x64` or `128x128` `.npy` patches.
- `reconstruct_matrix.py`: Post-processing utility to assemble predicted patches back into full contact matrices.

### Downstream & Evaluation
- `downstream_analysis/`: Scripts for A/B compartment analysis, TAD detection (EmbedTAD), and loop calling (HiCGNN).
- `eval_scripts/`: Metrics calculation and visualization (MOC, SCC, HiC map plotting).

## Getting Started

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate hicinterpolate
# OR
pip install -r requirements.txt
```

### Training
Configure `configs/config.yaml` with your data paths, then run:
```bash
torchrun --standalone --nproc_per_node=1 hicinterpolate.py --distributed --train --config config
```

### Inference
Configure `src/inference/config.yaml` and run:
```bash
python inference.py --config config
```

### Biological Analysis
Use the scripts in `downstream_analysis/` or `eval_scripts/` for post-inference validation. 
*Note: `HowToRun.md` references a `dsa.py` entry point which may be deprecated or missing; use individual scripts in `downstream_analysis/` instead.*

## Development Conventions

1. **Configuration**: Always use `OmegaConf` via `configs/config.py`. Avoid hardcoding paths.
2. **Distributed Training**: Use `torchrun` and `DDP`. The codebase is designed for multi-GPU scaling.
3. **Data Format**: Input patches are strictly `.npy` files, typically $64 \times 64$ or $128 \times 128$.
4. **Metrics**: Validation uses a "Composite Score" combining biological (SCC, HiCRep, GenomeDISCO) and structural (F1, Density) metrics.
