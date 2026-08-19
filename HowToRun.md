# HiCInterpolate Usage Guide

## Overview

HiCInterpolate is implemented in Python 3.9 with PyTorch and designed to leverage GPU acceleration. This guide will walk you through the installation, training, and analysis workflows.

## System Requirements

### Python Version
- Python 3.9

### Key Dependencies
- **Deep Learning:** PyTorch, TensorFlow, torch-geometric
- **Scientific Computing:** NumPy, SciPy, pandas, scikit-learn
- **Genome Analysis:** NetworkX, cooler, cooltools
- **Visualization:** Matplotlib, seaborn
- **Utilities:** tqdm, omegaconf, torchmetrics, lpips, wandb, gensim, fastdtw
- **GPU Support:** CUDA (cupy-cuda11x, cugraph-cu11)

### Environment Setup Options

We provide three setup options. **We highly recommend Docker for the easiest setup:**

1. **Docker** (Recommended)
2. **Conda**
3. **pip**

## Getting Started

### Step 1: Clone Repository

```bash
git clone https://github.com/OluwadareLab/HiCInterpolate.git
cd HiCInterpolate
```

### Step 2: Environment Setup
#### Option A: Docker (Recommended)

**Using prebuilt Docker image** (fastest option):
```bash
docker pull oluwadarelab/hicinterpolate
```

**Or build from scratch:**
```bash
docker build -t hicinterpolate .
```

**Create and enter the container:**
```bash
docker run -itd --gpus all -v ${PWD}:${PWD} --name hicinterpolate oluwadarelab/hicinterpolate:latest
docker exec -it hicinterpolate bash
```

#### Option B: Conda Environment

```bash
conda create -n hicinterpolate python=3.9
conda activate hicinterpolate
```

#### Option C: pip Environment

```bash
python -m venv hicinterpolate
source hicinterpolate/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

---

### Step 3: Prepare Your Data

HiCInterpolate requires **64×64 square Hi-C contact matrices** in `.npy` format, extracted from full n×n Hi-C contact matrices.

**Data Conversion Utility:**

We provide a preprocessing script to convert `.cool` format matrices into 64×64 patches:

```
HiCInterpolate/
├── utils/
│   └── cool_to_square_matrix.py
```

**Download Sample Dataset:**

To use the dataset from our paper, download it here: [HiCInterpolate Dataset](https://doi.org/10.5281/zenodo.18319340)

---

### Step 4: Train the Model

**1. Configure training settings:**

Edit `configs/config.yaml` with your preferences:

```yaml
dir:
  root: <root_folder>
  model_state: <path_to_save_model_weights>
  output: <path_for_training_output>
  data: <dataset_folder>  # Should contain dataset_dict.txt and 64×64 patches (.npy)
```

**2. Start training:**

```bash
torchrun --standalone --nproc_per_node=1 hicinterpolate.py --distributed --train --test --config config
```

**3. Output files:**

Inside `model_state` folder:
- `hicinterpolate_64.pt` — Best model weights
- `hicinterpolate_64_snapshot.pt` — Latest weights (saved every 10 epochs)

Inside `output` folder:
- Training metrics and logs (CSV format)

---

### Step 5: Use Pretrained Models

**Best Model (25KB):**

[HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate.pt)
**Latest Snapshot:**
[Snapshot (100 epochs)](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_snapshot.pt)

**Other Model:**

***5KB***
[HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_5.pt)
**Latest Snapshot:**
[Snapshot (100 epochs)](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_snapshot_5.pt)

***10KB***
[HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_10.pt)
**Latest Snapshot:**
[Snapshot (100 epochs)](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_64_snapshot_10.pt)



---

### Step 6: Run Inference

**1. Preprocess inference data:**

Similar to training, prepare 64×64 patches in `.npy` format using our utility script:

```
HiCInterpolate/
├── utils/
│   └── inf_cool_to_square_matrix.py
```

**2. Configure inference settings:**

Edit `src/inference/config.yaml`:

```yaml
dir:
  root: <root_folder>
  model_state: <path_to_pretrained_model>  # Should contain hicinterpolate_64.pt
  output: <output_folder>  # Where predicted matrices will be saved
  data: <dataset_folder>  # 64×64 patches (.npy format)
```

**3. Run inference:**

```bash
python inference.py --config config
```

**4. Output files:**

- `inferenced.npy` — Predicted 64×64 patches structured as input
- Use `utils/reconstruct_matrix.py` to reconstruct the full n×n matrix

```
HiCInterpolate/
├── utils/
│   └── reconstruct_matrix.py
```

Usage: Provide the predicted `.npy` file, original matrix shape, and output directory.

---

### Step 7: Downstream Analysis

Perform various analyses on Hi-C contact matrices using our downstream analysis tools.

### Command-Line Interface

**General usage:**
```bash
python dsa.py [options] -i <input> -o <output> [-r resolution] [additional args]
```

### Available Analyses

#### A/B Compartments

Identify active (A) and inactive (B) compartments in a genomic region using cooltools `eigs_cis`. The selected region should span at least 50 bins.

**Required arguments:**
- `-ab` — Enable A/B compartment analysis
- `-sc` — Starting bin index
- `-ec` — Ending bin index

**Optional arguments:**
- `-c` — Chromosome identifier (default: `chr1`)

**Command:**
```bash
python dsa.py -ab -i /path/to/input.txt -o /path/to/output -r 10000 -c 11 -sc 500 -ec 1000
```

**Output:**
- `ab_compartment.png` — Compartment visualization

---

#### Chromatin Loops

Detect chromatin loops with Mustache after converting the contact matrix to `.cool`. Requires `cooler` and the Mustache CLI (`pip install mustache-hic`).

**Required arguments:**
- `-l` — Enable loop analysis
- `-c` — Chromosome identifier

**Command:**
```bash
python dsa.py -l -i /path/to/input.txt -o /path/to/output -r 10000 -c 21
```

**Output:**
- `chr21.mustache.tsv` — Loop calls in Mustache TSV format

| BIN1_CHR | BIN1_START | BIN1_END | BIN2_CHR | BIN2_START | BIN2_END | FDR      |
|----------|------------|----------|----------|------------|----------|----------|
| chr21    | 9660000    | 9670000  | chr21    | 13450000   | 13460000 | 2.8e-9   |
| chr21    | 32650000   | 32660000 | chr21    | 34570000   | 34580000 | 0.0012   |

---

#### Topologically Associating Domains (TADs)

Identify TAD boundaries and regions with EmbedTAD.

**Required arguments:**
- `-t` — Enable TAD analysis

**Command:**
```bash
python dsa.py -t -i /path/to/input.txt -o /path/to/output -r 10000
```

**Output:**
- EmbedTAD BED-like files under the output directory (Start, End)

| Start     | End       |
|-----------|-----------|
| 9440000   | 9550000   |
| 9670000   | 9830000   |
| 9870000   | 10070000  |

---

#### 3D Genome Structure

Reconstruct 3D coordinates with FLAMINGOr and write a PDB structure. Requires `Rscript` with the FLAMINGOr package and `downstream_analysis/flamingo_main_run.R`.

**Required arguments:**
- `-s` — Enable structure prediction
- `-r` — Resolution (bin size in bp)
- `-c` — Chromosome identifier
- `-sc` — Starting bin index
- `-ec` — Ending bin index

**Command:**
```bash
python dsa.py -s -i /path/to/input.txt -o /path/to/output -r 10000 -c 11 -sc 2500 -ec 3000
```

**Output:**
- `flamingo_coords.tsv` — Predicted 3D coordinates
- `flamingo_structure.pdb` — Structure in PDB format

| Atom | Serial | Name | Residue | X       | Y       | Z       |
|------|--------|------|---------|---------|---------|---------|
| ATOM | 1      | CA   | MET     | -56.821 | 14.029  | -35.235 |
| ATOM | 2      | CA   | MET     | -53.479 | 11.632  | -35.380 |

---

### Input/Output Parameters

| Parameter | Flag | Description | Required |
|-----------|------|-------------|----------|
| Input matrix | `-i, --input` | Hi-C matrix in `.txt` or `.npy` format | Yes |
| Output directory | `-o, --output` | Directory for results | Yes |
| Resolution | `-r, --resolution` | Bin size in base pairs | Yes |
| Chromosome | `-c, --chromosome` | Chromosome ID (`21` or `chr21`) | With `-l`, `-s`; optional with `-ab` |
| Start bin | `-sc, --start` | Starting bin index | With `-ab`, `-s` |
| End bin | `-ec, --end` | Ending bin index | With `-ab`, `-s` |

### Combining Multiple Analyses

Run multiple analyses in a single command by combining flags (ensure all required arguments are provided):

```bash
python dsa.py -ab -l -t -s -i /path/to/input.txt -o /path/to/output -r 10000 -c 11 -sc 500 -ec 1000
```
