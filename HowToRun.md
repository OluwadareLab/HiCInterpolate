# HiCInterpolate Usage Guide
## Package Requirements
We used python3.9 and pytorch to implement our HiCInterpolate model. By default we used GPU. Here is the package list we used in this model development.

* torch
* torchvision
* torchaudio
* tensorflow
* torch-geometric
* torch-sparse
* torch-scatter
* cupy-cuda11x
* cugraph-cu11
* numpy
* scipy
* pandas
* scikit-learn
* networkx
* matplotlib
* seaborn
* cooler
* tqdm
* omegaconf
* torchmetrics
* lpips
* wandb
* gensim
* fastdtw

We provided 3 environment setup options. You can choose one of them according to your preferences. We highly recommend using our prebuilt Docker image. Follow the steps one by one to get into HiCInterpolate.

## Step 1: Clone Repository
```bash
git clone https://github.com/OluwadareLab/HiCInterpolate.git
cd HiCInterpolate
```

## Step 2: Environment Setup
### Option A: Docker (Recommended)
  You can use our prebuilt Docker image or build it from scratch using the Dockerfile we provided.
  - Create HiCInterpolate Docker image:
    - Option (i): Use prebuilt Docker image (**Highly recommanded**):
      ```bash
      docker pull oluwadarelab/hicinterpolate
      ```
    - Option (ii): Build Docker image from scratch:
      ```bash
      docker build -t hicinterpolate .
      ```
  - Create Docker Container:
    ```bash
    docker run -itd --gpus all -v ${PWD}:${PWD} --name hicinterpolate oluwadarelab/hicinterpolate:latest
    ```
    OR
    ```bash
    docker run -itd --gpus all -v ${PWD}:${PWD} --name hicinterpolate hicinterpolate:latest
    ```
  - Enter Docker Container
    ```bash
    docker exec -it hicinterpolate bash
    ```

### Option B: Conda Environment
  - Create conda environment:
    ```bash
    conda create -n hicinterpolate -y
    ```
  - Activate your conda environment:
    ```bash
    conda activate hicinterpolate
    ```

### Option C: Python pip Environment
  - Create Python pip environment:
    ```bash
    pip install --upgrade pip
    ```
  - Activate your pip environment:
    ```bash
    pip install --upgrade pip
    ```
  - Install required dependencies:
    ```bash
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```
## Step 3: Data
In this step, you will prepare your dataset. HiCInterpolate requires a *64x64* square Hi-C contact matrix in *.npy* format from *nxn* full Hi-C contact matrix. To accelerate the process, we provided a utility script to convert .cool format Hi-C contact matrix into *nxn* Hi-C contact matrix, dividing them into *64x64* patch sizes. You will find cool_to_square_matrix.py under utils folder:

```text
HiCInterpolate/
├── utils/
│   ├── cool_to_square_matrix.py
```

If you want our dataset used in this study to train our model, please download it from: [HiCInterpolate Dataset](https://doi.org/10.5281/zenodo.18319340)

## Step 4: Train HiCInterpolate
To train HiCInterpolate, first you need to define your configurations in config.yaml file. After that, run the training command to train HiCInterpolate. You will find the config.yaml file inside the configs folder:

```text
HiCInterpolate/
├── configs/
│   └── config.yaml
```

- Prepare your config.yaml file:
Edit the following configurations according to your preferences.

```text
dir:
  root: <root folder>
  model_state: <where you want to save your model states>
  output: <output folder where you want to save training details>
  data: <dataset folder where your datasets are. This should contain the dataset_dict.txt file as a dataset dictionary and the 6xx64 patchs as .npy format>
```

- Execute training
```bash
torchrun --standalone --nproc_per_node=1 hicinterpolate.py --distributed --train --test --config config
```

- Output
  * Inside the model state folder, you will find two model weights:
    - hicinterpolate_64.pt: this is our best model
    - hicinterpolate_64_snapshot.pt: this is the last weights after every 10 epochs.
  * Inside the output folder, you will find training details such as cvs file recording every epoch's metrics.


## Step 5: Pretrained Model
If you want to use our best pretrained model weight for your usage, you can download it from: [HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_64.pt)

Additionally, we also provided our last snapshot out of 300 epochs. You will find it from: [Snapshot](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/hicinterpolate_64_snapshot.pt)


## Step 6: Inference
- Data preprocessing
Like training steps, you need to provide a 64x64 patch size of your data in .npy format. We provided a utility script for data preprocessing for inference. You will find inf_cool_to_square_matrix.py under the utils folder:

```text
HiCInterpolate/
├── utils/
│   ├── inf_cool_to_square_matrix.py
```

- Prepare your config.yaml file:
You will find a different config.yaml file for inference inside the src/inference folder:
```text
HiCInterpolate/
├── src/
│   └── inference/
│       └── config.yaml
```

Edit the following configurations according to your preferences.

```text
dir:
  root: <root folder>
  model_state: <where your model weights are saved. It should be hicinterpolate_64.pt named file.>
  output: <output folder where you want to save predicted Hi-C contact matrix in .npy. format.>
  data: <dataset folder where your datasets are. This should contain the dataset_dict.txt file as a dataset dictionary and the 6xx64 patchs as .npy format>
```

- Execute inference
```bash
torchrun inference.py --config config
```
- Output
  * `inferenced.npy` file, and it contains the predicted 64x64 patches and is structured as you provided the input 64x64 patches. If you want to reconstruct the original nxn Hi-C contact matrix for downstream analysis, we provided a reconstruct function in the utils folder.
```text
HiCInterpolate/
├── utils/
│   ├── reconstruct_matrix.py
```
It requires a predicted .npy file, the original matrix shape, and the output folder path.

## Step 7: Downstream Analysis
This program provides several command-line flags and arguments that allow users to run different downstream analyses on Hi-C contact matrices.

* `-ab` or `--ab-compartments` flag enables A/B compartment analysis. When this flag is used, the user must also specify the genomic region of interest using the `--start` and `--end` arguments, which define the start and end bin indices of the region.

* `-l` or `--loop` flag enables chromatin loop analysis. This analysis requires the chromosome identifier provided through the `--chromosome` argument and the genome assembly specified by the `--genome_id` argument. Supported genome identifiers include hg19, hg38, mm9, and mm10.

* `-t` or `--tads` flag enables Topologically Associating Domain (TAD) analysis. This flag does not require additional arguments beyond the input matrix and resolution.

* `-s` or `--structure` flag enables reconstruction of 3D genome structure coordinates. Similar to A/B compartment analysis, this option requires the `--start` and `--end` arguments to define the genomic region to be modeled.

* `-i` or `--input` argument specifies the input Hi-C contact matrix. The input must be a square N × N matrix provided in either `.txt` or `.npy` format.

* `-o` or `--output` argument defines the output directory where all generated results will be saved.

* `-r` or `--resolution` argument specifies the resolution, or bin size, of the Hi-C contact matrix and should be provided as an integer value.

* `-c` or `--chromosome` argument specifies the chromosome number or name and is required when loop analysis is enabled.

* `-g` or `--genome_id` argument specifies the genome assembly to be used for loop analysis. If not explicitly provided, it defaults to `hg19`.

* `-sc` or `--start` argument specifies the starting bin index of the genomic region and is required for A/B compartment analysis and 3D structure prediction.

* `-ec` or `--end` argument specifies the ending bin index of the genomic region and is required for A/B compartment analysis and 3D structure prediction.

Multiple analysis flags can be combined in a single run, provided that all required arguments for each selected analysis are supplied.


### A/B Compartment:
```bash
python dsa.py -ab -i /path/to/input.txt -o /path/to/output -r 10000 -sc 500 -ec 1000
```

* Output:
You will find ab_compartment.png file under your output folder.
![A/B Compartments](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/chr11_yt.png)

### Chromatin Loops
```bash
python dsa.py -l -i /path/to/input.txt -o /path/to/output -r 10000 -c 21 -g hg38
```

* Output:
- You will find a loops subdirectory under your output directory. Inside loops directory, you will find merged_loops.bedpe fild containg chromatin loops coordinates and other details.

| chr1  | x1       | x2       | chr2  | y1       | y2       | name | score | strand1 | strand2 | color      | observed | expectedBL | expectedDonut | expectedH | expectedV | fdrBL        | fdrDonut     | fdrH        | fdrV        | numCollapsed | centroid1 | centroid2 | radius  |
|-------|----------|----------|-------|----------|----------|------|-------|---------|---------|------------|----------|------------|---------------|-----------|-----------|--------------|--------------|------------|------------|--------------|-----------|-----------|---------|
| chr21 | 9660000  | 9670000  | chr21 | 13450000 | 13460000 | .    | .     | .       | .       | 0,255,255  | 51.0     | 12.553924  | 12.484794     | 12.641066 | 12.764403 | 2.765284E-9 | 4.4349378E-9 | 3.326869E-9 | 4.239684E-9 | 6            | 9665000  | 13450000 | 11180   |
| chr21 | 32650000 | 32660000 | chr21 | 34570000 | 34580000 | .    | .     | .       | .       | 0,255,255  | 38.0     | 13.415261  | 13.349878     | 13.635471 | 13.872615 | 0.00117912   | 0.0018885597 | 0.0013067651 | 0.0016970497 | 1            | 32655000 | 34575000 | 0       |
| chr21 | 44210000 | 44220000 | chr21 | 44910000 | 44920000 | .    | .     | .       | .       | 0,255,255  | 172.0    | 14.372463  | 14.253437     | 14.980931 | 14.691927 | 0.0          | 0.0          | 0.0         | 0.0         | 10           | 44216000 | 44917000 | 20125   |
| chr21 | 30730000 | 30740000 | chr21 | 31370000 | 31380000 | .    | .     | .       | .       | 0,255,255  | 70.0     | 15.406109  | 19.444122     | 15.603013 | 16.010157 | 6.8534376E-20 | 1.6903339E-15 | 1.4837505E-15 | 1.4664573E-15 | 6            | 30736666 | 31373333 | 11785   |
| chr21 | 9690000  | 9700000  | chr21 | 9970000  | 9980000  | .    | .     | .       | .       | 0,255,255  | 140.0    | 16.30356   | 23.35135      | 20.614822 | 19.670992 | 0.0          | 0.0          | 0.0         | 0.0         | 9            | 9695000  | 9975000  | 14142   |


### TADs

```bash
python dsa.py -t -i /path/to/input.txt -o /path/to/output -r 10000
```

* Output:
<*>.txt (BED-like) file contains TAD regions as follows-

| Start     | End       |
|-----------|-----------|
| 9440000   | 9550000   |
| 9670000   | 9830000   |
| 9870000   | 10070000  |
| 10100000  | 10690000  |
| 10830000  | 11010000  |
| 11200000  | 14340000  |
| 14350000  | 15200000  |
| 15200000  | 15300000  |
| 15300000  | 15410000  |
| 15410000  | 15520000  |


### 3D Structure

```bash
python dsa.py -s -i /path/to/input.txt -o /path/to/output -sc 2500 -ec 3000
```

* Output:
You will find <sc>_<ec>_structure.pdb file in your output directory contains predicted coordinates of 3D structures.

| Atom | Serial | Name | Residue | Chain | X       | Y       | Z       | Occupancy | TempFactor |
|------|--------|------|---------|-------|---------|---------|---------|-----------|------------|
| ATOM | 1      | CA   | MET     | B1    | -56.821 | 14.029  | -35.235 | 0.20      | 10.00      |
| ATOM | 2      | CA   | MET     | B2    | -53.479 | 11.632  | -35.380 | 0.20      | 10.00      |
| ATOM | 3      | CA   | MET     | B3    | -54.224 | 11.505  | -35.357 | 0.20      | 10.00      |
| ATOM | 4      | CA   | MET     | B4    | -53.434 | 10.759  | -35.399 | 0.20      | 10.00      |
| ATOM | 5      | CA   | MET     | B5    | -51.788 | 10.521  | -36.385 | 0.20      | 10.00      |
| ATOM | 6      | CA   | MET     | B6    | -52.202 | 10.266  | -35.742 | 0.20      | 10.00      |
| ATOM | 7      | CA   | MET     | B7    | -51.906 | 10.359  | -35.867 | 0.20      | 10.00      |
| ATOM | 8      | CA   | MET     | B8    | -51.583 | 10.242  | -36.271 | 0.20      | 10.00      |
| ATOM | 9      | CA   | MET     | B9    | -51.797 | 10.264  | -35.798 | 0.20      | 10.00      |
