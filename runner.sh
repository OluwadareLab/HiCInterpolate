#!/bin/bash
#SBATCH --job-name=hicinterpolate_training_2
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=infinite

# 1. Initialize and activate Conda
source /opt/miniconda3/etc/profile.d/conda.sh  # Adjust path to your conda installation if needed
conda activate hicinterpolate
cd /home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate

torchrun --standalone --nproc_per_node=1 hicinterpolate.py --train --config config_5k_512
torchrun --standalone --nproc_per_node=1 hicinterpolate.py --train --config config_10k_512
torchrun --standalone --nproc_per_node=1 hicinterpolate.py --train --config config_25k_512