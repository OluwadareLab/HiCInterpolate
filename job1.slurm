#!/bin/bash

#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --nodelist=g001
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=30-00:00:00
#SBATCH --job-name=hici
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate/hici_%j.log

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate film

echo "Job started on $(hostname) at $(date)"

CUDA_VISIBLE_DEVICES=0 python train.py --load-snapshot --config config_256_set_1 &
CUDA_VISIBLE_DEVICES=1 python train.py --load-snapshot --config config_256_set_2 &
wait

echo "Job finished at $(date)"