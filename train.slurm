#!/bin/bash

#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --nodelist=g001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --job-name=hici
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate_Set_1/_%j.log

# === Load your modules or conda env ===
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate film

# === Torchrun setup ===
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE | awk '{ print $1; exit }')
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

echo "Job started on $(hostname) at $(date)"

torchrun \
--nproc-per-node=1 \
--nnodes=1 \
--node_rank=$SLURM_NODEID \
--rdzv-id=$RANDOM \
--rdzv-backend=c10d \
--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
train.py \
--epochs 1000 \
--batch-size 32 \
--save-every 10 \
--distributed \
--load-snapshot

echo "Job finished at $(date)"

# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py --epochs 1000 --batch-size 12 --save-every 10 --data-augmentation --distributed --load-snapshot
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py --epochs 100 --batch-size 32 --save-every 10 --distributed

