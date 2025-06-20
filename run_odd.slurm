#!/bin/bash

#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --nodelist=g001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=30-00:00:00
#SBATCH --job-name=hici
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate/hici_%j.log

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate film

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE | awk '{ print $1; exit }')

echo "Job started on $(hostname) at $(date)"

PORT_1=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
PORT_2=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# === Run job 1 in background ===
CUDA_VISIBLE_DEVICES=0 torchrun \
--nproc-per-node=1 \
--nnodes=1 \
--node_rank=0 \
--rdzv-id=job1_$RANDOM \
--rdzv-backend=c10d \
--rdzv-endpoint=$MASTER_ADDR:$PORT_1 \
--master_addr=$MASTER_ADDR \
--master_port=$PORT_1 \
train.py \
--distributed \
--load-snapshot \
--config config_128_set_1 &

# === Run job 2 in background ===
CUDA_VISIBLE_DEVICES=1 torchrun \
--nproc-per-node=1 \
--nnodes=1 \
--node_rank=0 \
--rdzv-id=job2_$RANDOM \
--rdzv-backend=c10d \
--rdzv-endpoint=$MASTER_ADDR:$PORT_2 \
--master_addr=$MASTER_ADDR \
--master_port=$PORT_2 \
train.py \
--distributed \
--load-snapshot \
--config config_128_set_2 &

# Wait for both jobs to complete
wait

echo "Job finished at $(date)"
