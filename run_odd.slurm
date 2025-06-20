#!/bin/bash

#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --nodelist=g001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=30-00:00:00
#SBATCH --job-name=hici_6job
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate/hici_6job_%j.log

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate film

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE | awk '{ print $1; exit }')

echo "Job started on $(hostname) at $(date)"

# Dynamically get 6 free ports
read PORT1 PORT2 PORT3 PORT4 PORT5 PORT6 < <(python -c "
import socket
def get_port(): s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); return p
print(get_port(), get_port(), get_port(), get_port(), get_port(), get_port())
")

# === GPU 0 Jobs ===
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job1_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT1 \
  --master_addr=$MASTER_ADDR --master_port=$PORT1 \
  train.py --distributed --load-snapshot --config config_128_set_1 &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job2_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT2 \
  --master_addr=$MASTER_ADDR --master_port=$PORT2 \
  train.py --distributed --load-snapshot --config config_128_set_2 &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job3_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT3 \
  --master_addr=$MASTER_ADDR --master_port=$PORT3 \
  train.py --distributed --load-snapshot --config config_128_set_3 &

# === GPU 1 Jobs ===
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job4_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT4 \
  --master_addr=$MASTER_ADDR --master_port=$PORT4 \
  train.py --distributed --load-snapshot --config config_128_set_4 &

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job5_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT5 \
  --master_addr=$MASTER_ADDR --master_port=$PORT5 \
  train.py --distributed --load-snapshot --config config_128_set_5 &

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job6_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT6 \
  --master_addr=$MASTER_ADDR --master_port=$PORT6 \
  train.py --distributed --load-snapshot --config config_128_set_6 &

wait
echo "Job finished at $(date)"
