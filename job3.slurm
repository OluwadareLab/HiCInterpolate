#!/bin/bash

#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --nodelist=g001
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=30-00:00:00
#SBATCH --job-name=hici
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate/hici_%j.log

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate film

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE | awk '{ print $1; exit }')

read PORT1 PORT2 < <(python -c "
import socket
s1 = socket.socket(); s1.bind(('', 0))
s2 = socket.socket(); s2.bind(('', 0))
print(s1.getsockname()[1], s2.getsockname()[1])
s1.close(); s2.close()
")
echo "PORT1=$PORT1"
echo "PORT2=$PORT2"

echo "Job started on $(hostname) at $(date)"

# === GPU 0 Jobs ===
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job1_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT1 \
  --master_addr=$MASTER_ADDR --master_port=$PORT1 \
  train.py --distributed --load-snapshot --config config_256_set_3 &

# === GPU 1 Jobs ===
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 torchrun --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv-id=job4_$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$PORT2 \
  --master_addr=$MASTER_ADDR --master_port=$PORT2 \
  train.py --distributed --load-snapshot --config config_256_set_4 &
wait

echo "Job finished at $(date)"