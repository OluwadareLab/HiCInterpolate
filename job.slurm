#!/bin/bash

#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --nodelist=g002
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=30-00:00:00
#SBATCH --job-name=HI
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate/job_hicinterpolate_%j.log

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate film

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_ADDR=$(getent ahostsv4 $MASTER_NODE | awk '{ print $1; exit }')
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

CONFIG_FILE=$1

echo "Job ($CONFIG_FILE) started on $(hostname) at $(date)"

python3 hicinterpolate.py --train --test --config config_256_set_1 &
python3 hicinterpolate.py --train --test --config config_256_set_2 &
wait

echo "Job finished at $(date)"