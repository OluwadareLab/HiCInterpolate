#!/bin/bash

# sbatch job1.slurm config_64_set_1
# sbatch job2.slurm config_64_set_2
# sbatch job1.slurm config_64_set_3
# sbatch job2.slurm config_64_set_4
# sbatch job1.slurm config_64_set_5
# sbatch job2.slurm config_64_set_6

# sbatch job1.slurm config_128_set_1
# sbatch job2.slurm config_128_set_2
# sbatch job1.slurm config_128_set_3
# sbatch job2.slurm config_128_set_4
# sbatch job1.slurm config_128_set_5
# sbatch job2.slurm config_128_set_6

# sbatch job1.slurm config_256_set_1
# sbatch job2.slurm config_256_set_2
# sbatch job1.slurm config_256_set_3
# sbatch job2.slurm config_256_set_4
# sbatch job1.slurm config_256_set_5
# sbatch job2.slurm config_256_set_6

# sbatch job1.slurm config_512_set_1
# sbatch job2.slurm config_512_set_2
# sbatch job1.slurm config_512_set_3
# sbatch job2.slurm config_512_set_4
# sbatch job1.slurm config_512_set_5
# sbatch job2.slurm config_512_set_6


# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_64_set_1
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_64_set_2
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_64_set_3
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_64_set_4
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_64_set_5
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_64_set_6
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --train --test --config config_64_set_1_kr_w_rand_AdamW_cosin
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --load-snapshot --train --test --config config_64_set_1_kr_w_rand
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_128_set_2
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_128_set_3
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_128_set_4
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_128_set_5
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_128_set_6
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_256_set_1
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_256_set_2
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_256_set_3
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_256_set_4
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_256_set_5
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_256_set_6
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_512_set_1
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_512_set_2
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_512_set_3
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_512_set_4
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_512_set_5
# torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_512_set_6


torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --train --test --config config_256_set_1_kr_log_clip_norm_AdamW_cosin
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --train --test --config config_256_set_1_kr_log_clip_norm_diag_AdamW_cosin

/mmfs1/home/hchowdhu/data/hicinterpolate
/mmfs1/home/hchowdhu/data/triplets/kr_log_clip_norm_diag
sbatch job1.slurm config_256_set_1_kr_log_clip_norm_diag_slurm

python3 hicinterpolate.py --train --test --config config_256_set_1_kr_log_clip_norm_diag

python3 hicinterpolate.py --train --test --config config_256_set_1_kr_log_clip_norm_diag_test