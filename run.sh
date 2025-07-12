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
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 hicinterpolate.py --distributed --test --config config_128_set_1
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
