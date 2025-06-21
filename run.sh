#!/bin/bash

torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py train.py --distributed --config config_64_set_1
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py train.py --distributed --config config_64_set_2
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py train.py --distributed --config config_64_set_3
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py train.py --distributed --config config_64_set_4
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py train.py --distributed --config config_64_set_5
torchrun --nproc-per-node 1 --nnodes 1 --node_rank 0 train.py train.py --distributed --config config_64_set_6
