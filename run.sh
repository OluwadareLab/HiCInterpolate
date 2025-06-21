#!/bin/bash

sbatch job1.slurm config_128_set_1 &
sbatch job2.slurm config_256_set_1 &
sbatch job1.slurm config_128_set_2 &
sbatch job2.slurm config_256_set_2 &
wait
sbatch job1.slurm config_128_set_3 &
sbatch job2.slurm config_256_set_3 &
sbatch job1.slurm config_128_set_4 &
sbatch job2.slurm config_256_set_4 &
wait
sbatch job1.slurm config_128_set_5 &
sbatch job2.slurm config_256_set_5 &
sbatch job1.slurm config_128_set_6 &
sbatch job2.slurm config_256_set_6 &
wait

