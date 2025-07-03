#!/bin/bash

sbatch job1.slurm config_512_set_1
sbatch job2.slurm config_512_set_2
sbatch job1.slurm config_512_set_3
sbatch job2.slurm config_512_set_4
sbatch job1.slurm config_512_set_5
sbatch job2.slurm config_512_set_6