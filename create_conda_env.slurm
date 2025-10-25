#!/bin/bash

#SBATCH --partition=compute-long
#SBATCH --nodes=1
#SBATCH --nodelist=c010
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30-00:00:00
#SBATCH --job-name=CHIE
#SBATCH --output=/home/hchowdhu/Documents/HiCInterpolate/job_create_hicinterpolate_env_%j.log
#SBATCH --error=/home/hchowdhu/Documents/HiCInterpolate/job_create_hicinterpolate_env_%j.err

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh


echo "Creating conda environment from environment.yml ..."
conda env create -f environment.yml

echo "Environment list:"
conda env list

# ======= Optional: Export to check packages =======
conda activate $(head -1 environment.yml | cut -d' ' -f2)
conda list > installed_packages.txt

echo "Conda environment creation complete."