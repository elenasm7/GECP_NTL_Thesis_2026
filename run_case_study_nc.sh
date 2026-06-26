#!/bin/bash
#SBATCH --job-name=case_calculation
#SBATCH --partition=rome
#SBATCH --time=00-1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate NTL_paper

python Analysing_case_study_nc.py 
