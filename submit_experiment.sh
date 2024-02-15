#!/bin/bash

#SBATCH --partition=bluemoon
#SBATCH --nodes=1
# #SBATCH --ncpu=100
#SBATCH --ntasks=100
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=experiment_test_1
#SBATCH --output=./outfiles/%x_%j.out

set -x 

cd /gpfs1/home/k/t/ktbielaw/projects/cilia-diff

source /gpfs1/home/k/t/ktbielaw/anaconda3/bin/activate ciliadiff

python3 run_exp.py --exp $1 --vacc
