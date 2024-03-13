#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --job-name=experiment_test_1
#SBATCH --output=./outfiles/%x_%j.out

set -x 

cd /gpfs1/home/k/t/ktbielaw/projects/cilia-diff

source /gpfs1/home/k/t/ktbielaw/anaconda3/bin/activate ciliadiff

python3 run_exp.py --exp $1 --vacc
