#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --job-name=ciliadiff_exp_%j
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=03:00:00

set -x

# echo "$1" "$2"
module load singularity

# singularity exec --nv ~/taichi-vacc-x11_latest.sif --bind . python3 ~/projects/cilia-diff/run_trial.py --file $1

singularity exec --nv ~/taichi-vacc-x11_latest.sif python3 ~/projects/cilia-diff/run_trial.py --file $1
