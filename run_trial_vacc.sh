#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=ciliadiff_exp_%j
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=00:10:00

set -x

# echo "$1" "$2"
conda activate ciliadiff
python3 run_exp.py experiment_file_example.txt exp_example

module load singularity

singularity exec --nv --bind ~/projects/cilia-diff:/home/user/cilia-diff cuda_12.3.1-runtime-ubuntu20.04.sif /bin/bash -c "pip3 install taichi && python3 /home/user/cilia-diff/run_trial.py $1"