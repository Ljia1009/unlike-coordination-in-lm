#!/bin/bash

#SBATCH --job-name=train_and
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=jiamuluo@uw.edu

#SBATCH --account=stf
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=03-12:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=train_and.out # where STDOUT goes
#SBATCH --error=train_and.error # where STDERR goes

# Your programs to run.
eval "$(/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/bin/conda shell.bash hook)"
conda activate coordination
python3 -m models.train_only_and > train_and_log
conda deactivate