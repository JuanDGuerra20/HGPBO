#!/bin/bash
#SBATCH --mem=64G
#SBATCH --job-name=MultiProc_GPBO
#SBATCH --cpus-per-task=15

module load python/3.10
source $HOME/hgpbo_venv/bin/activate

python efficient_general_2d.py