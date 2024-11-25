#!/bin/bash
#SBATCH --mem=64G
#SBATCH --job-name=MultiProc_GPBO
#SBATCH --cpus-per-task=15
#SBATCH --output=logs/3D/%A_%a_multi_process_synthetic_datasets.out

module load python/3.10
source $HOME/hgpbo_venv/bin/activate
source .env

cd 3D_testing
python ucb_efficient_general.py