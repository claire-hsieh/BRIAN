#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH --mem 150G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=clhsieh@ucdavis.edu
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu-a100-h
#SBATCH --account=quonbiogrp
#SBATCH --output=/home/gluetown/output/8_26/%A_%a.out
#SBATCH --error=/home/gluetown/error/dir/slurm-err%j.txt


# #SBATCH --array=0-8%3

echo "Running on $(hostname)"

# source ~/.bashrc
# conda activate esmfold

source /home/gluetown/miniconda3/etc/profile.d/conda.sh
conda activate /home/gluetown/miniconda3/envs/esmfold

# declare -a start_indices=(30 40 50 60 70 80 90 100) # 8 jobs
# declare -a end_indices=(40 50 60 70 80 90 100 101)

# start=${start_indices[$SLURM_ARRAY_TASK_ID]}
# end=${end_indices[$SLURM_ARRAY_TASK_ID]}

# echo "$(start) $(end)"

# python3 get_esm.py $start $end
python3 get_esm.py 4 5

