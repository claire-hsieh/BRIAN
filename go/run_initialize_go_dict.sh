#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem 100G
#SBATCH --gres=gpu:0
#SBATCH --mail-user=clhsieh@ucdavis.edu
#SBATCH --mail-type=FAIL
#SBATCH --partition=med
#SBATCH --account=quonbiogrp
#SBATCH --output=/home/gluetown/output/initialize_go_dict%A_%a.out
#SBATCH --error=/home/gluetown/error/dir/initialize_go_dict%j.txt

echo "Running on $(hostname)"

source /home/gluetown/miniconda3/etc/profile.d/conda.sh
conda activate /home/gluetown/miniconda3/envs/esm2

srun python3 initialize_go_dict.py
