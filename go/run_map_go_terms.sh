#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem 100G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mail-user=clhsieh@ucdavis.edu
#SBATCH --mail-type=FAIL
#SBATCH --partition=med
#SBATCH --account=quonbiogrp
#SBATCH --output=/home/gluetown/output/map_go_terms%A_%a.out
#SBATCH --error=/home/gluetown/error/dir/map_go_terms%j.txt
echo "Running on $(hostname)"

source /home/gluetown/miniconda3/etc/profile.d/conda.sh
conda activate /home/gluetown/miniconda3/envs/esm2

srun python3 map_go_terms.py