#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH --mem 100G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mail-user=clhsieh@ucdavis.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=quonbiogrp
#SBATCH --output=/home/gluetown/output/deepgo%A_%a.out
#SBATCH --error=/home/gluetown/error/dir/deepgo%j.txt
#SBATCH --partition=med
echo "Running on $(hostname)"

source /home/gluetown/miniconda3/etc/profile.d/conda.sh
conda activate /home/gluetown/miniconda3/envs/esm2

srun python3 /home/gluetown/brain/scripts/go/deepgo_predictions.py
