#!/bin/bash
#SBATCH --job-name=subset_imagenet_ibp_run_emb_128
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -p rtx2080
#SBATCH --qos=big


cd repo/ibp
echo "--- test run ---"
#conda activate hypnettorch
source activate hypnettorch-3
echo "--- running main.py---"
python -u main.py
