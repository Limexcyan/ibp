#!/bin/bash
#SBATCH --job-name=subset_imagenet
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -p rtx2080
#SBATCH --qos=big

cd repo/ibp
echo "--- test run ---"
source activate hypnettorch

echo "--- running main.py---"
python -u main.py