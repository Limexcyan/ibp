#!/bin/bash
#SBATCH --job-name=split_mnist_run
#SBATCH --mem=4G
#SBATCH --cpus-per-tasks=2
#SBATCH --partition=student

# /$HOME/miniconda3/envs/hypnettorch/bin/python /$HOME/ibp/main.py
cd $HOME/ibp
echo "--- test run ---"
conda activate hypnettorch

echo "--- running main.py---"
python -u main.py
