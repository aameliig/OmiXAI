#!/bin/bash
#SBATCH --job-name=omixai_pfi
#SBATCH --output=logs/pfi_%j.out
#SBATCH --error=logs/pfi_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu

CONDA_ENV=omixai
PROJECT_DIR=$HOME/OmiXAI

conda activate $CONDA_ENV
export PYTHONPATH=$HOME/Sparse_vector:$PYTHONPATH

cd $PROJECT_DIR

python scripts/run_pfi_rf.py \
    --features results/feature_matrix_flat.npy \
    --labels   results/labels_flat.npy \
    --omixai   results/omixai_ranking.csv \
    --n_repeats 10 \
    --out_dir  results/
