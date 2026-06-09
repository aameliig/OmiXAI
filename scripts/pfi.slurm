#!/bin/bash
#SBATCH --job-name=omixai_pfi
#SBATCH --output=logs/pfi_%j.out
#SBATCH --error=logs/pfi_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1

CONDA_ENV=omixai
PROJECT_DIR=$HOME/OmiXAI

conda activate $CONDA_ENV
export PYTHONPATH=$HOME/DNA/Sparse_vector:$PYTHONPATH

mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

python scripts/run_pfi_rf.py \
    --features      results/feature_matrix_flat.npy \
    --labels        results/labels_flat.npy \
    --omixai        results/omixai_ranking.csv \
    --feature_names results/feature_names.json \
    --n_repeats     10 \
    --out_dir       results/
