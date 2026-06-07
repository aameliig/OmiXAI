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

MODULE_CUDA=cuda/11.8
CONDA_ENV=omixai
PROJECT_DIR=$HOME/OmiXAI

module load $MODULE_CUDA
conda activate $CONDA_ENV

cd $PROJECT_DIR

# Run AFTER omixai.slurm — needs results/omixai_ranking.csv
python scripts/run_pfi_rf.py \
    --features results/feature_matrix_flat.npy \
    --labels   results/labels_flat.npy \
    --omixai   results/omixai_ranking.csv \
    --n_repeats 10 \
    --out_dir  results/
