#!/bin/bash
#SBATCH --job-name=omixai_retrain
#SBATCH --output=logs/retrain_%A_%a.out
#SBATCH --error=logs/retrain_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-5

K_VALUES=(50 100 300 500 704 1946)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

CONDA_ENV=omixai
PROJECT_DIR=$HOME/OmiXAI
DATA_DIR=$HOME/DNA

conda activate $CONDA_ENV
export PYTHONPATH=$HOME/DNA/Sparse_vector:$PYTHONPATH

mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

echo "Running k=$K (array task $SLURM_ARRAY_TASK_ID)"

python scripts/retrain_topk.py \
    --ranking       results/omixai_ranking.csv \
    --pfi_scores    results/pfi_rf_scores.npy \
    --feature_names results/feature_names.json \
    --data_dir      $DATA_DIR \
    --width         100 \
    --n_epochs      15 \
    --k             $K \
    --out_dir       results/
