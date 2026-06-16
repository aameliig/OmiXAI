#!/bin/bash
#SBATCH --job-name=omixai_retrain
#SBATCH --output=logs/retrain_%A_%a.out
#SBATCH --error=logs/retrain_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-5

K_VALUES=(50 100 300 500 704 1946)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=$HOME/OmiXAI
DATA_DIR=$HOME/DNA/z_dna

module load Python/Google_Colab_GPU_2024
PYTHON=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
export PYTHONPATH=$HOME/DNA:$HOME/DNA/Sparse_vector:$PYTHONPATH
export OMIXAI_FEATURE_ORDER=$HOME/OmiXAI/feature_order.csv

mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

echo "Running k=$K (array task $SLURM_ARRAY_TASK_ID)"

$PYTHON scripts/retrain_topk.py \
    --ranking       results/omixai_ranking.csv \
    --pfi_scores    results/pfi_rf_scores.npy \
    --feature_names results/feature_names.json \
    --data_dir      "$DATA_DIR" \
    --width         100 \
    --n_epochs      15 \
    --k             $K \
    --out_dir       results/
