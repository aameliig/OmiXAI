#!/bin/bash
#SBATCH --job-name=rt_omixai
#SBATCH --output=logs/rtom_%A_%a.out
#SBATCH --error=logs/rtom_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-5

K_VALUES=(50 100 300 500 700 1946)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=$HOME/OmiXAI
DATA_DIR=$HOME/DNA/z_dna

module load Python/Google_Colab_GPU_2024
PYTHON=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
export PYTHONPATH=$HOME/DNA:$HOME/DNA/Sparse_vector:$PYTHONPATH
export OMIXAI_FEATURE_ORDER=$HOME/OmiXAI/feature_order.csv

mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

echo "OmiXAI retrain k=$K"
$PYTHON scripts/retrain_faithful.py \
    --data_dir "$DATA_DIR" \
    --ranking  results/omixai_ranking.csv \
    --k        $K \
    --n_epochs 10 \
    --out_dir  results/
