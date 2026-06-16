#!/bin/bash
#SBATCH --job-name=eval_orig
#SBATCH --output=logs/evalorig_%j.out
#SBATCH --error=logs/evalorig_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1

PROJECT_DIR=$HOME/OmiXAI
DATA_DIR=$HOME/DNA/z_dna
WEIGHTS=$HOME/OmiXAI/weights/Cousine_GraphZSAGEConv_v5_lin_F1=77.75_epoch=17.pt

module load Python/Google_Colab_GPU_2024
PYTHON=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
export PYTHONPATH=$HOME/DNA:$HOME/DNA/Sparse_vector:$PYTHONPATH
export OMIXAI_FEATURE_ORDER=$HOME/OmiXAI/feature_order.csv

mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

$PYTHON scripts/eval_orig.py "$WEIGHTS" "$DATA_DIR" 100
