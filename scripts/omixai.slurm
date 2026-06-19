#!/bin/bash
#SBATCH --job-name=omixai_gnn
#SBATCH --output=logs/omixai_%j.out
#SBATCH --error=logs/omixai_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1

PROJECT_DIR=$HOME/OmiXAI
WEIGHTS=$HOME/OmiXAI/weights/Cousine_GraphZSAGEConv_v5_lin_F1=77.75_epoch=17.pt
DATA_DIR=$HOME/DNA/z_dna

module load Python/Google_Colab_GPU_2024
PYTHON=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
export PYTHONPATH=$HOME/DNA:$HOME/DNA/Sparse_vector:$PYTHONPATH
export OMIXAI_FEATURE_ORDER=$HOME/OmiXAI/feature_order.csv

mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/results
cd $PROJECT_DIR

# METHOD=hybrid (gradient ensemble, default) or METHOD=pfi (permutation importance)
METHOD=${METHOD:-hybrid}

$PYTHON scripts/run_interpret.py \
    --model    "$WEIGHTS" \
    --data_dir "$DATA_DIR" \
    --method   "$METHOD" \
    --width    100 \
    --out_dir  results/
