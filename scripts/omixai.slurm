#!/bin/bash
#SBATCH --job-name=omixai_gnn
#SBATCH --output=logs/omixai_%j.out
#SBATCH --error=logs/omixai_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1

module load Python/Google_Colab_GPU_2024

PYTHON=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
PROJECT_DIR=$HOME/OmiXAI
WEIGHTS=$HOME/weights/graphmzc.pt
DATA_DIR=$HOME/DNA/z_dna

export PYTHONPATH=$HOME/DNA:$PYTHONPATH

mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/results
cd $PROJECT_DIR

$PYTHON scripts/run_omixai_gnn.py \
    --model    $WEIGHTS \
    --data_dir $DATA_DIR \
    --width    100 \
    --out_dir  results/
