#!/bin/bash
#SBATCH --job-name=omixai_gnn
#SBATCH --output=logs/omixai_%j.out
#SBATCH --error=logs/omixai_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

CONDA_ENV=omixai
PROJECT_DIR=$HOME/OmiXAI
WEIGHTS=$HOME/weights/graphmzc.pt
DATA_DIR=$HOME/DNA

conda activate $CONDA_ENV

# SparseVector: not a pip package, add to Python path
export PYTHONPATH=$HOME/Sparse_vector:$PYTHONPATH

mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/results

cd $PROJECT_DIR

python scripts/run_omixai_gnn.py \
    --model    $WEIGHTS \
    --data_dir $DATA_DIR \
    --width    100 \
    --out_dir  results/
