#!/bin/bash
#SBATCH --job-name=retrain_topk
#SBATCH --output=logs/retrain_%j.out
#SBATCH --error=logs/retrain_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rocky
#SBATCH --gres=gpu:v100:1

PROJECT_DIR=$HOME/OmiXAI
DATA_DIR=$HOME/DNA/z_dna

module load Python/Google_Colab_GPU_2024
PYTHON=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
export PYTHONPATH=$HOME/DNA:$HOME/DNA/Sparse_vector:$PYTHONPATH
export OMIXAI_FEATURE_ORDER=$HOME/OmiXAI/feature_order.csv

mkdir -p $PROJECT_DIR/logs $PROJECT_DIR/results
cd $PROJECT_DIR

# ARM=omixai (default) or ARM=pfi. K empty = full sweep [50,100,300,500,700,all].
ARM=${ARM:-omixai}
K_ARG=${K:+--k $K}

if [ "$ARM" = "pfi" ]; then
    $PYTHON scripts/run_retrain.py --data_dir "$DATA_DIR" $K_ARG \
        --pfi_scores results/pfi_dl/pfi_dl_scores.npy \
        --feature_names results/feature_names.json
else
    $PYTHON scripts/run_retrain.py --data_dir "$DATA_DIR" $K_ARG \
        --ranking results/omixai_ranking.csv
fi
