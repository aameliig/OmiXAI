"""
Run OmiXAI interpretation on a trained GraphMZC model.

Data layout expected (mirrors vladislareon/z_dna repo):
  $DATA_DIR/
    hg38_dna/              — per-chromosome DNA chunks (joblib pkl)
    hg38_zdna/sparse/      — Z-DNA label SparseVectors (ZDNA_cousine.pkl)
    hg38_features/sparse/  — omics feature SparseVectors (one pkl per feature)

Produces:
  $OUT_DIR/omixai_gnn_scores.npy   — raw attribution dict, one array per method
  $OUT_DIR/omixai_ranking.csv      — hybrid-ranked feature list
  $OUT_DIR/feature_matrix_flat.npy — per-interval mean features (for RF-PFI)
  $OUT_DIR/labels_flat.npy         — per-interval binary labels (for RF-PFI)

Note on train/test separation:
  Feature importance is computed exclusively from True Positives in the TRAIN
  split. The test split is kept sealed for downstream validation after retraining
  with selected features.

Usage:
  python scripts/run_omixai_gnn.py \\
      --model    ~/weights/graphmzc.pt \\
      --data_dir ~/DNA \\
      --width    100 \\
      --out_dir  results/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from joblib import load
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from omixai import OmiXAI, GraphMZC, get_train_test_split_graph

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]


def load_chromosome(chrom: str, dna_dir: str) -> str:
    files = sorted(f for f in os.listdir(dna_dir) if f"{chrom}_" in f)
    return "".join(load(os.path.join(dna_dir, f)) for f in files)


def load_all_data(data_dir: str):
    dna_dir      = os.path.join(data_dir, "hg38_dna")
    zdna_path    = os.path.join(data_dir, "hg38_zdna", "sparse", "ZDNA_cousine.pkl")
    features_dir = os.path.join(data_dir, "hg38_features", "sparse")

    print("Loading DNA sequences...")
    DNA = {chrom: load_chromosome(chrom, dna_dir) for chrom in tqdm(CHROMS)}

    print("Loading Z-DNA labels...")
    ZDNA = load(zdna_path)

    print("Loading omics features...")
    feature_names = [f[:-4] for f in os.listdir(features_dir) if f.endswith(".pkl")]
    DNA_features  = {feat: load(os.path.join(features_dir, f"{feat}.pkl"))
                     for feat in tqdm(feature_names)}

    return DNA, ZDNA, DNA_features, feature_names


def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(10)
    DNA, ZDNA, DNA_features, feature_names = load_all_data(args.data_dir)
    n_features = len(feature_names)
    print(f"Omics features: {n_features}")

    train_dataset, test_dataset = get_train_test_split_graph(
        args.width, CHROMS, feature_names, DNA, DNA_features, ZDNA
    )

    loader_params = dict(batch_size=1, num_workers=4, shuffle=False)
    loader_train  = DataLoader(train_dataset, **loader_params)

    model = GraphMZC(n_features=n_features)
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model = model.to(device).eval()
    print(f"Model loaded: {args.model}")

    # Feature importance from TRAIN TPs only (test set kept sealed)
    pipeline = OmiXAI(model=model, model_type="gnn", n_features=n_features, device=device)
    scores   = pipeline.interpret(loader_train, width=args.width)

    np.save(out / "omixai_gnn_scores.npy", scores)

    ranking = pipeline.rank_features(feature_names=feature_names)
    ranking.to_csv(out / "omixai_ranking.csv")
    print(f"\nTop 20 features:\n{ranking.head(20).to_string()}")

    # Flat feature matrix for RF-PFI (mean per interval)
    print("\nBuilding flat feature matrix for RF-PFI...")
    X_list, y_list = [], []
    for i in tqdm(range(len(test_dataset))):
        item = test_dataset.get(i)
        X_list.append(item.x.squeeze(0)[:, 4:].numpy().mean(axis=0))
        y_list.append(int(item.y.squeeze(0).numpy().any()))
    np.save(out / "feature_matrix_flat.npy", np.array(X_list, dtype=np.float32))
    np.save(out / "labels_flat.npy",         np.array(y_list, dtype=np.int32))
    print(f"Saved all results to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--width",    type=int, default=100)
    parser.add_argument("--out_dir",  default="results/")
    main(parser.parse_args())
