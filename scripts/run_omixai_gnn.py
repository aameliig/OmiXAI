"""
Run OmiXAI interpretation on a trained GraphMZC model.

Data layout expected on the cluster (mirrors vladislareon/z_dna repo):
  $DATA_DIR/
    hg38_dna/          — per-chromosome DNA chunks (joblib pkl, loaded by chrom_reader)
    hg38_zdna/sparse/  — Z-DNA label SparseVectors  (ZDNA_cousine.pkl / ZDNA_shin.pkl)
    hg38_features/sparse/  — omics feature SparseVectors (one pkl per feature)

Produces:
  $OUT_DIR/omixai_gnn_scores.npy   — raw attribution dict, one array per method
  $OUT_DIR/omixai_ranking.csv      — hybrid-ranked feature list (discovery on train TPs only)

Note on train/test separation:
  Feature importance is computed exclusively from True Positives in the
  TRAIN split. The test split is kept sealed and used only for the final
  model performance evaluation after retraining with selected features.
  This ensures no information from the test set influences feature selection.

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
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gnn import GraphMZC
from data.graph_dataset import GraphGenomicDataset, get_train_test_split_graph
from omixai import OmiXAI

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]


# ------------------------------------------------------------------
# Data loading  (mirrors the notebook pattern exactly)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------

def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    np.random.seed(10)
    DNA, ZDNA, DNA_features, feature_names = load_all_data(args.data_dir)
    n_features = len(feature_names)
    print(f"Omics features: {n_features}")

    train_dataset, test_dataset = get_train_test_split_graph(
        args.width, CHROMS, feature_names, DNA, DNA_features, ZDNA
    )

    # batch_size=1 required for node-level GNN inference
    loader_params = dict(batch_size=1, num_workers=4, shuffle=False)
    loader_train = DataLoader(train_dataset, **loader_params)
    loader_test  = DataLoader(test_dataset,  **loader_params)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = GraphMZC(n_features=n_features)
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model = model.to(device).eval()
    print(f"Model loaded: {args.model}")

    # ------------------------------------------------------------------
    # 3. OmiXAI — feature importance from TRAIN TPs only
    #    (test loader kept sealed for downstream validation)
    # ------------------------------------------------------------------
    pipeline = OmiXAI(model=model, model_type="gnn", n_features=n_features, device=device)

    methods = ["IG", "IxG", "GB", "Deconv", "Saliency", "GNNExplainer"]
    scores = pipeline.interpret(loader_train, methods=methods, width=args.width)

    np.save(out / "omixai_gnn_scores.npy", scores)
    print(f"Saved raw scores → {out}/omixai_gnn_scores.npy")

    # ------------------------------------------------------------------
    # 4. Hybrid ranking
    # ------------------------------------------------------------------
    ranking = pipeline.rank_features(feature_names=feature_names)
    ranking.to_csv(out / "omixai_ranking.csv")
    print(f"Saved ranking → {out}/omixai_ranking.csv")
    print("\nTop 20 features:")
    print(ranking.head(20).to_string())

    # ------------------------------------------------------------------
    # 5. Save flat feature matrix for RF-PFI (mean per interval per feature)
    # ------------------------------------------------------------------
    print("\nBuilding flat feature matrix for RF-PFI...")
    _save_flat_matrix(test_dataset, feature_names, n_features, args.width, out)
    print(f"Saved flat matrix → {out}/feature_matrix_flat.npy  /  labels_flat.npy")


def _save_flat_matrix(dataset, feature_names, n_features, width, out: Path):
    """
    Collapse per-nucleotide features to per-interval by taking the mean.
    Produces (n_intervals, n_features) and (n_intervals,) arrays for RF.
    """
    X_list, y_list = [], []
    for i in tqdm(range(len(dataset)), desc="building matrix"):
        item = dataset.get(i)
        # item.x shape: (1, width, n_features+4); skip first 4 DNA channels
        x_mean = item.x.squeeze(0)[:, 4:].numpy().mean(axis=0)  # (n_features,)
        y_label = int(item.y.squeeze(0).numpy().any())           # 1 if any Z-DNA
        X_list.append(x_mean)
        y_list.append(y_label)

    np.save(out / "feature_matrix_flat.npy", np.array(X_list, dtype=np.float32))
    np.save(out / "labels_flat.npy",         np.array(y_list, dtype=np.int32))


# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True,
                        help="path to model weights (.pt)")
    parser.add_argument("--data_dir", required=True,
                        help="root data directory (contains hg38_dna/, hg38_zdna/, hg38_features/)")
    parser.add_argument("--width",    type=int, default=100)
    parser.add_argument("--out_dir",  default="results/")
    main(parser.parse_args())
