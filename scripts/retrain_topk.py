"""
Retrain GraphMZC on top-k features selected by OmiXAI hybrid ranking.

Reproduces Tables 2 and 3 from the paper: F1 and AUC at different k values.
Also runs the same experiment using PFI-RF ranking for comparison (R1).

Usage:
    python3 scripts/retrain_topk.py \
        --ranking      results/omixai_ranking.csv \
        --pfi_scores   results/pfi_rf_scores.npy \
        --data_dir     ~/DNA \
        --width        100 \
        --n_epochs     15 \
        --out_dir      results/

Outputs:
    results/retrain_table.csv   — F1 / AUC at each k for OmiXAI and PFI rankings
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omixai import GraphMZC, get_train_test_split_graph
from omixai.training import train_gnn, evaluate_gnn

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]
K_VALUES = [50, 100, 300, 500, 704, 1946]   # mirrors paper tables


def load_data(data_dir: str):
    dna_dir      = os.path.join(data_dir, "hg38_dna")
    zdna_path    = os.path.join(data_dir, "hg38_zdna", "sparse", "ZDNA_cousine.pkl")
    features_dir = os.path.join(data_dir, "hg38_features", "sparse")

    DNA   = {}
    files = sorted(os.listdir(dna_dir))
    for chrom in tqdm(CHROMS, desc="DNA"):
        chrom_files = sorted(f for f in files if f"{chrom}_" in f)
        DNA[chrom]  = "".join(load(os.path.join(dna_dir, f)) for f in chrom_files)

    ZDNA         = load(zdna_path)
    feature_names = [f[:-4] for f in os.listdir(features_dir) if f.endswith(".pkl")]
    DNA_features  = {feat: load(os.path.join(features_dir, f"{feat}.pkl"))
                     for feat in tqdm(feature_names, desc="omics")}
    return DNA, ZDNA, DNA_features, feature_names


def train_and_eval(feature_subset: list[str], DNA, ZDNA, DNA_features,
                   width: int, n_epochs: int, device) -> dict:
    np.random.seed(10)
    train_ds, test_ds = get_train_test_split_graph(
        width, CHROMS, feature_subset, DNA, DNA_features, ZDNA
    )
    loader_params = dict(batch_size=1, num_workers=4, shuffle=False)
    train_loader  = DataLoader(train_ds, **loader_params, shuffle=True)
    test_loader   = DataLoader(test_ds,  **loader_params)

    model = GraphMZC(n_features=len(feature_subset)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_gnn(model, opt, n_epochs, train_loader, test_loader, width)
    metrics = evaluate_gnn(model, test_loader, width)
    return metrics


def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    DNA, ZDNA, DNA_features, all_features = load_data(args.data_dir)

    print("Loading rankings...")
    omixai_rank = pd.read_csv(args.ranking, index_col=0)["mean_deviation"]
    omixai_order = omixai_rank.sort_values(ascending=False).index.tolist()

    pfi_order = None
    if args.pfi_scores:
        pfi_scores = np.load(args.pfi_scores)
        # pfi_scores[i] corresponds to the i-th column of feature_matrix_flat.npy,
        # which was built in the order saved to feature_names.json — NOT the order
        # os.listdir returns here. Mapping through all_features (os.listdir) would
        # assign each PFI score to the wrong feature. Use the canonical order.
        if args.feature_names:
            pfi_feat_names = json.loads(Path(args.feature_names).read_text())
        else:
            print("WARNING: --feature_names not given; falling back to os.listdir "
                  "order for PFI mapping. This is only correct if os.listdir here "
                  "matches the order used to build the PFI matrix.")
            pfi_feat_names = all_features
        if len(pfi_feat_names) != len(pfi_scores):
            raise ValueError(
                f"feature_names ({len(pfi_feat_names)}) != pfi_scores "
                f"({len(pfi_scores)}) — mismatched runs."
            )
        pfi_order = [pfi_feat_names[i] for i in np.argsort(pfi_scores)[::-1]]

    # --k selects a single k (for parallel job arrays); default runs all
    run_ks = [args.k] if args.k else K_VALUES

    rows = []
    for k in run_ks:
        print(f"\n=== k = {k} ===")

        # OmiXAI top-k
        subset_omixai = [f for f in omixai_order[:k] if f in DNA_features]
        print(f"OmiXAI: retraining with {len(subset_omixai)} features...")
        m = train_and_eval(subset_omixai, DNA, ZDNA, DNA_features, args.width, args.n_epochs, device)
        row = {"k": k, "method": "OmiXAI",
               "f1": m["f1"], "auc": m["auc"], "precision": m["prec"], "recall": m["rec"]}
        rows.append(row)
        print(f"  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

        # PFI top-k (if available)
        if pfi_order:
            subset_pfi = [f for f in pfi_order[:k] if f in DNA_features]
            print(f"PFI-RF: retraining with {len(subset_pfi)} features...")
            m2 = train_and_eval(subset_pfi, DNA, ZDNA, DNA_features, args.width, args.n_epochs, device)
            row2 = {"k": k, "method": "PFI-RF",
                    "f1": m2["f1"], "auc": m2["auc"], "precision": m2["prec"], "recall": m2["rec"]}
            rows.append(row2)
            print(f"  F1={m2['f1']:.4f}  AUC={m2['auc']:.4f}")

    table = pd.DataFrame(rows)
    # per-k output so job arrays don't overwrite each other
    suffix = f"_k{args.k}" if args.k else ""
    table.to_csv(out / f"retrain_table{suffix}.csv", index=False)
    print(f"\nSaved → {out}/retrain_table{suffix}.csv")

    if not args.k:
        # full table summary + Wilcoxon only when all k are present
        print(table.pivot(index="k", columns="method", values=["f1", "auc"]).to_string())
        if pfi_order:
            omixai_f1 = table[table.method == "OmiXAI"]["f1"].values
            pfi_f1    = table[table.method == "PFI-RF"]["f1"].values
            if len(omixai_f1) > 1:
                stat, p = wilcoxon(omixai_f1, pfi_f1, alternative="greater")
                print(f"\nWilcoxon (OmiXAI F1 > PFI F1): stat={stat:.3f}  p={p:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking",    required=True,  help="results/omixai_ranking.csv")
    parser.add_argument("--pfi_scores", default=None,   help="results/pfi_rf_scores.npy")
    parser.add_argument("--feature_names", default=None,
                        help="results/feature_names.json — canonical order used to "
                             "build the PFI matrix (needed to map pfi_scores to names)")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--width",      type=int, default=100)
    parser.add_argument("--n_epochs",   type=int, default=15)
    parser.add_argument("--out_dir",    default="results/")
    parser.add_argument("--k",          type=int, default=None,
                        help="run a single k value (for SLURM job arrays); omit to run all")
    main(parser.parse_args())
