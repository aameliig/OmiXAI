"""
Compare old ranking (train+test TPs) with new ranking (train-only TPs).

Answers Reviewer 2.1: validates that restricting interpretation to train TPs
does not materially change the feature ranking.

Usage:
    python3 scripts/compare_old_new_ranking.py \
        --old_dir  interpretation/CNN_0.88/ \
        --new_csv  results/omixai_ranking.csv \
        --features z_dna/hg38_features/sparse/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def load_old_ranking(old_dir: str, features_dir: str) -> pd.Series:
    """
    Reconstruct hybrid ranking from saved .pt / .npy attribution tensors
    (original results computed on train + test TPs).
    """
    old_dir = Path(old_dir)
    feature_names = [f[:-4] for f in os.listdir(features_dir) if f.endswith(".pkl")]

    scores = {}
    for f in sorted(old_dir.glob("*.pt")):
        t = torch.load(f, map_location="cpu").numpy()
        # tensors may include DNA channels — drop first 4 if size matches
        if len(t) == len(feature_names) + 4:
            t = t[4:]
        if len(t) == len(feature_names):
            scores[f.stem] = t

    for f in sorted(old_dir.glob("*.npy")):
        t = np.load(f)
        if len(t) == len(feature_names) + 4:
            t = t[4:]
        if len(t) == len(feature_names):
            scores[f.stem] = t

    if not scores:
        raise FileNotFoundError(f"No .pt/.npy attribution files found in {old_dir}")

    df = pd.DataFrame(scores, index=feature_names)
    pct = pd.DataFrame(index=df.index)
    for col in df.columns:
        mu = df[col].mean()
        pct[col] = (df[col] - mu) / mu * 100 if mu != 0 else 0.0

    pct["mean_deviation"] = pct.mean(axis=1)
    return pct["mean_deviation"].sort_values(ascending=False)


def main(args):
    print("Loading old ranking (train + test TPs)...")
    old_rank = load_old_ranking(args.old_dir, args.features)

    print("Loading new ranking (train TPs only)...")
    new_df   = pd.read_csv(args.new_csv, index_col=0)
    new_rank = new_df["mean_deviation"]

    # align on common features
    common = old_rank.index.intersection(new_rank.index)
    old_v  = old_rank.loc[common].values
    new_v  = new_rank.loc[common].values

    rho, pval = spearmanr(old_v, new_v)
    print(f"\nSpearman ρ (old vs new) = {rho:.4f}  (p = {pval:.2e})")
    print(f"Features compared: {len(common)}")

    old_order = old_rank.loc[common].sort_values(ascending=False).index
    new_order = new_rank.loc[common].sort_values(ascending=False).index

    print("\nOverlap between old and new top-k:")
    for k in (20, 50, 100, 300):
        shared = len(set(old_order[:k]) & set(new_order[:k]))
        print(f"  top-{k:>4}: {shared}/{k}  ({shared/k:.0%})")

    print("\nTop-20 features — old ranking:")
    print(old_order[:20].tolist())
    print("\nTop-20 features — new ranking (train-only):")
    print(new_order[:20].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_dir",  required=True,
                        help="directory with old .pt/.npy attribution tensors "
                             "(e.g. interpretation/CNN_0.88/)")
    parser.add_argument("--new_csv",  required=True,
                        help="new ranking CSV (results/omixai_ranking.csv)")
    parser.add_argument("--features", required=True,
                        help="path to hg38_features/sparse/ (to get feature names)")
    main(parser.parse_args())
