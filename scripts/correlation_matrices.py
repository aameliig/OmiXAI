"""
Generate correlation matrices between XAI methods at different top-k cutoffs.

Answers Reviewer 2.4: Figure 4 equivalent for k = 50, 100, 300, 500, 1000.

Usage:
    python3 scripts/correlation_matrices.py \
        --scores_dir  results/ \
        --old_dir     interpretation/GraphZSAGEConv/ \
        --out_dir     results/supplementary/

    Pass --scores_dir if you have omixai_gnn_scores.npy from the new run.
    Pass --old_dir to use the original .pt files from the repo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr


def load_scores_from_npy(scores_path: str) -> dict[str, np.ndarray]:
    raw = np.load(scores_path, allow_pickle=True).item()
    return {k: np.array(v) for k, v in raw.items()}


def load_scores_from_pt(pt_dir: str, n_features: int) -> dict[str, np.ndarray]:
    scores = {}
    for f in sorted(Path(pt_dir).glob("*.pt")):
        t = torch.load(f, map_location="cpu").numpy()
        if len(t) == n_features + 4:
            t = t[4:]
        if len(t) == n_features:
            # clean up filename to get method name
            name = f.stem.replace("mean_GraphZSAGEConv_v5_lin_", "").replace("mean_", "")
            scores[name] = t
    return scores


def spearman_matrix(scores: dict[str, np.ndarray], top_k: int) -> pd.DataFrame:
    """
    Compute Spearman correlation between methods restricted to top-k features
    (ranked by each method's own scores).
    """
    methods = list(scores.keys())
    n = len(methods)
    mat = np.eye(n)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:
                continue
            idx1 = np.argsort(scores[m1])[::-1][:top_k]
            idx2 = np.argsort(scores[m2])[::-1][:top_k]
            # union of top-k indices from both methods
            union = np.union1d(idx1, idx2)
            rho, _ = spearmanr(scores[m1][union], scores[m2][union])
            mat[i, j] = mat[j, i] = round(rho, 2)

    return pd.DataFrame(mat, index=methods, columns=methods)


def plot_matrix(corr: pd.DataFrame, k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.zeros_like(corr.values, dtype=bool)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title(f"Spearman correlation — top {k} features", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.scores_npy:
        scores = load_scores_from_npy(args.scores_npy)
    elif args.old_dir:
        n_features = int(args.n_features)
        scores = load_scores_from_pt(args.old_dir, n_features)
    else:
        raise ValueError("Provide either --scores_npy or --old_dir")

    print(f"Methods loaded: {list(scores.keys())}")

    k_values = [50, 100, 300, 500, 1000]
    for k in k_values:
        print(f"Computing correlation matrix for top-{k}...")
        corr = spearman_matrix(scores, k)
        plot_matrix(corr, k, out / f"corr_top{k}.png")
        corr.to_csv(out / f"corr_top{k}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_npy", default=None,
                        help="results/omixai_gnn_scores.npy (new run)")
    parser.add_argument("--old_dir",    default=None,
                        help="interpretation/GraphZSAGEConv/ (original .pt files)")
    parser.add_argument("--n_features", default=1946,
                        help="number of omics features (default 1946)")
    parser.add_argument("--out_dir",    default="results/supplementary/")
    main(parser.parse_args())
