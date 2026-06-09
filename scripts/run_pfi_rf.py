"""
PFI via Random Forest — fast alternative to DL-based permutation importance.

Why RF instead of DL:
  Running PFI directly on GraphMZC would require ~1950 features × 5 repeats
  × full dataset inference, which takes several days even on GPU. Using a
  Random Forest trained on the same data gives a well-established reference
  ranking that is standard in bioinformatics feature importance comparisons.

Workflow:
  1. Load preprocessed omics feature matrix and Z-DNA labels.
  2. Train RandomForestClassifier (fast, parallelised).
  3. Extract built-in impurity-based importance + run sklearn permutation_importance.
  4. Compare with OmiXAI hybrid ranking (saved in results/).

Usage:
  python scripts/run_pfi_rf.py \
      --features  path/to/feature_matrix.npy \
      --labels    path/to/labels.npy \
      --omixai    results/omixai_ranking.csv \
      --out_dir   results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------

def load_data(features_path: str, labels_path: str):
    """
    Load flat feature matrix and labels.

    Expected shapes:
      X : (n_intervals, n_features)   — mean omics value per interval per feature
      y : (n_intervals,)              — binary label (1 = Z-DNA positive)
    """
    X = np.load(features_path)
    y = np.load(labels_path)
    return X, y


def train_rf(X_train, y_train, n_jobs: int = -1) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def run_pfi(rf, X_test, y_test, n_repeats: int = 10, n_jobs: int = -1) -> np.ndarray:
    """
    sklearn permutation_importance on test set.
    Returns mean importance scores of shape (n_features,).
    """
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=n_repeats,
        scoring="f1",
        n_jobs=n_jobs,
        random_state=42,
    )
    return result.importances_mean


def compare_with_omixai(
    pfi_scores: np.ndarray,
    omixai_path: str,
    feature_names: list[str],
    top_k_values: tuple[int, ...] = (50, 100, 300, 500),
) -> dict:
    """
    Align PFI scores with the OmiXAI ranking BY FEATURE NAME, then compare.

    omixai_ranking.csv is sorted by importance, so its row order is the ranking,
    NOT the feature order. pfi_scores, in contrast, is in feature-matrix column
    order (== feature_names). Comparing them positionally (the previous bug)
    correlates unrelated features. We therefore index both by feature name and
    intersect.
    """
    if len(feature_names) != len(pfi_scores):
        raise ValueError(
            f"feature_names ({len(feature_names)}) and pfi_scores "
            f"({len(pfi_scores)}) lengths differ — wrong feature_names.json?"
        )

    omixai_df  = pd.read_csv(omixai_path, index_col=0)          # index = feature name
    pfi_series = pd.Series(pfi_scores, index=feature_names)     # index = feature name

    common = omixai_df.index.intersection(pfi_series.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping feature names between OmiXAI ranking and PFI — "
            "feature_names.json does not match the ranking CSV."
        )

    omixai_common = omixai_df.loc[common, "mean_deviation"]
    pfi_common    = pfi_series.loc[common]

    rho, pval = spearmanr(omixai_common.values, pfi_common.values)

    omixai_top = omixai_common.sort_values(ascending=False).index
    pfi_top    = pfi_common.sort_values(ascending=False).index
    overlap = {
        k: len(set(omixai_top[:k]) & set(pfi_top[:k])) / k
        for k in top_k_values
    }

    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "overlap_at_k": overlap,
        "n_common": int(len(common)),
    }


def print_results(stats: dict) -> None:
    print(f"\nSpearman ρ (OmiXAI vs PFI-RF) = {stats['spearman_rho']:.3f}  "
          f"(p = {stats['spearman_pval']:.2e})   [{stats['n_common']} common features]")
    print("\nFeature overlap  OmiXAI ∩ PFI-RF:")
    for k, frac in stats["overlap_at_k"].items():
        print(f"  top-{k:>4} :  {frac:.1%}  ({int(frac * k)}/{k} features)")


# ------------------------------------------------------------------

def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = load_data(args.features, args.labels)
    pos_frac = y.sum() / len(y)
    print(f"  X shape: {X.shape}, positives: {y.sum()}/{len(y)} ({pos_frac:.1%})")
    if pos_frac > 0.5:
        print("  WARNING: >50% positive. labels_flat.npy looks positive-skewed "
              "(the balanced test split should be ~25%). You are likely running on "
              "a STALE feature_matrix_flat.npy/labels_flat.npy — regenerate them "
              "with run_omixai_gnn.py before trusting these PFI numbers.")

    feature_names = None
    if args.feature_names:
        feature_names = json.loads(Path(args.feature_names).read_text())
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names ({len(feature_names)}) != matrix columns "
                f"({X.shape[1]}). The matrix and feature_names.json are from "
                f"different runs."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Random Forest...")
    rf = train_rf(X_train, y_train)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, rf.predict(X_test), digits=3))

    print(f"Running PFI (n_repeats={args.n_repeats})...")
    pfi_scores = run_pfi(rf, X_test, y_test, n_repeats=args.n_repeats)
    np.save(out / "pfi_rf_scores.npy", pfi_scores)
    print(f"  Saved → {out}/pfi_rf_scores.npy")

    # intrinsic RF importance (fast, for reference)
    np.save(out / "rf_impurity_scores.npy", rf.feature_importances_)

    summary = pd.DataFrame({
        "pfi_rf_importance": pfi_scores,
        "rf_impurity": rf.feature_importances_,
    })
    if feature_names is not None:
        summary.index = feature_names          # name-indexed for downstream joins
        summary.index.name = "feature"
    summary.to_csv(out / "pfi_rf_summary.csv")
    print(f"  Saved → {out}/pfi_rf_summary.csv")

    if args.omixai:
        if feature_names is None:
            raise ValueError(
                "--omixai comparison requires --feature_names (results/feature_names.json) "
                "to align PFI scores with the ranking by feature name."
            )
        print("Comparing with OmiXAI ranking...")
        stats = compare_with_omixai(pfi_scores, args.omixai, feature_names)
        print_results(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  required=True, help="path to feature matrix (.npy), shape (n_intervals, n_features)")
    parser.add_argument("--labels",    required=True, help="path to label vector (.npy), shape (n_intervals,)")
    parser.add_argument("--omixai",    default=None,  help="path to OmiXAI ranking CSV (results/omixai_ranking.csv)")
    parser.add_argument("--feature_names", default=None,
                        help="results/feature_names.json — canonical feature order "
                             "matching the matrix columns (required for --omixai compare)")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--out_dir",   default="results/")
    main(parser.parse_args())
