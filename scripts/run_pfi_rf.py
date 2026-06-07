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
    feature_names: list[str] | None = None,
    top_k_values: tuple[int, ...] = (50, 100, 300, 500),
) -> dict:
    omixai_df = pd.read_csv(omixai_path, index_col=0)
    omixai_scores = omixai_df["mean_deviation"].values

    rho, pval = spearmanr(omixai_scores, pfi_scores)

    omixai_order = np.argsort(omixai_scores)[::-1]
    pfi_order = np.argsort(pfi_scores)[::-1]

    overlap = {
        k: len(set(omixai_order[:k]) & set(pfi_order[:k])) / k
        for k in top_k_values
    }

    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "overlap_at_k": overlap,
        "omixai_scores": omixai_scores,
        "pfi_scores": pfi_scores,
    }


def print_results(stats: dict) -> None:
    print(f"\nSpearman ρ (OmiXAI vs PFI-RF) = {stats['spearman_rho']:.3f}  "
          f"(p = {stats['spearman_pval']:.2e})")
    print("\nFeature overlap  OmiXAI ∩ PFI-RF:")
    for k, frac in stats["overlap_at_k"].items():
        print(f"  top-{k:>4} :  {frac:.1%}  ({int(frac * k)}/{k} features)")


# ------------------------------------------------------------------

def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = load_data(args.features, args.labels)
    print(f"  X shape: {X.shape}, positives: {y.sum()}/{len(y)}")

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

    if args.omixai:
        print("Comparing with OmiXAI ranking...")
        stats = compare_with_omixai(pfi_scores, args.omixai)
        print_results(stats)

        summary = pd.DataFrame({
            "pfi_rf_importance": pfi_scores,
            "rf_impurity": rf.feature_importances_,
        })
        summary.to_csv(out / "pfi_rf_summary.csv")
        print(f"  Saved → {out}/pfi_rf_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  required=True, help="path to feature matrix (.npy), shape (n_intervals, n_features)")
    parser.add_argument("--labels",    required=True, help="path to label vector (.npy), shape (n_intervals,)")
    parser.add_argument("--omixai",    default=None,  help="path to OmiXAI ranking CSV (results/omixai_ranking.csv)")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--out_dir",   default="results/")
    main(parser.parse_args())
