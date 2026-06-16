"""
Run OmiXAI interpretation on a trained GraphMZC model.

Data layout expected (mirrors vladislareon/z_dna repo):
  $DATA_DIR/
    hg38_dna/              — per-chromosome DNA chunks (joblib pkl)
    hg38_zdna/sparse/      — Z-DNA label SparseVectors (ZDNA_cousine.pkl)
    hg38_features/sparse/  — omics feature SparseVectors (one pkl per feature)

Caching:
  On first run the loaded DNA, features and train/test split are saved to
  --cache_dir (default: ~/omixai_cache). Subsequent runs skip the slow
  loading/splitting step and restore from cache in seconds.
  Use --no_cache to force a fresh reload.

Usage:
  python scripts/run_omixai_gnn.py \\
      --model    ~/weights/graphmzc.pt \\
      --data_dir ~/DNA/z_dna \\
      --width    100 \\
      --out_dir  results/
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from joblib import load, dump
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from omixai import OmiXAI, GraphMZC, get_train_test_split_graph
from genome_cache import load_genome_cached

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_chromosome(chrom: str, dna_dir: str) -> str:
    files = sorted(f for f in os.listdir(dna_dir) if f"{chrom}_" in f)
    return "".join(load(os.path.join(dna_dir, f)) for f in files)


def load_all_data(data_dir: str):
    # Delegates to the shared one-file genome cache (DNA + ZDNA + omics features).
    # First run builds ~/omixai_cache/genome.joblib; later runs read it in seconds.
    # Feature order is the raw os.listdir order (matches the trained weights) and
    # is frozen in the cache, so every script shares the exact same ordering.
    return load_genome_cached(data_dir)


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def cache_path(cache_dir: Path, width: int) -> Path:
    return cache_dir / f"split_w{width}.pkl"


def save_cache(cache_dir: Path, width: int, train_intervals, test_intervals, feature_names):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path(cache_dir, width), "wb") as f:
        pickle.dump({
            "train_intervals": train_intervals,
            "test_intervals":  test_intervals,
            "feature_names":   feature_names,
        }, f)
    print(f"Split cached → {cache_path(cache_dir, width)}")


def load_cache(cache_dir: Path, width: int):
    with open(cache_path(cache_dir, width), "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    out       = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(10)

    # -- data loading with optional cache --
    if not args.no_cache and cache_path(cache_dir, args.width).exists():
        print(f"Loading split from cache: {cache_path(cache_dir, args.width)}")
        cached       = load_cache(cache_dir, args.width)
        feature_names = cached["feature_names"]
        n_features    = len(feature_names)
        print(f"Omics features: {n_features}")

        # still need DNA/features for the dataset __getitem__
        DNA, ZDNA, DNA_features, _ = load_all_data(args.data_dir)

        from omixai.data import GraphGenomicDataset
        train_dataset = GraphGenomicDataset(
            CHROMS, feature_names, DNA, DNA_features, ZDNA,
            cached["train_intervals"], args.width
        )
        test_dataset = GraphGenomicDataset(
            CHROMS, feature_names, DNA, DNA_features, ZDNA,
            cached["test_intervals"], args.width
        )
    else:
        DNA, ZDNA, DNA_features, feature_names = load_all_data(args.data_dir)
        n_features = len(feature_names)
        print(f"Omics features: {n_features}")

        train_dataset, test_dataset = get_train_test_split_graph(
            args.width, CHROMS, feature_names, DNA, DNA_features, ZDNA
        )

        if not args.no_cache:
            save_cache(cache_dir, args.width,
                       train_dataset.intervals,
                       test_dataset.intervals,
                       feature_names)

    # Persist the canonical feature order. Every downstream script (PFI compare,
    # retrain top-k, old/new comparison) must load THIS file rather than re-deriving
    # feature_names from os.listdir — otherwise feature index i means different
    # things in different scripts and the rankings silently misalign.
    import json
    (out / "feature_names.json").write_text(json.dumps(feature_names))
    print(f"Saved canonical feature order ({len(feature_names)} features) "
          f"→ {out}/feature_names.json")

    # Filter train dataset to positive intervals only (intervals containing Z-DNA).
    # Interpretation only makes sense for True Positives, and iterating over
    # negative intervals wastes GPU time with no useful TP signal.
    pos_intervals = [iv for iv in train_dataset.intervals
                     if ZDNA[iv[0]][int(iv[1]):int(iv[2])].any()]
    print(f"Train intervals: {len(train_dataset.intervals)} total, "
          f"{len(pos_intervals)} positive (used for interpretation)")

    from omixai.data import GraphGenomicDataset
    interp_dataset = GraphGenomicDataset(
        CHROMS, feature_names, DNA, DNA_features, ZDNA,
        pos_intervals, args.width
    )

    loader_params  = dict(batch_size=1, num_workers=4, shuffle=False)
    loader_interp  = DataLoader(interp_dataset, **loader_params)

    # -- model --
    model = GraphMZC(n_features=n_features)
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model = model.to(device).eval()
    print(f"Model loaded: {args.model}")

    # -- OmiXAI: positive train intervals only --
    pipeline = OmiXAI(model=model, n_features=n_features, model_type='gnn', device=device)
    scores   = pipeline.interpret(loader_interp, width=args.width)

    np.save(out / "omixai_gnn_scores.npy", scores)

    ranking = pipeline.rank_features(feature_names=feature_names)
    ranking.to_csv(out / "omixai_ranking.csv")
    print(f"\nTop 20 features:\n{ranking.head(20).to_string()}")

    # -- flat matrix for RF-PFI (built from the BALANCED test split) --
    # Column order matches feature_names (DNA channels 0:4 dropped), so the PFI
    # scores produced from this matrix align 1:1 with feature_names.json.
    print("\nBuilding flat feature matrix for RF-PFI (interval-level labels)...")
    X_list, y_list = [], []
    for i in tqdm(range(len(test_dataset))):
        item = test_dataset.get(i)
        X_list.append(item.x.squeeze(0)[:, 4:].numpy().mean(axis=0))
        # label = 1 if the interval contains ANY Z-DNA nucleotide. Class 1 = Z-DNA.
        y_list.append(int(item.y.squeeze(0).numpy().any()))

    y_arr = np.array(y_list, dtype=np.int32)
    pos_frac = float(y_arr.mean()) if len(y_arr) else 0.0
    print(f"Flat matrix: {len(y_arr)} intervals, "
          f"{int(y_arr.sum())} positive (class 1 = Z-DNA) = {pos_frac:.1%}")
    if pos_frac > 0.5:
        print("WARNING: >50% positive. The test split is 1:3 balanced and should "
              "be ~25% positive. A positive-skewed matrix means it was built over "
              "the positive-only interpretation set (or a stale file is present). "
              "Delete old feature_matrix_flat.npy/labels_flat.npy and rerun.")

    np.save(out / "feature_matrix_flat.npy", np.array(X_list, dtype=np.float32))
    np.save(out / "labels_flat.npy",         y_arr)
    print(f"Done. Results in {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--width",     type=int, default=100)
    parser.add_argument("--out_dir",   default="results/")
    parser.add_argument("--cache_dir", default=str(Path.home() / "omixai_cache"))
    parser.add_argument("--no_cache",  action="store_true",
                        help="ignore existing cache and reload from scratch")
    main(parser.parse_args())
