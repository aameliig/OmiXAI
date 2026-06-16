"""
One-file cache for the loaded genome (DNA + Z-DNA labels + omics features).

Reading the per-chromosome DNA chunks and ~2000 per-feature pickles takes
minutes and currently happens on EVERY job — and each retrain_topk array task
reloads the genome independently. This caches the parsed objects into a single
(compressed) joblib file, so subsequent runs read one file instead of thousands.

What is cached: the RAW loaded data only (DNA, ZDNA, DNA_features, feature_names).
NOT the per-interval processed tensors — those depend on the chosen feature
subset (retrain uses a different subset per k) and would not be reusable. The
cached feature_names is the raw os.listdir order (matching the trained weights);
freezing it in the cache also guarantees every script shares the exact order.

Disable with env var OMIXAI_NO_GENOME_CACHE=1, or just delete the cache file.
Default location: ~/omixai_cache/genome.joblib
"""
from __future__ import annotations

import os
from pathlib import Path

from joblib import load, dump
from tqdm import tqdm

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]


def _default_cache_path() -> Path:
    return Path.home() / "omixai_cache" / "genome.joblib"


def _load_chromosome(chrom: str, dna_dir: str) -> str:
    files = sorted(f for f in os.listdir(dna_dir) if f"{chrom}_" in f)
    return "".join(load(os.path.join(dna_dir, f)) for f in files)


def _resolve_feature_order(features_dir: str) -> list[str]:
    """
    Decide the feature column order.

    The trained weights expect features in the EXACT order used during training
    on Colab — which was os.listdir order there, and is NOT reproducible on the
    cluster. The authors saved that order to "Список признаков.csv"
    (column 'feature'). Point env var OMIXAI_FEATURE_ORDER at that CSV to use it.

    Falls back to raw os.listdir order if the env var is not set (will NOT match
    the weights on a different machine — only correct if listdir matches training).
    """
    available = {f[:-4] for f in os.listdir(features_dir) if f.endswith(".pkl")}

    order_file = os.environ.get("OMIXAI_FEATURE_ORDER")
    if not order_file:
        print("WARNING: OMIXAI_FEATURE_ORDER not set — using raw os.listdir order. "
              "This only matches the trained weights if listdir equals the training "
              "order. Set it to the saved feature CSV for correct alignment.")
        return [f[:-4] for f in os.listdir(features_dir) if f.endswith(".pkl")]

    import csv
    with open(order_file, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if "feature" not in rows[0]:
        raise ValueError(f"{order_file} has no 'feature' column (got {list(rows[0])})")
    order = [r["feature"] for r in rows]

    present = [f for f in order if f in available]
    missing = [f for f in order if f not in available]
    extra   = [f for f in available if f not in set(order)]
    print(f"Feature order from {order_file}: {len(order)} listed, "
          f"{len(present)} present on disk, {len(missing)} missing, "
          f"{len(extra)} on disk but not in CSV.")
    if missing:
        print(f"  WARNING: {len(missing)} CSV features missing on disk "
              f"(first few: {missing[:5]}). n_features will differ from training "
              f"→ weights may not load.")
    if extra:
        print(f"  NOTE: {len(extra)} disk features not in CSV are DROPPED "
              f"(first few: {extra[:5]}).")
    return present


def load_genome(data_dir: str):
    """Load DNA, Z-DNA labels and all omics features from disk (slow path)."""
    dna_dir      = os.path.join(data_dir, "hg38_dna")
    zdna_path    = os.path.join(data_dir, "hg38_zdna", "sparse", "ZDNA_cousine.pkl")
    features_dir = os.path.join(data_dir, "hg38_features", "sparse")

    print("Loading DNA sequences...")
    DNA = {chrom: _load_chromosome(chrom, dna_dir) for chrom in tqdm(CHROMS)}

    print("Loading Z-DNA labels...")
    ZDNA = load(zdna_path)

    feature_names = _resolve_feature_order(features_dir)
    print(f"Loading {len(feature_names)} omics features...")
    DNA_features = {feat: load(os.path.join(features_dir, f"{feat}.pkl"))
                    for feat in tqdm(feature_names)}

    return DNA, ZDNA, DNA_features, feature_names


def load_genome_cached(data_dir: str, cache_path=None):
    """
    Return (DNA, ZDNA, DNA_features, feature_names), using a one-file cache.

    First call builds the cache; later calls read it. Set
    OMIXAI_NO_GENOME_CACHE=1 to bypass entirely.
    """
    if os.environ.get("OMIXAI_NO_GENOME_CACHE"):
        return load_genome(data_dir)

    cache_path = Path(cache_path) if cache_path else _default_cache_path()

    if cache_path.exists():
        print(f"Loading genome from cache: {cache_path}")
        d = load(cache_path)
        return d["DNA"], d["ZDNA"], d["DNA_features"], d["feature_names"]

    DNA, ZDNA, DNA_features, feature_names = load_genome(data_dir)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving genome cache → {cache_path} (one-time, compressed)...")
        dump(
            {"DNA": DNA, "ZDNA": ZDNA,
             "DNA_features": DNA_features, "feature_names": feature_names},
            cache_path, compress=3,
        )
        print("Genome cache saved.")
    except Exception as e:                       # disk quota, perms, etc.
        print(f"WARNING: could not write genome cache ({e}); continuing without it.")

    return DNA, ZDNA, DNA_features, feature_names
