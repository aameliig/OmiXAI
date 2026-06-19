"""
Thin runner for top-k feature-reduction retraining (paper Table 3).

Uses the package function omixai.training.retrain_topk (GraphMZC, canonical
stratified split, batch=32, seed=42, best-epoch). Pick the feature ranking arm:

  OmiXAI arm:  --ranking results/omixai_ranking.csv
  PFI arm:     --pfi_scores results/pfi_dl/pfi_dl_scores.npy \\
               --feature_names results/feature_names.json

Usage:
  python scripts/run_retrain.py --data_dir DIR --ranking results/omixai_ranking.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

from omixai.data import load_genome_cached, CHROMS
from omixai.training import retrain_topk


def main(args):
    DNA, ZDNA, DNA_features, feat = load_genome_cached(args.data_dir)

    if args.pfi_scores:
        scores = np.load(args.pfi_scores)
        names = json.loads(Path(args.feature_names).read_text())
        ranked = [names[i] for i in np.argsort(scores)[::-1]]
        method = "PFI"
    else:
        import pandas as pd
        rank = pd.read_csv(args.ranking, index_col=0)["mean_deviation"]
        ranked = rank.sort_values(ascending=False).index.tolist()
        method = "OmiXAI"

    k_list = [args.k] if args.k else [50, 100, 300, 500, 700, None]
    df = retrain_topk(ranked, CHROMS, DNA, DNA_features, ZDNA,
                      k_list=k_list, width=args.width, n_epochs=args.n_epochs,
                      num_workers=args.num_workers, method_name=method)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix = (f"_k{args.k}" if args.k else "") + ("_pfi" if args.pfi_scores else "_omixai")
    df.to_csv(out / f"retrain{suffix}.csv", index=False)
    print(f"\nSaved → {out}/retrain{suffix}.csv\n{df.to_string(index=False)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ranking", default="results/omixai_ranking.csv")
    p.add_argument("--pfi_scores", default=None)
    p.add_argument("--feature_names", default="results/feature_names.json")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="results/")
    main(p.parse_args())
