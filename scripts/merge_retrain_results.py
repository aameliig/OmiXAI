"""
Merge per-k CSV files produced by retrain_topk.slurm job array into a single table.

Usage (after all array jobs finish):
    python scripts/merge_retrain_results.py --out_dir results/
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def main(args):
    out = Path(args.out_dir)
    files = sorted(glob.glob(str(out / "retrain_table_k*.csv")))

    if not files:
        print("No per-k files found. Check that the job array completed.")
        return

    dfs = [pd.read_csv(f) for f in files]
    table = pd.concat(dfs, ignore_index=True).sort_values(["k", "method"])
    table.to_csv(out / "retrain_table.csv", index=False)
    print(f"Merged {len(files)} files → {out}/retrain_table.csv")
    print(table.pivot(index="k", columns="method", values=["f1", "auc"]).to_string())

    # Wilcoxon: OmiXAI F1 > PFI-RF F1 across k values
    if "PFI-RF" in table["method"].values:
        omixai_f1 = table[table.method == "OmiXAI"]["f1"].values
        pfi_f1    = table[table.method == "PFI-RF"]["f1"].values
        if len(omixai_f1) > 1:
            stat, p = wilcoxon(omixai_f1, pfi_f1, alternative="greater")
            print(f"\nWilcoxon (OmiXAI F1 > PFI-RF F1): stat={stat:.3f}  p={p:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/")
    main(parser.parse_args())
