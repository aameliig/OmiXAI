import os

import numpy as np
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils import data
from tqdm import trange


def load_chromosome(chrom: str, dna_dir: str) -> str:
    files = sorted(f for f in os.listdir(dna_dir) if f"{chrom}_" in f)
    return "".join(load(os.path.join(dna_dir, f)) for f in files)


class GenomicDataset(data.Dataset):
    """
    PyTorch Dataset for fixed-length genomic intervals.

    Each item is a tuple (X, y) where:
      X : float32 array of shape (width, n_omics + 4)  — one-hot DNA + omics features
      y : int array of shape (width,)                  — per-nucleotide binary labels
    """

    def __init__(
        self,
        chroms: list[str],
        feature_names: list[str],
        dna_source: dict,
        features_source: dict,
        labels_source: dict,
        intervals: list,
    ) -> None:
        self.chroms = chroms
        self.feature_names = feature_names
        self.dna_source = dna_source
        self.features_source = features_source
        self.labels_source = labels_source
        self.intervals = intervals
        self._encoder = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx: int):
        chrom, begin, end = self.intervals[idx]
        begin, end = int(begin), int(end)

        dna_ohe = self._encoder.transform(list(self.dna_source[chrom][begin:end].upper()))

        omics_cols = [self.features_source[f][chrom][begin:end] for f in self.feature_names]
        if omics_cols:
            X = np.hstack((dna_ohe, np.array(omics_cols).T / 1000)).astype(np.float32)
        else:
            X = dna_ohe.astype(np.float32)

        y = self.labels_source[chrom][begin:end]
        return X, y


def get_train_test_split(
    width: int,
    chroms: list[str],
    feature_names: list[str],
    dna: dict,
    dna_features: dict,
    labels: dict,
) -> tuple[GenomicDataset, GenomicDataset]:
    """
    Build train and test GenomicDatasets with stratified split by chromosome.

    Positive intervals (containing at least one labelled nucleotide) are
    balanced 1:3 against negative intervals sampled without replacement.
    """
    pos, neg = [], []
    for chrom in chroms:
        for start in trange(0, labels[chrom].shape - width, width, desc=chrom):
            interval = [chrom, start, min(start + width, labels[chrom].shape)]
            (pos if labels[chrom][start : start + width].any() else neg).append(interval)

    neg = np.array(neg)[
        np.random.choice(len(neg), size=len(pos) * 3, replace=False)
    ].tolist()

    all_intervals = pos + neg
    strata = [f"{int(i < 400)}_{iv[0]}" for i, iv in enumerate(all_intervals)]
    train_idx, test_idx = next(StratifiedKFold().split(all_intervals, strata))

    return (
        GenomicDataset(chroms, feature_names, dna, dna_features, labels,
                       [all_intervals[i] for i in train_idx]),
        GenomicDataset(chroms, feature_names, dna, dna_features, labels,
                       [all_intervals[i] for i in test_idx]),
    )
