import numpy as np
import torch
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data, Dataset
from tqdm import trange


def _linear_edge_index(width: int) -> torch.Tensor:
    """Bidirectional edges for a linear chain of `width` nodes."""
    src = []
    dst = []
    for i in range(width - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return torch.tensor([src, dst], dtype=torch.long)


class GraphGenomicDataset(Dataset):
    """
    PyG Dataset for fixed-length genomic intervals represented as graphs.

    Each nucleotide position is a node connected to its immediate neighbours.
    Node features: one-hot DNA (4 channels) + omics / k-mer features.
    """

    def __init__(
        self,
        chroms: list[str],
        feature_names: list[str],
        dna_source: dict,
        features_source: dict,
        labels: dict,
        intervals: list,
        width: int,
    ) -> None:
        self.chroms = chroms
        self.feature_names = feature_names
        self.dna_source = dna_source
        self.features_source = features_source
        self.labels = labels
        self.intervals = intervals
        self._encoder = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))
        self._edge_index = _linear_edge_index(width)
        super().__init__(root=None)

    def len(self) -> int:
        return len(self.intervals)

    def get(self, idx: int) -> Data:
        chrom, begin, end = self.intervals[idx]
        begin, end = int(begin), int(end)

        dna_ohe = self._encoder.transform(list(self.dna_source[chrom][begin:end].upper()))

        omics_cols = [self.features_source[f][chrom][begin:end] for f in self.feature_names]
        if omics_cols:
            X = np.hstack((dna_ohe, np.array(omics_cols).T / 1000)).astype(np.float32)
        else:
            X = dna_ohe.astype(np.float32)

        x = torch.tensor(X, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(self.labels[chrom][begin:end], dtype=torch.int64).unsqueeze(0)

        return Data(x=x, edge_index=self._edge_index, y=y)


def get_train_test_split_graph(
    width: int,
    chroms: list[str],
    feature_names: list[str],
    dna: dict,
    dna_features: dict,
    labels: dict,
) -> tuple[GraphGenomicDataset, GraphGenomicDataset]:
    """
    Build train and test GraphGenomicDatasets with stratified split by chromosome.
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
        GraphGenomicDataset(chroms, feature_names, dna, dna_features, labels,
                            [all_intervals[i] for i in train_idx], width),
        GraphGenomicDataset(chroms, feature_names, dna, dna_features, labels,
                            [all_intervals[i] for i in test_idx], width),
    )
