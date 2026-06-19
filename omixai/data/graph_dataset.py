import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data, Dataset


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
    neg_ratio: int = 2,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[GraphGenomicDataset, GraphGenomicDataset]:
    """
    Build train and test GraphGenomicDatasets using the canonical
    class-stratified split (see `stratified_split_intervals`).
    """
    train_intervals, test_intervals = stratified_split_intervals(
        width, chroms, labels, neg_ratio=neg_ratio, test_size=test_size, seed=seed
    )
    return (
        GraphGenomicDataset(chroms, feature_names, dna, dna_features, labels,
                            train_intervals, width),
        GraphGenomicDataset(chroms, feature_names, dna, dna_features, labels,
                            test_intervals, width),
    )


def stratified_split_intervals(width, chroms, labels, neg_ratio=2,
                               test_size=0.2, seed=42):
    """
    Canonical class-stratified train/test split over fixed-width intervals.

    Positives = intervals containing >=1 labelled nucleotide; negatives are
    sampled at `neg_ratio` x positives. The split is stratified by
    (class, chromosome) via StratifiedShuffleSplit, so both folds keep the same
    class balance. This replaces the legacy `int(i < 400)` strata, which (with
    shuffle=False) sent ~all positives to the test fold.

    Returns
    -------
    (train_intervals, test_intervals) : two lists of [chrom, begin, end]
    """
    np.random.seed(seed)
    pos, neg = [], []
    for c in chroms:
        n = labels[c].shape
        for st in range(0, n - width, width):
            iv = [c, st, min(st + width, n)]
            (pos if labels[c][st:st + width].any() else neg).append(iv)
    pos = np.array(pos, dtype=object)
    neg = np.array(neg, dtype=object)
    neg = neg[np.random.choice(len(neg), size=len(pos) * neg_ratio, replace=False)]
    equalized = [[r[0], int(r[1]), int(r[2])] for r in np.concatenate([pos, neg])]
    strat = np.array([f"{int(i < len(pos))}_{iv[0]}" for i, iv in enumerate(equalized)])
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed).split(equalized, strat))
    return [equalized[i] for i in tr], [equalized[i] for i in te]
