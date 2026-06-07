import os
import numpy as np

from joblib import load
from tqdm import trange

from torch.utils import data
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold

# Graph dataset
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data

def chrom_reader(chrom):
    files = sorted([i for i in os.listdir(f'z_dna/hg38_dna/') if f"{chrom}_" in i])
    return ''.join([load(f"z_dna/hg38_dna/{file}") for file in files])


class GraphDataset(Dataset):
    def __init__(self, chroms, features,
                 dna_source, features_source,
                 labels, intervals, width,
                 transform=None, pre_transform=None, pre_filter=None):
        self.chroms = chroms
        self.features = features
        self.dna_source = dna_source
        self.features_source = features_source
        self.labels = labels
        self.intervals = intervals
        self.le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))

        self.ei = [[],[]]
        for i in range(width-1):
            self.ei[0].append(i)
            self.ei[0].append(i+1)
            self.ei[1].append(i+1)
            self.ei[1].append(i)
        super().__init__(None, None, None)

    def len(self):
        return len(self.intervals)

    def get(self, idx):
        interval = self.intervals[idx]
        chrom = interval[0]
        begin = int(interval[1])
        end = int(interval[2])
        dna_OHE = self.le.transform(list(self.dna_source[chrom][begin:end].upper()))

        feature_matr = []
        for feature in self.features:
            source = self.features_source[feature]
            feature_matr.append(source[chrom][begin:end])

        if len(feature_matr) > 0:
            X = np.hstack((dna_OHE, np.array(feature_matr).T/1000)).astype(np.float32)
        else:
            X = dna_OHE.astype(np.float32)
        X = torch.tensor(X, dtype=torch.float)

        edge_index = torch.tensor(np.array(self.ei), dtype=torch.long)

        y = self.labels[interval[0]][interval[1]: interval[2]]
        y = torch.tensor(y, dtype=torch.int64)

        return Data(x=X.unsqueeze(0), edge_index=edge_index, y=y.unsqueeze(0))




def get_train_test_dataset_edges_graph(width, chroms, feature_names, DNA, DNA_features, ZDNA):
    ei_1 = [[],[]]

    for i in range(width-1):
        if i+1 < width:
            ei_1[0].append(i)
            ei_1[0].append(i+1)
            ei_1[1].append(i+1)
            ei_1[1].append(i)

    edge = torch.tensor(np.array(ei_1), dtype=torch.long)

    ints_in = []
    ints_out = []

    for chrm in chroms:
      for st in trange(0, ZDNA[chrm].shape - width, width):
          interval = [st, min(st + width, ZDNA[chrm].shape)]
          if ZDNA[chrm][interval[0]: interval[1]].any():
              ints_in.append([chrm, interval[0], interval[1]])
          else:
              ints_out.append([chrm, interval[0], interval[1]])

    ints_in = np.array(ints_in)
    ints_out = np.array(ints_out)[np.random.choice(range(len(ints_out)), size=len(ints_in) * 3, replace=False)]

    equalized = np.vstack((ints_in, ints_out))
    equalized = [[inter[0], int(inter[1]), int(inter[2])] for inter in equalized]

    train_inds, test_inds = next(StratifiedKFold().split(equalized, [f"{int(i < 400)}_{elem[0]}"
                                                                    for i, elem
                                                                    in enumerate(equalized)]))

    train_intervals, test_intervals = [equalized[i] for i in train_inds], [equalized[i] for i in test_inds]

    train_dataset = GraphDataset(chroms, feature_names,
                        DNA, DNA_features,
                        ZDNA, train_intervals,width)

    test_dataset = GraphDataset(chroms, feature_names,
                        DNA, DNA_features,
                        ZDNA, test_intervals, width)

    return train_dataset, test_dataset, edge







