import os
import numpy as np

from joblib import load
from tqdm import trange

from torch.utils import data
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold

def chrom_reader(chrom):
    files = sorted([i for i in os.listdir(f'z_dna/hg38_dna/') if f"{chrom}_" in i])
    return ''.join([load(f"z_dna/hg38_dna/{file}") for file in files])


class Dataset(data.Dataset):
    def __init__(self, chroms, features,
                 dna_source, features_source,
                 labels_source, intervals):
        self.chroms = chroms
        self.features = features
        self.dna_source = dna_source
        self.features_source = features_source
        self.labels_source = labels_source
        self.intervals = intervals
        self.le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, index):
        interval = self.intervals[index]
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
        y = self.labels_source[interval[0]][interval[1]: interval[2]]

        return (X, y)



def get_train_test_dataset(width, chroms, feature_names, DNA, DNA_features, ZDNA):
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

    equalized = ints_in
    equalized = [[inter[0], int(inter[1]), int(inter[2])] for inter in equalized]

    train_inds, test_inds = next(StratifiedKFold().split(equalized, [f"{int(i < 400)}_{elem[0]}"
                                                                    for i, elem
                                                                    in enumerate(equalized)]))

    train_intervals, test_intervals = [equalized[i] for i in train_inds], [equalized[i] for i in test_inds]

    train_dataset = Dataset(chroms, feature_names,
                        DNA, DNA_features,
                        ZDNA, train_intervals)

    test_dataset = Dataset(chroms, feature_names,
                        DNA, DNA_features,
                        ZDNA, test_intervals)

    return train_dataset, test_dataset













