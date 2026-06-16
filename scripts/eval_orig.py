"""Ground-truth A/B: evaluate the checkpoint with the ORIGINAL model class,
ORIGINAL dataset and ORIGINAL per-interval metric. If this gives ~0.9 AUC /
~0.73 F1, the weights are fine and the bug is in our refactor. If it also gives
~0.1, the problem is the checkpoint/data/feature-order/env."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graph model framework"))   # original modules
sys.path.insert(0, str(ROOT / "scripts"))                 # genome_cache

from genome_cache import load_genome_cached
import graph_data_preparation as gdp          # ORIGINAL split + dataset
from graph_model import GraphZSAGEConv_v5_lin  # ORIGINAL model class

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "M"]]


def test_quiet(model, loader, width):
    aucs, prs, recs, f1s = [], [], [], []
    model.eval()
    with torch.no_grad():
        for dt in loader:
            x, edge, y = dt.x.cuda(), dt.edge_index.cuda(), dt.y.cuda().long()
            edge = edge[:, (edge < width).all(dim=0)]
            out = model(x, edge)
            pred = torch.argmax(out, dim=-1).cpu().numpy().flatten()
            prob = nn.Softmax(dim=-1)(out)[..., 1].cpu().numpy().flatten()
            yt = y.cpu().numpy().flatten()
            if np.std(yt) != 0:
                aucs.append(roc_auc_score(yt, prob))
                prs.append(precision_score(yt, pred, zero_division=0))
                recs.append(recall_score(yt, pred, zero_division=0))
            else:
                aucs.append(0); prs.append(0); recs.append(0)
            f1s.append(f1_score(yt, pred, zero_division=0))
    return map(np.mean, (aucs, prs, recs, f1s))


def main():
    model_path = sys.argv[1]
    data_dir   = sys.argv[2]
    width      = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    np.random.seed(10)
    DNA, ZDNA, DNA_features, feat = load_genome_cached(data_dir)
    print(f"Features: {len(feat)}")

    np.random.seed(10)   # original split uses np.random + StratifiedKFold
    _, test_ds, edges = gdp.get_train_test_dataset_edges_graph(
        width, CHROMS, feat, DNA, DNA_features, ZDNA)
    loader = DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=False)

    model = GraphZSAGEConv_v5_lin(top_count=len(feat), edge=edges).cuda()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    print(f"Model loaded: {model_path}")

    auc, pr, rec, f1 = test_quiet(model, loader, width)
    print("\n=== ORIGINAL test() (per-interval means) ===")
    print(f"  AUC={auc:.4f}  P={pr:.4f}  R={rec:.4f}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
