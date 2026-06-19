"""
Faithful reproduction of the paper's top-k retraining (Table 3), ported from
"3. Запуски на топе омиксных признаков/Запуск GraphZSAGE запуски на топ фичах.ipynb".

Key points (these differ from the earlier retrain_topk.py, which used the wrong
model and setup and collapsed):
  - model  : GraphZSAGEConv_13L with GroupNorm after every conv (deep GNN needs it)
  - batch  : 32  (not 1 → ~7x faster, stable)
  - seed   : set_random_seed(42) before model init
  - opt    : Adam(lr=1e-4, weight_decay=1e-4)
  - split  : StratifiedShuffleSplit(test_size=0.2) stratified by class+chromosome,
             negatives sampled 2x positives  (properly balanced)
  - loss   : NLLLoss (no class weight)
  - epochs : 10, report the best epoch (argmax val F1)

Select top-k features by name from a ranking:
  --ranking results/omixai_ranking.csv      (OmiXAI arm)
  --pfi_scores results/pfi_dl/pfi_dl_scores.npy --feature_names results/feature_names.json  (PFI arm)

k is the number of OMICS features; the model input dim is k+4 (DNA channels).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from genome_cache import load_genome_cached, CHROMS


def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ model
_CONV = [(1024, 1024), (1024, 512), (512, 512), (512, 256), (256, 256),
         (256, 128), (128, 128), (128, 64), (64, 64), (64, 32), (32, 32), (32, 2)]
_NORM = [(512, 1024), (512, 1024), (256, 512), (256, 512), (128, 256), (128, 256),
         (64, 128), (64, 128), (32, 64), (32, 64), (16, 32), (16, 32)]


class GraphZSAGEConv_13L(nn.Module):
    def __init__(self, top_count: int):
        super().__init__()
        dims = [(top_count, 1024)] + _CONV
        self.convs = nn.ModuleList([SAGEConv(i, o) for i, o in dims])          # 13 convs
        self.norms = nn.ModuleList([nn.GroupNorm(g, c) for g, c in _NORM])     # 12 norms

    def forward(self, x, edge):
        for i in range(12):
            x = self.convs[i](x, edge)
            x = x.permute(0, 2, 1)
            x = self.norms[i](x)
            x = x.permute(0, 2, 1)
            x = F.relu(x)
        x = self.convs[12](x, edge)
        return F.log_softmax(x, dim=-1)


def _linear_edge(width: int) -> torch.Tensor:
    src, dst = [], []
    for i in range(width - 1):
        src += [i, i + 1]; dst += [i + 1, i]
    return torch.tensor([src, dst], dtype=torch.long)


# ------------------------------------------------------------------ data / split
def build_split(width, ZDNA):
    np.random.seed(10)
    ins, outs = [], []
    for c in CHROMS:
        n = ZDNA[c].shape
        for st in range(0, n - width, width):
            iv = [c, st, min(st + width, n)]
            (ins if ZDNA[c][st:st + width].any() else outs).append(iv)
    ins = np.array(ins, dtype=object)
    outs = np.array(outs, dtype=object)
    sel = np.random.choice(len(outs), size=len(ins) * 2, replace=False)
    outs = outs[sel]
    np.random.seed(42)
    equalized = [[r[0], int(r[1]), int(r[2])] for r in np.vstack((ins, outs))]
    labels = np.array([1] * len(ins) + [0] * len(outs))
    strat = np.array([f"{l}_{iv[0]}" for l, iv in zip(labels, equalized)])
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                         random_state=42).split(equalized, strat))
    return [equalized[i] for i in tr], [equalized[i] for i in te]


class GDataset(torch.utils.data.Dataset):
    """Returns Data(x=(1,width,F), y=(1,width)); only full-width intervals."""
    def __init__(self, intervals, feats, DNA, DNA_features, ZDNA, width):
        self.iv = [iv for iv in intervals if int(iv[2]) - int(iv[1]) == width]
        self.feats, self.DNA, self.DF, self.Z, self.w = feats, DNA, DNA_features, ZDNA, width
        self.le = LabelBinarizer().fit(np.array([["A"], ["C"], ["T"], ["G"]]))

    def __len__(self): return len(self.iv)

    def __getitem__(self, i):
        c, b, e = self.iv[i][0], int(self.iv[i][1]), int(self.iv[i][2])
        ohe = self.le.transform(list(self.DNA[c][b:e].upper()))
        cols = [self.DF[f][c][b:e] for f in self.feats]
        X = np.hstack((ohe, np.array(cols).T / 1000)).astype(np.float32) if cols else ohe.astype(np.float32)
        x = torch.tensor(X, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(self.Z[c][b:e], dtype=torch.int64).unsqueeze(0)
        return Data(x=x, y=y)


# ------------------------------------------------------------------ train / eval
def _metrics(out, y):
    pred = out.argmax(dim=-1).cpu().numpy().flatten()
    prob = nn.Softmax(dim=-1)(out)[..., 1].detach().cpu().numpy().flatten()
    yt = y.cpu().numpy().flatten()
    if np.std(yt) == 0:
        return 0.0, 0.0, 0.0, f1_score(yt, pred, zero_division=0)
    return (roc_auc_score(yt, prob), precision_score(yt, pred, zero_division=0),
            recall_score(yt, pred, zero_division=0), f1_score(yt, pred, zero_division=0))


def run_epoch(model, opt, loader, edge, device, train):
    model.train() if train else model.eval()
    aucs, f1s = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for dt in loader:
            x, y = dt.x.to(device), dt.y.to(device).long()
            out = model(x, edge)
            a, p, r, f = _metrics(out, y)
            aucs.append(a); f1s.append(f)
            if train:
                opt.zero_grad()
                nn.NLLLoss()(out.permute(0, 2, 1), y).backward()
                opt.step()
    return float(np.mean(aucs)), float(np.mean(f1s))


def train_and_eval(feat_subset, DNA, DNA_features, ZDNA, train_iv, test_iv, width, n_epochs, device):
    set_random_seed(42)
    tr_ds = GDataset(train_iv, feat_subset, DNA, DNA_features, ZDNA, width)
    te_ds = GDataset(test_iv,  feat_subset, DNA, DNA_features, ZDNA, width)
    params = dict(batch_size=32, num_workers=4, shuffle=True)
    tr_ld = DataLoader(tr_ds, **params)
    te_ld = DataLoader(te_ds, batch_size=32, num_workers=4, shuffle=False)

    edge = _linear_edge(width).to(device)
    set_random_seed(42)
    model = GraphZSAGEConv_13L(len(feat_subset) + 4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best = {"f1": -1, "auc": 0}
    for ep in range(n_epochs):
        t0 = time.time()
        run_epoch(model, opt, tr_ld, edge, device, train=True)
        v_auc, v_f1 = run_epoch(model, None, te_ld, edge, device, train=False)
        if v_f1 > best["f1"]:
            best = {"f1": v_f1, "auc": v_auc, "epoch": ep + 1}
        print(f"  epoch {ep+1}/{n_epochs}  valF1={v_f1:.4f} valAUC={v_auc:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return best


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DNA, ZDNA, DNA_features, feat = load_genome_cached(args.data_dir)
    train_iv, test_iv = build_split(args.width, ZDNA)
    print(f"split: train={len(train_iv)} test={len(test_iv)}")

    if args.pfi_only:
        scores = np.load(args.pfi_scores)
        names = json.loads(Path(args.feature_names).read_text())
        order = [names[i] for i in np.argsort(scores)[::-1]]
        method = "PFI"
    else:
        rank = __import__("pandas").read_csv(args.ranking, index_col=0)["mean_deviation"]
        order = rank.sort_values(ascending=False).index.tolist()
        method = "OmiXAI"

    run_ks = [args.k] if args.k else [50, 100, 300, 500, 700, 1946]
    rows = []
    for k in run_ks:
        subset = [f for f in order[:k] if f in DNA_features]
        print(f"\n=== {method} k={k} ({len(subset)} omics + 4 DNA) ===", flush=True)
        b = train_and_eval(subset, DNA, DNA_features, ZDNA, train_iv, test_iv,
                           args.width, args.n_epochs, device)
        rows.append({"k": k, "method": method, "best_epoch": b["epoch"],
                     "f1": b["f1"], "auc": b["auc"]})
        print(f"  BEST: F1={b['f1']:.4f} AUC={b['auc']:.4f} @epoch {b['epoch']}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix = (f"_k{args.k}" if args.k else "") + ("_pfi" if args.pfi_only else "_omixai")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"retrain_faithful{suffix}.csv", index=False)
    print(f"\nSaved → {out}/retrain_faithful{suffix}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ranking", default="results/omixai_ranking.csv")
    p.add_argument("--pfi_scores", default="results/pfi_dl/pfi_dl_scores.npy")
    p.add_argument("--feature_names", default="results/feature_names.json")
    p.add_argument("--pfi_only", action="store_true")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--out_dir", default="results/")
    main(p.parse_args())
