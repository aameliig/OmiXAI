import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# optional: IPython / matplotlib only available in notebooks
try:
    from IPython.display import clear_output as _clear_output
    import matplotlib.pyplot as plt
    _INTERACTIVE = True
except ImportError:
    _INTERACTIVE = False

def _clear():
    if _INTERACTIVE:
        _clear_output()

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _nll_loss(output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.NLLLoss()(output.permute(0, 2, 1), y)


def _batch_metrics(output, y, pred):
    y_flat = y.cpu().numpy().flatten()
    p_flat = pred.cpu().numpy().flatten()
    prob = nn.Softmax(dim=-1)(output)[..., 1].detach().cpu().numpy().flatten()

    if np.std(y_flat) == 0:
        return 0.0, 0.0, 0.0, 0.0

    auc = roc_auc_score(y_flat, prob)
    prec = precision_score(y_flat, p_flat, zero_division=0)
    rec = recall_score(y_flat, p_flat, zero_division=0)
    f1 = f1_score(y_flat, p_flat, zero_division=0)
    return auc, prec, rec, f1


def _train_epoch(model, optimizer, loader, width: int) -> dict:
    model.train()
    logs = {k: [] for k in ("auc", "prec", "rec", "f1", "loss")}

    for dt in loader:
        x = dt.x.to(_DEVICE)
        edge = dt.edge_index.to(_DEVICE)
        y = dt.y.to(_DEVICE).long()

        valid = (edge < width).all(dim=0)
        edge = edge[:, valid]

        optimizer.zero_grad()
        out = model(x, edge)
        pred = torch.argmax(out, dim=-1)

        auc, prec, rec, f1 = _batch_metrics(out, y, pred)
        loss = _nll_loss(out, y)
        loss.backward()
        optimizer.step()

        logs["auc"].append(auc)
        logs["prec"].append(prec)
        logs["rec"].append(rec)
        logs["f1"].append(f1)
        logs["loss"].append(loss.item())

    return logs


def evaluate(model, loader, width: int) -> dict:
    model.eval()
    logs = {k: [] for k in ("auc", "prec", "rec", "f1", "loss")}

    with torch.no_grad():
        for dt in loader:
            x = dt.x.to(_DEVICE)
            edge = dt.edge_index.to(_DEVICE)
            y = dt.y.to(_DEVICE).long()

            valid = (edge < width).all(dim=0)
            edge = edge[:, valid]

            out = model(x, edge)
            pred = torch.argmax(out, dim=-1)

            auc, prec, rec, f1 = _batch_metrics(out, y, pred)
            loss = _nll_loss(out, y)

            logs["auc"].append(auc)
            logs["prec"].append(prec)
            logs["rec"].append(rec)
            logs["f1"].append(f1)
            logs["loss"].append(loss.item())

    return {k: np.mean(v) for k, v in logs.items()}


def _plot(train_hist, val_hist, title, batch_size, n_show=20):
    plt.figure(figsize=(n_show, 4))
    plt.title(title)
    n = len(val_hist)
    t_arr = np.array([None] * (batch_size * n_show) + train_hist)
    v_arr = np.array([None] * n_show + val_hist)
    plt.plot(np.linspace(n - n_show + 1, n + 1, (n_show + 1) * batch_size),
             t_arr[-(n_show + 1) * batch_size:], c="red", label="train")
    plt.plot(np.linspace(n - n_show + 1, n + 1, n_show + 1),
             v_arr[-n_show - 1:], c="green", label="test")
    plt.ylim((0, 1))
    plt.yticks(np.linspace(0, 1, 11))
    plt.legend()
    plt.grid()
    plt.show()


def train(model, optimizer, n_epochs: int, train_loader, test_loader, width: int) -> dict:
    """
    Full training loop with per-epoch evaluation and live plotting.

    Returns dict of validation metric lists (auc, prec, rec, f1, loss).
    """
    history = {k: [] for k in ("auc", "prec", "rec", "f1", "loss")}
    train_f1_log = []

    for epoch in range(n_epochs):
        t0 = time.time()
        print(f"Epoch {epoch + 1}/{n_epochs}")

        tr = _train_epoch(model, optimizer, train_loader, width)
        val = evaluate(model, test_loader, width)

        train_f1_log.extend(tr["f1"])
        for k in history:
            history[k].append(val[k])

        _clear()
        if _INTERACTIVE:
            _plot(train_f1_log, history["f1"], "F1", len(tr["f1"]))
            _plot(tr["auc"], [val["auc"]] * len(tr["auc"]), "AUC", len(tr["auc"]))

        elapsed = time.time() - t0
        print(f"  time {elapsed/60:.1f} min  |  "
              f"AUC {val['auc']:.4f}  F1 {val['f1']:.4f}  "
              f"Prec {val['prec']:.4f}  Rec {val['rec']:.4f}")

    return history
