"""
Permutation Feature Importance (PFI) for ConvMZC and GraphMZC.

For each omics feature, its values are permuted across all genomic intervals
(breaking feature-label association) and the F1 drop is measured. The result
can be compared with OmiXAI hybrid ranking via compare_rankings().
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from tqdm import tqdm

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DNA_CHANNELS = 4


# ------------------------------------------------------------------
# GNN
# ------------------------------------------------------------------

def compute_pfi_gnn(
    model,
    loader,
    n_features: int,
    width: int = 100,
    n_repeats: int = 5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """
    Permutation Feature Importance for a GNN model.

    Feature values are permuted across all intervals in the loader
    (not within a single interval), which correctly breaks the
    association between a feature and the genomic label.

    Parameters
    ----------
    model      : trained GraphMZC
    loader     : DataLoader yielding PyG Data objects
    n_features : number of omics features (excluding DNA channels)
    width      : sequence window length
    n_repeats  : permutations per feature; 5 gives stable estimates
    device     : defaults to CUDA if available

    Returns
    -------
    importance : ndarray (n_features,) — mean F1 drop per feature
    base_f1    : float — F1 before any permutation
    """
    device = device or _DEVICE
    model = model.to(device).eval()

    print("Loading data...")
    all_x, all_edges, all_y = [], [], []
    for dt in tqdm(loader, desc="data"):
        all_x.append(dt.x.cpu())
        all_edges.append(dt.edge_index.cpu())
        all_y.append(dt.y.cpu())

    base_f1 = _eval_gnn(model, all_x, all_edges, all_y, width, device)
    print(f"Baseline F1 = {base_f1:.4f}")

    importance = np.zeros(n_features, dtype=np.float64)

    for i in tqdm(range(n_features), desc="PFI"):
        drops = [
            base_f1 - _eval_gnn_permuted(model, all_x, all_edges, all_y, i, width, device)
            for _ in range(n_repeats)
        ]
        importance[i] = float(np.mean(drops))

    return importance, base_f1


def _eval_gnn(model, all_x, all_edges, all_y, width, device) -> float:
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, edge, y in zip(all_x, all_edges, all_y):
            x, edge, y = x.to(device), edge.to(device), y.to(device).long()
            valid = (edge < width).all(dim=0)
            out = model(x, edge[:, valid].squeeze())
            pred = torch.argmax(out, dim=-1)
            y_pred.extend(pred.cpu().numpy().flatten())
            y_true.extend(y.cpu().numpy().flatten())
    return f1_score(y_true, y_pred, zero_division=0)


def _eval_gnn_permuted(model, all_x, all_edges, all_y, feat_idx, width, device) -> float:
    col = feat_idx + _DNA_CHANNELS
    all_vals = torch.cat([x[:, :, col].reshape(-1) for x in all_x])
    all_vals = all_vals[torch.randperm(len(all_vals))]

    y_pred, y_true = [], []
    offset = 0
    with torch.no_grad():
        for x_orig, edge, y in zip(all_x, all_edges, all_y):
            n = x_orig.shape[1]
            x = x_orig.clone().to(device)
            x[:, :, col] = all_vals[offset : offset + n].to(device)
            offset += n

            edge, y = edge.to(device), y.to(device).long()
            valid = (edge < width).all(dim=0)
            out = model(x, edge[:, valid].squeeze())
            pred = torch.argmax(out, dim=-1)
            y_pred.extend(pred.cpu().numpy().flatten())
            y_true.extend(y.cpu().numpy().flatten())
    return f1_score(y_true, y_pred, zero_division=0)


# ------------------------------------------------------------------
# CNN
# ------------------------------------------------------------------

def compute_pfi_cnn(
    model,
    loader,
    n_features: int,
    width: int = 100,
    n_repeats: int = 5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """
    Permutation Feature Importance for a CNN model.

    Parameters match compute_pfi_gnn; loader yields (x, y) tensor pairs.
    """
    device = device or _DEVICE
    model = model.to(device).eval()

    print("Loading data...")
    all_x, all_y = [], []
    for x, y in tqdm(loader, desc="data"):
        all_x.append(x.cpu())
        all_y.append(y.cpu())

    base_f1 = _eval_cnn(model, all_x, all_y, device)
    print(f"Baseline F1 = {base_f1:.4f}")

    importance = np.zeros(n_features, dtype=np.float64)

    for i in tqdm(range(n_features), desc="PFI"):
        drops = [
            base_f1 - _eval_cnn_permuted(model, all_x, all_y, i, width, device)
            for _ in range(n_repeats)
        ]
        importance[i] = float(np.mean(drops))

    return importance, base_f1


def _eval_cnn(model, all_x, all_y, device) -> float:
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in zip(all_x, all_y):
            x, y = x.to(device), y.to(device).long()
            out = model(x)
            if out.dim() == 2:
                out = out.unsqueeze(0)
            pred = torch.argmax(out, dim=2)
            y_pred.extend(pred.cpu().numpy().flatten())
            y_true.extend(y.cpu().numpy().flatten())
    return f1_score(y_true, y_pred, zero_division=0)


def _eval_cnn_permuted(model, all_x, all_y, feat_idx, width, device) -> float:
    col = feat_idx + _DNA_CHANNELS
    # x shape: (1, width, n_features+4)
    all_vals = torch.cat([x[:, :, col].reshape(-1) for x in all_x])
    all_vals = all_vals[torch.randperm(len(all_vals))]

    y_pred, y_true = [], []
    offset = 0
    with torch.no_grad():
        for x_orig, y in zip(all_x, all_y):
            n = x_orig.shape[1]
            x = x_orig.clone().to(device)
            x[:, :, col] = all_vals[offset : offset + n].reshape(x_orig.shape[0], n).to(device)
            offset += n

            y = y.to(device).long()
            out = model(x)
            if out.dim() == 2:
                out = out.unsqueeze(0)
            pred = torch.argmax(out, dim=2)
            y_pred.extend(pred.cpu().numpy().flatten())
            y_true.extend(y.cpu().numpy().flatten())
    return f1_score(y_true, y_pred, zero_division=0)


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------

def compare_rankings(
    omixai_scores: np.ndarray,
    pfi_scores: np.ndarray,
    top_k_values: tuple[int, ...] = (50, 100, 300, 500),
) -> dict:
    """
    Compare OmiXAI hybrid ranking with PFI ranking.

    Returns Spearman rank correlation and feature overlap fractions at top-k.
    """
    rho, pval = spearmanr(omixai_scores, pfi_scores)

    omixai_top = np.argsort(omixai_scores)[::-1]
    pfi_top = np.argsort(pfi_scores)[::-1]

    overlap = {
        k: len(set(omixai_top[:k]) & set(pfi_top[:k])) / k
        for k in top_k_values
    }

    return {"spearman_rho": float(rho), "spearman_pval": float(pval), "overlap_at_k": overlap}


def print_comparison(stats: dict) -> None:
    print(f"Spearman ρ = {stats['spearman_rho']:.3f}  (p = {stats['spearman_pval']:.2e})")
    print("Overlap with OmiXAI at top-k:")
    for k, frac in stats["overlap_at_k"].items():
        print(f"  k = {k:>4}  →  {frac:.1%}")
