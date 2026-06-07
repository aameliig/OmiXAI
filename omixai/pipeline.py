"""
OmiXAI — ensemble XAI pipeline for CNN and GNN models trained on genomic features.

Workflow:
    1. OmiXAI(model, model_type, n_features)
    2. .interpret(*loaders, methods, width)   → attribution dict
    3. .rank_features(feature_names)          → ranked DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from captum.attr import Deconvolution, GuidedBackprop, InputXGradient, IntegratedGradients
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# first 4 channels in the feature tensor are one-hot DNA (A/T/G/C)
_DNA_CHANNELS = 4

CNN_METHODS = ("IG", "IxG", "GB", "Deconv")
GNN_METHODS = ("IG", "IxG", "GB", "Deconv", "Saliency", "GNNExplainer")


class OmiXAI:
    """
    Parameters
    ----------
    model       : trained PyTorch model (ConvMZC or GraphMZC)
    model_type  : 'cnn' or 'gnn'
    n_features  : number of omics / k-mer features (excluding one-hot DNA channels)
    device      : defaults to CUDA if available
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        n_features: int,
        device: torch.device | None = None,
    ) -> None:
        if model_type not in ("cnn", "gnn"):
            raise ValueError("model_type must be 'cnn' or 'gnn'")

        self.device = device or _DEVICE
        self.model = model.to(self.device).eval()
        self.model_type = model_type
        self.n_features = n_features
        self._scores: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(
        self,
        *loaders,
        methods: list[str] | None = None,
        width: int = 100,
        n_steps_ig: int = 50,
    ) -> dict[str, np.ndarray]:
        """
        Compute mean attribution vectors over True Positive predictions.

        Pass both train and test loaders to use all available True Positives:
            pipeline.interpret(train_loader, test_loader, width=100)

        Parameters
        ----------
        *loaders    : DataLoader(s) — (x, y) for CNN or PyG Data for GNN
        methods     : attribution methods; defaults to all for the model type
        width       : sequence window length in nucleotides
        n_steps_ig  : integration steps for Integrated Gradients (default 50)

        Returns
        -------
        dict  method_name → mean attribution array of shape (n_features,)
        """
        available = CNN_METHODS if self.model_type == "cnn" else GNN_METHODS
        methods = list(methods or available)

        unknown = [m for m in methods if m not in available]
        if unknown:
            raise ValueError(
                f"Methods {unknown} not available for '{self.model_type}'. "
                f"Choose from: {available}"
            )

        for name in methods:
            total = np.zeros(self.n_features, dtype=np.float64)
            count = 0

            for loader in loaders:
                t, c = (
                    self._run_cnn(loader, name, width, n_steps_ig)
                    if self.model_type == "cnn"
                    else self._run_gnn(loader, name, width)
                )
                total += t
                count += c

            self._scores[name] = total / max(count, 1)
            print(f"{name}: averaged over {count} TP regions")

        return self._scores

    def rank_features(
        self,
        feature_names: list[str] | None = None,
        scores: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """
        Apply hybrid ranking to stored (or provided) attribution scores.

        For each method the per-feature percentage deviation from the method mean
        is computed, then averaged across methods. Features are sorted in
        descending order (higher = more important).

        Returns
        -------
        DataFrame with column 'mean_deviation', indexed by feature name.
        """
        src = scores or self._scores
        if not src:
            raise RuntimeError("No attribution scores — run interpret() first.")

        df = pd.DataFrame(src)
        if feature_names is not None:
            df.index = feature_names[: len(df)]

        pct = pd.DataFrame(index=df.index)
        for col in df.columns:
            mu = df[col].mean()
            pct[col] = (df[col] - mu) / mu * 100 if mu != 0 else 0.0

        pct["mean_deviation"] = pct.mean(axis=1)
        return pct[["mean_deviation"]].sort_values("mean_deviation", ascending=False)

    # ------------------------------------------------------------------
    # CNN
    # ------------------------------------------------------------------

    def _run_cnn(
        self, loader, method: str, width: int, n_steps_ig: int
    ) -> tuple[np.ndarray, int]:
        explainer = self._build_captum(method)
        total = np.zeros(self.n_features, dtype=np.float64)
        count = 0

        for x, y_true in tqdm(loader, desc=f"CNN/{method}", leave=False):
            x = x.to(self.device)
            y_true = y_true.to(self.device).long()

            with torch.no_grad():
                out = self.model(x)
                pred = torch.argmax(out, dim=1).reshape(1, width)

            tp = [i for i in range(width) if pred[0][i] == y_true[0][i] == 1]
            if not tp:
                continue

            attrs = self._captum_attr(explainer, x, method, n_steps_ig)
            attrs = torch.squeeze(attrs, dim=0)
            tp_mean = attrs[tp, _DNA_CHANNELS:].mean(dim=0)
            total += tp_mean.cpu().detach().numpy()
            count += 1

        return total, count

    # ------------------------------------------------------------------
    # GNN
    # ------------------------------------------------------------------

    def _run_gnn(self, loader, method: str, width: int) -> tuple[np.ndarray, int]:
        explainer_obj = self._build_gnn_explainer(method)
        total = np.zeros(self.n_features, dtype=np.float64)
        count = 0

        for dt in tqdm(loader, desc=f"GNN/{method}", leave=False):
            x = dt.x.to(self.device)
            edge = dt.edge_index.to(self.device)
            y = dt.y.to(self.device).long()

            valid = (edge < width).all(dim=0)
            edge = edge[:, valid]

            with torch.no_grad():
                out = self.model(x, edge.squeeze())
                pred = torch.argmax(out, dim=-1)

            tp = [i for i in range(width) if pred[0][i] == y[0][i] == 1]
            if not tp:
                continue

            explanation = explainer_obj(x.squeeze(), edge)
            mask = explanation.node_mask  # (width, n_features + _DNA_CHANNELS)
            tp_mean = mask[tp, _DNA_CHANNELS:].mean(dim=0)
            total += tp_mean.cpu().detach().numpy()
            count += 1

        return total, count

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_captum(self, method: str):
        return {
            "IG":     IntegratedGradients,
            "IxG":    InputXGradient,
            "GB":     GuidedBackprop,
            "Deconv": Deconvolution,
        }[method](self.model)

    def _build_gnn_explainer(self, method: str) -> Explainer:
        algo = (
            GNNExplainer(epochs=50)
            if method == "GNNExplainer"
            else CaptumExplainer("Saliency" if method == "Saliency" else method)
        )
        return Explainer(
            model=self.model,
            algorithm=algo,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

    @staticmethod
    def _captum_attr(
        explainer, x: torch.Tensor, method: str, n_steps_ig: int
    ) -> torch.Tensor:
        if method == "IG":
            return explainer.attribute(x, target=1, n_steps=n_steps_ig)
        return explainer.attribute(x, target=1)
