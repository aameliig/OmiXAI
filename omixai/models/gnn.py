import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphMZC(torch.nn.Module):
    """
    13-layer GraphSAGE model for node-level classification on genomic graphs.

    Each nucleotide position is a node; edges connect adjacent positions.
    Node features: one-hot DNA (4 channels) + omics / k-mer features.

    Parameters
    ----------
    n_features : number of omics / k-mer features (excluding DNA channels)
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        in_dim = n_features + 4  # +4 for one-hot DNA

        self.convs = nn.ModuleList([
            SAGEConv(in_dim, 1800),
            SAGEConv(1800,   1650),
            SAGEConv(1650,   1500),
            SAGEConv(1500,   1350),
            SAGEConv(1350,   1200),
            SAGEConv(1200,   1050),
            SAGEConv(1050,    900),
            SAGEConv(900,     750),
            SAGEConv(750,     600),
            SAGEConv(600,     450),
            SAGEConv(450,     300),
            SAGEConv(300,     150),
            SAGEConv(150,      64),
        ])

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = edge.to(x.device)
        for conv in self.convs:
            x = F.relu(conv(x, edge))
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=-1)
