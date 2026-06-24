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
        in_dim = n_features + 4

        # named conv1..conv13 to match saved state_dict keys
        self.conv1  = SAGEConv(in_dim, 1800)
        self.conv2  = SAGEConv(1800,   1650)
        self.conv3  = SAGEConv(1650,   1500)
        self.conv4  = SAGEConv(1500,   1350)
        self.conv5  = SAGEConv(1350,   1200)
        self.conv6  = SAGEConv(1200,   1050)
        self.conv7  = SAGEConv(1050,    900)
        self.conv8  = SAGEConv(900,     750)
        self.conv9  = SAGEConv(750,     600)
        self.conv10 = SAGEConv(600,     450)
        self.conv11 = SAGEConv(450,     300)
        self.conv12 = SAGEConv(300,     150)
        self.conv13 = SAGEConv(150,      64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = edge.to(x.device)
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4,
                     self.conv5, self.conv6, self.conv7, self.conv8,
                     self.conv9, self.conv10, self.conv11, self.conv12,
                     self.conv13]:
            x = F.relu(conv(x, edge))
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=-1)
