import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMZC(nn.Module):
    """
    12-layer convolutional model for nucleotide-interval classification.

    Input tensor shape: (batch, 1, width, n_omics + 4)
    where 4 = one-hot DNA channels (A/T/G/C).

    Parameters
    ----------
    width        : sequence window length in nucleotides
    n_features   : number of omics / k-mer features (excluding DNA channels)
    """

    def __init__(self, width: int, n_features: int) -> None:
        super().__init__()
        self.width = width
        self.n_features = n_features

        self.seq = nn.Sequential(
            nn.Conv2d(1,   4,   kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(2,  4),
            nn.Conv2d(4,   8,   kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(4,  8),
            nn.Conv2d(8,   16,  kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(8,  16),
            nn.Conv2d(16,  32,  kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(16, 32),
            nn.Conv2d(32,  64,  kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(16, 64),
            nn.Conv2d(64,  128, kernel_size=(5, 5), padding=2), nn.ReLU(), nn.GroupNorm(32, 128),
            nn.Conv2d(128, 64,  kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(32, 64),
            nn.Conv2d(64,  32,  kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(16, 32),
            nn.Conv2d(32,  16,  kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(8,  16),
            nn.Conv2d(16,  8,   kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(4,  8),
            nn.Conv2d(8,   4,   kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(4,  4),
            nn.Conv2d(4,   1,   kernel_size=(3, 3), padding=1), nn.ReLU(), nn.GroupNorm(1,  1),
            nn.AlphaDropout(p=0.2),
            nn.Linear(n_features + 4, 500),
            nn.AlphaDropout(p=0.2),
            nn.SELU(),
            nn.Linear(500, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, 1, self.width, self.n_features + 4)
        x = self.seq(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=-1)
