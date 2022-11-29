import sys

import torch
import torch.nn.functional as F


# add dnnlib and torch_utils to PYTHONPATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path("./src")

from torch_utils import training_stats

#----------------------------------------------------------------------------

def variance_loss(z: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z.

    Args:
        z (torch.Tensor): NxD Tensor containing projected features from view 1.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z))
    return std_loss

#----------------------------------------------------------------------------

def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z.size()

    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (N - 1)

    diag = torch.eye(D, device=z.device)
    cov_loss = cov_z[~diag.bool()].pow_(2).sum() / D
    return cov_loss
    