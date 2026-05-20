"""
PCA utility for VSV extraction.
Adapted from VISTA/myutils.py.
"""

import torch
import torch.nn as nn


def _svd_flip(u, v):
    """Sign flip so that the largest absolute value in each column of u is positive."""
    max_abs_cols = torch.argmax(torch.abs(u), dim=0)
    signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1], device=u.device)])
    u = u * signs
    v = v * signs.unsqueeze(1)
    return u, v


class PCA(nn.Module):
    """Incremental-free PCA via full SVD. GPU-compatible."""

    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "PCA":
        """
        Fit PCA on X of shape [N, D].
        Registers mean_ [1, D] and components_ [n_components, D] as buffers.
        """
        n, d = X.shape
        k = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))          # [1, D]
        Z = X - self.mean_                                               # center
        U, S, Vh = torch.linalg.svd(Z.float(), full_matrices=False)
        U, Vt = _svd_flip(U, Vh)
        self.register_buffer("components_", Vt[:k])                     # [k, D]
        self.register_buffer("explained_variance_", (S[:k] ** 2) / (n - 1))
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "components_"), "PCA must be fit before transform"
        return (X - self.mean_) @ self.components_.t()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)
