import torch
import torch.nn as nn


class VIBBottleneck(nn.Module):
    """Variational Information Bottleneck for ego-status tokens.

    Compresses ego-status information through a stochastic bottleneck so that
    only trajectory-relevant information passes to the VLM, mitigating the
    ego-status shortcut problem identified in SpaceDrive+.

    During training, samples z ~ N(mu, sigma^2) via the reparameterisation
    trick and penalises KL[q(z|x) || N(0,I)].  At inference time the mean
    is used directly (deterministic).

    Args:
        input_dim: Hidden dimension of ego tokens (PE encoder output dim).
        z_dim:     Bottleneck latent dimension.  ``None`` → same as *input_dim*
                   (no dimensionality reduction, only information compression).
        beta:      KL divergence loss weight (returned loss is pre-scaled).
    """

    def __init__(self, input_dim: int, z_dim: int = None, beta: float = 1e-3):
        super().__init__()
        if z_dim is None:
            z_dim = input_dim

        self.mu_head = nn.Linear(input_dim, z_dim)
        self.logvar_head = nn.Linear(input_dim, z_dim)
        self.proj_back = (
            nn.Linear(z_dim, input_dim) if z_dim != input_dim else nn.Identity()
        )
        self.beta = beta

    def forward(
        self, ego_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_pe: ``(B, num_tokens, dim)`` — assembled ego feature tensor.

        Returns:
            z_proj:  ``(B, num_tokens, dim)`` — bottleneck-filtered tokens.
            kl_loss: scalar — ``beta * KL[q(z|x) || N(0,I)]``.
        """
        mu = self.mu_head(ego_pe)
        logvar = self.logvar_head(ego_pe)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        z_proj = self.proj_back(z)

        kl_per_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_elem.mean()

        return z_proj, self.beta * kl_loss
