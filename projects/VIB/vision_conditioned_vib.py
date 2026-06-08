import torch
import torch.nn as nn


class VisionConditionedVIB(nn.Module):
    """Vision-Conditioned Variational Information Bottleneck for ego tokens.

    Unlike the plain :class:`VIBBottleneck` which compresses ego tokens toward a
    fixed standard-normal prior ``N(0, I)``, this module compresses ego tokens
    toward a *vision-conditioned* prior ``p(z|vision) = N(mu_p, sigma_p^2)``
    produced by a small PriorNet on top of pure vision features.

    Minimising ``KL[q(z|ego) || p(z|vision)]`` removes the information that the
    vision stream already knows (KL-free, but useless for trajectory
    reconstruction) and keeps only the complementary ego information (incurs KL,
    but helps reconstruction).  This mitigates the ego-status shortcut while
    preserving the genuinely useful ego signal (precise speed, acceleration...).

    During training, samples ``z ~ N(mu_q, sigma_q^2)`` via the
    reparameterisation trick; at inference time the posterior mean is used
    directly (deterministic).

    Args:
        input_dim:  Hidden dimension of ego tokens (PE encoder output dim).
        vision_dim: Dimension of the pooled vision summary feeding the PriorNet.
        z_dim:      Bottleneck latent dimension.  ``None`` → same as *input_dim*
                    (no dimensionality reduction, only information compression).
        beta:       KL divergence loss weight (the returned loss is pre-scaled).
        prior_hidden_dim: Hidden dimension of the PriorNet MLP.  ``None`` →
                    ``vision_dim // 2``.
        logvar_clamp: Symmetric clamp range applied to both posterior and prior
                    log-variances for numerical stability.
    """

    def __init__(
        self,
        input_dim: int,
        vision_dim: int,
        z_dim: int = None,
        beta: float = 1e-3,
        prior_hidden_dim: int = None,
        logvar_clamp: float = 10.0,
    ):
        super().__init__()
        if z_dim is None:
            z_dim = input_dim
        if prior_hidden_dim is None:
            prior_hidden_dim = max(vision_dim // 2, 1)

        self.z_dim = z_dim
        self.beta = beta
        self.logvar_clamp = logvar_clamp

        # Encoder: q(z|ego) — posterior from ego feature.
        self.enc_mu = nn.Linear(input_dim, z_dim)
        self.enc_logvar = nn.Linear(input_dim, z_dim)

        # PriorNet: p(z|vision) — prior from pure vision summary.
        self.prior_mu = nn.Sequential(
            nn.Linear(vision_dim, prior_hidden_dim),
            nn.ReLU(),
            nn.Linear(prior_hidden_dim, z_dim),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(vision_dim, prior_hidden_dim),
            nn.ReLU(),
            nn.Linear(prior_hidden_dim, z_dim),
        )

        # Restore z back to the original ego token dimension.
        self.proj_back = (
            nn.Linear(z_dim, input_dim) if z_dim != input_dim else nn.Identity()
        )

    def forward(
        self,
        ego_feature: torch.Tensor,
        vision_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_feature:    ``(B, num_tokens, input_dim)`` — assembled ego tokens.
            vision_summary: ``(B, vision_dim)`` — pooled, ego-free vision feature.
                            Must NOT be detached so gradients reach the vision
                            encoder through the PriorNet.

        Returns:
            z_proj:  ``(B, num_tokens, input_dim)`` — bottleneck-filtered tokens.
            kl_loss: scalar — ``beta * KL[q(z|ego) || p(z|vision)]``.
            raw_kl: scalar — unscaled KL for logging/diagnostics.
        """
        # Align the PriorNet input dtype with the encoder weights (vision
        # features may arrive in bf16 while the VIB params are fp32).
        target_dtype = self.enc_mu.weight.dtype
        vision_summary = vision_summary.to(target_dtype)
        ego_feature = ego_feature.to(target_dtype)

        # --- Encoder: q(z|ego) ---
        mu_q = self.enc_mu(ego_feature)
        logvar_q = self.enc_logvar(ego_feature).clamp(
            -self.logvar_clamp, self.logvar_clamp
        )

        # --- PriorNet: p(z|vision) ---
        mu_p = self.prior_mu(vision_summary)
        logvar_p = self.prior_logvar(vision_summary).clamp(
            -self.logvar_clamp, self.logvar_clamp
        )

        # Broadcast the per-sample prior across the ego token dimension.
        if ego_feature.dim() == 3 and mu_p.dim() == 2:
            mu_p = mu_p.unsqueeze(1).expand_as(mu_q)
            logvar_p = logvar_p.unsqueeze(1).expand_as(logvar_q)

        # --- Reparameterisation ---
        if self.training:
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z = mu_q + std_q * eps
        else:
            z = mu_q

        z_proj = self.proj_back(z)

        raw_kl = self._kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

        return z_proj, self.beta * raw_kl, raw_kl

    @staticmethod
    def _kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
        """KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)] averaged over all elements.

        = 0.5 * ( logvar_p - logvar_q - 1
                  + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2 )
        """
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()

        kl = 0.5 * (
            logvar_p
            - logvar_q
            - 1.0
            + var_q / var_p
            + (mu_q - mu_p).pow(2) / var_p
        )
        return kl.mean()
