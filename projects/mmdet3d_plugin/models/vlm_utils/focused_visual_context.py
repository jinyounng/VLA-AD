import math
from typing import Optional

import torch
import torch.nn as nn


class FocusedVisualContext(nn.Module):
    """Dynamic top-k visual context fusion for trajectory/action hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        query_mode: str = "mean",
        fusion_mode: str = "gated_residual",
        alpha: float = 1.0,
    ):
        super().__init__()
        if query_mode != "mean":
            raise ValueError(f"Unsupported focus query_mode: {query_mode}")
        if fusion_mode != "gated_residual":
            raise ValueError(f"Unsupported focus fusion_mode: {fusion_mode}")

        self.hidden_dim = hidden_dim
        self.query_mode = query_mode
        self.fusion_mode = fusion_mode
        self.alpha = alpha

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.last_stats = {}

    def forward(
        self,
        q_action: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        top_k: int = 64,
    ) -> torch.Tensor:
        if q_action.dim() not in (2, 3):
            raise ValueError(f"q_action must be [N, D] or [N, T_action, D], got {q_action.shape}")
        if visual_tokens.dim() != 3:
            raise ValueError(f"visual_tokens must be [N, N_vis, D], got {visual_tokens.shape}")
        if q_action.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                f"q_action and visual_tokens batch mismatch: {q_action.shape[0]} vs {visual_tokens.shape[0]}"
            )
        if q_action.shape[-1] != self.hidden_dim or visual_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden dim mismatch: q_action {q_action.shape[-1]}, "
                f"visual_tokens {visual_tokens.shape[-1]}, expected {self.hidden_dim}"
            )

        original_dtype = q_action.dtype
        proj_dtype = self.q_proj.weight.dtype
        q_action_proj = q_action.to(proj_dtype)
        visual_tokens = visual_tokens.to(proj_dtype)

        if q_action_proj.dim() == 3:
            q_focus = q_action_proj.mean(dim=1)
        else:
            q_focus = q_action_proj

        q = self.q_proj(q_focus)
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)

        scores = torch.einsum("nd,nvd->nv", q, k) / math.sqrt(self.hidden_dim)
        if visual_mask is not None:
            if visual_mask.shape != scores.shape:
                raise ValueError(f"visual_mask shape {visual_mask.shape} does not match scores {scores.shape}")
            visual_mask = visual_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~visual_mask, torch.finfo(scores.dtype).min)

        n_vis = visual_tokens.shape[1]
        k_select = max(1, min(int(top_k), n_vis))
        top_scores, top_idx = torch.topk(scores, k=k_select, dim=1)

        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        selected_v = torch.gather(v, dim=1, index=gather_idx)

        attn = torch.softmax(top_scores, dim=-1)
        context = torch.einsum("nk,nkd->nd", attn, selected_v)

        if q_action_proj.dim() == 3:
            context_for_fusion = context.unsqueeze(1).expand(-1, q_action_proj.shape[1], -1)
        else:
            context_for_fusion = context

        gate = torch.sigmoid(self.gate_proj(torch.cat([q_action_proj, context_for_fusion], dim=-1)))
        q_focused = q_action_proj + self.alpha * gate * context_for_fusion

        with torch.no_grad():
            attn_entropy = -(attn * attn.clamp_min(1e-12).log()).sum(dim=-1).mean()
            self.last_stats = {
                "topk_score_mean": top_scores.mean().detach(),
                "attn_entropy": attn_entropy.detach(),
                "context_norm": context.norm(dim=-1).mean().detach(),
                "q_action_norm": q_action_proj.norm(dim=-1).mean().detach(),
                "gate_mean": gate.mean().detach(),
            }

        return q_focused.to(original_dtype)
