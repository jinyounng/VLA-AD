#!/usr/bin/env python3
"""Standalone KL Attention Loss utilities for SpaceDrive.

This file intentionally does not modify SpaceDrive model or dataset code.
Use it from an experiment-specific model/config when you want to add KLAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep this module independent from SpaceDrive's model/dataset imports.  The
# values mirror projects/mmdet3d_plugin/datasets/utils/constants.py for Qwen2.5-VL.
IMAGE_TOKEN_INDEX = 151655
VISION_START_TOKEN_INDEX = 151652
VISION_END_TOKEN_INDEX = 151653
POS_INDICATOR_TOKEN_INDEX = 151665


@dataclass(frozen=True)
class KLALTokenSpec:
    image_token_id: int = IMAGE_TOKEN_INDEX
    vision_start_token_id: int = VISION_START_TOKEN_INDEX
    vision_end_token_id: int = VISION_END_TOKEN_INDEX
    pos_indicator_token_id: int = POS_INDICATOR_TOKEN_INDEX


class KLALGTAttentionLoader:
    """Load precomputed per-sample KLAL GT maps.

    Expected file layout:
        <gt_dir>/<sample_token>.pt

    Each file should contain a tensor shaped ``(num_visual_tokens,)``.
    """

    def __init__(self, gt_dir: str | Path, eps: float = 1e-8):
        self.gt_dir = Path(gt_dir)
        self.eps = eps

    def load_one(
        self,
        sample_token: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        path = self.gt_dir / f"{sample_token}.pt"
        if not path.exists():
            raise FileNotFoundError(f"KLAL GT attention map not found: {path}")
        gt = torch.load(path, map_location="cpu")
        if not torch.is_tensor(gt):
            gt = torch.as_tensor(gt)
        gt = gt.float().flatten()
        gt = gt.clamp_min(self.eps)
        gt = gt / gt.sum().clamp_min(self.eps)
        if dtype is not None:
            gt = gt.to(dtype=dtype)
        if device is not None:
            gt = gt.to(device=device)
        return gt

    def load_batch(
        self,
        sample_tokens: Sequence[str],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        maps = [self.load_one(token, device=device, dtype=dtype) for token in sample_tokens]
        return torch.stack(maps, dim=0)

    def load_from_img_metas(
        self,
        img_metas: Sequence[Mapping],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        tokens = [str(meta.get("sample_idx") or meta.get("token")) for meta in img_metas]
        return self.load_batch(tokens, device=device, dtype=dtype)


def _as_layer_indices(num_layers: int, layers: Sequence[int] | int | str | None) -> list[int]:
    if layers is None or layers == "all":
        return list(range(num_layers))
    if isinstance(layers, int):
        layers = [layers]
    out = []
    for layer in layers:
        idx = int(layer)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise IndexError(f"KLAL layer index out of range: {layer} for {num_layers} layers")
        out.append(idx)
    return out


def _last_token_positions(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """Return last occurrence position per sample, or -1 when missing."""
    mask = input_ids.eq(token_id)
    positions = torch.arange(input_ids.size(1), device=input_ids.device).view(1, -1)
    last = torch.where(mask, positions, positions.new_full((1, input_ids.size(1)), -1)).amax(dim=1)
    return last


def _last_token_positions_with_mask(
    input_ids: torch.Tensor,
    token_id: int,
    allowed_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return last token occurrence per sample, optionally constrained by a mask."""
    mask = input_ids.eq(token_id)
    if allowed_mask is not None:
        mask = mask & allowed_mask.to(device=input_ids.device, dtype=torch.bool)
    positions = torch.arange(input_ids.size(1), device=input_ids.device).view(1, -1)
    return torch.where(mask, positions, positions.new_full((1, input_ids.size(1)), -1)).amax(dim=1)


def _visual_token_masks(input_ids: torch.Tensor, token_spec: KLALTokenSpec) -> torch.Tensor:
    """Build a per-sample mask for visual tokens only.

    Prefer IMAGE_TOKEN_INDEX because Qwen processor expands images into visual-token
    positions. Restrict to the first vision_start..vision_end span when those
    wrapper tokens are present, so ego-status IMAGE_TOKEN_INDEX insertions after
    VISION_END are excluded.
    """
    visual_mask = input_ids.eq(token_spec.image_token_id)
    start_pos = _last_or_first_positions(input_ids, token_spec.vision_start_token_id, first=True)
    end_pos = _last_or_first_positions(input_ids, token_spec.vision_end_token_id, first=False)

    seq_pos = torch.arange(input_ids.size(1), device=input_ids.device).view(1, -1)
    has_span = (start_pos >= 0) & (end_pos >= 0) & (end_pos > start_pos)
    span_mask = (seq_pos > start_pos.view(-1, 1)) & (seq_pos < end_pos.view(-1, 1))
    visual_mask = torch.where(has_span.view(-1, 1), visual_mask & span_mask, visual_mask)
    return visual_mask


def _last_or_first_positions(input_ids: torch.Tensor, token_id: int, *, first: bool) -> torch.Tensor:
    mask = input_ids.eq(token_id)
    positions = torch.arange(input_ids.size(1), device=input_ids.device).view(1, -1)
    fill = input_ids.size(1) if first else -1
    reduced = torch.where(mask, positions, positions.new_full((1, input_ids.size(1)), fill))
    out = reduced.amin(dim=1) if first else reduced.amax(dim=1)
    if first:
        out = torch.where(out.eq(input_ids.size(1)), out.new_full(out.shape, -1), out)
    return out


def _stack_visual_attention(
    layer_attention: torch.Tensor,
    query_positions: torch.Tensor,
    visual_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract query-to-visual attention from one layer.

    Args:
        layer_attention: ``(B, heads, seq, seq)``
        query_positions: ``(B,)``
        visual_mask: ``(B, seq)``

    Returns:
        padded_attn: ``(B, max_visual_tokens)``
        valid_mask: ``(B, max_visual_tokens)``
    """
    if layer_attention.dim() != 4:
        raise ValueError(f"Expected attention shape (B,H,S,S), got {tuple(layer_attention.shape)}")
    batch = layer_attention.size(0)
    batch_idx = torch.arange(batch, device=layer_attention.device)
    head_mean = layer_attention[batch_idx, :, query_positions].mean(dim=1)

    visual_counts = visual_mask.sum(dim=1)
    max_tokens = int(visual_counts.max().item())
    if max_tokens <= 0:
        raise ValueError("No visual tokens found in input_ids.")

    padded = head_mean.new_zeros((batch, max_tokens))
    valid = torch.zeros((batch, max_tokens), dtype=torch.bool, device=head_mean.device)
    for b in range(batch):
        values = head_mean[b, visual_mask[b]]
        n = values.numel()
        padded[b, :n] = values
        valid[b, :n] = True
    return padded, valid


def normalize_distribution(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    values = values.float().clamp_min(eps)
    if mask is not None:
        values = values.masked_fill(~mask, 0.0)
    denom = values.sum(dim=-1, keepdim=True).clamp_min(eps)
    return values / denom


class KLAttentionLoss(nn.Module):
    """KL divergence between answer-token attention and GT visual attention maps."""

    def __init__(
        self,
        klal_lambda: float = 0.1,
        layers: Sequence[int] | int | str | None = "all",
        token_spec: KLALTokenSpec | None = None,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.klal_lambda = float(klal_lambda)
        self.layers = layers
        self.token_spec = token_spec or KLALTokenSpec()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        *,
        attentions: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        gt_attention_map: torch.Tensor,
        query_allowed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if attentions is None:
            raise ValueError("attentions is None. Call lm_head/model with output_attentions=True.")
        if len(attentions) == 0:
            raise ValueError("attentions is empty.")
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids shape (B,S), got {tuple(input_ids.shape)}")

        query_pos = _last_token_positions_with_mask(
            input_ids,
            self.token_spec.pos_indicator_token_id,
            query_allowed_mask,
        )
        valid_query = query_pos.ge(0)
        if not bool(valid_query.all()):
            missing = torch.nonzero(~valid_query, as_tuple=False).flatten().tolist()
            raise ValueError(f"Missing POS_INDICATOR token for samples: {missing}")

        visual_mask = _visual_token_masks(input_ids, self.token_spec)
        selected_layers = _as_layer_indices(len(attentions), self.layers)
        layer_losses = []
        layer_attn_probs = []

        for layer_idx in selected_layers:
            attn_values, valid_visual = _stack_visual_attention(
                attentions[layer_idx],
                query_pos.to(attentions[layer_idx].device),
                visual_mask.to(attentions[layer_idx].device),
            )
            gt = gt_attention_map.to(device=attn_values.device, dtype=attn_values.dtype)
            gt = gt.reshape(gt.shape[0], -1)
            if gt.shape != attn_values.shape:
                raise ValueError(
                    f"GT map shape {tuple(gt.shape)} does not match visual attention "
                    f"shape {tuple(attn_values.shape)} at layer {layer_idx}. "
                    "Check grid size/camera order and visual token extraction."
                )

            attn_prob = normalize_distribution(attn_values, valid_visual, self.eps)
            gt_prob = normalize_distribution(gt, valid_visual, self.eps)
            kl_per_sample = F.kl_div(attn_prob.clamp_min(self.eps).log(), gt_prob, reduction="none").sum(dim=-1)
            layer_losses.append(kl_per_sample)
            layer_attn_probs.append(attn_prob.detach())

        kl_per_sample = torch.stack(layer_losses, dim=0).mean(dim=0)
        if self.reduction == "mean":
            loss = kl_per_sample.mean()
        elif self.reduction == "sum":
            loss = kl_per_sample.sum()
        else:
            loss = kl_per_sample

        scaled_loss = loss * self.klal_lambda
        info = {
            "klal_raw": loss.detach() if torch.is_tensor(loss) else torch.as_tensor(loss),
            "klal_scaled": scaled_loss.detach(),
            "klal_per_sample": kl_per_sample.detach(),
            "klal_visual_token_count": visual_mask.sum(dim=1).detach(),
            "klal_query_pos": query_pos.detach(),
            "klal_layers": torch.tensor(selected_layers, device=input_ids.device),
            "klal_mean_attn": torch.stack(layer_attn_probs, dim=0).mean(dim=0),
        }
        return scaled_loss, info


def compute_klal_loss_from_lm_output(
    lm_output,
    *,
    input_ids: torch.Tensor,
    gt_attention_map: torch.Tensor,
    klal_lambda: float = 0.1,
    layers: Sequence[int] | int | str | None = "all",
    token_spec: KLALTokenSpec | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convenience wrapper for SpaceDrive/Qwen output objects."""
    attentions = getattr(lm_output, "attentions", None)
    if attentions is None and isinstance(lm_output, Mapping):
        attentions = lm_output.get("attentions")
    module = KLAttentionLoss(klal_lambda=klal_lambda, layers=layers, token_spec=token_spec, eps=eps)
    return module(attentions=attentions, input_ids=input_ids, gt_attention_map=gt_attention_map)


__all__ = [
    "KLALGTAttentionLoader",
    "KLALTokenSpec",
    "KLAttentionLoss",
    "compute_klal_loss_from_lm_output",
    "normalize_distribution",
]
