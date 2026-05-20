#!/usr/bin/env python3
"""Experiment-only SpaceDrive wrapper with KL Attention Loss.

This module registers a new detector type, ``SpaceDriveKLAL``, without editing
the original SpaceDrive implementation.  It reuses SpaceDrive.forward_train_vlm
and temporarily wraps the LM head forward call to:

1. request ``output_attentions=True``;
2. capture the exact ``input_ids`` after SpaceDrive inserts ego/PE tokens;
3. add ``loss_klal`` to the returned loss dict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.models.vlm.spacedrive import SpaceDrive

from .klal_loss import KLALGTAttentionLoader, KLAttentionLoss


IGNORE_INDEX = -100


def _coerce_gt_attention_map(gt_attention_map: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if hasattr(gt_attention_map, "data"):
        gt_attention_map = gt_attention_map.data
    if isinstance(gt_attention_map, (list, tuple)):
        values = []
        for item in gt_attention_map:
            if hasattr(item, "data"):
                item = item.data
            if torch.is_tensor(item):
                values.append(item.flatten())
            else:
                values.append(torch.as_tensor(item).flatten())
        gt_attention_map = torch.stack(values, dim=0)
    elif not torch.is_tensor(gt_attention_map):
        gt_attention_map = torch.as_tensor(gt_attention_map)
    gt_attention_map = gt_attention_map.float()
    if gt_attention_map.dim() == 1:
        gt_attention_map = gt_attention_map.unsqueeze(0)
    return gt_attention_map.to(device=device, dtype=dtype)


def _sample_tokens_from_img_metas(img_metas: Sequence[Mapping]) -> list[str]:
    tokens = []
    for meta in img_metas:
        token = meta.get("sample_idx") or meta.get("token")
        if token is None:
            raise KeyError("img_metas entry lacks sample_idx/token for KLAL GT loading.")
        tokens.append(str(token))
    return tokens


@DETECTORS.register_module()
class SpaceDriveKLAL(SpaceDrive):
    """SpaceDrive variant that adds KLAL as an auxiliary loss."""

    def __init__(
        self,
        *args,
        use_klal: bool = False,
        klal_gt_dir: str | None = None,
        klal_lambda: float = 0.1,
        klal_layers: Sequence[int] | int | str | None = "all",
        klal_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_klal = use_klal
        self.klal_lambda = klal_lambda
        self.klal_layers = klal_layers
        self.klal_eps = klal_eps
        self.klal_gt_dir = klal_gt_dir
        self.klal_gt_loader = KLALGTAttentionLoader(klal_gt_dir, eps=klal_eps) if klal_gt_dir else None
        self.klal_loss_fn = KLAttentionLoss(
            klal_lambda=klal_lambda,
            layers=klal_layers,
            eps=klal_eps,
        )

    def _load_klal_gt_attention(
        self,
        img_metas,
        captured_input_ids: torch.Tensor,
        data: Mapping[str, Any],
    ) -> torch.Tensor:
        if "gt_attention_map" in data and data["gt_attention_map"] is not None:
            return _coerce_gt_attention_map(
                data["gt_attention_map"],
                captured_input_ids.device,
                torch.float32,
            )

        if self.klal_gt_loader is None:
            raise ValueError(
                "KLAL is enabled but neither data['gt_attention_map'] nor klal_gt_dir was provided."
            )
        tokens = _sample_tokens_from_img_metas(img_metas)
        return self.klal_gt_loader.load_batch(
            tokens,
            device=captured_input_ids.device,
            dtype=torch.float32,
        )

    def forward_train_vlm(
        self,
        img_metas,
        input_ids,
        vlm_labels,
        vlm_attn_mask,
        pixel_values,
        image_grid_thw,
        coords_pos_tensor,
        **data,
    ):
        if not self.use_klal:
            return super().forward_train_vlm(
                img_metas,
                input_ids,
                vlm_labels,
                vlm_attn_mask,
                pixel_values,
                image_grid_thw,
                coords_pos_tensor,
                **data,
            )

        captured: dict[str, Any] = {}
        original_forward = self.lm_head.forward

        def forward_with_klal(module_self, *args, **kwargs):
            kwargs["output_attentions"] = True
            output = original_forward(*args, **kwargs)
            captured["lm_output"] = output
            captured["input_ids"] = kwargs.get("input_ids")
            return output

        self.lm_head.forward = MethodType(forward_with_klal, self.lm_head)
        try:
            losses = super().forward_train_vlm(
                img_metas,
                input_ids,
                vlm_labels,
                vlm_attn_mask,
                pixel_values,
                image_grid_thw,
                coords_pos_tensor,
                **data,
            )
        finally:
            self.lm_head.forward = original_forward

        lm_output = captured.get("lm_output")
        captured_input_ids = captured.get("input_ids")
        if lm_output is None or captured_input_ids is None:
            raise RuntimeError("KLAL failed to capture lm_head output/input_ids.")

        gt_attention_map = self._load_klal_gt_attention(img_metas, captured_input_ids, data)
        labels = getattr(lm_output, "labels", None)
        query_allowed_mask = labels.ne(IGNORE_INDEX) if torch.is_tensor(labels) else None
        has_gt_planning = data.get("has_gt_planning")
        if torch.is_tensor(has_gt_planning) and not bool(has_gt_planning.any()):
            zero = captured_input_ids.new_tensor(0.0, dtype=torch.float32)
            losses["loss_klal"] = zero
            losses["klal_raw"] = zero.detach()
            losses["klal_visual_tokens"] = zero.detach()
            return losses

        loss_klal, klal_info = self.klal_loss_fn(
            attentions=lm_output.attentions,
            input_ids=captured_input_ids,
            gt_attention_map=gt_attention_map,
            query_allowed_mask=query_allowed_mask,
        )
        losses["loss_klal"] = loss_klal
        losses["klal_raw"] = klal_info["klal_raw"]
        losses["klal_visual_tokens"] = klal_info["klal_visual_token_count"].float().mean()
        return losses
