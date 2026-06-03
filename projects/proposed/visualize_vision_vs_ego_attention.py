#!/usr/bin/env python3
"""Analyse how much the SpaceDrive+ VLM attends to Vision vs Ego tokens
during waypoint prediction using gradient-based input attribution.

For each sample the script:
  1. Runs a training-style forward pass (with ground-truth waypoints).
  2. Back-propagates from `loss_pos` (waypoint loss) to the fused
     input embeddings at the first transformer decoder layer.
  3. Measures the L2 gradient magnitude at every token position and
     partitions it into **Vision**, **Ego**, and **Text** groups.

Results are saved as a JSON summary and a bar-chart PNG.

Usage
-----
    python projects/proposed/visualize_vision_vs_ego_attention.py \
        --config  projects/configs/spacedrive/spacedrive_plus_qwen.py \
        --checkpoint workspace/spacedrive_plus_qwen_klal/latest.pth \
        --num-samples 20 \
        --out-dir workspace/attention_analysis/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mmcv  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import DataContainer as DC  # noqa: E402
from mmcv.parallel import MMDataParallel, collate  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from projects.mmdet3d_plugin.datasets.utils.constants import (  # noqa: E402
    IMAGE_TOKEN_INDEX,
    POS_INDICATOR_TOKEN_INDEX,
    VISION_END_TOKEN_INDEX,
)


# ======================================================================
# Helpers
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",
                    default="projects/configs/spacedrive/spacedrive_plus_qwen.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--out-dir", default="workspace/attention_analysis/")
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args()


def unwrap_dc(data, device: torch.device):
    """Recursively unwrap mmcv DataContainer and move tensors to *device*."""
    if isinstance(data, DC):
        inner = data.data
        if isinstance(inner, list):
            inner = inner[0]
        if isinstance(inner, torch.Tensor) and not data.cpu_only:
            inner = inner.to(device)
        return inner
    if isinstance(data, dict):
        return {k: unwrap_dc(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(unwrap_dc(v, device) for v in data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def classify_tokens(input_ids: torch.Tensor):
    """Return bool masks (vision, ego, text) for a 1-D input_ids tensor.

    Token layout after ego insertion::

        ... VISION_START [IMAGE_TOKEN×N_vis] VISION_END
        [IMAGE_TOKEN]                       ← ego feature
        [POS_IND IMAGE_TOKEN] × ego_len     ← ego PE
        ... text / waypoint tokens ...
    """
    seq_len = input_ids.shape[0]
    vision = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    ego = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)

    ve_pos = (input_ids == VISION_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(ve_pos) == 0:
        vision[:] = input_ids == IMAGE_TOKEN_INDEX
        return vision, ego, ~vision

    last_ve = ve_pos[-1].item()

    for i in range(seq_len):
        if i <= last_ve and input_ids[i].item() == IMAGE_TOKEN_INDEX:
            vision[i] = True

    i = last_ve + 1
    while i < seq_len:
        tok = input_ids[i].item()
        if tok in (IMAGE_TOKEN_INDEX, POS_INDICATOR_TOKEN_INDEX):
            ego[i] = True
            i += 1
        else:
            break

    text = ~(vision | ego)
    return vision, ego, text


def find_module_by_cls(model, cls_substr: str):
    """Return the first module whose class name contains *cls_substr*."""
    for _name, mod in model.named_modules():
        if cls_substr in type(mod).__name__:
            return mod
    return None


def find_decoder_layers(model):
    """Return an ordered list of transformer decoder layers."""
    layers = []
    for name, mod in model.named_modules():
        if "DecoderLayer" in type(mod).__name__:
            layers.append((name, mod))
    return layers


# ======================================================================
# Hooks
# ======================================================================

class AttributionCapture:
    """Captures the fused embeddings entering the first decoder layer and
    the ``input_ids`` entering the Qwen VLM backbone so we can compute
    per-token gradient attribution after the backward pass.
    """

    def __init__(self):
        self.fused_embeds: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self._handles: list = []

    # -- decoder-layer pre-hook --
    def _layer_pre_hook(self, _module, args):
        h = args[0] if args else None
        if h is not None and isinstance(h, torch.Tensor):
            h.requires_grad_(True)
            h.retain_grad()
            self.fused_embeds = h

    # -- VLM model pre-hook (captures input_ids) --
    def _vlm_pre_hook(self, _module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None and isinstance(ids, torch.Tensor):
            self.input_ids = ids.detach().clone()

    def register(self, model):
        # 1. Hook first decoder layer
        dec_layers = find_decoder_layers(model)
        if not dec_layers:
            raise RuntimeError("Cannot find decoder layers in the model.")
        name0, layer0 = dec_layers[0]
        print(f"[Hook] first decoder layer: {name0}")
        self._handles.append(
            layer0.register_forward_pre_hook(self._layer_pre_hook)
        )
        # 2. Hook CustomQwen2_5_VLModel to capture input_ids after ego insertion
        vlm_model = find_module_by_cls(model, "CustomQwen2_5_VLModel")
        if vlm_model is None:
            vlm_model = find_module_by_cls(model, "CustomQwen3VLModel")
        if vlm_model is not None:
            print(f"[Hook] VLM backbone: {type(vlm_model).__name__}")
            self._handles.append(
                vlm_model.register_forward_pre_hook(
                    self._vlm_pre_hook, with_kwargs=True
                )
            )
        else:
            print("[Warning] Could not find VLM backbone module for input_ids hook")
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self):
        self.fused_embeds = None
        self.input_ids = None


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # ---- config ----
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "custom_imports"):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    # ---- build model (Qwen loads onto current CUDA device) ----
    print("Building model …")
    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()

    # ---- load checkpoint ----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing keys : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    # Disable gradient checkpointing so gradients flow cleanly
    for mod in model.modules():
        if hasattr(mod, "gradient_checkpointing_disable"):
            mod.gradient_checkpointing_disable()

    model.eval()

    # Wrap with MMDataParallel for automatic DataContainer scatter
    model_dp = MMDataParallel(model, device_ids=[args.gpu_id])

    # ---- dataset & loader ----
    print("Building dataset …")
    dataset = build_dataset(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate, samples_per_gpu=1),
    )

    # ---- hooks ----
    # Register on the unwrapped model (model_dp.module)
    cap = AttributionCapture().register(model)

    # ---- iterate samples ----
    records: list[dict] = []
    num_done = 0

    for batch_idx, raw_batch in enumerate(loader):
        if num_done >= args.num_samples:
            break

        cap.reset()
        model.zero_grad(set_to_none=True)

        try:
            with torch.enable_grad():
                losses = model_dp(return_loss=True, **raw_batch)

                target = losses.get("loss_pos")
                if target is None or not target.requires_grad:
                    total = sum(
                        v for k, v in losses.items()
                        if isinstance(v, torch.Tensor) and v.requires_grad and "loss" in k
                    )
                    target = total

                target.backward()
        except Exception as e:
            print(f"  [skip] sample {batch_idx}: {e}")
            continue

        has_gt = losses.get("loss_pos")
        if has_gt is not None and not has_gt.requires_grad:
            continue

        if cap.fused_embeds is None or cap.fused_embeds.grad is None:
            print(f"  [skip] sample {batch_idx}: no gradient captured")
            continue

        if cap.input_ids is None:
            print(f"  [skip] sample {batch_idx}: input_ids not captured")
            continue

        # ---- compute attribution ----
        grad_norm = cap.fused_embeds.grad.float().norm(dim=-1)[0]  # (seq_len,)
        ids = cap.input_ids[0]  # (seq_len,)

        vis_mask, ego_mask, txt_mask = classify_tokens(ids)
        total_attr = grad_norm.sum().item()
        if total_attr < 1e-12:
            continue

        vis_attr = grad_norm[vis_mask].sum().item()
        ego_attr = grad_norm[ego_mask].sum().item()
        txt_attr = grad_norm[txt_mask].sum().item()

        rec = {
            "sample_idx": batch_idx,
            "seq_len": int(ids.shape[0]),
            "n_vision": int(vis_mask.sum()),
            "n_ego": int(ego_mask.sum()),
            "n_text": int(txt_mask.sum()),
            "vision_pct": vis_attr / total_attr * 100,
            "ego_pct": ego_attr / total_attr * 100,
            "text_pct": txt_attr / total_attr * 100,
            "vision_per_token": vis_attr / max(int(vis_mask.sum()), 1),
            "ego_per_token": ego_attr / max(int(ego_mask.sum()), 1),
            "text_per_token": txt_attr / max(int(txt_mask.sum()), 1),
        }
        records.append(rec)
        num_done += 1
        print(
            f"  [{num_done}/{args.num_samples}] idx={batch_idx}  "
            f"Vision={rec['vision_pct']:.1f}%  Ego={rec['ego_pct']:.1f}%  "
            f"Text={rec['text_pct']:.1f}%"
        )

    cap.remove()

    if not records:
        print("No valid samples processed.")
        return

    # ---- aggregate ----
    avg = {
        "vision_pct": np.mean([r["vision_pct"] for r in records]),
        "ego_pct": np.mean([r["ego_pct"] for r in records]),
        "text_pct": np.mean([r["text_pct"] for r in records]),
        "vision_per_token": np.mean([r["vision_per_token"] for r in records]),
        "ego_per_token": np.mean([r["ego_per_token"] for r in records]),
        "text_per_token": np.mean([r["text_per_token"] for r in records]),
    }
    summary = {"avg": avg, "samples": records}

    json_path = out_dir / "attribution_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Total attribution %
    labels = ["Vision", "Ego", "Text"]
    vals = [avg["vision_pct"], avg["ego_pct"], avg["text_pct"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = axes[0].bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Attribution (%)")
    axes[0].set_title("Total Gradient Attribution by Token Type")
    axes[0].set_ylim(0, max(vals) * 1.2)

    # (b) Per-token attribution (normalised)
    per_tok = [avg["vision_per_token"], avg["ego_per_token"], avg["text_per_token"]]
    bars2 = axes[1].bar(labels, per_tok, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars2, per_tok):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Mean |∇| per token")
    axes[1].set_title("Per-Token Attribution (normalised by count)")

    # (c) Per-sample stacked bar (bottom strip)
    plt.tight_layout()
    plot_path = out_dir / "vision_vs_ego_attribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    # Per-sample breakdown plot
    if len(records) > 1:
        fig2, ax2 = plt.subplots(figsize=(max(8, len(records) * 0.6), 5))
        x = np.arange(len(records))
        vis_vals = [r["vision_pct"] for r in records]
        ego_vals = [r["ego_pct"] for r in records]
        txt_vals = [r["text_pct"] for r in records]
        ax2.bar(x, vis_vals, label="Vision", color=colors[0])
        ax2.bar(x, ego_vals, bottom=vis_vals, label="Ego", color=colors[1])
        ax2.bar(x, txt_vals,
                bottom=[v + e for v, e in zip(vis_vals, ego_vals)],
                label="Text", color=colors[2])
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Attribution (%)")
        ax2.set_title("Per-Sample Attribution Breakdown")
        ax2.legend(loc="upper right")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(r["sample_idx"]) for r in records],
                            rotation=45, fontsize=8)
        plt.tight_layout()
        fig2.savefig(out_dir / "per_sample_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved plot: {out_dir / 'per_sample_breakdown.png'}")

    # ---- print summary ----
    print("\n" + "=" * 50)
    print("         ATTRIBUTION SUMMARY")
    print("=" * 50)
    print(f"  Samples analysed : {len(records)}")
    print(f"  Vision (total %) : {avg['vision_pct']:.1f}%")
    print(f"  Ego    (total %) : {avg['ego_pct']:.1f}%")
    print(f"  Text   (total %) : {avg['text_pct']:.1f}%")
    print(f"  Vision per-token : {avg['vision_per_token']:.6f}")
    print(f"  Ego    per-token : {avg['ego_per_token']:.6f}")
    print(f"  Text   per-token : {avg['text_per_token']:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
