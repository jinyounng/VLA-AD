#!/usr/bin/env python3
"""Analyze trajectory-token attention to vision tokens vs ego-status tokens.

This script runs SpaceDrive+ (Qwen) on sampled val items, then:
1) generates trajectories,
2) re-runs a teacher-forced forward on prompt+generated tokens with output_attentions=True,
3) compares trajectory-token attention toward
   - vision token positions
   - ego-status token positions
4) reports/plots straight vs turn statistics.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer

_REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(_REPO_ROOT))

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.utils.constants import (
    IMAGE_TOKEN_INDEX,
    POS_INDICATOR_TOKEN_INDEX,
    VISION_END_TOKEN_INDEX,
    VISION_START_TOKEN_INDEX,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ann-file", default=None, help="val pkl used for straight/turn classification")
    p.add_argument("--output-dir", default="workspace/attention_analysis_plus")
    p.add_argument("--straight-samples", type=int, default=100)
    p.add_argument("--turn-samples", type=int, default=100)
    p.add_argument("--first-samples", type=int, default=0,
                   help="Analyze the first N dataloader samples without straight/turn filtering.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _unwrap(x):
    if isinstance(x, DataContainer):
        return x.data[0]
    return x


def _first_item(x):
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    return x


def _to_device_tensor(batch, key, device):
    x = _unwrap(batch[key])
    x = _first_item(x)
    if not torch.is_tensor(x):
        raise TypeError(f"{key} is not a tensor after unwrapping: {type(x)}")
    return x.to(device)


def _ensure_batched_tensor(x, dims_without_batch):
    if x.dim() == dims_without_batch:
        return x.unsqueeze(0)
    return x


def _get_image_feature_model(lm_head):
    candidates = [
        lm_head,
        getattr(lm_head, "model", None),
        getattr(getattr(lm_head, "base_model", None), "model", None),
        getattr(getattr(getattr(lm_head, "base_model", None), "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "get_image_features"):
            return candidate
    raise AttributeError("Could not find a module with get_image_features under model.lm_head")


def _extract_token(batch):
    metas = _unwrap(batch["img_metas"])
    m0 = _first_item(metas)
    if not isinstance(m0, dict):
        raise TypeError(f"img_metas first item is not dict: {type(m0)}")
    return str(m0["sample_idx"])


def _find_last_vision_anchor(input_ids_1d, model):
    """Find insertion anchor right after the vision token span."""
    ve = (input_ids_1d == VISION_END_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if ve.numel() > 0:
        return int(ve.max().item())
    img_tok_id = int(model.lm_head.config.image_token_id)
    img = (input_ids_1d == img_tok_id).nonzero(as_tuple=False).squeeze(-1)
    if img.numel() > 0:
        return int(img.max().item())
    # Fallback: right after BOS to avoid empty reduction crashes.
    return 0


def classify_tokens(ann_file, n_straight, n_turn, seed):
    with open(ann_file, "rb") as f:
        data = torch.load(f, map_location="cpu") if ann_file.endswith(".pt") else None
    if data is None:
        import pickle
        with open(ann_file, "rb") as f:
            data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data

    straight, turn = [], []
    for info in infos:
        token = str(info["token"])
        desc = str(info.get("gt_planning_command_desc", "")).lower()
        is_turn = any(k in desc for k in ["turn", "left", "right", "u-turn", "uturn"])
        is_straight = ("straight" in desc) and not is_turn
        if is_turn:
            turn.append(token)
        elif is_straight:
            straight.append(token)

    rng = random.Random(seed)
    straight = rng.sample(straight, min(n_straight, len(straight)))
    turn = rng.sample(turn, min(n_turn, len(turn)))
    return set(straight), set(turn)


@torch.no_grad()
def prepare_prefill_inputs(model, sample):
    """Mirror SpaceDrive.test_generation_pts prefill preparation."""
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    pixel_values = sample["pixel_values"]
    image_grid_thw = sample["image_grid_thw"]
    data = sample["data"]
    img = sample["img"]
    img_metas = sample["img_metas"]

    B = pixel_values.shape[0]
    pos_embed = None
    coords3d = None
    if model.vis_3d_pos:
        depth = model.depth_prediction(img, data["intrinsics"], img_metas=img_metas)
        location = model.prepare_location(image_grid_thw, pixel_values)
        pos_embed, coords3d = model.position_embeding(data, location, img_metas, depth, image_grid_thw)

    ego_feature = None
    if model.ego_status is not None:
        rec_can_bus = torch.cat([data["command"].unsqueeze(-1), data["can_bus"]], dim=-1)
        ego_feature = torch.empty(B, 0, model.llm_hidden_dim, device=rec_can_bus.device)

        if "feature" in model.ego_status:
            ego_mlp_input = torch.cat(
                [
                    model.memory_canbus.reshape(B, -1),
                    rec_can_bus.reshape(B, -1),
                    model.memory_egopose.reshape(B, -1, 16).reshape(B, -1),
                ],
                dim=-1,
            )
            ego_token = model.ego_status_mlp(ego_mlp_input).unsqueeze(1)
            ego_feature = torch.cat([ego_feature, ego_token], dim=1)

            last_vision_end = _find_last_vision_anchor(input_ids[0], model)
            ins = torch.tensor(
                [VISION_START_TOKEN_INDEX, IMAGE_TOKEN_INDEX, VISION_END_TOKEN_INDEX],
                device=input_ids.device,
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids[:, : last_vision_end + 1], ins, input_ids[:, last_vision_end + 1 :]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask[:, : last_vision_end + 1], torch.ones_like(ins), attention_mask[:, last_vision_end + 1 :]],
                dim=-1,
            )

        if "PE" in model.ego_status:
            past_xyz = model.memory_egopose[:, : model.ego_status_len, :3, 3]
            encoded = model.position_encoder(past_xyz.reshape(B, -1, 3)).reshape(B, model.ego_status_len, -1)
            ego_feature = torch.cat([ego_feature, encoded], dim=1)

            last_vision_end = _find_last_vision_anchor(input_ids[0], model)
            len_past = encoded.shape[1]
            ins = torch.tensor(
                [POS_INDICATOR_TOKEN_INDEX, IMAGE_TOKEN_INDEX] * len_past,
                device=input_ids.device,
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids[:, : last_vision_end + 1], ins, input_ids[:, last_vision_end + 1 :]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask[:, : last_vision_end + 1], torch.ones_like(ins), attention_mask[:, last_vision_end + 1 :]],
                dim=-1,
            )

    return input_ids, attention_mask, pixel_values, image_grid_thw, pos_embed, ego_feature, coords3d


@torch.no_grad()
def generate_then_collect_attn(model, prepared, max_new_tokens):
    input_ids, attn_mask, pixel_values, image_grid_thw, pos_embed, ego_feature, coords3d = prepared
    prefill_len = input_ids.shape[1]

    gen = model.lm_head.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attn_mask,
        pos_emb=pos_embed,
        planning_only=model.planning_only if model.io_3d_pos else False,
        single_coords_only=model.single_coords_only if model.io_3d_pos else False,
        ego_feature=ego_feature if (ego_feature is not None and ego_feature.numel() > 0) else None,
        enable_pe_input=model.enable_pe_input if model.io_3d_pos else False,
        pos_index=coords3d if model.use_rope else None,
        output_hidden_states=False,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **model._extra_lm_forward_kwargs(input_ids),
    )

    full_seq = gen.sequences[0]
    generated = full_seq[prefill_len:]
    traj_rel = (generated == POS_INDICATOR_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if traj_rel.numel() == 0:
        return None
    traj_abs = (traj_rel + prefill_len).tolist()

    full_input_ids = full_seq.unsqueeze(0)
    full_attn = torch.ones_like(full_input_ids, device=full_input_ids.device)
    fw = model.lm_head(
        input_ids=full_input_ids,
        attention_mask=full_attn,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pos_emb=pos_embed,
        planning_only=model.planning_only if model.io_3d_pos else False,
        single_coords_only=model.single_coords_only if model.io_3d_pos else False,
        ego_feature=ego_feature if (ego_feature is not None and ego_feature.numel() > 0) else None,
        enable_pe_input=model.enable_pe_input if model.io_3d_pos else False,
        pos_index=coords3d if model.use_rope else None,
        output_attentions=True,
        return_dict=True,
        **model._extra_lm_forward_kwargs(full_input_ids),
    )

    # Token groups in prefill section
    image_token_positions = (input_ids[0] == model.lm_head.config.image_token_id).nonzero(as_tuple=False).squeeze(-1)
    image_feature_model = _get_image_feature_model(model.lm_head)
    n_image_features = image_feature_model.get_image_features(
        pixel_values.reshape(-1, pixel_values.shape[-1]),
        image_grid_thw.reshape(-1, image_grid_thw.shape[-1]),
    ).shape[0]
    vision_token_indices = image_token_positions[:n_image_features].tolist()
    ego_token_indices = image_token_positions[n_image_features:].tolist()
    if len(ego_token_indices) == 0:
        return None

    vision_per_layer = []
    ego_per_layer = []
    vision_mass_per_layer = []
    ego_mass_per_layer = []
    for layer_attn in fw.attentions:
        # layer_attn: (B, heads, q_len, kv_len)
        a = layer_attn[0]  # (heads, q_len, kv_len)
        qidx = torch.tensor(traj_abs, device=a.device, dtype=torch.long)
        k_vision = torch.tensor(vision_token_indices, device=a.device, dtype=torch.long)
        k_ego = torch.tensor(ego_token_indices, device=a.device, dtype=torch.long)

        vision_attn = a[:, qidx][:, :, k_vision]
        ego_attn = a[:, qidx][:, :, k_ego]

        # Mean is per-token attention; mass is total attention to the token group.
        vision_per_layer.append(vision_attn.mean().item())
        ego_per_layer.append(ego_attn.mean().item())
        vision_mass_per_layer.append(vision_attn.sum(dim=-1).mean().item())
        ego_mass_per_layer.append(ego_attn.sum(dim=-1).mean().item())

    return {
        "vision_layer_mean": vision_per_layer,
        "ego_layer_mean": ego_per_layer,
        "vision_layer_mass": vision_mass_per_layer,
        "ego_layer_mass": ego_mass_per_layer,
        "trajectory_token_indices": traj_abs,
        "vision_token_count": len(vision_token_indices),
        "ego_token_count": len(ego_token_indices),
    }


def make_plots(straight_v, straight_e, turn_v, turn_e, straight_vm, straight_em, turn_vm, turn_em, out_png):
    layers = np.arange(len(straight_v))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(layers, straight_v, label="vision", color="tab:blue")
    axes[0, 0].plot(layers, straight_e, label="ego", color="tab:red")
    axes[0, 0].set_title("Straight: per-token mean")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Attention weight")
    axes[0, 0].legend()

    axes[0, 1].plot(layers, turn_v, label="vision", color="tab:blue")
    axes[0, 1].plot(layers, turn_e, label="ego", color="tab:red")
    axes[0, 1].set_title("Turn: per-token mean")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Attention weight")
    axes[0, 1].legend()

    bars = [
        float(np.mean(straight_v)),
        float(np.mean(turn_v)),
        float(np.mean(straight_e)),
        float(np.mean(turn_e)),
    ]
    axes[0, 2].bar(
        ["vision_straight", "vision_turn", "ego_straight", "ego_turn"],
        bars,
        color=["tab:blue", "tab:blue", "tab:red", "tab:red"],
    )
    axes[0, 2].set_title("Overall per-token mean")
    axes[0, 2].tick_params(axis="x", rotation=20)

    axes[1, 0].plot(layers, straight_vm, label="vision", color="tab:blue")
    axes[1, 0].plot(layers, straight_em, label="ego", color="tab:red")
    axes[1, 0].set_title("Straight: group mass")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Attention mass")
    axes[1, 0].legend()

    axes[1, 1].plot(layers, turn_vm, label="vision", color="tab:blue")
    axes[1, 1].plot(layers, turn_em, label="ego", color="tab:red")
    axes[1, 1].set_title("Turn: group mass")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Attention mass")
    axes[1, 1].legend()

    mass_bars = [
        float(np.mean(straight_vm)),
        float(np.mean(turn_vm)),
        float(np.mean(straight_em)),
        float(np.mean(turn_em)),
    ]
    axes[1, 2].bar(
        ["vision_straight", "vision_turn", "ego_straight", "ego_turn"],
        mass_bars,
        color=["tab:blue", "tab:blue", "tab:red", "tab:red"],
    )
    axes[1, 2].set_title("Overall group mass")
    axes[1, 2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def make_all_plot(all_v, all_e, all_vm, all_em, out_png):
    layers = np.arange(len(all_v))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(layers, all_v, label="vision", color="tab:blue")
    axes[0].plot(layers, all_e, label="ego", color="tab:red")
    axes[0].set_title("All: per-token mean")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Attention weight")
    axes[0].legend()

    axes[1].plot(layers, all_vm, label="vision", color="tab:blue")
    axes[1].plot(layers, all_em, label="ego", color="tab:red")
    axes[1].set_title("All: group mass")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Attention mass")
    axes[1].legend()

    axes[2].bar(
        ["vision_mean", "ego_mean", "vision_mass", "ego_mass"],
        [float(np.mean(all_v)), float(np.mean(all_e)), float(np.mean(all_vm)), float(np.mean(all_em))],
        color=["tab:blue", "tab:red", "tab:blue", "tab:red"],
    )
    axes[2].set_title("Overall")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    first_mode = args.first_samples > 0
    target_map = {}
    if not first_mode:
        if args.ann_file is None:
            raise ValueError("--ann-file is required unless --first-samples is set.")
        straight_pool, turn_pool = classify_tokens(
            args.ann_file, args.straight_samples, args.turn_samples, args.seed
        )
        for t in straight_pool:
            target_map[t] = "straight"
        for t in turn_pool:
            target_map[t] = "turn"
    else:
        straight_pool, turn_pool = set(), set()

    per_class = defaultdict(list)
    detail_rows = []

    for batch in data_loader:
        token = _extract_token(batch)
        if first_mode:
            cls = "all"
        else:
            if token not in target_map:
                continue
            cls = target_map[token]
            if cls == "straight" and len(per_class["straight"]) >= len(straight_pool):
                continue
            if cls == "turn" and len(per_class["turn"]) >= len(turn_pool):
                continue

        # build sample tensors
        sample = {
            "img": _ensure_batched_tensor(_to_device_tensor(batch, "img", device), 4),
            "input_ids": _ensure_batched_tensor(_to_device_tensor(batch, "input_ids", device), 1),
            "pixel_values": _ensure_batched_tensor(_to_device_tensor(batch, "pixel_values", device), 2),
            "image_grid_thw": _ensure_batched_tensor(_to_device_tensor(batch, "image_grid_thw", device), 2),
            "attention_mask": _ensure_batched_tensor(_to_device_tensor(batch, "attention_mask", device), 1),
            "img_metas": _unwrap(batch["img_metas"]),
            "data": {
                "lidar2img": _ensure_batched_tensor(_to_device_tensor(batch, "lidar2img", device), 3),
                "intrinsics": _ensure_batched_tensor(_to_device_tensor(batch, "intrinsics", device), 3),
                "extrinsics": _ensure_batched_tensor(_to_device_tensor(batch, "extrinsics", device), 3),
                "timestamp": _ensure_batched_tensor(_to_device_tensor(batch, "timestamp", device), 0),
                "img_timestamp": _ensure_batched_tensor(_to_device_tensor(batch, "img_timestamp", device), 1),
                "ego_pose": _ensure_batched_tensor(_to_device_tensor(batch, "ego_pose", device), 2),
                "ego_pose_inv": _ensure_batched_tensor(_to_device_tensor(batch, "ego_pose_inv", device), 2),
                "command": _ensure_batched_tensor(_to_device_tensor(batch, "command", device), 0),
                "can_bus": _ensure_batched_tensor(_to_device_tensor(batch, "can_bus", device), 1),
            },
        }

        # independent sample analysis (no temporal carry-over)
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        if getattr(model, "ego_status", None) is not None:
            sample["data"]["intrinsics"] = sample["data"]["intrinsics"].float()
            model.pre_update_memory(sample["data"])

        prepared = prepare_prefill_inputs(model, sample)
        out = generate_then_collect_attn(model, prepared, args.max_new_tokens)
        if out is None:
            continue
        per_class[cls].append(out)
        detail_rows.append({"token": token, "class": cls, **out})

        if first_mode:
            if len(per_class["all"]) >= args.first_samples:
                break
        elif len(per_class["straight"]) >= len(straight_pool) and len(per_class["turn"]) >= len(turn_pool):
            break

    if first_mode and len(per_class["all"]) == 0:
        raise RuntimeError("No samples were successfully analyzed in --first-samples mode.")
    if not first_mode and (len(per_class["straight"]) == 0 or len(per_class["turn"]) == 0):
        raise RuntimeError(
            f"Insufficient analyzed samples. straight={len(per_class['straight'])}, turn={len(per_class['turn'])}"
        )

    def agg(class_rows, key):
        arr = np.array([r[key] for r in class_rows], dtype=np.float32)
        return arr.mean(axis=0).tolist()

    out_png = os.path.join(args.output_dir, "attention_analysis.png")
    out_json = os.path.join(args.output_dir, "attention_results.json")

    if first_mode:
        all_v = agg(per_class["all"], "vision_layer_mean")
        all_e = agg(per_class["all"], "ego_layer_mean")
        all_vm = agg(per_class["all"], "vision_layer_mass")
        all_em = agg(per_class["all"], "ego_layer_mass")
        make_all_plot(all_v, all_e, all_vm, all_em, out_png)

        payload = {
            "mode": "first_samples",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "ann_file": args.ann_file,
            "counts": {
                "first_samples_target": args.first_samples,
                "all_used": len(per_class["all"]),
            },
            "layerwise_mean": {
                "all": {"vision": all_v, "ego": all_e},
            },
            "layerwise_mass": {
                "all": {"vision": all_vm, "ego": all_em},
            },
            "overall_mean": {
                "vision_all": float(np.mean(all_v)),
                "ego_all": float(np.mean(all_e)),
            },
            "overall_mass": {
                "vision_all": float(np.mean(all_vm)),
                "ego_all": float(np.mean(all_em)),
            },
            "sample_details_first_20": detail_rows[:20],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[done] saved: {out_json}")
        print(f"[done] saved: {out_png}")
        return

    straight_v = agg(per_class["straight"], "vision_layer_mean")
    straight_e = agg(per_class["straight"], "ego_layer_mean")
    turn_v = agg(per_class["turn"], "vision_layer_mean")
    turn_e = agg(per_class["turn"], "ego_layer_mean")
    straight_vm = agg(per_class["straight"], "vision_layer_mass")
    straight_em = agg(per_class["straight"], "ego_layer_mass")
    turn_vm = agg(per_class["turn"], "vision_layer_mass")
    turn_em = agg(per_class["turn"], "ego_layer_mass")

    make_plots(straight_v, straight_e, turn_v, turn_e, straight_vm, straight_em, turn_vm, turn_em, out_png)

    payload = {
        "mode": "straight_turn",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "ann_file": args.ann_file,
        "counts": {
            "straight_target": args.straight_samples,
            "turn_target": args.turn_samples,
            "straight_used": len(per_class["straight"]),
            "turn_used": len(per_class["turn"]),
        },
        "layerwise_mean": {
            "straight": {"vision": straight_v, "ego": straight_e},
            "turn": {"vision": turn_v, "ego": turn_e},
        },
        "layerwise_mass": {
            "straight": {"vision": straight_vm, "ego": straight_em},
            "turn": {"vision": turn_vm, "ego": turn_em},
        },
        "overall_mean": {
            "vision_straight": float(np.mean(straight_v)),
            "vision_turn": float(np.mean(turn_v)),
            "ego_straight": float(np.mean(straight_e)),
            "ego_turn": float(np.mean(turn_e)),
        },
        "overall_mass": {
            "vision_straight": float(np.mean(straight_vm)),
            "vision_turn": float(np.mean(turn_vm)),
            "ego_straight": float(np.mean(straight_em)),
            "ego_turn": float(np.mean(turn_em)),
        },
        "sample_details_first_20": detail_rows[:20],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[done] saved: {out_json}")
    print(f"[done] saved: {out_png}")


if __name__ == "__main__":
    main()
