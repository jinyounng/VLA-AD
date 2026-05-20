# ------------------------------------------------------------------------
# SpaceDrive attention analysis (test.py-based).
# Measures trajectory-token attention to vision vs ego tokens.
# ------------------------------------------------------------------------
import argparse
import json
import os
import pickle
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, DataContainer
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
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
    parser = argparse.ArgumentParser(description="SpaceDrive+ attention analysis")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument(
        "--ann-file", required=True,
        help="nuScenes val annotation pkl for straight/turn sampling",
    )
    parser.add_argument("--straight-samples", type=int, default=100)
    parser.add_argument("--turn-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--output-dir", default="workspace/attention_vs_ego")
    parser.add_argument(
        "--first-n-samples",
        type=int,
        default=0,
        help="If > 0, ignore straight/turn split and analyze first N successful samples.",
    )
    return parser.parse_args()


# ====================== helpers ======================

def _unwrap(x):
    """Recursively unwrap DataContainer / nested lists to get the raw tensor."""
    if isinstance(x, DataContainer):
        x = x.data
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    return x


def _to_dev(batch, key, device):
    """Extract a tensor from *batch*, add missing batch dim if needed, move to *device*."""
    x = _unwrap(batch[key])
    if not torch.is_tensor(x):
        raise TypeError(f"{key} is not tensor: {type(x)}")
    if key in ("input_ids", "attention_mask") and x.dim() == 1:
        x = x.unsqueeze(0)
    elif key in ("pixel_values", "image_grid_thw") and x.dim() == 2:
        x = x.unsqueeze(0)
    elif key in ("command",) and x.dim() == 0:
        x = x.unsqueeze(0)
    elif key in ("can_bus",) and x.dim() == 1:
        x = x.unsqueeze(0)
    return x.to(device)


def _sample_token(batch):
    """Return (sample_idx_str, [meta_dict]) from a data-loader batch."""
    metas = _unwrap(batch["img_metas"])
    if not isinstance(metas, dict):
        raise TypeError(f"img_metas type: {type(metas)}")
    return str(metas["sample_idx"]), [metas]


def _classify_tokens(ann_file, n_straight, n_turn, seed):
    """Split annotation tokens into *straight* and *turn* sets."""
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    straight, turn = [], []
    for info in infos:
        token = str(info["token"])
        desc = str(info.get("gt_planning_command_desc", "")).lower()
        is_turn = any(k in desc for k in ("left", "right", "turn", "u-turn", "uturn"))
        if is_turn:
            turn.append(token)
        elif "straight" in desc:
            straight.append(token)
    rng = random.Random(seed)
    return (
        set(rng.sample(straight, min(len(straight), n_straight))),
        set(rng.sample(turn, min(len(turn), n_turn))),
    )


def _last_vision_anchor(input_ids_1d, lm_head):
    """Position of the last vision/image-pad token in *input_ids_1d*.

    Used to determine where ego tokens should be inserted.
    """
    ve = (input_ids_1d == VISION_END_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if ve.numel() > 0:
        return int(ve.max().item())
    img_tok_id = int(lm_head.config.image_token_id)
    img = (input_ids_1d == img_tok_id).nonzero(as_tuple=False).squeeze(-1)
    if img.numel() > 0:
        return int(img.max().item())
    return 0


# ====================== core routines ======================

@torch.no_grad()
def _prepare_prefill(model, sample):
    """Build the full prefill sequence (images + ego tokens).

    Returns
    -------
    tuple
        (input_ids, attention_mask, pixel_values, image_grid_thw,
         pos_embed, ego_feature, coords3d,
         vision_indices,   # positions of IMAGE_TOKEN for *vision* features
         ego_indices)      # positions of IMAGE_TOKEN for *ego* features
    """
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    pixel_values = sample["pixel_values"]
    image_grid_thw = sample["image_grid_thw"]
    data = sample["data"]
    img = sample["img"]
    img_metas = sample["img_metas"]
    B = pixel_values.shape[0]

    image_tok_id = int(model.lm_head.config.image_token_id)

    # vision IMAGE_TOKEN count *before* ego insertion
    n_vision = int((input_ids[0] == image_tok_id).sum().item())

    # ---- 3-D visual position encoding ----
    pos_embed = None
    coords3d = None
    if model.vis_3d_pos:
        depth = model.depth_prediction(img, data["intrinsics"], img_metas=img_metas)
        location = model.prepare_location(image_grid_thw, pixel_values)
        pos_embed, coords3d = model.position_embeding(
            data, location, img_metas, depth, image_grid_thw,
        )

    # ---- ego status tokens ----
    ego_feature = None
    if model.ego_status is not None:
        rec_can_bus = torch.cat(
            [data["command"].unsqueeze(-1), data["can_bus"]], dim=-1,
        )
        ego_feature = torch.empty(
            B, 0, model.llm_hidden_dim, device=rec_can_bus.device,
        )

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
            anchor = _last_vision_anchor(input_ids[0], model.lm_head)
            ins = torch.tensor(
                [VISION_START_TOKEN_INDEX, IMAGE_TOKEN_INDEX, VISION_END_TOKEN_INDEX],
                device=input_ids.device,
            ).unsqueeze(0)
            input_ids = torch.cat(
                [input_ids[:, :anchor + 1], ins, input_ids[:, anchor + 1:]],
                dim=-1,
            )
            attention_mask = torch.cat(
                [attention_mask[:, :anchor + 1], torch.ones_like(ins),
                 attention_mask[:, anchor + 1:]],
                dim=-1,
            )

        if "PE" in model.ego_status:
            past_xyz = model.memory_egopose[:, :model.ego_status_len, :3, 3]
            encoded = model.position_encoder(
                past_xyz.reshape(B, -1, 3),
            ).reshape(B, model.ego_status_len, -1)
            ego_feature = torch.cat([ego_feature, encoded], dim=1)
            anchor = _last_vision_anchor(input_ids[0], model.lm_head)
            ins = torch.tensor(
                [POS_INDICATOR_TOKEN_INDEX, IMAGE_TOKEN_INDEX] * encoded.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
            input_ids = torch.cat(
                [input_ids[:, :anchor + 1], ins, input_ids[:, anchor + 1:]],
                dim=-1,
            )
            attention_mask = torch.cat(
                [attention_mask[:, :anchor + 1], torch.ones_like(ins),
                 attention_mask[:, anchor + 1:]],
                dim=-1,
            )

    # Split IMAGE_TOKEN positions into vision vs ego.
    # Original vision tokens occupy the first *n_vision* positions;
    # everything inserted afterwards is ego.
    all_img_pos = (
        (input_ids[0] == image_tok_id)
        .nonzero(as_tuple=False)
        .squeeze(-1)
        .tolist()
    )
    vision_indices = all_img_pos[:n_vision]
    ego_indices = all_img_pos[n_vision:]

    return (
        input_ids, attention_mask, pixel_values, image_grid_thw,
        pos_embed, ego_feature, coords3d,
        vision_indices, ego_indices,
    )


@torch.no_grad()
def _analyze_one(model, prefill, max_new_tokens):
    """Generate trajectory, run teacher-forced forward, return layer-wise attention."""
    (
        input_ids, attn_mask, pixel_values, image_grid_thw,
        pos_embed, ego_feature, coords3d,
        vision_indices, ego_indices,
    ) = prefill
    prefill_len = input_ids.shape[1]

    if len(vision_indices) == 0:
        return None

    ego_kw = (
        {"ego_feature": ego_feature}
        if ego_feature is not None and ego_feature.numel() > 0
        else {"ego_feature": None}
    )

    shared_kw = dict(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pos_emb=pos_embed,
        planning_only=model.planning_only if model.io_3d_pos else False,
        single_coords_only=model.single_coords_only if model.io_3d_pos else False,
        enable_pe_input=model.enable_pe_input if model.io_3d_pos else False,
        pos_index=coords3d if model.use_rope else None,
        **ego_kw,
        **model._extra_lm_forward_kwargs(input_ids),
    )

    # ---- step 1: generate trajectory ----
    gen = model.lm_head.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=False,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **shared_kw,
    )
    seq = gen.sequences[0]
    del gen
    torch.cuda.empty_cache()

    generated = seq[prefill_len:]
    traj_rel = (generated == POS_INDICATOR_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if traj_rel.numel() == 0:
        return None
    traj_abs = (traj_rel + prefill_len).tolist()

    # ---- step 2: teacher-forced forward with attentions ----
    full_ids = seq.unsqueeze(0)
    full_mask = torch.ones_like(full_ids)

    fw = model.lm_head(
        input_ids=full_ids,
        attention_mask=full_mask,
        output_attentions=True,
        return_dict=True,
        **shared_kw,
    )

    if not hasattr(fw, "attentions") or fw.attentions is None:
        del fw
        return None

    # ---- step 3: extract attention weights ----
    dev = input_ids.device
    traj_idx = torch.tensor(traj_abs, device=dev, dtype=torch.long)
    v_idx = torch.tensor(vision_indices, device=dev, dtype=torch.long)
    e_idx = torch.tensor(ego_indices, device=dev, dtype=torch.long) if ego_indices else None

    vision_layers, ego_layers = [], []
    for layer_attn in fw.attentions:
        a = layer_attn[0]                         # (heads, seq, seq)
        traj_attn = a[:, traj_idx]                 # (heads, n_traj, seq)
        vision_layers.append(traj_attn[:, :, v_idx].mean().item())
        ego_layers.append(
            traj_attn[:, :, e_idx].mean().item()
            if e_idx is not None and e_idx.numel() > 0
            else 0.0
        )
        del a, traj_attn

    del fw
    torch.cuda.empty_cache()

    return {
        "vision_layer_mean": vision_layers,
        "ego_layer_mean": ego_layers,
        "trajectory_token_indices": traj_abs,
    }


# ====================== plotting ======================

def _plot_and_save(straight_v, straight_e, turn_v, turn_e, out_png):
    layers = np.arange(len(straight_v))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(layers, straight_v, color="tab:blue", label="vision")
    axes[0].plot(layers, straight_e, color="tab:red", label="ego")
    axes[0].set_title("Straight")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Attention")
    axes[0].legend()

    axes[1].plot(layers, turn_v, color="tab:blue", label="vision")
    axes[1].plot(layers, turn_e, color="tab:red", label="ego")
    axes[1].set_title("Turn")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Attention")
    axes[1].legend()

    bars = [np.mean(straight_v), np.mean(turn_v),
            np.mean(straight_e), np.mean(turn_e)]
    axes[2].bar(
        ["vision_straight", "vision_turn", "ego_straight", "ego_turn"],
        bars,
        color=["tab:blue", "tab:blue", "tab:red", "tab:red"],
    )
    axes[2].set_title("Overall mean")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ====================== main ======================

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            importlib.import_module(_module_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    # ---- dataset / dataloader ----
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # ---- model ----
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0]).eval()
    m = model.module

    use_first_n = args.first_n_samples > 0
    if use_first_n:
        straight_set, turn_set, target_cls = set(), set(), {}
        per_class = {"all": []}
        n_target = args.first_n_samples
        print(f"[info] mode=first_n, targets: {n_target}")
    else:
        # ---- classify straight / turn ----
        straight_set, turn_set = _classify_tokens(
            args.ann_file, args.straight_samples, args.turn_samples, args.seed,
        )
        target_cls = {}
        for t in straight_set:
            target_cls[t] = "straight"
        for t in turn_set:
            target_cls[t] = "turn"
        per_class = {"straight": [], "turn": []}
        n_target = len(straight_set) + len(turn_set)
        print(f"[info] mode=split, targets: {len(straight_set)} straight, {len(turn_set)} turn")

    details = []

    # ---- iterate over dataset ----
    for batch_idx, batch in enumerate(data_loader):
        try:
            token, img_metas = _sample_token(batch)
        except (TypeError, KeyError):
            continue

        if use_first_n:
            cls = "all"
            if len(per_class["all"]) >= args.first_n_samples:
                break
        else:
            if token not in target_cls:
                continue
            cls = target_cls[token]
            cap = len(straight_set) if cls == "straight" else len(turn_set)
            if len(per_class[cls]) >= cap:
                continue

        if hasattr(m, "reset_memory"):
            m.reset_memory()

        try:
            dev = torch.device("cuda")
            sample = {
                "img": _to_dev(batch, "img", dev),
                "input_ids": _to_dev(batch, "input_ids", dev),
                "pixel_values": _to_dev(batch, "pixel_values", dev),
                "image_grid_thw": _to_dev(batch, "image_grid_thw", dev),
                "attention_mask": _to_dev(batch, "attention_mask", dev),
                "img_metas": img_metas,
                "data": {
                    "lidar2img": _to_dev(batch, "lidar2img", dev),
                    "intrinsics": _to_dev(batch, "intrinsics", dev),
                    "extrinsics": _to_dev(batch, "extrinsics", dev),
                    "timestamp": _to_dev(batch, "timestamp", dev),
                    "img_timestamp": _to_dev(batch, "img_timestamp", dev),
                    "ego_pose": _to_dev(batch, "ego_pose", dev),
                    "ego_pose_inv": _to_dev(batch, "ego_pose_inv", dev),
                    "command": _to_dev(batch, "command", dev),
                    "can_bus": _to_dev(batch, "can_bus", dev),
                },
            }

            if m.ego_status is not None:
                m.pre_update_memory(sample["data"])

            prefill = _prepare_prefill(m, sample)
            out = _analyze_one(m, prefill, args.max_new_tokens)
        except Exception as e:
            print(f"[warn] skip {token}: {e}")
            torch.cuda.empty_cache()
            continue

        if out is None:
            print(f"[skip] {token}: no trajectory tokens generated")
            continue

        per_class[cls].append(out)
        details.append({"token": token, "class": cls, **out})
        done = len(per_class["all"]) if use_first_n else (len(per_class["straight"]) + len(per_class["turn"]))
        print(
            f"[{done}/{n_target}] {cls:>8s}  {token}  "
            f"vis={np.mean(out['vision_layer_mean']):.6f}  "
            f"ego={np.mean(out['ego_layer_mean']):.6f}"
        )

        if use_first_n:
            if len(per_class["all"]) >= args.first_n_samples:
                break
        else:
            if (len(per_class["straight"]) >= len(straight_set)
                    and len(per_class["turn"]) >= len(turn_set)):
                break

    # ---- aggregate ----
    if use_first_n:
        if len(per_class["all"]) == 0:
            raise RuntimeError("No samples processed successfully")
    else:
        if len(per_class["straight"]) == 0 and len(per_class["turn"]) == 0:
            raise RuntimeError("No samples processed successfully")

    def _mean_layers(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float64).mean(axis=0).tolist()

    if use_first_n:
        n_layers = len(per_class["all"][0]["vision_layer_mean"])
        all_v = _mean_layers(per_class["all"], "vision_layer_mean")
        all_e = _mean_layers(per_class["all"], "ego_layer_mean")
        # keep plotting function unchanged (duplicate curves on both panels)
        straight_v, straight_e, turn_v, turn_e = all_v, all_e, all_v, all_e
    else:
        n_layers = max(
            len(per_class["straight"][0]["vision_layer_mean"]) if per_class["straight"] else 0,
            len(per_class["turn"][0]["vision_layer_mean"]) if per_class["turn"] else 0,
        )
        straight_v = _mean_layers(per_class["straight"], "vision_layer_mean") if per_class["straight"] else [0.0] * n_layers
        straight_e = _mean_layers(per_class["straight"], "ego_layer_mean") if per_class["straight"] else [0.0] * n_layers
        turn_v = _mean_layers(per_class["turn"], "vision_layer_mean") if per_class["turn"] else [0.0] * n_layers
        turn_e = _mean_layers(per_class["turn"], "ego_layer_mean") if per_class["turn"] else [0.0] * n_layers

    # ---- save ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "attention_results.json")
    out_png = os.path.join(args.output_dir, "attention_analysis.png")
    if use_first_n:
        payload = {
            "mode": "first_n",
            "counts": {
                "target": args.first_n_samples,
                "used": len(per_class["all"]),
            },
            "layerwise_mean": {
                "all": {"vision": straight_v, "ego": straight_e},
            },
            "overall_mean": {
                "vision_all": float(np.mean(straight_v)),
                "ego_all": float(np.mean(straight_e)),
            },
            "sample_details_first_20": details[:20],
        }
    else:
        payload = {
            "mode": "split",
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
            "overall_mean": {
                "vision_straight": float(np.mean(straight_v)),
                "vision_turn": float(np.mean(turn_v)),
                "ego_straight": float(np.mean(straight_e)),
                "ego_turn": float(np.mean(turn_e)),
            },
            "sample_details_first_20": details[:20],
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _plot_and_save(straight_v, straight_e, turn_v, turn_e, out_png)

    print(f"\n[done] {out_json}")
    print(f"[done] {out_png}")
    if use_first_n:
        print(f"all: {len(per_class['all'])} samples")
    else:
        print(
            f"straight: {len(per_class['straight'])} samples, "
            f"turn: {len(per_class['turn'])} samples"
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
