#!/usr/bin/env python3
"""Visualize KLAL GT attention maps over original camera images."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay KLAL attention maps on camera images.")
    parser.add_argument(
        "--ann-file",
        default="data/nuscenes/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="SpaceDrive nuScenes info PKL used to generate the maps.",
    )
    parser.add_argument(
        "--map-dir",
        default="workspace/klal_gt_attention_maps",
        help="Directory containing <sample_token>.pt or debug <sample_token>.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output visualization directory. Defaults to <map-dir>/visualizations_overlay.",
    )
    parser.add_argument("--image-size", nargs=2, type=int, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--token-h", type=int, default=None)
    parser.add_argument("--token-w", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tokens", nargs="*", default=None, help="Optional explicit sample tokens.")
    parser.add_argument("--camera-order", nargs="+", default=list(DEFAULT_CAMERA_ORDER))
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    path_str = str(path)
    if path_str.startswith("./"):
        path_str = path_str[2:]
    return Path(path_str)


def load_infos(ann_file: Path) -> dict:
    with ann_file.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    return {info.get("token") or info.get("sample_idx"): info for info in infos}


def infer_grid(args: argparse.Namespace, map_dir: Path) -> tuple[int, int]:
    if args.token_h is not None and args.token_w is not None:
        return args.token_h, args.token_w

    meta_path = map_dir / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu")
        if "token_h" in meta and "token_w" in meta:
            return int(meta["token_h"]), int(meta["token_w"])

    raise ValueError("Cannot infer token grid. Pass --token-h and --token-w.")


def load_cam_maps(path: Path, token_h: int, token_w: int, num_views: int) -> np.ndarray:
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(npz_path)
        if "cam_maps" in data:
            return data["cam_maps"].astype(np.float32)

    attn = torch.load(path, map_location="cpu").float().numpy()
    expected = num_views * token_h * token_w
    if attn.size != expected:
        raise ValueError(f"{path} has {attn.size} values, expected {expected}.")
    return attn.reshape(num_views, token_h, token_w).astype(np.float32)


def heat_rgba(attn: np.ndarray, alpha: int) -> np.ndarray:
    attn = attn.astype(np.float32)
    attn = attn / (float(attn.max()) + 1e-12)
    red = np.clip(255 * 1.8 * attn, 0, 255)
    green = np.clip(255 * 1.6 * np.maximum(0, 1 - np.abs(attn - 0.50) / 0.50) * attn, 0, 255)
    blue = np.clip(255 * (0.15 + 0.15 * (1 - attn)), 0, 255)
    alpha_arr = np.clip(alpha * attn, 0, alpha)
    return np.stack([red, green, blue, alpha_arr], axis=-1).astype(np.uint8)


def make_overlay(
    info: dict,
    cam_maps: np.ndarray,
    camera_order: list[str],
    image_size: tuple[int, int],
    alpha: int,
) -> Image.Image:
    target_h, target_w = image_size
    panel_w, panel_h = 320, 350
    panels = []

    for idx, cam_name in enumerate(camera_order):
        cam_info = info["cams"][cam_name]
        img = Image.open(resolve_path(cam_info["data_path"])).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        heat = Image.fromarray(heat_rgba(cam_maps[idx], alpha))
        heat = heat.resize((target_w, target_h), Image.Resampling.BILINEAR)
        overlay = Image.alpha_composite(img.convert("RGBA"), heat).convert("RGB")

        panel = Image.new("RGB", (panel_w, panel_h), "white")
        panel.paste(overlay.resize((panel_w, panel_w), Image.Resampling.BILINEAR), (0, 0))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 326), f"{cam_name}  max={float(cam_maps[idx].max()):.3g}", fill=(0, 0, 0))
        panels.append(panel)

    sheet = Image.new("RGB", (panel_w * 3, panel_h * 2), "white")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % 3) * panel_w, (idx // 3) * panel_h))
    return sheet


def main() -> None:
    args = parse_args()
    ann_file = Path(args.ann_file)
    map_dir = Path(args.map_dir)
    out_dir = Path(args.out_dir) if args.out_dir else map_dir / "visualizations_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    info_by_token = load_infos(ann_file)
    token_h, token_w = infer_grid(args, map_dir)

    if args.tokens:
        tokens = args.tokens
    else:
        paths = sorted(p for p in map_dir.glob("*.pt") if p.name != "metadata.pt")
        tokens = [p.stem for p in paths]
    if args.limit is not None:
        tokens = tokens[: args.limit]

    for token in tokens:
        if token not in info_by_token:
            print(f"[skip] {token}: not found in {ann_file}")
            continue
        pt_path = map_dir / f"{token}.pt"
        if not pt_path.exists() and not pt_path.with_suffix(".npz").exists():
            print(f"[skip] {token}: no .pt/.npz in {map_dir}")
            continue
        cam_maps = load_cam_maps(pt_path, token_h, token_w, len(args.camera_order))
        sheet = make_overlay(
            info_by_token[token],
            cam_maps,
            list(args.camera_order),
            tuple(args.image_size),
            args.alpha,
        )
        out_path = out_dir / f"{token}_overlay.png"
        sheet.save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
