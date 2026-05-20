#!/usr/bin/env python3
"""Visualize turn failure cases (understeer / oversteer): 6 cameras + BEV (GT vs pred trajectories only).

Turn classification: OpenDriveVLA gt_ego_fut_cmd from cached_nuscenes_info.pkl.
Understeer criterion: |θ_net|_pred < |θ_net|_gt (net heading change, first→last segment).

Ego frame: x forward, y left. Plot lateral as −y so vehicle left matches viewer left.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from nuscenes.nuscenes import NuScenes

CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

CAM_LABELS = {
    "CAM_FRONT_LEFT": "Front-Left",
    "CAM_FRONT": "Front",
    "CAM_FRONT_RIGHT": "Front-Right",
    "CAM_BACK_LEFT": "Back-Left",
    "CAM_BACK": "Back",
    "CAM_BACK_RIGHT": "Back-Right",
}

BEV_FRONT, BEV_REAR = 30.0, 10.0
BEV_SIDE = 15.0


def extract_coords_from_text(traj_text: str) -> np.ndarray:
    full_match = re.search(
        r"\[<POS_INDICATOR>\(([\d\.\-]+, [\d\.\-]+)\)\s*"
        r"([\s\S]*<POS_INDICATOR>\([\d\.\-]+, [\d\.\-]+\)\s*)*",
        traj_text,
    )
    if full_match is None:
        coords = re.findall(r"\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)", traj_text)
        if not coords:
            return np.empty((0, 2), dtype=np.float32)
    else:
        coords = re.findall(r"\(\+?[\d\.\-]+, \+?[\d\.\-]+\)", full_match.group(0))
    parsed = [tuple(map(float, re.findall(r"-?\d+\.\d+|-?\d+", c))) for c in coords]
    if not parsed:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(parsed, dtype=np.float32)


def net_heading_change_xy(pts: np.ndarray) -> float:
    """Signed heading change from first segment to last segment [rad]."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return float("nan")
    d0 = pts[1] - pts[0]
    d1 = pts[-1] - pts[-2]
    a0 = np.arctan2(d0[1], d0[0])
    a1 = np.arctan2(d1[1], d1[0])
    d = a1 - a0
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return float(d)


def net_heading_deg(pts: np.ndarray) -> float:
    """Net heading change in degrees."""
    v = net_heading_change_xy(pts)
    return float("nan") if np.isnan(v) else float(np.degrees(v))


def mission_goal_from_cmd(cmd_vec) -> str:
    """cmd_vec: [right, left, forward] as in OpenDriveVLA."""
    right, left, forward = cmd_vec
    if right > 0:
        return "turn_right"
    if left > 0:
        return "turn_left"
    return "keep_forward"


def load_cached_mission_map(cached_path: str) -> Dict[str, str]:
    cache = pickle.load(open(cached_path, "rb"))
    mission_map: Dict[str, str] = {}
    for token, value in cache.items():
        cmd_vec = value.get("gt_ego_fut_cmd")
        if cmd_vec is None or len(cmd_vec) < 3:
            continue
        mission_map[str(token)] = mission_goal_from_cmd(cmd_vec)
    return mission_map


def compute_l2_3s(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    n = min(len(pred_xy), len(gt_xy), 6)
    if n == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(pred_xy[:n] - gt_xy[:n], axis=1)))


def draw_bev(ax: plt.Axes, info: dict, pred_xy: np.ndarray) -> None:
    """BEV: x = −y_ego (lateral), y = x_ego (forward). GT / Pred only."""
    gt_xy = info["gt_planning"][0, :6, :2]
    ax.set_facecolor("#fafafa")

    gt_full = np.vstack([[0, 0], gt_xy])
    pred_full = np.vstack([[0, 0], pred_xy])

    ax.plot(
        -gt_full[:, 1],
        gt_full[:, 0],
        "o-",
        color="#2e7d32",
        ms=5,
        lw=2.2,
        label="GT",
        zorder=5,
    )
    ax.plot(
        -pred_full[:, 1],
        pred_full[:, 0],
        "s-",
        color="#c62828",
        ms=5,
        lw=2.2,
        label="Pred",
        zorder=5,
    )

    for i in range(1, gt_full.shape[0]):
        ax.annotate(
            str(i),
            (-gt_full[i, 1], gt_full[i, 0]),
            fontsize=6,
            color="#2e7d32",
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    for i in range(1, pred_full.shape[0]):
        ax.annotate(
            str(i),
            (-pred_full[i, 1], pred_full[i, 0]),
            fontsize=6,
            color="#c62828",
            fontweight="bold",
            ha="right",
            va="top",
        )

    ax.set_xlim(-BEV_SIDE, BEV_SIDE)
    ax.set_ylim(-BEV_REAR, BEV_FRONT)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral (m, −y ego)", fontsize=9)
    ax.set_ylabel("Longitudinal (m)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        "",
        xy=(0, BEV_FRONT - 1),
        xytext=(0, BEV_FRONT - 4),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax.text(0.5, BEV_FRONT - 2, "front", fontsize=7, color="gray", ha="left")


def load_camera_images(info: dict, nusc: NuScenes, nuscenes_root: str) -> dict:
    imgs = {}
    for cam_name in CAMERA_ORDER:
        cam_info = info["cams"][cam_name]
        data_path = cam_info["data_path"]
        for prefix in ("./data/nuscenes/", "data/nuscenes/"):
            if data_path.startswith(prefix):
                data_path = os.path.join(nuscenes_root, data_path[len(prefix) :])
                break
        if os.path.exists(data_path):
            imgs[cam_name] = Image.open(data_path)
        else:
            imgs[cam_name] = Image.new("RGB", (1600, 900), (128, 128, 128))
    return imgs


def load_predictions(
    infos: List[dict], pred_path: str
) -> Tuple[Dict[str, np.ndarray], int]:
    preds: Dict[str, np.ndarray] = {}
    for d in infos:
        token = d["token"]
        pf = os.path.join(pred_path, token)
        if not os.path.exists(pf):
            continue
        try:
            with open(pf) as f:
                pred_data = json.load(f)
            coords = extract_coords_from_text(pred_data[0]["A"])
            if coords.shape[0] >= 6:
                preds[token] = coords[:6, :2]
        except Exception:
            continue
    return preds, len(infos)


def classify_records(
    preds: Dict[str, np.ndarray],
    info_by_token: Dict[str, dict],
    mission_map: Dict[str, str],
) -> List[dict]:
    records = []
    for token, pred_xy in preds.items():
        d = info_by_token[token]
        gt_xy = d["gt_planning"][0, :6, :2]
        mask = d["gt_planning_mask"][0]
        if not mask.all():
            continue

        mission = mission_map.get(token)
        if mission is None or mission == "keep_forward":
            continue

        gt_angle = net_heading_deg(gt_xy)
        pred_angle = net_heading_deg(pred_xy)
        if np.isnan(gt_angle) or np.isnan(pred_angle):
            continue

        l2_3s = compute_l2_3s(pred_xy, gt_xy)

        if abs(pred_angle) < abs(gt_angle):
            steer_cat = "under_steer"
        elif abs(pred_angle) > abs(gt_angle):
            steer_cat = "over_steer"
        else:
            steer_cat = "good"

        records.append(
            {
                "token": token,
                "steer_cat": steer_cat,
                "l2_3s": l2_3s,
                "gt_angle": gt_angle,
                "pred_angle": pred_angle,
                "mission_goal": mission,
            }
        )
    return records


def select_default(
    records: List[dict], top_k: int
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    under = sorted(
        [r for r in records if r["steer_cat"] == "under_steer"],
        key=lambda r: r["l2_3s"],
        reverse=True,
    )[:top_k]
    over = sorted(
        [r for r in records if r["steer_cat"] == "over_steer"],
        key=lambda r: r["l2_3s"],
        reverse=True,
    )[:top_k]
    good = sorted(
        [r for r in records if r["steer_cat"] == "good"],
        key=lambda r: r["l2_3s"],
    )[:top_k]
    selected = under + over + good
    return under, over, good, selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nuscenes-root",
        default="/data/jykim/projects/OpenDriveVLA/data/nuscenes/",
        help="nuScenes dataroot",
    )
    p.add_argument(
        "--anno-pkl",
        default=None,
        help="Annotation PKL (default: <nuscenes-root>/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl)",
    )
    p.add_argument(
        "--pred-path",
        default="/data/jykim/projects/SpaceDrive/workspace/spacedrive_plus_qwen/_results_planning_only/",
        help="Directory with per-token prediction JSON files",
    )
    p.add_argument(
        "--cached-nusc-info",
        default="data/nuscenes/cached_nuscenes_info.pkl",
        help="Path to cached_nuscenes_info.pkl (OpenDriveVLA gt_ego_fut_cmd)",
    )
    p.add_argument(
        "--save-dir",
        default="/data/jykim/projects/SpaceDrive/workspace/understeer_vis/",
        help="Output directory for PNGs",
    )
    p.add_argument("--top-k", type=int, default=5, help="Per category (under/over/good)")
    p.add_argument(
        "--tokens",
        nargs="*",
        default=None,
        help="If set, render only these sample tokens (skip classification selection)",
    )
    p.add_argument("--verbose", action="store_true", help="NuScenes loader verbose")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nuscenes_root = os.path.expanduser(args.nuscenes_root)
    anno_pkl = args.anno_pkl or os.path.join(
        nuscenes_root, "nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl"
    )
    pred_path = os.path.expanduser(args.pred_path)
    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading mission map from cached_nuscenes_info.pkl …")
    mission_map: Dict[str, str] = {}
    if args.cached_nusc_info and os.path.exists(args.cached_nusc_info):
        mission_map = load_cached_mission_map(args.cached_nusc_info)
        print(f"  {len(mission_map)} tokens loaded")
    else:
        print(f"  WARNING: not found at '{args.cached_nusc_info}'")

    print("Loading annotation PKL …")
    with open(anno_pkl, "rb") as f:
        anno_data = pickle.load(f)
    infos: List[dict] = anno_data["infos"]
    info_by_token = {d["token"]: d for d in infos}

    print("Loading predictions …")
    preds, n_infos = load_predictions(infos, pred_path)
    print(f"Loaded {len(preds)} predictions out of {n_infos} annotations")

    if args.tokens:
        selected = []
        for t in args.tokens:
            if t not in info_by_token:
                print(f"Warning: unknown token {t}, skip")
                continue
            if t not in preds:
                print(f"Warning: no prediction for {t}, skip")
                continue
            selected.append(
                {
                    "token": t,
                    "steer_cat": "manual",
                    "l2_3s": compute_l2_3s(
                        preds[t],
                        info_by_token[t]["gt_planning"][0, :6, :2],
                    ),
                    "gt_angle": net_heading_deg(
                        info_by_token[t]["gt_planning"][0, :6, :2]
                    ),
                    "pred_angle": net_heading_deg(preds[t]),
                    "mission_goal": mission_map.get(t, "unknown"),
                }
            )
    else:
        records = classify_records(preds, info_by_token, mission_map)
        print(f"\nTurn samples with predictions: {len(records)}")
        for cat in ("under_steer", "over_steer", "good"):
            n = sum(1 for r in records if r["steer_cat"] == cat)
            print(f"  {cat}: {n}")
        _, _, _, selected = select_default(records, args.top_k)
        print(f"\nSelected {len(selected)} samples:")
        for r in selected:
            print(
                f"  [{r['steer_cat']:>12s}]  L2_3s={r['l2_3s']:.3f}  "
                f"θ_net GT={r['gt_angle']:+.1f}°  θ_net Pred={r['pred_angle']:+.1f}°  "
                f"mission={r['mission_goal']}"
            )

    print("\nLoading NuScenes …")
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_root, verbose=args.verbose
    )

    saved_files: List[str] = []
    for rec in selected:
        token = rec["token"]
        info = info_by_token[token]
        pred_xy = preds[token]
        imgs = load_camera_images(info, nusc, nuscenes_root)

        fig = plt.figure(figsize=(24, 14))
        gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1.4], hspace=0.15, wspace=0.04)

        for ci, cam_name in enumerate(CAMERA_ORDER):
            ax = fig.add_subplot(gs[0, ci])
            ax.imshow(imgs[cam_name])
            ax.set_title(CAM_LABELS[cam_name], fontsize=10, fontweight="bold")
            ax.axis("off")

        ax_bev = fig.add_subplot(gs[1, 1:5])
        draw_bev(ax_bev, info, pred_xy)

        cat = rec["steer_cat"]
        l2 = rec["l2_3s"]
        mission = rec.get("mission_goal", "")
        fig.suptitle(
            f"[{cat.upper().replace('_', ' ')}]  L2_3s={l2:.3f}m   "
            f"θ_net GT={rec['gt_angle']:+.1f}°  θ_net Pred={rec['pred_angle']:+.1f}°\n"
            f"Token: {token}   Mission: {mission}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        fname = f"{token}_{cat}_{l2:.3f}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_files.append(fpath)
        print(f"  Saved: {fname}")

    print(f"\nDone — {len(saved_files)} figures saved to {save_dir}")


if __name__ == "__main__":
    main()
