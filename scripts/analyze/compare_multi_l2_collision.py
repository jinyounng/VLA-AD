#!/usr/bin/env python3
"""Compare L2 / object-collision across multiple prediction folders, split by turn vs straight.

Turn vs straight uses OpenDriveVLA-style mission from cached_nuscenes_info.pkl
(gt_ego_fut_cmd: [right, left, forward]), same as analyze_turn_understeer.py / compare_turn_l2_collision.py.

By default only samples where *all* runs have a valid prediction are counted (fair comparison).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from nuscenes.eval.common.utils import Quaternion

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


try:
    from planning_utils import PlanningMetric
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = repo_root / "scripts" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from planning_utils import PlanningMetric


# ── mission (turn vs straight) ────────────────────────────────────────────


def mission_goal_from_cmd(cmd_vec) -> str:
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


def split_from_mission(mission: str) -> str:
    if mission in ("turn_left", "turn_right"):
        return "turn"
    if mission == "keep_forward":
        return "straight"
    return "unknown"


# ── preds ────────────────────────────────────────────────────────────────


def parse_coords_from_text(traj_text: str) -> np.ndarray:
    full_match = re.search(
        r"\[<POS_INDICATOR>\(([\d\.-]+, [\d\.-]+)\)\s*([\s\S]*<POS_INDICATOR>\([\d\.-]+, [\d\.-]+\)\s*)*",
        traj_text,
    )
    if full_match is None:
        coords = re.findall(r"\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)", traj_text)
        if not coords:
            return np.empty((0, 2), dtype=np.float32)
    else:
        coords = re.findall(r"\(\+?[\d\.-]+, \+?[\d\.-]+\)", full_match.group(0))

    parsed = [tuple(map(float, re.findall(r"-?\d+\.\d+|-?\d+", c))) for c in coords]
    if not parsed:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(parsed, dtype=np.float32)


def load_preds(pred_path: str, infos) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    pred_path = pred_path if pred_path.endswith("/") else pred_path + "/"
    for data in infos:
        token = data["token"]
        pred_file = pred_path + token
        if not os.path.exists(pred_file):
            continue
        try:
            with open(pred_file, "r", encoding="utf8") as f:
                pred_data = json.load(f)
            traj_text = pred_data[0]["A"]
            coords = parse_coords_from_text(traj_text)
            if coords.shape[0] >= 6:
                preds[token] = coords[:6, :2]
        except Exception:
            continue
    return preds


def append_tangent_directions(traj_xy: np.ndarray) -> np.ndarray:
    directions = []
    if np.linalg.norm(traj_xy[0]) < 0.5:
        directions.append(0.0)
    else:
        directions.append(np.arctan2(traj_xy[0][1], traj_xy[0][0]))
    for i in range(1, len(traj_xy)):
        vec = traj_xy[i] - traj_xy[i - 1]
        if np.linalg.norm(vec) < 0.3:
            angle = directions[-1]
        else:
            angle = np.arctan2(vec[1], vec[0])
        directions.append(angle)
    return np.concatenate([traj_xy, np.array(directions).reshape(-1, 1)], axis=-1)


# ── metrics ────────────────────────────────────────────────────────────────


def init_counter():
    return {
        "n": 0,
        "l2_1s_sum": 0.0,
        "l2_2s_sum": 0.0,
        "l2_3s_sum": 0.0,
        "obj_col_1s_sum": 0.0,
        "obj_col_2s_sum": 0.0,
        "obj_col_3s_sum": 0.0,
    }


def parse_runs(runs_arg: List[str]) -> List[Tuple[str, str]]:
    """Each item is 'label:path' or 'label=path'."""
    out: List[Tuple[str, str]] = []
    for item in runs_arg:
        sep = ":" if ":" in item else "="
        if sep not in item:
            raise ValueError(f"Invalid --runs entry (need label:path): {item}")
        label, path = item.split(sep, 1)
        label = label.strip()
        path = path.strip().rstrip("/")
        if not label or not path:
            raise ValueError(f"Invalid --runs entry: {item}")
        out.append((label, path))
    return out


def format_row(label: str, split: str, c: dict) -> str:
    n = c["n"]
    if n == 0:
        return f"{label:16s}  {split:10s}  n=0"
    l1 = c["l2_1s_sum"] / n
    l2 = c["l2_2s_sum"] / n
    l3 = c["l2_3s_sum"] / n
    o1 = 100.0 * c["obj_col_1s_sum"] / n
    o2 = 100.0 * c["obj_col_2s_sum"] / n
    o3 = 100.0 * c["obj_col_3s_sum"] / n
    return (
        f"{label:16s}  {split:10s}  n={n:5d}  "
        f"L2@1/2/3s {l1:.4f}/{l2:.4f}/{l3:.4f}  "
        f"ObjCol% {o1:.3f}/{o2:.3f}/{o3:.3f}"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Space-separated list label:path to each _results_planning_only directory",
    )
    p.add_argument("--base-path", type=str, required=True, help="nuScenes base (e.g. data/nuscenes)")
    p.add_argument(
        "--anno-path",
        default="nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="Annotation pkl filename under base-path",
    )
    p.add_argument(
        "--cached-nusc-info",
        default="data/nuscenes/cached_nuscenes_info.pkl",
        help="cached_nuscenes_info.pkl (gt_ego_fut_cmd)",
    )
    p.add_argument(
        "--union",
        action="store_true",
        help="If set, count each run on its own available preds (not intersection across runs)",
    )
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def main():
    args = parse_args()
    runs = parse_runs(args.runs)
    anno_file = os.path.join(args.base_path, args.anno_path)
    key_infos = pickle.load(open(anno_file, "rb"))
    infos = key_infos["infos"]

    if not os.path.exists(args.cached_nusc_info):
        raise FileNotFoundError(args.cached_nusc_info)
    mission_map = load_cached_mission_map(args.cached_nusc_info)

    pred_maps = {}
    run_iter = runs
    if not args.no_progress:
        run_iter = tqdm(runs, desc="Load predictions", unit="run")
    for label, path in run_iter:
        pred_maps[label] = load_preds(path, infos)
    planning_metric = PlanningMetric(args.base_path)
    ego_boxes = np.array([[0, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])

    # counters[label][split] where split in turn | straight
    counters: Dict[str, Dict[str, dict]] = {
        label: {"turn": init_counter(), "straight": init_counter()} for label, _ in runs
    }
    stats = {
        "total_infos": len(infos),
        "skipped_no_mission": 0,
        "skipped_unknown_mission": 0,
        "skipped_invalid_mask": 0,
        "skipped_missing_pred_intersection": 0,
        "used_turn": 0,
        "used_straight": 0,
    }

    sample_iter = infos
    if not args.no_progress:
        sample_iter = tqdm(
            infos,
            desc="L2 / collision (turn vs straight)",
            unit="sample",
            total=len(infos),
        )

    for data in sample_iter:
        token = data["token"]
        mission = mission_map.get(token)
        if mission is None:
            stats["skipped_no_mission"] += 1
            continue
        sp = split_from_mission(mission)
        if sp == "unknown":
            stats["skipped_unknown_mission"] += 1
            continue

        labels = [lbl for lbl, _ in runs]
        if not args.union:
            if any(token not in pred_maps[lbl] for lbl in labels):
                stats["skipped_missing_pred_intersection"] += 1
                continue
        else:
            if not any(token in pred_maps[lbl] for lbl in labels):
                continue

        mask = data["gt_planning_mask"][0]
        if not bool(mask.all()):
            stats["skipped_invalid_mask"] += 1
            continue

        gt_xy = data["gt_planning"][0, :6, :2]
        gt_t = torch.from_numpy(gt_xy).unsqueeze(0)
        gt_agent_boxes = np.concatenate([data["gt_boxes"], data["gt_velocity"]], -1)
        gt_agent_feats = np.concatenate(
            [
                data["gt_fut_traj"][:, :6].reshape(-1, 12),
                data["gt_fut_traj_mask"][:, :6],
                data["gt_fut_yaw"][:, :6],
                data["gt_fut_idx"],
            ],
            -1,
        )
        bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)

        split_key = sp  # "turn" | "straight"
        if split_key == "turn":
            stats["used_turn"] += 1
        else:
            stats["used_straight"] += 1

        for lbl in labels:
            if args.union and token not in pred_maps[lbl]:
                continue
            pred_xy = pred_maps[lbl][token]
            pred_t = torch.from_numpy(pred_xy).unsqueeze(0)
            pred_yaw = append_tangent_directions(pred_xy)
            pred_mask = np.concatenate(
                [
                    pred_yaw[..., :2].reshape(1, -1),
                    np.ones_like(pred_yaw[..., :1]).reshape(1, -1),
                    pred_yaw[..., 2:].reshape(1, -1),
                ],
                axis=-1,
            )
            _ = planning_metric.get_ego_seg(ego_boxes, pred_mask, add_rec=True)
            c = counters[lbl][split_key]
            for sec_idx, cur_time in enumerate((2, 4, 6), start=1):
                ade = float(np.mean(np.linalg.norm(pred_xy[:cur_time] - gt_xy[:cur_time], axis=1)))
                c[f"l2_{sec_idx}s_sum"] += ade
                _oc, obj_box_coll = planning_metric.evaluate_coll(
                    pred_t[:, :cur_time],
                    gt_t[:, :cur_time],
                    torch.from_numpy(bev_seg[1:]).unsqueeze(0),
                )
                c[f"obj_col_{sec_idx}s_sum"] += float(obj_box_coll.max().item())
            c["n"] += 1

    print("=" * 88)
    print("Multi-run L2 / ObjCol (turn vs straight, mission from cached_nuscenes_info.pkl)")
    print("=" * 88)
    mode = "union (per-run own preds)" if args.union else "intersection (all runs have pred)"
    print(f"Mode: {mode}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()
    for split in ("turn", "straight"):
        print(f"--- {split.upper()} ---")
        for lbl, _path in runs:
            print(format_row(lbl, split, counters[lbl][split]))
        print()

    if args.save_json:
        out = {
            "config": {
                "runs": [{"label": l, "path": p} for l, p in runs],
                "base_path": args.base_path,
                "anno_path": args.anno_path,
                "cached_nusc_info": args.cached_nusc_info,
                "union": args.union,
            },
            "stats": stats,
            "counters": {
                lbl: {split: dict(counters[lbl][split]) for split in ("turn", "straight")}
                for lbl, _ in runs
            },
        }
        with open(args.save_json, "w", encoding="utf8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
