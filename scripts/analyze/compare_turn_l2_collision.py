#!/usr/bin/env python3
"""Compare turn-scene L2/collision metrics between two prediction folders.

Turn classification follows OpenDriveVLA mission command in
cached_nuscenes_info.pkl (gt_ego_fut_cmd: [right, left, forward]).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from nuscenes.eval.common.utils import Quaternion

try:
    from planning_utils import PlanningMetric
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = repo_root / "scripts" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from planning_utils import PlanningMetric


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


def parse_coords_from_text(traj_text: str) -> np.ndarray:
    import re

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
    for data in infos:
        token = data["token"]
        pred_file = os.path.join(pred_path, token)
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-a", type=str, required=True, help="Model A _results_planning_only path")
    parser.add_argument("--pred-b", type=str, required=True, help="Model B _results_planning_only path")
    parser.add_argument("--label-a", type=str, default="A", help="Display name for model A")
    parser.add_argument("--label-b", type=str, default="B", help="Display name for model B")
    parser.add_argument("--base-path", type=str, required=True, help="Path to nuscenes base directory")
    parser.add_argument(
        "--anno-path",
        type=str,
        default="nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="Annotation pkl filename under base-path",
    )
    parser.add_argument(
        "--cached-nusc-info",
        type=str,
        default="data/nuscenes/cached_nuscenes_info.pkl",
        help="Path to cached_nuscenes_info.pkl (OpenDriveVLA gt_ego_fut_cmd)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save comparison details JSON",
    )
    return parser.parse_args()


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


def format_metrics(label: str, c: dict):
    n = c["n"]
    if n == 0:
        print(f"\n--- {label} ---")
        print("No valid turn samples.")
        return
    print(f"\n--- {label} (n={n}) ---")
    print(f"L2 @1s: {c['l2_1s_sum'] / n:.4f}  @2s: {c['l2_2s_sum'] / n:.4f}  @3s: {c['l2_3s_sum'] / n:.4f}")
    print(
        "ObjCol @1s: "
        f"{100.0 * c['obj_col_1s_sum'] / n:.4f}%  "
        f"@2s: {100.0 * c['obj_col_2s_sum'] / n:.4f}%  "
        f"@3s: {100.0 * c['obj_col_3s_sum'] / n:.4f}%"
    )


def main():
    args = parse_args()

    anno_file = os.path.join(args.base_path, args.anno_path)
    key_infos = pickle.load(open(anno_file, "rb"))
    infos = key_infos["infos"]

    if not os.path.exists(args.cached_nusc_info):
        raise FileNotFoundError(f"cached_nuscenes_info.pkl not found: {args.cached_nusc_info}")
    mission_map = load_cached_mission_map(args.cached_nusc_info)

    preds_a = load_preds(args.pred_a, infos)
    preds_b = load_preds(args.pred_b, infos)
    planning_metric = PlanningMetric(args.base_path)
    ego_boxes = np.array([[0, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])

    counter_a = init_counter()
    counter_b = init_counter()

    compared_tokens = 0
    skipped_no_mission = 0
    skipped_non_turn = 0
    skipped_missing_pred = 0
    skipped_invalid_mask = 0
    details = []

    for data in infos:
        token = data["token"]

        if token not in preds_a or token not in preds_b:
            skipped_missing_pred += 1
            continue

        mission = mission_map.get(token)
        if mission is None:
            skipped_no_mission += 1
            continue
        if mission not in ("turn_left", "turn_right"):
            skipped_non_turn += 1
            continue

        gt_xy = data["gt_planning"][0, :6, :2]
        mask = data["gt_planning_mask"][0]
        if not bool(mask.all()):
            skipped_invalid_mask += 1
            continue

        compared_tokens += 1
        pred_a = preds_a[token]
        pred_b = preds_b[token]

        pred_a_t = torch.from_numpy(pred_a).unsqueeze(0)
        pred_b_t = torch.from_numpy(pred_b).unsqueeze(0)
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

        e2g_r_mat = Quaternion(data["ego2global_rotation"]).rotation_matrix
        e2g_t = data["ego2global_translation"]
        drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)
        _ = drivable_seg  # kept for parity with existing metric setup

        pred_a_yaw = append_tangent_directions(pred_a)
        pred_b_yaw = append_tangent_directions(pred_b)
        pred_a_mask = np.concatenate(
            [
                pred_a_yaw[..., :2].reshape(1, -1),
                np.ones_like(pred_a_yaw[..., :1]).reshape(1, -1),
                pred_a_yaw[..., 2:].reshape(1, -1),
            ],
            axis=-1,
        )
        pred_b_mask = np.concatenate(
            [
                pred_b_yaw[..., :2].reshape(1, -1),
                np.ones_like(pred_b_yaw[..., :1]).reshape(1, -1),
                pred_b_yaw[..., 2:].reshape(1, -1),
            ],
            axis=-1,
        )
        _ = planning_metric.get_ego_seg(ego_boxes, pred_a_mask, add_rec=True)
        _ = planning_metric.get_ego_seg(ego_boxes, pred_b_mask, add_rec=True)

        token_detail = {"token": token, "mission_goal": mission}
        for sec_idx, cur_time in enumerate((2, 4, 6), start=1):
            ade_a = float(np.mean(np.linalg.norm(pred_a[:cur_time] - gt_xy[:cur_time], axis=1)))
            ade_b = float(np.mean(np.linalg.norm(pred_b[:cur_time] - gt_xy[:cur_time], axis=1)))

            counter_a[f"l2_{sec_idx}s_sum"] += ade_a
            counter_b[f"l2_{sec_idx}s_sum"] += ade_b

            _obj_coll_a, obj_box_coll_a = planning_metric.evaluate_coll(
                pred_a_t[:, :cur_time],
                gt_t[:, :cur_time],
                torch.from_numpy(bev_seg[1:]).unsqueeze(0),
            )
            _obj_coll_b, obj_box_coll_b = planning_metric.evaluate_coll(
                pred_b_t[:, :cur_time],
                gt_t[:, :cur_time],
                torch.from_numpy(bev_seg[1:]).unsqueeze(0),
            )
            obj_a = float(obj_box_coll_a.max().item())
            obj_b = float(obj_box_coll_b.max().item())
            counter_a[f"obj_col_{sec_idx}s_sum"] += obj_a
            counter_b[f"obj_col_{sec_idx}s_sum"] += obj_b

            token_detail[f"l2_{sec_idx}s_a"] = ade_a
            token_detail[f"l2_{sec_idx}s_b"] = ade_b
            token_detail[f"obj_col_{sec_idx}s_a"] = obj_a
            token_detail[f"obj_col_{sec_idx}s_b"] = obj_b

        counter_a["n"] += 1
        counter_b["n"] += 1
        details.append(token_detail)

    print("=" * 72)
    print("Turn-only comparison (mission in {turn_left, turn_right})")
    print("=" * 72)
    print(f"infos total: {len(infos)}")
    print(f"pred loaded: {args.label_a}={len(preds_a)}  {args.label_b}={len(preds_b)}")
    print(f"compared turn samples (intersection): {compared_tokens}")
    print(f"skipped missing prediction in either side: {skipped_missing_pred}")
    print(f"skipped no mission command: {skipped_no_mission}")
    print(f"skipped keep_forward samples: {skipped_non_turn}")
    print(f"skipped invalid planning mask: {skipped_invalid_mask}")

    format_metrics(args.label_a, counter_a)
    format_metrics(args.label_b, counter_b)

    if compared_tokens > 0:
        print("\n--- Delta (B - A) ---")
        print(
            f"L2 @1s: {(counter_b['l2_1s_sum'] - counter_a['l2_1s_sum']) / compared_tokens:+.4f}  "
            f"@2s: {(counter_b['l2_2s_sum'] - counter_a['l2_2s_sum']) / compared_tokens:+.4f}  "
            f"@3s: {(counter_b['l2_3s_sum'] - counter_a['l2_3s_sum']) / compared_tokens:+.4f}"
        )
        print(
            "ObjCol @1s: "
            f"{100.0 * (counter_b['obj_col_1s_sum'] - counter_a['obj_col_1s_sum']) / compared_tokens:+.4f}%  "
            f"@2s: {100.0 * (counter_b['obj_col_2s_sum'] - counter_a['obj_col_2s_sum']) / compared_tokens:+.4f}%  "
            f"@3s: {100.0 * (counter_b['obj_col_3s_sum'] - counter_a['obj_col_3s_sum']) / compared_tokens:+.4f}%"
        )

    if args.save_json:
        out = {
            "config": {
                "pred_a": args.pred_a,
                "pred_b": args.pred_b,
                "label_a": args.label_a,
                "label_b": args.label_b,
                "base_path": args.base_path,
                "anno_path": args.anno_path,
                "cached_nusc_info": args.cached_nusc_info,
            },
            "summary": {
                "compared_turn_samples": compared_tokens,
                "skipped_missing_pred": skipped_missing_pred,
                "skipped_no_mission": skipped_no_mission,
                "skipped_non_turn": skipped_non_turn,
                "skipped_invalid_mask": skipped_invalid_mask,
                "metrics_a": counter_a,
                "metrics_b": counter_b,
            },
            "details": details,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved comparison JSON: {args.save_json}")


if __name__ == "__main__":
    main()
