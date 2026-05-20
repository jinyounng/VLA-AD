#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Analyze turn vs non-turn samples and under-steering on turn samples.
#
# Turn classification: OpenDriveVLA's gt_ego_fut_cmd from cached_nuscenes_info.pkl
# Understeer criteria: 3 angle metrics from OpenDriveVLA
#   - Σ|Δθ| (sum of absolute segment turns)
#   - |θ_net| (net heading change, first→last segment)
#   - |θ_chord| (chord angle, first→last point displacement)
# ------------------------------------------------------------------------

import argparse
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nuscenes.eval.common.utils import Quaternion
from planning_utils import PlanningMetric


# ── OpenDriveVLA angle metrics ────────────────────────────────────────────

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


def sum_abs_segment_turns(pts: np.ndarray) -> float:
    """Sum of |delta angle| between consecutive segments [rad]."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return float("nan")
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    s = 0.0
    for i in range(len(ang) - 1):
        da = ang[i + 1] - ang[i]
        while da > np.pi:
            da -= 2 * np.pi
        while da < -np.pi:
            da += 2 * np.pi
        s += abs(da)
    return float(s)


def trajectory_chord_angle_xy(pts: np.ndarray) -> float:
    """Direction angle of displacement vector from first to last point [rad]."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return float("nan")
    d = pts[-1] - pts[0]
    return float(np.arctan2(d[1], d[0]))


def gt_severity_deg(gt_xy: np.ndarray, mode: str) -> float:
    """Scalar [deg] for binning by GT severity."""
    if mode == "net_heading":
        v = net_heading_change_xy(gt_xy)
    elif mode == "sum_abs":
        v = sum_abs_segment_turns(gt_xy)
    elif mode == "chord":
        v = trajectory_chord_angle_xy(gt_xy)
    else:
        raise ValueError(mode)
    if np.isnan(v):
        return float("nan")
    if mode == "sum_abs":
        return float(v) * 180.0 / np.pi
    return abs(float(v)) * 180.0 / np.pi


# ── Turn classification (OpenDriveVLA style) ─────────────────────────────

def mission_goal_from_cmd(cmd_vec) -> str:
    """cmd_vec: [right, left, forward] as in OpenDriveVLA."""
    right, left, forward = cmd_vec
    if right > 0:
        return "turn_right"
    if left > 0:
        return "turn_left"
    return "keep_forward"


def load_cached_mission_map(cached_path: str) -> Dict[str, str]:
    """Load cached_nuscenes_info.pkl → {token: mission_goal}."""
    cache = pickle.load(open(cached_path, "rb"))
    mission_map: Dict[str, str] = {}
    for token, value in cache.items():
        cmd_vec = value.get("gt_ego_fut_cmd")
        if cmd_vec is None or len(cmd_vec) < 3:
            continue
        mission_map[str(token)] = mission_goal_from_cmd(cmd_vec)
    return mission_map


# ── Helpers ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze turn/understeer from SpaceDrive outputs (OpenDriveVLA criteria).")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to _results_planning_only directory")
    parser.add_argument("--base_path", type=str, required=True, help="Path to nuscenes base directory")
    parser.add_argument(
        "--anno_path", type=str,
        default="nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="Annotation pkl filename under base_path",
    )
    parser.add_argument(
        "--cached-nusc-info", type=str,
        default="data/nuscenes/cached_nuscenes_info.pkl",
        help="Path to cached_nuscenes_info.pkl (OpenDriveVLA gt_ego_fut_cmd)",
    )
    parser.add_argument(
        "--show-bins", action="store_true",
        help="Show GT severity binned understeer table",
    )
    parser.add_argument(
        "--bin-by", choices=("net_heading", "sum_abs", "chord"),
        default="net_heading",
        help="Angle metric used for binning [deg]",
    )
    parser.add_argument(
        "--bin-edges-deg", type=str, default="15,30,45,60",
        help="Bin boundary degrees, comma-separated",
    )
    parser.add_argument(
        "--bins-per-split", action="store_true",
        help="Show binned table per turn_left/turn_right (default: ALL TURN only)",
    )
    parser.add_argument(
        "--save-details-json", type=str, default=None,
        help="Optional path to save per-sample details JSON",
    )
    return parser.parse_args()


def _normalize_pred_path(pred_path: str) -> str:
    return pred_path if pred_path.endswith("/") else pred_path + "/"


def _extract_coords_from_text(traj_text: str) -> np.ndarray:
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
    preds = {}
    pred_path = _normalize_pred_path(pred_path)
    for data in infos:
        token = data["token"]
        pred_file = pred_path + token
        if not os.path.exists(pred_file):
            continue
        try:
            with open(pred_file, "r", encoding="utf8") as f:
                pred_data = json.load(f)
            traj_text = pred_data[0]["A"]
            coords = _extract_coords_from_text(traj_text)
            if coords.shape[0] > 0:
                preds[token] = coords
        except Exception:
            continue
    return preds


def safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


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


def bin_labels_from_edges(edges: List[float]) -> List[str]:
    n = len(edges) + 1
    lab: List[str] = []
    for i in range(n):
        if i == 0:
            lab.append(f"[0, {edges[0]:g})")
        elif i < len(edges):
            lab.append(f"[{edges[i - 1]:g}, {edges[i]:g})")
        else:
            lab.append(f"[{edges[-1]:g}, inf)")
    return lab


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    anno_file = os.path.join(args.base_path, args.anno_path)
    key_infos = pickle.load(open(anno_file, "rb"))
    infos = key_infos["infos"]
    preds = load_preds(args.pred_path, infos)
    planning_metric = PlanningMetric(args.base_path)
    ego_boxes = np.array([[0, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])

    # Load mission goals from OpenDriveVLA cached_nuscenes_info.pkl
    mission_map: Dict[str, str] = {}
    if args.cached_nusc_info and os.path.exists(args.cached_nusc_info):
        mission_map = load_cached_mission_map(args.cached_nusc_info)
        print(f"Loaded mission map: {len(mission_map)} tokens from {args.cached_nusc_info}")
    else:
        print(f"WARNING: cached_nuscenes_info.pkl not found at '{args.cached_nusc_info}'. "
              "No mission-based turn classification available.")

    edges = [float(x.strip()) for x in args.bin_edges_deg.split(",") if x.strip()]
    edges = sorted(edges)
    bin_labs = bin_labels_from_edges(edges) if args.show_bins else []

    # Per-split counters: turn_left, turn_right, keep_forward, unknown
    splits = ["turn_left", "turn_right", "all_turn", "keep_forward"]
    counters = {}
    for sp in splits:
        counters[sp] = {
            "n": 0, "sum_lt": 0, "net_lt": 0, "chord_lt": 0,
            "l2_1s_sum": 0.0, "l2_2s_sum": 0.0, "l2_3s_sum": 0.0,
            "obj_col_1s_sum": 0.0, "obj_col_2s_sum": 0.0, "obj_col_3s_sum": 0.0,
            "boundary_1s_sum": 0.0, "boundary_2s_sum": 0.0, "boundary_3s_sum": 0.0,
        }

    total_pred = 0
    no_mission = 0
    details = []
    binned_records: Dict[str, List[dict]] = {"turn_left": [], "turn_right": [], "all_turn": []}

    for data in infos:
        token = data["token"]
        if token not in preds:
            continue
        pred_xy = preds[token][:6, :2]
        gt_xy = data["gt_planning"][0, :6, :2]

        mission = mission_map.get(token, None)
        if mission is None:
            no_mission += 1
            continue

        total_pred += 1
        is_turn = mission in ("turn_left", "turn_right")

        # Compute 3 angle metrics (OpenDriveVLA style)
        sum_abs_p = sum_abs_segment_turns(pred_xy)
        sum_abs_g = sum_abs_segment_turns(gt_xy)
        net_p = net_heading_change_xy(pred_xy)
        net_g = net_heading_change_xy(gt_xy)
        chord_p = trajectory_chord_angle_xy(pred_xy)
        chord_g = trajectory_chord_angle_xy(gt_xy)

        any_nan = any(np.isnan(x) for x in (sum_abs_p, sum_abs_g, net_p, net_g, chord_p, chord_g))

        group = mission if is_turn else "keep_forward"

        def update_counter(sp: str):
            counters[sp]["n"] += 1
            if not any_nan:
                if sum_abs_p < sum_abs_g:
                    counters[sp]["sum_lt"] += 1
                if abs(net_p) < abs(net_g):
                    counters[sp]["net_lt"] += 1
                if abs(chord_p) < abs(chord_g):
                    counters[sp]["chord_lt"] += 1

        update_counter(group)
        if is_turn:
            update_counter("all_turn")

        # L2 + collision/boundary metrics
        try:
            gt_traj_full = data["gt_planning"][0, :6, :2]
            mask = data["gt_planning_mask"][0]
            fut_valid_flag = bool(mask.all())
            if fut_valid_flag:
                pred_t = torch.from_numpy(pred_xy).unsqueeze(0)
                gt_t = torch.from_numpy(gt_traj_full).unsqueeze(0)
                for sec_idx, cur_time in enumerate((2, 4, 6), start=1):
                    ade = float(
                        sum(
                            np.sqrt(
                                (pred_xy[i, 0] - gt_traj_full[i, 0]) ** 2
                                + (pred_xy[i, 1] - gt_traj_full[i, 1]) ** 2
                            )
                            for i in range(cur_time)
                        )
                        / cur_time
                    )
                    for sp_key in ([group, "all_turn"] if is_turn else [group]):
                        counters[sp_key][f"l2_{sec_idx}s_sum"] += ade

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
                pred_traj_yaw = append_tangent_directions(pred_xy)
                pred_traj_mask = np.concatenate(
                    [
                        pred_traj_yaw[..., :2].reshape(1, -1),
                        np.ones_like(pred_traj_yaw[..., :1]).reshape(1, -1),
                        pred_traj_yaw[..., 2:].reshape(1, -1),
                    ],
                    axis=-1,
                )
                ego_seg = planning_metric.get_ego_seg(ego_boxes, pred_traj_mask, add_rec=True)
                for sec_idx, cur_time in enumerate((2, 4, 6), start=1):
                    _obj_coll, obj_box_coll = planning_metric.evaluate_coll(
                        pred_t[:, :cur_time], gt_t[:, :cur_time], torch.from_numpy(bev_seg[1:]).unsqueeze(0)
                    )
                    for sp_key in ([group, "all_turn"] if is_turn else [group]):
                        counters[sp_key][f"obj_col_{sec_idx}s_sum"] += float(obj_box_coll.max().item())
                    rec_out = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[0:1] == 1)).sum() > 0
                    out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[1 : cur_time + 1] == 1)).sum() > 0
                    if out_of_drivable and (not rec_out):
                        for sp_key in ([group, "all_turn"] if is_turn else [group]):
                            counters[sp_key][f"boundary_{sec_idx}s_sum"] += 1.0
        except Exception:
            pass

        # Binned records (for turns only)
        if is_turn and not any_nan:
            sev = gt_severity_deg(gt_xy, args.bin_by)
            if not np.isnan(sev):
                rec = {
                    "sev": sev,
                    "sum_lt": sum_abs_p < sum_abs_g,
                    "net_lt": abs(net_p) < abs(net_g),
                    "chord_lt": abs(chord_p) < abs(chord_g),
                }
                binned_records[mission].append(rec)
                binned_records["all_turn"].append(rec)

        details.append({
            "token": token,
            "mission_goal": mission,
            "is_turn": is_turn,
            "sum_abs_pred_rad": float(sum_abs_p) if not np.isnan(sum_abs_p) else None,
            "sum_abs_gt_rad": float(sum_abs_g) if not np.isnan(sum_abs_g) else None,
            "net_heading_pred_rad": float(net_p) if not np.isnan(net_p) else None,
            "net_heading_gt_rad": float(net_g) if not np.isnan(net_g) else None,
            "chord_pred_rad": float(chord_p) if not np.isnan(chord_p) else None,
            "chord_gt_rad": float(chord_g) if not np.isnan(chord_g) else None,
        })

    # ── Print results ────────────────────────────────────────────────────
    n_turn_left = counters["turn_left"]["n"]
    n_turn_right = counters["turn_right"]["n"]
    n_all_turn = counters["all_turn"]["n"]
    n_forward = counters["keep_forward"]["n"]

    print("=" * 60)
    print("Turn classification: OpenDriveVLA gt_ego_fut_cmd")
    print("Understeer criteria: OpenDriveVLA 3-metric (pred < gt)")
    print("=" * 60)
    print(f"total valid predictions: {total_pred}")
    print(f"skipped (no mission goal): {no_mission}")
    print(f"turn_left: {n_turn_left}  turn_right: {n_turn_right}  all_turn: {n_all_turn}  keep_forward: {n_forward}")

    for sp in ("turn_left", "turn_right", "all_turn", "keep_forward"):
        c = counters[sp]
        n = c["n"]
        print(f"\n--- {sp.upper()} (n={n}) ---")
        if n == 0:
            continue

        if sp != "keep_forward":
            print(f"  Σ|Δθ|_pred < Σ|Δθ|_gt:     {c['sum_lt']:4d} / {n} = {100 * c['sum_lt'] / n:.2f}%")
            print(f"  |θ_net|_pred < |θ_net|_gt:  {c['net_lt']:4d} / {n} = {100 * c['net_lt'] / n:.2f}%")
            print(f"  |θ_chord|_pred < |θ_chord|_gt: {c['chord_lt']:4d} / {n} = {100 * c['chord_lt'] / n:.2f}%")

        print(f"  L2 @1s: {c['l2_1s_sum'] / n:.4f}  @2s: {c['l2_2s_sum'] / n:.4f}  @3s: {c['l2_3s_sum'] / n:.4f}")
        print(f"  ObjCol @1s: {c['obj_col_1s_sum'] * 100 / n:.4f}%  @2s: {c['obj_col_2s_sum'] * 100 / n:.4f}%  @3s: {c['obj_col_3s_sum'] * 100 / n:.4f}%")
        print(f"  Boundary @1s: {c['boundary_1s_sum'] * 100 / n:.4f}%  @2s: {c['boundary_2s_sum'] * 100 / n:.4f}%  @3s: {c['boundary_3s_sum'] * 100 / n:.4f}%")

    # ── Binned analysis ──────────────────────────────────────────────────
    if args.show_bins:
        print(f"\n{'=' * 60}")
        print(f"GT severity bins (bin-by={args.bin_by}, edges={edges} deg)")
        print(f"{'=' * 60}")

        splits_to_show = [("ALL TURN", "all_turn")]
        if args.bins_per_split:
            splits_to_show = [
                ("Turn LEFT", "turn_left"),
                ("Turn RIGHT", "turn_right"),
                ("ALL TURN", "all_turn"),
            ]

        for title, key in splits_to_show:
            rows = binned_records.get(key, [])
            if not rows:
                print(f"\n  {title}: no valid samples")
                continue
            nb = len(edges) + 1
            print(f"\n  {title} (total valid: {len(rows)})")
            for b in range(nb):
                sub = [r for r in rows if int(np.digitize(r["sev"], np.asarray(edges))) == b]
                n = len(sub)
                lab = bin_labs[b]
                if n == 0:
                    print(f"    {lab}: n=0")
                    continue
                s1 = sum(1 for r in sub if r["sum_lt"])
                s2 = sum(1 for r in sub if r["net_lt"])
                s3 = sum(1 for r in sub if r["chord_lt"])
                print(
                    f"    {lab}: n={n}  "
                    f"Σ|Δθ| under {100 * s1 / n:.1f}%  "
                    f"|θ_net| under {100 * s2 / n:.1f}%  "
                    f"|θ_chord| under {100 * s3 / n:.1f}%"
                )

    # ── Save details ─────────────────────────────────────────────────────
    if args.save_details_json:
        out = {
            "config": {
                "pred_path": args.pred_path,
                "base_path": args.base_path,
                "cached_nusc_info": args.cached_nusc_info,
                "bin_by": args.bin_by,
                "bin_edges_deg": edges,
            },
            "summary": {
                "total_pred": total_pred,
                "no_mission_skipped": no_mission,
                "counters": counters,
            },
            "details": details,
        }
        with open(args.save_details_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved details to: {args.save_details_json}")


if __name__ == "__main__":
    main()
