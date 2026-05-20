#!/usr/bin/env python3
"""Build text-only planning dataset from nuScenes annotation PKL.

This converts each sample into:
1) A natural-language scene summary (ego/object/map/mission)
2) A GT trajectory target text in ego XY coordinates

The output JSONL can be used for text-only trajectory prediction experiments.
"""

import argparse
import json
import math
import os
import pickle
from io import BufferedReader
from typing import Dict, List, Optional, Tuple

import numpy as np


CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


class _PickleFallbackObject:
    """Placeholder object for unavailable optional classes in pickles."""

    def __init__(self, *args, **kwargs):
        self._state = {}

    def __setstate__(self, state):
        self._state = state


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that tolerates missing optional deps (e.g., shapely)."""

    def find_class(self, module, name):
        if module.startswith("shapely."):
            return _PickleFallbackObject
        return super().find_class(module, name)


def safe_pickle_load(fp: BufferedReader):
    try:
        return pickle.load(fp)
    except ModuleNotFoundError as e:
        # Retry with fallback resolver for optional packages serialized in PKL.
        if "shapely" not in str(e):
            raise
        fp.seek(0)
        return _SafeUnpickler(fp).load()


def mission_goal_from_cmd(cmd_vec) -> str:
    right, left, forward = cmd_vec
    if right > 0:
        return "turn_right"
    if left > 0:
        return "turn_left"
    return "keep_forward"


def load_cached_mission_map(cached_path: Optional[str]) -> Dict[str, str]:
    if not cached_path or not os.path.exists(cached_path):
        return {}
    with open(cached_path, "rb") as f:
        cache = safe_pickle_load(f)
    mission_map: Dict[str, str] = {}
    for token, value in cache.items():
        cmd_vec = value.get("gt_ego_fut_cmd")
        if cmd_vec is None or len(cmd_vec) < 3:
            continue
        mission_map[str(token)] = mission_goal_from_cmd(cmd_vec)
    return mission_map


def format_xy_points(xy: np.ndarray, digits: int = 2) -> str:
    pts = []
    for x, y in xy:
        pts.append(f"({x:.{digits}f}, {y:.{digits}f})")
    return "[" + ", ".join(pts) + "]"


def infer_ego_state_from_can_bus(
    can_bus: np.ndarray,
    speed_idx: int = 10,
    yaw_rate_idx: int = 9,
) -> Tuple[float, float]:
    """Infer ego speed/yaw-rate from current can_bus features only."""
    cb = np.asarray(can_bus, dtype=np.float32).reshape(-1)
    speed = float(cb[speed_idx]) if speed_idx < cb.shape[0] else 0.0
    yaw_rate = float(cb[yaw_rate_idx]) if yaw_rate_idx < cb.shape[0] else 0.0
    return speed, yaw_rate


def _curve_label_from_centerline(pts_xy: np.ndarray) -> str:
    if pts_xy.shape[0] < 2:
        return "unknown"
    order = np.argsort(pts_xy[:, 0])
    pts = pts_xy[order]
    x = pts[:, 0]
    y = pts[:, 1]

    # Focus on forward region in ego frame.
    mask = (x >= -2.0) & (x <= 35.0)
    if mask.sum() >= 2:
        x = x[mask]
        y = y[mask]

    dy = float(y[-1] - y[0])
    if abs(dy) < 1.0:
        return "mostly straight"
    if dy > 0:
        return "curving left"
    return "curving right"


def _normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _lane_heading_from_points(pts_xy: np.ndarray) -> float:
    if pts_xy.shape[0] < 2:
        return 0.0
    i0 = int(np.argmin(np.linalg.norm(pts_xy, axis=1)))
    i1 = min(i0 + 1, pts_xy.shape[0] - 1)
    if i1 == i0:
        i1 = max(i0 - 1, 0)
    vec = pts_xy[i1] - pts_xy[i0]
    return float(math.atan2(vec[1], vec[0]))


def _curvature_radius_from_centerline(pts_xy: np.ndarray) -> Optional[float]:
    if pts_xy.shape[0] < 3:
        return None
    order = np.argsort(pts_xy[:, 0])
    pts = pts_xy[order]
    x = pts[:, 0]
    y = pts[:, 1]
    mask = (x >= 0.0) & (x <= 25.0)
    if mask.sum() < 3:
        return None
    x = x[mask]
    y = y[mask]
    try:
        a, b, _ = np.polyfit(x, y, 2)
    except Exception:
        return None
    kappa = abs(2.0 * a) / max((1.0 + b * b) ** 1.5, 1e-6)
    if kappa < 1e-4:
        return None
    return float(1.0 / kappa)


def _estimate_lane_width(ego_lane_xy: np.ndarray, lane_xy_list: List[np.ndarray]) -> Optional[float]:
    if ego_lane_xy.shape[0] == 0:
        return None
    h0 = _lane_heading_from_points(ego_lane_xy)
    best = None
    for other in lane_xy_list:
        if other.shape[0] == 0:
            continue
        h1 = _lane_heading_from_points(other)
        dh = abs(_normalize_angle(h1 - h0))
        if dh > 0.45:
            continue
        d = float(np.min(np.linalg.norm(ego_lane_xy[:, None, :] - other[None, :, :], axis=-1)))
        if d < 2.0 or d > 6.0:
            continue
        if best is None or d < best:
            best = d
    return best


def _world_to_ego_xy(points_xy: np.ndarray, lane_entry: dict) -> np.ndarray:
    pose = lane_entry.get("pose", {})
    rot = np.asarray(pose.get("rotation", np.eye(3)), dtype=np.float32)
    trans = np.asarray(pose.get("translation", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
    if rot.shape != (3, 3):
        rot = np.eye(3, dtype=np.float32)
    if trans.shape[0] < 2:
        trans = np.zeros((3,), dtype=np.float32)
    pts3 = np.concatenate([points_xy, np.zeros((points_xy.shape[0], 1), dtype=np.float32)], axis=1)
    ego = (rot.T @ (pts3 - trans.reshape(1, 3)).T).T
    return ego[:, :2]


def _summarize_traffic_elements(lane_entry: dict) -> str:
    ann = lane_entry.get("annotation", {})
    tes = ann.get("traffic_element", [])
    if not tes:
        return "Traffic control: none annotated nearby"

    traffic_lights: List[Tuple[float, int]] = []
    road_signs: List[Tuple[float, int]] = []
    for te in tes:
        points = np.asarray(te.get("points", []), dtype=np.float32)
        if points.size == 0:
            continue
        if points.ndim == 2 and points.shape[1] >= 2:
            center_world = points[:, :2].mean(axis=0, keepdims=True)
        else:
            continue
        center_ego = _world_to_ego_xy(center_world, lane_entry)[0]
        x, y = float(center_ego[0]), float(center_ego[1])
        if x < 0.0 or x > 80.0 or abs(y) > 25.0:
            continue
        d = math.hypot(x, y)
        cat = int(te.get("category", -1))
        attr = int(te.get("attribute", -1))
        if cat == 1:
            traffic_lights.append((d, attr))
        elif cat == 2:
            road_signs.append((d, attr))

    parts = []
    if traffic_lights:
        traffic_lights.sort(key=lambda z: z[0])
        nearest = traffic_lights[0]
        parts.append(f"traffic_lights ahead {len(traffic_lights)} (nearest {nearest[0]:.1f}m, attr {nearest[1]})")
    if road_signs:
        road_signs.sort(key=lambda z: z[0])
        nearest = road_signs[0]
        parts.append(f"road_signs ahead {len(road_signs)} (nearest {nearest[0]:.1f}m, attr {nearest[1]})")
    if not parts:
        return "Traffic control: none in front corridor"
    return "Traffic control: " + "; ".join(parts)


def describe_map_from_lane_entry(lane_entry: Optional[dict]) -> str:
    """Map summary from current-frame lane annotation (no GT future usage)."""
    if not lane_entry:
        return "lane annotation unavailable"
    ann = lane_entry.get("annotation", {})
    lanes = ann.get("lane_centerline", [])
    if not lanes:
        return "no lane centerline nearby"

    lane_stats = []
    all_lane_xy: List[np.ndarray] = []
    connector_info: List[Tuple[float, float]] = []
    for lane in lanes:
        pts = np.asarray(lane.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        pts_xy = pts[:, :2]
        all_lane_xy.append(pts_xy)
        dist = float(np.min(np.linalg.norm(pts_xy, axis=1)))
        curve = _curve_label_from_centerline(pts_xy)
        is_conn = bool(lane.get("is_intersection_or_connector", False))
        heading = _lane_heading_from_points(pts_xy)
        radius = _curvature_radius_from_centerline(pts_xy)
        lane_stats.append((dist, curve, is_conn, heading, radius, pts_xy))
        if is_conn:
            forward_pts = pts_xy[pts_xy[:, 0] > 0.0]
            if forward_pts.shape[0] > 0:
                start_d = float(np.min(np.linalg.norm(forward_pts, axis=1)))
                connector_info.append((start_d, heading))

    if not lane_stats:
        return "lane geometry unavailable"

    lane_stats.sort(key=lambda z: z[0])
    nearest_dist, nearest_curve, _is_conn, nearest_heading, nearest_radius, nearest_xy = lane_stats[0]
    nearby_count = sum(1 for d, *_ in lane_stats if d < 15.0)
    width = _estimate_lane_width(nearest_xy, [xy for *_, xy in lane_stats[1:]])
    lane_text = (
        f"ego-nearest lane {nearest_curve}, heading {nearest_heading:.2f} rad, "
        f"distance {nearest_dist:.1f}m"
    )
    if nearest_radius is not None:
        lane_text += f", radius ~{nearest_radius:.1f}m over next 25m"
    lane_text += f"; {nearby_count} lane centerlines within 15m"
    if width is not None:
        lane_text += f"; estimated lane width {width:.1f}m"

    if connector_info:
        connector_info.sort(key=lambda z: z[0])
        d0, h0 = connector_info[0]
        connector_text = f"nearest connector starts at {d0:.1f}m ahead, heading {h0:.2f} rad"
    else:
        connector_text = "no connector lane in front corridor"

    traffic_text = _summarize_traffic_elements(lane_entry)
    return lane_text + "; " + connector_text + "; " + traffic_text


def mission_to_text(mission: Optional[str], command_desc: Optional[str]) -> str:
    if mission == "turn_left":
        return "Mission: turn left at next intersection"
    if mission == "turn_right":
        return "Mission: turn right at next intersection"
    if mission == "keep_forward":
        return "Mission: keep forward"

    if command_desc and isinstance(command_desc, str) and command_desc.strip():
        desc = command_desc.strip()
        if not desc.lower().startswith("mission"):
            return f"Mission: {desc}"
        return desc
    return "Mission: unknown"


def summarize_objects(
    info: dict,
    top_k: int = 8,
    max_distance: float = 40.0,
) -> str:
    boxes = info.get("gt_boxes", None)
    if boxes is None or len(boxes) == 0:
        return "Objects: none"

    names = info.get("gt_names", None)
    vel = info.get("gt_velocity", None)

    candidates: List[Tuple[float, str]] = []
    for i in range(len(boxes)):
        x = float(boxes[i][0])
        y = float(boxes[i][1])
        dist = math.hypot(x, y)
        if dist > max_distance:
            continue

        name = "object"
        if names is not None and i < len(names):
            name = str(names[i])

        speed = None
        vx, vy = 0.0, 0.0
        if vel is not None and i < len(vel):
            vx = float(vel[i][0])
            vy = float(vel[i][1])
            speed = math.hypot(vx, vy)

        heading = float(boxes[i][6]) if len(boxes[i]) > 6 else 0.0
        length = float(boxes[i][3]) if len(boxes[i]) > 3 else 0.0
        width = float(boxes[i][4]) if len(boxes[i]) > 4 else 0.0
        if abs(_normalize_angle(heading)) < 0.7:
            orient = "same_direction"
        elif abs(abs(_normalize_angle(heading)) - math.pi) < 0.7:
            orient = "oncoming"
        else:
            orient = "crossing"
        if speed is None:
            desc = (
                f"{name} at ({x:.1f}, {y:.1f}), heading {heading:.2f} rad ({orient}), "
                f"size {length:.1f}x{width:.1f}m"
            )
        elif speed < 0.2:
            desc = (
                f"{name} at ({x:.1f}, {y:.1f}), heading {heading:.2f} rad ({orient}), "
                f"velocity ({vx:.1f}, {vy:.1f}) m/s, stationary, size {length:.1f}x{width:.1f}m"
            )
        else:
            desc = (
                f"{name} at ({x:.1f}, {y:.1f}), heading {heading:.2f} rad ({orient}), "
                f"velocity ({vx:.1f}, {vy:.1f}) m/s, speed {speed:.1f} m/s, size {length:.1f}x{width:.1f}m"
            )
        candidates.append((dist, desc))

    if not candidates:
        return "Objects: none nearby"

    candidates.sort(key=lambda x: x[0])
    lines = [d for _, d in candidates[:top_k]]
    return "Objects: " + "; ".join(lines)


def camera_from_xy(x: float, y: float) -> str:
    """Assign object to one of 6 camera sectors using ego-frame azimuth."""
    az_deg = math.degrees(math.atan2(y, x))  # 0: front, +left
    if -30.0 <= az_deg < 30.0:
        return "CAM_FRONT"
    if 30.0 <= az_deg < 90.0:
        return "CAM_FRONT_LEFT"
    if 90.0 <= az_deg < 150.0:
        return "CAM_BACK_LEFT"
    if az_deg >= 150.0 or az_deg < -150.0:
        return "CAM_BACK"
    if -150.0 <= az_deg < -90.0:
        return "CAM_BACK_RIGHT"
    return "CAM_FRONT_RIGHT"


def summarize_objects_multicam(
    info: dict,
    max_objects_per_camera: int = 6,
    max_distance: float = 50.0,
) -> str:
    boxes = info.get("gt_boxes", None)
    if boxes is None or len(boxes) == 0:
        return "Per-camera observations:\n" + "\n".join([f"{cam}: none" for cam in CAMERA_ORDER])

    names = info.get("gt_names", None)
    vel = info.get("gt_velocity", None)
    per_cam: Dict[str, List[Tuple[float, str]]] = {cam: [] for cam in CAMERA_ORDER}

    for i in range(len(boxes)):
        x = float(boxes[i][0])
        y = float(boxes[i][1])
        dist = math.hypot(x, y)
        if dist > max_distance:
            continue
        cam = camera_from_xy(x, y)

        name = "object"
        if names is not None and i < len(names):
            name = str(names[i])

        speed = None
        vx, vy = 0.0, 0.0
        if vel is not None and i < len(vel):
            vx = float(vel[i][0])
            vy = float(vel[i][1])
            speed = math.hypot(vx, vy)

        heading = float(boxes[i][6]) if len(boxes[i]) > 6 else 0.0
        length = float(boxes[i][3]) if len(boxes[i]) > 3 else 0.0
        width = float(boxes[i][4]) if len(boxes[i]) > 4 else 0.0
        if abs(_normalize_angle(heading)) < 0.7:
            orient = "same_direction"
        elif abs(abs(_normalize_angle(heading)) - math.pi) < 0.7:
            orient = "oncoming"
        else:
            orient = "crossing"

        if speed is None:
            desc = (
                f"{name} at ({x:.1f}, {y:.1f}), {dist:.1f}m, heading {heading:.2f} rad ({orient}), "
                f"size {length:.1f}x{width:.1f}m"
            )
        elif speed < 0.2:
            desc = (
                f"{name} at ({x:.1f}, {y:.1f}), {dist:.1f}m, heading {heading:.2f} rad ({orient}), "
                f"velocity ({vx:.1f}, {vy:.1f}) m/s, stationary, size {length:.1f}x{width:.1f}m"
            )
        else:
            desc = (
                f"{name} at ({x:.1f}, {y:.1f}), {dist:.1f}m, heading {heading:.2f} rad ({orient}), "
                f"velocity ({vx:.1f}, {vy:.1f}) m/s, speed {speed:.1f} m/s, size {length:.1f}x{width:.1f}m"
            )
        per_cam[cam].append((dist, desc))

    lines = []
    for cam in CAMERA_ORDER:
        rows = sorted(per_cam[cam], key=lambda z: z[0])[:max_objects_per_camera]
        if not rows:
            lines.append(f"{cam}: none")
        else:
            lines.append(f"{cam}: " + "; ".join([r[1] for r in rows]))
    return "Per-camera observations:\n" + "\n".join(lines)


def build_prompt(
    info: dict,
    lane_entry: Optional[dict],
    mission_map: Dict[str, str],
    top_k_objects: int,
    max_obj_distance: float,
    camera_text: bool,
    max_objects_per_camera: int,
    speed_idx: int,
    yaw_rate_idx: int,
) -> str:
    token = str(info["token"])
    mission = mission_map.get(token)
    command_desc = info.get("gt_planning_command_desc", None)
    mission_text = mission_to_text(mission, command_desc)

    speed, yaw_rate = infer_ego_state_from_can_bus(
        info.get("can_bus", np.zeros((13,), dtype=np.float32)),
        speed_idx=speed_idx,
        yaw_rate_idx=yaw_rate_idx,
    )
    ego_line = f"Ego state: speed {speed:.1f} m/s, heading_rate {yaw_rate:.2f} rad/s"
    if camera_text:
        obj_line = summarize_objects_multicam(
            info,
            max_objects_per_camera=max_objects_per_camera,
            max_distance=max_obj_distance,
        )
    else:
        obj_line = summarize_objects(info, top_k=top_k_objects, max_distance=max_obj_distance)
    map_line = "Map: " + describe_map_from_lane_entry(lane_entry)

    return "\n".join([ego_line, obj_line, map_line, mission_text])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=str, required=True, help="nuScenes base directory")
    parser.add_argument(
        "--anno-path",
        type=str,
        default="nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="annotation PKL filename under base-path",
    )
    parser.add_argument(
        "--cached-nusc-info",
        type=str,
        default="data/nuscenes/cached_nuscenes_info.pkl",
        help="cached_nuscenes_info.pkl for OpenDriveVLA mission labels",
    )
    parser.add_argument(
        "--lane-path",
        type=str,
        default=None,
        help="lane annotation pickle (default: <base-path>/data_dict_sample.pkl)",
    )
    parser.add_argument("--output", type=str, required=True, help="output JSONL path")
    parser.add_argument("--top-k-objects", type=int, default=8, help="max objects in text summary")
    parser.add_argument(
        "--camera-text",
        action="store_true",
        help="emit 6-camera textual observations (CAM_FRONT/...) instead of one merged object line",
    )
    parser.add_argument(
        "--max-objects-per-camera",
        type=int,
        default=6,
        help="max objects listed per camera when --camera-text is set",
    )
    parser.add_argument("--max-object-distance", type=float, default=40.0, help="object distance threshold (m)")
    parser.add_argument(
        "--turn-only",
        action="store_true",
        help="keep only turn_left / turn_right samples by mission goal",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="max number of output samples (0 means no limit)",
    )
    parser.add_argument(
        "--speed-index",
        type=int,
        default=10,
        help="can_bus index used as ego speed [m/s]",
    )
    parser.add_argument(
        "--yaw-rate-index",
        type=int,
        default=9,
        help="can_bus index used as ego heading rate [rad/s]",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    anno_file = os.path.join(args.base_path, args.anno_path)
    with open(anno_file, "rb") as f:
        key_infos = safe_pickle_load(f)
    infos = key_infos["infos"]
    mission_map = load_cached_mission_map(args.cached_nusc_info)
    lane_path = args.lane_path or os.path.join(args.base_path, "data_dict_sample.pkl")
    lane_map = {}
    if os.path.exists(lane_path):
        with open(lane_path, "rb") as f:
            lane_map = safe_pickle_load(f)
    else:
        print(f"WARNING: lane annotation not found at {lane_path}. Map text will be unavailable.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

    num_total = 0
    num_written = 0
    num_skip_mask = 0
    num_skip_turn_filter = 0
    num_no_mission = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for info in infos:
            num_total += 1
            token = str(info["token"])
            gt = info.get("gt_planning", None)
            mask = info.get("gt_planning_mask", None)
            if gt is None or mask is None:
                continue

            gt_xy = np.asarray(gt[0, :6, :2], dtype=np.float32)
            valid_mask = np.asarray(mask[0]).astype(bool)
            if not valid_mask.all():
                num_skip_mask += 1
                continue

            mission = mission_map.get(token)
            if mission is None:
                num_no_mission += 1
            if args.turn_only and mission not in ("turn_left", "turn_right"):
                num_skip_turn_filter += 1
                continue

            prompt = build_prompt(
                info=info,
                lane_entry=lane_map.get(info.get("lane_info")) if isinstance(lane_map, dict) else None,
                mission_map=mission_map,
                top_k_objects=args.top_k_objects,
                max_obj_distance=args.max_object_distance,
                camera_text=args.camera_text,
                max_objects_per_camera=args.max_objects_per_camera,
                speed_idx=args.speed_index,
                yaw_rate_idx=args.yaw_rate_index,
            )

            traj_target_text = format_xy_points(gt_xy, digits=2)

            sample = {
                "token": token,
                "location": info.get("location", ""),
                "mission_goal": mission,
                "prompt": prompt,
                "target_trajectory_xy": gt_xy.tolist(),
                "target_trajectory_text": traj_target_text,
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            num_written += 1
            if args.max_samples > 0 and num_written >= args.max_samples:
                break

    print("=" * 72)
    print("Built text-only planning dataset")
    print("=" * 72)
    print(f"total infos: {num_total}")
    print(f"written samples: {num_written}")
    print(f"skipped (invalid planning mask): {num_skip_mask}")
    if args.turn_only:
        print(f"skipped (non-turn by mission): {num_skip_turn_filter}")
    print(f"samples without mission label: {num_no_mission}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
