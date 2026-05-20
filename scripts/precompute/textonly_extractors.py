"""Section extractors for the text-only planning dataset.

Each ``extract_*`` function takes the sample *info* dict (and optionally the
lane PKL entry) and returns a plain-text block that goes into the prompt.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from textonly_geometry import (
    CAMERA_ORDER,
    camera_from_xy,
    curvature_from_points,
    curve_direction_from_points,
    estimate_lane_width,
    lane_heading_at_closest,
    normalize_angle,
    time_to_cross_ego_arc,
)

# ===================================================================
# Ego
# ===================================================================

def extract_ego(info: dict) -> str:
    """``=== EGO ===`` block from can_bus."""
    cb = np.asarray(info.get("can_bus", np.zeros(13)), dtype=np.float64).ravel()
    speed = float(cb[10]) if cb.shape[0] > 10 else 0.0
    yaw_rate = float(cb[9]) if cb.shape[0] > 9 else 0.0

    yaw_deg_per_s = math.degrees(yaw_rate)
    turn_dir = "left" if yaw_rate > 0.01 else ("right" if yaw_rate < -0.01 else "straight")

    return (
        f"Speed {speed:.1f} m/s, yaw_rate {yaw_rate:.3f} rad/s "
        f"(turning {turn_dir} ~{abs(yaw_deg_per_s):.1f} deg/s)"
    )


# ===================================================================
# Objects (per-camera grouping)
# ===================================================================

_ORIENT_THRESHOLDS = [
    (0.7, "same direction"),
    (math.pi - 0.7, "crossing"),
]


def _orient_label(heading: float) -> str:
    abs_h = abs(normalize_angle(heading))
    for thr, label in _ORIENT_THRESHOLDS:
        if abs_h < thr:
            return label
    return "oncoming"


def _state_label_from_attr(attr: str, speed: float) -> str:
    """Create a single object state label, prioritizing gt_attrs."""
    if not attr:
        return f"moving at {speed:.1f} m/s" if speed >= 0.2 else "stationary"
    a = attr.lower()
    if "parked" in a:
        return "parked"
    if "stopped" in a:
        return "stopped"
    if "moving" in a:
        return f"moving at {speed:.1f} m/s"
    if "standing" in a:
        return "standing"
    if "sitting" in a or "lying" in a:
        return "sitting/lying"
    if "with_rider" in a:
        return "with rider"
    if "without_rider" in a:
        return "without rider"
    return attr.split(".")[-1]


def _future_traj_note(
    obj_fut: Optional[np.ndarray],
    obj_xy: np.ndarray,
    obj_vel: np.ndarray,
    ego_speed: float,
    ego_yaw_rate: float,
) -> str:
    """Generate a short note about the object's predicted path risk."""
    ttc = time_to_cross_ego_arc(obj_xy, obj_vel, ego_speed, ego_yaw_rate)
    if ttc is not None:
        return f"predicted path crosses ego turn zone in ~{ttc}s"

    return ""


def extract_objects(
    info: dict,
    max_distance: float = 50.0,
    max_per_cam: int = 8,
) -> str:
    """``=== OBJECTS ===`` block grouped by camera."""
    boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))), dtype=np.float64)
    names = info.get("gt_names", [])
    attrs = info.get("gt_attrs", [])
    vel = np.asarray(info.get("gt_velocity", np.empty((0, 2))), dtype=np.float64)
    valid_flag = info.get("valid_flag", None)
    fut_traj = info.get("gt_fut_traj", None)  # (N, T, 2)

    cb = np.asarray(info.get("can_bus", np.zeros(13)), dtype=np.float64).ravel()
    ego_speed = float(cb[10]) if cb.shape[0] > 10 else 0.0
    ego_yaw_rate = float(cb[9]) if cb.shape[0] > 9 else 0.0

    n = boxes.shape[0]

    per_cam: Dict[str, List[Tuple[float, str]]] = {c: [] for c in CAMERA_ORDER}

    for i in range(n):
        if valid_flag is not None and i < len(valid_flag) and not valid_flag[i]:
            continue

        x, y = float(boxes[i, 0]), float(boxes[i, 1])
        dist = math.hypot(x, y)
        if dist > max_distance:
            continue

        heading = float(boxes[i, 6])
        name = str(names[i]) if i < len(names) else "object"
        attr = str(attrs[i]) if i < len(attrs) else ""
        orient = _orient_label(heading)

        vx = float(vel[i, 0]) if i < vel.shape[0] else 0.0
        vy = float(vel[i, 1]) if i < vel.shape[0] else 0.0
        speed = math.hypot(vx, vy)
        state_label = _state_label_from_attr(attr, speed)

        cam = camera_from_xy(x, y)

        obj_fut = None
        if fut_traj is not None and i < fut_traj.shape[0]:
            obj_fut = np.asarray(fut_traj[i], dtype=np.float64)

        fut_note = _future_traj_note(
            obj_fut,
            np.array([x, y]),
            np.array([vx, vy]),
            ego_speed,
            ego_yaw_rate,
        )

        parts = [f"{name} at ({x:.1f}, {y:.1f}), {dist:.1f}m, {orient}"]
        parts.append(state_label)
        if fut_note:
            parts.append(f"— {fut_note}")

        desc = ", ".join(parts)
        per_cam[cam].append((dist, desc))

    lines = []
    for cam in CAMERA_ORDER:
        entries = sorted(per_cam[cam], key=lambda z: z[0])[:max_per_cam]
        if not entries:
            lines.append(f"{cam}: none")
        else:
            cam_lines = [f"- {e[1]}" for e in entries]
            lines.append(f"{cam}:\n" + "\n".join(cam_lines))

    return "\n".join(lines)


# ===================================================================
# Map
# ===================================================================

def _summarize_traffic_elements(lane_entry: dict) -> List[str]:
    """Traffic lights / road signs from lane annotation.

    Traffic element points are in global frame and need world→ego conversion
    via the lane entry's ego2global pose.
    """
    ann = lane_entry.get("annotation", {})
    tes = ann.get("traffic_element", [])
    if not tes:
        return []
    pose = lane_entry.get("pose", {})
    rot = np.asarray(pose.get("rotation", np.eye(3)), dtype=np.float64).reshape(3, 3)
    trans = np.asarray(pose.get("translation", np.zeros(3)), dtype=np.float64).reshape(3)

    parts: List[str] = []
    tl_dists: List[float] = []
    rs_dists: List[float] = []

    for te in tes:
        pts = np.asarray(te.get("points", []), dtype=np.float64)
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 2:
            continue
        center_world = pts[:, :2].mean(axis=0)
        diff = center_world - trans[:2]
        center_ego = rot[:2, :2].T @ diff
        ex, ey = float(center_ego[0]), float(center_ego[1])
        if ex < -5.0 or ex > 80.0 or abs(ey) > 25.0:
            continue
        d = math.hypot(ex, ey)
        cat = int(te.get("category", -1))
        if cat == 1:
            tl_dists.append(d)
        elif cat == 2:
            rs_dists.append(d)

    if tl_dists:
        tl_dists.sort()
        parts.append(f"Traffic light at {tl_dists[0]:.0f}m ahead (state unknown)")
    if rs_dists:
        rs_dists.sort()
        parts.append(f"Road sign at {rs_dists[0]:.0f}m ahead")

    return parts


def _geom_to_points(g) -> Optional[np.ndarray]:
    """Extract Nx2 coordinate array from a shapely geometry or np.ndarray."""
    if isinstance(g, np.ndarray):
        pts = g.astype(np.float64)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            return pts[:, :2]
        return None
    if hasattr(g, "exterior"):  # Polygon
        coords = np.asarray(g.exterior.coords, dtype=np.float64)
        return coords[:, :2] if coords.shape[1] >= 2 else None
    if hasattr(g, "coords"):  # LineString / LinearRing
        coords = np.asarray(g.coords, dtype=np.float64)
        return coords[:, :2] if coords.ndim == 2 and coords.shape[1] >= 2 else None
    return None


def _map_geoms_text(info: dict) -> List[str]:
    """Extract ped crossing / divider / boundary from map_geoms (MapTR format).

    Keys: 0=ped_crossing, 1=divider, 2=boundary.
    Values are lists of shapely LineString/Polygon (ego frame).
    """
    mg = info.get("map_geoms", None)
    if not mg or not isinstance(mg, dict):
        return []

    label_map = {0: "Crosswalk", 1: "Divider", 2: "Road boundary"}
    parts: List[str] = []

    for key in sorted(label_map.keys()):
        geoms = mg.get(key, [])
        if not geoms:
            continue
        label = label_map[key]
        min_dist = float("inf")
        direction = ""
        for g in geoms:
            pts_2d = _geom_to_points(g)
            if pts_2d is None or pts_2d.shape[0] == 0:
                continue
            dists = np.linalg.norm(pts_2d, axis=1)
            d = float(dists.min())
            if d < min_dist:
                min_dist = d
                closest_idx = int(np.argmin(dists))
                cx, cy = float(pts_2d[closest_idx, 0]), float(pts_2d[closest_idx, 1])
                if cx > 2.0:
                    direction = "ahead"
                elif cx < -2.0:
                    direction = "behind"
                else:
                    direction = "left" if cy > 0 else "right"
        if min_dist < 50.0:
            parts.append(f"{label} at {min_dist:.0f}m {direction}")

    return parts


def extract_map(info: dict, lane_entry: Optional[dict]) -> str:
    """``=== MAP ===`` block combining lane PKL + map_geoms."""
    sections: List[str] = []

    # --- lane centerlines from lane PKL ---
    # NOTE: lane centerline points are already in ego frame in this PKL.
    if lane_entry:
        ann = lane_entry.get("annotation", {})
        lanes = ann.get("lane_centerline", [])

        lane_egos: List[Tuple[float, str, bool, float, Optional[float], np.ndarray]] = []
        connector_infos: List[Tuple[float, float, str]] = []

        for lane in lanes:
            pts = np.asarray(lane.get("points", []), dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            pts_ego = pts[:, :2]
            dist = float(np.min(np.linalg.norm(pts_ego, axis=1)))
            if dist > 30.0:
                continue
            is_conn = bool(lane.get("is_intersection_or_connector", False))
            heading = lane_heading_at_closest(pts_ego, flip_backward=not is_conn)
            kappa = curvature_from_points(pts_ego)
            direction = curve_direction_from_points(pts_ego)
            lane_egos.append((dist, direction, is_conn, heading, kappa, pts_ego))

            if is_conn:
                fwd = pts_ego[pts_ego[:, 0] > 0.0]
                if fwd.shape[0] > 0:
                    start_d = float(np.min(np.linalg.norm(fwd, axis=1)))
                    connector_infos.append((start_d, heading, direction))

        if lane_egos:
            lane_egos.sort(key=lambda z: z[0])
            nd, ncurve, _, nh, nk, nxy = lane_egos[0]
            nearby = sum(1 for d, *_ in lane_egos if d < 15.0)
            width = estimate_lane_width(nxy, [xy for *_, xy in lane_egos[1:]])

            ego_lane_parts = [f"Ego-nearest lane: distance {nd:.1f}m, {ncurve}"]
            if nk is not None:
                ego_lane_parts.append(f"(κ={nk:.3f}/m)")
            sections.append(" ".join(ego_lane_parts))

            width_str = f", width ~{width:.1f}m" if width else ""
            sections.append(f"{nearby} lane centerlines within 15m{width_str}")

            if connector_infos:
                connector_infos.sort(key=lambda z: z[0])
                cd, ch, cdir = connector_infos[0]
                sections.append(
                    f"Connector ahead: starts {cd:.1f}m, {cdir} to heading {ch:.2f} rad"
                )

        sections.extend(_summarize_traffic_elements(lane_entry))
    else:
        sections.append("Lane annotation unavailable")

    # --- map_geoms (ped crossing / divider / boundary) ---
    sections.extend(_map_geoms_text(info))

    return "\n".join(sections) if sections else "Map data unavailable"


# ===================================================================
# Mission
# ===================================================================

def extract_mission(info: dict) -> str:
    """``=== MISSION ===`` block — use gt_planning_command_desc directly."""
    desc = info.get("gt_planning_command_desc", "")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    cmd = info.get("gt_planning_command", None)
    if cmd is not None:
        cmd_int = int(np.asarray(cmd).ravel()[0]) if hasattr(cmd, '__len__') else int(cmd)
        mapping = {0: "Turn right", 1: "Turn left", 2: "Go straight"}
        return mapping.get(cmd_int, "Unknown command")
    return "Unknown"


# ===================================================================
# Scene description / metadata
# ===================================================================

def extract_scene_header(info: dict) -> str:
    """Location + scene description if available."""
    loc = info.get("location", "unknown")
    desc = info.get("description", "")
    parts = [f"Location: {loc}"]
    if desc:
        parts.append(f"Context: {desc}")
    return "\n".join(parts)


# ===================================================================
# Full prompt assembly
# ===================================================================

def build_full_prompt(info: dict, lane_entry: Optional[dict]) -> str:
    """Assemble the complete multi-section prompt string."""
    blocks = [
        ("SCENE", extract_scene_header(info)),
        ("EGO", extract_ego(info)),
        ("OBJECTS", extract_objects(info)),
        ("MAP", extract_map(info, lane_entry)),
        ("MISSION", extract_mission(info)),
    ]
    parts = []
    for title, body in blocks:
        parts.append(f"=== {title} ===\n{body}")
    return "\n\n".join(parts)
