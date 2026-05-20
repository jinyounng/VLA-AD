"""Geometry helpers for text-only planning dataset builder.

Coordinate convention (ego frame):
  x = forward, y = left, z = up
  yaw positive = counter-clockwise (left turn)
"""

import math
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    a = a % (2.0 * math.pi)
    if a >= math.pi:
        a -= 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def world_to_ego_points(
    points_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Transform Nx2 or Nx3 world-frame points into ego frame.

    Parameters
    ----------
    points_world : (N, 2) or (N, 3)
    rotation : (3, 3)  ego2global rotation matrix
    translation : (3,) ego2global translation
    """
    pts = np.asarray(points_world, dtype=np.float64)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)

    ndim = pts.shape[-1]
    if ndim == 2:
        pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float64)], axis=1)

    ego = (rot.T @ (pts - trans).T).T
    return ego[:, :ndim]


def world_to_ego_points_from_pose_dict(
    points_xy: np.ndarray,
    pose: dict,
) -> np.ndarray:
    """Convenience wrapper using the ``pose`` dict from lane PKL entries."""
    rot = np.asarray(pose.get("rotation", np.eye(3)), dtype=np.float64)
    trans = np.asarray(pose.get("translation", np.zeros(3)), dtype=np.float64)
    if rot.shape != (3, 3):
        rot = np.eye(3, dtype=np.float64)
    return world_to_ego_points(points_xy, rot, trans)


# ---------------------------------------------------------------------------
# Camera assignment
# ---------------------------------------------------------------------------

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)

_CAM_AZIMUTH_RANGES = [
    ("CAM_FRONT",       -30.0,  30.0),
    ("CAM_FRONT_RIGHT", -90.0, -30.0),
    ("CAM_BACK_RIGHT", -150.0, -90.0),
    # CAM_BACK handled as default (>=150 or <-150)
    ("CAM_BACK_LEFT",   90.0, 150.0),
    ("CAM_FRONT_LEFT",  30.0,  90.0),
]


def camera_from_xy(x: float, y: float) -> str:
    """Assign ego-frame point to one of 6 nuScenes cameras.

    Azimuth = atan2(-y, x) because y-left but azimuth CW from front.
    Using atan2(y, x) with x-forward, y-left:
      Front ~ 0°, Front-Left ~ +60°, Back-Left ~ +120°, Back ~ ±180°
      Front-Right ~ -60°, Back-Right ~ -120°
    """
    az_deg = math.degrees(math.atan2(y, x))
    for cam, lo, hi in _CAM_AZIMUTH_RANGES:
        if lo <= az_deg < hi:
            return cam
    return "CAM_BACK"


# ---------------------------------------------------------------------------
# Curvature from polyfit
# ---------------------------------------------------------------------------

def curvature_from_points(pts_xy: np.ndarray, x_lo: float = 0.0, x_hi: float = 25.0) -> Optional[float]:
    """Estimate curvature kappa (1/m) via 2nd-order polyfit on ego-frame pts.

    Returns None when insufficient points or kappa < 5e-3 (near-straight).
    """
    if pts_xy.shape[0] < 3:
        return None
    order = np.argsort(pts_xy[:, 0])
    pts = pts_xy[order]
    mask = (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
    if mask.sum() < 3:
        return None
    x, y = pts[mask, 0], pts[mask, 1]
    try:
        a, b, _ = np.polyfit(x, y, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    kappa = abs(2.0 * a) / max((1.0 + b * b) ** 1.5, 1e-9)
    if kappa < 5e-3:
        return None
    return float(kappa)


def curve_direction_from_points(pts_xy: np.ndarray) -> str:
    """Return 'curving left', 'curving right', or 'near-straight'."""
    if pts_xy.shape[0] < 2:
        return "unknown"
    order = np.argsort(pts_xy[:, 0])
    pts = pts_xy[order]
    mask = (pts[:, 0] >= -2.0) & (pts[:, 0] <= 35.0)
    if mask.sum() >= 2:
        pts = pts[mask]
    dy = float(pts[-1, 1] - pts[0, 1])
    if abs(dy) < 1.0:
        return "near-straight"
    # Project convention: +y means right side.
    return "curving right" if dy > 0 else "curving left"


# ---------------------------------------------------------------------------
# Lane heading
# ---------------------------------------------------------------------------

def lane_heading_at_closest(pts_xy: np.ndarray, flip_backward: bool = True) -> float:
    """Heading (rad) of the lane at the point closest to ego origin.

    If ``flip_backward`` and the resulting vector points behind ego (x<0),
    flip by pi so it points forward.  Connector lanes should pass
    ``flip_backward=False``.
    """
    if pts_xy.shape[0] < 2:
        return 0.0
    dists = np.linalg.norm(pts_xy, axis=1)
    i0 = int(np.argmin(dists))
    i1 = min(i0 + 1, pts_xy.shape[0] - 1)
    if i1 == i0:
        i1 = max(i0 - 1, 0)
    vec = pts_xy[i1] - pts_xy[i0]
    heading = math.atan2(vec[1], vec[0])
    if flip_backward and vec[0] < 0:
        heading = normalize_angle(heading + math.pi)
    return float(heading)


# ---------------------------------------------------------------------------
# Estimated lane width
# ---------------------------------------------------------------------------

def estimate_lane_width(
    ego_lane_xy: np.ndarray,
    other_lanes: list,
) -> Optional[float]:
    """Heuristic lane-width from adjacent parallel lanes."""
    if ego_lane_xy.shape[0] == 0 or not other_lanes:
        return None
    h0 = lane_heading_at_closest(ego_lane_xy)
    best = None
    for other in other_lanes:
        if other.shape[0] == 0:
            continue
        h1 = lane_heading_at_closest(other)
        if abs(normalize_angle(h1 - h0)) > 0.45:
            continue
        d = float(np.min(np.linalg.norm(
            ego_lane_xy[:, None, :] - other[None, :, :], axis=-1
        )))
        if 2.0 < d < 6.0 and (best is None or d < best):
            best = d
    return best


# ---------------------------------------------------------------------------
# Kinematic baseline (constant turn-rate extrapolation)
# ---------------------------------------------------------------------------

def kinematic_baseline(
    speed: float,
    yaw_rate: float,
    n_steps: int = 6,
    dt: float = 0.5,
) -> np.ndarray:
    """Constant-speed constant-yaw-rate extrapolation in ego frame.

    Returns (n_steps, 2) trajectory.
    """
    traj = np.zeros((n_steps, 2), dtype=np.float64)
    x, y, theta = 0.0, 0.0, 0.0
    for t in range(n_steps):
        theta += yaw_rate * dt
        x += speed * math.cos(theta) * dt
        y += speed * math.sin(theta) * dt
        traj[t] = [x, y]
    return traj


# ---------------------------------------------------------------------------
# Turn severity from GT trajectory
# ---------------------------------------------------------------------------

_TURN_BUCKETS = [
    (5.0, "straight"),
    (15.0, "mild_turn"),
    (30.0, "moderate_turn"),
    (60.0, "sharp_turn"),
]


def turn_severity(gt_xy: np.ndarray) -> Tuple[float, str]:
    """Compute yaw change (degrees) between first and last segment of GT traj.

    Returns (yaw_change_deg, bucket_label).
    Segments shorter than *min_seg* are skipped to avoid noise from near-zero
    displacement (e.g., stationary vehicle).
    """
    if gt_xy.shape[0] < 2:
        return 0.0, "straight"

    min_seg = 0.05  # metres — ignore sub-5cm steps
    d = gt_xy[1:] - gt_xy[:-1]
    lengths = np.linalg.norm(d, axis=1)
    valid = lengths > min_seg
    if valid.sum() < 2:
        return 0.0, "straight"

    headings = np.arctan2(d[valid, 1], d[valid, 0])
    yaw_change = abs(normalize_angle(float(headings[-1] - headings[0])))
    yaw_deg = math.degrees(yaw_change)
    bucket = "very_sharp_turn"
    for threshold, label in _TURN_BUCKETS:
        if yaw_deg < threshold:
            bucket = label
            break
    return round(yaw_deg, 1), bucket


# ---------------------------------------------------------------------------
# Object-ego interaction helpers
# ---------------------------------------------------------------------------

def time_to_cross_ego_arc(
    obj_xy: np.ndarray,
    obj_vel: np.ndarray,
    ego_speed: float,
    ego_yaw_rate: float,
    horizon: float = 4.0,
    dt: float = 0.1,
    corridor_half_width: float = 2.0,
) -> Optional[float]:
    """Estimate when an object's future path crosses the ego's planned arc.

    Returns seconds until crossing, or None if no crossing within *horizon*.
    Simple forward sim with constant velocities.
    """
    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    ovx, ovy = float(obj_vel[0]), float(obj_vel[1])

    ex, ey, etheta = 0.0, 0.0, 0.0
    t = 0.0
    while t < horizon:
        t += dt
        ox += ovx * dt
        oy += ovy * dt
        etheta += ego_yaw_rate * dt
        ex += ego_speed * math.cos(etheta) * dt
        ey += ego_speed * math.sin(etheta) * dt
        dist = math.hypot(ox - ex, oy - ey)
        if dist < corridor_half_width:
            return round(t, 1)
    return None
