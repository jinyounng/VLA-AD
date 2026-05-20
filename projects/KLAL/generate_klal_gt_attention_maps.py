#!/usr/bin/env python3
"""Precompute KLAL GT attention maps for SpaceDrive visual tokens.

This script is intentionally standalone: it does not modify the training or
model code.  It reads SpaceDrive/nuScenes info PKLs, projects 3D boxes and
local HD-map geometries into each camera, rasterizes them on the Qwen visual
token grid, smooths, normalizes, and saves one ``.pt`` distribution per sample.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is expected in this repo env.
    Image = None

try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
except Exception:  # pragma: no cover - fallback below handles this.
    scipy_gaussian_filter = None


DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

MAP_CLASS_TO_ID = {
    "ped_crossing": 0,
    "crosswalk": 0,
    "divider": 1,
    "road_boundary": 2,
    "boundary": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KLAL GT attention maps from SpaceDrive nuScenes infos."
    )
    parser.add_argument(
        "--ann-file",
        default="data/nuscenes/nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl",
        help="SpaceDrive nuScenes info PKL.",
    )
    parser.add_argument(
        "--data-root",
        default="data/nuscenes",
        help="Root used to resolve camera image paths in the PKL.",
    )
    parser.add_argument(
        "--out-dir",
        default="workspace/klal_gt_attention_maps",
        help="Directory where <sample_token>.pt files are written.",
    )
    parser.add_argument("--image-size", nargs=2, type=int, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument(
        "--grid-mode",
        choices=("ceil", "floor"),
        default="ceil",
        help=(
            "ceil matches Qwen2.5-VL/SpaceDrive 640px -> 23x23 visual tokens; "
            "floor matches the literal formula in KLAL.md."
        ),
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--map-weight", type=float, default=1.0)
    parser.add_argument(
        "--map-sample-step",
        type=float,
        default=0.5,
        help="Meter interval used to densify HD-map polylines before projection.",
    )
    parser.add_argument(
        "--map-classes",
        nargs="+",
        default=None,
        help=(
            "Optional map classes to rasterize. Accepts ids or names: "
            "0/ped_crossing/crosswalk, 1/divider, 2/road_boundary/boundary."
        ),
    )
    parser.add_argument(
        "--distance-weight",
        action="store_true",
        help="Down-weight boxes by 1 / distance in lidar frame.",
    )
    parser.add_argument(
        "--camera-order",
        nargs="+",
        default=list(DEFAULT_CAMERA_ORDER),
        help="Camera concat order. Defaults to SpaceDrive info['cams'] order.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate maps even if output .pt already exists.",
    )
    parser.add_argument(
        "--save-debug-npz",
        action="store_true",
        help="Also save per-camera maps as compressed npz for inspection.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path, data_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if str(path).startswith("data/") or str(path).startswith("./data/"):
        return Path(str(path).lstrip("./"))
    return data_root / path


def visual_grid(image_h: int, image_w: int, patch_size: int, merge_size: int, mode: str) -> tuple[int, int]:
    stride = patch_size * merge_size
    if mode == "ceil":
        return math.ceil(image_h / stride), math.ceil(image_w / stride)
    return image_h // stride, image_w // stride


def scale_intrinsic_for_target(
    intrinsic: np.ndarray,
    raw_hw: tuple[int, int] | None,
    target_hw: tuple[int, int],
) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
    if intrinsic.shape == (4, 4):
        intrinsic_3 = intrinsic[:3, :3].copy()
    else:
        intrinsic_3 = intrinsic[:3, :3].copy()

    if raw_hw is not None:
        raw_h, raw_w = raw_hw
        target_h, target_w = target_hw
        sx = target_w / float(raw_w)
        sy = target_h / float(raw_h)
        intrinsic_3[0, 0] *= sx
        intrinsic_3[0, 2] *= sx
        intrinsic_3[1, 1] *= sy
        intrinsic_3[1, 2] *= sy
    return intrinsic_3


def image_hw(cam_info: dict, data_root: Path) -> tuple[int, int] | None:
    if Image is None or "data_path" not in cam_info:
        return None
    img_path = resolve_path(cam_info["data_path"], data_root)
    if not img_path.exists():
        return None
    with Image.open(img_path) as img:
        w, h = img.size
    return h, w


def quaternion_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Convert nuScenes [w, x, y, z] quaternions to a 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def lidar_to_cam_matrix(cam_info: dict) -> np.ndarray:
    """Return a 4x4 lidar-to-camera transform from info['cams'][cam]."""
    # Match projects/mmdet3d_plugin/datasets/nuscenes_dataset.py exactly:
    # that loader builds cam2lidar from sensor2ego_* and then inverts it.
    if "sensor2ego_rotation" in cam_info and "sensor2ego_translation" in cam_info:
        cam2lidar = np.eye(4, dtype=np.float64)
        cam2lidar[:3, :3] = quaternion_to_matrix(cam_info["sensor2ego_rotation"])
        cam2lidar[:3, 3] = np.asarray(cam_info["sensor2ego_translation"], dtype=np.float64)
        return np.linalg.inv(cam2lidar)
    if "sensor2lidar_rotation" in cam_info and "sensor2lidar_translation" in cam_info:
        cam2lidar = np.eye(4, dtype=np.float64)
        cam2lidar[:3, :3] = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64)
        cam2lidar[:3, 3] = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64)
        return np.linalg.inv(cam2lidar)
    if "extrinsics" in cam_info:
        return np.asarray(cam_info["extrinsics"], dtype=np.float64)
    raise KeyError("Camera info lacks sensor2lidar_* or extrinsics.")


def yaw_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def box_corners_lidar(box: Sequence[float]) -> np.ndarray:
    """Create 8 corners for boxes shaped [x, y, z, w, l, h, yaw]."""
    x, y, z, w, l, h, yaw = [float(v) for v in box[:7]]
    x_c = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z_c = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    corners = np.stack([x_c, y_c, z_c], axis=1)
    return corners @ yaw_rotation(yaw).T + np.array([x, y, z], dtype=np.float64)


def project_points(
    points_lidar: np.ndarray,
    intrinsic: np.ndarray,
    lidar2cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_lidar, dtype=np.float64)
    homo = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    cam = homo @ lidar2cam.T
    depth = cam[:, 2]
    valid = depth > 1e-5
    uvw = cam[:, :3] @ intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    return uv, valid


def raster_rect(
    grid: np.ndarray,
    min_u: float,
    min_v: float,
    max_u: float,
    max_v: float,
    image_hw_: tuple[int, int],
    patch_size: int,
    merge_size: int,
    weight: float,
) -> None:
    image_h, image_w = image_hw_
    stride = patch_size * merge_size
    min_u = float(np.clip(min_u, 0, image_w - 1))
    max_u = float(np.clip(max_u, 0, image_w - 1))
    min_v = float(np.clip(min_v, 0, image_h - 1))
    max_v = float(np.clip(max_v, 0, image_h - 1))
    if max_u <= min_u or max_v <= min_v:
        return
    j0 = max(0, int(math.floor(min_u / stride)))
    j1 = min(grid.shape[1] - 1, int(math.floor(max_u / stride)))
    i0 = max(0, int(math.floor(min_v / stride)))
    i1 = min(grid.shape[0] - 1, int(math.floor(max_v / stride)))
    grid[i0 : i1 + 1, j0 : j1 + 1] += float(weight)


def raster_points(
    grid: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    image_hw_: tuple[int, int],
    patch_size: int,
    merge_size: int,
    weight: float,
) -> None:
    image_h, image_w = image_hw_
    stride = patch_size * merge_size
    pts = uv[valid]
    if pts.size == 0:
        return
    in_img = (pts[:, 0] >= 0) & (pts[:, 0] < image_w) & (pts[:, 1] >= 0) & (pts[:, 1] < image_h)
    pts = pts[in_img]
    if pts.size == 0:
        return
    jj = np.clip((pts[:, 0] // stride).astype(np.int64), 0, grid.shape[1] - 1)
    ii = np.clip((pts[:, 1] // stride).astype(np.int64), 0, grid.shape[0] - 1)
    np.add.at(grid, (ii, jj), float(weight))


def gaussian_filter_np(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    if scipy_gaussian_filter is not None:
        return scipy_gaussian_filter(arr, sigma=sigma)

    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
    return out


def shapely_coords(geom) -> list[np.ndarray]:
    """Extract Nx2 arrays from LineString/MultiLineString/Polygon-like geometries."""
    if geom is None:
        return []
    if hasattr(geom, "geoms"):
        out: list[np.ndarray] = []
        for g in geom.geoms:
            out.extend(shapely_coords(g))
        return out
    if hasattr(geom, "exterior"):
        return [np.asarray(geom.exterior.coords, dtype=np.float64)]
    if hasattr(geom, "coords"):
        return [np.asarray(geom.coords, dtype=np.float64)]
    arr = np.asarray(geom, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return [arr[:, :2]]
    return []


def densify_polyline(xy: np.ndarray, step: float) -> np.ndarray:
    if xy.shape[0] < 2 or step <= 0:
        return xy[:, :2]
    dense = [xy[0, :2]]
    for start, end in zip(xy[:-1, :2], xy[1:, :2]):
        dist = float(np.linalg.norm(end - start))
        if dist <= 1e-8:
            continue
        n = max(1, int(math.ceil(dist / step)))
        for t in np.linspace(1.0 / n, 1.0, n):
            dense.append(start * (1.0 - t) + end * t)
    return np.asarray(dense, dtype=np.float64)


def parse_map_classes(classes: Sequence[str] | None) -> set[int] | None:
    if classes is None:
        return None
    parsed: set[int] = set()
    for cls in classes:
        key = str(cls).lower()
        if key.isdigit():
            parsed.add(int(key))
        elif key in MAP_CLASS_TO_ID:
            parsed.add(MAP_CLASS_TO_ID[key])
        else:
            raise ValueError(f"Unknown map class: {cls}")
    return parsed


def raster_boxes_for_camera(
    cam_map: np.ndarray,
    info: dict,
    intrinsic: np.ndarray,
    lidar2cam: np.ndarray,
    target_hw: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))), dtype=np.float64)
    valid_flag = info.get("valid_flag", None)
    for idx, box in enumerate(boxes):
        if valid_flag is not None and len(valid_flag) > idx and not bool(valid_flag[idx]):
            continue
        corners = box_corners_lidar(box)
        uv, valid = project_points(corners, intrinsic, lidar2cam)
        if not np.any(valid):
            continue
        uv_valid = uv[valid]
        if uv_valid.size == 0:
            continue
        min_u, min_v = uv_valid.min(axis=0)
        max_u, max_v = uv_valid.max(axis=0)
        weight = args.bbox_weight
        if args.distance_weight:
            dist = max(np.linalg.norm(box[:2]), 1.0)
            weight = weight / dist
        raster_rect(
            cam_map,
            min_u,
            min_v,
            max_u,
            max_v,
            target_hw,
            args.patch_size,
            args.merge_size,
            weight,
        )


def raster_map_for_camera(
    cam_map: np.ndarray,
    info: dict,
    intrinsic: np.ndarray,
    lidar2cam: np.ndarray,
    target_hw: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    map_geoms = info.get("map_geoms", {})
    if not map_geoms:
        return
    allowed_map_classes = parse_map_classes(args.map_classes)
    for map_class, geom_list in map_geoms.items():
        if allowed_map_classes is not None and int(map_class) not in allowed_map_classes:
            continue
        for geom in geom_list:
            for xy in shapely_coords(geom):
                if xy.shape[0] == 0:
                    continue
                xy = densify_polyline(xy, args.map_sample_step)
                points = np.concatenate(
                    [xy[:, :2], np.zeros((xy.shape[0], 1), dtype=np.float64)], axis=1
                )
                uv, valid = project_points(points, intrinsic, lidar2cam)
                raster_points(
                    cam_map,
                    uv,
                    valid,
                    target_hw,
                    args.patch_size,
                    args.merge_size,
                    args.map_weight,
                )


def build_attention_map(info: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    target_hw = tuple(args.image_size)
    token_h, token_w = visual_grid(
        target_hw[0], target_hw[1], args.patch_size, args.merge_size, args.grid_mode
    )
    data_root = Path(args.data_root)
    cam_maps = []
    cams = info["cams"]

    for cam_name in args.camera_order:
        if cam_name not in cams:
            raise KeyError(f"{cam_name} not found in info['cams']; available={list(cams)}")
        cam_info = cams[cam_name]
        raw_hw = image_hw(cam_info, data_root)
        intrinsic = scale_intrinsic_for_target(cam_info["cam_intrinsic"], raw_hw, target_hw)
        lidar2cam = lidar_to_cam_matrix(cam_info)
        cam_map = np.zeros((token_h, token_w), dtype=np.float64)

        raster_boxes_for_camera(cam_map, info, intrinsic, lidar2cam, target_hw, args)
        raster_map_for_camera(cam_map, info, intrinsic, lidar2cam, target_hw, args)
        cam_map = gaussian_filter_np(cam_map, args.sigma)
        cam_maps.append(cam_map)

    gt_map = np.concatenate([m.reshape(-1) for m in cam_maps], axis=0)
    if gt_map.sum() <= 0:
        gt_map[:] = 1.0
    gt_map = gt_map + args.eps
    gt_map = gt_map / gt_map.sum()
    return gt_map.astype(np.float32), np.stack(cam_maps).astype(np.float32)


def main() -> None:
    args = parse_args()
    ann_file = Path(args.ann_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ann_file.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    end = None if args.limit is None else args.start + args.limit
    selected = infos[args.start : end]

    token_h, token_w = visual_grid(
        args.image_size[0], args.image_size[1], args.patch_size, args.merge_size, args.grid_mode
    )
    print(
        f"Generating {len(selected)} maps: views={len(args.camera_order)}, "
        f"grid={token_h}x{token_w}, tokens={len(args.camera_order) * token_h * token_w}"
    )

    for n, info in enumerate(selected, start=args.start):
        token = info.get("token") or info.get("sample_idx")
        if token is None:
            raise KeyError(f"Info at index {n} lacks token/sample_idx.")
        out_path = out_dir / f"{token}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        gt_map, cam_maps = build_attention_map(info, args)
        torch.save(torch.from_numpy(gt_map), out_path)
        if args.save_debug_npz:
            np.savez_compressed(out_dir / f"{token}.npz", cam_maps=cam_maps, gt_map=gt_map)
        if (n - args.start + 1) % 100 == 0:
            print(f"[{n - args.start + 1}/{len(selected)}] wrote {out_path}")

    meta = {
        "ann_file": str(ann_file),
        "image_size": list(args.image_size),
        "patch_size": args.patch_size,
        "merge_size": args.merge_size,
        "grid_mode": args.grid_mode,
        "token_h": token_h,
        "token_w": token_w,
        "camera_order": list(args.camera_order),
        "sigma": args.sigma,
        "eps": args.eps,
    }
    torch.save(meta, out_dir / "metadata.pt")
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
