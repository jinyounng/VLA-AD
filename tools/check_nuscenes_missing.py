#!/usr/bin/env python3
"""List asset paths referenced by nuScenes info PKL that are missing on disk."""
import argparse
import os
import pickle
from os import path as osp


def resolve_path(path: str, data_root: str) -> str:
    if not path or not isinstance(path, str):
        return path
    normalized = path.replace("\\", "/")
    for prefix in ("./data/nuscenes/", "data/nuscenes/"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix) :].lstrip("/")
            return osp.join(data_root, rel)
    return path


def collect_paths(info: dict, data_root: str):
    paths = []
    p = resolve_path(info.get("lidar_path"), data_root)
    if p:
        paths.append(p)
    for sw in info.get("sweeps") or []:
        if isinstance(sw, dict) and sw.get("data_path"):
            paths.append(resolve_path(sw["data_path"], data_root))
    for cam_info in (info.get("cams") or {}).values():
        if isinstance(cam_info, dict) and cam_info.get("data_path"):
            paths.append(resolve_path(cam_info["data_path"], data_root))
    return paths


def load_infos(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data_infos" in data:
        return data["data_infos"]
    raise ValueError(f"Unsupported PKL structure: {type(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="e.g. /path/to/data/nuscenes/ (with trailing slash ok)",
    )
    parser.add_argument(
        "--pkl",
        required=True,
        help="nuscenes2d_ego_temporal_infos_*.pkl",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional: write missing paths (one per line) to this file",
    )
    args = parser.parse_args()
    data_root = osp.abspath(args.data_root.rstrip("/") + "/")

    infos = load_infos(args.pkl)
    missing = []
    seen = set()
    n_refs = 0
    for info in infos:
        for p in collect_paths(info, data_root):
            n_refs += 1
            if p in seen:
                continue
            seen.add(p)
            if not osp.isfile(p):
                missing.append(p)

    print(f"infos: {len(infos)}  path_refs: {n_refs}  unique_paths_checked: {len(seen)}")
    print(f"missing: {len(missing)}")
    for p in missing:
        print(p)
    if args.out:
        os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(missing) + ("\n" if missing else ""))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
