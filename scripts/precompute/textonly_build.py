#!/usr/bin/env python3
"""Build a text-only planning dataset from nuScenes annotation PKLs.

Converts each sample into a rich natural-language scene description (ego state,
per-camera objects with attributes & future-path notes, map topology, mission)
plus the 6-step GT trajectory target.  Output is a single JSONL file suitable
for LLM trajectory-prediction ablation.

Usage
-----
    python textonly_build.py \
        --base-path /data/nuscenes \
        --output workspace/textonly_planning.jsonl

See ``--help`` for full options.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from io import BufferedReader
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Ensure sibling modules are importable when run as a script.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from textonly_extractors import build_full_prompt, extract_mission
from textonly_geometry import kinematic_baseline, turn_severity

# ---------------------------------------------------------------------------
# Safe pickle loading (shapely fallback)
# ---------------------------------------------------------------------------

class _PickleFallback:
    def __init__(self, *a, **k):
        self._state = {}
    def __setstate__(self, s):
        self._state = s


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("shapely."):
            return _PickleFallback
        return super().find_class(module, name)


def safe_pickle_load(fp: BufferedReader):
    try:
        return pickle.load(fp)
    except ModuleNotFoundError as e:
        if "shapely" not in str(e):
            raise
        fp.seek(0)
        return _SafeUnpickler(fp).load()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base-path", type=str, required=True,
        help="nuScenes data root (parent of the PKL files)",
    )
    p.add_argument(
        "--anno-pkl", type=str,
        default="nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="Annotation PKL filename (relative to --base-path)",
    )
    p.add_argument(
        "--lane-pkl", type=str, default="data_dict_sample.pkl",
        help="Lane annotation PKL filename (relative to --base-path)",
    )
    p.add_argument("--output", "-o", type=str, required=True, help="Output JSONL path")

    # Filtering
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Cap output at N samples (0 = unlimited)",
    )
    p.add_argument(
        "--min-visibility", type=int, default=2,
        help="Minimum nuScenes visibility token for objects [1-4]",
    )
    p.add_argument(
        "--max-object-distance", type=float, default=50.0,
        help="Object distance threshold in meters",
    )
    p.add_argument(
        "--max-objects-per-camera", type=int, default=8,
        help="Max objects listed under each camera heading",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()

    # -- Load annotation PKL --
    anno_path = os.path.join(args.base_path, args.anno_pkl)
    print(f"Loading annotations: {anno_path}")
    with open(anno_path, "rb") as f:
        data = safe_pickle_load(f)
    infos = data["infos"]
    print(f"  {len(infos)} samples loaded")

    # -- Load lane PKL --
    lane_path = os.path.join(args.base_path, args.lane_pkl)
    lane_map: Dict = {}
    if os.path.exists(lane_path):
        print(f"Loading lane annotations: {lane_path}")
        with open(lane_path, "rb") as f:
            lane_map = safe_pickle_load(f)
        print(f"  {len(lane_map)} lane entries loaded")
    else:
        print(f"WARNING: lane PKL not found at {lane_path}; map text will be limited.")

    # -- Prepare output --
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stats = dict(total=0, written=0, skip_mask=0)

    with open(args.output, "w", encoding="utf-8") as fout:
        for info in infos:
            stats["total"] += 1

            # --- Validate GT planning ---
            gt = info.get("gt_planning")
            mask = info.get("gt_planning_mask")
            if gt is None or mask is None:
                stats["skip_mask"] += 1
                continue
            gt = np.asarray(gt, dtype=np.float64)
            mask = np.asarray(mask, dtype=np.float64)
            gt_xy = gt[0, :6, :2]
            valid = mask[0].astype(bool)
            if not valid.all():
                stats["skip_mask"] += 1
                continue

            # --- Lane entry lookup ---
            lane_key = info.get("lane_info")
            lane_entry: Optional[dict] = None
            if lane_key is not None and isinstance(lane_map, dict):
                lane_entry = lane_map.get(lane_key)

            # --- Build prompt ---
            prompt = build_full_prompt(info, lane_entry)

            # --- Turn severity ---
            yaw_deg, bucket = turn_severity(gt_xy)

            # --- Kinematic baseline ---
            cb = np.asarray(info.get("can_bus", np.zeros(13)), dtype=np.float64).ravel()
            speed = float(cb[10]) if cb.shape[0] > 10 else 0.0
            yaw_rate = float(cb[9]) if cb.shape[0] > 9 else 0.0
            kb = kinematic_baseline(speed, yaw_rate, n_steps=6, dt=0.5)

            # --- Mission text ---
            mission_text = extract_mission(info)

            # --- Compose output record ---
            record = {
                "token": str(info["token"]),
                "location": info.get("location", ""),
                "scene_description": info.get("description", ""),
                "mission": mission_text,
                "turn_severity": {
                    "yaw_change_deg": yaw_deg,
                    "bucket": bucket,
                },
                "prompt": prompt,
                "target_trajectory_xy": _round_traj(gt_xy),
                "kinematic_baseline_xy": _round_traj(kb),
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

            if args.max_samples > 0 and stats["written"] >= args.max_samples:
                break

    elapsed = time.time() - t0
    print("=" * 68)
    print(f"Text-only planning dataset built in {elapsed:.1f}s")
    print(f"  Total samples scanned : {stats['total']}")
    print(f"  Written               : {stats['written']}")
    print(f"  Skipped (invalid mask): {stats['skip_mask']}")
    print(f"  Output                : {args.output}")
    print("=" * 68)


def _round_traj(xy: np.ndarray, digits: int = 3) -> list:
    """Convert (N,2) array to list of [x,y] rounded for JSON."""
    return [[round(float(r[0]), digits), round(float(r[1]), digits)] for r in xy]


if __name__ == "__main__":
    main()
