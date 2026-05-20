#!/usr/bin/env python3
"""
Cross-View Feature Similarity Experiment for SpaceDrive.

Measures whether SpaceDrive's visual encoder produces similar features
for the same 3D object when it appears in multiple camera views simultaneously.

Steps:
  1. Use nuScenes devkit to project 3D annotations onto 6 camera views.
  2. Find objects visible in 2+ cameras.
  3. Load SpaceDrive's Qwen2.5-VL visual encoder and extract per-view features.
  4. Crop features for overlapping objects, compute cosine similarity.
  5. Compare same-object vs different-object similarity.
  6. Split analysis by turn vs non-turn samples.
"""

import argparse
import os
import sys
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from pyquaternion import Quaternion


# ---------------------------------------------------------------------------
# nuScenes camera order (must match SpaceDrive's convention)
# ---------------------------------------------------------------------------
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

IMG_H, IMG_W = 640, 640  # SpaceDrive resize target


# ===================================================================
# Part 1 — Project 3D annotations to camera views
# ===================================================================

def get_sample_camera_data(nusc: NuScenes, sample_token: str) -> Dict[str, dict]:
    """Return calibration + file info for each of the 6 cameras in a sample."""
    sample = nusc.get("sample", sample_token)
    cam_data = {}
    for cam_name in CAMERA_NAMES:
        sd_token = sample["data"][cam_name]
        sd_record = nusc.get("sample_data", sd_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        ego_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
        cam_data[cam_name] = {
            "sd_record": sd_record,
            "cs_record": cs_record,
            "ego_record": ego_record,
            "intrinsic": np.array(cs_record["camera_intrinsic"]),
            "filename": sd_record["filename"],
            "width": sd_record["width"],
            "height": sd_record["height"],
        }
    return cam_data


def project_annotation_to_camera(
    nusc: NuScenes,
    ann_token: str,
    cam_info: dict,
    margin: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Project a sample_annotation 3D box into a camera view.

    Returns (corners_2d [2, 8], bbox [x1, y1, x2, y2] in original image coords)
    or None if the annotation is not visible.
    """
    ann = nusc.get("sample_annotation", ann_token)
    cs_record = cam_info["cs_record"]
    ego_record = cam_info["ego_record"]
    intrinsic = cam_info["intrinsic"]
    orig_w, orig_h = cam_info["width"], cam_info["height"]

    # Global -> ego
    box_center = np.array(ann["translation"])
    box_size = ann["size"]  # w, l, h
    box_rotation = Quaternion(ann["rotation"])

    # Transform center to ego frame
    ego_t = np.array(ego_record["translation"])
    ego_r = Quaternion(ego_record["rotation"])
    center_ego = ego_r.inverse.rotate(box_center - ego_t)

    # Transform center to sensor frame
    sensor_t = np.array(cs_record["translation"])
    sensor_r = Quaternion(cs_record["rotation"])
    center_sensor = sensor_r.inverse.rotate(center_ego - sensor_t)

    # If behind camera, skip
    if center_sensor[2] <= 0:
        return None

    # Full rotation in sensor frame
    rot_sensor = sensor_r.inverse * ego_r.inverse * box_rotation

    # Build 3D corners in sensor frame
    from nuscenes.utils.data_classes import Box
    box3d = Box(center_sensor, box_size, rot_sensor)
    corners_3d = box3d.corners()  # (3, 8)

    # Any corner behind camera -> partial visibility
    if np.any(corners_3d[2, :] <= 0):
        return None

    # Project to image
    corners_2d = view_points(corners_3d, np.array(intrinsic), normalize=True)[:2]  # (2, 8)

    x_min, x_max = corners_2d[0].min(), corners_2d[0].max()
    y_min, y_max = corners_2d[1].min(), corners_2d[1].max()

    # Check if within image bounds (with margin)
    if x_max < margin or x_min > orig_w - margin:
        return None
    if y_max < margin or y_min > orig_h - margin:
        return None

    # Clip to image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(orig_w, x_max)
    y_max = min(orig_h, y_max)

    bbox_area = (x_max - x_min) * (y_max - y_min)
    if bbox_area < 100:  # skip tiny projections
        return None

    bbox = np.array([x_min, y_min, x_max, y_max])
    return corners_2d, bbox


def find_multiview_objects(
    nusc: NuScenes,
    sample_token: str,
) -> List[dict]:
    """Find objects that appear in 2+ camera views for one sample.

    Returns list of dicts:
      {
        'ann_token': str,
        'category': str,
        'instance_token': str,
        'views': {cam_name: {'bbox_orig': [x1,y1,x2,y2], 'bbox_resized': [x1,y1,x2,y2]}},
      }
    """
    cam_data = get_sample_camera_data(nusc, sample_token)
    sample = nusc.get("sample", sample_token)

    results = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        views = {}
        for cam_name in CAMERA_NAMES:
            proj = project_annotation_to_camera(nusc, ann_token, cam_data[cam_name])
            if proj is not None:
                _, bbox_orig = proj
                w_orig, h_orig = cam_data[cam_name]["width"], cam_data[cam_name]["height"]
                scale_x = IMG_W / w_orig
                scale_y = IMG_H / h_orig
                bbox_resized = np.array([
                    bbox_orig[0] * scale_x,
                    bbox_orig[1] * scale_y,
                    bbox_orig[2] * scale_x,
                    bbox_orig[3] * scale_y,
                ])
                views[cam_name] = {
                    "bbox_orig": bbox_orig.tolist(),
                    "bbox_resized": bbox_resized.tolist(),
                }

        if len(views) >= 2:
            results.append({
                "ann_token": ann_token,
                "category": ann["category_name"],
                "instance_token": ann["instance_token"],
                "views": views,
            })

    return results


# ===================================================================
# Part 2 — SpaceDrive feature extraction
# ===================================================================

def load_qwen_visual_encoder(model_path: str, device: str = "cuda"):
    """Load only the Qwen2.5-VL visual model + processor for feature extraction."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, processor


def load_spacedrive_visual_encoder(
    model_path: str,
    checkpoint_path: str,
    device: str = "cuda",
):
    """Load SpaceDrive's fine-tuned Qwen2.5-VL (with LoRA) visual encoder.

    This loads the full SpaceDrive checkpoint and extracts the visual backbone,
    which may have been affected by LoRA fine-tuning on the q/k/v/o projections.
    Note: In SpaceDrive's default config, LoRA targets are in the LLM attention
    layers, not the vision tower. So the vision encoder weights should be
    identical to the pretrained Qwen2.5-VL unless the checkpoint modifies them.
    We load the full model anyway to be safe and to allow comparing.
    """
    from transformers import AutoProcessor

    # SpaceDrive's internal imports use absolute paths like
    # ``from projects.mmdet3d_plugin...``, so the repo root must be on sys.path.
    _spacedrive_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if _spacedrive_root not in sys.path:
        sys.path.insert(0, _spacedrive_root)

    from projects.mmdet3d_plugin.models.vlm_utils.custom_qwen import (
        CustomQwen2_5_VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_path)

    base_model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, local_files_only=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading SpaceDrive checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # SpaceDrive wraps the model in MMDet3D detector; extract the lm_head submodule keys
        visual_keys = {
            k.replace("lm_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("lm_head.") and "visual" in k
        }
        if visual_keys:
            missing, unexpected = base_model.load_state_dict(visual_keys, strict=False)
            print(f"  Loaded {len(visual_keys)} visual keys "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("  No separate visual keys in checkpoint (vision tower unchanged from pretrained)")

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    return base_model, processor


def extract_per_view_features(
    model,
    processor,
    images: List[Image.Image],
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Extract visual encoder features for each camera view image.

    The Qwen2.5-VL image processor handles patching / merging internally.
    We call the vision model directly (model.visual) to get patch-level features.

    Returns:
        List of tensors, one per view. Each has shape (H_patches, W_patches, C).
    """
    visual_processed = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = visual_processed["pixel_values"].to(device, dtype=torch.bfloat16)
    image_grid_thw = visual_processed["image_grid_thw"].to(device)

    # Qwen2.5-VL visual encoder: outputs (total_patches, hidden_dim)
    with torch.no_grad():
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)

    # Split per view using image_grid_thw
    per_view_features = []
    offset = 0
    for i in range(image_grid_thw.shape[0]):
        t, h, w = image_grid_thw[i].tolist()
        # After Qwen's merge operation, effective spatial size is
        # (h // merge_size, w // merge_size) * t
        merge_size = getattr(processor.image_processor, "merge_size", 2)
        h_merged = h // merge_size
        w_merged = w // merge_size
        n_tokens = int(t * h_merged * w_merged)
        view_feats = image_embeds[offset : offset + n_tokens]  # (n_tokens, C)
        # Reshape to spatial: use the last frame if t > 1 (images have t=1)
        view_feats = view_feats.reshape(t, h_merged, w_merged, -1)
        if t > 1:
            view_feats = view_feats[-1:]  # keep last temporal frame
        view_feats = view_feats.squeeze(0)  # (h_merged, w_merged, C)
        per_view_features.append(view_feats)
        offset += n_tokens

    return per_view_features


# ===================================================================
# Part 3 — Feature cropping & cosine similarity
# ===================================================================

def bbox_to_feature_region(
    bbox_resized: List[float],
    feat_h: int,
    feat_w: int,
    stride: int = 14,
    merge_size: int = 2,
) -> Tuple[int, int, int, int]:
    """Map a resized-image-space bbox to feature map grid indices.

    The effective stride from image to feature map is stride * merge_size.
    """
    effective_stride = stride * merge_size
    x1, y1, x2, y2 = bbox_resized

    fi_y1 = max(0, int(y1 / effective_stride))
    fi_x1 = max(0, int(x1 / effective_stride))
    fi_y2 = min(feat_h, int(np.ceil(y2 / effective_stride)))
    fi_x2 = min(feat_w, int(np.ceil(x2 / effective_stride)))

    # Ensure at least 1x1 region
    if fi_y2 <= fi_y1:
        fi_y2 = fi_y1 + 1
    if fi_x2 <= fi_x1:
        fi_x2 = fi_x1 + 1

    fi_y2 = min(fi_y2, feat_h)
    fi_x2 = min(fi_x2, feat_w)

    return fi_y1, fi_x1, fi_y2, fi_x2


def crop_and_pool_feature(
    feat_map: torch.Tensor,
    bbox_resized: List[float],
    stride: int = 14,
    merge_size: int = 2,
) -> torch.Tensor:
    """Crop feature map at bbox location and average-pool to a single vector.

    Args:
        feat_map: (H, W, C)
        bbox_resized: [x1, y1, x2, y2] in resized image space

    Returns:
        Pooled feature vector of shape (C,)
    """
    h, w, c = feat_map.shape
    y1, x1, y2, x2 = bbox_to_feature_region(bbox_resized, h, w, stride, merge_size)
    cropped = feat_map[y1:y2, x1:x2, :]  # (rh, rw, C)
    pooled = cropped.reshape(-1, c).mean(dim=0)  # (C,)
    return pooled


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()


# ===================================================================
# Part 4 — Turn / non-turn classification
# ===================================================================

def classify_turn(
    nusc: NuScenes,
    sample_token: str,
    info_lookup: Optional[Dict[str, dict]] = None,
    angle_threshold_deg: float = 10.0,
) -> bool:
    """Determine if a sample is a turn sample.

    Uses gt_planning trajectory curvature if info_lookup is provided,
    otherwise falls back to command description from annotation PKL.
    """
    if info_lookup is not None and sample_token in info_lookup:
        info = info_lookup[sample_token]
        cmd_desc = info.get("gt_planning_command_desc", "")
        if isinstance(cmd_desc, str):
            low = cmd_desc.lower()
            if any(k in low for k in ("turning left", "turning right",
                                       "turning sharp left", "turning sharp right")):
                return True

        gt_planning = info.get("gt_planning")
        if gt_planning is not None:
            traj = np.array(gt_planning)[0, :6, :2]
            angle = _traj_turn_angle_deg(traj)
            if abs(angle) >= angle_threshold_deg:
                return True

    return False


def _traj_turn_angle_deg(traj_xy: np.ndarray) -> float:
    if traj_xy is None or len(traj_xy) < 3:
        return 0.0
    diffs = np.diff(traj_xy[:, :2], axis=0)
    valid = np.linalg.norm(diffs, axis=1) > 1e-6
    diffs = diffs[valid]
    if len(diffs) < 2:
        return 0.0
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.unwrap(headings)
    return float(np.degrees(headings[-1] - headings[0]))


# ===================================================================
# Part 5 — Main experiment
# ===================================================================

def load_info_lookup(anno_pkl_path: str) -> Dict[str, dict]:
    """Load annotation PKL and build a token -> info dict."""
    with open(anno_pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if "infos" in data else data
    lookup = {}
    for info in infos:
        lookup[info["token"]] = info
    return lookup


def run_experiment(args):
    print("=" * 70)
    print("Cross-View Feature Similarity Experiment")
    print("=" * 70)

    # --- Load nuScenes ---
    print(f"\n[1/5] Loading nuScenes from {args.nuscenes_root} ...")
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=True)

    # --- Build info lookup for turn classification ---
    info_lookup = None
    if args.anno_pkl and os.path.exists(args.anno_pkl):
        print(f"  Loading annotation PKL: {args.anno_pkl}")
        info_lookup = load_info_lookup(args.anno_pkl)

    # --- Determine which samples to process ---
    if args.split == "val":
        scene_splits = {s["token"]: s for s in nusc.scene}
        val_scenes = [s for s in nusc.scene if s["name"] in _get_val_scene_names()]
        sample_tokens = []
        for scene in val_scenes:
            token = scene["first_sample_token"]
            while token:
                sample_tokens.append(token)
                sample = nusc.get("sample", token)
                token = sample["next"]
    else:
        sample_tokens = [s["token"] for s in nusc.sample]

    if args.max_samples > 0:
        sample_tokens = sample_tokens[: args.max_samples]

    print(f"  Processing {len(sample_tokens)} samples")

    # --- Find multi-view objects ---
    print("\n[2/5] Finding objects visible in 2+ camera views ...")
    sample_mv_objects = {}
    total_mv_objects = 0
    for st in tqdm(sample_tokens, desc="Projecting annotations"):
        mv_objs = find_multiview_objects(nusc, st)
        if mv_objs:
            sample_mv_objects[st] = mv_objs
            total_mv_objects += len(mv_objs)

    print(f"  Found {total_mv_objects} multi-view objects across {len(sample_mv_objects)} samples")

    if total_mv_objects == 0:
        print("No multi-view objects found. Exiting.")
        return

    # --- Load visual encoder ---
    if args.spacedrive_ckpt:
        print(f"\n[3/5] Loading SpaceDrive fine-tuned encoder from {args.model_path} "
              f"+ {args.spacedrive_ckpt} ...")
        model, processor = load_spacedrive_visual_encoder(
            args.model_path, args.spacedrive_ckpt, device=args.device,
        )
    else:
        print(f"\n[3/5] Loading Qwen2.5-VL visual encoder from {args.model_path} ...")
        model, processor = load_qwen_visual_encoder(args.model_path, device=args.device)
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    stride = args.stride
    print(f"  stride={stride}, merge_size={merge_size}, effective_stride={stride * merge_size}")

    # --- Extract features and compute similarities ---
    print("\n[4/5] Extracting features and computing similarities ...")

    results_same_object = []      # cosine sim for same object across views
    results_diff_object = []      # cosine sim for different objects (baseline)
    results_by_category = defaultdict(lambda: {"same": [], "diff": []})
    results_turn = {"same": [], "diff": []}
    results_non_turn = {"same": [], "diff": []}

    processed = 0
    for sample_token, mv_objs in tqdm(sample_mv_objects.items(), desc="Feature extraction"):
        # Load 6 camera images
        cam_data = get_sample_camera_data(nusc, sample_token)
        images = []
        for cam_name in CAMERA_NAMES:
            img_path = os.path.join(args.nuscenes_root, cam_data[cam_name]["filename"])
            img = Image.open(img_path).convert("RGB").resize((IMG_W, IMG_H))
            images.append(img)

        # Extract per-view features
        try:
            per_view_feats = extract_per_view_features(model, processor, images, device=args.device)
        except Exception as e:
            print(f"  Warning: feature extraction failed for {sample_token}: {e}")
            continue

        # Build cam_name -> feature map lookup
        cam_feat_map = {}
        for i, cam_name in enumerate(CAMERA_NAMES):
            cam_feat_map[cam_name] = per_view_feats[i]

        # Determine turn/non-turn
        is_turn = classify_turn(nusc, sample_token, info_lookup, args.turn_angle_threshold)
        group = results_turn if is_turn else results_non_turn

        # --- Same-object similarity: for each multi-view object, compare
        #     features between all pairs of views ---
        obj_features = {}  # (obj_idx, cam_name) -> pooled feature
        for obj_idx, obj in enumerate(mv_objs):
            for cam_name, view_info in obj["views"].items():
                feat_map = cam_feat_map[cam_name]
                pooled = crop_and_pool_feature(
                    feat_map, view_info["bbox_resized"], stride, merge_size
                )
                obj_features[(obj_idx, cam_name)] = pooled

            # Pairwise same-object similarity
            view_names = list(obj["views"].keys())
            for i in range(len(view_names)):
                for j in range(i + 1, len(view_names)):
                    fa = obj_features[(obj_idx, view_names[i])]
                    fb = obj_features[(obj_idx, view_names[j])]
                    sim = cosine_sim(fa, fb)
                    record = {
                        "sample_token": sample_token,
                        "ann_token": obj["ann_token"],
                        "category": obj["category"],
                        "view_a": view_names[i],
                        "view_b": view_names[j],
                        "cosine_similarity": sim,
                        "is_turn": is_turn,
                    }
                    results_same_object.append(record)
                    results_by_category[obj["category"]]["same"].append(sim)
                    group["same"].append(sim)

        # --- Different-object similarity (CROSS-VIEW only) ---
        # To be a fair baseline against same-object cross-view comparison,
        # diff-object pairs must also compare features across *different*
        # camera views.  Same-view diff inflates similarity because the
        # two crops share viewpoint, lighting, and background.
        #
        # Strategy: for every pair of different objects that each appear in
        # 2+ views, pick view pairs (vi, vj) where vi != vj.
        # We also collect all single-view objects in this sample to expand
        # the diff pool (any object visible in any view can participate).
        all_obj_features = {}  # (global_obj_idx, cam_name) -> pooled feature
        all_obj_meta = []      # list of (obj_idx_in_mv, category, views_dict)

        # Include multi-view objects (already computed above)
        for obj_idx, obj in enumerate(mv_objs):
            for cam_name in obj["views"]:
                all_obj_features[(obj_idx, cam_name)] = obj_features[(obj_idx, cam_name)]
            all_obj_meta.append((obj_idx, obj["category"], obj["views"], obj["ann_token"]))

        # Build cross-view diff pairs: obj_i in view_a vs obj_j in view_b (i!=j, a!=b)
        if len(all_obj_meta) >= 2:
            for i in range(len(all_obj_meta)):
                for j in range(i + 1, len(all_obj_meta)):
                    idx_i, cat_i, views_i, ann_i = all_obj_meta[i]
                    idx_j, cat_j, views_j, ann_j = all_obj_meta[j]
                    # All cross-view pairs between these two objects
                    cross_pairs = [
                        (vi, vj) for vi in views_i for vj in views_j if vi != vj
                    ]
                    # Limit to avoid combinatorial explosion
                    for vi, vj in cross_pairs[:4]:
                        fa = all_obj_features.get((idx_i, vi))
                        fb = all_obj_features.get((idx_j, vj))
                        if fa is not None and fb is not None:
                            sim = cosine_sim(fa, fb)
                            record = {
                                "sample_token": sample_token,
                                "ann_a": ann_i,
                                "ann_b": ann_j,
                                "cat_a": cat_i,
                                "cat_b": cat_j,
                                "view_a": vi,
                                "view_b": vj,
                                "cosine_similarity": sim,
                                "is_turn": is_turn,
                            }
                            results_diff_object.append(record)
                            results_by_category[cat_i]["diff"].append(sim)
                            results_by_category[cat_j]["diff"].append(sim)
                            group["diff"].append(sim)

        processed += 1
        if processed % 50 == 0:
            _print_interim(results_same_object, results_diff_object)

    # --- Report ---
    print("\n" + "=" * 70)
    print("[5/5] RESULTS")
    print("=" * 70)
    _print_full_report(
        results_same_object,
        results_diff_object,
        results_by_category,
        results_turn,
        results_non_turn,
    )

    # --- Save ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        _save_results(
            args.output_dir,
            results_same_object,
            results_diff_object,
            results_by_category,
            results_turn,
            results_non_turn,
        )


# ===================================================================
# Reporting helpers
# ===================================================================

def _stats(values):
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _print_interim(same, diff):
    s_sims = [r["cosine_similarity"] for r in same]
    d_sims = [r["cosine_similarity"] for r in diff]
    s = _stats(s_sims)
    d = _stats(d_sims)
    print(f"  [interim] same-obj: n={s['n']}, mean={s['mean']:.4f} | "
          f"diff-obj: n={d['n']}, mean={d['mean']:.4f}")


def _print_full_report(same, diff, by_cat, turn, non_turn):
    s_sims = [r["cosine_similarity"] for r in same]
    d_sims = [r["cosine_similarity"] for r in diff]

    s = _stats(s_sims)
    d = _stats(d_sims)

    print("\n--- Overall ---")
    print(f"Same-object cross-view similarity:  n={s['n']:>6d}  "
          f"mean={s['mean']:.4f}  std={s['std']:.4f}  median={s['median']:.4f}  "
          f"[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"Diff-object cross-view (baseline):  n={d['n']:>6d}  "
          f"mean={d['mean']:.4f}  std={d['std']:.4f}  median={d['median']:.4f}  "
          f"[{d['min']:.4f}, {d['max']:.4f}]")

    if s["n"] > 0 and d["n"] > 0:
        gap = s["mean"] - d["mean"]
        print(f"  *** Gap (same - diff): {gap:+.4f} ***")

    # Per-category
    print("\n--- Per-category ---")
    for cat in sorted(by_cat.keys()):
        cs = _stats(by_cat[cat]["same"])
        cd = _stats(by_cat[cat]["diff"])
        if cs["n"] >= 5:
            print(f"  {cat:<35s}  same: n={cs['n']:>4d} mean={cs['mean']:.4f}  |  "
                  f"diff: n={cd['n']:>4d} mean={cd['mean']:.4f}")

    # Turn vs non-turn
    print("\n--- Turn vs Non-turn ---")
    for label, grp in [("TURN", turn), ("NON-TURN", non_turn)]:
        gs = _stats(grp["same"])
        gd = _stats(grp["diff"])
        print(f"  [{label}]")
        print(f"    Same-object: n={gs['n']:>5d}  mean={gs['mean']:.4f}  std={gs['std']:.4f}")
        print(f"    Diff-object: n={gd['n']:>5d}  mean={gd['mean']:.4f}  std={gd['std']:.4f}")
        if gs["n"] > 0 and gd["n"] > 0:
            print(f"    Gap (same-diff): {gs['mean'] - gd['mean']:+.4f}")

    # View-pair breakdown
    print("\n--- View-pair breakdown (same-object) ---")
    pair_sims = defaultdict(list)
    for r in same:
        pair = tuple(sorted([r["view_a"], r["view_b"]]))
        pair_sims[pair].append(r["cosine_similarity"])
    for pair in sorted(pair_sims.keys()):
        ps = _stats(pair_sims[pair])
        print(f"  {pair[0]:<20s} <-> {pair[1]:<20s}  n={ps['n']:>4d}  mean={ps['mean']:.4f}")


def _save_results(output_dir, same, diff, by_cat, turn, non_turn):
    s_sims = [r["cosine_similarity"] for r in same]
    d_sims = [r["cosine_similarity"] for r in diff]

    summary = {
        "overall": {
            "same_object": _stats(s_sims),
            "diff_object": _stats(d_sims),
        },
        "turn": {
            "same_object": _stats(turn["same"]),
            "diff_object": _stats(turn["diff"]),
        },
        "non_turn": {
            "same_object": _stats(non_turn["same"]),
            "diff_object": _stats(non_turn["diff"]),
        },
        "per_category": {
            cat: {"same": _stats(v["same"]), "diff": _stats(v["diff"])}
            for cat, v in by_cat.items()
        },
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "same_object_pairs.json"), "w") as f:
        json.dump(same, f, indent=2)

    with open(os.path.join(output_dir, "diff_object_pairs.json"), "w") as f:
        json.dump(diff[:10000], f, indent=2)  # cap size

    print(f"\nResults saved to {output_dir}/")


def _get_val_scene_names():
    """Official nuScenes val split scene names."""
    return [
        'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
        'scene-0016', 'scene-0017', 'scene-0018', 'scene-0035', 'scene-0036',
        'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094',
        'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099',
        'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104',
        'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109',
        'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270',
        'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
        'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330',
        'scene-0331', 'scene-0332', 'scene-0344', 'scene-0345', 'scene-0346',
        'scene-0347', 'scene-0348', 'scene-0349', 'scene-0350', 'scene-0351',
        'scene-0352', 'scene-0353', 'scene-0354', 'scene-0355', 'scene-0356',
        'scene-0357', 'scene-0358', 'scene-0359', 'scene-0360', 'scene-0361',
        'scene-0362', 'scene-0363', 'scene-0364', 'scene-0365', 'scene-0366',
        'scene-0367', 'scene-0368', 'scene-0369', 'scene-0370', 'scene-0371',
        'scene-0372', 'scene-0373', 'scene-0374', 'scene-0375', 'scene-0376',
        'scene-0377', 'scene-0378', 'scene-0379', 'scene-0380', 'scene-0381',
        'scene-0382', 'scene-0383', 'scene-0384', 'scene-0385', 'scene-0386',
        'scene-0388', 'scene-0389', 'scene-0390', 'scene-0391', 'scene-0392',
        'scene-0393', 'scene-0394', 'scene-0395', 'scene-0396', 'scene-0397',
        'scene-0398', 'scene-0399', 'scene-0400', 'scene-0401', 'scene-0402',
        'scene-0403', 'scene-0405', 'scene-0406', 'scene-0407', 'scene-0408',
        'scene-0410', 'scene-0411', 'scene-0412', 'scene-0413', 'scene-0414',
        'scene-0415', 'scene-0416', 'scene-0417', 'scene-0418', 'scene-0419',
        'scene-0420', 'scene-0421', 'scene-0422', 'scene-0423', 'scene-0424',
        'scene-0425', 'scene-0426', 'scene-0427', 'scene-0428', 'scene-0429',
        'scene-0430', 'scene-0431', 'scene-0432', 'scene-0433', 'scene-0434',
        'scene-0435', 'scene-0436', 'scene-0437', 'scene-0438', 'scene-0439',
        'scene-0440', 'scene-0441', 'scene-0442', 'scene-0443', 'scene-0444',
        'scene-0445', 'scene-0446', 'scene-0447', 'scene-0448', 'scene-0449',
        'scene-0450', 'scene-0451', 'scene-0452', 'scene-0453', 'scene-0454',
        'scene-0455', 'scene-0456', 'scene-0457', 'scene-0458', 'scene-0459',
        'scene-0461', 'scene-0462', 'scene-0463', 'scene-0464', 'scene-0465',
        'scene-0467', 'scene-0468', 'scene-0469', 'scene-0471', 'scene-0472',
        'scene-0474', 'scene-0475', 'scene-0476', 'scene-0477', 'scene-0478',
        'scene-0479', 'scene-0480', 'scene-0499', 'scene-0500', 'scene-0501',
        'scene-0502', 'scene-0504', 'scene-0505', 'scene-0506', 'scene-0507',
        'scene-0508', 'scene-0509', 'scene-0510', 'scene-0511', 'scene-0512',
        'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518',
        'scene-0525', 'scene-0526', 'scene-0527', 'scene-0528', 'scene-0529',
        'scene-0530', 'scene-0531', 'scene-0532', 'scene-0533', 'scene-0534',
        'scene-0535', 'scene-0536', 'scene-0537', 'scene-0538', 'scene-0539',
        'scene-0541', 'scene-0542', 'scene-0543', 'scene-0544', 'scene-0545',
        'scene-0546', 'scene-0566', 'scene-0568', 'scene-0570', 'scene-0571',
        'scene-0572', 'scene-0573', 'scene-0574', 'scene-0575', 'scene-0576',
        'scene-0577', 'scene-0578', 'scene-0580', 'scene-0582', 'scene-0583',
        'scene-0584', 'scene-0585', 'scene-0586', 'scene-0587', 'scene-0588',
        'scene-0589', 'scene-0590', 'scene-0591', 'scene-0592', 'scene-0593',
        'scene-0594', 'scene-0595', 'scene-0596', 'scene-0597', 'scene-0598',
        'scene-0599', 'scene-0600', 'scene-0639', 'scene-0640', 'scene-0641',
        'scene-0642', 'scene-0643', 'scene-0644', 'scene-0645', 'scene-0646',
        'scene-0647', 'scene-0648', 'scene-0649', 'scene-0650', 'scene-0651',
        'scene-0652', 'scene-0653', 'scene-0654', 'scene-0655', 'scene-0656',
        'scene-0657', 'scene-0658', 'scene-0659', 'scene-0660', 'scene-0661',
        'scene-0662', 'scene-0663', 'scene-0664', 'scene-0665', 'scene-0666',
        'scene-0667', 'scene-0668', 'scene-0669', 'scene-0670', 'scene-0671',
        'scene-0672', 'scene-0673', 'scene-0674', 'scene-0675', 'scene-0676',
        'scene-0677', 'scene-0678', 'scene-0679', 'scene-0681', 'scene-0683',
        'scene-0684', 'scene-0685', 'scene-0686', 'scene-0687', 'scene-0688',
        'scene-0689', 'scene-0695', 'scene-0696', 'scene-0697', 'scene-0698',
        'scene-0700', 'scene-0701', 'scene-0703', 'scene-0704', 'scene-0705',
        'scene-0706', 'scene-0707', 'scene-0708', 'scene-0709', 'scene-0710',
        'scene-0711', 'scene-0712', 'scene-0713', 'scene-0714', 'scene-0715',
        'scene-0716', 'scene-0717', 'scene-0718', 'scene-0719', 'scene-0726',
        'scene-0727', 'scene-0728', 'scene-0730', 'scene-0731', 'scene-0733',
        'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738',
        'scene-0786', 'scene-0787', 'scene-0789', 'scene-0790', 'scene-0791',
        'scene-0792', 'scene-0803', 'scene-0804', 'scene-0805', 'scene-0806',
        'scene-0808', 'scene-0809', 'scene-0810', 'scene-0811', 'scene-0812',
        'scene-0813', 'scene-0815', 'scene-0816', 'scene-0817', 'scene-0819',
        'scene-0820', 'scene-0821', 'scene-0822', 'scene-0847', 'scene-0848',
        'scene-0849', 'scene-0850', 'scene-0851', 'scene-0852', 'scene-0853',
        'scene-0854', 'scene-0855', 'scene-0856', 'scene-0858', 'scene-0860',
        'scene-0861', 'scene-0862', 'scene-0863', 'scene-0864', 'scene-0865',
        'scene-0866', 'scene-0868', 'scene-0869', 'scene-0870', 'scene-0871',
        'scene-0872', 'scene-0873', 'scene-0875', 'scene-0876', 'scene-0877',
        'scene-0878', 'scene-0880', 'scene-0882', 'scene-0883', 'scene-0884',
        'scene-0885', 'scene-0886', 'scene-0887', 'scene-0888', 'scene-0889',
        'scene-0890', 'scene-0891', 'scene-0892', 'scene-0893', 'scene-0894',
        'scene-0895', 'scene-0896', 'scene-0897', 'scene-0898', 'scene-0899',
        'scene-0900', 'scene-0901', 'scene-0902', 'scene-0903', 'scene-0945',
        'scene-0947', 'scene-0949', 'scene-0952', 'scene-0953', 'scene-0955',
        'scene-0956', 'scene-0957', 'scene-0958', 'scene-0959', 'scene-0960',
        'scene-0961', 'scene-0975', 'scene-0976', 'scene-0977', 'scene-0978',
        'scene-0979', 'scene-0980', 'scene-0981', 'scene-0982', 'scene-0983',
        'scene-0984', 'scene-0988', 'scene-0989', 'scene-0990', 'scene-0991',
        'scene-0992', 'scene-0994', 'scene-0995', 'scene-0996', 'scene-0997',
        'scene-0998', 'scene-0999', 'scene-1000', 'scene-1001', 'scene-1002',
        'scene-1003', 'scene-1004', 'scene-1005', 'scene-1006', 'scene-1007',
        'scene-1008', 'scene-1009', 'scene-1010', 'scene-1011', 'scene-1012',
        'scene-1013', 'scene-1014', 'scene-1015', 'scene-1016', 'scene-1017',
        'scene-1018', 'scene-1019', 'scene-1020', 'scene-1021', 'scene-1022',
        'scene-1023', 'scene-1024', 'scene-1025', 'scene-1044', 'scene-1045',
        'scene-1046', 'scene-1047', 'scene-1048', 'scene-1049', 'scene-1050',
        'scene-1051', 'scene-1052', 'scene-1053', 'scene-1054', 'scene-1055',
        'scene-1056', 'scene-1057', 'scene-1058', 'scene-1074', 'scene-1075',
        'scene-1076', 'scene-1077', 'scene-1078', 'scene-1079', 'scene-1080',
        'scene-1081', 'scene-1082', 'scene-1083', 'scene-1084', 'scene-1085',
        'scene-1086', 'scene-1087', 'scene-1088', 'scene-1089', 'scene-1090',
        'scene-1091', 'scene-1092', 'scene-1093', 'scene-1094', 'scene-1095',
        'scene-1096', 'scene-1097', 'scene-1098', 'scene-1099', 'scene-1100',
        'scene-1101', 'scene-1102', 'scene-1104', 'scene-1105', 'scene-1106',
        'scene-1107', 'scene-1108', 'scene-1109', 'scene-1110',
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-view feature similarity experiment for SpaceDrive"
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default="/data/jykim/projects/OpenDriveVLA/data/nuscenes/",
    )
    parser.add_argument(
        "--nuscenes_version",
        type=str,
        default="v1.0-trainval",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens",
        help="Path to Qwen2.5-VL model (the base visual encoder)",
    )
    parser.add_argument(
        "--anno_pkl",
        type=str,
        default="/data/jykim/projects/OpenDriveVLA/data/nuscenes/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl",
        help="Annotation PKL for turn/non-turn classification",
    )
    parser.add_argument(
        "--spacedrive_ckpt",
        type=str,
        default=None,
        help="Optional: SpaceDrive checkpoint .pth (loads LoRA-fine-tuned visual encoder)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "all"],
    )
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--turn_angle_threshold", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="workspace/cross_view_similarity/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
