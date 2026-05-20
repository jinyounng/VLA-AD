#!/usr/bin/env python3
"""Precompute per-sample UniDepth predictions for SpaceDrive.

Output format:
  <output_dir>/<sample_idx>.pt

Each file stores:
  {
    "depth": Tensor[num_views, H, W] (cpu),
    "camera_order": List[str],
    "sample_idx": str,
  }
"""

import argparse
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute UniDepth cache for SpaceDrive.")
    parser.add_argument("--ann-file", required=True, help="Path to *.pkl annotation file.")
    parser.add_argument("--data-root", required=True, help="NuScenes root used by configs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-sample .pt files.")
    parser.add_argument("--image-size", type=int, default=640, help="Image resize size (default: 640).")
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0).")
    parser.add_argument("--backbone", choices=["s", "b", "l"], default="l", help="UniDepth v2 ViT size.")
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--start", type=int, default=0, help="Start index in annotation list.")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive). -1 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def resolve_sample_path(data_root, path):
    if not path or not isinstance(path, str):
        return path
    normalized = path.replace("\\", "/")
    for prefix in ("./data/nuscenes/", "data/nuscenes/"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):].lstrip("/")
            return os.path.join(data_root, rel)
    return path


def read_ann_infos(ann_file):
    data = mmcv.load(ann_file)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported annotation structure in {ann_file}")


def load_views(cam_infos, data_root, image_size):
    tensors = []
    cam_names = []
    for cam_name, cam_info in cam_infos.items():
        image_path = resolve_sample_path(data_root, cam_info["data_path"])
        img = mmcv.imread(image_path, flag="color")  # HWC, BGR, uint8
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        img = mmcv.imresize(img, (image_size, image_size))
        img = img[:, :, ::-1]  # BGR -> RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW
        tensors.append(torch.from_numpy(img).to(torch.float32))
        cam_names.append(cam_name)
    return torch.stack(tensors, dim=0), cam_names


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    unidepth_repo = repo_root / "unidepth"
    if str(unidepth_repo) not in sys.path:
        sys.path.insert(0, str(unidepth_repo))

    from unidepth.models import UniDepthV2  # pylint: disable=import-error

    infos = read_ann_infos(args.ann_file)
    start = max(args.start, 0)
    end = len(infos) if args.end < 0 else min(args.end, len(infos))
    infos = infos[start:end]

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"unidepth-v2-vit{args.backbone}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}").eval().to(args.device)
    model.interpolation_mode = "bilinear"

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    skipped = 0
    for info in tqdm(infos, desc="Precomputing depth"):
        sample_idx = str(info["token"])
        out_path = out_dir / f"{sample_idx}.pt"
        if out_path.exists() and not args.overwrite:
            continue

        try:
            views, cam_names = load_views(info["cams"], args.data_root, args.image_size)
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            print(f"[WARN] Skipping {sample_idx}: {e}")
            continue

        views = views.to(args.device, non_blocking=True)

        with torch.inference_mode():
            depth = model.infer(views)["depth"]  # [num_views, 1, H, W]
        depth = depth.squeeze(1)  # [num_views, H, W]
        depth = depth.to(save_dtype).cpu().contiguous()

        torch.save(
            {
                "depth": depth,
                "camera_order": cam_names,
                "sample_idx": sample_idx,
            },
            out_path,
        )

    print(f"Saved depth cache to: {out_dir}")
    if skipped:
        print(f"[WARN] Skipped {skipped} samples due to missing images.")


if __name__ == "__main__":
    main()
