"""
SpaceDrive evaluation with VSV (Visual Steering Vector) applied.

Identical to scripts/test/test.py, but inserts VSVLayer into the LLM's
transformer MLP blocks before running inference.

Usage:
    # 1. Extract VSV first (one-time)
    python vsv/extract_vsv.py \\
        projects/configs/spacedrive/spacedrive_qwen.py \\
        ckpts/your_checkpoint.pth \\
        --num-samples 200 \\
        --output vsv/vsv_direction.pt

    # 2. Evaluate with VSV
    python vsv/test_with_vsv.py \\
        projects/configs/spacedrive/spacedrive_qwen.py \\
        ckpts/your_checkpoint.pth \\
        --vsv-path vsv/vsv_direction.pt \\
        --vsv-lambda 0.5 \\
        --out results_vsv.pkl

    # Lambda sweep: 0.1, 0.5, 1.0, 2.0
    for lam in 0.1 0.5 1.0 2.0; do
        python vsv/test_with_vsv.py \\
            projects/configs/spacedrive/spacedrive_qwen.py \\
            ckpts/your_checkpoint.pth \\
            --vsv-path vsv/vsv_direction.pt \\
            --vsv-lambda $lam \\
            --out results_vsv_lam${lam}.pkl
    done
"""

import argparse
import os
import sys
import warnings

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from vsv.llm_layers import add_vsv_layers, remove_vsv_layers


def parse_args():
    parser = argparse.ArgumentParser(description="SpaceDrive eval with VSV")
    parser.add_argument("config", help="Test config file")
    parser.add_argument("checkpoint", help="Checkpoint file")

    # VSV arguments
    parser.add_argument("--vsv-path", default=None,
                        help="Path to pre-extracted VSV direction (.pt). "
                             "If not provided, runs baseline (no VSV).")
    parser.add_argument("--vsv-lambda", type=float, default=0.5,
                        help="VSV steering strength (default: 0.5)")
    parser.add_argument("--vsv-layers", default=None,
                        help="Optional layer slice: 'start,end' or 'ss,se,ts,te'")

    # Standard test.py arguments
    parser.add_argument("--out", help="Output result file (.pkl)")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    parser.add_argument("--format-only", action="store_true")
    parser.add_argument("--eval", type=str, nargs="+")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir")
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--tmpdir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--eval-options", nargs="+", action=DictAction)
    parser.add_argument("--launcher",
                        choices=["none", "pytorch", "slurm", "mpi"],
                        default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError("--options and --eval-options cannot both be specified")
    if args.options:
        warnings.warn("--options is deprecated; use --eval-options")
        args.eval_options = args.options

    assert (
        args.out or args.eval or args.format_only or args.show or args.show_dir
    ), "Specify at least one of --out / --eval / --format-only / --show / --show-dir"

    if args.eval and args.format_only:
        raise ValueError("--eval and --format-only cannot both be specified")

    if args.out and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("Output file must be a .pkl file")

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports"):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if getattr(cfg, "plugin", False):
        import importlib
        if getattr(cfg, "plugin_dir", None):
            plugin_dir = cfg.plugin_dir
            parts = [p for p in plugin_dir.split("/") if p]
            mod_path = ".".join(parts)
            importlib.import_module(mod_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # Dataset
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # Model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16"):
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if args.checkpoint != "none":
        ckpt = load_checkpoint(model, args.checkpoint, map_location="cpu")
        model.CLASSES = ckpt.get("meta", {}).get("CLASSES", dataset.CLASSES)
        if "PALETTE" in ckpt.get("meta", {}):
            model.PALETTE = ckpt["meta"]["PALETTE"]
    else:
        print("[VSV] Warning: no checkpoint loaded")
        if hasattr(dataset, "CLASSES"):
            model.CLASSES = dataset.CLASSES
        if hasattr(dataset, "PALETTE"):
            model.PALETTE = dataset.PALETTE

    # ---- Apply VSV ----
    if args.vsv_path is not None:
        direction = torch.load(args.vsv_path, map_location="cpu")
        print(f"[VSV] Loaded direction from {args.vsv_path}, shape={direction.shape}")

        # Get LLM reference before wrapping with MMDataParallel
        lm_head = model.lm_head
        add_vsv_layers(lm_head, direction, lambda_val=args.vsv_lambda,
                       tar_layers=args.vsv_layers)
    else:
        print("[VSV] No --vsv-path given — running baseline (no VSV)")

    # ---- Inference ----
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    # ---- Save / Eval ----
    if args.out:
        import mmcv
        print(f"\nWriting results to {args.out}")
        mmcv.dump(outputs, args.out)

    kwargs = args.eval_options or {}
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect",
                    "save_best", "rule", "dynamic_intervals"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(kwargs)
        dataset.evaluate(outputs, args.eval, **eval_kwargs)

    return model.module.save_path if not distributed else None


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
