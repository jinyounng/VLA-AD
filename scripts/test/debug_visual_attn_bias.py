import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import DataContainer
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.utils.constants import IMAGE_TOKEN_INDEX
from visual_attention_bias import VisualAttentionBiasConfig, install_visual_attention_bias


def parse_args():
    parser = argparse.ArgumentParser(description="Run one-sample visual attention bias debug inference.")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--sample-index", type=int, default=0, help="0-based sample index in the test dataloader")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings using key=value pairs",
    )
    parser.add_argument("--visual-attn-bias", action="store_true")
    parser.add_argument("--visual-attn-alpha", type=float, default=1.0)
    parser.add_argument("--visual-token-start", type=int, default=-1)
    parser.add_argument("--visual-token-end", type=int, default=-1)
    parser.add_argument("--visual-attn-layer-start", type=int, default=0)
    parser.add_argument("--visual-attn-layer-end", type=int, default=-1)
    parser.add_argument("--visual-attn-image-token-id", type=int, default=IMAGE_TOKEN_INDEX)
    parser.add_argument("--no-force-eager-attn", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=64)
    parser.add_argument(
        "--debug-output",
        default=None,
        help="optional path to dump visual attention debug JSON",
    )
    return parser.parse_args()


def _load_config(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib

        if hasattr(cfg, "plugin_dir"):
            module_dir = os.path.dirname(cfg.plugin_dir).split("/")
        else:
            module_dir = os.path.dirname(args.config).split("/")
        module_path = module_dir[0]
        for part in module_dir[1:]:
            module_path = module_path + "." + part
        importlib.import_module(module_path)

    return cfg


def _prepare_test_loader(cfg):
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

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return dataset, data_loader


def _unwrap(x):
    if isinstance(x, DataContainer):
        return x.data[0]
    return x


def _first_item(x):
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    return x


def main():
    args = parse_args()
    cfg = _load_config(args)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    dataset, data_loader = _prepare_test_loader(cfg)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint != "none":
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    if torch.cuda.is_available():
        model = model.cuda()
    if args.visual_attn_bias:
        bias_cfg = VisualAttentionBiasConfig(
            enabled=True,
            alpha=args.visual_attn_alpha,
            visual_start=args.visual_token_start,
            visual_end=args.visual_token_end,
            layer_start=args.visual_attn_layer_start,
            layer_end=args.visual_attn_layer_end,
            image_token_id=args.visual_attn_image_token_id,
            force_eager_attention=not args.no_force_eager_attn,
            debug=True,
            debug_limit=args.debug_limit,
        )
        patched_layers = install_visual_attention_bias(model, bias_cfg)
        print(
            "Visual attention bias debug enabled: "
            f"alpha={bias_cfg.alpha}, "
            f"visual_range=[{bias_cfg.visual_start}, {bias_cfg.visual_end}), "
            f"layers=[{bias_cfg.layer_start}, {bias_cfg.layer_end}], "
            f"patched_attention_modules={patched_layers}"
        )

    model = MMDataParallel(model, device_ids=[0] if torch.cuda.is_available() else None)
    model.eval()

    selected = None
    for idx, data in enumerate(data_loader):
        if idx == args.sample_index:
            selected = data
            break
    if selected is None:
        raise IndexError(f"sample_index={args.sample_index} out of range for dataset length {len(dataset)}")

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **selected)

    state = getattr(model.module, "_spacedrive_visual_attn_bias_state", None)
    if state is None:
        print("No visual attention bias state found on model.")
        return

    img_metas = _unwrap(selected["img_metas"])
    meta0 = _first_item(img_metas)
    sample_id = str(meta0["sample_idx"])
    payload = {
        "sample_index": args.sample_index,
        "sample_id": sample_id,
        "alpha": args.visual_attn_alpha,
        "visual_range_requested": [args.visual_token_start, args.visual_token_end],
        "visual_range_runtime": list(state.runtime_range) if state.runtime_range is not None else None,
        "layer_range": [args.visual_attn_layer_start, args.visual_attn_layer_end],
        "debug_records": state.debug_records,
        "result_preview": result[:1] if isinstance(result, list) else result,
    }

    debug_output = args.debug_output
    if debug_output is None:
        debug_output = os.path.join("workspace", f"visual_attn_debug_{sample_id}.json")
    os.makedirs(os.path.dirname(debug_output), exist_ok=True)
    with open(debug_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved debug output to {debug_output}")
    print(f"Collected {len(state.debug_records)} attention debug records for sample {sample_id}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
