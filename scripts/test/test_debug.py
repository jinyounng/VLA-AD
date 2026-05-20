#!/usr/bin/env python3
# ------------------------------------------------------------------------
# SpaceDrive debug test entry (non-intrusive)
# - Original files are untouched.
# - Dumps one input sample summary before normal test loop.
# ------------------------------------------------------------------------

import argparse
import json
import os
from pathlib import Path
import warnings

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test debug runner")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
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
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        "--dump-one-input",
        action="store_true",
        help="Dump one dataloader sample summary and continue normal test.",
    )
    parser.add_argument(
        "--debug-only-one-sample",
        action="store_true",
        help="Dump one sample then exit without running full evaluation.",
    )
    parser.add_argument(
        "--dump-one-input-out",
        default=None,
        help="Path to JSON summary output. Default: <work_dir>/debug_one_input.json",
    )

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError("--options and --eval-options cannot be both specified")
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def _safe_repr(value, max_len=160):
    text = repr(value)
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def _summarize_obj(x, depth=0, max_depth=3):
    if depth > max_depth:
        return {"type": type(x).__name__, "note": "max_depth_reached"}

    if hasattr(x, "data") and hasattr(x, "cpu_only"):
        data = getattr(x, "data", None)
        return {
            "type": type(x).__name__,
            "cpu_only": bool(getattr(x, "cpu_only", False)),
            "stack": bool(getattr(x, "stack", False)),
            "data_summary": _summarize_obj(data, depth + 1, max_depth),
        }

    if torch.is_tensor(x):
        return {
            "type": "Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[str(k)] = _summarize_obj(v, depth + 1, max_depth)
        return {"type": "dict", "keys": list(x.keys()), "items": out}

    if isinstance(x, (list, tuple)):
        preview = [_summarize_obj(v, depth + 1, max_depth) for v in list(x)[:3]]
        return {
            "type": type(x).__name__,
            "len": len(x),
            "preview_first_3": preview,
        }

    if isinstance(x, (str, int, float, bool)) or x is None:
        return {"type": type(x).__name__, "value": x}

    return {"type": type(x).__name__, "repr": _safe_repr(x)}


def _unwrap_datacontainer(x):
    if hasattr(x, "data") and hasattr(x, "cpu_only"):
        return x.data
    return x


def _pick_first_tensor_from_nested(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            found = _pick_first_tensor_from_nested(item)
            if found is not None:
                return found
    if isinstance(x, dict):
        for item in x.values():
            found = _pick_first_tensor_from_nested(item)
            if found is not None:
                return found
    return None


def _pick_first_dict_from_nested(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            found = _pick_first_dict_from_nested(item)
            if found is not None:
                return found
    return None


def _try_decode_prompt(first_batch, cfg):
    decoded = None
    decode_error = None
    token_count = None

    input_ids_raw = first_batch.get("input_ids")
    if input_ids_raw is None:
        return decoded, decode_error, token_count

    input_ids_unwrapped = _unwrap_datacontainer(input_ids_raw)
    input_ids_tensor = _pick_first_tensor_from_nested(input_ids_unwrapped)
    if input_ids_tensor is None:
        return decoded, "input_ids tensor not found in batch", token_count

    # Expect [seq] or [1, seq]
    if input_ids_tensor.dim() >= 2:
        ids_1d = input_ids_tensor[0]
    else:
        ids_1d = input_ids_tensor
    token_count = int(ids_1d.numel())

    tokenizer_path = getattr(cfg, "tokenizer_path", None)
    if not tokenizer_path:
        return decoded, "tokenizer_path missing in config", token_count

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        decoded = tokenizer.decode(ids_1d.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception as e:
        decode_error = f"{type(e).__name__}: {e}"

    return decoded, decode_error, token_count


def _dump_one_input(data_loader, cfg, output_path=None):
    iterator = iter(data_loader)
    first_batch = next(iterator)
    img_metas_unwrapped = _unwrap_datacontainer(first_batch.get("img_metas"))
    first_meta = _pick_first_dict_from_nested(img_metas_unwrapped) if img_metas_unwrapped is not None else None
    decoded_prompt, decode_error, input_token_count = _try_decode_prompt(first_batch, cfg)

    payload = {
        "note": "First batch summary from test dataloader",
        "config": str(getattr(cfg, "filename", "unknown")),
        "batch_top_level_keys": list(first_batch.keys()),
        "debug_prompt": {
            "decoded_text": decoded_prompt,
            "decode_error": decode_error,
            "input_token_count": input_token_count,
        },
        "debug_meta": {
            "sample_idx": first_meta.get("sample_idx") if isinstance(first_meta, dict) else None,
            "scene_token": first_meta.get("scene_token") if isinstance(first_meta, dict) else None,
            "filename": first_meta.get("filename") if isinstance(first_meta, dict) else None,
        },
        "summary": _summarize_obj(first_batch, depth=0, max_depth=3),
    }

    if output_path is None:
        work_dir = getattr(cfg, "work_dir", ".")
        output_path = os.path.join(work_dir, "debug_one_input.json")
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[debug] dumped first input summary to: {output_path}")


def main():
    args = parse_args()

    debug_dump_only = args.dump_one_input and args.debug_only_one_sample
    assert debug_dump_only or args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation with --out/--eval/--format-only/--show/--show-dir "
        "(or use --dump-one-input --debug-only-one-sample for debug dump only mode)."
    )
    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib

        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config).split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            importlib.import_module(_module_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    distributed = args.launcher != "none"
    if distributed:
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

    if args.dump_one_input:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            _dump_one_input(data_loader, cfg, args.dump_one_input_out)
        if args.debug_only_one_sample:
            print("[debug] --debug-only-one-sample enabled. Exit after first input dump.")
            return None

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if args.checkpoint != "none":
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES
        if "PALETTE" in checkpoint.get("meta", {}):
            model.PALETTE = checkpoint["meta"]["PALETTE"]
    else:
        print("No checkpoint file is specified, the model will not load any checkpoint.")
        if hasattr(dataset, "PALETTE"):
            model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    return model.module.save_path


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
