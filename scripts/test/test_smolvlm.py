# ------------------------------------------------------------------------
# SpaceDrive - SmolVLM Inference / Evaluation Script
# Generates text responses from NuScenes multi-view images + VQA questions
# ------------------------------------------------------------------------

import os
import sys
import json
import argparse
import logging
import pickle
from typing import List, Dict

import torch
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: None

class _MockDTensor:
    pass

try:
    from torch.distributed.tensor import DTensor as _real_DTensor
except ImportError:
    _real_DTensor = _MockDTensor
    import torch.distributed.tensor as _dt
    _dt.DTensor = _MockDTensor

import transformers
if not hasattr(transformers.modeling_utils, "DTensor"):
    transformers.modeling_utils.DTensor = _real_DTensor

from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

logger = logging.getLogger(__name__)

CAMERA_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT",
]


def resolve_image_path(data_root: str, raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    for prefix in ("./data/nuscenes/", "data/nuscenes/"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):].lstrip("/")
            return os.path.join(data_root, rel)
    if os.path.isabs(normalized):
        return normalized
    return os.path.join(data_root, normalized)


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLM inference on NuScenes VQA")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model or LoRA adapter dir")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path (required for LoRA checkpoints)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="NuScenes data root")
    parser.add_argument("--anno_path", type=str,
                        default="nuscenes2d_ego_temporal_infos_val.pkl")
    parser.add_argument("--vqa_dir", type=str, default="vqa/val/")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--system_message", type=str,
                        default="You are a helpful autonomous driving assistant.")
    parser.add_argument("--use_all_cameras", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit number of samples (-1 = all)")
    return parser.parse_args()


def load_model_and_processor(args):
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))

    if is_lora:
        base_path = args.base_model_path
        if base_path is None:
            with open(os.path.join(args.model_path, "adapter_config.json")) as f:
                adapter_cfg = json.load(f)
            base_path = adapter_cfg.get("base_model_name_or_path", None)
        if base_path is None:
            raise ValueError("LoRA checkpoint detected but --base_model_path not provided")
        logger.info(f"Loading base model from {base_path}")
        model = AutoModelForVision2Seq.from_pretrained(
            base_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        logger.info(f"Loading LoRA adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    else:
        logger.info(f"Loading full model from {args.model_path}")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.eval()
    return model, processor


def build_test_samples(data_root, anno_path, vqa_dir, use_all_cameras):
    pkl_path = os.path.join(data_root, anno_path)
    logger.info(f"Loading annotations from {pkl_path}")
    with open(pkl_path, "rb") as f:
        anno_data = pickle.load(f)
    infos = anno_data["infos"]

    vqa_path = os.path.join(data_root, vqa_dir)
    samples = []

    for info in infos:
        token = info["token"]
        vqa_file = os.path.join(vqa_path, f"{token}.json")
        if not os.path.exists(vqa_file):
            continue

        with open(vqa_file) as f:
            qa_pairs = json.load(f)

        cam_paths = []
        if use_all_cameras:
            for cam_name in CAMERA_ORDER:
                cam_paths.append(
                    resolve_image_path(data_root, info["cams"][cam_name]["data_path"])
                )
        else:
            cam_paths.append(
                resolve_image_path(data_root, info["cams"]["CAM_FRONT"]["data_path"])
            )

        location = info.get("location", "").split("-")[0]

        for qa in qa_pairs:
            question = qa.get("question", qa.get("Q", ""))
            answer = qa.get("answer", qa.get("A", ""))
            if not question:
                continue
            samples.append({
                "token": token,
                "cam_paths": cam_paths,
                "question": question,
                "reference": answer,
                "location": location,
            })

    return samples


@torch.no_grad()
def run_generation(model, processor, samples, args) -> List[Dict]:
    results = []
    device = next(model.parameters()).device

    for sample in tqdm(samples, desc="Generating"):
        images = [Image.open(p).convert("RGB") for p in sample["cam_paths"]]

        image_content = [{"type": "image"} for _ in images]
        location_prefix = f"You are driving in {sample['location']}. " if sample["location"] else ""

        messages = [
            {
                "role": "user",
                "content": image_content + [
                    {"type": "text", "text": location_prefix + sample["question"]},
                ],
            },
        ]

        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = processor(
            text=[text_input],
            images=[images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_new_tokens + 1024,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
        )
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        output_ids = model.generate(**inputs, **gen_kwargs)
        trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()

        results.append({
            "token": sample["token"],
            "question": sample["question"],
            "reference": sample["reference"],
            "generated": generated,
        })

    return results


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    model, processor = load_model_and_processor(args)

    samples = build_test_samples(
        args.data_root, args.anno_path, args.vqa_dir, args.use_all_cameras
    )
    logger.info(f"Loaded {len(samples)} test samples")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logger.info(f"Limiting to {len(samples)} samples")

    if args.output_path is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.output_path = os.path.join("workspace", f"smolvlm_results_{model_name}.json")

    results = run_generation(model, processor, samples, args)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output_path}")

    for r in results[:3]:
        print(f"\n{'='*60}")
        print(f"[Question]  {r['question'][:150]}")
        print(f"[Reference] {r['reference'][:200]}")
        print(f"[Generated] {r['generated'][:200]}")


if __name__ == "__main__":
    main()
