# ------------------------------------------------------------------------
# SpaceDrive - SmolVLM Fine-tuning Trainer
# Standalone HuggingFace Trainer for SmolVLM-256M (Vision-Language Model)
# Uses NuScenes multi-view images + VQA data
# ------------------------------------------------------------------------

import os
import json
import random
import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import transformers
if not hasattr(transformers.modeling_utils, "DTensor"):
    transformers.modeling_utils.DTensor = _real_DTensor

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ──────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="ckpts/SmolVLM-256M-Instruct",
        metadata={"help": "Path to pretrained SmolVLM checkpoint"},
    )
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of LoRA target modules"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "torch dtype: float32, float16, bfloat16"},
    )


@dataclass
class DataArguments:
    data_root: str = field(
        metadata={"help": "NuScenes data root (contains samples/, vqa/, etc.)"},
    )
    anno_path: str = field(
        default="nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl",
        metadata={"help": "Annotation pkl filename (relative to data_root)"},
    )
    vqa_dir: str = field(
        default="vqa/train/",
        metadata={"help": "VQA directory (relative to data_root)"},
    )
    eval_vqa_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Eval VQA directory (relative to data_root)"},
    )
    eval_anno_path: Optional[str] = field(
        default=None,
        metadata={"help": "Eval annotation pkl filename"},
    )
    max_length: int = field(default=2048)
    system_message: str = field(
        default="You are a helpful autonomous driving assistant.",
    )
    use_all_cameras: bool = field(
        default=False,
        metadata={"help": "If True, use all 6 cameras. If False, use CAM_FRONT only."},
    )


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

CAMERA_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT",
]


def resolve_image_path(data_root: str, raw_path: str) -> str:
    """Resolve a NuScenes relative image path to an absolute path."""
    normalized = raw_path.replace("\\", "/")
    for prefix in ("./data/nuscenes/", "data/nuscenes/"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):].lstrip("/")
            return os.path.join(data_root, rel)
    if os.path.isabs(normalized):
        return normalized
    return os.path.join(data_root, normalized)


class NuScenesVLMDataset(Dataset):
    """NuScenes VQA dataset for SmolVLM fine-tuning.

    Each sample pairs camera image(s) with a QA conversation.
    Returns dict with 'images' (list of PIL) and 'messages' (chat format).
    """

    def __init__(
        self,
        data_root: str,
        anno_path: str,
        vqa_dir: str,
        processor: AutoProcessor,
        system_message: str,
        max_length: int,
        use_all_cameras: bool = False,
    ):
        self.data_root = data_root
        self.processor = processor
        self.system_message = system_message
        self.max_length = max_length
        self.use_all_cameras = use_all_cameras

        pkl_path = os.path.join(data_root, anno_path)
        logger.info(f"Loading annotations from {pkl_path}")
        with open(pkl_path, "rb") as f:
            anno_data = pickle.load(f)
        infos = anno_data["infos"]

        vqa_path = os.path.join(data_root, vqa_dir)
        logger.info(f"Loading VQA from {vqa_path}")

        self.samples = []
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

            for qa in qa_pairs:
                question = qa.get("question", qa.get("Q", ""))
                answer = qa.get("answer", qa.get("A", ""))
                if not question or not answer:
                    continue
                self.samples.append({
                    "cam_paths": cam_paths,
                    "question": question,
                    "answer": answer,
                    "location": info.get("location", "").split("-")[0],
                })

        logger.info(f"Built {len(self.samples)} VLM samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        for p in sample["cam_paths"]:
            img = Image.open(p).convert("RGB")
            images.append(img)

        image_content = [{"type": "image"} for _ in images]
        location_prefix = f"You are driving in {sample['location']}. " if sample["location"] else ""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": image_content + [
                    {"type": "text", "text": location_prefix + sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]

        return {"images": images, "messages": messages}


# ──────────────────────────────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────────────────────────────

class SmolVLMCollator:
    """Collates VLM samples: applies chat template, tokenizes with images."""

    def __init__(self, processor: AutoProcessor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[dict]) -> dict:
        texts = []
        all_images = []

        for sample in batch:
            text = self.processor.apply_chat_template(
                sample["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text)
            all_images.append(sample["images"])

        inputs = self.processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = IGNORE_INDEX

        # Mask everything before the assistant response
        for i, text in enumerate(texts):
            parts = text.split("assistant\n")
            if len(parts) >= 2:
                prompt_part = "assistant\n".join(parts[:-1]) + "assistant\n"
                prompt_ids = self.processor.tokenizer.encode(
                    prompt_part, add_special_tokens=False
                )
                labels[i, : len(prompt_ids)] = IGNORE_INDEX

        inputs["labels"] = labels
        return inputs


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )
    logger.info(f"Training args: {training_args}")

    set_seed(training_args.seed)

    # ── Model & Processor ──
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)

    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # ── LoRA ──
    if model_args.use_lora:
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ── Data ──
    train_dataset = NuScenesVLMDataset(
        data_root=data_args.data_root,
        anno_path=data_args.anno_path,
        vqa_dir=data_args.vqa_dir,
        processor=processor,
        system_message=data_args.system_message,
        max_length=data_args.max_length,
        use_all_cameras=data_args.use_all_cameras,
    )

    eval_dataset = None
    if data_args.eval_vqa_dir:
        eval_anno = data_args.eval_anno_path or data_args.anno_path.replace("train", "val")
        eval_dataset = NuScenesVLMDataset(
            data_root=data_args.data_root,
            anno_path=eval_anno,
            vqa_dir=data_args.eval_vqa_dir,
            processor=processor,
            system_message=data_args.system_message,
            max_length=data_args.max_length,
            use_all_cameras=data_args.use_all_cameras,
        )

    collator = SmolVLMCollator(processor=processor, max_length=data_args.max_length)

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()

    safe_save_dir = os.path.join(training_args.output_dir, "final_model")
    if model_args.use_lora:
        model.save_pretrained(safe_save_dir)
    else:
        trainer.save_model(safe_save_dir)
    processor.save_pretrained(safe_save_dir)
    logger.info(f"Model saved to {safe_save_dir}")


if __name__ == "__main__":
    main()
