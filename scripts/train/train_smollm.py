# ------------------------------------------------------------------------
# SpaceDrive - SmolLM Fine-tuning Trainer
# Standalone HuggingFace Trainer for SmolLM-135M (text-only LLM)
# ------------------------------------------------------------------------

import os
import json
import random
import logging
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
import transformers

if not hasattr(transformers.modeling_utils, "DTensor"):
    transformers.modeling_utils.DTensor = _real_DTensor
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
        default="ckpts/SmolLM-135M",
        metadata={"help": "Path to pretrained SmolLM checkpoint"},
    )
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of LoRA target modules"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "torch dtype: float32, float16, bfloat16"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to training data (JSON, JSONL, or directory of JSONs)"},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data (same formats as data_path)"},
    )
    max_length: int = field(default=2048)
    data_format: str = field(
        default="auto",
        metadata={"help": "Data format: 'auto', 'conversations', 'nuscenes_vqa'"},
    )
    system_message: str = field(
        default="You are a helpful autonomous driving assistant.",
    )


# ──────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────

def load_json_or_jsonl(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            return json.load(f)


def load_nuscenes_vqa_dir(directory: str) -> List[dict]:
    """Load NuScenes-style VQA directory where each file is {sample_idx}.json
    containing a list of {question, answer} pairs."""
    samples = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath) as f:
            qa_pairs = json.load(f)
        convs = []
        for pair in qa_pairs:
            q = pair.get("question", pair.get("Q", ""))
            a = pair.get("answer", pair.get("A", ""))
            convs.append({"from": "human", "value": q})
            convs.append({"from": "gpt", "value": a})
        if convs:
            samples.append({"conversations": convs})
    return samples


def load_data(path: str, data_format: str) -> List[dict]:
    if data_format == "nuscenes_vqa" or (
        data_format == "auto" and os.path.isdir(path)
    ):
        return load_nuscenes_vqa_dir(path)

    raw = load_json_or_jsonl(path)
    if not raw:
        return []

    first = raw[0]
    if "conversations" in first:
        return raw
    if "question" in first and "answer" in first:
        return [
            {
                "conversations": [
                    {"from": "human", "value": item["question"]},
                    {"from": "gpt", "value": item["answer"]},
                ]
            }
            for item in raw
        ]
    raise ValueError(
        f"Unrecognized data format in {path}. Expected 'conversations' key "
        "or 'question'/'answer' keys."
    )


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class ConversationDataset(Dataset):
    """Tokenizes multi-turn conversations for causal LM fine-tuning.
    User turns are masked (IGNORE_INDEX); assistant turns are supervised."""

    ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    def __init__(
        self,
        data: List[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        system_message: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = system_message

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        convs = sample["conversations"]

        input_ids, labels = self._tokenize_conversation(convs)

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def _tokenize_conversation(self, convs: List[dict]):
        """Build input_ids and labels with chat template formatting.

        Uses <|im_start|>role\ncontent<|im_end|>\n format
        (compatible with SmolLM's special tokens).
        """
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)

        input_ids: List[int] = []
        labels: List[int] = []

        if self.system_message:
            system_tokens = self._encode_turn("system", self.system_message,
                                              im_start_id, im_end_id, nl_ids)
            input_ids.extend(system_tokens)
            labels.extend([IGNORE_INDEX] * len(system_tokens))

        for conv in convs:
            role = self.ROLE_MAP.get(conv.get("from", conv.get("role", "")), "user")
            content = conv.get("value", conv.get("content", ""))

            turn_tokens = self._encode_turn(role, content, im_start_id, im_end_id, nl_ids)
            input_ids.extend(turn_tokens)

            if role == "assistant":
                # mask the header: <|im_start|>assistant\n
                header = [im_start_id] + self.tokenizer.encode(
                    "assistant", add_special_tokens=False
                ) + nl_ids
                n_header = len(header)
                labels.extend([IGNORE_INDEX] * n_header + turn_tokens[n_header:])
            else:
                labels.extend([IGNORE_INDEX] * len(turn_tokens))

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _encode_turn(self, role: str, content: str, im_start_id, im_end_id, nl_ids):
        role_ids = self.tokenizer.encode(role, add_special_tokens=False)
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)
        return [im_start_id] + role_ids + nl_ids + content_ids + [im_end_id] + nl_ids


# ──────────────────────────────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PaddingCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


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

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── Model ──
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if model_args.use_lora:
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ── Data ──
    logger.info(f"Loading training data from {data_args.data_path}")
    train_data = load_data(data_args.data_path, data_args.data_format)
    logger.info(f"Loaded {len(train_data)} training samples")

    train_dataset = ConversationDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        system_message=data_args.system_message,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_data = load_data(data_args.eval_data_path, data_args.data_format)
        logger.info(f"Loaded {len(eval_data)} eval samples")
        eval_dataset = ConversationDataset(
            data=eval_data,
            tokenizer=tokenizer,
            max_length=data_args.max_length,
            system_message=data_args.system_message,
        )

    collator = PaddingCollator(tokenizer=tokenizer)

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()

    # Save final model
    safe_save_dir = os.path.join(training_args.output_dir, "final_model")
    if model_args.use_lora:
        model.save_pretrained(safe_save_dir)
    else:
        trainer.save_model(safe_save_dir)
    tokenizer.save_pretrained(safe_save_dir)
    logger.info(f"Model saved to {safe_save_dir}")


if __name__ == "__main__":
    main()
