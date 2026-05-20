# ------------------------------------------------------------------------
# SpaceDrive - SmolLM Inference / Evaluation Script
# ------------------------------------------------------------------------

import os
import sys
import json
import argparse
import logging
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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, "..", "train")
sys.path.insert(0, TRAIN_DIR)
from train_smollm import load_data, IGNORE_INDEX


def parse_args():
    parser = argparse.ArgumentParser(description="SmolLM inference / evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model (full) or LoRA adapter dir")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path (required for LoRA checkpoints)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Test data path (JSON, JSONL, or directory)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save generated results (JSON)")
    parser.add_argument("--data_format", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy decoding")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--system_message", type=str,
                        default="You are a helpful autonomous driving assistant.")
    parser.add_argument("--eval_loss", action="store_true",
                        help="Compute avg loss / perplexity on the test set")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    return parser.parse_args()


def load_model_and_tokenizer(args):
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
            raise ValueError("LoRA checkpoint detected but --base_model_path not provided "
                             "and base_model_name_or_path not found in adapter_config.json")
        logger.info(f"Loading base model from {base_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        logger.info(f"Loading LoRA adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    else:
        logger.info(f"Loading full model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def build_prompt(conversations: List[dict], system_message: str, tokenizer) -> str:
    """Build chat-formatted prompt from conversation turns (user turns only)."""
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    parts = []
    if system_message:
        parts.append(f"{im_start}system\n{system_message}{im_end}\n")

    for conv in conversations:
        role_raw = conv.get("from", conv.get("role", ""))
        role = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}.get(role_raw, role_raw)
        content = conv.get("value", conv.get("content", ""))

        if role == "assistant":
            break
        parts.append(f"{im_start}{role}\n{content}{im_end}\n")

    parts.append(f"{im_start}assistant\n")
    return "".join(parts)


def get_reference(conversations: List[dict]) -> str:
    """Extract the first assistant response as reference."""
    for conv in conversations:
        role = conv.get("from", conv.get("role", ""))
        if role in ("gpt", "assistant"):
            return conv.get("value", conv.get("content", ""))
    return ""


@torch.no_grad()
def run_generation(model, tokenizer, test_data, args) -> List[Dict]:
    results = []
    device = next(model.parameters()).device

    for sample in tqdm(test_data, desc="Generating"):
        convs = sample["conversations"]
        prompt = build_prompt(convs, args.system_message, tokenizer)
        reference = get_reference(convs)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_length).to(device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        output_ids = model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({"prompt": prompt, "reference": reference, "generated": generated})

    return results


@torch.no_grad()
def run_eval_loss(model, tokenizer, test_data, args) -> Dict:
    from train_smollm import ConversationDataset, PaddingCollator
    from torch.utils.data import DataLoader

    dataset = ConversationDataset(
        data=test_data, tokenizer=tokenizer,
        max_length=args.max_length, system_message=args.system_message)
    collator = PaddingCollator(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0

    for batch in tqdm(loader, desc="Eval loss"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        mask = batch["labels"] != IGNORE_INDEX
        total_loss += outputs.loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    import math
    ppl = math.exp(min(avg_loss, 100))
    return {"avg_loss": avg_loss, "perplexity": ppl, "total_tokens": total_tokens}


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    model, tokenizer = load_model_and_tokenizer(args)
    test_data = load_data(args.data_path, args.data_format)
    logger.info(f"Loaded {len(test_data)} test samples")

    if args.output_path is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.output_path = os.path.join("workspace", f"smollm_results_{model_name}.json")

    outputs = {}

    if args.eval_loss:
        metrics = run_eval_loss(model, tokenizer, test_data, args)
        logger.info(f"Loss: {metrics['avg_loss']:.4f}  |  PPL: {metrics['perplexity']:.2f}  |  Tokens: {metrics['total_tokens']}")
        outputs["metrics"] = metrics

    results = run_generation(model, tokenizer, test_data, args)
    outputs["results"] = results

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output_path}")

    for r in results[:3]:
        print(f"\n{'='*60}")
        print(f"[Reference] {r['reference'][:200]}")
        print(f"[Generated] {r['generated'][:200]}")


if __name__ == "__main__":
    main()
