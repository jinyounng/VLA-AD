#!/usr/bin/env python3
"""Text-only trajectory inference using a SpaceDrive (Qwen2.5-VL) checkpoint.

Instead of model.generate(), this script uses a single **forward pass** with
a teacher-forced answer template containing <POS_INDICATOR><POS_EMBEDDING>
slots.  Hidden states at the <POS_INDICATOR> positions are extracted and
decoded to (x, y) via the trained ``mlp_output_coords``.

This mirrors the training-time forward pass and avoids the problem of the
model not generating <POS_INDICATOR> tokens when image features are absent.

Usage
-----
    python infer_textonly.py \
        --checkpoint workspace/spacedrive_qwen/latest.pth \
        --prompts workspace/textonly_prompts.jsonl \
        --output workspace/textonly_results/ \
        --base-model ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens/
"""

import argparse
import json
import os
import pickle
import sys
import time
import contextlib
import io
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from projects.mmdet3d_plugin.models.vlm_utils.custom_qwen import (
    CustomQwen2_5_VLForConditionalGeneration,
)
from projects.mmdet3d_plugin.datasets.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    POS_EMBEDDING_TOKEN,
    POS_EMBEDDING_TOKEN_INDEX,
    POS_INDICATOR_TOKEN,
    POS_INDICATOR_TOKEN_INDEX,
)
from projects.mmdet3d_plugin.models.vlm_utils.positional_encoding import PositionalEncoding3D

NUM_PLAN_STEPS = 6

_SLOT = f"{POS_INDICATOR_TOKEN}{POS_EMBEDDING_TOKEN}"
ANSWER_TEMPLATE = (
    "Here is the planning trajectory ["
    + ", ".join([_SLOT] * NUM_PLAN_STEPS)
    + "]."
)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(base_model_path: str, lora_rank: int = 16):
    """Load base Qwen2.5-VL, apply LoRA, and create new_lm_head / new_embed_tokens."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map={"": torch.cuda.current_device()} if device.type == "cuda" else {"": "cpu"},
        local_files_only=True,
    )

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    hidden_dim = model.base_model.model.config.text_config.hidden_size

    # new_lm_head: small linear for POS_INDICATOR/POS_EMBEDDING logits
    new_lm_head = nn.Linear(hidden_dim, 2, bias=False).to(device=device, dtype=torch_dtype)
    model.base_model.model.new_lm_head = new_lm_head

    # new_embed_tokens: fine-tuned embeddings for the 2 special tokens
    new_embed_tokens = nn.Embedding(2, hidden_dim).to(device=device, dtype=torch_dtype)
    model.base_model.model.model.language_model.new_embed_tokens = new_embed_tokens

    return model


def build_mlp(hidden_dim: int = 3584):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 3),
    )


def build_ego_status_mlp(hidden_dim: int = 3584, ego_status_len: int = 2):
    in_dim = 14 * ego_status_len + 14 + 16 * ego_status_len
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def build_position_encoder(hidden_dim: int = 3584):
    # PositionalEncoding3D constructor is verbose (prints freq tensor).
    with contextlib.redirect_stdout(io.StringIO()):
        return PositionalEncoding3D(
            hidden_dim,
            dtype_override=torch.float32,
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model, mlp, checkpoint_path: str):
    """Load SpaceDrive detector checkpoint into PEFT model + MLP."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # In the detector, the VLM is stored under `lm_head.*`.
    # Our PEFT model params are `base_model.model.*`, so strip the `lm_head.` prefix.
    vlm_sd = {}
    for k, v in sd.items():
        if k.startswith("lm_head."):
            vlm_sd[k[len("lm_head."):]] = v

    missing, unexpected = model.load_state_dict(vlm_sd, strict=False)
    n_loaded = len(vlm_sd) - len(unexpected)
    print(f"  VLM: loaded {n_loaded}/{len(vlm_sd)} keys "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    # MLP: checkpoint keys are `mlp_output_coords.0.weight`, etc.
    mlp_sd = {}
    for k, v in sd.items():
        if k.startswith("mlp_output_coords."):
            mlp_sd[k[len("mlp_output_coords."):]] = v
    if mlp_sd:
        mlp.load_state_dict(mlp_sd, strict=True)
        print(f"  MLP: loaded {len(mlp_sd)} keys")
    else:
        print("  WARNING: no mlp_output_coords found in checkpoint")

    return sd


def load_optional_modules(
    sd: Dict[str, torch.Tensor],
    ego_status_mlp: Optional[nn.Module],
    position_encoder: Optional[nn.Module],
):
    if ego_status_mlp is not None:
        ego_sd = {}
        for k, v in sd.items():
            if k.startswith("ego_status_mlp."):
                ego_sd[k[len("ego_status_mlp."):]] = v
        if ego_sd:
            ego_status_mlp.load_state_dict(ego_sd, strict=True)
            print(f"  EgoStatusMLP: loaded {len(ego_sd)} keys")
        else:
            print("  WARNING: ego_status_mlp keys not found in checkpoint")

    if position_encoder is not None:
        pe_sd = {}
        for k, v in sd.items():
            if k.startswith("position_encoder."):
                pe_sd[k[len("position_encoder."):]] = v
        if pe_sd:
            missing, unexpected = position_encoder.load_state_dict(pe_sd, strict=False)
            print(
                f"  PositionEncoder: loaded {len(pe_sd)} keys "
                f"({len(missing)} missing, {len(unexpected)} unexpected)"
            )
        else:
            print("  WARNING: position_encoder keys not found in checkpoint")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def build_input_ids(
    tokenizer,
    q_text: str,
    device: torch.device,
    num_ego_tokens: int = 0,
):
    """Build input_ids with teacher-forced answer template.

    The Q field looks like:
        system\nYou are a helpful assistant.\nuser\n...\nassistant\n

    We parse system/user parts, then append the assistant answer template
    containing <POS_INDICATOR><POS_EMBEDDING> slot pairs.

    Returns
    -------
    input_ids : (1, L) long tensor
    attention_mask : (1, L)
    pe_mask : (L,) bool -- True at POS_EMBEDDING positions
    """
    lines = q_text.strip().split("\n") if q_text.strip() else []

    system_msg = ""
    user_msg = ""
    state = None
    for line in lines:
        stripped = line.strip()
        if stripped == "system":
            state = "system"
            continue
        elif stripped == "user":
            state = "user"
            continue
        elif stripped == "assistant":
            state = "assistant"
            continue
        if state == "system":
            system_msg += line + "\n"
        elif state == "user":
            user_msg += line + "\n"

    messages = []
    if system_msg.strip():
        messages.append({"role": "system", "content": system_msg.strip()})
    messages.append({"role": "user", "content": user_msg.strip()})
    messages.append({"role": "assistant", "content": ANSWER_TEMPLATE})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, return_tensors="pt", padding=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    if num_ego_tokens > 0:
        # Inject IMAGE_TOKEN placeholders directly in token space.
        # We place them right after BOS so they become deterministic slots for
        # ego-feature embedding replacement.
        insert_ids = torch.full(
            (1, num_ego_tokens),
            IMAGE_TOKEN_INDEX,
            dtype=input_ids.dtype,
            device=device,
        )
        insert_mask = torch.ones((1, num_ego_tokens), dtype=attn_mask.dtype, device=device)
        input_ids = torch.cat([input_ids[:, :1], insert_ids, input_ids[:, 1:]], dim=1)
        attn_mask = torch.cat([attn_mask[:, :1], insert_mask, attn_mask[:, 1:]], dim=1)

    pe_mask = (input_ids[0] == POS_EMBEDDING_TOKEN_INDEX)

    return input_ids, attn_mask, pe_mask


def load_ego_source_map(anno_pkl: str) -> Dict[str, Dict[str, float]]:
    data = pickle.load(open(anno_pkl, "rb"))
    info_map: Dict[str, Dict[str, float]] = {}
    for info in data["infos"]:
        token = str(info["token"])
        info_map[token] = {
            "command": float(info.get("gt_planning_command", 0)),
            "can_bus": info["can_bus"],
        }
    return info_map


def build_ego_feature(
    token: str,
    ego_source_map: Dict[str, Dict[str, float]],
    ego_status_mlp: nn.Module,
    position_encoder: nn.Module,
    device: torch.device,
    ego_status_len: int,
) -> Optional[torch.Tensor]:
    if token not in ego_source_map:
        return None

    src = ego_source_map[token]
    command = torch.tensor([[src["command"]]], device=device, dtype=torch.float32)  # (1,1)
    can_bus = torch.tensor(src["can_bus"], device=device, dtype=torch.float32).view(1, -1)  # (1,13)
    rec_can_bus = torch.cat([command, can_bus], dim=-1)  # (1,14)

    # Emulate SpaceDrive memory at standalone-sample inference:
    # no temporal stream -> zero past can-bus / zero past pose.
    memory_canbus = torch.zeros((1, ego_status_len, 14), device=device, dtype=torch.float32)
    memory_egopose = torch.zeros((1, ego_status_len, 4, 4), device=device, dtype=torch.float32)

    ego_mlp_input = torch.cat(
        [
            memory_canbus.reshape(1, -1),
            rec_can_bus.reshape(1, -1),
            memory_egopose.reshape(1, -1, 16).reshape(1, -1),
        ],
        dim=-1,
    )
    ego_token = ego_status_mlp(ego_mlp_input).unsqueeze(1)  # (1,1,H)

    past_xyz = memory_egopose[:, :ego_status_len, :3, 3]  # (1,L,3), all zeros
    encoded_past_xyz = position_encoder(past_xyz.reshape(1, -1, 3)).reshape(1, ego_status_len, -1)

    ego_feature = torch.cat([ego_token, encoded_past_xyz], dim=1)  # (1, 1+L, H)
    return ego_feature


def build_inputs_embeds_with_ego(
    model,
    input_ids: torch.LongTensor,
    ego_feature: torch.Tensor,
) -> torch.Tensor:
    base = model.base_model.model.model
    inputs_embeds = base.get_input_embeddings()(input_ids)

    # Keep special-token embedding behavior consistent with custom_qwen.py.
    if hasattr(base.language_model, "new_embed_tokens"):
        is_new_token = input_ids >= POS_INDICATOR_TOKEN_INDEX
        if is_new_token.any():
            rel_ids = input_ids[is_new_token] - POS_INDICATOR_TOKEN_INDEX
            new_embeds = base.language_model.new_embed_tokens(rel_ids)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[is_new_token] = new_embeds.to(inputs_embeds.dtype)

    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
    need = ego_feature.shape[1]
    if image_positions.numel() < need:
        raise RuntimeError(
            f"Not enough <image> placeholders for ego tokens: "
            f"found {image_positions.numel()}, need {need}"
        )

    target_positions = image_positions[:need]
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, target_positions, :] = ego_feature[0].to(inputs_embeds.dtype)
    return inputs_embeds


# ---------------------------------------------------------------------------
# Forward + decode
# ---------------------------------------------------------------------------

def forward_and_decode(
    model, mlp,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    pe_mask: torch.BoolTensor,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> List[List[float]]:
    """Single forward pass → extract hidden states at <POS_INDICATOR> → MLP decode.

    Training extracts hidden states by rolling the POS_EMBEDDING mask one
    position left (``torch.roll(mask, -1, dim=1)``).  Equivalently, for each
    POS_EMBEDDING at position *i*, the hidden state at position *i-1*
    (the POS_INDICATOR) is decoded.
    """
    result = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        return_dict=True,
    )

    # `last_hidden_state` is the transformer output before lm_head
    # (set by CustomQwen2_5_VLForConditionalGeneration.forward).
    hs = result.last_hidden_state  # (1, seq_len, hidden_dim)
    if hs is None:
        raise RuntimeError(
            "last_hidden_state is None.  Check model forward return.")

    pe_indices = pe_mask.nonzero(as_tuple=False).squeeze(-1)
    indicator_indices = pe_indices - 1  # POS_INDICATOR is one position before POS_EMBEDDING

    pos_hidden = hs[0, indicator_indices]  # (num_pe, hidden_dim)

    with torch.no_grad():
        decoded = mlp(pos_hidden.float())  # (num_pe, 3)

    coords = []
    for i in range(decoded.shape[0]):
        x, y = float(decoded[i, 0]), float(decoded[i, 1])
        coords.append([round(x, 4), round(y, 4)])

    return coords


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model",
                    default="ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens/")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=0, help="0 = unlimited")
    p.add_argument("--use-ego-status", action="store_true",
                   help="Inject SpaceDrive-plus ego status token(s) in text-only inference.")
    p.add_argument("--anno-pkl", type=str, default="",
                   help="nuScenes anno pkl with can_bus + gt_planning_command (required if --use-ego-status).")
    p.add_argument("--ego-status-len", type=int, default=2,
                   help="Past ego status length used in plus model (default: 2).")
    p.add_argument(
        "--ego-only",
        action="store_true",
        help="Ignore scene prompt text and predict from ego-status tokens only.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    t0 = time.time()

    print(f"[1/4] Loading base model: {args.base_model}")
    model = build_model(args.base_model, lora_rank=args.lora_rank)
    hidden_dim = model.base_model.model.config.text_config.hidden_size

    mlp = build_mlp(hidden_dim).to(device)
    ego_status_mlp = None
    position_encoder = None
    ego_source_map = None
    ego_token_count = 0
    if args.use_ego_status:
        if not args.anno_pkl:
            raise ValueError("--anno-pkl is required when --use-ego-status is enabled.")
        ego_status_mlp = build_ego_status_mlp(hidden_dim, args.ego_status_len).to(device)
        position_encoder = build_position_encoder(hidden_dim).to(device)
        ego_source_map = load_ego_source_map(args.anno_pkl)
        ego_token_count = 1 + args.ego_status_len
        print(f"  Ego status enabled: {len(ego_source_map)} token entries, {ego_token_count} ego token(s)")
    if args.ego_only and not args.use_ego_status:
        raise ValueError("--ego-only requires --use-ego-status.")

    print(f"[2/4] Loading checkpoint: {args.checkpoint}")
    sd = load_checkpoint(model, mlp, args.checkpoint)
    load_optional_modules(sd, ego_status_mlp, position_encoder)

    model.eval()
    mlp.eval()
    if ego_status_mlp is not None:
        ego_status_mlp.eval()
    if position_encoder is not None:
        position_encoder.eval()

    print(f"[3/4] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[4/4] Running inference → {args.output}")
    os.makedirs(args.output, exist_ok=True)

    count = 0
    with open(args.prompts, "r") as f:
        for line in f:
            rec = json.loads(line)
            token = rec["token"]
            q_text = "" if args.ego_only else rec.get("Q", "")

            input_ids, attn_mask, pe_mask = build_input_ids(
                tokenizer, q_text, device, num_ego_tokens=ego_token_count
            )

            inputs_embeds = None
            if args.use_ego_status:
                ego_feature = build_ego_feature(
                    token=token,
                    ego_source_map=ego_source_map,
                    ego_status_mlp=ego_status_mlp,
                    position_encoder=position_encoder,
                    device=device,
                    ego_status_len=args.ego_status_len,
                )
                if ego_feature is None:
                    print(f"  WARN: token {token[:16]} missing in anno map; fallback to no-ego.")
                else:
                    inputs_embeds = build_inputs_embeds_with_ego(model, input_ids, ego_feature)

            n_pe = pe_mask.sum().item()
            if n_pe != NUM_PLAN_STEPS:
                print(f"  WARN: {token[:16]} has {n_pe} POS_EMBEDDING slots, "
                      f"expected {NUM_PLAN_STEPS}")

            coords = forward_and_decode(
                model, mlp, input_ids, attn_mask, pe_mask, inputs_embeds=inputs_embeds
            )

            output_text = ANSWER_TEMPLATE
            for c in coords:
                output_text = output_text.replace(
                    POS_EMBEDDING_TOKEN, f"({c[0]:.2f}, {c[1]:.2f})", 1
                )

            out_record = [{
                "Q": q_text,
                "A": output_text,
                "trajectory_xy": coords,
                "mode": "ego_only" if args.ego_only else "prompt_plus_ego",
            }]
            out_path = os.path.join(args.output, token)
            with open(out_path, "w") as fout:
                json.dump(out_record, fout)

            count += 1
            if count % 50 == 0 or count == 1:
                elapsed = time.time() - t0
                print(f"  [{count}] {token[:16]}... "
                      f"coords={coords[0]} ({elapsed:.1f}s)")

            if args.max_samples > 0 and count >= args.max_samples:
                break

    elapsed = time.time() - t0
    print(f"\nDone: {count} samples in {elapsed:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
