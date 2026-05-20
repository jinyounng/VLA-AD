"""
VSVLayer and transformer layer management utilities.
Adapted from VISTA/llm_layers.py for SpaceDrive's Qwen2.5-VL backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class VSVLayer(nn.Module):
    """
    Inserted after each transformer MLP block.
    Steers hidden states toward the visual direction while preserving norm.

    vsv  : [N, hidden_dim] tensor  (N directions per layer; typically N=1)
    lam  : list of N lambda scalars
    """

    def __init__(self, vsv: torch.Tensor, lam: list, simple_mode: bool = False):
        super().__init__()
        # Register as buffer so it moves with the module's device/dtype
        self.register_buffer("vsv", vsv.float())   # [N, hidden_dim]
        self.lam = lam
        self.simple_mode = simple_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vsv is None:
            return x

        orig_dtype = x.dtype
        x = x.float()
        original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)

        y = torch.zeros_like(x)

        if self.simple_mode:
            vsv_vec = self.vsv[0]                               # [hidden_dim]
            lam_val = self.lam[0]
            y = lam_val * F.normalize(vsv_vec, dim=-1).expand_as(x)
        else:
            for i in range(len(self.vsv)):
                vsv_vec = self.vsv[i]                           # [hidden_dim]
                lam_val = self.lam[i]
                # adaptive lambda: stronger steering when x points opposite to vsv
                cos_sim = F.cosine_similarity(x, -vsv_vec.view(1, 1, -1), dim=-1)  # [B, T]
                lambda_sim = (1.0 + torch.clamp(cos_sim, min=0.0)).unsqueeze(-1)   # [B, T, 1]
                y = y + lam_val * lambda_sim * F.normalize(vsv_vec, dim=-1).expand_as(x)
            y = y / len(self.vsv)

        # Direction-shift with norm preservation
        x = F.normalize(F.normalize(x, p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        return x.to(orig_dtype)


# ---------------------------------------------------------------------------
# Helper utilities (model-agnostic)
# ---------------------------------------------------------------------------

def get_nested_attr(obj, attr_path: str):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path: str, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model: nn.Module, path: str = ""):
    """Recursively find the longest nn.ModuleList in a PyTorch model."""
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        child_path = f"{path}.{name}" if path else name

        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = child_path

        sub_path, sub_len = find_longest_modulelist(child, child_path)
        if sub_len > longest_len:
            longest_len = sub_len
            longest_path = sub_path

    return longest_path, longest_len


def find_module(block: nn.Module, keywords: list):
    """Find a sub-module by keyword search in its name."""
    for name, module in block.named_modules():
        if any(kw in name for kw in keywords):
            return module
    submodule_names = [n for n, _ in block.named_modules()]
    raise ValueError(f"Could not find {keywords} in: {submodule_names}")


def get_layers_path(model: nn.Module) -> str:
    longest_path, _ = find_longest_modulelist(model)
    return longest_path


# Try these paths before falling back to the longest-ModuleList heuristic.
# Needed because vision encoders often have more blocks than text decoder layers,
# and the heuristic would pick the wrong one (e.g. Qwen2.5-VL: 32 vision vs 28 text).
_DECODER_LAYER_PATHS = [
    "model.language_model.layers",  # Qwen2.5-VL (CustomQwen2_5_VLForConditionalGeneration)
    "model.layers",                  # LLaMA / Qwen2-VL variants
    "language_model.model.layers",   # other HF layouts
    "transformer.h",                 # GPT-2
]


def get_layers(model: nn.Module) -> nn.ModuleList:
    for path in _DECODER_LAYER_PATHS:
        try:
            layers = get_nested_attr(model, path)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                print(f"[VSV] Using decoder layers at path: {path} ({len(layers)} layers)")
                return layers
        except AttributeError:
            continue
    # Fallback: longest ModuleList heuristic
    path = get_layers_path(model)
    print(f"[VSV] Fallback: using longest ModuleList at path: {path}")
    return get_nested_attr(model, path)


# ---------------------------------------------------------------------------
# Add / remove VSV hooks
# ---------------------------------------------------------------------------

_MLP_KEYWORDS = ["mlp", "feedforward", "ffn"]


def add_vsv_layers(
    llm: nn.Module,
    direction: torch.Tensor,
    lambda_val: float,
    tar_layers: str = None,
):
    """
    Insert VSVLayer after each transformer MLP.

    Args:
        llm       : the LLM module (e.g. model.module.lm_head)
        direction : [num_layers, hidden_dim] tensor (output of extract_vsv)
        lambda_val: steering strength scalar
        tar_layers: optional "start,end" or "ss,se,ts,te" layer slice string
    """
    layers = get_layers(llm)
    assert len(direction) == len(layers), (
        f"direction has {len(direction)} layers but model has {len(layers)} layers"
    )

    vsv_slice = direction
    layer_slice = layers

    if tar_layers is not None:
        parts = [int(x) for x in tar_layers.split(",")]
        if len(parts) == 2:
            s, e = parts
            layer_slice = layers[s:e]
            vsv_slice = direction[s:e]
        elif len(parts) == 4:
            ss, se, ts, te = parts
            layer_slice = layers[ss:se]
            vsv_slice = direction[ts:te]
        else:
            raise ValueError(f"Invalid tar_layers: {tar_layers}")

    lam_list = [lambda_val]

    for i, layer in enumerate(layer_slice):
        original_mlp = find_module(layer, _MLP_KEYWORDS)
        # vsv_slice[i]: [hidden_dim] → unsqueeze to [1, hidden_dim] for VSVLayer
        vsv_i = vsv_slice[i].unsqueeze(0).to(original_mlp.parameters().__next__().device)
        layer.mlp = nn.Sequential(original_mlp, VSVLayer(vsv_i, lam_list))

    print(f"[VSV] Inserted VSVLayer into {len(layer_slice)} transformer layers (lambda={lambda_val})")


def remove_vsv_layers(llm: nn.Module):
    """Remove all VSVLayer hooks from transformer MLP blocks."""
    layers = get_layers(llm)
    removed = 0
    for layer in layers:
        try:
            mlp = find_module(layer, _MLP_KEYWORDS)
        except ValueError:
            continue
        if isinstance(mlp, nn.Sequential) and isinstance(mlp[-1], VSVLayer):
            layer.mlp = mlp[0]
            removed += 1
    print(f"[VSV] Removed VSVLayer from {removed} transformer layers")
