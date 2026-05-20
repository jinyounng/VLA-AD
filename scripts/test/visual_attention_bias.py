import functools
import types
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VisualAttentionBiasConfig:
    enabled: bool = False
    alpha: float = 1.0
    visual_start: int = -1
    visual_end: int = -1
    layer_start: int = 0
    layer_end: int = -1
    image_token_id: Optional[int] = None
    force_eager_attention: bool = True
    debug: bool = False
    debug_limit: int = 32


class _VisualAttentionBiasState:
    def __init__(self, cfg: VisualAttentionBiasConfig):
        self.cfg = cfg
        self.runtime_range: Optional[Tuple[int, int]] = None
        self.active_layer: Optional[int] = None
        self._softmax_depth = 0
        self._orig_softmax = F.softmax
        self._orig_sdpa = getattr(F, "scaled_dot_product_attention", None)
        self.debug_records = []
        self._debug_keys = set()

    def set_runtime_range_from_input_ids(self, input_ids):
        if self.cfg.visual_start >= 0 and self.cfg.visual_end >= 0:
            self.runtime_range = (self.cfg.visual_start, self.cfg.visual_end)
            return
        if input_ids is None or self.cfg.image_token_id is None:
            self.runtime_range = None
            return
        image_pos = (input_ids[0] == self.cfg.image_token_id).nonzero(as_tuple=False).flatten()
        if image_pos.numel() == 0:
            self.runtime_range = None
            return
        self.runtime_range = (int(image_pos.min().item()), int(image_pos.max().item()) + 1)

    def current_visual_range(self, key_len: int) -> Optional[Tuple[int, int]]:
        if not self.cfg.enabled or self.cfg.alpha == 0:
            return None
        start_end = self.runtime_range
        if start_end is None:
            start_end = (self.cfg.visual_start, self.cfg.visual_end)
        start, end = start_end
        if start < 0 or end < 0:
            return None
        start = max(0, min(start, key_len))
        end = max(start, min(end, key_len))
        if start == end:
            return None
        return start, end

    def _maybe_record_debug(self, tensor, visual_range):
        if (
            not self.cfg.debug
            or self.active_layer is None
            or visual_range is None
            or len(self.debug_records) >= self.cfg.debug_limit
        ):
            return

        start, end = visual_range
        debug_key = (self.active_layer, tensor.shape[-2], tensor.shape[-1], start, end)
        if debug_key in self._debug_keys:
            return
        self._debug_keys.add(debug_key)

        detached = tensor.detach().to(torch.float32)
        visual_slice = detached[..., start:end]
        if start > 0:
            non_visual_left = detached[..., :start]
        else:
            non_visual_left = None
        if end < detached.shape[-1]:
            non_visual_right = detached[..., end:]
        else:
            non_visual_right = None

        non_visual_parts = [x for x in (non_visual_left, non_visual_right) if x is not None and x.numel() > 0]
        if non_visual_parts:
            non_visual_tensor = torch.cat(non_visual_parts, dim=-1)
            non_visual_mean = float(non_visual_tensor.mean().item())
        else:
            non_visual_mean = None

        record = {
            "layer_idx": int(self.active_layer),
            "query_len": int(detached.shape[-2]),
            "key_len": int(detached.shape[-1]),
            "visual_start": int(start),
            "visual_end": int(end),
            "alpha": float(self.cfg.alpha),
            "visual_mean_before": float(visual_slice.mean().item()),
            "visual_mean_after": float((visual_slice + self.cfg.alpha).mean().item()),
            "non_visual_mean": non_visual_mean,
        }
        self.debug_records.append(record)
        msg = (
            "[visual-attn-debug] "
            f"layer={record['layer_idx']} "
            f"q={record['query_len']} "
            f"k={record['key_len']} "
            f"visual=[{record['visual_start']},{record['visual_end']}) "
            f"visual_mean_before={record['visual_mean_before']:.6f} "
            f"visual_mean_after={record['visual_mean_after']:.6f}"
        )
        if record["non_visual_mean"] is not None:
            msg += f" non_visual_mean={record['non_visual_mean']:.6f}"
        print(msg)

    def _compute_sdpa_scores(self, query, key, scale=None):
        scale_factor = scale if scale is not None else (1.0 / (query.shape[-1] ** 0.5))
        return torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    def biased_softmax(self, input, dim=None, _stacklevel=3, dtype=None):
        if (
            self.active_layer is not None
            and dim in (-1, input.dim() - 1)
            and input.dim() >= 4
            and torch.is_floating_point(input)
        ):
            visual_range = self.current_visual_range(input.shape[-1])
            if visual_range is not None:
                self._maybe_record_debug(input, visual_range)
                start, end = visual_range
                input = input.clone()
                input[..., start:end] = input[..., start:end] + self.cfg.alpha
        return self._orig_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    def biased_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if self.active_layer is not None:
            visual_range = self.current_visual_range(key.shape[-2])
            if visual_range is not None and not (attn_mask is None and is_causal):
                if self.cfg.debug and len(self.debug_records) < self.cfg.debug_limit:
                    debug_scores = self._compute_sdpa_scores(query, key, scale=scale)
                    self._maybe_record_debug(debug_scores, visual_range)
                start, end = visual_range
                bias = torch.zeros(
                    (1, 1, query.shape[-2], key.shape[-2]),
                    dtype=query.dtype,
                    device=query.device,
                )
                bias[..., start:end] = self.cfg.alpha
                attn_mask = bias if attn_mask is None else attn_mask + bias.to(attn_mask.dtype)
        kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        if scale is not None:
            kwargs["scale"] = scale
        if enable_gqa:
            kwargs["enable_gqa"] = enable_gqa
        try:
            return self._orig_sdpa(query, key, value, **kwargs)
        except TypeError:
            kwargs.pop("scale", None)
            kwargs.pop("enable_gqa", None)
            return self._orig_sdpa(query, key, value, **kwargs)

    def __enter__(self):
        self._softmax_depth += 1
        if self._softmax_depth == 1:
            F.softmax = self.biased_softmax
            if self._orig_sdpa is not None:
                F.scaled_dot_product_attention = self.biased_scaled_dot_product_attention
        return self

    def __exit__(self, exc_type, exc, tb):
        self._softmax_depth -= 1
        if self._softmax_depth == 0:
            F.softmax = self._orig_softmax
            if self._orig_sdpa is not None:
                F.scaled_dot_product_attention = self._orig_sdpa


def _iter_decoder_attention_modules(model) -> Iterable[Tuple[int, str, torch.nn.Module]]:
    layer_modules = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__.lower().endswith("decoderlayer"):
            layer_modules.append((module_name, module))

    for layer_idx, (layer_name, layer) in enumerate(layer_modules):
        for child_name, child in layer.named_modules():
            if child_name and child.__class__.__name__.lower().endswith("attention"):
                if all(hasattr(child, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    name = f"{layer_name}.{child_name}"
                    yield layer_idx, name, child


def _patch_attention_forward(attn_module, layer_idx: int, state: _VisualAttentionBiasState):
    if getattr(attn_module, "_spacedrive_visual_attn_bias_patched", False):
        return False

    original_forward = attn_module.forward

    @functools.wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        previous_layer = state.active_layer
        state.active_layer = layer_idx
        try:
            with state:
                return original_forward(*args, **kwargs)
        finally:
            state.active_layer = previous_layer

    attn_module.forward = types.MethodType(patched_forward, attn_module)
    attn_module._spacedrive_visual_attn_bias_patched = True
    return True


def _patch_one_generate(module, state: _VisualAttentionBiasState):
    if not hasattr(module, "generate") or getattr(module, "_spacedrive_visual_attn_bias_generate_patched", False):
        return False
    original_generate = module.generate

    @functools.wraps(original_generate)
    def patched_generate(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        state.set_runtime_range_from_input_ids(input_ids)
        return original_generate(*args, **kwargs)

    module.generate = patched_generate
    module._spacedrive_visual_attn_bias_generate_patched = True
    return True


def _patch_generate(model, state: _VisualAttentionBiasState):
    patched = 0
    patched += int(_patch_one_generate(model, state))
    for module in model.modules():
        if module is model:
            continue
        patched += int(_patch_one_generate(module, state))
    return patched


def _force_eager_attention(model):
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"


def install_visual_attention_bias(model, cfg: VisualAttentionBiasConfig) -> int:
    if not cfg.enabled:
        return 0

    if cfg.force_eager_attention:
        _force_eager_attention(model)

    state = _VisualAttentionBiasState(cfg)
    patched = 0
    for layer_idx, _, attn in _iter_decoder_attention_modules(model):
        if layer_idx < cfg.layer_start:
            continue
        if cfg.layer_end >= 0 and layer_idx > cfg.layer_end:
            continue
        patched += int(_patch_attention_forward(attn, layer_idx, state))

    _patch_generate(model, state)
    model._spacedrive_visual_attn_bias_state = state
    return patched
