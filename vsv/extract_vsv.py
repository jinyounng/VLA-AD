"""
Extract VSV (Visual Steering Vector) from SpaceDrive's Qwen2.5-VL model.

For each calibration sample:
  1. Forward pass with real pixel_values  → last-token hidden state per layer
  2. Forward pass with no-image input     → last-token hidden state per layer
  3. diff = h_with_img - h_no_img

Stack all diffs, apply PCA (rank=1), reshape to [num_layers, hidden_dim].

Usage:
    python vsv/extract_vsv.py \\
        projects/configs/spacedrive/spacedrive_qwen.py \\
        ckpts/your_checkpoint.pth \\
        --num-samples 200 \\
        --output vsv/vsv_direction.pt
"""

import argparse
import os
import sys
import traceback
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

# Make sure the project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from vsv.llm_layers import get_layers
from vsv.pca_utils import PCA

# Qwen2.5-VL vision token IDs
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_TOKEN_ID = 151655


# ---------------------------------------------------------------------------
# Forward-hook based hidden-state extractor
# ---------------------------------------------------------------------------

class HiddenStateCollector:
    """
    Registers forward hooks on transformer layers.
    After a forward pass, self.hidden[layer_idx] contains the last-token
    hidden state (float32, CPU) for that layer.
    """

    def __init__(self, llm: nn.Module):
        self.layers = get_layers(llm)
        self.hidden: list = []   # filled after forward
        self._hooks: list = []

    def __enter__(self):
        self.hidden = [None] * len(self.layers)
        for idx, layer in enumerate(self.layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            # out can be a tuple (hidden_state, ...) or just a tensor
            tensor = out[0] if isinstance(out, tuple) else out
            # Extract last token: [B, T, D] or [T, D] → [D]
            if tensor.dim() == 3:
                self.hidden[idx] = tensor[0, -1, :].float().cpu()
            elif tensor.dim() == 2:
                self.hidden[idx] = tensor[-1, :].float().cpu()
            else:
                self.hidden[idx] = tensor.float().cpu().flatten()[-1:]
        return hook

    def get_stacked(self) -> torch.Tensor:
        """Return [num_layers, hidden_dim] tensor."""
        return torch.stack(self.hidden, dim=0)


# ---------------------------------------------------------------------------
# "No-image" input construction
# ---------------------------------------------------------------------------

def remove_vision_tokens(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Remove all vision spans (<|vision_start|> … <|vision_end|>) from input_ids.
    input_ids: [1, seq_len]
    Returns: [1, new_seq_len]
    """
    ids = input_ids[0].tolist()
    result = []
    inside = False
    for tok in ids:
        if tok == VISION_START_ID:
            inside = True
        elif tok == VISION_END_ID:
            inside = False
        elif not inside:
            result.append(tok)
    return torch.tensor([result], dtype=input_ids.dtype, device=input_ids.device)


def make_no_image_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Build (input_ids_no_img, attn_mask_no_img) with vision tokens removed.
    """
    ids_no_img = remove_vision_tokens(input_ids)
    attn_no_img = torch.ones(
        (1, ids_no_img.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device
    )
    return ids_no_img, attn_no_img


# ---------------------------------------------------------------------------
# Unwrap helpers
# ---------------------------------------------------------------------------

def get_llm(model: nn.Module) -> nn.Module:
    """Return the lm_head module from a (possibly MMDataParallel-wrapped) SpaceDrive model."""
    m = model.module if hasattr(model, "module") else model
    return m.lm_head


def get_llm_forward_kwargs(batch: dict, device: torch.device):
    """
    Extract the minimal kwargs needed for a direct lm_head forward call.
    Mirrors the preprocessing done in SpaceDrive.test_generation_pts.
    """
    from mmcv.parallel import DataContainer as DC

    def _unwrap(v):
        while True:
            if isinstance(v, DC):
                v = v.data
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[0]
            else:
                break
        return v

    def _first(v):
        v = _unwrap(v)
        if isinstance(v, torch.Tensor):
            # Ensure at least 2D so model always sees a batch dimension.
            # 0-D scalar → [1, 1]; 1-D [N] → [1, N]; 2-D+ untouched.
            if v.dim() == 0:
                v = v.unsqueeze(0).unsqueeze(0)
            elif v.dim() == 1:
                v = v.unsqueeze(0)
            return v.to(device)
        return v

    input_ids = _first(batch.get("input_ids"))
    pixel_values = _first(batch.get("pixel_values"))
    image_grid_thw = _first(batch.get("image_grid_thw"))
    attention_mask = _first(batch.get("attention_mask"))

    # Qwen2.5-VL expects pixel_values flat: [total_patches, patch_dim]
    if pixel_values is not None and pixel_values.dim() > 2:
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
    if image_grid_thw is not None and image_grid_thw.dim() > 2:
        image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])

    return input_ids, pixel_values, image_grid_thw, attention_mask


# ---------------------------------------------------------------------------
# VSV extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_vsv(
    model: nn.Module,
    data_loader,
    num_samples: int,
    rank: int = 1,
) -> torch.Tensor:
    """
    Extract VSV direction of shape [num_layers, hidden_dim].

    Args:
        model       : SpaceDrive (MMDataParallel or bare)
        data_loader : mmdet3d test dataloader
        num_samples : number of calibration samples
        rank        : PCA rank (default 1)
    """
    llm = get_llm(model)
    device = next(llm.parameters()).device

    diffs = []      # list of [num_layers * hidden_dim] tensors

    pbar = tqdm(total=num_samples, desc="[VSV] Extracting", unit="sample", disable=False, dynamic_ncols=True, file=sys.stderr)
    for batch_idx, batch in enumerate(data_loader):
        if len(diffs) >= num_samples:
            break

        try:
            input_ids, pixel_values, image_grid_thw, attention_mask = \
                get_llm_forward_kwargs(batch, device)
        except Exception as e:
            pbar.set_postfix_str(f"skip batch {batch_idx}: {e}")
            traceback.print_exc(file=sys.stderr)
            continue

        if input_ids is None or pixel_values is None:
            continue

        # ---- Debug: print shapes on first batch ----
        if batch_idx == 0:
            print(f"\n[VSV] input_ids:       {input_ids.shape if input_ids is not None else None}", file=sys.stderr)
            print(f"[VSV] pixel_values:    {pixel_values.shape if pixel_values is not None else None}", file=sys.stderr)
            print(f"[VSV] image_grid_thw:  {image_grid_thw.shape if image_grid_thw is not None else None}", file=sys.stderr)
            print(f"[VSV] attention_mask:  {attention_mask.shape if attention_mask is not None else None}", file=sys.stderr)

        # ---- Forward with image ----
        try:
            collector_img = HiddenStateCollector(llm)
            with collector_img:
                llm(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
            h_img = collector_img.get_stacked()   # [num_layers, hidden_dim]
        except Exception as e:
            pbar.set_postfix_str(f"fwd(img) failed: {e}")
            traceback.print_exc(file=sys.stderr)
            continue

        # ---- Forward without image ----
        try:
            ids_no_img, attn_no_img = make_no_image_inputs(input_ids, attention_mask)
            collector_no = HiddenStateCollector(llm)
            with collector_no:
                llm(
                    input_ids=ids_no_img,
                    pixel_values=None,
                    image_grid_thw=None,
                    attention_mask=attn_no_img,
                    use_cache=False,
                    output_hidden_states=False,
                )
            h_no = collector_no.get_stacked()     # [num_layers, hidden_dim]
        except Exception as e:
            pbar.set_postfix_str(f"fwd(no-img) failed: {e}")
            traceback.print_exc(file=sys.stderr)
            continue

        diff = (h_img - h_no).view(-1)             # [num_layers * hidden_dim]
        diffs.append(diff)
        pbar.update(1)

    pbar.close()

    if len(diffs) < 2:
        raise RuntimeError(f"Too few valid samples ({len(diffs)}) to fit PCA. Check data/model.")

    print(f"[VSV] Running PCA on {len(diffs)} samples ...")
    fit_data = torch.stack(diffs).float()               # [N, num_layers * hidden_dim]

    pca = PCA(n_components=rank).to(fit_data.device)
    pca.fit(fit_data)

    # Reconstruct direction: components_ [rank, D] + mean_ [1, D]
    direction = (
        pca.components_.sum(dim=0, keepdim=True) + pca.mean_
    ).mean(0)                                           # [num_layers * hidden_dim]

    num_layers = h_img.shape[0]
    hidden_dim = h_img.shape[1]
    direction = direction.view(num_layers, hidden_dim)  # [num_layers, hidden_dim]

    print(f"[VSV] Extracted direction: {direction.shape}")
    return direction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract VSV for SpaceDrive")
    parser.add_argument("config", help="SpaceDrive test config file")
    parser.add_argument("checkpoint", help="SpaceDrive checkpoint (.pth)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of calibration samples")
    parser.add_argument("--output", default="vsv/vsv_direction.pt",
                        help="Output path for VSV tensor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # plugin import
    if getattr(cfg, "plugin", False) and getattr(cfg, "plugin_dir", None):
        import importlib
        parts = cfg.plugin_dir.split("/")
        mod_path = ".".join(parts[:-1]) if parts[-1] == "" else ".".join(parts)
        importlib.import_module(mod_path.replace("/", "."))

    set_random_seed(args.seed)

    # dataset / dataloader — use train set for calibration (avoid test leakage).
    # Train pipeline includes annotation loaders that break in test_mode, so
    # replace it with the test pipeline (no ann_info required).
    import copy
    train_cfg = copy.deepcopy(cfg.data.train)
    train_cfg.test_mode = True
    train_cfg.pipeline = cfg.data.test.pipeline
    dataset = build_dataset(train_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg:
        wrap_fp16_model(model)

    if args.checkpoint != "none":
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    else:
        print("[VSV] Warning: no checkpoint loaded — using random weights")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    direction = extract_vsv(model, data_loader, num_samples=args.num_samples)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(direction, args.output)
    print(f"[VSV] Saved direction ({direction.shape}) → {args.output}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
