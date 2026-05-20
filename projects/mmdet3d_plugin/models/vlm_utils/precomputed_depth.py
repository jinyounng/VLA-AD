from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence

import torch


class PrecomputedDepthStore:
    """Load per-sample precomputed depth tensors from disk.

    Expected file layout:
      <root>/<sample_idx>.pt

    Each file may store either:
      - a tensor of shape [num_views, H, W]
      - a dict containing {"depth": tensor}
    """

    _SENTINEL = object()

    def __init__(self, root: str, cache_size: int = 256):
        self.root = Path(root).expanduser()
        self._cache: OrderedDict[str, Optional[torch.Tensor]] = OrderedDict()
        self._cache_size = max(cache_size, 0)

    def _load_one(self, sample_idx: str) -> Optional[torch.Tensor]:
        cached = self._cache.get(sample_idx, self._SENTINEL)
        if cached is not self._SENTINEL:
            self._cache.move_to_end(sample_idx)
            return cached

        path = self.root / f"{sample_idx}.pt"
        if not path.exists():
            self._put_cache(sample_idx, None)
            return None

        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if "depth" not in data:
                return None
            data = data["depth"]

        if not isinstance(data, torch.Tensor):
            return None

        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        if data.dim() != 3:
            self._put_cache(sample_idx, None)
            return None
        data = data.contiguous()
        self._put_cache(sample_idx, data)
        return data

    def _put_cache(self, key: str, value: Optional[torch.Tensor]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[key] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def load_batch(
        self,
        sample_ids: Sequence[str],
        num_views: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        depth_list: List[torch.Tensor] = []
        for sample_id in sample_ids:
            depth = self._load_one(sample_id)
            if depth is None:
                return None
            if depth.shape[0] != num_views or depth.shape[1] != height or depth.shape[2] != width:
                return None
            depth_list.append(depth)

        if not depth_list:
            return None
        return torch.stack(depth_list, dim=0).to(device=device, dtype=dtype)
