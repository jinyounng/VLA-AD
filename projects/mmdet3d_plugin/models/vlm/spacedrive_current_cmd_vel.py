from contextlib import contextmanager

import torch
from mmdet.models import DETECTORS

from .spacedrive import SpaceDrive


@DETECTORS.register_module()
class SpaceDriveCurrentCommandVelocity(SpaceDrive):
    """SpaceDrive variant that keeps only current command and velocity ego status.

    This class intentionally leaves the base SpaceDrive implementation untouched.
    It reuses the normal plus-model training/generation path, but temporarily
    masks ego-status tensors before the base class builds ``ego_feature``:

    - current command is preserved because it is concatenated separately from
      ``data["command"]`` into ``rec_can_bus``.
    - current velocity is preserved from ``data["can_bus"][:, 10:13]``.
    - all other current can_bus fields are zeroed.
    - all past can_bus and past egopose memory fields are zeroed.

    Use with ``ego_status="feature"``. Keeping ``PE`` enabled would reinsert
    past egopose positional tokens, which is outside the intended ablation.
    """

    velocity_slice = slice(10, 13)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ego_status is not None and "PE" in self.ego_status:
            raise ValueError(
                "SpaceDriveCurrentCommandVelocity should be used without PE "
                "ego status. Set ego_status='feature'."
            )

    def _mask_current_can_bus(self, data):
        if "can_bus" not in data or data["can_bus"] is None:
            return data

        patched = dict(data)
        can_bus = patched["can_bus"].clone()
        velocity = can_bus[:, self.velocity_slice].clone()
        can_bus.zero_()
        can_bus[:, self.velocity_slice] = velocity
        patched["can_bus"] = can_bus
        return patched

    @contextmanager
    def _current_command_velocity_only(self):
        original_memory_canbus = getattr(self, "memory_canbus", None)
        original_memory_egopose = getattr(self, "memory_egopose", None)
        try:
            if torch.is_tensor(original_memory_canbus):
                self.memory_canbus = torch.zeros_like(original_memory_canbus)
            if torch.is_tensor(original_memory_egopose):
                self.memory_egopose = torch.zeros_like(original_memory_egopose)
            yield
        finally:
            self.memory_canbus = original_memory_canbus
            self.memory_egopose = original_memory_egopose

    def forward_train_vlm(self, img_metas, input_ids, vlm_labels, vlm_attn_mask,
                          pixel_values, image_grid_thw, coords_pos_tensor, **data):
        patched_data = self._mask_current_can_bus(data)
        with self._current_command_velocity_only():
            return super().forward_train_vlm(
                img_metas,
                input_ids,
                vlm_labels,
                vlm_attn_mask,
                pixel_values,
                image_grid_thw,
                coords_pos_tensor,
                **patched_data,
            )

    def test_generation_pts(self, img, img_metas, input_ids, pixel_values,
                            image_grid_thw, attention_mask, **data):
        patched_data = self._mask_current_can_bus(data)
        with self._current_command_velocity_only():
            return super().test_generation_pts(
                img,
                img_metas,
                input_ids,
                pixel_values,
                image_grid_thw,
                attention_mask,
                **patched_data,
            )
