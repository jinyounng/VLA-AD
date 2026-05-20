#!/usr/bin/env python3
"""Inference with one past memory_canbus component ablated.

This runtime patch leaves source model code untouched. During generation only,
it zeros one selected component:
  - memory_canbus[:, :L, SPACEDRIVE_ABLATE_CANBUS_INDEX]

Optionally, if SPACEDRIVE_ABLATE_WITH_PAST_XYZ_ZERO=1, it also zeros:
  - memory_egopose[:, :L, :3, 3]

The original temporal memory is restored before the normal update step.

memory_canbus layout:
  0      command
  1:14   raw can_bus[0:13]
"""

import importlib.util
import os
from pathlib import Path

import torch


def _load_base_test_main():
    test_py = Path(__file__).with_name("test.py")
    spec = importlib.util.spec_from_file_location("spacedrive_base_test", test_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _ablation_index():
    raw = os.environ.get("SPACEDRIVE_ABLATE_CANBUS_INDEX")
    if raw is None:
        raise RuntimeError("SPACEDRIVE_ABLATE_CANBUS_INDEX must be set")
    idx = int(raw)
    if idx < 0 or idx >= 14:
        raise RuntimeError(f"SPACEDRIVE_ABLATE_CANBUS_INDEX must be in [0, 13], got {idx}")
    return idx


def _patch_ablation():
    from projects.mmdet3d_plugin.models.vlm.spacedrive import SpaceDrive

    original_test_generation = SpaceDrive.test_generation
    if getattr(original_test_generation, "_past_canbus_index_ablation", False):
        return

    idx = _ablation_index()
    with_past_xyz_zero = os.environ.get("SPACEDRIVE_ABLATE_WITH_PAST_XYZ_ZERO", "0") == "1"
    debug = os.environ.get("SPACEDRIVE_ABLATE_DEBUG", "0") == "1"

    def wrapped_test_generation(self, img, img_metas, **data):
        memory_canbus = getattr(self, "memory_canbus", None)
        memory_egopose = getattr(self, "memory_egopose", None)
        if memory_canbus is None or (with_past_xyz_zero and memory_egopose is None):
            return original_test_generation(self, img, img_metas, **data)

        original_memory_canbus = memory_canbus
        original_memory_egopose = memory_egopose
        try:
            ablated_memory_canbus = memory_canbus.clone()
            ego_status_len = min(
                int(getattr(self, "ego_status_len", ablated_memory_canbus.shape[1])),
                ablated_memory_canbus.shape[1],
            )
            if with_past_xyz_zero:
                ego_status_len = min(ego_status_len, memory_egopose.shape[1])

            ablated_memory_canbus[:, :ego_status_len, idx] = 0
            self.memory_canbus = ablated_memory_canbus
            if with_past_xyz_zero:
                ablated_memory_egopose = memory_egopose.clone()
                ablated_memory_egopose[:, :ego_status_len, :3, 3] = 0
                self.memory_egopose = ablated_memory_egopose

            if debug:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
                if rank == 0:
                    print(
                        "[ablate_past_canbus_index] zeroed "
                        f"memory_canbus[:, :{ego_status_len}, {idx}]"
                        + (" with past xyz zero" if with_past_xyz_zero else "")
                    )

            return original_test_generation(self, img, img_metas, **data)
        finally:
            self.memory_canbus = original_memory_canbus
            if with_past_xyz_zero:
                self.memory_egopose = original_memory_egopose

    wrapped_test_generation._past_canbus_index_ablation = True
    SpaceDrive.test_generation = wrapped_test_generation


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    _patch_ablation()
    main = _load_base_test_main()
    main()
