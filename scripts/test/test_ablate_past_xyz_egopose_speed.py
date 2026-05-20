#!/usr/bin/env python3
"""Inference with past xyz/PE, full past ego-pose, and past speed ablated.

This runtime patch leaves source model code untouched. During generation only,
it zeros:
  - memory_egopose[:, :L, :, :]         past ego pose and past xyz PE
  - memory_canbus[:, :L, 11]            past speed, from raw can_bus[10]

The original temporal memory is restored before the normal update step.
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


def _patch_ablation():
    from projects.mmdet3d_plugin.models.vlm.spacedrive import SpaceDrive

    original_test_generation = SpaceDrive.test_generation
    if getattr(original_test_generation, "_past_xyz_egopose_speed_ablation", False):
        return

    debug = os.environ.get("SPACEDRIVE_ABLATE_DEBUG", "0") == "1"

    def wrapped_test_generation(self, img, img_metas, **data):
        memory_egopose = getattr(self, "memory_egopose", None)
        memory_canbus = getattr(self, "memory_canbus", None)
        if memory_egopose is None or memory_canbus is None:
            return original_test_generation(self, img, img_metas, **data)

        original_memory_egopose = memory_egopose
        original_memory_canbus = memory_canbus
        try:
            ablated_memory_egopose = memory_egopose.clone()
            ablated_memory_canbus = memory_canbus.clone()
            ego_status_len = min(
                int(getattr(self, "ego_status_len", ablated_memory_egopose.shape[1])),
                ablated_memory_egopose.shape[1],
                ablated_memory_canbus.shape[1],
            )

            ablated_memory_egopose[:, :ego_status_len] = 0
            # memory_canbus = [command, can_bus[0], ..., can_bus[12]]
            # Codebase convention: raw can_bus[10] is ego speed.
            ablated_memory_canbus[:, :ego_status_len, 11] = 0

            self.memory_egopose = ablated_memory_egopose
            self.memory_canbus = ablated_memory_canbus

            if debug:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
                if rank == 0:
                    print(
                        "[ablate_past_xyz_egopose_speed] zeroed "
                        f"memory_egopose[:, :{ego_status_len}] and "
                        f"memory_canbus[:, :{ego_status_len}, 11]"
                    )

            return original_test_generation(self, img, img_metas, **data)
        finally:
            self.memory_egopose = original_memory_egopose
            self.memory_canbus = original_memory_canbus

    wrapped_test_generation._past_xyz_egopose_speed_ablation = True
    SpaceDrive.test_generation = wrapped_test_generation


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    _patch_ablation()
    main = _load_base_test_main()
    main()
