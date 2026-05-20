#!/usr/bin/env python3
"""Inference with all past motion memory ablated.

This runtime patch leaves source model code untouched. During generation only,
it zeros the full past motion memory used by ego-status:
  - memory_canbus[:, :L, :]
  - memory_egopose[:, :L, :, :]
Then it restores the original temporal memory before the normal update step.
Current command/current can_bus are still passed normally.
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
    if getattr(original_test_generation, "_past_motion_all_ablation", False):
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
            ablated_memory_canbus[:, :ego_status_len] = 0

            self.memory_egopose = ablated_memory_egopose
            self.memory_canbus = ablated_memory_canbus

            if debug:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
                if rank == 0:
                    print(
                        "[ablate_past_motion_all] zeroed "
                        f"memory_canbus/memory_egopose for {ego_status_len} past frames"
                    )

            return original_test_generation(self, img, img_metas, **data)
        finally:
            self.memory_egopose = original_memory_egopose
            self.memory_canbus = original_memory_canbus

    wrapped_test_generation._past_motion_all_ablation = True
    SpaceDrive.test_generation = wrapped_test_generation


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    _patch_ablation()
    main = _load_base_test_main()
    main()
