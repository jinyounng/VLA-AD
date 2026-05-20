# ------------------------------------------------------------------------
# SpaceDrive
# Past ego-coordinate ablation test entrypoint.
# ------------------------------------------------------------------------
"""Run the normal SpaceDrive test loop with past ego xyz coordinates ablated.

This file intentionally does not modify the model implementation.  It patches
the test-time generation method in memory, zeros only the translation column of
``memory_egopose`` while ego features are built, then restores the original
memory before SpaceDrive updates temporal state for the next sample.
"""

import importlib.util
import os
from pathlib import Path

import torch


def _load_base_test_main():
    test_py = Path(__file__).with_name('test.py')
    spec = importlib.util.spec_from_file_location('spacedrive_base_test', test_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _patch_past_ego_xyz_ablation():
    from projects.mmdet3d_plugin.models.vlm.spacedrive import SpaceDrive

    original_test_generation = SpaceDrive.test_generation
    if getattr(original_test_generation, '_past_ego_xyz_ablation', False):
        return

    debug = os.environ.get('SPACEDRIVE_ABLATE_DEBUG', '0') == '1'

    def wrapped_test_generation(self, img, img_metas, **data):
        memory_egopose = getattr(self, 'memory_egopose', None)
        if memory_egopose is None:
            return original_test_generation(self, img, img_metas, **data)

        original_memory_egopose = memory_egopose
        try:
            ablated_memory_egopose = memory_egopose.clone()
            ego_status_len = min(
                int(getattr(self, 'ego_status_len', ablated_memory_egopose.shape[1])),
                ablated_memory_egopose.shape[1],
            )
            ablated_memory_egopose[:, :ego_status_len, :3, 3] = 0
            self.memory_egopose = ablated_memory_egopose

            if debug:
                rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                if rank == 0:
                    print(
                        '[ablate_past_ego_xyz] zeroed '
                        f'memory_egopose[:, :{ego_status_len}, :3, 3] for this generation'
                    )

            return original_test_generation(self, img, img_metas, **data)
        finally:
            self.memory_egopose = original_memory_egopose

    wrapped_test_generation._past_ego_xyz_ablation = True
    SpaceDrive.test_generation = wrapped_test_generation


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    _patch_past_ego_xyz_ablation()
    main = _load_base_test_main()
    save_path = main()
