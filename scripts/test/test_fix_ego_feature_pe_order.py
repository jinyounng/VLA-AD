#!/usr/bin/env python3
"""Inference entrypoint with a runtime-only ego feature order fix.

This does not modify the model source. It monkeypatches Qwen's SpaceDrive
forward path so feature+PE ego features are scattered into the fake image-token
slots in the same order those slots appear in input_ids.
"""

import torch

from projects.mmdet3d_plugin.models.vlm_utils.custom_qwen import CustomQwen2_5_VLModel


_ORIG_QWEN_FORWARD = CustomQwen2_5_VLModel.forward


def _forward_with_fixed_ego_feature_pe_order(self, *args, **kwargs):
    ego_feature = kwargs.get("ego_feature", None)
    if (
        ego_feature is not None
        and getattr(ego_feature, "ndim", 0) == 3
        and ego_feature.shape[1] == 3
    ):
        # SpaceDrive currently builds ego_feature as [feature, PE1, PE2], while
        # input_ids contain fake image slots as [PE1, PE2, feature].
        kwargs["ego_feature"] = torch.cat([ego_feature[:, 1:], ego_feature[:, :1]], dim=1)
    return _ORIG_QWEN_FORWARD(self, *args, **kwargs)


CustomQwen2_5_VLModel.forward = _forward_with_fixed_ego_feature_pe_order

from scripts.test.test import main  # noqa: E402


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
