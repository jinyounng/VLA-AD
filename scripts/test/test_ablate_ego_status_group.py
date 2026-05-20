#!/usr/bin/env python3
"""Inference with one semantic ego-status group ablated.

This runtime patch leaves source model code untouched. Select exactly one group
with SPACEDRIVE_ABLATE_EGO_GROUP.

Supported groups, excluding command:
  past_quaternion      memory_canbus[:, :L, 1:5]
  past_acceleration    memory_canbus[:, :L, 5:8]
  past_angular_velocity memory_canbus[:, :L, 8:11]
  past_velocity        memory_canbus[:, :L, 11:14]
  past_egopose         memory_egopose[:, :L, :, :]

  current_quaternion   data["can_bus"][:, 0:4]
  current_acceleration data["can_bus"][:, 4:7]
  current_angular_velocity data["can_bus"][:, 7:10]
  current_velocity     data["can_bus"][:, 10:13]
  current_egopose      current ego_pose/ego_pose_inv used for memory localization

For all groups, the normal temporal memory update is preserved after generation.
"""

import importlib.util
import os
from pathlib import Path

import torch


PAST_CANBUS_GROUPS = {
    "past_quaternion": slice(1, 5),
    "past_acceleration": slice(5, 8),
    "past_angular_velocity": slice(8, 11),
    "past_velocity": slice(11, 14),
}

CURRENT_CANBUS_GROUPS = {
    "current_quaternion": slice(0, 4),
    "current_acceleration": slice(4, 7),
    "current_angular_velocity": slice(7, 10),
    "current_velocity": slice(10, 13),
}

SUPPORTED_GROUPS = set(PAST_CANBUS_GROUPS) | set(CURRENT_CANBUS_GROUPS) | {
    "past_egopose",
    "current_egopose",
}


def _load_base_test_main():
    test_py = Path(__file__).with_name("test.py")
    spec = importlib.util.spec_from_file_location("spacedrive_base_test", test_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _group_name():
    group = os.environ.get("SPACEDRIVE_ABLATE_EGO_GROUP")
    if group not in SUPPORTED_GROUPS:
        supported = ", ".join(sorted(SUPPORTED_GROUPS))
        raise RuntimeError(
            f"SPACEDRIVE_ABLATE_EGO_GROUP must be one of: {supported}. Got {group!r}"
        )
    return group


def _clone_state(model):
    state = {}
    for name in ("memory_canbus", "memory_egopose", "sample_time"):
        value = getattr(model, name, None)
        state[name] = value.clone() if torch.is_tensor(value) else value
    memory_count = getattr(model, "memory_count", 0)
    state["memory_count"] = memory_count.clone() if torch.is_tensor(memory_count) else memory_count
    return state


def _restore_state(model, state):
    for name, value in state.items():
        setattr(model, name, value)


def _identity_like_pose(pose):
    batch = pose.shape[0]
    return torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(batch, 1, 1)


def _patch_ablation():
    from projects.mmdet3d_plugin.models.vlm.spacedrive import SpaceDrive

    group = _group_name()
    debug = os.environ.get("SPACEDRIVE_ABLATE_DEBUG", "0") == "1"

    original_test_generation = SpaceDrive.test_generation
    original_forward_test = SpaceDrive.forward_test

    if not getattr(original_test_generation, "_ego_status_group_ablation", False):

        def wrapped_test_generation(self, img, img_metas, **data):
            original_memory_canbus = getattr(self, "memory_canbus", None)
            original_memory_egopose = getattr(self, "memory_egopose", None)

            patched_memory_canbus = None
            patched_memory_egopose = None
            patched_data = dict(data)
            ego_status_len = None

            try:
                if group in PAST_CANBUS_GROUPS:
                    if original_memory_canbus is not None:
                        patched_memory_canbus = original_memory_canbus.clone()
                        ego_status_len = min(
                            int(getattr(self, "ego_status_len", patched_memory_canbus.shape[1])),
                            patched_memory_canbus.shape[1],
                        )
                        patched_memory_canbus[:, :ego_status_len, PAST_CANBUS_GROUPS[group]] = 0
                        self.memory_canbus = patched_memory_canbus

                elif group == "past_egopose":
                    if original_memory_egopose is not None:
                        patched_memory_egopose = original_memory_egopose.clone()
                        ego_status_len = min(
                            int(getattr(self, "ego_status_len", patched_memory_egopose.shape[1])),
                            patched_memory_egopose.shape[1],
                        )
                        patched_memory_egopose[:, :ego_status_len] = 0
                        self.memory_egopose = patched_memory_egopose

                elif group in CURRENT_CANBUS_GROUPS and "can_bus" in patched_data:
                    can_bus = patched_data["can_bus"].clone()
                    can_bus[:, CURRENT_CANBUS_GROUPS[group]] = 0
                    patched_data["can_bus"] = can_bus

                if debug:
                    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
                    if rank == 0:
                        suffix = f", L={ego_status_len}" if ego_status_len is not None else ""
                        print(f"[ablate_ego_status_group] group={group}{suffix}")

                return original_test_generation(self, img, img_metas, **patched_data)
            finally:
                if patched_memory_canbus is not None:
                    self.memory_canbus = original_memory_canbus
                if patched_memory_egopose is not None:
                    self.memory_egopose = original_memory_egopose

        wrapped_test_generation._ego_status_group_ablation = True
        SpaceDrive.test_generation = wrapped_test_generation

    if group == "current_egopose" and not getattr(original_forward_test, "_current_egopose_ablation", False):

        def wrapped_forward_test(self, img, img_metas, rescale, **data):
            if self.ego_status is None:
                return original_forward_test(self, img, img_metas, rescale, **data)

            state_before_pre_update = _clone_state(self)

            ablated_data = dict(data)
            if "ego_pose" in ablated_data:
                ablated_data["ego_pose"] = _identity_like_pose(ablated_data["ego_pose"])
            if "ego_pose_inv" in ablated_data:
                ablated_data["ego_pose_inv"] = _identity_like_pose(ablated_data["ego_pose_inv"])

            self.pre_update_memory(ablated_data)

            generation_data = dict(data)
            for key in generation_data:
                if key not in ["question_text"]:
                    generation_data[key] = generation_data[key][0].unsqueeze(0)

            output = self.test_generation(img, img_metas, **generation_data)

            _restore_state(self, state_before_pre_update)
            self.pre_update_memory(data)

            for key in data:
                if key not in ["question_text"]:
                    data[key] = data[key][0].unsqueeze(0)

            rec_can_bus = torch.cat([data["command"].unsqueeze(-1), data["can_bus"]], dim=-1).unsqueeze(1)
            batch = rec_can_bus.shape[0]
            rec_ego_pose = torch.eye(4, device=rec_can_bus.device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)
            self.post_update_memory(data, rec_ego_pose, rec_can_bus)

            if debug:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
                if rank == 0:
                    print("[ablate_ego_status_group] group=current_egopose")

            return output

        wrapped_forward_test._current_egopose_ablation = True
        SpaceDrive.forward_test = wrapped_forward_test


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    _patch_ablation()
    main = _load_base_test_main()
    main()
