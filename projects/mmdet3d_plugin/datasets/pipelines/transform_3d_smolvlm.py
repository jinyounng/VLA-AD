# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# SmolVLM-specific data pipeline transforms.
#
# Key bug fixes over transform_3d.py:
#   1. do_image_splitting=False  → 1 tile per camera, perfect PE alignment
#   2. image_grid_thw derived from model config (not hardcoded [[1,8,8]])
#   3. '\nAssistant:' prefix properly masked in labels (Bug #3)
#   4. coords_pos_tensor correctly handled per-batch-item
# ------------------------------------------------------------------------

import json
import os
import re
import copy
import random
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from transformers import AutoProcessor, AutoTokenizer

from ..utils.constants import (
    DEFAULT_IMAGE_TOKEN, POS_EMBEDDING_TOKEN, POS_INDICATOR_TOKEN, IGNORE_INDEX,
)
from .transform_3d import (
    LoadAnnoatationVQA, LoadAnnoatationVQATest, format_number,
)


def _derive_smolvlm_grid(processor_path: str):
    """Read vision config and return (grid_size, tokens_per_image).

    For SmolVLM-256M: image_size=512, patch_size=16, scale_factor=4
    → grid_size = 512 // 16 // 4 = 8  →  64 tokens per image tile.
    """
    cfg_path = os.path.join(processor_path, 'config.json')
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    vis_cfg = model_cfg.get('vision_config', {})
    img_size    = vis_cfg.get('image_size', 512)
    patch_size  = vis_cfg.get('patch_size', 16)
    scale_factor = model_cfg.get('scale_factor', 4)
    grid_size = img_size // patch_size // scale_factor  # 8
    return grid_size, grid_size * grid_size              # 8, 64


# ---------------------------------------------------------------------------
# Training transform
# ---------------------------------------------------------------------------

@PIPELINES.register_module()
class LoadAnnoatationVQASmolVLM(LoadAnnoatationVQA):
    """SmolVLM-specific training VQA loader.

    Inherits data-loading utilities (preprocess_vqa, convert_coords_to_pos_embedding,
    etc.) from LoadAnnoatationVQA and overrides __init__ / __call__ to fix the
    SmolVLM-specific bugs.
    """

    def __init__(self, processor, tokenizer=None, max_length=131072, **kwargs):
        # Let base class do all the book-keeping (file paths, lane info, flags …).
        super().__init__(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            llm_type='smolvlm',
            **kwargs,
        )

        # ── FIX #1: reload processor with do_image_splitting=False ──────────
        # With splitting enabled every 640×640 camera image produces a local
        # tile AND a global thumbnail (≥2 tiles/camera).  The position-encoding
        # pipeline computes one 8×8 grid per camera, so the extra global-
        # thumbnail tokens receive zero PE → completely wrong spatial context.
        # Disabling splitting gives exactly 1 tile per camera, 64 tokens/camera,
        # which matches the PE grid perfectly.
        self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length)
        self.processor.image_processor.do_image_splitting = False
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.processor.tokenizer = self.tokenizer

        # ── FIX #2: derive token grid from model config ──────────────────────
        self._grid_size, self._tokens_per_image = _derive_smolvlm_grid(processor)

        # ── FIX #3: compute '\nAssistant:' prefix length for label masking ───
        # The Idefics3 chat template appends '\nAssistant: ' before the answer.
        # Those tokens must be masked so the model isn't taught to predict them.
        self._assistant_prefix_len = len(
            self.processor.tokenizer('\nAssistant:', add_special_tokens=False)['input_ids']
        )

    # ------------------------------------------------------------------
    def __call__(self, results):
        # ── 1. Load planning trajectory ────────────────────────────────
        traj = None
        traj_question = None
        has_gt_planning = False

        if 'gt_planning' in results:
            planning_traj = results['gt_planning'][:, :, :2]
            mask = results['gt_planning_mask'][:].any(axis=-1)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                formatted_points = ', '.join(
                    f"({format_number(p[0], 2)}, {format_number(p[1], 2)})"
                    for p in planning_traj
                )
                traj_question = "Please provide the planning trajectory for the ego car without reasons."
                if self.load_ego_command_in_question:
                    traj_question = results['command_desc'] + traj_question
                traj = f"Here is the planning trajectory [{formatted_points}]."
                if self.load_high_level_command:
                    traj = results['command_desc'] + traj

        # ── 2. Build sources ───────────────────────────────────────────
        sources = []
        prompt = f"You are driving in {results['location']}. "

        if not self.planning_only:
            sources = self.preprocess_vqa(results, traj)

        random.shuffle(sources)

        has_gt_planning = (
            not self.counter_only
            and 'gt_planning' in results
            and traj is not None
        )

        if has_gt_planning:
            sources = [[
                {"from": "human", "value": traj_question},
                {"from": "gpt",   "value": traj},
            ]] + sources

        if not sources:
            sources = [[{"from": "human", "value": ""}]]

        # ── 3. Build SmolVLM conversation ──────────────────────────────
        vqa_anno = [item for pair in sources for item in pair]
        image_prefix = (DEFAULT_IMAGE_TOKEN + '\n') * len(results['img'])
        vqa_anno[0]['value'] = image_prefix + prompt + vqa_anno[0]['value']

        # Substitute coordinate tokens
        coords_pos_list = []
        if self.load_3d_pos:
            iterate = (
                [i for i, c in enumerate(vqa_anno) if c['from'] == 'human']
                if self.counter_only
                else range(len(vqa_anno))
            )
            for i in iterate:
                modified, coords = self.convert_coords_to_pos_embedding(vqa_anno[i]['value'])
                vqa_anno[i]['value'] = modified
                coords_pos_list.append(coords)
            if self.single_token_output and len(vqa_anno) > 1 and vqa_anno[1]['value']:
                vqa_anno[1]['value'] = (
                    vqa_anno[1]['value'].split(POS_INDICATOR_TOKEN)[0]
                    + POS_INDICATOR_TOKEN + POS_EMBEDDING_TOKEN + ']'
                )

        # Convert to Idefics3 role/content format
        for item in vqa_anno:
            item['role'] = 'user' if 'human' in item.pop('from') else 'assistant'
            item['content'] = [{"type": "text", "text": item.pop('value')}]

        text_prompt = self.processor.apply_chat_template(
            [vqa_anno], add_generation_prompt=False
        )

        # ── 4. Tokenise + encode images ────────────────────────────────
        results['img'] = [img.astype(np.uint8) for img in results['img']]
        inputs = self.processor(
            images=results['img'], text=text_prompt, return_tensors='pt'
        )

        input_ids   = inputs['input_ids']   # (1, seq_len)
        pixel_values = inputs['pixel_values']
        labels      = input_ids.clone()

        # Mask <image> tokens
        img_tok_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        labels[input_ids == img_tok_id] = IGNORE_INDEX

        # ── FIX #3: mask user turn AND '\nAssistant:' prefix ──────────
        eou_id = self.processor.tokenizer.convert_tokens_to_ids('<end_of_utterance>')
        eou_pos = (input_ids[0] == eou_id).nonzero(as_tuple=True)[0]
        if len(eou_pos) > 0:
            first_eou = eou_pos[0].item()
            mask_end  = first_eou + 1 + self._assistant_prefix_len
            labels[0, :mask_end] = IGNORE_INDEX
        else:
            labels[0, :input_ids.shape[1] // 2] = IGNORE_INDEX

        vlm_labels = labels[0]
        input_ids  = input_ids[0]

        # ── FIX #2: image_grid_thw from config-derived grid_size ──────
        num_cameras = len(results['img'])
        image_grid_thw = torch.tensor(
            [[1, self._grid_size, self._grid_size]] * num_cameras,
            device=input_ids.device,
        ).reshape(num_cameras, 3)

        # ── 5. Store results ───────────────────────────────────────────
        results['input_ids']       = input_ids
        results['vlm_labels']      = vlm_labels
        results['image_grid_thw']  = image_grid_thw
        results['pixel_values']    = pixel_values
        results['coords_pos_tensor'] = (
            torch.cat(coords_pos_list, dim=0)
            if self.load_3d_pos and coords_pos_list
            else torch.empty((0, 2))
        )
        results['has_gt_planning'] = has_gt_planning
        return results


# ---------------------------------------------------------------------------
# Test / inference transform
# ---------------------------------------------------------------------------

@PIPELINES.register_module()
class LoadAnnoatationVQATestSmolVLM(LoadAnnoatationVQATest):
    """SmolVLM-specific test VQA loader."""

    def __init__(self, processor, tokenizer=None, max_length=131072, **kwargs):
        super().__init__(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            llm_type='smolvlm',
            **kwargs,
        )

        # Same fixes as training class
        self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length)
        self.processor.image_processor.do_image_splitting = False
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.processor.tokenizer = self.tokenizer

        self._grid_size, self._tokens_per_image = _derive_smolvlm_grid(processor)

    # ------------------------------------------------------------------
    def __call__(self, results):
        sources = self.preprocess_vqa(results)
        prompt  = f"You are driving in {results['location']}. "

        coords_pos_list = []

        if 'planning' in self.load_type:
            for anno in sources:
                if self.load_ego_command_in_question:
                    anno[0]['value'] = results['command_desc'] + anno[0]['value']
                anno[0]['value'] = (
                    (DEFAULT_IMAGE_TOKEN + '\n') * len(results['img'])
                    + prompt + anno[0]['value']
                )
        else:
            if 'counter' in self.load_type:
                sources = [sources[self.counter_idx]]
            for anno in sources:
                anno[0]['value'] = (
                    (DEFAULT_IMAGE_TOKEN + '\n') * len(results['img'])
                    + prompt + anno[0]['value']
                )
            if self.load_3d_pos:
                for i, anno in enumerate(sources):
                    modified, coords = self.convert_coords_to_pos_embedding(anno[0]['value'])
                    anno[0]['value'] = modified
                    coords_pos_list.append(coords)

        # Use last (or only) source for SmolVLM
        anno = sources[-1]

        for item in anno:
            item['role'] = 'user' if 'human' in item.pop('from') else 'assistant'
            item['content'] = [{"type": "text", "text": item.pop('value')}]

        text_prompt = self.processor.apply_chat_template(
            [anno], add_generation_prompt=True
        )

        results['img'] = [img.astype(np.uint8) for img in results['img']]
        inputs = self.processor(
            images=results['img'], text=text_prompt, return_tensors='pt'
        )

        input_ids    = inputs['input_ids'][0]
        pixel_values = inputs['pixel_values']

        # FIX #2: config-derived grid
        num_cameras = len(results['img'])
        image_grid_thw = torch.tensor(
            [[1, self._grid_size, self._grid_size]] * num_cameras,
            device=input_ids.device,
        ).reshape(num_cameras, 3)

        results['input_ids']      = input_ids
        results['attention_mask'] = torch.ones_like(input_ids)
        results['pixel_values']   = pixel_values
        results['image_grid_thw'] = image_grid_thw

        results['coords_pos_list']   = coords_pos_list
        results['coords_pos_tensor'] = (
            torch.cat(coords_pos_list, dim=0)
            if coords_pos_list
            else torch.empty((0, 2))
        )
        return results
