# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Qwen3-VL variant: same training/inference loop as SpaceDrive but passes
# multimodal RoPE token-type ids required by transformers Qwen3VLModel.

import torch
from mmdet.models import DETECTORS

from .spacedrive import SpaceDrive


@DETECTORS.register_module()
class SpaceDriveQwen3VL(SpaceDrive):
    """SpaceDrive detector using Qwen3-VL (`CustomQwen3VLForConditionalGeneration`)."""

    def __init__(self, **kwargs):
        super(SpaceDriveQwen3VL, self).__init__(**kwargs)
        self.lm_type = 'qwen3vl'
        lm_head = kwargs.get('lm_head')
        if lm_head is not None and getattr(self, 'lm_head', None) is not None:
            cfg = self.lm_head.base_model.model.config
            if hasattr(cfg, 'text_config') and getattr(cfg.text_config, 'hidden_size', None) is not None:
                self.llm_hidden_dim = cfg.text_config.hidden_size

    def _infer_mm_token_type_ids(self, input_ids):
        """Match HF Qwen3-VL: 0=text, 1=image token span, 2=video token span."""
        cfg = self.lm_head.base_model.model.config
        img_id = cfg.image_token_id
        vid_id = cfg.video_token_id
        mm = torch.zeros_like(input_ids, dtype=torch.int32, device=input_ids.device)
        mm[input_ids == img_id] = 1
        mm[input_ids == vid_id] = 2
        return mm

    def _extra_lm_forward_kwargs(self, input_ids):
        if input_ids is None:
            return {}
        return {'mm_token_type_ids': self._infer_mm_token_type_ids(input_ids)}
