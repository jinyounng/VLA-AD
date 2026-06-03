"""SpaceDrive + Variational Information Bottleneck (VIB)

Inserts a VIB layer between the ego-status encoders (MLP + 3-D PE) and the
VLM token stream.  The bottleneck stochastically compresses ego tokens so
that only trajectory-relevant information passes through, mitigating the
ego-status shortcut discovered in SpaceDrive+.

Architecture delta w.r.t. SpaceDrive+:
    ego_feature  (B, 1+ego_status_len, H)
         |
    [VIBBottleneck]  ← mu, logvar heads + reparameterisation
         |
    z_ego  (B, 1+ego_status_len, H)   →  VLM  (replaces raw ego tokens)

Loss:  total = trajectory_loss + vlm_loss + beta * KL[q(z|x) || N(0,I)]
"""

import torch
from mmdet.models import DETECTORS

from ...datasets.utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    POS_INDICATOR_TOKEN_INDEX,
    VISION_END_TOKEN_INDEX,
)
from .spacedrive import SpaceDrive
from projects.VIB.vib_bottleneck import VIBBottleneck


@DETECTORS.register_module()
class SpaceDriveVIB(SpaceDrive):
    """SpaceDrive+ with a Variational Information Bottleneck on ego tokens."""

    def __init__(self, vib_beta: float = 1e-3, vib_z_dim: int = None, **kwargs):
        super().__init__(**kwargs)

        assert (
            self.ego_status is not None
        ), "SpaceDriveVIB requires ego_status to be set (e.g. 'feature+PE')"

        self.vib = VIBBottleneck(
            input_dim=self.llm_hidden_dim,
            z_dim=vib_z_dim,
            beta=vib_beta,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_train_vlm(
        self,
        img_metas,
        input_ids,
        vlm_labels,
        vlm_attn_mask,
        pixel_values,
        image_grid_thw,
        coords_pos_tensor,
        **data,
    ):
        B = pixel_values.shape[0]

        # ---- 3-D PE for vision tokens (unchanged) ----
        pos_embed = None
        coords3d = None
        if self.vis_3d_pos:
            depth = self.depth_prediction(
                data["img"], data["intrinsics"], img_metas=img_metas
            )
            location = self.prepare_location(image_grid_thw, pixel_values)
            pos_embed, coords3d = self.position_embeding(
                data, location, img_metas, depth, image_grid_thw,
                False, data["img"], sample_idx=img_metas[0]["sample_idx"],
            )

        # ---- IO coords PE (unchanged) ----
        io_coords_pos = None
        gt_coords_xy = None
        has_gt_planning = None
        if self.io_3d_pos:
            io_coords = coords_pos_tensor
            z_dim = torch.zeros(io_coords.shape[0]).unsqueeze(-1).to(io_coords.device)
            gt_coords_xy = io_coords
            io_coords = torch.cat((io_coords, z_dim), dim=-1).unsqueeze(0)
            if self.input_pe_mlp:
                io_coords_pos = self.position_encoder_mlp(io_coords)
            else:
                io_coords_pos = self.position_encoder(io_coords)
                io_coords_pos = io_coords_pos.detach()
            has_gt_planning = data.get("has_gt_planning", None)

        # ---- Ego feature construction (identical to SpaceDrive+) ----
        kl_loss = torch.tensor(0.0, device=pixel_values.device)
        ego_feature = None

        if self.ego_status is not None:
            rec_can_bus = torch.cat(
                [data["command"].unsqueeze(-1), data["can_bus"]], dim=-1
            )
            ego_feature = torch.empty(
                B, 0, self.llm_hidden_dim, device=rec_can_bus.device
            )

            if "feature" in self.ego_status:
                ego_mlp_input = torch.cat([
                    self.memory_canbus.reshape(B, -1),
                    rec_can_bus.reshape(B, -1),
                    self.memory_egopose.reshape(B, -1, 16).reshape(B, -1),
                ], dim=-1)
                ego_token = self.ego_status_mlp(ego_mlp_input).unsqueeze(1)
                ego_feature = torch.cat([ego_feature, ego_token], dim=1)

                if input_ids is not None:
                    if (input_ids[0] == VISION_END_TOKEN_INDEX).sum() == 0:
                        last_vision_end_token = (
                            (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
                        )
                    else:
                        last_vision_end_token = (
                            (input_ids[0] == VISION_END_TOKEN_INDEX).nonzero().max()
                        )
                    insert_input_ids = torch.tensor(
                        [IMAGE_TOKEN_INDEX], device=input_ids.device
                    ).unsqueeze(0).repeat(B, 1)
                    insert_labels = torch.tensor(
                        [IGNORE_INDEX], device=vlm_labels.device
                    ).unsqueeze(0).repeat(B, 1)
                    insert_attn_mask = torch.tensor(
                        [1], device=vlm_attn_mask.device
                    ).unsqueeze(0).repeat(B, 1)
                    input_ids = torch.cat([
                        input_ids[:, :last_vision_end_token + 1],
                        insert_input_ids,
                        input_ids[:, last_vision_end_token + 1:],
                    ], dim=-1)
                    vlm_labels = torch.cat([
                        vlm_labels[:, :last_vision_end_token + 1],
                        insert_labels,
                        vlm_labels[:, last_vision_end_token + 1:],
                    ], dim=-1)
                    vlm_attn_mask = torch.cat([
                        vlm_attn_mask[:, :last_vision_end_token + 1],
                        insert_attn_mask,
                        vlm_attn_mask[:, last_vision_end_token + 1:],
                    ], dim=-1)

            if "PE" in self.ego_status:
                past_xyz = self.memory_egopose[:, : self.ego_status_len, :3, 3]
                encoded_past_xyz = self.position_encoder(
                    past_xyz.reshape(B, -1, 3)
                ).reshape(B, self.ego_status_len, -1)
                ego_feature = torch.cat([ego_feature, encoded_past_xyz], dim=1)

                if input_ids is not None:
                    if (input_ids[0] == VISION_END_TOKEN_INDEX).sum() == 0:
                        last_vision_end_token = (
                            (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
                        )
                    else:
                        last_vision_end_token = (
                            (input_ids[0] == VISION_END_TOKEN_INDEX).nonzero().max()
                        )
                    len_past_pos = encoded_past_xyz.shape[1]
                    insert_input_ids = torch.tensor(
                        [POS_INDICATOR_TOKEN_INDEX, IMAGE_TOKEN_INDEX] * len_past_pos,
                        device=input_ids.device,
                    ).unsqueeze(0).repeat(B, 1)
                    insert_labels = torch.tensor(
                        [IGNORE_INDEX] * len_past_pos * 2,
                        device=vlm_labels.device,
                    ).unsqueeze(0).repeat(B, 1)
                    insert_attn_mask = torch.tensor(
                        [1] * len_past_pos * 2,
                        device=vlm_attn_mask.device,
                    ).unsqueeze(0).repeat(B, 1)
                    input_ids = torch.cat([
                        input_ids[:, :last_vision_end_token + 1],
                        insert_input_ids,
                        input_ids[:, last_vision_end_token + 1:],
                    ], dim=-1)
                    vlm_labels = torch.cat([
                        vlm_labels[:, :last_vision_end_token + 1],
                        insert_labels,
                        vlm_labels[:, last_vision_end_token + 1:],
                    ], dim=-1)
                    vlm_attn_mask = torch.cat([
                        vlm_attn_mask[:, :last_vision_end_token + 1],
                        insert_attn_mask,
                        vlm_attn_mask[:, last_vision_end_token + 1:],
                    ], dim=-1)

            # ===== VIB: stochastically compress ego tokens =====
            ego_feature, kl_loss = self.vib(ego_feature)

        # ---- VLM forward (unchanged) ----
        losses = dict()

        if self.with_lm_head:
            lm_loss = self.lm_head(
                input_ids=input_ids,
                attention_mask=vlm_attn_mask,
                labels=vlm_labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pos_emb=pos_embed,
                io_coords_pos=io_coords_pos,
                loss_pos_lambda=self.loss_pos_lambda if self.io_3d_pos else None,
                loss_for_pos=self.pe_decode_method,
                include_semantic_posemb=(
                    self.include_semantic_posemb if self.io_3d_pos else False
                ),
                supervise_semantic_posemb=(
                    self.supervise_semantic_posemb if self.io_3d_pos else False
                ),
                planning_only=self.planning_only if self.io_3d_pos else False,
                single_coords_only=(
                    self.single_coords_only if self.io_3d_pos else False
                ),
                has_gt_planning=has_gt_planning if self.io_3d_pos else None,
                gt_coords_xy=gt_coords_xy if self.io_3d_pos else None,
                ego_feature=(
                    ego_feature
                    if self.ego_status
                    and ego_feature is not None
                    and ego_feature.numel() > 0
                    else None
                ),
                enable_pe_input=(
                    self.enable_pe_input if self.io_3d_pos else False
                ),
                pos_index=coords3d if self.use_rope else None,
                **self._extra_lm_forward_kwargs(input_ids),
            )

            losses.update(vlm_loss=lm_loss["loss"])

            # ---- Trajectory loss computation (unchanged) ----
            if self.io_3d_pos:
                if self.pe_decode_method in [
                    "l2_coords_mlp", "l2_coords_mlp_2layer",
                ] and self.io_3d_pos:
                    if len(lm_loss["output_pos"].shape) == 2:
                        lm_loss["output_pos"] = lm_loss["output_pos"].unsqueeze(0)
                    if not has_gt_planning.any():
                        sampled_coords = (
                            torch.zeros(B, 1, 3).to(lm_loss["output_pos"].device)
                            * (
                                torch.tensor(self.pc_range[3:6])
                                - torch.tensor(self.pc_range[0:3])
                            ).to(lm_loss["output_pos"].device)
                            + torch.tensor(self.pc_range[0:3]).to(
                                lm_loss["output_pos"].device
                            )
                        )
                        if self.input_pe_mlp:
                            sampled_coords_pos = self.position_encoder_mlp(
                                sampled_coords
                            )
                        else:
                            sampled_coords_pos = self.position_encoder(sampled_coords)
                        if self.use_vae_to_replace_mlp:
                            decoded_output_pos, loss_vae_gen = self.vae_output_coords(
                                sampled_coords_pos, sampled_coords[:, :, :2]
                            )
                            losses["loss_vae_gen"] = loss_vae_gen * 0
                            lm_loss["loss_pos"] = (
                                self.loss_pos_func(
                                    decoded_output_pos[:, :, :2],
                                    sampled_coords[:, :, :2],
                                )
                                * 0
                            )
                        else:
                            decoded_output_pos = self.mlp_output_coords(
                                sampled_coords_pos
                            ).reshape(B, -1, 3)[:, :, :2]
                            lm_loss["loss_pos"] = (
                                self.loss_pos_func(
                                    decoded_output_pos, sampled_coords[:, :, :2]
                                )
                                * 0
                            )
                    else:
                        if self.use_vae_to_replace_mlp:
                            decoded_output_pos, loss_vae_gen = self.vae_output_coords(
                                lm_loss["output_pos"], lm_loss["gt_coords_xy"]
                            )
                            losses["loss_vae_gen"] = loss_vae_gen
                        else:
                            decoded_output_pos = self.mlp_output_coords(
                                lm_loss["output_pos"]
                            )
                        if len(decoded_output_pos.shape) == 4:
                            decoded_output_pos = decoded_output_pos.squeeze(1)
                        decoded_output_pos = decoded_output_pos[:, :, :2]
                        lm_loss["loss_pos"] = (
                            self.loss_pos_func(
                                decoded_output_pos, lm_loss["gt_coords_xy"]
                            )
                            * self.loss_pos_lambda
                        )
                elif self.pe_decode_method in ["cosine", "l2"]:
                    pass
                else:
                    raise NotImplementedError(
                        f"pe_decode_method {self.pe_decode_method} not implemented"
                    )
                losses.update(loss_pos=lm_loss["loss_pos"])

        if self.learnable_pe_scaling:
            losses.update(pe_scaling=self.pe_scaling)

        # ===== VIB KL divergence loss =====
        losses["loss_vib_kl"] = kl_loss

        return losses

    # ------------------------------------------------------------------
    # Inference — intercept ego_feature before it enters VLM generation
    # ------------------------------------------------------------------

    def test_generation_pts(
        self, img, img_metas, input_ids, pixel_values, image_grid_thw,
        attention_mask, **data,
    ):
        vib = self.vib
        orig_generate = self.lm_head.generate

        def _vib_generate(*args, **kwargs):
            ef = kwargs.get("ego_feature")
            if ef is not None:
                kwargs = dict(kwargs)
                kwargs["ego_feature"], _ = vib(ef)
            return orig_generate(*args, **kwargs)

        self.lm_head.generate = _vib_generate
        try:
            return super().test_generation_pts(
                img, img_metas, input_ids, pixel_values,
                image_grid_thw, attention_mask, **data,
            )
        finally:
            self.lm_head.generate = orig_generate
