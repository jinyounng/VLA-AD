"""SpaceDrive + Vision-Conditioned Variational Information Bottleneck (VC-VIB)

Extends :class:`SpaceDriveVIB`'s idea, but replaces the fixed ``N(0, I)`` prior
with a *vision-conditioned* prior ``p(z|vision)``.  A small PriorNet maps a pure
(ego-free) pooled vision summary to ``(mu_p, sigma_p)`` and the ego tokens are
compressed toward this prior:

    ego_feature  (B, 1+ego_status_len, H)
         |                                  vision  →  [vision encoder]  →  summary
    [VisionConditionedVIB]  ← KL(q(z|ego) || p(z|vision))      (PriorNet)
         |
    z_ego  (B, 1+ego_status_len, H)   →  VLM  (replaces raw ego tokens)

Effect: information already present in vision (direction, road layout...) is
KL-free yet useless for reconstruction → dropped from ``z``; complementary ego
information (precise speed/acceleration...) incurs KL but helps planning → kept.

Loss:  total = trajectory_loss + vlm_loss + beta * KL[q(z|ego) || p(z|vision)]

Note: this model leaves the base ``SpaceDriveVIB`` / ``VIBBottleneck`` code
untouched.  The pooled vision summary is obtained by running the (shared) Qwen
vision encoder via ``get_image_features``; gradients are *not* detached so the
PriorNet trains the vision encoder as intended.
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
from projects.VIB.vision_conditioned_vib import VisionConditionedVIB


@DETECTORS.register_module()
class SpaceDriveVCVIB(SpaceDrive):
    """SpaceDrive+ with a Vision-Conditioned Information Bottleneck on ego tokens."""

    def __init__(
        self,
        vib_beta: float = 1e-3,
        vib_z_dim: int = None,
        vib_prior_hidden_dim: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            self.ego_status is not None
        ), "SpaceDriveVCVIB requires ego_status to be set (e.g. 'feature+PE')"

        # vision_dim == llm_hidden_dim: `get_image_features` already projects
        # vision tokens into the LLM hidden space.
        self.vib = VisionConditionedVIB(
            input_dim=self.llm_hidden_dim,
            vision_dim=self.llm_hidden_dim,
            z_dim=vib_z_dim,
            beta=vib_beta,
            prior_hidden_dim=vib_prior_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Vision summary helpers
    # ------------------------------------------------------------------

    def _unwrap_vlm_model(self):
        """Return the inner ``CustomQwen2_5_VLModel`` (handles the LoRA wrapper)."""
        m = self.lm_head
        # peft wrapping: PeftModel.base_model.model -> *ForConditionalGeneration
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        # *ForConditionalGeneration.model -> *VLModel (has get_image_features)
        return m.model

    def _build_vision_summary(
        self, pixel_values, image_grid_thw, B, pos_embed=None, pos_index=None
    ):
        """Pool the pure vision features into a per-sample ``(B, H)`` summary.

        The features come from the same vision encoder the VLM uses, but are
        pooled *before* ever mixing with ego/text tokens, so they represent
        ego-free vision context.  The returned image embeddings are the same
        tensors passed into the VLM, including visual PE/RoPE when enabled.
        Gradients flow (no detach).
        """
        vlm = self._unwrap_vlm_model()

        pv = pixel_values.reshape(-1, pixel_values.shape[-1])
        grid = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])

        image_embeds = vlm.get_image_features(pv, grid)  # (N_total, H)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(list(image_embeds), dim=0)

        if pos_index is not None:
            image_embeds = vlm.rope_3d(image_embeds.unsqueeze(0), pos_index).squeeze(0)
        elif pos_embed is not None:
            pos_embed = pos_embed.reshape(image_embeds.shape)
            if pos_embed.shape[0] != image_embeds.shape[0]:
                raise ValueError(
                    f"Positional embedding shape {pos_embed.shape} does not "
                    f"match image embedding shape {image_embeds.shape}"
                )
            image_embeds = image_embeds + pos_embed

        n_total = image_embeds.shape[0]
        # All views share the same resolution → tokens split evenly across the
        # batch.  (Training/eval here use samples_per_gpu=1, i.e. B=1.)
        if n_total % B == 0:
            vision_summary = image_embeds.reshape(
                B, n_total // B, image_embeds.shape[-1]
            ).mean(dim=1)
        else:
            vision_summary = image_embeds.mean(dim=0, keepdim=True).expand(
                B, image_embeds.shape[-1]
            )
        return vision_summary, image_embeds

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

        # ---- Pure vision summary for the VC-VIB prior ----
        # Built before ego/text mixing so the PriorNet only sees vision.
        vision_summary, image_embeds = self._build_vision_summary(
            pixel_values, image_grid_thw, B, pos_embed=pos_embed,
            pos_index=coords3d if self.use_rope else None,
        )

        # ---- Ego feature construction (identical to SpaceDrive+) ----
        kl_loss = torch.tensor(0.0, device=pixel_values.device)
        raw_kl = torch.tensor(0.0, device=pixel_values.device)
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

            # ===== VC-VIB: compress ego tokens toward p(z|vision) =====
            ego_feature, kl_loss, raw_kl = self.vib(ego_feature, vision_summary)

        # ---- VLM forward (unchanged) ----
        losses = dict()

        if self.with_lm_head:
            lm_loss = self.lm_head(
                input_ids=input_ids,
                attention_mask=vlm_attn_mask,
                labels=vlm_labels,
                pixel_values=None,
                image_grid_thw=image_grid_thw,
                precomputed_image_embeds=image_embeds,
                pos_emb=None,
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
                pos_index=None,
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

        # ===== VC-VIB KL divergence loss (already beta-scaled) =====
        losses["loss_vib_kl"] = kl_loss
        losses["vib_raw_kl"] = raw_kl.detach()

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
                bsz = ef.shape[0]
                vs, image_embeds = self._build_vision_summary(
                    kwargs["pixel_values"],
                    kwargs["image_grid_thw"],
                    bsz,
                    pos_embed=kwargs.get("pos_emb"),
                    pos_index=kwargs.get("pos_index"),
                )
                kwargs["precomputed_image_embeds"] = image_embeds
                kwargs["pixel_values"] = None
                kwargs["pos_emb"] = None
                kwargs["pos_index"] = None
                if vs.shape[0] != ef.shape[0]:
                    if vs.shape[0] == 1:
                        vs = vs.expand(ef.shape[0], -1)
                    else:
                        vs = vs[: ef.shape[0]]
                kwargs["ego_feature"], _, _ = vib(ef, vs)
            return orig_generate(*args, **kwargs)

        self.lm_head.generate = _vib_generate
        try:
            return super().test_generation_pts(
                img, img_metas, input_ids, pixel_values,
                image_grid_thw, attention_mask, **data,
            )
        finally:
            self.lm_head.generate = orig_generate
