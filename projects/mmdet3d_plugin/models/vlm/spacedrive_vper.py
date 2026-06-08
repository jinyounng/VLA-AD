"""SpaceDrive + VPER v3

Architecture:
    VLM(vision + command, ego 제외) → PE Decoder → base_wp
                                    → Gate Head  → gate
    Raw ego features               → Delta MLP  → raw_delta

    final_wp = base_wp + gate * max_delta * tanh(raw_delta)

Ego features (e_ego, φ(c^ego_τ)) are computed with the same encoders as
SpaceDrive+ but are **not** injected into the VLM token stream.  Instead they
feed the lightweight Delta MLP whose output is gated by the VLM hidden state.
"""

import json

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS

from ...datasets.utils.constants import (
    POS_EMBEDDING_TOKEN,
    POS_EMBEDDING_TOKEN_INDEX,
    POS_INDICATOR_TOKEN_INDEX,
)
from .spacedrive import SpaceDrive
from .vper_modules import DeltaMLP, GateHead


@DETECTORS.register_module()
class SpaceDriveVPER(SpaceDrive):
    """SpaceDrive with VPER v3 ego-gated delta waypoint refinement."""

    def __init__(
        self,
        max_delta_x=2.0,
        max_delta_y=1.0,
        delta_mlp_hidden_dim=256,
        vper_base_loss_weight=0.3,
        num_waypoints=6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            self.ego_status is not None
            and "feature" in self.ego_status
            and "PE" in self.ego_status
        ), "SpaceDriveVPER requires ego_status='feature+PE'"

        # ---- VPER-specific modules ----
        self.gate_head = GateHead(self.llm_hidden_dim)

        # D_ego = llm_hidden_dim (e_ego) + ego_status_len * llm_hidden_dim (φ_ego flat)
        ego_dim = self.llm_hidden_dim * (1 + self.ego_status_len)
        self.delta_mlp = DeltaMLP(
            ego_dim=ego_dim,
            hidden_dim=delta_mlp_hidden_dim,
            num_waypoints=num_waypoints,
        )

        self.register_buffer(
            "vper_max_delta", torch.tensor([max_delta_x, max_delta_y])
        )
        self.vper_base_loss_weight = vper_base_loss_weight

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_vper_ego_features(self, data, B):
        """Compute ego features for the Delta MLP (NOT injected into the VLM).

        Returns:
            ego_features: (B, D_ego) where D_ego = llm_hidden_dim * (1 + ego_status_len)
        """
        rec_can_bus = torch.cat(
            [data["command"].unsqueeze(-1), data["can_bus"]], dim=-1
        )

        # e_ego – MLP over canbus + ego-pose history
        ego_mlp_input = torch.cat(
            [
                self.memory_canbus.reshape(B, -1),
                rec_can_bus.reshape(B, -1),
                self.memory_egopose.reshape(B, -1, 16).reshape(B, -1),
            ],
            dim=-1,
        )
        e_ego = self.ego_status_mlp(ego_mlp_input)  # (B, llm_hidden_dim)

        # φ(c^ego_τ) – 3-D positional encoding of past ego positions
        past_xyz = self.memory_egopose[:, : self.ego_status_len, :3, 3]
        phi_ego = self.position_encoder(past_xyz.reshape(B, -1, 3)).reshape(
            B, self.ego_status_len, -1
        )  # (B, ego_status_len, llm_hidden_dim)

        return torch.cat([e_ego, phi_ego.flatten(1)], dim=-1)  # (B, D_ego)

    def _vper_decode_waypoints(self, output_pos, ego_features):
        """Apply VPER composition: final_wp = base_wp + gate * Δ_max * tanh(raw_δ).

        Args:
            output_pos:   (B, N, hidden_dim)  VLM hidden states at waypoint slots.
            ego_features: (B, D_ego)          Concatenated ego features.

        Returns:
            final_wp, base_wp, gate, delta – each (B, N, 2).
        """
        base_wp = self.mlp_output_coords(output_pos)[:, :, :2]  # (B, N, 2)
        gate = self.gate_head(output_pos)  # (B, N, 2)

        num_wp = output_pos.shape[1]
        raw_deltas = [self.delta_mlp(ego_features, j) for j in range(num_wp)]
        raw_delta = torch.stack(raw_deltas, dim=1)  # (B, N, 2)

        delta = gate * self.vper_max_delta * torch.tanh(raw_delta)
        final_wp = base_wp + delta

        return final_wp, base_wp, gate, delta

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
                data,
                location,
                img_metas,
                depth,
                image_grid_thw,
                False,
                data["img"],
                sample_idx=img_metas[0]["sample_idx"],
            )

        # ---- IO coords PE (unchanged) ----
        io_coords_pos = None
        gt_coords_xy = None
        has_gt_planning = None
        if self.io_3d_pos:
            io_coords = coords_pos_tensor  # (num_coords, 2)
            z_dim = torch.zeros(io_coords.shape[0]).unsqueeze(-1).to(io_coords.device)
            gt_coords_xy = io_coords
            io_coords = torch.cat((io_coords, z_dim), dim=-1).unsqueeze(0)

            if self.input_pe_mlp:
                io_coords_pos = self.position_encoder_mlp(io_coords)
            else:
                io_coords_pos = self.position_encoder(io_coords)
                io_coords_pos = io_coords_pos.detach()

            has_gt_planning = data.get("has_gt_planning", None)

        # ---- VPER: compute ego features, do NOT inject into VLM ----
        ego_features = self._compute_vper_ego_features(data, B)

        # ---- VLM forward (ego_feature=None → ego bypasses VLM) ----
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
                single_coords_only=self.single_coords_only if self.io_3d_pos else False,
                has_gt_planning=has_gt_planning if self.io_3d_pos else None,
                gt_coords_xy=gt_coords_xy if self.io_3d_pos else None,
                ego_feature=None,
                enable_pe_input=self.enable_pe_input if self.io_3d_pos else False,
                pos_index=coords3d if self.use_rope else None,
                **self._extra_lm_forward_kwargs(input_ids),
            )

            losses.update(vlm_loss=lm_loss["loss"])

            # ---- VPER waypoint loss ----
            if self.io_3d_pos and self.pe_decode_method in [
                "l2_coords_mlp",
                "l2_coords_mlp_2layer",
            ]:
                if len(lm_loss["output_pos"].shape) == 2:
                    lm_loss["output_pos"] = lm_loss["output_pos"].unsqueeze(0)

                if not has_gt_planning.any():
                    # No GT planning available → produce a zero-gradient placeholder loss.
                    device = lm_loss["output_pos"].device
                    pc_lo = torch.tensor(self.pc_range[0:3], device=device)
                    pc_hi = torch.tensor(self.pc_range[3:6], device=device)
                    sampled_coords = (
                        torch.zeros(B, 1, 3, device=device) * (pc_hi - pc_lo) + pc_lo
                    )
                    if self.input_pe_mlp:
                        sampled_pos = self.position_encoder_mlp(sampled_coords)
                    else:
                        sampled_pos = self.position_encoder(sampled_coords)
                    decoded = self.mlp_output_coords(sampled_pos).reshape(B, -1, 3)[
                        :, :, :2
                    ]
                    # Keep all VPER params in the graph so DDP doesn't complain.
                    # ego_status_mlp is also disconnected because ego bypasses VLM.
                    dummy_vper = (
                        sum(p.sum() * 0 for p in self.gate_head.parameters())
                        + sum(p.sum() * 0 for p in self.delta_mlp.parameters())
                        + sum(p.sum() * 0 for p in self.ego_status_mlp.parameters())
                    )
                    lm_loss["loss_pos"] = (
                        self.loss_pos_func(decoded, sampled_coords[:, :, :2]) * 0
                        + dummy_vper
                    )
                    losses["gate_mean"] = torch.tensor(0.0, device=device)
                    losses["delta_norm"] = torch.tensor(0.0, device=device)
                    losses["base_l2"] = torch.tensor(0.0, device=device)
                    losses["huber_base"] = torch.tensor(0.0, device=device)
                    losses["huber_final"] = torch.tensor(0.0, device=device)
                else:
                    output_pos = lm_loss["output_pos"]
                    gt_xy = lm_loss["gt_coords_xy"]

                    final_wp, base_wp, gate, delta = self._vper_decode_waypoints(
                        output_pos, ego_features
                    )

                    loss_final = self.loss_pos_func(final_wp, gt_xy)
                    loss_base = self.loss_pos_func(base_wp, gt_xy)
                    lm_loss["loss_pos"] = (
                        loss_final + self.vper_base_loss_weight * loss_base
                    ) * self.loss_pos_lambda

                    losses["gate_mean"] = gate.mean().detach()
                    losses["delta_norm"] = delta.norm(dim=-1).mean().detach()
                    losses["base_l2"] = F.mse_loss(
                        base_wp.detach(), gt_xy.detach()
                    )
                    # Huber values logged separately for comparison with the
                    # original SpaceDrive+ loss_pos (== huber_base). Keys must NOT
                    # contain "loss" or mmdet would sum them into the total loss.
                    losses["huber_base"] = loss_base.detach()
                    losses["huber_final"] = loss_final.detach()

                losses.update(loss_pos=lm_loss["loss_pos"])

            elif self.io_3d_pos and self.pe_decode_method in ["cosine", "l2"]:
                losses.update(loss_pos=lm_loss["loss_pos"])

        if self.learnable_pe_scaling:
            losses.update(pe_scaling=self.pe_scaling)

        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def test_generation_pts(
        self, img, img_metas, input_ids, pixel_values, image_grid_thw,
        attention_mask, **data,
    ):
        """VPER inference: generate without ego tokens, apply VPER decode."""
        if "question_text" in data:
            question_text = data["question_text"]
        else:
            question_text = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            if "coords_pos_tensor" in data:
                formatted_points = ", ".join(
                    f"({self.format_number(point[0], 2)}, "
                    f"{self.format_number(point[1], 2)})"
                    for point in data["coords_pos_tensor"][0].cpu().numpy()
                )
                question_text = question_text.replace(
                    ", , , , , ", str(formatted_points)
                )

        B = pixel_values.shape[0]

        # ---- 3-D PE for vision ----
        pos_embed = None
        coords3d = None
        if self.vis_3d_pos:
            depth = self.depth_prediction(
                img, data["intrinsics"], img_metas=img_metas
            )
            location = self.prepare_location(image_grid_thw, pixel_values)
            pos_embed, coords3d = self.position_embeding(
                data, location, img_metas, depth, image_grid_thw
            )

        # ---- IO coords PE ----
        io_coords_pos = None
        if self.io_3d_pos and "coords_pos_tensor" in data:
            io_coords = data["coords_pos_tensor"]
            z_dim = (
                torch.zeros(io_coords.shape[1])
                .unsqueeze(-1)
                .repeat(B, 1, 1)
                .to(io_coords.device)
            )
            io_coords = torch.cat((io_coords, z_dim), dim=-1)
            io_coords_pos = self.position_encoder(io_coords).detach()

        # ---- VPER ego features (not injected into VLM) ----
        ego_features = self._compute_vper_ego_features(data, B)

        # ---- Autoregressive generation (ego_feature=None) ----
        generated_text = []
        if self.with_lm_head:
            mmcv.mkdir_or_exist(self.save_path)

            for i, input_ids_i in enumerate(input_ids):
                input_ids_i = input_ids_i.unsqueeze(0)
                len_input_ids = input_ids_i.shape[1]

                generate_kwargs = dict(
                    input_ids=input_ids_i,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    pos_emb=pos_embed,
                    loss_pos_lambda=(
                        self.loss_pos_lambda if self.io_3d_pos else None
                    ),
                    include_semantic_posemb=(
                        self.include_semantic_posemb if self.io_3d_pos else False
                    ),
                    planning_only=(
                        self.planning_only if self.io_3d_pos else False
                    ),
                    single_coords_only=(
                        self.single_coords_only if self.io_3d_pos else False
                    ),
                    ego_feature=None,
                    enable_pe_input=False,
                    pos_index=coords3d if self.use_rope else None,
                    coords_encoder=(
                        self.position_encoder_mlp
                        if (
                            not self.single_token_output
                            and self.pe_decode_method is not None
                            and "mlp" in self.pe_decode_method
                            and self.input_pe_mlp
                        )
                        else self.position_encoder
                    ),
                    coords_decoder=(
                        self.mlp_output_coords
                        if (
                            not self.single_token_output
                            and self.pe_decode_method is not None
                            and "mlp" in self.pe_decode_method
                            and not self.use_vae_to_replace_mlp
                        )
                        else None
                    ),
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    max_new_tokens=100,
                    use_cache=True,
                )

                if self.lm_type == "qwenvl25":
                    generate_kwargs.update(
                        self._extra_lm_forward_kwargs(input_ids_i)
                    )

                outputs = self.lm_head.generate(**generate_kwargs)

                output_ids = outputs["sequences"][0][len_input_ids:].unsqueeze(0)

                if self.io_3d_pos:
                    last_hidden_state = (
                        outputs["hidden_states_for_output"]
                        .reshape(1, -1, self.llm_hidden_dim)[0][len_input_ids - 1 :]
                        .unsqueeze(0)
                    )

                    # Locate POS_INDICATOR positions
                    pos_indicator_mask = output_ids[0] == POS_INDICATOR_TOKEN_INDEX

                    if pos_indicator_mask.sum() > 1:
                        shifted = torch.roll(pos_indicator_mask, shifts=1, dims=-1)
                        pos_indicator_mask = pos_indicator_mask & ~shifted

                    if pos_indicator_mask.sum() == 0:
                        generated_text.append(
                            dict(
                                Q=question_text,
                                A=self.tokenizer.batch_decode(
                                    output_ids, skip_special_tokens=False
                                ),
                            )
                        )
                        continue

                    pos_embedding_mask = torch.roll(
                        pos_indicator_mask, shifts=1, dims=-1
                    )
                    pos_embedding_mask[0] = False
                    pos_embedding_index = pos_embedding_mask.nonzero()

                    pos_embedding = last_hidden_state[
                        0, pos_embedding_index, :
                    ].reshape(B, -1, self.llm_hidden_dim)

                    output_ids[0, pos_embedding_index] = POS_EMBEDDING_TOKEN_INDEX
                    output_text = self.tokenizer.batch_decode(
                        output_ids, skip_special_tokens=False
                    )[0]

                    # ---- VPER decode ----
                    if "mlp" in self.pe_decode_method:
                        with torch.no_grad():
                            pos_emb_f32 = pos_embedding.to(torch.float32)
                            ego_f32 = ego_features.to(torch.float32)
                            final_wp, _, _, _ = self._vper_decode_waypoints(
                                pos_emb_f32, ego_f32
                            )
                            interpolated_pos = final_wp[0]  # (N, 2)
                    else:
                        with torch.no_grad():
                            _, interpolated_pos = self.position_encoder.decode_pos(
                                pos_embedding,
                                self.pos_emb_grid,
                                self.pc_range,
                                self.voxel_size,
                                sim_method=self.pe_decode_method,
                            )
                            interpolated_pos = interpolated_pos.squeeze(0)

                    for n in range(interpolated_pos.shape[0]):
                        x = float(interpolated_pos[n][0])
                        y = float(interpolated_pos[n][1])
                        coord_str = f"({x:.2f}, {y:.2f})"
                        output_text = output_text.replace(
                            POS_EMBEDDING_TOKEN, coord_str, 1
                        )

                    print(f"generated text: {output_text}")
                    generated_text.append(
                        dict(Q=question_text, A=output_text)
                    )
                else:
                    generated_text.append(
                        dict(
                            Q=question_text,
                            A=self.tokenizer.batch_decode(
                                output_ids, skip_special_tokens=True
                            ),
                        )
                    )

            with open(
                self.save_path + img_metas[0]["sample_idx"], "w"
            ) as file:
                json.dump(generated_text, file)

        return generated_text
