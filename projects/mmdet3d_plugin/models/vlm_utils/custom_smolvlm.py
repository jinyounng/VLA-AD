# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Hugging Face Transformers (https://github.com/huggingface/transformers)
# Copyright (c) The Hugging Face team. All rights reserved.
# ------------------------------------------------------------------------

from transformers import Idefics3ForConditionalGeneration
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3Model,
    Idefics3CausalLMOutputWithPast,
    Idefics3BaseModelOutputWithPast,
)
import torch

from transformers.utils import auto_docstring, can_return_tuple

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from ...datasets.utils.constants import (
    IGNORE_INDEX,
    SMOLVLM_POS_INDICATOR_TOKEN_INDEX as POS_INDICATOR_TOKEN_INDEX,
    SMOLVLM_POS_EMBEDDING_TOKEN_INDEX as POS_EMBEDDING_TOKEN_INDEX,
)

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]

import os


@dataclass
class CustomGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    hidden_states_for_output: Optional[torch.FloatTensor] = None


@dataclass
class CustomIdefics3CausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
    loss_pos: Optional[torch.FloatTensor] = None
    output_pos: Optional[torch.FloatTensor] = None
    gt_pos: Optional[torch.FloatTensor] = None
    output_pe_mask: Optional[torch.BoolTensor] = None
    gt_coords_xy: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state_original: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None


class CustomIdefics3ForConditionalGeneration(Idefics3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomIdefics3Model(config)
        self.l2_loss = nn.MSELoss()
        self.llm_hidden_dim = config.text_config.hidden_size
        self.position_encoder = None
        self.pc_range = None
        self.voxel_size = None
        self.pos_emb_grid = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        # SpaceDrive-specific args
        pos_emb: Optional[torch.Tensor] = None,
        io_coords_pos: Optional[torch.Tensor] = None,
        loss_pos_lambda: Optional[torch.FloatTensor] = None,
        loss_for_pos: str = 'l2',
        include_semantic_posemb=False,
        supervise_semantic_posemb=False,
        planning_only: Optional[bool] = False,
        single_coords_only: Optional[bool] = False,
        has_gt_planning=torch.zeros(1),
        gt_coords_xy: Optional[torch.Tensor] = None,
        coords_encoder=None,
        coords_decoder=None,
        ego_feature: Optional[torch.Tensor] = None,
        enable_pe_input=False,
        pos_index: Optional[torch.Tensor] = None,
        # Qwen-specific args that we accept but ignore
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CustomIdefics3CausalLMOutputWithPast]:

        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        io_coords_pos_mask = None
        weighted_mask = None
        if io_coords_pos is not None:
            weighted_tokens = [POS_INDICATOR_TOKEN_INDEX]
            weighted_mask = torch.ones(self.config.text_config.vocab_size)
            weighted_mask[weighted_tokens] = 3.0
            weighted_mask = weighted_mask.float()

            io_coords_pos = io_coords_pos.to(dtype=self.model.dtype, device=self.model.device)
            io_coords_pos_mask = (input_ids == POS_EMBEDDING_TOKEN_INDEX).unsqueeze(-1)

            io_coords_pos_indices = io_coords_pos_mask.nonzero(as_tuple=False)
            scattered_io_coords_pos = torch.full(
                (input_ids.shape[0], input_ids.shape[1], io_coords_pos.shape[-1]),
                0, dtype=io_coords_pos.dtype, device=io_coords_pos.device,
            )
            if io_coords_pos_indices.shape[0] == io_coords_pos.shape[1]:
                scattered_io_coords_pos[io_coords_pos_indices[:, 0], io_coords_pos_indices[:, 1]] = io_coords_pos.reshape(-1, io_coords_pos.shape[-1])
                io_coords_pos = scattered_io_coords_pos.reshape(B, -1, io_coords_pos.shape[-1])
            else:
                scattered_io_coords_pos[io_coords_pos_indices[:, 0], io_coords_pos_indices[:, 1]] = io_coords_pos[:, :io_coords_pos_indices.shape[0]].reshape(-1, io_coords_pos.shape[-1])
                io_coords_pos = scattered_io_coords_pos.reshape(B, -1, io_coords_pos.shape[-1])

        # Some collators provide pixel values with an extra singleton axis: (B, 1, N, C, H, W).
        if pixel_values is not None and pixel_values.dim() == 6 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            pos_emb=pos_emb,
            io_coords_pos_mask=io_coords_pos_mask,
            enable_pe_input=enable_pe_input,
            io_coords_pos=io_coords_pos,
            include_semantic_posemb=include_semantic_posemb,
            ego_feature=ego_feature,
        )

        hidden_states = outputs[0]
        hidden_states_original = hidden_states
        if hidden_states.dtype != self.lm_head.weight.dtype:
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)

        logits = self.lm_head(hidden_states)
        if hasattr(self, 'new_lm_head'):
            if self.new_lm_head.weight.dtype != logits.dtype:
                hidden_states = hidden_states.to(self.new_lm_head.weight.dtype)
            new_logits = self.new_lm_head(hidden_states)
            new_logits = new_logits.to(logits.dtype)
            pos_indicator_id = POS_INDICATOR_TOKEN_INDEX
            pos_embedding_id = POS_EMBEDDING_TOKEN_INDEX
            logits = logits.index_copy(2, torch.arange(pos_indicator_id, pos_embedding_id + 1, device=logits.device), new_logits)

        if io_coords_pos_mask is not None:
            output_mask = (labels != IGNORE_INDEX).unsqueeze(-1).expand_as(hidden_states)
            output_pe_mask = torch.roll(io_coords_pos_mask, shifts=-1, dims=-2)
            output_mask_hidden = torch.roll(output_mask, shifts=-1, dims=-2)
            output_pe_mask = output_pe_mask & output_mask_hidden
            output_pos = hidden_states.masked_select(output_pe_mask).view(-1, hidden_states.shape[-1])
            if include_semantic_posemb:
                semantic_pos_emb = self.model.get_input_embeddings()(torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=io_coords_pos.device).unsqueeze(0)).squeeze(0)
                output_pos = output_pos - semantic_pos_emb

            gt_pos = io_coords_pos.masked_select(io_coords_pos_mask & output_mask).view(-1, io_coords_pos.shape[-1])

            if gt_coords_xy is not None:
                output_xy_mask = output_mask.masked_select(io_coords_pos_mask).view(-1, hidden_states.shape[-1])
                output_xy_mask = output_xy_mask[:, :2]
                gt_coords_xy = gt_coords_xy.masked_select(output_xy_mask).view(-1, gt_coords_xy.shape[-1]).to(gt_pos.dtype)

        loss = None
        if labels is not None:
            if io_coords_pos_mask is not None:
                if not (supervise_semantic_posemb and include_semantic_posemb):
                    labels = labels.masked_fill(io_coords_pos_mask.squeeze(-1), IGNORE_INDEX)

            if planning_only and single_coords_only:
                raise ValueError("planning_only and single_coords_only cannot be both True at the same time.")

            if not has_gt_planning.any():
                output_pos = torch.empty((0, self.llm_hidden_dim)).to(labels.device)
                gt_pos = torch.empty((0, self.llm_hidden_dim)).to(labels.device)
                gt_coords_xy = torch.empty((0, 2)).to(labels.device)
            else:
                num_valid = has_gt_planning.sum()
                if planning_only:
                    if io_coords_pos is not None:
                        output_pos = output_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :6, :]
                        gt_pos = gt_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :6, :]
                        gt_coords_xy = gt_coords_xy.reshape(num_valid, -1, 2)[:, :6, :]
                elif single_coords_only:
                    if io_coords_pos is not None:
                        output_pos = output_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :1, :].unsqueeze(1)
                        gt_pos = gt_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :1, :].unsqueeze(1)
                        gt_coords_xy = gt_coords_xy.reshape(num_valid, -1, 2)[:, :6, :]

            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, weight=weighted_mask, num_items_in_batch=torch.tensor(B))

            if io_coords_pos_mask is not None:
                if loss_for_pos in ('l2_coords_mlp', 'l2_coords_mlp_2layer'):
                    output_pos = output_pos.to(torch.float32)
                    gt_coords_xy = gt_coords_xy.to(torch.float32)
                    loss_pos = torch.tensor(0.0, device=output_pos.device, dtype=output_pos.dtype)
                elif loss_for_pos == 'cosine':
                    loss_pos = 1 - torch.cosine_similarity(output_pos, gt_pos, dim=-1).mean()
                elif loss_for_pos == 'l2':
                    loss_pos = self.l2_loss(output_pos, gt_pos)
                else:
                    raise ValueError(f"Unknown loss_for_pos: {loss_for_pos}")
                loss_pos = loss_pos * loss_pos_lambda

        if coords_encoder is not None and coords_decoder is not None:
            hs_dtype = hidden_states.dtype
            if include_semantic_posemb:
                semantic_pos_emb = self.model.get_input_embeddings()(torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=hidden_states.device).unsqueeze(0)).squeeze(0)
                hidden_states = hidden_states - semantic_pos_emb
            hidden_states = hidden_states.to(torch.float32)
            decoded_coords = coords_decoder(hidden_states)[:, :, :3]
            if decoded_coords.shape[-1] < 3:
                decoded_coords = torch.cat([decoded_coords, torch.zeros((*decoded_coords.shape[:-1], 1), device=decoded_coords.device, dtype=decoded_coords.dtype)], dim=-1)
            decoded_coords[:, :, -1] = 0
            reencoded_pe = coords_encoder(decoded_coords)
            hidden_states = reencoded_pe.to(hs_dtype)
            if include_semantic_posemb:
                hidden_states = hidden_states + semantic_pos_emb

        if labels is not None and io_coords_pos_mask is not None:
            return CustomIdefics3CausalLMOutputWithPast(
                loss=loss,
                loss_pos=loss_pos if io_coords_pos_mask is not None else None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                last_hidden_state=hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=outputs.image_hidden_states,
                output_pos=output_pos,
                gt_pos=gt_pos,
                output_pe_mask=output_pe_mask,
                gt_coords_xy=gt_coords_xy,
                labels=labels,
            )
        return CustomIdefics3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            last_hidden_state=hidden_states,
            last_hidden_state_original=hidden_states_original,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        inputs_embeds = None
        input_embeds_for_output = None

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        if_next_token_pos = False
        next_token_pos_countdown = -200

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):

            model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, inputs_embeds=inputs_embeds, **model_kwargs)
            model_inputs['enable_pe_input'] = model_kwargs.get('enable_pe_input', False)

            if if_next_token_pos and not model_inputs['enable_pe_input']:
                model_inputs["input_ids"] = torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=model_inputs["input_ids"].device).reshape(model_inputs["input_ids"].shape)
            elif if_next_token_pos and model_inputs['enable_pe_input']:
                model_inputs["inputs_embeds"] = inputs_embeds[:, -1, :].unsqueeze(1)
                model_inputs["input_ids"] = None

            if_next_token_pos = False
            next_token_pos_countdown -= 1

            if model_inputs["input_ids"] is not None and model_inputs["input_ids"][0, -1] == POS_INDICATOR_TOKEN_INDEX:
                if_next_token_pos = True
                next_token_pos_countdown = 0

            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                if 'io_coords_pos' in model_inputs:
                    model_inputs.pop('io_coords_pos')
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            if return_dict_in_generate:
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )
                    if inputs_embeds is None:
                        inputs_embeds = outputs.last_hidden_state
                        input_embeds_for_output = outputs.last_hidden_state_original
                    else:
                        inputs_embeds = torch.cat([inputs_embeds, outputs.last_hidden_state], dim=1)
                        input_embeds_for_output = torch.cat([input_embeds_for_output, outputs.last_hidden_state_original], dim=1)

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            if if_next_token_pos:
                unfinished_sequences = unfinished_sequences
            else:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids, scores=scores, logits=raw_logits,
                    encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions, cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return CustomGenerateDecoderOnlyOutput(
                    sequences=input_ids, scores=scores, logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=inputs_embeds,
                    hidden_states_for_output=input_embeds_for_output,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


class CustomIdefics3Model(Idefics3Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        # SpaceDrive-specific args
        pos_emb: Optional[torch.Tensor] = None,
        io_coords_pos_mask: Optional[torch.Tensor] = None,
        io_coords_pos: Optional[torch.Tensor] = None,
        include_semantic_posemb: Optional[bool] = False,
        ego_feature: Optional[torch.Tensor] = None,
        enable_pe_input: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, Idefics3BaseModelOutputWithPast]:

        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            use_cache = False

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # Keep the native dtype to avoid SDPA mask/query dtype mismatch
        # (the attention mask may be built from inputs_embeds dtype internally).

        if enable_pe_input and io_coords_pos is not None and io_coords_pos_mask is not None:
            inputs_embeds = inputs_embeds.masked_scatter(io_coords_pos_mask, io_coords_pos)

        # Vision encoding
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # Add pos_emb to image features before merging
            if pos_emb is not None:
                pos_emb_flat = pos_emb.reshape(-1, pos_emb.shape[-1]).to(image_hidden_states.dtype)
                if pos_emb_flat.shape[0] != image_hidden_states.shape[0]:
                    # Sequence-mode batching can alter visual token count; align safely.
                    target = image_hidden_states.shape[0]
                    current = pos_emb_flat.shape[0]
                    if current > target:
                        pos_emb_flat = pos_emb_flat[:target]
                    else:
                        pad = torch.zeros(
                            target - current,
                            pos_emb_flat.shape[-1],
                            dtype=pos_emb_flat.dtype,
                            device=pos_emb_flat.device,
                        )
                        pos_emb_flat = torch.cat([pos_emb_flat, pad], dim=0)
                image_hidden_states = image_hidden_states + pos_emb_flat

            if ego_feature is not None:
                image_hidden_states_reshaped = image_hidden_states.reshape(B, -1, image_hidden_states.shape[-1])
                image_hidden_states_reshaped = torch.cat([image_hidden_states_reshaped, ego_feature], dim=-2)
                image_hidden_states = image_hidden_states_reshaped.reshape(-1, image_hidden_states_reshaped.shape[-1])

            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        if attention_mask is not None:
            # SDPA expects bool mask (or same floating dtype as query).
            attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
        )

        return Idefics3BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )
