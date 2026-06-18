# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from flow_grpo.bagel.data.data_utils import pil_img2rgb
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, gen_use_vit=True, nonmarkov_mode=None):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.gen_use_vit = gen_use_vit
        self.nonmarkov_mode = nonmarkov_mode
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    def gen_image(
        self,
        image_shape,
        gen_context,  # KV cache of the input context after being encoded by the LLM
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",

        num_timesteps=50,
        timestep_shift=3.0,

        # for grpo learn
        learn=False,
        sample=None,  # dict containing latents, prev_latents, log_probs, timesteps collected from the main script
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
        pe_loss=None,
    ):
        # Do not set the initial latent to be the same for the same prompt in eval mode
        if noise_level==0:
            generators=None
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(  # pure noise latent as the starting point for image generation
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
            generators=generators
        ) 
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        if learn:
            clipfrac, clipfrac_gt_one, clipfrac_lt_one, policy_loss, kl_loss, loss = self.model.generate_image_learn(
                sample=sample,
                grpo_config=grpo_config,
                accelerator=accelerator,
                optimizer=optimizer,
                transformer=transformer,
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                noise_level=noise_level,
                pe_loss=pe_loss,
            )
            return {
                "clipfrac": clipfrac,
                "clipfrac_gt_one": clipfrac_gt_one,
                "clipfrac_lt_one": clipfrac_lt_one,
                "policy_loss": policy_loss,
                "kl_loss": kl_loss,
                "loss": loss,
            }
        else:
            unpacked_latent, all_latents, all_log_probs, timesteps = self.model.generate_image(
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                noise_level=noise_level,
                sample_sde_window_size=grpo_config.sample.sde_window_size,
                sample_sde_window_range=grpo_config.sample.sde_window_range,
                process_index=getattr(accelerator, 'process_index', 0),
                device=getattr(accelerator, 'device', 'cuda'),
            )
            image = self.decode_image(unpacked_latent[0].float(), image_shape)
            return {
                "image": image,
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "timesteps": timesteps
            }

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample
        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].float()
        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, pad_to_max_length: bool = False):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            pad_to_max_length=pad_to_max_length,
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    def gen_text_with_logprobs(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, pad_to_max_length: bool = False):
        """Same as gen_text, but also returns token_ids and per-token log_probs (used for PE GRPO sampling)."""
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        seq, log_probs = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            pad_to_max_length=pad_to_max_length,
            return_log_probs=True,
            **generation_input,
        )
        # OOV token detection: check IDs against full vocab (base + added tokens)
        ids = seq[:, 0]
        full_vocab_size = len(self.tokenizer)  # includes added special tokens
        oov_mask = ids >= full_vocab_size
        if oov_mask.any():
            oov_positions = oov_mask.nonzero(as_tuple=True)[0].tolist()
            oov_ids = ids[oov_mask].tolist()
            # Replace OOV tokens with eos_token_id to prevent tokenizer.decode from crashing
            # (byte-level decoding cannot handle token IDs beyond the vocabulary)
            ids = ids.clone()
            ids[oov_mask] = self.new_token_ids['eos_token_id']
            seq = seq.clone()
            seq[:, 0] = ids
        output = self.tokenizer.decode(ids)
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        # seq: (seq_len, batch), log_probs: (seq_len, batch) — log_probs[i] = log P(seq[i] | context, seq[:i])
        return output, seq[:, 0], log_probs[:, 0]  # returns (text, token_ids, log_probs)

    def pe_teacher_forced_logprobs(self, pe_context, pe_token_ids, pe_token_len, temperature=1.0):
        """
        PE training phase: perform a teacher-forced forward over the stored PE token sequence
        based on the reconstructed PE context, and compute new log_probs.

        Args:
            pe_context: Context containing the KV cache (system prompt + pe_question already encoded).
            pe_token_ids: Stored PE token sequence [bos, tok1, ..., tokK].
            pe_token_len: Valid sequence length.
            temperature: Temperature consistent with the sampling phase.

        Returns:
            new_log_probs: shape (pe_token_len - 1,)
        """
        past_key_values = pe_context['past_key_values']
        kv_lens = pe_context['kv_lens']
        ropes = pe_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        new_log_probs = self.model.generate_text_learn(
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input['packed_key_value_indexes'],
            key_values_lens=generation_input['key_values_lens'],
            packed_query_position_ids=generation_input['packed_query_position_ids'],
            token_ids=pe_token_ids,
            token_len=pe_token_len,
            temperature=temperature,
        )
        return new_log_probs
        
    def init_context_with_image(self, image: Image.Image):
        """
        Initialize (gen_context, cfg_img_context) and encode the reference image into gen_context.
        Call this once outside the multi-round loop to avoid re-encoding the reference image each round.

        The reference image is encoded only into gen_context (full conditioning),
        not into cfg_img_context (image-free CFG conditioning).

        Returns:
            gen_context:     Context with the reference image KV cache included.
            cfg_img_context: Empty context (text only, no image).
        """
        gen_context = self.init_gen_context()
        cfg_img_context = self.init_gen_context()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            image_resized = self.vae_transform.resize_transform(pil_img2rgb(image))
            gen_context = self.update_context_image(
                image_resized, gen_context, vae=True, vit=self.gen_use_vit
            )
        return gen_context, cfg_img_context

    def _build_pe_question(self, history_prompts: List[str], current_prompt: str) -> str:
        """Build the PE question text (shared by sampling and training). Uses different pe_question based on nonmarkov_mode."""
        history_text = "\n".join(
            f"Round {k+1}: {p.strip()}" for k, p in enumerate(history_prompts)
        )
        if self.nonmarkov_mode == 'cm':
            pe_question = (
                f"Editing history:\n{history_text}\n\n"
                f"Current instruction to enhance: \"{current_prompt.strip()}\"\n\n"
                f"Please enhance the current instruction by doing the following:\n"
                f"If any earlier round specifies a persistent global constraint "
                f"(for example, 'Ensure all subsequent edits are yellow', 'Use leather material for all edits'; "
                f"note these are just examples — use the actual constraint from the editing history), "
                f"explicitly incorporate that constraint into the current instruction.\n"
                f"Think briefly, under 100 words.\n"
                f"Output ONLY the enhanced instruction. No explanation, no extra text."
            )
        elif self.nonmarkov_mode == 'cu':
            pe_question = (
                f"Editing history:\n{history_text}\n\n"
                f"Current instruction to enhance: \"{current_prompt.strip()}\"\n\n"
                f"Please enhance the current instruction by doing the following:\n"
                f"Replace any vague pronouns (it, them, they, its, the one, etc.) "
                f"with a specific object. Based on the objects visible in the image "
                f"and the previous editing instructions, infer what object the user "
                f"most likely refers to with the vague pronoun, and substitute the "
                f"pronoun with that specific object's name.\n"
                f"Think briefly, under 100 words.\n"
                f"Output ONLY the enhanced instruction. No explanation, no extra text."
            )
        else:
            pe_question = (
                f"Editing history:\n{history_text}\n\n"
                f"Current instruction to rewrite: \"{current_prompt.strip()}\"\n\n"
                f"Given the editing history above, rewrite the current instruction to be fully "
                f"self-contained — resolve any ambiguities, implicit references, or missing context "
                f"from the conversation history, so the instruction can be executed independently "
                f"without referring back to prior rounds.\n"
                f"Think briefly, under 100 words.\n"
                f"Output ONLY the rewritten instruction. No explanation, no extra text."
            )
        return pe_question

    def _pe_enhance_prompt_in_context(
        self, gen_context, history_prompts: List[str],
        current_prompt: str, pe_max_token_n: int = 600,
        grpo_sample: bool = False, pe_temperature: float = 0.7,
    ):
        """
        Perform in-context PE in latent space based on the current gen_context to resolve pronoun references.
        Deep-copies gen_context to avoid polluting the original context.

        Args:
            grpo_sample: If True, use do_sample=True and return (text, fallback_reason, token_ids, log_probs).
        """
        pe_question = self._build_pe_question(history_prompts, current_prompt)

        pe_context = deepcopy(gen_context)
        pe_context = self.update_context_text(VLM_THINK_SYSTEM_PROMPT, pe_context)
        pe_context = self.update_context_text(pe_question, pe_context)

        if grpo_sample:
            # Step 1: autoregressive sampling to obtain the token sequence
            # (gen_text_with_logprobs deep-copies pe_context internally, so the original is not modified)
            raw, pe_token_ids, _pe_log_probs_step = self.gen_text_with_logprobs(
                pe_context, max_length=pe_max_token_n,
                do_sample=True, temperature=pe_temperature,
                pad_to_max_length=True,
            )
            # Step 2: recompute old_log_probs via batch forward (generate_text_learn) to ensure
            # the same FlashAttention path as pe_teacher_forced_logprobs at training time.
            # Autoregressive generate_text uses flash_attn(query_len=1) while training batch forward
            # uses flash_attn(query_len=N); these use different CUDA kernels and produce different numerics.
            # pe_context is unmodified (deepcopy inside gen_text_with_logprobs) and can be reused directly.
            # pe_token_ids may be on CPU (generate_text moves to output_device); ensure it matches the model device.
            _device = pe_context['past_key_values'].key_cache[0].device
            pe_log_probs = self.pe_teacher_forced_logprobs(
                pe_context, pe_token_ids.to(_device), len(pe_token_ids),
                temperature=pe_temperature,
            )
        else:
            raw = self.gen_text(pe_context, max_length=pe_max_token_n, do_sample=False, pad_to_max_length=True)
            pe_token_ids, pe_log_probs = None, None

        raw = (raw or "").strip()

        if not raw:
            result = (current_prompt, "empty_output")
        elif "</think>" in raw:
            after_think = raw.split("</think>", 1)[1].strip()
            if after_think:
                result = (after_think, None)
            else:
                result = (current_prompt, "empty_after_think")
        else:
            result = (current_prompt, "no_think_end_tag")

        if grpo_sample:
            return result[0], result[1], pe_token_ids, pe_log_probs
        return result[0], result[1]

    def interleave_inference_incremental(
        self,
        prompt: str,
        gen_context,
        cfg_img_context,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        grpo_config=None,
        accelerator=None,
        noise_level=0.7,
        generators=None,
        skip_context_image_update=False,
    ):
        """
        Incremental single-round inference that reuses the preceding KV cache without rebuilding history.

        Each call proceeds as follows:
          1. cfg_text_context = deepcopy(gen_context)  [snapshot before the current prompt]
          2. gen_context      += current prompt
          3. cfg_img_context  += current prompt (image-free CFG conditioning)
          4. call gen_image() to generate the image
          5. gen_context      += generated image  [for reuse in the next round]
             (skipped when skip_context_image_update=True)

        Args:
            prompt:                    Editing instruction for the current round.
            gen_context:               Full context with history KV cache (reference image + prior rounds).
            cfg_img_context:           Text-only CFG conditioning context (no image).
            skip_context_image_update: If True, skip appending the generated image to gen_context.
                                       The caller must invoke update_context_image() manually
                                       to use an annotated version of the image instead.

        Returns:
            output_dict:     {"image", "all_latents", "all_log_probs", "timesteps"}
            gen_context:     Updated full context (generated image appended when skip_context_image_update=False).
            cfg_img_context: Updated text context (current prompt appended).
        """
        import numpy as np
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Snapshot before the current prompt → cfg_text (text-free CFG) conditioning
            cfg_text_context = deepcopy(gen_context)
            # Append current prompt to gen and cfg_img contexts
            gen_context = self.update_context_text(prompt, gen_context)
            cfg_img_context = self.update_context_text(prompt, cfg_img_context)

            output_dict = self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                learn=False,
                noise_level=noise_level,
                generators=generators,
                grpo_config=grpo_config,
                accelerator=accelerator,
            )

            if not skip_context_image_update:
                # Append the generated image to gen_context for reuse in the next round
                img_tensor = output_dict["image"]  # float32 (3, H, W) in [0, 1]
                arr = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                gen_img_pil = Image.fromarray(arr)
                gen_img_resized = self.vae_transform.resize_transform(pil_img2rgb(gen_img_pil))
                gen_context = self.update_context_image(
                    gen_img_resized, gen_context, vae=True, vit=self.gen_use_vit
                )

        return output_dict, gen_context, cfg_img_context

    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],  # can contain images and/or text prompts
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        learn=False,
        sample=None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
        pad_to_max_length=False,
        pe_data=None,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output, vit=understanding_output or self.gen_use_vit)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n, pad_to_max_length=pad_to_max_length)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                # --- PE GRPO loss (computed during training when turn > 0 and PE data is available) ---
                pe_loss = None
                pe_loss_for_log = None
                if learn and pe_data is not None:
                    pe_base_context = deepcopy(cfg_text_context)
                    pe_question = self._build_pe_question(
                        pe_data["pe_history_prompts"], pe_data["pe_current_prompt"]
                    )
                    pe_base_context = self.update_context_text(VLM_THINK_SYSTEM_PROMPT, pe_base_context)
                    pe_base_context = self.update_context_text(pe_question, pe_base_context)

                    new_log_probs = self.pe_teacher_forced_logprobs(
                        pe_base_context,
                        pe_data["pe_token_ids"],
                        pe_data["pe_token_len"],
                        temperature=pe_data.get("pe_temperature", 0.7),
                    )
                    old_log_probs = pe_data["pe_log_probs"][:len(new_log_probs)]
                    pe_ratio = torch.exp(new_log_probs - old_log_probs)
                    pe_advantages = pe_data["advantages"]

                    pe_unclipped = -pe_advantages * pe_ratio
                    pe_clipped = -pe_advantages * torch.clamp(
                        pe_ratio,
                        1.0 - grpo_config.train.clip_range_lt,
                        1.0 + grpo_config.train.clip_range_gt,
                    )
                    pe_loss_value = torch.mean(torch.maximum(pe_unclipped, pe_clipped))
                    pe_loss_for_log = pe_loss_value.detach()

                    # PE clipfrac statistics (aligned with the FM loss clipfrac tracking in bagel.py)
                    pe_clipfrac = torch.mean(
                        ((pe_ratio - 1.0 > grpo_config.train.clip_range_gt) | (1.0 - pe_ratio > grpo_config.train.clip_range_lt)).float()
                    ).detach()
                    pe_clipfrac_gt_one = torch.mean(
                        (pe_ratio - 1.0 > grpo_config.train.clip_range_gt).float()
                    ).detach()
                    pe_clipfrac_lt_one = torch.mean(
                        (1.0 - pe_ratio > grpo_config.train.clip_range_lt).float()
                    ).detach()

                    # Only contribute PE loss to the total loss when contribute_loss=True
                    if pe_data.get("contribute_loss", True):
                        pe_loss = pe_loss_value

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,

                    # for grpo learn
                    learn=learn,
                    sample=sample,
                    grpo_config=grpo_config,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    transformer=transformer,
                    noise_level=noise_level,
                    generators=generators,
                    pe_loss=pe_loss,  # passed in to be combined with the FM loss
                )

                # Inject PE loss metrics into the gen_image return dict
                if learn and isinstance(img, dict):
                    img["pe_loss"] = pe_loss_for_log if pe_loss_for_log is not None else img.get("policy_loss", torch.tensor(0.0)) * 0 - 1
                    img["pe_clipfrac"] = pe_clipfrac if pe_loss_for_log is not None else torch.tensor(-1.0)
                    img["pe_clipfrac_gt_one"] = pe_clipfrac_gt_one if pe_loss_for_log is not None else torch.tensor(-1.0)
                    img["pe_clipfrac_lt_one"] = pe_clipfrac_lt_one if pe_loss_for_log is not None else torch.tensor(-1.0)
                    img["pe_ratio_mean"] = pe_ratio.mean().detach() if pe_loss_for_log is not None else torch.tensor(-1.0)

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)
        return output_list[0]
