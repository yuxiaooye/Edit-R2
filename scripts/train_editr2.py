


import sys
import gc
from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from flow_grpo.fsdp_utils import save_fsdp_checkpoint, register_optimizer_offload_hooks
from diffusers.utils.torch_utils import is_compiled_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# bagel
from flow_grpo.bagel.data.data_utils import add_special_tokens, pil_img2rgb
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache
from flow_grpo.bagel.inferencer import InterleaveInferencer

import numpy as np
import flow_grpo.rewards
from flow_grpo.edival_dataset import EDiValPromptImageDataset
from flow_grpo.stat_tracking import PerPromptStatTracker
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from huggingface_hub import snapshot_download


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)



class DistributedKRepeatSampler(Sampler):
    """Each unique prompt is sampled k times per epoch across all GPUs."""

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k cannot divide n*b: k={k}, num_replicas={num_replicas}, batch_size={batch_size}"
        )
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                per_card.append(shuffled_samples[start : start + self.batch_size])
            yield per_card[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _build_interleave_list(history_images, history_prompts):
    """
    Build the BAGEL in-context multi-turn input list.

    Args:
        history_images:  List[PIL.Image] — images for rounds 0..T:
                         [0] = original reference image, [t] = image generated in round t-1 (t>=1)
        history_prompts: List[str]       — edit instructions for rounds 0..T, same length as history_images
    Returns:
        List[PIL.Image | str] — interleaved list: [img0, p0, img1, p1, ..., imgT, pT]
    """
    assert len(history_images) == len(history_prompts), (
        f"history_images ({len(history_images)}) and history_prompts ({len(history_prompts)}) must have the same length"
    )
    input_list = []
    for img, prompt in zip(history_images, history_prompts):
        input_list.append(img)
        input_list.append(prompt)
    return input_list


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image (H, W, 3) -> float32 tensor (3, H, W), value range [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """float32 tensor (3, H, W), value range [0, 1] -> PIL Image."""
    arr = (t.cpu().float().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_generators(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def _extract_turn_metadata(metadatas, turn_idx):
    """Extract the turn_idx-th metadata slice from a batch of metadata."""
    turn_metadata = []
    for meta in metadatas:
        turn_meta = {}
        for key, value in meta.items():
            if isinstance(value, list) and key in ("instruction", "task_type", "formatted_instruction", "unchanged_objects", "available_objects", "all_objects", "bg_consistency"):
                # idx = min(turn_idx, len(value) - 1)
                if turn_idx >= len(value): 
                    raise ValueError(f"Check the number of turns in the dataset — the current data does not meet requirements.")
                idx = turn_idx
                turn_meta[key] = value[idx]
            else:
                turn_meta[key] = value
        turn_metadata.append(turn_meta)
    return turn_metadata


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


# ---------------------------------------------------------------------------
# IC CoT
# ---------------------------------------------------------------------------

_PE_QUESTION_TEMPLATE = (
    "Editing history:\n{history}\n\n"
    "Current instruction to enhance: \"{current}\"\n\n"
    "Please enhance the current instruction by doing BOTH of the following:\n"
    "1. Replace any vague pronouns (it, them, they, its, the one, etc.) "
    "with the specific object names visible in the image.\n"
    "2. If any earlier round specifies a persistent global constraint "
    "(e.g., 'Ensure all subsequent edits are yellow', 'Use leather material for all edits'), "
    "explicitly incorporate that constraint into the current instruction.\n"
    "Think briefly, under 100 words.\n"
    "Output ONLY the enhanced instruction. No explanation, no extra text."
)


def enhance_prompt_bagel(
    inferencer,
    history_prompts: list,
    current_image,
    current_prompt: str,
) -> str:
    """
    Use BAGEL's own understanding pathway (ViT) to enhance the current prompt (PE).

    Uses single-image VQA format (understanding_output=True), only the und parameters,
    without touching moe_gen parameters. Ensure the model has ViT loaded (visual_und=True)
    before calling.

    Args:
        inferencer:      InterleaveInferencer instance
        history_prompts: list of prompts from previous turns (excluding current), serialized into the question
        current_image:   input image for the current turn (PIL.Image), used as visual input
        current_prompt:  the prompt for the current turn to be enhanced

    Returns:
        Enhanced prompt string; returns the original current_prompt on failure.
    """
    from flow_grpo.bagel.data.data_utils import pil_img2rgb

    history_text = "\n".join(
        f"Round {i + 1}: {p.strip()}" for i, p in enumerate(history_prompts)
    )
    question = _PE_QUESTION_TEMPLATE.format(history=history_text, current=current_prompt.strip())

    if not isinstance(current_image, Image.Image):
        current_image = Image.fromarray(current_image)

    result = inferencer(
        image=pil_img2rgb(current_image),
        text=question,
        think=True,
        understanding_output=True,
        do_sample=False,
        max_think_token_n=600,
        pad_to_max_length=True,
    )

    raw = (result.get("text") if isinstance(result, dict) else result or "").strip()

    if not raw:
        logger.warning("Failed to generate enhanced prompt; falling back to original prompt.")
        return current_prompt, "empty_output"

    # Think-mode output format: `<think>...</think> enhanced instruction`
    if "</think>" in raw:
        after_think = raw.split("</think>", 1)[1].strip()
        if after_think:
            return after_think, None
        return current_prompt, "empty_after_think"

    # No think tag: conservatively return the original prompt (avoids garbled output or truncated think)
    return current_prompt, "no_think_end_tag"



# ---------------------------------------------------------------------------
# Eval function (multi-turn in-context, incremental KV cache reuse)
# ---------------------------------------------------------------------------

def eval_fn(
    inferencer,
    inference_hyper,
    test_dataloader,
    tokenizer,
    config,
    accelerator,
    global_step,
    eval_reward_fn,
    executor,
    autocast,
    num_turns,
    transformer=None,
    prefix="",
    use_prompt_enhance=False,
    use_ic_cot=False,
):
    assert not (use_prompt_enhance and use_ic_cot), \
        "use_prompt_enhance and use_ic_cot cannot both be True"
    all_turn_images = {i: [] for i in range(num_turns)}
    all_turn_prompts = {i: [] for i in range(num_turns)}
    all_turn_enhanced_prompts = {i: [] for i in range(num_turns)}  # Enhanced prompts
    all_ref_images_batches = []
    all_rewards = defaultdict(list)
    all_ga_responses = {i: [] for i in range(num_turns)}
    all_cu_responses = {i: [] for i in range(num_turns)}

    # PE fallback accumulation across all test batches and turns (for wandb monitoring)
    _pe_fallback_total_eval = 0
    _pe_total_eval = 0

    for test_batch in tqdm(
        test_dataloader,
        desc="Eval (multi-turn):",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, metadatas, ref_images, _ = test_batch

        # Parse per-turn instructions from "p0 | p1 | ..." format; repeat the last if fewer than num_turns
        all_turn_instructions = []
        for p in prompts:
            instrs = [s.strip() for s in p.split(" | ")]
            while len(instrs) < num_turns:
                instrs.append(instrs[-1])
            all_turn_instructions.append(instrs)
        base_images = ref_images

        current_ref_images_history = [[img] for img in ref_images]

        # Collect reference image tensors for wandb display
        ref_tensor_list = [
            pil_to_tensor(img) if isinstance(img, Image.Image) else img.cpu().float()
            for img in ref_images
        ]
        local_ref_tensor = torch.stack(ref_tensor_list, dim=0).to(accelerator.device)
        gathered_ref_tensor = accelerator.gather(local_ref_tensor).cpu()
        all_ref_images_batches.append(gathered_ref_tensor)

        # Initialize KV cache for each sample (including reference image)
        _t_init = time.time()
        with torch.no_grad():
            eval_gen_contexts = []
            eval_cfg_img_contexts = []
            for idx in range(len(prompts)):
                ref_pil = (
                    ref_images[idx] if isinstance(ref_images[idx], Image.Image)
                    else tensor_to_pil(ref_images[idx])
                )
                gc_ctx, cic_ctx = inferencer.init_context_with_image(ref_pil)
                eval_gen_contexts.append(gc_ctx)
                eval_cfg_img_contexts.append(cic_ctx)

        # Original (pre-PE) prompt history for reward evaluation, ensuring fair comparison with/without PE
        original_prompts_per_turn = [[] for _ in range(len(prompts))]
        # Formatted instruction history for CU evaluation (explicit object names with brackets)
        format_instructions_per_turn = [[] for _ in range(len(prompts))]

        for turn_idx in range(num_turns):
            turn_metadata = _extract_turn_metadata(metadatas, turn_idx)
            current_ref_images_flat = [h[-1] for h in current_ref_images_history]

            # Phase 1: PE (pad_to_max_length ensures equal forward-pass count across ranks, preventing AllGather deadlock)
            enhanced_prompts = [all_turn_instructions[idx][turn_idx] for idx in range(len(prompts))]
            # Save original prompts before PE modifies enhanced_prompts (for reward evaluation)
            original_prompts_this_turn = list(enhanced_prompts)
            for k in range(len(prompts)):
                original_prompts_per_turn[k].append(original_prompts_this_turn[k])
                # Collect formatted instructions for CU evaluation
                format_instructions_per_turn[k].append(turn_metadata[k].get("formatted_instruction", original_prompts_this_turn[k]))
            if use_prompt_enhance and turn_idx > 0:
                _t_pe_phase = time.time()
                _pe_fallback_count = 0
                _pe_fallback_reasons = {}
                with torch.no_grad():
                    for idx in range(len(prompts)):
                        _t_pe_single = time.time()
                        history_prompts_pe = [all_turn_instructions[idx][t] for t in range(turn_idx)]
                        current_img_pe = current_ref_images_history[idx][-1]
                        if not isinstance(current_img_pe, Image.Image):
                            current_img_pe = tensor_to_pil(current_img_pe)
                        enhanced_prompts[idx], _pe_fallback_reason = enhance_prompt_bagel(
                            inferencer,
                            history_prompts=history_prompts_pe,
                            current_image=current_img_pe,
                            current_prompt=enhanced_prompts[idx],
                        )
                        if _pe_fallback_reason:
                            _pe_fallback_count += 1
                            _pe_fallback_reasons[_pe_fallback_reason] = _pe_fallback_reasons.get(_pe_fallback_reason, 0) + 1
                _pe_total = len(prompts)
                _pe_fallback_total_eval += _pe_fallback_count
                _pe_total_eval += _pe_total

            if use_ic_cot and turn_idx > 0:
                _t_pe_phase = time.time()
                _pe_fallback_count = 0
                _pe_fallback_reasons = {}
                with torch.no_grad():
                    for idx in range(len(prompts)):
                        _t_pe_single = time.time()
                        history_prompts_pe = [all_turn_instructions[idx][t] for t in range(turn_idx)]
                        enhanced_prompts[idx], _pe_fallback_reason = inferencer._pe_enhance_prompt_in_context(
                            gen_context=eval_gen_contexts[idx],
                            history_prompts=history_prompts_pe,
                            current_prompt=enhanced_prompts[idx],
                            pe_max_token_n=300,
                        )
                        if _pe_fallback_reason:
                            _pe_fallback_count += 1
                            _pe_fallback_reasons[_pe_fallback_reason] = _pe_fallback_reasons.get(_pe_fallback_reason, 0) + 1
                _pe_total = len(prompts)
                _pe_fallback_total_eval += _pe_fallback_count
                _pe_total_eval += _pe_total

            # Phase 2: Diffusion generation (normal FSDP sharding, fixed denoising steps, all ranks synchronized)
            _t_turn = time.time()
            images = []
            with torch.no_grad():
                for idx in range(len(prompts)):
                    output_dict, eval_gen_contexts[idx], eval_cfg_img_contexts[idx] = (
                        inferencer.interleave_inference_incremental(
                            prompt=enhanced_prompts[idx],
                            gen_context=eval_gen_contexts[idx],
                            cfg_img_context=eval_cfg_img_contexts[idx],
                            cfg_text_scale=config.sample.eval_guidance_scale,
                            cfg_img_scale=inference_hyper["cfg_img_scale"],
                            cfg_interval=inference_hyper["cfg_interval"],
                            timestep_shift=inference_hyper["timestep_shift"],
                            num_timesteps=config.sample.eval_num_steps,
                            cfg_renorm_min=inference_hyper["cfg_renorm_min"],
                            cfg_renorm_type=inference_hyper["cfg_renorm_type"],
                            image_shapes=(config.resolution, config.resolution),
                            noise_level=0,
                            grpo_config=config,
                            accelerator=accelerator,
                        )
                    )
                    images.append(output_dict["image"])
            images = torch.stack(images, dim=0)

            # Use pre-PE original prompts for reward evaluation, ensuring fair comparison with/without PE
            cur_turn_raw_prompts = original_prompts_this_turn
            # Build GA inputs: full image sequence [base, gen_0, ..., gen_{turn_idx}] and corresponding prompts
            if "edival_client_ga" in config.reward_fn:
                ga_images_history = [
                    [img if isinstance(img, Image.Image) else tensor_to_pil(img)
                     for img in current_ref_images_history[k]]  # [base, gen_0..gen_{turn_idx-1}]
                    + [tensor_to_pil(images[k])]                # + gen_{turn_idx}
                    for k in range(len(prompts))
                ]
                ga_prompts_history = [
                    original_prompts_per_turn[k]  # per-turn original (pre-PE) prompts, turn_idx+1 total
                    for k in range(len(prompts))
                ]
            else:
                ga_images_history = None
                ga_prompts_history = None
            # Build CU inputs: full image sequence + prompts + format_instructions
            if "edival_client_cu" in config.reward_fn:
                cu_images_history = [
                    [img if isinstance(img, Image.Image) else tensor_to_pil(img)
                     for img in current_ref_images_history[k]]
                    + [tensor_to_pil(images[k])]
                    for k in range(len(prompts))
                ]
                cu_prompts_history = [
                    original_prompts_per_turn[k]
                    for k in range(len(prompts))
                ]
                cu_format_instructions_history = [
                    format_instructions_per_turn[k]
                    for k in range(len(prompts))
                ]
            else:
                cu_images_history = None
                cu_prompts_history = None
                cu_format_instructions_history = None
            rewards_future = executor.submit(
                eval_reward_fn,
                images,
                cur_turn_raw_prompts,
                turn_metadata,
                current_ref_images_flat,
                base_images=base_images,
                only_strict=False,
                ga_images_history=ga_images_history,
                ga_prompts_history=ga_prompts_history,
                cu_images_history=cu_images_history,
                cu_prompts_history=cu_prompts_history,
                cu_format_instructions_history=cu_format_instructions_history,
            )
            time.sleep(0)
            rewards, extra_info = rewards_future.result()

            if extra_info.get("ga_responses"):
                local_ga = extra_info["ga_responses"]
                if accelerator.num_processes > 1:
                    gathered_ga = [None] * accelerator.num_processes
                    torch.distributed.all_gather_object(gathered_ga, local_ga)
                    merged_ga = []
                    for rank_responses in gathered_ga:
                        merged_ga.extend(rank_responses)
                    all_ga_responses[turn_idx].extend(merged_ga)
                else:
                    all_ga_responses[turn_idx].extend(local_ga)

            if extra_info.get("cu_responses"):
                local_cu = extra_info["cu_responses"]
                if accelerator.num_processes > 1:
                    gathered_cu = [None] * accelerator.num_processes
                    torch.distributed.all_gather_object(gathered_cu, local_cu)
                    merged_cu = []
                    for rank_responses in gathered_cu:
                        merged_cu.extend(rank_responses)
                    all_cu_responses[turn_idx].extend(merged_cu)
                else:
                    all_cu_responses[turn_idx].extend(local_cu)

            for key, value in rewards.items():
                rewards_gather = (
                    accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                )
                all_rewards[f"turn{turn_idx}_{key}"].append(rewards_gather)

            gathered_images = accelerator.gather(images.to(accelerator.device)).cpu()
            all_turn_images[turn_idx].append(gathered_images)

            gathered_prompt_ids = accelerator.gather(
                tokenizer(
                    cur_turn_raw_prompts,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
            ).cpu().numpy()
            gathered_batch_prompts = tokenizer.batch_decode(gathered_prompt_ids, skip_special_tokens=True)
            all_turn_prompts[turn_idx].extend(gathered_batch_prompts)

            # Collect PE-enhanced prompts (for saving to pe_prompts.json)
            if accelerator.num_processes > 1:
                gathered_enhanced = [None] * accelerator.num_processes
                torch.distributed.all_gather_object(gathered_enhanced, list(enhanced_prompts))
                merged_enhanced = []
                for rank_prompts in gathered_enhanced:
                    merged_enhanced.extend(rank_prompts)
                all_turn_enhanced_prompts[turn_idx].extend(merged_enhanced)
            else:
                all_turn_enhanced_prompts[turn_idx].extend(list(enhanced_prompts))

            new_pil_images = [tensor_to_pil(img) for img in images]
            for i, new_img in enumerate(new_pil_images):
                current_ref_images_history[i].append(new_img)

    if accelerator.is_main_process:
        final_rewards = {k: np.concatenate(v) for k, v in all_rewards.items()}
        concat_turn_images = {i: torch.cat(all_turn_images[i], dim=0) for i in range(num_turns)}

        eval_save_dir = os.path.join(
            config.logdir, config.run_name, "eval_images", f"step_{global_step}"
        )
        os.makedirs(eval_save_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            wandb_log_dict = {}

            # Save original reference images
            if all_ref_images_batches:
                orig_save_dir = os.path.join(eval_save_dir, "original")
                os.makedirs(orig_save_dir, exist_ok=True)
                all_ref_flat = []
                for batch_ref in all_ref_images_batches:
                    all_ref_flat.extend(batch_ref)
                num_orig = len(all_ref_flat)
                for idx in range(num_orig):
                    pil = all_ref_flat[idx]
                    if not isinstance(pil, Image.Image):
                        pil = tensor_to_pil(pil)
                    pil.resize((config.resolution, config.resolution)).save(
                        os.path.join(orig_save_dir, f"{idx}.jpg")
                    )

            for turn_idx in range(num_turns):
                turn_images = concat_turn_images[turn_idx]
                turn_prompts_log = all_turn_prompts[turn_idx]
                num_to_save = len(turn_images)

                turn_save_dir = os.path.join(eval_save_dir, f"turn{turn_idx}")
                os.makedirs(turn_save_dir, exist_ok=True)
                turn_tmpdir = os.path.join(tmpdir, f"turn{turn_idx}")
                os.makedirs(turn_tmpdir, exist_ok=True)

                for idx in range(num_to_save):
                    pil = tensor_to_pil(turn_images[idx])
                    pil.save(os.path.join(turn_tmpdir, f"{idx}.jpg"))
                    pil.save(os.path.join(turn_save_dir, f"{idx}.jpg"))

                turn_reward_keys = [k for k in final_rewards if k.startswith(f"turn{turn_idx}_")]
                sampled_rewards_log = [
                    {k: final_rewards[k][i] for k in turn_reward_keys}
                    for i in range(num_to_save)
                ]
                wandb_log_dict[f"{prefix}eval_images_turn{turn_idx}"] = [
                    wandb.Image(
                        os.path.join(turn_tmpdir, f"{idx}.jpg"),
                        caption=f"turn{turn_idx}: {turn_prompts_log[idx]:.200} | "
                        + " | ".join(f"{k}: {v:.2f}" for k, v in r.items() if v != -10),
                    )
                    for idx, r in enumerate(sampled_rewards_log)
                ]

            # Save metadata JSON
            metadata_json = {
                "global_step": global_step,
                "num_turns": num_turns,
                "samples": [
                    {
                        "turn": turn_idx,
                        "idx": idx,
                        "prompt": all_turn_prompts[turn_idx][idx],
                        "rewards": {
                            k: float(final_rewards[k][idx])
                            for k in final_rewards if k.startswith(f"turn{turn_idx}_")
                        },
                        **({"ga_response": all_ga_responses[turn_idx][idx]}
                           if idx < len(all_ga_responses[turn_idx]) else {}),
                        **({"cu_response": all_cu_responses[turn_idx][idx]}
                           if idx < len(all_cu_responses[turn_idx]) else {}),
                    }
                    for turn_idx in range(num_turns)
                    for idx in range(len(concat_turn_images[turn_idx]))
                ],
            }
            with open(os.path.join(eval_save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata_json, f, indent=2, ensure_ascii=False)

            # Save before/after PE enhanced prompt comparison
            pe_prompts_json = {
                "global_step": global_step,
                "num_turns": num_turns,
                "pe_prompts": [
                    {
                        "turn": turn_idx,
                        "idx": idx,
                        "before": all_turn_prompts[turn_idx][idx],
                        "after": all_turn_enhanced_prompts[turn_idx][idx],
                    }
                    for turn_idx in range(num_turns)
                    for idx in range(len(all_turn_prompts[turn_idx]))
                    if idx < len(all_turn_enhanced_prompts[turn_idx])
                ]
            }
            with open(os.path.join(eval_save_dir, "pe_prompts.json"), "w", encoding="utf-8") as f:
                json.dump(pe_prompts_json, f, indent=2, ensure_ascii=False)

            # Save the initial ref_image for the entire trace
            if all_ref_images_batches:
                ref_img_tensor = all_ref_images_batches[0][0]  # take the first sample of the first batch
                ref_pil = tensor_to_pil(ref_img_tensor) if not isinstance(ref_img_tensor, Image.Image) else ref_img_tensor
                ref_tmp_path = os.path.join(tmpdir, "ref_image.jpg")
                ref_pil.save(ref_tmp_path)
                wandb_log_dict[f"{prefix}eval_ref_image"] = wandb.Image(ref_tmp_path)

            wandb_log_dict.update({
                f"{prefix}eval_reward_{key}": np.mean(value[value != -10])
                for key, value in final_rewards.items()
            })
            # PE fallback ratio monitoring
            if _pe_total_eval > 0:
                wandb_log_dict[f"{prefix}eval_pe_fallback_ratio"] = _pe_fallback_total_eval / _pe_total_eval
            wandb.log(wandb_log_dict, step=global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    config = FLAGS.config

    if hasattr(config, "edival_if_coef"):
        os.environ["EDIVAL_IF_COEF"] = str(config.edival_if_coef)

    if hasattr(config, "edival_if_num_votes"):
        os.environ["EDIVAL_IF_NUM_VOTES"] = str(config.edival_if_num_votes)

    if hasattr(config, "edival_ga_num_votes"):
        os.environ["EDIVAL_GA_NUM_VOTES"] = str(config.edival_ga_num_votes)

    if hasattr(config, "edival_ga_temperature"):
        os.environ["EDIVAL_GA_TEMPERATURE"] = str(config.edival_ga_temperature)

    num_turns = getattr(config, "num_turns", 2)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name + "_" + unique_id) if config.run_name else unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # For multi-turn, multiply gradient accumulation steps by number of turns
        gradient_accumulation_steps=(
            config.train.gradient_accumulation_steps
            * config.sample.train_batch_size
            * config.sample.sde_window_size
            * num_turns
        ),
    )
    accelerator.state.fsdp_plugin.activation_checkpointing = config.activation_checkpointing
    accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ["Qwen2MoTDecoderLayer"]

    os.environ["WANDB_API_KEY"] = "f8fd18480e9bfbe6ff0a96cfe0a378a9984cbe8e"  

    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
            name=config.run_name,
            config=config.to_dict(),
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    model_path = config.pretrained.model
    if not os.path.exists(model_path):
        model_local_dir = snapshot_download(repo_id=model_path)
    else:
        model_local_dir = model_path

    llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    if config.vae_donot_sample:
        vae_model.reg.sample = False

    # PE requires ViT (understanding pathway); whether to load depends on config switch
    use_prompt_enhance = getattr(config, "use_prompt_enhance", False)
    use_ic_cot = getattr(config, "use_ic_cot", False)
    use_ic_cot_grpo_loss = getattr(config, "use_ic_cot_grpo_loss", False)
    pe_temperature = getattr(config, "pe_temperature", 0.7)
    pe_max_token_n = getattr(config, "pe_max_token_n", 300)
    assert not (use_prompt_enhance and use_ic_cot), \
        "use_prompt_enhance and use_ic_cot cannot both be True"
    if use_ic_cot:
    use_vit = getattr(config, "use_vit", False)

    if use_prompt_enhance or use_ic_cot or use_vit:
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_local_dir, "vit_config.json")
            
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, bagel_config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    else:
        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=False,
            llm_config=llm_config,
            vit_config=None,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            model = Bagel(language_model, None, bagel_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(*config.vae_transform)
    vit_transform = ImageTransform(*config.vit_transform)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
        device_map={"": f"cuda:{accelerator.local_process_index}"},
        offload_buffers=False,
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/tmp/offload",
    )
    model = model.eval()

    # --- Text Key Scaling: inject text_key_scale from config into LLM config ---
    _tks = getattr(config, 'text_key_scale', 1.0)
    model.language_model.config.text_key_scale = _tks
    if _tks != 1.0:

    # KL regularization reference model
    if config.train.beta > 0:
        language_model_ref = Qwen2ForCausalLM(llm_config)
        language_model_ref.load_state_dict(model.language_model.state_dict())
        language_model_ref.to(
            device=f"cuda:{accelerator.local_process_index}", dtype=inference_dtype
        )
        language_model_ref.eval()
        language_model_ref.requires_grad_(False)
        language_model_ref.config.text_key_scale = _tks  # keep consistent with the policy model

    vae_model.requires_grad_(False)
    model.requires_grad_(False)



    inference_hyper = dict(
        cfg_img_scale=config.sample.cfg_img_scale,
        cfg_interval=[0, 1.0],
        timestep_shift=config.train.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type=config.cfg_renorm_type,
        image_shapes=(config.resolution, config.resolution),
    )

    nonmarkov_mode = getattr(config, "nonmarkov_mode", None)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        gen_use_vit=getattr(config, "gen_use_vit", True),
        nonmarkov_mode=nonmarkov_mode,
    )

    vae_model.to(accelerator.device, dtype=inference_dtype)
    model.to(accelerator.device, dtype=inference_dtype)


    if config.use_lora:
        target_modules = [
            "self_attn.q_proj_moe_gen",
            "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen",
            "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ]
        if use_ic_cot_grpo_loss:
            target_modules += [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.language_model = get_peft_model(model.language_model, transformer_lora_config)
        for name, param in model.language_model.named_parameters():
            if "lora" in name:
                param.data = param.data.to(dtype=inference_dtype)


        resume_lora_ckpt = getattr(config, "resume_lora_checkpoint", None)
        if resume_lora_ckpt:
            logger.info(f"Loading LoRA checkpoint from {resume_lora_ckpt}")
            from safetensors.torch import load_file as load_safetensors
            ckpt_state = load_safetensors(resume_lora_ckpt)
            missing, unexpected = model.language_model.load_state_dict(ckpt_state, strict=False)
            logger.info(f"LoRA checkpoint loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        for name, param in model.language_model.named_parameters():
            if "moe_gen" in name:
                param.requires_grad = True

    transformer = model.language_model
    transformer.config.use_cache = False
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    ema = None

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    if config.fsdp_optimizer_offload:
        register_optimizer_offload_hooks(optimizer)

    # -------------------------------------------------------------------------
    # Dataset (edival format only)
    # -------------------------------------------------------------------------
    train_dataset = EDiValPromptImageDataset(config.dataset, config.resolution, "train")
    test_dataset = EDiValPromptImageDataset(config.dataset, config.resolution, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=EDiValPromptImageDataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=EDiValPromptImageDataset.collate_fn,
        shuffle=False,
        num_workers=4,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(accelerator.device, config.reward_fn)

    if config.train.beta > 0:
        transformer, language_model_ref, optimizer, train_dataloader, test_dataloader = (
            accelerator.prepare(
                transformer, language_model_ref, optimizer, train_dataloader, test_dataloader
            )
        )
        model.language_model_ref = language_model_ref
    else:
        transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, test_dataloader
        )
    model.language_model = transformer

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = ( # number of samples per epoch
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
        * num_turns
    )
    total_train_batch_size = ( # effective actual training batch size
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running BAGEL in-context multi-turn GRPO training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Num Turns  = {num_turns}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device  = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps  = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total samples per epoch      = {samples_per_epoch}")
    logger.info(f"  Total train batch size       = {total_train_batch_size}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    train_iter = iter(train_dataloader)
    global_step = 0

    for epoch in range(first_epoch, config.num_epochs):

        # ---------------------------------------------------------------
        # EVAL + CHECKPOINT
        # ---------------------------------------------------------------
        transformer.eval()

        if not config.debug and epoch % config.save_freq == 0 and epoch > 0:
            save_fsdp_checkpoint(
                config.save_dir, transformer, global_step, accelerator.process_index
            )

        if not config.debug and epoch % config.eval_freq == 0:
            if hasattr(config, "markov") and config.markov:
                eval_fn_markov(                
                    inferencer,
                    inference_hyper,
                    test_dataloader,
                    tokenizer,
                    config,
                    accelerator,
                    global_step,
                    eval_reward_fn,
                    executor,
                    autocast,
                    num_turns
                )
            else:
                eval_fn(
                    inferencer,
                    inference_hyper,
                    test_dataloader,
                    tokenizer,
                    config,
                    accelerator,
                    global_step,
                    eval_reward_fn,
                    executor,
                    autocast,
                    num_turns,
                    transformer=transformer,
                    use_prompt_enhance=use_prompt_enhance,
                    use_ic_cot=use_ic_cot,
                )
            if config.val_only: 
                sys.exit()

        # ---------------------------------------------------------------
        # SAMPLING (multi-turn, incremental KV cache reuse)
        # ---------------------------------------------------------------
        gc.collect()
        torch.cuda.empty_cache()
        transformer.eval()
        samples = []

        # PE fallback accumulation across all sample batches and turns (for wandb monitoring)
        _pe_fallback_total_sample = 0
        _pe_total_sample = 0

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, metadatas, ref_images, prompt_with_image_paths = next(train_iter)

            # Parse multi-turn instructions: prompts[k] = "instruction_0 | instruction_1 | ..."
            all_turn_instructions = []
            for p in prompts:
                instrs = [s.strip() for s in p.split(" | ")]
                while len(instrs) < num_turns:
                    instrs.append(instrs[-1])
                all_turn_instructions.append(instrs)
            base_images = ref_images

            # Store the full prompt for stat_tracker (shared key across all turns)
            prompt_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # Tokenize each turn's instruction separately for training replay decoding
            per_turn_prompt_ids_list = []
            for t in range(num_turns):
                t_instrs = [
                    all_turn_instructions[k][t] if t < len(all_turn_instructions[k]) else ""
                    for k in range(len(prompts))
                ]
                t_ids = tokenizer(
                    t_instrs,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                per_turn_prompt_ids_list.append(t_ids)
            # per_turn_prompt_ids shape: (B, num_turns, 256)
            per_turn_prompt_ids = torch.stack(per_turn_prompt_ids_list, dim=1)
            # Save original (pre-PE) prompt ids for reconstructing the PE question during training
            orig_per_turn_prompt_ids = per_turn_prompt_ids.clone()

            generators = create_generators(prompt_with_image_paths, base_seed=42)

            # History image list: history[k] = [ref_img, gen_0, gen_1, ...]
            current_ref_images_history = [[img] for img in ref_images]

            # Initialize KV cache for each sample (including reference image)
            _t_kv_init = time.time()
            with torch.no_grad():
                sample_gen_contexts = []
                sample_cfg_img_contexts = []
                for idx in range(len(prompts)):
                    ref_pil = (
                        ref_images[idx] if isinstance(ref_images[idx], Image.Image)
                        else tensor_to_pil(ref_images[idx])
                    )
                    gc_ctx, cic_ctx = inferencer.init_context_with_image(ref_pil)
                    sample_gen_contexts.append(gc_ctx)
                    sample_cfg_img_contexts.append(cic_ctx)

            # Original (pre-PE) prompt history for reward evaluation, ensuring fair comparison with/without PE
            original_prompts_per_turn = [[] for _ in range(len(prompts))]
            # Formatted instruction history for CU evaluation (explicit object names with brackets)
            format_instructions_per_turn = [[] for _ in range(len(prompts))]

            # Collect pre/post-PE prompts for saving
            turn_pe_prompts_log = {i: [] for i in range(num_turns)}

            for turn_idx in range(num_turns):
                turn_metadata = _extract_turn_metadata(metadatas, turn_idx)
                current_ref_images_flat = [h[-1] for h in current_ref_images_history]

                # Phase 1: PE (pad_to_max_length ensures equal forward-pass count across ranks, preventing AllGather deadlock)
                enhanced_prompts = [all_turn_instructions[idx][turn_idx] for idx in range(len(prompts))]
                # Save original prompts before PE modifies enhanced_prompts (for reward evaluation)
                original_prompts_this_turn = list(enhanced_prompts)
                for k in range(len(prompts)):
                    original_prompts_per_turn[k].append(original_prompts_this_turn[k])
                    # Collect formatted instructions for CU evaluation
                    format_instructions_per_turn[k].append(turn_metadata[k].get("formatted_instruction", original_prompts_this_turn[k]))
                if use_prompt_enhance and turn_idx > 0:
                    _t_pe_phase = time.time()
                    _pe_fallback_count = 0
                    _pe_fallback_reasons = {}
                    with torch.no_grad():
                        for idx in range(len(prompts)):
                            _t_pe_single = time.time()
                            history_prompts_pe = [all_turn_instructions[idx][t] for t in range(turn_idx)]
                            current_img_pe = current_ref_images_history[idx][-1]
                            if not isinstance(current_img_pe, Image.Image):
                                current_img_pe = tensor_to_pil(current_img_pe)
                            enhanced_prompts[idx], _pe_fallback_reason = enhance_prompt_bagel(
                                inferencer,
                                history_prompts=history_prompts_pe,
                                current_image=current_img_pe,
                                current_prompt=enhanced_prompts[idx],
                            )
                            if _pe_fallback_reason:
                                _pe_fallback_count += 1
                                _pe_fallback_reasons[_pe_fallback_reason] = _pe_fallback_reasons.get(_pe_fallback_reason, 0) + 1
                            new_ids = tokenizer(
                                enhanced_prompts[idx],
                                padding="max_length",
                                max_length=256,
                                truncation=True,
                                return_tensors="pt",
                            ).input_ids[0].to(accelerator.device)
                            per_turn_prompt_ids[idx, turn_idx] = new_ids
                    _pe_total = len(prompts)
                    _pe_fallback_total_sample += _pe_fallback_count
                    _pe_total_sample += _pe_total

                # PE data collection (turn 0 has no PE, zero-padded)
                batch_pe_token_ids = []
                batch_pe_log_probs = []
                batch_pe_token_lens = []

                if use_ic_cot and turn_idx > 0:
                    _t_pe_phase = time.time()
                    _pe_fallback_count = 0
                    _pe_fallback_reasons = {}
                    with torch.no_grad():
                        for idx in range(len(prompts)):
                            _t_pe_single = time.time()
                            history_prompts_pe = [all_turn_instructions[idx][t] for t in range(turn_idx)]
                            # Must use autocast to align with the autocast inside interleave_inference during training;
                            # otherwise update_context_text / generate_text take different precision paths,
                            # causing minor PE KV cache deviation -> pe_ratio != 1.0 at on-policy time
                            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                                enhanced_prompts[idx], _pe_fallback_reason, pe_tids, pe_lps = (
                                    inferencer._pe_enhance_prompt_in_context(
                                        gen_context=sample_gen_contexts[idx],
                                        history_prompts=history_prompts_pe,
                                        current_prompt=enhanced_prompts[idx],
                                        pe_max_token_n=pe_max_token_n,
                                        grpo_sample=True,
                                        pe_temperature=pe_temperature,
                                    )
                                )
                            if _pe_fallback_reason:
                                _pe_fallback_count += 1
                                _pe_fallback_reasons[_pe_fallback_reason] = _pe_fallback_reasons.get(_pe_fallback_reason, 0) + 1

                            # Collect PE token data (padded to pe_max_token_n)
                            pe_len = len(pe_tids)
                            padded_tids = torch.zeros(pe_max_token_n, dtype=torch.long, device=accelerator.device)
                            padded_lps = torch.zeros(pe_max_token_n, device=accelerator.device)
                            padded_tids[:pe_len] = pe_tids
                            padded_lps[:min(len(pe_lps), pe_max_token_n)] = pe_lps[:pe_max_token_n]
                            batch_pe_token_ids.append(padded_tids)
                            batch_pe_log_probs.append(padded_lps)
                            batch_pe_token_lens.append(pe_len)

                            new_ids = tokenizer(
                                enhanced_prompts[idx],
                                padding="max_length",
                                max_length=256,
                                truncation=True,
                                return_tensors="pt",
                            ).input_ids[0].to(accelerator.device)
                            per_turn_prompt_ids[idx, turn_idx] = new_ids
                    _pe_total = len(prompts)
                    _pe_fallback_total_sample += _pe_fallback_count
                    _pe_total_sample += _pe_total
                else:
                    # turn 0 or PE-latent not enabled: zero-pad
                    for idx in range(len(prompts)):
                        batch_pe_token_ids.append(torch.zeros(pe_max_token_n, dtype=torch.long, device=accelerator.device))
                        batch_pe_log_probs.append(torch.zeros(pe_max_token_n, device=accelerator.device))
                        batch_pe_token_lens.append(0)

                # Collect pre/post-PE prompts (all samples, regardless of whether PE took effect)
                for idx in range(len(prompts)):
                    turn_pe_prompts_log[turn_idx].append({
                        "idx": idx,
                        "before": original_prompts_this_turn[idx],
                        "after": enhanced_prompts[idx],
                    })

                # Phase 2: Diffusion generation (normal FSDP sharding, fixed denoising steps, all ranks synchronized)
                _t_turn = time.time()
                images = []
                latents = []
                log_probs = []
                timesteps_list = []
                with torch.no_grad():
                    for idx in range(len(prompts)):
                        generator = generators[idx : idx + 1] if config.sample.same_latent else None
                        output_dict, sample_gen_contexts[idx], sample_cfg_img_contexts[idx] = (
                            inferencer.interleave_inference_incremental(
                                prompt=enhanced_prompts[idx],
                                gen_context=sample_gen_contexts[idx],
                                cfg_img_context=sample_cfg_img_contexts[idx],
                                cfg_text_scale=config.sample.guidance_scale,
                                cfg_img_scale=inference_hyper["cfg_img_scale"],
                                cfg_interval=inference_hyper["cfg_interval"],
                                timestep_shift=inference_hyper["timestep_shift"],
                                num_timesteps=config.sample.num_steps,
                                cfg_renorm_min=inference_hyper["cfg_renorm_min"],
                                cfg_renorm_type=inference_hyper["cfg_renorm_type"],
                                image_shapes=(config.resolution, config.resolution),
                                noise_level=config.sample.noise_level,
                                generators=generator,
                                grpo_config=config,
                                accelerator=accelerator,
                            )
                        )
                        images.append(output_dict["image"])
                        latents.append(output_dict["all_latents"])
                        log_probs.append(output_dict["all_log_probs"])
                        timesteps_list.append(output_dict["timesteps"])

                stacked_latents = torch.stack(
                    [torch.stack(inner, dim=0) for inner in latents], dim=0
                )
                stacked_log_probs = torch.stack(
                    [torch.stack(inner, dim=0) for inner in log_probs], dim=0
                )
                stacked_timesteps = torch.stack(timesteps_list, dim=0)
                stacked_images = torch.stack(images, dim=0)

                # Store the current turn's history images (including reference image)
                # for training replay to reconstruct the interleave context
                # shape: (B, turn_idx+1, 3, H, W)
                ref_images_history_tensor = torch.stack([
                    torch.stack(
                        [pil_to_tensor(img) for img in current_ref_images_history[k]], dim=0
                    )
                    for k in range(len(ref_images))
                ], dim=0).to(accelerator.device)

                cur_ref_images_tensor = torch.stack(
                    [pil_to_tensor(img) for img in current_ref_images_flat], dim=0
                ).to(accelerator.device)

                turn_idx_tensor = torch.full(
                    (len(prompts),), turn_idx, dtype=torch.long, device=accelerator.device
                )

                # Use pre-PE original prompts for reward evaluation, ensuring fair comparison with/without PE
                cur_turn_raw_prompts = original_prompts_this_turn
                # Build GA inputs: full image sequence [base, gen_0, ..., gen_{turn_idx}] and corresponding prompts
                if "edival_client_ga" in config.reward_fn:
                    ga_images_history = [
                        [img if isinstance(img, Image.Image) else tensor_to_pil(img)
                         for img in current_ref_images_history[k]]  # [base, gen_0..gen_{turn_idx-1}]
                        + [tensor_to_pil(stacked_images[k])]        # + gen_{turn_idx}
                        for k in range(len(prompts))
                    ]
                    ga_prompts_history = [
                        original_prompts_per_turn[k]  # per-turn original (pre-PE) prompts, turn_idx+1 total
                        for k in range(len(prompts))
                    ]
                else:
                    ga_images_history = None
                    ga_prompts_history = None
                # Build CU inputs: full image sequence + prompts + format_instructions
                if "edival_client_cu" in config.reward_fn:
                    cu_images_history = [
                        [img if isinstance(img, Image.Image) else tensor_to_pil(img)
                         for img in current_ref_images_history[k]]
                        + [tensor_to_pil(stacked_images[k])]
                        for k in range(len(prompts))
                    ]
                    cu_prompts_history = [
                        original_prompts_per_turn[k]
                        for k in range(len(prompts))
                    ]
                    cu_format_instructions_history = [
                        format_instructions_per_turn[k]
                        for k in range(len(prompts))
                    ]
                else:
                    cu_images_history = None
                    cu_prompts_history = None
                    cu_format_instructions_history = None
                rewards_future = executor.submit(
                    reward_fn,
                    stacked_images,
                    cur_turn_raw_prompts,
                    turn_metadata,
                    current_ref_images_flat,
                    base_images=base_images,
                    only_strict=True,
                    ga_images_history=ga_images_history,
                    ga_prompts_history=ga_prompts_history,
                    cu_images_history=cu_images_history,
                    cu_prompts_history=cu_prompts_history,
                    cu_format_instructions_history=cu_format_instructions_history,
                )
                time.sleep(0)

                samples.append({
                    "prompt_ids": prompt_ids,
                    "per_turn_prompt_ids": per_turn_prompt_ids,
                    "orig_per_turn_prompt_ids": orig_per_turn_prompt_ids,
                    "timesteps": stacked_timesteps,
                    "latents": stacked_latents[:, :-1],
                    "prev_latents": stacked_latents[:, 1:],
                    "log_probs": stacked_log_probs,
                    "ref_images": cur_ref_images_tensor,
                    "ref_images_history": ref_images_history_tensor,
                    "gen_images": stacked_images,
                    "turn_idx_stored": turn_idx_tensor,
                    # PE GRPO data
                    "pe_token_ids": torch.stack(batch_pe_token_ids, dim=0),
                    "pe_log_probs": torch.stack(batch_pe_log_probs, dim=0),
                    "pe_token_lens": torch.tensor(batch_pe_token_lens, dtype=torch.long, device=accelerator.device),
                    "rewards": rewards_future,
                })

                # Append current turn's generated image to history for the next turn
                new_pil_images = [tensor_to_pil(img) for img in stacked_images]
                for k, new_img in enumerate(new_pil_images):
                    current_ref_images_history[k].append(new_img)

        # Wait for all reward computations to complete, and collect per-turn images and prompts for wandb logging
        # Organize data per turn_idx for per-turn image logging
        turn_images_for_wandb = {i: [] for i in range(num_turns)}  # turn_idx -> list of (image_tensor, prompt, rewards)
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, _ = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            # Collect per-turn image data
            turn_idx = int(sample["turn_idx_stored"][0].item())
            # gen_images: (B, 3, H, W), the images actually generated this turn
            gen_img = sample["gen_images"]  # (B, 3, H, W)
            # Decode the current turn's prompt
            per_turn_ids = sample["per_turn_prompt_ids"]  # (B, num_turns, 256)
            for b in range(gen_img.shape[0]):
                turn_images_for_wandb[turn_idx].append({
                    "image": gen_img[b],  # (3, H, W)
                    "prompt_ids": per_turn_ids[b, turn_idx],  # (256,)
                    "rewards": {k: sample["rewards"][k][b].item() for k in sample["rewards"]},
                })

        # ---  mark used_to_learn ---
        # samples are arranged as [b0t0, b0t1, ..., b1t0, b1t1, ...], every num_turns form one batch group
        # If a sample has edival_if=0 at turn i, that sample's used_to_learn=False for subsequent turns i+1, i+2, ...
        _filter_total_marked = 0  # number of sample-turns marked as non-learning
        _filter_total_eligible = 0  # total number of sample-turns with turn>0
        num_batch_groups = len(samples) // num_turns
        for batch_g in range(num_batch_groups):
            batch_samples = samples[batch_g * num_turns : (batch_g + 1) * num_turns]
            B = batch_samples[0]["turn_idx_stored"].shape[0]
            skip_set = set()  # set of sample indices to skip (within the batch)

            for t, s in enumerate(batch_samples):
                # Create used_to_learn flag
                used_to_learn = torch.ones(B, dtype=torch.bool, device=accelerator.device)
                if t > 0:
                    _filter_total_eligible += B
                    for idx in skip_set:
                        used_to_learn[idx] = False
                    n_marked = len(skip_set)
                    _filter_total_marked += n_marked
                    if n_marked > 0:
                        pass
                s["used_to_learn"] = used_to_learn

                # Check edival_if for the current turn, update skip_set for subsequent turns
                if "edival_client_if" in s["rewards"]:
                    if_scores = s["rewards"]["edival_client_if"]
                    for idx in range(B):
                        if idx not in skip_set and if_scores[idx].item() == 0.0:
                            skip_set.add(idx)

        # Pad ref_images_history to the same history length to ensure consistent shape during collate
        _max_hist = max(s["ref_images_history"].shape[1] for s in samples)
        for s in samples:
            hist = s["ref_images_history"]
            if hist.shape[1] < _max_hist:
                pad = hist[:, -1:].expand(-1, _max_hist - hist.shape[1], -1, -1, -1).clone()
                s["ref_images_history"] = torch.cat([hist, pad], dim=1)

        # collate
        tensor_keys = [k for k in samples[0] if k != "rewards"]
        samples_collated = {}
        for k in tensor_keys:
            samples_collated[k] = torch.cat([s[k] for s in samples], dim=0)
        samples_collated["rewards"] = {
            sub_key: torch.cat([s["rewards"][sub_key] for s in samples], dim=0)
            for sub_key in samples[0]["rewards"]
        }
        samples = samples_collated

        # Group rewards by turn_idx for separate logging
        # Use turn_idx_stored to distinguish data from different turns
        turn_rewards_grouped = {i: defaultdict(list) for i in range(num_turns)}
        turn_idx_tensor = samples["turn_idx_stored"]  # (total_batch_size,)
        for turn_idx in range(num_turns):
            mask = turn_idx_tensor == turn_idx  # find samples belonging to this turn
            if mask.sum() > 0:
                for key in samples["rewards"]:
                    turn_rewards_grouped[turn_idx][key] = samples["rewards"][key][mask]

        # Log sampled images per turn during training (similar to eval_fn)
        # Gather must be executed on all processes, not inside is_main_process
        if epoch % 5 == 0:
            _train_gathered_images = {}
            _train_gathered_prompt_ids = {}
            _train_gathered_rewards = {}
            for turn_idx in range(num_turns):
                turn_data = turn_images_for_wandb[turn_idx]
                if len(turn_data) == 0:
                    continue
                _imgs = torch.stack([d["image"] for d in turn_data], dim=0).to(accelerator.device)
                _pids = torch.stack([d["prompt_ids"] for d in turn_data], dim=0).to(accelerator.device)
                _rew_keys = list(turn_data[0]["rewards"].keys())
                _rews = {
                    k: torch.tensor([d["rewards"][k] for d in turn_data], device=accelerator.device)
                    for k in _rew_keys
                }
                _train_gathered_images[turn_idx] = accelerator.gather(_imgs).cpu()
                _train_gathered_prompt_ids[turn_idx] = accelerator.gather(_pids).cpu()
                _train_gathered_rewards[turn_idx] = {
                    k: accelerator.gather(v).cpu().numpy() for k, v in _rews.items()
                }

            # Gather PE prompts across ranks
            _train_gathered_pe_prompts = {}
            for turn_idx in range(num_turns):
                local_pe = turn_pe_prompts_log[turn_idx]
                if accelerator.num_processes > 1:
                    gathered_pe = [None] * accelerator.num_processes
                    torch.distributed.all_gather_object(gathered_pe, local_pe)
                    merged_pe = []
                    for rank_entries in gathered_pe:
                        merged_pe.extend(rank_entries)
                    _train_gathered_pe_prompts[turn_idx] = merged_pe
                else:
                    _train_gathered_pe_prompts[turn_idx] = local_pe

            if accelerator.is_main_process:
                train_save_dir = os.path.join(
                    config.logdir, config.run_name, "train_images", f"step_{global_step}"
                )
                os.makedirs(train_save_dir, exist_ok=True)

                # Save before/after PE enhanced prompt comparison
                pe_prompts_json = {
                    "global_step": global_step,
                    "num_turns": num_turns,
                    "pe_prompts": [
                        {
                            "turn": turn_idx,
                            "idx": entry["idx"],
                            "before": entry["before"],
                            "after": entry["after"],
                        }
                        for turn_idx in range(num_turns)
                        for entry in _train_gathered_pe_prompts.get(turn_idx, [])
                    ]
                }
                with open(os.path.join(train_save_dir, "pe_prompts.json"), "w", encoding="utf-8") as f:
                    json.dump(pe_prompts_json, f, indent=2, ensure_ascii=False)

                with tempfile.TemporaryDirectory() as tmpdir:
                    wandb_log_dict = {}
                    for turn_idx in range(num_turns):
                        if turn_idx not in _train_gathered_images:
                            continue
                        turn_images_gathered = _train_gathered_images[turn_idx]
                        turn_prompt_ids_gathered = _train_gathered_prompt_ids[turn_idx].numpy()
                        turn_rewards_gathered = _train_gathered_rewards[turn_idx]

                        turn_save_dir = os.path.join(train_save_dir, f"turn{turn_idx}")
                        os.makedirs(turn_save_dir, exist_ok=True)
                        turn_tmpdir = os.path.join(tmpdir, f"turn{turn_idx}")
                        os.makedirs(turn_tmpdir, exist_ok=True)

                        for idx in range(len(turn_images_gathered)):
                            pil = tensor_to_pil(turn_images_gathered[idx])
                            pil.save(os.path.join(turn_tmpdir, f"{idx}.jpg"))
                            pil.save(os.path.join(turn_save_dir, f"{idx}.jpg"))

                        decoded_prompts = tokenizer.batch_decode(
                            turn_prompt_ids_gathered, skip_special_tokens=True
                        )
                        wandb_log_dict[f"train_images_turn{turn_idx}"] = [
                            wandb.Image(
                                os.path.join(turn_tmpdir, f"{idx}.jpg"),
                                caption=f"turn{turn_idx}: {decoded_prompts[idx]:.200} | "
                                + " | ".join(
                                    f"{k}: {v:.2f}"
                                    for k, v in {k: turn_rewards_gathered[k][idx] for k in turn_rewards_gathered}.items()
                                    if v != -10
                                ),
                            )
                            for idx in range(len(turn_images_gathered))
                        ]

                    # Log the initial ref_image for the entire trace (take the first sample of collated samples)
                    if "ref_images_history" in samples:
                        ref_img_tensor = samples["ref_images_history"][0, 0]  # (3, H, W)
                        ref_pil = tensor_to_pil(ref_img_tensor)
                        ref_tmp_path = os.path.join(tmpdir, "ref_image.jpg")
                        ref_pil.save(ref_tmp_path)
                        wandb_log_dict["ref_image"] = wandb.Image(ref_tmp_path)

                    if wandb_log_dict:
                        wandb.log(wandb_log_dict, step=global_step)

        # Reward bookkeeping
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1)
        gathered_rewards = {
            key: accelerator.gather(value).cpu().numpy()
            for key, value in samples["rewards"].items()
        }

        # Gather per-turn rewards and log them separately
        gathered_turn_rewards = {i: {} for i in range(num_turns)}
        for turn_idx in range(num_turns):
            for key in samples["rewards"]:
                turn_values = turn_rewards_grouped[turn_idx][key]
                if len(turn_values) > 0:
                    gathered_turn_rewards[turn_idx][key] = (
                        accelerator.gather(turn_values).cpu().numpy()
                    )

        if accelerator.is_main_process:
            wandb_log_dict = {"epoch": epoch}
            # Log average of all rewards (backward compatible)
            for key, value in gathered_rewards.items():
                if "_strict_accuracy" not in key and "_accuracy" not in key:
                    wandb_log_dict[f"reward_{key}"] = value.mean()
            
            # Log per-turn rewards separately, format: turn{turn_idx}_{key}
            for turn_idx in range(num_turns):
                for key, value in gathered_turn_rewards[turn_idx].items():
                    if "_strict_accuracy" not in key and "_accuracy" not in key:
                        # Filter out invalid values (-10) before computing mean
                        valid_mask = value != -10
                        if valid_mask.sum() > 0:
                            wandb_log_dict[f"turn{turn_idx}_reward_{key}"] = value[valid_mask].mean()
            
            # PE fallback ratio monitoring
            if _pe_total_sample > 0:
                wandb_log_dict["sample_pe_fallback_ratio"] = _pe_fallback_total_sample / _pe_total_sample
            # early stop ratio monitoring (samples in turn>0 marked non-learning due to prior IF=0)
            if _filter_total_eligible > 0:
                wandb_log_dict["sample_early_stop_ratio"] = _filter_total_marked / _filter_total_eligible
            wandb.log(wandb_log_dict, step=global_step)

        # Compute advantages
        if config.per_prompt_stat_tracking:
            prompt_ids_gathered = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            gathered_prompts = tokenizer.batch_decode(prompt_ids_gathered, skip_special_tokens=True)
            advantages = stat_tracker.update(gathered_prompts, gathered_rewards["avg"])
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                gathered_prompts, gathered_rewards
            )
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (
                gathered_rewards["avg"] - gathered_rewards["avg"].mean()
            ) / (gathered_rewards["avg"].std() + 1e-4)

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(
                accelerator.num_processes, -1, advantages.shape[-1]
            )[accelerator.process_index].to(accelerator.device)
        )
        del samples["rewards"]

        # --- set advantage=0 for samples with used_to_learn=False ---
        # Save original advantage for PE GRPO loss first (PE should not be affected)
        samples["pe_advantages"] = samples["advantages"].clone()

        filter_mask = ~samples["used_to_learn"]  # True = do not learn
        n_filtered = filter_mask.sum().item()
        if n_filtered > 0:
            samples["advantages"][filter_mask] = 0.0

        total_batch_size, num_timesteps = samples["timesteps"].shape

        # ---------------------------------------------------------------
        # TRAINING (multi-turn: each sample rebuilds the full interleave context)
        # ---------------------------------------------------------------
        transformer.train()
        for inner_epoch in range(config.train.num_inner_epochs):
            mini_batch_size = max(
                total_batch_size // (config.sample.num_batches_per_epoch * num_turns), 1
            )

            samples_batched = {
                k: v.reshape(-1, mini_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # BAGEL norm convention: norm layers have training=False
            transformer.train()
            transformer.module.training = False
            transformer.module.model.training = False
            if config.use_lora:
                transformer.module.model.model.training = False
                for layer in transformer.module.model.model.layers:
                    layer.module.training = False
                    layer.module.self_attn.training = False
            else:
                for layer in transformer.module.model.layers:
                    layer.module.training = False
                    layer.module.self_attn.training = False

            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                sample["dtimesteps"] = torch.cat(
                    [
                        sample["timesteps"][:, :-1] - sample["timesteps"][:, 1:],
                        sample["timesteps"][:, -1].unsqueeze(1),
                    ],
                    dim=1,
                )
                bs = sample["timesteps"].shape[0]

                for j in tqdm(
                    range(bs),
                    desc="Batch",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    cur_sample = {k: v[j] for k, v in sample.items()}
                    # filtered sample still runs forward+backward for DDP sync, but advantage=0
                    if not cur_sample["used_to_learn"].item():
                        pass
                    turn_idx_j = int(cur_sample["turn_idx_stored"].item())  # 0 means the sample is the first edit; similarly for other values

                    # Decode per-turn instructions for this sample (PE-enhanced version, for KV cache reconstruction)
                    per_turn_ids_j = cur_sample["per_turn_prompt_ids"]  # (num_turns, 256)
                    all_turn_instrs_j = [
                        tokenizer.decode(per_turn_ids_j[t], skip_special_tokens=True)
                        for t in range(turn_idx_j + 1)
                    ]

                    # Reconstruct history images from stored tensors (PIL format)
                    hist_imgs_j = [
                        tensor_to_pil(cur_sample["ref_images_history"][t])
                        for t in range(turn_idx_j + 1)
                    ]

                    # Build the full interleave input list for this turn
                    input_list_j = _build_interleave_list(hist_imgs_j, all_turn_instrs_j) # [img0, p0, img1, p1, ..., imgT, pT]

                    # Build PE GRPO data (only when turn > 0 and PE data is available)
                    pe_data_j = None
                    if use_ic_cot_grpo_loss and use_ic_cot and turn_idx_j > 0 and int(cur_sample["pe_token_lens"].item()) > 0:
                        # Decode history instructions from original prompt ids (pre-PE version) for PE question reconstruction
                        orig_ids_j = cur_sample["orig_per_turn_prompt_ids"]  # (num_turns, 256)
                        pe_history_prompts = [
                            tokenizer.decode(orig_ids_j[t], skip_special_tokens=True)
                            for t in range(turn_idx_j)
                        ]
                        pe_current_prompt = tokenizer.decode(orig_ids_j[turn_idx_j], skip_special_tokens=True)
                        pe_data_j = {
                            "pe_token_ids": cur_sample["pe_token_ids"],
                            "pe_log_probs": cur_sample["pe_log_probs"],
                            "pe_token_len": int(cur_sample["pe_token_lens"].item()),
                            "pe_temperature": pe_temperature,
                            "pe_history_prompts": pe_history_prompts,
                            "pe_current_prompt": pe_current_prompt,
                            "advantages": cur_sample["pe_advantages"],
                            "contribute_loss": use_ic_cot_grpo_loss,
                        }

                    with autocast():
                        output_list = inferencer.interleave_inference(
                            input_list_j,
                            noise_level=config.sample.noise_level,
                            learn=True,
                            sample=cur_sample,
                            grpo_config=config,
                            accelerator=accelerator,
                            optimizer=optimizer,
                            transformer=transformer,
                            num_timesteps=config.sample.num_steps,
                            cfg_text_scale=config.sample.guidance_scale,
                            pe_data=pe_data_j,
                            **inference_hyper,
                        )
                    output_dict = output_list[0]

                    info["clipfrac"].append(output_dict["clipfrac"])
                    info["clipfrac_gt_one"].append(output_dict["clipfrac_gt_one"])
                    info["clipfrac_lt_one"].append(output_dict["clipfrac_lt_one"])
                    info["policy_loss"].append(output_dict["policy_loss"])
                    info["kl_loss"].append(output_dict["kl_loss"])
                    info["loss"].append(output_dict["loss"])
                    if "pe_loss" in output_dict:
                        info["pe_loss"].append(output_dict["pe_loss"])
                    if "pe_clipfrac" in output_dict and output_dict["pe_clipfrac"] >= 0:
                        info["pe_clipfrac"].append(output_dict["pe_clipfrac"])
                        info["pe_clipfrac_gt_one"].append(output_dict["pe_clipfrac_gt_one"])
                        info["pe_clipfrac_lt_one"].append(output_dict["pe_clipfrac_lt_one"])
                        info["pe_ratio_mean"].append(output_dict["pe_ratio_mean"])

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

                    if config.train.ema:
                        ema.step(transformer_trainable_parameters, global_step)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
