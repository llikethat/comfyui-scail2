# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from safetensors.torch import load_file
import gc

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model_scail import SCAILModel
from .modules.model_scail2 import SCAIL2Model
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.lora import fuse_lora_with_diff_b
from .utils.scail_utils import extract_and_compress_mask_to_latent

class SCAIL2Pipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        scail_safetensors_path, 
        scail_config_path="./config.json",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        lora_path=None,
        lora_alpha=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.lora_path = lora_path
        self.lora_alpha = lora_alpha

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanSCAILModel from {scail_safetensors_path}")
        self.model = SCAIL2Model.from_config(scail_config_path)
        state_dict = load_file(scail_safetensors_path)
        self.model.load_state_dict(state_dict)
        if self.lora_path is not None:
            if self.lora_alpha is None:
                self.lora_alpha = 1.0
            self.fuse_lora(self.lora_path, self.lora_alpha)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def fuse_lora(self, lora_path, alpha=1.0):
        logging.info(f"Fusing LoRA from {lora_path}, strength = {alpha}.")
        lora_state_dict = load_file(lora_path)
        fuse_lora_with_diff_b(self.model, lora_state_dict, alpha=alpha)

    def generate(self,
                 input_prompt,
                 img,
                 ref_mask_img: torch.Tensor,
                 pose_video: torch.Tensor,
                 driving_mask_video: torch.Tensor,
                 replace_flag: bool,
                 segment_len=81,
                 segment_overlap=5,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt=None,
                 seed=-1,
                 offload_model=True,
                 additional_ref_imgs: list[torch.Tensor] = None,
                 additional_ref_mask_imgs: list[torch.Tensor] = None,
                 **kwargs):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (torch.Tensor):
                Input image tensor. Shape: [3, H, W], Range: (-1, 1)
            ref_mask_img (torch.Tensor):
                Input image mask tensor. Shape: [3, H, W], Range: (-1, 1)
            pose_video (torch.Tensor):
                Input pose video. Shape: [T, C, H, W]
            driving_mask_video (torch.Tensor):
                Input driving mask tensor. Shape: [3, T, H, W], Range: (-1, 1)
            replace_flag (bool):
                True for replacement mode, False for animation mode
            segment_len (`int`, *optional*, defaults to 81):
                Number of pixel frames sampled in each segment.
            segment_overlap (`int`, *optional*, defaults to 5):
                Number of pixel frames shared with the previous segment as clean history.
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to None):
                Negative prompt for content exclusion. If not given, use ""
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, T, H, W).
        """
        if segment_len <= 0:
            raise ValueError("segment_len must be positive")
        if segment_overlap <= 0 or segment_overlap >= segment_len:
            raise ValueError("segment_overlap must be in (0, segment_len)")

        pose_video = pose_video.to(self.device)
        driving_mask_video = driving_mask_video.to(self.device)
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device) # 3 H W
        else:
            img = img.to(self.device) # 3 H W, -1 ~ 1
        ori_img = img.unsqueeze(0).to(self.device) # 1, 3, H, W

        if not isinstance(ref_mask_img, torch.Tensor):
            ref_mask_img = TF.to_tensor(ref_mask_img).sub_(0.5).div_(0.5).to(self.device) # 3 H W
        else:
            ref_mask_img = ref_mask_img.to(self.device) # 3 H W, -1 ~ 1

        if additional_ref_imgs is not None:
            if additional_ref_mask_imgs is None:
                raise ValueError('additional_ref_mask_imgs is required when additional_ref_imgs is provided.')
            if isinstance(additional_ref_imgs, torch.Tensor):
                additional_ref_imgs = [additional_ref_imgs]
            if isinstance(additional_ref_mask_imgs, torch.Tensor):
                additional_ref_mask_imgs = [additional_ref_mask_imgs]
            if len(additional_ref_imgs) != len(additional_ref_mask_imgs):
                raise ValueError(
                    'additional_ref_imgs and additional_ref_mask_imgs must have the same length, '
                    'got %d and %d.' % (len(additional_ref_imgs), len(additional_ref_mask_imgs)))
            additional_ref_imgs = [
                TF.to_tensor(u).sub_(0.5).div_(0.5).to(self.device)
                if not isinstance(u, torch.Tensor) else u.to(self.device)
                for u in additional_ref_imgs
            ]
            additional_ref_mask_imgs = [
                TF.to_tensor(u).sub_(0.5).div_(0.5).to(self.device)
                if not isinstance(u, torch.Tensor) else u.to(self.device)
                for u in additional_ref_mask_imgs
            ]
        elif additional_ref_mask_imgs is not None:
            raise ValueError('additional_ref_mask_imgs requires additional_ref_imgs.')
        num_frames = pose_video.shape[0]
        if driving_mask_video.shape[1] != num_frames:
            raise ValueError(
                f"pose_video and driving_mask_video must have the same frame count, "
                f"got {num_frames} and {driving_mask_video.shape[1]}")

        def build_segments(total_frames):
            if total_frames <= segment_len:
                keep = ((total_frames - 1) // self.vae_stride[0]) * self.vae_stride[0] + 1
                return [(0, keep)]
            segments = []
            start = 0
            stride = segment_len - segment_overlap
            while start < total_frames:
                end = start + segment_len
                if end > total_frames:
                    break
                segments.append((start, end))
                start += stride
            return segments

        segments = build_segments(num_frames)
        if len(segments) == 0:
            raise ValueError(
                f"No valid segment was produced for {num_frames} frames. "
                f"Use a longer driving video or reduce segment_len.")
        if len(segments) > 1:
            logging.info(
                f"Sampling {len(segments)} segments with segment_len={segment_len}, "
                f"segment_overlap={segment_overlap}.")

        ref_latent = self.vae.encode([rearrange(ori_img, 't c h w -> c t h w')])[0]
        
        additional_ref_latent = None
        additional_ref_mask_latent_28ch = None
        if additional_ref_imgs is not None:
            additional_ref_latents = []
            additional_ref_mask_latents = []
            for additional_ref_img, additional_ref_mask_img in zip(additional_ref_imgs, additional_ref_mask_imgs):
                ori_additional_ref_img = additional_ref_img.unsqueeze(0).to(self.device)
                additional_ref_latents.append(
                    self.vae.encode([rearrange(ori_additional_ref_img, 't c h w -> c t h w')])[0]
                )
                additional_ref_mask_latents.append(
                    extract_and_compress_mask_to_latent(
                        additional_ref_mask_img.unsqueeze(1), additional_spatial_downsample=1
                    )
                )
            additional_ref_latent = torch.cat(additional_ref_latents, dim=1)
            additional_ref_mask_latent_28ch = torch.cat(additional_ref_mask_latents, dim=1)
        ref_mask_latent_28ch = extract_and_compress_mask_to_latent(
            ref_mask_img.unsqueeze(1), additional_spatial_downsample=1
        )  # (28, 1, H_lat, W_lat)
        lat_c = ref_latent.shape[0]

        # TODO: support sequence_parallel
        max_seq_len = 1e10
        # max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        #     self.patch_size[1] * self.patch_size[2])
        # max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt is None:
            n_prompt = ""

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        def apply_clean_history(latent, history_latent):
            if history_latent is None:
                return latent
            history_t = history_latent.shape[1]
            latent[:, :history_t] = history_latent.to(device=latent.device, dtype=latent.dtype)
            return latent

        output_segments = []
        prev_history_pixel = None

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            def build_sample_scheduler():
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")
                return sample_scheduler, timesteps

            def sample_func(latent, arg_c, arg_null, history_latent):
                if offload_model:
                    self.model.to(self.device)
                latent = apply_clean_history(latent, history_latent)
                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = [apply_clean_history(latent.to(self.device), history_latent)]
                    timestep = [t]

                    timestep = torch.stack(timestep).to(self.device)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()
                    if guide_scale <= 1.0:
                        noise_pred = noise_pred_cond
                    else:
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                    latent = latent.to(
                        torch.device('cpu') if offload_model else self.device)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = apply_clean_history(temp_x0.squeeze(0), history_latent)

                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                if offload_model:
                    self.model.cpu()
                    torch.cuda.empty_cache()

                if self.rank == 0:
                    videos = self.vae.decode(x0)
                return videos

            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                logging.info(
                    f"Processing segment {seg_idx + 1}/{len(segments)}: "
                    f"frames [{seg_start}, {seg_end})")
                sample_scheduler, timesteps = build_sample_scheduler()

                pose_segment = pose_video[seg_start:seg_end]
                smpl_render_video = F.interpolate(
                    pose_segment, scale_factor=0.5, mode='bilinear', align_corners=False)
                pose_latent = self.vae.encode([rearrange(smpl_render_video, 't c h w -> c t h w')])[0]

                lat_t = pose_latent.shape[1]
                _, lat_h, lat_w = ref_latent.shape[1:]

                null_noisy_mask = torch.zeros(
                    ref_mask_latent_28ch.shape[0], lat_t, lat_h, lat_w,
                    device=self.device, dtype=ref_mask_latent_28ch.dtype)
                ref_masks = torch.cat([ref_mask_latent_28ch, null_noisy_mask], dim=1)

                driving_mask_segment = driving_mask_video[:, seg_start:seg_end]
                driving_mask_segment = F.interpolate(
                    driving_mask_segment, scale_factor=0.5, mode='bilinear', align_corners=False)
                driving_masks = extract_and_compress_mask_to_latent(
                    driving_mask_segment, additional_spatial_downsample=1
                )

                history_latent = None
                history_mask = None
                if seg_idx > 0:
                    if prev_history_pixel is None:
                        raise RuntimeError("Missing previous segment history frames.")
                    history_latent = self.vae.encode([
                        prev_history_pixel.to(self.device, dtype=self.param_dtype)
                    ])[0]
                    history_t = min(history_latent.shape[1], lat_t)
                    history_mask = torch.zeros(
                        4, lat_t, lat_h, lat_w, device=self.device, dtype=torch.float32)
                    history_mask[:, :history_t] = 1
                    logging.info(
                        f"Using {prev_history_pixel.shape[1]} clean history frames "
                        f"({history_t} latent frames).")

                noise = torch.randn(
                    lat_c,
                    lat_t,
                    lat_h,
                    lat_w,
                    dtype=torch.float32,
                    generator=seed_g,
                    device=self.device)

                arg_c = {
                    'context': [context[0]],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'ref_latents': [ref_latent],
                    'ref_masks': [ref_masks],
                    'pose_latents': [pose_latent],
                    'driving_masks': [driving_masks],
                    'history_mask': [history_mask] if history_mask is not None else None,
                    'replace_flag': replace_flag,
                    'additional_ref_latents': None if additional_ref_latent is None else [additional_ref_latent],
                    'additional_ref_masks': None if additional_ref_mask_latent_28ch is None else [additional_ref_mask_latent_28ch],
                }

                arg_null = {
                    'context': context_null,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'ref_latents': [ref_latent],
                    'ref_masks': [ref_masks],
                    'pose_latents': [pose_latent],
                    'driving_masks': [driving_masks],
                    'history_mask': [history_mask] if history_mask is not None else None,
                    'replace_flag': replace_flag,
                    'additional_ref_latents': None if additional_ref_latent is None else [additional_ref_latent],
                    'additional_ref_masks': None if additional_ref_mask_latent_28ch is None else [additional_ref_mask_latent_28ch],
                }

                if offload_model:
                    torch.cuda.empty_cache()

                videos = sample_func(noise, arg_c, arg_null, history_latent)
                segment_video = videos[0] if self.rank == 0 else None
                if self.rank == 0:
                    if seg_idx == 0:
                        output_segments.append(segment_video.cpu())
                    else:
                        output_segments.append(segment_video[:, segment_overlap:].cpu())
                    if seg_idx < len(segments) - 1:
                        prev_history_pixel = segment_video[:, -segment_overlap:].contiguous()

                del noise, pose_latent, ref_masks, driving_masks, sample_scheduler
                if history_latent is not None:
                    del history_latent, history_mask
                if offload_model:
                    torch.cuda.empty_cache()

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0:
            return torch.cat(output_segments, dim=1).to(self.device)
        return None
