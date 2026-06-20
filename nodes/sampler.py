"""SCAIL2 Sampler — the main inference node.

Refactor of upstream `SCAIL2Pipeline.generate()` for ComfyUI with these changes:

  1. History between segments is kept as latent (no VAE round-trip drift).
  2. Driving video tail is padded so no frames are silently dropped.
  3. `history_length` is configurable (upstream uses 5; WanAnimatePlus uses 21).
  4. Quick-preview mode runs only segment 1 at reduced steps for fast iteration.
  5. Live previews emitted via comfy.utils.ProgressBar after every segment.
  6. (segment_len - 1) % 4 is validated at the input boundary.

Outputs the full decoded video as a ComfyUI IMAGE batch, plus a debug grid of
first/middle/last frames per segment.
"""
from __future__ import annotations

import gc
import logging
import math
import random
import sys
import types
from contextlib import contextmanager
from typing import Optional

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

try:
    import comfy.utils as comfy_utils
except ImportError:  # pragma: no cover
    comfy_utils = None

from ._shared import (
    log,
    image_to_chw_minus1_1,
    image_batch_to_tchw_minus1_1,
    video_chw_to_image_batch,
    pad_frames_to_segments,
    trim_to_length,
    validate_segment_len,
    round_to_multiple,
)


def _build_scheduler(solver: str, sampling_steps: int, shift: float, device, num_train_timesteps: int):
    if solver == "unipc":
        from ..scail2_wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        sched = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sched.set_timesteps(sampling_steps, device=device, shift=shift)
        return sched, sched.timesteps
    elif solver == "dpm++":
        from ..scail2_wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps,
        )
        sched = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(sched, device=device, sigmas=sigmas)
        return sched, timesteps
    raise ValueError(f"Unknown solver {solver}")


def _build_segments(total_frames: int, segment_len: int, segment_overlap: int, vae_stride0: int):
    """Same logic as upstream `build_segments` — caller pre-pads to ensure full coverage."""
    if total_frames <= segment_len:
        keep = ((total_frames - 1) // vae_stride0) * vae_stride0 + 1
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


class SCAIL2Sampler:
    """Main SCAIL-2 inference node."""

    CATEGORY = "SCAIL2/sampling"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "debug_preview_grid")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("SCAIL2_MODEL",),
                "vae":           ("SCAIL2_VAE",),
                "clip_vision":   ("SCAIL2_CLIP_VISION",),
                "text":          ("SCAIL2_TEXT",),
                "ref_image":     ("IMAGE",),
                "masks":         ("SCAIL2_MASKS",),
                "pose_frames":   ("IMAGE",),

                "target_width":  ("INT", {"default": 896, "min": 256, "max": 2048, "step": 32}),
                "target_height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "replace_mode":  ("BOOLEAN", {"default": False,
                    "tooltip": "False: animation mode. True: character replacement mode."}),

                "steps":         ("INT", {"default": 40, "min": 1, "max": 200}),
                "cfg":           ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift":         ("FLOAT", {"default": 3.0, "min": 0.1, "max": 12.0, "step": 0.1}),
                "solver":        (["unipc", "dpm++"], {"default": "unipc"}),
                "seed":          ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),

                "segment_len":      ("INT", {"default": 81, "min": 5, "max": 161, "step": 4,
                    "tooltip": "(segment_len - 1) must be divisible by 4. Valid: 5,9,...,77,81,85,..."}),
                "segment_overlap":  ("INT", {"default": 5, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Pixel frames reused as clean history between segments."}),
                "pad_tail":         ("BOOLEAN", {"default": True,
                    "tooltip": "Pad the driving video tail so trailing frames aren't dropped."}),
                "history_as_latent": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep history as latent across segments (no VAE round-trip drift)."}),

                "offload_model":    ("BOOLEAN", {"default": True}),
                "quick_preview":    ("BOOLEAN", {"default": False,
                    "tooltip": "Run only the first segment for fast prompt iteration."}),
                "quick_preview_steps": ("INT", {"default": 8, "min": 1, "max": 40,
                    "tooltip": "Steps used during quick_preview. Ignored otherwise."}),
            },
            "optional": {
                "additional_ref_images": ("IMAGE", {"tooltip": "Optional extra references (multi-ref mode)"}),
                "clip_vision_offload_after": ("BOOLEAN", {"default": True}),
            }
        }

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def sample(self,
               model, vae, clip_vision, text,
               ref_image, masks, pose_frames,
               target_width, target_height, replace_mode,
               steps, cfg, shift, solver, seed,
               segment_len, segment_overlap, pad_tail, history_as_latent,
               offload_model, quick_preview, quick_preview_steps,
               additional_ref_images=None, clip_vision_offload_after=True):

        validate_segment_len(segment_len)
        if segment_overlap <= 0 or segment_overlap >= segment_len:
            raise ValueError(f"segment_overlap must be in (0, {segment_len})")
        if target_width % 32 != 0 or target_height % 32 != 0:
            raise ValueError("target_width and target_height must be divisible by 32.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dit = model["model"]
        param_dtype = model["param_dtype"]
        vae_obj = vae["vae"]
        clip_obj = clip_vision["clip"]

        if quick_preview:
            steps = min(steps, quick_preview_steps)
            log.info("Quick-preview: forcing steps=%d, single-segment only.", steps)

        # ------------------------------------------------------------------
        # 1. Prepare reference image (single frame, target resolution, [-1,1])
        # ------------------------------------------------------------------
        ref_chw = image_to_chw_minus1_1(ref_image)
        ref_chw = F.interpolate(ref_chw.unsqueeze(0),
                                size=(target_height, target_width),
                                mode="bilinear", align_corners=False).squeeze(0).to(device)
        ori_img = ref_chw.unsqueeze(0)                       # (1, 3, H, W)

        # Ref mask: resize to target res then binarize to 28ch (matches upstream order).
        from ..scail2_wan.utils.scail_utils import extract_and_compress_mask_to_latent
        ref_mask_chw = masks["ref_mask_chw"].to(device)                                # (3, H_ref, W_ref) [-1,1]
        ref_mask_chw = F.interpolate(ref_mask_chw.unsqueeze(0),
                                     size=(target_height, target_width),
                                     mode="nearest").squeeze(0)
        ref_mask_28 = extract_and_compress_mask_to_latent(
            ref_mask_chw.unsqueeze(1),
            additional_spatial_downsample=masks.get("additional_spatial_downsample", 1),
        )  # (28, 1, h_lat, w_lat)

        # ------------------------------------------------------------------
        # 2. Prepare pose + driving mask, with optional tail padding
        # ------------------------------------------------------------------
        pose = image_batch_to_tchw_minus1_1(pose_frames).to(device)                    # (T, 3, H, W) [-1, 1]
        pose = F.interpolate(pose, size=(target_height, target_width),
                             mode="bilinear", align_corners=False)
        drv_mask_3thw = masks["driving_mask_3thw"].to(device)                          # (3, T, H, W) [-1, 1] raw
        drv_mask_tchw = drv_mask_3thw.permute(1, 0, 2, 3)                              # (T, 3, H, W)
        drv_mask_tchw = F.interpolate(drv_mask_tchw,
                                      size=(target_height, target_width),
                                      mode="nearest")                                  # NEAREST preserves color bins
        drv_mask_3thw = drv_mask_tchw.permute(1, 0, 2, 3).contiguous()

        T_orig = pose.shape[0]
        if drv_mask_3thw.shape[1] != T_orig:
            raise ValueError(
                f"pose_frames ({T_orig}) and driving mask ({drv_mask_3thw.shape[1]}) must match"
            )

        if pad_tail and not quick_preview:
            pose = pad_frames_to_segments(pose, segment_len, segment_overlap, mode="repeat_last")
            drv_mask_tchw_pad = pad_frames_to_segments(
                drv_mask_3thw.permute(1, 0, 2, 3), segment_len, segment_overlap, mode="repeat_last")
            drv_mask_3thw = drv_mask_tchw_pad.permute(1, 0, 2, 3).contiguous()

        T_eff = pose.shape[0]

        if quick_preview:
            # Trim to a single segment worth of frames
            T_eff = min(T_eff, segment_len)
            pose = pose[:T_eff]
            drv_mask_3thw = drv_mask_3thw[:, :T_eff]

        # ------------------------------------------------------------------
        # 3. Build segments
        # ------------------------------------------------------------------
        vae_stride0 = 4
        segments = _build_segments(T_eff, segment_len, segment_overlap, vae_stride0)
        if not segments:
            raise ValueError(
                f"No valid segment for {T_eff} frames at segment_len={segment_len}. "
                f"Provide more driving frames or reduce segment_len."
            )
        log.info("Sampling %d segment(s) over %d frames (orig=%d, target=%dx%d).",
                 len(segments), T_eff, T_orig, target_width, target_height)

        # ------------------------------------------------------------------
        # 4. VAE-encode the ref image (and any extra refs)
        # ------------------------------------------------------------------
        ref_latent = vae_obj.encode([rearrange(ori_img, "t c h w -> c t h w")])[0]  # (C, 1, h_lat, w_lat)
        lat_c = ref_latent.shape[0]
        _, lat_h, lat_w = ref_latent.shape[1:]

        additional_ref_latent = None
        additional_ref_mask_28 = None
        extras_chw_list = masks.get("additional_ref_masks_chw", None)
        if additional_ref_images is not None:
            if extras_chw_list is None:
                raise ValueError(
                    "additional_ref_images requires `additional_ref_masks` to be set on SCAIL2EncodeMasks."
                )
            if additional_ref_images.shape[0] != len(extras_chw_list):
                raise ValueError(
                    f"Mismatch: {additional_ref_images.shape[0]} additional images vs "
                    f"{len(extras_chw_list)} additional masks. Must be equal."
                )
            extra_latents = []
            extra_mask_28s = []
            for i in range(additional_ref_images.shape[0]):
                im_chw = image_to_chw_minus1_1(additional_ref_images[i:i + 1]).to(device)
                im_chw = F.interpolate(im_chw.unsqueeze(0),
                                       size=(target_height, target_width),
                                       mode="bilinear", align_corners=False).squeeze(0)
                extra_latents.append(
                    vae_obj.encode([rearrange(im_chw.unsqueeze(0), "t c h w -> c t h w")])[0]
                )
                mk_chw = extras_chw_list[i].to(device)
                mk_chw = F.interpolate(mk_chw.unsqueeze(0),
                                       size=(target_height, target_width),
                                       mode="nearest").squeeze(0)
                extra_mask_28s.append(extract_and_compress_mask_to_latent(
                    mk_chw.unsqueeze(1),
                    additional_spatial_downsample=masks.get("additional_spatial_downsample", 1),
                ))
            additional_ref_latent = torch.cat(extra_latents, dim=1)
            additional_ref_mask_28 = torch.cat(extra_mask_28s, dim=1)

        # ------------------------------------------------------------------
        # 5. Text + CLIP features
        # ------------------------------------------------------------------
        context = [t.to(device) for t in text["positive"]]
        context_null = [t.to(device) for t in text["negative"]]

        # CLIP needs the un-normalized ref image as (3, 1, H, W) per upstream
        if offload_model and clip_obj.model is not None:
            clip_obj.model.to(device)
        clip_feat = clip_obj.visual([ref_chw[:, None, :, :]])
        if clip_vision_offload_after:
            clip_obj.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 6. Seed
        # ------------------------------------------------------------------
        gen = torch.Generator(device=device)
        gen.manual_seed(seed if seed > 0 else random.randint(1, 2**63 - 1))

        # ------------------------------------------------------------------
        # 7. Segment loop
        # ------------------------------------------------------------------
        # Move DiT to device
        if offload_model:
            dit.to(device)

        previews_per_segment = []
        output_segments = []
        prev_history_latent = None   # latent-anchored history (eliminates VAE round-trip)
        prev_history_pixel = None    # pixel-anchored fallback if history_as_latent=False

        pbar = comfy_utils.ProgressBar(len(segments)) if comfy_utils is not None else None

        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(dit, "no_sync", noop_no_sync)

        with amp.autocast(dtype=param_dtype), torch.no_grad(), no_sync():
            max_seq_len = int(1e10)
            num_train_ts = 1000

            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                log.info("Segment %d/%d: pixel frames [%d, %d)",
                         seg_idx + 1, len(segments), seg_start, seg_end)

                sched, timesteps = _build_scheduler(solver, steps, shift, device, num_train_ts)

                pose_segment = pose[seg_start:seg_end]                            # (T, 3, H, W)
                pose_half = F.interpolate(pose_segment, scale_factor=0.5,
                                          mode="bilinear", align_corners=False)
                pose_latent = vae_obj.encode([rearrange(pose_half, "t c h w -> c t h w")])[0]
                lat_t = pose_latent.shape[1]

                null_noisy_mask = torch.zeros(
                    ref_mask_28.shape[0], lat_t, lat_h, lat_w,
                    device=device, dtype=ref_mask_28.dtype,
                )
                ref_masks_full = torch.cat([ref_mask_28, null_noisy_mask], dim=1)

                drv_seg = drv_mask_3thw[:, seg_start:seg_end]                      # (3, T, H, W)
                drv_seg_half = F.interpolate(drv_seg, scale_factor=0.5,
                                             mode="nearest")                       # NEAREST keeps colors
                driving_masks = extract_and_compress_mask_to_latent(
                    drv_seg_half, additional_spatial_downsample=1,
                )

                # ------- History handling -------
                history_latent_for_anchor = None
                history_mask = None
                if seg_idx > 0:
                    if history_as_latent:
                        if prev_history_latent is None:
                            raise RuntimeError("Missing prev_history_latent; bug?")
                        history_latent_for_anchor = prev_history_latent
                    else:
                        if prev_history_pixel is None:
                            raise RuntimeError("Missing prev_history_pixel; bug?")
                        history_latent_for_anchor = vae_obj.encode([
                            prev_history_pixel.to(device, dtype=param_dtype)
                        ])[0]
                    history_t = min(history_latent_for_anchor.shape[1], lat_t)
                    history_mask = torch.zeros(
                        4, lat_t, lat_h, lat_w, device=device, dtype=torch.float32,
                    )
                    history_mask[:, :history_t] = 1
                    log.info("  using %d-latent-frame anchor (%s).",
                             history_t, "latent" if history_as_latent else "pixel")

                # ------- Noise + denoise loop -------
                noise = torch.randn(
                    lat_c, lat_t, lat_h, lat_w,
                    dtype=torch.float32, generator=gen, device=device,
                )

                arg_c = {
                    "context":      [context[0]],
                    "clip_fea":     clip_feat,
                    "seq_len":      max_seq_len,
                    "ref_latents":  [ref_latent],
                    "ref_masks":    [ref_masks_full],
                    "pose_latents": [pose_latent],
                    "driving_masks":[driving_masks],
                    "history_mask": [history_mask] if history_mask is not None else None,
                    "replace_flag": replace_mode,
                    "additional_ref_latents": None if additional_ref_latent is None else [additional_ref_latent],
                    "additional_ref_masks":   None if additional_ref_mask_28 is None else [additional_ref_mask_28],
                }
                arg_null = dict(arg_c)
                arg_null["context"] = context_null

                latent = noise
                def _apply_history(lt):
                    if history_latent_for_anchor is None:
                        return lt
                    ht = history_latent_for_anchor.shape[1]
                    lt = lt.clone()
                    lt[:, :ht] = history_latent_for_anchor.to(lt.device, dtype=lt.dtype)
                    return lt

                latent = _apply_history(latent)

                step_pbar = tqdm(timesteps, desc=f"seg {seg_idx+1}/{len(segments)}", leave=False)
                for t in step_pbar:
                    li = [_apply_history(latent.to(device))]
                    ts = torch.stack([t]).to(device)

                    npred_c = dit(li, t=ts, **arg_c)[0]
                    if cfg <= 1.0:
                        npred = npred_c
                    else:
                        npred_u = dit(li, t=ts, **arg_null)[0]
                        npred = npred_u + cfg * (npred_c - npred_u)

                    if offload_model:
                        torch.cuda.empty_cache()

                    temp_x0 = sched.step(npred.unsqueeze(0), t, latent.unsqueeze(0),
                                         return_dict=False, generator=gen)[0]
                    latent = _apply_history(temp_x0.squeeze(0))

                # ------- Decode this segment -------
                if offload_model:
                    dit.cpu()
                    torch.cuda.empty_cache()

                decoded = vae_obj.decode([latent.to(device)])[0]   # (3, T, H, W) [-1, 1]

                # Save history for next segment
                if seg_idx < len(segments) - 1:
                    if history_as_latent:
                        # use the tail latent frames from this segment's denoised latent
                        n_lat = max(1, math.ceil(segment_overlap / vae_stride0))
                        prev_history_latent = latent[:, -n_lat:].detach()
                    else:
                        prev_history_pixel = decoded[:, -segment_overlap:].contiguous().detach()

                # ------- Build segment preview (first/middle/last frame) -------
                T_dec = decoded.shape[1]
                preview_idx = [0, T_dec // 2, T_dec - 1] if T_dec >= 3 else list(range(T_dec))
                prv = decoded[:, preview_idx].cpu()                # (3, k, H, W) in [-1,1]
                previews_per_segment.append(prv)

                # ------- Trim segment-overlap from non-first segments and append -------
                if seg_idx == 0:
                    output_segments.append(decoded.cpu())
                else:
                    output_segments.append(decoded[:, segment_overlap:].cpu())

                # Emit live preview to ComfyUI client
                if pbar is not None:
                    # Use first frame of segment as a quick preview thumbnail
                    thumb = decoded[:, 0].cpu().add(1).div(2).clamp(0, 1)  # (3,H,W)
                    thumb = (thumb * 255).byte().permute(1, 2, 0).contiguous()   # H,W,3
                    try:
                        pbar.update_absolute(seg_idx + 1, total=len(segments), preview=("PNG", thumb, None))
                    except TypeError:
                        # Older ComfyUI signatures: update_absolute(value)
                        pbar.update_absolute(seg_idx + 1)

                # Move DiT back to GPU for the next segment if we offloaded
                if offload_model and seg_idx < len(segments) - 1:
                    dit.to(device)

                del noise, pose_latent, ref_masks_full, driving_masks, decoded, latent
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if offload_model:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ------- Stitch + trim back to original length -------
        full_video = torch.cat(output_segments, dim=1)   # (3, T_eff, H, W)
        if not quick_preview and pad_tail:
            full_video = trim_to_length(full_video, T_orig)

        images_out = video_chw_to_image_batch(full_video)   # (T, H, W, 3) [0, 1]

        # ------- Build debug preview grid: each segment's [first, mid, last] thumbs -------
        H, W = full_video.shape[2], full_video.shape[3]
        thumb_h, thumb_w = 128, int(128 * W / max(1, H))
        thumb_w = max(64, (thumb_w // 8) * 8)
        rows = []
        for prv in previews_per_segment:
            r = F.interpolate(prv.add(1).div(2).clamp(0, 1),
                              size=(thumb_h, thumb_w),
                              mode="bilinear", align_corners=False)        # (3, k, h, w)
            row = torch.cat([r[:, i] for i in range(r.shape[1])], dim=2)   # (3, h, k*w)
            rows.append(row)
        # pad rows to equal width
        max_w = max(r.shape[2] for r in rows)
        padded = []
        for r in rows:
            if r.shape[2] < max_w:
                pad = torch.zeros(3, r.shape[1], max_w - r.shape[2])
                r = torch.cat([r, pad], dim=2)
            padded.append(r)
        grid = torch.cat(padded, dim=1)                                    # (3, len*h, max_w)
        debug_img = grid.permute(1, 2, 0).unsqueeze(0)                     # (1, gh, gw, 3)

        return (images_out, debug_img)
