"""Encoding nodes — text prompts (T5) and colored masks.

The mask node passes through raw normalized images; binarization to the 28-channel
latent happens inside the sampler at the *target* resolution. This avoids a
resolution mismatch when the ref_mask is uploaded at a different size than the
sampler's target_height/target_width.
"""
from __future__ import annotations

import torch

from ._shared import (
    log,
    image_to_chw_minus1_1,
    image_batch_to_3thw_minus1_1,
)


class SCAIL2EncodeText:
    """Encode positive + negative prompts through umt5-xxl."""

    CATEGORY = "SCAIL2/encoding"
    RETURN_TYPES = ("SCAIL2_TEXT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "encode"

    _WAN_NEG = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t5": ("SCAIL2_T5",),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "offload_after_encode": ("BOOLEAN", {"default": True,
                    "tooltip": "Move T5 back to CPU after encoding to free GPU memory."}),
            },
            "optional": {
                "use_wan_default_negative": ("BOOLEAN", {"default": False,
                    "tooltip": "Use the long Wan default negative prompt instead of `negative`."}),
            }
        }

    def encode(self, t5, positive: str, negative: str, offload_after_encode: bool,
               use_wan_default_negative: bool = False):
        encoder = t5["t5"]

        if use_wan_default_negative:
            negative = self._WAN_NEG

        if torch.cuda.is_available():
            encoder.model.to("cuda")
            enc_device = torch.device("cuda")
        else:
            enc_device = torch.device("cpu")

        log.info("Encoding text. positive=%d chars, negative=%d chars", len(positive), len(negative))
        with torch.no_grad():
            pos = encoder([positive], enc_device)
            neg = encoder([negative], enc_device)

        if offload_after_encode:
            encoder.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pos = [p.cpu() for p in pos]
            neg = [n.cpu() for n in neg]

        return ({
            "positive": pos,
            "negative": neg,
            "positive_text": positive,
            "negative_text": negative,
        },)


class SCAIL2EncodeMasks:
    """Bundle ref + driving (+ optional additional-ref) masks for the sampler.

    Stores normalized RGB tensors. Binarization to the 28-channel latent happens
    inside the sampler after resizing to target_height/target_width.

    Mask color semantics (from upstream README):
        Black  -> background should NOT be visible
        White  -> background should be visible
        Color  -> character region <-> driving motion correspondence

    Use the SCAIL2 Debug Inputs node first to verify your mask colors are
    saturated. Anything not at the extremes (threshold 225/255) is silently
    dropped by the binarizer.
    """

    CATEGORY = "SCAIL2/encoding"
    RETURN_TYPES = ("SCAIL2_MASKS",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "bundle"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_mask": ("IMAGE", {"tooltip": "RGB mask for the reference character image"}),
                "driving_mask_frames": ("IMAGE", {"tooltip": "Per-frame RGB driving masks (same length as pose video)"}),
                "additional_spatial_downsample": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "additional_ref_masks": ("IMAGE", {"tooltip": "Optional extra-reference masks; pair with extra-reference images in the sampler"}),
            }
        }

    def bundle(self, ref_mask, driving_mask_frames, additional_spatial_downsample: int,
               additional_ref_masks=None):
        ref_chw = image_to_chw_minus1_1(ref_mask)                          # (3, H, W) [-1, 1]
        drv_3thw = image_batch_to_3thw_minus1_1(driving_mask_frames)       # (3, T, H, W) [-1, 1]

        extras_list = None
        if additional_ref_masks is not None:
            extras_list = []
            for i in range(additional_ref_masks.shape[0]):
                rm = additional_ref_masks[i:i + 1]
                extras_list.append(image_to_chw_minus1_1(rm))              # (3, H, W) each

        out = {
            "ref_mask_chw": ref_chw,
            "driving_mask_3thw": drv_3thw,
            "additional_ref_masks_chw": extras_list,
            "additional_spatial_downsample": additional_spatial_downsample,
        }
        log.info(
            "Bundled masks: ref=%s, driving=%s, extras=%s",
            tuple(ref_chw.shape), tuple(drv_3thw.shape),
            None if extras_list is None else [tuple(e.shape) for e in extras_list],
        )
        return (out,)
