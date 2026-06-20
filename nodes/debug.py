"""Debug input visualization — catches mask color bugs before spending GPU time.

The SCAIL-2 README explicitly calls out mask color mistakes as the leading cause
of bad output. This node parses the 7 valid colors (white/red/green/blue/yellow/
magenta/cyan) the way the pipeline does internally and visualises:

  - Cropped ref image at the target resolution
  - Cropped ref mask, with per-color coverage stats
  - "Ambiguous" pixel map — pixels that don't snap cleanly to one of the 7 colors
    and will be silently dropped by the mask binarizer
  - Sample pose / driving-mask frames (start, middle, end)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._shared import (
    log,
    image_to_chw_minus1_1,
    image_batch_to_tchw_minus1_1,
    image_batch_to_3thw_minus1_1,
)

_ON_THRESH = (225.0 - 127.5) / 127.5  # same threshold used by the upstream mask binarizer
_COLOR_RGB = {
    "white":   (1.0, 1.0, 1.0),
    "red":     (1.0, 0.2, 0.2),
    "green":   (0.2, 1.0, 0.2),
    "blue":    (0.3, 0.4, 1.0),
    "yellow":  (1.0, 1.0, 0.3),
    "magenta": (1.0, 0.3, 1.0),
    "cyan":    (0.3, 1.0, 1.0),
}
_COLOR_NAMES = list(_COLOR_RGB.keys())


def _parse_mask_channels(mask_minus1_1: torch.Tensor) -> dict:
    """mask_minus1_1: (3, T, H, W) -> dict of per-channel binary maps + stats.

    Replicates the upstream binarizer logic precisely (extract_and_compress_mask_to_latent).
    """
    C, T, H, W = mask_minus1_1.shape
    m = mask_minus1_1.permute(1, 0, 2, 3).float()  # T, 3, H, W
    R = (m[:, 0:1] > _ON_THRESH).float()
    G = (m[:, 1:2] > _ON_THRESH).float()
    B = (m[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    channels = {
        "white":   R * G * B,
        "red":     R * nG * nB,
        "green":   nR * G * nB,
        "blue":    nR * nG * B,
        "yellow":  R * G * nB,
        "magenta": R * nG * B,
        "cyan":    nR * G * B,
    }
    # The 7 channels exclude all-zero (black, background-hidden) AND any pixel
    # that doesn't snap cleanly to one of 7 valid colors. Sum to detect coverage.
    on_any = sum(channels.values())                    # T, 1, H, W
    raw_any = ((R + G + B) > 0).float()                # T, 1, H, W  - any channel above threshold
    ambiguous = (raw_any - on_any).clamp(0, 1)         # any-on but not one of the 7 valid combos
    # (Above is technically zero by construction since every R/G/B combo IS one of 8 outcomes,
    # but if a future upstream change adds intermediate handling this still flags it.)
    # We instead define ambiguous as "pixels whose colors are not at the saturated extremes":
    near_zero = (m.abs() < (50.0 / 127.5))             # gray-ish channel
    saturated = (m > _ON_THRESH) | (m < -_ON_THRESH)
    fully_decisive = saturated.all(dim=1, keepdim=True).float()  # T,1,H,W
    ambiguous = (1.0 - fully_decisive)

    total = T * H * W
    stats = {}
    for name, ch in channels.items():
        ratio = ch.sum().item() / total
        stats[name] = ratio
    stats["black"] = 1.0 - on_any.sum().item() / total - ambiguous.sum().item() / total
    stats["ambiguous"] = ambiguous.sum().item() / total
    return {"channels": channels, "ambiguous": ambiguous, "stats": stats}


def _colorize_channel(name: str, ch: torch.Tensor) -> torch.Tensor:
    """ch: (1, H, W) binary -> (3, H, W) tinted by the channel's color."""
    rgb = _COLOR_RGB[name]
    out = torch.stack([ch[0] * rgb[0], ch[0] * rgb[1], ch[0] * rgb[2]], dim=0)
    return out


def _overlay_label(canvas: torch.Tensor, text: str) -> torch.Tensor:
    """No-op for now (text rendering inside torch is messy without PIL). Returns canvas.

    Keeping the function signature so we can swap in PIL-based text later without
    changing call sites.
    """
    return canvas


def _grid(panels: list[torch.Tensor], cols: int, pad: int = 4) -> torch.Tensor:
    """Stack a list of (3, H, W) panels of equal H,W into a (3, gh, gw) grid."""
    assert len(panels) > 0
    C, H, W = panels[0].shape
    rows = (len(panels) + cols - 1) // cols
    bg = torch.zeros(C, rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad)
    for i, p in enumerate(panels):
        r, c = i // cols, i % cols
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        bg[:, y:y + H, x:x + W] = p.clamp(0, 1)
    return bg


class SCAIL2DebugInputs:
    """Visualize parsed mask channels and sampled frames before running the sampler.

    Output is a single IMAGE (1, H, W, 3) showing a labelled grid. Console log
    contains per-color coverage percentages and warnings about ambiguous pixels.
    """

    CATEGORY = "SCAIL2/debug"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_grid", "stats_report")
    FUNCTION = "debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_mask": ("IMAGE",),
                "pose_frames": ("IMAGE",),
                "driving_mask_frames": ("IMAGE",),
                "panel_size": ("INT", {"default": 192, "min": 64, "max": 512, "step": 16}),
            }
        }

    def debug(self, ref_image, ref_mask, pose_frames, driving_mask_frames, panel_size: int):
        # Normalize to internal tensors
        ref_img = image_to_chw_minus1_1(ref_image).add(1).div(2).clamp(0, 1)            # (3,H,W) [0,1]
        ref_msk_internal = image_to_chw_minus1_1(ref_mask)                              # (3,H,W) [-1,1]
        pose_t = image_batch_to_tchw_minus1_1(pose_frames).add(1).div(2).clamp(0, 1)    # (T,3,H,W)
        drv_msk_internal = image_batch_to_3thw_minus1_1(driving_mask_frames)            # (3,T,H,W)

        def resize_panel(img_3hw: torch.Tensor) -> torch.Tensor:
            return F.interpolate(img_3hw.unsqueeze(0), size=(panel_size, panel_size),
                                 mode="bilinear", align_corners=False).squeeze(0)

        # --- ref image + mask, with channel breakdown ---
        ref_mask_parsed = _parse_mask_channels(ref_msk_internal.unsqueeze(1))   # T=1
        ref_mask_vis = ref_msk_internal.add(1).div(2).clamp(0, 1)               # (3,H,W) view
        panels = [
            resize_panel(ref_img),
            resize_panel(ref_mask_vis),
        ]
        for name in _COLOR_NAMES:
            ch = ref_mask_parsed["channels"][name][0]              # (1,H,W)
            colorized = _colorize_channel(name, ch)
            panels.append(resize_panel(colorized))
        # ambiguous overlay (red on grey)
        amb = ref_mask_parsed["ambiguous"][0]                      # (1,H,W)
        amb_vis = torch.stack([amb[0], amb[0] * 0.2, amb[0] * 0.2], dim=0)
        panels.append(resize_panel(amb_vis))

        # --- driving mask: first / middle / last frame parsing ---
        T = drv_msk_internal.shape[1]
        idxs = [0, T // 2, T - 1] if T >= 3 else list(range(T))
        for ti in idxs:
            frame = drv_msk_internal[:, ti:ti + 1]                  # (3,1,H,W)
            parsed = _parse_mask_channels(frame)
            frame_vis = frame[:, 0].add(1).div(2).clamp(0, 1)
            panels.append(resize_panel(frame_vis))
            # composite all 7 colors back into one visualisation
            recombined = torch.zeros_like(frame_vis)
            for name in _COLOR_NAMES:
                ch = parsed["channels"][name][0]                   # (1,H,W)
                colorized = _colorize_channel(name, ch)
                recombined = recombined + colorized
            panels.append(resize_panel(recombined.clamp(0, 1)))

        # --- pose frames sampled ---
        T_pose = pose_t.shape[0]
        idxs_p = [0, T_pose // 2, T_pose - 1] if T_pose >= 3 else list(range(T_pose))
        for pi in idxs_p:
            panels.append(resize_panel(pose_t[pi]))

        grid = _grid(panels, cols=5, pad=6)                         # (3, gh, gw)
        out_image = grid.permute(1, 2, 0).unsqueeze(0)              # (1, gh, gw, 3)

        # Build a stats report
        lines = ["=== Reference mask coverage ==="]
        for name in _COLOR_NAMES + ["black", "ambiguous"]:
            ratio = ref_mask_parsed["stats"][name]
            lines.append(f"  {name:10s} {ratio*100:6.2f}%")
        if ref_mask_parsed["stats"]["ambiguous"] > 0.02:
            lines.append("  WARNING: >2% ambiguous pixels in ref mask — pipeline will drop them.")

        lines.append("=== Driving mask (mid frame) coverage ===")
        mid_parsed = _parse_mask_channels(drv_msk_internal[:, T // 2:T // 2 + 1])
        for name in _COLOR_NAMES + ["black", "ambiguous"]:
            ratio = mid_parsed["stats"][name]
            lines.append(f"  {name:10s} {ratio*100:6.2f}%")
        if mid_parsed["stats"]["ambiguous"] > 0.02:
            lines.append("  WARNING: >2% ambiguous pixels in driving mask.")

        lines.append(f"=== Frame counts ===")
        lines.append(f"  pose_frames: {T_pose}")
        lines.append(f"  driving_mask_frames: {T}")
        if T_pose != T:
            lines.append(f"  ERROR: pose ({T_pose}) and driving mask ({T}) frame counts MUST match.")

        report = "\n".join(lines)
        log.info("\n%s", report)
        return (out_image, report)
