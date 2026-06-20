"""Shared helpers used across SCAIL2 node modules."""
from __future__ import annotations

import os
import logging

import torch
import torch.nn.functional as F

try:
    import folder_paths  # ComfyUI provides this
except ImportError:  # pragma: no cover - allows static checks outside ComfyUI
    folder_paths = None


log = logging.getLogger("ComfyUI-SCAIL2")

# Resolve filesystem layout once. ComfyUI puts these on import.
if folder_paths is not None:
    MODELS_DIR = folder_paths.models_dir
else:
    MODELS_DIR = os.environ.get("COMFY_MODELS_DIR", "./models")

SCAIL2_SUBDIR = os.path.join(MODELS_DIR, "scail2")
TOKENIZERS_SUBDIR = os.path.join(SCAIL2_SUBDIR, "tokenizers")


def ensure_dirs():
    """Create the scail2 subdirs so first-run users see where to drop tokenizers."""
    os.makedirs(SCAIL2_SUBDIR, exist_ok=True)
    os.makedirs(TOKENIZERS_SUBDIR, exist_ok=True)


def list_files(category: str, exts=(".safetensors", ".pth", ".pt", ".bin")) -> list[str]:
    """List files in a ComfyUI folder_paths category, filtered by extension."""
    if folder_paths is None:
        return []
    try:
        names = folder_paths.get_filename_list(category)
    except Exception:
        return []
    return [n for n in names if n.lower().endswith(exts)]


def resolve_model_path(category: str, name: str) -> str:
    if folder_paths is None:
        return name
    return folder_paths.get_full_path(category, name)


def list_tokenizer_dirs() -> list[str]:
    """List subdirectories under models/scail2/tokenizers/ as candidate HF tokenizer dirs."""
    if not os.path.isdir(TOKENIZERS_SUBDIR):
        return []
    return sorted(
        d for d in os.listdir(TOKENIZERS_SUBDIR)
        if os.path.isdir(os.path.join(TOKENIZERS_SUBDIR, d))
    )


def resolve_tokenizer_dir(name: str) -> str:
    """Resolve a tokenizer name to an absolute path. Accepts absolute paths verbatim."""
    if os.path.isabs(name):
        return name
    return os.path.join(TOKENIZERS_SUBDIR, name)


# ---------------------------------------------------------------------------
# Tensor conversions between ComfyUI IMAGE format and the SCAIL pipeline format
# ---------------------------------------------------------------------------
# ComfyUI IMAGE: (B, H, W, C), float32, range [0, 1]
# SCAIL pipeline single-image: (C, H, W), float32, range [-1, 1]
# SCAIL pipeline video frames:  (T, C, H, W), range [-1, 1]
# SCAIL pipeline mask video:    (3, T, H, W), range [-1, 1]
# ---------------------------------------------------------------------------

def image_to_chw_minus1_1(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI single IMAGE (B, H, W, C) with B==1 -> (3, H, W) in [-1, 1]."""
    if image.dim() != 4 or image.shape[0] < 1:
        raise ValueError(f"Expected IMAGE of shape (B,H,W,C); got {tuple(image.shape)}")
    if image.shape[0] > 1:
        log.warning("Image batch has %d entries; using only first.", image.shape[0])
    x = image[0]                       # H, W, C
    x = x.permute(2, 0, 1).contiguous()  # C, H, W
    x = x.float().mul_(2.0).sub_(1.0)
    return x


def image_batch_to_tchw_minus1_1(images: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE batch (T, H, W, C) -> (T, C, H, W) in [-1, 1]."""
    if images.dim() != 4:
        raise ValueError(f"Expected IMAGE batch of shape (T,H,W,C); got {tuple(images.shape)}")
    x = images.permute(0, 3, 1, 2).contiguous()
    x = x.float().mul_(2.0).sub_(1.0)
    return x


def image_batch_to_3thw_minus1_1(images: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE batch (T, H, W, C) -> (3, T, H, W) in [-1, 1] for mask videos.

    If the input has 1 channel (MASK-style passed as IMAGE), broadcasts to 3.
    """
    if images.dim() != 4:
        raise ValueError(f"Expected IMAGE batch of shape (T,H,W,C); got {tuple(images.shape)}")
    x = images
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, -1, 3)
    elif x.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels in last dim; got {x.shape[-1]}")
    x = x.permute(3, 0, 1, 2).contiguous()  # 3, T, H, W
    x = x.float().mul_(2.0).sub_(1.0)
    return x


def video_chw_to_image_batch(video: torch.Tensor) -> torch.Tensor:
    """SCAIL pipeline output (C, T, H, W) in [-1, 1] -> ComfyUI IMAGE (T, H, W, C) in [0, 1]."""
    if video.dim() != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected (3, T, H, W); got {tuple(video.shape)}")
    x = video.permute(1, 2, 3, 0).contiguous()   # T, H, W, C
    x = x.float().add_(1.0).div_(2.0).clamp_(0.0, 1.0)
    return x


def round_to_multiple(x: int, m: int) -> int:
    return int(round(x / m)) * m


def validate_segment_len(segment_len: int, vae_temporal_stride: int = 4) -> None:
    """SCAIL-2 requires (segment_len - 1) divisible by VAE temporal stride."""
    if segment_len < vae_temporal_stride + 1:
        raise ValueError(f"segment_len must be >= {vae_temporal_stride + 1}")
    if (segment_len - 1) % vae_temporal_stride != 0:
        raise ValueError(
            f"(segment_len - 1) must be divisible by {vae_temporal_stride}; "
            f"got segment_len={segment_len}. Try 81, 77, 73, 69, ... or 85, 89, 93, ..."
        )


def pad_frames_to_segments(
    frames: torch.Tensor,
    segment_len: int,
    segment_overlap: int,
    mode: str = "repeat_last",
) -> torch.Tensor:
    """Pad a (T, ...) tensor so the upstream segment builder covers every frame.

    Upstream `build_segments` silently drops the trailing frames that don't fit a
    full segment. We pad the tail with the last frame (or mirror) so all frames
    are processed; the caller is responsible for truncating decoded output back
    to the original length.
    """
    if mode not in ("repeat_last", "mirror"):
        raise ValueError(f"Unknown pad mode {mode}")
    total = frames.shape[0]
    if total <= segment_len:
        # single segment, upstream code handles tail
        return frames

    stride = segment_len - segment_overlap
    # number of segments upstream would produce
    n_full = 1 + max(0, (total - segment_len) // stride)
    last_end = segment_len + (n_full - 1) * stride
    if last_end >= total:
        return frames

    # we need at least one more segment; pad to reach (n_full+1)-th segment end
    needed_end = segment_len + n_full * stride
    pad = needed_end - total
    if mode == "repeat_last":
        tail = frames[-1:].expand(pad, *frames.shape[1:])
    else:  # mirror
        # mirror the last `pad` frames
        take = min(pad, total)
        mirror = frames[-take:].flip(0)
        if pad > take:
            extra = frames[:1].expand(pad - take, *frames.shape[1:])
            tail = torch.cat([mirror, extra], dim=0)
        else:
            tail = mirror
    log.info("Padded driving tail by %d frame(s) to keep all frames in coverage.", pad)
    return torch.cat([frames, tail], dim=0)


def trim_to_length(video: torch.Tensor, length: int) -> torch.Tensor:
    """Trim the temporal dim of a (C, T, H, W) tensor back to `length`."""
    if video.shape[1] <= length:
        return video
    return video[:, :length].contiguous()
