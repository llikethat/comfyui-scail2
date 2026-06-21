"""ComfyUI-SCAIL2

Faithful wrapper of the SCAIL-2 release repo (https://huggingface.co/zai-org/SCAIL-2)
as ComfyUI custom nodes. Adds long-video improvements over the upstream segmented sampler.

Standard install layout:
    ComfyUI/models/diffusion_models/   <- SCAIL-2.safetensors (converted DiT)
    ComfyUI/models/vae/                <- Wan2.1_VAE.pth
    ComfyUI/models/text_encoders/      <- models_t5_umt5-xxl-enc-bf16.pth
    ComfyUI/models/clip_vision/        <- models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
    ComfyUI/models/loras/              <- optional Lightx2v LoRA .safetensors
    ComfyUI/models/scail2/tokenizers/  <- umt5-xxl/ (HF tokenizer dir)
"""
import logging
import sys

_log = logging.getLogger("ComfyUI-SCAIL2")

# Startup dependency check. The CLIP/VAE/T5 loaders don't need diffusers, but
# the DiT (SCAIL2Model) inherits from diffusers.ConfigMixin, so the sampler
# cannot run without it. Surface this at pack load time so users don't discover
# it mid-workflow.
_missing = []
for _pkg, _import_name in [
    ("diffusers", "diffusers"),
    ("einops", "einops"),
    ("easydict", "easydict"),
    ("ftfy", "ftfy"),
    ("safetensors", "safetensors"),
]:
    try:
        __import__(_import_name)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    _msg = (
        f"\n{'=' * 70}\n"
        f"ComfyUI-SCAIL2: missing required packages: {', '.join(_missing)}\n"
        f"From your ComfyUI Python environment, run:\n"
        f"  pip install {' '.join(_missing)}\n"
        f"or:\n"
        f"  pip install -r custom_nodes/ComfyUI-SCAIL2/requirements.txt\n"
        f"The nodes will register but the sampler will fail until these are installed.\n"
        f"{'=' * 70}"
    )
    print(_msg, file=sys.stderr)
    _log.warning(_msg)


from .nodes.loaders import (
    SCAIL2ModelLoader,
    SCAIL2VAELoader,
    SCAIL2T5Loader,
    SCAIL2CLIPVisionLoader,
    SCAIL2LoRALoader,
)
from .nodes.encoders import (
    SCAIL2EncodeText,
    SCAIL2EncodeMasks,
)
from .nodes.debug import SCAIL2DebugInputs
from .nodes.sampler import SCAIL2Sampler


NODE_CLASS_MAPPINGS = {
    "SCAIL2ModelLoader": SCAIL2ModelLoader,
    "SCAIL2VAELoader": SCAIL2VAELoader,
    "SCAIL2T5Loader": SCAIL2T5Loader,
    "SCAIL2CLIPVisionLoader": SCAIL2CLIPVisionLoader,
    "SCAIL2LoRALoader": SCAIL2LoRALoader,
    "SCAIL2EncodeText": SCAIL2EncodeText,
    "SCAIL2EncodeMasks": SCAIL2EncodeMasks,
    "SCAIL2DebugInputs": SCAIL2DebugInputs,
    "SCAIL2Sampler": SCAIL2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2ModelLoader": "SCAIL2 Model Loader",
    "SCAIL2VAELoader": "SCAIL2 VAE Loader",
    "SCAIL2T5Loader": "SCAIL2 T5 Text Encoder Loader",
    "SCAIL2CLIPVisionLoader": "SCAIL2 CLIP Vision Loader",
    "SCAIL2LoRALoader": "SCAIL2 LoRA Loader",
    "SCAIL2EncodeText": "SCAIL2 Encode Text",
    "SCAIL2EncodeMasks": "SCAIL2 Encode Masks (ref + driving)",
    "SCAIL2DebugInputs": "SCAIL2 Debug Inputs",
    "SCAIL2Sampler": "SCAIL2 Sampler",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
