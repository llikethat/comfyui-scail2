"""ComfyUI-SCAIL2

Faithful wrapper of the SCAIL-2 release repo (https://huggingface.co/zai-org/SCAIL-2)
as ComfyUI custom nodes. Adds long-video improvements over the upstream segmented sampler.

Standard install layout:
    ComfyUI/models/diffusion_models/   <- SCAIL-2.safetensors (converted DiT)
    ComfyUI/models/vae/                <- Wan2.1_VAE.pth
    ComfyUI/models/text_encoders/      <- models_t5_umt5-xxl-enc-bf16.pth
    ComfyUI/models/clip_vision/        <- models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
    ComfyUI/models/loras/              <- optional Lightx2v LoRA .safetensors
    ComfyUI/models/scail2/tokenizers/  <- umt5-xxl/, xlm-roberta-large/ (HF tokenizer dirs)
"""

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
