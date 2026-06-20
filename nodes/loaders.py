"""Loader nodes — load SCAIL-2 model components from standard ComfyUI dirs."""
from __future__ import annotations

import gc
import json
import logging
import os

import torch
from safetensors.torch import load_file

from ._shared import (
    log,
    list_files,
    resolve_model_path,
    list_tokenizer_dirs,
    resolve_tokenizer_dir,
    ensure_dirs,
)

# Path to bundled config JSONs (config-14b.json / config-1.3b.json) inside this node pack.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCES_DIR = os.path.join(_PKG_DIR, "resources")

_VARIANT_CONFIG = {
    "SCAIL-14B": "config-14b.json",
    "SCAIL-1.3B": "config-1.3b.json",
}


def _dtype_from_str(s: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[s]


class SCAIL2ModelLoader:
    """Load the SCAIL-2 DiT (.safetensors, post-convert.py)."""

    CATEGORY = "SCAIL2/loaders"
    RETURN_TYPES = ("SCAIL2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        ensure_dirs()
        models = list_files("diffusion_models", exts=(".safetensors",))
        return {
            "required": {
                "model_name": (models if models else ["<place SCAIL-2.safetensors in models/diffusion_models/>"],),
                "variant": (list(_VARIANT_CONFIG.keys()), {"default": "SCAIL-14B"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "load_device": (["cpu", "cuda"], {"default": "cpu", "tooltip": "Initial load device. The sampler moves it to GPU when needed."}),
            }
        }

    def load(self, model_name: str, variant: str, precision: str, load_device: str):
        # Lazy import — model_scail2 imports flash_attn which we patched, but still
        # heavy. Keep node import time light by deferring this until load.
        from ..scail2_wan.modules.model_scail2 import SCAIL2Model

        config_path = os.path.join(_RESOURCES_DIR, _VARIANT_CONFIG[variant])
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Bundled config missing: {config_path}")

        model_path = resolve_model_path("diffusion_models", model_name)
        if model_path is None or not os.path.isfile(model_path):
            raise FileNotFoundError(f"SCAIL-2 model not found: {model_name}")

        log.info("Loading SCAIL-2 DiT from %s (variant=%s, precision=%s)", model_path, variant, precision)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        log.info("DiT config: dim=%d layers=%d heads=%d in_dim=%d mask_dim=%d",
                 cfg_dict["dim"], cfg_dict["num_layers"], cfg_dict["num_heads"],
                 cfg_dict["in_dim"], cfg_dict["mask_dim"])

        model = SCAIL2Model.from_config(config_path)
        state_dict = load_file(model_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("State dict missing keys: %d (showing first 5): %s", len(missing), missing[:5])
        if unexpected:
            log.warning("State dict unexpected keys: %d (showing first 5): %s", len(unexpected), unexpected[:5])

        model.eval().requires_grad_(False)
        dtype = _dtype_from_str(precision)
        model.to(dtype)
        if load_device == "cuda" and torch.cuda.is_available():
            model.to("cuda")

        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bundle = {
            "model": model,
            "variant": variant,
            "config": cfg_dict,
            "param_dtype": dtype,
            "lora_applied": [],
        }
        return (bundle,)


class SCAIL2VAELoader:
    """Load Wan2.1 VAE (.pth)."""

    CATEGORY = "SCAIL2/loaders"
    RETURN_TYPES = ("SCAIL2_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        vaes = list_files("vae", exts=(".pth", ".safetensors"))
        return {
            "required": {
                "vae_name": (vaes if vaes else ["<place Wan2.1_VAE.pth in models/vae/>"],),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    def load(self, vae_name: str, device: str):
        from ..scail2_wan.modules.vae import WanVAE

        vae_path = resolve_model_path("vae", vae_name)
        if vae_path is None or not os.path.isfile(vae_path):
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        log.info("Loading Wan VAE from %s", vae_path)
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        vae = WanVAE(vae_pth=vae_path, device=dev)
        return ({"vae": vae, "device": dev},)


class SCAIL2T5Loader:
    """Load the umt5-xxl T5 text encoder."""

    CATEGORY = "SCAIL2/loaders"
    RETURN_TYPES = ("SCAIL2_T5",)
    RETURN_NAMES = ("t5",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        ensure_dirs()
        t5s = list_files("text_encoders", exts=(".pth", ".bin", ".safetensors"))
        tokenizers = list_tokenizer_dirs() or ["umt5-xxl"]
        return {
            "required": {
                "t5_name": (t5s if t5s else ["<place models_t5_umt5-xxl-enc-bf16.pth in models/text_encoders/>"],),
                "tokenizer": (tokenizers, {"tooltip": "Subdir under models/scail2/tokenizers/, or absolute path"}),
                "device": (["cpu", "cuda"], {"default": "cpu", "tooltip": "T5 is ~10GB. Keep on CPU and offload, or pre-load on GPU."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    def load(self, t5_name: str, tokenizer: str, device: str, precision: str):
        from ..scail2_wan.modules.t5 import T5EncoderModel

        ckpt = resolve_model_path("text_encoders", t5_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"T5 checkpoint not found: {t5_name}")
        tok_dir = resolve_tokenizer_dir(tokenizer)
        if not os.path.isdir(tok_dir):
            raise FileNotFoundError(
                f"Tokenizer directory not found: {tok_dir}\n"
                f"Place the umt5-xxl HF tokenizer dir under models/scail2/tokenizers/."
            )

        dtype = _dtype_from_str(precision)
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        log.info("Loading umt5-xxl T5 from %s (tokenizer=%s, device=%s)", ckpt, tok_dir, dev)
        t5 = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=dev,
            checkpoint_path=ckpt,
            tokenizer_path=tok_dir,
            shard_fn=None,
        )
        return ({"t5": t5, "device": dev, "dtype": dtype},)


class SCAIL2CLIPVisionLoader:
    """Load the open_clip XLM-RoBERTa-ViT-Huge-14 vision encoder."""

    CATEGORY = "SCAIL2/loaders"
    RETURN_TYPES = ("SCAIL2_CLIP_VISION",)
    RETURN_NAMES = ("clip_vision",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        ensure_dirs()
        clips = list_files("clip_vision", exts=(".pth", ".safetensors", ".bin"))
        tokenizers = ["(unused)"] + list_tokenizer_dirs()
        return {
            "required": {
                "clip_name": (clips if clips else ["<place CLIP vision in models/clip_vision/>"],),
                "tokenizer": (tokenizers, {"default": "(unused)",
                    "tooltip": "CLIPModel does not use the tokenizer — leave as '(unused)'. Field kept for forward compatibility."}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }

    def load(self, clip_name: str, tokenizer: str, precision: str):
        from ..scail2_wan.modules.clip import CLIPModel

        ckpt = resolve_model_path("clip_vision", clip_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"CLIP vision not found: {clip_name}")
        # CLIPModel stores tokenizer_path but never reads it — pass any string.
        tok_dir = "" if tokenizer == "(unused)" else resolve_tokenizer_dir(tokenizer)

        dtype = _dtype_from_str(precision)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLIP vision from %s", ckpt)
        clip = CLIPModel(
            dtype=dtype,
            device=dev,
            checkpoint_path=ckpt,
            tokenizer_path=tok_dir,
        )
        return ({"clip": clip, "device": dev, "dtype": dtype},)


class SCAIL2LoRALoader:
    """Fuse a Lightx2v (or compatible) LoRA into a loaded SCAIL-2 DiT.

    Chainable: feed the output back into another SCAIL2LoRALoader to stack LoRAs.
    """

    CATEGORY = "SCAIL2/loaders"
    RETURN_TYPES = ("SCAIL2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "fuse"

    @classmethod
    def INPUT_TYPES(cls):
        loras = list_files("loras", exts=(".safetensors",))
        return {
            "required": {
                "model": ("SCAIL2_MODEL",),
                "lora_name": (loras if loras else ["<place LoRA in models/loras/>"],),
                "alpha": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            }
        }

    def fuse(self, model, lora_name: str, alpha: float):
        from ..scail2_wan.utils.lora import fuse_lora_with_diff_b

        lora_path = resolve_model_path("loras", lora_name)
        if lora_path is None or not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA not found: {lora_name}")
        log.info("Fusing LoRA %s with alpha=%.3f", lora_path, alpha)
        lora_state = load_file(lora_path)
        fuse_lora_with_diff_b(model["model"], lora_state, alpha=alpha)
        # Return a shallow-copied bundle so downstream caches treat it as a new object.
        out = dict(model)
        out["lora_applied"] = list(model.get("lora_applied", [])) + [
            {"name": lora_name, "alpha": alpha}
        ]
        del lora_state
        gc.collect()
        return (out,)
