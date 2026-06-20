# ComfyUI-SCAIL2

A faithful ComfyUI wrapper for [SCAIL-2](https://github.com/zai-org/SCAIL-2) (zai-org/SCAIL-2), the character animation model that bypasses intermediate pose representations.

This node pack vendors the official `wan/` package from the SCAIL-2 release repo and exposes it as ComfyUI nodes. It is intentionally distinct from `kijai/ComfyUI-WanVideoWrapper` and `wuwukaka/ComfyUI-WanAnimatePlus`: where those packs evolve the model loading to fit a larger Wan ecosystem, this pack preserves upstream sampling math exactly so behavior stays aligned with the SCAIL-2 paper and release.

## What this pack adds beyond upstream

The upstream `SCAIL2Pipeline.generate()` has known long-video footguns. This pack fixes them in the sampler while keeping every other tensor operation identical:

1. **Latent-anchored history** — between segments, history is kept as latent instead of decoded → pixel → re-encoded. Eliminates VAE round-trip drift over long videos.
2. **Tail padding** — driving videos whose length doesn't align to `segment_len + k * (segment_len - segment_overlap)` no longer silently truncate. Trailing frames are repeat-padded for coverage and trimmed back after decoding.
3. **`segment_len` validation** — `(segment_len - 1) % 4 == 0` is enforced at the input boundary with a clear error.
4. **`quick_preview` mode** — runs only segment 1 at reduced steps for fast prompt iteration.
5. **Live per-segment previews** — each segment's first frame is streamed back to the UI as it finishes via `comfy.utils.ProgressBar`.
6. **Debug input visualization** — a separate node parses the 7-color mask binarizer the way the model does, reports per-color coverage and flags ambiguous pixels, catching mask bugs before GPU time is spent.

## Install

```
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-SCAIL2
cd ComfyUI-SCAIL2
pip install -r requirements.txt
```

`flash_attn` is recommended but not required — the vendored `attention.py` falls back to PyTorch SDPA when flash_attn is missing.

## Downloads

All files listed below have been verified against the actual HuggingFace repos. Total disk: ~32 GB for the bf16 stack, less if you swap in quantized variants.

| Component | File | Size | Destination |
|-----------|------|------|-------------|
| **DiT** (pre-converted, recommended) | [`wan2.1_14B_SCAIL_2_fp16.safetensors`](https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/wan2.1_14B_SCAIL_2_fp16.safetensors) | ~14 GB | `models/diffusion_models/` |
| **VAE** | [`Wan2.1_VAE.pth`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/Wan2.1_VAE.pth) | ~250 MB | `models/vae/` |
| **T5** umt5-xxl (bf16) | [`models_t5_umt5-xxl-enc-bf16.pth`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth) | 11.4 GB | `models/text_encoders/` |
| **T5 tokenizer** (4 files) | [`spiece.model`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/umt5-xxl/spiece.model) · [`tokenizer.json`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/umt5-xxl/tokenizer.json) · [`tokenizer_config.json`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/umt5-xxl/tokenizer_config.json) · [`special_tokens_map.json`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/umt5-xxl/special_tokens_map.json) | ~22 MB | `models/scail2/tokenizers/umt5-xxl/` |
| **CLIP vision** (visual-only) | [`models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth`](https://huggingface.co/zai-org/SCAIL-2/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth) | 2.53 GB | `models/clip_vision/` |
| **Lightx2v LoRA** (optional, step-distill) | [`Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/resolve/main/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors) | ~300 MB | `models/loras/` |

### One-shot download script

Drop this into your ComfyUI root and run it. Uses the official `huggingface_hub` CLI (`pip install -U "huggingface_hub[cli]"`).

```bash
#!/usr/bin/env bash
set -euo pipefail
COMFY_ROOT="${COMFY_ROOT:-$(pwd)}"
cd "$COMFY_ROOT"

mkdir -p models/diffusion_models models/vae models/text_encoders \
         models/clip_vision models/loras models/scail2/tokenizers/umt5-xxl

# 1. DiT (pre-converted)
hf download Comfy-Org/SCAIL-2 wan2.1_14B_SCAIL_2_fp16.safetensors \
    --local-dir models/diffusion_models

# 2. VAE + 3. T5 + 4. T5 tokenizer + 5. CLIP vision (all from zai-org/SCAIL-2)
hf download zai-org/SCAIL-2 Wan2.1_VAE.pth \
    --local-dir models/vae
hf download zai-org/SCAIL-2 umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/text_encoders --local-dir-use-symlinks False
mv models/text_encoders/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth models/text_encoders/
rmdir models/text_encoders/umt5-xxl 2>/dev/null || true

hf download zai-org/SCAIL-2 \
    umt5-xxl/spiece.model umt5-xxl/tokenizer.json \
    umt5-xxl/tokenizer_config.json umt5-xxl/special_tokens_map.json \
    --local-dir models/scail2/tokenizers
# Result: models/scail2/tokenizers/umt5-xxl/{spiece.model, ...}

hf download zai-org/SCAIL-2 models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth \
    --local-dir models/clip_vision

# 6. Lightx2v LoRA (optional)
hf download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v \
    loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors \
    --local-dir models/loras --local-dir-use-symlinks False
mv models/loras/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors models/loras/
rmdir models/loras/loras 2>/dev/null || true

echo "Done. Total downloaded: ~32 GB."
```

Resulting tree:

```
ComfyUI/models/
├── diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors
├── vae/Wan2.1_VAE.pth
├── text_encoders/models_t5_umt5-xxl-enc-bf16.pth
├── clip_vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
├── loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
└── scail2/tokenizers/umt5-xxl/{spiece.model, tokenizer.json, tokenizer_config.json, special_tokens_map.json}
```

### Lower-VRAM variants (not yet supported by this pack)

These exist if you can't fit the bf16 stack, but the loader needs additional work to handle them — flagged as roadmap, not v0.1:

- **GGUF quantized DiT** — [`realrebelai/SCAIL-2_GGUF`](https://huggingface.co/realrebelai/SCAIL-2_GGUF) (Q2_K=6 GB ... Q8_0=17.7 GB). Requires loading via [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF). Our `SCAIL2 Model Loader` does not currently route through it.
- **fp8 T5** — [`Kijai/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors`](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/umt5-xxl-enc-fp8_e4m3fn.safetensors) (~5 GB vs 11.4 GB). Our T5 loader uses `torch.load` (.pth); supporting safetensors + fp8 needs a different key-mapping path.

If you want either supported, open an issue.

### Skipping the conversion step

The official `zai-org/SCAIL-2` repo ships the DiT as an FSDP shard (`model/1/fsdp2_rank_0000_checkpoint.pt`) that requires running upstream's `convert.py` to produce a usable `.safetensors`. The `Comfy-Org/SCAIL-2` mirror linked above is exactly that conversion already done — drop it in and skip the step entirely.

## Nodes

| Node | Purpose |
|------|---------|
| `SCAIL2 Model Loader` | Load the DiT (`.safetensors` after `convert.py`). Variant `SCAIL-14B` or `SCAIL-1.3B`. |
| `SCAIL2 VAE Loader` | Wan2.1 VAE. |
| `SCAIL2 T5 Text Encoder Loader` | umt5-xxl + tokenizer. CPU-resident by default; moved to GPU for encoding. |
| `SCAIL2 CLIP Vision Loader` | XLM-RoBERTa-ViT-Huge-14 vision encoder. |
| `SCAIL2 LoRA Loader` | Optional, chainable — fuses Lightx2v or compatible LoRA into the DiT. |
| `SCAIL2 Encode Text` | T5 encode positive + negative. |
| `SCAIL2 Encode Masks (ref + driving)` | Converts RGB masks to the 28-channel binary latents the model expects. |
| `SCAIL2 Debug Inputs` | Parses the 7-color mask the way the model does, with coverage stats and an ambiguous-pixel overlay. Use this before the sampler to catch mask bugs. |
| `SCAIL2 Sampler` | Main inference. Outputs IMAGE batch + a debug preview grid of per-segment thumbs. |

## Mask color semantics (from upstream README)

The mask is **not optional**, even in single-character animation mode. Encoded as RGB at saturated extremes:

- **Black** — background here should *not* be visible
- **White** — background here *should* be visible
- **Color** (red / green / blue / yellow / magenta / cyan) — encodes the correspondence between a character region and the driving motion. Use consistent colors for the same character across ref mask, driving mask, and extra reference masks.

Anything that isn't at the saturated extreme on every channel is silently dropped by the binarizer (threshold ≈ 225/255 per channel). The `SCAIL2 Debug Inputs` node flags this. If you see `WARNING: >2% ambiguous pixels`, re-export your masks with hard thresholding before generating.

## Long-video controls

| Parameter | Default | Notes |
|-----------|---------|-------|
| `segment_len` | 81 | Pixel frames per segment. `(segment_len - 1) % 4 == 0`. |
| `segment_overlap` | 5 | History pixel frames reused between adjacent segments. |
| `pad_tail` | True | Pad trailing frames so none are dropped. Off when `quick_preview` is on. |
| `history_as_latent` | True | Keep history as latent (no VAE drift). Set False to match upstream behavior bit-for-bit. |

`segment_overlap` can be increased toward ~21 for harder motion (matches the stronger anchor `WanAnimatePlus` uses). Higher overlap = stronger continuity, slightly higher VRAM, slightly slower throughput.

## Quick preview workflow

Toggle `quick_preview = True` on the sampler. The sampler will:
1. Skip tail padding.
2. Trim the driving video to one `segment_len` segment.
3. Clamp `steps` to `quick_preview_steps` (default 8).
4. Return the decoded single segment.

Use this for prompt iteration and mask validation. Toggle off for the full long-video run.

## Known limitations

- T5 and CLIP tokenizers must be present as HuggingFace tokenizer directories under `models/scail2/tokenizers/`. There's no auto-download yet.
- The sampler is single-GPU. The upstream USP / FSDP paths are vendored but not exposed as node options.
- Multi-reference mode is supported but, as upstream notes, the model is not optimized for it. For better multi-ref results see WanAnimatePlus's `image-as-short-video` mocking trick.
- This pack does not include pose preprocessing (SCAIL-Pose). Provide preprocessed `pose` and `driving_mask_frames` from your own pipeline.

## License

Apache-2.0, matching the upstream SCAIL-2 release.
