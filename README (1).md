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

## Model files (standard ComfyUI layout)

```
ComfyUI/models/
├── diffusion_models/
│   └── SCAIL-2.safetensors                                           # converted DiT
├── vae/
│   └── Wan2.1_VAE.pth
├── text_encoders/
│   └── models_t5_umt5-xxl-enc-bf16.pth
├── clip_vision/
│   └── models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
├── loras/
│   └── lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors   # optional
└── scail2/
    └── tokenizers/
        ├── umt5-xxl/             # HF tokenizer directory (spiece.model + config files)
        └── xlm-roberta-large/    # HF tokenizer directory
```

Download from `hf download zai-org/SCAIL-2`, then run `convert.py` from the upstream repo to produce `SCAIL-2.safetensors`. The tokenizer subdirectories are inside the same HuggingFace download — copy them to `models/scail2/tokenizers/`.

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
