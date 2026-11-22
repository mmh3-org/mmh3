# Models

mmh3 finds each checkpoint under its ComfyUI name in a models directory laid out like ComfyUI's
`models` folder, so a ComfyUI installation's `models` directory works as is. `mmh3` takes the
directory with `--models DIR` or `MMH3_MODELS`, and single files with `--dit`, `--text-encoder`,
`--video-vae` and `--audio-vae`.

`make download-models` runs `tools/download-models.sh`, which downloads these files into `models`
with the Hugging Face CLI (`hf`), or with curl when hf is not installed:

| File | From |
| --- | --- |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | yniw/MiniMax-H3-mmh3 |
| `patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors` | yniw/MiniMax-H3-mmh3 |

With `--video-vae fp16`, the script downloads the FP16 video VAE,
`vae/minimax_h3_video_vae_fp16.safetensors` from Comfy-Org/MiniMax-H3, instead of the INT8 one.
`--ref2va` also downloads the ref2va DiT and its 8-step Turbo LoRA for [reference
images](ref2va.md), `--no-fasth3` skips the [FastH3](fasth3.md) patch, `--lightx2v-turbo` also
downloads the [Turbo LoRA](lightx2v-turbo.md),
`loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` from
lightx2v/Minimax-h3-Turbo, `--taomate` also downloads the [TaoMate-H3](taomate.md) LoRA,
`loras/minimax_h3_taomate_3step_lora_rank128_bf16.safetensors` from yniw/MiniMax-H3-mmh3, `--pdmd`
also downloads the [PDMD](pdmd.md) LoRA,
`loras/minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors` from yniw/MiniMax-H3-mmh3, `--dmad`
also downloads the [DMAD](dmad.md) LoRA,
`loras/minimax_h3_dmad_4step_full_critic_rank128_bf16.safetensors` from yniw/MiniMax-H3-mmh3,
`--veda` also downloads the predictor of [Veda](veda.md)'s sparse attention,
`veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` from
Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview, `--latent-upscaler` also downloads the
latent upscaler of [draft steps](draft-steps.md),
`latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` from
LBH-123-AI/Minimax_h3_latent_Upscaler, `--image` also downloads the DiT and the image VAE of
[Image](image.md), `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` and
`vae/minimax_h3_image_vae_fp16.safetensors` from Comfy-Org/MiniMax-H3-Image, and `--models DIR`
downloads into another directory.
`make download-models LORA=pdmd`, `LORA=dmad`, `LORA=turbo` or `LORA=taomate` passes `--pdmd`,
`--dmad`, `--lightx2v-turbo` or `--taomate` to the script.

### pi-H3

`--pih3 2|4` and `--pih3-dwd 2|4` select the video VAE and DWD variants. The downloader
places the converted patch in `patches`. [yniw/MiniMax-H3-mmh3](https://huggingface.co/yniw/MiniMax-H3-mmh3)

| Variant | Patch |
| --- | --- |
| Video VAE | `minimax_h3_pih3_2step.safetensors` or `minimax_h3_pih3_4step.safetensors` |
| DWD | `minimax_h3_pih3_2step_dwd.safetensors` or `minimax_h3_pih3_4step_dwd.safetensors` |

Each video VAE patch takes about 0.5 GB and each DWD patch takes about 2.2 GB.

`make download-models LORA=pih3` and `make download-models LORA=pih3-dwd` default to two steps.
Set `PIH3_STEPS=4` to select the 4-step adapter. Both presets skip the FastH3 patch. The DWD preset
also sets `--video-vae none` for text-to-video. To add first- or last-frame conditioning, download
the video VAE with `tools/download-models.sh --models DIR --no-fasth3`. See [pi-H3](pih3.md).

### Model lookup

Patches go in `patches` of the models directory. `--patch`, `--dwd` and `--lora` each accept a path
or a file name. mmh3 looks up file names in `patches` and then `loras` for `--patch` and `--dwd`.
For `--lora`, it looks up the file name in `loras`.

`generate` decodes with the INT8 video VAE when the models directory has it, and with the FP16
one otherwise. The INT8 one is faster, and on FastH3 videos its pixels are about 61 dB PSNR from the
FP16 decode, closer than rounding them to 8 bits. `tools/models/video_vae_int8.py` builds it.
