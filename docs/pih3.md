# pi-H3

[pi-H3](https://github.com/Lakonik/LakonLab/blob/2ee2b334374ec91b7cee371a85a524bf4ba644d3/docs/piH3.md)
distills MiniMax H3 into two- and four-step students. Both variants use the pruned FL2VA DiT and
the audio VAE decoder. The DWD variant produces RGB with a pixel refiner that reads the final
DiT features.

## Download and generate

Use `LORA=pih3` for the video VAE or `LORA=pih3-dwd` for DWD on one local CUDA or Metal device. Both
default to two steps. Set `PIH3_STEPS=4` on both commands to select the matching 4-step adapter.

```sh
make download-models LORA=pih3-dwd PIH3_STEPS=4
make generate LORA=pih3-dwd PIH3_STEPS=4
```

The download fetches the converted patch directly into `models/patches`.
[Distributed patches](https://huggingface.co/yniw/MiniMax-H3-mmh3) · [Conversion instructions](../tools/models/README.md#pih3_patchpy)

## Video VAE output

The Make presets select the backend and its attention precision. For a direct CLI invocation on
Metal, omit `--attention-precision int8-fp8` in the examples below to use the Mac's default.

```sh
tools/download-models.sh --no-fasth3 --pih3 2
target/release/mmh3 generate --prompt-file prompt.txt --out pih3.mp4 \
  --width 1344 --height 768 --frames 124 --seed 1 \
  --schedule pih3 --steps 2 --attention sol --attention-precision int8-fp8 --sparse-start 0 \
  --patch minimax_h3_pih3_2step.safetensors
```

The schedule uses the published raw sigma maximum of 0.93 for two steps and 0.96 for four, with
a final interval scale of 0.6, the default video and audio shifts of 12 and 3, and Euler sampling.
For four steps, use `--pih3 4`, `--steps 4` and `--patch minimax_h3_pih3_4step.safetensors`.

## Speed on DGX Spark

The benchmark used the Makefile's Tokyo rain prompt and seed 1 to generate 124 frames (5.17 s)
at 1344×768. All runs used Sol-Attn from the first step at tau 1.3, INT8/FP8 attention, the INT8
video VAE and native MP4 output with audio. The totals are CLI wall times and include model
loading and encoding.

| Model and DiT linear precision | DGX Sparks | Two DiT steps | Video decode and submission | Total |
| --- | ---: | ---: | ---: | ---: |
| PDMD, INT8 | 1 | 27.6–27.8 s | 15.8–15.9 s | 51.20–51.70 s |
| pi-H3, INT8 | 1 | 27.2–27.3 s | 15.8 s | 50.94–50.96 s |
| pi-H3, NVFP4 | 1 | 22.6–22.7 s | 16.0–16.1 s | 47.23–47.82 s |
| pi-H3, INT8 | 2 | 16.2–16.6 s | 10.5–10.7 s | 35.03–35.05 s |

PDMD and pi-H3 INT8 were measured on the same Spark. The first INT8 runs on one Spark took
57.78 s (PDMD) and 54.48 s (pi-H3), including selection of additional GEMM tilings. The first
distributed run took 36.84 s. Its remote worker stayed running across all three invocations.
NVFP4 was measured on the other Spark with `--linear-precision nvfp4`.

For the distributed run, put the patch in both machines' models directories and add
`--local-worker --worker HOST:PORT` to the generation command. Both Sparks share each DiT step
and the video decode. See [Distributed generation](distributed.md) for worker setup.

## DWD

Use `--video-vae none` for text-to-video with audio. To use `--first-frame` or `--last-frame`,
pass `--video-vae int8` to download the video VAE. Both paths use the audio VAE decoder.

```sh
tools/download-models.sh --no-fasth3 --video-vae none --pih3-dwd 2
target/release/mmh3 generate --prompt-file prompt.txt --out pih3-dwd.mp4 \
  --width 1344 --height 768 --frames 124 --seed 1 \
  --schedule pih3 --steps 2 --attention sol --attention-precision int8-fp8 --sparse-start 0 \
  --dwd minimax_h3_pih3_2step_dwd.safetensors
```

`--dwd` applies the DiT patch and uses the refiner from the same file. DWD takes the final step's
target video rows through its own LayerNorm and time modulation, expands them to a grid of
16-pixel patches, and runs eight Sol blocks with alternating shifted windows. Each five-latent
block contributes 17 frames, and a two-latent tail contributes five. First- and last-frame
conditioning still uses the video VAE encoder. The audio path keeps the audio VAE decoder.

DWD runs on one local CUDA or Metal device and accepts the pi-H3 Euler schedule with FL2VA
inputs. Use `--devices` to pick a single card on a multi-GPU host. DWD can use dense or Sol
attention in the DiT. For four steps, use `--pih3-dwd 4`, `--steps 4` and
`--dwd minimax_h3_pih3_4step_dwd.safetensors`.

### DWD benchmark

These runs used the Tokyo rain prompt and seed 1 at 1344×768 and 124 frames. They used Sol-Attn at
tau 1.3 and INT8/FP8 DiT attention on one DGX Spark. The video output stage includes decoder
loading and submission to NVENC.

| Variant | Steps | DiT linear precision | Video output stage | Total |
| --- | ---: | --- | ---: | ---: |
| Video VAE | 2 | INT8 | 15.6–15.7 s | 50.43–50.96 s |
| DWD | 2 | INT8 | 3.0–3.1 s | 38.19–38.44 s |
| DWD | 2 | NVFP4 | 3.3 s | 34.28–34.49 s |
| DWD | 4 | INT8 | 3.0 s | 64.51 s |

The 2-step INT8 VAE and DWD runs were interleaved on one DGX Spark. The first DWD invocation took
43.75 s, with 6.1 s in the video output stage, while additional GEMM shapes were tuned. The NVFP4
runs used `--linear-precision nvfp4` on the other Spark, where the first invocation took 37.72 s.
The 4-step result is one run.

### Reference check

The CUDA refiner was compared with [LakonLab's FP32 refiner](https://github.com/Lakonik/LakonLab/blob/2ee2b334374ec91b7cee371a85a524bf4ba644d3/lakonlab/models/architectures/sol_h3/sol_h3.py)
on the same synthetic features and 22-frame grid, including spatial edge windows and a partial
temporal block. The pixel PSNR was 80.66 dB for FP16 refiner weights and 52.49 dB for the published
INT8 ConvRot refiner. The FP32 reference used the
[full-precision adapter](https://huggingface.co/Lakonik/pi-H3/blob/4a864589b0fac3d14768f932c930c4dcf1a17b56/pi_h3_2step_dwd_adapter/diffusion_pytorch_model.safetensors).

With LakonLab installed, generate the reference case and compare either published weight file.

```sh
python tools/golden/pih3_dwd.py --adapter /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/dwd_reference.safetensors
MMH3_DWD_REFERENCE=/path/to/dwd_reference.safetensors \
  MMH3_DWD_WEIGHTS=/path/to/pi_h3_2step_dwd_adapter_int8_convrot.safetensors \
  cargo test -p mmh3-cuda --lib vae::dwd::tests::matches_reference_pixels -- --ignored --nocapture
```

On macOS, run the same test with `-p mmh3-metal` to compare the Metal refiner against the same
reference.
