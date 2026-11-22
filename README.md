# mmh3

mmh3 is a standalone Rust inference engine for generating video with audio and generating or
editing images. It includes the tokenizer, Qwen3-VL text encoder, diffusion transformer, samplers
and VAEs. [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) ·
[MiniMax H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)

Install the prerequisites for your backend, then run the examples from the repository root.
[CUDA on Linux and Windows (WSL)](docs/cuda.md) · [Metal on macOS](docs/metal.md)

The benchmarks below use one device and include model loading and file output.

## Generate video

```sh
make download-models
make generate
```

This downloads the models into `models`, builds for your backend and writes `out.mp4` with audio
from the sample prompt. The default Make preset uses four FastH3 steps. Set `PROMPT`, `SEED` and
`OUT` to change the prompt, seed and output path. [Video usage](docs/usage.md) ·
[FastH3](docs/fasth3.md)

| Backend | Device | Default clip | Total |
| --- | --- | --- | ---: |
| CUDA | DGX Spark | 1344×768, 124 frames | 80 s |
| Metal | M4 Max | 672×384, 73 frames | 153 s |
| Metal | M6 | 672×384, 73 frames | 67 s |

### Faster video

Use the two-step pi-H3 DWD preset. Its pixel refiner produces video frames directly from the
final DiT features. On CUDA with NVFP4 support, add
`OUTPUT_OPTIONS="--linear-precision nvfp4"` to the generation command for the fastest CUDA result.
For the two-step PDMD preset, use `LORA=pdmd` on both commands. [pi-H3](docs/pih3.md) ·
[PDMD](docs/pdmd.md)

```sh
make download-models LORA=pih3-dwd
make generate LORA=pih3-dwd
```

The faster presets use the same clip sizes as the default runs.

| Backend | Device | Preset | Total |
| --- | --- | --- | ---: |
| CUDA | DGX Spark | 2-step pi-H3 DWD with NVFP4 | 34.3–34.5 s |
| Metal | M6 | 2-step pi-H3 DWD | 54.5–61.8 s |
| Metal | M6 | 2-step PDMD, earlier benchmark | 45 s |

In the later Metal comparison, PDMD took 65.9 s on its second run and DWD took 54.5 s.
[CUDA benchmark](docs/pih3.md#dwd-benchmark) · [Metal comparison](docs/pih3.md#speed-on-metal)

## Generate images

```sh
make
tools/download-models.sh --image
target/release/mmh3 image --models models --out out.png --width 4096 --height 4096 \
  --prompt "A red panda in a knitted scarf sits on a mossy log in a misty pine forest at dawn."
```

This writes a 4096×4096 PNG with the model's default six steps. [Image generation and editing](docs/image.md)

| Backend | Device | Total at 4096×4096 |
| --- | --- | ---: |
| CUDA | DGX Spark | 63 s |
| Metal | M5 Max | 165 s |
| Metal | M6 | 337 s |

### Faster images

On CUDA and the M6, use PDMD with three steps at half resolution, the latent upscaler and one
step at full resolution. Set `--attention-precision int8-fp8` on CUDA or `int8-fp16` on Metal.
On CUDA, add `--linear-precision nvfp4`. On the M5 Max, `--draft-steps 2 --attention sol`
with FP16 attention and bilinear enlargement gave its fastest recorded time.
[Faster image commands](docs/image.md#faster) ·
[Image benchmarks](docs/image.md#speed)

| Backend | Device | Settings | Total at 4096×4096 |
| --- | --- | --- | ---: |
| CUDA | DGX Spark | PDMD, 3 draft steps, NVFP4 | 23 s |
| Metal | M5 Max | 2 draft steps, Sol-Attn, FP16 attention | 89 s |
| Metal | M6 | PDMD, 3 draft steps, INT8/FP16 attention | 140 s |

## Documentation

- Set up model files, build the engine and choose an output format. [Models](docs/models.md) ·
  [Build](docs/build.md) · [Output](docs/output.md)
- Generate video from keyframes or from reference images, sounds and clips.
  [First and last frames](docs/fl2va.md) · [Reference media](docs/ref2va.md)
- Split video generation across CUDA and Metal workers over RDMA or sockets, or serve video and
  image requests over HTTP with models kept in memory. [Distributed generation](docs/distributed.md) ·
  [Server](docs/server.md)
- Find the other presets, per-stage timings and accuracy checks. [Documentation index](docs/README.md) ·
  [Performance and accuracy](docs/performance.md)

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
