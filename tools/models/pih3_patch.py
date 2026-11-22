# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Converts a pi-H3 adapter, including the ComfyUI ConvRot DWD release, for mmh3."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from checkpoint import DTYPES, SafeTensors, bf16_bytes, write_safetensors
from diffusers_lora import LAYERS, NAME, PREFIX, block_factors, block_updates

OUTPUT_TENSORS = {
    "norm_out.norm.weight": "final_layer.norm.weight",
    "norm_out.linear.weight": "final_layer.adaln_proj.linear.weight",
    "norm_out.linear.bias": "final_layer.adaln_proj.linear.bias",
    "proj_out.weight": "final_layer.video_out.weight",
    "proj_out.bias": "final_layer.video_out.bias",
    "audio_proj_out.weight": "final_layer.audio_out.weight",
    "audio_proj_out.bias": "final_layer.audio_out.bias",
}
BLOCK_COUNT = 50
RANK = 64


def replacement(source, base, original, name):
    shape = source.shape(original)
    if shape != base.shape(name):
        raise ValueError(f"{original}: replacement does not fit {name} in the base")
    dtype = base.dtype(name)
    if dtype == source.dtype(original):
        return dtype, shape, source.read(original).tobytes()
    if dtype not in ("BF16", "F16", "F32"):
        raise ValueError(
            f"{name}: incompatible replacement dtype {source.dtype(original)}"
        )
    values = source.load(original)
    data = (
        bf16_bytes(values)
        if dtype == "BF16"
        else values.astype(DTYPES[dtype]).tobytes()
    )
    return dtype, shape, data


def add_factors(tensors, base, name, up, down):
    outputs, inputs = base.shape(f"{name}.weight")
    if up.shape != (outputs, RANK) or down.shape != (RANK, inputs):
        raise ValueError(f"{name}: rank-{RANK} factors do not fit the base")
    tensors[f"{PREFIX}{name}.lora_A.weight"] = ("BF16", down.shape, bf16_bytes(down))
    tensors[f"{PREFIX}{name}.lora_B.weight"] = ("BF16", up.shape, bf16_bytes(up))
    # The released factors have unit scale. mmh3 applies alpha / rank.
    tensors[f"{PREFIX}{name}.alpha"] = ("BF16", (), bf16_bytes([RANK]))


def convert(source, base):
    factors = [name for name in source.header if NAME.fullmatch(name)]
    blocks = block_factors(factors)
    expected_blocks = {("transformer_blocks", index) for index in range(BLOCK_COUNT)}
    if blocks.keys() != expected_blocks:
        raise ValueError(f"expected LoRA factors for all {BLOCK_COUNT} pi-H3 blocks")
    replacements = {
        f"transformer_blocks.{index}.adaln_proj.linear.weight": (
            f"blocks.{index}.adaln_proj.linear.weight"
        )
        for index in range(BLOCK_COUNT)
    }
    replacements.update(OUTPUT_TENSORS)
    if source.header.keys() != set(factors) | replacements.keys():
        raise ValueError("expected the tensor layout of a non-DWD pi-H3 adapter")

    tensors = {}
    for (_, index), layers in sorted(blocks.items()):
        block = f"blocks.{index}"
        if layers.keys() != LAYERS.keys() or any(
            pair.keys() != {"A", "B"} for pair in layers.values()
        ):
            raise ValueError(
                f"{block}: expected paired factors for attention output and MLP"
            )
        for name, up, down in block_updates(source, block, layers):
            add_factors(tensors, base, name, up, down)

    for original, name in replacements.items():
        tensors[f"{PREFIX}{name}"] = replacement(source, base, original, name)
    return tensors


def convert_dwd(source, base):
    if "blocks.0.adaln_proj.linear.weight" not in source:
        raise ValueError("use the pi-H3 DWD adapter named *_int8_convrot.safetensors")
    tensors = {}
    used = set()

    def copy(name, prefix):
        tensors[f"{prefix}{name}"] = (
            source.dtype(name),
            source.shape(name),
            source.read(name).tobytes(),
        )
        used.add(name)

    def replace(name):
        tensors[f"{PREFIX}{name}"] = replacement(source, base, name, name)
        used.add(name)

    for name in OUTPUT_TENSORS.values():
        replace(name)
    for index in range(BLOCK_COUNT):
        block = f"blocks.{index}"
        replace(f"{block}.adaln_proj.linear.weight")
        if index >= BLOCK_COUNT - 4:
            replace(f"{block}.adaln_proj.linear.bias")
        for layer in LAYERS.values():
            name = f"{block}.{layer}"
            if index < BLOCK_COUNT - 4:
                up, down = f"{name}.lora_B.weight", f"{name}.lora_A.weight"
                # The ComfyUI adapter already has gate before up in its fc1 rows.
                add_factors(tensors, base, name, source.load(up), source.load(down))
                used.update((up, down))
            else:
                quant = f"{name}.comfy_quant"
                if source.dtype(f"{name}.weight") != "I8" or json.loads(
                    source.read(quant).tobytes()
                ) != json.loads(base.read(quant).tobytes()):
                    raise ValueError(f"{name}: expected the base's INT8 ConvRot format")
                used.add(quant)
                replace(f"{name}.weight")
                replace(f"{name}.weight_scale")
    for suffix, shape in [("weight", [10752, 8]), ("bias", [10752])]:
        name = f"dwd_norm_out.linear.{suffix}"
        if source.shape(name) != shape or source.dtype(name) != "F16":
            raise ValueError(f"{name}: incompatible DWD AdaLN tensor")
        copy(name, PREFIX)
    for name in source.header:
        if name.startswith("dwd_pixel_refiner."):
            copy(name, "")
    if source.header.keys() != used:
        raise ValueError(
            f"unexpected DWD tensors {sorted(source.header.keys() - used)}"
        )
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, type=Path, help="the pi-H3 safetensors file"
    )
    parser.add_argument("--base", required=True, type=Path, help="the pruned FL2VA DiT")
    parser.add_argument("--out", required=True, type=Path, help="patch file to write")
    args = parser.parse_args()
    if args.out.resolve() in (args.adapter.resolve(), args.base.resolve()):
        parser.error("--out must be different from --adapter and --base")
    try:
        source, base = SafeTensors(args.adapter), SafeTensors(args.base)
        conversion = convert_dwd if "dwd_norm_out.linear.weight" in source else convert
        tensors = conversion(source, base)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as directory:
        output = Path(directory) / args.out.name
        write_safetensors(
            output, tensors, {"source": f"Lakonik/pi-H3 ({args.adapter.name})"}
        )
        output.replace(args.out)
    print(f"wrote {args.out} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
