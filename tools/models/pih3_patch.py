# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Converts a non-DWD pi-H3 adapter into an mmh3 DiT patch."""

import argparse
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
            outputs, inputs = base.shape(f"{name}.weight")
            if up.shape != (outputs, RANK) or down.shape != (RANK, inputs):
                raise ValueError(f"{name}: rank-{RANK} factors do not fit the base")
            tensors[f"{PREFIX}{name}.lora_A.weight"] = (
                "BF16",
                down.shape,
                bf16_bytes(down),
            )
            tensors[f"{PREFIX}{name}.lora_B.weight"] = (
                "BF16",
                up.shape,
                bf16_bytes(up),
            )
            # The released factors have unit scale. mmh3 applies alpha / rank.
            tensors[f"{PREFIX}{name}.alpha"] = ("BF16", (), bf16_bytes([RANK]))

    for original, name in replacements.items():
        values = source.load(original)
        if list(values.shape) != base.shape(name):
            raise ValueError(f"{original}: replacement does not fit {name} in the base")
        dtype = base.dtype(name)
        if dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"{name}: unsupported replacement dtype {dtype}")
        data = (
            bf16_bytes(values)
            if dtype == "BF16"
            else values.astype(DTYPES[dtype]).tobytes()
        )
        tensors[f"{PREFIX}{name}"] = (dtype, values.shape, data)
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
        tensors = convert(SafeTensors(args.adapter), SafeTensors(args.base))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as directory:
        output = Path(directory) / args.out.name
        write_safetensors(
            output, tensors, {"source": f"Lakonik/pi-H3 ({args.adapter.name})"}
        )
        output.replace(args.out)
    print(
        f"wrote {args.out}: {BLOCK_COUNT * len(LAYERS)} LoRAs, {BLOCK_COUNT + len(OUTPUT_TENSORS)} replacements"
    )


if __name__ == "__main__":
    main()
