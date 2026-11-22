"""Writes FP32 pi-H3 DWD reference pixels for the CUDA refiner test.

Run in an environment with LakonLab installed. Pass the full-precision
pi_h3_2step_dwd_adapter/diffusion_pytorch_model.safetensors as --adapter.
"""

import argparse
from pathlib import Path

import torch
from lakonlab.models.architectures.sol_h3.sol_h3 import SolH3DWDHead
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "dwd_pixel_refiner."
GRID = (7, 9, 10)
HIDDEN = 5376


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.resolve() == args.adapter.resolve():
        parser.error("--out must be different from --adapter")

    with torch.device("meta"):
        head = SolH3DWDHead(HIDDEN, (4, 32, 32), (1, 32, 32), 5)
    with safe_open(args.adapter, framework="pt") as file:
        names = file.keys()
        weights = {
            name.removeprefix(PREFIX): file.get_tensor(name).to(
                args.device, torch.float32
            )
            for name in names
            if name.startswith(PREFIX)
        }
    head.load_state_dict(weights, assign=True)
    head.eval()

    generator = torch.Generator().manual_seed(2026)
    features = torch.randn((*GRID, HIDDEN), generator=generator).half()
    with torch.inference_mode():
        rows = features.view(1, -1, HIDDEN).to(args.device, torch.float32)
        rgb = head(None, rows, GRID)[0].permute(1, 0, 2, 3)
        mean = rgb.new_tensor([-0.03, -0.088, -0.188])[:, None, None, None]
        std = rgb.new_tensor([0.458, 0.448, 0.45])[:, None, None, None]
        rgb = ((rgb * std + mean + 1) * 0.5).clamp_(0, 1).contiguous().cpu()
    save_file({"features": features, "rgb": rgb}, args.out)


if __name__ == "__main__":
    main()
