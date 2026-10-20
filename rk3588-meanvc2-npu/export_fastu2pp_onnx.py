#!/usr/bin/env python3
"""Export the fixed-shape 120 ms Fast-U2++ streaming step to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    model = torch.jit.load(str(args.checkpoint), map_location="cpu").eval()

    fbank = torch.randn(1, 19, 80, dtype=torch.float32)
    offset = torch.tensor(8, dtype=torch.int64)
    required_cache_size = torch.tensor(8, dtype=torch.int64)
    attention_cache = torch.zeros(6, 4, 8, 128, dtype=torch.float32)
    convolution_cache = torch.zeros(6, 1, 256, 8, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (
            fbank,
            offset,
            required_cache_size,
            attention_cache,
            convolution_cache,
        ),
        str(args.output),
        input_names=[
            "fbank",
            "offset",
            "required_cache_size",
            "attention_cache",
            "convolution_cache",
        ],
        output_names=["bottleneck", "attention_cache_out", "convolution_cache_out"],
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"saved_onnx={args.output}")


if __name__ == "__main__":
    main()
