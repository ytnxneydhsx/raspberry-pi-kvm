#!/usr/bin/env python3
"""Convert the fixed-shape MeanVC2 DiT ONNX graph to RKNN FP16."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def check(code: int, step: str) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with RKNN error {code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization-level", type=int, default=3)
    args = parser.parse_args()

    rknn = RKNN(verbose=True)
    try:
        check(
            rknn.config(
                target_platform="rk3588",
                optimization_level=args.optimization_level,
            ),
            "config",
        )
        check(rknn.load_onnx(model=str(args.onnx)), "load_onnx")
        check(rknn.build(do_quantization=False), "build")
        check(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    print(f"saved_rknn={args.output}")


if __name__ == "__main__":
    main()
