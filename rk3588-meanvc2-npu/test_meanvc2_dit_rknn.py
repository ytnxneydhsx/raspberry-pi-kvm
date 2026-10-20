#!/usr/bin/env python3
"""Run and validate the fixed-shape MeanVC2 DiT RKNN model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


INPUT_NAMES = ("x", "t", "r", "cond", "speaker", "gtm_key", "gtm_value")


def check(code: int, step: str) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with RKNN error {code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--loops", type=int, default=20)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    inputs = [fixture[name].astype(np.float32) for name in INPUT_NAMES]
    expected = fixture["expected"].astype(np.float32)

    runtime = RKNNLite(verbose=False)
    try:
        check(runtime.load_rknn(str(args.model)), "load_rknn")
        check(
            runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL),
            "init_runtime",
        )
        output = runtime.inference(inputs=inputs)[0].astype(np.float32)
        start = time.perf_counter()
        for _ in range(args.loops):
            output = runtime.inference(inputs=inputs)[0].astype(np.float32)
        elapsed = time.perf_counter() - start
    finally:
        runtime.release()

    diff = output - expected
    print(f"output_shape={tuple(output.shape)}")
    print(f"loops={args.loops}")
    print(f"total_ms={elapsed * 1000:.3f}")
    print(f"mean_ms={elapsed * 1000 / args.loops:.3f}")
    print(f"max_abs_error={float(np.max(np.abs(diff))):.6f}")
    print(f"mean_abs_error={float(np.mean(np.abs(diff))):.6f}")
    print(f"cosine_similarity={float(np.vdot(output.ravel(), expected.ravel()) / (np.linalg.norm(output) * np.linalg.norm(expected))):.8f}")
    if not np.isfinite(output).all():
        raise SystemExit("MeanVC2 DiT RKNN output contains non-finite values")


if __name__ == "__main__":
    main()
