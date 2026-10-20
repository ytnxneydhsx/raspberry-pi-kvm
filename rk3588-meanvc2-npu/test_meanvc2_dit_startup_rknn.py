#!/usr/bin/env python3
"""Validate an empty/12/24-frame MeanVC2 DiT startup RKNN graph."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


BASE_INPUT_NAMES = (
    "x",
    "t",
    "r",
    "cond",
    "speaker",
    "gtm_key",
    "gtm_value",
    "rope_freqs",
)
CACHE_INPUT_NAMES = tuple(
    f"{kind}_{layer}" for layer in range(4) for kind in ("key", "value")
)
OUTPUT_NAMES = ("velocity",) + tuple(
    f"{kind}_{layer}_out" for layer in range(4) for kind in ("key", "value")
)


def cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.vdot(actual.ravel(), expected.ravel())
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--loops", type=int, default=10)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    input_names = BASE_INPUT_NAMES + tuple(
        name for name in CACHE_INPUT_NAMES if name in fixture
    )
    inputs = [fixture[name].astype(np.float32) for name in input_names]
    expected = [
        fixture[f"expected_{name}"].astype(np.float32) for name in OUTPUT_NAMES
    ]
    runtime = RKNNLite(verbose=False)
    if runtime.load_rknn(str(args.model)) != 0:
        raise RuntimeError(f"failed to load RKNN model: {args.model}")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
        raise RuntimeError("failed to initialize RKNN runtime")
    try:
        outputs = runtime.inference(inputs=inputs)
        started = time.perf_counter()
        for _ in range(args.loops):
            outputs = runtime.inference(inputs=inputs)
        elapsed = time.perf_counter() - started
    finally:
        runtime.release()

    for name, actual, reference in zip(OUTPUT_NAMES, outputs, expected, strict=True):
        print(
            f"{name}: shape={tuple(actual.shape)} mean_abs_error="
            f"{float(np.mean(np.abs(actual - reference))):.6f} "
            f"cosine={cosine(actual, reference):.8f}"
        )
    print(f"mean_npu_ms={elapsed * 1000 / args.loops:.3f}")


if __name__ == "__main__":
    main()
