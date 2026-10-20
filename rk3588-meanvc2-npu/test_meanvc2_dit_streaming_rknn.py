#!/usr/bin/env python3
"""Validate the fixed-cache streaming MeanVC2 DiT RKNN graph."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


INPUT_NAMES = (
    "x", "t", "r", "cond", "speaker", "gtm_key", "gtm_value", "rope_freqs",
    "key_0", "value_0", "key_1", "value_1", "key_2", "value_2", "key_3", "value_3",
)
OUTPUT_NAMES = (
    "velocity", "key_0_out", "value_0_out", "key_1_out", "value_1_out",
    "key_2_out", "value_2_out", "key_3_out", "value_3_out",
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
    parser.add_argument("--loops", type=int, default=20)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    inputs = [fixture[name].astype(np.float32) for name in INPUT_NAMES]
    expected = [fixture[f"expected_{name}"].astype(np.float32) for name in OUTPUT_NAMES]
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
            f"{float(np.mean(np.abs(actual - reference))):.6f} cosine={cosine(actual, reference):.8f}"
        )
    for layer in range(4):
        cache_length = 36 if layer < 2 else 24
        preserved = cache_length - 12
        actual_key = outputs[1 + layer * 2].reshape(1, 2, cache_length, 64)[
            :, :, :preserved, :
        ]
        input_key = fixture[f"key_{layer}"].reshape(1, 2, cache_length, 64)[
            :, :, -preserved:, :
        ]
        print(
            f"key_{layer}_preserved_mean_abs_error="
            f"{float(np.mean(np.abs(actual_key - input_key))):.6f}"
        )
    print(f"mean_npu_ms={elapsed * 1000 / args.loops:.3f}")


if __name__ == "__main__":
    main()
