#!/usr/bin/env python3
"""Validate a streaming Fast-U2++ RKNN model over consecutive cache updates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rknnlite.api import RKNNLite


def cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.linalg.norm(actual) * np.linalg.norm(expected)
    return float(np.vdot(actual.ravel(), expected.ravel()) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--position-table", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = torch.jit.load(str(args.checkpoint), map_location="cpu").eval()
    position_table = np.load(args.position_table).astype(np.float32)
    runtime = RKNNLite(verbose=False)
    if runtime.load_rknn(str(args.model)) != 0:
        raise RuntimeError(f"failed to load RKNN model: {args.model}")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
        raise RuntimeError("failed to initialize RKNN runtime")

    torch_attention = torch.zeros(6, 4, 8, 128, dtype=torch.float32)
    torch_convolution = torch.zeros(6, 1, 256, 8, dtype=torch.float32)
    npu_attention = torch_attention.numpy().reshape(1, 192, 128).copy()
    npu_convolution = torch_convolution.numpy().reshape(1, 1536, 8).copy()
    elapsed = 0.0
    worst_mean = 0.0
    worst_cosine = 1.0
    try:
        with torch.no_grad():
            for step in range(args.steps):
                fbank = torch.randn(1, 19, 80, dtype=torch.float32)
                offset = torch.tensor(8 + step * 4, dtype=torch.int64)
                required = torch.tensor(8, dtype=torch.int64)
                torch_attention_in = torch_attention.numpy().copy()
                torch_convolution_in = torch_convolution.numpy().copy()
                expected, torch_attention, torch_convolution = model(
                    fbank,
                    offset,
                    required,
                    torch_attention,
                    torch_convolution,
                )
                started = time.perf_counter()
                actual, npu_attention, npu_convolution = runtime.inference(
                    inputs=[
                        fbank.numpy(),
                        npu_attention,
                        npu_convolution,
                        position_table[:, int(offset) : int(offset) + 12, :],
                    ]
                )
                elapsed += time.perf_counter() - started
                teacher_actual, teacher_attention, teacher_convolution = runtime.inference(
                    inputs=[
                        fbank.numpy(),
                        torch_attention_in.reshape(1, 192, 128),
                        torch_convolution_in.reshape(1, 1536, 8),
                        position_table[:, int(offset) : int(offset) + 12, :],
                    ]
                )
                fixed_position_actual = runtime.inference(
                    inputs=[
                        fbank.numpy(),
                        torch_attention_in.reshape(1, 192, 128),
                        torch_convolution_in.reshape(1, 1536, 8),
                        position_table[:, 8:20, :],
                    ]
                )[0]

                expected_np = expected.numpy()
                mean_error = float(np.mean(np.abs(actual - expected_np)))
                cosine = cosine_similarity(actual, expected_np)
                teacher_mean_error = float(
                    np.mean(np.abs(teacher_actual - expected_np))
                )
                fixed_position_error = float(
                    np.mean(np.abs(fixed_position_actual - expected_np))
                )
                position_effect = float(
                    np.mean(np.abs(teacher_actual - fixed_position_actual))
                )
                attention_error = float(
                    np.mean(
                        np.abs(
                            npu_attention.reshape(6, 4, 8, 128)
                            - torch_attention.numpy()
                        )
                    )
                )
                convolution_error = float(
                    np.mean(
                        np.abs(
                            npu_convolution.reshape(6, 1, 256, 8)
                            - torch_convolution.numpy()
                        )
                    )
                )
                worst_mean = max(worst_mean, mean_error)
                worst_cosine = min(worst_cosine, cosine)
                print(
                    f"step={step} offset={int(offset)} mean_abs_error={mean_error:.6f} "
                    f"cosine_similarity={cosine:.8f} teacher_mean_abs_error={teacher_mean_error:.6f} "
                    f"fixed_position_error={fixed_position_error:.6f} "
                    f"position_effect={position_effect:.6f} "
                    f"attention_cache_error={attention_error:.6f} "
                    f"convolution_cache_error={convolution_error:.6f}"
                )
    finally:
        runtime.release()

    print(f"mean_npu_ms={elapsed * 1000 / args.steps:.3f}")
    print(f"worst_mean_abs_error={worst_mean:.6f}")
    print(f"worst_cosine_similarity={worst_cosine:.8f}")
    if not np.isfinite(npu_attention).all() or not np.isfinite(npu_convolution).all():
        sys.exit("Fast-U2++ RKNN cache contains non-finite values")


if __name__ == "__main__":
    main()
