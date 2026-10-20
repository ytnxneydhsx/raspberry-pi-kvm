#!/usr/bin/env python3
"""Compare consecutive steady-state DiT NPU cache updates with PyTorch."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rknnlite.api import RKNNLite


def cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.vdot(actual.ravel(), expected.ravel())
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )


def flatten_caches(caches):
    values = []
    for key, value in caches:
        values.extend((key.numpy().reshape(1, -1, 64), value.numpy().reshape(1, -1, 64)))
    return values


def unflatten_caches(values):
    caches = []
    for layer in range(4):
        length = 36 if layer < 2 else 24
        key = torch.from_numpy(values[layer * 2].reshape(1, 2, length, 64).copy())
        value = torch.from_numpy(values[layer * 2 + 1].reshape(1, 2, length, 64).copy())
        caches.append((key, value))
    return caches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    root = args.meanvc2_root.resolve()
    sys.path.insert(0, str(root / "runtime"))
    from run_rt import _load_vc_model

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = _load_vc_model(
        str(root / "src/config/config_120ms_40ms.json"),
        str(root / "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
        "cpu",
    )
    for module in model.modules():
        if hasattr(module, "native_rms_norm"):
            module.native_rms_norm = False
    speaker = torch.randn(1, 256)
    with torch.no_grad():
        gtm_key, gtm_value = model.gtm(speaker)
        cpu_cache = None
        for index in range(3):
            _, cpu_cache = model(
                torch.randn(1, 16, 80),
                torch.tensor([1.0]),
                torch.tensor([0.5]),
                cond=torch.randn(1, 16, 256),
                spks=speaker,
                offset=index * 12,
                is_inference=True,
                kv_cache=cpu_cache,
                gtm_kv=(gtm_key, gtm_value),
            )
    npu_caches = flatten_caches(cpu_cache)

    runtime = RKNNLite(verbose=False)
    if runtime.load_rknn(str(args.model)) != 0:
        raise RuntimeError(f"failed to load RKNN model: {args.model}")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
        raise RuntimeError("failed to initialize RKNN runtime")

    npu_seconds = 0.0
    try:
        with torch.no_grad():
            for step in range(args.steps):
                offset = 36 + step * 12
                x = torch.randn(1, 16, 80)
                cond = torch.randn(1, 16, 256)
                expected, cpu_cache = model(
                    x,
                    torch.tensor([1.0]),
                    torch.tensor([0.5]),
                    cond=cond,
                    spks=speaker,
                    offset=offset,
                    is_inference=True,
                    kv_cache=cpu_cache,
                    gtm_kv=(gtm_key, gtm_value),
                )
                rope = model.rotary_embed.forward_from_seq_len(offset + 16)[0][:, -52:, :]
                inputs = [
                    x.numpy(),
                    np.asarray([1.0], dtype=np.float32),
                    np.asarray([0.5], dtype=np.float32),
                    cond.numpy(),
                    speaker.numpy(),
                    gtm_key.numpy(),
                    gtm_value.numpy(),
                    rope.numpy(),
                    *npu_caches,
                ]
                started = time.perf_counter()
                outputs = runtime.inference(inputs=inputs)
                npu_seconds += time.perf_counter() - started
                actual = outputs[0].astype(np.float32)
                npu_caches = [value.astype(np.float32) for value in outputs[1:]]
                expected_np = expected.numpy()
                cache_reference = flatten_caches(cpu_cache)
                cache_error = max(
                    float(np.mean(np.abs(actual_cache - expected_cache)))
                    for actual_cache, expected_cache in zip(npu_caches, cache_reference, strict=True)
                )
                print(
                    f"step={step} offset={offset} mean_abs_error="
                    f"{float(np.mean(np.abs(actual - expected_np))):.6f} "
                    f"cosine={cosine(actual, expected_np):.8f} max_cache_mean_error={cache_error:.6f}"
                )
    finally:
        runtime.release()
    print(f"mean_npu_ms={npu_seconds * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
