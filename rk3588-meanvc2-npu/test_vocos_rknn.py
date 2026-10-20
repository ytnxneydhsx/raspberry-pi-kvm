#!/usr/bin/env python3
"""Validate Vocos RKNN frontend and finish ISTFT on the CPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from rknnlite.api import RKNNLite


def check(code: int, step: str) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with RKNN error {code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--vocos", type=Path, required=True)
    parser.add_argument("--loops", type=int, default=50)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    mel = fixture["mel"].astype(np.float32)
    expected_hidden = fixture["expected_hidden"].astype(np.float32)
    expected_spectral = fixture["expected_spectral"].astype(np.float32)
    expected_waveform = fixture["expected_waveform"].astype(np.float32)

    runtime = RKNNLite(verbose=False)
    try:
        check(runtime.load_rknn(str(args.model)), "load_rknn")
        check(
            runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL),
            "init_runtime",
        )
        hidden = runtime.inference(inputs=[mel])[0].astype(np.float32)
        start = time.perf_counter()
        for _ in range(args.loops):
            hidden = runtime.inference(inputs=[mel])[0].astype(np.float32)
        npu_elapsed = time.perf_counter() - start
    finally:
        runtime.release()

    hidden_diff = hidden - expected_hidden
    vocoder = torch.jit.load(str(args.vocos)).eval()
    with torch.no_grad():
        spectral_tensor = vocoder.head.out(torch.from_numpy(hidden)).transpose(1, 2)
    spectral = spectral_tensor.numpy()
    spectral_diff = spectral - expected_spectral
    magnitude_log, phase = spectral_tensor.chunk(2, dim=1)
    magnitude = torch.exp(magnitude_log).clamp(max=100.0)
    spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    start = time.perf_counter()
    with torch.no_grad():
        waveform = vocoder.head.istft(spectrum).numpy()
    istft_elapsed = time.perf_counter() - start
    waveform_diff = waveform - expected_waveform

    cosine = float(
        np.vdot(spectral.ravel(), expected_spectral.ravel())
        / (np.linalg.norm(spectral) * np.linalg.norm(expected_spectral))
    )
    print(f"spectral_shape={tuple(spectral.shape)}")
    print(f"npu_mean_ms={npu_elapsed * 1000 / args.loops:.3f}")
    print(f"hidden_max_abs_error={float(np.max(np.abs(hidden_diff))):.6f}")
    print(f"hidden_mean_abs_error={float(np.mean(np.abs(hidden_diff))):.6f}")
    print(f"spectral_max_abs_error={float(np.max(np.abs(spectral_diff))):.6f}")
    print(f"spectral_mean_abs_error={float(np.mean(np.abs(spectral_diff))):.6f}")
    print(f"spectral_cosine_similarity={cosine:.8f}")
    print(f"cpu_istft_ms={istft_elapsed * 1000:.3f}")
    print(f"waveform_max_abs_error={float(np.max(np.abs(waveform_diff))):.6f}")
    print(f"waveform_mean_abs_error={float(np.mean(np.abs(waveform_diff))):.6f}")
    if not np.isfinite(hidden).all() or not np.isfinite(waveform).all():
        raise SystemExit("Vocos RKNN produced non-finite values")


if __name__ == "__main__":
    main()
