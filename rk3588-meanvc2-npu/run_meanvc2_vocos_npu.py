#!/usr/bin/env python3
"""Run MeanVC2 with its Vocos backbone on the RK3588 NPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rknnlite.api import RKNNLite


def load_meanvc_runtime(root: Path):
    sys.path.insert(0, str(root / "runtime"))
    import run_rt

    return run_rt


class VocosNpuMixin:
    """Replace the fixed 14-frame Vocos backbone with RKNN inference."""

    def init_vocos_npu(self, model_path: Path) -> None:
        self.vocos_npu = RKNNLite(verbose=False)
        if self.vocos_npu.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"failed to load Vocos RKNN model: {model_path}")
        if self.vocos_npu.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
            raise RuntimeError("failed to initialize RKNN runtime")
        self.vocos_npu_calls = 0
        self.vocos_cpu_calls = 0
        self.vocos_npu_seconds = 0.0
        self.vocos_cpu_seconds = 0.0

    def close_vocos_npu(self) -> None:
        if getattr(self, "vocos_npu", None) is not None:
            self.vocos_npu.release()
            self.vocos_npu = None

    def _decode_mel(self, mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]
            mel = (mel + 1) / 2

            started = time.perf_counter()
            if mel.shape[-1] == 14:
                hidden_np = self.vocos_npu.inference(
                    inputs=[mel.detach().cpu().numpy().astype(np.float32)]
                )[0]
                hidden = torch.from_numpy(hidden_np.astype(np.float32, copy=False))
                spectral = self.vocoder.head.out(hidden).transpose(1, 2)
                magnitude_log, phase = spectral.chunk(2, dim=1)
                magnitude = torch.exp(magnitude_log).clamp(max=100.0)
                spectrum = torch.complex(
                    magnitude * torch.cos(phase),
                    magnitude * torch.sin(phase),
                )
                wav = self.vocoder.head.istft(spectrum).squeeze().cpu().numpy()
                self.vocos_npu_calls += 1
                self.vocos_npu_seconds += time.perf_counter() - started
            else:
                wav = self.vocoder.decode(mel).squeeze().detach().cpu().numpy()
                self.vocos_cpu_calls += 1
                self.vocos_cpu_seconds += time.perf_counter() - started

            if self.last_wav is not None:
                wav = wav[self.vocoder_wav_pad:]
                front = wav[:self.vocoder_wav_overlap]
                smooth_front = (
                    self.last_wav * self.down_linspace.numpy()
                    + front * self.up_linspace.numpy()
                )
                new_wav = np.concatenate(
                    [smooth_front, wav[self.vocoder_wav_overlap:-self.vocoder_wav_overlap]]
                )
            else:
                new_wav = wav[:-self.vocoder_wav_overlap]

            self.last_wav = wav[-self.vocoder_wav_overlap:]
        return new_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--vocos-rknn", type=Path, required=True)
    parser.add_argument("--target-emb", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.meanvc2_root.resolve()
    runtime = load_meanvc_runtime(root)

    class VocosNpuRunner(VocosNpuMixin, runtime.VCRunner):
        pass

    runner = VocosNpuRunner(
        target_wav=None,
        target_emb=args.target_emb,
        device="cpu",
        model="120ms",
        threads=args.threads,
    )
    runner.init_vocos_npu(args.vocos_rknn.resolve())
    try:
        runner.process_file(args.input, args.output, seed=args.seed)
        print(f"[Vocos NPU] NPU calls: {runner.vocos_npu_calls}")
        print(f"[Vocos NPU] CPU fallback calls: {runner.vocos_cpu_calls}")
        print(f"[Vocos NPU] Hybrid decode time: {runner.vocos_npu_seconds + runner.vocos_cpu_seconds:.3f}s")
    finally:
        runner.close_vocos_npu()


if __name__ == "__main__":
    main()
