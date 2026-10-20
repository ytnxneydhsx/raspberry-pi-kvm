#!/usr/bin/env python3
"""Run MeanVC2 with steady-state DiT and Vocos stages on the RK3588 NPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rknnlite.api import RKNNLite
from torchaudio.compliance import kaldi

from run_meanvc2_vocos_npu import VocosNpuMixin, load_meanvc_runtime


class AsrNpuMixin:
    """Run the fixed 120 ms Fast-U2++ streaming encoder on RKNN."""

    def init_asr_npu(
        self, model_path: Path, position_table_path: Path, enabled: bool
    ) -> None:
        self.asr_npu_enabled = enabled
        self.asr_npu_calls = 0
        self.asr_npu_seconds = 0.0
        if not enabled:
            self.asr_npu = None
            return
        self.asr_npu = RKNNLite(verbose=False)
        if self.asr_npu.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"failed to load ASR RKNN model: {model_path}")
        if self.asr_npu.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
            raise RuntimeError("failed to initialize ASR RKNN runtime")
        self.asr_position_table = np.load(position_table_path).astype(np.float32)

    def close_asr_npu(self) -> None:
        if getattr(self, "asr_npu", None) is not None:
            self.asr_npu.release()
            self.asr_npu = None

    def _encode_chunk(self, samples: np.ndarray) -> torch.Tensor | None:
        if not self.asr_npu_enabled:
            return super()._encode_chunk(samples)
        with torch.no_grad():
            padded = (
                np.concatenate((self.samples_cache, samples))
                if self.samples_cache.size
                else samples
            )
            if len(padded) < 400:
                self.samples_cache = padded
                return None

            wav_t = torch.from_numpy(padded * (1 << 15)).unsqueeze(0)
            fbanks = kaldi.fbank(
                wav_t,
                frame_length=25,
                frame_shift=10,
                snip_edges=True,
                num_mel_bins=80,
                energy_floor=0.0,
                dither=0.0,
                sample_frequency=16000,
            )
            self.samples_cache = padded[160 * fbanks.shape[0] :]
            if self.frame_cache is not None:
                fbanks = torch.cat([self.frame_cache, fbanks], dim=0)

            if self.asr_offset >= 4000:
                cache_frames = self.asr_att_cache.size(2)
                self.asr_offset = max(cache_frames, self.asr_offset - 4000)

            bns = []
            index = 0
            while index + self._bn_window <= fbanks.shape[0]:
                fbank = fbanks[index : index + self._bn_window].unsqueeze(0)
                started = time.perf_counter()
                output, attention_cache, convolution_cache = self.asr_npu.inference(
                    inputs=[
                        fbank.numpy().astype(np.float32),
                        self.asr_att_cache.numpy().astype(np.float32).reshape(1, 192, 128),
                        self.asr_cnn_cache.numpy().astype(np.float32).reshape(1, 1536, 8),
                        self.asr_position_table[
                            :, self.asr_offset : self.asr_offset + 12, :
                        ],
                    ]
                )
                self.asr_npu_seconds += time.perf_counter() - started
                self.asr_npu_calls += 1
                self.asr_att_cache = torch.from_numpy(
                    attention_cache.astype(np.float32, copy=False).reshape(6, 4, 8, 128).copy()
                )
                self.asr_cnn_cache = torch.from_numpy(
                    convolution_cache.astype(np.float32, copy=False).reshape(6, 1, 256, 8).copy()
                )
                bns.append(torch.from_numpy(output.astype(np.float32, copy=False)).squeeze(0))
                self.asr_offset += self._asr_offset_step
                index += self._bn_stride
            self.frame_cache = fbanks[index:]
            if not bns:
                return None

            bn = torch.cat(bns, dim=0).unsqueeze(0)
            if self.encoder_output_cache is not None:
                bn = torch.cat([self.encoder_output_cache, bn], dim=1)
            self.encoder_output_cache = bn[:, -1:, :]
            if bn.shape[1] < 2:
                return None
            bn_up = F.interpolate(
                bn.transpose(1, 2), size=17, mode="linear", align_corners=True
            ).transpose(1, 2)
            return bn_up[:, 1:, :]


class DiTSteadyNpuMixin:
    """Use RKNN after the first three CPU chunks establish fixed-size KV caches."""

    def init_dit_npu(self, model_path: Path, npu_ode_steps: int) -> None:
        self.dit_npu = RKNNLite(verbose=False)
        if self.dit_npu.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"failed to load DiT RKNN model: {model_path}")
        if self.dit_npu.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
            raise RuntimeError("failed to initialize DiT RKNN runtime")
        self.dit_npu_calls = 0
        self.dit_npu_chunks = 0
        self.dit_cpu_chunks = 0
        self.dit_npu_seconds = 0.0
        self.dit_cpu_seconds = 0.0
        self.dit_cpu_ode_calls = 0
        self.dit_npu_ode_steps = npu_ode_steps

    def close_dit_npu(self) -> None:
        if getattr(self, "dit_npu", None) is not None:
            self.dit_npu.release()
            self.dit_npu = None

    @staticmethod
    def _steady_cache(caches) -> bool:
        if caches is None or len(caches) != 4:
            return False
        expected = (36, 36, 24, 24)
        return all(
            tuple(key.shape) == (1, 2, length, 64)
            and tuple(value.shape) == (1, 2, length, 64)
            for (key, value), length in zip(caches, expected, strict=True)
        )

    @staticmethod
    def _flatten_cache(caches) -> list[np.ndarray]:
        flattened = []
        for key, value in caches:
            flattened.extend(
                (
                    key.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 64),
                    value.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 64),
                )
            )
        return flattened

    @staticmethod
    def _restore_cache(outputs: list[np.ndarray]):
        caches = []
        for layer in range(4):
            length = 36 if layer < 2 else 24
            key = torch.from_numpy(
                outputs[layer * 2].astype(np.float32, copy=False).reshape(1, 2, length, 64).copy()
            )
            value = torch.from_numpy(
                outputs[layer * 2 + 1].astype(np.float32, copy=False).reshape(1, 2, length, 64).copy()
            )
            caches.append((key, value))
        return caches

    def _vc_step(self, cond: torch.Tensor) -> torch.Tensor:
        if self.vc_offset >= 4000 or not self._steady_cache(self.vc_kv_cache):
            started = time.perf_counter()
            mel = super()._vc_step(cond)
            self.dit_cpu_seconds += time.perf_counter() - started
            self.dit_cpu_chunks += 1
            return mel

        step_size = 0.5
        with torch.no_grad():
            x = torch.randn(1, cond.shape[1], 80, device=cond.device, dtype=cond.dtype)
            if self.noise_cache is not None:
                x[:, : self.block_size, :] = self.noise_cache
            self.noise_cache = x[:, -self.block_size :, :].clone()

            pre_kv_cache = self.vc_kv_cache
            pre_cache = self._flatten_cache(pre_kv_cache)
            rope = self.vc.rotary_embed.forward_from_seq_len(self.vc_offset + cond.shape[1])[0]
            rope = rope[:, -52:, :].detach().cpu().numpy().astype(np.float32)
            common = [
                cond.detach().cpu().numpy().astype(np.float32),
                self.vc_spk_emb.detach().cpu().numpy().astype(np.float32),
                self.vc_gtm_kv[0].detach().cpu().numpy().astype(np.float32),
                self.vc_gtm_kv[1].detach().cpu().numpy().astype(np.float32),
                rope,
            ]
            for index in range(2):
                t_value = 1.0 - index * step_size
                r_value = max(0.0, t_value - step_size)
                if index < self.dit_npu_ode_steps:
                    npu_started = time.perf_counter()
                    outputs = self.dit_npu.inference(
                        inputs=[
                            x.detach().cpu().numpy().astype(np.float32),
                            np.asarray([t_value], dtype=np.float32),
                            np.asarray([r_value], dtype=np.float32),
                            *common,
                            *pre_cache,
                        ]
                    )
                    self.dit_npu_seconds += time.perf_counter() - npu_started
                    velocity = torch.from_numpy(outputs[0].astype(np.float32, copy=False))
                    new_cache = self._restore_cache(outputs[1:])
                    self.dit_npu_calls += 1
                else:
                    cpu_started = time.perf_counter()
                    velocity, new_cache = self.vc(
                        x,
                        torch.tensor([t_value]),
                        torch.tensor([r_value]),
                        cond=cond,
                        spks=self.vc_spk_emb,
                        offset=self.vc_offset,
                        is_inference=True,
                        kv_cache=pre_kv_cache,
                        gtm_kv=self.vc_gtm_kv,
                    )
                    self.dit_cpu_seconds += time.perf_counter() - cpu_started
                    self.dit_cpu_ode_calls += 1
                x = x - step_size * velocity
                if index == 1:
                    self.vc_kv_cache = new_cache
            self.dit_npu_chunks += 1
            self.vc_offset += self.chunk_size
            return x[:, : self.chunk_size, :].transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--asr-rknn", type=Path, required=True)
    parser.add_argument("--asr-position-table", type=Path, required=True)
    parser.add_argument("--dit-rknn", type=Path, required=True)
    parser.add_argument("--vocos-rknn", type=Path, required=True)
    parser.add_argument("--asr-device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dit-npu-steps", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--target-emb", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.meanvc2_root.resolve()
    sys.path.insert(0, str(root / "npu"))
    runtime = load_meanvc_runtime(root)

    class MeanVC2NpuRunner(
        AsrNpuMixin, DiTSteadyNpuMixin, VocosNpuMixin, runtime.VCRunner
    ):
        pass

    runner = MeanVC2NpuRunner(
        target_wav=None,
        target_emb=args.target_emb,
        device="cpu",
        model="120ms",
        threads=args.threads,
    )
    runner.init_asr_npu(
        args.asr_rknn.resolve(),
        args.asr_position_table.resolve(),
        enabled=args.asr_device == "npu",
    )
    runner.init_dit_npu(args.dit_rknn.resolve(), args.dit_npu_steps)
    runner.init_vocos_npu(args.vocos_rknn.resolve())
    try:
        runner.process_file(args.input, args.output, seed=args.seed)
        print(
            f"[ASR NPU] calls: {runner.asr_npu_calls}, time: {runner.asr_npu_seconds:.3f}s"
        )
        print(
            f"[DiT NPU] NPU chunks: {runner.dit_npu_chunks}, calls: {runner.dit_npu_calls}, "
            f"CPU ODE calls: {runner.dit_cpu_ode_calls}, CPU warm-up chunks: "
            f"{runner.dit_cpu_chunks}, time: "
            f"{runner.dit_npu_seconds + runner.dit_cpu_seconds:.3f}s"
        )
        print(
            f"[Vocos NPU] NPU calls: {runner.vocos_npu_calls}, CPU fallback calls: "
            f"{runner.vocos_cpu_calls}, time: "
            f"{runner.vocos_npu_seconds + runner.vocos_cpu_seconds:.3f}s"
        )
    finally:
        runner.close_asr_npu()
        runner.close_dit_npu()
        runner.close_vocos_npu()


if __name__ == "__main__":
    main()
