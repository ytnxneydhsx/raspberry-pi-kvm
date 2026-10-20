#!/usr/bin/env python3
"""Export the RKNN-friendly Vocos backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class VocosBackboneWrapper(nn.Module):
    def __init__(self, backbone: torch.jit.ScriptModule):
        super().__init__()
        self.backbone = backbone

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.backbone(mel, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=14)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(42)
    vocoder = torch.jit.load(str(args.vocos)).eval()
    backbone = torch.jit.script(VocosBackboneWrapper(vocoder.backbone)).eval()
    mel = torch.rand(1, 80, args.frames)

    with torch.no_grad():
        hidden = backbone(mel)
        spectral = vocoder.head.out(hidden).transpose(1, 2)
        expected_waveform = vocoder.decode(mel)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        backbone,
        (mel,),
        str(args.output),
        input_names=["mel"],
        output_names=["hidden"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    np.savez(
        args.fixture,
        mel=mel.numpy(),
        expected_hidden=hidden.numpy(),
        expected_spectral=spectral.numpy(),
        expected_waveform=expected_waveform.numpy(),
    )
    print(f"saved_onnx={args.output}")
    print(f"saved_fixture={args.fixture}")
    print(f"hidden_shape={tuple(hidden.shape)}")
    print(f"spectral_shape={tuple(spectral.shape)}")
    print(f"waveform_shape={tuple(expected_waveform.shape)}")


if __name__ == "__main__":
    main()
