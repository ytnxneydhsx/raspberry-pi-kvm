#!/usr/bin/env python3
"""Export one fixed-shape MeanVC2 DiT inference step to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class StatelessDiTStep(nn.Module):
    """A fixed 120 ms MeanVC2 DiT step used for RKNN compatibility tests."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: torch.Tensor,
        speaker: torch.Tensor,
        gtm_key: torch.Tensor,
        gtm_value: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self.model(
            x,
            t,
            r,
            cache=None,
            cond=cond,
            spks=speaker,
            offset=0,
            mask=None,
            is_inference=True,
            is_uncondition=False,
            cfg_mask=None,
            kv_cache=None,
            gtm_kv=(gtm_key, gtm_value),
        )
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
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
    # PyTorch 2.5's legacy ONNX exporter has no aten::rms_norm symbolic.
    # MeanVC2 already carries an algebraically equivalent portable path for
    # older PyTorch releases; select it only in this export copy.
    for module in model.modules():
        if hasattr(module, "native_rms_norm"):
            module.native_rms_norm = False
    wrapper = StatelessDiTStep(model).eval()

    x = torch.randn(1, 16, 80)
    t = torch.tensor([1.0], dtype=torch.float32)
    r = torch.tensor([0.5], dtype=torch.float32)
    cond = torch.randn(1, 16, 256)
    speaker = torch.randn(1, 256)
    with torch.no_grad():
        gtm_key, gtm_value = model.gtm(speaker)
        expected = wrapper(x, t, r, cond, speaker, gtm_key, gtm_value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (x, t, r, cond, speaker, gtm_key, gtm_value),
        str(args.output),
        input_names=["x", "t", "r", "cond", "speaker", "gtm_key", "gtm_value"],
        output_names=["velocity"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    np.savez(
        args.fixture,
        x=x.numpy(),
        t=t.numpy(),
        r=r.numpy(),
        cond=cond.numpy(),
        speaker=speaker.numpy(),
        gtm_key=gtm_key.numpy(),
        gtm_value=gtm_value.numpy(),
        expected=expected.numpy(),
    )
    print(f"saved_onnx={args.output}")
    print(f"saved_fixture={args.fixture}")
    print(f"expected_shape={tuple(expected.shape)}")


if __name__ == "__main__":
    main()
