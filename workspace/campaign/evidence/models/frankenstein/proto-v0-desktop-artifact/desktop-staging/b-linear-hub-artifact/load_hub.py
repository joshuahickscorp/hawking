#!/usr/bin/env python3
"""Standalone loader for the B_LINEAR hub Desktop artifact.

Requires only: Python 3 + torch. Reconstructs the residual and applies
bypass vs bridge-on without the Grok worktree or full hawking lab package.

Usage:
  python load_hub.py --artifact-dir . --hidden-npy /path/to/L37.npy
  python load_hub.py --artifact-dir . --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSubspaceInit(nn.Module):
    def __init__(self, d_model: int = 4096, rank: int = 16, scale: float = 0.05):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.scale = float(scale)
        self.a_lo = nn.Linear(d_model, rank, bias=False)
        self.a_hi = nn.Linear(rank, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (self.a_hi(self.a_lo(x)) + self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class TinyRouter(nn.Module):
    def __init__(self, d_model: int, n_choices: int, gate_hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, gate_hidden)
        self.fc2 = nn.Linear(gate_hidden, n_choices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean-pool if sequence
        h = x.mean(dim=-2) if x.dim() >= 3 else x
        return self.fc2(F.silu(self.fc1(h)))


def load_linear_from_checkpoint(ckpt_path: Path) -> LinearSubspaceInit:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert blob.get("capability_claim") is not True
    state = blob["runtime_state_dict"]
    spec = blob.get("spec") or {}
    mod = LinearSubspaceInit(
        d_model=int(spec.get("d_student") or 4096),
        rank=int(spec.get("rank") or 16),
        scale=0.05,
    )
    # Map linear_init.* keys
    mapped = {}
    for k, v in state.items():
        if k.startswith("linear_init."):
            mapped[k[len("linear_init.") :]] = v
    missing, unexpected = mod.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected
    return mod


def force_one_hot(router: TinyRouter, idx: int, scale: float = 50.0) -> None:
    with torch.no_grad():
        router.fc2.weight.zero_()
        router.fc2.bias.zero_()
        router.fc2.bias[idx] = float(scale)


@torch.no_grad()
def apply(mod: LinearSubspaceInit, x: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "bypass":
        return x
    if mode == "bridge_on":
        return mod(x)
    raise SystemExit(f"unknown mode {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=Path("."))
    ap.add_argument("--hidden-npy", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    ckpt = args.artifact_dir / "BEST_BALANCED.pt"
    receipt = args.artifact_dir / "ARTIFACT_RECEIPT.json"
    if receipt.is_file():
        rec = json.loads(receipt.read_text())
        assert rec.get("capability_claim") is not True
        print("receipt_arm", rec.get("arm"))
        print("receipt_capability_claim", rec.get("capability_claim"))
    mod = load_linear_from_checkpoint(ckpt)
    print("loaded_linear", mod.d_model, mod.rank, "scale", mod.scale)
    if args.self_test:
        x = torch.randn(4, mod.d_model)
        y_off = apply(mod, x, mode="bypass")
        y_on = apply(mod, x, mode="bridge_on")
        assert torch.equal(y_off, x)
        assert torch.isfinite(y_on).all()
        print("self_test_ok", "max_abs_delta", float((y_on - y_off).abs().max()))
        return
    if args.hidden_npy is None:
        raise SystemExit("pass --hidden-npy or --self-test")
    import numpy as np
    arr = np.load(args.hidden_npy)
    x = torch.from_numpy(arr.astype("float32"))
    y_off = apply(mod, x, mode="bypass")
    y_on = apply(mod, x, mode="bridge_on")
    print("shape", list(x.shape))
    print("bypass_identity", bool(torch.equal(y_off, x)))
    print("bridge_on_finite", bool(torch.isfinite(y_on).all()))
    print("max_abs_delta", float((y_on - y_off).abs().max()))


if __name__ == "__main__":
    main()
