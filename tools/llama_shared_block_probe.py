#!/usr/bin/env python3
"""Offline capability probe for a weight-shared multi-block Llama student.

The student maps a captured early residual state to a captured late residual
state by applying one learned residual block repeatedly.  This is a real
shared-depth grammar: artifact bytes are charged once, while each recurrence's
MACs and sequential matvecs are charged separately.  Passing this surface
gate is still neither model quality nor a TPS claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = "hawking.tg.llama_shared_block_probe.v1"
SEED = 17


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class SharedResidual(torch.nn.Module):
    def __init__(self, hidden: int, width: int, steps: int, step_rank: int = 0) -> None:
        super().__init__()
        self.expand = torch.nn.Linear(hidden, width)
        self.project = torch.nn.Linear(width, hidden)
        self.steps = steps
        self.step_rank = step_rank
        self.adapters = torch.nn.ModuleList()
        if step_rank:
            for _ in range(steps):
                self.adapters.append(torch.nn.Sequential(
                    torch.nn.Linear(hidden, step_rank, bias=False),
                    torch.nn.Linear(step_rank, hidden, bias=False),
                ))
        # A learned residual scale begins conservatively; repeated additions
        # are otherwise numerically unstable before the shared block learns.
        self.scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for step in range(self.steps):
            delta = self.project(torch.nn.functional.silu(self.expand(state)))
            if self.adapters:
                delta = delta + self.adapters[step](state)
            state = state + self.scale * delta
        return state


def physical(hidden: int, width: int, steps: int, step_rank: int = 0) -> dict[str, int]:
    # Two matrices plus their bias vectors and one recurrence scale. Means are
    # codec state and must be stored, but the recurrent weights remain unique.
    block = hidden * width + width + width * hidden + hidden + 1
    adapters = steps * (2 * hidden * step_rank) if step_rank else 0
    stored = block + adapters + hidden + hidden
    return {
        "stored_parameters": stored,
        "prospective_fp16_artifact_bytes": stored * 2,
        "unique_active_parameters_per_token": stored,
        "prospective_unique_active_fp16_bytes_per_token": stored * 2,
        "executed_macs_per_token": steps * (2 * hidden * width + 2 * hidden * step_rank),
        "sequential_matvecs_per_token": steps * (2 + (2 if step_rank else 0)),
        "shared_recurrence_steps": steps,
        "step_rank": step_rank,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step-rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.width, args.steps, args.epochs, args.batch) < 1 or args.step_rank < 0:
        raise SystemExit("all geometry and optimizer counts must be positive")
    receipt = json.loads(args.capture_receipt.read_text())
    dataset_hash = sha256(args.dataset)
    if receipt.get("dataset", {}).get("sha256") != dataset_hash:
        raise SystemExit("dataset hash does not match capture receipt")
    data = np.load(args.dataset)
    x = np.asarray(data["inputs"], dtype=np.float32)
    y = np.asarray(data["targets"], dtype=np.float32)
    heldout = np.asarray(data["heldout"], dtype=bool)
    data.close()
    if x.ndim != 2 or x.shape != y.shape or heldout.shape != (len(x),):
        raise SystemExit("dataset must have paired equal-width vectors and one heldout bit per row")
    fit_mask = ~heldout
    if fit_mask.sum() < 8192 or heldout.sum() < 2048:
        raise SystemExit("sealed dataset below capability minimum")
    x_mean = x[fit_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = y[fit_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = SharedResidual(x.shape[1], args.width, args.steps, args.step_rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    fit_rows = np.flatnonzero(fit_mask)
    for epoch in range(args.epochs):
        order = np.random.default_rng(SEED + epoch).permutation(fit_rows)
        for start in range(0, len(order), args.batch):
            rows = order[start:start + args.batch]
            xb = torch.from_numpy(x[rows] - x_mean).to(device)
            yb = torch.from_numpy(y[rows] - y_mean).to(device)
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(xb), yb).backward()
            optimizer.step()
    def normalized_rmse(mask: np.ndarray) -> float:
        error = baseline = 0.0
        with torch.no_grad():
            for start in range(0, int(mask.sum()), args.batch):
                rows = np.flatnonzero(mask)[start:start + args.batch]
                predicted = model(torch.from_numpy(x[rows] - x_mean).to(device)).float().cpu().numpy() + y_mean
                error += float(np.square(predicted - y[rows], dtype=np.float64).sum())
                baseline += float(np.square(y[rows] - y_mean, dtype=np.float64).sum())
        return float(np.sqrt(error / max(baseline, np.finfo(np.float64).tiny)))
    fit_score, heldout_score = normalized_rmse(fit_mask), normalized_rmse(heldout)
    result = {
        "schema": SCHEMA,
        "status": "OFFLINE_SURFACE_GATE_PASS_RUNTIME_REQUIRED" if heldout_score <= 0.10 else "OFFLINE_SURFACE_GATE_FAILED",
        "dataset_sha256": dataset_hash,
        "device": str(device),
        "architecture": {"shared_block": "linear_silu_linear_residual", "width": args.width, "steps": args.steps, "step_rank": args.step_rank, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": SEED},
        "score": {"fit_normalized_rmse": fit_score, "heldout_normalized_rmse": heldout_score},
        "physical": physical(x.shape[1], args.width, args.steps, args.step_rank),
        "runtime_eligibility": "NO: requires emitted artifact, generated-token capability, artifact/runtime parity, and strict matched decode evidence",
        "tps_claim": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
