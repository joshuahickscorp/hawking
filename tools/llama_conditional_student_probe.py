#!/usr/bin/env python3
"""Capability-gate a hard-routed, source-derived Llama FFN student.

This is an offline experiment only.  The router is a fixed random projection
followed by fit-only k-means centroids; each generated token executes exactly
one compact SiLU expert.  It deliberately bills both stored and *active*
parameters so routing capacity cannot be mistaken for a decode-speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = "hawking.tg.llama_conditional_student_probe.v1"
SEED = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kmeans(values: np.ndarray, experts: int, iterations: int) -> np.ndarray:
    """Deterministic, fit-only Lloyd routing centroids."""
    if len(values) < experts:
        raise ValueError("fewer fit rows than experts")
    rng = np.random.default_rng(SEED)
    centers = values[rng.choice(len(values), size=experts, replace=False)].copy()
    for _ in range(iterations):
        d2 = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        updated = centers.copy()
        for expert in range(experts):
            members = values[labels == expert]
            if len(members):
                updated[expert] = members.mean(axis=0, dtype=np.float64)
        if np.array_equal(updated, centers):
            break
        centers = updated
    return centers.astype(np.float32, copy=False)


def route(values: np.ndarray, projection: np.ndarray, centers: np.ndarray) -> np.ndarray:
    features = values @ projection
    return ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--router-dim", type=int, default=16)
    parser.add_argument("--kmeans-iterations", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.experts, args.width, args.router_dim, args.epochs, args.batch) < 1:
        raise SystemExit("all geometry and training arguments must be positive")
    capture = json.loads(args.capture_receipt.read_text())
    dataset_hash = sha256(args.dataset)
    if capture.get("dataset", {}).get("sha256") != dataset_hash:
        raise SystemExit("dataset hash does not match sealed capture receipt")
    data = np.load(args.dataset)
    x = np.asarray(data["inputs"], dtype=np.float32)
    y = np.asarray(data["targets"], dtype=np.float32)
    heldout = np.asarray(data["heldout"], dtype=bool)
    data.close()
    fit_mask = ~heldout
    if fit_mask.sum() < 8192 or heldout.sum() < 2048:
        raise SystemExit("sealed dataset below capability minimum")
    x_mean = x[fit_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = y[fit_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    rng = np.random.default_rng(SEED)
    projection = (rng.standard_normal((x.shape[1], args.router_dim), dtype=np.float32)
                  / np.sqrt(np.float32(x.shape[1]))).astype(np.float32)
    fit_features = (x[fit_mask] - x_mean) @ projection
    centroids = kmeans(fit_features, args.experts, args.kmeans_iterations)
    labels = route(x - x_mean, projection, centroids)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    models = [torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], args.width), torch.nn.SiLU(),
        torch.nn.Linear(args.width, y.shape[1]),
    ).to(device) for _ in range(args.experts)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) for model in models]
    for expert, (model, optimizer) in enumerate(zip(models, optimizers)):
        rows = np.flatnonzero(fit_mask & (labels == expert))
        if not len(rows):
            continue
        for epoch in range(args.epochs):
            order = np.random.default_rng(SEED + expert * 97 + epoch).permutation(rows)
            for start in range(0, len(order), args.batch):
                part = order[start:start + args.batch]
                xb = torch.from_numpy(x[part] - x_mean).to(device)
                yb = torch.from_numpy(y[part] - y_mean).to(device)
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(xb), yb).backward()
                optimizer.step()
    def score(mask: np.ndarray) -> float:
        total = baseline = 0.0
        for expert, model in enumerate(models):
            rows = np.flatnonzero(mask & (labels == expert))
            with torch.no_grad():
                for start in range(0, len(rows), args.batch):
                    part = rows[start:start + args.batch]
                    if not len(part):
                        continue
                    pred = model(torch.from_numpy(x[part] - x_mean).to(device)).float().cpu().numpy() + y_mean
                    total += float(np.square(pred - y[part], dtype=np.float64).sum())
                    baseline += float(np.square(y[part] - y_mean, dtype=np.float64).sum())
        return float(np.sqrt(total / max(baseline, np.finfo(np.float64).tiny)))
    fit_score, heldout_score = score(fit_mask), score(heldout)
    per_expert = x.shape[1] * args.width + args.width + args.width * y.shape[1] + y.shape[1]
    router_params = projection.size + centroids.size + x_mean.size
    stored = args.experts * per_expert + router_params + y_mean.size
    active = per_expert + router_params + y_mean.size
    result = {
        "schema": SCHEMA,
        "status": "OFFLINE_SURFACE_GATE_PASS_RUNTIME_REQUIRED" if heldout_score <= 0.10 else "OFFLINE_SURFACE_GATE_FAILED",
        "dataset_sha256": dataset_hash,
        "device": str(device),
        "architecture": {"experts": args.experts, "width": args.width, "activation": "silu", "router": "fixed_random_projection_fit_only_kmeans_nearest_centroid", "router_dim": args.router_dim, "kmeans_iterations": args.kmeans_iterations, "top_k": 1, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": SEED},
        "score": {"fit_normalized_rmse": fit_score, "heldout_normalized_rmse": heldout_score},
        "routing": {"fit_rows_per_expert": [int((fit_mask & (labels == i)).sum()) for i in range(args.experts)], "heldout_rows_per_expert": [int((heldout & (labels == i)).sum()) for i in range(args.experts)]},
        "physical": {"stored_parameters": int(stored), "prospective_fp16_artifact_bytes": int(stored * 2), "active_parameters_per_token": int(active), "prospective_active_fp16_bytes_per_token": int(active * 2), "executed_macs_per_token": int(x.shape[1] * args.router_dim + 2 * x.shape[1] * args.width), "sequential_matvecs_per_token": 3},
        "runtime_eligibility": "NO: requires emitted artifact, generated-token capability, artifact/runtime parity, and strict matched decode evidence",
        "tps_claim": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
