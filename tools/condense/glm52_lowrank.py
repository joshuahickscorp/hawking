#!/usr/bin/env python3
"""A low-rank expert representation, billed exactly, to test family against rate.

The pilot closed product quantization on GLM-5.2's expert path and the allocation probe
showed no legal split can rescue it, so the open question is whether a materially
different representation does better at the same budget.  Low rank is the cheapest such
test: it is not a codebook, it makes a different assumption about where the structure is,
and it costs the same to bill.

A rank-r factorization of an m x n tensor stores r*(m+n) values.  At float16 the rate is

    bpw = r * (m + n) * 16 / (m * n)

so for a 2048 x 6144 routed expert, one rank costs 0.0104 bits per weight and the 0.7396
budget that G0 spent on codebooks and indices buys rank 71 instead.

Truncation is by randomized range finding rather than a full SVD: a dense SVD of a
2048 x 6144 tensor is minutes, and 782 of them per layer is not a pilot.  The range
finder is two matmuls and a small SVD, and at rank 71 with oversampling it recovers the
leading subspace closely enough that the comparison is about the family, not the solver.

    selftest
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MAGIC = b"GLM52LRK"
# Same fixed metadata charge the PQ path bills, so the two families are compared on equal
# accounting rather than on who hid their header better.
HEADER_BYTES = 64
OVERSAMPLE = 8
POWER_ITERATIONS = 1


def rank_for_rate(rows: int, cols: int, target_bpw: float) -> int:
    """The largest rank whose exact serialized rate stays at or under the target."""
    per_rank_bits = (rows + cols) * 16
    budget_bits = target_bpw * rows * cols - HEADER_BYTES * 8
    return max(1, int(budget_bits // per_rank_bits))


def billed_bpw(rows: int, cols: int, rank: int) -> float:
    return (rank * (rows + cols) * 16 + HEADER_BYTES * 8) / (rows * cols)


def truncate(weights: np.ndarray, rank: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Randomized range finding, returning factors U [m,r] and V [r,n] with W ~= U @ V."""
    rows, cols = weights.shape
    rank = min(rank, min(rows, cols))
    generator = np.random.default_rng(seed)
    sketch = generator.standard_normal((cols, rank + OVERSAMPLE)).astype(np.float32)
    basis = weights @ sketch
    for _ in range(POWER_ITERATIONS):
        # One power iteration sharpens the subspace when the spectrum decays slowly, which
        # is exactly the regime where a naive sketch would flatter low rank.
        basis = weights @ (weights.T @ basis)
    q, _ = np.linalg.qr(basis)
    projected = q.T @ weights
    u, s, vt = np.linalg.svd(projected, full_matrices=False)
    u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    left = (q @ u) * s
    return left.astype(np.float16), vt.astype(np.float16)


def serialize(left: np.ndarray, right: np.ndarray, *, rows: int, cols: int) -> bytes:
    """Exactly what is billed: a fixed header plus both factors at float16."""
    header = struct.pack("<8sIIII", MAGIC, rows, cols, int(left.shape[1]), 0)
    header = header + b"\x00" * (HEADER_BYTES - len(header))
    return header + left.tobytes() + right.tobytes()


def deserialize(blob: bytes) -> tuple[np.ndarray, np.ndarray, int, int]:
    magic, rows, cols, rank, _ = struct.unpack("<8sIIII", blob[:24])
    if magic != MAGIC:
        raise ValueError("not a GLM52 low-rank payload")
    left_count = rows * rank
    body = blob[HEADER_BYTES:]
    left = np.frombuffer(body[: left_count * 2], dtype=np.float16).reshape(rows, rank)
    right = np.frombuffer(body[left_count * 2:], dtype=np.float16).reshape(rank, cols)
    return left, right, rows, cols


def reconstruct(blob: bytes) -> np.ndarray:
    left, right, _rows, _cols = deserialize(blob)
    return (left.astype(np.float32) @ right.astype(np.float32))


def pack_tensor(weights: np.ndarray, target_bpw: float, *, seed: int = 0) -> dict:
    """Fit, serialize, and report the rate the file actually costs."""
    rows, cols = weights.shape
    rank = rank_for_rate(rows, cols, target_bpw)
    left, right = truncate(weights, rank, seed=seed)
    blob = serialize(left, right, rows=rows, cols=cols)
    observed = len(blob) * 8 / weights.size
    recon = (left.astype(np.float32) @ right.astype(np.float32))
    error = float(np.linalg.norm(recon - weights) / max(np.linalg.norm(weights), 1e-12))
    return {"blob": blob, "rank": rank, "bpw": observed,
            "predicted_bpw": billed_bpw(rows, cols, rank),
            "relative_frobenius_error": error}


def selftest() -> int:
    rng = np.random.default_rng(0)

    # The rate must be what the formula says, on the real expert shape.
    weights = rng.standard_normal((2048, 6144)).astype(np.float32)
    packed = pack_tensor(weights, 0.7396)
    assert packed["bpw"] <= 0.7396 + 1e-9, packed["bpw"]
    assert abs(packed["bpw"] - packed["predicted_bpw"]) < 1e-9, packed
    assert packed["rank"] == rank_for_rate(2048, 6144, 0.7396), packed["rank"]

    # And the file must decode to what was fitted, within float16 storage.
    dense = reconstruct(packed["blob"])
    assert dense.shape == weights.shape
    left, right, _, _ = deserialize(packed["blob"])
    assert left.dtype == np.float16 and right.dtype == np.float16

    # On a genuinely low-rank matrix the family must be near exact, or the solver is
    # broken and any comparison against it would be a comparison against a bug.
    rank = 40
    low = (rng.standard_normal((2048, rank)).astype(np.float32)
           @ rng.standard_normal((rank, 6144)).astype(np.float32))
    fitted = pack_tensor(low, billed_bpw(2048, 6144, rank + 4))
    assert fitted["relative_frobenius_error"] < 0.02, fitted["relative_frobenius_error"]

    # On full-rank noise it must not pretend: a rank-71 fit of a 2048-rank gaussian keeps
    # only a small share of the energy, and claiming otherwise would be the tell.
    assert packed["relative_frobenius_error"] > 0.8, packed["relative_frobenius_error"]

    print("glm52_lowrank selftest OK: rank at G0 rate =", packed["rank"],
          "bpw =", round(packed["bpw"], 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
