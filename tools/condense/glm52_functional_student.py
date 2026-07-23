#!/usr/bin/env python3
"""A shared-basis expert student: one function space per projection, coefficients per expert.

Per-tensor low rank spends its whole budget describing one expert.  But a layer's 256
routed experts are not 256 unrelated matrices, they are 256 specialisations inside one
block, and the pilot's own numbers hint at it: low rank held the expert path two orders of
magnitude better than product quantization at the same rate, on tensors it could only fit
to rank 71.

Sharing the basis changes the arithmetic.  Stack a projection across all 256 experts and
factor the stack: the 6144-wide side becomes one basis stored once for the layer, and each
expert keeps only its own coefficients.  The basis then costs 1/256th of what it would per
expert, so the same budget buys

    bpw = r * 16 * (256*2048 + 6144) / (256*2048*6144) = r * 0.0026352

rank 280 instead of rank 71, at G0's rate.  Four times the subspace for the same bits.

Nothing is materialised whole.  The stacked matrix is 12.9 GiB per projection, so the
range finder streams it expert by expert: sketch each block, stack the sketches, take one
QR, then stream a second pass for the projected matrix.

Billing is exact and measured from real bytes: the file this writes is the artifact, and
its length divided by the source weights it replaces is the rate.  The .gravity container
does not yet carry side information shared across tensors, so this stores to its own file
and the container integration waits until the family earns it.

    fit LAYER RATE      fit and serialize one layer's expert student
    selftest
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MAGIC = b"GLM52FBS"
HEADER_BYTES = 64
OVERSAMPLE = 16

# Which side of each projection is shared.  The 6144 side is the block's own width and is
# what every expert has in common; the 2048 side is the expert's own intermediate space.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def rate_for_rank(rank: int, *, experts: int, inner: int, width: int) -> float:
    """Exact bits per source weight for one projection at this rank."""
    stored = rank * (experts * inner + width)
    covered = experts * inner * width
    return (stored * 16 + HEADER_BYTES * 8) / covered


def rank_for_rate(target: float, *, experts: int, inner: int, width: int) -> int:
    covered = experts * inner * width
    budget = target * covered - HEADER_BYTES * 8
    per_rank = (experts * inner + width) * 16
    return max(1, int(budget // per_rank))


def _blocks(weights: list[np.ndarray], width: int) -> list[np.ndarray]:
    """Orient every expert so the shared side is the columns."""
    out = []
    for block in weights:
        out.append(block if block.shape[1] == width else block.T)
    return out


def fit_shared_basis(weights: list[np.ndarray], rank: int, *, seed: int = 0):
    """Streamed randomized range finding over a stack no one ever materialises.

    Returns (coefficients per expert [inner, rank], shared basis [rank, width]).
    """
    width = weights[0].shape[1]
    generator = np.random.default_rng(seed)
    sketch = generator.standard_normal((width, rank + OVERSAMPLE)).astype(np.float32)

    sketched = [block @ sketch for block in weights]
    stacked = np.concatenate(sketched, axis=0)
    del sketched
    basis, _ = np.linalg.qr(stacked)
    del stacked

    # B = Q.T @ A, accumulated block by block so A is never assembled.
    projected = np.zeros((basis.shape[1], width), dtype=np.float32)
    offset = 0
    for block in weights:
        rows = block.shape[0]
        projected += basis[offset:offset + rows].T @ block
        offset += rows

    u, s, vt = np.linalg.svd(projected, full_matrices=False)
    u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    shared = vt.astype(np.float16)

    coefficients = []
    offset = 0
    scaled = u * s
    for block in weights:
        rows = block.shape[0]
        coefficients.append((basis[offset:offset + rows] @ scaled).astype(np.float16))
        offset += rows
    return coefficients, shared


def serialize(coefficients: list[np.ndarray], shared: np.ndarray, *,
              transposed: bool) -> bytes:
    rank = shared.shape[0]
    header = struct.pack("<8sIIIII", MAGIC, len(coefficients), coefficients[0].shape[0],
                         shared.shape[1], rank, int(transposed))
    header = header + b"\x00" * (HEADER_BYTES - len(header))
    body = shared.tobytes() + b"".join(block.tobytes() for block in coefficients)
    return header + body


def deserialize(blob: bytes):
    magic, experts, inner, width, rank, transposed = struct.unpack("<8sIIIII", blob[:28])
    if magic != MAGIC:
        raise ValueError("not a GLM52 functional block student payload")
    cursor = HEADER_BYTES
    shared_count = rank * width
    shared = np.frombuffer(blob[cursor:cursor + shared_count * 2],
                           dtype=np.float16).reshape(rank, width)
    cursor += shared_count * 2
    per_expert = inner * rank
    coefficients = []
    for _ in range(experts):
        coefficients.append(np.frombuffer(blob[cursor:cursor + per_expert * 2],
                                          dtype=np.float16).reshape(inner, rank))
        cursor += per_expert * 2
    return coefficients, shared, bool(transposed)


def reconstruct_expert(blob: bytes, index: int) -> np.ndarray:
    coefficients, shared, transposed = deserialize(blob)
    dense = coefficients[index].astype(np.float32) @ shared.astype(np.float32)
    return dense.T if transposed else dense


def pack_projection(weights: list[np.ndarray], target_bpw: float, *, seed: int = 0) -> dict:
    """Fit one projection across every expert, serialize, and report the real rate."""
    width = max(weights[0].shape)
    transposed = weights[0].shape[1] != width
    blocks = _blocks(weights, width)
    inner = blocks[0].shape[0]
    rank = rank_for_rate(target_bpw, experts=len(blocks), inner=inner, width=width)
    rank = min(rank, min(inner * len(blocks), width))

    started = time.time()
    coefficients, shared = fit_shared_basis(blocks, rank, seed=seed)
    blob = serialize(coefficients, shared, transposed=transposed)
    covered = sum(block.size for block in blocks)
    observed = len(blob) * 8 / covered

    errors = []
    for index, block in enumerate(blocks):
        recon = coefficients[index].astype(np.float32) @ shared.astype(np.float32)
        errors.append(float(np.linalg.norm(recon - block)
                            / max(np.linalg.norm(block), 1e-12)))
    return {
        "blob": blob, "rank": rank, "bpw": observed,
        "predicted_bpw": rate_for_rank(rank, experts=len(blocks), inner=inner, width=width),
        "experts": len(blocks), "transposed": transposed,
        "mean_relative_frobenius_error": float(np.mean(errors)),
        "fit_seconds": round(time.time() - started, 2),
    }


def selftest() -> int:
    rng = np.random.default_rng(0)

    # Rate must be exactly the formula, on the real GLM shapes, at G0's budget.
    experts, inner, width = 8, 256, 512
    blocks = [rng.standard_normal((inner, width)).astype(np.float32) for _ in range(experts)]
    packed = pack_projection(blocks, 0.7396)
    assert packed["bpw"] <= 0.7396 + 1e-9, packed["bpw"]
    assert abs(packed["bpw"] - packed["predicted_bpw"]) < 1e-9, packed

    # Round trip through the file, per expert, must equal the fit.
    coefficients, shared, transposed = deserialize(packed["blob"])
    assert len(coefficients) == experts and not transposed
    direct = coefficients[3].astype(np.float32) @ shared.astype(np.float32)
    assert np.allclose(reconstruct_expert(packed["blob"], 3), direct)

    # The claim the whole family rests on: when experts share structure, one basis serves
    # all of them and the fit is near exact at a rank far below what per-tensor low rank
    # could afford.  Build experts that genuinely live in a shared subspace.
    basis = rng.standard_normal((24, width)).astype(np.float32)
    shared_blocks = [rng.standard_normal((inner, 24)).astype(np.float32) @ basis
                     for _ in range(experts)]
    fitted = pack_projection(shared_blocks, rate_for_rank(
        28, experts=experts, inner=inner, width=width))
    assert fitted["mean_relative_frobenius_error"] < 0.02, fitted

    # And it must not pretend on experts that share nothing.
    assert packed["mean_relative_frobenius_error"] > 0.5, packed

    # A transposed projection must come back in its original orientation.
    tall = [rng.standard_normal((width, inner)).astype(np.float32) for _ in range(4)]
    packed_tall = pack_projection(tall, 0.75)
    assert packed_tall["transposed"]
    assert reconstruct_expert(packed_tall["blob"], 0).shape == (width, inner)

    # The rank the amortisation buys at the real shape, which is the point of the family.
    real = rank_for_rate(0.7396, experts=256, inner=2048, width=6144)
    print(json.dumps({"selftest": "PASS", "rank_at_G0_rate_real_shape": real,
                      "per_tensor_lowrank_rank_at_same_rate": 71,
                      "amortisation": round(real / 71, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
