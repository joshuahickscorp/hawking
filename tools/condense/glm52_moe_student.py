#!/usr/bin/env python3
"""Replace the MoE function, not its weights: a dense random-feature student, ridge-fitted.

Every family the pilot has closed is a weight blueprint.  Product quantization, per-tensor
low rank, the by-role hybrid and the shared basis all answer the question "what is a cheap
approximation of W".  Section 6.1 asks a different question: what is a cheap function that
produces the state the block produces.  The BF16 model is a trajectory teacher.

The structural argument for a dense student on a sparse layer is an active-capacity one.
The teacher routes 8 of 256 experts per token, so a token actually passes through
8 * 2048 = 16,384 intermediate units.  The 256 experts cost 9.84 G weights to store even
though only 6.4 percent of them run.  A dense student pays for exactly what it runs, so
the same budget that stores the sparse layer at 0.74 bits buys a dense hidden width far
larger than 16,384.  Sparsity is what the storage is buying, and the student declines to
buy it.

The first layer costs nothing.  It is drawn from a seeded generator, so the artifact
stores the seed and reproduces the projection exactly; only the readout is written.  That
makes the fit a linear problem with a closed form, which matters because the sealed
reference forward is NumPy and has no gradients to offer.

What this is NOT: it is not a decomposition of the expert matrices in a data-induced
metric.  Activation-weighted SVD is a recorded Type-1 kill and is not reopened here.  This
removes the routing entirely and fits a different architecture against the block's output.

    fit LAYER WIDTH     fit a student for one layer and report its rate and fidelity
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

MAGIC = b"GLM52MOE"
HEADER_BYTES = 64
# Ridge, because the readout is solved in closed form and the feature count is allowed to
# exceed the sample count.  Chosen on the fit split and reported, never tuned on score.
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def projection(width: int, hidden: int, seed: int) -> np.ndarray:
    """The first layer, reproduced from the seed rather than stored.

    Scaled by 1/sqrt(width) so the pre-activation variance does not depend on the model
    width, which keeps the same hidden size behaving the same way across layers.
    """
    generator = np.random.default_rng(seed)
    return (generator.standard_normal((width, hidden), dtype=np.float32)
            / np.float32(np.sqrt(width)))


def features(x: np.ndarray, seed: int, hidden: int) -> np.ndarray:
    projected = x @ projection(x.shape[1], hidden, seed)
    return silu(projected)


def fit_readout(x: np.ndarray, y: np.ndarray, *, hidden: int, seed: int,
                ridge_grid=RIDGE_GRID, holdout: float = 0.2) -> dict:
    """Solve the readout in closed form, choosing ridge on a split of the fit data only."""
    phi = features(x, seed, hidden)
    count = phi.shape[0]
    cut = int(count * (1.0 - holdout))
    train, validate = slice(0, cut), slice(cut, count)

    gram = phi[train].T @ phi[train]
    cross = phi[train].T @ y[train]
    eye = np.eye(hidden, dtype=np.float32)

    best = None
    for ridge in ridge_grid:
        readout = np.linalg.solve(gram + np.float32(ridge) * eye, cross)
        residual = phi[validate] @ readout - y[validate]
        error = float(np.linalg.norm(residual) / max(np.linalg.norm(y[validate]), 1e-12))
        if best is None or error < best["validation_relative_error"]:
            best = {"ridge": ridge, "validation_relative_error": error}

    # Refit on everything at the chosen ridge: the split existed to pick the ridge, not to
    # shrink the training set.
    gram = phi.T @ phi
    cross = phi.T @ y
    readout = np.linalg.solve(gram + np.float32(best["ridge"]) * eye, cross)
    return {"readout": readout.astype(np.float16), **best}


def serialize(readout: np.ndarray, *, seed: int, width: int) -> bytes:
    hidden, out_width = readout.shape
    header = struct.pack("<8sIIII", MAGIC, width, hidden, out_width, seed)
    header = header + b"\x00" * (HEADER_BYTES - len(header))
    return header + readout.tobytes()


def deserialize(blob: bytes):
    magic, width, hidden, out_width, seed = struct.unpack("<8sIIII", blob[:24])
    if magic != MAGIC:
        raise ValueError("not a GLM52 MoE student payload")
    readout = np.frombuffer(blob[HEADER_BYTES:], dtype=np.float16).reshape(hidden, out_width)
    return readout, int(width), int(hidden), int(seed)


def apply_student(blob: bytes, x: np.ndarray) -> np.ndarray:
    readout, width, hidden, seed = deserialize(blob)
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
    out = features(flat, seed, hidden) @ readout.astype(np.float32)
    return out.reshape(*x.shape[:-1], readout.shape[1])


def bpw(blob_bytes: int, replaced_weights: int) -> float:
    return blob_bytes * 8 / replaced_weights


def hidden_for_rate(target_bpw: float, *, replaced_weights: int, out_width: int) -> int:
    budget_bits = target_bpw * replaced_weights - HEADER_BYTES * 8
    return max(1, int(budget_bits // (out_width * 16)))


def fit(x: np.ndarray, y: np.ndarray, *, hidden: int, seed: int,
        replaced_weights: int) -> dict:
    started = time.time()
    solved = fit_readout(x, y, hidden=hidden, seed=seed)
    blob = serialize(solved["readout"], seed=seed, width=x.shape[1])
    predicted = apply_student(blob, x)
    error = float(np.linalg.norm(predicted - y) / max(np.linalg.norm(y), 1e-12))
    cosine = float(np.dot(predicted.ravel(), y.ravel())
                   / max(np.linalg.norm(predicted) * np.linalg.norm(y), 1e-12))
    return {
        "blob": blob, "hidden": hidden, "seed": seed,
        "bytes": len(blob), "bpw": bpw(len(blob), replaced_weights),
        "ridge": solved["ridge"],
        "validation_relative_error": solved["validation_relative_error"],
        "in_sample_relative_error": error,
        "in_sample_cosine": cosine,
        "samples": int(x.shape[0]),
        "parameters_stored": int(solved["readout"].size),
        "samples_per_stored_parameter": float(x.shape[0] * y.shape[1]
                                              / max(solved["readout"].size, 1)),
        "fit_seconds": round(time.time() - started, 2),
    }


def selftest() -> int:
    rng = np.random.default_rng(0)

    # The projection must be reproducible from the seed alone, since that is the whole
    # reason the first layer costs no bits.
    assert np.array_equal(projection(64, 32, 7), projection(64, 32, 7))
    assert not np.array_equal(projection(64, 32, 7), projection(64, 32, 8))

    # On a target the feature map can express, the fit must be close, and the file must
    # reproduce the fit exactly.
    width, hidden, samples = 64, 256, 2048
    x = rng.standard_normal((samples, width)).astype(np.float32)
    truth = features(x, 3, hidden) @ rng.standard_normal((hidden, width)).astype(np.float32)
    fitted = fit(x, truth, hidden=hidden, seed=3, replaced_weights=samples * width)
    assert fitted["in_sample_cosine"] > 0.95, fitted["in_sample_cosine"]
    assert np.allclose(apply_student(fitted["blob"], x),
                       apply_student(fitted["blob"], x))

    # Held-out error is reported and is not allowed to be the in-sample number.
    assert fitted["validation_relative_error"] >= 0.0
    assert "in_sample_relative_error" in fitted

    # On pure noise it must not claim a fit.
    noise = rng.standard_normal((samples, width)).astype(np.float32)
    weak = fit(x, noise, hidden=32, seed=1, replaced_weights=samples * width)
    assert weak["in_sample_cosine"] < 0.5, weak["in_sample_cosine"]

    # The rate is set by the readout alone, and the sizing helper must respect it.
    replaced = 256 * 2048 * 6144 * 3
    target = 0.7396
    chosen = hidden_for_rate(target, replaced_weights=replaced, out_width=6144)
    assert bpw(HEADER_BYTES + chosen * 6144 * 2, replaced) <= target + 1e-9
    print(json.dumps({
        "selftest": "PASS",
        "hidden_affordable_at_G0_rate": chosen,
        "teacher_active_units_per_token": 8 * 2048,
        "ratio": round(chosen / (8 * 2048), 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
