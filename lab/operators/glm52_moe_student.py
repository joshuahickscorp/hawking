"""Recomposed science module glm52_moe_student (C-SCI-R1)."""
from __future__ import annotations
import sys as _sys_a1
from pathlib import Path as _Path_a1
import json
import struct
import sys
import time
from pathlib import Path
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
HERE = _A1_CONDENSE
MAGIC = b'GLM52MOE'
HEADER_BYTES = 64
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))

def projection(width: int, hidden: int, seed: int) -> np.ndarray:
    """The first layer, reproduced from the seed rather than stored.

    Scaled by 1/sqrt(width) so the pre-activation variance does not depend on the model
    width, which keeps the same hidden size behaving the same way across layers.
    """
    generator = np.random.default_rng(seed)
    return generator.standard_normal((width, hidden), dtype=np.float32) / np.float32(np.sqrt(width))

def features(x: np.ndarray, seed: int, hidden: int) -> np.ndarray:
    projected = x @ projection(x.shape[1], hidden, seed)
    return silu(projected)

def fit_readout(x: np.ndarray, y: np.ndarray, *, hidden: int, seed: int, ridge_grid=RIDGE_GRID, holdout: float=0.2) -> dict:
    """Solve the readout in closed form, choosing ridge on a split of the fit data only."""
    phi = features(x, seed, hidden)
    count = phi.shape[0]
    cut = int(count * (1.0 - holdout))
    train, validate = (slice(0, cut), slice(cut, count))
    gram = phi[train].T @ phi[train]
    cross = phi[train].T @ y[train]
    eye = np.eye(hidden, dtype=np.float32)
    best = None
    for ridge in ridge_grid:
        readout = np.linalg.solve(gram + np.float32(ridge) * eye, cross)
        residual = phi[validate] @ readout - y[validate]
        error = float(np.linalg.norm(residual) / max(np.linalg.norm(y[validate]), 1e-12))
        if best is None or error < best['validation_relative_error']:
            best = {'ridge': ridge, 'validation_relative_error': error}
    gram = phi.T @ phi
    cross = phi.T @ y
    readout = np.linalg.solve(gram + np.float32(best['ridge']) * eye, cross)
    return {'readout': readout.astype(np.float16), **best}

def serialize(readout: np.ndarray, *, seed: int, width: int) -> bytes:
    hidden, out_width = readout.shape
    header = struct.pack('<8sIIII', MAGIC, width, hidden, out_width, seed)
    header = header + b'\x00' * (HEADER_BYTES - len(header))
    return header + readout.tobytes()

def deserialize(blob: bytes):
    magic, width, hidden, out_width, seed = struct.unpack('<8sIIII', blob[:24])
    if magic != MAGIC:
        raise ValueError('not a GLM52 MoE student payload')
    readout = np.frombuffer(blob[HEADER_BYTES:], dtype=np.float16).reshape(hidden, out_width)
    return (readout, int(width), int(hidden), int(seed))

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

def fit(x: np.ndarray, y: np.ndarray, *, hidden: int, seed: int, replaced_weights: int) -> dict:
    started = time.time()
    solved = fit_readout(x, y, hidden=hidden, seed=seed)
    blob = serialize(solved['readout'], seed=seed, width=x.shape[1])
    predicted = apply_student(blob, x)
    error = float(np.linalg.norm(predicted - y) / max(np.linalg.norm(y), 1e-12))
    cosine = float(np.dot(predicted.ravel(), y.ravel()) / max(np.linalg.norm(predicted) * np.linalg.norm(y), 1e-12))
    return {'blob': blob, 'hidden': hidden, 'seed': seed, 'bytes': len(blob), 'bpw': bpw(len(blob), replaced_weights), 'ridge': solved['ridge'], 'validation_relative_error': solved['validation_relative_error'], 'in_sample_relative_error': error, 'in_sample_cosine': cosine, 'samples': int(x.shape[0]), 'parameters_stored': int(solved['readout'].size), 'samples_per_stored_parameter': float(x.shape[0] * y.shape[1] / max(solved['readout'].size, 1)), 'fit_seconds': round(time.time() - started, 2)}
