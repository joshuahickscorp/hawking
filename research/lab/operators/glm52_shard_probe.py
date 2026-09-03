"""Recomposed science module glm52_shard_probe (C-SCI-R1)."""
from __future__ import annotations
import sys as _sys_a1
from pathlib import Path as _Path_a1
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
PROBE_SCHEMA = 'hawking.glm52.shard_weight_probe.v2'
CHUNK_ELEMENTS = 64 << 20

def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Exact widening: BF16 is the high 16 bits of the IEEE-754 FP32 pattern."""
    return (raw.astype(np.uint32) << 16).view(np.float32)
_ALL_PATTERNS = np.arange(65536, dtype=np.uint16)
with np.errstate(invalid='ignore'):
    _BF16_VALUE = _bf16_to_f32(_ALL_PATTERNS).astype(np.float64)
_BF16_EXPONENT = (_ALL_PATTERNS >> 7 & 255).astype(np.int64)
_BF16_FINITE = np.isfinite(_BF16_VALUE)
_BF16_IS_ZERO = _ALL_PATTERNS & 32767 == 0
_BF16_IS_NEGATIVE = _ALL_PATTERNS >> 15 == 1
_BF16_VALUE_SAFE = np.where(_BF16_FINITE, _BF16_VALUE, 0.0)
_BF16_VALUE_SAFE_SQ = np.square(_BF16_VALUE_SAFE)

def _entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy of an exact value histogram, in bits per element."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    nonzero = counts[counts > 0].astype(np.float64) / total
    return float(-(nonzero * np.log2(nonzero)).sum())

def _stats_from_histogram(counts: np.ndarray) -> dict:
    """Exact tensor statistics reduced from the complete BF16 value histogram."""
    total = int(counts.sum())
    if total == 0:
        return {'elements': 0}
    finite_counts = np.where(_BF16_FINITE, counts, 0)
    finite_total = int(finite_counts.sum())
    present = finite_counts > 0
    accum = float((finite_counts * _BF16_VALUE_SAFE).sum())
    accum_sq = float((finite_counts * _BF16_VALUE_SAFE_SQ).sum())
    mean = accum / finite_total if finite_total else 0.0
    variance = max(accum_sq / finite_total - mean * mean, 0.0) if finite_total else 0.0
    values_present = _BF16_VALUE[present]
    exponent_hist = np.bincount(_BF16_EXPONENT, weights=counts.astype(np.float64), minlength=256).astype(np.int64)
    occupied = np.nonzero(exponent_hist)[0]
    return {'elements': total, 'min': float(values_present.min()) if values_present.size else 0.0, 'max': float(values_present.max()) if values_present.size else 0.0, 'absmax': float(np.abs(values_present).max()) if values_present.size else 0.0, 'mean': mean, 'std': math.sqrt(variance), 'rms': math.sqrt(accum_sq / finite_total) if finite_total else 0.0, 'zero_fraction': float(counts[_BF16_IS_ZERO].sum()) / total, 'negative_fraction': float(counts[_BF16_IS_NEGATIVE].sum()) / total, 'nonfinite_count': total - finite_total, 'zeroth_order_entropy_bits': _entropy_bits(counts), 'exponent_histogram': exponent_hist.tolist(), 'exponent_span_log2': int(occupied.max() - occupied.min()) if occupied.size else 0}

def _probe_f32(handle, start: int, elements: int) -> dict:
    """Direct chunked path for the few F32 control tensors; no 2^32 table is possible."""
    handle.seek(start)
    exponent_hist = np.zeros(256, dtype=np.int64)
    total = zeros = negatives = nonfinite = 0
    accum = accum_sq = 0.0
    minimum, maximum, absmax = (math.inf, -math.inf, 0.0)
    remaining = elements
    while remaining > 0:
        take = min(CHUNK_ELEMENTS, remaining)
        raw = handle.read(take * 4)
        if len(raw) != take * 4:
            raise ValueError(f'short read: want {take * 4}, got {len(raw)}')
        values = np.frombuffer(raw, dtype=np.float32)
        bits = values.view(np.uint32)
        exponent_hist += np.bincount((bits >> 23 & 255).astype(np.int64), minlength=256)
        zeros += int(np.count_nonzero(values == 0))
        negatives += int(np.count_nonzero(np.signbit(values)))
        finite = values[np.isfinite(values)]
        nonfinite += int(values.size - finite.size)
        if finite.size:
            wide = finite.astype(np.float64)
            accum += float(wide.sum())
            accum_sq += float(np.square(wide).sum())
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
            absmax = max(absmax, float(np.abs(finite).max()))
        total += take
        remaining -= take
    mean = accum / total if total else 0.0
    variance = max(accum_sq / total - mean * mean, 0.0) if total else 0.0
    occupied = np.nonzero(exponent_hist)[0]
    return {'elements': total, 'min': minimum if total else 0.0, 'max': maximum if total else 0.0, 'absmax': absmax, 'mean': mean, 'std': math.sqrt(variance), 'rms': math.sqrt(accum_sq / total) if total else 0.0, 'zero_fraction': zeros / total if total else 0.0, 'negative_fraction': negatives / total if total else 0.0, 'nonfinite_count': nonfinite, 'zeroth_order_entropy_bits': None, 'exponent_histogram': exponent_hist.tolist(), 'exponent_span_log2': int(occupied.max() - occupied.min()) if occupied.size else 0}

def probe_tensor(handle, row: dict) -> dict:
    """Exact statistics for one tensor, read at its sealed absolute byte range."""
    start = int(row['absolute_start'])
    payload = int(row['payload_bytes'])
    dtype = row['dtype']
    if dtype == 'BF16':
        elements = payload // 2
        handle.seek(start)
        counts = np.zeros(65536, dtype=np.int64)
        remaining = elements
        while remaining > 0:
            take = min(CHUNK_ELEMENTS, remaining)
            raw = handle.read(take * 2)
            if len(raw) != take * 2:
                raise ValueError(f'short read for {row['name']}: want {take * 2}, got {len(raw)}')
            counts += np.bincount(np.frombuffer(raw, dtype=np.uint16), minlength=65536)
            remaining -= take
        stats = _stats_from_histogram(counts)
    else:
        stats = _probe_f32(handle, start, payload // 4)
    return {'name': row['name'], 'category': row['category'], 'section': row['section'], 'layer': row['layer'], 'expert': row['expert'], 'indexshare_group': row['indexshare_group'], 'budget_class': row['provisional_budget_class'], 'dtype': dtype, 'shape': row['shape'], **stats}

def probe_shard(shard_path: Path, rows: list[dict]) -> dict:
    """Probe every sealed tensor in one resident shard, in sealed offset order."""
    ordered = sorted(rows, key=lambda r: int(r['absolute_start']))
    started = time.time()
    with open(shard_path, 'rb', buffering=0) as handle:
        tensors = [probe_tensor(handle, row) for row in ordered]
    elapsed = time.time() - started
    scored = [t for t in tensors if t.get('zeroth_order_entropy_bits') is not None]
    scored_elements = sum((t['elements'] for t in scored))
    return {'schema': PROBE_SCHEMA, 'shard': shard_path.name, 'tensor_count': len(tensors), 'elements': sum((t['elements'] for t in tensors)), 'shard_zeroth_order_entropy_bits_per_weight': sum((t['zeroth_order_entropy_bits'] * t['elements'] for t in scored)) / scored_elements if scored_elements else None, 'probe_seconds': round(elapsed, 2), 'probed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'tensors': tensors}
