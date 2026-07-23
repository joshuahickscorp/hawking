#!/usr/bin/env python3.12
"""One-pass weight evidence capture for a resident GLM-5.2 BF16 source shard.

The source streams past exactly once and a refetch costs the full wire time, so every
shard is mined for the cheap statistics the later representation work needs before its
body is evicted.  Storage is negligible against the shards themselves: roughly 2 KB per
tensor against a 5.34 GB shard.

Reads only at the byte offsets already sealed in GLM52_SHARD_DEPENDENCY_GRAPH.json, so
this parses no safetensors header and invents no taxonomy -- the campaign's own tensor
classification (category, layer, expert, indexshare group, budget class) is carried
straight through onto every row.

The whole probe rests on one property: BF16 has only 65536 distinct values, so an exact
value histogram is a *complete* summary of the tensor.  Every statistic below is then
derived by reducing over 65536 bins instead of over billions of weights, which makes the
pass roughly an order of magnitude cheaper than widening the data to FP32, removes float
accumulation-order error entirely, and makes the result exactly reproducible regardless
of chunk size.  The per-chunk cost is a single ``bincount``.

Captured per tensor, and why it earns the CPU:

* zeroth-order entropy in bits/weight, from the exact value histogram -- what a perfect
  context-free entropy coder achieves on this tensor, which is the empirical reference
  point every sub-1-BPW rate claim has to be argued against;
* exponent histogram (256 bins) and log2 span -- the magnitude distribution driving G3
  asymmetric organ allocation and G6 codebook design;
* min/max/absmax/mean/std/rms, zero and negative fractions -- outlier and dynamic-range
  evidence for D1 protected directions.

F32 tensors (76 router weights out of 59,585) cannot use a 2^32 table and take a direct
chunked path instead.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROBE_SCHEMA = "hawking.glm52.shard_weight_probe.v2"
# 64M elements = 128 MB of uint16 in flight per worker, so several probe workers stay well
# inside the RAM the campaign is allowed to touch.  Results are chunk-size invariant.
CHUNK_ELEMENTS = 64 << 20


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Exact widening: BF16 is the high 16 bits of the IEEE-754 FP32 pattern."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


# Every BF16 value, and the exponent each bit pattern belongs to.  Built once; all
# per-tensor statistics are reductions of a histogram against these tables.
_ALL_PATTERNS = np.arange(65536, dtype=np.uint16)
with np.errstate(invalid="ignore"):  # 2048 of the 65536 patterns are NaN/Inf by definition
    _BF16_VALUE = _bf16_to_f32(_ALL_PATTERNS).astype(np.float64)
_BF16_EXPONENT = ((_ALL_PATTERNS >> 7) & 0xFF).astype(np.int64)
_BF16_FINITE = np.isfinite(_BF16_VALUE)
_BF16_IS_ZERO = (_ALL_PATTERNS & 0x7FFF) == 0
_BF16_IS_NEGATIVE = (_ALL_PATTERNS >> 15) == 1
# NaN times a zero count is still NaN, which would poison the moment sums, so arithmetic
# uses a table with the non-finite patterns replaced by an exact zero contribution.
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
        return {"elements": 0}
    finite_counts = np.where(_BF16_FINITE, counts, 0)
    finite_total = int(finite_counts.sum())
    present = finite_counts > 0

    accum = float((finite_counts * _BF16_VALUE_SAFE).sum())
    accum_sq = float((finite_counts * _BF16_VALUE_SAFE_SQ).sum())
    mean = accum / finite_total if finite_total else 0.0
    variance = max(accum_sq / finite_total - mean * mean, 0.0) if finite_total else 0.0

    values_present = _BF16_VALUE[present]
    exponent_hist = np.bincount(_BF16_EXPONENT, weights=counts.astype(np.float64),
                                minlength=256).astype(np.int64)
    occupied = np.nonzero(exponent_hist)[0]

    return {
        "elements": total,
        "min": float(values_present.min()) if values_present.size else 0.0,
        "max": float(values_present.max()) if values_present.size else 0.0,
        "absmax": float(np.abs(values_present).max()) if values_present.size else 0.0,
        "mean": mean,
        "std": math.sqrt(variance),
        "rms": math.sqrt(accum_sq / finite_total) if finite_total else 0.0,
        "zero_fraction": float(counts[_BF16_IS_ZERO].sum()) / total,
        "negative_fraction": float(counts[_BF16_IS_NEGATIVE].sum()) / total,
        "nonfinite_count": total - finite_total,
        "zeroth_order_entropy_bits": _entropy_bits(counts),
        "exponent_histogram": exponent_hist.tolist(),
        "exponent_span_log2": int(occupied.max() - occupied.min()) if occupied.size else 0,
    }


def _probe_f32(handle, start: int, elements: int) -> dict:
    """Direct chunked path for the few F32 control tensors; no 2^32 table is possible."""
    handle.seek(start)
    exponent_hist = np.zeros(256, dtype=np.int64)
    total = zeros = negatives = nonfinite = 0
    accum = accum_sq = 0.0
    minimum, maximum, absmax = math.inf, -math.inf, 0.0
    remaining = elements
    while remaining > 0:
        take = min(CHUNK_ELEMENTS, remaining)
        raw = handle.read(take * 4)
        if len(raw) != take * 4:
            raise ValueError(f"short read: want {take * 4}, got {len(raw)}")
        values = np.frombuffer(raw, dtype=np.float32)
        bits = values.view(np.uint32)
        exponent_hist += np.bincount(((bits >> 23) & 0xFF).astype(np.int64), minlength=256)
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
    return {
        "elements": total,
        "min": minimum if total else 0.0, "max": maximum if total else 0.0, "absmax": absmax,
        "mean": mean, "std": math.sqrt(variance),
        "rms": math.sqrt(accum_sq / total) if total else 0.0,
        "zero_fraction": zeros / total if total else 0.0,
        "negative_fraction": negatives / total if total else 0.0,
        "nonfinite_count": nonfinite,
        "zeroth_order_entropy_bits": None,  # a 2^32 value histogram is not worth building
        "exponent_histogram": exponent_hist.tolist(),
        "exponent_span_log2": int(occupied.max() - occupied.min()) if occupied.size else 0,
    }


def probe_tensor(handle, row: dict) -> dict:
    """Exact statistics for one tensor, read at its sealed absolute byte range."""
    start = int(row["absolute_start"])
    payload = int(row["payload_bytes"])
    dtype = row["dtype"]

    if dtype == "BF16":
        elements = payload // 2
        handle.seek(start)
        counts = np.zeros(65536, dtype=np.int64)
        remaining = elements
        while remaining > 0:
            take = min(CHUNK_ELEMENTS, remaining)
            raw = handle.read(take * 2)
            if len(raw) != take * 2:
                raise ValueError(f"short read for {row['name']}: want {take * 2}, got {len(raw)}")
            counts += np.bincount(np.frombuffer(raw, dtype=np.uint16), minlength=65536)
            remaining -= take
        stats = _stats_from_histogram(counts)
    else:
        stats = _probe_f32(handle, start, payload // 4)

    return {
        "name": row["name"], "category": row["category"], "section": row["section"],
        "layer": row["layer"], "expert": row["expert"],
        "indexshare_group": row["indexshare_group"],
        "budget_class": row["provisional_budget_class"],
        "dtype": dtype, "shape": row["shape"],
        **stats,
    }


def probe_shard(shard_path: Path, rows: list[dict]) -> dict:
    """Probe every sealed tensor in one resident shard, in sealed offset order."""
    ordered = sorted(rows, key=lambda r: int(r["absolute_start"]))
    started = time.time()
    with open(shard_path, "rb", buffering=0) as handle:
        tensors = [probe_tensor(handle, row) for row in ordered]
    elapsed = time.time() - started
    scored = [t for t in tensors if t.get("zeroth_order_entropy_bits") is not None]
    scored_elements = sum(t["elements"] for t in scored)
    return {
        "schema": PROBE_SCHEMA,
        "shard": shard_path.name,
        "tensor_count": len(tensors),
        "elements": sum(t["elements"] for t in tensors),
        "shard_zeroth_order_entropy_bits_per_weight": (
            sum(t["zeroth_order_entropy_bits"] * t["elements"] for t in scored) / scored_elements
            if scored_elements else None
        ),
        "probe_seconds": round(elapsed, 2),
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tensors": tensors,
    }


def selftest() -> int:
    """Exactness checks on a synthetic buffer with known values.  No campaign files touched."""
    import tempfile

    known = np.array([1.0, -2.0, 0.0, 0.5, 4.0, -0.25, 0.0, 8.0], dtype=np.float32)
    bits = (known.view(np.uint32) >> 16).astype(np.uint16)  # exact: these values fit BF16
    row = {
        "name": "synthetic.weight", "category": "routed_expert", "section": "main_text",
        "layer": 3, "expert": 7, "indexshare_group": None,
        "provisional_budget_class": "COMPRESSIBLE_CANDIDATE", "dtype": "BF16",
        "shape": [2, 4], "absolute_start": 128, "payload_bytes": bits.nbytes,
    }

    def _probe(chunk: int) -> dict:
        global CHUNK_ELEMENTS
        original = CHUNK_ELEMENTS
        CHUNK_ELEMENTS = chunk
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "synthetic.safetensors"
                path.write_bytes(b"\x00" * 128 + bits.tobytes())
                with open(path, "rb", buffering=0) as handle:
                    return probe_tensor(handle, row)
        finally:
            CHUNK_ELEMENTS = original

    got = _probe(1 << 20)
    assert got["elements"] == 8, got["elements"]
    assert got["max"] == 8.0 and got["min"] == -2.0, (got["min"], got["max"])
    assert got["absmax"] == 8.0, got["absmax"]
    assert got["zero_fraction"] == 0.25, got["zero_fraction"]
    assert got["negative_fraction"] == 0.25, got["negative_fraction"]
    # reference in float64: the histogram path accumulates in float64, so comparing against
    # numpy's float32 reduction would be testing numpy's rounding, not this module
    exact = known.astype(np.float64)
    assert abs(got["mean"] - float(exact.mean())) < 1e-12, got["mean"]
    assert abs(got["std"] - float(exact.std())) < 1e-12, got["std"]
    assert got["nonfinite_count"] == 0

    # six distinct values, one appearing twice out of eight -> known exact entropy
    expected = -sum(p * math.log2(p) for p in [2 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
    assert abs(got["zeroth_order_entropy_bits"] - expected) < 1e-12, got["zeroth_order_entropy_bits"]

    chunked = _probe(3)  # force many chunks over the same bytes
    assert chunked == got, "chunk size changed the result"

    print(json.dumps({"selftest": "PASS", "entropy_bits": got["zeroth_order_entropy_bits"],
                      "chunk_invariant": True, "schema": PROBE_SCHEMA}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    sys.stderr.write("import this module; only `selftest` runs standalone\n")
    raise SystemExit(2)
