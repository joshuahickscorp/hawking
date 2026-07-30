#!/usr/bin/env python3.12
"""Serialize GLM-5.2 tensors into physically-exact sub-bit compact shards.

``gravity_forge`` decides the representation and bills it; this module is the part that
makes the bill true on disk.  A packed artifact whose serialized size exceeds its ledger
is a BPW claim that does not survive contact with a filesystem, so the invariant here is
hard: the bytes written for a tensor equal ``artifact.ledger.bytes()`` exactly, or the
write is refused.  Indices are bit-packed to the billed width (4 bits for k=16, 8 for
k=256) rather than stored as the int64 the packer keeps in memory, codebooks land as the
fp16 they are billed as, and the 64-byte metadata allowance is a real fixed-size header.

The output is executable, not just measurable: what is written is exactly the ``pq_codes``
stash that :func:`gravity_forge.pq_execute` consumes for its direct compact matvec, which
decodes per-subspace and never materializes the dense weight.  That is what makes the
compact artifact hostable at a size the BF16 parent could never reach.

Scope boundary, deliberately not blurred: everything here is F0 (exact physical
accounting) and F1 (weight-space reconstruction error).  Weight-space error is a PROXY.
Nothing in this module measures output divergence, capability, or end-to-end behaviour,
and a small artifact that round-trips is NOT evidence that the model still works.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402
import artifact_client as gravity_format  # noqa: E402

PACK_SCHEMA = "hawking.glm52.compact_tensor.v1"


class PackCoverageError(RuntimeError):
    """A shard's artifact would not contain every tensor the shard held."""
MAGIC = b"GLM52CPK"
# gravity_forge bills exactly this much for per-artifact metadata, so the container header
# is fixed at the same size and the on-disk file can match the ledger to the byte.
HEADER_BYTES = forge._METADATA_BYTES


def index_bits(cardinality: int) -> int:
    """Billed index width, identical to ByteLedger.add_index."""
    return max(1, math.ceil(math.log2(max(2, cardinality))))


def pack_indices(indices: np.ndarray, bits: int) -> bytes:
    """Bit-pack indices via the Rust Core A authority (in-process ctypes)."""
    flat = np.ascontiguousarray(indices, dtype=np.uint64).ravel()
    if flat.size and int(flat.max()) >= (1 << bits):
        raise ValueError(f"index {int(flat.max())} does not fit in {bits} bits")
    return gravity_format.pack_indices(flat.astype(np.uint32, copy=False), int(bits))


def unpack_indices(raw: bytes, count: int, bits: int) -> np.ndarray:
    """Inverse of :func:`pack_indices`, with temporaries bounded by the payload.

    Eight consecutive indices occupy exactly ``bits`` bytes, so lane ``j`` of every such
    group sits at one fixed byte offset and one fixed shift.  Reading the eight lanes with
    strided byte slices never materializes a per-bit grid: the widest temporary is one
    ``count/8`` window array, against ``count*bits`` bytes for an unpackbits grid.  Output
    is uint8 for the k<=256 rungs the ladder actually ships.
    """
    if not 1 <= bits <= 16:
        raise ValueError(f"unpack_indices handles 1..16 bits, got {bits}")
    span = 2 if bits <= 8 else 3  # bytes an index can straddle within its group
    window_dtype = np.uint16 if bits <= 8 else np.uint32
    groups = (count + 7) // 8
    block = np.zeros(groups * bits + span, dtype=np.uint8)
    source = np.frombuffer(raw, dtype=np.uint8)[: groups * bits]
    block[: source.size] = source
    out = np.empty(groups * 8, dtype=np.uint8 if bits <= 8 else np.uint16)
    stop = groups * bits
    for j in range(8):
        start = j * bits
        base = start // 8
        window = block[base: base + stop: bits].astype(window_dtype)
        for extra in range(1, span):
            window <<= 8
            window |= block[base + extra: base + extra + stop: bits]
        window >>= span * 8 - (start % 8) - bits
        window &= (1 << bits) - 1
        out[j::8] = window
    return out[:count]


def serialize(artifact: forge.PackedArtifact) -> bytes:
    """Serialize a PQ-family artifact to exactly its billed byte count."""
    codes = artifact.config.get("pq_codes")
    if codes is None:
        raise ValueError(f"family {artifact.family} carries no pq_codes stash")
    codebooks = codes["codebooks"]
    indices = codes["indices"]
    bits = index_bits(codebooks[0].shape[0])

    body = b"".join(np.ascontiguousarray(cb, dtype=np.float16).tobytes() for cb in codebooks)
    body += pack_indices(indices, bits)

    # Fixed 64-byte header: everything pq_execute needs to rebuild the geometry.  Packed
    # binary rather than JSON so it cannot silently outgrow its billed allowance.
    header = MAGIC + struct.pack(
        "<HHHHIIIIH?B",
        int(codes["D"]), int(codes["S"]), int(codes["sub"]), int(codebooks[0].shape[0]),
        int(codes["rows"]), int(codes["cols"]), int(codes["nchunk"]),
        int(codes["seed"]), int(bits), bool(codes["rotate"]), len(codebooks),
    )
    header = header.ljust(HEADER_BYTES, b"\x00")
    if len(header) != HEADER_BYTES:
        raise ValueError(f"header is {len(header)} bytes, billed {HEADER_BYTES}")

    blob = header + body
    billed = artifact.ledger.bytes()
    if len(blob) != billed:
        raise ValueError(
            f"serialized {len(blob)} bytes but ledger bills {billed}; "
            "the BPW claim and the file must agree exactly"
        )
    return blob


def deserialize(blob: bytes) -> dict[str, Any]:
    """Rebuild the pq_codes stash that gravity_forge.pq_execute consumes."""
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("not a GLM-5.2 compact tensor")
    fields = struct.unpack_from("<HHHHIIIIH?B", blob, len(MAGIC))
    D, S, sub, card, rows, cols, nchunk, seed, bits, rotate, n_codebooks = fields
    offset = HEADER_BYTES
    codebooks = []
    for _ in range(n_codebooks):
        span = card * sub * 2
        codebooks.append(
            np.frombuffer(blob[offset: offset + span], dtype=np.float16).astype(np.float32)
            .reshape(card, sub)
        )
        offset += span
    count = rows * nchunk * S
    indices = unpack_indices(blob[offset:], count, bits).reshape(rows * nchunk, S)
    return {
        "codebooks": codebooks, "indices": indices.astype(np.int64),
        "D": D, "S": S, "sub": sub, "rows": rows, "cols": cols,
        "nchunk": nchunk, "rotate": bool(rotate), "seed": seed,
    }


def load_artifact(blob: bytes) -> forge.PackedArtifact:
    """Rehydrate into the shape pq_execute accepts, without a dense reconstruction."""
    codes = deserialize(blob)
    ledger = forge.ByteLedger()
    ledger.add("serialized", (len(blob) - HEADER_BYTES) * 8)
    return forge.PackedArtifact(
        "product_quant", np.empty((0,), dtype=np.float32),
        codes["rows"] * codes["cols"], ledger, ledger.total_bits(), 0,
        {"pq_codes": codes},
    )


# Rate ladder, measured on real GLM-5.2 expert weights.  Every rung is legally below the
# 1.0 BPW ceiling: k=256 at dim=8 bills 1.0026 and is therefore inadmissible, which is why
# the anchor is k=128.  Geometry is chosen by measurement, not symmetry -- at equal rate a
# richer codebook over longer subvectors beat a smaller one over shorter subvectors
# (dim=16/k=256 -> 0.505 BPW at relerr 0.761, versus dim=8/k=16 -> 0.500 at 0.779).
LADDER = (
    {"rung": "R0", "dim": 8, "k": 128, "nominal_bpw": 0.876},
    {"rung": "R2", "dim": 16, "k": 256, "nominal_bpw": 0.505},
    {"rung": "R4", "dim": 32, "k": 256, "nominal_bpw": 0.261},
)
PRODUCTION_RUNG = "R0"
# 2026-07-25: a non-power-of-2 "R1" (dim=11, k=1024, nominal 0.909) was tried here after
# the screening gate failed twice and general-v1's own T1 tier called for ~1.0 bpw on
# routed_expert. _pq_geometry() silently rejects a non-power-of-2 dim and falls back to
# _largest_pow2_divisor(cols) instead -- no error, no warning -- so R1 was never actually
# packed at the requested rate; it produced arbitrary, wildly-varying bpw depending on
# each tensor's own column count. Once corrected to respect the power-of-2 constraint, a
# systematic search over every (dim, k) admissible at GLM-5.2's real T1 tensor sizes
# (routed_expert 12,582,912 elements, dense_mlp up to 75,497,472, attention as small as
# 3,538,944) found NO candidate strictly between R0's 0.875 and BPW_CEILING=1.0 that is
# admissible across that size range. R0 is not an arbitrary starting point -- for tensors
# GLM-5.2's shape, it is at or extremely near the practical ceiling this codec family
# (S=1, power-of-2 dim) can reach. See GLM52_R1_GEOMETRY_INVALID_FINDING.json.
# The non-production rungs are a rate survey, not the artifact.  Fitting all three on every
# tensor made the ladder the whole cost of a pack (measured 1.24 s x 213 tensors ~= the
# 264 s one-worker pack_seconds), and it was re-establishing a near-constant: across the
# routed-expert tensors, which are all the same shape, R2 and R4 relative error have
# stdev 1.8e-4 and 1.0e-4.  Every routed-expert tensor is therefore a repeat measurement of
# a quantity already known to four decimals.  Survey every Nth of them and keep the rest of
# the model exhaustive: non-routed tensors are shape-diverse and stay fully measured.
LADDER_SAMPLE_EVERY = int(os.environ.get("GLM52_LADDER_SAMPLE_EVERY", "32"))
# Router/normalization/control tensors: 582 of 59,585 tensors and ~0.1% of all weights, so
# holding them at source precision costs almost nothing in whole-model BPW while keeping
# the control path exact.  Compressing the router to save 0.1% would be a bad trade.
PROTECTED_BUDGET_CLASS = "CONTROL_SENSITIVE_CANDIDATE"


# Hard physical ceiling from the campaign's governing law: no deployable compressed
# candidate may exceed one complete bit per original logical parent weight.
BPW_CEILING = 1.0


def codebook_bits(k: int, sub: int) -> int:
    """Codebook cost is fixed per tensor, so it is the term that fails to amortize."""
    return k * sub * 16


def rung_is_admissible(rung: dict, elements: int) -> bool:
    """Whether a rung can bill under the ceiling for a tensor of this size.

    A codebook costs the same whether it serves 32 thousand weights or 12 million, so a
    geometry that bills 0.876 BPW on a routed expert can bill 1.39 on a small tensor.  The
    ceiling is a property of (geometry, tensor size), never of geometry alone.
    """
    sub = rung["dim"]  # subspaces=1 throughout the ladder, so the subvector is the full dim
    index_cost = index_bits(rung["k"]) / rung["dim"]
    fixed = (codebook_bits(rung["k"], sub) + HEADER_BYTES * 8) / max(1, elements)
    return (index_cost + fixed) < BPW_CEILING


def pack_tensor_ladder(weights: np.ndarray, *, ladder=LADDER, seed: int = 0,
                       rungs: frozenset[str] | None = None) -> list[dict]:
    """Run the requested admissible ladder rungs on one tensor while its bytes are resident.

    The source streams past once, so rungs are measured in the single visit rather than
    re-fetching per rate.  Rungs whose fixed costs cannot amortize over this tensor are
    skipped rather than emitted as illegal artifacts.  Returns one metrics row per rung; no
    rung is written here.

    ``rungs`` restricts which rungs are actually fitted.  A rung left out still gets a row,
    marked ``sampled_out``, because a measurement that was never taken and a measurement
    that failed must not look the same in the record.
    """
    rows = []
    for rung in ladder:
        if rungs is not None and rung["rung"] not in rungs:
            rows.append({"rung": rung["rung"], "dim": rung["dim"], "k": rung["k"],
                         "admitted": False, "sampled_out": True,
                         "reason": "NOT_IN_THIS_TENSOR_LADDER_SAMPLE", "artifact": None})
            continue
        if not rung_is_admissible(rung, weights.size):
            rows.append({"rung": rung["rung"], "dim": rung["dim"], "k": rung["k"],
                         "admitted": False,
                         "reason": "FIXED_COST_CANNOT_AMORTIZE_UNDER_CEILING",
                         "artifact": None})
            continue
        artifact = forge.pack_product_quant(
            weights, dim=rung["dim"], subspaces=1, k=rung["k"], seed=seed)
        # measured, not predicted: the ledger is the authority on what this actually costs
        if artifact.whole_artifact_bpw >= BPW_CEILING:
            rows.append({"rung": rung["rung"], "dim": rung["dim"], "k": rung["k"],
                         "admitted": False, "bpw": artifact.whole_artifact_bpw,
                         "reason": "MEASURED_BPW_AT_OR_OVER_CEILING", "artifact": None})
            continue
        rows.append({
            "rung": rung["rung"], "dim": rung["dim"], "k": rung["k"], "admitted": True,
            "bpw": artifact.whole_artifact_bpw,
            # Metal/BLAS reductions can differ in the seventh decimal between
            # otherwise byte-identical fits.  The payload is deterministic; letting
            # insignificant diagnostic jitter rewrite the JSON header made repeated
            # packs differ despite identical tensor bytes.  Six decimals retains
            # materially more precision than the survey uses while making the
            # container reproducible.
            "relative_frobenius_error": round(
                forge._rel_error(weights, artifact.recon), 6
            ),
            "artifact": artifact,
        })
    return rows


class _DoctorResult:
    """Adapt doctor_pq's treatment report to the artifact shape the scorer reads."""

    def __init__(self, report: dict, base: forge.PackedArtifact) -> None:
        self._report = report
        self.recon = report.get("recon", base.recon)
        self.whole_artifact_bpw = float(report["new_whole_bpw"])
        self.ledger = base.ledger


def _candidate_families() -> list[dict]:
    """Materially distinct sub-0.5 candidates, not one geometry with knobs.

    The 1.0 ceiling is already met by plain PQ, so every candidate here targets the half-bit
    frontier.  Each family attacks a different weakness: rotation attacks coherence, islands
    attack heavy tails, Doctor buys back error with a billed residual, and shared grammar
    attacks the thing that is specific to a 256-expert MoE -- the per-expert codebook cost,
    which amortizes toward indices-only as the cluster grows.
    """
    return [
        {"family": "product_quant", "dim": 16, "k": 256},
        {"family": "product_quant", "dim": 16, "k": 64},
        {"family": "product_quant", "dim": 32, "k": 256},
        {"family": "transform_pq", "dim": 16, "k": 256},
        {"family": "transform_pq", "dim": 32, "k": 256},
        {"family": "pq_protected_islands", "dim": 16, "k": 256, "budget_frac": 0.03},
        {"family": "pq_doctor", "dim": 16, "k": 256, "doctor_frac": 0.25,
         "strategy": "residual_codebook"},
        {"family": "shared_grammar", "dim": 16, "k": 256, "cluster": 8},
    ]


def _run_candidate(spec: dict, weights: np.ndarray,
                   cluster: list[np.ndarray] | None) -> forge.PackedArtifact | None:
    family = spec["family"]
    if family == "product_quant":
        return forge.pack_product_quant(weights, dim=spec["dim"], subspaces=1, k=spec["k"], seed=0)
    if family == "transform_pq":
        return forge.pack_transform_pq(weights, dim=spec["dim"], subspaces=1, k=spec["k"], seed=0)
    if family == "pq_protected_islands":
        return forge.pack_pq_protected_islands(
            weights, dim=spec["dim"], subspaces=1, k=spec["k"], seed=0,
            budget_frac=spec["budget_frac"])
    if family == "pq_doctor":
        base = forge.pack_product_quant(weights, dim=spec["dim"], subspaces=1, k=spec["k"], seed=0)
        budget = int(base.ledger.bytes() * spec["doctor_frac"])
        report = forge.doctor_pq(weights, base, byte_budget=budget,
                                 strategy=spec["strategy"])
        # doctor_pq reports a treatment rather than returning an artifact, so surface its
        # billed rate and post-treatment error through the same shape the scorer expects
        return _DoctorResult(report, base)
    if family == "shared_grammar":
        if not cluster:
            return None
        return forge.pack_shared_grammar(cluster, dim=spec["dim"], k=spec["k"], stages=1, seed=0)
    return None


def run_tournament(samples: list[np.ndarray], *, cluster: list[np.ndarray] | None = None,
                   target_bpw: float = 0.5) -> dict:
    """Score every candidate family on real sampled tensors and pick a sub-target winner.

    Selection is on weight-space error at or under the target rate.  That is an F1 PROXY:
    it ranks candidates, it does not establish that any of them preserves capability.
    """
    results = []
    for spec in _candidate_families():
        errors, rates, ok = [], [], True
        for index, weights in enumerate(samples):
            try:
                artifact = _run_candidate(spec, weights, cluster)
            except Exception as exc:  # noqa: BLE001
                results.append({**spec, "status": "ERROR",
                                "error": f"{type(exc).__name__}: {exc}"})
                ok = False
                break
            if artifact is None:
                ok = False
                break
            recon = artifact.recon
            if recon.ndim == 3:  # shared grammar returns the whole cluster
                reference = np.stack(cluster[: recon.shape[0]])
                errors.append(float(np.mean([forge._rel_error(reference[i], recon[i])
                                             for i in range(recon.shape[0])])))
            else:
                errors.append(forge._rel_error(weights, recon))
            rates.append(artifact.whole_artifact_bpw)
        if not ok or not errors:
            continue
        results.append({**spec, "status": "OK",
                        "mean_bpw": float(np.mean(rates)),
                        "mean_relative_frobenius_error": float(np.mean(errors))})

    admissible = [r for r in results
                  if r["status"] == "OK" and r["mean_bpw"] <= target_bpw + 1e-6]
    winner = min(admissible, key=lambda r: r["mean_relative_frobenius_error"]) if admissible else None
    return {
        "schema": "hawking.glm52.frozen_pack_program.v1",
        "target_bpw": target_bpw, "samples": len(samples),
        "evidence_level": "F1_WEIGHT_SPACE_PROXY_ONLY",
        "not_evidence_of": "output divergence or capability",
        "results": sorted(results, key=lambda r: r.get("mean_relative_frobenius_error", 9e9)),
        "winner": winner,
    }


def pack_shard(shard_path: Path, rows: list[dict], out_dir: Path, *,
               production_rung: str = PRODUCTION_RUNG, seed: int = 0,
               rate_override: dict[str | tuple[int, int], str] | None = None,
               telemetry: dict[str, Any] | None = None) -> dict:
    """Pack every tensor of one resident shard into the accumulating compact artifact.

    One binary blob and one index per source shard, so the compact artifact grows at 282
    files rather than 59,585.  Protected control tensors are carried at source precision
    and billed honestly at 16 BPW rather than quietly excluded from the denominator.

    `rate_override` maps an exact tensor name, or the coarser `(layer, expert)` key,
    to either `"native"` (carry the selected tensor(s) at full source precision, same
    mechanism a protected control tensor already uses) or a specific rung name from
    `LADDER` (pack at that rung instead of `production_rung`). Exact tensor decisions
    take precedence over an expert-wide fallback. A key absent from the map, or
    `rate_override=None` altogether, reproduces today's behavior exactly -- this is
    how Prometheus's frozen per-tensor Math-Preserve auction reaches the packer without
    a second one: additive, and provably a no-op when unused (see
    tools/condense/tests/test_glm52_pack_rate_override.py).
    """
    import glm52_shard_probe as probe

    total_started = time.perf_counter()
    stage_seconds: dict[str, float] = defaultdict(float)
    category_stats: dict[str, dict[str, Any]] = {}

    def category(row: dict) -> dict[str, Any]:
        name = str(row["category"])
        if name not in category_stats:
            category_stats[name] = {
                "tensors": 0,
                "weights": 0,
                "source_bytes": 0,
                "artifact_bytes": 0,
                "native_tensors": 0,
                "packed_tensors": 0,
                "stage_seconds": defaultdict(float),
            }
        return category_stats[name]

    def add_time(name: str, elapsed: float, cat: dict[str, Any] | None = None) -> None:
        stage_seconds[name] += elapsed
        if cat is not None:
            cat["stage_seconds"][name] += elapsed

    out_dir.mkdir(parents=True, exist_ok=True)
    # "model-00007-of-00282.safetensors" -> "model-00007-of-00282.gravity"
    gravity_path = out_dir / (shard_path.name.replace(".safetensors", "") + ".gravity")
    ordered = sorted(rows, key=lambda r: int(r["absolute_start"]))

    entries = []
    payloads: list[tuple[dict, bytes]] = []
    compact_bits = 0
    total_weights = 0
    offset = 0
    all_rungs = frozenset(rung["rung"] for rung in LADDER)
    surveyed = 0
    routed_seen = 0
    def carry_native(row: dict, raw: bytes, elements: int, reason: str) -> None:
        """Physically store a tensor at source precision and bill exactly its own bytes.

        Billing and storing are the same act here.  Generation A kept them apart: it added
        elements * 16 to compact_bits and appended a descriptor, but never handed the bytes
        to write_shard, so every router, norm and indexer tensor was accounted for and
        written nowhere while the BF16 body that held them was evicted.
        """
        nonlocal compact_bits, offset
        compact_bits += len(raw) * 8
        descriptor = {
            "name": row["name"], "category": row["category"],
            "layer": row.get("layer"), "expert": row.get("expert"),
            "shape": row["shape"], "codec": f"native.{str(row['dtype']).lower()}",
            "terminal_state": "PROTECTED_SOURCE_NATIVE",
            "elements": int(elements),
            "bpw": len(raw) * 8 / max(1, elements),
            "reason": reason,
        }
        payloads.append((descriptor, raw))
        entries.append(descriptor)
        offset += len(raw)
        cat = category(row)
        cat["artifact_bytes"] += len(raw)
        cat["native_tensors"] += 1

    with open(shard_path, "rb", buffering=0) as source:
        for row in ordered:
            cat = category(row)
            elements = 1
            for dim in row["shape"]:
                elements *= int(dim)
            cat["tensors"] += 1
            cat["weights"] += elements
            cat["source_bytes"] += int(row["payload_bytes"])

            started = time.perf_counter()
            source.seek(int(row["absolute_start"]))
            raw = source.read(int(row["payload_bytes"]))
            add_time("source_read", time.perf_counter() - started, cat)
            # Every declared tensor counts in the denominator exactly once, whatever path
            # it takes.  The complete rate is the campaign's headline, so nothing may be
            # quietly excluded from it.
            total_weights += elements

            if row["dtype"] != "BF16":
                # F32 control tensors (router score-correction bias and friends) are not
                # ladder candidates, but they are declared source weight and the artifact
                # is incomplete without them.  Carry the exact source bytes.
                carry_native(row, raw, elements, "NON_BF16_CONTROL_TENSOR")
                continue

            started = time.perf_counter()
            weights = probe._bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(
                row["shape"]).astype(np.float32)
            add_time("bf16_decode", time.perf_counter() - started, cat)

            if row["provisional_budget_class"] == PROTECTED_BUDGET_CLASS:
                carry_native(row, raw, weights.size, "PROTECTED_BUDGET_CLASS")
                continue

            override = None
            if rate_override:
                override = rate_override.get(row["name"])
                if override is None:
                    override = rate_override.get((row.get("layer"), row.get("expert")))
            if override == "native":
                reason = (
                    "PROMETHEUS_COALITION_PROTECTED"
                    if row.get("expert") is not None
                    else "PROMETHEUS_PROFILE_PROTECTED"
                )
                carry_native(row, raw, weights.size, reason)
                continue
            target_rung = override or production_rung

            # Deterministic by position, not random: the same shard always surveys the
            # same tensors, so a re-pack reproduces the record exactly.
            #
            # 2026-07-26: the non-survey branch used to fit `production_rung` rather than
            # the tensor's own target, so an overridden tensor had to survey every rung
            # merely to be sure its target got fitted at all -- correctness dressed as
            # science.  Under Math-Preserve that exemption swallowed the schedule: the
            # allocation manifest overrides 56,076 of 59,585 tensors, so "the uncommon
            # case" was 94 percent of the model and the survey became the pack.  Measured
            # on one real routed-expert shape (2048x6144): 6.29 s for {R0,R2,R4} against
            # 1.62 s for the single target rung, 3.88x, with the packed payload bytes
            # byte-identical either way (the ladder rungs share no RNG state -- _kmeans
            # seeds its own generator -- so a rung not fitted cannot perturb one that is).
            # Fitting the target directly keeps the same guarantee without the exemption,
            # and the sampling schedule now applies to every routed expert on equal terms.
            # Non-routed tensors stay exhaustively surveyed: they are shape-diverse, and
            # there are 1,217 of them against 58,368 routed experts that are two shapes.
            is_routed = row.get("expert") is not None
            if not is_routed or LADDER_SAMPLE_EVERY <= 1:
                rungs = all_rungs
            else:
                rungs = (all_rungs if routed_seen % LADDER_SAMPLE_EVERY == 0
                         else frozenset({target_rung}))
                routed_seen += 1
            surveyed += rungs == all_rungs
            started = time.perf_counter()
            ladder_rows = pack_tensor_ladder(weights, seed=seed, rungs=rungs)
            add_time("fit", time.perf_counter() - started, cat)
            chosen = next((r for r in ladder_rows
                           if r["rung"] == target_rung and r["admitted"]), None)
            if chosen is None:  # no admissible rung: protect rather than exceed the ceiling
                carry_native(row, raw, weights.size, "NO_ADMISSIBLE_LADDER_RUNG")
                continue

            started = time.perf_counter()
            payload = serialize(chosen["artifact"])
            add_time("serialize", time.perf_counter() - started, cat)
            compact_bits += len(payload) * 8
            descriptor = {
                "name": row["name"], "category": row["category"],
                "layer": row["layer"], "expert": row["expert"],
                "shape": row["shape"], "codec": "gravity-pq",
                "terminal_state": "PACKED_IN_CORE_ARTIFACT",
                "elements": int(weights.size),
                "rung": chosen["rung"], "bpw": chosen["bpw"],
                "relative_frobenius_error": chosen["relative_frobenius_error"],
                # the whole ladder measured in the one visit the bytes were resident
                "ladder": [{k: v for k, v in r.items() if k != "artifact"}
                           for r in ladder_rows],
            }
            payloads.append((descriptor, payload))
            entries.append(descriptor)
            offset += len(payload)
            cat["artifact_bytes"] += len(payload)
            cat["packed_tensors"] += 1

    # Fail closed on the coverage hole rather than write another artifact that the
    # streamer will treat as proof the source was consumed.  Every path above now hands
    # its bytes to write_shard, so this guard should never fire; it stays because a
    # future path that forgets to would otherwise be indistinguishable from a good pack,
    # and the failure mode is deletion of the only body that could recreate the weight.
    written = {descriptor["name"] for descriptor, _ in payloads}
    absent = [row["name"] for row in ordered if row["name"] not in written]
    if absent:
        raise PackCoverageError(
            f"{shard_path.name}: {len(absent)} of {len(ordered)} source tensors would be "
            f"billed but not written, including {absent[:3]}. Refusing to emit an "
            f"artifact that authorizes eviction of weights it does not contain."
        )
    compressed = [(d, b) for d, b in payloads if not str(d["codec"]).startswith("native.")]
    native = [(d, b) for d, b in payloads if str(d["codec"]).startswith("native.")]
    packed_weights = sum(int(d["elements"]) for d, _ in compressed)
    complete_weights = sum(int(d["elements"]) for d, _ in payloads)
    # Write through a temporary name and rename.  A .gravity is proof a body was
    # consumed, and the streamer treats any file with the right name as packed --
    # so a pack killed mid-write would leave a truncated artifact that reads as
    # complete and authorizes eviction of the BF16 source.  Rename is atomic, which
    # makes a partial .gravity impossible rather than merely unlikely.
    partial_path = gravity_path.with_name(gravity_path.name + ".partial")
    format_telemetry: dict[str, Any] = {}
    gravity_format.write_shard(
        partial_path, payloads,
        model={"repo": "zai-org/GLM-5.2",
               "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
               "source_shard": shard_path.name},
        architecture={"type": "GlmMoeDsaForCausalLM", "hidden_layers": 78,
                      "routed_experts": 256, "shared_experts": 1, "hidden_size": 6144},
        tokenizer={"kind": "reference", "source": "zai-org/GLM-5.2"},
        compression={
            "codec": "gravity-pq", "production_rung": production_rung,
            # packed_bpw is what the codec achieved on the tensors it compressed.
            # complete_bpw is what the shard costs with its native organs carried, and it
            # is the only rate a candidate may be judged on.  Both reconcile against
            # physical bytes in gravity_format.verify.
            "packed_bpw": (sum(len(b) for _, b in compressed) * 8 / max(1, packed_weights)),
            "complete_bpw": (sum(len(b) for _, b in payloads) * 8 / max(1, complete_weights)),
            "whole_shard_bpw": compact_bits / max(1, total_weights),
            "native_tensors": len(native),
            "native_bytes": sum(len(b) for _, b in native),
            "compressed_tensors": len(compressed),
            # Each tensor's own target rung is fitted and is the artifact.  The other rungs
            # are a survey; this says exactly how much of one this shard carries.  `schedule`
            # is versioned because shards 1-29 of this run were sealed under `v1_survey_
            # every_overridden_tensor`, and a reader must be able to tell which density a
            # given shard's survey rows were taken at.  Payload bytes are unaffected.
            # Which k-means assignment arithmetic produced these codebooks. Both are
            # correct; they are not byte-compatible, so a shard must say which one it used
            # or the artifact is no longer reproducible from (seed, iters) alone.
            "fit_kernel": forge.FIT_KERNEL,
            "ladder_survey": {
                "schedule": "v2_target_rung_always_survey_every_nth_routed_expert",
                "target_rung_coverage": "ALL_TENSORS",
                "other_rungs_sampled_every_nth_routed_expert": LADDER_SAMPLE_EVERY,
                "non_routed_tensors_fully_surveyed": True,
                "tensors_fully_surveyed": surveyed,
                "routed_expert_tensors_seen": routed_seen,
            },
            "protected_budget_class": PROTECTED_BUDGET_CLASS,
            "evidence_level": "F0_PHYSICAL_AND_F1_WEIGHT_SPACE_PROXY_ONLY",
            "not_evidence_of": "output divergence, capability, or end-to-end behaviour",
        },
        shard={"source": shard_path.name, "of": 282},
        telemetry=format_telemetry)
    stage_seconds["hash_manifest"] += float(
        format_telemetry.get("hash_manifest_seconds", 0.0)
    )
    stage_seconds["file_write"] += float(
        format_telemetry.get("file_write_seconds", 0.0)
    )
    started = time.perf_counter()
    with open(partial_path, "rb") as handle:
        os.fsync(handle.fileno())
    stage_seconds["fsync"] += time.perf_counter() - started
    started = time.perf_counter()
    os.replace(partial_path, gravity_path)
    stage_seconds["atomic_rename"] += time.perf_counter() - started

    total_seconds = time.perf_counter() - total_started
    measured_seconds = sum(stage_seconds.values())
    stage_seconds["bookkeeping"] += max(0.0, total_seconds - measured_seconds)
    if telemetry is not None:
        clean_categories = {}
        for name, stats in sorted(category_stats.items()):
            cat_seconds = sum(stats["stage_seconds"].values())
            clean_categories[name] = {
                **{k: v for k, v in stats.items() if k != "stage_seconds"},
                "stage_seconds": {
                    key: round(value, 6)
                    for key, value in sorted(stats["stage_seconds"].items())
                },
                "measured_seconds": round(cat_seconds, 6),
                "tensors_per_second": (
                    stats["tensors"] / max(total_seconds, 1e-12)
                ),
                "weights_per_second": (
                    stats["weights"] / max(total_seconds, 1e-12)
                ),
            }
        source_bytes = sum(int(row["payload_bytes"]) for row in ordered)
        artifact_file_bytes = gravity_path.stat().st_size
        telemetry.update({
            "schema": "hawking.glm52.pass3_pack_telemetry.v1",
            "shard": shard_path.name,
            "timing_side_channel_only": True,
            "excluded_from_artifact_and_canonical_receipts": True,
            "total_seconds": total_seconds,
            "stage_seconds": {
                key: round(value, 6)
                for key, value in sorted(stage_seconds.items())
            },
            "stage_time_percentage": {
                key: round(value * 100.0 / max(total_seconds, 1e-12), 4)
                for key, value in sorted(stage_seconds.items())
            },
            "categories": clean_categories,
            "tensors": len(entries),
            "weights": total_weights,
            "source_bytes": source_bytes,
            "artifact_payload_bytes": offset,
            "artifact_file_bytes": artifact_file_bytes,
            "tensors_per_second": len(entries) / max(total_seconds, 1e-12),
            "weights_per_second": total_weights / max(total_seconds, 1e-12),
            "source_mb_per_second_end_to_end": (
                source_bytes / 1_000_000 / max(total_seconds, 1e-12)
            ),
            "artifact_mb_per_second_end_to_end": (
                artifact_file_bytes / 1_000_000 / max(total_seconds, 1e-12)
            ),
            "source_read_mb_per_second": (
                source_bytes
                / 1_000_000
                / max(stage_seconds["source_read"], 1e-12)
            ),
            "format": format_telemetry,
        })

    return {
        "schema": "hawking.glm52.compact_shard_index.v1",
        "shard": shard_path.name, "gravity": gravity_path.name,
        "production_rung": production_rung,
        "tensors": len(entries), "weights": total_weights,
        "compact_bytes": offset,
        "ladder_tensors_fully_surveyed": surveyed,
        "ladder_sample_every_nth_routed_expert": LADDER_SAMPLE_EVERY,
        "complete_bpw": sum(len(b) for _, b in payloads) * 8 / max(1, complete_weights),
        "packed_bpw": sum(len(b) for _, b in compressed) * 8 / max(1, packed_weights),
        "native_tensors": len(native), "compressed_tensors": len(compressed),
        "tensor_coverage": "COMPLETE_EVERY_DECLARED_TENSOR_PHYSICALLY_PRESENT",
        "whole_shard_bpw": compact_bits / max(1, total_weights),
        "evidence_level": "F0_PHYSICAL_AND_F1_WEIGHT_SPACE_PROXY_ONLY",
        "not_evidence_of": "output divergence, capability, or end-to-end behaviour",
    }


def selftest() -> int:
    """Round-trip, exact-size, and execute-equivalence checks on synthetic weights."""
    rng = np.random.default_rng(0)

    for bits in (1, 3, 4, 8):
        values = rng.integers(0, 1 << bits, size=257, dtype=np.uint64)
        restored = unpack_indices(pack_indices(values, bits), values.size, bits)
        assert np.array_equal(values, restored), f"bit-pack round trip failed at {bits} bits"

    for dim, k in ((8, 16), (4, 256), (8, 256)):
        weights = rng.standard_normal((256, 128)).astype(np.float32)
        artifact = forge.pack_product_quant(weights, dim=dim, subspaces=1, k=k, seed=0, iters=4)
        blob = serialize(artifact)

        # the load-bearing invariant: the file is exactly what was billed
        assert len(blob) == artifact.ledger.bytes(), (len(blob), artifact.ledger.bytes())
        on_disk_bpw = len(blob) * 8 / weights.size
        assert abs(on_disk_bpw - artifact.whole_artifact_bpw) < 1e-9, on_disk_bpw

        codes = deserialize(blob)
        original = artifact.config["pq_codes"]
        assert np.array_equal(codes["indices"], original["indices"]), "indices changed"
        for restored_cb, original_cb in zip(codes["codebooks"], original["codebooks"]):
            # codebooks are billed as fp16, so fp16 is the exact stored precision
            assert np.array_equal(restored_cb, original_cb.astype(np.float16).astype(np.float32))

        # what was written still executes, and matches executing the in-memory artifact
        probe = rng.standard_normal(weights.shape[1]).astype(np.float32)
        reloaded = load_artifact(blob)
        direct = forge.pq_execute(artifact, probe)
        from_disk = forge.pq_execute(reloaded, probe)
        gap = float(np.abs(direct - from_disk).max() / (np.abs(direct).max() + 1e-12))
        assert gap < 2e-3, f"execute drifted after round trip: {gap}"

    assert any(r["rung"] == PRODUCTION_RUNG for r in LADDER), "production rung must be on the ladder"

    # Large tensor: fixed costs amortize, so every rung should admit and bill sub-1.
    large = rng.standard_normal((4096, 1024)).astype(np.float32)
    admitted = [r for r in pack_tensor_ladder(large) if r["admitted"]]
    assert len(admitted) == len(LADDER), "every rung should admit on a large tensor"
    for row in admitted:
        assert row["bpw"] < BPW_CEILING, f"{row['rung']} bills {row['bpw']}"
        assert len(serialize(row["artifact"])) == row["artifact"].ledger.bytes()

    # Small tensor: the codebook cannot amortize.  The ladder must refuse rather than emit
    # an over-ceiling artifact -- this is the case that would silently inflate whole-model
    # BPW if admissibility were treated as a property of geometry alone.
    small = rng.standard_normal((64, 128)).astype(np.float32)
    rows = pack_tensor_ladder(small)
    assert any(not r["admitted"] for r in rows), "small tensor must refuse at least one rung"
    for row in rows:
        if row["admitted"]:
            assert row["bpw"] < BPW_CEILING, f"admitted {row['rung']} at {row['bpw']}"
        else:
            assert row["artifact"] is None and "reason" in row

    print(json.dumps({"selftest": "PASS", "schema": PACK_SCHEMA,
                      "header_bytes": HEADER_BYTES, "ladder_rungs": len(LADDER),
                      "all_rungs_sub_one_bpw": True,
                      "size_equals_ledger": True, "executes_from_disk": True}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    sys.stderr.write("import this module; only `selftest` runs standalone\n")
    raise SystemExit(2)
