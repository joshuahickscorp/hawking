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
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402
import gravity_format  # noqa: E402

PACK_SCHEMA = "hawking.glm52.compact_tensor.v1"
MAGIC = b"GLM52CPK"
# gravity_forge bills exactly this much for per-artifact metadata, so the container header
# is fixed at the same size and the on-disk file can match the ledger to the byte.
HEADER_BYTES = forge._METADATA_BYTES


def index_bits(cardinality: int) -> int:
    """Billed index width, identical to ByteLedger.add_index."""
    return max(1, math.ceil(math.log2(max(2, cardinality))))


def pack_indices(indices: np.ndarray, bits: int) -> bytes:
    """Bit-pack indices at exactly the billed width, most-significant bit first."""
    flat = np.ascontiguousarray(indices, dtype=np.uint64).ravel()
    if flat.size and int(flat.max()) >= (1 << bits):
        raise ValueError(f"index {int(flat.max())} does not fit in {bits} bits")
    spread = (flat[:, None] >> np.arange(bits - 1, -1, -1, dtype=np.uint64)) & np.uint64(1)
    return np.packbits(spread.ravel().astype(np.uint8)).tobytes()


def unpack_indices(raw: bytes, count: int, bits: int) -> np.ndarray:
    """Inverse of :func:`pack_indices`."""
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[: count * bits]
    grid = unpacked.reshape(count, bits).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(bits - 1, -1, -1, dtype=np.uint64))
    return (grid * weights).sum(axis=1)


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


def pack_tensor_ladder(weights: np.ndarray, *, ladder=LADDER, seed: int = 0) -> list[dict]:
    """Run every admissible ladder rung on one tensor while its bytes are resident.

    The source streams past once, so all rungs are measured in the single visit rather than
    re-fetching per rate.  Rungs whose fixed costs cannot amortize over this tensor are
    skipped rather than emitted as illegal artifacts.  Returns one metrics row per admitted
    rung; no rung is written here.
    """
    rows = []
    for rung in ladder:
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
            "relative_frobenius_error": forge._rel_error(weights, artifact.recon),
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
               production_rung: str = PRODUCTION_RUNG, seed: int = 0) -> dict:
    """Pack every tensor of one resident shard into the accumulating compact artifact.

    One binary blob and one index per source shard, so the compact artifact grows at 282
    files rather than 59,585.  Protected control tensors are carried at source precision
    and billed honestly at 16 BPW rather than quietly excluded from the denominator.
    """
    import glm52_shard_probe as probe

    out_dir.mkdir(parents=True, exist_ok=True)
    # "model-00007-of-00282.safetensors" -> "model-00007-of-00282.gravity"
    gravity_path = out_dir / (shard_path.name.replace(".safetensors", "") + ".gravity")
    ordered = sorted(rows, key=lambda r: int(r["absolute_start"]))

    entries = []
    payloads: list[tuple[dict, bytes]] = []
    compact_bits = 0
    total_weights = 0
    offset = 0
    with open(shard_path, "rb", buffering=0) as source:
        for row in ordered:
            if row["dtype"] != "BF16":
                continue  # F32 control tensors ride the protected path
            source.seek(int(row["absolute_start"]))
            raw = source.read(int(row["payload_bytes"]))
            weights = probe._bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(
                row["shape"]).astype(np.float32)
            total_weights += weights.size

            if row["provisional_budget_class"] == PROTECTED_BUDGET_CLASS:
                compact_bits += weights.size * 16
                entries.append({"name": row["name"], "category": row["category"],
                                "terminal_state": "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES",
                                "billed_bpw": 16.0, "elements": int(weights.size)})
                continue

            ladder_rows = pack_tensor_ladder(weights, seed=seed)
            chosen = next((r for r in ladder_rows
                           if r["rung"] == production_rung and r["admitted"]), None)
            if chosen is None:  # no admissible rung: protect rather than exceed the ceiling
                compact_bits += weights.size * 16
                entries.append({"name": row["name"], "category": row["category"],
                                "terminal_state": "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES",
                                "billed_bpw": 16.0, "elements": int(weights.size),
                                "reason": "NO_ADMISSIBLE_LADDER_RUNG"})
                continue

            payload = serialize(chosen["artifact"])
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

    packed_weights = sum(int(d["elements"]) for d, _ in payloads)
    # Write through a temporary name and rename.  A .gravity is proof a body was
    # consumed, and the streamer treats any file with the right name as packed --
    # so a pack killed mid-write would leave a truncated artifact that reads as
    # complete and authorizes eviction of the BF16 source.  Rename is atomic, which
    # makes a partial .gravity impossible rather than merely unlikely.
    partial_path = gravity_path.with_name(gravity_path.name + ".partial")
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
            "packed_bpw": (sum(len(b) for _, b in payloads) * 8 / max(1, packed_weights)),
            "whole_shard_bpw": compact_bits / max(1, total_weights),
            "protected_budget_class": PROTECTED_BUDGET_CLASS,
            "evidence_level": "F0_PHYSICAL_AND_F1_WEIGHT_SPACE_PROXY_ONLY",
            "not_evidence_of": "output divergence, capability, or end-to-end behaviour",
        },
        shard={"source": shard_path.name, "of": 282})
    with open(partial_path, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial_path, gravity_path)

    return {
        "schema": "hawking.glm52.compact_shard_index.v1",
        "shard": shard_path.name, "gravity": gravity_path.name,
        "production_rung": production_rung,
        "tensors": len(entries), "weights": total_weights,
        "compact_bytes": offset,
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
