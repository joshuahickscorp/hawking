#!/usr/bin/env python3.12
"""FLOP-and-byte ledger: separate what compression bought in bytes from what it bought in arithmetic.

Mandate section 9 (FLOP paradigm) and section 7.2 (active-byte ledger).  The distinction
this module exists to make auditable is simple and unflattering: a representation that
executes the same dense multiply-accumulate count as its BF16 parent bought *bytes*, not
*arithmetic*.  Gravity's R0 decode-FMA kernel is exactly that case -- it reads 7.1% of the
BF16 weight bytes and then performs ``rows*nchunk*D == rows*cols`` fused multiply-adds,
which is the dense MAC count to the operation.  Nothing here is allowed to blur that.

Two derived ratios are reported and both are DESCRIPTIVE, never capability claims:

  * ``arithmetic_compression = dense_equivalent_macs / executed_arithmetic``
  * ``byte_compression      = dense_equivalent_bytes / executed_bytes``

They say how much work and how much traffic a representation removed relative to a dense
BF16 reference executing the same mathematical contraction.  They say nothing about output
divergence, perplexity, or whether the compressed model still works.  A representation can
report a byte compression of 18x and be functionally destroyed; F0 physical accounting and
F1 weight-space error do not license any F2+ claim.

Discipline that makes the ledger trustworthy: every field is explicitly set or explicitly
``None``.  ``None`` means UNMEASURED, and an UNMEASURED field poisons every ratio it feeds
rather than silently contributing a zero.  A number that was never measured never becomes
a favourable one by accident.

Analytic cost models for the two grammars are here so that a projected result can be
checked against a measured one at arbitrary ``(rows, cols, D, k)``:

  * decode-FMA (what the Metal kernel does today): decode inside the accumulation, one
    fused multiply-add per original weight.  ``rows*nchunk*D`` MACs.
  * lookup-linear (what the representation permits): precompute ``codebook @ x`` once per
    chunk, then every output row is a gather-and-add over ``nchunk`` table entries.
    ``nchunk*k*D + rows*nchunk`` ops -- 5.33x fewer at the gate/up geometry.

Both models include the per-threadgroup re-read terms that
``gravity_metal.bytes_read_per_matvec`` omits: that function reports the artifact's
indices plus one codebook, but every threadgroup stages its own copy of the codebook and
of ``x``, and the kernel uploads 8-bit indices where R0 bills 7-bit.  At the gate/up
geometry the real device traffic is 1,785,856 B, not the 1,574,912 B it reports.

Roofline constants are the MEASURED ones for this machine (Mac15,14, M3 Ultra, 60 GPU
cores): 736 GB/s sustained read, 17,703 GFLOP/s fp32 FMA.  The 819 GB/s vendor figure is
not this machine's roof and is deliberately absent.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_adapter as adapter  # noqa: E402
import gravity_format  # noqa: E402

LEDGER_SCHEMA = "hawking.gravity.active_byte_ledger.v1"
ROOFLINE_SCHEMA = "hawking.gravity.token_roofline.v1"
RECORD_SCHEMA = "hawking.gravity.flop_ledger_record.v1"

# ---------------------------------------------------------------------------
# MEASURED machine constants.  Source: this campaign's own microbenchmarks on
# Mac15,14 / Apple M3 Ultra / 60 GPU cores / 96 GiB unified.  The 819 GB/s vendor
# number is NOT this machine's roof and is not used anywhere in this module.
MEASURED_READ_BYTES_PER_S = 736.0e9          # best-median sustained GPU read (759.8e9 max)
MEASURED_FMA_FLOPS_PER_S = 17_703.0e9        # fp32 FMA
MEASURED_RIDGE_FLOP_PER_BYTE = 24.05
MEASURED_COMMAND_BUFFER_FIXED_S = 215.8e-6
MEASURED_MARGINAL_DISPATCH_S = 0.71e-6
# The measured roof is an fp32 FMA benchmark, and an FMA is conventionally counted as two
# floating-point operations.  Records count MACs; this constant is the only place the two
# conventions meet, so a reader can re-derive the roofline under the other convention.
ROOF_FLOP_PER_MAC = 2

# Production representation, rung R0 (tools/condense/glm52_pack.py LADDER).
R0_D = 8
R0_K = 128
R0_SUBSPACES = 1
R0_ROTATE = False
R0_ARTIFACT_INDEX_BITS = 7                   # what the file stores and what R0 bills
KERNEL_INDEX_BITS = 8                        # what gravity_metal.py uploads today
CODEBOOK_DTYPE_BYTES = 2                     # fp16 on disk, deserialized to fp32
METADATA_BYTES = 64                          # glm52_pack.HEADER_BYTES, a real fixed header
BF16_BYTES = 2

# gravity_metal.py constants that shape the re-read terms.
KERNEL_THREADS = 256
THREADGROUP_MEMORY_LIMIT = 32768             # measured maxThreadgroupMemoryLength


class FlopLedgerError(RuntimeError):
    """A ledger input is inconsistent with the artifact or with the architecture."""


# ---------------------------------------------------------------------------
# 1. The per-workload record.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkloadRecord:
    """One workload's arithmetic and traffic, with UNMEASURED kept distinct from zero.

    Every field is either an explicit number or ``None``.  ``None`` means UNMEASURED and
    propagates: a ratio that would need it comes back ``None`` rather than treating the
    missing term as free.  There is no default that quietly means zero.

    Fields, in the order the mandate names them:

    ``dense_equivalent_macs``
        What a BF16 reference would execute for the same contraction, in MACs.
    ``dense_equivalent_bytes``
        What that reference would read for the same contraction (weights, BF16).
    ``executed_flops``
        Floating-point operations actually executed, in MAC-equivalents (one FMA = 1).
    ``executed_int_ops``
        Integer/bit operations: index unpack, index load, gather address arithmetic.
    ``representation_overhead_ops``
        Work that exists only because the representation exists: codebook construction or
        staging, table construction, metadata handling.
    ``bytes_read`` / ``bytes_written``
        Device traffic actually moved, including any re-reads.
    ``dispatches``
        Kernel dispatches, so fixed command-buffer cost can be attributed.
    ``latency_s``
        Wall or GPU seconds, whichever the ``note`` says.  ``None`` unless measured.
    """

    workload: str
    dense_equivalent_macs: int | None
    dense_equivalent_bytes: int | None
    executed_flops: int | None
    executed_int_ops: int | None
    representation_overhead_ops: int | None
    bytes_read: int | None
    bytes_written: int | None
    dispatches: int | None
    latency_s: float | None
    evidence: str
    note: str = ""

    @property
    def executed_arithmetic(self) -> int | None:
        """Every executed operation, float and integer and representation overhead."""
        parts = (self.executed_flops, self.executed_int_ops,
                 self.representation_overhead_ops)
        if any(part is None for part in parts):
            return None
        return sum(parts)  # type: ignore[arg-type]

    @property
    def executed_bytes(self) -> int | None:
        if self.bytes_read is None or self.bytes_written is None:
            return None
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_compression(self) -> float | None:
        """Dense-equivalent arithmetic over executed arithmetic.  Descriptive only.

        This counts index and staging work against the representation, so it can fall
        BELOW 1.0: decode-FMA runs the dense MAC count and then pays index loads on top.
        """
        return _ratio(self.dense_equivalent_macs, self.executed_arithmetic)

    @property
    def flop_compression(self) -> float | None:
        """Dense-equivalent MACs over executed floating-point MACs alone.

        The classifier keys on this rather than on :attr:`arithmetic_compression`, because
        the question "did the representation remove arithmetic" is a question about the
        floating-point contraction; index and staging ops are an additive tax on top and
        are reported separately rather than mixed into the same ratio.
        """
        return _ratio(self.dense_equivalent_macs, self.executed_flops)

    @property
    def byte_compression(self) -> float | None:
        """Dense-equivalent bytes over executed bytes.  Descriptive only."""
        return _ratio(self.dense_equivalent_bytes, self.executed_bytes)

    @property
    def arithmetic_intensity_flop_per_byte(self) -> float | None:
        return _ratio(self.executed_arithmetic, self.executed_bytes)

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema"] = RECORD_SCHEMA
        row["executed_arithmetic"] = self.executed_arithmetic
        row["executed_bytes"] = self.executed_bytes
        row["arithmetic_compression"] = self.arithmetic_compression
        row["flop_compression"] = self.flop_compression
        row["byte_compression"] = self.byte_compression
        row["arithmetic_intensity_flop_per_byte"] = self.arithmetic_intensity_flop_per_byte
        row["unmeasured_fields"] = sorted(k for k, v in asdict(self).items() if v is None)
        return row


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    """UNMEASURED in, UNMEASURED out.  A missing term never becomes a zero."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


# ---------------------------------------------------------------------------
# 2. Analytic cost models for the two grammars.
# ---------------------------------------------------------------------------
def nchunk_for(cols: int, D: int) -> int:
    if cols % D:
        raise FlopLedgerError(f"cols={cols} is not divisible by D={D}; PQ geometry undefined")
    return cols // D


def packed_tensor_bytes(rows: int, cols: int, *, D: int = R0_D, k: int = R0_K,
                        index_bits: int = R0_ARTIFACT_INDEX_BITS) -> int:
    """Bytes one PQ tensor occupies on disk, matching glm52_pack.serialize exactly.

    Header (64 B, billed) + one fp16 codebook (k*D halves) + bit-packed indices at the
    billed width.  Validated against the real shard header in :func:`validate_against_shard`.
    """
    chunks = nchunk_for(cols, D)
    index_bytes = (rows * chunks * index_bits + 7) // 8
    return METADATA_BYTES + k * D * CODEBOOK_DTYPE_BYTES + index_bytes


def decode_fma_cost(rows: int, cols: int, *, D: int = R0_D, k: int = R0_K,
                    threads: int = KERNEL_THREADS,
                    index_bits: int = KERNEL_INDEX_BITS,
                    stage_x: bool | None = None) -> dict[str, Any]:
    """The grammar gravity_metal.py implements: one FMA per original weight.

    Executed MACs are ``rows*nchunk*D``, which equals ``rows*cols``: the dense count.  The
    device traffic is where this grammar wins, and the win is smaller than
    ``gravity_metal.bytes_read_per_matvec`` reports because that function counts one
    codebook and no ``x``, while every threadgroup stages its own copy of both.
    """
    chunks = nchunk_for(cols, D)
    groups = (rows + threads - 1) // threads
    codebook_bytes = k * D * CODEBOOK_DTYPE_BYTES
    x_bytes = chunks * D * 4
    if stage_x is None:  # gravity_metal.py:207
        stage_x = (k * D + chunks * D) * 4 <= THREADGROUP_MEMORY_LIMIT
    index_bytes = (rows * chunks * index_bits + 7) // 8
    # Re-read terms: the codebook is staged per threadgroup; x is staged per threadgroup
    # when it fits, and otherwise streamed per (thread, chunk) straight from device memory.
    codebook_reread = groups * codebook_bytes
    x_reread = groups * x_bytes if stage_x else rows * chunks * D * 4
    return {
        "grammar": "decode-FMA",
        "rows": rows, "cols": cols, "D": D, "k": k, "nchunk": chunks,
        "threads_per_threadgroup": threads, "threadgroups": groups, "stage_x": stage_x,
        "macs": rows * chunks * D,
        "index_ops": rows * chunks,
        "codebook_staging_ops": groups * k * D,
        "index_bytes": index_bytes,
        "codebook_bytes_single": codebook_bytes,
        "codebook_bytes_reread": codebook_reread,
        "x_bytes_single": x_bytes,
        "x_bytes_reread": x_reread,
        "device_bytes_read": index_bytes + codebook_reread + x_reread,
        "bytes_written": rows * 4,
        "reported_by_gravity_metal": index_bytes + codebook_bytes,
        "dense_bf16_weight_bytes": rows * cols * BF16_BYTES,
    }


def lookup_linear_cost(rows: int, cols: int, *, D: int = R0_D, k: int = R0_K,
                       table_dtype_bytes: int = 4) -> dict[str, Any]:
    """The grammar the representation permits but no kernel implements yet.

    ``codebook @ x`` is the same for every row, so build it once per chunk
    (``nchunk*k*D`` MACs) and then each output row is ``nchunk`` gather-adds.  Arithmetic
    stops scaling with ``rows*cols`` and starts scaling with ``rows*nchunk``, which is the
    only structural route from "fewer bytes" to "fewer operations" at this rate.
    """
    chunks = nchunk_for(cols, D)
    table_macs = chunks * k * D
    gather_adds = rows * chunks
    return {
        "grammar": "lookup-linear",
        "rows": rows, "cols": cols, "D": D, "k": k, "nchunk": chunks,
        "table_macs": table_macs,
        "gather_adds": gather_adds,
        "total_ops": table_macs + gather_adds,
        "table_bytes": chunks * k * table_dtype_bytes,
        "table_dtype_bytes": table_dtype_bytes,
        "index_ops": gather_adds,
        "dense_equivalent_macs": rows * chunks * D,
        "reduction_vs_decode_fma": (rows * chunks * D) / (table_macs + gather_adds),
    }


# ---------------------------------------------------------------------------
# 3. Whole-model active-byte ledger, derived from the architecture.
# ---------------------------------------------------------------------------
# Categories glm52_contract.classify_tensor marks CONTROL_SENSITIVE_CANDIDATE: they ride at
# source precision (16 BPW) rather than being packed, so their active bytes are dense.
PROTECTED_ORGANS = frozenset({
    "normalization", "indexer", "router", "router_control",
    "mtp_normalization", "mtp_head_norm",
})


def official_geometry() -> adapter.Geometry:
    """The pinned official architecture, straight from the adapter's own contract."""
    config = dict(adapter.OFFICIAL_CONFIG_CONTRACT)
    config["indexer_types"] = list(adapter.OFFICIAL_INDEXER_TYPES)
    config["mlp_layer_types"] = [
        "dense" if layer < int(config["first_k_dense_replace"]) else "sparse"
        for layer in range(int(config["num_hidden_layers"]))
    ]
    return adapter.validate_config(config)


def ladder_admissible(elements: int, *, D: int = R0_D, k: int = R0_K,
                      index_bits: int = R0_ARTIFACT_INDEX_BITS) -> bool:
    """glm52_pack.rung_is_admissible: fixed codebook cost must amortize under 1.0 BPW."""
    fixed_bits = k * D * CODEBOOK_DTYPE_BYTES * 8 + METADATA_BYTES * 8
    return (index_bits / D) + fixed_bits / max(1, elements) < 1.0


def active_tensor_rows(geometry: adapter.Geometry) -> list[dict[str, Any]]:
    """Every weight tensor read to decode ONE token, with its packed and dense bytes.

    Derived, not assumed: names and shapes come from ``glm52_adapter.tensor_spec``, the
    dense/sparse split from ``first_k_dense_replace``, the IndexShare full/shared split
    from ``geometry.indexer_type``, and the routed-expert count from
    ``num_experts_per_tok``.  The MTP layer is excluded and says so; it is a speculative
    head, not part of the single-token main-text path.
    """
    rows: list[dict[str, Any]] = []

    def add(name: str, *, gather_rows: int | None = None, note: str = "") -> None:
        spec = adapter.tensor_spec(geometry, name)
        shape = spec.shape
        elements = math.prod(shape)
        protected = spec.organ in PROTECTED_ORGANS or not ladder_admissible(elements)
        if len(shape) == 1 or protected:
            packed = elements * (4 if spec.dtype == "F32" else BF16_BYTES)
            packed_kind = "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES"
        else:
            packed = packed_tensor_bytes(shape[0], shape[1])
            packed_kind = "PACKED_IN_CORE_ARTIFACT"
        dense = elements * (4 if spec.dtype == "F32" else BF16_BYTES)
        macs = 0 if len(shape) == 1 else shape[0] * shape[1]
        if gather_rows is not None:  # a row gather, not a contraction
            fraction = gather_rows / shape[0]
            packed = (METADATA_BYTES + R0_K * R0_D * CODEBOOK_DTYPE_BYTES
                      + (gather_rows * nchunk_for(shape[1], R0_D) * R0_ARTIFACT_INDEX_BITS + 7) // 8)
            dense = gather_rows * shape[1] * BF16_BYTES
            macs = 0
            packed_kind = "PACKED_ROW_GATHER"
            note = note or f"one row of {shape[0]} ({fraction:.3g} of the tensor)"
        if packed_kind == "PACKED_IN_CORE_ARTIFACT":
            cost = decode_fma_cost(shape[0], shape[1])
            kernel_bytes = cost["device_bytes_read"] + cost["bytes_written"]
            stage_x = cost["stage_x"]
        else:
            kernel_bytes, stage_x = packed, None
        rows.append({
            "name": name, "organ": spec.organ, "shape": list(shape), "dtype": spec.dtype,
            "elements": elements, "terminal_state": packed_kind,
            "active_bytes": packed, "dense_bf16_bytes": dense,
            "kernel_device_bytes": kernel_bytes, "stage_x": stage_x,
            "dense_equivalent_macs": macs, "note": note,
        })

    add("model.embed_tokens.weight", gather_rows=1)
    for layer in range(geometry.num_hidden_layers):
        for name in adapter._layer_names(geometry, layer):
            spec = adapter.tensor_spec(geometry, name)
            if spec.organ == "routed_expert" and spec.expert >= geometry.num_experts_per_tok:
                continue  # top-k routing: only num_experts_per_tok of n_routed_experts fire
            add(name, note=("top-%d of %d routed experts; shapes are identical so which "
                            "experts fire does not change the byte total"
                            % (geometry.num_experts_per_tok, geometry.n_routed_experts))
                if spec.organ == "routed_expert" else "")
    add("model.norm.weight")
    add("lm_head.weight")
    return rows


def kv_read_model(geometry: adapter.Geometry, *, context_length: int,
                  kv_cache_dtype_bytes: int | None = None,
                  index_cache_dtype_bytes: int | None = None) -> dict[str, Any]:
    """Per-token KV traffic.  Shapes derive from the config; the cache dtype does not.

    Derivable from the repo: the latent width per cached position is
    ``kv_lora_rank + qk_rope_head_dim`` (the row count of ``kv_a_proj_with_mqa``), one
    entry per layer per position, and DSA attends to at most ``index_topk`` positions.  The
    indexer scores every position it can see, but only the 'full' IndexShare layers hold
    their own index keys.

    NOT derivable from the repo: the KV cache element dtype.  There is no inference runtime
    in this tree, so nothing here fixes it.  Pass it explicitly or the byte totals come back
    ``None`` (UNMEASURED) with the missing input named.
    """
    latent_width = geometry.kv_lora_rank + geometry.qk_rope_head_dim
    attended = min(context_length, geometry.index_topk)
    full_layers = sum(1 for layer in range(geometry.num_hidden_layers)
                      if geometry.indexer_type(layer) == "full")
    latent_entries = geometry.num_hidden_layers * attended * latent_width
    index_entries = full_layers * context_length * geometry.index_head_dim
    unknowns = []
    if kv_cache_dtype_bytes is None:
        unknowns.append({"field": "kv_cache_dtype_bytes",
                         "missing_input": "the serving runtime's KV latent cache dtype; "
                                          "no inference runtime exists in this repo"})
    if index_cache_dtype_bytes is None:
        unknowns.append({"field": "index_cache_dtype_bytes",
                         "missing_input": "the serving runtime's IndexShare key cache dtype; "
                                          "no inference runtime exists in this repo"})
    return {
        "context_length": context_length,
        "index_topk": geometry.index_topk,
        "attended_positions": attended,
        "latent_width_per_position": latent_width,
        "latent_width_derivation": "kv_lora_rank + qk_rope_head_dim = "
                                   f"{geometry.kv_lora_rank} + {geometry.qk_rope_head_dim}",
        "layers_reading_latents": geometry.num_hidden_layers,
        "full_indexshare_layers": full_layers,
        "kv_latent_elements": latent_entries,
        "index_key_elements": index_entries,
        "kv_cache_dtype_bytes": kv_cache_dtype_bytes,
        "index_cache_dtype_bytes": index_cache_dtype_bytes,
        "kv_latent_bytes": (None if kv_cache_dtype_bytes is None
                            else latent_entries * kv_cache_dtype_bytes),
        "index_key_bytes": (None if index_cache_dtype_bytes is None
                            else index_entries * index_cache_dtype_bytes),
        "unknowns": unknowns,
        "evidence": "ANALYTIC_FROM_CONFIG_SHAPES_ONLY",
    }


def validate_against_shard(shard: Path) -> dict[str, Any]:
    """Check every analytic per-tensor byte figure against the real sealed artifact.

    The shard header is the authority on what is really stored.  For each distinct
    (category, shape) present it compares :func:`packed_tensor_bytes` to the header's own
    byte count; any nonzero delta means the analytic model has drifted from the packer and
    the ledger below it is not to be trusted.
    """
    header = gravity_format.read_header(shard)
    geometry = official_geometry()
    seen: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for tensor in header["tensors"]:
        shape = tuple(int(dim) for dim in tensor["shape"])
        key = (tensor["category"], shape)
        predicted = packed_tensor_bytes(shape[0], shape[1])
        row = seen.setdefault(key, {
            "category": tensor["category"], "shape": list(shape),
            "elements": int(tensor["elements"]),
            "header_bytes": int(tensor["bytes"]), "analytic_bytes": predicted,
            "delta_bytes": int(tensor["bytes"]) - predicted,
            "header_bpw": float(tensor["bpw"]),
            "analytic_bpw": predicted * 8 / int(tensor["elements"]),
            "occurrences": 0,
            "adapter_shape_agrees": list(shape) == list(
                adapter.tensor_spec(geometry, tensor["name"]).shape),
        })
        row["occurrences"] += 1
    rows = sorted(seen.values(), key=lambda r: (r["category"], r["shape"]))
    return {
        "shard": str(shard),
        "shard_source": header["model"]["source_shard"],
        "body_sha256": header["integrity"]["body_sha256"],
        "tensor_count": int(header["integrity"]["tensor_count"]),
        "production_rung": header["compression"]["production_rung"],
        "packed_bpw": float(header["compression"]["packed_bpw"]),
        "distinct_geometries": rows,
        "all_analytic_bytes_match_header": all(r["delta_bytes"] == 0 for r in rows),
        "all_shapes_match_adapter": all(r["adapter_shape_agrees"] for r in rows),
        "read_only": True,
    }


def active_byte_ledger(shard: Path, *, context_length: int = 4096,
                       kv_cache_dtype_bytes: int | None = None,
                       index_cache_dtype_bytes: int | None = None) -> dict[str, Any]:
    """The whole-model per-token ledger, validated against the real artifact header."""
    geometry = official_geometry()
    validation = validate_against_shard(shard)
    if not validation["all_analytic_bytes_match_header"]:
        raise FlopLedgerError(
            f"analytic byte model disagrees with {shard.name}: {validation['distinct_geometries']}")

    rows = active_tensor_rows(geometry)
    by_organ: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = by_organ.setdefault(row["organ"], {
            "organ": row["organ"], "tensors": 0, "elements": 0,
            "active_bytes": 0, "dense_bf16_bytes": 0, "kernel_device_bytes": 0,
            "dense_equivalent_macs": 0, "tensors_without_x_staging": 0,
            "terminal_states": set(),
        })
        bucket["tensors"] += 1
        bucket["elements"] += row["elements"]
        bucket["active_bytes"] += row["active_bytes"]
        bucket["dense_bf16_bytes"] += row["dense_bf16_bytes"]
        bucket["kernel_device_bytes"] += row["kernel_device_bytes"]
        bucket["tensors_without_x_staging"] += int(row["stage_x"] is False)
        bucket["dense_equivalent_macs"] += row["dense_equivalent_macs"]
        bucket["terminal_states"].add(row["terminal_state"])
    for bucket in by_organ.values():
        bucket["terminal_states"] = sorted(bucket["terminal_states"])

    active_bytes = sum(row["active_bytes"] for row in rows)
    dense_bytes = sum(row["dense_bf16_bytes"] for row in rows)
    dense_macs = sum(row["dense_equivalent_macs"] for row in rows)
    # What the kernel would really move, including per-threadgroup re-reads and the 8-bit
    # index upload where the artifact bills 7-bit.
    kernel_bytes = sum(row["kernel_device_bytes"] for row in rows)
    unstaged = [row for row in rows if row["stage_x"] is False]

    kv = kv_read_model(geometry, context_length=context_length,
                       kv_cache_dtype_bytes=kv_cache_dtype_bytes,
                       index_cache_dtype_bytes=index_cache_dtype_bytes)
    kv_bytes = (None if kv["kv_latent_bytes"] is None or kv["index_key_bytes"] is None
                else kv["kv_latent_bytes"] + kv["index_key_bytes"])

    return {
        "schema": LEDGER_SCHEMA,
        "authority": "Every per-tensor byte figure was checked against the sealed .gravity "
                     "shard header, which is the authority on what is really stored; shapes "
                     "come from glm52_adapter.tensor_spec, never from a guess.",
        "evidence_level": "F0_PHYSICAL_ACCOUNTING_ONLY",
        "not_evidence_of": "output divergence, capability, or end-to-end behaviour",
        "artifact_validation": validation,
        "representation": {"rung": "R0", "D": R0_D, "k": R0_K, "subspaces": R0_SUBSPACES,
                           "rotate": R0_ROTATE, "artifact_index_bits": R0_ARTIFACT_INDEX_BITS,
                           "kernel_index_bits": KERNEL_INDEX_BITS,
                           "kernel_index_stream_inflation": KERNEL_INDEX_BITS / R0_ARTIFACT_INDEX_BITS},
        "architecture": {
            "hidden_size": geometry.hidden_size,
            "layers": geometry.num_hidden_layers,
            "dense_mlp_layers": geometry.first_k_dense_replace,
            "sparse_moe_layers": geometry.num_hidden_layers - geometry.first_k_dense_replace,
            "attention_blocks": geometry.num_hidden_layers,
            "routed_experts": geometry.n_routed_experts,
            "shared_experts": geometry.n_shared_experts,
            "experts_per_token": geometry.num_experts_per_tok,
            "full_indexshare_layers": sum(1 for layer in range(geometry.num_hidden_layers)
                                          if geometry.indexer_type(layer) == "full"),
            "shared_indexshare_layers": sum(1 for layer in range(geometry.num_hidden_layers)
                                            if geometry.indexer_type(layer) == "shared"),
            "vocab_size": geometry.vocab_size,
            "mtp_layer_excluded": geometry.mtp_layer,
            "mtp_exclusion_reason": "speculative head; not on the single-token main-text path",
        },
        "per_organ": sorted(by_organ.values(), key=lambda b: -b["active_bytes"]),
        "totals": {
            "active_tensors_per_token": len(rows),
            "active_bytes_per_token": active_bytes,
            "dense_bf16_bytes_per_token": dense_bytes,
            "byte_compression": _ratio(dense_bytes, active_bytes),
            "dense_equivalent_macs_per_token": dense_macs,
            "kernel_device_bytes_per_token": kernel_bytes,
            "kernel_amplification_vs_artifact": _ratio(kernel_bytes, active_bytes),
            "kernel_byte_compression_vs_dense": _ratio(dense_bytes, kernel_bytes),
        },
        # Where the byte win evaporates: a tensor whose x does not fit alongside the
        # codebook in 32 KB of threadgroup memory falls off the staged path and re-reads x
        # once per (thread, chunk).  These are analytic properties of gravity_metal.py as
        # written, not of the representation.
        "x_staging_failures": {
            "tensors_per_token": len(unstaged),
            "threadgroup_memory_limit_bytes": THREADGROUP_MEMORY_LIMIT,
            "kernel_device_bytes": sum(row["kernel_device_bytes"] for row in unstaged),
            "artifact_bytes": sum(row["active_bytes"] for row in unstaged),
            "distinct_shapes": [list(shape) for shape in
                                sorted({tuple(row["shape"]) for row in unstaged})],
            "cause": "gravity_metal.py:207 stages x only when (k*D + nchunk*D)*4 fits in "
                     "maxThreadgroupMemoryLength; above that the else-branch at "
                     "gravity_metal.py:98 streams x from device memory per chunk per thread",
        },
        "kv_reads": kv,
        "totals_including_kv": {
            "kv_bytes_per_token": kv_bytes,
            "active_bytes_per_token": None if kv_bytes is None else active_bytes + kv_bytes,
        },
        "unknowns": kv["unknowns"],
    }


# ---------------------------------------------------------------------------
# 4. Token roofline.
# ---------------------------------------------------------------------------
def token_roofline(ledger: dict[str, Any], *, bytes_per_token: int | None = None,
                   ops_per_token: int | None = None) -> dict[str, Any]:
    """Both bounds, reported side by side, with the binding one named.

    Bytes go into the MEASURED 736 GB/s sustained read.  Operations go into the MEASURED
    17,703 GFLOP/s fp32 FMA roof, converted at ``ROOF_FLOP_PER_MAC``.  Whichever time is
    larger binds; the arithmetic intensity against the measured 24.05 flop/byte ridge says
    the same thing a second way.
    """
    totals = ledger["totals"]
    active_bytes = totals["active_bytes_per_token"] if bytes_per_token is None else bytes_per_token
    # decode-FMA executes the dense MAC count.  That is the finding, not an approximation.
    executed = totals["dense_equivalent_macs_per_token"] if ops_per_token is None else ops_per_token

    executed_flop = None if executed is None else executed * ROOF_FLOP_PER_MAC
    bandwidth_s = _ratio(active_bytes, MEASURED_READ_BYTES_PER_S)
    compute_s = _ratio(executed_flop, MEASURED_FMA_FLOPS_PER_S)
    binds = ("UNMEASURED" if bandwidth_s is None or compute_s is None
             else ("COMPUTE" if compute_s > bandwidth_s else "BANDWIDTH"))
    intensity = _ratio(executed_flop, active_bytes)
    return {
        "schema": ROOFLINE_SCHEMA,
        "measured_constants": {
            "sustained_read_bytes_per_s": MEASURED_READ_BYTES_PER_S,
            "sustained_read_note": "best-median measured on this machine; the 819 GB/s "
                                   "vendor figure is NOT this machine's roof and is unused",
            "fp32_fma_flops_per_s": MEASURED_FMA_FLOPS_PER_S,
            "roof_flop_per_mac": ROOF_FLOP_PER_MAC,
            "ridge_flop_per_byte": MEASURED_RIDGE_FLOP_PER_BYTE,
            "command_buffer_fixed_s": MEASURED_COMMAND_BUFFER_FIXED_S,
            "marginal_dispatch_s": MEASURED_MARGINAL_DISPATCH_S,
        },
        "active_bytes_per_token": active_bytes,
        "executed_macs_per_token": executed,
        "bandwidth_bound_s_per_token": bandwidth_s,
        "compute_bound_s_per_token": compute_s,
        "binds": binds,
        "bound_tokens_per_s": _ratio(1.0, max(x for x in (bandwidth_s, compute_s)
                                              if x is not None)) if binds != "UNMEASURED" else None,
        "arithmetic_intensity_flop_per_byte": intensity,
        "ridge_comparison": (None if intensity is None else
                             ("ABOVE_RIDGE_COMPUTE_BOUND" if intensity > MEASURED_RIDGE_FLOP_PER_BYTE
                              else "BELOW_RIDGE_BANDWIDTH_BOUND")),
        "evidence_level": "F0_ANALYTIC_ROOFLINE_FROM_MEASURED_MACHINE_CONSTANTS",
        "not_evidence_of": "achieved throughput; the measured custom kernel reaches 0.19% "
                           "of the bandwidth roof and 0.13% of the compute roof",
    }


# ---------------------------------------------------------------------------
# 5. Classifier.
# ---------------------------------------------------------------------------
CLASSIFICATIONS = (
    "FEWER_BYTES_SAME_ARITHMETIC",
    "FEWER_FLOPS_AND_BYTES",
    "CONDITIONAL_COMPUTE",
    "NATIVE_FUNCTIONAL_RUNTIME",
    "UNMEASURED_INSUFFICIENT",
)
SAME_ARITHMETIC_TOLERANCE = 0.02  # 2%: below this, "fewer operations" is noise, not a result


def classify(records: list[WorkloadRecord], *,
             tolerance: float = SAME_ARITHMETIC_TOLERANCE) -> dict[str, Any]:
    """Name what a set of workloads actually bought.  The rule, in order:

    0. If any record needed for the totals is UNMEASURED -> ``UNMEASURED_INSUFFICIENT``.
       Nothing is classified from a number that was never measured.
    1. If any workload has dense-equivalent MACs but executes zero arithmetic, the saving
       came from *not running* work, not from a cheaper representation of it ->
       ``CONDITIONAL_COMPUTE``.
    2. Else if ``flop_compression <= 1 + tolerance`` -- the floating-point contraction is
       unchanged, or worse -- the representation bought bytes and nothing else ->
       ``FEWER_BYTES_SAME_ARITHMETIC``.  The reason string carries the exact excess when the
       executed total exceeds dense, which is the normal decode-FMA case: the dense MAC
       count runs and index loads are paid on top of it.
    3. Else if flop compression exceeds 1 and no record pays any representation overhead,
       the stored form is executed directly rather than reconstructed ->
       ``NATIVE_FUNCTIONAL_RUNTIME``.
    4. Else -> ``FEWER_FLOPS_AND_BYTES``.

    The rule keys on floating-point work, not on the executed total, because a taxonomy
    that let index-load bookkeeping decide the verdict would name the same kernel
    differently for a change that touched no arithmetic at all.
    """
    if not records:
        raise FlopLedgerError("classification needs at least one record")

    incomplete = sorted(r.workload for r in records
                        if r.dense_equivalent_macs is None or r.dense_equivalent_bytes is None
                        or r.executed_flops is None or r.executed_arithmetic is None
                        or r.executed_bytes is None)
    if incomplete:
        return {
            "classification": "UNMEASURED_INSUFFICIENT",
            "rule": "at least one record carries an UNMEASURED field the totals require",
            "unmeasured_records": incomplete,
            "arithmetic_compression": None, "flop_compression": None,
            "byte_compression": None,
        }

    dense_macs = sum(r.dense_equivalent_macs for r in records)          # type: ignore[misc]
    dense_bytes = sum(r.dense_equivalent_bytes for r in records)        # type: ignore[misc]
    executed_flops = sum(r.executed_flops for r in records)             # type: ignore[misc]
    executed_ops = sum(r.executed_arithmetic for r in records)          # type: ignore[misc]
    executed_bytes = sum(r.executed_bytes for r in records)             # type: ignore[misc]
    arithmetic = _ratio(dense_macs, executed_ops)
    flop = _ratio(dense_macs, executed_flops)
    byte = _ratio(dense_bytes, executed_bytes)
    overhead = sum(r.representation_overhead_ops for r in records)      # type: ignore[misc]

    skipped = sorted(r.workload for r in records
                     if r.dense_equivalent_macs and not r.executed_arithmetic)
    if skipped:
        verdict, reason = "CONDITIONAL_COMPUTE", (
            f"{len(skipped)} workload(s) execute no arithmetic at all: the saving is skipped "
            "work, not a cheaper representation")
    elif flop is None or flop <= 1.0 + tolerance:
        verdict, reason = "FEWER_BYTES_SAME_ARITHMETIC", (
            f"flop compression {flop:.4f} does not exceed 1 by more than {tolerance:.0%}: the "
            f"dense MAC count still runs, and counting index and staging work the executed "
            f"total is {1.0 / arithmetic:.4f}x dense while bytes fall {byte:.4f}x")
    elif overhead == 0:
        verdict, reason = "NATIVE_FUNCTIONAL_RUNTIME", (
            f"flop compression {flop:.4f} with zero representation overhead: the stored form "
            "is executed directly, never reconstructed")
    else:
        verdict, reason = "FEWER_FLOPS_AND_BYTES", (
            f"flop compression {flop:.4f} and byte compression {byte:.4f}, "
            f"paying {overhead} representation-overhead ops")
    return {
        "classification": verdict, "rule": reason,
        "records": len(records),
        "dense_equivalent_macs": dense_macs, "dense_equivalent_bytes": dense_bytes,
        "executed_flops": executed_flops,
        "executed_arithmetic": executed_ops, "executed_bytes": executed_bytes,
        "representation_overhead_ops": overhead,
        "arithmetic_compression": arithmetic, "flop_compression": flop,
        "byte_compression": byte,
        "tolerance": tolerance,
        "descriptive_only": "These ratios describe work and traffic removed relative to a "
                            "dense BF16 reference.  They are not capability claims.",
    }


def matvec_record(rows: int, cols: int, *, workload: str, latency_s: float | None = None,
                  evidence: str = "ANALYTIC_COST_MODEL", note: str = "") -> WorkloadRecord:
    """A decode-FMA matvec as a record, so the ledger and the classifier share one model."""
    cost = decode_fma_cost(rows, cols)
    return WorkloadRecord(
        workload=workload,
        dense_equivalent_macs=rows * cols,
        dense_equivalent_bytes=rows * cols * BF16_BYTES + cols * 4,
        executed_flops=cost["macs"],
        executed_int_ops=cost["index_ops"],
        representation_overhead_ops=cost["codebook_staging_ops"],
        bytes_read=cost["device_bytes_read"],
        bytes_written=cost["bytes_written"],
        dispatches=1,
        latency_s=latency_s,
        evidence=evidence,
        note=note or f"decode-FMA, {cost['threadgroups']} threadgroups, stage_x={cost['stage_x']}",
    )


# MEASURED, not modelled: the down-projection geometry timed on this machine.  Latency is
# GPU time; the wall figure was 0.5057 ms.  Kept as a constant so the record type is
# exercised against a real measurement and not only against the analytic model.
MEASURED_DOWN_PROJ_LATENCY_GPU_S = 0.2096e-3
MEASURED_DOWN_PROJ_LATENCY_WALL_S = 0.5057e-3
MEASURED_DENSE_FP16_MPS_LATENCY_S = 0.3674e-3


def measured_down_proj_record() -> WorkloadRecord:
    return matvec_record(
        6144, 2048, workload="R0 down_proj [6144,2048] custom Metal v2",
        latency_s=MEASURED_DOWN_PROJ_LATENCY_GPU_S,
        evidence="MEASURED_GPU_TIME_ANALYTIC_OP_AND_BYTE_COUNTS",
        note="GPU time measured 0.2096 ms (wall 0.5057 ms); dense fp16 MPS measured "
             "0.3674 ms at the same geometry, so the custom kernel is 0.727x dense here. "
             "Operation and byte counts are analytic, latency is measured.")


# ---------------------------------------------------------------------------
# Report emission.
# ---------------------------------------------------------------------------
DEFAULT_SHARD = Path("/Users/scammermike/Desktop/GLM52-Gravity-SubBit/"
                     "model-00002-of-00282.gravity")


def emit(out_dir: Path, shard: Path = DEFAULT_SHARD, *,
         context_length: int = 4096) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = active_byte_ledger(shard, context_length=context_length)
    roofline = token_roofline(ledger)

    records = [matvec_record(2048, 6144, workload="R0 gate_proj [2048,6144]"),
               matvec_record(2048, 6144, workload="R0 up_proj [2048,6144]"),
               measured_down_proj_record()]
    roofline["classification"] = classify(records)
    roofline["records"] = [record.to_json() for record in records]
    roofline["grammar_comparison"] = {
        "gate_up_decode_fma": decode_fma_cost(2048, 6144),
        "gate_up_lookup_linear": lookup_linear_cost(2048, 6144),
        "down_decode_fma": decode_fma_cost(6144, 2048),
        "down_lookup_linear": lookup_linear_cost(6144, 2048),
    }

    paths = {}
    for name, payload in (("GLM52_ACTIVE_BYTE_LEDGER.json", ledger),
                          ("GLM52_TOKEN_ROOFLINE.json", roofline)):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths[name] = path
    return paths


def _iter_summary(ledger: dict[str, Any]) -> Iterator[str]:
    totals = ledger["totals"]
    yield f"active bytes/token   {totals['active_bytes_per_token']:,}"
    yield f"dense BF16/token     {totals['dense_bf16_bytes_per_token']:,}"
    yield f"byte compression     {totals['byte_compression']:.3f}x"
    yield f"dense MACs/token     {totals['dense_equivalent_macs_per_token']:,}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "emit":
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else (
            HERE.parent.parent / "reports" / "condense" / "breakthrough")
        written = emit(target)
        print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print("\n".join(_iter_summary(active_byte_ledger(DEFAULT_SHARD))))
        raise SystemExit(0)
    sys.stderr.write("usage: gravity_flop_ledger.py [emit [OUT_DIR]|summary]\n")
    raise SystemExit(2)
