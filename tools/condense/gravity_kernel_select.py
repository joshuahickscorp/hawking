#!/usr/bin/env python3
"""KERNEL SELECTION MATRIX -- which grammar executes which geometry, and on what evidence.

Three grammars now exist for the same R0 artifact and none of them wins everywhere:

  * ``production_v2``  -- gravity_metal.py, one thread per row, 1D grid.  GRID_UNDERFILLED.
  * ``decode_fma_2d``  -- Track A's (row tile x chunk block) split.  Wins where the table
                          build would not amortize.
  * ``lookup_linear``  -- Track B's on-chip per-block table.  Wins where ``rows >> k*D``.

The choice is NOT a hand-written if-branch over magic numbers.  Three tables carry it:

  ``KERNELS``          what each grammar can physically consume, with the source line of
                       every refusal.  A kernel that cannot decode the artifact is not a
                       candidate no matter how fast it is (mandate 2).
  ``SHAPE_PRIORS``     which variant shapes are worth timing, each row naming the sealed
                       measurement that put it there.  No constant appears without a
                       citation.
  ``CRITERIA``         the ranking, in order, with the noise band and the source of the
                       band.  ``select_kernel`` walks this list; it contains no geometry
                       knowledge at all.

Everything above answers on a machine with no Metal device, which is what the test file
exercises.  ``main()`` adds the GPU: it times every admissible candidate on REAL tensors
from the live campaign's shards (read-only, safe-age enforced), grades every one against
``gravity_forge.pq_execute`` before timing it, runs the adversarial index distributions,
and re-bills the whole token under the selected kernels.

Activation source is SYNTHETIC everywhere -- see grf.teacher_activation_status().
The router is absent from every shard, so expert selection is a FIXED LIST, not routing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gravity_bench_lab as lab            # noqa: E402
import gravity_flop_ledger as fl           # noqa: E402
import gravity_forge as forge              # noqa: E402
import gravity_metal                       # noqa: E402
import gravity_metal_lab_b as lb           # noqa: E402
import gravity_real_fixtures as grf        # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "condense" / "breakthrough"
REPORT_PATH = REPORT_DIR / "GLM52_KERNEL_SELECTION_MATRIX.json"

BASELINE_REPORT = "reports/condense/breakthrough/GLM52_BREAKTHROUGH_BASELINE.json"
TRACK_A_REPORT = "reports/condense/breakthrough/GLM52_TRACK_A_BENCHMARK.json"
TRACK_B_REPORT = "reports/condense/breakthrough/GLM52_TRACK_B_BENCHMARK.json"

PARITY_GATE = 2e-3                  # gravity_metal.py's own tolerance; see module docstring
BANDWIDTH_ROOF_GB_S = lab.BANDWIDTH_ROOF_GB_S
THREADGROUP_MEMORY_LIMIT = 32768    # MEASURED: Mac15,14 maxThreadgroupMemoryLength

# Whole-token totals this file must reproduce, not assume.  Both come from
# gravity_flop_ledger.active_tensor_rows(official_geometry()) and are asserted in main().
PRODUCTION_TOKEN_BYTES = 38_579_106_528
ARTIFACT_FLOOR_BYTES = 5_010_218_784


class KernelSelectError(RuntimeError):
    """A selection cannot be made, or would misdescribe the evidence behind it."""


# ------------------------------------------------------------------ table 1: what can decode what

@dataclass(frozen=True)
class Kernel:
    """One grammar's ARTIFACT COMPATIBILITY, stated as the refusals in its own source.

    Every field here is a property of a compiled kernel, not a tuning knob.  ``accepts``
    returns the reasons rather than a bool so a rejection can be published verbatim.
    """

    name: str
    entry_point: str
    subspaces: tuple[int, ...]
    rotate_supported: bool
    k_max: int
    index_bits_in_kernel: int
    d_multiple_of_4: bool
    rows_multiple_of_4: bool
    rungs: tuple[str, ...]
    scratch_model: Callable[[int, int, int], int]   # (k, D, cbs) -> threadgroup bytes
    refusal_sites: tuple[tuple[str, str], ...]
    provenance: str

    def accepts(self, artifact: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
        """Can this kernel consume this artifact at all?  Reasons, not a bare bool."""
        reasons: list[str] = []
        if int(artifact["S"]) not in self.subspaces:
            reasons.append(f"subspaces={artifact['S']} not in {self.subspaces}")
        if artifact["rotate"] and not self.rotate_supported:
            reasons.append("artifact is rotated; kernel has no rotation path")
        if int(artifact["k"]) > self.k_max:
            reasons.append(f"k={artifact['k']} exceeds k_max={self.k_max}")
        if int(artifact["index_bits_on_disk"]) > self.index_bits_in_kernel:
            reasons.append(
                f"disk index width {artifact['index_bits_on_disk']} b exceeds the "
                f"{self.index_bits_in_kernel} b the kernel loads")
        if self.d_multiple_of_4 and int(artifact["D"]) % 4 != 0:
            reasons.append(f"float4 inner loop needs D%4==0, D={artifact['D']}")
        if self.rows_multiple_of_4 and int(artifact["rows"]) % 4 != 0:
            reasons.append(f"row4 tile needs rows%4==0, rows={artifact['rows']}")
        rung = artifact.get("rung")
        if rung is not None and rung not in self.rungs:
            reasons.append(f"rung {rung} not in {self.rungs}")
        if self.scratch_model(int(artifact["k"]), int(artifact["D"]), 1) > THREADGROUP_MEMORY_LIMIT:
            reasons.append("codebook alone exceeds threadgroup memory")
        return (not reasons), tuple(reasons)

    def to_json(self) -> dict[str, Any]:
        return {
            "kernel": self.name, "entry_point": self.entry_point,
            "subspaces_supported": list(self.subspaces),
            "rotate_supported": self.rotate_supported,
            "k_max": self.k_max, "index_bits_in_kernel": self.index_bits_in_kernel,
            "requires_D_multiple_of_4": self.d_multiple_of_4,
            "requires_rows_multiple_of_4": self.rows_multiple_of_4,
            "rungs_supported": list(self.rungs),
            "refusal_sites": [{"condition": c, "source": s} for c, s in self.refusal_sites],
            "provenance": self.provenance,
        }


KERNELS: dict[str, Kernel] = {
    "production_v2": Kernel(
        name="production_v2",
        entry_point="gravity_metal.GravityMetalDecoder.matvec (gravity_pq_matvec_rows)",
        subspaces=(1,), rotate_supported=False, k_max=256, index_bits_in_kernel=8,
        d_multiple_of_4=False, rows_multiple_of_4=False, rungs=("R0",),
        scratch_model=lambda k, D, cbs: k * D * 4,
        refusal_sites=(
            ("S != 1", "gravity_metal.py matvec: 'this kernel handles subspaces == 1 only'"),
            ("rotate", "gravity_metal.py matvec: 'rotated geometry is not wired into this "
                       "kernel yet'"),
            ("k > 256", "gravity_metal.py matvec: 'k > 256 exceeds the one-byte index this "
                        "kernel uses'"),
        ),
        provenance="the shipped kernel; 1D grid, const uint row = gid, GRID_UNDERFILLED",
    ),
    "decode_fma_2d": Kernel(
        name="decode_fma_2d",
        entry_point="gravity_metal_lab_b.TrackBDecoder (dfma_split) == Track A's grammar",
        subspaces=(1,), rotate_supported=False, k_max=256, index_bits_in_kernel=8,
        d_multiple_of_4=True, rows_multiple_of_4=False, rungs=("R0",),
        scratch_model=lambda k, D, cbs: (k * D + cbs * D) * 4,
        refusal_sites=(
            ("D % 4 != 0", "gravity_metal_lab_b.dfma_plan: 'the float4 inner loop needs D % 4 == 0'"),
            ("index >= k", "gravity_metal_lab_b.TrackBDecoder.upload: 'index ... out of range for k'"),
            ("scratch > limit", "gravity_metal_lab_b.dfma_plan: 'cbs=... needs ... B, limit ...'"),
        ),
        provenance="GLM52_TRACK_A_BENCHMARK.json: first configuration faster than dense fp16",
    ),
    "lookup_linear": Kernel(
        name="lookup_linear",
        entry_point="gravity_metal_lab_b.TrackBDecoder (ll_blk_f32_r4), table on chip",
        subspaces=(1,), rotate_supported=False, k_max=256, index_bits_in_kernel=8,
        d_multiple_of_4=False, rows_multiple_of_4=True, rungs=("R0",),
        scratch_model=lambda k, D, cbs: (k * D + cbs * D) * 4 + cbs * k * 4,
        refusal_sites=(
            ("rows % 4 != 0", "gravity_metal_lab_b.ll_plan: 'row4 needs rows % 4 == 0'"),
            ("scratch > limit", "gravity_metal_lab_b.ll_plan: 'cbs=... needs ... B of "
                                "threadgroup memory ... limit is ...'"),
            ("index >= k", "gravity_metal_lab_b.TrackBDecoder.upload: 'index ... out of range'"),
        ),
        provenance="GLM52_TRACK_B_BENCHMARK.json: SPLIT_BY_GEOMETRY_TABLE_BUILD_FRACTION_BINDS",
    ),
}

# The fourth candidate mandate 1 names.  It has no compiled kernel, and inventing one to
# fill a slot in a table would be a fabricated row, so it is declared here as what it is.
HYBRID_WITHIN_TENSOR = {
    "candidate": "hybrid_within_tensor",
    "status": "UNIMPLEMENTED_NOT_MEASURED",
    "reason": "mixing the two grammars WITHIN one matvec would need a kernel that builds "
              "the lookup table for some chunk blocks and decodes the rest per row in the "
              "same dispatch. No such kernel is compiled in this worktree. Both existing "
              "grammars already carry the 2D (row tile x chunk block) split, so the split "
              "is not the axis a within-tensor hybrid would add.",
    "what_does_exist": "hybrid_mixed_grammar_batch -- one command buffer per layer in which "
                       "each tensor is encoded with the grammar this matrix selected for its "
                       "geometry. That is cross-tensor, is buildable from "
                       "gravity_metal_lab_b.TrackBDecoder.run_batch today, and is MEASURED "
                       "below.",
}


# ------------------------------------------------------------------ table 2: which shapes to time

SHAPE_PRIORS: tuple[dict[str, Any], ...] = (
    {
        "grammar": "decode_fma_2d", "field": "blocks", "values": (1, 8, 32, 64),
        "measurement": f"{TRACK_A_REPORT}: median-of-medians by chunk-block count, "
                       "gate 0.9913 -> 0.3602 -> 0.3291 -> 0.3158 ms at blocks 1/8/32/64; "
                       "attention 2.2086 -> 0.5782 -> 0.4745 -> 0.4714. Saturation 32-64; "
                       "blocks=1 (the production grid: one dispatch, no reduce tax) is the "
                       "worst row at every geometry. blocks=1 is swept HERE because it is "
                       "the only way to put a GPU clock on the production grid -- "
                       "gravity_metal.matvec exposes no GPUEndTime.",
    },
    {
        "grammar": "decode_fma_2d", "field": "tpg", "values": (64, 256),
        "measurement": f"{TRACK_A_REPORT}: best shapes were gate tpg256/blk64, "
                       "down tpg64/blk64, attention tpg64/blk32.",
    },
    {
        "grammar": "lookup_linear", "field": "cbs", "values": (16, 32, 52),
        "measurement": f"{TRACK_B_REPORT}: the fp32 table caps cbs at 52 "
                       "(4096 + 544*cbs <= 32768); half-precision T raises it to 99 and "
                       "buys no time while costing 823x-1189x in parity, so it is not swept.",
    },
    {
        "grammar": "lookup_linear", "field": "tpg", "values": (256, 1024),
        "measurement": f"{TRACK_B_REPORT}: row4 (uchar4 load, 4 accumulators) is in every "
                       "winning variant at both ends; tpg sets rows per thread.",
    },
)

# Levers the sealed sweeps already closed.  Carried so the matrix says why it did not sweep
# them, rather than silently not sweeping them.
CLOSED_LEVERS: tuple[dict[str, Any], ...] = (
    {"lever": "native 7-bit unpack", "verdict": "NATIVE_7BIT_NEUTRAL_WITHIN_NOISE",
     "measurement": f"{TRACK_A_REPORT}: b7/u8 time ratio median 0.995/1.018/0.997 at the "
                    "three geometries; it removes 12.5% of the index stream, which is "
                    "0.4%-2.6% of executed bytes."},
    {"lever": "half-precision lookup table", "verdict": "REJECTED_PAYS_ACCURACY_FOR_NOTHING",
     "measurement": f"{TRACK_B_REPORT}: GPU ratio 0.999/0.998, parity 823x/1189x worse."},
    {"lever": "lookup table in device memory", "verdict": "NEGATIVE_CONTROL_KEPT",
     "measurement": f"{TRACK_B_REPORT}: 15.84x slower on GPU time at gate, 5.39x at down."},
    {"lever": "codebook sharing across experts", "verdict": "UNREACHABLE_FROM_CURRENT_ARTIFACTS",
     "measurement": f"{TRACK_B_REPORT}: 3.19x on GPU time with 16 real index streams on one "
                    "real book, but 60 of 60 codebooks on one shard hash distinctly. Needs a "
                    "full re-pack of 1.507 TB plus a new quality gate; quality cost UNMEASURED."},
    {"lever": "command-buffer batching", "verdict": "ORTHOGONAL_ALREADY_MEASURED",
     "measurement": f"{BASELINE_REPORT}: 1.18x gate/up, 1.58x down, entirely from removing "
                    "command buffers; per-dispatch GPU time does not improve with batching."},
)


# ------------------------------------------------------------------ table 3: the ranking

@dataclass(frozen=True)
class Criterion:
    key: str
    direction: str            # "min" | "max"
    role: str                 # "hard_filter" | "primary" | "tiebreak"
    band: float | None
    source: str


CRITERIA: tuple[Criterion, ...] = (
    Criterion("artifact_compatible", "max", "hard_filter", None,
              "mandate 2: a kernel that cannot consume the artifact is not a candidate "
              "regardless of speed. Evaluated by Kernel.accepts against the shard header."),
    Criterion("parity_relative_l2", "min", "hard_filter", PARITY_GATE,
              "gravity_metal.py's own tolerance, 2e-3. A 1e-6 gate would fail for the right "
              "reason and be misread: real codebooks are fp16 on disk so the kernel's fp16 "
              f"cast is a no-op ({BASELINE_REPORT}: codebook_fp16_roundtrip_lossless=true, "
              "max delta exactly 0.0)."),
    Criterion("latency_wall_median_ms", "min", "eliminate_only", 0.20,
              "what a caller pays, and the only metric all four implementations expose, so "
              "it is where production_v2 is judged. It ELIMINATES and never DECIDES: the "
              "measured 215.8 us command-buffer fixed cost is 70-90% of every candidate's "
              "wall at one matvec, so a wall gap of a few percent is host noise on a shared "
              f"constant, not a grammar fact ({TRACK_B_REPORT}: 'a shape chosen there is "
              "chosen under the wrong regime'). The 20% band is that constant's share made "
              "explicit: 20% of a ~0.27 ms wall is 54 us, about a quarter of the constant."),
    Criterion("latency_gpu_median_ms", "min", "primary", 0.05,
              "GPUEndTime-GPUStartTime from the driver: the component the GRAMMAR controls. "
              f"The 5% band is the campaign's standing rule on a contended box "
              f"({TRACK_B_REPORT}: verdicts require median AND min to clear +-5%). "
              "gravity_metal.matvec exposes no GPU clock, so production_v2 is dropped here "
              "with that reason recorded; its grid is measured under a clock as "
              "decode_fma_2d blocks=1."),
    Criterion("latency_gpu_min_ms", "min", "primary", 0.05,
              "the confirming statistic. Track B's law is that median AND min must both "
              "clear the band; when they disagree the candidates are inside noise and the "
              "decision falls through to the analytic criteria below."),
    Criterion("executed_total_bytes", "min", "tiebreak", None,
              "mandate 2 bandwidth criterion. ANALYTIC upper bound from "
              "gravity_metal_lab_b.ll_cost / dfma_cost / gravity_metal.matvec_bytes."),
    Criterion("executed_fp_ops", "min", "tiebreak", None,
              "mandate 2 arithmetic criterion. Same analytic ledgers."),
    Criterion("scratch_bytes", "min", "tiebreak", None,
              "mandate 2 resource criterion: threadgroup memory caps residency per core. "
              f"{BASELINE_REPORT}/diagnostic: 28,672 B of scratch capped a fused grid at "
              "~1 TG/core and cost 4.8x."),
)


def _stat(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None or value == lab.UNMEASURED else float(value)


def select_kernel(candidates: list[dict[str, Any]],
                  criteria: Iterable[Criterion] = CRITERIA) -> dict[str, Any]:
    """Rank measured candidate rows by CRITERIA.  No geometry knowledge lives here.

    ``candidates`` are dicts carrying at least ``kernel``, ``artifact_compatible``,
    ``incompatibility_reasons`` and whichever metric keys the criteria name.  Missing
    metrics are skipped rather than coerced, so a criterion nobody measured cannot decide.

    Three roles, and the difference between them is the whole point of the table:
    ``hard_filter`` removes what cannot run or cannot be trusted, ``eliminate_only``
    narrows the pool but may never announce a winner, ``primary``/``tiebreak`` may decide.
    """
    if not candidates:
        raise KernelSelectError("no candidates offered")
    trail: list[dict[str, Any]] = []
    pool = list(candidates)

    for criterion in criteria:
        if criterion.role == "hard_filter":
            if criterion.key == "artifact_compatible":
                keep = [c for c in pool if c.get("artifact_compatible")]
            else:
                keep = [c for c in pool
                        if _stat(c, criterion.key) is not None
                        and _stat(c, criterion.key) <= (criterion.band or float("inf"))]
            trail.append({
                "criterion": criterion.key, "role": criterion.role,
                "band": criterion.band, "source": criterion.source,
                "rejected": [{"kernel": c["kernel"],
                              "reason": c.get("incompatibility_reasons")
                              or f"{criterion.key}={_stat(c, criterion.key)}"}
                             for c in pool if c not in keep],
                "survivors": [c["kernel"] for c in keep],
            })
            pool = keep
            if not pool:
                raise KernelSelectError(
                    f"every candidate was rejected by hard filter {criterion.key}")
            continue

        scored = [(c, _stat(c, criterion.key)) for c in pool]
        unmeasured = [c["kernel"] for c, v in scored if v is None]
        scored = [(c, v) for c, v in scored if v is not None]
        if not scored:
            trail.append({"criterion": criterion.key, "role": criterion.role,
                          "outcome": "NOT_MEASURED_ON_ANY_SURVIVOR",
                          "source": criterion.source})
            continue
        best_value = min(v for _, v in scored) if criterion.direction == "min" \
            else max(v for _, v in scored)
        band = criterion.band or 0.0
        limit = best_value * (1 + band) if criterion.direction == "min" \
            else best_value * (1 - band)
        inside = [c for c, v in scored
                  if (v <= limit if criterion.direction == "min" else v >= limit)]
        decided = len(inside) == 1 and criterion.role != "eliminate_only"
        trail.append({
            "criterion": criterion.key, "role": criterion.role, "direction": criterion.direction,
            "band": criterion.band, "source": criterion.source,
            "best_value": best_value,
            "values": {c["kernel"]: v for c, v in scored},
            "dropped_for_no_measurement": unmeasured,
            "inside_band": [c["kernel"] for c in inside],
            "decided": decided,
        })
        pool = inside
        if decided:
            break

    winner = pool[0]
    return {
        "selected": winner["kernel"],
        "selected_config": winner.get("config"),
        "decided_by": next((t["criterion"] for t in trail if t.get("decided")),
                           "TIEBREAK_EXHAUSTED_FIRST_SURVIVOR"),
        "tie_survivors": [c["kernel"] for c in pool],
        "decision_trail": trail,
    }


# ------------------------------------------------------------------ artifact facts

def artifact_facts(fixture: grf.Fixture) -> dict[str, Any]:
    """What the SHARD says about this tensor.  Compatibility is judged against this."""
    codes = fixture.codes
    book = np.asarray(codes["codebooks"][0])
    return {
        "shard": fixture.shard, "tensor": fixture.tensor, "sha256": fixture.sha256,
        "rows": int(codes["rows"]), "cols": int(codes["cols"]),
        "nchunk": int(codes["nchunk"]), "D": int(codes["D"]), "S": int(codes["S"]),
        "sub": int(codes["sub"]), "k": int(book.shape[0]),
        "rotate": bool(codes["rotate"]),
        "rung": fixture.descriptor.get("rung"),
        "bpw": fixture.descriptor.get("bpw"),
        "index_bits_on_disk": 7,
        "index_bits_source": "R0 packs 7-bit indices; glm52_pack.pack_indices(.., 7)",
        "codebook_dtype_on_disk": "float16",
    }


def unique_read_bytes(*, rows: int, nchunk: int, D: int, k: int, blocks: int) -> int:
    """Lower bound: every distinct byte once, which is what a perfect cache would move.

    Track A refuted the re-read upper bound at the attention geometry by measuring above
    the 736 GB/s roof, so both bounds are carried and neither is called 'the' traffic.
    """
    partial = 0 if blocks <= 1 else rows * blocks * 4
    return rows * nchunk + k * D * 2 + nchunk * D * 4 + partial


# ------------------------------------------------------------------ candidate construction

def candidate_configs(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand SHAPE_PRIORS into the admissible variants for one geometry.

    Every value comes from the table; the only thing computed here is whether the device
    would refuse it, which is asked of the plan functions rather than re-derived.
    """
    priors = {(row["grammar"], row["field"]): row["values"] for row in SHAPE_PRIORS}
    rows, nchunk = facts["rows"], facts["nchunk"]
    D, k = facts["D"], facts["k"]
    out: list[dict[str, Any]] = []

    out.append({"kernel": "production_v2", "config": {"threads": gravity_metal.THREADS},
                "shape": None})

    for blocks in priors[("decode_fma_2d", "blocks")]:
        blocks = min(blocks, nchunk)
        cbs = (nchunk + blocks - 1) // blocks
        for tpg in priors[("decode_fma_2d", "tpg")]:
            try:
                shape = lb.dfma_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=cbs, tpg=tpg,
                                     threadgroup_memory_limit=THREADGROUP_MEMORY_LIMIT)
            except lb.TrackBError:
                continue
            out.append({"kernel": "decode_fma_2d",
                        "config": {"blocks": shape["blocks"], "cbs": cbs, "tpg": tpg},
                        "shape": shape})

    for cbs in priors[("lookup_linear", "cbs")]:
        cbs = min(cbs, nchunk)
        for tpg in priors[("lookup_linear", "tpg")]:
            try:
                shape = lb.ll_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=cbs, tpg=tpg,
                                   half_table=False, row4=(rows % 4 == 0),
                                   threadgroup_memory_limit=THREADGROUP_MEMORY_LIMIT)
            except lb.TrackBError:
                continue
            out.append({"kernel": "lookup_linear",
                        "config": {"cbs": cbs, "tpg": tpg, "blocks": shape["blocks"]},
                        "shape": shape})

    # de-duplicate shapes that collapsed onto each other after the nchunk clip
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    for entry in out:
        key = (entry["kernel"], json.dumps(entry["config"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def cost_of(kernel: str, facts: dict[str, Any], shape: dict[str, Any] | None,
            codes: dict[str, Any]) -> dict[str, Any]:
    """The analytic byte/op ledger for one candidate, from the module that owns it."""
    rows, cols = facts["rows"], facts["cols"]
    nchunk, D, k = facts["nchunk"], facts["D"], facts["k"]
    if kernel == "production_v2":
        raw = gravity_metal.matvec_bytes(
            codes, threadgroup_memory_limit=THREADGROUP_MEMORY_LIMIT)
        blocks = 1
        cost = {
            "grammar": "production_v2",
            "executed_fp_macs": rows * nchunk * D,
            "executed_fp_adds": 0,
            "executed_fp_ops": 2 * rows * nchunk * D,
            "executed_gather_ops": rows * nchunk,
            "executed_read_bytes": raw["executed_read_bytes"],
            "executed_total_bytes": raw["executed_total_bytes"],
            "dense_equivalent_macs": rows * nchunk * D,
            "arithmetic_reduction_vs_decode_fma": 1.0,
            "table_bytes_written_to_device": 0,
            "table_bytes_on_chip": 0,
            "logical_artifact_bytes": raw["logical_artifact_bytes"],
            "dense_bf16_bytes": raw["dense_bf16_bytes"],
            "executed_read_bpw": raw["executed_read_bpw"],
            "scratch_bytes": k * D * 4,
            "threadgroups": raw["threadgroups"],
            "threads_in_flight": rows,
            "stage_x": raw["stage_x"],
        }
    elif kernel == "decode_fma_2d":
        cost = dict(lb.dfma_cost(rows=rows, cols=cols, nchunk=nchunk, D=D, k=k, shape=shape))
        blocks = shape["blocks"]
        cost.update(scratch_bytes=shape["scratch_bytes"], threadgroups=shape["threadgroups"],
                    threads_in_flight=shape["threads_in_flight"], stage_x=True)
    elif kernel == "lookup_linear":
        cost = dict(lb.ll_cost(rows=rows, cols=cols, nchunk=nchunk, D=D, k=k, shape=shape))
        blocks = shape["blocks"]
        cost.update(scratch_bytes=shape["scratch_bytes"], threadgroups=shape["threadgroups"],
                    threads_in_flight=shape["threads_in_flight"], stage_x=True)
    else:
        raise KernelSelectError(f"no cost model for {kernel}")
    cost["unique_read_bytes"] = unique_read_bytes(
        rows=rows, nchunk=nchunk, D=D, k=k, blocks=blocks)
    cost["blocks"] = blocks
    return cost


def roofline_of(bytes_moved: int, seconds: float) -> dict[str, Any]:
    if seconds <= 0:
        return {"achieved_gb_s": lab.UNMEASURED, "fraction_of_bandwidth_roof": lab.UNMEASURED}
    gb_s = bytes_moved / seconds / 1e9
    return {"achieved_gb_s": gb_s, "fraction_of_bandwidth_roof": gb_s / BANDWIDTH_ROOF_GB_S,
            "exceeds_roof": gb_s > BANDWIDTH_ROOF_GB_S}


# ------------------------------------------------------------------ geometry census

def geometry_census() -> dict[str, Any]:
    """Every geometry class in a token, with its per-token multiplicity.

    Derived from gravity_flop_ledger.active_tensor_rows, not typed in, so the class list
    and the token weights cannot drift apart.
    """
    rows = fl.active_tensor_rows(fl.official_geometry())
    classes: dict[str, dict[str, Any]] = {}
    for row in rows:
        shape = tuple(row["shape"])
        if row["terminal_state"] != "PACKED_IN_CORE_ARTIFACT":
            key = f"NOT_KERNEL_BILLED::{row['terminal_state']}"
            slot = classes.setdefault(key, {
                "geometry_class": key, "organ": row["organ"], "shape": None,
                "per_token_tensors": 0, "active_bytes": 0, "kernel_device_bytes": 0,
                "dense_equivalent_macs": 0, "kernel_selectable": False,
            })
        else:
            key = f"{row['organ']}::{shape[0]}x{shape[1]}"
            slot = classes.setdefault(key, {
                "geometry_class": key, "organ": row["organ"], "shape": list(shape),
                "rows": shape[0], "cols": shape[1],
                "nchunk": fl.nchunk_for(shape[1], fl.R0_D),
                "per_token_tensors": 0, "active_bytes": 0, "kernel_device_bytes": 0,
                "dense_equivalent_macs": 0, "kernel_selectable": True,
            })
        slot["per_token_tensors"] += 1
        slot["active_bytes"] += row["active_bytes"]
        slot["kernel_device_bytes"] += row["kernel_device_bytes"]
        slot["dense_equivalent_macs"] += row["dense_equivalent_macs"]
    return {
        "source": "gravity_flop_ledger.active_tensor_rows(official_geometry())",
        "tensor_rows_per_token": len(rows),
        "production_token_bytes": sum(r["kernel_device_bytes"] for r in rows),
        "artifact_floor_bytes": sum(r["active_bytes"] for r in rows),
        "dense_equivalent_macs": sum(r["dense_equivalent_macs"] for r in rows),
        "classes": dict(sorted(classes.items())),
    }


# ------------------------------------------------------------------ adversarial index sets

def adversarial_indices(codes: dict[str, Any], mode: str) -> dict[str, Any]:
    """A copy of the codes with the index stream replaced.  Codebook untouched.

    ``all_same``  every lookup hits codeword 0: the hot extreme, one cache line for the
                  whole gather, and the worst case for a decode-FMA branch predictor claim.
    ``spread``    round-robin over all k codewords in index order: the cold extreme, every
                  threadgroup touches the whole codebook on every chunk.
    """
    if mode not in ("all_same", "spread"):
        raise KernelSelectError(f"unknown adversarial mode {mode!r}")
    out = dict(codes)
    k = int(np.asarray(codes["codebooks"][0]).shape[0])
    n = int(codes["rows"]) * int(codes["nchunk"])
    dtype = np.asarray(codes["indices"]).dtype
    flat = (np.zeros(n, dtype=dtype) if mode == "all_same"
            else (np.arange(n) % k).astype(dtype))
    out["indices"] = flat.reshape(-1, 1)
    return out


# ------------------------------------------------------------------ GPU driver

def _artifact_of(codes: dict[str, Any]) -> forge.PackedArtifact:
    ledger = forge.ByteLedger()
    ledger.add("indices", int(np.asarray(codes["indices"]).size) * 7)
    return forge.PackedArtifact(
        "product_quant", np.empty((0,), dtype=np.float32),
        int(codes["rows"]) * int(codes["cols"]), ledger, ledger.total_bits(), 0,
        {"pq_codes": codes})


def _parity(got: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    diff = reference - got
    return {
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-30)),
        "max_abs_error": float(np.abs(diff).max()),
        "cosine": float(reference @ got / ((np.linalg.norm(reference)
                                            * np.linalg.norm(got)) + 1e-30)),
        "finite": bool(np.isfinite(got).all()),
        "gate": PARITY_GATE,
    }


@dataclass
class Bench:
    """One geometry's GPU session: uploads reused, references cached, samples kept."""

    dec: Any
    prod: Any
    reps: int
    warmup: int
    gpu_seconds: float = 0.0

    def time(self, fn: Callable[[], Any], spec: lab.BenchSpec,
             gpu: bool) -> tuple[lab.TimingStats, lab.TimingStats | None]:
        start = time.perf_counter()
        if gpu:
            wall, gpu_stats = lb.measure_both(fn, spec, self.dec)
        else:
            wall, gpu_stats = lab.measure(fn, spec), None
        self.gpu_seconds += time.perf_counter() - start
        return wall, gpu_stats


def run_candidate(bench: Bench, facts: dict[str, Any], codes: dict[str, Any],
                  entry: dict[str, Any], x: np.ndarray, reference: np.ndarray,
                  spec: lab.BenchSpec, key: str) -> dict[str, Any]:
    """Compatibility, then parity, then -- only if both hold -- timing."""
    kernel = KERNELS[entry["kernel"]]
    ok, reasons = kernel.accepts(facts)
    row: dict[str, Any] = {
        "kernel": entry["kernel"], "config": entry["config"],
        "artifact_compatible": ok, "incompatibility_reasons": list(reasons),
    }
    if not ok:
        row["status"] = "REFUSED_ARTIFACT_INCOMPATIBLE"
        return row

    cost = cost_of(entry["kernel"], facts, entry["shape"], codes)
    row.update({k: cost[k] for k in ("executed_total_bytes", "executed_read_bytes",
                                     "executed_fp_ops", "executed_gather_ops",
                                     "scratch_bytes", "threadgroups", "threads_in_flight",
                                     "unique_read_bytes", "blocks")})
    row["cost_model"] = cost

    if entry["kernel"] == "production_v2":
        call = lambda: bench.prod.matvec(codes, x, key=key)          # noqa: E731
        gpu_timed = False
    else:
        max_blocks = max(1, entry["shape"]["blocks"])
        up = bench.dec.upload(codes, f"{key}::{max_blocks}", max_blocks=max_blocks)
        bench.dec.set_x(up, x)
        job = [(up, entry["shape"])]
        call = lambda: bench.dec.run_batch(job)                      # noqa: E731
        gpu_timed = True

    try:
        got = np.asarray(call()).ravel()
    except Exception as exc:                                          # noqa: BLE001
        row["status"] = "DISPATCH_FAILED"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["parity"] = _parity(got, reference)
    row["parity_relative_l2"] = row["parity"]["relative_l2"]
    if row["parity_relative_l2"] > PARITY_GATE:
        row["status"] = "NOT_TIMED_PARITY_FAILED"
        return row

    wall, gpu = bench.time(call, spec, gpu_timed)
    row["status"] = "TIMED"
    row["timing_wall_ms"] = lb.timing_json(wall)
    row["latency_wall_median_ms"] = wall.median_ms
    row["latency_wall_min_ms"] = wall.min_ms
    if gpu is not None:
        row["timing_gpu_ms"] = lb.timing_json(gpu)
        row["latency_gpu_median_ms"] = gpu.median_ms
        row["latency_gpu_min_ms"] = gpu.min_ms
    billed = gpu.median_ms if gpu is not None else wall.median_ms
    row["roofline_executed_upper_bound"] = roofline_of(cost["executed_total_bytes"],
                                                       billed / 1e3)
    row["roofline_unique_lower_bound"] = roofline_of(cost["unique_read_bytes"], billed / 1e3)
    row["roofline_billed_on"] = "gpu_execution median" if gpu is not None else "wall median"
    return row


def dense_baseline(bench: Bench, codes: dict[str, Any], x: np.ndarray,
                   spec: lab.BenchSpec) -> dict[str, Any]:
    """dense fp16 on MPS, the only thing this kernel has to beat to be worth shipping."""
    try:
        import torch
    except ImportError as exc:                                        # noqa: BLE001
        return {"status": "UNMEASURED", "reason": f"torch unavailable: {exc}"}
    if not torch.backends.mps.is_available():
        return {"status": "UNMEASURED", "reason": "MPS unavailable"}
    book = np.asarray(codes["codebooks"][0], dtype=np.float32)
    idx = np.asarray(codes["indices"])[:, 0]
    dense = book[idx].reshape(int(codes["rows"]), int(codes["cols"]))
    w = torch.from_numpy(dense).to("mps", torch.float16)
    xv = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("mps", torch.float16)
    del dense

    def call():
        y = w @ xv
        torch.mps.synchronize()
        return y

    wall, _ = bench.time(call, spec, gpu=False)
    out = {"status": "TIMED", "timing_wall_ms": lb.timing_json(wall),
           "latency_wall_median_ms": wall.median_ms, "latency_wall_min_ms": wall.min_ms,
           "bytes_moved": int(codes["rows"]) * int(codes["cols"]) * 2}
    out["roofline"] = roofline_of(out["bytes_moved"], wall.median_ms / 1e3)
    del w, xv
    torch.mps.empty_cache()
    return out


def run_geometry(name: str, fixture: grf.Fixture, bench: Bench,
                 *, with_dense: bool) -> dict[str, Any]:
    codes = fixture.codes
    facts = artifact_facts(fixture)
    x = fixture.activation(seed=7)
    reference = np.asarray(forge.pq_execute(_artifact_of(codes), x)).ravel()
    spec = lab.BenchSpec(
        rows=facts["rows"], cols=facts["cols"], batch=1, input_seed=7,
        input_dtype="float32", output_dtype="float32",
        warmup=bench.warmup, reps=bench.reps,
        sync_boundary="per_call_host_sync", dependency_shape="independent_calls",
        pack_in_timed_region=False, unpack_in_timed_region=False)

    candidates = [run_candidate(bench, facts, codes, entry, x, reference, spec,
                                fixture.cache_key)
                  for entry in candidate_configs(facts)]
    timed = [c for c in candidates if c["status"] == "TIMED"]
    if not timed:
        raise KernelSelectError(f"{name}: nothing survived to timing")

    # One row per kernel: its own best config, so the ranking compares GRAMMARS not shapes.
    # Ranked by the GPU clock where the kernel reports one, because picking the
    # representative on the noisy wall and then judging it on the GPU clock lets host
    # contention decide which shape represents a grammar -- which made the [2048,6144]
    # pick flip between runs.
    def rank_key(row: dict[str, Any]) -> float:
        return row.get("latency_gpu_median_ms", row["latency_wall_median_ms"])

    best_per_kernel: dict[str, dict[str, Any]] = {}
    for row in timed:
        cur = best_per_kernel.get(row["kernel"])
        if cur is None or rank_key(row) < rank_key(cur):
            best_per_kernel[row["kernel"]] = row
    for row in candidates:
        if not row["artifact_compatible"] and row["kernel"] not in best_per_kernel:
            best_per_kernel[row["kernel"]] = row

    decision = select_kernel(list(best_per_kernel.values()))
    chosen = best_per_kernel[decision["selected"]]

    result: dict[str, Any] = {
        "geometry_class": name,
        "fixture": fixture.as_json(),
        "artifact_facts": facts,
        "index_distribution": grf.index_distribution(codes),
        "activation_source": fixture.activation_source,
        "expert_selection": "FIXED_LIST_ROUTER_ABSENT_FROM_EVERY_SHARD",
        "candidates_timed": len(timed),
        "best_config_per_kernel": {k: v.get("config") for k, v in best_per_kernel.items()
                                   if v.get("status") == "TIMED"},
        "candidates": candidates,
        "decision": decision,
        "selected": {
            "kernel": chosen["kernel"], "config": chosen["config"],
            "parity": chosen["parity"],
            "latency_wall_ms": chosen["timing_wall_ms"],
            "latency_gpu_ms": chosen.get("timing_gpu_ms"),
            "executed_total_bytes": chosen["executed_total_bytes"],
            "unique_read_bytes": chosen["unique_read_bytes"],
            "scratch_bytes": chosen["scratch_bytes"],
            "threads_in_flight": chosen["threads_in_flight"],
            "roofline_executed_upper_bound": chosen["roofline_executed_upper_bound"],
            "roofline_unique_lower_bound": chosen["roofline_unique_lower_bound"],
        },
    }
    if with_dense:
        result["dense_fp16_mps"] = dense_baseline(bench, codes, x, spec)
        dense = result["dense_fp16_mps"]
        if dense.get("status") == "TIMED":
            result["selected"]["vs_dense_fp16_wall"] = {
                "median": dense["latency_wall_median_ms"] / chosen["latency_wall_median_ms"],
                "min": dense["latency_wall_min_ms"] / chosen["latency_wall_min_ms"],
            }
    prod = best_per_kernel.get("production_v2")
    if prod is not None and prod.get("status") == "TIMED":
        result["selected"]["vs_production_v2_wall"] = {
            "median": prod["latency_wall_median_ms"] / chosen["latency_wall_median_ms"],
            "min": prod["latency_wall_min_ms"] / chosen["latency_wall_min_ms"],
        }
    return result


def run_adversarial(fixture: grf.Fixture, bench: Bench, selected: str,
                    best_config: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Does the selection survive an index distribution the packer never produced?

    The two real ends (skewed gate_proj, near-uniform down_proj) are already covered by the
    geometry rows.  These two are constructed and impossible: every lookup on one codeword,
    and a perfectly spread stream.  Both grammars are re-timed on each; the verdict holds
    only if the same kernel still wins.
    """
    out: dict[str, Any] = {"fixture": fixture.as_json(), "selected_on_real_indices": selected,
                           "cases": {}}
    facts = artifact_facts(fixture)
    x = fixture.activation(seed=11)
    for mode in ("all_same", "spread"):
        codes = adversarial_indices(fixture.codes, mode)
        reference = np.asarray(forge.pq_execute(_artifact_of(codes), x)).ravel()
        spec = lab.BenchSpec(
            rows=facts["rows"], cols=facts["cols"], batch=1, input_seed=11,
            input_dtype="float32", output_dtype="float32",
            warmup=bench.warmup, reps=bench.reps,
            sync_boundary="per_call_host_sync", dependency_shape="independent_calls",
            pack_in_timed_region=False, unpack_in_timed_region=False)
        # every grammar at ITS OWN best shape on the real indices, so a rival cannot be
        # handicapped by an inherited config (blocks=1 is in the sweep; it is the
        # production grid and would lose for the wrong reason)
        keep = [e for e in candidate_configs(facts)
                if e["kernel"] != "production_v2"
                and e["config"] == best_config.get(e["kernel"])]
        rows = [run_candidate(bench, facts, codes, e, x, reference, spec,
                              f"{fixture.cache_key}::adv::{mode}")
                for e in keep]
        timed = [r for r in rows if r["status"] == "TIMED"]
        decision = select_kernel(timed) if timed else None
        out["cases"][mode] = {
            "index_distribution": grf.index_distribution(codes),
            "candidates_at_their_own_best_shape": [e["config"] for e in keep],
            "candidates": rows,
            "winner": decision["selected"] if decision else "NONE_TIMED",
            "selection_holds": bool(decision and decision["selected"] == selected),
            "construction": ("every lookup on codeword 0 (hot extreme)" if mode == "all_same"
                             else "round-robin over all k codewords (cold extreme)"),
        }
    return out


def run_hybrid_batch(jobs: list[tuple[grf.Fixture, str, dict[str, dict[str, Any]]]],
                     bench: Bench) -> dict[str, Any]:
    """The hybrid that actually exists: one command buffer, per-tensor grammar from the matrix.

    Compared against the same working set forced to a single grammar -- each at ITS OWN
    best measured shape for that geometry, not at an inherited one -- so the mixture is
    measured against the alternatives it replaces rather than against a handicapped rival.
    """
    def build(force: str | None) -> list[tuple[dict, dict]] | None:
        built = []
        for fixture, kernel, best in jobs:
            facts = artifact_facts(fixture)
            use = force or kernel
            config = best.get(use)
            if config is None:
                return None
            entries = [e for e in candidate_configs(facts)
                       if e["kernel"] == use and e["config"] == config]
            if not entries:
                return None
            entry = entries[0]
            up = bench.dec.upload(fixture.codes, f"{fixture.cache_key}::{entry['shape']['blocks']}",
                                  max_blocks=entry["shape"]["blocks"])
            bench.dec.set_x(up, fixture.activation(seed=3))
            built.append((up, entry["shape"]))
        return built

    spec = lab.BenchSpec(
        rows=sum(artifact_facts(f)["rows"] for f, _, _ in jobs),
        cols=sum(artifact_facts(f)["cols"] for f, _, _ in jobs),
        batch=1, input_seed=3, input_dtype="float32", output_dtype="float32",
        warmup=bench.warmup, reps=bench.reps,
        sync_boundary="per_batch_gpu_fence", dependency_shape="independent_calls",
        pack_in_timed_region=False, unpack_in_timed_region=False)

    out: dict[str, Any] = {
        "definition": "one command buffer, each tensor encoded with the grammar this matrix "
                      "selected for its geometry",
        "tensors": [{"tensor": f.tensor, "grammar": k, "config": b.get(k)}
                    for f, k, b in jobs],
        "variants": {},
    }
    for label, force in (("hybrid_mixed_grammar", None),
                         ("all_decode_fma_2d", "decode_fma_2d"),
                         ("all_lookup_linear", "lookup_linear")):
        built = build(force)
        if built is None:
            out["variants"][label] = {"status": "NOT_APPLICABLE_TO_EVERY_GEOMETRY"}
            continue
        call = lambda b=built: bench.dec.run_batch(b)                # noqa: E731
        try:
            call()
        except Exception as exc:                                      # noqa: BLE001
            out["variants"][label] = {"status": "DISPATCH_FAILED",
                                      "error": f"{type(exc).__name__}: {exc}"}
            continue
        wall, gpu = bench.time(call, spec, gpu=True)
        out["variants"][label] = {
            "status": "TIMED", "command_buffers": 1,
            "timing_wall_ms": lb.timing_json(wall), "timing_gpu_ms": lb.timing_json(gpu),
            "latency_wall_median_ms": wall.median_ms,
            "latency_gpu_median_ms": gpu.median_ms,
        }
    timed = {k: v for k, v in out["variants"].items() if v.get("status") == "TIMED"}
    if "hybrid_mixed_grammar" in timed:
        base = timed["hybrid_mixed_grammar"]["latency_gpu_median_ms"]
        out["hybrid_gpu_speedup_vs_single_grammar"] = {
            k: v["latency_gpu_median_ms"] / base
            for k, v in timed.items() if k != "hybrid_mixed_grammar"}
        out["verdict"] = ("HYBRID_WINS" if all(r > 1.05 for r in
                                               out["hybrid_gpu_speedup_vs_single_grammar"].values())
                          else "HYBRID_WITHIN_NOISE_OF_BEST_SINGLE_GRAMMAR")
    return out


# ------------------------------------------------------------------ mandate 6: whole token

def shape_stability(geometries: dict[str, Any]) -> dict[str, Any]:
    """Do two tensors of the SAME shape get the same grammar?

    They must, or the matrix is reading contention rather than geometry.  This is the
    matrix auditing itself: the selection rule only ever sees measured rows, so two rows
    from the same shape are an internal replicate that nobody had to plan for.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, g in geometries.items():
        facts = g["artifact_facts"]
        key = f"{facts['rows']}x{facts['cols']}"
        gpu = {c["kernel"]: c.get("latency_gpu_median_ms")
               for c in g["candidates"] if c["status"] == "TIMED"
               and c.get("latency_gpu_median_ms") is not None
               and c["config"] == g["best_config_per_kernel"].get(c["kernel"])}
        groups.setdefault(key, []).append(
            {"geometry_class": name, "selected": g["selected"]["kernel"],
             "gpu_median_ms_by_kernel": gpu,
             "decided_by": g["decision"]["decided_by"]})
    out = {}
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        picks = {r["selected"] for r in rows}
        spreads = []
        for r in rows:
            vals = [v for v in r["gpu_median_ms_by_kernel"].values() if v]
            if len(vals) >= 2:
                spreads.append(max(vals) / min(vals))
        out[key] = {
            "replicates": rows,
            "agree": len(picks) == 1,
            "grammar_gpu_spread_ratio": spreads,
            "verdict": ("STABLE" if len(picks) == 1 else
                        "UNSTABLE_GRAMMARS_ARE_INSIDE_THE_NOISE_BAND_AT_THIS_SHAPE"),
        }
    return {
        "model": "two real tensors of one shape are an internal replicate of the selection",
        "shapes": out,
        "verdict": ("EVERY_REPLICATED_SHAPE_AGREES" if all(v["agree"] for v in out.values())
                    else "AT_LEAST_ONE_SHAPE_SPLITS_BETWEEN_GRAMMARS_INSIDE_NOISE"),
        "consequence": "where a shape splits, the two grammars are interchangeable at that "
                       "geometry and the byte ledger, not the clock, should pick.",
    }


def token_ledger(census: dict[str, Any], selections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Re-bill every tensor a token reads under the kernel this matrix selected for it.

    Classes with no measured tensor (nothing on a safe shard) are billed by the same rule
    with ``selection_basis = EXTRAPOLATED``, and are listed so the extrapolation is visible
    rather than folded into a total.
    """
    executed = unique = 0
    rows: list[dict[str, Any]] = []
    for key, entry in census["classes"].items():
        if not entry["kernel_selectable"]:
            executed += entry["active_bytes"]
            unique += entry["active_bytes"]
            rows.append({"geometry_class": key, "kernel": "NOT_KERNEL_BILLED",
                         "per_token_tensors": entry["per_token_tensors"],
                         "bytes_per_token": entry["active_bytes"],
                         "selection_basis": "carried at source precision"})
            continue
        pick = selections.get(key)
        if pick is None:
            rows.append({"geometry_class": key, "kernel": "UNSELECTED_NO_SAFE_TENSOR",
                         "per_token_tensors": entry["per_token_tensors"],
                         "bytes_per_token": entry["kernel_device_bytes"],
                         "selection_basis": "PRODUCTION_KERNEL_BYTES_CARRIED_UNCHANGED"})
            executed += entry["kernel_device_bytes"]
            unique += entry["kernel_device_bytes"]
            continue
        n = entry["per_token_tensors"]
        executed += pick["executed_total_bytes"] * n
        unique += pick["unique_read_bytes"] * n
        rows.append({
            "geometry_class": key, "kernel": pick["kernel"], "config": pick["config"],
            "per_token_tensors": n,
            "executed_bytes_per_tensor": pick["executed_total_bytes"],
            "unique_bytes_per_tensor": pick["unique_read_bytes"],
            "executed_bytes_per_token": pick["executed_total_bytes"] * n,
            "unique_bytes_per_token": pick["unique_read_bytes"] * n,
            "production_bytes_per_token": entry["kernel_device_bytes"],
            "selection_basis": pick.get("selection_basis", "MEASURED_ON_A_REAL_TENSOR"),
        })
    floor = census["artifact_floor_bytes"]
    production = census["production_token_bytes"]
    return {
        "model": "ANALYTIC byte model per kernel, multiplied by the ledger's per-token counts",
        "rows": rows,
        "production_kernel_bytes_per_token": production,
        "selected_executed_bytes_per_token": executed,
        "selected_unique_bytes_per_token": unique,
        "artifact_floor_bytes_per_token": floor,
        "reduction_vs_production_executed": production / executed if executed else None,
        "reduction_vs_production_unique": production / unique if unique else None,
        "executed_over_floor": executed / floor,
        "unique_over_floor": unique / floor,
        "bandwidth_roof_gb_s": BANDWIDTH_ROOF_GB_S,
        "token_ceiling_tok_s": {
            "production_kernel": 1e9 * BANDWIDTH_ROOF_GB_S / production,
            "selected_executed_upper_bound": 1e9 * BANDWIDTH_ROOF_GB_S / executed,
            "selected_unique_lower_bound": 1e9 * BANDWIDTH_ROOF_GB_S / unique,
            "artifact_floor": 1e9 * BANDWIDTH_ROOF_GB_S / floor,
        },
        "caveat": "the executed model is an upper bound whose re-read terms the caches "
                  "partly serve; Track A measured above the 736 GB/s roof under it at the "
                  "attention geometry. True traffic lies between executed and unique.",
    }


# ------------------------------------------------------------------ orchestration

CLASS_FOR_FIXTURE = {
    ("routed_expert", "gate"): "routed_expert::2048x6144",
    ("routed_expert", "up"): "routed_expert::2048x6144",
    ("routed_expert", "down"): "routed_expert::6144x2048",
    ("shared_expert", "gate"): "shared_expert::2048x6144",
    ("shared_expert", "up"): "shared_expert::2048x6144",
    ("shared_expert", "down"): "shared_expert::6144x2048",
}


def collect_fixtures(layer: int | None) -> list[tuple[str, grf.Fixture]]:
    """One real tensor per geometry class, from safe shards only."""
    index = grf.layer_index()
    layers = grf.executable_layers(index)
    if not layers:
        raise KernelSelectError("no safe layer is MoE-executable")
    chosen = layer if layer is not None else layers[0]
    entry = index[chosen]
    root = Path(grf.ARTIFACT_DIR)
    picked: list[tuple[str, grf.Fixture]] = []

    expert = entry["complete_experts"][0]
    for projection in ("gate", "down"):
        name = f"model.layers.{chosen}.mlp.experts.{expert}.{projection}_proj.weight"
        fixture = grf._fixture(root / entry["experts"][str(expert)][projection], name)
        picked.append((CLASS_FOR_FIXTURE[("routed_expert", projection)], fixture))
    for projection in ("gate", "down"):
        name = f"model.layers.{chosen}.mlp.shared_experts.{projection}_proj.weight"
        fixture = grf._fixture(root / entry["shared_expert"][projection], name)
        picked.append((CLASS_FOR_FIXTURE[("shared_expert", projection)], fixture))
    for name, shard in sorted(entry["attention"].items()):
        fixture = grf._fixture(root / shard, name)
        rows, cols = fixture.shape
        picked.append((f"attention::{rows}x{cols}", fixture))

    # dense MLP lives only in layers < first_k_dense_replace; find it wherever it is safe
    for path in grf.safe_shards(root):
        header = __import__("gravity_format").read_header(path)
        for tensor in header["tensors"]:
            kind, _, _, _ = grf.classify(tensor["name"])
            if kind != "dense_mlp":
                continue
            rows, cols = tensor["shape"]
            key = f"dense_mlp::{rows}x{cols}"
            if any(k == key for k, _ in picked):
                continue
            picked.append((key, grf._fixture(path, tensor["name"])))
        if sum(1 for k, _ in picked if k.startswith("dense_mlp")) >= 2:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    started = time.time()
    census = geometry_census()
    if census["production_token_bytes"] != PRODUCTION_TOKEN_BYTES:
        raise KernelSelectError(
            f"ledger says {census['production_token_bytes']} B/token, this file was written "
            f"against {PRODUCTION_TOKEN_BYTES}")
    if census["artifact_floor_bytes"] != ARTIFACT_FLOOR_BYTES:
        raise KernelSelectError(
            f"ledger floor {census['artifact_floor_bytes']} != {ARTIFACT_FLOOR_BYTES}")

    fixtures = collect_fixtures(args.layer)
    bench = Bench(dec=lb.TrackBDecoder(), prod=gravity_metal.decoder(),
                  reps=args.reps, warmup=args.warmup)

    geometries: dict[str, Any] = {}
    selections: dict[str, dict[str, Any]] = {}
    hybrid_jobs: list[tuple[grf.Fixture, str, dict[str, dict[str, Any]]]] = []
    for name, fixture in fixtures:
        result = run_geometry(name, fixture, bench, with_dense=True)
        geometries[name] = result
        pick = result["selected"]
        selections[name] = {
            "kernel": pick["kernel"], "config": pick["config"],
            "executed_total_bytes": pick["executed_total_bytes"],
            "unique_read_bytes": pick["unique_read_bytes"],
        }
        hybrid_jobs.append((fixture, pick["kernel"], result["best_config_per_kernel"]))

    # the two real index ends the shards actually carry, plus the two constructed extremes
    skewed = next(f for k, f in fixtures if k == "routed_expert::2048x6144")
    uniform = next(f for k, f in fixtures if k == "routed_expert::6144x2048")
    adversarial = {
        "real_skewed_gate_proj": {
            "tensor": skewed.tensor,
            "index_distribution": grf.index_distribution(skewed.codes),
            "selected": selections["routed_expert::2048x6144"]["kernel"],
            "note": "the real hot end; measured as a geometry row above",
        },
        "real_near_uniform_down_proj": {
            "tensor": uniform.tensor,
            "index_distribution": grf.index_distribution(uniform.codes),
            "selected": selections["routed_expert::6144x2048"]["kernel"],
            "note": "the real cold end; measured as a geometry row above",
        },
        "constructed_on_skewed_geometry": run_adversarial(
            skewed, bench, selections["routed_expert::2048x6144"]["kernel"],
            geometries["routed_expert::2048x6144"]["best_config_per_kernel"]),
        "constructed_on_uniform_geometry": run_adversarial(
            uniform, bench, selections["routed_expert::6144x2048"]["kernel"],
            geometries["routed_expert::6144x2048"]["best_config_per_kernel"]),
    }
    holds = all(case["selection_holds"]
                for block in ("constructed_on_skewed_geometry", "constructed_on_uniform_geometry")
                for case in adversarial[block]["cases"].values())
    adversarial["verdict"] = ("SELECTION_HOLDS_UNDER_EVERY_ADVERSARIAL_DISTRIBUTION" if holds
                              else "SELECTION_FLIPS_UNDER_AT_LEAST_ONE_DISTRIBUTION")

    hybrid = run_hybrid_batch(hybrid_jobs, bench)
    ledger = token_ledger(census, selections)

    report = {
        "report": "GLM52_KERNEL_SELECTION_MATRIX",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds": time.time() - started,
        "gpu_seconds_in_timed_regions": bench.gpu_seconds,
        "machine_facts": lab.MACHINE_FACTS,
        "activation_source": grf.SYNTHETIC,
        "activation_status": grf.teacher_activation_status(),
        "expert_selection": "FIXED_LIST_ROUTER_ABSENT_FROM_EVERY_SHARD",
        "bench_spec_policy": {
            "reps": args.reps, "warmup": args.warmup,
            "law": "gravity_bench_lab.BenchSpec; every candidate for one geometry shares one "
                   "spec, so no comparison crosses differing specs",
            "statistics": "min and median are the trustworthy readings on this contended "
                          "box; p95 and max are contention and are carried, never averaged",
        },
        "tables": {
            "kernels": {name: k.to_json() for name, k in KERNELS.items()},
            "shape_priors": list(SHAPE_PRIORS),
            "closed_levers": list(CLOSED_LEVERS),
            "criteria": [{"criterion": c.key, "direction": c.direction, "role": c.role,
                          "band": c.band, "source": c.source} for c in CRITERIA],
            "hybrid_within_tensor": HYBRID_WITHIN_TENSOR,
        },
        "geometry_census": census,
        "geometries": geometries,
        "selection_summary": {
            name: {
                "shape": g["artifact_facts"]["rows"] and
                         [g["artifact_facts"]["rows"], g["artifact_facts"]["cols"]],
                "selected": g["selected"]["kernel"],
                "config": g["selected"]["config"],
                "decided_by": g["decision"]["decided_by"],
                "parity_relative_l2": g["selected"]["parity"]["relative_l2"],
                "latency_median_ms": g["selected"]["latency_wall_ms"]["median"],
                "latency_min_ms": g["selected"]["latency_wall_ms"]["min"],
                "gpu_median_ms": (g["selected"]["latency_gpu_ms"] or {}).get("median"),
                "achieved_gb_s_executed_upper": g["selected"][
                    "roofline_executed_upper_bound"]["achieved_gb_s"],
                "fraction_of_roof_executed_upper": g["selected"][
                    "roofline_executed_upper_bound"]["fraction_of_bandwidth_roof"],
                "achieved_gb_s_unique_lower": g["selected"][
                    "roofline_unique_lower_bound"]["achieved_gb_s"],
                "vs_dense_fp16_wall": g["selected"].get("vs_dense_fp16_wall"),
                "vs_production_v2_wall": g["selected"].get("vs_production_v2_wall"),
            } for name, g in geometries.items()
        },
        "artifact_compatibility_gate": {
            "candidates_offered": sum(len(g["candidates"]) for g in geometries.values()),
            "candidates_refused_as_incompatible": sum(
                1 for g in geometries.values() for c in g["candidates"]
                if not c["artifact_compatible"]),
            "finding": "the gate is live and never fires on THESE artifacts: every safe "
                       "shard is rung R0 (S=1, rotate=False, k=128, D=8, 7-bit indices) and "
                       "every tensor has rows % 4 == 0, so all three grammars can decode "
                       "everything. It fires in the unit tests, which is the only place a "
                       "rotated / multi-subspace / k>256 / odd-rows artifact exists.",
            "what_would_be_refused": {
                "rotated artifact": "all three -- no kernel has a rotation path",
                "S > 1": "all three",
                "k > 256": "all three -- the index buffer is uint8",
                "D not a multiple of 4": "decode_fma_2d only (float4 inner loop)",
                "rows not a multiple of 4": "lookup_linear only (uchar4 row tile)",
            },
        },
        "shape_stability": shape_stability(geometries),
        "adversarial": adversarial,
        "hybrid_mixed_grammar_batch": hybrid,
        "whole_token_ledger": ledger,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out} ({args.out.stat().st_size} B) "
          f"gpu_timed={bench.gpu_seconds:.1f}s wall={report['wall_seconds']:.1f}s")
    return 0


def selftest() -> int:
    """Pure-python checks: the tables answer without a GPU."""
    r0 = {"S": 1, "rotate": False, "k": 128, "D": 8, "rows": 2048, "cols": 6144,
          "nchunk": 768, "index_bits_on_disk": 7, "rung": "R0"}
    for name, kernel in KERNELS.items():
        ok, reasons = kernel.accepts(r0)
        assert ok, (name, reasons)
    assert not KERNELS["lookup_linear"].accepts({**r0, "rows": 2049})[0]
    assert not KERNELS["production_v2"].accepts({**r0, "rotate": True})[0]
    decision = select_kernel([
        {"kernel": "a", "artifact_compatible": True, "parity_relative_l2": 1e-7,
         "latency_wall_median_ms": 1.0},
        {"kernel": "b", "artifact_compatible": True, "parity_relative_l2": 1e-7,
         "latency_wall_median_ms": 0.5},
    ])
    assert decision["selected"] == "b", decision
    census = geometry_census()
    assert census["production_token_bytes"] == PRODUCTION_TOKEN_BYTES
    assert census["artifact_floor_bytes"] == ARTIFACT_FLOOR_BYTES
    print("gravity_kernel_select selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
