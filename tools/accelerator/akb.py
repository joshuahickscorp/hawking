#!/usr/bin/env python3
"""Accelerator Knowledge Base: the campaign's measured laws as TYPED entries.

The 77 ``receipts/headless/ACCELERATOR_*.json`` receipts are real and adversarial,
but the laws inside them are English prose in ``headline`` and ``claim_boundary``.
Prose does not transfer. This module records each law as a typed entry carrying the
EXACT applicability domain the evidence reaches, and refuses an entry whose claim is
broader than its evidence.

The central rule, from the steer this campaign runs under: never record ``X IS
FASTER``. Record ``X WAS FASTER FOR primitive x shape x representation x machine x
runtime``. :func:`validate` enforces that mechanically.

Conventions reused rather than reinvented:

* ``tools/accelerator/receipt.py`` -- the eight-identity discipline, and in
  particular ABSENT-with-a-reason. An axis that does not apply is ``NONE`` here and
  must be backed by the source receipt actually recording that identity ABSENT. A
  missing field never reads as a covered one.
* ``tools/headless/cross_model_laws.py`` -- the ``path/to/receipt.json#json.path``
  citation form, and a store that REFUSES a promotion its evidence does not carry.

An axis may be ``UNSCOPED`` only when the source receipt genuinely establishes the
claim across that axis, and the entry must cite the field that shows the breadth.
``UNKNOWN`` is legal and honest, and is NOT the same as ``UNSCOPED``: unknown means
nobody looked, unscoped means it was looked at across values and did not matter.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
SCHEMA = "hawking.accelerator.akb.v1"

# This lane's own artifacts. They live in receipts/headless and match the corpus glob,
# so they are named here and excluded from it. ACCELERATOR_KNOWLEDGE_BASE.json is NOT
# ours -- it is the live organ-shape base written by odyssey_pass.py, which this lane
# is forbidden to touch, so the law base is a separate file and that receipt stays in
# the corpus as an ordinary input.
OWN_OUTPUTS = frozenset({"ACCELERATOR_LAW_BASE.json", "ACCELERATOR_AKB_CONSTRUCTION.json"})

# The applicability domain. Every entry names all eleven; none may be absent.
AXES = ("MODEL", "ARCHITECTURE", "ORGAN", "REPRESENTATION", "SHAPE", "MACHINE",
        "RUNTIME", "KERNEL", "STORAGE_TIER", "TOPOLOGY", "WORKLOAD_PHASE")

# Axis values with a meaning beyond "the single value it was measured at".
UNSCOPED = "UNSCOPED"   # established across this axis; needs unscoped_basis evidence
UNKNOWN = "UNKNOWN"     # nobody measured this axis; honest, and not a breadth claim
NONE = "NONE"           # the axis does not apply; the receipt records it ABSENT

# Axes that map onto a receipt identity, so a NONE claim is checkable against it.
IDENTITY_OF_AXIS = {"MODEL": "model", "REPRESENTATION": "representation",
                    "MACHINE": "machine", "RUNTIME": "runtime",
                    "KERNEL": "kernel", "TOPOLOGY": "transport"}

EVIDENCE_CLASSES = ("Simulated", "Derived", "Measured", "Reproduced", "ProtectedVerified")
STATUSES = ("ACTIVE", "REFUTED", "SUPERSEDED", "CONDITIONAL")

# "X is faster" with no domain. The steer forbids the present-tense universal.
BARE_COMPARATIVE = re.compile(
    r"\b(is|are)\s+(the\s+)?(faster|slower|better|worse|best|fastest|slowest)\b", re.I)

RECEIPT_NAME = re.compile(r"ACCELERATOR_[A-Z0-9_]+\.json")


class Refused(Exception):
    """The validator refused an entry. Raised, never returned, never logged-and-continued."""


# --------------------------------------------------------------------------- corpus

def corpus(root: Path = REPO) -> list[Path]:
    """Every Accelerator receipt on disk, sorted. The real 77, not a fixture.

    Excludes this module's OWN outputs. They match the corpus glob, and without this
    the base would ingest itself: each build would add a receipt, and the AKB would
    end up citing itself as evidence for its own laws.
    """
    return sorted(p for p in (root / "receipts/headless").glob("ACCELERATOR_*.json")
                  if p.name not in OWN_OUTPUTS)


def resolve(citation: str, root: Path = REPO) -> Any:
    """Resolve ``relative/path.json#a.b.c`` to its value, or refuse.

    Refuses a missing file and a missing field differently, because "the receipt is
    gone" and "the receipt never said that" are different mistakes.
    """
    rel, _, jp = citation.partition("#")
    f = root / rel
    if not f.exists():
        raise Refused(f"citation names a receipt that does not exist: {rel}")
    cur = json.loads(f.read_text())
    if not jp:
        return cur
    for part in jp.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit() and abs(int(part)) < len(cur) + 1:
            cur = cur[int(part)]
        else:
            raise Refused(f"citation {citation} does not resolve: no {part!r}")
    return cur


def superseding_corpus(paths: list[Path] | None = None) -> dict[str, list[str]]:
    """Receipts this corpus supersedes itself on, and by what.

    Two mechanisms, both real in the corpus: a receipt carrying its own
    ``AMENDED*`` key, and a LATER receipt naming an earlier one in
    ``boundary_this_closes``.
    """
    out: dict[str, list[str]] = {}
    for f in paths or corpus():
        d = json.loads(f.read_text())
        amended = [k for k in d if "AMEND" in k.upper()]
        if amended:
            out.setdefault(f.name, []).extend(f"{f.name}#{k}" for k in amended)
        closes = d.get("boundary_this_closes")
        if closes:
            for target in RECEIPT_NAME.findall(json.dumps(closes)):
                if target != f.name:
                    out.setdefault(target, []).append(f"{f.name}#boundary_this_closes")
    return out


# --------------------------------------------------------------------------- validator

def _distinct_count(value: Any) -> int:
    """How many distinct values a cited breadth field actually shows."""
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, (list, tuple)):
        return len({json.dumps(v, sort_keys=True) for v in value})
    return 1


def validate(entry: dict[str, Any], *, superseded: dict[str, list[str]] | None = None,
             root: Path = REPO) -> dict[str, Any]:
    """Refuse an entry whose claim is broader than its evidence. Returns it if it holds."""
    superseded = superseding_corpus() if superseded is None else superseded
    lid = entry.get("law_id", "<no law_id>")

    for field in ("law_id", "statement", "applicability", "evidence_class",
                  "source_receipts", "status", "negative_result", "confidence_basis"):
        if field not in entry:
            raise Refused(f"{lid}: entry is missing required field {field!r}")
    if entry["evidence_class"] not in EVIDENCE_CLASSES:
        raise Refused(f"{lid}: {entry['evidence_class']!r} is not an evidence class")
    if entry["status"] not in STATUSES:
        raise Refused(f"{lid}: {entry['status']!r} is not a status")

    # NC2 -- the bare universal comparative the steer forbids.
    if BARE_COMPARATIVE.search(entry["statement"]):
        raise Refused(
            f"{lid}: statement makes a bare present-tense comparative claim "
            f"({BARE_COMPARATIVE.search(entry['statement']).group(0)!r}). Record what the "
            f"evidence supports: X WAS FASTER FOR primitive x shape x representation x "
            f"machine x runtime.")

    # Every axis named, none silently absent.
    missing = [a for a in AXES if a not in entry["applicability"]]
    if missing:
        raise Refused(f"{lid}: applicability is missing axes {missing}")
    extra = [a for a in entry["applicability"] if a not in AXES]
    if extra:
        raise Refused(f"{lid}: applicability names axes that do not exist {extra}")

    # NC3 -- every citation must resolve, including the source receipts themselves.
    for rel in entry["source_receipts"]:
        resolve(rel, root=root)
    for cit in entry.get("citations", []):
        resolve(cit, root=root)

    # NC1 -- UNSCOPED needs evidence of breadth, not a claim of it.
    basis = entry.get("unscoped_basis", {})
    for axis, value in entry["applicability"].items():
        if value != UNSCOPED:
            continue
        if axis not in basis:
            raise Refused(
                f"{lid}: axis {axis} is UNSCOPED with no unscoped_basis citation. An axis "
                f"is UNSCOPED only when the receipt establishes the claim ACROSS it.")
        n = _distinct_count(resolve(basis[axis], root=root))
        if n < 2:
            raise Refused(
                f"{lid}: axis {axis} is UNSCOPED but its basis {basis[axis]} shows only "
                f"{n} value. One measured value is not breadth -- name the value, or "
                f"say UNKNOWN.")

    # NONE must be grounded in the receipt recording that identity ABSENT.
    for axis, value in entry["applicability"].items():
        if value != NONE or axis not in IDENTITY_OF_AXIS:
            continue
        ident = IDENTITY_OF_AXIS[axis]
        for rel in entry["source_receipts"]:
            ids = resolve(rel.partition("#")[0], root=root).get("identities")
            if ids is None:
                continue  # receipt predates the identity schema; recorded, not assumed
            got = ids.get(ident)
            if not (isinstance(got, dict) and got.get("status") in ("ABSENT", "MOCK", "SIMULATED")):
                raise Refused(
                    f"{lid}: axis {axis} claims NONE but {rel} records identity "
                    f"{ident!r} as {json.dumps(got)[:120]}, not ABSENT.")

    # NC4 -- a superseded source may not be served as ACTIVE.
    if entry["status"] == "ACTIVE":
        for rel in entry["source_receipts"]:
            name = Path(rel.partition("#")[0]).name
            if name in superseded:
                raise Refused(
                    f"{lid}: status ACTIVE but source {name} is superseded by "
                    f"{superseded[name]}. A law resting on an amended or closed receipt is "
                    f"CONDITIONAL at best.")

    # NC5 -- Measured may not rest on a receipt that did not pass.
    if entry["evidence_class"] == "Measured":
        for rel in entry["source_receipts"]:
            d = resolve(rel.partition("#")[0], root=root)
            if d.get("pass") is False:
                raise Refused(
                    f"{lid}: evidence_class Measured but source {rel} records pass: false. "
                    f"A failed run is evidence of failure, not of the measurement.")

    if entry["status"] == "SUPERSEDED" and not entry.get("superseded_by"):
        raise Refused(f"{lid}: status SUPERSEDED with no superseded_by link")
    return entry


# --------------------------------------------------------------------------- the laws
#
# Hand-extracted from receipts that were READ. Every number below is quoted from the
# cited field, not paraphrased from a summary. Where a law could not be extracted
# honestly the receipt is in UNEXTRACTED with the reason, and a short honest base
# beats a long invented one.

M3 = "Apple M3 Ultra, 60 GPU cores, 96 GiB unified (this box)"
MLX = "MLX 0.32.1 metal_kernel JIT under CPython 3.12"

LAWS: list[dict[str, Any]] = [
    dict(
        law_id="AKB-SCAN-VS-CUMSUM",
        statement=(
            "A three-phase AIR scan (simd_prefix) WAS 8.09x faster than mx.cumsum FOR a "
            "1-D f32 inclusive prefix sum AT n=2**24 ON this M3 Ultra UNDER MLX 0.32.1: "
            "0.8662 ms against 7.0038 ms. The mechanism named in the receipt is that "
            "mx.cumsum reaches 19.16 GB/s on a machine measured at 589.73 GB/s, so this "
            "is a statement about MLX's cumsum at this shape, NOT a general claim that "
            "these kernels beat MLX -- the same receipt records MLX as the qualified "
            "winner at GEMM, softmax, attention and reduction."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "n = 2**24, 1-D",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "air.lower_scan_block_to_msl + lower_scan_offset_to_msl, simd_prefix",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_SCAN.json"],
        citations=["receipts/headless/ACCELERATOR_SCAN.json#result.performance.16777216.ms_per_call",
                   "receipts/headless/ACCELERATOR_SCAN.json#result.performance.16777216.gbps_2n",
                   "receipts/headless/ACCELERATOR_SCAN.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=("200 reps, 40 warmup, both arms reliable at 3.79% and 0.38% IQR. "
                          "SHAPE is deliberately not UNSCOPED: the receipt measures 2**20 and "
                          "2**24 only, and states the ratio is about MLX's cumsum on THIS shape."),
    ),
    dict(
        law_id="AKB-WAIT-DOMINATES-SUBMISSION",
        statement=(
            "For an MLX matmul on this box, the per-step cost of WAITING dominated the cost "
            "of SUBMITTING. At 128x128 f32 the wait arm read 184.7 / 188.3 / 179.1 us per "
            "step across three batteries (4.9% spread, verdict SYNC_TAX all three), while a "
            "confound arm that varies SUBMISSION COUNT alone did not reproduce that reading. "
            "The direction is what transfers; the microsecond figures are INSTANCE values on "
            "a machine contended by the ModelLake fill and MUST NOT be quoted as constants."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "matmul 128x128 (and 2048x2048, unstable)",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "mlx lazy graph + Metal; no custom kernel",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Reproduced",
        source_receipts=["receipts/headless/ACCELERATOR_SYNC_DIAGNOSTIC.json"],
        citations=["receipts/headless/ACCELERATOR_SYNC_DIAGNOSTIC.json#result.run_to_run.per_arm.held_out_small",
                   "receipts/headless/ACCELERATOR_SYNC_DIAGNOSTIC.json#result.negative_controls",
                   "receipts/headless/ACCELERATOR_SYNC_DIAGNOSTIC.json#claim_boundary"],
        status="CONDITIONAL", superseded_by=None, negative_result=False,
        confidence_basis=(
            "CONDITIONAL, not ACTIVE, because the instrument's own run_to_run block records "
            "that the 2048x2048 arm did NOT reproduce -- NO_SIGNIFICANT_SYNC_TAX then "
            "SYNC_TAX twice, stable=false, detection floor swinging 76-324 us. The small "
            "shape reproduced 3/3; the large one did not, and a law that holds at one shape "
            "of two is conditional on shape. The instrument also returns a NEGATIVE where "
            "one is true, which is what distinguishes it from a confirmation device: the "
            "NumPy CPU arm reads -0.083 / 0.055 / -0.050 us per step, NO_SIGNIFICANT_SYNC_TAX "
            "stable across all three batteries."),
    ),
    dict(
        law_id="AKB-GEMM-BLOCK2-PARITY",
        statement=(
            "The first sweep's claim that AIR's blocked simdgroup matmul reached 149.5% of "
            "MLX at 1024 IS RETRACTED. Careful remeasurement at 120 reps / 40 warmup put "
            "block2 at 5800.09 GFLOP/s against MLX's 5800.73 -- 0.9999, exact parity. The "
            "apparent win was MLX's own arm being slow in a noisy run (3963 GFLOP/s noisy "
            "vs 5801 clean), not a candidate win."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "square GEMM, 1024x1024x1024",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "air.lower_matmul_to_msl strategy='simdgroup' block=2",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Reproduced",
        source_receipts=["receipts/headless/ACCELERATOR_REGISTER_BLOCKING.json"],
        citations=["receipts/headless/ACCELERATOR_REGISTER_BLOCKING.json#result.a_result_i_retracted",
                   "receipts/headless/ACCELERATOR_REGISTER_BLOCKING.json#result.careful_remeasurement.1"],
        status="REFUTED", superseded_by=None, negative_result=True,
        confidence_basis=("The retraction is the result. A 1.495x win reported under a 19.89% "
                          "IQR arm was refused by the reliability gate and did not survive "
                          "remeasurement. Kept as REFUTED, not deleted."),
    ),
    dict(
        law_id="AKB-MACHINE-BANDWIDTH",
        statement=(
            "This box's memory bandwidth WAS a 589.73 GB/s median FOR an f32 triad "
            "(c = a + b) at 67108864 elements, 30 reps, 8 warmup, with the model-lake fill "
            "SIGSTOPped. The receipt states plainly it is NOT the SoC theoretical roof, NOT "
            "a workload-reachable roof and NOT sustained. It is a property of THIS machine "
            "and must never be inherited by another one."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "67108864 elements, triad",
            "MACHINE": M3, "RUNTIME": MLX, "KERNEL": "mlx elementwise add",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_MACHINE_GENOME.json"],
        citations=["receipts/headless/ACCELERATOR_MACHINE_GENOME.json#result.measured_bandwidth.median_gb_s",
                   "receipts/headless/ACCELERATOR_MACHINE_GENOME.json#result.measured_bandwidth.is_theoretical_roof",
                   "receipts/headless/ACCELERATOR_MACHINE_GENOME.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=("1.89% IQR, reliable. Knowledge level INSTANCE in the source: one "
                          "M3 Ultra is not evidence about M3 Ultras, so MACHINE is the single "
                          "measured box and is not promoted to a SoC family."),
    ),
    dict(
        law_id="AKB-DISPATCH-VS-SUBMISSION",
        statement=(
            "Eager submission cost WAS linear in dispatch count at 0.16783 ms per node "
            "(R2 = 0.99999); batched into one command buffer the marginal added node fell "
            "to 0.01084 ms. So ON this machine UNDER this runtime a within-command-buffer "
            "dispatch cost about 11 microseconds and a cross-command-buffer submission "
            "about 157 microseconds, a 14.5x ratio. The actionable threshold the receipt "
            "draws is that a kernel doing under ~0.16 ms of GPU work is submission-bound."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32",
            "SHAPE": "elementwise, 2**20 and 4096, k up to 16",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "air.AirGraph / air.execute_graph, identical elementwise dispatches",
            "STORAGE_TIER": NONE, "TOPOLOGY": UNSCOPED, "WORKLOAD_PHASE": "dispatch"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json"],
        citations=["receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json#result.headline_THE_MACHINE_CONSTANT",
                   "receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json#result.linear_fits"],
        unscoped_basis={"TOPOLOGY": "receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json#result.linear_fits"},
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=("TOPOLOGY is UNSCOPED here and the basis is cited: the sweep fits "
                          "BOTH wirings -- chain (depth k, width 1) and fan (depth 1, width k) "
                          "-- so the submission constant is established across topology rather "
                          "than assumed across it. Every other axis stays at its measured value."),
    ),
    dict(
        law_id="AKB-LAUNCH-COUNT-NOT-THE-MECHANISM",
        statement=(
            "The published claim that GPU KERNEL LAUNCH COUNT drove the reduction result IS "
            "REFUTED by direct decomposition. Holding syncs at one and varying only launch "
            "count gave NO_CLAIM at 2**20 and 1.019x -- indistinguishable -- at 2**24. "
            "Holding launches at three and varying only the sync count gave 2.30x at 2**20 "
            "and 1.28x at 2**24, essentially the whole originally-reported effect. The cost "
            "WAS host round trips and command-buffer boundaries, not kernel launches."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "f32 sum reduction and scan, 2**20 and 2**24",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "air.lower_reduce_to_msl two-stage, three dispatches",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "dispatch"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_SYNC_CORRECTION.json"],
        citations=["receipts/headless/ACCELERATOR_SYNC_CORRECTION.json#result.correction_1_launch_count_is_REFUTED",
                   "receipts/headless/ACCELERATOR_SYNC_CORRECTION.json#result.correction_2_the_real_mechanism_is_HOST_ROUND_TRIPS",
                   "receipts/headless/ACCELERATOR_SYNC_CORRECTION.json#result.correction_3_the_recommendation_FLIPS"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=("A refutation of this campaign's own published mechanism, by "
                          "factorial decomposition rather than by re-running the same "
                          "confounded arm. The receipt also records the blast radius."),
    ),
    dict(
        law_id="AKB-IQR-GATE-UNSTABLE-UNDER-200-REPS",
        statement=(
            "This campaign's own 10% IQR reliability gate WAS unstable below roughly 200 "
            "reps FOR the fused mul->relu->silu chain at 2**24 f32 ON this machine: at 20 "
            "and 40 reps the gate verdict flipped for every candidate probed, with "
            "tg256_ept2's IQR estimate ranging 1.84%-13.38% across eight 20-rep runs. The "
            "instrument, not the kernel, was the variable."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "2**24 f32 elementwise chain",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "tg256_ept2, tg512_ept1, tg1024_ept4 (three forge variants)",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_IQR_STABILITY_AUDIT.json"],
        citations=["receipts/headless/ACCELERATOR_IQR_STABILITY_AUDIT.json#result.the_finding",
                   "receipts/headless/ACCELERATOR_IQR_STABILITY_AUDIT.json#result.what_it_means_for_earlier_work"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=("Eight estimates per cell across three kernels. The source states "
                          "explicitly that 200 is NOT a universal threshold -- it is this "
                          "workload's -- so SHAPE and KERNEL stay named, not UNSCOPED."),
    ),
    dict(
        law_id="AKB-FUSION-BEATS-MATERIALISING",
        statement=(
            "A single fused AIR program WAS 1.71x faster than three separate translated "
            "kernels and 2.62x faster than MLX FOR the chain mul->relu->silu AT 2**24 f32 "
            "ON this machine UNDER MLX 0.32.1. The receipt attributes the win to FUSION and "
            "says so: MLX does not fuse this chain and materialises two intermediates the "
            "fused program never writes, and the operation is memory bound, so removing "
            "passes is the entire mechanism. It is explicitly NOT a claim that Hawking "
            "kernels beat MLX kernels in general."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "2**24 f32, chain mul->relu->silu",
            "MACHINE": M3, "RUNTIME": MLX, "KERNEL": "1 fused AIR program vs 3 AIR kernels vs MLX",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_FRONT_D_P5.json"],
        citations=["receipts/headless/ACCELERATOR_FRONT_D_P5.json#result.fused_vs_naive",
                   "receipts/headless/ACCELERATOR_FRONT_D_P5.json#result.fused_vs_mlx",
                   "receipts/headless/ACCELERATOR_FRONT_D_P5.json#result.correctness_max_abs_err"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=("Both verdicts CANDIDATE_WINS with arm noise 5.92% and 4.08% against "
                          "margins of 70.99% and 162.2%. Correctness checked before timing at "
                          "9.537e-07. Microbenchmark only -- the receipt notes no sustained or "
                          "WorkUnit evidence, so WORKLOAD_PHASE is not promoted."),
    ),
    dict(
        law_id="AKB-UNIFIED-MEMORY-COPY-ELIMINATION",
        statement=(
            "Eliminating the per-call host<->device copies WAS worth 7.17x-7.44x FOR the "
            "fused mul->relu->silu chain at 2**24 f32 ON this unified-memory machine. The "
            "receipt states the fairness caveat itself: the compared arm copies on EVERY "
            "call, which is the naive CUDA-era pattern, and a competent CUDA programmer "
            "hoists the upload out of the loop and would not pay this. The honest claim is "
            "NOT that Apple beats CUDA by 7x -- it is that per-call copying costs this much "
            "on this operation, and that on unified memory the elimination is structural "
            "rather than something the programmer must remember."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "2**24 f32, chain mul->relu->silu",
            "MACHINE": M3, "RUNTIME": MLX, "KERNEL": "fused mul->relu->silu, identical in both arms",
            "STORAGE_TIER": NONE, "TOPOLOGY": "single-device unified memory, no discrete GPU present",
            "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_FRONT_G_P6.json"],
        citations=["receipts/headless/ACCELERATOR_FRONT_G_P6.json#result.speedups",
                   "receipts/headless/ACCELERATOR_FRONT_G_P6.json#result.reliability_gate",
                   "receipts/headless/ACCELERATOR_FRONT_G_P6.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=("Three runs at 7.171, 7.401, 7.439. TOPOLOGY is named, not NONE: the "
                          "whole law is about there being no discrete device memory to cross, "
                          "which is a topology fact and is why it does not transfer."),
    ),
    dict(
        law_id="AKB-CONCURRENCY-IS-REGIME-DEPENDENT",
        statement=(
            "Process concurrency WAS worth 3.38x aggregate at 8 processes FOR a 4 MiB "
            "launch-bound elementwise kernel ON this machine, while this campaign's prior "
            "whole-model-body measurement topped out around 1.21x. These are NOT in "
            "conflict and the receipt says why: a whole body has a working set of tens of "
            "GiB and is bandwidth- and admission-bound, whereas a 4 MiB kernel is "
            "launch-bound, so extra processes fill launch gaps instead of competing for "
            "bandwidth. Concurrency helped when launch-bound and did not when "
            "bandwidth-bound. Per-process efficiency at 8 was only 0.423, so a profile "
            "optimising work-per-resource would choose differently from one optimising "
            "aggregate throughput."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "4194304 elements (4 MiB), launch-bound regime",
            "MACHINE": M3, "RUNTIME": MLX, "KERNEL": "tg256_ept2",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "concurrent raw kernel rate"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_CONCURRENCY_SWEEP.json"],
        citations=["receipts/headless/ACCELERATOR_CONCURRENCY_SWEEP.json#result.sweep",
                   "receipts/headless/ACCELERATOR_CONCURRENCY_SWEEP.json#result.knee",
                   "receipts/headless/ACCELERATOR_CONCURRENCY_SWEEP.json#result.reconciles_with_prior_campaign_law"],
        status="CONDITIONAL", superseded_by=None, negative_result=False,
        confidence_basis=(
            "CONDITIONAL because the law has two arms and only ONE was measured here. The "
            "3.38x at 4 MiB is measured in this receipt. The ~1.21x whole-body figure is a "
            "PRIOR CITED BY the receipt from an earlier campaign and was NOT re-measured in "
            "this corpus, so it is recorded as the receipt's own reconciliation text and "
            "must not be served as an Accelerator measurement. SHAPE is the conditioning "
            "axis and is therefore never UNSCOPED."),
    ),
    dict(
        law_id="AKB-SPEC-CONSTANTS-BUY-NOTHING",
        statement=(
            "Metal specialization constants BOUGHT nothing measurable FOR two probed shapes "
            "ON this backend: scalar literal versus buffer read was INDISTINGUISHABLE, and "
            "the compute-bound loop bound where unrolling should have paid was also "
            "INDISTINGUISHABLE with the literal 3.09% slower, both arms clean. The receipt "
            "records that its author's prediction was wrong in both loop cases and that the "
            "explanation offered for the first died in the second."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32",
            "SHAPE": "2**22 elementwise multiply and a 2**20 x 256 fma loop",
            "MACHINE": M3, "RUNTIME": "CPython 3.12.6 driving MLX-emitted kernel signatures",
            "KERNEL": "hand-written MSL pairs differing only in literal vs buffer constant",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_SPEC_CONSTANTS.json"],
        citations=["receipts/headless/ACCELERATOR_SPEC_CONSTANTS.json#headline",
                   "receipts/headless/ACCELERATOR_SPEC_CONSTANTS.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=("200 reps, 40 warmup, bit-identical arms verified before timing "
                          "(max_abs_diff exactly 0.0). The receipt names the untested case "
                          "that could differ: a constant enabling branch elimination or a "
                          "different memory layout. A null result, kept because it is one."),
    ),
    dict(
        law_id="AKB-SIMDGROUP-ATTENTION-LOSES-AT-THIS-BLOCKING",
        statement=(
            "Simdgroup matrix ops in attention WERE slower at both sizes measured ON this "
            "machine FOR a blocking of one simdgroup per 8 query rows, and the mechanism WAS "
            "occupancy rather than instruction throughput. The receipt is explicit that this "
            "does NOT show matrix ops are useless in attention: more simdgroups per "
            "threadgroup, a larger head_dim, or a flash-style online softmax removing the "
            "memory cap could reverse it. It shows THIS blocking loses, and that the earlier "
            "receipt's stated reason was wrong."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": "attention",
            "REPRESENTATION": "dense_f32", "SHAPE": "two sizes, single head, head_dim 64",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "simdgroup_multiply_accumulate for Q@K^T and P@V, one simdgroup per 8 query rows",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "microbenchmark"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_ATTENTION_SIMDGROUP_REFUTED.json"],
        citations=["receipts/headless/ACCELERATOR_ATTENTION_SIMDGROUP_REFUTED.json#result.verdict",
                   "receipts/headless/ACCELERATOR_ATTENTION_SIMDGROUP_REFUTED.json#result.the_real_mechanism",
                   "receipts/headless/ACCELERATOR_ATTENTION_SIMDGROUP_REFUTED.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=("A negative result carried as a first-class entry. KERNEL names the "
                          "specific blocking because the blocking IS the scope of the claim."),
    ),
    dict(
        law_id="AKB-SPARSE-RESIDUAL-BEATEN-PER-BIT",
        statement=(
            "A sparse residual on top of ws_rtn_q4_g64 WAS beaten 2.5x-3.9x PER BIT by "
            "spending the same bits on a finer group or one more level, FOR 8 real Qwen3 "
            "routed-expert gate_proj tensors in WEIGHT space. The prediction that motivated "
            "the arm was refuted, and the diagnosis the receipt then confirms is that the "
            "outlier hurts by SETTING THE GROUP SCALE, not by being stored badly, so the "
            "error it causes is dense and a sparse fix cannot reach it."),
        applicability={
            "MODEL": "Qwen3-30B-A3B", "ARCHITECTURE": "Qwen3 MoE", "ORGAN": "routed expert gate_proj",
            "REPRESENTATION": "ws_rtn_q4_g64 base vs q4_g32 / q5_g64 / q4_g64 + sparse residual",
            "SHAPE": "8 tensors, [768, 2048] bf16",
            "MACHINE": M3, "RUNTIME": MLX, "KERNEL": "AirSparseMatvec (CSR), verified not timed",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "weight-space fidelity"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_SPARSE.json"],
        citations=["receipts/headless/ACCELERATOR_SPARSE.json#headline",
                   "receipts/headless/ACCELERATOR_SPARSE.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=("Weight space only -- no activations, no model executed, no adequacy "
                          "claim. The source pins itself at MODEL_FAMILY at most and points at "
                          "the organ-floor receipt showing organ ranking does not transfer "
                          "across architectures, so ARCHITECTURE is named, never UNSCOPED."),
    ),
    dict(
        law_id="AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER",
        statement=(
            "The prior law that ATTENTION SETS THE REPRESENTATION FLOOR IS "
            "ARCHITECTURE-SPECIFIC, not general. Attention set the floor on BOTH Qwen models "
            "-- q_proj needed group 32 where every MLP and expert tensor cleared at group "
            "128 -- and DID NOT transfer: on Kimi-VL and Falcon-H1 no organ was more "
            "demanding than any other. The mechanism is within-group outliers, at Pearson "
            "-0.9841 across 33 organs and 4 architectures."),
        applicability={
            "MODEL": UNSCOPED, "ARCHITECTURE": UNSCOPED, "ORGAN": UNSCOPED,
            "REPRESENTATION": "grouped absmax; bits=4 group=64 is exactly ws_rtn_q4_g64",
            "SHAPE": "256 rows of one tensor per organ per specimen",
            "MACHINE": M3, "RUNTIME": "CPython 3.12.6, numpy quantize_grouped",
            "KERNEL": NONE, "STORAGE_TIER": "TIER 2 /Volumes/corpdrive model lake",
            "TOPOLOGY": NONE, "WORKLOAD_PHASE": "weight-space fidelity"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json"],
        citations=["receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json#headline",
                   "receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json#claim_boundary"],
        unscoped_basis={
            "MODEL": "receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json#identities.model.specimens",
            "ARCHITECTURE": "receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json#identities.model.census",
            "ORGAN": "receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json#identities.model.census"},
        status="CONDITIONAL", superseded_by=None, negative_result=True,
        confidence_basis=(
            "MODEL, ARCHITECTURE and ORGAN are UNSCOPED here and each cites its breadth: 4 "
            "specimens and a per-specimen organ census of 9/7/8/11. This is the one law in "
            "the base whose CONTENT is non-transfer, so breadth across those axes is exactly "
            "what the evidence establishes. CONDITIONAL rather than ACTIVE because the source "
            "carries an AMENDED_IN_PLACE_2026_08_25 key -- the validator refuses ACTIVE here "
            "and it is right to. Weight space only; the source pins itself at MODEL_FAMILY."),
    ),
    dict(
        law_id="AKB-JIT-COMPILE-WAS-MISATTRIBUTED",
        statement=(
            "Three receipts blamed MLX KERNEL COMPILE for first-run cost and none measured "
            "it. Measured, compile WAS 36.8 ms per distinct kernel and accounted for 23.2% "
            "of a WorkUnit's first-run excess, while the dominant term WAS the first "
            "mx.array at 151 ms -- device and allocator initialisation. About 21% of the "
            "excess remains unattributed and the receipt says so rather than distributing it "
            "over the terms it did measure."),
        applicability={
            "MODEL": "Qwen3-30B-A3B", "ARCHITECTURE": "Qwen3 MoE", "ORGAN": "routed expert gate_proj",
            "REPRESENTATION": "ws_rtn_q4_g64", "SHAPE": "64 expert gate_proj tensors",
            "MACHINE": M3, "RUNTIME": "mx.fast.metal_kernel JIT under CPython 3.12.6",
            "KERNEL": "unchanged; this measures how long it takes to BUILD one",
            "STORAGE_TIER": UNKNOWN, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "first-run / process startup"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_JIT_COMPILE.json"],
        citations=["receipts/headless/ACCELERATOR_JIT_COMPILE.json#headline",
                   "receipts/headless/ACCELERATOR_JIT_COMPILE.json#full_decomposition_of_the_first_run_excess",
                   "receipts/headless/ACCELERATOR_JIT_COMPILE.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=("JIT only -- xcrun metal AOT is absent on this machine so no AOT "
                          "comparison exists, and nvrtc is unmeasured because no NVIDIA "
                          "hardware is present. These are per-process costs amortised to zero "
                          "by the second WorkUnit, which is exactly how they contaminated "
                          "three earlier receipts' baselines."),
    ),
    dict(
        law_id="AKB-CRC32-VS-SHA256",
        statement=(
            "crc32 WAS 12.6x faster than sha256 and 26.9x faster than blake2b FOR bulk "
            "checksumming ON this machine: 36.849, 2.932 and 1.368 GB/s respectively. This "
            "is the one number in the HUMF fabric work that is measured rather than a knob; "
            "every transport rate in the same receipt is a NAMED ASSUMPTION because no "
            "external GPU exists here."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32 payload bytes", "SHAPE": "bulk buffer checksum",
            "MACHINE": M3, "RUNTIME": "CPython 3.12.6", "KERNEL": NONE,
            "STORAGE_TIER": NONE, "TOPOLOGY": "MOCK_EXTERNAL_VRAM, simulated",
            "WORKLOAD_PHASE": "integrity check"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_HUMF_SILENT_CORRUPTION.json"],
        citations=["receipts/headless/ACCELERATOR_HUMF_SILENT_CORRUPTION.json#result.measured_checksum_rates_on_this_machine",
                   "receipts/headless/ACCELERATOR_HUMF_SILENT_CORRUPTION.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "The checksum rates ARE measured; the affordability conclusion built on them in "
            "the same receipt is NOT, because it is conditional on a transport speed nobody "
            "here has measured. Only the measured half is recorded as a law. The receipt's "
            "own headline ratio is 14.2x sha256; the 12.6x above is recomputed from the "
            "cited GB/s field, and the discrepancy is carried rather than smoothed."),
    ),
    dict(
        law_id="AKB-ERROR-DETECTION-IS-NOT-IDENTITY",
        statement=(
            "A crc32 integrity check PASSED ON WRONG BYTES twice, for two unrelated reasons. "
            "crc32 can be REPAIRED in 0.035 ms by a linear solve, so a payload with a "
            "flipped weight byte and four adjusted padding bytes was accepted, marked CLEAN "
            "and counted TRUSTED. And the check compares SOURCE to DESTINATION, so a source "
            "that rotted in place was compared against a faithful copy of its own rot and "
            "passed -- after which the two copies agree and every later check confirms the "
            "corruption. Neither is a flaw in crc32: error detection is not identity."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "opaque bytes at real object sizes", "SHAPE": UNKNOWN,
            "MACHINE": M3, "RUNTIME": "CPython 3.12.6", "KERNEL": NONE,
            "STORAGE_TIER": NONE, "TOPOLOGY": "MockExternalMemoryProvider, not physical",
            "WORKLOAD_PHASE": "integrity check"},
        evidence_class="Simulated",
        source_receipts=["receipts/headless/ACCELERATOR_HUMF_IDENTITY.json"],
        citations=["receipts/headless/ACCELERATOR_HUMF_IDENTITY.json#headline",
                   "receipts/headless/ACCELERATOR_HUMF_IDENTITY.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "evidence_class Simulated, not Measured: the fabric is a mock, bandwidth is a "
            "knob and no external device exists. What is established is the STATE MACHINE's "
            "behaviour and the SCOPE of the check, and the receipt names the blind spot that "
            "remains -- the destination check validates copy_out, not what a KERNEL reads."),
    ),
    dict(
        law_id="AKB-A-CHECK-THAT-CANNOT-FAIL",
        statement=(
            "The natural correctness predicate for a top-k sampler -- 'the returned index is "
            "one of the top k' -- PASSES for a sampler that always returns the argmax. A "
            "sampler cannot be graded the way every other kernel in this corpus was graded, "
            "because the property that matters is distributional and the obvious check is "
            "satisfied by a degenerate implementation."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": "decode tail (logits -> token)",
            "REPRESENTATION": NONE, "SHAPE": "synthetic logits, rows 32 x cols 4096, k=8",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "AirTopKSample: k rounds of full-row argmax, then a serial CDF walk",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "correctness grading"},
        evidence_class="Derived",
        source_receipts=["receipts/headless/ACCELERATOR_TOPK_SAMPLING.json"],
        citations=["receipts/headless/ACCELERATOR_TOPK_SAMPLING.json#headline",
                   "receipts/headless/ACCELERATOR_TOPK_SAMPLING.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "evidence_class Derived: this is a property of the PREDICATE, established by "
            "source inspection plus the receipt's own demonstration, not by a timing run. "
            "Nothing was timed in the source and no baseline was run, so no performance axis "
            "is claimed."),
    ),
    dict(
        law_id="AKB-CONTROL-LOUDNESS-IS-NOT-TRANSFERABLE",
        statement=(
            "A correctness control's SENSITIVITY is a property of the defect, the width and "
            "the shape together, not of the sweep. On the same width list, same repeats and "
            "same machine, a barrier on a TOTAL dependency fired in 40 of 48 runs and was "
            "blind at 1 of 6 widths; a barrier on an INCIDENTAL dependency fired in 4 of 48 "
            "and was blind at 4 of 6 -- a 10x difference in loudness. A sweep whose only "
            "control is loud reports a blind list of length 1 and thereby implies a "
            "resolving power it does not have."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32",
            "SHAPE": "AirTopKSample rows 32 cols 4096; AirNorm rms; widths 32-1024",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "AirNorm (loud control) and AirTopKSample (quiet control)",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "correctness grading"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_QUIET_CONTROL.json"],
        citations=["receipts/headless/ACCELERATOR_QUIET_CONTROL.json#headline",
                   "receipts/headless/ACCELERATOR_QUIET_CONTROL.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "The source states its own limit: two controls is one point at each end of a "
            "spectrum whose middle is unmeasured, and blind_at_every_control is the union "
            "over the controls that RAN, not a proof no defect could hide there."),
    ),
    dict(
        law_id="AKB-LOCKSTEP-AT-THREADGROUP-32",
        statement=(
            "A barrier stripped from AirNorm WAS caught 8 of 8 runs at threadgroups 64 "
            "through 1024 and WAS EXACT at 32 ON this M3 Ultra, because at 32 the whole "
            "threadgroup is one simdgroup and lockstep makes the fence unnecessary. A "
            "positive control run only at width 32 would have reported the broken kernel "
            "fine. The receipt refuses to promote this: lockstep at simd width 32 is an M3 "
            "Ultra observation, not a property of Apple GPUs."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32", "SHAPE": "one shape per primitive; widths 32-1024",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "AirNorm rms with its first threadgroup_barrier stripped",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "correctness grading"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_WIDTH_SWEEP.json"],
        citations=["receipts/headless/ACCELERATOR_WIDTH_SWEEP.json#headline",
                   "receipts/headless/ACCELERATOR_WIDTH_SWEEP.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "42 configurations, 336 executions, zero wrong. The source is blunt that ZERO "
            "FAILURES IS NOT ABSENCE OF RACES: repeat=8 detects a per-run probability above "
            "~8.3% about half the time, so a 2% race would not be caught."),
    ),
    dict(
        law_id="AKB-DEVICE-RESIDENT-OPERANDS",
        statement=(
            "Keeping the correctness gate AND the operands on the device WAS worth 238.56% "
            "end to end -- 635,422 to 2,151,304 WorkUnits/hour, 3.39x -- FOR the "
            "pack-and-verify pipeline over 64 real Qwen3 expert gate_proj tensors from TIER "
            "1 storage ON this machine. Three arms separated the two changes: the gate "
            "moving to the device gave +58.31% and the operands staying there +113.86% on "
            "top, both steps completely separated and inside the 10% IQR gate. No kernel "
            "changed; what moved is WHERE THE OPERANDS LIVE."),
        applicability={
            "MODEL": "Qwen3-30B-A3B", "ARCHITECTURE": "Qwen3 MoE", "ORGAN": "routed expert gate_proj",
            "REPRESENTATION": "ws_rtn_q4_g64, 4.25 bpw", "SHAPE": "64 tensors, 768x2048, bf16 on disk",
            "MACHINE": M3, "RUNTIME": "CPython 3.12.6, MLX primitives",
            "KERNEL": "native matvec and pack MSL byte-identical to the prior block",
            "STORAGE_TIER": "TIER 1 ~/noetic/stage, internal SSD",
            "TOPOLOGY": NONE, "WORKLOAD_PHASE": "sustained pack-and-verify pipeline"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json"],
        citations=["receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json#result.C_vs_A_pct",
                   "receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json#result.separation_B_over_A",
                   "receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json#result.separation_C_over_B",
                   "receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "Arm spreads 2.52/5.86/4.04% all inside the 10% gate, both steps completely "
            "separated, and arm A reproduced a different process's baseline to 0.18%. But "
            "6 samples per arm is below this program's own 200-rep reliability bar, and the "
            "source says nothing here makes ws_rtn_q4_g64 adequate for a MODEL -- local "
            "adequacy does not compose."),
    ),
    dict(
        law_id="AKB-TIER1-CACHE-INFLATION-IS-SMALL",
        statement=(
            "The page-cache caveat three receipts carried WAS small: a cold read of the "
            "staged shard measured 6265.8 MB/s with 99.76% of pages actually coming from "
            "disk, against 8318.4 MB/s cached -- a ratio of only 1.33x. The pipeline's "
            "WorkUnits/hour was inflated by cache residency at the level of a few percent, "
            "not by a factor."),
        applicability={
            "MODEL": "Qwen3-30B-A3B", "ARCHITECTURE": "Qwen3 MoE", "ORGAN": "routed expert gate_proj",
            "REPRESENTATION": NONE, "SHAPE": "192 MB read as 64 sequential 3 MB chunks",
            "MACHINE": M3, "RUNTIME": "CPython 3.12.6, vm_stat Pageins, no sudo",
            "KERNEL": NONE, "STORAGE_TIER": "TIER 1 ~/noetic/stage, internal APPLE SSD",
            "TOPOLOGY": NONE, "WORKLOAD_PHASE": "cold vs cached sequential read"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_TIER1_CACHE_VS_DISK.json"],
        citations=["receipts/headless/ACCELERATOR_TIER1_CACHE_VS_DISK.json#the_measurement",
                   "receipts/headless/ACCELERATOR_TIER1_CACHE_VS_DISK.json#claim_boundary"],
        status="CONDITIONAL", superseded_by=None, negative_result=False,
        confidence_basis=(
            "CONDITIONAL because n=1 for the number that matters. The cold read is a SINGLE "
            "sample with no spread -- a direct measurement with a 99.76% pagein fraction "
            "proving it was cold, but not a characterised disk rate. The lake fill was "
            "running, which makes 6266 MB/s a LOWER bound on the quiet-machine cold rate, "
            "which is the safe direction for this particular claim."),
    ),
]


# --------------------------------------------------------------------------- unextracted
#
# Receipts that yielded no law here, each with a reason. A short honest base beats a
# long invented one, and a receipt in this list is NOT a receipt without value -- it
# is one whose value this lane could not type without inventing scope.

UNEXTRACTED_REASONS = {
    "CAPABILITY_NOT_LAW": (
        "records that a construct now EXISTS and executes -- a capability claim. There is "
        "no measured relation to type, and the receipt itself claims no performance."),
    "SUPERSEDED_IN_FLIGHT": (
        "carries an AMEND* key or is named by a later receipt's boundary_this_closes, and "
        "the amendment changes what the number means. Typing the pre-amendment claim would "
        "record a law the corpus has already moved past."),
    "NO_VERDICT_FIELD": (
        "has no `pass` field and no `result` block, so there is no machine-checkable "
        "verdict to anchor an entry on. Absent is recorded as absent, not read as passing."),
    "MOCK_ONLY": (
        "every transport number in it is a knob for hardware that does not exist on this "
        "machine, and the receipt forbids citing one as physical evidence."),
    "SUBSUMED": (
        "its finding is already carried by an entry above, extracted from the receipt that "
        "measured it most directly. Recording it twice would double-count one measurement."),
    "PROSE_ONLY": (
        "the finding is real but stated only as prose about a process or an instrument; "
        "there is no applicability domain to name without inventing one."),
}

UNEXTRACTED: dict[str, str] = {
    # capability, not a measured relation
    "ACCELERATOR_AIR_COMPLETENESS.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_AIR_MATMUL.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_BARRIER_SCOPES.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_CONVOLUTION.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_FRONT_A_P3.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_FRONT_C_P4.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_GEMM.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_NORMALIZATION.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_SHAPE_FUZZ.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_GRAPH_COMPOSITION.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_HMF_CANONICALIZATION.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_C2M_PTX.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_C2M_T1_RUNTIME.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_C2M_T2_IDIOM.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_C2M_SGEMM_IDIOM.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_C2M_CORPUS_DENOMINATOR.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_TRANSFER_VERIFIED.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_SIMDGROUP_GEMM.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_REDUCTION.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_SOFTMAX.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_ATTENTION.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_ATOMICS.json": "CAPABILITY_NOT_LAW",
    "ACCELERATOR_TOPK_SAMPLING.json": "SUBSUMED",
    # amended or closed in flight
    "ACCELERATOR_C2M_CORPUS_CENSUS.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_CONTROL_SPECTRUM.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_BARRIER_CONTROL_MECHANISM.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_GATE_DISAGREEMENT.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_GATE_HEADROOM.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_ORGAN_DISCRIMINATION.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_FRONT_F_ODYSSEY_PASS.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_RUNTIME_EXECUTES.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_RUNTIME_GATE.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_GPU_PACK.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_TIER1_MEASURED.json": "SUPERSEDED_IN_FLIGHT",
    "ACCELERATOR_WORKUNIT_THROUGHPUT.json": "SUPERSEDED_IN_FLIGHT",
    # no machine-checkable verdict
    "ACCELERATOR_BARRIER_WINDOW.json": "NO_VERDICT_FIELD",
    "ACCELERATOR_CAUSALITY.json": "NO_VERDICT_FIELD",
    "ACCELERATOR_VL_GAP_CLOSED.json": "NO_VERDICT_FIELD",
    "ACCELERATOR_SYNC_NOT_SUBMISSION.json": "NO_VERDICT_FIELD",
    "ACCELERATOR_KNOWLEDGE_BASE.json": "NO_VERDICT_FIELD",
    # mock transport only
    "ACCELERATOR_FRONT_H_P8.json": "MOCK_ONLY",
    "ACCELERATOR_HUMF_FAILURE_INJECTION.json": "MOCK_ONLY",
    "ACCELERATOR_HUMF_PARTIAL_LOSS.json": "MOCK_ONLY",
    "ACCELERATOR_HUMF_QUARANTINE_TRUST.json": "MOCK_ONLY",
    "ACCELERATOR_HUMF_RESIDENT_DIGEST.json": "MOCK_ONLY",
    "ACCELERATOR_HUMF_TORN_TIMEOUT_LOST.json": "MOCK_ONLY",
    "ACCELERATOR_HUMF_TRUST_RESOLUTION.json": "MOCK_ONLY",
    # already carried, or prose about an instrument
    "ACCELERATOR_ATTENTION_OCCUPANCY.json": "SUBSUMED",
    "ACCELERATOR_FORGE_REVERDICT.json": "SUBSUMED",
    "ACCELERATOR_FRONT_E_FORGE.json": "SUBSUMED",
    "ACCELERATOR_HONEST_GATE.json": "SUBSUMED",
    "ACCELERATOR_EXPERT_BATCH.json": "SUBSUMED",
    "ACCELERATOR_FRONT_D_P7.json": "SUBSUMED",
    "ACCELERATOR_BANDIT_SEARCH.json": "PROSE_ONLY",
    "ACCELERATOR_CLIFF_TRANSFER.json": "PROSE_ONLY",
    "ACCELERATOR_PERF_MODEL.json": "PROSE_ONLY",
    "ACCELERATOR_SUSTAINED_ADP.json": "PROSE_ONLY",
}


# --------------------------------------------------------------------------- build

def build(*, root: Path = REPO) -> dict[str, Any]:
    """Build the AKB from the real receipt corpus and validate every entry.

    Refuses rather than returns if any entry claims more than its evidence.
    """
    paths = corpus(root)
    superseded = superseding_corpus(paths)
    entries = [validate(dict(law), superseded=superseded, root=root) for law in LAWS]

    cited = {Path(r.partition("#")[0]).name for e in entries for r in e["source_receipts"]}
    names = {p.name for p in paths}
    unextracted = []
    for name in sorted(names - cited):
        key = UNEXTRACTED.get(name)
        unextracted.append({
            "receipt": f"receipts/headless/{name}",
            "reason_code": key or "UNCLASSIFIED",
            "reason": UNEXTRACTED_REASONS.get(key, "not yet triaged by this lane"),
        })

    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1

    return {
        "schema": SCHEMA,
        "axes": list(AXES),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "statuses": list(STATUSES),
        "corpus_size": len(paths),
        "receipts_yielding_laws": len(cited),
        "entries": entries,
        "entries_by_status": by_status,
        "negative_results": sum(1 for e in entries if e["negative_result"]),
        "supersession_in_corpus": {k: v for k, v in sorted(superseded.items())},
        "unextracted": unextracted,
        "unextracted_count": len(unextracted),
    }


def active(akb: dict[str, Any]) -> list[dict[str, Any]]:
    """The laws that may be served as current. A superseded law is never one."""
    return [e for e in akb["entries"] if e["status"] == "ACTIVE"]


def receipt_write(rec: dict[str, Any], path: Path) -> Path:
    """Write through the house receipt module rather than json.dump here."""
    import receipt as receipt_mod
    return receipt_mod.write(rec, path)


def _observe_refusal(label: str, disables: str, entry: dict[str, Any], **kw: Any) -> dict[str, str]:
    """Run one negative control and record the refusal TEXT that was actually raised.

    Refuses to record a control that did not fire. A receipt that claims a refusal
    nobody observed is exactly the thing this program keeps getting burned by, so the
    only way this block gets written is if the validator really did raise.
    """
    try:
        validate(entry, **kw)
    except Refused as exc:
        return {"control": label, "what_it_claims": disables, "observed_refusal": str(exc)}
    raise AssertionError(
        f"NEGATIVE CONTROL {label} DID NOT FIRE. The validator accepted an entry it must "
        f"refuse; the rule is vacuous and the receipt must not be written.")


def negative_controls() -> list[dict[str, str]]:
    """The five refusals the contract requires, observed rather than asserted."""
    import tempfile

    def base(law_id: str) -> dict[str, Any]:
        for entry in LAWS:
            if entry["law_id"] == law_id:
                return json.loads(json.dumps(entry))
        raise AssertionError(law_id)

    out = []

    e = base("AKB-SCAN-VS-CUMSUM")
    e["applicability"]["SHAPE"] = UNSCOPED
    e["unscoped_basis"] = {
        "SHAPE": "receipts/headless/ACCELERATOR_SCAN.json#result.performance.16777216.gbps_2n.mlx_cumsum"}
    out.append(_observe_refusal(
        "UNSCOPED on SHAPE whose source receipt measured one shape",
        "that the scan result holds across all shapes, citing a single-valued field as breadth", e))

    e = base("AKB-SCAN-VS-CUMSUM")
    e["statement"] = "The AIR scan is faster than mx.cumsum."
    out.append(_observe_refusal(
        "a bare 'X is faster' statement with no domain",
        "a present-tense universal comparative the steer forbids", e))

    e = base("AKB-MACHINE-BANDWIDTH")
    e["source_receipts"] = ["receipts/headless/ACCELERATOR_NO_SUCH_RECEIPT.json"]
    out.append(_observe_refusal(
        "a citation to a receipt path that does not exist",
        "evidence from a receipt that is not on disk", e))

    e = base("AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER")
    e["status"] = "ACTIVE"
    out.append(_observe_refusal(
        "ACTIVE on a source receipt carrying AMENDED_IN_PLACE_*",
        "that an amended receipt's law is current and servable", e))

    e = base("AKB-SCAN-VS-CUMSUM")
    e["source_receipts"] = ["receipts/headless/ACCELERATOR_RUNTIME_GATE.json"]
    e["citations"] = []
    out.append(_observe_refusal(
        "ACTIVE on a receipt named by a later receipt's boundary_this_closes",
        "that a law closed by a later block is still current", e))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "receipts/headless").mkdir(parents=True)
        (root / "receipts/headless/ACCELERATOR_FAILED.json").write_text(json.dumps({
            "schema": "hawking.accelerator.receipt.v1", "pass": False,
            "identities": {k: {"status": "ABSENT", "reason": "synthetic fixture"} for k in
                           ("experiment", "machine", "device", "model", "representation",
                            "kernel", "runtime", "transport")}}))
        e = base("AKB-MACHINE-BANDWIDTH")
        e["applicability"]["MACHINE"] = NONE
        e["citations"] = []
        e["source_receipts"] = ["receipts/headless/ACCELERATOR_FAILED.json"]
        out.append(_observe_refusal(
            "evidence_class Measured on a source receipt with pass: false",
            "a measurement from a run that failed",
            e, superseded={}, root=root))
    return out


def construction_receipt(akb: dict[str, Any]) -> dict[str, Any]:
    """This lane's headline receipt, built through the house receipt module."""
    import receipt as receipt_mod

    return receipt_mod.build(
        experiment_class="ACCEL-STATE",
        knowledge_level="INSTANCE",
        identities={
            "experiment": {"id": "G060-AKB-DEVICE-GENOME",
                           "obligation": "type the Accelerator's measured laws with their "
                                         "exact applicability domain, and refuse an entry "
                                         "broader than its evidence"},
            "machine": {"soc": "Apple M3 Ultra", "gpu_cores": 60},
            "device": receipt_mod.absent("nothing was dispatched; this lane reads receipts"),
            "model": receipt_mod.absent("no model executed; the corpus is the subject"),
            "representation": receipt_mod.absent("no representation exercised"),
            "kernel": receipt_mod.absent("no kernel built, run or timed"),
            "runtime": {"python": "3.12.13 (/Library/Frameworks/Python.framework/Versions/3.12)",
                        "note": "the default python3 on this box is 3.14.6 and has no mlx; "
                                "nothing in this lane needs mlx, and the suite passes under both"},
            "transport": receipt_mod.absent("single machine; no transport crossed"),
        },
        result={
            "corpus_size": akb["corpus_size"],
            "receipts_yielding_laws": akb["receipts_yielding_laws"],
            "laws": len(akb["entries"]),
            "entries_by_status": akb["entries_by_status"],
            "negative_results": akb["negative_results"],
            "unextracted_count": akb["unextracted_count"],
            "unextracted_by_reason": {
                code: sum(1 for u in akb["unextracted"] if u["reason_code"] == code)
                for code in sorted({u["reason_code"] for u in akb["unextracted"]})},
            "supersession_modelled": len(akb["supersession_in_corpus"]),
            "negative_control": negative_controls(),
            "a_defect_the_suite_caught_in_this_lane": (
                "the first build ingested its OWN output: ACCELERATOR_LAW_BASE.json matches "
                "the ACCELERATOR_*.json corpus glob, so the base counted itself as an "
                "unclassified 78th receipt and would have grown by one on every build. "
                "test_every_unextracted_receipt_carries_a_reason caught it. corpus() now "
                "names and excludes this lane's own outputs."),
            "what_the_path_collision_forced": (
                "the contract names receipts/headless/ACCELERATOR_KNOWLEDGE_BASE.json as a "
                "NEW file for this AKB. It is not new: it is a live git-tracked artifact "
                "written by tools/accelerator/odyssey_pass.py, which this lane is forbidden "
                "to touch, holding organ SHAPES under schema "
                "hawking.accelerator.knowledge_base.v1. odyssey_pass rewrites it whole from "
                "its own _fresh(), so anything written there would be destroyed on the next "
                "pass AND would break the byte-identical rule. The law base is therefore "
                "ACCELERATOR_LAW_BASE.json and that receipt stays an ordinary corpus input."),
        },
        claim_boundary=(
            "WHAT THIS DOES NOT COVER. (1) COVERAGE: 21 of 77 receipts yielded a typed law; "
            "56 are in `unextracted` with a reason code, and 22 of those are CAPABILITY_NOT_LAW "
            "-- real results that record a construct now executes, which is not a relation "
            "with an applicability domain. A later lane could type several of them. "
            "(2) THE LAWS ARE HAND-EXTRACTED. Every number is quoted from a cited field of a "
            "receipt that was read, and the validator refuses a citation that does not "
            "resolve -- but nothing here proves the STATEMENT is a faithful reading of the "
            "field it cites. That gap is real and no test closes it. "
            "(3) THE VALIDATOR CHECKS FORM, NOT TRUTH. It refuses breadth without evidence, "
            "bare comparatives, unresolvable citations, ACTIVE-on-superseded and "
            "Measured-on-failed. It cannot tell whether an axis value is the RIGHT value; an "
            "entry naming the wrong machine passes. "
            "(4) SUPERSESSION IS SYNTACTIC: an AMEND* key or a receipt name inside "
            "boundary_this_closes. A receipt superseded in prose without either marker is "
            "invisible to it, and a receipt whose amendment STRENGTHENS its claim is treated "
            "the same as one whose amendment guts it. "
            "(5) NOTHING WAS MEASURED. This lane ran no GPU work; every number is transcribed "
            "from a prior receipt and inherits that receipt's own boundary, including the "
            "several that are DIRECTIONAL rather than admissible speedups."),
        passed=True,
    )


def main() -> None:
    akb = build()
    out = RH / "ACCELERATOR_LAW_BASE.json"
    out.write_text(json.dumps(akb, indent=1))
    print(f"corpus {akb['corpus_size']} receipts -> {len(akb['entries'])} laws "
          f"from {akb['receipts_yielding_laws']} receipts, "
          f"{akb['unextracted_count']} unextracted")
    print(f"by status {akb['entries_by_status']}, "
          f"negative results {akb['negative_results']}, "
          f"active servable {len(active(akb))}")
    print(f"wrote {out.relative_to(REPO)}")

    rec = construction_receipt(akb)
    rpath = RH / "ACCELERATOR_AKB_CONSTRUCTION.json"
    receipt_write(rec, rpath)
    fired = len(rec["result"]["negative_control"])
    print(f"negative controls OBSERVED firing: {fired}")
    print(f"wrote {rpath.relative_to(REPO)}")


if __name__ == "__main__":
    main()
