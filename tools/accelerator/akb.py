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

# S032 §13. Membership was decided by a FILENAME PREFIX, which is the same
# name-filter defect this program fixed in bench.machine_quiescence: a receipt
# named otherwise is INVISIBLE, and invisible reads identically to "triaged and
# found empty". The steer forbids both available shortcuts -- do not widen to every
# receipt in the directory, and do not let a classifier GUESS which civilization
# owns the 348 foreign ones. So a receipt DECLARES itself instead.
EVIDENCE_DOMAIN = "accelerator"

# The six scopes S032 §13 names. A declaring receipt must carry all of them, so a
# half-filled declaration cannot buy membership -- the same rule the AKB already
# applies to an entry's eleven applicability axes.
REGISTRATION_KEYS = ("evidence_domain", "civilization", "program",
                     "machine_scope", "representation_scope", "kernel_scope")


def registration(path: Path) -> dict | None:
    """The receipt's own membership declaration, or None if it does not declare.

    A declaration that is INCOMPLETE is not a declaration: it returns None rather
    than a partial dict, because a receipt that names its domain and nothing else
    would otherwise join the corpus while telling the reader nothing about what its
    evidence covers.
    """
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return None
    reg = doc.get("akb_registration")
    if not isinstance(reg, dict):
        return None
    if any(k not in reg for k in REGISTRATION_KEYS):
        return None
    if reg.get("evidence_domain") != EVIDENCE_DOMAIN:
        return None
    return reg


def corpus(root: Path = REPO) -> list[Path]:
    """Every Accelerator receipt on disk, sorted. Two routes, and the AKB reports
    which one each receipt arrived by.

    DECLARED -- the receipt carries a complete `akb_registration` naming this
    evidence_domain. This is the route S032 §13 asks for and the only one that
    does not depend on what a file was called.

    LEGACY GLOB -- the filename starts with ACCELERATOR_. Retained because 80-odd
    receipts predate the declaration and rewriting them all in one pass would be a
    bulk edit nobody reviewed; the build reports how many arrive this way so the
    legacy route's size is visible and can shrink.

    Excludes this module's OWN outputs either way: they match the glob, and without
    this the base would ingest itself -- each build adding a receipt until the AKB
    cited itself as evidence for its own laws.
    """
    d = root / "receipts/headless"
    by_glob = {p for p in d.glob("ACCELERATOR_*.json")}
    by_decl = {p for p in d.glob("*.json") if registration(p) is not None}
    return sorted((by_glob | by_decl) - {d / n for n in OWN_OUTPUTS},
                  key=lambda q: q.name)


def membership_routes(root: Path = REPO) -> dict:
    """How each corpus member got in. A route nobody can see is a route nobody
    can shrink."""
    d = root / "receipts/headless"
    out = {"declared": [], "legacy_glob_only": []}
    for p in corpus(root):
        (out["declared"] if registration(p) is not None
         else out["legacy_glob_only"]).append(p.name)
    out["declared_count"] = len(out["declared"])
    out["legacy_glob_only_count"] = len(out["legacy_glob_only"])
    out["note"] = (
        "membership by DECLARATION does not depend on the filename; membership by "
        "the legacy glob does. The legacy count is the size of the remaining "
        "filename dependence and should fall, never rise.")
    return out


def outside_scope(root: Path = REPO) -> list[str]:
    """Receipts in the same directory that the corpus glob CANNOT SEE.

    The corpus scopes itself by FILENAME PREFIX, which is a name filter with the
    same defect this program just fixed in bench: a receipt named TOKEN_*, or
    CAPABILITY_*, or FUSION_* is neither extracted NOR refused by
    test_every_unextracted_receipt_carries_a_reason -- IT IS INVISIBLE, and
    invisible reads identically to `triaged and found empty`.

    The scope itself is defensible (this is the ACCELERATOR knowledge base) and it
    is NOT widened here by guesswork, because pulling in every headless receipt
    would ingest Q80, noetic and civilization work this lane cannot type. What
    changes is that the exclusion is now REPORTED rather than silent, so a reader
    can see that membership depends on what a file was named.
    """
    seen = {p.name for p in corpus(root)} | set(OWN_OUTPUTS)
    return sorted(p.name for p in (root / "receipts/headless").glob("*.json")
                  if p.name not in seen)


# Accelerator receipts that do NOT start with ACCELERATOR_ and are therefore
# invisible to the corpus glob. Listed EXPLICITLY rather than inferred, because a
# content classifier would be this lane guessing which of 348 headless receipts
# are its own -- and a wrong guess ingests another campaign's work as Accelerator
# evidence. Named here, the gap is a short auditable list instead of a silence.
KNOWN_ACCELERATOR_OUTSIDE_SCOPE = (
    "TOKEN_EXECUTION_ATLAS.json",
    "TOKEN_EXECUTION_ATLAS_COUNTS.json",
    "TOKEN_GRAPH_REDUCTION_TIMED.json",
    "CAPABILITY_FUSED_GRAPH_CLEARED.json",
    "FUSION_GAIN_IS_LENGTH_INDEPENDENT.json",
)


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


# receipt name -> law ids an amendment in it explicitly declares unaffected.
# Populated by superseding_corpus, which is the only thing that reads amendments.
_RECONCILED: dict[str, dict[str, set[str]]] = {}
_EXEMPT: dict[str, set[str]] = {}


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
            # An amendment may DECLARE which laws it leaves standing. Explicit
            # registration, not a semantic guess about scope (S032 §13) -- an
            # amendment that withdraws one figure from a receipt does not
            # automatically invalidate a law resting on a different measurement in
            # the same file, but nothing may INFER that; the amendment has to say
            # so, by law_id, in the receipt.
            for k in amended:
                blk = d.get(k) if isinstance(d.get(k), dict) else {}
                for lid in (blk or {}).get("laws_unaffected", []):
                    _EXEMPT.setdefault(f.name, set()).add(str(lid))
                # A THIRD STATE, because two were not enough. `laws_unaffected`
                # says the amendment does not bear on that law. But an amendment
                # that DOES bear on a law and whose correction has ALREADY BEEN
                # WRITTEN INTO that law's statement is neither unaffected nor
                # unreconciled, and calling it unaffected would be false.
                # `laws_reconciled` names those -- and it is NOT a promise: the
                # law's own statement must CITE the amending receipt, checked in
                # NC4, so a reconciliation nobody performed still refuses.
                # The acceptable citations are the receipts the AMENDMENT ITSELF
                # names -- usually the one that did the correcting -- plus this
                # file. Deriving them from the block rather than from the file
                # name is what lets an in-file amendment point at the receipt
                # that actually carries the new measurement.
                stems = {Path(m).stem for m in RECEIPT_NAME.findall(json.dumps(blk))}
                stems.add(Path(f.name).stem)
                for lid in (blk or {}).get("laws_reconciled", []):
                    _RECONCILED.setdefault(f.name, {})[str(lid)] = stems
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

    # A value that READS as a sentinel but is not one. "NONE -- holds for any
    # kernel" looks like NONE to a reviewer and is a named value to every check
    # below, so it slips the grounding rule while claiming its breadth. Same for
    # a prose UNSCOPED. Caught after writing one by accident in this lane's own
    # bandwidth-ceiling law.
    for axis, value in entry["applicability"].items():
        if not isinstance(value, str):
            continue
        head = value.strip().upper()
        for sentinel in (NONE, UNSCOPED, UNKNOWN):
            if value != sentinel and head.startswith(sentinel):
                raise Refused(
                    f"{lid}: axis {axis} is {value[:60]!r}, which READS as {sentinel} and is "
                    f"treated as a named value by every check below. Use the bare sentinel and "
                    f"ground it, or name the value the evidence actually covers.")

    # NONE must be grounded in the receipt recording that identity ABSENT.
    for axis, value in entry["applicability"].items():
        if value != NONE or axis not in IDENTITY_OF_AXIS:
            continue
        ident = IDENTITY_OF_AXIS[axis]
        for rel in entry["source_receipts"]:
            ids = resolve(rel.partition("#")[0], root=root).get("identities")
            if ids is None:
                # The receipt predates the identity schema, so the NONE cannot be
                # checked against anything. Skipping SILENTLY is the check that
                # cannot fail: an ungrounded NONE on MACHINE would read exactly like
                # a grounded one, and NONE on MACHINE is the widest over-claim
                # available -- it turns an M3 Ultra result into a universal. Record
                # it so "checked and grounded" is distinguishable from "could not
                # check". Not raised, because refusing here would retroactively
                # invalidate every law citing a pre-schema receipt.
                entry.setdefault("none_claims_not_grounded", []).append(
                    {"axis": axis, "identity": ident, "receipt": rel,
                     "why": "source receipt has no identities block to check against"})
                continue
            got = ids.get(ident)
            if not (isinstance(got, dict) and got.get("status") in ("ABSENT", "MOCK", "SIMULATED")):
                raise Refused(
                    f"{lid}: axis {axis} claims NONE but {rel} records identity "
                    f"{ident!r} as {json.dumps(got)[:120]}, not ABSENT.")

    # NC4 -- a superseded source may not be served as ACTIVE.
    if entry["status"] == "ACTIVE":
        for rel in entry["source_receipts"]:
            name = Path(rel.partition("#")[0]).name
            if name in superseded and lid not in _EXEMPT.get(name, ()):
                if lid in _RECONCILED.get(name, {}):
                    # Claimed reconciled -- so the law must SHOW it, by citing the
                    # receipt that amended it. A declaration alone would let a law
                    # opt out of the check it exists to fail.
                    amenders = _RECONCILED[name][lid]
                    if not any(a in entry["statement"] for a in amenders):
                        raise Refused(
                            f"{lid}: the amendment to {name} claims this law is RECONCILED, "
                            f"but the law's own statement cites none of {sorted(amenders)}. "
                            "A reconciliation nobody wrote into the law is a promise, not a "
                            "correction.")
                    continue
                raise Refused(
                    f"{lid}: status ACTIVE but source {name} is superseded by "
                    f"{superseded[name]}. A law resting on an amended or closed receipt is "
                    f"CONDITIONAL at best. If the amendment does not bear on THIS law, "
                    f"the amendment must say so by naming {lid} in laws_unaffected; if it "
                    f"DOES bear on it and the law already carries the correction, name it "
                    f"in laws_reconciled and cite the amending receipt in the statement.")

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
HK = "hawking-core release-fast Rust/Metal decoder, binary d34044cffae8f320"

LAWS: list[dict[str, Any]] = [
    dict(
        law_id="AKB-GRAPH-BOUNDARY-LEVER-IS-NEARLY-EXHAUSTED",
        statement=(
            "The 628-dispatch decode graph holds 401 dispatches at boundaries whose intermediate "
            "is consumed once and never escapes -- 63.9% of the graph, and deleting all of them "
            "would leave 227. Together they are worth about 1.10 ms of a 28.0208 ms token (3.9%) "
            "in dispatch cost at the 2.75 us marginal measured for non-operand-sharing levers, "
            "plus 31.68 MB of intermediate traffic = 0.32% of the token's weight bytes. Against "
            "the 1.65x of headroom the byte wall leaves, EVERY REMAINING GRAPH BOUNDARY TOGETHER "
            "IS ABOUT 4% OF A 65% GAP. Two of twelve boundaries are undeletable and marked so: "
            "GQA's qkv and rope/cache outputs ESCAPE into the KV cache. The clearest abstraction "
            "tax by ratio is ba_to_decay_beta -> gated_delta_decode, which spends a whole "
            "dispatch moving 384 BYTES, 48 times per token."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A)",
            "ARCHITECTURE": ("Qwen3.8 hybrid; fused, the two layer kinds DIVERGE -- 48 DeltaNet "
                             "at 10 dispatches, 16 GQA at 9, where unfused both were 15"),
            "ORGAN": "whole decode graph, resolved to producer/consumer pairs",
            "REPRESENTATION": "HQ30UQ4 g64 mixer + affine_q2 g64 MLP",
            "SHAPE": "batch 1 decode; intermediate sizes assume f32 activations",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": "the 628-dispatch fused graph, histogram from a real 6-token trace",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode"},
        evidence_class="Derived",
        source_receipts=["receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json"],
        citations=[
            "receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json#THE_BOUNDARY_CENSUS",
            "receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json"
            "#AND_THAT_IS_THE_CEILING_ON_EVERY_REMAINING_BOUNDARY_TOGETHER",
            "receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "Derived, and the class says so. The dispatch COUNTS are exact -- the histogram "
            "reconciles to 628 and the layer arithmetic closes. The 3.9% is 401 multiplied by a "
            "rate measured on three OTHER levers, and the one lever that broke that rate broke "
            "it by 6.6x, so a boundary of an unanticipated shape would make this LOW rather than "
            "high. The boundary list is STRUCTURAL: producer/consumer pairs read off the "
            "histogram and the layer shape, with no buffer traced to prove single consumption -- "
            "the two escapes are marked because their escape is known, not because a tool found "
            "them. Nothing here was built or timed."),
    ),
    dict(
        law_id="AKB-DELTANET-STATE-IS-3-PERCENT-OF-THE-TOKEN",
        statement=(
            "The DeltaNet recurrent state is 48 value heads x 128 value dim x 128 key dim of f32 "
            "= 3.146 MB PER LAYER, read and written every token across 48 layers = 301.99 MB per "
            "decode token, 3.06% of the 9.868 GB of weights -- and it appears in no weight "
            "ranking because it is not a weight. With the conv cache (22.02 MB) and the GQA KV "
            "read at context 60 (7.86 MB), non-weight traffic is 331.87 MB per token, 3.36%. So "
            "the token reads about 10.20 GB rather than 9.87 and the byte-wall fraction moves "
            "from 60.7% to about 62.7%. THE KV TERM IS THE ONLY ONE THAT GROWS WITH CONTEXT: at "
            "8192 tokens it would be 1.07 GB per token, 10.9% of the weight bytes, which is the "
            "first term in this accounting that makes the ceiling context-dependent."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A)",
            "ARCHITECTURE": ("Qwen3.8 hybrid: 48 DeltaNet layers with 48 value heads at 128x128 "
                             "state, 16 GQA layers with 4 kv heads at head_dim 256"),
            "ORGAN": "DeltaNet recurrent state, DeltaNet conv cache, GQA KV cache",
            "REPRESENTATION": "f32 state; the state is NOT quantized and this is why it is "
                              "invisible to an EBPW accounting",
            "SHAPE": "batch 1 decode; the KV figure is at context 60 and is linear in context",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": "the 628-dispatch fused graph",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode"},
        evidence_class="Derived",
        source_receipts=["receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json"],
        citations=[
            "receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json"
            "#WHAT_THE_WEIGHT_ATLAS_EXCLUDED_IS_NOW_A_NUMBER",
            "receipts/headless/ACCELERATOR_628_BOUNDARY_CENSUS.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "Derived from the artifact's own config -- linear_num_value_heads, "
            "linear_value_head_dim and linear_key_head_dim -- times f32 and the layer count. It "
            "upgrades a NAMED EXCLUSION in ACCELERATOR_TOKEN_BYTE_ATLAS_628 to a number, which "
            "is what that receipt asked for. It assumes the state is read AND written once per "
            "layer per token, which is what a recurrence does and was not traced. The activation "
            "floor is excluded as too crude to state."),
    ),
    dict(
        law_id="AKB-INPUT-OPERAND-REUSE-IS-NOT-THE-MLP-FUSION-WIN",
        statement=(
            "Loading the input vector ONCE for two accumulators instead of twice did NOT make a "
            "dual-accumulator affine_q2 matvec faster at the sealed resident's real MLP shape. "
            "Two arms, one dispatch each, identical weights read and identical output to "
            "1.913e-07 against a float64 oracle, differing only in whether the input slice is "
            "loaded once or twice through two unaliased parameters: the REUSE arm measured 0.3715 "
            "ms against RELOAD's 0.3646, 1.9% SLOWER, with reload faster in 3 of 3 admitted "
            "sweeps. An 8-20% advantage for reuse -- the size at which it was proposed as the "
            "mechanism behind 84% of the gate_up fusion win -- is excluded. MECHANISM: the kernel "
            "issues 8 shift-and-mask unpacks plus 2 fused multiply-adds per byte pair, and "
            "against that instruction stream one extra L1 load is free. The regime cross-checks "
            "-- 149.9 GB/s here against the 161.9 GB/s ACCELERATOR_EXPERT_BATCH measured for a "
            "native q4 arm it independently called arithmetic-bound on the unpack."),
        applicability={
            "MODEL": NONE,
            "ARCHITECTURE": NONE,
            "ORGAN": "MLP gate/up projection pair",
            "REPRESENTATION": "affine_q2 group64 at 2.5 bpw, the artifact's own MLP codec",
            "SHAPE": "17408 x 5120, the artifact's real MLP shape; one shape only",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": ("synthetic dual-accumulator matvec, threadgroup 128, 2 rows per "
                       "threadgroup -- NOT the production kernel"),
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-token decode shape (one input vector)"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_MLP_OPERAND_REUSE_REFUTED.json"],
        citations=[
            "receipts/headless/ACCELERATOR_MLP_OPERAND_REUSE_REFUTED.json#measured",
            "receipts/headless/ACCELERATOR_MLP_OPERAND_REUSE_REFUTED.json"
            "#WHAT_THIS_ELIMINATES_FROM_S032_5_S_THIRTEEN_CANDIDATES",
            "receipts/headless/ACCELERATOR_MLP_OPERAND_REUSE_REFUTED.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "3 sweeps admitted and 0 refused under a pre-registered quiescence gate (max_rss "
            "0.13-0.15 GiB). The arms carry ~15% IQR, ABOVE the program's 10% gate, so this "
            "refutes the hypothesis AT THE SIZE IT WAS PROPOSED and resolves no small effect in "
            "either direction -- it rests on sign consistency 3/3 plus a median in the wrong "
            "direction, not on a clean gate. The control's defence against common-subexpression "
            "elimination is ARCHITECTURAL (two unaliased buffer parameters) and unverified "
            "against generated assembly, because xcrun metal is ABSENT on this machine. Says "
            "NOTHING about the production 17.35 us/dispatch measurement -- it refutes one "
            "proposed mechanism for it. The arm that would settle what IS left -- prologue, "
            "reduction and epilogue together -- is BUILT, CORRECT and UNTIMED, blocked on a "
            "quiesced window."),
    ),
    dict(
        law_id="AKB-964-DISPATCHES-PER-DECODE-TOKEN",
        statement=(
            "The sealed-3.14 resident's production decode graph runs 964 dispatches per token, "
            "MEASURED by delta so prefill cancels: (15424 - 11568) / (6 - 2) = 964.000 exactly. "
            "The arithmetic 1 + 64*15 + 3 is structurally confirmed and BOTH LAYER KINDS COST "
            "EXACTLY 15 BY DIFFERENT ROUTES -- a DeltaNet layer spends 2 rmsnorm + 2 "
            "add_residual + 6 matvec + 1 swiglu + 4 DeltaNet-specific, a GQA layer 2 + 2 + 7 "
            "matvec + 1 + 3 GQA-specific -- which the schedule file could not have shown. The "
            "matvecs split CLEANLY BY ORGAN with nothing mixed inside one: 209 uniform-q4 take "
            "every mixer projection plus the lm_head, 192 affine-q2 take every MLP projection."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A), 3.1393 complete EBPW",
            "ARCHITECTURE": "Qwen3.8 hybrid, 48 DeltaNet + 16 GQA layers",
            "ORGAN": "whole decode graph, resolved per kernel family",
            "REPRESENTATION": "mixed HQ30UQ4 g64 mixer + affine_q2 g64 MLP",
            "SHAPE": "batch 1 decode, 11-token prompt; dispatch count is structural and was "
                     "not varied over prompts",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": "the 964-dispatch production graph, all four fusion levers off",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/TOKEN_EXECUTION_ATLAS_COUNTS.json"],
        citations=["receipts/headless/TOKEN_EXECUTION_ATLAS_COUNTS.json#headline",
                   "receipts/headless/TOKEN_EXECUTION_ATLAS_COUNTS.json"
                   "#the_arithmetic_is_now_STRUCTURALLY_confirmed",
                   "receipts/headless/TOKEN_EXECUTION_ATLAS_COUNTS.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "The delta method removes prefill exactly and lands on a whole number, 964.000. It "
            "became measurable only after a real defect was fixed: the runtime pushed one label "
            "per dispatch into a Vec and the session harvest collapsed it into a HashSet, "
            "destroying the multiplicity the question is about. COUNTS ONLY -- nothing here is "
            "timed, and AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST establishes that this count "
            "does not predict the graph's cost. This receipt's OWN earlier histogram was "
            "withdrawn for describing a uniform-q4 body; what is recorded here is the "
            "re-measured version."),
    ),
    dict(
        law_id="AKB-FUSION-GAIN-IS-LENGTH-INDEPENDENT",
        statement=(
            "The four pre-existing fusion levers were worth 2.1-2.3% of wall at EVERY generation "
            "length measured -- 64, 256 and 1024 new tokens -- for 21.6% fewer dispatches, with "
            "output identical: clean floors 30.2791 ms/token unfused against 29.6304 fused, "
            "unfused reproducing 30.2282-30.3325 across 5 clean runs in two sessions (0.35% "
            "spread) and fused 29.5537-29.6653 (0.38%), with complete separation. THIS TIGHTENS "
            "THE CEILING RATHER THAN LOOSENING IT: a length-independent gain means the dispatch "
            "ladder does not improve at longer generations either, which the +29.09% it replaced "
            "would have implied."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A)",
            "ARCHITECTURE": "Qwen3.8 hybrid, 48 DeltaNet + 16 GQA layers",
            "ORGAN": "whole decode graph", "REPRESENTATION": "sealed-3.14, unchanged across arms",
            "SHAPE": "64, 256 and 1024 new tokens at max_seq_len 2048, one prompt",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": "964-dispatch control against the 756-dispatch three-lever graph",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode"},
        evidence_class="Reproduced",
        source_receipts=["receipts/headless/FUSION_GAIN_IS_LENGTH_INDEPENDENT.json"],
        citations=["receipts/headless/FUSION_GAIN_IS_LENGTH_INDEPENDENT.json"
                   "#THE_29_PERCENT_IS_REFUTED",
                   "receipts/headless/FUSION_GAIN_IS_LENGTH_INDEPENDENT.json"
                   "#what_this_costs_and_what_survives",
                   "receipts/headless/FUSION_GAIN_IS_LENGTH_INDEPENDENT.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "Reproduced: both floors recur across two sessions and two processes at 0.35-0.38% "
            "spread. Admitted by COMPLETE SEPARATION of two tight clusters after excluding 12 of "
            "24 runs as outliers -- an enormous exclusion rate, stated as such, and the reason "
            "the receipt refuses a rep count as its basis. This law REFUTES the +29.09% its own "
            "predecessor published, which came from unpaired runs on a contended machine. SHAPE "
            "is not UNSCOPED: length-independence is established over 64 to 1024 and says "
            "nothing beyond 1024."),
    ),
    dict(
        law_id="AKB-DECODE-BYTES-ARE-THE-MLP-NOT-THE-HEAD",
        statement=(
            "One decode token of the sealed-3.14 resident reads 9,868,249,760 weight bytes, and "
            "the ranking by TOKEN bytes is MLP 54.19% over 128 dispatches, DeltaNet projections "
            "29.93% over 96, GQA projections 9.03% over 48, lm_head 6.84% in ONE. The derivation "
            "from element counts and codec bpw reconciles against the artifact's own "
            "payload_bytes to 0.101%, and the residual has a name -- MIX_REPORT records "
            "f32_bytes 10,584,840 against an unaccounted 10,651,416. THE PER-DISPATCH RANKING "
            "DISAGREES WITH THE PER-TOKEN ONE: the head moves 675,430,400 bytes in one dispatch "
            "against 55,705,600 for the largest MLP dispatch, 12.13x, and is still 6.84% of the "
            "token because the MLP runs its dispatch 128 times. Deleting the head entirely -- "
            "which no exact algorithm can do -- caps at 6.84%, and the full logit tensor that "
            "argmax fusion would remove is 0.0201% of the token's bytes."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A), 3.1393 complete EBPW",
            "ARCHITECTURE": ("Qwen3.8 hybrid: 48 DeltaNet + 16 GQA (full_attention_interval 4), "
                             "H 5120, I 17408, V 248320, untied head"),
            "ORGAN": "whole body, resolved by organ",
            "REPRESENTATION": "HQ30UQ4 g64 mixer/head at 4.25 bpw + affine_q2 g64 MLP at 2.5 bpw",
            "SHAPE": "batch 1 decode, one token; KV and activation traffic EXCLUDED",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": "the 628-dispatch fused decode graph",
            "STORAGE_TIER": "weights resident in unified memory; not a storage-read accounting",
            "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode (prefill amortises the head differently)"},
        evidence_class="Derived",
        source_receipts=["receipts/headless/ACCELERATOR_TOKEN_BYTE_ATLAS_628.json"],
        citations=[
            "receipts/headless/ACCELERATOR_TOKEN_BYTE_ATLAS_628.json"
            "#THE_DERIVATION_RECONCILES_AND_THAT_IS_WHAT_MAKES_IT_A_MEASUREMENT",
            "receipts/headless/ACCELERATOR_TOKEN_BYTE_ATLAS_628.json#ORGAN_ROLLUP",
            "receipts/headless/ACCELERATOR_TOKEN_BYTE_ATLAS_628.json#WHAT_THIS_ATLAS_IS_NOT"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "Derived, not Measured, and the class says so: exact element counts times codec bpw. "
            "Its strength is the reconciliation -- 0.101% unaccounted landing on an "
            "independently recorded f32_bytes. Its weakness is that the ms figures divide bytes "
            "by ONE effective bandwidth and families demonstrably do not share one "
            "(ACCELERATOR_EXPERT_BATCH measured 161.9 vs 426.4 GB/s on this machine), so only "
            "the BYTE ranking is claimed. Non-projection dispatches read activations and no "
            "weights and are absent from a weight-byte ranking by construction, not free."),
    ),
    dict(
        law_id="AKB-NORM-BOUNDS-CANNOT-PRUNE-THIS-HEAD",
        statement=(
            "No exact norm-based rejection bound prunes the sealed-3.14 lm_head. Row norms are "
            "near-equinorm -- coefficient of variation 0.1288 on the likely head and 0.1646 on "
            "the companion table -- so the Cauchy-Schwarz test |x.w| <= ||x|| ||w|| rejects "
            "0.000% and 0.073% of the 248320 rows at an alignment cos of 0.10 and 0.236% and "
            "0.976% at 0.30; pruning half would need cos about 0.63. The two-stage prefix bound "
            "dies the same way because energy is uniform across dimensions: median tail-norm "
            "ratio tracks sqrt(1-k/D) at 0.995-1.023 for k from 0.25D to 0.90D on both tables, "
            "and the top 10% of 64-dimension groups holds 13.3-13.7% of the squared norm against "
            "a uniform 10%, so a prefix bound at k=D/2 is 1.41x tighter than one that already "
            "prunes under 1%. MECHANISM: the head has no preferred magnitude and no preferred "
            "dimension subspace, and its discrimination is entirely full-dimensional DIRECTION, "
            "which is what every norm bound discards."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A), untied [248320, 5120] head",
            "ARCHITECTURE": "Qwen3.8 hybrid, tie_word_embeddings false",
            "ORGAN": "lm_head (and the companion embedding table, which refutes less strongly)",
            "REPRESENTATION": "HQ30UQ4 g64 at 4.25 bpw; norms taken on the QUANTIZED weights",
            "SHAPE": "248320 rows x 5120 dims; tail profile on 4096 seeded rows at 64-dim groups",
            "MACHINE": M3, "RUNTIME": HK,
            # The bare sentinel, GROUNDED: the source receipt records KernelIdentity
            # ABSENT with a reason, because nothing was dispatched. Writing this as
            # prose starting with the word NONE is what the sentinel guard caught --
            # it reads as NONE to a reviewer and validates as a named value.
            "KERNEL": NONE,
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "greedy single-token selection at decode"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_LM_HEAD_EXACT_BOUNDS_REFUTED.json"],
        citations=[
            "receipts/headless/ACCELERATOR_LM_HEAD_EXACT_BOUNDS_REFUTED.json"
            "#MEASURED_WITHOUT_DEQUANTIZING_ANYTHING",
            "receipts/headless/ACCELERATOR_LM_HEAD_EXACT_BOUNDS_REFUTED.json"
            "#THE_SECOND_FAMILY_DIES_THE_SAME_WAY_AND_I_PREDICTED_THAT_TOO",
            "receipts/headless/ACCELERATOR_LM_HEAD_EXACT_BOUNDS_REFUTED.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "Both halves were PREDICTED IN WRITING BEFORE THE RUN with a named falsifier and "
            "both confirmed -- CV under 0.30 as predicted, and tail decay tracking the uniform "
            "sqrt(1-k/D) within 2.3% as predicted. Norms are exact over all 248320 rows of both "
            "tables, computed from the packed codes with no dequantization. The weakness is that "
            "NO FORWARD PASS RAN: cos(theta) is a free parameter and the curve is reported "
            "across it, so what is established is that no plausible alignment prunes materially, "
            "not that a measured alignment obtains. MODEL is deliberately not UNSCOPED -- a "
            "heavy-tailed head elsewhere would reopen the family. Says NOTHING about low-rank "
            "factorization: a matrix can be per-row isotropic and still low rank."),
    ),
    dict(
        law_id="AKB-CHAT-TEMPLATE-ARM-MOVES-CAPABILITY",
        statement=(
            "The chat-template arm changed the sealed-3.14 resident's measured capability by "
            "five of forty-three cases with no byte of the artifact altered: 30/43 on "
            "open_think against 35/43 on pre_closed_think, same artifact_inventory_sha "
            "1aff5df85bda1108, same binary, same chat_template file. The whole movement is on "
            "structured_output, 5/15 -> 10/15, and eight empty replies become zero -- under "
            "open_think eight calls generated 1135-1536 tokens and returned nothing after the "
            "think block was stripped. The think arm held THREE TIMES the token budget "
            "(capability_suite.py:277) and still lost, so the budget asymmetry favours the "
            "losing arm. Per-token decode is arm-dependent and small in the same direction: at "
            "one fixed prompt, pre_closed_think reads 28.0208 ms/token against open_think's "
            "28.3018, a 1.003% advantage, because the open render is 65 prompt tokens against "
            "25 and carries a longer KV cache. Capability outweighs decode 16.6x, taking "
            "arm-matched accepted TPS 24.65 -> 29.05."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A), 3.1393 complete EBPW",
            "ARCHITECTURE": "Qwen3.8 hybrid, 48 DeltaNet + 16 GQA layers",
            "ORGAN": NONE,
            "REPRESENTATION": ("UNCHANGED across arms: HQ30UQ4 g64 mixer + affine_q2 g64 MLP. "
                               "The arm is a SERVING MODE, not a representation."),
            "SHAPE": "batch 1 decode; 43-case capability suite; timing at one 21-token prompt",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": "the 628-dispatch fused decode graph, identical in both arms",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode with a rendered chat prompt"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_RESIDENT_TEMPLATE_ARM.json"],
        citations=[
            "receipts/headless/ACCELERATOR_RESIDENT_TEMPLATE_ARM.json#THE_FINDING",
            "receipts/headless/ACCELERATOR_RESIDENT_TEMPLATE_ARM.json"
            "#RAW_TPS_IS_ARM_DEPENDENT_AND_I_MEASURED_IT_RATHER_THAN_ASSUMING_IT",
            "receipts/headless/ACCELERATOR_RESIDENT_TEMPLATE_ARM.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "ONE capability run per arm, unpaired -- admissible because the claim is a PASS "
            "COUNT, which does not drift with machine load, and the same runs' wall times are "
            "explicitly refused (the no_think run's own machine_state records "
            "worst_repetition_spread_pct 63.8). The timing half is 3 admitted sweeps under the "
            "pre-registered quiescence gate with per-arm spreads of 0.18-0.34%, 0 refused. "
            "MODEL is deliberately not UNSCOPED: the same effect was measured on the 2.60-EBPW "
            "body at 0/43 vs 14/43, which is two bodies of one family and not breadth. Nothing "
            "here says the arm is better in general -- a task that needs chain-of-thought is "
            "exactly what it removes, and the suite's one such item fails under both arms."),
    ),
    dict(
        law_id="AKB-BANDWIDTH-CEILING-BOUNDS-ACCEPTED-TPS",
        statement=(
            "Single-request decode of the sealed-3.14 body reads 9,878,898,416 weight bytes per "
            "token -- the whole payload except the embedding table, which is read as one row "
            "while the untied lm_head is read in full. Against this box's measured 589.73 GB/s "
            "that sets a floor of 16.752 ms per token, so the raw-TPS ceiling is 187.40 divided "
            "by complete EBPW and the accepted-TPS ceiling is that times capability over 43. The "
            "measured graph sits at 27.5896 ms, which is 358.1 GB/s or 60.7% of the wall, leaving "
            "1.65x of total headroom for ALL execution-graph work combined. 50 accepted TPS "
            "therefore requires capability/EBPW >= 11.473; the resident is at 30/3.1393 = 9.556, "
            "20.1% short, so at 30/43 the target is unreachable by 1.20x even at infinite graph "
            "efficiency. The 2.5970-EBPW specimen's accepted CEILING at 14/43 is 23.49, below the "
            "resident's present accepted measurement of 25.29."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A), 26,895,998,464 parent params",
            "ARCHITECTURE": "Qwen3.8 hybrid, 64 layers, untied 248320x5120 embed and lm_head",
            "ORGAN": "whole body; the bound is over total resident weight bytes",
            "REPRESENTATION": "complete EBPW 3.1393 (HQ30UQ4 g64 mixer + affine_q2 g64 MLP)",
            "SHAPE": "batch 1, one token, KV traffic EXCLUDED so the bound is optimistic",
            "MACHINE": M3, "RUNTIME": HK,
            # NOT the NONE sentinel. A descriptive string starting with the word
            # NONE would read as NONE to a human and as a named value to
            # validate(), which is the worst of both. The FLOOR is kernel-
            # independent arithmetic; the 358.1 GB/s and 60.7% figures are not,
            # so the axis names the kernels that produced them.
            "KERNEL": ("the 628-dispatch fused graph: qwen80_add_residual_rmsnorm_tg, "
                       "qwen_uniform_q4_group64_matvec_{qkv,pair_concat,geo_tpr64}, "
                       "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128"),
            "STORAGE_TIER": "resident in unified memory; not a storage-read bound",
            "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode (prefill amortises the lm_head differently)"},
        evidence_class="Derived",
        source_receipts=["receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"],
        citations=[
            "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"
            "#FINDING_7_THE_BANDWIDTH_WALL_TURNS_S031_19_INTO_A_CONSTRAINT_CURVE",
            "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"
            "#FINDING_7_THE_BANDWIDTH_WALL_TURNS_S031_19_INTO_A_CONSTRAINT_CURVE.claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "Derived, NOT Measured, and the class says so: it is arithmetic over two independent "
            "measurements -- the artifact's payload_bytes and MACHINE_GENOME's 589.73 GB/s triad "
            "median at 1.89% IQR. Its weakest input is that KV-cache traffic is excluded, which "
            "makes every ceiling here an UPPER BOUND; that is the safe direction for the "
            "unreachability claim and the unsafe direction for the 59.7 raw figure, which must "
            "not be quoted as attainable. KERNEL is NONE because the bound is over bytes and no "
            "kernel can move fewer than the weights it reads; TOPOLOGY is NONE and is grounded in "
            "the source receipt recording transport ABSENT."),
    ),
    dict(
        law_id="AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST",
        statement=(
            "Two decode graphs at the SAME dispatch count differed in wall time. On the "
            "sealed-3.14 mixed-codec Qwen3.8 body, add_rmsnorm+gqa_qkv+dn_inproj fusion and "
            "mlp_swiglu+gqa_qkv+dn_inproj fusion BOTH measure 756 dispatches per decode "
            "token -- byte-identical trace totals of 9072 at n=2 and 12096 at n=6 -- and "
            "read 28.8697 ms and 27.7872 ms per token, 3.75% apart with complete separation "
            "and per-arm spreads of 0.25% and 0.21%. The metric fails in the other direction "
            "too: a 692-dispatch graph measured 27.7757 ms, indistinguishable from the "
            "756-dispatch one at 0.04%. The marginal cost of a removed dispatch was 2.64, "
            "2.86 and 17.35 us on the GPU for three levers measured in one session, a 6.6x "
            "range, so no single us-per-dispatch constant can be multiplied by a count. "
            "DISPATCHES_PER_TOKEN is a structural fact about the graph and was NOT a "
            "predictor of its cost here."),
        applicability={
            "MODEL": "Qwen3.8-27B sealed-3.14 (NOETIC_PARENT_A), 3.1393 complete EBPW",
            "ARCHITECTURE": "Qwen3.8 hybrid, 48 DeltaNet + 16 GQA layers",
            "ORGAN": "whole decode graph; the outlier lever is the MLP gate/up pair",
            "REPRESENTATION": "mixed: HQ30UQ4 g64 mixer + HGRAVF01 affine_q2 g64 MLP",
            "SHAPE": "batch 1 decode, 11-token prompt, 49 generated tokens, max_seq_len 2048",
            "MACHINE": M3, "RUNTIME": HK,
            "KERNEL": ("qwen80_add_residual_rmsnorm_tg, qwen_uniform_q4_*_matvec_{qkv,"
                       "pair_concat}, qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128"),
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode (NOT prefill; S031 §7 keeps them separate)"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"],
        citations=[
            "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"
            "#FINDING_2_TWO_GRAPHS_AT_THE_SAME_DISPATCH_COUNT_DIFFER_BY_3_75_PERCENT",
            "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"
            "#FINDING_3_A_DISPATCH_IS_NOT_A_UNIT_OF_COST.marginal_us_per_removed_dispatch",
            "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "3 admitted sweeps per arm under a PRE-REGISTERED admission gate -- "
            "bench.machine_quiescence sampled before and after each run, admitted only with "
            "no process over 2 GiB RSS at either sample, 0 runs refused. Admitted by complete "
            "separation of clusters spreading 0.12-0.37%, not by a rep count. SHAPE and "
            "WORKLOAD_PHASE are deliberately not UNSCOPED: one prompt, one length, decode "
            "only. The 17.35 us outlier's MECHANISM is not established -- the receipt rules "
            "out host (flat at 1.5-1.8% of wall) and DRAM bandwidth (a 34.8x overshoot of "
            "this machine's measured 589.73 GB/s) and names no third cause."),
    ),
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
        law_id="AKB-A-REMOVED-DISPATCH-IS-PRICED-BY-ITS-BYTES",
        statement=(
            "COST PER REMOVED DISPATCH IS NOT A WELL-DEFINED QUANTITY unless the bytes held "
            "constant are stated. Fusing two half-width GEMV kernels into one full-width "
            "kernel removes a dispatch while moving ZERO extra bytes and returns 1.042 us; "
            "adding a dispatch that reads more weights costs 70.708 us. Same machine, same "
            "shape, same session -- a factor of 68. Doubling dispatches AND weights costs "
            "+23.76%, not +100%, so dispatches overlap and the machine is bandwidth-bound. "
            "OPERAND REUSE IS REFUTED as the mechanism at +0.418% against a pre-registered "
            "8-20%, with complete_separation False."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP GEMV; the production comparison point is the gate_up pair",
            "REPRESENTATION": "grouped absmax group=64 operands",
            "SHAPE": "rows=17408 cols=5120, 8704 threadgroups per dispatch",
            "MACHINE": M3, "RUNTIME": "MLX 0.32.1 mx.fast.metal_kernel JIT, 200 reps, 3 sweeps",
            "KERNEL": "hand-written MSL: single_only / reuse / reload / two_no_add / two_plus_add",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-token decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_PRICE.json"],
        citations=["receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_PRICE.json#result.THE_SEPARATION",
                   "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_PRICE.json#result.OPERAND_REUSE_IS_REFUTED",
                   "receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_PRICE.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "ONE shape, THREE sweeps, and BENCH_STATE CONTENDED -- so the absolute values are "
            "upper bounds and the 0.418% reuse figure sits INSIDE the 5.86-8.84% IQR and must "
            "not be read as a small positive. What carries the law is the SEPARATION: 1.042 vs "
            "70.708 us is a factor of 68, far outside that noise, and the +23.76% overlap "
            "figure is likewise well above it. The refutation is sound rather than merely "
            "unresolved because the pre-registered 8% floor WOULD have cleared this noise. "
            "MODEL and ARCHITECTURE are NONE because the arms are synthetic; the receipt "
            "withdraws the earlier 8.6 us figure AND the framing that produced it, and does "
            "NOT refute production's 17.35 us, which is a bytes-plus-dispatch number."),
    ),
    dict(
        law_id="AKB-THE-REDUCTION-TAIL-IS-FREE",
        statement=(
            "THE CROSS-LANE REDUCTION TAIL COSTS NOTHING MEASURABLE, AND DELETING IT "
            "ENTIRELY BOUNDS EVERY POSSIBLE REDUCTION VARIANT. On a 17408x5120 q4_g64 "
            "matvec at tpr64 the shipped serial tail -- one lane summing 64 threadgroup "
            "slots while 63 idle -- is INDISTINGUISHABLE from simd_sum at 0.981x and from a "
            "six-step tree at 0.960x, and a DELETION CONTROL that removes the barrier, the "
            "threadgroup array and the reduction altogether measures 0.985x. All eight "
            "minima across two runs and four arms span 5.3%, and the ranking REVERSES "
            "between runs -- the serial tail took the shortest minimum in one run and the "
            "longest in the other -- "
            "which is the signature of no effect. The deletion control is the decisive arm "
            "because it bounds what ANY tail can buy rather than ranking three tails against "
            "each other: IF REMOVING THE REDUCTION OUTRIGHT BUYS NOTHING, NO REDUCTION CAN. "
            "This closes the one structural candidate "
            "AKB-SIX-LEVERS-ELIMINATED-AND-THE-FLOOR-MECHANISM-IS-UNRESOLVED left named and "
            "untested, taking the count to SEVEN levers eliminated by measurement. THE "
            "FLOOR'S MECHANISM IS STILL NOT NAMED and no eighth story is offered -- what "
            "changed is that the kernel-level candidate list is now EMPTY rather than "
            "holding one open item."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64, identical in every arm",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 at tg128",
            "MACHINE": M3, "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps, two runs",
            "KERNEL": "native matvec with four reduction tails: serial, simd_sum, tree, deleted",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-token decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json"],
        citations=[
            "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json#THE_RESULT_IS_THAT_THE_TAIL_IS_FREE",
            "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json#P3_THE_BOUND",
            "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json#why_the_deletion_control_stores_from_every_lane",
            "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "The arms are comparable BY CONSTRUCTION and a test asserts it: the kernel body "
            "is byte-identical across all four sources, so the element count is identical "
            "and a difference is the tail and nothing else. The deletion control's store is "
            "UNCONDITIONAL on purpose, because predicating it on lane==0 lets the compiler "
            "sink the pure loop into the branch and the arm would measure dead-code "
            "elimination; the guard is verified by the measurement itself, since the arm "
            "lands ON the ~300 G elem/s element floor rather than 64x above it. It also ADDS "
            "64-fold store traffic, which makes the bound CONSERVATIVE. Anti-vacuity holds: "
            "the deletion control reads rel_err 0.986 against the float64 oracle while the "
            "three real tails read 1.4-2.0e-07, and three mutations were watched failing. "
            "Run 1 is CONTAMINATED at 105-259% round-spread so minima and an ordinal "
            "rounds-won count are reported beside every median; run 2 at 16-24% is the "
            "cleaner one and is the run in which the serial tail comes LAST. ONE shape, ONE "
            "tpr, ONE machine, INSTANCE -- the tail's share scales with TPR and with the "
            "per-element work in front of it, so a much smaller GROUPS/TPR ratio would not "
            "inherit this. tpr=32 with simd_sum, which needs no threadgroup memory and no "
            "barrier at all, is CONFOUNDED with geometry and NOT RUN. No shipped kernel "
            "changed and neither variant earns a place."),
    ),
    dict(
        law_id="AKB-SIX-LEVERS-ELIMINATED-AND-THE-FLOOR-MECHANISM-IS-UNRESOLVED",
        statement=(
            "THE PER-ELEMENT MATVEC FLOOR SURVIVES OPERAND REUSE AND INSTRUCTION-LEVEL "
            "PARALLELISM, THE TWO STRONGEST KERNEL-SIDE CANDIDATES, AND ITS MECHANISM IS "
            "STILL NOT NAMED. Staging x in threadgroup memory -- 20,480 bytes inside "
            "Metal's 32,768 allowance, the mechanism register blocking measured worth 2.48x "
            "for GEMM -- LOSES to reading x from device, 0.870x at 16 rows per stage "
            "and 0.499x at 2, "
            "reproduced in direction and magnitude across two runs, so the x reads were "
            "already cache-served and staging adds a pass, a barrier and threadgroup "
            "pressure while removing nothing. Four independent accumulators, breaking the "
            "serial dependency chain with the SAME loads, decode and element count, are "
            "INDISTINGUISHABLE at 0.987x on medians and 1.054x on minima, 8 of 14 rounds -- "
            "a coin flip. SIX LEVERS ARE NOW ELIMINATED BY MEASUREMENT: weight bytes, load "
            "instructions, decode arithmetic, the weight reads entirely, operand reuse and "
            "ILP. NO SEVENTH MECHANISM IS OFFERED -- this program's scorecard is five "
            "diagnoses written down and two wrong, and the one structural feature no arm "
            "varied, the serial cross-lane reduction tail, is recorded as a NAMED UNTESTED "
            "CANDIDATE rather than an explanation. AMENDED 2026-08-26 by "
            "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json: THAT CANDIDATE "
            "IS NOW DEAD, not pending -- simd_sum 0.981x, a tree 0.960x, and DELETING the "
            "reduction entirely 0.985x, so the count is SEVEN levers and the candidate list "
            "at the kernel level is EMPTY."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64, identical in every arm",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 at tg 128 and 1024",
            "MACHINE": M3, "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps",
            "KERNEL": "native matvec: device x, staged x at two amortisations, four accumulators",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-token decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_TWO_MORE_LEVERS_DIE.json",
                         "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json"],
        citations=["receipts/headless/ACCELERATOR_TWO_MORE_LEVERS_DIE.json#P1_OPERAND_REUSE_IS_REFUTED_AND_NOT_NARROWLY",
                   "receipts/headless/ACCELERATOR_TWO_MORE_LEVERS_DIE.json#P3_INSTRUCTION_LATENCY_IS_INDISTINGUISHABLE",
                   "receipts/headless/ACCELERATOR_TWO_MORE_LEVERS_DIE.json#THE_MECHANISM_IS_UNRESOLVED_AND_I_AM_NOT_REACHING_FOR_A_SIXTH_STORY",
                   "receipts/headless/ACCELERATOR_TWO_MORE_LEVERS_DIE.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "BOTH pre-registered predictions land on the wrong side, which is what carries "
            "it. The four-arm run is CONTAMINATED -- arm A shows 272.6% round-spread -- so "
            "medians are unreliable and minima and an ordinal rounds-won count are reported "
            "beside every one; what survives contamination is that the staging arms "
            "reproduce their direction and magnitude across two independent runs, and that "
            "no staged or ILP arm wins a majority of rounds. The ILP verdict rests on ONE "
            "run and its two estimators disagree in direction, which is why it is called "
            "indistinguishable rather than a small loss. Correctness precedes every timing "
            "at 1.8-2.0e-07 against a shared float64 oracle and the staging barrier control "
            "fired 8 of 8. ONE shape, ONE machine, INSTANCE; MODEL and ARCHITECTURE are "
            "NONE because the weights are synthetic at a borrowed shape. No shipped kernel "
            "changed and neither variant earns a place."),
    ),
    dict(
        law_id="AKB-THE-INSTANT-IS-THE-COST",
        statement=(
            "WHAT A SIMDGROUP ASKS FOR AT ONE INSTANT COSTS 1.11x TO 1.41x, AT AN ADDRESS SET THAT "
            "IS IDENTICAL LANE BY LANE. Rotating each lane's inner sequence by its own lane index "
            "leaves the per-lane group SET unchanged -- not the same count, THE SAME GROUPS -- so "
            "the per-simdgroup loop-wide set, its span, its fragment count and every stride in it "
            "are identical by construction, and the only thing that moves is WHICH ITERATION asks "
            "for WHICH: simdgroup 0's first iteration goes from ONE contiguous 1024-byte run to "
            "SEVENTEEN runs at GROUPS=80, and to THIRTY-TWO at GROUPS=256. Measured on an Apple M3 "
            "Ultra at 17408x5120 tpr64: 1.2199x, 1.1103x and 1.1668x across three runs losing 12, "
            "11 and 14 of 14 round-robin rounds, and 1.4095x at 17408x16384 losing 14 of 14, so "
            "HARDER SHATTERING COSTS MORE and the structural ladder was computed before the timing. "
            "THIS IS THE FIRST CONFIRMED MECHANISM AFTER TWELVE ELIMINATIONS -- weight bytes, load "
            "instructions, decode arithmetic, the weight reads at all, operand reuse, ILP, the "
            "reduction tail, the x reads, displacement (ill-posed), adjacency, fragmentation, "
            "stride magnitude and span -- and it explains them rather than replacing them, because "
            "not one of those levers changed what a simdgroup asks for at an instant while the "
            "shipped kernel already issues ONE contiguous request per iteration. THE CAUSE IS NOT "
            "NAMED: coalescer width, transaction count and queue occupancy are not separated here. "
            "AMENDED BY ACCELERATOR_THE_PENALTY_IS_A_STEP: THE WIDTH IS WRONG AND THE COST IS A "
            "STEP. A block rotation at k=32 leaves EVERY SIMDGROUP ITERATION one contiguous run "
            "and still costs 1.2434x and 1.2341x at 14 of 14 rounds twice, so the separating "
            "width is the THREADGROUP, whose request splits from one run to two there; and a "
            "six-rung granularity ladder is NOT MONOTONE, with k=4 reproducibly worst, so the "
            "penalty is a STEP at the first departure from a single contiguous request "
            "and not a gradient in the run count. AMENDED AGAIN BY "
            "ACCELERATOR_THE_WIDTH_IS_THE_ROW: THE CHARGED WIDTH IS THE ROW, NOT THE "
            "THREADGROUP. Sweeping the threadgroup width 64/128/256/512 takes the CONTROL from "
            "one contiguous threadgroup request to eight and costs NOTHING -- 0.9861/0.9936/"
            "1.0146 and 1.0035/1.0068/1.0029 against tg=64 across two runs, at coin-flip ordinal "
            "counts -- while the k=32 penalty holds 1.209x to 1.267x at EVERY width, tracking the "
            "64 lanes of a row, which a lane rotation never lets the threadgroup move. So the "
            "unit is excluded at 32 lanes and at 128-512 threads and located at 64, WHERE THE "
            "ROW, THE THREADS-PER-ROW AND A 2-SIMDGROUP PAIR COINCIDE AND ARE NOT SEPARATED. "
            "NARROWED AGAIN BY ACCELERATOR_THE_UNIT_IS_THE_ROW_AT_ITS_OWN_WIDTH, WHICH SEPARATES "
            "THEM: THE UNIT IS THE ROW AT WHATEVER WIDTH TPR SETS AND NOT A HARDWARE CONSTANT. "
            "At tpr=32 a row is ONE simdgroup while a fixed 64-lane hardware width would span TWO "
            "rows at unrelated addresses, so the two readings predict non-overlapping bands -- "
            "and the first departure there costs 1.2213x and 1.2456x at 14 of 14 rounds twice, "
            "inside the row band and twice the hardware-constant band, with per-lane work held "
            "identical at four groups and 256 elements. The step is sharper at the narrower row: "
            "two runs costs 22-25% while everything from two runs to thirty-two adds a further "
            "six points, and the k=4 anomaly does NOT reproduce at the same run count. "
            "NOTHING IS ADOPTED -- both arms are correct and the rotated one LOSES AT EVERY SHAPE "
            "MEASURED, so the shipped order already sits at the best point on this axis."),
        evidence_domain="accelerator", civilization="I-D_ACCELERATOR",
        machine_scope="Apple M3 Ultra, INSTANCE",
        representation_scope="ws_rtn_q4_g64",
        kernel_scope="native matvec under a per-lane iteration rotation",
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64",
            "SHAPE": "rows=17408 at cols 5120 (1.25 groups/lane) and 16384 (4 groups/lane), group 64, tpr64 at tg128",
            "MACHINE": M3,
            "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps, three runs at 5120 and one at 16384",
            "KERNEL": "native matvec with each lane's inner sequence rotated by its own lane index",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "a SINGLY SUBMITTED decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_THE_INSTANT_IS_THE_COST.json",
                         "receipts/headless/ACCELERATOR_THE_PENALTY_IS_A_STEP.json",
                         "receipts/headless/ACCELERATOR_THE_WIDTH_IS_THE_ROW.json",
                         "receipts/headless/ACCELERATOR_THE_UNIT_IS_THE_ROW_AT_ITS_OWN_WIDTH.json"],
        citations=[
            "receipts/headless/ACCELERATOR_THE_INSTANT_IS_THE_COST.json#P1_CONFIRMED_PAST_THE_TOP_OF_MY_OWN_BAND",
            "receipts/headless/ACCELERATOR_THE_INSTANT_IS_THE_COST.json#P2_CONFIRMED_AND_IT_IS_MONOTONE",
            "receipts/headless/ACCELERATOR_THE_INSTANT_IS_THE_COST.json#WHAT_THIS_MEANS_FOR_THE_EIGHT_DEAD_LEVERS",
            "receipts/headless/ACCELERATOR_THE_PENALTY_IS_A_STEP.json#P2_IS_THE_DISCRIMINATOR_AND_IT_IS_REFUTED_PAST_MY_FALSIFIER",
            "receipts/headless/ACCELERATOR_THE_PENALTY_IS_A_STEP.json#WHAT_THE_SIX_RUNGS_SAY_TOGETHER"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "THE ANTI-VACUITY CONTROL IS DIFFERENT IN KIND A THIRD TIME: a reordering permutes the "
            "same terms so every arm MUST be correct, and the replacements are the per-lane sets "
            "asserted EQUAL lane by lane, the order asserted DIFFERENT so a rotation that silently "
            "reduced to the shipped order cannot tie and read as a finding, the loop-wide span and "
            "fragment count asserted EQUAL rather than assumed, and an executing test running the "
            "generated Metal against a float64 oracle because the previous block watched a mutation "
            "survive a suite that pinned only its own Python. The reorder is visible in the answer: "
            "rel_err moves 2.021e-07 to 2.030e-07 and 2.512e-07 to 2.562e-07, independent evidence "
            "it landed. rot0 is the control and pays the identical runtime modulo in the shipped "
            "order, measured free at 5120 (1.0997 / 0.9862 / 1.0169, opposite directions). Three "
            "mutations watched failing. TWO POINTS on the groups-per-lane axis give a DIRECTION and "
            "not a shape, and the two shapes differ in total work, a confound named rather than "
            "buried -- what is not confounded is that rot loses to rot0 at each shape separately. "
            "BENCH_STATE CONTENDED, round-spreads 14.6% to 333.7%, minima the estimator and the "
            "ordinal counts carrying the verdict. ONE tpr, ONE machine, INSTANCE. No model "
            "executed, no capability run, no adequacy claim moves."),
    ),
    dict(
        law_id="AKB-LANE-ORDER-COSTS",
        statement=(
            "WHICH LANE TOUCHES WHICH ADDRESS COSTS ABOUT 8% AT FULL ELEMENT COUNT, AT AN IDENTICAL "
            "ADDRESS SET. Permuting lane -> (lane*37) % 64, a bijection that leaves the row's "
            "footprint, the threadgroup's footprint, the iteration count, the element count and the "
            "multiset of per-lane iteration counts UNCHANGED, measures 1.0802x and 1.0793x across two "
            "runs -- AGREEING TO 0.08% -- and loses 27 OF 28 round-robin rounds. This is the FIRST "
            "NAMED COMPONENT of the footprint mechanism ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT left "
            "unnamed: the simdgroup's 32 lanes touch the SAME addresses at the same instant either "
            "way, so it is not reach, not capacity and not cold lines -- the memory system cares which "
            "lane asks. AMENDED BY ACCELERATOR_THE_SIMDGROUP_SPAN_DOUBLES: THAT EXCLUSION RESTED ON "
            "WRONG ARITHMETIC. The address set is identical over 64 LANES and NOT over a 32-LANE "
            "SIMDGROUP, the coalescing unit, where the identity spans a CONTIGUOUS 1024 BYTES and "
            "EVERY permutation spans 2048 -- so reach and capacity were NEVER EXCLUDED and that "
            "exclusion must not be cited. The MEASUREMENT stands and survives ROTATING THE ARM "
            "ORDER, which had never been controlled for; the magnitude softens to a 1.03-1.11x "
            "range over six runs, and a reversal that keeps adjacency costs like a scatter, so "
            "adjacency is not the variable either. AMENDED AGAIN BY "
            "ACCELERATOR_SPAN_IS_REFUTED_COALESCING_IS_NOT: THE SPAN CANDIDATE IS WITHDRAWN. A "
            "BLOCKED PARTITION varies it without a permutation -- same per-lane count lane by "
            "lane, simd0 cut from 2560 bytes in two fragments to a contiguous 1536 -- and it is "
            "1.0315x and 1.0303x SLOWER, so span joins fragmentation and stride magnitude as "
            "EXCLUDED. What survives is PER-ITERATION COALESCING, which no arm had varied: the "
            "shipped assignment is the only one issuing ONE contiguous request per iteration, "
            "and at this shape, tpr and machine no other arm has beaten it. A CANDIDATE, "
            "NOT A CLAIM. IT DOES NOT HAPPEN AT THIN WORK: 0.9911 and 1.0202 in opposite directions, 5 "
            "and 7 of 14 rounds, so the effect scales with concurrent requests per lane, which is a "
            "CANDIDATE and not a claim. THE MAGNITUDES COINCIDE WITH THE FOOTPRINT EFFECT (1.082x and "
            "1.098x) AND THAT IS NOT AN IDENTITY CLAIM, because confining the footprint moves the "
            "address set AND the ordering together and these arms do not separate them. AND THE BAND "
            "WAS TOO WIDE TO SEE IT: 1.08x sits INSIDE the pre-registered +-10% indistinguishable "
            "band, so the pre-registered verdict is CONFIRMED while the effect is real -- the ORDINAL "
            "rounds-won count and the cross-run agreement are what carry it, never the ratio."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 at tg128; full and 2-element-per-group work",
            "MACHINE": M3,
            "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps, two runs",
            "KERNEL": "native matvec with lanes permuted over groups by an odd multiplier",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "a SINGLY SUBMITTED decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_LANE_ORDER_COSTS.json",
                         "receipts/headless/ACCELERATOR_THE_SIMDGROUP_SPAN_DOUBLES.json",
                         "receipts/headless/ACCELERATOR_SPAN_IS_REFUTED_COALESCING_IS_NOT.json"],
        citations=[
            "receipts/headless/ACCELERATOR_LANE_ORDER_COSTS.json#P1_LANE_ORDER_COSTS_AT_FULL_ELEMENT_COUNT",
            "receipts/headless/ACCELERATOR_LANE_ORDER_COSTS.json#WHAT_THIS_NAMES_THAT_WAS_UNNAMED",
            "receipts/headless/ACCELERATOR_LANE_ORDER_COSTS.json#P2_AND_IT_DOES_NOT_HAPPEN_AT_THIN_WORK",
            "receipts/headless/ACCELERATOR_LANE_ORDER_COSTS.json#WHAT_THIS_PROBE_STRUCTURALLY_CANNOT_SEPARATE_SAID_IN_ADVANCE",
            "receipts/headless/ACCELERATOR_THE_SIMDGROUP_SPAN_DOUBLES.json#AND_THE_CENTRAL_ARITHMETIC_OF_MY_PREVIOUS_BLOCK_IS_WRONG",
            "receipts/headless/ACCELERATOR_THE_SIMDGROUP_SPAN_DOUBLES.json#THE_ORDER_ARTIFACT_I_HAD_NOT_CONTROLLED_FOR",
            "receipts/headless/ACCELERATOR_SPAN_IS_REFUTED_COALESCING_IS_NOT.json#P1_IS_MINE_AND_IT_IS_REFUTED_PAST_MY_OWN_FALSIFIER",
            "receipts/headless/ACCELERATOR_SPAN_IS_REFUTED_COALESCING_IS_NOT.json#WHAT_SURVIVES_IS_A_PROPERTY_NO_ARM_HAD_EVER_VARIED"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "THE CONTROL IS THE IDENTITY PERMUTATION, NOT THE SHIPPED KERNEL: _lane1 carries the same "
            "multiply and modulo as _lane37, so the comparison prices ORDER and not two extra "
            "instructions, and that arithmetic is separately measured free (0.99x / 1.0398x, mixed in "
            "direction). THE ANTI-VACUITY CONTROL IS DIFFERENT IN KIND from every other probe in this "
            "family: those were WRONG BY CONSTRUCTION and had to be, while a bijection MUST BE "
            "CORRECT, so its replacement is a bijection verified over all 64 lanes rather than "
            "inferred from coprimality, an assertion that the sources differ in EXACTLY ONE LINE, and "
            "a refusal of even multipliers which would skip and double-visit groups while looking "
            "plausible. lane37 reads rel_err 1.952e-07 against lane1's 2.021e-07 -- different, which "
            "is itself evidence the permutation moved which lane accumulates what. Three mutations "
            "watched failing, including collapsing the permutation to the identity, which would tie "
            "and read as a finding. BENCH_STATE CONTENDED at 36-261% round-spread; minima are the "
            "estimator and the ordinal count carries the verdict. ONE permutation, ONE shape, ONE "
            "machine, INSTANCE. The reordered kernel is CORRECT and therefore a legitimate candidate, "
            "and it is SLOWER, so nothing is adopted and no shipped kernel changed."),
    ),
    dict(
        law_id="AKB-THE-ENTRY-COST-IS-FOOTPRINT",
        statement=(
            "THE LOOP-ENTRY COST OF AN ISOLATED q4_g64 MATVEC IS THE ADDRESS FOOTPRINT, NOT THE LOOP. "
            "Confining every address to ONE group while holding the loop bound, the iteration count, "
            "the element count, the scale loads, the reduction, the grid and the stores IDENTICAL "
            "measures 1.472x and 1.453x at two elements per group, winning 13 of 14 and 12 of 14 "
            "round-robin rounds -- and the confined arm lands ON the trivial one-store kernel, 0.2116 "
            "and 0.2061 ms against 0.2144 and 0.1998, so the ENTIRE loop-entry gap "
            "AKB-ENTERING-THE-LOOP-COSTS-MORE-THAN-RUNNING-IT measured is address spread. AT THE FULL "
            "ELEMENT COUNT the same variable is worth 1.082x and 1.098x, 11 and 12 of 14 rounds: "
            "spreading a row's addresses over 2560 bytes instead of 64 costs 8-10% at production "
            "shape, which is the FIRST LEVER IN EIGHT BLOCKS TO MOVE ANYTHING. It does NOT contradict "
            "the weight reads measuring free by deletion "
            "(ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE): that control deleted packed's 2560 "
            "bytes per row while LEAVING x's 20480 and the scales in place, so the footprint barely "
            "moved and the effect could not appear -- a control that deletes one of three operands is "
            "not a footprint experiment. THE CONFINED ARM IS WRONG BY CONSTRUCTION and is NOT A "
            "CANDIDATE KERNEL; this says what the cost is MADE OF, not that a faster kernel exists. "
            "The MECHANISM is not named -- cold lines, TLB reach, cache capacity and coalescing are "
            "NOT separated here."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 at tg128; 2 and 64 elements per group",
            "MACHINE": M3,
            "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps, two runs",
            "KERNEL": "native matvec with every address confined to N groups at fixed work",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "a SINGLY SUBMITTED decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT.json"],
        citations=[
            "receipts/headless/ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT.json#P1_IS_MINE_AND_IT_IS_REFUTED",
            "receipts/headless/ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT.json#P2_footprint_at_FULL_element_count",
            "receipts/headless/ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT.json#HOW_THIS_RECONCILES_WITH_THE_WEIGHT_READS_BEING_FREE_RATHER_THAN_CONTRADICTING_IT",
            "receipts/headless/ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT.json#WHAT_THIS_IS_NOT"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "My own P1 predicted INDISTINGUISHABLE within 10% with a 1.15x falsifier, on the strength "
            "of three earlier free-read results, and it is refuted past that falsifier TWICE. P2 "
            "predicted the same at full element count and is refuted too. The single-variable claim is "
            "asserted LINE BY LINE against the base source rather than believed, the index still "
            "varies with g so the body cannot be hoisted whole, and local at the FULL group count "
            "reproduces the shipped kernel exactly so the family contains what executes. Every "
            "confined probe is WRONG at rel_err ~0.98 and separately checked NON-DEGENERATE, because "
            "a folded loop is also wrong. Three mutations watched failing. BENCH_STATE CONTENDED with "
            "round-spreads 110-375%, so minima are the estimator and the ORDINAL 13/14 and 12/14 "
            "rounds-won counts are what carry the verdict rather than the ratios. ONE shape, ONE tpr, "
            "ONE machine, INSTANCE. No shipped kernel changed."),
    ),
    dict(
        law_id="AKB-ENTERING-THE-LOOP-COSTS-MORE-THAN-RUNNING-IT",
        statement=(
            "IN AN ISOLATED q4_g64 MATVEC THE FIRST ELEMENTS OF THE INNER LOOP COST FAR MORE THAN "
            "THE REST. Going from NO element work to TWO elements per 64-wide group costs "
            "0.069-0.099 ms across three runs; going from two to SIXTY-FOUR costs 0.006-0.031 ms. "
            "The first two elements cost 2.2x to 15x what the remaining sixty-two cost, in every "
            "run, which is roughly 80x to 500x per element -- a WIDE RANGE because both terms are "
            "differences between small contended quantities, so the DIRECTION transfers and the "
            "magnitude is a range. THIS NAMES A SHAPE THAT WAS ALREADY MEASURED AND UNEXPLAINED: "
            "ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER found a FLAT low region, three arms "
            "within 0.2% across an 8x work range, and called its linear model wrong in shape "
            "without naming the shape. The cost sits in the FIRST iteration, so widening 2 to 16 "
            "adds almost nothing and a knee appears only near full width. AND THE REDUCTION IS NOT "
            "THE TERM: isolated with a single variable it ties three times (-0.0015, +0.0030, "
            "-0.0009 ms) and priced at IDENTICAL STORE TRAFFIC it is 0.0023 ms = 3.4% of the gap. "
            "The mechanism of the first-iteration cost is NOT NAMED -- first touch of cold cache "
            "lines is the obvious candidate and it is recorded as untested."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64, identical where read",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 at tg128, 2 vs 64 elements per group",
            "MACHINE": M3,
            "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps, three runs",
            "KERNEL": "native matvec decomposed into trivial, reduction-only, loop-without-barrier, loop-with-barrier and full",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "a SINGLY SUBMITTED decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_ENTERING_THE_LOOP_COSTS_MORE_THAN_RUNNING_IT.json"],
        citations=[
            "receipts/headless/ACCELERATOR_ENTERING_THE_LOOP_COSTS_MORE_THAN_RUNNING_IT.json#THE_RESULT_THIS_BLOCK_DID_NOT_SET_OUT_TO_GET",
            "receipts/headless/ACCELERATOR_ENTERING_THE_LOOP_COSTS_MORE_THAN_RUNNING_IT.json#P1_THE_REDUCTION_IS_REFUTED_AS_THE_TERM",
            "receipts/headless/ACCELERATOR_ENTERING_THE_LOOP_COSTS_MORE_THAN_RUNNING_IT.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "The named suspect was REFUTED BY ARITHMETIC BEFORE ANY ARM RAN -- 1.39M scale loads "
            "over 1.11M threads is 1.25 per thread and cannot be a fifth of a kernel -- so no arm "
            "was spent on it and the correction is recorded as pre-run. My own P1 predicted the "
            "reduction was 60% of the term and it is refuted three times. A DEFECT IN MY OWN "
            "DESIGN is what makes the barrier price trustworthy: the obvious from-above arm also "
            "turns one predicated store per row into an unconditional store from every lane, 64 "
            "fold, and its disagreement with the from-below route in SIGN is what exposed it; the "
            "equal-store arm was built for that and the two routes then agree. Every probe is "
            "WRONG BY CONSTRUCTION and separately checked NON-DEGENERATE, with the reduction-only "
            "arm's accumulator depending on the ROW as well as the lane because a per-lane "
            "constant would reduce to the same value everywhere. Three mutations watched failing, "
            "and a fourth REPORTED SURVIVING because it hit an identical line in a different tail "
            "-- re-anchored on two lines, then caught. BENCH_STATE CONTENDED with round-spreads to "
            "375%; minima are the estimator and medians sit beside them. ONE shape, ONE tpr, ONE "
            "machine, INSTANCE. No shipped kernel changed."),
    ),
    dict(
        law_id="AKB-GRID-LAUNCH-IS-NOT-FREE",
        statement=(
            "DISPATCHING MORE THREADGROUPS COSTS TIME EVEN WITH NO WORK TO DO, at about 2-6 "
            "nanoseconds per threadgroup on this chip. A trivial one-store-per-row kernel holding "
            "its OUTPUT fixed while the grid sweeps 136 to 8704 threadgroups -- 64x, with the work "
            "fixed at nothing -- rose 12.0% and 35.4% across two runs, MONOTONICALLY from tpr4 "
            "upward in both. The direction reproduces and the MAGNITUDE DOES NOT, so the price is "
            "a range and not a value. THIS DOES NOT CONTRADICT the earlier measurement that tpr1 "
            "took longer than tpr64 on the REAL kernel: there the wide grid had 64x the "
            "parallelism to apply to real work and that gain outweighs the launch cost, while here "
            "there is no work to parallelise so the launch cost is the only effect left. A LAUNCH "
            "COST IS INVISIBLE WHENEVER THE GRID BUYS PARALLELISM, which is why it took a kernel "
            "that does nothing to see it. AMENDED BY ACCELERATOR_THE_WIDTH_IS_THE_ROW: THE "
            "ATTRIBUTION TO THREADGROUPS IS NOT SUPPORTED BY A CLEANER AXIS. That sweep moved "
            "THREAD COUNT AND THREADGROUP COUNT TOGETHER; holding threads FIXED at 1,114,112 on a "
            "REAL kernel while the threadgroup count falls 8x, 17408 to 2176, moves NOTHING -- "
            "1.5% one way in one run and 0.3% the other in the second, the direction BACKWARDS "
            "for a launch cost in both, bounding it under 0.5 ns per threadgroup against the 2-6 "
            "reported here. EITHER the price is per THREAD rather than per threadgroup, which the "
            "confounded axis could not separate, OR real work hides it; BOTH ARE LIVE AND NEITHER "
            "IS PICKED. The 12.0% and 35.4% rises are NOT withdrawn -- what is withdrawn is "
            "reading them as a per-threadgroup price."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "one f16 scale read per row; no weights read",
            "SHAPE": "17408 outputs, grid swept 17408 to 1114112 threads at threadgroup 128",
            "MACHINE": M3,
            "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps, twice",
            "KERNEL": "a trivial one-store-per-row probe, WRONG BY CONSTRUCTION",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "a SINGLY SUBMITTED dispatch doing no work"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS.json",
                         "receipts/headless/ACCELERATOR_THE_WIDTH_IS_THE_ROW.json"],
        citations=[
            "receipts/headless/ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS.json#P2_THE_GRID_AXIS_AND_MY_PREDICTION_IS_REFUTED",
            "receipts/headless/ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS.json#THE_DECOMPOSITION",
            "receipts/headless/ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=False,
        confidence_basis=(
            "My own pre-registered prediction was FLAT WITHIN 15% and it is REFUTED, which is what "
            "carries this. The arms differ ONLY in launch: all seven return BITWISE IDENTICAL "
            "output, pinned by a test, so a timing difference cannot be a different computation. "
            "The probe is WRONG BY CONSTRUCTION at rel_err 1.000 against the float64 oracle -- the "
            "anti-vacuity condition -- and separately checked NON-DEGENERATE at 17408 of 17408 "
            "nonzero with 1289 distinct values, because a kernel storing a CONSTANT is also wrong "
            "and would time an empty dispatch rather than a cheap one. Every buffer is referenced, "
            "multiplied by zero where unused, because MLX binds the signature from the named "
            "inputs and an unreferenced buffer would make this a different dispatch from its "
            "comparands. Three mutations were watched failing. THE GRID IS VARIED BY CHANGING "
            "THREADS-PER-ROW, so lane count moves with it -- legitimate only because this kernel "
            "has NO cross-lane structure at all, and a kernel that did would not inherit this. "
            "BENCH_STATE CONTENDED with round-spreads 14-472%; minima are the estimator and "
            "medians sit beside them. ONE shape, ONE machine, INSTANCE. No shipped kernel changed."),
    ),
    dict(
        law_id="AKB-AN-ISOLATED-MATVEC-IS-MOSTLY-NOT-ITS-ELEMENTS",
        statement=(
            "AT THIS SHAPE 86% OF AN ISOLATED q4_g64 MATVEC'S MEASURED TIME DOES NOT DEPEND ON "
            "ITS ELEMENT COUNT. Holding the grid, the lane assignment, the group loop, the "
            "reduction and the stores FIXED and deleting 96.9% of the inner iterations returns "
            "18%; a five-point sweep over element counts spanning 32x fits a FIXED cost of "
            "0.2450 ms on minima and 0.2600 on medians, 86.1% and 84.8% of the full kernel, and "
            "the three narrowest arms read FLAT to 0.2% across an 8x work range so the shape is a "
            "FLOOR WITH A KNEE rather than a line. Removing each named per-element operation "
            "confirms it from the other side: half the x reads 0.947x, the multiply 0.979x, and "
            "EVERY x READ removed 0.987x and 1.019x. This is the SIXTH independent sighting of "
            "the ~0.16-0.25 ms per-submission floor and the fitted 0.2450 agrees to 0.5% with the "
            "0.2438 ms intercept ACCELERATOR_EXPERT_BATCH fitted on a different kernel. IT "
            "EXPLAINS THE SEVEN DEAD LEVERS rather than adding an eighth: fewer bytes, fewer "
            "loads, cheaper decode, no weight reads, operand reuse, ILP and the whole reduction "
            "tail could not move a number five parts fixed to one part element work. PRODUCTION "
            "DOES NOT PAY IT -- 402 weight-carrying dispatches inside a 29.29 ms token average "
            "0.0729 ms each, 3.4x BELOW this isolated floor, because 964 dispatches share a "
            "submission. SUBMISSION AND GRID LAUNCH ARE NOT SEPARATED HERE and no claim divides "
            "the 0.245 ms between them. AMENDED 2026-08-26 by "
            "receipts/headless/ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS.json, WHICH SEPARATES "
            "THEM: a trivial one-store-per-row probe holding the OUTPUT fixed while the grid "
            "sweeps 136 to 8704 threadgroups costs 50-64% of the baseline, extrapolates to "
            "0.152-0.163 ms at zero threadgroups -- landing on the ~0.157 ms submission "
            "ACCELERATOR_GRAPH_SUBMISSION measured directly -- and RISES 12% and 35% across the "
            "64x grid, so grid launch is NOT free at about 2-6 ns per threadgroup. THE FIXED "
            "COST IS NOT ONE THING: submission is the largest term, grid launch is real and was "
            "invisible to an axis that held the grid fixed, and 18-30% is the GROUP LOOP, its "
            "1.39M scale loads and the reduction -- per-row work an INNER-loop sweep holds "
            "constant and therefore deposits in the intercept. A FIXED INTERCEPT IS ONLY AS "
            "SPECIFIC AS THE AXIS THAT WAS VARIED. The 86%-not-elements measurement STANDS and "
            "is corroborated, the element term measuring under 15% there by another route."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64, identical in every arm",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 at tg128, element count swept 32x",
            "MACHINE": M3, "RUNTIME": "MLX mx.fast.metal_kernel JIT, 14 round-robin rounds x 20 reps",
            "KERNEL": "native matvec with one per-element operation removed per arm, and an element-count sweep at fixed grid",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "a SINGLY SUBMITTED decode-shaped GEMV, CONTENDED machine -- and the "
                              "singly-submitted part is the whole point"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json",
                         "receipts/headless/ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS.json"],
        citations=[
            "receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json#THE_FIT_MAKES_IT_A_NUMBER",
            "receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json#THE_SIXTH_SIGHTING_AND_IT_EXPLAINS_THE_WHOLE_WALL",
            "receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json#P1_THE_x_READ_DELETION_TIES",
            "receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "Two routes to the same floor: the regression intercept is 0.2450 ms and the flat "
            "region's own minimum is 0.2475, 1% apart. The linear model is WRONG IN SHAPE and the "
            "receipt says so -- R2 0.888 with three arms flat across 8x work -- so the intercept "
            "is quoted as a floor estimate and the slope is not offered as a per-element cost. "
            "Every probe is WRONG BY CONSTRUCTION at rel_err 0.98-4.88 against the baseline's "
            "2.02e-07, which is the anti-vacuity condition, and each is separately checked "
            "NON-DEGENERATE at 17408 of 17408 nonzero rows because a folded loop is also wrong. "
            "The widest sweep arm is asserted BYTE-IDENTICAL to the shipped kernel so the family "
            "cannot exclude what executes, and a template-drift guard raises rather than letting "
            "a no-op replacement make a probe BE the baseline. Four mutations were watched "
            "failing -- and two earlier ones reported surviving without ever landing, a shell "
            "escaping error that reads exactly like a suite that cannot detect them. BENCH_STATE "
            "CONTENDED with round-spreads 16-271%, so minima and an ordinal rounds-won count sit "
            "beside every median. ONE shape, ONE tpr, ONE machine, INSTANCE: the 86% share is "
            "this shape's, and a kernel with far more element work per submission would show "
            "less, which is the finding rather than a caveat. No shipped kernel changed and no "
            "production claim moves."),
    ),
    dict(
        law_id="AKB-THE-MATVEC-FLOOR-IS-THE-ELEMENT-NOT-THE-BYTE",
        statement=(
            "A REPRESENTATION-NATIVE MATVEC AT THE RESIDENT'S MLP SHAPE IS BOUND BY ITS "
            "ELEMENT COUNT, NOT BY ITS BYTES AND NOT BY ITS UNPACK. A deletion control "
            "with the SAME geometry, reduction and element count that NEVER READS THE "
            "PACKED ARRAY measures 0.2897 ms against the real kernel's 0.2825 -- the real "
            "kernel FASTER, a statistical tie -- so 44.6 MB of weight traffic costs "
            "NOTHING MEASURABLE. Two candidate levers are eliminated by measurement: "
            "4-byte loads cut load instructions FOURFOLD and change nothing (1.004x, 7 of "
            "12 round-robin rounds, a coin flip), and a 256-entry LUT removing the "
            "mask/shift/convert/subtract chain is 16.1% SLOWER (1 of 12 rounds). The "
            "floor is one x read and one fused multiply-add per weight at 307.7 G elem/s, "
            "and weight traffic below ~171 MB at this element count hides beneath it. "
            "CONSEQUENCE FOR THE INSTRUMENT: effective GB/s is weight_bytes/time, so when "
            "time is set by elements it measures THE REPRESENTATION'S DENSITY AND NOT THE "
            "MACHINE -- the same kernel would report half the GB/s at half the bpw without "
            "running any faster."
            "AMENDED 2026-08-26 by "
            "receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json: THIS "
            "LAW'S OWN NAME IS WRONG. The floor is NEITHER the byte NOR the element. Halving "
            "the x reads reads 0.947x, removing the MULTIPLY 0.979x, and REMOVING EVERY x READ "
            "0.987x and 1.019x across two runs -- so neither operation the phrase 'one x read "
            "and one fused multiply-add' names costs anything. Deleting 96.9% of the element "
            "work at a FIXED GRID returns 18%, and a five-point sweep fits a FIXED cost of "
            "0.2450 ms = 86.1% of the full kernel, with eight times the element work reading "
            "FLAT to 0.2%. What the arms here measured is unchanged; what they tied AGAINST was "
            "mostly not element work. The ~171 MB crossover is WITHDRAWN."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE,
            "ORGAN": "MLP-shaped GEMV at 89.1M weights, a shape borrowed from the resident",
            "REPRESENTATION": "ws_rtn_q4_g64 in every arm; the control reads no weights",
            "SHAPE": "rows=17408 cols=5120, group 64, tpr64 tg128",
            "MACHINE": M3, "RUNTIME": "MLX mx.fast.metal_kernel JIT, 12-16 round-robin rounds x 20 reps",
            "KERNEL": "native matvec under byte / word / lut unpacks plus a no-weights control",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-token decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE.json",
                         "receipts/headless/ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER.json"],
        citations=["receipts/headless/ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE.json#BOTH_PREDICTIONS_LAND_ON_THE_WRONG_SIDE",
                   "receipts/headless/ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE.json#AND_THE_DELETION_CONTROL_SAYS_WHAT_THE_WALL_IS",
                   "receipts/headless/ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE.json#AN_INSTRUMENT_CORRECTION_THAT_MATTERS_MORE_THAN_THE_ARMS",
                   "receipts/headless/ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "BOTH pre-registered predictions land on the WRONG side, which is what carries "
            "it: the load-width lever was predicted to win 1.5x and is refuted at 1.004x, "
            "the LUT was predicted not to win and LOSES. The first sweep was DISCARDED and "
            "why is part of the evidence -- sequential arms put byte first and it read "
            "116.4% IQR at 3.4x its own previous measurement, so a transient lands on "
            "whoever holds the clock; round-robin then reproduced the previous block's "
            "byte figure to 4.8%. The floor probe's estimator is the MINIMUM of round "
            "medians, stated because contention only adds time, and its control carries "
            "63.1% round-spread so 'the bytes are free' is DIRECTIONAL -- admitted because "
            "the tie runs in the wrong direction for a cost, not by a clean gate. THE "
            "DENSE DECOMPOSITION IS REJECTED BY THIS PROGRAM'S OWN IMPLAUSIBILITY RULE at "
            "an implied 617.3 GB/s over the measured 589.73 roof, since the arms have "
            "different launch geometries; no dense number is claimed. ONE shape, ONE "
            "machine, INSTANCE; the ~171 MB crossover scales with the element count and is "
            "this shape's number. Correctness precedes every timing at 2.0-2.2e-07 against "
            "a shared float64 oracle. No shipped kernel changed."),
    ),
    dict(
        law_id="AKB-THE-UNPACK-IS-THE-WALL-NOT-THE-BYTES",
        statement=(
            "A REPRESENTATION-NATIVE MATVEC ON THIS MACHINE IS ARITHMETIC-BOUND ON ITS "
            "UNPACK, AND THE THREAD GEOMETRY IS NOT THE LEVER. At the resident's real MLP "
            "shape 17408x5120, dense f32 sustains 405.7 GB/s while ws_rtn_q4_g64 native "
            "reaches 138.7 GB/s at one thread per row and 160.9 GB/s at the resident's own "
            "SIXTY-FOUR threads per row -- 1.16x for a 64x change in lanes, against a "
            "pre-registered 1.5x that is REFUTED. AMENDED 2026-08-26 by "
            "ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE: THE UNPACK IS NOT THE "
            "WALL EITHER -- a deletion control reading NO packed bytes measures 0.2897 ms "
            "against the real kernel's 0.2825, a LUT decode is 16.1% SLOWER and 4-byte "
            "loads change nothing at 1.004x, so the wall is the PER-ELEMENT rate at "
            "307.7 G elem/s and weight traffic under ~171 MB hides beneath it. The "
            "native arm reads 7.53x FEWER BYTES "
            "for only 2.99x less time, converting 40% of its own byte ratio and burning "
            "the rest on the unpack, and its best geometry sits at 27.3% of the measured "
            "589.73 GB/s roof. THE BODY-LEVEL COROLLARY IS ALREADY IN PRODUCTION DATA: "
            "three bodies of one parent at 3.1393 / 2.9802 / 2.5970 complete EBPW record "
            "34.14 / 34.12 / 33.12 raw TPS, so Spearman(EBPW, TPS) is +1.000 where "
            "bandwidth-bound demands -1.000, effective bandwidth FALLS 337.3 -> 318.8 -> "
            "266.8 GB/s as bytes fall, and a 17.3% byte cut returned -3.0% instead of "
            "+20.9%. FEWER BYTES ARE NECESSARY AND NOT SUFFICIENT."),
        applicability={
            "MODEL": ("kernel arms NONE -- synthetic weights at a borrowed shape; the body "
                      "corollary is the three NOETIC_PARENT_A bodies at 3.1393 / 2.9802 / "
                      "2.5970 complete EBPW"),
            "ARCHITECTURE": "kernel arms NONE; the body corollary is qwen3_5, 48 DeltaNet + 16 GQA",
            "ORGAN": "MLP-shaped GEMV; the body-level corollary is whole-decoder",
            "REPRESENTATION": "ws_rtn_q4_g64 measured; the body corollary spans the "
                              "sealed mixed hq30uq4 / hgrafv01 ladder",
            "SHAPE": "rows=17408 cols=5120, group 64",
            "MACHINE": M3, "RUNTIME": "MLX mx.fast.metal_kernel JIT, 200 reps, 40 warmup",
            "KERNEL": "hand-written MSL: dense f32, native tpr1, native tpr64",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-token decode-shaped GEMV, CONTENDED machine"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json"],
        citations=["receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json#THE_WALL",
                   "receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json#PREDICTIONS",
                   "receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json#THE_PRODUCTION_EVIDENCE_THAT_SETTLES_THE_DIRECTION",
                   "receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "BENCH_STATE CONTENDED with the operator's 40.63 GiB job at 102.1% CPU left "
            "running, so every absolute GB/s is a LOWER BOUND and the arm-to-arm ratios "
            "measured back to back in one process are what carries it; arms clear the 10% "
            "gate at 6.30 / 7.68 / 9.68% IQR. The timed kernels are ws_rtn_q4_g64 while the "
            "resident's MLP is affine_q2 group 32 at 2.50 bpw, a DIFFERENT format reading "
            "fewer bytes and doing MORE unpack per byte -- the DIRECTION transfers and the "
            "absolute rate does not, which is why MODEL and ARCHITECTURE are NONE. The "
            "three-body TPS figures are RECORDED in S031 s0 and not re-measured, and their "
            "3.0% spread could be noise; what is not noise-sized is the ABSENCE of the "
            "+20.9% a bandwidth-bound model demands. Correctness precedes every timing at "
            "2.0e-07 to 1.2e-06 against a shared float64 oracle, with the tpr64 barrier "
            "control watched firing. TWO geometry points only; no optimum is claimed."),
    ),
    dict(
        law_id="AKB-THE-DISPATCH-LADDER-CANNOT-REACH-THE-ACCEPTED-TPS-TARGET",
        statement=(
            "THE DECODE GRAPH'S COUNT COLUMN AND ITS BYTES COLUMN DISAGREE BY THREE ORDERS "
            "OF MAGNITUDE. On the sealed 3.1393 resident, 402 of 964 dispatches -- 41.6% of "
            "the count -- carry 99.893% of the weight bytes; the other 562 carry 0.107%. "
            "Priced at the measured 1.042 us for a byte-free removal, deleting EVERY "
            "weight-free dispatch, far past the S031 s4 target of 200, returns at most "
            "0.586 ms against a 29.29 ms token, about 2% -- the same order as the 2.19% the "
            "fusion block measured end to end from a different direction. AND THE CEILING IS "
            "HARD: at 9.879 GB of weight traffic per token the sealed body runs at "
            "337.27 GB/s = 57.2% of this machine's measured 589.73 GB/s roof, so a PERFECT "
            "machine tops out at 59.70 raw TPS while 50 ACCEPTED TPS at the 30/43 capability "
            "floor needs 71.67. NO DISPATCH COUNT REACHES THE TARGET; only fewer bytes do, "
            "which is S031 s2's own order with a number attached. AMENDED 2026-08-26 by "
            "ACCELERATOR_UNPACK_IS_THE_WALL: fewer bytes are NECESSARY and NOT "
            "SUFFICIENT -- the 17.3% byte cut from 3.1393 to 2.5970 EBPW returned "
            "-3.0% raw TPS, not the +20.9% bandwidth-bound demands."),
        applicability={
            "MODEL": "sealed-3.14 (NOETIC_PARENT_A), 26.896G parent params",
            "ARCHITECTURE": "qwen3_5 64-layer hybrid, 48 DeltaNet + 16 GQA",
            "ORGAN": "whole decode graph, all 14 kernel families",
            "REPRESENTATION": "mixed hq30uq4 / hgrafv01 / f32v2 at 3.1393 complete EBPW",
            "SHAPE": "single-token decode, 11-token prompt",
            "MACHINE": M3, "RUNTIME": "Hawking Rust Metal decode, dispatch counter",
            "KERNEL": "all 964 dispatches of the resident graph",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "single-request decode; says nothing about prefill"},
        evidence_class="Derived",
        source_receipts=["receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json"],
        citations=["receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json#THE_COUNT_COLUMN_AND_THE_BYTES_COLUMN_DISAGREE",
                   "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json#THE_CEILING",
                   "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json#three_reconciliations_that_make_it_falsifiable",
                   "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json#claim_boundary"],
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "NOTHING WAS TIMED. The bytes are read off the artifact's own catalog and the TPS "
            "figures are arithmetic over a RECORDED 34.14 raw TPS and a RECORDED 589.73 GB/s "
            "roof -- a DERIVATION, not an experiment, and classed Derived for that reason. "
            "What carries it is the reconciliation: the attribution sums to the catalog total, "
            "the catalog total equals bytes on disk at ZERO delta, and 8*bytes/params "
            "reproduces the sealed 3.139300850311054 to twelve places; and the map derived "
            "from the tensor inventory reproduces the runtime's independently measured 964, "
            "having FAILED at 948 until a missing weight-free kernel was found. Weight traffic "
            "ONLY -- activations, KV and the DeltaNet state are uncounted, and every uncounted "
            "byte LOWERS the ceiling, so the conclusion sits on the safe side of its own "
            "omission: including the largest omission moves 59.70 to 57.93, nowhere near "
            "71.67. ONE artifact, INSTANCE, no kernel changed and no adequacy claim moves."),
    ),
    dict(
        law_id="AKB-WEIGHT-METRICS-INVERT-ON-ATTENTION-Q",
        statement=(
            "For attention q_proj under ws_rtn_q4_g64, WEIGHT-SPACE FIDELITY RANKS FOUR "
            "SPECIMENS IN EXACTLY THE WRONG ORDER against real-activation error. Cosine runs "
            "0.988150 / 0.989516 / 0.994001 / 0.994252 while real-activation relative error "
            "runs 0.032677 / 0.033354 / 0.056519 / 0.065670 -- monotone and fully reversed, "
            "Spearman -1.000 on two token sets and -0.800 on two more. It is NOT the "
            "scale-invariance defect: weight_rel_err, which is not scale-invariant, inverts "
            "identically. The GAUSSIAN activation proxy fails on the SAME boundary, "
            "overstating error by up to 1.6x on q_proj while landing within 7% on the expert "
            "gate. CONSEQUENCE: the weight-space floor procedure assigns 4.500 bpw to the "
            "specimen whose real activations tolerate the representation BEST and 4.125 to "
            "the one that tolerates it WORST, Spearman(floor_bpw, real_err) = -1.000. "
            "SHARPENED BY A SECOND RECEIPT, and the sharpening is CONFOUND-FREE where the "
            "original was not: moving each specimen from ONE FLAT SETTING to ITS OWN "
            "weight-space floor -- same weights, same activations, only the setting moves, "
            "so no cross-architecture comparison is involved -- changes real-activation "
            "error MONOTONICALLY BY RANK. The best improves 14.21%, the second is exactly "
            "unchanged, the third worsens 10.43%, the fourth worsens 9.63%. THE PROCEDURE "
            "IMPROVES THE ALREADY-GOOD AND DEGRADES THE ALREADY-BAD, and the spread it is "
            "meant to equalise comes out 1.33x WIDER than using one flat setting for "
            "everybody (0.043958 against 0.032994). NO CORRECTED BUDGET FOLLOWS: the "
            "re-floored ordering is bar-independent only at the COARSE grouping, and the "
            "Kimi/Falcon pair swaps at 2 of 5 bars."),
        applicability={
            "MODEL": UNSCOPED, "ARCHITECTURE": UNSCOPED,
            "ORGAN": "attention q_proj; the expert gate does NOT invert and is the contrast",
            "REPRESENTATION": "grouped absmax bits=4 group=64, exactly ws_rtn_q4_g64",
            "SHAPE": "layer-0 q_proj, [4096,2048] / [3072,2048] / [1536,3072]",
            "MACHINE": M3, "RUNTIME": "CPython 3.12, mlx_lm weights + numpy float32 reference",
            "KERNEL": NONE, "STORAGE_TIER": "TIER 2 /Volumes/corpdrive model lake",
            "TOPOLOGY": NONE,
            "WORKLOAD_PHASE": "representation fidelity against REAL activations, decode position 0"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json",
                         "receipts/headless/ACCELERATOR_REFLOOR_ON_ACTIVATIONS.json"],
        citations=["receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json#result.THE_ANSWER_IS_ORGAN_DEPENDENT",
                   "receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json#result.THE_BIT_BUDGET_IS_INVERTED_ACROSS_SPECIMENS",
                   "receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json#claim_boundary",
                   "receipts/headless/ACCELERATOR_REFLOOR_ON_ACTIVATIONS.json#result.THE_CONFOUND_FREE_HEADLINE",
                   "receipts/headless/ACCELERATOR_REFLOOR_ON_ACTIVATIONS.json#result.P2_REFUTED_IN_PART_AND_I_DO_NOT_GET_TO_QUOTE_A_BUDGET"],
        unscoped_basis={
            "MODEL": "receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json#result.q_proj_rows",
            "ARCHITECTURE": "receipts/headless/ACCELERATOR_ACTIVATION_VS_WEIGHT_SPACE.json#result.architectures_measured"},
        status="ACTIVE", superseded_by=None, negative_result=True,
        confidence_basis=(
            "Four specimens are THREE independent architecture groups -- the Qwen pair differs "
            "at the 4th decimal -- so architecture, shape and cosine are NOT separated, and the "
            "receipt names that confound rather than burying it. What raises this above one "
            "ordering flipping on noise is that the inversion is monotone on all four points, "
            "holds within the twin pair, and reproduces across four independent token sets. The "
            "contrast case is what makes it a boundary rather than a blanket claim: the expert "
            "gate does not invert. Its mechanism is already in this base as "
            "AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER -- within-group outliers concentrated in q_proj "
            "at Pearson -0.9841 -- so this is that weight-space law's activation-space "
            "consequence, not an unexplained correlation. THE SECOND RECEIPT REMOVES THE "
            "CONFOUND FROM THE CORE CLAIM: its within-specimen deltas hold one specimen's "
            "weights and activations fixed and move only the setting, so the monotone-by-rank "
            "result needs no cross-architecture comparison at all. The SPREAD figure and the "
            "bar-based reflooring still do, which is why the budget is NOT quoted."),
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
            "resolving power it does not have. EXTENDED 2026-08-26 by "
            "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json, which "
            "amends receipts/headless/ACCELERATOR_BARRIER_WINDOW.json: loudness is not "
            "transferable BETWEEN controls and it is not even STABLE FOR ONE CONTROL. A "
            "barrier stripped from gravity_native's serial reduction tail -- classified "
            "LOUD by the shipped prior, no upstream barrier -- fired 8 of 8 in one process "
            "and 4 of 60 = 0.067 under ground truth in another, with the intact kernel "
            "wrong 0 of 60. So the asymmetry that receipt encoded, that the loud case is a "
            "reliable prediction while the quiet one is not, DOES NOT SURVIVE, and a third "
            "window-closing mechanism is named: THE READING THREAD BEING THE SLOWEST ONE, "
            "a property of the work distribution across lanes that a prior reading barrier "
            "structure cannot see."),
        applicability={
            "MODEL": NONE, "ARCHITECTURE": NONE, "ORGAN": NONE,
            "REPRESENTATION": "dense_f32",
            "SHAPE": "AirTopKSample rows 32 cols 4096; AirNorm rms; widths 32-1024",
            "MACHINE": M3, "RUNTIME": MLX,
            "KERNEL": "AirNorm (loud control) and AirTopKSample (quiet control)",
            "STORAGE_TIER": NONE, "TOPOLOGY": NONE, "WORKLOAD_PHASE": "correctness grading"},
        evidence_class="Measured",
        source_receipts=["receipts/headless/ACCELERATOR_QUIET_CONTROL.json",
                         "receipts/headless/ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json"],
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
    # A hardware refusal and a precision measurement of its workaround; the
    # AKB's axes are about kernel performance and this is about a TYPE.
    "ACCELERATOR_FP64_IS_A_HARDWARE_REFUSAL.json": "CAPABILITY_NOT_LAW",
    # A NEGATIVE capability probe with a named instrument gate; nothing timed.
    "ACCELERATOR_OCCUPANCY_PROBE.json": "CAPABILITY_NOT_LAW",
    # Records that the pinned CUDA corpus is absent from this machine and that the
    # census could not say so. A fact about EVIDENCE AVAILABILITY and an instrument
    # refusal; there is no measured relation to type and nothing is timed.
    "ACCELERATOR_C2M_CORPUS_ABSENT.json": "CAPABILITY_NOT_LAW",
    # Adds a BUDGETED SWEEP to the fabric and the reporting discipline that stops a
    # skipped copy reading as a clean one. Bookkeeping over a mock provider with
    # nothing timed; the only number in it is an ESTIMATE from a rate measured in
    # ACCELERATOR_HUMF_IDENTITY, which this base already carries.
    "ACCELERATOR_HUMF_SCRUB.json": "CAPABILITY_NOT_LAW",
    # Records that all four Odyssey specimens now EXECUTE ON REAL LAKE WEIGHTS
    # once the storage gate opened. Its two measured numbers are a storage bus
    # rate and an mmap page-in rate, neither of which is a kernel law on this
    # base's axes (primitive x shape x representation x machine). The finding
    # that a 51.5% load still looks peaked is an INSTRUMENT caveat, and this
    # base already refuses to type those as laws (PROSE_ONLY exists for it).
    "ACCELERATOR_REAL_LAKE_WEIGHTS.json": "CAPABILITY_NOT_LAW",
    # An IR capability and a legality rule; the physical law it rests on
    # (lockstep at simd width 32) is already carried by the barrier-scopes work.
    "ACCELERATOR_SIMDGROUP_SCOPE_EXECUTES.json": "CAPABILITY_NOT_LAW",
    # A trust-model result over a MOCK fabric; no physical relation is measured.
    "ACCELERATOR_HUMF_CONSISTENT_CORRUPTION.json": "MOCK_NOT_PHYSICAL",
    # An integration result and six defect fixes; no measured physical relation.
    "HCLI_RESIDENT_END_TO_END.json": "CAPABILITY_NOT_LAW",
    # A coverage census over receipt SCHEMA, not a physical relation.
    "ACCELERATOR_BENCH_STATE_IS_A_SCHEMA_REQUIREMENT.json": "METHOD_NOT_LAW",
    # An endpoint that now EXISTS and refuses correctly; no measured relation.
    "ACCELERATOR_SEALED_RESIDENT_ENDPOINT.json": "CAPABILITY_NOT_LAW",
    # A path-and-permission repair, not a measured relation about execution.
    "ACCELERATOR_HCLI_VERIFIER_PATH_SEMANTICS.json": "CAPABILITY_NOT_LAW",
    # Newly VISIBLE via akb_registration (S032 §13) rather than via a filename.
    # Each carries a real result and each is superseded or subsumed by something
    # already in LAWS, so being visible earns a REASON rather than an entry.
    "TOKEN_GRAPH_REDUCTION_TIMED.json": "SUPERSEDED_IN_FLIGHT",
    "CAPABILITY_FUSED_GRAPH_CLEARED.json": "SUPERSEDED_IN_FLIGHT",
    "HCLI_RESIDENT_SEAL.json": "SUBSUMED",
    # an instrument, not a measured relation
    "ACCELERATOR_QUIESCENCE_INSTRUMENT.json": "PROSE_ONLY",
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
        "outside_scope_count": len(outside_scope(root)),
        "known_accelerator_outside_scope": [
            n for n in KNOWN_ACCELERATOR_OUTSIDE_SCOPE
            if n in set(outside_scope(root))],
        "scope_is_a_filename_prefix": (
            "membership is decided by the ACCELERATOR_* glob, so a receipt named "
            "otherwise is neither extracted nor refused -- it is invisible. The "
            "excluded names are listed rather than silently dropped."),
        "receipts_yielding_laws": len(cited),
        "membership_routes": membership_routes(root),
        "none_claims_not_grounded_count": sum(
            len(e.get("none_claims_not_grounded", [])) for e in entries),
        "none_claims_not_grounded_note": (
            "a NONE on an identity-backed axis that could not be checked because its "
            "source receipt predates the identity schema. Reported, not refused: these "
            "are unverified breadth claims, and an unreported one reads identically to "
            "a verified one."),
        "entries": entries,
        "entries_by_status": by_status,
        "negative_results": sum(1 for e in entries if e["negative_result"]),
        "supersession_in_corpus": {k: v for k, v in sorted(superseded.items())},
        # Which laws an amendment explicitly declared it does NOT reach, by law_id.
        # Published so the exemption is auditable from the artifact rather than
        # living only in a module global.
        "amendment_exemptions": {k: sorted(v) for k, v in sorted(_EXEMPT.items())},
        # Kept SEPARATE from the exemptions on purpose: "this amendment does not
        # reach that law" and "it reaches it and the law already carries the
        # correction" are different statuses, and one list would let a reader
        # take a reconciled law for an untouched one.
        "amendment_reconciliations": {k: sorted(v) for k, v in sorted(_RECONCILED.items())},
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
