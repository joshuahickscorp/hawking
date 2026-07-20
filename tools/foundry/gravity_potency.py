#!/usr/bin/env python3.12
"""Gravity Potency Ratchet: immutable method generations, potency vector, laws.

The ratchet exists so that each finished parent makes the NEXT parent start harder,
never softer. It is deliberately made of four refusals:

  1. a method generation is immutable once sealed, and only a new parent completion
     with a sealed evidence review can promote the next generation;
  2. the potency vector is a VECTOR: this module refuses to collapse it into one
     number, because a single score hides which axis regressed;
  3. the no-senility law refuses a parent program that starts timid, and refuses a
     starting-rate raise justified only by parameter count;
  4. the aggressive-gravity ordering law refuses a rate raise while any lever at the
     current rate is unexhausted.

Everything here is byte plan bookkeeping. It launches nothing and reads no weights.
LAW (sealed into every generation): byte plan != capability. Only a real
parent-vs-packed forward (mean symmetric KL <= 0.10 AND next-token argmax agreement
>= 0.95) can select a frontier.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional

_HERE = Path(__file__).resolve().parent
_CONDENSE = _HERE.parent / "condense"
if str(_CONDENSE) not in sys.path:
    sys.path.insert(0, str(_CONDENSE))

from eco_common import (  # noqa: E402
    EcoError, atomic_write_json, hash_value, is_sha256, now_iso, read_json_safe,
    seal_field, sealed,
)

SCHEMA_REGISTRY = "hawking.foundry.gravity_method_registry.v1"
SCHEMA_POTENCY_ROW = "hawking.foundry.gravity_potency_row.v1"
SCHEMA_ATLAS = "hawking.foundry.negative_transfer_atlas.v1"
SCHEMA_REVIEW = "hawking.foundry.evidence_review.v1"

REGISTRY_NAME = "GRAVITY_METHOD_REGISTRY.json"
LEDGER_NAME = "GRAVITY_POTENCY_LEDGER.jsonl"
ATLAS_NAME = "NEGATIVE_TRANSFER_ATLAS.json"

# Quality contract. Never weakened after a failure; a generation that wants different
# numbers must argue them in its own sealed review, and tightening only.
MAX_MEAN_SYMMETRIC_KL = Fraction(1, 10)
MIN_ARGMAX_AGREEMENT = Fraction(19, 20)
# 88 calibration tokens is not evidence: median routing split 63.6 percent stable,
# 26.1 percent of cells never route.
MIN_CAPABILITY_TOKENS = 1000

# Exact total-bit identities. A rounded decimal is not an identity.
RATE_LADDER: tuple[Fraction, ...] = (
    Fraction(1, 1), Fraction(17, 20), Fraction(7, 10), Fraction(3, 5),
    Fraction(1, 2), Fraction(2, 5), Fraction(1, 3), Fraction(1, 4),
)
QUALITY_ANCHOR_MIN = Fraction(17, 20)   # "near 1 BPW"
HIGH_SUBBIT_MAX = Fraction(1, 3)        # "high sub-bit challenger"

# Aggressive-gravity ordering: exhaust all of these AT a rate before raising it.
LEVER_ORDER: tuple[str, ...] = (
    "representation",
    "organ_allocation",
    "sharing_scope",
    "subvector_dimension",
    "protected_islands",
    "doctor_within_budget",
    "routing_aware_allocation",
)


class PotencyError(EcoError):
    """Fail-closed ratchet error."""


# ── paths ─────────────────────────────────────────────────────────────────────────────
def foundry_dir() -> Path:
    return Path(os.environ.get("HAWKING_FOUNDRY_DIR", str(_HERE)))


def registry_path() -> Path:
    return foundry_dir() / REGISTRY_NAME


def ledger_path() -> Path:
    return foundry_dir() / LEDGER_NAME


def atlas_path() -> Path:
    return foundry_dir() / ATLAS_NAME


# ── rate identities ───────────────────────────────────────────────────────────────────
def rate_identity(q: Fraction) -> dict[str, Any]:
    q = Fraction(q)
    return {"num": q.numerator, "den": q.denominator,
            "label": f"{q.numerator}/{q.denominator}", "value": float(q)}


def parse_rate(text: Any) -> Fraction:
    """Exact rational only. '0.85' is refused; pass '17/20'."""
    if isinstance(text, Fraction):
        return text
    if isinstance(text, dict) and "num" in text and "den" in text:
        return Fraction(int(text["num"]), int(text["den"]))
    s = str(text).strip()
    if "/" in s:
        n, d = s.split("/", 1)
        return Fraction(int(n), int(d))
    if s.lstrip("-").isdigit():
        return Fraction(int(s), 1)
    raise PotencyError(f"rate must be an exact rational 'n/d', not {text!r}")


def on_ladder(q: Fraction) -> bool:
    return Fraction(q) in RATE_LADDER


def capability_pass(mean_symmetric_kl: float, argmax_agreement: float) -> bool:
    """The only gate that may select a frontier. Byte plan != capability."""
    return (Fraction(str(mean_symmetric_kl)) <= MAX_MEAN_SYMMETRIC_KL
            and Fraction(str(argmax_agreement)) >= MIN_ARGMAX_AGREEMENT)


# ── 1. immutable method generations ───────────────────────────────────────────────────
def _generation_key(n: int) -> str:
    return f"GRAVITY_METHOD_V{int(n)}"


def _v1_generation() -> dict[str, Any]:
    """Sealed from the F0 (GPT-OSS-120B) + F1 (Qwen3-235B) evidence."""
    gen = {
        "generation": 1,
        "key": _generation_key(1),
        "method_version": "GRAVITY_METHOD_V1",
        "sealed_at": now_iso(),
        "source_revision": {
            "parent": "openai/gpt-oss-120b",
            "status": "released_after_harvest",
            "note": "re-downloadable from the pinned revision; harvest completed before release",
            "pinned": True,
        },
        "provider_revision": {
            "foundry": "hawking.deep-architecture-foundry",
            "engine": "tools/condense/gravity_forge.py + gptoss_subbit_packer.py",
            "forward": "tools/condense/gptoss_real_forward.py",
        },
        "candidate_priors": {
            "organ_inversion": {
                "claim": "mlp1 (gate+up) is the SENSITIVE organ; mlp2 (down) tolerates more.",
                "evidence": "F0 gpt-oss-120b real forward; F1 qwen3-235b dominant_failure_organ=gate",
                "action": "allocate bits to gate/up first, spend the slack on down",
            },
            "routing_frequency_allocation": {
                "claim": "allocate rate by measured expert routing frequency",
                "status": "alive",
                "requirement": "~1000 calibration tokens; 88 is NOT enough "
                               "(median split 63.6 percent stable, 26.1 percent of cells never route)",
            },
            "space_filling_gain": {
                "claim": "rel_error falls monotonically with VQ dimension at fixed rate",
                "measured": {"d8k16": 0.782, "d16k256": 0.757, "d32k65536": 0.668},
                "status": "alive",
            },
            "row_norm_stratification": {
                "claim": "94 percent of gate/up rows collapse onto ONE codeword "
                         "because row norms span 1e-5..0.91; stratify by row norm",
                "status": "alive_untested",
            },
            "honest_boundary_f0": {
                "claim": "sub-bit uniform AND treated both collapsed on a real forward",
                "status": "sealed_negative_result",
            },
        },
        "quality_contract": {
            "law": "byte plan != capability; only a real parent-vs-packed forward selects a frontier",
            "mean_symmetric_kl_max": str(MAX_MEAN_SYMMETRIC_KL),
            "next_token_argmax_agreement_min": str(MIN_ARGMAX_AGREEMENT),
            "min_capability_tokens": MIN_CAPABILITY_TOKENS,
            "weakening_forbidden": True,
            "enforcer": "tools/foundry/quality_contract.py (same thresholds)",
        },
        "kernel_set": {
            "representation": ["pq_vq", "trellis_free_scalar"],
            "vq_dimensions": [8, 16, 32],
            "organ_split": ["mlp1_gate_up", "mlp2_down", "attn", "embed_head_passthru"],
            "doctor": ["doctor_static", "doctor_conditional"],
        },
        "storage_policy": {
            "expert_cache_cap_bytes": 20 * 1024 ** 3,
            "rationale": "a single lockstep pass has ZERO cross-layer reuse; a 64GB cap gave 0 "
                         "evictions, drove RAM 70->18GB and swap to 906MB free. Aggressive RAM "
                         "only where real reuse exists.",
            "source_release": "release source after harvest; re-download from pinned revision",
        },
        "rate_ladder": [rate_identity(r) for r in RATE_LADDER],
        "promoted_by_review_sha256": None,
        "parents_completed": ["gpt-oss-120b:F0"],
        "parents_in_flight": ["qwen3-235b:F1"],
    }
    return seal_field(gen, "sha256")


def seal_v1(*, overwrite: bool = False) -> dict[str, Any]:
    """Write the registry with V1 sealed. Refuses to overwrite an existing registry."""
    path = registry_path()
    if path.exists() and not overwrite:
        return load_registry()
    doc = {"schema": SCHEMA_REGISTRY, "created_at": now_iso(),
           "generations": {_generation_key(1): _v1_generation()}}
    atomic_write_json(path, doc)
    return doc


def load_registry() -> dict[str, Any]:
    """Read + verify. A mutated sealed generation is a hard failure."""
    doc = read_json_safe(registry_path())
    if doc.get("schema") != SCHEMA_REGISTRY:
        raise PotencyError(f"not a gravity method registry: {doc.get('schema')!r}")
    for key, gen in sorted(doc.get("generations", {}).items()):
        if not sealed(gen, "sha256"):
            raise PotencyError(f"{key} is mutated or unsealed: sha256 does not match its body")
    return doc


def latest_generation(doc: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    doc = doc or load_registry()
    gens = doc.get("generations", {})
    if not gens:
        raise PotencyError("registry holds no generations")
    return max(gens.values(), key=lambda g: int(g["generation"]))


def _review_failures(review: Any) -> list[str]:
    bad: list[str] = []
    if not isinstance(review, dict):
        return ["evidence review is missing (promotion requires a sealed review artifact)"]
    if review.get("schema") != SCHEMA_REVIEW:
        bad.append(f"review schema must be {SCHEMA_REVIEW}, got {review.get('schema')!r}")
    if not sealed(review, "sha256"):
        bad.append("review is not sealed (sha256 does not match its body)")
    for field in ("parent_id", "reviewer", "verdict", "capability_receipt_sha256"):
        if not review.get(field):
            bad.append(f"review is missing {field}")
    if review.get("verdict") not in (None, "accept"):
        bad.append(f"review verdict is {review.get('verdict')!r}, not 'accept'")
    if review.get("capability_receipt_sha256") and not is_sha256(review["capability_receipt_sha256"]):
        bad.append("capability_receipt_sha256 is not a sha256")
    return bad


def promote(evidence: dict[str, Any]) -> dict[str, Any]:
    """Promote the next generation. REFUSES without a sealed evidence review.

    evidence: {"review": <sealed review artifact>, "generation": {...new bindings...}}
    A new parent completion is the only thing that can promote a generation, and the
    completion has to have survived review. Sealed generations are never rewritten.
    """
    failures = _review_failures((evidence or {}).get("review"))
    if failures:
        raise PotencyError("promotion refused: " + "; ".join(failures))
    review = evidence["review"]
    body = evidence.get("generation")
    if not isinstance(body, dict) or not body:
        raise PotencyError("promotion refused: no generation bindings supplied")

    doc = load_registry()
    prev = latest_generation(doc)
    n = int(prev["generation"]) + 1
    key = _generation_key(n)
    if key in doc["generations"]:
        raise PotencyError(f"promotion refused: {key} already sealed (generations are immutable)")

    contract = dict(prev["quality_contract"])
    contract.update(body.get("quality_contract", {}))
    if (Fraction(str(contract["mean_symmetric_kl_max"])) > MAX_MEAN_SYMMETRIC_KL
            or Fraction(str(contract["next_token_argmax_agreement_min"])) < MIN_ARGMAX_AGREEMENT
            or int(contract.get("min_capability_tokens", MIN_CAPABILITY_TOKENS)) < MIN_CAPABILITY_TOKENS):
        raise PotencyError("promotion refused: quality contract weakened; thresholds only tighten")

    gen = {k: v for k, v in prev.items() if k != "sha256"}
    gen.update(body)
    gen.update({
        "generation": n, "key": key, "method_version": key,
        "method_version_name": key, "sealed_at": now_iso(),
        "quality_contract": contract,
        "promoted_by_review_sha256": review["sha256"],
        "parents_completed": list(prev.get("parents_completed", [])) + [review["parent_id"]],
    })
    doc["generations"][key] = seal_field(gen, "sha256")
    atomic_write_json(registry_path(), doc)
    return doc["generations"][key]


# ── 2. potency vector ─────────────────────────────────────────────────────────────────
POTENCY_AXES: tuple[str, ...] = (
    "lowest_physical_bpw",           # smallest byte plan ever built
    "lowest_functional_bpw",         # smallest that still produced finite, non-degenerate logits
    "lowest_capability_passing_bpw",  # smallest that PASSED the real-forward contract
    "quality_at_fixed_bpw",          # {rate_label: {mean_symmetric_kl, argmax_agreement}}
    "doctor_recovery_per_bit",       # capability regained per bit of doctor reserve
    "runtime_seconds",
    "peak_ram_bytes",
    "source_bytes",
    "artifact_bytes",
    "time_to_frontier_seconds",
    "energy_joules",                 # None where not measurable
    "transfer_success_next_parent",  # None until the next parent runs
)


def potency_row(parent_id: str, *, note: str = "", **axes: Any) -> dict[str, Any]:
    unknown = set(axes) - set(POTENCY_AXES)
    if unknown:
        raise PotencyError(f"unknown potency axes: {sorted(unknown)}")
    row = {"schema": SCHEMA_POTENCY_ROW, "parent_id": str(parent_id),
           "recorded_at": now_iso(), "note": note,
           "method_version": latest_generation()["method_version"]}
    row.update({axis: axes.get(axis) for axis in POTENCY_AXES})
    return seal_field(row, "sha256")


def append_potency(row: dict[str, Any]) -> dict[str, Any]:
    """Append-only. Rows are never edited; a correction is a new row."""
    if not sealed(row, "sha256"):
        row = seal_field({k: v for k, v in row.items() if k != "sha256"}, "sha256")
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def read_potency(parent_id: Optional[str] = None) -> list[dict[str, Any]]:
    path = ledger_path()
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if parent_id is None or r.get("parent_id") == parent_id]


def collapse_to_score(*_args: Any, **_kwargs: Any) -> float:
    """Deliberately unimplemented. Potency is a vector."""
    raise PotencyError(
        "refused: potency is a vector, not a score. A single number hides which axis "
        "regressed (a lower BPW bought with a collapsed forward is not progress). "
        "Use report_potency() and read all axes."
    )


def report_potency(parent_id: Optional[str] = None) -> str:
    rows = read_potency(parent_id)
    if not rows:
        return "potency ledger: no rows"
    out: list[str] = []
    for row in rows:
        out.append(f"parent {row['parent_id']}  method {row.get('method_version')}  "
                   f"recorded {row.get('recorded_at')}")
        for axis in POTENCY_AXES:
            value = row.get(axis)
            out.append(f"  {axis:<32} {'unmeasured' if value is None else value}")
        out.append("  single_score                     REFUSED (potency is a vector)")
        if row.get("note"):
            out.append(f"  note: {row['note']}")
    return "\n".join(out)


# ── 3. no-senility law ────────────────────────────────────────────────────────────────
_SIZE_WORDS = ("larger", "bigger", "more param", "parameter count", "param count",
               "size", "scale", "huge", "giant", "235b", "685b", "1t")


def check_no_senility(program: dict[str, Any],
                      previous_parent_evidence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """A bigger model is not permission to start softer.

    program: {"parent_id", "rates": [rate identities], "start_rate", "start_rate_reason",
              "start_rate_evidence" (optional sha256 of a measurement)}
    previous_parent_evidence: {"parent_id", "lowest_credible_bpw", "start_rate"}
    """
    failures: list[str] = []
    raw = program.get("rates") or []
    if not raw:
        failures.append("program declares no rates")
    rates: list[Fraction] = []
    for item in raw:
        try:
            q = parse_rate(item)
        except PotencyError as exc:
            failures.append(str(exc))
            continue
        if not on_ladder(q):
            failures.append(f"rate {q.numerator}/{q.denominator} is not on the exact rate ladder")
        rates.append(q)

    if rates and not any(r >= QUALITY_ANCHOR_MIN for r in rates):
        failures.append(f"no quality anchor near 1 BPW (need a rate >= {QUALITY_ANCHOR_MIN})")
    if rates and not any(r <= HIGH_SUBBIT_MAX for r in rates):
        failures.append(f"no high sub-bit challenger (need a rate <= {HIGH_SUBBIT_MAX})")

    prev = previous_parent_evidence or {}
    if prev:
        if prev.get("lowest_credible_bpw") is None:
            raise PotencyError("previous_parent_evidence needs lowest_credible_bpw")
        prev_low = parse_rate(prev["lowest_credible_bpw"])
        if not any(r <= prev_low for r in rates):
            failures.append(
                f"does not cover the previous parent's lowest credible region "
                f"({prev_low.numerator}/{prev_low.denominator} from {prev.get('parent_id')})")
        if not any(r < prev_low for r in rates):
            failures.append(
                f"no lower-rate stress point below the previous parent's "
                f"{prev_low.numerator}/{prev_low.denominator}")
        if prev.get("start_rate") is not None and program.get("start_rate") is not None:
            start = parse_rate(program["start_rate"])
            prev_start = parse_rate(prev["start_rate"])
            if start > prev_start:
                reason = str(program.get("start_rate_reason") or "")
                lowered = reason.lower()
                if not is_sha256(program.get("start_rate_evidence")):
                    failures.append(
                        f"starting rate raised {prev_start} -> {start} without a measured "
                        f"start_rate_evidence sha256")
                elif any(word in lowered for word in _SIZE_WORDS):
                    failures.append(
                        f"starting rate raised {prev_start} -> {start} on a size argument "
                        f"({reason!r}); parameter count is not permission to start timid")
    return {"ok": not failures, "failures": failures,
            "rates": [rate_identity(r) for r in sorted(set(rates), reverse=True)]}


# ── 4. aggressive-gravity ordering ────────────────────────────────────────────────────
def check_rate_discipline(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """history: ordered attempts, each {"rate": identity, "lever": name, "exhausted": bool}.

    A rate may only be raised once every lever in LEVER_ORDER is exhausted at the
    current rate, and levers are worked in canonical order.
    """
    failures: list[str] = []
    done: dict[Fraction, set[str]] = {}
    current: Optional[Fraction] = None
    for i, step in enumerate(history):
        rate = parse_rate(step["rate"])
        lever = step.get("lever")
        if lever not in LEVER_ORDER:
            failures.append(f"step {i}: unknown lever {lever!r}")
            continue
        if current is not None and rate > current:
            missing = [x for x in LEVER_ORDER if x not in done.get(current, set())]
            if missing:
                failures.append(
                    f"step {i}: rate raised {current} -> {rate} with levers unexhausted at "
                    f"{current}: {missing}")
        seen = done.setdefault(rate, set())
        earlier = [x for x in LEVER_ORDER[:LEVER_ORDER.index(lever)] if x not in seen]
        if earlier:
            failures.append(f"step {i}: lever {lever!r} attempted at {rate} before {earlier}")
        if step.get("exhausted", True):
            seen.add(lever)
        current = rate
    remaining = ([x for x in LEVER_ORDER if x not in done.get(current, set())]
                 if current is not None else list(LEVER_ORDER))
    return {"ok": not failures, "failures": failures,
            "current_rate": rate_identity(current) if current is not None else None,
            "next_lever": remaining[0] if remaining else None,
            "may_raise_rate": current is not None and not remaining}


# ── 5. negative transfer atlas ────────────────────────────────────────────────────────
def _atlas_seed() -> dict[str, Any]:
    entries = {
        "inter_expert_redundancy": {
            "lever": "delta coding / shared low-rank bases / cluster-mean subtraction across experts",
            "killed_by": "mean pairwise cosine between experts = 1e-4; experts are mutually orthogonal",
            "parent": "gpt-oss-120b:F0",
            "verdict": "dead on arrival: there is no shared component to subtract",
            "reopen_condition": "a future parent measures mean pairwise expert cosine >= 0.10 "
                                "on its own weights",
        },
        "entropy_coded_pq_indices": {
            "lever": "entropy coding of trained PQ indices",
            "killed_by": "measured 0.0 to 0.7 percent, not the hoped 10 to 25 percent; "
                         "Lloyd-optimal indices are near-uniform by construction",
            "parent": "gpt-oss-120b:F0",
            "verdict": "the gain is inside the noise of the byte plan",
            "reopen_condition": "a future parent uses NON-Lloyd (biased or stratified) codebooks "
                                "whose measured index entropy is <= 0.9 of uniform",
        },
        "posthoc_scalar_gain": {
            "lever": "post-hoc scalar gain correction on a PQ artifact",
            "killed_by": "optimal gain pinned at exactly 1.0; k-means reconstruction is a "
                         "conditional mean so the residual is orthogonal to the reconstruction, "
                         "and cosine is gain-invariant",
            "parent": "gpt-oss-120b:F0",
            "verdict": "algebraically pinned, not merely unhelpful",
            "reopen_condition": "a future parent uses a NON-conditional-mean quantizer whose "
                                "residual is measurably non-orthogonal to the reconstruction",
        },
        "ternary_factorization": {
            "lever": "ternary factorization instead of VQ",
            "killed_by": "loses to VQ at matched rate",
            "parent": "gpt-oss-120b:F0",
            "verdict": "dominated at every matched rate tested",
            "reopen_condition": "a future parent diagnoses a weight distribution where ternary "
                                "beats VQ at a matched exact rate on a real forward",
        },
        "large_expert_cache": {
            "lever": "aggressive expert cache (64GB cap) for a single lockstep pass",
            "killed_by": "0 evictions at a 64GB cap; RAM 70->18GB free and swap down to 906MB free. "
                         "A single lockstep pass has zero cross-layer reuse",
            "parent": "gpt-oss-120b:F0",
            "verdict": "correct cap is ~20GB; aggressive RAM only where real reuse exists",
            "reopen_condition": "a future parent's schedule has MEASURED cross-layer expert reuse "
                                "(cache hit rate > 0) instead of a single lockstep pass",
        },
        "uniform_subbit_allocation": {
            "lever": "uniform sub-bit allocation across organs (and its treated variant)",
            "killed_by": "F0 real forward: both the uniform and the treated artifact collapsed",
            "parent": "gpt-oss-120b:F0",
            "verdict": "superseded by organ inversion (mlp1 gate+up sensitive, mlp2 down tolerant), "
                       "confirmed by F1 qwen3-235b dominant_failure_organ = gate",
            "reopen_condition": "a future parent measures NO organ sensitivity split "
                                "(gate and down degrade within 10 percent of each other)",
        },
        "calibration_88_tokens": {
            "lever": "routing-frequency allocation calibrated on ~88 tokens",
            "killed_by": "at 88 tokens the median routing split is only 63.6 percent stable and "
                         "26.1 percent of cells never route at all",
            "parent": "gpt-oss-120b:F0",
            "verdict": "the lever is alive, the 88-token calibration is dead; needs ~1000 tokens",
            "reopen_condition": "never at 88; use >= 1000 calibration tokens",
        },
    }
    doc = {"schema": SCHEMA_ATLAS, "created_at": now_iso(), "entries": entries}
    return seal_field(doc, "sha256")


def seal_atlas(*, overwrite: bool = False) -> dict[str, Any]:
    path = atlas_path()
    if path.exists() and not overwrite:
        return load_atlas()
    atomic_write_json(path, _atlas_seed())
    return load_atlas()


def load_atlas() -> dict[str, Any]:
    doc = read_json_safe(atlas_path())
    if doc.get("schema") != SCHEMA_ATLAS:
        raise PotencyError(f"not a negative transfer atlas: {doc.get('schema')!r}")
    return doc


def atlas_check(lever_id: str, diagnosis: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Blocked unless a NEW parent's diagnosis reopens the lever.

    diagnosis: {"parent_id": ..., "reopens": [lever_id, ...], "measurement": "..."}
    The reopening parent must differ from the parent that killed the lever, and must
    cite the measurement that changes the verdict.
    """
    entry = load_atlas()["entries"].get(lever_id)
    if entry is None:
        return {"lever": lever_id, "blocked": False, "reason": "not in the atlas"}
    d = diagnosis or {}
    reopened = (lever_id in (d.get("reopens") or [])
                and bool(d.get("measurement"))
                and d.get("parent_id") not in (None, entry["parent"]))
    return {"lever": lever_id, "blocked": not reopened,
            "killed_by": entry["killed_by"], "verdict": entry["verdict"],
            "reopen_condition": entry["reopen_condition"],
            "reopened_by": d.get("parent_id") if reopened else None}


# ── selftest / CLI ────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    checks = {
        "rate_ladder_exact": all(isinstance(r, Fraction) for r in RATE_LADDER),
        "rounded_rate_refused": _raises(lambda: parse_rate("0.85")),
        "score_refused": _raises(collapse_to_score),
        "v1_seals": sealed(_v1_generation(), "sha256"),
        "atlas_seals": sealed(_atlas_seed(), "sha256"),
        "contract": capability_pass(0.05, 0.97) and not capability_pass(0.2, 0.97),
    }
    return {"ok": all(checks.values()), "checks": checks}


def _raises(fn) -> bool:
    try:
        fn()
    except PotencyError:
        return True
    return False


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "seal":
        seal_v1()
        seal_atlas()
        print(f"registry: {registry_path()}\natlas:    {atlas_path()}")
    elif cmd == "report":
        print(report_potency(argv[2] if len(argv) > 2 else None))
    elif cmd == "generation":
        print(json.dumps(latest_generation(), indent=2, sort_keys=True))
    elif cmd == "selftest":
        print(json.dumps(selftest(), indent=2, sort_keys=True))
    elif cmd == "status":
        gen = latest_generation()
        print(f"method {gen['method_version']} sha256 {gen['sha256'][:16]} "
              f"parents {gen['parents_completed']}")
        print(f"potency rows: {len(read_potency())}")
        print(f"atlas entries: {len(load_atlas()['entries'])}")
    else:
        print(f"usage: {argv[0]} [seal|status|report [parent]|generation|selftest]")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
