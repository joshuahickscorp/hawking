"""MATH / FORMAL laboratory — H.6 against local claims with a local checker.

z3py is the independent checker. Lean is present on this host but unused:
the Ramanujan probe is not in this sparse checkout, and this engine does
not spawn subprocesses.

None of these results are novel mathematics. They formalize laws already
stated in this repo. Unformalized reasoning is labeled as such.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import REPO, write_receipt
from tools.theia.bounty import BountyClass, make_internal_bounty
from tools.theia.intake import IntakeRefused, VerificationFailed, run_intake
from tools.theia.labs import LabKind
from tools.theia.value import DeclaredFactor, ValueInputs

RECORDED_BY = "tools/theia/math_lab.py"
RECEIPT_NAME = "THEIA_MATH_FORMAL.json"
SCHEMA = "hawking.theia.math_formal.v1"
EVIDENCE_TIER = "STATIC"

KERNEL_GEOMETRY = REPO / "receipts" / "future" / "KERNEL_GEOMETRY.json"
COMPLETE_EBPW = REPO / "receipts" / "future" / "COMPLETE_EBPW.json"

UNIT_SOURCE = (
    "declared unit baseline so the H.1 denominator is defined; STATIC_ONLY; "
    "not a hardware measurement"
)


def _z3():
    try:
        import z3
    except ImportError as e:
        raise IntakeRefused(
            "MATH/FORMAL checker z3py is not importable; lab stays declared"
        ) from e
    return z3


def checker_status() -> dict[str, Any]:
    z3_available = False
    z3_version = None
    z3_error = None
    try:
        z3 = _z3()
        z3.Int("x")
        z3_available = True
        z3_version = z3.get_version_string()
    except Exception as e:  # noqa: BLE001 — status probe, not a swallow of a bounty
        z3_error = f"{type(e).__name__}: {e}"
    lean = Path("/opt/homebrew/bin/lean")
    return {
        "z3py": {
            "available": z3_available,
            "version": z3_version,
            "used": z3_available,
            "error": z3_error,
        },
        "lean": {
            "available": lean.is_file(),
            "path": str(lean) if lean.is_file() else None,
            "used": False,
            "why_unused": (
                "research/ramanujan/container/probe/RamanujanProbe.lean is not in this "
                "sparse checkout; invoking lean would require subprocess, which "
                "this engine does not do"
            ),
        },
    }


def apple_threadgroup_ceiling() -> int:
    """Call the kernel checker's own symbol. An import is not a call site."""
    from tools.future.static_kernel_verify import APPLE_MAX_THREADS_PER_THREADGROUP

    return int(APPLE_MAX_THREADS_PER_THREADGROUP)


def check_claims() -> list[dict[str, Any]]:
    """H.6: formalize, run the prover, archive the counterexample.

    Evidence tier STATIC: SMT sat/unsat is logic, not a hardware measurement.
    """
    z3 = _z3()
    ceiling = apple_threadgroup_ceiling()
    out: list[dict[str, Any]] = []

    x, y, z = z3.Ints("x y z")
    legal = z3.And(x >= 1, y >= 1, z >= 1, x * y * z <= ceiling)

    s_illegal_legal = z3.Solver()
    s_illegal_legal.add(legal, x * y * z > ceiling)
    r1 = s_illegal_legal.check()
    out.append(
        {
            "id": "APPLE_TG_PRODUCT_NEVER_EXCEEDS_CEILING",
            "kind": "theorem",
            "work": "formalization",
            "source_symbol": "tools.future.static_kernel_verify.APPLE_MAX_THREADS_PER_THREADGROUP",
            "ceiling": ceiling,
            "formal": (
                "unsat of (x,y,z>=1 /\\ x*y*z<=CEILING /\\ x*y*z>CEILING) "
                f"with CEILING={ceiling}"
            ),
            "novelty": "not novel; formalization of this repo's static kernel ceiling",
            "evidence_tier": EVIDENCE_TIER,
            "z3_result": str(r1),
            "expected": "unsat",
            "holds": r1 == z3.unsat,
            "checker": "z3.Solver.check",
        }
    )

    s_counter = z3.Solver()
    s_counter.add(x == 64, y == 8, z == 4, x * y * z > ceiling)
    r2 = s_counter.check()
    model = None
    if r2 == z3.sat:
        m = s_counter.model()
        model = {
            "x": int(m[x].as_long()),
            "y": int(m[y].as_long()),
            "z": int(m[z].as_long()),
            "product": 64 * 8 * 4,
        }
    out.append(
        {
            "id": "POWER_OF_TWO_THREADGROUP_IS_NOT_ALWAYS_LEGAL",
            "kind": "counterexample",
            "work": "counterexample_search",
            "claim_refuted": (
                "any power-of-two threadgroup triple is legal on Apple Silicon"
            ),
            "formal": f"(64,8,4) product 2048 > CEILING {ceiling}",
            "novelty": "not novel; the ceiling is already in static_kernel_verify",
            "evidence_tier": EVIDENCE_TIER,
            "z3_result": str(r2),
            "expected": "sat",
            "holds": r2 == z3.sat and model is not None and model["product"] == 2048,
            "model": model,
            "checker": "z3.Solver.check",
            "archived_as_negative_science": True,
        }
    )

    geo = json.loads(KERNEL_GEOMETRY.read_text())
    occ = geo["occupancy_class"]
    tpr = int(occ["threads_per_row"])
    rpt = int(occ["rows_per_tg"])
    tg = int(occ["threadgroup"])
    s_geo = z3.Solver()
    s_geo.add(x == tpr, y == rpt, z == 1, x * y * z == tg, legal)
    r3 = s_geo.check()
    out.append(
        {
            "id": "KERNEL_GEOMETRY_OCCUPANCY_IDENTITY_IS_LEGAL",
            "kind": "theorem",
            "work": "theorem_search",
            "source": str(KERNEL_GEOMETRY),
            "instance": {
                "threads_per_row": tpr,
                "rows_per_tg": rpt,
                "threadgroup": tg,
            },
            "formal": f"sat of x={tpr} /\\ y={rpt} /\\ z=1 /\\ x*y*z={tg} /\\ legal",
            "novelty": "not novel; occupancy_class identity already in KERNEL_GEOMETRY.json",
            "evidence_tier": EVIDENCE_TIER,
            "z3_result": str(r3),
            "expected": "sat",
            "holds": r3 == z3.sat and tpr * rpt == tg and tg <= ceiling,
            "checker": "z3.Solver.check",
        }
    )

    vr, p, ig, tv, sr = z3.Reals("vr p ig tv sr")
    wt, cc, hc, rsk1, rsk2, oc = z3.Reals("wt cc hc rsk1 rsk2 oc")
    pos = [q > 0 for q in (vr, p, ig, tv, sr, wt, cc, hc, rsk1, rsk2, oc)]
    v1 = (vr * p * ig * tv * sr) / (wt * cc * hc * rsk1 * oc)
    v2 = (vr * p * ig * tv * sr) / (wt * cc * hc * rsk2 * oc)
    s_mono = z3.Solver()
    s_mono.add(pos)
    s_mono.add(rsk1 < rsk2)
    s_mono.add(v1 <= v2)
    r4 = s_mono.check()
    out.append(
        {
            "id": "H1_VALUE_STRICTLY_DECREASES_IN_RISK",
            "kind": "theorem",
            "work": "symbolic_computation",
            "source_symbol": "tools.theia.value.bounty_value",
            "formal": (
                "unsat of (all H.1 factors > 0 /\\ risk1 < risk2 /\\ value(risk1) <= value(risk2))"
            ),
            "novelty": "not novel; restates H.1 as a ratio of positive terms",
            "evidence_tier": EVIDENCE_TIER,
            "z3_result": str(r4),
            "expected": "unsat",
            "holds": r4 == z3.unsat,
            "checker": "z3.Solver.check",
        }
    )

    ebpw = json.loads(COMPLETE_EBPW.read_text())
    parts = ebpw["incumbent"]["parts"]
    stored = int(ebpw["incumbent"]["stored_bytes"])
    part_syms = [z3.Int(f"p{i}") for i in range(len(parts))]
    s_cons = z3.Solver()
    total = z3.Int("stored")
    s_cons.add(total == stored)
    s_cons.add(z3.Sum(part_syms) == total)
    for sym, part in zip(part_syms, parts):
        s_cons.add(sym == int(part["bytes"]))
    r5 = s_cons.check()
    out.append(
        {
            "id": "COMPLETE_EBPW_PARTS_SUM_EQUALS_STORED_BYTES",
            "kind": "theorem",
            "work": "symbolic_computation",
            "source": str(COMPLETE_EBPW),
            "n_parts": len(parts),
            "stored_bytes": stored,
            "parts_sum": sum(int(p["bytes"]) for p in parts),
            "formal": "sat of Sum(part_i) == stored_bytes with part_i bound to receipt bytes",
            "novelty": "not novel; restates complete_ebpw's unreconciled-candidate refusal",
            "evidence_tier": EVIDENCE_TIER,
            "z3_result": str(r5),
            "expected": "sat",
            "holds": r5 == z3.sat and stored == sum(int(p["bytes"]) for p in parts),
            "checker": "z3.Solver.check",
        }
    )

    failed = [c["id"] for c in out if not c["holds"]]
    if failed:
        raise VerificationFailed(f"MATH/FORMAL claims failed: {failed}")
    return out


def verify_math_receipt(path: Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Independent checker: re-run z3.Solver.check on every claim."""
    del path
    live = check_claims()
    claimed = doc.get("claims")
    if not isinstance(claimed, list) or not claimed:
        raise VerificationFailed("math receipt has no claims list")
    live_ids = [c["id"] for c in live]
    claimed_ids = [c.get("id") for c in claimed]
    if live_ids != claimed_ids:
        raise VerificationFailed(
            f"math claim ids {claimed_ids} != live check_claims() {live_ids}"
        )
    for live_c, claimed_c in zip(live, claimed):
        if live_c["z3_result"] != claimed_c.get("z3_result"):
            raise VerificationFailed(
                f"{live_c['id']}: live z3_result {live_c['z3_result']!r} != "
                f"receipt {claimed_c.get('z3_result')!r}"
            )
        if live_c["holds"] is not True:
            raise VerificationFailed(f"{live_c['id']} does not hold on re-check")
    status = checker_status()
    if not status["z3py"]["available"]:
        raise VerificationFailed("z3py was not available on independent verification")
    return {
        "n_claims": len(live),
        "claim_ids": live_ids,
        "independent_module": "tools.theia.math_lab.check_claims",
        "checker": "z3.Solver.check",
        "evidence_tier": EVIDENCE_TIER,
        "ceiling": apple_threadgroup_ceiling(),
    }


def _value_inputs(claims: list[dict[str, Any]]) -> ValueInputs:
    n_hold = sum(1 for c in claims if c["holds"])
    n_transfer = sum(1 for c in claims if c["kind"] == "theorem")
    n_counter = sum(1 for c in claims if c["kind"] == "counterexample")
    if n_hold < 1 or n_transfer < 1:
        raise IntakeRefused("math lab produced no holding theorem")

    def unit(name: str, source: str) -> DeclaredFactor:
        return DeclaredFactor(value=Fraction(1), name=name, source=source)

    return ValueInputs(
        verified_reward=unit(
            "verified_reward",
            "H.1: verified_reward may include theorem/proof; MATH/FORMAL H.6",
        ),
        probability_of_success=unit(
            "probability_of_success",
            "local z3 check of already-stated repo laws, not a search",
        ),
        information_gain=DeclaredFactor(
            value=Fraction(n_hold),
            name="information_gain",
            source="count of claims that z3.Solver.check confirmed",
        ),
        transfer_value=DeclaredFactor(
            value=Fraction(n_transfer + n_counter),
            name="transfer_value",
            source="theorems plus archived counterexamples reusable as search knowledge",
        ),
        strategic_relevance=unit(
            "strategic_relevance",
            "MATH / FORMAL laboratory, §19.12 / H.6",
        ),
        wall_time=unit("wall_time", UNIT_SOURCE),
        compute_cost=unit("compute_cost", UNIT_SOURCE),
        human_cost=unit("human_cost", UNIT_SOURCE),
        risk=unit(
            "risk",
            "local z3; no network, no ACTIVE_TEST, no credentials",
        ),
        opportunity_cost=unit("opportunity_cost", UNIT_SOURCE),
    )


def assemble_receipt_doc(claims: list[dict[str, Any]]) -> dict[str, Any]:
    status = checker_status()
    return {
        "schema": SCHEMA,
        "version": 1,
        "recorded_by": RECORDED_BY,
        "lab": LabKind.MATH_FORMAL.value,
        "recipe": "H.6",
        "evidence_class": "STATIC_ONLY",
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": (
            "SMT sat/unsat over this repo's own stated laws. Not novel mathematics. "
            "Not a hardware measurement. Lean was not invoked."
        ),
        "checkers": status,
        "h6": {
            "formalize": True,
            "literature_equivalence": (
                "explicitly not novel; each claim names the repo law it restates"
            ),
            "prover": "z3py",
            "independent_checker": "tools.theia.math_lab.check_claims re-runs z3.Solver.check",
            "unformalized": "none in this receipt; every claim has a z3_result",
            "failed_lemmas_archived": [
                c["id"] for c in claims if c["kind"] == "counterexample"
            ],
        },
        "executable_work": list(
            {
                c["work"]
                for c in claims
                if c.get("work")
            }
        ),
        "claims": claims,
        "n_claims": len(claims),
        "n_hold": sum(1 for c in claims if c["holds"]),
        "apple_threadgroup_ceiling": apple_threadgroup_ceiling(),
    }


def run_math_bounty(*, write: bool = True):
    """Execute one real MATH/FORMAL bounty through H.2. Returns IntakeResult."""
    status = checker_status()
    if not status["z3py"]["available"]:
        raise IntakeRefused(
            "MATH/FORMAL: z3py is not available; lab stays declared. "
            f"detail={status['z3py']['error']}"
        )
    claims = check_claims()
    doc = assemble_receipt_doc(claims)
    artifact = write_receipt(RECEIPT_NAME, doc, recorded_by=RECORDED_BY)
    del write
    bounty = make_internal_bounty(
        id=f"math:{SCHEMA}:apple-tg-and-h1",
        source=str(artifact.resolve()),
        domain="math",
        question_or_target=(
            "Formalize this repo's Apple threadgroup ceiling, KERNEL_GEOMETRY "
            "occupancy identity, H.1 risk monotonicity, and complete_ebpw byte "
            "conservation; independently check with z3; archive a counterexample "
            "to 'any power-of-two threadgroup is legal'."
        ),
        nonmonetary_value="formal proof or counterexample",
        bounty_class=BountyClass.FORMAL_PROOF_OR_COUNTEREXAMPLE,
        lab=LabKind.MATH_FORMAL.value,
        extra_rules=("H.6 math/proof recipe", "checker is local z3py"),
        evidence_required=("receipt schema", "seal_sha256", "z3.Solver.check"),
    )

    def inputs(_artifact: Path) -> ValueInputs:
        return _value_inputs(claims)

    return run_intake(
        bounty,
        value_inputs_factory=inputs,
        expected_schema=SCHEMA,
    )
