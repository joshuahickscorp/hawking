"""F0 pipeline instrument and F1 premise-retrieval training loop on tiny fixtures.

Local forge stages (from the Ramanujan handoff contract: F0-F9 small trainable system):

  F0  diagnose  -- inventory what is wired, what is gated, what is refused
  F1  train     -- premise-retrieval training loop on the fixture corpus

Both stages are `NON_PRODUCTION_AUTHORITY`. Neither flips `RAMANUJAN_RESEARCH_AUTHORIZED`.
Neither generates teacher traces from Math-Preserve (hash-REFUSED). Neither needs Lean,
Mathlib, or the network.

The point of F0 as an *instrument* rather than a paragraph is that a later stage can
read the receipt and refuse to pretend a missing component is present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ramanujan.economics import ALLOCATOR, CURRENCY, REWARDS, BranchAccount
from ramanujan.limits import LimitRegistry
from ramanujan.roles import (
    GENERATOR_IDS,
    PROMOTER_IDS,
    ROLE_CATALOG,
    generators_may_not_promote,
)
from ramanujan.search import AUTHORITY, PremiseRetrieval, SearchEconomics
from ramanujan.sovereignty import SovereigntyHooks, SovereigntyRefused
from ramanujan.stores import STORE_NAMES

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
PREMISE_CORPUS_PATH = FIXTURE_DIR / "premise_corpus.json"


def load_premise_fixture(path: Path | None = None) -> dict:
    p = path or PREMISE_CORPUS_PATH
    data = json.loads(p.read_text())
    if data.get("authority") != AUTHORITY:
        raise ValueError(
            f"premise fixture must declare authority={AUTHORITY!r}; got {data.get('authority')!r}"
        )
    return data


# --------------------------------------------------------------------------
# F0 -- diagnose
# --------------------------------------------------------------------------
def f0_diagnose(
    limits: LimitRegistry | None = None,
    premise_path: Path | None = None,
) -> dict[str, Any]:
    """F0 pipeline instrument: deterministic readiness inventory against fixtures.

    Reports what is present, what is structurally enforced, and what is still gated.
    Does not train. Does not search. Does not claim research authorization.
    """
    reg = limits or LimitRegistry()
    hooks = SovereigntyHooks(limits=reg)
    # F0 itself must pass the forge gate (local fixture stage).
    gate = hooks.forge_gate("F0")
    if not gate["allowed"]:
        raise RuntimeError(f"F0 forge gate refused: {gate['reason']}")

    fixture_ok = False
    fixture_meta: dict[str, Any] = {}
    try:
        fx = load_premise_fixture(premise_path)
        fixture_ok = True
        fixture_meta = {
            "n_premises": len(fx.get("premises", {})),
            "n_queries": len(fx.get("queries", [])),
            "authority": fx.get("authority"),
        }
    except (OSError, ValueError, json.JSONDecodeError) as e:
        fixture_meta = {"error": str(e)}

    retriever = PremiseRetrieval(corpus=(fx.get("premises") if fixture_ok else {}) or {})

    components = {
        "roles": {
            "status": "present",
            "n_roles": len(ROLE_CATALOG),
            "generators": sorted(GENERATOR_IDS),
            "promoters": sorted(PROMOTER_IDS),
            "generators_may_not_promote": generators_may_not_promote(),
        },
        "stores": {
            "status": "present",
            "names": list(STORE_NAMES),
            "count": len(STORE_NAMES),
        },
        "economics": {
            "status": "present",
            "allocator": ALLOCATOR,
            "currency": CURRENCY,
            "reward_kinds": sorted(REWARDS),
            "search_economics": "ramanujan.search.SearchEconomics",
            "session_economics": "ramanujan.economics.SessionEconomics",
        },
        "limit_registry": {
            "status": "present",
            "ids": list(reg.ids()),
            "RAMANUJAN_RESEARCH_AUTHORIZED": reg.research_authorized(),
            "consultable": True,
        },
        "sovereignty": {
            "status": "present",
            "hooks": "ramanujan.sovereignty.SovereigntyHooks",
            "local_forge": sorted(hooks.as_public_dict()["local_forge_stages"]),
            "gated_forge": sorted(hooks.as_public_dict()["gated_forge_stages"]),
        },
        "graveyard": {
            "status": "present",
            "law": "burial is not deletion; free resurrection is refused",
        },
        "premise_retrieval": {
            "status": "present_untrained" if not retriever.trained else "present_trained",
            **retriever.status(),
            "fixture": fixture_meta,
            "fixture_ok": fixture_ok,
        },
        "lean_mathlib": {
            "status": "partial",
            "note": (
                "Lean/Mathlib may be present on this host (see toolchain_selftest); "
                "fixture Lean remains NON_PRODUCTION_AUTHORITY for this layer"
            ),
        },
        "math_preserve_teacher_traces": {
            "status": "REFUSED",
            "limit": "L-TEACHER-01",
            "note": "never generate teacher traces from the collapsed Math-Preserve artifact",
        },
    }

    return {
        "stage": "F0",
        "name": "diagnose",
        "authority": AUTHORITY,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "forge_gate": gate,
        "components": components,
        "ready_for_f1": fixture_ok and generators_may_not_promote(),
        "blocked_from_research": not reg.research_authorized(),
    }


# --------------------------------------------------------------------------
# F1 -- premise-retrieval training loop
# --------------------------------------------------------------------------
def f1_train_premise_retrieval(
    premise_path: Path | None = None,
    steps: int = 40,
    limits: LimitRegistry | None = None,
) -> dict[str, Any]:
    """F1 training loop: fit PremiseRetrieval on the labelled fixture corpus.

    Consults the Limit Registry before training so the registry is on the path rather
    than decorative. Refuses Math-Preserve teacher-trace actions. Does not set
    RAMANUJAN_RESEARCH_AUTHORIZED.
    """
    reg = limits or LimitRegistry()
    hooks = SovereigntyHooks(limits=reg)
    hooks.require_forge_stage("F1")

    # Training on fixtures is not "run_research" and is not a Math-Preserve trace.
    # Explicitly refuse the forbidden action so the consult log shows the block.
    teacher = reg.consult("teacher_trace_from_math_preserve", role_id="librarian")
    if teacher.allowed:
        raise RuntimeError(
            "L-TEACHER-01 must block teacher_trace_from_math_preserve; registry is misconfigured"
        )

    # Local fixture training is allowed without research authorization; the fence
    # blocks run_research, not instrument loops on NON_PRODUCTION fixtures.
    research = reg.consult("run_research", role_id="director")
    if research.allowed:
        raise RuntimeError(
            "RAMANUJAN_RESEARCH_AUTHORIZED must remain false; registry is misconfigured"
        )
    # Sovereignty path: refuse_research must raise while unauthorized.
    try:
        hooks.refuse_research(role_id="director")
    except SovereigntyRefused:
        pass
    else:
        raise RuntimeError("refuse_research must raise while unauthorized")

    fx = load_premise_fixture(premise_path)
    retriever = PremiseRetrieval(corpus=dict(fx["premises"]))
    before = _ranking_quality(retriever, fx["queries"])
    train_receipt = retriever.train(fx["queries"], steps=steps)
    after = _ranking_quality(retriever, fx["queries"])

    return {
        "stage": "F1",
        "name": "premise_retrieval_train",
        "authority": AUTHORITY,
        "RAMANUJAN_RESEARCH_AUTHORIZED": reg.research_authorized(),
        "retriever": retriever.status(),
        "train": train_receipt,
        "quality_before": before,
        "quality_after": after,
        "improved": after["mean_reciprocal_rank"] >= before["mean_reciprocal_rank"],
        "limit_consults": list(reg.consult_log),
        "teacher_trace_blocked": not teacher.allowed,
        "note": (
            "Trainable and trained on fixtures only. Label remains honest: "
            f"{retriever.label}. Not production premise selection."
        ),
        "retriever_object": retriever,
    }


def _ranking_quality(retriever: PremiseRetrieval, queries: list[dict]) -> dict:
    """Mean reciprocal rank of the first relevant hit. Fixture metric only."""
    rrs: list[float] = []
    hits_at_1 = 0
    for q in queries:
        ranking = retriever.retrieve(q["goal"], k=max(3, len(retriever.corpus)))
        relevant = set(q.get("relevant", []))
        rr = 0.0
        for i, (name, _score) in enumerate(ranking, start=1):
            if name in relevant:
                rr = 1.0 / i
                if i == 1:
                    hits_at_1 += 1
                break
        rrs.append(rr)
    n = max(1, len(rrs))
    return {
        "n_queries": len(rrs),
        "mean_reciprocal_rank": sum(rrs) / n,
        "hits_at_1": hits_at_1,
        "retriever_label": retriever.label,
        "trained": retriever.trained,
    }


def run_f0_f1(
    premise_path: Path | None = None,
    steps: int = 40,
) -> dict[str, Any]:
    """Run F0 then F1. Returns a combined receipt. No multi-hour anything."""
    reg = LimitRegistry()
    f0 = f0_diagnose(limits=reg, premise_path=premise_path)
    if not f0["ready_for_f1"]:
        return {
            "authority": AUTHORITY,
            "f0": f0,
            "f1": None,
            "status": "F1_SKIPPED_F0_NOT_READY",
        }
    f1 = f1_train_premise_retrieval(premise_path=premise_path, steps=steps, limits=reg)
    # Drop the live object from the JSON-friendly receipt.
    f1_public = {k: v for k, v in f1.items() if k != "retriever_object"}
    return {
        "authority": AUTHORITY,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "f0": f0,
        "f1": f1_public,
        "status": "F0_F1_COMPLETE_ON_FIXTURES",
    }


def demo_branch_under_budget(max_expansions: int = 3) -> dict:
    """Small economics demo used by the receipt: a branch that exhausts and records why."""
    from ramanujan.search import ProofState

    def tactics(s: ProofState):
        if s.goal.startswith("g"):
            n = int(s.goal[1:])
            if n > 0:
                yield "peel", ProofState(f"g{n - 1}", s.hyps)
            else:
                yield "close", ProofState("True", s.hyps)

    branch = BranchAccount(
        branch_id="demo_exhaust",
        economics=SearchEconomics(max_expansions=max_expansions, max_depth=100),
    )
    from ramanujan.economics import run_branch_search

    result, branch = run_branch_search(
        branch,
        ProofState("g50"),
        tactics,
        heuristic=lambda s: float(int(s.goal[1:])) if s.goal.startswith("g") else 0.0,
    )
    return {
        "found": result.found,
        "stopped_by": result.stopped_by,
        "branch_score": branch.score(),
        "halt": None
        if branch.halt is None
        else {
            "reason": branch.halt.reason,
            "spent": branch.halt.spent,
            "value_earned": branch.halt.value_earned,
        },
    }
