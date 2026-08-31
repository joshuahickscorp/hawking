"""ADAPTIVE_VERIFICATION — kill on the cheapest falsifier so expensive child work never launches.

Expensive proof is the enemy of throughput. This sidecar is the ordered
refusal to launch a later meta-funnel gate after an earlier stage has
already killed the candidate, and the ledger that names the funnel work
that was therefore not done.

It does not fork `meta_funnel.Funnel`, `negative_index.refuse_if_dead`, or
`flash_schools.cheapest_falsifier`. It sequences them. A cheap screen that
kills is a refusal to spend later gates. A cheap screen that PASSES proves
nothing except "not yet dead" and is never reportable as verification.
Missing input is a recorded refusal, never a silent pass. A candidate the
negative index already killed never enters a stage.

STATIC_ONLY. No GPU. No lease. No hardware number.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tools.future._common import write_receipt
from tools.future import meta_funnel as mf
from tools.future import negative_index as ni

RECEIPT = "ADAPTIVE_VERIFICATION.json"
SCHEMA = "hawking.future.adaptive_verification.v1"
RECORDED_BY = "tools/future/adaptive_verification.py"

VERDICT_KILLED = "KILLED"
VERDICT_REFUSED = "REFUSED"
VERDICT_REFUSED_DEAD = "REFUSED_DEAD"
VERDICT_NOT_YET_DEAD = "NOT_YET_DEAD"

FALSIFIER_STAGE_ID = "F"
FALSIFIER_NAME = "cheapest_falsifier"
FALSIFIER_COST_CLASS = "CHEAP_FALSIFIER"
FALSIFIER_WORKUNIT = "future.adaptive_verification.cheapest_falsifier"
FALSIFIER_ORIGIN = "tools/future/flash_schools.py:CANDIDATE_FIELDS.cheapest_falsifier"
FUNNEL_ORIGIN = "tools/future/meta_funnel.py:GATES"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Passing an early "
    "stage is not evidence of anything except not-yet-dead. Survival of "
    "every stage is NOT_YET_DEAD, never verified, never a promotion."
)


class ScreenError(ValueError):
    """A screen was asked to invent a pass or a verification."""


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    cost_class: str
    cost_rank: int
    required_input: str
    can_decide: str
    cannot_decide: str
    workunit: str
    origin: str

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cost_class": self.cost_class,
            "cost_rank": self.cost_rank,
            "required_input": self.required_input,
            "can_decide": self.can_decide,
            "cannot_decide": self.cannot_decide,
            "workunit": self.workunit,
            "origin": self.origin,
            "passing_proves": "not yet dead at this stage",
            "passing_does_not_prove": self.cannot_decide,
        }


OnStage = Callable[[Stage, dict[str, Any]], None]
RefuseFn = Callable[[dict[str, Any]], dict[str, Any] | None]


def funnel_workunit(gate: mf.Gate) -> str:
    """Identity of the child work a later gate would have been.

    Derived from `meta_funnel.GATES`, not invented. `saved()` returns
    these names so the work-not-done is falsifiable against the funnel.
    """
    return f"future.meta_funnel.gate.{gate.id}.{gate.name}"


def funnel_child_workunits() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "workunit": funnel_workunit(g),
            "gate_id": g.id,
            "gate_name": g.name,
            "cost_class": g.cost_class,
            "required_input": g.required_input,
            "can_decide": g.passing_proves,
            "cannot_decide": g.passing_does_not_prove,
            "origin": FUNNEL_ORIGIN,
        }
        for g in mf.GATES
    )


def _falsifier_stage() -> Stage:
    return Stage(
        id=FALSIFIER_STAGE_ID,
        name=FALSIFIER_NAME,
        cost_class=FALSIFIER_COST_CLASS,
        cost_rank=0,
        required_input="falsifier_observation",
        can_decide=(
            "Whether the named cheap observation already kills the candidate. "
            "A kill here is a refusal to launch any meta-funnel gate."
        ),
        cannot_decide=(
            "Teacher fit, held-out numerical, route identity, tokens, "
            "capability, physical NR, complete NX, EBPW, or promotion. "
            "A pass here is not-yet-dead, not verified."
        ),
        workunit=FALSIFIER_WORKUNIT,
        origin=FALSIFIER_ORIGIN,
    )


def _funnel_stage(gate: mf.Gate) -> Stage:
    return Stage(
        id=str(gate.id),
        name=gate.name,
        cost_class=gate.cost_class,
        cost_rank=gate.id,
        required_input=gate.required_input,
        can_decide=gate.passing_proves,
        cannot_decide=gate.passing_does_not_prove,
        workunit=funnel_workunit(gate),
        origin=FUNNEL_ORIGIN,
    )


def _named_falsifier(candidate: Mapping[str, Any]) -> Any:
    raw = candidate.get("cheapest_falsifier")
    if raw is None:
        inputs = candidate.get("inputs")
        if isinstance(inputs, dict):
            raw = inputs.get("cheapest_falsifier")
    return raw


def has_cheapest_falsifier(candidate: Mapping[str, Any]) -> bool:
    raw = _named_falsifier(candidate)
    if raw is None:
        return False
    if isinstance(raw, str) and not raw.strip():
        return False
    if isinstance(raw, (list, tuple, dict)) and not raw:
        return False
    return True


def _stages(candidate: Mapping[str, Any]) -> tuple[Stage, ...]:
    out: list[Stage] = []
    if has_cheapest_falsifier(candidate):
        out.append(_falsifier_stage())
    out.extend(_funnel_stage(g) for g in mf.GATES)
    ranks = [s.cost_rank for s in out]
    if ranks != sorted(ranks):
        raise ScreenError("ladder is not cheapest-first")
    return tuple(out)


def ladder(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ordered verification stages for this candidate, cheapest first.

    Each stage states what it can decide and what it cannot. Passing is
    never promotion and never verification.
    """
    return [s.public() for s in _stages(candidate)]


def _proposal(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": candidate.get("model"),
        "organ": candidate.get("organ"),
        "representation": candidate.get("representation"),
        "hypothesis_family": (
            candidate.get("hypothesis_family")
            or candidate.get("family")
            or candidate.get("technique")
            or candidate.get("mechanism")
        ),
        "technique": candidate.get("technique"),
        "mechanism": candidate.get("mechanism"),
        "lever": candidate.get("lever"),
        "family": candidate.get("family"),
        "machine": candidate.get("machine"),
    }


def _consult_index(
    candidate: Mapping[str, Any],
    refuse_fn: RefuseFn | None,
) -> dict[str, Any]:
    """Refuse before any stage when the index already killed this idea.

    Lookup is live so tests can monkeypatch `negative_index.refuse_if_dead`.
    An ingest failure is recorded as COPED_UNAVAILABLE, not as a pass of a
    stage — no stage has run yet.
    """
    fn: RefuseFn = refuse_fn if refuse_fn is not None else ni.refuse_if_dead
    try:
        hit = fn(_proposal(dict(candidate)))
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError, TypeError) as exc:
        return {
            "consulted": True,
            "state": "COPED_UNAVAILABLE",
            "hit": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if hit:
        return {
            "consulted": True,
            "state": "HIT",
            "hit": dict(hit),
            "reason": hit.get("reason") or "negative index already killed this hypothesis",
        }
    return {
        "consulted": True,
        "state": "CLEAR",
        "hit": None,
        "reason": "no refuse-eligible scar matched; stages may run",
    }


def _falsifier_observation(candidate: Mapping[str, Any]) -> Any:
    inputs = candidate.get("inputs")
    if isinstance(inputs, dict) and "falsifier_observation" in inputs:
        return inputs["falsifier_observation"]
    if "falsifier_observation" in candidate:
        return candidate["falsifier_observation"]
    spec = _named_falsifier(candidate)
    if isinstance(spec, dict):
        return spec
    return None


def _eval_falsifier(value: Any) -> tuple[str, str]:
    if isinstance(value, dict) and value.get("fired") is True:
        return (
            VERDICT_KILLED,
            value.get("mechanism") or "named cheapest falsifier fired",
        )
    if isinstance(value, dict) and value.get("fired") is False:
        return (
            "PASSED",
            "cheapest falsifier did not fire; not yet dead; proves nothing else",
        )
    return mf._eval_from_status(
        value,
        "cheapest falsifier did not fire; not yet dead; proves nothing else",
        "named cheapest falsifier fired",
    )


def _run_falsifier(candidate: dict[str, Any], stage: Stage) -> dict[str, Any]:
    raw = _falsifier_observation(candidate)
    state = mf.input_state(raw)
    if mf.is_absent(raw):
        return {
            "verdict": VERDICT_REFUSED,
            "reason": (
                f"required input {stage.required_input!r} is {state}; "
                "a named cheapest falsifier without an observation is not a pass "
                "and does not unlock later funnel gates"
            ),
            "input_state": state,
            "required_input": stage.required_input,
        }
    verdict, reason = _eval_falsifier(raw)
    return {
        "verdict": verdict,
        "reason": reason,
        "input_state": state,
        "required_input": stage.required_input,
    }


def _run_funnel_gate(
    candidate: dict[str, Any],
    stage: Stage,
    funnel: mf.Funnel,
) -> dict[str, Any]:
    gate = mf.GATES_BY_ID[int(stage.id)]
    result = funnel.advance(candidate, gate)
    return {
        "verdict": result.verdict,
        "reason": result.reason,
        "input_state": result.input_state,
        "required_input": result.required_input,
        "scar_id": (result.scar or {}).get("scar_id") if result.scar else None,
    }


def _saved_from_gates(
    launched_gate_ids: set[int],
    *,
    not_launched_because: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for unit in funnel_child_workunits():
        if unit["gate_id"] in launched_gate_ids:
            continue
        row = dict(unit)
        row["not_launched_because"] = not_launched_because
        out.append(row)
    return out


def _stage_log(stage: Stage, result: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(result["verdict"])
    passing = verdict == "PASSED"
    return {
        **stage.public(),
        "verdict": VERDICT_NOT_YET_DEAD if passing else verdict,
        "raw_verdict": verdict,
        "reason": result["reason"],
        "input_state": result.get("input_state"),
        "scar_id": result.get("scar_id"),
        "proves": "not yet dead at this stage" if passing else result["reason"],
        "does_not_prove": stage.cannot_decide,
    }


def screen(
    candidate: Mapping[str, Any],
    *,
    refuse_fn: RefuseFn | None = None,
    funnel: mf.Funnel | None = None,
    on_stage: OnStage | None = None,
) -> dict[str, Any]:
    """Run the ladder, stop at the first kill or refusal, name the cost.

    Later stages are not called after a stop. Survival of every stage is
    `NOT_YET_DEAD` with `verified=False`. The negative index is consulted
    before any stage; a hit launches zero stages.
    """
    cand = dict(candidate)
    stages = _stages(cand)
    funnel = funnel if funnel is not None else mf.Funnel()
    index = _consult_index(cand, refuse_fn)
    if index["state"] == "HIT":
        reason = str(index["reason"])
        return {
            "id": cand.get("id"),
            "verdict": VERDICT_REFUSED_DEAD,
            "verified": False,
            "killed_by": None,
            "killed_by_workunit": None,
            "refused_by": "negative_index",
            "reason": reason,
            "cost": {
                "stages_executed": 0,
                "cost_class_paid": None,
                "cost_rank_paid": None,
                "later_stages_launched": 0,
                "funnel_gates_launched": 0,
            },
            "stages_run": [],
            "stages_not_run": [s.public() for s in stages],
            "saved": _saved_from_gates(
                set(),
                not_launched_because=(
                    "negative index already killed this hypothesis; "
                    "no funnel gate was launched"
                ),
            ),
            "negative_index": index,
            "claim_boundary": CLAIM_BOUNDARY,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    log: list[dict[str, Any]] = []
    launched_gate_ids: set[int] = set()
    stall: dict[str, Any] | None = None
    stall_stage: Stage | None = None
    for stage in stages:
        if on_stage is not None:
            on_stage(stage, cand)
        if stage.id == FALSIFIER_STAGE_ID:
            result = _run_falsifier(cand, stage)
        else:
            result = _run_funnel_gate(cand, stage, funnel)
            launched_gate_ids.add(int(stage.id))
        log.append(_stage_log(stage, result))
        if result["verdict"] != "PASSED":
            stall = result
            stall_stage = stage
            break

    remaining = [s.public() for s in stages[len(log) :]]
    if stall is None:
        if remaining:
            raise ScreenError("ladder claimed survival while stages remain")
        reason = (
            "every scheduled stage passed; still not verified, still not a "
            "promotion (STATIC_ONLY). Passing is not-yet-dead."
        )
        return {
            "id": cand.get("id"),
            "verdict": VERDICT_NOT_YET_DEAD,
            "verified": False,
            "killed_by": None,
            "killed_by_workunit": None,
            "refused_by": None,
            "reason": reason,
            "cost": {
                "stages_executed": len(log),
                "cost_class_paid": stages[-1].cost_class if stages else None,
                "cost_rank_paid": stages[-1].cost_rank if stages else None,
                "later_stages_launched": 0,
                "funnel_gates_launched": len(launched_gate_ids),
            },
            "stages_run": log,
            "stages_not_run": remaining,
            "saved": [],
            "negative_index": index,
            "claim_boundary": CLAIM_BOUNDARY,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    assert stall_stage is not None
    public_verdict = (
        VERDICT_KILLED if stall["verdict"] == VERDICT_KILLED else VERDICT_REFUSED
    )
    why = (
        f"{public_verdict.lower()} at {stall_stage.name} "
        f"(cost_class={stall_stage.cost_class}); later funnel gates not launched"
    )
    return {
        "id": cand.get("id"),
        "verdict": public_verdict,
        "verified": False,
        "killed_by": stall_stage.name if public_verdict == VERDICT_KILLED else None,
        "killed_by_workunit": stall_stage.workunit if public_verdict == VERDICT_KILLED else None,
        "refused_by": stall_stage.name if public_verdict == VERDICT_REFUSED else None,
        "reason": stall["reason"],
        "cost": {
            "stages_executed": len(log),
            "cost_class_paid": stall_stage.cost_class,
            "cost_rank_paid": stall_stage.cost_rank,
            "later_stages_launched": 0,
            "funnel_gates_launched": len(launched_gate_ids),
        },
        "stages_run": log,
        "stages_not_run": remaining,
        "saved": _saved_from_gates(launched_gate_ids, not_launched_because=why),
        "negative_index": index,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "scar_id": stall.get("scar_id"),
    }


def saved(
    candidate: Mapping[str, Any],
    *,
    screen_result: Mapping[str, Any] | None = None,
    refuse_fn: RefuseFn | None = None,
    funnel: mf.Funnel | None = None,
    on_stage: OnStage | None = None,
) -> list[dict[str, Any]]:
    """Child work that was not launched, named from `meta_funnel.GATES`.

    An invented count is not this function. Each row is a real funnel
    gate with its workunit identity. Empty only when every funnel gate
    actually ran (still not verified).
    """
    result = screen_result if screen_result is not None else screen(
        candidate, refuse_fn=refuse_fn, funnel=funnel, on_stage=on_stage
    )
    rows = result.get("saved")
    if not isinstance(rows, list):
        raise ScreenError("screen result is missing saved child work")
    return list(rows)


# ---------------------------------------------------------------------------
# Proofs the receipt has to have actually run, not declared.
# ---------------------------------------------------------------------------


def _plan() -> dict[str, Any]:
    return {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": False,
        "regions": [
            {
                "kind": "shared_generator",
                "bits_class": "shared",
                "family": "shared_basis",
                "organ": "routed_experts",
            }
        ],
    }


def _candidate(cid: str, *, cheapest: Any = None, observation: Any = None, **inputs: Any) -> dict[str, Any]:
    plan = inputs.pop("allocation_plan", None) or _plan()
    body = mf._default_inputs(allocation_plan=plan)
    body.update(inputs)
    cand: dict[str, Any] = {
        "id": cid,
        "family": "shared_basis",
        "organ": "routed_experts",
        "technique": "shared_basis_plus_nf_residual",
        "model": mf.FLASH_MODEL,
        "allocation_plan": plan,
        "inputs": body,
        "passed_gates": [],
    }
    if cheapest is not None:
        cand["cheapest_falsifier"] = cheapest
    if observation is not None:
        cand["falsifier_observation"] = observation
        cand["inputs"]["falsifier_observation"] = observation
    return cand


def _clear(_proposal: dict[str, Any]) -> None:
    return None


def _all_pass_inputs() -> dict[str, Any]:
    ok = {"status": "PASSED"}
    return {
        "teacher_corpus": dict(ok),
        "held_out_numerical": dict(ok),
        "route_traces": dict(ok),
        "logit_token": dict(ok),
        "bounded_capability": dict(ok),
        "physical_nr": dict(ok),
        "complete_nx": dict(ok),
        "ebpw_ledger": {"status": "PASSED", "all_required_bytes_included": True},
    }


def prove_kill_at_cheapest_falsifier_launches_zero_funnel_gates() -> dict[str, Any]:
    """NEGATIVE CONTROL: a stage-1 kill must not call later stages."""
    called: list[str] = []
    funnel = mf.Funnel()
    original = funnel.advance
    advanced: list[str] = []

    def wrapped(candidate: dict[str, Any], gate: int | str | mf.Gate) -> mf.AdvanceResult:
        g = mf.resolve_gate(gate)
        advanced.append(g.name)
        return original(candidate, gate)

    funnel.advance = wrapped  # type: ignore[method-assign]
    cand = _candidate(
        "proof.kill.falsifier",
        cheapest="Kill if the cheap observation already fires.",
        observation={"fired": True, "mechanism": "cheap observation killed it"},
        **_all_pass_inputs(),
    )
    result = screen(
        cand,
        refuse_fn=_clear,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    saved_units = saved(cand, screen_result=result)
    later = [g.name for g in mf.GATES]
    holds = (
        result["verdict"] == VERDICT_KILLED
        and result["killed_by"] == FALSIFIER_NAME
        and result["verified"] is False
        and called == [FALSIFIER_NAME]
        and advanced == []
        and result["cost"]["later_stages_launched"] == 0
        and result["cost"]["funnel_gates_launched"] == 0
        and [u["gate_name"] for u in saved_units] == later
        and all(u["origin"] == FUNNEL_ORIGIN for u in saved_units)
    )
    if not holds:
        raise ScreenError(
            "cheapest-falsifier kill leaked later work: "
            f"called={called} advanced={advanced} saved={len(saved_units)}"
        )
    return {
        "holds": True,
        "killed_by": result["killed_by"],
        "stages_called": list(called),
        "funnel_advance_called": list(advanced),
        "saved_workunits": [u["workunit"] for u in saved_units],
        "n_saved": len(saved_units),
        "n_funnel_gates": len(mf.GATES),
    }


def prove_survive_is_not_verified() -> dict[str, Any]:
    """NEGATIVE CONTROL: surviving every stage is not verification."""
    cand = _candidate("proof.survive", **_all_pass_inputs())
    result = screen(cand, refuse_fn=_clear)
    blob = json.dumps(result, sort_keys=True)
    holds = (
        result["verdict"] == VERDICT_NOT_YET_DEAD
        and result["verified"] is False
        and result.get("status") not in {"VERIFIED", "PASSED_ALL"}
        and '"VERIFIED"' not in blob
        and result["saved"] == []
        and result["cost"]["funnel_gates_launched"] == len(mf.GATES)
    )
    if not holds:
        raise ScreenError(f"survival was reported as verification: {result['verdict']}")
    return {
        "holds": True,
        "verdict": result["verdict"],
        "verified": result["verified"],
        "n_stages_run": len(result["stages_run"]),
        "passing_proves": "not yet dead",
    }


def prove_missing_input_is_not_pass() -> dict[str, Any]:
    """NEGATIVE CONTROL: absent teacher corpus is REFUSED, not PASSED."""
    called: list[str] = []
    cand = _candidate("proof.missing.teacher")
    result = screen(
        cand,
        refuse_fn=_clear,
        on_stage=lambda s, _c: called.append(s.name),
    )
    later = [g.name for g in mf.GATES if g.id > 2]
    holds = (
        result["verdict"] == VERDICT_REFUSED
        and result["refused_by"] == "real_teacher_fit"
        and result["verified"] is False
        and called == ["analytical_structure_screen", "real_teacher_fit"]
        and all(name not in called for name in later)
        and result["stages_run"][-1]["raw_verdict"] == VERDICT_REFUSED
        and result["stages_run"][-1]["verdict"] != "PASSED"
    )
    if not holds:
        raise ScreenError(
            f"missing input was treated as a pass: called={called} verdict={result['verdict']}"
        )
    return {
        "holds": True,
        "refused_by": result["refused_by"],
        "input_state": result["stages_run"][-1]["input_state"],
        "later_stages_called": [n for n in later if n in called],
        "saved_workunits": [u["workunit"] for u in result["saved"]],
    }


def prove_negative_index_blocks_all_stages() -> dict[str, Any]:
    """NEGATIVE CONTROL: a dead index hit launches zero stages."""
    called: list[str] = []
    funnel = mf.Funnel()
    original = funnel.advance
    advanced: list[str] = []

    def wrapped(candidate: dict[str, Any], gate: int | str | mf.Gate) -> mf.AdvanceResult:
        g = mf.resolve_gate(gate)
        advanced.append(g.name)
        return original(candidate, gate)

    funnel.advance = wrapped  # type: ignore[method-assign]

    def refuse(_proposal: dict[str, Any]) -> dict[str, Any]:
        return {
            "refused": True,
            "scar_id": "PROOF-DEAD",
            "source_path": "tools/future/negative_index.py",
            "reason": "known-dead hypothesis; rediscovery is not free",
            "hypothesis_family": "cross_expert_structure",
        }

    cand = _candidate("proof.dead.index", **_all_pass_inputs())
    result = screen(
        cand,
        refuse_fn=refuse,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    holds = (
        result["verdict"] == VERDICT_REFUSED_DEAD
        and result["refused_by"] == "negative_index"
        and called == []
        and advanced == []
        and result["cost"]["stages_executed"] == 0
        and result["cost"]["funnel_gates_launched"] == 0
        and len(result["saved"]) == len(mf.GATES)
    )
    if not holds:
        raise ScreenError(
            f"dead-index candidate leaked stages: called={called} advanced={advanced}"
        )
    return {
        "holds": True,
        "verdict": result["verdict"],
        "stages_called": list(called),
        "funnel_advance_called": list(advanced),
        "saved_workunits": [u["workunit"] for u in result["saved"]],
    }


def prove_ladder_states_what_it_cannot_decide() -> dict[str, Any]:
    rows = ladder(_candidate("proof.ladder.shape"))
    funnel_rows = [r for r in rows if r["origin"] == FUNNEL_ORIGIN]
    holds = (
        len(funnel_rows) == len(mf.GATES)
        and all(r["cannot_decide"] for r in rows)
        and all(r["can_decide"] for r in rows)
        and all(r["passing_proves"] == "not yet dead at this stage" for r in rows)
        and [r["name"] for r in funnel_rows] == [g.name for g in mf.GATES]
        and [r["cost_rank"] for r in rows] == sorted(r["cost_rank"] for r in rows)
    )
    if not holds:
        raise ScreenError("ladder lost a funnel stage or its cannot-decide clause")
    return {
        "holds": True,
        "n_stages": len(rows),
        "funnel_stage_names": [r["name"] for r in funnel_rows],
        "workunits": [r["workunit"] for r in funnel_rows],
    }


def build() -> Path:
    from tools.future import flash_schools as fs

    families, provenance = mf.recover_families()
    scars: list[Any] | None
    index_state = "CLEAR"
    try:
        scars = ni.ingest()
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        scars = None
        index_state = f"COPED_UNAVAILABLE:{type(exc).__name__}"

    def refuse(proposal: dict[str, Any]) -> dict[str, Any] | None:
        if scars is None:
            return None
        return ni.refuse_if_dead(proposal, scars=scars)

    recovered_screens = [screen(dict(c), refuse_fn=refuse) for c in families]
    n_saved = sum(len(r["saved"]) for r in recovered_screens)

    proofs = {
        "kill_at_cheapest_falsifier_launches_zero_funnel_gates": (
            prove_kill_at_cheapest_falsifier_launches_zero_funnel_gates()
        ),
        "survive_is_not_verified": prove_survive_is_not_verified(),
        "missing_input_is_not_pass": prove_missing_input_is_not_pass(),
        "negative_index_blocks_all_stages": prove_negative_index_blocks_all_stages(),
        "ladder_states_what_it_cannot_decide": prove_ladder_states_what_it_cannot_decide(),
    }
    if not all(p.get("holds") for p in proofs.values()):
        raise ScreenError("a mandatory proof did not hold")

    recovered_implementation = [
        "tools/future/meta_funnel.py GATES / Funnel.advance / Funnel.run / input_state / is_absent — composed, not forked; saved() names these gates",
        "tools/future/negative_index.py refuse_if_dead — consulted before any stage",
        "tools/future/flash_schools.py CANDIDATE_FIELDS.cheapest_falsifier — first ladder stage when the candidate names one",
        "tools/future/candidate_planner.py descendants_of — lineage invalidation already exists; this module names funnel child work, not queue descendants",
        "tools/future/qualification_pipeline.py STAGES — GPU sequencer; not forked (this ladder has no lease and no GPU)",
        "tools/future/evidence_dag.py V0–V9 — evidence-level hierarchy, a different ladder; not forked",
        "tools/future/scar_scheduling.py admit — scheduling-side scar refusal; the pre-stage gate here is refuse_if_dead",
    ]
    gaps_closed = [
        "ladder(candidate): cheapest-first stages, each with can_decide / cannot_decide, derived from meta_funnel.GATES plus an optional cheapest_falsifier stage",
        "screen(candidate): stop at the first kill or refusal; later stages are not called; cost is the cost_class paid, not a hardware number",
        "saved(candidate): names the concrete un-launched future.meta_funnel.gate.N.name workunits from GATES, not an invented count",
        "negative_index.refuse_if_dead is consulted before any stage; a hit launches zero stages",
        "survival of every stage is NOT_YET_DEAD with verified=false; a pass is not verification",
        "missing required input is REFUSED, never silently PASSED",
    ]
    negative_findings = [
        "this module is not in tools/future/orchestration.py BINDINGS; invoke() will raise UnknownBinding until the connector is extended. Naming a frontier here is not a binding",
        "saved() names funnel gates, not candidate_planner descendants — queue lineage is a different child-work graph and was not merged",
        "flash_schools cheapest_falsifier is a prose kill criterion; without falsifier_observation the cheapest stage REFUSES rather than inventing a measurement",
        f"negative-index consult during recovered-family screens: {index_state}",
        "passing cheapest_falsifier or any funnel gate is not evidence of teacher fit, tokens, capability, NR, NX, EBPW, or promotion",
        "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
    ]

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Make cheap-falsifier-then-stop automatic: a candidate killed at "
            "the cheapest stage never launches later meta-funnel gates, and "
            "the work not done is named from those gates."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "api": {
            "ladder": "tools.future.adaptive_verification.ladder(candidate)",
            "screen": "tools.future.adaptive_verification.screen(candidate)",
            "saved": "tools.future.adaptive_verification.saved(candidate)",
        },
        "funnel_child_workunits": list(funnel_child_workunits()),
        "flash_schools_cheapest_falsifier_field": "cheapest_falsifier" in fs.CANDIDATE_FIELDS,
        "proofs": proofs,
        "recovered_family_screens": [
            {
                "id": r.get("id"),
                "verdict": r["verdict"],
                "verified": r["verified"],
                "killed_by": r.get("killed_by"),
                "refused_by": r.get("refused_by"),
                "funnel_gates_launched": r["cost"]["funnel_gates_launched"],
                "n_saved": len(r["saved"]),
                "saved_workunits": [u["workunit"] for u in r["saved"]],
            }
            for r in recovered_screens
        ],
        "recovery_provenance": provenance,
        "counts": {
            "funnel_gates": len(mf.GATES),
            "recovered_families_screened": len(recovered_screens),
            "saved_workunits_across_recovered": n_saved,
            "proofs_held": sum(1 for p in proofs.values() if p.get("holds")),
        },
        "recovered_implementation": recovered_implementation,
        "gaps_closed": gaps_closed,
        "negative_findings": negative_findings,
        "resident_callable": {
            "entry_point": "tools.future.adaptive_verification.screen(candidate)",
            "workunit": (
                "one CPU_ANALYSIS unit; screen one candidate cheapest-first "
                "and name the meta_funnel gates that were therefore not launched"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.VERIFICATION.repro",
            "fails_closed": (
                "absent input -> REFUSED, never PASSED; negative-index hit -> "
                "REFUSED_DEAD and zero stages run; survival -> NOT_YET_DEAD "
                "with verified=false, never VERIFIED"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
