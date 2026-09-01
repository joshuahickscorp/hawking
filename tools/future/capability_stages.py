#!/usr/bin/env python3
"""CAPABILITY STAGES — what it costs to ask "is the model still capable".

SUB2 has been measuring LOCAL_FUNCTIONAL_FIDELITY (hidden-state cosine).
That is not capability. This sidecar is the staged evaluation API so a
caller can choose depth and terminate cheaply. The caller supplies the
component. This module does not choose which representation or organ to
test.

Five stages, cheapest first:

    LOCAL_FUNCTIONAL_FIDELITY  hidden-state cosine (wraps the existing map)
    LOGIT_TOKEN                next-token logits + argmax, KL, top-k
    FAST_CAPABILITY            a small fixed prompt set with checkable answers
    HCLI_MISSION_SUBSET        valid structured HCLI work-request emission
    EXPENSIVE_QUALIFICATION    declared; this sidecar refuses politely

Each stage reports what it measured, wall cost in seconds, and an explicit
statement of what it does NOT establish. A stage that cannot run SKIPS with
a reason and is never counted as a pass.

    python3 tools/future/capability_stages.py --build
    python3 -m pytest tools/future/test_capability_stages.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not spawn a resident.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.future import aux_capability_screen as acs
from tools.future import capability_information_map as cim
from tools.future import functional_role_probe as fp
from tools.future import qualification_pipeline as qp
from tools.future import resident_provider as rp
from tools.future import workunit_species as ws
from tools.future._common import write_receipt


RECEIPT = "CAPABILITY_STAGES.json"
SCHEMA = "hawking.future.capability_stages.v1"
VERSION = 1
RECORDED_BY = "tools/future/capability_stages.py"

LOCAL_FUNCTIONAL_FIDELITY = "LOCAL_FUNCTIONAL_FIDELITY"
LOGIT_TOKEN = "LOGIT_TOKEN"
FAST_CAPABILITY = "FAST_CAPABILITY"
HCLI_MISSION_SUBSET = "HCLI_MISSION_SUBSET"
EXPENSIVE_QUALIFICATION = "EXPENSIVE_QUALIFICATION"

STAGE_IDS: tuple[str, ...] = (
    LOCAL_FUNCTIONAL_FIDELITY,
    LOGIT_TOKEN,
    FAST_CAPABILITY,
    HCLI_MISSION_SUBSET,
    EXPENSIVE_QUALIFICATION,
)

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
NOT_RUN = "NOT_RUN"

OVERALL_FAIL = "FAIL"
OVERALL_PASS_THROUGH_DEPTH = "PASS_THROUGH_DEPTH"
OVERALL_PARTIAL = "PARTIAL"
OVERALL_INCOMPLETE = "INCOMPLETE"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate. "
    "A stage PASS is only the measurement that stage names. Hidden-state "
    "cosine is not capability (functional_role_probe / S031 §9). Next-token "
    "argmax is not capability (meta_funnel logit_token_validation). A SKIPPED "
    "stage is not a pass. This sidecar never chooses the component under "
    "test; a missing component.id is a refusal. Wall seconds in the receipt "
    "are judge time on caller-supplied evidence in this process, not resident "
    "generate time and not a qualified token cost. evidence_class STATIC_ONLY. "
    "gpu_authority false."
)


class StagesRefuse(ValueError):
    """The staged evaluator refused rather than guessing or choosing science."""


class StageInputMissing(StagesRefuse):
    """A stage was asked to run without the evidence it measures."""

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"REFUSED: {stage} cannot run: {reason}")


class ExpensiveQualificationRefused(StagesRefuse):
    """EXPENSIVE_QUALIFICATION is declared; this sidecar will not run it."""

    def __init__(self, reason: str = "") -> None:
        self.stage = EXPENSIVE_QUALIFICATION
        self.reason = reason or (
            "EXPENSIVE_QUALIFICATION is declared but this sidecar has no GPU "
            "authority; qualification_pipeline.execute always refuses. "
            "This is a skip, not a pass."
        )
        super().__init__(f"REFUSED: {self.reason}")


@dataclass(frozen=True)
class Stage:
    id: str
    rank: int
    measures: str
    does_not_establish: str
    required_input: str
    origin: str
    evidence_cost_reason: str

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": self.rank,
            "cheapest_first_index": self.rank,
            "measures": self.measures,
            "does_not_establish": self.does_not_establish,
            "required_input": self.required_input,
            "origin": self.origin,
            "evidence_production_cost": "UNMEASURED",
            "evidence_production_reason": self.evidence_cost_reason,
        }


_STAGES: tuple[Stage, ...] = (
    Stage(
        id=LOCAL_FUNCTIONAL_FIDELITY,
        rank=0,
        measures=(
            "Hidden-state cosine between caller-supplied incumbent and "
            "candidate activations, judged against "
            "capability_information_map.HIDDEN_COSINE_BAR. This is the "
            "SUB2 LOCAL_FUNCTIONAL_FIDELITY instrument "
            f"(functional_role_probe.MEASURED_LEVEL={fp.MEASURED_LEVEL})."
        ),
        does_not_establish=(
            "Not capability, not next-token identity, not checkable-task "
            "competence, not HCLI mission competence, and not qualification. "
            "S031 §9 forbids local cosine as the final verdict. A role can "
            "carry unequal capability value while carrying equal fidelity value."
        ),
        required_input="hidden_a and hidden_b (incumbent vs candidate hidden state)",
        origin=(
            "tools/future/capability_information_map.py::_cosine, "
            "HIDDEN_COSINE_BAR; tools/future/functional_role_probe.py:"
            "MEASURED_LEVEL"
        ),
        evidence_cost_reason=(
            "producing hidden states is a CPU replay or a generate this "
            "sidecar does not run; only the cosine judge is timed here"
        ),
    ),
    Stage(
        id=LOGIT_TOKEN,
        rank=1,
        measures=(
            "Next-token logits: KL(incumbent || candidate), top-k agreement, "
            "logits cosine, and argmax agreement. KL and top-k are the "
            "screen; argmax is a side report and is not parity."
        ),
        does_not_establish=(
            "Not checkable-task competence, not HCLI mission competence, "
            "and not qualification. meta_funnel gate 5: token-level identity "
            "on a stated probe is not bounded capability. Argmax agreement is "
            "not logit parity (aux_capability_screen.ArgmaxAloneParityRefuse)."
        ),
        required_input="logits_a and logits_b (incumbent vs candidate next-token logits)",
        origin=(
            "tools/future/aux_capability_screen.py::kl_divergence, "
            "topk_agreement, report_logit_parity, softmax; "
            "tools/future/capability_information_map.py::_cosine"
        ),
        evidence_cost_reason=(
            "producing logits requires a forward / LM-head this sidecar "
            "does not run; only KL/top-k/argmax on supplied logits are timed"
        ),
    ),
    Stage(
        id=FAST_CAPABILITY,
        rank=2,
        measures=(
            "A small fixed prompt set with deterministic predicates "
            "(capital of France, HCLI stale-artifact choice, 17*19). "
            "Degenerate replies that contain a checkable token still fail "
            "(resident_provider quality / MEASURED_DEGENERATE_CHOICE)."
        ),
        does_not_establish=(
            "HCLI mission competence, protected qualification, or that a "
            "live resident generate would match these answers. A three-item "
            "checkable floor is not the capability contract; the larger "
            "headless suite is not imported and is not this stage."
        ),
        required_input="answers keyed by probe id (fact-capital, fact-choice, fact-arith)",
        origin=(
            "tools/future/resident_provider.py::PROMPT_FRANCE, PROMPT_CHOICE, "
            "quality, MEASURED_DEGENERATE_CHOICE; "
            "tools/headless/capability_suite.py fact-arith cited not imported"
        ),
        evidence_cost_reason=(
            "producing answers requires a generate this sidecar does not "
            "spawn; only the deterministic predicates are timed"
        ),
    ),
    Stage(
        id=HCLI_MISSION_SUBSET,
        rank=3,
        measures=(
            "Whether caller-supplied emissions are valid structured HCLI "
            "work requests under workunit_species.validate_emitted_unit "
            "(core fields, claim_boundary, verifier, HCLI-safe effect_class)."
        ),
        does_not_establish=(
            "Protected qualification, throughput, or that the resident will "
            "emit this under a live mission loop. Valid work-request shape "
            "is necessary and not sufficient for mission competence."
        ),
        required_input="emissions: one or more WorkUnit mappings or JSON objects",
        origin=(
            "tools/future/workunit_species.py::validate_emitted_unit, "
            "emit_hcli_workunit"
        ),
        evidence_cost_reason=(
            "producing a live mission emission requires the resident; only "
            "WorkUnit validation is timed here"
        ),
    ),
    Stage(
        id=EXPENSIVE_QUALIFICATION,
        rank=4,
        measures=(
            "Nothing on this sidecar. The stage is declared so a caller can "
            "name the depth. qualification_pipeline.execute always raises; "
            "stages 10-12 of that pipeline emit a REQUEST/SPEC and stop."
        ),
        does_not_establish=(
            "Anything, when SKIPPED. A polite refusal is not a pass and is "
            "not a fail of the model. This sidecar has no GPU authority and "
            "will not seize an HCLI lease."
        ),
        required_input=(
            "a protected GPU lease and a QUIESCENT machine, which this "
            "sidecar will not take or coerce"
        ),
        origin="tools/future/qualification_pipeline.py::execute, STAGES",
        evidence_cost_reason=(
            "protected qualification requires a GPU lease this sidecar will "
            "not take; qualification_pipeline.execute always refuses"
        ),
    ),
)
STAGES_BY_ID: dict[str, Stage] = {s.id: s for s in _STAGES}


FAST_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "fact-capital",
        "prompt": rp.PROMPT_FRANCE,
        "origin": "tools/future/resident_provider.py:PROMPT_FRANCE",
        "expect_contains": ("paris",),
    },
    {
        "id": "fact-choice",
        "prompt": rp.PROMPT_CHOICE,
        "origin": "tools/future/resident_provider.py:PROMPT_CHOICE",
        "expect_contains": ("hbm_doctor.py",),
    },
    {
        "id": "fact-arith",
        "prompt": "Compute 17 * 19. Reply with only the number.",
        "origin": "tools/headless/capability_suite.py:fact-arith (cited, not imported)",
        "expect_number": 323,
    },
)
FAST_PROBE_IDS: tuple[str, ...] = tuple(p["id"] for p in FAST_PROBES)


OnStage = Callable[[str, Mapping[str, Any]], None]


def catalog() -> list[dict[str, Any]]:
    """Declared stages, cheapest first. Each names what it does not establish."""
    rows = [s.public() for s in _STAGES]
    ranks = [r["rank"] for r in rows]
    if ranks != sorted(ranks):
        raise StagesRefuse("ladder is not cheapest-first")
    if [r["id"] for r in rows] != list(STAGE_IDS):
        raise StagesRefuse("catalog order drifted from STAGE_IDS")
    if fp.MEASURED_LEVEL != LOCAL_FUNCTIONAL_FIDELITY:
        raise StagesRefuse(
            "functional_role_probe.MEASURED_LEVEL is no longer "
            "LOCAL_FUNCTIONAL_FIDELITY; this wrapper would be lying"
        )
    for row in rows:
        if not str(row.get("does_not_establish") or "").strip():
            raise StagesRefuse(f"{row['id']} is missing does_not_establish")
    return rows


def counts_as_pass(row: Mapping[str, Any] | None) -> bool:
    """The only legal pass tally. SKIPPED / NOT_RUN / FAIL are not passes."""
    if not isinstance(row, Mapping):
        raise StagesRefuse("counts_as_pass needs a stage result mapping")
    if "verdict" not in row:
        raise StagesRefuse("stage result has no verdict; cannot count it as a pass")
    return row["verdict"] == PASS


def passed_stage_ids(report: Mapping[str, Any]) -> list[str]:
    stages = report.get("stages")
    if not isinstance(stages, list):
        raise StagesRefuse("report.stages is missing")
    return [str(r["id"]) for r in stages if counts_as_pass(r)]


def require_component(subject: Mapping[str, Any] | None) -> dict[str, Any]:
    """Caller supplies the component. This module will not pick one."""
    if not isinstance(subject, Mapping):
        raise StagesRefuse(
            "REFUSED: evaluate() needs a mapping subject; the caller supplies "
            "the component and this module does not choose one"
        )
    comp = subject.get("component")
    if not isinstance(comp, Mapping):
        raise StagesRefuse(
            "REFUSED: caller did not supply component; this module does not "
            "choose which representation or organ to evaluate"
        )
    cid = str(comp.get("id") or "").strip()
    if not cid:
        raise StagesRefuse(
            "REFUSED: caller did not supply component.id; this module does "
            "not choose which component to evaluate"
        )
    return dict(comp)


def _stage(stage_id: str) -> Stage:
    if stage_id not in STAGES_BY_ID:
        raise StagesRefuse(f"unknown stage {stage_id!r}; known: {list(STAGE_IDS)}")
    return STAGES_BY_ID[stage_id]


def _upto(max_stage: str | None) -> tuple[Stage, ...]:
    if max_stage is None:
        return _STAGES
    want = _stage(max_stage)
    return tuple(s for s in _STAGES if s.rank <= want.rank)


def _nested(subject: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    raw = subject.get(stage_id) or subject.get(stage_id.lower())
    if isinstance(raw, Mapping):
        return raw
    return {}


def _vec(value: Any, *, stage: str, name: str) -> np.ndarray:
    if value is None:
        raise StageInputMissing(stage, f"{name} is missing")
    if isinstance(value, (str, bytes, Mapping)):
        raise StageInputMissing(stage, f"{name} is not an array")
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise StageInputMissing(stage, f"{name} is not an array: {exc}") from exc
    if arr.size == 0:
        raise StageInputMissing(stage, f"{name} is empty")
    return arr


def hidden_cosine(a: Any, b: Any) -> float:
    """Wrap capability_information_map._cosine. Not a new instrument."""
    return float(cim._cosine(np.asarray(a), np.asarray(b)))


def _pick_pair(
    subject: Mapping[str, Any],
    nested: Mapping[str, Any],
    *,
    names: tuple[str, ...],
) -> tuple[Any, Any] | None:
    for src in (nested, subject):
        got = [src.get(n) for n in names]
        if all(v is not None for v in got):
            return got[0], got[1]
    return None


def run_local_functional_fidelity(subject: Mapping[str, Any]) -> dict[str, Any]:
    nested = _nested(subject, LOCAL_FUNCTIONAL_FIDELITY)
    pair = _pick_pair(
        subject,
        nested,
        names=("hidden_a", "hidden_b"),
    )
    if pair is None:
        raise StageInputMissing(
            LOCAL_FUNCTIONAL_FIDELITY,
            "hidden_a and hidden_b are required; this stage does not invent activations",
        )
    a = _vec(pair[0], stage=LOCAL_FUNCTIONAL_FIDELITY, name="hidden_a")
    b = _vec(pair[1], stage=LOCAL_FUNCTIONAL_FIDELITY, name="hidden_b")
    if a.shape != b.shape:
        raise StageInputMissing(
            LOCAL_FUNCTIONAL_FIDELITY,
            f"hidden_a shape {a.shape} != hidden_b shape {b.shape}",
        )
    cosine = hidden_cosine(a, b)
    bar = float(cim.HIDDEN_COSINE_BAR)
    if cosine != cosine:  # NaN
        return {
            "verdict": SKIPPED,
            "reason": "cosine is undefined (zero-norm hidden state); not a pass",
            "measurement": {"cosine": None, "bar": bar, "n": int(a.size)},
        }
    passed = cosine >= bar
    return {
        "verdict": PASS if passed else FAIL,
        "reason": (
            f"hidden-state cosine {cosine:.6f} "
            f"{'>=' if passed else '<'} bar {bar}"
        ),
        "measurement": {
            "cosine": float(cosine),
            "bar": bar,
            "n": int(a.size),
            "instrument": fp.MEASURED_LEVEL,
            "wrapped": "capability_information_map._cosine",
        },
    }


def run_logit_token(subject: Mapping[str, Any]) -> dict[str, Any]:
    nested = _nested(subject, LOGIT_TOKEN)
    pair = _pick_pair(subject, nested, names=("logits_a", "logits_b"))
    if pair is None:
        raise StageInputMissing(
            LOGIT_TOKEN,
            "logits_a and logits_b are required; argmax alone is not this stage",
        )
    a = _vec(pair[0], stage=LOGIT_TOKEN, name="logits_a")
    b = _vec(pair[1], stage=LOGIT_TOKEN, name="logits_b")
    if a.shape != b.shape:
        raise StageInputMissing(
            LOGIT_TOKEN,
            f"logits_a shape {a.shape} != logits_b shape {b.shape}",
        )
    k = int(acs.TOPK)
    kl = float(acs.kl_divergence(acs.softmax(a), acs.softmax(b)))
    topk = float(acs.topk_agreement(a, b, k))
    argmax_a = int(np.argmax(a))
    argmax_b = int(np.argmax(b))
    argmax_ag = 1.0 if argmax_a == argmax_b else 0.0
    parity = acs.report_logit_parity(
        kl_nats=kl,
        top_k_agreement=topk,
        argmax_agreement=argmax_ag,
        k=k,
        n_rows=1,
    )
    logits_cos = hidden_cosine(a, b)
    kl_bar = float(acs.LOGIT_KL_BAR)
    topk_bar = float(acs.TOPK_AGREE_BAR)
    passed = kl <= kl_bar and topk >= topk_bar
    return {
        "verdict": PASS if passed else FAIL,
        "reason": (
            f"KL {kl:.6f} (bar {kl_bar}) top-{k} agreement {topk:.4f} "
            f"(bar {topk_bar}) argmax_agreement {argmax_ag} "
            f"(argmax is not the pass criterion)"
        ),
        "measurement": {
            **parity,
            "logits_cosine": None if logits_cos != logits_cos else float(logits_cos),
            "argmax_a": argmax_a,
            "argmax_b": argmax_b,
            "kl_bar": kl_bar,
            "top_k_bar": topk_bar,
            "wrapped": "aux_capability_screen.report_logit_parity",
        },
    }


def _reply_text(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    if isinstance(reply, Mapping):
        for key in ("text", "raw_text", "generated_text", "answer"):
            val = reply.get(key)
            if isinstance(val, str):
                return val
    raise StageInputMissing(FAST_CAPABILITY, "reply has no text")


def _reply_quality(reply: Any) -> tuple[str | None, str | None]:
    """(quality, reason). quality None means UNPROVEN, not CLEAN."""
    if isinstance(reply, str):
        return None, "reply is a bare string; CLEAN vs DEGENERATE is UNPROVEN"
    if not isinstance(reply, Mapping):
        return None, "reply is not a mapping; quality is UNPROVEN"
    try:
        return str(rp.quality(reply)), None
    except rp.QualityUnproven as exc:
        return None, str(exc)


def _probe_passes(probe: Mapping[str, Any], text: str) -> tuple[bool, str]:
    if not (text or "").strip():
        return False, "empty reply"
    needles = probe.get("expect_contains") or ()
    if needles and not all(n.lower() in text.lower() for n in needles):
        return False, f"expected all of {tuple(needles)}"
    number = probe.get("expect_number")
    if number is not None:
        nums = [int(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*", text)]
        if int(number) not in nums:
            return False, f"expected {number} among {nums[:8]}"
    return True, "predicate held"


def run_fast_capability(subject: Mapping[str, Any]) -> dict[str, Any]:
    nested = _nested(subject, FAST_CAPABILITY)
    answers = nested.get("answers", subject.get("answers"))
    if not isinstance(answers, Mapping) or not answers:
        raise StageInputMissing(
            FAST_CAPABILITY,
            "answers keyed by probe id are required; this stage does not generate",
        )
    missing = [pid for pid in FAST_PROBE_IDS if pid not in answers]
    if missing:
        raise StageInputMissing(
            FAST_CAPABILITY,
            f"answers missing {missing}; a partial set is not a pass of the fixed probe set",
        )
    items: list[dict[str, Any]] = []
    any_fail = False
    for probe in FAST_PROBES:
        reply = answers[probe["id"]]
        text = _reply_text(reply)
        quality, quality_reason = _reply_quality(reply)
        pred_ok, pred_reason = _probe_passes(probe, text)
        if quality == rp.QUALITY_DEGENERATE:
            ok = False
            reason = (
                "quality is DEGENERATE; a checkable token fished from loop "
                "garbage is not a pass (resident_provider.MEASURED_DEGENERATE_CHOICE)"
            )
        elif not pred_ok:
            ok = False
            reason = pred_reason
        else:
            ok = True
            reason = pred_reason
        any_fail = any_fail or not ok
        items.append(
            {
                "id": probe["id"],
                "origin": probe["origin"],
                "passed": ok,
                "reason": reason,
                "quality": quality if quality is not None else "UNPROVEN",
                "quality_reason": quality_reason,
                "text_len": len(text),
            }
        )
    passed = not any_fail
    return {
        "verdict": PASS if passed else FAIL,
        "reason": (
            f"{sum(1 for i in items if i['passed'])}/{len(items)} checkable probes passed"
        ),
        "measurement": {
            "n_probes": len(items),
            "n_passed": sum(1 for i in items if i["passed"]),
            "items": items,
            "wrapped": "resident_provider.quality / PROMPT_FRANCE / PROMPT_CHOICE",
        },
    }


def _as_emission_rows(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        return [raw]
    raise StageInputMissing(
        HCLI_MISSION_SUBSET, "emissions must be a mapping, JSON string, or list"
    )


def run_hcli_mission_subset(subject: Mapping[str, Any]) -> dict[str, Any]:
    nested = _nested(subject, HCLI_MISSION_SUBSET)
    raw = nested.get("emissions", subject.get("emissions"))
    rows = _as_emission_rows(raw)
    if not rows:
        raise StageInputMissing(
            HCLI_MISSION_SUBSET,
            "emissions are required; this stage does not invent work requests",
        )
    valid: list[str] = []
    invalid: list[dict[str, Any]] = []
    for item in rows:
        obj: Any = item
        if isinstance(item, str):
            try:
                obj = json.loads(item)
            except json.JSONDecodeError as exc:
                invalid.append({"id": None, "reason": f"not JSON: {exc}"})
                continue
        if not isinstance(obj, Mapping):
            invalid.append({"id": None, "reason": "emission is not an object"})
            continue
        try:
            ws.validate_emitted_unit(obj)
        except ws.WorkUnitShapeError as exc:
            invalid.append({"id": obj.get("id"), "reason": str(exc)})
            continue
        valid.append(str(obj.get("id")))
    passed = bool(valid) and not invalid
    return {
        "verdict": PASS if passed else FAIL,
        "reason": (
            f"{len(valid)} valid structured work request(s), {len(invalid)} invalid"
        ),
        "measurement": {
            "n_emissions": len(rows),
            "n_valid": len(valid),
            "n_invalid": len(invalid),
            "valid_ids": valid,
            "invalid": invalid,
            "wrapped": "workunit_species.validate_emitted_unit",
        },
    }


def expensive_qualification_declaration() -> dict[str, Any]:
    """Cite the existing pipeline. Do not run it."""
    return {
        "declared": True,
        "pipeline_stages": list(qp.STAGES),
        "n_pipeline_stages": len(qp.STAGES),
        "execute_always_refuses": True,
        "gpu_authority": False,
        "kind": "SPEC",
        "source": "tools/future/qualification_pipeline.py::execute",
        "reason": (
            "qualification_pipeline.execute raises unless --execute, an "
            "existing HCLI lease, and QUIESCENT all hold — and then raises "
            "anyway because this sidecar has no GPU authority"
        ),
    }


def run_expensive_qualification(subject: Mapping[str, Any]) -> dict[str, Any]:
    _ = subject
    raise ExpensiveQualificationRefused()


_RUNNERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    LOCAL_FUNCTIONAL_FIDELITY: run_local_functional_fidelity,
    LOGIT_TOKEN: run_logit_token,
    FAST_CAPABILITY: run_fast_capability,
    HCLI_MISSION_SUBSET: run_hcli_mission_subset,
    EXPENSIVE_QUALIFICATION: run_expensive_qualification,
}


def run_stage(stage_id: str, subject: Mapping[str, Any]) -> dict[str, Any]:
    """Run one stage. Missing evidence RAISES. Does not default."""
    require_component(subject)
    stage = _stage(stage_id)
    runner = _RUNNERS[stage.id]
    body = runner(subject)
    return _finish_row(stage, body)


def _finish_row(stage: Stage, body: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(body.get("verdict") or "")
    if verdict not in {PASS, FAIL, SKIPPED, NOT_RUN}:
        raise StagesRefuse(f"{stage.id} returned illegal verdict {verdict!r}")
    row = stage.public()
    row.update(dict(body))
    row["id"] = stage.id
    row["verdict"] = verdict
    row["counts_as_pass"] = counts_as_pass(row)
    row["does_not_establish"] = stage.does_not_establish
    row["measures"] = stage.measures
    if verdict == SKIPPED and not str(row.get("reason") or "").strip():
        raise StagesRefuse(f"{stage.id} SKIPPED without a reason")
    return row


def _skip_row(stage: Stage, reason: str) -> dict[str, Any]:
    return _finish_row(
        stage,
        {
            "verdict": SKIPPED,
            "reason": reason,
            "measurement": None,
        },
    )


def _not_run_row(stage: Stage, reason: str) -> dict[str, Any]:
    return _finish_row(
        stage,
        {
            "verdict": NOT_RUN,
            "reason": reason,
            "measurement": None,
        },
    )


def _cost_record(row: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(str(row["id"]))
    wall = row.get("wall_seconds")
    verdict = str(row.get("verdict"))
    measured = isinstance(wall, (int, float)) and not isinstance(wall, bool)
    if verdict in {SKIPPED, NOT_RUN}:
        return {
            "id": stage.id,
            "verdict": verdict,
            "wall_seconds": float(wall) if measured else None,
            "wall_cost": "MEASURED" if measured else "UNMEASURED",
            "evaluation_cost": "UNMEASURED",
            "reason": (
                str(row.get("reason") or "")
                or "stage did not run; skip-path wall is not the cost of the evaluation"
            ),
            "evidence_production_cost": "UNMEASURED",
            "evidence_production_reason": stage.evidence_cost_reason,
        }
    if not measured:
        return {
            "id": stage.id,
            "verdict": verdict,
            "wall_seconds": None,
            "wall_cost": "UNMEASURED",
            "evaluation_cost": "UNMEASURED",
            "reason": str(row.get("unmeasured_reason") or "stage wall was not timed"),
            "evidence_production_cost": "UNMEASURED",
            "evidence_production_reason": stage.evidence_cost_reason,
        }
    return {
        "id": stage.id,
        "verdict": verdict,
        "wall_seconds": float(wall),
        "wall_cost": "MEASURED",
        "evaluation_cost": "MEASURED",
        "reason": (
            "judge wall time on caller-supplied evidence in this process; "
            "not resident generate time"
        ),
        "evidence_production_cost": "UNMEASURED",
        "evidence_production_reason": stage.evidence_cost_reason,
    }


def _overall(rows: Sequence[Mapping[str, Any]]) -> str:
    verdicts = [str(r["verdict"]) for r in rows]
    if FAIL in verdicts:
        return OVERALL_FAIL
    if PASS in verdicts and SKIPPED not in verdicts and NOT_RUN not in verdicts:
        return OVERALL_PASS_THROUGH_DEPTH
    if PASS in verdicts:
        return OVERALL_PARTIAL
    return OVERALL_INCOMPLETE


def evaluate(
    subject: Mapping[str, Any],
    *,
    max_stage: str | None = None,
    stop_on_fail: bool = True,
    on_stage: OnStage | None = None,
) -> dict[str, Any]:
    """Run stages cheapest first. Missing evidence SKIPS. SKIP is not a pass.

    The caller supplies `subject["component"]`. This function will not pick
    a representation, organ, or specimen.
    """
    component = require_component(subject)
    planned = _upto(max_stage)
    log: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for stage in planned:
        if stop_reason is not None:
            skipped = _not_run_row(stage, stop_reason)
            skipped["cost"] = _cost_record(skipped)
            log.append(skipped)
            continue
        if on_stage is not None:
            on_stage(stage.id, subject)
        t0 = time.perf_counter()
        try:
            row = run_stage(stage.id, subject)
        except StageInputMissing as exc:
            row = _skip_row(stage, exc.reason)
        except ExpensiveQualificationRefused as exc:
            row = _skip_row(stage, exc.reason)
        row["wall_seconds"] = float(time.perf_counter() - t0)
        row["cost"] = _cost_record(row)
        row["counts_as_pass"] = counts_as_pass(row)
        log.append(row)
        if row["verdict"] == FAIL and stop_on_fail:
            stop_reason = (
                f"terminated after FAIL at {stage.id}; later stages were not spent"
            )

    n_pass = sum(1 for r in log if counts_as_pass(r))
    n_fail = sum(1 for r in log if r["verdict"] == FAIL)
    n_skipped = sum(1 for r in log if r["verdict"] == SKIPPED)
    n_not_run = sum(1 for r in log if r["verdict"] == NOT_RUN)
    overall = _overall(log)
    return {
        "component": component,
        "max_stage": max_stage if max_stage is not None else STAGE_IDS[-1],
        "stop_on_fail": bool(stop_on_fail),
        "stages": log,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skipped": n_skipped,
        "n_not_run": n_not_run,
        "passed_stage_ids": [r["id"] for r in log if counts_as_pass(r)],
        "overall": overall,
        "establishes_capability": False,
        "why_not_capability": (
            "even PASS_THROUGH_DEPTH through FAST_CAPABILITY or "
            "HCLI_MISSION_SUBSET is not qualification, and "
            "EXPENSIVE_QUALIFICATION cannot run on this sidecar. "
            "A SKIPPED stage is excluded from n_pass."
        ),
        "terminated_because": stop_reason,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def fixture_subject() -> dict[str, Any]:
    """Caller-supplied fixture so --build can time the judges. Not a scientific pick."""
    hidden_a = np.ones(5120, dtype=np.float64)
    hidden_b = hidden_a + 1e-6
    logits_a = np.zeros(64, dtype=np.float64)
    logits_a[0] = 5.0
    logits_a[1] = 4.0
    logits_a[2] = 3.0
    logits_b = logits_a.copy()
    unit = ws.emit_hcli_workunit(
        id="future.capability_stages.fixture",
        role="capability_stage_fixture",
        description="fixture work request for HCLI_MISSION_SUBSET timing",
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.capability_stages",
        provider="future.capability_stages",
        effect_class="READ_ONLY",
    )
    return {
        "component": {
            "id": "FIXTURE.caller_supplied",
            "kind": "FIXTURE",
            "note": (
                "build() supplies a fixture so the judges can be timed. "
                "This is not a scientific selection of a resident organ."
            ),
        },
        "hidden_a": hidden_a,
        "hidden_b": hidden_b,
        "logits_a": logits_a,
        "logits_b": logits_b,
        "answers": {
            "fact-capital": {
                "text": "Paris",
                "max_new_tokens": 16,
                "generated_tokens": 2,
                "new_token_ids": [rp.EOS_IM_END_ID],
            },
            "fact-choice": {
                "text": "hbm_doctor.py",
                "max_new_tokens": 16,
                "generated_tokens": 2,
                "new_token_ids": [rp.EOS_IM_END_ID],
            },
            "fact-arith": {
                "text": "323",
                "max_new_tokens": 16,
                "generated_tokens": 1,
                "new_token_ids": [rp.EOS_IM_END_ID],
            },
        },
        "emissions": [unit],
    }


def _py(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    return value


def build() -> Any:
    catalog_rows = catalog()
    report = _py(evaluate(fixture_subject()))
    costs = [_cost_record(row) for row in report["stages"]]
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Staged evaluation API: five increasing depths at which a caller "
            "can ask whether a supplied component is still capable, and stop "
            "cheaply. LOCAL_FUNCTIONAL_FIDELITY is not capability."
        ),
        "question": (
            "What does it cost to ask 'is the model still capable' at five "
            "increasing depths?"
        ),
        "answer": (
            "The judge cost of the four cheap stages is MEASURED wall_seconds "
            "on a caller-supplied fixture in this process. Evidence-production "
            "cost (hidden replay, logits, generate, live mission) is UNMEASURED "
            "because this sidecar does not spawn the resident. "
            "EXPENSIVE_QUALIFICATION is UNMEASURED: qualification_pipeline."
            "execute always refuses and the skip is not a pass."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "chooses_component": False,
        "caller_supplies_component": True,
        "stages": catalog_rows,
        "ladder_order": list(STAGE_IDS),
        "n_stages": len(STAGE_IDS),
        "skip_is_not_a_pass": True,
        "pass_verdict": PASS,
        "non_pass_verdicts": [FAIL, SKIPPED, NOT_RUN],
        "bars_reused": {
            "hidden_cosine_bar": float(cim.HIDDEN_COSINE_BAR),
            "logit_kl_bar": float(acs.LOGIT_KL_BAR),
            "top_k": int(acs.TOPK),
            "top_k_agree_bar": float(acs.TOPK_AGREE_BAR),
            "local_level": fp.MEASURED_LEVEL,
        },
        "fast_probes": [
            {
                "id": p["id"],
                "prompt": p["prompt"],
                "origin": p["origin"],
                "expect_contains": list(p.get("expect_contains") or ()),
                "expect_number": p.get("expect_number"),
            }
            for p in FAST_PROBES
        ],
        "expensive_qualification": expensive_qualification_declaration(),
        "reused_not_rebuilt": {
            LOCAL_FUNCTIONAL_FIDELITY: (
                "capability_information_map._cosine + HIDDEN_COSINE_BAR; "
                "functional_role_probe.MEASURED_LEVEL"
            ),
            LOGIT_TOKEN: (
                "aux_capability_screen.kl_divergence, topk_agreement, "
                "softmax, report_logit_parity"
            ),
            FAST_CAPABILITY: (
                "resident_provider.PROMPT_FRANCE, PROMPT_CHOICE, quality"
            ),
            HCLI_MISSION_SUBSET: (
                "workunit_species.validate_emitted_unit, emit_hcli_workunit"
            ),
            EXPENSIVE_QUALIFICATION: (
                "qualification_pipeline.execute (always refuses); STAGES cited"
            ),
        },
        "fixture_run": report,
        "per_stage_cost": costs,
        "what_the_wall_seconds_are": (
            "perf_counter around each stage judge on the fixture_run subject. "
            "They are not GPU ns, not token_ns, and not a qualified cost."
        ),
        "what_is_not_measured_here": (
            "resident generate time, CPU catalog replay of sealed-3.14, "
            "protected-accelerator-bench wall, GPU lease occupancy. Those "
            "are UNMEASURED with a per-stage reason, not guessed."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Any:
    return build()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true", help="emit the sealed receipt")
    ap.add_argument("--selftest", action="store_true", help="alias of --build")
    args = ap.parse_args(argv)
    if args.build or args.selftest:
        out = build()
        print(out)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
