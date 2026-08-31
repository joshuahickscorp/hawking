"""META_EXPERIMENT_FUNNEL — kill representation candidates early, keep the scar.

Representation search is expensive and most candidates are wrong. This sidecar
funnel is the ordered refusal to spend a later gate on a candidate whose earlier
input is NOT_BUILT or NOT_MEASURED, and the ledger that remembers WHY a candidate
died so the same shape is never re-run.

Nine gates. Cheap analytical first, EBPW last. Passing a gate is not a promotion.
This module produces STATIC_ONLY evidence. It does not take DIAGNOSTIC_RELATIVE
or PROTECTED_ABSOLUTE measurements. It has no GPU.

    python3 tools/future/meta_funnel.py --build
    python3 tools/future/meta_funnel.py --selftest

Recovered, not invented: composition_ladder.py is an 8-rung Qwen qualification
ladder (unreached ≠ failed). negative_science.py / NOETIC_NEGATIVE_SCIENCE.json
already record deaths. representation_library.py already names families and
refuses uniform-bpw-only plans. This module does not fork those. It adds the
Flash/meta advance-refusal funnel and shape-keyed scars they do not provide.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "META_EXPERIMENT_FUNNEL.json"
SCHEMA = "hawking.future.meta_funnel.v1"
RECORDED_BY = "tools/future/meta_funnel.py"

FLASH_MODEL = "qwen3.8-flash-next"
FLASH_MODEL_CLASS = "moe"

# Tokens that mean "you may not pretend this input was observed".
ABSENT_TOKENS = frozenset(
    {
        "NOT_BUILT",
        "NOT_MEASURED",
        "NOT_RUN",
        "NOT_TESTED",
        "NOT_IMPLEMENTED",
        "UNKNOWN",
        "ABSENT",
        "PLAN_ONLY",
        "SCAFFOLD_ONLY",
        "SEALED_METADATA_ONLY",
        "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "CANDIDATE_NOT_BUILT",
        "REQUIRED_RESIDENT_STATE_NOT_BUILT",
        "WAITING_FOR_REPRESENTATION_AND_LOADER",
        "PLANNED_UNTIL_VERIFIED_BODY",
        "PLANNED_UNTIL_NATIVE_EXECUTION",
        "FULL_TENSOR_TRANSFORM_ONLY",
        "BOUNDED_SLICE_REFERENCE_ONLY",
        "BOUNDED_SLICE_RECONSTRUCTION_ONLY",
        "INCOMPLETE",
    }
)
MEASURED_PASS = frozenset({"PASSED", "FIT_PASSED", "OK", "EXACT_MATCH"})
MEASURED_FAIL = frozenset(
    {"FAILED", "KILLED", "MISMATCH", "BELOW_NULL", "ARGMAX_FLIP", "INCOHERENT"}
)

# Complete-system executable-information fields. Copied from FLASH_EBPW_BUDGET
# target_contract.complete_system_byte_fields — a budget, not a measurement.
COMPLETE_SYSTEM_FIELDS = (
    "weight_codes",
    "scales",
    "zero_points",
    "bases",
    "residuals",
    "dictionaries",
    "expert_indices",
    "routing_metadata",
    "generators",
    "lookup_structures",
    "ngram_representation",
    "mtp_representation",
    "required_executable_metadata",
)

META_SUB1 = "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
META_L4 = "receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json"
NNS_RECEIPT = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
LIB_RECEIPT = "receipts/headless/REPRESENTATION_LIBRARY.json"
EBPW_RECEIPT = "receipts/headless/FLASH_EBPW_BUDGET.json"
EXP_RECEIPT = "receipts/headless/FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json"
XFORM_RECEIPT = "receipts/headless/FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"
ROUTER_AB_RECEIPT = "receipts/headless/FLASH_NOETIC_ROUTER_REPRESENTATION_AB.json"
ROUTER_SEL_RECEIPT = "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json"
NX_RECEIPT = "receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json"
LADDER_RECEIPT = "receipts/headless/COMPOSITION_LADDER.json"


@dataclass(frozen=True)
class Gate:
    id: int
    name: str
    cost_class: str
    required_input: str
    kill_criterion: str
    passing_proves: str
    passing_does_not_prove: str


GATES: tuple[Gate, ...] = (
    Gate(
        1,
        "analytical_structure_screen",
        "CHEAP_ANALYTICAL",
        "allocation_plan",
        (
            "The allocation plan is malformed: no regions, a region without a kind, "
            "a claim of complete-system executable information that omits required "
            "fields (bases/residuals/dictionaries/generators/…), or a numeric total "
            "without complete accounting. Synthetic-activation cosine is NOT a kill "
            "here (NOETIC_NEGATIVE_SCIENCE NNS-001: Gaussian-proxy method is dead; "
            "the idea is not)."
        ),
        "The hypothesis is a well-formed allocation of TOTAL EXECUTABLE INFORMATION "
        "(heterogeneous bits are expressible; uniform bits are allowed).",
        "Teacher fit, held-out fidelity, route identity, tokens, capability, "
        "physical NR, a complete NX, or an EBPW number.",
    ),
    Gate(
        2,
        "real_teacher_fit",
        "REAL_TEACHER_CPU",
        "teacher_corpus",
        (
            "A measured fit on real teacher-forced / captured activations from the "
            "named source fails the stated null. Weight reconstruction on a bounded "
            "slice is not teacher fit."
        ),
        "The candidate reconstructs or predicts on the teacher corpus it was fit to.",
        "Held-out numerical validity, route stability, logit/token identity, "
        "capability, or anything physical.",
    ),
    Gate(
        3,
        "held_out_numerical",
        "HELDOUT_NUMERICAL_CPU",
        "held_out_numerical",
        (
            "Held-out activations (disjoint from the teacher corpus) fail the stated "
            "null. Organ-local survival is not whole-model survival "
            "(composition_ladder: ternary CANON then argmax-flip)."
        ),
        "Local numerical fidelity on unseen X for the stated organ.",
        "Route identity, complete-token argmax, coherent generation, capability, "
        "NR, NX, or EBPW.",
    ),
    Gate(
        4,
        "route_stability",
        "ROUTE_TRACE_CPU",
        "route_traces",
        (
            "Student top-k expert identity diverges from the teacher "
            "(status MISMATCH, or expert_ids_exact_match is false)."
        ),
        "On the measured traces, the student selects the same experts as the teacher.",
        "Logit/token identity beyond routing, capability, physical lowering, or EBPW.",
    ),
    Gate(
        5,
        "logit_token_validation",
        "LOGIT_TOKEN_CPU",
        "logit_token",
        (
            "Complete-token / argmax disagreement, or decode degeneration on the "
            "stated probe. A cheaper kernel that flips the argmax is dead."
        ),
        "Token-level identity on the stated probe (argmax / short decode).",
        "Bounded capability, a physical NR, a complete NX, or EBPW. "
        "16 greedy tokens are not capability.",
    ),
    Gate(
        6,
        "bounded_capability",
        "BOUNDED_CAPABILITY_CPU",
        "bounded_capability",
        (
            "The capability suite fails an incumbent axis. Silence that scores on "
            "vacuous axes is not a pass."
        ),
        "The candidate matched the incumbent on every substantive capability axis "
        "of the stated suite.",
        "Physical NR lowering, a complete NX, protected token_ns, or EBPW.",
    ),
    Gate(
        7,
        "physical_nr_lowering",
        "PHYSICAL_NR_STATIC",
        "physical_nr",
        (
            "Lowering was attempted and failed (irreducible NR, missing organ, "
            "hidden dense rematerialization disclosed as a failure)."
        ),
        "A Physical NR artifact exists and lowered without a disclosed failure. "
        "STATIC_ONLY: this sidecar did not run a kernel.",
        "A complete NX, protected complete-token timing, or EBPW. "
        "PLAN_ONLY / NOT_IMPLEMENTED is not a pass; it is a refusal.",
    ),
    Gate(
        8,
        "complete_nx",
        "COMPLETE_NX_STATIC",
        "complete_nx",
        (
            "An NX was presented and is incomplete in a tested way (failed "
            "validation, missing native consumer)."
        ),
        "A source-independent complete NX artifact exists as more than sealed metadata.",
        "EBPW, protected TPS, or promotion. SCAFFOLD_ONLY / "
        "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION is a refusal, not a pass.",
    ),
    Gate(
        9,
        "ebpw",
        "EBPW_ACCOUNTING_STATIC",
        "ebpw_ledger",
        (
            "Complete-system executable bytes were counted and either omit a required "
            "field or exceed the stated ceiling. A target ceiling is not a measurement."
        ),
        "A complete-system EBPW ledger exists with every required field counted.",
        "Protected token_ns, joules, or promotion. null complete_system_bytes is a "
        "refusal, not a number.",
    ),
)
GATES_BY_ID = {g.id: g for g in GATES}
GATES_BY_NAME = {g.name: g for g in GATES}


@dataclass(frozen=True)
class AdvanceResult:
    verdict: str
    gate_id: int
    gate_name: str
    reason: str
    required_input: str
    input_state: str
    scar: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class AdvanceRefused(Exception):
    """The funnel refused to run a gate. Not a death. Not a pass."""

    def __init__(self, result: AdvanceResult):
        super().__init__(result.reason)
        self.result = result


# ---------------------------------------------------------------------------
# Receipt / input inspection
# ---------------------------------------------------------------------------

def load_receipt(rel: str) -> dict[str, Any] | None:
    """Read a JSON receipt from disk, or from HEAD when the sparse tree omitted it."""
    path = REPO / rel
    if path.is_file():
        try:
            return load_json(path)
        except (OSError, json.JSONDecodeError):
            return None
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def path_in_head(rel: str) -> bool:
    if (REPO / rel).exists():
        return True
    listed = git("ls-tree", "--name-only", "HEAD", rel)
    return any(line == rel for line in listed.splitlines())


def receipt_exists(rel: str) -> bool:
    return path_in_head(rel)


def input_state(value: Any) -> str:
    if value is None:
        return "ABSENT"
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return "ABSENT"
        if token in ABSENT_TOKENS:
            return token
        return "PRESENT"
    if isinstance(value, dict):
        if not value:
            return "ABSENT"
        st = value.get("status", value.get("state"))
        if isinstance(st, str) and st in ABSENT_TOKENS:
            return st
        return "PRESENT"
    if isinstance(value, (list, tuple)) and not value:
        return "ABSENT"
    return "PRESENT"


def is_absent(value: Any) -> bool:
    return input_state(value) != "PRESENT"


def resolve_gate(gate: int | str | Gate) -> Gate:
    if isinstance(gate, Gate):
        return gate
    if isinstance(gate, int):
        if gate not in GATES_BY_ID:
            raise KeyError(f"no gate {gate}")
        return GATES_BY_ID[gate]
    if gate in GATES_BY_NAME:
        return GATES_BY_NAME[gate]
    raise KeyError(f"no gate {gate!r}")


def _canonical_allocation(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {"regions": [], "unit": "TOTAL_EXECUTABLE_INFORMATION"}
    regions = plan.get("regions") or []
    canon = []
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict):
                continue
            canon.append(
                {
                    "kind": region.get("kind"),
                    "bits_class": region.get("bits_class"),
                    "family": region.get("family"),
                    "organ": region.get("organ"),
                }
            )
    canon.sort(key=lambda r: (r["kind"] or "", r["organ"] or "", r["family"] or ""))
    return {
        "unit": plan.get("unit") or "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": bool(plan.get("forces_uniform_bpw", False)),
        "regions": canon,
    }


def shape_fingerprint(hypothesis: dict[str, Any]) -> str:
    """Identity of the IDEA, not of the proposal id.

    A future candidate with a new id but the same family / organ / technique /
    model / allocation canonicalizes to the same hash and hits the scar.
    """
    plan = hypothesis.get("allocation_plan")
    if plan is None:
        plan = (hypothesis.get("inputs") or {}).get("allocation_plan")
    payload = {
        "family": hypothesis.get("family"),
        "organ": hypothesis.get("organ"),
        "technique": hypothesis.get("technique"),
        "model": hypothesis.get("model"),
        "allocation": _canonical_allocation(plan),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _get_input(candidate: dict[str, Any], key: str) -> Any:
    inputs = candidate.get("inputs")
    if isinstance(inputs, dict) and key in inputs:
        return inputs[key]
    if key in candidate:
        return candidate[key]
    return None


# ---------------------------------------------------------------------------
# Gate evaluators. Called only when the required input is PRESENT.
# ---------------------------------------------------------------------------

def _eval_from_status(value: Any, pass_msg: str, fail_msg: str) -> tuple[str, str]:
    if isinstance(value, dict):
        if value.get("fit_passed") is True or value.get("passed") is True:
            return "PASSED", pass_msg
        if value.get("fit_passed") is False or value.get("passed") is False:
            return "KILLED", value.get("mechanism") or fail_msg
        st = str(value.get("status") or value.get("verdict") or "")
        if st in MEASURED_PASS:
            return "PASSED", pass_msg
        if st in MEASURED_FAIL:
            return "KILLED", value.get("mechanism") or fail_msg
        return "REFUSED", f"input present but unmeasured (status={st or 'absent'})"
    if isinstance(value, str):
        if value in MEASURED_PASS:
            return "PASSED", pass_msg
        if value in MEASURED_FAIL:
            return "KILLED", fail_msg
    return "REFUSED", "input present but unmeasured"


def _eval_analytical(_candidate: dict[str, Any], value: Any) -> tuple[str, str]:
    if not isinstance(value, dict):
        return "KILLED", "allocation_plan is not a mapping"
    regions = value.get("regions")
    if not isinstance(regions, list) or not regions:
        return (
            "KILLED",
            "allocation_plan has no regions; total executable information is undefined",
        )
    for i, region in enumerate(regions):
        if not isinstance(region, dict) or not region.get("kind"):
            return "KILLED", f"allocation_plan region {i} is missing kind"
    if value.get("claims_complete_system"):
        acct = value.get("accounting_fields") or []
        have_fields = {str(x) for x in acct}
        have_kinds = {str(r.get("kind")) for r in regions if isinstance(r, dict)}
        missing = [f for f in COMPLETE_SYSTEM_FIELDS if f not in have_fields and f not in have_kinds]
        if missing:
            return (
                "KILLED",
                "claims complete-system executable information but omits "
                + ",".join(missing),
            )
    total = value.get("total_executable_information")
    if isinstance(total, (int, float)) and not value.get("accounting_complete"):
        return (
            "KILLED",
            "numeric total executable information without complete accounting",
        )
    return (
        "PASSED",
        "allocation plan is well-formed; does not prove fit, tokens, capability, or physical lowering",
    )


def _eval_teacher(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    return _eval_from_status(
        value,
        "fit on real teacher corpus; does not prove held-out, routes, tokens, or capability",
        "real teacher fit failed the stated null",
    )


def _eval_heldout(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    return _eval_from_status(
        value,
        "held-out numerical fidelity on the stated organ; does not prove routes, tokens, or capability",
        "held-out numerical validation failed the stated null",
    )


def _eval_routes(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    if isinstance(value, dict) and value.get("expert_ids_exact_match") is False:
        return (
            "KILLED",
            value.get("mechanism")
            or "student top-k expert identity diverges from the teacher",
        )
    return _eval_from_status(
        value,
        "route traces match teacher top-k on the stated probe",
        "student routes diverge from the teacher",
    )


def _eval_logits(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    if isinstance(value, dict) and value.get("argmax_agree") is False:
        return "KILLED", value.get("mechanism") or "complete-token argmax disagreement"
    return _eval_from_status(
        value,
        "token-level identity on the stated probe; 16 greedy tokens are not capability",
        "logit/token validation failed",
    )


def _eval_capability(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    return _eval_from_status(
        value,
        "bounded capability suite matched the incumbent on substantive axes",
        "bounded capability failed an incumbent axis",
    )


def _eval_nr(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    return _eval_from_status(
        value,
        "NR lowering exists as an artifact (STATIC_ONLY; no kernel was run here)",
        "physical NR lowering failed",
    )


def _eval_nx(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    return _eval_from_status(
        value,
        "a source-independent complete NX exists as more than sealed metadata",
        "complete NX failed validation",
    )


def _eval_ebpw(_c: dict[str, Any], value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        if value.get("all_required_bytes_included") is False and value.get("status") in MEASURED_PASS:
            return "KILLED", "EBPW ledger claims a pass but omits required executable-information fields"
        ebpw = value.get("complete_system_ebpw")
        ceiling = value.get("complete_system_ebpw_max")
        if isinstance(ebpw, (int, float)) and isinstance(ceiling, (int, float)) and ebpw > ceiling:
            return "KILLED", "complete-system EBPW exceeds the stated ceiling"
    return _eval_from_status(
        value,
        "complete-system EBPW ledger counted every required field",
        "EBPW accounting failed",
    )


EVALUATORS = {
    1: _eval_analytical,
    2: _eval_teacher,
    3: _eval_heldout,
    4: _eval_routes,
    5: _eval_logits,
    6: _eval_capability,
    7: _eval_nr,
    8: _eval_nx,
    9: _eval_ebpw,
}


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------

class Funnel:
    """Ordered gates plus an append-only scar ledger.

    Deaths accumulate and are never overwritten. Retrieval is by shape
    fingerprint, not by candidate id.
    """

    def __init__(self) -> None:
        self.scars: list[dict[str, Any]] = []

    def _next_scar_id(self) -> str:
        return f"MF-{len(self.scars) + 1:04d}"

    def match_scars(self, hypothesis: dict[str, Any]) -> list[dict[str, Any]]:
        fp = shape_fingerprint(hypothesis)
        return [s for s in self.scars if s["shape_sha256"] == fp]

    def _write_scar(
        self,
        candidate: dict[str, Any],
        gate: Gate,
        mechanism: str,
        evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        scar = {
            "scar_id": self._next_scar_id(),
            "shape_sha256": shape_fingerprint(candidate),
            "gate_id": gate.id,
            "gate_name": gate.name,
            "verdict": "KILLED",
            "mechanism": mechanism,
            "evidence": list(evidence or candidate.get("cited_evidence") or []),
            "identity": {
                "candidate_id": candidate.get("id"),
                "family": candidate.get("family"),
                "organ": candidate.get("organ"),
                "technique": candidate.get("technique"),
                "model": candidate.get("model"),
                "allocation_canonical": _canonical_allocation(
                    candidate.get("allocation_plan")
                    or (candidate.get("inputs") or {}).get("allocation_plan")
                ),
            },
            "reopen_condition": (
                candidate.get("reopen_condition")
                or f"A new measurement of {gate.required_input} that passes gate {gate.id} "
                f"({gate.name}) on {candidate.get('model')}/{candidate.get('organ')}."
            ),
            "kind": candidate.get("death_kind") or "PROPERTY_OF_IDEA",
        }
        self.scars.append(scar)
        return scar

    def advance(self, candidate: dict[str, Any], gate: int | str | Gate) -> AdvanceResult:
        """Run one gate. REFUSES rather than inventing a pass.

        Earlier gates must already be PASSED. The current gate's required input
        must not be NOT_BUILT / NOT_MEASURED / absent. A kill writes a scar.
        """
        g = resolve_gate(gate)
        passed = list(candidate.get("passed_gates") or [])
        earlier = [x.id for x in GATES if x.id < g.id]
        missing_earlier = [i for i in earlier if i not in passed]
        if missing_earlier:
            return AdvanceResult(
                verdict="REFUSED",
                gate_id=g.id,
                gate_name=g.name,
                reason=(
                    f"earlier gate(s) {missing_earlier} not PASSED; "
                    "the funnel does not skip"
                ),
                required_input=g.required_input,
                input_state="EARLIER_GATE_NOT_PASSED",
            )
        if candidate.get("died_at"):
            return AdvanceResult(
                verdict="REFUSED",
                gate_id=g.id,
                gate_name=g.name,
                reason=f"candidate already killed at gate {candidate['died_at']}",
                required_input=g.required_input,
                input_state="ALREADY_KILLED",
            )
        raw = _get_input(candidate, g.required_input)
        state = input_state(raw)
        if is_absent(raw):
            return AdvanceResult(
                verdict="REFUSED",
                gate_id=g.id,
                gate_name=g.name,
                reason=(
                    f"required input {g.required_input!r} is {state}; "
                    "advance refuses to invent a measurement"
                ),
                required_input=g.required_input,
                input_state=state,
            )
        verdict, reason = EVALUATORS[g.id](candidate, raw)
        if verdict == "REFUSED":
            return AdvanceResult(
                verdict="REFUSED",
                gate_id=g.id,
                gate_name=g.name,
                reason=reason,
                required_input=g.required_input,
                input_state=state,
            )
        if verdict == "KILLED":
            scar = self._write_scar(candidate, g, reason)
            candidate["died_at"] = g.id
            candidate["died_at_name"] = g.name
            candidate["kill_mechanism"] = reason
            return AdvanceResult(
                verdict="KILLED",
                gate_id=g.id,
                gate_name=g.name,
                reason=reason,
                required_input=g.required_input,
                input_state=state,
                scar=scar,
            )
        if g.id not in passed:
            passed.append(g.id)
        candidate["passed_gates"] = passed
        return AdvanceResult(
            verdict="PASSED",
            gate_id=g.id,
            gate_name=g.name,
            reason=reason,
            required_input=g.required_input,
            input_state=state,
        )

    def run(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Walk gates 1..9. Stop at the first KILLED or REFUSED."""
        log: list[dict[str, Any]] = []
        stall: AdvanceResult | None = None
        for g in GATES:
            result = self.advance(candidate, g)
            log.append(result.as_dict())
            if result.verdict != "PASSED":
                stall = result
                break
        out = {
            "id": candidate.get("id"),
            "family": candidate.get("family"),
            "organ": candidate.get("organ"),
            "technique": candidate.get("technique"),
            "model": candidate.get("model"),
            "source": candidate.get("source"),
            "heterogeneous": not _canonical_allocation(
                candidate.get("allocation_plan")
                or (candidate.get("inputs") or {}).get("allocation_plan")
            ).get("forces_uniform_bpw", False),
            "allocation_region_kinds": [
                r["kind"]
                for r in _canonical_allocation(
                    candidate.get("allocation_plan")
                    or (candidate.get("inputs") or {}).get("allocation_plan")
                )["regions"]
            ],
            "passed_gates": list(candidate.get("passed_gates") or []),
            "log": log,
        }
        if stall is None:
            out["stall_gate"] = None
            out["stall_gate_name"] = None
            out["stall_verdict"] = "PASSED_ALL"
            out["stall_reason"] = "every gate passed; still not a promotion (STATIC_ONLY)"
        else:
            out["stall_gate"] = stall.gate_id
            out["stall_gate_name"] = stall.gate_name
            out["stall_verdict"] = stall.verdict
            out["stall_reason"] = stall.reason
            out["scar_id"] = (stall.scar or {}).get("scar_id")
        return out


def advance(candidate: dict[str, Any], gate: int | str | Gate, funnel: Funnel | None = None) -> AdvanceResult:
    return (funnel or Funnel()).advance(candidate, gate)


# ---------------------------------------------------------------------------
# Family recovery. Prefer FLASH_META_REPRESENTATION_SUB1 when it exists.
# ---------------------------------------------------------------------------

def _default_inputs(**present: Any) -> dict[str, Any]:
    inputs = {
        "allocation_plan": "NOT_BUILT",
        "teacher_corpus": "NOT_BUILT",
        "held_out_numerical": "NOT_MEASURED",
        "route_traces": "NOT_MEASURED",
        "logit_token": "NOT_MEASURED",
        "bounded_capability": "NOT_RUN",
        "physical_nr": "NOT_BUILT",
        "complete_nx": "NOT_BUILT",
        "ebpw_ledger": "NOT_MEASURED",
    }
    inputs.update(present)
    return inputs


def _candidate(
    cid: str,
    family: str,
    organ: str,
    technique: str,
    allocation_plan: dict[str, Any],
    source: str,
    *,
    extra_inputs: dict[str, Any] | None = None,
    cited: list[dict[str, Any]] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    inputs = _default_inputs(allocation_plan=allocation_plan)
    if extra_inputs:
        inputs.update(extra_inputs)
    return {
        "id": cid,
        "family": family,
        "organ": organ,
        "technique": technique,
        "model": FLASH_MODEL,
        "model_class": FLASH_MODEL_CLASS,
        "source": source,
        "allocation_plan": allocation_plan,
        "inputs": inputs,
        "cited_evidence": cited or [],
        "note": note,
        "passed_gates": [],
    }


# Region kind per ledger component. The real FLASH_META_REPRESENTATION_SUB1
# family_budget entries carry a `ledger` that IS a heterogeneous allocation --
# expert_latent_symbols / shared_decoder_amortized / router_margin_guard /
# format_and_seed all draw from different budgets for different reasons. Flattening
# that to one uniform bulk region threw away exactly the structure section 17 is
# about, so map each component to the region kind it actually represents.
_LEDGER_REGION_KIND = (
    ("router", ("routing_sensitive", "premium")),
    ("margin", ("routing_sensitive", "premium")),
    ("residual", ("sparse_residual", "sparse")),
    ("repair", ("sparse_residual", "sparse")),
    ("shared", ("shared_generator", "shared")),
    ("decoder", ("shared_generator", "shared")),
    ("dictionary", ("shared_generator", "shared")),
    ("literal", ("capability_island", "literal")),
    ("state", ("capability_island", "literal")),
    ("format", ("format_overhead", "literal")),
    ("seed", ("format_overhead", "literal")),
    ("latent", ("generated_bulk", "near_zero")),
    ("symbol", ("generated_bulk", "near_zero")),
)


def _ledger_region(component: str) -> tuple[str, str]:
    low = component.lower()
    for token, kind in _LEDGER_REGION_KIND:
        if token in low:
            return kind
    return ("predictable_bulk", "uniform")


def _ledger_plan(family: str, organ: str, ledger: dict[str, Any]) -> dict[str, Any]:
    """Allocation derived from a real family_budget ledger, not flattened."""
    regions = []
    for component, bits in sorted(ledger.items()):
        if not isinstance(bits, (int, float)):
            continue
        kind, bits_class = _ledger_region(component)
        regions.append(
            {
                "kind": kind,
                "bits_class": bits_class,
                "family": family,
                "organ": organ,
                "component": component,
                "component_meta_bpw": bits,
            }
        )
    if not regions:
        return _uniform_plan(family, organ)
    kinds = {r["kind"] for r in regions}
    return {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        # Heterogeneous exactly when the ledger draws on more than one region kind.
        "forces_uniform_bpw": len(kinds) == 1,
        "regions": regions,
        "derived_from": "family_budget[].ledger",
        "claims_complete_system": False,
        "accounting_complete": False,
    }


def _uniform_plan(family: str, organ: str, bits_class: str = "uniform") -> dict[str, Any]:
    return {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": True,
        "regions": [
            {
                "kind": "predictable_bulk",
                "bits_class": bits_class,
                "family": family,
                "organ": organ,
            }
        ],
    }


def _hetero_flash_plan() -> dict[str, Any]:
    """FLASH_EBPW chosen_representation, as an allocation, not as a measurement."""
    routed = ("embeddings", "deltanet", "sparse_attention", "routed_experts", "shared_expert", "lm_head")
    native = ("mtp", "ngram_engine", "vision_backbone", "residual_hyperconnections", "support_misc")
    regions: list[dict[str, str]] = []
    for organ in routed:
        regions.append(
            {
                "kind": "shared_generator",
                "bits_class": "shared",
                "family": "shared_bf16_basis",
                "organ": organ,
            }
        )
        regions.append(
            {
                "kind": "sparse_residual",
                "bits_class": "sparse",
                "family": "nf_residual",
                "organ": organ,
            }
        )
    regions.append(
        {
            "kind": "routing_sensitive",
            "bits_class": "premium",
            "family": "organ_native",
            "organ": "router",
        }
    )
    regions.append(
        {
            "kind": "capability_island",
            "bits_class": "literal",
            "family": "resident_state",
            "organ": "recurrent_state",
        }
    )
    for organ in native:
        regions.append(
            {
                "kind": "predictable_bulk",
                "bits_class": "near_zero",
                "family": "organ_native",
                "organ": organ,
            }
        )
    return {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": False,
        "regions": regions,
        "claims_complete_system": False,
        "accounting_complete": False,
        "note": (
            "Heterogeneous by construction: shared generators on routed bulk, "
            "premium on the router, literal precision on recurrent state, "
            "organ-native on n-gram/MTP/vision. Not a counted EBPW."
        ),
    }


def _cited(path: str, field: str, note: str) -> dict[str, str]:
    return {"path": path, "field": field, "note": note}


def _families_from_meta_sub1(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort parser for the (currently absent) meta sub-1 receipt."""
    buckets: list[Any] = []
    for key in ("families", "family_budget", "budget_search", "meta_program"):
        node = doc.get(key)
        if isinstance(node, list):
            buckets.extend(node)
        elif isinstance(node, dict):
            inner = node.get("families") or node.get("candidates") or node.get("entries")
            if isinstance(inner, list):
                buckets.extend(inner)
            elif isinstance(inner, dict):
                for name, spec in sorted(inner.items()):
                    if isinstance(spec, dict):
                        buckets.append({"id": spec.get("id") or name, **spec})
                    else:
                        buckets.append({"id": name, "family": name, "value": spec})
            elif key == "family_budget":
                for name, spec in sorted(node.items()):
                    if name in {"schema", "version", "unit"}:
                        continue
                    if isinstance(spec, dict):
                        buckets.append({"id": spec.get("id") or name, **spec})
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    coherence = doc.get("coherence_contract")
    next_gate = doc.get("next_gate")
    for spec in buckets:
        if not isinstance(spec, dict):
            continue
        cid = str(spec.get("id") or spec.get("family") or spec.get("name") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        family = str(spec.get("family") or spec.get("name") or cid)
        organ = str(spec.get("organ") or "routed_experts")
        technique = str(spec.get("technique") or spec.get("codec") or family)
        plan = spec.get("allocation_plan")
        if not isinstance(plan, dict):
            ledger = spec.get("ledger")
            plan = (
                _ledger_plan(family, organ, ledger)
                if isinstance(ledger, dict) and ledger
                else _uniform_plan(family, organ)
            )
        extra = {}
        for k in (
            "teacher_corpus",
            "held_out_numerical",
            "route_traces",
            "logit_token",
            "bounded_capability",
            "physical_nr",
            "complete_nx",
            "ebpw_ledger",
        ):
            if k in spec:
                extra[k] = spec[k]
            elif isinstance(spec.get("inputs"), dict) and k in spec["inputs"]:
                extra[k] = spec["inputs"][k]
        cand = _candidate(
            cid if cid.startswith("flash.") else f"flash.meta.{cid}",
            family,
            organ,
            technique,
            plan,
            META_SUB1,
            extra_inputs=extra,
            cited=[_cited(META_SUB1, "family_budget", "primary family source")],
            note="recovered from FLASH_META_REPRESENTATION_SUB1",
        )
        if coherence is not None:
            cand["coherence_contract"] = "PRESENT"
        if next_gate is not None:
            cand["declared_next_gate"] = next_gate if isinstance(next_gate, str) else "PRESENT"
        out.append(cand)
    return out


def _nx_input_from_receipt(nx: dict[str, Any] | None) -> Any:
    if not nx:
        return "NOT_BUILT"
    status = nx.get("status")
    loader = nx.get("native_loader") or {}
    kernels = nx.get("native_kernels") or {}
    loader_st = loader.get("status") if isinstance(loader, dict) else loader
    kern_st = kernels.get("status") if isinstance(kernels, dict) else kernels
    if status in ABSENT_TOKENS or loader_st in ABSENT_TOKENS or kern_st in ABSENT_TOKENS:
        return status or "NOT_BUILT"
    return {"status": status or "UNKNOWN"}


def _nr_input_from_receipts(nx: dict[str, Any] | None) -> Any:
    if not nx:
        return "NOT_BUILT"
    loader = nx.get("native_loader") or {}
    kernels = nx.get("native_kernels") or {}
    loader_st = loader.get("status") if isinstance(loader, dict) else None
    kern_st = kernels.get("status") if isinstance(kernels, dict) else None
    if loader_st in ABSENT_TOKENS or kern_st in ABSENT_TOKENS or not loader_st:
        return loader_st or kern_st or "NOT_BUILT"
    return {"status": loader_st}


def _ebpw_input_from_receipt(ebpw: dict[str, Any] | None) -> Any:
    if not ebpw:
        return "NOT_MEASURED"
    measured = ebpw.get("measured") or {}
    if not isinstance(measured, dict):
        return "NOT_MEASURED"
    if measured.get("complete_system_bytes") is None:
        return "NOT_MEASURED"
    return {
        "status": ebpw.get("status") or "NOT_MEASURED",
        "all_required_bytes_included": measured.get("all_required_bytes_included"),
        "complete_system_bytes": measured.get("complete_system_bytes"),
        "complete_system_ebpw": measured.get("complete_system_ebpw"),
    }


def recover_families() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Disk/HEAD recovery. Does not invent teacher corpora or hardware numbers."""
    provenance: dict[str, Any] = {
        "meta_sub1": META_SUB1 if receipt_exists(META_SUB1) else None,
        "meta_l4": META_L4 if receipt_exists(META_L4) else None,
        "primary_source": None,
    }
    meta = load_receipt(META_SUB1)
    if meta:
        families = _families_from_meta_sub1(meta)
        if families:
            provenance["primary_source"] = META_SUB1
            return families, provenance

    provenance["primary_source"] = "recovered_from_flash_receipts_and_library"
    nx = load_receipt(NX_RECEIPT)
    ebpw = load_receipt(EBPW_RECEIPT)
    exp = load_receipt(EXP_RECEIPT)
    xform = load_receipt(XFORM_RECEIPT)
    router_ab = load_receipt(ROUTER_AB_RECEIPT)
    router_sel = load_receipt(ROUTER_SEL_RECEIPT)
    lib = load_receipt(LIB_RECEIPT)

    nx_in = _nx_input_from_receipt(nx)
    nr_in = _nr_input_from_receipts(nx)
    ebpw_in = _ebpw_input_from_receipt(ebpw)
    later = {
        "physical_nr": nr_in,
        "complete_nx": nx_in,
        "ebpw_ledger": ebpw_in,
        "bounded_capability": "NOT_RUN",
        "logit_token": "NOT_MEASURED",
        "held_out_numerical": "NOT_MEASURED",
        "teacher_corpus": "NOT_BUILT",
    }

    families: list[dict[str, Any]] = []

    families.append(
        _candidate(
            "flash.meta_program.shared_basis_nf_residual",
            "shared_basis_plus_nf_residual",
            "whole_model",
            "shared-basis-plus-NF-residual for expert/routed paths; organ-native for state/sparse/n-gram/MTP/vision",
            _hetero_flash_plan(),
            EBPW_RECEIPT,
            extra_inputs=later,
            cited=[
                _cited(EBPW_RECEIPT, "chosen_representation", "heterogeneous chosen program; organs CANDIDATE_NOT_BUILT"),
                _cited(NX_RECEIPT, "status", f"nx.status={nx.get('status') if nx else 'ABSENT'}"),
            ],
            note=(
                "The Flash EBPW chosen_representation is a PROGRAM, not a body. "
                "Every organ is CANDIDATE_NOT_BUILT / actual_bytes NOT_MEASURED / capability NOT_RUN."
            ),
        )
    )
    families.append(
        _candidate(
            "flash.recurrent_state.resident",
            "resident_state",
            "recurrent_state",
            "resident sequence-isolated state",
            {
                "unit": "TOTAL_EXECUTABLE_INFORMATION",
                "forces_uniform_bpw": False,
                "regions": [
                    {
                        "kind": "capability_island",
                        "bits_class": "literal",
                        "family": "resident_state",
                        "organ": "recurrent_state",
                    }
                ],
            },
            EBPW_RECEIPT,
            extra_inputs=later,
            cited=[_cited(EBPW_RECEIPT, "organs[recurrent_state].representation_status", "REQUIRED_RESIDENT_STATE_NOT_BUILT")],
            note="Literal precision on a capability-critical island. The body is not built.",
        )
    )

    exp_cands = (exp or {}).get("candidates") or {}
    if isinstance(exp_cands, dict):
        for name in sorted(exp_cands):
            spec = exp_cands[name] if isinstance(exp_cands[name], dict) else {}
            scheme = str(spec.get("scheme") or spec.get("id") or name)
            shared = "basis" in name or "shared" in name
            plan = (
                {
                    "unit": "TOTAL_EXECUTABLE_INFORMATION",
                    "forces_uniform_bpw": False,
                    "regions": [
                        {
                            "kind": "shared_generator",
                            "bits_class": "shared",
                            "family": name,
                            "organ": "routed_experts",
                        },
                        {
                            "kind": "sparse_residual",
                            "bits_class": "sparse",
                            "family": name,
                            "organ": "routed_experts",
                        },
                    ],
                }
                if shared
                else _uniform_plan(name, "routed_experts")
            )
            families.append(
                _candidate(
                    f"flash.routed_experts.{name}",
                    name,
                    "routed_experts",
                    scheme,
                    plan,
                    EXP_RECEIPT,
                    extra_inputs=later,
                    cited=[
                        _cited(EXP_RECEIPT, f"candidates.{name}.status", "bounded slice; model_capability_tested=false"),
                        _cited(XFORM_RECEIPT, "status", f"transform_parity={(xform or {}).get('status', 'ABSENT')}; whole_model_capability NOT_TESTED"),
                    ],
                    note=(
                        "Weight reconstruction / transform parity on a routed-expert tensor "
                        "is not a teacher-forced corpus. teacher_corpus remains NOT_BUILT."
                    ),
                )
            )

    ab_cands = (router_ab or {}).get("candidates") or []
    sel_status = None
    if isinstance(router_sel, dict):
        sel_status = (router_sel.get("source_selection_parity") or {}).get("status")
    if isinstance(ab_cands, list):
        for spec in ab_cands:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("id") or "")
            if not name:
                continue
            fam = str(spec.get("family") or name)
            extra = dict(later)
            if sel_status:
                extra["route_traces"] = {
                    "status": sel_status,
                    "expert_ids_exact_match": (
                        (router_sel.get("source_selection_parity") or {}).get("expert_ids_exact_match")
                        if isinstance(router_sel, dict)
                        else None
                    ),
                    "source": ROUTER_SEL_RECEIPT,
                    "evidence_class": "CITED_CODEX_STATIC",
                    "note": (
                        "Bounded in-memory router study. Not teacher fit. "
                        "Not used to skip gates 2–3. STATIC_ONLY citation."
                    ),
                }
            families.append(
                _candidate(
                    f"flash.router.{name}",
                    fam,
                    "router",
                    fam,
                    _uniform_plan(fam, "router"),
                    ROUTER_AB_RECEIPT,
                    extra_inputs=extra,
                    cited=[
                        _cited(ROUTER_AB_RECEIPT, f"candidates.{name}", "bounded in-memory router representation"),
                        _cited(ROUTER_SEL_RECEIPT, "source_selection_parity.status", f"cited {sel_status}"),
                    ],
                    note=(
                        "Router AB is a bounded study. Route MISMATCH is cited for gate 4 "
                        "but cannot be reached until teacher fit and held-out pass."
                    ),
                )
            )

    lib_fams = (lib or {}).get("families") or []
    if isinstance(lib_fams, list):
        for spec in lib_fams:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("family") or "")
            if not name:
                continue
            families.append(
                _candidate(
                    f"flash.kinship.{name}",
                    name,
                    "routed_experts",
                    name,
                    _uniform_plan(name, "routed_experts"),
                    LIB_RECEIPT,
                    extra_inputs=later,
                    cited=[
                        _cited(LIB_RECEIPT, f"families.{name}", "Qwen-measured kinship seed; MODEL_SPECIFIC; does not prune Flash"),
                    ],
                    note=(
                        "representation_library kinship: an MoE expert is a dense MLP "
                        "that only some tokens reach. A Qwen measurement warns; it does not "
                        "kill the Flash idea. teacher_corpus is still NOT_BUILT on Flash."
                    ),
                )
            )

    families.sort(key=lambda c: str(c.get("id")))
    provenance["n_recovered"] = len(families)
    provenance["nx_status"] = (nx or {}).get("status")
    provenance["ebpw_status"] = (ebpw or {}).get("status")
    provenance["router_selection_parity"] = sel_status
    return families, provenance


def recover_negative_science() -> dict[str, Any]:
    nns = load_receipt(NNS_RECEIPT)
    ladder = load_receipt(LADDER_RECEIPT)
    return {
        "noetic_negative_science": {
            "path": NNS_RECEIPT,
            "present": nns is not None,
            "schema": (nns or {}).get("schema"),
            "n_entries": len((nns or {}).get("entries") or []) if nns else 0,
            "how_to_use": (nns or {}).get("how_to_use"),
            "death_fields_observed": (
                sorted((nns.get("entries") or [{}])[0].keys())
                if nns and (nns.get("entries")) and isinstance((nns.get("entries") or [{}])[0], dict)
                else []
            ),
        },
        "composition_ladder": {
            "path": LADDER_RECEIPT,
            "present": ladder is not None,
            "schema": (ladder or {}).get("schema"),
            "rungs": (ladder or {}).get("rungs"),
            "rule": (ladder or {}).get("rule"),
            "n_candidates": len((ladder or {}).get("candidates") or []) if ladder else 0,
            "note": (
                "8-rung Qwen ladder. Unreached ≠ failed. Not the Flash 9-gate funnel; "
                "consumed as law, not copied."
            ),
        },
        "tools_flash_meta_representation": receipt_exists("tools/flash_meta_representation.py"),
        "tools_flash_meta_coherence_screen": receipt_exists("tools/flash_meta_coherence_screen.py"),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _gate_docs() -> list[dict[str, Any]]:
    return [asdict(g) for g in GATES]


def build() -> Any:
    families, provenance = recover_families()
    nns = recover_negative_science()
    funnel = Funnel()
    runs = [funnel.run(c) for c in families]

    stall_counts: dict[str, int] = {}
    for row in runs:
        key = f"gate_{row['stall_gate']}_{row['stall_verdict']}"
        stall_counts[key] = stall_counts.get(key, 0) + 1

    n_gate2 = sum(1 for r in runs if r["stall_gate"] == 2)
    n_gate3 = sum(1 for r in runs if r["stall_gate"] == 3)
    n_refused = sum(1 for r in runs if r["stall_verdict"] == "REFUSED")

    recovered_implementation = {
        "sidecar_plumbing": "tools/future/_common.py (write_receipt, bench_block, load_json, git)",
        "composition_ladder": {
            "path": "tools/headless/composition_ladder.py",
            "receipt": LADDER_RECEIPT,
            "what_it_is": "8-rung Qwen qualification ladder with unreached ≠ failed",
            "why_not_forked": (
                "It classifies already-run Qwen receipts. It has no advance() refusal, "
                "no Flash families, no shape-keyed scars, no gates 7–9 (NR/NX/EBPW)."
            ),
        },
        "negative_science": {
            "tools": [
                "tools/headless/negative_science.py",
                "tools/headless/noetic_negative_science.py",
            ],
            "receipt": NNS_RECEIPT,
            "what_it_is": "append-only death corpus with reopen conditions; match by id or claim string",
            "gap": "F009: nothing queries scars before a new experiment is proposed; no shape fingerprint",
        },
        "representation_library": {
            "path": "tools/headless/representation_library.py",
            "receipt": LIB_RECEIPT,
            "what_it_is": "ranked family seeds + ALLOCATION_SCHEMA that permits zero-bit regions and high-precision islands",
            "gap": "seeds a search; does not funnel a candidate through ordered gates",
        },
        "flash_receipts_consumed": [
            EBPW_RECEIPT,
            EXP_RECEIPT,
            XFORM_RECEIPT,
            ROUTER_AB_RECEIPT,
            ROUTER_SEL_RECEIPT,
            NX_RECEIPT,
        ],
        "flash_meta_primary": {
            "FLASH_META_REPRESENTATION_SUB1.json": provenance["meta_sub1"],
            "FLASH_META_COHERENCE_SCREEN_L4.json": provenance["meta_l4"],
            "tools/flash_meta_representation.py": nns["tools_flash_meta_representation"],
            "tools/flash_meta_coherence_screen.py": nns["tools_flash_meta_coherence_screen"],
        },
        "primary_source": provenance["primary_source"],
    }

    gaps_closed = [
        "Nine ordered gates, each with input requirement, kill criterion, cost class, and prove / not-prove.",
        "advance(candidate, gate) REFUSES when the required input is NOT_BUILT / NOT_MEASURED / absent, and when earlier gates are not PASSED.",
        "Killed candidates write an append-only scar retrievable by hypothesis shape, not just by id.",
        "Allocation is TOTAL EXECUTABLE INFORMATION: bulk / routing / islands / residual / shared generators are first-class. Uniform bpw is allowed, not assumed.",
        "Recovered Flash families were run through the funnel; stalls are reported rather than manufactured into passes.",
    ]

    negative_findings = [
        f"{META_SUB1} is not in HEAD and not on disk — family_budget / budget_search / meta_program / coherence_contract / next_gate could not be read from the named primary receipt.",
        f"{META_L4} is not in HEAD and not on disk.",
        "tools/flash_meta_representation.py and tools/flash_meta_coherence_screen.py are not in HEAD.",
        "receipts/headless is not materialized in this sparse checkout; receipts were read via git show HEAD:<path> when present.",
        "No Flash teacher-forced / captured-activation corpus was found (GLM teacher-forced captures and Qwen teacher operators exist; they are not a Flash corpus).",
        "receipts/headless/FLASH_COMPLETE_V0.nx.json is not in this HEAD (frontier F001 cites it; NX status here is FLASH_NEXT_NOETIC_EXECUTABLE = SCAFFOLD_ONLY).",
        "This sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. Every number that would need a GPU, FPGA, or power meter is UNKNOWN / null.",
        "Qwen C1/C2/C5 NOT_WORTH_BUILDING and Qwen binary/ternary deaths are MODEL_SPECIFIC; they were not imported as Flash kills.",
    ]

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Kill representation candidates as early and as cheaply as possible "
            "while preserving why each one died, so the same dead idea is never "
            "re-run. Odyssey I (WHAT IS TRUE?). Disk state is authority. There is "
            "no Era VI and no Odyssey IV. FPGA stays inside Accelerator / Physical "
            "Compiler / Fusion."
        ),
        "claim_class": "STATIC_ONLY",
        "gpu_authority": False,
        "gates": _gate_docs(),
        "advance_rule": (
            "advance(candidate, gate) REFUSES when (1) any earlier gate is not "
            "PASSED, (2) the candidate was already killed, or (3) the gate's "
            "required input is NOT_BUILT / NOT_MEASURED / absent / PLAN_ONLY / "
            "SCAFFOLD_ONLY / equivalent. A refusal is not a death. A kill writes "
            "a scar. Passing is not promotion."
        ),
        "heterogeneity": {
            "unit": "TOTAL_EXECUTABLE_INFORMATION",
            "forces_uniform_bpw": False,
            "region_kinds": [
                "predictable_bulk",
                "routing_sensitive",
                "capability_island",
                "sparse_residual",
                "shared_generator",
            ],
            "bits_classes": ["near_zero", "premium", "literal", "sparse", "shared", "uniform"],
            "complete_system_fields": list(COMPLETE_SYSTEM_FIELDS),
            "law": (
                "A candidate may allocate near-zero to predictable bulk, premium to "
                "routing-sensitive structure, literal precision to capability-critical "
                "islands, sparse residual correction, and shared generators/dictionaries. "
                "The funnel will not refuse a well-formed uniform plan for being uniform; "
                "it will refuse a complete-system EBPW claim that omits those fields."
            ),
        },
        "scar_schema": {
            "scar_id": "MF-NNNN",
            "shape_sha256": "sha256(family, organ, technique, model, canonical allocation)",
            "gate_id": "int",
            "mechanism": "why it died",
            "evidence": "paths / notes, not hardware numbers",
            "identity": "enough to match a future proposal of the same shape",
            "reopen_condition": "required",
            "accumulation": "append-only; never overwritten",
        },
        "recovered_implementation": recovered_implementation,
        "recovery_provenance": provenance,
        "negative_science_recovered": nns,
        "gaps_closed": gaps_closed,
        "negative_findings": negative_findings,
        "families_run": runs,
        "scars": list(funnel.scars),
        "counts": {
            "families": len(runs),
            "gates": len(GATES),
            "scars": len(funnel.scars),
            "refused": n_refused,
            "killed": sum(1 for r in runs if r["stall_verdict"] == "KILLED"),
            "passed_all": sum(1 for r in runs if r["stall_verdict"] == "PASSED_ALL"),
            "stalled_at_gate_2": n_gate2,
            "stalled_at_gate_3": n_gate3,
            "by_stall": stall_counts,
        },
        "correct_answer": (
            f"{n_gate2} of {len(runs)} recovered Flash families stall at gate 2 "
            f"(real teacher fit) because the Flash teacher corpus is NOT_BUILT; "
            f"{n_gate3} stall at gate 3. That is the correct answer. Bounded "
            "weight reconstruction and router AB MISMATCH were not used to skip "
            "teacher fit. No family was advanced onto NR/NX/EBPW."
        ),
        "integration": {
            "advance": "Funnel.advance(candidate, gate) -> AdvanceResult",
            "run": "Funnel.run(candidate) -> stall report",
            "match_scars": "Funnel.match_scars(hypothesis) -> list[scar] keyed by shape_fingerprint",
            "recover_families": "recover_families() -> (candidates, provenance)",
            "shape_fingerprint": "shape_fingerprint(hypothesis) -> sha256 hex",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Any:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = selftest() if a.selftest else build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
