"""Generic capability evaluation — no single hardcoded benchmark.

Roadmap §48 territory: one interface usable for the current resident, a
future NX, representation experiments, and heterogeneous execution.

A capability does not exist until something CALLS it. This module's
call site is CapabilityEvalRegistry.run -> Evaluator.evaluate. Importing
the module is not an evaluation.

The live resident is out of scope. A CompletionSource with
talks_to_live_daemon=True is refused. Fixtures prove the interface.

Recovered existing implementations (not adequate as the generic interface):

* tools/headless/capability_suite.py — one hardcoded Qwen suite
* tools/headless/capability_contract.py — hardcoded llama.cpp Q5_K incumbent
* crates/hawking-eval/src/lib.rs — substring Task/run_suite, one suite shape

    python3 tools/future/capability_eval.py --build
    python3 -m pytest tools/future/test_capability_eval.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from tools.future._common import git, write_receipt
from tools.future.experiment_receipt import attach


RECEIPT = "CAPABILITY_EVAL.json"
SCHEMA = "hawking.future.capability_eval.v1"
RECORDED_BY = "tools/future/capability_eval.py"
VERSION = 1

ADMITTED_DOMAINS: tuple[str, ...] = (
    "reasoning",
    "coding",
    "tool_use",
    "mission_autonomy",
    "domain_capability",
)

ADMITTED_SUBJECT_KINDS: tuple[str, ...] = (
    "resident",
    "nx",
    "representation",
    "heterogeneous",
)

# Axes a representation candidate and an incumbent are both scored on.
# Billing axes match tools.future.complete_ebpw.COMPARE_AXES; fidelity
# flags (execute_match, chain_complete) are owned by the family chain.
REPRESENTATION_AXES: tuple[str, ...] = (
    "complete_ebpw",
    "stored_bytes",
    "stored_bpw",
    "billed_ms",
    "executable_bytes",
    "is_sub2_executable",
    "reconstructs_dense_parent",
    "consumes_representation_directly",
    "execute_match",
    "chain_complete",
)

EVIDENCE_TIERS: tuple[str, ...] = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)


class LiveResidentRefused(RuntimeError):
    """Evaluation of the live daemon is out of scope."""


class EvalRefused(ValueError):
    """The interface refused a call (unknown domain/kind/evaluator)."""


@dataclass(frozen=True)
class EvalItem:
    id: str
    domain: str
    prompt: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSubject:
    kind: str
    identity: str

    def __post_init__(self) -> None:
        if self.kind not in ADMITTED_SUBJECT_KINDS:
            raise EvalRefused(
                f"subject kind {self.kind!r} is not one of {ADMITTED_SUBJECT_KINDS}"
            )


class CompletionSource(ABC):
    talks_to_live_daemon: bool = False

    @abstractmethod
    def complete(self, item: EvalItem) -> str:
        raise NotImplementedError


class ScriptedSource(CompletionSource):
    """Deterministic fixture. Does not talk to any model or daemon."""

    talks_to_live_daemon = False

    def __init__(self, answers: Mapping[str, str]) -> None:
        self.answers = dict(answers)

    def complete(self, item: EvalItem) -> str:
        if item.id not in self.answers:
            raise EvalRefused(f"scripted source has no answer for {item.id}")
        return self.answers[item.id]


class LiveResidentSource(CompletionSource):
    """Named so a caller cannot accidentally hit the daemon."""

    talks_to_live_daemon = True

    def complete(self, item: EvalItem) -> str:
        raise LiveResidentRefused(
            f"refusing to complete {item.id!r} via the live resident; "
            "evaluation of the live daemon is out of scope"
        )


class Evaluator(ABC):
    id: str
    domain: str

    @abstractmethod
    def items(self) -> tuple[EvalItem, ...]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, subject: EvalSubject, source: CompletionSource) -> dict[str, Any]:
        raise NotImplementedError


def _extract_code(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1)
    return text if "def " in text else None


def _extract_int(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text or "")
    if not nums:
        return None
    return int(nums[0])


class ReasoningPredicateEvaluator(Evaluator):
    """Reasoning scored by deterministic predicates over the completion text.

    Not a named public benchmark. Items are local fixtures so the interface
    can run without a model.
    """

    id = "reasoning.predicate"
    domain = "reasoning"

    def items(self) -> tuple[EvalItem, ...]:
        return (
            EvalItem(
                id="arith-17-19",
                domain=self.domain,
                prompt="Compute 17 * 19. Reply with only the number.",
                meta={"expect_int": 323},
            ),
            EvalItem(
                id="modules-after-edits",
                domain=self.domain,
                prompt=(
                    "A repo has 12 modules. 3 are deleted, then 5 are added, "
                    "then a quarter of the total are split in two. How many "
                    "modules are there at the end? Reply with only the number."
                ),
                meta={"expect_int": 21},
            ),
        )

    def evaluate(self, subject: EvalSubject, source: CompletionSource) -> dict[str, Any]:
        outcomes = []
        for item in self.items():
            text = source.complete(item)
            got = _extract_int(text)
            expect = item.meta["expect_int"]
            passed = got == expect
            outcomes.append(
                {
                    "id": item.id,
                    "domain": item.domain,
                    "passed": passed,
                    "got": got,
                    "expect": expect,
                    "completion_excerpt": (text or "")[:240],
                }
            )
        passed_n = sum(1 for o in outcomes if o["passed"])
        return {
            "evaluator_id": self.id,
            "domain": self.domain,
            "subject_kind": subject.kind,
            "subject_identity": subject.identity,
            "outcomes": outcomes,
            "passed": passed_n,
            "total": len(outcomes),
            "evidence_tier": "FUNCTIONAL_SIM",
            "scoring": "deterministic_integer_predicate",
            "hardcoded_benchmark": None,
        }


class CodingExecuteEvaluator(Evaluator):
    """Coding scored by compiling and executing the emitted function.

    Genuinely different from ReasoningPredicateEvaluator: the score is the
    return value of a called function, not a substring/integer predicate on
    the raw text.
    """

    id = "coding.execute"
    domain = "coding"

    def items(self) -> tuple[EvalItem, ...]:
        return (
            EvalItem(
                id="dedupe-stable",
                domain=self.domain,
                prompt=(
                    "Write a Python function `dedupe(xs)` that removes "
                    "duplicates while preserving first-seen order. Reply with "
                    "a single python code block."
                ),
                meta={
                    "fn": "dedupe",
                    "cases": [
                        {"args": [[1, 1, 2, 3, 2]], "expect": [1, 2, 3]},
                        {"args": [["a", "b", "a"]], "expect": ["a", "b"]},
                        {"args": [[]], "expect": []},
                    ],
                },
            ),
            EvalItem(
                id="clamp-range",
                domain=self.domain,
                prompt=(
                    "Write a Python function `clamp(x, lo, hi)` that returns x "
                    "bounded to [lo, hi]. Reply with a single python code block."
                ),
                meta={
                    "fn": "clamp",
                    "cases": [
                        {"args": [5, 0, 10], "expect": 5},
                        {"args": [-2, 0, 10], "expect": 0},
                        {"args": [99, 0, 10], "expect": 10},
                    ],
                },
            ),
        )

    def evaluate(self, subject: EvalSubject, source: CompletionSource) -> dict[str, Any]:
        outcomes = []
        for item in self.items():
            text = source.complete(item)
            passed, detail = self._score(item, text)
            outcomes.append(
                {
                    "id": item.id,
                    "domain": item.domain,
                    "passed": passed,
                    "detail": detail,
                    "fn": item.meta["fn"],
                    "n_cases": len(item.meta["cases"]),
                }
            )
        passed_n = sum(1 for o in outcomes if o["passed"])
        return {
            "evaluator_id": self.id,
            "domain": self.domain,
            "subject_kind": subject.kind,
            "subject_identity": subject.identity,
            "outcomes": outcomes,
            "passed": passed_n,
            "total": len(outcomes),
            "evidence_tier": "FUNCTIONAL_SIM",
            "scoring": "compile_and_execute_against_cases",
            "hardcoded_benchmark": None,
        }

    @staticmethod
    def _score(item: EvalItem, text: str) -> tuple[bool, str]:
        code = _extract_code(text)
        if not code:
            return False, "no python code in the completion"
        fn_name = item.meta["fn"]
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"does not parse: {exc}"
        defined = any(
            isinstance(n, ast.FunctionDef) and n.name == fn_name for n in ast.walk(tree)
        )
        if not defined:
            return False, f"no function named {fn_name}"
        ns: dict[str, Any] = {}
        try:
            exec(compile(tree, "<capability_eval>", "exec"), ns, ns)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            return False, f"exec failed: {type(exc).__name__}: {exc}"
        fn = ns.get(fn_name)
        if not callable(fn):
            return False, f"{fn_name} is not callable"
        for i, case in enumerate(item.meta["cases"]):
            try:
                got = fn(*case["args"])
            except Exception as exc:  # noqa: BLE001
                return False, f"case {i} raised {type(exc).__name__}: {exc}"
            if got != case["expect"]:
                return False, f"case {i}: got {got!r} expect {case['expect']!r}"
        return True, "all cases passed"


class CapabilityEvalRegistry:
    """One interface. Evaluators register; none is a default benchmark."""

    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, evaluator: Evaluator) -> None:
        if evaluator.domain not in ADMITTED_DOMAINS:
            raise EvalRefused(
                f"evaluator {evaluator.id} domain {evaluator.domain!r} "
                f"is not admitted; admitted={ADMITTED_DOMAINS}"
            )
        if evaluator.id in self._evaluators:
            raise EvalRefused(f"duplicate evaluator id {evaluator.id}")
        self._evaluators[evaluator.id] = evaluator

    def list_evaluators(self) -> tuple[str, ...]:
        return tuple(self._evaluators)

    def admits_domain(self, domain: str) -> bool:
        return domain in ADMITTED_DOMAINS

    def admits_subject_kind(self, kind: str) -> bool:
        return kind in ADMITTED_SUBJECT_KINDS

    def run(
        self,
        evaluator_id: str,
        subject: EvalSubject,
        source: CompletionSource,
    ) -> dict[str, Any]:
        if getattr(source, "talks_to_live_daemon", False):
            raise LiveResidentRefused(
                "CapabilityEvalRegistry.run refuses a live-resident source"
            )
        ev = self._evaluators.get(evaluator_id)
        if ev is None:
            raise EvalRefused(
                f"no evaluator registered as {evaluator_id!r}; "
                f"have {list(self._evaluators)}"
            )
        # CALL SITE: Evaluator.evaluate. An import is not this.
        return ev.evaluate(subject, source)

    def run_all(
        self,
        subject: EvalSubject,
        source: CompletionSource,
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for eid in self.list_evaluators():
            out[eid] = self.run(eid, subject, source)
        return out


def score_representation_family(
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    incumbent_id: str,
    incumbent: Mapping[str, Any],
) -> dict[str, Any]:
    """Hook: score a candidate family on the same axes as an incumbent.

    CALL SITE: tools.future.representation_lab.verify_family. An import of
    this function is not a score. A missing axis is a refusal, not a zero.
    Live-resident subjects are out of scope.
    """
    subject = EvalSubject(kind="representation", identity=candidate_id)
    missing_c = [k for k in REPRESENTATION_AXES if k not in candidate]
    missing_i = [k for k in REPRESENTATION_AXES if k not in incumbent]
    if missing_c or missing_i:
        raise EvalRefused(
            f"candidate missing {missing_c}, incumbent missing {missing_i}; "
            "scoring on different axes is refused"
        )
    per_axis: dict[str, Any] = {}
    numeric_wins = 0
    numeric_n = 0
    for axis in REPRESENTATION_AXES:
        c_val = candidate[axis]
        i_val = incumbent[axis]
        cell: dict[str, Any] = {"candidate": c_val, "incumbent": i_val}
        if isinstance(c_val, bool) and isinstance(i_val, bool):
            cell["same"] = c_val == i_val
        elif isinstance(c_val, (int, float)) and isinstance(i_val, (int, float)) and not (
            isinstance(c_val, bool) or isinstance(i_val, bool)
        ):
            delta = float(c_val) - float(i_val)
            cell["delta"] = delta
            numeric_n += 1
            # Lower complete_ebpw / stored_bytes / billed_ms / executable_bytes
            # is a win; booleans are not ranked here.
            if axis in {
                "complete_ebpw",
                "stored_bytes",
                "stored_bpw",
                "billed_ms",
                "executable_bytes",
            }:
                cell["candidate_smaller"] = delta < 0
                if delta < 0:
                    numeric_wins += 1
        per_axis[axis] = cell
    return {
        "evaluator_id": "representation.family_axes",
        "domain": "domain_capability",
        "subject_kind": subject.kind,
        "subject_identity": subject.identity,
        "incumbent_id": incumbent_id,
        "axes": list(REPRESENTATION_AXES),
        "per_axis": per_axis,
        "n_axes": len(REPRESENTATION_AXES),
        "n_cost_axes_candidate_smaller": numeric_wins,
        "n_cost_axes": numeric_n,
        "same_axes_as_incumbent": True,
        "hardcoded_benchmark": None,
        "evidence_tier": "STATIC",
        "call_site": (
            "score_representation_family -> EvalSubject(kind='representation')"
        ),
    }


def passing_scripted_source() -> ScriptedSource:
    return ScriptedSource(
        {
            "arith-17-19": "323",
            "modules-after-edits": "21",
            "dedupe-stable": (
                "```python\n"
                "def dedupe(xs):\n"
                "    out = []\n"
                "    seen = set()\n"
                "    for x in xs:\n"
                "        if x not in seen:\n"
                "            seen.add(x)\n"
                "            out.append(x)\n"
                "    return out\n"
                "```"
            ),
            "clamp-range": (
                "```python\n"
                "def clamp(x, lo, hi):\n"
                "    if x < lo:\n"
                "        return lo\n"
                "    if x > hi:\n"
                "        return hi\n"
                "    return x\n"
                "```"
            ),
        }
    )


def failing_scripted_source() -> ScriptedSource:
    return ScriptedSource(
        {
            "arith-17-19": "0",
            "modules-after-edits": "12",
            "dedupe-stable": "```python\ndef dedupe(xs):\n    return xs\n```",
            "clamp-range": "```python\ndef clamp(x, lo, hi):\n    return x\n```",
        }
    )


def default_registry() -> CapabilityEvalRegistry:
    """Two genuinely different evaluators, one interface."""
    reg = CapabilityEvalRegistry()
    reg.register(ReasoningPredicateEvaluator())
    reg.register(CodingExecuteEvaluator())
    return reg


def recover_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/headless/capability_suite.py",
            "what": (
                "Doctor/Tabula capability gate: one hardcoded Qwen suite with "
                "fixed items and axes (knowledge/reasoning/coding/...)."
            ),
            "adequate_for_this_lane": False,
            "why_not_adequate": (
                "A single suite is the defect. The generic interface must not "
                "hardcode one benchmark or one subject."
            ),
        },
        {
            "path": "tools/headless/capability_contract.py",
            "what": "Hardcoded incumbent llama.cpp Q5_K vs a fixed candidate list.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "Tied to one production body and four named runs.",
        },
        {
            "path": "crates/hawking-eval/src/lib.rs",
            "what": "Task/score/run_suite substring harness plus support-halo judge.",
            "adequate_for_this_lane": False,
            "why_not_adequate": (
                "One Task shape (prompt + expect substrings). Not a registry of "
                "domain evaluators, and the live serve path is out of scope."
            ),
        },
    ]


def build() -> Any:
    reg = default_registry()
    source = passing_scripted_source()
    fail_source = failing_scripted_source()
    subjects = (
        EvalSubject(kind="representation", identity="fixture-representation"),
        EvalSubject(kind="nx", identity="fixture-nx"),
    )
    runs = []
    for subject in subjects:
        # CALL SITE: registry.run for each registered evaluator.
        results = reg.run_all(subject, source)
        runs.append(
            {
                "subject_kind": subject.kind,
                "subject_identity": subject.identity,
                "results": results,
            }
        )
    fail_subject = EvalSubject(kind="heterogeneous", identity="fixture-fail")
    fail_results = reg.run_all(fail_subject, fail_source)

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Generic capability-evaluation interface. No single hardcoded "
            "benchmark. Live resident is refused."
        ),
        "admitted_domains": list(ADMITTED_DOMAINS),
        "admitted_subject_kinds": list(ADMITTED_SUBJECT_KINDS),
        "registered_evaluators": [
            {
                "id": eid,
                "domain": ev.domain,
                "class": type(ev).__name__,
                "n_items": len(ev.items()),
                "scoring": (
                    "deterministic_integer_predicate"
                    if ev.domain == "reasoning"
                    else "compile_and_execute_against_cases"
                ),
            }
            for eid in reg.list_evaluators()
            for ev in (reg._evaluators[eid],)
        ],
        "evaluators_run": [r["results"] for r in runs],
        "runs": runs,
        "negative_control_failing_source": fail_results,
        "live_resident_refused": True,
        "hardcoded_benchmark": None,
        "evidence_tier": "FUNCTIONAL_SIM",
        "recovered_implementation": recover_implementation(),
        "call_sites": (
            "CapabilityEvalRegistry.run -> ReasoningPredicateEvaluator.evaluate",
            "CapabilityEvalRegistry.run -> CodingExecuteEvaluator.evaluate",
            "CodingExecuteEvaluator._score -> exec of compiled AST",
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    # Prove the live-resident refusal actually fires (call site, not a comment).
    refused = False
    try:
        reg.run(
            "reasoning.predicate",
            EvalSubject(kind="resident", identity="must-not-touch"),
            LiveResidentSource(),
        )
    except LiveResidentRefused:
        refused = True
    if not refused:
        raise LiveResidentRefused("LiveResidentSource was not refused")
    doc["live_resident_refusal_fired"] = True
    doc = attach(
        doc,
        producer=RECORDED_BY,
        location=f"receipts/future/{RECEIPT}",
        claim=(
            "A generic capability-evaluation interface runs at least two "
            "distinct evaluators and refuses a live-resident source."
        ),
        verdict="ACCEPT",
        evidence_tier="FUNCTIONAL_SIM",
        scope="fixture subjects; live resident is out of scope",
        facts=[
            {"claim": "reasoning.predicate and coding.execute both ran"},
            {"claim": "LiveResidentSource was refused"},
        ],
        hypotheses=[],
        negative_controls=[
            {
                "id": "failing_source",
                "what": "a failing CompletionSource is rejected by both evaluators",
                "observed": fail_results,
            },
            {
                "id": "live_resident_refused",
                "what": "talks_to_live_daemon=True is a refusal, not a skip",
            },
        ],
        failures=[],
        resource_usage={"gpu_authority": False},
        qualification="FUNCTIONAL_SIM over fixtures; not a live-resident score",
        contamination=[],
        uncertainty=[
            "fixture items are not a substitute for a domain benchmark corpus"
        ],
        falsifier=(
            "a live-resident source is accepted, only one evaluator is "
            "registered, or evaluate() is never called"
        ),
        next_actions=[],
        receipts=[f"receipts/future/{RECEIPT}"],
        tests={"live_resident_refusal_fired": True},
        evidence=[{"call_sites": doc["call_sites"]}],
    )
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generic capability evaluation interface")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    print(build())
    return 0


if __name__ == "__main__":
    from tools.future._common import require_known_flags

    require_known_flags(["--build"])
    raise SystemExit(main())
