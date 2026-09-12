"""Bounded, resumable Gravity search owned by HCLI.

This is the search behavior around the existing Doctor/Gravity/NR/NX
artifacts.  It does not invent a second model runtime: an evaluator supplies a
real patient-runner receipt, while this module owns candidate choice,
checkpointing, terminal classification, and the complete-system verifier.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .latency import now_ns, seconds_from_ns, since_ns, wall_elapsed_ns


SCHEMA = "hawking.hcli.odyssey.gravity_gauntlet.v1"
TARGET_HIT = "TARGET_HIT"
PROVEN_UNABLE = "PROVEN_UNABLE"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
TERMINAL_DISPOSITIONS = frozenset({TARGET_HIT, PROVEN_UNABLE, BUDGET_EXHAUSTED})
TARGET_COMPLETE_EBPW = 1.0
TARGET_REQUIREMENTS = (
    "complete accounting",
    "capability",
    "execution",
    "independent verifier",
    "magnitude adequacy",
    "validated utilization",
)


@dataclass(frozen=True)
class Candidate:
    id: str
    specimen: str
    spec: str
    parent_id: str | None
    mutation: str
    expected_effect: str
    representation_class: str = "candidate"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "candidate"


def _spec_bits(spec: str) -> int | None:
    # None means "this spec carries no precision information", which is NOT the
    # same as q4. Returning a default here made an unrunnable string
    # indistinguishable from a measured 4-bit candidate and fed a number nobody
    # measured into the search order.
    m = re.search(r"q(\d+)", spec)
    return int(m.group(1)) if m else None


def _spec_group(spec: str) -> int | None:
    m = re.search(r"g(\d+)", spec)
    return int(m.group(1)) if m else None


def candidate_space(specimen: str, specs: Iterable[str]) -> list[Candidate]:
    """Create a deterministic representation frontier from explicit specs."""
    out: list[Candidate] = []
    seen: set[str] = set()
    for i, spec in enumerate(specs):
        spec = str(spec).strip()
        if not spec or spec in seen:
            continue
        seen.add(spec)
        rep_class = (
            "NEGATIVE_CONTROL_MAGNITUDE_0.01W"
            if "negative-control" in spec
            else "candidate"
        )
        out.append(
            Candidate(
                id=f"{_slug(specimen)}-{_slug(spec)}",
                specimen=specimen,
                spec=spec,
                parent_id=None,
                mutation="initial_representation" if i == 0 else "frontier_candidate",
                expected_effect=_expected_effect(spec, rep_class),
                representation_class=rep_class,
            )
        )
    if not out:
        raise ValueError("candidate space is empty")
    return out


def _expected_effect(spec: str, rep_class: str) -> str:
    if rep_class.startswith("NEGATIVE_CONTROL"):
        return "deliberately destroys magnitude while preserving direction"
    bits, group = _spec_bits(spec), _spec_group(spec)
    if bits is None:
        return f"test the {spec} representation frontier (no uniform precision)"
    return f"test q{bits} / group{group if group is not None else 'unset'} storage frontier"


def _candidate_doc(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "specimen": candidate.specimen,
        "spec": candidate.spec,
        "parent_id": candidate.parent_id,
        "mutation": candidate.mutation,
        "expected_effect": candidate.expected_effect,
        "representation_class": candidate.representation_class,
    }


def _candidate_from_doc(doc: Mapping[str, Any]) -> Candidate:
    return Candidate(
        id=str(doc["id"]),
        specimen=str(doc["specimen"]),
        spec=str(doc["spec"]),
        parent_id=doc.get("parent_id"),
        mutation=str(doc["mutation"]),
        expected_effect=str(doc["expected_effect"]),
        representation_class=str(doc.get("representation_class") or "candidate"),
    )


def _atomic_write(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_receipt(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return json.loads(json.dumps(dict(value)))
    return json.loads(Path(value).read_text())


def _verification_wall_ns(data: Mapping[str, Any]) -> int | None:
    value = data.get("verification_wall_ns")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    legacy = data.get("verification_wall_s")
    if isinstance(legacy, (int, float)) and not isinstance(legacy, bool):
        return max(0, int(round(float(legacy) * 1_000_000_000)))
    return None


def _measured_complete_bpw(receipt: Mapping[str, Any]) -> tuple[float | None, str | None]:
    raw = receipt.get("complete_ebpw", receipt.get("complete_bpw"))
    if raw is None and isinstance(receipt.get("accounting"), Mapping):
        raw = receipt["accounting"].get("complete_bpw")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "complete EBPW is absent or non-numeric"
    labels = receipt.get("labels")
    label = labels.get("complete_bpw") if isinstance(labels, Mapping) else None
    evidence = str(receipt.get("_evidence") or "")
    if label is not None and "MEASURED" not in str(label).upper():
        return None, f"complete EBPW label is not measured: {label!r}"
    if label is None and "MEASURED" not in evidence.upper() and receipt.get("measurement_state") != "MEASURED":
        return None, "complete EBPW has no measured provenance"
    return value, None


def _accounting_check(receipt: Mapping[str, Any], complete_bpw: float | None) -> tuple[bool, str]:
    accounting = receipt.get("accounting")
    if not isinstance(accounting, Mapping):
        return False, "complete accounting block is absent"
    if complete_bpw is None:
        return False, "complete accounting cannot bind to an absent EBPW"
    bound = accounting.get("complete_bpw")
    if bound is not None:
        try:
            if abs(float(bound) - complete_bpw) > 0.002:
                return False, "accounting.complete_bpw disagrees with receipt complete EBPW"
        except (TypeError, ValueError):
            return False, "accounting.complete_bpw is not numeric"
    if not any(accounting.get(k) is not None for k in ("complete_bytes", "executable_bytes", "disk_tensors")):
        return False, "complete accounting has no executable/persistent byte evidence"
    return True, "complete accounting reconciles"


def magnitude_adequacy(candidate: Candidate, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a direction-preserving but magnitude-destroyed representation."""
    ratio = receipt.get("magnitude_ratio")
    direction = receipt.get("direction_similarity")
    if candidate.representation_class.startswith("NEGATIVE_CONTROL"):
        try:
            failed = ratio is None or float(ratio) < 0.5
        except (TypeError, ValueError):
            failed = True
        return {
            "adequate": not failed,
            "verdict": "REJECTED_MAGNITUDE_DESTROYED" if failed else "CONTROL_DID_NOT_FAIL",
            "direction_similarity": direction,
            "magnitude_ratio": ratio,
            "reason": "0.01W must fail even when cosine/direction is preserved",
        }
    if ratio is None:
        return {"adequate": False, "verdict": "UNVALIDATED", "direction_similarity": direction, "magnitude_ratio": None,
                "reason": "magnitude adequacy was not measured"}
    try:
        ratio_f = float(ratio)
    except (TypeError, ValueError):
        return {"adequate": False, "verdict": "FAIL", "direction_similarity": direction, "magnitude_ratio": ratio,
                "reason": "magnitude ratio is not numeric"}
    return {"adequate": ratio_f >= 0.5, "verdict": "PASS" if ratio_f >= 0.5 else "FAIL",
            "direction_similarity": direction, "magnitude_ratio": ratio,
            "reason": "measured magnitude ratio"}


def observe(candidate: Candidate, receipt: Mapping[str, Any] | str | Path, *, target: float = TARGET_COMPLETE_EBPW) -> dict[str, Any]:
    raw = _read_receipt(receipt)
    complete_bpw, complete_error = _measured_complete_bpw(raw)
    accounting_ok, accounting_reason = _accounting_check(raw, complete_bpw)
    doctor_verdict = str(raw.get("verdict") or "")
    capability_signal = bool(raw.get("capability_ok")) or doctor_verdict in {"CANDIDATE_PASS", "PASS"}
    execution_complete = bool(raw.get("execution_complete") or raw.get("native_execution_complete"))
    verifier_independent = bool(raw.get("verifier_independent") or raw.get("independent_verifier"))
    utilization = raw.get("utilization")
    utilization_measured = isinstance(utilization, Mapping) and bool(utilization.get("validated_fields"))
    magnitude = magnitude_adequacy(candidate, raw)
    reject_reasons = [x for x in (complete_error, accounting_reason if not accounting_ok else None) if x]
    if not magnitude["adequate"]:
        reject_reasons.append(magnitude["reason"])
    doctor = raw.get("doctor")
    doctor_wall = doctor if isinstance(doctor, Mapping) else {}
    measured_wall_ns = wall_elapsed_ns(raw)
    if measured_wall_ns is None:
        measured_wall_ns = wall_elapsed_ns(doctor_wall)
    verification_wall_ns = _verification_wall_ns(raw)
    return {
        "receipt": raw.get("out") or raw.get("receipt_path"),
        "complete_ebpw": complete_bpw,
        "complete_ebpw_measured": complete_error is None,
        "complete_accounting": accounting_ok,
        "accounting_reason": accounting_reason,
        "capability_signal": capability_signal,
        "capability_status": raw.get("capability_status") or doctor_verdict or "UNKNOWN",
        "execution_complete": execution_complete,
        "verifier_independent": verifier_independent,
        "utilization_measured": utilization_measured,
        "magnitude_adequacy": magnitude,
        "persistent_bytes": raw.get("stored_bytes") or (raw.get("accounting") or {}).get("complete_bytes"),
        "wall_ns": measured_wall_ns,
        "wall_s": seconds_from_ns(measured_wall_ns),
        "verification_wall_ns": verification_wall_ns,
        "verification_wall_s": seconds_from_ns(verification_wall_ns),
        "resource_measurements": raw.get("resource_measurements") or raw.get("utilization"),
        "nr_release_verified": bool(raw.get("nr_release_verified") or raw.get("release_verified")),
        "target_eligible": bool(
            complete_error is None
            and accounting_ok
            and complete_bpw is not None
            and complete_bpw <= target
            and capability_signal
            and execution_complete
            and verifier_independent
            and utilization_measured
            and magnitude["adequate"]
        ),
        "reject_reasons": reject_reasons,
        "source_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
    }


def _candidate_priority(candidate: Candidate, *, capability_signal: bool) -> tuple[int]:
    bits = _spec_bits(candidate.spec)
    if bits is None:
        # No precision to descend or climb. Rank after every candidate that has
        # one; Python's stable sort then preserves explicit frontier order here.
        return (1 << 16,)
    # After a capability-preserving result, descend bits first. After a
    # capability loss, climb precision first. Group choices stay in the
    # explicit frontier order supplied by the Doctor.
    # Python's sort is stable: equal precision candidates retain the explicit
    # frontier order supplied by the Doctor. That order is evidence, not a
    # hidden universal preference for one group size.
    return (bits if not capability_signal else -bits,)


def proven_unable_bound(bound: Mapping[str, Any], *, target: float = TARGET_COMPLETE_EBPW) -> tuple[bool, str]:
    required = ("limiting_mechanism", "measured_evidence", "assumptions", "search_region", "reopen_condition")
    missing = [k for k in required if not bound.get(k)]
    if missing:
        return False, f"PROVEN_UNABLE is missing {missing}"
    if bound.get("proven") is not True:
        return False, "bound.proven must be true"
    try:
        upper = float(bound["upper_bound_complete_ebpw"])
    except (KeyError, TypeError, ValueError):
        return False, "upper_bound_complete_ebpw is required and numeric"
    if upper <= target:
        return False, "the stated bound does not exclude the target"
    return True, "independently justified bound excludes the target"


class GravityGauntlet:
    """Single-writer search state machine; every step is checkpointed."""

    def __init__(self, state_path: str | Path, specimen: str, candidates: Iterable[Candidate], budget: int,
                 target: float = TARGET_COMPLETE_EBPW) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.path = Path(state_path)
        self.candidates = list(candidates)
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if any(c.specimen != specimen for c in self.candidates):
            raise ValueError("all candidates must belong to the same specimen")
        self.specimen = specimen
        self.budget = int(budget)
        self.target = float(target)
        self.state = self._load_or_init()
        self._evaluated_ids = {
            row["candidate"]["id"]
            for row in self.state.get("iterations", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("candidate"), Mapping)
            and row["candidate"].get("id")
        }
        self._iteration_by_id = {
            row["candidate"]["id"]: row
            for row in self.state.get("iterations", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("candidate"), Mapping)
            and row["candidate"].get("id")
        }

    def _load_or_init(self) -> dict[str, Any]:
        docs = {_candidate_doc(c)["id"]: c for c in self.candidates}
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("schema") != SCHEMA:
                raise ValueError("checkpoint schema mismatch")
            if state.get("specimen") != self.specimen:
                raise ValueError("checkpoint specimen mismatch")
            if set(state.get("candidate_space", {})) != set(docs):
                raise ValueError("candidate space changed; start a new gauntlet")
            if list((state.get("target") or {}).get("requires") or []) != list(TARGET_REQUIREMENTS):
                state.setdefault("target", {})["requires"] = list(TARGET_REQUIREMENTS)
                _atomic_write(self.path, state)
            self._normalize_cost(state)
            return state
        state = {
            "schema": SCHEMA,
            "authority": "HCLI single writer; verifier decides target status",
            "specimen": self.specimen,
            "target": {"complete_ebpw_lte": self.target, "requires": list(TARGET_REQUIREMENTS)},
            "budget": {"max_evaluations": self.budget, "used": 0},
            "candidate_space": {c.id: _candidate_doc(c) for c in self.candidates},
            "frontier": [c.id for c in self.candidates],
            "iterations": [],
            "best_candidate_id": None,
            "nr": {"current_candidate_id": None, "released_candidate_ids": [], "release_verified": True},
            "terminal": None,
            "cost": {
                "candidate_evaluations": 0,
                "wall_ns": 0,
                "verification_wall_ns": 0,
                "wall_s": 0.0,
                "verification_wall_s": 0.0,
                "timing_unit": "ns",
                "resource_cost": {},
            },
            "created_at": time.time(),
            "writer": {"pid": os.getpid(), "mode": "single_writer"},
        }
        self._normalize_cost(state)
        return state

    @staticmethod
    def _normalize_cost(state: dict[str, Any]) -> None:
        """Backfill exact duration totals once, then keep them incremental."""
        cost = state.setdefault("cost", {})
        rows = state.get("iterations") or []
        wall_ns = sum(
            wall_elapsed_ns(row.get("observation", {})) or 0
            for row in rows
            if isinstance(row, Mapping)
        )
        verification_wall_ns = sum(
            _verification_wall_ns(row.get("observation", {})) or 0
            if isinstance(row, Mapping) and isinstance(row.get("observation"), Mapping)
            else 0
            for row in rows
        )
        cost["wall_ns"] = wall_ns
        cost["verification_wall_ns"] = verification_wall_ns
        cost["timing_unit"] = "ns"
        cost["wall_s"] = seconds_from_ns(wall_ns) or 0.0
        cost["verification_wall_s"] = seconds_from_ns(verification_wall_ns) or 0.0

    def _checkpoint(self) -> None:
        self.state["updated_at"] = time.time()
        _atomic_write(self.path, self.state)

    def _remaining(self) -> list[Candidate]:
        return [
            _candidate_from_doc(self.state["candidate_space"][cid])
            for cid in self.state["frontier"]
            if cid not in self._evaluated_ids
        ]

    def _choose_next(self, observation: Mapping[str, Any], *, exclude_id: str | None = None) -> Candidate | None:
        remaining = self._remaining()
        if exclude_id is not None:
            remaining = [c for c in remaining if c.id != exclude_id]
        if not remaining:
            return None
        if observation.get("magnitude_adequacy", {}).get("verdict") == "REJECTED_MAGNITUDE_DESTROYED":
            # A negative control is an adequacy regression, never a search win.
            remaining = [c for c in remaining if not c.representation_class.startswith("NEGATIVE_CONTROL")] or remaining
        signal = bool(observation.get("capability_signal"))
        return sorted(remaining, key=lambda c: _candidate_priority(c, capability_signal=signal))[0]

    def _update_best(self, candidate: Candidate, observation: Mapping[str, Any]) -> None:
        if observation.get("complete_ebpw") is None or not observation.get("complete_ebpw_measured"):
            return
        current_id = self.state.get("best_candidate_id")
        current = self._iteration_by_id.get(current_id) if current_id else None
        # Capability first, then bytes. Ranking on EBPW alone crowns whatever is
        # smallest, including a body that cannot generate -- the headline number
        # would then advertise a broken artifact.
        def rank(obs: Mapping[str, Any]) -> tuple[int, float]:
            return (0 if obs.get("capability_signal") else 1, float(obs["complete_ebpw"]))

        if current is None or rank(observation) < rank(current["observation"]):
            self.state["best_candidate_id"] = candidate.id

    def step(self, candidate: Candidate, receipt: Mapping[str, Any] | str | Path) -> dict[str, Any]:
        if self.state.get("terminal"):
            return self.state
        if self.state["budget"]["used"] >= self.budget:
            self._finish_budget("allocated candidate budget is exhausted")
            return self.state
        if candidate.id in self._evaluated_ids:
            raise ValueError(f"candidate {candidate.id} was already evaluated")
        started_ns = now_ns()
        observation = observe(candidate, receipt, target=self.target)
        prior = self.state["nr"].get("current_candidate_id")
        if prior and prior != candidate.id:
            self.state["nr"]["released_candidate_ids"].append(prior)
            self.state["nr"]["release_verified"] = bool(
                self.state["iterations"][-1]["observation"].get("nr_release_verified")
            )
        candidate_doc = _candidate_doc(candidate)
        if prior:
            candidate_doc["parent_id"] = prior
            candidate_doc["mutation"] = "evidence_guided_precision_step"
        next_candidate = self._choose_next(observation, exclude_id=candidate.id)
        step_wall_ns = since_ns(started_ns)
        row = {
            "candidate": candidate_doc,
            "observation": observation,
            "decision": {
                "next_candidate_id": next_candidate.id if next_candidate else None,
                "reason": (
                    "capability signal preserved; descend bits/group for information gain"
                    if observation.get("capability_signal")
                    else "capability signal weak/absent; prefer higher-precision survivor"
                ),
            },
            "wall_ns": step_wall_ns,
            "wall_s": seconds_from_ns(step_wall_ns) or 0.0,
            "timing_unit": "ns",
        }
        self.state["iterations"].append(row)
        self._evaluated_ids.add(candidate.id)
        self._iteration_by_id[candidate.id] = row
        self.state["budget"]["used"] += 1
        self.state["nr"]["current_candidate_id"] = candidate.id
        self._update_best(candidate, observation)
        cost = self.state["cost"]
        cost["candidate_evaluations"] = self.state["budget"]["used"]
        cost["wall_ns"] = int(cost.get("wall_ns") or 0) + int(observation.get("wall_ns") or 0)
        cost["verification_wall_ns"] = int(cost.get("verification_wall_ns") or 0) + int(observation.get("verification_wall_ns") or 0)
        cost["timing_unit"] = "ns"
        cost["wall_s"] = seconds_from_ns(cost["wall_ns"]) or 0.0
        cost["verification_wall_s"] = seconds_from_ns(cost["verification_wall_ns"]) or 0.0
        if observation.get("target_eligible"):
            self.state["terminal"] = {"disposition": TARGET_HIT, "candidate_id": candidate.id, "reason": "complete EBPW and all required gates passed"}
        elif self.state["budget"]["used"] >= self.budget or next_candidate is None:
            self._finish_budget("allocated search ended before target hit")
        self._checkpoint()
        return self.state

    def _finish_budget(self, reason: str) -> None:
        best_id = self.state.get("best_candidate_id")
        best_row = self._iteration_by_id.get(best_id) if best_id else None
        self.state["terminal"] = {
            "disposition": BUDGET_EXHAUSTED,
            "reason": reason,
            "best_candidate_id": self.state.get("best_candidate_id"),
            "best_complete_ebpw": self.best_complete_ebpw(),
            "best_capability_status": (best_row or {}).get("observation", {}).get("capability_status"),
            "best_execution_complete": (best_row or {}).get("observation", {}).get("execution_complete"),
            "best_verifier_independent": (best_row or {}).get("observation", {}).get("verifier_independent"),
            "remaining_frontier": [c.id for c in self._remaining()],
        }

    def finalize_proven_unable(self, bound: Mapping[str, Any]) -> dict[str, Any]:
        if self.state.get("terminal") and self.state["terminal"]["disposition"] == TARGET_HIT:
            raise ValueError("a target hit cannot be rewritten as PROVEN_UNABLE")
        ok, reason = proven_unable_bound(bound, target=self.target)
        if not ok:
            raise ValueError(reason)
        self.state["terminal"] = {"disposition": PROVEN_UNABLE, "bound": dict(bound), "reason": reason}
        self._checkpoint()
        return self.state

    def best_complete_ebpw(self) -> float | None:
        # The EBPW OF the best candidate, not the minimum over all of them.
        # A bare min disagrees with best_candidate_id whenever the smallest body
        # is one that lost capability, putting two different meanings of "best"
        # in the same terminal block.
        best_id = self.state.get("best_candidate_id")
        row = self._iteration_by_id.get(best_id) if best_id else None
        if row is not None and row["observation"].get("complete_ebpw") is not None:
            return float(row["observation"]["complete_ebpw"])
        vals = [float(x) for x in (y["observation"].get("complete_ebpw") for y in self.state["iterations"]) if x is not None]
        return min(vals) if vals else None

    def run(self, evaluator: Callable[[Candidate], Mapping[str, Any] | str | Path], max_steps: int | None = None) -> dict[str, Any]:
        steps = 0
        while not self.state.get("terminal") and self._remaining() and self.state["budget"]["used"] < self.budget:
            if max_steps is not None and steps >= max_steps:
                break
            if not self.state["iterations"]:
                candidate = self._remaining()[0]
            else:
                candidate = self._choose_next(self.state["iterations"][-1]["observation"])
            if candidate is None:
                break
            try:
                receipt = evaluator(candidate)
            except Exception as exc:  # preserve a failed experiment as evidence
                receipt = {
                    "receipt_path": None,
                    "_evidence": "MEASURED (runner failure)",
                    "error": f"{type(exc).__name__}: {exc}",
                    "verdict": "RUNNER_FAILURE",
                }
            self.step(candidate, receipt)
            steps += 1
        if not self.state.get("terminal") and (self.state["budget"]["used"] >= self.budget or not self._remaining()):
            self._finish_budget("allocated search ended before target hit")
            self._checkpoint()
        return self.state


def run_from_receipts(state_path: str | Path, specimen: str, specs: Iterable[str], budget: int,
                      receipt_dir: str | Path) -> dict[str, Any]:
    """Run against existing measured patient receipts without re-running them."""
    root = Path(receipt_dir)
    candidates = candidate_space(specimen, specs)
    engine = GravityGauntlet(state_path, specimen, candidates, budget)

    def evaluator(candidate: Candidate) -> Path:
        path = root / f"{specimen}_GRAVITY_{candidate.spec}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    return engine.run(evaluator)


def run_patient_runner(state_path: str | Path, specimen: str, specs: Iterable[str], budget: int,
                       runner: str | Path, weights: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Evaluate candidates sequentially through the existing patient runner."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidates = candidate_space(specimen, specs)
    engine = GravityGauntlet(state_path, specimen, candidates, budget)

    def evaluator(candidate: Candidate) -> Path:
        out = root / f"{specimen}_GRAVITY_{candidate.spec}.json"
        cmd = [sys.executable, str(runner), "--oxx", specimen, "--weights", str(weights), "--gravity", candidate.spec, "--out", str(out)]
        subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
        return out

    return engine.run(evaluator)
