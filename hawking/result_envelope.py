"""Canonical result envelope shared by Engine, AgentOS, and receipts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RESULT_SCHEMA = "hawking.agentos.result.v1"


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


@dataclass
class ResultEnvelope:
    verdict: str
    claim: Optional[str] = None
    verified_facts: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    tests: Optional[Dict[str, Any]] = None
    negative_controls: Optional[List[Dict[str, Any]]] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)
    resource_use: Dict[str, Any] = field(default_factory=dict)
    uncertainty: List[str] = field(default_factory=list)
    blocker: Optional[str] = None
    next_action: Optional[str] = None
    receipt_paths: List[str] = field(default_factory=list)
    # Mutation/completion evidence is additive to the historical envelope.
    # These fields let a restart or a Web client distinguish a model claim
    # from an accepted, attributable change without replacing the existing
    # receipt schema.
    changed_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    diff: Dict[str, Any] = field(default_factory=dict)
    completion: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        # Keep both spellings at the boundary while callers migrate.  The
        # canonical field is resource_use; resource_usage was used by the
        # earlier delegation envelope.
        return {
            "schema": RESULT_SCHEMA,
            "verdict": self.verdict,
            "claim": self.claim,
            "verified_facts": list(self.verified_facts),
            "hypotheses": list(self.hypotheses),
            "evidence": list(self.evidence),
            "artifacts": list(self.artifacts),
            "tests": self.tests,
            "negative_controls": self.negative_controls,
            "failures": list(self.failures),
            "resource_use": dict(self.resource_use),
            "resource_usage": dict(self.resource_use),
            "uncertainty": list(self.uncertainty),
            "blocker": self.blocker,
            "next_action": self.next_action,
            "receipt_paths": list(self.receipt_paths),
            "changed_artifacts": list(self.changed_artifacts),
            "verification": dict(self.verification),
            "limitations": list(self.limitations),
            "diff": dict(self.diff),
            "completion": self.completion,
        }


def build_result_envelope(
    *,
    goal: str,
    result: Optional[Dict[str, Any]],
    evidence: Optional[Iterable[Dict[str, Any]]] = None,
    validation: Any = None,
    runtime_provenance: Optional[Iterable[Dict[str, Any]]] = None,
    model_calls: Optional[Iterable[Dict[str, Any]]] = None,
    receipt_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Derive an honest envelope from an Engine execution.

    Model prose is a hypothesis until deterministic validation passes.  This
    is intentionally a small derivation function so a new provider cannot
    promote its own text merely by returning a different JSON shape.
    """
    body = result if isinstance(result, dict) else {}
    status = str(body.get("status") or "unknown").lower()
    validation_dict = validation if isinstance(validation, dict) else {}
    validated = validation is True or validation_dict.get("ok") is True
    if status in {"failed", "cancelled"} or body.get("kind") == "error":
        verdict = "BLOCKED"
    elif validated:
        verdict = "ACCEPT"
    elif status == "unverified" or validation is None:
        verdict = "UNVERIFIED"
    else:
        verdict = "BLOCKED"

    facts: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    content = str(body.get("content") or "").strip()
    if content:
        entry = {"claim": content, "source": "engine_result"}
        if verdict == "ACCEPT":
            facts.append(entry)
        else:
            hypotheses.append(entry)

    evidence_rows: List[Dict[str, Any]] = []
    for item in evidence or ():
        if not isinstance(item, dict):
            continue
        row = {"path": item.get("path"), "identity": item.get("identity")}
        if item.get("path"):
            evidence_rows.append(row)

    artifacts: List[Dict[str, Any]] = []
    for item in _as_list(validation_dict.get("files")):
        if isinstance(item, dict):
            artifacts.append(dict(item))

    tests = None
    checks = validation_dict.get("checks")
    if isinstance(checks, list):
        tests = {"checks": checks}
    elif validation_dict:
        tests = {
            key: validation_dict[key]
            for key in ("command", "exit_code", "output", "reason", "ok")
            if key in validation_dict
        }
        if not tests:
            tests = None

    failures: List[Dict[str, Any]] = []
    if body.get("error") or body.get("error_type"):
        failures.append({
            "kind": body.get("error_type") or "error",
            "message": body.get("error") or "execution failed",
        })
    if isinstance(validation_dict, dict) and validation_dict.get("ok") is False:
        failures.append({
            "kind": "validation",
            "reason": validation_dict.get("reason") or "validation failed",
        })

    blocker = None
    if failures:
        blocker = str(failures[-1].get("message") or failures[-1].get("reason") or "execution did not verify")
    elif verdict != "ACCEPT":
        blocker = "deterministic verification did not establish the claim"

    uncertainty = []
    if verdict != "ACCEPT":
        uncertainty.append("model output is not a verified fact")
    if body.get("rolled_back"):
        uncertainty.append("mutation was rolled back")

    paths = [str(receipt_path)] if receipt_path else []
    changed_artifacts = []
    for item in _as_list(body.get("changed_files") or body.get("files")):
        if isinstance(item, dict):
            changed_artifacts.append(dict(item))
    if not changed_artifacts:
        changed_artifacts = [dict(item) for item in artifacts if item.get("changed")]

    completion_raw = body.get("completion_packet")
    completion = dict(completion_raw) if isinstance(completion_raw, dict) else None
    verification = {}
    if completion is not None and isinstance(completion.get("verification"), dict):
        verification.update(completion["verification"])
    if validation_dict:
        verification.setdefault("validation_ok", validation_dict.get("ok") is True)
        verification.setdefault("checks", validation_dict.get("checks") or [])
    verification.setdefault("diff_inspected", bool(body.get("diff_inspected")))
    limitations = []
    if completion is not None and isinstance(completion.get("limitations"), list):
        limitations.extend(str(item) for item in completion["limitations"])
    if verdict != "ACCEPT" and not limitations:
        limitations.append("deterministic verification did not establish the claim")
    raw_diff = str(body.get("diff") or "")
    diff_summary = {
        "inspected": bool(body.get("diff_inspected") or raw_diff),
        "sha256": hashlib.sha256(raw_diff.encode("utf-8")).hexdigest() if raw_diff else None,
        "chars": len(raw_diff),
    }
    envelope = ResultEnvelope(
        verdict=verdict,
        claim=str(goal or "") or None,
        verified_facts=facts,
        hypotheses=hypotheses,
        evidence=evidence_rows,
        artifacts=artifacts,
        tests=tests,
        negative_controls=(
            validation_dict.get("negative_controls")
            if isinstance(validation_dict.get("negative_controls"), list)
            else None
        ),
        failures=failures,
        resource_use={
            "runtime_provenance": list(runtime_provenance or []),
            "model_calls": list(model_calls or []),
        },
        uncertainty=uncertainty,
        blocker=blocker,
        next_action=(
            "re-run the named deterministic tests"
            if verdict == "ACCEPT"
            else "supply or repair a deterministic verifier before accepting"
        ),
        receipt_paths=paths,
        changed_artifacts=changed_artifacts,
        verification=verification,
        limitations=list(dict.fromkeys(limitations)),
        diff=diff_summary,
        completion=completion,
    )
    return envelope.to_dict()


__all__ = ["RESULT_SCHEMA", "ResultEnvelope", "build_result_envelope"]
