"""Hawking-owned Auto planning, worker packets, and routing evidence.

Auto is deliberately a control-plane decision, not a provider prompt.  This
module produces a small, inspectable CognitionPlan from deterministic signals,
keeps routing inspectable, and persists a redacted Worker Packet for each
WorkUnit.  Provider conversation state is never used as Goal authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .auto_mode import (
    CORE_AUTO_MODELS,
    ESCALATION_AUTO_MODELS,
    MODEL_ROLE_BIASES,
    REMOTE_AUTO_ROSTER,
    request_text,
)
from .persist import atomic_write_json

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]

AUTO_PLAN_SCHEMA = "hawking.auto.cognition_plan.v1"
WORKER_PACKET_SCHEMA = "hawking.worker_packet.v1"
METRICS_SCHEMA = "hawking.auto.model_metrics.v1"
AUTO_REFILL_SCHEMA = "hawking.auto.refill.v1"
AUTO_CONTINUATION_SCHEMA = "hawking.auto.continuation.v1"
AUTO_EXECUTION_AUDIT_SCHEMA = "hawking.auto.execution_audit.v3"

# P4_AUTO_EXECUTION_AUDIT_V3: a worker packet is only admissible for execution
# when its declared task class is one of the accepted routing labels.  The
# legacy labels and the empirical labels are both accepted; anything else is a
# routing defect that must be surfaced before a provider is dispatched.
AUTO_EXECUTION_AUDIT_REQUIRED_FIELDS = (
    "workunit_id",
    "task_class",
    "goal_id",
)


def audit_worker_packet_for_execution(packet: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deterministic execution-audit verdict for one worker packet.

    The audit is a pure function of the packet bytes: it never mutates the
    packet, never consults provider state, and never invents authority.  A
    packet is ``admissible`` only when every required field is present and
    non-empty and the declared ``task_class`` is an accepted routing label.
    """
    if not isinstance(packet, Mapping):
        return {
            "schema": AUTO_EXECUTION_AUDIT_SCHEMA,
            "admissible": False,
            "reasons": ["packet_not_mapping"],
            "missing_fields": list(AUTO_EXECUTION_AUDIT_REQUIRED_FIELDS),
            "task_class": None,
        }

    missing_fields: List[str] = []
    for field in AUTO_EXECUTION_AUDIT_REQUIRED_FIELDS:
        value = packet.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field)

    task_class = packet.get("task_class")
    task_class_label = task_class if isinstance(task_class, str) else None
    # Normalize the declared label before matching: routing labels are
    # case-insensitive and tolerate surrounding whitespace, so a packet that
    # declares "  Repo_Engineering " is the same routing decision as
    # "repo_engineering".  The normalized label is what the audit reports so
    # downstream receipts are byte-stable regardless of packet formatting.
    task_class_normalized = (
        task_class_label.strip().lower() if task_class_label is not None else None
    )

    reasons: List[str] = []
    if missing_fields:
        reasons.append("missing_required_fields")
    if task_class_normalized is None:
        reasons.append("task_class_not_string")
    elif task_class_normalized not in AUTO_TASK_CLASSES:
        reasons.append("task_class_not_accepted")

    return {
        "schema": AUTO_EXECUTION_AUDIT_SCHEMA,
        "admissible": not reasons,
        "reasons": reasons,
        "missing_fields": missing_fields,
        "task_class": task_class_normalized,
    }

# The old 12/4 and 25% burst values were an artificial Hawking admission
# ceiling.  Keep the historical values as telemetry only; actual admission is
# unbounded by worker count and is gated by real resource/provider health,
# dependency readiness, budget authority, and disjoint writer leases.
BASE_MAX_REMOTE_COGNITION_WORKERS = 12
BASE_MAX_PREMIUM_WORKERS = 4
FRONTIER_BURST_FACTOR = 1.25
UNBOUNDED_CONCURRENCY = None
MAX_REMOTE_COGNITION_WORKERS = UNBOUNDED_CONCURRENCY
MAX_PREMIUM_WORKERS = UNBOUNDED_CONCURRENCY
MAX_AUTOTUNED_WORKERS = UNBOUNDED_CONCURRENCY
AUTO_TASK_CLASSES = (
    "deterministic",
    "research",
    "repo_engineering",
    "debugging",
    "planning",
    "architecture",
    "review",
    "browser",
    "desktop",
    "document",
    "reverse_engineering",
    "model_science",
    "benchmark",
    "multimodal",
    "physical_engineering",
    "synthesis",
    "general",
    # Empirical routing labels. The legacy labels above remain accepted in
    # packets and receipts; these finer classes are the measured-value keys.
    "source_mapping",
    "routine_implementation",
    "large_refactor",
    "architecture_planning",
    "execution_plan",
    "bug_repair",
    "test_design",
    "falsification",
    "security_analysis",
    "high_value_review",
    "hardware_reasoning",
    "long_context_synthesis",
)

TASK_CLASS_ALIASES = {
    "research": "source_mapping",
    "repo_engineering": "routine_implementation",
    "debugging": "bug_repair",
    "planning": "architecture_planning",
    "architecture": "architecture_planning",
    "review": "high_value_review",
    "benchmark": "test_design",
    "physical_engineering": "hardware_reasoning",
    "synthesis": "long_context_synthesis",
    "model_science": "long_context_synthesis",
}

# Every alias target must be a real, routable task class.  A typo here would
# silently route a WorkUnit to a class with no measured priors, so the mapping
# is validated at import time and fails loudly instead of degrading routing.
_UNROUTABLE_ALIAS_TARGETS = tuple(
    sorted(
        {
            target
            for target in TASK_CLASS_ALIASES.values()
            if target not in AUTO_TASK_CLASSES
        }
    )
)
if _UNROUTABLE_ALIAS_TARGETS:  # pragma: no cover - guards a static typo
    raise RuntimeError(
        "TASK_CLASS_ALIASES targets are not routable task classes: "
        + ", ".join(_UNROUTABLE_ALIAS_TARGETS)
    )


def canonical_task_class(task_class: str) -> str:
    """Resolve a task-class label to its routable canonical class.

    Legacy labels (``research``, ``repo_engineering``, ...) are folded onto the
    empirical routing labels via :data:`TASK_CLASS_ALIASES`.  Unknown labels are
    returned unchanged so callers keep their existing fallback behavior, and
    already-canonical labels are returned as-is.
    """
    if not isinstance(task_class, str):
        return task_class
    return TASK_CLASS_ALIASES.get(task_class, task_class)

GLM_TRIAL_CLASSES = frozenset({
    "source_mapping", "routine_implementation", "architecture_planning",
    "test_design", "security_analysis", "long_context_synthesis",
})

MODEL_PRIORS = {
    "repo_engineering": "deepseek/deepseek-v4.1-flash",
    "debugging": "deepseek/deepseek-v4.1-flash",
    "research": "deepseek/deepseek-v4.1-flash",
    "benchmark": "deepseek/deepseek-v4.1-flash",
    "planning": "moonshotai/kimi-k3",
    "architecture": "moonshotai/kimi-k3",
    "review": "moonshotai/kimi-k3",
    "browser": "deepseek/deepseek-v4.1-flash",
    "desktop": "deepseek/deepseek-v4.1-flash",
    "document": "deepseek/deepseek-v4.1-flash",
    "reverse_engineering": "deepseek/deepseek-v4.1-flash",
    "synthesis": "moonshotai/kimi-k3",
    "multimodal": "deepseek/deepseek-v4.1-flash",
    "physical_engineering": "deepseek/deepseek-v4.1-flash",
    "deterministic": "deepseek/deepseek-v4.1-flash",
    "general": "deepseek/deepseek-v4.1-flash",
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(openrouter_api_key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_METRICS_THREAD_LOCK = threading.Lock()


@contextmanager
def _performance_lock(path: Path):
    """Serialize model-metric read/modify/write across worker processes."""
    lock_path = path.with_name(f".{path.name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _METRICS_THREAD_LOCK:
        handle = lock_path.open("a+")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()


@dataclass(frozen=True)
class WorkerLease:
    """An in-flight Hawking worker reservation, never a provider identity."""

    lease_id: str
    worker_id: str
    goal_id: str
    workunit_id: str
    worker_attempt_id: str
    model: str
    acquired_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "goal_id": self.goal_id,
            "workunit_id": self.workunit_id,
            "worker_attempt_id": self.worker_attempt_id,
            "model": self.model,
            "acquired_at": self.acquired_at,
        }


class WorkerAdmissionError(RuntimeError):
    """An explicit Hawking/provider admission policy refused another worker."""


def recommend_concurrency(
    task_class: str,
    *,
    active: int = 0,
    accepted: int = 0,
    attempts: int = 0,
    failure_rate: float = 0.0,
    budget_remaining_usd: float | None = None,
) -> dict[str, Any]:
    """Recommend an evidence-based target; it is not a hard admission ceiling.

    Read/research lanes can use a large frontier, while complex planning and
    development get enough lanes to form a real work graph. Parallel mutation
    still requires disjoint effect scopes. Poor outcomes reduce the target;
    this function never launches or cancels work.
    """
    kind = str(task_class or "general").strip().lower()
    base = 10 if kind in {"research", "deterministic", "source", "document"} else 6
    if kind in {"planning", "architecture", "review", "synthesis"}:
        base = 8
    if kind in {"repo_engineering", "debugging", "desktop", "mutation"}:
        base = 6
    if attempts >= 3 and (float(failure_rate) >= 0.5 or accepted == 0):
        base = max(1, base - 1)
    if budget_remaining_usd is not None and float(budget_remaining_usd) < 0.50:
        base = min(base, 2)
    # This is a sizing recommendation, not a hard admission ceiling.  Keep
    # the evidence-based target finite while allowing the frontier to admit
    # every independent runnable lane when resources permit.
    target = max(1, base)
    return {
        "schema": "hawking.auto.concurrency_recommendation.v1",
        "task_class": kind,
        "target": target,
        "active": max(0, int(active)),
        "slots": max(0, target - max(0, int(active))),
        "evidence": {"accepted": max(0, int(accepted)), "attempts": max(0, int(attempts)), "failure_rate": max(0.0, min(1.0, float(failure_rate)))},
        "claim_boundary": "recommendation only; RemoteWorkerAdmission remains authoritative",
    }


class RemoteWorkerAdmission:
    """Admit remote cognition without an artificial Hawking count ceiling.

    ``None`` means unbounded by Hawking worker count.  Explicit finite values
    remain available for an operator/provider resource policy, but the default
    path must not turn an arbitrary 5/15 slot value into a project barrier.
    """

    def __init__(
        self,
        *,
        max_workers: Optional[int] = MAX_REMOTE_COGNITION_WORKERS,
        max_premium_workers: Optional[int] = MAX_PREMIUM_WORKERS,
    ) -> None:
        self.max_workers = (
            None if max_workers is None else max(1, int(max_workers))
        )
        self.max_premium_workers = (
            None if max_premium_workers is None else max(1, int(max_premium_workers))
        )
        self._lock = threading.RLock()
        self._leases: Dict[str, WorkerLease] = {}

    @staticmethod
    def _is_premium(model: str) -> bool:
        return str(model or "").strip().lower().startswith("moonshotai/kimi-k3")

    def acquire(
        self,
        *,
        model: str,
        goal_id: str = "",
        workunit_id: str = "",
        worker_attempt_id: str = "",
        worker_id: Optional[str] = None,
    ) -> WorkerLease:
        model = str(model or "").strip()
        if not model:
            raise WorkerAdmissionError("remote worker model is required")
        with self._lock:
            if self.max_workers is not None and len(self._leases) >= self.max_workers:
                raise WorkerAdmissionError("Hawking remote concurrency limit reached")
            premium = sum(self._is_premium(item.model) for item in self._leases.values())
            if (
                self._is_premium(model)
                and self.max_premium_workers is not None
                and premium >= self.max_premium_workers
            ):
                raise WorkerAdmissionError("Hawking premium-model concurrency limit reached")
            lease = WorkerLease(
                lease_id=f"LEASE-{uuid.uuid4().hex}",
                worker_id=str(worker_id or f"WORKER-{uuid.uuid4().hex}"),
                goal_id=str(goal_id or ""),
                workunit_id=str(workunit_id or ""),
                worker_attempt_id=str(worker_attempt_id or ""),
                model=model,
                acquired_at=time.time(),
            )
            self._leases[lease.lease_id] = lease
            return lease

    def release(self, lease_id: str) -> bool:
        with self._lock:
            return self._leases.pop(str(lease_id), None) is not None

    @contextmanager
    def lease(self, **kwargs: Any):
        current = self.acquire(**kwargs)
        try:
            yield current
        finally:
            self.release(current.lease_id)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [item.to_dict() for item in self._leases.values()]


def _redact(value: Any, *, limit: int = 1600) -> str:
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]", text)
    return text[:limit]


def _redact_packet_value(value: Any, *, limit: int = 1600) -> Any:
    """Redact packet data without flattening structured durable references."""
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name.casefold() in {
                "api_key", "access_token", "authorization", "password",
                "secret", "private_key", "bearer", "token",
            }:
                out[name] = "[redacted]"
            else:
                out[name] = _redact_packet_value(child, limit=limit)
        return out
    if isinstance(value, list):
        return [_redact_packet_value(item, limit=limit) for item in value[:32]]
    if isinstance(value, tuple):
        return [_redact_packet_value(item, limit=limit) for item in value[:32]]
    if isinstance(value, str):
        return _redact(value, limit=limit)
    return value


def _tool_names(request: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(request, Mapping):
        return []
    names: List[str] = []
    tools = request.get("tools")
    if not isinstance(tools, list):
        return names
    for item in tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        name = function.get("name") if isinstance(function, Mapping) else item.get("name")
        if name and str(name) not in names:
            names.append(str(name)[:120])
    return names[:24]


def classify_request(request: Optional[Mapping[str, Any]] = None, *, text: str = "") -> Dict[str, Any]:
    """Classify work from explicit signals without asking a model to self-route."""
    source = request_text(request) if isinstance(request, Mapping) else ""
    # Request text and direct text must share one normalized classifier input.
    # Previously only the direct ``text=`` path case-folded, so an OpenAI
    # message mentioning ``macOS`` bypassed the ``desktop`` rule.
    source = " ".join(part for part in (source, str(text or "")) if part).casefold().strip()
    tools = _tool_names(request)
    if tools:
        source = f"{source} {' '.join(tools).casefold()}".strip()

    # Specific physical-world classes must win over generic words such as
    # "review" or "visual".  A desktop accessibility repair that asks for an
    # independent review is still desktop work, and must not silently route as
    # generic Kimi review work.
    rules = (
        ("browser", ("playwright", "browser", "dom", "web page", "web ui")),
        ("desktop", ("macos", "desktop", "accessibility", "axuielement", "screen capture", "cg event")),
        ("document", ("pdf", "docx", "pptx", "xlsx", "ocr", "document extraction")),
        ("reverse_engineering", ("rizin", "ghidra", "pyghidra", "decompile", "binary analysis", "reverse engineering")),
        ("physical_engineering", ("egpu", "hardware", "device placement", "thermal", "gpu lane", "physical engineering")),
        ("multimodal", ("multimodal", "image", "visual")),
        ("benchmark", ("benchmark", "throughput", "tokens per second", "tps")),
        ("model_science", ("modellake", "model science", "representation", "decode")),
        # Explicit implementation/test language is an execution request even
        # when it also asks for audit or review; Auto must not collapse that
        # into a two-worker review topology.
        ("repo_engineering", ("repo", "repository", "implement", "edit", "write", "git", "test")),
        ("architecture", ("architecture", "system design", "long horizon", "odyssey", "consolidate")),
        ("review", ("review", "audit", "adversarial", "falsify", "critique")),
        ("planning", ("plan", "roadmap", "decompose", "phase plan")),
        ("debugging", ("debug", "failure", "repair", "regression", "broken")),
        ("research", ("research", "search", "inspect", "identify", "find", "explain")),
    )
    task_class = "general"
    signals: List[str] = []
    for candidate, markers in rules:
        hits = [marker for marker in markers if marker in source]
        if hits:
            task_class = candidate
            signals.extend(hits[:4])
            break
    if not source or (not tools and source in {"hello", "hi", "thanks"}):
        task_class = "deterministic" if tools else "general"
    complexity_score = len(signals)
    complexity_score += min(4, len(re.findall(r"\b(and|then|each|every|multiple|parallel|all)\b", source)))
    complexity_score += min(4, sum(
        marker in source for marker in (
            "decompose", "independent", "audit", "review", "implement",
            "architecture", "long horizon", "phase", "workunit",
        )
    ))
    complexity_score += 1 if len(source) > 500 else 0
    complexity = "simple" if complexity_score <= 2 else "compound" if complexity_score <= 5 else "complex"
    required = {
        "deterministic": ["fs.read", "git.status"],
        "research": ["fs.search", "fs.read"],
        "repo_engineering": ["fs.search", "fs.read", "repo.edit", "tests.run", "git.status", "git.diff"],
        "debugging": ["fs.search", "fs.read", "tests.run", "repo.edit", "git.diff"],
        "planning": ["fs.search", "fs.read", "git.status"],
        "architecture": ["fs.search", "fs.read", "git.status", "git.diff"],
        "review": ["fs.search", "fs.read", "git.diff"],
        "browser": ["browser.observe", "browser.find", "browser.verify"],
        "desktop": ["macos.observe", "macos.changed"],
        "document": ["fs.read", "receipt.read"],
        "reverse_engineering": ["fs.read", "receipt.read"],
        "model_science": ["fs.read", "receipt.read", "tests.run"],
        "benchmark": ["tests.run", "receipt.read"],
        "multimodal": ["fs.read"],
        "physical_engineering": ["receipt.read", "fs.read", "tests.run"],
        "synthesis": ["receipt.read", "fs.read"],
        "general": [],
    }.get(task_class, [])
    return {
        "task_class": task_class,
        "complexity": complexity,
        "complexity_score": complexity_score,
        "signals": signals,
        "required_capabilities": required,
        "tool_names": tools,
        "text_chars": len(source),
    }


def _available_models(allowed_models: Iterable[str]) -> List[str]:
    allowed = {str(item).strip() for item in allowed_models if str(item).strip()}
    return [model for model in REMOTE_AUTO_ROSTER if not allowed or model in allowed]


def _routing_task_class(task_class: str) -> str:
    value = str(task_class or "general").strip().lower()
    return TASK_CLASS_ALIASES.get(value, value)


def _performance_rows(workspace: str | Path | None) -> Dict[str, Mapping[str, Any]]:
    if not workspace:
        return {}
    path = Path(workspace).expanduser().resolve() / ".hawking" / "auto" / "model-performance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    rows = payload.get("models") if isinstance(payload, Mapping) else {}
    return {
        str(key): value for key, value in rows.items()
        if isinstance(value, Mapping)
    } if isinstance(rows, Mapping) else {}


def _model_metric_row(
    rows: Mapping[str, Mapping[str, Any]],
    model: str,
    task_class: str,
) -> Dict[str, Any]:
    routed = _routing_task_class(task_class)
    candidates = [
        rows.get(f"{model}::{routed}"),
        rows.get(f"{model}::{task_class}"),
        rows.get(f"{model}::general"),
    ]
    merged: Dict[str, Any] = {}
    # Prefer task-specific evidence, then use general evidence only for fields
    # absent from the task row. This prevents a large easy-task history from
    # hiding a model's hard architecture failures.
    for candidate in reversed(candidates):
        if isinstance(candidate, Mapping):
            merged.update(candidate)
    return merged


def model_routing_value(
    model: str,
    task_class: str,
    *,
    performance_rows: Mapping[str, Mapping[str, Any]] | None = None,
    independent: bool = False,
) -> Dict[str, Any]:
    """Return inspectable measured value for one model/task class.

    This is deliberately a routing signal, not an acceptance decision. New
    models get an explicit exploration opportunity; established models are
    compared using acceptance, quality, cost, latency, failure, and (for
    reviewers) independence evidence.
    """
    row = _model_metric_row(performance_rows or {}, str(model), task_class)
    attempts = max(0, int(row.get("attempts") or 0))
    accepted = max(0, int(row.get("accepted") or 0))
    cost = max(0.0, float(row.get("cost_usd") or 0.0))
    wall = max(0.0, float(row.get("wall_time_s") or 0.0))
    failures = row.get("failure_classes") if isinstance(row.get("failure_classes"), Mapping) else {}
    transport_failures = max(0, int(row.get("provider_transport_failures") or 0))
    failure_count = sum(max(0, int(value or 0)) for value in failures.values()) + transport_failures
    reversals = max(0, int(row.get("review_overturns") or 0))
    tool_samples = max(0, int(row.get("tool_compliance_samples") or 0))
    tool_successes = max(0, int(row.get("tool_compliance_successes") or 0))
    grounding_samples = max(0, int(row.get("source_grounding_samples") or 0))
    grounding_sum = max(0.0, float(row.get("source_grounding_sum") or 0.0))
    semantic_samples = max(0, int(row.get("provider_semantic_samples") or 0))
    semantic_successes = max(0, int(row.get("provider_semantic_successes") or 0))
    terminal_samples = max(0, int(row.get("terminal_packet_samples") or 0))
    terminal_successes = max(0, int(row.get("terminal_packet_successes") or 0))
    tool_rate = tool_successes / max(1, tool_samples) if tool_samples else 1.0
    grounding_rate = grounding_sum / max(1, grounding_samples) if grounding_samples else 1.0
    semantic_rate = semantic_successes / max(1, semantic_samples) if semantic_samples else 1.0
    terminal_rate = terminal_successes / max(1, terminal_samples) if terminal_samples else 1.0
    acceptance_rate = (accepted + 0.25) / (attempts + 1.0)
    failure_rate = min(1.0, failure_count / max(1, attempts))
    quality = max(
        0.05,
        acceptance_rate * (
            0.40
            + 0.20 * tool_rate
            + 0.15 * grounding_rate
            + 0.15 * semantic_rate
            + 0.10 * terminal_rate
        )
        - (0.35 * failure_rate)
        - (0.20 * reversals / max(1, attempts)),
    )
    mean_wall = wall / max(1, attempts)
    mean_cost = cost / max(1, attempts)
    # Keep units bounded and comparable when one provider reports zero cost or
    # latency. Unknown economics are penalized mildly, not treated as free.
    economics = max(0.25, (1.0 + mean_cost) * (1.0 + mean_wall / 120.0))
    independence_bonus = 1.0 + (0.15 if independent and attempts == 0 else 0.0)
    value = (quality * independence_bonus * max(0.05, 1.0 - 0.5 * failure_rate)) / economics
    return {
        "model": str(model), "task_class": _routing_task_class(task_class),
        "attempts": attempts, "accepted": accepted,
        "acceptance_rate": round(acceptance_rate, 6),
        "failure_rate": round(failure_rate, 6),
        "transport_failure_rate": round(transport_failures / max(1, attempts), 6),
        "transport_failures": transport_failures,
        "tool_compliance_rate": round(tool_rate, 6),
        "source_grounding_rate": round(grounding_rate, 6),
        "provider_semantic_success_rate": round(semantic_rate, 6),
        "terminal_packet_success_rate": round(terminal_rate, 6),
        "mean_cost_usd": round(mean_cost, 8),
        "mean_wall_time_s": round(mean_wall, 6),
        "routing_value": round(value, 8),
        "exploration_eligible": attempts == 0,
        "independence_bonus": round(independence_bonus, 4),
    }


def select_measured_model(
    task_class: str,
    candidates: Sequence[str],
    *,
    workspace: str | Path | None = None,
    preferred: str = "",
    role: str = "",
    escalation: bool = False,
) -> tuple[str, Dict[str, Any]]:
    """Select by measured value with bounded GLM exploration and escalation."""
    live = [str(model).strip() for model in candidates if str(model).strip()]
    if not live:
        return str(preferred or ""), {"reason": "no_candidates", "candidates": []}
    rows = _performance_rows(workspace)
    routing_class = _routing_task_class(task_class)
    role_text = str(role or "").casefold()
    if escalation:
        escalation_live = [model for model in live if model in ESCALATION_AUTO_MODELS]
        if escalation_live:
            if "fals" in role_text or "review" in role_text or "advers" in role_text:
                order = ["nvidia/nemotron-3-ultra-550b-a55b", "qwen/qwen3.8-2.4t-a95b"]
            else:
                order = ["qwen/qwen3.8-2.4t-a95b", "nvidia/nemotron-3-ultra-550b-a55b"]
            for model in order:
                if model in escalation_live:
                    return model, {"reason": "information_gain_escalation", "value": model_routing_value(model, routing_class, performance_rows=rows, independent=True)}
    glm = "z-ai/glm-5.3"
    if glm in live and routing_class in GLM_TRIAL_CLASSES:
        glm_value = model_routing_value(glm, routing_class, performance_rows=rows)
        if glm_value["exploration_eligible"]:
            return glm, {"reason": "unmeasured_glm_trial", "value": glm_value}
    values = [
        (model, model_routing_value(model, routing_class, performance_rows=rows, independent=bool(role and model != preferred)))
        for model in live if model in CORE_AUTO_MODELS or model not in ESCALATION_AUTO_MODELS
    ]
    if not values:
        values = [(model, model_routing_value(model, routing_class, performance_rows=rows)) for model in live]
    # A route that has accumulated only semantic failures must not monopolize
    # a mutation frontier merely because it has a small historical acceptance
    # prior. Rotate among the measured pool by least exposure until one route
    # demonstrates semantic success. This is bounded diversification, not a
    # blind worker multiplier: transport and resource gates remain authoritative.
    measured = [item for item in values if item[1]["attempts"] > 0]
    if len(measured) >= 2 and all(
            item[1]["provider_semantic_success_rate"] == 0.0
        for item in measured
    ):
        # When every measured route has zero recorded semantic success, raw
        # transport rate alone is not value: it can make one imperfect route
        # monopolize all fresh retries. Prefer a route with prior accepted
        # work first (the acceptance counter is an independent Hawking-owned
        # signal), then rotate among the remaining candidates by exposure and
        # transport health. Pure no-success pools still get least-exposed
        # diversification; fall back to the complete pool only when every
        # route is transport-dead.
        transport_live = [
            item for item in measured
            if float(item[1].get("transport_failure_rate") or 0.0) < 0.9
        ]
        diversity_pool = transport_live or measured
        selected, selected_value = min(
            diversity_pool,
            key=lambda item: (
                -item[1]["accepted"],
                item[1]["attempts"],
                item[1]["transport_failure_rate"],
                0 if item[0] != preferred else 1,
                item[0],
            ),
        )
        return selected, {
            "reason": "semantic_failure_diversification",
            "role": role,
            "preferred": preferred,
            "selected": selected_value,
            "preferred_value": next(
                (value for model, value in values if model == preferred), None
            ),
            "candidates": [value for _, value in values],
        }
    # Initial biases break ties; measured value wins after evidence exists.
    preferred_value = next((value for model, value in values if model == preferred), None)
    selected, selected_value = max(
        values,
        key=lambda item: (item[1]["routing_value"], 1 if item[0] == preferred else 0),
    )
    return selected, {
        "reason": "measured_value" if any(value["attempts"] for _, value in values) else "initial_role_bias",
        "role": role,
        "preferred": preferred,
        "selected": selected_value,
        "preferred_value": preferred_value,
        "candidates": [value for _, value in values],
    }


def select_active_diversity_route(
    task_class: str,
    candidates: Sequence[str],
    selected: str,
    *,
    active_model_counts: Mapping[str, int],
    workspace: str | Path | None = None,
    minimum_active_lanes: int = 4,
    dominant_share: float = 0.75,
) -> tuple[str, Dict[str, Any]]:
    """Keep a qualified alternate route alive when one model dominates.

    Measured value remains the primary selector.  Once a route owns at least
    75% of a sufficiently large active pool, however, Auto reserves the next
    admission for the least-exposed candidate whose measured transport rate
    is not dead.  This is a bounded exploration floor for parallel execution,
    not a worker-count ceiling or a blind retry policy.
    """
    selected = str(selected or "").strip()
    counts = {
        str(model).strip(): max(0, int(count or 0))
        for model, count in (active_model_counts or {}).items()
        if str(model).strip()
    }
    total_active = sum(counts.values())
    selected_count = counts.get(selected, 0)
    if (
        not selected
        or total_active < max(1, int(minimum_active_lanes))
        or selected_count / max(1, total_active) < float(dominant_share)
    ):
        return selected, {"reason": "measured_value", "selected": selected}
    pool = [str(model).strip() for model in candidates if str(model).strip()]
    rows = _performance_rows(workspace)
    alternatives = []
    for model in pool:
        if model == selected:
            continue
        value = model_routing_value(
            model,
            _routing_task_class(task_class),
            performance_rows=rows,
            independent=True,
        )
        if float(value.get("transport_failure_rate") or 0.0) >= 0.90:
            continue
        alternatives.append((model, value))
    if not alternatives:
        return selected, {"reason": "measured_value_no_live_alternate", "selected": selected}
    alternate, value = min(
        alternatives,
        key=lambda item: (
            counts.get(item[0], 0),
            float(item[1].get("transport_failure_rate") or 0.0),
            -float(item[1].get("routing_value") or 0.0),
            item[0],
        ),
    )
    return alternate, {
        "reason": "active_lane_diversification",
        "selected": alternate,
        "replaced": selected,
        "active_model_counts": counts,
        "selected_value": value,
        "dominant_share": float(dominant_share),
    }


def _authoritative_active_model_counts(
    root: Path,
    goal_id: str,
    queue: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Count active routes from the queue plus authoritative WorkUnit records.

    The queue is a durable projection and can lag a just-admitted owner.  Route
    diversity must therefore use the WorkUnit ledger as a second source of
    truth, while de-duplicating queue rows that already name the same owner.
    """
    counts: Dict[str, int] = {}
    counted_ids: set[str] = set()

    def add(item: Mapping[str, Any], *, workunit_id: str = "") -> None:
        identifier = str(workunit_id or item.get("workunit_id") or "").strip()
        if identifier:
            if identifier in counted_ids:
                return
            counted_ids.add(identifier)
        model = str(
            item.get("worker_model")
            or item.get("model_override")
            or item.get("selected_model")
            or (
                item.get("auto_route", {}).get("selected_model")
                if isinstance(item.get("auto_route"), Mapping) else ""
            )
            or ""
        ).strip()
        if model:
            counts[model] = counts.get(model, 0) + 1

    workunits_root = root / ".hawking" / "workunits"
    if workunits_root.is_dir():
        for state_path in workunits_root.glob("WORKUNIT-*.json"):
            try:
                record = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, Mapping):
                continue
            if str(record.get("goal_id") or "").strip() != str(goal_id).strip():
                continue
            if str(record.get("state") or "").upper() not in {"RUNNING", "ADMITTED"}:
                continue
            add(record, workunit_id=str(record.get("workunit_id") or state_path.stem))
    # Queue rows are only a fallback for an owner whose authoritative record
    # has not materialized yet. If both exist, the WorkUnit record wins because
    # it contains the route actually assigned to the live owner.
    for item in queue:
        if str(item.get("state") or "").upper() in {"ADMITTED", "RUNNING"}:
            add(item)
    return counts


def _first_available(preferred: str, candidates: Sequence[str]) -> str:
    return preferred if preferred in candidates else (candidates[0] if candidates else preferred)


def build_cognition_plan(
    request: Optional[Mapping[str, Any]] = None,
    *,
    allowed_models: Iterable[str] = REMOTE_AUTO_ROSTER,
    pinned_model: Optional[str] = None,
    budget_usd: Optional[float] = None,
    goal_id: str = "",
) -> Dict[str, Any]:
    """Build a bounded topology; actual worker identities are assigned at dispatch."""
    classification = classify_request(request)
    candidates = _available_models(allowed_models)
    workspace = request.get("hawking_workspace") if isinstance(request, Mapping) else None
    requested = str(pinned_model or "").strip()
    if requested and requested not in candidates:
        requested = _first_available(requested, candidates)
    preferred_primary = requested or _first_available(
        MODEL_PRIORS.get(classification["task_class"], MODEL_PRIORS["general"]), candidates
    )
    task_class = classification["task_class"]
    primary, primary_route = select_measured_model(
        task_class, candidates, workspace=workspace, preferred=preferred_primary,
        role="primary",
    )
    complex_task = classification["complexity"] == "complex"
    target = recommend_concurrency(task_class)["target"]
    workers: List[Dict[str, Any]] = []

    def add(slot: str, model: str, role: str, purpose: str) -> None:
        if not model:
            return
        workers.append({
            "slot_id": slot,
            "model": model,
            "role": role,
            "purpose": purpose,
            "identity": "assigned by Hawking at dispatch",
            "state": "planned",
        })

    if requested:
        add("W1", primary, "pinned", "complete the bounded pinned WorkUnit")
        topology = "pinned"
    elif task_class in {"architecture", "planning", "synthesis"} and complex_task and len(candidates) >= 2:
        kimi = _first_available("moonshotai/kimi-k3", candidates)
        deepseek = _first_available("deepseek/deepseek-v4.1-flash", candidates)
        glm, glm_route = select_measured_model(
            task_class, candidates, workspace=workspace, preferred="z-ai/glm-5.3",
            role="alternative_synthesis",
        )
        lanes = (
            (kimi, "plan", "produce a bounded dependency graph"),
            (deepseek, "source-map", "map current owners and constraints"),
            (glm, "alternative", "produce an independent crossover synthesis"),
            (deepseek, "falsifier", "seek contradictions and obsolete paths"),
            (deepseek, "scaffold", "prepare independently actionable tranches"),
            (kimi, "synthesis", "converge grounded evidence into a plan"),
            (glm, "verification", "identify focused acceptance evidence"),
            (kimi, "adversarial", "challenge high-risk assumptions"),
        )
        for index, (model, role, purpose) in enumerate(lanes[:target], start=1):
            add(f"W{index}", model, role, purpose)
        topology = "parallel planning lanes → evidence convergence"
    elif task_class in {"repo_engineering", "debugging"} and complex_task:
        deepseek = _first_available("deepseek/deepseek-v4.1-flash", candidates)
        implementer, implementer_route = select_measured_model(
            task_class, candidates, workspace=workspace, preferred=deepseek,
            role="implementation",
        )
        glm, glm_route = select_measured_model(
            task_class, candidates, workspace=workspace, preferred="z-ai/glm-5.3",
            role="falsification",
        )
        reviewer, reviewer_route = select_measured_model(
            "review", candidates, workspace=workspace, preferred="moonshotai/kimi-k3",
            role="review",
        )
        lanes = (
            ("source-map", "identify owners, tests, and effect scopes"),
            ("implementation", "execute one admitted effect scope"),
            ("test-map", "identify deterministic verification obligations"),
            ("falsifier", "seek incompatible callers and regressions"),
            ("repair", "prepare a disjoint repair tranche"),
            ("review", "review grounded diff and evidence"),
        )
        lane_models = {
            "source-map": deepseek, "implementation": implementer,
            "test-map": deepseek, "falsifier": glm if "z-ai/glm-5.3" in candidates else deepseek,
            "repair": implementer, "review": reviewer,
        }
        for index, (role, purpose) in enumerate(lanes[:target], start=1):
            add(f"W{index}", lane_models.get(role, deepseek), role, purpose)
        topology = "parallel development lanes; writers require disjoint effect scopes"
    elif task_class in {"research", "model_science"} and complex_task:
        researcher, researcher_route = select_measured_model(
            task_class, candidates, workspace=workspace, preferred="deepseek/deepseek-v4.1-flash",
            role="source_mapping",
        )
        falsifier = "z-ai/glm-5.3" if "z-ai/glm-5.3" in candidates else researcher
        add("W1", researcher, "researcher", "bounded hypothesis A")
        add("W2", falsifier, "falsifier", "independent falsifier")
        add("W3", _first_available("moonshotai/kimi-k3", candidates), "synthesizer", "converge accepted evidence")
        topology = "research × 2 → synthesis"
    elif task_class in {"review", "multimodal"} and complex_task:
        add("W1", primary, "specialist", "produce an independent bounded assessment")
        add("W2", primary, "reviewer", "cross-check the specialist evidence")
        topology = "specialist → reviewer"
    else:
        add("W1", primary, "implementer" if task_class in {"repo_engineering", "debugging"} else "worker", "complete one bounded WorkUnit")
        topology = "single worker"

    if budget_usd is not None:
        try:
            budget = max(0.0, float(budget_usd))
        except (TypeError, ValueError):
            budget = 0.0
    else:
        budget = None
    return {
        "schema": AUTO_PLAN_SCHEMA,
        "plan_id": f"PLAN-{uuid.uuid4().hex[:12].upper()}",
        "goal_id": str(goal_id or ""),
        "created_at": time.time(),
        "task": classification,
        "topology": topology,
        "workers": workers[:],
        "policy": {
            "worker_policy": "PINNED" if requested else "AUTO",
            "allowed_models": list(candidates),
            "max_remote_workers": MAX_REMOTE_COGNITION_WORKERS,
            "max_premium_workers": MAX_PREMIUM_WORKERS,
            "remote_concurrency": "unbounded by Hawking count; resource/provider gates remain authoritative",
            "effect_scope_rule": "parallel writes require disjoint scope leases; read-only lanes may fan out",
            "budget_authority": "parent Goal",
            "measured_value_routing": True,
            "task_class_alias": _routing_task_class(task_class),
            "primary_route": primary_route,
            "role_biases": {key: list(value) for key, value in MODEL_ROLE_BIASES.items()},
            "core_models": list(CORE_AUTO_MODELS),
            "escalation_models": list(ESCALATION_AUTO_MODELS),
            "escalation_rule": "Qwen/Nemotron require explicit information-gain or disagreement escalation",
            "exploration_rule": "unmeasured GLM receives bounded comparable trials before measured rebalance",
        },
        "budget_usd": budget,
        "execution": "planned; worker identities are provisioned only by Hawking",
        "claim_boundary": "routing plan only; no provider request or worker was started",
    }


def worker_packet(
    goal: Mapping[str, Any],
    workunit: Mapping[str, Any],
    *,
    role: str = "worker",
    plan_id: str = "",
    current_evidence: Sequence[Any] = (),
    failed_approaches: Sequence[Any] = (),
    dependencies: Sequence[Any] = (),
    authority: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the bounded memory image delivered to a fresh worker instance."""
    goal_id = str(goal.get("goal_id") or workunit.get("goal_id") or "")
    workunit_id = str(workunit.get("workunit_id") or goal.get("workunit_id") or "")
    model = str(workunit.get("worker_model") or goal.get("resident_assignment") or "")
    worker_id = str(workunit.get("worker_id") or f"WORKER-{uuid.uuid4().hex}")
    worker_attempt_id = str(workunit.get("worker_attempt_id") or f"ATTEMPT-{uuid.uuid4().hex}")
    authorized = list(
        (authority or {}).get("capabilities")
        if isinstance(authority, Mapping) and isinstance((authority or {}).get("capabilities"), list)
        else goal.get("authorized_capabilities") or []
    )[:32]
    task_class = str((goal.get("classification") or goal.get("task_class") or "general"))
    playbooks = {
        "repo_engineering": ["identify owner", "inspect focused test", "act", "test", "repair", "diff"],
        "debugging": ["reproduce", "localize", "repair", "rerun", "diff"],
        "research": ["search bounded corpus", "read owner", "record evidence", "stop when grounded"],
        "review": ["inspect evidence", "challenge claim", "record independent finding"],
        "browser": ["observe the fixed browser session", "resolve one semantic target", "act only under Goal authority", "verify the visible delta"],
        "desktop": ["observe native WorldState", "inspect bounded AX facts", "obtain a separate input lease before action", "verify semantic delta"],
        "document": ["inspect native structure first", "extract bounded spans", "OCR only unresolved regions", "record confidence and source engine"],
        "reverse_engineering": ["confirm authorized binary scope", "inspect cached metadata first", "record derived interpretation with engine provenance"],
        "physical_engineering": ["read measured resource state", "avoid protected lanes", "measure before changing placement", "record physical evidence"],
    }
    packet = {
        "schema": WORKER_PACKET_SCHEMA,
        "packet_id": f"PACKET-{uuid.uuid4().hex[:12].upper()}",
        "created_at": time.time(),
        "plan_id": str(plan_id or goal.get("auto_plan", {}).get("plan_id") or ""),
        "goal_id": goal_id,
        "workunit_id": workunit_id,
        "worker_id": worker_id,
        "worker_attempt_id": worker_attempt_id,
        "worker": {
            "model": model,
            "provider": "openrouter" if "/" in model else "hawking",
            "role": str(role or "worker"),
            "state": str(workunit.get("state") or "QUEUED"),
        },
        "root_goal": _redact(goal.get("objective"), limit=1200),
        "parent_goal_objective": _redact(
            goal.get("parent_goal_objective") or goal.get("objective"), limit=1600
        ),
        "workunit": _redact(workunit.get("objective") or goal.get("objective"), limit=1200),
        "accepted_phase": str(
            goal.get("accepted_phase") or goal.get("phase") or ""
        )[:200],
        "attachments": [
            _redact_packet_value(item, limit=700)
            for item in (goal.get("attachments") or [])[:8]
            if isinstance(item, Mapping)
        ],
        "swift_helper_state": _redact(goal.get("swift_helper_state") or {}, limit=1200),
        "completion_obligations": [_redact(item, limit=500) for item in (goal.get("acceptance") or workunit.get("acceptance") or [])[:12]],
        "authority": dict(authority or {
            "tools": "Hawking registry only",
            "resources": goal.get("authority_boundary") or "Goal-scoped resources",
            "mutation": "Goal contract and single-writer lock",
            "forbidden": ["Flash/Pulsar", "production promotion", "provider self-authorization"],
        }),
        "capability_catalog": authorized,
        "playbook": playbooks.get(task_class, ["inspect", "act only within scope", "verify", "report evidence"]),
        "current_evidence": [_redact(item, limit=700) for item in list(current_evidence)[:20]],
        "failed_approaches": [_redact(item, limit=500) for item in list(failed_approaches)[:12]],
        "dependencies": [_redact(item, limit=500) for item in list(dependencies)[:12]],
        "output_contract": "return structured evidence; Hawking decides acceptance",
        "budget": {
            "authorized_usd": goal.get("budget_authorized_usd"),
            "plan": goal.get("budget_plan") or {},
            "authority": "parent Goal",
        },
        "claim_boundary": "Worker Packet is bounded context, not a completion receipt.",
    }
    packet["digest"] = hashlib.sha256(
        json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return packet


def packet_path(workspace: str | Path, workunit_id: str, worker_attempt_id: Optional[str] = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(workunit_id or "worker"))
    attempt = re.sub(r"[^A-Za-z0-9_.-]", "_", str(worker_attempt_id or "").strip())
    suffix = f".{attempt}" if attempt else ""
    return Path(workspace).expanduser().resolve() / ".hawking" / "auto" / "packets" / f"{safe}{suffix}.json"


def persist_worker_packet(workspace: str | Path, packet: Mapping[str, Any]) -> Path:
    workunit_id = str(packet.get("workunit_id") or packet.get("packet_id") or "worker")
    attempt_path = packet_path(
        workspace,
        workunit_id,
        str(packet.get("worker_attempt_id") or "") or None,
    )
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(attempt_path, dict(packet))

    # Keep the old WorkUnit-only path as a symlink to the latest attempt.  The
    # attempt-scoped file is authoritative and preserves concurrent/restarted
    # workers; the stable path is only a compatibility read door for existing
    # callers and never duplicates the packet bytes.
    legacy_path = packet_path(workspace, workunit_id)
    if legacy_path != attempt_path:
        try:
            if not legacy_path.exists() and not legacy_path.is_symlink():
                os.symlink(attempt_path.name, legacy_path)
            elif legacy_path.is_symlink():
                replacement = legacy_path.with_name(
                    f".{legacy_path.name}.{uuid.uuid4().hex}.tmp"
                )
                os.symlink(attempt_path.name, replacement)
                os.replace(replacement, legacy_path)
            else:
                # A pre-existing legacy regular file is user/state data; do
                # not overwrite it merely to publish a compatibility alias.
                return attempt_path
        except OSError:
            # The unique attempt record is still durable even if a filesystem
            # cannot provide the optional legacy alias.
            return attempt_path
        return legacy_path
    return attempt_path


def _record_model_outcome_locked(
    path: Path,
    *,
    model: str,
    task_class: str,
    accepted: bool,
    cost_usd: float,
    repaired: bool,
    review_overturn: bool,
    wall_time_s: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    tool_calls: int,
    failure_class: Optional[str],
    tool_compliance: Optional[bool],
    source_grounding_quality: Optional[float],
    provider_semantic_success: Optional[bool],
    terminal_packet_success: Optional[bool],
    provider_transport_failure: bool,
    independence_value: Optional[float],
) -> Dict[str, Any]:
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        current = {}
    if not isinstance(current, dict):
        current = {}
    current.setdefault("schema", METRICS_SCHEMA)
    current.setdefault("updated_at", time.time())
    rows = current.setdefault("models", {})
    if not isinstance(rows, dict):
        rows = {}
        current["models"] = rows
    key = f"{str(model).strip()}::{str(task_class).strip() or 'general'}"
    row = rows.get(key)
    if not isinstance(row, dict):
        row = {
        "model": str(model), "task_class": str(task_class or "general"),
        "attempts": 0, "accepted": 0, "repaired": 0,
        "review_overturns": 0, "cost_usd": 0.0,
        "wall_time_s": 0.0, "input_tokens": 0, "output_tokens": 0,
        "tool_calls": 0, "failure_classes": {},
        "tool_compliance_samples": 0, "tool_compliance_successes": 0,
        "source_grounding_samples": 0, "source_grounding_sum": 0.0,
        "provider_semantic_samples": 0, "provider_semantic_successes": 0,
        "terminal_packet_samples": 0, "terminal_packet_successes": 0,
        "provider_transport_failures": 0, "independence_value_sum": 0.0,
        "independence_value_samples": 0,
        }
        rows[key] = row
    row["attempts"] = int(row.get("attempts", 0)) + 1
    row["accepted"] = int(row.get("accepted", 0)) + int(bool(accepted))
    row["repaired"] = int(row.get("repaired", 0)) + int(bool(repaired))
    row["review_overturns"] = int(row.get("review_overturns", 0)) + int(bool(review_overturn))
    row["cost_usd"] = round(float(row.get("cost_usd", 0.0)) + max(0.0, float(cost_usd or 0.0)), 8)
    if wall_time_s is not None:
        row["wall_time_s"] = round(float(row.get("wall_time_s", 0.0)) + max(0.0, float(wall_time_s)), 6)
    if input_tokens is not None:
        row["input_tokens"] = int(row.get("input_tokens", 0)) + max(0, int(input_tokens))
    if output_tokens is not None:
        row["output_tokens"] = int(row.get("output_tokens", 0)) + max(0, int(output_tokens))
    row["tool_calls"] = int(row.get("tool_calls", 0)) + max(0, int(tool_calls or 0))
    if failure_class:
        failures = row.setdefault("failure_classes", {})
        if not isinstance(failures, dict):
            failures = {}
            row["failure_classes"] = failures
        failure_key = str(failure_class)[:120]
        failures[failure_key] = int(failures.get(failure_key, 0)) + 1
    if tool_compliance is not None:
        row["tool_compliance_samples"] = int(row.get("tool_compliance_samples", 0)) + 1
        row["tool_compliance_successes"] = int(row.get("tool_compliance_successes", 0)) + int(bool(tool_compliance))
    if source_grounding_quality is not None:
        row["source_grounding_samples"] = int(row.get("source_grounding_samples", 0)) + 1
        row["source_grounding_sum"] = round(float(row.get("source_grounding_sum", 0.0)) + max(0.0, min(1.0, float(source_grounding_quality))), 6)
    if provider_semantic_success is not None:
        row["provider_semantic_samples"] = int(row.get("provider_semantic_samples", 0)) + 1
        row["provider_semantic_successes"] = int(row.get("provider_semantic_successes", 0)) + int(bool(provider_semantic_success))
    if terminal_packet_success is not None:
        row["terminal_packet_samples"] = int(row.get("terminal_packet_samples", 0)) + 1
        row["terminal_packet_successes"] = int(row.get("terminal_packet_successes", 0)) + int(bool(terminal_packet_success))
    if provider_transport_failure:
        row["provider_transport_failures"] = int(row.get("provider_transport_failures", 0)) + 1
    if independence_value is not None:
        row["independence_value_samples"] = int(row.get("independence_value_samples", 0)) + 1
        row["independence_value_sum"] = round(float(row.get("independence_value_sum", 0.0)) + max(0.0, min(1.0, float(independence_value))), 6)
    current["updated_at"] = time.time()
    atomic_write_json(path, current)
    return row


def record_model_outcome(
    workspace: str | Path,
    *,
    model: str,
    task_class: str,
    accepted: bool,
    cost_usd: float = 0.0,
    repaired: bool = False,
    review_overturn: bool = False,
    wall_time_s: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    tool_calls: int = 0,
    failure_class: Optional[str] = None,
    tool_compliance: Optional[bool] = None,
    source_grounding_quality: Optional[float] = None,
    provider_semantic_success: Optional[bool] = None,
    terminal_packet_success: Optional[bool] = None,
    provider_transport_failure: bool = False,
    independence_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Update durable routing evidence; counters are never provider claims."""
    path = Path(workspace).expanduser().resolve() / ".hawking" / "auto" / "model-performance.json"
    with _performance_lock(path):
        return _record_model_outcome_locked(
            path,
            model=model,
            task_class=task_class,
            accepted=accepted,
            cost_usd=cost_usd,
            repaired=repaired,
            review_overturn=review_overturn,
            wall_time_s=wall_time_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            failure_class=failure_class,
            tool_compliance=tool_compliance,
            source_grounding_quality=source_grounding_quality,
            provider_semantic_success=provider_semantic_success,
            terminal_packet_success=terminal_packet_success,
            provider_transport_failure=provider_transport_failure,
            independence_value=independence_value,
        )


def refill_runnable_frontier(
    workspace: str | Path,
    goal_id: str,
    *,
    efficient_concurrency: Optional[int] = MAX_REMOTE_COGNITION_WORKERS,
    max_new: Optional[int] = None,
    background: bool = True,
) -> Dict[str, Any]:
    """Serialize Auto refill decisions across concurrent terminal owners."""
    root = Path(workspace).expanduser().resolve()
    lock_path = root / ".hawking" / "auto" / "refill-state.json"
    with _performance_lock(lock_path):
        return _refill_runnable_frontier_locked(
            root,
            goal_id,
            efficient_concurrency=efficient_concurrency,
            max_new=max_new,
            background=background,
        )


_AUTO_CONTINUATION_TASK_CLASSES = {
    "P2_PRODUCT_COMMANDS_HWEB": "bug_repair",
    "P4_GOAL_WORKUNIT_AUTO_EXECUTIONPLAN": "routine_implementation",
    "P5_PERCEPTION_ARSENAL": "routine_implementation",
    "P6_GRAVITY_IMPLEMENTATION_CONTRACT_V2": "model_science",
    "P7_NR_EXECUTION_BOUNDARY_V3": "model_science",
    "P8_NOVA_IMPLEMENTATION_CONTRACT_V2": "model_science",
    "P8_NOVA_HELDOUT_GATE_DISCOVERY": "test_design",
    "P8_NOVA_CANDIDATE_LIVE_EVIDENCE": "test_design",
    "P9_DARKMATTER_OWNER_RECONCILIATION_V2": "architecture_planning",
    "P9_DARKMATTER_PHYSICAL_PLAN_IMPLEMENTATION_V1": "model_science",
    "P10_NX_IMPLEMENTATION_CONTRACT_V2": "model_science",
    "P11_NATIVE_CPU_IMPLEMENTATION_CONTRACT_V2": "hardware_reasoning",
    "P12_APPLE_BACKEND": "hardware_reasoning",
    "P13_NVIDIA_BACKEND": "hardware_reasoning",
    "P13_NVIDIA_IMPLEMENTATION_CONTRACT_V2": "hardware_reasoning",
    "P14_PCIE_IMPLEMENTATION_CONTRACT_V2": "hardware_reasoning",
    "P15_AUTOTUNING_FOCUSED_CONTRACT_V4": "routine_implementation",
    "P16_ODYSSEY_MODELLAKE": "routine_implementation",
    "P17_CODEBASE_COLLAPSE": "large_refactor",
    "P18_TUNNEL": "bug_repair",
    "P19_SANDBOX": "routine_implementation",
    "P20_FINAL_HANDOFF_FOCUSED_REVIEW_V3": "architecture_planning",
}


def _production_focus_files(item: Mapping[str, Any]) -> set[str]:
    """Return the declared non-test/non-report paths a lane may mutate.

    Auto may expand only pre-planned accepted heads that name at least one
    production/configuration path.  A focused test alone is useful evidence,
    but it is not enough authority to create a mutation lane.
    """
    paths: set[str] = set()
    for raw in (item.get("focus_files") or []):
        path = str(raw or "").strip().replace("\\", "/")
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        if not path or lower.startswith(("docs/", "receipts/", ".hawking/")):
            continue
        if "/tests/" in lower or "/test/" in lower:
            continue
        if name.startswith("test_") or name.endswith(("_test.py", "_tests.py")):
            continue
        paths.add(path)
    return paths

_FRESH_ROUTE_INSTRUCTION = (
    "Use the explicitly qualified alternate model route for this fresh attempt; "
    "do not replay the prior provider turn."
)

# A qualified alternate route is a single fresh recovery attempt, not a
# recursive retry family.  The former unlimited chain created thousands of
# queue/graph projections with the same writer scope, made provider prompts
# grow, and consumed paid requests without increasing independent progress.
# This is a retry-safety bound only; it is never a worker-count ceiling.
MAX_ROUTE_REOPEN_DEPTH = 1
_ROUTE_REOPEN_TOKEN = "_ROUTE_REOPEN_V2"
_COMPACT_ROUTE_REOPEN_RE = re.compile(r"_ROUTE_REOPEN_V2_G(\d+)$")
_AUTO_IMPLEMENTATION_VERSION_RE = re.compile(r"_AUTO_IMPLEMENTATION_V(\d+)")


def _auto_regression_test_path(
    source_tranche: str,
    focus_tests: Sequence[str],
) -> str:
    """Give an Auto mutation a dedicated proving-test path.

    Existing focused tests are valuable regression context, but are normally
    green before a new source change.  The engine rightly refuses those as a
    proof of causality.  Auto therefore reserves one narrow, initially absent
    test file in the same test directory.  A provider must propose the source
    and that test in one transaction; pre-mutation collection of the absent
    test is non-green, while the post-mutation test proves the new behavior.
    """
    candidates = [str(item).split("::", 1)[0].strip() for item in focus_tests]
    first = next((item for item in candidates if item), "")
    parent = Path(first).parent if first else Path("hawking/tests")
    slug = re.sub(r"[^a-z0-9]+", "_", str(source_tranche).lower()).strip("_")
    slug = slug[:96] or "continuation"
    return str(parent / f"test_auto_{slug}_regression.py")


def _route_reopen_depth(tranche_id: Any) -> int:
    """Count durable route-reopen generations, including compact ids.

    Existing queue rows used repeated ``_ROUTE_REOPEN_V2`` tokens.  Keep those
    ids readable and backwards-compatible while using ``_G<N>`` for newly
    generated rows, which prevents id/context growth from becoming the next
    source of provider failures.
    """
    text = str(tranche_id or "")
    compact = _COMPACT_ROUTE_REOPEN_RE.search(text)
    if compact:
        return int(compact.group(1))
    return text.count(_ROUTE_REOPEN_TOKEN)


def _route_reopen_compact_id(tranche_id: Any, depth: int) -> str:
    """Return a bounded-size id for a route-reopen generation."""
    root = str(tranche_id or "").split(_ROUTE_REOPEN_TOKEN, 1)[0].strip()
    return f"{root}{_ROUTE_REOPEN_TOKEN}_G{int(depth)}"


def _with_fresh_route_instruction(objective: Any) -> str:
    """Keep route-reopen prompts bounded and idempotent.

    Reopen generations are durable queue rows.  Re-appending the same route
    sentence to an already reopened objective inflated provider context on
    every recovery pass and made structured mutation turns less reliable.
    Strip only trailing copies of this exact supervisor instruction, then add
    one canonical copy.  The rest of the provider objective remains intact.
    """
    value = str(objective or "").rstrip()
    suffix = re.escape(_FRESH_ROUTE_INSTRUCTION)
    value = re.sub(rf"(?:\s+{suffix})+$", "", value).rstrip()
    return f"{value} {_FRESH_ROUTE_INSTRUCTION}".strip()


def _route_reopen_scope(item: Mapping[str, Any]) -> str:
    """Return the writer/effect scope that owns an alternate route attempt."""
    plan = item.get("mutation_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    for value in (
        item.get("writer_scope"), item.get("effect_scope"),
        plan.get("writer_scope"), plan.get("effect_scope"),
        item.get("source_tranche"),
    ):
        scope = str(value or "").strip()
        if scope:
            return scope
    return str(item.get("tranche_id") or item.get("tranche") or "").split(
        _ROUTE_REOPEN_TOKEN, 1
    )[0].strip()


def _auto_implementation_generation(item: Mapping[str, Any]) -> int:
    """Order competing historical implementation contracts newest-first."""
    match = _AUTO_IMPLEMENTATION_VERSION_RE.search(
        str(item.get("tranche_id") or item.get("tranche") or "")
    )
    return int(match.group(1)) if match else 0


def _route_reopen_priority(item: Mapping[str, Any]) -> int:
    try:
        return int(item.get("priority", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _compact_stale_route_reopen_projection(
    source: Mapping[str, Any],
    queue: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str], Dict[str, Any]]:
    """Discard only obsolete route-reopen *projections* from an Auto queue.

    WorkUnit records and their result/dead-letter packets remain the durable
    authority.  A queue row is merely the scheduler projection, so retaining
    every terminal/retry row forever makes ordinary refill scan thousands of
    duplicate writer scopes.  Keep live and acceptance-pending rows intact;
    remove only stale generated reopen rows that cannot be admitted under the
    one-fresh-route policy.  The accompanying graph projection is reduced by
    the exact same ids to keep its runnable frontier truthful and cheap.
    """
    live_states = {
        "ADMITTED", "RUNNING", "CHECKPOINTING", "COMPACTING",
        "COMPLETE", "COMPLETE_PENDING_ACCEPTANCE",
    }
    # A route recovery belongs to its source tranche, not to every historical
    # implementation-version spelling of that tranche.  Earlier V2 spools
    # retained V1..V4 terminal projections, then legitimately made one
    # recovery for each spelling.  Those recoveries are duplicate effect
    # attempts, rather than independent parallel work.  Once a source has a
    # recovery projection, its old terminal base projections can disappear
    # from the scheduler view.  The WorkUnit/dead-letter ledger is untouched.
    recovered_sources = {
        str(item.get("source_tranche") or "").strip()
        for item in queue
        if str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
        and bool(str(item.get("reopen_of") or "").strip())
    }
    live_recovery_sources = {
        str(item.get("source_tranche") or "").strip()
        for item in queue
        if str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
        and bool(str(item.get("reopen_of") or "").strip())
        and str(item.get("state") or "").strip().upper() in live_states
    }
    # Preserve one not-yet-admitted recovery when no recovery for its source is
    # live.  Without this, compaction could delete the very qualified route
    # it was meant to retain before the admission loop saw it.
    ready_recovery_candidates: Dict[str, List[Mapping[str, Any]]] = {}
    for item in queue:
        source_tranche = str(item.get("source_tranche") or "").strip()
        if (
            source_tranche
            and str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and bool(str(item.get("reopen_of") or "").strip())
            and str(item.get("state") or "").strip().upper() in {"READY", "RUNNABLE", "PENDING"}
            and source_tranche not in live_recovery_sources
            and _route_reopen_depth(item.get("tranche_id") or item.get("tranche")) <= MAX_ROUTE_REOPEN_DEPTH
        ):
            ready_recovery_candidates.setdefault(source_tranche, []).append(item)
    protected_ready_recoveries: set[str] = set()
    for candidates in ready_recovery_candidates.values():
        selected = max(
            candidates,
            key=lambda candidate: (
                _auto_implementation_generation(candidate),
                _route_reopen_priority(candidate),
                str(candidate.get("tranche_id") or candidate.get("tranche") or ""),
            ),
        )
        tranche_id = str(selected.get("tranche_id") or selected.get("tranche") or "").strip()
        if tranche_id:
            protected_ready_recoveries.add(tranche_id)
    retained: List[Dict[str, Any]] = []
    removed_ids: List[str] = []
    removed_by_state: Dict[str, int] = {}
    for raw_item in queue:
        item = dict(raw_item)
        tranche_id = str(item.get("tranche_id") or item.get("tranche") or "").strip()
        state = str(item.get("state") or "READY").strip().upper()
        stale_reopen = (
            str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and bool(str(item.get("reopen_of") or "").strip())
            and state not in live_states
            and tranche_id not in protected_ready_recoveries
        )
        source_tranche = str(item.get("source_tranche") or "").strip()
        stale_base_after_recovery = (
            str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and not bool(str(item.get("reopen_of") or "").strip())
            and state == "TERMINAL_PROVIDER_OUTCOME"
            and source_tranche in recovered_sources
        )
        if not stale_reopen and not stale_base_after_recovery:
            retained.append(item)
            continue
        if tranche_id:
            removed_ids.append(tranche_id)
        removed_by_state[state] = removed_by_state.get(state, 0) + 1

    removed = set(removed_ids)
    graph_source = source.get("graph")
    graph = dict(graph_source) if isinstance(graph_source, Mapping) else {"nodes": []}
    graph["nodes"] = [
        dict(node)
        for node in (graph.get("nodes") or [])
        if isinstance(node, Mapping)
        and str(node.get("id") or "").strip() not in removed
    ]
    graph_state = {
        str(key): dict(value)
        for key, value in (source.get("graph_frontier_state") or {}).items()
        if str(key) not in removed and isinstance(value, Mapping)
    }
    graph_runnable = [
        str(item).strip()
        for item in (source.get("graph_runnable_frontier") or [])
        if str(item).strip() and str(item).strip() not in removed
    ]
    return retained, graph, removed_ids, {
        "graph_frontier_state": graph_state,
        "graph_runnable_frontier": graph_runnable,
        "removed_by_state": removed_by_state,
    }


def _compact_generated_tranche_projection(
    generated_ids: set[str],
    queue: Sequence[Mapping[str, Any]],
) -> tuple[set[str], List[str]]:
    """Keep the generated-id index aligned with its live queue projection.

    ``generated_tranches`` is an acceleration index for the Auto queue, not
    the WorkUnit ledger.  It must retain the canonical contract and the one
    permitted fresh-route recovery even when their queue rows have already
    been compacted; otherwise the materializer would resurrect that work.
    The old recursive loop appended every later retry id, including
    ever-growing ids that no longer had a queue row.  Keep current
    Auto-generated contracts and bounded (depth <= 1) history only. Terminal
    WorkUnit records, receipts, and dead letters remain untouched.
    """
    projected = {
        str(item.get("tranche_id") or item.get("tranche") or "").strip()
        for item in queue
        if str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
        and str(item.get("tranche_id") or item.get("tranche") or "").strip()
    }
    bounded_history = {
        tranche_id
        for tranche_id in generated_ids
        if _route_reopen_depth(tranche_id) <= MAX_ROUTE_REOPEN_DEPTH
    }
    retained = projected | bounded_history
    removed = sorted(generated_ids - retained)
    return retained, removed


def _materialize_auto_continuations(
    goal: Mapping[str, Any],
    *,
    max_new: Optional[int] = None,
    workspace: str | Path | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Turn explicit read-evidence heads into the next Auto work frontier.

    The ordinary refill controller is intentionally admission-only.  That is
    the right boundary for legacy Goals, but it left an explicitly opted-in
    long-haul Goal idle after its source-mapping queue was accepted.  This
    small deterministic scaffold is Auto planning, not Codex planning: it
    consumes the Goal's durable accepted heads and creates bounded
    implementation contracts with disjoint tranche scopes.  It never retries
    terminal provider outcomes, invents hardware availability, or promotes a
    provider packet.

    A Goal must opt in with ``auto_continuation_policy.enabled`` and name the
    allowed source tranches.  The generated contracts remain ordinary queue
    entries, so the existing dependency, writer-scope, budget, and WorkUnit
    admission gates still own execution.
    """
    source = dict(goal)
    policy = source.get("auto_continuation_policy")
    if not isinstance(policy, Mapping) or policy.get("enabled") is not True:
        return source, {"schema": AUTO_CONTINUATION_SCHEMA, "generated": [], "skipped": [{"reason": "not_enabled"}]}

    allowed = {
        str(item).strip()
        for item in (policy.get("source_tranches") or _AUTO_CONTINUATION_TASK_CLASSES)
        if str(item).strip()
    }
    queue = [dict(item) for item in (source.get("auto_refill_queue") or []) if isinstance(item, Mapping)]
    # V2 can widen from the fixed initial phase subset without asking Codex to
    # invent a plan: accepted read-evidence packets already carry their exact
    # source/test focus.  Admit only those packets that name a production
    # focus path; unscoped evidence remains read evidence until Auto supplies
    # a bounded contract through its normal planning route.
    if policy.get("expand_accepted_focused_heads") is True:
        allowed.update(
            str(item.get("tranche_id") or item.get("tranche") or "").strip()
            for item in queue
            if str(item.get("state") or "").upper() == "ACCEPTED_READ_EVIDENCE"
            and _production_focus_files(item)
        )
    # Compact legacy reopen rows in place before they are admitted again.
    # `_with_fresh_route_instruction` already protects newly-created rows,
    # but queue records from before that repair remain durable and can keep
    # spawning oversized WorkUnit contracts indefinitely. This is a
    # semantics-preserving supervisor normalization: it retains one fresh
    # route instruction and never changes scope, acceptance, or state.
    compacted_objectives: List[Dict[str, Any]] = []
    for item in queue:
        objective = str(item.get("objective") or "")
        if _FRESH_ROUTE_INSTRUCTION not in objective:
            continue
        compacted = _with_fresh_route_instruction(objective)
        if compacted == objective:
            continue
        item["objective"] = compacted
        compacted_objectives.append({
            "tranche_id": str(item.get("tranche_id") or item.get("tranche") or ""),
            "old_chars": len(objective),
            "new_chars": len(compacted),
        })
    queue, graph, stale_reopen_ids, projection_compaction = (
        _compact_stale_route_reopen_projection(source, queue)
    )
    historical_generated_ids = {
        str(item).strip()
        for item in (policy.get("generated_tranches") or [])
        if str(item).strip()
    }
    generated_ids, stale_generated_ids = _compact_generated_tranche_projection(
        historical_generated_ids, queue,
    )
    existing_ids = {
        str(item.get("tranche_id") or item.get("tranche") or "").strip()
        for item in queue
        if str(item.get("tranche_id") or item.get("tranche") or "").strip()
    }
    accepted = {
        str(item.get("tranche") or "").strip()
        for item in (source.get("accepted_heads") or [])
        if isinstance(item, Mapping) and str(item.get("tranche") or "").strip()
    }
    terminal = {
        str(item.get("tranche") or item.get("tranche_id") or "").strip()
        for key in ("blocked_tranches", "frontier_outcomes")
        for item in (source.get(key) or [])
        if isinstance(item, Mapping)
        and str(item.get("tranche") or item.get("tranche_id") or "").strip()
        and str(item.get("state") or item.get("classification") or "").upper()
        in {"TERMINAL_PROVIDER_OUTCOME", "OUTPUT_UNUSABLE", "CANCELLED", "BLOCKED"}
    }
    # The queue is a compatibility projection and can be the first durable
    # place that learns a child reached a provider terminal state. Include it
    # here so the same refill tick can materialize a fresh qualified route;
    # otherwise a reconciled ADMITTED row would remain terminal forever while
    # the graph falsely reports runnable work.
    terminal.update(
        str(item.get("tranche") or item.get("tranche_id") or "").strip()
        for item in queue
        if isinstance(item, Mapping)
        and str(item.get("tranche") or item.get("tranche_id") or "").strip()
        and str(item.get("state") or item.get("classification") or "").upper()
        in {"TERMINAL_PROVIDER_OUTCOME", "OUTPUT_UNUSABLE", "CANCELLED", "BLOCKED"}
    )
    generated: List[Dict[str, Any]] = []
    rerouted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    raw_limit = max_new if max_new is not None else policy.get("max_generated_per_refill", 9)
    if raw_limit is None or str(raw_limit).strip().lower() in {"unbounded", "all", "auto"}:
        # This bounds only this deterministic materialization pass by the
        # actual eligible graph, never by an arbitrary worker count.
        limit = max(0, len(allowed))
    else:
        try:
            limit = max(0, int(raw_limit))
        except (TypeError, ValueError):
            limit = 9

    nodes = [dict(node) for node in (graph.get("nodes") or []) if isinstance(node, Mapping)]
    node_ids = {str(node.get("id") or "").strip() for node in nodes if str(node.get("id") or "").strip()}

    for item in queue:
        base = str(item.get("tranche_id") or item.get("tranche") or "").strip()
        if not base or base not in allowed:
            continue
        if base not in accepted:
            skipped.append({"tranche_id": base, "reason": "not_accepted_read_evidence"})
            continue
        if base in terminal:
            skipped.append({"tranche_id": base, "reason": "terminal_provider_outcome"})
            continue
        # A new explicit generation is the safe spool/recovery seam after a
        # protocol repair.  It creates a fresh bounded contract while leaving
        # every prior WorkUnit and dead letter immutable; the default remains
        # V1 for legacy Goals that do not opt into another generation.
        generation = str(policy.get("continuation_version") or "V1").strip().upper()
        if not re.fullmatch(r"V[0-9]+", generation):
            generation = "V1"
        continuation = f"{base}_AUTO_IMPLEMENTATION_{generation}"
        if continuation in generated_ids or continuation in existing_ids:
            continue
        if limit <= 0:
            break
        source_files = [str(value) for value in (item.get("focus_files") or []) if str(value).strip()]
        existing_tests = [str(value) for value in (item.get("focus_tests") or []) if str(value).strip()]
        # A dedicated absent test is the red-before-green proving surface;
        # existing tests remain source-mapping context, not the causal proof.
        focused_tests = [_auto_regression_test_path(base, existing_tests)]
        existing_regression_test = bool(
            workspace is not None
            and (Path(workspace).expanduser().resolve() / focused_tests[0]).is_file()
        )
        task_class = _AUTO_CONTINUATION_TASK_CLASSES.get(base, "routine_implementation")
        writer_scope = f"auto-continuation/{base.lower()}"
        acceptance_canary = bool(policy.get("acceptance_canary_generation") == generation)
        canary_model = (
            str(policy.get("acceptance_canary_model") or "").strip()
            if acceptance_canary else ""
        )
        physical_note = (
            " Hardware and device execution remain withheld unless the existing protected gate admits them."
            if task_class == "hardware_reasoning" else ""
        )
        generated_item = {
            "tranche_id": continuation,
            "priority": int(item.get("priority", 0) or 0) + 25,
            "state": "READY",
            "objective": (
                f"Advance {base} beyond accepted read evidence with one bounded, substantive implementation "
                f"change in the declared scope. Inspect the current owner and existing receipt, implement "
                f"the narrowest missing executable behavior, run a focused test that exercises that behavior, "
                f"and record exact evidence. Do not satisfy this tranche with a phase/contract/qualification "
                f"marker, a version constant, an import-only assertion, or a test-only change. Do not repeat "
                f"source mapping, replay terminal provider outcomes, or claim phase/release completion."
                f"{physical_note}"
            ),
            "acceptance": [
                "A production-source delta changes an observable behavior in the declared scope, or a precise current blocker is recorded.",
                "A focused test/check exercises that changed behavior with reproducible output; marker-only, import-only, and test-only evidence is insufficient.",
                "The result names changed files, evidence paths, and an explicit no-release/no-hardware-qualification boundary.",
            ],
            "focus_files": source_files,
            "focus_tests": focused_tests,
            "writer_scope": writer_scope,
            "task_class": task_class,
            "goal_mode": "discrete_goal",
            "mutation_plan": {
                "writer_scope": writer_scope,
                "effect_scope": writer_scope,
                "source_tranche": base,
                "allow_existing_regression_tests": existing_regression_test,
                "dedicated_regression_test": focused_tests[0],
                "acceptance_canary": acceptance_canary,
                "acceptance_canary_generation": generation if acceptance_canary else "",
                "require_substantive_behavior": True,
                "non_goals": [
                    "do not replay terminal provider outcomes",
                    "do not add phase, contract, qualification, or version markers as the implementation",
                    "do not use test-only or import-only changes as implementation evidence",
                    "do not weaken hardware quiescence or protected-resource gates",
                    "do not promote provider prose or claim release completion",
                ],
            },
            "generated_by": AUTO_CONTINUATION_SCHEMA,
            "source_tranche": base,
            "dependencies": [base],
            "depends_on": [base],
        }
        if canary_model in REMOTE_AUTO_ROSTER:
            generated_item["model_override"] = canary_model
        queue.append(generated_item)
        generated_ids.add(continuation)
        existing_ids.add(continuation)
        generated.append({"tranche_id": continuation, "source_tranche": base, "writer_scope": writer_scope})
        if continuation not in node_ids:
            nodes.append({"id": continuation, "dependencies": [base], "generated_by": AUTO_CONTINUATION_SCHEMA})
            node_ids.add(continuation)
        limit -= 1

    # A mutation-capable lane can produce a terminal provider outcome before
    # it reaches repo.edit.  If the Goal explicitly names a freshly qualified
    # alternate route, create one new route-scoped contract.  This is not a
    # retry of the provider turn: the tranche id, model route, and admission
    # evidence are all new, while the old dead letter remains authoritative.
    reopen = policy.get("route_reopen")
    if isinstance(reopen, Mapping) and reopen.get("enabled") is True:
        reopen_classes = {
            str(value).strip().upper()
            for value in (reopen.get("failure_classes") or ["MUTATION_PAYLOAD_MISSING"])
            if str(value).strip()
        }
        configured_model = str(reopen.get("model") or "").strip()
        route_selection: Dict[str, Any] = {}
        fresh_route_receipt = str(reopen.get("fresh_route_receipt") or "").strip()
        route_receipts = reopen.get("fresh_route_receipts")
        route_receipts = route_receipts if isinstance(route_receipts, Mapping) else {}
        # READY reopen rows may have been materialized before the measured
        # failure ledger learned that their original route was semantically
        # unusable. Re-evaluate those rows at refill time so a durable spool
        # cannot pin the active frontier to one provider indefinitely.
        model_pool = reopen.get("model_pool") or CORE_AUTO_MODELS
        candidates = [
            str(model).strip() for model in model_pool
            if str(model).strip() in REMOTE_AUTO_ROSTER
        ]
        if candidates:
            for ready in queue:
                if (
                    str(ready.get("state") or "").upper() != "READY"
                    or str(ready.get("generated_by") or "") != AUTO_CONTINUATION_SCHEMA
                    or not str(ready.get("reopen_of") or "").strip()
                ):
                    continue
                task_class = str(
                    ready.get("task_class") or "routine_implementation"
                ).strip()
                preferred = MODEL_PRIORS.get(
                    _routing_task_class(task_class), MODEL_PRIORS["general"]
                )
                selected_model, selection = select_measured_model(
                    task_class,
                    candidates,
                    workspace=workspace,
                    preferred=preferred,
                    role="implementation",
                )
                current_model = str(ready.get("model_override") or "").strip()
                if (
                    selected_model
                    and selected_model != current_model
                    and selection.get("reason") == "semantic_failure_diversification"
                ):
                    ready["model_override"] = selected_model
                    ready["route_selection"] = dict(selection)
                    rerouted.append({
                        "tranche_id": ready.get("tranche_id"),
                        "reopen_of": ready.get("reopen_of"),
                        "from_model": current_model,
                        "model_override": selected_model,
                        "route_selection": dict(selection),
                    })
        # A scope may have historical V1/V2/... implementation contracts.
        # Their provider failures are not independent work. Prefer one newest
        # eligible contract, and never create a second recovery while a route
        # for that same writer scope is already live or queued.
        reopen_scopes = {
            _route_reopen_scope(item)
            for item in queue
            if str(item.get("reopen_of") or "").strip()
        }
        terminal_candidates = [
            item for item in queue
            if str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and str(item.get("state") or "").upper() == "TERMINAL_PROVIDER_OUTCOME"
        ]
        for item in sorted(
            terminal_candidates,
            key=lambda candidate: (
                -_auto_implementation_generation(candidate),
                -_route_reopen_priority(candidate),
                str(candidate.get("tranche_id") or candidate.get("tranche") or ""),
            ),
        ):
            # A fresh qualified route is one recovery attempt.  A reopened
            # packet that reaches a terminal state is useful failure evidence,
            # not a license to retry the same writer scope recursively.
            if str(item.get("reopen_of") or "").strip():
                skipped.append({
                    "tranche_id": item.get("tranche_id"),
                    "reason": "route_reopen_chain_exhausted",
                })
                continue
            classification = str(
                item.get("released_classification") or item.get("classification") or ""
            ).strip().upper()
            if classification not in reopen_classes:
                continue
            writer_scope = _route_reopen_scope(item)
            if writer_scope and writer_scope in reopen_scopes:
                skipped.append({
                    "tranche_id": item.get("tranche_id"),
                    "reason": "route_reopen_scope_already_recovered",
                    "writer_scope": writer_scope,
                })
                continue
            prior_id = str(item.get("workunit_id") or "").strip()
            alternate_model = configured_model
            if configured_model.casefold() in {"auto", "hawking-auto", "measured", "measured-auto"}:
                pool = reopen.get("model_pool") or CORE_AUTO_MODELS
                candidates = [
                    str(model).strip() for model in pool
                    if str(model).strip() in REMOTE_AUTO_ROSTER
                ]
                task_class = str(item.get("task_class") or "routine_implementation").strip()
                preferred = MODEL_PRIORS.get(_routing_task_class(task_class), MODEL_PRIORS["general"])
                alternate_model, route_selection = select_measured_model(
                    task_class,
                    candidates,
                    workspace=workspace,
                    preferred=preferred,
                    role="implementation",
                )
            selected_route_receipt = str(
                route_receipts.get(alternate_model) or fresh_route_receipt
            ).strip()
            if not prior_id or not alternate_model or not selected_route_receipt:
                skipped.append({"tranche_id": item.get("tranche_id"), "reason": "reopen_contract_incomplete"})
                continue
            source_id = str(item.get("tranche_id") or "").strip()
            next_depth = 1
            # Preserve the existing public shape for the sole fresh route
            # recovery.  Further generations are intentionally prohibited.
            v2_id = f"{source_id}_ROUTE_REOPEN_V2"
            if not v2_id or v2_id in existing_ids:
                continue
            reopened = dict(item)
            reopened.update({
                "tranche_id": v2_id,
                "state": "READY",
                "workunit_id": None,
                "released_at": None,
                "released_workunit_state": None,
                "released_classification": None,
                "result_path": None,
                "model_override": alternate_model,
                "reopen_of": prior_id,
                "route_reopen_generation": next_depth,
                "route_reopen_root": source_id.split(_ROUTE_REOPEN_TOKEN, 1)[0].strip(),
                "fresh_route_receipt": selected_route_receipt,
                "priority": int(item.get("priority", 0) or 0) + 5,
                "objective": _with_fresh_route_instruction(item.get("objective")),
            })
            queue.append(reopened)
            if writer_scope:
                reopen_scopes.add(writer_scope)
            existing_ids.add(v2_id)
            generated_ids.add(v2_id)
            generated_item = {
                "tranche_id": v2_id,
                "source_tranche": item.get("source_tranche"),
                "reopen_of": prior_id,
                "model_override": alternate_model,
            }
            if route_selection:
                reopened["route_selection"] = dict(route_selection)
                generated_item["route_selection"] = dict(route_selection)
            generated.append(generated_item)
            if v2_id not in node_ids:
                source_tranche = str(item.get("source_tranche") or "").strip()
                nodes.append({"id": v2_id, "dependencies": [source_tranche] if source_tranche else [], "generated_by": AUTO_CONTINUATION_SCHEMA})
                node_ids.add(v2_id)

    updated = dict(source)
    if generated or rerouted or compacted_objectives or stale_reopen_ids or stale_generated_ids:
        updated["auto_refill_queue"] = queue
        updated["graph"] = {**graph, "nodes": nodes}
        updated.update({
            "graph_frontier_state": projection_compaction["graph_frontier_state"],
            "graph_runnable_frontier": projection_compaction["graph_runnable_frontier"],
        })
        updated["auto_continuation_policy"] = {
            **dict(policy),
            "schema": AUTO_CONTINUATION_SCHEMA,
            "generated_tranches": sorted(generated_ids),
            "last_materialized_at": time.time(),
        }
    return updated, {
        "schema": AUTO_CONTINUATION_SCHEMA,
        "generated": generated,
        "rerouted": rerouted,
        "compacted_objectives": compacted_objectives[-32:],
        "stale_route_reopen_projection": {
            "removed_count": len(stale_reopen_ids),
            "removed_by_state": projection_compaction["removed_by_state"],
            "tranche_ids": stale_reopen_ids[-32:],
            "claim_boundary": "Only redundant queue/graph projections were removed; WorkUnit records and result/dead-letter packets remain authoritative.",
        },
        "stale_generated_tranche_projection": {
            "removed_count": len(stale_generated_ids),
            "tranche_ids": stale_generated_ids[-32:],
            "claim_boundary": "Only stale Auto generated-id index entries were removed; queue contracts, WorkUnit records, and result/dead-letter packets remain authoritative.",
        },
        "skipped": skipped[-32:],
        "claim_boundary": "Auto-generated bounded continuation contracts only; no provider result or phase/release acceptance.",
    }


def _reconcile_auto_queue_workunits(
    root: Path, goal: Mapping[str, Any]
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Release queue entries whose admitted WorkUnit already reached a terminal state.

    Queue state is a projection, not the WorkUnit authority.  A daemon crash or
    a provider dead-letter can therefore leave an ``ADMITTED`` queue row after
    the canonical WorkUnit has finished.  Without this repair Auto sees neither
    a free lane nor a fresh terminal edge to reopen, so an otherwise runnable
    graph silently stalls.  This helper only copies terminal owner facts; it
    never promotes provider prose or invents a retry contract.
    """
    updated = dict(goal)
    queue = [dict(item) for item in (goal.get("auto_refill_queue") or []) if isinstance(item, Mapping)]
    reconciled: List[Dict[str, Any]] = []
    terminal_states = {
        "OUTPUT_UNUSABLE", "PAUSED_PROVIDER", "CANCELLED", "BLOCKED",
        "COMPLETE", "COMPLETE_PENDING_ACCEPTANCE",
    }
    for item in queue:
        state = str(item.get("state") or "READY").upper()
        if state not in {"ADMITTED", "RUNNING"}:
            continue
        workunit_id = str(item.get("workunit_id") or "").strip()
        if not workunit_id or "/" in workunit_id or "\\" in workunit_id:
            continue
        path = root / ".hawking" / "workunits" / f"{workunit_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        workunit_state = str(record.get("state") or "").upper()
        if workunit_state not in terminal_states:
            continue
        failure = workunit_state not in {"COMPLETE", "COMPLETE_PENDING_ACCEPTANCE"}
        projected_state = "TERMINAL_PROVIDER_OUTCOME" if failure else "COMPLETE"
        item.update({
            "state": projected_state,
            "released_at": time.time(),
            "released_workunit_state": workunit_state,
            "released_classification": str(record.get("classification") or "").strip(),
            "result_path": f"receipts/future/workunits/{workunit_id}_RESULT_PACKET.json",
        })
        reconciled.append({
            "tranche_id": str(item.get("tranche_id") or item.get("tranche") or "").strip(),
            "workunit_id": workunit_id,
            "projected_state": projected_state,
            "classification": str(record.get("classification") or "").strip(),
        })
    if reconciled:
        updated["auto_refill_queue"] = queue
    return updated, reconciled


def _shared_acceptance_circuit(
    root: Path,
    goal_id: str,
    *,
    threshold: int = 3,
) -> Dict[str, Any]:
    """Detect a shared mutation-gate failure before Auto fans it out.

    Provider turns that reach a typed proposal but fail before ``repo.edit``
    are evidence about Hawking's shared acceptance seam, not independent
    provider failures.  Keep one explicitly marked canary available until a
    later accepted WorkUnit closes the circuit.  Read-only WorkUnits are not
    examined or constrained here.
    """
    events: List[tuple[float, str, str]] = []
    workunits = root / ".hawking" / "workunits"
    if not workunits.is_dir():
        return {"open": False, "failures": [], "accepted_after_failure": False}
    applied_mutation_workunits: set[str] = set()
    intents = root / ".hawking" / "mutation-intents"
    if intents.is_dir():
        for path in intents.glob("*.json"):
            try:
                intent = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            request = intent.get("request") if isinstance(intent, Mapping) else {}
            if (
                isinstance(request, Mapping)
                and str(request.get("goal_id") or "") == str(goal_id)
                and str(intent.get("status") or "").upper() in {"APPLIED", "VERIFIED"}
            ):
                workunit_id = str(request.get("workunit_id") or "").strip()
                if workunit_id:
                    applied_mutation_workunits.add(workunit_id)
    for path in workunits.glob("WORKUNIT-*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(record.get("goal_id") or "") != str(goal_id):
            continue
        at = path.stat().st_mtime
        state = str(record.get("state") or "").upper()
        classification = str(record.get("classification") or "").upper()
        if state == "OUTPUT_UNUSABLE" and classification == "MUTATION_PROPOSAL_UNACCEPTED":
            events.append((at, "rejected", str(record.get("workunit_id") or path.stem)))
        elif (
            state in {"COMPLETE", "COMPLETE_PENDING_ACCEPTANCE"}
            and str(record.get("workunit_id") or path.stem) in applied_mutation_workunits
        ):
            events.append((at, "accepted", str(record.get("workunit_id") or path.stem)))
    events.sort(reverse=True)
    newest_accept = max((at for at, kind, _ in events if kind == "accepted"), default=0.0)
    failures = [(at, wid) for at, kind, wid in events if kind == "rejected" and at > newest_accept]
    return {
        "open": len(failures) >= max(1, int(threshold)),
        "failures": [wid for _at, wid in failures[:16]],
        "accepted_after_failure": bool(newest_accept and (not failures or newest_accept > failures[0][0])),
        "threshold": threshold,
    }


def _refill_runnable_frontier_locked(
    workspace: str | Path,
    goal_id: str,
    *,
    efficient_concurrency: Optional[int] = MAX_REMOTE_COGNITION_WORKERS,
    max_new: Optional[int] = None,
    background: bool = True,
) -> Dict[str, Any]:
    """Admit already-planned, dependency-valid lanes until the frontier is empty.

    This is the Auto refill seam, not a planner.  A durable Goal supplies
    ``auto_refill_queue`` entries containing the tranche objective, acceptance
    contract, and writer scope.  The refill controller only filters that
    queue against the canonical graph/frontier, avoids active writer-scope
    conflicts, and calls the existing Goal/WorkUnit admission owner.  It
    never invents objectives, retries a known terminal tranche, or promotes
    provider output.
    """
    root = Path(workspace).expanduser().resolve()
    from .goal_surface import (
        dispatch_child_workunit,
        load_goal,
        reconcile_goal_frontier,
        update_goal,
    )

    goal = load_goal(root, str(goal_id).strip())
    if not goal:
        return {
            "schema": AUTO_REFILL_SCHEMA,
            "goal_id": str(goal_id),
            "admitted": [],
            "skipped": [{"reason": "goal_missing"}],
            "claim_boundary": "no admission",
        }
    if str(goal.get("status") or "").upper() != "RUNNING":
        return {
            "schema": AUTO_REFILL_SCHEMA,
            "goal_id": str(goal.get("goal_id") or goal_id),
            "admitted": [],
            "skipped": [{"reason": "goal_not_running", "status": goal.get("status")}],
            "claim_boundary": "no admission",
        }

    # Recover queue truth before materialization.  This is deliberately a
    # cheap local projection: the WorkUnit record is authoritative, while the
    # queue must expose its terminal edge so route reopen and free-lane refill
    # can happen on this same Auto tick.
    goal, queue_reconciled = _reconcile_auto_queue_workunits(root, goal)
    if queue_reconciled:
        persisted = update_goal(
            root,
            str(goal.get("goal_id") or goal_id),
            auto_refill_queue=goal.get("auto_refill_queue"),
            auto_refill_last_run={
                "at": time.time(),
                "queue_reconciled": queue_reconciled[-32:],
            },
        )
        if persisted:
            goal = persisted

    # A V2 Goal may explicitly opt into Auto continuation scaffolding.  Run
    # this before graph projection so generated implementation nodes become
    # dependency-valid on the same refill tick.  Legacy Goals remain strictly
    # admission-only because they do not carry the opt-in policy.
    goal, continuation = _materialize_auto_continuations(goal, workspace=root)
    continuation_changed = bool(
        continuation.get("generated")
        or continuation.get("rerouted")
        or continuation.get("compacted_objectives")
        or (continuation.get("stale_route_reopen_projection") or {}).get("removed_count")
        or (continuation.get("stale_generated_tranche_projection") or {}).get("removed_count")
    )
    if continuation_changed:
        persisted = update_goal(
            root,
            str(goal.get("goal_id") or goal_id),
            graph=goal.get("graph"),
            graph_frontier_state=goal.get("graph_frontier_state"),
            graph_runnable_frontier=goal.get("graph_runnable_frontier"),
            auto_refill_queue=goal.get("auto_refill_queue"),
            auto_continuation_policy=goal.get("auto_continuation_policy"),
            auto_continuation_last_run=continuation,
        )
        if persisted:
            goal = persisted

    # The WorkUnit projection below knows whether children are still active,
    # but a parent acceptance is a tranche-level fact.  Rebuild that small
    # graph projection before examining the queue so an accepted dependency
    # releases every independently runnable sibling on the same pass.  This
    # deliberately does *not* construct a queue contract: planning remains
    # Hawking Auto's job and admission still requires an explicit objective,
    # acceptance contract, and effect scope.
    goal, graph_projection_changed = _project_tranche_graph(goal)
    if graph_projection_changed:
        persisted = update_goal(root, str(goal.get("goal_id") or goal_id), **{
            "graph_frontier_state": goal["graph_frontier_state"],
            "graph_runnable_frontier": goal["graph_runnable_frontier"],
        })
        if persisted:
            goal = persisted

    projection = reconcile_goal_frontier(root, str(goal.get("goal_id") or goal_id), persist=True)
    active_ids = [
        str(item).strip() for item in (projection.get("active_workunit_ids") or [])
        if str(item).strip()
    ]
    if efficient_concurrency is None:
        ceiling: Optional[int] = None
        slots: Optional[int] = None
    else:
        try:
            ceiling = max(1, int(efficient_concurrency))
        except (TypeError, ValueError):
            ceiling = None
        slots = None if ceiling is None else max(0, ceiling - len(active_ids))
    if max_new is not None:
        try:
            requested_slots = max(0, int(max_new))
            slots = requested_slots if slots is None else min(slots, requested_slots)
        except (TypeError, ValueError):
            slots = 0

    graph_bound = "graph_runnable_frontier" in goal
    graph_frontier = {
        str(item).strip() for item in (goal.get("graph_runnable_frontier") or [])
        if str(item).strip()
    }
    graph_nodes = goal.get("graph")
    graph_node_ids = {
        str(node.get("id") or "").strip()
        for node in (graph_nodes.get("nodes") or [])
        if isinstance(node, Mapping) and str(node.get("id") or "").strip()
    } if isinstance(graph_nodes, Mapping) else set()
    queue = [dict(item) for item in (goal.get("auto_refill_queue") or []) if isinstance(item, Mapping)]
    # Count already-admitted route assignments so a qualified alternate route
    # cannot be starved by repeated measured-value ties.  This is advisory
    # diversity telemetry; writer-scope admission remains the hard boundary.
    active_model_counts = _authoritative_active_model_counts(
        root,
        str(goal.get("goal_id") or goal_id),
        queue,
    )
    terminal_tranches = {
        str(item.get("tranche") or item.get("tranche_id") or "").strip()
        for key in ("blocked_tranches", "frontier_outcomes")
        for item in (goal.get(key) or [])
        if isinstance(item, Mapping)
        and str(item.get("tranche") or item.get("tranche_id") or "").strip()
    }
    admitted_tranches = {
        str(item.get("tranche_id") or "").strip()
        for item in queue
        if str(item.get("state") or "").upper() in {"ADMITTED", "RUNNING", "COMPLETE"}
    }

    def _active_scope(workunit_id: str) -> str:
        path = root / ".hawking" / "workunits" / f"{workunit_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            record = {}
        for value in (record.get("writer_scope"), record.get("effect_scope")):
            if str(value or "").strip():
                return str(value).strip()
        contract_path = str(record.get("contract_path") or "").strip()
        if contract_path:
            try:
                contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                contract = {}
            mutation_plan = contract.get("mutation_plan")
            if isinstance(mutation_plan, Mapping) and str(mutation_plan.get("writer_scope") or "").strip():
                return str(mutation_plan["writer_scope"]).strip()
        return ""

    def _active_focus_files(workunit_id: str) -> set[str]:
        """Read the immutable child contract's production mutation focus."""
        path = root / ".hawking" / "workunits" / f"{workunit_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return set()
        contract_path = str(record.get("contract_path") or "").strip()
        if not contract_path:
            return set()
        try:
            contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return set()
        return _production_focus_files(contract) if isinstance(contract, Mapping) else set()

    active_scopes = {scope for scope in (_active_scope(item) for item in active_ids) if scope}
    active_focus_files = set().union(
        *(_active_focus_files(item) for item in active_ids)
    ) if active_ids else set()
    # The queue is a projection and can briefly lag a WorkUnit that was
    # admitted by a concurrent supervisor tick.  Scan the small authoritative
    # WorkUnit state directory as a second fence so an orphaned live owner
    # cannot permit a second writer for the same effect scope.  This does not
    # cancel or rewrite the orphan; it only makes the next admission fail
    # closed until the live owner drains.
    workunits_root = root / ".hawking" / "workunits"
    if workunits_root.is_dir():
        for state_path in workunits_root.glob("WORKUNIT-*.json"):
            try:
                record = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, Mapping):
                continue
            if str(record.get("goal_id") or "").strip() != str(goal.get("goal_id") or goal_id):
                continue
            if str(record.get("state") or "").upper() not in {"RUNNING", "ADMITTED"}:
                continue
            scope = _active_scope(str(record.get("workunit_id") or state_path.stem))
            if scope:
                active_scopes.add(scope)
            active_focus_files.update(
                _active_focus_files(str(record.get("workunit_id") or state_path.stem))
            )
    admitted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    changed = False

    def _has_fresh_reopen_evidence(item: Mapping[str, Any]) -> bool:
        """Allow a terminal tranche only with an explicit fresh route proof.

        A terminal provider result is never retried by merely re-adding its
        tranche id.  Reopening is a separate durable contract: it names the
        prior WorkUnit and points at a receipt produced by a newly qualified
        route.  Keep the receipt inside the workspace so a caller cannot turn
        an arbitrary external path into a retry authorization.
        """
        prior = str(item.get("reopen_of") or "").strip()
        receipt = str(item.get("fresh_route_receipt") or "").strip()
        if not prior or not receipt:
            return False
        try:
            receipt_path = (root / receipt).resolve()
            receipt_path.relative_to(root)
            route_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not receipt_path.is_file() or not isinstance(route_receipt, Mapping):
            return False
        def _int_field(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0
        if route_receipt.get("tool_call_qualified") is True:
            return True
        qualification = route_receipt.get("qualification")
        if isinstance(qualification, Mapping):
            checks = qualification.get("checks")
            tool_call = checks.get("tool_call") if isinstance(checks, Mapping) else None
            if isinstance(tool_call, Mapping):
                if (
                    _int_field(tool_call.get("http_status")) == 200
                    and _int_field(tool_call.get("tool_call_count")) > 0
                    and bool(tool_call.get("arguments_present"))
                ):
                    return True
        probes = route_receipt.get("live_gateway_probes")
        if isinstance(probes, list):
            return any(
                isinstance(probe, Mapping)
                and probe.get("accepted") is True
                and _int_field(probe.get("gateway_status")) == 200
                and bool(probe.get("tool_name") or probe.get("generated_text"))
                for probe in probes
            )
        return False

    candidates: List[tuple[int, int, Dict[str, Any]]] = []
    for index, item in enumerate(queue):
        tranche = str(item.get("tranche_id") or item.get("tranche") or "").strip()
        state = str(item.get("state") or "READY").upper()
        if not tranche:
            skipped.append({"index": index, "reason": "missing_tranche_id"})
            continue
        if state not in {"READY", "RUNNABLE", "PENDING"}:
            continue
        mutation_plan = item.get("mutation_plan")
        source_tranche = str(item.get("source_tranche") or "").strip()
        if not source_tranche and isinstance(mutation_plan, Mapping):
            source_tranche = str(mutation_plan.get("source_tranche") or "").strip()
        # A route-reopen continuation is a new admission contract, not a new
        # graph node.  Its lineage is still graph-bound through the original
        # source tranche, while the explicit fresh-route receipt authorizes
        # reopening a terminal provider edge.  Without this exception the
        # graph projection could contain only the previous reopen depth and
        # strand the next bounded continuation in READY forever.
        lineage_runnable = (
            str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and bool(str(item.get("reopen_of") or "").strip())
            and source_tranche in graph_node_ids
            and _has_fresh_reopen_evidence(item)
        )
        if graph_bound and tranche not in graph_frontier and not lineage_runnable:
            skipped.append({"tranche_id": tranche, "reason": "not_graph_runnable"})
            continue
        if tranche in terminal_tranches and not _has_fresh_reopen_evidence(item):
            skipped.append({"tranche_id": tranche, "reason": "known_terminal_or_admitted"})
            continue
        if tranche in admitted_tranches:
            skipped.append({"tranche_id": tranche, "reason": "known_terminal_or_admitted"})
            continue
        if not str(item.get("objective") or "").strip():
            skipped.append({"tranche_id": tranche, "reason": "missing_objective"})
            continue
        acceptance = item.get("acceptance")
        if not isinstance(acceptance, list) or not any(str(value).strip() for value in acceptance):
            skipped.append({"tranche_id": tranche, "reason": "missing_acceptance"})
            continue
        try:
            priority = int(item.get("priority", 0))
        except (TypeError, ValueError):
            priority = 0
        candidates.append((-priority, index, item))

    acceptance_circuit = _shared_acceptance_circuit(
        root, str(goal.get("goal_id") or goal_id),
    )
    if acceptance_circuit["open"]:
        active_canary_generation = ""
        continuation_policy = goal.get("auto_continuation_policy")
        if isinstance(continuation_policy, Mapping):
            active_canary_generation = str(
                continuation_policy.get("acceptance_canary_generation") or ""
            ).strip().upper()
        canaries = [
            candidate for candidate in candidates
            if (
                bool((candidate[2].get("mutation_plan") or {}).get("acceptance_canary"))
                and str((candidate[2].get("mutation_plan") or {}).get(
                    "acceptance_canary_generation"
                ) or "").strip().upper() == active_canary_generation
            )
        ]
        # The mutation circuit only constrains effectful lanes. Read-only and
        # unrelated science work may continue while the shared acceptance seam
        # is unhealthy.
        readonly = [
            candidate for candidate in candidates
            if not bool((candidate[2].get("mutation_plan") or {}).get("acceptance_canary"))
            and not bool((candidate[2].get("mutation_plan") or {}).get("require_source_mutation"))
        ]
        candidates = sorted(readonly) + sorted(canaries)[:1]
        if not candidates:
            skipped.append({
                "reason": "shared_acceptance_circuit_open",
                "failures": acceptance_circuit["failures"],
            })

    for _negative_priority, index, item in sorted(candidates):
        if slots is not None and slots <= 0:
            break
        tranche = str(item.get("tranche_id") or item.get("tranche") or "").strip()
        if (
            _route_reopen_depth(tranche) > MAX_ROUTE_REOPEN_DEPTH
            and str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
        ):
            skipped.append({"tranche_id": tranche, "reason": "route_reopen_depth_exhausted"})
            continue
        # Re-evaluate a READY route-reopen contract at the final admission
        # seam.  This protects the live frontier when an older continuation
        # packet was materialized before semantic-failure telemetry existed.
        # The queue remains the authority for the durable contract; only the
        # admitted provider override is diversified here.
        route_reopen_policy = goal.get("auto_continuation_policy", {}).get("route_reopen") \
            if isinstance(goal.get("auto_continuation_policy"), Mapping) else None
        if (
            isinstance(route_reopen_policy, Mapping)
            and route_reopen_policy.get("enabled") is True
            and str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and str(item.get("reopen_of") or "").strip()
        ):
            pool = route_reopen_policy.get("model_pool") or CORE_AUTO_MODELS
            route_candidates = [
                str(model).strip() for model in pool
                if str(model).strip() in REMOTE_AUTO_ROSTER
            ]
            task_class = str(item.get("task_class") or "routine_implementation").strip()
            preferred = MODEL_PRIORS.get(_routing_task_class(task_class), MODEL_PRIORS["general"])
            selected_model, selection = select_measured_model(
                task_class,
                route_candidates,
                workspace=root,
                preferred=preferred,
                role="implementation",
            )
            selected_model, diversity_selection = select_active_diversity_route(
                task_class,
                route_candidates,
                selected_model,
                active_model_counts=active_model_counts,
                workspace=root,
            )
            if diversity_selection.get("reason") == "active_lane_diversification":
                selection = {**dict(selection), **diversity_selection}
            if (
                selection.get("reason") in {
                    "semantic_failure_diversification",
                    "active_lane_diversification",
                }
                and selected_model
                and selected_model != str(item.get("model_override") or "").strip()
            ):
                item = dict(item)
                item["model_override"] = selected_model
                item["route_selection"] = dict(selection)
                queue[index] = item
        plan = item.get("mutation_plan") if isinstance(item.get("mutation_plan"), Mapping) else {}
        # Durable queue rows may predate the existing-test propagation repair.
        # Reconcile this execution fact at the final admission seam so a fresh
        # provider cannot be instructed to create a regression file that is
        # already present.  This changes no scope or acceptance obligation;
        # it only selects replace/retain semantics from current disk truth.
        focused_test_paths = [
            str(value).strip()
            for value in (item.get("focus_tests") or [])
            if str(value).strip()
        ]
        if focused_test_paths and any(
            (root / path).is_file() for path in focused_test_paths
        ):
            plan = {**dict(plan), "allow_existing_regression_tests": True}
            item = dict(item)
            item["mutation_plan"] = dict(plan)
            queue[index] = item
        if (
            isinstance(route_reopen_policy, Mapping)
            and route_reopen_policy.get("enabled") is True
            and str(item.get("generated_by") or "") == AUTO_CONTINUATION_SCHEMA
            and str(item.get("reopen_of") or "").strip()
        ):
            plan = {**dict(plan), "allow_existing_regression_tests": True}
            item = dict(item)
            item["mutation_plan"] = dict(plan)
            queue[index] = item
        writer_scope = str(item.get("writer_scope") or plan.get("writer_scope") or tranche).strip()
        if writer_scope in active_scopes:
            skipped.append({"tranche_id": tranche, "reason": "writer_scope_conflict", "writer_scope": writer_scope})
            continue
        candidate_focus_files = _production_focus_files(item)
        mutation_bundle = item.get("mutation_bundle") or plan.get("mutation_bundle") or plan
        requires_source = bool(
            isinstance(mutation_bundle, Mapping)
            and mutation_bundle.get("require_source_mutation") is True
        )
        existing_production_focus = {
            path for path in candidate_focus_files
            if not path.startswith("tests/")
            and not "/tests/" in path
            and (root / path).is_file()
        }
        if requires_source and not existing_production_focus:
            skipped.append({
                "tranche_id": tranche,
                "reason": "invalid_contract_missing_production_scope",
                "focus_files": sorted(candidate_focus_files),
            })
            continue
        conflicting_files = sorted(candidate_focus_files & active_focus_files)
        if conflicting_files:
            skipped.append({
                "tranche_id": tranche,
                "reason": "focus_file_conflict",
                "paths": conflicting_files,
            })
            continue
        try:
            budget = float(item.get("budget_usd", goal.get("budget_authorized_usd") or 0.5))
        except (TypeError, ValueError):
            skipped.append({"tranche_id": tranche, "reason": "invalid_budget"})
            continue
        try:
            result = dispatch_child_workunit(
                root,
                str(goal.get("goal_id") or goal_id),
                objective=str(item["objective"]),
                acceptance=[str(value) for value in item["acceptance"] if str(value).strip()],
                focus_files=[str(value) for value in (item.get("focus_files") or []) if str(value).strip()],
                focus_tests=[str(value) for value in (item.get("focus_tests") or []) if str(value).strip()],
                budget_usd=budget,
                required_capabilities=(
                    [str(value) for value in item.get("required_capabilities") if str(value).strip()]
                    if isinstance(item.get("required_capabilities"), list) else None
                ),
                external_roots=(
                    [str(value) for value in item.get("external_roots") if str(value).strip()]
                    if isinstance(item.get("external_roots"), list) else None
                ),
                mutation_plan={**dict(plan), "writer_scope": writer_scope},
                goal_mode=str(item.get("goal_mode") or "worker_research"),
                task_class=str(item.get("task_class") or "").strip() or None,
                model_override=str(item.get("model_override") or "").strip() or None,
                background=background,
            )
        except Exception as exc:  # admission failure is evidence, not a refill loop
            skipped.append({"tranche_id": tranche, "reason": "admission_error", "error": f"{type(exc).__name__}: {exc}"[:400]})
            continue
        workunit_id = str(result.get("workunit_id") or "").strip()
        if not workunit_id:
            skipped.append({"tranche_id": tranche, "reason": "admission_missing_workunit_id"})
            continue
        updated_item = dict(item)
        updated_item.update({
            "state": "ADMITTED",
            "workunit_id": workunit_id,
            "writer_scope": writer_scope,
            "admitted_at": time.time(),
            "selected_model": str(
                ((result.get("auto_route") or {}).get("selected_model"))
                if isinstance(result.get("auto_route"), Mapping) else ""
            ),
        })
        queue[index] = updated_item
        admitted.append({"tranche_id": tranche, "workunit_id": workunit_id, "writer_scope": writer_scope})
        active_ids.append(workunit_id)
        active_scopes.add(writer_scope)
        active_focus_files.update(candidate_focus_files)
        admitted_model = str(
            updated_item.get("model_override")
            or updated_item.get("selected_model")
            or ""
        ).strip()
        if admitted_model:
            active_model_counts[admitted_model] = active_model_counts.get(admitted_model, 0) + 1
        admitted_tranches.add(tranche)
        if slots is not None:
            slots -= 1
        changed = True

    if changed or queue != (goal.get("auto_refill_queue") or []):
        update_goal(
            root,
            str(goal.get("goal_id") or goal_id),
            auto_refill_queue=queue,
            auto_refill_last_run={
                "at": time.time(),
                "active_before": len(projection.get("active_workunit_ids") or []),
                "ceiling": ceiling,
                "admitted": list(admitted),
                "skipped": list(skipped)[-32:],
            },
        )
    return {
        "schema": AUTO_REFILL_SCHEMA,
        "goal_id": str(goal.get("goal_id") or goal_id),
        "active_before": len(projection.get("active_workunit_ids") or []),
        "ceiling": ceiling,
        "admitted": admitted,
        "queue_reconciled": queue_reconciled[-32:],
        "acceptance_circuit": acceptance_circuit,
        "skipped": skipped[-32:],
        "claim_boundary": "Auto admitted only pre-specified runnable WorkUnit contracts; no tranche planning or provider acceptance.",
    }


def _project_tranche_graph(goal: Mapping[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Project durable tranche acceptance into the V2 dependency graph.

    This is intentionally a projection, not planning: a graph node becomes
    runnable when its dependencies have independent acceptance evidence, but
    it cannot be admitted until an already-durable ``auto_refill_queue``
    contract exists.  Keeping those concerns separate makes recovery cheap:
    after a daemon restart or an accepted review, Auto sees the same runnable
    frontier without rereading the master objective or replaying workers.
    """
    # ``graph`` is the durable V2 field; accept the descriptive alias as a
    # compatibility input for freshly created/older Goal records.
    graph = goal.get("graph") or goal.get("dependency_graph")
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    if not isinstance(nodes, list):
        return dict(goal), False

    accepted = {
        str(item.get("tranche") or "").strip()
        for item in (goal.get("accepted_heads") or [])
        if isinstance(item, Mapping) and str(item.get("tranche") or "").strip()
    }
    current = {
        str(key): dict(value)
        for key, value in (goal.get("graph_frontier_state") or {}).items()
        if isinstance(value, Mapping)
    }
    for tranche, row in current.items():
        if str(row.get("state") or "").upper() == "ACCEPTED_READ_EVIDENCE":
            accepted.add(tranche)

    projected: Dict[str, Dict[str, Any]] = {}
    runnable: List[str] = []
    terminal_states = {
        "TERMINAL_PROVIDER_OUTCOME",
        "COMPLETE_PENDING_ACCEPTANCE",
        "WORKUNIT_COMPLETE_PENDING_ACCEPTANCE",
    }
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        tranche = str(node.get("id") or "").strip()
        if not tranche:
            continue
        dependencies = [str(value).strip() for value in (node.get("dependencies") or []) if str(value).strip()]
        row = dict(current.get(tranche) or {})
        row.setdefault("dependencies", dependencies)
        if tranche in accepted:
            # Preserve review metadata and never reinterpret acceptance as a
            # phase/release completion.
            row.setdefault("state", "ACCEPTED_READ_EVIDENCE")
            row["unresolved_dependencies"] = []
        else:
            unresolved = [dependency for dependency in dependencies if dependency not in accepted]
            row["unresolved_dependencies"] = unresolved
            if str(row.get("state") or "").upper() in terminal_states:
                # A terminal provider/output edge requires a fresh explicit
                # reopen contract; dependency projection must never turn it
                # back into ordinary runnable work merely because it has no
                # unresolved graph dependencies.
                pass
            elif unresolved:
                if str(row.get("state") or "").upper() not in {
                    "WORKUNIT_COMPLETE_PENDING_ACCEPTANCE", "TERMINAL_PROVIDER_OUTCOME",
                }:
                    row["state"] = "WAITING_DEPENDENCIES"
            else:
                row["state"] = "RUNNABLE_AWAITING_CONTRACT"
                runnable.append(tranche)
        projected[tranche] = row

    changed = (
        projected != (goal.get("graph_frontier_state") or {})
        or runnable != (goal.get("graph_runnable_frontier") or [])
    )
    if not changed:
        return dict(goal), False
    updated = dict(goal)
    updated["graph_frontier_state"] = projected
    updated["graph_runnable_frontier"] = runnable
    return updated, True


__all__ = [
    "AUTO_PLAN_SCHEMA", "AUTO_TASK_CLASSES", "MODEL_PRIORS",
    "AUTO_CONTINUATION_SCHEMA",
    "BASE_MAX_REMOTE_COGNITION_WORKERS", "BASE_MAX_PREMIUM_WORKERS",
    "FRONTIER_BURST_FACTOR",
    "MAX_REMOTE_COGNITION_WORKERS", "MAX_PREMIUM_WORKERS", "MAX_AUTOTUNED_WORKERS", "WorkerLease",
    "WorkerAdmissionError", "RemoteWorkerAdmission",
    "classify_request", "build_cognition_plan", "worker_packet",
    "packet_path", "persist_worker_packet", "record_model_outcome", "recommend_concurrency",
    "model_routing_value", "select_measured_model",
    "refill_runnable_frontier",
]
def _p4_auto_implementation_contract_v2_scope_marker():
    """Bounded P4_AUTO_IMPLEMENTATION_CONTRACT_V2 scope hook."""
    return "P4_AUTO_IMPLEMENTATION_CONTRACT_V2"
def _p4_auto_implementation_contract_probe(state):
    """Bounded P4 hook: report whether a discrete goal is complete.

    Returns True only when the supplied state mapping explicitly marks the
    goal as complete; any other value (including missing keys) is False.
    """
    if not isinstance(state, dict):
        return False
    return state.get("discrete_goal_complete") is True
