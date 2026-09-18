"""Hawking-owned Auto routing for the OpenAI/Open WebUI surface.

Auto is a Hawking selector, not an OpenRouter model and not a second agent
runtime.  This module only classifies the request and chooses an already
admitted live model from the remote roster.  The gateway remains responsible
for credentials, cost policy, cancellation, receipts, and the actual call.

The selector is deliberately small and explainable.  It is a first routing
policy that can later consume measured local-role qualification without
changing the OpenAI surface or Open WebUI integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


AUTO_MODEL_ID = "hawking-auto"
AUTO_POLICY_REVISION = "hawking-auto-aggressive-v2-measured-diversity-unbounded-14"

# These are exact remote identities observed in the current catalog. The live
# gateway allow/deny policy still filters this roster at request time; adding
# an identity here is pool admission, not a claim that every route is healthy
# or that every model should receive routine traffic.
REMOTE_AUTO_ROSTER = (
    "deepseek/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
    "qwen/qwen3.8-2.4t-a95b",
    "nvidia/nemotron-3-ultra-550b-a55b",
)

CORE_AUTO_MODELS = (
    "deepseek/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
)
ESCALATION_AUTO_MODELS = (
    "qwen/qwen3.8-2.4t-a95b",
    "nvidia/nemotron-3-ultra-550b-a55b",
)
MODEL_ROLE_BIASES = {
    "deepseek/deepseek-v4.1-flash": (
        "implementation", "source_mapping", "routine_implementation", "bug_repair",
        "test_design", "routine_read_evidence",
    ),
    "moonshotai/kimi-k3": (
        "architecture", "architecture_planning", "deep_synthesis", "hard_debugging",
        "high_value_review", "long_context_synthesis",
    ),
    "z-ai/glm-5.3": (
        "planning", "coding", "agentic_source_work", "security_research", "alternative_synthesis",
    ),
    "qwen/qwen3.8-2.4t-a95b": (
        "disagreement_resolution", "deep_planning_backup", "long_context_synthesis",
    ),
    "nvidia/nemotron-3-ultra-550b-a55b": (
        "architectural_diversity", "independent_review", "falsification", "hard_disagreement",
    ),
}


class AutoRouteUnavailable(RuntimeError):
    """No live, policy-admitted remote candidate can serve Auto."""


@dataclass(frozen=True)
class AutoRoute:
    """A sanitized routing decision suitable for response metadata/receipts."""

    model_id: str
    task_class: str
    reason: str
    candidates: tuple[str, ...]
    requested_profile: Optional[str] = None
    cognition_plan: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "requested_model": AUTO_MODEL_ID,
            "selected_model": self.model_id,
            "runtime": "remote",
            "provider": "openrouter",
            "task_class": self.task_class,
            "reason": self.reason,
            "candidates": list(self.candidates),
            "requested_profile": self.requested_profile,
            "policy": AUTO_POLICY_REVISION,
        }
        if isinstance(self.cognition_plan, Mapping):
            payload["cognition_plan"] = dict(self.cognition_plan)
        return payload


def is_auto_model(value: object) -> bool:
    """Accept the canonical id and harmless UI aliases, never provider ids."""
    normalized = str(value or "").strip().casefold()
    return normalized in {AUTO_MODEL_ID, "hawking/auto", "hawking:auto", "auto"}


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""


def request_text(request: Optional[Mapping[str, Any]]) -> str:
    """Extract objective text without forwarding Hawking control metadata."""
    if not isinstance(request, Mapping):
        return ""
    direct = []
    for key in ("hawking_objective", "objective", "purpose"):
        value = request.get(key)
        if isinstance(value, str):
            direct.append(value)
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if str(message.get("role") or "").casefold() in {"user", "system"}:
                direct.append(_content_text(message.get("content")))
    return " ".join(item for item in direct if item).casefold()


def _profile(request: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(request, Mapping):
        return None
    for key in ("hawking_auto_profile", "hawking_route", "auto_profile"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold().replace("_", "-")
    return None


def _available(
    *,
    allowed_model_ids: Iterable[str] = (),
    available_model_ids: Optional[Iterable[str]] = None,
) -> list[str]:
    allowed = {str(item).strip() for item in allowed_model_ids if str(item).strip()}
    catalog = (
        {str(item).strip() for item in available_model_ids if str(item).strip()}
        if available_model_ids is not None else None
    )
    return [
        model for model in REMOTE_AUTO_ROSTER
        if (not allowed or model in allowed)
        and (catalog is None or model in catalog)
    ]


def _escalation_requested(request: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(request, Mapping):
        return False
    value = request.get("hawking_auto_escalation") or request.get("hawking_information_gain")
    return bool(value is True or str(value).strip().casefold() in {"1", "true", "yes", "on", "high"})


def choose_remote_route(
    request: Optional[Mapping[str, Any]],
    *,
    allowed_model_ids: Iterable[str] = (),
    available_model_ids: Optional[Iterable[str]] = None,
) -> AutoRoute:
    """Choose a remote model using explicit task signals and live availability."""
    candidates = _available(
        allowed_model_ids=allowed_model_ids,
        available_model_ids=available_model_ids,
    )
    if not candidates:
        raise AutoRouteUnavailable(
            "Hawking Auto has no live roster model inside the current gateway policy"
        )

    profile = _profile(request)
    objective = request_text(request)
    has_tools = isinstance(request, Mapping) and bool(request.get("tools"))
    choices: list[tuple[str, str, str]]
    escalation = _escalation_requested(request) or profile in {
        "qwen", "nemotron", "escalation", "diversity",
    }
    if profile in {"qwen", "nemotron", "escalation", "diversity"}:
        preferred = (
            "qwen/qwen3.8-2.4t-a95b"
            if profile in {"qwen", "escalation"}
            else "nvidia/nemotron-3-ultra-550b-a55b"
        )
        choices = [("escalation", preferred, "explicit information-gain/escalation profile")]
    elif profile in {"glm", "crossover", "alternative"}:
        choices = [("crossover", "z-ai/glm-5.3", "explicit GLM crossover trial")]
    elif profile in {"fast", "tools", "build", "execute", "worker"} or has_tools or any(
        token in objective for token in (
            "git", "repo", "tool", "write", "test", "build", "execute",
            "goal", "workunit", "mutation", "implementation",
        )
    ):
        choices = [("execution", "deepseek/deepseek-v4.1-flash", "tool/build signals")]
    elif profile in {"premium", "kimi", "independent", "quality", "deep", "architecture", "judge"} or "kimi" in objective:
        choices = [("independent", "moonshotai/kimi-k3", "independent/deep reasoning profile")]
    elif profile in {"multimodal", "vision"} or any(
        token in objective for token in ("image", "multimodal", "visual")
    ):
        choices = [("multimodal", "moonshotai/kimi-k3", "multimodal signal")]
    elif any(
        token in objective for token in (
            "verify", "verification", "audit", "evaluate", "review", "falsify",
            "compare", "grade", "adversarial",
        )
    ):
        choices = [("verification", "moonshotai/kimi-k3", "review/independent signals")]
    else:
        choices = [("general", "deepseek/deepseek-v4.1-flash", "default quality/value route")]

    task_class, preferred, reason = choices[0]
    # Qwen/Nemotron stay out of ordinary traffic even when live. They enter
    # only through an explicit information-gain profile or a future measured
    # arbitrage decision from Auto's richer planner.
    if not escalation:
        candidates = [model for model in candidates if model in CORE_AUTO_MODELS] or candidates
    selected = preferred if preferred in candidates else candidates[0]
    if selected != preferred:
        reason = f"{reason}; preferred route unavailable, selected first live candidate"
    plan: Optional[Mapping[str, Any]] = None
    try:
        # Lazy import avoids the intentional auto_mode ↔ auto_orchestration
        # dependency cycle: the orchestration module owns the richer plan, but
        # the lightweight route remains usable by isolated catalog tests.
        from .auto_orchestration import build_cognition_plan

        plan = build_cognition_plan(
            request,
            allowed_models=candidates,
            goal_id=str(request.get("goal_id") or "") if isinstance(request, Mapping) else "",
        )
    except Exception:
        # Routing must not fail merely because optional plan persistence or a
        # future plan schema is unavailable. The selected model remains the
        # authoritative admission decision.
        plan = None
    return AutoRoute(
        model_id=selected,
        task_class=task_class,
        reason=reason,
        candidates=tuple(candidates),
        requested_profile=profile,
        cognition_plan=plan,
    )


__all__ = [
    "AUTO_MODEL_ID",
    "AUTO_POLICY_REVISION",
    "CORE_AUTO_MODELS",
    "ESCALATION_AUTO_MODELS",
    "MODEL_ROLE_BIASES",
    "REMOTE_AUTO_ROSTER",
    "AutoRoute",
    "AutoRouteUnavailable",
    "choose_remote_route",
    "is_auto_model",
    "request_text",
]
