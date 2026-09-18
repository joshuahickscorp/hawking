"""Provider-neutral cognition providers for Hawking WorkUnits.

This module is deliberately an adapter, not a second mission system.  A
WorkUnit still belongs to :mod:`hawking.mission`, admission still belongs to the
existing scheduler, and acceptance still belongs to the deterministic
verifier.  The adapters here only turn a bounded provider request into the
same ``ModelProvider``/``GenerationResponse`` contract used by local
residents.

The two supported provider selectors are explicit and credential-free::

    openrouter:vendor/model-id
    hawking:native-profile-or-resident

OpenRouter model ids are intentionally opaque.  No model catalog is required
to use one.  The ``hawking`` selector is the local provider-neutral native
cognition path; it does not create a second scheduler or acceptance authority.

Remote identity is semantic.  It never invents a local PID, port, process, or
resident model path.  A remote receipt records provider/model/request/slot,
usage, cost, retries, latency, and the provider request id without retaining
credentials or hidden reasoning.
"""
from __future__ import annotations

import json
import getpass
import os
import re
import socket
import subprocess
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .providers import (
    FEATURES,
    Capability,
    CapabilityContract,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
    ResidentProfile,
)


REMOTE_RECEIPT_SCHEMA = "hawking.remote.cognition.receipt.v1"
REMOTE_IDENTITY_SCHEMA = "hawking.remote.cognition.identity.v1"
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_REMOTE_TIMEOUT_S = 60.0
DEFAULT_REMOTE_ATTEMPTS = 3
DEFAULT_REMOTE_BACKOFF_S = 2.0
DEFAULT_REMOTE_MAX_BACKOFF_S = 20.0
DEFAULT_CATALOG_TTL_S = 300.0
DEFAULT_OPENROUTER_KEYCHAIN_SERVICE = "OpenRouter"
# There is no artificial Hawking worker-count ceiling.  A finite value remains
# available as an explicit operator/provider policy, but the default is
# unbounded and real resource, provider-health, budget, dependency, and effect
# scope gates remain authoritative.
DEFAULT_OPENROUTER_CONCURRENCY: Optional[int] = None

# Current OpenRouter endpoint metadata, observed 2026-09-16. These opaque
# identifiers are endpoint `tag` values, not display-name guesses. They are
# preferences only: OpenRouter may fail over within the exact requested model.
DEFAULT_MODEL_PROVIDER_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "deepseek/deepseek-v4.1-flash": ("together",),
    "qwen/qwen3.8-flash": ("alibaba",),
}

# OpenRouter reports that GLM 5.3 requires its reasoning mode.  Omitting the
# optional wire hint lets that route use its provider-required default; sending
# ``reasoning: {effort: none}`` is a deterministic HTTP 400.
MODELS_REQUIRING_PROVIDER_REASONING = frozenset({"z-ai/glm-5.3"})


def _workunit_tool_name_map(names: List[str]) -> Dict[str, str]:
    """Map Hawking tool ids to provider-safe function names.

    Hawking's typed registry uses dotted names (``fs.search``), while the
    OpenAI-compatible function-call boundary used by OpenRouter accepts only
    letters, digits, ``_`` and ``-``. Keep the registry id authoritative and
    translate only at that wire boundary; the reverse mapping is applied before
    any registry dispatch.
    """
    result: Dict[str, str] = {}
    owners: Dict[str, str] = {}
    for raw_name in names:
        name = str(raw_name)
        base = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "hawking_tool"
        base = base[:64]
        candidate = base
        suffix = 1
        while candidate in owners and owners[candidate] != name:
            marker = f"_{suffix}"
            candidate = f"{base[:max(1, 64 - len(marker))]}{marker}"
            suffix += 1
        owners[candidate] = name
        result[name] = candidate
    return result

_HIDDEN_KEY_RE = re.compile(
    r"(?:reasoning|reasoning_content|thinking|thoughts?|chain[_-]?of[_-]?thought|analysis)",
    re.IGNORECASE,
)
_HIDDEN_TAG_RE = re.compile(
    r"<\s*(?:think|analysis|reasoning|thought)\b[^>]*>.*?<\s*/\s*(?:think|analysis|reasoning|thought)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_SECRET_FIELD_RE = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|authorization|password|secret|token)\s*[:=]\s*[\"']?)[^\"'\s,}]+"
)


class RemoteCognitionError(RuntimeError):
    """A structured provider failure safe to place in a WorkUnit result."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        provider: str,
        model_id: str,
        request_id: Optional[str] = None,
        remote_request_id: Optional[str] = None,
        recoverable: bool = False,
        status_code: Optional[int] = None,
        retry_after_s: Optional[float] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.code = str(code)
        self.provider = str(provider)
        self.model_id = str(model_id)
        self.request_id = request_id
        self.remote_request_id = remote_request_id
        self.recoverable = bool(recoverable)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.details = _safe_mapping(details or {})
        super().__init__(self._safe_message(message))

    @staticmethod
    def _safe_message(message: Any) -> str:
        text = str(message or "")
        text = _BEARER_RE.sub(r"\1[REDACTED]", text)
        return _SECRET_FIELD_RE.sub(r"\1[REDACTED]", text)[:1200]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "provider": self.provider,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "remote_request_id": self.remote_request_id,
            "recoverable": self.recoverable,
            "status_code": self.status_code,
            "retry_after_s": self.retry_after_s,
            "details": _safe_mapping(self.details),
        }


class RemoteAuthError(RemoteCognitionError):
    """The provider rejected or could not find the configured credential."""


class RemoteRateLimitError(RemoteCognitionError):
    """The bounded retry budget was exhausted on a rate limit."""


class RemoteCostError(RemoteCognitionError):
    """A paid request was not explicitly authorized or exceeded its budget."""


class RemoteCancelledError(RemoteCognitionError):
    """The caller cancelled a remote request or task."""


@dataclass(frozen=True)
class ProviderSelector:
    """The exact provider/model identity selected by a WorkUnit."""

    provider: str
    model_id: str

    @classmethod
    def parse(cls, value: Any) -> "ProviderSelector":
        text = str(value or "").strip()
        if ":" not in text:
            raise ValueError(
                "remote provider selector must be '<provider>:<model-id>'"
            )
        provider, model = text.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in {"openrouter", "hawking"}:
            raise ValueError(f"unsupported remote provider {provider!r}")
        if not model:
            raise ValueError("remote provider selector requires a model id")
        if any(ord(char) < 32 for char in model):
            raise ValueError("remote model id contains a control character")
        return cls(provider=provider, model_id=model)

    @property
    def value(self) -> str:
        return f"{self.provider}:{self.model_id}"


def is_remote_selector(value: Any) -> bool:
    text = str(value or "").strip()
    return text.lower().startswith(("openrouter:", "hawking:"))


@dataclass(frozen=True)
class RemoteExecutionIdentity:
    """Provider-owned identity; local process topology is explicitly absent."""

    provider: str
    model_id: str
    backend: str
    protocol: str
    transport: Optional[str] = None
    remote_slot: Optional[str] = None
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    remote_request_id: Optional[str] = None
    schema: str = REMOTE_IDENTITY_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "model_id": self.model_id,
            "backend": self.backend,
            "runtime": "remote",
            "protocol": self.protocol,
            "transport": self.transport,
            "remote_slot": self.remote_slot,
            "endpoint": _public_endpoint(self.endpoint),
            "request_id": self.request_id,
            "remote_request_id": self.remote_request_id,
            # These are deliberately explicit.  A remote provider does not
            # own a local child process or loopback listener.
            "pid": None,
            "port": None,
            "model_residency": "provider-managed",
        }


@dataclass
class RemoteCostPolicy:
    """Fail-closed spend policy shared by one provider registration.

    A zero or missing limit means paid remote cognition is blocked.  A caller
    may authorize a WorkUnit explicitly with ``metadata`` containing
    ``cost_authorization: {"max_usd": ...}``; the provider-level limit is
    still the safer default for long-lived AgentOS registrations.
    """

    max_cost_usd: float = 0.0
    mission_limit_usd: Optional[float] = None
    monthly_limit_usd: Optional[float] = None
    spent_usd: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @classmethod
    def from_environment(cls, workspace: Optional[Path] = None) -> "RemoteCostPolicy":
        raw = os.environ.get("HAWKING_REMOTE_MAX_COST_USD")
        if raw is None:
            # Keep the historical variable readable during migration; no
            # credential or spend is inferred from it unless it is numeric.
            raw = os.environ.get("HAWKING_REMOTE_MAX_COST_USD")
        if raw in (None, "") and workspace is not None:
            # Reuse the existing Hawking/HCLI layered configuration owner when
            # a workspace is known.  The environment remains the higher-
            # precedence compatibility path during the rename migration.
            try:
                from .config import Config

                loaded = Config(str(workspace)).load()
                policy = loaded.get("remote_cost_policy")
                if isinstance(policy, Mapping):
                    raw = policy.get("max_cost_usd")
                if raw in (None, ""):
                    raw = loaded.get("remote_max_cost_usd")
            except Exception:
                raw = None
        mission_raw = os.environ.get("HAWKING_REMOTE_MISSION_MAX_COST_USD")
        monthly_raw = os.environ.get("HAWKING_REMOTE_MONTHLY_MAX_COST_USD")
        try:
            limit = max(0.0, float(raw)) if raw not in (None, "") else 0.0
        except (TypeError, ValueError):
            limit = 0.0

        def optional_limit(value: Any, config_key: str) -> Optional[float]:
            candidate = value
            if candidate in (None, "") and workspace is not None:
                try:
                    from .config import Config

                    loaded = Config(str(workspace)).load()
                    candidate = loaded.get(config_key)
                except Exception:
                    candidate = None
            try:
                return max(0.0, float(candidate)) if candidate not in (None, "") else None
            except (TypeError, ValueError):
                return None

        return cls(
            max_cost_usd=limit,
            mission_limit_usd=optional_limit(mission_raw, "remote_mission_max_cost_usd"),
            monthly_limit_usd=optional_limit(monthly_raw, "remote_monthly_max_cost_usd"),
        )

    def _authorized_limit(self, metadata: Optional[Mapping[str, Any]]) -> float:
        limit = float(self.max_cost_usd)
        auth = (metadata or {}).get("cost_authorization")
        if isinstance(auth, Mapping) and auth.get("max_usd") is not None:
            try:
                limit = max(0.0, float(auth["max_usd"]))
            except (TypeError, ValueError):
                limit = 0.0
        return limit

    def authorize(
        self,
        *,
        provider: str,
        model_id: str,
        request_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> float:
        limit = self._authorized_limit(metadata)
        estimated: Optional[float] = None
        raw_estimated = (metadata or {}).get("estimated_cost_usd")
        if raw_estimated is not None:
            try:
                estimated = max(0.0, float(raw_estimated))
            except (TypeError, ValueError):
                estimated = None
        if limit <= 0.0:
            raise RemoteCostError(
                "COST_AUTHORIZATION_REQUIRED",
                "paid remote cognition is blocked until an explicit USD limit is configured",
                provider=provider,
                model_id=model_id,
                request_id=request_id,
                details={"limit_usd": limit, "estimated_cost_usd": estimated},
            )
        with self._lock:
            if self.mission_limit_usd is not None and self.spent_usd >= float(self.mission_limit_usd):
                raise RemoteCostError(
                    "MISSION_COST_LIMIT_EXCEEDED",
                    "mission remote-cost limit has been exhausted",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"mission_limit_usd": self.mission_limit_usd, "spent_usd": self.spent_usd},
                )
            if self.monthly_limit_usd is not None and self.spent_usd >= float(self.monthly_limit_usd):
                raise RemoteCostError(
                    "MONTHLY_COST_LIMIT_EXCEEDED",
                    "monthly remote-cost limit has been exhausted",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"monthly_limit_usd": self.monthly_limit_usd, "spent_usd": self.spent_usd},
                )
            if estimated is not None and estimated > limit:
                raise RemoteCostError(
                    "WORKUNIT_COST_LIMIT_EXCEEDED",
                    "estimated WorkUnit cost exceeds its explicit limit",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"limit_usd": limit, "estimated_cost_usd": estimated},
                )
            if self.mission_limit_usd is not None and estimated is not None and self.spent_usd + estimated > float(self.mission_limit_usd):
                raise RemoteCostError(
                    "MISSION_COST_LIMIT_EXCEEDED",
                    "estimated WorkUnit cost exceeds the mission limit",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"mission_limit_usd": self.mission_limit_usd, "spent_usd": self.spent_usd, "estimated_cost_usd": estimated},
                )
        return limit

    def observe(
        self,
        cost_usd: Optional[float],
        *,
        limit_usd: float,
        provider: str,
        model_id: str,
        request_id: str,
    ) -> Optional[float]:
        if cost_usd is None:
            return None
        try:
            cost = max(0.0, float(cost_usd))
        except (TypeError, ValueError):
            return None
        with self._lock:
            if cost > limit_usd:
                raise RemoteCostError(
                    "OBSERVED_COST_EXCEEDED",
                    "provider-reported cost exceeded the explicit WorkUnit limit",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"limit_usd": limit_usd, "observed_cost_usd": cost},
                )
            next_spent = self.spent_usd + cost
            if self.mission_limit_usd is not None and next_spent > float(self.mission_limit_usd):
                raise RemoteCostError(
                    "MISSION_COST_LIMIT_EXCEEDED",
                    "provider-reported cost exceeded the mission limit",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"mission_limit_usd": self.mission_limit_usd, "spent_usd": self.spent_usd, "observed_cost_usd": cost},
                )
            if self.monthly_limit_usd is not None and next_spent > float(self.monthly_limit_usd):
                raise RemoteCostError(
                    "MONTHLY_COST_LIMIT_EXCEEDED",
                    "provider-reported cost exceeded the monthly limit",
                    provider=provider,
                    model_id=model_id,
                    request_id=request_id,
                    details={"monthly_limit_usd": self.monthly_limit_usd, "spent_usd": self.spent_usd, "observed_cost_usd": cost},
                )
            self.spent_usd = next_spent
        return cost


def _csv_policy_value(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(
        item.strip()
        for item in str(raw).split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class OpenRouterGatewayPolicy:
    """Hawking-owned policy for the generic OpenRouter gateway.

    The policy is intentionally small and configuration-backed.  It keeps
    provider/model admission, privacy routing, request ceilings, retry/timeout
    bounds, and concurrency in Hawking while OpenRouter remains the transport
    and marketplace.  No catalog or credential lookup happens at import time.
    """

    endpoint: str = DEFAULT_OPENROUTER_ENDPOINT
    allowed_models: Tuple[str, ...] = ()
    denied_models: Tuple[str, ...] = ()
    allowed_providers: Tuple[str, ...] = ()
    denied_providers: Tuple[str, ...] = ()
    require_zdr: bool = False
    allow_fallbacks: bool = True
    allow_model_fallbacks: bool = False
    require_parameters: bool = False
    provider_sort: Optional[str] = None
    provider_preferences: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_PROVIDER_PREFERENCES),
        compare=False,
        repr=False,
    )
    max_output_tokens: Optional[int] = None
    timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S
    max_attempts: int = DEFAULT_REMOTE_ATTEMPTS
    concurrency: Optional[int] = DEFAULT_OPENROUTER_CONCURRENCY
    catalog_ttl_s: float = DEFAULT_CATALOG_TTL_S
    cost_policy: RemoteCostPolicy = field(
        default_factory=RemoteCostPolicy,
        compare=False,
        repr=False,
    )

    @classmethod
    def from_environment(
        cls,
        workspace: Optional[Path] = None,
    ) -> "OpenRouterGatewayPolicy":
        values: Dict[str, Any] = {}
        raw_config: Mapping[str, Any] = {}
        if workspace is not None:
            try:
                from .config import Config

                loaded = Config(str(workspace)).load()
                candidate = loaded.get("openrouter_gateway") or loaded.get("remote_gateway")
                if isinstance(candidate, Mapping):
                    raw_config = candidate
            except Exception:
                raw_config = {}

        def value(env_name: str, *config_names: str, default: Any = None) -> Any:
            raw = os.environ.get(env_name)
            if raw not in (None, ""):
                return raw
            for name in config_names:
                if name in raw_config and raw_config[name] not in (None, ""):
                    return raw_config[name]
            return default

        endpoint = str(value(
            "HAWKING_OPENROUTER_ENDPOINT",
            "endpoint",
            default=DEFAULT_OPENROUTER_ENDPOINT,
        )).strip() or DEFAULT_OPENROUTER_ENDPOINT
        allowed_models = _csv_policy_value(value(
            "HAWKING_OPENROUTER_ALLOWED_MODELS",
            "allowed_models",
            default=None,
        ))
        denied_models = _csv_policy_value(value(
            "HAWKING_OPENROUTER_DENIED_MODELS",
            "denied_models",
            default=None,
        ))
        allowed_providers = _csv_policy_value(value(
            "HAWKING_OPENROUTER_ALLOWED_PROVIDERS",
            "allowed_providers",
            default=None,
        ))
        denied_providers = _csv_policy_value(value(
            "HAWKING_OPENROUTER_DENIED_PROVIDERS",
            "denied_providers",
            default=None,
        ))

        def boolean(env_name: str, *config_names: str, default: bool) -> bool:
            raw = value(env_name, *config_names, default=default)
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        def number(env_name: str, *config_names: str, default: Any) -> Any:
            raw = value(env_name, *config_names, default=default)
            try:
                return type(default)(raw)
            except (TypeError, ValueError):
                return default

        max_output_raw = value(
            "HAWKING_OPENROUTER_MAX_OUTPUT_TOKENS",
            "max_output_tokens",
            default=None,
        )
        try:
            max_output_tokens = int(max_output_raw) if max_output_raw not in (None, "") else None
        except (TypeError, ValueError):
            max_output_tokens = None
        provider_sort = str(value(
            "HAWKING_OPENROUTER_SORT",
            "provider_sort",
            default="",
        ) or "").strip().lower() or None
        configured_preferences = raw_config.get("provider_preferences")
        preferences: Dict[str, Tuple[str, ...]] = dict(DEFAULT_MODEL_PROVIDER_PREFERENCES)
        if isinstance(configured_preferences, Mapping):
            for model_id, raw_providers in configured_preferences.items():
                if isinstance(raw_providers, str):
                    providers = _csv_policy_value(raw_providers)
                elif isinstance(raw_providers, (list, tuple)):
                    providers = tuple(str(item).strip() for item in raw_providers if str(item).strip())
                else:
                    continue
                if providers:
                    preferences[str(model_id).strip()] = providers
        raw_concurrency = value(
            "HAWKING_OPENROUTER_CONCURRENCY",
            "concurrency",
            default=DEFAULT_OPENROUTER_CONCURRENCY,
        )
        if isinstance(raw_concurrency, str) and raw_concurrency.strip().lower() in {
            "", "none", "null", "unlimited", "unbounded", "auto",
        }:
            configured_concurrency = None
        elif raw_concurrency in (None, ""):
            configured_concurrency = None
        else:
            try:
                parsed_concurrency = int(raw_concurrency)
            except (TypeError, ValueError):
                parsed_concurrency = 0
            # Zero is the explicit, conventional spelling for unlimited in
            # this local policy surface.  Negative/invalid values fail open to
            # the unbounded default rather than inventing a one-slot barrier.
            configured_concurrency = parsed_concurrency if parsed_concurrency > 0 else None
        return cls(
            endpoint=endpoint,
            allowed_models=allowed_models,
            denied_models=denied_models,
            allowed_providers=allowed_providers,
            denied_providers=denied_providers,
            require_zdr=boolean("HAWKING_OPENROUTER_REQUIRE_ZDR", "require_zdr", default=False),
            allow_fallbacks=boolean("HAWKING_OPENROUTER_ALLOW_FALLBACKS", "allow_fallbacks", default=True),
            allow_model_fallbacks=boolean("HAWKING_OPENROUTER_ALLOW_MODEL_FALLBACKS", "allow_model_fallbacks", default=False),
            require_parameters=boolean("HAWKING_OPENROUTER_REQUIRE_PARAMETERS", "require_parameters", default=False),
            provider_sort=provider_sort,
            provider_preferences=preferences,
            max_output_tokens=max_output_tokens if max_output_tokens and max_output_tokens > 0 else None,
            timeout_s=max(0.1, float(number("HAWKING_OPENROUTER_TIMEOUT_S", "timeout_s", default=DEFAULT_REMOTE_TIMEOUT_S))),
            max_attempts=max(1, int(number("HAWKING_OPENROUTER_MAX_ATTEMPTS", "max_attempts", default=DEFAULT_REMOTE_ATTEMPTS))),
            concurrency=configured_concurrency,
            catalog_ttl_s=max(0.0, float(number("HAWKING_OPENROUTER_CATALOG_TTL_S", "catalog_ttl_s", default=DEFAULT_CATALOG_TTL_S))),
            cost_policy=RemoteCostPolicy.from_environment(workspace),
        )

    def admit_model(self, model_id: str) -> None:
        model = str(model_id or "").strip()
        if not model:
            raise RemoteCognitionError(
                "MODEL_REQUIRED",
                "an exact OpenRouter model id is required",
                provider="openrouter",
                model_id="",
            )
        if model in self.denied_models or (
            self.allowed_models and model not in self.allowed_models
        ):
            raise RemoteCognitionError(
                "MODEL_NOT_ALLOWED",
                f"OpenRouter model {model!r} is outside Hawking's configured model policy",
                provider="openrouter",
                model_id=model,
                recoverable=False,
            )

    def validate_request(self, model_id: str, max_tokens: Optional[int]) -> None:
        self.admit_model(model_id)
        if self.max_output_tokens is not None and max_tokens is not None:
            if int(max_tokens) > self.max_output_tokens:
                raise RemoteCognitionError(
                    "REQUEST_TOKEN_LIMIT_EXCEEDED",
                    f"requested output tokens exceed Hawking's limit of {self.max_output_tokens}",
                    provider="openrouter",
                    model_id=str(model_id),
                    recoverable=False,
                )

    def transport_options(
        self, model_id: str = "", requested: Any = None,
    ) -> Dict[str, Any]:
        """Return the bounded OpenRouter provider-routing object."""
        # Compatibility with callers that supplied only a provider mapping.
        if isinstance(model_id, Mapping) and requested is None:
            requested = model_id
            model_id = ""
        requested_map = requested if isinstance(requested, Mapping) else {}
        provider: Dict[str, Any] = {}
        for key in (
            "order",
            "allow_fallbacks",
            "require_parameters",
            "data_collection",
            "quantizations",
            "sort",
            "ignore",
        ):
            if key in requested_map:
                provider[key] = _safe_value(requested_map[key])
        preferred = tuple(self.provider_preferences.get(str(model_id).strip(), ()))
        if preferred:
            # Same-model endpoint preference. Keep availability fallback unless
            # a stronger explicit policy below requires otherwise.
            provider["order"] = list(preferred)
            provider["allow_fallbacks"] = bool(self.allow_fallbacks)
        if self.allowed_providers:
            provider["order"] = list(self.allowed_providers)
            provider["allow_fallbacks"] = False
        if self.denied_providers:
            if provider.get("order"):
                provider["order"] = [
                    item for item in provider["order"]
                    if str(item) not in self.denied_providers
                ]
            provider["ignore"] = list(self.denied_providers)
            provider["allow_fallbacks"] = False
        if self.require_zdr:
            provider["data_collection"] = "deny"
        if not self.allow_fallbacks:
            provider["allow_fallbacks"] = False
        if self.require_parameters:
            provider["require_parameters"] = True
        if self.provider_sort:
            provider["sort"] = self.provider_sort
        return {"provider": provider} if provider else {}


_CATALOG_CACHE_LOCK = Lock()
_CATALOG_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def cached_openrouter_catalog(
    policy: OpenRouterGatewayPolicy,
    *,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch/cache OpenRouter metadata only when a catalog request asks for it."""
    key = policy.endpoint
    now = time.time()
    with _CATALOG_CACHE_LOCK:
        cached = _CATALOG_CACHE.get(key)
        if not force and cached is not None and now - cached[0] < policy.catalog_ttl_s:
            return _copy(cached[1])
    provider = OpenRouterProvider(
        "__catalog__",
        endpoint=policy.endpoint,
        timeout=policy.timeout_s,
        max_attempts=policy.max_attempts,
        key_resolver=_keychain_resolver_from_environment(),
        catalog_ttl_s=policy.catalog_ttl_s,
    )
    models = provider.list_models(force=force, timeout=policy.timeout_s)
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE[key] = (time.time(), list(models))
    return _copy(models)


def _copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _safe_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        if _HIDDEN_KEY_RE.search(name):
            continue
        if name.lower().replace("-", "_") in {
            "api_key",
            "access_token",
            "authorization",
            "password",
            "secret",
            "token",
        }:
            result[name] = "[REDACTED]"
        elif isinstance(item, Mapping):
            result[name] = _safe_mapping(item)
        elif isinstance(item, list):
            result[name] = [_safe_value(child) for child in item]
        else:
            result[name] = _safe_value(item)
    return result


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _strip_hidden_reasoning(value)
    return _copy(value)


def _strip_hidden_reasoning(text: str) -> str:
    return _HIDDEN_TAG_RE.sub("", str(text)).strip()


def _redact_text(text: Any) -> str:
    value = _strip_hidden_reasoning(str(text or ""))
    value = _BEARER_RE.sub(r"\1[REDACTED]", value)
    return _SECRET_FIELD_RE.sub(r"\1[REDACTED]", value)[:1200]


def _public_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    parsed = urllib.parse.urlparse(str(endpoint))
    if not parsed.scheme or not parsed.netloc:
        return str(endpoint).split("?", 1)[0].split("#", 1)[0]
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_retry_after(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _header(headers: Any, *names: str) -> Optional[str]:
    if headers is None:
        return None
    for name in names:
        try:
            value = headers.get(name)
        except AttributeError:
            value = None
        if value is None:
            try:
                value = headers.get(name.lower())
            except AttributeError:
                value = None
        if value is not None:
            return str(value)
    return None


class _HTTPStatus(Exception):
    def __init__(self, status: int, body: str, headers: Any = None) -> None:
        self.status = int(status)
        self.body = body
        self.headers = headers
        super().__init__(f"HTTP {status}")


def _read_body(response: Any) -> bytes:
    raw = response.read()
    if isinstance(raw, bytes):
        return raw
    return str(raw or "").encode("utf-8")


def _read_json(response: Any) -> Tuple[Any, Any, int]:
    try:
        body = _read_body(response)
        headers = getattr(response, "headers", None)
        status = int(getattr(response, "status", getattr(response, "code", 200)))
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    text = body.decode("utf-8", errors="replace")
    if status < 200 or status >= 300:
        raise _HTTPStatus(status, _redact_text(text), headers)
    try:
        return json.loads(text), headers, status
    except (TypeError, ValueError) as exc:
        raise ValueError("remote provider returned invalid JSON") from exc


def _extract_remote_request_id(data: Any, headers: Any) -> Optional[str]:
    if isinstance(data, Mapping):
        for key in ("id", "request_id", "requestId"):
            if data.get(key) is not None:
                return str(data[key])
    return _header(headers, "x-request-id", "x-openrouter-request-id", "request-id")


def _extract_cost(data: Mapping[str, Any], usage: Mapping[str, Any]) -> Optional[float]:
    for source in (usage, data):
        for key in ("cost", "cost_usd", "total_cost"):
            value = source.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
    return None


def _content_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        # Do not strip edge whitespace here: in an SSE delta the leading
        # space belongs to the next token ("hello" + " world").
        return _HIDDEN_TAG_RE.sub("", value)
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("type") or "text").lower()
            if "reason" in kind or "think" in kind or "analysis" in kind:
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(_HIDDEN_TAG_RE.sub("", text))
        return "".join(parts) or None
    return None


def _tool_call_has_semantics(value: Any) -> bool:
    """Return whether one wire tool call identifies an actual callable."""
    if not isinstance(value, Mapping):
        return False
    function = value.get("function")
    if isinstance(function, Mapping):
        return bool(str(function.get("name") or "").strip())
    # Retain compatibility with the older ``function_call`` shape.
    return bool(str(value.get("name") or "").strip())


def _normalized_response(data: Any) -> Tuple[Optional[str], Optional[str], Dict[str, Any], Dict[str, Any]]:
    safe = _safe_value(data)
    if not isinstance(safe, dict):
        safe = {"value": safe}
    choices = safe.get("choices")
    text: Optional[str] = None
    finish: Optional[str] = None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        finish = str(choice.get("finish_reason")) if choice.get("finish_reason") is not None else None
        message = choice.get("message")
        if isinstance(message, Mapping):
            text = _content_text(message.get("content"))
        if text is None:
            text = _content_text(choice.get("text"))
    if text is None:
        text = _content_text(safe.get("output_text"))
    usage_raw = safe.get("usage")
    usage = dict(usage_raw) if isinstance(usage_raw, Mapping) else {}
    return text, finish, usage, safe


def _semantic_response_problem(
    request: GenerationRequest,
    payload: Mapping[str, Any],
    text: Optional[str],
    finish: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Reject HTTP-success envelopes that cannot be a semantic completion.

    HTTP status is transport evidence only.  In particular, a provider may
    answer ``200 / finish_reason=length`` with a null message and no tool call.
    That must never become a successful Hawking turn, especially when the
    request explicitly required a function or structured JSON result.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return {
            "code": "REMOTE_SEMANTIC_INVALID",
            "message": "provider returned HTTP success without a usable completion choice",
            "reason": "missing_choices",
        }
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        # Chat-completion callers must receive a message object.  Do not let a
        # malformed/legacy text field masquerade as a worker completion.
        return {
            "code": "REMOTE_SEMANTIC_INVALID",
            "message": "provider returned HTTP success without an assistant message",
            "reason": "missing_message",
        }

    raw_calls = message.get("tool_calls")
    tool_calls = (
        [item for item in raw_calls if _tool_call_has_semantics(item)]
        if isinstance(raw_calls, list) else []
    )
    if _tool_call_has_semantics(message.get("function_call")):
        tool_calls.append(message["function_call"])

    tool_choice = request.transport_options.get("tool_choice")
    required_tool = (
        tool_choice == "required"
        or isinstance(tool_choice, Mapping)
        and str(tool_choice.get("type") or "").lower() in {"function", "tool"}
    )
    response_format = request.response_schema
    if response_format is None:
        candidate = request.transport_options.get("response_format")
        response_format = candidate if isinstance(candidate, Mapping) else None
    structured = isinstance(response_format, Mapping) and bool(
        response_format.get("type")
        or response_format.get("json_schema")
        or response_format.get("schema")
    )

    has_content = bool(isinstance(text, str) and text.strip())
    finish_text = str(finish or "").strip().lower()
    # A named choice is meaningful only while a tool surface is actually
    # advertised.  Proposal synthesis intentionally closes that surface after
    # grounding; OpenAI-compatible request normalization can retain a legacy
    # choice flag even though ``tools`` is empty.  Rejecting its JSON response
    # as a missing tool call turns a valid Hawking-owned proposal boundary into
    # a false HTTP-200 semantic miss.
    if required_tool and request.tools and not tool_calls:
        return {
            "code": "REMOTE_SEMANTIC_INCOMPLETE",
            "message": "provider returned HTTP success without the required tool call",
            "reason": "required_tool_call_missing",
            "finish_reason": finish_text or None,
        }
    # A tool-call turn is an intermediate semantic completion, even when the
    # overall request also carries a terminal JSON response format.  Several
    # OpenAI-compatible providers (including the live Kimi route) correctly
    # return ``finish_reason=tool_calls`` with ``content=null``.  Applying the
    # terminal response-format requirement to that turn converts valid HTTP
    # 200 tool traffic into a false 502.  The named-call check above remains
    # fail-closed: an empty or unnamed call is not enough to cross the boundary.
    if structured and not tool_calls:
        if not has_content:
            return {
                "code": "REMOTE_SEMANTIC_INCOMPLETE",
                "message": "provider returned HTTP success without required structured content",
                "reason": "structured_content_missing",
                "finish_reason": finish_text or None,
            }
        try:
            json.loads(str(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            # A few OpenAI-compatible routes return a complete typed object
            # inside a short fence/prefix/suffix despite response_format. The
            # bounded structured extractor below can recover that object;
            # arbitrary prose still follows the fail-closed path.
            if _structured_text(str(text)):
                return None
            if request.transport_options.get("hawking_allow_read_only_prose_result") is True:
                # A Hawking-owned read observation already exists and the
                # caller explicitly opted into the read-only prose normalizer.
                # This does not waive the later completion contract or permit
                # mutation/acceptance from provider prose.
                return None
            return {
                "code": "REMOTE_SEMANTIC_INVALID",
                "message": "provider returned non-JSON content for a structured response",
                "reason": "structured_content_invalid_json",
                "finish_reason": finish_text or None,
            }
    # Empty content on a truncation/content-filter/error finish is never a
    # usable completion.  This is the exact 200/null/length failure observed
    # on the live route probe.  Also reject an entirely empty stop because it
    # carries no semantic result for a chat request.
    if not has_content and not tool_calls:
        return {
            "code": "REMOTE_SEMANTIC_INCOMPLETE",
            "message": "provider returned HTTP success without content or a tool call",
            "reason": "empty_assistant_message",
            "finish_reason": finish_text or None,
        }
    return None


def _semantic_miss_response(
    payload: Mapping[str, Any], problem: Mapping[str, Any]
) -> Dict[str, Any]:
    """Represent a provider HTTP-200 semantic miss without fabricating 502.

    A required tool call that never arrived is still unusable for a mutation
    contract, but it is not an upstream gateway outage. Return a typed
    Hawking-owned status envelope so the WorkUnit can classify/recover from
    the actual semantic condition without spending a transport retry.
    """
    safe = _safe_value(payload)
    result = dict(safe) if isinstance(safe, Mapping) else {}
    choices_raw = result.get("choices")
    choices = (
        [dict(item) for item in choices_raw if isinstance(item, Mapping)]
        if isinstance(choices_raw, list) else []
    )
    message = dict(choices[0].get("message") or {}) if choices else {}
    status = {
        "schema": "hawking.provider.semantic_miss.v1",
        "status": "MUTATION_PAYLOAD_MISSING",
        "provider_semantic_code": str(problem.get("code") or "REMOTE_SEMANTIC_INCOMPLETE"),
        "reason": str(problem.get("reason") or "semantic_completion_missing"),
    }
    message.update({"role": "assistant", "content": json.dumps(status, sort_keys=True)})
    message.pop("tool_calls", None)
    message.pop("function_call", None)
    if choices:
        choices[0]["message"] = message
        choices[0]["finish_reason"] = "stop"
    else:
        choices = [{"message": message, "finish_reason": "stop"}]
    result["choices"] = choices
    result["hawking_semantic_miss"] = status
    return result


def _surface_semantic_miss_as_response(
    request: GenerationRequest, problem: Mapping[str, Any]
) -> bool:
    """Allow only an explicit typed WorkUnit route to receive the envelope.

    The envelope represents an upstream HTTP-success semantic miss. It covers
    both an incomplete response (for example, a missing required tool call)
    and invalid structured content (for example, prose under json_object).
    Other callers retain the generic fail-closed gateway error behavior.
    """
    return (
        request.transport_options.get("hawking_semantic_miss_as_response") is True
        and str(problem.get("code") or "") in {
            "REMOTE_SEMANTIC_INCOMPLETE", "REMOTE_SEMANTIC_INVALID",
        }
    )


def _response_tool_calls(
    response: GenerationResponse, *, fallback_read_tool: str = "",
) -> List[Dict[str, Any]]:
    raw = response.raw if isinstance(response.raw, Mapping) else {}
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return []
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        return [
            dict(call) for call in calls
            if _tool_call_has_semantics(call)
        ]
    # Some otherwise OpenAI-compatible routes serialize a requested tool call
    # into assistant JSON instead of the wire ``message.tool_calls`` field.
    # It is still only a *proposal* until it passes the same named-tool and
    # registry checks below; normalize this narrow shape so a real read does
    # not become owner-side "theater" and strand the completion contract.
    content = message.get("content")
    if isinstance(content, str):
        try:
            encoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            encoded = None
        encoded_calls = encoded.get("tool_calls") if isinstance(encoded, Mapping) else None
        if isinstance(encoded_calls, list):
            normalized: List[Dict[str, Any]] = []
            for item in encoded_calls:
                if not isinstance(item, Mapping):
                    continue
                function = item.get("function")
                name = (
                    str(function.get("name") or "").strip()
                    if isinstance(function, Mapping)
                    else str(item.get("name") or "").strip()
                )
                arguments = (
                    function.get("arguments") if isinstance(function, Mapping)
                    else item.get("arguments", {})
                )
                if not name:
                    continue
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments if isinstance(arguments, Mapping) else {})
                normalized.append({
                    "id": str(item.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
            if normalized:
                return normalized
        # A few routes honor a named *read* tool choice but serialize only
        # that function's argument object in assistant content.  The caller
        # supplies this fallback only for a forced read; never infer a write
        # operation from content.  Registry validation still owns the path
        # and argument schema before any read is executed.
        if fallback_read_tool in {
            "fs.read", "filesystem.read", "source.read",
            "fs.search", "filesystem.search", "source.search",
        } and isinstance(encoded, Mapping) and "tool_calls" not in encoded:
            return [{
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": fallback_read_tool,
                    "arguments": json.dumps(dict(encoded)),
                },
            }]
    function = message.get("function_call")
    if _tool_call_has_semantics(function):
        return [{
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": dict(function),
        }]
    return []


def _workunit_completion_contract(context: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize an explicit Hawking-owned completion contract.

    Provider prose is never a completion authority.  This contract is opt-in
    because ordinary cognition and generic client passthrough must remain
    useful even when they are intentionally read-only.  A caller that owns a
    mutation WorkUnit can require real registry evidence here; the provider
    loop will then ask for bounded continuation instead of accepting a prose
    stop as completion.
    """
    raw = context.get("completion_contract")
    if not isinstance(raw, Mapping):
        return None
    required = raw.get("required_successful_tools") or raw.get("required_tools") or []
    if isinstance(required, str):
        required = [required]
    if not isinstance(required, (list, tuple, set, frozenset)):
        required = []
    names = sorted({str(item).strip() for item in required if str(item).strip()})
    any_required = raw.get("required_any_successful_tools") or []
    if isinstance(any_required, str):
        any_required = [any_required]
    if not isinstance(any_required, (list, tuple, set, frozenset)):
        any_required = []
    any_names = sorted({str(item).strip() for item in any_required if str(item).strip()})
    verified_required = raw.get("required_verified_tools") or []
    if isinstance(verified_required, str):
        verified_required = [verified_required]
    if not isinstance(verified_required, (list, tuple, set, frozenset)):
        verified_required = []
    verified_names = sorted({
        str(item).strip() for item in verified_required if str(item).strip()
    })
    try:
        continuation = int(raw.get("max_continuation_turns", 4))
    except (TypeError, ValueError):
        continuation = 4
    try:
        read_only_limit = int(raw.get("read_only_tool_call_limit", 0))
    except (TypeError, ValueError):
        read_only_limit = 0
    structured_keys = raw.get("structured_required_keys") or []
    if isinstance(structured_keys, str):
        structured_keys = [structured_keys]
    if not isinstance(structured_keys, (list, tuple, set, frozenset)):
        structured_keys = []
    structured_status_values = raw.get("structured_status_values") or []
    if isinstance(structured_status_values, str):
        structured_status_values = [structured_status_values]
    if not isinstance(structured_status_values, (list, tuple, set, frozenset)):
        structured_status_values = []
    first_tool = str(
        raw.get("required_first_tool") or raw.get("first_tool") or ""
    ).strip()
    focus_paths = raw.get("required_focus_paths") or []
    if isinstance(focus_paths, str):
        focus_paths = [focus_paths]
    if not isinstance(focus_paths, (list, tuple, set, frozenset)):
        focus_paths = []
    return {
        "required_successful_tools": (
            sorted(set(names) | ({"tests.run"} if bool(raw.get("require_final_tests_pass")) else set()))
        ),
        "required_any_successful_tools": any_names,
        "required_verified_tools": verified_names,
        "required_focus_paths": sorted({
            str(path).strip().replace("\\", "/")
            for path in focus_paths
            if str(path).strip()
        })[:32],
        "require_final_tests_pass": bool(raw.get("require_final_tests_pass")),
        "require_structured_result": bool(raw.get("require_structured_result")),
        "structured_required_keys": sorted({
            str(name).strip() for name in structured_keys if str(name).strip()
        })[:24],
        "structured_status_values": sorted({
            str(value).strip() for value in structured_status_values if str(value).strip()
        })[:12],
        "required_first_tool": first_tool,
        "max_continuation_turns": max(0, min(8, continuation)),
        "read_only_tool_call_limit": max(0, min(31, read_only_limit)),
        # A mutation-proposal turn is intentionally different from the legacy
        # direct-tool contract: the provider emits bounded data and Hawking
        # owns the repo.edit/test/status/diff effects. Preserve this bit while
        # normalizing the transport contract; dropping it silently re-enabled
        # the provider-forced repo.edit door.
        "mutation_proposal_mode": bool(raw.get("mutation_proposal_mode")),
        "require_regression_test_mutation": bool(raw.get("require_regression_test_mutation")),
        "required_regression_test_paths": [
            str(path).strip().replace("\\", "/").split("::", 1)[0]
            for path in (raw.get("required_regression_test_paths") or [])
            if str(path).strip()
        ][:4],
        "reminder": str(raw.get("reminder") or "").strip()[:1600],
    }


def _mutation_proposal_serialization_instruction(contract: Mapping[str, Any]) -> str:
    """Describe the compact intent first; retain literal proposals only for legacy routes."""
    paths = [str(path).strip() for path in (contract.get("required_regression_test_paths") or []) if str(path).strip()]
    existing = {
        str(path).strip() for path in (contract.get("existing_regression_test_paths") or []) if str(path).strip()
    }
    paired = bool(contract.get("require_regression_test_mutation")) and bool(paths)
    base = (
        "The admitted read budget is complete. Return exactly one compact JSON object. "
        "If any successful fs.read result contains source_anchor, the ONLY valid mutation shape is "
        "{s:MUTATE, anchor:<source-anchor-id>, op:replace, body:<replacement>, tests:[...]}: "
        "Hawking resolves the anchor, source digest, workspace, lease, and exact old bytes locally. "
        "Do not return old_text, source digests, workspace authority, lease data, Git revision, or repeated evidence. "
        "In compact mode do not emit status=MUTATION_PROPOSED, operations, edits, canonical_root, "
        "expected_base_hashes, or any legacy proposal envelope; those fields are invalid here. "
        "If no source_anchor is present, return {s:BLOCKED, reason:MISSING_SOURCE_ANCHOR, need:[...]} instead of "
        "reconstructing literal source. Legacy status=MUTATION_PROPOSED literal proposals are accepted only for "
        "explicit compatibility routes and still pass Hawking's canonical gates. "
    )
    if paired:
        base += (
            "operations MUST contain exactly two non-noop literal-byte operations: one substantive "
            "production-source operation and one test-file operation with complete pytest source at "
            "the dedicated regression path " + json.dumps(paths) + ". Use op=replace with an exact "
            "observed test anchor when a listed path already exists (existing paths: " + json.dumps(sorted(existing)) + "); "
            "use op=create only for an absent path. Naming it only in required_tests "
            "is invalid; a source-only payload will be rejected before repo.edit. Keep the source "
            "replacement narrowly anchored and under 1600 characters; never paste or duplicate a whole "
            "handler/doc block when a local branch or helper edit is sufficient. "
        )
    else:
        base += "operations MUST contain one real source operation. "
    return base + (
        "For legacy fallback, use op=replace with path, exact old_text, and complete new_text, or create with nonempty "
        "new_text/new_lines. For replace, old_text and new_text MUST differ and old_text MUST "
        "be an exact observed slice; never submit a function signature or any other no-op "
        "replacement. If the source already satisfies the intended behavior, choose a different "
        "small observable source behavior or report a precise blocker rather than emitting a "
        "no-op. change, kind, source_edit, anchor-only, and prose are invalid. "
        "Do not call tools, repeat archaeology, or emit prose."
    )


def _tool_trace_completion(
    trace: List[Dict[str, Any]],
    final: Optional[GenerationResponse],
    contract: Optional[Mapping[str, Any]],
    prior_tool_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive completion only from Hawking-owned tool observations."""
    prior_tool_evidence = (
        prior_tool_evidence if isinstance(prior_tool_evidence, Mapping) else {}
    )
    successful: set[str] = {
        str(name).strip()
        for name in (prior_tool_evidence.get("successful_tools") or [])
        if str(name).strip()
    }
    verified: set[str] = {
        str(name).strip()
        for name in (prior_tool_evidence.get("verified_tools") or [])
        if str(name).strip()
    }
    if not contract:
        return {"complete": True, "unmet": [], "successful_tools": []}
    # A passing tests.run is durable evidence for this WorkUnit; a later
    # provider turn must not erase it merely because that turn did not repeat
    # the test call.
    tests_passed = "tests.run" in successful
    for entry in trace:
        if not isinstance(entry, Mapping) or entry.get("ok") is not True:
            continue
        tool = str(entry.get("tool") or "")
        if not tool:
            continue
        value = entry.get("result")
        if isinstance(value, Mapping):
            value = value.get("value")
        if tool == "repo.edit" and isinstance(value, Mapping):
            accepted_statuses = {"accepted"}
            if contract.get("verification_mode") == (
                "engine_validation_and_status_diff_no_focused_tests"
            ):
                accepted_statuses.add("unproven")
            if not (
                str(value.get("status") or "").lower() in accepted_statuses
                and value.get("applied") is True
                and value.get("rolled_back") is not True
            ):
                continue
        if tool == "filesystem.write" and isinstance(value, Mapping):
            if value.get("changed") is not True:
                continue
        if tool == "tests.run":
            if isinstance(value, Mapping):
                tests_passed = bool(
                    value.get("verified") is True
                    and value.get("returncode") == 0
                )
            if not tests_passed:
                continue
        successful.add(tool)
        # Some tool calls are intentionally successful even when their
        # observed predicate is false (notably browser.verify). A successful
        # transport/action call is not proof of the requested page state.
        if isinstance(value, Mapping) and value.get("verified") is True:
            verified.add(tool)
    unmet = [
        name for name in contract.get("required_successful_tools", [])
        if str(name).strip() and str(name) not in successful
    ]
    any_required = [
        str(name) for name in (contract.get("required_any_successful_tools") or [])
        if str(name).strip()
    ]
    if any_required and not any(name in successful for name in any_required):
        unmet.append("one_of:" + ",".join(any_required))
    unmet.extend(
        f"{name}:verified"
        for name in contract.get("required_verified_tools", [])
        if str(name).strip() and str(name) not in verified
    )
    required_focus_paths = [
        str(path).strip().replace("\\", "/")
        for path in (contract.get("required_focus_paths") or [])
        if str(path).strip()
    ]
    if required_focus_paths:
        observed_paths: set[str] = set()
        for entry in trace:
            if not isinstance(entry, Mapping) or entry.get("ok") is not True:
                continue
            tool = str(entry.get("tool") or "").strip()
            if tool not in {"fs.read", "filesystem.read", "receipt.read", "receipt.inspect"}:
                continue
            arguments = entry.get("arguments")
            candidates: list[Any] = []
            if isinstance(arguments, Mapping):
                candidates.extend([arguments.get("path"), arguments.get("file")])
                values = arguments.get("paths")
                if isinstance(values, (list, tuple, set)):
                    candidates.extend(values)
            result = entry.get("result")
            if isinstance(result, Mapping):
                result = result.get("value")
            if isinstance(result, Mapping):
                candidates.extend([result.get("path"), result.get("file")])
            for value in candidates:
                token = str(value or "").strip().replace("\\", "/")
                if token:
                    observed_paths.add(token)
        missing_paths = [
            path for path in required_focus_paths
            if not any(observed == path or observed.endswith("/" + path) for observed in observed_paths)
        ]
        if missing_paths:
            unmet.append("focus_paths:" + ",".join(missing_paths))
    if contract.get("require_final_tests_pass") and not tests_passed:
        unmet.append("tests.run:verified")
    if contract.get("require_structured_result"):
        raw_structured = _structured_text(final.text if final is not None else "")
        structured = _normalized_research_result(
            final.text if final is not None else "", contract,
        )
        if not structured and contract.get("allow_read_only_prose_result"):
            structured = _normalized_read_only_prose_result(
                final.text if final is not None else "",
                contract,
                successful,
            )
        # Keep the old missing-key diagnostics for malformed/empty packets;
        # only a valid terminal observation packet receives defaultable fields.
        if not structured:
            structured = raw_structured
        if not structured:
            unmet.append("structured_result")
        else:
            missing_keys = [
                str(name) for name in (contract.get("structured_required_keys") or [])
                if str(name) and str(name) not in structured
            ]
            if missing_keys:
                unmet.append("structured_result:" + ",".join(missing_keys))
            allowed_status = {
                str(value).casefold() for value in (contract.get("structured_status_values") or [])
                if str(value)
            }
            if allowed_status and str(structured.get("status") or "").casefold() not in allowed_status:
                unmet.append("structured_result:terminal_status")
    # Preserve order while keeping the seal compact and deterministic.
    unmet = list(dict.fromkeys(str(item) for item in unmet))
    return {
        "complete": not unmet,
        "unmet": unmet,
        "successful_tools": sorted(successful),
        "verified_tools": sorted(verified),
    }


def _structured_result_response_format(
    trace: List[Dict[str, Any]],
    contract: Optional[Mapping[str, Any]],
    prior_tool_evidence: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, str]]:
    """Use provider JSON mode only after Hawking has the required evidence.

    A read-only worker first needs a legal opportunity to call a read tool.
    Once the daemon has that durable observation, JSON mode makes the final
    completion packet machine-readable without preventing the necessary tool
    call on the initial provider request.
    """
    if not isinstance(contract, Mapping) or not contract.get("require_structured_result"):
        return None
    completion = _tool_trace_completion(trace, None, contract, prior_tool_evidence)
    outstanding = [
        str(item) for item in (completion.get("unmet") or [])
        if str(item) != "structured_result"
    ]
    return {"type": "json_object"} if not outstanding else None


def _read_only_evidence_is_ready_for_terminal_packet(
    trace: Sequence[Mapping[str, Any]],
    contract: Optional[Mapping[str, Any]],
    prior_tool_evidence: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Whether a research turn has earned a tool-free terminal JSON reply."""
    if not isinstance(contract, Mapping) or not contract.get("require_structured_result"):
        return False
    if (
        contract.get("required_successful_tools")
        or contract.get("required_verified_tools")
        or contract.get("required_focus_paths")
    ):
        return False
    required = {
        str(name).strip()
        for name in (contract.get("required_any_successful_tools") or [])
        if str(name).strip()
    }
    if not required:
        return False
    successful = {
        str(name).strip()
        for name in ((prior_tool_evidence or {}).get("successful_tools") or [])
        if str(name).strip()
    }
    successful.update(
        str(entry.get("tool") or "").strip()
        for entry in trace
        if isinstance(entry, Mapping) and entry.get("ok") is True
    )
    return bool(required & successful)


def _terminal_research_packet_instruction() -> str:
    """Compact final-turn grammar after Hawking has already read evidence."""
    return (
        "Hawking has completed the admitted read-only observation. Do not call, "
        "name, or describe any tool. Return exactly one JSON object with these "
        "keys: observations (non-empty array of grounded strings), sources "
        "(array), recommendations (array), unresolved_questions (array), "
        "child_worker_ids (array), and status set to COMPLETE or "
        "STRUCTURED_RESULT_READY."
    )


def _next_bounded_mutation_tool(
    trace: List[Dict[str, Any]],
    contract: Optional[Mapping[str, Any]],
    prior_tool_evidence: Optional[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    calls_used: int,
) -> Optional[str]:
    """Return the next owed BUILD door after bounded orientation.

    A BUILD provider may inspect first, but provider preference is not an
    execution policy.  Once the declared read-only allowance is consumed,
    Hawking advances the existing completion contract by forcing the next
    admitted typed door.  This is still fail-closed: the tool must already be
    projected, write authority and the BUILD budget must be present, and the
    completion evidence is derived from Hawking observations only.
    """
    if not isinstance(contract, Mapping):
        return None
    budget = context.get("tool_budget")
    scope = budget.get("scope") if isinstance(budget, Mapping) else ""
    # The completion contract and explicit write authority are the canonical
    # admission boundary.  Older callers may not carry a scheduler scope
    # label, so do not make that legacy metadata a second veto.
    if context.get("tool_write_authority") is not True:
        return None
    required = {
        str(name).strip()
        for name in (contract.get("required_successful_tools") or [])
        if str(name).strip()
    }
    if "repo.edit" not in required:
        return None
    try:
        read_limit = int(contract.get("read_only_tool_call_limit") or 0)
    except (TypeError, ValueError):
        read_limit = 0
    # A real read observation is enough orientation for a bounded mutation
    # Goal.  Do not leave the next door hostage to a provider-specific call
    # counter: some routes stop after one or two reads and emit empty prose
    # while still ignoring an otherwise valid tool_choice.
    successful_trace_tools = {
        str(entry.get("tool") or "").strip()
        for entry in trace
        if isinstance(entry, Mapping) and entry.get("ok") is True
    }
    allowed_names = {
        str(name).strip()
        for name in (context.get("allowed_tool_names") or [])
        if str(name).strip()
    }
    if context.get("fresh_provider_session") is True:
        for name in ("fs.search", "filesystem.search", "source.search"):
            if name in allowed_names:
                return name
    # A provider may legally stop in prose before making its first read. For
    # a declared bounded mutation contract, make the initial orientation door
    # deterministic as well: the model still supplies the search arguments,
    # while Hawking narrows the admitted schema to a read-only search. The
    # next round is then forced through the observed repo.edit door.
    if not successful_trace_tools and calls_used == 0:
        if "fs.search" in allowed_names:
            return "fs.search"
        if "filesystem.search" in allowed_names:
            return "filesystem.search"
    read_observed = bool(successful_trace_tools & {
        "fs.read", "filesystem.read", "fs.search", "filesystem.search",
        "source.search", "source.read", "source.owner", "source.outline",
    })
    if not read_observed and calls_used < max(1, read_limit):
        return None
    completion = _tool_trace_completion(
        trace, None, contract, prior_tool_evidence,
    )
    successful = {
        str(name).strip()
        for name in (completion.get("successful_tools") or [])
        if str(name).strip()
    }
    # This ordering is the declared BUILD contract, not a second scheduler.
    # If a tool failed, it remains owed and is retried with the provider's
    # latest structured observation in the conversation.
    for name in ("repo.edit", "tests.run", "git.status", "git.diff"):
        if name in required and name not in successful:
            return name
    return None


def _request_summary(request: GenerationRequest) -> Dict[str, Any]:
    return {
        "request_id": request.request_id,
        "model": request.model,
        "message_count": len(request.messages),
        "roles": [str(item.get("role")) for item in request.messages if isinstance(item, Mapping)],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "has_response_schema": request.response_schema is not None,
        "tool_count": len(request.tools),
        "transport_options": _safe_mapping(request.transport_options),
    }


def _workunit_packet(wu: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the bounded provider packet; never serialize the live context."""
    allowed = {
        "objective": context.get("objective") or getattr(wu, "description", ""),
        "scope": context.get("scope") or context.get("acceptance") or [],
        "authority": context.get("authority") or "Hawking Mission verifier and operator contract",
        "context": context.get("context") or context.get("context_memory") or {},
        "tests": context.get("tests") or context.get("acceptance") or [],
        "stop": context.get("stop") or context.get("stop_conditions") or [],
        "cost": context.get("cost") or context.get("remote_cost_authorization") or {},
        "resources": context.get("resources") or {"resource_class": getattr(wu, "resource_class", None)},
        "unit_id": getattr(wu, "id", None),
        "role": getattr(wu, "role", None),
    }
    packet = _safe_mapping(allowed)
    encoded = json.dumps(packet, sort_keys=True, ensure_ascii=False)
    if len(encoded) <= 8000:
        return packet
    # Keep the contract fields visible while refusing to turn a provider
    # request into a transcript or an unbounded memory replay.
    return {
        "unit_id": packet.get("unit_id"),
        "role": packet.get("role"),
        "objective": str(packet.get("objective") or "")[:1200],
        "scope": packet.get("scope"),
        "authority": packet.get("authority"),
        "context": "[TRUNCATED_TO_BOUND_WORKUNIT_PACKET]",
        "tests": packet.get("tests"),
        "stop": packet.get("stop"),
        "cost": packet.get("cost"),
        "resources": packet.get("resources"),
    }


def _structured_text(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError):
        # Recover one complete JSON object from a provider wrapper. Do not
        # search or interpret arbitrary prose beyond the bounded object; if
        # raw_decode cannot consume an object, the caller receives {} and the
        # existing semantic rejection remains in force.
        try:
            start = candidate.find("{")
            if start < 0:
                return {}
            value, _end = json.JSONDecoder().raw_decode(candidate[start:])
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return _safe_mapping(value) if isinstance(value, Mapping) else {}


_RESEARCH_DEFAULTABLE_KEYS = frozenset({
    "sources", "recommendations", "unresolved_questions", "child_worker_ids",
})


def _normalized_research_result(
    text: Optional[str],
    contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize a bounded provider research packet without granting authority.

    Providers may return a terminal JSON packet with substantive observations
    while omitting empty bookkeeping arrays. Those fields are Hawking-owned
    defaults, not provider authority. Prose, empty observations, and
    nonterminal status remain unusable; read evidence is checked separately by
    ``_tool_trace_completion``.
    """
    value = _structured_text(text)
    observations = value.get("observations")
    if not isinstance(observations, list) or not any(
        str(item or "").strip() for item in observations
    ):
        return {}
    raw_status = str(value.get("status") or "").strip()
    allowed = {
        str(item).strip().casefold()
        for item in (contract or {}).get("structured_status_values", [])
        if str(item).strip()
    }
    if not raw_status or (allowed and raw_status.casefold() not in allowed):
        return {}
    normalized = dict(value)
    normalized["status"] = raw_status.upper()
    for key in _RESEARCH_DEFAULTABLE_KEYS:
        current = normalized.get(key)
        if current is None:
            normalized[key] = []
        elif not isinstance(current, list):
            return {}
    required = {
        str(item).strip()
        for item in (contract or {}).get("structured_required_keys", [])
        if str(item).strip()
    }
    if any(key not in normalized for key in (required - _RESEARCH_DEFAULTABLE_KEYS)):
        return {}
    return normalized


def _normalized_read_only_prose_result(
    text: Optional[str],
    contract: Optional[Mapping[str, Any]],
    successful_tools: set[str],
) -> Dict[str, Any]:
    """Compile a bounded read-only provider answer after real tool evidence.

    This is intentionally separate from the strict JSON normalizer: prose is
    never a mutation proposal or authority, but it is a useful research result
    once Hawking has already observed the required read-only operation.
    """
    value = str(text or "").strip()
    if not value or not any(
        str(name).strip() in successful_tools
        for name in (contract or {}).get("required_any_successful_tools", [])
    ):
        return {}
    if len(value) > 4000:
        value = value[:4000]
    normalized: Dict[str, Any] = {
        "observations": [value],
        "sources": [],
        "recommendations": [],
        "unresolved_questions": [],
        "child_worker_ids": [],
        "status": "COMPLETE",
    }
    required = {
        str(item).strip()
        for item in (contract or {}).get("structured_required_keys", [])
        if str(item).strip()
    }
    return normalized if required <= set(normalized) else {}


def _mutation_proposal_from_text(text: Optional[str]) -> Optional[Any]:
    """Prose is never treated as an edit; only explicit typed JSON qualifies."""
    raw = str(text or "").strip()
    if raw.startswith("HAWKING_PATCH_V1") or raw.startswith("HAWKING_EDIT_V1"):
        return raw
    value = _structured_text(raw)
    compact_status = str(value.get("s") or "").strip().upper()
    if compact_status == "MUTATE" and value.get("body") is not None:
        return value
    status = str(value.get("status") or "").strip().upper()
    if status == "MUTATION_PROPOSED":
        return value
    if (
        status in {"PATCH_PROPOSED", "EDIT_PROPOSED", "REPAIR_PROPOSED"}
        and isinstance(value.get("operations"), list)
        and value.get("operations")
    ):
        value["status"] = "MUTATION_PROPOSED"
        return value
    if status not in {"MUTATION_PROPOSED"}:
        return None
    return value


def _read_only_calls_from_proposal(
    proposal: Any,
    *,
    allowed_names: set[str],
    successful_tools: set[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Recover a provider's explicit read proposal as bounded observations.

    Some qualified routes return ``MUTATION_PROPOSED`` with only ``fs.read``
    or ``fs.search`` rows after the initial forced read.  Those are not source
    mutations and must never reach the engine as an operation with an empty
    ``op``.  They are nonetheless unambiguous, already within the projected
    capability surface, and can supply the final focused observation before a
    tool-free patch-synthesis request.  Mixed read/write payloads remain a
    normal proposal and still fail closed through the mutation boundary.
    """
    if not isinstance(proposal, Mapping):
        return []
    raw_ops = proposal.get("operations") or proposal.get("operation") or []
    if isinstance(raw_ops, Mapping):
        raw_ops = [raw_ops]
    if not isinstance(raw_ops, list) or not raw_ops:
        return []
    aliases = {
        "read": ("fs.read", "filesystem.read", "source.read"),
        "fs.read": ("fs.read", "filesystem.read", "source.read"),
        "filesystem.read": ("filesystem.read", "fs.read", "source.read"),
        "source.read": ("source.read", "fs.read", "filesystem.read"),
        "search": ("fs.search", "filesystem.search", "source.search"),
        "fs.search": ("fs.search", "filesystem.search", "source.search"),
        "filesystem.search": ("filesystem.search", "fs.search", "source.search"),
        "source.search": ("source.search", "fs.search", "filesystem.search"),
    }
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for raw in raw_ops:
        if not isinstance(raw, Mapping):
            return []
        kind = str(
            raw.get("op") or raw.get("kind") or raw.get("operation")
            or raw.get("tool") or ""
        ).strip().lower()
        candidates = aliases.get(kind)
        if not candidates:
            return []
        name = next((item for item in candidates if item in allowed_names), "")
        if not name:
            return []
        arguments = {
            str(key): value for key, value in raw.items()
            if key not in {"op", "kind", "operation", "tool", "tool_name"}
        }
        calls.append((name, arguments))
    # Prefer an observation from a new tool family (normally focused search
    # after the forced source read), while retaining deterministic order.
    fresh = [item for item in calls if item[0] not in successful_tools]
    return fresh or calls


def _response_proposal_text(response: Any) -> str:
    """Recover typed proposal content even when a provider also emits calls."""
    text = str(getattr(response, "text", "") or "")
    if text.strip():
        return text
    raw = getattr(response, "raw", None)
    if isinstance(raw, Mapping):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping):
                return str(message.get("content") or "")
    return ""


class _RemoteProviderBase:
    is_remote_cognition = True

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        workspace: Optional[Path] = None,
        remote_slot: Optional[str] = None,
        cost_policy: Optional[RemoteCostPolicy] = None,
    ) -> None:
        self.provider_name = str(provider)
        self.model_id = str(model_id)
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        self.remote_slot = str(remote_slot) if remote_slot else None
        self.cost_policy = cost_policy or RemoteCostPolicy.from_environment(self.workspace)
        self._last_receipt: Optional[Dict[str, Any]] = None
        self._health = ProviderHealth(
            state="unknown",
            ready=False,
            provider=self.provider_name,
            detail="remote provider has not been probed",
            recoverable=True,
        )
        self._cancel_events: Dict[str, Event] = {}
        self._cancel_requests: Dict[str, Tuple[GenerationRequest, float]] = {}
        self._cancel_lock = Lock()

    @property
    def last_receipt(self) -> Optional[Dict[str, Any]]:
        return _copy(self._last_receipt) if self._last_receipt else None

    def _identity(self, *, request_id: Optional[str] = None, remote_request_id: Optional[str] = None) -> Dict[str, Any]:
        return RemoteExecutionIdentity(
            provider=self.provider_name,
            model_id=self.model_id,
            backend="remote",
            protocol="provider-managed",
            transport=self.provider_name,
            remote_slot=self.remote_slot,
            request_id=request_id,
            remote_request_id=remote_request_id,
        ).to_dict()

    def identity(self) -> Dict[str, Any]:
        return self._identity()

    def capabilities(self) -> CapabilityContract:
        return CapabilityContract(
            features={
                feature: Capability(
                    state="unknown",
                    enforcement="unknown",
                    source="remote provider capability not probed",
                )
                for feature in FEATURES
            }
        )

    def supports(self, feature: str) -> Optional[bool]:
        return self.capabilities().supports(feature)

    def health(self) -> ProviderHealth:
        return self._health

    def profile(self) -> ResidentProfile:
        identity = self.identity()
        return ResidentProfile(
            profile_id=f"{self.provider_name}:{self.model_id}",
            provider=self.provider_name,
            model_id=self.model_id,
            artifact={"model_identity": f"{self.provider_name}:{self.model_id}"},
            runtime={
                "backend": "remote",
                "runtime": "remote",
                "protocol": identity.get("protocol"),
                "transport": identity.get("transport"),
                "pid": None,
                "port": None,
                "remote_slot": self.remote_slot,
            },
            capabilities=self.capabilities(),
            limits={"remote": True, "pid": None, "port": None},
            qualification={"status": "unprobed", "authority": "provider observation only"},
            metadata={"identity_snapshot": identity},
        )

    def _event_for(
        self,
        request_id: str,
        request: Optional[GenerationRequest] = None,
        started_at: Optional[float] = None,
    ) -> Event:
        event = Event()
        with self._cancel_lock:
            self._cancel_events[request_id] = event
            if request is not None:
                self._cancel_requests[request_id] = (
                    request,
                    float(started_at if started_at is not None else time.time()),
                )
        return event

    def _drop_event(self, request_id: str) -> None:
        with self._cancel_lock:
            self._cancel_events.pop(request_id, None)
            self._cancel_requests.pop(request_id, None)

    def cancel(self, request_id: str) -> Dict[str, Any]:
        key = str(request_id or "")
        with self._cancel_lock:
            event = self._cancel_events.get(key)
            context = self._cancel_requests.get(key)
        if event is not None:
            event.set()
            if context is not None:
                request, started_at = context
                self._failure_receipt(
                    request=request,
                    started_at=started_at,
                    status="cancelled",
                    error=RemoteCancelledError(
                        "REMOTE_CANCELLED",
                        "remote request cancelled by the Hawking gateway client",
                        provider=self.provider_name,
                        model_id=self.model_id,
                        request_id=key,
                        recoverable=False,
                    ),
                )
        return {"request_id": key, "cancel_requested": event is not None, "pid": None, "port": None}

    def stop(self) -> Dict[str, Any]:
        return {"gone": True, "provider": self.provider_name, "pid": None, "port": None}

    def _cancelled(self, request: Optional[GenerationRequest], event: Optional[Event]) -> bool:
        if event is not None and event.is_set():
            return True
        metadata = request.metadata if request is not None else {}
        callback = metadata.get("is_cancelled") if isinstance(metadata, Mapping) else None
        if callable(callback):
            try:
                return bool(callback())
            except Exception:
                return False
        external = metadata.get("cancel_event") if isinstance(metadata, Mapping) else None
        return bool(external.is_set()) if hasattr(external, "is_set") else False

    def _failure_receipt(
        self,
        *,
        request: Optional[GenerationRequest],
        started_at: float,
        finished_at: Optional[float] = None,
        status: str = "failed",
        error: Optional[RemoteCognitionError] = None,
        retries: int = 0,
        remote_request_id: Optional[str] = None,
        usage: Optional[Mapping[str, Any]] = None,
        cost_usd: Optional[float] = None,
        provider_route: Optional[str] = None,
    ) -> Dict[str, Any]:
        finished = time.time() if finished_at is None else finished_at
        estimated_cost = None
        if request is not None and isinstance(request.metadata, Mapping):
            candidate = request.metadata.get("estimated_cost_usd")
            try:
                estimated_cost = float(candidate) if candidate is not None else None
            except (TypeError, ValueError):
                estimated_cost = None
        transport = request.transport_options if request is not None else {}
        preferred_provider = None
        if isinstance(transport, Mapping):
            routing = transport.get("provider")
            if isinstance(routing, Mapping):
                order = routing.get("order")
                if isinstance(order, (list, tuple)) and order:
                    preferred_provider = str(order[0])
        payload: Dict[str, Any] = {
            "schema": REMOTE_RECEIPT_SCHEMA,
            "status": status,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "provider_route": provider_route,
            "preferred_provider": preferred_provider,
            "resolved_provider": provider_route,
            "request": _request_summary(request) if request else None,
            "request_id": request.request_id if request else None,
            "remote_request_id": remote_request_id,
            "remote_slot": (
                self.remote_slot
                or (
                    request.metadata.get("remote_slot") or request.metadata.get("slot")
                    if request is not None and isinstance(request.metadata, Mapping)
                    else None
                )
            ),
            "pid": None,
            "port": None,
            "started_at": started_at,
            "finished_at": finished,
            "latency_ms": max(0.0, (finished - started_at) * 1000.0),
            "retries": max(0, int(retries)),
            "usage": _safe_mapping(usage or {}),
            "cost_usd": cost_usd,
            "cost_accounting": {
                "estimated_cost_usd": estimated_cost,
                "provider_reported_cost_usd": cost_usd,
                "final_accounted_cost_usd": cost_usd if status == "completed" else None,
            },
            "error": error.to_dict() if error else None,
        }
        self._last_receipt = _safe_mapping(payload)
        return _copy(self._last_receipt)

    def failure_result(self, error: BaseException, request: Optional[GenerationRequest] = None) -> Dict[str, Any]:
        if isinstance(error, RemoteCognitionError):
            structured = error
        else:
            structured = RemoteCognitionError(
                "REMOTE_PROVIDER_ERROR",
                f"{type(error).__name__}: {error}",
                provider=self.provider_name,
                model_id=self.model_id,
                request_id=request.request_id if request else None,
                recoverable=True,
            )
        receipt = self._last_receipt or self._failure_receipt(
            request=request,
            started_at=time.time(),
            error=structured,
        )
        return {
            "status": "failed",
            "backend": self.provider_name,
            "provider": self.provider_name,
            "model": self.model_id,
            "error": structured.to_dict(),
            "remote_cognition": receipt,
            "provider_receipt": receipt,
            "validation": {"ok": False, "reason": structured.code},
            "operations": [],
            "tests": [],
            "questions": [],
        }


class OpenRouterProvider(_RemoteProviderBase):
    """Lazy OpenRouter provider for an arbitrary authorized model id."""

    def __init__(
        self,
        model_id: str,
        *,
        workspace: Optional[Path] = None,
        endpoint: str = DEFAULT_OPENROUTER_ENDPOINT,
        cost_policy: Optional[RemoteCostPolicy] = None,
        key_resolver: Optional[Callable[[], Optional[str]]] = None,
        urlopen: Optional[Callable[..., Any]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
        timeout: float = DEFAULT_REMOTE_TIMEOUT_S,
        max_attempts: int = DEFAULT_REMOTE_ATTEMPTS,
        backoff_s: float = DEFAULT_REMOTE_BACKOFF_S,
        max_backoff_s: float = DEFAULT_REMOTE_MAX_BACKOFF_S,
        remote_slot: Optional[str] = None,
        catalog_ttl_s: float = DEFAULT_CATALOG_TTL_S,
    ) -> None:
        super().__init__(
            provider="openrouter",
            model_id=str(model_id).strip(),
            workspace=workspace,
            remote_slot=remote_slot,
            cost_policy=cost_policy,
        )
        if not self.model_id:
            raise ValueError("OpenRouter model id is required")
        self.endpoint = str(endpoint or DEFAULT_OPENROUTER_ENDPOINT).strip()
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OpenRouter endpoint must be an http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("OpenRouter endpoint URL credentials are not allowed")
        self._key_resolver = key_resolver
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleeper or time.sleep
        self.timeout = max(0.1, float(timeout))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_s = max(0.0, float(backoff_s))
        self.max_backoff_s = max(0.0, float(max_backoff_s))
        self.catalog_ttl_s = max(0.0, float(catalog_ttl_s))
        self._catalog: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._last_attempts = 0
        self._last_remote_request_id: Optional[str] = None

    def _identity(self, *, request_id: Optional[str] = None, remote_request_id: Optional[str] = None) -> Dict[str, Any]:
        identity = RemoteExecutionIdentity(
            provider="openrouter",
            model_id=self.model_id,
            backend="remote",
            protocol="openai-chat-completions",
            transport="openrouter",
            remote_slot=self.remote_slot,
            endpoint=self.endpoint,
            request_id=request_id,
            remote_request_id=remote_request_id,
        ).to_dict()
        identity["credential_source"] = "OPENROUTER_API_KEY or configured keychain resolver"
        identity["credential_value"] = "[REDACTED]"
        return identity

    def _api_key(self) -> str:
        token = os.environ.get("OPENROUTER_API_KEY")
        if token:
            return token
        if callable(self._key_resolver):
            try:
                token = self._key_resolver()
            except Exception as exc:
                raise RemoteAuthError(
                    "OPENROUTER_CREDENTIAL_UNAVAILABLE",
                    f"configured OpenRouter secret resolver failed: {type(exc).__name__}",
                    provider="openrouter",
                    model_id=self.model_id,
                    recoverable=False,
                ) from exc
        if token:
            return str(token)
        raise RemoteAuthError(
            "OPENROUTER_API_KEY_MISSING",
            "OPENROUTER_API_KEY is not configured and no keychain resolver returned a credential",
            provider="openrouter",
            model_id=self.model_id,
            recoverable=False,
        )

    def _headers(self, api_key: str, request_id: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hawking-remote-cognition/1",
        }
        if request_id:
            headers["X-Request-ID"] = str(request_id)
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    def _cancel_error(self, request: GenerationRequest) -> RemoteCancelledError:
        return RemoteCancelledError(
            "REMOTE_CANCELLED",
            "remote request cancelled before completion",
            provider="openrouter",
            model_id=self.model_id,
            request_id=request.request_id,
            recoverable=False,
        )

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(max(0.0, float(retry_after)), self.max_backoff_s)
        return min(self.backoff_s * (2 ** max(0, attempt - 1)), self.max_backoff_s)

    def _request_json(
        self,
        *,
        url: str,
        method: str,
        body: Optional[Mapping[str, Any]],
        request: Optional[GenerationRequest],
        timeout: float,
        event: Optional[Event],
    ) -> Tuple[Any, Any, int, int, Optional[str]]:
        request_id = request.request_id if request else f"catalog-{uuid.uuid4()}"
        api_key = self._api_key()
        payload = json.dumps(body, sort_keys=True).encode("utf-8") if body is not None else None
        headers = self._headers(api_key, request_id)
        attempt_limit = (
            1
            if request is not None
            and request.transport_options.get("hawking_fast_failover") is True
            else self.max_attempts
        )
        for attempt in range(1, attempt_limit + 1):
            self._last_attempts = attempt
            if self._cancelled(request, event):
                if request is not None:
                    raise self._cancel_error(request)
                raise RemoteCancelledError(
                    "REMOTE_CANCELLED",
                    "remote catalog request cancelled",
                    provider="openrouter",
                    model_id=self.model_id,
                    request_id=request_id,
                )
            req = urllib.request.Request(url, data=payload, headers=headers, method=method)
            try:
                try:
                    response = self._urlopen(req, timeout=timeout)
                    data, response_headers, status = _read_json(response)
                except urllib.error.HTTPError as exc:
                    try:
                        body_text = _redact_text(exc.read().decode("utf-8", errors="replace"))
                    finally:
                        exc.close()
                    raise _HTTPStatus(exc.code, body_text, getattr(exc, "headers", None)) from exc
                remote_id = _extract_remote_request_id(data, response_headers)
                self._last_remote_request_id = remote_id
                return data, response_headers, status, attempt - 1, remote_id
            except _HTTPStatus as exc:
                remote_id = _header(exc.headers, "x-request-id", "x-openrouter-request-id", "request-id")
                self._last_remote_request_id = remote_id
                retry_after = _parse_retry_after(_header(exc.headers, "retry-after"))
                if exc.status in {401, 403}:
                    raise RemoteAuthError(
                        f"OPENROUTER_AUTH_HTTP_{exc.status}",
                        f"OpenRouter authentication rejected (HTTP {exc.status})",
                        provider="openrouter",
                        model_id=self.model_id,
                        request_id=request_id,
                        remote_request_id=remote_id,
                        status_code=exc.status,
                        recoverable=False,
                        details={"response": exc.body},
                    ) from exc
                retryable = exc.status == 429 or 500 <= exc.status <= 599
                if retryable and attempt < attempt_limit:
                    self._sleep(self._backoff(attempt, retry_after))
                    continue
                if exc.status == 429:
                    raise RemoteRateLimitError(
                        "OPENROUTER_RATE_LIMITED",
                        "OpenRouter rate limit exhausted the bounded retry budget",
                        provider="openrouter",
                        model_id=self.model_id,
                        request_id=request_id,
                        remote_request_id=remote_id,
                        status_code=exc.status,
                        retry_after_s=retry_after,
                        recoverable=True,
                        details={"response": exc.body, "attempts": attempt},
                    ) from exc
                raise RemoteCognitionError(
                    f"OPENROUTER_HTTP_{exc.status}",
                    f"OpenRouter request failed with HTTP {exc.status}: {exc.body}",
                    provider="openrouter",
                    model_id=self.model_id,
                    request_id=request_id,
                    remote_request_id=remote_id,
                    status_code=exc.status,
                    recoverable=retryable,
                    details={"response": exc.body, "attempts": attempt},
                ) from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
                if attempt < attempt_limit:
                    self._sleep(self._backoff(attempt, None))
                    continue
                raise RemoteCognitionError(
                    "OPENROUTER_NETWORK_TIMEOUT" if isinstance(exc, (socket.timeout, TimeoutError)) else "OPENROUTER_NETWORK_ERROR",
                    f"OpenRouter request failed after {attempt} attempt(s): {type(exc).__name__}",
                    provider="openrouter",
                    model_id=self.model_id,
                    request_id=request_id,
                    recoverable=True,
                    details={"attempts": attempt},
                ) from exc
            except ValueError as exc:
                raise RemoteCognitionError(
                    "OPENROUTER_INVALID_JSON",
                    str(exc),
                    provider="openrouter",
                    model_id=self.model_id,
                    request_id=request_id,
                    recoverable=False,
                    details={"attempts": attempt},
                ) from exc
        raise AssertionError("bounded remote retry loop fell through")

    def _request_stream(
        self,
        *,
        url: str,
        body: Mapping[str, Any],
        request: GenerationRequest,
        timeout: float,
        event: Optional[Event],
    ) -> Tuple[Any, int, Optional[str]]:
        """Open one bounded SSE response, retrying only before a stream exists."""
        request_id = request.request_id
        api_key = self._api_key()
        payload = json.dumps(body, sort_keys=True).encode("utf-8")
        headers = self._headers(api_key, request_id)
        headers["Accept"] = "text/event-stream"
        attempt_limit = (
            1
            if request.transport_options.get("hawking_fast_failover") is True
            else self.max_attempts
        )
        for attempt in range(1, attempt_limit + 1):
            self._last_attempts = attempt
            if self._cancelled(request, event):
                raise self._cancel_error(request)
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                try:
                    response = self._urlopen(req, timeout=timeout)
                    status = int(getattr(response, "status", getattr(response, "code", 200)))
                    if status < 200 or status >= 300:
                        body_text = _redact_text(_read_body(response).decode("utf-8", errors="replace"))
                        raise _HTTPStatus(status, body_text, getattr(response, "headers", None))
                except urllib.error.HTTPError as exc:
                    try:
                        body_text = _redact_text(exc.read().decode("utf-8", errors="replace"))
                    finally:
                        exc.close()
                    raise _HTTPStatus(exc.code, body_text, getattr(exc, "headers", None)) from exc
                remote_id = _extract_remote_request_id(None, getattr(response, "headers", None))
                self._last_remote_request_id = remote_id
                return response, attempt - 1, remote_id
            except _HTTPStatus as exc:
                remote_id = _header(exc.headers, "x-request-id", "x-openrouter-request-id", "request-id")
                self._last_remote_request_id = remote_id
                retry_after = _parse_retry_after(_header(exc.headers, "retry-after"))
                if exc.status in {401, 403}:
                    raise RemoteAuthError(
                        f"OPENROUTER_AUTH_HTTP_{exc.status}",
                        f"OpenRouter authentication rejected (HTTP {exc.status})",
                        provider="openrouter",
                        model_id=self.model_id,
                        request_id=request_id,
                        remote_request_id=remote_id,
                        status_code=exc.status,
                        recoverable=False,
                        details={"response": exc.body},
                    ) from exc
                retryable = exc.status == 429 or 500 <= exc.status <= 599
                if retryable and attempt < attempt_limit:
                    self._sleep(self._backoff(attempt, retry_after))
                    continue
                if exc.status == 429:
                    raise RemoteRateLimitError(
                        "OPENROUTER_RATE_LIMITED",
                        "OpenRouter rate limit exhausted the bounded retry budget",
                        provider="openrouter",
                        model_id=self.model_id,
                        request_id=request_id,
                        remote_request_id=remote_id,
                        status_code=exc.status,
                        retry_after_s=retry_after,
                        recoverable=True,
                        details={"response": exc.body, "attempts": attempt},
                    ) from exc
                raise RemoteCognitionError(
                    f"OPENROUTER_HTTP_{exc.status}",
                    f"OpenRouter request failed with HTTP {exc.status}: {exc.body}",
                    provider="openrouter",
                    model_id=self.model_id,
                    request_id=request_id,
                    remote_request_id=remote_id,
                    status_code=exc.status,
                    recoverable=retryable,
                    details={"response": exc.body, "attempts": attempt},
                ) from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
                if attempt < attempt_limit:
                    self._sleep(self._backoff(attempt, None))
                    continue
                raise RemoteCognitionError(
                    "OPENROUTER_NETWORK_TIMEOUT" if isinstance(exc, (socket.timeout, TimeoutError)) else "OPENROUTER_NETWORK_ERROR",
                    f"OpenRouter stream failed after {attempt} attempt(s): {type(exc).__name__}",
                    provider="openrouter",
                    model_id=self.model_id,
                    request_id=request_id,
                    recoverable=True,
                    details={"attempts": attempt},
                ) from exc
        raise AssertionError("bounded remote stream retry loop fell through")

    @staticmethod
    def _stream_records(response: Any):
        """Yield decoded SSE data records without buffering the completion."""
        readline = getattr(response, "readline", None)
        if not callable(readline):
            raw = response.read()
            lines = raw.splitlines() if isinstance(raw, (bytes, bytearray)) else str(raw).splitlines()
            iterator = iter(lines)
        else:
            # ``http.client`` returns bytes, while small in-process test and
            # embedding transports often return text.  A fixed bytes sentinel
            # would spin forever on the latter after EOF.
            def _lines():
                while True:
                    line = readline()
                    if line in (b"", ""):
                        break
                    yield line
            iterator = _lines()
        data_lines: List[str] = []
        for raw_line in iterator:
            if raw_line in (b"", ""):
                if not data_lines:
                    continue
                raw = "\n".join(data_lines).strip()
                data_lines = []
                if raw == "[DONE]":
                    break
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, Mapping):
                    yield dict(value)
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, (bytes, bytearray)) else str(raw_line)
            line = line.rstrip("\r\n")
            if not line:
                if data_lines:
                    raw = "\n".join(data_lines).strip()
                    data_lines = []
                    if raw == "[DONE]":
                        break
                    try:
                        value = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, Mapping):
                        yield dict(value)
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            raw = "\n".join(data_lines).strip()
            if raw != "[DONE]":
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError):
                    value = None
                if isinstance(value, Mapping):
                    yield dict(value)

    def generate_stream(
        self,
        request: GenerationRequest,
        *,
        timeout: Optional[float] = None,
    ):
        """Stream OpenRouter SSE deltas through the common provider contract."""
        normalized = request if isinstance(request, GenerationRequest) else GenerationRequest.from_mapping(request)
        if normalized.model and normalized.model != self.model_id:
            raise RemoteCognitionError(
                "MODEL_ID_MISMATCH",
                f"request model {normalized.model!r} does not match selected OpenRouter model {self.model_id!r}",
                provider="openrouter",
                model_id=self.model_id,
                request_id=normalized.request_id,
                recoverable=False,
            )
        started = time.time()
        event = self._event_for(normalized.request_id, normalized, started)
        response = None
        remote_id = None
        retries = 0
        usage: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        provider_route: Optional[str] = None
        stream_text_parts: List[str] = []
        stream_tool_calls: List[Dict[str, Any]] = []
        try:
            limit = self.cost_policy.authorize(
                provider="openrouter",
                model_id=self.model_id,
                request_id=normalized.request_id,
                metadata=normalized.metadata,
            )
            payload = normalized.to_payload()
            payload["model"] = self.model_id
            payload["stream"] = True
            response, retries, remote_id = self._request_stream(
                url=self.endpoint,
                body=payload,
                request=normalized,
                timeout=float(timeout if timeout is not None else self.timeout),
                event=event,
            )
            for record in self._stream_records(response):
                if self._cancelled(normalized, event):
                    raise self._cancel_error(normalized)
                if record.get("id") is not None and remote_id is None:
                    remote_id = str(record.get("id"))
                if record.get("provider") not in (None, ""):
                    provider_route = str(record.get("provider"))
                record_usage = record.get("usage")
                if isinstance(record_usage, Mapping):
                    usage.update(dict(record_usage))
                choices = record.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice.get("finish_reason"))
                delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
                text = _content_text(delta.get("content"))
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for position, raw_call in enumerate(tool_calls):
                        if not isinstance(raw_call, Mapping):
                            continue
                        raw_index = raw_call.get("index")
                        try:
                            index = int(raw_index) if raw_index is not None else position
                        except (TypeError, ValueError):
                            index = position
                        while len(stream_tool_calls) <= index:
                            stream_tool_calls.append({
                                "index": len(stream_tool_calls),
                                "type": "function",
                                "function": {},
                            })
                        accumulated = stream_tool_calls[index]
                        for key in ("id", "type"):
                            value = raw_call.get(key)
                            if value not in (None, ""):
                                accumulated[key] = value
                        raw_function = raw_call.get("function")
                        if isinstance(raw_function, Mapping):
                            function = accumulated.setdefault("function", {})
                            if not isinstance(function, dict):
                                function = {}
                                accumulated["function"] = function
                            name = raw_function.get("name")
                            if name not in (None, ""):
                                function["name"] = name
                            arguments = raw_function.get("arguments")
                            if arguments not in (None, ""):
                                previous = function.get("arguments")
                                if previous in (None, ""):
                                    function["arguments"] = arguments
                                elif isinstance(previous, str) and isinstance(arguments, str):
                                    function["arguments"] = previous + arguments
                                else:
                                    function["arguments"] = arguments
                if text or tool_calls:
                    if text:
                        stream_text_parts.append(text)
                    yield {
                        "type": "delta",
                        "text": text or "",
                        "tool_calls": _safe_value(tool_calls) if tool_calls else None,
                        "raw": _safe_value(record),
                    }
            stream_payload: Dict[str, Any] = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "".join(stream_text_parts) or None,
                        **({
                            "tool_calls": [
                                call for call in stream_tool_calls
                                if _tool_call_has_semantics(call)
                            ]
                        } if any(
                            _tool_call_has_semantics(call)
                            for call in stream_tool_calls
                        ) else {}),
                    },
                    "finish_reason": finish_reason or "stop",
                }]
            }
            semantic_problem = _semantic_response_problem(
                normalized,
                stream_payload,
                "".join(stream_text_parts) or None,
                finish_reason or "stop",
            )
            semantic_miss = None
            if semantic_problem:
                if _surface_semantic_miss_as_response(normalized, semantic_problem):
                    stream_payload = _semantic_miss_response(stream_payload, semantic_problem)
                    miss_text, finish_reason, _unused_usage, _unused_safe = _normalized_response(stream_payload)
                    semantic_miss = dict(stream_payload["hawking_semantic_miss"])
                    if miss_text:
                        yield {
                            "type": "delta",
                            "text": miss_text,
                            "tool_calls": None,
                            "raw": _safe_value(stream_payload),
                        }
                else:
                    raise RemoteCognitionError(
                        str(semantic_problem["code"]),
                        str(semantic_problem["message"]),
                        provider="openrouter",
                        model_id=self.model_id,
                        request_id=normalized.request_id,
                        remote_request_id=remote_id,
                        recoverable=True,
                        status_code=502,
                        details=semantic_problem,
                    )
            cost = _extract_cost({}, usage)
            observed_cost = self.cost_policy.observe(
                cost,
                limit_usd=limit,
                provider="openrouter",
                model_id=self.model_id,
                request_id=normalized.request_id,
            )
            receipt = self._finish_receipt(
                request=normalized,
                started_at=started,
                status="semantic_incomplete" if semantic_miss else "completed",
                attempts=retries + 1,
                remote_request_id=remote_id,
                usage=usage,
                cost_usd=observed_cost,
                provider_route=provider_route,
            )
            usage.update({
                "cost_usd": observed_cost,
                "remote_request_id": remote_id,
                "latency_ms": receipt.get("latency_ms"),
                "retries": retries,
            })
            if semantic_miss:
                usage["hawking_semantic_miss"] = semantic_miss
            yield {
                "type": "done",
                "finish_reason": finish_reason or "stop",
                "usage": usage,
                "receipt": receipt,
                "remote_request_id": remote_id,
            }
        except RemoteCognitionError as exc:
            observed_error_cost = None
            if isinstance(exc.details, Mapping):
                candidate = exc.details.get("observed_cost_usd")
                try:
                    observed_error_cost = float(candidate) if candidate is not None else None
                except (TypeError, ValueError):
                    observed_error_cost = None
            self._finish_receipt(
                request=normalized,
                started_at=started,
                status="cancelled" if isinstance(exc, RemoteCancelledError) else "failed",
                attempts=max(1, int(getattr(self, "_last_attempts", 1) or 1)),
                remote_request_id=exc.remote_request_id or remote_id or getattr(self, "_last_remote_request_id", None),
                cost_usd=observed_error_cost,
                error=exc,
            )
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            self._drop_event(normalized.request_id)

    def _finish_receipt(
        self,
        *,
        request: GenerationRequest,
        started_at: float,
        status: str,
        attempts: int,
        remote_request_id: Optional[str],
        usage: Optional[Mapping[str, Any]] = None,
        cost_usd: Optional[float] = None,
        error: Optional[RemoteCognitionError] = None,
        provider_route: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._failure_receipt(
            request=request,
            started_at=started_at,
            status=status,
            retries=max(0, attempts - 1),
            remote_request_id=remote_request_id,
            usage=usage,
            cost_usd=cost_usd,
            error=error,
            provider_route=provider_route,
        )

    def generate(self, request: GenerationRequest, *, timeout: Optional[float] = None) -> GenerationResponse:
        normalized = request if isinstance(request, GenerationRequest) else GenerationRequest.from_mapping(request)
        if normalized.model and normalized.model != self.model_id:
            raise RemoteCognitionError(
                "MODEL_ID_MISMATCH",
                f"request model {normalized.model!r} does not match selected OpenRouter model {self.model_id!r}",
                provider="openrouter",
                model_id=self.model_id,
                request_id=normalized.request_id,
                recoverable=False,
            )
        started = time.time()
        event = self._event_for(normalized.request_id, normalized, started)
        limit = 0.0
        try:
            limit = self.cost_policy.authorize(
                provider="openrouter",
                model_id=self.model_id,
                request_id=normalized.request_id,
                metadata=normalized.metadata,
            )
            if self._cancelled(normalized, event):
                raise self._cancel_error(normalized)
            payload = normalized.to_payload()
            payload["model"] = self.model_id
            data, headers, _status, retries, remote_id = self._request_json(
                url=self.endpoint,
                method="POST",
                body=payload,
                request=normalized,
                timeout=float(timeout if timeout is not None else self.timeout),
                event=event,
            )
            if self._cancelled(normalized, event):
                raise self._cancel_error(normalized)
            text, finish, usage, safe = _normalized_response(data)
            semantic_problem = _semantic_response_problem(
                normalized, safe, text, finish
            )
            semantic_miss = None
            if semantic_problem:
                if _surface_semantic_miss_as_response(normalized, semantic_problem):
                    safe = _semantic_miss_response(safe, semantic_problem)
                    text, finish, usage, safe = _normalized_response(safe)
                    semantic_miss = dict(safe["hawking_semantic_miss"])
                else:
                    raise RemoteCognitionError(
                        str(semantic_problem["code"]),
                        str(semantic_problem["message"]),
                        provider="openrouter",
                        model_id=self.model_id,
                        request_id=normalized.request_id,
                        remote_request_id=remote_id,
                        recoverable=True,
                        status_code=502,
                        details=semantic_problem,
                    )
            cost = _extract_cost(data if isinstance(data, Mapping) else {}, usage)
            observed_cost = self.cost_policy.observe(
                cost,
                limit_usd=limit,
                provider="openrouter",
                model_id=self.model_id,
                request_id=normalized.request_id,
            )
            provider_route = None
            if isinstance(data, Mapping) and data.get("provider") not in (None, ""):
                provider_route = str(data.get("provider"))
            receipt = self._finish_receipt(
                request=normalized,
                started_at=started,
                status="semantic_incomplete" if semantic_miss else "completed",
                attempts=retries + 1,
                remote_request_id=remote_id,
                usage=usage,
                cost_usd=observed_cost,
                provider_route=provider_route,
            )
            safe["remote_cognition"] = receipt
            usage = dict(usage)
            usage.update(
                {
                    "cost_usd": observed_cost,
                    "remote_request_id": remote_id,
                    "latency_ms": receipt.get("latency_ms"),
                    "retries": retries,
                }
            )
            if semantic_miss:
                usage["hawking_semantic_miss"] = semantic_miss
            response = GenerationResponse(
                text=text,
                raw=safe,
                finish_reason=finish,
                usage=usage,
                provider="openrouter",
                request_id=normalized.request_id,
            )
            return response
        except RemoteCognitionError as exc:
            observed_error_cost = None
            if isinstance(exc.details, Mapping):
                candidate = exc.details.get("observed_cost_usd")
                try:
                    observed_error_cost = float(candidate) if candidate is not None else None
                except (TypeError, ValueError):
                    observed_error_cost = None
            self._finish_receipt(
                request=normalized,
                started_at=started,
                status="cancelled" if isinstance(exc, RemoteCancelledError) else "failed",
                attempts=max(1, int(getattr(self, "_last_attempts", 1) or 1)),
                remote_request_id=exc.remote_request_id or getattr(self, "_last_remote_request_id", None),
                cost_usd=observed_error_cost,
                error=exc,
            )
            raise
        finally:
            self._drop_event(normalized.request_id)

    def _workunit_tool_schemas(self, context: Mapping[str, Any]) -> Tuple[Any, List[str]]:
        """Project Hawking's already-authorized registry into OpenAI tools."""
        registry = context.get("tool_registry")
        if registry is None or context.get("hawking_tools") is False:
            return (), []
        try:
            from .chat_tools import openai_schemas
            from .tool_registry import READ_ONLY, RESEARCH

            extra_schemas = {
                str(item.get("function", {}).get("name")): dict(item)
                for item in (context.get("extra_tool_schemas") or [])
                if isinstance(item, Mapping)
                and isinstance(item.get("function"), Mapping)
                and item.get("function", {}).get("name")
            }

            write_authority = bool(
                context.get("tool_write_authority") is True
                and context.get("mutation_lock_held") is True
            )
            # Some Hawking-owned orchestration doors are intentionally not
            # registry mutations.  They are injected only by the daemon's
            # explicit worker loop and must name themselves in this separate
            # allow-set; ordinary callers cannot smuggle extra schemas into a
            # read-only projection.
            extra_allowed = {
                str(item).strip()
                for item in (context.get("extra_tool_names") or [])
                if str(item).strip()
            }
            requested = context.get("allowed_tool_names")
            if requested is None:
                # Preserve every existing consolidated discovery door. Add the
                # already-registered first-order leaves alongside it so models
                # that support typed function calls can choose an exact
                # contract (fs.search) instead of guessing against a union
                # contract (fs + op=search). Nothing previously projected is
                # removed; second-order aliases such as filesystem.search stay
                # hidden exactly as they were by the public discovery surface.
                requested = [
                    item.get("name")
                    for item in registry.discover()
                    if isinstance(item, Mapping)
                ]
                tools = getattr(registry, "_tools", {})
                if isinstance(tools, Mapping):
                    consolidated = {
                        str(spec.alias_of)
                        for spec in tools.values()
                        if getattr(spec, "alias_of", None)
                        and getattr(tools.get(spec.alias_of), "alias_of", None) is None
                    }
                    for name, spec in tools.items():
                        alias_of = getattr(spec, "alias_of", None)
                        if alias_of and str(alias_of) in consolidated:
                            if str(name) not in requested:
                                requested.append(str(name))
                else:
                    requested = [
                        item.get("name")
                        for item in registry.discover()
                        if isinstance(item, Mapping)
                    ]
            allowed: List[str] = []
            for name in requested if isinstance(requested, (list, tuple, set, frozenset)) else ():
                spec = registry.get(str(name))
                if spec is None:
                    if str(name) in extra_schemas and (
                        write_authority or str(name) in extra_allowed
                    ):
                        allowed.append(str(name))
                    continue
                mutation = str(getattr(spec, "mutation", ""))
                if mutation not in {READ_ONLY, RESEARCH} and not write_authority:
                    continue
                allowed.append(str(name))
            # A typed MutationProposal is parsed into the canonical
            # ``repo.edit`` builder call by Hawking, not executed directly by
            # the provider.  Keep that internal bridge in the session menu
            # whenever the daemon supplied its proposal executor; otherwise
            # parse_calls recognizes the proposal but rejects it as an
            # unoffered tool before the canonical mutation gate sees it.
            if write_authority and callable(context.get("mutation_proposal_executor")):
                if "repo.edit" not in allowed:
                    allowed.append("repo.edit")
            schemas = openai_schemas(
                registry,
                names=allowed,
                write=write_authority,
                allowed_names=allowed,
            )
            existing_schema_names = {
                str(item.get("function", {}).get("name"))
                for item in schemas
                if isinstance(item, Mapping)
                and isinstance(item.get("function"), Mapping)
            }
            schemas.extend(
                dict(extra_schemas[name])
                for name in allowed
                if name in extra_schemas and name not in existing_schema_names
            )
            wire_names = _workunit_tool_name_map(allowed)
            for schema in schemas:
                function = schema.get("function") if isinstance(schema, Mapping) else None
                if not isinstance(function, dict):
                    continue
                canonical = str(function.get("name") or "")
                if canonical in wire_names:
                    function["name"] = wire_names[canonical]
            return tuple(schemas), allowed
        except Exception:
            # A projection failure must withhold tools, never widen the request.
            return (), []

    def _generate_with_workunit_tools(
        self,
        request: GenerationRequest,
        context: Mapping[str, Any],
        schemas: Tuple[Any, ...],
        allowed_names: List[str],
    ) -> Tuple[GenerationResponse, List[Dict[str, Any]]]:
        """Run a bounded Hawking-owned tool loop over the same provider."""
        registry = context.get("tool_registry")
        try:
            max_calls = max(0, min(32, int(context.get("max_tool_calls", 8))))
        except (TypeError, ValueError):
            max_calls = 8
        completion_contract = _workunit_completion_contract(context)
        prior_tool_evidence = context.get("prior_tool_evidence")
        continuation_turns = 0
        conversation = [dict(message) for message in request.messages]
        trace: List[Dict[str, Any]] = []
        aggregate: Dict[str, Any] = {}
        model_receipts: List[Dict[str, Any]] = []
        # This is bounded diagnostic provenance for the canonical proposal
        # handoff.  It answers the only question that matters at the seam:
        # did the provider emit a typed proposal, and did Hawking invoke the
        # executor?  It contains neither source contents nor credentials.
        proposal_bridge: List[Dict[str, Any]] = []
        final: Optional[GenerationResponse] = None
        calls_used = 0
        phase_nudges = 0
        wire_names = _workunit_tool_name_map(allowed_names)
        registry_names = {wire: canonical for canonical, wire in wire_names.items()}
        mutation_plan = (
            context.get("mutation_plan")
            if isinstance(context.get("mutation_plan"), Mapping)
            else {}
        )
        proposal_mode = bool(
            isinstance(completion_contract, Mapping)
            and completion_contract.get("mutation_proposal_mode")
        )

        def owed_mutation_tool() -> Optional[str]:
            """Return the next contract door for a bounded mutation continuation."""
            if proposal_mode:
                # The provider is not the mutation authority on this turn. It
                # returns a MutationProposal/envelope; the supplied Hawking
                # executor performs the canonical repo.edit transaction.
                return None
            # The declared completion contract is the canonical mutation
            # authority.  Older callers supplied a mutation-plan kind, but
            # ordinary bounded Build WorkUnits (including P1 serialization
            # repair) legitimately carry only the required tool contract.
            # Keeping the kind gate here allowed a provider to stop after a
            # successful read even though Hawking had already admitted the
            # typed repo.edit door.
            required_contract_tools = {
                str(name).strip()
                for name in (completion_contract or {}).get(
                    "required_successful_tools", []
                )
                if str(name).strip()
            }
            if "repo.edit" not in required_contract_tools:
                return None
            successful = {
                str(name).strip()
                for name in (prior_tool_evidence or {}).get("successful_tools", [])
                if str(name).strip()
            }
            verified = {
                str(name).strip()
                for name in (prior_tool_evidence or {}).get("verified_tools", [])
                if str(name).strip()
            }
            for entry in trace:
                if not isinstance(entry, Mapping) or entry.get("ok") is not True:
                    continue
                tool = str(entry.get("tool") or "").strip()
                value = entry.get("result")
                if isinstance(value, Mapping):
                    value = value.get("value")
                if tool == "repo.edit" and isinstance(value, Mapping):
                    if (
                        str(value.get("status") or "").lower() == "accepted"
                        and value.get("applied") is True
                        and value.get("rolled_back") is not True
                    ):
                        successful.add(tool)
                elif tool == "tests.run" and isinstance(value, Mapping):
                    if value.get("verified") is True and value.get("returncode") == 0:
                        successful.add(tool)
                        verified.add(tool)
                else:
                    successful.add(tool)
            if "repo.edit" not in successful:
                return "repo.edit"
            if "tests.run" not in verified:
                return "tests.run"
            if "git.status" not in successful:
                return "git.status"
            if "git.diff" not in successful:
                return "git.diff"
            return None

        for turn in range(max_calls + 1):
            round_payload = request.to_dict()
            provider_schemas = list(schemas)
            if proposal_mode:
                # Keep read evidence available, but do not advertise or accept
                # the mutating/verification doors on the proposal wire. This
                # prevents a provider's direct tool call from becoming an
                # alternate mutation path while leaving Hawking's own
                # executor responsible for the four canonical evidence edges.
                blocked_wire_names = {
                    wire_names.get(name, name)
                    for name in {
                        "repo.edit", "tests.run", "git.status", "git.diff",
                    }
                }
                provider_schemas = [
                    schema for schema in provider_schemas
                    if not (
                        isinstance(schema, Mapping)
                        and isinstance(schema.get("function"), Mapping)
                        and str(schema["function"].get("name") or "") in blocked_wire_names
                    )
                ]
            round_payload.update({
                "messages": conversation,
                "tools": provider_schemas,
                "request_id": f"{request.request_id}:tool-{turn}",
            })
            response_format = _structured_result_response_format(
                trace, completion_contract, prior_tool_evidence,
            )
            if response_format is not None:
                round_payload["response_format"] = response_format
            terminal_read_only_packet = _read_only_evidence_is_ready_for_terminal_packet(
                trace, completion_contract, prior_tool_evidence,
            )
            if terminal_read_only_packet:
                # The real read observation is already in the conversation.
                # Request the terminal evidence packet without another broad
                # tool menu that can induce faux tool-call JSON.
                round_payload.pop("tools", None)
                round_payload.pop("tool_choice", None)
                terminal_options = dict(round_payload.get("transport_options") or {})
                terminal_options.pop("tool_choice", None)
                round_payload["transport_options"] = terminal_options
                round_payload["messages"] = conversation + [{
                    "role": "user",
                    "content": _terminal_research_packet_instruction(),
                }]
                options = dict(round_payload.get("transport_options") or {})
                options["hawking_allow_read_only_prose_result"] = True
                round_payload["transport_options"] = options
                round_payload["max_tokens"] = max(
                    int(round_payload.get("max_tokens") or 0), 1536,
                )
            if isinstance(completion_contract, Mapping) and completion_contract.get("mutation_proposal_mode"):
                # A mutation proposal is still provider data, never authority,
                # but an HTTP-200 prose response is not useful data.  Require
                # a JSON object at this boundary so the semantic gate rejects
                # empty/prose success immediately and the normalizer can
                # accept either MUTATION_PROPOSED JSON or the legacy explicit
                # HAWKING_PATCH_V1/HAWKING_EDIT_V1 envelope when a route does
                # not honor JSON mode.
                round_payload["response_format"] = {"type": "json_object"}
                # Patch synthesis is a bounded serialization step, not an
                # open-ended reasoning turn.  Prevent the provider from
                # consuming the entire small decode budget in hidden thought
                # and returning an empty visible payload.
                if self.model_id not in MODELS_REQUIRING_PROVIDER_REASONING:
                    round_payload["reasoning"] = {"effort": "none"}
                else:
                    # A caller may have carried this optional optimization
                    # from a prior route. Never impose it on a model whose
                    # upstream contract says reasoning is mandatory.
                    round_payload.pop("reasoning", None)
                round_payload["max_tokens"] = max(int(round_payload.get("max_tokens") or 0), 1536)
                # Do not leave a provider in an open-ended read loop after the
                # admitted orientation budget is spent.  The next round is a
                # serialization boundary: remove the read menu and require a
                # typed proposal.  This keeps repo mutation Hawking-owned while
                # preventing repeated fs.read/search calls from ending as
                # prose-only MUTATION_PAYLOAD_MISSING outcomes.
                try:
                    read_limit = int(
                        completion_contract.get("read_only_tool_call_limit") or 0
                    )
                except (TypeError, ValueError):
                    read_limit = 0
                if read_limit > 0 and calls_used >= read_limit and not _tool_trace_completion(
                    trace, None, completion_contract, prior_tool_evidence
                ).get("complete"):
                    round_payload.pop("tools", None)
                    round_payload.pop("tool_choice", None)
                    # GenerationRequest stores OpenAI extension fields under
                    # transport_options.  Clearing only the top-level wire
                    # key left the named orientation read mandatory after the
                    # tool menu was deliberately closed, so a valid JSON
                    # mutation proposal was rejected as a missing tool call.
                    proposal_options = dict(round_payload.get("transport_options") or {})
                    proposal_options.pop("tool_choice", None)
                    round_payload["transport_options"] = proposal_options
                    round_payload["messages"] = conversation + [{
                        "role": "user",
                        "content": _mutation_proposal_serialization_instruction(
                            completion_contract
                        ),
                    }]
            forced_tool = owed_mutation_tool()
            if (
                forced_tool is None
                and proposal_mode
                and context.get("require_initial_read") is True
                and calls_used == 0
            ):
                prior_successful = {
                    str(name).strip()
                    for name in (prior_tool_evidence or {}).get(
                        "successful_tools", []
                    )
                    if str(name).strip()
                } if isinstance(prior_tool_evidence, Mapping) else set()
                if not prior_successful.intersection({
                    "fs.read", "filesystem.read", "fs.search", "filesystem.search",
                }):
                    for candidate in ("fs.read", "filesystem.read"):
                        if candidate in allowed_names:
                            forced_tool = candidate
                            break
            if forced_tool is None and not proposal_mode and not terminal_read_only_packet:
                forced_tool = _next_bounded_mutation_tool(
                    trace,
                    completion_contract,
                    prior_tool_evidence,
                    context,
                    calls_used=calls_used,
                )
            if (
                forced_tool is None
                and calls_used == 0
                and isinstance(completion_contract, Mapping)
            ):
                # Read-only research contracts may name a first evidence door.
                # Enforce it at the provider boundary so a prose opening cannot
                # consume the bounded route before Hawking observes source.
                candidate = str(
                    completion_contract.get("required_first_tool") or ""
                ).strip()
                if candidate in allowed_names:
                    forced_tool = candidate
            if forced_tool and forced_tool in allowed_names:
                wire_tool = wire_names.get(forced_tool, forced_tool)
                # Some provider routes have accepted an exact tool_choice but
                # still selected a different function from a large menu. Once
                # Hawking has crossed the bounded orientation phase, remove
                # that ambiguity as well: the next request contains only the
                # already-admitted owed door.
                round_payload["tools"] = [
                    schema for schema in schemas
                    if isinstance(schema, Mapping)
                    and isinstance(schema.get("function"), Mapping)
                    and str(schema["function"].get("name") or "") == wire_tool
                ]
                round_payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": wire_tool},
                }
                # Preserve the exact named choice on every qualified route.
                # Kimi's generic ``required`` fallback may choose an
                # unrelated read tool, which defeats an explicit focused-file
                # contract even though the route can honor the named call.
                if forced_tool in {
                    "fs.search", "filesystem.search", "source.search",
                    "fs.read", "filesystem.read",
                }:
                    # The live qualified Kimi and DeepSeek routes require
                    # the same JSON response constraint alongside a named
                    # read tool. Without it they can emit HTTP 200 prose,
                    # which the semantic gate correctly rejects but which
                    # wastes an entire bounded WorkUnit turn.
                    round_payload["response_format"] = {"type": "json_object"}
                    # The first orientation call only needs compact typed
                    # arguments/results. Keep it independent from the larger
                    # mutation decode budget so a provider cannot spend the
                    # bounded turn narrating before it reaches the read door.
                    try:
                        round_payload["max_tokens"] = min(
                            int(round_payload.get("max_tokens") or 256), 256
                        )
                    except (TypeError, ValueError):
                        round_payload["max_tokens"] = 256
            response = self.generate(
                GenerationRequest.from_mapping(round_payload),
                timeout=context.get("timeout"),
            )
            final = response
            if isinstance(response.raw, Mapping) and isinstance(
                response.raw.get("remote_cognition"), Mapping
            ):
                model_receipts.append(dict(response.raw["remote_cognition"]))
            for key, value in (response.usage or {}).items():
                if key in {"prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"}:
                    try:
                        aggregate[key] = float(aggregate.get(key, 0.0)) + float(value or 0.0)
                    except (TypeError, ValueError):
                        continue
                elif key not in aggregate:
                    aggregate[key] = value
            tool_calls = _response_tool_calls(
                response,
                fallback_read_tool=(
                    forced_tool if forced_tool in {
                        "fs.read", "filesystem.read", "source.read",
                        "fs.search", "filesystem.search", "source.search",
                    } else ""
                ),
            )
            proposal_text = _response_proposal_text(response)
            proposal = (
                _mutation_proposal_from_text(proposal_text)
                if proposal_mode else None
            )
            if proposal_mode:
                proposal_bridge.append({
                    "turn": turn,
                    "typed_proposal": proposal is not None,
                    "response_chars": len(proposal_text),
                    "has_tool_calls": bool(tool_calls),
                })
            read_only_proposal_calls = _read_only_calls_from_proposal(
                proposal,
                allowed_names=allowed_names,
                successful_tools={
                    str(entry.get("tool") or "").strip()
                    for entry in trace
                    if isinstance(entry, Mapping) and entry.get("ok") is True
                },
            ) if proposal_mode else []
            try:
                read_limit = int(
                    (completion_contract or {}).get("read_only_tool_call_limit") or 0
                )
            except (TypeError, ValueError):
                read_limit = 0
            if (
                read_only_proposal_calls
                and read_limit > calls_used
                and (registry is not None or callable(context.get("tool_dispatch")))
            ):
                # Treat a read-only proposed operation as a request for the
                # already-admitted Hawking read surface, never as a mutation.
                # Execute at most the remaining bounded orientation budget;
                # the following turn is tool-free patch synthesis.
                dispatcher = context.get("tool_dispatch")
                remaining = max(0, read_limit - calls_used)
                for index, (name, arguments) in enumerate(read_only_proposal_calls[:remaining]):
                    wire_name = wire_names.get(name, name)
                    call_id = f"proposal_read_{uuid.uuid4().hex[:12]}_{index}"
                    tool_result = (
                        dispatcher(name, arguments)
                        if callable(dispatcher)
                        else registry.invoke(name, arguments)
                    )
                    observation = tool_result.to_dict() if hasattr(tool_result, "to_dict") else {
                        "ok": bool(getattr(tool_result, "ok", False)),
                        "value": getattr(tool_result, "value", None),
                        "error": getattr(tool_result, "error", None),
                    }
                    trace.append({
                        "tool": name,
                        "wire_tool": wire_name,
                        "arguments": _safe_value(arguments),
                        "dispatched": True,
                        "ok": bool(observation.get("ok")),
                        "result": _safe_value(observation),
                        "recovered_from": "read_only_mutation_proposal",
                    })
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": wire_name,
                        "content": json.dumps(_safe_value(observation), default=str),
                    })
                    calls_used += 1
                final = response
                continue
            # Some OpenAI-compatible routes retain a tool-call envelope from
            # the preceding bounded read while placing the complete typed
            # proposal in assistant content. A proposal is Hawking-owned data;
            # recognize it before the direct-call branch so the valid payload
            # is not discarded as provider theater.
            if not tool_calls or proposal is not None:
                proposal_executor = context.get("mutation_proposal_executor")
                if proposal is not None and callable(proposal_executor):
                    proposal_bridge[-1]["executor_available"] = True
                    compact_intent = (
                        proposal_mode
                        and isinstance(proposal, Mapping)
                        and str(proposal.get("s") or proposal.get("status") or "").upper()
                        == "MUTATE"
                        and bool(proposal.get("anchor"))
                        and "body" in proposal
                    )
                    legacy_literal = (
                        isinstance(proposal, Mapping)
                        and isinstance(proposal.get("operations"), list)
                        and any(
                            isinstance(row, Mapping)
                            and any(key in row for key in ("old_text", "new_text", "old_lines", "new_lines"))
                            for row in proposal.get("operations", [])
                        )
                    )
                    compact_contract = (
                        isinstance(proposal, Mapping)
                        and isinstance(proposal.get("completion_contract"), Mapping)
                        and str(
                            proposal.get("completion_contract", {}).get("compact_worker_schema") or ""
                        ) == "hawking.compact_worker.v1"
                    )
                    if compact_contract and legacy_literal and not compact_intent:
                        execution = {
                            "status": "rejected",
                            "applied": False,
                            "reason": "COMPACT_MUTATION_INTENT_REQUIRED",
                            "failure_class": "MUTATION_PAYLOAD_MISSING",
                        }
                    else:
                        execution = proposal_executor(proposal)
                    proposal_bridge[-1]["executor_invoked"] = True
                    if isinstance(execution, Mapping):
                        proposal_bridge[-1]["execution_status"] = str(
                            execution.get("status") or ""
                        )
                    execution_trace = (
                        execution.get("post_mutation_trace")
                        if isinstance(execution, Mapping)
                        else None
                    )
                    if not isinstance(execution_trace, list) or not execution_trace:
                        # Keep a canonical repo.edit-shaped receipt even when
                        # a direct test caller supplies only the narrow
                        # proposal executor. The result is still Hawking's
                        # validation output; provider prose never fills it.
                        execution_value = dict(execution) if isinstance(execution, Mapping) else {}
                        # ``execute_mutation_proposal`` historically returned
                        # the engine packet without duplicating its applied
                        # flags at the adapter root.  The fallback receipt
                        # must still satisfy the same machine-owned predicate
                        # as the full post-mutation trace; otherwise an
                        # already-applied HTTP-200 proposal is needlessly
                        # continued and eventually dead-lettered.
                        execution_value.setdefault(
                            "applied",
                            bool(execution_value.get("status") or "")
                            in {"accepted", "unproven"},
                        )
                        execution_value.setdefault("rolled_back", False)
                        # Preserve the compatibility meaning of legacy
                        # adapters that proved the write landed but used a
                        # non-canonical status label.  The receipt is still
                        # explicitly accepted as an effect until focused tests
                        # verify the tranche; the separate tests.run gate
                        # still prevents implementation acceptance when tests
                        # are required. This keeps an applied mutation from
                        # being misclassified as a missing repo.edit edge.
                        if (
                            execution_value.get("applied") is True
                            and execution_value.get("rolled_back") is not True
                            and str(execution_value.get("status") or "").lower()
                            not in {"accepted", "unproven"}
                        ):
                            execution_value["status"] = "accepted"
                        execution_trace = [{
                            "tool": "repo.edit",
                            "dispatched": True,
                            "ok": str(execution_value.get("status") or "").lower()
                               in {"accepted", "unproven"},
                            "result": {"ok": True, "value": _safe_value(execution_value)},
                        }]
                    else:
                        # A legacy executor may return only the follow-up
                        # verification rows (for example git.status/diff)
                        # while the mutation itself is recorded in the nested
                        # engine packet.  Never let that partial projection
                        # hide an actually applied repo.edit from the owner.
                        has_repo_edit = any(
                            isinstance(row, Mapping)
                            and str(row.get("tool") or "") == "repo.edit"
                            and row.get("ok") is True
                            for row in execution_trace
                        )
                        execution_status = str(
                            (execution or {}).get("status") or ""
                        ).lower()
                        execution_applied = (execution or {}).get("applied")
                        execution_rolled_back = (execution or {}).get("rolled_back")
                        if isinstance(execution, Mapping) and isinstance(
                            execution.get("execution"), Mapping
                        ):
                            inner_execution = execution["execution"]
                            if execution_applied is None:
                                execution_applied = inner_execution.get("applied")
                            if execution_rolled_back is None:
                                execution_rolled_back = inner_execution.get("rolled_back")
                        applied_without_rollback = (
                            execution_applied is True
                            and execution_rolled_back is not True
                        )
                        if not has_repo_edit and (
                            execution_status in {"accepted", "unproven"}
                            or applied_without_rollback
                        ):
                            execution_value = dict(execution)
                            execution_value.setdefault("applied", True)
                            execution_value.setdefault("rolled_back", False)
                            execution_value.setdefault("status", "unproven")
                            execution_trace.insert(0, {
                                "tool": "repo.edit",
                                "dispatched": True,
                                "ok": True,
                                "result": {
                                    "ok": True,
                                    "value": _safe_value(execution_value),
                                },
                            })
                    trace.extend(
                        dict(row) for row in execution_trace
                        if isinstance(row, Mapping)
                    )
                    trace.append({
                        "tool": "mutation.proposal",
                        "dispatched": True,
                        "ok": str(execution.get("status") or "").lower() in {"accepted", "unproven"},
                        "result": {"ok": True, "value": _safe_value(execution)},
                    })
                    # A typed proposal is complete when Hawking has already
                    # performed the canonical mutation and verification chain.
                    # Do not spend a second remote turn asking the provider to
                    # narrate that fact: a HTTP-200 prose response or a
                    # transport timeout there used to discard an otherwise
                    # accepted WorkUnit.  Keep the repair continuation only
                    # for proposals whose Hawking-owned evidence is still
                    # incomplete.
                    proposal_completion = _tool_trace_completion(
                        trace, response, completion_contract, prior_tool_evidence,
                    )
                    if proposal_completion.get("complete") is True:
                        final = response
                        break
                    conversation.extend([
                        {"role": "assistant", "content": proposal_text},
                        {"role": "user", "content": (
                            (
                                "The previous proposal contained only read/inspect "
                                "observations; Hawking discarded those because they "
                                "are not mutation effects. Return one bounded effect "
                                "operation now (replace/create/insert/append), with "
                                "an exact observed anchor and the focused test. Do "
                                "not emit read, fs.read, search, status, or diff "
                                "entries in operations.\n"
                                if str((execution or {}).get("reason") or "")
                                == "EMPTY_MUTATION_PROPOSAL" else
                                (
                                    "The previous payload was rejected before execution "
                                    "because it omitted the required regression-test "
                                    "operation. Return one real source operation and "
                                    "one real operation targeting one of these test "
                                    "paths: "
                                    + json.dumps(
                                        _safe_value((execution or {}).get("test_mutation_paths") or [])
                                    )
                                    + ". Include the focused test invocation and do not "
                                    "repeat the source-only payload.\n"
                                    if "MISSING_REQUIRED_TEST_OPERATION" in str(
                                        (execution or {}).get("reason") or ""
                                ) else
                    (
                        "The previous compact proposal was not applied. Return one "
                        "executable repair using only {s:MUTATE,anchor,op:replace,body,tests}. "
                        "Use one of these Hawking-issued anchors from the successful fs.read: "
                        + json.dumps([
                            str(
                                ((row.get("result") or {}).get("value") or {}).get(
                                    "source_anchor", {}
                                ).get("anchor")
                            )
                            for row in trace
                            if isinstance(row, Mapping)
                            and isinstance(row.get("result"), Mapping)
                            and isinstance((row.get("result") or {}).get("value"), Mapping)
                            and isinstance(
                                ((row.get("result") or {}).get("value") or {}).get(
                                    "source_anchor"
                                ), Mapping
                            )
                            and str(
                                (((row.get("result") or {}).get("value") or {}).get(
                                    "source_anchor", {}
                                ) or {}).get("anchor") or ""
                            ).strip()
                        ])
                        + ". Return exactly one of those anchors. Never return "
                        "path, old_text, old_lines, new_text, digests, leases, or authority. "
                                        "If the anchor is unavailable, return {s:BLOCKED,reason:MISSING_SOURCE_ANCHOR}.\n"
                                        if proposal_mode and not bool(
                                            (completion_contract or {}).get("require_regression_test_mutation")
                                        ) else
                                        "The previous proposal was not applied. In compact "
                                        "proposal mode return exactly one JSON mutation intent: "
                                        "{s:MUTATE,anchor,op:replace,body,tests}. Do not return "
                                        "old_text, new_text, path, digests, leases, or prose.\n"
                                    )
                                )
                            ) +
                            json.dumps(_safe_value(execution), default=str)[:12000]
                        )},
                    ])
                    continuation_turns += 1
                    if continuation_turns <= completion_contract.get("max_continuation_turns", 1):
                        continue
                elif proposal is not None:
                    proposal_bridge[-1]["executor_available"] = False
                completion = _tool_trace_completion(
                    trace, response, completion_contract, prior_tool_evidence
                )
                if (
                    completion_contract
                    and completion["unmet"]
                    and calls_used < max_calls
                    and continuation_turns
                    < completion_contract["max_continuation_turns"]
                ):
                    # Preserve the provider's final turn as context, but do
                    # not treat it as acceptance evidence.  This is the
                    # measured DeepSeek gap: useful archaeology followed by a
                    # prose stop before the write/test boundary.
                    conversation.append({
                        "role": "assistant",
                        "content": proposal_text,
                    })
                    unmet = ", ".join(completion["unmet"])
                    reminder = completion_contract.get("reminder") or (
                        "Continue the same bounded WorkUnit. Hawking has not "
                        "accepted the unit because the following evidence is "
                        "still missing: " + unmet + "."
                    )
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"{reminder}\n"
                            f"Still owed: {unmet}. Use only the projected "
                            "Hawking tools, perform no fabricated calls, and "
                            "return the requested structured result only after "
                            "the evidence is real."
                        ),
                    })
                    continuation_turns += 1
                    continue
                break
            if calls_used >= max_calls:
                break
            raw_message = (
                response.raw.get("choices", [{}])[0].get("message", {})
                if isinstance(response.raw, Mapping) else {}
            )
            conversation.append({
                "role": "assistant",
                "content": raw_message.get("content")
                if isinstance(raw_message, Mapping) else response.text,
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                wire_name = str(function.get("name") or "")
                name = registry_names.get(wire_name, wire_name)
                call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}")
                raw_arguments = function.get("arguments") or "{}"
                arguments: Dict[str, Any] = {}
                if calls_used >= max_calls:
                    observation = {
                        "ok": False,
                        "error": "tool call budget exhausted",
                        "failure_class": "TOOL_CALL_BUDGET_EXHAUSTED",
                    }
                    trace.append({
                        "tool": name,
                        "wire_tool": wire_name,
                        "arguments": {},
                        "dispatched": False,
                        "ok": False,
                        "error": observation["error"],
                    })
                else:
                    try:
                        parsed_arguments = (
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str) else raw_arguments
                        )
                        if not isinstance(parsed_arguments, Mapping):
                            raise ValueError("tool arguments must be a JSON object")
                        arguments = dict(parsed_arguments)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        observation = {
                            "ok": False,
                            "error": f"invalid tool arguments: {type(exc).__name__}",
                            "failure_class": "INVALID_ARGUMENTS",
                        }
                        trace.append({
                            "tool": name,
                            "wire_tool": wire_name,
                            "arguments": {},
                            "dispatched": False,
                            "ok": False,
                            "error": observation["error"],
                        })
                    else:
                        dispatcher = context.get("tool_dispatch")
                        if proposal_mode and name in {
                            "repo.edit", "tests.run", "git.status", "git.diff",
                        }:
                            observation = {
                                "ok": False,
                                "error": (
                                    "direct mutation/verification calls are disabled; "
                                    "return a Hawking mutation proposal"
                                ),
                                "failure_class": "MUTATION_PROPOSAL_REQUIRED",
                            }
                            trace.append({
                                "tool": name,
                                "wire_tool": wire_name,
                                "arguments": _safe_value(arguments),
                                "dispatched": False,
                                "ok": False,
                                "error": observation["error"],
                                "failure_class": observation["failure_class"],
                            })
                        elif name not in allowed_names or (
                            registry is None and not callable(dispatcher)
                        ):
                            observation = {
                                "ok": False,
                                "error": "tool is outside the WorkUnit capability projection",
                                "failure_class": "PERMISSION_DENIED",
                            }
                            trace.append({
                                "tool": name,
                                "wire_tool": wire_name,
                                "arguments": _safe_value(arguments),
                                "dispatched": False,
                                "ok": False,
                                "error": observation["error"],
                            })
                        else:
                            tool_result = (
                                dispatcher(name, arguments)
                                if callable(dispatcher)
                                else registry.invoke(name, arguments)
                            )
                            observation = tool_result.to_dict() if hasattr(tool_result, "to_dict") else {
                                "ok": bool(getattr(tool_result, "ok", False)),
                                "value": getattr(tool_result, "value", None),
                                "error": getattr(tool_result, "error", None),
                            }
                            trace.append({
                                "tool": name,
                                "wire_tool": wire_name,
                                "arguments": _safe_value(arguments),
                                "dispatched": True,
                                "ok": bool(observation.get("ok")),
                                "result": _safe_value(observation),
                            })
                    calls_used += 1
                conversation.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": wire_name,
                    "content": json.dumps(_safe_value(observation), default=str),
                })
            if (
                completion_contract
                and completion_contract.get("read_only_tool_call_limit", 0)
                and calls_used >= completion_contract["read_only_tool_call_limit"]
                and phase_nudges < 1
                and not _tool_trace_completion(
                    trace, response, completion_contract, prior_tool_evidence
                )["complete"]
            ):
                required = ", ".join(
                    completion_contract.get("required_successful_tools") or []
                )
                conversation.append({
                    "role": "user",
                    "content": (
                        "Hawking has enough bounded archaeology for this WorkUnit. "
                        "Stop searching and reading unrelated material now. The "
                        "next turn must execute the owed evidence tools, not "
                        "another exploratory call: " + required + "."
                    ),
                })
                phase_nudges += 1
        if final is None:
            raise RemoteCognitionError(
                "WORKUNIT_TOOL_LOOP_EMPTY",
                "Hawking WorkUnit tool loop produced no provider response",
                provider="openrouter",
                model_id=self.model_id,
                request_id=request.request_id,
                recoverable=True,
            )
        final.request_id = request.request_id
        if aggregate:
            for key, value in list(aggregate.items()):
                if isinstance(value, float) and value.is_integer():
                    aggregate[key] = int(value)
            final.usage = aggregate
        if isinstance(final.raw, dict):
            final.raw["hawking_tool_trace"] = _safe_value(trace)
            final.raw["hawking_tool_call_count"] = calls_used
            final.raw["hawking_remote_cognition_calls"] = _safe_value(model_receipts)
            final.raw["hawking_proposal_bridge"] = _safe_value(proposal_bridge)
            final.raw["hawking_tool_admission"] = {
                "write_authority": bool(context.get("tool_write_authority")),
                "repo_edit_offered": (
                    "repo.edit" in allowed_names and not proposal_mode
                ),
                "mutation_proposal_mode": proposal_mode,
                "tool_count": len(allowed_names),
                "prior_successful_count": len(
                    (prior_tool_evidence or {}).get("successful_tools") or []
                ) if isinstance(prior_tool_evidence, Mapping) else 0,
            }
            completion = _tool_trace_completion(
                trace, final, completion_contract, prior_tool_evidence
            )
            final.raw["hawking_completion"] = {
                **completion,
                "required_successful_tools": list(
                    (completion_contract or {}).get("required_successful_tools", [])
                ),
                "continuation_turns": continuation_turns,
                "phase_nudges": phase_nudges,
                "max_continuation_turns": (
                    (completion_contract or {}).get("max_continuation_turns", 0)
                ),
                "calls_used": calls_used,
                "reason": (
                    "evidence_contract_satisfied"
                    if completion["complete"]
                    else "evidence_contract_incomplete"
                ),
            }
        return final, trace

    def generate_stream_with_workunit_tools(
        self,
        request: GenerationRequest,
        context: Mapping[str, Any],
        schemas: Tuple[Any, ...],
        allowed_names: List[str],
    ):
        """Stream a bounded Hawking tool loop without dropping tool schemas.

        ``generate_stream`` is deliberately a transport primitive: it does
        not know which local capability is allowed to answer a tool call.
        H-Web needs both that real incremental transport and the same
        capability boundary used by WorkUnits.  This adapter keeps the
        provider turn/continuation protocol here, while the supplied
        dispatcher remains Hawking's authority.

        The provider's interim text is forwarded as it arrives.  A model that
        elects to call a tool generally emits no prose before the call; if it
        does, that is still a genuine provider delta rather than a fabricated
        progress message.  Tool-call chunks are accumulated until the stream
        ends, then only the admitted dispatcher is invoked.
        """
        registry = context.get("tool_registry")
        try:
            max_calls = max(0, min(32, int(context.get("max_tool_calls", 8))))
        except (TypeError, ValueError):
            max_calls = 8
        completion_contract = _workunit_completion_contract(context)
        prior_tool_evidence = context.get("prior_tool_evidence")
        proposal_mode = bool(
            isinstance(completion_contract, Mapping)
            and completion_contract.get("mutation_proposal_mode")
        )
        conversation = [dict(message) for message in request.messages]
        wire_names = _workunit_tool_name_map(allowed_names)
        registry_names = {wire: canonical for canonical, wire in wire_names.items()}
        trace: List[Dict[str, Any]] = []
        aggregate: Dict[str, Any] = {}
        model_receipts: List[Dict[str, Any]] = []
        calls_used = 0
        last_done: Dict[str, Any] = {}
        continuation_turns = 0

        def add_usage(usage: Any) -> None:
            if not isinstance(usage, Mapping):
                return
            for key, value in usage.items():
                if key in {"prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"}:
                    try:
                        aggregate[key] = float(aggregate.get(key, 0.0)) + float(value or 0.0)
                    except (TypeError, ValueError):
                        continue
                elif key not in aggregate:
                    aggregate[key] = value

        def merge_calls(chunks: Any, records: Dict[str, Dict[str, Any]]) -> None:
            for position, raw in enumerate(chunks or []):
                if not isinstance(raw, Mapping):
                    continue
                index = raw.get("index")
                key = str(index if index is not None else raw.get("id") or position)
                call = records.setdefault(key, {
                    "index": index if index is not None else position,
                    "id": raw.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": raw.get("type") or "function",
                    "function": {"name": "", "arguments": ""},
                })
                if raw.get("id"):
                    call["id"] = str(raw["id"])
                function = raw.get("function") if isinstance(raw.get("function"), Mapping) else {}
                target = call["function"]
                if function.get("name"):
                    target["name"] = str(function["name"])
                arguments = function.get("arguments")
                if arguments not in (None, ""):
                    target["arguments"] = str(target.get("arguments") or "") + str(arguments)

        def dispatch(call: Mapping[str, Any]) -> Dict[str, Any]:
            nonlocal calls_used
            function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
            wire_name = str(function.get("name") or "")
            name = registry_names.get(wire_name, wire_name)
            raw_arguments = function.get("arguments") or "{}"
            arguments: Dict[str, Any] = {}
            if calls_used >= max_calls:
                observation: Dict[str, Any] = {
                    "ok": False,
                    "error": "tool call budget exhausted",
                    "failure_class": "TOOL_CALL_BUDGET_EXHAUSTED",
                }
                trace.append({"tool": name, "wire_tool": wire_name, "arguments": {},
                              "dispatched": False, "ok": False,
                              "error": observation["error"]})
                return observation
            try:
                parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                if not isinstance(parsed, Mapping):
                    raise ValueError("tool arguments must be a JSON object")
                arguments = dict(parsed)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                observation = {
                    "ok": False,
                    "error": f"invalid tool arguments: {type(exc).__name__}",
                    "failure_class": "INVALID_ARGUMENTS",
                }
                trace.append({"tool": name, "wire_tool": wire_name, "arguments": {},
                              "dispatched": False, "ok": False,
                              "error": observation["error"]})
            else:
                dispatcher = context.get("tool_dispatch")
                if proposal_mode and name in {
                    "repo.edit", "tests.run", "git.status", "git.diff",
                }:
                    observation = {
                        "ok": False,
                        "error": (
                            "direct mutation/verification calls are disabled; "
                            "return a Hawking mutation proposal"
                        ),
                        "failure_class": "MUTATION_PROPOSAL_REQUIRED",
                    }
                    trace.append({"tool": name, "wire_tool": wire_name,
                                  "arguments": _safe_value(arguments),
                                  "dispatched": False, "ok": False,
                                  "error": observation["error"],
                                  "failure_class": observation["failure_class"]})
                elif name not in allowed_names or (registry is None and not callable(dispatcher)):
                    observation = {
                        "ok": False,
                        "error": "tool is outside the WorkUnit capability projection",
                        "failure_class": "PERMISSION_DENIED",
                    }
                    trace.append({"tool": name, "wire_tool": wire_name,
                                  "arguments": _safe_value(arguments), "dispatched": False,
                                  "ok": False, "error": observation["error"]})
                else:
                    result = dispatcher(name, arguments) if callable(dispatcher) else registry.invoke(name, arguments)
                    observation = result.to_dict() if hasattr(result, "to_dict") else {
                        "ok": bool(getattr(result, "ok", False)),
                        "value": getattr(result, "value", None),
                        "error": getattr(result, "error", None),
                    }
                    trace.append({"tool": name, "wire_tool": wire_name,
                                  "arguments": _safe_value(arguments), "dispatched": True,
                                  "ok": bool(observation.get("ok")),
                                  "result": _safe_value(observation)})
            calls_used += 1
            return observation

        for turn in range(max_calls + 1):
            payload = request.to_dict()
            provider_schemas = list(schemas)
            if proposal_mode:
                blocked_wire_names = {
                    wire_names.get(name, name)
                    for name in {"repo.edit", "tests.run", "git.status", "git.diff"}
                }
                provider_schemas = [
                    schema for schema in provider_schemas
                    if not (
                        isinstance(schema, Mapping)
                        and isinstance(schema.get("function"), Mapping)
                        and str(schema["function"].get("name") or "") in blocked_wire_names
                    )
                ]
            payload.update({
                "messages": conversation,
                "tools": provider_schemas,
                "request_id": f"{request.request_id}:stream-tool-{turn}",
            })
            terminal_read_only_packet = _read_only_evidence_is_ready_for_terminal_packet(
                trace, completion_contract, prior_tool_evidence,
            )
            if terminal_read_only_packet:
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                payload["response_format"] = {"type": "json_object"}
                payload["messages"] = conversation + [{
                    "role": "user",
                    "content": _terminal_research_packet_instruction(),
                }]
                options = dict(payload.get("transport_options") or {})
                options["hawking_allow_read_only_prose_result"] = True
                payload["transport_options"] = options
                payload["max_tokens"] = max(
                    int(payload.get("max_tokens") or 0), 1536,
                )
            forced_tool = None if proposal_mode else _next_bounded_mutation_tool(
                trace,
                completion_contract,
                prior_tool_evidence,
                context,
                calls_used=calls_used,
            )
            if (
                forced_tool is None
                and calls_used == 0
                and isinstance(completion_contract, Mapping)
                and not terminal_read_only_packet
            ):
                candidate = str(
                    completion_contract.get("required_first_tool") or ""
                ).strip()
                if candidate in allowed_names:
                    forced_tool = candidate
            if forced_tool and forced_tool in allowed_names:
                wire_tool = wire_names.get(forced_tool, forced_tool)
                payload["tools"] = [
                    schema for schema in schemas
                    if isinstance(schema, Mapping)
                    and isinstance(schema.get("function"), Mapping)
                    and str(schema["function"].get("name") or "") == wire_tool
                ]
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": wire_tool},
                }
                if forced_tool in {
                    "fs.search", "filesystem.search", "source.search",
                    "fs.read", "filesystem.read",
                }:
                    payload["response_format"] = {"type": "json_object"}
                    try:
                        payload["max_tokens"] = min(
                            int(payload.get("max_tokens") or 256), 256
                        )
                    except (TypeError, ValueError):
                        payload["max_tokens"] = 256
            streamed_request = GenerationRequest.from_mapping(payload)
            call_records: Dict[str, Dict[str, Any]] = {}
            round_text: List[str] = []
            done: Dict[str, Any] = {}
            for item in self.generate_stream(streamed_request, timeout=context.get("timeout")):
                if item.get("type") == "delta":
                    text = str(item.get("text") or "")
                    if text:
                        round_text.append(text)
                        yield {"type": "delta", "text": text, "tool_calls": None}
                    if item.get("tool_calls"):
                        merge_calls(item.get("tool_calls"), call_records)
                elif item.get("type") == "done":
                    done = dict(item)
            last_done = done
            add_usage(done.get("usage"))
            if isinstance(done.get("receipt"), Mapping):
                model_receipts.append(dict(done["receipt"]))
            calls = [call for _key, call in sorted(
                call_records.items(), key=lambda item: int(item[1].get("index") or 0)
            )]
            proposal_text = "".join(round_text)
            proposal = (
                _mutation_proposal_from_text(proposal_text)
                if proposal_mode else None
            )
            if not calls or proposal is not None:
                if proposal is not None:
                    proposal_executor = context.get("mutation_proposal_executor")
                    if proposal is not None and callable(proposal_executor):
                        execution = proposal_executor(proposal)
                        execution_trace = (
                            execution.get("post_mutation_trace")
                            if isinstance(execution, Mapping) else None
                        )
                        if not isinstance(execution_trace, list) or not execution_trace:
                            execution_trace = [{
                                "tool": "repo.edit",
                                "dispatched": True,
                                "ok": str((execution or {}).get("status") or "").lower()
                                in {"accepted", "unproven"},
                                "result": {"ok": True, "value": _safe_value(execution)},
                            }]
                        trace.extend(
                            dict(row) for row in execution_trace
                            if isinstance(row, Mapping)
                        )
                        trace.append({
                            "tool": "mutation.proposal",
                            "dispatched": True,
                            "ok": str(execution.get("status") or "").lower()
                            in {"accepted", "unproven"},
                            "result": {"ok": True, "value": _safe_value(execution)},
                        })
                        proposal_completion = _tool_trace_completion(
                            trace, None, completion_contract, prior_tool_evidence,
                        )
                        if proposal_completion.get("complete") is True:
                            # The stream has already emitted the provider
                            # proposal; close with Hawking's durable evidence
                            # instead of requiring a second provider decode.
                            yield {
                                "type": "done",
                                "finish_reason": "hawking_mutation_complete",
                                "usage": aggregate or dict(done.get("usage") or {}),
                                "receipt": done.get("receipt"),
                                "tool_trace": _safe_value(trace),
                                "tool_admission": {
                                    "write_authority": bool(context.get("tool_write_authority")),
                                    "repo_edit_offered": False,
                                    "mutation_proposal_mode": True,
                                    "tool_count": len(allowed_names),
                                    "effect_ceiling": "MUTATE" if context.get("tool_write_authority") else "READ",
                                },
                                "remote_cognition_calls": _safe_value(model_receipts),
                                "hawking_completion": {
                                    **proposal_completion,
                                    "required_successful_tools": list(
                                        (completion_contract or {}).get("required_successful_tools", [])
                                    ),
                                    "continuation_turns": continuation_turns,
                                    "calls_used": calls_used,
                                },
                                "calls_used": calls_used,
                            }
                            return
                        yield {"type": "tool", "trace": _safe_value(trace[-1])}
                        conversation.extend([
                            {"role": "assistant", "content": proposal_text},
                            {"role": "user", "content": (
                                (
                                    "The previous proposal contained only read/inspect "
                                    "observations; Hawking discarded those because they "
                                    "are not mutation effects. Return one bounded effect "
                                    "operation now (replace/create/insert/append), with "
                                    "an exact observed anchor and the focused test. Do "
                                    "not emit read, fs.read, search, status, or diff "
                                    "entries in operations.\n"
                                    if str((execution or {}).get("reason") or "")
                                    == "EMPTY_MUTATION_PROPOSAL" else
                                    (
                                        "The previous proposal was not applied. In compact "
                                        "proposal mode return exactly one JSON mutation intent: "
                                        "{s:MUTATE,anchor,op:replace,body,tests}. Do not return "
                                        "old_text, new_text, path, digests, leases, or prose.\n"
                                        if proposal_mode else
                                        "The previous proposal was not applied. Return one "
                                        "executable repair: op=replace with path, exact old_text, "
                                        "and complete literal new_text (or create with nonempty "
                                        "new_text / literal new_lines). change, kind, source_edit, "
                                        "anchor-only, and prose instructions are invalid.\n"
                                    )
                                ) +
                                json.dumps(_safe_value(execution), default=str)[:12000]
                            )},
                        ])
                        continuation_turns += 1
                        if continuation_turns <= completion_contract.get("max_continuation_turns", 1):
                            continue
                completion_probe = GenerationResponse(
                    text="".join(round_text),
                    raw={"choices": [{"message": {
                        "content": "".join(round_text),
                    }}]},
                    usage=dict(aggregate or done.get("usage") or {}),
                    provider=self.provider_name,
                    request_id=request.request_id,
                    finish_reason=str(done.get("finish_reason") or "stop"),
                )
                completion = _tool_trace_completion(
                    trace, completion_probe, completion_contract,
                    prior_tool_evidence,
                )
                if (
                    completion_contract
                    and completion["unmet"]
                    and calls_used < max_calls
                    and continuation_turns
                    < completion_contract["max_continuation_turns"]
                ):
                    conversation.append({
                        "role": "assistant",
                        "content": "".join(round_text),
                    })
                    unmet = ", ".join(completion["unmet"])
                    reminder = completion_contract.get("reminder") or (
                        "Continue the same bounded Hawking WorkUnit. Missing evidence: "
                        + unmet
                    )
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"{reminder}\nStill owed: {unmet}. Use only the "
                            "projected Hawking tools; do not claim completion "
                            "without their observations."
                        ),
                    })
                    continuation_turns += 1
                    continue
                for key, value in list(aggregate.items()):
                    if isinstance(value, float) and value.is_integer():
                        aggregate[key] = int(value)
                yield {
                    "type": "done",
                    "finish_reason": str(done.get("finish_reason") or "stop"),
                    "usage": aggregate or dict(done.get("usage") or {}),
                    "receipt": done.get("receipt"),
                    "tool_trace": _safe_value(trace),
                    "tool_admission": {
                        "write_authority": bool(context.get("tool_write_authority")),
                        "repo_edit_offered": (
                            "repo.edit" in allowed_names and not proposal_mode
                        ),
                        "mutation_proposal_mode": proposal_mode,
                        "tool_count": len(allowed_names),
                        "effect_ceiling": "MUTATE" if context.get("tool_write_authority") else "READ",
                    },
                    "remote_cognition_calls": _safe_value(model_receipts),
                    "hawking_completion": {
                        **completion,
                        "required_successful_tools": list(
                            (completion_contract or {}).get("required_successful_tools", [])
                        ),
                        "continuation_turns": continuation_turns,
                        "calls_used": calls_used,
                    },
                    "calls_used": calls_used,
                }
                return
            conversation.append({
                "role": "assistant",
                "content": "".join(round_text) or None,
                "tool_calls": calls,
            })
            for call in calls:
                observation = dispatch(call)
                function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                yield {"type": "tool", "trace": _safe_value(trace[-1])}
                conversation.append({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    "name": str(function.get("name") or ""),
                    "content": json.dumps(_safe_value(observation), default=str),
                })
        final_probe = GenerationResponse(
            text="",
            raw={"choices": [{"message": {"content": ""}}]},
            usage=dict(aggregate or last_done.get("usage") or {}),
            provider=self.provider_name,
            request_id=request.request_id,
            finish_reason="tool_calls",
        )
        completion = _tool_trace_completion(
            trace, final_probe, completion_contract, prior_tool_evidence,
        )
        yield {
            "type": "done",
            "finish_reason": "tool_calls",
            "usage": aggregate or dict(last_done.get("usage") or {}),
            "receipt": last_done.get("receipt"),
            "tool_trace": _safe_value(trace),
            "tool_admission": {
                "write_authority": bool(context.get("tool_write_authority")),
                "repo_edit_offered": (
                    "repo.edit" in allowed_names and not proposal_mode
                ),
                "mutation_proposal_mode": proposal_mode,
                "tool_count": len(allowed_names),
                "effect_ceiling": "MUTATE" if context.get("tool_write_authority") else "READ",
            },
            "remote_cognition_calls": _safe_value(model_receipts),
            "hawking_completion": {
                **completion,
                "required_successful_tools": list(
                    (completion_contract or {}).get("required_successful_tools", [])
                ),
                "continuation_turns": continuation_turns,
                "calls_used": calls_used,
            },
            "calls_used": calls_used,
        }

    def execute_workunit(self, wu: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one bounded cognition unit and return claims for central validation."""
        packet = _workunit_packet(wu, context)
        prompt = str(context.get("prompt") or getattr(wu, "description", ""))
        instruction = (
            "Return a single JSON object with bounded fields: observations, "
            "operations, tests, questions, status, and content. Operations "
            "are claims for Hawking's existing mutation/verifier boundary; "
            "do not perform or claim acceptance of them.\n\n"
            "WORKUNIT CONTRACT:\n"
            + json.dumps(packet, sort_keys=True, ensure_ascii=False)
        )
        metadata: Dict[str, Any] = {
            "unit_id": getattr(wu, "id", None),
            "workunit_packet": packet,
        }
        authorization = context.get("remote_cost_authorization") or context.get("cost")
        if isinstance(authorization, Mapping):
            metadata["cost_authorization"] = dict(authorization)
        for key in ("is_cancelled", "cancel_event", "estimated_cost_usd"):
            if key in context:
                metadata[key] = context[key]
        tool_schemas, allowed_tool_names = self._workunit_tool_schemas(context)
        request = GenerationRequest(
            messages=(
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ),
            model=self.model_id,
            max_tokens=int(context.get("max_tokens") or os.environ.get("HAWKING_MODEL_TOKENS", "1024")),
            temperature=(float(context["temperature"]) if context.get("temperature") is not None else None),
            response_schema=(dict(context["response_schema"]) if isinstance(context.get("response_schema"), Mapping) else None),
            tools=tuple(tool_schemas),
            metadata=metadata,
            request_id=f"wu-{getattr(wu, 'id', 'work')}-{uuid.uuid4().hex[:12]}",
        )
        started = time.time()
        if tool_schemas:
            response, tool_trace = self._generate_with_workunit_tools(
                request,
                context,
                tuple(tool_schemas),
                allowed_tool_names,
            )
        else:
            response = self.generate(request, timeout=context.get("timeout"))
            tool_trace = []
        structured = _structured_text(response.text)
        observations = structured.get("observations")
        if not isinstance(observations, list):
            observations = [response.text] if response.text else []
        operations = structured.get("operations")
        tests = structured.get("tests")
        questions = structured.get("questions")
        completion = (
            response.raw.get("hawking_completion")
            if isinstance(response.raw, Mapping)
            else None
        )
        return {
            "status": "observed",
            "backend": "openrouter",
            "provider": "openrouter",
            "model": self.model_id,
            "content": response.text,
            "observations": _safe_value(observations),
            "operations": _safe_value(operations if isinstance(operations, list) else []),
            "tests": _safe_value(tests if isinstance(tests, list) else []),
            "questions": _safe_value(questions if isinstance(questions, list) else []),
            "usage": _safe_mapping(response.usage),
            "cost": response.usage.get("cost_usd"),
            "wall_s": max(0.0, time.time() - started),
            "request_id": response.request_id,
            "tool_trace": _safe_value(tool_trace),
            "tool_call_count": len(tool_trace),
            "completion": _safe_value(completion),
            "remote_cognition": _safe_value(
                response.raw.get("remote_cognition")
                if isinstance(response.raw, Mapping)
                else self.last_receipt
            ),
            "provider_response": response.to_dict(),
            "validation": {
                "ok": False,
                "reason": "REMOTE_COGNITION_REQUIRES_CENTRAL_VALIDATION",
            },
        }

    def _catalog_url(self) -> str:
        parsed = urllib.parse.urlparse(self.endpoint)
        path = parsed.path
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")]
        elif path.endswith("/completions"):
            path = path[: -len("/completions")]
        path = path.rstrip("/")
        if not path.endswith("/v1"):
            path += "/v1"
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path + "/models", "", "", ""))

    def list_models(self, *, force: bool = False, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Explicitly fetch a short-lived catalog; selection never requires it."""
        now = time.time()
        if not force and self._catalog is not None and now - self._catalog[0] < self.catalog_ttl_s:
            return _copy(self._catalog[1])
        data, _headers, _status, _retries, _remote_id = self._request_json(
            url=self._catalog_url(),
            method="GET",
            body=None,
            request=None,
            timeout=float(timeout if timeout is not None else self.timeout),
            event=None,
        )
        values = data.get("data") if isinstance(data, Mapping) else []
        models = [_safe_mapping(item) for item in values] if isinstance(values, list) else []
        self._catalog = (now, models)
        return _copy(models)




class HawkingNativeProvider(_RemoteProviderBase):
    """Hawking's native cognition provider under the common provider contract.

    This absorbs the useful bounded-worker semantics that previously lived
    behind the Grok adapter: one WorkUnit packet in, one structured observation
    out, with Mission/verifier/MutationLock authority remaining centralized.
    It never launches a vendor bot or creates a second scheduler.
    """

    # This is the in-process Hawking provider under the shared contract.  It
    # must not be mistaken for a remote transport merely because the common
    # adapter base also hosts OpenRouter.
    is_remote_cognition = False

    def __init__(
        self,
        model_id: str = "native-resident",
        *,
        workspace: Optional[Path] = None,
        backend: Any = None,
        engine: Any = None,
        model_path: Optional[str] = None,
        remote_slot: Optional[str] = None,
        cost_policy: Optional[RemoteCostPolicy] = None,
    ) -> None:
        super().__init__(
            provider="hawking",
            model_id=str(model_id or "native-resident"),
            workspace=workspace,
            remote_slot=remote_slot,
            cost_policy=cost_policy or RemoteCostPolicy(max_cost_usd=0.0),
        )
        self._backend = backend
        self._engine = engine
        self._model_path = str(model_path or "").strip() or None
        self._native_request_ids: Dict[str, Any] = {}

    def _resolve_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._engine is not None:
            return self._engine
        try:
            from .backends import NativeRuntimeBackend
            from .hawking_native import config_for_model_path

            selected = self._model_path or os.environ.get("HAWKING_NATIVE_CONFIG") or os.environ.get("HAWKING_MODEL_PATH")
            if not selected:
                selected = str(Path(__file__).resolve().with_name("hawking-native.sealed-3.14.json"))
            config_for_model_path(selected)
            self._backend = NativeRuntimeBackend(model_path=selected)
            return self._backend
        except Exception as exc:
            raise RemoteCognitionError(
                "HAWKING_NATIVE_UNAVAILABLE",
                f"Hawking native profile is unavailable: {type(exc).__name__}: {exc}",
                provider="hawking",
                model_id=self.model_id,
                recoverable=True,
            ) from exc

    def identity(self) -> Dict[str, Any]:
        identity: Dict[str, Any] = {}
        backend = self._backend or self._engine
        identity_fn = getattr(backend, "identity", None) if backend is not None else None
        if callable(identity_fn):
            try:
                candidate = identity_fn()
                if isinstance(candidate, Mapping):
                    identity = dict(candidate)
            except Exception:
                identity = {}
        identity.update({
            "schema": REMOTE_IDENTITY_SCHEMA,
            "provider": "hawking",
            "model_id": str(identity.get("model_id") or self.model_id),
            "backend": "hawking-native",
            "runtime": "hawking-native",
            "protocol": str(identity.get("protocol") or "hawking-native"),
            "transport": str(identity.get("transport") or "in-process"),
            "model_residency": "hawking-owned",
            "remote_slot": self.remote_slot,
            "request_id": None,
            "remote_request_id": None,
            "pid": identity.get("pid"),
            "port": identity.get("port"),
        })
        return _safe_mapping(identity)

    def profile(self) -> ResidentProfile:
        backend = self._backend or self._engine
        if backend is not None:
            try:
                profile = ResidentProfile.from_backend(
                    backend,
                    profile_id=f"hawking:{self.model_id}",
                    model_id=str(self.identity().get("model_id") or self.model_id),
                )
                profile.provider = "hawking"
                profile.runtime["backend"] = "hawking-native"
                return profile
            except Exception:
                pass
        identity = self.identity()
        return ResidentProfile(
            profile_id=f"hawking:{self.model_id}",
            provider="hawking",
            model_id=str(identity.get("model_id") or self.model_id),
            runtime={"backend": "hawking-native", "runtime": "hawking-native", "pid": identity.get("pid"), "port": identity.get("port")},
            capabilities=self.capabilities(),
            qualification={"status": "unprobed", "authority": "Hawking native observation only"},
            metadata={"identity_snapshot": identity},
        )

    def capabilities(self) -> CapabilityContract:
        backend = self._backend or self._engine
        supports = getattr(backend, "supports", None) if backend is not None else None
        if callable(supports):
            return CapabilityContract.from_backend(backend)
        return CapabilityContract(
            features={
                feature: Capability(
                    state="unknown",
                    enforcement="unknown",
                    source="Hawking native capability not probed",
                )
                for feature in FEATURES
            }
        )

    def health(self) -> ProviderHealth:
        backend = self._backend or self._engine
        ready_fn = getattr(backend, "ready", None) if backend is not None else None
        if callable(ready_fn):
            try:
                ready = bool(ready_fn(0.0))
                return ProviderHealth(
                    state="healthy" if ready else "degraded",
                    ready=ready,
                    provider="hawking",
                    detail="native runtime readiness observation",
                    recoverable=not ready,
                )
            except Exception as exc:
                return ProviderHealth(
                    state="degraded",
                    ready=False,
                    provider="hawking",
                    detail=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                )
        return ProviderHealth(
            state="unknown",
            ready=False,
            provider="hawking",
            detail="native provider has not been started",
            recoverable=True,
        )

    def _complete(self, request: GenerationRequest, timeout: Optional[float]) -> GenerationResponse:
        if self._cancelled(request, None):
            raise RemoteCancelledError(
                "HAWKING_NATIVE_CANCELLED",
                "Hawking native request was cancelled before execution",
                provider="hawking",
                model_id=self.model_id,
                request_id=request.request_id,
                recoverable=True,
            )
        backend = self._resolve_backend()
        payload = request.to_payload()
        payload["model"] = str(request.model or self.model_id)
        complete = getattr(backend, "complete", None)
        if not callable(complete):
            raise RemoteCognitionError(
                "HAWKING_NATIVE_CONTRACT",
                "Hawking native provider has no complete() operation",
                provider="hawking",
                model_id=self.model_id,
                request_id=request.request_id,
                recoverable=False,
            )
        started = time.time()
        raw_result = complete(payload, timeout=timeout)
        raw = getattr(raw_result, "raw", raw_result)
        text = getattr(raw_result, "text", None)
        if text is None:
            text = _content_text(raw)
        usage: Dict[str, Any] = {}
        if isinstance(raw_result, Mapping):
            usage_value = raw_result.get("usage")
            if isinstance(usage_value, Mapping):
                usage = dict(usage_value)
        else:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(raw_result, key, None)
                if value is not None:
                    usage[key] = value
        if isinstance(raw, Mapping):
            usage_value = raw.get("usage")
            if isinstance(usage_value, Mapping):
                usage.update(dict(usage_value))
        safe_raw = _safe_value(raw if isinstance(raw, Mapping) else {"content": text})
        identity = self.identity()
        receipt = self._failure_receipt(
            request=request,
            started_at=started,
            finished_at=time.time(),
            status="completed",
            usage=usage,
            remote_request_id=None,
        )
        receipt.update({
            "provider": "hawking",
            "model_id": self.model_id,
            "runtime": "hawking-native",
            "pid": identity.get("pid"),
            "port": identity.get("port"),
            "remote_slot": self.remote_slot,
        })
        self._last_receipt = _safe_mapping(receipt)
        if isinstance(safe_raw, dict):
            safe_raw["hawking_cognition"] = self._last_receipt
        return GenerationResponse(
            text=_strip_hidden_reasoning(str(text)) if text is not None else None,
            raw=safe_raw,
            usage=usage,
            provider="hawking",
            request_id=request.request_id,
        )

    def generate(self, request: GenerationRequest, *, timeout: Optional[float] = None) -> GenerationResponse:
        normalized = request if isinstance(request, GenerationRequest) else GenerationRequest.from_mapping(request)
        if normalized.model and normalized.model not in {self.model_id, "hawking", "native", "native-resident"}:
            raise RemoteCognitionError(
                "MODEL_ID_MISMATCH",
                f"request model {normalized.model!r} does not match selected Hawking model {self.model_id!r}",
                provider="hawking",
                model_id=self.model_id,
                request_id=normalized.request_id,
                recoverable=False,
            )
        return self._complete(normalized, timeout)

    def execute_workunit(self, wu: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
        packet = _workunit_packet(wu, context)
        resource_class = str(getattr(wu, "resource_class", "") or "").upper()
        if resource_class == "MUTATION" and context.get("mutation_lock_held") is not True:
            error = RemoteCognitionError(
                "MUTATION_LOCK_REQUIRED",
                "Hawking mutation cognition requires the existing Mission MutationLock",
                provider="hawking",
                model_id=self.model_id,
                recoverable=True,
            )
            return self.failure_result(error)
        prompt = str(context.get("prompt") or getattr(wu, "description", ""))
        instruction = (
            "Return one bounded JSON object with observations, operations, tests, "
            "questions, status, and content. Operations are proposals for "
            "Hawking's central mutation/verifier path; do not self-accept them.\n\n"
            "WORKUNIT CONTRACT:\n" + json.dumps(packet, sort_keys=True, ensure_ascii=False)
        )
        request = GenerationRequest(
            messages=(
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ),
            model=self.model_id,
            max_tokens=int(context.get("max_tokens") or os.environ.get("HAWKING_MODEL_TOKENS", "1024")),
            temperature=(float(context["temperature"]) if context.get("temperature") is not None else None),
            response_schema=(dict(context["response_schema"]) if isinstance(context.get("response_schema"), Mapping) else None),
            metadata={
                "unit_id": getattr(wu, "id", None),
                "workunit_packet": packet,
                "is_cancelled": context.get("is_cancelled"),
                "cancel_event": context.get("cancel_event"),
            },
            request_id=f"wu-{getattr(wu, "id", "work")}-{uuid.uuid4().hex[:12]}",
        )
        started = time.time()
        try:
            response = self.generate(request, timeout=context.get("timeout"))
        except RemoteCognitionError as exc:
            return self.failure_result(exc, request)
        structured = _structured_text(response.text)
        observations = structured.get("observations")
        if not isinstance(observations, list):
            observations = [response.text] if response.text else []
        return {
            "status": "observed",
            "backend": "hawking-native",
            "provider": "hawking",
            "model": self.model_id,
            "content": response.text,
            "observations": _safe_value(observations),
            "operations": _safe_value(structured.get("operations") if isinstance(structured.get("operations"), list) else []),
            "tests": _safe_value(structured.get("tests") if isinstance(structured.get("tests"), list) else []),
            "questions": _safe_value(structured.get("questions") if isinstance(structured.get("questions"), list) else []),
            "usage": _safe_mapping(response.usage),
            "cost": None,
            "wall_s": max(0.0, time.time() - started),
            "request_id": response.request_id,
            "hawking_cognition": _safe_value(self.last_receipt),
            "provider_response": response.to_dict(),
            "validation": {"ok": False, "reason": "HAWKING_COGNITION_REQUIRES_CENTRAL_VALIDATION"},
        }

    def cancel(self, request_id: str) -> Dict[str, Any]:
        return super().cancel(request_id)

# Historical spelling retained only as an inert import compatibility alias.
GrokProvider = HawkingNativeProvider


def _keychain_resolver_from_environment() -> Optional[Callable[[], Optional[str]]]:
    # The canonical secure location is deliberately usable with no project
    # setup: OpenRouter/<current macOS user>.  Explicit Hawking variables stay
    # supported as narrow overrides for managed hosts.  The value is read only
    # inside the resolver and is never placed in a receipt, prompt, or response.
    service = (
        os.environ.get("HAWKING_OPENROUTER_KEYCHAIN_SERVICE")
        or DEFAULT_OPENROUTER_KEYCHAIN_SERVICE
    )
    account = (
        os.environ.get("HAWKING_OPENROUTER_KEYCHAIN_ACCOUNT")
        or os.environ.get("USER")
        or getpass.getuser()
    )
    if not service or not account:
        return None

    def resolve() -> Optional[str]:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        value = (result.stdout or "").strip()
        return value if result.returncode == 0 and value else None

    return resolve


def openrouter_credential_available() -> bool:
    """Return only whether the secure OpenRouter credential can be resolved."""
    token = os.environ.get("OPENROUTER_API_KEY")
    if token:
        return True
    resolver = _keychain_resolver_from_environment()
    if not callable(resolver):
        return False
    try:
        return bool(resolver())
    except Exception:
        return False


def provider_for_selector(
    selector: Any,
    *,
    workspace: Optional[Path] = None,
    **kwargs: Any,
) -> Any:
    """Construct a provider lazily; construction performs no network call."""
    parsed = ProviderSelector.parse(selector)
    if parsed.provider == "openrouter":
        kwargs.setdefault("key_resolver", _keychain_resolver_from_environment())
        return OpenRouterProvider(parsed.model_id, workspace=workspace, **kwargs)
    return HawkingNativeProvider(parsed.model_id, workspace=workspace, **kwargs)


__all__ = [
    "DEFAULT_OPENROUTER_ENDPOINT",
    "GrokProvider",
    "HawkingNativeProvider",
    "OpenRouterProvider",
    "DEFAULT_OPENROUTER_KEYCHAIN_SERVICE",
    "OpenRouterGatewayPolicy",
    "ProviderSelector",
    "RemoteAuthError",
    "RemoteCancelledError",
    "RemoteCognitionError",
    "RemoteCostError",
    "RemoteCostPolicy",
    "RemoteRateLimitError",
    "RemoteExecutionIdentity",
    "is_remote_selector",
    "provider_for_selector",
    "openrouter_credential_available",
    "cached_openrouter_catalog",
]
