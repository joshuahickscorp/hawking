"""Model-neutral provider, capability, role, and provenance contracts.

HCLI historically called its local model ``qwen`` even when the selected
artifact was an MLX directory or a GGUF file.  That made the current Hawking
resident look like the architecture of the product.  This module is the
small, serialisable contract between AgentOS and any model provider:

* a provider has an identity and a capability contract;
* a profile closes over the artifact, tokenizer, runtime, representation,
  limits, fallbacks, and receipts that make a result reproducible;
* roles express policy (generalist, science, verifier, vision, frontier), not
  a hard-coded model name.

The concrete transports remain in :mod:`hcli.backends`.  This module does not
start a process, schedule work, or make a model call.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


PROFILE_SCHEMA = "hcli.provider.profile.v1"
CAPABILITY_SCHEMA = "hcli.provider.capabilities.v1"
ROLE_SCHEMA = "hcli.provider.roles.v1"
GENERATION_REQUEST_SCHEMA = "hcli.provider.generation_request.v1"
GENERATION_RESPONSE_SCHEMA = "hcli.provider.generation_response.v1"
HEALTH_SCHEMA = "hcli.provider.health.v1"
FAILURE_SCHEMA = "hcli.provider.failure.v1"
RECEIPT_SCHEMA = "hcli.provider.receipt.v1"

FEATURES: Tuple[str, ...] = (
    "response_format",
    "grammar",
    "chat_template_kwargs",
    "prefix_cache",
    "slots",
    "vision",
    "tool_calling",
    "streaming",
)

ROLES: Tuple[str, ...] = (
    "generalist",
    "science",
    "verifier",
    "vision",
    "frontier",
)


def _copy(value: Any) -> Any:
    """Return a JSON-safe defensive copy where possible."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _now() -> float:
    return time.time()


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower().replace("-", "_")
            if re.match(
                r"(?i)^(?:api[_-]?key|access[_-]?token|authorization|auth|password|secret|private[_-]?key|bearer|token|key|(?:hf|gh|github|openai|anthropic)[_-]?(?:token|key))$",
                lowered,
            ):
                result[name] = "[REDACTED]"
            else:
                result[name] = _safe_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return _copy(value)


def _safe_mapping(value: Any) -> Dict[str, Any]:
    """Copy provider metadata without retaining credential-shaped values."""
    result = _safe_value(value)
    return result if isinstance(result, dict) else {}


@dataclass(frozen=True)
class GenerationRequest:
    """Provider-neutral generation request.

    The payload remains OpenAI-shaped at the transport edge because that is a
    useful interoperability format, but AgentOS does not require a provider to
    be HTTP or to use a particular model family. Provider-specific options stay
    in ``metadata`` and are never promoted to core semantics.
    """

    messages: Tuple[Mapping[str, Any], ...] = ()
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    response_schema: Optional[Mapping[str, Any]] = None
    tools: Tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"gen-{uuid.uuid4()}")
    schema: str = GENERATION_REQUEST_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationRequest":
        messages = value.get("messages") or ()
        tools = value.get("tools") or ()
        if not isinstance(messages, (list, tuple)):
            raise TypeError("generation request messages must be an array")
        if not isinstance(tools, (list, tuple)):
            raise TypeError("generation request tools must be an array")
        response_schema = value.get("response_schema")
        if response_schema is None:
            response_schema = value.get("response_format")
        return cls(
            messages=tuple(item for item in messages if isinstance(item, Mapping)),
            model=str(value["model"]) if value.get("model") is not None else None,
            max_tokens=(int(value["max_tokens"]) if value.get("max_tokens") is not None else None),
            temperature=(float(value["temperature"]) if value.get("temperature") is not None else None),
            response_schema=(dict(response_schema) if isinstance(response_schema, Mapping) else None),
            tools=tuple(item for item in tools if isinstance(item, Mapping)),
            metadata=dict(value.get("metadata") or {}),
            request_id=str(value.get("request_id") or f"gen-{uuid.uuid4()}"),
            schema=str(value.get("schema") or GENERATION_REQUEST_SCHEMA),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "messages": [_copy(dict(item)) for item in self.messages],
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_schema": _copy(self.response_schema),
            "tools": [_copy(dict(item)) for item in self.tools],
            "metadata": _copy(dict(self.metadata)),
        }

    def to_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.pop("schema", None)
        payload.pop("request_id", None)
        payload["messages"] = [_copy(dict(item)) for item in self.messages]
        if self.response_schema is not None:
            payload["response_format"] = _copy(dict(self.response_schema))
        payload.pop("response_schema", None)
        if self.tools:
            payload["tools"] = [_copy(dict(item)) for item in self.tools]
        else:
            payload.pop("tools", None)
        payload.pop("metadata", None)
        return payload


@dataclass
class GenerationResponse:
    """Normalized response independent of transport/provider."""

    text: Optional[str] = None
    raw: Any = None
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    provider: Optional[str] = None
    request_id: Optional[str] = None
    degraded_features: List[str] = field(default_factory=list)
    schema: str = GENERATION_RESPONSE_SCHEMA

    @classmethod
    def from_completion(cls, value: Any, *, provider: Optional[str] = None) -> "GenerationResponse":
        raw = getattr(value, "raw", value)
        text = getattr(value, "text", None)
        finish = getattr(value, "finish_reason", None)
        usage: Dict[str, Any] = {}
        request_id = None
        degraded = list(getattr(value, "degraded", None) or [])
        if isinstance(raw, Mapping):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                choice = choices[0]
                finish = finish or choice.get("finish_reason")
                message = choice.get("message")
                if text is None and isinstance(message, Mapping):
                    candidate = message.get("content")
                    if isinstance(candidate, str):
                        text = candidate
            if isinstance(raw.get("usage"), Mapping):
                usage = dict(raw["usage"])
            request_id = str(raw.get("id")) if raw.get("id") is not None else None
        result = cls(
            text=text if isinstance(text, str) else (str(text) if text is not None else None),
            raw=_copy(raw),
            finish_reason=str(finish) if finish is not None else None,
            usage=usage,
            provider=provider,
            request_id=request_id,
            degraded_features=degraded,
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "text": self.text,
            "raw": _copy(self.raw),
            "finish_reason": self.finish_reason,
            "usage": _copy(self.usage),
            "provider": self.provider,
            "request_id": self.request_id,
            "degraded_features": list(self.degraded_features),
        }


@dataclass(frozen=True)
class ProviderHealth:
    """Observed provider health; it is not a qualification claim."""

    state: str = "unknown"  # healthy | degraded | unavailable | unknown
    ready: bool = False
    provider: Optional[str] = None
    detail: Optional[str] = None
    observed_at: float = field(default_factory=_now)
    recoverable: bool = True
    schema: str = HEALTH_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "state": self.state,
            "ready": self.ready,
            "provider": self.provider,
            "detail": self.detail,
            "observed_at": self.observed_at,
            "recoverable": self.recoverable,
        }


@dataclass(frozen=True)
class ProviderFailure:
    """Structured failure that tells AgentOS whether retry/recovery is safe."""

    code: str
    message: str
    recoverable: bool = True
    provider: Optional[str] = None
    request_id: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema: str = FAILURE_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "provider": self.provider,
            "request_id": self.request_id,
            "details": _copy(dict(self.details)),
        }


@dataclass
class ProviderReceipt:
    """Durable, secret-free observation of one provider call."""

    provider: str
    model_id: str
    request: Dict[str, Any] = field(default_factory=dict)
    response: Optional[Dict[str, Any]] = None
    health: Optional[Dict[str, Any]] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)
    profile_id: Optional[str] = None
    started_at: float = field(default_factory=_now)
    finished_at: Optional[float] = None
    receipt_id: str = field(default_factory=lambda: f"provider-{uuid.uuid4()}")
    schema: str = RECEIPT_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "profile_id": self.profile_id,
            "request": _safe_value(self.request),
            "response": _safe_value(self.response),
            "health": _safe_value(self.health),
            "failures": _safe_value(self.failures),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class ModelProvider(Protocol):
    """Minimal provider contract AgentOS needs from a model source."""

    def generate(self, request: GenerationRequest, *, timeout: Optional[float] = None) -> GenerationResponse:
        ...

    def capabilities(self) -> "CapabilityContract":
        ...

    def health(self) -> ProviderHealth:
        ...

    def profile(self) -> "ResidentProfile":
        ...


class ResidentProvider(ModelProvider, Protocol):
    """Provider with an explicit lifecycle (local resident or remote session)."""

    def start(self, **kwargs: Any) -> None:
        ...

    def stop(self) -> Any:
        ...


@dataclass(frozen=True)
class Capability:
    """One observed capability, with its enforcement strength."""

    state: str = "unknown"  # supported | unsupported | unknown
    enforcement: str = "unknown"  # enforced | degraded | unavailable | unknown
    source: Optional[str] = None
    observed_at: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "enforcement": self.enforcement,
            "source": self.source,
            "observed_at": self.observed_at,
            "note": self.note,
        }


@dataclass(frozen=True)
class CapabilityContract:
    """The capability surface AgentOS may rely on for a provider."""

    features: Mapping[str, Capability] = field(default_factory=dict)
    schema: str = CAPABILITY_SCHEMA

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        *,
        feature_names: Sequence[str] = FEATURES,
        observed_at: Optional[float] = None,
    ) -> "CapabilityContract":
        observed = _now() if observed_at is None else float(observed_at)
        supports = getattr(backend, "supports", None)
        values: Dict[str, Capability] = {}
        for feature in feature_names:
            if not callable(supports):
                values[feature] = Capability(
                    source="backend has no supports()",
                    observed_at=observed,
                )
                continue
            try:
                observed_support = supports(feature)
                if observed_support is None:
                    values[feature] = Capability(
                        source="backend.supports",
                        observed_at=observed,
                        note="provider did not declare this capability",
                    )
                    continue
                supported = bool(observed_support)
            except Exception as exc:  # noqa: BLE001 - a census must not crash
                values[feature] = Capability(
                    source="backend.supports",
                    observed_at=observed,
                    note=f"probe error: {type(exc).__name__}: {exc}",
                )
                continue
            values[feature] = Capability(
                state="supported" if supported else "unsupported",
                enforcement="enforced" if supported else "degraded",
                source="backend.supports",
                observed_at=observed,
            )
        return cls(features=values)

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilityContract":
        if not isinstance(value, Mapping):
            return cls()
        raw = value.get("features", value)
        if not isinstance(raw, Mapping):
            return cls()
        features: Dict[str, Capability] = {}
        for name, item in raw.items():
            if isinstance(item, Capability):
                features[str(name)] = item
            elif isinstance(item, dict):
                features[str(name)] = Capability(
                    state=str(item.get("state") or "unknown"),
                    enforcement=str(item.get("enforcement") or "unknown"),
                    source=item.get("source"),
                    observed_at=item.get("observed_at"),
                    note=item.get("note"),
                )
            else:
                supported = bool(item)
                features[str(name)] = Capability(
                    state="supported" if supported else "unsupported",
                    enforcement="unknown",
                )
        return cls(features=features, schema=str(value.get("schema") or CAPABILITY_SCHEMA))

    def supports(self, feature: str) -> Optional[bool]:
        aliases = {
            "json_schema": "response_format",
            "response_format_json_schema": "response_format",
            "grammar_gbnf": "grammar",
            "prompt_prefix_cache": "prefix_cache",
            "continuous_batching_slots": "slots",
        }
        item = self.features.get(aliases.get(str(feature), str(feature)))
        if item is None or item.state == "unknown":
            return None
        return item.state == "supported"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "features": {
                str(name): item.to_dict()
                for name, item in sorted(self.features.items())
            },
        }


@dataclass
class ResidentProfile:
    """Reproducibility closure for a model/provider selection.

    Fields deliberately remain dictionaries at the boundary.  Different
    runtimes have different compiler, representation, and machine-genome
    vocabularies; forcing those into one lossy dataclass would hide the exact
    information the receipt is meant to preserve.
    """

    profile_id: str
    provider: str
    model_id: str
    artifact: Dict[str, Any] = field(default_factory=dict)
    tokenizer: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    compiler: Dict[str, Any] = field(default_factory=dict)
    representation: Dict[str, Any] = field(default_factory=dict)
    capabilities: CapabilityContract = field(default_factory=CapabilityContract)
    prompt_contract: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)
    fallbacks: List[Dict[str, Any] | str] = field(default_factory=list)
    hot_bytes: Optional[int] = None
    machine_genome: Dict[str, Any] = field(default_factory=dict)
    receipts: List[str] = field(default_factory=list)
    qualification: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = PROFILE_SCHEMA
    environment: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidentProfile":
        data = dict(value)
        caps = CapabilityContract.from_mapping(data.get("capabilities"))
        qualification = data.get("qualification")
        if not isinstance(qualification, dict):
            qualification = {"status": qualification} if qualification else {}
        fallbacks = data.get("fallbacks")
        if not isinstance(fallbacks, list):
            fallbacks = [] if fallbacks is None else [fallbacks]
        receipts = data.get("receipts")
        if not isinstance(receipts, list):
            receipts = [] if receipts is None else [receipts]
        return cls(
            profile_id=str(data.get("profile_id") or data.get("id") or "profile"),
            provider=str(data.get("provider") or "unknown"),
            model_id=str(data.get("model_id") or data.get("model") or "unknown"),
            artifact=dict(data.get("artifact") or {}),
            tokenizer=dict(data.get("tokenizer") or {}),
            runtime=dict(data.get("runtime") or {}),
            compiler=dict(data.get("compiler") or {}),
            representation=dict(data.get("representation") or {}),
            capabilities=caps,
            prompt_contract=dict(data.get("prompt_contract") or {}),
            generation=dict(data.get("generation") or {}),
            limits=dict(data.get("limits") or {}),
            fallbacks=list(fallbacks),
            hot_bytes=(int(data["hot_bytes"]) if data.get("hot_bytes") is not None else None),
            machine_genome=dict(data.get("machine_genome") or {}),
            receipts=[str(item) for item in receipts],
            qualification=qualification,
            metadata=dict(data.get("metadata") or {}),
            schema=str(data.get("schema") or PROFILE_SCHEMA),
            environment=dict(data.get("environment") or data.get("env") or {}),
        )

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        *,
        profile_id: Optional[str] = None,
        model_id: Optional[str] = None,
        receipts: Optional[Iterable[str]] = None,
    ) -> "ResidentProfile":
        identity: Dict[str, Any] = {}
        identity_fn = getattr(backend, "identity", None)
        if callable(identity_fn):
            try:
                value = identity_fn()
                if isinstance(value, dict):
                    identity = value
            except Exception as exc:  # noqa: BLE001 - provenance must survive probes
                identity = {"identity_error": f"{type(exc).__name__}: {exc}"}
        provider = str(
            identity.get("provider")
            or identity.get("runtime")
            or identity.get("backend")
            or type(backend).__name__
        )
        chosen_model = str(
            model_id
            or identity.get("model_id")
            or identity.get("model_identity")
            or identity.get("model_path")
            or "unknown"
        )
        artifact = {
            key: identity[key]
            for key in (
                "model_path",
                "model_identity",
                "model_bytes",
                "artifact_root",
                "artifact_inventory",
            )
            if key in identity
        }
        tokenizer = {
            key: identity[key]
            for key in ("tokenizer", "tokenizer_sha256_16")
            if key in identity
        }
        runtime = {
            key: identity[key]
            for key in (
                "backend",
                "runtime",
                "runtime_build",
                "binary",
                "binary_sha256_16",
                "protocol",
                "mode",
                "pid",
                "port",
            )
            if key in identity
        }
        if isinstance(identity.get("current_runtime"), Mapping):
            runtime["current_runtime"] = _safe_mapping(identity["current_runtime"])
        environment = _safe_mapping(
            identity.get("environment")
            or identity.get("env")
            or identity.get("fusion_env")
        )
        limits = {
            key: identity[key]
            for key in ("context", "max_seq_len", "n_slots", "decode_concurrency")
            if key in identity
        }
        if isinstance(identity.get("limits"), Mapping):
            limits.update(_safe_mapping(identity["limits"]))
        generation = dict(identity.get("generation") or {})
        representation = _safe_mapping(identity.get("representation"))
        if identity.get("physical_ebpw") is not None:
            representation.setdefault("physical_ebpw", identity["physical_ebpw"])
        if identity.get("quantisation") is not None:
            representation.setdefault("quantisation", identity["quantisation"])
        if identity.get("require_fusion_env") is not None:
            representation.setdefault("require_fusion_env", bool(identity["require_fusion_env"]))
        caps = CapabilityContract.from_backend(backend)
        declared_caps = CapabilityContract.from_mapping(identity.get("capabilities"))
        if declared_caps.features:
            merged_caps = dict(declared_caps.features)
            # A live supports() probe is stronger than a profile declaration.
            merged_caps.update(caps.features)
            caps = CapabilityContract(features=merged_caps)
        known_fallbacks = identity.get("fallbacks") or identity.get("degraded_features") or []
        if not isinstance(known_fallbacks, list):
            known_fallbacks = [known_fallbacks]
        hot = identity.get("hot_bytes")
        if hot is None:
            hot = identity.get("model_bytes")
        try:
            hot_bytes = int(hot) if hot is not None else None
        except (TypeError, ValueError):
            hot_bytes = None
        qualification = identity.get("qualification")
        if not isinstance(qualification, dict):
            qualification = {"status": qualification} if qualification else {}
        return cls(
            profile_id=profile_id or f"{provider}:{chosen_model}",
            provider=provider,
            model_id=chosen_model,
            artifact=artifact,
            tokenizer=tokenizer,
            runtime=runtime,
            compiler=dict(identity.get("compiler") or {}),
            representation=representation,
            capabilities=caps,
            prompt_contract=dict(identity.get("prompt_contract") or {}),
            generation=generation,
            limits=limits,
            fallbacks=list(known_fallbacks),
            hot_bytes=hot_bytes,
            machine_genome=dict(identity.get("machine_genome") or {}),
            receipts=[str(item) for item in (receipts or identity.get("receipts") or [])],
            qualification=qualification,
            metadata={
                "identity_snapshot": _safe_mapping(identity),
                "observed_at": _now(),
            },
            environment=environment,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "artifact": _copy(self.artifact),
            "tokenizer": _copy(self.tokenizer),
            "runtime": _copy(self.runtime),
            "compiler": _copy(self.compiler),
            "representation": _copy(self.representation),
            "capabilities": self.capabilities.to_dict(),
            "prompt_contract": _copy(self.prompt_contract),
            "generation": _copy(self.generation),
            "limits": _copy(self.limits),
            "fallbacks": _copy(self.fallbacks),
            "hot_bytes": self.hot_bytes,
            "machine_genome": _copy(self.machine_genome),
            "receipts": list(self.receipts),
            "qualification": _copy(self.qualification),
            "metadata": _copy(self.metadata),
            "environment": _copy(self.environment),
        }

    def fingerprint(self) -> str:
        """Stable identity of the profile, excluding observation timestamps."""
        payload = self.to_dict()
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("observed_at", None)
            identity = metadata.get("identity_snapshot")
            if isinstance(identity, dict):
                for key in ("pid", "port", "created", "timestamp"):
                    identity.pop(key, None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def supports(self, feature: str) -> Optional[bool]:
        """Return the profile's declared capability, if it is known."""
        return self.capabilities.supports(feature)


@dataclass(frozen=True)
class RolePolicy:
    """Provider preference policy for a cognition role."""

    role: str
    providers: Tuple[str, ...]
    requires: Tuple[str, ...] = ()
    fallback_roles: Tuple[str, ...] = ()
    note: str = ""
    budget: Optional[float] = None
    latency_threshold_s: Optional[float] = None
    capability_threshold: Optional[Mapping[str, Any]] = None
    fallback_chain: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RolePolicy":
        if not isinstance(value, Mapping):
            raise TypeError("role policy must be an object")
        providers = value.get("providers") or value.get("preference") or ()
        if isinstance(providers, str):
            providers = (providers,)
        requires = value.get("requires") or ()
        if isinstance(requires, str):
            requires = (requires,)
        fallback_roles = value.get("fallback_roles") or ()
        if isinstance(fallback_roles, str):
            fallback_roles = (fallback_roles,)
        fallback_chain = value.get("fallback_chain") or ()
        if isinstance(fallback_chain, str):
            fallback_chain = (fallback_chain,)
        return cls(
            role=str(value.get("role") or "generalist"),
            providers=tuple(str(item) for item in providers),
            requires=tuple(str(item) for item in requires),
            fallback_roles=tuple(str(item) for item in fallback_roles),
            note=str(value.get("note") or ""),
            budget=float(value["budget"]) if value.get("budget") is not None else None,
            latency_threshold_s=(float(value["latency_threshold_s"]) if value.get("latency_threshold_s") is not None else None),
            capability_threshold=(dict(value["capability_threshold"]) if isinstance(value.get("capability_threshold"), Mapping) else None),
            fallback_chain=tuple(str(item) for item in fallback_chain),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "providers": list(self.providers),
            "requires": list(self.requires),
            "fallback_roles": list(self.fallback_roles),
            "note": self.note,
            "budget": self.budget,
            "latency_threshold_s": self.latency_threshold_s,
            "capability_threshold": _copy(self.capability_threshold),
            "fallback_chain": list(self.fallback_chain),
        }


DEFAULT_ROLE_POLICIES: Dict[str, RolePolicy] = {
    "generalist": RolePolicy("generalist", ("resident", "local", "remote")),
    "science": RolePolicy("science", ("resident", "specialist", "local", "remote")),
    "coding": RolePolicy("coding", ("specialist", "resident", "local", "remote")),
    "verifier": RolePolicy(
        "verifier", ("cpu", "resident", "local"), ("deterministic_verification",)
    ),
    "vision": RolePolicy(
        "vision", ("vmcp", "multimodal", "remote", "local"), ("vision",)
    ),
    "frontier": RolePolicy("frontier", ("remote", "resident", "local")),
}


class RoleRouter:
    """Resolve roles against provider names and capability contracts."""

    def __init__(
        self,
        policies: Optional[Mapping[str, RolePolicy]] = None,
    ) -> None:
        self.policies = dict(policies or DEFAULT_ROLE_POLICIES)

    def choose(
        self,
        role: str,
        providers: Mapping[str, Any] | Iterable[str],
    ) -> Dict[str, Any]:
        role_name = str(role or "generalist").strip().lower()
        role_aliases = {
            "resident_generalist": "generalist",
            "science_specialist": "science",
            "coding_specialist": "coding",
            "verifier_assistant": "verifier",
            "vision/perception": "vision",
            "vision_perception": "vision",
            "remote_frontier": "frontier",
        }
        role_name = role_aliases.get(role_name, role_name)
        policy = self.policies.get(role_name) or self.policies["generalist"]
        values = (
            dict(providers)
            if isinstance(providers, Mapping)
            else {str(name): str(name) for name in providers}
        )
        attempted: List[Dict[str, Any]] = []
        preferences = list(policy.providers)
        for fallback in policy.fallback_chain:
            if fallback not in preferences:
                preferences.append(fallback)
        for preference in preferences:
            for provider_name, provider_value in values.items():
                if not self._matches_preference(provider_name, preference):
                    continue
                requirements = list(policy.requires)
                if isinstance(policy.capability_threshold, Mapping):
                    requirements.extend(str(key) for key in policy.capability_threshold)
                missing = [
                    requirement
                    for requirement in requirements
                    if not self._supports(provider_name, provider_value, requirement)
                ]
                if missing:
                    attempted.append({"provider": provider_name, "missing": missing})
                    continue
                return {
                    "role": role_name,
                    "provider": provider_name,
                    "fallback": not policy.providers or preference != policy.providers[0],
                    "policy": policy.to_dict(),
                    "reason": "first available provider satisfying role policy",
                }
        return {
            "role": role_name,
            "provider": None,
            "fallback": False,
            "policy": policy.to_dict(),
            "reason": (
                "no provider satisfies the role policy"
                if not attempted
                else f"capability requirements not met: {attempted}"
            ),
        }

    @staticmethod
    def _matches_preference(provider: str, preference: str) -> bool:
        name = str(provider).strip().lower()
        wanted = str(preference).strip().lower()
        if name == wanted:
            return True
        aliases = {
            "resident": {"native", "noetic_native", "hawking_native", "hawking-native"},
            "local": {"mlx", "llamacpp", "llama.cpp", "native", "noetic_native", "hawking_native", "hawking-native"},
            "remote": {"openai", "openai-compatible", "openai_compatible", "http", "https"},
        }
        return name in aliases.get(wanted, set())

    @staticmethod
    def _supports(provider: str, value: Any, requirement: str) -> bool:
        # Deterministic verification is a capability of the verifier/tool
        # plane, not a claim that a language model can make about itself.
        if requirement == "deterministic_verification" and str(provider).lower() in {
            "cpu", "tool", "vmcp"
        }:
            return True
        supports = getattr(value, "supports", None)
        if callable(supports):
            try:
                return supports(requirement) is True
            except Exception:
                return False
        if isinstance(value, ResidentProfile):
            return value.supports(requirement) is True
        if isinstance(value, CapabilityContract):
            return value.supports(requirement) is True
        if isinstance(value, Mapping):
            profile = value.get("profile")
            if isinstance(profile, Mapping):
                return RoleRouter._supports(provider, ResidentProfile.from_mapping(profile), requirement)
            caps = CapabilityContract.from_mapping(value.get("capabilities", value))
            return caps.supports(requirement) is True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": ROLE_SCHEMA,
            "roles": {name: policy.to_dict() for name, policy in sorted(self.policies.items())},
        }


def profile_from_backend(backend: Any, **kwargs: Any) -> ResidentProfile:
    """Convenience adapter used by RuntimeInterface and receipts."""
    return ResidentProfile.from_backend(backend, **kwargs)


def host_profile() -> Dict[str, Any]:
    """Small non-secret machine identity for a provider receipt."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


__all__ = [
    "CAPABILITY_SCHEMA",
    "DEFAULT_ROLE_POLICIES",
    "FAILURE_SCHEMA",
    "FEATURES",
    "GENERATION_REQUEST_SCHEMA",
    "GENERATION_RESPONSE_SCHEMA",
    "HEALTH_SCHEMA",
    "PROFILE_SCHEMA",
    "RECEIPT_SCHEMA",
    "ROLES",
    "ROLE_SCHEMA",
    "Capability",
    "CapabilityContract",
    "GenerationRequest",
    "GenerationResponse",
    "ModelProvider",
    "ProviderFailure",
    "ProviderHealth",
    "ProviderReceipt",
    "ResidentProfile",
    "ResidentProvider",
    "RolePolicy",
    "RoleRouter",
    "host_profile",
    "profile_from_backend",
]
