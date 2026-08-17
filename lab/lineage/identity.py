"""Genesis instance identity: artifact, representation, runtime/kernel genome."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from lab.lineage.canon import (
    bpw_key,
    labeled_sha,
    require_mapping,
    require_nonempty_str,
    require_sha256,
    utc_now,
)

SCHEMA = "hawking.lineage.genesis_instance.v1"

# Sealed Qwen3.8 Genesis identity (tournament 2026-08-16).
GENESIS_MODEL = "PocketAiHub/Qwen3.8-27B-Abliterated-MLX"
GENESIS_BASE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GENESIS_ARTIFACT_PATH = "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
GENESIS_ARTIFACT_MANIFEST_PATH = f"{GENESIS_ARTIFACT_PATH}/manifest.json"
# SHA-256 of the sealed manifest bytes, not a synthetic lineage label.
GENESIS_ARTIFACT_MANIFEST_SHA = (
    "d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df"
)
GENESIS_BINARY = "ascension_qwen38_hybrid_greedy"
GENESIS_COMPLETE_TOKEN_NS = 35_227_918
GENESIS_BPW = 4.2527
GENESIS_TPS = 28.4
FIRST_RUNG_COMPLETE_TOKEN_NS = 10_000_000

DEFAULT_CAPABILITY_CONTRACT: dict[str, float] = {
    "coherence": 1.0,
    "complete_token_discipline": 1.0,
    "engineering": 1.0,
}

DEFAULT_BENCHMARK_FINGERPRINT = labeled_sha(
    "bench/complete-token/qwen38/greedy/3prompt/gpu-cb-timestamps"
)


class IdentityError(ValueError):
    """Instance identity is incomplete or inconsistent."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of file bytes, refusing absent/non-file artifacts."""
    artifact = Path(path)
    if not artifact.is_file():
        raise IdentityError(f"artifact identity source is not a file: {artifact}")
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IdentityError(f"cannot hash artifact identity source {artifact}: {exc}") from exc
    return digest.hexdigest()


@dataclass
class Invoker:
    """Who is asking the promotion gate to run.

    Parent and child identities are refused. Only an external principal
    (protected controller, human operator, or the lineage gate process)
    may invoke the gate.
    """

    principal: str
    identity: str
    acting_as: str = "external"

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "identity": self.identity,
            "acting_as": self.acting_as,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Invoker":
        data = require_mapping(raw, "invoker")
        return cls(
            principal=require_nonempty_str(data.get("principal"), "invoker.principal"),
            identity=require_nonempty_str(data.get("identity"), "invoker.identity"),
            acting_as=str(data.get("acting_as") or "external"),
        )


EXTERNAL_PRINCIPALS: frozenset[str] = frozenset(
    {
        "protected_controller",
        "human_operator",
        "lineage_gate",
    }
)

SELF_PRINCIPALS: frozenset[str] = frozenset(
    {
        "parent",
        "child",
        "current",
        "candidate",
        "genesis",
        "genesis_self",
        "sandbox_model",
        "self",
    }
)


@dataclass
class GenesisInstance:
    """One named Genesis occupant. Copies are stored in lineage slots."""

    instance_id: str
    generation: int
    artifact_sha: str
    runtime_sha: str
    kernel_genome_sha: str
    representation_bpw: float
    complete_token_ns: int
    capability: dict[str, float]
    physical_bpw: float | None = None
    silent_fallback_ids: tuple[str, ...] = ()
    benchmark_fingerprint: str = DEFAULT_BENCHMARK_FINGERPRINT
    identity: dict[str, str] = field(default_factory=dict)
    lane: str = "integrator"
    valid: bool = True
    live: bool = False
    terminated: bool = False
    launched: bool = False
    role: str = ""
    research_state: dict[str, Any] | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        self.instance_id = require_nonempty_str(self.instance_id, "instance_id")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 0:
            raise IdentityError("generation must be a non-negative int")
        self.artifact_sha = require_sha256(self.artifact_sha, "artifact_sha")
        self.runtime_sha = require_sha256(self.runtime_sha, "runtime_sha")
        self.kernel_genome_sha = require_sha256(self.kernel_genome_sha, "kernel_genome_sha")
        self.representation_bpw = float(bpw_key(self.representation_bpw))
        if self.physical_bpw is not None:
            if (
                isinstance(self.physical_bpw, bool)
                or not isinstance(self.physical_bpw, (int, float))
                or not math.isfinite(float(self.physical_bpw))
                or float(self.physical_bpw) <= 0.0
            ):
                raise IdentityError("physical_bpw must be a positive number when present")
            self.physical_bpw = float(self.physical_bpw)
        if not isinstance(self.complete_token_ns, int) or isinstance(self.complete_token_ns, bool):
            raise IdentityError("complete_token_ns must be an int")
        if self.complete_token_ns <= 0:
            raise IdentityError("complete_token_ns must be positive")
        if not isinstance(self.capability, dict) or not self.capability:
            raise IdentityError("capability contract map is required")
        cleaned: dict[str, float] = {}
        for key, value in self.capability.items():
            if not isinstance(key, str) or not key.strip():
                raise IdentityError("capability axis names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise IdentityError(f"capability[{key!r}] must be numeric")
            cleaned[key] = float(value)
        self.capability = cleaned
        self.silent_fallback_ids = tuple(
            require_nonempty_str(item, "silent_fallback_id") for item in self.silent_fallback_ids
        )
        self.benchmark_fingerprint = require_nonempty_str(
            self.benchmark_fingerprint, "benchmark_fingerprint"
        )
        self.identity = dict(self.identity or {})
        if not self.created_at:
            self.created_at = utc_now()

    @property
    def tps(self) -> float:
        return 1_000_000_000.0 / float(self.complete_token_ns)

    def copy(self) -> "GenesisInstance":
        return GenesisInstance(
            instance_id=self.instance_id,
            generation=self.generation,
            artifact_sha=self.artifact_sha,
            runtime_sha=self.runtime_sha,
            kernel_genome_sha=self.kernel_genome_sha,
            representation_bpw=self.representation_bpw,
            complete_token_ns=self.complete_token_ns,
            capability=dict(self.capability),
            physical_bpw=self.physical_bpw,
            silent_fallback_ids=tuple(self.silent_fallback_ids),
            benchmark_fingerprint=self.benchmark_fingerprint,
            identity=dict(self.identity),
            lane=self.lane,
            valid=self.valid,
            live=self.live,
            terminated=self.terminated,
            launched=self.launched,
            role=self.role,
            research_state=None if self.research_state is None else dict(self.research_state),
            created_at=self.created_at,
        )

    def invoke_promotion_gate(self, *args: Any, **kwargs: Any) -> None:
        """Neither parent nor child may invoke the gate on itself."""
        from lab.lineage.promotion import SelfCertificationRefused

        raise SelfCertificationRefused(
            f"{self.instance_id} may not invoke the promotion gate on itself "
            "(promotion authority is external)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "instance_id": self.instance_id,
            "generation": self.generation,
            "artifact_sha": self.artifact_sha,
            "runtime_sha": self.runtime_sha,
            "kernel_genome_sha": self.kernel_genome_sha,
            "representation_bpw": self.representation_bpw,
            "physical_bpw": self.physical_bpw,
            "complete_token_ns": self.complete_token_ns,
            "tps": self.tps,
            "capability": dict(self.capability),
            "silent_fallback_ids": list(self.silent_fallback_ids),
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "identity": dict(self.identity),
            "lane": self.lane,
            "valid": self.valid,
            "live": self.live,
            "terminated": self.terminated,
            "launched": self.launched,
            "role": self.role,
            "research_state": None if self.research_state is None else dict(self.research_state),
            "created_at": self.created_at,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GenesisInstance":
        data = require_mapping(raw, "genesis_instance")
        return cls(
            instance_id=str(data["instance_id"]),
            generation=int(data["generation"]),
            artifact_sha=str(data["artifact_sha"]),
            runtime_sha=str(data["runtime_sha"]),
            kernel_genome_sha=str(data["kernel_genome_sha"]),
            representation_bpw=float(data["representation_bpw"]),
            complete_token_ns=int(data["complete_token_ns"]),
            capability=dict(data.get("capability") or {}),
            physical_bpw=data.get("physical_bpw"),
            silent_fallback_ids=tuple(data.get("silent_fallback_ids") or ()),
            benchmark_fingerprint=str(
                data.get("benchmark_fingerprint") or DEFAULT_BENCHMARK_FINGERPRINT
            ),
            identity=dict(data.get("identity") or {}),
            lane=str(data.get("lane") or "integrator"),
            valid=bool(data.get("valid", True)),
            live=bool(data.get("live", False)),
            terminated=bool(data.get("terminated", False)),
            launched=bool(data.get("launched", False)),
            role=str(data.get("role") or ""),
            research_state=data.get("research_state"),
            created_at=str(data.get("created_at") or ""),
        )


def make_qwen38_genesis(
    *,
    instance_id: str = "genesis-qwen38-g0",
    artifact_sha: str = GENESIS_ARTIFACT_MANIFEST_SHA,
) -> GenesisInstance:
    """The seated Hawking Genesis parent as of 2026-08-16."""
    return GenesisInstance(
        instance_id=instance_id,
        generation=0,
        artifact_sha=artifact_sha,
        runtime_sha=labeled_sha("runtime/ascension_qwen38_hybrid_greedy"),
        kernel_genome_sha=labeled_sha(
            "genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64"
        ),
        representation_bpw=GENESIS_BPW,
        complete_token_ns=GENESIS_COMPLETE_TOKEN_NS,
        capability=dict(DEFAULT_CAPABILITY_CONTRACT),
        silent_fallback_ids=(),
        benchmark_fingerprint=DEFAULT_BENCHMARK_FINGERPRINT,
        identity={
            "model": GENESIS_MODEL,
            "base_rev": GENESIS_BASE_REV,
            "artifact": GENESIS_ARTIFACT_PATH,
            "artifact_manifest": GENESIS_ARTIFACT_MANIFEST_PATH,
            "artifact_sha_authority": "sha256(manifest.json bytes)",
            "binary": GENESIS_BINARY,
        },
        lane="integrator",
        valid=True,
        live=False,
        launched=False,
        role="current",
    )


def as_instance(value: GenesisInstance | Mapping[str, Any], name: str = "instance") -> GenesisInstance:
    if isinstance(value, GenesisInstance):
        return value
    return GenesisInstance.from_mapping(require_mapping(value, name))


def as_invoker(value: Invoker | Mapping[str, Any]) -> Invoker:
    if isinstance(value, Invoker):
        return value
    return Invoker.from_mapping(value)
