"""Shared immutable types for credential-broker preflight and inventory.

Generalises the sealed shapes already used by:
- ``lab.operators.kimi_k3_source_admission`` (official repo + 40-char revision +
  LFS hash inventory + license + architecture facts + claim boundary)
- ``lab.operators.deepseek_v4_stream_executor`` (pinned revision, range plans,
  storage_policy floor + max_inflight, source_retention_paths)
- ``lab.operators.glm52_source_fetch`` (manifest files/sizes, VERIFIED ledger,
  disk floor)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TypeError_(ValueError):
    """Broker type invariant failed closed."""


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError_(f"{label} must be a non-empty string")
    return value.strip()


def _require_nonneg_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError_(f"{label} must be a non-negative integer")
    return value


def _require_pos_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError_(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class ImmutableRevision:
    """Pinned immutable Hub (or equivalent) commit identity.

    Mirrors Kimi/DeepSeek practice: never bind ``main`` alone; pin the resolved
    40-character commit before any range plan or body transfer is admissible.
    """

    commit: str
    requested: str = "main"

    def __post_init__(self) -> None:
        commit = _require_str(self.commit, "revision.commit")
        if _COMMIT_RE.fullmatch(commit) is None:
            raise TypeError_(
                f"revision.commit must be a 40-character lowercase hex commit, got {commit!r}"
            )
        object.__setattr__(self, "commit", commit.lower())
        object.__setattr__(self, "requested", _require_str(self.requested, "revision.requested"))

    def as_dict(self) -> dict[str, str]:
        return {"commit": self.commit, "requested": self.requested}


@dataclass(frozen=True)
class OfficialSource:
    """Official repository binding before any acquisition.

    Bible §7 pre-acquisition checklist item: official source + license.
    """

    repository: str
    revision: ImmutableRevision
    license_id: str
    license_file_sha256: str | None = None
    private: bool = False
    gated: bool = False
    source_authority: str = "official_hub_metadata"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", _require_str(self.repository, "repository"))
        if not isinstance(self.revision, ImmutableRevision):
            raise TypeError_("revision must be an ImmutableRevision")
        object.__setattr__(self, "license_id", _require_str(self.license_id, "license_id"))
        if self.license_file_sha256 is not None:
            digest = _require_str(self.license_file_sha256, "license_file_sha256").lower()
            if _SHA256_RE.fullmatch(digest) is None:
                raise TypeError_("license_file_sha256 must be 64 hex characters")
            object.__setattr__(self, "license_file_sha256", digest)
        if self.private or self.gated:
            # Gated/private sources still use the broker; models still never see the token.
            pass
        object.__setattr__(
            self,
            "source_authority",
            _require_str(self.source_authority, "source_authority"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision.as_dict(),
            "license_id": self.license_id,
            "license_file_sha256": self.license_file_sha256,
            "private": bool(self.private),
            "gated": bool(self.gated),
            "source_authority": self.source_authority,
        }


@dataclass(frozen=True)
class FileEntry:
    """One exact file in a hash inventory (control asset or weight shard)."""

    path: str
    bytes: int
    sha256: str | None = None
    lfs_sha256: str | None = None
    kind: str = "weight"  # weight | control | other

    def __post_init__(self) -> None:
        path = _require_str(self.path, "file.path")
        if "/" in path and path.startswith("/"):
            raise TypeError_("file.path must be a repository-relative path, not absolute")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "bytes", _require_nonneg_int(self.bytes, "file.bytes"))
        for label, value in (("sha256", self.sha256), ("lfs_sha256", self.lfs_sha256)):
            if value is None:
                continue
            digest = _require_str(value, label).lower()
            if _SHA256_RE.fullmatch(digest) is None:
                raise TypeError_(f"{label} must be 64 hex characters")
            object.__setattr__(self, label, digest)
        kind = _require_str(self.kind, "file.kind")
        if kind not in {"weight", "control", "other"}:
            raise TypeError_("file.kind must be weight|control|other")
        object.__setattr__(self, "kind", kind)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HashInventory:
    """Exact files, sizes, and hash identities for a pinned revision.

    Pattern from Kimi ``weight_shards`` / GLM ``GLM52_OFFICIAL_MANIFEST`` /
    DeepSeek range ``expected_capture_sha256``.
    """

    files: tuple[FileEntry, ...]
    revision: ImmutableRevision

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple) or not self.files:
            raise TypeError_("hash inventory requires at least one FileEntry")
        if not all(isinstance(f, FileEntry) for f in self.files):
            raise TypeError_("files must be FileEntry instances")
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise TypeError_("hash inventory contains duplicate paths")
        if not isinstance(self.revision, ImmutableRevision):
            raise TypeError_("inventory revision must be ImmutableRevision")

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def weight_bytes(self) -> int:
        return sum(f.bytes for f in self.files if f.kind == "weight")

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision.as_dict(),
            "files": [f.as_dict() for f in self.files],
            "total_bytes": self.total_bytes,
            "weight_bytes": self.weight_bytes,
            "file_count": len(self.files),
        }


@dataclass(frozen=True)
class ArchitectureClassification:
    """Architecture facts derived from official config metadata only."""

    model_type: str
    architectures: tuple[str, ...]
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    vocab_size: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_type", _require_str(self.model_type, "model_type"))
        if not isinstance(self.architectures, tuple) or not self.architectures:
            raise TypeError_("architectures must be a non-empty tuple of strings")
        for a in self.architectures:
            _require_str(a, "architecture")
        for label, value in (
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("vocab_size", self.vocab_size),
        ):
            if value is not None:
                _require_pos_int(value, label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "vocab_size": self.vocab_size,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class StorageForecast:
    """Storage plan: peak source, intermediate, sealed Gravity, residual floor."""

    peak_source_bytes: int
    peak_intermediate_bytes: int
    sealed_artifact_bytes: int
    protected_floor_bytes: int
    max_inflight_bytes: int
    no_full_source_accumulation: bool = True

    def __post_init__(self) -> None:
        for label in (
            "peak_source_bytes",
            "peak_intermediate_bytes",
            "sealed_artifact_bytes",
            "protected_floor_bytes",
            "max_inflight_bytes",
        ):
            _require_nonneg_int(getattr(self, label), label)
        if not self.no_full_source_accumulation:
            raise TypeError_(
                "Bible §7 forbids accumulating full source models + duplicate intermediates; "
                "no_full_source_accumulation must be True"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeMemoryForecast:
    """Runtime-memory forecast for the intended Gravity/runtime path."""

    peak_resident_bytes: int
    working_set_bytes: int
    kv_or_state_bytes: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        for label in ("peak_resident_bytes", "working_set_bytes", "kv_or_state_bytes"):
            _require_nonneg_int(getattr(self, label), label)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GravityPlanSummary:
    """Pointer to the intended Gravity transform plan (not the transform itself)."""

    plan_id: str
    transform_family: str
    target_artifact_schema: str
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _require_str(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "transform_family", _require_str(self.transform_family, "transform_family")
        )
        object.__setattr__(
            self,
            "target_artifact_schema",
            _require_str(self.target_artifact_schema, "target_artifact_schema"),
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificPurpose:
    """Why this acquisition exists; without purpose, acquisition is refused."""

    purpose_id: str
    statement: str
    programme: str  # e.g. bootstrap_qwen_30b, bootstrap_qwen_80b, frankenstein_donor
    success_metric: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose_id", _require_str(self.purpose_id, "purpose_id"))
        object.__setattr__(self, "statement", _require_str(self.statement, "statement"))
        object.__setattr__(self, "programme", _require_str(self.programme, "programme"))
        object.__setattr__(
            self, "success_metric", _require_str(self.success_metric, "success_metric")
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RangeRequest:
    """One approved byte range for stream download/resume/verify.

    DeepSeek currently admits header-only ranges; GLM admits full-shard ranges
    under a schedule. This type is family-agnostic; the executor enforces kind
    limits per model programme.
    """

    range_id: str
    path: str
    start: int
    end: int
    expected_sha256: str | None = None
    kind: str = "body"  # body | header | control

    def __post_init__(self) -> None:
        object.__setattr__(self, "range_id", _require_str(self.range_id, "range_id"))
        object.__setattr__(self, "path", _require_str(self.path, "path"))
        start = _require_nonneg_int(self.start, "start")
        end = _require_pos_int(self.end, "end")
        if end <= start:
            raise TypeError_("range end must be > start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if self.expected_sha256 is not None:
            digest = _require_str(self.expected_sha256, "expected_sha256").lower()
            if _SHA256_RE.fullmatch(digest) is None:
                raise TypeError_("expected_sha256 must be 64 hex characters")
            object.__setattr__(self, "expected_sha256", digest)
        kind = _require_str(self.kind, "kind")
        if kind not in {"body", "header", "control"}:
            raise TypeError_("range kind must be body|header|control")
        object.__setattr__(self, "kind", kind)

    @property
    def expected_bytes(self) -> int:
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "range_id": self.range_id,
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "expected_bytes": self.expected_bytes,
            "expected_sha256": self.expected_sha256,
            "kind": self.kind,
        }


def inventory_from_mapping(value: Mapping[str, Any], revision: ImmutableRevision) -> HashInventory:
    """Build a HashInventory from a Kimi/GLM-style file list mapping."""
    rows = value.get("files") or value.get("weight_shards") or value.get("files_list")
    if not isinstance(rows, list) or not rows:
        raise TypeError_("mapping must contain a non-empty files or weight_shards list")
    files: list[FileEntry] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError_("inventory row must be a mapping")
        path = row.get("path") or row.get("filename")
        size = row.get("bytes") if "bytes" in row else row.get("logical_bytes")
        kind = row.get("kind")
        if kind is None:
            kind = "weight" if str(path).endswith(".safetensors") else "control"
        files.append(
            FileEntry(
                path=str(path),
                bytes=int(size),
                sha256=row.get("sha256"),
                lfs_sha256=row.get("lfs_sha256") or row.get("hub_lfs_sha256"),
                kind=str(kind),
            )
        )
    return HashInventory(files=tuple(files), revision=revision)
