"""Receipt seal + document integrity (single seal family; historical normalize)."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from lab.layout import resolve_workspace_path
from lab.semantic_taxonomy import (
    CONDENSE_OPERATION,
    SemanticTaxonomyError,
    normalize_semantic_tags,
)

RECEIPT_SCHEMA = "hawking.lab.receipt.v1"
GATE_EVIDENCE_SCHEMA = "hawking.lab.gate_evidence.v1"


class SealIntegrityError(ValueError):
    """Integrity failure; subclasses ValueError so receipt callers keep working."""


def _existing_workspace_path(path: str | Path) -> Path:
    """Use the compact workspace location when a historic path still names it.

    This is deliberately a lookup fallback.  Generic relative paths keep their
    caller-defined meaning (for example, a receipt beside a gate envelope).
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    resolved = resolve_workspace_path(candidate)
    return resolved if resolved.exists() else candidate


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SealIntegrityError(f"{label} must be a non-empty string")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise SealIntegrityError(f"{label} must be a 64-character sha256 or null")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SealIntegrityError(f"{label} must be a hexadecimal sha256 or null") from exc
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    data = value if isinstance(value, (bytes, bytearray)) else _canonical(value)
    return hashlib.sha256(data).hexdigest()


def seal(
    value: Mapping[str, Any],
    *,
    key: str = "seal_sha256",
    seal_key: str | None = None,
) -> dict[str, Any]:
    """Canonical seal. ``seal_key`` is accepted as an alias of ``key``."""
    k = seal_key if seal_key is not None else key
    unsigned = {kk: vv for kk, vv in value.items() if kk != k}
    return {**unsigned, k: _sha256_hex(unsigned)}


def verify(
    value: Mapping[str, Any],
    *,
    label: str = "document",
    key: str = "seal_sha256",
    seal_key: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SealIntegrityError(f"{label} is not a JSON object")
    k = seal_key if seal_key is not None else key
    recorded = value.get(k)
    expected = seal(dict(value), key=k)[k]
    if recorded != expected:
        raise SealIntegrityError(
            f"{label} seal mismatch: recorded={recorded!r} expected={expected}"
        )
    return dict(value)


# Historical names retained as aliases (assignments, not extra defs).
seal_document = seal
verify_document_seal = verify


def reject_resealed_substitution(
    observed: Mapping[str, Any],
    expected_builder: Callable[[], Mapping[str, Any]],
    *,
    label: str = "binding",
    match: str = "exact deterministic runtime",
) -> dict[str, Any]:
    verify(observed, label=label)
    expected = expected_builder()
    if _canonical(dict(observed)) != _canonical(dict(expected)):
        raise SealIntegrityError(f"{label} is not the {match}")
    return dict(observed)


def inspect_launcher_node(
    path: Path,
    *,
    label: str = "launcher",
    expected_mode: int | None = 0o755,
    require_single_hard_link: bool = True,
    refuse_symlink: bool = True,
) -> os.stat_result:
    clean = Path(path)
    try:
        st = os.lstat(clean)
    except OSError as exc:
        raise SealIntegrityError(f"cannot stat {label}: {exc}") from exc
    if refuse_symlink and stat.S_ISLNK(st.st_mode):
        raise SealIntegrityError(f"{label} must be a regular file, not a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise SealIntegrityError(f"{label} must be a regular file")
    if require_single_hard_link and st.st_nlink != 1:
        raise SealIntegrityError(
            f"{label} must not be a hard-link farm (nlink={st.st_nlink})"
        )
    if expected_mode is not None and stat.S_IMODE(st.st_mode) != expected_mode:
        raise SealIntegrityError(
            f"{label} mode must be {expected_mode:04o}, got {stat.S_IMODE(st.st_mode):04o}"
        )
    return st


def preflight_must_not_use_subprocess(
    *, subprocess_used: bool, label: str = "preflight"
) -> None:
    if subprocess_used:
        raise SealIntegrityError(f"{label} must not call subprocess")


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _git_commit(repo: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def read_jsonl_ledger(path: str | Path) -> Iterator[dict[str, Any]]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            yield row


def _declared_artifact_references(value: object) -> tuple[str, ...]:
    """Return document-declared references without probing the referenced data.

    Receipt normalisation must remain safe for historical documents.  This
    helper deliberately makes no path, digest, or network availability claim;
    it only gives the semantic taxonomy enough syntax to count what the record
    already declared.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def read_any_receipt(path: str | Path) -> dict[str, Any]:
    """Normalize a lab or legacy campaign receipt into a common shape."""
    path = _existing_workspace_path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"receipt root must be object: {path}")
    schema = str(raw.get("schema") or "")
    if schema == RECEIPT_SCHEMA or schema.startswith("hawking.lab."):
        out = dict(raw)
        out.setdefault("raw_schema", schema)
        try:
            out["semantic_tags"] = normalize_semantic_tags(
                raw.get("semantic_tags"),
                operation=CONDENSE_OPERATION,
                artifact_kind="lab_receipt",
                raw_schema=schema,
                missing_status=(
                    "historical_unlabeled" if raw.get("semantic_tags") is None else None
                ),
                artifact_references=_declared_artifact_references(raw.get("artifacts")),
            )
        except SemanticTaxonomyError as exc:
            raise SealIntegrityError(f"receipt semantic tags are invalid: {exc}") from exc
        return out
    status = str(raw.get("status") or raw.get("verdict") or "unknown")
    try:
        tags = normalize_semantic_tags(
            raw.get("semantic_tags"),
            operation=CONDENSE_OPERATION,
            artifact_kind="legacy_receipt",
            raw_schema=schema or None,
            missing_status="historical_unlabeled",
            artifact_references=_declared_artifact_references(raw.get("artifacts")),
        )
    except SemanticTaxonomyError as exc:
        raise SealIntegrityError(f"receipt semantic tags are invalid: {exc}") from exc
    return {
        "schema": RECEIPT_SCHEMA,
        "raw_schema": schema or "legacy",
        "campaign_id": raw.get("campaign_id") or raw.get("id") or "",
        "commit": raw.get("commit") or raw.get("git_commit") or "",
        "inputs": raw.get("inputs") or {},
        "method": raw.get("method") or raw.get("reproduction") or "",
        "measurement": raw.get("measurement") or raw.get("summary") or {},
        "verdict": status,
        "phase": raw.get("phase") or "",
        "at": raw.get("at") or raw.get("timestamp") or "",
        "artifacts": raw.get("artifacts") or [],
        "path": str(path),
        "seal_sha256": raw.get("seal_sha256") or "",
        "semantic_tags": tags,
    }


@dataclass
class Receipt:
    campaign_id: str
    verdict: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    method: Mapping[str, Any] = field(default_factory=dict)
    measurement: Mapping[str, Any] = field(default_factory=dict)
    commit: str = ""
    phase: str = ""
    status: str = ""
    at: str = ""
    schema: str = RECEIPT_SCHEMA
    reproduction: str = ""
    artifacts: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    semantic_tags: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        try:
            tags = normalize_semantic_tags(
                self.semantic_tags or None,
                operation=CONDENSE_OPERATION,
                artifact_kind="lab_receipt",
                raw_schema=self.schema,
                artifact_references=_declared_artifact_references(self.artifacts),
            )
        except SemanticTaxonomyError as exc:
            raise SealIntegrityError(f"receipt semantic tags are invalid: {exc}") from exc
        body = {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "commit": self.commit or _git_commit(),
            "inputs": dict(self.inputs),
            "method": dict(self.method),
            "measurement": dict(self.measurement),
            "verdict": self.verdict,
            "phase": self.phase or self.status,
            "status": self.status or self.verdict,
            "at": self.at or _utc_now(),
            "reproduction": self.reproduction,
            "artifacts": list(self.artifacts),
            "summary": dict(self.summary),
            "semantic_tags": tags,
        }
        return seal(body)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Receipt":
        verify(raw, label="receipt")
        schema = str(raw.get("schema") or RECEIPT_SCHEMA)
        try:
            tags = normalize_semantic_tags(
                raw.get("semantic_tags"),
                operation=CONDENSE_OPERATION,
                artifact_kind="lab_receipt",
                raw_schema=schema,
                missing_status=(
                    "historical_unlabeled" if raw.get("semantic_tags") is None else None
                ),
                artifact_references=_declared_artifact_references(raw.get("artifacts")),
            )
        except SemanticTaxonomyError as exc:
            raise SealIntegrityError(f"receipt semantic tags are invalid: {exc}") from exc
        return cls(
            campaign_id=str(raw.get("campaign_id") or ""),
            verdict=str(raw.get("verdict") or raw.get("status") or ""),
            inputs=dict(raw.get("inputs") or {}),
            method=dict(raw.get("method") or {}),
            measurement=dict(raw.get("measurement") or {}),
            commit=str(raw.get("commit") or ""),
            phase=str(raw.get("phase") or ""),
            status=str(raw.get("status") or ""),
            at=str(raw.get("at") or ""),
            schema=schema,
            reproduction=str(raw.get("reproduction") or ""),
            artifacts=tuple(str(x) for x in raw.get("artifacts") or ()),
            summary=dict(raw.get("summary") or {}),
            semantic_tags=tags,
        )


@dataclass(frozen=True)
class GateEvidence:
    """A sealed, independently reviewed gate outcome.

    A boolean in a campaign plan is a request to check a gate, not evidence that
    it passed.  This binds a result to its receipt, the exact source/artifact
    identities when applicable, the measurement mode and three distinct review
    roles.  The record deliberately does not interpret benchmark contents; its
    job is to make a later promotion re-check the immutable identity envelope.
    """

    gate_id: str
    result: str
    receipt_path: str
    receipt_sha256: str
    family: str
    model: str
    measurement_mode: str
    builder: str
    challenger: str
    verifier: str
    source_sha256: str | None = None
    artifact_sha256: str | None = None
    device: str | None = None
    lease_id: str | None = None
    schema: str = GATE_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "gate_id": self.gate_id,
            "result": self.result,
            "receipt_path": self.receipt_path,
            "receipt_sha256": self.receipt_sha256,
            "family": self.family,
            "model": self.model,
            "measurement_mode": self.measurement_mode,
            "builder": self.builder,
            "challenger": self.challenger,
            "verifier": self.verifier,
            "source_sha256": self.source_sha256,
            "artifact_sha256": self.artifact_sha256,
            "device": self.device,
            "lease_id": self.lease_id,
        }
        self._validate(body)
        return seal(body)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GateEvidence":
        value = verify(raw, label="gate evidence")
        cls._validate(value)
        return cls(
            gate_id=str(value["gate_id"]),
            result=str(value["result"]),
            receipt_path=str(value["receipt_path"]),
            receipt_sha256=str(value["receipt_sha256"]),
            family=str(value["family"]),
            model=str(value["model"]),
            measurement_mode=str(value["measurement_mode"]),
            builder=str(value["builder"]),
            challenger=str(value["challenger"]),
            verifier=str(value["verifier"]),
            source_sha256=value.get("source_sha256"),
            artifact_sha256=value.get("artifact_sha256"),
            device=value.get("device"),
            lease_id=value.get("lease_id"),
            schema=str(value["schema"]),
        )

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> None:
        if value.get("schema") != GATE_EVIDENCE_SCHEMA:
            raise SealIntegrityError(
                f"gate evidence schema must be {GATE_EVIDENCE_SCHEMA!r}"
            )
        for key in (
            "gate_id",
            "receipt_path",
            "family",
            "model",
            "measurement_mode",
            "builder",
            "challenger",
            "verifier",
        ):
            _required_string(value.get(key), f"gate evidence.{key}")
        if value.get("result") not in {"PASS", "FAIL"}:
            raise SealIntegrityError("gate evidence.result must be PASS or FAIL")
        receipt_sha256 = value.get("receipt_sha256")
        if receipt_sha256 is None:
            raise SealIntegrityError("gate evidence.receipt_sha256 is required")
        _optional_sha256(receipt_sha256, "gate evidence.receipt_sha256")
        _optional_sha256(value.get("source_sha256"), "gate evidence.source_sha256")
        _optional_sha256(value.get("artifact_sha256"), "gate evidence.artifact_sha256")
        if value.get("builder") == value.get("challenger"):
            raise SealIntegrityError("gate evidence.builder and challenger must differ")
        if value.get("verifier") in {value.get("builder"), value.get("challenger")}:
            raise SealIntegrityError("gate evidence.verifier must be independent")


class ReceiptAuthority:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, campaign_id: str) -> Path:
        safe = campaign_id.replace("/", "_")
        return self.root / f"{safe}.receipt.json"

    def gate_path_for(self, gate_id: str) -> Path:
        safe = gate_id.replace("/", "_")
        return self.root / "gates" / f"{safe}.gate.json"

    def write(self, receipt: Receipt | Mapping[str, Any]) -> Path:
        if isinstance(receipt, Receipt):
            payload = receipt.to_dict()
            campaign_id = receipt.campaign_id
        else:
            payload = seal(dict(receipt))
            campaign_id = str(payload.get("campaign_id") or "unknown")
        path = self.path_for(campaign_id)
        _atomic_write_text(
            path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        return path

    def read(self, campaign_id: str) -> dict[str, Any]:
        path = self.path_for(campaign_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return verify(raw, label=str(path))

    def read_path(self, path: Path) -> dict[str, Any]:
        return read_any_receipt(Path(path))

    def write_gate_evidence(self, evidence: GateEvidence | Mapping[str, Any]) -> Path:
        payload = evidence.to_dict() if isinstance(evidence, GateEvidence) else dict(evidence)
        checked = GateEvidence.from_dict(payload)
        path = self.gate_path_for(checked.gate_id)
        _atomic_write_text(
            path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        return path

    def read_gate_evidence(self, path: str | Path, *, expected_gate: str | None = None) -> GateEvidence:
        evidence_path = _existing_workspace_path(path)
        raw = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence = GateEvidence.from_dict(raw)
        if expected_gate is not None and evidence.gate_id != expected_gate:
            raise SealIntegrityError(
                f"gate evidence {evidence_path} is for {evidence.gate_id!r}, "
                f"not required gate {expected_gate!r}"
            )
        receipt_path = Path(evidence.receipt_path)
        if not receipt_path.is_absolute():
            workspace_receipt = _existing_workspace_path(receipt_path)
            receipt_path = (
                workspace_receipt
                if workspace_receipt.is_absolute()
                else (evidence_path.parent / receipt_path).resolve()
            )
        if not receipt_path.is_file():
            raise SealIntegrityError(
                f"gate evidence {evidence.gate_id!r} names missing receipt {receipt_path}"
            )
        raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verified_receipt = verify(raw_receipt, label=f"gate receipt {receipt_path}")
        if verified_receipt.get("seal_sha256") != evidence.receipt_sha256:
            raise SealIntegrityError(
                f"gate evidence {evidence.gate_id!r} receipt seal does not match"
            )
        return evidence
