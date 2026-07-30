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

RECEIPT_SCHEMA = "hawking.lab.receipt.v1"


class SealIntegrityError(ValueError):
    """Integrity failure; subclasses ValueError so receipt callers keep working."""


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


def read_any_receipt(path: str | Path) -> dict[str, Any]:
    """Normalize a lab or legacy campaign receipt into a common shape."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"receipt root must be object: {path}")
    schema = str(raw.get("schema") or "")
    if schema == RECEIPT_SCHEMA or schema.startswith("hawking.lab."):
        out = dict(raw)
        out.setdefault("raw_schema", schema)
        return out
    status = str(raw.get("status") or raw.get("verdict") or "unknown")
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

    def to_dict(self) -> dict[str, Any]:
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
        }
        return seal(body)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Receipt":
        verify(raw, label="receipt")
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
            schema=str(raw.get("schema") or RECEIPT_SCHEMA),
            reproduction=str(raw.get("reproduction") or ""),
            artifacts=tuple(str(x) for x in raw.get("artifacts") or ()),
            summary=dict(raw.get("summary") or {}),
        )


class ReceiptAuthority:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, campaign_id: str) -> Path:
        safe = campaign_id.replace("/", "_")
        return self.root / f"{safe}.receipt.json"

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
