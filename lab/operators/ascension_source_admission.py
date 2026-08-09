"""Metadata-only Hugging Face source admission for Ascension's two managers.

This is deliberately a controller-side transport surface.  It can use an
already authenticated Hugging Face credential store to pin a public/gated
source revision, enumerate exact files, and fetch only tiny control files such
as ``config.json`` and a license.  It never returns a token, writes one to a
receipt, downloads a model body, starts a model runtime, or changes a
qualification state.

The resulting documents are *candidate metadata*, not controller-certified
manager source receipts.  They give the later protected source/architecture
gate an exact, reproducible input rather than a floating Hub ``main`` branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lab.operators.ascension_lifecycle import DEFAULT_ROOT, MODEL_30B, MODEL_80B
from lab.receipts import seal


SCHEMA = "hawking.ascension.source_admission_candidate.v1"
SUMMARY_SCHEMA = "hawking.ascension.source_admission_summary.v1"


class SourceAdmissionError(RuntimeError):
    """A metadata request cannot produce a trustworthy candidate document."""


@dataclass(frozen=True)
class SourceTarget:
    key: str
    artifact_id: str
    model_id: str
    repository: str
    family: str
    role: str


TARGETS: dict[str, SourceTarget] = {
    "qwen30": SourceTarget(
        key="qwen30",
        artifact_id="QWEN30_SOURCE_METADATA_CANDIDATE",
        model_id=MODEL_30B,
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        family="QWEN3_MOE",
        role="executor",
    ),
    "qwen80": SourceTarget(
        key="qwen80",
        artifact_id="QWEN80_SOURCE_METADATA_CANDIDATE",
        model_id=MODEL_80B,
        repository="Qwen/Qwen3-Coder-Next",
        family="QWEN3_NEXT",
        role="reviewer",
    ),
}


@dataclass(frozen=True)
class SourceAdmissionPaths:
    root: Path
    records_root: Path
    cache_root: Path
    summary_path: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "SourceAdmissionPaths":
        resolved = Path(root).expanduser().resolve()
        records = resolved / "source-admission"
        return cls(
            root=resolved,
            records_root=records,
            cache_root=records / "metadata-cache",
            summary_path=records / "SOURCE_ADMISSION_STATUS.json",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _regular_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise SourceAdmissionError(f"source-admission path must be a real directory: {path}")
    os.chmod(path, 0o750)


def bootstrap_layout(root: str | Path = DEFAULT_ROOT) -> SourceAdmissionPaths:
    paths = SourceAdmissionPaths.from_root(root)
    _regular_directory(paths.root)
    _regular_directory(paths.records_root)
    _regular_directory(paths.cache_root)
    return paths


def _hub_client() -> tuple[Any, Callable[..., str]]:
    """Import hub transport lazily so controller-only tests require no network."""

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment integration
        raise SourceAdmissionError("huggingface_hub is required for source metadata capture") from exc
    return HfApi(), hf_hub_download


def _safe_authentication_state(client: Any) -> dict[str, Any]:
    """Probe authentication without recording account identity or secret material."""

    try:
        client.whoami(token=True)
    except Exception as exc:  # Public sources remain queryable without auth.
        return {
            "authenticated": False,
            "credential_source": "none_or_unavailable",
            "detail": f"authentication unavailable: {type(exc).__name__}",
            "token_material_recorded": False,
        }
    return {
        "authenticated": True,
        "credential_source": "huggingface_local_credential_store",
        "token_material_recorded": False,
        "token_export_forbidden": True,
    }


def _file_kind(name: str) -> str:
    if name.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth")):
        return "weight"
    if name in {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}:
        return "control"
    if name.lower().startswith("license"):
        return "license"
    return "other"


def _sibling_row(item: Any) -> dict[str, Any]:
    name = str(getattr(item, "rfilename", ""))
    lfs = getattr(item, "lfs", None)
    oid = getattr(lfs, "oid", None) if lfs is not None else None
    size = getattr(item, "size", None)
    return {
        "path": name,
        "bytes": int(size) if isinstance(size, int) and size >= 0 else None,
        "lfs_sha256": str(oid).lower() if isinstance(oid, str) and len(oid) == 64 else None,
        "kind": _file_kind(name),
    }


def _download_control_file(
    downloader: Callable[..., str],
    *,
    repository: str,
    revision: str,
    filename: str,
    cache_root: Path,
) -> Path | None:
    try:
        location = downloader(
            repo_id=repository,
            filename=filename,
            revision=revision,
            token=True,
            cache_dir=str(cache_root),
        )
    except Exception:
        return None
    path = Path(location)
    # ``huggingface_hub`` materializes snapshots as symlinks into its own blob
    # cache.  Rejecting every symlink would therefore make a perfectly valid
    # metadata-only source look incomplete.  Resolve it, but only accept a
    # regular target that remains inside this controller-owned cache root.
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(cache_root.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def capture_source_metadata(
    target: SourceTarget,
    *,
    root: str | Path = DEFAULT_ROOT,
    client: Any | None = None,
    downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Capture an immutable source candidate using controller-owned transport.

    ``token=True`` asks the Hub client to use its configured secure credential
    store when present.  The credential is never read into this module's
    serialized document, passed to a model, or printed.
    """

    paths = bootstrap_layout(root)
    if client is None or downloader is None:
        imported_client, imported_downloader = _hub_client()
        client = client or imported_client
        downloader = downloader or imported_downloader

    authentication = _safe_authentication_state(client)
    try:
        info = client.model_info(target.repository, revision="main", files_metadata=True, token=True)
    except Exception as exc:
        document = seal(
            {
                "schema": SCHEMA,
                "artifact_id": target.artifact_id,
                "status": "METADATA_CAPTURE_FAILED",
                "authority_level": "candidate",
                "recorded_at": _utc_now(),
                "target": {
                    "model_id": target.model_id,
                    "repository": target.repository,
                    "family": target.family,
                    "role": target.role,
                },
                "authentication": authentication,
                # Error messages from third-party transports are not a safe
                # receipt surface: an implementation could echo a credential
                # or signed URL.  Retain only the machine-readable class.
                "error": {"type": type(exc).__name__, "message": "Hub metadata request failed"},
                "claim_boundary": {
                    "no_model_body_downloaded": True,
                    "no_token_material_recorded": True,
                    "not_controller_certified": True,
                },
            }
        )
        _atomic_json(paths.records_root / f"{target.artifact_id}.json", document)
        return document

    revision = str(getattr(info, "sha", "") or "").lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise SourceAdmissionError(f"Hub returned no immutable 40-character revision for {target.repository}")
    siblings = [_sibling_row(item) for item in list(getattr(info, "siblings", ()) or ())]
    siblings.sort(key=lambda item: str(item["path"]))
    file_names = {str(item["path"]) for item in siblings}
    config_path = _download_control_file(
        downloader,
        repository=target.repository,
        revision=revision,
        filename="config.json",
        cache_root=paths.cache_root,
    )
    config: dict[str, Any] | None = None
    config_digest: str | None = None
    if config_path is not None:
        try:
            decoded = json.loads(config_path.read_text(encoding="utf-8"))
            config = dict(decoded) if isinstance(decoded, Mapping) else None
            config_digest = _sha256(config_path)
        except (OSError, json.JSONDecodeError):
            config = None

    license_path: Path | None = None
    for filename in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if filename in file_names:
            license_path = _download_control_file(
                downloader,
                repository=target.repository,
                revision=revision,
                filename=filename,
                cache_root=paths.cache_root,
            )
            if license_path is not None:
                break

    card_data = getattr(info, "cardData", None)
    card_license = None
    if isinstance(card_data, Mapping):
        value = card_data.get("license")
        card_license = str(value) if isinstance(value, str) and value.strip() else None
    weight_files = [row for row in siblings if row["kind"] == "weight"]
    known_weight_bytes = sum(int(row["bytes"] or 0) for row in weight_files)
    unknown_size_count = sum(1 for row in siblings if row["bytes"] is None)
    document = seal(
        {
            "schema": SCHEMA,
            "artifact_id": target.artifact_id,
            "status": "CANDIDATE_METADATA_CAPTURED",
            "authority_level": "candidate",
            "recorded_at": _utc_now(),
            "target": {
                "model_id": target.model_id,
                "repository": target.repository,
                "family": target.family,
                "role": target.role,
            },
            "source": {
                "repository": target.repository,
                "revision": revision,
                "revision_requested": "main",
                "official_namespace_expected": target.repository.startswith("Qwen/"),
                "private": bool(getattr(info, "private", False)),
                "gated": bool(getattr(info, "gated", False)),
                "license_card_value": card_license,
                "license_file": license_path.name if license_path else None,
                "license_file_sha256": _sha256(license_path) if license_path else None,
            },
            "inventory": {
                "files": siblings,
                "file_count": len(siblings),
                "weight_file_count": len(weight_files),
                "known_weight_bytes": known_weight_bytes,
                "unknown_size_count": unknown_size_count,
            },
            "architecture": {
                "config_sha256": config_digest,
                "model_type": config.get("model_type") if config else None,
                "architectures": config.get("architectures") if config else None,
                "torch_dtype": config.get("torch_dtype") if config else None,
                "hidden_size": config.get("hidden_size") if config else None,
                "num_hidden_layers": config.get("num_hidden_layers") if config else None,
                "num_experts": config.get("num_experts") if config else None,
                "num_experts_per_tok": config.get("num_experts_per_tok") if config else None,
                "vocab_size": config.get("vocab_size") if config else None,
                "config_captured": config is not None,
            },
            "authentication": authentication,
            "claim_boundary": {
                "metadata_only": True,
                "no_model_body_downloaded": True,
                "no_model_loaded": True,
                "no_token_material_recorded": True,
                "token_never_exposed_to_model": True,
                "not_controller_certified": True,
                "not_permission_to_stream_model_body": True,
            },
        }
    )
    _atomic_json(paths.records_root / f"{target.artifact_id}.json", document)
    return document


def capture_all_sources(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Capture both manager source candidates in fixed Bible order."""

    paths = bootstrap_layout(root)
    documents = [capture_source_metadata(target, root=paths.root) for target in TARGETS.values()]
    all_captured = all(document.get("status") == "CANDIDATE_METADATA_CAPTURED" for document in documents)
    summary = seal(
        {
            "schema": SUMMARY_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "ALL_METADATA_CAPTURED" if all_captured else "METADATA_CAPTURE_PARTIAL",
            "candidate_order": [target.model_id for target in TARGETS.values()],
            "records": [
                {
                    "artifact_id": document["artifact_id"],
                    "status": document["status"],
                    "repository": document["target"]["repository"],
                    "revision": (document.get("source") or {}).get("revision"),
                    "record_path": str(paths.records_root / f"{document['artifact_id']}.json"),
                    "seal_sha256": document["seal_sha256"],
                    "no_model_body_downloaded": (document.get("claim_boundary") or {}).get(
                        "no_model_body_downloaded"
                    ) is True,
                }
                for document in documents
            ],
            "claim_boundary": {
                "records_are_candidate_metadata_not_qualification": True,
                "no_token_material_recorded": True,
                "no_model_body_downloaded": True,
            },
        }
    )
    _atomic_json(paths.summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="capture metadata for a manager source target")
    capture.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    capture.add_argument("--target", choices=tuple(TARGETS) + ("all",), default="all")
    status = sub.add_parser("status", help="print the latest source-admission summary")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        if args.target == "all":
            result = capture_all_sources(args.root)
        else:
            result = capture_source_metadata(TARGETS[args.target], root=args.root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        paths = SourceAdmissionPaths.from_root(args.root)
        try:
            result = json.loads(paths.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(json.dumps({"state": "ABSENT", "summary_path": str(paths.summary_path)}))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unknown source-admission command {args.command!r}")


__all__ = [
    "MODEL_30B",
    "MODEL_80B",
    "SCHEMA",
    "SUMMARY_SCHEMA",
    "SourceAdmissionError",
    "SourceAdmissionPaths",
    "SourceTarget",
    "TARGETS",
    "bootstrap_layout",
    "capture_all_sources",
    "capture_source_metadata",
]


if __name__ == "__main__":
    raise SystemExit(main())
