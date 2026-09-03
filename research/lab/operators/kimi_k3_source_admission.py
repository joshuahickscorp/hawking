#!/usr/bin/env python3.12
"""Official metadata-only admission for the Kimi K3 source.

The Kimi grandparent lane must not start from a press report or be confused
with the older Kimi K2.6 artifact.  This operator binds the official public
Hub revision, license, control assets, safetensors shard inventory, and LFS
SHA-256 identities without materialising a weight shard or retaining a Hub/Xet
cache.  It is source provenance only -- never permission to acquire teacher
traces or launch a parent-model stream.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
EVIDENCE_ROOT = WORKSPACE_ROOT / "campaign" / "evidence" / "models" / "kimi-k3"
RUNTIME_ROOT = WORKSPACE_ROOT / "campaign" / "evidence" / "runtime" / "kimi-k3-admission"

REPOSITORY = "moonshotai/Kimi-K3"
ADMISSION_SCHEMA = "hawking.kimi_k3.official_source_admission.v1"
ADMISSION_STATUS = "KIMI_K3_OFFICIAL_SOURCE_ADMITTED_METADATA_ONLY"
ADMISSION_NAME = "KIMI_K3_SOURCE_ADMISSION.json"
MIN_FREE_FLOOR_BYTES = 15 * 1024**3
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_RE = re.compile(r"^model-\d{5}-of-\d{6}\.safetensors$")


class KimiK3SourceAdmissionError(RuntimeError):
    """The official K3 source could not be safely and immutably bound."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise KimiK3SourceAdmissionError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise KimiK3SourceAdmissionError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise KimiK3SourceAdmissionError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path, label: str) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise KimiK3SourceAdmissionError(f"{label} must be a non-symlink directory")
        return
    path.mkdir(parents=True, exist_ok=False)


def _atomic_create(path: Path, raw: bytes) -> str:
    if path.exists():
        _regular_file(path, "existing admission receipt")
        existing = path.read_bytes()
        if existing != raw:
            raise KimiK3SourceAdmissionError(
                f"refusing to overwrite a different immutable admission receipt: {path}"
            )
        return _sha256(existing)
    _ensure_dir(path.parent, "admission receipt parent")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


def _floor_check(path: Path) -> None:
    try:
        free = shutil.disk_usage(path).free
    except OSError as exc:
        raise KimiK3SourceAdmissionError(f"cannot measure free space: {exc}") from exc
    if free < MIN_FREE_FLOOR_BYTES:
        raise KimiK3SourceAdmissionError("15 GiB free-space floor is not satisfied")


def _configure_environment(runtime_root: Path) -> dict[str, str]:
    """Set all relevant Hub/Xet variables before the lazy Hub import."""

    if "huggingface_hub" in sys.modules or "hf_xet" in sys.modules:
        raise KimiK3SourceAdmissionError(
            "Kimi source admission must set Hub/Xet environment before importing Hugging Face modules"
        )
    _ensure_dir(runtime_root, "Kimi admission runtime root")
    paths = {
        "HF_HOME": runtime_root / "hf-home",
        "HF_HUB_CACHE": runtime_root / "hub-cache",
        "HF_XET_CACHE": runtime_root / "xet-cache",
    }
    for path in paths.values():
        _ensure_dir(path, f"Kimi admission cache root {path.name}")
    environment = {
        "HF_HOME": str(paths["HF_HOME"]),
        "HF_HUB_CACHE": str(paths["HF_HUB_CACHE"]),
        "HF_XET_CACHE": str(paths["HF_XET_CACHE"]),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "false",
        "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE": "true",
    }
    os.environ.update(environment)
    return environment


def _fetch_bytes(url: str, *, label: str, expected_bytes: int | None) -> bytes:
    """Fetch a small official control asset into RAM only; no token is sent."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hawking-kimi-k3-source-admission/1", "Accept": "*/*"},
    )
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            if expected_bytes is not None and len(body) != expected_bytes:
                raise KimiK3SourceAdmissionError(
                    f"{label} byte count mismatch: expected {expected_bytes}, got {len(body)}"
                )
            return body
        except (OSError, KimiK3SourceAdmissionError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(0.25 * (2**attempt))
    raise KimiK3SourceAdmissionError(f"cannot fetch {label}: {error}")


def _cache_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(root)))
    return sorted(files)


def _lfs_sha256(sibling: object) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    value = getattr(lfs, "sha256", None) if lfs is not None else None
    return value if isinstance(value, str) and len(value) == 64 else None


def _sibling_size(sibling: object) -> int | None:
    value = getattr(sibling, "size", None)
    return value if isinstance(value, int) and value >= 0 else None


def _card_license(info: object) -> str | None:
    card = getattr(info, "card_data", None)
    if isinstance(card, Mapping):
        value = card.get("license")
    else:
        value = getattr(card, "license", None)
    return value if isinstance(value, str) else None


def _metadata_asset(
    *,
    filename: str,
    siblings: Mapping[str, object],
    hf_hub_url: Any,
    revision: str,
) -> tuple[dict[str, Any], bytes]:
    sibling = siblings.get(filename)
    if sibling is None:
        raise KimiK3SourceAdmissionError(f"official source lacks required control asset {filename}")
    size = _sibling_size(sibling)
    raw = _fetch_bytes(
        hf_hub_url(REPOSITORY, filename=filename, revision=revision),
        label=filename,
        expected_bytes=size,
    )
    digest = _sha256(raw)
    expected_lfs = _lfs_sha256(sibling)
    if expected_lfs is not None and digest != expected_lfs:
        raise KimiK3SourceAdmissionError(
            f"official control asset SHA mismatch for {filename}: {digest} != {expected_lfs}"
        )
    return {
        "bytes": len(raw),
        "sha256": digest,
        "hub_lfs_sha256": expected_lfs,
        "lfs_identity_verified": expected_lfs is None or expected_lfs == digest,
    }, raw


def build_admission(
    *,
    revision: str = "main",
    workspace: str | Path = WORKSPACE_ROOT,
    runtime_root: str | Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    """Bind the official K3 source identity without transferring model bodies."""

    workspace_path = _absolute(workspace, "workspace")
    runtime = _absolute(runtime_root, "runtime root")
    if not workspace_path.is_dir():
        raise KimiK3SourceAdmissionError(f"workspace is not a directory: {workspace_path}")
    _floor_check(workspace_path)
    environment = _configure_environment(runtime)

    # Import strictly after setting the Xet environment.
    from huggingface_hub import HfApi, hf_hub_url  # type: ignore[import-not-found]

    try:
        info = HfApi().model_info(REPOSITORY, revision=revision, files_metadata=True)
    except Exception as exc:  # Hugging Face exposes several transport exception classes.
        raise KimiK3SourceAdmissionError(f"cannot query official Kimi K3 source: {exc}") from exc
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise KimiK3SourceAdmissionError(f"official K3 revision is not an immutable commit: {commit!r}")
    if getattr(info, "private", None) is not False or getattr(info, "gated", None) not in (False, None):
        raise KimiK3SourceAdmissionError("Kimi K3 source is not an ungated public source")

    sibling_values = list(getattr(info, "siblings", None) or [])
    siblings: dict[str, object] = {}
    for sibling in sibling_values:
        name = getattr(sibling, "rfilename", None)
        if not isinstance(name, str) or not name or name in siblings:
            raise KimiK3SourceAdmissionError("official K3 source has malformed or duplicate sibling names")
        siblings[name] = sibling
    if not siblings:
        raise KimiK3SourceAdmissionError("official K3 source exposes no files")

    required_controls = (
        "LICENSE",
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        "tiktoken.model",
        "tokenizer_config.json",
    )
    metadata_assets: dict[str, Any] = {}
    control_bytes = 0
    raw_controls: dict[str, bytes] = {}
    for filename in required_controls:
        entry, raw = _metadata_asset(
            filename=filename, siblings=siblings, hf_hub_url=hf_hub_url, revision=commit
        )
        metadata_assets[filename] = entry
        raw_controls[filename] = raw
        control_bytes += len(raw)

    try:
        config = json.loads(raw_controls["config.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KimiK3SourceAdmissionError(f"cannot parse official Kimi K3 config: {exc}") from exc
    if not isinstance(config, Mapping):
        raise KimiK3SourceAdmissionError("official Kimi K3 config is not an object")
    if config.get("model_type") != "kimi_k3":
        raise KimiK3SourceAdmissionError("official Kimi K3 config does not declare model_type kimi_k3")
    try:
        index = json.loads(raw_controls["model.safetensors.index.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KimiK3SourceAdmissionError(f"cannot parse official Kimi K3 index: {exc}") from exc
    if not isinstance(index, Mapping):
        raise KimiK3SourceAdmissionError("official Kimi K3 safetensors index is not an object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise KimiK3SourceAdmissionError("official Kimi K3 index has no weight map")

    weights: list[dict[str, Any]] = []
    for name, sibling in sorted(siblings.items()):
        if not _WEIGHT_RE.fullmatch(name):
            continue
        size = _sibling_size(sibling)
        digest = _lfs_sha256(sibling)
        if size is None or digest is None:
            raise KimiK3SourceAdmissionError(f"Kimi K3 weight lacks LFS size/hash identity: {name}")
        weights.append({"path": name, "bytes": size, "lfs_sha256": digest})
    if len(weights) < 2:
        raise KimiK3SourceAdmissionError("official K3 source has an implausible safetensors shard inventory")
    declared_weight_files = {value for value in weight_map.values() if isinstance(value, str)}
    indexed_weights = {weight["path"] for weight in weights}
    if declared_weight_files != indexed_weights:
        raise KimiK3SourceAdmissionError(
            "Kimi K3 safetensors index does not exactly match the official shard inventory"
        )

    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise KimiK3SourceAdmissionError("official K3 config lacks text_config")
    architecture = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "text_model_type": text_config.get("model_type"),
        "text_architectures": text_config.get("architectures"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "hidden_size": text_config.get("hidden_size"),
        "n_routed_experts": text_config.get("n_routed_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "torch_dtype": config.get("dtype"),
    }
    hub_cache_files = _cache_files(Path(environment["HF_HUB_CACHE"]))
    xet_cache_files = _cache_files(Path(environment["HF_XET_CACHE"]))
    # huggingface_hub may record an agent-harness capability table beneath
    # HF_HOME. It contains no source bytes, and is not a Hub/Xet cache. Keep
    # it visible in the receipt but fail on any other unexpected home payload.
    home_runtime_files = _cache_files(Path(environment["HF_HOME"]))
    non_source_runtime_files = [
        name for name in home_runtime_files if name == ".agent_harnesses.json"
    ]
    if set(home_runtime_files) != set(non_source_runtime_files):
        raise KimiK3SourceAdmissionError(
            "metadata-only Kimi K3 admission wrote an unexpected HF_HOME payload"
        )
    if hub_cache_files or xet_cache_files:
        raise KimiK3SourceAdmissionError(
            "metadata-only Kimi K3 admission unexpectedly wrote a local Hub/Xet cache"
        )
    admission = {
        "schema": ADMISSION_SCHEMA,
        "status": ADMISSION_STATUS,
        "source": {
            "repository": REPOSITORY,
            "revision": commit,
            "requested_revision": revision,
            "private": False,
            "gated": False,
            "hub_card_license": _card_license(info),
            "license_file_sha256": metadata_assets["LICENSE"]["sha256"],
            "license_file_bytes": metadata_assets["LICENSE"]["bytes"],
            "source_file_count": len(siblings),
            "weight_shard_count": len(weights),
            "weight_shard_bytes": sum(weight["bytes"] for weight in weights),
            "weight_shards": weights,
            "index_weight_entry_count": len(weight_map),
            "metadata_assets": metadata_assets,
            "architecture_facts": architecture,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
            "hf_xet": importlib.metadata.version("hf_xet"),
            "environment_set_before_hub_import": True,
            "environment": environment,
        },
        "storage": {
            "hard_free_floor_bytes": MIN_FREE_FLOOR_BYTES,
            "control_metadata_downloaded_in_memory_bytes": control_bytes,
            "source_body_persisted": False,
            "persistent_hub_cache_files": hub_cache_files,
            "persistent_xet_cache_files": xet_cache_files,
            "non_source_runtime_files": non_source_runtime_files,
            "full_weight_shards_downloaded": 0,
        },
        "claim_boundary": {
            "metadata_only": True,
            "weight_payload_verified_by_lfs_identity_only": True,
            "weights_materialized_or_runtime_loaded": False,
            "teacher_trace_acquisition_authorized": False,
            "parent_restream_authorized": False,
            "does_not_override_ramanujan_completion_gate": True,
            "does_not_establish_local_runtime_or_capability": True,
        },
    }
    return seal(admission)


def validate_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        document = verify(value, label="Kimi K3 source admission")
    except SealIntegrityError as exc:
        raise KimiK3SourceAdmissionError(str(exc)) from exc
    if document.get("schema") != ADMISSION_SCHEMA or document.get("status") != ADMISSION_STATUS:
        raise KimiK3SourceAdmissionError("Kimi K3 source admission has the wrong schema or status")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise KimiK3SourceAdmissionError("Kimi K3 source admission lacks source binding")
    if source.get("repository") != REPOSITORY or _COMMIT_RE.fullmatch(str(source.get("revision"))) is None:
        raise KimiK3SourceAdmissionError("Kimi K3 source admission lacks an immutable official identity")
    if source.get("private") is not False or source.get("gated") is not False:
        raise KimiK3SourceAdmissionError("Kimi K3 source admission is not public and ungated")
    weights = source.get("weight_shards")
    if not isinstance(weights, list) or len(weights) < 2:
        raise KimiK3SourceAdmissionError("Kimi K3 source admission lacks a shard inventory")
    if any(
        not isinstance(row, Mapping)
        or not _WEIGHT_RE.fullmatch(str(row.get("path")))
        or not isinstance(row.get("bytes"), int)
        or not isinstance(row.get("lfs_sha256"), str)
        for row in weights
    ):
        raise KimiK3SourceAdmissionError("Kimi K3 source admission has malformed shard identity")
    storage = document.get("storage")
    claims = document.get("claim_boundary")
    if not isinstance(storage, Mapping) or not isinstance(claims, Mapping):
        raise KimiK3SourceAdmissionError("Kimi K3 source admission lacks storage/claim boundaries")
    if storage.get("source_body_persisted") is not False or storage.get("full_weight_shards_downloaded") != 0:
        raise KimiK3SourceAdmissionError("Kimi K3 admission claims a persisted or downloaded weight body")
    if storage.get("persistent_hub_cache_files") != [] or storage.get("persistent_xet_cache_files") != []:
        raise KimiK3SourceAdmissionError("Kimi K3 admission claims a persistent Hub/Xet cache")
    if claims.get("teacher_trace_acquisition_authorized") is not False:
        raise KimiK3SourceAdmissionError("Kimi K3 admission incorrectly authorizes teacher traces")
    return document


def write_admission(
    *,
    revision: str = "main",
    workspace: str | Path = WORKSPACE_ROOT,
    runtime_root: str | Path = RUNTIME_ROOT,
    out: str | Path = EVIDENCE_ROOT / ADMISSION_NAME,
) -> dict[str, Any]:
    document = build_admission(revision=revision, workspace=workspace, runtime_root=runtime_root)
    validate_admission(document)
    output = _absolute(out, "admission output")
    _atomic_create(output, _canonical(document) + b"\n")
    return {
        "status": document["status"],
        "path": str(output),
        "seal_sha256": document["seal_sha256"],
        "repository": document["source"]["repository"],
        "revision": document["source"]["revision"],
        "weight_shard_count": document["source"]["weight_shard_count"],
        "weight_shard_bytes": document["source"]["weight_shard_bytes"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--out", type=Path, default=EVIDENCE_ROOT / ADMISSION_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = write_admission(
            revision=args.revision,
            workspace=args.workspace,
            runtime_root=args.runtime_root,
            out=args.out,
        )
    except KimiK3SourceAdmissionError as exc:
        raise SystemExit(f"Kimi K3 source admission error: {exc}") from exc
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
