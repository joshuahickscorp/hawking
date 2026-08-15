"""Bind the unrun Q30 raw-final-logit successor executable.

This is a CPU/build evidence step only.  It fingerprints the compiled
diagnostic successor and its source against the sealed six-vector contract;
it never invokes the binary, loads a model, opens Metal/MPS, or touches the
live Q30 service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as retention_contract
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_CONTRACT = (
    DEFAULT_ROOT / "raw-final-logit-retention-successor/receipts"
    / "QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_"
    "07260bb96d09dab6ba7b0955c4f72da541404dfb5c38117dffe944173a9e8e34.json"
)
DEFAULT_BINARY = (
    REPO_ROOT
    / "workspace/ops/build/rust/debug/examples"
    / "ascension_qwen30_quality_repack_all_layer_current_trace_diagnostic"
)
DEFAULT_SOURCE = REPO_ROOT / "crates/hawking-core/examples/ascension_qwen30_quality_repack_all_layer_current_trace_diagnostic.rs"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "raw-final-logit-retention-successor/build-bindings"

SCHEMA = "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_executor_build_binding.v1"
STATUS = "PREPARED_RAW_FINAL_LOGIT_RETENTION_EXECUTOR_BUILD_BOUND_NOT_RUN"


class RawFinalLogitRetentionBuildBindingError(RuntimeError):
    """An unrun successor binary is not bound to the prepared contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise RawFinalLogitRetentionBuildBindingError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise RawFinalLogitRetentionBuildBindingError(f"{label} is not an object")
    return dict(checked)


def _regular(path: Path, *, label: str, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise RawFinalLogitRetentionBuildBindingError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RawFinalLogitRetentionBuildBindingError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RawFinalLogitRetentionBuildBindingError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise RawFinalLogitRetentionBuildBindingError(f"{label} must be executable")
    return path.resolve(strict=True)


def _evidence(path: Path) -> dict[str, Any]:
    stat_result = path.stat()
    return {"path": str(path), "bytes": stat_result.st_size, "sha256": _sha256_file(path)}


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_binding(*, contract_path: Path, binary_path: Path, source_path: Path) -> dict[str, Any]:
    contract_path = _regular(contract_path, label="raw retention contract")
    contract = _sealed(contract_path, label="raw retention contract")
    if contract.get("schema") != retention_contract.SCHEMA or contract.get("status") != retention_contract.STATUS:
        raise RawFinalLogitRetentionBuildBindingError("raw retention contract schema/status drifted")
    binary_path = _regular(binary_path, label="raw retention successor binary", executable=True)
    source_path = _regular(source_path, label="raw retention successor source")
    source = source_path.read_text(encoding="utf-8")
    if f'const RAW_FINAL_LOGIT_RETENTION_MODE: &str = "{retention_contract.NATIVE_MODE}";' not in source:
        raise RawFinalLogitRetentionBuildBindingError("source does not expose the named raw-final-logit retention mode")
    for required in (
        "checked_f32le_bytes",
        "raw_final_logit_payload",
        "all_four_replay_prior_98db_full_f32le_hashes",
        "source_teacher_capture_is_currently_blocked",
    ):
        if required not in source:
            raise RawFinalLogitRetentionBuildBindingError(f"source lacks required fail-closed retention guard {required!r}")
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "raw_final_logit_retention_contract": {
                **_evidence(contract_path),
                "seal_sha256": contract.get("seal_sha256"),
            },
            "executor_binary": _evidence(binary_path),
            "executor_source": _evidence(source_path),
            "mode": retention_contract.NATIVE_MODE,
            "execution_contract": {
                "requires_new_outer_receipt_last_capture_and_new_quiet_lease": True,
                "requires_four_direct_packed_raw_f32le_vectors": True,
                "requires_each_native_hash_to_match_immutable_prior_98db_witness": True,
                "requires_two_source_teacher_vectors_from_separate_source_memory_lease": True,
                "source_memory_preflight_currently_blocked_is_a_hard_runtime_refusal": True,
                "cannot_emit_six_vector_metric_or_quality_claim": True,
            },
            "claim_boundary": {
                "build_and_file_identity_only": True,
                "does_not_execute_the_binary": True,
                "does_not_open_source_weight_payloads_or_load_source_model": True,
                "does_not_create_metal_or_mps_context": True,
                "does_not_take_or_grant_gpu_or_memory_lease": True,
                "does_not_touch_live_qwen30_server_watcher_adapter_or_hcli": True,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_binding(contract_path=args.contract, binary_path=args.binary, source_path=args.source)
        if args.output is None:
            output = DEFAULT_OUTPUT_ROOT / f"QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_EXECUTOR_{result['executor_binary']['sha256']}.json"
        else:
            output = args.output
        _atomic_json(output, result)
    except RawFinalLogitRetentionBuildBindingError as exc:
        print(f"Q30 raw-final-logit retention build binding refused: {exc}")
        return 2
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
