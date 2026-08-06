#!/usr/bin/env python3
"""Seal a campaign-canonical wrapper around a raw Rust SIMDgroup sweep.

The bounded Metal sweep intentionally writes its own Rust/``serde_json``
producer receipt.  That producer's self-seal is useful as an opaque raw-run
field, but it is not byte-compatible with :mod:`lab.receipts` canonical JSON.
This utility never rewrites that raw evidence.  Instead it validates the
required raw component metrics, hashes the exact raw bytes, mirrors the
baseline-admission fields, and seals a new Python-canonical wrapper with
``lab.receipts.seal``.

The wrapper remains strictly raw-weight component evidence: it expressly does
not turn act_quant, source-forward parity, a V4 runtime, HCLI, or BASE_TRUE_TPS
into a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify  # noqa: E402


SCHEMA = "hawking.gravity.deepseek_v4.raw_weight_simdgroup_splitk_sweep.v1"
STATUS = "PASS_REAL_M3_METAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_NOT_SOURCE_FORWARD_OR_RUNTIME"
_REQUIRED_SCOPE_TRUE = (
    "raw_weight_component_only",
    "not_source_forward_parity",
    "not_a_full_model_load",
    "not_a_full_43_layer_runtime_adapter",
    "not_a_token_or_generation",
    "not_a_BASE_TRUE_TPS_measurement",
    "not_a_runtime_kernel_promotion",
    "same_sealed_full_gravity_artifact_before_and_after",
    "same_deterministic_input_and_raw_weight_cpu_reference_before_and_after",
)
_SOURCE_PATHS = (
    "crates/hawking-core/examples/gravity_deepseek_v4_simdgroup_splitk_sweep.rs",
    "crates/hawking-core/shaders/matmul.metal",
    "crates/hawking-core/src/metal/mod.rs",
    "Cargo.lock",
)


class SweepSealError(ValueError):
    """The raw run cannot be safely wrapped as campaign evidence."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regular_file(path: Path, label: str) -> Path:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise SweepSealError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise SweepSealError(f"{label} must be a regular non-symlink file")
    return path.resolve()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SweepSealError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SweepSealError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    rendered = _string(value, label)
    if len(rendered) != 64:
        raise SweepSealError(f"{label} must be a SHA-256 hex digest")
    try:
        int(rendered, 16)
    except ValueError as exc:
        raise SweepSealError(f"{label} must be a SHA-256 hex digest") from exc
    return rendered


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SweepSealError(f"{label} must be an integer >= {minimum}")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise SweepSealError(f"{label} is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SweepSealError(f"{label} root must be an object")
    return value, raw_bytes


def _lab_verify_outcome(value: Mapping[str, Any]) -> dict[str, Any]:
    """Describe, rather than conceal, the expected Rust/Python seal mismatch."""

    recorded = _digest(value.get("seal_sha256"), "raw Rust receipt seal_sha256")
    try:
        verify(value, label="raw Rust producer receipt")
    except SealIntegrityError as exc:
        expected = seal(value)["seal_sha256"]
        return {
            "status": "MISMATCH_EXPECTED_RUST_SERDE_JSON_NOT_LAB_CANONICAL",
            "recorded_rust_producer_seal_sha256": recorded,
            "expected_lab_receipts_seal_sha256": expected,
            "error": str(exc),
        }
    return {
        "status": "MATCHED_LAB_RECEIPTS_CANONICAL",
        "recorded_rust_producer_seal_sha256": recorded,
        "expected_lab_receipts_seal_sha256": recorded,
        "error": None,
    }


def _validate_raw_run(value: Mapping[str, Any]) -> None:
    if value.get("schema") != SCHEMA:
        raise SweepSealError("raw Rust run schema does not match the bounded sweep contract")
    if value.get("status") != STATUS:
        raise SweepSealError("raw Rust run status does not match a passing bounded sweep")
    scope = _mapping(value.get("scope"), "raw Rust run scope")
    for field in _REQUIRED_SCOPE_TRUE:
        if scope.get(field) is not True:
            raise SweepSealError(f"raw Rust run scope.{field} must be explicitly true")
    metal = _mapping(value.get("metal"), "raw Rust run Metal evidence")
    dispatches = _integer(
        metal.get("aggregate_real_gpu_dispatches"),
        "raw Rust run aggregate_real_gpu_dispatches",
        minimum=1,
    )
    command_buffers = _integer(
        metal.get("aggregate_command_buffers"),
        "raw Rust run aggregate_command_buffers",
        minimum=1,
    )
    waits = _integer(
        metal.get("aggregate_cpu_visible_waits"),
        "raw Rust run aggregate_cpu_visible_waits",
        minimum=1,
    )
    if dispatches != command_buffers or dispatches != waits:
        raise SweepSealError("raw Rust run aggregate command/dispatch/wait accounting mismatch")
    if metal.get("fallback") is not False or _integer(metal.get("fallback_count"), "fallback_count") != 0:
        raise SweepSealError("raw Rust run reports a fallback")
    artifact = _mapping(value.get("artifact_binding"), "raw Rust artifact binding")
    _digest(artifact.get("manifest_file_sha256"), "raw Rust artifact manifest_file_sha256")
    _digest(artifact.get("manifest_seal_sha256"), "raw Rust artifact manifest_seal_sha256")
    artifact_path = Path(_string(artifact.get("path"), "raw Rust artifact path"))
    manifest_path = _regular_file(artifact_path / "manifest.json", "raw Rust artifact manifest")
    if _sha256_file(manifest_path) != artifact["manifest_file_sha256"]:
        raise SweepSealError("raw Rust artifact manifest bytes differ from its reported hash")
    before_after = _mapping(value.get("before_after"), "raw Rust before_after")
    for family in ("fp8_control", "fp4_routed_expert"):
        result = _mapping(before_after.get(family), f"raw Rust {family} before_after")
        if result.get("p50_outcome") != "CANDIDATE_GPU_P50_WIN_NOT_PROMOTED":
            raise SweepSealError(f"raw Rust {family} was not an unpromoted candidate win")
        if result.get("same_raw_weight_input_and_cpu_reference") is not True:
            raise SweepSealError(f"raw Rust {family} did not preserve raw CPU reference")
        _integer(
            result.get("authority_serial_winner_gpu_p50_us"),
            f"raw Rust {family} serial p50",
            minimum=1,
        )
        _integer(
            result.get("candidate_parallel_winner_gpu_p50_us"),
            f"raw Rust {family} candidate p50",
            minimum=1,
        )
    for key in ("serial_authority_before", "optional_parallel_candidates_after"):
        components = _mapping(value.get(key), f"raw Rust {key}")
        if set(("fp8_control", "fp4_routed_expert")) - set(components):
            raise SweepSealError(f"raw Rust {key} lacks a component family")


def _source_hashes() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative in _SOURCE_PATHS:
        path = _regular_file(REPO_ROOT / relative, f"source {relative}")
        rows[relative] = {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    binary = REPO_ROOT / "workspace/ops/build/rust/release/examples/gravity_deepseek_v4_simdgroup_splitk_sweep"
    if binary.exists():
        binary = _regular_file(binary, "release SIMDgroup sweep binary")
        rows["release_binary"] = {
            "path": str(binary),
            "sha256": _sha256_file(binary),
            "bytes": binary.stat().st_size,
        }
    script = _regular_file(Path(__file__), "campaign wrapper script")
    rows["campaign_wrapper_script"] = {
        "path": str(script),
        "sha256": _sha256_file(script),
        "bytes": script.stat().st_size,
    }
    return rows


def _prior_raw_receipt(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    document, raw_bytes = _read_json(path, label)
    return {
        "path": str(path),
        "file_sha256": _sha256_bytes(raw_bytes),
        "bytes": len(raw_bytes),
        "recorded_rust_producer_seal_sha256": document.get("seal_sha256"),
        "lab_receipts_verification": _lab_verify_outcome(document),
        "unaltered": True,
    }


def _write_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise SweepSealError(f"refusing to overwrite existing canonical receipt {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    # `x` prevents replacement of any concurrent output. A receipt is small,
    # and fsync makes the one-shot evidence file durable before reporting it.
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_wrapper(
    *, raw_path: Path, prior_v1: Path, prior_v2: Path, out_path: Path
) -> dict[str, Any]:
    raw_path = _regular_file(raw_path, "fresh raw Rust SIMDgroup run")
    raw_document, raw_bytes = _read_json(raw_path, "fresh raw Rust SIMDgroup run")
    _validate_raw_run(raw_document)
    raw_lab = _lab_verify_outcome(raw_document)
    if raw_lab["status"] != "MISMATCH_EXPECTED_RUST_SERDE_JSON_NOT_LAB_CANONICAL":
        raise SweepSealError(
            "fresh raw run did not expose the expected Rust/Python canonicalization boundary"
        )
    artifact = _mapping(raw_document["artifact_binding"], "raw Rust artifact binding")
    raw_source = _mapping(raw_document["serial_authority_before"], "raw serial authority")
    candidate_source = _mapping(raw_document["optional_parallel_candidates_after"], "raw candidates")
    prior = {
        "v1": _prior_raw_receipt(prior_v1, "prior raw Rust v1 receipt"),
        "v2": _prior_raw_receipt(prior_v2, "prior raw Rust v2 receipt"),
    }
    wrapper = {
        "schema": SCHEMA,
        "status": STATUS,
        "receipt_revision": {
            "revision": "v3_campaign_python_canonical_wrapper",
            "predecessor_raw_rust_receipts_preserved": True,
            "correction": "Rust serde_json producer seals in v1/v2 are not byte-compatible with lab.receipts canonical JSON. This v3 wrapper is a fresh raw run, binds its exact raw bytes, and uses lab.receipts.seal without altering earlier files.",
        },
        # Mirror the strict freezer-facing fields exactly from the fresh raw
        # run. The complete raw document remains nested below for audit.
        "scope": raw_document["scope"],
        "reproduction": raw_document.get("reproduction"),
        "artifact_binding": raw_document["artifact_binding"],
        "metal": raw_document["metal"],
        "before_after": raw_document["before_after"],
        "serial_authority_before": raw_source,
        "optional_parallel_candidates_after": candidate_source,
        "raw_rust_run_evidence": {
            "path": str(raw_path),
            "bytes": len(raw_bytes),
            "file_sha256": _sha256_bytes(raw_bytes),
            "producer_recorded_seal_sha256": raw_document["seal_sha256"],
            "lab_receipts_verification": raw_lab,
            "raw_run_document": raw_document,
        },
        "prior_raw_rust_receipts": prior,
        "source_hashes": _source_hashes(),
        "canonicalization": {
            "wrapper_authority": "lab.receipts.seal / canonical UTF-8 JSON (sort_keys, compact separators, ensure_ascii=False)",
            "wrapper_verified_in_memory_before_write": True,
            "raw_rust_producer_seal_is_not_reused_as_campaign_seal": True,
        },
        "freshness": {"wrapped_at_utc": _utc_now()},
        "claim_boundary": {
            "raw_weight_component_only": True,
            "model_py_activation_quantization_executed": False,
            "source_forward_parity": False,
            "full_43_layer_runtime": False,
            "HCLI_measurement": False,
            "BASE_TRUE_TPS": False,
            "component_candidate_promoted": False,
        },
        "next_boundary": "act_quant -> source-native FP8 projection remains the next faithful unit; this wrapper does not convert raw-matvec evidence into source-forward or runtime evidence.",
    }
    sealed = seal(wrapper)
    verify(sealed, label="fresh campaign-canonical SIMDgroup wrapper")
    if sealed["artifact_binding"]["manifest_seal_sha256"] != artifact["manifest_seal_sha256"]:
        raise SweepSealError("canonical wrapper lost the raw artifact manifest binding")
    return sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="fresh Rust-produced raw v3 JSON")
    parser.add_argument("--prior-v1", required=True, type=Path, help="immutable raw Rust v1 JSON")
    parser.add_argument("--prior-v2", required=True, type=Path, help="immutable raw Rust v2 JSON")
    parser.add_argument("--out", required=True, type=Path, help="new canonical v3 JSON; must not exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        wrapper = build_wrapper(
            raw_path=args.raw,
            prior_v1=args.prior_v1,
            prior_v2=args.prior_v2,
            out_path=args.out,
        )
        _write_new_canonical_json(args.out, wrapper)
        stored, _ = _read_json(args.out.resolve(), "stored canonical v3 wrapper")
        verify(stored, label="stored canonical v3 wrapper")
    except (OSError, SweepSealError, SealIntegrityError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": stored["status"],
                "receipt": str(args.out.resolve()),
                "seal_sha256": stored["seal_sha256"],
                "raw_run_file_sha256": stored["raw_rust_run_evidence"]["file_sha256"],
                "lab_receipts_verify": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
