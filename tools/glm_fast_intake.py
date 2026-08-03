#!/usr/bin/env python3.12
"""Fail-closed GLM custom-format admission for Hide and Ramanujan.

This is deliberately an *intake* gate, not a benchmark wrapper.  A historic
pack, a synthetic parity result, a host-state forward, or a throughput number
without an operator-selected target cannot make an artifact eligible for a
speed-sensitive consumer.  The gate binds the current artifact directory,
parity receipt, and benchmark receipt through the exact index SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "hawking.glm52.fast_intake.v1"
SUPPORTED_INDEXES = ("model.gravity.index.json", "model.activation_aware.index.json")
FAST_FLAGS = (
    "resident_state",
    "compact_mla",
    "device_dsa",
    "compact_attention_icb",
    "device_router",
    "gpu_native_bf16_head",
    "gpu_lm_head_icb",
    "expert_wave",
    "expert_wave_concurrent",
    "expert_table_hit",
    "expert_table_icb",
)


class IntakeError(ValueError):
    """An input cannot support a safe admission decision."""


@dataclass(frozen=True)
class Target:
    min_decode_tps: float | None
    max_decode_p99_ms: float | None
    min_context_tokens: int | None
    min_decode_tokens: int

    def missing(self) -> list[str]:
        missing: list[str] = []
        if self.min_decode_tps is None:
            missing.append("min_decode_tps")
        if self.max_decode_p99_ms is None:
            missing.append("max_decode_p99_ms")
        if self.min_context_tokens is None:
            missing.append("min_context_tokens")
        return missing

    def validate(self) -> None:
        if self.min_decode_tps is not None and self.min_decode_tps <= 0:
            raise IntakeError("min_decode_tps must be positive")
        if self.max_decode_p99_ms is not None and self.max_decode_p99_ms <= 0:
            raise IntakeError("max_decode_p99_ms must be positive")
        if self.min_context_tokens is not None and self.min_context_tokens <= 0:
            raise IntakeError("min_context_tokens must be positive")
        if self.min_decode_tokens <= 0:
            raise IntakeError("min_decode_tokens must be positive")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_admission() -> Path:
    return repo_root() / "workspace/campaign/evidence/models/glm52/GLM52_SOURCE_ADMISSION.json"


def default_parity() -> Path:
    return repo_root() / "workspace/campaign/evidence/models/glm52/GLM52_REFERENCE_PARITY.json"


def default_measurement() -> Path:
    return repo_root() / "workspace/campaign/evidence/models/glm52/GLM52_MATH_PRESERVE_BASE_TPS.json"


def default_ramanujan_lock() -> Path:
    return repo_root() / "ramanujan/records/runtime/RAMANUJAN_ENVIRONMENT_LOCK.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise IntakeError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IntakeError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise IntakeError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate(status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


def artifact_index(artifact_dir: Path | None) -> tuple[Path | None, str | None, dict[str, Any] | None]:
    if artifact_dir is None or not artifact_dir.is_dir():
        return None, None, None
    present = [artifact_dir / name for name in SUPPORTED_INDEXES if (artifact_dir / name).is_file()]
    if len(present) != 1:
        return None, None, None
    index = present[0]
    try:
        return index, sha256_file(index), load_json(index)
    except IntakeError:
        return index, None, None


def capacity_probe_path(artifact_dir: Path | None) -> Path:
    """Find an existing filesystem path without treating a missing artifact as fatal."""
    candidate = artifact_dir if artifact_dir is not None else repo_root()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def nearest_rank_p99_ms(samples: list[float]) -> float:
    """Nearest-rank p99, fail-closed when the raw per-token series is absent."""
    if not samples:
        raise IntakeError("decode_ms_per_token_all must be a non-empty numeric list")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in samples):
        raise IntakeError("decode_ms_per_token_all must contain positive numeric values")
    ordered = sorted(float(value) for value in samples)
    rank = max(0, (99 * len(ordered) + 99) // 100 - 1)
    return ordered[rank]


def evaluate(
    *,
    target: Target,
    artifact_dir: Path | None,
    source_admission_path: Path,
    parity_path: Path,
    measurement_path: Path,
    ramanujan_lock_path: Path,
) -> dict[str, Any]:
    """Evaluate current files without fetching, packing, or starting research."""
    target.validate()
    source = load_json(source_admission_path)
    parity = load_json(parity_path)
    measurement = load_json(measurement_path)
    ramanujan = load_json(ramanujan_lock_path)
    index_path, index_sha, index = artifact_index(artifact_dir)
    free = shutil.disk_usage(capacity_probe_path(artifact_dir)).free
    gates: dict[str, dict[str, Any]] = {}

    missing_target = target.missing()
    gates["TARGET_CONTRACT"] = (
        gate("BLOCKED", "speed target must be supplied by the operator; 'godly' is not a measurable threshold", missing=missing_target)
        if missing_target
        else gate("PASS", "explicit decode throughput, p99, context, and sample-size targets are bound", min_decode_tps=target.min_decode_tps, max_decode_p99_ms=target.max_decode_p99_ms, min_context_tokens=target.min_context_tokens, min_decode_tokens=target.min_decode_tokens)
    )

    source_bytes = source.get("source", {}).get("logical_bytes")
    source_resident = source.get("source", {}).get("complete_source_resident")
    body_stream = source.get("admission_gates", {}).get("body_stream")
    if source_resident is True and body_stream is True:
        gates["SOURCE_AND_PACK_PATH"] = gate("PASS", "source body is admitted and the pack path is available")
    else:
        gates["SOURCE_AND_PACK_PATH"] = gate(
            "BLOCKED",
            "a full GLM source rehydrate/pack path has not been re-admitted; do not infer it from historical receipts",
            logical_source_bytes=source_bytes,
            complete_source_resident=source_resident,
            body_stream_admitted=body_stream,
            blockers=source.get("body_stream_blockers", []),
        )

    if index_path is None or index_sha is None or index is None:
        gates["ARTIFACT_ASSEMBLY"] = gate(
            "BLOCKED",
            "expected exactly one readable custom-format index in a present artifact directory",
            artifact_dir=str(artifact_dir) if artifact_dir else None,
            supported_indexes=list(SUPPORTED_INDEXES),
        )
    else:
        gates["ARTIFACT_ASSEMBLY"] = gate(
            "PASS",
            "one readable custom-format index is present",
            index=str(index_path),
            index_sha256=index_sha,
            index_schema=index.get("schema"),
        )

    parity_status = str(parity.get("status", ""))
    parity_index = parity.get("artifact_index_sha256") or parity.get("index_sha256")
    parity_source = parity.get("source_revision") or parity.get("revision")
    source_revision = source.get("revision")
    if index_sha and parity_status == "PASS" and parity_index == index_sha and parity_source == source_revision:
        gates["ORACLE_PARITY"] = gate("PASS", "source revision and live artifact index are bound to a PASS oracle receipt", index_sha256=index_sha)
    else:
        gates["ORACLE_PARITY"] = gate(
            "BLOCKED",
            "parity must be a source-parent PASS receipt bound to this exact index; synthetic or unbound historical evidence is insufficient",
            receipt_status=parity.get("status"),
            receipt_index_sha256=parity_index,
            receipt_source_revision=parity_source,
            required_source_revision=source_revision,
            live_index_sha256=index_sha,
        )

    measurement_artifact = measurement.get("artifact")
    measurement_index = measurement_artifact.get("index_sha256") if isinstance(measurement_artifact, dict) else None
    run = measurement.get("run_configuration") if isinstance(measurement.get("run_configuration"), dict) else {}
    resolved = run.get("resolved") if isinstance(run.get("resolved"), dict) else {}
    missing_fast_flags = [name for name in FAST_FLAGS if resolved.get(name) is not True]
    full_logits_readback = resolved.get("full_logits_readback")
    rows = measurement.get("measurements")
    rows = rows if isinstance(rows, list) else []
    checked_rows: list[dict[str, Any]] = []
    fastest_error: str | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            samples = row.get("decode_ms_per_token_all")
            if not isinstance(samples, list):
                raise IntakeError("no raw per-token timings")
            p99 = nearest_rank_p99_ms(samples)
            device = row.get("device_execution")
            waits = device.get("command_buffer_waits_per_token_all") if isinstance(device, dict) else None
            device_ok = (
                isinstance(device, dict)
                and device.get("backend") == "metal"
                and isinstance(device.get("device_name"), str)
                and bool(device["device_name"].strip())
                and device.get("resident_state") is True
                and isinstance(waits, list)
                and len(waits) == len(samples)
                and all(isinstance(value, int) and value > 0 for value in waits)
            )
            row_info = {
                "context_tokens": row.get("context_tokens"),
                "decode_tokens": row.get("decode_tokens"),
                "decode_tps": row.get("base_true_decode_tps"),
                "decode_p99_ms": p99,
                "device_execution_proven": device_ok,
                "token_only_device_head": row.get("output_modes") == ["token_plus_topk_diagnostics"],
            }
            checked_rows.append(row_info)
        except IntakeError as exc:
            fastest_error = str(exc)

    target_is_set = not missing_target
    qualified_rows = []
    if target_is_set:
        for row in checked_rows:
            if (
                isinstance(row["context_tokens"], int)
                and row["context_tokens"] >= target.min_context_tokens
                and isinstance(row["decode_tokens"], int)
                and row["decode_tokens"] >= target.min_decode_tokens
                and isinstance(row["decode_tps"], (int, float))
                and row["decode_tps"] >= target.min_decode_tps
                and row["decode_p99_ms"] <= target.max_decode_p99_ms
                and row["device_execution_proven"]
                and row["token_only_device_head"]
            ):
                qualified_rows.append(row)

    measurement_bound = bool(index_sha and measurement_index == index_sha and measurement.get("verify_hash") is True)
    gpu_ready = measurement_bound and not missing_fast_flags and full_logits_readback is False and any(row["device_execution_proven"] for row in checked_rows)
    gates["GPU_FAST_DECODE"] = (
        gate("PASS", "measurement proves resident Metal decode, nonzero command-buffer waits, device token sampling, and all fast-path flags", index_sha256=index_sha)
        if gpu_ready
        else gate(
            "BLOCKED",
            "speed evidence must be index-bound and prove real resident Metal decode; configuration names alone do not qualify",
            measurement_index_sha256=measurement_index,
            live_index_sha256=index_sha,
            verify_hash=measurement.get("verify_hash"),
            missing_fast_flags=missing_fast_flags,
            full_logits_readback=full_logits_readback,
            checked_rows=checked_rows,
        )
    )
    gates["DECODE_PERFORMANCE"] = (
        gate("PASS", "at least one complete row meets the operator-selected decode contract", qualifying_rows=qualified_rows)
        if target_is_set and measurement_bound and gpu_ready and qualified_rows
        else gate(
            "BLOCKED",
            "no index-bound GPU decode row meets all throughput, tail-latency, context, and sample-size requirements",
            checked_rows=checked_rows,
            timing_error=fastest_error,
        )
    )

    all_runtime_pass = all(gates[name]["status"] == "PASS" for name in ("TARGET_CONTRACT", "ARTIFACT_ASSEMBLY", "ORACLE_PARITY", "GPU_FAST_DECODE", "DECODE_PERFORMANCE"))
    gates["HIDE_HANDOFF"] = (
        gate("PASS", "Hide may consume this exact artifact under the sealed performance contract")
        if all_runtime_pass
        else gate("BLOCKED", "Hide may not expose a custom-format artifact until all runtime evidence gates pass")
    )
    research_authorized = ramanujan.get("RAMANUJAN_RESEARCH_AUTHORIZED") is True
    gates["RAMANUJAN_SANDBOX"] = (
        gate("PASS", "runtime is fast-qualified and Ramanujan research is authorized")
        if all_runtime_pass and research_authorized
        else gate(
            "BLOCKED",
            "no Ramanujan sandbox launch: it needs both a fast-qualified runtime and explicit research authorization",
            research_authorized=research_authorized,
        )
    )

    return {
        "schema": SCHEMA,
        "status": "PASS" if gates["RAMANUJAN_SANDBOX"]["status"] == "PASS" else "BLOCKED",
        "intent": "fast custom-format GLM admission; this command never fetches weights, packs artifacts, starts Hide, or starts Ramanujan",
        "free_bytes_observed": free,
        "target": {
            "min_decode_tps": target.min_decode_tps,
            "max_decode_p99_ms": target.max_decode_p99_ms,
            "min_context_tokens": target.min_context_tokens,
            "min_decode_tokens": target.min_decode_tokens,
        },
        "inputs": {
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
            "source_admission": str(source_admission_path),
            "parity": str(parity_path),
            "measurement": str(measurement_path),
            "ramanujan_lock": str(ramanujan_lock_path),
        },
        "gates": gates,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "verify"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--source-admission", type=Path, default=default_source_admission())
    parser.add_argument("--parity", type=Path, default=default_parity())
    parser.add_argument("--measurement", type=Path, default=default_measurement())
    parser.add_argument("--ramanujan-lock", type=Path, default=default_ramanujan_lock())
    parser.add_argument("--min-decode-tps", type=float)
    parser.add_argument("--max-decode-p99-ms", type=float)
    parser.add_argument("--min-context-tokens", type=int)
    parser.add_argument("--min-decode-tokens", type=int, default=32)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    target = Target(args.min_decode_tps, args.max_decode_p99_ms, args.min_context_tokens, args.min_decode_tokens)
    if args.command == "verify" and target.missing():
        print("glm-fast-intake: verify requires --min-decode-tps, --max-decode-p99-ms, and --min-context-tokens", file=sys.stderr)
        return 2
    try:
        result = evaluate(
            target=target,
            artifact_dir=args.artifact_dir,
            source_admission_path=args.source_admission,
            parity_path=args.parity,
            measurement_path=args.measurement,
            ramanujan_lock_path=args.ramanujan_lock,
        )
    except IntakeError as exc:
        print(f"glm-fast-intake: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
