#!/usr/bin/env python3
"""Capture a two-step PLE-inclusive Flash source reference through MLX-VLM.

This is an isolated research/reference oracle, not a production Hawking
runtime.  It binds the exact installed implementation and runtime, consumes
the already verified source-shard ledger, persists raw F32 logits, and records
cache/PLE state-boundary identities.  It deliberately does not create or fake
the machine-admin owner signature required for V2 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawking.persist import atomic_write_json, sha256_file_or_none  # noqa: E402
from tools.odyssey import flash_repeated_accepted_decode as gate  # noqa: E402


RUNTIME_LOCK = Path(__file__).with_name("flash_external_reference_mlx_runtime.lock.json")
WORKER = Path(__file__).with_name("flash_external_greedy_reference_mlx_worker.py")
DEFAULT_LEDGER = ROOT / "receipts/headless/FLASH_SOURCE_SHARD_LEDGER_20260912.json"
DEFAULT_OUT = ROOT / "receipts/headless/FLASH_EXTERNAL_GREEDY_REFERENCE_MLX_20260912"
CAPTURE_SCHEMA = "hawking.flash.external_greedy_reference_capture.v1"
CAPTURE_STATUS = "CAPTURED_EXTERNAL_COMPLETE_SOURCE_GREEDY_REFERENCE__OWNER_AUTHORIZATION_REQUIRED"
AUTHORIZATION_REQUEST_SCHEMA = "hawking.flash.external_greedy_reference_owner_authorization_request.v1"
GENERATED_STEPS = 2
EXPECTED_EXECUTION_SCHEDULE = "exact_qwen4_exp_row_addressed_ple_and_experts_with_evaluated_sublayer_boundaries"


def _sha256(path: Path) -> str:
    value = sha256_file_or_none(path)
    if value is None:
        raise ValueError(f"cannot hash required artifact: {path}")
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file: {resolved}")
    if info.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {resolved}")
    return resolved


def _executable(path: Path, label: str) -> tuple[Path, Path]:
    """Validate a locked executable while retaining its selected venv path.

    uv environments intentionally expose Python through a symlink to a shared,
    hard-linked interpreter. Evidence artifacts keep the stricter `_regular`
    rule; executable identity is closed by the selected path, resolved target,
    runtime lock, package versions, and imported implementation hashes.
    """
    selected = path.expanduser()
    target = selected.resolve(strict=True)
    info = target.stat()
    if not stat.S_ISREG(info.st_mode) or not selected.exists():
        raise ValueError(f"{label} must resolve to a regular file: {selected}")
    if not os.access(target, os.X_OK):
        raise ValueError(f"{label} is not executable: {target}")
    return selected, target


def _runtime_lock() -> dict[str, Any]:
    lock_path = _regular(RUNTIME_LOCK, "MLX reference runtime lock")
    lock = gate._json(lock_path)
    if lock.get("schema") != "hawking.flash.external_reference_mlx_runtime_lock.v1":
        raise ValueError("MLX reference runtime lock has an incompatible schema")
    selected_python, expected_python = _executable(
        Path(str(lock.get("python_executable", ""))),
        "locked MLX reference Python",
    )
    packages = lock.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("MLX reference runtime lock omitted package versions")
    for name, expected in packages.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("MLX reference runtime lock contains a malformed package entry")
    probe_code = (
        "import importlib.metadata,json;"
        "names=" + repr(sorted(packages)) + ";"
        "print(json.dumps({n:importlib.metadata.version(n) for n in names}))"
    )
    probe = subprocess.run(
        [str(selected_python), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(probe.stdout)
    for name, expected in packages.items():
        actual = observed.get(name)
        if actual != expected:
            raise ValueError(f"MLX reference runtime package drift: {name} {actual} != {expected}")
    return {
        "path": str(lock_path),
        "sha256": _sha256(lock_path),
        "python_executable": str(selected_python),
        "python_executable_resolved": str(expected_python),
        "python_executable_sha256": _sha256(expected_python),
        "packages": observed,
    }


def _source_ledger(model_root: Path, ledger_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_path = _regular(ledger_path, "verified source shard ledger")
    ledger = gate._json(ledger_path)
    manifest = gate._canonical_model_lake_manifest(model_root)
    index = gate._source_file(model_root, "model.safetensors.index.json", "source safetensors index")
    binding = {
        "path": str(ledger_path),
        "sha256": _sha256(ledger_path),
        "schema": gate.SOURCE_SHARD_LEDGER_SCHEMA,
        "status": gate.SOURCE_SHARD_LEDGER_STATUS,
        "seal_sha256": ledger.get("seal_sha256"),
    }
    verified = gate._verify_shard_ledger(
        binding,
        contract_dir=ledger_path.parent,
        model_root=model_root,
        manifest_sha256=_sha256(manifest),
        index_path=index,
        index_sha256=_sha256(index),
    )
    return binding, verified


def _implementation_paths(runtime: dict[str, Any]) -> list[Path]:
    python = runtime["python_executable"]
    module_names = [
        "mlx_vlm.utils",
        "mlx_vlm.models.cache",
        "mlx_vlm.models.switch_layers",
        "mlx_vlm.models.qwen3_5.language",
        "mlx_vlm.models.qwen4_exp.qwen4_exp",
        "mlx_vlm.models.qwen4_exp.language",
        "mlx_vlm.models.qwen4_exp.fp8",
    ]
    probe_code = (
        "import importlib,json;"
        "names=" + repr(module_names) + ";"
        "print(json.dumps({n:importlib.import_module(n).__file__ for n in names}))"
    )
    probe = subprocess.run([python, "-c", probe_code], check=True, capture_output=True, text=True)
    paths = json.loads(probe.stdout)
    return sorted({_regular(Path(paths[name]), name) for name in module_names})


def _write_implementation_manifest(out_dir: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    files = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in _implementation_paths(runtime)
    ]
    manifest = {
        "schema": "hawking.flash.external_reference_mlx_implementation.v1",
        "status": "PINNED_EXTERNAL_REFERENCE_IMPLEMENTATION",
        "files": files,
        "architecture": "Qwen4ExpForConditionalGeneration / Qwen4ExpTextConfig",
        "execution_role": "isolated PLE-inclusive source oracle only",
        "independent_from_hawking_native_executor": True,
        "claim_boundary": "source-code closure only; semantic correctness still requires independent review and owner authorization",
    }
    path = out_dir / "provider-implementation.json"
    atomic_write_json(path, manifest)
    return {"path": str(path), "sha256": _sha256(path)}


def _reserve_output(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _artifact_binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def _write_capture_progress(out_dir: Path, document: dict[str, Any]) -> None:
    atomic_write_json(out_dir / "capture-progress.json", document)


def capture(model_root: Path, ledger_path: Path, out_dir: Path) -> dict[str, Any]:
    model_root = model_root.expanduser().resolve()
    lane = gate._protected_native_lane()
    if not lane["clean"]:
        raise ValueError(f"protected native/provider lane is not clean: {lane['matches']}")
    runtime = _runtime_lock()
    ledger_binding, verified_ledger = _source_ledger(model_root, ledger_path)
    out_dir = _reserve_output(out_dir)
    implementation = _write_implementation_manifest(out_dir, runtime)
    producer = _artifact_binding(_regular(WORKER, "external source provider worker"))
    packager = _artifact_binding(Path(__file__).resolve())
    runtime_artifact = {"path": runtime["path"], "sha256": runtime["sha256"]}
    if len({implementation["path"], producer["path"], runtime_artifact["path"]}) != 3:
        raise ValueError("provider implementation, producer, and runtime lock must be distinct artifacts")

    started = time.time_ns()
    progress: dict[str, Any] = {
        "schema": CAPTURE_SCHEMA,
        "status": "RUNNING_EXTERNAL_SOURCE_GREEDY_CAPTURE",
        "source": {
            "model": gate.REPO_ID,
            "pinned_revision": gate.PINNED_REVISION,
            "model_root": str(model_root),
            "shard_ledger": ledger_binding,
        },
        "provider": {
            "implementation": implementation,
            "runtime_lock": runtime_artifact,
            "producer": producer,
            "hawking_packager": packager,
            "runtime": runtime,
        },
        "prompt_token_ids": list(gate.PROMPT_IDS),
        "generated_token_ids": [],
        "steps": [],
        "started_unix_ns": started,
        "claim_boundary": "in-flight external source capture; not an admitted reference",
    }
    _write_capture_progress(out_dir, progress)

    worker_command = [
        runtime["python_executable"],
        str(Path(producer["path"])),
        "--root", str(model_root),
        "--out-dir", str(out_dir),
        "--prompt-json", json.dumps(list(gate.PROMPT_IDS), separators=(",", ":")),
        "--steps", str(GENERATED_STEPS),
        "--vocab-size", "248320",
    ]
    try:
        worker_run = subprocess.run(
            worker_command,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        progress.update({
            "status": "FAILED_EXTERNAL_PROVIDER_RUNTIME__NO_NEW_SOURCE_RESULT",
            "failure": {
                "provider_exit_code": exc.returncode,
                "stdout_tail": (exc.stdout or "")[-12_000:],
                "stderr_tail": (exc.stderr or "")[-12_000:],
                "scientific_disposition": "none",
            },
            "finished_unix_ns": time.time_ns(),
            "claim_boundary": "provider/runtime failure only; not source evidence, teacher admission, or a Flash scientific disposition",
        })
        _write_capture_progress(out_dir, progress)
        raise RuntimeError("external source provider failed; terminal evidence is preserved") from exc
    del worker_run
    provider_capture_path = _regular(out_dir / "provider-capture.json", "external provider capture")
    provider_capture = gate._json(provider_capture_path)
    if (
        provider_capture.get("schema") != "hawking.flash.external_greedy_reference_mlx_worker.v1"
        or provider_capture.get("status") != "CAPTURED_TWO_STEP_GREEDY_F32"
        or provider_capture.get("root") != str(model_root)
        or provider_capture.get("prompt_token_ids") != list(gate.PROMPT_IDS)
        or provider_capture.get("no_sampling") is not True
        or provider_capture.get("incremental_cache_continuity") is not True
        or provider_capture.get("execution_schedule") != EXPECTED_EXECUTION_SCHEDULE
    ):
        raise ValueError("external provider capture has an incompatible identity or execution contract")
    ple_storage = provider_capture.get("ple_storage_manifest")
    if not isinstance(ple_storage, dict):
        raise ValueError("external provider capture omitted its row-addressed PLE manifest")
    ple_manifest_path = _regular(
        Path(str(ple_storage.get("path", ""))),
        "external provider PLE range manifest",
    )
    if (
        ple_manifest_path.parent != out_dir
        or ple_storage.get("sha256") != _sha256(ple_manifest_path)
        or ple_storage.get("layout") != "safetensors_ranges"
        or ple_storage.get("source_mutated") is not False
        or ple_storage.get("weight_payload_copied") is not False
    ):
        raise ValueError("external provider PLE range manifest binding differs")
    ple_manifest_artifact = _artifact_binding(ple_manifest_path)
    references = gate._require_token_list(
        provider_capture.get("generated_token_ids"),
        "external provider generated_token_ids",
    )
    provider_steps = provider_capture.get("steps")
    if len(references) != GENERATED_STEPS or not isinstance(provider_steps, list) or len(provider_steps) != GENERATED_STEPS:
        raise ValueError("external provider capture omitted a generated step")
    steps: list[dict[str, Any]] = []
    for generation_index, (token_id, raw_step) in enumerate(zip(references, provider_steps, strict=True)):
        if not isinstance(raw_step, dict):
            raise ValueError("external provider capture contains a malformed step")
        expected_input = [*gate.PROMPT_IDS, *references[:generation_index]]
        payload_binding = raw_step.get("logits_f32")
        if not isinstance(payload_binding, dict):
            raise ValueError("external provider capture omitted raw F32 logits")
        payload = _regular(Path(str(payload_binding.get("path", ""))), "external provider logits")
        raw = payload.read_bytes()
        observed_argmax = gate._f32_argmax(raw, elements=248_320, label=f"source logits step {generation_index}")
        if (
            raw_step.get("generation_index") != generation_index
            or raw_step.get("input_token_ids") != expected_input
            or raw_step.get("argmax_token_id") != token_id
            or observed_argmax != token_id
            or payload_binding.get("sha256") != hashlib.sha256(raw).hexdigest()
            or payload_binding.get("dtype") != "F32_LE"
            or payload_binding.get("elements") != 248_320
            or payload_binding.get("bytes") != 248_320 * 4
        ):
            raise ValueError("external provider logits or token boundary failed independent packaging verification")
        state_boundary = raw_step.get("state_boundary")
        if not isinstance(state_boundary, dict) or state_boundary.get("schema") != "hawking.flash.external_source_state_boundary.v1":
            raise ValueError("external provider capture omitted state boundary identity")
        steps.append({**raw_step, "expected_token_id": token_id, "logits_f32": _artifact_binding(payload) | {
            "dtype": "F32_LE", "elements": 248_320, "bytes": 248_320 * 4,
        }})
    ple_layers = provider_capture.get("ple_layer_ids")
    if not isinstance(ple_layers, list) or not ple_layers:
        raise ValueError("external provider capture instantiated no PLE layers")

    manifest_path = gate._canonical_model_lake_manifest(model_root)
    config_path = gate._source_file(model_root, "config.json", "source config")
    index_path = gate._source_file(model_root, "model.safetensors.index.json", "source safetensors index")
    tokenizer_path = gate._source_file(model_root, "tokenizer.json", "source tokenizer")
    manifest_sha256 = _sha256(manifest_path)
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)
    tokenizer_sha256 = _sha256(tokenizer_path)
    provider_hashes = {
        "implementation_sha256": implementation["sha256"],
        "runtime_lock_sha256": runtime_artifact["sha256"],
        "producer_sha256": producer["sha256"],
        "ple_storage_manifest_sha256": ple_manifest_artifact["sha256"],
    }
    trace = {
        "schema": gate.LOGITS_TRACE_SCHEMA,
        "status": gate.LOGITS_TRACE_STATUS,
        "source": {
            "model": gate.REPO_ID,
            "pinned_revision": gate.PINNED_REVISION,
            "config_sha256": config_sha256,
            "safetensors_index_sha256": index_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "model_lake_manifest_sha256": manifest_sha256,
            "shard_ledger_sha256": ledger_binding["sha256"],
        },
        "provider_artifacts": provider_hashes,
        "prompt_token_ids": list(gate.PROMPT_IDS),
        "generated_token_ids": references,
        "sampling": {"method": "greedy_argmax", "do_sample": False, "temperature": 0},
        "dtype": "F32_LE",
        "argmax_tie_break": "lowest_token_id",
        "vocab_size": 248_320,
        "ple_execution": {
            "inclusive": True,
            "instantiated_ple_layer_ids": ple_layers,
            "incremental_cache_continuity": True,
            "execution_schedule": EXPECTED_EXECUTION_SCHEDULE,
            "row_addressed_storage_manifest": ple_manifest_artifact,
        },
        "steps": steps,
    }
    trace = {**trace, "seal_sha256": gate._compact_utf8_sorted_seal(trace)}
    trace_path = out_dir / "external-logits-trace.json"
    atomic_write_json(trace_path, trace)

    contract = {
        "schema": gate.EXTERNAL_REFERENCE_SCHEMA,
        "status": gate.EXTERNAL_REFERENCE_STATUS,
        "source": {
            "model": gate.REPO_ID,
            "pinned_revision": gate.PINNED_REVISION,
            "config_sha256": config_sha256,
            "safetensors_index_sha256": index_sha256,
            "ple_inclusive": True,
            "model_lake_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "repo": gate.REPO_ID,
                "revision": gate.PINNED_REVISION,
            },
            "shard_ledger": ledger_binding,
        },
        "tokenizer": {"file": "tokenizer.json", "sha256": tokenizer_sha256},
        "prompt_token_ids": list(gate.PROMPT_IDS),
        "generated_token_ids": references,
        "sampling": {"method": "greedy_argmax", "do_sample": False, "temperature": 0},
        "provider": {
            "independent_from_hawking_native_executor": True,
            "implementation_version": "mlx-vlm-0.7.0-qwen4-exp",
            "implementation": implementation,
            "runtime_lock": runtime_artifact,
            "producer": producer,
            "execution_schedule": EXPECTED_EXECUTION_SCHEDULE,
            "ple_storage_manifest": ple_manifest_artifact,
        },
        "logits_trace": {
            "path": str(trace_path),
            "sha256": _sha256(trace_path),
            "schema": gate.LOGITS_TRACE_SCHEMA,
            "status": gate.LOGITS_TRACE_STATUS,
            "seal_sha256": trace["seal_sha256"],
        },
    }
    contract = {**contract, "seal_sha256": gate._compact_utf8_sorted_seal(contract)}
    contract_path = out_dir / "external-reference.v2.json"
    atomic_write_json(contract_path, contract)

    authorization_request = {
        "schema": AUTHORIZATION_REQUEST_SCHEMA,
        "status": "OWNER_ACTION_REQUIRED",
        "authorization_schema": gate.OWNER_AUTHORIZATION_SCHEMA,
        "authorization_scope": gate.OWNER_AUTHORIZATION_SCOPE,
        "owner_public_key_path": str(gate.pinned_owner_public_key_path()),
        "required_payload_bindings": {
            "reference_contract_sha256": _sha256(contract_path),
            "reference_contract_seal_sha256": contract["seal_sha256"],
            "provider_implementation_sha256": implementation["sha256"],
            "provider_runtime_lock_sha256": runtime_artifact["sha256"],
            "provider_producer_sha256": producer["sha256"],
            "logits_trace_sha256": _sha256(trace_path),
            "logits_trace_seal_sha256": trace["seal_sha256"],
            "shard_ledger_sha256": ledger_binding["sha256"],
            "shard_ledger_seal_sha256": ledger_binding["seal_sha256"],
            "model_lake_manifest_sha256": manifest_sha256,
        },
        "owner_must_supply": ["issued_unix_ns", "expires_unix_ns", "nonce", "signature_ed25519_hex"],
        "claim_boundary": "unsigned admission request only; it grants no execution or evidence authority",
    }
    atomic_write_json(out_dir / "owner-authorization-request.json", authorization_request)

    final = {
        **progress,
        "status": CAPTURE_STATUS,
        "generated_token_ids": references,
        "steps": steps,
        "ple_layer_count": len(ple_layers),
        "contract": _artifact_binding(contract_path),
        "logits_trace": _artifact_binding(trace_path),
        "source_shard_ledger": verified_ledger,
        "finished_unix_ns": time.time_ns(),
        "elapsed_ns": time.time_ns() - started,
        "max_resident_set_size_bytes": provider_capture.get("max_resident_set_size_bytes"),
        "model_loaded": True,
        "gpu_or_provider_server_started": False,
        "owner_authorization_present": False,
        "claim_boundary": "external oracle capture only; V2 admission remains false until the machine-admin owner authorization verifies",
    }
    atomic_write_json(out_dir / "capture.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    model_root = args.root.expanduser().resolve()
    runtime = _runtime_lock()
    ledger_binding, verified = _source_ledger(model_root, args.ledger)
    lane = gate._protected_native_lane()
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_EXTERNAL_GREEDY_REFERENCE_MLX",
            "model_root": str(model_root),
            "runtime": runtime,
            "source_shard_ledger": verified,
            "out_dir": str(args.out_dir.expanduser().resolve()),
            "out_exists": args.out_dir.expanduser().resolve().exists(),
            "protected_native_lane": lane,
            "generated_steps": GENERATED_STEPS,
            "claim_boundary": "preflight only; no model load, forward, logits, or teacher admission",
        }, indent=2))
        return 0
    if not args.execute:
        raise ValueError("external source capture requires explicit --execute")
    if not lane["clean"]:
        raise ValueError(f"protected native/provider lane is not clean: {lane['matches']}")
    result = capture(model_root, Path(ledger_binding["path"]), args.out_dir)
    print(json.dumps({
        "status": result["status"],
        "generated_token_ids": result["generated_token_ids"],
        "contract": result["contract"],
        "elapsed_ns": result["elapsed_ns"],
        "owner_authorization_present": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
