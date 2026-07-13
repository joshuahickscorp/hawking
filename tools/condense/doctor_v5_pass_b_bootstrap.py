#!/usr/bin/env python3.12
"""Create and validate the reviewed 0.5B Pass-B registry and pilot spec."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PASS_B = ROOT / "reports/condense/doctor_v5_pass_b"
SPEC_PATH = PASS_B / "pilot_specs/0.5B-q4-control.json"
REGISTRY_PATH = PASS_B / "adapter_registry.json"
MANIFEST_PATH = PASS_B / "parameter_manifests/0.5B.json"
ADAPTER_PATH = HERE / "doctor_v5_strand_control_adapter.py"
ABI_PATH = HERE / "doctor_v5_adapter_abi.py"
WORKER_PATH = HERE / "doctor_v5_pass_b_worker.py"
PARAMETER_MODULE = HERE / "doctor_v5_parameter_manifest.py"
SPEC_SCHEMA = "hawking.doctor_v5_pass_b_strand_control_spec.v1"
ADAPTER_ID = "doctor-v5-strand-q4-control"
OPERATION = "condense_pilot"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise RuntimeError(f"symlink input forbidden: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block); total += len(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
            or total != after.st_size:
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest(), total


def _relative(path: Path) -> str:
    return str(path.resolve(strict=True).relative_to(ROOT.resolve()))


def _input(role: str, path: Path) -> dict[str, Any]:
    digest, size = _sha_file(path)
    return {"role": role, "path": _relative(path), "sha256": digest, "bytes": size}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _write_once(abi: Any, path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or _load(path) != value:
            raise RuntimeError(f"refusing to replace a different reviewed artifact: {path}")
        return
    abi.atomic_json(path, value)


def prepare() -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import doctor_v5_adapter_abi as abi
    import doctor_v5_parameter_manifest as parameter

    census_path = ROOT / "reports/condense/doctor_v5_scale/0.5B/census.json"
    if not MANIFEST_PATH.exists():
        parameter.build_manifest(census_path, MANIFEST_PATH)
    manifest = parameter.read_json(MANIFEST_PATH)
    errors = parameter.validate_manifest(manifest, verify_files=True)
    if errors:
        raise RuntimeError("0.5B parameter manifest invalid: " + "; ".join(errors))

    model_dir = ROOT / "scratch/qwen-05b"
    program = {
        "schema": "hawking.doctor_v5_pass_b_strand_control_program.v1",
        "adapter_id": ADAPTER_ID,
        "operation": OPERATION,
        "profile": "strand-scalar-quality-rhtcols-v1",
        "bits": 4,
        "threads": 8,
        "quantizer_argv_suffix": [
            "--bits", "4", "--quality", "--rht-cols", "--outlier-channel", "1",
            "--outlier-bits", "8", "--sdsq-sideinfo", "--c2f-outl",
            "--ragged-v2",
        ],
        "attest_roots": True,
        "candidate_must_decode_from_packed_archive": True,
        "evaluation": ["ppl_bench", "multi_eval"],
        "device": "cpu",
        "dtype": "bfloat16",
    }
    resources = {
        "schema": "hawking.doctor_v5_pass_b_resource_admission.v1",
        "memory_pressure_level": 1,
        "swap_used_bytes": 0,
        "require_ac_power": True,
        "require_nominal_thermal": True,
        "disk_reserve_bytes": 150_000_000_000,
        "scratch_budget_bytes": 12_000_000_000,
        "monitor_interval_seconds": 5,
        "gate_failure": "terminate_process_group_and_resume_from_last_phase",
    }
    input_paths = {
        "adapter_abi": ABI_PATH,
        "pass_a_index": ROOT / "reports/condense/doctor_v5_scale/index.json",
        "source_census": census_path,
        "parameter_manifest": MANIFEST_PATH,
        "worker": WORKER_PATH,
        "quantizer": ROOT / "vendor/strand-quant/target/release/quantize-model",
        "attestor": ROOT / "vendor/strand-decode-kernel/target/release/attest-strand",
        "decoder": ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors",
        "ppl_bench": HERE / "ppl_bench.py",
        "multi_eval": HERE / "multi_eval.py",
        "source_weights": model_dir / "model.safetensors",
        "model_config": model_dir / "config.json",
        "generation_config": model_dir / "generation_config.json",
        "tokenizer": model_dir / "tokenizer.json",
        "tokenizer_config": model_dir / "tokenizer_config.json",
        "tokenizer_merges": model_dir / "merges.txt",
        "tokenizer_vocab": model_dir / "vocab.json",
    }
    inputs = sorted((_input(role, path) for role, path in input_paths.items()),
                    key=lambda row: row["role"])
    spec = {
        "schema": SPEC_SCHEMA,
        "label": "0.5B",
        "adapter_id": ADAPTER_ID,
        "operation": OPERATION,
        "profile": program["profile"],
        "bits": 4,
        "model_family": "qwen2",
        "backend": "apple_silicon_cpu",
        "seed": 20260713,
        "program": program,
        "program_spec_sha256": _hash_value(program),
        "resource_admission": resources,
        "resource_admission_sha256": _hash_value(resources),
        "disk_reserve_bytes": resources["disk_reserve_bytes"],
        "scratch_budget_bytes": resources["scratch_budget_bytes"],
        "maximum_scratch_bytes": resources["scratch_budget_bytes"],
        "inputs": inputs,
        "quality_claims_permitted": False,
        "source_deletion_permitted": False,
    }
    _write_once(abi, SPEC_PATH, spec)
    spec_sha, _ = _sha_file(SPEC_PATH)

    adapter_sha, _ = _sha_file(ADAPTER_PATH)
    python = Path(sys.executable)
    if python.is_symlink() or not stat.S_ISREG(python.lstat().st_mode):
        raise RuntimeError("Python interpreter is not a regular non-symlink file")
    python = python.resolve(strict=True)
    python_sha, _ = _sha_file(python)
    entry = {
        "adapter_id": ADAPTER_ID,
        "adapter_version": "1",
        "source_path": _relative(ADAPTER_PATH),
        "source_sha256": adapter_sha,
        "executable_path": str(python),
        "executable_sha256": python_sha,
        "entrypoint_argv": [str(python), _relative(ADAPTER_PATH), "run",
                            "--request", "{request_path}"],
        "operations": [OPERATION],
        "model_families": ["qwen2"],
        "backends": ["apple_silicon_cpu"],
        "request_schema": abi.REQUEST_SCHEMA,
        "result_schema": abi.RESULT_SCHEMA,
        "checkpoint_schema": abi.CHECKPOINT_SCHEMA,
        "reviewed": True,
        "execution_only_not_quality_evidence": True,
    }
    registry = abi.build_registry([entry])
    _write_once(abi, REGISTRY_PATH, registry)
    errors = abi.validate_registry(registry, verify_files=True, base_dir=ROOT)
    if errors:
        raise RuntimeError("adapter registry invalid: " + "; ".join(errors))
    return {
        "schema": "hawking.doctor_v5_pass_b_bootstrap_receipt.v1",
        "status": "ready",
        "registry": {"path": str(REGISTRY_PATH), "sha256": _sha_file(REGISTRY_PATH)[0],
                     "registry_sha256": registry["registry_sha256"]},
        "pilot_spec": {"path": str(SPEC_PATH), "sha256": spec_sha,
                       "program_spec_sha256": spec["program_spec_sha256"],
                       "resource_admission_sha256": spec["resource_admission_sha256"]},
        "parameter_manifest": {"path": str(MANIFEST_PATH),
                               "sha256": _sha_file(MANIFEST_PATH)[0],
                               "exact_stored_parameters": manifest["parameter_authority"][
                                   "exact_distinct_stored_parameter_count"]},
        "source_deletion_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "selftest"))
    args = parser.parse_args()
    if args.command == "selftest":
        assert _hash_value({"a": 1, "b": 2}) == _hash_value({"b": 2, "a": 1})
        print("doctor_v5_pass_b_bootstrap.py selftest OK")
        return 0
    print(json.dumps(prepare(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
