#!/usr/bin/env python3.12
"""Build and verify a self-contained handoff packet for The Appendix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import appendix_catalog
import appendix_contract
import appendix_physical_counter_authority
import appendix_physical_counter_executor
import appendix_physical_counter_request
import appendix_postrun
import appendix_scaffold
import spec_reentry_scaffold
import tq_runtime_probe
import tq_runtime_matrix


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_handoff.v1"
SOURCE_FILES = (
    "crates/hawking-core/shaders/strand_bitslice.metal",
    "crates/hawking-core/build.rs",
    "crates/hawking-core/src/kernels/mod.rs",
    "crates/hawking-core/src/lib.rs",
    "crates/hawking-core/src/metal/mod.rs",
    "crates/hawking-core/src/metal/physical_signpost.c",
    "crates/hawking-core/src/model/qwen_dense.rs",
    "crates/hawking-core/src/speculate/governor.rs",
    "crates/hawking-core/src/speculate/router.rs",
    "crates/hawking-core/src/tq.rs",
    "crates/hawking-core/src/tq_gpu.rs",
    "crates/hawking/Cargo.toml",
    "crates/hawking/src/tq_device_probe.rs",
    "crates/hawking/src/tq_spec_probe.rs",
    "crates/hawking/src/process_joule.rs",
    "docs/env_flags.md",
    "docs/plans/APPENDIX.md",
    "docs/plans/APPENDIX_HANDOFF.md",
    "docs/plans/appendix_counter_authority_allowed_signers",
    "docs/plans/appendix_counter_authority_registry.json",
    "docs/plans/hawking_event_horizon_status.md",
    "docs/plans/spec_decode_reentry_appendix_2026_07_14.md",
    "docs/plans/spec_decode_studio_readiness_2026_07_12.md",
    "docs/plans/tq_compute_for_memory_appendix_2026_07_14.md",
    "tools/condense/appendix_catalog.py",
    "tools/condense/appendix_cheap_gates.py",
    "tools/condense/appendix_contract.py",
    "tools/condense/appendix_corpus.py",
    "tools/condense/appendix_device_runner.py",
    "tools/condense/doctor_v5_local_observer.py",
    "tools/condense/appendix_handoff.py",
    "tools/condense/appendix_ledger.py",
    "tools/condense/appendix_postrun.py",
    "tools/condense/appendix_physical_counter_collector.py",
    "tools/condense/appendix_physical_counter_authority.py",
    "tools/condense/appendix_physical_counter_executor.py",
    "tools/condense/appendix_physical_counter_normalizer.py",
    "tools/condense/appendix_physical_counter_request.py",
    "tools/condense/appendix_physical_evidence_gate.py",
    "tools/condense/appendix_physical_release_packet.py",
    "tools/condense/appendix_physical_release_state.py",
    "tools/condense/appendix_process_joule_collector.py",
    "tools/condense/appendix_scaffold.py",
    "tools/condense/appendix_xctrace_export_adapter.py",
    "tools/condense/physical_counter_attestation.py",
    "tools/condense/spec_reentry_scaffold.py",
    "tools/condense/spec_receipt_contract.py",
    "tools/condense/tq_runtime_probe.py",
    "tools/condense/spec_tq_runner.py",
    "tools/condense/tq_receipt_contract.py",
    "tools/condense/tq_runtime_matrix.py",
    "tools/condense/tests/test_appendix_catalog.py",
    "tools/condense/tests/test_appendix_cheap_gates.py",
    "tools/condense/tests/test_appendix_contract.py",
    "tools/condense/tests/test_appendix_corpus.py",
    "tools/condense/tests/test_appendix_device_runner.py",
    "tools/condense/tests/test_appendix_handoff.py",
    "tools/condense/tests/test_appendix_ledger.py",
    "tools/condense/tests/test_appendix_postrun.py",
    "tools/condense/tests/test_appendix_physical_counter_collector.py",
    "tools/condense/tests/test_appendix_physical_counter_authority.py",
    "tools/condense/tests/test_appendix_physical_counter_executor.py",
    "tools/condense/tests/test_appendix_physical_counter_normalizer.py",
    "tools/condense/tests/test_appendix_physical_counter_request.py",
    "tools/condense/tests/test_appendix_physical_evidence_gate.py",
    "tools/condense/tests/test_appendix_physical_release_packet.py",
    "tools/condense/tests/test_appendix_physical_release_state.py",
    "tools/condense/tests/test_appendix_process_joule_collector.py",
    "tools/condense/tests/test_appendix_scaffold.py",
    "tools/condense/tests/test_appendix_xctrace_export_adapter.py",
    "tools/condense/tests/physical_counter_fixtures.py",
    "tools/condense/tests/test_physical_counter_attestation.py",
    "tools/condense/tests/test_spec_reentry_scaffold.py",
    "tools/condense/tests/test_spec_receipt_contract.py",
    "tools/condense/tests/test_spec_tq_runner.py",
    "tools/condense/tests/test_tq_runtime_probe.py",
    "tools/condense/tests/test_tq_receipt_contract.py",
    "tools/condense/tests/test_tq_runtime_matrix.py",
    "vendor/strand-decode-kernel/Cargo.toml",
    "vendor/strand-decode-kernel/shaders/strand_bitslice.metal",
    "vendor/strand-decode-kernel/shaders/strand_bitslice_staged.metal",
    "vendor/strand-decode-kernel/src/block_walk.rs",
    "vendor/strand-decode-kernel/src/metal.rs",
    "vendor/strand-decode-kernel/src/bin/gate-bitslice.rs",
    "vendor/strand-decode-kernel/src/bin/gate-bitslice-staged.rs",
    "vendor/strand-decode-kernel/src/bin/gate-coopwindow.rs",
    "vendor/strand-decode-kernel/src/bin/gate-tablecompact.rs",
    "vendor/strand-decode-kernel/src/bin/gate-token-buffer.rs",
    "vendor/strand-quant/src/codebook.rs",
    "vendor/strand-quant/src/decode.rs",
    "vendor/strand-quant/src/tests.rs",
    "vendor/strand-quant/src/trellis.rs",
)
GENERATED_ARTIFACTS = (
    "reports/appendix/tq_runtime_static_probe.json",
    "reports/appendix/tq_runtime_static_probe.receipt.json",
    "reports/appendix/tq_runtime_device_matrix.json",
    "reports/appendix/appendix_postrun_plan.json",
    "reports/appendix/appendix_cheap_gates.json",
    "reports/appendix/appendix_release_packet_cheap_gates.json",
    "reports/appendix/physical_release/release_plan.json",
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _source_manifest() -> list[dict]:
    rows = []
    for relative in SOURCE_FILES:
        path = ROOT / relative
        rows.append({
            "path": relative,
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": _sha_file(path) if path.is_file() else None,
        })
    return rows


def _artifact_manifest() -> list[dict]:
    rows = []
    for relative in GENERATED_ARTIFACTS:
        path = ROOT / relative
        rows.append({
            "path": relative,
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": _sha_file(path) if path.is_file() else None,
        })
    return rows


def _stamp_packet(packet: dict) -> dict:
    stamped = copy.deepcopy(packet)
    stamped.pop("packet_sha256", None)
    stamped["packet_sha256"] = appendix_contract.canonical_sha256(stamped)
    return stamped


def build_packet() -> dict:
    master = appendix_scaffold.build_plan()
    catalog = appendix_catalog.build_catalog()
    spec = spec_reentry_scaffold.build_matrix("CORPUS")
    probe = tq_runtime_probe.build_probe()
    device_matrix = tq_runtime_matrix.build_matrix(probe)
    postrun = appendix_postrun.build_plan()
    counter_authority = appendix_physical_counter_authority.load_default_registry()
    counter_executor = appendix_physical_counter_executor.execution_capability_contract()
    counter_request = appendix_physical_counter_request.build_config()
    return _stamp_packet({
        "schema": SCHEMA,
        "name": "The Appendix",
        "source_commit": _source_commit(),
        "active_run_contract": {
            "is_primary_corpus": True,
            "must_not_stop_or_modify": True,
            "heavy_execution_deferred": True,
        },
        "fingerprints": {
            "master_plan_sha256": appendix_contract.canonical_sha256(master),
            "capability_catalog_sha256": appendix_contract.canonical_sha256(catalog),
            "spec_matrix_sha256": appendix_contract.canonical_sha256(spec),
            "static_probe_sha256": appendix_contract.canonical_sha256(probe),
            "tq_device_matrix_sha256": appendix_contract.canonical_sha256(device_matrix),
            "postrun_plan_sha256": appendix_contract.canonical_sha256(postrun),
            "counter_authority_registry_sha256": counter_authority["registry_sha256"],
            "counter_executor_capability_sha256": counter_executor["capability_sha256"],
            "counter_request_builder_config_sha256": counter_request["config_sha256"],
        },
        "coverage": {
            "capability_sectors": len(catalog["sectors"]),
            "master_cells": len(master["cells"]),
            "spec_cells": len(spec["cells"]),
            "static_tq_cells": len(probe["cells"]),
            "tq_device_cells": device_matrix["counts"],
            "implemented_tq_runtime_paths": ["stored", "compact", "hashed", "computed"],
            "accounted_future_tq_recipes": ["compact_hashed", "compact_computed", "repacked_lut"],
            "mapped_postrun_gates": postrun["counts"]["mapped_existing_gates"],
            "missing_postrun_adapters": postrun["counts"]["missing_artifact_adapters"],
        },
        "source_manifest": _source_manifest(),
        "generated_artifacts": _artifact_manifest(),
        "incorporation_order": [
            "read docs/plans/APPENDIX.md and docs/plans/APPENDIX_HANDOFF.md",
            "run appendix_handoff.py --audit",
            "run all cheap gates listed in this packet",
            "preserve the active corpus and its heavy lease",
            "after the run, freeze the corpus index and bind every receipt",
            "use appendix_postrun.py to distinguish vendor research from final Hawking evidence",
            "build both release probes after the run and admit TQ device cells before TQ-native speculative verification",
        ],
        "cheap_gate_commands": [
            "python3.12 tools/condense/appendix_handoff.py --audit",
            "python3.12 tools/condense/appendix_cheap_gates.py --verify-release-packet reports/appendix/appendix_release_packet_cheap_gates.json",
            "python3.12 tools/condense/appendix_postrun.py --selftest",
            "python3.12 -m pytest -q tools/condense/tests/test_appendix_*.py tools/condense/tests/test_spec_*.py tools/condense/tests/test_tq_*.py",
            "cargo test -p hawking-core --features tq --lib tq_gpu::tests::runtime_ -- --test-threads=1",
            "cargo test -p hawking-core --features tq --lib tq_gpu::tests::gpu_ -- --test-threads=1",
            "cargo test -p hawking-core --features tq --lib speculate::router::tests -- --test-threads=1",
            "cargo check -p hawking-core --features tq",
            "cargo check -p hawking --features tq --bin hawking-tq-device-probe --bin hawking-tq-spec-probe",
            "git diff --check",
        ],
        "deferred_device_gates": [
            "Metal compilation and host/device record-size probe",
            "stored/compact/hashed/computed fused GEMV parity",
            "occupancy, realized bandwidth, latency, energy, pressure, swap, and thermal receipts",
            "ragged-row TQ GPU design for cols not divisible by 256",
            "compact+hashed, compact+computed, and layout-repacked lookup kernels",
            "artifact-bound Hawking-core device receipts from the implemented lease runner",
            "TQ-native B=1..8 batched-verifier device parity from the implemented lease runner",
        ],
        "known_non_appendix_blockers": [
            "local xcrun cannot locate the optional offline Metal compiler; runtime Metal source compilation remains the execution path",
            "vendor strand-quant all-target test builds encounter the pre-existing quantize-model Args drift; use --lib for library gates",
        ],
    })


def verify_packet(packet: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, dict):
        return ["handoff packet must be an object"]
    if packet.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    unstamped = copy.deepcopy(packet)
    claimed = unstamped.pop("packet_sha256", None)
    if claimed != appendix_contract.canonical_sha256(unstamped):
        errors.append("packet_sha256 mismatch")
    if packet.get("source_commit") != _source_commit():
        errors.append("source_commit differs from current HEAD")
    contract_value = packet.get("active_run_contract", {})
    if not (
        contract_value.get("is_primary_corpus") is True
        and contract_value.get("must_not_stop_or_modify") is True
        and contract_value.get("heavy_execution_deferred") is True
    ):
        errors.append("active-run safety contract is missing or weakened")

    expected_fingerprints = build_packet()["fingerprints"]
    if packet.get("fingerprints") != expected_fingerprints:
        errors.append("plan/catalog/spec/probe fingerprint mismatch")

    source_rows = packet.get("source_manifest")
    if not isinstance(source_rows, list):
        errors.append("source_manifest must be a list")
    else:
        by_path = {row.get("path"): row for row in source_rows if isinstance(row, dict)}
        for relative in SOURCE_FILES:
            row = by_path.get(relative)
            path = ROOT / relative
            if row is None:
                errors.append(f"source manifest missing {relative}")
            elif not path.is_file():
                errors.append(f"required source file missing: {relative}")
            elif row.get("sha256") != _sha_file(path) or row.get("size") != path.stat().st_size:
                errors.append(f"source file fingerprint mismatch: {relative}")

    artifact_rows = packet.get("generated_artifacts")
    if not isinstance(artifact_rows, list):
        errors.append("generated_artifacts must be a list")
    else:
        for row in artifact_rows:
            if not isinstance(row, dict) or not row.get("exists"):
                continue
            path = ROOT / str(row.get("path"))
            if not path.is_file() or row.get("sha256") != _sha_file(path):
                errors.append(f"generated artifact fingerprint mismatch: {row.get('path')}")
    return errors


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _selftest() -> int:
    packet = build_packet()
    assert verify_packet(packet) == []
    assert packet == build_packet()
    broken = copy.deepcopy(packet)
    broken["source_manifest"][0]["sha256"] = "0" * 64
    broken = _stamp_packet(broken)
    assert any("source file fingerprint" in error for error in verify_packet(broken))
    print("appendix_handoff.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", action="store_true")
    parser.add_argument("--write", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.packet:
        print(json.dumps(build_packet(), indent=2, sort_keys=True))
        return 0
    if args.write is not None:
        _atomic_json(args.write, build_packet())
        return 0
    if args.verify is not None:
        errors = verify_packet(_load(args.verify))
    elif args.audit:
        errors = verify_packet(build_packet())
    else:
        parser.error("choose --packet, --write, --verify, --audit, or --selftest")
        return 64
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
