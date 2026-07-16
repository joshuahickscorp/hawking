#!/usr/bin/env python3.12
"""Fail-closed post-run bridge from the corpus to TQ hardware and spec-decode gates.

This module is a deterministic plan/status surface.  It never launches Metal,
opens a model, hashes the live corpus, or executes a speculative proposer.  Its
main job is to distinguish useful vendor microbenchmarks from the Hawking-core
and artifact-bound evidence still required for a runtime decision.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

import appendix_contract
import spec_reentry_scaffold
import tq_runtime_matrix
import tq_runtime_probe


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_postrun.v1"
DEFAULT_PROBE = ROOT / "target" / "release" / "hawking-tq-device-probe"
DEFAULT_SPEC_PROBE = ROOT / "target" / "release" / "hawking-tq-spec-probe"


def _gate(
    gate_id: str,
    *,
    tier: str,
    command: list[str] | None,
    runtime_paths: list[str],
    source_surface: str,
    strict_receipt: bool,
    limitations: list[str],
    finalize_command: list[str] | None = None,
) -> dict:
    return {
        "id": gate_id,
        "tier": tier,
        "command": command,
        "runtime_paths": runtime_paths,
        "source_surface": source_surface,
        "strict_device_receipt_emitted": strict_receipt,
        "finalize_command": finalize_command,
        "requires_exclusive_heavy_lease": tier != "static_compile",
        "limitations": limitations,
    }


def gate_inventory() -> list[dict]:
    vendor_manifest = "vendor/strand-decode-kernel/Cargo.toml"
    vendor_shader = "vendor/strand-decode-kernel/shaders/strand_bitslice.metal"
    hawking_shader = "crates/hawking-core/shaders/strand_bitslice.metal"
    shared = ["cargo", "run", "--release", "--manifest-path", vendor_manifest, "--bin"]
    return [
        _gate(
            "compile_vendor_tq_gates",
            tier="static_compile",
            command=[
                "cargo", "check", "--manifest-path", vendor_manifest,
                "--bin", "gate-bitslice", "--bin", "gate-tablecompact",
                "--bin", "gate-coopwindow", "--bin", "gate-token-buffer",
                "--bin", "gate-bitslice-staged",
            ],
            runtime_paths=["stored", "compact", "hashed", "computed"],
            source_surface="Rust host compilation only",
            strict_receipt=False,
            limitations=["does not invoke Metal", "does not prove Hawking-core dispatch"],
        ),
        _gate(
            "build_hawking_tq_release_probes",
            tier="static_compile",
            command=[
                "cargo", "build", "--release", "-p", "hawking", "--features", "tq",
                "--bin", "hawking-tq-device-probe",
                "--bin", "hawking-tq-spec-probe",
            ],
            runtime_paths=["stored", "compact", "hashed", "computed"],
            source_surface=(
                "crates/hawking/src/tq_device_probe.rs + "
                "crates/hawking/src/tq_spec_probe.rs"
            ),
            strict_receipt=False,
            limitations=[
                "build-only gate; does not invoke Metal or open an artifact",
                "the resulting binaries must still be hash-bound by each raw bundle",
                "must run after the active Doctor heavy owner releases the machine",
            ],
        ),
        _gate(
            "vendor_bitslice_identity",
            tier="vendor_microbench",
            command=shared + ["gate-bitslice"],
            runtime_paths=["stored", "computed"],
            source_surface=vendor_shader,
            strict_receipt=False,
            limitations=[
                "ad-hoc text output",
                "only waits for strand-qat, so the Appendix owner interlock must wrap it",
                "vendor shader evidence is not Hawking-core end-to-end evidence",
            ],
        ),
        _gate(
            "vendor_compact_metadata",
            tier="vendor_microbench",
            command=shared + ["gate-tablecompact"],
            runtime_paths=["stored", "compact"],
            source_surface="self-contained MSL in gate-tablecompact.rs",
            strict_receipt=False,
            limitations=[
                "self-contained experiment rather than deployed Hawking shader",
                "fixed synthetic ffn_down cells",
                "ad-hoc text output",
            ],
        ),
        _gate(
            "vendor_hashed_window",
            tier="vendor_microbench",
            command=shared + ["gate-coopwindow"],
            runtime_paths=["stored", "hashed"],
            source_surface="self-contained MSL in gate-coopwindow.rs",
            strict_receipt=False,
            limitations=[
                "hash+quantile and cooperative-window research, not Acklam computed mode",
                "fixed synthetic ffn_down cells",
                "ad-hoc text output",
            ],
        ),
        _gate(
            "vendor_token_command_buffer",
            tier="vendor_microbench",
            command=shared + ["gate-token-buffer"],
            runtime_paths=["stored"],
            source_surface=vendor_shader,
            strict_receipt=False,
            limitations=[
                "synthetic 0.5B tensor count with columns padded to 256",
                "best-of-30 timing is not the Appendix percentile contract",
                "does not serve a real .tq artifact",
            ],
        ),
        _gate(
            "vendor_staged_decode_writes",
            tier="vendor_microbench",
            command=shared + ["gate-bitslice-staged", "--", "--force"],
            runtime_paths=["stored"],
            source_surface="vendor/strand-decode-kernel/shaders/strand_bitslice_staged.metal",
            strict_receipt=False,
            limitations=[
                "--force is only safe inside the shared heavy lease after owner recheck",
                "small-shape microbench",
                "decode-only result cannot establish token speed",
            ],
        ),
        _gate(
            "hawking_core_runtime_matrix",
            tier="hawking_core_artifact",
            command=[
                "python3.12", "tools/condense/appendix_device_runner.py",
                "--run-raw", "<artifact.tq>", "--runtime-path", "<mode>",
                "--cell-id", "<matrix-cell-id>",
                "--residual-artifact", "<independent-residual-artifact.tq>",
                "--residual-tensor", "<residual-tensor-name>",
                "--output", "<raw-bundle.json>",
            ],
            runtime_paths=["stored", "compact", "hashed", "computed"],
            source_surface=hawking_shader,
            strict_receipt=True,
            finalize_command=[
                "python3.12", "tools/condense/appendix_device_runner.py",
                "--finalize", "<raw-bundle.json>", "--counters", "<physical-counters.json>",
                "--cell-id", "<matrix-cell-id>", "--output", "<receipt.json>",
            ],
            limitations=[
                "release probe binary must be built before the post-run lease",
                "occupancy/bandwidth/energy counters must be supplied from bound physical captures",
                "residual coverage is credited only when the explicit second artifact executes the accumulate reduction; omit both residual flags for an honest single-pass cell",
            ],
        ),
        _gate(
            "tq_native_batched_verifier",
            tier="hawking_core_artifact",
            command=[
                "python3.12", "tools/condense/spec_tq_runner.py", "--run-raw",
                "--weights", "<model.gguf>", "--artifact", "<artifact.tq>",
                "--prompts", "<token-prompts.json>", "--runtime-path", "<mode>",
                "--output", "<spec-raw-bundle.json>",
            ],
            runtime_paths=["stored", "compact", "hashed", "computed"],
            source_surface="Hawking-core verifier path",
            strict_receipt=True,
            finalize_command=[
                "python3.12", "tools/condense/spec_tq_runner.py", "--finalize",
                "<spec-raw-bundle.json>", "--counters", "<physical-counters.json>",
                "--parity-output", "<parity-receipt.json>",
                "--curve-output", "<curve-receipt.json>",
            ],
            limitations=[
                "release spec probe binary and a hash-bound corpus token-prompt set are required",
                "all seven linears per layer must be TQ-owned and GPU-resident",
                "physical energy/GPU-time/byte counters are required to finalize the cost curve",
                "must be exact before any proposer timing is admissible",
            ],
        ),
    ]


def _stage(stage_id: str, depends_on: list[str], gates: list[str], output: str) -> dict:
    return {
        "id": stage_id,
        "depends_on": depends_on,
        "gates": gates,
        "output": output,
        "state": "deferred",
    }


def build_plan(label: str = "CORPUS") -> dict:
    probe = tq_runtime_probe.build_probe()
    device = tq_runtime_matrix.build_matrix(probe)
    spec = spec_reentry_scaffold.build_matrix(label)
    gates = gate_inventory()
    stages = [
        _stage("A0_freeze_corpus", [], [], "hawking.appendix_corpus_index.v3"),
        _stage(
            "A1_compile_surfaces",
            ["A0_freeze_corpus"],
            ["compile_vendor_tq_gates", "build_hawking_tq_release_probes"],
            "compile logs and two hash-bindable release probes",
        ),
        _stage(
            "A2_vendor_research",
            ["A1_compile_surfaces"],
            [
                "vendor_bitslice_identity", "vendor_compact_metadata",
                "vendor_hashed_window", "vendor_token_command_buffer",
                "vendor_staged_decode_writes",
            ],
            "raw research evidence only",
        ),
        _stage(
            "A3_hawking_tq_device_matrix",
            ["A1_compile_surfaces"],
            ["hawking_core_runtime_matrix"],
            "hawking.tq_runtime_device.v1 receipts",
        ),
        _stage(
            "B0_tq_batched_verifier",
            ["A3_hawking_tq_device_matrix"],
            ["tq_native_batched_verifier"],
            "hawking.spec_tq_batched_parity.v1 receipts",
        ),
        _stage(
            "B1_spec_cost_and_proposers",
            ["B0_tq_batched_verifier"],
            [],
            "P1-P3 Appendix spec receipts",
        ),
        _stage(
            "B2_parallel_tree_composition",
            ["B1_spec_cost_and_proposers"],
            [],
            "P4-P6 Appendix spec receipts",
        ),
    ]
    ids = {stage["id"] for stage in stages}
    if any(not set(stage["depends_on"]) <= ids for stage in stages):
        raise AssertionError("post-run stage dependency closure failed")
    gate_ids = {gate["id"] for gate in gates}
    if any(not set(stage["gates"]) <= gate_ids for stage in stages):
        raise AssertionError("post-run gate reference closure failed")
    payload = {
        "schema": SCHEMA,
        "label": label,
        "execution_supported": False,
        "reason_execution_is_fail_closed": (
            "both Hawking-core artifact adapters are implemented but raw device execution is "
            "lease-gated; physical counter captures and later proposer/tree adapters remain"
        ),
        "active_corpus_must_not_be_mutated": True,
        "source_probe_sha256": appendix_contract.canonical_sha256(probe),
        "tq_device_matrix_sha256": appendix_contract.canonical_sha256(device),
        "spec_matrix_sha256": appendix_contract.canonical_sha256(spec),
        "counts": {
            "tq_static_cells": len(probe["cells"]),
            "tq_device_cells": device["counts"],
            "spec_cells": len(spec["cells"]),
            "mapped_existing_gates": sum(gate["command"] is not None for gate in gates),
            "missing_artifact_adapters": sum(gate["command"] is None for gate in gates),
        },
        "gates": gates,
        "stages": stages,
        "promotion_rule": (
            "vendor microbenchmarks may select an implementation candidate, but only "
            "artifact-bound Hawking-core receipts may change a runtime default"
        ),
    }
    payload["plan_sha256"] = appendix_contract.canonical_sha256(payload)
    return payload


def _metal_compiler() -> str | None:
    if shutil.which("xcrun") is None:
        return None
    proc = subprocess.run(
        ["xcrun", "--find", "metal"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def status(
    plan: dict | None = None,
    *,
    active_owners: list[dict] | None = None,
    metal_compiler: str | None | object = ...,
    probe_available: bool | None = None,
    spec_probe_available: bool | None = None,
    platform_name: str | None = None,
) -> dict:
    plan = build_plan() if plan is None else plan
    owners = spec_reentry_scaffold.active_heavy_owners() if active_owners is None else active_owners
    compiler = _metal_compiler() if metal_compiler is ... else metal_compiler
    probe = DEFAULT_PROBE.is_file() if probe_available is None else probe_available
    spec_probe = (
        DEFAULT_SPEC_PROBE.is_file()
        if spec_probe_available is None
        else spec_probe_available
    )
    host_platform = platform.system() if platform_name is None else platform_name
    ready = not owners and host_platform == "Darwin" and probe
    spec_ready = not owners and host_platform == "Darwin" and spec_probe
    return {
        "schema": "hawking.appendix_postrun_status.v1",
        "plan_sha256": plan["plan_sha256"],
        "active_heavy_owner_count": len(owners),
        "active_heavy_owners": owners,
        "metal_compiler": compiler,
        "offline_metal_compiler_required": False,
        "runtime_metal_source_compilation": host_platform == "Darwin",
        "release_probe_available": probe,
        "release_spec_probe_available": spec_probe,
        "device_environment_ready": ready,
        "device_raw_execution_ready": ready,
        "spec_raw_execution_ready": spec_ready,
        "execution_ready": False,
        "blockers": [
            *(["active heavy owners still own the machine"] if owners else []),
            *(["device runner requires macOS"] if host_platform != "Darwin" else []),
            *(["release hawking-tq-device-probe binary is not built"] if not probe else []),
            *(["release hawking-tq-spec-probe binary is not built"] if not spec_probe else []),
            "physical counter captures and a hash-bound corpus token-prompt set are still required",
            "P2-P6 proposer, learned-draft, tree, and composition evidence remains deferred",
        ],
    }


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _selftest() -> int:
    first = build_plan()
    assert first == build_plan()
    assert first["counts"]["tq_static_cells"] == 1036
    assert first["counts"]["tq_device_cells"]["deferred"] == 496
    assert first["counts"]["mapped_existing_gates"] == 9
    assert first["counts"]["missing_artifact_adapters"] == 0
    assert not first["execution_supported"]
    busy = status(
        first, active_owners=[{"pid": 1}], metal_compiler=None,
        probe_available=True, spec_probe_available=True, platform_name="Darwin",
    )
    assert not busy["device_environment_ready"]
    idle = status(
        first, active_owners=[], metal_compiler=None,
        probe_available=True, spec_probe_available=True, platform_name="Darwin",
    )
    assert idle["device_environment_ready"]
    assert idle["device_raw_execution_ready"]
    assert idle["spec_raw_execution_ready"]
    assert not idle["execution_ready"]
    print("appendix_postrun.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--write", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    plan = build_plan()
    if args.write is not None:
        _atomic_json(args.write, plan)
        return 0
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.status:
        report = status(plan)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["device_environment_ready"] else 75
    parser.error("choose --plan, --status, --write, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
