"""CPU-only tests for the future Qwen80 true-input all-ten MoE outer launcher.

The fake executable below only writes synthetic receipts into a temporary
directory.  It never loads an artifact or opens a Metal context.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_true_input_all_ten_moe_graph_launcher as launcher


ROUTE_IDS = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494]
ROUTE_WEIGHTS = [0.1] * launcher.TOP_K


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sealed_json(path: Path, payload: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    document = seal(payload)
    _write_json(path, document)
    return document, _evidence(path)


def _inputs(tmp_path: Path) -> dict[str, Path | str | list[int] | list[float]]:
    manifest = tmp_path / "manifest.json"
    manifest_document, manifest_evidence = _sealed_json(
        manifest,
        {"schema": launcher.MANIFEST_SCHEMA, "status": "FIXTURE_CURRENT_COMPLETE_ARTIFACT"},
    )

    admission = tmp_path / "admission-current.json"
    admission_document, admission_evidence = _sealed_json(
        admission,
        {
            "schema": launcher.ADMISSION_SCHEMA,
            "status": launcher.ADMISSION_STATUS,
            "complete_manifest": {
                "path": manifest_evidence["path"],
                "document_sha256": manifest_evidence["sha256"],
                "seal_sha256": manifest_document["seal_sha256"],
            },
            "admission_receipt": {"seal_sha256": "a" * 64},
        },
    )

    router_inner = tmp_path / "router-inner.json"
    _write_json(
        router_inner,
        {
            "schema": launcher.ROUTER_INNER_SCHEMA,
            "status": launcher.ROUTER_INNER_STATUS,
            "mode": "metal",
            "component_only": True,
            "metal_device_or_dispatch_performed": True,
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_current_path": admission_evidence["path"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
        },
    )
    router_inner_evidence = _evidence(router_inner)

    router_outer = tmp_path / "router-outer.json"
    router_outer_document, router_outer_evidence = _sealed_json(
        router_outer,
        {
            "schema": launcher.ROUTER_OUTER_SCHEMA,
            "status": launcher.ROUTER_OUTER_STATUS,
            "source_binding": {
                "manifest": manifest_evidence,
                "admission_current": admission_evidence,
            },
            "inner_probe_capture": {
                "present": True,
                "path": router_inner_evidence["path"],
                "sha256": router_inner_evidence["sha256"],
                "schema": launcher.ROUTER_INNER_SCHEMA,
                "status": launcher.ROUTER_INNER_STATUS,
                "mode": "metal",
                "metal_performed": True,
            },
        },
    )

    route_plan = tmp_path / "all-ten-route-plan.json"
    waves: list[dict[str, object]] = []
    for index, expert_id in enumerate(ROUTE_IDS):
        projections = {
            projection: {"artifact_sha256": f"{index + offset:x}".zfill(64)}
            for offset, projection in enumerate(("gate", "up", "down"), start=1)
        }
        waves.append(
            {
                "wave_index": index,
                "expert_id": expert_id,
                "normalized_weight": ROUTE_WEIGHTS[index],
                **projections,
            }
        )
    _write_json(
        route_plan,
        {
            "schema": launcher.ROUTE_PLAN_SCHEMA,
            "status": launcher.ROUTE_PLAN_STATUS,
            "layer": 0,
            "router_evidence": {
                "source_stable_route_ids": ROUTE_IDS,
                "source_stable_normalized_weights": ROUTE_WEIGHTS,
            },
            "deterministic_waves": waves,
        },
    )
    route_plan_evidence = _evidence(route_plan)

    first_residual = tmp_path / "first-residual-outer.json"
    first_residual_document, first_residual_evidence = _sealed_json(
        first_residual,
        {
            "schema": launcher.FIRST_RESIDUAL_SCHEMA,
            "status": launcher.FIRST_RESIDUAL_STATUS,
            "source_binding": {
                "manifest": manifest_evidence,
                "admission_current": admission_evidence,
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_pointer_seal_sha256": admission_document["seal_sha256"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
            "first_residual_output": {
                "layer": 0,
                "linear_state_slot": 0,
                "elements": launcher.HIDDEN,
                "same_command_graph_required": True,
                "sha256": "b" * 64,
            },
        },
    )

    bridge = tmp_path / "typed-bridge.json"
    bridge_document, bridge_evidence = _sealed_json(
        bridge,
        {
            "schema": launcher.BRIDGE_SCHEMA,
            "status": launcher.BRIDGE_STATUS,
            "source_binding": {
                "manifest": manifest_evidence,
                "admission_current": admission_evidence,
                "route_plan": route_plan_evidence,
                "first_residual_receipt": first_residual_evidence,
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_pointer_seal_sha256": admission_document["seal_sha256"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
            "typed_bridge": {
                "layer": 0,
                "route_count": launcher.TOP_K,
                "first_residual_elements": launcher.HIDDEN,
                "same_command_graph_required": True,
                "first_residual_output_sha256": "b" * 64,
                "first_residual_receipt_seal_sha256": first_residual_document["seal_sha256"],
                "compact_section_sha256": {
                    "gate_scales": "1" * 64,
                    "gate_signs": "2" * 64,
                    "up_scales": "3" * 64,
                    "up_signs": "4" * 64,
                    "down_scales": "5" * 64,
                    "down_signs": "6" * 64,
                },
            },
        },
    )

    # This is deliberately unsealed static ABI authority.  The sealed quiet
    # lease below pins its exact raw document digest before a future device
    # child may use it; it is not device or execution evidence by itself.
    fixed_abi = tmp_path / "fixed-abi-contract.json"
    _write_json(
        fixed_abi,
        {
            "schema": launcher.FIXED_ABI_SCHEMA,
            "status": launcher.FIXED_ABI_STATUS,
            "source_binding": {
                "model_id": launcher.FIXED_ABI_MODEL_ID,
                "model_key": launcher.FIXED_ABI_MODEL_KEY,
                "source_repository": launcher.FIXED_ABI_SOURCE_REPOSITORY,
                "source_revision": launcher.FIXED_ABI_SOURCE_REVISION,
                "source_config_sha256": launcher.FIXED_ABI_SOURCE_CONFIG_SHA256,
                "source_shard": launcher.FIXED_ABI_SOURCE_SHARD,
                "source_shard_sha256": launcher.FIXED_ABI_SOURCE_SHARD_SHA256,
                "manifest_schema": launcher.MANIFEST_SCHEMA,
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_receipt_seal_sha256": "a" * 64,
                "source_body_audit_seal_sha256": launcher.FIXED_ABI_SOURCE_BODY_AUDIT_SEAL,
                "source_revalidation_seal_sha256": launcher.FIXED_ABI_SOURCE_REVALIDATION_SEAL,
            },
            "geometry": {
                "layer": 0,
                "hidden": launcher.HIDDEN,
                "intermediate": 512,
                "experts": 512,
                "top_k": launcher.TOP_K,
                "group_size": 128,
                "rms_epsilon": "1e-6",
            },
            "external_authority": {
                "route_plan_schema": launcher.ROUTE_PLAN_SCHEMA,
                "route_plan_status": launcher.ROUTE_PLAN_STATUS,
                "first_residual_schema": launcher.FIRST_RESIDUAL_SCHEMA,
                "first_residual_status": launcher.FIRST_RESIDUAL_STATUS,
                "typed_bridge_schema": launcher.BRIDGE_SCHEMA,
                "typed_bridge_status": launcher.BRIDGE_STATUS,
                "route_payloads_materialized_here": False,
                "first_residual_materialized_here": False,
                "expected_topk_witness_materialized_here": False,
                "route_tensor_sha256s_materialized_here": False,
            },
            "fixed_14_dispatch_abi": [
                {"ordinal": index, "kernel": kernel}
                for index, kernel in enumerate(launcher.FIXED_ABI_KERNELS, start=1)
            ],
            "claim_boundary": {
                "artifact_scan_or_payload_open_performed": False,
                "metal_context_or_dispatch_performed": False,
                "runtime_watcher_server_registry_or_hcli_changed": False,
                "token_or_tps_claim": False,
                "execution_status": "PREPARED_NOT_EXECUTED",
            },
        },
    )
    fixed_abi_evidence = _evidence(fixed_abi)

    lease = tmp_path / "quiet-component-lease.json"
    lease_document, lease_evidence = _sealed_json(
        lease,
        {
            "schema": launcher.LEASE_SCHEMA,
            "status": launcher.LEASE_STATUS,
            "lease_id": "fixture-q80-true-input-all-ten",
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "automatic_retry_prohibited": True,
                "outer_reaped_capture_required": True,
            },
            "execution_policy": {
                "component": launcher.LEASE_COMPONENT,
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "artifact_binding": {
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
            "typed_bridge_binding": {
                "path": bridge_evidence["path"],
                "document_sha256": bridge_evidence["sha256"],
                "schema": launcher.BRIDGE_SCHEMA,
                "status": launcher.BRIDGE_STATUS,
                "seal_sha256": bridge_document["seal_sha256"],
            },
            "fixed_abi_contract_binding": {
                "path": fixed_abi_evidence["path"],
                "document_sha256": fixed_abi_evidence["sha256"],
                "schema": launcher.FIXED_ABI_SCHEMA,
                "status": launcher.FIXED_ABI_STATUS,
            },
        },
    )
    return {
        "manifest": manifest,
        "admission": admission,
        "router_inner": router_inner,
        "router_outer": router_outer,
        "route_plan": route_plan,
        "first_residual": first_residual,
        "typed_bridge": bridge,
        "fixed_abi": fixed_abi,
        "lease": lease,
        "manifest_sha256": str(manifest_evidence["sha256"]),
        "manifest_seal": str(manifest_document["seal_sha256"]),
        "admission_pointer_seal": str(admission_document["seal_sha256"]),
        "admission_receipt_seal": "a" * 64,
        "router_outer_seal": str(router_outer_document["seal_sha256"]),
        "first_residual_seal": str(first_residual_document["seal_sha256"]),
        "bridge_sha256": str(bridge_evidence["sha256"]),
        "bridge_seal": str(bridge_document["seal_sha256"]),
        "fixed_abi_sha256": str(fixed_abi_evidence["sha256"]),
        "lease_sha256": str(lease_evidence["sha256"]),
        "lease_seal": str(lease_document["seal_sha256"]),
        "route_plan_sha256": str(route_plan_evidence["sha256"]),
    }


def _probe(tmp_path: Path, body: str) -> tuple[Path, Path]:
    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{body}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _config(
    tmp_path: Path,
    probe: Path,
    inputs: dict[str, Path | str | list[int] | list[float]],
    *,
    include_lease: bool = True,
) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        admission_current=inputs["admission"],  # type: ignore[arg-type]
        router_receipt=inputs["router_inner"],  # type: ignore[arg-type]
        router_outer_receipt=inputs["router_outer"],  # type: ignore[arg-type]
        route_plan=inputs["route_plan"],  # type: ignore[arg-type]
        first_residual_receipt=inputs["first_residual"],  # type: ignore[arg-type]
        typed_bridge_receipt=inputs["typed_bridge"],  # type: ignore[arg-type]
        fixed_abi_contract=inputs["fixed_abi"],  # type: ignore[arg-type]
        lease_receipt=inputs["lease"] if include_lease else None,  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        workers=1,
        timeout_seconds=10.0,
    )


def _inner_receipt(config: launcher.LaunchConfig, inputs: dict[str, Path | str | list[int] | list[float]]) -> dict[str, object]:
    assert config.lease_receipt is not None
    return {
        "schema": launcher.EXPECTED_INNER_SCHEMA,
        "status": launcher.EXPECTED_INNER_STATUS,
        "mode": "metal",
        "metal_device_or_dispatch_performed": True,
        "component_only": True,
        "complete_layer_or_token_performed": False,
        "durable_capture": {
            "receipt_written_last_is_completion_marker": True,
            "outer_reaped_capture_required": True,
            "replay_guarded": True,
        },
        "artifact_binding": {
            "manifest_document_sha256": inputs["manifest_sha256"],
            "manifest_seal_sha256": inputs["manifest_seal"],
            "admission_pointer_seal_sha256": inputs["admission_pointer_seal"],
            "admission_receipt_seal_sha256": inputs["admission_receipt_seal"],
        },
        "typed_bridge_binding": {
            "receipt_path": str(config.typed_bridge_receipt.resolve()),
            "receipt_document_sha256": inputs["bridge_sha256"],
            "seal_sha256": inputs["bridge_seal"],
            "schema": launcher.BRIDGE_SCHEMA,
            "status": launcher.BRIDGE_STATUS,
            "first_residual_output_sha256": "b" * 64,
        },
        "fixed_abi_contract_binding": {
            "path": str(config.fixed_abi_contract.resolve()),
            "document_sha256": inputs["fixed_abi_sha256"],
            "schema": launcher.FIXED_ABI_SCHEMA,
            "status": launcher.FIXED_ABI_STATUS,
        },
        "route_plan_binding": {
            "path": str(config.route_plan.resolve()),
            "sha256": inputs["route_plan_sha256"],
        },
        "route_guard_readback": {
            "value": 1,
            "passed": True,
            "observed_ids": ROUTE_IDS,
            "expected_ids": ROUTE_IDS,
            "observed_weights": ROUTE_WEIGHTS,
            "expected_weights": ROUTE_WEIGHTS,
        },
        "readback_parity": {
            "all_ten_route_witnesses": launcher.TOP_K,
            "all_ten_route_cpu_device_parity_passed": True,
            "shared_expert_cpu_device_parity_passed": True,
            "routed_sum_cpu_device_parity_passed": True,
            "second_residual_cpu_device_parity_passed": True,
        },
        "metal_execution_policy": {
            "strict_math_required": True,
            "timing_or_benchmarking_allowed": False,
            "complete_layer_or_token_allowed": False,
            "tps_or_tg_claim_allowed": False,
            "lease_binding": {
                "receipt_path": str(config.lease_receipt.resolve()),
                "receipt_document_sha256": inputs["lease_sha256"],
                "seal_sha256": inputs["lease_seal"],
                "schema": launcher.LEASE_SCHEMA,
                "status": launcher.LEASE_STATUS,
            },
        },
    }


def _inner_body(receipt: dict[str, object]) -> str:
    rendered = shlex.quote(json.dumps(receipt, sort_keys=True))
    return (
        'capture=""; previous=""; '
        'for value in "$@"; do '
        'if [ "$previous" = "--capture-dir" ]; then capture="$value"; break; fi; '
        'previous="$value"; done; '
        'mkdir "$capture"; '
        f"printf '%s\\n' {rendered} > \"$capture/receipt.json\"; "
        'echo "child stdout"; echo "child stderr" >&2; exit 0'
    )


def test_requires_fresh_component_lease_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, include_lease=False)

    with pytest.raises(launcher.TrueInputAllTenMoeGraphLauncherError, match="lease"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_reaps_valid_inner_and_replays_without_second_child(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(_inner_receipt(config, inputs))}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    first = launcher.run_attempt(config)
    second = launcher.run_attempt(config)

    assert first["status"] == "CAPTURED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_TERMINAL_COMPONENT_ONLY"
    assert first["inner_probe_capture"]["binding_valid"] is True
    assert first["source_binding"]["route_plan"]["sha256"] == inputs["route_plan_sha256"]
    assert first["source_binding"]["typed_bridge_receipt_seal_sha256"] == inputs["bridge_seal"]
    assert second == first
    assert marker.read_text(encoding="utf-8") == "run"
    verify(first)
    persisted = json.loads((config.capture_dir / launcher.TERMINAL_FILENAME).read_text(encoding="utf-8"))
    assert persisted == first


def test_zero_exit_with_failed_route_guard_is_refused_and_retained(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    invalid = _inner_receipt(config, inputs)
    invalid["route_guard_readback"] = {
        "value": 0,
        "passed": False,
        "observed_ids": ROUTE_IDS,
        "expected_ids": ROUTE_IDS,
        "observed_weights": ROUTE_WEIGHTS,
        "expected_weights": ROUTE_WEIGHTS,
    }
    probe.write_text(
        "#!/bin/sh\n" f"printf run >> {shlex.quote(str(marker))}\n" f"{_inner_body(invalid)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT")
    assert receipt["inner_probe_capture"]["binding_valid"] is False
    assert "route guard" in receipt["inner_probe_capture"]["binding_error"]
    assert marker.read_text(encoding="utf-8") == "run"
    verify(receipt)


def test_typed_bridge_first_residual_drift_refuses_before_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    bridge = inputs["typed_bridge"]
    assert isinstance(bridge, Path)
    bad_bridge = json.loads(bridge.read_text(encoding="utf-8"))
    bad_bridge.pop("seal_sha256")
    typed = dict(bad_bridge["typed_bridge"])
    typed["first_residual_output_sha256"] = "c" * 64
    bad_bridge["typed_bridge"] = typed
    _write_json(bridge, seal(bad_bridge))
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)

    with pytest.raises(launcher.TrueInputAllTenMoeGraphLauncherError, match="first-residual identity"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_first_residual_accepts_current_pointer_reseal_with_stable_receipt(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    admission = inputs["admission"]
    manifest = inputs["manifest"]
    first_residual = inputs["first_residual"]
    assert isinstance(admission, Path)
    assert isinstance(manifest, Path)
    assert isinstance(first_residual, Path)
    current = json.loads(admission.read_text(encoding="utf-8"))
    current.pop("seal_sha256")
    current["recorded_at"] = "fixture-reseal"
    _write_json(admission, seal(current))
    manifest_evidence, manifest_seal = launcher._bind_manifest(manifest)
    admission_evidence, _, admission_receipt_seal = launcher._bind_admission(
        admission, manifest_evidence, manifest_seal
    )

    _, _, output_sha = launcher._bind_first_residual(
        first_residual,
        manifest=manifest_evidence,
        manifest_seal=manifest_seal,
        admission=admission_evidence,
        admission_receipt_seal=admission_receipt_seal,
    )

    assert output_sha == "b" * 64


def test_first_residual_refuses_current_pointer_manifest_or_admission_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    admission = inputs["admission"]
    manifest = inputs["manifest"]
    first_residual = inputs["first_residual"]
    assert isinstance(admission, Path)
    assert isinstance(manifest, Path)
    assert isinstance(first_residual, Path)
    current = json.loads(admission.read_text(encoding="utf-8"))
    current.pop("seal_sha256")
    current["admission_receipt"] = {"seal_sha256": "b" * 64}
    _write_json(admission, seal(current))
    manifest_evidence, manifest_seal = launcher._bind_manifest(manifest)
    admission_evidence, _, admission_receipt_seal = launcher._bind_admission(
        admission, manifest_evidence, manifest_seal
    )

    with pytest.raises(launcher.TrueInputAllTenMoeGraphLauncherError, match="first-residual artifact authority"):
        launcher._bind_first_residual(
            first_residual,
            manifest=manifest_evidence,
            manifest_seal=manifest_seal,
            admission=admission_evidence,
            admission_receipt_seal=admission_receipt_seal,
        )


def test_typed_bridge_accepts_current_pointer_reseal_with_stable_receipt(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    admission = inputs["admission"]
    manifest = inputs["manifest"]
    route_plan = inputs["route_plan"]
    first_residual = inputs["first_residual"]
    bridge = inputs["typed_bridge"]
    assert isinstance(admission, Path)
    assert isinstance(manifest, Path)
    assert isinstance(route_plan, Path)
    assert isinstance(first_residual, Path)
    assert isinstance(bridge, Path)
    current = json.loads(admission.read_text(encoding="utf-8"))
    current.pop("seal_sha256")
    current["recorded_at"] = "fixture-typed-bridge-reseal"
    _write_json(admission, seal(current))
    manifest_evidence, manifest_seal = launcher._bind_manifest(manifest)
    admission_evidence, _, admission_receipt_seal = launcher._bind_admission(
        admission, manifest_evidence, manifest_seal
    )
    route_evidence = launcher._file_evidence(route_plan, "route plan")
    first_evidence, first_seal, output_sha = launcher._bind_first_residual(
        first_residual,
        manifest=manifest_evidence,
        manifest_seal=manifest_seal,
        admission=admission_evidence,
        admission_receipt_seal=admission_receipt_seal,
    )

    _, bridge_seal = launcher._bind_typed_bridge(
        bridge,
        manifest=manifest_evidence,
        manifest_seal=manifest_seal,
        admission=admission_evidence,
        admission_receipt_seal=admission_receipt_seal,
        route_plan=route_evidence,
        first_residual=first_evidence,
        first_residual_seal=first_seal,
        first_residual_output_sha256=output_sha,
    )

    assert bridge_seal


def test_typed_bridge_refuses_current_pointer_manifest_or_admission_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    admission = inputs["admission"]
    manifest = inputs["manifest"]
    route_plan = inputs["route_plan"]
    first_residual = inputs["first_residual"]
    bridge = inputs["typed_bridge"]
    assert isinstance(admission, Path)
    assert isinstance(manifest, Path)
    assert isinstance(route_plan, Path)
    assert isinstance(first_residual, Path)
    assert isinstance(bridge, Path)
    current = json.loads(admission.read_text(encoding="utf-8"))
    current.pop("seal_sha256")
    current["admission_receipt"] = {"seal_sha256": "b" * 64}
    _write_json(admission, seal(current))
    manifest_evidence, manifest_seal = launcher._bind_manifest(manifest)
    admission_evidence, _, admission_receipt_seal = launcher._bind_admission(
        admission, manifest_evidence, manifest_seal
    )
    route_evidence = launcher._file_evidence(route_plan, "route plan")
    # Keep the historical antecedent unchanged so this exercises the typed
    # bridge's own immutable admission-receipt binding rather than stopping
    # at the earlier first-residual gate.
    first_document = json.loads(first_residual.read_text(encoding="utf-8"))
    first_evidence = launcher._file_evidence(first_residual, "first residual")
    first_seal = str(first_document["seal_sha256"])
    output_sha = "b" * 64

    with pytest.raises(launcher.TrueInputAllTenMoeGraphLauncherError, match="typed bridge artifact authority"):
        launcher._bind_typed_bridge(
            bridge,
            manifest=manifest_evidence,
            manifest_seal=manifest_seal,
            admission=admission_evidence,
            admission_receipt_seal=admission_receipt_seal,
            route_plan=route_evidence,
            first_residual=first_evidence,
            first_residual_seal=first_seal,
            first_residual_output_sha256=output_sha,
        )



def test_unsealed_fixed_abi_plan_is_pinned_by_the_sealed_lease_before_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    fixed_abi = inputs["fixed_abi"]
    assert isinstance(fixed_abi, Path)
    # Whitespace leaves the static plan's semantics intact but changes its raw
    # document identity.  The outer launcher must reject it through the sealed
    # lease binding rather than treating an unsealed plan as mutable authority.
    fixed_abi.write_text(fixed_abi.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)

    with pytest.raises(launcher.TrueInputAllTenMoeGraphLauncherError, match="fixed ABI identity drifted"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()
