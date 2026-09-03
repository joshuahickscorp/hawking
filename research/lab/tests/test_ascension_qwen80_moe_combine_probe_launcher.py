"""Focused non-GPU tests for the future Qwen80 MoE-combine outer launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_moe_combine_probe_launcher as launcher


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


def _top10_binding(
    router: Path, router_outer: Path, router_outer_seal: str
) -> dict[str, object]:
    return {
        "router_receipt_path": str(router.resolve()),
        "router_receipt_sha256": _sha256(router),
        "router_outer_receipt_path": str(router_outer.resolve()),
        "router_outer_receipt_sha256": _sha256(router_outer),
        "router_outer_receipt_seal_sha256": router_outer_seal,
        "ids": list(launcher.SOURCE_TOP10_IDS),
    }


def _inputs(tmp_path: Path) -> dict[str, Path | str]:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        seal({"schema": launcher.CURRENT_MANIFEST_SCHEMA, "status": "fixture complete artifact"}),
    )
    manifest_evidence = _evidence(manifest)
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))

    admission = tmp_path / "admission-current.json"
    _write_json(
        admission,
        seal(
            {
                "schema": launcher.CURRENT_ADMISSION_SCHEMA,
                "status": launcher.CURRENT_ADMISSION_STATUS,
                "complete_manifest": {
                    "path": manifest_evidence["path"],
                    "document_sha256": manifest_evidence["sha256"],
                    "seal_sha256": manifest_document["seal_sha256"],
                },
                "admission_receipt": {"seal_sha256": "a" * 64},
            }
        ),
    )
    admission_evidence = _evidence(admission)
    admission_document = json.loads(admission.read_text(encoding="utf-8"))

    router = tmp_path / "router-inner-receipt.json"
    _write_json(
        router,
        {
            "schema": launcher.UPSTREAM_ROUTER_INNER_SCHEMA,
            "status": launcher.UPSTREAM_ROUTER_INNER_STATUS,
            "mode": "metal",
            "component_only": True,
            "metal_device_or_dispatch_performed": True,
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_current_path": admission_evidence["path"],
                "admission_receipt_seal_sha256": "a" * 64,
                "layer": 0,
                "experts_per_token": 10,
            },
            "source_stable_top10_router": {
                "ids": list(launcher.SOURCE_TOP10_IDS),
                "device_ids": list(launcher.SOURCE_TOP10_IDS),
                "device_ids_exact_match": True,
                "ids_unique_and_in_range": True,
                "renormalized_weights": [0.1] * 10,
            },
        },
    )
    router_evidence = _evidence(router)

    router_outer = tmp_path / "router-outer-receipt.json"
    _write_json(
        router_outer,
        seal(
            {
                "schema": launcher.UPSTREAM_ROUTER_OUTER_SCHEMA,
                "status": launcher.UPSTREAM_ROUTER_OUTER_STATUS,
                "source_binding": {
                    "manifest": manifest_evidence,
                    "admission_current": admission_evidence,
                },
                "inner_probe_capture": {
                    "present": True,
                    "path": router_evidence["path"],
                    "sha256": router_evidence["sha256"],
                    "schema": launcher.UPSTREAM_ROUTER_INNER_SCHEMA,
                    "status": launcher.UPSTREAM_ROUTER_INNER_STATUS,
                    "mode": "metal",
                    "metal_performed": True,
                },
            }
        ),
    )
    router_outer_evidence = _evidence(router_outer)
    router_outer_document = json.loads(router_outer.read_text(encoding="utf-8"))
    top10_binding = _top10_binding(
        router, router_outer, str(router_outer_document["seal_sha256"])
    )

    cpu_inner = tmp_path / "moe-combine-cpu-inner.json"
    _write_json(
        cpu_inner,
        {
            "schema": launcher.EXPECTED_INNER_SCHEMA,
            "status": launcher.EXPECTED_CPU_STATUS,
            "mode": "cpu-oracle",
            "metal_device_or_dispatch_performed": False,
            "component_only": True,
            "routed_expert_aggregation_performed": True,
            "shared_expert_add_performed": True,
            "second_residual_performed": True,
            "complete_layer_or_token_performed": False,
            "durable_capture": {"receipt_written_last_is_completion_marker": True},
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "manifest_seal_sha256": manifest_document["seal_sha256"],
                "admission_current_path": admission_evidence["path"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
            "source_top10_binding": top10_binding,
        },
    )
    cpu_inner_evidence = _evidence(cpu_inner)

    baseline = tmp_path / "sealed-moe-combine-cpu-baseline.json"
    _write_json(
        baseline,
        seal(
            {
                "schema": launcher.CPU_BASELINE_WRAPPER_SCHEMA,
                "status": launcher.CPU_BASELINE_WRAPPER_STATUS,
                "source_binding": {
                    "manifest": manifest_evidence,
                    "manifest_seal_sha256": manifest_document["seal_sha256"],
                    "admission_current": admission_evidence,
                    "admission_receipt_seal_sha256": "a" * 64,
                    "source_top10_binding": top10_binding,
                },
                "cpu_inner_receipt": cpu_inner_evidence,
            }
        ),
    )
    baseline_evidence = _evidence(baseline)
    baseline_document = json.loads(baseline.read_text(encoding="utf-8"))

    lease = tmp_path / "moe-combine-quiet-metal-lease.json"
    _write_json(
        lease,
        seal(
            {
                "schema": launcher.MOE_COMBINE_LEASE_SCHEMA,
                "status": launcher.MOE_COMBINE_LEASE_STATUS,
                "execution_policy": {
                    "component": launcher.MOE_COMBINE_LEASE_COMPONENT,
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
                "source_top10_binding": top10_binding,
                "cpu_baseline_binding": {
                    "receipt_path": baseline_evidence["path"],
                    "receipt_document_sha256": baseline_evidence["sha256"],
                    "schema": launcher.CPU_BASELINE_WRAPPER_SCHEMA,
                    "status": launcher.CPU_BASELINE_WRAPPER_STATUS,
                    "seal_sha256": baseline_document["seal_sha256"],
                },
            }
        ),
    )
    return {
        "manifest": manifest,
        "admission": admission,
        "router": router,
        "router_outer": router_outer,
        "cpu_inner": cpu_inner,
        "baseline": baseline,
        "lease": lease,
        "manifest_sha256": str(manifest_evidence["sha256"]),
        "manifest_seal": str(manifest_document["seal_sha256"]),
        "admission_pointer_seal": str(admission_document["seal_sha256"]),
        "admission_receipt_seal": "a" * 64,
        "router_sha256": str(router_evidence["sha256"]),
        "router_outer_sha256": str(router_outer_evidence["sha256"]),
        "router_outer_seal": str(router_outer_document["seal_sha256"]),
        "baseline_sha256": str(baseline_evidence["sha256"]),
        "baseline_seal": str(baseline_document["seal_sha256"]),
        "lease_sha256": _sha256(lease),
        "lease_seal": str(json.loads(lease.read_text(encoding="utf-8"))["seal_sha256"]),
    }


def _probe(tmp_path: Path, body: str) -> tuple[Path, Path]:
    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    probe.write_text(
        "#!/bin/sh\n"
        f"printf run >> {shlex.quote(str(marker))}\n"
        f"{body}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _config(
    tmp_path: Path,
    probe: Path,
    inputs: dict[str, Path | str],
    *,
    include_lease: bool = True,
) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        admission_current=inputs["admission"],  # type: ignore[arg-type]
        router_receipt=inputs["router"],  # type: ignore[arg-type]
        router_outer_receipt=inputs["router_outer"],  # type: ignore[arg-type]
        cpu_baseline_receipt=inputs["baseline"],  # type: ignore[arg-type]
        lease_receipt=inputs["lease"] if include_lease else None,  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        mode="metal",
        workers=2,
        timeout_seconds=10.0,
    )


def _inner_metal_body(config: launcher.LaunchConfig, inputs: dict[str, Path | str]) -> str:
    assert config.lease_receipt is not None
    receipt = {
        "schema": launcher.EXPECTED_INNER_SCHEMA,
        "status": launcher.EXPECTED_METAL_STATUS,
        "mode": "metal",
        "metal_device_or_dispatch_performed": True,
        "component_only": True,
        "routed_expert_aggregation_performed": True,
        "shared_expert_add_performed": True,
        "second_residual_performed": True,
        "complete_layer_or_token_performed": False,
        "durable_capture": {"receipt_written_last_is_completion_marker": True},
        "artifact_binding": {
            "manifest_path": str(config.manifest.resolve()),
            "manifest_document_sha256": inputs["manifest_sha256"],
            "manifest_seal_sha256": inputs["manifest_seal"],
            "admission_current_path": str(config.admission_current.resolve()),
            "admission_pointer_seal_sha256": inputs["admission_pointer_seal"],
            "admission_receipt_seal_sha256": inputs["admission_receipt_seal"],
        },
        "source_top10_binding": {
            "router_receipt_path": str(config.router_receipt.resolve()),
            "router_receipt_sha256": inputs["router_sha256"],
            "router_outer_receipt_path": str(config.router_outer_receipt.resolve()),
            "router_outer_receipt_sha256": inputs["router_outer_sha256"],
            "router_outer_receipt_seal_sha256": inputs["router_outer_seal"],
            "ids": list(launcher.SOURCE_TOP10_IDS),
        },
        "cpu_baseline_binding": {
            "receipt_path": str(config.cpu_baseline_receipt.resolve()),
            "receipt_document_sha256": inputs["baseline_sha256"],
            "schema": launcher.CPU_BASELINE_WRAPPER_SCHEMA,
            "status": launcher.CPU_BASELINE_WRAPPER_STATUS,
            "seal_sha256": inputs["baseline_seal"],
        },
        "metal_execution_policy": {
            "strict_math_required": True,
            "timing_or_benchmarking_allowed": False,
            "complete_layer_or_token_allowed": False,
            "tps_or_tg_claim_allowed": False,
            "lease_binding": {
                "receipt_path": str(config.lease_receipt.resolve()),
                "receipt_document_sha256": inputs["lease_sha256"],
                "schema": launcher.MOE_COMBINE_LEASE_SCHEMA,
                "status": launcher.MOE_COMBINE_LEASE_STATUS,
                "seal_sha256": inputs["lease_seal"],
            },
        },
    }
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


def test_metal_requires_component_only_lease_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, include_lease=False)

    with pytest.raises(launcher.MoeCombineProbeLauncherError, match="lease"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_malformed_sealed_baseline_refuses_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    baseline = inputs["baseline"]
    assert isinstance(baseline, Path)
    _write_json(
        baseline,
        seal(
            {
                "schema": launcher.CPU_BASELINE_WRAPPER_SCHEMA,
                "status": launcher.CPU_BASELINE_WRAPPER_STATUS,
            }
        ),
    )
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs)

    with pytest.raises(launcher.MoeCombineProbeLauncherError, match="CPU baseline"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_launcher_reaps_nonzero_child_and_persists_stdout_stderr(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, 'echo "child stdout"; echo "child stderr" >&2; exit 7')
    config = _config(tmp_path, probe, inputs)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_NONZERO")
    assert receipt["child"]["terminal"] == {
        "reaped": True,
        "timed_out": False,
        "returncode": 7,
        "exit_code": 7,
        "signal": None,
    }
    assert receipt["one_shot"]["terminal_receipt_written_last"] is True
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "child stdout\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "child stderr\n"
    persisted = json.loads((config.capture_dir / launcher.TERMINAL_FILENAME).read_text(encoding="utf-8"))
    assert persisted == receipt


def test_outer_launcher_reaps_signal_terminated_child(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, 'echo "before signal"; kill -TERM $$')
    config = _config(tmp_path, probe, inputs)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_SIGNAL")
    assert receipt["child"]["terminal"] == {
        "reaped": True,
        "timed_out": False,
        "returncode": -15,
        "exit_code": None,
        "signal": 15,
    }
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"


def test_outer_launcher_replays_terminal_evidence_without_a_second_child(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 3")
    config = _config(tmp_path, probe, inputs)

    first = launcher.run_attempt(config)
    second = launcher.run_attempt(config)

    assert first["status"].endswith("CHILD_NONZERO")
    assert second == first
    assert marker.read_text(encoding="utf-8") == "run"


def test_outer_launcher_binds_sealed_baseline_and_exact_source_top10_to_future_inner(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    probe.write_text(
        "#!/bin/sh\n"
        f"printf run >> {shlex.quote(str(marker))}\n"
        f"{_inner_metal_body(config, inputs)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    receipt = launcher.run_attempt(config)

    assert receipt["status"] == "CAPTURED_QWEN80_MOE_COMBINE_OUTER_TERMINAL_COMPONENT_ONLY"
    assert receipt["inner_probe_capture"]["binding_valid"] is True
    assert receipt["source_binding"]["source_top10_ids"] == list(launcher.SOURCE_TOP10_IDS)
    assert receipt["source_binding"]["cpu_baseline_seal_sha256"] == inputs["baseline_seal"]
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
