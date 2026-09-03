"""Focused non-GPU tests for the future Qwen80 shared-expert outer launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_shared_expert_probe_launcher as launcher


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


def _inputs(tmp_path: Path) -> dict[str, Path | str]:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"fixture": "manifest"})
    manifest_evidence = _evidence(manifest)

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
                },
                "admission_receipt": {"seal_sha256": "a" * 64},
            }
        ),
    )
    admission_evidence = _evidence(admission)
    admission_document = json.loads(admission.read_text(encoding="utf-8"))

    cpu_inner = tmp_path / "shared-expert-cpu-inner.json"
    _write_json(
        cpu_inner,
        {
            "schema": launcher.EXPECTED_INNER_SCHEMA,
            "status": launcher.EXPECTED_CPU_STATUS,
            "mode": "cpu-oracle",
            "metal_device_or_dispatch_performed": False,
            "shared_expert_only": True,
            "durable_capture": {"receipt_written_last_is_completion_marker": True},
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "admission_current_path": admission_evidence["path"],
                "admission_receipt_seal_sha256": "a" * 64,
            },
        },
    )
    cpu_inner_evidence = _evidence(cpu_inner)

    baseline = tmp_path / "sealed-shared-expert-cpu-baseline.json"
    _write_json(
        baseline,
        seal(
            {
                "schema": launcher.CPU_BASELINE_WRAPPER_SCHEMA,
                "status": launcher.CPU_BASELINE_WRAPPER_STATUS,
                "source_binding": {
                    "manifest": manifest_evidence,
                    "admission_current": admission_evidence,
                    "admission_receipt_seal_sha256": "a" * 64,
                },
                "cpu_inner_receipt": cpu_inner_evidence,
            }
        ),
    )
    baseline_evidence = _evidence(baseline)
    baseline_document = json.loads(baseline.read_text(encoding="utf-8"))

    lease = tmp_path / "shared-expert-quiet-metal-lease.json"
    _write_json(
        lease,
        seal(
            {
                "schema": launcher.SHARED_EXPERT_LEASE_SCHEMA,
                "status": launcher.SHARED_EXPERT_LEASE_STATUS,
                "execution_policy": {
                    "component": launcher.SHARED_EXPERT_LEASE_COMPONENT,
                    "quiet_qwen80_device_lease": True,
                    "strict_math": True,
                    "timing_or_benchmarking_allowed": False,
                    "complete_layer_or_token_allowed": False,
                    "tps_or_tg_claim_allowed": False,
                },
                "artifact_binding": {
                    "manifest_document_sha256": manifest_evidence["sha256"],
                    "admission_receipt_seal_sha256": "a" * 64,
                },
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
        "cpu_inner": cpu_inner,
        "baseline": baseline,
        "lease": lease,
        "manifest_sha256": str(manifest_evidence["sha256"]),
        "admission_pointer_seal": str(admission_document["seal_sha256"]),
        "admission_receipt_seal": "a" * 64,
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
        "shared_expert_only": True,
        "routed_expert_sum_performed": False,
        "moe_combine_performed": False,
        "second_residual_performed": False,
        "durable_capture": {"receipt_written_last_is_completion_marker": True},
        "artifact_binding": {
            "manifest_path": str(config.manifest.resolve()),
            "manifest_document_sha256": inputs["manifest_sha256"],
            "admission_current_path": str(config.admission_current.resolve()),
            "admission_pointer_seal_sha256": inputs["admission_pointer_seal"],
            "admission_receipt_seal_sha256": inputs["admission_receipt_seal"],
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
                "schema": launcher.SHARED_EXPERT_LEASE_SCHEMA,
                "status": launcher.SHARED_EXPERT_LEASE_STATUS,
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


def test_metal_requires_a_sealed_component_lease_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, include_lease=False)

    with pytest.raises(launcher.SharedExpertProbeLauncherError, match="lease"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()


def test_outer_launcher_reaps_nonzero_child_and_replays_without_second_launch(tmp_path: Path) -> None:
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
    assert receipt["one_shot"]["automatic_retry_disabled"] is True
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "child stdout\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "child stderr\n"
    persisted = json.loads((config.capture_dir / launcher.TERMINAL_FILENAME).read_text(encoding="utf-8"))
    assert persisted == receipt

    assert launcher.run_attempt(config) == receipt
    assert marker.read_text(encoding="utf-8") == "run"


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


def test_outer_launcher_binds_sealed_cpu_baseline_to_strict_metal_inner_receipt(
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

    assert receipt["status"] == "CAPTURED_QWEN80_SHARED_EXPERT_OUTER_TERMINAL_COMPONENT_ONLY"
    assert receipt["inner_probe_capture"]["binding_valid"] is True
    assert receipt["source_binding"]["cpu_baseline_seal_sha256"] == inputs["baseline_seal"]
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
