"""Focused non-GPU tests for the Qwen80 route-0/expert-65 outer launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
import stat
from pathlib import Path

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_routed_expert_probe_launcher as launcher


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
    router_receipt = tmp_path / "router-inner-receipt.json"
    _write_json(
        router_receipt,
        {
            "schema": launcher.UPSTREAM_ROUTER_INNER_SCHEMA,
            "status": launcher.UPSTREAM_ROUTER_INNER_STATUS,
            "mode": "metal",
            "component_only": True,
            "metal_device_or_dispatch_performed": True,
            "artifact_binding": {
                "manifest_path": manifest_evidence["path"],
                "manifest_document_sha256": manifest_evidence["sha256"],
                "admission_current_path": str(admission.resolve()),
                "admission_receipt_seal_sha256": "a" * 64,
            },
        },
    )
    router_outer = tmp_path / "router-outer-receipt.json"
    _write_json(
        router_outer,
        seal(
            {
                "schema": launcher.UPSTREAM_ROUTER_OUTER_SCHEMA,
                "status": launcher.UPSTREAM_ROUTER_OUTER_STATUS,
                "source_binding": {
                    "manifest": manifest_evidence,
                    "admission_current": _evidence(admission),
                },
                "inner_probe_capture": {
                    "present": True,
                    "path": str(router_receipt.resolve()),
                    "sha256": _sha256(router_receipt),
                    "schema": launcher.UPSTREAM_ROUTER_INNER_SCHEMA,
                    "status": launcher.UPSTREAM_ROUTER_INNER_STATUS,
                    "mode": "metal",
                    "metal_performed": True,
                },
            }
        ),
    )
    return {
        "manifest": manifest,
        "admission": admission,
        "router_receipt": router_receipt,
        "router_outer": router_outer,
        "manifest_sha256": str(manifest_evidence["sha256"]),
        "admission_pointer_seal": json.loads(admission.read_text(encoding="utf-8"))["seal_sha256"],
        "router_outer_seal": json.loads(router_outer.read_text(encoding="utf-8"))["seal_sha256"],
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
    mode: str = "cpu-oracle",
    lease_receipt: Path | None = None,
) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        admission_current=inputs["admission"],  # type: ignore[arg-type]
        router_receipt=inputs["router_receipt"],  # type: ignore[arg-type]
        router_outer_receipt=inputs["router_outer"],  # type: ignore[arg-type]
        capture_dir=tmp_path / "outer-capture",
        mode=mode,
        workers=2,
        timeout_seconds=10.0,
        lease_receipt=lease_receipt,
    )


def _inner_cpu_body(config: launcher.LaunchConfig, inputs: dict[str, Path | str]) -> str:
    receipt = {
        "schema": launcher.EXPECTED_INNER_SCHEMA,
        "status": launcher.EXPECTED_CPU_STATUS,
        "mode": "cpu-oracle",
        "one_selected_expert_only": True,
        "metal_device_or_dispatch_performed": False,
        "durable_capture": {"receipt_written_last_is_completion_marker": True},
        "artifact_binding": {
            "manifest_path": str(config.manifest.resolve()),
            "manifest_document_sha256": inputs["manifest_sha256"],
            "admission_current_path": str(config.admission_current.resolve()),
            "admission_pointer_seal_sha256": inputs["admission_pointer_seal"],
            "admission_receipt_seal_sha256": "a" * 64,
        },
        "route_evidence": {
            "router_receipt_path": str(config.router_receipt.resolve()),
            "router_receipt_sha256": _sha256(config.router_receipt),
            "router_outer_receipt_path": str(config.router_outer_receipt.resolve()),
            "router_outer_receipt_sha256": _sha256(config.router_outer_receipt),
            "router_outer_receipt_seal_sha256": inputs["router_outer_seal"],
            "selected_expert": 65,
            "selected_route_index": 0,
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


def test_outer_launcher_binds_expected_cpu_inner_receipt_to_router_outer_chain(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 99")
    config = _config(tmp_path, probe, inputs)
    probe.write_text(
        "#!/bin/sh\n"
        f"printf run >> {shlex.quote(str(marker))}\n"
        f"{_inner_cpu_body(config, inputs)}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)

    receipt = launcher.run_attempt(config)

    assert receipt["status"] == "CAPTURED_QWEN80_ROUTED_EXPERT_65_OUTER_TERMINAL_COMPONENT_ONLY"
    assert receipt["inner_probe_capture"]["binding_valid"] is True
    assert receipt["inner_probe_capture"]["status"] == launcher.EXPECTED_CPU_STATUS
    assert receipt["source_binding"]["router_outer_seal_sha256"] == inputs["router_outer_seal"]
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"


def test_metal_mode_requires_route65_lease_before_any_child_starts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    probe, marker = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, inputs, mode="metal")

    with pytest.raises(launcher.RoutedExpertProbeLauncherError, match="lease"):
        launcher.run_attempt(config)

    assert not config.capture_dir.exists()
    assert not marker.exists()
