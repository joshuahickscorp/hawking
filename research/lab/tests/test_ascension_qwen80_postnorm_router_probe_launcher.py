"""Focused terminal-capture tests for the Qwen80 postnorm/router launcher."""
from __future__ import annotations

import json
import shlex
import stat

from lab.receipts import verify
from lab.operators import ascension_qwen80_postnorm_router_probe_launcher as launcher


def _probe(tmp_path, body: str):
    marker = tmp_path / "runs.txt"
    path = tmp_path / "ascension_qwen80_direct_packed_postnorm_router_top10"
    path.write_text(
        "#!/bin/sh\n"
        f"printf run >> {shlex.quote(str(marker))}\n"
        f"{body}\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path, marker


def _config(tmp_path, probe, *, mode: str = "cpu-oracle"):
    manifest = tmp_path / "manifest.json"
    admission = tmp_path / "admission.json"
    manifest.write_text("{}\n", encoding="utf-8")
    admission.write_text("{}\n", encoding="utf-8")
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=manifest,
        admission_current=admission,
        capture_dir=tmp_path / "capture",
        mode=mode,
        workers=2,
        timeout_seconds=10.0,
    )


def test_outer_launcher_reaps_nonzero_child_and_never_retries_same_capture(tmp_path) -> None:
    probe, marker = _probe(tmp_path, 'echo "child stdout"; echo "child stderr" >&2; exit 7')
    config = _config(tmp_path, probe)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_NONZERO")
    assert receipt["child"]["terminal"] == {
        "reaped": True,
        "timed_out": False,
        "returncode": 7,
        "exit_code": 7,
        "signal": None,
    }
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "child stdout\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "child stderr\n"
    persisted = json.loads((config.capture_dir / launcher.TERMINAL_FILENAME).read_text(encoding="utf-8"))
    assert persisted == receipt
    assert launcher.run_attempt(config) == receipt
    assert marker.read_text(encoding="utf-8") == "run"


def test_outer_launcher_binds_zero_exit_to_expected_inner_component_receipt(tmp_path) -> None:
    body = (
        'capture=""; previous=""; '
        'for value in "$@"; do '
        'if [ "$previous" = "--capture-dir" ]; then capture="$value"; break; fi; '
        'previous="$value"; done; '
        'mkdir "$capture"; '
        "printf '%s\\n' "
        "'{\"status\":\"EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_CPU_ORACLE_READY_METAL_LEASE_REQUIRED\",\"mode\":\"cpu-oracle\",\"metal_device_or_dispatch_performed\":false}' "
        '> "$capture/receipt.json"; exit 0'
    )
    probe, marker = _probe(tmp_path, body)
    config = _config(tmp_path, probe)

    receipt = launcher.run_attempt(config)

    assert receipt["status"] == "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"
    assert receipt["inner_probe_capture"]["status"] == launcher.EXPECTED_CPU_STATUS
    assert receipt["inner_probe_capture"]["metal_performed"] is False
    assert receipt["child"]["terminal"]["reaped"] is True
    assert marker.read_text(encoding="utf-8") == "run"
    verify(receipt)


def test_metal_mode_requires_an_immutable_lease_receipt(tmp_path) -> None:
    probe, _ = _probe(tmp_path, "exit 0")
    config = _config(tmp_path, probe, mode="metal")
    try:
        launcher.run_attempt(config)
    except launcher.PostnormRouterLauncherError as error:
        assert "lease" in str(error)
    else:
        raise AssertionError("metal launch without a lease was accepted")
