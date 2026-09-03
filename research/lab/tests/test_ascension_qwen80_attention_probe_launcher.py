"""Focused non-GPU terminal-capture tests for the Qwen80 attention launcher."""
from __future__ import annotations

import json
import shlex
import stat

from lab.receipts import verify
from lab.operators import ascension_qwen80_attention_probe_launcher as launcher


def _probe_script(tmp_path, *, body: str) -> tuple[object, object]:
    marker = tmp_path / "child-runs.txt"
    probe = tmp_path / "ascension_qwen80_direct_packed_attention_stage"
    probe.write_text(
        "#!/bin/sh\n"
        f"printf run >> {shlex.quote(str(marker))}\n"
        f"{body}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe, marker


def _config(tmp_path, probe) -> launcher.LaunchConfig:
    return launcher.LaunchConfig(
        probe_bin=probe,
        manifest=tmp_path / "missing-manifest.json",
        expected_manifest_seal_sha256="a" * 64,
        expected_source_audit_seal_sha256="b" * 64,
        expected_source_revision="unit-test-revision",
        capture_dir=tmp_path / "outer-capture",
        mode="cpu-oracle",
        timeout_seconds=10.0,
    )


def test_outer_launcher_seals_nonzero_child_and_replays_without_second_launch(tmp_path) -> None:
    probe, marker = _probe_script(
        tmp_path, body='echo "probe stdout"; echo "probe stderr" >&2; exit 7'
    )
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
    assert receipt["child"]["pid"] > 0
    assert receipt["inner_probe_capture"]["present"] is False
    assert receipt["one_shot"]["automatic_retry_disabled"] is True
    verify(receipt)
    assert marker.read_text(encoding="utf-8") == "run"
    assert (config.capture_dir / launcher.OUTER_STDOUT).read_text(encoding="utf-8") == "probe stdout\n"
    assert (config.capture_dir / launcher.OUTER_STDERR).read_text(encoding="utf-8") == "probe stderr\n"
    persisted = json.loads(
        (config.capture_dir / launcher.TERMINAL_RECEIPT).read_text(encoding="utf-8")
    )
    assert persisted == receipt

    replay = launcher.run_attempt(config)

    assert replay == receipt
    assert marker.read_text(encoding="utf-8") == "run"


def test_outer_launcher_seals_signal_termination_and_reaps_child(tmp_path) -> None:
    probe, marker = _probe_script(tmp_path, body='echo "before signal"; kill -TERM $$')
    config = _config(tmp_path, probe)

    receipt = launcher.run_attempt(config)

    assert receipt["status"].endswith("CHILD_SIGNAL")
    assert receipt["child"]["terminal"]["reaped"] is True
    assert receipt["child"]["terminal"]["returncode"] == -15
    assert receipt["child"]["terminal"]["exit_code"] is None
    assert receipt["child"]["terminal"]["signal"] == 15
    assert marker.read_text(encoding="utf-8") == "run"
    verify(receipt)
