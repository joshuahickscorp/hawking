"""Acceptance: DSV4F writer output is readable by doctor6 collect_expert_activations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    ActivationWeightedRepackError,
    collect_expert_activations,
)
from lab.operators.dsv4f_activation_capture import collect_via_doctor6, read_f32le


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _emit_via_example(out: Path) -> Path:
    root = _repo_root()
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "hawking-core",
        "--example",
        "dsv4f_activation_capture",
        "--",
        "--output-dir",
        str(out),
        "--tiny",
        "--layers",
        "3",
        "--probes",
        "2",
        "--tokens-per-probe",
        "12",
        "--max-hidden-tokens-per-expert",
        "3",
        "--experts",
        "16",
        "--row-threshold",
        "2",
        "--capture-set",
        "required",
    ]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=root, env=env)
    return out


def test_doctor6_reads_writer_output(tmp_path: Path) -> None:
    run = _emit_via_example(tmp_path / "dsv4f-cap")
    cap = json.loads((run / "capture-result.json").read_text(encoding="utf-8"))
    stacked, prov = collect_expert_activations(run, cap)
    assert stacked, "doctor6 must yield at least one (layer, expert) array"
    hidden = int(cap["runtime_binding"]["hidden"])
    print(f"doctor6_key_count={len(stacked)}")
    print(f"doctor6_layers_with_hidden_hits={prov.get('layers_with_hidden_hits')}")
    print(f"doctor6_token_expert_pairs={prov.get('token_expert_pairs')}")
    shown = 0
    for (layer, expert), arr in sorted(stacked.items()):
        assert arr.dtype == np.float32
        assert arr.ndim == 2
        assert arr.shape[1] == hidden
        if shown < 4:
            print(f"doctor6_organ_shape L{layer}.E{expert}={arr.shape}")
            shown += 1
    via_helper, _ = collect_via_doctor6(run, cap)
    assert set(via_helper) == set(stacked)
    for key in stacked:
        np.testing.assert_array_equal(via_helper[key], stacked[key])


def test_short_row_raises_in_doctor6(tmp_path: Path) -> None:
    run = _emit_via_example(tmp_path / "dsv4f-short")
    cap = json.loads((run / "capture-result.json").read_text(encoding="utf-8"))
    # Truncate the first retained hidden so collect must raise.
    victim = None
    for probe in cap["probes"]:
        for step in probe["steps"]:
            for layer_row in step["layers"]:
                meta = layer_row.get("router_input_hidden_f32le")
                if meta:
                    victim = run / meta["relative_path"]
                    expected = int(meta["elements"])
                    break
            if victim is not None:
                break
        if victim is not None:
            break
    assert victim is not None and victim.is_file()
    raw = victim.read_bytes()
    victim.write_bytes(raw[:4])
    raised = None
    try:
        collect_expert_activations(run, cap)
    except ActivationWeightedRepackError as exc:
        raised = exc
    assert raised is not None, "doctor6 must raise on a short hidden row"
    assert "hidden size mismatch" in str(raised)
    raised = None
    try:
        read_f32le(victim, expected)
    except ActivationWeightedRepackError as exc:
        raised = exc
    assert raised is not None, "read_f32le must raise on a short hidden row"
    assert "hidden size mismatch" in str(raised)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_doctor6_reads_writer_output(p / "a")
        test_short_row_raises_in_doctor6(p / "b")
    print("PASS test_dsv4f_activation_capture_writer")
    sys.exit(0)
