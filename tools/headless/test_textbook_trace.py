"""The checker must fail on each way a claim can be untraceable."""
import json, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import textbook_trace as tt

REPO = Path(__file__).resolve().parents[2]
TB = REPO / "docs/ultragoals/QWEN_TEXTBOOK_V1.md"


def _check(body):
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    return tt.check(path)


def test_value_disagreement_is_caught():
    _, problems, _, _ = _check(
        "Claim 9.99 EBPW [receipts/headless/WHOLE_MODEL_NATIVE.json#compile.complete_ebpw].\n")
    assert any(p["kind"] == "VALUE_DISAGREES" for p in problems), problems


def test_dead_pointer_is_caught():
    _, problems, _, _ = _check("See [receipts/headless/NO_SUCH.json#a.b].\n")
    assert any(p["kind"] == "DEAD_POINTER" for p in problems)


def test_unresolvable_path_is_caught():
    _, problems, _, _ = _check(
        "Claim 1.0 bpw [receipts/headless/WHOLE_MODEL_NATIVE.json#no.such.key].\n")
    assert any(p["kind"] == "UNRESOLVABLE" for p in problems)


def test_uncited_number_is_caught():
    _, _, untraceable, _ = _check("Throughput was 412.5 GB/s.\n")
    assert untraceable


def test_the_real_textbook_passes():
    claims, problems, untraceable, _ = tt.check(TB)
    assert problems == [] and untraceable == [], (problems, untraceable)
    assert len(claims) >= 20


def test_cli_exit_codes():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("Bad 9.99 [receipts/headless/WHOLE_MODEL_NATIVE.json#compile.complete_ebpw].\n")
        bad = f.name
    tool = str(Path(__file__).parent / "textbook_trace.py")
    assert subprocess.run([sys.executable, tool, bad], capture_output=True).returncode != 0
    assert subprocess.run([sys.executable, tool, str(TB)], capture_output=True).returncode == 0
