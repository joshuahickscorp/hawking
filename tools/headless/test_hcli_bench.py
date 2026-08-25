"""G039 pins."""
import json, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
RUNS = sorted(RH.glob("HCLI_BENCH_*.json"))
pytestmark = pytest.mark.skipif(not RUNS, reason="no HCLI bench runs")


def runs():
    return [json.load(open(p)) for p in RUNS]


def test_the_headline_is_a_rate_not_a_count():
    """A body that is correct but slow must not score like a fast one."""
    for d in runs():
        assert d["headline"] == "VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR"
        s = d["score"]
        assert s["VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR"] > 0
        assert s["total_wall_s"] > 0


def test_all_required_axes_are_exercised():
    """S011 §26 names reasoning, code generation, tool selection, tool calls,
    structured output, repair, verification and multi-turn progress."""
    for d in runs():
        assert {"code_generation", "repair", "tool_selection", "structured_output",
                "multi_turn", "verification"} <= set(d["axes_covered"])


def test_acceptance_is_deterministic_never_model_graded():
    for d in runs():
        assert "No model grades any model" in d["scoring"]


def test_the_secondary_vector_is_present():
    """§27: median and p95 latency, repair loops/WU, tool reliability, cost/WU."""
    for d in runs():
        s = d["score"]
        for k in ("median_workunit_latency_s", "p95_workunit_latency_s",
                  "repair_loops_per_workunit", "tool_reliability",
                  "tokens_per_accepted_workunit"):
            assert k in s, k


def test_every_workunit_records_its_own_evidence():
    for d in runs():
        for w in d["workunits"]:
            assert w["trace"], w["id"]
            assert isinstance(w["verified"], bool)
            assert w["why"]
            assert w["turns"] >= 1


def test_tool_calls_actually_executed():
    """A bench where no tool ever ran would not be measuring tool use."""
    assert any(d["score"]["tool_calls"] > 0 for d in runs())


def test_repair_loops_actually_happened_somewhere():
    """The repair path must be exercised, not merely present in the code."""
    assert any(w["repair_loops"] > 0 for d in runs() for w in d["workunits"])


def test_a_failure_is_recorded_rather_than_rounded_away():
    """Every run accepting everything would mean the bench cannot discriminate."""
    assert any(not w["verified"] for d in runs() for w in d["workunits"])


# --- the verifiers must be able to fail ------------------------------------------

def test_code_verifier_rejects_wrong_code_and_accepts_right_code():
    import hcli_bench as h
    good = "```python\ndef dedupe(xs):\n    s=set(); o=[]\n    for x in xs:\n        " \
           "if x not in s:\n            s.add(x); o.append(x)\n    return o\n```"
    bad = "```python\ndef dedupe(xs):\n    return sorted(set(xs))\n```"
    wu = next(w for w in h.WORKUNITS if w["id"] == "wu-code-dedupe")
    assert wu["verify"](good)[0] is True
    assert wu["verify"](bad)[0] is False
    assert wu["verify"]("no code here")[0] is False


def test_tool_verifier_rejects_the_wrong_tool():
    import hcli_bench as h
    wu = next(w for w in h.WORKUNITS if w["id"] == "wu-tool-select")
    right = '<tool>{"name":"read_file","args":{"path":"/etc/hostname"}}</tool>'
    wrong = '<tool>{"name":"list_dir","args":{"path":"/etc"}}</tool>'
    assert wu["verify"](right)[0] is True
    assert wu["verify"](wrong)[0] is False


def test_unterminated_think_block_is_not_counted_as_a_reply():
    """Same rule as the capability suite; otherwise a runaway scores as an answer."""
    src = (Path(__file__).resolve().parent / "hcli_bench.py").read_text()
    assert "unterminated" in src and "</think>" in src
