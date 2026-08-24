"""The causal law is applied to the claims already made, not only to future ones.

A law that governs only future work leaves the existing claims unexamined, and the
existing claims are the ones the campaign is currently reasoning from.

The audit found a systematic gap rather than a one-off: the campaign is diligent
about no-op comparisons (fused vs unfused, candidate vs incumbent) and carries
strong sentinels (dense_w_materialized, expanded_to_q4, weights_ptr_shared), but
rarely runs a deliberately WORSE candidate to prove the gate can still reject.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P = REPO / "receipts" / "headless" / "CAUSAL_BENCHMARK_LAW.json"


def _d():
    return json.loads(P.read_text())


def test_every_claim_has_a_sentinel_and_a_noop_control():
    """Those two are what stop a no-op from scoring. Their absence is the
    condition that invalidated the earlier self-opt harness."""
    bad = [r["receipt"] for r in _d()["audited"] if r.get("would_a_noop_pass")]
    assert bad == [], f"a no-op could pass these claims: {bad}"


def test_the_audit_is_retroactive():
    names = {r["receipt"] for r in _d()["audited"]}
    # Claims from the parent campaign must be included, not just new ones.
    assert "NOETIC_DISPATCH_FUSION" in names
    assert "AFFINE2_G64_LSFIT" in names
    assert len(names) >= 8


def test_the_bad_control_gap_is_recorded_not_hidden():
    missing_bad = [r["receipt"] for r in _d()["audited"] if "bad_control" in r.get("missing", [])]
    assert missing_bad, "if this ever empties, the gap is closed and this test should be updated"
    # And the survivors are the ones that really do carry a worse arm.
    survive = {r["receipt"] for r in _d()["audited"] if r["verdict"] == "SURVIVES_THE_LAW"}
    assert "NOETIC_PARENT_A" in survive, "it keeps runtime_div as a deliberate slow arm"


def test_limits_are_stated():
    lim = _d()["limits"]
    assert any("not for" in x or "not its correctness" in x.lower() or "presence" in x.lower()
               for x in lim), "must say it checks presence, not correctness"


def test_the_motivating_failures_are_named():
    why = " ".join(_d()["why_this_is_not_theoretical"]).lower()
    assert "identical code" in why and "llama-server" in why


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print(f"ok  {n}")
    print("5/5 passed")
