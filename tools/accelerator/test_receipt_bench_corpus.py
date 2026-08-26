"""S032 §3 across the whole corpus, not just receipts built through receipt.build.

The rule is only real if it binds the artifacts that already exist. 296 of the 297
receipts quoting a duration carried no machine state at all when this was written,
and an absent state reads as QUIESCED to every downstream reader -- which is the
claim none of them made.
"""
import json
from pathlib import Path

import receipt as R

RH = R.REPO / "receipts" / "headless"
# A raw data dump is not a receipt and never made a claim.
RECEIPT_MARKERS = {"schema", "receipt", "identities", "result", "date"}


def receipts_quoting_timing():
    for f in sorted(RH.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or not (RECEIPT_MARKERS & set(d)):
            continue
        if R._timing_keys(d, "root"):
            yield f, d


def test_ANTI_VACUITY_the_corpus_actually_contains_performance_receipts():
    """Without this, a glob that matched nothing would make the rule below pass."""
    found = list(receipts_quoting_timing())
    assert len(found) > 100, f"only {len(found)} timing receipts found"


def test_EVERY_receipt_quoting_timing_states_the_machine_it_was_measured_under():
    missing = [f.name for f, d in receipts_quoting_timing()
               if not isinstance(d.get("bench"), dict)]
    assert not missing, (
        f"{len(missing)} performance receipts carry no bench block, so they read "
        f"as quiet: {missing[:8]}\n"
        f"fix: python3 tools/accelerator/stamp_bench_state.py\n"
        f"THIS DRIFTS ON PURPOSE. A launchd job (tools/odyssey_driver.sh) "
        f"regenerates receipts every five minutes without a bench block, so a "
        f"one-time backfill cannot hold -- eighteen had already lost their stamp "
        f"forty minutes after the first pass.")


def test_NO_receipt_claims_QUIESCED_without_a_sample_that_earned_it():
    """UNKNOWN is cheap and honest. QUIESCED is a claim, and a backfill that
    stamped it everywhere would be worse than the gap it closed."""
    for f, d in receipts_quoting_timing():
        b = d["bench"]
        assert b.get("state") in R.BENCH_STATES, (f.name, b.get("state"))
        if b["state"] == "QUIESCED":
            q = b.get("quiescence")
            assert isinstance(q, dict) and q.get("quiet") is True, f.name
            assert not q.get("n_contenders"), f.name


def test_every_bench_block_names_a_machine_and_a_time():
    for f, d in receipts_quoting_timing():
        for k in ("recorded_at", "machine", "rule"):
            assert d["bench"].get(k), (f.name, k)


def test_the_backfilled_state_is_LABELLED_as_backfilled():
    """A reader must be able to tell a state that was recorded at measurement time
    from one reconstructed afterwards. They are not the same evidence."""
    back = [f.name for f, d in receipts_quoting_timing()
            if "backfill" in str(d["bench"].get("recorded_by", ""))]
    assert back, "the backfill left no trace of itself"
    for f, d in receipts_quoting_timing():
        if "backfill" in str(d["bench"].get("recorded_by", "")):
            assert d["bench"].get("provenance"), f.name


def test_the_stamper_is_a_TOOL_and_its_check_mode_agrees_with_this_file():
    """A backfill that lives in a scratchpad script is a backfill nobody can rerun,
    and this corpus needs rerunning: a live writer keeps dropping the field."""
    import stamp_bench_state as S
    assert S.main(["--check"]) == 0


def test_the_stamper_is_IDEMPOTENT():
    """Running it twice must not restamp, or every run would rewrite the corpus and
    a real drift would be invisible in the diff."""
    import stamp_bench_state as S
    before = {f.name: raw for f, raw, _ in S.performance_receipts()}
    assert S.main([]) == 0
    after = {f.name: f.read_text() for f, _, _ in S.performance_receipts()}
    changed = [n for n in before if before[n] != after.get(n)]
    assert not changed, changed[:5]
