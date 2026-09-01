"""Continuity, not similarity: compaction must not lose what the resident knows.

The grade is whether a reader of the compacted kernel could continue
correctly. Textual similarity is not the grade. Each of the five
corruptions must be caught against the REAL kernel; a compactor that
passes its own corrupted input is worthless.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.future import hcli_compactor as hc
from tools.future._common import RECEIPTS


@pytest.fixture(scope="module")
def kernel():
    return hc.load_kernel()


@pytest.fixture(scope="module")
def compacted(kernel):
    return hc.compact(kernel)


def test_missing_kernel_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "kernel_path", lambda: tmp_path / "NO_SUCH.json")
    with pytest.raises(hc.CompactorRefused, match="cannot be read"):
        hc.load_kernel()


def test_missing_required_field_refuses(kernel):
    k = copy.deepcopy(kernel)
    del k["scars"]
    with pytest.raises(hc.CompactorRefused, match="missing"):
        hc.compact(k)


def test_hypothesis_without_verdict_refuses(kernel):
    k = copy.deepcopy(kernel)
    k["hypotheses"][0].pop("verdict")
    with pytest.raises(hc.CompactorRefused, match="verdict"):
        hc.compact(k)


def test_real_kernel_compaction_preserves_hypothesis_verdicts(kernel, compacted):
    assert hc.hypothesis_verdicts(kernel) == hc.hypothesis_verdicts(compacted)
    assert hc.hypothesis_verdicts(kernel), "the live kernel has recorded verdicts"
    for hid, verdict in hc.hypothesis_verdicts(kernel).items():
        assert compacted["hypotheses"]
        got = {h["id"]: h["verdict"] for h in compacted["hypotheses"]}
        assert got[hid] == verdict


def test_real_kernel_compaction_preserves_scars(kernel, compacted):
    assert [_scar(s) for s in kernel["scars"]] == [_scar(s) for s in compacted["scars"]]
    assert len(compacted["scars"]) >= 1


def _scar(s):
    return s if isinstance(s, str) else json.dumps(s, sort_keys=True)


def test_honest_compaction_passes_continuity(kernel, compacted):
    report = hc.evaluate_continuity(kernel, compacted)
    assert report["ok"] is True, report["losses"]
    assert report["could_continue"] is True
    assert report["grade"] == "CONTINUITY"
    assert report["losses"] == []
    assert "textual similarity is not the grade" in report["not_the_grade"]


def test_removing_a_live_hypothesis_is_caught(kernel, compacted):
    assert hc.live_hypotheses(kernel), "the live kernel has a live hypothesis"
    bad = hc.corrupt_remove_live_hypothesis(compacted)
    report = hc.evaluate_continuity(kernel, bad)
    assert report["ok"] is False
    assert hc.LIVE_HYPOTHESIS in report["loss_kinds"], report


def test_removing_a_scar_is_caught(kernel, compacted):
    bad = hc.corrupt_remove_scar(compacted)
    report = hc.evaluate_continuity(kernel, bad)
    assert report["ok"] is False
    assert hc.SCAR in report["loss_kinds"], report


def test_altering_the_target_is_caught(kernel, compacted):
    bad = hc.corrupt_alter_target(compacted)
    report = hc.evaluate_continuity(kernel, bad)
    assert report["ok"] is False
    assert hc.CURRENT_TARGET in report["loss_kinds"], report


def test_removing_the_latest_refutation_is_caught(kernel, compacted):
    ref = hc.latest_refutation(kernel)
    assert ref is not None and ref.get("id"), "the live kernel has a refutation"
    bad = hc.corrupt_remove_latest_refutation(compacted)
    report = hc.evaluate_continuity(kernel, bad)
    assert report["ok"] is False
    assert hc.LATEST_REFUTATION in report["loss_kinds"], report
    assert ref["id"] not in hc.hypothesis_verdicts(bad)


def test_removing_a_wake_condition_is_caught(kernel, compacted):
    assert hc.wake_conditions(kernel), "the live kernel has a wake condition"
    bad = hc.corrupt_remove_wake_condition(compacted)
    report = hc.evaluate_continuity(kernel, bad)
    assert report["ok"] is False
    assert hc.WAKE_CONDITION in report["loss_kinds"], report


def test_textual_similarity_is_not_the_grade(kernel, compacted):
    """Near-identical-with-a-hole fails; rewritten-prose-with-the-surface passes."""
    similar = copy.deepcopy(kernel)
    similar["scars"] = list(similar["scars"])[:-1]
    similar_report = hc.evaluate_continuity(kernel, similar)
    assert similar_report["ok"] is False
    assert hc.SCAR in similar_report["loss_kinds"]

    rewritten = copy.deepcopy(compacted)
    for it in rewritten.get("iterations") or []:
        if "belief_update" in it:
            it["belief_update"] = "rewritten deliberation the evaluator must ignore"
    rewritten_report = hc.evaluate_continuity(kernel, rewritten)
    assert rewritten_report["ok"] is True, rewritten_report["losses"]


def test_evaluator_does_not_grade_a_compactor_against_its_own_input(kernel):
    """A compactor that 'passes' the kernel it just stripped a scar from is worthless."""
    stripped = hc.corrupt_remove_scar(kernel)
    compacted_stripped = hc.compact(stripped)
    assert hc.evaluate_continuity(stripped, compacted_stripped)["ok"] is True
    vs_real = hc.evaluate_continuity(kernel, compacted_stripped)
    assert vs_real["ok"] is False
    assert hc.SCAR in vs_real["loss_kinds"]


def test_tried_params_measured_state_and_objective_are_preserved(kernel, compacted):
    assert compacted["objective"] == kernel["objective"]
    assert compacted["measured_state"] == kernel["measured_state"]
    assert set(hc.tried_params(compacted)) >= set(hc.tried_params(kernel))
    assert hc.unsupported_requests(compacted) == hc.unsupported_requests(kernel)
    assert hc.next_work(compacted)
    assert hc.active_work(compacted) or hc.measurements(kernel) == []


def test_degenerate_empty_turns_are_discarded(kernel):
    k = copy.deepcopy(kernel)
    k.setdefault("iterations", []).append({
        "n": 9999,
        "parsed": False,
        "degenerated": True,
        "belief_update": "this tail is noise and must not survive compaction " * 20,
        "live_hypotheses": None,
        "results": [],
        "results_summary": ["no work was accepted from that turn"],
        "reply_chars": 4000,
    })
    compacted, stats = hc.compact_with_stats(k)
    ns = [it.get("n") for it in compacted.get("iterations") or []]
    assert 9999 not in ns
    assert stats["discarded_by_category"]["degenerate_tails"]["n"] >= 1
    assert stats["discarded_by_category"]["degenerate_tails"]["bytes"] > 0
    assert hc.evaluate_continuity(k, compacted)["ok"] is True


def test_compacted_kernel_is_smaller(kernel, compacted):
    before = hc._nbytes(kernel)
    after = hc._nbytes(compacted)
    assert after < before
    _, stats = hc.compact_with_stats(kernel)
    assert stats["compression_ratio"] < 1.0
    assert stats["bytes_saved"] > 0


def test_receipt_reports_compression_ratio_and_discarded_categories(kernel):
    doc = hc.build()
    assert "compression_ratio" in doc
    assert 0 < doc["compression_ratio"] < 1.0
    cats = doc["discarded_by_category"]
    for name in ("repeated_prose", "obsolete_deliberation", "degenerate_tails"):
        assert name in cats, name
        assert "n" in cats[name] and "bytes" in cats[name]
    assert doc["continuity"]["ok"] is True
    assert doc["all_negative_controls_caught"] is True
    for name in hc.CORRUPTIONS:
        assert doc["negative_controls"][name]["caught"] is True, name
    assert doc["preserved"]["hypothesis_verdicts"] == hc.hypothesis_verdicts(kernel)
    assert doc["preserved"]["n_scars"] == len(kernel["scars"])
    assert doc["grade_is_not"] == "textual similarity"
    assert "choose the next hypothesis" in doc["what_this_module_does_not_do"]


def test_build_writes_a_receipt_that_parses():
    doc = hc.build()
    path = hc.write_receipt(hc.RECEIPT_NAME, doc, hc.RECORDED_BY)
    assert path == RECEIPTS / hc.RECEIPT_NAME
    loaded = json.loads(path.read_text())
    assert loaded["schema"] == hc.SCHEMA
    assert loaded["seal_sha256"]
    assert loaded["compression_ratio"] == doc["compression_ratio"]
    assert "discarded_by_category" in loaded
    json.loads(json.dumps(loaded))


def test_kernel_path_resolves_the_live_kernel():
    p = hc.kernel_path()
    assert p.is_file()
    assert p.name == "HCLI_MISSION_KERNEL.json"
    assert Path(p).as_posix().endswith(hc.KERNEL_REL)
