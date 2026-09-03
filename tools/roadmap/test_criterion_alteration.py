"""criterion_altered was a constant, so half the BUILT law proved nothing.

The law is "verdict ACCEPTED and criterion_altered false and a command". The
middle clause was written as the literal False at fourteen sites across all six
acceptance lanes and computed at none of them. A receipt cannot be trusted to
report that it did not move its own goalposts, so the auditor recomputes it from
the source the receipt cites.
"""
from __future__ import annotations

import json
from collections import Counter

from tools.roadmap import lineage
from tools.roadmap.auditor import REPO, criterion_matches_its_source as check

ACCEPTANCE = REPO / "receipts" / "acceptance"


def _real_span_and_text() -> tuple[int, int, str]:
    lines = lineage.roadmap_lines()
    start, end = 7332, 7358
    return start, end, "\n".join(lines[start - 1:end])


def test_a_matching_span_is_recognised():
    start, end, text = _real_span_and_text()
    verdict, why = check({"criterion_quoted": f"## header\n{text}",
                          "criterion_source": {"start_line": start, "end_line": end}})
    assert verdict == "MATCHES", why


def test_a_moved_goalpost_is_caught():
    """The defect the flag was supposed to prevent and could not."""
    start, end, _ = _real_span_and_text()
    verdict, why = check({
        "criterion_quoted": "a criterion this gate can definitely satisfy",
        "criterion_source": {"start_line": start, "end_line": end},
    })
    assert verdict == "ALTERED", why
    assert str(start) in why


def test_a_self_declared_alteration_is_still_honoured():
    assert check({"criterion_altered": True, "criterion_quoted": "x"})[0] == "ALTERED"
    assert check({"criterion_weakened": True, "criterion_quoted": "x"})[0] == "ALTERED"


def test_an_empty_span_is_alteration_not_a_match():
    """A span past the end of the roadmap quotes nothing, and nothing must never
    read as agreement."""
    verdict, why = check({"criterion_quoted": "anything",
                          "criterion_source": {"start_line": 99000, "end_line": 99010}})
    assert verdict == "ALTERED", why


def test_an_unciteable_receipt_is_unverifiable_not_trusted():
    assert check({"criterion_quoted": "text with no span"})[0] == "UNVERIFIABLE"
    assert check({})[0] == "UNVERIFIABLE"


def test_the_supplement_shape_is_checked_against_the_supplement():
    sup = json.loads((REPO / "civilization" / "GATE_CRITERIA_SUPPLEMENT.json").read_text())
    want = sup["gates"]["FPGA_HWIR"]["criterion"]
    doc = {"criterion_quoted": want,
           "criterion_source": {"pointer": "gates.FPGA_HWIR.criterion"}}
    assert check(doc)[0] == "MATCHES"
    doc["criterion_quoted"] = "something easier"
    assert check(doc)[0] == "ALTERED"


def test_no_accepted_receipt_in_the_corpus_has_an_altered_criterion():
    """Documents the current state. A failure here is a real finding, not noise."""
    counts: Counter[str] = Counter()
    altered = []
    for path in sorted(ACCEPTANCE.glob("*.json")):
        if "." in path.stem:
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or str(doc.get("verdict", "")).upper() != "ACCEPTED":
            continue
        verdict, why = check(doc)
        counts[verdict] += 1
        if verdict == "ALTERED":
            altered.append(f"{path.stem}: {why}")
    assert not altered, "ACCEPTED receipts whose criterion no longer matches: " + "; ".join(altered)
    assert counts["MATCHES"] >= 18, counts


def test_the_number_of_unverifiable_acceptances_only_goes_down():
    """Eight gates are accepted against a criterion nobody can re-check.

    That is not a pass and not a failure -- it is the size of the hole, pinned so
    it cannot quietly grow. Lower it by giving those receipts a citeable span.
    """
    graph = json.loads((REPO / "civilization" / "CAPABILITY_GRAPH.json").read_text())
    unverifiable = [
        gid for gid, g in graph["gates"].items()
        for e in (g.get("accepted") or {}).get("evidence") or []
        if e.get("criterion_source_check") == "UNVERIFIABLE"
    ]
    assert len(unverifiable) <= 8, sorted(unverifiable)
