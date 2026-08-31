import json

from tools.future import global_frontier as gf


def test_every_entry_carries_the_section_78_fields():
    for e in gf.FRONTIER:
        for field in gf.REQUIRED_FIELDS:
            assert field in e, f"{e.get('id')} missing {field}"
        assert e["classification"] in {
            "MISSING", "WEAK", "STALE", "BLOCKED", "HIGH_VALUE_INTEGRATION"
        }


def test_ids_unique():
    ids = [e["id"] for e in gf.FRONTIER]
    assert len(ids) == len(set(ids))


def test_missing_claims_are_probe_backed_not_asserted():
    # Every MISSING entry must prove absence by executing a probe, and the
    # probe must actually be an absence/presence test that ran.
    for e in gf.FRONTIER:
        r = gf.run_probe(e["probe"])
        assert "holds" in r
        if e["classification"] == "MISSING" and r["kind"] == "absent":
            # Either the gap is still open, or this campaign closed it. What must
            # never happen is a MISSING claim contradicted by something OUTSIDE the
            # sidecar, which would mean the claim was wrong when it was written.
            assert r["holds"] or r.get("resolved_by_sidecar"), (
                f"{e['id']} claims MISSING but found non-sidecar {r['hits'][:3]}"
            )


def test_absent_probe_actually_detects_something_that_exists():
    # Negative control: the absence probe must be capable of failing.
    r = gf._probe_absent("mutation_surface.py")
    assert r["hits"], "absence probe found nothing for a file that exists"
    assert r["holds"] is False


def test_build_emits_sealed_receipt():
    out = gf.build()
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.claude_global_frontier.v1"
    assert doc["seal_sha256"]
    assert doc["bench"]["gpu_authority"] is False
    assert doc["counts"]["total"] == len(gf.FRONTIER)
    assert all("probe_result" in e for e in doc["entries"])


def test_verify_is_green_now():
    assert gf.verify() == 0
