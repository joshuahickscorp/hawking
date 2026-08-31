"""Teacher-corpus contract tests.

The load-bearing test is the negative control: a corpus that only meets its
row threshold by duplicating rows must be REFUSED, and a same-size diverse
corpus must pass. A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json

import pytest

from tools.future import teacher_corpus as tc
from tools.future._common import RECEIPTS, load_json


def test_build_emits_sealed_receipt():
    out = tc.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "TEACHER_CORPUS_CONTRACT.json"
    assert doc["schema"] == "hawking.future.teacher_corpus.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    # Re-seal matches.
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    import hashlib

    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["selftest"]["diverse_accepted"] is True
    assert doc["selftest"]["duplicated_refused"] is True
    assert "THRESHOLD_MET_ONLY_BY_DUPLICATION" in doc["selftest"]["duplicated_codes"]
    assert doc["capture_workunits"]
    assert all(u["executed"] is False for u in doc["capture_workunits"])
    # load_json is the sidecar reader; the receipt must round-trip.
    assert load_json(out)["schema"] == doc["schema"]


def test_manifest_binds_specimen_layer_surface_provenance():
    rows = tc.make_diverse_corpus(16)
    man = tc.build_manifest(rows)
    assert man["schema"] == "hawking.future.teacher_corpus.manifest.v1"
    assert man["n_rows"] == 16
    assert man["specimens"]
    spec = man["specimens"][0]
    assert spec["model"]
    assert spec["pinned_revision"]
    assert spec["seal_sha256"]
    assert man["captures"]
    for cap in man["captures"]:
        assert cap["layer"] >= 0
        assert cap["surface"] in tc.SURFACES
        assert cap["seal_sha256"]
        assert cap["pinned_revision"]
        assert cap["route_union"]
        assert cap["provenance_chain"]
        for p in cap["provenance_chain"]:
            assert p["kind"]
            assert p["authority"] in tc.AUTHORITIES
            assert p["source_path"]
    for row in man["rows"]:
        assert row["content_sha256"]
        assert row["envelope_sha256"]
        assert "route_union_membership" in row
        assert set(row["route_union_membership"]) <= set(row["route_ids"])


def test_row_metadata_route_union_and_content_hash():
    rows = tc.make_diverse_corpus(8)
    annotated = tc.annotate_corpus(rows)
    unions = tc.route_unions(annotated)
    assert unions
    hashes = [r["content_sha256"] for r in annotated]
    assert len(hashes) == len(set(hashes))
    # Copying a row with a new id must collide on content and differ on envelope.
    copy = dict(annotated[0])
    copy["row_id"] = "copied"
    copy = tc.annotate_row(copy)
    assert copy["content_sha256"] == annotated[0]["content_sha256"]
    assert copy["envelope_sha256"] != annotated[0]["envelope_sha256"]


def test_diversity_measures_have_definitions_and_thresholds():
    rows = tc.make_diverse_corpus(32)
    div = tc.compute_diversity(rows)
    for name in (
        "row_diversity",
        "prompt_diversity",
        "token_position_diversity",
        "route_diversity",
        "capability_domain_diversity",
    ):
        block = div[name]
        assert "measure" in block
        assert "definition" in block and block["definition"]
        assert "formula" in block
        assert "threshold" in block
        assert "inadequate" in block
        assert block["inadequate"] is False
        assert name in tc.DIVERSITY_MEASURES
        assert tc.DIVERSITY_MEASURES[name]["inadequate_below"] is not None
    assert div["row_diversity"]["n_unique"] == 32
    assert div["prompt_diversity"]["n_unique"] >= tc.MIN_PROMPTS_FOR_FIT
    assert div["capability_domain_diversity"]["n_unique"] >= tc.MIN_DOMAINS_FOR_FIT


def test_dedup_by_content_hash():
    duped = tc.make_duplicated_corpus(20, unique=5)
    keep, groups = tc.exact_dedup(duped)
    assert len(keep) == 5
    collided = [ids for ids in groups.values() if len(ids) > 1]
    assert collided
    assert sum(len(ids) for ids in groups.values()) == 20


def test_near_duplicate_detection():
    # Long shared payload with a one-character tail change is a near-dup at 0.80.
    base = "the quick brown fox jumps over the lazy dog " * 6
    spec = tc.FIXTURE_SPECIMEN
    prov = tc._fixture_provenance("captured")
    a = tc.make_row(
        row_id="a",
        specimen=spec,
        layer=0,
        surface="hidden_pre_mlp",
        prompt_id="p0",
        prompt_text="alpha",
        token_position=0,
        route_ids=[1],
        capability_domain="prose",
        payload=base,
        provenance=prov,
    )
    b = tc.make_row(
        row_id="b",
        specimen=spec,
        layer=0,
        surface="hidden_pre_mlp",
        prompt_id="p1",
        prompt_text="beta",
        token_position=1,
        route_ids=[2],
        capability_domain="code",
        payload=base + "!",
        provenance=prov,
    )
    c = tc.make_row(
        row_id="c",
        specimen=spec,
        layer=1,
        surface="hidden_pre_mlp",
        prompt_id="p2",
        prompt_text="gamma",
        token_position=2,
        route_ids=[3],
        capability_domain="math",
        payload="completely different activation payload xyz",
        provenance=prov,
    )
    hits = tc.find_near_duplicates([a, b, c], field="payload")
    assert 0 in hits and 1 in hits
    assert all(h["jaccard"] >= tc.JACCARD_NEAR_DUP for h in hits[0])
    assert 2 not in hits


def test_negative_control_duplication_refused_and_diverse_passes():
    """Both halves are required. The refusal must actually fire."""
    n = 32
    diverse = tc.make_diverse_corpus(n)
    duped = tc.make_duplicated_corpus(n, unique=4)

    ok = tc.validate_corpus(diverse, min_rows=n, raise_on_refuse=True)
    assert ok["accepted"] is True
    assert ok["n_rows"] == n
    assert ok["n_unique_content"] == n
    assert ok["refusals"] == []

    with pytest.raises(tc.CorpusRefused) as ei:
        tc.validate_corpus(duped, min_rows=n, raise_on_refuse=True)
    err = ei.value
    assert "REFUSED" in str(err)
    assert "THRESHOLD_MET_ONLY_BY_DUPLICATION" in err.codes
    assert err.result["n_rows"] == n
    assert err.result["n_unique_content"] == 4
    assert err.result["n_unique_content"] < n
    # Same size, opposite verdict.
    assert ok["n_rows"] == err.result["n_rows"]
    assert ok["accepted"] is True
    assert err.result["accepted"] is False


def test_uniform_route_distribution_refused():
    rows = tc.make_uniform_route_corpus(32, n_routes=8)
    with pytest.raises(tc.CorpusRefused) as ei:
        tc.validate_corpus(rows, min_rows=32, raise_on_refuse=True)
    assert "ROUTE_DISTRIBUTION_SUSPICIOUSLY_UNIFORM" in ei.value.codes


def test_position_degeneracy_refused():
    rows = tc.make_position_degenerate_corpus(32)
    with pytest.raises(tc.CorpusRefused) as ei:
        tc.validate_corpus(rows, min_rows=32, raise_on_refuse=True)
    assert "POSITION_DEGENERACY" in ei.value.codes


def test_synthesised_padding_refused_even_when_payloads_are_unique():
    """Unique synthetic rows that close min_rows still fail if labelled synthesised
    or if they cycle routes. Unique-but-fake is still fake."""
    honest = tc.make_diverse_corpus(6)
    pad = tc.make_uniform_route_corpus(26)
    rows = honest + pad
    with pytest.raises(tc.CorpusRefused) as ei:
        tc.validate_corpus(rows, min_rows=32, raise_on_refuse=True)
    assert "SYNTHESISED_OR_DUPLICATED_TO_THRESHOLD" in ei.value.codes or (
        "ROUTE_DISTRIBUTION_SUSPICIOUSLY_UNIFORM" in ei.value.codes
    )


def test_missing_specimen_binding_refused():
    rows = tc.make_diverse_corpus(16)
    rows[3]["specimen"] = {"model": "x", "pinned_revision": "", "seal_sha256": ""}
    rows[3] = tc.annotate_row(rows[3])
    with pytest.raises(tc.CorpusRefused) as ei:
        tc.validate_corpus(rows, min_rows=8, raise_on_refuse=True)
    assert "MISSING_SPECIMEN_OR_PROVENANCE_BINDING" in ei.value.codes


def test_honest_small_corpus_is_inadequate_not_fabricated():
    rows = tc.make_diverse_corpus(6)
    result = tc.validate_corpus(rows, min_rows=32, raise_on_refuse=True)
    assert result["accepted"] is False
    assert result["refusals"] == []
    assert "UNIQUE_ROWS_BELOW_MIN" in result["inadequacy"]


def test_workunits_emitted_not_executed():
    units = tc.emit_capture_workunits()
    assert units
    ids = [u["id"] for u in units]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    for u in units:
        assert u["role"] == "teacher_capture"
        assert u["status"] == "pending"
        assert u["executed"] is False
        assert u["resource_class"] == "GPU_EXCLUSIVE"
        p = u["payload"]
        assert p["specimen"]["model"]
        assert p["specimen"]["pinned_revision"]
        assert p["specimen"]["seal_sha256"]
        assert p["layer_range"] == [0, 4]
        assert p["surface"] in tc.SURFACES
        assert p["target_row_count"] == tc.BOUNDED_TARGET_ROWS
        assert p["diversity_target"]["validator"].endswith("validate_corpus")
        assert "GPU" in u["execution_forbidden_reason"] or "lease" in u["execution_forbidden_reason"]


def test_content_hash_excludes_row_id_and_provenance():
    rows = tc.make_diverse_corpus(2)
    a = dict(rows[0])
    b = dict(rows[0])
    b["row_id"] = "other-id"
    b["provenance"] = tc._fixture_provenance("duplicated")
    assert tc.content_sha256_of(a) == tc.content_sha256_of(b)


def test_selftest_function_proves_both_halves():
    result = tc.selftest()
    assert result["diverse_accepted"] is True
    assert result["duplicated_refused"] is True
    assert "THRESHOLD_MET_ONLY_BY_DUPLICATION" in result["duplicated_codes"]
    assert result["duplicated_unique"] < result["duplicated_n"]
    assert result["diverse_unique"] == result["diverse_n"]
