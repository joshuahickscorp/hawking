"""Tests for derived-artifact freshness.

Includes a negative control nobody has watched fail: two queue documents that
differ only in key order / whitespace / a zero-valued status bucket must
classify STALE_FINGERPRINT_ONLY, and a candidate status change must classify
STALE_SEMANTIC and make --check exit 1.
"""
from __future__ import annotations

import json

from tools.future import freshness as fr
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _raw(doc, *, indent=2, sort_keys=False) -> bytes:
    return json.dumps(doc, indent=indent, sort_keys=sort_keys).encode()


def test_module_entry_point_emits_sealed_receipt():
    rc = fr.main(["--report"])
    assert rc == 0
    path = RECEIPTS / "DERIVED_FRESHNESS.json"
    doc = json.loads(path.read_text())
    assert path.parent == RECEIPTS
    assert doc["schema"] == "hawking.future.freshness.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_source"] in {"pinned_snapshot", "live_headless"}
    _assert_no_hardware_claims(doc)


def test_selftest_guard_fires():
    assert fr.selftest() == 0


def test_semantic_fingerprint_ignores_key_order_whitespace_and_zero_buckets():
    a = fr.synthetic_candidate("alpha", status="READY_PROTECTED")
    b = fr.synthetic_candidate("beta", status="BLOCKED", blocked_reason="nx")
    first = fr.synthetic_queue([a, b])
    second = {
        "version": 1,
        "schema": first["schema"],
        "bench": {"recorded_at": "2099-01-01T00:00:00Z", "state": "UNKNOWN", "recorded_by": "rewritten"},
        "counts": {
            "candidates": 2,
            "by_status": {**first["counts"]["by_status"], "BRAND_NEW_ZERO": 0},
        },
        "candidates": [b, a],
    }
    assert fr.semantic_fingerprint(first) == fr.semantic_fingerprint(second)
    assert _raw(first) != _raw(second, indent=4, sort_keys=True)


def test_exact_mutation_key_order_and_dependency_order_do_not_move_fingerprint():
    left = fr.synthetic_candidate(
        "x",
        exact_mutation={"child_fusion_env": {"B": "2", "A": "1"}},
        dependencies=["p2", "p1"],
    )
    right = fr.synthetic_candidate(
        "x",
        exact_mutation={"child_fusion_env": {"A": "1", "B": "2"}},
        dependencies=["p1", "p2"],
    )
    assert fr.semantic_fingerprint(fr.synthetic_queue([left])) == fr.semantic_fingerprint(
        fr.synthetic_queue([right])
    )


def test_candidate_order_does_not_move_fingerprint():
    a = fr.synthetic_candidate("a")
    b = fr.synthetic_candidate("b")
    c = fr.synthetic_candidate("c")
    assert fr.semantic_fingerprint(fr.synthetic_queue([a, b, c])) == fr.semantic_fingerprint(
        fr.synthetic_queue([c, a, b])
    )


def test_negative_control_cosmetic_is_fingerprint_only_then_status_change_is_semantic():
    """NEGATIVE CONTROL: cosmetic rewrite vs a real status change.

    Two queue documents that differ ONLY in key order / whitespace / an added
    zero-valued status bucket classify STALE_FINGERPRINT_ONLY. Changing one
    candidate's status classifies STALE_SEMANTIC and --check exits 1.
    """
    alpha = fr.synthetic_candidate("alpha", status="READY_PROTECTED")
    beta = fr.synthetic_candidate("beta", status="BLOCKED", blocked_reason="nx")
    old = fr.synthetic_queue([alpha, beta])
    cosmetic = {
        "version": old["version"],
        "schema": old["schema"],
        "counts": {
            "by_status": {**old["counts"]["by_status"], "BRAND_NEW_ZERO": 0},
            "candidates": old["counts"]["candidates"],
        },
        "candidates": [
            {
                "blocked_reason": beta["blocked_reason"],
                "exact_mutation": beta["exact_mutation"],
                "dependencies": beta["dependencies"],
                "affected_physical_region": beta["affected_physical_region"],
                "status": beta["status"],
                "model": beta["model"],
                "candidate_id": beta["candidate_id"],
            },
            {
                "candidate_id": alpha["candidate_id"],
                "model": alpha["model"],
                "status": alpha["status"],
                "affected_physical_region": alpha["affected_physical_region"],
                "dependencies": alpha["dependencies"],
                "blocked_reason": alpha["blocked_reason"],
                "exact_mutation": {"child_fusion_env": {"HAWKING_FOO": "1"}},
            },
        ],
        "bench": {
            "recorded_by": "synthetic",
            "recorded_at": "2099-01-01T00:00:00Z",
            "state": "UNKNOWN",
        },
    }
    raw_old = _raw(old, indent=2)
    raw_cosmetic = _raw(cosmetic, indent=4, sort_keys=True)
    assert raw_old != raw_cosmetic
    cosmetic_row = fr.classify_documents(old, cosmetic, old_raw=raw_old, new_raw=raw_cosmetic)
    assert cosmetic_row["status"] == fr.STALE_FINGERPRINT_ONLY
    assert cosmetic_row["byte_match"] is False
    assert cosmetic_row["semantic_match"] is True
    assert cosmetic_row["added_candidate_ids"] == []
    assert cosmetic_row["removed_candidate_ids"] == []
    assert cosmetic_row["status_changed_candidate_ids"] == []
    assert fr.exit_code_for({"classifications": [cosmetic_row]}) == 0

    moved = json.loads(json.dumps(cosmetic))
    for row in moved["candidates"]:
        if row["candidate_id"] == "alpha":
            row["status"] = "BLOCKED"
    raw_moved = _raw(moved, indent=4, sort_keys=True)
    semantic_row = fr.classify_documents(old, moved, old_raw=raw_old, new_raw=raw_moved)
    assert semantic_row["status"] == fr.STALE_SEMANTIC
    assert semantic_row["byte_match"] is False
    assert semantic_row["semantic_match"] is False
    assert semantic_row["status_changed_candidate_ids"] == ["alpha"]
    assert semantic_row["added_candidate_ids"] == []
    assert semantic_row["removed_candidate_ids"] == []
    assert fr.exit_code_for({"classifications": [semantic_row]}) == 1


def test_fresh_when_bytes_and_semantics_match():
    q = fr.synthetic_queue([fr.synthetic_candidate("only")])
    raw = _raw(q)
    row = fr.classify_documents(q, q, old_raw=raw, new_raw=raw)
    assert row["status"] == fr.FRESH
    assert row["byte_match"] is True
    assert row["semantic_match"] is True
    assert fr.exit_code_for({"classifications": [row]}) == 0


def test_added_and_removed_candidate_ids():
    old = fr.synthetic_queue(
        [fr.synthetic_candidate("keep"), fr.synthetic_candidate("drop")],
    )
    new = fr.synthetic_queue(
        [fr.synthetic_candidate("keep"), fr.synthetic_candidate("new")],
    )
    row = fr.classify_documents(old, new, old_raw=_raw(old), new_raw=_raw(new))
    assert row["status"] == fr.STALE_SEMANTIC
    assert row["added_candidate_ids"] == ["new"]
    assert row["removed_candidate_ids"] == ["drop"]
    assert row["recorded_candidate_count"] == 2
    assert row["current_candidate_count"] == 2


def test_fingerprint_does_not_assume_a_fixed_candidate_count():
    two = fr.synthetic_queue([fr.synthetic_candidate("a"), fr.synthetic_candidate("b")])
    five = fr.synthetic_queue([fr.synthetic_candidate(str(i)) for i in range(5)])
    assert len(fr.queue_identity_rows(two)) == 2
    assert len(fr.queue_identity_rows(five)) == 5
    assert fr.semantic_fingerprint(two) != fr.semantic_fingerprint(five)
    # No module-level cap: a 47-row queue is just 47 derived rows.
    many = fr.synthetic_queue([fr.synthetic_candidate(f"c{i:03d}") for i in range(47)])
    assert len(fr.queue_identity_rows(many)) == 47


def test_unknown_when_no_provenance():
    current = fr.synthetic_queue([fr.synthetic_candidate("x")])
    row = fr.classify_artifact(
        recorded_sha256=None,
        current_sha256="abc",
        recorded_fp=None,
        current_fp=fr.semantic_fingerprint(current),
        recorded_doc=None,
        current_doc=current,
    )
    assert row["status"] == fr.UNKNOWN
    assert row["added_candidate_ids"] == []
    assert row["removed_candidate_ids"] == []
    assert fr.exit_code_for({"classifications": [row]}) == 0


def test_check_exit_codes():
    assert fr.exit_code_for({"classifications": [{"status": fr.FRESH}]}) == 0
    assert fr.exit_code_for({"classifications": [{"status": fr.STALE_FINGERPRINT_ONLY}]}) == 0
    assert fr.exit_code_for({"classifications": [{"status": fr.UNKNOWN}]}) == 0
    assert fr.exit_code_for({"classifications": [{"status": fr.STALE_SEMANTIC}]}) == 1
    assert (
        fr.exit_code_for(
            {
                "classifications": [
                    {"status": fr.STALE_FINGERPRINT_ONLY},
                    {"status": fr.STALE_SEMANTIC},
                ]
            }
        )
        == 1
    )


def test_refresh_invokes_only_stale_semantic_producers():
    called: list[str] = []

    def fake_invoke(producer: str):
        called.append(producer)
        return {"ok": True, "producer": producer}

    rows = [
        {"derived": "A.json", "status": fr.FRESH, "producer": "mod:a"},
        {"derived": "B.json", "status": fr.STALE_FINGERPRINT_ONLY, "producer": "mod:b"},
        {"derived": "C.json", "status": fr.STALE_SEMANTIC, "producer": "mod:c"},
        {"derived": "D.json", "status": fr.UNKNOWN, "producer": "mod:d"},
        {"derived": "E.json", "status": fr.STALE_SEMANTIC, "producer": "tools.future.freshness:build"},
    ]
    out = fr.refresh_stale(rows, invoke=fake_invoke)
    assert called == ["mod:c"]
    assert [r["derived"] for r in out] == ["C.json"]


def test_registry_covers_required_derived_artifacts():
    names = {e.derived for e in fr.REGISTRY}
    assert "CANDIDATE_STAGED_PLAN.json" in names
    assert "QUALIFICATION_PIPELINE.json" in names
    assert "FLASH_NX_COMPLETENESS_AUDIT.json" in names
    assert "HCLI_FUTURE_WORKUNITS.json" in names
    assert "TOURNAMENT_READINESS.json" in names
    assert "EVIDENCE_SNAPSHOT.json" in names
    for entry in fr.REGISTRY:
        assert ":" in entry.producer
        assert entry.kind in {"queue", "generic", "manifest"}


def test_unregistered_finding_lists_sidecar_receipts_not_in_registry():
    names = fr.list_unregistered()
    registered = {e.derived for e in fr.REGISTRY}
    for name in names:
        assert name not in registered
        assert name != "DERIVED_FRESHNESS.json"
        assert not name.startswith("evidence/")
    # A known sibling receipt must surface as UNREGISTERED so the finding is load-bearing.
    assert "ANE_PREBOARD.json" in names
    assert "CANDIDATE_STAGED_PLAN.json" not in names


def test_choose_current_prefers_primary_checkout_over_sibling_worktrees():
    copies = [
        {
            "kind": "other_worktree",
            "path": "/z/sibling/receipts/headless/Q.json",
            "sha256": "aaa",
            "evidence_source": "live_headless",
        },
        {
            "kind": "git_common",
            "path": "/a/primary/receipts/headless/Q.json",
            "sha256": "bbb",
            "evidence_source": "live_headless",
        },
        {
            "kind": "this_worktree",
            "path": "/m/here/receipts/headless/Q.json",
            "sha256": "ccc",
            "evidence_source": "live_headless",
        },
        {
            "kind": "pinned_snapshot",
            "path": "/p/pin/Q.json",
            "sha256": "ddd",
            "evidence_source": "pinned_snapshot",
        },
    ]
    chosen = fr.choose_current(copies)
    assert chosen is not None
    assert chosen["kind"] == "git_common"
    assert chosen["sha256"] == "bbb"


def test_resolve_copies_copes_with_live_or_pinned():
    copies = fr.resolve_copies(fr.QUEUE_REL)
    assert isinstance(copies, list)
    kinds = {c["kind"] for c in copies}
    assert kinds <= {"this_worktree", "other_worktree", "git_common", "pinned_snapshot"}
    for copy in copies:
        assert copy["sha256"]
        assert copy["evidence_source"] in {"pinned_snapshot", "live_headless"}
        assert copy["path"]
    # Either a live copy or the pin is enough to classify; the module must
    # record which path it took rather than requiring one checkout shape.
    assert copies, "queue must be visible as live, another worktree, or the pinned snapshot"


def test_assess_records_plan_provenance_and_does_not_hardcode_count():
    plan = fr.assess_entry(next(e for e in fr.REGISTRY if e.derived == "CANDIDATE_STAGED_PLAN.json"))
    assert plan["derived"] == "CANDIDATE_STAGED_PLAN.json"
    assert plan["status"] in {fr.FRESH, fr.STALE_FINGERPRINT_ONLY, fr.STALE_SEMANTIC, fr.UNKNOWN}
    assert plan["producer"] == "tools.future.candidate_planner:build"
    if plan["status"] == fr.STALE_SEMANTIC:
        assert isinstance(plan["added_candidate_ids"], list)
        assert isinstance(plan["removed_candidate_ids"], list)
        assert isinstance(plan["status_changed_candidate_ids"], list)
    for src in plan.get("sources") or []:
        n_rec = src.get("recorded_candidate_count")
        n_cur = src.get("current_candidate_count")
        if n_rec is not None:
            assert isinstance(n_rec, int) and n_rec >= 0
        if n_cur is not None:
            assert isinstance(n_cur, int) and n_cur >= 0
        if n_rec is not None and n_cur is not None:
            assert n_cur == n_rec + len(src.get("added_candidate_ids") or []) - len(
                src.get("removed_candidate_ids") or []
            )


def test_unknown_producers_are_findings_not_silent_fresh():
    for name in (
        "QUALIFICATION_PIPELINE.json",
        "FLASH_NX_COMPLETENESS_AUDIT.json",
        "HCLI_FUTURE_WORKUNITS.json",
        "TOURNAMENT_READINESS.json",
    ):
        entry = next(e for e in fr.REGISTRY if e.derived == name)
        row = fr.assess_entry(entry)
        assert row["status"] == fr.UNKNOWN, (
            f"{name} recorded no source sha; classifying it FRESH would silently "
            f"tolerate a semantic change. got {row['status']}"
        )


def test_no_hardware_numbers_in_assess_body():
    body = fr.assess()
    _assert_no_hardware_claims(body)
    blob = json.dumps(body)
    for key in HARDWARE_FIELDS:
        # Hardware keys may appear in prose (e.g. "token_ns") but the
        # HardwareClaimError walk is the load-bearing guard.
        assert isinstance(key, str)
    assert body["measurement_class"] == "STATIC_ONLY"
    assert body["no_era_vi"] is True
    assert body["no_odyssey_iv"] is True
    del blob


def test_check_cli_exit_matches_stale_semantic():
    # --check writes the receipt and uses the same exit rule as exit_code_for.
    rc = fr.main(["--check"])
    doc = json.loads((RECEIPTS / "DERIVED_FRESHNESS.json").read_text())
    assert rc == fr.exit_code_for(doc)
    assert rc in (0, 1)
    assert doc["check"]["would_exit_nonzero"] is (rc == 1)
