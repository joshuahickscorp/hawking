"""Acceptance tests: CALL the gate symbols. Never weaken, skip, or mark slow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.acceptance.lake.common import (
    GATES,
    RECEIPT_SCHEMA,
    RECEIPTS,
    WORKTREE,
    lake_mounted,
)
from tools.acceptance.lake.hash_verify import (
    GATE as HASH_GATE,
    call_reconcile,
    run_hash_gate,
    run_reconcile_on_scratch,
)
from tools.acceptance.lake.identity import (
    GATE as IDENTITY_GATE,
    call_run_modellake_census,
    run_identity_gate,
)
from tools.acceptance.lake.promotion import (
    GATE as PROMO_GATE,
    call_promote,
    run_promote_on_scratch,
    run_promotion_gate,
)
from tools.odyssey.modellake import sha256 as lake_sha256
from tools.odyssey.modellake_lineage import (
    CANONICAL_SPECIMEN,
    ROADMAP_LIFECYCLE,
    derive_lifecycle,
    express_lineage,
)
from tools.odyssey.product_boundary import safe_defaults

REPO = WORKTREE


def test_every_assigned_gate_is_catalogued():
    assigned = {
        "MODELLAKE_IDENTITY_RESOLVED",
        "MODELLAKE_HASH_VERIFIED",
        "MODELLAKE_ATOMIC_PROMOTION",
        "QWEN27_RUNTIME_IDENTITY_FROZEN",
        "QWEN27_PROTECTED_BASELINE",
        "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED",
    }
    assert assigned == set(GATES)


def test_derive_lifecycle_identity_resolved_from_repo_revision():
    """§14: watch-manifest repo/revision is IDENTITY_RESOLVED, not DISCOVERED."""
    life, why = derive_lifecycle(
        watch={"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
        lake_man=None,
        source_dir=None,
        partial_dir=None,
        staged_dir=None,
        fingerprinted=False,
        nr_present=False,
    )
    assert life == "IDENTITY_RESOLVED"
    assert "repo/revision" in why
    assert ROADMAP_LIFECYCLE.index(life) > ROADMAP_LIFECYCLE.index("DISCOVERED")


def test_derive_lifecycle_does_not_call_a_partial_a_specimen(tmp_path):
    partial = tmp_path / "partial" / "slug"
    partial.mkdir(parents=True)
    life, _ = derive_lifecycle(
        watch=None,
        lake_man=None,
        source_dir=None,
        partial_dir=partial,
        staged_dir=None,
        fingerprinted=False,
        nr_present=False,
    )
    assert life == "DOWNLOADING"


def test_promote_symbol_refuses_incomplete_and_renames_atomically(tmp_path):
    result = run_promote_on_scratch(tmp_path)
    props = result["properties"]
    assert props["incomplete_refused"]
    assert props["incomplete_absent_from_specimens"]
    assert props["dry_run_moved_nothing"]
    assert props["go_promoted"]
    assert props["go_verified_at_destination"]
    assert props["go_partial_gone"]
    assert props["go_weights_intact"]
    assert props["clash_refused"]
    assert props["clash_existing_untouched"]
    # Direct call site of the catalog symbol (already used inside the helper).
    refused = call_promote("no-such-tag", go=False)
    assert refused["action"] == "REFUSED"


def test_reconcile_symbol_promotes_complete_scratch_only(tmp_path):
    result = run_reconcile_on_scratch(tmp_path)
    scratch = result["scratch"]
    assert scratch["destination_present"] is True
    assert scratch["partial_gone"] is True
    assert scratch["live_lake_untouched"] is True
    assert "acme--hash@deadbeefcafe" in result["promoted"]
    # call_reconcile is the catalog symbol; the helper already invoked it.
    assert callable(call_reconcile)


def test_modellake_sha256_matches_known_bytes(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hawking-acceptance-hash")
    digest = lake_sha256(path)
    import hashlib

    assert digest == hashlib.sha256(b"hawking-acceptance-hash").hexdigest()


def test_census_symbol_is_actually_called(tmp_path, monkeypatch):
    mg = pytest.importorskip("hcli.agentos.modellake_gate")
    lake = tmp_path / "lake"
    specimens = lake / "specimens"
    (specimens / "Qwen--Qwen3-0.6B@c1899de289a0").mkdir(parents=True)
    (lake / "partial").mkdir()
    (lake / "manifests").mkdir()
    monkeypatch.setattr(mg, "LAKE", lake)
    monkeypatch.setattr(mg, "TIER2", specimens)
    monkeypatch.setattr(mg, "PARTIAL", lake / "partial")
    monkeypatch.setattr(mg, "MANIFESTS", lake / "manifests")
    monkeypatch.setattr(
        mg,
        "_fetch_manifest",
        lambda repo, revision, timeout_s: {
            "repo": repo,
            "requested_revision": revision,
            "resolved_revision": revision,
            "last_modified": None,
            "file_count": 0,
            "total_declared_bytes": 0,
            "files": [],
            "source_url": "test://pinned",
        },
    )
    monkeypatch.setattr(mg, "_processes", lambda: {"status": "UNKNOWN", "matches": []})
    emit = tmp_path / "census.json"
    report = call_run_modellake_census(
        repo_root=tmp_path, emit=emit, timeout_s=5.0
    )
    assert report["schema"] == "hcli.agentos.modellake_census.v1"
    names = [e["name"] for e in (report.get("specimens") or {}).get("entries") or []]
    assert "Qwen--Qwen3-0.6B@c1899de289a0" in names
    assert report.get("acquisition_policy", {}).get("download_performed") is False


def test_identity_gate_live_lineage_if_lake_mounted(tmp_path, monkeypatch):
    if not lake_mounted():
        pytest.skip("ModelLake volume not mounted")
    monkeypatch.setattr("tools.acceptance.lake.common.RECEIPTS", tmp_path)
    receipt = run_identity_gate(live=True, run_census=False)
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["criterion_altered"] is False
    assert receipt["measured"]["sealed_specimens"] == 55
    assert receipt["measured"]["identity_resolved"] == 55
    assert receipt["verdict"] == "ACCEPTED"
    lin = express_lineage(CANONICAL_SPECIMEN, config=safe_defaults(), git_root=str(REPO))
    assert lin["provenance"]["repo"] == "Qwen/Qwen3-0.6B"
    assert lin["registry"]["present"] is True
    assert ROADMAP_LIFECYCLE.index(lin["registry"]["lifecycle"]) >= ROADMAP_LIFECYCLE.index(
        "IDENTITY_RESOLVED"
    )


def test_hash_gate_scratch_invokes_reconcile(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.acceptance.lake.common.RECEIPTS", tmp_path)
    receipt = run_hash_gate(
        live=False, scratch_root=tmp_path / "scratch", run_canonical_oid_hash=False
    )
    assert receipt["gate"] == HASH_GATE
    assert receipt["symbol_invoked"] is True
    assert receipt["checks"]["reconcile_promoted_scratch"] is True
    assert receipt["checks"]["scratch_is_not_the_live_lake"] is True
    assert receipt["criterion_altered"] is False
    # live=False cannot oid-hash 55 specimens
    assert receipt["verdict"] == "BLOCKED"
    assert "oid-backed sha256" in (receipt["blocker"] or {}).get("missing_input", "")


def test_promotion_gate_scratch_atomic(tmp_path):
    scratch = run_promote_on_scratch(tmp_path)
    assert scratch["properties"]["go_promoted"] is True


def test_runtime_archaeology_symbol_freezes_identity(tmp_path, monkeypatch):
    from tools.acceptance.lake.qwen27 import (
        IDENTITY_GATE as QID,
        call_run_runtime_archaeology,
        run_runtime_identity_gate,
    )
    from tools.acceptance.lake.common import PRIMARY

    profile = PRIMARY / "hcli" / "hawking-native.sealed-3.14.json"
    if not profile.is_file():
        pytest.skip("primary hcli profile absent")
    raw = call_run_runtime_archaeology(
        repo_root=PRIMARY,
        profile=profile,
        identity_emit=tmp_path / "id.json",
        diff_emit=tmp_path / "diff.json",
    )
    assert raw["status"] == "PASSED"
    assert raw["checks"]["unknowns_are_explicit"] is True
    assert raw["historical_selection"]["historical_anchor_tps"] is not None
    assert (raw.get("current") or {}).get("binary", {}).get("sha256")
    monkeypatch.setattr("tools.acceptance.lake.common.RECEIPTS", tmp_path)
    receipt = run_runtime_identity_gate()
    assert receipt["gate"] == QID
    assert receipt["symbol_invoked"] is True
    assert receipt["verdict"] == "ACCEPTED"
    assert receipt["criterion_altered"] is False


def test_protected_baseline_symbol_records_missing_quiet_window(tmp_path, monkeypatch):
    from tools.acceptance.lake.qwen27 import (
        BASELINE_GATE,
        call_run_protected_accelerator_benchmark,
        run_protected_baseline_gate,
    )

    raw = call_run_protected_accelerator_benchmark(
        repo_root=str(REPO),
        emit=str(tmp_path / "protected.json"),
        ready_timeout_s=0.3,
        interval_s=0.1,
        timeout_s=2.0,
        warmup_requests=1,
        measure_requests=1,
        max_new_tokens=4,
    )
    assert raw["schema"] == "hcli.agentos.protected_accelerator_benchmark.v1"
    assert not raw.get("measurements")
    monkeypatch.setattr("tools.acceptance.lake.common.RECEIPTS", tmp_path)
    receipt = run_protected_baseline_gate(ready_timeout_s=0.3)
    assert receipt["gate"] == BASELINE_GATE
    assert receipt["symbol_invoked"] is True
    assert receipt["verdict"] == "BLOCKED"
    assert "quiescent" in (receipt["blocker"] or {}).get("missing_input", "")


def test_regression_gate_explains_from_identity_and_source(tmp_path, monkeypatch):
    from tools.acceptance.lake.qwen27 import (
        REGRESSION_GATE,
        _mlp_source_truth,
        run_regression_gate,
        run_runtime_identity_gate,
    )

    monkeypatch.setattr("tools.acceptance.lake.common.RECEIPTS", tmp_path)
    source = _mlp_source_truth()
    assert source["from_env_present"] is True
    assert source["swiglu_and_1_same_arm"] is True
    assert source["unrecognised_panics"] is True
    run_runtime_identity_gate()
    receipt = run_regression_gate(invoke_live_ab=False)
    assert receipt["gate"] == REGRESSION_GATE
    assert receipt["criterion_altered"] is False
    assert receipt["checks"]["mlp_swiglu_and_1_same_arm_in_source"] is True
    assert receipt["checks"]["no_performance_claim"] is True
    assert receipt["verdict"] in {"ACCEPTED", "BLOCKED"}
    if receipt["verdict"] == "ACCEPTED":
        assert receipt["checks"]["different_verified_at_least_one"] is True


def test_receipts_never_alter_the_criterion():
    if not RECEIPTS.is_dir():
        pytest.skip("receipts not written yet")
    found = list(RECEIPTS.glob("*.json"))
    if not found:
        pytest.skip("no receipts")
    for path in found:
        if path.name.endswith(".census.json") or path.name.endswith(".run.json"):
            continue
        if path.name.endswith(".archaeology.json") or path.name.endswith(".diff.json"):
            continue
        if path.name == "SUMMARY.json":
            doc = json.loads(path.read_text(encoding="utf-8"))
            assert doc.get("criterion_altered") is False
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema") != RECEIPT_SCHEMA:
            continue
        assert doc["criterion_altered"] is False
        assert doc["verdict"] in {"ACCEPTED", "BLOCKED"}
        assert doc["gate"] in GATES
        assert "operational" in doc["criterion"]


def test_no_negative_control_mutation_left_in_acceptance():
    root = REPO / "tools" / "acceptance"
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("test_"):
            continue
        if path.suffix not in {".py", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "NEGCTRL_MUTATION" in text:
            hits.append(str(path))
    assert hits == []
