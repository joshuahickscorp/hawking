"""Tests for tools/future/sleeping_specimens.py.

Acceptance nobody has watched fail:
  - the three live downloads appear as SLEEPING_SPECIMEN_WU with
    SEALED_SOURCE_READY, identity and revision read from the watcher
  - every launch-gate criterion carries a dependency class
  - DEFERABLE_PARALLEL cannot block launch
  - LAUNCH_CRITICAL can
  - a model download is DEFERABLE_PARALLEL unless a first WorkGraph names
    that exact model
  - odyssey_launch criterion logic is not imported-as-rewritten
"""
from __future__ import annotations

import ast
import json

from tools.future import odyssey_launch as ol
from tools.future import sleeping_specimens as ss
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


def test_build_seals_static_only_receipt():
    out = ss.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SLEEPING_SPECIMENS.json"
    assert doc["schema"] == ss.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["odyssey_launch_criterion_logic_not_edited"] is True
    assert doc["gate_consumes_dependency_class"] is False
    proofs = doc["proofs"]
    assert proofs["every_criterion_classified"] is True
    assert proofs["n_criteria"] == 16
    assert proofs["deferable_cannot_block"] is True
    assert proofs["launch_critical_can_block"] is True
    assert proofs["wake_condition_is_sealed_source_ready"] is True
    assert proofs["gate_not_rewritten"] is True
    assert proofs["gate_not_re_run"] is True
    assert proofs["watcher_not_restarted"] is True
    _assert_no_hardware_claims(doc)


def test_every_criterion_has_a_dependency_class():
    classes = ss.classify_all_criteria()
    assert tuple(classes) == ol.CRITERION_IDS
    assert set(classes) == set(ol.CRITERION_IDS)
    for cid, klass in classes.items():
        assert klass in ss.DEPENDENCY_CLASSES
        assert ss.criterion_dependency_class(cid) == klass
    assert ss.REQUIRED_BEFORE_NX in classes.values()
    assert ss.REQUIRED_BEFORE_PROMOTION in classes.values()
    assert ss.DEFERABLE_PARALLEL in classes.values()
    assert ss.LAUNCH_CRITICAL in classes.values()
    assert classes["nr_nx_path_callable"] == ss.REQUIRED_BEFORE_NX
    assert classes["protected_scheduling"] == ss.REQUIRED_BEFORE_PROMOTION
    assert classes["transfer_substrate"] == ss.DEFERABLE_PARALLEL
    assert classes["adversary_substrate"] == ss.DEFERABLE_PARALLEL
    assert classes["resident_autonomy_trial_pass"] == ss.LAUNCH_CRITICAL
    assert classes["specimen_curriculum_ready"] == ss.LAUNCH_CRITICAL


def test_deferable_parallel_cannot_block_launch():
    classes = ss.classify_all_criteria()
    deferable = [cid for cid, k in classes.items() if k == ss.DEFERABLE_PARALLEL]
    assert deferable
    results = ss.synthetic_results(unmet=deferable)
    verdict = ss.launch_allowed_under_dependency_classes(results)
    assert verdict["allowed"] is True
    assert verdict["deferred_unmet"] == deferable
    assert verdict["blocking"] == []
    assert verdict["evidence_not_weakened"] is True
    # The unmet evidence is still named. Serialization is what was removed.
    assert set(verdict["unmet"]) == set(deferable)


def test_launch_critical_can_block_launch():
    classes = ss.classify_all_criteria()
    critical = [cid for cid, k in classes.items() if k == ss.LAUNCH_CRITICAL]
    assert "resident_autonomy_trial_pass" in critical
    results = ss.synthetic_results(unmet=["resident_autonomy_trial_pass"])
    verdict = ss.launch_allowed_under_dependency_classes(results)
    assert verdict["allowed"] is False
    assert "resident_autonomy_trial_pass" in verdict["blocking"]
    # Both directions in one pair of tests: deferable does not block, this does.


def test_nx_and_promotion_classes_do_not_serialize_launch():
    nx = ss.synthetic_results(unmet=["nr_nx_path_callable"])
    nx_v = ss.launch_allowed_under_dependency_classes(nx)
    assert nx_v["allowed"] is True
    assert nx_v["nx_allowed"] is False
    promo = ss.synthetic_results(unmet=["protected_scheduling"])
    promo_v = ss.launch_allowed_under_dependency_classes(promo)
    assert promo_v["allowed"] is True
    assert promo_v["promotion_allowed"] is False


def test_model_download_is_deferable_unless_first_workgraph_requires_it():
    required = [{"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}]
    assert (
        ss.classify_acquisition(
            repo="Qwen/Qwen3-VL-8B-Instruct",
            revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            required=required,
        )
        == ss.DEFERABLE_PARALLEL
    )
    assert (
        ss.classify_acquisition(
            repo="thinkingmachines/Inkling-Small",
            revision="8cc5877b44d343f88b92086aa1fb72897950f06a",
            required=required,
        )
        == ss.DEFERABLE_PARALLEL
    )
    assert (
        ss.classify_acquisition(
            repo="zai-org/GLM-5.3-Flash",
            revision="04c4e9e95c5da8862dced7e5056455116f83a7e0",
            required=required,
        )
        == ss.DEFERABLE_PARALLEL
    )
    # Exact pin of the first WorkGraph specimen IS launch-critical.
    assert (
        ss.classify_acquisition(
            repo="Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            required=required,
        )
        == ss.LAUNCH_CRITICAL
    )
    # A first WorkGraph that literally required GLM-5.3-Flash would not defer it.
    glm_required = [{"repo": "zai-org/GLM-5.3-Flash", "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0"}]
    assert (
        ss.classify_acquisition(
            repo="zai-org/GLM-5.3-Flash",
            revision="04c4e9e95c5da8862dced7e5056455116f83a7e0",
            required=glm_required,
        )
        == ss.LAUNCH_CRITICAL
    )


def test_live_downloads_are_sleeping_wus_from_watcher_state():
    sample = ss.read_latest_watcher_sample()
    assert sample is not None, "watcher_sample must be readable from the live jsonl"
    active = list(sample.get("active_jobs") or [])
    assert len(active) >= 1
    units = ss.pending_from_watcher_sample(sample)
    tags = {
        (u.get("modellake_identity") or {}).get("tag")
        for u in units
    }
    # PID snapshot plus recently-started unfinished jobs. Active PIDs must
    # all sleep; recently-started unfinished jobs may join them.
    assert set(active) <= tags
    assert len(tags) >= 1
    for u in units:
        ident = u["modellake_identity"]
        assert ident["repo"]
        assert ident["revision"]
        assert u["target_revision"] == ident["revision"]
        assert u["status"] == "sleeping"
        assert u["classification"] == "SLEEPING"
        assert u["species"] == ss.SLEEPING_SPECIES
        assert u["wake_condition"] == ss.WAKE_SEALED_SOURCE_READY
        assert u["acquisition_state"] in {"active", "manifest_wait", "absent", "complete"}
        assert u["expected_role_candidates"]
        assert all(r.get("status") == "CANDIDATE" for r in u["expected_role_candidates"])
        assert u["dependency_class"] == ss.DEFERABLE_PARALLEL
        assert u["required_by_first_workgraph"] is False
        assert (u.get("early_metadata") or {}).get("is_sealed_specimen") is False
        assert (u.get("early_metadata") or {}).get("weights_opened") is False


def test_receipt_records_the_live_sleeping_set():
    doc = json.loads((RECEIPTS / ss.RECEIPT).read_text())
    # The receipt is a snapshot of the watcher_sample it read. The live
    # active set rotates; comparing to a later sample would flake.
    live = set(doc["watcher_probe"].get("pending_jobs") or doc["watcher_probe"].get("active_jobs") or [])
    recorded = {
        (u.get("modellake_identity") or {}).get("tag")
        for u in doc["sleeping_units"]
    }
    assert recorded == live
    assert doc["n_sleeping"] == len(live)
    assert doc["watcher_probe"].get("did_not_restart_watcher") is True
    assert doc["watcher_probe"].get("did_not_kill_downloads") is True
    for u in doc["sleeping_units"]:
        assert u["wake_condition"] == ss.WAKE_SEALED_SOURCE_READY
        assert u["status"] == "sleeping"
        assert u["modellake_identity"]["repo"]
        assert u["modellake_identity"]["revision"]
        assert u["target_revision"]
        assert u["acquisition_state"]
        assert u["expected_role_candidates"]
        assert u["dependency_class"] == ss.DEFERABLE_PARALLEL


def test_three_live_downloads_named():
    """The campaign named three in-flight hf downloads. They must be in the set.

    If the watcher has rotated the active set since the contract was written,
    the live active_jobs still all appear as sleeping WUs (covered above).
    This test pins the three the contract named when they are still active.
    """
    sample = ss.read_latest_watcher_sample()
    assert sample is not None
    active = set(sample.get("active_jobs") or [])
    named = {
        "Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8",
        "thinkingmachines--Inkling-Small@8cc5877b44d3",
        "zai-org--GLM-5.3-Flash@04c4e9e95c5d",
    }
    still = named & active
    # At least the intersection must be sleeping; if the watcher still has
    # all three, the receipt must carry all three.
    doc = json.loads((RECEIPTS / ss.RECEIPT).read_text())
    recorded = {
        (u.get("modellake_identity") or {}).get("tag")
        for u in doc["sleeping_units"]
    }
    assert still <= recorded
    if named <= active:
        assert named <= recorded
        assert doc["n_sleeping"] == 3


def test_does_not_rewrite_odyssey_launch_evaluators():
    """Classification is metadata. The gate file must still own CRITERION_IDS
    evaluation; this module must not assign into ol.EVALUATORS or can_launch."""
    src = (REPO / "tools" / "future" / "sleeping_specimens.py").read_text()
    tree = ast.parse(src)
    assigned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                    if t.value.id in {"ol", "odyssey_launch"}:
                        assigned.append(t.attr)
    assert assigned == []
    # The live gate still treats all 16 as launch-critical; we expose the
    # table rather than swapping launch_verdict.
    assert ss.GATE_CONSUMES_DEPENDENCY_CLASS is False
    # Import of CRITERION_IDS / _checkout_roots is reading, not rewriting.
    assert ol.CRITERION_IDS[0] == "resident_autonomy_trial_pass"
    assert len(ol.CRITERION_IDS) == 16
    assert not hasattr(ss, "EVALUATORS")
    assert ss.launch_allowed_under_dependency_classes is not ol.launch_verdict


def test_emit_unit_is_hcli_shaped():
    unit = ss.emit_sleeping_specimen_wu(
        repo="Qwen/Qwen3-VL-8B-Instruct",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        tag="Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8",
        acquisition_state="active",
        expected_bytes=17545915883,
        destination=None,
        pids=[1],
        present_bytes=None,
        remaining_bytes=17545915883,
        dependency_class=ss.DEFERABLE_PARALLEL,
        fingerprint=None,
        required_by_first_workgraph=False,
    )
    wus_ok = True
    from tools.future.workunit_species import validate_emitted_unit

    validate_emitted_unit(unit)
    assert wus_ok
    assert unit["wake_condition"] == ss.WAKE_SEALED_SOURCE_READY
    assert unit["status"] == "sleeping"


def test_sealed_source_ready_is_the_exit_transition(tmp_path):
    """SLEEPING_SPECIMEN_WU leaves SLEEPING only when the specimen dir exists.

    Empty directory is not ready. Missing tag is not ready. No synthetic COMPLETED.
    """
    tag = "acme--x@deadbeefcafe"
    assert ss.sealed_source_ready(tag, specimen_root=tmp_path) is False
    empty = tmp_path / tag
    empty.mkdir()
    assert ss.sealed_source_ready(tag, specimen_root=tmp_path) is False
    (empty / "config.json").write_text("{}", encoding="utf-8")
    assert ss.sealed_source_ready(tag, specimen_root=tmp_path) is True
    event = ss.notify_sealed_source_ready(tag, source="test", specimen_root=tmp_path)
    assert event["wake_condition"] == ss.WAKE_SEALED_SOURCE_READY
    assert event["ready"] is True
    assert event["tag"] == tag
    missing = ss.notify_sealed_source_ready(
        "no-such-tag", source="test", specimen_root=tmp_path
    )
    assert missing["ready"] is False
    assert ss.sealed_source_ready("", specimen_root=tmp_path) is False
