"""SELF_MEASURED_DIRTY: envelope, legitimate uses, promotion closed."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import contamination as C
from tools.future import dirty_measure as D
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _proc(*, pid=1, name="idle", cpu=0.0, rss=0.1):
    return {"pid": pid, "name": name, "cpu_pct": cpu, "rss_gib": rss, "state": "S"}


def _ok_processes(rows):
    return {
        "status": "OK",
        "method": "ps_enumerate",
        "cpu_pct_available": True,
        "no_name_filter": True,
        "n_enumerated": len(rows),
        "all": rows,
        "reason": None,
    }


def _probes(*, processes, load=None, memory=None, gpu=None, thermal=None, identity=None):
    return {
        "processes": processes,
        "load": load or {"status": "OK", "load_1m": 0.2, "load_5m": 0.2, "load_15m": 0.2, "ncpu": 28},
        "memory": memory
        or {
            "status": "OK",
            "pressure_level": 0,
            "pressure_name": "normal",
            "pages": {},
            "bytes": {},
        },
        "gpu_occupancy": gpu
        or {
            "status": "OK",
            "device_utilization_pct": 0,
            "renderer_utilization_pct": 0,
            "tiler_utilization_pct": 0,
            "gpu_core_count": 60,
        },
        "thermal": thermal or {"status": "UNKNOWN", "reason": "test"},
        "machine_identity": identity or {"hash": "test-identity", "fields": {"hw.model": "test"}},
    }


def _env(**kwargs):
    declared = kwargs.pop("declared_resident", None)
    ordinal = kwargs.pop("benchmark_ordinal", 0)
    return D.dirty_snapshot(
        probes=_probes(**kwargs),
        declared_resident=declared,
        benchmark_ordinal=ordinal,
    )


def _duration_faster_b(n=None):
    n = C.MIN_PAIRS if n is None else n
    return [(10.0, 8.0)] * n


def _duration_slower_b(n=None):
    n = C.MIN_PAIRS if n is None else n
    return [(8.0, 10.0)] * n


def test_entry_point_runs_and_seals_receipt():
    out = D.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DIRTY_MEASUREMENT.json"
    assert doc["schema"] == D.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["evidence_class"] == D.EVIDENCE_CLASS
    assert doc["gpu_authority"] is False
    assert doc["resident_loaded"]["did_not_start_resident"] is True
    assert "gpu_processes" in doc
    assert "memory_pressure" in doc
    assert "thermal_state" in doc
    assert "competing_workloads" in doc
    assert doc["contamination_fingerprint"]
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["can_hcli_invoke"] is True
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["workunit"]["id"] == "future.dirty-measure.self-snapshot"
    assert doc["resident_callable"]["workunit"]["output_receipt_path"] == "receipts/future/DIRTY_MEASUREMENT.json"
    assert doc["resident_callable"]["sleeping_protected_followup"]["status"] == "SLEEPING"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_receipt_contains_no_hardware_measurement_fields():
    doc = json.loads(D.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"{here} = {v!r} is a hardware field")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_envelope_carries_resident_gpu_memory_thermal_competing_fingerprint():
    env = _env(
        processes=_ok_processes([_proc(pid=9, name="Python", cpu=5.0, rss=3.5)]),
        thermal={"status": "UNKNOWN", "reason": "platform does not expose thermal"},
        declared_resident={"model": "sealed-3.14", "pid": 9, "process_name": "Python"},
    )
    assert env["evidence_class"] == D.EVIDENCE_CLASS
    assert env["measurement_class"] == "STATIC_ONLY"
    assert env["promotable"] is False
    assert env["resident_loaded"]["model"] == "sealed-3.14"
    assert env["resident_loaded"]["pid"] == 9
    assert env["resident_loaded"]["status"] == "OK"
    assert env["gpu_processes"]["no_name_filter"] is True
    assert env["memory_pressure"]["status"] == "OK"
    thermal = env["thermal_state"]
    assert thermal == "UNKNOWN" or (isinstance(thermal, dict) and thermal.get("status") in {"OK", "UNKNOWN"})
    assert env["competing_workloads"]
    assert env["contamination_fingerprint"]
    assert env["contamination_class"] in C.CONTAMINATION_CLASSES


def test_resident_model_is_unknown_unless_declared_never_guessed():
    env = _env(processes=_ok_processes([_proc(pid=4, name="Python", cpu=1.0, rss=9.0)]))
    loaded = env["resident_loaded"]
    assert loaded["model"] == "UNKNOWN"
    assert loaded["pid"] == 4
    assert loaded["status"] == "PARTIAL"
    assert loaded["did_not_start_resident"] is True


def test_fingerprint_stable_for_same_state_differs_when_pid_changes():
    rows = [_proc(pid=9, name="Python", cpu=5.0, rss=3.5)]
    a = _env(processes=_ok_processes(rows))
    b = _env(processes=_ok_processes(rows))
    assert a["contamination_fingerprint"] == b["contamination_fingerprint"]
    # Recompute from the snapshot: same bytes, no wall-clock.
    again = D.contamination_fingerprint(a["snapshot"])
    assert again == a["contamination_fingerprint"]
    shifted = _env(processes=_ok_processes([_proc(pid=10, name="Python", cpu=5.0, rss=3.5)]))
    assert shifted["contamination_fingerprint"] != a["contamination_fingerprint"]


def test_fingerprint_ignores_rss_jitter_inside_the_same_class():
    a = _env(processes=_ok_processes([_proc(pid=9, name="Python", cpu=5.0, rss=3.5)]))
    b = _env(processes=_ok_processes([_proc(pid=9, name="Python", cpu=5.0, rss=3.7)]))
    assert a["contamination_class"] == b["contamination_class"]
    assert a["contamination_fingerprint"] == b["contamination_fingerprint"]


def test_cheap_paired_ab_reports_faster_arm_and_ratio_not_magnitude():
    env = _env(processes=_ok_processes([_proc()]))
    rec = D.cheap_paired_ab(_duration_faster_b(), quantity_kind="duration", envelope=env)
    assert rec["faster_arm"] == "B"
    assert rec["median_ratio"] == pytest.approx(0.8)
    assert rec["ratio_iqr"] == pytest.approx(0.0)
    assert rec["sufficient_for_decision"] is True
    assert rec["evidence_class"] == D.EVIDENCE_CLASS
    assert rec["measurement_class"] == "STATIC_ONLY"
    assert rec["spread"]["bootstrap_ci95"]
    for key in HARDWARE_FIELDS:
        assert key not in rec
    assert "latency" not in rec
    assert "absolute_tps" not in rec
    assert "mean" not in rec["ab_stats"]
    assert rec["ab_stats"][D.DIRTY_BINDING_KEY]["evidence_class"] == D.EVIDENCE_CLASS


def test_exact_tie_is_a_direction_decision_and_does_not_prune():
    env = _env(processes=_ok_processes([_proc()]))
    pairs = [(10.0, 10.0)] * C.MIN_PAIRS
    rec = D.effect_direction(pairs, quantity_kind="duration", envelope=env)
    assert rec["faster_arm"] == "TIE"
    assert rec["sufficient_for_direction"] is True
    pruned = D.prune_dominated(
        [{"a": "x", "b": "y", "pairs": pairs}],
        quantity_kind="duration",
        envelope=env,
    )
    assert pruned["pruned"] == []
    assert pruned["sufficient_for_decision"] is True
    assert "x" in pruned["survivors"] and "y" in pruned["survivors"]


def test_effect_direction_is_explicit_api_and_quantity_kind_matters():
    env = _env(processes=_ok_processes([_proc()]))
    duration = D.effect_direction(_duration_faster_b(), quantity_kind="duration", envelope=env)
    rate = D.effect_direction(_duration_faster_b(), quantity_kind="rate", envelope=env)
    assert duration["use"] == "direction"
    assert duration["faster_arm"] == "B"
    assert rate["faster_arm"] == "A"
    with pytest.raises(ValueError, match="quantity_kind"):
        D.effect_direction(_duration_faster_b(), quantity_kind="tokens", envelope=env)


def test_rank_candidates_orders_sufficient_dirty_results():
    env = _env(processes=_ok_processes([_proc()]))
    ranked = D.rank_candidates(
        [
            {"id": "slow", "pairs": _duration_slower_b()},
            {"id": "fast", "pairs": _duration_faster_b()},
        ],
        quantity_kind="duration",
        envelope=env,
    )
    assert ranked["use"] == "rank"
    assert ranked["sufficient_for_decision"] is True
    assert [r["id"] for r in ranked["ranked"]] == ["fast", "slow"]
    assert ranked["ranked"][0]["rank"] == 1


def test_noisy_dirty_result_is_not_used_to_rank():
    env = _env(processes=_ok_processes([_proc()]))
    short = [(10.0, 8.0)] * 2
    ranked = D.rank_candidates(
        [
            {"id": "ok", "pairs": _duration_faster_b()},
            {"id": "also", "pairs": _duration_slower_b()},
            {"id": "noisy", "pairs": short},
        ],
        quantity_kind="duration",
        envelope=env,
    )
    assert ranked["sufficient_for_decision"] is True
    assert [r["id"] for r in ranked["ranked"]] == ["ok", "also"]
    assert any(u["id"] == "noisy" for u in ranked["unranked"])
    only_noise = D.rank_candidates(
        [{"id": "a", "pairs": short}, {"id": "b", "pairs": short}],
        quantity_kind="duration",
        envelope=env,
    )
    assert only_noise["sufficient_for_decision"] is False
    assert only_noise["ranked"] == []
    assert "noisy" in only_noise["reason"] or "n_rankable" in only_noise["reason"]


def test_use_dispatcher_is_the_explicit_api():
    env = _env(processes=_ok_processes([_proc()]))
    rec = D.use("cheap_paired_ab", _duration_faster_b(), quantity_kind="duration", envelope=env)
    assert rec["use"] == "cheap_paired_ab"
    with pytest.raises(ValueError, match="legitimate uses"):
        D.use("promote", _duration_faster_b(), quantity_kind="duration", envelope=env)


def test_prune_dominated_only_when_ci_excludes_one_and_is_not_protected_reject():
    env = _env(processes=_ok_processes([_proc()]))
    pruned = D.prune_dominated(
        [{"a": "keep", "b": "slow", "pairs": _duration_slower_b()}],
        quantity_kind="duration",
        envelope=env,
    )
    assert pruned["use"] == "prune_dominated"
    assert [p["id"] for p in pruned["pruned"]] == ["slow"]
    assert pruned["pruned"][0]["status"] == "PRUNED_SELF_MEASURED_DIRTY"
    assert pruned["pruned"][0]["not_a_protected_reject"] is True
    assert "keep" in pruned["survivors"]
    noisy = D.prune_dominated(
        [{"a": "x", "b": "y", "pairs": [(10.0, 10.1), (10.0, 9.9)]}],
        quantity_kind="duration",
        envelope=env,
    )
    assert noisy["pruned"] == []
    assert noisy["sufficient_for_decision"] is False


def test_direction_refuses_absolute_latency_or_tps():
    env = _env(processes=_ok_processes([_proc()]))
    rec = D.cheap_paired_ab(_duration_faster_b(), quantity_kind="duration", envelope=env)
    with pytest.raises(D.DirtyMagnitudeRefused, match="accepted_tps"):
        D._assert_no_magnitude({"accepted_tps": rec["median_ratio"]})
    with pytest.raises(D.DirtyMagnitudeRefused, match="latency"):
        D._assert_no_magnitude({"latency": 12.0})
    with pytest.raises(D.DirtyMagnitudeRefused):
        D.rank_candidates(
            [{"id": "x", "accepted_tps": 24.4, "pairs": _duration_faster_b()}],
            quantity_kind="duration",
            envelope=env,
        )


def test_write_receipt_refuses_hardware_fields_as_backstop():
    with pytest.raises(HardwareClaimError, match="tps"):
        write_receipt(
            "_DIRTY_MEASURE_SHOULD_NOT_LAND.json",
            {"schema": "test", "tps": 12.0},
            "tools/future/test_dirty_measure.py",
        )


def test_negative_control_dirty_record_offered_for_promotion_is_refused():
    env = _env(processes=_ok_processes([_proc()]))
    rec = D.cheap_paired_ab(_duration_faster_b(), quantity_kind="duration", envelope=env)
    with pytest.raises(C.PromotionRefused, match="SELF_MEASURED_DIRTY"):
        D.offer_for_promotion(rec)
    with pytest.raises(C.PromotionRefused, match="SELF_MEASURED_DIRTY"):
        D.offer_for_promotion({**dict(rec), "promotable": True, "offered_as": "PROTECTED_ABSOLUTE", "caveat": "ok"})
    with pytest.raises(C.PromotionRefused):
        C.assert_promotable(rec)
    with pytest.raises(C.PromotionRefused):
        D.assert_promotable(rec)
    with pytest.raises(C.PromotionRefused, match="frozen"):
        rec["evidence_class"] = "PROTECTED_ABSOLUTE"
    with pytest.raises(C.PromotionRefused, match="frozen"):
        rec["measurement_class"] = "PROTECTED_ABSOLUTE"


def test_negative_control_no_field_copy_path_to_protected_absolute():
    """A guard nobody has watched fail is not a guard.

    contamination.assert_promotable accepts a QUIESCENT PROTECTED_ABSOLUTE
    shell that copies dirty ratios. That hole is the field-copy path; this
    module must refuse it.
    """
    env = _env(processes=_ok_processes([_proc()]))
    rec = D.cheap_paired_ab(_duration_faster_b(), quantity_kind="duration", envelope=env)
    stolen = {
        "measurement_class": "PROTECTED_ABSOLUTE",
        "contamination_class": "QUIESCENT",
        "evidence_class": "PROTECTED_ABSOLUTE",
        "promotable": True,
        "caveat": "copied from dirty",
        "ab_stats": {
            "median_ratio": rec["median_ratio"],
            "ratio_q1": rec["ratio_q1"],
            "ratio_q3": rec["ratio_q3"],
            "ratio_iqr": rec["ratio_iqr"],
            "bootstrap_ci95": rec["bootstrap_ci95"],
            "sufficient_for_decision": True,
            "n_kept": rec["ab_stats"]["n_kept"],
            "reason": "copied",
        },
        "median_ratio": rec["median_ratio"],
    }
    # Watch the landed gate fail to catch this (the hole), then watch ours fire.
    C.assert_promotable(stolen)  # must not raise — otherwise we are not testing field-copy
    with pytest.raises(C.PromotionRefused, match="values match"):
        D.ingest_as_protected(stolen)
    with pytest.raises(C.PromotionRefused, match="field copy"):
        D.assert_promotable(stolen)
    with pytest.raises(C.PromotionRefused, match="field copy"):
        D.as_protected_absolute(stolen)
    with pytest.raises(C.PromotionRefused, match="copying its value"):
        D.copy_value_as(rec, "PROTECTED_ABSOLUTE")
    with pytest.raises(C.PromotionRefused, match="cannot mint"):
        D.mint_protected_absolute(values=dict(stolen))
    wholesale = {
        "measurement_class": "PROTECTED_ABSOLUTE",
        "contamination_class": "QUIESCENT",
        "ab_stats": dict(rec["ab_stats"]),
    }
    with pytest.raises(C.PromotionRefused):
        D.ingest_as_protected(wholesale)
    assert D.is_dirty_sourced(stolen) is True
    assert D.is_dirty_sourced(wholesale) is True


def test_public_functions_do_not_return_protected_absolute():
    src = Path(D.__file__).read_text()
    tree = ast.parse(src)
    hits = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            for d in ast.walk(child.value):
                if not isinstance(d, ast.Dict):
                    continue
                for key, val in zip(d.keys, d.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "measurement_class"
                        and isinstance(val, ast.Constant)
                        and val.value == "PROTECTED_ABSOLUTE"
                    ):
                        hits.append(node.name)
    assert hits == []
    assert D._public_return_protected_hits() == []


def test_sleeping_followup_is_not_a_synthetic_protected_result():
    unit = D.emit_workunit(sleeping_protected=True)
    assert unit["status"] == "SLEEPING"
    assert unit["measurement_class"] == "STATIC_ONLY"
    assert unit["may_promote"] is False
    assert "synthetic" in unit["blocked_reason"].lower() or "cannot become" in unit["blocked_reason"]
    runnable = D.emit_workunit()
    assert runnable["status"] == "pending"
    assert runnable["command"][-1] == "--snapshot"
    assert runnable["output_receipt_path"] == "receipts/future/DIRTY_MEASUREMENT.json"
    for name in D.HCLI_CORE_FIELDS:
        assert name in runnable
        assert name in unit


def test_queue_policy_is_recorded_without_assuming_checkout_shape():
    pol = D.queue_policy()
    assert "present" in pol
    assert "protected_start_requires_machine_quiescence" in pol
    if pol["present"]:
        assert pol["protected_start_requires_machine_quiescence"] is True
        assert pol["diagnostic_results_do_not_promote"] is True
    recovered = D._recovered()
    for row in recovered:
        assert "on_disk_in_this_worktree" in row
        assert isinstance(row["on_disk_in_this_worktree"], bool)


def test_selftest_watches_the_gates_fail():
    result = D.selftest()
    assert result["offer_for_promotion_raises"]["fired"] is True
    assert result["ingest_field_copy_raises"]["fired"] is True
    assert result["landed_contamination_gate_would_accept_field_copy"] is True
    assert result["no_public_return_of_protected_absolute"] is True
    assert result["sleeping_protected_followup"] == "SLEEPING"
    assert result["noisy_insufficient"] is True


def test_mixed_fingerprints_refuse_to_rank():
    env_a = _env(processes=_ok_processes([_proc(pid=1, name="Python", cpu=5.0, rss=3.5)]))
    env_b = _env(processes=_ok_processes([_proc(pid=2, name="Python", cpu=5.0, rss=3.5)]))
    rec_a = D.cheap_paired_ab(_duration_faster_b(), quantity_kind="duration", envelope=env_a)
    rec_b = D.cheap_paired_ab(_duration_slower_b(), quantity_kind="duration", envelope=env_b)
    ranked = D.rank_candidates(
        [{"id": "a", "dirty_record": rec_a}, {"id": "b", "dirty_record": rec_b}],
        quantity_kind="duration",
        envelope=env_a,
    )
    assert ranked["sufficient_for_decision"] is False
    assert ranked["ranked"] == []
    assert "fingerprint" in ranked["reason"]
