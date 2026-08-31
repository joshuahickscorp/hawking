"""Contamination science: snapshot, class, A/B stats, promotion gate."""
from __future__ import annotations

import inspect
import json
import pathlib

import pytest

from tools.future import contamination as C
from tools.future import status_causality as sc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def _proc(*, pid=1, name="idle", cpu=0.0, rss=0.1, extra=None):
    row = {"pid": pid, "name": name, "cpu_pct": cpu, "rss_gib": rss, "state": "S"}
    if extra:
        row.update(extra)
    return row


def _probes(*, processes, load=None, memory=None, gpu=None, thermal=None):
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
        "machine_identity": {"hash": "test-identity", "fields": {"hw.model": "test"}},
    }


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


def _snap(**kwargs):
    return C.snapshot(benchmark_ordinal=kwargs.pop("benchmark_ordinal", 0), probes=_probes(**kwargs))


def _ok_stats():
    return C.paired_ab_stats(C.SYNTHETIC_PAIRS)


def test_build_emits_sealed_receipt():
    out = C.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "CONTAMINATION_SCIENCE.json"
    assert doc["schema"] == C.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["contamination_class"] in C.CONTAMINATION_CLASSES
    assert doc["snapshot"]["gpu_processes"]["no_name_filter"] is True
    assert "benchmark_ordinal" in doc["snapshot"]
    assert doc["snapshot"]["machine_identity"]["hash"]
    assert doc["gate"]["selftest"]["diagnostic_relative_raises"]["fired"] is True
    assert doc["gate"]["selftest"]["heavy_raises"]["fired"] is True
    assert doc["gate"]["selftest"]["quiescent_protected_passes"] is True
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_receipt_contains_no_hardware_measurement_fields():
    doc = json.loads(C.build().read_text())

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


def test_live_receipt_is_not_promotable():
    doc = json.loads(C.build().read_text())
    with pytest.raises(C.PromotionRefused):
        C.assert_promotable(doc)


def test_snapshot_shape_and_ordinal():
    snap = _snap(processes=_ok_processes([_proc()]), benchmark_ordinal=3)
    assert snap["benchmark_ordinal"] == 3
    assert snap["thermal_state"] == "UNKNOWN" or snap["thermal_state"]["status"] in {"OK", "UNKNOWN"}
    assert snap["gpu_processes"]["no_name_filter"] is True
    assert "names" not in inspect.signature(C.probe_processes).parameters
    assert set(inspect.signature(C.probe_processes).parameters) == set()


def test_classify_quiescent_when_required_probes_are_clean():
    snap = _snap(processes=_ok_processes([_proc(cpu=1.0, rss=0.2)]))
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "QUIESCENT"
    assert got["required_probes"]["processes_cpu_pct"] is True
    assert got["contamination_evidence"]


def test_classify_unknown_when_required_probe_fails_never_optimistic_quiescent():
    processes = {
        "status": "FAILED",
        "method": None,
        "cpu_pct_available": False,
        "no_name_filter": True,
        "reason": "ps exited 1",
        "n_enumerated": 0,
        "all": [],
    }
    snap = _snap(processes=processes)
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "UNKNOWN"
    assert got["contamination_class"] != "QUIESCENT"
    assert any(e["kind"] == "probe_failed" for e in got["contamination_evidence"])


def test_classify_partial_ps_without_cpu_cannot_be_quiescent():
    processes = {
        "status": "PARTIAL",
        "method": "libproc_enumerate_rss",
        "cpu_pct_available": False,
        "no_name_filter": True,
        "reason": "ps failed; RSS only",
        "n_enumerated": 2,
        "all": [_proc(cpu=None, rss=0.1), _proc(pid=2, cpu=None, rss=0.2)],
    }
    snap = _snap(processes=processes)
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "UNKNOWN"


def test_classify_light_from_rss_contender():
    snap = _snap(processes=_ok_processes([_proc(name="Python", cpu=5.0, rss=3.5)]))
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "LIGHT"
    assert snap["resident_local_model"]["name"] == "Python"
    assert snap["gpu_processes"]["processes"][0]["pid"] == 1


def test_classify_heavy_from_rss():
    snap = _snap(processes=_ok_processes([_proc(name="mlx", cpu=10.0, rss=19.4)]))
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "HEAVY"
    assert any(e["contributes"] == "HEAVY" and e["probe"] == "rss" for e in got["contamination_evidence"])


def test_classify_heavy_from_cpu():
    snap = _snap(processes=_ok_processes([_proc(name="build", cpu=450.0, rss=0.4)]))
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "HEAVY"
    assert any(e["probe"] == "cpu" for e in got["contamination_evidence"])


def test_classify_heavy_wins_even_when_ps_cpu_is_missing():
    processes = {
        "status": "PARTIAL",
        "method": "libproc_enumerate_rss",
        "cpu_pct_available": False,
        "no_name_filter": True,
        "reason": "ps failed",
        "n_enumerated": 1,
        "all": [_proc(name="Python", cpu=None, rss=19.4)],
    }
    snap = _snap(processes=processes)
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "HEAVY"


def test_classify_light_from_gpu_occupancy():
    snap = _snap(
        processes=_ok_processes([_proc(cpu=1.0, rss=0.1)]),
        gpu={
            "status": "OK",
            "device_utilization_pct": 11,
            "renderer_utilization_pct": 9,
            "tiler_utilization_pct": 9,
        },
    )
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "LIGHT"


def test_classify_heavy_from_memory_pressure():
    snap = _snap(
        processes=_ok_processes([_proc()]),
        memory={"status": "OK", "pressure_level": 2, "pressure_name": "urgent", "pages": {}, "bytes": {}},
    )
    got = C.classify_contamination(snap)
    assert got["contamination_class"] == "HEAVY"


def test_paired_ab_reports_median_iqr_count_and_never_a_mean():
    stats = C.paired_ab_stats([(10.0, 12.0)] * 7)
    assert stats["n_kept"] == 7
    assert stats["median_ratio"] == pytest.approx(1.2)
    assert stats["ratio_iqr"] == pytest.approx(0.0)
    assert stats["sufficient_for_decision"] is True
    assert stats["bootstrap_ci95"][0] <= stats["median_ratio"] <= stats["bootstrap_ci95"][1]
    assert "mean" not in stats
    assert "average" not in stats
    assert "mean_ratio" not in stats
    assert stats["summary"].startswith("median")


def test_paired_ab_insufficient_samples():
    stats = C.paired_ab_stats([(10.0, 11.0)] * 3)
    assert stats["n_kept"] == 3
    assert stats["sufficient_for_decision"] is False
    assert "min_pairs=7" in stats["reason"]
    assert stats["median_ratio"] == pytest.approx(1.1)


def test_paired_ab_drops_non_positive_a_and_is_deterministic():
    pairs = [(0.0, 1.0)] + [(10.0, 11.0)] * 7
    a = C.paired_ab_stats(pairs)
    b = C.paired_ab_stats(pairs)
    assert a["n_dropped"] == 1
    assert a["n_kept"] == 7
    assert a == b
    assert a["bootstrap_ci95"] == b["bootstrap_ci95"]


def test_paired_ab_iqr_is_not_a_bare_median():
    # seven pairwise ratios: 1,1,1,1,2,2,2 → median 1, IQR > 0
    pairs = [(10.0, 10.0)] * 4 + [(10.0, 20.0)] * 3
    stats = C.paired_ab_stats(pairs)
    assert stats["median_ratio"] == pytest.approx(1.0)
    assert stats["ratio_iqr"] > 0
    assert stats["ratio_q1"] != stats["ratio_q3"]


def test_assert_promotable_is_discriminating_not_blanket():
    """Negative control: the gate must fire, and must also let a clean record through."""
    good = {
        "measurement_class": "PROTECTED_ABSOLUTE",
        "contamination_class": "QUIESCENT",
        "ab_stats": _ok_stats(),
    }
    C.assert_promotable(good)  # must not raise

    with pytest.raises(C.PromotionRefused, match="DIAGNOSTIC_RELATIVE"):
        C.assert_promotable(
            {
                "measurement_class": "DIAGNOSTIC_RELATIVE",
                "offered_as": "PROTECTED_ABSOLUTE",
                "contamination_class": "QUIESCENT",
                "ab_stats": _ok_stats(),
            }
        )

    with pytest.raises(C.PromotionRefused, match="HEAVY"):
        C.assert_promotable(
            {
                "measurement_class": "PROTECTED_ABSOLUTE",
                "contamination_class": "HEAVY",
                "ab_stats": _ok_stats(),
            }
        )


def test_assert_promotable_refuses_light_unknown_and_short_samples():
    stats = _ok_stats()
    with pytest.raises(C.PromotionRefused, match="LIGHT"):
        C.assert_promotable(
            {
                "measurement_class": "PROTECTED_ABSOLUTE",
                "contamination_class": "LIGHT",
                "ab_stats": stats,
            }
        )
    with pytest.raises(C.PromotionRefused, match="UNKNOWN"):
        C.assert_promotable(
            {
                "measurement_class": "PROTECTED_ABSOLUTE",
                "contamination_class": "UNKNOWN",
                "ab_stats": stats,
            }
        )
    with pytest.raises(C.PromotionRefused, match="insufficient"):
        C.assert_promotable(
            {
                "measurement_class": "PROTECTED_ABSOLUTE",
                "contamination_class": "QUIESCENT",
                "ab_stats": C.paired_ab_stats([(10.0, 10.1)] * 2),
            }
        )


def test_assert_promotable_does_not_accept_qualified_protected_as_alias():
    with pytest.raises(C.PromotionRefused):
        C.assert_promotable(
            {
                "measurement_class": "QUALIFIED_PROTECTED",
                "contamination_class": "QUIESCENT",
                "ab_stats": _ok_stats(),
            }
        )


def test_selftest_watches_the_gate_fire():
    result = C.selftest()
    assert result["quiescent_protected_passes"] is True
    assert result["diagnostic_relative_raises"]["fired"] is True
    assert result["heavy_raises"]["fired"] is True
    assert "DIAGNOSTIC_RELATIVE" in result["diagnostic_relative_raises"]["message"]
    assert "HEAVY" in result["heavy_raises"]["message"]


def test_gpu_process_list_is_enumeration_not_a_name_filter():
    rows = [
        _proc(pid=9, name="fileproviderd", cpu=100.0, rss=0.01),
        _proc(pid=8, name="Python", cpu=5.0, rss=17.6),
        _proc(pid=7, name="idle", cpu=0.4, rss=0.05),
    ]
    snap = _snap(processes=_ok_processes(rows))
    names = {p["name"] for p in snap["gpu_processes"]["processes"]}
    assert "fileproviderd" in names
    assert "Python" in names
    assert "idle" not in names
    assert snap["gpu_processes"]["no_name_filter"] is True
    assert "modellake" not in json.dumps(snap["gpu_processes"]["attribution"])


# ---------------------------------------------------------------------------
# G007 consumer: build records the five causality fields.
# ---------------------------------------------------------------------------


def test_build_records_the_five_causality_fields():
    """A coverage number no test defends will drift back to zero."""
    out = C.build()
    doc = json.loads(out.read_text())
    assert C.records_five_fields(doc)
    src = pathlib.Path(C.__file__).read_text()
    assert "sc.emit(" in src
    assert doc["contamination_class"] in C.CONTAMINATION_CLASSES
    assert "snapshot" in doc["probe_performed"] or "classify_contamination" in doc["probe_performed"]
    assert doc["direct_observation"] != doc["contamination_class"]
    assert "contamination_class=" in doc["direct_observation"]
    assert doc["causality_verdict"] in {sc.SUPPORTED, sc.OVERREACHING, sc.UNTESTED}


def test_unsupplied_observation_records_untested_not_a_restatement():
    result = {"contamination_class": "HEAVY", "contamination_reason": "rss"}
    rec = C.record_contamination_causality(
        result, probe_performed="", direct_observation=""
    )
    assert rec["verdict"] == sc.UNTESTED
    assert rec["direct_observation"] in ("", None)
    assert rec["direct_observation"] != "HEAVY"
    assert "HEAVY" not in str(rec["direct_observation"] or "")
    assert result["contamination_class"] == "HEAVY"
    assert rec["interpretation"] != rec["direct_observation"]


def test_overreaching_does_not_override_contamination_class(monkeypatch):
    def overreach(status, **kwargs):
        return {
            "probe_performed": kwargs.get("probe_performed") or "p",
            "direct_observation": kwargs.get("direct_observation") or "o",
            "interpretation": kwargs.get("interpretation") or status,
            "confidence": {
                "level": "LOW",
                "about": "a",
                "would_raise": "b",
                "would_lower": "c",
            },
            "alternatives": [
                {
                    "hypothetical": "h",
                    "consistent_with_observation": True,
                    "consistent_with_claim": False,
                }
            ],
            "verdict": sc.OVERREACHING,
            "falsifier": "f",
            "probe_kind": sc.PROBE_MEASURED_FLAGS,
            "claim_kind": sc.CLAIM_OBJECT_ABSENCE,
        }

    monkeypatch.setattr(C.sc, "emit", overreach)
    out = C.build()
    doc = json.loads(out.read_text())
    assert doc["contamination_class"] in C.CONTAMINATION_CLASSES
    assert doc["causality_verdict"] == sc.OVERREACHING


def test_coverage_receipt_names_contamination_as_recording():
    path = RECEIPTS / "STATUS_CAUSALITY_COVERAGE.json"
    doc = json.loads(path.read_text())
    assert "contamination" in doc["recording_five_fields"]
    assert doc["n_gates"] == 18
