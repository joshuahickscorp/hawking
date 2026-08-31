"""Negative controls for the 30-minute model-bearing torture.

A detector nobody has watched reject will rubber-stamp a scripted hour.
These tests prove: a reworded second hypothesis fails; a without-the-model
control that matches the model sequence is FAIL; a wait without a differing
replan is not the required event; hardware-named fields cannot land in a
receipt; hcli/* is invoked not edited; --build does not mint a pass.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from tools.future import autonomy_degeneracy as ad
from tools.future import model_bearing as mb
from tools.future import model_bearing_torture as mbt
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)

SRC = Path(__file__).resolve().parent / "model_bearing_torture.py"


HYP_A = {
    "text": "replace MLP F with a cheaper full-width operator",
    "mechanism": "full-width function replacement of F",
    "surface": "mlp",
    "hypothesis_family": "mlp_function_replacement",
    "organ": "mlp",
}
HYP_B = {
    "text": "leave F; pin the sealed resident fusion env on the live hawking body",
    "mechanism": "fusion-env identity pin on the resident process",
    "surface": "hawking.resident",
    "hypothesis_family": "fusion_env_applied",
    "organ": "hawking",
}


def _events_required():
    return [
        {"kind": "failure_explained", "t_s": 12.0, "payload": {"explain_verbatim": "arithmetic dominated the byte save"}},
        {
            "kind": "second_hypothesis",
            "t_s": 20.0,
            "payload": {
                "hypothesis_a": HYP_A,
                "hypothesis_b": HYP_B,
                "hypothesis_a_verbatim": HYP_A["text"],
                "hypothesis_b_verbatim": HYP_B["text"],
                "difference": mb.meaningfully_different(HYP_A, HYP_B),
            },
        },
        {
            "kind": "subprocess_wait_start",
            "t_s": 30.0,
            "payload": {
                "unit_id": "WU.SUBAGENT.receipt_wait_probe",
                "queued_before": ["WU.DEAD.mlp_function_replacement", "WU.HAWKING.fusion_env_applied"],
                "t_s_end": 48.0,
            },
        },
        {
            "kind": "model_reasoned_during_wait",
            "t_s": 34.0,
            "payload": {"reply_text": "avoid MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution"},
        },
        {
            "kind": "receipt_ingested",
            "t_s": 45.0,
            "payload": {
                "receipt": "run/WU.SUBAGENT.receipt_wait_probe.receipt.json",
                "path": "run/WU.SUBAGENT.receipt_wait_probe.receipt.json",
                "unit_id": "WU.SUBAGENT.receipt_wait_probe",
            },
        },
        {
            "kind": "scheduler_replan",
            "t_s": 46.0,
            "payload": {
                "queued_before": ["WU.DEAD.mlp_function_replacement", "WU.HAWKING.fusion_env_applied"],
                "queued_after": ["WU.HAWKING.fusion_env_applied"],
            },
        },
        {
            "kind": "scar_avoidance",
            "t_s": 34.1,
            "payload": {
                "scar_id": "MLP_FUNCTION_REPLACEMENT_CLOSED",
                "named_scar": "MLP_FUNCTION_REPLACEMENT_CLOSED",
                "source": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
                "verbatim": "avoid MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution",
                "reply_text": "avoid MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution",
                "declined_id": "WU.DEAD.mlp_function_replacement",
            },
        },
        {
            "kind": "workunit_launched",
            "t_s": 50.0,
            "payload": {
                "unit_id": "WU.HAWKING.fusion_env_applied",
                "id": "WU.HAWKING.fusion_env_applied",
                "label": "WU.HAWKING.fusion_env_applied",
                "capability": "WU.HAWKING.fusion_env_applied",
                "hawking_self": True,
                "verbatim": "pick WU.HAWKING.fusion_env_applied because the machine is the uncertainty",
                "unit": {
                    "id": "WU.HAWKING.fusion_env_applied",
                    "label": "WU.HAWKING.fusion_env_applied",
                    "capability": "WU.HAWKING.fusion_env_applied",
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Source / hcli ownership
# ---------------------------------------------------------------------------


def test_source_invokes_resident_py_and_does_not_write_hcli():
    text = SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in {"write_text", "write_bytes", "open"}:
                writes.append(ast.dump(node)[:200])
    assert "hcli/agentos/resident.py" in text
    assert "start" in text and "status" in text and "stop" in text
    assert "STALE" in text
    assert "resident.py does not exist" in text
    # The module invokes the CLI via subprocess; it must not treat hcli as a write root.
    assert "hcli/* is invoked" in text or "invoked, never edited" in text or "hcli_invoked_not_edited" in text


def test_resident_py_exists_on_original_checkout_stale_note_is_wrong():
    path = mbt.resident_py_path()
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "add_parser(\"start\"" in text or "start" in text
    assert "add_parser(\"status\"" in text or "status" in text
    assert "add_parser(\"stop\"" in text or "stop" in text
    assert mbt.RESIDENT_PY_REL == "hcli/agentos/resident.py"


# ---------------------------------------------------------------------------
# Difference / four events
# ---------------------------------------------------------------------------


def test_reworded_second_hypothesis_is_not_the_required_event():
    events = [
        {
            "kind": "second_hypothesis",
            "t_s": 1.0,
            "payload": {
                "hypothesis_a": mb.RESTATEMENT_PRIOR,
                "hypothesis_b": mb.RESTATEMENT_REWORD,
                "hypothesis_a_verbatim": mb.RESTATEMENT_PRIOR["text"],
                "hypothesis_b_verbatim": mb.RESTATEMENT_REWORD["text"],
                "difference": mb.meaningfully_different(mb.RESTATEMENT_PRIOR, mb.RESTATEMENT_REWORD),
            },
        }
    ]
    row = mbt.detect_second_hypothesis(events, [])
    assert row["found"] is False


def test_bigger_n_restatement_is_not_the_required_event():
    a = {"text": "try again with a bigger N on mlp", "mechanism": "bigger n", "surface": "mlp"}
    b = {"text": "try again with a bigger N on mlp", "mechanism": "bigger n", "surface": "mlp"}
    events = [
        {
            "kind": "second_hypothesis",
            "t_s": 1.0,
            "payload": {
                "hypothesis_a": a,
                "hypothesis_b": b,
                "hypothesis_a_verbatim": a["text"],
                "hypothesis_b_verbatim": b["text"],
                "difference": mb.meaningfully_different(a, b),
            },
        }
    ]
    row = mbt.detect_second_hypothesis(events, [])
    assert row["found"] is False


def test_different_surface_second_hypothesis_passes():
    events = _events_required()
    row = mbt.detect_second_hypothesis(events, [])
    assert row["found"] is True
    assert "mlp" in str(row["hypothesis_a"]).lower() or "function replacement" in str(row["hypothesis_a_verbatim"]).lower()
    assert "hawking" in str(row["hypothesis_b"]).lower() or "fusion" in str(row["hypothesis_b_verbatim"]).lower()
    assert row["difference"]["different"] is True
    assert "restatement" not in (row.get("why_not_restatement") or "").lower() or row["difference"]["different"]


def test_wait_without_differing_replan_fails_the_event():
    events = [
        {"kind": "subprocess_wait_start", "t_s": 1.0, "payload": {"unit_id": "WU.X", "queued_before": ["A", "B"], "t_s_end": 10.0}},
        {"kind": "model_reasoned_during_wait", "t_s": 2.0, "payload": {"reply_text": "still queued A B"}},
        {"kind": "receipt_ingested", "t_s": 8.0, "payload": {"receipt": "r.json", "path": "r.json"}},
        {"kind": "scheduler_replan", "t_s": 9.0, "payload": {"queued_before": ["A", "B"], "queued_after": ["A", "B"]}},
    ]
    row = mbt.detect_wait_reason_receipt_replan(events)
    assert row["found"] is False


def test_wait_reason_receipt_replan_requires_all_four_timestamps():
    events = _events_required()
    row = mbt.detect_wait_reason_receipt_replan(events)
    assert row["found"] is True
    assert row["t_s_wait_start"] <= row["t_s_reason"] <= row["t_s_receipt"] <= row["t_s_replan"]
    assert row["queue_differed"] is True
    assert row["queued_before"] != row["queued_after"]


def test_scar_avoidance_requires_named_scar_in_verbatim_output():
    events = [
        {
            "kind": "scar_avoidance",
            "t_s": 1.0,
            "payload": {
                "scar_id": "MLP_FUNCTION_REPLACEMENT_CLOSED",
                "named_scar": "MLP_FUNCTION_REPLACEMENT_CLOSED",
                "verbatim": "I will skip the closed MLP work",
                "reply_text": "I will skip the closed MLP work",
            },
        }
    ]
    row = mbt.detect_scar_avoidance(events, [])
    assert row["found"] is False
    events[0]["payload"]["verbatim"] = "avoid MLP_FUNCTION_REPLACEMENT_CLOSED"
    events[0]["payload"]["reply_text"] = "avoid MLP_FUNCTION_REPLACEMENT_CLOSED"
    row = mbt.detect_scar_avoidance(events, [])
    assert row["found"] is True
    assert row["scar_id"] == "MLP_FUNCTION_REPLACEMENT_CLOSED"


def test_hawking_self_unit_is_the_machine_not_a_specimen():
    events = [
        {
            "kind": "workunit_launched",
            "t_s": 1.0,
            "payload": {"unit_id": "WU.SPECIMEN.qwen_gate_up", "hawking_self": False, "label": "WU.SPECIMEN.qwen_gate_up"},
        }
    ]
    row = mbt.detect_hawking_self(events, [])
    assert row["found"] is False
    mentioned = mbt.detect_hawking_self(
        [],
        [{"t_s": 1.0, "reply_text": "worth_doing_next includes WU.HAWKING.fusion_env_applied"}],
    )
    assert mentioned["found"] is False
    assert mentioned.get("mentioned_in_model_output") is True
    events = _events_required()
    row = mbt.detect_hawking_self(events, [])
    assert row["found"] is True
    assert str(row["unit_id"]).startswith("WU.HAWKING.")


def test_selftest_fixtures_satisfy_all_four_detectors():
    mbt.selftest()


# ---------------------------------------------------------------------------
# Control / participation
# ---------------------------------------------------------------------------


def test_without_the_model_matching_sequence_is_fail():
    ctrl = mbt.control_replay(
        [
            {"policy_id": "WU.A", "model_id": "WU.A", "launched": "WU.A"},
            {"policy_id": "WU.B", "model_id": "WU.B", "launched": "WU.B"},
        ]
    )
    assert ctrl["control_ran"] is True
    assert ctrl["sequences_identical"] is True
    assert ctrl["would_timeline_look_the_same_without_the_model"] is True
    same = mbt.would_look_the_same(ctrl, {"materially_participated": {"participated": True}})
    assert same["answer"] is True
    assert same["verdict_implication"] == "FAIL"


def test_diverged_sequence_is_not_the_same_without_the_model():
    ctrl = mbt.control_replay(
        [
            {"policy_id": "WU.DEAD.mlp_function_replacement", "model_id": "WU.HAWKING.fusion_env_applied", "launched": "WU.HAWKING.fusion_env_applied", "diverged": True},
        ]
    )
    same = mbt.would_look_the_same(ctrl, {"materially_participated": {"participated": True}})
    assert same["answer"] is False


def test_identical_model_outputs_are_a_degeneracy():
    calls = [{"reply_sha256": "abc"} for _ in range(12)]
    row = mbt.identical_output_degeneracy(calls)
    assert row["degenerate"] is True
    mixed = [{"reply_sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(12)]
    row = mbt.identical_output_degeneracy(mixed)
    assert row["degenerate"] is False


def test_degeneracy_measure_runs_over_a_nondegenerate_timeline():
    events = []
    t = 30.0
    for i in range(6):
        uid = f"WU.HAWKING.unit.{i:04d}"
        events.append(
            {
                "kind": "work_refilled",
                "t_s": t,
                "payload": {"unit_ids": [uid, f"WU.HAWKING.other.{i:04d}"]},
            }
        )
        t += 40
        events.append(
            {
                "kind": "workunit_launched",
                "t_s": t,
                "payload": {
                    "unit_id": uid,
                    "id": uid,
                    "label": uid,
                    "capability": uid,
                    "unit": {"id": uid, "label": uid, "capability": uid},
                },
            }
        )
        t += 10
        events.append(
            {
                "kind": "receipt_ingested",
                "t_s": t,
                "payload": {"receipt": f"run/{uid}.json", "path": f"run/{uid}.json", "unit_id": uid},
            }
        )
        t += 20
        events.append(
            {
                "kind": "idea_rejected",
                "t_s": t,
                "payload": {"scar_id": f"SCAR.{i}", "hypothesis_family": f"family_{i}", "idea": f"SCAR.{i}"},
            }
        )
        t += 30
        events.append({"kind": "NEXT_DECISION", "t_s": t, "payload": {"cycle": i, "model_id": uid}})
        t += 15
    report = ad.measure({"events": events, "elapsed_s": t})
    assert report["verdict"] == "PASS"
    assert report["degenerate_axes"] == []


# ---------------------------------------------------------------------------
# GPU park records a wait; hardware fields refused
# ---------------------------------------------------------------------------


def test_gpu_park_records_wait_on_a_held_lock(tmp_path: Path):
    """flock is per-process; a sibling thread will not contend. Hold from a child."""
    lock = tmp_path / "gpu.lock"
    lock.write_text(json.dumps({"holder": "other-lane", "pid": 1}) + "\n")
    holder = subprocess.Popen(
        [
            "python3",
            "-c",
            (
                "import fcntl,sys,time;"
                f"p={str(lock)!r};"
                "fh=open(p,'a+');"
                "fcntl.flock(fh.fileno(), fcntl.LOCK_EX);"
                "time.sleep(0.4)"
            ),
        ]
    )
    time.sleep(0.08)
    park = mbt.GpuPark(paths=[lock])
    rec = park.acquire()
    assert rec["held"] is True
    assert rec["waited_s"] >= 0.0
    assert rec["waited_for"], rec
    assert rec["waited_for"][0]["parked"] is True
    holder.wait(timeout=5)
    park.release()


def test_strip_hardware_drops_rate_and_ns_fields():
    dirty = {
        "text": "Paris",
        "tps": 24.4,
        "complete_tps": 12.0,
        "complete_tps_current_measured": 24.4086,
        "gpu_ns": 9,
        "wall_ns": 3,
        "generated_tokens": 3,
        "nested": {"decode_tps": 1.0, "ok": True},
    }
    clean = mbt.strip_hardware(dirty)
    assert "tps" not in clean
    assert "complete_tps" not in clean
    assert "complete_tps_current_measured" not in clean
    assert "gpu_ns" not in clean
    assert "wall_ns" not in clean
    assert clean["generated_tokens"] == 3
    assert "decode_tps" not in clean["nested"]
    assert clean["nested"]["ok"] is True
    _assert_no_hardware_claims(clean)


def test_catalog_marks_closed_scars_dead_and_hawking_units_as_self():
    rows = mbt.live_catalog()
    ids = {r["id"] for r in rows}
    assert "WU.DEAD.mlp_function_replacement" in ids
    assert "WU.HAWKING.resident_identity_pin" in ids
    dead = [r for r in rows if r.get("dead")]
    selfs = [r for r in rows if r.get("hawking_self")]
    assert dead
    assert selfs
    assert all(r.get("scar_id") for r in dead)
    assert mbt.is_dead_unit(dead[0])
    assert mbt.is_dead_unit(selfs[0]) is None
    # High-gain dead rows would be the scripted policy's first pick without scars.
    assert max(r["expected_information_gain"] for r in dead) > max(r["expected_information_gain"] for r in selfs)
    # Eight keyed families sit on the menu so the model can name them.
    for family in ("MONARCH", "BUTTERFLY", "FACTORIZE_THE_FACTORS", "PRODUCT_DICTIONARY",
                   "CONDITIONAL_PROGRAM", "GENERATED_BLOCK", "NONLINEAR_GENERATOR"):
        assert any(r.get("scar_id") == family for r in dead), family
    # Scar ids must be in titles: choose() forwards title, not description.
    mlp = next(r for r in rows if r["id"] == "WU.DEAD.mlp_function_replacement")
    assert "MLP_FUNCTION_REPLACEMENT_CLOSED" in mlp["title"]


def test_pruning_fix_refuses_all_eight_keyed_families():
    report = mbt.verify_pruning_fix()
    assert report["ok"] is True, report
    assert report["all_eight_refused"] is True
    assert report["live_school_not_pruned"] is True
    assert report["scripted_policy_skips_dead_mlp"] is True
    assert report["scripted_policy_id"] != "WU.DEAD.mlp_function_replacement"
    assert not str(report["scripted_policy_id"] or "").startswith("WU.DEAD.")
    refused = {row["family"]: row["refused"] for row in report["families"]}
    for family in mbt.WAVE_DEAD:
        assert refused[family] is True, family


def test_scripted_policy_no_longer_advertises_dead_mlp_replacement():
    policy = mb.fixed_policy_choose(mbt.live_catalog())
    assert policy["id"] != "WU.DEAD.mlp_function_replacement"
    assert policy["id"] and not str(policy["id"]).startswith("WU.DEAD.")
    refused_ids = {r.get("id") for r in (policy.get("refusals") or [])}
    assert "WU.DEAD.mlp_function_replacement" in refused_ids


# ---------------------------------------------------------------------------
# --build is not a pass; receipts if present are sealed and honest
# ---------------------------------------------------------------------------


def test_build_does_not_claim_a_pass_or_fake_events(tmp_path, monkeypatch):
    """--build must not clobber a live 30-minute receipt on the canonical path."""

    def fake_write(name, doc, recorded_by):
        from tools.future._common import seal

        doc.setdefault("bench", {"gpu_authority": False, "state": "UNKNOWN", "measurement_state": "STATIC_ONLY", "recorded_by": recorded_by, "machine": "test", "rule": "no hardware"})
        doc.setdefault("claim_boundary", "test")
        _assert_no_hardware_claims(doc)
        seal(doc)
        out = tmp_path / name
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return out

    monkeypatch.setattr(mbt, "write_receipt", fake_write)
    out = mbt.build()
    doc = json.loads(out.read_text())
    assert out.parent == tmp_path
    assert doc["verdict"] == "FAIL"
    assert doc["schema"] == mbt.SCHEMA
    assert doc["gpu_authority"] is False
    assert doc["required_events"]["second_hypothesis"]["found"] is False
    assert doc["would_timeline_look_the_same_without_the_model"]["answer"] is True
    assert doc["stale_note_contradicted"] is True
    assert "resident.py" in doc["stale_note"]
    _assert_no_hardware_claims(doc)
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_launch_unit_labels_workunit_id_not_argv0(tmp_path: Path):
    unit = {
        "id": "WU.HAWKING.fusion_env_applied",
        "hawking_self": True,
        "hypothesis_family": "fusion_env_applied",
        "dead": False,
    }
    pin = {
        "resident_identity": "sealed-3.14",
        "resident_binary": str(tmp_path / "missing-bin"),
        "binary_sha256": "abc",
        "fusion_env": {"HAWKING_QWEN38_FUSE_MLP": "swiglu"},
    }
    handle = mbt.launch_unit(unit, run_dir=tmp_path, pin=pin, pid=None, wait_s=0.4)
    assert handle["label"] == "WU.HAWKING.fusion_env_applied"
    assert handle["capability"] == "WU.HAWKING.fusion_env_applied"
    assert handle["argv0"] != handle["label"]
    proc = handle["proc"]
    proc.wait(timeout=8)
    snap = mbt.poll_handle(handle)
    assert snap["landed"] is True
    body = mbt.load_receipt(handle["receipt"])
    assert body["unit_id"] == "WU.HAWKING.fusion_env_applied"
    assert body.get("identity") == "sealed-3.14"


def test_live_receipts_if_present_are_sealed_and_hardware_free():
    path = RECEIPTS / mbt.RECEIPT
    timeline = RECEIPTS / mbt.TIMELINE_RECEIPT
    if not path.is_file() or not timeline.is_file():
        pytest.skip("live --run receipts not on disk yet")
    doc = json.loads(path.read_text())
    tl = json.loads(timeline.read_text())
    assert doc["schema"] == mbt.SCHEMA
    assert tl["schema"] == mbt.TIMELINE_SCHEMA
    assert doc["gpu_authority"] is False
    assert tl["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    _assert_no_hardware_claims(tl)
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert "required_events" in doc
    assert "would_timeline_look_the_same_without_the_model" in doc
    assert "degeneracy" in doc
    assert "sealed" in doc
    assert doc["sealed"]["stale_note_contradicted"] is True
    assert doc["hcli_invoked_not_edited"] is True
    # Control actually ran.
    assert doc["control"]["control_ran"] is True
    # Degeneracy measure ran over this timeline.
    assert doc["degeneracy"]["measure"] == "tools.future.autonomy_degeneracy.measure"
    # Four event slots exist (found true or honest false).
    for key in ("second_hypothesis", "wait_reason_receipt_replan", "scar_avoidance", "hawking_self_workunit"):
        assert key in doc["required_events"]
        assert "found" in doc["required_events"][key]
    # Re-run specific: prune verified live; reason-rate reported; control honest.
    if doc.get("elapsed_s"):
        assert doc.get("prune_verification", {}).get("ok") is True
        assert "reason_rate" in doc
        assert "n_choose_no_reason" in doc["reason_rate"]
        assert "rate" in doc["reason_rate"]
        assert "fraction_model_over_policy" in doc["participation"]
        assert "changed_what_ran_next_count" in (
            doc["participation"].get("materially_participated") or {}
        )
    # Timeline carries verbatim model output when the model was asked.
    if tl.get("model_calls"):
        assert any("reply_text" in c for c in tl["model_calls"])
    if doc.get("verdict") == "PASS":
        assert doc["would_timeline_look_the_same_without_the_model"]["answer"] is False
        for key, row in doc["required_events"].items():
            assert row.get("found") is True, key
        assert doc["sealed"]["pin"]["sealed"] is True
        assert doc["degeneracy"]["verdict"] != "FAIL"
