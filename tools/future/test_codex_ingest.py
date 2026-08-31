"""Tests for the Codex receipt ingest sidecar.

Includes a negative control the classifier must refuse (PROTECTED_REJECT is
SCAR, never LAW) and a negative control the idempotence guard must be capable
of failing (a perturbed cursor makes --assert-idempotent exit non-zero).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.future import codex_ingest as ci
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


NOW = "2026-08-29T18:00:00Z"


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _classify_doc(obj: dict) -> ci.Classification:
    return ci.classify_bytes(json.dumps(obj).encode())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_protected_reject_is_scar_not_law():
    """NEGATIVE CONTROL: a PROTECTED_REJECT receipt must never classify as LAW."""
    cls = _classify_doc({"schema": "test.v1", "status": "PROTECTED_REJECT", "pass": True})
    assert cls.label == "SCAR"
    assert cls.label != "LAW"
    assert cls.token == "PROTECTED_REJECT"
    assert cls.field == "status"
    assert cls.confidence >= 0.9


def test_protected_reject_token_in_body_is_scar():
    cls = ci.classify_bytes(b'{"note": "gate returned PROTECTED_REJECT on this candidate"}')
    assert cls.label == "SCAR"
    assert "PROTECTED_REJECT" in cls.token


def test_diagnostic_reject_is_scar():
    cls = _classify_doc({"status": "DIAGNOSTIC_REJECT"})
    assert cls.label == "SCAR"
    assert cls.token == "DIAGNOSTIC_REJECT"


def test_protected_pass_is_law():
    cls = _classify_doc({"status": "PROTECTED_PASS"})
    assert cls.label == "LAW"
    assert cls.token == "PROTECTED_PASS"


def test_diagnostic_pass_is_law():
    cls = _classify_doc({"headline": "DIAGNOSTIC_PASS under a contaminated window"})
    assert cls.label == "LAW"
    assert cls.token == "DIAGNOSTIC_PASS"


def test_status_passed_is_law():
    cls = _classify_doc({"status": "PASSED", "result": {"hypothesis": "x"}})
    assert cls.label == "LAW"
    assert cls.token == "PASSED"


def test_status_verified_is_law():
    cls = _classify_doc({"status": "VERIFIED"})
    assert cls.label == "LAW"


def test_not_for_promotion_plus_passed_is_scar():
    """A passing A/B that is NOT_FOR_PROMOTION must not become a law."""
    cls = _classify_doc({"NOT_FOR_PROMOTION": True, "status": "PASSED", "pass": True})
    assert cls.label == "SCAR"
    assert cls.field == "NOT_FOR_PROMOTION"


def test_refuted_verdict_is_scar_even_if_pass_true():
    cls = _classify_doc({"pass": True, "result": {"verdict": "REFUTED", "hypothesis": "simdgroup"}})
    assert cls.label == "SCAR"
    assert cls.token == "REFUTED"
    assert cls.field == "result.verdict"


def test_pass_false_is_scar():
    cls = _classify_doc({"schema": "hawking.headless.x.v1", "pass": False})
    assert cls.label == "SCAR"
    assert cls.field == "pass"


def test_blocked_status_is_scar():
    cls = _classify_doc({"status": "BLOCKED", "blockers": ["x"]})
    assert cls.label == "SCAR"
    assert cls.token == "BLOCKED"


def test_failed_gate_is_scar():
    cls = _classify_doc({"gate": "REJECTED"})
    assert cls.label == "SCAR"


def test_negative_science_block_is_scar():
    cls = _classify_doc({"negative_science": {"id": "QN-BINARY-INJURY", "level": "MODEL_SPECIFIC"}})
    assert cls.label == "SCAR"
    assert cls.field == "negative_science"


def test_census_without_verdict_is_neutral():
    cls = _classify_doc(
        {
            "schema": "hawking.accelerator.corpus_census.v1",
            "corpus_size": 104,
            "entries": [{"id": "x"}],
        }
    )
    assert cls.label == "NEUTRAL"


def test_census_with_pass_true_is_still_neutral():
    """A census that records pass:true is 'the census ran', not a physical law."""
    cls = _classify_doc(
        {
            "schema": "hawking.accelerator.receipt.v1",
            "THE_CENSUS": {"n": 3},
            "pass": True,
        }
    )
    assert cls.label == "NEUTRAL"


def test_plan_status_is_neutral():
    cls = _classify_doc({"status": "SCAFFOLD_ONLY", "schema": "hcli.agentos.flash_next.v1"})
    assert cls.label == "NEUTRAL"


def test_classification_is_pure_function_of_bytes():
    raw = json.dumps({"status": "PROTECTED_PASS"}).encode()
    a = ci.classify_bytes(raw)
    b = ci.classify_bytes(raw)
    assert a == b
    # Path must not matter: same bytes, same class.
    assert ci.classify_bytes(raw).label == "LAW"


def test_binary_artifact_is_neutral():
    cls = ci.classify_bytes(b"\x00\x01\xffgzip")
    assert cls.label == "NEUTRAL"


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------


def test_law_delta_names_every_consumer_and_nulls_hardware():
    raw = json.dumps(
        {
            "status": "PROTECTED_PASS",
            "knowledge_level": "INSTANCE",
            "experiment_class": "ACCEL-KERNEL",
            "result": {"hypothesis": "fused mlp wins on this organ"},
            "tps": 33.7,
            "token_ns": 123456,
        }
    ).encode()
    cls = ci.classify_bytes(raw)
    delta = ci.emit_delta("receipts/headless/X.json", "ab" * 32, raw, cls)
    assert delta is not None
    assert delta["classification"] == "LAW"
    for key in (
        "odyssey_ii_law_candidate",
        "odyssey_iii_attack_target",
        "architecture_atlas_behaviour_reference",
        "physical_graph_candidate_semantic",
        "learned_physical_compiler_row",
        "hwir_projection",
    ):
        assert key in delta
    assert delta["odyssey_ii_law_candidate"]["proposed_scope"] == ci.SCOPE_MODEL_LOCAL
    assert delta["odyssey_ii_law_candidate"]["sidecar_promotion_authority"] is False
    measured = delta["learned_physical_compiler_row"]["measured"]
    assert measured["tps"] is None
    assert measured["token_ns"] is None
    assert measured["gpu_ns"] is None
    assert delta["learned_physical_compiler_row"]["bench_state"] == "UNKNOWN"


def test_scar_delta_kills_hypotheses():
    raw = json.dumps(
        {
            "status": "PROTECTED_REJECT",
            "result": {"hypothesis": "binary healing restores generation"},
            "reopen_condition": "a coherent body faster than q2f",
        }
    ).encode()
    cls = ci.classify_bytes(raw)
    delta = ci.emit_delta("receipts/headless/Y.json", "cd" * 32, raw, cls)
    assert delta is not None
    assert delta["classification"] == "SCAR"
    inv = delta["invalidation"]
    assert inv["kills"]
    assert inv["makes_redundant"]
    assert inv["level"] == "MODEL_SPECIFIC"
    assert inv["sidecar_must_not_promote"] is True
    assert inv["reopen_condition"] == "a coherent body faster than q2f"


def test_neutral_emits_no_delta():
    raw = json.dumps({"schema": "hawking.headless.census.v1", "entries": []}).encode()
    cls = ci.classify_bytes(raw)
    assert cls.label == "NEUTRAL"
    assert ci.emit_delta("receipts/headless/Z.json", "ee" * 32, raw, cls) is None


def test_law_delta_does_not_trip_hardware_claim_guard(tmp_path, monkeypatch):
    """Source receipts may contain tps; our delta must not copy them as numbers."""
    raw = json.dumps({"status": "PROTECTED_PASS", "tps": 12.5, "gpu_ns": 99}).encode()
    cls = ci.classify_bytes(raw)
    delta = ci.emit_delta("receipts/headless/HW.json", "ff" * 32, raw, cls)
    # write_receipt must accept the delta document.
    monkeypatch.setattr("tools.future._common.RECEIPTS", tmp_path)
    out = write_receipt("_delta_hw_guard.json", {"schema": "test", "version": 1, "delta": delta}, "test")
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["delta"]["learned_physical_compiler_row"]["measured"]["tps"] is None


# ---------------------------------------------------------------------------
# Cursor / ingest
# ---------------------------------------------------------------------------


def test_ingest_first_scan_classifies_and_second_is_empty(tmp_path):
    root = tmp_path / "headless"
    _dump(root / "law.json", {"status": "PROTECTED_PASS", "result": {"hypothesis": "h"}})
    _dump(root / "scar.json", {"status": "PROTECTED_REJECT"})
    _dump(root / "census.json", {"schema": "hawking.headless.census.v1", "entries": [1]})

    first = ci.ingest(root=root, previous={}, now=NOW)
    assert first["scan"]["n_new"] == 3
    assert first["scan"]["n_changed"] == 0
    assert first["scan"]["n_deltas"] == 2  # law + scar, not census
    assert first["scan"]["n_active_deltas"] == 2
    labels = {Path(r["relpath"]).name: r["label"] for r in first["classified_this_scan"]}
    assert labels["law.json"] == "LAW"
    assert labels["scar.json"] == "SCAR"
    assert labels["census.json"] == "NEUTRAL"
    assert first["schema"] == ci.SCHEMA
    assert first["vocabulary"]["no_era_vi"] is True
    assert first["vocabulary"]["no_odyssey_iv"] is True

    second = ci.ingest(root=root, previous=first, now="2026-08-29T18:01:00Z")
    assert second["scan"]["n_new"] == 0
    assert second["scan"]["n_changed"] == 0
    assert second["scan"]["n_deltas"] == 0
    assert second["classified_this_scan"] == []
    assert second["deltas_this_scan"] == []
    assert second["scan"]["n_active_deltas"] == 2
    assert {d["source"] for d in second["active_deltas"]} == {
        d["source"] for d in first["active_deltas"]
    }
    # first_seen is stable; last_classified is not bumped on a no-op
    for key, rec in first["cursor"].items():
        assert second["cursor"][key]["sha256"] == rec["sha256"]
        assert second["cursor"][key]["first_seen"] == rec["first_seen"]
        assert second["cursor"][key]["last_classified"] == rec["last_classified"]


def test_mtime_alone_is_not_a_change(tmp_path):
    root = tmp_path / "headless"
    target = root / "stable.json"
    _dump(target, {"status": "VERIFIED"})
    first = ci.ingest(root=root, previous={}, now=NOW)
    os.utime(target, (0, 0))
    second = ci.ingest(root=root, previous=first, now="2026-08-29T19:00:00Z")
    assert second["scan"]["n_new"] == 0
    assert second["scan"]["n_changed"] == 0
    assert second["scan"]["n_deltas"] == 0
    # mtime is recorded, just not used as the detector
    key = next(iter(second["cursor"]))
    assert second["cursor"][key]["mtime"] == 0.0


def test_content_change_is_detected_by_hash(tmp_path):
    root = tmp_path / "headless"
    target = root / "flip.json"
    _dump(target, {"status": "PROTECTED_PASS"})
    first = ci.ingest(root=root, previous={}, now=NOW)
    _dump(target, {"status": "PROTECTED_REJECT"})
    second = ci.ingest(root=root, previous=first, now="2026-08-29T20:00:00Z")
    assert second["scan"]["n_new"] == 0
    assert second["scan"]["n_changed"] == 1
    assert second["scan"]["n_deltas"] == 1
    assert second["classified_this_scan"][0]["label"] == "SCAR"
    assert second["deltas_this_scan"][0]["classification"] == "SCAR"
    key = next(iter(second["cursor"]))
    assert second["cursor"][key]["first_seen"] == NOW
    assert second["cursor"][key]["last_classified"] == "2026-08-29T20:00:00Z"


def test_missing_file_is_not_new(tmp_path):
    root = tmp_path / "headless"
    _dump(root / "keep.json", {"status": "VERIFIED"})
    _dump(root / "gone.json", {"status": "VERIFIED"})
    first = ci.ingest(root=root, previous={}, now=NOW)
    (root / "gone.json").unlink()
    second = ci.ingest(root=root, previous=first, now="2026-08-29T21:00:00Z")
    assert second["scan"]["n_new"] == 0
    assert second["scan"]["n_changed"] == 0
    assert second["scan"]["n_missing_from_disk"] == 1
    gone_key = [k for k in second["cursor"] if k.endswith("gone.json")][0]
    assert second["cursor"][gone_key].get("present") is False


def test_files_are_opened_read_only(tmp_path, monkeypatch):
    root = tmp_path / "headless"
    _dump(root / "x.json", {"status": "VERIFIED"})
    opens = []
    real_open = os.open

    def spy_open(path, flags, *args, **kwargs):
        if str(path).endswith("x.json"):
            opens.append(flags)
            # O_RDONLY is 0 on POSIX; O_ACCMODE is the actual access-mode mask.
            assert (flags & os.O_ACCMODE) == os.O_RDONLY
            assert not (flags & os.O_WRONLY)
            assert not (flags & os.O_RDWR)
            assert not (flags & os.O_CREAT)
            assert not (flags & os.O_TRUNC)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)
    ci.ingest(root=root, previous={}, now=NOW)
    assert opens, "expected the artifact to be opened"


# ---------------------------------------------------------------------------
# Idempotence guard — the refusal must be able to fire
# ---------------------------------------------------------------------------


def test_assert_idempotent_refusal_fires_on_perturbed_cursor(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: a guard nobody has watched fail is not a guard.

    After a clean --once, --assert-idempotent must exit 0. After the cursor is
    artificially perturbed, the same flag must exit non-zero.
    """
    headless = tmp_path / "headless"
    _dump(headless / "scar.json", {"status": "PROTECTED_REJECT", "pass": True})
    receipt_name = "_TEST_CODEX_INGEST_STATE.json"
    state_path = RECEIPTS / receipt_name
    monkeypatch.setattr(ci, "HEADLESS", headless)
    monkeypatch.setattr(ci, "RECEIPT", receipt_name)
    try:
        assert ci.main(["--once"]) == 0
        assert state_path.is_file()
        doc = json.loads(state_path.read_text())
        assert doc["scan"]["n_new"] == 1
        assert doc["classified_this_scan"][0]["label"] == "SCAR"

        # Unchanged directory: the guard is quiet.
        assert ci.main(["--once", "--assert-idempotent"]) == 0
        quiet = json.loads(state_path.read_text())
        assert quiet["scan"]["n_new"] == 0
        assert quiet["scan"]["n_changed"] == 0

        # Perturb the cursor (lie about the hash). The guard must fire.
        lied = json.loads(state_path.read_text())
        key = next(iter(lied["cursor"]))
        lied["cursor"][key]["sha256"] = "00" * 32
        state_path.write_text(json.dumps(lied, indent=1, sort_keys=True) + "\n")
        rc = ci.main(["--once", "--assert-idempotent"])
        assert rc == 1, "idempotence assertion must be capable of failing"
    finally:
        if state_path.exists():
            state_path.unlink()


def test_assert_idempotent_also_fires_if_cursor_entry_deleted(tmp_path, monkeypatch):
    headless = tmp_path / "headless"
    _dump(headless / "law.json", {"status": "VERIFIED"})
    receipt_name = "_TEST_CODEX_INGEST_STATE_DELETED.json"
    state_path = RECEIPTS / receipt_name
    monkeypatch.setattr(ci, "HEADLESS", headless)
    monkeypatch.setattr(ci, "RECEIPT", receipt_name)
    try:
        assert ci.main(["--once"]) == 0
        doc = json.loads(state_path.read_text())
        doc["cursor"] = {}
        state_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        assert ci.main(["--once", "--assert-idempotent"]) == 1
    finally:
        if state_path.exists():
            state_path.unlink()


# ---------------------------------------------------------------------------
# build() / selftest() against the live corpus
# ---------------------------------------------------------------------------


def test_build_and_selftest_emit_sealed_receipt():
    out = ci.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == ci.RECEIPT
    assert doc["schema"] == ci.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["cursor"]
    assert "recovered_implementation" in doc
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["scan"]["n_on_disk"] >= 1
    # Hardware-claim guard: a number in a hardware field must not sneak in.
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_must_not_exist.json",
            {"schema": "test", "tps": 1.0},
            "test",
        )
    leaked = RECEIPTS / "_must_not_exist.json"
    if leaked.exists():
        leaked.unlink()
