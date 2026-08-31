"""Tests for tools/future/specimen_events.py.

Acceptance nobody has watched fail:
  - the seal transition is a durable replayable event, not an in-memory callback
  - mission id and phase_transition are unchanged across a specimen arrival
  - a helper that restarts Odyssey is refused
  - transfer-and-scar lookup is scheduled before new search
  - a partial-weight experiment cannot be recorded as specimen science
  - already-sealed specimens are cited, not re-hashed
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import specimen_events as se
from tools.future import workgraph as wg
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_build_seals_static_only_receipt():
    out = se.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SPECIMEN_EVENTS.json"
    assert doc["schema"] == se.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    proofs = doc["proofs"]
    assert proofs["durable_replayable_event"] is True
    assert proofs["mission_id_unchanged"] is True
    assert proofs["phase_transition_unchanged"] is True
    assert proofs["launch_seal_unchanged"] is True
    assert proofs["restarts_odyssey"] is False
    assert proofs["restart_helper_refused"] is True
    assert proofs["transfer_before_new_search"] is True
    assert proofs["partial_weight_science_refused"] is True
    assert proofs["cite_did_not_rehash"] is True
    assert proofs["fingerprint_did_not_open_weights"] is True
    assert proofs["workgraph_added"] is True
    assert proofs["workgraph_bounded"] is True
    _assert_no_hardware_claims(doc)


def test_demonstration_says_replay_when_not_live():
    doc = json.loads((RECEIPTS / se.RECEIPT).read_text())
    if doc.get("live_arrival"):
        assert doc["demonstration_mode"] == "live_arrival"
        assert doc["replay"] is False
    else:
        assert doc["demonstration_mode"] == "replay"
        assert doc["replay"] is True
        assert "replay" in str(doc.get("demonstration_why") or "").lower()


def test_event_log_survives_restart_and_replays(tmp_path):
    path = tmp_path / "events.json"
    log = se.SpecimenEventLog(path)
    specimen = {"repo": "example/Model", "revision": "abc123", "tag": "example--Model@abc123"}
    a = se.make_transition_event(
        from_state=se.DOWNLOADING,
        to_state=se.COMPLETE_UNSEALED,
        specimen=specimen,
        replay=True,
        source="test",
    )
    b = se.make_transition_event(
        from_state=se.COMPLETE_UNSEALED,
        to_state=se.SEALED_SOURCE_SPECIMEN,
        specimen=specimen,
        replay=True,
        source="test",
    )
    assert log.append(a)["inserted"] is True
    assert log.append(a)["inserted"] is False  # idempotent
    assert log.append(b)["inserted"] is True
    fresh = se.SpecimenEventLog(path)
    rows = fresh.replay()
    assert [r["event_id"] for r in rows] == [a["event_id"], b["event_id"]]
    assert [r["to_state"] for r in rows] == [se.COMPLETE_UNSEALED, se.SEALED_SOURCE_SPECIMEN]
    sealed = fresh.transitions_to(se.SEALED_SOURCE_SPECIMEN)
    assert len(sealed) == 1
    assert sealed[0]["kind"] == se.KIND_TRANSITION


def test_illegal_transition_is_refused():
    specimen = {"repo": "x/y", "revision": "1"}
    with pytest.raises(se.IllegalTransition):
        se.make_transition_event(
            from_state=se.SEALED_SOURCE_SPECIMEN,
            to_state=se.DOWNLOADING,
            specimen=specimen,
            replay=True,
            source="test",
        )


def test_mission_id_and_phase_unchanged_across_arrival():
    """The test that fails if a specimen arrival resets either field."""
    mission = se.load_running_mission()
    before = (
        mission.mission_id,
        mission.phase_transition,
        mission.launch_seal_sha256,
    )
    assert mission.phase_transition == "STARTED"
    specimen = {
        "repo": "tiiuae/Falcon-H1-7B-Instruct",
        "revision": "41e72f27effbab80cd45b6e884688452253a3686",
        "tag": "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb",
        "bytes_hashed": 15182220635,
    }
    event = se.make_transition_event(
        from_state=se.COMPLETE_UNSEALED,
        to_state=se.SEALED_SOURCE_SPECIMEN,
        specimen=specimen,
        replay=True,
        source="test",
    )
    fp = se.fingerprint_from_config(
        {
            "architectures": ["FalconH1ForCausalLM"],
            "model_type": "falcon_h1",
            "hidden_size": 3072,
            "num_hidden_layers": 44,
        },
        total_size=15182220635,
    )
    result = se.apply_sealed_arrival(
        event,
        mission,
        fingerprint=fp,
        cite={"action": "CITED_EXISTING_SEAL", "rehashed": False},
        laws=[{"law_id": "LAW-TEST", "scope": "GENERIC_CANDIDATE", "organ_class": "mlp"}],
        scar_rows=[
            {"hypothesis_family": fam, "refused": True, "scar": {"scar_id": fam}}
            for fam in se.WAVE_DEAD_FAMILIES
        ],
        roles=[{"role": "small_dense_alternate_architecture_transfer", "status": "CANDIDATE"}],
    )
    after = (
        mission.mission_id,
        mission.phase_transition,
        mission.launch_seal_sha256,
    )
    assert after == before
    assert result["mission_id_unchanged"] is True
    assert result["phase_transition_unchanged"] is True
    assert result["launch_seal_unchanged"] is True
    assert result["restarts_odyssey"] is False
    untouched = se.launch_receipt_untouched(mission)
    assert untouched["mission_id_unchanged"] is True
    assert untouched["phase_transition_unchanged"] is True
    assert untouched["seal_unchanged"] is True
    assert untouched["phase_transition"] == "STARTED"
    assert result["workgraph"]["n_units"] >= 3
    # Existing first-wave units are still the launch graph; we added another.
    assert mission.existing_unit_ids
    assert result["workgraph"]["unit_ids"]
    assert set(result["workgraph"]["unit_ids"]).isdisjoint(set(mission.existing_unit_ids))


def test_rebind_identity_and_restart_helper_fail():
    mission = se.load_running_mission()
    with pytest.raises(se.MissionRestartForbidden):
        mission.rebind_identity(mission_id="odyssey-i/forged")
    with pytest.raises(se.MissionRestartForbidden):
        se.restart_odyssey()


def test_arrival_that_rewrote_identity_would_fail(monkeypatch):
    """If apply_sealed_arrival rebound the mission, this test fails."""
    mission = se.load_running_mission()
    original = se.RunningMission.add_workgraph

    def sabotage(self, snapshot):
        object.__setattr__(self, "mission_id", "odyssey-i/RESTARTED") if False else None
        self.mission_id = "odyssey-i/RESTARTED"
        self.phase_transition = "NOT_STARTED"
        return original(self, snapshot)

    monkeypatch.setattr(se.RunningMission, "add_workgraph", sabotage)
    event = se.make_transition_event(
        from_state=se.COMPLETE_UNSEALED,
        to_state=se.SEALED_SOURCE_SPECIMEN,
        specimen={"repo": "x/y", "revision": "1", "tag": "x--y@1", "bytes_hashed": 100},
        replay=True,
        source="test",
    )
    with pytest.raises(se.MissionRestartForbidden):
        se.apply_sealed_arrival(
            event,
            mission,
            fingerprint=se.fingerprint_from_config({"model_type": "x"}, total_size=100),
            cite={"action": "CITED_EXISTING_SEAL", "rehashed": False},
            laws=[],
            scar_rows=[],
            roles=[{"role": "deferred_lake_entry", "status": "CANDIDATE"}],
        )


def test_transfer_and_scars_run_before_new_search():
    fp = se.fingerprint_from_config(
        {
            "architectures": ["FalconH1ForCausalLM"],
            "model_type": "falcon_h1",
            "hidden_size": 3072,
        },
        weight_names=["model.layers.0.mamba.A_log", "model.layers.0.feed_forward.gate_proj.weight"],
        total_size=15_000_000_000,
    )
    laws = [{"law_id": "LAW-GENERIC", "scope": "GENERIC_CANDIDATE", "organ_class": "mlp"}]
    scars = [{"hypothesis_family": fam, "refused": True} for fam in se.WAVE_DEAD_FAMILIES]
    experiments = se.cheapest_first_experiments(
        fingerprint=fp, laws=laws, scars=scars, size_bytes=15_000_000_000
    )
    phases = [e["campaign_phase"] for e in experiments]
    first_search = next((i for i, p in enumerate(phases) if p == se.CAMPAIGN_NEW_SEARCH), None)
    last_transfer = max(i for i, p in enumerate(phases) if p == se.CAMPAIGN_TRANSFER)
    if first_search is not None:
        assert last_transfer < first_search
    snapshot = se.plan_bounded_workgraph(
        specimen={"repo": "tiiuae/Falcon-H1-7B-Instruct", "revision": "41e72f27effb"},
        fingerprint=fp,
        experiments=experiments,
        cite={"action": "CITED_EXISTING_SEAL", "rehashed": False},
        roles=[{"role": "small_dense_alternate_architecture_transfer", "status": "CANDIDATE"}],
    )
    assert se.transfer_runs_before_new_search(snapshot) is True
    assert snapshot["transfer_and_scar_ids"]
    search_ids = snapshot["new_search_ids"]
    by_id = {u["id"]: u for u in snapshot["units"]}
    for sid in search_ids:
        deps = set(by_id[sid]["dependencies"])
        assert deps & set(snapshot["transfer_and_scar_ids"])
    for tid in snapshot["transfer_and_scar_ids"]:
        deps = set(by_id[tid]["dependencies"])
        assert not (deps & set(search_ids))
    # WAVE_DEAD families are consulted, not emitted as new-search schools.
    for fam in se.WAVE_DEAD_FAMILIES:
        assert fam not in json.dumps([by_id[s].get("description") for s in search_ids])


def test_partial_weight_experiment_cannot_be_recorded_as_science():
    with pytest.raises(se.PartialWeightScienceError):
        se.record_specimen_science(
            specimen_state=se.DOWNLOADING,
            experiment={"kind": "gravity", "stage": "gravity", "requires_weights": True},
            sealed=False,
        )
    with pytest.raises(se.PartialWeightScienceError):
        se.record_specimen_science(
            specimen_state=se.COMPLETE_UNSEALED,
            experiment={"kind": "doctor", "stage": "doctor"},
            sealed=False,
        )
    with pytest.raises(se.PartialWeightScienceError):
        se.record_specimen_science(
            specimen_state=se.DOWNLOADING,
            experiment={"kind": "architecture_note", "requires_weights": False},
            sealed=False,
        )
    early = se.record_early_metadata(
        specimen={"repo": "x/y"},
        state=se.DOWNLOADING,
        payload={"filenames": ["config.json"]},
    )
    assert early["is_specimen_science"] is False
    assert early["kind"] == se.KIND_EARLY_METADATA
    ok = se.record_specimen_science(
        specimen_state=se.SEALED_SOURCE_SPECIMEN,
        experiment={"kind": "transfer_law", "requires_weights": False},
        sealed=True,
    )
    assert ok["is_specimen_science"] is True


def test_unsealed_event_does_not_wake_or_add():
    mission = se.load_running_mission()
    event = se.make_transition_event(
        from_state=se.DOWNLOADING,
        to_state=se.COMPLETE_UNSEALED,
        specimen={"repo": "x/y", "revision": "1"},
        replay=True,
        source="test",
    )
    with pytest.raises(se.UnsealedSourceError):
        se.apply_sealed_arrival(event, mission)
    wake = se.SchedulerWake()
    with pytest.raises(se.UnsealedSourceError):
        wake.wake(event, mission)


def test_cite_existing_seal_does_not_rehash():
    rows = se.load_verification_rows()
    assert rows, "SPECIMEN_VERIFICATION.json must have whole-tree rows"
    row = rows[0]
    cited = se.cite_existing_seal(row, expected_revision=None)
    assert cited["rehashed"] is False
    assert cited["action"] == "CITED_EXISTING_SEAL"
    assert cited["whole_tree_verified"] is True


def test_bounded_not_exhaustive():
    huge_fp = se.fingerprint_from_config({"model_type": "glm"}, total_size=se.HUGE_BYTES)
    huge_exp = se.cheapest_first_experiments(
        fingerprint=huge_fp, laws=[], scars=[], size_bytes=se.HUGE_BYTES
    )
    assert len(huge_exp) <= se.MAX_UNITS_HUGE
    assert all(e["campaign_phase"] != se.CAMPAIGN_NEW_SEARCH for e in huge_exp)
    snap = se.plan_bounded_workgraph(
        specimen={"repo": "zai-org/GLM-5.3-Flash", "revision": "04c4e9e95c5d"},
        fingerprint=huge_fp,
        experiments=huge_exp,
        cite={"action": "CITED_EXISTING_SEAL", "rehashed": False},
        roles=[{"role": "deferred_lake_entry", "status": "CANDIDATE"}],
    )
    assert snap["n_units"] <= se.MAX_UNITS_HUGE
    assert snap["n_units"] < len(wg.LANE_IDS) + 13  # not a full 13-stage explosion
    assert not snap["new_search_ids"]


def test_wave_dead_families_are_the_campaign_scars():
    assert se.WAVE_DEAD_FAMILIES == (
        "MLP_FUNCTION_REPLACEMENT",
        "MONARCH",
        "BUTTERFLY",
        "FACTORIZE_THE_FACTORS",
        "PRODUCT_DICTIONARY",
        "CONDITIONAL_PROGRAM",
        "GENERATED_BLOCK",
        "NONLINEAR_GENERATOR",
    )


def test_arrival_workgraph_uses_workgraph_lanes():
    fp = se.fingerprint_from_config({"model_type": "falcon_h1"}, total_size=1_000_000_000)
    snap = se.plan_bounded_workgraph(
        specimen={"repo": "tiiuae/Falcon-H1-7B-Instruct", "revision": "41e72f"},
        fingerprint=fp,
        experiments=se.cheapest_first_experiments(
            fingerprint=fp,
            laws=[{"law_id": "L", "scope": "GENERIC_CANDIDATE"}],
            scars=[{"hypothesis_family": "MLP_FUNCTION_REPLACEMENT", "refused": True}],
            size_bytes=1_000_000_000,
        ),
        cite={"action": "CITED_EXISTING_SEAL", "rehashed": False},
        roles=[],
    )
    for u in snap["units"]:
        assert u["resource_lane"] in wg.LANE_IDS
        assert u["gpu_authority"] is False
        assert u["evidence_class"] == "STATIC_ONLY"
