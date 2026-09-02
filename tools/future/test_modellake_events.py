"""MODEL_SEALED must emit work, and a claim is not a seal.

Load-bearing failures:
  - a missing watcher log reads as 'nothing is downloading'
  - the same seal twice emits once
  - units are built by frontiers._item, so a dead school is refused
  - COMPLETE_UNSEALED does not emit, even if the log claims it is done
"""
from __future__ import annotations

import json

import pytest

from tools.future import frontiers as fr
from tools.future import modellake_events as me
from tools.future import modellake_scheduler_view as mv
from tools.future import specimen_registry as sr
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def _install_tmp_lake(tmp_path, monkeypatch, *, sealed=(), unsealed=()):
    """Point the registry at a synthesized lake. Never writes the live volume."""
    lake = tmp_path / "lake"
    specimens = lake / "specimens"
    manifests = lake / "manifests"
    partial = lake / "partial"
    specimens.mkdir(parents=True)
    manifests.mkdir()
    partial.mkdir()
    for sid in sealed:
        body = specimens / sid
        body.mkdir()
        (body / "config.json").write_text(json.dumps({"model_type": "synth"}))
        (body / "weights.bin").write_bytes(b"w")
        (manifests / f"{sid}.json").write_text(
            json.dumps({"resolved_sha": "abc123", "bytes": 1})
        )
    for sid in unsealed:
        body = specimens / sid
        body.mkdir()
        (body / "config.json").write_text(json.dumps({"model_type": "synth"}))
        (body / "weights.bin").write_bytes(b"w")
    monkeypatch.setattr(sr, "LAKE", lake)
    monkeypatch.setattr(sr, "SPECIMENS", specimens)
    monkeypatch.setattr(sr, "MANIFESTS", manifests)
    monkeypatch.setattr(sr, "PARTIAL", partial)


def _skip_if_lake_unmounted():
    if not sr.LAKE.is_dir():
        pytest.skip(f"{sr.LAKE} is not attached")


def test_the_six_triggers_are_exactly_s027_and_the_view():
    assert me.SEAL_TRIGGERS == mv.SEAL_TRIGGERS
    assert len(me.SEAL_TRIGGERS) == 6
    assert me.SEAL_TRIGGERS[0] == "fingerprint"
    assert me.SEAL_TRIGGERS[-1] == "possible prefetch or load"


def test_a_missing_watcher_log_refuses_rather_than_reporting_nothing_downloading(
    monkeypatch,
):
    monkeypatch.setattr(me, "WATCH_LOG", me.REPO / "no" / "such.jsonl")
    with pytest.raises(me.LakeEventRefused, match="watcher is not running"):
        me.consume()
    with pytest.raises(me.LakeEventRefused, match="nothing is downloading"):
        me._require_watch_log()


def test_an_empty_watcher_log_refuses(tmp_path, monkeypatch):
    p = tmp_path / "watch.jsonl"
    p.write_text("")
    monkeypatch.setattr(me, "WATCH_LOG", p)
    with pytest.raises(me.LakeEventRefused, match="no parseable event"):
        me._tail()


def test_the_live_registry_partitions_sealed_from_unsealed(tmp_path, monkeypatch):
    """Counts drift as the watcher completes downloads (33 unsealed, not the
    29 an earlier snapshot saw) - assert the partition invariant, not a
    number the live lake will outdate on its own."""
    _install_tmp_lake(
        tmp_path, monkeypatch,
        sealed=("synth--sealed@aaa",),
        unsealed=("synth--unsealed@bbb",),
    )
    b = sr.seal_backlog()
    sealed = me.sealed_from_disk()
    reg = sr.registry()
    assert b["n_sealed"] > 0
    assert b["n_complete_unsealed"] + b["n_sealed"] <= len(reg)
    assert len(sealed) == b["n_sealed"]
    reg_sealed = {
        r["id"] for r in reg
        if r["lifecycle"] in ("SEALED_SOURCE", "FINGERPRINTED")
    }
    assert {r["id"] for r in sealed} == reg_sealed
    assert reg_sealed.isdisjoint(set(b["ids"]))


def test_detection_reads_a_manifest_file_not_a_claim():
    _skip_if_lake_unmounted()
    sealed = me.sealed_from_disk()
    assert sealed
    for row in sealed:
        path = sr.MANIFESTS / f"{row['id']}.json"
        assert path.is_file(), row["id"]
        man = json.loads(path.read_text())
        assert man.get("resolved_sha"), row["id"]
        assert man.get("bytes"), row["id"]
        assert me.manifest_is_seal(row["id"]) is True


def test_a_watcher_cache_manifest_without_bytes_is_not_a_seal(tmp_path, monkeypatch):
    """A file in manifests/ is a watcher cache, not a seal, unless it carries
    both resolved_sha and bytes."""
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    sid = "synth--cache-only@aaa"
    sealed_sid = "synth--sealed@bbb"
    (manifests / f"{sid}.json").write_text(json.dumps({
        "resolved_sha": "abc",
        "expected": 4,
        "files": ["a"],
        "sizes": {},
    }))
    (manifests / f"{sealed_sid}.json").write_text(json.dumps({
        "resolved_sha": "def",
        "bytes": 4096,
    }))
    monkeypatch.setattr(sr, "MANIFESTS", manifests)
    man_path = sr.MANIFESTS / f"{sid}.json"
    assert man_path.is_file(), "the trap: a file exists and is still not a seal"
    man = json.loads(man_path.read_text())
    assert not man.get("bytes")
    assert me.manifest_is_seal(sid) is False
    assert me.manifest_is_seal(sealed_sid) is True


def test_complete_unsealed_does_not_emit(tmp_path, monkeypatch):
    _install_tmp_lake(
        tmp_path, monkeypatch,
        sealed=("synth--sealed@aaa",),
        unsealed=("synth--unsealed@bbb",),
    )
    b = sr.seal_backlog()
    assert b["n_complete_unsealed"] == len(b["ids"])
    for sid in b["ids"]:
        assert me.manifest_is_seal(sid) is False
    sealed_ids = {r["id"] for r in me.sealed_from_disk()}
    assert sealed_ids.isdisjoint(set(b["ids"]))


def test_a_log_claim_without_a_manifest_is_not_a_seal(tmp_path, monkeypatch):
    unsealed = "synth--unsealed@deadbeef"
    _install_tmp_lake(tmp_path, monkeypatch, unsealed=(unsealed,))
    p = tmp_path / "watch.jsonl"
    p.write_text(
        json.dumps({"event": "already_complete", "job": unsealed, "ts": "now"}) + "\n"
        + json.dumps({"event": "download_exit", "job": unsealed, "returncode": 0}) + "\n"
    )
    monkeypatch.setattr(me, "WATCH_LOG", p)
    monkeypatch.setattr(me.lc, "per_specimen", lambda: [])
    claims = me.log_completion_claims()
    assert unsealed in claims
    emitted_ids = {row["specimen_id"] for row in me.consume(ledger=set())}
    assert unsealed not in emitted_ids
    assert unsealed in me.claims_without_seal()


def test_the_same_seal_twice_emits_once():
    _skip_if_lake_unmounted()
    sealed = me.sealed_from_disk()
    assert sealed
    ledger: set[str] = set()
    first = me.emit_for_seal(sealed[0], ledger)
    second = me.emit_for_seal(sealed[0], ledger)
    assert len(first) == 6
    assert [u["hypothesis_family"] for u in first] == [
        me.TRIGGER_FAMILY[t] for t in me.SEAL_TRIGGERS
    ]
    assert second == []
    assert sealed[0]["id"] in ledger


def test_consume_is_idempotent_across_the_whole_sealed_set(tmp_path, monkeypatch):
    _skip_if_lake_unmounted()
    p = tmp_path / "watch.jsonl"
    p.write_text(json.dumps({"event": "watcher_sample", "active_jobs": []}) + "\n")
    monkeypatch.setattr(me, "WATCH_LOG", p)
    n_sealed_now = len(me.sealed_from_disk())
    ledger: set[str] = set()
    first = me.consume(ledger)
    second = me.consume(ledger)
    assert len(first) == n_sealed_now > 0
    assert all(row["n_units"] == len(me.SEAL_TRIGGERS) for row in first)
    assert sum(row["n_units"] for row in first) == n_sealed_now * len(me.SEAL_TRIGGERS)
    assert second == []
    assert len(ledger) == n_sealed_now


def test_units_are_created_through_frontiers_item_not_hand_built(monkeypatch):
    _skip_if_lake_unmounted()
    calls: list[str] = []
    orig = me.fr._item

    def wrapped(**kwargs):
        calls.append(kwargs["id"])
        return orig(**kwargs)

    monkeypatch.setattr(me.fr, "_item", wrapped)
    sealed = me.sealed_from_disk()[0]
    ledger: set[str] = set()
    units = me.emit_for_seal(sealed, ledger)
    assert len(units) == 6
    assert calls == [u["id"] for u in units]
    assert all(u["species"] == me.SEAL_SPECIES for u in units)
    assert all(u["kind"] == "NEXT_WORK" for u in units)
    assert all(u.get("redundancy_key") for u in units)
    assert all("Static sidecar artifact" in (u.get("claim_boundary") or "") for u in units)
    n = len(calls)
    assert me.emit_for_seal(sealed, ledger) == []
    assert len(calls) == n


def test_a_dead_school_unit_is_refused_rather_than_created():
    _skip_if_lake_unmounted()
    sealed = me.sealed_from_disk()[0]
    with pytest.raises(fr.DeadSchoolRefused, match="zero overlapable"):
        me.emit_trigger(
            sealed,
            "fingerprint",
            title="reorder the dispatches",
            detail="permute the dispatch order of the token graph",
        )


def test_a_missing_specimen_id_refuses_rather_than_emitting():
    with pytest.raises(me.LakeEventRefused, match="missing"):
        me.emit_for_seal({"lifecycle": "FINGERPRINTED"}, ledger=set())


def test_emit_for_seal_refuses_an_unsealed_specimen(tmp_path, monkeypatch):
    sid = "synth--unsealed@deadbeef"
    _install_tmp_lake(tmp_path, monkeypatch, unsealed=(sid,))
    row = next(r for r in sr.registry() if r["id"] == sid)
    with pytest.raises(me.LakeEventRefused, match="not a seal"):
        me.emit_for_seal(row, ledger=set())


def test_prefetch_and_economics_cite_the_measured_rate_not_a_guess():
    _skip_if_lake_unmounted()
    sealed = me.sealed_from_disk()
    ledger: set[str] = set()
    units = {u["hypothesis_family"]: u for u in me.emit_for_seal(sealed[0], ledger)}
    eco = units["modellake_seal_economics"]
    pre = units["modellake_seal_prefetch"]
    assert "cold load is" in eco["detail"]
    assert "guess" not in eco["detail"]
    assert "S027 §8" in pre["detail"]
    assert "minutes" in pre["detail"]


def test_build_writes_a_receipt_that_parses():
    _skip_if_lake_unmounted()
    # A live count, fetched right before build() derives its own - not a
    # frozen literal the lake will drift past the moment more specimens seal.
    n_sealed_now = len(me.sealed_from_disk())
    out = me.build()
    # build() returns the doc; --build writes it. Call write via main path
    # by sealing through the public builder used by --build.
    from tools.future._common import write_receipt

    path = write_receipt(me.RECEIPT_NAME, dict(out), me.RECORDED_BY)
    doc = json.loads(path.read_text())
    assert path.parent == RECEIPTS
    assert path.name == "MODELLAKE_EVENTS.json"
    assert doc["schema"] == me.SCHEMA
    assert doc["is_this_wired"] is True
    assert doc["n_emitted_specimens"] == n_sealed_now
    assert doc["n_emitted_units"] == n_sealed_now * len(me.SEAL_TRIGGERS)
    assert doc["through_frontiers_item"] is True
    assert doc["idempotent"] is True
    assert doc["human_notified"] is False
    assert doc["no_conversational_boundary"] is True
    assert doc["detection"]["n_complete_unsealed"] == sr.seal_backlog()["n_complete_unsealed"]
    assert doc["detection"]["n_sealed_on_disk"] == n_sealed_now
    assert doc["seal_sha256"]
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)


def test_emitted_rows_carry_what_a_consumer_needs_without_re_deriving_it():
    """G101 wired a consumer that emits WorkUnits into a receipt; this is the
    audit that the receipt itself carries fingerprint/family/scar/economics
    content, not just prose instructing someone else to go compute it."""
    _skip_if_lake_unmounted()
    rows = me.consume(ledger=set())
    assert rows
    for row in rows:
        arch = row["architecture"]
        assert arch["model_type"] == row["model_type"]
        assert "hidden_size" in arch and "num_hidden_layers" in arch

        fam = row["family"]
        assert fam["model_type"] == (row["model_type"] or "UNKNOWN")
        assert row["specimen_id"] not in fam["sibling_ids"]
        assert fam["n_siblings"] == len(fam["sibling_ids"])
        assert fam["is_novel_architecture"] == (fam["n_siblings"] == 0)

        scars = row["relevant_scars"]
        assert scars["n_matches"] >= scars["n_model_specific"] >= 0
        assert len(scars["matches"]) <= me.MAX_RELEVANT_SCARS

        eco = row["economics"]
        assert eco["id"] == row["specimen_id"]
        assert eco["cold_load_seconds"] > 0
        assert eco["fits_in_uma"] is None  # never guessed here

        assert "not implied" in row["residency_decision"]
        assert "HCLI" in row["residency_decision"]


def test_role_and_scar_trigger_details_cite_the_real_counts_not_generic_prose(
    monkeypatch,
):
    """Mutation target: stub _family_context/_relevant_scars to return a
    distinctive count and confirm the WorkUnit detail text actually carries
    it through, rather than the fixed sentence the triggers used to emit."""
    _skip_if_lake_unmounted()
    sealed = me.sealed_from_disk()[0]
    model_type = (sealed.get("architecture") or {}).get("model_type") or "UNKNOWN"

    fake_family = {
        "model_type": model_type, "sibling_ids": ["sentinel-sibling@zzz"],
        "n_siblings": 1, "is_novel_architecture": False,
    }
    fake_scars = {
        "queried_model": model_type,
        "matches": [{
            "scar_id": "SENTINEL_SCAR", "verdict": "FALSIFIED", "level": "MODEL_SPECIFIC",
            "model": model_type, "models": [model_type], "organ": "unrecorded",
            "hypothesis_family": "sentinel", "failure_mechanism": "x",
            "reopen_condition": "x", "source_path": "x", "match_score": 99,
        }],
        "n_matches": 1, "n_model_specific": 1,
    }
    monkeypatch.setattr(me, "_family_context", lambda *a, **k: fake_family)
    monkeypatch.setattr(me, "_relevant_scars", lambda *a, **k: fake_scars)

    ledger: set[str] = set()
    units = {u["hypothesis_family"]: u for u in me.emit_for_seal(sealed, ledger)}
    assert "sentinel-sibling@zzz" in units["modellake_seal_role"]["detail"]
    assert "SENTINEL_SCAR" in units["modellake_seal_laws_scars"]["detail"]


def test_an_unmounted_lake_refuses_rather_than_reporting_zero_seals(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(sr, "LAKE", Path("/no/such/volume"))
    with pytest.raises(sr.RegistryRefused, match="is not attached"):
        me.sealed_from_disk()
