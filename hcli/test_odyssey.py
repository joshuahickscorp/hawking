"""The resident can see and drive the live Odyssey mission through HCLI.

Odyssey (tools/odyssey_ctl.py) is real and mid-flight -- O003 sealed, O006 on
disk, O010-O013 queued -- but HCLI has no odyssey verb at all. hcli/odyssey.py
is the connector. This locks down two things:

1. inspection ingests the *existing* state (no fixture, no re-run) and never
   writes anything -- proven by hashing a patient packet before and after.
2. every verb that can mutate Odyssey state, spawn a subprocess, or start a
   download refuses with PermissionError unless confirm=True, the same gate
   hcli/tool_registry.py already uses for benchmark.run/accelerator.benchmark
   -- and it refuses *before* touching the driver at all (no subprocess is
   spawned to reach the refusal).
3. the "own" verbs (ingest/add_to_eligibility/park_specimen/record_law/
   record_scar/create_transfer_probe/create_adversarial_probe) write only to
   a ledger this module owns, gate on confirm=True the same way, refuse
   *before* the ledger file is touched, refuse a specimen HCLI invented, and
   let a transfer/adversarial probe follow a law with no other ordering --
   proving the campaign's I/II/III overlap has no barrier beyond that.

Runnable two ways:

    python3 -m pytest hcli/test_odyssey.py -q
    python3 hcli/test_odyssey.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from hcli import odyssey


def test_status_ingests_real_mid_flight_state():
    """No fixture: this must reflect the sealed patient actually on disk."""
    result = odyssey.status()
    assert result["ok"] is True, result
    assert "HAWKING ODYSSEY-I" in result["stdout"]
    assert "O003" in result["stdout"]
    assert "SEALED" in result["stdout"]


# One test that shelled out to the real driver three times in a row was a 3.0s
# SERIAL critical path -- the longest single item in the suite once the hardware
# probes were dealt with, and nothing can run faster than its slowest test.
# Parametrised, the same three real invocations run on three workers at once.
@pytest.mark.parametrize("verb", [
    "queue",
    "value",
    # `economics` is 2.72s of real driver computation on its own -- the whole
    # suite's critical path, since nothing finishes before its slowest test.
    # `queue` and `value` already prove the shell-out bridge is reachable and
    # read-only; this one adds the heavy computation. Deselected by default,
    # run with `-m slow`. Drop the marker if you want it back on every run.
    pytest.param("economics", marks=pytest.mark.slow),
])
def test_queue_value_economics_are_read_only_and_succeed(verb):
    result = getattr(odyssey, verb)()
    assert result["ok"] is True, (verb, result)


def test_patient_reads_without_writing():
    """Named 'inspect', so a patient read must not perturb the file it reads."""
    before = odyssey.patient("O003")
    assert before["found"] is True, before
    digest_before = hashlib.sha256(open(before["path"], "rb").read()).hexdigest()

    odyssey.patient("O003")  # a second read must not mutate anything either

    digest_after = hashlib.sha256(open(before["path"], "rb").read()).hexdigest()
    assert digest_before == digest_after


def test_patient_reports_absence_for_unknown_oxx_without_writing():
    result = odyssey.patient("O999")
    assert result["found"] is False
    assert result["oxx"] == "O999"


def test_admit_check_is_read_only_despite_the_verb_name():
    """cmd_admit only prints a memgate/worker_gate/disk decision; it never
    writes ODYSSEY_STATE.json, so it belongs with the inspect verbs."""
    result = odyssey.admit_check("hcli-odyssey-probe", 1.0)
    assert result["exit_code"] in (0, 1), result
    assert "slug=hcli-odyssey-probe" in result["stdout"]


def test_dry_run_paths_execute_the_real_driver_and_launch_nothing():
    """dry_run=True needs no confirm because the driver itself launches
    nothing in that mode -- assert the driver's own words say so."""
    result = odyssey.harvest(dry_run=True)
    assert result["ok"] is True, result
    assert "mode=dry-run" in result["stdout"]

    result = odyssey.acquire_next(dry_run=True)
    assert result["ok"] is True, result
    assert "DRY-RUN" in result["stdout"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: odyssey.harvest(dry_run=False),
        lambda: odyssey.write_packet("O003"),
        lambda: odyssey.run(dry_run=False),
        lambda: odyssey.cycle(dry_run=False),
        lambda: odyssey.retire("O013"),
        lambda: odyssey.acquire_next(dry_run=False),
        lambda: odyssey.completions(rebuild=True),
    ],
    ids=["harvest", "write_packet", "run", "cycle", "retire", "acquire_next", "completions_rebuild"],
)
def test_mutating_verbs_refuse_without_confirm(call, monkeypatch):
    """Refusal must happen before the driver is ever invoked -- patch
    subprocess.run to explode if a mutating verb reaches it unconfirmed."""

    def _boom(*args, **kwargs):
        raise AssertionError("mutating verb spawned the driver without confirm=True")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(PermissionError):
        call()


# --------------------------------------------------------------------------
# own -- ledger verbs (ingest/eligibility/park/law/scar/probes)
# --------------------------------------------------------------------------

@pytest.fixture
def isolated_ledger(monkeypatch, tmp_path):
    """Point the ledger at a scratch file so tests never touch the real one."""
    path = tmp_path / "HCLI_LEDGER.json"
    monkeypatch.setattr(odyssey, "LEDGER_PATH", path)
    return path


def test_ingest_reflects_real_mid_flight_specimens(isolated_ledger):
    """No fixture: O003/O006 must show up exactly as they sit on disk."""
    result = odyssey.ingest()
    assert result["specimen_count"] >= 14, result["specimen_count"]
    by_oxx = {s["oxx"]: s for s in result["specimens"]}
    assert by_oxx["O003"]["on_disk"] is True, by_oxx["O003"]
    assert by_oxx["O006"]["on_disk"] is True, by_oxx["O006"]
    assert result["eligible"] == [] and result["laws"] == []


def test_own_verbs_refuse_without_confirm_before_touching_the_ledger(isolated_ledger, monkeypatch):
    """Refusal must happen before HCLI_LEDGER.json is ever written."""

    def _boom(*args, **kwargs):
        raise AssertionError("mutating verb wrote the ledger without confirm=True")

    monkeypatch.setattr(odyssey, "atomic_write_json", _boom)
    calls = [
        lambda: odyssey.add_to_eligibility("O003"),
        lambda: odyssey.park_specimen("O003", "superseded"),
        lambda: odyssey.record_law("MLP density floor ~2.25 bpw"),
        lambda: odyssey.record_scar("LAW001", "attack found a gap"),
        lambda: odyssey.create_transfer_probe("LAW001", "O006"),
        lambda: odyssey.create_adversarial_probe("LAW001", "zero the gate proj"),
    ]
    for call in calls:
        with pytest.raises(PermissionError):
            call()
    assert not isolated_ledger.exists()


def test_add_to_eligibility_refuses_a_specimen_hcli_invented(isolated_ledger):
    with pytest.raises(ValueError):
        odyssey.add_to_eligibility("O999", confirm=True)
    assert not isolated_ledger.exists()


def test_add_to_eligibility_is_idempotent_by_oxx(isolated_ledger):
    first = odyssey.add_to_eligibility("O003", note="cheap-lab", confirm=True)
    assert first["entry"]["oxx"] == "O003"
    second = odyssey.add_to_eligibility("O003", note="re-noted", confirm=True)
    assert second["entry"]["note"] == "re-noted"
    ledger = json.loads(isolated_ledger.read_text())
    assert len(ledger["eligible"]) == 1, ledger["eligible"]
    ingested = odyssey.ingest()
    assert ingested["eligible"] == ledger["eligible"]


def test_park_specimen_requires_a_reason(isolated_ledger):
    with pytest.raises(ValueError):
        odyssey.park_specimen("O003", "", confirm=True)
    assert not isolated_ledger.exists()


def test_probes_require_a_law_already_recorded(isolated_ledger):
    """No global phase barrier, but a probe still needs a real target: it
    cannot attack or transfer-test a law nobody has claimed yet."""
    with pytest.raises(ValueError):
        odyssey.create_transfer_probe("LAW001", "O006", confirm=True)
    with pytest.raises(ValueError):
        odyssey.create_adversarial_probe("LAW001", "zero the gate proj", confirm=True)
    with pytest.raises(ValueError):
        odyssey.record_scar("LAW001", "gap found", confirm=True)


def test_law_then_transfer_and_adversarial_probes_overlap_immediately(isolated_ledger):
    """As soon as a law is claimed, both II and III may act on it -- same turn,
    no waiting on each other."""
    law = odyssey.record_law("MLP density floor ~2.25 bpw", evidence="q2f", source_oxx="O003", confirm=True)
    law_id = law["entry"]["id"]

    transfer = odyssey.create_transfer_probe(law_id, "O006", confirm=True)
    assert transfer["entry"]["status"] == "PROPOSED"
    adversarial = odyssey.create_adversarial_probe(law_id, "zero the gate proj", confirm=True)
    assert adversarial["entry"]["status"] == "PROPOSED"

    scar = odyssey.record_scar(law_id, "adversarial probe found a gap", confirm=True)
    ledger = json.loads(isolated_ledger.read_text())
    assert ledger["laws"][0]["id"] == law_id
    assert ledger["transfer_probes"][0]["law_id"] == law_id
    assert ledger["adversarial_probes"][0]["law_id"] == law_id
    assert ledger["scars"][0] == scar["entry"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
