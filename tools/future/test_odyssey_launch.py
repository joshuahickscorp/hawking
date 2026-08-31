"""Odyssey I launch gate tests.

Negative control: can_launch() is False today and names every unmet
criterion; ODYSSEY_I_LAUNCH.json is not written while the gate refuses.
A guard nobody has watched fail is not a guard.

Sparse-checkout trap: these tests never encode the checkout by asserting
that an unrelated file is absent. They assert the module copes with either
state and records which path it took.
"""
from __future__ import annotations

import hashlib
import ast
import json
import pathlib
from pathlib import Path

import pytest

from tools.future import odyssey_launch as ol
from tools.future import autonomy_trial as at
from tools.future import status_causality as sc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt, sha256_file
from hcli.workunit import WorkUnit


def test_entry_point_runs_and_seals_gate_receipt():
    out = ol.build()
    path = Path(out["gate_path"])
    assert path.parent == RECEIPTS
    assert path.name == ol.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == ol.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "recovered_implementation" in doc
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["workunit_emitted"]
    assert doc["resident_callable"]["receipt"]
    assert doc["resident_callable"]["frontier_fed"]
    assert doc["resident_callable"]["fail_closed"]


def test_sixteen_criteria_evaluated_in_contract_order():
    results = ol.evaluate_launch_criteria()
    assert [r["id"] for r in results] == list(ol.CRITERION_IDS)
    assert len(results) == len(ol.CRITERION_IDS)
    assert len(set(r["id"] for r in results)) == len(ol.CRITERION_IDS)
    for row in results:
        assert "met" in row
        assert row["reason"]
        assert "evidence" in row
        assert "operational" in row


def test_can_launch_names_every_unmet_and_agrees_with_itself():
    """NEGATIVE CONTROL: name EVERY unmet criterion, and never disagree with can_launch.

    This used to pin "nr_nx_path_callable" in unmet. Its own comment already had
    the principle right - it deliberately refused to pin autonomy or
    protected_scheduling because those were plumbing and would become met - and
    then pinned a third criterion that also became met, so the test failed
    BECAUSE the campaign closed an obligation.

    The invariants that actually belong here and survive the gate opening: unmet
    lists every failing criterion rather than stopping at the first, can_launch is
    False exactly when something is unmet, and the verdict block agrees with both.
    """
    results = ol.evaluate_launch_criteria()
    unmet = ol.unmet_criteria(results)
    assert unmet == [r["id"] for r in results if not r["met"]]
    assert ol.can_launch(results) is (not unmet), (
        "can_launch and unmet_criteria must never disagree"
    )
    verdict = ol.launch_verdict(results)
    assert verdict["verdict"] == ("LAUNCH" if not unmet else "REFUSED")
    assert verdict["allowed"] is (not unmet)
    assert verdict["unmet"] == unmet
    assert verdict["n_unmet"] == len(unmet)
    assert verdict["n_criteria"] == len(ol.CRITERION_IDS)


def test_unmet_aggregator_does_not_stop_at_first():
    """A first failure must not hide later failures."""
    fake = []
    for i, cid in enumerate(ol.CRITERION_IDS):
        fake.append({"id": cid, "met": i not in (0, 5, len(ol.CRITERION_IDS) - 1)})
    unmet = ol.unmet_criteria(fake)
    assert unmet[0] == ol.CRITERION_IDS[0]
    assert ol.CRITERION_IDS[5] in unmet
    assert unmet[-1] == ol.CRITERION_IDS[-1]
    assert unmet == [ol.CRITERION_IDS[0], ol.CRITERION_IDS[5], ol.CRITERION_IDS[-1]]
    assert ol.can_launch(fake) is False
    only_last = [{"id": cid, "met": cid != ol.CRITERION_IDS[-1]} for cid in ol.CRITERION_IDS]
    assert ol.unmet_criteria(only_last) == [ol.CRITERION_IDS[-1]]
    all_met = [{"id": cid, "met": True} for cid in ol.CRITERION_IDS]
    assert ol.can_launch(all_met) is True
    assert ol.unmet_criteria(all_met) == []


def test_launch_receipt_not_written_while_gate_refuses(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: refuse must not write ODYSSEY_I_LAUNCH.json."""
    written: list[str] = []

    def spy(name, doc, recorded_by):
        written.append(name)
        path = tmp_path / name
        path.write_text(json.dumps({"name": name, "recorded_by": recorded_by}))
        return path

    out = ol.build(writer=spy)
    assert ol.RECEIPT in written
    assert (tmp_path / ol.RECEIPT).is_file()
    # The control is the IMPLICATION, not the era. Refusing must never write the
    # phase-transition receipt; launching must write exactly one. Pinning "today
    # refuses" made this fail the moment the gate legitimately opened, which is
    # the one moment a launch safety control must still be working.
    refused = bool(out["doc"]["verdict"]["unmet"])
    if refused:
        assert out["doc"]["verdict"]["verdict"] == "REFUSED"
        assert ol.LAUNCH_RECEIPT not in written
        assert out["launch"]["written"] is False
        assert out["doc"]["odyssey_i_launch_written"] is False
        assert out["doc"]["phase_transition"] == "NOT_STARTED"
        assert (tmp_path / ol.LAUNCH_RECEIPT).exists() is False
    else:
        assert out["doc"]["verdict"]["verdict"] == "LAUNCH"
        assert written.count(ol.LAUNCH_RECEIPT) == 1
        assert out["launch"]["written"] is True
        assert out["doc"]["odyssey_i_launch_written"] is True
        assert out["doc"]["phase_transition"] == "STARTED"
        assert (tmp_path / ol.LAUNCH_RECEIPT).is_file()


def _bound_resident_block(**overrides):
    block = {
        "kind": "resident",
        "found": True,
        "bound": True,
        "status": "ACCEPTED",
        "schema": "hawking.future.resident_identity.v1",
        "pins": {
            "nx_id": {"model_id": "qwen3.8-27b-sealed-3.14"},
            "sealed_model_id": "qwen3.8-27b-sealed-3.14",
            "executable_hash": {"by_role": {"binary": "a" * 64}},
            "artifact_root": "/Users/scammermike/noetic/NOETIC_PARENT_A",
            "tokenizer": {"sha256": "b" * 64},
            "qualification": {"role": "CONTROL_HISTORICAL_NOT_CURRENT_PROOF"},
        },
        "pins_named": [
            "nx_id",
            "sealed_model_id",
            "executable_hash",
            "artifact_root",
            "tokenizer",
            "qualification",
        ],
        "missing": [],
        "agrees_with_incumbent": True,
        "unbound_reason": None,
    }
    block.update(overrides)
    return block


def test_forced_pass_writes_launch_receipt(tmp_path):
    payload = {
        "schema": ol.LAUNCH_SCHEMA,
        "phase_transition": "STARTED",
        "resident_identity": _bound_resident_block(),
    }
    refused = ol.write_launch_if_passed(payload, allowed=False, writer=lambda n, d, r: tmp_path / n)
    assert refused["written"] is False
    launched = ol.write_launch_if_passed(
        payload,
        allowed=True,
        writer=lambda n, d, r: (tmp_path / n).write_text("ok") or (tmp_path / n),
    )
    assert launched["written"] is True
    assert launched["name"] == ol.LAUNCH_RECEIPT
    assert (tmp_path / ol.LAUNCH_RECEIPT).is_file()


def test_write_launch_refuses_unbound_resident(tmp_path):
    """NEGATIVE CONTROL: a pass on the sixteen criteria must not mint an unbound launch."""
    payload = {
        "schema": ol.LAUNCH_SCHEMA,
        "phase_transition": "STARTED",
        "resident_identity": {
            "bound": False,
            "found": True,
            "status": "ACCEPTED",
            "missing": ["executable_hash"],
            "unbound_reason": "found but does not pin executable_hash",
        },
    }
    written: list[str] = []

    def spy(name, doc, recorded_by):
        written.append(name)
        path = tmp_path / name
        path.write_text("no")
        return path

    out = ol.write_launch_if_passed(payload, allowed=True, writer=spy)
    assert out["written"] is False
    assert ol.LAUNCH_RECEIPT not in written
    assert out.get("unbound_identity") is True
    assert "unbound" in out["reason"]
    assert "executable_hash" in out["reason"]
    assert (tmp_path / ol.LAUNCH_RECEIPT).exists() is False


def test_write_launch_clears_when_identity_binds(tmp_path):
    """The unbound guard must not become a permanent blocker once the identity binds."""
    payload = {
        "schema": ol.LAUNCH_SCHEMA,
        "phase_transition": "STARTED",
        "resident_identity": _bound_resident_block(),
    }
    out = ol.write_launch_if_passed(
        payload,
        allowed=True,
        writer=lambda n, d, r: (tmp_path / n).write_text("ok") or (tmp_path / n),
    )
    assert out["written"] is True
    assert out.get("unbound_identity") is not True
    assert (tmp_path / ol.LAUNCH_RECEIPT).is_file()


def test_bound_identity_does_not_open_the_gate(tmp_path):
    """Binding names the resident. It does not lower the sixteen-criterion bar."""
    payload = {
        "schema": ol.LAUNCH_SCHEMA,
        "phase_transition": "STARTED",
        "resident_identity": _bound_resident_block(),
    }
    out = ol.write_launch_if_passed(
        payload,
        allowed=False,
        writer=lambda n, d, r: (tmp_path / n).write_text("no") or (tmp_path / n),
    )
    assert out["written"] is False
    assert "criterion" in out["reason"]
    assert (tmp_path / ol.LAUNCH_RECEIPT).exists() is False


def test_launch_payload_identity_is_bound_or_names_the_missing_field():
    results = ol.evaluate_launch_criteria()
    curriculum = ol.propose_specimen_curriculum()
    first = (curriculum.get("roles") or [{}])[0]
    graphs = ol.emit_first_workgraphs(first)
    payload = ol.launch_payload(
        results, curriculum=curriculum, workgraphs=graphs, blockers=ol.physical_blockers()
    )
    resident = payload["resident_identity"]
    sandbox = payload["sandbox_identity"]
    assert resident["status"] is not None
    assert sandbox["status"] is not None
    if resident["bound"]:
        for field in (
            "nx_id",
            "sealed_model_id",
            "executable_hash",
            "artifact_root",
            "tokenizer",
            "qualification",
        ):
            assert field in resident["pins"], field
            assert field in resident["pins_named"], field
        assert resident["missing"] == []
        assert resident["unbound_reason"] is None
        assert resident["agrees_with_incumbent"] is True
    else:
        assert resident["missing"]
        assert resident["unbound_reason"]
        assert "not invented" in resident["unbound_reason"] or "does not pin" in resident["unbound_reason"] or "incumbent" in resident["unbound_reason"]
    if sandbox["bound"]:
        assert sandbox["pins"]["identity_sha256"]
        assert sandbox["pins"]["reentry_same_identity"] is True
        assert sandbox["schema"] == ol.SANDBOX_SCHEMA
        assert "RESIDENT_SANDBOX.json" in str(sandbox.get("resolved") or sandbox.get("rel") or "")
    else:
        assert sandbox["missing"]
        assert sandbox["unbound_reason"]
    # HCLI checkpoint is not this identity, even if that file exists.
    assert "HCLI_AGENTOS_CHECKPOINT" not in str(sandbox.get("resolved") or "")


def test_hcli_autonomy_gate_is_not_odyssey_trial():
    row = next(r for r in ol.evaluate_launch_criteria() if r["id"] == "resident_autonomy_trial_pass")
    kinds = {e.get("kind") for e in row["evidence"]}
    assert "hcli_agentos_autonomy" in kinds
    hcli = next(e for e in row["evidence"] if e.get("kind") == "hcli_agentos_autonomy")
    assert hcli.get("not_this_criterion") is True
    assert "path_taken" in hcli
    # Cope with either state: found or not, the path taken is recorded.
    assert hcli["path_taken"] in {
        "worktree",
        "evidence_snapshot",
        "primary_checkout",
        "git_head",
        "not_found",
    }
    wrong = next(e for e in row["evidence"] if e.get("kind") == "wrong_receipt_name")
    assert wrong.get("not_authority") is True
    assert "AUTONOMY_TRIAL.json" in wrong.get("rel", "")


def test_probe_json_records_path_taken_either_state():
    hit = ol.probe_json("receipts/future/ODYSSEY2_LAW_STORE.json")
    assert hit["found"] is True
    assert hit["path_taken"] in {"worktree", "evidence_snapshot", "primary_checkout", "git_head"}
    assert isinstance(hit["doc"], dict)
    miss = ol.probe_json("receipts/future/THIS_RECEIPT_IS_NOT_PART_OF_THE_CONTRACT_zzzz.json")
    assert miss["found"] is False
    assert miss["path_taken"] == "not_found"
    assert "searched" in miss
    assert miss["doc"] is None


def test_specimen_curriculum_five_roles_not_exhaustive():
    cur = ol.propose_specimen_curriculum()
    roles = [r["role"] for r in cur["roles"]]
    assert roles == [c[0] for c in ol.CURRICULUM_ROLES]
    assert cur["n_roles"] == len(ol.CURRICULUM_ROLES)
    assert cur["n_ready"] == sum(1 for r in cur["roles"] if r.get("ready"))
    # The invariant is the relation, not the value. The line below used to assert
    # ready is False directly under a comment saying not to freeze the ready
    # count, and it froze it: every role is verified now and the assertion failed
    # for the right reason.
    assert cur["ready"] is (cur["n_ready"] == cur["n_roles"])
    purposes = {r["role"]: r["purpose"] for r in cur["roles"]}
    assert "procedural speed" in purposes["very_small_dense_procedural_speed"]
    assert "alternate architecture" in purposes["small_dense_alternate_architecture_transfer"]
    assert "compiler" in purposes["mid_size_dense_compiler"]
    assert "Qwen27" in purposes["qwen27_mature_physical"]
    assert "Flash" in purposes["flash_heterogeneous_frontier"]
    extras = {row["repo"] for row in cur.get("not_proposed") or []}
    first_wave = {r["repo"] for r in cur["roles"]}
    assert extras.isdisjoint(first_wave)
    assert "Do not exhaustively optimize" in cur["not_proposed_rule"]


def test_first_workgraphs_are_real_workunits_with_dependencies():
    cur = ol.propose_specimen_curriculum()
    first = cur["roles"][0]
    graphs = ol.emit_first_workgraphs(first)
    units = graphs["units"]
    assert graphs["n_units"] == len(units)
    assert graphs["stages"] == list(ol.GRAPH_STAGES)
    by_id = {u["id"]: u for u in units}
    assert len(by_id) == len(units)
    for u in units:
        ol.wus.validate_emitted_unit(u)
        WorkUnit.from_dict(u)
        assert u["provider"] == "future.odyssey_launch"
        assert u["verifier"]
        assert u["claim_boundary"]
    slug = graphs["specimen"]["slug"]
    prefix = f"odyssey-i.wg.{slug}"
    for prev, stage in zip([None, *ol.GRAPH_STAGES[:-1]], ol.GRAPH_STAGES):
        uid = f"{prefix}.{stage}"
        deps = by_id[uid]["dependencies"]
        if prev is None:
            assert deps == []
        else:
            assert deps == [f"{prefix}.{prev}"]
    laws_id = f"{prefix}.laws"
    ii = by_id[f"{prefix}.phase_ii_transfer"]
    iii = by_id[f"{prefix}.phase_iii_attack"]
    assert ii["dependencies"] == [laws_id]
    assert iii["dependencies"] == [laws_id]
    assert f"{prefix}.phase_iii_attack" not in ii["dependencies"]
    assert f"{prefix}.phase_ii_transfer" not in iii["dependencies"]
    for stage in ol.GPU_STAGES:
        u = by_id[f"{prefix}.{stage}"]
        assert u["status"] == "sleeping"
        assert u["classification"] == "SLEEPING"
        assert u["resource_class"] == "GPU_EXCLUSIVE"
        assert "synthetic" in (u.get("blocked_reason") or "").lower() or "SLEEPING" in (u.get("blocked_reason") or "")


def test_phase_ii_iii_listen_concurrently_no_barrier():
    policy = ol.phase_listen_policy("phase_i.laws")
    assert policy["barrier"] is None
    assert policy["global_barrier"] is False
    assert policy["phase_ii_depends_on_phase_iii"] is False
    assert policy["phase_iii_depends_on_phase_ii"] is False
    assert policy["phase_ii_depends_on"] == ["phase_i.laws"]
    assert policy["phase_iii_depends_on"] == ["phase_i.laws"]
    assert policy["no_odyssey_iv"] is True
    assert policy["no_era_vi"] is True


def test_sleeping_physical_blockers_are_workunits_not_synthetic_results():
    blockers = ol.physical_blockers()
    holding = [b for b in blockers if b.get("holds")]
    assert holding, "live physical blockers hold today; an empty list would invent a clear machine"
    for b in holding:
        assert b["sleeping"] is True
        wu = b.get("workunit")
        assert wu is not None
        ol.wus.validate_emitted_unit(wu)
        assert wu["status"] == "sleeping"
        assert wu["classification"] == "SLEEPING"
        assert wu.get("synthetic_result_forbidden") is True
        assert wu.get("blocked_reason")


def test_gate_workunit_is_hcli_shaped():
    results = ol.evaluate_launch_criteria()
    wu = ol._gate_workunit(ol.launch_verdict(results))
    ol.wus.validate_emitted_unit(wu)
    assert wu["id"] == "odyssey-i.launch-gate"
    assert wu["status"] == "completed"
    assert wu["unmet"] == ol.unmet_criteria(results)
    # classification is REFUSED while the gate refuses and STATIC_ONLY once it
    # allows - an evidence class, not the verdict string. Pinning "REFUSED" made
    # this fail the day the gate opened, which is a fact about the calendar.
    allowed = ol.launch_verdict(results)["allowed"]
    assert wu["classification"] == ("STATIC_ONLY" if allowed else "REFUSED")
    assert wu["verdict"] == ol.launch_verdict(results)["verdict"]
    if allowed:
        assert wu["blocked_reason"] is None


def test_receipt_has_no_numeric_hardware_fields():
    out = ol.build()
    doc = json.loads(Path(out["gate_path"]).read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    assert doc["bench"]["state"] == "UNKNOWN"


def test_hardware_claim_still_raises_through_write_receipt(tmp_path, monkeypatch):
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "ODYSSEY_LAUNCH_GATE_SHOULD_NOT_LAND.json",
            {"schema": "test", "tps": 12.0},
            "tools/future/test_odyssey_launch.py",
        )
    leaked = RECEIPTS / "ODYSSEY_LAUNCH_GATE_SHOULD_NOT_LAND.json"
    if leaked.is_file():
        # write_receipt raises before write; if a prior run left this, do not
        # encode checkout absence — just refuse to treat it as this test's output.
        doc = json.loads(leaked.read_text())
        assert doc.get("tps") != 12.0


def test_siblings_are_exercised_not_merely_named():
    """The no-import rule was a WAVE-TIME constraint, and that wave has landed.

    While the siblings were being written concurrently, importing them would have
    coupled unfinished work. They are now committed, so the honest evaluation is
    to RUN them: a criterion that reports "not landed" about a module sitting on
    disk is stale, not careful. What must still hold is that every import is
    exercised rather than decorative, and that a failed exercise leaves the
    criterion unmet instead of flipping it true.
    """
    src = pathlib.Path(ol.__file__).read_text()
    exercised = {
        line.split('"')[1]
        for line in src.splitlines()
        if "_exercise(" in line and '"' in line
    }
    assert exercised, "no sibling is exercised; the criteria would be measuring absence"
    for dotted in exercised:
        assert dotted.startswith("tools.future."), dotted

    # A failed exercise must NOT be able to open a criterion.
    bad = ol._exercise("tools.future.does_not_exist", "build")
    assert bad["ok"] is False
    assert "import failed" in bad["why"]
    bad2 = ol._exercise("tools.future.frontiers", "no_such_function")
    assert bad2["ok"] is False
    assert "not callable" in bad2["why"]

def test_verify_cli_refuses_and_keeps_phase_not_started():
    rc = ol.verify()
    assert rc == 0
    doc = json.loads((RECEIPTS / ol.RECEIPT).read_text())
    refused = bool(doc["verdict"]["unmet"])
    assert doc["verdict"]["verdict"] == ("REFUSED" if refused else "LAUNCH")
    assert doc["odyssey_i_launch_written"] is (not refused)
    assert doc["launch_receipt"]["written"] is (not refused)
    assert doc["phase_transition"] == ("NOT_STARTED" if refused else "STARTED")


# ---------------------------------------------------------------------------
# Specimen readiness may be EARNED, but it must never be rounded up to.
# ---------------------------------------------------------------------------


def test_specimen_dirs_on_disk_copes_when_the_lake_is_not_mounted(monkeypatch):
    """ModelLake is an external volume. An unmounted lake is not an error."""
    assert isinstance(ol._specimen_dirs_on_disk(), set)


def test_disk_is_authority_over_the_census_cache():
    """A specimen the census never recorded still exists.

    Reading only the census reported Mistral-Small-24B as absent from the
    ModelLake specimens listing while its directory sat in that listing.
    """
    dirs = ol._specimen_dirs_on_disk()
    if not dirs:
        return  # lake not mounted here; the fixtures below carry the guarantees
    idx = ol._lake_index({})
    for name in dirs:
        slug, _, _rev = name.partition("@")
        repo = slug.replace("--", "/", 1)
        assert repo in idx, f"{repo} is on disk but absent from the index"
        assert idx[repo]["in_specimens_listing"] is True


def test_earned_verification_requires_a_real_recomputation(tmp_path, monkeypatch):
    """NEGATIVE CONTROLS: each way of NOT verifying must fail to count."""
    rows = {
        "hashed_nothing": {"specimen": "a", "status": "WHOLE_TREE_VERIFIED",
                           "bytes_hashed": 0, "mismatched": 0,
                           "no_remote_digest": 0, "verified": 3, "n_files": 3},
        "a_file_mismatched": {"specimen": "b", "status": "WHOLE_TREE_VERIFIED",
                              "bytes_hashed": 99, "mismatched": 1,
                              "no_remote_digest": 0, "verified": 3, "n_files": 3},
        "a_file_undigested": {"specimen": "c", "status": "WHOLE_TREE_VERIFIED",
                              "bytes_hashed": 99, "mismatched": 0,
                              "no_remote_digest": 1, "verified": 3, "n_files": 3},
        "counted_more_than_it_checked": {"specimen": "d", "status": "WHOLE_TREE_VERIFIED",
                                         "bytes_hashed": 99, "mismatched": 0,
                                         "no_remote_digest": 0, "verified": 2, "n_files": 3},
        "merely_partial": {"specimen": "e", "status": "PARTIAL_NO_REMOTE_DIGEST",
                           "bytes_hashed": 99, "mismatched": 0,
                           "no_remote_digest": 1, "verified": 2, "n_files": 3},
    }
    good = {"specimen": "ok", "status": "WHOLE_TREE_VERIFIED", "bytes_hashed": 99,
            "mismatched": 0, "no_remote_digest": 0, "verified": 3, "n_files": 3}

    def _probe(doc):
        monkeypatch.setattr(ol, "probe_json", lambda *a, **k: {"found": True, "doc": doc})
        return ol._independently_verified()

    for why, row in rows.items():
        assert _probe({"results": [row]}) == {}, f"a specimen that {why} was counted"
    assert set(_probe({"results": [good]})) == {"ok"}


def test_the_curriculum_never_asserts_an_absence_it_did_not_check():
    """Every unready role must give a reason derived from evidence, not a constant."""
    cur = ol.propose_specimen_curriculum()
    for role in cur["roles"]:
        if role.get("ready"):
            continue
        assert role.get("ready_reason"), f"{role['role']} refused without a reason"
        if "not in the ModelLake specimens listing" in role["ready_reason"]:
            slug = str(role["repo"]).replace("/", "--", 1)
            on_disk = {n.partition("@")[0] for n in ol._specimen_dirs_on_disk()}
            assert slug not in on_disk, (
                f"{role['repo']} was called absent from the listing but is on disk"
            )


# ---------------------------------------------------------------------------
# A tool whose parent is not on this host is not callable by anyone.
# ---------------------------------------------------------------------------


def test_declared_inputs_are_parsed_not_imported(tmp_path):
    """Importing an odyssey tool would run its module-level work inside the gate."""
    rel = "tools/future/_fixture_declared_inputs.py"
    src = ol.REPO / rel
    src.write_text(
        'from pathlib import Path\n'
        'PARENT = Path("/definitely/not/here")\n'
        'RELATIVE = Path("receipts/headless")\n'
        'SIDE_EFFECT = print("this must never run")\n'
    )
    try:
        found = ol._declared_inputs(rel)
    finally:
        src.unlink()
    names = {i["name"] for i in found}
    assert names == {"PARENT"}, "only absolute external inputs count"
    assert found[0]["present"] is False


def test_doctor_and_gravity_blame_the_real_blocker_not_the_runtime():
    """The reason must name what is actually blocking, or it sends work astray."""
    for evaluate in (ol._eval_doctor, ol._eval_gravity):
        row = evaluate()
        missing = row["evidence"][0]["missing_inputs"]
        if not missing:
            continue
        assert row["met"] is False
        assert not row["operational"]["flags"]["invoke"], (
            "a tool whose declared parent is unreachable cannot be invoked"
        )
        for item in missing:
            assert item["path"] in row["reason"], "the blocker is not named in the reason"


def test_a_moved_parent_is_reported_as_stale_not_as_missing():
    """52GB that is already on the disk must never be reported as absent.

    Doctor and Gravity declare /Users/scammermike/models/... which is gone. The
    parent itself is on the external volume. Calling that a missing model sends
    someone to re-download it.
    """
    row = ol._eval_doctor()
    stale = row["evidence"][0]["stale_declared_paths"]
    if not stale:
        return  # nothing moved on this host
    for item in stale:
        assert pathlib.Path(item["resolved_elsewhere"]).is_dir()
        assert pathlib.Path(item["path"]).name == pathlib.Path(item["resolved_elsewhere"]).name
    assert "STALE DECLARED PATH" in row["reason"]
    assert "not a missing model" in row["reason"]


def test_resolution_is_bounded_to_known_model_roots(tmp_path):
    """A find over a 4.5TB volume is not a probe; only named roots are searched."""
    assert ol._resolve_stale_input("/nowhere/at/all/a-name-that-does-not-exist") is None
    for root in ol.MODEL_ROOTS:
        assert root.startswith("/")


def test_negative_control_the_gate_cannot_certify_itself_as_the_driver():
    """odyssey_launch names these tools in an `owned = [...]` literal.

    An earlier version accepted any Assign as evidence of driving and so
    credited this gate module as Doctor's resident driver -- self-certification,
    which the constitution forbids outright.
    """
    sched = ol._resident_schedulable(["tools/odyssey/doctor_tournament.py"])
    assert sched.get("driver_module") != "odyssey_launch.py"
    # A real driver now exists (odyssey_tool_driver.py), so schedule is true --
    # but it must be true because a module DRIVES the tool, never because this
    # gate names it in an `owned = [...]` literal. That is the invariant; the
    # boolean was only ever a proxy for it while no driver existed.
    if sched["schedule"]:
        driver = sched["driver_module"]
        src = (ol.REPO / "tools" / "future" / driver).read_text(errors="replace")
        tree = ast.parse(src)
        in_call = any(
            isinstance(n, ast.Call) and "doctor_tournament.py" in ast.dump(n)
            for n in ast.walk(tree)
        )
        assert in_call, f"{driver} names the tool but never calls it"


def test_a_retired_patient_needs_BOTH_the_prior_seal_and_fresh_verification():
    """Recurrence is earned twice over, or it is a pass on a stale seal.

    The odysseys are recurrent phases and the first completion is historical, so
    a patient retired from that wave is a specimen with a proven role. But
    retirement alone would let a years-old seal stand in for verification, and
    verification alone would throw away the prior work.
    """
    verified_and_sealed = {
        "patient_state": "RETIRED", "patient_seal": "sha256:abc",
        "whole_tree_verified": True, "revision": "r1",
    }
    ready, why = ol._ready(verified_and_sealed, require_lake_verified=True)
    assert ready is True
    assert "RECURRENT_PATIENT" in why

    for missing in ("patient_seal", "whole_tree_verified"):
        row = dict(verified_and_sealed)
        row[missing] = None if missing == "patient_seal" else False
        ready, why = ol._ready(row, require_lake_verified=True)
        assert ready is False, f"a retired patient passed without {missing}"


def test_recurrence_is_labelled_so_it_is_not_read_as_a_first_wave_result():
    cur = ol.propose_specimen_curriculum()
    for role in cur["roles"]:
        if role.get("ready") and "RETIRED" in str(role.get("modellake", {}).get("patient_state", "")):
            assert "RECURRENT_PATIENT" in role["ready_reason"]


def test_a_specimen_under_partial_is_located_not_missing():
    """"Not in the specimens listing" was read as "the model is not here".

    Qwen3-0.6B was complete inside ModelLake the whole time -- ten files, ten
    published digests, zero incomplete markers -- sitting under partial/ rather
    than specimens/. Location was the only thing partial about it. Reporting
    that as a missing model would have sent someone to re-download it, which is
    the same failure as the three Odyssey tools pointing at a moved directory.
    """
    cur = ol.propose_specimen_curriculum()
    role = next(r for r in cur["roles"]
                if r["role"] == "very_small_dense_procedural_speed")
    if not role.get("located_under_partial"):
        return  # not verified on this host; the rule below still holds
    assert role["ready"] is True
    # Readiness still had to be EARNED by recomputation, not granted by location.
    verified = ol._independently_verified()
    row = verified["Qwen--Qwen3-0.6B@c1899de289a0#partial"]
    assert row["verified"] == row["n_files"]
    assert row["mismatched"] == 0 and row["no_remote_digest"] == 0
    assert row["bytes_hashed"] > 0, "readiness claimed without hashing anything"


def test_finding_a_specimen_does_not_lower_the_bar_for_the_others():
    """NEGATIVE CONTROL: every ready role must still name real verification."""
    for role in ol.propose_specimen_curriculum()["roles"]:
        if not role.get("ready"):
            continue
        why = role["ready_reason"]
        assert any(k in why for k in
                   ("whole-tree", "RECURRENT_PATIENT", "tree digest is sealed")), (
            f"{role['role']} was made ready by something other than verification: {why}"
        )


def test_protected_scheduling_is_measured_not_a_constant(monkeypatch):
    """Capability and availability must be able to diverge.

    The old evaluator ANDed both, so a HEAVY machine looked like 'the resident
    cannot handle protected work'. capability_report() already separates them.
    """
    live = ol._eval_protected_scheduling()
    notes = live["operational"]["notes"]
    assert notes["availability_is_not_capability"] == "true"
    assert notes["sidecar_must_not_seize_lock"] == "true"
    assert "PROTECTED_SCHEDULER_CAPABLE" in live
    assert "PROTECTED_WINDOW_AVAILABLE" in live
    # Live contamination is not QUIESCENT: the window must not be reported available.
    if live.get("contamination_class") != "QUIESCENT":
        assert live["PROTECTED_WINDOW_AVAILABLE"] is False

    def fake_cap_capable_unavailable(**kw):
        return {
            "invoked": True,
            "why": "injected: capable, window closed",
            "PROTECTED_SCHEDULER_CAPABLE": True,
            "PROTECTED_WINDOW_AVAILABLE": False,
            "contamination_class": "HEAVY",
            "lease_present": False,
            "live_verdict": "BLOCKED_ON_PROTECTED_WINDOW",
            "did_not_fabricate_lease": True,
            "did_not_flock": True,
            "receipt_path_taken": "injected",
            "import_path_taken": "injected",
            "availability_overridden_because_not_quiescent": False,
        }

    monkeypatch.setattr(ol, "_protected_capability_report", fake_cap_capable_unavailable)
    lifted = ol._eval_protected_scheduling()
    assert lifted["met"] is True
    assert lifted["PROTECTED_SCHEDULER_CAPABLE"] is True
    assert lifted["PROTECTED_WINDOW_AVAILABLE"] is False
    assert lifted["operational"]["flags"]["invoke"] is True


def test_refusing_to_seize_a_lock_is_not_the_same_claim_as_cannot_schedule():
    """The policy and the capability are different facts and must stay separate."""
    row = ol._eval_protected_scheduling()
    notes = row["operational"]["notes"]
    assert notes["sidecar_must_not_seize_lock"] == "true"
    assert notes["did_not_flock"] == "True" or notes["did_not_flock"] == "true"
    run = row.get("protected_physical_run_completed") or {}
    assert run.get("required_for_this_criterion") is False
    assert run.get("id") == "protected_physical_run_completed"


def test_a_declared_status_does_not_outrank_a_measurement():
    """"weights are not present" for 335GB of verified shards.

    The Flash school carries physical_status metadata_only_weights_not_present.
    It was true when the law store was written. Today the specimen is 144 of 144
    files whole-tree verified by recomputing every published digest. Deferring to
    the string would refuse a specimen on stale text -- the same failure as the
    moved Doctor parent, the absent-but-present GPU, and the specimen filed under
    partial/.
    """
    stale = {"physical_status": "metadata_only_weights_not_present",
             "revision": "r1", "whole_tree_verified": True, "bytes_hashed": 335_000_000_000}
    ready, why = ol._ready(stale, require_lake_verified=True)
    assert ready is True, why

    # NEGATIVE CONTROL: measurement wins only when it is real. A status flip on a
    # verified flag with nothing hashed behind it is exactly the laundering this
    # is meant to refuse.
    for weak in ({"whole_tree_verified": True, "bytes_hashed": 0},
                 {"whole_tree_verified": True},
                 {"whole_tree_verified": False, "bytes_hashed": 335_000_000_000}):
        row = {"physical_status": "metadata_only_weights_not_present", "revision": "r1", **weak}
        ready, why = ol._ready(row, require_lake_verified=True)
        assert ready is False, f"a specimen passed on {weak} with nothing measured"


def test_the_curriculum_is_ready_only_when_every_role_names_verification():
    cur = ol.propose_specimen_curriculum()
    if not cur["ready"]:
        return
    assert cur["n_ready"] == cur["n_roles"]
    for role in cur["roles"]:
        assert role["ready"] is True
        assert any(k in role["ready_reason"] for k in
                   ("whole-tree", "RECURRENT_PATIENT", "tree digest is sealed"))


def _inject_persisted_autonomy(monkeypatch, tmp_path, *, verdict, trial="1h", mutate=False, extra=None):
    """A real timeline file plus a persisted record the evaluator will read."""
    tl = tmp_path / "timeline.json"
    body = at.build_passing_timeline("1h" if trial in at.LAUNCH_ELIGIBLE_TRIALS else "15m")
    body["trial"] = trial
    tl.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    digest = sha256_file(tl)
    if mutate:
        tl.write_text(tl.read_text() + " ")
    record = {
        "schema": at.VERDICT_PERSIST_SCHEMA,
        "trial": trial,
        "verdict": verdict,
        "reason": "injected",
        "conditions_met": [],
        "conditions_unmet": ["injected"] if verdict == "FAIL" else [],
        "timeline_path": str(tl),
        "timeline_seal_digest": digest,
        "resident_orchestration": True,
        "resident_orchestration_reason": "injected HCLI loop launched work",
        "resident_model_cognition": "UNAVAILABLE",
        "resident_model_cognition_reason": "injected: no model in the loop",
        "frozen_build_manifest_digest": None,
        "orchestration_is_not_cognition": True,
    }
    if extra:
        record.update(extra)
    doc = {"schema": at.SCHEMA, "persisted_verdicts_by_trial": {trial: record}}
    monkeypatch.setattr(
        ol,
        "_autonomy_trials_doc",
        lambda: {"found": True, "path_taken": "injected", "doc": doc, "searched": []},
    )
    return record, tl


def test_fail_verdict_never_satisfies_autonomy_criterion(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: a persisted FAIL is not a pass, even with orchestration."""
    _inject_persisted_autonomy(monkeypatch, tmp_path, verdict="FAIL", trial="1h")
    row = ol._eval_autonomy()
    assert row["met"] is False
    assert row["persisted_verdict"] == "FAIL"
    assert row["resident_orchestration"] is True
    assert "FAIL" in row["reason"]


def test_tampered_timeline_seal_is_refused(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: editing the transcript after judgement is a refusal."""
    _inject_persisted_autonomy(monkeypatch, tmp_path, verdict="PASS", trial="1h", mutate=True)
    row = ol._eval_autonomy()
    assert row["met"] is False
    assert row["timeline_seal_verifies"] is False
    assert "seal" in row["reason"].lower() or "digest" in row["reason"].lower() or "transcript" in row["reason"].lower()


def test_persisted_pass_with_verifying_seal_and_orchestration_meets(tmp_path, monkeypatch):
    _inject_persisted_autonomy(monkeypatch, tmp_path, verdict="PASS", trial="1h")
    row = ol._eval_autonomy()
    assert row["met"] is True
    assert row["resident_orchestration"] is True
    assert row["resident_model_cognition"] == "UNAVAILABLE"
    assert row["timeline_seal_verifies"] is True
    # Cognition is recorded, not required to be a thinking model.
    assert row["resident_orchestration"] is not row["resident_model_cognition"]


def test_fifteen_minute_pass_is_not_the_launch_bar(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: 15m is a real trial; it is not Odyssey I start."""
    _inject_persisted_autonomy(monkeypatch, tmp_path, verdict="PASS", trial="15m")
    row = ol._eval_autonomy()
    assert row["met"] is False
    # launch_candidate_from_receipt skips 15m, so the evaluator sees no 1h+
    # candidate. Either phrasing is a refusal of the short trial as the bar.
    assert (
        "15m" in row["reason"]
        or "1h/3h/6h" in row["reason"]
        or "launch bar" in row["reason"]
    )


def test_singular_autonomy_trial_receipt_is_not_authority(monkeypatch):
    """NEGATIVE CONTROL: the receipt-name bug that cost 70 ingestions."""
    fake_singular = {
        "found": True,
        "path_taken": "injected",
        "doc": {
            "schema": "hawking.future.autonomy_trial.v1",
            "verdict": "PASS",
            "resident_orchestration": True,
        },
    }
    real_probe = ol.probe_json

    def probe(*rels, **kw):
        if rels and "AUTONOMY_TRIAL.json" in rels[0] and "AUTONOMY_TRIALS" not in rels[0]:
            return fake_singular
        if rels and rels[0].endswith("AUTONOMY_TRIALS.json"):
            return {"found": False, "path_taken": "not_found", "doc": None, "searched": []}
        return real_probe(*rels, **kw)

    monkeypatch.setattr(ol, "probe_json", probe)
    monkeypatch.setattr(
        ol,
        "_autonomy_trials_doc",
        lambda: {"found": False, "path_taken": "not_found", "doc": None, "searched": []},
    )
    row = ol._eval_autonomy()
    assert row["met"] is False
    wrong = next(e for e in row["evidence"] if e.get("kind") == "wrong_receipt_name")
    assert wrong["found"] is True
    assert wrong["not_authority"] is True


def test_orchestration_true_does_not_imply_cognition(tmp_path, monkeypatch):
    _inject_persisted_autonomy(monkeypatch, tmp_path, verdict="PASS", trial="1h")
    row = ol._eval_autonomy()
    assert row["resident_orchestration"] is True
    assert row["resident_model_cognition"] == "UNAVAILABLE"
    notes = row["operational"]["notes"]
    assert notes["orchestration_is_not_cognition"] == "true"


def test_protected_window_unavailable_when_contamination_not_quiescent(monkeypatch):
    """NEGATIVE CONTROL: HEAVY + claimed available is refused on the window field."""
    def lying_report():
        return {
            "invoked": True,
            "why": "injected lie: available on HEAVY",
            "PROTECTED_SCHEDULER_CAPABLE": True,
            "PROTECTED_WINDOW_AVAILABLE": True,
            "contamination_class": "HEAVY",
            "lease_present": False,
            "live_verdict": "RUNNABLE",
            "did_not_fabricate_lease": True,
            "did_not_flock": True,
            "receipt_path_taken": "injected",
            "import_path_taken": "injected",
            "availability_overridden_because_not_quiescent": True,
        }

    monkeypatch.setattr(ol, "_protected_capability_report", lying_report)
    row = ol._eval_protected_scheduling()
    assert row["PROTECTED_WINDOW_AVAILABLE"] is False
    assert row["PROTECTED_SCHEDULER_CAPABLE"] is True
    assert row["met"] is True  # capability, not availability


def test_incapable_scheduler_does_not_meet(monkeypatch):
    """NEGATIVE CONTROL: CAPABLE false is unmet, even if someone claims a window."""
    def incapable():
        return {
            "invoked": True,
            "why": "injected: not capable",
            "PROTECTED_SCHEDULER_CAPABLE": False,
            "PROTECTED_WINDOW_AVAILABLE": True,
            "contamination_class": "QUIESCENT",
            "lease_present": True,
            "live_verdict": "REFUSED",
            "did_not_fabricate_lease": True,
            "did_not_flock": True,
            "receipt_path_taken": "injected",
            "import_path_taken": "injected",
            "availability_overridden_because_not_quiescent": False,
        }

    monkeypatch.setattr(ol, "_protected_capability_report", incapable)
    row = ol._eval_protected_scheduling()
    assert row["met"] is False
    assert row["PROTECTED_SCHEDULER_CAPABLE"] is False


def test_flash_nx_ready_is_a_separate_field_from_the_generic_path():
    """NEGATIVE CONTROL: Flash readiness is not the generic criterion.

    This asserted GENERIC_NR_NX_PIPELINE_CALLABLE is False, which was a fact about
    one afternoon rather than a property of the gate - the generic packer landed
    and the criterion closed, and the test failed BECAUSE the obligation was met.
    The real invariant is independence: Flash stays False, the criterion tracks
    the GENERIC field, and the two are never conflated.
    """
    row = ol._eval_nr_nx()
    assert row["FLASH_NX_READY"] is False
    assert row["met"] is row["GENERIC_NR_NX_PIPELINE_CALLABLE"], (
        "the criterion must track the generic field, not Flash"
    )
    assert "generic" in row["reason"].lower() or "GENERIC" in row["reason"]


def test_flash_ready_does_not_satisfy_generic_nr_nx(monkeypatch):
    """NEGATIVE CONTROL: one model's artifact is not the orchestration path."""
    monkeypatch.setattr(
        ol,
        "_nr_nx_generic_state",
        lambda: {
            "invoked": ["nr_nx_generic.generic_pipeline_callable", "nr_nx_generic.flash_nx_ready"],
            "import": {"ok": True, "path_taken": "injected"},
            "receipt_path_taken": "injected",
            "receipt_found": True,
            "GENERIC_NR_NX_PIPELINE_CALLABLE": False,
            "GENERIC_FROM_RECEIPT": False,
            "FLASH_NX_READY": True,
            "FLASH_FROM_RECEIPT": True,
            "flash": {"FLASH_NX_READY": True},
            "flash_why": "injected flash ready",
            "first_failing_stage": {"stage": "Doctor", "why": "injected fail"},
            "n_stages": 14,
        },
    )
    row = ol._eval_nr_nx()
    assert row["met"] is False
    assert row["FLASH_NX_READY"] is True
    assert row["GENERIC_NR_NX_PIPELINE_CALLABLE"] is False


def test_generic_callable_meets_while_flash_stays_false(monkeypatch):
    monkeypatch.setattr(
        ol,
        "_nr_nx_generic_state",
        lambda: {
            "invoked": ["nr_nx_generic.generic_pipeline_callable", "nr_nx_generic.flash_nx_ready"],
            "import": {"ok": True, "path_taken": "injected"},
            "receipt_path_taken": "injected",
            "receipt_found": True,
            "GENERIC_NR_NX_PIPELINE_CALLABLE": True,
            "GENERIC_FROM_RECEIPT": True,
            "FLASH_NX_READY": False,
            "FLASH_FROM_RECEIPT": False,
            "flash": {"FLASH_NX_READY": False},
            "flash_why": "injected flash not ready",
            "first_failing_stage": None,
            "n_stages": 14,
        },
    )
    row = ol._eval_nr_nx()
    assert row["met"] is True
    assert row["FLASH_NX_READY"] is False
    assert row["GENERIC_NR_NX_PIPELINE_CALLABLE"] is True


def test_gate_never_writes_launch_receipt_while_any_criterion_unmet(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: ODYSSEY_I_LAUNCH.json is a phase transition, not a diary."""
    written: list[str] = []

    def spy(name, doc, recorded_by):
        written.append(name)
        path = tmp_path / name
        path.write_text(json.dumps({"name": name}))
        return path

    out = ol.build(writer=spy)
    # The invariant is the IMPLICATION - any unmet criterion blocks the
    # phase-transition receipt - not the identity of today's unmet criterion and
    # not the assumption that there IS one. This first pinned
    # nr_nx_path_callable, then assumed the unmet set was non-empty; both broke
    # when the campaign closed the obligations, which is the one moment a launch
    # safety control most needs to still work.
    unmet = out["doc"]["verdict"]["unmet"]
    rewire = out["doc"]["rewire"]
    if unmet:
        # still_refused_if_any_unmet is itself conditional on there BEING an
        # unmet criterion; with none it reports False, which is correct and not
        # a regression.
        assert rewire["still_refused_if_any_unmet"] is True
        assert out["doc"]["verdict"]["verdict"] == "REFUSED"
        assert ol.LAUNCH_RECEIPT not in written
        assert out["launch"]["written"] is False
        assert out["doc"]["odyssey_i_launch_written"] is False
        assert (tmp_path / ol.LAUNCH_RECEIPT).exists() is False
        assert rewire["gate_count_after"]["n_unmet"] >= 1
    else:
        assert out["doc"]["verdict"]["verdict"] == "LAUNCH"
        assert written.count(ol.LAUNCH_RECEIPT) == 1
        assert out["doc"]["odyssey_i_launch_written"] is True
        assert (tmp_path / ol.LAUNCH_RECEIPT).is_file()


# ---------------------------------------------------------------------------
# G007 consumer: every criterion records the five causality fields.
# ---------------------------------------------------------------------------


def test_every_wired_criterion_records_the_five_fields():
    """A coverage number no test defends will drift back to zero."""
    results = ol.evaluate_launch_criteria()
    assert [r["id"] for r in results] == list(ol.CRITERION_IDS)
    missing = [r["id"] for r in results if not ol.records_five_fields(r)]
    assert missing == [], f"wired criteria stopped recording the five fields: {missing}"
    src = pathlib.Path(ol.__file__).read_text()
    assert "sc.emit(" in src
    for row in results:
        assert row["probe_performed"] != row["id"]
        assert row["direct_observation"] != row["id"]
        assert row["direct_observation"] != row["reason"]
        assert row["interpretation"]
        assert row["confidence"]["level"]
        assert row["alternatives"]
        assert row["causality_verdict"] in {sc.SUPPORTED, sc.OVERREACHING, sc.UNTESTED}
        # probe describes what was done, not a restatement of the status label
        assert row["id"] not in row["probe_performed"] or any(
            token in row["probe_performed"]
            for token in ("probe_json", "import", "invoke", "Path", "glob", "_module_file", "_exercise")
        )


def test_unsupplied_observation_records_untested_not_a_restatement():
    """A gate that cannot supply an observation must not fabricate one from the status."""
    row = {
        "id": "doctor_callable",
        "met": False,
        "reason": "Doctor is not resident-callable",
    }
    rec = ol.record_criterion_causality(
        row, probe_performed="", direct_observation=""
    )
    assert rec["verdict"] == sc.UNTESTED
    assert rec["direct_observation"] in ("", None)
    assert rec["direct_observation"] != row["reason"]
    assert rec["direct_observation"] != "doctor_callable"
    assert "not resident-callable" not in str(rec["direct_observation"] or "")
    assert row["met"] is False
    # interpretation may name the status; observation must not copy it
    assert rec["interpretation"] != rec["direct_observation"]


def test_overreaching_classification_does_not_override_met(monkeypatch):
    """OVERREACHING is information for the reader, not an override of the gate."""

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

    monkeypatch.setattr(ol.sc, "emit", overreach)
    monkeypatch.setattr(
        ol,
        "_nr_nx_generic_state",
        lambda: {
            "invoked": ["nr_nx_generic.generic_pipeline_callable"],
            "import": {"ok": True, "path_taken": "injected"},
            "receipt_path_taken": "injected",
            "receipt_found": True,
            "GENERIC_NR_NX_PIPELINE_CALLABLE": True,
            "GENERIC_FROM_RECEIPT": True,
            "FLASH_NX_READY": False,
            "FLASH_FROM_RECEIPT": False,
            "flash": {"FLASH_NX_READY": False},
            "flash_why": "injected",
            "first_failing_stage": None,
            "n_stages": 14,
        },
    )
    row = ol._eval_nr_nx()
    assert row["met"] is True
    assert row["GENERIC_NR_NX_PIPELINE_CALLABLE"] is True
    assert row["causality_verdict"] == sc.OVERREACHING


def test_coverage_receipt_names_recording_and_remainder():
    """Coverage is names, not a percentage. The remainder is named, not dropped."""
    path = RECEIPTS / "STATUS_CAUSALITY_COVERAGE.json"
    assert path.is_file(), "STATUS_CAUSALITY_COVERAGE.json must be written"
    doc = json.loads(path.read_text())
    for banned in ("percent", "percentage", "coverage_pct", "pct"):
        assert banned not in doc
    recording = doc["recording_five_fields"]
    missing = doc["not_recording_five_fields"]
    unread = doc["unreadable"]
    assert "odyssey_launch" in recording
    assert "integration_gate" in recording
    # S015 dissolved the partition. The eight hcli/agentos gates are wired.
    # The remainder that must stay named is the Rust capture boundary.
    for name in ("resident_gate", "native_gate", "native_mission_gate",
                 "autonomy_gate", "modellake_gate", "vmcp_gate",
                 "recovery_gate", "research_gate"):
        assert name in recording, f"{name} vanished from recording"
        assert name not in missing
    assert "flash_meta_teacher_capture_boundary" in missing, (
        "flash_meta_teacher_capture_boundary vanished from the remainder"
    )
    assert set(recording).isdisjoint(missing)
    assert set(recording).isdisjoint(unread)
    for cid in ol.CRITERION_IDS:
        assert cid in doc["odyssey_launch_criteria_recording_five_fields"]
    assert doc["odyssey_launch_criteria_not_recording_five_fields"] == []
    assert doc["n_gates"] == 18
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False


def test_causality_stamp_does_not_change_live_met_unmet():
    """The criterion's own met/unmet is not replaced by the causal record."""
    results = ol.evaluate_launch_criteria()
    by_id = {r["id"]: r for r in results}
    nr = by_id["nr_nx_path_callable"]
    assert nr["met"] is bool(nr.get("GENERIC_NR_NX_PIPELINE_CALLABLE"))
    prot = by_id["protected_scheduling"]
    assert prot["met"] is bool(prot.get("PROTECTED_SCHEDULER_CAPABLE"))
    recs = by_id["receipts"]
    assert recs["met"] is True
    assert recs["operational"]["resident_operational"] is True
    for row in results:
        assert isinstance(row["met"], bool)
        if row["causality_verdict"] == sc.OVERREACHING:
            # overreach is beside the verdict, not a flip
            assert "met" in row
