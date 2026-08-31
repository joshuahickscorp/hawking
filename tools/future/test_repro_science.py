"""Negative control for tools/future/repro_science.py.

The twelve fault injectors ARE the negative control: every one must be shown
to be DETECTED. A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import repro_science as rs
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


def test_build_emits_sealed_receipt():
    out = rs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "REPRO_SCIENCE.json"
    assert doc["schema"] == rs.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert rs.seal_is_valid(doc)
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["fault_injection"]["all_detected"] is True
    assert doc["fault_injection"]["n_detected"] == len(rs.FAULT_NAMES)
    assert doc["claim_downgrade"]["transitivity_holds"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert len(doc["eras"]) == 5
    assert len(doc["odysseys"]) == 3
    assert "VI" not in "".join(doc["eras"])
    assert "IV" not in "".join(doc["odysseys"])


def test_selftest_emits_the_same_receipt():
    out = rs.selftest()
    assert out.name == "REPRO_SCIENCE.json"
    assert rs.seal_is_valid(json.loads(out.read_text()))


def test_identity_reproduces():
    genome = rs.fixture_machine_genome()
    inputs = {"specimen": rs.fixture_specimen_hash(), "route_corpus": "a" * 64}
    kwargs = dict(
        inputs=inputs,
        code_sha256=rs.fixture_code_sha256(),
        compiler="sidecar-static-compiler-pin",
        machine_genome=genome,
    )
    assert rs.experiment_identity(**kwargs) == rs.experiment_identity(**kwargs)


def test_identity_changes_with_input():
    genome = rs.fixture_machine_genome()
    code = rs.fixture_code_sha256()
    compiler = "sidecar-static-compiler-pin"
    a = rs.experiment_identity(
        inputs={"specimen": "a" * 64},
        code_sha256=code,
        compiler=compiler,
        machine_genome=genome,
    )
    b = rs.experiment_identity(
        inputs={"specimen": "b" * 64},
        code_sha256=code,
        compiler=compiler,
        machine_genome=genome,
    )
    assert a != b


def test_identity_changes_with_code_and_genome_and_compiler():
    genome = rs.fixture_machine_genome()
    inputs = {"specimen": "a" * 64}
    base = rs.experiment_identity(
        inputs=inputs,
        code_sha256="a" * 64,
        compiler="c1",
        machine_genome=genome,
    )
    assert (
        base
        != rs.experiment_identity(
            inputs=inputs,
            code_sha256="b" * 64,
            compiler="c1",
            machine_genome=genome,
        )
    )
    other = dict(genome)
    other["arch"] = "x86_64"
    assert (
        base
        != rs.experiment_identity(
            inputs=inputs,
            code_sha256="a" * 64,
            compiler="c1",
            machine_genome=other,
        )
    )
    assert (
        base
        != rs.experiment_identity(
            inputs=inputs,
            code_sha256="a" * 64,
            compiler="c2",
            machine_genome=genome,
        )
    )


def test_provenance_traces_claim_to_code_inputs_machine():
    g = rs.example_provenance_graph()
    traced = rs.trace_claim(g, "CLAIM_GRANDCHILD")
    kinds = {n["kind"] for n in traced}
    ids = {n["id"] for n in traced}
    assert kinds >= {"claim", "output", "experiment", "input", "code", "machine"}
    assert {"CLAIM_GRANDCHILD", "CLAIM_CHILD", "CLAIM_PARENT", "OUT1", "EXP1", "CODE", "SPECIMEN", "MACHINE"} <= ids


def test_unknown_claim_fail_closes():
    with pytest.raises(rs.FailClosed) as ei:
        rs.trace_claim(rs.example_provenance_graph(), "NO_SUCH_CLAIM")
    assert ei.value.fault == "unknown_claim"


def test_receipt_reader_suite_is_flagged():
    dead = (
        "import json\n"
        "from pathlib import Path\n"
        "def test_repro():\n"
        "    doc = json.loads(Path('receipts/future/REPRO_SCIENCE.json').read_text())\n"
        "    assert doc['schema'] == 'hawking.future.repro_science.v1'\n"
    )
    assert rs.is_receipt_reader_suite(dead) is True


def test_live_suite_is_not_flagged():
    live = (
        "from tools.future.repro_science import build\n"
        "def test_repro():\n"
        "    out = build()\n"
        "    assert out.exists()\n"
    )
    assert rs.is_receipt_reader_suite(live) is False


def test_mutation_makes_live_verification_fail_and_dead_suite_is_caught():
    proof = rs.run_mutation_canaries()
    assert proof["live_verification_fails_on_mutation"] is True
    assert proof["dead_verification_stayed_green"] is True
    assert proof["dead_verification_caught"] is True
    assert proof["receipt_reader_suite_flagged"] is True
    assert proof["leftover_negative_control_caught"] is True
    assert proof["this_module_has_no_leftover_canary"] is True


def test_leftover_canary_detected_on_injected_source():
    mutated = "def add(a, b):\n    return a - b  # " + rs.CANARY_MARKER + "\n"
    assert rs.leftover_canary_present(mutated) is True
    assert rs.leftover_canary_present(Path(rs.__file__).read_text()) is False


def test_replication_bundle_complete_and_incomplete_refuses():
    w = rs.healthy_world()
    bundle = rs.make_replication_bundle(
        experiment_identity_value=w.experiment_identity,
        inputs=[{"name": "specimen", "sha256": w.specimen_hash, "role": "specimen"}],
        code_identity=w.code_sha256,
        machine_genome_pin=w.machine_genome,
    )
    rs.assert_bundle_complete(bundle)
    broken = dict(bundle)
    del broken["recipe_steps"]
    with pytest.raises(rs.FailClosed) as ei:
        rs.assert_bundle_complete(broken)
    assert ei.value.fault == "incomplete_replication_bundle"


def test_healthy_world_admits():
    assert rs.admit(rs.healthy_world()) == "ADMITTED"


@pytest.mark.parametrize("fault", rs.FAULT_NAMES)
def test_each_fault_is_detected_and_resume_is_safe(fault):
    healthy = rs.healthy_world()
    snap = rs.checkpoint(healthy)
    tainted = rs.inject(fault, healthy)
    with pytest.raises(rs.FailClosed) as ei:
        rs.admit(tainted)
    assert ei.value.fault == fault, f"expected {fault}, got {ei.value.fault}: {ei.value.reason}"
    assert ei.value.reason
    restored = rs.resume(snap)
    assert rs.admit(restored) == "ADMITTED"


def test_negative_control_all_twelve_faults_detected():
    """The whole fault-injection suite IS the negative control.

    Every named fault must fire a refusal with that exact name. A suite that
    reports PASS because it never injected anything has not watched the guard fail.
    """
    rows = rs.run_fault_suite()
    names = [r["fault"] for r in rows]
    assert names == list(rs.FAULT_NAMES)
    assert len(names) == 12
    missed = [r["fault"] for r in rows if not r["detected"]]
    assert missed == [], f"guards did not fire: {missed}"
    unresumed = [r["fault"] for r in rows if not r["resumed"]]
    assert unresumed == [], f"resume failed: {unresumed}"
    for row in rows:
        assert row["injected"] is True
        assert row["matched"] == row["fault"]
        assert row["reason"], f"{row['fault']} refused without a reason"


def test_skip_is_not_pass():
    with pytest.raises(rs.FailClosed) as ei:
        rs.finalize_verdict("SKIP")
    assert ei.value.fault == "skip_as_pass"
    with pytest.raises(rs.FailClosed):
        rs.finalize_verdict("skipped")
    assert rs.finalize_verdict("PASS") == "PASS"
    assert rs.finalize_verdict("FAIL") == "FAIL"
    assert rs.finalize_verdict("REFUSED") == "REFUSED"
    w = rs.healthy_world()
    w.proposed_verdict = "SKIP"
    with pytest.raises(rs.FailClosed) as ei:
        rs.admit(w)
    assert ei.value.fault == "skip_as_pass"


def test_invalidate_parent_downgrades_grandchild():
    led = rs.new_ledger()
    rs.ledger_add(led, "E_parent", "evidence")
    rs.ledger_add(led, "E_other", "evidence")
    rs.ledger_add(led, "C_child", "claim")
    rs.ledger_add(led, "C_grandchild", "claim")
    rs.ledger_add(led, "C_unrelated", "claim")
    rs.ledger_link(led, "C_child", "E_parent")
    rs.ledger_link(led, "C_grandchild", "C_child")
    rs.ledger_link(led, "C_unrelated", "E_other")
    assert rs.ledger_status(led, "C_grandchild") == "VALID"
    after = rs.ledger_invalidate(led, "E_parent")
    assert after["E_parent"] == "INVALID"
    assert after["C_child"] == "DOWNGRADED"
    assert after["C_grandchild"] == "DOWNGRADED"
    assert after["C_unrelated"] == "VALID"
    assert after["E_other"] == "VALID"
    proof = rs.transitive_downgrade_proof()
    assert proof["transitivity_holds"] is True


def test_physical_killed_subprocess_fail_closes():
    proof = rs.physical_killed_subprocess_proof()
    assert proof["detected"] is True
    assert proof["physical"] is True
    assert proof["resumed"] is True
    assert proof["fault"] == "killed_subprocess"


def test_corrupt_receipt_is_a_broken_seal_not_a_wrong_field_name():
    w = rs.healthy_world()
    assert rs.seal_is_valid(w.receipt)
    tainted = rs.inject("corrupt_receipt", w)
    assert not rs.seal_is_valid(tainted.receipt)
    with pytest.raises(rs.FailClosed) as ei:
        rs.admit(tainted)
    assert ei.value.fault == "corrupt_receipt"


def test_write_receipt_still_rejects_hardware_numbers():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "MUST_NOT_EXIST.json",
            {"schema": "nope", "tps": 12.0},
            "tools/future/test_repro_science.py",
        )
    assert not (RECEIPTS / "MUST_NOT_EXIST.json").exists()
