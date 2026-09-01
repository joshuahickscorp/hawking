"""Tests for the 22-frontier operating layer.

Negative controls are load-bearing: next_work must still return safe work
when GPU_PROTECTED and ANE are blocked; is_idle must be false against the
real current book; a redundant low-information unit is refused while a
novel one is admitted.
"""
from __future__ import annotations

import json

import pytest

from tools.future import frontiers as fr
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims
from tools.future.workunit_species import HCLI_CORE_FIELDS


@pytest.fixture(scope="module")
def book():
    return fr.load_book()


@pytest.fixture(scope="module")
def receipt_path(book):
    return fr.build(available_lanes=fr.THIS_HOST_LANES, book=book)


@pytest.fixture(scope="module")
def doc(receipt_path):
    return json.loads(receipt_path.read_text())


def test_entry_point_runs_and_seals_receipt(receipt_path, doc):
    assert receipt_path.parent == RECEIPTS
    assert receipt_path.name == "FRONTIER_STATE.json"
    assert doc["schema"] == "hawking.future.frontiers.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["produces_diagnostic_relative"] is False
    assert doc["produces_protected_absolute"] is False
    _assert_no_hardware_claims(doc)


def test_twenty_two_named_frontiers_are_persistent(book, doc):
    names = list(fr.FRONTIER_NAMES)
    assert names == list(doc["frontier_names"])
    assert len(names) == len(set(names))
    assert set(doc["frontiers"]) == set(names)
    assert doc["n_frontiers"] == len(names)
    for name in names:
        row = doc["frontiers"][name]
        assert row["name"] == name
        assert row["status"] in {"ACTIVE", "BLOCKED", "EXHAUSTED"}
        assert "open_questions" in row
        assert "blocked" in row
        assert "next_work" in row
        assert set(row["counts"]) >= {"open_questions", "blocked", "next_work"}
        # Counts are derived from the lists, not pinned.
        assert row["counts"]["open_questions"] == len(row["open_questions"])
        assert row["counts"]["blocked"] == len(row["blocked"])
        assert row["counts"]["next_work"] == len(row["next_work"])


def test_no_era_vi_or_odyssey_iv(doc):
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "VI" not in doc["eras"]
    assert all("IV" not in o for o in doc["odysseys"])


def test_next_work_still_returns_safe_work_when_gpu_and_ane_blocked(book):
    """Negative control: a blocked hardware lane must not idle the daemon."""
    available = [lane for lane in fr.ALL_LANES if lane not in {fr.LANE_GPU_PROTECTED, fr.LANE_ANE}]
    units = book.next_work(available)
    assert units, "next_work() returned nothing while GPU_PROTECTED and ANE are blocked"
    kinds = set()
    for unit in units:
        req = set(unit.get("required_lanes") or [])
        assert fr.LANE_GPU_PROTECTED not in req, unit["id"]
        assert fr.LANE_ANE not in req, unit["id"]
        assert req <= set(available)
        assert unit.get("resource_class") != "GPU_EXCLUSIVE"
        assert unit.get("status") == "pending"
        assert unit.get("classification") != "SLEEPING"
        assert unit.get("evidence_class") == "STATIC_ONLY"
        assert unit.get("bench_state") == "UNKNOWN"
        missing = [name for name in HCLI_CORE_FIELDS if name not in unit]
        assert not missing, f"{unit['id']} missing HCLI fields {missing}"
        kinds.add(unit.get("frontier"))
    # Must still yield CPU analysis, simulation, representation, tooling, Odyssey.
    cpu_frontiers = kinds & {
        "TOOLS",
        "ODYSSEY_TRANSFER",
        "ODYSSEY_ADVERSARY",
        "MODEL_REPRESENTATION",
        "PHYSICAL_GRAPH",
        "FPGA",
        "VERIFICATION",
        "DECODING",
        "HCLI_SELF",
        "CHILD_RESIDENT",
        "CONTEXT",
        "MEMORY",
        "EXPERIMENT_TURNAROUND",
        "MODEL_EXECUTION",
        "MODEL_CAPABILITY",
        "ACTIVE_BYTES",
        "GPU_KERNELS",
        "STATE",
        "LATENCY",
        "TPS",
        "ANE",
        "ARCHITECTURE_REPATRIATION",
    }
    assert cpu_frontiers, f"no CPU-class frontiers in next_work: {sorted(kinds)}"
    assert kinds & {"TOOLS", "ODYSSEY_TRANSFER", "ODYSSEY_ADVERSARY", "MODEL_REPRESENTATION"}


def test_next_work_functional_api_matches_book(book):
    a = book.next_work(fr.THIS_HOST_LANES)
    b = fr.next_work(fr.THIS_HOST_LANES, book=book)
    assert [u["id"] for u in a] == [u["id"] for u in b]
    assert a  # real current frontier has CPU work


def test_is_idle_is_false_against_the_real_current_frontier(book, doc):
    """Negative control: the §140 failure condition is false right now."""
    assert book.is_idle() is False
    assert fr.is_idle(book=book) is False
    assert doc["is_idle"] is False
    proof = book.idle_proof()
    assert proof["is_idle"] is False
    assert proof["n_active"] >= 1
    assert proof["active"]
    # Hardware-blocked frontiers must not drag the whole book to idle.
    assert set(proof["blocked"]) <= set(fr.FRONTIER_NAMES)


def test_sleeping_gpu_and_ane_units_carry_wake_conditions(book):
    sleeping = book.sleeping_units()
    assert sleeping, "hardware work must exist as SLEEPING units, not vanish"
    gpu = [u for u in sleeping if fr.LANE_GPU_PROTECTED in (u.get("required_lanes") or [])]
    ane = [u for u in sleeping if fr.LANE_ANE in (u.get("required_lanes") or [])]
    assert gpu, "GPU_PROTECTED work must sleep, not disappear"
    assert ane, "ANE work must sleep, not disappear"
    for unit in gpu + ane:
        assert unit["status"] == "blocked"
        assert unit["classification"] == "SLEEPING"
        wake = unit.get("wake_condition") or {}
        assert wake.get("all_of"), unit["id"]
        never = wake.get("never") or []
        joined = " ".join(never).lower()
        assert "synthetic" in joined
        assert unit.get("blocked_reason")
        # Ready-protected identities are derived, not a pinned integer.
        if unit["id"] == "FT.GPU_KERNELS.ready-protected":
            ids = unit.get("ready_protected_ids") or []
            assert ids, "identity set must be recovered from disk/snapshot"
            assert unit.get("n_ready_protected") == len(ids)


def test_redundant_low_information_unit_is_refused_and_novel_is_admitted(book):
    """Negative control: the busywork guard has to actually fire."""
    open_items = [
        i
        for i in book.items
        if i["kind"] == "NEXT_WORK" and not fr._item_sleeping(i, book.wake)
    ]
    assert open_items
    original = dict(open_items[0])
    clone = dict(original)
    clone["id"] = original["id"] + ".copy"
    clone["expected_information_gain"] = fr.INFO_LOW
    refused = fr.admit(
        clone,
        queued=open_items,
        book_items=book.items,
        scar_doc=book.scar_doc,
    )
    assert refused["admitted"] is False
    assert refused["refused"] is True
    assert "redundant" in refused["reason"] or "low-information" in refused["reason"]
    assert refused["redundancy"] >= fr.REDUNDANCY_LOW_GAIN

    novel = {
        "id": "FT.CHILD_RESIDENT.novel-identity-hash-mismatch-policy",
        "frontier": "CHILD_RESIDENT",
        "title": "Bind a child-resident identity receipt against the install contract without launching",
        "detail": "Targets identity-hash mismatch policy only; distinct from the catalog dry-run.",
        "hypothesis_family": "child_resident_identity_hash_mismatch_policy",
        "expected_information_gain": fr.INFO_HIGH,
        "description": "Novel identity-mismatch policy dry-run for a child resident, no process launch",
    }
    accepted = fr.admit(
        novel,
        queued=open_items,
        book_items=book.items,
        scar_doc=book.scar_doc,
    )
    assert accepted["admitted"] is True, accepted
    assert accepted["refused"] is False
    assert accepted["expected_information_gain"] >= fr.INFO_MEDIUM


def test_scar_dead_hypothesis_is_refused_at_admission(book):
    dead = {
        "id": "FT.VERIFICATION.replay-cross-expert",
        "frontier": "VERIFICATION",
        "title": "Retry trivial global expert sharing on qwen3-80b",
        "hypothesis_family": "cross_expert_structure",
        "model": "qwen3-80b",
        "expected_information_gain": fr.INFO_HIGH,
        "description": "rediscover cross-expert structure on qwen80",
    }
    decision = fr.admit(
        dead,
        queued=[],
        book_items=book.items,
        scar_doc=book.scar_doc,
    )
    if book.scar_doc is None:
        pytest.skip("NEGATIVE_SCIENCE_INDEX not visible in this checkout; admit coped")
    assert decision["admitted"] is False
    assert decision["scar_overlap"]
    assert decision["scar_overlap"].get("refused") is True
    assert "dead" in decision["reason"].lower() or "rediscovery" in decision["reason"].lower()


def test_zero_gain_is_refused():
    decision = fr.admit(
        {"id": "busywork.empty", "title": "nudge", "expected_information_gain": 0},
        queued=[],
        book_items=[],
        scar_doc=None,
    )
    assert decision["admitted"] is False
    assert "zero" in decision["reason"] or "busywork" in decision["reason"]


def test_movement_is_unknown_until_a_real_verified_move(book, doc):
    mov = book.movement()
    assert mov["state"] == "UNKNOWN"
    assert mov["n_moves"] == 0
    assert mov["moves"] == []
    assert mov["supporting_rates"]["moves_per_wall_second"] == "UNKNOWN"
    assert doc["movement"]["state"] == "UNKNOWN"
    with pytest.raises(fr.UnverifiedMoveError):
        book.record_move(
            frontier="TOOLS",
            before={"open": 3},
            after={"open": 1},
            evidence=["receipts/future/FRONTIER_STATE.json"],
            verified=False,
        )
    with pytest.raises(fr.DominatedMoveError):
        book.record_move(
            frontier="TOOLS",
            before={"open": 1},
            after={"open": 3},
            evidence=["receipts/future/FRONTIER_STATE.json"],
            verified=True,
        )
    with pytest.raises(HardwareClaimError):
        book.record_move(
            frontier="TPS",
            before={"tps": 1},
            after={"tps": 2},
            evidence=["synthetic"],
            verified=True,
        )
    # Recording a real move is allowed for the API, but build() does not seed
    # a baseline. Isolate so the module-scoped book stays UNKNOWN for other tests.
    isolated = fr.load_book()
    isolated.record_move(
        frontier="TOOLS",
        before={"open": 4, "resolved": 0},
        after={"open": 3, "resolved": 1},
        evidence=["receipts/future/CLAUDE_GLOBAL_FRONTIER.json"],
        verified=True,
        higher_better=("resolved",),
    )
    assert isolated.movement()["n_moves"] == 1
    assert isolated.movement()["supporting_rates"]["moves_per_wall_second"] == "UNKNOWN"


def test_copes_with_unseen_handoff_and_queue():
    """Sparse checkout: unseen is a recovery path, not an absence assertion."""
    book = fr.load_book(
        overrides={
            "CODEX_ACCELERATOR_HANDOFF.json": None,
            "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json": None,
        }
    )
    taken = book.recovery["paths_taken"]
    assert taken["CODEX_ACCELERATOR_HANDOFF.json"]
    assert "unseen" in taken["CODEX_ACCELERATOR_HANDOFF.json"] or taken[
        "CODEX_ACCELERATOR_HANDOFF.json"
    ].startswith("override")
    assert len(book.frontiers()) == len(fr.FRONTIER_NAMES)
    units = book.next_work(fr.THIS_HOST_LANES)
    assert units
    assert book.is_idle() is False
    # Identity set still recovered from a snapshot, not assumed empty-because-unseen.
    assert book.queue_identity["ready_protected_ids"]


def test_global_frontier_is_consumed_not_rewritten(doc):
    consumed = doc["global_frontier_consumed"]
    assert consumed["path"] == "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
    assert consumed["n_entries"] >= 1
    # The campaign tracker receipt must still be the one global_frontier.py owns.
    tracker = json.loads((RECEIPTS / "CLAUDE_GLOBAL_FRONTIER.json").read_text())
    assert tracker["schema"] == "hawking.future.claude_global_frontier.v1"


def test_resident_callable_names_fail_closed_path(doc):
    rc = doc["resident_callable"]
    assert rc["can_hcli_invoke"] is True
    assert rc["entry_point"]
    assert "WorkUnit" in rc["workunit_emitted"]
    assert rc["receipt_written"].endswith("FRONTIER_STATE.json")
    assert "twenty-two" in rc["frontier_fed"] or "22" in rc["frontier_fed"]
    assert rc["fail_closed"]
    assert any("SLEEPING" in x or "synthetic" in x.lower() for x in rc["fail_closed"])
    assert "wakeup.py" in rc["integration_swaps"]
    assert "resident_api.py" in rc["integration_swaps"]


def test_selftest_seals_and_proves_the_negative_controls(book):
    out = fr.selftest()
    assert out.name == "FRONTIER_STATE.json"
    body = json.loads(out.read_text())
    assert body["is_idle"] is False
    assert body["n_next_work"] >= 1
    for unit in body["next_work"]:
        req = set(unit.get("required_lanes") or [])
        assert fr.LANE_GPU_PROTECTED not in req
        assert fr.LANE_ANE not in req


def test_refill_never_empty_while_active(book):
    first = book.next_work(fr.THIS_HOST_LANES)
    again = book.refill(fr.THIS_HOST_LANES)
    assert first and again
    assert [u["id"] for u in first] == [u["id"] for u in again]
    assert book.is_idle() is False


def test_emitted_units_round_trip_hcli_shape(book):
    units = book.next_work(fr.THIS_HOST_LANES)
    from hcli.workunit import WorkUnit

    for unit in units[:5]:
        wu = WorkUnit.from_dict(dict(unit))
        assert wu.id == unit["id"]
        assert wu.verifier


def test_no_numeric_hardware_fields_in_receipt(doc):
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in HARDWARE_FIELDS:
                    assert not isinstance(value, (int, float)), here
                walk(value, here)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(doc)


def test_counts_are_derived_from_recovered_identities(book, doc):
    ident = book.queue_identity
    assert ident["n_ready_protected"] == len(ident["ready_protected_ids"])
    assert ident["n_blocked"] == len(ident["blocked_ids"])
    assert doc["n_next_work"] == len(doc["next_work"])
    assert doc["n_sleeping"] == len(doc["sleeping"])
    proof = doc["idle_proof"]
    assert proof["n_active"] == len(proof["active"])
    assert proof["n_blocked"] == len(proof["blocked"])
    assert proof["n_exhausted"] == len(proof["exhausted"])
    assert proof["n_frontiers"] == len(fr.FRONTIER_NAMES)
    assert proof["n_active"] + proof["n_blocked"] + proof["n_exhausted"] == len(fr.FRONTIER_NAMES)


def test_a_dispatch_reordering_unit_is_refused_at_proposal_time():
    """S026 §4: the scar must bite before GPU time is spent, not after."""
    with pytest.raises(fr.DeadSchoolRefused, match="zero overlapable"):
        fr._item(
            id="FT.MODEL_EXECUTION.test.reorder",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Permute the dispatch order of the token graph",
            detail="sweep command-order permutations for overlap",
            required_lanes=(),
            gain=5,
            species="probe",
            verifier="x",
            evidence=(),
        )


def test_the_refusal_reads_the_hypothesis_family_too():
    with pytest.raises(fr.DeadSchoolRefused):
        fr._item(
            id="FT.MODEL_EXECUTION.test.family",
            frontier="MODEL_EXECUTION",
            kind="NEXT_WORK",
            title="Try a different launch shape",
            detail="nothing suspicious in this text",
            required_lanes=(),
            gain=5,
            species="probe",
            verifier="x",
            evidence=(),
            hypothesis_family="top-level overlap",
        )


def test_asking_whether_the_scar_still_holds_is_not_refused():
    """It closes a school, not a word. Questions stay legal."""
    item = fr._item(
        id="FT.MODEL_EXECUTION.test.question",
        frontier="MODEL_EXECUTION",
        kind="OPEN_QUESTION",
        title="Does dispatch reordering have slack once drafting exists?",
        detail="the scar's own reopen condition names speculative drafting",
        required_lanes=(),
        gain=5,
        species="question",
        verifier="x",
        evidence=(),
    )
    assert item["id"].endswith("test.question")


def test_the_live_catalog_contains_no_unit_from_the_dead_school():
    assert len(fr._catalog()) > 0
