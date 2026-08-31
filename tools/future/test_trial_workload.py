"""Negative controls for the frozen-autonomy trial workload composer.

A composer nobody has watched refuse is a composer that will silently pad a
trial with copies. These tests prove it can reject an unknown trial id, an
unbound frontier item, a duplicate work identity, a 3h/6h set with no replan
pair (naming what is missing), and a GPU unit leaked as pending.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import adaptive_verification as av
from tools.future import autonomy_trial as at
from tools.future import meta_funnel as mf
from tools.future import negative_index as ni
from tools.future import odyssey_launch as ol
from tools.future import orchestration as orch
from tools.future import phase_listeners as pl
from tools.future import specimen_verify as sv
from tools.future import trial_workload as tw
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)


@pytest.fixture(scope="module")
def book():
    return tw.load_book()


def _clear(_proposal: dict) -> None:
    return None


def test_unknown_trial_id_is_refused_not_defaulted():
    with pytest.raises(tw.WorkloadRefused, match="unknown trial_id") as excinfo:
        tw.mix("12h")
    assert "trial_id" in excinfo.value.missing
    with pytest.raises(tw.WorkloadRefused, match="unknown trial_id"):
        tw.compose("")
    with pytest.raises(tw.WorkloadRefused, match="unknown trial_id"):
        tw.compose("not-a-trial")


def test_select_falcon_refuses_a_substitute(book):
    with pytest.raises(tw.WorkloadRefused, match="Falcon-H1-7B") as excinfo:
        tw.select_falcon([])
    assert tw.ROLE_FAST in excinfo.value.missing
    with pytest.raises(tw.WorkloadRefused, match="Falcon"):
        tw.select_falcon(["Qwen--Qwen3-30B-A3B@ad44e777bcd1"])
    with pytest.raises(tw.WorkloadRefused, match="LONG_SPECIMEN_VERIFY"):
        tw.select_long(["tiiuae--Falcon-H1-7B-Instruct@41e72f27effb"])


def test_select_falcon_copes_with_live_listing_or_absence():
    """No skip: lake mounted or not, the helper either finds Falcon or refuses."""
    names = sv.list_specimens()
    try:
        row = tw.select_falcon(names)
    except tw.WorkloadRefused as exc:
        assert "Falcon" in str(exc)
        assert tw.ROLE_FAST in exc.missing
        return
    assert "falcon-h1" in row["name"].lower().replace("_", "-")


def test_mix_longer_trials_require_the_declared_roles_and_replan():
    for tid in ("15m", "1h", "3h", "6h"):
        spec = tw.mix(tid)
        assert spec["trial_id"] == tid
        assert spec["duration_s"] == at.TRIAL_DURATION_S[tid]
        assert spec["gpu_authority"] is False
        assert spec["evidence_class"] == "STATIC_ONLY"
        assert tw.ROLE_FAST in spec["required_roles"]
        assert tw.ROLE_NEGATIVE in spec["required_roles"]
        assert spec["replan_required"] is (tid in tw.LONGER_TRIALS)
    long = tw.mix("3h")
    for role in (
        tw.ROLE_FAST,
        tw.ROLE_LONG,
        tw.ROLE_NEGATIVE,
        tw.ROLE_SCREEN,
        tw.ROLE_HCLI,
        tw.ROLE_O2,
        tw.ROLE_O3,
    ):
        assert role in long["required_roles"]
        assert long["proportions"][role]["min_units"] == 1
        assert long["proportions"][role]["why"]
    assert tw.mix("15m")["replan_required"] is False
    assert tw.ROLE_LONG not in tw.mix("15m")["required_roles"]
    assert tw.ROLE_LONG not in tw.mix("1h")["required_roles"]


def test_compose_15m_is_bound_mixed_and_has_no_long_unit(book):
    try:
        doc = tw.compose("15m", book=book)
    except tw.WorkloadRefused as exc:
        # Lake unmounted: FAST cannot be substituted. That is coping, not a skip.
        assert "Falcon" in str(exc)
        return
    assert doc["admitted"] is True
    assert doc["gpu_authority"] is False
    roles = {u["mix_role"] for u in doc["units"]}
    assert tw.ROLE_FAST in roles
    assert tw.ROLE_NEGATIVE in roles
    assert tw.ROLE_HCLI in roles
    assert tw.ROLE_LONG not in roles
    falcon = [u for u in doc["units"] if u.get("mix_role") == tw.ROLE_FAST]
    assert falcon and "falcon-h1" in str(falcon[0].get("specimen") or falcon[0]["description"]).lower()
    for unit in doc["units"]:
        assert unit["module"] in orch.BINDINGS
        assert orch.BINDINGS[unit["module"]][0] == unit["frontier_id"]
        assert tw._item_by_id(book, unit["frontier_id"]) is not None
        assert unit["gpu_authority"] is False
        assert unit["worth_doing_anyway"]
        ident = at.work_identity(unit)
        assert ident[0] and ident[1] and ident[3]


def test_compose_3h_and_6h_carry_frontier_derived_replan_pairs(book):
    try:
        three = tw.compose("3h", book=book)
        six = tw.compose("6h", book=book)
    except tw.WorkloadRefused as exc:
        assert "Falcon" in str(exc) or "LONG" in str(exc)
        return
    for doc, tid in ((three, "3h"), (six, "6h")):
        assert doc["admitted"] is True
        assert doc["n_replan_pairs"] >= 1
        assert doc["mix"]["replan_required"] is True
        roles = {u["mix_role"] for u in doc["units"]}
        for role in tw.mix(tid)["required_roles"]:
            assert role in roles
        long_units = [u for u in doc["units"] if u.get("mix_role") == tw.ROLE_LONG]
        assert long_units
        long_name = str(long_units[0].get("specimen") or "")
        assert "falcon-h1" not in long_name.lower()
        ids = {u["frontier_id"] for u in doc["units"]}
        for pair in doc["replan_pairs"]:
            assert pair["cause_frontier_id"] in ids
            assert pair["effect_frontier_id"] in ids
            assert pair["how"]
            assert pair["derived_from"]
            assert tw._item_by_id(book, pair["cause_frontier_id"])
            assert tw._item_by_id(book, pair["effect_frontier_id"])
            assert pair["cause_id"] != pair["effect_id"]


def test_3h_workload_with_no_replan_pair_is_refused_naming_the_gap(book):
    """NEGATIVE CONTROL: 3h/6h without a replan pair must not look admitted."""
    fillers = (
        ("freshness.py", tw.ROLE_FAST),
        ("hmf_objects.py", tw.ROLE_LONG),
        ("turnaround.py", tw.ROLE_NEGATIVE),
        ("ane_preboard.py", tw.ROLE_SCREEN),
        ("decode_civilization.py", tw.ROLE_HCLI),
        ("hbm_doctor.py", tw.ROLE_O2),
        ("fusion_sim.py", tw.ROLE_O3),
    )
    units = []
    for module, role in fillers:
        units.append(
            tw.make_unit(
                module,
                description=(
                    f"advance {orch.BINDINGS[module][0]} by running {module} "
                    "as isolated CPU science that does not change another unit's priority"
                ),
                mix_role=role,
                book=book,
                why_worth_doing="each of these is real bound CPU work; together they form no replan pair",
            )
        )
    pairs = tw.replan_pairs(units, book)
    assert pairs == [], pairs
    with pytest.raises(tw.WorkloadRefused, match="replan pair") as excinfo:
        tw.admit_workload(units, "3h", book=book)
    assert "replan_pair" in excinfo.value.missing
    with pytest.raises(tw.WorkloadRefused, match="replan pair"):
        tw.admit_workload(units, "6h", book=book)
    # 15m does not require a replan pair, but it does require its own roles.
    # The filler roles are not 15m's. Construct a 15m-shaped set with no pair.
    short = [
        tw.make_unit(
            "freshness.py",
            description="resync derived artifacts with semantic fingerprints, not sha-only",
            mix_role=tw.ROLE_FAST,
            book=book,
            why_worth_doing="derived freshness is F017 CPU work and stands as a receipt",
        ),
        tw.make_unit(
            "hmf_objects.py",
            description="audit HMF managed-object legal transitions; tri-state coherence must not collapse to a boolean",
            mix_role=tw.ROLE_NEGATIVE,
            book=book,
            why_worth_doing="HMF transition audit is memory-frontier work",
        ),
        tw.make_unit(
            "turnaround.py",
            description="remeasure CPU-side experiment-turnaround phases; leave GPU phases UNKNOWN",
            mix_role=tw.ROLE_HCLI,
            book=book,
            why_worth_doing="CPU turnaround phases are the honest latency work on this host",
        ),
    ]
    assert tw.replan_pairs(short, book) == []
    ok = tw.admit_workload(short, "15m", book=book)
    assert ok["admitted"] is True
    assert ok["n_replan_pairs"] == 0


def test_unbound_frontier_item_is_rejected(book):
    unit = tw.make_unit(
        "freshness.py",
        description="resync derived artifacts with semantic fingerprints, not sha-only",
        mix_role=tw.ROLE_FAST,
        book=book,
        why_worth_doing="F017 CPU work",
    )
    unit["frontier_id"] = "FT.FAKE.not-in-the-book"
    with pytest.raises(tw.WorkloadRefused, match="not bound to a real frontier item") as excinfo:
        tw.admit_unit(unit, book=book)
    assert "frontier_item" in excinfo.value.missing
    unit["frontier_id"] = "FT.MEMORY.hmf"
    with pytest.raises(tw.WorkloadRefused, match="mismatched"):
        tw.admit_unit(unit, book=book)
    ghost = dict(unit)
    ghost["frontier_id"] = orch.BINDINGS["freshness.py"][0]
    ghost["module"] = "not_a_real_module.py"
    ghost["capability"] = "not_a_real_module.py"
    with pytest.raises(tw.WorkloadRefused, match="BINDINGS"):
        tw.admit_unit(ghost, book=book)


def test_duplicate_work_identity_is_rejected_even_when_ids_differ(book):
    a = tw.make_unit(
        "negative_index.py",
        description="rebuild the scar index that prunes work before it is scheduled",
        mix_role=tw.ROLE_NEGATIVE,
        book=book,
        why_worth_doing="scar index is campaign next work",
    )
    b = dict(a)
    b["id"] = a["id"] + ".copy"
    assert a["id"] != b["id"]
    assert at.work_identity(a) == at.work_identity(b)
    with pytest.raises(tw.WorkloadRefused, match="duplicate work identity") as excinfo:
        tw.admit_workload([a, b], "15m", book=book)
    assert "distinct_work" in excinfo.value.missing


def test_gpu_unit_is_emitted_sleeping_not_pending(book):
    unit = tw.make_unit(
        "teacher_corpus.py",
        description=(
            "Flash teacher capture is blocked on Metal; this unit exists so the "
            "composer must park it rather than invent a corpus"
        ),
        mix_role="GPU_PARKED",
        book=book,
        why_worth_doing="naming the blocked capture is honest; running it is not",
    )
    assert unit["status"] == "blocked"
    assert unit["classification"] == "SLEEPING"
    assert unit["resource_class"] in at.GPU_RESOURCE
    assert unit["gpu_authority"] is False
    ok, why = at.is_valid_workunit(unit)
    assert ok is False
    assert "blocked" in why or "sleeping" in why or "GPU" in why
    leaked = dict(unit)
    leaked["status"] = "pending"
    parked = tw.admit_unit(leaked, book=book)
    assert parked["status"] == "blocked"
    assert parked["classification"] == "SLEEPING"


def test_padding_generic_description_is_refused(book):
    with pytest.raises(tw.WorkloadRefused, match="padding") as excinfo:
        tw.make_unit(
            "freshness.py",
            description="do work",
            mix_role=tw.ROLE_FAST,
            book=book,
            why_worth_doing="should never be admitted",
        )
    assert "worth_doing_anyway" in excinfo.value.missing


def test_bound_negative_science_capability_can_actually_refuse():
    """EXECUTED capability: naming refuse_if_dead is not evidence it fires."""
    dead = ni.refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert dead is not None
    assert dead["refused"] is True
    live = ni.refuse_if_dead(
        {
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "hwir_node_types",
        }
    )
    assert live is None


def test_bound_multi_fidelity_screen_can_kill_expensive_children():
    """EXECUTED capability: a cheap falsifier kill must not launch funnel gates."""
    called: list[str] = []
    funnel = mf.Funnel()
    original = funnel.advance
    advanced: list[str] = []

    def wrapped(candidate, gate):
        g = mf.resolve_gate(gate)
        advanced.append(g.name)
        return original(candidate, gate)

    funnel.advance = wrapped  # type: ignore[method-assign]
    cand = av._candidate(
        "trial.workload.kill.falsifier",
        cheapest="Kill if the cheap observation already fires.",
        observation={"fired": True, "mechanism": "cheap observation killed it"},
    )
    result = av.screen(cand, refuse_fn=_clear, funnel=funnel, on_stage=lambda s, _c: called.append(s.name))
    assert result["verdict"] == av.VERDICT_KILLED
    assert result["verified"] is False
    assert result["cost"]["funnel_gates_launched"] == 0
    assert advanced == []
    saved = av.saved(cand, screen_result=result)
    assert [u["gate_name"] for u in saved] == [g.name for g in mf.GATES]
    assert called == [av.FALSIFIER_NAME]


def test_odyssey_iii_listener_can_emit_zero_on_empty_store():
    """NEGATIVE CONTROL: spawn is not a default. Empty Phase I emits nothing."""
    store = pl.load_law_store(laws=[])
    assert store["n"] == 0
    assert store["reason_code"] == pl.EMPTY_STORE
    q = pl.qualifying_laws(laws=[])
    assert q["n"] == 0
    assert q["laws"] == []


def test_catalog_replan_edges_are_derived_from_live_bindings(book):
    edges = tw.catalog_replan_edges(book)
    assert edges, "the live catalog must yield at least one priority-changing evidence edge"
    for edge in edges:
        assert tw._item_by_id(book, edge["cause_frontier_id"])
        assert tw._item_by_id(book, edge["effect_frontier_id"])
        assert edge["cause_module"] in orch.BINDINGS
        assert orch.BINDINGS[edge["cause_module"]][0] == edge["cause_frontier_id"]
        rec = tw._receipt_literal(edge["cause_module"])
        assert rec == edge["evidence_receipt"]
    mechanisms = tw.recovered_mechanism_edges(book)
    assert any(e["cause_module"] == "adaptive_verification.py" for e in mechanisms)
    assert any(e["cause_module"] == "odyssey2_law_store.py" for e in mechanisms)
    kids = av.funnel_child_workunits()
    assert len(kids) == len(mf.GATES) == 9
    assert "specimen_curriculum_ready" in ol.CRITERION_IDS


def test_compose_does_not_pad_with_capability_cycling(book):
    try:
        doc = tw.compose("3h", book=book)
    except tw.WorkloadRefused as exc:
        assert "Falcon" in str(exc) or "LONG" in str(exc)
        return
    modules = [u["module"] for u in doc["units"]]
    assert "pytest" not in " ".join(modules)
    assert modules.count("specimen_verify.py") == 2
    identities = [at.work_identity(u) for u in doc["units"]]
    assert len(identities) == len(set(identities))
    flood = at.busywork_flood(doc["units"])
    assert flood["flood"] is False


def test_build_emits_sealed_static_only_receipt(book):
    out = tw.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == tw.RECEIPT
    assert doc["schema"] == tw.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        if key in doc:
            assert not isinstance(doc[key], (int, float))
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == f"receipts/future/{tw.RECEIPT}"
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert set(doc["trial_ids"]) == set(at.TRIAL_IDS)
    for tid in at.TRIAL_IDS:
        row = doc["composed"][tid]
        if row.get("admitted"):
            assert row["n_units"] >= 3
            if tid in tw.LONGER_TRIALS:
                assert row["n_replan_pairs"] >= 1


def test_receipt_rejects_a_hardware_number():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 1})


def test_module_parses_and_has_no_stubs():
    src = Path(tw.__file__).read_text()
    ast.parse(src)
    for needle in ("raise NotImplementedError", "TODO", "\n    pass\n"):
        assert needle not in src
