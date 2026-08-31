"""Claim scope: widening without a named replicating specimen must RAISE.

Acceptance the rest of the module is worthless without:

* a law carries its available-specimen set and a three-axis scope tier
* widen() cannot broaden MODEL_LOCAL / ORGAN_LOCAL / MACHINE_LOCAL without
  a named replicating specimen
* every experiment binds the six identity fields
* at least one existing campaign law is retro-scoped, and any over-broad
  statement is narrowed with the narrowing recorded
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import claim_scope as cs
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


PARENT = cs.PARENT_SPECIMEN
FALCON = "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb"
FLASH = "Qwen--Qwen3.8-Flash-Next@34567a4712bc"


def _identity(**kwargs):
    base = {
        "specimen_seal": {"specimen": PARENT, "tree_digest": "ab" * 32},
        "model_revision": {"model_id": cs.PARENT_MODEL_ID},
        "resident_identity": {"resident_identity": cs.PARENT_RESIDENT},
        "code_and_build_identity": {"git_head": "deadbeef"},
        "machine_genome": {"receipt": cs.MACHINE_GENOME_REL, "remeasured": False},
        "laws_scars_version": {"digest": "cd" * 32},
    }
    base.update(kwargs)
    return cs.bind_experiment(**base)


def _timeline(*sealed_rows, in_flight=()):
    specs = {}
    sealed = []
    for row in sealed_rows:
        specs[row["specimen"]] = dict(row)
        if row.get("sealed"):
            sealed.append(row)
    return {
        "lake": "/fixture",
        "lake_mounted": False,
        "n_known": len(specs),
        "n_sealed": len(sealed),
        "n_in_flight": len(in_flight),
        "specimens": specs,
        "sealed": sealed,
        "in_flight": list(in_flight),
        "rule": "fixture timeline for unit tests; production uses load_timeline()",
    }


def _sealed(name, at, **extra):
    row = {
        "specimen": name,
        "sealed": True,
        "sealed_at": at,
        "sealed_at_source": "fixture",
        "status": "SEALED",
        "owner": extra.pop("owner", "modellake"),
    }
    row.update(extra)
    return row


def _claim(**kwargs) -> cs.Claim:
    ident = kwargs.pop("experiment_identity", None) or _identity()
    tested = kwargs.pop("tested_specimens", (PARENT,))
    as_of = kwargs.pop("as_of", "2026-08-30T16:00:00Z")
    available = kwargs.pop("available_specimens", tested)
    tl = kwargs.pop("_timeline", None)
    defaults = dict(
        law_id="LAW-TEST",
        statement="a model-local observation used by tests",
        tested_specimens=tuple(tested),
        available_specimens=tuple(available),
        as_of=as_of,
        as_of_source="test fixture",
        scope=cs.narrow_scope(),
        original_scope=cs.narrow_scope(),
        scope_kind="ORIGINAL",
        organ="mlp",
        machine=cs.PARENT_MACHINE,
        parent=PARENT,
        evidence_refs=("receipts/future/MLP_ALU_ROOFLINE.json",),
        experiment_identity=ident,
    )
    defaults.update(kwargs)
    claim = cs.Claim(**defaults)
    if tl is not None:
        return cs.validate_claim(claim, timeline=tl)
    return claim


EARLY = _timeline(
    _sealed(PARENT, "2026-08-30T15:44:16Z", owner="local_directory", tree_digest="ab" * 32),
)
LATER = _timeline(
    _sealed(PARENT, "2026-08-30T15:44:16Z", owner="local_directory", tree_digest="ab" * 32),
    _sealed(FALCON, "2026-08-31T09:56:23Z"),
    _sealed(FLASH, "2026-08-31T09:56:23Z"),
)


def test_law_carries_available_set_and_scope_tier():
    """Acceptance: a law carries its available-specimen set and a scope tier."""
    claim = _claim(available_specimens=(PARENT,), _timeline=EARLY)
    claim = cs.validate_claim(claim, timeline=EARLY)
    d = claim.to_dict()
    assert d["available_specimens"] == [PARENT]
    assert d["tested_specimens"] == [PARENT]
    assert d["scope"] == {
        "model": "MODEL_LOCAL",
        "organ": "ORGAN_LOCAL",
        "machine": "MACHINE_LOCAL",
    }
    assert d["scope_kind"] == "ORIGINAL"
    assert d["evidence_universe"] == f"within currently sealed specimens {PARENT}"
    assert "all hawking" not in d["evidence_universe"].lower()


def test_scope_cannot_widen_without_named_replicating_specimen():
    """Acceptance: widen() RAISES. A flag is not a gate."""
    claim = cs.validate_claim(_claim(), timeline=EARLY)
    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(claim, "model", replicating_specimen=None, replication_receipt=None, timeline=EARLY)
    assert ei.value.reason == "no_named_replicating_specimen"
    assert claim.scope["model"] == "MODEL_LOCAL"

    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(claim, "model", replicating_specimen="", replication_receipt="r.json", timeline=EARLY)
    assert ei.value.reason == "no_named_replicating_specimen"

    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(
            claim,
            "model",
            replicating_specimen="all Hawking models",
            replication_receipt="r.json",
            timeline=EARLY,
        )
    assert ei.value.reason == "no_named_replicating_specimen"

    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(claim, "organ", replicating_specimen=None, replication_receipt="r.json", timeline=EARLY)
    assert ei.value.reason == "no_named_replicating_specimen"

    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(claim, "machine", replicating_specimen="  ", replication_receipt="r.json", timeline=EARLY)
    assert ei.value.reason == "no_named_replicating_specimen"


def test_named_specimen_without_receipt_is_not_a_replication():
    claim = cs.validate_claim(_claim(), timeline=LATER)
    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(
            claim,
            "model",
            replicating_specimen=FALCON,
            replication_receipt=None,
            timeline=LATER,
            at="2026-08-31T12:00:00Z",
        )
    assert ei.value.reason == "replication_has_no_receipt"
    assert claim.scope["model"] == "MODEL_LOCAL"


def test_unsealed_specimen_cannot_widen():
    claim = cs.validate_claim(_claim(), timeline=EARLY)
    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(
            claim,
            "model",
            replicating_specimen=FALCON,
            replication_receipt="receipts/future/FAKE_REPLICATION.json",
            timeline=EARLY,
            at="2026-08-30T16:00:00Z",
        )
    assert ei.value.reason == "replicating_specimen_not_sealed"


def test_original_tested_specimen_is_not_a_replicating_specimen():
    claim = cs.validate_claim(_claim(), timeline=LATER)
    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(
            claim,
            "model",
            replicating_specimen=PARENT,
            replication_receipt="receipts/future/FAKE.json",
            timeline=LATER,
            at="2026-08-31T12:00:00Z",
        )
    assert ei.value.reason == "replicating_specimen_already_in_original_scope"


def test_widen_with_named_sealed_specimen_and_receipt():
    claim = cs.validate_claim(_claim(), timeline=LATER)
    out = cs.widen(
        claim,
        "model",
        replicating_specimen=FALCON,
        replication_receipt="receipts/future/FAKE_FALCON_REPLICATION.json",
        timeline=LATER,
        at="2026-08-31T12:00:00Z",
    )
    assert out.scope["model"] == "MODEL_REPLICATED"
    assert out.scope["organ"] == "ORGAN_LOCAL"
    assert out.scope["machine"] == "MACHINE_LOCAL"
    assert out.scope_kind == "REPLICATION"
    assert FALCON in out.tested_specimens
    assert FALCON in out.replicating_specimens
    assert claim.scope["model"] == "MODEL_LOCAL"
    assert claim.scope_kind == "ORIGINAL"
    assert out.original_scope["model"] == "MODEL_LOCAL"


def test_organ_widen_requires_distinct_organ():
    claim = cs.validate_claim(_claim(), timeline=LATER)
    with pytest.raises(cs.ScopeViolation) as ei:
        cs.widen(
            claim,
            "organ",
            replicating_specimen=PARENT,
            replication_receipt="r.json",
            replicating_organ="mlp",
            timeline=LATER,
            at="2026-08-31T12:00:00Z",
        )
    assert ei.value.reason == "replicating_organ_not_distinct"
    out = cs.widen(
        claim,
        "organ",
        replicating_specimen=PARENT,
        replication_receipt="r.json",
        replicating_organ="deltanet",
        timeline=LATER,
        at="2026-08-31T12:00:00Z",
    )
    assert out.scope["organ"] == "ORGAN_REPLICATED"
    assert out.scope["model"] == "MODEL_LOCAL"


def test_failed_transfer_does_not_widen():
    claim = cs.validate_claim(_claim(), timeline=LATER)
    out = cs.record_failed_transfer(
        claim,
        specimen=FALCON,
        receipt="receipts/future/FAKE_FAILED.json",
        why="did not replicate ARM A jump",
        at="2026-08-31T12:00:00Z",
        timeline=LATER,
    )
    assert out.scope == claim.scope
    assert out.scope_kind == "ORIGINAL"
    assert out.failed_transfers[0]["specimen"] == FALCON
    assert out.scope["model"] == "MODEL_LOCAL"


def test_hindsight_contamination_refuses_later_specimen_on_earlier_law():
    """A law from 16:00 on the 30th cannot list Flash sealed the next morning."""
    with pytest.raises(cs.HindsightContamination) as ei:
        cs.validate_claim(
            _claim(
                as_of="2026-08-30T16:00:00Z",
                available_specimens=(PARENT, FLASH),
            ),
            timeline=LATER,
        )
    assert ei.value.reason == "specimen_not_sealed_at_as_of"
    assert ei.value.specimen == FLASH


def test_available_at_is_time_indexed():
    early = cs.available_at("2026-08-30T16:00:00Z", LATER)
    late = cs.available_at("2026-08-31T12:00:00Z", LATER)
    assert early == [PARENT]
    assert PARENT in late and FALCON in late and FLASH in late
    assert set(early) <= set(late)


def test_in_flight_is_not_available():
    tl = _timeline(
        _sealed(PARENT, "2026-08-30T15:44:16Z"),
        in_flight=(
            {
                "specimen": "Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8",
                "in_flight": True,
                "sealed": False,
            },
        ),
    )
    tl["specimens"]["Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8"] = {
        "specimen": "Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8",
        "sealed": False,
        "in_flight": True,
        "sealed_at": None,
    }
    names = cs.available_at("2026-08-31T23:00:00Z", tl)
    assert "Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8" not in names
    assert PARENT in names


def test_experiment_binds_six_identity_fields():
    """Acceptance: every experiment binds the six identity fields."""
    ident = _identity()
    for field in cs.EXPERIMENT_IDENTITY_FIELDS:
        assert ident[field]
    for drop in cs.EXPERIMENT_IDENTITY_FIELDS:
        kwargs = {
            "specimen_seal": ident["specimen_seal"],
            "model_revision": ident["model_revision"],
            "resident_identity": ident["resident_identity"],
            "code_and_build_identity": ident["code_and_build_identity"],
            "machine_genome": ident["machine_genome"],
            "laws_scars_version": ident["laws_scars_version"],
        }
        kwargs[drop] = None
        with pytest.raises(cs.ExperimentIdentityError) as ei:
            cs.bind_experiment(**kwargs)
        assert drop in ei.value.missing


def test_bind_experiment_fill_defaults_still_has_six_fields():
    ident = cs.bind_experiment(fill_defaults=True)
    for field in cs.EXPERIMENT_IDENTITY_FIELDS:
        assert ident.get(field), field
    seal = ident["specimen_seal"]
    assert seal.get("specimen") == PARENT
    assert seal.get("tree_digest")
    assert ident["model_revision"]["model_id"] == cs.PARENT_MODEL_ID
    assert ident["machine_genome"]["remeasured"] is False
    assert ident["laws_scars_version"]["digest"]
    assert ident["code_and_build_identity"]["git_head"]


def test_universal_conclusion_refused_on_model_local():
    claim = cs.validate_claim(_claim(), timeline=EARLY)
    with pytest.raises(cs.OverbroadConclusion) as ei:
        cs.conclude(claim, "All Hawking models behave this way")
    assert ei.value.reason == "universal_claim_on_model_local"
    ok = cs.conclude(claim, "ARM A jumped on this parent")
    assert ok["evidence_universe"].startswith("within currently sealed specimens")
    assert PARENT in ok["evidence_universe"]


def test_campaign_laws_are_retro_scoped_and_overbroad_narrowed():
    """Acceptance: at least one existing campaign law is retro-scoped; over-broad statements narrowed."""
    tl = cs.load_timeline()
    claims, narrowings = cs.retro_campaign_laws(timeline=tl)
    ids = [c.law_id for c in claims]
    assert "LAW-MLP-ARITHMETIC-SENSITIVITY" in ids
    assert "LAW-BROADCAST-AUX-NON-CRITICALITY" in ids
    assert "LAW-MLP-FUNCTION-REPLACEMENT-CLOSED" in ids
    assert "LAW-PROBE-UNDERSELLS-TOKEN" in ids
    assert "LAW-497P4-ANCHOR" in ids
    assert any(c.narrowed for c in claims), "at least one over-broad statement must be narrowed"
    assert narrowings, "narrowing must be recorded, not silent"
    for claim in claims:
        assert claim.scope == cs.narrow_scope()
        assert claim.scope_kind == "ORIGINAL"
        assert claim.tested_specimens == (PARENT,)
        assert PARENT in claim.available_specimens
        assert FLASH not in claim.tested_specimens
        assert FALCON not in claim.tested_specimens
        for axis in cs.AXES:
            assert claim.scope[axis] == cs.NARROW_TIERS[axis]
        for field in cs.EXPERIMENT_IDENTITY_FIELDS:
            assert claim.experiment_identity.get(field), (claim.law_id, field)
        assert claim.evidence_universe().startswith("within currently sealed specimens")
        assert "all Hawking models" not in claim.statement
    closed = next(c for c in claims if c.law_id == "LAW-MLP-FUNCTION-REPLACEMENT-CLOSED")
    assert closed.narrowed
    assert closed.teacher_corpus_rows == 45076
    assert closed.layers == (3, 31, 38, 63)
    assert "MODEL_SPECIFIC" in (closed.statement + (closed.narrowing or ""))
    arith = next(c for c in claims if c.law_id == "LAW-MLP-ARITHMETIC-SENSITIVITY")
    assert arith.narrowed
    assert "MIXED" in arith.statement
    assert "ALU-bound" in (arith.statement_before_narrowing or "")
    roof = next(c for c in claims if c.law_id == "LAW-497P4-ANCHOR")
    assert roof.narrowed
    assert "DRAM" in roof.statement or "DRAM" in (roof.narrowing or "")


def test_retro_function_replacement_does_not_inherit_later_seals_as_tested():
    """Hindsight: a law made before the five-role verification must not read as tested on six."""
    tl = cs.load_timeline()
    claims, _ = cs.retro_campaign_laws(timeline=tl)
    closed = next(c for c in claims if c.law_id == "LAW-MLP-FUNCTION-REPLACEMENT-CLOSED")
    assert closed.tested_specimens == (PARENT,)
    # Flash's independent verification is 2026-08-31T09:56:23Z; structured
    # operator was recorded at 08:25 the same day. Flash may be available
    # later, never tested here.
    assert FLASH not in closed.tested_specimens
    if closed.as_of and closed.as_of != "UNKNOWN":
        as_of = cs.parse_iso(closed.as_of)
        flash_row = (tl.get("specimens") or {}).get(FLASH) or {}
        flash_at = cs.parse_iso(flash_row.get("sealed_at"))
        if as_of and flash_at and as_of < flash_at:
            assert FLASH not in closed.available_specimens


def test_timeline_reads_real_seal_times_not_a_fixture():
    tl = cs.load_timeline()
    parent = (tl.get("specimens") or {}).get(PARENT) or {}
    assert parent.get("sealed") is True, "authorized-external parent must be sealed"
    assert parent.get("sealed_at"), "seal time must be read, not left empty"
    assert parent.get("sealed_at") != "UNKNOWN"
    src = str(parent.get("sealed_at_source") or "")
    assert "EXTERNAL_SPECIMEN_SEAL" in src or "SPECIMEN_VERIFICATION" in src
    assert parent.get("tree_digest"), "tree digest is the identity, location is not"
    # Fixture times would be a constant in this test file. The live reader
    # must agree with the receipt on disk.
    edoc = json.loads((RECEIPTS / "EXTERNAL_SPECIMEN_SEAL.json").read_text())
    assert parent["tree_digest"] == edoc["tree_digest"]
    assert parent["sealed_at"] == edoc["bench"]["recorded_at"]
    sealed_names = {r["specimen"] for r in tl.get("sealed") or []}
    assert PARENT in sealed_names
    for flying in tl.get("in_flight") or []:
        assert flying["specimen"] not in sealed_names
        assert flying.get("sealed") is False


def test_live_available_set_grows_and_does_not_include_in_flight():
    tl = cs.load_timeline()
    parent_at = ((tl.get("specimens") or {}).get(PARENT) or {}).get("sealed_at")
    if not parent_at:
        pytest.skip("parent seal time unread")
    early = cs.available_at(parent_at, tl)
    late = cs.available_at("2099-01-01T00:00:00Z", tl)
    assert PARENT in early
    assert set(early) <= set(late)
    flying = {r["specimen"] for r in tl.get("in_flight") or []}
    assert flying.isdisjoint(set(late))


def test_build_writes_receipt_without_hardware_claims():
    out = cs.build()
    path = Path(out["path"])
    assert path == RECEIPTS / cs.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == cs.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["n_laws"] == 5
    assert doc["n_narrowed"] >= 1
    assert doc["seal_sha256"]
    assert doc["refusal_witness"]["widen_without_specimen"]["raised"] is True
    assert doc["refusal_witness"]["widen_without_specimen"]["reason"] == "no_named_replicating_specimen"
    assert doc["refusal_witness"]["universal_conclusion"]["raised"] is True
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")
    for law in doc["laws"]:
        assert law["scope"]["model"] == "MODEL_LOCAL"
        assert law["scope"]["organ"] == "ORGAN_LOCAL"
        assert law["scope"]["machine"] == "MACHINE_LOCAL"
        assert PARENT in law["tested_specimens"]
        for field in cs.EXPERIMENT_IDENTITY_FIELDS:
            assert law["experiment_identity"][field], field
        assert law["evidence_universe"].startswith("within currently sealed specimens")
    assert any(n.get("correction_not_downgrade") for n in doc["narrowings"])
    assert doc["timeline"]["n_sealed"] >= 1
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
