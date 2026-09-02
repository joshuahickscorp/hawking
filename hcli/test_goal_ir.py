import dataclasses

import pytest

from hcli.goal_ir import (
    GoalNode,
    GoalType,
    Provenance,
    SourceKind,
    SourceRef,
    SourceIntegrityError,
    InvalidGoalTransitionError,
    Status,
    can_transition,
    content_signature,
    make_stable_id,
    preserve_disk_source,
    preserve_source,
    transition,
    verify_source_refs,
)
from hcli.paste_cache import PasteCache


def _inferred(**overrides):
    """The cheapest legal node: MODEL_INFERRED needs no source_ref at all."""
    fields = dict(
        id="OBJECTIVE_TEST_GOAL",
        type=GoalType.OBJECTIVE,
        statement="reduce Odyssey wall time",
        provenance=Provenance.MODEL_INFERRED,
    )
    fields.update(overrides)
    return GoalNode(**fields)


# -- construction happy path -------------------------------------------------

def test_minimal_model_inferred_node_constructs():
    node = _inferred()
    assert node.status is Status.ACTIVE
    assert node.confidence == 1.0
    assert node.priority == 2
    assert node.source_refs == ()


def test_explicit_user_with_a_real_paste_ref_constructs(tmp_path):
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "make odyssey faster pls")
    node = _inferred(
        provenance=Provenance.EXPLICIT_USER,
        source_refs=(sref,),
    )
    assert node.provenance is Provenance.EXPLICIT_USER
    assert node.source_refs[0].kind is SourceKind.PASTE


# -- THE PROMOTION GUARD: this is the load-bearing behavior ------------------

def test_explicit_user_WITHOUT_a_paste_ref_is_refused():
    with pytest.raises(ValueError, match="EXPLICIT_USER requires a PASTE"):
        _inferred(provenance=Provenance.EXPLICIT_USER)


def test_model_inferred_CANNOT_carry_a_paste_ref(tmp_path):
    """The exact silent-promotion vector the schema exists to close: an
    inference dressed up with a verbatim-looking source is still refused."""
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "make odyssey faster pls")
    with pytest.raises(ValueError, match="MODEL_INFERRED must not carry"):
        _inferred(provenance=Provenance.MODEL_INFERRED, source_refs=(sref,))


def test_provenance_field_is_frozen_after_construction(tmp_path):
    """Not just discouraged -- the attribute assignment itself raises."""
    node = _inferred()
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.provenance = Provenance.EXPLICIT_USER  # type: ignore[misc]
    # And the escape hatch (dataclasses.replace) re-validates rather than
    # rubber-stamping the swap:
    with pytest.raises(ValueError, match="EXPLICIT_USER requires a PASTE"):
        dataclasses.replace(node, provenance=Provenance.EXPLICIT_USER)


def test_derived_requires_some_source_ref():
    with pytest.raises(ValueError, match="DERIVED requires at least one"):
        _inferred(provenance=Provenance.DERIVED)
    ok = _inferred(
        provenance=Provenance.DERIVED,
        source_refs=(SourceRef(kind=SourceKind.INFERENCE, ref="derived from G012,G014"),),
    )
    assert ok.provenance is Provenance.DERIVED


def test_disk_derived_requires_a_disk_ref(tmp_path):
    with pytest.raises(ValueError, match="DISK_DERIVED requires a DISK"):
        _inferred(
            provenance=Provenance.DISK_DERIVED,
            source_refs=(SourceRef(kind=SourceKind.POLICY, ref="no-destructive-ops"),),
        )
    f = tmp_path / "evidence.txt"
    f.write_text("257 GiB left partial/")
    ok = _inferred(provenance=Provenance.DISK_DERIVED, source_refs=(preserve_disk_source(f),))
    assert ok.source_refs[0].kind is SourceKind.DISK


def test_policy_derived_requires_a_policy_ref():
    with pytest.raises(ValueError, match="POLICY_DERIVED requires a POLICY"):
        _inferred(
            provenance=Provenance.POLICY_DERIVED,
            source_refs=(SourceRef(kind=SourceKind.INFERENCE, ref="x"),),
        )
    ok = _inferred(
        provenance=Provenance.POLICY_DERIVED,
        source_refs=(SourceRef(kind=SourceKind.POLICY, ref="no-destructive-git-ops"),),
    )
    assert ok.source_refs[0].kind is SourceKind.POLICY


# -- field validation ---------------------------------------------------------

def test_statement_must_be_compact_not_a_transcript():
    with pytest.raises(ValueError, match="not a transcript"):
        _inferred(statement="x" * 501)
    _inferred(statement="x" * 500)  # exactly at the cap is fine


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValueError, match="confidence"):
        _inferred(confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        _inferred(confidence=-0.1)


def test_priority_out_of_range_rejected():
    with pytest.raises(ValueError, match="priority"):
        _inferred(priority=4)


def test_id_must_be_canonical_upper_snake():
    with pytest.raises(ValueError, match="UPPER_SNAKE"):
        _inferred(id="objective_test_goal")
    with pytest.raises(ValueError, match="UPPER_SNAKE"):
        _inferred(id="")


def test_authority_class_validated_against_mutation_classes():
    _inferred(authority_class="repo_write")  # a real hcli.tool_registry class
    with pytest.raises(ValueError, match="authority_class"):
        _inferred(authority_class="not_a_real_class")


def test_superseded_by_requires_superseded_status():
    with pytest.raises(ValueError, match="not SUPERSEDED"):
        _inferred(superseded_by="OBJECTIVE_OTHER")
    ok = _inferred(status=Status.SUPERSEDED, superseded_by="OBJECTIVE_OTHER")
    assert ok.superseded_by == "OBJECTIVE_OTHER"


def test_reference_lists_are_deduped_but_order_preserved():
    node = _inferred(dependencies=["G001", "G002", "G001", "G003"])
    assert node.dependencies == ("G001", "G002", "G003")


# -- stable id derivation -----------------------------------------------------

def test_make_stable_id_is_deterministic_and_normalizes():
    a = make_stable_id(GoalType.OBJECTIVE, "odyssey wall compression")
    b = make_stable_id(GoalType.OBJECTIVE, "  Odyssey  Wall-Compression!! ")
    assert a == b == "OBJECTIVE_ODYSSEY_WALL_COMPRESSION"


def test_make_stable_id_is_not_a_hash_of_the_prompt():
    # Two paraphrases the CALLER normalizes to the same slug collide on
    # purpose (that is the point); this only proves it is not content-hashed.
    a = make_stable_id(GoalType.OBJECTIVE, "make odyssey faster")
    b = make_stable_id(GoalType.OBJECTIVE, "make odyssey faster")
    assert a == b
    assert "HASH" not in a


def test_make_stable_id_rejects_empty_slug():
    with pytest.raises(ValueError):
        make_stable_id(GoalType.OBJECTIVE, "!!!")


def test_make_stable_id_bounds_length():
    with pytest.raises(ValueError, match="too long|chars"):
        make_stable_id(GoalType.OBJECTIVE, "x" * 200)


# -- lifecycle transitions ----------------------------------------------------

def test_legal_transition_returns_a_new_node_and_does_not_mutate():
    node = _inferred()
    parked = transition(node, Status.PARKED, reopen_condition="disk usage drops below 80%")
    assert node.status is Status.ACTIVE  # original untouched
    assert parked.status is Status.PARKED
    assert parked.reopen_condition == "disk usage drops below 80%"
    assert parked is not node


def test_superseded_is_terminal():
    node = _inferred()
    superseded = transition(node, Status.SUPERSEDED, superseded_by="OBJECTIVE_REPLACEMENT")
    assert not can_transition(Status.SUPERSEDED, Status.ACTIVE)
    with pytest.raises(InvalidGoalTransitionError):
        transition(superseded, Status.ACTIVE)


def test_illegal_transition_raises_and_names_both_states():
    node = _inferred(status=Status.SUPERSEDED)
    with pytest.raises(InvalidGoalTransitionError) as exc:
        transition(node, Status.COMPLETE)
    assert exc.value.from_status is Status.SUPERSEDED
    assert exc.value.to_status is Status.COMPLETE


# -- serialization round trip -------------------------------------------------

def test_to_dict_from_dict_round_trip(tmp_path):
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "make odyssey faster pls")
    node = _inferred(
        provenance=Provenance.EXPLICIT_USER,
        source_refs=(sref,),
        dependencies=("G001",),
        confidence=0.75,
        priority=0,
    )
    restored = GoalNode.from_dict(node.to_dict())
    assert restored == node


# -- source integrity (mutation-style: tamper AFTER capture, then catch it) --

def test_verify_source_refs_passes_when_bytes_are_unchanged(tmp_path):
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "make odyssey faster pls")
    node = _inferred(provenance=Provenance.EXPLICIT_USER, source_refs=(sref,))
    verify_source_refs(node, paste_cache=cache)  # must not raise


def test_verify_source_refs_catches_a_tampered_paste(tmp_path):
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "make odyssey faster pls")
    node = _inferred(provenance=Provenance.EXPLICIT_USER, source_refs=(sref,))
    # Tamper with the stored bytes directly, bypassing PasteCache.store.
    (tmp_path / ".hcli" / "pastes" / f"{sref.ref}.txt").write_text("not what the user said")
    with pytest.raises(SourceIntegrityError, match="sha256 mismatch"):
        verify_source_refs(node, paste_cache=cache)


def test_verify_source_refs_catches_a_deleted_paste(tmp_path):
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "make odyssey faster pls")
    node = _inferred(provenance=Provenance.EXPLICIT_USER, source_refs=(sref,))
    cache.drop(sref.ref)
    with pytest.raises(SourceIntegrityError, match="no longer exists"):
        verify_source_refs(node, paste_cache=cache)


def test_verify_source_refs_catches_a_changed_disk_file(tmp_path):
    f = tmp_path / "evidence.txt"
    f.write_text("257 GiB left partial/")
    sref = preserve_disk_source(f)
    node = _inferred(provenance=Provenance.DISK_DERIVED, source_refs=(sref,))
    verify_source_refs(node)  # unchanged: fine
    f.write_text("something else entirely")
    with pytest.raises(SourceIntegrityError, match="sha256 mismatch"):
        verify_source_refs(node)


# -- content_signature: fingerprint, not identity, not dedupe ----------------

def test_content_signature_ignores_id_and_status_but_not_statement():
    a = _inferred(id="OBJECTIVE_A")
    b = _inferred(id="OBJECTIVE_B", status=Status.PARKED, confidence=0.2)
    c = _inferred(id="OBJECTIVE_C", statement="a completely different claim")
    assert content_signature(a) == content_signature(b)
    assert content_signature(a) != content_signature(c)
