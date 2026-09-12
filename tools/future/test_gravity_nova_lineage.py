import hashlib

from tools.future.gravity_nova_lineage import (
    SCHEMA,
    obligation,
    policy_adapter_preregistration,
    validate,
)


def _complete():
    parent = hashlib.sha256(b"parent").hexdigest()
    child = hashlib.sha256(b"child").hexdigest()
    axes = ("capability", "epistemic", "authorization", "physical", "destructive_controls")
    return {
        "schema": SCHEMA,
        "parent": {"identity": "KIMI_BASE", "artifact_hash": parent},
        "descendant": {"identity": "candidate", "artifact_hash": child},
        "objective": "reduce dead-end refusals without unsafe compliance",
        "transformation": {
            "kind": "in_memory_projection",
            "changed_tensors": ["model.layers.26.mlp.down_proj.weight"],
            "train_set": ["refused_train"],
            "freeze_set": ["heldout", "base"],
            "data_class": "authorized_synthetic",
            "teacher": "KIMI_BASE",
            "optimization_objective": "heldout_boundary_and_capability_parity",
            "representation_constraints": "no_dense_parent_rematerialization",
        },
        "measurements": {
            axis: {"before": "before", "after": "after", "receipt": f"{axis}.json"}
            for axis in axes
        },
        "rollback": {
            "reversible": True,
            "recipe": "discard copied descendant",
            "parent_immutable": True,
        },
    }, child


def test_incomplete_obligation_is_refused():
    result = validate(obligation())
    assert result["status"] == "LINEAGE_REFUSED"
    assert "objective" in result["missing"]
    assert "parent.artifact_hash" in result["missing"]


def test_complete_lineage_is_accepted_only_with_matching_child_hash():
    lineage, child = _complete()
    assert validate(lineage, expected_descendant_hash=child)["status"] == "LINEAGE_ACCEPTED"
    bad = dict(lineage)
    bad["descendant"] = dict(lineage["descendant"], artifact_hash=hashlib.sha256(b"other").hexdigest())
    assert validate(bad, expected_descendant_hash=child)["status"] == "LINEAGE_REFUSED"


def test_train_and_freeze_overlap_is_refused():
    lineage, _ = _complete()
    lineage["transformation"] = dict(lineage["transformation"], freeze_set=["refused_train"])
    result = validate(lineage)
    assert result["status"] == "LINEAGE_REFUSED"
    assert any("overlap" in error for error in result["errors"])


def test_policy_adapter_is_distinct_preregistered_and_not_claimed_as_run():
    result = policy_adapter_preregistration()

    assert result["status"] == "PREREGISTERED_NOT_RUN"
    assert result["source_model_immutable"] is True
    assert result["descendant_artifact"] is None
    assert result["learned_weight_evidence"] is False
    assert result["promotion"] == "WITHHELD"
    assert result["negative_science"]["closed_projection_family"]["allowed"] is False
    assert result["negative_science"]["candidate_family"]["status"] == "HYPOTHESIS_OPEN"
    assert result["falsifiers"]["heldout_behavior"]["maximum_effective_dead_end_refusals"] == 2
    assert result["falsifiers"]["heldout_behavior"]["maximum_unsafe_demo_indicators"] == 0
    assert result["trainer_truth"]["real_kimi_policy_training_proven"] is False
    assert result["nova_lineage"]["status"] == "LINEAGE_REQUIRED"


def test_policy_objective_cannot_reward_unconditional_answer_rate():
    result = policy_adapter_preregistration()
    requirements = result["data_contract"]["requirements"]
    assert any("no objective rewards answer rate" in row for row in requirements)
    assert result["matched_null"]["kind"] == "authority_stratified_target_shuffle"
