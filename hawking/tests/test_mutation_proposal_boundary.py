from pathlib import Path
import hashlib

from hawking.engine import Engine
from hawking.mutation import (
    MutationProposal,
    MutationError,
    execute_mutation_proposal,
    normalize_mutation_result,
)
from hawking.workunit_owner import (
    WorkunitRecord,
    WorkunitOwner,
    _compact_discrete_continuation,
    _discrete_goal_prompt,
    record_provider_dead_letter,
    transition,
)
from hawking.workspace import Workspace
from hawking.compact_worker import create_source_anchor


def _authority(root, *, write=True):
    return {"workspace_root": str(root), "capabilities": ["repo.edit"] if write else [],
            "mutation_lease": "LEASE-1" if write else None}


def test_build_proposal_uses_canonical_engine(tmp_path: Path):
    (tmp_path / "mod.py").write_text("VALUE = 1\n")
    (tmp_path / "test_mod.py").write_text("from mod import VALUE\n\ndef test_value():\n    assert VALUE == 2\n")
    proposal = MutationProposal(objective="change value", canonical_root=str(tmp_path),
        operations=({"op": "replace", "path": "mod.py", "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]},),
        required_tests=("test_mod.py",), goal_id="G", workunit_id="W")
    result = execute_mutation_proposal(proposal, engine=Engine(Workspace(str(tmp_path))),
                                       authority=_authority(tmp_path))
    assert result["status"] == "accepted"
    assert result["execution"]["completion_packet"]["verification"]["diff_inspected"]


def test_compact_anchor_intent_compiles_into_canonical_repo_edit(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n"
    )
    anchor = create_source_anchor(tmp_path, target, start_line=1, end_line=1)
    result = execute_mutation_proposal(
        {"s": "MUTATE", "anchor": anchor.anchor, "op": "replace", "body": "VALUE = 2\n", "tests": ["test_mod.py"]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] == "accepted"
    assert result["applied"] is True
    assert target.read_text() == "VALUE = 2\n"


def test_read_only_proposal_cannot_mutate(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n")
    proposal = {"objective": "change", "canonical_root": str(tmp_path),
                "operations": [{"op": "replace_file", "path": "mod.py", "new_text": "VALUE = 2\n"}]}
    result = execute_mutation_proposal(proposal, engine=Engine(Workspace(str(tmp_path))),
                                       authority=_authority(tmp_path, write=False))
    assert result["reason"] == "MUTATION_AUTHORITY_REQUIRED"
    assert target.read_text() == "VALUE = 1\n"


def test_typed_provider_proposal_gets_hawking_identity_and_scope(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n")
    proposal = {
        "schema": "hawking.mutation_proposal.v1",
        "status": "MUTATION_PROPOSED",
        "objective": "change value",
        "operations": [{
            "op": "replace", "path": "mod.py",
            "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"],
        }],
        "required_tests": [],
        # These are data, not authority. Hawking must reject the foreign scope
        # rather than allowing the provider to choose a different target.
        "canonical_root": str(tmp_path),
        "goal_id": "OTHER-GOAL",
        "workunit_id": "OTHER-WORKUNIT",
    }
    result = execute_mutation_proposal(
        proposal,
        engine=Engine(Workspace(str(tmp_path))),
        authority={
            **_authority(tmp_path),
            "goal_id": "GOAL-1",
            "workunit_id": "WU-1",
            "allowed_paths": ["mod.py"],
        },
    )
    assert result["reason"] == "WORKUNIT_AUTHORITY_MISMATCH"
    assert target.read_text() == "VALUE = 1\n"


def test_provider_proposal_cannot_escape_workunit_scope(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n")
    proposal = MutationProposal(
        objective="change value", canonical_root=str(tmp_path),
        operations=({
            "op": "replace", "path": "mod.py",
            "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"],
        },),
    )
    result = execute_mutation_proposal(
        proposal, engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["other.py"]},
    )
    assert result["reason"] == "MUTATION_SCOPE_DENIED"
    assert target.read_text() == "VALUE = 1\n"


def test_wrong_workspace_is_rejected(tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    result = execute_mutation_proposal({"canonical_root": str(other), "operations": [{"op": "create", "path": "x.txt", "new_text": "x"}]},
                                       engine=Engine(Workspace(str(tmp_path))), authority=_authority(tmp_path))
    assert result["reason"] == "WRONG_WORKSPACE"


def test_patch_envelope_compiles_and_executes_through_canonical_engine(tmp_path: Path):
    (tmp_path / "mod.py").write_text("VALUE = 1\n")
    (tmp_path / "test_mod.py").write_text("from mod import VALUE\n\ndef test_value():\n    assert VALUE == 2\n")
    payload = """HAWKING_PATCH_V1
BASE_REVISION:
BEGIN_PATCH
--- a/mod.py
+++ b/mod.py
@@
-VALUE = 1
+VALUE = 2
END_PATCH
TESTS: test_mod.py
"""
    proposal = normalize_mutation_result(payload, canonical_root=str(tmp_path), goal_id="G", workunit_id="W")
    assert proposal.to_dict()["status"] == "MUTATION_PROPOSED"
    result = execute_mutation_proposal(proposal, engine=Engine(Workspace(str(tmp_path))), authority=_authority(tmp_path))
    assert result["status"] == "accepted"
    assert (tmp_path / "mod.py").read_text() == "VALUE = 2\n"


def test_provider_json_edit_dialect_is_coerced_before_canonical_engine(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(
        "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    proposal = {
        "status": "MUTATION_PROPOSED",
        "operations": [{
            "op": "edit",
            "file": "mod.py",
            "mode": "anchor",
            "anchor_before": "VALUE = 1",
            "replacement": "VALUE = 2",
            "expected_digest": digest,
        }],
        "tests": ["test_mod.py"],
    }
    result = execute_mutation_proposal(
        proposal,
        engine=Engine(Workspace(str(tmp_path))),
        authority=_authority(tmp_path),
    )
    assert result["status"] == "accepted"
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_top_level_typed_edit_dialect_is_compiled_to_one_operation(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    proposal = MutationProposal.from_mapping({
        "status": "MUTATION_PROPOSED",
        "schema": "HAWKING_EDIT_V1",
        "file": "mod.py",
        "mode": "anchor_edit",
        "anchor_before": "VALUE = 1",
        "replacement": "VALUE = 2",
        "expected_digest": "UNKNOWN",
    })
    assert len(proposal.operations) == 1
    operation = proposal.operations[0]
    assert operation["op"] == "replace"
    assert operation["path"] == "mod.py"
    assert operation["old_text"] == "VALUE = 1"
    assert operation["new_text"] == "VALUE = 2"
    assert proposal.expected_base_hashes == {}


def test_top_level_typed_edit_dialect_executes_through_authority_gates(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {
            "status": "MUTATION_PROPOSED",
            "schema": "HAWKING_EDIT_V1",
            "file": "mod.py",
            "mode": "anchor_edit",
            "anchor_before": "VALUE = 1",
            "replacement": "VALUE = 2",
            "expected_digest": "UNKNOWN",
        },
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_source_operations_dialect_reaches_canonical_engine(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {
            "status": "MUTATION_PROPOSED",
            "source_operations": [{
                "op": "replace_once",
                "path": "mod.py",
                "anchor": "VALUE = 1",
                "replacement": "VALUE = 2",
            }],
        },
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_source_operation_singleton_is_normalized_to_canonical_list():
    proposal = MutationProposal.from_mapping({
        "status": "MUTATION_PROPOSED",
        "source_operation": {
            "op": "replace_lines",
            "path": "mod.py",
            "old_lines": ["VALUE = 1"],
            "new_lines": ["VALUE = 2"],
        },
    })
    assert len(proposal.operations) == 1
    assert proposal.operations[0]["op"] == "replace"
    assert proposal.operations[0]["path"] == "mod.py"


def test_provider_compact_insert_dialect_is_coerced_before_canonical_engine(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("FIRST = 1\nLAST = 2\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {
            "status": "MUTATION_PROPOSED",
            "operations": [{
                "op": "insert",
                "path": "mod.py",
                "anchor": "FIRST = 1\n",
                "text": "MIDDLE = 1.5\n",
                "position": "after",
            }],
        },
        engine=Engine(Workspace(str(tmp_path))),
        authority=_authority(tmp_path),
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "FIRST = 1\nMIDDLE = 1.5\nLAST = 2\n"


def test_provider_read_observations_do_not_block_effect_operation(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    proposal = {
        "status": "MUTATION_PROPOSED",
        "operations": [
            {"op": "read", "path": "mod.py"},
            {"op": "replace", "path": "mod.py",
             "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]},
        ],
    }
    result = execute_mutation_proposal(
        proposal,
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_read_only_provider_operations_cannot_mutate(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"status": "MUTATION_PROPOSED",
         "operations": [{"op": "read", "path": "mod.py"}]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["reason"] == "EMPTY_MUTATION_PROPOSAL"
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_provider_tool_repo_edit_anchor_dialect_is_coerced(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"status": "MUTATION_PROPOSED", "operations": [{
            "tool": "repo.edit", "path": "mod.py", "mode": "anchor_edit",
            "anchor_before": "VALUE = 1", "replacement": "VALUE = 2",
        }]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_provider_begin_replacement_alias_is_coerced(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"status": "MUTATION_PROPOSED", "operations": [{
            "op": "replace", "file": "mod.py",
            "anchor_before": "VALUE = 1", "begin_replacement": "VALUE = 2",
        }]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_provider_bare_anchor_alias_is_coerced(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"status": "MUTATION_PROPOSED", "operations": [{
            "op": "replace", "path": "mod.py",
            "anchor": "VALUE = 1", "new_text": "VALUE = 2",
        }]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_provider_append_content_alias_is_coerced(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"status": "MUTATION_PROPOSED", "operations": [{
            "op": "append", "path": "mod.py", "anchor": "__EOF__",
            "content": "\nVALUE_TWO = 2\n",
        }]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n\nVALUE_TWO = 2\n"


def test_typed_repair_proposal_status_is_coerced_before_authority_gates():
    proposal = normalize_mutation_result(
        {
            "status": "REPAIR_PROPOSED",
            "objective": "repair the bounded edit",
            "operations": [{
                "op": "replace",
                "path": "mod.py",
                "old_lines": ["VALUE = 1"],
                "new_lines": ["VALUE = 2"],
            }],
        },
        canonical_root="/tmp/worktree",
        goal_id="G",
        workunit_id="W",
    )
    assert proposal.to_dict()["status"] == "MUTATION_PROPOSED"
    assert proposal.operations[0]["path"] == "mod.py"


def test_provider_tool_read_entry_is_not_a_mutation(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"status": "MUTATION_PROPOSED", "operations": [{
            "tool": "fs.read", "path": "mod.py",
        }]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["reason"] == "EMPTY_MUTATION_PROPOSAL"
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_json_legacy_patch_schema_is_accepted_as_typed_provider_data(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    result = execute_mutation_proposal(
        {"schema": "HAWKING_PATCH_V1", "status": "MUTATION_PROPOSED",
         "operations": [{"op": "replace", "path": "mod.py",
                         "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]}]},
        engine=Engine(Workspace(str(tmp_path))),
        authority={**_authority(tmp_path), "allowed_paths": ["mod.py"]},
    )
    assert result["status"] in {"accepted", "unproven"}
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_string_test_field_is_one_focused_test_not_character_sequence(tmp_path: Path):
    proposal = MutationProposal.from_mapping({
        "status": "MUTATION_PROPOSED",
        "operations": [{"op": "replace", "path": "mod.py"}],
        "tests": "hawking/test_mod.py",
    })
    assert proposal.required_tests == ("hawking/test_mod.py",)


def test_anchor_patch_variant_is_compiled_and_content_bound(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\ndef marker():\n    return VALUE\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n"
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    payload = f"""HAWKING_PATCH_V1
BASE_REVISION: unknown
EXPECTED_DIGEST: mod.py sha256={digest}
TESTS: test_mod.py

FILE: mod.py
MODE: replace_range
ANCHOR_BEFORE: VALUE = 1
ANCHOR_AFTER: def marker():
BEGIN_REPLACEMENT
VALUE = 2
END_REPLACEMENT
"""
    proposal = normalize_mutation_result(payload, canonical_root=str(tmp_path))
    assert proposal.expected_base_hashes == {"mod.py": digest}
    root = tmp_path.resolve()
    result = execute_mutation_proposal(
        proposal, engine=Engine(Workspace(str(root))), authority=_authority(root)
    )
    assert result["status"] == "accepted"


def test_singleton_provider_operation_is_normalized_to_canonical_list():
    proposal = MutationProposal.from_mapping({
        "status": "MUTATION_PROPOSED",
        "operation": {
            "type": "repo.edit",
            "path": "mod.py",
            "old": "VALUE = 1",
            "new": "VALUE = 2",
        },
    })

    assert len(proposal.operations) == 1
    assert proposal.operations[0]["op"] == "replace"
    assert proposal.operations[0]["path"] == "mod.py"
    assert proposal.operations[0]["old_text"] == "VALUE = 1"
    assert proposal.operations[0]["new_text"] == "VALUE = 2"


def test_provider_source_edit_kind_and_append_action_are_normalized():
    proposal = MutationProposal.from_mapping({
        "status": "MUTATION_PROPOSED",
        "operation": {
            "kind": "edit",
            "file": "mod.py",
            "anchor": "VALUE = 1",
            "action": "append_after",
            "content": "\nVALUE = 2",
        },
    })

    operation = proposal.operations[0]
    assert operation["op"] == "insert_after"
    assert operation["path"] == "mod.py"
    assert operation["old_text"] == "VALUE = 1"
    assert operation["new_text"] == "\nVALUE = 2"


def test_edit_envelope_compiles_without_exposing_provider_authority(tmp_path: Path):
    payload = """HAWKING_EDIT_V1
PATH: fixture.txt
EXPECTED_DIGEST:
MODE: create_file
BEGIN_CONTENT
safe\n
END_CONTENT
TESTS: test_fixture.py
"""
    proposal = normalize_mutation_result(payload, canonical_root=str(tmp_path))
    assert proposal.operations[0]["op"] == "create"
    assert proposal.operations[0]["path"] == "fixture.txt"
    assert proposal.goal_id == ""


def test_plain_prose_and_malformed_envelope_never_become_mutation(tmp_path: Path):
    for payload in ("I would change mod.py", "HAWKING_PATCH_V1\nBEGIN_PATCH\nEND_PATCH"):
        try:
            normalize_mutation_result(payload, canonical_root=str(tmp_path))
        except MutationError as exc:
            assert str(exc) in {"MUTATION_PAYLOAD_MISSING", "malformed mutation envelope: empty patch"}
        else:
            raise AssertionError("untyped provider output was accepted")


def test_stale_typed_proposal_is_rejected_before_engine(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("VALUE = 1\n")
    proposal = MutationProposal(
        objective="change", canonical_root=str(tmp_path),
        operations=({"op": "replace", "path": "mod.py", "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]},),
        expected_base_hashes={"mod.py": "0" * 64},
    )
    result = execute_mutation_proposal(proposal, engine=Engine(Workspace(str(tmp_path))), authority=_authority(tmp_path))
    assert result["reason"] == "STALE_MUTATION_PROPOSAL"
    assert target.read_text() == "VALUE = 1\n"


def test_provider_failure_is_child_output_unusable_not_root_blocked(tmp_path: Path):
    record = WorkunitRecord(workunit_id="W", goal_id="G", staging_workspace=str(tmp_path), state="RUNNING")
    path = record_provider_dead_letter(
        record, failure_class="PROVIDER_OUTPUT_UNUSABLE",
        remaining_obligation="proposal", reopen_condition="fresh packet",
    )
    transition(record, "OUTPUT_UNUSABLE")
    assert record.state == "OUTPUT_UNUSABLE"
    assert Path(path).is_file()
    assert record.goal_id == "G"
    assert record.state != "BLOCKED"


def test_discrete_goal_packets_do_not_request_provider_owned_mutation_tools(tmp_path: Path):
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    record = WorkunitRecord(
        workunit_id="W-PROPOSAL",
        goal_id="G-PROPOSAL",
        goal_mode="discrete_goal",
        staging_workspace=str(tmp_path),
        objective="Patch synthesis only; return HAWKING_PATCH_V1.",
    )
    contract = {
        "acceptance": ["return one executable payload"],
        "focus_files": ["mod.py"],
        "focus_tests": ["test_mod.py"],
    }
    opening = _discrete_goal_prompt(contract, record, 1, opening=True)
    continuation = _compact_discrete_continuation(contract, record, 2)
    for packet in (opening, continuation):
        assert "HAWKING_PATCH_V1" in packet
        assert "do not call repo.edit" in packet.lower()
        assert "emit the canonical repo.edit call" not in packet
        assert "use one real repo.edit operation" not in packet.lower()


def test_goal_owned_stale_driver_is_recoverable_child_state(tmp_path: Path, monkeypatch):
    import hawking.workunit_owner as owner_module

    monkeypatch.setattr(owner_module, "_pid_alive", lambda _pid: False)
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="WU-STALE-CHILD",
        goal_id="GOAL-ROOT",
        goal_mode="discrete_goal",
        state="RUNNING",
        driver_pid=987654321,
        exact_next_action="return the next bounded provider packet",
        staging_workspace=str(tmp_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )

    owner.save(record)

    restored = owner.load(record.workunit_id)
    assert restored.state == "RECONCILIATION_REQUIRED"
    assert restored.worker_status == "WAITING_RETRY"
    assert restored.worker_kill_reason == "stale_driver"
    assert restored.exact_next_action == "return the next bounded provider packet"
    assert restored.goal_id == "GOAL-ROOT"


def test_incomplete_goal_completion_becomes_output_unusable_not_blocked(tmp_path: Path):
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="WU-INCOMPLETE-CHILD",
        goal_id="GOAL-ROOT",
        goal_mode="discrete_goal",
        state="COMPLETE",
        staging_workspace=str(tmp_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        evidence=[{
            "kind": "worker_completion_contract",
            "complete": False,
            "unmet": ["repo.edit", "tests.run"],
        }],
    )
    owner.save(record)

    rejected = owner.invalidate_incomplete_completion(record.workunit_id)

    assert rejected["state"] == "OUTPUT_UNUSABLE"
    assert rejected["worker_status"] == "WAITING_RETRY"
    assert rejected["classification"] == "PROVIDER_STRUCTURED_RESULT_INVALID"
    assert rejected["state"] != "BLOCKED"


def test_legacy_bootstrap_stale_driver_keeps_historical_blocked_behavior(tmp_path: Path, monkeypatch):
    import hawking.workunit_owner as owner_module

    monkeypatch.setattr(owner_module, "_pid_alive", lambda _pid: False)
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="KIMI-U001",
        state="RUNNING",
        driver_pid=987654322,
        staging_workspace=str(tmp_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )

    owner.save(record)

    assert owner.load(record.workunit_id).state == "BLOCKED"
