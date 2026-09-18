"""Deterministic acceptance probes for the Arsenal A0/A1/B1 tranche."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from hawking.engine import Engine
from hawking.auto_orchestration import (
    RemoteWorkerAdmission,
    WorkerAdmissionError,
    build_cognition_plan,
    classify_request,
    worker_packet,
)
from hawking.auto_mode import CORE_AUTO_MODELS
from hawking.execution_binding import (
    BINDING_SCHEMA,
    build_candidate_manifest,
    persist_candidate_manifest,
)
from hawking.mutation import (
    MutationConflict,
    MutationError,
    MutationExecutor,
    MutationIntentStore,
    build_mutation_request,
    reconcile_mutation_intent,
)
from hawking.tool_registry import READ_ONLY, REVERSIBLE_REPO, default_tool_registry
from hawking.workspace import Workspace


def test_candidate_manifest_is_immutable_and_binds_current_authority(tmp_path: Path) -> None:
    manifest = build_candidate_manifest(
        tmp_path,
        goal_id="GOAL-A0",
        workunit_id="WU-A0",
        allowed_paths=["fixture.py"],
        verification_obligations={"tests": ["test_fixture.py"]},
        budget={"cap_usd": 0.0},
        resource_reservation={"class": "MUTATION", "slots": 1},
        capabilities=["fs.search", "repo.edit", "tests.run"],
    )
    payload = manifest.to_dict()
    assert payload["binding"]["schema"] == BINDING_SCHEMA
    assert payload["binding"]["goal_id"] == "GOAL-A0"
    assert payload["binding"]["allowed_paths"] == ["fixture.py"]
    path = persist_candidate_manifest(tmp_path, manifest)
    assert path.is_file()
    assert persist_candidate_manifest(tmp_path, manifest) == path
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_mutation_executor_refuses_a_stale_base(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("before\n", encoding="utf-8")
    request = build_mutation_request(
        tmp_path,
        [{"op": "replace_file", "path": "fixture.txt", "new_text": "after\n"}],
        metadata={
            "request_id": "REQ-stale",
            "expected_base_hashes": {"fixture.txt": hashlib.sha256(b"wrong\n").hexdigest()},
        },
    )
    with pytest.raises(MutationConflict):
        MutationExecutor(tmp_path).begin(request, [target])
    assert target.read_text(encoding="utf-8") == "before\n"
    assert not (tmp_path / ".hawking" / "mutation-intents" / "REQ-stale.json").exists()


def test_mutation_executor_enforces_the_admitted_path_scope(tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("before\n", encoding="utf-8")
    request = build_mutation_request(
        tmp_path,
        [{"op": "replace_file", "path": "outside.txt", "new_text": "after\n"}],
        metadata={"request_id": "REQ-scope", "allowed_paths": ["other.txt"]},
    )
    with pytest.raises(MutationError, match="outside admitted allowed_paths"):
        MutationExecutor(tmp_path).begin(request, [target])
    assert target.read_text(encoding="utf-8") == "before\n"
    assert not (tmp_path / ".hawking" / "mutation-intents" / "REQ-scope.json").exists()


def test_engine_records_verified_completion_packet_and_intent(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    test = tmp_path / "test_mod.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test.write_text(
        "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    engine = Engine(Workspace(str(tmp_path)))
    result = engine.apply_typed_mutation(
        [{"op": "replace", "path": "mod.py", "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]}],
        tests=["test_mod.py"],
        request={
            "request_id": "REQ-verified",
            "goal_id": "GOAL-A1",
            "workunit_id": "WU-A1",
            "worker_attempt_id": "ATTEMPT-1",
            "capability_id": "repo.edit",
        },
    )
    assert result["status"] == "accepted"
    assert result["intent_status"] == "VERIFIED"
    assert result["completion_packet"]["verification"]["diff_inspected"] is True
    assert result["completion_packet"]["verification"]["evidence_complete"] is True
    intent = json.loads(
        (tmp_path / ".hawking" / "mutation-intents" / "REQ-verified.json").read_text(encoding="utf-8")
    )
    assert intent["status"] == "VERIFIED"
    assert intent["request"]["goal_id"] == "GOAL-A1"
    assert intent["files"][0]["sha256_after"]


def test_filesystem_write_uses_the_same_typed_intent_executor(tmp_path: Path) -> None:
    registry = default_tool_registry(
        tmp_path,
        repo_root=tmp_path,
        permissions={READ_ONLY, REVERSIBLE_REPO},
    )
    result = registry.invoke(
        "filesystem.write",
        {
            "path": "fixture.txt",
            "content": "hello\n",
            "overwrite": False,
            "request_id": "REQ-fs-write",
            "goal_id": "GOAL-fs",
            "workunit_id": "WU-fs",
        },
    )
    assert result.ok, result.error
    assert result.value["intent_status"] == "VERIFIED"
    assert result.value["changed_files"][0]["path"] == "fixture.txt"


def test_unknown_outcome_reconciles_without_replaying(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("before\n", encoding="utf-8")
    request = build_mutation_request(
        tmp_path,
        [{"op": "replace_file", "path": "fixture.txt", "new_text": "after\n"}],
        metadata={"request_id": "REQ-reconcile"},
    )
    store = MutationIntentStore(tmp_path)
    store.write(
        request,
        status="UNKNOWN_OUTCOME",
        files=[{"path": "fixture.txt", "sha256_after": hashlib.sha256(b"after\n").hexdigest()}],
    )
    target.write_text("after\n", encoding="utf-8")
    result = reconcile_mutation_intent(tmp_path, "REQ-reconcile")
    # The bytes match the recorded post-state, but the intent carries no
    # durable verification-stage marker.  Recovery must not promote that to
    # VERIFIED after a crash between publish and the acceptance checks.
    assert result["status"] == "UNKNOWN_OUTCOME"
    assert result["reconciled"] is False
    assert result["candidate_matches_recorded_after"] is True
    assert target.read_text(encoding="utf-8") == "after\n"


def test_interrupted_applied_intent_never_claims_a_partial_publish(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("before\n", encoding="utf-8")
    request = build_mutation_request(
        tmp_path,
        [{"op": "replace_file", "path": "fixture.txt", "new_text": "after\n"}],
        metadata={"request_id": "REQ-partial"},
    )
    store = MutationIntentStore(tmp_path)
    store.write(
        request,
        status="APPLIED",
        files=[{
            "path": "fixture.txt",
            "sha256_before": hashlib.sha256(b"before\n").hexdigest(),
        }],
    )
    target.write_text("partial\n", encoding="utf-8")
    result = reconcile_mutation_intent(tmp_path, "REQ-partial")
    assert result["status"] == "UNKNOWN_OUTCOME"
    assert result["reconciled"] is False
    assert target.read_text(encoding="utf-8") == "partial\n"


def test_search_corpus_returns_scope_proof_and_explicit_expansion(tmp_path: Path) -> None:
    (tmp_path / "hawking").mkdir()
    (tmp_path / "receipts").mkdir()
    (tmp_path / "workspace" / "campaign").mkdir(parents=True)
    (tmp_path / "hawking" / "owner.py").write_text("marker source\n", encoding="utf-8")
    (tmp_path / "receipts" / "proof.json").write_text("marker evidence\n", encoding="utf-8")
    (tmp_path / "workspace" / "campaign" / "old.txt").write_text("marker campaign\n", encoding="utf-8")
    registry = default_tool_registry(tmp_path, repo_root=tmp_path, permissions={READ_ONLY})

    source = registry.invoke("fs.search", {"pattern": "marker", "corpus": "source"})
    assert source.ok, source.error
    assert {Path(row["path"]).name for row in source.value["matches"]} == {"owner.py"}
    assert source.value["searched_roots"]
    assert source.value["excluded_source_kinds"]
    assert source.value["expansion_route"]["next_corpus"] == "all-authorized"
    evidence = registry.invoke("fs.search", {"pattern": "marker", "corpus": "evidence"})
    assert evidence.ok, evidence.error
    assert {Path(row["path"]).name for row in evidence.value["matches"]} == {"proof.json"}
    assert registry.get("fs.search").input_schema["properties"]["corpus"]["enum"]


def test_search_scope_revision_changes_when_a_candidate_changes(tmp_path: Path) -> None:
    (tmp_path / "hawking").mkdir()
    target = tmp_path / "hawking" / "owner.py"
    target.write_text("one\n", encoding="utf-8")
    registry = default_tool_registry(tmp_path, repo_root=tmp_path, permissions={READ_ONLY})
    first = registry.invoke("fs.search", {"pattern": "one", "corpus": "source"})
    target.write_text("two\n", encoding="utf-8")
    second = registry.invoke("fs.search", {"pattern": "two", "corpus": "source"})
    assert first.ok and second.ok
    assert first.value["index_revision"] != second.value["index_revision"]


def test_search_scope_revision_does_not_use_mtime_as_content_identity(tmp_path: Path) -> None:
    (tmp_path / "hawking").mkdir()
    target = tmp_path / "hawking" / "owner.py"
    target.write_text("one\n", encoding="utf-8")
    registry = default_tool_registry(tmp_path, repo_root=tmp_path, permissions={READ_ONLY})
    first = registry.invoke("fs.search", {"pattern": "one", "corpus": "source"})
    original = target.stat()
    target.write_text("two\n", encoding="utf-8")
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    second = registry.invoke("fs.search", {"pattern": "two", "corpus": "source"})
    assert first.ok and second.ok
    assert first.value["index_revision"] != second.value["index_revision"]


def test_auto_worker_admission_bounds_remote_instances_without_sharing_identity() -> None:
    admission = RemoteWorkerAdmission(max_workers=2, max_premium_workers=1)
    first = admission.acquire(model="deepseek/deepseek-v4.1-flash", goal_id="A", workunit_id="A1")
    second = admission.acquire(model="deepseek/deepseek-v4.1-flash", goal_id="B", workunit_id="B1")
    assert first.worker_id != second.worker_id
    assert first.lease_id != second.lease_id
    with pytest.raises(WorkerAdmissionError):
        admission.acquire(model="moonshotai/kimi-k3", goal_id="C", workunit_id="C1")
    assert admission.release(first.lease_id) is True
    assert admission.release(first.lease_id) is False
    assert len(admission.snapshot()) == 1


def test_aggressive_auto_default_has_no_artificial_kimi_slot_ceiling() -> None:
    admission = RemoteWorkerAdmission()
    leases = [admission.acquire(model="moonshotai/kimi-k3") for _ in range(32)]
    assert len({lease.worker_id for lease in leases}) == 32
    assert admission.max_workers is None
    assert admission.max_premium_workers is None


def test_worker_packet_carries_capabilities_and_recoverable_identity() -> None:
    packet = worker_packet(
        {"goal_id": "GOAL-H1", "objective": "repair the repository", "task_class": "repo_engineering"},
        {"workunit_id": "WU-H1", "worker_model": "deepseek/deepseek-v4.1-flash"},
        authority={"capabilities": ["fs.search", "repo.edit", "tests.run"]},
    )
    assert packet["worker_id"].startswith("WORKER-")
    assert packet["worker_attempt_id"].startswith("ATTEMPT-")
    assert packet["capability_catalog"] == ["fs.search", "repo.edit", "tests.run"]
    assert packet["playbook"][-1] == "diff"
    assert packet["digest"]


def test_auto_classifies_new_world_interfaces_before_generic_review() -> None:
    desktop = classify_request(
        text="Implement the macOS accessibility action lease, then independently review its evidence."
    )
    document = classify_request(text="Extract tables from this scanned PDF using OCR only where needed.")
    reverse = classify_request(text="Use Rizin to inspect this authorized binary before decompiling it.")
    physical = classify_request(text="Measure eGPU device placement before changing the GPU lane.")

    assert desktop["task_class"] == "desktop"
    assert desktop["required_capabilities"] == ["macos.observe", "macos.changed"]
    assert document["task_class"] == "document"
    assert reverse["task_class"] == "reverse_engineering"
    assert physical["task_class"] == "physical_engineering"

    plan = build_cognition_plan(
        {"messages": [{"role": "user", "content": "Implement a macOS desktop accessibility fixture."}]}
    )
    assert plan["task"]["task_class"] == "desktop"
    # V2 routing is measured rather than DeepSeek-pinned; desktop work must
    # stay inside the admitted core pool while allowing Kimi/GLM diversity.
    assert plan["workers"][0]["model"] in CORE_AUTO_MODELS


def test_aggressive_auto_builds_parallel_plan_and_development_lanes() -> None:
    planning = build_cognition_plan(
        {"messages": [{"role": "user", "content": (
            "Create a long horizon architecture plan; decompose multiple independent "
            "workunits, review every phase, and build a parallel evidence graph."
        )}]},
        allowed_models=["deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"],
    )
    assert planning["topology"] == "parallel planning lanes → evidence convergence"
    assert len(planning["workers"]) == 8
    assert {row["model"] for row in planning["workers"]} == {
        "deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"
    }

    development = build_cognition_plan(
        {"messages": [{"role": "user", "content": (
            "Implement and test a repository repair, then audit multiple independent "
            "source owners and review the resulting Git diff."
        )}]},
        allowed_models=["deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"],
    )
    assert development["topology"] == "parallel development lanes; writers require disjoint effect scopes"
    assert len(development["workers"]) == 6
    assert development["policy"]["effect_scope_rule"].startswith("parallel writes")


def test_structural_source_tools_share_scope_and_never_write(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    source = tmp_path / "pkg" / "owner.py"
    source.write_text(
        "def canonical_owner(value):\n    return value\n\nresult = canonical_owner('x')\n",
        encoding="utf-8",
    )
    test = tmp_path / "tests" / "test_owner.py"
    test.write_text(
        "from pkg.owner import canonical_owner\n\ndef test_owner():\n    assert canonical_owner('x') == 'x'\n",
        encoding="utf-8",
    )
    registry = default_tool_registry(tmp_path, repo_root=tmp_path, permissions={READ_ONLY})

    owner = registry.invoke("source.owner", {"query": "canonical_owner"})
    assert owner.ok, owner.error
    assert owner.value["owner"]["path"] == "pkg/owner.py"
    outline = registry.invoke("source.outline", {"path": "pkg/owner.py"})
    assert outline.ok, outline.error
    assert outline.value["symbols"][0]["name"] == "canonical_owner"
    refs = registry.invoke("source.references", {"symbol": "canonical_owner"})
    assert refs.ok, refs.error
    assert refs.value["count"] >= 2
    ast_query = registry.invoke("source.ast_query", {"pattern": "call:canonical_owner"})
    assert ast_query.ok, ast_query.error
    assert ast_query.value["count"] >= 1
    affected = registry.invoke("source.affected_tests", {"paths": ["pkg/owner.py"]})
    assert affected.ok, affected.error
    assert affected.value["tests"][0]["path"] == "tests/test_owner.py"
    preview = registry.invoke(
        "source.rewrite_preview",
        {"scope": "pkg/owner.py", "pattern": "return value", "replacement": "return value.strip()"},
    )
    assert preview.ok, preview.error
    assert preview.value["would_write"] is False
    assert source.read_text(encoding="utf-8").endswith("return value\n\nresult = canonical_owner('x')\n")
