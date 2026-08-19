"""Live-controller proof: protected evidence -> rebind -> activation -> promotion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import lab.lineage.lifecycle as lifecycle
from lab.lineage.canon import labeled_sha
from lab.lineage.identity import GenesisInstance
from lab.lineage.lifecycle import (
    CANDIDATE_SCHEMA,
    PAIR_RECEIPT_SCHEMA,
    BenchmarkProfile,
    CandidateInbox,
    CandidateSpec,
    LifecycleError,
    PromotionController,
    WorkerRegistry,
)
from lab.lineage.state import LineageState
from lab.lineage.testing import science_payload
from lab.receipts import seal, verify


REPO = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executable(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _artifact(root: Path, *, bpw: float) -> tuple[Path, str]:
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
                "complete_physical_bpw": bpw,
            },
            sort_keys=True,
        )
    )
    return root, _sha(manifest)


def _instance(
    *,
    instance_id: str,
    generation: int,
    artifact_sha: str,
    resident_sha: str,
    kernel_sha: str,
    bpw: float,
    wall: int,
    artifact: Path,
) -> GenesisInstance:
    return GenesisInstance(
        instance_id=instance_id,
        generation=generation,
        artifact_sha=artifact_sha,
        runtime_sha=resident_sha,
        kernel_genome_sha=kernel_sha,
        representation_bpw=bpw,
        physical_bpw=bpw,
        complete_token_ns=wall,
        capability={
            "coherence": 1.0,
            "complete_token_discipline": 1.0,
            "engineering": 1.0,
        },
        benchmark_fingerprint=labeled_sha("fixture-benchmark"),
        identity={
            "model": "PocketAiHub/Qwen3.8-27B-Abliterated-MLX",
            "artifact": str(artifact),
        },
    )


def _pair(
    *,
    parent: BenchmarkProfile,
    child: BenchmarkProfile,
) -> dict:
    parent_reps = [35_400_000, 35_200_000, 35_300_000]
    child_reps = [30_100_000, 30_000_000, 29_900_000]
    greedy = [
        {"prompt": "The capital of France is", "parent_ids": [1, 2], "child_ids": [1, 2]},
        {"prompt": "2 + 2 =", "parent_ids": [3, 4], "child_ids": [3, 4]},
        {"prompt": "Once upon a time", "parent_ids": [5, 6], "child_ids": [5, 6]},
    ]
    return seal(
        {
            "schema": PAIR_RECEIPT_SCHEMA,
            "status": "PASS",
            "capture_origin_attested": True,
            "timing_authority": "MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "parent": parent.to_dict(),
            "child": child.to_dict(),
            "parent_complete_token_ns_reps": parent_reps,
            "child_complete_token_ns_reps": child_reps,
            "greedy_token_ids": greedy,
            "protected_tests": [
                {"name": "coherence_greedy_ids", "status": "PASS"},
                {"name": "complete_token_ledger_closed", "status": "PASS"},
                {"name": "no_silent_fallback", "status": "PASS"},
            ],
        }
    )


def _fixture(tmp_path: Path) -> dict:
    parent_artifact, parent_artifact_sha = _artifact(tmp_path / "parent-artifact", bpw=4.25)
    child_artifact, child_artifact_sha = _artifact(tmp_path / "child-artifact", bpw=4.0)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    resident = _executable(tmp_path / "genesis-resident", b"resident stable binary\n")
    parent_bench = _executable(tmp_path / "parent-benchmark", b"parent benchmark\n")
    child_bench = _executable(tmp_path / "child-benchmark", b"child benchmark\n")
    parent_kernel = tmp_path / "parent.metal"
    parent_kernel.write_text("kernel parent")

    parent = _instance(
        instance_id="genesis-g0",
        generation=0,
        artifact_sha=parent_artifact_sha,
        resident_sha=_sha(resident),
        kernel_sha=_sha(parent_kernel),
        bpw=4.25,
        wall=35_300_000,
        artifact=parent_artifact,
    )
    parent.identity.update(
        {
            "protected_verification": str(tmp_path / "parent-verification.json"),
            "resident_executable": str(resident),
            "tokenizer": str(tokenizer),
        }
    )
    (tmp_path / "parent-verification.json").write_text(
        json.dumps(
            {
                "protected_binding": {
                    "artifact_manifest_sha256": parent_artifact_sha,
                    "runtime_executable_path": str(parent_bench),
                    "runtime_executable_sha256": _sha(parent_bench),
                    "kernel_source_path": str(parent_kernel),
                    "kernel_source_sha256": _sha(parent_kernel),
                }
            }
        )
    )
    child = _instance(
        instance_id="genesis-g1",
        generation=1,
        artifact_sha=child_artifact_sha,
        resident_sha=_sha(resident),
        kernel_sha=_sha(parent_kernel),
        bpw=4.0,
        wall=30_000_000,
        artifact=child_artifact,
    )

    lineage = LineageState()
    lineage.install(parent)
    state = tmp_path / "GENESIS_LINEAGE_CURRENT.json"
    state.write_text(json.dumps(lineage.to_dict(), indent=2))
    request = {
        "schema": CANDIDATE_SCHEMA,
        "candidate": child.to_dict(),
        "artifact_root": str(child_artifact),
        "tokenizer": str(tokenizer),
        "resident_executable": str(resident),
        "benchmark_runtime": str(child_bench),
        "kernel_source": str(parent_kernel),
        "parent_payload": science_payload(),
        "world_state": {"fixture": True},
    }
    request_path = tmp_path / "candidate.json"
    request_path.write_text(json.dumps(request, indent=2))
    parent_profile = BenchmarkProfile(parent, parent_artifact, tokenizer, parent_bench, parent_kernel)
    spec = CandidateSpec.from_mapping(request, repo=REPO)
    return {
        "parent": parent,
        "child": child,
        "state": state,
        "request": request_path,
        "spec": spec,
        "parent_profile": parent_profile,
        "registry": WorkerRegistry(tmp_path / "workers.json"),
        "checkpoint_root": tmp_path / "checkpoints",
        "candidate_root": tmp_path / "candidate-results",
    }


def test_full_live_controller_rebinds_before_sacrificial_promotion(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    active = {"instance": fx["parent"]}

    def health() -> dict:
        instance = active["instance"]
        return {
            "ok": True,
            "body_resident": True,
            "artifact_sha": instance.artifact_sha,
            "generation": instance.generation,
            "reload_error": None,
        }

    def activate(spec: CandidateSpec) -> dict:
        active["instance"] = spec.instance
        return health()

    def benchmark(parent: BenchmarkProfile, child: BenchmarkProfile, _out: Path) -> dict:
        assert parent.instance.instance_id == fx["parent"].instance_id
        assert child.instance.instance_id == fx["child"].instance_id
        return _pair(parent=parent, child=child)

    controller = PromotionController(
        repo=REPO,
        state_path=fx["state"],
        worker_registry=fx["registry"],
        checkpoint_root=fx["checkpoint_root"],
        candidate_root=fx["candidate_root"],
        health=health,
        activate=activate,
        benchmark=benchmark,
    )
    workers = controller.bootstrap_workers()
    assert {row["session_role"] for row in workers} == {"child_a", "child_b"}

    result = controller.promote(fx["request"])
    verify(result, label="lifecycle result")
    assert result["outcome"] == "PROMOTED"
    assert result["authority_moved"] is True
    assert active["instance"].instance_id == fx["child"].instance_id
    state = json.loads(fx["state"].read_text())
    assert state["slots"]["CURRENT"]["instance_id"] == fx["child"].instance_id
    assert state["slots"]["LAST_KNOWN_GOOD"]["instance_id"] == fx["parent"].instance_id
    assert state["slots"]["LAST_KNOWN_GOOD"]["terminated"] is True
    rebound, _ = fx["registry"].load()
    assert {row["bound_generation"]["generation"] for row in rebound} == {1}
    assert (fx["checkpoint_root"] / "gravity" / "checkpoint.json").is_file()
    assert (fx["checkpoint_root"] / "kernel" / "checkpoint.json").is_file()


def test_controller_derives_child_token_wall_from_protected_pair_not_worker_guess(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    request = json.loads(fx["request"].read_text())
    # A sandbox worker may estimate this field, but only the protected pair
    # gets to establish it. A deliberately wrong positive estimate must not
    # prevent a valid candidate from ever reaching the external gate.
    request["candidate"]["complete_token_ns"] = 1
    fx["request"].write_text(json.dumps(request))
    active = {"instance": fx["parent"]}

    def health() -> dict:
        instance = active["instance"]
        return {
            "ok": True,
            "body_resident": True,
            "artifact_sha": instance.artifact_sha,
            "generation": instance.generation,
            "reload_error": None,
        }

    def activate(spec: CandidateSpec) -> dict:
        active["instance"] = spec.instance
        return health()

    controller = PromotionController(
        repo=REPO,
        state_path=fx["state"],
        worker_registry=fx["registry"],
        checkpoint_root=fx["checkpoint_root"],
        candidate_root=fx["candidate_root"],
        health=health,
        activate=activate,
        benchmark=lambda parent, child, _out: _pair(parent=parent, child=child),
    )
    controller.bootstrap_workers()
    result = controller.promote(fx["request"])
    assert result["outcome"] == "PROMOTED"
    state = json.loads(fx["state"].read_text())
    child = state["slots"]["CURRENT"]
    assert child["complete_token_ns"] == 30_000_000
    assert child["identity"]["candidate_declared_complete_token_ns"] == "1"
    assert child["identity"]["complete_token_ns_authority"] == "protected_pair_upper_median"


def test_candidate_builder_stages_real_files_without_claiming_protected_speed(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    request = lifecycle.build_candidate_request(
        authority_repo=REPO,
        state_path=fx["state"],
        origin_repo=tmp_path,
        instance_id="genesis-g1-staged",
        artifact_root=fx["spec"].artifact_root,
        next_bottleneck="test candidate representation against protected complete-token pair",
    )
    spec = CandidateSpec.from_mapping(request, repo=tmp_path)
    assert spec.instance.generation == 1
    # The parent wall is explicitly only a placeholder; the controller replaces
    # it with the protected paired median before the gate can inspect it.
    assert spec.instance.complete_token_ns == fx["parent"].complete_token_ns
    assert spec.instance.identity["candidate_wall_is_untrusted"] == "controller derives protected paired median"
    assert request["world_state"]["candidate_draft"]["prepared_by"] == "agentos_hcli"


def test_candidate_activation_failure_keeps_parent_authoritative_and_workers_unmoved(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    active = {"instance": fx["parent"]}

    def health() -> dict:
        instance = active["instance"]
        return {
            "ok": True,
            "body_resident": True,
            "artifact_sha": instance.artifact_sha,
            "generation": instance.generation,
            "reload_error": None,
        }

    controller = PromotionController(
        repo=REPO,
        state_path=fx["state"],
        worker_registry=fx["registry"],
        checkpoint_root=fx["checkpoint_root"],
        candidate_root=fx["candidate_root"],
        health=health,
        activate=lambda _spec: health(),
        benchmark=lambda parent, child, _out: _pair(parent=parent, child=child),
    )
    before = controller.bootstrap_workers()
    with pytest.raises(LifecycleError, match="activation failed"):
        controller.promote(fx["request"])
    state = json.loads(fx["state"].read_text())
    assert state["slots"]["CURRENT"]["instance_id"] == fx["parent"].instance_id
    assert active["instance"].instance_id == fx["parent"].instance_id
    after, _ = fx["registry"].load()
    assert after == before


def test_runtime_binary_change_rebinds_then_promotes_via_managed_restart(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    changed = tmp_path / "new-resident"
    _executable(changed, b"new resident binary\n")
    request = json.loads(fx["request"].read_text())
    request["resident_executable"] = str(changed)
    request["candidate"]["runtime_sha"] = _sha(changed)
    fx["request"].write_text(json.dumps(request))
    active = {"instance": fx["parent"]}
    observed: list[str] = []

    def health() -> dict:
        instance = active["instance"]
        return {
            "ok": True,
            "body_resident": True,
            "artifact_sha": instance.artifact_sha,
            "generation": instance.generation,
            "reload_error": None,
        }

    def restart(spec: CandidateSpec) -> dict:
        # The old parent stayed authoritative while both AgentOS workers were
        # checkpointed/rebound. The state now selects G1 but explicitly does
        # not claim it is live until this observed start succeeds.
        state = json.loads(fx["state"].read_text())
        assert state["slots"]["CURRENT"]["instance_id"] == spec.instance.instance_id
        assert state["slots"]["CURRENT"]["live"] is False
        assert state["slots"]["LAST_KNOWN_GOOD"]["terminated"] is False
        rebound, _ = fx["registry"].load()
        assert {row["bound_generation"]["generation"] for row in rebound} == {1}
        assert active["instance"].instance_id == fx["parent"].instance_id
        observed.append("parent_rebound_then_stopped")
        active["instance"] = spec.instance
        return health()

    controller = PromotionController(
        repo=REPO,
        state_path=fx["state"],
        worker_registry=fx["registry"],
        checkpoint_root=fx["checkpoint_root"],
        candidate_root=fx["candidate_root"],
        health=health,
        activate=lambda _spec: (_ for _ in ()).throw(AssertionError("must not hot reload runtime")),
        restart=restart,
        benchmark=lambda parent, child, _out: _pair(parent=parent, child=child),
    )
    controller.bootstrap_workers()
    result = controller.promote(fx["request"])
    verify(result, label="managed-runtime-promotion")
    assert result["activation_mode"] == "managed_exec_restart"
    assert observed == ["parent_rebound_then_stopped"]
    assert active["instance"].instance_id == fx["child"].instance_id
    state = json.loads(fx["state"].read_text())
    assert state["slots"]["CURRENT"]["instance_id"] == fx["child"].instance_id
    assert state["slots"]["CURRENT"]["live"] is True
    assert state["slots"]["LAST_KNOWN_GOOD"]["terminated"] is True


def test_runtime_candidate_failure_rolls_back_state_and_worker_bindings(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    changed = tmp_path / "bad-resident"
    _executable(changed, b"bad resident binary\n")
    request = json.loads(fx["request"].read_text())
    request["resident_executable"] = str(changed)
    request["candidate"]["runtime_sha"] = _sha(changed)
    fx["request"].write_text(json.dumps(request))
    active = {"instance": fx["parent"]}
    stops: list[str] = []

    def health() -> dict:
        instance = active["instance"]
        return {
            "ok": True,
            "body_resident": True,
            "artifact_sha": instance.artifact_sha,
            "generation": instance.generation,
            "reload_error": None,
        }

    controller = PromotionController(
        repo=REPO,
        state_path=fx["state"],
        worker_registry=fx["registry"],
        checkpoint_root=fx["checkpoint_root"],
        candidate_root=fx["candidate_root"],
        health=health,
        restart=lambda _spec: health(),
        stop=lambda: stops.append("non-authoritative-body-stopped") or True,
        benchmark=lambda parent, child, _out: _pair(parent=parent, child=child),
    )
    before = controller.bootstrap_workers()
    with pytest.raises(LifecycleError, match="CURRENT was restored"):
        controller.promote(fx["request"])
    state = json.loads(fx["state"].read_text())
    assert state["slots"]["CURRENT"]["instance_id"] == fx["parent"].instance_id
    after, _ = fx["registry"].load()
    assert after == before
    assert stops == ["non-authoritative-body-stopped"]


def test_kernel_change_without_runtime_replacement_is_refused(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    changed_kernel = tmp_path / "changed.metal"
    changed_kernel.write_text("new kernel source")
    request = json.loads(fx["request"].read_text())
    request["kernel_source"] = str(changed_kernel)
    request["candidate"]["kernel_genome_sha"] = _sha(changed_kernel)
    fx["request"].write_text(json.dumps(request))
    controller = PromotionController(
        repo=REPO,
        state_path=fx["state"],
        worker_registry=fx["registry"],
        checkpoint_root=fx["checkpoint_root"],
        candidate_root=fx["candidate_root"],
        health=lambda: {
            "ok": True,
            "body_resident": True,
            "artifact_sha": fx["parent"].artifact_sha,
            "generation": 0,
            "reload_error": None,
        },
        benchmark=lambda *_args: (_ for _ in ()).throw(AssertionError("must not benchmark")),
    )
    with pytest.raises(LifecycleError, match="kernel genome"):
        controller.promote(fx["request"])


def test_candidate_inbox_preserves_request_through_claim_and_failure_archive(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    raw = fx["request"].read_bytes()
    inbox = CandidateInbox(tmp_path / "candidate-inbox")
    submitted = inbox.submit(fx["request"], repo=REPO)
    assert submitted.read_bytes() == raw
    assert inbox.status()["inbox"] == [submitted.name]
    active = inbox.claim_next()
    assert active is not None
    assert active.read_bytes() == raw
    assert inbox.status()["inbox"] == []
    assert inbox.status()["active"] == [active.name]
    archived = inbox.complete(
        active,
        outcome="failed",
        record={"schema": "fixture", "error": "controller refused fixture"},
    )
    assert archived.read_bytes() == raw
    assert archived.with_suffix(".lifecycle.json").is_file()
    assert inbox.status()["active"] == []


def test_candidate_inbox_seals_the_isolated_worker_origin_for_relative_paths(
    tmp_path: Path,
) -> None:
    origin = tmp_path / "gravity-worktree"
    origin.mkdir()
    fx = _fixture(origin)
    request = json.loads(fx["request"].read_text())
    for key in (
        "artifact_root",
        "tokenizer",
        "resident_executable",
        "benchmark_runtime",
        "kernel_source",
    ):
        request[key] = str(Path(request[key]).relative_to(origin))
    relative_request = origin / "candidate-relative.json"
    relative_request.write_text(json.dumps(request, indent=2))
    inbox = CandidateInbox(tmp_path / "candidate-inbox")
    submitted = inbox.submit(relative_request, repo=origin)
    active = inbox.claim_next()
    assert active is not None
    assert inbox.origin_repo_for(active, fallback_repo=REPO) == origin.resolve()
    archived = inbox.complete(active, outcome="failed", record={"schema": "fixture"})
    assert archived.with_suffix(".submission.json").is_file()
    assert not (inbox.submissions / f"{submitted.name}.submission.json").exists()


def test_external_inbox_controller_uses_the_sealed_worker_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = tmp_path / "kernel-worktree"
    origin.mkdir()
    fx = _fixture(origin)
    request = json.loads(fx["request"].read_text())
    for key in (
        "artifact_root",
        "tokenizer",
        "resident_executable",
        "benchmark_runtime",
        "kernel_source",
    ):
        request[key] = str(Path(request[key]).relative_to(origin))
    request_path = origin / "relative-request.json"
    request_path.write_text(json.dumps(request))
    inbox_root = tmp_path / "candidate-inbox"
    CandidateInbox(inbox_root).submit(request_path, repo=origin)
    seen: dict[str, Path] = {}

    class FakeController:
        def __init__(self, **_kwargs) -> None:
            pass

        def bootstrap_workers(self) -> list[dict]:
            return []

        def promote(self, _request: Path, *, candidate_repo: Path | None = None) -> dict:
            assert candidate_repo is not None
            seen["origin"] = candidate_repo
            return {"outcome": "REJECT", "authority_moved": False}

    monkeypatch.setattr(lifecycle, "PromotionController", FakeController)
    result = lifecycle.process_candidate_inbox_once(
        repo=REPO,
        candidate_root=inbox_root,
        state_path=fx["state"],
        worker_registry_path=fx["registry"].path,
        checkpoint_root=fx["checkpoint_root"],
    )
    assert result["outcome"] == "NOT_PROMOTED"
    assert seen["origin"] == origin.resolve()


def test_worker_registry_provisions_separate_sparse_homes_without_resetting_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _fixture(tmp_path)["parent"]
    registry = WorkerRegistry(tmp_path / "workers.json")
    registry.bootstrap(
        generation=lifecycle.generation_record(parent, repo_head="0" * 40),
        repo=REPO,
    )
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_git(repo: Path, argv, *, label: str) -> str:
        args = tuple(str(item) for item in argv)
        calls.append((Path(repo), args))
        if args == ("rev-parse", "--show-toplevel"):
            return str(Path(repo).resolve())
        if args[:4] == ("worktree", "add", "--detach", "--no-checkout"):
            Path(args[-2]).mkdir(parents=True)
        return ""

    monkeypatch.setattr(lifecycle, "_git_checked", fake_git)
    homes = tmp_path / "isolated-homes"
    workers = registry.provision_worktrees(repo=REPO, worktree_root=homes)
    assert {row["worker_id"] for row in workers} == {"gravity", "kernel"}
    for worker in workers:
        durable = worker["durable_task_state"]
        assert durable["worktree_isolated"] is True
        assert Path(durable["worktree"]).parent == homes
        assert durable["worktree"] != str(REPO)
    assert sum(1 for _repo, args in calls if args[:2] == ("worktree", "add")) == 2
    assert all("reset" not in args for _repo, args in calls)
