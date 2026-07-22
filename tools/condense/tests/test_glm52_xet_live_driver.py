#!/usr/bin/env python3.12
"""Zero-network adversarial tests for the production live-Xet driver."""
from __future__ import annotations

import copy
import json
import os
import pathlib
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest


TESTS = pathlib.Path(__file__).resolve().parent
CONDENSE = TESTS.parent
for entry in (str(CONDENSE), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import glm52_state as state  # noqa: E402
import glm52_xet_autotune as autotune  # noqa: E402
import glm52_xet_live as live  # noqa: E402
import glm52_xet_live_driver as driver  # noqa: E402
import test_glm52_xet_autotune as authority_helpers  # noqa: E402
import test_glm52_xet_live as live_helpers  # noqa: E402
from glm52_common import Glm52Error, seal  # noqa: E402


class _Events:
    def __init__(self, events: Sequence[Mapping[str, Any]]) -> None:
        self.events = [copy.deepcopy(dict(item)) for item in events]

    def verified_events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.events)


class _Lease:
    def __init__(self, path: pathlib.Path, epoch: str) -> None:
        path.write_text("{}", encoding="utf-8")
        self.path = path
        self.owner = "glm52-controller"
        self._handle = path.open("r+", encoding="utf-8")
        self.epoch = epoch

    def assert_held(self) -> None:
        assert self._handle is not None

    def probe(self) -> dict[str, Any]:
        return {
            "lock_state": "HELD_BY_THIS_HANDLE",
            "live_lock_held": True,
            "held_by_this_handle": True,
            "owner_record_ok": True,
            "owner": self.owner,
            "owner_pid": os.getpid(),
            "owner_pid_alive": True,
            "controller_epoch": self.epoch,
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class _Controller:
    def __init__(
        self,
        authority: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        lease: _Lease,
    ) -> None:
        self.campaign_id = authority["campaign_id"]
        self.source_revision = authority["revision"]
        self.controller_epoch = authority["controller_epoch"]
        self.expected_contract_sha256 = authority["expected_contract_sha256"]
        self.lease = lease
        count = checkpoint["event_count"]
        last = {
            "seq": count - 1,
            "kind": "STATE_TRANSITION",
            "chain_sha256": checkpoint["event_head_hash"],
            "payload": {
                "to_state": "AUTOTUNE_XET",
                "transition_intent": copy.deepcopy(authority["transition_intent"]),
                "telegram_delivery": copy.deepcopy(
                    authority["telegram_delivery_receipt"]
                ),
            },
        }
        self.events = _Events([{} for _ in range(count - 1)] + [last])
        self.checkpoint = copy.deepcopy(dict(checkpoint))

    def resume(self, *, recover_single_tail: bool) -> dict[str, Any]:
        assert recover_single_tail is False
        return copy.deepcopy(self.checkpoint)

    def status(self) -> dict[str, Any]:
        return {
            "durable_state_ok": True,
            "state": "AUTOTUNE_XET",
            "live_worker_lease_ok": True,
        }

    def _controller_anchor(self, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        assert checkpoint["seal_sha256"] == self.checkpoint["seal_sha256"]
        return {"anchor_sha256": "7" * 64}


def _concrete_authority_world(tmp_path: pathlib.Path):
    plan = live_helpers._fake_plan()
    authority = authority_helpers._authority(plan)
    ref = authority["committed_controller_checkpoint"]
    checkpoint = {
        "schema": ref["checkpoint_schema"],
        "campaign_id": ref["campaign_id"],
        "source_revision": ref["source_revision"],
        "controller_epoch": ref["controller_epoch"],
        "expected_contract_sha256": ref["expected_contract_sha256"],
        "state": ref["state"],
        "last_claim_id": ref["last_claim_id"],
        "event_count": ref["event_count"],
        "event_head_hash": ref["event_head_hash"],
        "window_event_count": ref["window_event_count"],
        "window_event_head_hash": ref["window_event_head_hash"],
        "seal_sha256": ref["checkpoint_seal_sha256"],
    }
    runtime = SimpleNamespace(
        campaign_id=authority["campaign_id"],
        expected_contract={
            "seal_sha256": authority["expected_contract_sha256"],
            "state_gates": {
                "AUTOTUNE_XET": {
                    "required_artifacts": {
                        "xet_autotune_plan": {
                            "expected_seal_sha256": plan["seal_sha256"]
                        }
                    }
                }
            },
        },
        telegram_auth=authority_helpers._auth_config(),
    )
    lease = _Lease(tmp_path / "controller.lease", authority["controller_epoch"])
    controller = _Controller(authority, checkpoint, lease)
    return plan, authority, runtime, controller, lease


def test_production_authority_verifier_replays_exact_event_checkpoint_and_lease(
    tmp_path: pathlib.Path,
) -> None:
    plan, authority, runtime, controller, lease = _concrete_authority_world(tmp_path)
    try:
        verifier = driver.ProductionExecutionAuthorityVerifier(runtime, controller)
        assert autotune.validate_execution_authority(
            authority,
            plan_seal=plan["seal_sha256"],
            expected_contract_sha256=authority["expected_contract_sha256"],
            verifier=verifier,
        ) == authority
        binding = verifier.current_binding(
            intent=authority["transition_intent"],
            receipt=authority["telegram_delivery_receipt"],
        )
        assert binding["checkpoint_seal_sha256"] == (
            authority["committed_controller_checkpoint"]["checkpoint_seal_sha256"]
        )
        assert binding["telegram_receipt_seal_sha256"] == driver._sha(
            authority["telegram_delivery_receipt"]
        )
        assert driver._is_sha256(binding["lease_identity_sha256"])
    finally:
        lease.close()


@pytest.mark.parametrize("failed_gate", ["intent", "telegram", "checkpoint", "lease"])
def test_each_production_authority_mismatch_refuses_before_any_live_process(
    failed_gate: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, authority, runtime, controller, lease = _concrete_authority_world(tmp_path)
    spawned: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        spawned.append("spawned")
        raise AssertionError("live child/process path was reached")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(live, "execute_trial", forbidden)
    if failed_gate == "intent":
        runtime.telegram_auth = state.TelegramAuthConfig(
            hmac_key=b"different-production-verifier-test-key-material!!",
            expected_chat_identity_digest=runtime.telegram_auth.expected_chat_identity_digest,
        )
    elif failed_gate == "telegram":
        controller.events.events[-1]["payload"]["telegram_delivery"] = copy.deepcopy(
            authority["telegram_delivery_receipt"]
        )
        controller.events.events[-1]["payload"]["telegram_delivery"]["message_id"] += 1
    elif failed_gate == "checkpoint":
        controller.checkpoint["seal_sha256"] = "f" * 64
    else:
        monkeypatch.setattr(
            lease,
            "probe",
            lambda: {
                **_Lease.probe(lease),
                "live_lock_held": False,
                "held_by_this_handle": False,
            },
        )
    try:
        verifier = driver.ProductionExecutionAuthorityVerifier(runtime, controller)
        with pytest.raises(Glm52Error, match="independent"):
            autotune.validate_execution_authority(
                authority,
                plan_seal=plan["seal_sha256"],
                expected_contract_sha256=authority["expected_contract_sha256"],
                verifier=verifier,
            )
        assert spawned == []
    finally:
        lease.close()


@pytest.mark.parametrize("case", ["absent", "mismatched"])
def test_authority_refusal_precedes_plan_probe_git_resource_and_live_child(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = live_helpers._fake_plan()
    contract = {"seal_sha256": "c" * 64}
    runtime = SimpleNamespace()
    controller = SimpleNamespace()
    authority = None if case == "absent" else seal({"schema": "wrong"})
    calls: list[str] = []

    monkeypatch.setenv(live.EXECUTE_ENV, "1")
    monkeypatch.setattr(
        driver,
        "_require_official_runtime",
        lambda *_args, **_kwargs: contract,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("forbidden")
        raise AssertionError("post-authority execution path was reached")

    monkeypatch.setattr(live, "validate_live_plan", forbidden)
    monkeypatch.setattr(live, "execute_trial", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    class NoGit:
        def preflight(self) -> dict[str, Any]:
            return forbidden()

    class NoSample:
        def sample(self, _pid: int) -> Mapping[str, Any]:
            return forbidden()

    with pytest.raises(Glm52Error):
        driver._prepare_under_lease(
            runtime,
            controller,
            plan,
            contract,
            authority,
            resource_sampler=NoSample(),
            provenance=NoGit(),
        )
    assert calls == []


class _Provenance:
    def __init__(self) -> None:
        self.receipt = seal({"schema": driver.PROVENANCE_SCHEMA, "status": "PASS"})
        self.checks = 0
        self.final_checks = 0

    def assert_current(self, receipt: Mapping[str, Any]) -> None:
        assert dict(receipt) == self.receipt
        self.checks += 1

    def assert_final_current(self, receipt: Mapping[str, Any]) -> None:
        assert dict(receipt) == self.receipt
        self.final_checks += 1


class _AuthorityBinding:
    def __init__(self) -> None:
        self.binding = {
            "controller_epoch": autotune.CONTROLLER_EPOCH,
            "checkpoint_seal_sha256": "1" * 64,
            "lease_identity_sha256": "2" * 64,
            "telegram_receipt_seal_sha256": "3" * 64,
        }

    def current_binding(self, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(self.binding)


class _ResourceGuard:
    def __init__(self) -> None:
        self.baseline = live_helpers._resource_sample(os.getpid())
        self.last = copy.deepcopy(self.baseline)
        self.checks = 0

    def assert_safe(self) -> dict[str, Any]:
        self.checks += 1
        return copy.deepcopy(self.last)

    def sample(self, pid: int) -> Mapping[str, Any]:
        value = copy.deepcopy(self.last)
        value["pid"] = pid
        return value


class _MemoryWriter:
    def __init__(self) -> None:
        self.binding: dict[str, Any] | None = None
        self.values: dict[str, dict[str, Any]] = {}
        self.terminal: dict[str, Any] | None = None

    def begin(self, binding: Mapping[str, Any]) -> None:
        assert self.binding is None
        self.binding = copy.deepcopy(dict(binding))

    def write(self, name: str, value: Mapping[str, Any]) -> None:
        assert name not in self.values
        self.values[name] = copy.deepcopy(dict(value))

    def finish(self, receipt: Mapping[str, Any]) -> None:
        self.terminal = copy.deepcopy(dict(receipt))


def test_orchestration_executes_exact_12_then_two_full_hashes_and_attests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = live_helpers._fake_plan()

    def accept_plan(
        candidate: Mapping[str, Any],
        *,
        root: pathlib.Path = live.REPO_ROOT,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        del root, rebuild
        return copy.deepcopy(dict(candidate))

    monkeypatch.setattr(live, "validate_live_plan", accept_plan)
    monkeypatch.setattr(
        autotune,
        "validate_execution_authority",
        lambda authority, **_kwargs: copy.deepcopy(dict(authority)),
    )
    calls: list[str] = []

    def execute(
        candidate_plan: Mapping[str, Any],
        spec: Mapping[str, Any],
        capability: Mapping[str, Any],
        *,
        capability_verifier: Any,
        resource_sampler: Any,
        network_counter: Any,
        root: pathlib.Path,
    ) -> dict[str, Any]:
        del resource_sampler, network_counter, root
        assert capability_verifier.verify_live_capability(
            capability, plan=candidate_plan, spec=spec
        ) is True
        trial_id = spec["trial"]["trial_id"]
        calls.append(trial_id)
        largest_digest = (
            candidate_plan["largest_shard_validation"]["lfs_sha256"]
            if spec["trial"]["kind"] == "FULL_LARGEST_SHARD_VALIDATION"
            else None
        )
        return live_helpers._result_for_spec(
            spec,
            elapsed=1.0 + len(calls),
            actual_network=spec["trial"]["planned_payload_bytes"],
            largest_digest=largest_digest,
        )

    monkeypatch.setattr(live, "execute_trial", execute)
    monkeypatch.setattr(
        driver.schedule_freeze,
        "attest_xet_autotune_result",
        lambda raw, *_args, **_kwargs: seal({
            "schema": "test.attested.v1",
            "status": "PASS",
            "raw_result_seal_sha256": raw["seal_sha256"],
        }),
    )
    evidence_auth = state.EvidenceAuthConfig(
        hmac_key=b"driver-test-evidence-key-material-is-long-enough!!",
        campaign_id=driver.OFFICIAL_CAMPAIGN_ID,
        source_revision=live.REVISION,
    )
    runtime = SimpleNamespace(
        campaign_id=driver.OFFICIAL_CAMPAIGN_ID,
        source_revision=live.REVISION,
        controller_epoch=autotune.CONTROLLER_EPOCH,
        evidence_auth=evidence_auth,
    )
    checkpoint = {"seal_sha256": "1" * 64}
    controller = SimpleNamespace(
        resume=lambda recover_single_tail: copy.deepcopy(checkpoint),
        _controller_anchor=lambda value: {"anchor_sha256": "7" * 64},
    )
    authority = seal({
        "transition_intent": {"seal_sha256": "4" * 64},
        "telegram_delivery_receipt": {"hmac_sha256": "5" * 64},
    })
    provenance = _Provenance()
    guard = _ResourceGuard()
    prepared = driver.PreparedRun(
        plan=plan,
        contract={"seal_sha256": "6" * 64},
        authority=authority,
        authority_verifier=_AuthorityBinding(),
        provenance=provenance,
        provenance_receipt=provenance.receipt,
        resource_guard=guard,
        resource_baseline=copy.deepcopy(guard.baseline),
    )
    writer = _MemoryWriter()
    outcome = driver._execute_prepared(
        runtime,
        controller,
        prepared,
        network_counter=SimpleNamespace(),
        writer=writer,
    )

    expected = [row["trial_id"] for row in plan["trial_matrix"]]
    assert calls[:12] == expected
    assert calls[12:] == ["LARGEST_ACQUISITION", "LARGEST_STEADY"]
    assert len(calls) == 14
    assert outcome["raw_result"]["coverage"]["trial_ids_in_plan_order"] == expected
    assert outcome["driver_receipt"]["coverage"] == {
        "trial_ids": expected,
        "trial_count": 12,
        "largest_validation_lanes": ["acquisition", "steady"],
        "largest_validation_count": 2,
    }
    assert writer.terminal == outcome["driver_receipt"]
    assert driver.RAW_RESULT_NAME in writer.values
    assert driver.ATTESTED_RESULT_NAME in writer.values
    assert driver.DRIVER_RECEIPT_NAME in writer.values
    assert provenance.checks >= 28
    assert provenance.final_checks == 1
    assert guard.checks == 16


def test_capability_is_issued_in_process_exactly_once_and_rechecks_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = live_helpers._fake_plan()
    monkeypatch.setattr(
        live,
        "validate_live_plan",
        lambda candidate, **_kwargs: copy.deepcopy(dict(candidate)),
    )
    provenance = _Provenance()
    authority_verifier = _AuthorityBinding()
    authority = {
        "transition_intent": {"seal_sha256": "4" * 64},
        "telegram_delivery_receipt": {"hmac_sha256": "5" * 64},
    }
    issuer = driver.ProductionCapabilityIssuerVerifier(
        authority_verifier=authority_verifier,
        authority=authority,
        provenance=provenance,
        provenance_receipt=provenance.receipt,
    )
    cap = issuer.issue(
        plan=plan,
        trial_id="DEFAULT_UNSET",
        kind="BOUNDED_XET_BODY_RANGE",
        maximum_network_bytes=10_000,
        timeout_seconds=10,
    )
    spec = live.build_trial_spec(
        plan,
        "DEFAULT_UNSET",
        capability_seal_sha256=cap["seal_sha256"],
        campaign_consumed_bytes=0,
        trial_network_cap_bytes=10_000,
        rebuild_plan=False,
    )
    assert issuer.verify_live_capability(cap, plan=plan, spec=spec) is True
    assert issuer.verify_live_capability(cap, plan=plan, spec=spec) is False


def test_terminal_currentness_refuses_a_different_deterministic_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = live_helpers._fake_plan()
    rebuilt = copy.deepcopy(plan)
    rebuilt["status"] = "DIFFERENT"
    monkeypatch.setattr(live, "validate_live_plan", lambda *_args, **_kwargs: rebuilt)
    provenance = _Provenance()
    binding = _AuthorityBinding()
    guard = _ResourceGuard()
    prepared = driver.PreparedRun(
        plan=plan,
        contract={"seal_sha256": "6" * 64},
        authority=seal({
            "transition_intent": {"seal_sha256": "4" * 64},
            "telegram_delivery_receipt": {"hmac_sha256": "5" * 64},
        }),
        authority_verifier=binding,
        provenance=provenance,
        provenance_receipt=provenance.receipt,
        resource_guard=guard,
        resource_baseline=copy.deepcopy(guard.baseline),
    )
    with pytest.raises(driver.DriverError, match="rebuild differs"):
        driver._terminal_currentness(
            prepared, initial_authority_binding=binding.binding
        )
    assert provenance.final_checks == 0


def test_terminal_currentness_repeats_remote_then_refuses_lease_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = live_helpers._fake_plan()
    monkeypatch.setattr(
        live,
        "validate_live_plan",
        lambda candidate, **_kwargs: copy.deepcopy(dict(candidate)),
    )
    monkeypatch.setattr(
        autotune,
        "validate_execution_authority",
        lambda authority, **_kwargs: copy.deepcopy(dict(authority)),
    )
    provenance = _Provenance()
    binding = _AuthorityBinding()
    initial = copy.deepcopy(binding.binding)
    binding.binding["lease_identity_sha256"] = "9" * 64
    guard = _ResourceGuard()
    prepared = driver.PreparedRun(
        plan=plan,
        contract={"seal_sha256": "6" * 64},
        authority=seal({
            "transition_intent": {"seal_sha256": "4" * 64},
            "telegram_delivery_receipt": {"hmac_sha256": "5" * 64},
        }),
        authority_verifier=binding,
        provenance=provenance,
        provenance_receipt=provenance.receipt,
        resource_guard=guard,
        resource_baseline=copy.deepcopy(guard.baseline),
    )
    with pytest.raises(driver.DriverError, match="binding changed"):
        driver._terminal_currentness(
            prepared, initial_authority_binding=initial
        )
    assert provenance.final_checks == 1
    assert guard.checks == 1


def test_git_provenance_requires_clean_exact_pushed_head() -> None:
    head = "a" * 40
    remote_queries = 0

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal remote_queries
        args = list(argv[3:])
        stdout = ""
        returncode = 0
        if args == ["rev-parse", "HEAD"]:
            stdout = head + "\n"
        elif args == ["symbolic-ref", "--short", "HEAD"]:
            stdout = driver.OFFICIAL_BRANCH + "\n"
        elif args == [
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
        ]:
            stdout = f"{driver.OFFICIAL_REMOTE}/{driver.OFFICIAL_BRANCH}\n"
        elif args[:2] == ["ls-files", "--error-unmatch"]:
            stdout = "\n".join(args[2:]) + "\n"
        elif args[:2] == ["diff", "--quiet"]:
            pass
        elif args == ["status", "--porcelain=v1", "--untracked-files=all"]:
            pass
        elif args == ["remote", "get-url", "origin"]:
            stdout = "git@github.com:joshuahickscorp/hawking.git\n"
        elif args == [
            "ls-remote", "--heads", "--exit-code", "origin", driver.OFFICIAL_BRANCH,
        ]:
            remote_queries += 1
            stdout = f"{head}\trefs/heads/{driver.OFFICIAL_BRANCH}\n"
        else:
            returncode = 99
        return subprocess.CompletedProcess(argv, returncode, stdout, "unexpected")

    verifier = driver.ProductionGitProvenance(
        driver.REPO_ROOT,
        runner=runner,
        required_paths=("tools/condense/glm52_xet_live_driver.py",),
    )
    receipt = verifier.preflight()
    assert receipt["status"] == "PASS_CLEAN_HEAD_PUSHED_EXACTLY"
    verifier.assert_current(receipt)
    verifier.assert_final_current(receipt)
    assert remote_queries == 2


def test_git_final_currentness_rejects_upstream_advance() -> None:
    head = "a" * 40
    remote_head = head

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv[3:])
        stdout = ""
        returncode = 0
        if args == ["rev-parse", "HEAD"]:
            stdout = head + "\n"
        elif args == ["symbolic-ref", "--short", "HEAD"]:
            stdout = driver.OFFICIAL_BRANCH + "\n"
        elif args == [
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
        ]:
            stdout = f"{driver.OFFICIAL_REMOTE}/{driver.OFFICIAL_BRANCH}\n"
        elif args[:2] == ["ls-files", "--error-unmatch"]:
            stdout = "\n".join(args[2:]) + "\n"
        elif args[:2] == ["diff", "--quiet"]:
            pass
        elif args == ["status", "--porcelain=v1", "--untracked-files=all"]:
            pass
        elif args == ["remote", "get-url", "origin"]:
            stdout = "git@github.com:joshuahickscorp/hawking.git\n"
        elif args == [
            "ls-remote", "--heads", "--exit-code", "origin", driver.OFFICIAL_BRANCH,
        ]:
            stdout = f"{remote_head}\trefs/heads/{driver.OFFICIAL_BRANCH}\n"
        else:
            returncode = 99
        return subprocess.CompletedProcess(argv, returncode, stdout, "unexpected")

    verifier = driver.ProductionGitProvenance(
        driver.REPO_ROOT,
        runner=runner,
        required_paths=("tools/condense/glm52_xet_live_driver.py",),
    )
    receipt = verifier.preflight()
    remote_head = "b" * 40
    with pytest.raises(driver.DriverError, match="not pushed exactly"):
        verifier.assert_final_current(receipt)


def test_provenance_covers_every_planner_input_and_runtime_lock() -> None:
    required = set(driver.PROVENANCE_PATHS)
    assert {name for name, _schema, _statuses in autotune.INPUT_CONTRACTS} <= required
    assert "GLM52_XET_AUTOTUNE_PLAN.json" in required
    assert "tools/condense/requirements-glm52.txt" in required


def _production_writer(tmp_path: pathlib.Path) -> driver.ProductionArtifactWriter:
    artifact_root = tmp_path / "artifacts"
    controller_root = tmp_path / "controller"
    artifact_root.mkdir()
    controller_root.mkdir()
    return driver.ProductionArtifactWriter(artifact_root, controller_root)


def _test_receipt() -> dict[str, Any]:
    return seal({"schema": "test.driver_receipt.v1", "status": "PASS"})


def test_production_writer_accepts_only_canonical_lowercase_json_and_is_one_use(
    tmp_path: pathlib.Path,
) -> None:
    writer = _production_writer(tmp_path)
    receipt = _test_receipt()
    try:
        writer.begin({"authority": "test"})
        writer.write("RESULT.json", {"schema": "test.result.v1", "status": "PASS"})
        with pytest.raises(driver.DriverError, match="unsafe or duplicated"):
            writer.write("RESULT.JSON", {"status": "PASS"})
        with pytest.raises(driver.DriverError, match="unsafe or duplicated"):
            writer.write("../RESULT.json", {"status": "PASS"})
        writer.write(driver.DRIVER_RECEIPT_NAME, receipt)
        writer.finish(receipt)
        marker = json.loads(writer.marker_path.read_text(encoding="utf-8"))
        assert marker["status"] == "PASS_COMPLETE"
        assert marker["written_artifacts"] == [
            "RESULT.json", driver.DRIVER_RECEIPT_NAME,
        ]
        assert [row["name"] for row in marker["artifact_inventory"]] == \
            marker["written_artifacts"]
        with pytest.raises(driver.DriverError, match="not active"):
            writer.write("LATE.json", {"status": "PASS"})
        with pytest.raises(driver.DriverError, match="more than once"):
            writer.finish(receipt)
        with pytest.raises(driver.DriverError, match="fresh one-use"):
            writer.begin({"authority": "again"})
    finally:
        writer.close()


@pytest.mark.parametrize("mutation", ["delete", "replace", "hardlink"])
def test_terminal_writer_revalidates_every_artifact_before_pass(
    mutation: str,
    tmp_path: pathlib.Path,
) -> None:
    writer = _production_writer(tmp_path)
    receipt = _test_receipt()
    target = writer.artifact_root / "RESULT.json"
    try:
        writer.begin({"authority": "test"})
        writer.write("RESULT.json", {"schema": "test.result.v1", "status": "PASS"})
        writer.write(driver.DRIVER_RECEIPT_NAME, receipt)
        if mutation == "delete":
            target.unlink()
        elif mutation == "replace":
            target.unlink()
            target.write_text("foreign", encoding="utf-8")
        else:
            os.link(target, writer.artifact_root / "FOREIGN.json")
        with pytest.raises((driver.DriverError, OSError)):
            writer.finish(receipt)
        marker = json.loads(writer.marker_path.read_text(encoding="utf-8"))
        assert marker["status"] == \
            "IN_PROGRESS_RESTART_REQUIRES_EXPLICIT_RECONCILIATION"
        if mutation == "replace":
            assert target.read_text(encoding="utf-8") == "foreign"
        with pytest.raises(driver.DriverError, match="more than once"):
            writer.finish(receipt)
    finally:
        writer.close()


def test_anchored_writer_refuses_replaced_artifact_directory(
    tmp_path: pathlib.Path,
) -> None:
    writer = _production_writer(tmp_path)
    moved = tmp_path / "artifacts-original"
    try:
        writer.begin({"authority": "test"})
        writer.artifact_root.rename(moved)
        writer.artifact_root.mkdir()
        with pytest.raises(driver.DriverError, match="directory identity changed"):
            writer.write("RESULT.json", {"status": "PASS"})
        assert not (writer.artifact_root / "RESULT.json").exists()
    finally:
        writer.close()


def test_exclusive_writer_never_deletes_a_racing_foreign_destination(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "RESULT.json"

    def racing_link(
        _source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        del src_dir_fd
        assert follow_symlinks is False
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, b"foreign")
        finally:
            os.close(descriptor)
        raise FileExistsError(target)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(driver.DriverError, match="refusing to overwrite"):
        driver.ProductionArtifactWriter._exclusive_json(
            destination, {"schema": "test", "status": "PASS"}
        )
    assert destination.read_text(encoding="utf-8") == "foreign"


def test_post_link_replacement_plus_directory_fsync_failure_preserves_foreign_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "RESULT.json"
    real_fsync = os.fsync
    injected = False

    def failing_directory_fsync(descriptor: int) -> None:
        nonlocal injected
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not injected:
            injected = True
            destination.unlink()
            destination.write_text("foreign-after-link", encoding="utf-8")
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", failing_directory_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        driver.ProductionArtifactWriter._exclusive_json(
            destination, {"schema": "test", "status": "PASS"}
        )
    assert injected is True
    assert destination.read_text(encoding="utf-8") == "foreign-after-link"


def test_replaced_marker_is_never_overwritten_or_unlinked_at_finish(
    tmp_path: pathlib.Path,
) -> None:
    writer = _production_writer(tmp_path)
    receipt = _test_receipt()
    try:
        writer.begin({"authority": "test"})
        writer.write(driver.DRIVER_RECEIPT_NAME, receipt)
        writer.marker_path.unlink()
        writer.marker_path.write_text("foreign-marker", encoding="utf-8")
        with pytest.raises(driver.DriverError):
            writer.finish(receipt)
        assert writer.marker_path.read_text(encoding="utf-8") == "foreign-marker"
    finally:
        writer.close()
