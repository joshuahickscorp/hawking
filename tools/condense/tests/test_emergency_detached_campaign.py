from __future__ import annotations

import json
import os
import signal
import stat
import sys
from pathlib import Path

import pytest


CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import emergency_detached_campaign as handoff  # noqa: E402


TEST_HEAD = "a" * 40


def _resources(
    *, swap_used: int = 0, swapouts: int = 0, boot: int = 1_700_000_000
) -> dict[str, int]:
    return {
        "available_ram_bytes": 32 * 1024**3,
        "swap_used_bytes": swap_used,
        "swapouts": swapouts,
        "free_disk_bytes": 500 * 1024**3,
        "boot_epoch_seconds": boot,
    }


def _authorization(*, executor_head: str = TEST_HEAD) -> dict:
    return handoff.seal({
        "schema": handoff.AUTHORIZATION_SCHEMA,
        "explicit_cli_flag": True,
        "session": os.fspath(handoff.SESSION),
        "executor_root": os.fspath(handoff.EXECUTOR_ROOT),
        "executor_head": executor_head,
        "scope": "EXACT_KIMI_PHASE2_SOURCE_RELEASE_ONLY",
        "authorized_by_uid": os.getuid(),
        "authorized_at": "2026-01-01T00:00:00Z",
    })


def test_resource_parsers_and_zero_growth_policy() -> None:
    vm = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free: 1000000.
Pages inactive: 4000000.
Pages speculative: 100000.
Swapouts: 7.
"""
    available, swapouts = handoff.parse_vm_stat(vm)
    assert available == 5_100_000 * 4096
    assert swapouts == 7
    assert handoff.parse_swap_used("total = 9.00G used = 1.25G free = 7.75G") == int(1.25 * 1024**3)
    assert handoff.parse_swap_used("used = 325.75M") == 325_75 * 1024**2 // 100
    assert handoff.parse_swap_used("used = 0.0009765625K") == 1
    assert handoff.parse_swap_used("used = 325.76M") == 341_584_118
    assert handoff.parse_swap_used("used = 0.1B") == 1
    with pytest.raises(handoff.HandoffError, match="omitted"):
        handoff.parse_swap_used("used = 1e3M")
    assert handoff.parse_boot_epoch("{ sec = 1700000000, usec = 1 }") == 1_700_000_000
    baseline = _resources(swap_used=1024**3, swapouts=7)
    assert handoff.resource_failures(baseline, baseline=baseline, disk_floor=5 * 1024**3) == []
    grown = {**baseline, "swap_used_bytes": baseline["swap_used_bytes"] + 1, "swapouts": 8}
    assert handoff.resource_failures(grown, baseline=baseline, disk_floor=5 * 1024**3) == [
        "SWAP_USED_GREW", "SWAPOUT_COUNTER_GREW"
    ]


def test_recovery_process_detection_requires_exact_session() -> None:
    session = Path("/private/session with spaces")
    rows = [
        (10, f"python kimi_k26_phase2_recovery.py generate --session {str(session)!r}"),
        (11, "python kimi_k26_phase2_recovery.py verify --session '/private/session with spaces'"),
        (12, "python other.py generate --session '/private/session with spaces'"),
        (13, "python kimi_k26_phase2_recovery.py generate --session /other"),
    ]
    assert handoff.recovery_generate_pids(rows, session) == [10]


def _state() -> dict:
    return handoff.seal({
        "schema": handoff.SCHEMA,
        "generation": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "phase": handoff.PHASE_WAIT_RECOVERY,
        "session": os.fspath(handoff.SESSION),
        "executor_root": os.fspath(handoff.EXECUTOR_ROOT),
        "executor_head": TEST_HEAD,
        "repo_root": os.fspath(handoff.REPO_ROOT),
        "kimi_release_authorization": _authorization(),
        "resource_baseline": _resources(),
        "glm_config": "/tmp/config",
        "glm_authority": "/tmp/authority",
        "glm_scratch_root": "/tmp/scratch",
        "child": None,
        "attempts": {},
        "not_before_epoch": 0,
        "last_resource_sample": _resources(),
        "block": None,
        "artifacts": {},
    })


def test_durable_store_repairs_snapshot_from_atomic_journal(tmp_path: Path) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    first = store.commit(_state(), "BOOTSTRAPPED", {"test": True})
    assert store.load() == first
    store.state_path.write_text("torn", encoding="utf-8")
    os.chmod(store.state_path, 0o600)
    assert store.load() == first
    assert stat.S_IMODE(store.state_path.stat().st_mode) == 0o600


def test_journal_chain_tamper_fails_closed(tmp_path: Path) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    store.commit(_state(), "BOOTSTRAPPED", {})
    path = store.journal / "00000000.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["event"] = "TAMPERED"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(handoff.HandoffError, match="seal mismatch"):
        store.load()


def test_exact_private_crash_temp_does_not_brick_journal(tmp_path: Path) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    crash_temp = store.journal / ".00000000.json.123.456.tmp"
    crash_temp.write_bytes(b"torn")
    os.chmod(crash_temp, 0o600)
    assert store.history() == []
    near_match = store.journal / ".00000000.json.123.456.tmp.foreign"
    near_match.write_bytes(b"foreign")
    os.chmod(near_match, 0o600)
    with pytest.raises(handoff.HandoffError, match="unexpected entries"):
        store.history()


def test_operation_roots_default_to_clean_executor() -> None:
    assert handoff.REPO_ROOT == handoff.EXECUTOR_ROOT
    assert handoff.REPO_ROOT == Path(
        "/Users/scammermike/Downloads/hawking-emergency-executor"
    )
    assert handoff.CONTROLLER_ROOT == handoff.EXECUTOR_ROOT


def test_executor_requires_exact_official_remote_head_and_pinned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    expected = TEST_HEAD
    pins = {"operation.py": "c" * 64}
    monkeypatch.setattr(handoff, "EXPECTED_OPERATION_FILE_SHA256", pins)
    monkeypatch.setattr(
        handoff, "_sha256_regular_file", lambda _root, relative: pins[relative]
    )
    calls: list[tuple[str, ...]] = []

    def fake_git(_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return expected
        if arguments == ("rev-parse", "@{u}"):
            return expected
        if arguments == (
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ):
            return handoff.OFFICIAL_CAMPAIGN_REF
        if arguments == ("remote", "get-url", "origin"):
            return handoff.OFFICIAL_ORIGIN_URL
        if arguments == ("branch", "--show-current"):
            return handoff.EXPECTED_EXECUTOR_BRANCH
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return "\n".join(pins)
        if arguments == (
            "ls-remote", "--exit-code", "origin", handoff.OFFICIAL_CAMPAIGN_REMOTE_REF
        ):
            return f"{expected}\t{handoff.OFFICIAL_CAMPAIGN_REMOTE_REF}"
        raise AssertionError(arguments)

    monkeypatch.setattr(handoff, "_git", fake_git)
    receipt = handoff.verify_release_executor(tmp_path)
    assert receipt["head"] == expected
    assert receipt["remote_head"] == expected
    assert receipt["origin_url"] == handoff.OFFICIAL_ORIGIN_URL
    assert receipt["operation_file_sha256"] == pins
    assert (
        "ls-remote", "--exit-code", "origin", handoff.OFFICIAL_CAMPAIGN_REMOTE_REF
    ) in calls


def test_unfinalized_operation_hash_pins_refuse_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        handoff, "EXPECTED_OPERATION_FILE_SHA256", {"operation.py": "__NOT_PINNED__"}
    )
    monkeypatch.setattr(
        handoff, "_git", lambda *_args: pytest.fail("git must not run without a real pin")
    )
    with pytest.raises(handoff.HandoffError, match="hash pins have not been finalized"):
        handoff.verify_release_executor(tmp_path)


def test_linked_worktree_is_rejected_as_emergency_executor(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("gitdir: /foreign/worktree\n", encoding="utf-8")
    with pytest.raises(handoff.HandoffError, match="not a standalone clone"):
        handoff.verify_release_executor(tmp_path)


def test_remote_head_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    pins = {"operation.py": "c" * 64}
    monkeypatch.setattr(handoff, "EXPECTED_OPERATION_FILE_SHA256", pins)
    monkeypatch.setattr(
        handoff, "_sha256_regular_file", lambda _root, relative: pins[relative]
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments in (("rev-parse", "HEAD"), ("rev-parse", "@{u}")):
            return TEST_HEAD
        if arguments == (
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ):
            return handoff.OFFICIAL_CAMPAIGN_REF
        if arguments == ("remote", "get-url", "origin"):
            return handoff.OFFICIAL_ORIGIN_URL
        if arguments == ("branch", "--show-current"):
            return handoff.EXPECTED_EXECUTOR_BRANCH
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return "operation.py"
        if arguments[:3] == ("ls-remote", "--exit-code", "origin"):
            return f"{'b' * 40}\t{handoff.OFFICIAL_CAMPAIGN_REMOTE_REF}"
        raise AssertionError(arguments)

    monkeypatch.setattr(handoff, "_git", fake_git)
    with pytest.raises(handoff.HandoffError, match="exact official remote HEAD"):
        handoff.verify_release_executor(tmp_path)


def test_origin_url_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    pins = {"operation.py": "c" * 64}
    monkeypatch.setattr(handoff, "EXPECTED_OPERATION_FILE_SHA256", pins)
    monkeypatch.setattr(
        handoff, "_sha256_regular_file", lambda _root, relative: pins[relative]
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments in (("rev-parse", "HEAD"), ("rev-parse", "@{u}")):
            return TEST_HEAD
        if arguments == (
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ):
            return handoff.OFFICIAL_CAMPAIGN_REF
        if arguments == ("remote", "get-url", "origin"):
            return "git@github.com:attacker/hawking.git"
        if arguments == ("branch", "--show-current"):
            return handoff.EXPECTED_EXECUTOR_BRANCH
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return "operation.py"
        if arguments[:3] == ("ls-remote", "--exit-code", "origin"):
            return f"{TEST_HEAD}\t{handoff.OFFICIAL_CAMPAIGN_REMOTE_REF}"
        raise AssertionError(arguments)

    monkeypatch.setattr(handoff, "_git", fake_git)
    with pytest.raises(handoff.HandoffError, match="exact official remote HEAD"):
        handoff.verify_release_executor(tmp_path)


def test_post_bootstrap_executor_verification_needs_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    pins = {"operation.py": "c" * 64}
    monkeypatch.setattr(handoff, "EXPECTED_OPERATION_FILE_SHA256", pins)
    monkeypatch.setattr(
        handoff, "_sha256_regular_file", lambda _root, relative: pins[relative]
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments in (("rev-parse", "HEAD"), ("rev-parse", "@{u}")):
            return TEST_HEAD
        if arguments == (
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ):
            return handoff.OFFICIAL_CAMPAIGN_REF
        if arguments == ("remote", "get-url", "origin"):
            return handoff.OFFICIAL_ORIGIN_URL
        if arguments == ("branch", "--show-current"):
            return handoff.EXPECTED_EXECUTOR_BRANCH
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return "operation.py"
        if arguments and arguments[0] == "ls-remote":
            pytest.fail("runtime verification must not query the network")
        raise AssertionError(arguments)

    monkeypatch.setattr(handoff, "_git", fake_git)
    receipt = handoff.verify_release_executor(tmp_path, require_remote=False)
    assert receipt["head"] == TEST_HEAD
    assert receipt["remote_head"] == TEST_HEAD
    assert receipt["remote_verified_live"] is False


def test_controller_waits_for_existing_generate_without_spawning(tmp_path: Path) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    state = store.commit(_state(), "BOOTSTRAPPED", {})
    spawned = []
    sample = _resources()
    command = (
        "python tools/condense/kimi_k26_phase2_recovery.py generate --session "
        + repr(os.fspath(handoff.SESSION))
    )
    controller = handoff.Controller(
        store,
        resource_sampler=lambda _root: sample,
        process_rows=lambda: [(321, command)],
        popen=lambda *args, **kwargs: spawned.append((args, kwargs)),
    )
    controller.state = state
    assert controller.advance_once() is False
    assert spawned == []
    assert store.load()["phase"] == handoff.PHASE_WAIT_RECOVERY


def _child_state(tmp_path: Path) -> dict:
    state = _state()
    body = {key: value for key, value in state.items() if key != "seal_sha256"}
    body["child"] = {
        "action": handoff.ACTION_RECOVERY_VERIFY,
        "action_id": "00000001-recovery_verify",
        "command_sha256": "f" * 64,
        "pid": 4242,
        "pgid": 4242,
        "process_started": "Tue Jul 21 01:02:03 2026",
        "process_command_sha256": "e" * 64,
        "stdout_path": os.fspath(tmp_path / "child.stdout.json"),
        "stderr_path": os.fspath(tmp_path / "child.stderr.log"),
        "start_resource_sample": _resources(),
        "started_at": "2026-01-01T00:00:00Z",
    }
    return handoff.seal(body)


def test_pid_reuse_is_never_signalled_or_treated_as_controller_child(
    tmp_path: Path,
) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    state = store.commit(_child_state(tmp_path), "BOOTSTRAPPED", {})
    signals: list[tuple[int, int]] = []
    reused = {
        "pid": 4242,
        "pgid": 4242,
        "process_started": "Tue Jul 21 09:09:09 2026",
        "process_command_sha256": "1" * 64,
    }
    controller = handoff.Controller(
        store,
        resource_sampler=lambda _root: _resources(),
        process_identity=lambda _pid: reused,
        process_rows=lambda: pytest.fail("broad process scan must not reconcile a child"),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    )
    controller.state = state
    outcome, document = controller._reconcile_child()
    assert (outcome, document) == ("RETRY", None)
    assert signals == []
    assert store.load()["child"] is None


def test_sampler_failure_stops_exact_restarted_child_and_retains_watchdog_state(
    tmp_path: Path,
) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    state = store.commit(_child_state(tmp_path), "BOOTSTRAPPED", {})
    exact = {
        "pid": 4242,
        "pgid": 4242,
        "process_started": "Tue Jul 21 01:02:03 2026",
        "process_command_sha256": "e" * 64,
    }
    stopped = False

    def identity(_pid: int) -> dict | None:
        return None if stopped else exact

    signals: list[tuple[int, int]] = []

    def killpg(pgid: int, sig: int) -> None:
        nonlocal stopped
        signals.append((pgid, sig))
        stopped = True

    controller = handoff.Controller(
        store,
        resource_sampler=lambda _root: (_ for _ in ()).throw(RuntimeError("sampler down")),
        process_identity=identity,
        process_rows=lambda: pytest.fail("broad process scan must not reconcile a child"),
        sleeper=lambda _seconds: None,
        killpg=killpg,
    )
    controller.state = state
    assert controller._reconcile_child() == ("WAIT", None)
    assert signals == [(4242, signal.SIGTERM)]
    persisted = store.load()
    assert persisted is not None and persisted["child"]["pid"] == 4242


def test_spawn_gap_reconciliation_recovers_exact_live_child_without_duplicate(
    tmp_path: Path,
) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    initial = _state()
    body = {key: value for key, value in initial.items() if key != "seal_sha256"}
    controller = handoff.Controller(store)
    controller.state = handoff.seal(body)
    argv, _cwd, _environment = controller._command(handoff.ACTION_RECOVERY_VERIFY)
    body["child"] = {
        "action": handoff.ACTION_RECOVERY_VERIFY,
        "action_id": "00000001-recovery_verify",
        "command_sha256": handoff.hashlib.sha256(
            b"\0".join(item.encode() for item in argv)
        ).hexdigest(),
        "pid": None,
        "stdout_path": os.fspath(tmp_path / "child.stdout.json"),
        "stderr_path": os.fspath(tmp_path / "child.stderr.log"),
        "start_resource_sample": _resources(),
        "started_at": "2026-01-01T00:00:00Z",
    }
    state = store.commit(body, "CHILD_INTENT", {})
    exact = {
        "pid": 4242,
        "pgid": 4242,
        "process_started": "Tue Jul 21 01:02:03 2026",
        "process_command_sha256": "e" * 64,
    }
    controller = handoff.Controller(
        store,
        resource_sampler=lambda _root: _resources(),
        process_rows=lambda: [(4242, handoff.shlex.join(argv))],
        process_identity=lambda _pid: exact,
        popen=lambda *_args, **_kwargs: pytest.fail("must not spawn a duplicate child"),
    )
    controller.state = state
    assert controller._reconcile_child() == ("WAIT", None)
    persisted = store.load()
    assert persisted is not None
    assert persisted["child"]["pid"] == 4242
    assert persisted["child"]["pgid"] == 4242


def test_campaign_baseline_is_not_reset_between_actions(tmp_path: Path) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    state = store.commit(_state(), "BOOTSTRAPPED", {})
    spawned: list[object] = []
    grown = _resources(swap_used=1, swapouts=1)
    controller = handoff.Controller(
        store,
        resource_sampler=lambda _root: grown,
        popen=lambda *args, **kwargs: spawned.append((args, kwargs)),
    )
    controller.state = state
    assert controller.run_child(handoff.ACTION_RECOVERY_VERIFY) == ("WAIT", None)
    persisted = store.load()
    assert persisted is not None
    assert persisted["resource_baseline"] == _resources()
    assert persisted["last_resource_sample"] == grown
    assert spawned == []


def test_verified_executor_head_must_remain_bootstrap_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    state = store.commit(_state(), "BOOTSTRAPPED", {})
    monkeypatch.setattr(
        handoff,
        "verify_release_executor",
        lambda _root, **_kwargs: {
            "head": "b" * 40,
            "root": os.fspath(handoff.EXECUTOR_ROOT),
        },
    )
    controller = handoff.Controller(store)
    controller.state = state
    assert controller._verify_executor_or_block() is False
    persisted = store.load()
    assert persisted["phase"] == handoff.PHASE_BLOCKED
    assert persisted["block"]["reason"] == "RELEASE_EXECUTOR_HEAD_CHANGED_AFTER_BOOTSTRAP"


def test_replacement_capsule_status_is_accepted_without_terminal_block(
    tmp_path: Path,
) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    initial = _state()
    body = {key: value for key, value in initial.items() if key != "seal_sha256"}
    body["phase"] = handoff.PHASE_VERIFY_RECOVERY
    state = store.commit(body, "BOOTSTRAPPED", {})
    document = handoff.seal({
        "status": "PASS_DETERMINISTIC_SEMANTIC_REPLACEMENT_CAPSULE"
    })
    controller = handoff.Controller(store)
    controller.state = state
    controller.run_child = lambda _action: ("DONE", document)  # type: ignore[method-assign]
    assert controller.advance_once() is True
    persisted = store.load()
    assert persisted["phase"] == handoff.PHASE_RELEASE_AUDIT
    assert persisted["recovery_capsule_mode"] == "DETERMINISTIC_SEMANTIC_REPLACEMENT"


def test_plist_is_caffeinated_and_does_not_touch_overnight_plist() -> None:
    value = handoff._validate_plist(handoff.PLIST_SOURCE)
    assert value["ProgramArguments"][:3] == ["/usr/bin/caffeinate", "-dimsu", "--"]
    assert value["ProgramArguments"][4] == os.fspath(
        handoff.EXECUTOR_ROOT / "tools/condense/emergency_detached_campaign.py"
    )
    assert value["WorkingDirectory"] == os.fspath(handoff.EXECUTOR_ROOT)
    assert handoff.PLIST_SOURCE == (
        Path(handoff.__file__).resolve().parents[2]
        / "deploy/launchd/com.hawking.emergency.detached.campaign.plist"
    )
    assert value["ProcessType"] == "Interactive"
    assert value["StandardOutPath"].startswith(os.fspath(handoff.STATE_ROOT))
    assert "overnight" not in handoff.PLIST_SOURCE.name


def test_bootstrap_requires_explicit_release_authorization_flag() -> None:
    args = handoff.build_parser().parse_args(["bootstrap"])
    with pytest.raises(handoff.HandoffError, match="explicit --authorize"):
        handoff._bootstrap(args)
    authorized = handoff.build_parser().parse_args(
        ["bootstrap", "--authorize-exact-kimi-release"]
    )
    assert authorized.authorize_exact_kimi_release is True


def test_release_authorization_is_private_durable_and_head_bound(
    tmp_path: Path,
) -> None:
    store = handoff.DurableStore(tmp_path / "state")
    store.prepare()
    authorization = handoff._durable_authorization(store, executor_head=TEST_HEAD)
    assert authorization["explicit_cli_flag"] is True
    assert authorization["executor_head"] == TEST_HEAD
    assert authorization["scope"] == "EXACT_KIMI_PHASE2_SOURCE_RELEASE_ONLY"
    assert stat.S_IMODE(store.authorization_path.stat().st_mode) == 0o600
    assert handoff._durable_authorization(store, executor_head=TEST_HEAD) == authorization
    with pytest.raises(handoff.HandoffError, match="changed executor_head"):
        handoff._durable_authorization(store, executor_head="b" * 40)


def test_source_contains_no_delete_implementation() -> None:
    source = Path(handoff.__file__).read_text(encoding="utf-8")
    forbidden = ("shutil.rmtree(", "os.unlink(", "os.remove(", ".unlink(")
    assert all(token not in source for token in forbidden)
