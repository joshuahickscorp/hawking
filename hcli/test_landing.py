"""The one remaining self-hosting gap: HCLI could read and test its own code
but had no way to land a commit. `git.checkout-safe` / `git.revert-safe`
correctly refuse forever; this is not about weakening that. It is a narrower,
separate capability: one governed path from a resident's typed tool call to
ONE local commit, split across three roles that do not trust each other
(RESIDENT proposes, IntegrationVerifier decides admissibility deterministically
by re-deriving every answer itself, LandingService is the only code that ever
shells out to `git commit`).

These tests drive the REAL `hcli.landing` module and the REAL `ToolRegistry`
against a scratch git repository built fresh under `tmp_path` for every test
(never the Hawking repo itself). They cover a clean accepted landing, a
refusal for each named admissibility condition, and proof that the resident's
other tool surfaces (`shell.exec`, `shell.readonly`) cannot reach a raw git
commit either.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from hcli._test_git import scratch_repo

from hcli.landing import LandingService
from hcli.tool_registry import DESTRUCTIVE, READ_ONLY, default_tool_registry

PASS_CMD = [sys.executable, "-c", "print('ok')"]
FAIL_CMD = [sys.executable, "-c", "import sys; sys.exit(1)"]


def _repo(tmp_path: Path) -> Path:
    """A scratch git repository, never the Hawking repo, fresh per test."""
    return scratch_repo(
        tmp_path / "repo",
        email="hcli-landing-test@example.com",
        name="hcli-landing-test",
        filename="README.md",
        body="scratch repo for hcli.landing tests\n",
    )


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()


def _registry(repo: Path):
    return default_tool_registry(repo, repo_root=repo)


def _propose(repo: Path, **overrides):
    args = {
        "branch": "main",
        "allowed_paths": ["feature.txt"],
        "test_command": PASS_CMD,
        "message": "feat(hcli): add the feature file",
    }
    args.update(overrides)
    return _registry(repo).invoke("git.land.propose", args)


# --- a clean, accepted landing ------------------------------------------------


def test_clean_proposal_lands_a_real_commit(tmp_path):
    repo = _repo(tmp_path)
    before = _head(repo)
    (repo / "feature.txt").write_text("new capability\n", encoding="utf-8")

    result = _propose(repo)

    assert result.ok is True, result.to_dict()
    value = result.value
    assert value["landed"] is True, value
    assert value["commit_sha"] and value["commit_sha"] != before
    assert value["changed_paths"] == ["feature.txt"]
    assert _head(repo) == value["commit_sha"], "a real commit must actually move HEAD"

    body = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    assert "feat(hcli): add the feature file" in body
    assert "Landed-By: hcli-autonomous-landing-service" in body
    assert "Co-Authored-By" not in body

    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    assert status.strip() == "", "the working tree must be clean after a real landing"


# --- one refusal per named admissibility condition ---------------------------


def test_refusal_not_a_git_repo(tmp_path):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    result = _registry(plain).invoke("git.land.propose", {
        "branch": "main", "allowed_paths": ["x.txt"], "test_command": PASS_CMD, "message": "m",
    })
    assert result.ok is True  # the TOOL call succeeded; the LANDING was refused
    assert result.value["landed"] is False
    assert result.value["reason"] == "NOT_A_GIT_REPO"


def test_refusal_branch_mismatch(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, branch="not-main")

    assert result.value["reason"] == "BRANCH_MISMATCH"
    assert result.value["landed"] is False
    assert _head(repo) == before


def test_refusal_empty_allowlist(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, allowed_paths=[])

    assert result.value["reason"] == "EMPTY_ALLOWLIST"
    assert _head(repo) == before


def test_refusal_path_outside_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, allowed_paths=["../escape.txt"])

    assert result.value["reason"] == "PATH_OUTSIDE_REPO"
    assert _head(repo) == before


def test_refusal_path_touches_governance_source(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, allowed_paths=["hcli/landing.py"])
    assert result.value["reason"] == "PATH_TOUCHES_GOVERNANCE_SOURCE"
    assert _head(repo) == before

    result = _propose(repo, allowed_paths=[".git/config"])
    assert result.value["reason"] == "PATH_TOUCHES_GOVERNANCE_SOURCE"
    assert _head(repo) == before


def test_refusal_dirty_outside_allowlist(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    (repo / "README.md").write_text("modified outside the allowlist\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, allowed_paths=["feature.txt"])

    assert result.value["reason"] == "DIRTY_OUTSIDE_ALLOWLIST"
    assert "README.md" in result.value["detail"]
    assert _head(repo) == before


def test_refusal_empty_diff(tmp_path):
    repo = _repo(tmp_path)  # nothing at all has changed
    before = _head(repo)

    result = _propose(repo, allowed_paths=["feature.txt"])

    assert result.value["reason"] == "EMPTY_DIFF"
    assert _head(repo) == before


def test_refusal_test_command_required(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, test_command=[])

    assert result.value["reason"] == "TEST_COMMAND_REQUIRED"
    assert _head(repo) == before


def test_refusal_tests_failed(tmp_path):
    """No self-certification: the proposal never claims tests passed, so the
    only way this can be admissible is for the verifier to have run
    FAIL_CMD itself and observed a non-zero exit."""
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, test_command=FAIL_CMD)

    assert result.value["reason"] == "TESTS_FAILED"
    assert _head(repo) == before


def test_refusal_message_required(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _propose(repo, message="   ")

    assert result.value["reason"] == "MESSAGE_REQUIRED"
    assert _head(repo) == before


# --- no self-certification is even expressible --------------------------------


def test_proposal_schema_has_no_field_to_smuggle_a_self_certification(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")

    result = _registry(repo).invoke("git.land.propose", {
        "branch": "main",
        "allowed_paths": ["feature.txt"],
        "test_command": PASS_CMD,
        "message": "m",
        "tests_passed": True,  # not a real field -- additionalProperties: False
    })

    assert result.ok is False
    assert result.failure_class == "INVALID_ARGUMENTS"


def test_landing_service_land_only_takes_a_proposal_never_a_report():
    """There is no back door that hands LandingService a pre-verified report
    and skips the check -- `land` re-derives admissibility itself, every
    time, from nothing but the proposal."""
    sig = inspect.signature(LandingService.land)
    assert list(sig.parameters) == ["self", "proposal"]


# --- the resident's other tool surfaces cannot reach a raw git commit --------


def test_shell_exec_cannot_commit(tmp_path):
    repo = _repo(tmp_path)
    (repo / "feature.txt").write_text("x\n", encoding="utf-8")
    before = _head(repo)

    result = _registry(repo).invoke("shell.exec", {"argv": ["git", "commit", "-am", "sneak it in"]})

    assert result.ok is False
    assert _head(repo) == before, "shell.exec must not be a second path to git commit"


def test_shell_readonly_cannot_run_git_at_all(tmp_path):
    repo = _repo(tmp_path)
    before = _head(repo)

    result = _registry(repo).invoke("shell.readonly", {"command": "git commit -m sneak"})

    assert result.ok is False
    assert _head(repo) == before


def test_only_git_land_propose_can_create_a_commit(tmp_path):
    repo = _repo(tmp_path)
    registry = _registry(repo)
    git_tools = [t for t in registry.discover() if t["name"].startswith("git.")]
    names = {t["name"] for t in git_tools}
    assert "git.land.propose" in names

    for spec in git_tools:
        if spec["name"] == "git.land.propose":
            continue
        # every other git.* tool is either read-only or a hard, unconditional
        # refusal -- neither can write history on its own.
        assert spec["mutation"] in (READ_ONLY, DESTRUCTIVE), spec

    before = _head(repo)
    for name in ("git.checkout-safe", "git.revert-safe", "git.checkout/revert-safe"):
        # DESTRUCTIVE is not in the default permission set at all, so these
        # are blocked before their (always-refusing) handler even runs.
        result = registry.invoke(name, {})
        assert result.ok is False
        assert result.failure_class == "PERMISSION_DENIED"
    assert _head(repo) == before


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in sorted(globals().items()):
            if not name.startswith("test_"):
                continue
            if fn.__code__.co_argcount:
                case_dir = Path(tmp) / name
                case_dir.mkdir()
                fn(case_dir)
            else:
                fn()
            print(f"ok  {name}")
    print("all green")


# --- a verifier that runs the tests is only honest if the tests can fail -----
# `["true"]`, `["sh","-c","exit 0"]` and `["python3","-c","raise SystemExit(0)"]`
# were each admissible: the command ran, exited 0, and landed the change without
# checking anything. That is self-certification smuggled through the command
# rather than through a field, which the schema was carefully shaped to prevent.


def _sub(tmp_path: Path, name: str) -> Path:
    """_repo() creates <parent>/repo, so the parent must exist first."""
    parent = tmp_path / name
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _with_change(repo: Path):
    (repo / "feature.txt").write_text("a real change\n", encoding="utf-8")


def test_a_test_command_that_cannot_fail_is_refused(tmp_path):
    for cmd in (["true"], ["sh", "-c", "exit 0"],
                [sys.executable, "-c", "raise SystemExit(0)"], [":"]):
        repo = _repo(_sub(tmp_path, cmd[-1][:6].replace("/", "_") or "c"))
        _with_change(repo)
        result = _propose(repo, test_command=cmd)
        payload = result.value if result.ok else result.to_dict()
        assert "TEST_COMMAND_INADMISSIBLE" in str(payload), (cmd, payload)


def test_success_laundering_is_refused(tmp_path):
    """`real_check || true` runs the check and throws the failure away."""
    repo = _repo(_sub(tmp_path, "laundered"))
    _with_change(repo)
    result = _propose(repo, test_command=["sh", "-c", "pytest -q || true"])
    payload = result.value if result.ok else result.to_dict()
    assert "TEST_COMMAND_INADMISSIBLE" in str(payload), payload


def test_an_unrunnable_command_refuses_instead_of_raising(tmp_path):
    """It used to raise FileNotFoundError out of check() - failing OPEN."""
    repo = _repo(_sub(tmp_path, "unrunnable"))
    _with_change(repo)
    result = _propose(repo, test_command=["no_such_binary_xyz"])
    payload = str(result.value if result.ok else result.to_dict())
    assert "TEST_COMMAND_INADMISSIBLE" in payload or "TEST_COMMAND_UNRUNNABLE" in payload


def test_a_real_test_command_is_still_admitted(tmp_path):
    """The guard must not make landing impossible, which is the failure mode on
    the other side of over-refusing."""
    repo = _repo(_sub(tmp_path, "real"))
    _with_change(repo)
    result = _propose(repo, test_command=PASS_CMD)
    assert "TEST_COMMAND_INADMISSIBLE" not in str(
        result.value if result.ok else result.to_dict()
    )


def test_every_file_that_can_weaken_the_check_is_refused(tmp_path):
    """Not just the file that performs the check.

    Listing only landing.py was a real, reproduced bypass: a proposal naming
    hcli/tool_registry.py LANDED, and that file defines every tool's mutation
    class, the default permission set, the git.checkout-safe/revert-safe
    refusals, and the git.land.propose handler itself. A resident could have
    landed a new unguarded tool shelling straight to `git commit`, after which
    every proposal is governed by rules it wrote for itself.
    """
    from hcli.landing import _ALWAYS_REFUSED_PREFIXES, IntegrationVerifier

    for target in (
        "hcli/landing.py",
        "hcli/tool_registry.py",
        "hcli/verifier_pipeline.py",
        "hcli/executors.py",
    ):
        assert target in _ALWAYS_REFUSED_PREFIXES, target
        repo = _repo(_sub(tmp_path, target.replace("/", "_")))
        (repo / "hcli").mkdir(exist_ok=True)
        (repo / target).write_text("weakened\n", encoding="utf-8")
        report = IntegrationVerifier().check(
            _landing_proposal(repo, allowed_paths=(target,))
        )
        assert report.admissible is False, target
        assert report.reason == "PATH_TOUCHES_GOVERNANCE_SOURCE", (target, report.reason)


def _landing_proposal(repo: Path, **overrides):
    from hcli.landing import LandingProposal

    args = {
        "repo_root": repo, "branch": "main", "allowed_paths": ("feature.txt",),
        "test_command": tuple(PASS_CMD), "message": "m",
    }
    args.update(overrides)
    return LandingProposal(**args)


def test_a_test_command_cannot_smuggle_a_governance_file_into_the_commit(tmp_path):
    """The fourth bypass, found by an adversarial lane and reproduced.

    changed_paths is computed BEFORE verification runs the test command, and
    the command is attacker-controlled code. A proposal declaring only
    feature.txt landed a commit containing hcli/landing.py overwritten with
    "PWNED_GOVERNANCE": the command wrote the file and `git add`-ed it, and the
    unqualified `git commit` took the whole index. The allowlist stopped the
    file being DECLARED; nothing stopped it being WRITTEN by the command the
    verifier itself runs.

    KNOWN REMAINING LIMIT, stated rather than hidden: the commit is refused and
    the tamper is named, but the test command still executes and can still
    modify the working tree. Sandboxing the verification step is the only real
    fix for that and it is not attempted here.
    """
    from hcli.landing import LandingProposal, LandingService

    repo = _repo(_sub(tmp_path, "tamper"))
    (repo / "hcli").mkdir(exist_ok=True)
    (repo / "hcli" / "landing.py").write_text("GOVERNANCE = 'strict'\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "governance"], cwd=repo, check=True, capture_output=True)
    (repo / "feature.txt").write_text("an ordinary feature\n", encoding="utf-8")
    before = _head(repo)

    attack = (
        'from pathlib import Path; import subprocess; '
        'Path("hcli/landing.py").write_text("PWNED_GOVERNANCE\\n"); '
        'subprocess.check_call(["git","add","--","hcli/landing.py"]); print("ok")'
    )
    result = LandingService().land(LandingProposal(
        repo_root=repo, branch="main", allowed_paths=("feature.txt",),
        test_command=(sys.executable, "-c", attack), message="feat: add feature",
    ))
    assert result.landed is False, "a smuggled governance edit was committed"
    assert result.reason == "TAMPERED_DURING_VERIFICATION", result.reason

    # The refusal means NO new commit at all, so assert HEAD did not move.
    # Inspecting HEAD's contents would only re-read the fixture's own
    # governance commit and prove nothing.
    assert _head(repo) == before, "a commit was created despite the refusal"


def test_the_commit_is_pathspec_limited_to_declared_paths(tmp_path):
    """Belt to the tamper check's braces: even if something is staged behind
    the service's back, an unqualified `git commit` must not sweep it in."""
    from hcli.landing import LandingService
    import inspect

    source = inspect.getsource(LandingService.land)
    assert '"commit", "-m", message,' in source and '"--", *report.changed_paths' in source, (
        "the commit is no longer pathspec-limited, so the whole index can ride along"
    )
