"""The one remaining self-hosting gap: HCLI can search, read, write and test its
own code but has no way to land a commit. `git.checkout-safe` and
`git.revert-safe` correctly refuse forever -- that refusal is governance, not a
bug, and it stays. This module adds a narrower, honest capability instead of
giving the resident arbitrary git authority: a governed path to ONE commit,
built from three roles that do not trust each other.

    RESIDENT              calls the `git.land.propose` typed tool (see
                           tool_registry.py). It can construct a
                           ``LandingProposal`` out of typed arguments; it has
                           no other way to reach git.

    IntegrationVerifier    deterministic admissibility, no model in the loop.
                           Every check re-derives its own answer from the
                           working tree or a subprocess it runs itself. A
                           proposal cannot self-certify because
                           ``LandingProposal`` has no "tests_passed" field for
                           a lie to live in -- the verifier always re-runs the
                           declared test command against the tree state that
                           exists right now.

    LandingService         the only code in this file (and, by construction,
                           the only code anywhere in HCLI) that shells out to
                           `git add` / `git commit`. It re-verifies every
                           proposal itself before touching git; it never
                           trusts a report handed to it from outside.

Fail closed. Every refusal names the condition that failed (one of
``REFUSAL_REASONS``) so a caller that distrusts the proposer can audit exactly
why a commit did not happen. Nothing here ever calls `git push` -- landing
means a local commit, push stays human.
"""
from __future__ import annotations

import dataclasses
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Paths a proposal can never declare, regardless of its own allowlist. This is
# what stops the governance from amending itself: a resident that could get
# its own verifier or landing-service source into the allowlist could land a
# change that weakens the very check meant to constrain it.
_ALWAYS_REFUSED_PREFIXES: Tuple[str, ...] = (
    ".git/",
    # Every file that can WEAKEN THE CHECK, not just the file that performs it.
    # Listing only landing.py was a real, reproduced bypass: a verifier built a
    # scratch repo, proposed `allowed_paths=["hcli/tool_registry.py"]` with a
    # passing test command, and it LANDED. tool_registry.py defines every tool's
    # mutation class, the default permission set, the unconditional
    # git.checkout-safe/git.revert-safe refusals, and the git.land.propose
    # handler itself -- so a resident could land a brand-new unguarded tool that
    # shells straight to `git commit`, and every later proposal would then be
    # governed by rules the resident had just written for itself.
    "hcli/landing.py",
    "hcli/tool_registry.py",
    # command_is_admissible is what stops `["true"]` from passing as a test.
    # A resident that can edit it can make tautologies admissible again and
    # then land anything.
    "hcli/verifier_pipeline.py",
    # _run_tool is the only route from a WorkUnit to the permission contract.
    # Edit it and the contract stops being consulted at all.
    "hcli/executors.py",
    # The sovereign verifier envelope. The resident may propose implementation
    # that makes these pass; it may not edit the gate until it passes. A gate
    # believed scientifically wrong is SUPERSEDED through protected review with
    # a negative control, never quietly relaxed.
    "hcli/test_goal_verifier_synthesis.py",
    "hcli/test_hcli_overhead.py",
    "hcli/test_self_mutation_e2e.py",
    "hcli/test_context_compiler_runtime.py",
    "hcli/test_qwen38_prefill_pipeline.py",
    "hcli/test_long_context_runtime.py",
    "hcli/test_deltanet_state_checkpoint.py",
    "hcli/test_autonomous_frontier_metabolism.py",
    "hcli/test_capability_callsite_reachability.py",
    "hcli/test_resident_protected_performance.py",
    "hcli/test_resident_successor_handoff.py",
    "hcli/test_negative_science_runtime.py",
    "hcli/test_resident_watch_control_plane.py",
    "tools/odyssey/test_modellake_retained_bytes.py",
    "tools/odyssey/test_odyssey_streaming_runtime.py",
    "receipts/sovereign/VERIFIER_MANIFEST.json",
)

REFUSAL_REASONS = frozenset({
    "NOT_A_GIT_REPO",
    "BRANCH_MISMATCH",
    "EMPTY_ALLOWLIST",
    "PATH_OUTSIDE_REPO",
    "PATH_TOUCHES_GOVERNANCE_SOURCE",
    "DIRTY_OUTSIDE_ALLOWLIST",
    "EMPTY_DIFF",
    "TEST_COMMAND_REQUIRED",
    "TEST_COMMAND_INADMISSIBLE",
    "TEST_COMMAND_UNRUNNABLE",
    "TESTS_FAILED",
    "TESTS_TIMEOUT",
    "TAMPERED_DURING_VERIFICATION",
    "MESSAGE_REQUIRED",
    "GIT_ADD_FAILED",
    "GIT_COMMIT_FAILED",
})

# Recorded in every commit this service lands. Deliberately not the GitHub
# `Co-Authored-By:` trailer format -- that names a *human* contributor, and no
# human wrote this diff. This just records, honestly, what did.
AUTHORSHIP_TRAILER = "Landed-By: hcli-autonomous-landing-service"


@dataclasses.dataclass(frozen=True)
class LandingProposal:
    """What a resident may assert. There is deliberately no ``tests_passed``
    or ``verified`` field: the schema itself gives a self-certification
    nowhere to live. Everything here is re-checked, never taken on faith."""

    repo_root: Path
    branch: str
    allowed_paths: Tuple[str, ...]
    test_command: Tuple[str, ...]
    message: str
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(
            self, "allowed_paths",
            tuple(dict.fromkeys(str(p) for p in (self.allowed_paths or ()))),
        )
        object.__setattr__(self, "test_command", tuple(str(a) for a in (self.test_command or ())))


@dataclasses.dataclass(frozen=True)
class VerificationReport:
    admissible: bool
    reason: Optional[str]
    detail: str
    changed_paths: Tuple[str, ...]
    test_returncode: Optional[int]
    test_stdout_tail: str
    test_stderr_tail: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LandingResult:
    landed: bool
    commit_sha: Optional[str]
    reason: Optional[str]
    detail: str
    report: VerificationReport


class IntegrationVerifier:
    """Deterministic admissibility. Nothing here is a model call, and nothing
    here trusts the proposal's own account of itself."""

    def __init__(self, git_argv: Sequence[str] = ("git",)) -> None:
        self._git = tuple(git_argv)

    def check(self, proposal: LandingProposal) -> VerificationReport:
        repo_root = proposal.repo_root

        # 1. must actually be a git repository rooted exactly here.
        probe = self._run(repo_root, "rev-parse", "--show-toplevel")
        if probe.returncode != 0:
            return self._refuse("NOT_A_GIT_REPO", probe.stderr.strip())
        try:
            toplevel = Path(probe.stdout.strip()).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return self._refuse("NOT_A_GIT_REPO", str(exc))
        if toplevel != repo_root:
            return self._refuse("NOT_A_GIT_REPO", f"toplevel {toplevel} != declared repo_root {repo_root}")

        # 2. the branch must be what the proposal claims -- not "some branch".
        branch_probe = self._run(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        if branch_probe.returncode != 0:
            return self._refuse("BRANCH_MISMATCH", branch_probe.stderr.strip())
        current_branch = branch_probe.stdout.strip()
        if current_branch != proposal.branch:
            return self._refuse(
                "BRANCH_MISMATCH",
                f"proposal claims {proposal.branch!r}, tree is on {current_branch!r}",
            )

        # 3. the allowlist itself must be sane before it is trusted for anything.
        if not proposal.allowed_paths:
            return self._refuse("EMPTY_ALLOWLIST", "allowed_paths must name at least one path")
        normalized_allowlist = set()
        for raw in proposal.allowed_paths:
            normalized = self._normalize(repo_root, raw)
            if normalized is None:
                return self._refuse("PATH_OUTSIDE_REPO", raw)
            if self._touches_governance(normalized):
                return self._refuse("PATH_TOUCHES_GOVERNANCE_SOURCE", normalized)
            normalized_allowlist.add(normalized)

        # 4. the working tree must be clean apart from the declared paths.
        status = self._run(repo_root, "status", "--porcelain")
        if status.returncode != 0:
            return self._refuse("DIRTY_OUTSIDE_ALLOWLIST", status.stderr.strip())
        changed = self._parse_status(status.stdout)
        outside = sorted(set(changed) - normalized_allowlist)
        if outside:
            return self._refuse("DIRTY_OUTSIDE_ALLOWLIST", ", ".join(outside[:10]))

        # 5. the diff must be non-empty -- a no-op proposal is not a landing.
        if not changed:
            return self._refuse("EMPTY_DIFF", "working tree has no changes to land")

        # 6. the test command must actually run, on THIS tree state, and pass.
        # No trust in a caller-supplied verdict is possible here: the schema
        # a proposal is built from (see LandingProposal) has no field to put
        # one in, so there is nothing to trust or distrust -- this is the
        # only source of truth about whether tests passed.
        if not proposal.test_command:
            return self._refuse("TEST_COMMAND_REQUIRED", "")
        # A verifier that RUNS the tests is only honest if the tests can fail.
        # `["true"]`, `["sh","-c","exit 0"]` and
        # `["python3","-c","raise SystemExit(0)"]` were all admissible here:
        # the command ran, exited 0, and landed the change without checking
        # anything. That is self-certification smuggled in through the command
        # instead of through a field. This repo already owns the detector for
        # exactly that shape -- tautologies, `|| true` success-laundering, and
        # first tokens outside the allowlist -- so use it rather than growing a
        # second opinion that can disagree with the first.
        try:
            from .verifier_pipeline import command_is_admissible

            admissible, why = command_is_admissible(shlex.join(proposal.test_command))
        except Exception:
            # FAIL CLOSED. A check that cannot be performed is not a pass, and
            # an attacker who can break the detector must not thereby be
            # granted the thing the detector was guarding.
            admissible, why = False, "VACUOUS_COMMAND"
        if not admissible:
            return self._refuse("TEST_COMMAND_INADMISSIBLE", why)
        try:
            test_proc = subprocess.run(
                list(proposal.test_command),
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=proposal.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._refuse("TESTS_TIMEOUT", str(exc))
        except OSError as exc:
            # A test command that cannot even be executed used to raise
            # FileNotFoundError straight out of check(), so the caller saw a
            # crash rather than a refusal -- failing OPEN by exception.
            return self._refuse("TEST_COMMAND_UNRUNNABLE", str(exc))
        if test_proc.returncode != 0:
            return VerificationReport(
                admissible=False, reason="TESTS_FAILED",
                detail=f"exit {test_proc.returncode}",
                changed_paths=tuple(sorted(changed)),
                test_returncode=test_proc.returncode,
                test_stdout_tail=test_proc.stdout[-2000:],
                test_stderr_tail=test_proc.stderr[-2000:],
            )

        # 7. there must be something to say about the change.
        if not proposal.message.strip():
            return self._refuse("MESSAGE_REQUIRED", "")

        return VerificationReport(
            admissible=True, reason=None, detail="",
            changed_paths=tuple(sorted(changed)),
            test_returncode=test_proc.returncode,
            test_stdout_tail=test_proc.stdout[-2000:],
            test_stderr_tail=test_proc.stderr[-2000:],
        )

    def _run(self, repo_root: Path, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self._git, "-C", str(repo_root), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )

    @staticmethod
    def _refuse(reason: str, detail: str) -> VerificationReport:
        return VerificationReport(
            admissible=False, reason=reason, detail=detail,
            changed_paths=(), test_returncode=None, test_stdout_tail="", test_stderr_tail="",
        )

    @staticmethod
    def _normalize(repo_root: Path, raw: str) -> Optional[str]:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            candidate = (repo_root / text).resolve()
            return candidate.relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            return None

    @staticmethod
    def _touches_governance(normalized_path: str) -> bool:
        for prefix in _ALWAYS_REFUSED_PREFIXES:
            stripped = prefix.rstrip("/")
            if normalized_path == stripped or normalized_path.startswith(prefix):
                return True
        return False

    @staticmethod
    def _parse_status(porcelain: str) -> List[str]:
        # ponytail: plain `--porcelain` (not `-z`) is quoted/escaped for exotic
        # filenames (embedded newlines, a literal " -> "). Every changed path
        # in this repo is plain ASCII in practice; upgrade to `-z` parsing if
        # that ever stops being true.
        paths: List[str] = []
        for line in porcelain.splitlines():
            if not line.strip():
                continue
            rest = line[3:]
            if " -> " in rest:
                rest = rest.split(" -> ", 1)[1]
            rest = rest.strip()
            if rest.startswith('"') and rest.endswith('"') and len(rest) >= 2:
                rest = rest[1:-1]
            if rest:
                paths.append(rest)
        return paths


class LandingService:
    """The sole writer. This is the only class in HCLI whose methods are
    allowed to run `git add` or `git commit`; nothing else in the process
    may. It never trusts a report it did not produce itself -- `land()`
    always re-verifies, even if a caller already holds a report."""

    def __init__(self, verifier: Optional[IntegrationVerifier] = None, git_argv: Sequence[str] = ("git",)) -> None:
        self._verifier = verifier or IntegrationVerifier(git_argv)
        self._git = tuple(git_argv)

    def _run_status(self, repo_root: Path) -> str:
        out = subprocess.run(
            [*self._git, "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30.0, check=False,
        )
        return out.stdout if out.returncode == 0 else ""

    def land(self, proposal: LandingProposal) -> LandingResult:
        report = self._verifier.check(proposal)
        if not report.admissible:
            return LandingResult(landed=False, commit_sha=None, reason=report.reason, detail=report.detail, report=report)

        repo_root = proposal.repo_root
        add = subprocess.run(
            [*self._git, "-C", str(repo_root), "add", "--", *report.changed_paths],
            capture_output=True, text=True, timeout=30.0, check=False,
        )
        if add.returncode != 0:
            return LandingResult(landed=False, commit_sha=None, reason="GIT_ADD_FAILED", detail=add.stderr.strip(), report=report)

        # RE-VERIFY AFTER THE TEST COMMAND RAN. changed_paths was computed
        # BEFORE verification executed attacker-controlled code. A proposal
        # declaring only feature.txt was landed with hcli/landing.py overwritten
        # to "PWNED_GOVERNANCE", because its test command wrote that file and
        # `git add`-ed it, and the unqualified `git commit` below took the whole
        # index. The path allowlist stopped the file being DECLARED; nothing
        # stopped it being WRITTEN by the command the verifier itself runs.
        after = self._verifier._parse_status(
            self._run_status(repo_root)
        )
        allowed = set(report.changed_paths)
        tampered = sorted(p for p in after if p not in allowed)
        if tampered:
            return LandingResult(
                landed=False, commit_sha=None, reason="TAMPERED_DURING_VERIFICATION",
                detail="paths changed while the test command ran: " + ", ".join(tampered[:5]),
                report=report,
            )

        message = f"{proposal.message.strip()}\n\n{AUTHORSHIP_TRAILER}"
        # PATHSPEC-LIMITED COMMIT. `git commit -m` without one commits the whole
        # index, so anything staged behind the service's back rides along. This
        # is the hard barrier; the tamper check above is the detector.
        commit = subprocess.run(
            [*self._git, "-C", str(repo_root), "commit", "-m", message,
             "--", *report.changed_paths],
            capture_output=True, text=True, timeout=30.0, check=False,
        )
        if commit.returncode != 0:
            return LandingResult(landed=False, commit_sha=None, reason="GIT_COMMIT_FAILED", detail=commit.stderr.strip(), report=report)

        sha_probe = subprocess.run(
            [*self._git, "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10.0, check=False,
        )
        sha = sha_probe.stdout.strip() if sha_probe.returncode == 0 else None
        return LandingResult(landed=True, commit_sha=sha, reason=None, detail="", report=report)


def propose_landing(
    repo_root: Any,
    *,
    branch: Any,
    allowed_paths: Any,
    test_command: Any,
    message: Any,
    timeout_s: Any = 300.0,
) -> Dict[str, Any]:
    """The one entry point a typed tool may call (see `git.land.propose` in
    tool_registry.py). Builds the proposal the resident described, then hands
    it straight to the sole writer -- which re-verifies it before it can
    touch git. Never raises for a governance refusal; the result says why."""
    proposal = LandingProposal(
        repo_root=Path(str(repo_root)),
        branch=str(branch or ""),
        allowed_paths=tuple(allowed_paths or ()),
        test_command=tuple(test_command or ()),
        message=str(message or ""),
        timeout_s=float(timeout_s or 300.0),
    )
    result = LandingService().land(proposal)
    return {
        "schema": "hcli.landing.result.v1",
        "landed": result.landed,
        "commit_sha": result.commit_sha,
        "reason": result.reason,
        "detail": result.detail,
        "changed_paths": list(result.report.changed_paths),
    }


__all__ = [
    "AUTHORSHIP_TRAILER",
    "REFUSAL_REASONS",
    "IntegrationVerifier",
    "LandingProposal",
    "LandingResult",
    "LandingService",
    "VerificationReport",
    "propose_landing",
]
