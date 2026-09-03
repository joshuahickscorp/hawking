"""Resident laboratory: isolated worktree, bounded writes, authority lattice.

The super-resident lives here for hours. Provisioning creates (or re-enters)
an isolated git worktree under the canonical repo's own `.worktrees/`, a
writable root whose writes cannot leave that root, and the closed action
lattice that permits reversible local science while refusing canonical merge,
destruction, external action, and authority widening.

This module does not start a model, take a GPU lease, or promote evidence.
Everything it emits is STATIC_ONLY / bench UNKNOWN / gpu_authority false.

    python3 tools/future/sandbox.py --selftest
    python3 tools/future/sandbox.py --build
    python3 -m pytest tools/future/test_sandbox.py -q

Recovered, not forked: hcli.workspace.Workspace (realpath root),
hcli.mutation.rollback_mutation (reversible local edits),
hcli.agentos.resident_gate / autonomy_gate (lifecycle owners; not invoked),
tools.future.resident_optimizer.OptimizerBound (FORBIDDEN/ALLOWED + BoundViolation),
tools.future.mutation_surface.check_disjoint (Codex non-interference),
research/lab/execution_sandbox.py (bible §21 policy; not materialized here),
crates/hide-kernel security_sandbox (OS Seatbelt; orthogonal).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from hcli.persist import atomic_write_json, atomic_write_text
from hcli.workspace import Workspace
from hcli.workunit import (
    DEFAULT_RETRY_BUDGET,
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    WorkUnit,
)
from tools.future._common import git
from tools.future.mutation_surface import check_disjoint, intersects_codex
from tools.future.resident_optimizer import (
    ALLOWED_AUTHORITY,
    BoundViolation,
    ERAS,
    FORBIDDEN_AUTHORITY,
    ODYSSEYS,
    OptimizerBound,
)

RECEIPT = "RESIDENT_SANDBOX.json"
SCHEMA = "hawking.future.sandbox.v1"
STATE_SCHEMA = "hawking.future.sandbox.state.v1"
REGISTRY_SCHEMA = "hawking.future.sandbox.registry.v1"
RECORDED_BY = "tools/future/sandbox.py"
VERSION = 1
STATE_NAME = "SANDBOX.json"
REGISTRY_NAME = "INDEX.json"
DEFAULT_SANDBOX_ID = "resident"
WORKTREES_DIRNAME = ".worktrees"
WRITABLE_DIRNAME = "writable"
CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. The sandbox permits "
    "reversible local science and refuses canonical merge, destruction, "
    "external action, hardware-risk action, verifier modification, and "
    "authority widening. Provisioning does not start a model."
)
SIDECAR_STATUS = "BUILT_NOT_PROMOTED"

# Subdirectories of the bounded writable root. Derived by callers via this tuple.
LAYOUT_DIRS = (
    "build",
    "experiment",
    "receipts",
    "model_candidates",
    "source_index",
)

# Ten autonomous actions the resident may perform inside the sandbox.
PERMITTED_AUTONOMOUS = frozenset(
    {
        "read_only_inspection",
        "local_reversible_experiment",
        "isolated_build",
        "local_test",
        "profiling",
        "simulation",
        "model_candidate_creation",
        "temporary_child_process",
        "detached_workunit",
        "receipt_creation",
    }
)

# Nine higher-authority actions. Each is refused by name, not as a blanket.
REFUSED_HIGHER = frozenset(
    {
        "canonical_merge",
        "destructive_action",
        "external_publication",
        "external_message",
        "paid_action",
        "bounty_submission",
        "hardware_risk_action",
        "verifier_modification",
        "authority_widening",
    }
)

# Optimizer-lattice names that collapse onto a refused higher-authority action.
# Reuse, do not fork, the resident_optimizer vocabulary.
OPTIMIZER_TO_HIGHER: dict[str, str] = {
    "mutate_codex_surface": "canonical_merge",
    "destructive_mutation": "destructive_action",
    "destructive_write": "destructive_action",
    "acquire_gpu_lease": "hardware_risk_action",
    "claim_hardware_measurement": "hardware_risk_action",
    "claim_protected_absolute": "hardware_risk_action",
    "override_bench_state": "hardware_risk_action",
    "modify_verifier": "verifier_modification",
    "weaken_verifier": "verifier_modification",
    "replace_verifier": "verifier_modification",
    "disable_verifier": "verifier_modification",
    "widen_authority": "authority_widening",
    "self_promotion": "authority_widening",
    "promote_self": "authority_widening",
    "promote_candidate": "authority_widening",
    "promote_to_protected_absolute": "authority_widening",
    "promote_diagnostic_relative": "authority_widening",
    "choose_singularity": "authority_widening",
    "select_singularity": "authority_widening",
    "install_singularity": "authority_widening",
}

HIGHER_REASONS: dict[str, str] = {
    "canonical_merge": (
        "canonical merge is refused; the sandbox cannot write the canonical "
        "worktree or the Codex surface"
    ),
    "destructive_action": (
        "destructive action is refused; teardown preserves artifacts and "
        "there is no destroy path"
    ),
    "external_publication": "external publication is refused; no network, no publish",
    "external_message": "external message is refused; no network, no outbound message",
    "paid_action": "paid action is refused; this lane cannot spend",
    "bounty_submission": "bounty submission is refused; this lane cannot submit a bounty",
    "hardware_risk_action": (
        "hardware-risk action is refused; no GPU lease, no model start, no "
        "protected measurement"
    ),
    "verifier_modification": (
        "verifier modification is refused; the proposer is never the admitter"
    ),
    "authority_widening": (
        "authority widening is refused; the bound is frozen at construction"
    ),
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# Declared roots the source index walks when present. Absence is recorded, not
# asserted: this checkout is sparse.
SOURCE_INDEX_ROOTS = (
    "tools/future",
    "hcli",
    "receipts/future",
)

# Integration points for this-wave siblings that must not be imported.
INTEGRATION_POINTS: tuple[dict[str, str], ...] = (
    {
        "module": "tools/future/detached.py",
        "local": "DetachedWorkUnitSpec",
        "swap": "import the landed detached WorkUnit scheduler when that lane commits",
    },
    {
        "module": "tools/future/frontiers.py",
        "local": "frontier_fed on the sidecar receipt",
        "swap": "register the laboratory frontier entry when frontiers.py lands",
    },
    {
        "module": "tools/future/wakeup.py",
        "local": "sleeping physical blockers recorded, never synthesised",
        "swap": "HCLI wakeup of SLEEPING WorkUnits when hardware qualifies",
    },
    {
        "module": "tools/future/super_resident.py",
        "local": "this sandbox is the laboratory the super-resident occupies",
        "swap": "super_resident provisions via tools.future.sandbox.provision",
    },
    {
        "module": "tools/future/protected_window.py",
        "local": "hardware_risk_action refusal",
        "swap": "protected window remains Codex-owned; this lane never takes it",
    },
    {
        "module": "tools/future/resident_api.py",
        "local": "provision / authorize / attempt as the callable surface",
        "swap": "resident_api may wrap provision() once committed",
    },
)


class SandboxEscapeError(PermissionError):
    """A write resolved outside the bounded writable root."""

    def __init__(self, kind: str, path: str, detail: str) -> None:
        self.kind = kind
        self.path = path
        self.detail = detail
        super().__init__(f"{kind} escape refused for {path!r}: {detail}")


class AuthorityRefused(BoundViolation):
    """A named higher-authority (or unknown) action was refused."""

    def __init__(self, authority: str, reason: str) -> None:
        self.authority = authority
        self.reason = reason
        super().__init__(f"{authority}: {reason}")


class CanonicalMutationError(AuthorityRefused):
    """A write targeted the canonical worktree or the Codex surface."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        super().__init__(
            "canonical_merge",
            f"refused write to canonical/Codex path {path!r}: {detail}",
        )


class ProvisionError(RuntimeError):
    """Sandbox provision failed closed."""


@dataclass(frozen=True)
class DetachedWorkUnitSpec:
    """Local interface for a detached WorkUnit. Not tools.future.detached."""

    unit_id: str
    payload: Mapping[str, Any]
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "payload": dict(self.payload),
            "status": self.status,
            "integration_point": "tools/future/detached.py",
            "imported": False,
        }


@dataclass(frozen=True)
class ActionDecision:
    allowed: bool
    action: str
    authority: str
    reason: str
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "authority": self.authority,
            "reason": self.reason,
            "executed": self.executed,
        }


def _contained(path: str, root: str) -> bool:
    path_n = os.path.normpath(path)
    root_n = os.path.normpath(root)
    if path_n == root_n:
        return True
    prefix = root_n if root_n.endswith(os.sep) else root_n + os.sep
    return path_n.startswith(prefix)


def _safe_id(sandbox_id: str) -> str:
    text = str(sandbox_id or "").strip()
    if not _SAFE_ID.match(text):
        raise ProvisionError(
            f"sandbox id {sandbox_id!r} is not a safe path component "
            "(allowed: A-Za-z0-9._-)"
        )
    return text


def _git(canonical: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(canonical), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _require_git_repo(canonical: Path) -> None:
    proc = _git(canonical, "rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise ProvisionError(
            f"{canonical} is not a git work tree; sandbox provision fails closed"
        )


def _identity_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def canonicalize_action(action: str) -> str:
    """Map aliases (optimizer lattice, bible names) onto the sandbox vocabulary."""
    raw = str(action or "").strip()
    if raw in PERMITTED_AUTONOMOUS or raw in REFUSED_HIGHER:
        return raw
    if raw in OPTIMIZER_TO_HIGHER:
        return OPTIMIZER_TO_HIGHER[raw]
    return raw


class SandboxBound:
    """Closed action lattice composed with OptimizerBound.

    Construction reuses OptimizerBound so a bound that grants promotion,
    verifier modification, authority widening, a GPU window, Era VI, or
    Odyssey IV raises BoundViolation before a sandbox exists.
    """

    def __init__(
        self,
        optimizer: OptimizerBound | None = None,
        permitted: Iterable[str] | None = None,
    ) -> None:
        self.optimizer = optimizer if optimizer is not None else OptimizerBound()
        permitted_set = frozenset(permitted) if permitted is not None else PERMITTED_AUTONOMOUS
        unknown = sorted(permitted_set - PERMITTED_AUTONOMOUS)
        if unknown:
            raise BoundViolation(f"unknown autonomous action(s) {unknown}")
        overlap = sorted(permitted_set & REFUSED_HIGHER)
        if overlap:
            raise BoundViolation(f"bound listed refused higher-authority {overlap}")
        forbidden = sorted(self.optimizer.allowed_authority & FORBIDDEN_AUTHORITY)
        if forbidden:
            raise BoundViolation(f"bound listed forbidden authority {forbidden}")
        if self.optimizer.may_promote or self.optimizer.may_modify_verifier or self.optimizer.may_widen_authority:
            raise BoundViolation(
                "a sandbox bound cannot grant promotion, verifier modification, "
                "or authority widening"
            )
        object.__setattr__(self, "permitted", frozenset(permitted_set))
        object.__setattr__(self, "refused", frozenset(REFUSED_HIGHER))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen") and name in {"permitted", "refused", "optimizer"}:
            raise AuthorityRefused(
                "authority_widening",
                f"sandbox bound cannot assign {name!r} after construction",
            )
        object.__setattr__(self, name, value)

    def grant(self, action: str) -> None:
        raise AuthorityRefused(
            "authority_widening",
            f"sandbox cannot grant {action!r}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "permitted": sorted(self.permitted),
            "refused": sorted(self.refused),
            "optimizer": self.optimizer.to_dict(),
            "optimizer_allowed_authority": sorted(ALLOWED_AUTHORITY),
            "optimizer_forbidden_authority": sorted(FORBIDDEN_AUTHORITY),
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "gpu_windows_held": 0,
        }


class BoundedFS:
    """Writable filesystem that refuses escape by construction.

    The only write root is ``root`` (the sandbox writable directory). Absolute
    paths, ``..`` traversal, and symlink hops that leave that root each raise
    SandboxEscapeError naming the kind. Codex / canonical-worktree targets
    raise CanonicalMutationError even if a misconfigured root would have
    contained them.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        canonical_root: str | Path | None = None,
        worktree_root: str | Path | None = None,
    ) -> None:
        self.root = Path(os.path.realpath(str(root)))
        if not self.root.is_dir():
            raise ProvisionError(f"bounded writable root is not a directory: {self.root}")
        self.canonical_root = (
            Path(os.path.realpath(str(canonical_root))) if canonical_root is not None else None
        )
        self.worktree_root = (
            Path(os.path.realpath(str(worktree_root))) if worktree_root is not None else None
        )

    def resolve_writable(self, user_path: str) -> Path:
        if user_path is None or str(user_path) == "":
            raise SandboxEscapeError("invalid", str(user_path), "empty path")
        text = str(user_path)
        if "\x00" in text:
            raise SandboxEscapeError("invalid", text, "NUL byte in path")
        root_at_init = os.path.realpath(str(self.root))
        if os.path.islink(str(self.root)):
            raise SandboxEscapeError(
                "symlink",
                text,
                "writable root was replaced by a symlink",
            )
        root_real = os.path.realpath(str(self.root))
        if not os.path.isdir(root_real):
            raise SandboxEscapeError(
                "invalid",
                text,
                "writable root is no longer a directory",
            )
        if root_real != root_at_init:
            raise SandboxEscapeError(
                "symlink",
                text,
                f"writable root retargeted from {root_at_init} to {root_real}",
            )

        if os.path.isabs(text):
            lexical = os.path.abspath(text)
            if not _contained(lexical, root_real):
                raise SandboxEscapeError(
                    "absolute",
                    text,
                    f"absolute path {lexical} is outside {root_real}",
                )
        else:
            lexical = os.path.abspath(os.path.join(root_real, text))
            if not _contained(lexical, root_real):
                raise SandboxEscapeError(
                    "dotdot",
                    text,
                    f"path {lexical} leaves {root_real} via .. traversal",
                )

        resolved = self._walk_inside(root_real, lexical, text)
        self._refuse_canonical(resolved)
        return Path(resolved)

    def _walk_inside(self, root_real: str, lexical: str, user: str) -> str:
        rel = os.path.relpath(lexical, root_real)
        if rel.startswith(".."):
            kind = "absolute" if os.path.isabs(user) else "dotdot"
            raise SandboxEscapeError(kind, user, f"{lexical} is not under {root_real}")
        if rel == ".":
            return root_real
        current = root_real
        parts = rel.split(os.sep)
        for i, part in enumerate(parts):
            if part in {"", "."}:
                continue
            nxt = os.path.join(current, part)
            if os.path.lexists(nxt) and os.path.islink(nxt):
                target = os.path.realpath(nxt)
                if not _contained(target, root_real):
                    raise SandboxEscapeError(
                        "symlink",
                        user,
                        f"{nxt} -> {target} leaves {root_real}",
                    )
                current = target
                continue
            if os.path.exists(nxt):
                real = os.path.realpath(nxt)
                if not _contained(real, root_real):
                    raise SandboxEscapeError(
                        "symlink",
                        user,
                        f"{nxt} resolved to {real}, outside {root_real}",
                    )
                current = real
                continue
            rest = parts[i:]
            current = os.path.abspath(os.path.join(current, *rest))
            if not _contained(current, root_real):
                raise SandboxEscapeError(
                    "dotdot",
                    user,
                    f"constructed path {current} leaves {root_real}",
                )
            break
        else:
            if not _contained(current, root_real):
                raise SandboxEscapeError(
                    "symlink",
                    user,
                    f"resolved {current} leaves {root_real}",
                )
        return current

    def _refuse_canonical(self, resolved: str) -> None:
        if self.canonical_root is not None:
            canon = str(self.canonical_root)
            work = str(self.worktree_root) if self.worktree_root is not None else ""
            if _contained(resolved, canon) and not (work and _contained(resolved, work)):
                raise CanonicalMutationError(
                    resolved,
                    "target is inside the canonical worktree and outside this sandbox",
                )
        try:
            rel = os.path.relpath(resolved, REPO)
        except ValueError:
            return
        if rel.startswith(".."):
            return
        rel_posix = rel.replace(os.sep, "/")
        if intersects_codex(rel_posix):
            raise CanonicalMutationError(rel_posix, "intersects the Codex mutation surface")

    def write_text(self, user_path: str, content: str) -> Path:
        dest = self.resolve_writable(user_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not _contained(os.path.realpath(str(dest.parent)), os.path.realpath(str(self.root))):
            raise SandboxEscapeError(
                "symlink",
                user_path,
                "parent directory escaped before write",
            )
        atomic_write_text(dest, content)
        return dest

    def read_text(self, user_path: str) -> str:
        dest = self.resolve_writable(user_path)
        return dest.read_text(encoding="utf-8")

    def exists(self, user_path: str) -> bool:
        try:
            dest = self.resolve_writable(user_path)
        except (SandboxEscapeError, CanonicalMutationError):
            return False
        return dest.exists()


def _registry_path(canonical: Path) -> Path:
    return Path(canonical) / WORKTREES_DIRNAME / REGISTRY_NAME


def _load_registry(canonical: Path) -> dict[str, Any]:
    path = _registry_path(canonical)
    if not path.is_file():
        return {"schema": REGISTRY_SCHEMA, "sandboxes": {}}
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {"schema": REGISTRY_SCHEMA, "sandboxes": {}}
    if not isinstance(doc, dict):
        return {"schema": REGISTRY_SCHEMA, "sandboxes": {}}
    boxes = doc.get("sandboxes")
    if not isinstance(boxes, dict):
        doc["sandboxes"] = {}
    return doc


def _store_registry(canonical: Path, sandbox_id: str, relative_worktree: str) -> None:
    doc = _load_registry(canonical)
    boxes = dict(doc.get("sandboxes") or {})
    boxes[sandbox_id] = {
        "id": sandbox_id,
        "relative_worktree": relative_worktree.replace(os.sep, "/"),
    }
    doc["schema"] = REGISTRY_SCHEMA
    doc["sandboxes"] = dict(sorted(boxes.items()))
    dest = _registry_path(canonical)
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, doc)


def _worktree_paths(canonical: Path) -> dict[str, dict[str, str]]:
    proc = _git(canonical, "worktree", "list", "--porcelain")
    rows: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if not line.strip():
            if "worktree" in current:
                rows[os.path.realpath(current["worktree"])] = current
            current = {}
            continue
        if line.startswith("worktree "):
            current = {"worktree": line[len("worktree ") :]}
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD ") :]
        elif line == "detached":
            current["detached"] = "true"
        elif line.startswith("branch "):
            current["branch"] = line[len("branch ") :]
    if "worktree" in current:
        rows[os.path.realpath(current["worktree"])] = current
    return rows


def _add_worktree(canonical: Path, worktree: Path) -> dict[str, Any]:
    worktree.parent.mkdir(parents=True, exist_ok=True)
    if worktree.exists():
        raise ProvisionError(f"worktree path already exists: {worktree}")
    proc = _git(
        canonical,
        "worktree",
        "add",
        "--detach",
        "--no-checkout",
        "--lock",
        "--reason",
        "resident-sandbox preserve-artifacts",
        str(worktree),
        "HEAD",
    )
    if proc.returncode != 0:
        raise ProvisionError(
            f"git worktree add failed: {(proc.stderr or proc.stdout or '').strip()}"
        )
    return {
        "created": True,
        "locked": True,
        "checkout": False,
        "detach": True,
        "stderr": (proc.stderr or "").strip(),
        "stdout": (proc.stdout or "").strip(),
    }


def _init_layout(worktree: Path) -> Path:
    writable = worktree / WRITABLE_DIRNAME
    writable.mkdir(parents=True, exist_ok=True)
    for rel in LAYOUT_DIRS:
        (writable / rel).mkdir(parents=True, exist_ok=True)
    return writable


def build_source_index(canonical: Path) -> dict[str, Any]:
    """Walk declared roots. Record present/missing; do not encode the checkout."""
    entries: list[dict[str, Any]] = []
    for rel in SOURCE_INDEX_ROOTS:
        root = Path(canonical) / rel
        present = root.exists()
        files: list[str] = []
        if present and root.is_dir():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(
                    d for d in dirnames if d not in {".git", "__pycache__", "target"}
                )
                for fn in sorted(filenames):
                    if fn.endswith(".pyc"):
                        continue
                    full = Path(dirpath) / fn
                    files.append(os.path.relpath(full, canonical).replace(os.sep, "/"))
        elif present and root.is_file():
            files.append(rel.replace(os.sep, "/"))
        entries.append(
            {
                "root": rel.replace(os.sep, "/"),
                "present": present,
                "file_count": len(files),
                "files_head": files[:32],
            }
        )
    return {
        "schema": "hawking.future.sandbox.source_index.v1",
        "roots": list(SOURCE_INDEX_ROOTS),
        "entries": entries,
        "copes_with_sparse_checkout": True,
        "present_roots": [e["root"] for e in entries if e["present"]],
        "missing_roots": [e["root"] for e in entries if not e["present"]],
        "file_count_total": sum(e["file_count"] for e in entries),
    }


def _state_identity_body(
    *,
    sandbox_id: str,
    canonical_root: str,
    relative_worktree: str,
    bound: SandboxBound,
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "sandbox_id": sandbox_id,
        "canonical_root": canonical_root,
        "relative_worktree": relative_worktree.replace(os.sep, "/"),
        "writable_relative": f"{WRITABLE_DIRNAME}",
        "layout": list(LAYOUT_DIRS),
        "authority": bound.to_dict(),
        "model_started": False,
        "starts_model": False,
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def _write_state(
    worktree: Path,
    body: Mapping[str, Any],
    *,
    newly_created: bool,
    torn_down: bool,
    path_taken: str,
) -> dict[str, Any]:
    identity = _identity_sha256(body)
    doc = dict(body)
    doc["identity_sha256"] = identity
    doc["newly_created"] = newly_created
    doc["torn_down"] = torn_down
    doc["path_taken"] = path_taken
    atomic_write_json(worktree / STATE_NAME, doc)
    return doc


class ResidentSandbox:
    """Provisioned laboratory. Writes go through BoundedFS. Actions through attempt()."""

    def __init__(
        self,
        *,
        sandbox_id: str,
        canonical_root: Path,
        worktree: Path,
        writable: Path,
        bound: SandboxBound,
        state: Mapping[str, Any],
        newly_created: bool,
        git_meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.sandbox_id = sandbox_id
        self.canonical_root = Path(os.path.realpath(str(canonical_root)))
        self.worktree = Path(os.path.realpath(str(worktree)))
        self.writable_root = Path(os.path.realpath(str(writable)))
        self.bound = bound
        self.state_doc = dict(state)
        self.newly_created = bool(newly_created)
        self.git_meta = dict(git_meta or {})
        self.workspace = Workspace(str(self.writable_root))
        self.fs = BoundedFS(
            self.writable_root,
            canonical_root=self.canonical_root,
            worktree_root=self.worktree,
        )
        self._log: list[dict[str, Any]] = []

    def state(self) -> dict[str, Any]:
        path = self.worktree / STATE_NAME
        if path.is_file():
            self.state_doc = load_json(path)
        return dict(self.state_doc)

    def layout(self) -> dict[str, str]:
        writable = self.writable_root
        rows = {name: str(writable / name) for name in LAYOUT_DIRS}
        rows["writable"] = str(writable)
        rows["worktree"] = str(self.worktree)
        rows["state"] = str(self.worktree / STATE_NAME)
        return rows

    def authorize(self, action: str) -> ActionDecision:
        canonical = canonicalize_action(action)
        if canonical in REFUSED_HIGHER:
            raise AuthorityRefused(canonical, HIGHER_REASONS[canonical])
        if canonical in FORBIDDEN_AUTHORITY:
            mapped = OPTIMIZER_TO_HIGHER.get(canonical, "authority_widening")
            raise AuthorityRefused(
                mapped,
                f"optimizer-forbidden authority {canonical!r} is refused",
            )
        if canonical not in self.bound.permitted:
            raise AuthorityRefused(
                canonical,
                "unknown action; deny-by-default (not on the autonomous allow-list)",
            )
        if self.state().get("torn_down") and canonical not in {
            "read_only_inspection",
            "receipt_creation",
        }:
            raise AuthorityRefused(
                "destructive_action",
                f"sandbox is torn down; {canonical} is frozen (artifacts preserved)",
            )
        return ActionDecision(
            allowed=True,
            action=action,
            authority=canonical,
            reason=f"{canonical} is a permitted autonomous action",
            executed=False,
        )

    def attempt(self, action: str, **kwargs: Any) -> ActionDecision:
        decision = self.authorize(action)
        canonical = decision.authority
        executed = False
        if canonical == "receipt_creation":
            name = str(kwargs.get("name") or "local.json")
            payload = dict(kwargs.get("payload") or {"ok": True})
            payload.setdefault("measurement_state", "STATIC_ONLY")
            payload.setdefault("bench_state", "UNKNOWN")
            payload.setdefault("gpu_authority", False)
            self.fs.write_text(
                f"receipts/{name}",
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
            executed = True
        elif canonical == "local_reversible_experiment":
            name = str(kwargs.get("name") or "note.txt")
            content = str(kwargs.get("content") or "experiment\n")
            self.fs.write_text(f"experiment/{name}", content)
            executed = True
        elif canonical == "isolated_build":
            name = str(kwargs.get("name") or "build.txt")
            self.fs.write_text(f"build/{name}", str(kwargs.get("content") or "isolated-build\n"))
            executed = True
        elif canonical == "model_candidate_creation":
            name = str(kwargs.get("name") or "candidate.json")
            body = dict(kwargs.get("payload") or {})
            body.setdefault("kind", "stub")
            body.setdefault("model_started", False)
            body.setdefault("measurement_state", "STATIC_ONLY")
            body.setdefault("bench_state", "UNKNOWN")
            self.fs.write_text(
                f"model_candidates/{name}",
                json.dumps(body, indent=2, sort_keys=True) + "\n",
            )
            executed = True
        elif canonical == "detached_workunit":
            spec = DetachedWorkUnitSpec(
                unit_id=str(kwargs.get("unit_id") or f"future.resident-sandbox.detached.{self.sandbox_id}"),
                payload=dict(kwargs.get("payload") or {"sandbox_id": self.sandbox_id}),
            )
            self.fs.write_text(
                "experiment/detached_workunit.json",
                json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
            )
            executed = True
        elif canonical == "temporary_child_process":
            # Permitted as a token. Provisioning does not spawn, and never a model.
            executed = False
        out = ActionDecision(
            allowed=True,
            action=action,
            authority=canonical,
            reason=decision.reason,
            executed=executed,
        )
        self._log.append(out.to_dict())
        return out

    def start_model(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityRefused(
            "hardware_risk_action",
            "sandbox does not start a model; provisioning only",
        )

    def merge_canonical(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityRefused("canonical_merge", HIGHER_REASONS["canonical_merge"])

    def teardown(self, *, preserve_artifacts: bool = True) -> dict[str, Any]:
        if not preserve_artifacts:
            raise AuthorityRefused(
                "destructive_action",
                "teardown cannot destroy artifacts; preserve_artifacts is required",
            )
        body = _state_identity_body(
            sandbox_id=self.sandbox_id,
            canonical_root=str(self.canonical_root),
            relative_worktree=os.path.relpath(self.worktree, self.canonical_root),
            bound=self.bound,
        )
        doc = _write_state(
            self.worktree,
            body,
            newly_created=False,
            torn_down=True,
            path_taken="teardown_preserved",
        )
        self.state_doc = doc
        self.newly_created = False
        surviving = sorted(
            p.name for p in self.writable_root.iterdir() if p.name != ".git"
        )
        return {
            "torn_down": True,
            "artifacts_preserved": True,
            "worktree_removed": False,
            "writable_entries": surviving,
            "worktree": str(self.worktree),
        }

    def emit_workunit(self) -> dict[str, Any]:
        return emit_provision_workunit(self.sandbox_id, workspace=str(self.worktree))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "canonical_root": str(self.canonical_root),
            "worktree": str(self.worktree),
            "writable_root": str(self.writable_root),
            "newly_created": self.newly_created,
            "layout": self.layout(),
            "state": self.state(),
            "bound": self.bound.to_dict(),
            "git_meta": self.git_meta,
            "workspace_git_root": self.workspace.git_root,
        }


def _reenter(
    *,
    sandbox_id: str,
    canonical: Path,
    worktree: Path,
    bound: SandboxBound,
    path_taken: str,
) -> ResidentSandbox:
    state_path = worktree / STATE_NAME
    if not state_path.is_file():
        raise ProvisionError(f"cannot re-enter {worktree}: {STATE_NAME} missing")
    writable = worktree / WRITABLE_DIRNAME
    if not writable.is_dir():
        writable = _init_layout(worktree)
    state = load_json(state_path)
    stored_id = str(state.get("sandbox_id") or sandbox_id)
    if stored_id != sandbox_id:
        raise ProvisionError(
            f"state sandbox_id {stored_id!r} does not match requested {sandbox_id!r}"
        )
    return ResidentSandbox(
        sandbox_id=sandbox_id,
        canonical_root=canonical,
        worktree=worktree,
        writable=writable,
        bound=bound,
        state=state,
        newly_created=False,
        git_meta={"path_taken": path_taken, "reentered": True},
    )


def provision(
    sandbox_id: str = DEFAULT_SANDBOX_ID,
    *,
    canonical_root: str | Path | None = None,
    bound: SandboxBound | None = None,
) -> ResidentSandbox:
    """Idempotent provision. Re-entry recovers the same sandbox, never a second one."""
    sid = _safe_id(sandbox_id)
    canonical = Path(os.path.realpath(str(canonical_root or REPO)))
    _require_git_repo(canonical)
    envelope = bound if bound is not None else SandboxBound()
    worktree = canonical / WORKTREES_DIRNAME / sid
    relative = os.path.join(WORKTREES_DIRNAME, sid)

    if (worktree / STATE_NAME).is_file() and (worktree / WRITABLE_DIRNAME).is_dir():
        box = _reenter(
            sandbox_id=sid,
            canonical=canonical,
            worktree=worktree,
            bound=envelope,
            path_taken="reentered_existing_state",
        )
        _store_registry(canonical, sid, relative)
        return box

    registered = (_load_registry(canonical).get("sandboxes") or {}).get(sid)
    if registered and (worktree / STATE_NAME).is_file():
        box = _reenter(
            sandbox_id=sid,
            canonical=canonical,
            worktree=worktree,
            bound=envelope,
            path_taken="reentered_registry",
        )
        return box

    listed = _worktree_paths(canonical)
    real_wt = os.path.realpath(str(worktree))
    if real_wt in listed and (worktree / STATE_NAME).is_file():
        return _reenter(
            sandbox_id=sid,
            canonical=canonical,
            worktree=worktree,
            bound=envelope,
            path_taken="reentered_git_worktree",
        )

    git_meta: dict[str, Any]
    if real_wt in listed:
        git_meta = {"created": False, "already_registered": True, **listed[real_wt]}
        path_taken = "completed_partial_worktree"
    else:
        git_meta = _add_worktree(canonical, worktree)
        path_taken = "provisioned_new_worktree"

    writable = _init_layout(worktree)
    index = build_source_index(canonical)
    (writable / "source_index" / "INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    body = _state_identity_body(
        sandbox_id=sid,
        canonical_root=str(canonical),
        relative_worktree=relative,
        bound=envelope,
    )
    state = _write_state(
        worktree,
        body,
        newly_created=True,
        torn_down=False,
        path_taken=path_taken,
    )
    _store_registry(canonical, sid, relative)
    return ResidentSandbox(
        sandbox_id=sid,
        canonical_root=canonical,
        worktree=worktree,
        writable=writable,
        bound=envelope,
        state=state,
        newly_created=True,
        git_meta=git_meta,
    )


def init_fixture_repo(path: str | Path) -> Path:
    """Tiny git repo for tests and selftest. Never the live Hawking tree."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ProvisionError(f"git init failed: {proc.stderr.strip()}")
    subprocess.run(
        ["git", "config", "user.email", "sandbox@hawking.invalid"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "resident-sandbox"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    (root / "README").write_text("resident sandbox fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def emit_provision_workunit(
    sandbox_id: str = DEFAULT_SANDBOX_ID,
    *,
    workspace: str | None = None,
) -> dict[str, Any]:
    sid = _safe_id(sandbox_id)
    unit = WorkUnit(
        id=f"future.resident-sandbox.provision.{sid}",
        role="science",
        description=(
            f"Provision isolated resident sandbox {sid!r} under .worktrees/ "
            "(no model start; bounded writes; authority lattice)."
        ),
        dependencies=[],
        resource_class="LIGHT_CONTROL",
        preferred_backend=None,
        provider="future.sandbox",
        verifier="future.sandbox.provision_contract",
        effect_class="REVERSIBLE",
        workspace=workspace or f".worktrees/{sid}",
        classification="STATIC_ONLY",
        status="pending",
        repair_depth=0,
    )
    row = unit.to_dict()
    row.update(
        {
            "claim_boundary": CLAIM_BOUNDARY,
            "species_local": "resident_sandbox_provision",
            "sandbox_id": sid,
            "requires_quiescence": False,
            "starts_model": False,
            "gpu_windows_held": 0,
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "budget": {
                "attempts": DEFAULT_RETRY_BUDGET,
                "max_repair_depth": MAX_REPAIR_DEPTH,
                "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
                "gpu_windows_held": 0,
                "gpu_windows_requested": 0,
            },
            "stop_conditions": [
                "stop when the sandbox is provisioned and SANDBOX.json is on disk",
                "stop when a higher-authority action is requested; refuse by name",
                "never start a model; never take a GPU lease",
                "never git worktree remove; teardown preserves artifacts",
            ],
            "integration_point_detached": "tools/future/detached.py (not imported)",
        }
    )
    WorkUnit.from_dict(dict(row))
    return row


def prove_filesystem_refusals(
    box: ResidentSandbox,
    *,
    outside: Path,
) -> list[dict[str, Any]]:
    """Watch absolute / .. / symlink writes fail. A guard nobody saw fail is not a guard."""
    outside = Path(outside)
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("untouched\n", encoding="utf-8")
    results: list[dict[str, Any]] = []

    def _trial(kind: str, thunk: Any, path_label: str) -> None:
        try:
            thunk()
        except SandboxEscapeError as exc:
            if exc.kind != kind:
                raise AssertionError(
                    f"escape kind {exc.kind!r} did not match expected {kind!r} for {path_label}"
                ) from exc
            if outside.read_text(encoding="utf-8") != "untouched\n":
                raise AssertionError(f"{kind} escape mutated the outside file")
            results.append(
                {
                    "kind": kind,
                    "path": path_label,
                    "refused": True,
                    "error": str(exc),
                }
            )
            return
        raise AssertionError(f"filesystem guard did not fire for {kind} ({path_label})")

    _trial(
        "absolute",
        lambda: box.fs.write_text(str(outside.resolve()), "pwned\n"),
        str(outside),
    )
    _trial(
        "dotdot",
        lambda: box.fs.write_text("../ESCAPE_DOTDOT.txt", "pwned\n"),
        "../ESCAPE_DOTDOT.txt",
    )
    link = box.writable_root / "escape_link"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(str(outside.resolve()), str(link))
    _trial("symlink", lambda: box.fs.write_text("escape_link", "pwned\n"), "escape_link")
    if outside.read_text(encoding="utf-8") != "untouched\n":
        raise AssertionError("symlink escape mutated the outside file")
    return results


def prove_higher_authority_refusals(box: ResidentSandbox) -> list[dict[str, Any]]:
    """Refuse each of the nine higher-authority actions separately, by name."""
    results: list[dict[str, Any]] = []
    for authority in sorted(REFUSED_HIGHER):
        try:
            box.attempt(authority)
        except AuthorityRefused as exc:
            if exc.authority != authority:
                raise AssertionError(
                    f"refusal named {exc.authority!r} not {authority!r}"
                ) from exc
            if authority not in str(exc):
                raise AssertionError(
                    f"refusal message did not name {authority}: {exc}"
                ) from exc
            results.append(
                {
                    "authority": authority,
                    "refused": True,
                    "error": str(exc),
                }
            )
            continue
        raise AssertionError(f"authority guard did not fire for {authority}")
    if {r["authority"] for r in results} != set(REFUSED_HIGHER):
        raise AssertionError("did not prove every refused higher-authority action")
    return results


def prove_module_disjoint() -> dict[str, Any]:
    here = Path(__file__).resolve()
    test = here.parent / "test_sandbox.py"
    paths = [str(here)]
    if test.is_file():
        paths.append(str(test))
    code = check_disjoint(paths)
    return {
        "exit_code": code,
        "disjoint": code == 0,
        "paths": [
            os.path.relpath(p, REPO).replace(os.sep, "/") for p in paths
        ],
    }


def recovered_implementation() -> list[dict[str, Any]]:
    rows = (
        (
            "hcli/workspace.py",
            "Workspace.realpath root + git-toplevel detect; too thin to bound writes, reused as the writable Workspace",
        ),
        (
            "hcli/mutation.py",
            "bounded reversible mutations, snapshots, rollback_mutation; recovered as the model for local_reversible_experiment",
        ),
        (
            "hcli/agentos/autonomy_gate.py",
            "A1-A5 autonomy qualification; not invoked (would start a resident)",
        ),
        (
            "hcli/agentos/resident_gate.py",
            "LIVE_RESIDENT_SEQUENTIAL_PROOF; lifecycle owner; this sandbox does not start it",
        ),
        (
            "hcli/agentos/resident.py",
            "does not exist in HEAD; resident_gate.py is the live boundary",
        ),
        (
            "hcli/persist.py",
            "atomic_write_text / atomic_write_json; used for SANDBOX.json, INDEX.json, bounded writes",
        ),
        (
            "hcli/workunit.py",
            "WorkUnit constructor + repair budgets; provision emits through it",
        ),
        (
            "tools/future/resident_optimizer.py",
            "OptimizerBound + BoundViolation + ALLOWED/FORBIDDEN_AUTHORITY; reused as the inner lattice",
        ),
        (
            "tools/future/mutation_surface.py",
            "check_disjoint / intersects_codex; composed so Codex paths cannot be written",
        ),
        (
            "research/lab/execution_sandbox.py",
            "bible §21 ExecutionSandboxPolicy; recovered via git show; not imported (sparse, different vocabulary)",
        ),
        (
            "research/lab/operators/ascension_sandbox.py",
            "detached research-sandbox controller; never loads a model; recovered via git show",
        ),
        (
            "crates/hide-kernel/src/security_sandbox.rs",
            "OS Seatbelt profile; fail-closed spawn; orthogonal to in-process BoundedFS",
        ),
        (
            "tools/grok_worktree_reaper.py",
            "git worktree remove only, never filesystem force-delete; teardown here does not remove",
        ),
        (
            "workspace/campaign/governance/odyssey/program/sandbox/POLICY.json",
            "Odyssey sandbox POLICY (deny-by-default fs/network); recovered via git show",
        ),
    )
    out: list[dict[str, Any]] = []
    for rel, what in rows:
        path = REPO / rel
        out.append(
            {
                "path": rel,
                "present": path.is_file(),
                "what": what,
            }
        )
    return out


def resident_callable(work_unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/sandbox.py --provision|--selftest|--build",
        "callable": "tools.future.sandbox.provision",
        "workunit": {
            "id": work_unit.get("id"),
            "species_local": work_unit.get("species_local"),
            "verifier": work_unit.get("verifier"),
            "effect_class": work_unit.get("effect_class"),
            "classification": work_unit.get("classification"),
            "starts_model": False,
            "emitted_by": "tools.future.sandbox.emit_provision_workunit",
        },
        "receipt": "receipts/future/RESIDENT_SANDBOX.json",
        "frontier_fed": {
            "name": "laboratory / Odyssey I WHAT IS TRUE?",
            "integration_point": "tools/future/frontiers.py (this wave; not imported)",
            "effect": (
                "a provisioned sandbox the resident can discover, invoke, "
                "schedule and verify; disk state persists; next work refills"
            ),
        },
        "fail_closed": {
            "filesystem_escape": "SandboxEscapeError (absolute | dotdot | symlink)",
            "higher_authority": "AuthorityRefused naming the authority",
            "optimizer_lattice": "BoundViolation at OptimizerBound / SandboxBound construction",
            "codex_surface": "CanonicalMutationError + mutation_surface.check_disjoint",
            "hardware": "HardwareClaimError via write_receipt",
            "model_start": "AuthorityRefused hardware_risk_action",
            "destroy": "AuthorityRefused destructive_action (teardown preserves)",
        },
    }


def _hermetic_proofs() -> dict[str, Any]:
    """Run provision / escape / authority / longevity proofs in a throwaway repo."""
    tmp = tempfile.TemporaryDirectory(prefix="resident-sandbox-")
    try:
        canonical = init_fixture_repo(Path(tmp.name) / "canonical")
        box = provision(DEFAULT_SANDBOX_ID, canonical_root=canonical)
        if not box.newly_created:
            raise AssertionError("first provision of a new fixture must be a creation")
        for name in LAYOUT_DIRS:
            if not (box.writable_root / name).is_dir():
                raise AssertionError(f"missing layout dir {name}")
        box.attempt("local_reversible_experiment", name="keep.txt", content="stay\n")
        again = provision(DEFAULT_SANDBOX_ID, canonical_root=canonical)
        if again.newly_created:
            raise AssertionError("second provision created a second sandbox")
        if os.path.realpath(str(again.worktree)) != os.path.realpath(str(box.worktree)):
            raise AssertionError("re-entry recovered a different worktree")
        if again.fs.read_text("experiment/keep.txt") != "stay\n":
            raise AssertionError("re-entry lost experiment artifacts")
        if again.state().get("identity_sha256") != box.state().get("identity_sha256"):
            raise AssertionError("re-entry recovered a different sandbox identity")

        outside = Path(tmp.name) / "outside.txt"
        fs_rows = prove_filesystem_refusals(again, outside=outside)
        auth_rows = prove_higher_authority_refusals(again)

        try:
            again.start_model()
            model_refused = False
            model_error = "start_model did not refuse"
        except AuthorityRefused as exc:
            model_refused = exc.authority == "hardware_risk_action"
            model_error = str(exc)
        if not model_refused:
            raise AssertionError(model_error)

        preserved = again.teardown(preserve_artifacts=True)
        if not (again.writable_root / "experiment" / "keep.txt").is_file():
            raise AssertionError("teardown deleted experiment artifacts")
        reentered = provision(DEFAULT_SANDBOX_ID, canonical_root=canonical)
        if reentered.newly_created:
            raise AssertionError("post-teardown provision created a second sandbox")
        if reentered.fs.read_text("experiment/keep.txt") != "stay\n":
            raise AssertionError("post-teardown re-entry lost artifacts")

        disjoint = prove_module_disjoint()
        if not disjoint["disjoint"]:
            raise AssertionError("sandbox module intersects the Codex mutation surface")

        unit = again.emit_workunit()
        permitted_ok = []
        for action in sorted(PERMITTED_AUTONOMOUS):
            # torn_down freezes most actions; prove permit on a fresh id
            permitted_ok.append(action)

        fresh = provision("permittest", canonical_root=canonical)
        for action in sorted(PERMITTED_AUTONOMOUS):
            decision = fresh.authorize(action)
            if not decision.allowed:
                raise AssertionError(f"permitted action {action} was not allowed")
            permitted_ok.append(action)

        try:
            SandboxBound(OptimizerBound(may_widen_authority=True))
            bound_ok = False
        except BoundViolation:
            bound_ok = True
        if not bound_ok:
            raise AssertionError("SandboxBound accepted may_widen_authority")

        return {
            "canonical_root": str(canonical),
            "worktree": str(again.worktree),
            "writable_root": str(again.writable_root),
            "sandbox_id": again.sandbox_id,
            "identity_sha256": again.state().get("identity_sha256"),
            "newly_created_first": True,
            "reentry_same_identity": True,
            "layout": list(LAYOUT_DIRS),
            "filesystem_refusals": fs_rows,
            "higher_authority_refusals": auth_rows,
            "model_start_refused": True,
            "model_start_error": model_error,
            "teardown": preserved,
            "disjoint": disjoint,
            "workunit_id": unit["id"],
            "permitted_authorized": sorted(set(permitted_ok)),
            "source_index": build_source_index(canonical),
            "production_worktrees_root": str(Path(REPO) / WORKTREES_DIRNAME),
            "git_meta": box.git_meta,
        }
    finally:
        tmp.cleanup()


def build() -> Path:
    proofs = _hermetic_proofs()
    unit = emit_provision_workunit(DEFAULT_SANDBOX_ID)
    fs_rows = proofs["filesystem_refusals"]
    auth_rows = proofs["higher_authority_refusals"]
    live_index = build_source_index(REPO)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": SIDECAR_STATUS,
        "promoted": False,
        "built": True,
        "purpose": (
            "Isolated laboratory a future super-resident occupies for hours: "
            "git worktree under .worktrees/, bounded writable filesystem, "
            "authority lattice, durable re-entry. Provisioning only; no model."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": (
                "Accelerator / Physical Compiler / Fusion; not a civilization "
                "and this module does not build an FPGA backend"
            ),
            "disk_state_is_authority": True,
            "models_think_tools_know_context_is_a_cache": True,
            "diagnostic_relative_never_promotes": True,
            "protected_absolute_not_emitted": True,
        },
        "provision": {
            "worktrees_dirname": WORKTREES_DIRNAME,
            "default_sandbox_id": DEFAULT_SANDBOX_ID,
            "layout_dirs": list(LAYOUT_DIRS),
            "state_name": STATE_NAME,
            "registry_name": REGISTRY_NAME,
            "git_worktree": (
                "git worktree add --detach --no-checkout --lock "
                "(reason: resident-sandbox preserve-artifacts)"
            ),
            "idempotent": True,
            "teardown_preserves_artifacts": True,
            "teardown_runs_worktree_remove": False,
            "production_worktrees_root": proofs["production_worktrees_root"],
            "hermetic_proof_worktree": proofs["worktree"],
            "path_taken": proofs.get("git_meta", {}).get("created"),
        },
        "bounded_filesystem": {
            "write_root": WRITABLE_DIRNAME,
            "refuses": ["absolute", "dotdot", "symlink"],
            "mechanism": "BoundedFS.resolve_writable walks components and refuses escape by kind",
            "proofs": fs_rows,
            "all_refused": bool(fs_rows) and all(r["refused"] for r in fs_rows),
            "kinds_proven": sorted({r["kind"] for r in fs_rows}),
        },
        "authority": {
            "permitted": sorted(PERMITTED_AUTONOMOUS),
            "refused": sorted(REFUSED_HIGHER),
            "optimizer_bound": SandboxBound().to_dict(),
            "refusals_proven": auth_rows,
            "all_higher_refused": bool(auth_rows)
            and all(r["refused"] for r in auth_rows)
            and {r["authority"] for r in auth_rows} == set(REFUSED_HIGHER),
            "model_start_refused": proofs["model_start_refused"],
        },
        "longevity": {
            "reentry_same_identity": proofs["reentry_same_identity"],
            "identity_sha256": proofs["identity_sha256"],
            "state_survives_process_restart": True,
            "second_provision_creates_second_sandbox": False,
        },
        "canonical_mutation": {
            "composed": "tools.future.mutation_surface.check_disjoint / intersects_codex",
            "disjoint": proofs["disjoint"],
            "writes_canonical_worktree": False,
            "writes_codex_surface": False,
        },
        "no_model_start": {
            "starts_model": False,
            "refused_as": "hardware_risk_action",
            "error": proofs["model_start_error"],
        },
        "source_index": live_index,
        "hermetic_source_index": proofs["source_index"],
        "work_unit": unit,
        "counts": {
            "permitted_autonomous": len(PERMITTED_AUTONOMOUS),
            "refused_higher": len(REFUSED_HIGHER),
            "layout_dirs": len(LAYOUT_DIRS),
            "filesystem_kinds_proven": len({r["kind"] for r in fs_rows}),
            "higher_refusals_proven": len(auth_rows),
            "source_index_files": live_index["file_count_total"],
            "hermetic_source_index_files": proofs["source_index"]["file_count_total"],
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": [
            "isolated git worktree under the canonical repo's own .worktrees/",
            "idempotent provision; re-entry recovers the same sandbox identity",
            "teardown preserves artifacts and refuses destroy",
            "BoundedFS refuses absolute, .. traversal, and symlink escape by kind",
            "ten autonomous actions permitted; nine higher-authority actions refused by name",
            "OptimizerBound / BoundViolation / FORBIDDEN_AUTHORITY reused, not forked",
            "mutation_surface.check_disjoint composed; Codex paths cannot be written",
            "SANDBOX.json + .worktrees/INDEX.json survive process restart",
            "provisioning does not start a model",
            "HCLI WorkUnit emitted for provision; verifier is not the proposer",
        ],
        "negative_findings": [
            "research/lab/execution_sandbox.py is bible §21 policy, not a worktree provisioner; not imported (different action vocabulary); present-or-missing is recorded in recovered_implementation",
            "OS-level Seatbelt confinement (hide-kernel security_sandbox) is orthogonal; BoundedFS is the in-process gate",
            "temporary_child_process is a permitted token; provisioning does not spawn, and a raw child could write outside unless OS-confined",
            "this-wave siblings (detached, frontiers, wakeup, super_resident, resident_api, protected_window) are not imported",
            "live Hawking .worktrees/ is not mutated by --selftest/--build; those use a hermetic fixture repo",
            "MetalContext reports no Metal-capable GPU on this host; that remains a SLEEPING physical blocker, not a synthetic result",
            "this lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
            "hcli/agentos/resident.py is recorded present/missing in recovered_implementation; resident_gate.py is the live sequential-proof boundary",
            "no GPU lease, no paid action, no bounty, no external message",
        ],
        "resident_callable": resident_callable(unit),
        "integration_points": list(INTEGRATION_POINTS),
        "sleeping_physical_blockers": [
            "MetalContext reports no Metal-capable GPU on this host",
            "xcrun cannot locate the Metal compiler under CommandLineTools",
            "protected bench lock files exist; holder pids unproven; flock would be a seizure",
            "qualification pipeline classifies the machine HEAVY and will not quiesce standing workers",
            "Flash source-independent NX is SCAFFOLD_ONLY, not qualified",
            "teacher capture is 0/256",
        ],
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--provision", action="store_true")
    ap.add_argument("--id", default=DEFAULT_SANDBOX_ID)
    ap.add_argument("--canonical", default=None, help="canonical git root (default: REPO)")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    if args.provision:
        box = provision(args.id, canonical_root=args.canonical)
        print(json.dumps(box.to_dict(), indent=2, sort_keys=True))
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
