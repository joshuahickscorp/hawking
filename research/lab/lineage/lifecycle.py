"""Durable external control plane for Genesis candidate qualification.

The lineage primitives already know how to gate a successor and transfer its
science.  This module connects them to the things a running organism needs:

* a durable registry for the logical worker tasks (not model children),
* protected parent/candidate benchmark and greedy-probe receipts,
* an external controller which can checkpoint/rebind workers before authority
  moves, and
* compare-and-swap state writes so a stale controller cannot overwrite a newer
  generation.

Candidate-authored JSON is a request, never promotion authority.  The
controller independently hashes the artifact/runtime/kernel, runs the supplied
benchmark executable under the one GPU lease, and asks the existing mechanical
promotion gate to decide from those derived values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lab.lineage.bus import ResearchBus
from lab.lineage.canon import require_sha256, utc_now
from lab.lineage.continuity import (
    WorkerCheckpointStore,
    migrate_workers,
    normalize_generation,
    normalize_worker,
)
from lab.lineage.identity import GenesisInstance, Invoker, file_sha256
from lab.lineage.promotion import REQUIRED_PROTECTED_TESTS, evaluate_promotion
from lab.lineage.state import LineageError, LineageState
from lab.lineage.transfer import pack_state, parent_research_payload, payload_checksum
from lab.qwen38_protected_run_verifier import verify_qwen38_capture
from lab.receipts import seal, verify


REPO = Path(__file__).resolve().parents[2]

CANDIDATE_SCHEMA = "hawking.genesis.candidate_request.v1"
CANDIDATE_SUBMISSION_SCHEMA = "hawking.genesis.candidate_submission.v1"
PAIR_REQUEST_SCHEMA = "hawking.genesis.protected_pair_request.v1"
PAIR_RECEIPT_SCHEMA = "hawking.genesis.protected_pair_receipt.v1"
WORKER_REGISTRY_SCHEMA = "hawking.genesis.worker_registry.v1"
LIFECYCLE_RECEIPT_SCHEMA = "hawking.genesis.lifecycle_receipt.v1"

DEFAULT_LINEAGE_PATH = REPO / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"
DEFAULT_WORKER_REGISTRY = REPO / "workspace" / "ops" / "genesis-workers.json"
DEFAULT_CHECKPOINT_ROOT = REPO / "workspace" / "ops" / "genesis-worker-checkpoints"
DEFAULT_CANDIDATE_ROOT = REPO / "workspace" / "ops" / "genesis-candidates"
# A model never writes the operator's dirty primary checkout.  Logical workers
# receive sparse linked worktrees here, which survive a resident generation
# change but remain separate from the authoritative live tree.
DEFAULT_WORKTREE_ROOT = REPO / "workspace" / "ops" / "local" / "genesis-agentos-worktrees"
DEFAULT_GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"

GPU_TIMING_AUTHORITY = "MTLCommandBuffer GPUStartTime/GPUEndTime after wait"
DEFAULT_GREEDY_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time",
)


class LifecycleError(ValueError):
    """A candidate request is incomplete, stale, or unsafe to activate."""


class LifecycleBusy(LifecycleError):
    """The protected GPU lane is occupied; the candidate remains pending."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise LifecycleError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise LifecycleError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{label} must be a JSON object")
    return value, raw


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    expected_previous_sha256: str | None,
) -> None:
    """Fsync + replace bytes only if no other controller won the race."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        if expected_previous_sha256 is not None:
            raise LifecycleError(f"{target} disappeared before compare-and-swap")
        mode = 0o644
    except OSError as exc:
        raise LifecycleError(f"cannot stat {target}: {exc}") from exc

    fd, name = tempfile.mkstemp(prefix=f".{target.name}.lifecycle-", dir=target.parent)
    tmp = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_previous_sha256 is None:
            if target.exists():
                raise LifecycleError(f"{target} appeared during compare-and-swap")
        else:
            try:
                observed = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError as exc:
                raise LifecycleError(f"cannot re-read {target}: {exc}") from exc
            if observed != expected_previous_sha256:
                raise LifecycleError(f"{target} changed during compare-and-swap")
        os.replace(tmp, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None,
) -> None:
    """Fsync + replace a JSON document only if no controller won the race."""
    _atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        expected_previous_sha256=expected_previous_sha256,
    )


def _resolve_path(value: object, *, repo: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise LifecycleError(f"cannot resolve {label} {candidate}: {exc}") from exc


def _regular_file(value: object, *, repo: Path, label: str, executable: bool = False) -> Path:
    path = _resolve_path(value, repo=repo, label=label)
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise LifecycleError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise LifecycleError(f"{label} is not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise LifecycleError(f"{label} is not executable: {path}")
    return path


def _directory(value: object, *, repo: Path, label: str) -> Path:
    path = _resolve_path(value, repo=repo, label=label)
    if not path.is_dir():
        raise LifecycleError(f"{label} is not a directory: {path}")
    return path


def _git_head(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LifecycleError(f"cannot determine repository head: {exc}") from exc
    head = result.stdout.strip().lower()
    if result.returncode != 0 or len(head) not in {40, 64} or any(c not in "0123456789abcdef" for c in head):
        raise LifecycleError(f"cannot determine a valid repository head: {result.stderr.strip()}")
    return head


def _git_checked(repo: Path, argv: Sequence[str], *, label: str) -> str:
    """Run a bounded Git operation without a shell or an implicit checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *argv],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LifecycleError(f"cannot {label}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2_000:]
        raise LifecycleError(f"cannot {label}: {detail}")
    return result.stdout.strip()


def generation_record(instance: GenesisInstance, *, repo_head: str) -> dict[str, Any]:
    """Project immutable instance identity into the continuity compiler shape."""
    if instance.physical_bpw is None:
        raise LifecycleError(f"{instance.instance_id} has no physical_bpw")
    record = {
        "generation": instance.generation,
        "artifact_sha": instance.artifact_sha,
        "runtime_sha": instance.runtime_sha,
        "repo_head": repo_head,
        "physical_bpw": instance.physical_bpw,
        "complete_token_ns": instance.complete_token_ns,
    }
    return normalize_generation(record)


class WorkerRegistry:
    """Atomic AgentOS worker registry; model/KV state never lives here."""

    def __init__(self, path: Path = DEFAULT_WORKER_REGISTRY) -> None:
        self.path = Path(path)

    def load(self) -> tuple[list[dict[str, Any]], str | None]:
        if not self.path.exists():
            return [], None
        raw, raw_bytes = _load_json_object(self.path, "worker registry")
        if raw.get("schema") != WORKER_REGISTRY_SCHEMA:
            raise LifecycleError(f"unexpected worker registry schema {raw.get('schema')!r}")
        workers = raw.get("workers")
        if not isinstance(workers, list):
            raise LifecycleError("worker registry.workers must be a list")
        normalized = [normalize_worker(item) for item in workers]
        if len({item["worker_id"] for item in normalized}) != len(normalized):
            raise LifecycleError("worker registry contains duplicate worker_id values")
        return normalized, hashlib.sha256(raw_bytes).hexdigest()

    def replace(
        self,
        workers: Iterable[Mapping[str, Any]],
        *,
        expected_previous_sha256: str | None,
    ) -> None:
        normalized = [normalize_worker(item) for item in workers]
        if len({item["worker_id"] for item in normalized}) != len(normalized):
            raise LifecycleError("refusing duplicate worker_id values")
        _atomic_write_json(
            self.path,
            {
                "schema": WORKER_REGISTRY_SCHEMA,
                "updated_at": utc_now(),
                "workers": normalized,
            },
            expected_previous_sha256=expected_previous_sha256,
        )

    def bootstrap(self, *, generation: Mapping[str, Any], repo: Path) -> list[dict[str, Any]]:
        """Install the two durable logical fronts only when no registry exists.

        These records are deliberately task state, not invented experiment
        findings.  They give child_a and child_b explicit durable homes so a
        later promotion cannot silently abandon the Gravity and kernel fronts.
        """
        existing, previous = self.load()
        if existing:
            return existing
        bound = normalize_generation(generation)
        base = {
            "repo": str(repo.resolve()),
            "worktree": str(repo.resolve()),
            "hypotheses": [],
            "findings": [],
            "receipts": [],
            "negative_science": [],
            "pending_experiments": [],
            "tool_results": [],
        }
        workers = [
            {
                "worker_id": "gravity",
                "session_role": "child_a",
                "task_id": "genesis-gravity-frontier",
                "task_contract": {"front": "Gravity/Doctor", "version": 1},
                "priority": 10,
                "state": "READY",
                "durable_task_state": {
                    **base,
                    "goal": "Lower Genesis physical BPW and unique-once weight bytes without capability loss.",
                    "subgoal": "Build and falsify successor representation artifacts.",
                    "NEXT_ACTION": "Read current negative science and select the smallest representation discriminator.",
                },
                "generation_dependencies": ["artifact", "runtime"],
                "bound_generation": bound,
            },
            {
                "worker_id": "kernel",
                "session_role": "child_b",
                "task_id": "genesis-kernel-frontier",
                "task_contract": {"front": "kernel/execution genome", "version": 1},
                "priority": 10,
                "state": "READY",
                "durable_task_state": {
                    **base,
                    "goal": "Lower the complete-token wall of the current Genesis execution genome.",
                    "subgoal": "Build and falsify kernel/runtime successor artifacts.",
                    "NEXT_ACTION": "Profile the current complete token and select the largest measured execution cost.",
                },
                "generation_dependencies": ["artifact", "runtime"],
                "bound_generation": bound,
            },
        ]
        self.replace(workers, expected_previous_sha256=previous)
        return [normalize_worker(item) for item in workers]

    def provision_worktrees(
        self,
        *,
        repo: Path,
        worktree_root: Path = DEFAULT_WORKTREE_ROOT,
    ) -> list[dict[str, Any]]:
        """Give each logical worker an isolated sparse linked worktree.

        The live checkout is intentionally often dirty: it contains resident
        state, operator changes, and other lanes.  An HCLI worker may edit and
        test only its own linked worktree.  This method never resets either
        checkout, never removes an existing worktree, and does not overwrite a
        worker's changes on repeated calls.
        """
        source = Path(repo).resolve()
        if not source.is_dir():
            raise LifecycleError(f"cannot provision worker worktrees: repo is missing {source}")
        top = _git_checked(source, ["rev-parse", "--show-toplevel"], label="find source Git root")
        if Path(top).resolve() != source:
            raise LifecycleError(
                f"cannot provision worker worktrees: repo must be its Git root ({source})"
            )
        workers, previous_sha = self.load()
        if previous_sha is None or not workers:
            raise LifecycleError("cannot provision worktrees before the worker registry is bootstrapped")
        base_commit = _git_head(source)
        root = Path(worktree_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        changed = False
        for worker in workers:
            durable = worker["durable_task_state"]
            recorded = Path(str(durable["worktree"])).expanduser()
            if not recorded.is_absolute():
                recorded = source / recorded
            recorded = recorded.resolve()
            # Preserve an already-provisioned worker home.  Otherwise migrate
            # the bootstrap default (the live source root) to a stable path.
            target = recorded if recorded != source else root / worker["worker_id"]
            if target.exists():
                try:
                    worker_top = _git_checked(
                        target,
                        ["rev-parse", "--show-toplevel"],
                        label=f"inspect worker worktree {worker['worker_id']}",
                    )
                except LifecycleError:
                    raise LifecycleError(
                        f"refusing to reuse non-Git worker worktree {target} for {worker['worker_id']}"
                    ) from None
                if Path(worker_top).resolve() != target.resolve():
                    raise LifecycleError(
                        f"worker worktree root mismatch for {worker['worker_id']}: {target}"
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                _git_checked(
                    source,
                    ["worktree", "add", "--detach", "--no-checkout", str(target), base_commit],
                    label=f"create isolated worktree for {worker['worker_id']}",
                )
                # Sparse checkout keeps a second worker home cheap even though
                # the primary repository contains large campaign artifacts.
                _git_checked(
                    target,
                    ["sparse-checkout", "init", "--cone"],
                    label=f"initialize sparse checkout for {worker['worker_id']}",
                )
                _git_checked(
                    target,
                    ["sparse-checkout", "set", "--cone", "lab", "tools", "contracts", ".cargo"],
                    label=f"select source paths for {worker['worker_id']}",
                )
                _git_checked(
                    target,
                    ["checkout", "--detach", base_commit],
                    label=f"populate isolated worktree for {worker['worker_id']}",
                )
            target_text = str(target.resolve())
            if durable.get("worktree") != target_text or durable.get("worktree_isolated") is not True:
                durable["worktree"] = target_text
                durable["worktree_isolated"] = True
                durable["worktree_base_commit"] = base_commit
                durable["worktree_provisioned_at"] = utc_now()
                changed = True
        if changed:
            self.replace(workers, expected_previous_sha256=previous_sha)
        return workers


@dataclass(frozen=True)
class BenchmarkProfile:
    """One independently hashed runnable arm of a protected A/B test."""

    instance: GenesisInstance
    artifact_root: Path
    tokenizer: Path
    benchmark_runtime: Path
    kernel_source: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance.instance_id,
            "generation": self.instance.generation,
            "artifact_root": str(self.artifact_root),
            "artifact_sha": self.instance.artifact_sha,
            "tokenizer": str(self.tokenizer),
            "benchmark_runtime": str(self.benchmark_runtime),
            "benchmark_runtime_sha256": file_sha256(self.benchmark_runtime),
            "kernel_source": str(self.kernel_source),
            "kernel_source_sha256": file_sha256(self.kernel_source),
        }


@dataclass(frozen=True)
class CandidateSpec:
    """A proposed successor plus exact files the external controller may run."""

    instance: GenesisInstance
    artifact_root: Path
    tokenizer: Path
    resident_executable: Path
    benchmark_runtime: Path
    kernel_source: Path
    parent_payload: dict[str, Any]
    world_state: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, repo: Path) -> "CandidateSpec":
        if raw.get("schema") != CANDIDATE_SCHEMA:
            raise LifecycleError(f"candidate request schema must be {CANDIDATE_SCHEMA}")
        candidate_raw = raw.get("candidate")
        if not isinstance(candidate_raw, Mapping):
            raise LifecycleError("candidate request.candidate must be an object")
        try:
            instance = GenesisInstance.from_mapping(candidate_raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise LifecycleError(f"invalid candidate instance: {exc}") from exc
        artifact_root = _directory(raw.get("artifact_root"), repo=repo, label="artifact_root")
        manifest = _regular_file(
            str(artifact_root / "manifest.json"), repo=repo, label="candidate manifest"
        )
        actual_artifact_sha = file_sha256(manifest)
        if actual_artifact_sha != instance.artifact_sha:
            raise LifecycleError(
                "candidate artifact_sha does not match independently hashed manifest bytes"
            )
        tokenizer = _regular_file(raw.get("tokenizer"), repo=repo, label="tokenizer")
        resident = _regular_file(
            raw.get("resident_executable"),
            repo=repo,
            label="candidate resident executable",
            executable=True,
        )
        if file_sha256(resident) != instance.runtime_sha:
            raise LifecycleError(
                "candidate runtime_sha does not match candidate resident executable bytes"
            )
        benchmark = _regular_file(
            raw.get("benchmark_runtime"),
            repo=repo,
            label="candidate benchmark runtime",
            executable=True,
        )
        kernel = _regular_file(raw.get("kernel_source"), repo=repo, label="candidate kernel source")
        if file_sha256(kernel) != instance.kernel_genome_sha:
            raise LifecycleError(
                "candidate kernel_genome_sha does not match candidate kernel source bytes"
            )
        try:
            manifest_json, _ = _load_json_object(manifest, "candidate manifest")
            manifest_bpw = float(manifest_json["complete_physical_bpw"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LifecycleError("candidate manifest.complete_physical_bpw is required") from exc
        if not math.isfinite(manifest_bpw) or manifest_bpw <= 0.0:
            raise LifecycleError("candidate manifest.complete_physical_bpw must be positive")
        if instance.physical_bpw is None or not math.isclose(
            float(instance.physical_bpw), manifest_bpw, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise LifecycleError(
                "candidate physical_bpw does not match independently read manifest"
            )
        payload = raw.get("parent_payload")
        if not isinstance(payload, Mapping):
            raise LifecycleError("candidate request.parent_payload must be an object")
        # payload_checksum normalizes every required field and rejects a blank next action.
        payload_copy = dict(payload)
        payload_checksum(payload_copy)
        world = raw.get("world_state") or {}
        if not isinstance(world, Mapping):
            raise LifecycleError("candidate request.world_state must be an object")
        return cls(
            instance=instance,
            artifact_root=artifact_root,
            tokenizer=tokenizer,
            resident_executable=resident,
            benchmark_runtime=benchmark,
            kernel_source=kernel,
            parent_payload=payload_copy,
            world_state=dict(world),
        )

    @property
    def benchmark_profile(self) -> BenchmarkProfile:
        return BenchmarkProfile(
            instance=self.instance,
            artifact_root=self.artifact_root,
            tokenizer=self.tokenizer,
            benchmark_runtime=self.benchmark_runtime,
            kernel_source=self.kernel_source,
        )

    def handoff_instance(self) -> GenesisInstance:
        """Return the child identity with the independently verified live paths.

        The request's free-form identity map is not enough for a restart after
        promotion.  Persist the paths and their measured hashes only after the
        controller has independently resolved every file above.
        """
        child = self.instance.copy()
        child.identity.update(
            {
                "artifact": str(self.artifact_root),
                "artifact_manifest": str(self.artifact_root / "manifest.json"),
                "artifact_sha_authority": "sha256(manifest.json bytes)",
                "tokenizer": str(self.tokenizer),
                "resident_executable": str(self.resident_executable),
                "resident_executable_sha256": file_sha256(self.resident_executable),
                "measurement_runtime": str(self.benchmark_runtime),
                "measurement_runtime_sha256": file_sha256(self.benchmark_runtime),
                "kernel_source": str(self.kernel_source),
                "kernel_source_sha256": file_sha256(self.kernel_source),
            }
        )
        return child


def build_candidate_request(
    *,
    authority_repo: Path,
    state_path: Path,
    origin_repo: Path,
    instance_id: str,
    artifact_root: str | Path,
    next_bottleneck: str,
    tokenizer: str | Path | None = None,
    resident_executable: str | Path | None = None,
    benchmark_runtime: str | Path | None = None,
    kernel_source: str | Path | None = None,
    world_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid but non-authoritative candidate request from real files.

    This is deliberately *staging*, not qualification. The worker supplies a
    changed artifact/runtime/kernel and a named next bottleneck; the helper
    hashes those exact bytes, carries the parent capability contract, and uses
    the parent wall only as a conservative placeholder. ``PromotionController``
    replaces that wall with a protected paired median before it can affect
    lineage.
    """
    controller_repo = Path(authority_repo).resolve()
    worker_repo = Path(origin_repo).resolve()
    state_raw, _ = _load_json_object(Path(state_path), "lineage state")
    lineage = LineageState.from_dict(state_raw)
    parent = lineage.current
    if parent is None or not parent.valid:
        raise LifecycleError("cannot build a candidate request without a valid CURRENT Genesis")
    if lineage.candidate is not None and lineage.candidate.valid:
        raise LifecycleError("cannot stage a second candidate while a valid CANDIDATE is present")
    candidate_artifact = _directory(str(artifact_root), repo=worker_repo, label="candidate artifact_root")
    manifest = _regular_file(
        str(candidate_artifact / "manifest.json"), repo=worker_repo, label="candidate manifest"
    )
    try:
        manifest_raw, _ = _load_json_object(manifest, "candidate manifest")
        physical_bpw = float(manifest_raw["complete_physical_bpw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleError("candidate manifest.complete_physical_bpw is required") from exc
    if not math.isfinite(physical_bpw) or physical_bpw <= 0.0:
        raise LifecycleError("candidate manifest.complete_physical_bpw must be positive")
    parent_tokenizer = _current_tokenizer(parent, repo=controller_repo)
    parent_profile = _current_profile(parent, repo=controller_repo, tokenizer=parent_tokenizer)
    parent_resident = _regular_file(
        parent.identity.get("resident_executable"),
        repo=controller_repo,
        label="CURRENT resident executable",
        executable=True,
    )
    selected_tokenizer = _regular_file(
        str(tokenizer) if tokenizer is not None else str(parent_tokenizer),
        repo=worker_repo,
        label="candidate tokenizer",
    )
    selected_resident = _regular_file(
        str(resident_executable) if resident_executable is not None else str(parent_resident),
        repo=worker_repo,
        label="candidate resident executable",
        executable=True,
    )
    selected_benchmark = _regular_file(
        str(benchmark_runtime) if benchmark_runtime is not None else str(parent_profile.benchmark_runtime),
        repo=worker_repo,
        label="candidate benchmark runtime",
        executable=True,
    )
    selected_kernel = _regular_file(
        str(kernel_source) if kernel_source is not None else str(parent_profile.kernel_source),
        repo=worker_repo,
        label="candidate kernel source",
    )
    child_identity = dict(parent.identity)
    child_identity.update(
        {
            "candidate_request_origin": str(worker_repo),
            "candidate_wall_is_untrusted": "controller derives protected paired median",
        }
    )
    child = GenesisInstance(
        instance_id=instance_id,
        generation=parent.generation + 1,
        artifact_sha=file_sha256(manifest),
        runtime_sha=file_sha256(selected_resident),
        kernel_genome_sha=file_sha256(selected_kernel),
        representation_bpw=physical_bpw,
        physical_bpw=physical_bpw,
        # This is a placeholder intentionally replaced by measured_child_from_pair.
        complete_token_ns=parent.complete_token_ns,
        capability=dict(parent.capability),
        silent_fallback_ids=tuple(parent.silent_fallback_ids),
        benchmark_fingerprint=parent.benchmark_fingerprint,
        identity=child_identity,
        lane="agentos_candidate_builder",
        valid=True,
    )
    payload = parent_research_payload(
        next_bottleneck=next_bottleneck,
        task_graph=[
            {
                "id": "candidate-protected-pair",
                "parent": parent.instance_id,
                "candidate": child.instance_id,
            }
        ],
        memories=[f"candidate staged from {worker_repo}"],
        genomes={
            "parent_runtime_sha": parent.runtime_sha,
            "candidate_runtime_sha": child.runtime_sha,
            "parent_kernel_sha": parent.kernel_genome_sha,
            "candidate_kernel_sha": child.kernel_genome_sha,
        },
        active_hypotheses=[
            {
                "id": f"candidate-{child.instance_id}",
                "text": f"candidate must beat CURRENT on {next_bottleneck.strip()}",
            }
        ],
        open_experiments=[
            {"id": "protected-parent-child-pair", "status": "requested"}
        ],
    )
    request = {
        "schema": CANDIDATE_SCHEMA,
        "candidate": child.to_dict(),
        "artifact_root": str(candidate_artifact),
        "tokenizer": str(selected_tokenizer),
        "resident_executable": str(selected_resident),
        "benchmark_runtime": str(selected_benchmark),
        "kernel_source": str(selected_kernel),
        "parent_payload": payload,
        "world_state": {
            "candidate_draft": {
                "prepared_by": "agentos_hcli",
                "origin_repo": str(worker_repo),
                "parent_instance_id": parent.instance_id,
            },
            **dict(world_state or {}),
        },
    }
    # Keep staging honest: it may create no request the controller would later
    # reject merely because required files or their hashes were malformed.
    CandidateSpec.from_mapping(request, repo=worker_repo)
    return request


class CandidateInbox:
    """Durable handoff queue between AgentOS work and the external gate.

    A resident worker may construct a candidate request, but it may not invoke
    promotion.  Submission validates and snapshots that request into an inbox.
    A separate controller process atomically claims one item, performs the
    protected A/B and lifecycle handoff, then preserves the original request in
    an outcome directory.  Nothing is silently discarded or retried after a
    controller crash.
    """

    _OUTCOMES = frozenset({"promoted", "not-promoted", "failed"})

    def __init__(self, root: Path = DEFAULT_CANDIDATE_ROOT) -> None:
        self.root = Path(root)

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def active(self) -> Path:
        return self.root / "active"

    @property
    def submissions(self) -> Path:
        """Controller-owned submission provenance, separate from model bytes."""
        return self.root / "submissions"

    def _outcome_dir(self, outcome: str) -> Path:
        if outcome not in self._OUTCOMES:
            raise LifecycleError(f"unknown candidate inbox outcome {outcome!r}")
        return self.root / outcome

    @staticmethod
    def _safe_name(instance_id: str, request_sha: str) -> str:
        safe_id = "".join(ch if ch.isalnum() or ch in ".-_" else "-" for ch in instance_id)
        return f"{safe_id}-{request_sha[:16]}.json"

    def _submission_path(self, request_name: str) -> Path:
        if not request_name.endswith(".json"):
            raise LifecycleError(f"invalid candidate request name {request_name!r}")
        return self.submissions / f"{request_name}.submission.json"

    def _submission_record(
        self,
        *,
        request_sha: str,
        request_path: Path,
        repo: Path,
        instance_id: str,
    ) -> dict[str, Any]:
        return seal(
            {
                "schema": CANDIDATE_SUBMISSION_SCHEMA,
                "submitted_at": utc_now(),
                "request_sha256": request_sha,
                "request_path": str(request_path.resolve()),
                # Relative paths in a candidate request are interpreted against
                # this independently recorded origin, not against whichever
                # controller process picks it up later.
                "origin_repo": str(repo.resolve()),
                "instance_id": instance_id,
            }
        )

    def _existing_submission_matches(self, path: Path, *, request_sha: str) -> bool:
        if not path.is_file():
            return False
        try:
            record, _ = _load_json_object(path, "candidate submission")
            verify(record, label="candidate submission")
        except (LifecycleError, ValueError):
            return False
        return (
            record.get("schema") == CANDIDATE_SUBMISSION_SCHEMA
            and record.get("request_sha256") == request_sha
        )

    def submit(self, request_path: Path, *, repo: Path = REPO) -> Path:
        """Validate and atomically snapshot a candidate-authored request."""
        request, raw = _load_json_object(Path(request_path), "candidate request")
        origin_repo = Path(repo).resolve()
        spec = CandidateSpec.from_mapping(request, repo=origin_repo)
        request_sha = hashlib.sha256(raw).hexdigest()
        target = self.inbox / self._safe_name(spec.instance.instance_id, request_sha)
        submission = self._submission_path(target.name)
        if target.exists():
            try:
                if hashlib.sha256(target.read_bytes()).hexdigest() == request_sha:
                    if not submission.exists():
                        _atomic_write_json(
                            submission,
                            self._submission_record(
                                request_sha=request_sha,
                                request_path=Path(request_path),
                                repo=origin_repo,
                                instance_id=spec.instance.instance_id,
                            ),
                            expected_previous_sha256=None,
                        )
                    elif not self._existing_submission_matches(submission, request_sha=request_sha):
                        raise LifecycleError(f"candidate submission provenance is invalid: {submission}")
                    return target
            except OSError as exc:
                raise LifecycleError(f"cannot inspect existing inbox request {target}: {exc}") from exc
            raise LifecycleError(f"candidate inbox name collision at {target}")
        # Record origin before publishing the raw request. A controller can
        # therefore never claim a valid relative-path request without the
        # worktree from which it was independently validated.
        if submission.exists():
            if not self._existing_submission_matches(submission, request_sha=request_sha):
                raise LifecycleError(f"candidate submission provenance collision at {submission}")
        else:
            _atomic_write_json(
                submission,
                self._submission_record(
                    request_sha=request_sha,
                    request_path=Path(request_path),
                    repo=origin_repo,
                    instance_id=spec.instance.instance_id,
                ),
                expected_previous_sha256=None,
            )
        _atomic_write_bytes(target, raw, expected_previous_sha256=None)
        return target

    def origin_repo_for(self, active_path: Path, *, fallback_repo: Path = REPO) -> Path:
        """Resolve the sealed source worktree for an active candidate request.

        Older/manual inbox entries have no sidecar and retain the historical
        controller-root interpretation. New HCLI submissions always carry this
        provenance, which allows their request paths to stay relative to an
        isolated worker worktree.
        """
        active_root = self.active.resolve()
        try:
            active = Path(active_path).resolve(strict=True)
        except OSError as exc:
            raise LifecycleError(f"cannot resolve active candidate request {active_path}: {exc}") from exc
        if active.parent != active_root or active.suffix != ".json":
            raise LifecycleError("candidate origin requested outside this inbox's active directory")
        sidecar = self._submission_path(active.name)
        if not sidecar.exists():
            return Path(fallback_repo).resolve()
        record, _ = _load_json_object(sidecar, "candidate submission")
        try:
            verify(record, label="candidate submission")
        except ValueError as exc:
            raise LifecycleError(f"candidate submission seal failed: {exc}") from exc
        if record.get("schema") != CANDIDATE_SUBMISSION_SCHEMA:
            raise LifecycleError("candidate submission has an unexpected schema")
        actual_sha = file_sha256(active)
        if record.get("request_sha256") != actual_sha:
            raise LifecycleError("candidate submission provenance does not match active request bytes")
        origin = _directory(record.get("origin_repo"), repo=Path(fallback_repo).resolve(), label="candidate origin repo")
        return origin

    def claim_next(self) -> Path | None:
        """Atomically move the oldest queued request into controller ownership."""
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.active.mkdir(parents=True, exist_ok=True)
        for queued in sorted(self.inbox.glob("*.json")):
            try:
                mode = queued.stat().st_mode
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise LifecycleError(f"cannot stat candidate inbox entry {queued}: {exc}") from exc
            if not stat.S_ISREG(mode) or queued.is_symlink():
                raise LifecycleError(f"candidate inbox entry is not a regular request file: {queued}")
            active = self.active / queued.name
            if active.exists():
                # A same-name active request is a durable indication that a
                # previous controller owns it. Do not overwrite its evidence.
                continue
            try:
                os.replace(queued, active)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise LifecycleError(f"cannot claim candidate inbox entry {queued}: {exc}") from exc
            return active
        return None

    def complete(
        self,
        active_path: Path,
        *,
        outcome: str,
        record: Mapping[str, Any],
    ) -> Path:
        """Archive a claimed request and a controller receipt/error together."""
        active_root = self.active.resolve()
        try:
            active = Path(active_path).resolve(strict=True)
        except OSError as exc:
            raise LifecycleError(f"cannot resolve active candidate request {active_path}: {exc}") from exc
        if active.parent != active_root or active.suffix != ".json":
            raise LifecycleError("refusing to archive a candidate outside this inbox's active directory")
        target_dir = self._outcome_dir(outcome)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / active.name
        if target.exists():
            raise LifecycleError(f"refusing to overwrite archived candidate request {target}")
        try:
            os.replace(active, target)
        except OSError as exc:
            raise LifecycleError(f"cannot archive candidate request {active}: {exc}") from exc
        sidecar = self._submission_path(active.name)
        if sidecar.exists():
            archived_sidecar = target.with_suffix(".submission.json")
            if archived_sidecar.exists():
                raise LifecycleError(
                    f"refusing to overwrite archived candidate submission provenance {archived_sidecar}"
                )
            try:
                os.replace(sidecar, archived_sidecar)
            except OSError as exc:
                raise LifecycleError(
                    f"cannot archive candidate submission provenance {sidecar}: {exc}"
                ) from exc
        _atomic_write_json(
            target.with_suffix(".lifecycle.json"),
            dict(record),
            expected_previous_sha256=None,
        )
        return target

    def status(self) -> dict[str, list[str]]:
        """Return durable queue state without interpreting a candidate as good."""
        result: dict[str, list[str]] = {}
        for name, path in (("inbox", self.inbox), ("active", self.active)):
            result[name] = sorted(item.name for item in path.glob("*.json")) if path.is_dir() else []
        return result


def _current_tokenizer(current: GenesisInstance, *, repo: Path) -> Path:
    """Resolve the parent's tokenizer independently of a proposed child.

    A tokenizer change is part of a candidate's behavior. Pairing the parent
    through the candidate tokenizer would invalidate the comparison, so prefer
    the CURRENT identity and otherwise use the seated resident discovery path.
    """
    identity = current.identity or {}
    recorded = identity.get("tokenizer")
    if recorded is not None:
        return _regular_file(recorded, repo=repo, label="CURRENT tokenizer")
    try:
        from tools.agentos.genesis_resident import discover_tokenizer

        discovered = discover_tokenizer(repo)
    except Exception as exc:
        raise LifecycleError(f"cannot discover CURRENT tokenizer: {exc}") from exc
    if discovered is None:
        raise LifecycleError("CURRENT tokenizer is unavailable")
    return _regular_file(discovered, repo=repo, label="CURRENT tokenizer")


def _current_profile(current: GenesisInstance, *, repo: Path, tokenizer: Path) -> BenchmarkProfile:
    identity = current.identity or {}
    artifact = _directory(identity.get("artifact"), repo=repo, label="CURRENT artifact")
    if file_sha256(artifact / "manifest.json") != current.artifact_sha:
        raise LifecycleError("CURRENT artifact SHA does not match its manifest")
    verification_path = _regular_file(
        identity.get("protected_verification"),
        repo=repo,
        label="CURRENT protected verification",
    )
    verification, _ = _load_json_object(verification_path, "CURRENT protected verification")
    binding = verification.get("protected_binding")
    if not isinstance(binding, Mapping):
        raise LifecycleError("CURRENT protected verification lacks protected_binding")
    runtime = _regular_file(
        binding.get("runtime_executable_path"),
        repo=repo,
        label="CURRENT benchmark runtime",
        executable=True,
    )
    kernel = _regular_file(
        binding.get("kernel_source_path"), repo=repo, label="CURRENT kernel source"
    )
    if binding.get("artifact_manifest_sha256") != current.artifact_sha:
        raise LifecycleError("CURRENT protected verification artifact hash disagrees with lineage")
    try:
        benchmark_sha = require_sha256(
            binding.get("runtime_executable_sha256"), "CURRENT benchmark runtime SHA"
        )
        kernel_sha = require_sha256(binding.get("kernel_source_sha256"), "CURRENT kernel SHA")
    except ValueError as exc:
        raise LifecycleError(str(exc)) from exc
    if file_sha256(runtime) != benchmark_sha or file_sha256(kernel) != kernel_sha:
        raise LifecycleError("CURRENT protected benchmark file changed since verification")
    return BenchmarkProfile(
        instance=current,
        artifact_root=artifact,
        tokenizer=tokenizer,
        benchmark_runtime=runtime,
        kernel_source=kernel,
    )


def _run_command(
    argv: Sequence[str],
    *,
    timeout_s: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(argv),
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LifecycleError(f"protected command failed to start: {exc}") from exc
    if result.returncode != 0:
        raise LifecycleError(
            "protected command failed: "
            f"exit={result.returncode} argv={list(argv)!r} stderr={result.stderr[-2000:]}"
        )
    return result


def _load_prompt_probe(path: Path, *, prompts: Sequence[str], label: str) -> list[dict[str, Any]]:
    raw, _ = _load_json_object(path, f"{label} greedy probe")
    rows = raw.get("prompts")
    if not isinstance(rows, list) or len(rows) != len(prompts):
        raise LifecycleError(f"{label} greedy probe does not contain every protected prompt")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise LifecycleError(f"{label} greedy probe row is not an object")
        prompt = str(row.get("prompt") or "")
        ids = row.get("new_token_ids")
        fallbacks = row.get("fallbacks")
        if prompt not in prompts or prompt in seen:
            raise LifecycleError(f"{label} greedy probe has an unexpected/duplicate prompt {prompt!r}")
        if type(fallbacks) is not int or fallbacks != 0:
            raise LifecycleError(f"{label} greedy probe recorded fallbacks={fallbacks!r}")
        if not isinstance(ids, list) or not ids or any(type(item) is not int for item in ids):
            raise LifecycleError(f"{label} greedy probe has invalid token ids for {prompt!r}")
        seen.add(prompt)
        normalized.append({"prompt": prompt, "ids": list(ids)})
    if set(prompts) != seen:
        raise LifecycleError(f"{label} greedy probe omitted a protected prompt")
    return normalized


def run_pair_unlocked(
    *,
    parent: BenchmarkProfile,
    child: BenchmarkProfile,
    out_dir: Path,
    pairs: int = 3,
    prompts: Sequence[str] = DEFAULT_GREEDY_PROMPTS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run protected A/B artifacts while the caller owns the GPU lease.

    Each A/B pair is a complete independent benchmark capture.  The derived
    parent and child headlines therefore form the three paired reps required by
    the promotion gate; raw captures remain on disk for a later verifier.
    """
    if pairs < 3:
        raise LifecycleError("protected parent/candidate comparison requires at least 3 pairs")
    prompts = tuple(str(prompt) for prompt in prompts)
    if len(prompts) < 3 or len(set(prompts)) != len(prompts) or any(not p.strip() for p in prompts):
        raise LifecycleError("protected greedy probe requires at least 3 distinct non-empty prompts")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    prompt_file = root / "protected-greedy-prompts.txt"
    prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    def capture(profile: BenchmarkProfile, *, arm: str, pair_index: int) -> dict[str, Any]:
        path = root / f"{pair_index:02d}-{arm}-complete-wall.json"
        argv = [
            str(profile.benchmark_runtime),
            "--artifact-root",
            str(profile.artifact_root),
            "--tokenizer",
            str(profile.tokenizer),
            "--complete-wall",
            "--pairs",
            "3",
            "--max-new-tokens",
            "32",
            "--max-seq-len",
            "8192",
            "--out",
            str(path),
        ]
        _run_command(argv, timeout_s=7200.0, runner=runner)
        if not path.is_file():
            raise LifecycleError(f"{arm} benchmark exited without its capture {path}")
        verified = verify_qwen38_capture(
            capture_path=path,
            artifact_root=profile.artifact_root,
            runtime_executable=profile.benchmark_runtime,
            kernel_source=profile.kernel_source,
        )
        return {
            "capture_path": str(path),
            "capture_sha256": file_sha256(path),
            "verification": verified,
            "complete_token_ns": verified["measurement"][
                "derived_headline_complete_wall_ns_per_token"
            ],
        }

    paired: list[dict[str, Any]] = []
    for index in range(1, pairs + 1):
        paired.append(
            {
                "pair": index,
                "parent": capture(parent, arm="parent", pair_index=index),
                "child": capture(child, arm="child", pair_index=index),
            }
        )

    def probe(profile: BenchmarkProfile, *, arm: str) -> list[dict[str, Any]]:
        path = root / f"{arm}-greedy-probe.json"
        argv = [
            str(profile.benchmark_runtime),
            "--artifact-root",
            str(profile.artifact_root),
            "--tokenizer",
            str(profile.tokenizer),
            "--prompts-file",
            str(prompt_file),
            "--max-new-tokens",
            "16",
            "--max-seq-len",
            "8192",
            "--out",
            str(path),
        ]
        _run_command(argv, timeout_s=7200.0, runner=runner)
        return _load_prompt_probe(path, prompts=prompts, label=arm)

    parent_probe = {row["prompt"]: row["ids"] for row in probe(parent, arm="parent")}
    child_probe = {row["prompt"]: row["ids"] for row in probe(child, arm="child")}
    greedy_rows = [
        {
            "prompt": prompt,
            "parent_ids": parent_probe[prompt],
            "child_ids": child_probe[prompt],
        }
        for prompt in prompts
    ]
    coherence = all(row["parent_ids"] == row["child_ids"] for row in greedy_rows)
    if not coherence:
        raise LifecycleError("candidate greedy token ids diverged from parent on protected prompts")
    parent_reps = [int(row["parent"]["complete_token_ns"]) for row in paired]
    child_reps = [int(row["child"]["complete_token_ns"]) for row in paired]
    document = {
        "schema": PAIR_RECEIPT_SCHEMA,
        "status": "PASS",
        "recorded_at": utc_now(),
        "timing_authority": GPU_TIMING_AUTHORITY,
        "parent": parent.to_dict(),
        "child": child.to_dict(),
        "pair_count": pairs,
        "paired_order": ["parent", "child"] * pairs,
        "pairs": paired,
        "parent_complete_token_ns_reps": parent_reps,
        "child_complete_token_ns_reps": child_reps,
        "greedy_token_ids": greedy_rows,
        "protected_tests": [
            {"name": "coherence_greedy_ids", "status": "PASS"},
            {"name": "complete_token_ledger_closed", "status": "PASS"},
            {"name": "no_silent_fallback", "status": "PASS"},
        ],
        "capture_origin_attested": True,
        "gpu_lease_owned_by_caller": True,
    }
    receipt = seal(document)
    _atomic_write_json(root / "PROTECTED_PAIR_RECEIPT.json", receipt, expected_previous_sha256=None)
    return receipt


def run_protected_pair(
    *,
    parent: BenchmarkProfile,
    child: BenchmarkProfile,
    out_dir: Path,
    gpu_lock: Path = DEFAULT_GPU_LOCK,
    owner: str = "genesis-protected-pair",
) -> dict[str, Any]:
    """Run :func:`run_pair_unlocked` below one external GPU lease process.

    The public wrapper intentionally launches a fresh interpreter under the
    existing lock script.  Holding the lease around all six captures prevents a
    queue race between adjacent parent/candidate arms.
    """
    request = {
        "schema": PAIR_REQUEST_SCHEMA,
        "parent": parent.to_dict(),
        "child": child.to_dict(),
        "out_dir": str(Path(out_dir)),
        "pairs": 3,
        "prompts": list(DEFAULT_GREEDY_PROMPTS),
    }
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "protected-pair-request.json"
    _atomic_write_json(request_path, request, expected_previous_sha256=None)
    if not gpu_lock.is_file() or not os.access(gpu_lock, os.X_OK):
        raise LifecycleError(f"protected GPU lock is unavailable: {gpu_lock}")
    argv = [
        str(gpu_lock),
        owner,
        sys.executable,
        str(REPO / "tools" / "genesis_lifecycle.py"),
        "benchmark-pair",
        "--request",
        str(request_path),
    ]
    _run_command(argv, timeout_s=43_200.0)
    receipt_path = root / "PROTECTED_PAIR_RECEIPT.json"
    receipt, _ = _load_json_object(receipt_path, "protected pair receipt")
    try:
        verify(receipt, label="protected pair receipt")
    except ValueError as exc:
        raise LifecycleError(f"protected pair receipt seal failed: {exc}") from exc
    if receipt.get("schema") != PAIR_RECEIPT_SCHEMA or receipt.get("status") != "PASS":
        raise LifecycleError("protected pair runner did not produce a PASS receipt")
    return receipt


def _profile_from_request(raw: Mapping[str, Any], *, repo: Path) -> BenchmarkProfile:
    instance_raw = {
        "instance_id": raw.get("instance_id"),
        "generation": raw.get("generation"),
        "artifact_sha": raw.get("artifact_sha"),
        # Benchmark profiles are only used here.  The independently checked
        # runtime/kernel values below are the real authority for a capture.
        "runtime_sha": raw.get("benchmark_runtime_sha256"),
        "kernel_genome_sha": raw.get("kernel_source_sha256"),
        "representation_bpw": 1.0,
        "physical_bpw": 1.0,
        "complete_token_ns": 1,
        "capability": {"protected_benchmark": 1.0},
        "identity": {"model": "PocketAiHub/Qwen3.8-27B-Abliterated-MLX"},
    }
    try:
        instance = GenesisInstance.from_mapping(instance_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleError(f"invalid protected-pair profile: {exc}") from exc
    artifact = _directory(raw.get("artifact_root"), repo=repo, label="pair artifact")
    tokenizer = _regular_file(raw.get("tokenizer"), repo=repo, label="pair tokenizer")
    runtime = _regular_file(
        raw.get("benchmark_runtime"), repo=repo, label="pair benchmark runtime", executable=True
    )
    kernel = _regular_file(raw.get("kernel_source"), repo=repo, label="pair kernel source")
    if file_sha256(artifact / "manifest.json") != instance.artifact_sha:
        raise LifecycleError("pair artifact manifest changed after request creation")
    if file_sha256(runtime) != instance.runtime_sha or file_sha256(kernel) != instance.kernel_genome_sha:
        raise LifecycleError("pair runtime/kernel changed after request creation")
    return BenchmarkProfile(instance, artifact, tokenizer, runtime, kernel)


def benchmark_pair_command(request_path: Path) -> int:
    """CLI target run inside ``gpu_lane_lock.sh``; never promotes anything."""
    request, _ = _load_json_object(request_path, "protected pair request")
    if request.get("schema") != PAIR_REQUEST_SCHEMA:
        raise LifecycleError("unexpected protected pair request schema")
    parent_raw = request.get("parent")
    child_raw = request.get("child")
    if not isinstance(parent_raw, Mapping) or not isinstance(child_raw, Mapping):
        raise LifecycleError("protected pair request needs parent and child profiles")
    parent = _profile_from_request(parent_raw, repo=REPO)
    child = _profile_from_request(child_raw, repo=REPO)
    out_dir = _directory(request.get("out_dir"), repo=REPO, label="protected pair output")
    pairs = request.get("pairs", 3)
    if type(pairs) is not int:
        raise LifecycleError("protected pair request.pairs must be an integer")
    prompts = request.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise LifecycleError("protected pair request.prompts must be a string list")
    run_pair_unlocked(parent=parent, child=child, out_dir=out_dir, pairs=pairs, prompts=prompts)
    return 0


def _upper_median(values: Sequence[int], *, label: str) -> int:
    if len(values) < 3 or any(type(value) is not int or value <= 0 for value in values):
        raise LifecycleError(f"{label} must contain at least three positive integer values")
    return sorted(values)[len(values) // 2]


def _pair_values(receipt: Mapping[str, Any], name: str) -> list[int]:
    values = receipt.get(name)
    if not isinstance(values, list):
        raise LifecycleError(f"protected pair receipt.{name} must be a list")
    return [int(value) if type(value) is int else 0 for value in values]


def _validate_pair_receipt(
    receipt: Mapping[str, Any],
    *,
    parent: BenchmarkProfile,
    child: BenchmarkProfile,
) -> tuple[list[int], list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        verify(receipt, label="protected pair receipt")
    except ValueError as exc:
        raise LifecycleError(f"protected pair receipt seal failed: {exc}") from exc
    if receipt.get("schema") != PAIR_RECEIPT_SCHEMA or receipt.get("status") != "PASS":
        raise LifecycleError("protected pair receipt is not a PASS receipt")
    if receipt.get("capture_origin_attested") is not True:
        raise LifecycleError("protected pair receipt did not attest capture origin")
    parent_block = receipt.get("parent")
    child_block = receipt.get("child")
    if not isinstance(parent_block, Mapping) or not isinstance(child_block, Mapping):
        raise LifecycleError("protected pair receipt lacks parent/child bindings")
    if (
        parent_block.get("artifact_sha") != parent.instance.artifact_sha
        or child_block.get("artifact_sha") != child.instance.artifact_sha
        or parent_block.get("benchmark_runtime_sha256") != file_sha256(parent.benchmark_runtime)
        or child_block.get("benchmark_runtime_sha256") != file_sha256(child.benchmark_runtime)
        or parent_block.get("kernel_source_sha256") != file_sha256(parent.kernel_source)
        or child_block.get("kernel_source_sha256") != file_sha256(child.kernel_source)
    ):
        raise LifecycleError("protected pair receipt identity does not match independently hashed arms")
    if receipt.get("timing_authority") != GPU_TIMING_AUTHORITY:
        raise LifecycleError("protected pair receipt used the wrong timing authority")
    parent_reps = _pair_values(receipt, "parent_complete_token_ns_reps")
    child_reps = _pair_values(receipt, "child_complete_token_ns_reps")
    _upper_median(parent_reps, label="parent complete-token reps")
    _upper_median(child_reps, label="child complete-token reps")
    greedy = receipt.get("greedy_token_ids")
    tests = receipt.get("protected_tests")
    if not isinstance(greedy, list) or not isinstance(tests, list):
        raise LifecycleError("protected pair receipt lacks greedy or protected-test evidence")
    return parent_reps, child_reps, [dict(row) for row in greedy if isinstance(row, Mapping)], [
        dict(row) for row in tests if isinstance(row, Mapping)
    ]


def _candidate_manifest_preimage(spec: CandidateSpec) -> str:
    try:
        # Promotion's generic gate hashes the supplied preimage.  Artifact
        # manifests are JSON and therefore must be UTF-8; preserving the exact
        # decoded bytes retains its independent SHA identity without an invented
        # serialized substitute.
        return (spec.artifact_root / "manifest.json").read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LifecycleError(f"cannot read candidate manifest preimage: {exc}") from exc


def promotion_evidence_from_pair(
    *,
    parent_profile: BenchmarkProfile,
    child: GenesisInstance,
    spec: CandidateSpec,
    pair_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build gate evidence only from controller-derived protected receipts."""
    parent = parent_profile.instance
    parent_reps, child_reps, greedy, tests = _validate_pair_receipt(
        pair_receipt,
        parent=parent_profile,
        child=spec.benchmark_profile,
    )
    return {
        "measurement": {
            "artifact_sha": child.artifact_sha,
            "complete_token_ns_reps": child_reps,
            "parent_complete_token_ns_reps": parent_reps,
            "regime": "warm",
            "timing_authority": GPU_TIMING_AUTHORITY,
            "parent_timing_authority": GPU_TIMING_AUTHORITY,
            "child_timing_authority": GPU_TIMING_AUTHORITY,
            "benchmark_fingerprint": child.benchmark_fingerprint,
            "paired": True,
            "alternating_reps": len(child_reps),
            "protected_pair_receipt": pair_receipt.get("seal_sha256"),
        },
        "representation": {
            "bpw": child.representation_bpw,
            "physical_bpw": child.physical_bpw,
            "receipt_ref": pair_receipt.get("seal_sha256"),
        },
        "genome": {
            "runtime_sha": child.runtime_sha,
            "kernel_genome_sha": child.kernel_genome_sha,
            "resident_executable_sha256": file_sha256(spec.resident_executable),
            "benchmark_runtime_sha256": file_sha256(spec.benchmark_runtime),
            "kernel_source_sha256": file_sha256(spec.kernel_source),
            "receipt_ref": pair_receipt.get("seal_sha256"),
        },
        "artifact_receipt": {
            "sha": child.artifact_sha,
            "preimage": _candidate_manifest_preimage(spec),
        },
        "protected_tests": tests,
        "greedy_token_ids": greedy,
        "state_transfer": {
            "payload": dict(spec.parent_payload),
            "checksum_sha256": payload_checksum(spec.parent_payload),
            "checksum_verified": True,
        },
        "rollback_artifact": {"valid": True, "instance_id": parent.instance_id},
        "parent_contract": {
            "capability": dict(parent.capability),
            "benchmark_fingerprint": parent.benchmark_fingerprint,
            "silent_fallback_ids": list(parent.silent_fallback_ids),
            "complete_token_ns": parent.complete_token_ns,
            "representation_bpw": parent.representation_bpw,
            "tps": parent.tps,
        },
        "new_silent_fallbacks": [],
    }


def measured_child_from_pair(
    *,
    parent_profile: BenchmarkProfile,
    child: GenesisInstance,
    spec: CandidateSpec,
    pair_receipt: Mapping[str, Any],
) -> GenesisInstance:
    """Replace an untrusted candidate wall estimate with protected truth.

    A worker cannot legally run the protected GPU comparison itself, so it
    cannot know the exact paired median in advance. Requiring it to guess that
    integer made valid new children impossible to submit. The declared value
    remains auditable in identity, while only the controller-derived upper
    median is allowed into the child identity or promotion gate.
    """
    _parent_reps, child_reps, _greedy, _tests = _validate_pair_receipt(
        pair_receipt,
        parent=parent_profile,
        child=spec.benchmark_profile,
    )
    measured = child.copy()
    declared = measured.complete_token_ns
    measured.complete_token_ns = _upper_median(child_reps, label="candidate complete-token reps")
    measured.identity.update(
        {
            "candidate_declared_complete_token_ns": str(declared),
            "complete_token_ns_authority": "protected_pair_upper_median",
            "protected_pair_receipt": str(pair_receipt.get("seal_sha256") or ""),
        }
    )
    return measured


HealthFn = Callable[[], Mapping[str, Any] | None]
ActivateFn = Callable[[CandidateSpec], Mapping[str, Any] | None]
RestartFn = Callable[[CandidateSpec], Mapping[str, Any] | None]
StopFn = Callable[[], bool]
BenchmarkFn = Callable[[BenchmarkProfile, BenchmarkProfile, Path], Mapping[str, Any]]


def _default_health(repo: Path) -> Mapping[str, Any] | None:
    from tools.agentos import genesis_resident

    return genesis_resident.health(genesis_resident.default_socket(repo), timeout=2.0)


def _default_activate(repo: Path, spec: CandidateSpec) -> Mapping[str, Any] | None:
    from tools.agentos import genesis_resident

    return genesis_resident.request_reload(
        genesis_resident.default_socket(repo),
        artifact=spec.artifact_root,
        generation=spec.instance.generation,
        artifact_sha=spec.instance.artifact_sha,
    )


def _default_stop(repo: Path) -> bool:
    """Stop only the health-attested resident currently bound to this repo.

    Socket shutdown is the normal path.  If a newly-started candidate answers
    health but has stopped servicing control requests, it is still a verified
    Genesis resident process and may be terminated so the supervisor can boot
    the durable CURRENT.  We never signal an unobserved PID.
    """
    from tools.agentos import genesis_resident

    socket_path = genesis_resident.default_socket(repo)
    if genesis_resident.request_stop(socket_path):
        return True
    observed = genesis_resident.health(socket_path, timeout=2.0)
    if not isinstance(observed, Mapping) or observed.get("body_resident") is not True:
        return False
    try:
        pid = int(observed.get("pid"))
    except (TypeError, ValueError):
        return False
    if not genesis_resident.process_alive(pid):
        return True
    try:
        os.kill(pid, 15)
    except OSError:
        return False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not genesis_resident.process_alive(pid):
            return True
        time.sleep(0.1)
    return not genesis_resident.process_alive(pid)


def _default_restart(repo: Path, spec: CandidateSpec) -> Mapping[str, Any] | None:
    """Ask the managed supervisor to replace the resident executable.

    The caller has already moved durable CURRENT to a child with
    ``live=False``.  ``genesis_forever.sh`` detects this proven stop, resolves
    the new CURRENT identity, and starts exactly that executable.  We wait for
    a matching health observation rather than treating a successful stop as a
    successful promotion.
    """
    from tools.agentos import genesis_resident

    if not _default_stop(repo):
        return None
    raw_timeout = os.environ.get("GENESIS_RUNTIME_RESTART_TIMEOUT_S", "900")
    try:
        timeout_s = float(raw_timeout)
    except ValueError:
        timeout_s = 900.0
    timeout_s = min(max(timeout_s, 30.0), 3_600.0)
    socket_path = genesis_resident.default_socket(repo)
    deadline = time.monotonic() + timeout_s
    last: Mapping[str, Any] | None = None
    while True:
        observed = genesis_resident.health(socket_path, timeout=2.0)
        if _health_matches(observed, spec.instance):
            return observed
        last = observed
        if time.monotonic() >= deadline:
            return last
        time.sleep(0.25)


def _default_benchmark(parent: BenchmarkProfile, child: BenchmarkProfile, out_dir: Path) -> Mapping[str, Any]:
    return run_protected_pair(parent=parent, child=child, out_dir=out_dir)


def _health_matches(health: Mapping[str, Any] | None, instance: GenesisInstance) -> bool:
    return bool(
        isinstance(health, Mapping)
        and health.get("ok") is True
        and health.get("body_resident") is True
        and health.get("artifact_sha") == instance.artifact_sha
        and health.get("generation") == instance.generation
        and not health.get("reload_error")
    )


def _directive_bindings() -> list[dict[str, Any]]:
    from tools.agentos.genesis_contract import contract_provenance

    binding = contract_provenance()
    contracts = binding.get("contracts")
    if not isinstance(contracts, list):
        raise LifecycleError("canonical Genesis contract provenance is malformed")
    return [dict(row) for row in contracts if isinstance(row, Mapping)]


def _result_path(root: Path, *, candidate: GenesisInstance, request_sha: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in candidate.instance_id)
    return root / f"{safe}-{candidate.artifact_sha[:12]}-{request_sha[:12]}" / "LIFECYCLE_RECEIPT.json"


class PromotionController:
    """Perform a protected live handoff; the candidate never holds authority.

    Artifact-only successors reload inside the already-live resident executable.
    A runtime/kernel successor uses a two-phase managed-exec handoff: durable
    workers rebind while the parent remains healthy; CURRENT then names the
    child *without claiming it is live*; the supervisor restarts against that
    identity; only an observed child health response retires the parent.
    """

    def __init__(
        self,
        *,
        repo: Path = REPO,
        state_path: Path = DEFAULT_LINEAGE_PATH,
        worker_registry: WorkerRegistry | None = None,
        checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
        candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
        health: HealthFn | None = None,
        activate: ActivateFn | None = None,
        restart: RestartFn | None = None,
        stop: StopFn | None = None,
        benchmark: BenchmarkFn | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.state_path = Path(state_path)
        self.registry = worker_registry or WorkerRegistry()
        self.checkpoint_root = Path(checkpoint_root)
        self.candidate_root = Path(candidate_root)
        self._health = health or (lambda: _default_health(self.repo))
        self._activate = activate or (lambda spec: _default_activate(self.repo, spec))
        self._restart = restart or (lambda spec: _default_restart(self.repo, spec))
        self._stop = stop or (lambda: _default_stop(self.repo))
        self._benchmark = benchmark or _default_benchmark

    def bootstrap_workers(self) -> list[dict[str, Any]]:
        state_raw, _ = _load_json_object(self.state_path, "lineage state")
        lineage = LineageState.from_dict(state_raw)
        current = lineage.current
        if current is None:
            raise LifecycleError("cannot bootstrap workers without CURRENT Genesis")
        return self.registry.bootstrap(
            generation=generation_record(current, repo_head=_git_head(self.repo)), repo=self.repo
        )

    def _write_result(
        self,
        *,
        candidate: GenesisInstance,
        request_sha: str,
        result: Mapping[str, Any],
    ) -> Path:
        path = _result_path(self.candidate_root, candidate=candidate, request_sha=request_sha)
        _atomic_write_json(path, seal(dict(result)), expected_previous_sha256=None)
        return path

    def _rollback_nominated(
        self,
        *,
        lineage: LineageState,
        expected_state_sha: str,
        reason: str,
    ) -> None:
        lineage.rollback(reason=reason)
        _atomic_write_json(
            self.state_path,
            lineage.to_dict(),
            expected_previous_sha256=expected_state_sha,
        )

    def _parent_reload_spec(
        self,
        *,
        parent: GenesisInstance,
        parent_profile: BenchmarkProfile,
        payload: Mapping[str, Any],
    ) -> CandidateSpec:
        """Build a verified reload target for the still-authoritative parent."""
        executable = _regular_file(
            parent.identity.get("resident_executable"),
            repo=self.repo,
            label="CURRENT resident executable",
            executable=True,
        )
        if file_sha256(executable) != parent.runtime_sha:
            raise LifecycleError(
                "CURRENT resident executable changed since its lineage runtime binding"
            )
        return CandidateSpec(
            instance=parent.copy(),
            artifact_root=parent_profile.artifact_root,
            tokenizer=parent_profile.tokenizer,
            resident_executable=executable,
            benchmark_runtime=parent_profile.benchmark_runtime,
            kernel_source=parent_profile.kernel_source,
            parent_payload=dict(payload),
            world_state={},
        )

    def _rollback_runtime_handoff(
        self,
        *,
        lineage: LineageState,
        expected_state_sha: str,
        workers: Sequence[Mapping[str, Any]],
        rebound_registry_sha: str,
        reason: str,
    ) -> None:
        """Restore durable G0 first, then ensure a wrong child cannot linger.

        A runtime replacement needs CURRENT=child for the supervisor to choose
        the child executable.  If that executable fails activation, restoring
        the state before stopping it makes the supervisor's next retry select
        LAST_KNOWN_GOOD rather than repeatedly booting the failed image.
        """
        lineage.rollback(reason=reason)
        _atomic_write_json(
            self.state_path,
            lineage.to_dict(),
            expected_previous_sha256=expected_state_sha,
        )
        try:
            self.registry.replace(workers, expected_previous_sha256=rebound_registry_sha)
        except Exception as exc:
            raise LifecycleError(
                "runtime candidate rollback restored CURRENT but could not restore "
                "the durable worker registry"
            ) from exc
        if not self._stop():
            raise LifecycleError(
                "runtime candidate rollback restored CURRENT but could not stop "
                "the non-authoritative resident; supervisor reconciliation is required"
            )

    def promote(
        self,
        request_path: Path,
        *,
        candidate_repo: Path | None = None,
    ) -> dict[str, Any]:
        """Benchmark, rebind, activate, then atomically move lineage authority.

        No candidate is able to self-certify. A benchmark failure does not
        write a CANDIDATE as if it were qualified. For an executable-image
        successor, the parent stays live through all worker rebind work and
        the durable state marks the child activation-pending rather than live.
        """
        request, request_bytes = _load_json_object(Path(request_path), "candidate request")
        request_sha = hashlib.sha256(request_bytes).hexdigest()
        origin = Path(candidate_repo or self.repo).resolve()
        spec = CandidateSpec.from_mapping(request, repo=origin)
        child = spec.handoff_instance()
        state_raw, state_bytes = _load_json_object(self.state_path, "lineage state")
        state_sha = hashlib.sha256(state_bytes).hexdigest()
        lineage = LineageState.from_dict(state_raw)
        parent = lineage.current
        if parent is None or not parent.valid:
            raise LifecycleError("CURRENT Genesis is absent or invalid")
        if child.generation <= parent.generation:
            raise LifecycleError("candidate generation must strictly exceed CURRENT before benchmarking")
        runtime_replacement = child.runtime_sha != parent.runtime_sha
        if not runtime_replacement and child.kernel_genome_sha != parent.kernel_genome_sha:
            raise LifecycleError(
                "candidate changes the kernel genome without changing the resident executable; "
                "a kernel successor must be carried by a managed runtime replacement"
            )
        if not _health_matches(self._health(), parent):
            raise LifecycleError("CURRENT resident health does not match lineage; refusing promotion")
        parent_profile = _current_profile(
            parent,
            repo=self.repo,
            tokenizer=_current_tokenizer(parent, repo=self.repo),
        )

        run_dir = _result_path(
            self.candidate_root, candidate=child, request_sha=request_sha
        ).parent
        pair = self._benchmark(parent_profile, spec.benchmark_profile, run_dir)
        child = measured_child_from_pair(
            parent_profile=parent_profile,
            child=child,
            spec=spec,
            pair_receipt=pair,
        )
        evidence = promotion_evidence_from_pair(
            parent_profile=parent_profile,
            child=child,
            spec=spec,
            pair_receipt=pair,
        )
        verdict = evaluate_promotion(
            parent=parent,
            child=child,
            evidence=evidence,
            invoker=Invoker(
                principal="protected_controller",
                identity="genesis-lifecycle-controller",
                acting_as="external",
            ),
            lineage=lineage,
        )
        if verdict.get("verdict") != "ACCEPT":
            result = {
                "schema": LIFECYCLE_RECEIPT_SCHEMA,
                "outcome": verdict.get("verdict"),
                "authority_moved": False,
                "reason": verdict.get("reason"),
                "candidate_id": child.instance_id,
                "parent_id": parent.instance_id,
                "protected_pair_receipt": pair.get("seal_sha256"),
                "verdict": verdict,
                "recorded_at": utc_now(),
            }
            self._write_result(candidate=child, request_sha=request_sha, result=result)
            return seal(result)

        existing_candidate = lineage.candidate
        if existing_candidate is not None and existing_candidate.valid and (
            existing_candidate.instance_id != child.instance_id
        ):
            raise LifecycleError(
                f"another valid CANDIDATE ({existing_candidate.instance_id}) is already staged"
            )
        lineage.nominate(child)
        lineage.snapshot_current_as_lkg()
        _atomic_write_json(
            self.state_path,
            lineage.to_dict(),
            expected_previous_sha256=state_sha,
        )
        nominated_sha = hashlib.sha256(self.state_path.read_bytes()).hexdigest()

        workers, registry_sha = self.registry.load()
        if len(workers) < 2:
            self._rollback_nominated(
                lineage=lineage,
                expected_state_sha=nominated_sha,
                reason="promotion refused: fewer than two durable logical workers",
            )
            raise LifecycleError("promotion needs at least two registered logical workers")
        parent_generation = generation_record(parent, repo_head=_git_head(self.repo))
        child_generation = generation_record(child, repo_head=_git_head(self.repo))
        if not _health_matches(self._health(), parent):
            self._rollback_nominated(
                lineage=lineage,
                expected_state_sha=nominated_sha,
                reason="parent resident stopped before worker rebind",
            )
            raise LifecycleError("parent resident stopped before worker rebind")
        bus = ResearchBus()
        migration = migrate_workers(
            workers=workers,
            old_generation=parent_generation,
            new_generation=child_generation,
            store=WorkerCheckpointStore(self.checkpoint_root),
            directives=_directive_bindings(),
            world_state={
                "CURRENT": parent.to_dict(),
                "CANDIDATE": child.to_dict(),
                **spec.world_state,
            },
            bus=bus,
            parent_live_before=True,
            parent_live_after=_health_matches(self._health(), parent),
            protected_test_slot_available=True,
        )
        if migration.get("status") != "PASS":
            self._rollback_nominated(
                lineage=lineage,
                expected_state_sha=nominated_sha,
                reason="worker migration did not produce PASS",
            )
            raise LifecycleError("worker migration did not produce PASS")

        rebound_workers = [
            WorkerCheckpointStore(self.checkpoint_root).load(worker["worker_id"])["worker"]
            for worker in workers
        ]
        package = pack_state(parent, spec.parent_payload, to=child)
        invoker = Invoker(
            principal="protected_controller",
            identity="genesis-lifecycle-controller",
            acting_as="external",
        )

        if runtime_replacement:
            # Runtime replacement cannot reload a process image. Persist the
            # durable worker rebind while G0 still answers, then make the
            # child identity visible to the managed supervisor.
            try:
                self.registry.replace(rebound_workers, expected_previous_sha256=registry_sha)
            except Exception:
                self._rollback_nominated(
                    lineage=lineage,
                    expected_state_sha=nominated_sha,
                    reason="failed to persist rebound worker registry before runtime handoff",
                )
                raise
            rebound_sha = hashlib.sha256(self.registry.path.read_bytes()).hexdigest()
            handover = lineage.handover(
                package=package,
                invoker=invoker,
                verdict=verdict,
                retire_parent=False,
                successor_live=False,
            )
            try:
                _atomic_write_json(
                    self.state_path,
                    lineage.to_dict(),
                    expected_previous_sha256=nominated_sha,
                )
            except Exception:
                self.registry.replace(workers, expected_previous_sha256=rebound_sha)
                raise
            handover_sha = hashlib.sha256(self.state_path.read_bytes()).hexdigest()

            activated = self._restart(spec)
            if not _health_matches(activated, child):
                self._rollback_runtime_handoff(
                    lineage=lineage,
                    expected_state_sha=handover_sha,
                    workers=workers,
                    rebound_registry_sha=rebound_sha,
                    reason="managed runtime candidate did not return matching health",
                )
                raise LifecycleError(
                    "managed runtime candidate activation failed; CURRENT was restored from "
                    "LAST_KNOWN_GOOD"
                )

            observed_live = lineage.mark_current_live(instance_id=child.instance_id)
            _atomic_write_json(
                self.state_path,
                lineage.to_dict(),
                expected_previous_sha256=handover_sha,
            )
            live_sha = hashlib.sha256(self.state_path.read_bytes()).hexdigest()
            parent_retirement = lineage.finalize_parent_retirement()
            _atomic_write_json(
                self.state_path,
                lineage.to_dict(),
                expected_previous_sha256=live_sha,
            )
            result = {
                "schema": LIFECYCLE_RECEIPT_SCHEMA,
                "outcome": "PROMOTED",
                "authority_moved": True,
                "candidate_id": child.instance_id,
                "parent_id": parent.instance_id,
                "current_id": lineage.current.instance_id if lineage.current else None,
                "protected_pair_receipt": pair.get("seal_sha256"),
                "verdict": verdict,
                "activation_mode": "managed_exec_restart",
                "activation_health": dict(activated),
                "worker_migration": migration,
                "handover": handover,
                "successor_observed_live": observed_live,
                "parent_retirement": parent_retirement,
                "recorded_at": utc_now(),
            }
            self._write_result(candidate=child, request_sha=request_sha, result=result)
            return seal(result)

        # An artifact-only test subject already proved it can execute under
        # the current process image. Reload only after the parent survived the
        # full checkpoint/rebind operation.
        activated = self._activate(spec)
        if not _health_matches(activated, child):
            self._rollback_nominated(
                lineage=lineage,
                expected_state_sha=nominated_sha,
                reason="candidate resident activation failed before authority moved",
            )
            raise LifecycleError("candidate resident activation failed before authority moved")

        launch = lineage.launch_successor(lambda _candidate: True)
        if not launch.ok:
            self._activate(
                self._parent_reload_spec(
                    parent=parent,
                    parent_profile=parent_profile,
                    payload=spec.parent_payload,
                )
            )
            self._rollback_nominated(
                lineage=lineage,
                expected_state_sha=nominated_sha,
                reason=launch.reason,
            )
            raise LifecycleError(f"candidate launch state failed: {launch.reason}")
        _atomic_write_json(
            self.state_path,
            lineage.to_dict(),
            expected_previous_sha256=nominated_sha,
        )
        launched_sha = hashlib.sha256(self.state_path.read_bytes()).hexdigest()

        # Persist rebound worker state before authority moves. If this write
        # fails, restore the still-authoritative parent rather than presenting
        # a live child with an uncommitted AgentOS rebind.
        try:
            self.registry.replace(rebound_workers, expected_previous_sha256=registry_sha)
        except Exception:
            self._activate(
                self._parent_reload_spec(
                    parent=parent,
                    parent_profile=parent_profile,
                    payload=spec.parent_payload,
                )
            )
            self._rollback_nominated(
                lineage=lineage,
                expected_state_sha=launched_sha,
                reason="failed to persist rebound worker registry",
            )
            raise
        rebound_sha = hashlib.sha256(self.registry.path.read_bytes()).hexdigest()

        handover = lineage.handover(
            package=package,
            invoker=invoker,
            verdict=verdict,
        )
        try:
            _atomic_write_json(
                self.state_path,
                lineage.to_dict(),
                expected_previous_sha256=launched_sha,
            )
        except Exception:
            # A failed lineage CAS must not leave the durable registry pointing
            # at an unseated child.  Restore it before surfacing the error.
            self.registry.replace(workers, expected_previous_sha256=rebound_sha)
            self._activate(
                self._parent_reload_spec(
                    parent=parent,
                    parent_profile=parent_profile,
                    payload=spec.parent_payload,
                )
            )
            try:
                lineage.rollback(reason="failed to seal artifact-only handover")
                _atomic_write_json(
                    self.state_path,
                    lineage.to_dict(),
                    expected_previous_sha256=launched_sha,
                )
            except Exception:
                # The failed CAS may mean another controller legitimately won
                # the state race. Do not overwrite its authority blindly.
                pass
            raise

        result = {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "outcome": "PROMOTED",
            "authority_moved": True,
            "candidate_id": child.instance_id,
            "parent_id": parent.instance_id,
            "current_id": lineage.current.instance_id if lineage.current else None,
            "protected_pair_receipt": pair.get("seal_sha256"),
            "verdict": verdict,
            "launch": launch.to_dict(),
            "worker_migration": migration,
            "handover": handover,
            "recorded_at": utc_now(),
        }
        self._write_result(candidate=child, request_sha=request_sha, result=result)
        return seal(result)


def process_candidate_inbox_once(
    *,
    repo: Path = REPO,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    state_path: Path = DEFAULT_LINEAGE_PATH,
    worker_registry_path: Path = DEFAULT_WORKER_REGISTRY,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Claim and externally evaluate one AgentOS-submitted candidate request.

    This is intentionally a one-shot controller operation. The ascent daemon
    may launch it, but it never receives the state mutation logic itself. A
    crash leaves a request visibly in ``active/`` rather than pretending it was
    rejected or silently trying a partially activated child again.
    """
    root = Path(candidate_root)
    inbox = CandidateInbox(root)
    active = inbox.claim_next()
    if active is None:
        return {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "outcome": "IDLE",
            "authority_moved": False,
            "recorded_at": utc_now(),
        }
    controller = PromotionController(
        repo=Path(repo),
        state_path=Path(state_path),
        worker_registry=WorkerRegistry(Path(worker_registry_path)),
        checkpoint_root=Path(checkpoint_root),
        candidate_root=root,
    )
    origin_repo = Path(repo).resolve()
    try:
        origin_repo = inbox.origin_repo_for(active, fallback_repo=Path(repo))
        controller.bootstrap_workers()
        result = controller.promote(active, candidate_repo=origin_repo)
    except Exception as exc:  # The request must not disappear on a controller failure.
        record = seal(
            {
                "schema": LIFECYCLE_RECEIPT_SCHEMA,
                "outcome": "FAILED",
                "authority_moved": False,
                "active_request": str(active),
                "candidate_origin_repo": str(origin_repo),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at": utc_now(),
            }
        )
        archived = inbox.complete(active, outcome="failed", record=record)
        return {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "outcome": "FAILED",
            "authority_moved": False,
            "archived_request": str(archived),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "recorded_at": utc_now(),
        }

    promoted = result.get("outcome") == "PROMOTED" and result.get("authority_moved") is True
    record = seal(
        {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "outcome": "PROMOTED" if promoted else "NOT_PROMOTED",
            "authority_moved": bool(promoted),
            "active_request": str(active),
            "candidate_origin_repo": str(origin_repo),
            "controller_result": result,
            "recorded_at": utc_now(),
        }
    )
    archived = inbox.complete(
        active,
        outcome="promoted" if promoted else "not-promoted",
        record=record,
    )
    return {
        "schema": LIFECYCLE_RECEIPT_SCHEMA,
        "outcome": "PROMOTED" if promoted else "NOT_PROMOTED",
        "authority_moved": bool(promoted),
        "archived_request": str(archived),
        "controller_result": result,
        "recorded_at": utc_now(),
    }


__all__ = [
    "BenchmarkProfile",
    "CANDIDATE_SCHEMA",
    "CANDIDATE_SUBMISSION_SCHEMA",
    "CandidateInbox",
    "CandidateSpec",
    "DEFAULT_CANDIDATE_ROOT",
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_LINEAGE_PATH",
    "DEFAULT_WORKTREE_ROOT",
    "DEFAULT_WORKER_REGISTRY",
    "GPU_TIMING_AUTHORITY",
    "LifecycleBusy",
    "LifecycleError",
    "PAIR_RECEIPT_SCHEMA",
    "PAIR_REQUEST_SCHEMA",
    "PromotionController",
    "WorkerRegistry",
    "benchmark_pair_command",
    "build_candidate_request",
    "generation_record",
    "measured_child_from_pair",
    "promotion_evidence_from_pair",
    "process_candidate_inbox_once",
    "run_pair_unlocked",
    "run_protected_pair",
]
