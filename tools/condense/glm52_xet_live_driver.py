#!/usr/bin/env python3.12
"""Production authority and orchestration for the GLM-5.2 live Xet autotune.

This is the only public parent for :mod:`glm52_xet_live`.  The lower-level live
module deliberately exposes only a private child and one-trial primitives.  A
production run reaches those primitives only after this module has proved, in
order, that:

* the exact sealed AUTOTUNE_XET transition and Telegram Bot API receipt are the
  transition currently committed by the controller;
* this process owns the controller singleton lease and the checkpoint/log replay
  is current and healthy;
* the repository is clean and its exact HEAD is present at the configured
  upstream (the remote check happens only after authority is accepted);
* a fresh Darwin resource sample passes the frozen 16 GiB RAM, 8 GiB swap,
  zero-swap-growth, zero-new-swapout, disk, allocation, and thermal policy; and
* every per-operation capability was issued in this process and is accepted once
  by a verifier that rechecks the live lease, state, provenance, and Telegram
  binding immediately before the child may spawn.

The campaign order is fixed: all 12 planner trials, profile selection, then one
full largest-shard hash for each selected lane.  No result is fabricated or
substituted: the existing live validators assemble the raw result and the
campaign evidence key producer-attests it before it is published.

Importing this module performs no network access, starts no process, and reads no
model body.  Tests replace the one-trial executor and use only in-memory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
for entry in (HERE, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import glm52_gravity as gravity  # noqa: E402
import glm52_schedule_freeze as schedule_freeze  # noqa: E402
import glm52_state as state  # noqa: E402
import glm52_xet_autotune as autotune  # noqa: E402
import glm52_xet_live as live  # noqa: E402
from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    canonical,
    read_sealed_json,
    seal,
    utc_now,
    verify_sealed,
)


OFFICIAL_CAMPAIGN_ID = "glm52-bf16-xet-gravity"
OFFICIAL_BRANCH = "campaign/glm52-bf16-xet-gravity"
OFFICIAL_REMOTE = "origin"
OFFICIAL_REMOTE_URL_SHA256 = (
    "73d175407fb7f9bbebdcf067c39be2e8fc99567328cd7ccc20bbe171454a0164"
)
DRIVER_SCHEMA = "hawking.glm52.xet_live_driver_receipt.v1"
RUN_MARKER_SCHEMA = "hawking.glm52.xet_live_driver_run_marker.v1"
PROVENANCE_SCHEMA = "hawking.glm52.xet_live_git_provenance.v1"
LEASE_BINDING_SCHEMA = "hawking.glm52.xet_live_lease_binding.v1"
CAPABILITY_TTL_GRACE_SECONDS = 30.0
TRIAL_TIMEOUT_SECONDS = 900.0
LARGEST_TIMEOUT_SECONDS = 1800.0
SAMPLE_INTERVAL_SECONDS = 0.05

RUN_MARKER_NAME = "GLM52_XET_AUTOTUNE_RUN.json"
RAW_RESULT_NAME = "GLM52_XET_AUTOTUNE_RAW_RESULT.json"
ATTESTED_RESULT_NAME = "GLM52_XET_AUTOTUNE_RESULT.json"
DRIVER_RECEIPT_NAME = "GLM52_XET_AUTOTUNE_DRIVER_RECEIPT.json"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ARTIFACT_STEM = re.compile(r"^[A-Z0-9][A-Z0-9_-]{0,154}$")

PROVENANCE_PATHS = tuple(name for name, _schema, _statuses in autotune.INPUT_CONTRACTS) + (
    "GLM52_XET_AUTOTUNE_PLAN.json",
    "tools/glm52_gravity.py",
    "tools/condense/glm52_common.py",
    "tools/condense/requirements-glm52.txt",
    "tools/condense/glm52_schedule_freeze.py",
    "tools/condense/glm52_state.py",
    "tools/condense/glm52_xet_autotune.py",
    "tools/condense/glm52_xet_live.py",
    "tools/condense/glm52_xet_live_driver.py",
)


def _safe_artifact_name(name: Any) -> bool:
    """Accept one canonical ASCII leaf with an exact lowercase JSON suffix."""
    return isinstance(name, str) \
        and len(name) <= 160 \
        and name.endswith(".json") \
        and _SAFE_ARTIFACT_STEM.fullmatch(name[:-5]) is not None


class DriverError(Glm52Error):
    """A production live-Xet driver invariant failed closed."""


class GitRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one bounded git command without a shell."""


class ArtifactWriter(Protocol):
    def begin(self, binding: Mapping[str, Any]) -> None:
        """Durably claim this authority before the first live operation."""

    def write(self, name: str, value: Mapping[str, Any]) -> None:
        """Durably publish one new, never-overwritten artifact."""

    def finish(self, receipt: Mapping[str, Any]) -> None:
        """Durably seal the successful terminal marker."""


def _clone(value: Any) -> Any:
    return json.loads(canonical(value).decode("utf-8"))


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _require_official_runtime(
    runtime: gravity.Runtime,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = state._validate_expected_contract(_clone(contract))  # noqa: SLF001
    except (state.StateError, TypeError, ValueError) as exc:
        raise DriverError(f"expected campaign contract is invalid: {exc}") from exc
    if runtime.campaign_id != OFFICIAL_CAMPAIGN_ID \
            or runtime.source_revision != live.REVISION \
            or runtime.controller_epoch != autotune.CONTROLLER_EPOCH \
            or runtime.allow_synthetic_contract is not False:
        raise DriverError("live Xet driver requires the exact official production runtime")
    if value.get("campaign_id") != runtime.campaign_id \
            or value.get("source_revision") != runtime.source_revision \
            or value.get("seal_sha256") != runtime.expected_contract.get("seal_sha256") \
            or value != runtime.expected_contract:
        raise DriverError("runtime expected contract differs from the supplied sealed contract")
    if runtime.telegram_auth.expected_chat_identity_digest != \
            value.get("expected_chat_identity_digest"):
        raise DriverError("runtime Telegram identity differs from the expected contract")
    return value


def _minimal_plan_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = verify_sealed(_clone(plan), label="GLM-5.2 Xet autotune plan")
    except (Glm52Error, TypeError, ValueError) as exc:
        raise DriverError(str(exc)) from exc
    if value.get("schema") != autotune.PLAN_SCHEMA \
            or value.get("status") != "PASS_OFFLINE_PLAN_BODY_NOT_READ" \
            or value.get("repo") != live.REPO_ID \
            or value.get("revision") != live.REVISION:
        raise DriverError("Xet plan schema/status/source identity mismatch")
    return value


def _checkpoint_ref(
    checkpoint: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": autotune.COMMITTED_CHECKPOINT_REF_SCHEMA,
        "checkpoint_schema": state.CHECKPOINT_SCHEMA,
        "campaign_id": checkpoint.get("campaign_id"),
        "source_revision": checkpoint.get("source_revision"),
        "controller_epoch": checkpoint.get("controller_epoch"),
        "expected_contract_sha256": checkpoint.get("expected_contract_sha256"),
        "state": checkpoint.get("state"),
        "last_claim_id": checkpoint.get("last_claim_id"),
        "transition_intent_sha256": intent.get("seal_sha256"),
        "telegram_receipt_hmac_sha256": receipt.get("hmac_sha256"),
        "checkpoint_seal_sha256": checkpoint.get("seal_sha256"),
        "event_count": checkpoint.get("event_count"),
        "event_head_hash": checkpoint.get("event_head_hash"),
        "window_event_count": checkpoint.get("window_event_count"),
        "window_event_head_hash": checkpoint.get("window_event_head_hash"),
    }


def _lease_binding(controller: state.Controller) -> dict[str, Any]:
    controller.lease.assert_held()
    handle = controller.lease._handle  # noqa: SLF001 - descriptor binding is the authority
    if handle is None:
        raise DriverError("controller lease handle disappeared")
    try:
        descriptor = os.fstat(handle.fileno())
        named = os.lstat(controller.lease.path)
    except OSError as exc:
        raise DriverError(f"cannot bind the live controller lease: {exc}") from exc
    if not stat.S_ISREG(descriptor.st_mode) or int(descriptor.st_nlink) != 1 \
            or stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode) \
            or int(named.st_nlink) != 1 \
            or (int(descriptor.st_dev), int(descriptor.st_ino)) != \
            (int(named.st_dev), int(named.st_ino)):
        raise DriverError("controller lease descriptor/name identity is unsafe")
    observation = controller.lease.probe()
    if observation.get("held_by_this_handle") is not True \
            or observation.get("live_lock_held") is not True \
            or observation.get("owner_record_ok") is not True \
            or observation.get("owner_pid") != os.getpid() \
            or observation.get("owner_pid_alive") is not True \
            or observation.get("controller_epoch") != controller.controller_epoch:
        raise DriverError("controller singleton lease is not live and owned by this process")
    body = {
        "schema": LEASE_BINDING_SCHEMA,
        "campaign_id": controller.campaign_id,
        "controller_epoch": controller.controller_epoch,
        "lease_path": os.path.abspath(os.fspath(controller.lease.path)),
        "device": int(descriptor.st_dev),
        "inode": int(descriptor.st_ino),
        "owner": observation.get("owner"),
        "owner_pid": int(observation["owner_pid"]),
    }
    return {**body, "lease_identity_sha256": _sha(body)}


class ProductionExecutionAuthorityVerifier:
    """Concrete authority verifier backed by the held controller and live keys."""

    def __init__(self, runtime: gravity.Runtime, controller: state.Controller) -> None:
        self.runtime = runtime
        self.controller = controller

    def _current(
        self,
        *,
        intent: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint = self.controller.resume(recover_single_tail=False)
        events = self.controller.events.verified_events()
        if checkpoint.get("state") != "AUTOTUNE_XET" \
                or checkpoint.get("event_count") != len(events) \
                or not events:
            raise DriverError("controller is not at the exact committed AUTOTUNE_XET entry")
        event = events[-1]
        payload = event.get("payload")
        if event.get("kind") != "STATE_TRANSITION" \
                or event.get("seq") != len(events) - 1 \
                or event.get("chain_sha256") != checkpoint.get("event_head_hash") \
                or not isinstance(payload, Mapping) \
                or payload.get("to_state") != "AUTOTUNE_XET" \
                or payload.get("transition_intent") != intent \
                or payload.get("telegram_delivery") != receipt:
            raise DriverError("live controller event does not contain the exact Xet authority")
        return checkpoint, dict(event)

    def _trusted(self, plan_seal: str, contract_seal: str) -> bool:
        return plan_seal == self.runtime.expected_contract["state_gates"][
            "AUTOTUNE_XET"
        ]["required_artifacts"]["xet_autotune_plan"]["expected_seal_sha256"] \
            and contract_seal == self.runtime.expected_contract["seal_sha256"]

    def verify_prepared_transition_intent_hmac(
        self,
        transition_intent: Mapping[str, Any],
        *,
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        try:
            value = state.validate_transition_intent(
                transition_intent, self.runtime.telegram_auth
            )
        except (state.StateError, TypeError, ValueError):
            return False
        return self._trusted(plan_seal, expected_contract_sha256) \
            and value.get("campaign_id") == self.runtime.campaign_id

    def verify_telegram_delivery_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        transition_intent: Mapping[str, Any],
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        try:
            state.validate_telegram_delivery_receipt(
                receipt, transition_intent, self.runtime.telegram_auth
            )
            self._current(intent=transition_intent, receipt=receipt)
        except (DriverError, state.StateError, TypeError, ValueError):
            return False
        return self._trusted(plan_seal, expected_contract_sha256)

    def verify_committed_controller_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        transition_intent: Mapping[str, Any],
        telegram_receipt: Mapping[str, Any],
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        try:
            current, _event = self._current(
                intent=transition_intent, receipt=telegram_receipt
            )
        except (DriverError, state.StateError, TypeError, ValueError):
            return False
        return dict(checkpoint) == _checkpoint_ref(
            current, intent=transition_intent, receipt=telegram_receipt
        ) and self._trusted(plan_seal, expected_contract_sha256)

    def verify_live_singleton_lease(
        self,
        checkpoint: Mapping[str, Any],
        *,
        transition_intent: Mapping[str, Any],
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        try:
            events = self.controller.events.verified_events()
            payload = events[-1].get("payload") if events else None
            if not isinstance(payload, Mapping) \
                    or payload.get("transition_intent") != transition_intent \
                    or not isinstance(payload.get("telegram_delivery"), Mapping):
                return False
            current, _event = self._current(
                intent=transition_intent,
                receipt=payload["telegram_delivery"],
            )
            _lease_binding(self.controller)
            status = self.controller.status()
        except (DriverError, state.StateError, TypeError, ValueError):
            return False
        return status.get("durable_state_ok") is True \
            and status.get("state") == "AUTOTUNE_XET" \
            and status.get("live_worker_lease_ok") is True \
            and dict(checkpoint) == _checkpoint_ref(
                current,
                intent=transition_intent,
                receipt=payload["telegram_delivery"],
            ) \
            and self._trusted(plan_seal, expected_contract_sha256)

    def current_binding(
        self,
        *,
        intent: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        checkpoint, _event = self._current(intent=intent, receipt=receipt)
        status = self.controller.status()
        lease = _lease_binding(self.controller)
        if status.get("durable_state_ok") is not True \
                or status.get("state") != "AUTOTUNE_XET":
            raise DriverError("live controller lease/state is not green")
        # The entry-authority callback above requires a fresh controller heartbeat.
        # During the bounded body run the unchanged AUTOTUNE checkpoint is itself an
        # authority invariant, so appending controller heartbeat events would revoke
        # every issued capability.  Continued liveness is instead proved by the same
        # process still owning the exact flock descriptor at each capability check.
        return {
            "controller_epoch": self.controller.controller_epoch,
            "checkpoint_seal_sha256": checkpoint["seal_sha256"],
            "lease_identity_sha256": lease["lease_identity_sha256"],
            "telegram_receipt_seal_sha256": _sha(receipt),
        }


def _default_git_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "Never",
        "LC_ALL": "C",
    })
    try:
        return subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError(f"git provenance command failed: {exc}") from exc


class ProductionGitProvenance:
    """Clean-worktree and exact remote-HEAD verifier with local rechecks."""

    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        *,
        runner: GitRunner = _default_git_runner,
        required_paths: Sequence[str] = PROVENANCE_PATHS,
    ) -> None:
        self.root = Path(repo_root).resolve()
        self.runner = runner
        self.required_paths = tuple(required_paths)
        self._receipt: dict[str, Any] | None = None

    def _git(self, *arguments: str) -> str:
        result = self.runner(("/usr/bin/git", "-C", os.fspath(self.root), *arguments))
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1:] or ["no diagnostic"]
            raise DriverError(
                f"git provenance command refused ({arguments[0]}): {detail[0]}"
            )
        return result.stdout

    def _local(self, *, require_clean: bool) -> dict[str, Any]:
        head = self._git("rev-parse", "HEAD").strip()
        branch = self._git("symbolic-ref", "--short", "HEAD").strip()
        upstream = self._git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ).strip()
        if _HEX40.fullmatch(head) is None \
                or branch != OFFICIAL_BRANCH \
                or upstream != f"{OFFICIAL_REMOTE}/{OFFICIAL_BRANCH}":
            raise DriverError("git HEAD/branch/upstream identity is invalid")
        remote_url = self._git("remote", "get-url", OFFICIAL_REMOTE).strip()
        remote_url_sha256 = hashlib.sha256(remote_url.encode("utf-8")).hexdigest()
        if remote_url_sha256 != OFFICIAL_REMOTE_URL_SHA256:
            raise DriverError("git origin URL differs from the pinned Hawking remote")
        tracked = self._git("ls-files", "--error-unmatch", *self.required_paths)
        if set(tracked.splitlines()) != set(self.required_paths):
            raise DriverError("one or more live-Xet production paths are not tracked")
        required_diff = self.runner((
            "/usr/bin/git", "-C", os.fspath(self.root), "diff", "--quiet",
            "HEAD", "--", *self.required_paths,
        ))
        if required_diff.returncode != 0:
            raise DriverError("live-Xet production source differs from committed HEAD")
        status = self._git("status", "--porcelain=v1", "--untracked-files=all")
        if require_clean and status:
            raise DriverError("production live-Xet launch requires a clean worktree")
        return {
            "head": head,
            "branch": branch,
            "upstream": upstream,
            "remote_url_sha256": remote_url_sha256,
        }

    def _upstream_head(self, local: Mapping[str, Any]) -> str:
        remote, remote_branch = str(local["upstream"]).split("/", 1)
        listing = self._git(
            "ls-remote", "--heads", "--exit-code", remote, remote_branch
        ).strip().splitlines()
        if len(listing) != 1:
            raise DriverError("git upstream remote returned an ambiguous branch head")
        fields = listing[0].split()
        remote_head = fields[0] if len(fields) == 2 else ""
        if remote_head != local["head"]:
            raise DriverError("current git HEAD is not pushed exactly to its upstream")
        return remote_head

    def _issued_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        if self._receipt is None or dict(receipt) != self._receipt:
            raise DriverError("git provenance receipt was not issued by this verifier")
        return self._receipt

    def preflight(self) -> dict[str, Any]:
        local = self._local(require_clean=True)
        remote_head = self._upstream_head(local)
        receipt = seal({
            "schema": PROVENANCE_SCHEMA,
            "status": "PASS_CLEAN_HEAD_PUSHED_EXACTLY",
            "repo_root": os.fspath(self.root),
            **local,
            "remote_head": remote_head,
            "required_paths": list(self.required_paths),
            "verified_at": utc_now(),
        })
        self._receipt = receipt
        return _clone(receipt)

    def assert_current(self, receipt: Mapping[str, Any]) -> None:
        issued = self._issued_receipt(receipt)
        local = self._local(require_clean=False)
        if any(
            local[key] != issued[key]
            for key in ("head", "branch", "upstream", "remote_url_sha256")
        ):
            raise DriverError("git provenance changed after production preflight")

    def assert_final_current(self, receipt: Mapping[str, Any]) -> None:
        """Repeat the complete clean local and exact upstream check at commit."""
        issued = self._issued_receipt(receipt)
        local = self._local(require_clean=True)
        if any(
            local[key] != issued[key]
            for key in ("head", "branch", "upstream", "remote_url_sha256")
        ):
            raise DriverError("git provenance changed before terminal publication")
        if self._upstream_head(local) != issued["remote_head"]:
            raise DriverError("git upstream head changed before terminal publication")


class CampaignResourceGuard:
    """Carry one campaign baseline through every per-trial live sample."""

    def __init__(self, sampler: live.ResourceSampler, policy: Mapping[str, Any]) -> None:
        self.sampler = sampler
        self.policy = dict(policy)
        self.baseline: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        if self.baseline is not None:
            raise DriverError("campaign resource guard was already started")
        sample = live._validate_resource_sample(  # noqa: SLF001
            self.sampler.sample(os.getpid()), pid=os.getpid()
        )
        live._enforce_resource_policy_sample(  # noqa: SLF001
            sample, baseline=sample, policy=self.policy
        )
        self.baseline = dict(sample)
        self.last = dict(sample)
        return _clone(sample)

    def sample(self, pid: int) -> Mapping[str, Any]:
        if self.baseline is None:
            raise DriverError("campaign resource guard has no baseline")
        sample = live._validate_resource_sample(  # noqa: SLF001
            self.sampler.sample(pid), pid=pid
        )
        live._enforce_resource_policy_sample(  # noqa: SLF001
            sample, baseline=self.baseline, policy=self.policy
        )
        self.last = dict(sample)
        return sample

    def assert_safe(self) -> dict[str, Any]:
        return _clone(self.sample(os.getpid()))


class ProductionCapabilityIssuerVerifier:
    """Issue exact in-process capabilities and consume each seal at most once."""

    def __init__(
        self,
        *,
        authority_verifier: ProductionExecutionAuthorityVerifier,
        authority: Mapping[str, Any],
        provenance: ProductionGitProvenance,
        provenance_receipt: Mapping[str, Any],
    ) -> None:
        self.authority_verifier = authority_verifier
        self.authority = _clone(authority)
        self.provenance = provenance
        self.provenance_receipt = _clone(provenance_receipt)
        self._issued: dict[str, dict[str, Any]] = {}
        self._consumed: set[str] = set()

    def issue(
        self,
        *,
        plan: Mapping[str, Any],
        trial_id: str,
        kind: str,
        maximum_network_bytes: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.provenance.assert_current(self.provenance_receipt)
        controller = self.authority_verifier.current_binding(
            intent=self.authority["transition_intent"],
            receipt=self.authority["telegram_delivery_receipt"],
        )
        capability = seal({
            "schema": live.CAPABILITY_SCHEMA,
            "status": "AUTHORIZED",
            "repo": live.REPO_ID,
            "revision": live.REVISION,
            "plan_seal_sha256": plan["seal_sha256"],
            "trial_id": trial_id,
            "allowed_kind": kind,
            "max_network_bytes": maximum_network_bytes,
            "controller": controller,
            "expires_unix_ns": time.time_ns() + int(
                (timeout_seconds + CAPABILITY_TTL_GRACE_SECONDS) * 1_000_000_000
            ),
            "credentials_serialized": False,
        })
        if capability["seal_sha256"] in self._issued:
            raise DriverError("duplicate live capability seal would be replayable")
        self._issued[capability["seal_sha256"]] = capability
        return _clone(capability)

    def verify_live_capability(
        self,
        capability: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
        spec: Mapping[str, Any],
    ) -> bool:
        try:
            seal_value = capability.get("seal_sha256")
            if not _is_sha256(seal_value) \
                    or seal_value in self._consumed \
                    or self._issued.get(str(seal_value)) != dict(capability):
                return False
            self.provenance.assert_current(self.provenance_receipt)
            current = self.authority_verifier.current_binding(
                intent=self.authority["transition_intent"],
                receipt=self.authority["telegram_delivery_receipt"],
            )
            expected = {
                "controller_epoch": current["controller_epoch"],
                "checkpoint_seal_sha256": current["checkpoint_seal_sha256"],
                "lease_identity_sha256": current["lease_identity_sha256"],
                "telegram_receipt_seal_sha256": current[
                    "telegram_receipt_seal_sha256"
                ],
            }
            if capability.get("controller") != expected \
                    or capability.get("plan_seal_sha256") != plan.get("seal_sha256") \
                    or capability.get("trial_id") != spec.get("trial", {}).get("trial_id") \
                    or capability.get("allowed_kind") != spec.get("trial", {}).get("kind") \
                    or capability.get("max_network_bytes") != spec.get(
                        "network_budget", {}
                    ).get("trial_network_cap_bytes"):
                return False
            self._consumed.add(str(seal_value))
            return True
        except (DriverError, Glm52Error, KeyError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class _FileRecord:
    name: str
    device: int
    inode: int
    size: int
    sha256: str

    def evidence(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "sha256": self.sha256,
        }


class ProductionArtifactWriter:
    """Exclusive durable JSON publication with a fully revalidated inventory."""

    _NEW = "NEW"
    _ACTIVE = "ACTIVE"
    _FINISHED = "FINISHED"
    _FAILED = "FAILED"

    def __init__(self, artifact_root: Path, controller_root: Path) -> None:
        self.artifact_root, self._artifact_fd, self._artifact_identity = \
            self._open_anchored_directory(artifact_root)
        try:
            self.controller_root, self._controller_fd, self._controller_identity = \
                self._open_anchored_directory(controller_root)
        except BaseException:
            os.close(self._artifact_fd)
            self._artifact_fd = -1
            raise
        self.marker_path = self.controller_root / RUN_MARKER_NAME
        self.marker: dict[str, Any] | None = None
        self._marker_record: _FileRecord | None = None
        self.written: list[str] = []
        self._records: dict[str, _FileRecord] = {}
        self._state = self._NEW

    @staticmethod
    def _render(value: Mapping[str, Any]) -> bytes:
        return json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode("utf-8") + b"\n"

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DriverError("durable JSON write made no progress")
            view = view[written:]

    @staticmethod
    def _read_all(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    @staticmethod
    def _open_anchored_directory(path: Path) -> tuple[Path, int, tuple[int, int]]:
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            if absolute.resolve(strict=True) != absolute:
                raise DriverError(f"artifact directory contains a symlink: {absolute}")
            descriptor = os.open(
                absolute,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise DriverError(f"cannot open anchored artifact directory {absolute}: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            named = os.lstat(absolute)
            identity = (int(opened.st_dev), int(opened.st_ino))
            if not stat.S_ISDIR(opened.st_mode) \
                    or stat.S_ISLNK(named.st_mode) \
                    or not stat.S_ISDIR(named.st_mode) \
                    or identity != (int(named.st_dev), int(named.st_ino)):
                raise DriverError(f"artifact directory identity is unsafe: {absolute}")
            return absolute, descriptor, identity
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _assert_directory_current(
        path: Path, descriptor: int, identity: tuple[int, int]
    ) -> None:
        if descriptor < 0:
            raise DriverError("artifact directory descriptor is closed")
        opened = os.fstat(descriptor)
        try:
            named = os.lstat(path)
        except OSError as exc:
            raise DriverError(f"artifact directory name disappeared: {path}") from exc
        if not stat.S_ISDIR(opened.st_mode) \
                or stat.S_ISLNK(named.st_mode) \
                or not stat.S_ISDIR(named.st_mode) \
                or (int(opened.st_dev), int(opened.st_ino)) != identity \
                or (int(named.st_dev), int(named.st_ino)) != identity:
            raise DriverError(f"artifact directory identity changed: {path}")

    @staticmethod
    def _descriptor_identity(descriptor: int) -> tuple[int, int]:
        found = os.fstat(descriptor)
        return int(found.st_dev), int(found.st_ino)

    @classmethod
    def _verify_descriptor_content(
        cls,
        descriptor: int,
        *,
        identity: tuple[int, int],
        rendered: bytes,
        links: int,
    ) -> os.stat_result:
        found = os.fstat(descriptor)
        if not stat.S_ISREG(found.st_mode) \
                or int(found.st_nlink) != links \
                or (int(found.st_dev), int(found.st_ino)) != identity \
                or int(found.st_size) != len(rendered) \
                or stat.S_IMODE(found.st_mode) & 0o077:
            raise DriverError("published JSON descriptor identity or mode is unsafe")
        if cls._read_all(descriptor) != rendered:
            raise DriverError("published JSON bytes differ from the exact rendered value")
        return found

    @staticmethod
    def _named_identity(directory_fd: int, name: str) -> os.stat_result:
        try:
            return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise DriverError(f"published JSON name is unavailable: {name}") from exc

    @classmethod
    def _open_exact_named(
        cls,
        directory_fd: int,
        name: str,
        *,
        identity: tuple[int, int],
        rendered: bytes,
        links: int,
        writable: bool = False,
    ) -> int:
        flags = (os.O_RDWR if writable else os.O_RDONLY) \
            | getattr(os, "O_NOFOLLOW", 0) \
            | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except OSError as exc:
            raise DriverError(f"cannot reopen exact published JSON {name}: {exc}") from exc
        try:
            cls._verify_descriptor_content(
                descriptor, identity=identity, rendered=rendered, links=links
            )
            named = cls._named_identity(directory_fd, name)
            if stat.S_ISLNK(named.st_mode) \
                    or (int(named.st_dev), int(named.st_ino)) != identity:
                raise DriverError(f"published JSON name was replaced: {name}")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @classmethod
    def _publish_json_at(
        cls,
        directory_fd: int,
        directory_path: Path,
        directory_identity: tuple[int, int],
        name: str,
        value: Mapping[str, Any],
    ) -> _FileRecord:
        if not _safe_artifact_name(name):
            raise DriverError(f"live-Xet artifact name is not canonical: {name!r}")
        cls._assert_directory_current(
            directory_path, directory_fd, directory_identity
        )
        rendered = cls._render(value)
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL \
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            temporary_fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        except OSError as exc:
            raise DriverError(f"cannot create private JSON staging inode: {exc}") from exc
        temporary_identity = cls._descriptor_identity(temporary_fd)
        destination_fd = -1
        try:
            cls._write_all(temporary_fd, rendered)
            os.fsync(temporary_fd)
            cls._verify_descriptor_content(
                temporary_fd,
                identity=temporary_identity,
                rendered=rendered,
                links=1,
            )
            named_temporary = cls._named_identity(directory_fd, temporary)
            if (int(named_temporary.st_dev), int(named_temporary.st_ino)) != \
                    temporary_identity:
                raise DriverError("private JSON staging name was replaced")
            try:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise DriverError(f"refusing to overwrite live-Xet artifact {name}: {exc}") from exc
            destination_fd = cls._open_exact_named(
                directory_fd,
                name,
                identity=temporary_identity,
                rendered=rendered,
                links=2,
            )
            os.fsync(directory_fd)
            cls._assert_directory_current(
                directory_path, directory_fd, directory_identity
            )
            named_destination = cls._named_identity(directory_fd, name)
            if (int(named_destination.st_dev), int(named_destination.st_ino)) != \
                    temporary_identity:
                raise DriverError("published JSON destination was replaced")

            # Remove only the staging name after proving that it still names our
            # exact inode.  A failed or ambiguous publication deliberately leaves
            # both names for explicit reconciliation; it never unlinks destination.
            named_temporary = cls._named_identity(directory_fd, temporary)
            if (int(named_temporary.st_dev), int(named_temporary.st_ino)) != \
                    temporary_identity:
                raise DriverError("private JSON staging name became ambiguous")
            os.unlink(temporary, dir_fd=directory_fd)
            os.fsync(directory_fd)
            cls._verify_descriptor_content(
                destination_fd,
                identity=temporary_identity,
                rendered=rendered,
                links=1,
            )
            named_destination = cls._named_identity(directory_fd, name)
            if (int(named_destination.st_dev), int(named_destination.st_ino)) != \
                    temporary_identity:
                raise DriverError("published JSON destination changed after fsync")
            return _FileRecord(
                name=name,
                device=temporary_identity[0],
                inode=temporary_identity[1],
                size=len(rendered),
                sha256=hashlib.sha256(rendered).hexdigest(),
            )
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)
            os.close(temporary_fd)
            # Never delete a destination after link, even on fsync/error paths:
            # another process may have replaced it.  Failed staging names are
            # likewise retained rather than risking deletion of a foreign inode.

    @classmethod
    def _exclusive_json(
        cls, path: Path, value: Mapping[str, Any]
    ) -> _FileRecord:
        absolute = Path(os.path.abspath(os.fspath(path)))
        directory, directory_fd, identity = cls._open_anchored_directory(
            absolute.parent
        )
        try:
            return cls._publish_json_at(
                directory_fd, directory, identity, absolute.name, value
            )
        finally:
            os.close(directory_fd)

    @classmethod
    def _record_rendered(cls, record: _FileRecord, value: Mapping[str, Any]) -> bytes:
        rendered = cls._render(value)
        if len(rendered) != record.size \
                or hashlib.sha256(rendered).hexdigest() != record.sha256:
            raise DriverError(f"in-memory value differs from recorded JSON {record.name}")
        return rendered

    @classmethod
    def _rewrite_record_at(
        cls,
        directory_fd: int,
        directory_path: Path,
        directory_identity: tuple[int, int],
        record: _FileRecord,
        old_value: Mapping[str, Any],
        new_value: Mapping[str, Any],
    ) -> _FileRecord:
        cls._assert_directory_current(
            directory_path, directory_fd, directory_identity
        )
        descriptor = cls._open_exact_named(
            directory_fd,
            record.name,
            identity=(record.device, record.inode),
            rendered=cls._record_rendered(record, old_value),
            links=1,
            writable=True,
        )
        rendered = cls._render(new_value)
        try:
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            cls._write_all(descriptor, rendered)
            os.fsync(descriptor)
            cls._verify_descriptor_content(
                descriptor,
                identity=(record.device, record.inode),
                rendered=rendered,
                links=1,
            )
            named = cls._named_identity(directory_fd, record.name)
            if (int(named.st_dev), int(named.st_ino)) != \
                    (record.device, record.inode):
                raise DriverError("run marker name was replaced during update")
            os.fsync(directory_fd)
            cls._assert_directory_current(
                directory_path, directory_fd, directory_identity
            )
            return _FileRecord(
                name=record.name,
                device=record.device,
                inode=record.inode,
                size=len(rendered),
                sha256=hashlib.sha256(rendered).hexdigest(),
            )
        finally:
            os.close(descriptor)

    def close(self) -> None:
        for attribute in ("_artifact_fd", "_controller_fd"):
            descriptor = getattr(self, attribute, -1)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, attribute, -1)

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    def begin(self, binding: Mapping[str, Any]) -> None:
        if self._state != self._NEW:
            raise DriverError("live-Xet writer is not a fresh one-use writer")
        self._state = "BEGINNING"
        marker = seal({
            "schema": RUN_MARKER_SCHEMA,
            "status": "IN_PROGRESS_RESTART_REQUIRES_EXPLICIT_RECONCILIATION",
            "binding": _clone(binding),
            "written_artifacts": [],
            "artifact_inventory": [],
            "started_at": utc_now(),
            "finished_at": None,
            "terminal_driver_receipt_seal_sha256": None,
        })
        try:
            self._marker_record = self._publish_json_at(
                self._controller_fd,
                self.controller_root,
                self._controller_identity,
                RUN_MARKER_NAME,
                marker,
            )
        except BaseException:
            self._state = self._FAILED
            raise
        self.marker = marker
        self._state = self._ACTIVE

    def _update_marker(self, *, terminal: Mapping[str, Any] | None = None) -> None:
        if self.marker is None or self._marker_record is None:
            raise DriverError("live-Xet run marker is not initialized")
        body = {
            key: _clone(value)
            for key, value in self.marker.items()
            if key != "seal_sha256"
        }
        body["written_artifacts"] = list(self.written)
        body["artifact_inventory"] = [
            self._records[name].evidence() for name in self.written
        ]
        if terminal is not None:
            body["status"] = "PASS_COMPLETE"
            body["finished_at"] = utc_now()
            body["terminal_driver_receipt_seal_sha256"] = terminal["seal_sha256"]
        marker = seal(body)
        self._marker_record = self._rewrite_record_at(
            self._controller_fd,
            self.controller_root,
            self._controller_identity,
            self._marker_record,
            self.marker,
            marker,
        )
        self.marker = marker

    def write(self, name: str, value: Mapping[str, Any]) -> None:
        if self._state != self._ACTIVE:
            raise DriverError("live-Xet writer is not active")
        if not _safe_artifact_name(name) or name in self._records:
            raise DriverError("live-Xet artifact name is unsafe or duplicated")
        self._state = "WRITING"
        try:
            record = self._publish_json_at(
                self._artifact_fd,
                self.artifact_root,
                self._artifact_identity,
                name,
                value,
            )
            self._records[name] = record
            self.written.append(name)
            self._update_marker()
        except BaseException:
            self._state = self._FAILED
            raise
        self._state = self._ACTIVE

    def finish(self, receipt: Mapping[str, Any]) -> None:
        if self._state != self._ACTIVE:
            raise DriverError("live-Xet writer cannot finish more than once")
        self._state = "FINISHING"
        held: list[int] = []
        try:
            if not self.written or self.written[-1] != DRIVER_RECEIPT_NAME:
                raise DriverError("terminal driver receipt is not the final artifact")
            receipt_record = self._records[DRIVER_RECEIPT_NAME]
            self._record_rendered(receipt_record, receipt)
            self._assert_directory_current(
                self.artifact_root, self._artifact_fd, self._artifact_identity
            )
            self._assert_directory_current(
                self.controller_root, self._controller_fd, self._controller_identity
            )
            # Hold and verify every exact inode and byte string through the only
            # transition of the marker to PASS_COMPLETE.
            for name in self.written:
                record = self._records[name]
                content = self._read_exact_by_hash(record)
                descriptor = self._open_exact_named(
                    self._artifact_fd,
                    name,
                    identity=(record.device, record.inode),
                    rendered=content,
                    links=1,
                )
                held.append(descriptor)
            self._update_marker(terminal=receipt)
        except BaseException:
            self._state = self._FAILED
            raise
        finally:
            for descriptor in held:
                os.close(descriptor)
        self._state = self._FINISHED

    def _read_exact_by_hash(self, record: _FileRecord) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
            | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(record.name, flags, dir_fd=self._artifact_fd)
        try:
            found = os.fstat(descriptor)
            content = self._read_all(descriptor)
            if not stat.S_ISREG(found.st_mode) \
                    or int(found.st_nlink) != 1 \
                    or (int(found.st_dev), int(found.st_ino)) != \
                    (record.device, record.inode) \
                    or len(content) != record.size \
                    or hashlib.sha256(content).hexdigest() != record.sha256:
                raise DriverError(f"artifact inventory changed: {record.name}")
            return content
        finally:
            os.close(descriptor)


def _operation_caps(plan: Mapping[str, Any]) -> list[int]:
    payloads = [int(row["planned_payload_bytes"]) for row in plan["trial_matrix"]]
    largest = int(plan["largest_shard_validation"]["bytes"])
    payloads.extend((largest, largest))
    planned = sum(payloads)
    hard = int(plan["network_budget"]["hard_cap_bytes"])
    if planned != int(plan["network_budget"]["planned_maximum_bytes"]) or hard < planned:
        raise DriverError("sealed plan network budget cannot cover the exact 14 operations")
    guard = hard - planned
    extras = [guard * payload // planned for payload in payloads]
    extras[-1] += guard - sum(extras)
    caps = [payload + extra for payload, extra in zip(payloads, extras)]
    if sum(caps) != hard or any(cap < payload for cap, payload in zip(caps, payloads)):
        raise DriverError("deterministic per-operation network cap allocation failed")
    return caps


def _profile_selections(
    plan: Mapping[str, Any], trial_results: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    policy = live._resource_policy_from_plan(plan)  # noqa: SLF001
    candidates: list[dict[str, Any]] = []
    expected = [row["trial_id"] for row in plan["trial_matrix"]]
    if len(trial_results) != 12:
        raise DriverError("profile selection requires all exact 12 live trial results")
    for expected_id, raw in zip(expected, trial_results):
        result = live.validate_trial_result(raw, plan=plan)
        if result["trial_binding"]["trial_id"] != expected_id \
                or result["status"] != "PASS_COMPLETE_MEASURED":
            raise DriverError(
                f"live Xet trial is absent, reordered, or incomplete: {expected_id}"
            )
        resource = result["resource_observations"]
        verdict = autotune.evaluate_resource_trial(
            resource["before"],
            resource["samples"],
            resource["after"],
            required_free_bytes=policy["required_free_disk_bytes"],
            required_available_ram_bytes=policy["minimum_available_ram_bytes"],
            trial_network_cap_bytes=result["network_accounting"][
                "trial_network_cap_bytes"
            ],
            maximum_swap_used_bytes=policy["maximum_swap_used_bytes"],
            maximum_swap_growth_bytes=policy["maximum_swap_growth_bytes"],
            maximum_materialized_raw_allocated_bytes=policy[
                "maximum_materialized_raw_allocated_bytes"
            ],
            heavy_lane_regressions=resource["heavy_lane_regressions"],
            complete_source_views=resource["complete_source_views"],
        )
        candidates.append(live._candidate_from_result(result, verdict))  # noqa: SLF001
    return {
        lane: autotune.select_profile(candidates, lane=lane)
        for lane in ("acquisition", "steady")
    }


@dataclass(frozen=True)
class PreparedRun:
    plan: dict[str, Any]
    contract: dict[str, Any]
    authority: dict[str, Any]
    authority_verifier: ProductionExecutionAuthorityVerifier
    provenance: ProductionGitProvenance
    provenance_receipt: dict[str, Any]
    resource_guard: CampaignResourceGuard
    resource_baseline: dict[str, Any]


def _terminal_currentness(
    prepared: PreparedRun,
    *,
    initial_authority_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild every input and repeat remote/authority gates after attestation."""
    prepared.provenance.assert_current(prepared.provenance_receipt)
    try:
        rebuilt = live.validate_live_plan(
            prepared.plan, root=REPO_ROOT, rebuild=True
        )
    except (Glm52Error, KeyError, TypeError, ValueError) as exc:
        raise DriverError(f"terminal live Xet plan rebuild failed: {exc}") from exc
    if canonical(rebuilt) != canonical(prepared.plan):
        raise DriverError("terminal plan rebuild differs from the prepared live plan")
    prepared.provenance.assert_final_current(prepared.provenance_receipt)
    final_sample = prepared.resource_guard.assert_safe()
    validated_authority = autotune.validate_execution_authority(
        prepared.authority,
        plan_seal=prepared.plan["seal_sha256"],
        expected_contract_sha256=prepared.contract["seal_sha256"],
        verifier=prepared.authority_verifier,
    )
    if canonical(validated_authority) != canonical(prepared.authority):
        raise DriverError("terminal authority validation returned a different authority")
    final_binding = prepared.authority_verifier.current_binding(
        intent=prepared.authority["transition_intent"],
        receipt=prepared.authority["telegram_delivery_receipt"],
    )
    if dict(final_binding) != dict(initial_authority_binding):
        raise DriverError("controller authority/lease binding changed during live Xet run")
    return _clone(final_binding), final_sample


def _prepare_under_lease(
    runtime: gravity.Runtime,
    controller: state.Controller,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    *,
    resource_sampler: live.ResourceSampler,
    provenance: ProductionGitProvenance,
) -> PreparedRun:
    """Complete every non-body gate; authority is checked before subprocess probes."""
    if os.environ.get(live.EXECUTE_ENV) != "1":
        raise DriverError(f"production driver requires {live.EXECUTE_ENV}=1")
    minimal_plan = _minimal_plan_identity(plan)
    expected_contract = _require_official_runtime(runtime, contract)
    if authority is None:
        raise DriverError("sealed Xet execution authority is required")
    authority_value = _clone(authority)
    authority_verifier = ProductionExecutionAuthorityVerifier(runtime, controller)
    # This structural + live validation happens before deterministic plan rebuild,
    # git, resource-command, network-counter, or model-child process execution.
    autotune.validate_execution_authority(
        authority_value,
        plan_seal=minimal_plan["seal_sha256"],
        expected_contract_sha256=expected_contract["seal_sha256"],
        verifier=authority_verifier,
    )
    try:
        validated_plan = live.validate_live_plan(
            minimal_plan, root=REPO_ROOT, rebuild=True
        )
    except (Glm52Error, KeyError, TypeError, ValueError) as exc:
        raise DriverError(f"live Xet plan rebuild failed: {exc}") from exc
    provenance_receipt = provenance.preflight()
    policy = live._resource_policy_from_plan(validated_plan)  # noqa: SLF001
    resource_guard = CampaignResourceGuard(resource_sampler, policy)
    baseline = resource_guard.start()
    provenance.assert_current(provenance_receipt)
    authority_verifier.current_binding(
        intent=authority_value["transition_intent"],
        receipt=authority_value["telegram_delivery_receipt"],
    )
    return PreparedRun(
        plan=validated_plan,
        contract=expected_contract,
        authority=authority_value,
        authority_verifier=authority_verifier,
        provenance=provenance,
        provenance_receipt=provenance_receipt,
        resource_guard=resource_guard,
        resource_baseline=baseline,
    )


def _artifact_prefix(ordinal: int, trial_id: str) -> str:
    return f"GLM52_XET_{ordinal:02d}_{trial_id}"


def _execute_prepared(
    runtime: gravity.Runtime,
    controller: state.Controller,
    prepared: PreparedRun,
    *,
    network_counter: live.NetworkBudgetCounter,
    writer: ArtifactWriter,
) -> dict[str, Any]:
    plan = prepared.plan
    authority = prepared.authority
    issuer = ProductionCapabilityIssuerVerifier(
        authority_verifier=prepared.authority_verifier,
        authority=authority,
        provenance=prepared.provenance,
        provenance_receipt=prepared.provenance_receipt,
    )
    caps = _operation_caps(plan)
    initial_authority_binding = prepared.authority_verifier.current_binding(
        intent=authority["transition_intent"],
        receipt=authority["telegram_delivery_receipt"],
    )
    run_binding = {
        "campaign_id": runtime.campaign_id,
        "source_revision": runtime.source_revision,
        "controller_epoch": runtime.controller_epoch,
        "expected_contract_sha256": prepared.contract["seal_sha256"],
        "plan_seal_sha256": plan["seal_sha256"],
        "authority_seal_sha256": authority["seal_sha256"],
        "provenance_seal_sha256": prepared.provenance_receipt["seal_sha256"],
        "resource_baseline_sha256": _sha(prepared.resource_baseline),
        "operation_caps_sha256": _sha(caps),
        "initial_authority_binding": _clone(initial_authority_binding),
    }
    writer.begin(run_binding)
    consumed = 0
    trial_results: list[dict[str, Any]] = []

    for ordinal, row in enumerate(plan["trial_matrix"]):
        prepared.resource_guard.assert_safe()
        prepared.provenance.assert_current(prepared.provenance_receipt)
        trial_id = row["trial_id"]
        cap = caps[ordinal]
        capability = issuer.issue(
            plan=plan,
            trial_id=trial_id,
            kind="BOUNDED_XET_BODY_RANGE",
            maximum_network_bytes=cap,
            timeout_seconds=TRIAL_TIMEOUT_SECONDS,
        )
        spec = live.build_trial_spec(
            plan,
            trial_id,
            capability_seal_sha256=capability["seal_sha256"],
            campaign_consumed_bytes=consumed,
            trial_network_cap_bytes=cap,
            timeout_seconds=TRIAL_TIMEOUT_SECONDS,
            sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
            root=REPO_ROOT,
            rebuild_plan=False,
        )
        prefix = _artifact_prefix(ordinal, trial_id)
        writer.write(f"{prefix}_CAPABILITY.json", capability)
        writer.write(f"{prefix}_SPEC.json", spec)
        result = live.execute_trial(
            plan,
            spec,
            capability,
            capability_verifier=issuer,
            resource_sampler=prepared.resource_guard,
            network_counter=network_counter,
            root=REPO_ROOT,
        )
        result = live.validate_trial_result(result, plan=plan)
        if result["status"] != "PASS_COMPLETE_MEASURED":
            raise DriverError(f"live Xet trial lacks required measurements: {trial_id}")
        writer.write(f"{prefix}_RESULT.json", result)
        trial_results.append(result)
        consumed += int(result["network_accounting"]["actual_network_bytes"])

    selections = _profile_selections(plan, trial_results)
    largest_evidence: list[dict[str, Any]] = []
    largest_results: list[dict[str, Any]] = []
    for lane_index, lane in enumerate(("acquisition", "steady"), start=12):
        prepared.resource_guard.assert_safe()
        prepared.provenance.assert_current(prepared.provenance_receipt)
        selected_id = selections[lane]["trial_id"]
        trial_id = f"LARGEST_{lane.upper()}"
        cap = caps[lane_index]
        capability = issuer.issue(
            plan=plan,
            trial_id=trial_id,
            kind="FULL_LARGEST_SHARD_VALIDATION",
            maximum_network_bytes=cap,
            timeout_seconds=LARGEST_TIMEOUT_SECONDS,
        )
        spec = live.build_largest_validation_spec(
            plan,
            lane=lane,
            selected_trial_id=selected_id,
            capability_seal_sha256=capability["seal_sha256"],
            campaign_consumed_bytes=consumed,
            trial_network_cap_bytes=cap,
            timeout_seconds=LARGEST_TIMEOUT_SECONDS,
            sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
            root=REPO_ROOT,
            rebuild_plan=False,
        )
        prefix = _artifact_prefix(lane_index, trial_id)
        writer.write(f"{prefix}_CAPABILITY.json", capability)
        writer.write(f"{prefix}_SPEC.json", spec)
        result = live.execute_trial(
            plan,
            spec,
            capability,
            capability_verifier=issuer,
            resource_sampler=prepared.resource_guard,
            network_counter=network_counter,
            root=REPO_ROOT,
        )
        result = live.validate_trial_result(result, plan=plan)
        evidence = live.build_largest_validation_evidence(
            plan, result, lane=lane, selected_trial_id=selected_id
        )
        writer.write(f"{prefix}_RESULT.json", result)
        writer.write(f"{prefix}_EVIDENCE.json", evidence)
        largest_results.append(result)
        largest_evidence.append(evidence)
        consumed += int(result["network_accounting"]["actual_network_bytes"])

    prepared.provenance.assert_current(prepared.provenance_receipt)
    prepared.authority_verifier.current_binding(
        intent=authority["transition_intent"],
        receipt=authority["telegram_delivery_receipt"],
    )
    raw_result = live.assemble_autotune_result(
        plan,
        trial_results,
        largest_evidence,
        required_free_bytes=autotune.REQUIRED_FREE_DISK_BYTES,
        required_available_ram_bytes=autotune.MINIMUM_AVAILABLE_RAM_BYTES,
        root=REPO_ROOT,
        rebuild_plan=False,
    )
    checkpoint = controller.resume(recover_single_tail=False)
    controller_anchor = controller._controller_anchor(checkpoint)  # noqa: SLF001
    attested = schedule_freeze.attest_xet_autotune_result(
        raw_result,
        plan,
        prepared.contract,
        auth=runtime.evidence_auth,
        controller_anchor_sha256=controller_anchor["anchor_sha256"],
        root=REPO_ROOT,
        rebuild_plan=False,
    )
    terminal_binding, final_sample = _terminal_currentness(
        prepared, initial_authority_binding=initial_authority_binding
    )
    receipt_body = {
        "schema": DRIVER_SCHEMA,
        "status": "PASS_ALL_12_TRIALS_AND_TWO_FULL_HASH_VALIDATIONS",
        "campaign_id": runtime.campaign_id,
        "source_revision": runtime.source_revision,
        "expected_contract_sha256": prepared.contract["seal_sha256"],
        "plan_seal_sha256": plan["seal_sha256"],
        "authority_seal_sha256": authority["seal_sha256"],
        "controller_anchor_sha256": controller_anchor["anchor_sha256"],
        "provenance": _clone(prepared.provenance_receipt),
        "resource_policy": live._resource_policy_from_plan(plan),  # noqa: SLF001
        "resource_baseline": _clone(prepared.resource_baseline),
        "resource_final": final_sample,
        "trial_result_seals": [item["seal_sha256"] for item in trial_results],
        "largest_result_seals": [item["seal_sha256"] for item in largest_results],
        "largest_evidence_seals": [item["seal_sha256"] for item in largest_evidence],
        "raw_result_seal_sha256": raw_result["seal_sha256"],
        "attested_result_seal_sha256": attested["seal_sha256"],
        "actual_network_bytes": consumed,
        "campaign_network_hard_cap_bytes": plan["network_budget"]["hard_cap_bytes"],
        "coverage": {
            "trial_ids": [row["trial_id"] for row in plan["trial_matrix"]],
            "trial_count": len(trial_results),
            "largest_validation_lanes": ["acquisition", "steady"],
            "largest_validation_count": len(largest_evidence),
        },
        "body_boundary": {
            "body_files_created_by_autotune_executor": 0,
            "full_model_downloaded": False,
            "streaming_schedule_refreeze_required": True,
        },
        "completed_at": utc_now(),
    }
    receipt = state.seal_producer_authenticated_evidence(
        receipt_body, auth=runtime.evidence_auth
    )
    writer.write(RAW_RESULT_NAME, raw_result)
    writer.write(ATTESTED_RESULT_NAME, attested)
    writer.write(DRIVER_RECEIPT_NAME, receipt)
    # Keep the last gap before the terminal marker to local, bounded checks.
    # The expensive deterministic rebuild and upstream query above already ran
    # after attestation; these checks prove neither local inputs, resources, nor
    # the exact held lease changed while the three terminal artifacts published.
    prepared.resource_guard.assert_safe()
    prepared.provenance.assert_current(prepared.provenance_receipt)
    current_binding = prepared.authority_verifier.current_binding(
        intent=authority["transition_intent"],
        receipt=authority["telegram_delivery_receipt"],
    )
    if current_binding != terminal_binding:
        raise DriverError("authority/lease binding changed before terminal marker")
    writer.finish(receipt)
    return {
        "raw_result": raw_result,
        "attested_result": attested,
        "driver_receipt": receipt,
    }


def run_production_autotune(
    runtime: gravity.Runtime,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Acquire the controller lease and run the exact production campaign."""
    controller = runtime.controller()
    with controller:
        return run_under_held_controller(
            runtime,
            controller,
            plan,
            contract,
            authority,
            scratch_root=scratch_root,
        )


def run_under_held_controller(
    runtime: gravity.Runtime,
    controller: state.Controller,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Run from a worker that already owns the exact production controller lease."""
    resource_root = Path(scratch_root).resolve(strict=True)
    artifact_root = Path(runtime.artifact_root).resolve(strict=True)
    if resource_root != artifact_root:
        raise DriverError(
            "resource sampling root must be the runtime's exact anchored artifact root"
        )
    if controller.campaign_id != runtime.campaign_id \
            or controller.source_revision != runtime.source_revision \
            or controller.controller_epoch != runtime.controller_epoch \
            or Path(controller.root).resolve(strict=True) != \
            Path(runtime.controller_root).resolve(strict=True) \
            or Path(controller.artifact_root).resolve(strict=True) != artifact_root:
        raise DriverError("held controller differs from the loaded production runtime")
    _lease_binding(controller)
    resource_sampler = live.DarwinResourceSampler(resource_root)
    network_counter = live.DarwinHostNetworkCounter()
    verifier = ProductionGitProvenance(REPO_ROOT)
    prepared = _prepare_under_lease(
        runtime,
        controller,
        plan,
        contract,
        authority,
        resource_sampler=resource_sampler,
        provenance=verifier,
    )
    writer = ProductionArtifactWriter(runtime.artifact_root, runtime.controller_root)
    try:
        return _execute_prepared(
            runtime,
            controller,
            prepared,
            network_counter=network_counter,
            writer=writer,
        )
    finally:
        writer.close()


def _command_run(args: argparse.Namespace) -> int:
    if args.execute_live is not True:
        raise DriverError("--execute-live is required")
    runtime = gravity.load_runtime(args.config)
    plan = read_sealed_json(args.plan)
    contract = read_sealed_json(runtime.expected_contract_path)
    authority = read_sealed_json(args.authority)
    outcome = run_production_autotune(
        runtime,
        plan,
        contract,
        authority,
        scratch_root=args.scratch_root,
    )
    print(json.dumps({
        "status": outcome["driver_receipt"]["status"],
        "raw_result_seal_sha256": outcome["raw_result"]["seal_sha256"],
        "attested_result_seal_sha256": outcome["attested_result"]["seal_sha256"],
        "driver_receipt_seal_sha256": outcome["driver_receipt"]["seal_sha256"],
    }, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the fully authority-gated live autotune")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--plan", type=Path, default=REPO_ROOT / "GLM52_XET_AUTOTUNE_PLAN.json"
    )
    run.add_argument("--authority", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--execute-live", action="store_true", required=True)
    run.set_defaults(handler=_command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (DriverError, Glm52Error, gravity.CliError, state.StateError, OSError) as exc:
        print(
            json.dumps({"status": "REFUSED", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
