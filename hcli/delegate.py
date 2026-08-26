"""Claude -> HCLI delegation surface: run / status / steer / result / abort.

This module is an ADAPTER. It owns no scheduling, no locking, no retry and no
persistence of its own. Everything durable here is one of:

* ``hcli.mission`` — ``Mission``, ``mission_dir``, ``mission_state_path``,
  ``Mission.checkpoint``, ``Mission.from_workspace``, ``Mission.cancel``
* ``hcli.steering`` — ``SteeringQueue`` / ``SteerEvent``
* ``hcli.resources`` — ``MutationLock`` (which is what carries ``pid_is_alive``
  and the process START TOKEN, so a recycled pid cannot impersonate a holder)
* ``hcli.persist`` — ``atomic_write_json`` / ``atomic_write_text``
* ``hcli.verifier_pipeline`` — ``run_pipeline`` / ``Verdict`` / ``ModelCaller``
* ``hcli.mutation`` — ``content_fingerprint``
* ``hcli.backends`` — ``_post_json`` for the real OpenAI-compatible endpoint

THE CLASSIFICATION LAW lives in exactly two functions and every path routes
through them:

* ``classify_claim(text, artifact)`` is the ONLY door into ``verified_facts``.
  ``artifact`` is a required positional parameter with no default, so a caller
  cannot promote a claim by forgetting an argument. A claim with no deterministic
  artifact — including anything a model said, at any level of confidence — is a
  hypothesis.
* ``verifier_outcome(record)`` is the ONLY judge of whether a verifier passed.
  ``TRUE`` with a nonzero exit code is ``failed``, not ``passed``.

``decide_verdict`` consumes both and can never return ACCEPT while a required
verifier is ``failed``.

Every verb is DISK-FIRST: ``status``, ``result``, ``steer`` and ``abort`` read
durable state and never require a live in-process ``Mission``. Only ``abort``
loads state through ``Mission`` machinery, and only to reuse ``cancel``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .mission import Mission, mission_dir, mission_state_path
from .persist import atomic_write_json
from .resources import MutationLock
from .steering import STEER_KINDS, SteeringQueue

SPEC_FILENAME = "delegation_spec.json"
ENVELOPE_FILENAME = "delegation_envelope.json"
CANCEL_FILENAME = "delegation_cancel.json"
DELEGATIONS_DIRNAME = "delegations"
SPEC_VERSION = 1
ENVELOPE_VERSION = 1
EXEC_VERB = "__delegate_exec"

VERIFY_MODES = ("standard", "strict")
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MAX_FIELD_CHARS = 4000
COMMAND_TIMEOUT_S = 600.0
ABORT_JOIN_TIMEOUT_S = 3.0


class DelegationError(RuntimeError):
    """Base for adapter refusals."""


class MissionNotFound(DelegationError):
    """No delegation spec at the resolved workspace."""


class DelegationBusy(DelegationError):
    """A live writer already holds this workspace."""


# ---------------------------------------------------------------------------
# Paths. mission_dir() is the authority; nothing here invents a layout.
# ---------------------------------------------------------------------------


def delegations_root(root: Optional[Union[str, Path]] = None) -> Path:
    return Path(root or os.getcwd()) / ".hcli" / DELEGATIONS_DIRNAME


def spec_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / SPEC_FILENAME


def envelope_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / ENVELOPE_FILENAME


def cancel_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / CANCEL_FILENAME


def resolve_workspace(
    mission: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> Path:
    """A mission handle is either a workspace path or a mission id.

    The filesystem is the registry: a mission id names a directory under
    ``<root>/.hcli/delegations/``. No index file, so nothing can go stale.
    """
    token = os.fspath(mission)
    candidates = [Path(token), delegations_root(root) / token]
    for ws in candidates:
        if spec_path(ws).is_file():
            return ws
    raise MissionNotFound(
        f"no {SPEC_FILENAME} for mission {token!r}; looked in "
        + ", ".join(str(mission_dir(c)) for c in candidates)
    )


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """(value, defect). Never raises, never guesses a value."""
    if not path.is_file():
        return None, f"missing file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable {path.name}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# THE CLASSIFICATION LAW
# ---------------------------------------------------------------------------

_ARTIFACT_KINDS = ("receipt", "command")


@dataclass(frozen=True)
class Artifact:
    """A deterministic support. Either a path that exists, or a command run."""

    kind: str
    ref: str
    exit_code: Optional[int] = None
    output: Optional[str] = None
    observed_at: Optional[float] = None
    # What exit code makes this command SUPPORT its claim. Defaults to 0, so a
    # command that FAILED cannot back an unqualified success claim. A refutation
    # sets this to the failing code, because there the failure IS the evidence.
    expected_exit: Optional[int] = 0

    def to_dict(self) -> Dict[str, Any]:
        text, clipped = _clip(self.output)
        return {
            "kind": self.kind,
            "ref": self.ref,
            "exit_code": self.exit_code,
            "expected_exit": self.expected_exit,
            "output": text,
            "output_truncated": clipped,
            "observed_at": self.observed_at,
        }


def artifact_defect(artifact: Optional[Artifact]) -> Optional[str]:
    """Why this artifact cannot support a verified fact. None = it can."""
    if artifact is None:
        return "no deterministic artifact supplied"
    if not isinstance(artifact, Artifact):
        return f"not an Artifact record: {type(artifact).__name__}"
    if artifact.kind not in _ARTIFACT_KINDS:
        return f"unknown artifact kind {artifact.kind!r}"
    ref = (artifact.ref or "").strip()
    if not ref:
        return f"{artifact.kind} artifact has an empty ref"
    if artifact.kind == "receipt":
        if not Path(ref).exists():
            return f"receipt path does not exist on disk: {ref}"
        return None
    if not isinstance(artifact.exit_code, int) or isinstance(artifact.exit_code, bool):
        return "command artifact has no recorded integer exit code"
    # A FAILED command is evidence of failure, not of the claim. The adversarial
    # audit found this promotion latent -- unreachable through build_envelope
    # today, but live the moment a new caller passes a nonzero code -- and the
    # steer requires the distinction be hard to violate ACCIDENTALLY. The escape
    # hatch is explicit, never a default: a refutation declares the failing code
    # as expected, so `verifier for X did not pass (exit_code=1)` stays verifiable.
    # This mirrors the AKB rule already in this repo: "evidence_class Measured but
    # source records pass: false -- a failed run is evidence of failure".
    expected = artifact.expected_exit
    if not isinstance(expected, int) or isinstance(expected, bool):
        return "command artifact has no integer expected_exit"
    if artifact.exit_code != expected:
        return (f"command exited {artifact.exit_code}, expected {expected}: a command "
                f"that did not do what was expected is evidence of that, not of the claim")
    return None


def classify_claim(text: str, artifact: Optional[Artifact]) -> Dict[str, Any]:
    """THE ONE DOOR into ``verified_facts``.

    ``artifact`` is positional and has NO default: promotion cannot happen by
    omission. Model prose reaches this function with ``artifact=None`` and
    stays a hypothesis no matter how it is worded.
    """
    claim = str(text)
    defect = artifact_defect(artifact)
    if defect is None:
        assert isinstance(artifact, Artifact)  # narrowed by artifact_defect
        return {
            "claim": claim,
            "class": "verified",
            "artifact": artifact.to_dict(),
            "reason": None,
        }
    return {
        "claim": claim,
        "class": "hypothesis",
        "artifact": artifact.to_dict() if isinstance(artifact, Artifact) else None,
        "reason": defect,
    }


def verifier_outcome(record: Dict[str, Any]) -> str:
    """THE ONE JUDGE: 'passed' | 'failed' | 'unresolved'.

    A model verdict of TRUE against a nonzero exit code is ``failed``. This is
    the function the adversarial mutation test (H) attacks; if mutating it does
    not break test A, the classifier is decoration.
    """
    verdict = str(record.get("verdict") or "").strip().upper()
    code = record.get("exit_code")
    if verdict == "TRUE" and isinstance(code, int) and code == 0:
        return "passed"
    if verdict == "UNVERIFIABLE":
        return "unresolved"
    return "failed"


def decide_verdict(
    *,
    verified_facts: Sequence[Dict[str, Any]],
    verifier_records: Sequence[Dict[str, Any]],
    blocker: Optional[str],
    defects: Sequence[str],
    remaining: Sequence[str],
) -> str:
    """ACCEPT | BLOCKED | INCONCLUSIVE. Never ACCEPT over a failed verifier."""
    if defects:
        return "INCONCLUSIVE"
    failed = [r for r in verifier_records if verifier_outcome(r) == "failed"]
    required_failed = [r for r in failed if r.get("required")]
    if required_failed:
        return "BLOCKED"
    if blocker:
        return "BLOCKED"
    if failed:
        # Non-required verifier failed: not a blocker, but not a clean accept.
        return "INCONCLUSIVE"
    if remaining:
        return "INCONCLUSIVE"
    if not verified_facts:
        return "INCONCLUSIVE"
    return "ACCEPT"


# Authority tiers. An actual command run beats a recorded number; newer durable
# disk state beats a stale receipt.
_AUTHORITY_TIER = {"command": 3, "disk": 2, "receipt": 1, "recorded": 0}


def pick_authority(
    candidates: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Resolve conflicting evidence for one value.

    Each candidate is ``{"value":..., "source": command|disk|receipt|recorded,
    "observed_at": float|None}``. Higher tier wins outright; within a tier the
    newer observation wins. ``None`` values never win.
    """
    usable = [c for c in candidates if c.get("value") is not None]
    if not usable:
        return None
    return max(
        usable,
        key=lambda c: (
            _AUTHORITY_TIER.get(str(c.get("source")), 0),
            float(c.get("observed_at") or 0.0),
        ),
    )


def _clip(text: Any, limit: int = MAX_FIELD_CHARS) -> Tuple[Optional[str], bool]:
    if text is None:
        return None, False
    body = text if isinstance(text, str) else str(text)
    if len(body) <= limit:
        return body, False
    return body[:limit] + f"\n[...truncated {len(body) - limit} chars]", True


# ---------------------------------------------------------------------------
# run — writes the spec BEFORE any work, returns a mission id, does not block
# ---------------------------------------------------------------------------


def build_spec(
    objective: str,
    *,
    mission_id: str,
    constraints: Sequence[str] = (),
    allowed_resources: Sequence[str] = (),
    protected_paths: Sequence[str] = (),
    verification: str = "standard",
    budget: Optional[Dict[str, Any]] = None,
    output_contract: Sequence[str] = (),
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    mode = (verification or "standard").strip().lower()
    if mode not in VERIFY_MODES:
        raise DelegationError(
            f"verification must be one of {VERIFY_MODES}, got {verification!r}"
        )
    if not (objective or "").strip():
        raise DelegationError("run requires a non-empty objective")
    return {
        "spec_version": SPEC_VERSION,
        "mission_id": mission_id,
        "objective": objective.strip(),
        "constraints": [str(c) for c in constraints],
        "allowed_resources": [str(r) for r in allowed_resources],
        "protected_paths": [str(p) for p in protected_paths],
        "verification": mode,
        "budget": dict(budget) if budget else None,
        "output_contract": [str(p) for p in output_contract],
        "endpoint": endpoint or os.environ.get("HCLI_ENDPOINT", DEFAULT_ENDPOINT),
        "created_at": time.time(),
    }


def run(
    objective: str,
    *,
    root: Optional[Union[str, Path]] = None,
    workspace: Optional[Union[str, Path]] = None,
    spawn: bool = True,
    **spec_kwargs: Any,
) -> Dict[str, Any]:
    """Start a delegated mission. Returns immediately with a mission id."""
    mission_id = str(spec_kwargs.pop("mission_id", "") or uuid.uuid4().hex[:12])
    ws = Path(workspace) if workspace is not None else delegations_root(root) / mission_id
    ws.mkdir(parents=True, exist_ok=True)

    # Exclusivity is MutationLock's job. run() does not become the writer; it
    # only refuses to start a second one over a live holder.
    lock = MutationLock(ws)
    if not lock.try_break_stale():
        record = lock.read() or {}
        raise DelegationBusy(
            f"workspace {ws} already has a live writer (pid={record.get('pid')}, "
            f"unit={record.get('unit_id')})"
        )

    spec = build_spec(objective, mission_id=mission_id, **spec_kwargs)
    spec["fingerprint_at_start"] = _fingerprint(spec["output_contract"], ws)
    mission_dir(ws).mkdir(parents=True, exist_ok=True)
    atomic_write_json(spec_path(ws), spec)

    # units={} on purpose. Mission(goal=...) would compile the objective into
    # `implement`/`validate` scheduler units, and this adapter never schedules
    # a unit — obligations are settled through verifier_pipeline. Left to
    # default they sit `pending` forever and get reported as real outstanding
    # work in every status and every envelope, including accepted ones.
    mission = Mission(
        ws, goal=spec["objective"], mission_id=mission_id, quiet=True, units={}
    )
    mission.phase = "delegated"
    mission.checkpoint()

    pid = None
    if spawn:
        pid = _spawn_executor(ws)
        if pid is not None:
            mission.register_child_pid(pid)
    return {
        "mission_id": mission_id,
        "workspace": str(ws),
        "spec": str(spec_path(ws)),
        "state": str(mission_state_path(ws)),
        "writer_pid": pid,
        "spawned": bool(pid),
    }


def _spawn_executor(workspace: Path) -> Optional[int]:
    repo = Path(__file__).resolve().parent.parent
    log = mission_dir(workspace) / "delegate_exec.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(log, "ab")
    except OSError:
        return None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hcli", EXEC_VERB, str(workspace)],
            cwd=str(repo),
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return None
    finally:
        handle.close()
    return proc.pid


def _fingerprint(paths: Sequence[str], workspace: Path) -> Optional[str]:
    if not paths:
        return None
    from .mutation import content_fingerprint

    try:
        return content_fingerprint(list(paths), root=str(workspace))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# status / steer / abort — disk-first
# ---------------------------------------------------------------------------


def status(
    mission: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Live snapshot from DURABLE state only. No in-process Mission needed."""
    ws = resolve_workspace(mission, root)
    spec, spec_defect = _read_json(spec_path(ws))
    state, state_defect = _read_json(mission_state_path(ws))
    envelope, _ = _read_json(envelope_path(ws))
    cancel, _ = _read_json(cancel_path(ws))
    spec = spec if isinstance(spec, dict) else {}
    state = state if isinstance(state, dict) else {}

    lock = MutationLock(ws)
    record = lock.read()
    writer = None
    if record:
        writer = {
            "pid": record.get("pid"),
            "unit_id": record.get("unit_id"),
            "acquired_at": record.get("acquired_at"),
            "alive": lock.holder_is_live(record),
        }

    units = state.get("units") if isinstance(state.get("units"), dict) else {}
    by_status: Dict[str, int] = {}
    for payload in units.values():
        if isinstance(payload, dict):
            key = str(payload.get("status") or "unknown")
            by_status[key] = by_status.get(key, 0) + 1

    return {
        "mission_id": spec.get("mission_id") or state.get("id"),
        "workspace": str(ws),
        "objective": spec.get("objective"),
        "verification": spec.get("verification"),
        "created_at": spec.get("created_at"),
        "phase": state.get("phase"),
        "last_checkpoint": state.get("last_checkpoint"),
        "units_by_status": by_status,
        "writer": writer,
        "steers_pending": len(_pending_steers(ws, state)),
        "cancel_requested": bool(cancel),
        "cancel_reason": (cancel or {}).get("reason") or state.get("cancel_reason"),
        "envelope_present": isinstance(envelope, dict),
        "verdict": (envelope or {}).get("verdict"),
        "blocker": (envelope or {}).get("blocker"),
        "defects": [d for d in (spec_defect, state_defect) if d],
    }


def _steering_queue(ws: Path, state: Optional[Dict[str, Any]] = None) -> SteeringQueue:
    if state is None:
        state, _ = _read_json(mission_state_path(ws))
    state = state if isinstance(state, dict) else {}
    spec, _ = _read_json(spec_path(ws))
    spec = spec if isinstance(spec, dict) else {}
    session = state.get("session_id") or spec.get("mission_id") or state.get("id")
    if not session:
        raise MissionNotFound(f"no session id recorded for {ws}")
    return SteeringQueue(str(ws), str(session))


def _pending_steers(ws: Path, state: Optional[Dict[str, Any]] = None) -> List[Any]:
    try:
        return _steering_queue(ws, state).pending()
    except Exception:
        return []


def steer(
    mission: Union[str, Path],
    text: str,
    kind: str = "knowledge",
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Queue a steer on DISK. Applies to future work; never rewrites history.

    Deliberately does NOT reload a ``Mission``: ``Mission.from_workspace`` fails
    every unit it finds ``running``, which would corrupt a live mission just
    because someone typed a steer. The executor owns the ledger and consumes
    the queue between obligations.
    """
    token = (kind or "knowledge").strip().lower()
    if token not in STEER_KINDS:
        raise DelegationError(f"kind must be one of {STEER_KINDS}, got {kind!r}")
    if not (text or "").strip():
        raise DelegationError("steer requires non-empty text")
    ws = resolve_workspace(mission, root)
    event = _steering_queue(ws).enqueue(text.strip(), kind=token)
    return {
        "mission_id": status(ws)["mission_id"],
        "steer_id": event.id,
        "kind": event.kind,
        "queued_at": event.timestamp,
        "applies_to": "future work only",
    }


def abort(
    mission: Union[str, Path],
    reason: str = "aborted by operator",
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Stop a delegation and leave lock + state recoverable."""
    ws = resolve_workspace(mission, root)
    reason = (reason or "aborted by operator").strip()
    atomic_write_json(cancel_path(ws), {"reason": reason, "at": time.time()})

    lock = MutationLock(ws)
    record = lock.read() or {}
    signalled = None
    pid = record.get("pid")
    try:
        pid = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid = None
    if pid and pid != os.getpid() and lock.holder_is_live(record):
        try:
            os.kill(pid, signal.SIGTERM)
            signalled = pid
        except OSError:
            signalled = None

    if signalled:
        # Bounded wait so the reported lock_free is usually true rather than
        # merely eventually true. A holder that outlives it is reported
        # honestly as lock_free=False; the next run() breaks it as stale.
        deadline = time.time() + ABORT_JOIN_TIMEOUT_S
        while time.time() < deadline and lock.holder_is_live(lock.read()):
            time.sleep(0.05)

    lock.release(record.get("unit_id"))
    lock_free = lock.try_break_stale()

    state, defect = _read_json(mission_state_path(ws))
    if isinstance(state, dict):
        state["phase"] = "cancelled"
        state["cancel_reason"] = reason
        state["last_checkpoint"] = time.time()
        atomic_write_json(mission_state_path(ws), state)

    envelope = build_envelope(ws, aborted_reason=reason)
    atomic_write_json(envelope_path(ws), envelope)
    return {
        "mission_id": envelope.get("mission_id"),
        "workspace": str(ws),
        "reason": reason,
        "signalled_pid": signalled,
        "lock_free": bool(lock_free),
        "verdict": envelope.get("verdict"),
        "state_defect": defect,
    }


# ---------------------------------------------------------------------------
# result — the envelope
# ---------------------------------------------------------------------------


def result(
    mission: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Return the envelope. Rebuilt from durable state when work is unfinished."""
    ws = resolve_workspace(mission, root)
    stored, _ = _read_json(envelope_path(ws))
    if isinstance(stored, dict) and stored.get("envelope_version") == ENVELOPE_VERSION:
        return stored
    return build_envelope(ws)


def _verifier_records(
    pipeline: Optional[Dict[str, Any]], mode: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize pipeline output. Missing fields are DEFECTS, never guesses."""
    records: List[Dict[str, Any]] = []
    defects: List[str] = []
    if pipeline is None:
        return records, defects
    if not isinstance(pipeline, dict):
        return records, [f"pipeline result is {type(pipeline).__name__}, not an object"]
    raw = pipeline.get("verdicts")
    if raw is None:
        return records, ["pipeline result has no 'verdicts'"]
    if not isinstance(raw, list):
        return records, ["pipeline 'verdicts' is not a list"]
    statements = {}
    for ob in pipeline.get("obligations") or []:
        if isinstance(ob, dict) and ob.get("id"):
            statements[str(ob["id"])] = ob
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            defects.append(f"verdict[{index}] is not an object")
            continue
        oid = str(item.get("obligation_id") or "")
        verdict = item.get("verdict")
        if verdict is None:
            defects.append(f"verdict[{index}] ({oid or 'unknown'}) has no verdict field")
        ob = statements.get(oid, {})
        consequential = ob.get("consequential")
        records.append(
            {
                "obligation_id": oid or None,
                "statement": ob.get("statement") or oid or None,
                "verdict": verdict,
                "command": item.get("command"),
                "exit_code": item.get("exit_code"),
                "output": item.get("output"),
                "evidence": item.get("evidence"),
                # strict: every verifier is required. standard: only the
                # obligations the planner marked consequential.
                "required": True if mode == "strict" else bool(consequential),
            }
        )
    return records, defects


def _tests_field(
    records: Sequence[Dict[str, Any]], spec: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Reconcile any recorded test count against an actual run. Run wins."""
    from .verifier_pipeline import _pytest_passed_count

    candidates: List[Dict[str, Any]] = []
    recorded = (spec.get("budget") or {}).get("expected_tests_passed")
    if recorded is not None:
        candidates.append(
            {"value": recorded, "source": "recorded", "observed_at": spec.get("created_at")}
        )
    command_ref = None
    for rec in records:
        command = str(rec.get("command") or "")
        if "pytest" not in command:
            continue
        counted = _pytest_passed_count(str(rec.get("output") or ""))
        if counted <= 0:
            continue
        command_ref = command
        candidates.append(
            {"value": counted, "source": "command", "observed_at": time.time()}
        )
    winner = pick_authority(candidates)
    if winner is None:
        return None
    return {
        "passed": winner["value"],
        "authority": winner["source"],
        "command": command_ref,
        "candidates": candidates,
    }


def build_envelope(
    workspace: Union[str, Path],
    *,
    pipeline: Optional[Dict[str, Any]] = None,
    blocker: Optional[str] = None,
    aborted_reason: Optional[str] = None,
    started_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Compose the delegation envelope. Every claim routes through classify_claim."""
    ws = Path(workspace)
    spec, spec_defect = _read_json(spec_path(ws))
    state, state_defect = _read_json(mission_state_path(ws))
    spec_ok = isinstance(spec, dict)
    spec = spec if spec_ok else {}
    state = state if isinstance(state, dict) else {}

    defects: List[str] = [d for d in (spec_defect, state_defect) if d]
    if spec_defect is None and not spec_ok:
        defects.append("delegation spec is not an object")

    if pipeline is None:
        stored, _ = _read_json(mission_dir(ws) / "pipeline_result.json")
        pipeline = stored if isinstance(stored, dict) else None

    mode = str(spec.get("verification") or "standard").lower()
    records, record_defects = _verifier_records(pipeline, mode)
    defects.extend(record_defects)

    verified: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    refutations: List[Dict[str, Any]] = []
    remaining: List[str] = []

    def file_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """classify_claim's answer decides the bucket. Nothing else does."""
        (verified if entry["class"] == "verified" else hypotheses).append(entry)
        return entry

    for rec in records:
        statement = rec.get("statement") or rec.get("obligation_id") or "(unnamed claim)"
        command = str(rec.get("command") or "").strip()
        art = (
            Artifact(
                kind="command",
                ref=command,
                exit_code=rec.get("exit_code"),
                output=rec.get("output"),
                observed_at=state.get("last_checkpoint"),
            )
            if command
            else None
        )
        outcome = verifier_outcome(rec)
        if outcome == "passed":
            file_entry(classify_claim(statement, art))
            continue
        if outcome == "failed":
            # The artifact supports the REFUTATION, not the statement. Here the
            # nonzero exit IS the evidence, so it is declared expected -- the one
            # place that is legitimate, and it is stated rather than defaulted.
            ref_art = (
                replace(art, expected_exit=art.exit_code)
                if isinstance(art, Artifact) and isinstance(art.exit_code, int)
                and not isinstance(art.exit_code, bool)
                else art
            )
            refutations.append(
                classify_claim(
                    f"verifier for {statement!r} did not pass "
                    f"(verdict={rec.get('verdict')}, exit_code={rec.get('exit_code')})",
                    ref_art,
                )
            )
            entry = file_entry(classify_claim(statement, None))
            entry["reason"] = (
                f"verifier FAILED: verdict={rec.get('verdict')} "
                f"exit_code={rec.get('exit_code')}"
            )
            continue
        entry = file_entry(classify_claim(statement, None))
        entry["reason"] = "verifier UNVERIFIABLE: no command settled it"
        remaining.append(statement)

    # Model prose. There is no wording that promotes this.
    # isinstance, not `pipeline or {}`: a truthy non-dict (a list, a bare
    # string) reached .get() and raised, throwing away the defect
    # _verifier_records had already recorded for exactly that input.
    answer = pipeline.get("answer") if isinstance(pipeline, dict) else None
    if answer:
        file_entry(classify_claim(str(answer), None))

    # Artifacts on disk. Presence is NOT verification (adversarial test B).
    artifacts: List[Dict[str, Any]] = []
    for declared in spec.get("output_contract") or []:
        path = Path(declared)
        if not path.is_absolute():
            path = ws / declared
        artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "verified": any(
                    str(path) in str(v.get("artifact", {}).get("ref") or "")
                    for v in verified
                ),
                "note": "presence on disk is not verification",
            }
        )

    cancel, _ = _read_json(cancel_path(ws))
    if aborted_reason is None and isinstance(cancel, dict):
        aborted_reason = cancel.get("reason")

    # BEFORE decide_verdict, not after. These used to be appended below the
    # verdict, which left decide_verdict's `if remaining: INCONCLUSIVE` rule
    # dead for them: an envelope could carry verdict ACCEPT and a
    # remaining_uncertainty naming work that had never run.
    for uid, payload in (state.get("units") or {}).items():
        if isinstance(payload, dict) and payload.get("status") in ("pending", "ready"):
            remaining.append(f"work unit {uid} still {payload.get('status')}")

    effective_blocker = blocker
    if effective_blocker is None and aborted_reason:
        effective_blocker = f"aborted: {aborted_reason}"
    if effective_blocker is None:
        # A BLOCKED verdict must always be able to say what blocked it.
        failed_required = [
            str(r.get("obligation_id") or r.get("statement") or "?")
            for r in records
            if r.get("required") and verifier_outcome(r) == "failed"
        ]
        if failed_required:
            effective_blocker = "required verifier(s) failed: " + ", ".join(
                failed_required
            )

    verdict = (
        "ABORTED"
        if aborted_reason
        else decide_verdict(
            verified_facts=verified,
            verifier_records=records,
            blocker=effective_blocker,
            defects=defects,
            remaining=remaining,
        )
    )

    fingerprint_now = _fingerprint(spec.get("output_contract") or [], ws)
    fingerprint_start = spec.get("fingerprint_at_start")
    if fingerprint_now is None or fingerprint_start is None:
        changed_files = None
    else:
        changed_files = {
            "fingerprint_at_start": fingerprint_start,
            "fingerprint_now": fingerprint_now,
            "changed": fingerprint_now != fingerprint_start,
            "paths": list(spec.get("output_contract") or []),
        }

    lock = MutationLock(ws)
    lock_record = lock.read()
    receipts = [
        str(p)
        for p in (spec_path(ws), mission_state_path(ws), envelope_path(ws))
        if p.is_file()
    ]

    truncated = any(
        (entry.get("artifact") or {}).get("output_truncated")
        for entry in list(verified) + list(refutations)
    )

    claim_source = pick_authority(
        [
            {
                "value": spec.get("objective"),
                "source": "receipt",
                "observed_at": spec.get("created_at"),
            },
            {
                "value": state.get("goal"),
                "source": "disk",
                "observed_at": state.get("last_checkpoint"),
            },
        ]
    )

    return {
        "envelope_version": ENVELOPE_VERSION,
        "mission_id": spec.get("mission_id") or state.get("id"),
        "workspace": str(ws),
        "state": state.get("phase"),
        "verdict": verdict,
        "claim": (claim_source or {}).get("value"),
        "verified_facts": verified,
        "hypotheses": hypotheses,
        "physical_measurements": None,
        "artifacts": artifacts,
        "changed_files": changed_files,
        "tests": _tests_field(records, spec),
        "negative_controls": None,
        "mutation_controls": None,
        "failed_attempts_that_change_interpretation": None,
        "refutations": refutations,
        "resource_usage": {
            "writer_pid": (lock_record or {}).get("pid"),
            "writer_alive": lock.holder_is_live(lock_record) if lock_record else False,
            "started_at": started_at or spec.get("created_at"),
            "elapsed_s": (
                time.time() - float(spec["created_at"])
                if spec.get("created_at")
                else None
            ),
            "allowed_resources": spec.get("allowed_resources") or None,
        },
        "blocker": effective_blocker,
        "remaining_uncertainty": remaining or None,
        "recommended_next_action": _next_action(verdict, effective_blocker, remaining),
        "receipt_paths": receipts,
        "defects": defects or None,
        "truncated": bool(truncated),
        "built_at": time.time(),
    }


def _next_action(
    verdict: str, blocker: Optional[str], remaining: Sequence[str]
) -> str:
    if verdict == "ACCEPT":
        return "consume the verified_facts; re-run the named commands to re-check"
    if verdict == "ABORTED":
        return "inspect the state file, then start a fresh mission if still needed"
    if verdict == "BLOCKED":
        return f"clear the blocker, then steer or re-run: {blocker or 'unnamed blocker'}"
    if remaining:
        return "steer the mission with what is missing, or re-run with --verify strict"
    return "treat nothing here as established; supply a deterministic verifier"


# ---------------------------------------------------------------------------
# The executor. Model access is behind ONE seam so tests stay offline.
# ---------------------------------------------------------------------------


_DELEGATE_MAX_TOKENS = int(os.environ.get("HCLI_DELEGATE_MAX_TOKENS", "4096"))


class ModelRunaway(RuntimeError):
    """The model produced no content because it spent its budget reasoning."""


def default_caller(endpoint: str, model: Optional[str] = None) -> Callable[..., Any]:
    """Real model seam: an OpenAI-compatible endpoint via backends._post_json.

    Point ``HCLI_ENDPOINT`` at whichever server is up — the 1B llama-server on
    :8080 or the resident 27B from ``~/models/serve-abliterated.sh``. No server
    up is a BLOCKED envelope naming the connection error, never a fake ACCEPT.
    """
    from .backends import (
        _post_json,
        completion_from_openai,
        make_structured_output_contract,
    )

    def caller(prompt: str, *, schema: Optional[dict] = None) -> Any:
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            # THINKING OFF, and this is not a preference. Measured against the
            # resident Qwen3.8-27B on mlx_lm.server: with thinking on, the plan
            # call came back finish_reason="length" having spent the whole budget
            # in a SEPARATE `message.reasoning` field and never emitting
            # `content` at all -- which arrived here as "" and surfaced as the
            # misleading `PlanError: planner returned no obligations`. With
            # enable_thinking=False the same prompt returns clean content and
            # zero reasoning. The rest of HCLI already does this
            # (backends.py MlxServerBackend.chat_template_args, and
            # config.enable_thinking defaults False); this seam bypassed the
            # backend via _post_json and so bypassed the setting too.
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": _DELEGATE_MAX_TOKENS,
        }
        if model:
            payload["model"] = model
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "delegation", "schema": schema},
            }

        def _complete(body, timeout=None):
            raw = _post_json(endpoint, body, float(timeout or 180.0), "delegate")
            return completion_from_openai(raw, [])

        # STRUCTURED OUTPUT GOES THROUGH THE EXISTING CONTRACT, not a bare post.
        # mlx_lm.server has NO response_format and NO grammar (backends.py says so
        # in its capability table), so a raw post just sends a field the server
        # ignores and the model answers in prose. Measured: the resident 27B
        # returned a markdown essay with LaTeX, `json.loads` raised, and the
        # planner reported the misleading `PlanError: planner returned no
        # obligations`. make_structured_output_contract is the machinery this repo
        # already has for exactly that backend: strip the field, inject the schema
        # instruction, VALIDATE every reply, retry a bounded number of times with
        # the rejection reason appended, then raise StructuredOutputExhausted --
        # never a silent pass.
        if schema:
            contract = make_structured_output_contract(None, schema)
            if contract is not None:
                completion = contract.enforce(_complete, payload, 180.0)
                parsed = (completion.raw or {}).get("_structured") \
                    if isinstance(completion.raw, dict) else None
                if parsed is not None:
                    return parsed
                try:
                    return json.loads(completion.text or "")
                except json.JSONDecodeError:
                    return completion.text or ""

        data = _post_json(endpoint, payload, 180.0, "delegate")
        completion = completion_from_openai(data, [])
        text = completion.text or ""

        # A runaway must be NAMED, not returned as an empty string. An empty
        # content with a populated reasoning field means the model spent its
        # budget thinking; silently passing "" downstream is what turned a
        # diagnosable runaway into an unrelated-looking planner error.
        if not text.strip():
            choices = (data or {}).get("choices") or [{}]
            msg = (choices[0] or {}).get("message") or {}
            reasoning = msg.get("reasoning") or ""
            finish = (choices[0] or {}).get("finish_reason")
            if reasoning or finish == "length":
                raise ModelRunaway(
                    f"model returned no content (finish_reason={finish!r}); "
                    f"{len(reasoning)} chars landed in `message.reasoning`. The "
                    f"budget was spent thinking. chat_template_kwargs."
                    f"enable_thinking=False is already set here, so this server "
                    f"is ignoring it -- check that it supports the field."
                )
        if schema:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text

    return caller


def shell_runner(
    workspace: Union[str, Path], protected_paths: Sequence[str] = ()
) -> Callable[[str], Tuple[int, str]]:
    """Run a verifier command. Refuses anything naming a protected path.

    ponytail: substring match on the protected path, not path resolution. A
    command that reaches a protected file by symlink or `cd ..` is not caught;
    upgrade to resolving argv paths if the blast radius ever justifies it.
    """
    guarded = [p for p in protected_paths if str(p).strip()]

    def runner(command: str) -> Tuple[int, str]:
        for path in guarded:
            if str(path) in command:
                return 126, f"REFUSED: command names protected path {path!r}"
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    return runner


def _goal_with_steers(objective: str, ws: Path) -> str:
    """Absorb pending steers into the goal handed to the pipeline."""
    queue = None
    try:
        queue = _steering_queue(ws)
        pending = queue.pending()
    except Exception:
        return objective
    if not pending:
        return objective
    lines = [f"[steer {e.kind}] {e.text}" for e in pending]
    queue.apply_pending()
    return objective + "\n\nOPERATOR STEERS (apply to remaining work):\n" + "\n".join(lines)


def execute_mission(
    workspace: Union[str, Path],
    *,
    caller: Optional[Callable[..., Any]] = None,
    run_command: Optional[Callable[[str], Tuple[int, str]]] = None,
) -> Dict[str, Any]:
    """Do the delegated work and write the envelope. Holds the writer lock."""
    from .verifier_pipeline import run_pipeline

    ws = Path(workspace)
    spec, defect = _read_json(spec_path(ws))
    lock = MutationLock(ws)
    mission_id = (spec or {}).get("mission_id") if isinstance(spec, dict) else None
    if not lock.acquire(str(mission_id or ws.name)):
        held = lock.read() or {}
        raise DelegationBusy(
            f"{ws} is held by pid={held.get('pid')} unit={held.get('unit_id')}"
        )
    started = time.time()
    pipeline: Optional[Dict[str, Any]] = None
    blocker: Optional[str] = None
    try:
        if not isinstance(spec, dict):
            blocker = defect or "delegation spec is not an object"
        elif cancel_path(ws).is_file():
            blocker = "cancel requested before execution started"
        else:
            goal = _goal_with_steers(str(spec.get("objective") or ""), ws)
            caller = caller or default_caller(
                str(spec.get("endpoint") or DEFAULT_ENDPOINT)
            )
            run_command = run_command or shell_runner(
                ws, spec.get("protected_paths") or []
            )
            try:
                raw = run_pipeline(goal, caller, run_command)
                pipeline = {
                    "goal": raw.get("goal"),
                    "answer": raw.get("answer"),
                    "verdicts": [vars(v) for v in raw.get("verdicts") or []],
                    "obligations": [vars(o) for o in raw.get("obligations") or []],
                }
                atomic_write_json(mission_dir(ws) / "pipeline_result.json", pipeline)
            except Exception as exc:
                blocker = f"{type(exc).__name__}: {exc}"
        envelope = build_envelope(
            ws, pipeline=pipeline, blocker=blocker, started_at=started
        )
        phase = "completed" if envelope["verdict"] == "ACCEPT" else "failed"
        # state.json and the envelope must not disagree about the phase.
        state, _ = _read_json(mission_state_path(ws))
        if isinstance(state, dict):
            state["phase"] = phase
            state["last_checkpoint"] = time.time()
            atomic_write_json(mission_state_path(ws), state)
            envelope["state"] = phase
        atomic_write_json(envelope_path(ws), envelope)
        return envelope
    finally:
        lock.release(str(mission_id or ws.name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(prog: str = "hcli") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="HCLI delegation surface"
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p_run = sub.add_parser("run", help="start a delegated mission (returns at once)")
    p_run.add_argument("--goal", required=True, help="objective for the mission")
    p_run.add_argument("--verify", choices=list(VERIFY_MODES), default="standard")
    p_run.add_argument("--budget", default=None, help="JSON object, e.g. '{\"minutes\":30}'")
    p_run.add_argument("--resources", action="append", default=[])
    p_run.add_argument("--protect", action="append", default=[])
    p_run.add_argument("--constraint", action="append", default=[])
    p_run.add_argument("--expect", action="append", default=[], help="output contract path")
    p_run.add_argument("--root", default=None, help="delegations root (default: cwd)")
    p_run.add_argument("--no-spawn", action="store_true", help="write the spec, run nothing")
    p_run.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="durable snapshot")
    p_status.add_argument("mission")
    p_status.add_argument("--root", default=None)
    p_status.add_argument("--json", action="store_true")

    p_steer = sub.add_parser("steer", help="queue guidance for future work")
    p_steer.add_argument("mission")
    p_steer.add_argument("text")
    p_steer.add_argument("--kind", choices=list(STEER_KINDS), default="knowledge")
    p_steer.add_argument("--root", default=None)
    p_steer.add_argument("--json", action="store_true")

    p_result = sub.add_parser("result", help="the evidence envelope")
    p_result.add_argument("mission")
    p_result.add_argument("--root", default=None)
    p_result.add_argument("--json", action="store_true")

    p_abort = sub.add_parser("abort", help="stop a delegation")
    p_abort.add_argument("mission")
    p_abort.add_argument("--reason", default="aborted by operator")
    p_abort.add_argument("--root", default=None)
    p_abort.add_argument("--json", action="store_true")
    return parser


def _render_result(env: Dict[str, Any]) -> str:
    lines = [
        f"mission {env.get('mission_id')}  state={env.get('state')}  "
        f"VERDICT={env.get('verdict')}",
    ]
    facts = env.get("verified_facts") or []
    lines.append(f"VERIFIED ({len(facts)}) — each backed by a deterministic artifact")
    for item in facts:
        art = item.get("artifact") or {}
        lines.append(f"  [VERIFIED] {item.get('claim')}")
        lines.append(f"             {art.get('kind')}: {art.get('ref')} "
                     f"(exit={art.get('exit_code')})")
    if not facts:
        lines.append("  (none)")
    hyps = env.get("hypotheses") or []
    lines.append(f"HYPOTHESIS ({len(hyps)}) — nothing here is established")
    for item in hyps:
        claim, _ = _clip(item.get("claim"), 300)
        lines.append(f"  [HYPOTHESIS] {claim}")
        if item.get("reason"):
            lines.append(f"               why not verified: {item['reason']}")
    for key in ("blocker", "remaining_uncertainty", "defects", "tests"):
        if env.get(key):
            lines.append(f"{key}: {json.dumps(env[key], default=str)[:600]}")
    if env.get("truncated"):
        lines.append("truncated: some artifact output was clipped")
    lines.append(f"next: {env.get('recommended_next_action')}")
    return "\n".join(lines)


def cli_main(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv))
    try:
        if args.verb == "run":
            budget = json.loads(args.budget) if args.budget else None
            out = run(
                args.goal,
                root=args.root,
                spawn=not args.no_spawn,
                verification=args.verify,
                budget=budget,
                allowed_resources=args.resources,
                protected_paths=args.protect,
                constraints=args.constraint,
                output_contract=args.expect,
            )
            print(json.dumps(out, indent=2) if args.json else out["mission_id"])
            return 0
        if args.verb == "status":
            out = status(args.mission, args.root)
            print(json.dumps(out, indent=2) if args.json else _render_status(out))
            return 0
        if args.verb == "steer":
            out = steer(args.mission, args.text, args.kind, args.root)
            print(json.dumps(out, indent=2) if args.json else f"steer {out['steer_id']} queued ({out['kind']})")
            return 0
        if args.verb == "result":
            out = result(args.mission, args.root)
            print(json.dumps(out, indent=2, default=str) if args.json else _render_result(out))
            return 0 if out.get("verdict") == "ACCEPT" else 2
        if args.verb == "abort":
            out = abort(args.mission, args.reason, args.root)
            print(json.dumps(out, indent=2) if args.json else
                  f"aborted {out['mission_id']} (lock_free={out['lock_free']})")
            return 0
    except DelegationError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    parser.error(f"unknown verb {args.verb!r}")
    return 3


def _render_status(snap: Dict[str, Any]) -> str:
    writer = snap.get("writer") or {}
    lines = [
        f"mission {snap.get('mission_id')}  phase={snap.get('phase')}",
        f"objective: {snap.get('objective')}",
        f"writer: pid={writer.get('pid')} alive={writer.get('alive')}",
        f"units: {snap.get('units_by_status') or {}}",
        f"steers pending: {snap.get('steers_pending')}",
        f"envelope: {snap.get('envelope_present')} verdict={snap.get('verdict')}",
    ]
    if snap.get("cancel_requested"):
        lines.append(f"cancel requested: {snap.get('cancel_reason')}")
    if snap.get("defects"):
        lines.append(f"defects: {snap['defects']}")
    return "\n".join(lines)


def exec_main(argv: Sequence[str]) -> int:
    """Entry point for the spawned worker (``hcli __delegate_exec <workspace>``)."""
    if not argv:
        print("usage: hcli __delegate_exec <workspace>", file=sys.stderr)
        return 2
    envelope = execute_mission(argv[0])
    print(json.dumps({"verdict": envelope.get("verdict")}))
    return 0 if envelope.get("verdict") == "ACCEPT" else 2
