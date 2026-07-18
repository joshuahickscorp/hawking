#!/usr/bin/env python3.12
"""One-use, bound, tamper-tested supersession transition for the successor control plane.

Master goal 2.2 + 5.5: the moment the legacy Doctor-v5 campaign reaches its signed
release boundary, control must hand off to the successor exactly once, and only if every
identity that the handoff was authorized against still matches the live world. This module
is that hand-off. It builds a ONE-USE transition intent, verifies it against live state,
runs a fail-closed gate that reuses `eco_activation.supersession_gate`, and executes the
transition at most once (append-only consumed receipt), with a rollback target.

A transition intent binds, immutably (sha256 self-seal), all of:
  - the exact legacy campaign plan hash it was authorized against;
  - the exact accepted reporter checkpoint identities;
  - the exact successor code commit + tree identity;
  - the exact successor config + queue-root identity;
  - the expected legacy terminal cell count;
  - the required quiescent state;
  - an operator authorization statement;
  - a generation number + expiry that prevent reuse;
  - a rollback target.

Authorization trust model (LABELED honestly): the preferred authorization is a real
detached signature over an operator key. When that is unavailable, the control is an
out-of-band, permission-restricted authorization file plus this object's sha256 self-seal.
That combination proves INTEGRITY (the intent was not altered after creation and binds
these exact identities) and POSSESSION (whoever placed the out-of-band file could write to
the restricted path), NOT cryptographic AUTHENTICITY. A forged authorization still cannot
execute a running campaign, because the supersession gate independently re-reads the live
campaign queue from disk and requires genuine terminal + reporter-sealed + quiescent state.

Non-interference: this module reads the campaign root read-only and writes only under
`reports/condense/event_horizon_successor/transition/`. It never signals or adopts a live
pid, never launches heavy compute, and never imports campaign-owned gitignored runtime.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, hash_value, now_iso, atomic_write_json, read_json_safe,
    is_sha256, repo_root,
)
from eco_activation import (  # noqa: E402
    ActivationConfig, supersession_gate, make_signature, TERMINAL_STATUSES,
)

SCHEMA_TRANSITION_INTENT = "hawking.successor.transition_intent.v1"
SCHEMA_TRANSITION_MANIFEST = "hawking.successor.transition_manifest.v1"
SCHEMA_TRANSITION_CONSUMED = "hawking.successor.transition_consumed.v1"

# The bound successor + legacy identity fields compared against the live manifest.
_LEGACY_FIELDS = ("legacy_plan_sha256", "expected_terminal_count")
_SUCCESSOR_FIELDS = ("successor_commit", "successor_tree_sha256",
                     "successor_config_sha256", "successor_queue_root_sha256")


class TransitionError(EcoError):
    """Fail-closed error in the supersession transition."""


def successor_state_root() -> Path:
    """The successor-only state namespace. Deliberately disjoint from the campaign
    namespace (`reports/condense/doctor_v5_ultra`) and from the eco layer
    (`reports/condense/frontier_eco`)."""
    return repo_root() / "reports" / "condense" / "event_horizon_successor"


@dataclasses.dataclass(frozen=True)
class TransitionConfig:
    campaign_root: Path
    state_root: Path

    @property
    def transition_dir(self) -> Path:
        return self.state_root / "transition"

    @property
    def manifest_path(self) -> Path:
        return self.transition_dir / "current.json"

    @property
    def previous_path(self) -> Path:
        return self.transition_dir / "previous.json"

    @property
    def consumed_dir(self) -> Path:
        return self.transition_dir / "consumed"

    def consumed_path(self, intent_sha256: str) -> Path:
        # keyed by the intent's self-seal so the same intent can be consumed at most once,
        # durably, across process invocations
        return self.consumed_dir / f"{intent_sha256}.json"

    def activation_cfg(self, expected_plan_sha256: str) -> ActivationConfig:
        return ActivationConfig(campaign_root=self.campaign_root, state_root=self.state_root,
                                expected_plan_sha256=expected_plan_sha256)


def default_config(campaign_root: str | os.PathLike[str] | None = None,
                   state_root: str | os.PathLike[str] | None = None) -> TransitionConfig:
    croot = Path(campaign_root) if campaign_root else \
        repo_root() / "reports" / "condense" / "doctor_v5_ultra"
    sroot = Path(state_root) if state_root else successor_state_root()
    return TransitionConfig(campaign_root=croot, state_root=sroot)


# ── checkpoint identity helpers ────────────────────────────────────────────────────────
def _checkpoint_shas(checkpoints: Any) -> list[str]:
    """Extract the accepted reporter checkpoint sha256 identities, sorted + deduped.

    A checkpoint value is either a sha256 string, or a dict carrying a sha256 self-seal
    (checkpoint_sha256 / state_sha256 / report_checkpoint_sha256). Anything else is ignored.
    """
    out: set[str] = set()
    if isinstance(checkpoints, dict):
        for value in checkpoints.values():
            if is_sha256(value):
                out.add(value)
            elif isinstance(value, dict):
                for k in ("checkpoint_sha256", "state_sha256", "report_checkpoint_sha256"):
                    if is_sha256(value.get(k)):
                        out.add(value[k])
    elif isinstance(checkpoints, (list, tuple)):
        for value in checkpoints:
            if is_sha256(value):
                out.add(value)
    return sorted(out)


# ── successor identity measurement (real path; injected in selftest) ───────────────────
def measure_successor_identity(successor_repo: str | os.PathLike[str] | None = None,
                               config_path: str | os.PathLike[str] | None = None,
                               queue_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Best-effort live successor identity: git commit + tree sha, config sha, queue-root sha.

    Real and environment-dependent (shells out to git, hashes files). The selftest never
    calls this; it injects a synthetic identity instead. Missing pieces come back as None
    so the caller (verify_intent) fails closed on a mismatch rather than crashing.
    """
    repo = Path(successor_repo) if successor_repo else repo_root()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo), text=True,
                                 capture_output=True, timeout=30)
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    config_sha = None
    if config_path and Path(config_path).is_file():
        try:
            config_sha = hash_value(read_json_safe(config_path))
        except EcoError:
            config_sha = None
    queue_root_sha = None
    if queue_root and Path(queue_root).is_dir():
        names = sorted(p.name for p in Path(queue_root).iterdir())
        queue_root_sha = hash_value({"queue_root": str(queue_root), "entries": names})
    return {
        "successor_commit": commit,
        # git tree object ids are sha1 (40 hex); we bind them verbatim as identity strings
        "successor_tree_sha256": tree,
        "successor_config_sha256": config_sha,
        "successor_queue_root_sha256": queue_root_sha,
    }


# ── intent construction ────────────────────────────────────────────────────────────────
def make_intent(*, legacy_plan_sha256: str,
                accepted_checkpoints: Any,
                successor_commit: str | None,
                successor_tree_sha256: str | None,
                successor_config_sha256: str | None,
                successor_queue_root_sha256: str | None,
                expected_terminal_count: int,
                rollback_target: str,
                authorization_statement: str,
                generation: int,
                operator_signature: dict[str, Any] | None = None,
                authorization_file: str | os.PathLike[str] | None = None,
                require_quiescent: bool = True,
                ttl_seconds: int = 3600,
                expires_at: str | None = None) -> dict[str, Any]:
    """Build a sealed, one-use transition intent binding every handoff identity.

    Exactly one authorization control must be present:
      - `operator_signature`: an `eco_activation.make_signature` object (preferred: a real
        detached signature would slot in here);
      - or `authorization_file`: a permission-restricted out-of-band authorization file,
        whose sha256 is bound. LABELED as integrity/possession control, not authenticity.
    """
    if not is_sha256(legacy_plan_sha256):
        raise TransitionError("legacy_plan_sha256 must be a sha256")
    if not isinstance(expected_terminal_count, int) or expected_terminal_count < 0:
        raise TransitionError("expected_terminal_count must be a non-negative int")
    if not isinstance(generation, int) or generation < 0:
        raise TransitionError("generation must be a non-negative int")

    authz_kind: str
    operator_signature_sha256: str | None = None
    authorization_file_sha256: str | None = None
    if operator_signature is not None:
        if not isinstance(operator_signature, dict) or not sealed(operator_signature, "signature_sha256"):
            raise TransitionError("operator_signature must be a sealed supersession signature")
        if operator_signature.get("plan_sha256") != legacy_plan_sha256:
            raise TransitionError("operator_signature binds a different plan_sha256")
        operator_signature_sha256 = operator_signature["signature_sha256"]
        authz_kind = "detached_signature_or_self_sealed_supersession_signature"
    elif authorization_file is not None:
        p = Path(authorization_file)
        if not p.is_file():
            raise TransitionError(f"authorization_file absent: {p}")
        authorization_file_sha256 = hash_value({"path": str(p), "bytes_sha256":
                                                _sha_file_hex(p)})
        authz_kind = "out_of_band_file_plus_self_seal"
    else:
        raise TransitionError("an operator_signature or an authorization_file is required")

    created = now_iso()
    if expires_at is None:
        exp_dt = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=ttl_seconds)
        expires_at = exp_dt.isoformat(timespec="seconds")

    intent = {
        "schema": SCHEMA_TRANSITION_INTENT,
        "generation": generation,
        "created_at": created,
        "expires_at": expires_at,
        "consumed": False,
        # legacy binding
        "legacy_plan_sha256": legacy_plan_sha256,
        "expected_terminal_count": expected_terminal_count,
        "accepted_checkpoints": _checkpoint_shas(accepted_checkpoints),
        "require_quiescent": bool(require_quiescent),
        # successor binding
        "successor_commit": successor_commit,
        "successor_tree_sha256": successor_tree_sha256,
        "successor_config_sha256": successor_config_sha256,
        "successor_queue_root_sha256": successor_queue_root_sha256,
        # authorization (see module docstring for the trust label)
        "authorization": {
            "kind": authz_kind,
            "statement": authorization_statement,
            "operator_signature_sha256": operator_signature_sha256,
            "authorization_file": str(authorization_file) if authorization_file else None,
            "authorization_file_sha256": authorization_file_sha256,
            "trust_label": ("integrity + possession control, NOT cryptographic authenticity "
                            "unless the operator_signature is a real detached signature"),
        },
        "rollback_target": rollback_target,
    }
    return seal_field(intent, "intent_sha256")


def _sha_file_hex(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# ── live manifest + verification ───────────────────────────────────────────────────────
def build_live_manifest(campaign_root: str | os.PathLike[str],
                        successor_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive the live comparison manifest. The LEGACY fields are read straight from the
    campaign `queue_state.json` on disk (a caller cannot fake them); the successor identity
    is measured live (injected as `successor_identity` in tests)."""
    qs = read_json_safe(Path(campaign_root) / "queue_state.json")
    cells = qs.get("cells", {})
    terminal = sum(1 for r in cells.values()
                   if isinstance(r, dict) and r.get("status") in TERMINAL_STATUSES) \
        if isinstance(cells, dict) else 0
    live = {
        "legacy_plan_sha256": qs.get("plan_sha256"),
        "expected_terminal_count": terminal,
        "accepted_checkpoints": _checkpoint_shas(qs.get("report_checkpoints")),
        "now": now_iso(),
    }
    sid = successor_identity if successor_identity is not None else measure_successor_identity()
    for field in _SUCCESSOR_FIELDS:
        live[field] = sid.get(field)
    return live


def _is_expired(intent: dict[str, Any], now: str | None = None) -> bool:
    exp = intent.get("expires_at")
    if not isinstance(exp, str):
        return True  # fail closed: no expiry means not usable
    try:
        exp_dt = _dt.datetime.fromisoformat(exp)
        now_dt = _dt.datetime.fromisoformat(now) if now else _dt.datetime.now(_dt.timezone.utc)
    except ValueError:
        return True
    if exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=_dt.timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=_dt.timezone.utc)
    return now_dt > exp_dt


def verify_intent(intent: dict[str, Any], live_manifest: dict[str, Any]) -> dict[str, Any]:
    """Check the self-seal, every bound identity against live values, expiry and consumed.

    Returns {ok, reasons}. Fail-closed: any missing/mismatched field is a reason."""
    reasons: list[str] = []
    if not isinstance(intent, dict) or intent.get("schema") != SCHEMA_TRANSITION_INTENT:
        return {"ok": False, "reasons": ["not a transition intent object"]}
    if not sealed(intent, "intent_sha256"):
        reasons.append("intent self-seal invalid (tampered)")
    if intent.get("consumed") is not False:
        reasons.append("intent already marked consumed")
    if not isinstance(intent.get("generation"), int):
        reasons.append("intent generation is not an int")
    if _is_expired(intent, live_manifest.get("now")):
        reasons.append(f"intent expired at {intent.get('expires_at')}")

    # legacy identity binding (live values are disk-derived and cannot be faked by a caller)
    if intent.get("legacy_plan_sha256") != live_manifest.get("legacy_plan_sha256"):
        reasons.append("legacy_plan_sha256 does not match live queue")
    if intent.get("expected_terminal_count") != live_manifest.get("expected_terminal_count"):
        reasons.append(f"expected_terminal_count {intent.get('expected_terminal_count')} "
                       f"!= live {live_manifest.get('expected_terminal_count')}")
    bound_cps = intent.get("accepted_checkpoints")
    live_cps = live_manifest.get("accepted_checkpoints")
    if live_cps is not None and bound_cps != live_cps:
        reasons.append("accepted_checkpoints identities do not match live reporter checkpoints")

    # successor identity binding
    for field in _SUCCESSOR_FIELDS:
        bound = intent.get(field)
        live = live_manifest.get(field)
        if bound is None:
            reasons.append(f"{field} not bound in intent")
        elif live is None:
            reasons.append(f"{field} could not be measured live")
        elif bound != live:
            reasons.append(f"{field} mismatch (intent {bound} != live {live})")

    # authorization presence (integrity/possession control; see docstring)
    authz = intent.get("authorization")
    if not isinstance(authz, dict) or not (
            is_sha256(authz.get("operator_signature_sha256"))
            or is_sha256(authz.get("authorization_file_sha256"))):
        reasons.append("no valid operator authorization bound (signature or oob file)")

    return {"ok": not reasons, "reasons": reasons}


# ── gate ────────────────────────────────────────────────────────────────────────────────
def evaluate_gate(campaign_root: str | os.PathLike[str], intent: dict[str, Any], *,
                  successor_identity: dict[str, Any] | None = None,
                  state_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Fail-closed handoff gate. Requires EVERY supersession condition (reused from
    `eco_activation.supersession_gate`: plan match, all cells terminal, reports sealed,
    checkpoints accepted, quiescent, operator signature valid) AND `verify_intent`.

    The gate independently re-reads live state from disk; it never accepts a caller-supplied
    all_pass or a caller-invented checkpoint."""
    plan = intent.get("legacy_plan_sha256")
    if not is_sha256(plan):
        return {"all_pass": False, "reasons": ["intent has no valid legacy_plan_sha256"],
                "supersession": None, "intent_verify": None}
    sroot = Path(state_root) if state_root else successor_state_root()
    cfg = TransitionConfig(campaign_root=Path(campaign_root), state_root=sroot)
    gate = supersession_gate(cfg.activation_cfg(plan))

    try:
        live = build_live_manifest(campaign_root, successor_identity)
        vi = verify_intent(intent, live)
    except EcoError as exc:
        vi = {"ok": False, "reasons": [f"live manifest unreadable: {exc}"]}

    reasons = list(gate.get("reasons", [])) + [f"intent: {r}" for r in vi["reasons"]]
    all_pass = bool(gate.get("all_pass")) and bool(vi["ok"])
    return {"all_pass": all_pass, "supersession": gate, "intent_verify": vi,
            "reasons": reasons}


# ── one-use execution + rollback ───────────────────────────────────────────────────────
def execute_transition(intent: dict[str, Any], campaign_root: str | os.PathLike[str],
                       go: bool, *, _test_identity_override: dict[str, Any] | None = None,
                       state_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Execute the supersession ONCE. Fail-closed and impossible to force:
      - it recomputes the gate itself (there is NO all_pass parameter to pass in);
      - it re-reads live campaign state from disk (an invented live manifest cannot pass the
        supersession portion, which reads queue_state directly);
      - it MEASURES the running successor identity (git tree/commit + config/queue-root hashes)
        rather than accepting it from the caller, so the bound-vs-running check is not vacuous;
        `_test_identity_override` is a test-only hook and must never be set in production;
      - it refuses if a consumed receipt for this exact intent already exists (durable
        one-use, append-only), or if `intent.consumed` is set, or if `go` is not True.
    On success it preserves the prior manifest as the rollback target, writes the activation
    manifest, and writes an append-only consumed receipt."""
    sroot = Path(state_root) if state_root else successor_state_root()
    cfg = TransitionConfig(campaign_root=Path(campaign_root), state_root=sroot)

    if not sealed(intent, "intent_sha256"):
        return {"executed": False, "refused": True, "reason": "intent self-seal invalid"}
    intent_sha = intent["intent_sha256"]

    # durable one-use guard: a consumed receipt for this intent means it already fired
    if cfg.consumed_path(intent_sha).exists():
        return {"executed": False, "refused": True,
                "reason": "intent already consumed (one-use receipt present)",
                "intent_sha256": intent_sha}
    if intent.get("consumed") is not False:
        return {"executed": False, "refused": True, "reason": "intent object marked consumed"}
    if not go:
        return {"executed": False, "refused": True, "reason": "go not set"}

    # production path MEASURES successor identity; only tests inject one
    gate = evaluate_gate(campaign_root, intent, successor_identity=_test_identity_override,
                         state_root=sroot)
    if not gate["all_pass"]:
        return {"executed": False, "refused": True, "reason": "gate not satisfied",
                "gate": gate}

    # atomic one-use CLAIM: close the TOCTOU between the exists-check above and the receipt
    # write below by creating the consumed receipt exclusively (O_EXCL). If a concurrent
    # execute already claimed this intent, refuse without side effects. The full receipt is
    # written over this claim at the end (only this process holds it).
    claim_path = cfg.consumed_path(intent_sha)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return {"executed": False, "refused": True,
                "reason": "intent consumed concurrently (exclusive claim lost)",
                "intent_sha256": intent_sha}
    os.write(_fd, b'{"status":"claimed"}\n')
    os.fsync(_fd)
    os.close(_fd)

    # preserve the pre-transition state as the rollback target
    if cfg.manifest_path.exists():
        atomic_write_json(cfg.previous_path, read_json_safe(cfg.manifest_path))
    else:
        pre = seal_field({"schema": SCHEMA_TRANSITION_MANIFEST, "active": False,
                          "note": "pre-transition default-off state (rollback target)",
                          "rollback_target": intent.get("rollback_target"),
                          "captured_at": now_iso()}, "manifest_sha256")
        atomic_write_json(cfg.previous_path, pre)

    manifest = {
        "schema": SCHEMA_TRANSITION_MANIFEST,
        "active": True,
        "generation": intent.get("generation"),
        "intent_sha256": intent_sha,
        "campaign_plan_sha256": intent.get("legacy_plan_sha256"),
        "successor_commit": intent.get("successor_commit"),
        "successor_tree_sha256": intent.get("successor_tree_sha256"),
        "successor_config_sha256": intent.get("successor_config_sha256"),
        "successor_queue_root_sha256": intent.get("successor_queue_root_sha256"),
        "rollback_target": intent.get("rollback_target"),
        "activated_at": now_iso(),
        "gate_supersession": gate["supersession"],
    }
    manifest = seal_field(manifest, "manifest_sha256")
    atomic_write_json(cfg.manifest_path, manifest)

    # append-only consumed receipt (durable one-use); its existence blocks re-execution
    receipt = seal_field({
        "schema": SCHEMA_TRANSITION_CONSUMED,
        "intent_sha256": intent_sha,
        "generation": intent.get("generation"),
        "manifest_sha256": manifest["manifest_sha256"],
        "consumed_at": now_iso(),
    }, "receipt_sha256")
    atomic_write_json(cfg.consumed_path(intent_sha), receipt)

    return {"executed": True, "refused": False, "intent_sha256": intent_sha,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(cfg.manifest_path)}


def rollback(campaign_root: str | os.PathLike[str] | None = None, *,
             state_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Restore the prior manifest (the rollback target captured at execute time), or return
    to explicit default-off when there is no prior manifest."""
    sroot = Path(state_root) if state_root else successor_state_root()
    croot = Path(campaign_root) if campaign_root else \
        repo_root() / "reports" / "condense" / "doctor_v5_ultra"
    cfg = TransitionConfig(campaign_root=croot, state_root=sroot)
    if cfg.previous_path.exists():
        prev = read_json_safe(cfg.previous_path)
        atomic_write_json(cfg.manifest_path, prev)
        return {"rolled_back": True, "restored": "previous_manifest",
                "active": bool(prev.get("active")),
                "manifest_sha256": prev.get("manifest_sha256")}
    if cfg.manifest_path.exists():
        off = seal_field({"schema": SCHEMA_TRANSITION_MANIFEST, "active": False,
                          "deactivated_at": now_iso()}, "manifest_sha256")
        atomic_write_json(cfg.manifest_path, off)
        return {"rolled_back": True, "restored": "default_off", "active": False}
    return {"rolled_back": False, "reason": "already default-off (no manifest)"}


def status(campaign_root: str | os.PathLike[str] | None = None, *,
           state_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    sroot = Path(state_root) if state_root else successor_state_root()
    croot = Path(campaign_root) if campaign_root else \
        repo_root() / "reports" / "condense" / "doctor_v5_ultra"
    cfg = TransitionConfig(campaign_root=croot, state_root=sroot)
    active = False
    manifest_sha = None
    if cfg.manifest_path.exists():
        try:
            m = read_json_safe(cfg.manifest_path)
            active = bool(m.get("active"))
            manifest_sha = m.get("manifest_sha256")
        except EcoError:
            active = False
    consumed = sorted(p.stem for p in cfg.consumed_dir.glob("*.json")) \
        if cfg.consumed_dir.exists() else []
    return {"schema": "hawking.successor.transition_status.v1", "active": active,
            "default_off": not active, "manifest_sha256": manifest_sha,
            "consumed_intents": consumed, "checked_at": now_iso()}


# ── selftest (fully offline; tempdirs + synthetic campaign + injected identities) ──────
def selftest() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        croot = base / "doctor_v5_ultra"
        sroot = base / "event_horizon_successor"
        croot.mkdir(parents=True)
        plan_sha = "a" * 64

        # synthetic reporter checkpoints and the successor identity we will bind + measure
        cp_sub, cp_120 = "b" * 64, "c" * 64
        succ_id = {
            "successor_commit": "e" * 40,
            "successor_tree_sha256": "f" * 40,
            "successor_config_sha256": "1" * 64,
            "successor_queue_root_sha256": "2" * 64,
        }

        def write_queue(all_terminal: bool, sealed_reports: bool) -> None:
            cells = {"c1": {"status": "complete"},
                     "c2": {"status": "complete" if all_terminal else "running"}}
            checkpoints = {"sub-120B": cp_sub if sealed_reports else None,
                           "120B": cp_120 if sealed_reports else None}
            atomic_write_json(croot / "queue_state.json",
                              {"plan_sha256": plan_sha, "cells": cells,
                               "report_checkpoints": checkpoints})

        # operator supersession signature, reused for both the gate and the intent binding
        sig = make_signature(plan_sha, signed_by="operator", statement="supersede after release")
        act_sig_path = ActivationConfig(campaign_root=croot, state_root=sroot,
                                        expected_plan_sha256=plan_sha).signature_path

        def make_valid_intent(*, expires_at: str | None = None) -> dict[str, Any]:
            return make_intent(
                legacy_plan_sha256=plan_sha,
                accepted_checkpoints={"sub-120B": cp_sub, "120B": cp_120},
                successor_commit=succ_id["successor_commit"],
                successor_tree_sha256=succ_id["successor_tree_sha256"],
                successor_config_sha256=succ_id["successor_config_sha256"],
                successor_queue_root_sha256=succ_id["successor_queue_root_sha256"],
                expected_terminal_count=2,
                rollback_target=str(sroot / "transition" / "previous.json"),
                authorization_statement="operator authorizes supersession at release boundary",
                generation=1,
                operator_signature=sig,
                expires_at=expires_at,
            )

        results: dict[str, Any] = {}

        # (1) running campaign -> gate fails, execute refuses
        write_queue(all_terminal=False, sealed_reports=False)
        intent = make_valid_intent()
        g1 = evaluate_gate(croot, intent, successor_identity=succ_id, state_root=sroot)
        if g1["all_pass"]:
            raise TransitionError("gate must fail on a running campaign")
        r1 = execute_transition(intent, croot, go=True, _test_identity_override=succ_id,
                                state_root=sroot)
        if r1["executed"]:
            raise TransitionError("execute must refuse on a running campaign")
        results["running_refused"] = True

        # bring the campaign to terminal + sealed + quiescent + signed
        write_queue(all_terminal=True, sealed_reports=True)
        atomic_write_json(act_sig_path, sig)

        # (2) terminal + sealed + signed + valid intent + go -> executes once
        intent2 = make_valid_intent()
        g2 = evaluate_gate(croot, intent2, successor_identity=succ_id, state_root=sroot)
        if not g2["all_pass"]:
            raise TransitionError(f"gate should pass now: {g2['reasons']}")
        r2 = execute_transition(intent2, croot, go=True, _test_identity_override=succ_id,
                                state_root=sroot)
        if not r2["executed"]:
            raise TransitionError(f"execute should succeed: {r2}")
        if not status(croot, state_root=sroot)["active"]:
            raise TransitionError("status should report active after execute")
        results["signed_go_executes_once"] = True

        # (3) second execute of the same intent -> refused (one-use consumed)
        r3 = execute_transition(intent2, croot, go=True, _test_identity_override=succ_id,
                                state_root=sroot)
        if r3["executed"]:
            raise TransitionError("second execute must refuse (one-use)")
        results["second_execute_refused"] = True

        # (4) tampered intent refused (flip a bound field without resealing)
        tampered = dict(intent2)
        tampered["successor_tree_sha256"] = "9" * 40  # seal no longer matches
        vt = verify_intent(tampered, build_live_manifest(croot, succ_id))
        if vt["ok"]:
            raise TransitionError("tampered intent must fail verify")
        rt = execute_transition(tampered, croot, go=True, _test_identity_override=succ_id,
                                state_root=sroot)
        if rt["executed"]:
            raise TransitionError("tampered intent must not execute")
        results["tampered_refused"] = True

        # (5) expired intent refused
        past = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=1)).isoformat(timespec="seconds")
        expired = make_valid_intent(expires_at=past)
        ve = verify_intent(expired, build_live_manifest(croot, succ_id))
        if ve["ok"]:
            raise TransitionError("expired intent must fail verify")
        re_ = execute_transition(expired, croot, go=True, _test_identity_override=succ_id,
                                 state_root=sroot)
        if re_["executed"]:
            raise TransitionError("expired intent must not execute")
        results["expired_refused"] = True

        # (6) rollback restores the prior (pre-transition) manifest -> default-off
        rb = rollback(croot, state_root=sroot)
        if not rb["rolled_back"]:
            raise TransitionError("rollback should succeed")
        if status(croot, state_root=sroot)["active"]:
            raise TransitionError("rollback must leave the layer inactive")
        results["rollback_restores"] = True

        # (7) a caller-supplied all_pass cannot bypass: execute takes NO such parameter, and
        #     even a fabricated matching live identity cannot pass while the campaign runs,
        #     because the supersession gate re-reads queue_state from disk.
        bypass_attempted = False
        try:
            execute_transition(intent2, croot, go=True, _test_identity_override=succ_id,
                               state_root=sroot, all_pass=True)  # type: ignore[call-arg]
        except TypeError:
            bypass_attempted = True  # no such knob exists
        if not bypass_attempted:
            raise TransitionError("execute_transition must not accept an all_pass override")
        write_queue(all_terminal=False, sealed_reports=False)  # campaign running again
        fresh_intent = make_valid_intent()
        forged_live = {  # a lying live manifest claiming everything is fine
            "legacy_plan_sha256": plan_sha, "expected_terminal_count": 2,
            "accepted_checkpoints": sorted([cp_sub, cp_120]), "now": now_iso(), **succ_id,
        }
        if verify_intent(fresh_intent, forged_live)["ok"] is not True:
            raise TransitionError("sanity: forged live should satisfy verify in isolation")
        rb7 = execute_transition(fresh_intent, croot, go=True, _test_identity_override=succ_id,
                                 state_root=sroot)
        if rb7["executed"]:
            raise TransitionError("running campaign must refuse regardless of caller inputs")
        results["all_pass_bypass_refused"] = True

    return {"ok": True, **results}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="One-use supersession transition (fail-closed).")
    ap.add_argument("--campaign-root", default=None)
    ap.add_argument("--state-root", default=None)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    sub.add_parser("rollback")
    args = ap.parse_args()
    if args.selftest or not args.cmd:
        if args.selftest:
            print(json.dumps(selftest(), indent=2, sort_keys=True))
            sys.exit(0)
    if args.cmd == "rollback":
        print(json.dumps(rollback(args.campaign_root, state_root=args.state_root),
                         indent=2, sort_keys=True))
    else:
        print(json.dumps(status(args.campaign_root, state_root=args.state_root),
                         indent=2, sort_keys=True))
