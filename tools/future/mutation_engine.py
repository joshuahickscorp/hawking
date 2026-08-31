"""MUTATION ENGINE — propose, apply, measure, and roll back a real change.

The frozen 1-hour autonomy trial produced 823 events and mutated nothing.
autonomy_run.py is a verifier: every unit is read-only analysis emitting a
receipt. A verifier cannot make Hawking faster, smaller, or better.

This module is the missing metabolism. It lets the resident propose a bounded
change, apply it inside a reversible scope, record before/after evidence with
an honest class, roll the world back, and return KEPT / ROLLED_BACK /
INCONCLUSIVE. Rollback is tested by digest, not declared.

Refuses: GPU lease, protected flock, Codex-surface writes, same-file double
apply, a KEPT verdict on dirty or unmeasured hardware evidence, hardware
performance numbers, a success shape for an absent input.

Cannot establish: that a KERNEL_OR_GPU or TOKEN_RATE mutation is a speedup
(this sidecar has no GPU authority and contamination is not clean); that a
RESIDENT_ARTIFACT child should succeed the incumbent (succession cannot
promote on SELF_MEASURED_DIRTY); that wiring this engine into autonomy_run
or orchestration BINDINGS has happened (those files are outside this lane).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import copy
import fnmatch
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import (
    REPO,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)
from tools.future import autonomy_run as ar
from tools.future import candidate_planner as cp
from tools.future import dirty_measure as dm
from tools.future import mutation_surface as ms
from tools.future import protected_window as pw
from tools.future import succession as succ
from tools.future.contamination import PromotionRefused

RECEIPT = "MUTATION_ENGINE.json"
SCHEMA = "hawking.future.mutation_engine.v1"
MUTATION_SCHEMA = "hawking.future.mutation.v1"
POLICY_SCHEMA = "hawking.future.mutation_engine.pipeline_policy.v1"
RECORDED_BY = "tools/future/mutation_engine.py"

KERNEL_OR_GPU = "KERNEL_OR_GPU"
REPRESENTATION_BPW = "REPRESENTATION_BPW"
TOKEN_RATE = "TOKEN_RATE"
PIPELINE_SELF = "PIPELINE_SELF"
RESIDENT_ARTIFACT = "RESIDENT_ARTIFACT"
MUTATION_CLASSES: tuple[str, ...] = (
    KERNEL_OR_GPU,
    REPRESENTATION_BPW,
    TOKEN_RATE,
    PIPELINE_SELF,
    RESIDENT_ARTIFACT,
)

VERDICT_KEPT = "KEPT"
VERDICT_ROLLED_BACK = "ROLLED_BACK"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICTS: tuple[str, ...] = (VERDICT_KEPT, VERDICT_ROLLED_BACK, VERDICT_INCONCLUSIVE)

PARK_PROTECTED = "BLOCKED_ON_PROTECTED_WINDOW"

# Classes whose *performance* claim needs a protected window this sidecar
# does not hold. Applying the overlay is allowed; KEPT is not.
NEEDS_PROTECTED: frozenset[str] = frozenset(
    {KERNEL_OR_GPU, TOKEN_RATE, REPRESENTATION_BPW, RESIDENT_ARTIFACT}
)

# The 1h trial: 4 refills returning the same 25 ids; 222 rejections of one table.
TRIAL_REFILL_IDS: tuple[str, ...] = tuple(f"FT.replay.{i:02d}" for i in range(25))
TRIAL_REFILL_COUNT = 4
TRIAL_SCAR_EVENTS = 222
TRIAL_SCAR_UNIQUE = 1

RECOVERED_REFILL_IDENTITY = "frontier_module_description"
RECOVERED_COMMIT_AT = "launch"

POLICY_NAME = "pipeline_policy.json"
KERNEL_OVERLAY = "overlays/kernel_or_gpu.json"
REP_OVERLAY = "overlays/representation_bpw.json"
TOKEN_OVERLAY = "overlays/token_rate.json"
CHILD_OVERLAY = "overlays/resident_artifact.json"

# Fusion toggle recovered from candidate_planner.STEM_SYNONYM, not invented.
FUSION_ENV_KEY = "HAWKING_FUSE_GQA_QKV"
# Host-ceremony key recovered from candidate_planner.HOST_CEREMONY_KEYS.
CEREMONY_KEY = "HAWKING_METAL_PIPELINE_CACHE_REUSE"

FRONTIER_TO_CLASS: dict[str, str] = {
    "GPU_KERNELS": KERNEL_OR_GPU,
    "MODEL_EXECUTION": KERNEL_OR_GPU,
    "TPS": TOKEN_RATE,
    "DECODING": TOKEN_RATE,
    "MODEL_REPRESENTATION": REPRESENTATION_BPW,
    "HCLI_SELF": PIPELINE_SELF,
    "VERIFICATION": PIPELINE_SELF,
    "EXPERIMENT_TURNAROUND": PIPELINE_SELF,
    "TOOLS": PIPELINE_SELF,
    "CHILD_RESIDENT": RESIDENT_ARTIFACT,
    "CONTEXT": PIPELINE_SELF,
}

# cpu-turnaround is host ceremony of the *loop*, not a GPU latency number.
FRONTIER_CLASS_OVERRIDES: dict[str, str] = {
    "FT.LATENCY.cpu-turnaround": PIPELINE_SELF,
    "FT.LATENCY.gpu-ns": TOKEN_RATE,
    "FT.TPS.accepted-token-cost": PIPELINE_SELF,
    "FT.TPS.protected-tps": TOKEN_RATE,
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. A mutation whose "
    "verdict needs protected measurement is parked BLOCKED_ON_PROTECTED_WINDOW "
    "and may only be INCONCLUSIVE here. SELF_MEASURED_DIRTY cannot KEPT."
)


class MutationRefused(ValueError):
    """A mutation was refused before it could pretend to succeed."""


class PartitionRefused(MutationRefused):
    """Target is outside the sidecar write partition. Raised before any write."""


class MutationConflictError(cp.IncompatibleMutationError):
    """Two mutations occupy the same file. The planner's cell rule, for artifacts."""


# Process-level bound engine. Unbound propose/apply refuse rather than touch
# the live tree. The resident must bind a reversible scope first.
_BOUND: list["MutationEngine | None"] = [None]


def bind(engine: "MutationEngine") -> "MutationEngine":
    _BOUND[0] = engine
    return engine


def unbind() -> None:
    _BOUND[0] = None


def _need() -> "MutationEngine":
    eng = _BOUND[0]
    if eng is None:
        raise MutationRefused(
            "no engine bound; refuse rather than mutate the live tree"
        )
    return eng


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _absent_digest() -> str:
    return _sha(b"")


def _clean(node: Any) -> Any:
    """Every public record is scanned. A hardware number is a fabrication here."""
    _assert_no_hardware_claims(node)
    _assert_no_magnitudes(node)
    return node


def _assert_no_magnitudes(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in dm.MAGNITUDE_FIELDS and isinstance(value, (int, float)):
                raise dm.DirtyMagnitudeRefused(
                    f"{here} = {value!r}: mutation engine refuses latency/TPS/"
                    "hardware magnitudes; work completed is a count, not a rate"
                )
            _assert_no_magnitudes(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_magnitudes(value, f"{path}[{i}]")


def recovered_pipeline_policy() -> dict[str, Any]:
    """Policy as autonomy_run.py actually runs it. Not an improvement."""
    return {
        "schema": POLICY_SCHEMA,
        "source": "tools/future/autonomy_run.py",
        "refill_watermark": ar.REFILL_WATERMARK,
        "refill_every": ar.REFILL_EVERY,
        "refill_interval_s": ar.REFILL_INTERVAL_S,
        "unit_budget_s": ar.UNIT_BUDGET_S,
        "refill_identity": RECOVERED_REFILL_IDENTITY,
        "identity_committed_at": RECOVERED_COMMIT_AT,
        "scar_refusal_once_per_identity": False,
        "stop_after_first": False,
        "gpu_authority": False,
    }


def simulate_trial_refills(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the frozen trial's refill table under a policy.

    Recovered identity is (frontier, module, description) with description
    varying per refill, so four refills of the same 25 ids look like 100
    new units. Committing (frontier, module) at queue time is the mutation
    that makes refill refuse the copies.
    """
    ident_mode = str(policy.get("refill_identity") or RECOVERED_REFILL_IDENTITY)
    stop_first = bool(policy.get("stop_after_first"))
    queued: list[str] = []
    claimed: set[str] = set()
    frontiers: list[str] = []
    replays = 0
    for r in range(TRIAL_REFILL_COUNT):
        for fid in TRIAL_REFILL_IDS:
            if stop_first and frontiers:
                replays += 1
                continue
            if ident_mode == "frontier_module":
                ident = fid
            elif ident_mode == "frontier_module_description":
                ident = f"{fid}|desc=refill-{r}"
            else:
                ident = f"{fid}|nonce={r}"
            if ident in claimed:
                replays += 1
                continue
            queued.append(ident)
            frontiers.append(fid)
            claimed.add(ident)
    unique = len(set(frontiers))
    return {
        "units_queued": len(queued),
        "unique_frontier_ids": unique,
        "replays_skipped": replays,
        "n_refills": TRIAL_REFILL_COUNT,
        "source_ids": len(TRIAL_REFILL_IDS),
        "busywork": len(queued) > unique,
    }


def simulate_scar_replays(policy: Mapping[str, Any]) -> dict[str, Any]:
    """222 rejections in one window, one identical table.

    Once-per-identity emits one refusal; replay_ok emits all 222.
    """
    once = bool(policy.get("scar_refusal_once_per_identity"))
    events = TRIAL_SCAR_UNIQUE if once else TRIAL_SCAR_EVENTS
    return {
        "refusal_events": events,
        "unique_ideas": TRIAL_SCAR_UNIQUE,
        "replayed_identical_table": not once,
    }


def _frontier_id(frontier: str | Mapping[str, Any]) -> str:
    if isinstance(frontier, Mapping):
        fid = frontier.get("id") or frontier.get("frontier")
        if not fid:
            raise MutationRefused("frontier mapping has no id; refuse rather than invent")
        return str(fid)
    text = str(frontier or "").strip()
    if not text:
        raise MutationRefused("absent frontier; refuse rather than invent a mutation")
    return text


def classify_frontier(frontier_id: str) -> str:
    if frontier_id in FRONTIER_CLASS_OVERRIDES:
        return FRONTIER_CLASS_OVERRIDES[frontier_id]
    name = frontier_id
    if name.startswith("FT."):
        parts = name.split(".")
        if len(parts) >= 2:
            name = parts[1]
    klass = FRONTIER_TO_CLASS.get(name)
    if klass is None:
        raise MutationRefused(
            f"frontier {frontier_id!r} maps to no mutation class; "
            f"known heads: {sorted(FRONTIER_TO_CLASS)}"
        )
    return klass


def _declared_repo_rel(target: str) -> str | None:
    t = target.replace("\\", "/").lstrip("./")
    if not t or t.startswith("/") or t.startswith("~"):
        return t or target
    for pat in ms.SIDECAR_OWNED + ms.CODEX_OWNED:
        prefix = pat.rstrip("*").rstrip("/")
        if t == prefix or t.startswith(prefix + "/") or fnmatch.fnmatch(t, pat):
            return t
    return None


def refuse_if_outside_partition(target: str) -> None:
    """The partition check. Must fire BEFORE any write, not after."""
    rel = _declared_repo_rel(target)
    if rel is None:
        return
    owner = ms.owner(rel)
    if owner != "SIDECAR" or ms.intersects_codex(rel):
        raise PartitionRefused(
            f"refused before apply: {rel} is {owner}-owned, not sidecar "
            f"(intersects_codex={ms.intersects_codex(rel)})"
        )


def _default_target(klass: str) -> str:
    return {
        KERNEL_OR_GPU: KERNEL_OVERLAY,
        REPRESENTATION_BPW: REP_OVERLAY,
        TOKEN_RATE: TOKEN_OVERLAY,
        PIPELINE_SELF: POLICY_NAME,
        RESIDENT_ARTIFACT: CHILD_OVERLAY,
    }[klass]


def _default_change(klass: str) -> dict[str, Any]:
    if klass == KERNEL_OR_GPU:
        return {
            "exact_mutation": {"child_fusion_env": {FUSION_ENV_KEY: "1"}},
            "measurement": "UNMEASURED",
        }
    if klass == REPRESENTATION_BPW:
        return {"rung_id": "R1.75", "claim": "UNMEASURED"}
    if klass == TOKEN_RATE:
        return {
            "host_ceremony": {CEREMONY_KEY: "1"},
            "measurement": "UNMEASURED",
        }
    if klass == PIPELINE_SELF:
        return {
            "refill_identity": "frontier_module",
            "identity_committed_at": "queue",
            "scar_refusal_once_per_identity": True,
        }
    return {
        "ladder": list(succ.LADDER_RUNGS),
        "child_status": "CANDIDATE",
        "succession_verdict": succ.VERDICT_INSUFFICIENT,
        "reason": succ.REASON_NO_EXECUTED_GPU_AUTHORITY,
    }


def _default_hypothesis(klass: str, frontier_id: str) -> str:
    if klass == PIPELINE_SELF:
        return (
            f"on {frontier_id}: commit refill identity at queue time as "
            "(frontier, module) and refuse duplicate scar-table replays, so "
            "the loop cannot re-emit the same 25 ids or 222 identical rejections"
        )
    if klass == KERNEL_OR_GPU:
        return (
            f"on {frontier_id}: overlay fusion toggle {FUSION_ENV_KEY}; "
            "dispatch-shape performance is unmeasured without a protected window"
        )
    if klass == TOKEN_RATE:
        return (
            f"on {frontier_id}: overlay host-ceremony {CEREMONY_KEY}; "
            "accepted-token rate is unmeasured without a protected window"
        )
    if klass == REPRESENTATION_BPW:
        return (
            f"on {frontier_id}: set representation rung_id on a scoped artifact; "
            "quality/EBPW is unmeasured without a protected window"
        )
    return (
        f"on {frontier_id}: propose a child via the succession ladder; "
        f"{succ.VERDICT_INSUFFICIENT} because this sidecar cannot mint "
        "PROTECTED evidence"
    )


def _seed_for(klass: str) -> dict[str, Any]:
    if klass == PIPELINE_SELF:
        return recovered_pipeline_policy()
    if klass == KERNEL_OR_GPU:
        return {
            "schema": "hawking.future.mutation_engine.kernel_overlay.v1",
            "source": "tools/future/candidate_planner.py",
            "exact_mutation": {"child_fusion_env": {}},
            "measurement": "UNMEASURED",
            "gpu_authority": False,
        }
    if klass == TOKEN_RATE:
        return {
            "schema": "hawking.future.mutation_engine.token_overlay.v1",
            "source": "tools/future/candidate_planner.py HOST_CEREMONY_KEYS",
            "host_ceremony": {k: "0" for k in sorted(cp.HOST_CEREMONY_KEYS)},
            "measurement": "UNMEASURED",
            "gpu_authority": False,
        }
    if klass == REPRESENTATION_BPW:
        return {
            "schema": "hawking.future.mutation_engine.representation_overlay.v1",
            "source": "tools/future/flash_bpw_ladder.py",
            "rung_id": "R_Q4",
            "claim": "UNMEASURED",
            "gpu_authority": False,
        }
    return {
        "schema": "hawking.future.mutation_engine.resident_overlay.v1",
        "source": "tools/future/succession.py",
        "ladder": list(succ.LADDER_RUNGS),
        "child_status": "ABSENT",
        "succession_verdict": succ.VERDICT_INSUFFICIENT,
        "reason": succ.REASON_SELF_MEASURED_DIRTY,
        "gpu_authority": False,
    }


class MutationEngine:
    """Reversible mutations, bound to one writable scope.

    Writes never leave `scope`. A Codex target is refused before the first
    byte. Two mutations cannot hold the same file. Protected-class mutations
    apply an overlay and park; they cannot KEPT.
    """

    def __init__(self, scope: str | Path) -> None:
        root = Path(scope).resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            rel: Path | None = root.relative_to(REPO.resolve())
        except ValueError:
            rel = None  # isolated lab directory outside the repo is the intended scope
        if rel is not None and rel.as_posix() != ".":
            posix = rel.as_posix()
            owner = ms.owner(posix)
            if owner != "SIDECAR" or ms.intersects_codex(posix):
                raise PartitionRefused(
                    f"engine scope {posix} is {owner}-owned; bind a sidecar "
                    "or isolated lab directory, not the Codex surface"
                )
        self.scope = root
        self._seq = 0
        self._mutations: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, str] = {}
        self._snaps: dict[str, bytes | None] = {}
        self._before: dict[str, str] = {}
        self._after: dict[str, str] = {}
        self._lease_calls = 0
        self._write_policy(recovered_pipeline_policy())

    def _write_policy(self, doc: Mapping[str, Any]) -> Path:
        path = self.scope / POLICY_NAME
        path.write_text(json.dumps(dict(doc), indent=1, sort_keys=True) + "\n")
        return path

    def _scope_path(self, target: str) -> Path:
        if Path(target).is_absolute() or ".." in Path(target).parts:
            raise MutationRefused(
                f"target {target!r} escapes the reversible scope; refuse before write"
            )
        refuse_if_outside_partition(target)
        dest = (self.scope / target).resolve()
        try:
            dest.relative_to(self.scope)
        except ValueError as exc:
            raise MutationRefused(
                f"target {target!r} resolved outside {self.scope}"
            ) from exc
        return dest

    def propose(
        self,
        frontier: str | Mapping[str, Any],
        *,
        mutation_class: str | None = None,
        target: str | None = None,
        change: Mapping[str, Any] | None = None,
        hypothesis: str | None = None,
    ) -> dict[str, Any]:
        fid = _frontier_id(frontier)
        klass = mutation_class or classify_frontier(fid)
        if klass not in MUTATION_CLASSES:
            raise MutationRefused(
                f"unknown mutation class {klass!r}; not in {MUTATION_CLASSES}"
            )
        ch = dict(change) if change is not None else _default_change(klass)
        if not ch:
            raise MutationRefused("empty change is a no-op; refuse")
        _clean(ch)
        tgt = target if target is not None else _default_target(klass)
        if not str(tgt).strip():
            raise MutationRefused("absent target; refuse rather than guess a path")
        # Partition check at propose, so a Codex mutation never reaches apply.
        refuse_if_outside_partition(tgt)
        needs = klass in NEEDS_PROTECTED
        self._seq += 1
        rec = {
            "schema": MUTATION_SCHEMA,
            "id": f"MUT.{klass}.{fid.replace('.', '_')}.{self._seq}",
            "mutation_class": klass,
            "frontier": fid,
            "hypothesis": hypothesis or _default_hypothesis(klass, fid),
            "target": tgt,
            "change": ch,
            "needs_protected_window": needs,
            "evidence_class_at_best": (
                dm.EVIDENCE_CLASS if needs else "STATIC_ONLY"
            ),
            "gpu_authority": False,
            "state": "PROPOSED",
            "parking": PARK_PROTECTED if needs else None,
            "verdict": None,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _clean(rec)
        self._mutations[rec["id"]] = rec
        return dict(rec)

    def apply(self, m: Mapping[str, Any]) -> dict[str, Any]:
        rec = self._resolve(m)
        if rec["state"] not in {"PROPOSED", "ROLLED_BACK"}:
            raise MutationRefused(
                f"{rec['id']} is {rec['state']}; apply is for PROPOSED mutations"
            )
        # Refuse-before-write: partition, then conflict, then snapshot, then write.
        refuse_if_outside_partition(rec["target"])
        dest = self._scope_path(rec["target"])
        key = str(dest)
        holder = self._locks.get(key)
        if holder and holder != rec["id"]:
            raise MutationConflictError(
                f"incompatible cell {{{holder}, {rec['id']}}}: both touch {rec['target']}"
            )
        snap = dest.read_bytes() if dest.is_file() else None
        before = _sha(snap if snap is not None else b"")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if snap is None:
            seed = _seed_for(rec["mutation_class"])
            dest.write_text(json.dumps(seed, indent=1, sort_keys=True) + "\n")
        try:
            doc = json.loads(dest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MutationRefused(f"target unreadable as JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise MutationRefused("target is not a JSON object; refuse")
        merged = copy.deepcopy(doc)
        _deep_update(merged, rec["change"])
        if json.dumps(merged, sort_keys=True) == json.dumps(doc, sort_keys=True):
            if snap is None and dest.is_file():
                dest.unlink()
            raise MutationRefused("NO_OP_MUTATION: change does not alter the artifact")
        _clean(merged)
        dest.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
        after = _sha(dest.read_bytes())
        if after == before:
            raise MutationRefused("NO_OP_MUTATION: after digest equals before")
        self._snaps[rec["id"]] = snap
        self._before[rec["id"]] = before
        self._after[rec["id"]] = after
        self._locks[key] = rec["id"]
        rec["state"] = "APPLIED"
        rec["applied_path"] = str(dest.relative_to(self.scope).as_posix())
        rec["before_digest"] = before
        rec["after_digest"] = after
        rec["parking"] = PARK_PROTECTED if rec["needs_protected_window"] else None
        _clean(rec)
        return {
            "id": rec["id"],
            "state": rec["state"],
            "applied_path": rec["applied_path"],
            "before_digest": before,
            "after_digest": after,
            "parking": rec["parking"],
            "gpu_authority": False,
            "lease_acquired": False,
            "lease_calls": self._lease_calls,
        }

    def evidence(self, m: Mapping[str, Any]) -> dict[str, Any]:
        rec = self._resolve(m)
        if rec["state"] != "APPLIED":
            raise MutationRefused(
                f"{rec['id']} is {rec['state']}; evidence requires an applied mutation"
            )
        dest = self._scope_path(rec["target"])
        now = _sha(dest.read_bytes()) if dest.is_file() else _absent_digest()
        if now != rec.get("after_digest"):
            raise MutationRefused(
                f"{rec['id']} after-digest drifted ({now} != {rec.get('after_digest')}); "
                "refuse to report evidence of a world we did not apply"
            )
        klass = rec["mutation_class"]
        needs = rec["needs_protected_window"]
        ev_class = dm.EVIDENCE_CLASS if needs else "STATIC_ONLY"
        work: dict[str, Any] | None = None
        if klass == PIPELINE_SELF:
            before_doc = recovered_pipeline_policy()
            snap = self._snaps.get(rec["id"])
            if snap is not None:
                try:
                    loaded = json.loads(snap.decode())
                    if isinstance(loaded, dict):
                        before_doc = loaded
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise MutationRefused("before-image is not JSON; refuse")
            after_doc = json.loads(dest.read_text())
            b_refill = simulate_trial_refills(before_doc)
            a_refill = simulate_trial_refills(after_doc)
            b_scar = simulate_scar_replays(before_doc)
            a_scar = simulate_scar_replays(after_doc)
            work = {
                "units_queued_before": b_refill["units_queued"],
                "units_queued_after": a_refill["units_queued"],
                "unique_frontier_ids_before": b_refill["unique_frontier_ids"],
                "unique_frontier_ids_after": a_refill["unique_frontier_ids"],
                "replays_skipped_before": b_refill["replays_skipped"],
                "replays_skipped_after": a_refill["replays_skipped"],
                "busywork_before": b_refill["busywork"],
                "busywork_after": a_refill["busywork"],
                "refusal_events_before": b_scar["refusal_events"],
                "refusal_events_after": a_scar["refusal_events"],
                "unit": "work_completed_counts",
            }
        body = {
            "id": rec["id"],
            "mutation_class": klass,
            "frontier": rec["frontier"],
            "hypothesis": rec["hypothesis"],
            "before_digest": rec["before_digest"],
            "after_digest": rec["after_digest"],
            "digest_changed": rec["before_digest"] != rec["after_digest"],
            "evidence_class": ev_class,
            "gpu_authority": False,
            "needs_protected_window": needs,
            "parking": rec["parking"],
            "work": work,
            "promotable": False,
            "measurement": "UNMEASURED" if needs else "work_completed_in_scope",
            "claim_boundary": CLAIM_BOUNDARY,
        }
        return _clean(body)

    def rollback(self, m: Mapping[str, Any]) -> dict[str, Any]:
        rec = self._resolve(m)
        if rec["state"] not in {"APPLIED"}:
            raise MutationRefused(
                f"{rec['id']} is {rec['state']}; rollback is for APPLIED mutations"
            )
        dest = self._scope_path(rec["target"])
        snap = self._snaps[rec["id"]]
        before = self._before[rec["id"]]
        if snap is None:
            if dest.is_file():
                dest.unlink()
            restored = b""
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(snap)
            restored = dest.read_bytes()
        got = _sha(restored)
        match = got == before
        if not match:
            raise MutationRefused(
                f"rollback digest mismatch for {rec['id']}: got {got} want {before}"
            )
        self._locks.pop(str(dest), None)
        rec["state"] = "ROLLED_BACK"
        rec["verdict"] = VERDICT_ROLLED_BACK
        rec["parking"] = rec["parking"] if rec["needs_protected_window"] else None
        out = {
            "id": rec["id"],
            "state": rec["state"],
            "before_digest": before,
            "restored_digest": got,
            "digest_match": True,
            "byte_identical": True,
            "verdict": VERDICT_ROLLED_BACK,
            "gpu_authority": False,
        }
        return _clean(out)

    def verdict(self, m: Mapping[str, Any]) -> dict[str, Any]:
        rec = self._resolve(m)
        if rec["state"] != "APPLIED":
            raise MutationRefused(
                f"{rec['id']} is {rec['state']}; verdict requires an applied mutation"
            )
        ev = self.evidence(rec)
        decision, reason = self._decide(rec, ev)
        if decision == VERDICT_ROLLED_BACK:
            rb = self.rollback(rec)
            out = {
                "id": rec["id"],
                "verdict": VERDICT_ROLLED_BACK,
                "reason": reason,
                "parking": rec.get("parking"),
                "evidence_class": ev["evidence_class"],
                "digest_match": rb["digest_match"],
                "gpu_authority": False,
                "promotable": False,
            }
            rec["verdict"] = VERDICT_ROLLED_BACK
            return _clean(out)
        rec["verdict"] = decision
        out = {
            "id": rec["id"],
            "verdict": decision,
            "reason": reason,
            "parking": rec.get("parking"),
            "evidence_class": ev["evidence_class"],
            "gpu_authority": False,
            "promotable": False,
            "work": ev.get("work"),
        }
        return _clean(out)

    def _decide(self, rec: Mapping[str, Any], ev: Mapping[str, Any]) -> tuple[str, str]:
        if rec["needs_protected_window"] or ev.get("parking") == PARK_PROTECTED:
            if ev["evidence_class"] != "STATIC_ONLY":
                return (
                    VERDICT_INCONCLUSIVE,
                    f"{rec['mutation_class']} needs a protected window; evidence is "
                    f"{ev['evidence_class']} on a sidecar with gpu_authority=false; "
                    f"parked {PARK_PROTECTED}; KEPT is refused",
                )
            return (
                VERDICT_INCONCLUSIVE,
                f"{rec['mutation_class']} needs protected measurement this "
                f"sidecar cannot take; parked {PARK_PROTECTED}",
            )
        if ev["evidence_class"] != "STATIC_ONLY":
            return (
                VERDICT_INCONCLUSIVE,
                f"evidence_class {ev['evidence_class']!r} cannot support KEPT",
            )
        if rec["mutation_class"] != PIPELINE_SELF:
            return (
                VERDICT_INCONCLUSIVE,
                f"{rec['mutation_class']} is not completable as work-completed "
                "STATIC_ONLY on this host",
            )
        work = ev.get("work") or {}
        unique_b = int(work.get("unique_frontier_ids_before") or 0)
        unique_a = int(work.get("unique_frontier_ids_after") or 0)
        queued_b = int(work.get("units_queued_before") or 0)
        queued_a = int(work.get("units_queued_after") or 0)
        skip_b = int(work.get("replays_skipped_before") or 0)
        skip_a = int(work.get("replays_skipped_after") or 0)
        if unique_a < unique_b:
            return (
                VERDICT_ROLLED_BACK,
                f"unique frontier work dropped {unique_b} -> {unique_a}; "
                "a mutation that loses work is not kept",
            )
        if queued_a >= queued_b and skip_a <= skip_b:
            return (
                VERDICT_ROLLED_BACK,
                f"no busywork reduction (queued {queued_b}->{queued_a}, "
                f"replays skipped {skip_b}->{skip_a}); rolled back",
            )
        if queued_a >= queued_b:
            return (
                VERDICT_ROLLED_BACK,
                f"queued work did not drop ({queued_b}->{queued_a}); rolled back",
            )
        return (
            VERDICT_KEPT,
            f"unique work held ({unique_b}->{unique_a}); queued copies dropped "
            f"{queued_b}->{queued_a}; replays skipped {skip_b}->{skip_a}",
        )

    def _resolve(self, m: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(m, str):
            mid = m
        else:
            mid = str((m or {}).get("id") or "")
        if not mid:
            raise MutationRefused("mutation has no id")
        rec = self._mutations.get(mid)
        if rec is None:
            raise MutationRefused(f"unknown mutation {mid!r}")
        return rec


def refuse_protected_lease(*_a: Any, **_k: Any) -> None:
    """The lease path, named so tests can watch it fire. apply() never calls it."""
    pw.acquire_lease()


def _deep_update(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if (
            key in dst
            and isinstance(dst[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def propose(frontier: str | Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    engine = kwargs.pop("engine", None)
    return (engine or _need()).propose(frontier, **kwargs)


def apply(m: Mapping[str, Any], *, engine: MutationEngine | None = None) -> dict[str, Any]:
    return (engine or _need()).apply(m)


def evidence(m: Mapping[str, Any], *, engine: MutationEngine | None = None) -> dict[str, Any]:
    return (engine or _need()).evidence(m)


def rollback(m: Mapping[str, Any], *, engine: MutationEngine | None = None) -> dict[str, Any]:
    return (engine or _need()).rollback(m)


def verdict(m: Mapping[str, Any], *, engine: MutationEngine | None = None) -> dict[str, Any]:
    return (engine or _need()).verdict(m)


def pipeline_self_cycle(engine: MutationEngine) -> dict[str, Any]:
    """The one mutation this host can honestly complete, end to end."""
    proposed = engine.propose("FT.HCLI_SELF.emit-workunits")
    applied = engine.apply(proposed)
    ev = engine.evidence(proposed)
    decided = engine.verdict(proposed)
    # Undo must still work after KEPT: a mutation engine without a proven
    # undo is a way to break the system autonomously.
    if decided["verdict"] == VERDICT_KEPT:
        undone = engine.rollback(proposed)
    else:
        undone = {
            "digest_match": decided.get("digest_match"),
            "byte_identical": decided.get("verdict") == VERDICT_ROLLED_BACK,
            "already_rolled_by_verdict": True,
        }
    return _clean(
        {
            "mutation_id": proposed["id"],
            "mutation_class": proposed["mutation_class"],
            "frontier": proposed["frontier"],
            "hypothesis": proposed["hypothesis"],
            "applied": applied,
            "evidence": ev,
            "verdict": decided,
            "rollback_after": undone,
            "rollback_digest_match": bool(
                (undone or {}).get("digest_match")
                or decided.get("digest_match")
            ),
        }
    )


def _proofs_in_scope(scope: Path) -> dict[str, Any]:
    """Drive every mandatory negative control. A guard never watched failing is not a guard."""
    proofs: dict[str, Any] = {}
    eng = MutationEngine(scope)

    # 1. PIPELINE_SELF end to end, rollback by digest.
    cycle = pipeline_self_cycle(eng)
    proofs["pipeline_self"] = {
        "holds": cycle["verdict"]["verdict"] == VERDICT_KEPT
        and cycle["rollback_digest_match"] is True,
        "verdict": cycle["verdict"]["verdict"],
        "rollback_digest_match": cycle["rollback_digest_match"],
        "work": (cycle["evidence"] or {}).get("work"),
        "mutation_id": cycle["mutation_id"],
    }

    # 2. Harmful PIPELINE_SELF must ROLLED_BACK (negative of KEPT).
    harmful = eng.propose(
        "FT.VERIFICATION.repro",
        change={"stop_after_first": True, "refill_identity": "frontier_module"},
    )
    eng.apply(harmful)
    hv = eng.verdict(harmful)
    proofs["harmful_rolled_back"] = {
        "holds": hv["verdict"] == VERDICT_ROLLED_BACK,
        "verdict": hv["verdict"],
        "reason": hv["reason"],
    }

    # 3. Protected-class mutation never KEPT on dirty evidence.
    kernel = eng.propose("FT.GPU_KERNELS.ready-protected")
    eng.apply(kernel)
    kev = eng.evidence(kernel)
    kv = eng.verdict(kernel)
    proofs["kernel_not_kept"] = {
        "holds": kv["verdict"] != VERDICT_KEPT
        and kv["verdict"] == VERDICT_INCONCLUSIVE
        and kev["evidence_class"] == dm.EVIDENCE_CLASS
        and kv.get("parking") == PARK_PROTECTED,
        "verdict": kv["verdict"],
        "evidence_class": kev["evidence_class"],
        "parking": kv.get("parking"),
    }
    eng.rollback(kernel)

    token = eng.propose("FT.TPS.protected-tps")
    eng.apply(token)
    tv = eng.verdict(token)
    proofs["token_rate_not_kept"] = {
        "holds": tv["verdict"] == VERDICT_INCONCLUSIVE
        and tv["verdict"] != VERDICT_KEPT,
        "verdict": tv["verdict"],
        "parking": tv.get("parking"),
    }
    eng.rollback(token)

    # 4. Outside-partition refused BEFORE apply (file must not appear).
    crate = "crates/hawking-core/src/engine.rs"
    outside_raised = False
    outside_before = False
    try:
        eng.propose("FT.GPU_KERNELS.static-warnings", target=crate)
    except PartitionRefused:
        outside_raised = True
        outside_before = not (eng.scope / crate).exists()
    proofs["partition_refuse_before_apply"] = {
        "holds": outside_raised and outside_before,
        "target": crate,
        "file_created": (eng.scope / crate).exists(),
    }

    # 5. Same-file conflict.
    a = eng.propose("FT.HCLI_SELF.emit-workunits")
    b = eng.propose("FT.TOOLS.frontiers-refill")
    eng.apply(a)
    conflict_raised = False
    try:
        eng.apply(b)
    except MutationConflictError:
        conflict_raised = True
    proofs["same_file_conflict"] = {
        "holds": conflict_raised,
        "a": a["id"],
        "b": b["id"],
        "target": a["target"],
    }
    eng.rollback(a)

    # 6. Dirty evidence cannot be offered as promotion.
    dirty_closed = False
    try:
        dm.offer_for_promotion(
            {
                "evidence_class": dm.EVIDENCE_CLASS,
                "measurement_class": "STATIC_ONLY",
                "gpu_authority": False,
            }
        )
    except PromotionRefused:
        dirty_closed = True
    proofs["dirty_cannot_promote"] = {"holds": dirty_closed}

    # 7. Hardware field refused on propose.
    hw_raised = False
    try:
        eng.propose("FT.HCLI_SELF.emit-workunits", change={"tps": 120.0})
    except (HardwareClaimError, dm.DirtyMagnitudeRefused, MutationRefused):
        hw_raised = True
    proofs["hardware_field_refused"] = {"holds": hw_raised}

    # 8. Unbound module-level propose refuses (does not touch the live tree).
    prev = _BOUND[0]
    unbind()
    unbound_raised = False
    try:
        propose("FT.HCLI_SELF.emit-workunits")
    except MutationRefused:
        unbound_raised = True
    if prev is not None:
        bind(prev)
    proofs["unbound_refuses"] = {"holds": unbound_raised}

    proofs["all_hold"] = all(
        bool(p.get("holds")) for p in proofs.values() if isinstance(p, dict) and "holds" in p
    )
    proofs["lease_calls"] = eng._lease_calls
    return proofs


def build() -> Path:
    with tempfile.TemporaryDirectory(prefix="hawking-mutation-") as tmp:
        proofs = _proofs_in_scope(Path(tmp))
    recovered = [
        "tools/future/mutation_surface.py — owner/intersects_codex/check_disjoint (partition)",
        "tools/future/autonomy_run.py — verifier loop, REFILL_* constants, seen_identity-at-launch",
        "tools/future/candidate_planner.py — IncompatibleMutationError, HOST_CEREMONY_KEYS, STEM_SYNONYM",
        "tools/future/dirty_measure.py — SELF_MEASURED_DIRTY, offer_for_promotion, MAGNITUDE_FIELDS",
        "tools/future/contamination.py — PromotionRefused",
        "tools/future/protected_window.py — acquire_lease raises; never called here",
        "tools/future/succession.py — LADDER_RUNGS, REFUSE_INSUFFICIENT_EVIDENCE",
        "tools/future/sandbox.py — reversible local science (cited, not forked)",
        "hcli/mutation.py — snapshot/rollback/content_fingerprint algorithm recovered",
        "tools/future/adaptive_verification.py — fail-closed screens, not-yet-dead is not verified",
        "tools/future/frontiers.py — admit/next_work/refill (busywork identity)",
        "tools/future/orchestration.py — invoke/BINDINGS (read; this module is not bound here)",
        "tools/future/qualification_pipeline.py — AuthorityBoundaryError, no GPU seize",
        "tools/future/accelerator_workunits.py — GPU species emitted SLEEPING",
        "tools/future/codex_behaviors.py — mutation_scope on WorkUnit species",
        "tools/future/_common.py — write_receipt, HARDWARE_FIELDS",
    ]
    gaps = [
        "no propose/apply/evidence/rollback/verdict engine existed; autonomy_run only verifies",
        "KERNEL_OR_GPU and TOKEN_RATE had no honest INCONCLUSIVE parking path",
        "rollback was declared in succession/hcli but not driven as a resident mutation cycle",
        "same-file mutation conflict was a planner cell rule, not an apply-time lock",
        "Codex-target refusal existed as a checker, not as a pre-apply gate on a mutation",
        "PIPELINE_SELF had no reversible policy artifact and no work-completed measurement",
    ]
    negatives = [
        "KERNEL_OR_GPU / TOKEN_RATE / REPRESENTATION_BPW performance cannot be measured here",
        "contamination is not QUIESCENT; SELF_MEASURED_DIRTY cannot KEPT and cannot promote",
        "this sidecar has no GPU lease and acquire_lease is never called (lease_calls="
        f"{proofs.get('lease_calls', 0)})",
        "orchestration BINDINGS does not yet name mutation_engine.py (WRITE list forbids editing it)",
        "autonomy_run.py is not yet driven by this engine (WRITE list forbids editing it)",
        "RESIDENT_ARTIFACT cannot promote: succession verdict is REFUSE_INSUFFICIENT_EVIDENCE",
        "a live resident body was not mutated; overlays live in a reversible lab scope",
    ]
    if not proofs.get("all_hold"):
        raise MutationRefused(f"proofs failed closed: {proofs}")
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Give the resident a reversible mutation cycle so autonomy can "
            "propose, apply, measure, and roll back real changes instead of "
            "only verifying receipts."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "mutation_classes": list(MUTATION_CLASSES),
        "verdicts": list(VERDICTS),
        "parking": PARK_PROTECTED,
        "needs_protected": list(NEEDS_PROTECTED),
        "completable_here": [PIPELINE_SELF],
        "recovered_loop_constants": {
            "refill_watermark": ar.REFILL_WATERMARK,
            "refill_every": ar.REFILL_EVERY,
            "refill_interval_s": ar.REFILL_INTERVAL_S,
            "unit_budget_s": ar.UNIT_BUDGET_S,
            "refill_identity": RECOVERED_REFILL_IDENTITY,
            "identity_committed_at": RECOVERED_COMMIT_AT,
        },
        "trial_table": {
            "refill_ids": len(TRIAL_REFILL_IDS),
            "refill_count": TRIAL_REFILL_COUNT,
            "scar_events": TRIAL_SCAR_EVENTS,
            "scar_unique": TRIAL_SCAR_UNIQUE,
        },
        "proofs": proofs,
        "fusion_env_key": FUSION_ENV_KEY,
        "ceremony_key": CEREMONY_KEY,
        "recovered_implementation": recovered,
        "gaps_closed": gaps,
        "negative_findings": negatives,
        "resident_callable": {
            "entry_point": "tools.future.mutation_engine.propose(frontier)",
            "workunit": (
                "one CPU_ANALYSIS unit; bind a reversible scope, then "
                "propose/apply/evidence/rollback/verdict. PIPELINE_SELF is "
                "completable as work-completed counts; KERNEL_OR_GPU and "
                "TOKEN_RATE park BLOCKED_ON_PROTECTED_WINDOW."
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "unbound engine, absent frontier, Codex target, same-file "
                "conflict, no-op change, dirty KEPT, hardware fields: all "
                "raise; never a success shape"
            ),
        },
        "next_workunits": [
            "bind mutation_engine.py in tools/future/orchestration.py BINDINGS "
            "(outside this lane's WRITE list)",
            "drive propose/apply from autonomy_run.py instead of only "
            "orchestration.invoke of read-only capabilities (outside this WRITE list)",
            "measure KERNEL_OR_GPU / TOKEN_RATE under a real HCLI protected "
            "window this sidecar must not seize",
        ],
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true")
    ap.parse_args(); out = build(); print(out); return 0


if __name__ == "__main__":
    raise SystemExit(main())
