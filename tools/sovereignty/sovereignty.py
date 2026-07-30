#!/usr/bin/env python3
"""Capability Sovereignty spine: the deterministic, gate-independent half.

Rev3 SS8 (Capability Sovereignty and The Forge), SS17 (Manifest Schema), and
Appendix C (Sovereignty Event) define a much larger system: a Forge pipeline that
diagnoses and restores suppressed capability, and metrics that require a served
model answering an evaluated prompt set. None of that is here. This module is only
the part that is true today, from data already on disk, with no runtime:

  1. continuity manifest   -- read from a real .gravity artifact header, verbatim
  2. sovereignty event log -- append-only JSONL, content-addressed ids
  3. refusal attribution   -- the six SS8.6 codes, generic refusal REFUSED
  4. limit registry        -- add/list/check, no invisible limits
  5. sovereignty metrics   -- the 3 of 5 (SS8.10) computable from the event log alone

GATED, requiring a served runtime and an evaluated prompt set (Forge F0/F3):
false_refusal_rate, boundary_error_rate. Never fabricated, never zeroed -- named
in gated_stages with a reason, same discipline as tools/prometheus/prometheus.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "condense"))
import artifact_client as G  # noqa: E402  (read_header)

DEFAULT_STATE_DIR = REPO / "state" / "sovereignty"
EVENT_LOG_NAME = "sovereignty_events.jsonl"

# SS8.3 Four Planes (+ capability = the fifth surface an event can name).
PLANES = {"capability", "policy", "permission", "evidence", "resource"}

# SS8.6 Refusal Attribution. A generic "I cannot help" is insufficient inside
# Hawking -- every refusal must classify as exactly one of these six.
REFUSAL_CODES = {
    "R-CAPABILITY": "model attempted and failed",
    "R-LEARNED": "inherited blanket refusal",
    "R-POLICY": "declared profile excludes the request",
    "R-PERMISSION": "reasoning is allowed, external action is not",
    "R-VERIFICATION": "generated claim failed evidence requirements",
    "R-RESOURCE": "memory, context, time, or compute exhausted",
}

# SS8.10 metrics that need a served model over an evaluated prompt set (Forge
# F0/F3). Computing these today would mean fabricating a number; refuse instead.
GATED_METRICS = {
    "false_refusal_rate": "GATED: needs a served model + evaluated permitted-task "
        "prompt set (Forge F0/F3). Definition: permitted tasks incorrectly refused "
        "/ permitted tasks.",
    "boundary_error_rate": "GATED: needs a served model + evaluated out-of-profile "
        "prompt set (Forge F0/F3). Definition: out-of-profile actions incorrectly "
        "authorized / out-of-profile action requests.",
}


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. Continuity manifest (Rev3 SS17 `continuity:` block)
# ---------------------------------------------------------------------------

def build_continuity_manifest(artifact_path: Path, *, requested_model: str | None = None,
                              decoder_abi: str | None = None,
                              profile_hash: str | None = None,
                              policy_hash: str | None = None,
                              system_prompt_hash: str | None = None,
                              tool_schema_hash: str | None = None,
                              fallback_allowed: bool = False,
                              fallback_events: list | None = None) -> dict:
    """Build a continuity manifest from a real .gravity artifact's own header.

    executed_model / source_weight_hash come from the header's `model` block
    (repo / revision); artifact_hash comes from `integrity.body_sha256`. A
    continuity manifest that cannot name the exact artifact it describes is
    worthless, so this raises on a missing/malformed header -- it never emits a
    placeholder hash. `requested_model` defaults to the executed model (no
    substitution); pass it explicitly to express one (SS8.4).
    """
    artifact_path = Path(artifact_path)
    header = G.read_header(artifact_path)  # raises GravityFormatError / OSError

    model_block = header.get("model") or {}
    executed_model = model_block.get("repo")
    source_weight_hash = model_block.get("revision")
    if not executed_model or not source_weight_hash:
        raise ValueError(
            f"{artifact_path}: header.model is missing repo/revision -- refusing to "
            "build a continuity manifest that cannot name its own artifact")

    body_sha256 = (header.get("integrity") or {}).get("body_sha256")
    if not body_sha256:
        raise ValueError(f"{artifact_path}: header.integrity.body_sha256 missing")

    return {
        "requested_model": requested_model or executed_model,
        "executed_model": executed_model,
        "source_weight_hash": source_weight_hash,
        "artifact_hash": body_sha256,
        # real facts read off the header, not invented: which reader ABI this
        # shard requires to be decoded at all.
        "decoder_abi": decoder_abi or f"{header['schema']}+v{header['format_version']}",
        "profile_hash": profile_hash,
        "policy_hash": policy_hash,
        "system_prompt_hash": system_prompt_hash,
        "tool_schema_hash": tool_schema_hash,
        "fallback_allowed": fallback_allowed,
        "fallback_events": fallback_events or [],
    }


def build_continuity_manifest_multi_shard(model_dir: Path, *, requested_model: str | None = None,
                                          decoder_abi: str | None = None,
                                          profile_hash: str | None = None,
                                          policy_hash: str | None = None,
                                          system_prompt_hash: str | None = None,
                                          tool_schema_hash: str | None = None,
                                          fallback_allowed: bool = False,
                                          fallback_events: list | None = None) -> dict:
    """Same contract as `build_continuity_manifest`, for a model assembled across
    many `.gravity` shards (`model_dir/model.gravity.index.json`).

    A single shard has no `integrity.body_sha256` for the WHOLE model -- that field
    is per-shard by construction. So `artifact_hash` here is the sha256 of the
    assembler's own manifest content (every tensor's owning shard, the synthesized
    architecture, the coverage verdict): change which shard covers which tensor, or
    which architecture was synthesized, and this hash changes. It names the exact
    assembled model the same way a single body hash names a single shard.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.gravity.index.json"
    manifest = json.loads(index_path.read_text())

    model_block = manifest.get("model") or {}
    executed_model = model_block.get("repo")
    source_weight_hash = model_block.get("revision")
    if not executed_model or not source_weight_hash:
        raise ValueError(
            f"{index_path}: model.repo/revision missing -- refusing to build a "
            "continuity manifest that cannot name its own artifact")
    if manifest.get("coverage", {}).get("verdict") != "COMPLETE":
        raise ValueError(
            f"{index_path}: coverage verdict is {manifest.get('coverage', {}).get('verdict')!r}, "
            "not COMPLETE -- refusing to seal continuity for an incomplete model")

    return {
        "requested_model": requested_model or executed_model,
        "executed_model": executed_model,
        "source_weight_hash": source_weight_hash,
        "artifact_hash": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "artifact_hash_basis": "sha256 of model.gravity.index.json (multi-shard: no "
                               "single body hash names the whole model)",
        "shard_count": manifest.get("shard_count"),
        "decoder_abi": decoder_abi or "hawking.gravity.shard_header.v1+v1",
        "profile_hash": profile_hash,
        "policy_hash": policy_hash,
        "system_prompt_hash": system_prompt_hash,
        "tool_schema_hash": tool_schema_hash,
        "fallback_allowed": fallback_allowed,
        "fallback_events": fallback_events or [],
    }


# ---------------------------------------------------------------------------
# 2. Sovereignty event log (Rev3 Appendix C)
# ---------------------------------------------------------------------------

def _event_id(fields: dict) -> str:
    """Content-addressed id: sha256 over the event's own sorted fields, truncated
    to 16 hex chars. Any field rewrite changes the id, so an event cannot be
    silently altered in place -- see verify_event."""
    canon = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def make_event(*, plane: str, owner: str | None = None, reason: str | None = None,
               scope: str | None = None, requested_model: str | None = None,
               executed_model: str | None = None, original_prompt_hash: str | None = None,
               effective_prompt_hash: str | None = None, transformation: str | None = None,
               user_visible: bool = True, reversible: bool = True,
               ledger_ref: str | None = None, timestamp: str | None = None) -> dict:
    """Appendix C sovereignty_event. Only `plane` is enforced here -- owner / reason
    / scope / ledger_ref are left optional so attribution_completeness has a real
    quantity to measure instead of a field that is always full by construction."""
    if plane not in PLANES:
        raise ValueError(f"invalid plane {plane!r}, must be one of {sorted(PLANES)}")
    fields = {
        "timestamp": timestamp or _now(),
        "plane": plane, "owner": owner, "reason": reason, "scope": scope,
        "requested_model": requested_model, "executed_model": executed_model,
        "original_prompt_hash": original_prompt_hash,
        "effective_prompt_hash": effective_prompt_hash,
        "transformation": transformation,
        "user_visible": user_visible, "reversible": reversible,
        "ledger_ref": ledger_ref,
    }
    return {"id": _event_id(fields), **fields}


def verify_event(event: dict) -> bool:
    """True iff `event["id"]` still matches its own content -- i.e. nothing in the
    event was rewritten since it was minted."""
    fields = {k: v for k, v in event.items() if k != "id"}
    return event.get("id") == _event_id(fields)


def log_path(state_dir: Path) -> Path:
    return Path(state_dir) / EVENT_LOG_NAME


def append_event(state_dir: Path, event: dict) -> Path:
    """Append-only: one JSON row per event, newline-delimited."""
    path = log_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def read_events(state_dir: Path) -> list[dict]:
    path = log_path(state_dir)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# 3. Refusal attribution (Rev3 SS8.6)
# ---------------------------------------------------------------------------

def attribute(code: str, detail: str) -> dict:
    """A generic 'I cannot help' is insufficient inside Hawking (SS8.6): an
    unclassified code raises instead of degrading to a generic refusal record."""
    if code not in REFUSAL_CODES:
        raise ValueError(
            f"unclassified refusal code {code!r}; must be one of "
            f"{sorted(REFUSAL_CODES)} -- a generic refusal is insufficient "
            "inside Hawking (SS8.6)")
    return {"code": code, "meaning": REFUSAL_CODES[code], "detail": detail}


# ---------------------------------------------------------------------------
# 4. Limit registry (Rev3 SS8.11)
# ---------------------------------------------------------------------------

def add_limit(registry: dict, *, id: str, owner: str, type: str, reason: str,
              scope: str, threshold, current_value=None,
              user_changeable: bool = False) -> dict:
    """The point of the registry is that no limit is invisible, so a limit with no
    owner or no reason is rejected HERE, at insert, not at read time."""
    if not owner:
        raise ValueError(f"limit {id!r}: owner is required (no invisible limits)")
    if not reason:
        raise ValueError(f"limit {id!r}: reason is required (no invisible limits)")
    if id in registry:
        raise ValueError(f"limit {id!r} already registered")
    entry = {
        "id": id, "owner": owner, "type": type, "reason": reason, "scope": scope,
        "threshold": threshold, "current_value": current_value,
        "user_changeable": user_changeable, "event_log": [],
    }
    registry[id] = entry
    return entry


def list_limits(registry: dict) -> list[dict]:
    return list(registry.values())


def check(registry: dict, id: str, value) -> dict:
    """Update current_value and, when the limit binds (value >= threshold), append
    a record to that limit's own event_log -- so a bound limit is visible on the
    limit entry itself, not just inferable from current_value."""
    entry = registry[id]
    entry["current_value"] = value
    bound = value >= entry["threshold"]
    if bound:
        entry["event_log"].append({"timestamp": _now(), "value": value,
                                   "threshold": entry["threshold"], "bound": True})
    return {"id": id, "value": value, "threshold": entry["threshold"], "bound": bound}


# ---------------------------------------------------------------------------
# 5. Sovereignty metrics (Rev3 SS8.10) -- the 3 of 5 that are pure functions of
# the event log. There is no served runtime yet, so "turns" does not exist as a
# tracked quantity; each logged sovereignty_event IS one recorded intervention,
# which is the only honest proxy available today.
# ponytail: event-count is the "turns" proxy; once a runtime logs real per-turn
# events, feed those in here instead -- the metric functions don't change.
# ---------------------------------------------------------------------------

def hidden_intervention_rate(events: list[dict]) -> float:
    """unreported transformations or substitutions / evaluated turns (SS8.10).
    Every transformation this module logs IS reported (append-only log); "hidden"
    is operationalized as logged-but-not-surfaced-to-the-user: user_visible=False.
    """
    if not events:
        return 0.0
    hidden = sum(1 for e in events if not e.get("user_visible", True))
    return hidden / len(events)


def model_continuity_rate(events: list[dict]) -> float:
    """turns executed by requested artifact / total turns (SS8.10). Each event
    carries requested_model/executed_model (Appendix C); it counts as a
    substitution iff both are set and differ. No events -> vacuously 1.0."""
    if not events:
        return 1.0
    substituted = sum(1 for e in events
                      if e.get("requested_model") and e.get("executed_model")
                      and e["requested_model"] != e["executed_model"])
    return (len(events) - substituted) / len(events)


def attribution_completeness(events: list[dict]) -> float:
    """interventions with owner, reason, scope, and record / all interventions
    (SS8.10). "record" is the ledger_ref field."""
    if not events:
        return 1.0
    complete = sum(1 for e in events
                   if e.get("owner") and e.get("reason") and e.get("scope")
                   and e.get("ledger_ref"))
    return complete / len(events)


# ---------------------------------------------------------------------------
# seal: assemble the full deterministic sovereignty block for one artifact
# ---------------------------------------------------------------------------

def seal(artifact_path: Path, state_dir: Path = DEFAULT_STATE_DIR, *,
         requested_model: str | None = None) -> dict:
    # A directory means a multi-shard assembled model; a file means one .gravity
    # shard. Dispatching on what was actually given, not on a flag the caller has
    # to remember to set correctly.
    if artifact_path.is_dir():
        manifest = build_continuity_manifest_multi_shard(
            artifact_path, requested_model=requested_model)
    else:
        manifest = build_continuity_manifest(artifact_path, requested_model=requested_model)
    events = read_events(state_dir)
    return {
        "continuity": manifest,
        "sovereignty": {
            "hidden_intervention_rate": hidden_intervention_rate(events),
            "model_continuity_rate": model_continuity_rate(events),
            "attribution_completeness": attribution_completeness(events),
        },
        "event_count": len(events),
        "gated_stages": dict(GATED_METRICS),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capability Sovereignty spine (deterministic half)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_seal = sub.add_parser("seal", help="continuity + sovereignty block for one artifact")
    ap_seal.add_argument("--artifact", required=True)
    ap_seal.add_argument("--state", default=str(DEFAULT_STATE_DIR))
    ap_seal.add_argument("--requested-model", default=None)

    ap_event = sub.add_parser("event", help="append one sovereignty event")
    ap_event.add_argument("--state", default=str(DEFAULT_STATE_DIR))
    ap_event.add_argument("--plane", required=True, choices=sorted(PLANES))
    ap_event.add_argument("--owner", required=True)
    ap_event.add_argument("--reason", required=True)
    ap_event.add_argument("--scope", required=True)
    ap_event.add_argument("--requested-model", default=None)
    ap_event.add_argument("--executed-model", default=None)
    ap_event.add_argument("--original-prompt-hash", default=None)
    ap_event.add_argument("--effective-prompt-hash", default=None)
    ap_event.add_argument("--transformation", default=None)
    ap_event.add_argument("--ledger-ref", default=None)
    ap_event.add_argument("--not-user-visible", action="store_true")
    ap_event.add_argument("--not-reversible", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "seal":
        result = seal(Path(a.artifact), Path(a.state), requested_model=a.requested_model)
        print(json.dumps(result, indent=2))
    else:
        event = make_event(
            plane=a.plane, owner=a.owner, reason=a.reason, scope=a.scope,
            requested_model=a.requested_model, executed_model=a.executed_model,
            original_prompt_hash=a.original_prompt_hash,
            effective_prompt_hash=a.effective_prompt_hash,
            transformation=a.transformation, ledger_ref=a.ledger_ref,
            user_visible=not a.not_user_visible, reversible=not a.not_reversible)
        path = append_event(Path(a.state), event)
        print(json.dumps({"appended": event, "log": str(path)}, indent=2))


if __name__ == "__main__":
    main()
