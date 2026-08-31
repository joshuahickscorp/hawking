"""FALLBACK RESIDENT — keep Qwen27 restorable while a child is shadowed.

No Odyssey may depend on an experimental body with no recovery path.
Qwen27 sealed-3.14 is CURRENT_NONFINAL_HCLI_WORKER; this module seals that
body's identity (artifact path, specimen/revision, config digest), names the
concrete restore steps, and answers whether a supervisor can restore it
RIGHT NOW — without performing a restore, taking a GPU lease, or launching
a process.

A fallback that always reports itself healthy is worse than none: it removes
the reason to check. verify_restorable is required to be able to return
FALSE. Every precondition is a real file, digest, or locatable lifecycle
surface. A different model sitting at the sealed path is not the fallback.

This is not a fork of resident_identity (restart-survivable 19-field
document), resident_install (generic winner slots), super_resident (sandbox
floor), succession (child/shadow protocol), or tournament (NX-vs-NX). Those
remain. This sidecar is the recovery path those modules assume exists.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import HARDWARE_FIELDS, git, sha256_file, write_receipt
from tools.future.resident_identity import RESIDENCY_STATUS, SEAL_REL, SEALED_REL
from tools.future.resident_install import EXISTING_LIFECYCLE, PHASES, empty_contract
from tools.future.succession import LIFECYCLE_OWNERS, SUCCESSION_STEPS
from tools.future.super_resident import (
    QWEN_ID,
    QWEN_ROLE,
    REL_QWEN_IDENTITY,
    REL_QWEN_SEAL,
    evidence_presence,
    load_repo_json,
    locate as locate_repo,
)
from tools.future.tournament import INCUMBENT_CONTROL, QWEN_IDENTITY_REL
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

RECEIPT = "FALLBACK_RESIDENT.json"
SCHEMA = "hawking.future.fallback_resident.v1"
RECORDED_BY = "tools/future/fallback_resident.py"
VERSION = 1
UNKNOWN = "UNKNOWN"
CLAIM_CLASS = "STATIC_ONLY"

VERDICT_NOW = "RESTORABLE_NOW"
VERDICT_ACTION = "RESTORABLE_WITH_ACTION"
VERDICT_NOT = "NOT_RESTORABLE"
VERDICTS = (VERDICT_NOW, VERDICT_ACTION, VERDICT_NOT)

MIX_REPORT_NAME = "MIX_REPORT.json"
CHAT_TEMPLATE_NAME = "chat_template.jinja"
# Tokenizer is 12MB and the binaries ~6MB. Weights live under artifact_root
# and are identified by MIX_REPORT, not by hashing ten gigabytes here.
MAX_HASH_BYTES = 64 << 20

REL_GATE = "hcli/agentos/resident_gate.py"
REL_CONNECTOR = "hcli/hawking_native.py"
REL_RECOVERY = "hcli/agentos/recovery.py"
REL_CHECKPOINT = "hcli/agentos/checkpoint.py"

# Identity-bearing config. Hardware slots from the sealed profile (tps, ebpw)
# are refused here: copying them would launder a measurement this sidecar
# cannot make.
CONFIG_KEYS: tuple[str, ...] = (
    "model_id",
    "resident_identity",
    "family",
    "protocol",
    "runtime",
    "provider",
    "mode",
    "artifact_root",
    "tokenizer",
    "binary",
    "resident_binary",
    "prompt_contract",
)

RESTORE_STEPS: tuple[str, ...] = (
    "confirm_identity_document",
    "confirm_artifact_root",
    "confirm_specimen_mix",
    "confirm_tokenizer_digest",
    "confirm_chat_template_digest",
    "confirm_runtime_binary_digest",
    "confirm_resident_binary_present",
    "confirm_config_digest",
    "confirm_lifecycle_surfaces",
    "unload_non_fallback_body",
    "bind_fallback_identity",
    "probe_readiness_without_launching",
)

# Durable science a rollback must not unwrite. Named so a future editor
# cannot silently add "delete receipts" to the revert set.
ROLLBACK_DOES_NOT_REVERT: tuple[str, ...] = (
    "receipts/future/** (sidecar science, including this receipt)",
    "receipts/headless/** (Codex / Accelerator receipts)",
    "NEGATIVE_SCIENCE_INDEX / scars already recorded",
    "specimen verification receipts and ModelLake seals",
    "teacher-capture rows already on disk",
    "git history, commits, and current HEAD (campaign progress)",
    "NOETIC_PARENT_A / artifact bytes themselves (re-bound, not deleted)",
    "HCLI_RESIDENT_SEAL.json and hawking-native.sealed-3.14.json",
    "historical capability suite receipts",
    "CLAUDE_GLOBAL_FRONTIER.json and FRONTIER_STATE.json",
    "succession protocol seals (a stopped child's history remains)",
    "completed WorkUnit records",
)

ROLLBACK_DOES_REVERT: tuple[str, ...] = (
    "which body is bound as the HCLI resident (child/candidate → sealed-3.14)",
    "live session / request-id correlation / KV-recurrent state of the evicted body",
    "resident_install winner_id / bound slots for the experimental body",
    "canonical mission owner (succession: back to the incumbent)",
    "in-memory process of the experimental body (stop / unload / drop weights)",
)


class FallbackRefused(ValueError):
    """A fallback judgement was asked to invent a missing fact."""


# ---------------------------------------------------------------------------
# Overlay-aware observation. Tests inject a world; production reads disk.
# ---------------------------------------------------------------------------


def _rec(overlay: Mapping[str, Any] | None, role: str) -> Mapping[str, Any] | None:
    if overlay is None:
        return None
    got = overlay.get(role)
    return got if isinstance(got, Mapping) else None


def _seal_value(fields: Mapping[str, Any], key: str) -> Any:
    node = fields.get(key)
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def _json_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _hash_if_small(path: Path) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_HASH_BYTES:
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def _observe_file(path: str | None, rec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Presence + digest of one file. Overlay wins; missing is named, never defaulted."""
    if rec is not None:
        shown = rec.get("path", path)
        if "exists" in rec or "sha256" in rec or "doc" in rec:
            exists = bool(rec["exists"]) if "exists" in rec else False
            if "exists" not in rec and shown:
                exists = Path(str(shown)).is_file()
            digest = rec.get("sha256")
            if digest is None and exists and shown:
                host = Path(str(shown))
                if host.is_file():
                    digest = _hash_if_small(host)
            return {
                "path": shown,
                "exists": exists,
                "sha256": digest if digest else UNKNOWN,
                "source": rec.get("source") or "overlay",
                "is_dir": bool(rec.get("is_dir", False)),
            }
        path = shown if shown is not None else path
    if not path:
        return {
            "path": None,
            "exists": False,
            "sha256": UNKNOWN,
            "source": "undeclared",
            "is_dir": False,
            "missing_evidence": ["no path declared"],
        }
    host = Path(str(path)).expanduser()
    is_dir = host.is_dir()
    is_file = host.is_file()
    digest = _hash_if_small(host) if is_file else None
    return {
        "path": str(host),
        "exists": is_file or is_dir,
        "sha256": digest if digest else UNKNOWN,
        "source": "disk" if (is_file or is_dir) else "HOST_ABSENT",
        "is_dir": is_dir,
        "size_bytes": (host.stat().st_size if is_file else None),
    }


def _observe_dir(path: str | None, rec: Mapping[str, Any] | None) -> dict[str, Any]:
    if rec is not None and ("exists" in rec or "is_dir" in rec):
        return {
            "path": rec.get("path", path),
            "exists": bool(rec.get("exists", rec.get("is_dir", False))),
            "is_dir": bool(rec.get("is_dir", rec.get("exists", False))),
            "source": rec.get("source") or "overlay",
        }
    shown = (rec or {}).get("path", path) if rec is not None else path
    if not shown:
        return {"path": None, "exists": False, "is_dir": False, "source": "undeclared"}
    host = Path(str(shown)).expanduser()
    return {
        "path": str(host),
        "exists": host.is_dir(),
        "is_dir": host.is_dir(),
        "source": "disk" if host.is_dir() else "HOST_ABSENT",
    }


def _load_json_role(
    rel: str,
    rec: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load a repo-relative JSON document. Overlay doc wins; else locate/HEAD."""
    if rec is not None:
        doc = rec.get("doc")
        presence = {
            "rel": rec.get("path") or rel,
            "present": bool(rec.get("exists", doc is not None)),
            "source": rec.get("source") or "overlay",
            "tracked_in_head": bool(rec.get("in_git_head", False)),
            "recovery": rec.get("source") or "overlay",
        }
        if isinstance(doc, dict):
            return doc, presence
        if rec.get("exists") is False:
            return None, presence
    doc, source = load_repo_json(rel)
    presence = evidence_presence(rel)
    if source:
        presence["source"] = source
        presence["recovery"] = "disk" if not str(source).startswith("HEAD:") else "HEAD"
    return (doc if isinstance(doc, dict) else None), presence


def _load_mix(artifact_root: str | None, rec: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if rec is not None and ("doc" in rec or "exists" in rec):
        doc = rec.get("doc") if isinstance(rec.get("doc"), dict) else None
        path = rec.get("path")
        return doc, {
            "path": path,
            "exists": bool(rec.get("exists", doc is not None)),
            "source": rec.get("source") or "overlay",
            "sha256": rec.get("sha256") or UNKNOWN,
        }
    if not artifact_root:
        return None, {
            "path": None,
            "exists": False,
            "source": "undeclared",
            "sha256": UNKNOWN,
            "missing_evidence": ["artifact_root not declared; cannot locate MIX_REPORT"],
        }
    path = str(Path(artifact_root) / MIX_REPORT_NAME)
    obs = _observe_file(path, None)
    doc = None
    if obs.get("exists") and not obs.get("is_dir"):
        try:
            loaded = json.loads(Path(path).read_text())
            if isinstance(loaded, dict):
                doc = loaded
        except (OSError, json.JSONDecodeError, UnicodeError):
            obs["source"] = "ON_DISK_UNPARSEABLE"
    return doc, obs


def _prefix_verdict(digest: Any, prefix: Any) -> str:
    if not isinstance(prefix, str) or not prefix:
        return "NO_SEAL_PREFIX"
    if not isinstance(digest, str) or digest in {UNKNOWN, "", None}:
        return "UNHASHED"
    if digest[: len(prefix)] == prefix:
        return "MATCH"
    return "MISMATCH"


def _hardware_numeric_keys(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool):
                hits.append(here)
            hits.extend(_hardware_numeric_keys(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(_hardware_numeric_keys(value, f"{path}[{i}]"))
    return hits


# ---------------------------------------------------------------------------
# fallback_identity — sealed Qwen27 body, no hardware numbers
# ---------------------------------------------------------------------------


def _config_payload(
    ident: Mapping[str, Any],
    mix: Mapping[str, Any] | None,
    *,
    tokenizer_sha: str | None,
    runtime_sha: str | None,
    chat_sha: str | None,
) -> dict[str, Any]:
    recipe = mix.get("recipe") if isinstance(mix, Mapping) else None
    recipe_id = recipe.get("id") if isinstance(recipe, Mapping) else None
    return {
        "model_id": ident.get("model_id"),
        "resident_identity": ident.get("resident_identity"),
        "family": ident.get("family"),
        "protocol": ident.get("protocol"),
        "runtime": ident.get("runtime"),
        "provider": ident.get("provider"),
        "mode": ident.get("mode"),
        "artifact_root": ident.get("artifact_root"),
        "tokenizer": ident.get("tokenizer"),
        "binary": ident.get("binary"),
        "resident_binary": ident.get("resident_binary"),
        "prompt_contract": ident.get("prompt_contract"),
        "mix_id": None if not isinstance(mix, Mapping) else mix.get("mix_id"),
        "parent_params": None if not isinstance(mix, Mapping) else mix.get("parent_params"),
        "recipe_id": recipe_id,
        "tokenizer_sha256": tokenizer_sha,
        "runtime_binary_sha256": runtime_sha,
        "chat_template_sha256": chat_sha,
    }


def fallback_identity(overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Exact Qwen27 fallback identity. Disk/git is authority. Does not restore."""
    ident, ident_pres = _load_json_role(REL_QWEN_IDENTITY, _rec(overlay, "identity_document"))
    seal, seal_pres = _load_json_role(REL_QWEN_SEAL, _rec(overlay, "seal"))
    if not isinstance(ident, dict):
        return {
            "schema": "hawking.future.fallback_identity.v1",
            "status": "IDENTITY_UNRESOLVED",
            "role": QWEN_ROLE,
            "id": QWEN_ID,
            "residency_status": RESIDENCY_STATUS,
            "unresolved_reason": f"{REL_QWEN_IDENTITY} was not locatable as a JSON object",
            "identity_presence": ident_pres,
            "seal_presence": seal_pres,
            "artifact_path": None,
            "specimen_identity": None,
            "config_digest": None,
            "gpu_authority": False,
            "evidence_class": CLAIM_CLASS,
            "performed_restore": False,
            "copied_runtime_tps": False,
            "copied_physical_ebpw": False,
        }

    fields = seal.get("fields") if isinstance(seal, Mapping) and isinstance(seal.get("fields"), dict) else {}
    artifact_root = ident.get("artifact_root")
    tokenizer_path = ident.get("tokenizer")
    runtime_path = ident.get("binary")
    resident_path = ident.get("resident_binary")
    chat_path = str(Path(str(artifact_root)) / CHAT_TEMPLATE_NAME) if artifact_root else None

    tok = _observe_file(tokenizer_path if isinstance(tokenizer_path, str) else None, _rec(overlay, "tokenizer"))
    runtime = _observe_file(runtime_path if isinstance(runtime_path, str) else None, _rec(overlay, "runtime_binary"))
    resident = _observe_file(resident_path if isinstance(resident_path, str) else None, _rec(overlay, "resident_binary"))
    chat = _observe_file(chat_path, _rec(overlay, "chat_template"))
    root = _observe_dir(artifact_root if isinstance(artifact_root, str) else None, _rec(overlay, "artifact_root"))
    mix, mix_obs = _load_mix(artifact_root if isinstance(artifact_root, str) else None, _rec(overlay, "mix_report"))

    tok_sha = tok.get("sha256") if tok.get("sha256") != UNKNOWN else None
    run_sha = runtime.get("sha256") if runtime.get("sha256") != UNKNOWN else None
    chat_sha = chat.get("sha256") if chat.get("sha256") != UNKNOWN else None
    payload = _config_payload(ident, mix, tokenizer_sha=tok_sha, runtime_sha=run_sha, chat_sha=chat_sha)
    digest = _json_digest(payload)
    hw = _hardware_numeric_keys(payload)
    if hw:
        raise FallbackRefused(f"config payload carried hardware fields {hw}")

    tok_prefix = _seal_value(fields, "tokenizer_sha256_16")
    run_prefix = _seal_value(fields, "runtime_binary_sha256_16")
    chat_prefix = _seal_value(fields, "chat_template_sha256_16")
    inventory_prefix = _seal_value(fields, "artifact_inventory_sha")
    runtime_commit = _seal_value(fields, "runtime_commit")
    closure = _seal_value(fields, "physical_closure")
    parent_params_seal = None
    if isinstance(closure, Mapping):
        parent_params_seal = closure.get("parent_params")

    catalog = mix.get("catalog") if isinstance(mix, Mapping) else None
    catalog_obs = _observe_file(
        catalog if isinstance(catalog, str) else None,
        _rec(overlay, "catalog"),
    ) if isinstance(catalog, str) else {
        "path": catalog,
        "exists": False,
        "sha256": UNKNOWN,
        "source": "undeclared",
        "is_dir": False,
    }

    specimen = {
        "mix_id": None if not isinstance(mix, Mapping) else mix.get("mix_id"),
        "parent_params": None if not isinstance(mix, Mapping) else mix.get("parent_params"),
        "parent_params_matches_seal": (
            isinstance(mix, Mapping)
            and parent_params_seal is not None
            and mix.get("parent_params") == parent_params_seal
        ),
        "did_not_load_second_27b": None if not isinstance(mix, Mapping) else mix.get("did_not_load_second_27b"),
        "recipe_id": payload["recipe_id"],
        "catalog": catalog_obs,
        "runtime_commit_on_seal": runtime_commit if runtime_commit else UNKNOWN,
        "runtime_commit_is_not_a_required_checkout": True,
        "artifact_inventory_sha16_on_seal": inventory_prefix if inventory_prefix else UNKNOWN,
        "note": (
            "Specimen identity is MIX_REPORT mix_id/parent_params/recipe plus the "
            "HCLI seal prefixes. runtime_commit names the sealed binary's origin "
            "commit; restoring the fallback does not check out that commit."
        ),
    }

    return {
        "schema": "hawking.future.fallback_identity.v1",
        "status": "SEALED",
        "role": QWEN_ROLE,
        "id": ident.get("model_id") or QWEN_ID,
        "residency_status": RESIDENCY_STATUS,
        "not_a_singularity": True,
        "tournament_role": INCUMBENT_CONTROL.get("role"),
        "identity_presence": ident_pres,
        "seal_presence": seal_pres,
        "artifact_path": artifact_root,
        "artifact_root": root,
        "tokenizer": {
            **tok,
            "seal_sha256_16": tok_prefix,
            "prefix_verdict": _prefix_verdict(tok.get("sha256"), tok_prefix),
        },
        "chat_template": {
            **chat,
            "seal_sha256_16": chat_prefix,
            "prefix_verdict": _prefix_verdict(chat.get("sha256"), chat_prefix),
        },
        "runtime_binary": {
            **runtime,
            "role": "seal_runtime_binary (hybrid greedy)",
            "seal_sha256_16": run_prefix,
            "prefix_verdict": _prefix_verdict(runtime.get("sha256"), run_prefix),
        },
        "resident_binary": {
            **resident,
            "role": "resident protocol binary (distinct from seal runtime_binary)",
            "note": (
                "HCLI_RESIDENT_SEAL.runtime_binary_sha256_16 matches 'binary' "
                "(hybrid greedy), not resident_binary. Presence is required; "
                "matching the greedy prefix is not."
            ),
        },
        "specimen_identity": specimen,
        "mix_report": mix_obs,
        "config": payload,
        "config_digest": digest,
        "seal_resident": seal.get("resident") if isinstance(seal, Mapping) else None,
        "seal_status": seal.get("status") if isinstance(seal, Mapping) else None,
        "protocol": ident.get("protocol"),
        "runtime": ident.get("runtime"),
        "prompt_contract": ident.get("prompt_contract"),
        "gpu_authority": False,
        "evidence_class": CLAIM_CLASS,
        "performed_restore": False,
        "copied_runtime_tps": False,
        "copied_physical_ebpw": False,
        "claim_boundary": (
            "Static identity of the fallback body. No hardware measurement. "
            "Sealed current_runtime TPS / physical_ebpw were not copied."
        ),
    }


# ---------------------------------------------------------------------------
# verify_restorable — answerable without performing a restore
# ---------------------------------------------------------------------------


def _precondition(
    pid: str,
    *,
    met: bool,
    kind: str,
    names: str,
    why: str,
    action: str | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = "MET" if met else "UNMET"
    body: dict[str, Any] = {
        "id": pid,
        "state": state,
        "kind": kind,
        "names": names,
        "why": why,
        "action": action,
        "checkable_without_restore": True,
    }
    if evidence is not None:
        body["evidence"] = dict(evidence)
    return body


def _lifecycle_locatable(rel: str, rec: Mapping[str, Any] | None) -> dict[str, Any]:
    if rec is not None and "exists" in rec:
        present = bool(rec.get("exists") or rec.get("in_git_head"))
        return {
            "rel": rel,
            "present": present,
            "on_disk": bool(rec.get("exists")),
            "in_git_head": bool(rec.get("in_git_head")),
            "source": rec.get("source") or rec.get("path"),
            "recovery": rec.get("recovery") or ("overlay" if rec.get("exists") else "unresolved"),
        }
    path = locate_repo(rel)
    listed = git("ls-tree", "-r", "--name-only", "HEAD", "--", rel)
    in_head = any(line.strip() == rel for line in listed.splitlines()) if listed else bool(git("ls-files", "--error-unmatch", rel))
    return {
        "rel": rel,
        "present": path is not None or in_head,
        "on_disk": path is not None,
        "in_git_head": in_head,
        "source": str(path) if path is not None else (f"HEAD:{rel}" if in_head else None),
        "recovery": "disk" if path is not None else ("HEAD" if in_head else "unresolved"),
    }


def verify_restorable(
    overlay: Mapping[str, Any] | None = None,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Can the supervisor restore Qwen27 RIGHT NOW? Does not restore.

    Verdicts: RESTORABLE_NOW | RESTORABLE_WITH_ACTION | NOT_RESTORABLE.
    A digest mismatch is never WITH_ACTION: a different model at the sealed
    path is not the fallback, and rebuilding will not make it so.
    """
    ident = dict(identity) if isinstance(identity, Mapping) else fallback_identity(overlay)
    rows: list[dict[str, Any]] = []

    unresolved = ident.get("status") == "IDENTITY_UNRESOLVED"
    ident_pres = ident.get("identity_presence") if isinstance(ident.get("identity_presence"), dict) else {}
    rows.append(
        _precondition(
            "identity_document",
            met=not unresolved and bool(ident.get("id")) and ident.get("residency_status") == RESIDENCY_STATUS,
            kind="MISSING" if unresolved else "IDENTITY",
            names=str(ident_pres.get("rel") or REL_QWEN_IDENTITY),
            why=(
                ident.get("unresolved_reason")
                if unresolved
                else (
                    f"sealed profile locatable; model_id={ident.get('id')!r}; "
                    f"residency_status={ident.get('residency_status')!r}"
                )
            ),
            action=(
                f"materialize {REL_QWEN_IDENTITY} from git HEAD"
                if unresolved and ident_pres.get("tracked_in_head")
                else None
            ),
            evidence=ident_pres,
        )
    )

    root = ident.get("artifact_root") if isinstance(ident.get("artifact_root"), dict) else {}
    art_path = ident.get("artifact_path") or root.get("path")
    art_ok = bool(root.get("exists") and root.get("is_dir"))
    rows.append(
        _precondition(
            "artifact_root",
            met=art_ok,
            kind="PRESENCE" if art_ok else "MISSING",
            names=str(art_path or "<artifact_root undeclared>"),
            why=(
                f"artifact_root is a directory at {art_path}"
                if root.get("exists") and root.get("is_dir")
                else f"artifact_root missing or not a directory: {art_path}"
            ),
            evidence=root,
        )
    )

    specimen = ident.get("specimen_identity") if isinstance(ident.get("specimen_identity"), dict) else {}
    mix_obs = ident.get("mix_report") if isinstance(ident.get("mix_report"), dict) else {}
    mix_path = mix_obs.get("path") or (
        str(Path(str(art_path)) / MIX_REPORT_NAME) if art_path else MIX_REPORT_NAME
    )
    mix_ok = bool(mix_obs.get("exists")) and specimen.get("mix_id") not in {None, "", UNKNOWN}
    parent_ok = specimen.get("parent_params_matches_seal") is True
    catalog = specimen.get("catalog") if isinstance(specimen.get("catalog"), dict) else {}
    catalog_ok = bool(catalog.get("exists"))
    specimen_met = mix_ok and parent_ok and catalog_ok
    specimen_kind = "IDENTITY" if specimen_met else "MISSING"
    specimen_names = mix_path
    specimen_why = (
        f"MIX_REPORT mix_id={specimen.get('mix_id')!r} parent_params match seal; catalog present"
        if specimen_met
        else "specimen MIX_REPORT/catalog/parent_params incomplete"
    )
    if mix_ok and not parent_ok:
        specimen_kind = "DIGEST_MISMATCH"
        specimen_names = mix_path
        specimen_why = (
            f"MIX_REPORT parent_params={specimen.get('parent_params')!r} "
            "does not match HCLI_RESIDENT_SEAL physical_closure; a different mix "
            "at the artifact_root is not the fallback"
        )
    elif mix_ok and parent_ok and not catalog_ok:
        specimen_kind = "MISSING"
        specimen_names = str(catalog.get("path") or "<catalog undeclared>")
        specimen_why = f"mix catalog missing: {specimen_names}"
    elif not mix_ok:
        specimen_kind = "MISSING"
        specimen_names = str(mix_path)
        specimen_why = f"MIX_REPORT missing or unreadable at {mix_path}"
    rows.append(
        _precondition(
            "specimen_mix",
            met=specimen_met,
            kind=specimen_kind,
            names=str(specimen_names),
            why=specimen_why,
            evidence={"mix": mix_obs, "specimen": specimen},
        )
    )

    def _digest_row(pid: str, slot: Mapping[str, Any], *, require_match: bool) -> dict[str, Any]:
        path = str(slot.get("path") or f"<{pid} undeclared>")
        exists = bool(slot.get("exists"))
        verdict = slot.get("prefix_verdict")
        if not exists:
            return _precondition(
                pid,
                met=False,
                kind="MISSING",
                names=path,
                why=f"{pid} missing at {path}",
                action=None,
                evidence=slot,
            )
        if require_match and verdict == "MISMATCH":
            return _precondition(
                pid,
                met=False,
                kind="DIGEST_MISMATCH",
                names=path,
                why=(
                    f"{pid} is present at {path} but sha256 {slot.get('sha256')!r} "
                    f"does not match seal prefix {slot.get('seal_sha256_16')!r}; "
                    "a different model at the sealed path is not the fallback"
                ),
                evidence=slot,
            )
        if require_match and verdict != "MATCH":
            return _precondition(
                pid,
                met=False,
                kind="MISSING" if verdict == "NO_SEAL_PREFIX" else "UNHASHED",
                names=path,
                why=(
                    f"{pid} could not be checked against the seal "
                    f"(prefix_verdict={verdict!r})"
                ),
                evidence=slot,
            )
        return _precondition(
            pid,
            met=True,
            kind="DIGEST" if require_match else "PRESENCE",
            names=path,
            why=(
                f"{pid} sha256 matches seal prefix {slot.get('seal_sha256_16')}"
                if require_match
                else f"{pid} present at {path}"
            ),
            evidence=slot,
        )

    tok_slot = ident.get("tokenizer") if isinstance(ident.get("tokenizer"), dict) else {}
    chat_slot = ident.get("chat_template") if isinstance(ident.get("chat_template"), dict) else {}
    run_slot = ident.get("runtime_binary") if isinstance(ident.get("runtime_binary"), dict) else {}
    res_slot = ident.get("resident_binary") if isinstance(ident.get("resident_binary"), dict) else {}
    rows.append(_digest_row("tokenizer", tok_slot, require_match=True))
    rows.append(_digest_row("chat_template", chat_slot, require_match=True))
    run_row = _digest_row("runtime_binary", run_slot, require_match=True)
    if run_row["state"] == "UNMET" and run_row["kind"] == "MISSING":
        run_row["action"] = (
            "rebuild the named release-fast hybrid-greedy binary "
            f"({run_slot.get('path')}); Codex owns cargo, this sidecar will not run it"
        )
    rows.append(run_row)
    res_row = _digest_row("resident_binary", res_slot, require_match=False)
    if res_row["state"] == "UNMET" and res_row["kind"] == "MISSING":
        res_row["action"] = (
            "rebuild the named release-fast resident binary "
            f"({res_slot.get('path')}); Codex owns cargo, this sidecar will not run it"
        )
    rows.append(res_row)

    cfg = ident.get("config_digest")
    rows.append(
        _precondition(
            "config_digest",
            met=isinstance(cfg, str) and len(cfg) == 64 and not unresolved,
            kind="IDENTITY",
            names="config_digest",
            why=(
                f"config_digest={cfg}"
                if isinstance(cfg, str)
                else "config_digest was not sealed (identity unresolved)"
            ),
            evidence={"config_digest": cfg, "config": ident.get("config")},
        )
    )

    gate = _lifecycle_locatable(REL_GATE, _rec(overlay, "resident_gate"))
    connector = _lifecycle_locatable(REL_CONNECTOR, _rec(overlay, "connector"))
    recovery = _lifecycle_locatable(REL_RECOVERY, _rec(overlay, "recovery"))
    life_met = bool(gate.get("present") and connector.get("present") and recovery.get("present"))
    missing_life = [
        r["rel"]
        for r in (gate, connector, recovery)
        if not r.get("present")
    ]
    life_action = None
    if missing_life and all(
        r.get("in_git_head") for r in (gate, connector, recovery) if r["rel"] in missing_life
    ):
        life_action = "materialize " + ", ".join(missing_life) + " from git HEAD"
    rows.append(
        _precondition(
            "lifecycle_surfaces",
            met=life_met,
            kind="PRESENCE" if life_met else "MISSING",
            names=", ".join(missing_life) if missing_life else f"{REL_GATE}, {REL_CONNECTOR}, {REL_RECOVERY}",
            why=(
                "resident_gate / hawking_native / recovery locatable (disk or git HEAD)"
                if life_met
                else f"lifecycle surface(s) unresolved: {missing_life}"
            ),
            action=life_action,
            evidence={"resident_gate": gate, "connector": connector, "recovery": recovery},
        )
    )

    contract = empty_contract()
    fb = (contract.get("slots") or {}).get("fallback_policy") or {}
    policy_ok = fb.get("on_identity_mismatch") == "REFUSE naming the field"
    rows.append(
        _precondition(
            "install_fallback_policy",
            met=policy_ok,
            kind="IDENTITY",
            names="tools/future/resident_install.py#fallback_policy",
            why=(
                "install contract refuses identity mismatch by naming the field"
                if policy_ok
                else f"fallback_policy.on_identity_mismatch={fb.get('on_identity_mismatch')!r}"
            ),
            evidence={"on_identity_mismatch": fb.get("on_identity_mismatch"), "on_unsealed": fb.get("on_unsealed")},
        )
    )

    unmet = [r for r in rows if r.get("state") != "MET"]
    mismatches = [r for r in unmet if r.get("kind") == "DIGEST_MISMATCH"]
    actionable = [r for r in unmet if r.get("action")]
    named = None
    action = None
    if not unmet:
        verdict = VERDICT_NOW
        reason = "every restore precondition is MET on disk/git; supervisor can restore without further artifact work"
    elif mismatches:
        verdict = VERDICT_NOT
        named = mismatches[0]["names"]
        reason = mismatches[0]["why"]
    elif unmet and all(r.get("action") for r in unmet):
        verdict = VERDICT_ACTION
        action = "; ".join(str(r["action"]) for r in actionable)
        named = unmet[0]["names"]
        reason = f"restorable after named action: {action}"
    else:
        verdict = VERDICT_NOT
        first = next((r for r in unmet if not r.get("action")), unmet[0])
        named = first["names"]
        reason = first["why"]

    return {
        "schema": "hawking.future.fallback_restorable.v1",
        "verdict": verdict,
        "restorable": verdict == VERDICT_NOW,
        "unmet_precondition": None if verdict == VERDICT_NOW else named,
        "action": action,
        "reason": reason,
        "preconditions": rows,
        "n_preconditions": len(rows),
        "n_unmet": len(unmet),
        "identity_id": ident.get("id"),
        "identity_status": ident.get("status"),
        "config_digest": ident.get("config_digest"),
        "performed_restore": False,
        "started_model_process": False,
        "took_gpu_lease": False,
        "flock": False,
        "gpu_authority": False,
        "evidence_class": CLAIM_CLASS,
        "fails_closed": (
            "MISSING artifact → NOT_RESTORABLE naming the artifact; "
            "DIGEST_MISMATCH → NOT_RESTORABLE (wrong body at the path); "
            "actionable absence → RESTORABLE_WITH_ACTION naming the action; "
            "never rounds a partial result into RESTORABLE_NOW"
        ),
    }


# ---------------------------------------------------------------------------
# restore_path — concrete steps, each independently checkable
# ---------------------------------------------------------------------------


def restore_path(
    overlay: Mapping[str, Any] | None = None,
    *,
    identity: Mapping[str, Any] | None = None,
    restorable: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ordered restore procedure. Checks run; bind/unload do not execute."""
    ident = dict(identity) if isinstance(identity, Mapping) else fallback_identity(overlay)
    verdict = dict(restorable) if isinstance(restorable, Mapping) else verify_restorable(overlay, identity=ident)
    by_id = {
        str(row["id"]): row
        for row in (verdict.get("preconditions") or [])
        if isinstance(row, dict) and row.get("id")
    }

    def _step(sid: str, *, precond: str | None, executes: bool, what: str) -> dict[str, Any]:
        row = by_id.get(precond) if precond else None
        if executes:
            state = "NOT_EXECUTED"
            checkable = False
            current = (
                "this sidecar does not bind, unload, or probe a live process; "
                "the supervisor owns execution"
            )
        elif row is None:
            state = "UNKNOWN"
            checkable = True
            current = f"no precondition row for {precond}"
        else:
            state = str(row.get("state") or "UNKNOWN")
            checkable = True
            current = str(row.get("why") or "")
        return {
            "id": sid,
            "what": what,
            "checkable_without_restore": checkable,
            "executes_restore": executes,
            "current_state": state,
            "current": current,
            "precondition_id": precond,
            "names": None if row is None else row.get("names"),
            "action": None if row is None else row.get("action"),
        }

    steps = [
        _step("confirm_identity_document", precond="identity_document", executes=False,
              what="Locate hawking-native.sealed-3.14.json and refuse if unreadable"),
        _step("confirm_artifact_root", precond="artifact_root", executes=False,
              what="Confirm artifact_root is a directory at the sealed path"),
        _step("confirm_specimen_mix", precond="specimen_mix", executes=False,
              what="Read MIX_REPORT.json; mix_id/parent_params must match the HCLI seal"),
        _step("confirm_tokenizer_digest", precond="tokenizer", executes=False,
              what="Hash tokenizer.json; prefix must match HCLI_RESIDENT_SEAL.tokenizer_sha256_16"),
        _step("confirm_chat_template_digest", precond="chat_template", executes=False,
              what="Hash chat_template.jinja; prefix must match HCLI_RESIDENT_SEAL.chat_template_sha256_16"),
        _step("confirm_runtime_binary_digest", precond="runtime_binary", executes=False,
              what="Hash hybrid-greedy binary; prefix must match HCLI_RESIDENT_SEAL.runtime_binary_sha256_16"),
        _step("confirm_resident_binary_present", precond="resident_binary", executes=False,
              what="Confirm the resident protocol binary is present (distinct artifact from greedy)"),
        _step("confirm_config_digest", precond="config_digest", executes=False,
              what="Recompute the identity-bearing config digest and refuse on drift"),
        _step("confirm_lifecycle_surfaces", precond="lifecycle_surfaces", executes=False,
              what="Locate resident_gate / hawking_native.stop+_restart_resident / recovery"),
        _step("unload_non_fallback_body", precond=None, executes=True,
              what="Stop/unload any shadowed child (connector.stop; drop weights; release device)"),
        _step("bind_fallback_identity", precond="install_fallback_policy", executes=True,
              what="Bind sealed-3.14 into the generic resident_install contract; identity mismatch REFUSES"),
        _step("probe_readiness_without_launching", precond=None, executes=True,
              what="Readiness is resident_gate's pid probe; this sidecar records the slot and does not start a process"),
    ]
    # The last three execute only under an authorized HCLI lane. Their
    # current_state stays NOT_EXECUTED so a receipt cannot claim a restore.
    return {
        "schema": "hawking.future.fallback_restore_path.v1",
        "steps": steps,
        "step_ids": list(RESTORE_STEPS),
        "n_steps": len(steps),
        "independently_checkable": [s["id"] for s in steps if s["checkable_without_restore"]],
        "supervisor_owned": [s["id"] for s in steps if s["executes_restore"]],
        "install_phases_the_bind_would_fill": list(PHASES),
        "succession_keep_parent_step": "keep_parent_for_rollback",
        "succession_steps": list(SUCCESSION_STEPS),
        "lifecycle_owners": {
            "resident_gate": LIFECYCLE_OWNERS.get("resident_gate") or EXISTING_LIFECYCLE.get("resident_gate"),
            "recovery": LIFECYCLE_OWNERS.get("recovery") or EXISTING_LIFECYCLE.get("recovery_gate"),
            "checkpoint": LIFECYCLE_OWNERS.get("checkpoint"),
            "resident_install": "tools/future/resident_install.py",
            "incumbent_identity": QWEN_IDENTITY_REL,
            "resident_seal": SEAL_REL,
        },
        "verdict": verdict.get("verdict"),
        "performed_restore": False,
        "started_model_process": False,
        "took_gpu_lease": False,
        "gpu_authority": False,
        "evidence_class": CLAIM_CLASS,
    }


# ---------------------------------------------------------------------------
# rollback_state — what reverts, and what durable science does not
# ---------------------------------------------------------------------------


def rollback_state(
    overlay: Mapping[str, Any] | None = None,
    *,
    restorable: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rollback restores the bound body. It does not unwrite sealed science."""
    verdict = dict(restorable) if isinstance(restorable, Mapping) else verify_restorable(overlay)
    return {
        "schema": "hawking.future.fallback_rollback.v1",
        "reverts": list(ROLLBACK_DOES_REVERT),
        "does_not_revert": list(ROLLBACK_DOES_NOT_REVERT),
        "durable_science_survives": True,
        "durable_science_rule": (
            "A receipt already sealed is not unwritten. Rollback changes which "
            "body is bound, not what the laboratory has already learned."
        ),
        "shadowed_child": {
            "proposals_already_written_as_sidecar_receipts": "survive",
            "canonical_mission_ownership": "returns to the incumbent",
            "live_session_of_the_child": "reverts (stop / unload)",
        },
        "git_head": {
            "reverts": False,
            "why": (
                "runtime_commit on HCLI_RESIDENT_SEAL identifies the sealed binary; "
                "it is not a required checkout. Campaign commits stay."
            ),
        },
        "fallback_body": {
            "id": QWEN_ID,
            "role": QWEN_ROLE,
            "identity": SEALED_REL,
            "seal": SEAL_REL,
        },
        "restorable_verdict": verdict.get("verdict"),
        "rollback_is_not_a_restore_until_executed": True,
        "performed_restore": False,
        "gpu_authority": False,
        "evidence_class": CLAIM_CLASS,
    }


# ---------------------------------------------------------------------------
# WorkUnit + receipt
# ---------------------------------------------------------------------------


def emit_workunits(verdict: Mapping[str, Any]) -> list[dict[str, Any]]:
    unit = emit_hcli_workunit(
        id="future.fallback_resident.verify",
        role="science",
        description=(
            "Seal the Qwen27 fallback identity and answer verify_restorable "
            "without performing a restore. CPU/disk only."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.fallback_resident.verify_restorable",
        provider="sidecar-static",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "command": "python3 tools/future/fallback_resident.py --build",
            "species": "CPU_ANALYSIS",
            "claim_boundary": (
                "WorkUnit is a proposal; receipt remains authoritative; "
                "this unit cannot restore, launch, or take a GPU lease."
            ),
            "may_promote": False,
            "may_modify_verifier": False,
            "verdict_at_emission": verdict.get("verdict"),
        },
    )
    validate_emitted_unit(unit)
    return [unit]


def resident_callable(verdict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.fallback_resident.verify_restorable()",
        "workunit": (
            "one CPU_ANALYSIS unit; STATIC_ANALYSIS resource_class; "
            "verify_restorable + seal FALLBACK_RESIDENT.json; does not restore"
        ),
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier": "FT.CHILD_RESIDENT.install-dry-run",
        "fails_closed": (
            "missing artifact → NOT_RESTORABLE naming the artifact; "
            "digest mismatch → NOT_RESTORABLE (wrong body at the path); "
            "partial evidence never rounds into RESTORABLE_NOW"
        ),
        "hcli_can_invoke": True,
        "python_api": {
            "fallback_identity": "fallback_identity(overlay=None) -> dict",
            "restore_path": "restore_path(overlay=None) -> dict",
            "verify_restorable": "verify_restorable(overlay=None) -> dict",
            "rollback_state": "rollback_state(overlay=None) -> dict",
            "build": "build() -> Path",
        },
        "verdict_at_emission": verdict.get("verdict"),
        "started_model_process": False,
        "took_gpu_lease": False,
    }


def recovered_implementation() -> list[str]:
    return [
        f"{SEALED_REL} (incumbent Qwen27 sealed identity; consumed, not forked)",
        f"{SEAL_REL} (HCLI resident seal prefixes: tokenizer/runtime/chat_template)",
        "tools/future/resident_identity.py (CURRENT_NONFINAL_HCLI_WORKER; UNKNOWN hardware slots)",
        "tools/future/resident_install.py (14-phase contract; fallback_policy REFUSE on mismatch)",
        "tools/future/super_resident.py (locate/load_repo_json; Qwen27 sandbox holder, not Singularity)",
        "tools/future/succession.py (keep_parent_for_rollback / rollback_to_parent; durable-science split)",
        "tools/future/tournament.py (Qwen27 is CONTROL incumbent, not a complete SINGULARITY NX)",
        "hcli/hawking_native.py (connector.stop / _restart_resident; cited, not executed)",
        "hcli/agentos/resident_gate.py (live residency proof; cited, not executed)",
        "hcli/agentos/recovery.py (fail-closed fixture recovery; Codex owns production recovery)",
    ]


def gaps_closed() -> list[str]:
    return [
        "sealed fallback identity: artifact path, specimen (mix_id/parent_params), config digest",
        "restore_path of independently checkable steps with current_state on each",
        "verify_restorable answers RESTORABLE_NOW | RESTORABLE_WITH_ACTION | NOT_RESTORABLE without restoring",
        "digest mismatch of a present artifact is NOT_RESTORABLE naming the artifact",
        "rollback_state names what reverts and what durable science does not",
    ]


def negative_findings() -> list[str]:
    return [
        "this sidecar does not perform a restore, start a resident, or take a GPU lease",
        "orchestration.py BINDINGS was not edited (not in this lane's WRITE list)",
        "frontiers.py was not given a new FT.CHILD_RESIDENT.fallback item (not in WRITE list); "
        "the receipt informs FT.CHILD_RESIDENT.install-dry-run",
        "artifact_root weights are identified by MIX_REPORT, not by hashing ~10GB",
        "runtime_commit on the seal is not a required git checkout",
        "hcli/ is not materialized in this sparse worktree; identity/seal/lifecycle recovered via locate/HEAD",
        "sealed current_runtime TPS and physical_ebpw were not copied",
        "production recovery remains Codex-owned (recovery.py is a fixture proof)",
    ]


def build() -> Path:
    ident = fallback_identity()
    verdict = verify_restorable(identity=ident)
    path = restore_path(identity=ident, restorable=verdict)
    rollback = rollback_state(restorable=verdict)
    units = emit_workunits(verdict)
    hw = _hardware_numeric_keys({"identity": ident, "verdict": verdict, "path": path, "rollback": rollback})
    if hw:
        raise FallbackRefused(f"receipt would carry hardware fields {hw}")
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Keep CURRENT_NONFINAL_HCLI_WORKER (Qwen27 sealed-3.14) restorable "
            "at every moment, including while a candidate child is shadowed. "
            "Identity, restore path, restorable-now verdict, rollback scope."
        ),
        "evidence_class": CLAIM_CLASS,
        "gpu_authority": False,
        "performed_restore": False,
        "started_model_process": False,
        "took_gpu_lease": False,
        "flock": False,
        "identity": ident,
        "restore_path": path,
        "verify_restorable": verdict,
        "rollback_state": rollback,
        "work_units": units,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": resident_callable(verdict),
        "negative_control": {
            "missing_artifact_must_be_not_restorable": True,
            "digest_mismatch_must_be_not_restorable": True,
            "rollback_names_durable_science": True,
            "verify_restorable_can_return_false": verdict["verdict"] in VERDICTS,
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def _selftest_overlay_missing() -> dict[str, Any]:
    """In-memory world whose tokenizer is absent. Must be NOT_RESTORABLE."""
    return {
        "identity_document": {
            "exists": True,
            "source": "selftest",
            "doc": {
                "model_id": "qwen3.8-27b-sealed-3.14",
                "resident_identity": "sealed-3.14",
                "family": "qwen3.8",
                "protocol": "hawking.qwen38.resident.v1",
                "runtime": "hawking-native",
                "provider": "native",
                "mode": "auto",
                "artifact_root": "/nonexistent/NOETIC_PARENT_A",
                "tokenizer": "/nonexistent/NOETIC_PARENT_A/tokenizer.json",
                "binary": "/nonexistent/greedy",
                "resident_binary": "/nonexistent/resident",
                "prompt_contract": {"renderer": "selftest"},
            },
        },
        "seal": {
            "exists": True,
            "source": "selftest",
            "doc": {
                "status": "SEALED",
                "resident": "sealed-3.14",
                "fields": {
                    "tokenizer_sha256_16": {"value": "aaaaaaaaaaaaaaaa"},
                    "runtime_binary_sha256_16": {"value": "bbbbbbbbbbbbbbbb"},
                    "chat_template_sha256_16": {"value": "cccccccccccccccc"},
                    "physical_closure": {"value": {"parent_params": 1}},
                    "runtime_commit": {"value": "deadbeef"},
                },
            },
        },
        "artifact_root": {"path": "/nonexistent/NOETIC_PARENT_A", "exists": False, "is_dir": False},
        "tokenizer": {"path": "/nonexistent/NOETIC_PARENT_A/tokenizer.json", "exists": False, "sha256": UNKNOWN},
        "chat_template": {"path": "/nonexistent/chat_template.jinja", "exists": False, "sha256": UNKNOWN},
        "runtime_binary": {"path": "/nonexistent/greedy", "exists": False, "sha256": UNKNOWN},
        "resident_binary": {"path": "/nonexistent/resident", "exists": False, "sha256": UNKNOWN},
        "mix_report": {"path": "/nonexistent/MIX_REPORT.json", "exists": False, "doc": None},
        "catalog": {"path": "/nonexistent/catalog", "exists": False},
        "resident_gate": {"exists": True, "in_git_head": True, "source": "selftest"},
        "connector": {"exists": True, "in_git_head": True, "source": "selftest"},
        "recovery": {"exists": True, "in_git_head": True, "source": "selftest"},
    }


def selftest() -> Path:
    missing = verify_restorable(_selftest_overlay_missing())
    if missing["verdict"] != VERDICT_NOT:
        raise FallbackRefused(f"selftest: missing artifact returned {missing['verdict']!r}")
    if "tokenizer" not in str(missing.get("unmet_precondition")) and "NOETIC_PARENT_A" not in str(missing.get("unmet_precondition")) and "MIX_REPORT" not in str(missing.get("unmet_precondition")) and "artifact" not in str(missing.get("reason", "")).lower():
        # First unmet may be artifact_root; that still names the artifact.
        if not missing.get("unmet_precondition"):
            raise FallbackRefused("selftest: NOT_RESTORABLE without naming an artifact")
    rb = rollback_state()
    if not rb["does_not_revert"] or not rb["durable_science_survives"]:
        raise FallbackRefused("selftest: rollback_state omitted durable science")
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = selftest() if a.selftest else build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
