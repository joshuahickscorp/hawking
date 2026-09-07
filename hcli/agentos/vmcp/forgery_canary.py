#!/usr/bin/env python3
"""Forgery canary for the receipts AgentOS actually trusts.

Attacks `CaptureBus.observe` / `CaptureBus.verify` with `ProjectStore` /
`ArtifactStore` — the same seam `hcli_integration.py` proved, in a
temporary project.

Adversaries (each must be attempted; UNDETECTED is a finding, not a skip):

  1. tampered_artifact — capture, then modify CAS bytes in place
  2. tampered_record   — leave CAS alone, edit summary_json (the digest field)
  3. truncated_receipt — cut the stored artifact, and the stored record, in half
  4. self_report       — adapter reports success and writes no artifact
  5. replay            — reuse a valid capture id for a DIFFERENT file
  6. stale_capture_after_subject_mutation — observe, overwrite the live file,
     then verify the original capture id and re-observe the same path

Positive control: an untampered capture must verify clean, or a canary that
always says DETECTED would pass.

    python3 hcli/agentos/vmcp/forgery_canary.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO / "receipts" / "headless" / "VMCP_FORGERY_CANARY.json"
RIGHTS = "local-file-owned-by-operator"
ABSENT_ID = "0" * 64


# --------------------------------------------------------------------------- locate


def locate_visionmcp_src() -> Path:
    """This worktree is sparse; visionmcp is not materialized here."""
    env = os.environ.get("VISIONMCP_SRC")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            REPO / "visionmcp" / "src",
            # Dedicated visionmcp checkout (has its own git). The hawking-copy
            # tree is an untracked copy whose `git rev-parse` walks up to the
            # hawking repo and would mis-attribute HEAD.
            Path("/Users/scammermike/Downloads/hawking/visionmcp/src"),
            Path.home() / ".searcher-donors" / "visionmcp" / "src",
            Path("/Users/scammermike/Downloads/hawking-copy/visionmcp/src"),
        ]
    )
    seen: set[Path] = set()
    for src in candidates:
        if not src.exists():
            continue
        src = src.resolve()
        if src in seen:
            continue
        seen.add(src)
        if (src / "visionmcp" / "perception" / "bus.py").is_file():
            return src
    raise FileNotFoundError(
        "visionmcp src not found. Set VISIONMCP_SRC to the package's src/ "
        "directory (the parent of the visionmcp package). This sparse worktree "
        "does not materialize visionmcp/, and git sparse-checkout add is denied."
    )


def git_info(cwd: Path) -> dict[str, Any]:
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        top = Path(toplevel).resolve()
        here = cwd.resolve()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        oneline = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        own = top == here
        pyproject = here / "pyproject.toml"
        if not own and pyproject.is_file() and "visionmcp" in pyproject.read_text(
            encoding="utf-8"
        )[:800]:
            # Copied tree living inside another git repo — do not claim that
            # repo's HEAD as visionmcp's.
            return {
                "head": None,
                "oneline": None,
                "cwd": str(cwd),
                "toplevel": str(top),
                "own_git": False,
                "note": f"not a git root; parent HEAD would be {head[:12]} ({oneline})",
            }
        return {
            "head": head,
            "oneline": oneline,
            "cwd": str(cwd),
            "toplevel": str(top),
            "own_git": own,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "oneline": None, "cwd": str(cwd), "error": str(exc)}


def jsonable(value: Any, *, limit: int = 4000) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v, limit=limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, limit=limit) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {"_bytes": len(value), "_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... <truncated {len(value) - limit} chars>"
        return value
    return repr(value)


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), indent=2, sort_keys=True, default=str)


# --------------------------------------------------------------------------- adapters (same FileEye the AgentOS integration registered)


class FileEye:
    """Deterministic sensor over a file. Same adapter as VMCP_AGENTOS_INTEGRATION.

    Identity is the path, not the content. Content lives in the artifact digest.
    """

    name = "file.eye"
    version = "1"

    def normalize_target(self, target: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(target["path"])).resolve()
        return {"id": f"file:{path}", "path": str(path)}

    def normalize_config(
        self, target: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "hash": str(config.get("hash", "sha256")),
            "max_bytes": int(config.get("max_bytes", 8_000_000)),
        }

    def environment(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "adapter_version": self.version,
            "hash": config["hash"],
        }

    def capture(self, target: dict[str, Any], config: dict[str, Any], sink: Any) -> Any:
        from visionmcp.perception.bus import CaptureOutcome

        path = Path(target["path"])
        limitations: list[str] = []
        if not path.is_file():
            return CaptureOutcome(
                summary={"present": False, "path": str(path)},
                limitations=["TARGET_ABSENT"],
            )
        data = path.read_bytes()
        if len(data) > config["max_bytes"]:
            data = data[: config["max_bytes"]]
            limitations.append("TRUNCATED_TO_MAX_BYTES")
        digest = hashlib.sha256(data).hexdigest()
        stat = path.stat()
        sink("bytes", data, "application/octet-stream", {"sha256": digest})
        return CaptureOutcome(
            summary={
                "present": True,
                "path": str(path),
                "size": stat.st_size,
                "mode": oct(stat.st_mode),
                "sha256": digest,
            },
            limitations=limitations,
        )


class EmptySuccessAdapter:
    """Reports success and writes no artifact. The self-report adversary."""

    name = "self.report"
    version = "1"

    def normalize_target(self, target: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(target["path"])).resolve()
        return {"id": f"self:{path}", "path": str(path)}

    def normalize_config(
        self, target: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return {"hash": "sha256"}

    def environment(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"adapter": self.name, "adapter_version": self.version}

    def capture(self, target: dict[str, Any], config: dict[str, Any], sink: Any) -> Any:
        from visionmcp.perception.bus import CaptureOutcome

        del sink, config
        return CaptureOutcome(
            summary={
                "ok": True,
                "verified": True,
                "present": True,
                "path": str(Path(target["path"]).resolve()),
                "sha256": "e" * 64,
            },
            limitations=[],
        )


# --------------------------------------------------------------------------- bus helpers


def make_bus(project_root: Path, *, with_self_report: bool = False):
    from visionmcp.perception.bus import AdapterRegistry, CaptureBus
    from visionmcp.projects.store import ProjectStore

    if (project_root / "project.db").is_file():
        project = ProjectStore.open(project_root)
    else:
        project_root.mkdir(parents=True, exist_ok=True)
        project = ProjectStore.create(project_root, name="vmcp-forgery-canary")
    registry = AdapterRegistry()
    registry.register(FileEye())
    if with_self_report:
        registry.register(EmptySuccessAdapter())
    return CaptureBus(project, registry)


def observe_file(bus, path: Path) -> dict[str, Any]:
    return bus.observe("file.eye", {"path": str(path)}, {}, rights_decision=RIGHTS)


def call_verify(bus, capture_id: str) -> dict[str, Any]:
    """Return the exact verify response, or the exception it raised."""
    try:
        payload = bus.verify(capture_id)
        return {"raised": False, "response": jsonable(payload)}
    except Exception as exc:
        return {
            "raised": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def bytes_role(record: dict[str, Any]) -> dict[str, Any]:
    for item in record.get("artifacts") or []:
        if item.get("role") == "bytes":
            return item
    raise RuntimeError(f"capture has no 'bytes' artifact: {record.get('artifacts')}")


def consume_as_agentos(bus, capture_id: str, claimed_path: Path) -> dict[str, Any]:
    """The AgentOS verifier from VMCP_AGENTOS_INTEGRATION.

    get() + verify() + summary.sha256 vs sha256(claimed_path). It does not
    check that the capture's recorded target is the claimed file. That is the
    consumption path this replay adversary is attacking.
    """
    try:
        record = bus.get(capture_id)
    except Exception as exc:
        return {
            "result": "GET_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if not record or record.get("status") != "COMPLETE":
        return {
            "result": "NO_EVIDENCE",
            "status": None if record is None else record.get("status"),
        }
    verified = call_verify(bus, capture_id)
    if verified["raised"]:
        return {
            "result": "VERIFY_ERROR",
            "verify": verified,
        }
    payload = verified["response"]
    if not payload.get("valid"):
        return {"result": "EVIDENCE_INVALID", "verify": payload}
    summary = record.get("summary") or {}
    if isinstance(summary, str):
        summary = json.loads(summary)
    observed = summary.get("sha256")
    actual = hashlib.sha256(claimed_path.read_bytes()).hexdigest()
    recorded_path = ((record.get("request") or {}).get("target") or {}).get("path")
    claimed = str(claimed_path.resolve())
    recorded = str(Path(recorded_path).resolve()) if recorded_path else None
    out = {
        "observed_sha256": observed,
        "claimed_sha256": actual,
        "recorded_target_path": recorded,
        "claimed_path": claimed,
        "subject_matches_recorded_target": recorded == claimed,
        "verify_valid": True,
        "verify": payload,
    }
    if observed != actual:
        out["result"] = "EVIDENCE_STALE"
        return out
    out["result"] = "SUPPORTED"
    return out


def verdict_from_verify(call: dict[str, Any]) -> str:
    """UNDETECTED iff verify returned valid=True. Anything else is DETECTED."""
    if call.get("raised"):
        return "DETECTED"
    response = call.get("response") or {}
    if response.get("valid") is True:
        return "UNDETECTED"
    return "DETECTED"


def distinguishable_from_absent(call: dict[str, Any], absent: dict[str, Any]) -> bool:
    return json.dumps(call, sort_keys=True, default=str) != json.dumps(
        absent, sort_keys=True, default=str
    )


# --------------------------------------------------------------------------- adversaries


def run_positive_control(bus, subject: Path) -> dict[str, Any]:
    record = observe_file(bus, subject)
    capture_id = record["capture_id"]
    verified = call_verify(bus, capture_id)
    ok = (
        record.get("status") == "COMPLETE"
        and not verified["raised"]
        and verified["response"].get("valid") is True
    )
    return {
        "id": "positive_control",
        "ok": ok,
        "capture_id": capture_id,
        "observe": {
            "status": record.get("status"),
            "authority": record.get("authority"),
            "reused": record.get("reused"),
            "summary": record.get("summary"),
            "artifact_roles": [a.get("role") for a in record.get("artifacts") or []],
        },
        "verify": verified,
        "verdict": "PASS" if ok else "FAIL",
    }


def run_tampered_artifact(bus, subject: Path) -> dict[str, Any]:
    attack = (
        "capture a file, then overwrite the stored CAS artifact bytes in place "
        "(same path, different content, digest in the record left unchanged)"
    )
    record = observe_file(bus, subject)
    capture_id = record["capture_id"]
    before = call_verify(bus, capture_id)
    if before.get("raised") or not (before.get("response") or {}).get("valid"):
        return {
            "id": "tampered_artifact",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "untampered capture did not verify, so a subsequent invalid "
            "result would not be attributable to the tamper",
            "before": before,
            "observe": {"capture_id": capture_id, "status": record.get("status")},
        }
    artifact = bytes_role(record)
    path = bus.artifacts.path_for(artifact["digest"])
    original = path.read_bytes()
    tampered = original + b"\nTAMPERED-CAS-BYTES\n"
    path.write_bytes(tampered)
    written = path.read_bytes()
    stuck = (
        written == tampered
        and hashlib.sha256(written).hexdigest() != artifact["digest"]
    )
    if not stuck:
        return {
            "id": "tampered_artifact",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "CAS overwrite did not stick",
            "path": str(path),
        }
    after = call_verify(bus, capture_id)
    return {
        "id": "tampered_artifact",
        "attack": attack,
        "capture_id": capture_id,
        "cas_path": str(path),
        "original_size": len(original),
        "tampered_size": len(tampered),
        "recorded_digest": artifact["digest"],
        "actual_digest_after": hashlib.sha256(tampered).hexdigest(),
        "before": before,
        "response": after,
        "verdict": verdict_from_verify(after),
    }


def run_tampered_record(bus, subject: Path) -> dict[str, Any]:
    attack = (
        "leave the artifact bytes alone; UPDATE observation_captures.summary_json "
        "so the recorded digest (summary.sha256) is a lie"
    )
    from visionmcp.core.util import canonical_json

    record = observe_file(bus, subject)
    capture_id = record["capture_id"]
    before = call_verify(bus, capture_id)
    if before.get("raised") or not (before.get("response") or {}).get("valid"):
        return {
            "id": "tampered_record",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "untampered capture did not verify",
            "before": before,
        }
    original_summary = dict(record["summary"])
    lying_summary = dict(original_summary)
    lying_summary["sha256"] = "f" * 64
    with bus.project.connection() as connection:
        connection.execute(
            "UPDATE observation_captures SET summary_json=? WHERE id=?",
            (canonical_json(lying_summary).decode(), capture_id),
        )
    reloaded = bus.get(capture_id)
    if (reloaded or {}).get("summary", {}).get("sha256") != "f" * 64:
        return {
            "id": "tampered_record",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "summary_json UPDATE did not stick",
            "reloaded_summary": None if reloaded is None else reloaded.get("summary"),
        }
    after = call_verify(bus, capture_id)
    return {
        "id": "tampered_record",
        "attack": attack,
        "capture_id": capture_id,
        "original_summary": original_summary,
        "lying_summary": lying_summary,
        "get_after_tamper": {
            "summary": reloaded.get("summary"),
            "status": reloaded.get("status"),
            "manifest_digest": reloaded.get("manifest_digest"),
        },
        "before": before,
        "response": {
            "get_after_tamper": {
                "summary": reloaded.get("summary"),
                "status": reloaded.get("status"),
                "manifest_digest": reloaded.get("manifest_digest"),
            },
            "verify": after,
        },
        "verdict": verdict_from_verify(after),
    }


def run_truncated_receipt(bus, artifact_subject: Path, record_subject: Path) -> dict[str, Any]:
    attack = (
        "cut the stored CAS artifact in half; separately cut the stored "
        "summary_json record in half. Compare each response to unknown-capture "
        "and to a deleted artifact, so 'absent' is not confused with 'truncated'."
    )
    absent = call_verify(bus, ABSENT_ID)

    art_record = observe_file(bus, artifact_subject)
    art_id = art_record["capture_id"]
    artifact = bytes_role(art_record)
    art_path = bus.artifacts.path_for(artifact["digest"])
    original = art_path.read_bytes()
    if len(original) < 16:
        return {
            "id": "truncated_receipt",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": f"artifact too small to cut in half ({len(original)} bytes)",
        }
    halved = original[: len(original) // 2]
    art_path.write_bytes(halved)
    if art_path.read_bytes() != halved:
        return {
            "id": "truncated_receipt",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "artifact truncation did not stick",
        }
    artifact_call = call_verify(bus, art_id)

    rec_record = observe_file(bus, record_subject)
    rec_id = rec_record["capture_id"]
    with bus.project.connection() as connection:
        row = connection.execute(
            "SELECT summary_json FROM observation_captures WHERE id=?",
            (rec_id,),
        ).fetchone()
        original_json = row["summary_json"]
        half_json = original_json[: len(original_json) // 2]
        connection.execute(
            "UPDATE observation_captures SET summary_json=? WHERE id=?",
            (half_json, rec_id),
        )
    record_call = call_verify(bus, rec_id)

    missing_subject = artifact_subject.parent / "truncated-missing.txt"
    missing_subject.write_text("missing-artifact-control\n" + ("M" * 64) + "\n")
    miss_record = observe_file(bus, missing_subject)
    miss_id = miss_record["capture_id"]
    miss_art = bytes_role(miss_record)
    miss_path = bus.artifacts.path_for(miss_art["digest"])
    miss_path.unlink()
    missing_call = call_verify(bus, miss_id)

    variants = {
        "artifact_halved": {
            "capture_id": art_id,
            "original_size": len(original),
            "truncated_size": len(halved),
            "response": artifact_call,
            "verdict": verdict_from_verify(artifact_call),
            "distinguishable_from_absent": distinguishable_from_absent(
                artifact_call, absent
            ),
            "distinguishable_from_missing_file": distinguishable_from_absent(
                artifact_call, missing_call
            ),
        },
        "record_halved": {
            "capture_id": rec_id,
            "original_summary_json_len": len(original_json),
            "truncated_summary_json_len": len(half_json),
            "truncated_summary_json": half_json,
            "response": record_call,
            "verdict": verdict_from_verify(record_call),
            "distinguishable_from_absent": distinguishable_from_absent(
                record_call, absent
            ),
        },
        "artifact_deleted_for_comparison": {
            "capture_id": miss_id,
            "response": missing_call,
            "verdict": verdict_from_verify(missing_call),
            "distinguishable_from_absent": distinguishable_from_absent(
                missing_call, absent
            ),
        },
    }
    # Headline: UNDETECTED if either cut still verifies valid. Do not call a
    # mixed result "partially detected".
    undetected = [
        name
        for name, body in variants.items()
        if name != "artifact_deleted_for_comparison" and body["verdict"] == "UNDETECTED"
    ]
    return {
        "id": "truncated_receipt",
        "attack": attack,
        "absent_capture_response": absent,
        "variants": variants,
        "response": {
            "artifact_halved": artifact_call,
            "record_halved": record_call,
            "absent_is_keyerror": absent.get("error_type") == "KeyError",
            "artifact_halved_distinguishable_from_absent": distinguishable_from_absent(
                artifact_call, absent
            ),
            "record_halved_distinguishable_from_absent": distinguishable_from_absent(
                record_call, absent
            ),
            "artifact_halved_distinguishable_from_deleted_file": distinguishable_from_absent(
                artifact_call, missing_call
            ),
        },
        "verdict": "UNDETECTED" if undetected else "DETECTED",
        "undetected_variants": undetected,
    }


def run_self_report(bus, subject: Path) -> dict[str, Any]:
    attack = (
        "adapter returns CaptureOutcome(summary={ok: True, verified: True}) "
        "without calling the artifact sink — success with no evidence body"
    )
    try:
        record = bus.observe(
            "self.report",
            {"path": str(subject)},
            {},
            rights_decision=RIGHTS,
        )
        observe_call: dict[str, Any] = {
            "raised": False,
            "response": jsonable(
                {
                    "capture_id": record.get("capture_id"),
                    "status": record.get("status"),
                    "summary": record.get("summary"),
                    "artifacts": record.get("artifacts"),
                    "reused": record.get("reused"),
                }
            ),
        }
        capture_id = record.get("capture_id")
    except Exception as exc:
        observe_call = {
            "raised": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        capture_id = None
        # The bus still writes a CAPTURING/INTERRUPTED row before the adapter
        # runs. Recover it so verify can be asked about the leftover.
        with bus.project.connection() as connection:
            row = connection.execute(
                "SELECT id,status FROM observation_captures WHERE adapter=? "
                "ORDER BY created_at DESC LIMIT 1",
                ("self.report",),
            ).fetchone()
        if row is not None:
            capture_id = row["id"]

    verify_call = call_verify(bus, capture_id) if capture_id else None
    leftover = bus.get(capture_id) if capture_id else None
    leftover_view = None
    if leftover is not None:
        leftover_view = {
            "capture_id": leftover.get("capture_id"),
            "status": leftover.get("status"),
            "summary": leftover.get("summary"),
            "artifacts": leftover.get("artifacts"),
            "manifest_digest": leftover.get("manifest_digest"),
        }

    # DETECTED if observe refused to complete, or verify did not return valid.
    if observe_call.get("raised"):
        verdict = "DETECTED"
    elif leftover_view and leftover_view.get("status") == "COMPLETE" and not leftover_view.get(
        "artifacts"
    ):
        # Completed with an empty body: the bus accepted the self-report.
        verdict = verdict_from_verify(verify_call or {"raised": False, "response": {}})
        if verify_call and not verify_call.get("raised") and (verify_call.get("response") or {}).get(
            "valid"
        ):
            verdict = "UNDETECTED"
    elif verify_call is not None:
        verdict = verdict_from_verify(verify_call)
        # A COMPLETE capture with artifacts would mean the adapter did write.
        if leftover_view and leftover_view.get("artifacts"):
            verdict = "NOT_ATTEMPTED"
            reason = "adapter unexpectedly wrote artifacts; self-report was not constructed"
        else:
            reason = None
    else:
        verdict = "NOT_ATTEMPTED"
        reason = "could not recover a capture id after observe"

    result: dict[str, Any] = {
        "id": "self_report",
        "attack": attack,
        "observe": observe_call,
        "capture_id": capture_id,
        "leftover": leftover_view,
        "response": {
            "observe": observe_call,
            "leftover": leftover_view,
            "verify": verify_call,
        },
        "verdict": verdict,
    }
    if verdict == "NOT_ATTEMPTED":
        result["reason"] = reason
    return result


def run_replay(bus, file_a: Path, file_b: Path) -> dict[str, Any]:
    attack = (
        "observe file A; present that valid capture_id as evidence for file B "
        "(different path, identical bytes). This is how the AgentOS verifier "
        "consumes evidence: CaptureBus.verify(capture_id) plus summary.sha256 "
        "vs sha256(claimed_path). It does not bind the capture to the subject."
    )
    if file_a.resolve() == file_b.resolve():
        return {
            "id": "replay",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "A and B resolved to the same path",
        }
    if file_a.read_bytes() != file_b.read_bytes():
        return {
            "id": "replay",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "A and B must share bytes so a content-only check cannot "
            "be mistaken for subject-binding",
        }
    record_a = observe_file(bus, file_a)
    capture_a = record_a["capture_id"]
    verify_a = call_verify(bus, capture_a)
    if verify_a.get("raised") or not (verify_a.get("response") or {}).get("valid"):
        return {
            "id": "replay",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "capture of A did not verify",
            "verify_a": verify_a,
        }

    consume_b = consume_as_agentos(bus, capture_a, file_b)
    consume_a = consume_as_agentos(bus, capture_a, file_a)
    # Observe B on its own — the bus must mint a different capture_id because
    # the path is in the identity. That is not the replay; it is the control
    # that identity includes the path at observe time, then drops it at verify.
    record_b = observe_file(bus, file_b)
    capture_b = record_b["capture_id"]

    verify_sig = str(inspect.signature(bus.verify))

    if consume_b.get("result") == "SUPPORTED" and not consume_b.get(
        "subject_matches_recorded_target"
    ):
        verdict = "UNDETECTED"
    elif consume_b.get("result") == "SUPPORTED":
        verdict = "NOT_ATTEMPTED"
        reason = "consume returned SUPPORTED but subject matched recorded target"
    else:
        verdict = "DETECTED"
        reason = None

    result = {
        "id": "replay",
        "attack": attack,
        "capture_id_a": capture_a,
        "capture_id_b": capture_b,
        "same_capture_id_for_different_paths": capture_a == capture_b,
        "file_a": str(file_a.resolve()),
        "file_b": str(file_b.resolve()),
        "shared_sha256": hashlib.sha256(file_a.read_bytes()).hexdigest(),
        "verify_signature": verify_sig,
        "verify_a": verify_a,
        "consume_a_as_a": consume_a,
        "response": consume_b,
        "verdict": verdict,
    }
    if verdict == "NOT_ATTEMPTED":
        result["reason"] = reason
    return result


def call_verify_live(bus, capture_id: str) -> dict[str, Any]:
    """verify(live_subject=True) if the bus offers it; else KeyError-shaped miss."""
    verify = getattr(bus, "verify", None)
    if not callable(verify):
        return {
            "raised": True,
            "error_type": "AttributeError",
            "error": "bus has no verify",
        }
    try:
        payload = verify(capture_id, live_subject=True)
        return {"raised": False, "response": jsonable(payload)}
    except TypeError:
        return {
            "raised": True,
            "error_type": "TypeError",
            "error": "verify does not accept live_subject",
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        return {
            "raised": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def call_subject_matches(bus, capture_id: str) -> dict[str, Any]:
    matcher = getattr(bus, "subject_matches", None)
    if not callable(matcher):
        return {
            "raised": True,
            "error_type": "AttributeError",
            "error": "bus has no subject_matches",
        }
    try:
        return {"raised": False, "response": jsonable(matcher(capture_id))}
    except Exception as exc:
        return {
            "raised": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_stale_capture_after_subject_mutation(bus, subject: Path) -> dict[str, Any]:
    attack = (
        "observe a file, overwrite it on disk, then verify the original "
        "capture id and re-observe the same path. Distinct from replay "
        "(same bytes, different path) and tampered_artifact (CAS bytes)."
    )
    record = observe_file(bus, subject)
    capture_id = record["capture_id"]
    digest_before = hashlib.sha256(subject.read_bytes()).hexdigest()
    stored_before = call_verify(bus, capture_id)
    if stored_before.get("raised") or not (stored_before.get("response") or {}).get("valid"):
        return {
            "id": "stale_capture_after_subject_mutation",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "untampered capture did not verify",
            "before": stored_before,
        }
    subject.write_bytes(subject.read_bytes() + b"\nMUTATED-LIVE-SUBJECT\n")
    digest_after = hashlib.sha256(subject.read_bytes()).hexdigest()
    if digest_after == digest_before:
        return {
            "id": "stale_capture_after_subject_mutation",
            "attack": attack,
            "verdict": "NOT_ATTEMPTED",
            "reason": "subject overwrite did not change the digest",
        }
    stored_after = call_verify(bus, capture_id)
    live_after = call_verify_live(bus, capture_id)
    freshness = call_subject_matches(bus, capture_id)
    reobserve = observe_file(bus, subject)
    reused = bool(reobserve.get("reused"))
    same_id = reobserve.get("capture_id") == capture_id
    live_detected = verdict_from_verify(live_after) == "DETECTED"
    freshness_response = freshness.get("response") or {}
    freshness_stale = (
        not freshness.get("raised")
        and freshness_response.get("matches") is False
        and freshness_response.get("reason") == "subject_changed"
    )
    reuse_bounded = reused is not True
    if (live_detected or freshness_stale) and reuse_bounded:
        verdict = "DETECTED"
        reason = None
    else:
        verdict = "UNDETECTED"
        reason = (
            "stored verify stayed valid for a mutated subject and "
            f"re-observe reused={reused} same_id={same_id}; "
            f"live_subject detected={live_detected} "
            f"subject_matches stale={freshness_stale}"
        )
    return {
        "id": "stale_capture_after_subject_mutation",
        "attack": attack,
        "capture_id": capture_id,
        "digest_before": digest_before,
        "digest_after": digest_after,
        "verify_stored_after_mutation": stored_after,
        "verify_live_subject_after_mutation": live_after,
        "subject_matches_after_mutation": freshness,
        "reobserve": {
            "reused": reused,
            "same_id": same_id,
            "capture_id": reobserve.get("capture_id"),
            "status": reobserve.get("status"),
            "summary": reobserve.get("summary"),
        },
        "response": {
            "digest_before": digest_before,
            "digest_after": digest_after,
            "verify_stored": stored_after,
            "verify_live_subject": live_after,
            "subject_matches": freshness,
            "reobserve_reused": reused,
            "reobserve_same_id": same_id,
        },
        "verdict": verdict,
        **({"reason": reason} if reason else {}),
    }


# --------------------------------------------------------------------------- main


def print_block(title: str, body: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"verdict: {body.get('verdict')}")
    if body.get("attack"):
        print(f"attack:  {body['attack']}")
    if body.get("reason"):
        print(f"reason:  {body['reason']}")
    print("response:")
    print(dumps(body.get("response")))


def main() -> int:
    started = time.time()
    vmcp_src = locate_visionmcp_src()
    if str(vmcp_src) not in sys.path:
        sys.path.insert(0, str(vmcp_src))

    import visionmcp
    from visionmcp.perception.bus import CaptureBus

    repo_git = git_info(REPO)
    vmcp_git = git_info(vmcp_src.parent if vmcp_src.name == "src" else vmcp_src)

    print("VMCP forgery canary")
    print(f"repo HEAD:     {repo_git.get('oneline') or repo_git}")
    print(f"visionmcp src: {vmcp_src}")
    print(f"visionmcp git: {vmcp_git.get('oneline') or vmcp_git}")
    print(f"visionmcp ver: {getattr(visionmcp, '__version__', None)}")
    print(f"CaptureBus.verify{inspect.signature(CaptureBus.verify)}")

    with tempfile.TemporaryDirectory(prefix="vmcp-forgery-canary-") as tmp:
        root = Path(tmp)
        project_root = root / "project"
        bus = make_bus(project_root, with_self_report=True)

        def write(name: str, payload: str) -> Path:
            path = root / name
            path.write_text(payload, encoding="utf-8")
            return path

        clean = write("clean.txt", "positive-control\n" + ("C" * 64) + "\n")
        tamper_art = write("tamper-art.txt", "tampered-artifact\n" + ("A" * 64) + "\n")
        tamper_rec = write("tamper-rec.txt", "tampered-record\n" + ("R" * 64) + "\n")
        trunc_art = write("trunc-art.txt", "truncated-artifact\n" + ("T" * 128) + "\n")
        trunc_rec = write("trunc-rec.txt", "truncated-record\n" + ("Q" * 128) + "\n")
        self_tgt = write("self-report.txt", "self-report-target\n")
        replay_a = write("replay-a.txt", "replay-shared-bytes\n" + ("B" * 64) + "\n")
        replay_b = write("replay-b.txt", "replay-shared-bytes\n" + ("B" * 64) + "\n")
        stale_tgt = write("stale-subject.txt", "stale-capture-subject\n" + ("S" * 64) + "\n")

        positive = run_positive_control(bus, clean)
        print()
        print("=" * 72)
        print("POSITIVE CONTROL (untampered capture must verify clean)")
        print("=" * 72)
        print(f"verdict: {positive['verdict']}")
        print(dumps({"observe": positive["observe"], "verify": positive["verify"]}))
        if not positive["ok"]:
            print(
                "\nINCONCLUSIVE: positive control failed. Subsequent DETECTED "
                "verdicts would not be distinguishable from a canary that "
                "always says detected."
            )

        adversaries = [
            run_tampered_artifact(bus, tamper_art),
            run_tampered_record(bus, tamper_rec),
            run_truncated_receipt(bus, trunc_art, trunc_rec),
            run_self_report(bus, self_tgt),
            run_replay(bus, replay_a, replay_b),
            run_stale_capture_after_subject_mutation(bus, stale_tgt),
        ]
        for body in adversaries:
            print_block(body["id"], body)

        # Re-check the positive control so a CAS collision cannot hide.
        still = call_verify(bus, positive["capture_id"])
        still_ok = not still["raised"] and (still.get("response") or {}).get("valid") is True
        print()
        print("=" * 72)
        print("POSITIVE CONTROL re-verify after attacks")
        print("=" * 72)
        print(f"verdict: {'PASS' if still_ok else 'FAIL'}")
        print(dumps(still))

        attempted = [a["id"] for a in adversaries if a.get("verdict") != "NOT_ATTEMPTED"]
        skipped = [a for a in adversaries if a.get("verdict") == "NOT_ATTEMPTED"]
        undetected = [a for a in adversaries if a.get("verdict") == "UNDETECTED"]
        detected = [a for a in adversaries if a.get("verdict") == "DETECTED"]

        handoff: list[dict[str, Any]] = []
        for body in undetected:
            if body["id"] == "tampered_record":
                handoff.append(
                    {
                        "adversary": "tampered_record",
                        "file": "visionmcp/src/visionmcp/perception/bus.py",
                        "line": 439,
                        "smallest_change": (
                            "In CaptureBus.verify, after the manifest JSON loads "
                            "and the existing capture_id/request/artifacts comparisons "
                            "(bus.py:439-448), also require "
                            "manifest.get('summary') == capture['summary'] and "
                            "manifest.get('limitations') == capture['limitations']. "
                            "The envelope already stores both (bus.py:298-299); "
                            "verify simply never reads them. That is the hole "
                            "summary_json sits in."
                        ),
                    }
                )
            elif body["id"] == "replay":
                handoff.append(
                    {
                        "adversary": "replay",
                        "file": "visionmcp/src/visionmcp/perception/bus.py",
                        "line": 395,
                        "smallest_change": (
                            "CaptureBus.verify(self, capture_id) takes no subject. "
                            "Add an optional claimed target "
                            "(verify(self, capture_id, *, claimed_target=None)) and, "
                            "when it is provided, fail with subject_mismatch unless "
                            "it equals capture['request']['target']. The AgentOS "
                            "verifier in hcli/agentos/vmcp/hcli_integration.py "
                            "(_VERIFIER, ~lines 155-171) must then pass claimed_path "
                            "through; a content-only sha256 check is not subject-binding."
                        ),
                    }
                )
            elif body["id"] == "stale_capture_after_subject_mutation":
                handoff.append(
                    {
                        "adversary": "stale_capture_after_subject_mutation",
                        "file": "visionmcp/src/visionmcp/perception/bus.py",
                        "smallest_change": (
                            "verify() must not silently claim a mutated subject is "
                            "current. Either fail when live_subject=True / "
                            "subject_matches reports subject_changed, or document "
                            "that verify is stored-capture only and bound observe() "
                            "so reused=True cannot be returned after the bytes change."
                        ),
                    }
                )
            else:
                handoff.append(
                    {
                        "adversary": body["id"],
                        "smallest_change": (
                            "verify returned valid=True after this attack; bind the "
                            "forged field to the envelope digest before returning valid."
                        ),
                    }
                )

        sixth = next(
            (
                a
                for a in adversaries
                if a["id"] == "stale_capture_after_subject_mutation"
            ),
            None,
        )
        complete = (
            positive["ok"]
            and still_ok
            and len(adversaries) == 6
            and not skipped
            and sixth is not None
            and sixth.get("verdict") == "DETECTED"
        )
        receipt = {
            "gate": "VMCP_FORGERY_CANARY",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - started, 4),
            "git_head": repo_git.get("head"),
            "git": repo_git,
            "visionmcp": {
                "version": getattr(visionmcp, "__version__", None),
                "src": str(vmcp_src),
                "git": vmcp_git,
                "in_this_worktree": (REPO / "visionmcp" / "src").is_dir(),
            },
            "seam": {
                "acquire": "visionmcp.perception.bus.CaptureBus.observe",
                "verify": "CaptureBus.verify",
                "verify_signature": str(inspect.signature(CaptureBus.verify)),
                "subject_matches": "CaptureBus.subject_matches",
                "stores": ["ProjectStore", "ArtifactStore"],
                "adapter": "file.eye (same adapter as VMCP_AGENTOS_INTEGRATION)",
                "profile": "CaptureBus directly — not the laboratory MCP profile",
            },
            "verify_decision": {
                "verify": (
                    "stored snapshot only: CAS bytes, envelope, event receipts. "
                    "Does not re-read the live subject. Payload scope is "
                    "stored_capture."
                ),
                "live_subject": (
                    "verify(capture_id, live_subject=True) also fails when the "
                    "live subject no longer matches the captured digest."
                ),
                "subject_matches": (
                    "CaptureBus.subject_matches(capture_id) re-reads the file "
                    "at request.target.path and compares it to the recorded "
                    "digest. That is the API for 'the file changed since'."
                ),
                "reuse": (
                    "observe() returns reused=True only when stored verify is "
                    "valid AND subject_matches allows reuse. A changed file "
                    "is recaptured (same request id, new bytes)."
                ),
            },
            "positive_control": positive,
            "positive_control_after_attacks": {
                "ok": still_ok,
                "verify": still,
            },
            "adversaries": adversaries,
            "verdicts": {a["id"]: a.get("verdict") for a in adversaries},
            "attempted": attempted,
            "not_attempted": [
                {"id": a["id"], "reason": a.get("reason")} for a in skipped
            ],
            "detected": [a["id"] for a in detected],
            "undetected": [a["id"] for a in undetected],
            "handoff": handoff,
            "result": "COMPLETE" if complete else "INCONCLUSIVE",
        }
        RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT_PATH.write_text(
            json.dumps(jsonable(receipt), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"positive_control: {'PASS' if positive['ok'] and still_ok else 'FAIL'}")
        for body in adversaries:
            print(f"{body['id']}: {body.get('verdict')}")
        print(f"attempted: {len(attempted)}/6")
        print(f"undetected: {[a['id'] for a in undetected]}")
        print(f"result: {receipt['result']}")
        print(f"receipt: {RECEIPT_PATH}")
        return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
