#!/usr/bin/env python3.12
"""Detached, fail-closed post-download processing queue for the Studio.

Only the verified 14B parent is admitted today.  The supervisor promotes that
parent with an atomic symlink (no model bytes are copied), waits for the normal
Studio run to release its machine slot, and then runs the ``studio`` and
``subbit`` lanes serially.  A lane is complete only when both its quantization
coverage receipt and model-completion receipt validate.

This daemon never deletes a source model.  Source release remains a separate,
artifact-bound lifecycle operation after a durable deployable artifact exists.
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

import procure
from ram_scheduler import classify_resource_state, resource_snapshot, thermal_output_ok
from studio_manifest import DEFAULT_HARDWARE
from floor_integrity import (
    FLOOR_POINT_SCHEMA,
    create_floor_binding,
    locked_upsert_floor_row,
    validate_floor_binding,
    validate_receipt_floor_row,
)

LABEL = "14B"
HF_ID = "Qwen/Qwen2.5-14B-Instruct"
SOURCE = ROOT / "scratch/staging/qwen-14b.partial"
CANONICAL = ROOT / "scratch/qwen-14b"
DOWNLOAD_MARKER = ROOT / "reports/condense/download_state/14B.verified.json"
PROMOTION_RECEIPT = ROOT / "reports/condense/processing_state/14B.promotion.json"
STATE = ROOT / "reports/condense/processing_queue_state.json"
PID_FILE = ROOT / "reports/condense/processing_queue.pid.json"
LOCK_FILE = ROOT / "reports/condense/processing_queue.lock"
MODEL_LOCK_FILE = ROOT / "reports/condense/processing_14B.lock"
LOG_FILE = ROOT / "reports/condense/processing_queue.log"
STUDIO_RUN_PID = ROOT / "reports/cron/studio_run.pid"
HEAVY_LOCK = ROOT / "reports/cron/studio_heavy.lock"
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"
SHARED_DRAIN = ROOT / "reports/cron/studio_drain.request"
DOWNLOAD_STATE_DIR = ROOT / "reports/condense/download_state"
DOWNLOAD_QUEUE_STATE = ROOT / "reports/condense/download_queue_state.json"
SOURCE_72 = "scratch/staging/qwen-72b.partial"
POLL_S = float(os.environ.get("HAWKING_PROCESSING_QUEUE_POLL_S", "30"))
LANES = ("studio", "subbit")
PROCESSING_SCRATCH_GB = 90.0
AUDIT_RECIPE_VERSION = "hawking.audit.recipe.2026-07-12.v2"
MAX_OUTPUT_ATTEMPTS = max(
    1, int(os.environ.get("HAWKING_PROCESSING_MAX_OUTPUT_ATTEMPTS", "3"))
)
OUTPUT_RETRY_DELAY_S = max(
    1.0, float(os.environ.get("HAWKING_PROCESSING_OUTPUT_RETRY_DELAY_S", "30"))
)

# This barrier is deliberately pinned here instead of accepting whichever names a coverage
# receipt claims were required.  A stale/forged receipt therefore cannot shrink either lane.
REQUIRED_LANE_CONFIGS = {
    "studio": (
        "f16", "4-AWQ", "3-AWQ", "2-AWQ", "1-AWQ",
        "3-AWQ+dr", "2-AWQ+dr", "1-AWQ+dr",
    ),
    "subbit": (
        "f16",
        "vtq-k1-d2-b256-frozen", "vtq-k2-d4-b256-frozen",
        "vtq-k1-d4-b256-frozen", "vtq-k2-d8-b256-frozen",
        "vtq-k1-d8-b256-frozen",
        "vtq-k1-d2-b256-learned", "vtq-k1-d3-b256-learned",
        "vtq-k1-d4-b256-learned", "vtq-k1-d8-b256-learned",
        "vtq-k1-d2-b256-learned+dr-r8",
        "vtq-k1-d3-b256-learned+dr-r8",
        "vtq-k1-d4-b256-learned+dr-r8",
        "vtq-k1-d8-b256-learned+dr-r8",
        "vtq-k1-d2-b2048-frozen",
        "vtq-k1-d4-b4096-frozen",
        "vtq-k1-d8-b8192-frozen",
    ),
}

_stop_requested = False


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _try_heavy_lease(path=None):
    path = pathlib.Path(path or HEAVY_LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = open(path, "a+")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    return lease


def _release_heavy_lease(lease):
    if lease is None:
        return
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
    finally:
        lease.close()


def _base_state():
    return {
        "schema": "hawking.processing_queue.v1",
        "created_at": _now(),
        "updated_at": _now(),
        "status": "new",
        "plan": [{"label": LABEL, "lanes": list(LANES), "admitted": True}],
        "model": LABEL,
        "lanes": {lane: {"status": "pending", "attempts": 0} for lane in LANES},
        "source_release": {
            "status": "blocked",
            "automatic_deletion": False,
            "reason": "durable deployable artifact inventory and artifact-bound verification are required",
        },
    }


def _load_state():
    state = _read_json(STATE, _base_state())
    if state.get("schema") != "hawking.processing_queue.v1":
        state = _base_state()
    state.setdefault("lanes", {})
    for lane in LANES:
        state["lanes"].setdefault(lane, {"status": "pending", "attempts": 0})
    state["plan"] = [{"label": LABEL, "lanes": list(LANES), "admitted": True}]
    state["source_release"] = _base_state()["source_release"]
    return state


def _update_state(status=None, **extra):
    state = _load_state()
    if status is not None:
        state["status"] = status
    state.update(extra)
    state["updated_at"] = _now()
    _atomic_json(STATE, state)
    return state


def _update_lane(lane, **updates):
    state = _load_state()
    row = dict(state["lanes"].get(lane, {}))
    row.update(updates)
    row["updated_at"] = _now()
    state["lanes"][lane] = row
    state["active_lane"] = lane
    state["updated_at"] = row["updated_at"]
    _atomic_json(STATE, state)
    return row


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_stat_fingerprint_for_dir(model_dir):
    """Recompute audit_ladder's cheap, mutation-sensitive parent identity."""
    tokenizer_names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
        "spiece.model", "sentencepiece.bpe.model",
    }
    manifest = {"model_dir": os.path.realpath(model_dir), "files": []}
    try:
        names = sorted(os.listdir(model_dir))
    except OSError:
        return None
    for name in names:
        if not (
            name in {"config.json", "generation_config.json", *tokenizer_names}
            or name.endswith(".safetensors")
            or name.endswith(".safetensors.index.json")
        ):
            continue
        path = os.path.join(model_dir, name)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        item = {"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if name.endswith(".json") or name in tokenizer_names:
            try:
                item["sha256"] = _sha256(path)
            except OSError:
                return None
        manifest["files"].append(item)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _rooted_path(value):
    path = pathlib.Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _audit_identity_problems(identity, lane):
    """Validate current source, evaluator, recipe environment, and executable evidence.

    Coverage already binds the identity document's bytes.  This second boundary deliberately
    recomputes everything mutable so a once-valid receipt cannot promote a changed model, eval
    corpus, baker, adapter contract, or Doctor recipe after a detached resume.
    """
    problems = []
    if not isinstance(identity, dict):
        return ["audit identity document is not an object"]
    expected_model_dir = ROOT / "scratch/qwen-14b"
    expected_identity = {
        "schema": "hawking.audit_identity.v1",
        "recipe_version": AUDIT_RECIPE_VERSION,
        "model": LABEL,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            problems.append(f"audit identity {key} is not the pinned value")
    expected_lane = (f"{lane}_full"
                     if os.environ.get("HAWKING_STUDIO_RESEARCH_FULL") == "1" else lane)
    if identity.get("lane") != expected_lane:
        problems.append("audit identity lane is not the requested lane")

    try:
        recorded_model_dir = os.path.realpath(str(identity.get("model_dir", "")))
        expected_real_model_dir = os.path.realpath(expected_model_dir)
        if recorded_model_dir != expected_real_model_dir:
            problems.append("audit identity model_dir is not the canonical 14B parent")
    except Exception:
        problems.append("audit identity model_dir is invalid")
    current_model_fingerprint = _model_stat_fingerprint_for_dir(expected_model_dir)
    if current_model_fingerprint is None:
        problems.append("canonical 14B parent cannot be fingerprinted")
    elif identity.get("model_fingerprint") != current_model_fingerprint:
        problems.append("audit identity model_fingerprint is stale")

    try:
        eval_path = _rooted_path(identity["eval_text_path"])
        expected_eval_path = _rooted_path(
            os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")
        ).resolve(strict=False)
        if eval_path.resolve(strict=False) != expected_eval_path:
            problems.append("audit identity eval text is not the current PPL_TEXT corpus")
        if not expected_eval_path.is_file():
            problems.append("audit identity eval text is absent")
        elif identity.get("eval_text_sha256") != _sha256(expected_eval_path):
            problems.append("audit identity eval_text_sha256 is stale")
    except Exception:
        problems.append("audit identity eval text binding is invalid")

    pinned_environment = {
        "device": "cpu",
        "dtype": "torch.bfloat16",
        "multiwindow": 4,
        "studio_tripwire": True,
        "bake_quality": os.environ.get("BAKE_QUALITY") == "1",
        "strand_f32_metric": os.environ.get("STRAND_F32_METRIC"),
        "strand_f32_search": os.environ.get("STRAND_F32_SEARCH"),
        "doctor_grad_accum": 4,
        "doctor_kd_topk": 64,
        "doctor_target_regex": None,
    }
    for key, expected in pinned_environment.items():
        if identity.get(key) != expected:
            problems.append(f"audit identity {key} is not the pinned Studio value")
    actmean = os.environ.get("BAKE_ACTMEAN")
    if actmean and not os.path.isfile(actmean):
        problems.append("current BAKE_ACTMEAN path is absent")
    expected_actmean_path = os.path.realpath(actmean) if actmean and os.path.isfile(actmean) else None
    expected_actmean_sha = _sha256(actmean) if actmean and os.path.isfile(actmean) else None
    if identity.get("bake_actmean_path") != expected_actmean_path:
        problems.append("audit identity bake_actmean_path is not the current Studio value")
    if identity.get("bake_actmean_sha256") != expected_actmean_sha:
        problems.append("audit identity bake_actmean_sha256 is stale")

    evidence = identity.get("evidence_files")
    if not isinstance(evidence, dict):
        problems.append("audit identity evidence_files is not an object")
        evidence = {}
    normalized_evidence = {}
    for raw_path, recorded_hash in evidence.items():
        try:
            evidence_path = _rooted_path(raw_path).resolve(strict=False)
        except Exception:
            problems.append(f"audit identity evidence path is invalid: {raw_path!r}")
            continue
        normalized_evidence[evidence_path] = recorded_hash
        if not evidence_path.is_file():
            problems.append(f"audit identity evidence is absent: {raw_path}")
        else:
            try:
                if recorded_hash != _sha256(evidence_path):
                    problems.append(f"audit identity evidence hash is stale: {raw_path}")
            except OSError:
                problems.append(f"audit identity evidence cannot be hashed: {raw_path}")

    required_evidence = [
        ROOT / "vendor/strand-quant/target/release/quantize-model",
        ROOT / "tools/condense/audit_ladder.py",
        ROOT / "tools/condense/doctor.py",
        ROOT / "tools/condense/multi_eval.py",
        ROOT / "tools/condense/adapter_contract.py",
        ROOT / "tools/condense/tripwire_gate.py",
    ]
    calibration = ROOT / "scratch/calib_corpus.txt"
    if calibration.is_file():
        required_evidence.append(calibration)
    for required_path in required_evidence:
        resolved = required_path.resolve(strict=False)
        if resolved not in normalized_evidence:
            try:
                display = str(required_path.relative_to(ROOT))
            except ValueError:
                display = str(required_path)
            problems.append(f"audit identity omits required evidence: {display}")
    return problems


def _download_marker_status():
    marker = _read_json(DOWNLOAD_MARKER, {})
    ok = procure._verified_marker_valid(
        marker,
        label=LABEL,
        hf_id=HF_ID,
        local_dir=str(SOURCE.relative_to(ROOT)),
        require_verify=True,
    )
    blockers = []
    if not ok:
        blockers.append("verified marker is absent, invalid, or not bound to the staged 14B path")
    if not SOURCE.is_dir():
        blockers.append(f"staged source directory is absent: {SOURCE}")
    if SOURCE.is_dir() and not (SOURCE / "config.json").is_file():
        blockers.append("staged source has no config.json")
    return {
        "ok": ok and not blockers,
        "path": str(DOWNLOAD_MARKER),
        "sha256": _sha256(DOWNLOAD_MARKER) if ok else None,
        "marker": marker if ok else None,
        "blockers": blockers,
    }


def _download_72_status():
    spec = procure._resolve("72B")
    _, marker_path = procure._checkpoint_paths("72B")
    marker = procure._read_json(marker_path, {})
    return {
        "ok": procure._verified_marker_valid(
            marker, label=spec.label, hf_id=spec.hf_id,
            local_dir=SOURCE_72, require_verify=True,
        ),
        "path": str(marker_path),
    }


def _promote(marker_status):
    """Atomically publish the canonical model name without moving model bytes."""
    if not marker_status.get("ok"):
        raise RuntimeError("14B download marker did not pass path-bound validation")
    expected_target = os.path.relpath(SOURCE, CANONICAL.parent)
    if CANONICAL.is_symlink():
        try:
            current_resolved = CANONICAL.resolve(strict=True)
            points_to_source = os.path.samefile(CANONICAL, SOURCE)
        except OSError as exc:
            current_resolved = None
            points_to_source = False
            broken_reason = f"{type(exc).__name__}: {exc}"
        else:
            broken_reason = None
        if not points_to_source:
            # Replacing a symlink is metadata-only; a real directory is never replaced.
            tmp = CANONICAL.with_name(f".{CANONICAL.name}.{os.getpid()}.tmp")
            try:
                os.symlink(expected_target, tmp)
                os.replace(tmp, CANONICAL)
                _fsync_dir(CANONICAL.parent)
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
            action = "reconciled-symlink"
            previous = {"resolved": str(current_resolved) if current_resolved else None,
                        "error": broken_reason}
        else:
            action = "already-promoted"
            previous = None
    elif os.path.lexists(CANONICAL):
        raise RuntimeError(f"refusing to replace non-symlink canonical path: {CANONICAL}")
    else:
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        tmp = CANONICAL.with_name(f".{CANONICAL.name}.{os.getpid()}.tmp")
        try:
            os.symlink(expected_target, tmp)
            os.replace(tmp, CANONICAL)
            _fsync_dir(CANONICAL.parent)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        action = "created-symlink"
        previous = None

    if not CANONICAL.is_symlink() or not os.path.samefile(CANONICAL, SOURCE):
        raise RuntimeError("canonical 14B symlink did not resolve to the verified source")
    receipt = {
        "schema": "hawking.model_promotion.v1",
        "status": "pass",
        "promoted_at": _now(),
        "label": LABEL,
        "hf_id": HF_ID,
        "source": str(SOURCE.relative_to(ROOT)),
        "canonical": str(CANONICAL.relative_to(ROOT)),
        "canonical_kind": "symlink",
        "symlink_target": os.readlink(CANONICAL),
        "action": action,
        "previous_symlink": previous,
        "download_marker": str(DOWNLOAD_MARKER.relative_to(ROOT)),
        "download_marker_sha256": marker_status["sha256"],
        "source_release": "blocked-until-durable-artifact-bound-verification",
    }
    _atomic_json(PROMOTION_RECEIPT, receipt)
    return receipt


def _thermal_snapshot():
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5, check=False
        )
        detail = (result.stdout + result.stderr).strip()
        return {"ok": thermal_output_ok(result.returncode, detail),
                "returncode": result.returncode, "detail": detail[-1000:]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _evaluate_resources(snapshot, thermal):
    blockers = []
    if not snapshot.get("ok"):
        blockers.append(f"resource snapshot unavailable: {snapshot.get('error', 'unknown')}")
    elif classify_resource_state(snapshot) != "green":
        blockers.append(
            f"memory is not green (pressure={snapshot.get('pressure_name')}, "
            f"swap={float(snapshot.get('swap_used_mb') or 0.0):.0f}MB)"
        )
    if "AC Power" not in str(snapshot.get("power_source", "")):
        blockers.append("AC power not confirmed")
    if not thermal.get("ok"):
        blockers.append("thermal/performance state is not green")
    usable = float(snapshot.get("disk_usable_now_gb") or 0.0)
    required = max(
        PROCESSING_SCRATCH_GB,
        float(snapshot.get("scratch_reserve_gb") or DEFAULT_HARDWARE.scratch_reserve_gb),
    )
    if usable < required:
        blockers.append(f"disk usable after reserve {usable:.1f}GB < scratch {required:.1f}GB")
    return {"ok": not blockers, "blockers": blockers,
            "resources": snapshot, "thermal": thermal}


def _resource_gate():
    return _evaluate_resources(resource_snapshot(), _thermal_snapshot())


def _studio_owner():
    info = _read_json(STUDIO_RUN_PID, {})
    pid = info.get("pid")
    return {"active": _pid_alive(pid), "pid": pid, "info": info}


def _orphan_heavy_work(ps_text=None):
    """Detect heavy Studio descendants even if their GO supervisor/RUN_PID died."""
    if ps_text is None:
        try:
            ps_text = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,command="], capture_output=True,
                text=True, timeout=5, check=True,
            ).stdout
        except Exception as exc:
            return [{"pid": None, "ppid": None,
                     "command": f"process inventory unavailable: {type(exc).__name__}: {exc}"}]
    patterns = (
        "tools/condense/audit_ladder.py",
        "tools/condense/studio_run.py --model",
        "tools/condense/doctor.py lora",
        "tools/condense/doctor.py blockwise",
        "tools/condense/doctor.py strand",
        "vendor/strand-quant/target/release/quantize-model",
    )
    rows = []
    for line in str(ps_text).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if pid == os.getpid() or not any(pattern in command for pattern in patterns):
            continue
        rows.append({"pid": pid, "ppid": ppid, "command": command[:300]})
    return rows


def _active_download_work():
    """Return actual controllers/children, excluding an idle queue supervisor.

    Excluding the queue supervisor avoids a deadlock: after 72B it deliberately
    waits for this processor at the pre-120B barrier.
    """
    rows = []
    seen = set()

    def add(pid, role, source, label=None):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return
        if pid == os.getpid() or pid in seen or not _pid_alive(pid):
            return
        seen.add(pid)
        rows.append({"pid": pid, "role": role, "label": label, "source": str(source)})

    if DOWNLOAD_STATE_DIR.exists():
        for path in DOWNLOAD_STATE_DIR.glob("*.pid.json"):
            info = _read_json(path, {})
            add(info.get("pid"), "download-controller", path, info.get("label"))
        for path in DOWNLOAD_STATE_DIR.glob("*.state.json"):
            info = _read_json(path, {})
            add(info.get("child_pid"), "download-child", path, info.get("label"))
    queue_state = _read_json(DOWNLOAD_QUEUE_STATE, {})
    add(queue_state.get("child_pid"), "download-queue-child", DOWNLOAD_QUEUE_STATE,
        queue_state.get("active_label"))
    return rows


def _lane_paths(lane):
    return {
        "coverage": ROOT / f"reports/cron/{lane}_{LABEL}.coverage.json",
        "complete": ROOT / f"reports/cron/{lane}_{LABEL}.complete.json",
        "audit": ROOT / f"reports/cron/{lane}_{LABEL}.jsonl",
        "identity": ROOT / f"reports/cron/{lane}_{LABEL}.identity.json",
        "log": ROOT / f"reports/condense/processing_queue_{LABEL}_{lane}.log",
    }


def _lane_validation(lane):
    paths = _lane_paths(lane)
    coverage = _read_json(paths["coverage"], {})
    complete = _read_json(paths["complete"], {})
    blockers = []
    canonical_required = REQUIRED_LANE_CONFIGS.get(lane)
    if canonical_required is None:
        return {
            "ok": False, "lane": lane, "blockers": [f"unknown processing lane: {lane}"],
            "coverage_path": str(paths["coverage"]),
            "complete_path": str(paths["complete"]), "audit_path": str(paths["audit"]),
            "identity_path": str(paths["identity"]), "required_configs": [],
            "successful_configs": [], "config_statuses": {},
        }
    canonical_required = list(canonical_required)
    expected_audit_set = (f"{lane}_full"
                          if os.environ.get("HAWKING_STUDIO_RESEARCH_FULL") == "1" else lane)
    if not (
        coverage.get("schema") == "hawking.studio_core_coverage.v1"
        and coverage.get("status") == "pass"
        and coverage.get("model") == LABEL
        and coverage.get("lane") == lane
    ):
        blockers.append("coverage receipt identity/status mismatch")
    if coverage.get("required_configs") != canonical_required:
        blockers.append("coverage required-config list does not match the pinned lane recipe")
    required_names = canonical_required
    configs = coverage.get("configs") if isinstance(coverage.get("configs"), dict) else {}
    successful_names = [
        name for name in required_names if configs.get(name, {}).get("status") == "pass"
    ]
    if not required_names or len(successful_names) != len(required_names):
        blockers.append(
            f"required config coverage is incomplete ({len(successful_names)}/{len(required_names)})"
        )
    if any(coverage.get(key) for key in (
        "missing_configs", "error_configs", "invalid_configs", "parse_errors",
        "identity_errors",
    )):
        blockers.append("coverage receipt records missing/error/invalid configs or identity errors")
    if coverage.get("identity_errors") != []:
        blockers.append("coverage identity_errors must be present and empty")
    audit_value = coverage.get("audit_jsonl")
    try:
        audit_path = pathlib.Path(str(audit_value))
        if not audit_path.is_absolute():
            audit_path = ROOT / audit_path
        if audit_path.resolve(strict=False) != paths["audit"].resolve(strict=False):
            blockers.append("coverage receipt is not bound to the expected audit JSONL")
    except Exception:
        blockers.append("coverage audit_jsonl path is invalid")
        audit_path = paths["audit"]
    if not audit_path.is_file():
        blockers.append("coverage audit JSONL is absent")
    elif coverage.get("audit_sha256") != _sha256(audit_path):
        blockers.append("coverage audit hash does not match the current JSONL")
    identity_doc = {}
    try:
        identity_path = pathlib.Path(str(coverage.get("audit_identity")))
        if not identity_path.is_absolute():
            identity_path = ROOT / identity_path
        if identity_path.resolve(strict=False) != paths["identity"].resolve(strict=False):
            blockers.append("coverage is not bound to the expected audit identity path")
        if not identity_path.is_file():
            blockers.append("coverage audit identity is absent")
        elif coverage.get("audit_identity_sha256") != _sha256(identity_path):
            blockers.append("coverage audit identity hash does not match current bytes")
        else:
            identity_doc = _read_json(identity_path, {})
    except Exception:
        blockers.append("coverage audit identity path is invalid")
        identity_path = paths["identity"]
    blockers.extend(_audit_identity_problems(identity_doc, lane))
    if not (coverage.get("completed_at") or coverage.get("generated_at") or coverage.get("timestamp")):
        blockers.append("coverage receipt has no timestamp")
    if not (
        complete.get("schema") == "hawking.studio_model_complete.v1"
        and complete.get("status") == "pass"
        and complete.get("model") == LABEL
        and complete.get("lane") == lane
        and complete.get("audit_set") == expected_audit_set
    ):
        blockers.append("model completion receipt identity/status mismatch")
    if complete.get("required_configs") != canonical_required:
        blockers.append("model completion required-config list does not match pinned lane recipe")
    try:
        complete_audit = pathlib.Path(str(complete.get("audit_jsonl")))
        if not complete_audit.is_absolute():
            complete_audit = ROOT / complete_audit
        if complete_audit.resolve(strict=False) != paths["audit"].resolve(strict=False):
            blockers.append("model completion is not bound to the expected audit JSONL")
    except Exception:
        blockers.append("model completion audit binding is invalid")
    try:
        complete_coverage = pathlib.Path(str(complete.get("coverage")))
        if not complete_coverage.is_absolute():
            complete_coverage = ROOT / complete_coverage
        if complete_coverage.resolve(strict=False) != paths["coverage"].resolve(strict=False):
            blockers.append("model completion is not bound to the expected coverage receipt")
        elif paths["coverage"].is_file() and complete.get("coverage_sha256") != _sha256(paths["coverage"]):
            blockers.append("model completion coverage hash does not match")
    except Exception:
        blockers.append("model completion coverage binding is invalid")
    receipt_doc = None
    try:
        receipt = pathlib.Path(str(complete.get("receipt")))
        if not receipt.is_absolute():
            receipt = ROOT / receipt
        if not receipt.is_file() or complete.get("receipt_sha256") != _sha256(receipt):
            blockers.append("model completion lane receipt hash does not match")
        else:
            receipt_doc = _read_json(receipt, {})
    except Exception:
        blockers.append("model completion lane receipt binding is invalid")

    if lane == "subbit":
        if complete.get("result_kind") != "research-campaign" or complete.get("floor_jsonl") is not None:
            blockers.append("subbit completion must be a floor-free research campaign")
        expected = {
            "schema": "hawking.subbit_campaign_complete.v1",
            "project": "hawking",
            "status": "research-complete",
            "model": LABEL,
            "lane": "subbit",
            "artifact_class": "reconstruction_oracle",
            "deployable": False,
            "product_gate": False,
            "floor_bpw": None,
            "required_configs": canonical_required,
            "audit_jsonl": str(paths["audit"].relative_to(ROOT)),
            "audit_sha256": _sha256(paths["audit"]) if paths["audit"].is_file() else None,
            "coverage": str(paths["coverage"].relative_to(ROOT)),
            "coverage_sha256": _sha256(paths["coverage"]) if paths["coverage"].is_file() else None,
        }
        if not isinstance(receipt_doc, dict) or any(
            receipt_doc.get(key) != value for key, value in expected.items()
        ):
            blockers.append("subbit research-campaign receipt semantics/bindings are invalid")
        if not isinstance(receipt_doc, dict) or not receipt_doc.get("promotion_blockers"):
            blockers.append("subbit campaign must preserve explicit promotion blockers")
    else:
        expected_floor = ROOT / "reports/cron/bit_floor_curve.jsonl"
        if complete.get("result_kind") != "deployable-floor-experiment":
            blockers.append("studio completion result-kind/floor binding is invalid")
        floor_ok, floor_problems = validate_floor_binding(
            complete, ROOT, "studio", LABEL, expected_floor, paths["audit"],
        )
        if not floor_ok:
            blockers.extend(f"studio floor binding: {problem}" for problem in floor_problems)
        try:
            if float(complete["floor_row"]["gate_pct"]) != float(
                os.environ.get("FLOOR_GATE_PCT", "2.0")
            ):
                blockers.append("studio floor gate_pct does not match current FLOOR_GATE_PCT")
        except (KeyError, TypeError, ValueError):
            blockers.append("studio floor gate_pct is missing or invalid")
        receipt_floor_ok, receipt_floor_problems = validate_receipt_floor_row(
            receipt_doc, complete.get("floor_row"),
        )
        if not receipt_floor_ok:
            blockers.extend(
                f"studio receipt/floor row: {problem}" for problem in receipt_floor_problems
            )
        artifact = str((receipt_doc or {}).get("condensed_artifact", ""))
        if not (
            isinstance(receipt_doc, dict)
            and receipt_doc.get("project") == "hawking"
            and receipt_doc.get("receipt_version") == "0.2"
            and LABEL in str(receipt_doc.get("source_model", ""))
            and str(paths["audit"].relative_to(ROOT)) in artifact
            and isinstance(receipt_doc.get("effective_bpw"), (int, float))
            and receipt_doc.get("effective_bpw") > 0
            and receipt_doc.get("claim_type") in {"baseline", "negative", "scale-point"}
            and receipt_doc.get("quality_gate") in {"pass", "warn", "fail"}
        ):
            blockers.append("studio floor/negative receipt semantics are invalid")
    return {
        "ok": not blockers,
        "lane": lane,
        "blockers": blockers,
        "coverage_path": str(paths["coverage"]),
        "complete_path": str(paths["complete"]),
        "audit_path": str(paths["audit"]),
        "identity_path": str(paths["identity"]),
        "required_configs": required_names,
        "successful_configs": successful_names,
        "config_statuses": {
            name: configs.get(name, {}).get("status") for name in required_names
        },
    }


def completion_status():
    """Public fail-closed barrier used by the download queue before 120B."""
    state = _load_state()
    marker = _download_marker_status()
    promotion = _read_json(PROMOTION_RECEIPT, {})
    promotion_ok = bool(
        promotion.get("schema") == "hawking.model_promotion.v1"
        and promotion.get("status") == "pass"
        and promotion.get("label") == LABEL
        and promotion.get("download_marker_sha256") == marker.get("sha256")
        and CANONICAL.is_symlink()
        and marker.get("ok")
    )
    if promotion_ok:
        try:
            promotion_ok = os.path.samefile(CANONICAL, SOURCE)
        except OSError:
            promotion_ok = False
    lanes = {lane: _lane_validation(lane) for lane in LANES}
    ok = state.get("status") == "complete" and promotion_ok and all(
        row["ok"] for row in lanes.values()
    )
    return {
        "ok": ok,
        "state_status": state.get("status"),
        "state_path": str(STATE),
        "download_marker_ok": marker.get("ok"),
        "promotion_ok": promotion_ok,
        "lanes": lanes,
        "source_release": state.get("source_release"),
    }


def _sleep_interruptible(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _stop_requested or SHARED_DRAIN.exists():
            return False
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    return True


def _terminate_child(proc, reason):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    _update_state(last_termination={"at": _now(), "pid": proc.pid, "reason": reason})
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _lane_progress(lane, returncode, validation):
    """Build stable semantic progress plus live audit telemetry for detached retries."""
    paths = _lane_paths(lane)
    coverage = _read_json(paths["coverage"], {})
    semantic = {
        "returncode": int(returncode),
        "successful_configs": validation.get("successful_configs", []),
        "config_statuses": validation.get("config_statuses", {}),
        "blockers": validation.get("blockers", []),
        "missing_configs": coverage.get("missing_configs", []),
        "error_configs": coverage.get("error_configs", []),
        "invalid_configs": coverage.get("invalid_configs", []),
        "parse_errors": coverage.get("parse_errors", []),
        "identity_errors": coverage.get("identity_errors", []),
    }
    fingerprint = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audit_bytes = None
    audit_sha256 = None
    try:
        audit_bytes = paths["audit"].stat().st_size
        audit_sha256 = _sha256(paths["audit"])
    except OSError:
        pass
    return {
        **semantic,
        "semantic_fingerprint": fingerprint,
        "audit_bytes": audit_bytes,
        "audit_sha256": audit_sha256,
        "observed_at": _now(),
    }


def _next_stalled_output_count(previous_fingerprint, previous_count, fingerprint):
    if previous_fingerprint == fingerprint:
        return max(0, int(previous_count or 0)) + 1
    return 1


def _output_retry_allowed(attempts_this_run, stalled_output_attempts):
    """Bound both total work in this daemon invocation and identical-output retries."""
    return (
        int(attempts_this_run) < MAX_OUTPUT_ATTEMPTS
        and int(stalled_output_attempts) < MAX_OUTPUT_ATTEMPTS
    )


def _effective_lane_returncode(child_returncode, validation_ok):
    child_returncode = int(child_returncode or 0)
    if child_returncode == 0 and validation_ok:
        return 0
    if child_returncode == 4 or not validation_ok:
        return 4
    return child_returncode


def _run_lane_once(lane, heavy_lease):
    validation = _lane_validation(lane)
    if validation["ok"]:
        _update_lane(lane, status="pass", validation=validation, child_pid=None,
                     completed_at=_now(), stalled_output_attempts=0,
                     output_failure_fingerprint=None)
        return 0
    gate = _resource_gate()
    downloads = _active_download_work()
    if not gate["ok"] or downloads:
        status = "paused-download" if downloads else "paused-resources"
        _update_lane(lane, status=status, child_pid=None, returncode=75,
                     safety=gate, download_activity=downloads)
        return 75
    paths = _lane_paths(lane)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    log = open(paths["log"], "ab", buffering=0)
    cmd = [sys.executable, str(ROOT / "tools/condense/studio_run.py"),
           "--model", LABEL, lane]
    child_env = os.environ.copy()
    child_env[HEAVY_LEASE_FD_ENV] = str(heavy_lease.fileno())
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        pass_fds=(heavy_lease.fileno(),),
    )
    log.close()
    state = _load_state()
    attempts = int(state["lanes"].get(lane, {}).get("attempts", 0)) + 1
    _update_state(status="running", child_pid=proc.pid, child_pgid=proc.pid,
                  active_lane=lane, child_log=str(paths["log"]))
    _update_lane(lane, status="running", attempts=attempts, child_pid=proc.pid,
                 command=cmd, started_at=_now())
    while proc.poll() is None:
        gate = _resource_gate()
        downloads = _active_download_work()
        _update_lane(lane, status="running", child_pid=proc.pid, safety=gate,
                     download_activity=downloads)
        if _stop_requested or SHARED_DRAIN.exists():
            _terminate_child(proc, "Studio/processing drain requested")
            _update_lane(lane, status="paused-drain", child_pid=None, returncode=130)
            _update_state(child_pid=None, child_pgid=None)
            return 130
        if not gate["ok"]:
            reason = "; ".join(gate["blockers"])
            _terminate_child(proc, reason)
            _update_lane(lane, status="paused-resources", child_pid=None,
                         returncode=75, safety=gate)
            _update_state(child_pid=None, child_pgid=None)
            return 75
        if downloads:
            _terminate_child(proc, "download controller/child became active")
            _update_lane(lane, status="paused-download", child_pid=None,
                         returncode=75, download_activity=downloads)
            _update_state(child_pid=None, child_pgid=None)
            return 75
        if not _sleep_interruptible(POLL_S):
            continue
    rc = int(proc.returncode or 0)
    _update_state(child_pid=None, child_pgid=None)
    validation = _lane_validation(lane)
    if rc == 0 and validation["ok"]:
        _update_lane(lane, status="pass", returncode=0, child_pid=None,
                     validation=validation, completed_at=_now(),
                     stalled_output_attempts=0, output_failure_fingerprint=None)
        return 0
    # studio_run returns 4 for fail-closed coverage/receipt output.  Any other child result that
    # still leaves invalid receipts is the same retry class: preserve the child code as evidence,
    # then retry only within the bounded semantic-progress budget below.
    effective_rc = _effective_lane_returncode(rc, validation["ok"])
    progress = _lane_progress(lane, effective_rc, validation)
    previous = _load_state()["lanes"].get(lane, {})
    stalled = _next_stalled_output_count(
        previous.get("output_failure_fingerprint"),
        previous.get("stalled_output_attempts"),
        progress["semantic_fingerprint"],
    )
    _update_lane(
        lane,
        status="retryable-output-failure" if effective_rc == 4 else "failed",
        returncode=effective_rc,
        child_returncode=rc,
        child_pid=None,
        validation=validation,
        progress=progress,
        output_failure_fingerprint=progress["semantic_fingerprint"],
        stalled_output_attempts=stalled,
    )
    return effective_rc


def _wait_for_admission():
    while True:
        if _stop_requested or SHARED_DRAIN.exists():
            return 130
        owner = _studio_owner()
        gate = _resource_gate()
        downloads = _active_download_work()
        download_72 = _download_72_status()
        orphan_heavy = [] if owner["active"] else _orphan_heavy_work()
        if (not owner["active"] and not orphan_heavy and not downloads
                and download_72["ok"] and gate["ok"]):
            return 0
        if owner["active"]:
            status = "waiting-studio"
        elif downloads:
            status = "waiting-download"
        elif orphan_heavy:
            status = "waiting-orphan-heavy"
        elif not download_72["ok"]:
            status = "waiting-72-verification"
        else:
            status = "waiting-resources"
        _update_state(
            status=status,
            studio_owner=owner,
            download_activity=downloads,
            download_72=download_72,
            safety=gate,
            orphan_heavy_work=orphan_heavy,
        )
        if not _sleep_interruptible(POLL_S):
            continue


def _acquire_admitted_lease():
    """Wait for green admission, acquire the shared lease, then recheck and claim RUN_PID."""
    while True:
        rc = _wait_for_admission()
        if rc != 0:
            return rc, None
        lease = _try_heavy_lease()
        if lease is None:
            _update_state(status="waiting-heavy-lease", heavy_lock=str(HEAVY_LOCK))
            if not _sleep_interruptible(POLL_S):
                continue
            continue

        # Admission can change between the last sample and flock acquisition.
        owner = _studio_owner()
        gate = _resource_gate()
        downloads = _active_download_work()
        download_72 = _download_72_status()
        orphan_heavy = [] if owner["active"] else _orphan_heavy_work()
        if (owner["active"] or orphan_heavy or downloads
                or not download_72["ok"] or not gate["ok"]):
            _release_heavy_lease(lease)
            _update_state(status="waiting-admission-recheck", studio_owner=owner,
                          safety=gate, download_activity=downloads, download_72=download_72,
                          orphan_heavy_work=orphan_heavy)
            if not _sleep_interruptible(POLL_S):
                continue
            continue
        if not _claim_studio_slot():
            _release_heavy_lease(lease)
            if not _sleep_interruptible(POLL_S):
                continue
            continue
        return 0, lease


def _claim_studio_slot():
    owner = _studio_owner()
    if owner["active"] and int(owner.get("pid") or -1) != os.getpid():
        return False
    _atomic_json(STUDIO_RUN_PID, {
        "pid": os.getpid(),
        "started_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "role": "processing-queue",
        "model": LABEL,
        "heavy_lock": str(HEAVY_LOCK),
    })
    return True


def _release_studio_slot():
    info = _read_json(STUDIO_RUN_PID, {})
    if info.get("pid") == os.getpid() and info.get("role") == "processing-queue":
        try:
            STUDIO_RUN_PID.unlink()
            _fsync_dir(STUDIO_RUN_PID.parent)
        except FileNotFoundError:
            pass


def run_queue():
    global _stop_requested
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    singleton = open(LOCK_FILE, "a+")
    model_lock = open(MODEL_LOCK_FILE, "a+")
    try:
        fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[processing-queue] another supervisor holds the singleton lock", file=sys.stderr)
        return 2
    try:
        fcntl.flock(model_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fcntl.flock(singleton.fileno(), fcntl.LOCK_UN)
        singleton.close()
        print("[processing-queue] the 14B model lock is held", file=sys.stderr)
        return 2

    def request_stop(_sig, _frame):
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _atomic_json(PID_FILE, {"schema": "hawking.processing_queue_pid.v1",
                            "pid": os.getpid(), "started_at": _now(), "log": str(LOG_FILE)})
    _update_state(status="validating", supervisor_pid=os.getpid())
    rc = 0
    heavy_lease = None
    try:
        marker = _download_marker_status()
        if not marker["ok"]:
            _update_state(status="blocked-download-marker", download_marker=marker)
            return 1
        _update_state(status="waiting-studio", download_marker=marker)
        rc, heavy_lease = _acquire_admitted_lease()
        if rc != 0:
            _update_state(status="paused-drain", paused_at=_now())
            return rc
        # Publish only after the shared heavy lease and RUN_PID claim. The direct
        # model subprocesses below remain under this lease for their whole lane.
        try:
            promotion = _promote(marker)
        except Exception as exc:
            _update_state(status="blocked-promotion", error=f"{type(exc).__name__}: {exc}")
            return 1
        _update_state(status="running", promotion=promotion)

        output_attempts = {lane: 0 for lane in LANES}
        for lane in LANES:
            while True:
                rc = _run_lane_once(lane, heavy_lease)
                if rc == 75:
                    _release_studio_slot()
                    _release_heavy_lease(heavy_lease)
                    heavy_lease = None
                    rc, heavy_lease = _acquire_admitted_lease()
                    if rc != 0:
                        break
                    continue
                if rc == 4:
                    output_attempts[lane] += 1
                    lane_state = _load_state()["lanes"].get(lane, {})
                    stalled = int(lane_state.get("stalled_output_attempts", 0))
                    if _output_retry_allowed(output_attempts[lane], stalled):
                        _update_lane(
                            lane,
                            status="waiting-output-retry",
                            retry_after_seconds=OUTPUT_RETRY_DELAY_S,
                            output_attempts_this_run=output_attempts[lane],
                            max_output_attempts=MAX_OUTPUT_ATTEMPTS,
                        )
                        if not _sleep_interruptible(OUTPUT_RETRY_DELAY_S):
                            rc = 130
                            break
                        continue
                break
            if rc != 0:
                break

        if rc == 0:
            _update_state(status="complete", active_lane=None, child_pid=None,
                          completed_at=_now(), terminal_reason="14B studio and subbit coverage passed")
        elif rc == 130:
            _update_state(status="paused-drain", child_pid=None, paused_at=_now())
        elif rc == 75:
            _update_state(status="waiting-studio", child_pid=None, studio_owner=_studio_owner())
        else:
            _update_state(status="blocked", child_pid=None, returncode=rc, blocked_at=_now())
        return rc
    finally:
        _release_studio_slot()
        _release_heavy_lease(heavy_lease)
        info = _read_json(PID_FILE, {})
        if info.get("pid") == os.getpid():
            try:
                PID_FILE.unlink()
                _fsync_dir(PID_FILE.parent)
            except FileNotFoundError:
                pass
        fcntl.flock(model_lock.fileno(), fcntl.LOCK_UN)
        fcntl.flock(singleton.fileno(), fcntl.LOCK_UN)
        model_lock.close()
        singleton.close()


def start_queue():
    if SHARED_DRAIN.exists():
        print(f"[processing-queue] Studio drain is active at {SHARED_DRAIN}", file=sys.stderr)
        return 130
    info = _read_json(PID_FILE, {})
    if _pid_alive(info.get("pid")):
        print(f"[processing-queue] already active pid={info['pid']}", file=sys.stderr)
        return 0
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab", buffering=0)
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "run"]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-dimsu", *cmd]
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    log.close()
    _atomic_json(PID_FILE, {"schema": "hawking.processing_queue_pid.v1", "pid": proc.pid,
                            "started_at": _now(), "log": str(LOG_FILE), "cmd": cmd})
    print(f"[processing-queue] detached pid={proc.pid}; log={LOG_FILE}", file=sys.stderr)
    return 0


def stop_queue():
    info = _read_json(PID_FILE, {})
    pid = info.get("pid")
    if not _pid_alive(pid):
        print("[processing-queue] not active", file=sys.stderr)
        return 0
    _update_state(status="stop-requested", stop_requested_at=_now())
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except OSError:
        os.kill(int(pid), signal.SIGTERM)
    print(f"[processing-queue] stop requested for pid={pid}", file=sys.stderr)
    return 0


def status():
    info = _read_json(PID_FILE, {})
    payload = {
        "schema": "hawking.processing_queue_status.v1",
        "generated_at": _now(),
        "active": _pid_alive(info.get("pid")),
        "pid": info.get("pid"),
        "state": _load_state(),
        "completion": completion_status(),
        "download_marker": _download_marker_status(),
        "studio_owner": _studio_owner(),
        "download_activity": _active_download_work(),
        "download_72": _download_72_status(),
        "resources": _resource_gate(),
        "drain_requested": SHARED_DRAIN.exists(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def selftest():
    global ROOT
    green = {
        "ok": True, "pressure_level": 1, "pressure_name": "normal", "swap_used_mb": 0.0,
        "power_source": "Now drawing from 'AC Power'", "disk_usable_now_gb": 100.0,
        "scratch_reserve_gb": 64.0,
    }
    assert _evaluate_resources(green, {"ok": True})["ok"]
    assert not _evaluate_resources({**green, "pressure_level": 2}, {"ok": True})["ok"]
    assert not _evaluate_resources({**green, "pressure_level": None}, {"ok": True})["ok"]
    assert not _evaluate_resources({**green, "swap_used_mb": None}, {"ok": True})["ok"]
    assert not _evaluate_resources({**green, "disk_usable_now_gb": 89}, {"ok": True})["ok"]
    assert not _evaluate_resources(green, {"ok": False})["ok"]
    with tempfile.TemporaryDirectory() as td:
        parent = pathlib.Path(td)
        source = parent / "source"
        canonical = parent / "canonical"
        source.mkdir()
        os.symlink("source", canonical)
        assert canonical.is_symlink() and os.path.samefile(canonical, source)
        lease_path = parent / "studio_heavy.lock"
        lease = _try_heavy_lease(lease_path)
        assert lease is not None
        assert _try_heavy_lease(lease_path) is None
        _release_heavy_lease(lease)
        lease = _try_heavy_lease(lease_path)
        inherited_env = os.environ.copy()
        inherited_env[HEAVY_LEASE_FD_ENV] = str(lease.fileno())
        wrapper = subprocess.run(
            [sys.executable, str(ROOT / "tools/condense/studio_run.py"),
             "--lease-selftest-child"],
            cwd=ROOT, env=inherited_env, pass_fds=(lease.fileno(),),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        )
        assert wrapper.returncode == 0, wrapper.stderr
        _release_heavy_lease(lease)
        # Simulate a SIGKILLed supervisor: closing its descriptor without LOCK_UN must leave the
        # inherited child descriptor holding admission until that child exits.
        lease = _try_heavy_lease(lease_path)
        keeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.25)"],
            pass_fds=(lease.fileno(),),
        )
        lease.close()
        assert _try_heavy_lease(lease_path) is None
        keeper.wait(timeout=5)
        reacquired = _try_heavy_lease(lease_path)
        assert reacquired is not None
        _release_heavy_lease(reacquired)
    assert _next_stalled_output_count("same", 1, "same") == 2
    assert _next_stalled_output_count("old", 99, "progress") == 1
    assert _output_retry_allowed(MAX_OUTPUT_ATTEMPTS - 1, MAX_OUTPUT_ATTEMPTS - 1)
    assert not _output_retry_allowed(MAX_OUTPUT_ATTEMPTS, 0)
    assert not _output_retry_allowed(0, MAX_OUTPUT_ATTEMPTS)
    assert _effective_lane_returncode(0, True) == 0
    assert _effective_lane_returncode(0, False) == 4
    assert _effective_lane_returncode(1, False) == 4
    assert _effective_lane_returncode(1, True) == 1
    assert _orphan_heavy_work(
        "101 1 python tools/condense/audit_ladder.py scratch/qwen-7b 7B studio out\n"
        "102 1 python harmless.py\n"
    ) == [{
        "pid": 101, "ppid": 1,
        "command": "python tools/condense/audit_ladder.py scratch/qwen-7b 7B studio out",
    }]
    original_root = ROOT
    identity_env_keys = (
        "PPL_TEXT", "HAWKING_STUDIO_RESEARCH_FULL", "FLOOR_GATE_PCT",
        "BAKE_QUALITY", "BAKE_ACTMEAN", "STRAND_F32_METRIC", "STRAND_F32_SEARCH",
    )
    original_identity_env = {key: os.environ.get(key) for key in identity_env_keys}
    for key in identity_env_keys:
        os.environ.pop(key, None)
    try:
        with tempfile.TemporaryDirectory() as td:
            ROOT = pathlib.Path(td)

            # A lane identity is accepted only against the current parent, evaluator, baker, and
            # recovery code.  Build a small but structurally faithful isolated Studio tree.
            model_dir = ROOT / "scratch/qwen-14b"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen2", "hidden_size": 8}) + "\n",
                encoding="utf-8",
            )
            (model_dir / "tokenizer.json").write_text(
                json.dumps({"version": "1.0", "model": {"type": "BPE"}}) + "\n",
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").write_bytes(b"synthetic-model-weights")
            eval_text = ROOT / "scratch/selftest_eval.txt"
            eval_text.write_text("detached Studio identity selftest\n", encoding="utf-8")
            os.environ["PPL_TEXT"] = str(eval_text)
            evidence_paths = [
                ROOT / "vendor/strand-quant/target/release/quantize-model",
                ROOT / "tools/condense/audit_ladder.py",
                ROOT / "tools/condense/doctor.py",
                ROOT / "tools/condense/multi_eval.py",
                ROOT / "tools/condense/adapter_contract.py",
                ROOT / "tools/condense/tripwire_gate.py",
                ROOT / "scratch/calib_corpus.txt",
            ]
            for index, path in enumerate(evidence_paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"selftest-evidence-{index}\n".encode())

            def build_lane_fixture(lane):
                paths = _lane_paths(lane)
                paths["audit"].parent.mkdir(parents=True, exist_ok=True)
                required = list(REQUIRED_LANE_CONFIGS[lane])
                paths["audit"].write_text(
                    "".join(json.dumps({"model": LABEL, "config": name}) + "\n"
                            for name in required),
                    encoding="utf-8",
                )
                identity = {
                    "schema": "hawking.audit_identity.v1",
                    "model": LABEL,
                    "lane": lane,
                    "recipe_version": AUDIT_RECIPE_VERSION,
                    "model_dir": os.path.realpath(model_dir),
                    "model_fingerprint": _model_stat_fingerprint_for_dir(model_dir),
                    "eval_text_path": os.path.realpath(eval_text),
                    "eval_text_sha256": _sha256(eval_text),
                    "device": "cpu",
                    "dtype": "torch.bfloat16",
                    "multiwindow": 4,
                    "studio_tripwire": True,
                    "bake_quality": os.environ.get("BAKE_QUALITY") == "1",
                    "bake_actmean_path": (
                        os.path.realpath(os.environ["BAKE_ACTMEAN"])
                        if os.environ.get("BAKE_ACTMEAN")
                        and os.path.isfile(os.environ["BAKE_ACTMEAN"]) else None
                    ),
                    "bake_actmean_sha256": (
                        _sha256(os.environ["BAKE_ACTMEAN"])
                        if os.environ.get("BAKE_ACTMEAN")
                        and os.path.isfile(os.environ["BAKE_ACTMEAN"]) else None
                    ),
                    "strand_f32_metric": os.environ.get("STRAND_F32_METRIC"),
                    "strand_f32_search": os.environ.get("STRAND_F32_SEARCH"),
                    "doctor_grad_accum": 4,
                    "doctor_kd_topk": 64,
                    "doctor_target_regex": None,
                    "evidence_files": {
                        str(path.relative_to(ROOT)): _sha256(path)
                        for path in evidence_paths
                    },
                }
                _atomic_json(paths["identity"], identity)
                coverage = {
                    "schema": "hawking.studio_core_coverage.v1",
                    "status": "pass",
                    "generated_at": _now(),
                    "model": LABEL,
                    "lane": lane,
                    "audit_jsonl": str(paths["audit"]),
                    "audit_sha256": _sha256(paths["audit"]),
                    "audit_identity": str(paths["identity"]),
                    "audit_identity_sha256": _sha256(paths["identity"]),
                    "identity_errors": [],
                    "required_configs": required,
                    "configs": {name: {"status": "pass"} for name in required},
                    "missing_configs": [],
                    "error_configs": [],
                    "invalid_configs": [],
                    "parse_errors": [],
                }
                _atomic_json(paths["coverage"], coverage)
                audit_relative = str(paths["audit"].relative_to(ROOT))
                if lane == "subbit":
                    receipt = ROOT / "receipts/official/14B-subbit-campaign.json"
                    receipt_doc = {
                        "schema": "hawking.subbit_campaign_complete.v1",
                        "project": "hawking",
                        "status": "research-complete",
                        "model": LABEL,
                        "lane": "subbit",
                        "artifact_class": "reconstruction_oracle",
                        "deployable": False,
                        "product_gate": False,
                        "floor_bpw": None,
                        "required_configs": required,
                        "audit_jsonl": audit_relative,
                        "audit_sha256": _sha256(paths["audit"]),
                        "coverage": str(paths["coverage"].relative_to(ROOT)),
                        "coverage_sha256": _sha256(paths["coverage"]),
                        "promotion_blockers": ["no deployable packed VTQ runtime"],
                    }
                    result_kind = "research-campaign"
                    floor_binding = {"floor_jsonl": None}
                else:
                    floor_curve = ROOT / "reports/cron/bit_floor_curve.jsonl"
                    floor_row = {
                        "schema": FLOOR_POINT_SCHEMA,
                        "model": LABEL,
                        "params_b": 14.0,
                        "floor_bpw": 4.0,
                        "winning_config": "4-AWQ",
                        "degr_pct": 1.0,
                        "gate_pct": 2.0,
                        "audit_jsonl": str(paths["audit"].resolve()),
                        "audit_sha256": _sha256(paths["audit"]),
                    }
                    locked_upsert_floor_row(floor_curve, LABEL, floor_row)
                    floor_binding = create_floor_binding(
                        ROOT, "studio", LABEL, floor_curve, paths["audit"],
                    )
                    receipt = ROOT / "receipts/official/14B-studio-floor.json"
                    receipt_doc = {
                        "project": "hawking",
                        "receipt_version": "0.2",
                        "source_model": "Qwen 14B synthetic selftest",
                        "condensed_artifact": f"4-AWQ @ 4.0 eff-bpw ({audit_relative})",
                        "effective_bpw": 4.0,
                        "claim_type": "scale-point",
                        "quality_gate": "pass",
                        "floor_point_sha256": floor_binding["floor_row_sha256"],
                    }
                    result_kind = "deployable-floor-experiment"
                _atomic_json(receipt, receipt_doc)
                complete = {
                    "schema": "hawking.studio_model_complete.v1",
                    "status": "pass",
                    "model": LABEL,
                    "lane": lane,
                    "audit_set": lane,
                    "required_configs": required,
                    "audit_jsonl": str(paths["audit"]),
                    "coverage": str(paths["coverage"]),
                    "coverage_sha256": _sha256(paths["coverage"]),
                    "receipt": str(receipt),
                    "receipt_sha256": _sha256(receipt),
                    "result_kind": result_kind,
                    **floor_binding,
                }
                _atomic_json(paths["complete"], complete)
                return paths, identity, coverage, receipt, receipt_doc, complete

            studio = build_lane_fixture("studio")
            studio_paths, studio_identity, studio_coverage, _, _, studio_complete = studio
            assert _lane_validation("studio")["ok"]

            # A receipt cannot redefine the lane to a cheaper subset.
            shrunk = json.loads(json.dumps(studio_coverage))
            shrunk["required_configs"] = shrunk["required_configs"][:-1]
            _atomic_json(studio_paths["coverage"], shrunk)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_paths["coverage"], studio_coverage)

            # Identity diagnostics and byte-level identity drift both fail closed.
            identity_error = json.loads(json.dumps(studio_coverage))
            identity_error["identity_errors"] = ["synthetic stale source"]
            _atomic_json(studio_paths["coverage"], identity_error)
            assert not _lane_validation("studio")["ok"]
            identity_field_missing = json.loads(json.dumps(studio_coverage))
            identity_field_missing.pop("identity_errors")
            _atomic_json(studio_paths["coverage"], identity_field_missing)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_paths["coverage"], studio_coverage)
            wrong_identity = dict(studio_identity)
            wrong_identity["lane"] = "subbit"
            _atomic_json(studio_paths["identity"], wrong_identity)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_paths["identity"], studio_identity)
            alternate_identity = studio_paths["identity"].with_name("alternate.identity.json")
            _atomic_json(alternate_identity, studio_identity)
            wrong_identity_path = json.loads(json.dumps(studio_coverage))
            wrong_identity_path["audit_identity"] = str(alternate_identity)
            wrong_identity_path["audit_identity_sha256"] = _sha256(alternate_identity)
            _atomic_json(studio_paths["coverage"], wrong_identity_path)
            path_validation = _lane_validation("studio")
            assert not path_validation["ok"]
            assert any("expected audit identity path" in row
                       for row in path_validation["blockers"])
            _atomic_json(studio_paths["coverage"], studio_coverage)

            def publish_studio_identity(identity_doc):
                """Republish all outer hashes so a negative test reaches semantic validation."""
                _atomic_json(studio_paths["identity"], identity_doc)
                coverage_doc = json.loads(json.dumps(studio_coverage))
                coverage_doc["audit_identity_sha256"] = _sha256(studio_paths["identity"])
                _atomic_json(studio_paths["coverage"], coverage_doc)
                complete_doc = json.loads(json.dumps(studio_complete))
                complete_doc["coverage_sha256"] = _sha256(studio_paths["coverage"])
                _atomic_json(studio_paths["complete"], complete_doc)
                return coverage_doc, complete_doc

            # Mutating the staged parent after an audit invalidates the receipt even when all outer
            # identity/coverage/completion hashes still agree with their historical bytes.
            model_config = model_dir / "config.json"
            model_config_bytes = model_config.read_bytes()
            model_config.write_bytes(model_config_bytes + b" ")
            stale_model = _lane_validation("studio")
            assert not stale_model["ok"]
            assert any("model_fingerprint is stale" in row for row in stale_model["blockers"])
            model_config.write_bytes(model_config_bytes)
            studio_identity = json.loads(json.dumps(studio_identity))
            studio_identity["model_fingerprint"] = _model_stat_fingerprint_for_dir(model_dir)
            studio_coverage, studio_complete = publish_studio_identity(studio_identity)
            assert _lane_validation("studio")["ok"]

            # The evaluator bytes are part of the recipe, not an advisory pathname.
            eval_bytes = eval_text.read_bytes()
            eval_text.write_bytes(eval_bytes + b"changed\n")
            stale_eval = _lane_validation("studio")
            assert not stale_eval["ok"]
            assert any("eval_text_sha256 is stale" in row for row in stale_eval["blockers"])
            eval_text.write_bytes(eval_bytes)
            assert _lane_validation("studio")["ok"]

            # Every recorded file is re-hashed, and adapter_contract is independently mandatory.
            adapter_contract = ROOT / "tools/condense/adapter_contract.py"
            adapter_bytes = adapter_contract.read_bytes()
            adapter_contract.write_bytes(adapter_bytes + b"changed\n")
            stale_evidence = _lane_validation("studio")
            assert not stale_evidence["ok"]
            assert any(
                "evidence hash is stale" in row and "adapter_contract" in row
                for row in stale_evidence["blockers"]
            )
            adapter_contract.write_bytes(adapter_bytes)
            assert _lane_validation("studio")["ok"]
            missing_adapter = json.loads(json.dumps(studio_identity))
            missing_adapter["evidence_files"].pop("tools/condense/adapter_contract.py")
            publish_studio_identity(missing_adapter)
            missing_evidence = _lane_validation("studio")
            assert not missing_evidence["ok"]
            assert any(
                "omits required evidence" in row and "adapter_contract" in row
                for row in missing_evidence["blockers"]
            )
            studio_coverage, studio_complete = publish_studio_identity(studio_identity)
            assert _lane_validation("studio")["ok"]

            # Republish each stale environment variant with valid outer hashes; the strict gate
            # must reject the recipe itself, including all Doctor controls.
            stale_environment = (
                ("device", "mps"),
                ("dtype", "torch.float32"),
                ("multiwindow", 1),
                ("studio_tripwire", False),
                ("bake_quality", not studio_identity.get("bake_quality")),
                ("bake_actmean_path", "/stale/actmean.json"),
                ("bake_actmean_sha256", "0" * 64),
                ("strand_f32_metric", "stale-selftest-value"),
                ("strand_f32_search", "stale-selftest-value"),
                ("doctor_grad_accum", 1),
                ("doctor_kd_topk", 32),
                ("doctor_target_regex", "q_proj"),
            )
            for field, stale_value in stale_environment:
                variant = json.loads(json.dumps(studio_identity))
                variant[field] = stale_value
                publish_studio_identity(variant)
                validation = _lane_validation("studio")
                assert not validation["ok"]
                assert any(field in row for row in validation["blockers"])
            studio_coverage, studio_complete = publish_studio_identity(studio_identity)
            assert _lane_validation("studio")["ok"]

            # Completion must independently repeat the pinned recipe.
            wrong_complete = dict(studio_complete)
            wrong_complete["required_configs"] = ["f16"]
            _atomic_json(studio_paths["complete"], wrong_complete)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_paths["complete"], studio_complete)
            assert _lane_validation("studio")["ok"]

            # Exact floor proof/hash and canonical curve-row bindings fail closed independently.
            floor_proof = ROOT / studio_complete["floor_jsonl"]
            proof_bytes = floor_proof.read_bytes()
            floor_proof.write_bytes(proof_bytes + b" ")
            assert not _lane_validation("studio")["ok"]
            floor_proof.write_bytes(proof_bytes)
            wrong_floor_hash = json.loads(json.dumps(studio_complete))
            wrong_floor_hash["floor_row_sha256"] = "0" * 64
            _atomic_json(studio_paths["complete"], wrong_floor_hash)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_paths["complete"], studio_complete)
            assert _lane_validation("studio")["ok"]
            studio_receipt = pathlib.Path(studio_complete["receipt"])
            studio_receipt_doc = _read_json(studio_receipt, {})
            wrong_receipt_floor = dict(studio_receipt_doc)
            wrong_receipt_floor["effective_bpw"] = 3.0
            _atomic_json(studio_receipt, wrong_receipt_floor)
            wrong_receipt_completion = dict(studio_complete)
            wrong_receipt_completion["receipt_sha256"] = _sha256(studio_receipt)
            _atomic_json(studio_paths["complete"], wrong_receipt_completion)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_receipt, studio_receipt_doc)
            _atomic_json(studio_paths["complete"], studio_complete)
            assert _lane_validation("studio")["ok"]
            wrong_receipt_hash = dict(studio_receipt_doc)
            wrong_receipt_hash["floor_point_sha256"] = "0" * 64
            _atomic_json(studio_receipt, wrong_receipt_hash)
            wrong_hash_completion = dict(studio_complete)
            wrong_hash_completion["receipt_sha256"] = _sha256(studio_receipt)
            _atomic_json(studio_paths["complete"], wrong_hash_completion)
            assert not _lane_validation("studio")["ok"]
            _atomic_json(studio_receipt, studio_receipt_doc)
            _atomic_json(studio_paths["complete"], studio_complete)
            assert _lane_validation("studio")["ok"]

            subbit = build_lane_fixture("subbit")
            subbit_paths, _, _, subbit_receipt, subbit_doc, subbit_complete = subbit
            assert _lane_validation("subbit")["ok"]
            false_product_claim = dict(subbit_doc)
            false_product_claim["deployable"] = True
            _atomic_json(subbit_receipt, false_product_claim)
            mutated_complete = dict(subbit_complete)
            mutated_complete["receipt_sha256"] = _sha256(subbit_receipt)
            _atomic_json(subbit_paths["complete"], mutated_complete)
            assert not _lane_validation("subbit")["ok"]
    finally:
        ROOT = original_root
        for key, value in original_identity_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("processing_queue.py selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "start":
        raise SystemExit(start_queue())
    if command == "run":
        raise SystemExit(run_queue())
    if command == "status":
        raise SystemExit(status())
    if command == "stop":
        raise SystemExit(stop_queue())
    if command == "--selftest":
        raise SystemExit(selftest())
    print("usage: processing_queue.py start|run|status|stop|--selftest", file=sys.stderr)
    raise SystemExit(2)
