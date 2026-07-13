#!/usr/bin/env python3.12
"""Detached, fail-closed representative-shard VTQ research queue.

This queue starts only after the strict 14B processing barrier.  It then waits
for path-bound verified 32B, 72B, and 120B download markers and runs one
``quantize-model --measure-only`` reconstruction oracle at a time.  It never
emits a packed model, PPL result, Doctor result, serving claim, or deployable
bit-floor point.

Each completed config is an independently durable checkpoint bound to the
download marker, representative source-shard SHA, quantizer SHA, exact command
and recipe, and exact metrics bytes.  A drain or resource-pressure stop simply
re-runs the current config; already validated receipts are skipped.

Usage:
  frontier_stream_queue.py start     # detached + caffeinated
  frontier_stream_queue.py run       # foreground supervisor
  frontier_stream_queue.py resume    # clear local drain and detach
  frontier_stream_queue.py status    # one JSON snapshot
  frontier_stream_queue.py drain     # request a restart-safe stop
  frontier_stream_queue.py selftest
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import math
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

import processing_queue
import procure
from ram_scheduler import resource_snapshot, thermal_output_ok
from studio_manifest import DEFAULT_HARDWARE

STATE = ROOT / "reports/condense/frontier_stream_queue_state.json"
PID_FILE = ROOT / "reports/condense/frontier_stream_queue.pid.json"
LOCK_FILE = ROOT / "reports/condense/frontier_stream_queue.lock"
LOG_FILE = ROOT / "reports/condense/frontier_stream_queue.log"
LOCAL_DRAIN = ROOT / "reports/condense/frontier_stream_queue.drain.request"
SHARED_DRAIN = ROOT / "reports/cron/studio_drain.request"
HEAVY_LOCK = ROOT / "reports/cron/studio_heavy.lock"
ARTIFACT_DIR = ROOT / "reports/condense/frontier_stream_queue"
BAKER = ROOT / "vendor/strand-quant/target/release/quantize-model"

POLL_S = max(1.0, float(os.environ.get("HAWKING_FRONTIER_STREAM_POLL_S", "15")))
RETRY_DELAY_S = max(
    1.0, float(os.environ.get("HAWKING_FRONTIER_STREAM_RETRY_DELAY_S", "30"))
)
MAX_FAILURES = max(
    1, int(os.environ.get("HAWKING_FRONTIER_STREAM_MAX_FAILURES", "3"))
)
THREADS = 1
DISK_WORKING_MARGIN_GB = 16.0
SUPPORTED_DTYPES = {"BF16", "F16", "F32"}
QUANTIZABLE_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
ACCOUNTING_SCOPE = (
    "logical_codec_stream_plus_required_lut_not_physical_packed_artifact"
)
ACCOUNTING_METHOD = (
    "exact encoder payload/trellis-side/OUTL bits + required per-tensor Q12 LUT bytes"
)

PLAN = (
    {
        "label": "32B", "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "local_dir": "scratch/staging/qwen-32b.partial",
    },
    {
        "label": "72B", "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "local_dir": "scratch/staging/qwen-72b.partial",
    },
    {
        "label": "120B", "hf_id": "openai/gpt-oss-120b",
        "local_dir": "scratch/staging/gpt-oss-120b.partial",
        "source_kind": "native MXFP4 original checkpoint",
    },
)


def _config(name, vec_dim, block_len=256, learned=False, control=False):
    return {
        "name": name,
        "bits": 1,
        "l_bits": 5,
        "vec_dim": int(vec_dim),
        "block_len": int(block_len),
        "learned_codebook": bool(learned),
        "block_amortization_control": bool(control),
    }


# k/d is below one for every primary point.  The three large-block points mirror
# the mandatory amortization controls in the Studio sub-bit ladder.
CONFIGS = (
    *(
        _config(f"vtq-k1-d{d}-b256-frozen", d)
        for d in (2, 3, 4, 8)
    ),
    *(
        _config(f"vtq-k1-d{d}-b256-learned", d, learned=True)
        for d in (2, 3, 4, 8)
    ),
    _config("vtq-k1-d2-b2048-frozen", 2, 2048, control=True),
    _config("vtq-k1-d4-b4096-frozen", 4, 4096, control=True),
    _config("vtq-k1-d8-b8192-frozen", 8, 8192, control=True),
)

_stop_requested = False
_sha_cache = {}


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


def _fsync_file(path):
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())
    _fsync_dir(pathlib.Path(path).parent)


def _atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {} if default is None else default


def _sha256(path):
    path = pathlib.Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    cached = _sha_cache.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if len(_sha_cache) >= 256:
        _sha_cache.clear()
    _sha_cache[key] = value
    return value


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rel(path):
    path = pathlib.Path(path)
    try:
        return str(path.resolve(strict=False).relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve(strict=False))


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _drain_requested():
    return _stop_requested or LOCAL_DRAIN.exists() or SHARED_DRAIN.exists()


def _base_state():
    return {
        "schema": "hawking.frontier_stream_queue.v1",
        "created_at": _now(),
        "updated_at": _now(),
        "status": "new",
        "plan": [dict(item) for item in PLAN],
        "configs": [dict(cfg) for cfg in CONFIGS],
        "items": {
            item["label"]: {
                "status": "pending",
                "configs": {
                    cfg["name"]: {"status": "pending", "attempts": 0, "failures": 0}
                    for cfg in CONFIGS
                },
            }
            for item in PLAN
        },
        "claim_limits": {
            "artifact_class": "reconstruction_oracle",
            "deployable": False,
            "product_gate": False,
            "ppl_measured": False,
            "doctor_run": False,
            "serving_quality_measured": False,
        },
    }


def _load_state():
    state = _read_json(STATE, _base_state())
    if state.get("schema") != "hawking.frontier_stream_queue.v1":
        state = _base_state()
    state["plan"] = [dict(item) for item in PLAN]
    state["configs"] = [dict(cfg) for cfg in CONFIGS]
    state.setdefault("items", {})
    for item in PLAN:
        row = state["items"].setdefault(item["label"], {"status": "pending"})
        row.setdefault("configs", {})
        for cfg in CONFIGS:
            row["configs"].setdefault(
                cfg["name"], {"status": "pending", "attempts": 0, "failures": 0}
            )
    return state


def _update_state(status=None, **updates):
    state = _load_state()
    if status is not None:
        state["status"] = status
    state.update(updates)
    state["updated_at"] = _now()
    _atomic_json(STATE, state)
    return state


def _update_item(label, **updates):
    state = _load_state()
    row = dict(state["items"][label])
    row.update(updates)
    row["updated_at"] = _now()
    state["items"][label] = row
    state["active_label"] = label
    state["updated_at"] = row["updated_at"]
    _atomic_json(STATE, state)
    return row


def _update_config(label, config, **updates):
    state = _load_state()
    item = dict(state["items"][label])
    configs = dict(item.get("configs", {}))
    row = dict(configs.get(config, {}))
    row.update(updates)
    row["updated_at"] = _now()
    configs[config] = row
    item["configs"] = configs
    item["active_config"] = config
    item["updated_at"] = row["updated_at"]
    state["items"][label] = item
    state["active_label"] = label
    state["active_config"] = config
    state["updated_at"] = row["updated_at"]
    _atomic_json(STATE, state)
    return row


def _sleep_interruptible(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _drain_requested():
            return False
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    return True


def _thermal_snapshot():
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        detail = (result.stdout + result.stderr).strip()
        return {
            "ok": thermal_output_ok(result.returncode, detail),
            "returncode": result.returncode,
            "detail": detail[-1000:],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _evaluate_safety(snapshot, thermal):
    blockers = []
    if not isinstance(snapshot, dict) or snapshot.get("ok") is not True:
        blockers.append("resource snapshot unavailable")
    if snapshot.get("pressure_level") != 1:
        blockers.append(
            f"memory pressure is not normal (level={snapshot.get('pressure_level')})"
        )
    swap = snapshot.get("swap_used_mb")
    if not isinstance(swap, (int, float)) or not math.isfinite(float(swap)):
        blockers.append("swap measurement unavailable")
    elif float(swap) > 0.001:
        blockers.append(f"swap must be zero (observed {float(swap):.3f}MB)")
    disk = snapshot.get("disk_free_gb")
    required_disk = DEFAULT_HARDWARE.disk_reserve_gb + DISK_WORKING_MARGIN_GB
    if not isinstance(disk, (int, float)) or float(disk) < required_disk:
        blockers.append(
            f"disk free {disk!r}GB < {required_disk:.1f}GB reserve+working margin"
        )
    if "AC Power" not in str(snapshot.get("power_source", "")):
        blockers.append("AC power not confirmed")
    if not isinstance(thermal, dict) or thermal.get("ok") is not True:
        blockers.append("thermal/performance state is not green")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "required_disk_free_gb": required_disk,
        "resources": snapshot,
        "thermal": thermal,
    }


def _safety():
    return _evaluate_safety(resource_snapshot(ROOT), _thermal_snapshot())


def _active_download_work():
    try:
        return processing_queue._active_download_work()
    except Exception as exc:
        # Detection failure is itself activity: fail closed instead of overlapping
        # a possibly-live Hugging Face child.
        return [{"role": "download-detection-error", "error": f"{type(exc).__name__}: {exc}"}]


def _processing_barrier():
    try:
        return processing_queue.completion_status()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _marker_status(item):
    _, marker_path = procure._checkpoint_paths(item["label"])
    marker_path = ROOT / marker_path
    marker = _read_json(marker_path, {})
    valid = procure._verified_marker_valid(
        marker,
        label=item["label"],
        hf_id=item["hf_id"],
        local_dir=item["local_dir"],
        require_verify=True,
    )
    source = ROOT / item["local_dir"]
    blockers = []
    if not valid:
        blockers.append("verified marker is missing or not bound to label/HF id/staged path")
    if not source.is_dir():
        blockers.append(f"staged source is absent: {item['local_dir']}")
    return {
        "ok": valid and not blockers,
        "path": _rel(marker_path),
        "sha256": _sha256(marker_path) if valid and marker_path.is_file() else None,
        "marker": marker if valid else None,
        "blockers": blockers,
    }


def _read_safetensor_header(path):
    path = pathlib.Path(path)
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("truncated safetensors length")
        header_len = int.from_bytes(raw_len, "little")
        if header_len <= 1 or header_len > min(path.stat().st_size - 8, 256 * 1024 * 1024):
            raise ValueError(f"invalid safetensors header length {header_len}")
        header = json.loads(handle.read(header_len))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    tensors = []
    for name, desc in header.items():
        if name == "__metadata__" or not isinstance(desc, dict):
            continue
        shape = desc.get("shape")
        dtype = desc.get("dtype")
        offsets = desc.get("data_offsets")
        if not (
            isinstance(shape, list) and all(isinstance(v, int) and v >= 0 for v in shape)
            and isinstance(dtype, str)
            and isinstance(offsets, list) and len(offsets) == 2
            and all(isinstance(v, int) and v >= 0 for v in offsets)
            and offsets[0] <= offsets[1]
        ):
            raise ValueError(f"invalid tensor descriptor: {name}")
        tensors.append({"name": name, "shape": shape, "dtype": dtype})
    if not tensors:
        raise ValueError("no tensor descriptors")
    quantizable = [
        row for row in tensors
        if len(row["shape"]) == 2
        and any(row["name"].endswith(suffix) for suffix in QUANTIZABLE_SUFFIXES)
    ]
    supported = [row for row in quantizable if row["dtype"] in SUPPORTED_DTYPES]
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(tensors),
        "dtypes": sorted({row["dtype"] for row in tensors}),
        "quantizable_tensor_count": len(quantizable),
        "supported_quantizable_tensor_count": len(supported),
        "unsupported_quantizable_tensor_count": len(quantizable) - len(supported),
    }


def _representative_shard(item):
    source = ROOT / item["local_dir"]
    candidates = sorted(source.rglob("*.safetensors")) if source.is_dir() else []
    inspected = []
    errors = []
    for path in candidates:
        try:
            inspected.append(_read_safetensor_header(path))
        except Exception as exc:
            errors.append({"path": _rel(path), "error": f"{type(exc).__name__}: {exc}"})
    readable = [
        row for row in inspected
        if row["supported_quantizable_tensor_count"] > 0
        and row["unsupported_quantizable_tensor_count"] == 0
    ]
    if not readable:
        return {
            "ok": False,
            "blocker": "no directly readable BF16/F16/F32 safetensors shard with supported projection tensors",
            "candidate_count": len(candidates),
            "inspected": inspected[:64],
            "errors": errors[:64],
        }
    # Maximum supported projection coverage, then smaller bytes, then lexical path.
    selected = sorted(
        readable,
        key=lambda row: (
            -row["supported_quantizable_tensor_count"], row["bytes"], row["path"]
        ),
    )[0]
    selected = dict(selected)
    selected["selection_method"] = (
        "max_supported_projection_count_then_smallest_shard_then_lexical_path"
    )
    selected["sha256"] = _sha256(ROOT / selected["path"])
    selected["ok"] = True
    return selected


def _format_blocker_path(label):
    return ARTIFACT_DIR / label / "source_format_blocker.json"


def _publish_format_blocker(item, marker, source):
    path = _format_blocker_path(item["label"])
    prior = _read_json(path, {})
    doc = {
        "schema": "hawking.frontier_source_format_blocker.v1",
        "status": "waiting-architecture-format-support",
        "success": False,
        "deployable": False,
        "label": item["label"],
        "hf_id": item["hf_id"],
        "source_kind": item.get("source_kind"),
        "first_observed_at": prior.get("first_observed_at") or _now(),
        "observed_at": _now(),
        "download_marker": marker.get("path"),
        "download_marker_sha256": marker.get("sha256"),
        "blocker": source.get("blocker"),
        "required_input": (
            "a quantize-model-readable safetensors shard containing BF16/F16/F32 "
            "projection tensors, or an explicitly reviewed MXFP4 decode adapter"
        ),
        "inspection": {
            "candidate_count": source.get("candidate_count"),
            "inspected": source.get("inspected", []),
            "errors": source.get("errors", []),
        },
    }
    _atomic_json(path, doc)
    return {"path": _rel(path), "sha256": _sha256(path), "document": doc}


def _paths(label, config):
    base = ARTIFACT_DIR / label / config
    return {
        "output_prefix": pathlib.Path(f"{base}.metrics"),
        "metrics": pathlib.Path(f"{base}.metrics.json"),
        "receipt": pathlib.Path(f"{base}.receipt.json"),
        "log": pathlib.Path(f"{base}.log"),
        "completion": ARTIFACT_DIR / label / "campaign.complete.json",
    }


def _recipe(config):
    return {
        "schema": "hawking.frontier_vtq_probe_recipe.v1",
        "bits": config["bits"],
        "l_bits": config["l_bits"],
        "nominal_payload_bpw": config["bits"] / config["vec_dim"],
        "vec_dim": config["vec_dim"],
        "block_len": config["block_len"],
        "learned_codebook": config["learned_codebook"],
        "learned_codebook_iters": 50,
        "learned_codebook_max_vectors": 16384,
        "rht": True,
        "rht_axis": "cols",
        "tail_biting": False,
        "affine_min": False,
        "affine_min_cli": "auto",
        "outlier_pct": 0.0,
        "outlier_bits": 8,
        "quality": False,
        "actmean": False,
        "measure_only": True,
        "encode_workers": THREADS,
        "environment": {
            "STRAND_NO_GPU": "1",
            "STRAND_F32_METRIC": "0",
            "STRAND_F32_SEARCH": "0",
        },
        "artifact_class": "reconstruction_oracle",
        "deployable": False,
        "product_gate": False,
    }


def _command(config, shard, output_prefix):
    command = [
        str(BAKER),
        "--input", str(shard),
        "--output", str(output_prefix),
        "--bits", str(config["bits"]),
        "--l", str(config["l_bits"]),
        "--no-tail-biting",
        "--affine-min", "auto",
        "--rht-cols",
        "--threads", str(THREADS),
        "--measure-only",
        "--vec-dim", str(config["vec_dim"]),
        "--block-len", str(config["block_len"]),
        "--outlier-channel", "0",
        "--outlier-bits", "8",
    ]
    if config["learned_codebook"]:
        command.append("--learned-codebook")
    return command


def _finite(value, *, positive=False):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (not positive or float(value) > 0)
    )


def _metrics_validation(metrics, config):
    blockers = []
    if not isinstance(metrics, dict):
        return {"ok": False, "blockers": ["metrics document is not an object"]}
    encoder = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    expected_encoder = {
        "bits": config["bits"],
        "l": config["l_bits"],
        "k": config["bits"],
        "rht": True,
        "rht_axis": "cols",
        "tail_biting": False,
        "affine_min": False,
        "calibrated": False,
        "block_hessian": False,
        "mixed_precision": False,
        "vec_dim": config["vec_dim"],
        "block_len": config["block_len"],
        "learned_codebook": config["learned_codebook"],
        "learned_codebook_iters": 50,
        "learned_codebook_max_vectors": 16384,
        "encode_workers": THREADS,
        "artifact_class": "reconstruction_oracle",
        "deployable": False,
    }
    if encoder != expected_encoder:
        blockers.append("encoder config does not exactly match the pinned recipe")
    tensors = metrics.get("tensors") if isinstance(metrics.get("tensors"), list) else []
    if not tensors:
        blockers.append("metrics contain no quantized tensors")
    keys = (
        "payload_bits", "trellis_side_bits", "outlier_side_bits", "required_lut_bytes"
    )
    sums = {key: 0 for key in keys}
    qcount = 0
    selected_tensors = 0
    selected_weights = 0
    vector_tensors = 0
    vector_weights = 0
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            blockers.append(f"tensor row {index} is not an object")
            continue
        n = tensor.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            blockers.append(f"tensor row {index} has invalid n")
            continue
        qcount += n
        if tensor.get("bits") != config["bits"]:
            blockers.append(f"tensor row {index} has wrong bit depth")
        if tensor.get("billing_complete") is not True \
                or tensor.get("billing_scope") != ACCOUNTING_SCOPE:
            blockers.append(f"tensor row {index} has incomplete/wrong accounting scope")
        for key in keys:
            value = tensor.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                blockers.append(f"tensor row {index} has invalid {key}")
            else:
                sums[key] += value
        vector_required = tensor.get("vector_lut_required") is True
        selected = tensor.get("learned_lut_selected") is True
        if vector_required:
            vector_tensors += 1
            vector_weights += n
            expected_lut = 52 + (1 << config["l_bits"]) * config["vec_dim"] * 4
            if tensor.get("required_lut_bytes") != expected_lut:
                blockers.append(f"tensor row {index} vector LUT bytes are not exact SDSC v2 cost")
        elif tensor.get("required_lut_bytes") != 0:
            blockers.append(f"tensor row {index} bills a vector LUT for a scalar tensor")
        if selected:
            selected_tensors += 1
            selected_weights += n
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else {}
    if aggregate.get("quantized_weights") != qcount or qcount <= 0:
        blockers.append("aggregate quantized-weight count does not equal tensor rows")
    for key, value in sums.items():
        if aggregate.get(key) != value:
            blockers.append(f"aggregate {key} does not equal tensor rows")
    if aggregate.get("outlier_side_bits") != 0:
        blockers.append("outliers must remain disabled for this sub-bit probe")
    if aggregate.get("learned_lut_selected_tensors") != selected_tensors \
            or aggregate.get("learned_lut_selected_weights") != selected_weights:
        blockers.append("aggregate learned-LUT selection counts do not equal tensor rows")
    if aggregate.get("vector_lut_required_tensors") != vector_tensors \
            or aggregate.get("vector_lut_required_weights") != vector_weights:
        blockers.append("aggregate required-vector-LUT counts do not equal tensor rows")
    if config["learned_codebook"]:
        if selected_tensors <= 0:
            blockers.append("learned recipe selected no learned LUT; cannot label this learned")
    elif selected_tensors != 0:
        blockers.append("frozen recipe contains learned-LUT selection")
    if vector_tensors <= 0 or vector_weights != qcount:
        blockers.append("vector recipe does not bill a required LUT for every quantized tensor")
    if aggregate.get("billing_complete") is not True \
            or aggregate.get("billing_scope") != ACCOUNTING_SCOPE:
        blockers.append("aggregate logical accounting is incomplete or wrong-scope")
    if aggregate.get("artifact_class") != "reconstruction_oracle" \
            or aggregate.get("deployable") is not False:
        blockers.append("metrics are not explicitly deployable=false reconstruction oracle")
    logical_bits = sum(sums[key] for key in keys[:-1]) + sums["required_lut_bytes"] * 8
    exact_bpw = logical_bits / qcount if qcount else float("nan")
    if not _finite(aggregate.get("oracle_effective_bpw"), positive=True) \
            or abs(float(aggregate.get("oracle_effective_bpw", 0)) - exact_bpw) > 1.1e-6:
        blockers.append("oracle_effective_bpw does not equal exact logical bits/weights")
    if not _finite(aggregate.get("weighted_rel_rms_pct")) \
            or float(aggregate.get("weighted_rel_rms_pct", -1)) < 0:
        blockers.append("weighted reconstruction RMS is absent or invalid")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "summary": {
            "quantized_weights": qcount,
            **sums,
            "logical_stream_bits_including_required_lut": logical_bits,
            "oracle_effective_bpw": exact_bpw if qcount else None,
            "weighted_rel_rms_pct": aggregate.get("weighted_rel_rms_pct"),
            "learned_lut_selected_tensors": selected_tensors,
            "learned_lut_selected_weights": selected_weights,
            "vector_lut_required_tensors": vector_tensors,
            "vector_lut_required_weights": vector_weights,
            "billing_scope": ACCOUNTING_SCOPE,
            "method": ACCOUNTING_METHOD,
        },
    }


def _attempt_identity(item, config, marker, shard, quantizer_sha, command, recipe):
    return _canonical_hash({
        "label": item["label"],
        "hf_id": item["hf_id"],
        "download_marker_sha256": marker["sha256"],
        "source_shard": shard["path"],
        "source_shard_sha256": shard["sha256"],
        "quantizer_sha256": quantizer_sha,
        "command": command,
        "recipe": recipe,
    })


def _receipt_validation(item, config, marker, shard, quantizer_sha):
    paths = _paths(item["label"], config["name"])
    command = _command(config, ROOT / shard["path"], paths["output_prefix"])
    recipe = _recipe(config)
    receipt = _read_json(paths["receipt"], {})
    blockers = []
    expected = {
        "schema": "hawking.frontier_vtq_probe_receipt.v1",
        "status": "pass",
        "artifact_class": "reconstruction_oracle",
        "deployable": False,
        "product_gate": False,
        "ppl_measured": False,
        "doctor_run": False,
        "serving_quality_measured": False,
        "label": item["label"],
        "hf_id": item["hf_id"],
        "config": config["name"],
        "download_marker": marker["path"],
        "download_marker_sha256": marker["sha256"],
        "source_shard": shard["path"],
        "source_shard_sha256": shard["sha256"],
        "source_shard_bytes": shard["bytes"],
        "quantizer": _rel(BAKER),
        "quantizer_sha256": quantizer_sha,
        "command": command,
        "command_sha256": _canonical_hash(command),
        "recipe": recipe,
        "recipe_sha256": _canonical_hash(recipe),
        "metrics": _rel(paths["metrics"]),
    }
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
        blockers.append("receipt identity/recipe/source/command binding mismatch")
    if not receipt.get("completed_at"):
        blockers.append("receipt completion timestamp is absent")
    current_marker = _marker_status(item)
    if not current_marker.get("ok") or current_marker.get("sha256") != marker.get("sha256"):
        blockers.append("download marker changed or no longer validates")
    try:
        if _sha256(ROOT / shard["path"]) != shard["sha256"]:
            blockers.append("source shard SHA changed")
    except OSError:
        blockers.append("source shard is absent")
    try:
        if _sha256(BAKER) != quantizer_sha:
            blockers.append("quantizer SHA changed")
    except OSError:
        blockers.append("quantizer is absent")
    if not paths["metrics"].is_file():
        blockers.append("metrics file is absent")
        metrics = {}
        metrics_sha = None
    else:
        metrics_sha = _sha256(paths["metrics"])
        metrics = _read_json(paths["metrics"], {})
    if receipt.get("metrics_sha256") != metrics_sha:
        blockers.append("receipt metrics SHA does not match current bytes")
    metrics_check = _metrics_validation(metrics, config)
    if not metrics_check["ok"]:
        blockers.extend(metrics_check["blockers"])
    if receipt.get("metrics_summary") != metrics_check.get("summary"):
        blockers.append("receipt metrics summary does not match exact metrics")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "receipt": _rel(paths["receipt"]),
        "receipt_sha256": _sha256(paths["receipt"]) if paths["receipt"].is_file() else None,
        "metrics": _rel(paths["metrics"]),
        "attempt_identity": _attempt_identity(
            item, config, marker, shard, quantizer_sha, command, recipe
        ),
    }


def _publish_receipt(item, config, marker, shard, quantizer_sha, command, recipe, safety):
    paths = _paths(item["label"], config["name"])
    _fsync_file(paths["metrics"])
    metrics = _read_json(paths["metrics"], {})
    checked = _metrics_validation(metrics, config)
    if not checked["ok"]:
        raise RuntimeError("invalid oracle metrics: " + "; ".join(checked["blockers"]))
    receipt = {
        "schema": "hawking.frontier_vtq_probe_receipt.v1",
        "status": "pass",
        "completed_at": _now(),
        "artifact_class": "reconstruction_oracle",
        "deployable": False,
        "product_gate": False,
        "ppl_measured": False,
        "doctor_run": False,
        "serving_quality_measured": False,
        "label": item["label"],
        "hf_id": item["hf_id"],
        "config": config["name"],
        "download_marker": marker["path"],
        "download_marker_sha256": marker["sha256"],
        "source_shard": shard["path"],
        "source_shard_sha256": shard["sha256"],
        "source_shard_bytes": shard["bytes"],
        "source_shard_selection": shard["selection_method"],
        "source_shard_quantizable_tensors": shard["supported_quantizable_tensor_count"],
        "quantizer": _rel(BAKER),
        "quantizer_sha256": quantizer_sha,
        "command": command,
        "command_sha256": _canonical_hash(command),
        "recipe": recipe,
        "recipe_sha256": _canonical_hash(recipe),
        "metrics": _rel(paths["metrics"]),
        "metrics_sha256": _sha256(paths["metrics"]),
        "metrics_summary": checked["summary"],
        "admission_safety": safety,
        "limitations": [
            "representative source shard only; not a full-model measurement",
            "reconstruction RMS only; no perplexity, task quality, or Doctor recovery was measured",
            "logical codec accounting is not a physical packed artifact byte count",
            "no packed VTQ artifact or serving runtime was produced",
            "cannot support model-fit, source-deletion, deployment, or product bit-floor claims",
        ],
    }
    _atomic_json(paths["receipt"], receipt)
    return receipt


def _try_heavy_lease():
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = open(HEAVY_LOCK, "a+")
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


def _terminate_child(proc, reason):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    _update_state(last_termination={"at": _now(), "pid": proc.pid, "reason": reason})
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _run_config(item, config, marker, shard, quantizer_sha, lease):
    # The child inherits the locked open-file description. If the detached supervisor is killed,
    # the quantizer keeps admission ownership until it exits, preventing a restarted queue from
    # launching a duplicate writer into the same config paths.
    paths = _paths(item["label"], config["name"])
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    command = _command(config, ROOT / shard["path"], paths["output_prefix"])
    recipe = _recipe(config)
    identity = _attempt_identity(item, config, marker, shard, quantizer_sha, command, recipe)
    state_row = _load_state()["items"][item["label"]]["configs"][config["name"]]
    if state_row.get("attempt_identity") != identity:
        state_row = _update_config(
            item["label"], config["name"], status="pending-new-identity",
            attempts=0, failures=0, attempt_identity=identity,
        )
    failures = int(state_row.get("failures", 0))
    if failures >= MAX_FAILURES:
        _update_config(
            item["label"], config["name"], status="blocked-retries",
            max_failures=MAX_FAILURES, attempt_identity=identity,
        )
        return 4
    safety = _safety()
    downloads = _active_download_work()
    if not safety["ok"] or downloads:
        _update_config(
            item["label"], config["name"],
            status="waiting-download" if downloads else "waiting-resources",
            safety=safety, download_activity=downloads, attempt_identity=identity,
        )
        return 75
    for stale in (paths["metrics"], paths["receipt"]):
        if stale.exists():
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            stale.rename(stale.with_name(f"{stale.name}.stale.{stamp}"))
            _fsync_dir(stale.parent)
    env = {
        **os.environ,
        "STRAND_NO_GPU": "1",
        "STRAND_F32_METRIC": "0",
        "STRAND_F32_SEARCH": "0",
    }
    log = open(paths["log"], "ab", buffering=0)
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        pass_fds=(lease.fileno(),),
    )
    log.close()
    attempts = int(state_row.get("attempts", 0)) + 1
    started = time.monotonic()
    _update_config(
        item["label"], config["name"], status="running", attempts=attempts,
        failures=failures, attempt_identity=identity, child_pid=proc.pid,
        command=command, command_sha256=_canonical_hash(command),
        recipe_sha256=_canonical_hash(recipe), started_at=_now(), safety=safety,
    )
    _update_state(
        status="running", child_pid=proc.pid, child_pgid=proc.pid,
        child_log=_rel(paths["log"]),
    )
    interrupted_reason = None
    while proc.poll() is None:
        safety = _safety()
        downloads = _active_download_work()
        _update_config(
            item["label"], config["name"], status="running", child_pid=proc.pid,
            elapsed_seconds=round(time.monotonic() - started, 1), safety=safety,
            download_activity=downloads,
        )
        if _drain_requested():
            interrupted_reason = "drain requested"
        elif downloads:
            interrupted_reason = "download controller/child became active"
        elif not safety["ok"]:
            interrupted_reason = "; ".join(safety["blockers"])
        if interrupted_reason:
            _terminate_child(proc, interrupted_reason)
            break
        _sleep_interruptible(POLL_S)
    rc = int(proc.returncode or 0)
    _update_state(child_pid=None, child_pgid=None)
    if interrupted_reason:
        _update_config(
            item["label"], config["name"], status="interrupted",
            child_pid=None, returncode=rc, interruption_reason=interrupted_reason,
        )
        return 130 if _drain_requested() else 75
    if rc != 0:
        failures += 1
        tail = ""
        try:
            with open(paths["log"], "rb") as handle:
                handle.seek(max(0, paths["log"].stat().st_size - 8192))
                tail = handle.read().decode(errors="replace")
        except OSError:
            pass
        _update_config(
            item["label"], config["name"],
            status="retryable-failure" if failures < MAX_FAILURES else "blocked-retries",
            child_pid=None, returncode=rc, failures=failures,
            max_failures=MAX_FAILURES, log_tail=tail,
        )
        return 4
    try:
        receipt = _publish_receipt(
            item, config, marker, shard, quantizer_sha, command, recipe, safety
        )
        checked = _receipt_validation(item, config, marker, shard, quantizer_sha)
        if not checked["ok"]:
            raise RuntimeError("post-commit receipt validation failed: " + "; ".join(checked["blockers"]))
    except Exception as exc:
        failures += 1
        _update_config(
            item["label"], config["name"],
            status="retryable-output-failure" if failures < MAX_FAILURES else "blocked-retries",
            child_pid=None, returncode=4, failures=failures,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 4
    _update_config(
        item["label"], config["name"], status="pass", child_pid=None,
        returncode=0, completed_at=receipt["completed_at"],
        receipt=checked["receipt"], receipt_sha256=checked["receipt_sha256"],
    )
    return 0


def _campaign_receipt(item, marker, shard, quantizer_sha):
    rows = []
    blockers = []
    for config in CONFIGS:
        checked = _receipt_validation(item, config, marker, shard, quantizer_sha)
        rows.append({
            "config": config["name"], "receipt": checked["receipt"],
            "receipt_sha256": checked["receipt_sha256"], "valid": checked["ok"],
        })
        if not checked["ok"]:
            blockers.append({"config": config["name"], "blockers": checked["blockers"]})
    if blockers:
        raise RuntimeError(f"campaign has invalid config receipts: {blockers}")
    path = _paths(item["label"], CONFIGS[0]["name"])["completion"]
    doc = {
        "schema": "hawking.frontier_vtq_shard_campaign.v1",
        "status": "research-complete",
        "completed_at": _now(),
        "label": item["label"],
        "hf_id": item["hf_id"],
        "artifact_class": "reconstruction_oracle",
        "scope": "representative-shard",
        "deployable": False,
        "product_gate": False,
        "floor_bpw": None,
        "ppl_measured": False,
        "doctor_run": False,
        "serving_quality_measured": False,
        "download_marker": marker["path"],
        "download_marker_sha256": marker["sha256"],
        "source_shard": shard["path"],
        "source_shard_sha256": shard["sha256"],
        "quantizer": _rel(BAKER),
        "quantizer_sha256": quantizer_sha,
        "required_configs": [cfg["name"] for cfg in CONFIGS],
        "config_receipts": rows,
        "promotion_blockers": [
            "representative shard is not full-model evidence",
            "no PPL/task/Doctor quality gate",
            "quantize-model vector path is a dense reconstruction oracle, not packed VTQ serving",
        ],
    }
    _atomic_json(path, doc)
    return {"path": _rel(path), "sha256": _sha256(path), "document": doc}


def _wait_for_probe_admission(label, config):
    while not _drain_requested():
        safety = _safety()
        downloads = _active_download_work()
        barrier = _processing_barrier()
        if safety["ok"] and not downloads and barrier.get("ok"):
            lease = _try_heavy_lease()
            if lease is not None:
                # Re-sample after acquisition.  No lease is ever held while merely
                # waiting for a marker, download, pressure recovery, or another owner.
                safety = _safety()
                downloads = _active_download_work()
                barrier = _processing_barrier()
                if safety["ok"] and not downloads and barrier.get("ok"):
                    return lease
                _release_heavy_lease(lease)
        status = (
            "waiting-14b-processing" if not barrier.get("ok") else
            "waiting-download" if downloads else
            "waiting-resources" if not safety["ok"] else
            "waiting-heavy-lease"
        )
        _update_state(
            status=status, active_label=label, active_config=config,
            safety=safety, download_activity=downloads, heavy_lock=_rel(HEAVY_LOCK),
            processing_barrier=barrier,
        )
        if not _sleep_interruptible(POLL_S):
            break
    return None


def run_queue():
    global _stop_requested
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    singleton = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[frontier-stream] another supervisor holds the singleton lock", file=sys.stderr)
        singleton.close()
        return 2

    def request_stop(_sig, _frame):
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _atomic_json(PID_FILE, {
        "schema": "hawking.frontier_stream_queue_pid.v1",
        "pid": os.getpid(), "started_at": _now(), "log": _rel(LOG_FILE),
    })
    rc = 0
    try:
        while not _drain_requested():
            barrier = _processing_barrier()
            if barrier.get("ok"):
                break
            _update_state(status="waiting-14b-processing", processing_barrier=barrier)
            if not _sleep_interruptible(POLL_S):
                break
        if _drain_requested():
            _update_state(status="paused-drain", paused_at=_now())
            return 130
        if not BAKER.is_file() or not os.access(BAKER, os.X_OK):
            _update_state(
                status="blocked-quantizer",
                blocker=f"release quantizer is absent or not executable: {_rel(BAKER)}",
            )
            return 1

        for item in PLAN:
            label = item["label"]
            marker = None
            while not _drain_requested():
                marker = _marker_status(item)
                downloads = _active_download_work()
                if marker["ok"] and not downloads:
                    break
                status = "waiting-download" if downloads else "waiting-download-marker"
                _update_item(label, status=status, download_marker=marker,
                             download_activity=downloads)
                _update_state(status=status, active_label=label,
                              download_activity=downloads)
                if not _sleep_interruptible(POLL_S):
                    break
            if _drain_requested():
                rc = 130
                break
            source = _representative_shard(item)
            if not source.get("ok"):
                blocker = _publish_format_blocker(item, marker, source)
                _update_item(
                    label, status="waiting-architecture-format-support",
                    download_marker=marker, source_format_blocker=blocker,
                )
                _update_state(
                    status="waiting-architecture-format-support", active_label=label,
                    blocker=blocker,
                )
                # Never convert this architecture/format blocker into success.
                while not _drain_requested():
                    if not _sleep_interruptible(POLL_S):
                        break
                    source = _representative_shard(item)
                    if source.get("ok"):
                        break
                    marker = _marker_status(item)
                    blocker = _publish_format_blocker(item, marker, source)
                    _update_item(
                        label, status="waiting-architecture-format-support",
                        download_marker=marker, source_format_blocker=blocker,
                    )
                if _drain_requested():
                    rc = 130
                    break
                if not source.get("ok"):
                    rc = 75
                    break
            quantizer_sha = _sha256(BAKER)
            _update_item(
                label, status="validating-receipts", download_marker=marker,
                representative_shard=source, quantizer_sha256=quantizer_sha,
            )
            for config in CONFIGS:
                checked = _receipt_validation(item, config, marker, source, quantizer_sha)
                if checked["ok"]:
                    _update_config(
                        label, config["name"], status="pass", receipt=checked["receipt"],
                        receipt_sha256=checked["receipt_sha256"],
                        attempt_identity=checked["attempt_identity"],
                    )
                    continue
                while not _drain_requested():
                    lease = _wait_for_probe_admission(label, config["name"])
                    if lease is None:
                        rc = 130
                        break
                    try:
                        run_rc = _run_config(
                            item, config, marker, source, quantizer_sha, lease
                        )
                    finally:
                        _release_heavy_lease(lease)
                    if run_rc == 0:
                        break
                    if run_rc in (75, 130):
                        if run_rc == 130:
                            rc = 130
                            break
                        continue
                    row = _load_state()["items"][label]["configs"][config["name"]]
                    if int(row.get("failures", 0)) >= MAX_FAILURES:
                        rc = 4
                        break
                    _update_config(
                        label, config["name"], status="waiting-retry",
                        retry_after_seconds=RETRY_DELAY_S,
                    )
                    if not _sleep_interruptible(RETRY_DELAY_S):
                        rc = 130
                        break
                if rc != 0:
                    break
            if rc != 0:
                break
            campaign = _campaign_receipt(item, marker, source, quantizer_sha)
            _update_item(
                label, status="research-complete", active_config=None,
                completion=campaign, completed_at=_now(),
            )
        if rc == 0:
            _update_state(
                status="complete", active_label=None, active_config=None,
                completed_at=_now(),
                terminal_reason="all representative-shard VTQ research receipts validated",
            )
        elif rc == 130:
            _update_state(status="paused-drain", child_pid=None, paused_at=_now())
        else:
            _update_state(status="blocked", child_pid=None, returncode=rc, blocked_at=_now())
        return rc
    finally:
        info = _read_json(PID_FILE, {})
        if info.get("pid") == os.getpid():
            try:
                PID_FILE.unlink()
                _fsync_dir(PID_FILE.parent)
            except FileNotFoundError:
                pass
        fcntl.flock(singleton.fileno(), fcntl.LOCK_UN)
        singleton.close()


def start_queue():
    if LOCAL_DRAIN.exists() or SHARED_DRAIN.exists():
        print("[frontier-stream] drain is active; use resume after the machine is ready", file=sys.stderr)
        return 130
    info = _read_json(PID_FILE, {})
    if _pid_alive(info.get("pid")):
        print(f"[frontier-stream] already active pid={info['pid']}", file=sys.stderr)
        return 0
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab", buffering=0)
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "run"]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    proc = subprocess.Popen(
        command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    log.close()
    _atomic_json(PID_FILE, {
        "schema": "hawking.frontier_stream_queue_pid.v1",
        "pid": proc.pid, "started_at": _now(), "log": _rel(LOG_FILE), "command": command,
    })
    print(f"[frontier-stream] detached pid={proc.pid}; log={LOG_FILE}", file=sys.stderr)
    return 0


def resume_queue():
    if LOCAL_DRAIN.exists():
        LOCAL_DRAIN.unlink()
        _fsync_dir(LOCAL_DRAIN.parent)
    if SHARED_DRAIN.exists():
        print(f"[frontier-stream] shared Studio drain remains active: {SHARED_DRAIN}", file=sys.stderr)
        return 130
    return start_queue()


def drain_queue():
    _atomic_json(LOCAL_DRAIN, {
        "schema": "hawking.frontier_stream_queue_drain.v1",
        "requested_at": _now(), "reason": "operator requested restart-safe drain",
    })
    info = _read_json(PID_FILE, {})
    if _pid_alive(info.get("pid")):
        try:
            os.killpg(int(info["pid"]), signal.SIGTERM)
        except OSError:
            try:
                os.kill(int(info["pid"]), signal.SIGTERM)
            except OSError:
                pass
        print(f"[frontier-stream] drain requested for pid={info['pid']}", file=sys.stderr)
    else:
        print("[frontier-stream] drain recorded; queue is not active", file=sys.stderr)
    return 0


def status():
    info = _read_json(PID_FILE, {})
    marker_rows = {item["label"]: _marker_status(item) for item in PLAN}
    payload = {
        "schema": "hawking.frontier_stream_queue_status.v1",
        "generated_at": _now(),
        "active": _pid_alive(info.get("pid")),
        "pid": info.get("pid"),
        "state": _load_state(),
        "processing_barrier": _processing_barrier(),
        "download_markers": marker_rows,
        "download_activity": _active_download_work(),
        "safety": _safety(),
        "local_drain_requested": LOCAL_DRAIN.exists(),
        "shared_drain_requested": SHARED_DRAIN.exists(),
        "heavy_lock": _rel(HEAVY_LOCK),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _synthetic_metrics(config, learned_selected=None):
    if learned_selected is None:
        learned_selected = config["learned_codebook"]
    n = 8192
    payload = (n // config["vec_dim"]) * config["bits"]
    side = 512
    lut = 52 + (1 << config["l_bits"]) * config["vec_dim"] * 4
    total = payload + side + lut * 8
    tensor = {
        "name": "model.layers.0.self_attn.q_proj.weight", "n": n,
        "bits": config["bits"], "bpw": round(total / n, 6),
        "payload_bits": payload, "trellis_side_bits": side,
        "outlier_side_bits": 0, "required_lut_bytes": lut,
        "vector_lut_required": True,
        "learned_lut_selected": learned_selected, "billing_complete": True,
        "billing_scope": ACCOUNTING_SCOPE, "rel_rms_pct": 42.0,
    }
    encoder = {
        "bits": config["bits"], "l": config["l_bits"], "k": config["bits"],
        "rht": True, "rht_axis": "cols", "tail_biting": False,
        "affine_min": False, "calibrated": False, "block_hessian": False,
        "mixed_precision": False, "vec_dim": config["vec_dim"],
        "block_len": config["block_len"],
        "learned_codebook": config["learned_codebook"],
        "learned_codebook_iters": 50, "learned_codebook_max_vectors": 16384,
        "encode_workers": THREADS, "artifact_class": "reconstruction_oracle",
        "deployable": False,
    }
    return {
        "tensors": [tensor],
        "aggregate": {
            "quantized_weights": n, "effective_bpw": round(total / n, 6),
            "oracle_effective_bpw": round(total / n, 6),
            "payload_bits": payload, "trellis_side_bits": side,
            "outlier_side_bits": 0, "required_lut_bytes": lut,
            "vector_lut_required_tensors": 1,
            "vector_lut_required_weights": n,
            "learned_lut_selected_tensors": 1 if learned_selected else 0,
            "learned_lut_selected_weights": n if learned_selected else 0,
            "billing_complete": True, "billing_scope": ACCOUNTING_SCOPE,
            "artifact_class": "reconstruction_oracle", "deployable": False,
            "weighted_rel_rms_pct": 42.0,
        },
        "config": encoder,
    }


def selftest():
    green = {
        "ok": True, "pressure_level": 1, "swap_used_mb": 0.0,
        "disk_free_gb": 400.0, "power_source": "Now drawing from 'AC Power'",
    }
    assert _evaluate_safety(green, {"ok": True})["ok"]
    assert not _evaluate_safety({**green, "pressure_level": 2}, {"ok": True})["ok"]
    assert not _evaluate_safety({**green, "swap_used_mb": 0.01}, {"ok": True})["ok"]
    assert not _evaluate_safety({**green, "disk_free_gb": 165.9}, {"ok": True})["ok"]
    assert not _evaluate_safety(green, {"ok": False})["ok"]
    names = [cfg["name"] for cfg in CONFIGS]
    assert len(names) == len(set(names)) == 11
    assert all(cfg["bits"] / cfg["vec_dim"] < 1 for cfg in CONFIGS)
    for config in CONFIGS:
        metrics = _synthetic_metrics(config)
        checked = _metrics_validation(metrics, config)
        assert checked["ok"], (config["name"], checked["blockers"])
        wrong = json.loads(json.dumps(metrics))
        wrong["aggregate"]["deployable"] = True
        assert not _metrics_validation(wrong, config)["ok"]
        wrong = json.loads(json.dumps(metrics))
        wrong["aggregate"]["payload_bits"] += 1
        assert not _metrics_validation(wrong, config)["ok"]
        if config["learned_codebook"]:
            assert not _metrics_validation(_synthetic_metrics(config, False), config)["ok"]
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tensor = {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16", "shape": [256, 256], "data_offsets": [0, 2],
            }
        }
        header = json.dumps(tensor, separators=(",", ":")).encode()
        path = root / "model.safetensors"
        with open(path, "wb") as handle:
            handle.write(len(header).to_bytes(8, "little"))
            handle.write(header)
            handle.write(b"\0\0")
        inspected = _read_safetensor_header(path)
        assert inspected["supported_quantizable_tensor_count"] == 1
        lock = root / "heavy.lock"
        first = open(lock, "a+")
        fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        second = open(lock, "a+")
        try:
            try:
                fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                raise AssertionError("second heavy lease unexpectedly succeeded")
            except BlockingIOError:
                pass
        finally:
            second.close()
            fcntl.flock(first.fileno(), fcntl.LOCK_UN)
            first.close()
    print("frontier_stream_queue selftest: PASS")
    return 0


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "start":
        return start_queue()
    if command == "run":
        return run_queue()
    if command == "resume":
        return resume_queue()
    if command == "status":
        return status()
    if command == "drain":
        return drain_queue()
    if command == "selftest":
        return selftest()
    print("usage: frontier_stream_queue.py {start|run|resume|status|drain|selftest}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
