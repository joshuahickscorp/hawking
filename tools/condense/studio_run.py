#!/usr/bin/env python3.12
"""studio_run.py — the chained, RAM-packed driver of the bit-floor-vs-scale experiment (plan §4).

This is the one command the Studio runs. It chains, for EACH model in the ladder, the safe recovery
stack and a binary search for that model's bit-floor, emits a receipt per floor point, and after the
whole ladder fits the floor-vs-scale curve. The parallelism is at MODEL granularity: each model's
pipeline runs serially inside one job (peak RAM = that model's doctor), and ram_scheduler packs whole
model-pipelines into the M3 Ultra 96 GB Studio. The default interactive-safe ladder runs the staged
0.5B/1.5B/7B wave, then 14B solo when present; 32B is held until a streamed path or an explicit,
measured memory waiver exists. Self-dispatching:
  studio_run.py --plan          # dry-run: print the per-model stages + the RAM-pack wave schedule
  studio_run.py --run           # schedule ALL models (packed) then fit the curve  [STUDIO]
  studio_run.py --model 7B      # run ONE model's full chain serially (what --run dispatches) [STUDIO]

Respects the §0 dead-ends and §6 proof discipline: effective bpw only, multiwindow eval, CPU-bf16
production numbers, judge on 7B+ (0.5B/1.5B are lab points, tagged baseline). Heavy stages are
guarded + checkpointed by the underlying tools. This driver also keeps a durable phase ledger so
an interruption or normal shutdown resumes at the first incomplete phase.
"""
import datetime
import fcntl
import hashlib
import math
import os
import re
import signal
import shutil
import sys
import json
import subprocess
import pathlib
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
TC = "tools/condense"
REC = "receipts/official"
FLOORS = "reports/cron/bit_floor_curve.jsonl"     # studio-lane floor curve (back-compat default)
RUN_STATE = pathlib.Path("reports/cron/studio_run_state.json")
RUN_PID = pathlib.Path("reports/cron/studio_run.pid")
WAIT_PID = pathlib.Path("reports/cron/studio_wait.pid")
DRAIN_REQUEST = pathlib.Path("reports/cron/studio_drain.request")
RUN_LOG = pathlib.Path("reports/cron/studio_run.log")
HEAVY_LOCK = pathlib.Path("reports/cron/studio_heavy.lock")
WAIT_LOCK = pathlib.Path("reports/cron/studio_wait.lock")
DOWNLOAD_QUEUE_PID = pathlib.Path("reports/condense/download_queue.pid.json")
DOWNLOAD_QUEUE_STATE_PATH = pathlib.Path("reports/condense/download_queue_state.json")
PROCESSING_QUEUE_PID = pathlib.Path("reports/condense/processing_queue.pid.json")
DOWNLOAD_STATE_DIR = pathlib.Path("reports/condense/download_state")
DOWNLOAD_PROCESS_BINDINGS = {
    "14B": ("Qwen/Qwen2.5-14B-Instruct", "scratch/staging/qwen-14b.partial"),
    "32B": ("Qwen/Qwen2.5-32B-Instruct", "scratch/staging/qwen-32b.partial"),
    "72B": ("Qwen/Qwen2.5-72B-Instruct", "scratch/staging/qwen-72b.partial"),
    "120B": ("openai/gpt-oss-120b", "scratch/staging/gpt-oss-120b.partial"),
    "DeepSeek-V4-Flash": (
        "deepseek-ai/DeepSeek-V4-Flash-DSpark",
        "scratch/staging/deepseek-v4-flash-dspark.partial",
    ),
    "Kimi-K2.6": ("moonshotai/Kimi-K2.6", "scratch/staging/kimi-k2.6.partial"),
}
sys.path.insert(0, TC)
from studio_manifest import DEFAULT_HARDWARE, FRONTIER_MODELS, frontier_by_label, frontier_labels
import procure
from ram_scheduler import HEAVY_LEASE_FD_ENV, inherited_lease_fds
from floor_integrity import (
    FLOOR_POINT_SCHEMA,
    canonical_row_sha256,
    create_floor_binding,
    locked_upsert_floor_row,
    validate_floor_binding,
    validate_receipt_floor_row,
)


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
    """Durably replace a JSON checkpoint; a power cut leaves the previous complete file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _try_heavy_lease(path=None):
    """Acquire the cross-supervisor heavy-work lease without blocking."""
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


def _active_run_pid():
    try:
        old = json.loads(RUN_PID.read_text()).get("pid")
        if old and int(old) != os.getpid():
            os.kill(int(old), 0)
            return int(old)
    except (FileNotFoundError, ProcessLookupError, ValueError, TypeError,
            json.JSONDecodeError, OSError):
        pass
    return None


def _load_run_state():
    try:
        with open(RUN_STATE) as f:
            state = json.load(f)
        if state.get("schema") == "hawking.studio_run_state.v1":
            return state
    except Exception:
        pass
    return {
        "schema": "hawking.studio_run_state.v1",
        "hardware_profile": DEFAULT_HARDWARE.name,
        "created_at": _now(),
        "updated_at": _now(),
        "phases": {},
    }


_BOOT_PHASE_EVIDENCE = None

# Every script or binary that can directly produce, admit, or summarize a checkpointed Studio
# phase. A detached process freezes these hashes at import time; a later on-disk edit forces a
# clean restart instead of allowing old in-memory code to bless new evidence.
PHASE_EVIDENCE_RELATIVE_PATHS = (
    "tools/condense/adapter_contract.py",
    "tools/condense/arch_coverage.py",
    "tools/condense/audit_ladder.py",
    "tools/condense/auto_bits.py",
    "tools/condense/awq.py",
    "tools/condense/bench_baselines.py",
    "tools/condense/calib_build.py",
    "tools/condense/codec_bakeoff.py",
    "tools/condense/codec_parallelism.py",
    "tools/condense/ctx_extend.py",
    "tools/condense/doctor.py",
    "tools/condense/download_queue.py",
    "tools/condense/eval_suite.py",
    "tools/condense/expert.py",
    "tools/condense/floor_integrity.py",
    "tools/condense/frontier_stream_queue.py",
    "tools/condense/frontier_common.py",
    "tools/condense/frontier_coverage.py",
    "tools/condense/frontier_receipts.py",
    "tools/condense/frontier_experiments.py",
    "tools/condense/frontier_experiment_runner.py",
    "tools/condense/kv.py",
    "tools/condense/ladder.py",
    "tools/condense/mixed_precision.py",
    "tools/condense/multi_eval.py",
    "tools/condense/procure.py",
    "tools/condense/processing_queue.py",
    "tools/condense/ram_scheduler.py",
    "tools/condense/ramcliff_bench.py",
    "tools/condense/receipt_verify.py",
    "tools/condense/residual.py",
    "tools/condense/scaling_law.py",
    "tools/condense/scorecard.py",
    "tools/condense/size_frontier.py",
    "tools/condense/spec_revive.py",
    "tools/condense/studio_manifest.py",
    "tools/condense/subbit.py",
    "tools/condense/tripwire_gate.py",
    "receipts/schema/condensation_receipt.schema.json",
    "scratch/calib_corpus.txt",
    "vendor/strand-quant/target/release/quantize-model",
)


def _current_phase_evidence():
    evidence_paths = [pathlib.Path(__file__).resolve(), *(
        ROOT / relative for relative in PHASE_EVIDENCE_RELATIVE_PATHS
    )]
    return {
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): _sha256_file(path)
        for path in evidence_paths if path.is_file()
    }


def _freeze_phase_evidence():
    global _BOOT_PHASE_EVIDENCE
    if _BOOT_PHASE_EVIDENCE is None:
        _BOOT_PHASE_EVIDENCE = _current_phase_evidence()
    return dict(_BOOT_PHASE_EVIDENCE)


def _phase_identity(name):
    """Bind a phase checkpoint to the code/recipe that produced it.

    A pre-upgrade detached GO may still finish and write ``status=pass`` using old in-memory
    functions. Status alone is therefore not resumable evidence. The next current-code resume trusts
    a phase only when this manifest matches; expensive model phases then reuse their stricter per-model
    hash-bound receipts instead of blindly repeating valid work.
    """
    boot_evidence = _freeze_phase_evidence()
    current_evidence = _current_phase_evidence()
    staged_models = {}
    for label, model_dir, *_rest in LADDER:
        path = pathlib.Path(model_dir)
        if path.is_dir():
            admission = _ladder_parent_admission(label, model_dir)
            staged_models[label] = {
                "model_dir": os.path.realpath(path),
                "model_fingerprint": _model_stat_fingerprint_for_dir(path),
                "download_admission": admission,
            }
    recipe = {
        "schema": "hawking.studio_phase_identity.v1",
        "phase": name,
        "hardware_profile": DEFAULT_HARDWARE.name,
        "core_research_configs": {
            lane: list(configs) for lane, configs in CORE_RESEARCH_CONFIGS.items()
        },
        "ladder": [list(row) for row in LADDER],
        "frontier_labels": list(frontier_labels()),
        "staged_models": staged_models,
        "environment": {
            key: os.environ.get(key) for key in (
                "HAWKING_STUDIO_RESEARCH_FULL", "HAWKING_ENABLE_SPEC_RESEARCH",
                "HAWKING_STUDIO_ALLOW_OVER_BUDGET",
                "FLOOR_GATE_PCT", "MULTIWINDOW", "STUDIO_TRIPWIRE",
                "DOCTOR_DEVICE", "DOCTOR_DTYPE", "DOCTOR_GRAD_ACCUM",
                "DOCTOR_KD_TOPK", "DOCTOR_TARGET_REGEX", "BAKE_QUALITY",
                "BAKE_ACTMEAN", "STRAND_F32_METRIC", "STRAND_F32_SEARCH",
                "LADDER_RETRY_ERRORS", "PPL_TEXT",
            )
        },
        "runtime_evidence_sha256": boot_evidence,
        "current_disk_evidence_sha256": current_evidence,
        "code_matches_runtime": boot_evidence == current_evidence,
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    recipe["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return recipe


def _set_phase(name, status, **extra):
    state = _load_run_state()
    row = dict(state.setdefault("phases", {}).get(name, {}))
    row.update(extra)
    current_identity = _phase_identity(name)
    started_identity = row.get("phase_identity")
    if status == "running" or started_identity is None:
        row["phase_identity"] = current_identity
    elif status in ("pass", "skipped") and started_identity != current_identity:
        # Never bless work produced across an on-disk code/recipe transition. Preserve the
        # identity captured at launch so a current-code resume reruns/revalidates the phase.
        status = "failed"
        row["identity_drift"] = {
            "detected_at": _now(),
            "started_identity_sha256": started_identity.get("identity_sha256")
                if isinstance(started_identity, dict) else None,
            "current_identity_sha256": current_identity.get("identity_sha256"),
        }
    row["status"] = status
    row["updated_at"] = _now()
    if status == "running":
        row["started_at"] = row.get("started_at") or row["updated_at"]
        row.pop("ended_at", None)
    elif status in ("pass", "failed", "interrupted", "skipped"):
        row["ended_at"] = row["updated_at"]
    state["phases"][name] = row
    state["updated_at"] = row["updated_at"]
    _atomic_json(RUN_STATE, state)
    return status


def _phase_done(name):
    row = _load_run_state().get("phases", {}).get(name, {})
    identity_ok = (
        row.get("status") in ("pass", "skipped")
        and row.get("phase_identity") == _phase_identity(name)
    )
    if not identity_ok:
        return False
    if name in {"P1_CONDENSE", "P2_SUBBIT"}:
        lane = "studio" if name == "P1_CONDENSE" else "subbit"
        staged = row["phase_identity"].get("staged_models", {})
        return bool(staged) and all(
            _have(label, lane) or _deferred_model_valid(label, lane)
            for label in staged
        )
    return True


def _draining():
    return DRAIN_REQUEST.exists()


def _checkpointed_phase(name, fn):
    """Run one phase once; only a durable pass/skipped record is trusted on resume."""
    if _phase_done(name):
        print(f"[studio] {name} checkpoint PASS — skip", file=sys.stderr)
        return 0
    if not _phase_identity(name).get("code_matches_runtime"):
        _set_phase(name, "failed", reason="on-disk code changed after this process started")
        print(f"[studio] {name} refused: on-disk code differs from loaded runtime; restart",
              file=sys.stderr)
        return 1
    if _draining():
        _set_phase(name, "interrupted", reason="drain requested before launch")
        return 130
    _set_phase(name, "running", pid=os.getpid())
    try:
        rc = int(fn() or 0)
    except KeyboardInterrupt:
        rc = 130
    except Exception as exc:
        _set_phase(name, "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"[studio] {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if rc == 0:
        if _set_phase(name, "pass", returncode=0) != "pass":
            print(f"[studio] {name} identity changed while running; checkpoint refused",
                  file=sys.stderr)
            return 1
    elif rc == 130 or _draining():
        _set_phase(name, "interrupted", returncode=rc, reason="drain/signal")
    else:
        _set_phase(name, "failed", returncode=rc)
    return rc


def _checkpointed_call(key, argv, env=None, acceptable=(0,)):
    """Checkpoint a command within its phase and terminate it cleanly when drain is requested."""
    if _phase_done(key):
        print(f"[studio] {key} checkpoint PASS — skip", file=sys.stderr)
        return 0
    if not _phase_identity(key).get("code_matches_runtime"):
        _set_phase(key, "failed", reason="on-disk code changed after this process started")
        return 1
    _set_phase(key, "running", command=list(argv), pid=os.getpid())
    child_env = dict(os.environ if env is None else env)
    lease_fds = inherited_lease_fds(os.environ)
    if lease_fds:
        child_env[HEAVY_LEASE_FD_ENV] = str(lease_fds[0])
    proc = subprocess.Popen(
        argv, env=child_env, start_new_session=True, pass_fds=lease_fds,
    )
    while proc.poll() is None:
        if _draining():
            print(f"[studio] drain: SIGTERM {key} pid={proc.pid}", file=sys.stderr)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print(f"[studio] drain grace expired: SIGKILL {key}", file=sys.stderr)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _set_phase(key, "interrupted", returncode=proc.wait(), reason="drain requested")
            return 130
        time.sleep(1)
    rc = proc.returncode
    if rc in acceptable:
        effective = _set_phase(key, "pass", returncode=rc)
        return 0 if effective == "pass" else 1
    _set_phase(key, "failed", returncode=rc)
    return rc


def _model_complete_path(label, set_name):
    return pathlib.Path(f"reports/cron/{set_name}_{label}.complete.json")


CORE_RESEARCH_CONFIGS = {
    "studio": (
        "f16", "4-AWQ", "3-AWQ", "2-AWQ", "1-AWQ",
        "3-AWQ+dr", "2-AWQ+dr", "1-AWQ+dr",
    ),
    "subbit": (
        # These are required *oracle measurements*, not deployable floor points.  The vector
        # codebooks are not serialized by .tq v2 yet, so audit/scaling code must keep every VTQ
        # row marked deployable=false until packed round-trip and runtime-parity gates pass.
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


def _audit_jsonl_path(label, set_name):
    return pathlib.Path(f"reports/cron/{set_name}_{label}.jsonl")


def _coverage_path(label, set_name):
    return pathlib.Path(f"reports/cron/{set_name}_{label}.coverage.json")


def _lane_receipt_path(label, set_name):
    """The authoritative receipt is lane-specific; no lane may alias another lane's proof."""
    if set_name == "subbit":
        return pathlib.Path(f"receipts/official/{label}-subbit-campaign.json")
    return pathlib.Path(f"receipts/official/{label}-{set_name}-floor.json")


def _legacy_studio_receipt_path(label):
    """Pre-lane path retained only as a compatibility copy of the Studio receipt."""
    return pathlib.Path(f"receipts/official/{label}-floor.json")


PREFIRE_DEVICE = "studio-m3ultra-96"
AUDIT_RECIPE_VERSION = "hawking.audit.recipe.2026-07-12.v2"


def _prefire_path(label):
    return pathlib.Path(f"reports/condense/prefire/{label}.json")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    return not positive or float(value) > 0.0


def _model_stat_fingerprint_for_dir(model_dir):
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
        if not (name in {"config.json", "generation_config.json", *tokenizer_names}
                or name.endswith(".safetensors") or name.endswith(".safetensors.index.json")):
            continue
        path = os.path.join(model_dir, name)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        item = {"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if name.endswith(".json") or name in tokenizer_names:
            item["sha256"] = _sha256_file(path)
        manifest["files"].append(item)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _audit_identity_path(audit_jsonl):
    value = str(audit_jsonl)
    return pathlib.Path(value[:-6] + ".identity.json" if value.endswith(".jsonl")
                        else value + ".identity.json")


def _audit_identity_valid(identity, label, set_name):
    try:
        model_dir = next(row[1] for row in LADDER if row[0] == label)
        expected_lane = (f"{set_name}_full"
                         if os.environ.get("HAWKING_STUDIO_RESEARCH_FULL") == "1"
                         else set_name)
        expected_eval_path = os.path.realpath(os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt"))
        actmean = os.environ.get("BAKE_ACTMEAN")
        actmean_exists = not actmean or os.path.isfile(actmean)
        expected_actmean_path = os.path.realpath(actmean) if actmean and actmean_exists else None
        expected_actmean_sha = _sha256_file(actmean) if actmean and actmean_exists else None
        evidence = identity.get("evidence_files")
        required_evidence = {
            "vendor/strand-quant/target/release/quantize-model",
            str(pathlib.Path(__file__).resolve().parents[0] / "audit_ladder.py"),
            "tools/condense/doctor.py", "tools/condense/multi_eval.py",
            "tools/condense/adapter_contract.py",
            "tools/condense/tripwire_gate.py",
        }
        if os.path.isfile("scratch/calib_corpus.txt"):
            required_evidence.add("scratch/calib_corpus.txt")
        # audit_ladder records __file__ exactly as invoked (normally tools/condense/...). Accept
        # either spelling while still requiring every recorded hash to match current bytes.
        evidence_names = set(evidence) if isinstance(evidence, dict) else set()
        has_audit = any(name.endswith("tools/condense/audit_ladder.py") for name in evidence_names)
        other_required = {name for name in required_evidence if not name.endswith("audit_ladder.py")}
        return bool(
            identity.get("schema") == "hawking.audit_identity.v1"
            and identity.get("recipe_version") == AUDIT_RECIPE_VERSION
            and identity.get("model") == label
            and identity.get("lane") == expected_lane
            and os.path.realpath(str(identity.get("model_dir", ""))) == os.path.realpath(model_dir)
            and identity.get("model_fingerprint") == _model_stat_fingerprint_for_dir(model_dir)
            and os.path.realpath(identity["eval_text_path"]) == expected_eval_path
            and identity.get("eval_text_sha256") == _sha256_file(expected_eval_path)
            and identity.get("device") == "cpu"
            and identity.get("dtype") == "torch.bfloat16"
            and identity.get("multiwindow") == 4
            and identity.get("studio_tripwire") is True
            and identity.get("bake_quality") == (os.environ.get("BAKE_QUALITY") == "1")
            and actmean_exists
            and identity.get("bake_actmean_path") == expected_actmean_path
            and identity.get("bake_actmean_sha256") == expected_actmean_sha
            and identity.get("strand_f32_metric") == os.environ.get("STRAND_F32_METRIC")
            and identity.get("strand_f32_search") == os.environ.get("STRAND_F32_SEARCH")
            and identity.get("doctor_grad_accum") == 4
            and identity.get("doctor_kd_topk") == 64
            and identity.get("doctor_target_regex") is None
            and has_audit
            and other_required.issubset(evidence_names)
            and all(os.path.isfile(path) and digest == _sha256_file(path)
                    for path, digest in evidence.items())
        )
    except Exception:
        return False


def _doctor_row_valid(row):
    """Require a finished Doctor plus exact serialized-adapter density accounting.

    A checkpoint is resumable work, not a completed recovery result.  This validation is repeated
    at the orchestration boundary so an older/in-memory audit process cannot create a new coverage
    manifest from a partial or historically under-billed ``+dr`` JSONL row.
    """
    doctor = row.get("doctor") if isinstance(row, dict) else None
    final = doctor.get("final") if isinstance(doctor, dict) else None
    accounting = doctor.get("adapter_accounting") if isinstance(doctor, dict) else None
    if not (
        isinstance(doctor, dict)
        and doctor.get("complete") is True
        and isinstance(final, dict)
        and final.get("stopped_early") is False
        and isinstance(accounting, dict)
        and accounting.get("schema") == "hawking.doctor_adapter_accounting.v1"
    ):
        return False
    for key in (
        "rank", "adapter_bytes", "quantized_weights", "adapter_effective_bpw",
        "base_effective_bpw", "total_effective_bpw",
    ):
        if not _finite_number(accounting.get(key), positive=True):
            return False
    if abs(
        float(accounting["adapter_effective_bpw"])
        - float(accounting["adapter_bytes"]) * 8.0 / float(accounting["quantized_weights"])
    ) > 1e-9:
        return False
    if abs(
        float(accounting["total_effective_bpw"])
        - float(accounting["base_effective_bpw"])
        - float(accounting["adapter_effective_bpw"])
    ) > 1e-9:
        return False
    return (
        _finite_number(row.get("eff_bpw"), positive=True)
        and abs(float(row["eff_bpw"]) - float(accounting["total_effective_bpw"])) <= 0.0011
    )


def _vtq_oracle_row_valid(config, row):
    """Validate a measured-but-explicitly-nondeployable vector-trellis oracle row."""
    identity = re.fullmatch(
        r"vtq-k(?P<bits>\d+)-d(?P<vec_dim>\d+)-b(?P<block_len>\d+)-"
        r"(?P<codebook>frozen|learned)(?:\+dr-r(?P<rank>\d+))?",
        config,
    )
    if identity is None:
        return False
    oracle = row.get("oracle") if isinstance(row, dict) else None
    accounting = oracle.get("accounting") if isinstance(oracle, dict) else None
    recipe = oracle.get("recipe") if isinstance(oracle, dict) else None
    if not (
        row.get("artifact_class") == "reconstruction_oracle"
        and row.get("deployable") is False
        and isinstance(oracle, dict)
        and oracle.get("schema") == "hawking.vtq_reconstruction_oracle.v1"
        and oracle.get("artifact_class") == "reconstruction_oracle"
        and oracle.get("deployable") is False
        and oracle.get("packed_artifact") is None
        and isinstance(accounting, dict)
        and isinstance(recipe, dict)
        and accounting.get("billing_scope")
            == "logical_codec_stream_plus_required_lut_not_physical_packed_artifact"
        and accounting.get("method")
            == "exact encoder payload/trellis-side/OUTL bits + required per-tensor Q12 LUT bytes"
    ):
        return False
    for key in ("quantized_weights", "logical_stream_bits_including_required_lut"):
        if not _finite_number(accounting.get(key), positive=True):
            return False
    for key in ("payload_bits", "trellis_side_bits", "outlier_side_bits", "required_lut_bytes"):
        value = accounting.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return False
    logical_bits = (
        accounting["payload_bits"] + accounting["trellis_side_bits"]
        + accounting["outlier_side_bits"] + accounting["required_lut_bytes"] * 8
    )
    if logical_bits != accounting["logical_stream_bits_including_required_lut"]:
        return False
    if not _finite_number(accounting.get("oracle_effective_bpw"), positive=True) \
            or abs(
                float(accounting["oracle_effective_bpw"])
                - float(logical_bits) / float(accounting["quantized_weights"])
            ) > 1e-9:
        return False
    if not (
        _finite_number(recipe.get("bits"), positive=True)
        and _finite_number(recipe.get("vec_dim"), positive=True)
        and _finite_number(recipe.get("block_len"), positive=True)
    ):
        return False
    expected_recipe = {
        "bits": int(identity.group("bits")),
        "l_bits": int(identity.group("bits")) + 4,
        "vec_dim": int(identity.group("vec_dim")),
        "block_len": int(identity.group("block_len")),
        "learned_codebook": identity.group("codebook") == "learned",
        "outlier_pct": 0.0,
        "awq_alpha": 0.0,
        "rht": "cols",
        "trellis_quality": False,
        "actmean": False,
        "learned_codebook_iters": 50,
        "learned_codebook_max_vectors": 16384,
        "input_dtype": "torch.bfloat16",
    }
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        return False
    workers = recipe.get("encode_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        return False
    if expected_recipe["learned_codebook"] and workers != 1:
        return False
    try:
        model_label = row.get("model")
        model_dir = next(entry[1] for entry in LADDER if entry[0] == model_label)
    except Exception:
        return False
    if recipe.get("source_fingerprint") != _model_stat_fingerprint_for_dir(model_dir):
        return False
    quantizer = "vendor/strand-quant/target/release/quantize-model"
    if recipe.get("quantizer_sha256") != _sha256_file(quantizer):
        return False
    recipe_sha = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if oracle.get("recipe_sha256") != recipe_sha:
        return False
    selected_tensors = accounting.get("learned_lut_selected_tensors")
    selected_weights = accounting.get("learned_lut_selected_weights")
    vector_tensors = accounting.get("vector_lut_required_tensors")
    vector_weights = accounting.get("vector_lut_required_weights")
    selected_fraction = accounting.get("learned_lut_selected_weight_fraction")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (selected_tensors, selected_weights, vector_tensors, vector_weights)):
        return False
    if not isinstance(selected_fraction, (int, float)) \
            or not math.isfinite(float(selected_fraction)) \
            or not 0.0 <= float(selected_fraction) <= 1.0:
        return False
    learned = identity.group("codebook") == "learned"
    expected_lut_bytes = vector_tensors * (
        52 + (1 << int(recipe["l_bits"])) * int(recipe["vec_dim"]) * 4
    )
    if not (
        vector_tensors > 0
        and vector_weights == accounting["quantized_weights"]
        and accounting["required_lut_bytes"] == expected_lut_bytes
        and selected_tensors <= vector_tensors
        and selected_weights <= vector_weights
    ):
        return False
    if learned:
        if not (selected_tensors > 0 and 0 < selected_weights <= accounting["quantized_weights"]):
            return False
    elif any((selected_tensors, selected_weights, selected_fraction)):
        return False
    if abs(
        float(selected_fraction)
        - float(selected_weights) / float(accounting["quantized_weights"])
    ) > 1e-9:
        return False
    rank = identity.group("rank")
    if rank is not None:
        doctor = row.get("doctor") if isinstance(row, dict) else None
        doctor_accounting = doctor.get("adapter_accounting") if isinstance(doctor, dict) else None
        if not isinstance(doctor_accounting, dict) \
                or doctor_accounting.get("rank") != int(rank):
            return False
        if doctor_accounting.get("quantized_weights") != accounting["quantized_weights"]:
            return False
        oracle_plus = accounting.get("oracle_plus_adapter_effective_bpw")
        if not _finite_number(oracle_plus, positive=True):
            return False
        if abs(float(doctor_accounting["base_effective_bpw"])
               - float(accounting["oracle_effective_bpw"])) > 1e-9:
            return False
        if abs(float(oracle_plus) - float(doctor_accounting["total_effective_bpw"])) > 1e-9:
            return False
    expected = accounting.get("oracle_plus_adapter_effective_bpw")
    if expected is None:
        expected = accounting.get("oracle_effective_bpw")
    return (
        _finite_number(expected, positive=True)
        and _finite_number(row.get("eff_bpw"), positive=True)
        and abs(float(row["eff_bpw"]) - float(expected)) <= 0.0011
    )


def _write_core_coverage(label, set_name, audit_jsonl=None, coverage_path=None):
    """Persist and return fail-closed core-config coverage for one model/lane.

    ``audit_ladder.py`` deliberately records per-config failures and continues, so its zero exit
    status alone is not completion. The last row for every required config must be a usable
    measurement. Binding this receipt to the audit JSONL hash prevents a later append/edit from
    silently inheriting stale coverage.
    """
    audit_jsonl = pathlib.Path(audit_jsonl or _audit_jsonl_path(label, set_name))
    coverage_path = pathlib.Path(coverage_path or _coverage_path(label, set_name))
    identity_path = _audit_identity_path(audit_jsonl)
    required = CORE_RESEARCH_CONFIGS.get(set_name)
    report = {
        "schema": "hawking.studio_core_coverage.v1",
        "generated_at": _now(),
        "status": "failed",
        "model": label,
        "lane": set_name,
        "audit_jsonl": str(audit_jsonl),
        "audit_sha256": None,
        "audit_identity": str(identity_path),
        "audit_identity_sha256": None,
        "identity_errors": [],
        "required_configs": list(required or ()),
        "configs": {},
        "missing_configs": [],
        "error_configs": [],
        "invalid_configs": [],
        "parse_errors": [],
    }
    if required is None:
        report["invalid_configs"] = [f"unknown lane: {set_name}"]
        _atomic_json(coverage_path, report)
        return False
    try:
        identity = json.loads(identity_path.read_text())
        report["audit_identity_sha256"] = _sha256_file(identity_path)
        if not _audit_identity_valid(identity, label, set_name):
            report["identity_errors"].append("audit identity is stale or does not match current source/recipe")
    except Exception as exc:
        report["identity_errors"].append(
            f"audit identity missing/unreadable: {type(exc).__name__}: {exc}"
        )
    if not audit_jsonl.is_file():
        report["missing_configs"] = list(required)
        report["parse_errors"] = [f"audit JSONL missing: {audit_jsonl}"]
        _atomic_json(coverage_path, report)
        return False

    latest = {}
    with open(audit_jsonl) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except Exception as exc:
                report["parse_errors"].append(
                    f"line {line_number}: {type(exc).__name__}: {exc}"
                )
                continue
            config = row.get("config") if isinstance(row, dict) else None
            if config in required:
                latest[config] = (line_number, row)

    for config in required:
        found = latest.get(config)
        if found is None:
            report["missing_configs"].append(config)
            report["configs"][config] = {"status": "missing"}
            continue
        line_number, row = found
        item = {"line": line_number}
        if row.get("error") is not None:
            item.update({"status": "error", "error": str(row.get("error"))[:500]})
            report["error_configs"].append(config)
        elif (row.get("model") != label
              or not _finite_number(row.get("ppl"), positive=True)
              or not _finite_number(row.get("eff_bpw"), positive=True)
              or not _finite_number(row.get("degr_pct"))
              or ("+dr" in config and not _doctor_row_valid(row))
              or (config.startswith("vtq-") and not _vtq_oracle_row_valid(config, row))):
            item.update({
                "status": "invalid",
                "model": row.get("model"),
                "ppl": row.get("ppl"),
                "eff_bpw": row.get("eff_bpw"),
                "degr_pct": row.get("degr_pct"),
                "doctor_complete": (row.get("doctor") or {}).get("complete")
                    if isinstance(row.get("doctor"), dict) else None,
                "doctor_accounting_schema": ((row.get("doctor") or {}).get(
                    "adapter_accounting") or {}).get("schema")
                    if isinstance((row.get("doctor") or {}).get("adapter_accounting"), dict)
                    else None,
                "artifact_class": row.get("artifact_class"),
                "deployable": row.get("deployable"),
                "oracle_schema": (row.get("oracle") or {}).get("schema")
                    if isinstance(row.get("oracle"), dict) else None,
            })
            report["invalid_configs"].append(config)
        else:
            item.update({
                "status": "pass",
                "ppl": float(row["ppl"]),
                "eff_bpw": float(row["eff_bpw"]),
                "degr_pct": float(row["degr_pct"]),
                "artifact_class": row.get("artifact_class"),
                "deployable": row.get("deployable"),
            })
        report["configs"][config] = item

    report["audit_sha256"] = _sha256_file(audit_jsonl)
    report["status"] = "pass" if not any((
        report["missing_configs"], report["error_configs"], report["invalid_configs"],
        report["parse_errors"], report["identity_errors"],
    )) else "failed"
    _atomic_json(coverage_path, report)
    return report["status"] == "pass"


def _core_coverage_valid(label, set_name, audit_jsonl=None, coverage_path=None):
    """Validate the durable coverage receipt against its exact current audit JSONL."""
    audit_jsonl = pathlib.Path(audit_jsonl or _audit_jsonl_path(label, set_name))
    coverage_path = pathlib.Path(coverage_path or _coverage_path(label, set_name))
    identity_path = _audit_identity_path(audit_jsonl)
    required = CORE_RESEARCH_CONFIGS.get(set_name)
    if required is None or not audit_jsonl.is_file():
        return False
    try:
        coverage = json.loads(coverage_path.read_text())
        identity = json.loads(identity_path.read_text())
        recorded_audit = pathlib.Path(coverage.get("audit_jsonl", ""))
        if not recorded_audit.is_absolute():
            recorded_audit = ROOT / recorded_audit
        actual_audit = audit_jsonl if audit_jsonl.is_absolute() else ROOT / audit_jsonl
        return bool(
            coverage.get("schema") == "hawking.studio_core_coverage.v1"
            and coverage.get("status") == "pass"
            and coverage.get("model") == label
            and coverage.get("lane") == set_name
            and coverage.get("audit_identity") == str(identity_path)
            and coverage.get("audit_identity_sha256") == _sha256_file(identity_path)
            and _audit_identity_valid(identity, label, set_name)
            and recorded_audit.resolve() == actual_audit.resolve()
            and coverage.get("required_configs") == list(required)
            and all(coverage.get("configs", {}).get(config, {}).get("status") == "pass"
                    for config in required)
            and not coverage.get("missing_configs")
            and not coverage.get("error_configs")
            and not coverage.get("invalid_configs")
            and not coverage.get("parse_errors")
            and not coverage.get("identity_errors")
            and coverage.get("audit_sha256") == _sha256_file(audit_jsonl)
        )
    except Exception:
        return False


def _receipt_matches_lane(path, label, set_name, audit_jsonl=None):
    """Reject a correctly named receipt whose payload was produced from the other lane."""
    audit_jsonl = pathlib.Path(audit_jsonl or _audit_jsonl_path(label, set_name))
    try:
        receipt = json.loads(pathlib.Path(path).read_text())
    except Exception:
        return False
    artifact = str(receipt.get("condensed_artifact", ""))
    candidates = {str(audit_jsonl)}
    candidates.add(str((audit_jsonl if audit_jsonl.is_absolute() else ROOT / audit_jsonl).resolve()))
    return (
        receipt.get("project") == "hawking"
        and label in str(receipt.get("source_model", ""))
        and any(candidate in artifact for candidate in candidates)
    )


def _write_subbit_campaign_receipt(label, audit_jsonl, coverage_path, receipt_path=None):
    """Commit completed VTQ research without manufacturing a deployable bit floor.

    Every current VTQ row is a dense reconstruction oracle.  Finishing all required experiments is
    an orchestration success, but there is no packed/resident candidate for ``scaling_law --floor``.
    This receipt advances the detached queue while keeping product promotion explicitly blocked.
    """
    audit_jsonl = pathlib.Path(audit_jsonl)
    coverage_path = pathlib.Path(coverage_path)
    latest = {}
    with open(audit_jsonl) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("config"):
                latest[row["config"]] = row
    required = list(CORE_RESEARCH_CONFIGS["subbit"])
    oracle_rows = [latest[name] for name in required if name != "f16"]
    if not oracle_rows or any(not _vtq_oracle_row_valid(row["config"], row)
                              for row in oracle_rows):
        raise ValueError("subbit campaign receipt requires every mandatory VTQ oracle row")
    min_density = min(oracle_rows, key=lambda row: float(row["eff_bpw"]))
    best_quality = min(oracle_rows, key=lambda row: float(row["degr_pct"]))
    receipt = {
        "schema": "hawking.subbit_campaign_complete.v1",
        "project": "hawking",
        "status": "research-complete",
        "generated_at": _now(),
        "model": label,
        "source_model": f"{label} reconstruction-oracle campaign",
        "lane": "subbit",
        "required_configs": required,
        "audit_jsonl": str(audit_jsonl),
        "audit_sha256": _sha256_file(audit_jsonl),
        "coverage": str(coverage_path),
        "coverage_sha256": _sha256_file(coverage_path),
        "condensed_artifact": f"VTQ reconstruction oracles ({audit_jsonl})",
        "artifact_class": "reconstruction_oracle",
        "deployable": False,
        "product_gate": False,
        "floor_bpw": None,
        "hypothesis_status": "not_yet_testable_as_deployable_artifact",
        "minimum_oracle": {
            "config": min_density["config"], "eff_bpw": min_density["eff_bpw"],
            "degr_pct": min_density["degr_pct"],
        },
        "best_quality_oracle": {
            "config": best_quality["config"], "eff_bpw": best_quality["eff_bpw"],
            "degr_pct": best_quality["degr_pct"],
        },
        "promotion_blockers": [
            "quantize-model does not yet bind learned VTQ LUTs into a produced packed archive",
            "no packed file-bpw / CPU runtime / Metal parity / direct-residency receipt exists",
            "reconstruction-oracle rows are ineligible for bit-floor and source-release claims",
        ],
    }
    path = pathlib.Path(receipt_path or _lane_receipt_path(label, "subbit"))
    _atomic_json(path, receipt)
    return path


def _subbit_campaign_valid(path, label, audit_jsonl, coverage_path):
    try:
        receipt = json.loads(pathlib.Path(path).read_text())
        return bool(
            receipt.get("schema") == "hawking.subbit_campaign_complete.v1"
            and receipt.get("project") == "hawking"
            and receipt.get("status") == "research-complete"
            and receipt.get("model") == label
            and receipt.get("lane") == "subbit"
            and receipt.get("required_configs") == list(CORE_RESEARCH_CONFIGS["subbit"])
            and receipt.get("artifact_class") == "reconstruction_oracle"
            and receipt.get("deployable") is False
            and receipt.get("product_gate") is False
            and receipt.get("floor_bpw") is None
            and receipt.get("audit_jsonl") == str(audit_jsonl)
            and receipt.get("audit_sha256") == _sha256_file(audit_jsonl)
            and receipt.get("coverage") == str(coverage_path)
            and receipt.get("coverage_sha256") == _sha256_file(coverage_path)
            and _receipt_matches_lane(path, label, "subbit", audit_jsonl)
        )
    except Exception:
        return False


def _prefire_valid(label, params, path=None):
    """A prior advisor checkpoint is reusable only for this model, size, and Studio profile."""
    path = pathlib.Path(path or _prefire_path(label))
    try:
        receipt = json.loads(path.read_text())
        bpw = receipt.get("target_bpw")
        return bool(
            receipt.get("schema") == "hawking.studio_prefire.v1"
            and receipt.get("status") == "pass"
            and receipt.get("model") == label
            and float(receipt.get("params_b")) == float(params)
            and receipt.get("device") == PREFIRE_DEVICE
            and _finite_number(bpw, positive=True)
            and receipt.get("auto_bits", {}).get("advisor_only") is True
            and receipt.get("auto_bits", {}).get("device") == PREFIRE_DEVICE
            and receipt.get("size_frontier", {}).get("device") == PREFIRE_DEVICE
            and _finite_number(receipt.get("doctor_plan", {}).get("target_bpw"), positive=True)
            and float(receipt.get("doctor_plan", {}).get("target_bpw")) == float(bpw)
        )
    except Exception:
        return False


def _run_prefire(label, params, env):
    """Checkpoint the cheap bit/size/Doctor advisors; never substitute them for the real ladder."""
    checkpoint = _prefire_path(label)
    if _prefire_valid(label, params, checkpoint):
        print(f"[{label}] pre-fire advisor checkpoint PASS — exhaustive ladder still follows",
              file=sys.stderr)
        return 0
    base = {
        "schema": "hawking.studio_prefire.v1",
        "status": "failed",
        "generated_at": _now(),
        "model": label,
        "params_b": params,
        "device": PREFIRE_DEVICE,
        "advisor_only": True,
        "does_not_replace": "audit_ladder core coverage, floor gate, or receipt verification",
    }

    commands = []

    def invoke(argv):
        commands.append(list(argv))
        if _draining():
            raise InterruptedError("drain requested before advisor launch")
        rc = subprocess.run(
            argv, env=env, pass_fds=inherited_lease_fds(os.environ)
        ).returncode
        if rc != 0:
            raise RuntimeError(f"advisor failed rc={rc}: {' '.join(argv)}")

    try:
        invoke([
            "python3.12", f"{TC}/auto_bits.py", "--params", str(params), "--label", label,
            "--device", PREFIRE_DEVICE,
        ])
        auto_path = pathlib.Path(f"reports/condense/{label}_autobits.json")
        auto = json.loads(auto_path.read_text())
        if (auto.get("model") != label or auto.get("device") != PREFIRE_DEVICE
                or float(auto.get("total_b")) != float(params)
                or auto.get("advisor_only") is not True):
            raise ValueError(f"auto-bits output identity mismatch: {auto_path}")
        recommended = auto.get("recommended_bpw")
        target_bpw = float(recommended) if _finite_number(recommended, positive=True) else 3.34

        invoke([
            "python3.12", f"{TC}/size_frontier.py", str(params), "--bpw", str(target_bpw),
            "--device", PREFIRE_DEVICE,
        ])
        size_path = pathlib.Path(
            f"reports/condense/size_{int(params)}b_{PREFIRE_DEVICE}.json"
        )
        size = json.loads(size_path.read_text())
        if (size.get("device") != PREFIRE_DEVICE
                or float(size.get("model_total_b")) != float(params)
                or float(size.get("bpw")) != target_bpw):
            raise ValueError(f"size-frontier output identity mismatch: {size_path}")

        invoke([
            "python3.12", f"{TC}/doctor.py", "registry", "--select", str(params),
            str(target_bpw), "--device", PREFIRE_DEVICE,
        ])
        doctor_path = pathlib.Path(f"reports/condense/doctor_plan_{int(params)}b.json")
        doctor = json.loads(doctor_path.read_text())
        if (float(doctor.get("params_b")) != float(params)
                or not _finite_number(doctor.get("target_bpw"), positive=True)):
            raise ValueError(f"Doctor plan output identity mismatch: {doctor_path}")

        receipt = {
            **base,
            "status": "pass",
            "completed_at": _now(),
            "target_bpw": target_bpw,
            "target_source": "auto_bits.recommended_bpw" if _finite_number(
                recommended, positive=True
            ) else "fallback-3.34",
            "commands": commands,
            "auto_bits_path": str(auto_path),
            "auto_bits_sha256": _sha256_file(auto_path),
            "auto_bits": auto,
            "size_frontier_path": str(size_path),
            "size_frontier_sha256": _sha256_file(size_path),
            "size_frontier": size,
            "doctor_plan_path": str(doctor_path),
            "doctor_plan_sha256": _sha256_file(doctor_path),
            "doctor_plan": doctor,
        }
        _atomic_json(checkpoint, receipt)
        print(f"[{label}] pre-fire PASS: advisor target={target_bpw} bpw on {PREFIRE_DEVICE}; "
              "starting exhaustive ladder", file=sys.stderr)
        return 0
    except InterruptedError as exc:
        _atomic_json(checkpoint, {**base, "status": "interrupted", "commands": commands,
                                  "error": str(exc), "updated_at": _now()})
        return 130
    except Exception as exc:
        _atomic_json(checkpoint, {**base, "commands": commands,
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "updated_at": _now()})
        print(f"[{label}] pre-fire FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


def _run_floor_with_lane_receipt(label, set_name, audit_jsonl, floor_jsonl, model_dir, env):
    """Run the legacy floor emitter in isolation, then commit its receipt to this lane only.

    ``scaling_law.py`` still emits ``<label>-floor.json`` relative to its cwd. A temporary cwd
    inside the repository lets git metadata resolve while ensuring a subbit run cannot even
    transiently overwrite the Studio compatibility receipt.
    """
    audit_jsonl = pathlib.Path(audit_jsonl)
    floor_jsonl = pathlib.Path(floor_jsonl)
    model_dir = pathlib.Path(model_dir)
    audit_abs = (audit_jsonl if audit_jsonl.is_absolute() else ROOT / audit_jsonl).resolve()
    floor_abs = (floor_jsonl if floor_jsonl.is_absolute() else ROOT / floor_jsonl).resolve()
    model_abs = (model_dir if model_dir.is_absolute() else ROOT / model_dir).resolve()
    temp_parent = ROOT / "reports/cron"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{set_name}_{label}_floor_", dir=temp_parent) as td:
        work = pathlib.Path(td)
        (work / "receipts/official").mkdir(parents=True)
        # Preserve the optional frozen-suite identity in the isolated receipt environment.
        for relative in (pathlib.Path("receipts/prompt_suite_v1.sha256"),
                         pathlib.Path("prompts/frozen/suite_v1.sha256")):
            source = ROOT / relative
            if source.is_file():
                destination = work / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        rc = subprocess.run([
            "python3.12", str((ROOT / TC / "scaling_law.py").resolve()), "--floor", label,
            str(audit_abs), str(floor_abs), str(model_abs),
        ], env=env, cwd=work, pass_fds=inherited_lease_fds(os.environ)).returncode
        if rc != 0:
            return rc
        generated = work / f"receipts/official/{label}-floor.json"
        try:
            receipt = json.loads(generated.read_text())
        except Exception as exc:
            print(f"[{label}] {set_name} floor receipt missing/invalid: {exc}", file=sys.stderr)
            return 4
        lane_receipt = _lane_receipt_path(label, set_name)
        _atomic_json(lane_receipt, receipt)
        if set_name == "studio":
            _atomic_json(_legacy_studio_receipt_path(label), receipt)
    if not _receipt_matches_lane(_lane_receipt_path(label, set_name), label, set_name, audit_jsonl):
        print(f"[{label}] {set_name} floor receipt does not match its audit lane", file=sys.stderr)
        return 4
    return 0


def floors_path(set_name="studio"):
    """Per-lane floor file so the studio and subbit lanes don't overwrite each other's curve."""
    return FLOORS if set_name == "studio" else f"reports/cron/bit_floor_{set_name}.jsonl"

# model ladder: label -> (hf dir, params(B), doctor peak GB, solo?, role)
# Measured/estimated peaks for the M3 Ultra 96 GB box. The scheduler's interactive-safe budget is
# lower than physical RAM so ChatGPT/Codex and macOS stay responsive. 32B remains listed for the
# research plan but is deliberately over budget and will not launch without an explicit override.
LADDER = [
    ("0.5B", "scratch/qwen-05b", 0.5, 10, False, "lab"),
    ("1.5B", "scratch/qwen-15b", 1.5, 10, False, "lab"),
    ("7B",   "scratch/qwen-7b",  7.0, 40, False, "substrate"),   # the honest mid; 1-bit judged here
    ("14B",  "scratch/staging/qwen-14b.partial", 14.0, 65, True,  "verified-payoff-solo"),
    ("32B",  "scratch/staging/qwen-32b.partial", 32.0, 85, True,
     "verified-streamed-capstone"),
]

# Download-backed ladder parents are consumed only from the exact path attested by a successful
# `hf cache verify` marker. The marker bytes and mutation-sensitive source fingerprint are then part
# of every Studio phase identity, so neither a swapped marker nor a changed parent can cross resume.
LADDER_DOWNLOAD_BINDINGS = {
    "14B": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "verified_source": "scratch/staging/qwen-14b.partial",
    },
    "32B": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "verified_source": "scratch/staging/qwen-32b.partial",
    },
}


def _ladder_parent_admission(label, model_dir):
    binding = LADDER_DOWNLOAD_BINDINGS.get(label)
    model_path = pathlib.Path(model_dir)
    fingerprint = _model_stat_fingerprint_for_dir(model_path) if model_path.is_dir() else None
    if binding is None:
        return {
            "required": False,
            "ok": model_path.is_dir() and fingerprint is not None,
            "model_dir": str(model_dir),
            "model_fingerprint": fingerprint,
            "blockers": [] if model_path.is_dir() and fingerprint is not None
                        else ["parent directory is absent or cannot be fingerprinted"],
        }
    expected_source = ROOT / binding["verified_source"]
    _state_path, marker_path_raw = procure._checkpoint_paths(label)
    marker_path = ROOT / marker_path_raw
    marker = procure._read_json(marker_path, {})
    blockers = []
    try:
        path_matches = os.path.samefile(model_path, expected_source)
    except OSError:
        path_matches = False
    if not path_matches:
        blockers.append("model path is not the exact verified staging parent")
    marker_ok = procure._verified_marker_valid(
        marker, label=label, hf_id=binding["hf_id"],
        local_dir=binding["verified_source"], require_verify=True,
    )
    if not marker_ok:
        blockers.append("hf verification marker is absent or not bound to label/HF id/staged path")
    marker_sha = _sha256_file(marker_path) if marker_ok and marker_path.is_file() else None
    if fingerprint is None:
        blockers.append("verified parent cannot be mutation-fingerprinted")
    return {
        "required": True,
        "ok": not blockers,
        "label": label,
        "hf_id": binding["hf_id"],
        "model_dir": str(model_dir),
        "verified_source": binding["verified_source"],
        "verified_marker": str(marker_path.relative_to(ROOT)),
        "verified_marker_sha256": marker_sha,
        "model_fingerprint": fingerprint,
        "blockers": blockers,
    }


def _full_model_scheduler_admission(est_gb, process_budget):
    if est_gb <= process_budget:
        return {"ok": True, "reason": "within interactive-safe process budget"}
    if os.environ.get("HAWKING_STUDIO_ALLOW_OVER_BUDGET") == "1":
        return {"ok": True, "reason": "explicit measured over-budget waiver"}
    return {
        "ok": False,
        "reason": (
            f"estimated peak {est_gb:.0f}GB exceeds {process_budget:.0f}GB process budget; "
            "full-model lane deferred to a streamed implementation"
        ),
    }


STREAMED_DEFER_NEXT = (
    "frontier_stream_queue representative-shard VTQ research; "
    "full Doctor/PPL awaits a measured streamed implementation"
)


def _deferred_model_path(label, lane):
    return pathlib.Path(f"reports/cron/{lane}_{label}.deferred.json")


def _deferred_model_document(label, lane, est_gb, process_budget, parent_admission):
    scheduler_admission = _full_model_scheduler_admission(est_gb, process_budget)
    if scheduler_admission["ok"]:
        raise ValueError("a runnable model cannot be recorded as memory-deferred")
    return {
        "schema": "hawking.studio_model_deferred.v1",
        "status": "deferred-memory-budget",
        "recorded_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "model": label,
        "lane": lane,
        "estimated_peak_gb": est_gb,
        "process_budget_gb": process_budget,
        "reason": scheduler_admission["reason"],
        "parent_admission": parent_admission,
        "safe_next_path": STREAMED_DEFER_NEXT,
        "automatic_memory_waiver": False,
    }


def _deferred_model_valid(label, lane, receipt_path=None):
    row = next((entry for entry in LADDER if entry[0] == label), None)
    if row is None:
        return False
    _label, model_dir, _params, est_gb, _solo, _role = row
    process_budget = float(getattr(
        DEFAULT_HARDWARE, "process_budget_gb", DEFAULT_HARDWARE.weight_budget_gb
    ))
    parent_admission = _ladder_parent_admission(label, model_dir)
    if not parent_admission["ok"]:
        return False
    scheduler_admission = _full_model_scheduler_admission(est_gb, process_budget)
    if scheduler_admission["ok"]:
        return False
    try:
        doc = json.loads(pathlib.Path(
            receipt_path or _deferred_model_path(label, lane)
        ).read_text())
    except Exception:
        return False
    expected = _deferred_model_document(
        label, lane, est_gb, process_budget, parent_admission
    )
    # Time documents when the decision was made, but every admission/budget/route semantic must be
    # byte-for-byte current. A verified-parent mutation or future streamed implementation therefore
    # invalidates the defer and makes the phase reconsider the model.
    expected["recorded_at"] = doc.get("recorded_at")
    return bool(expected["recorded_at"] and doc == expected)

# FRONTIER — the 100B+ research targets (the real prize). Kept in studio_manifest.py so procurement,
# RAM-cliff, and docs share one set of hardware/model facts. On this 96 GB box, only targets whose
# artifacts fit DEFAULT_HARDWARE.weight_budget_gb are resident candidates; the rest require a real
# paging/streaming path. Parent downloads are always governed separately by current free disk.
FRONTIER = [
    (m.label, m.local_dir, m.total_b, m.active_b, m.serve_bpw, m.moe, m.role, m.hf_id)
    for m in FRONTIER_MODELS
]

# the recovery stack run per model, cheapest-first (plan §2). Each entry: (stage, tool, note).
STACK = [
    ("L0 calib",      f"{TC}/calib_build.py",     "domain-matched corpus (input to all below)"),
    ("L1 AWQ",        f"{TC}/awq.py bake",        "alpha=0.5 pre-scale + bake"),
    ("L2 mixed-prec", f"{TC}/mixed_precision.py", "output-sensitivity bit allocation"),
    ("L3 residual",   f"{TC}/residual.py bake",   "full-rank residual (train-free ~1:1)"),
    ("L4 block-QAT",  f"{TC}/doctor.py blockwise","GATED: sharded/dtype/per-layer-resume proof required"),
    ("L5 GPTQ-Hess",  f"{TC}/doctor.py strand",   "GATED: requested-model + durable-state proof required"),
    ("L6 deep-KD",    f"{TC}/doctor.py lora",     "logit/feature KD polish on the full-rank base"),
]


def _have(label, set_name="studio"):
    """Trust a floor only when completion, hash-bound coverage, and this lane's receipt agree."""
    expected_audit_set = (f"{set_name}_full"
                          if os.environ.get("HAWKING_STUDIO_RESEARCH_FULL") == "1"
                          else set_name)
    complete_path = _model_complete_path(label, set_name)
    audit_jsonl = _audit_jsonl_path(label, set_name)
    coverage_path = _coverage_path(label, set_name)
    receipt_path = _lane_receipt_path(label, set_name)
    try:
        with open(complete_path) as f:
            complete = json.load(f)
        if not (complete.get("schema") == "hawking.studio_model_complete.v1"
                and complete.get("status") == "pass"
                and complete.get("model") == label
                and complete.get("lane") == set_name
                and complete.get("audit_set") == expected_audit_set
                and complete.get("audit_jsonl") == str(audit_jsonl)
                and complete.get("coverage") == str(coverage_path)
                and complete.get("coverage_sha256") == _sha256_file(coverage_path)
                and complete.get("receipt") == str(receipt_path)
                and complete.get("receipt_sha256") == _sha256_file(receipt_path)):
            return False
    except Exception:
        return False
    if not _core_coverage_valid(label, set_name, audit_jsonl, coverage_path):
        return False
    if not _receipt_matches_lane(receipt_path, label, set_name, audit_jsonl):
        return False
    if set_name == "subbit":
        return bool(
            complete.get("result_kind") == "research-campaign"
            and complete.get("floor_jsonl") is None
            and _subbit_campaign_valid(
                receipt_path, label, audit_jsonl, coverage_path
            )
        )
    valid_floor, _problems = validate_floor_binding(
        complete, ROOT, set_name, label, floors_path(set_name), audit_jsonl,
    )
    if not valid_floor:
        return False
    try:
        if float(complete["floor_row"]["gate_pct"]) != float(
            os.environ.get("FLOOR_GATE_PCT", "2.0")
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except Exception:
        return False
    receipt_ok, _receipt_problems = validate_receipt_floor_row(
        receipt, complete.get("floor_row"),
    )
    return receipt_ok


def run_model(label, set_name="studio"):
    """Run ONE model's full chain serially: (SUBBIT-0 gate) -> bake+recovery stack -> floor search ->
    receipt. Heavy; STUDIO only. set_name selects audit_ladder's config set ('studio' = the L0-L6
    bit-floor stack; 'subbit' = the sub-1/sub-2-bit frontier lane). Each step is checkpointed."""
    row = next(r for r in LADDER if r[0] == label)
    _, mdir, params, est_gb, _, role = row
    if not os.path.isdir(mdir):
        print(f"[{label}] SKIP — parent not staged at {mdir} (download on the Studio)", file=sys.stderr)
        return 2
    parent_admission = _ladder_parent_admission(label, mdir)
    if not parent_admission["ok"]:
        print(f"[{label}] BLOCKED — parent admission failed: "
              f"{'; '.join(parent_admission['blockers'])}", file=sys.stderr)
        return 4
    process_budget = float(getattr(DEFAULT_HARDWARE, "process_budget_gb",
                                   DEFAULT_HARDWARE.weight_budget_gb))
    scheduler_admission = _full_model_scheduler_admission(est_gb, process_budget)
    if not scheduler_admission["ok"]:
        print(f"[{label}] BLOCKED — {scheduler_admission['reason']}", file=sys.stderr)
        return 3
    ncpu = str(os.cpu_count() or 8)
    env = {**os.environ, "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "MULTIWINDOW": "4", "STUDIO_TRIPWIRE": "1", "DOCTOR_THREADS": ncpu,
           # Error rows are durable diagnostics, not completion. A later detached resume must
           # retry them after code/binary/resource repair instead of deadlocking coverage forever.
           "LADDER_RETRY_ERRORS": os.environ.get("LADDER_RETRY_ERRORS", "1"),
           "OMP_NUM_THREADS": ncpu, "MKL_NUM_THREADS": ncpu, "VECLIB_MAXIMUM_THREADS": ncpu,
           # Stop launching at yellow pressure and checkpoint at red; do not make tens of GB of swap.
           "DOCTOR_SWAP_CEIL": os.environ.get("DOCTOR_SWAP_CEIL", "2000"),
           "DOCTOR_SWAP_HARD_CEIL": os.environ.get("DOCTOR_SWAP_HARD_CEIL", "6000")}
    log = f"reports/cron/{set_name}_{label}.log"
    out = f"reports/cron/{set_name}_{label}"
    audit_set = f"{set_name}_full" if os.environ.get("HAWKING_STUDIO_RESEARCH_FULL") == "1" else set_name
    print(f"[{label}] {set_name} chain start (audit_set={audit_set}, role={role}, {params}B) -> {log}",
          file=sys.stderr)
    # Cheap, durable pre-fire advisors choose a starting density and recovery recipe for this exact
    # Studio profile. They are advisory only: successful return always continues into architecture
    # coverage and the exhaustive required-config ladder below.
    rc = _run_prefire(label, params, env)
    if rc != 0:
        return rc
    # Architecture coverage: which Doctor levers are arch-compatible for this model (dense here
    # today; the same check that flags Mamba2/RWKV-7 SSM state + MoE per-expert applicability).
    rc = subprocess.run(
        ["python3.12", f"{TC}/arch_coverage.py", mdir, label], env=env,
        pass_fds=inherited_lease_fds(os.environ),
    ).returncode
    if rc != 0:
        return rc
    # SUBBIT-0 GATE: measure the entropy/side-info floor first. If sub-1-bit dense is DEAD by the
    # floor, the subbit lane still runs (MoE/residual survive) but the gate is on record per model.
    if set_name == "subbit":
        rc = subprocess.run(
            ["python3.12", f"{TC}/subbit.py", "measure", mdir, label], env=env,
            pass_fds=inherited_lease_fds(os.environ),
        ).returncode
        if rc != 0:
            return rc
        # SUBBIT-4 probe: per-expert sensitivity decides MoE sub-bit allocation (gated to MoE dirs).
        if any(t in label.lower() for t in ("moe", "a22b", "a3b", "deepseek", "mixtral", "glm")):
            rc = subprocess.run(
                ["python3.12", f"{TC}/expert.py", "sensitivity", mdir, "--label", label,
                 "--bits", "1,2"], env=env, pass_fds=inherited_lease_fds(os.environ),
            ).returncode
            if rc != 0:
                return rc
    # Stage 1-2: the chosen stack via audit_ladder (bakes + AWQ + mixed + residual + L6 LoRA-KD +
    # L4 block-QAT + L5 GPTQ-Hessian), each checkpointed/guarded, multiwindow ppl + tripwire.
    audit_cmd = ["python3.12", f"{TC}/audit_ladder.py", mdir, label, audit_set, out]
    while True:
        rc = subprocess.run(
            audit_cmd, env=env, pass_fds=inherited_lease_fds(os.environ)
        ).returncode
        if rc != 75:
            break
        if _draining():
            return 130
        print(f"[{label}] audit model lock is busy; retrying in 30s", file=sys.stderr)
        time.sleep(30)
    if rc != 0:
        print(f"[{label}] audit failed rc={rc}; floor/receipt deliberately not emitted", file=sys.stderr)
        return rc
    audit_jsonl = _audit_jsonl_path(label, set_name)
    coverage_path = _coverage_path(label, set_name)
    if not _write_core_coverage(label, set_name, audit_jsonl, coverage_path):
        try:
            coverage = json.loads(coverage_path.read_text())
            details = {
                key: coverage.get(key) for key in (
                    "missing_configs", "error_configs", "invalid_configs", "parse_errors"
                ) if coverage.get(key)
            }
        except Exception:
            details = {"coverage": "unreadable"}
        print(f"[{label}] {set_name} core coverage FAILED: {details}; "
              "floor/receipt deliberately not emitted", file=sys.stderr)
        return 4
    # Stage 3-4: the scalar lane may select a deployable floor. The VTQ lane currently contains
    # reconstruction oracles only, so completing it writes a research-campaign receipt and never
    # feeds those rows to the deployable scaling law.
    if set_name == "subbit":
        try:
            receipt = _write_subbit_campaign_receipt(label, audit_jsonl, coverage_path)
        except Exception as exc:
            print(f"[{label}] subbit campaign receipt failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 4
        if not _subbit_campaign_valid(receipt, label, audit_jsonl, coverage_path):
            print(f"[{label}] subbit campaign receipt failed validation", file=sys.stderr)
            return 4
        result_kind = "research-campaign"
        floor_jsonl = None
        floor_binding = {"floor_jsonl": None}
    else:
        rc = _run_floor_with_lane_receipt(
            label, set_name, audit_jsonl, floors_path(set_name), mdir, env
        )
        if rc != 0:
            return rc
        receipt = _lane_receipt_path(label, set_name)
        rc = subprocess.run(
            ["python3.12", f"{TC}/receipt_verify.py", str(receipt)], env=env,
            pass_fds=inherited_lease_fds(os.environ),
        ).returncode
        if rc != 0:
            return rc
        result_kind = "deployable-floor-experiment"
        floor_curve = floors_path(set_name)
        try:
            floor_binding = create_floor_binding(
                ROOT, set_name, label, floor_curve, audit_jsonl,
            )
            valid_floor, floor_problems = validate_floor_binding(
                floor_binding, ROOT, set_name, label, floor_curve, audit_jsonl,
            )
            receipt_doc = json.loads(pathlib.Path(receipt).read_text())
            if not valid_floor:
                raise ValueError("; ".join(floor_problems))
            receipt_ok, receipt_problems = validate_receipt_floor_row(
                receipt_doc, floor_binding["floor_row"],
            )
            if not receipt_ok:
                raise ValueError("; ".join(receipt_problems))
        except Exception as exc:
            print(f"[{label}] floor completion binding failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 4
    completion = {
        "schema": "hawking.studio_model_complete.v1",
        "status": "pass",
        "completed_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "model": label,
        "lane": set_name,
        "audit_set": audit_set,
        "model_dir": mdir,
        "audit_jsonl": str(audit_jsonl),
        "coverage": str(coverage_path),
        "coverage_sha256": _sha256_file(coverage_path),
        "required_configs": list(CORE_RESEARCH_CONFIGS[set_name]),
        "result_kind": result_kind,
        "receipt": str(receipt),
        "receipt_sha256": _sha256_file(receipt),
        **floor_binding,
    }
    _atomic_json(_model_complete_path(label, set_name), completion)
    return 0


def run_all(set_name="studio"):
    """Schedule every model's chain, packed into RAM, then fit the curve."""
    from ram_scheduler import Scheduler, Job
    jobs = []
    process_budget = float(getattr(DEFAULT_HARDWARE, "process_budget_gb",
                                   DEFAULT_HARDWARE.weight_budget_gb))
    for lbl, mdir, _params, gb, solo, _role in LADDER:
        if _have(lbl, set_name):
            continue
        if not os.path.isdir(mdir):
            print(f"[{set_name}] {lbl} not staged — deferred", file=sys.stderr)
            continue
        parent_admission = _ladder_parent_admission(lbl, mdir)
        if not parent_admission["ok"]:
            print(f"[{set_name}] {lbl} parent not admitted — deferred: "
                  f"{'; '.join(parent_admission['blockers'])}", file=sys.stderr)
            continue
        scheduler_admission = _full_model_scheduler_admission(gb, process_budget)
        if not scheduler_admission["ok"]:
            deferred_path = _deferred_model_path(lbl, set_name)
            _atomic_json(deferred_path, _deferred_model_document(
                lbl, set_name, gb, process_budget, parent_admission
            ))
            print(f"[{set_name}] {lbl} safely deferred — {scheduler_admission['reason']} "
                  f"(receipt: {deferred_path})", file=sys.stderr)
            continue
        jobs.append(Job(
            lbl,
            ["python3.12", f"{TC}/studio_run.py", "--model", lbl, set_name],
            est_gb=gb,
            solo=solo,
            # `_have` above already validated any existing completion manifest. Leaving this unset
            # prevents Scheduler's existence-only shortcut from trusting a stale pre-coverage file.
            done_when=None,
            log=f"reports/cron/{set_name}_{lbl}.log",
            checkpoint_safe=True,
        ))
    scheduler = Scheduler(
        budget_gb=process_budget,
        statusf=f"reports/cron/{set_name}_sched.status",
        drain_file=str(DRAIN_REQUEST),
    )
    results = scheduler.run(jobs)
    if _draining():
        return 130
    failed = {name: rc for name, rc in results.items() if rc not in (0, 2)}
    if failed:
        print(f"[{set_name}] model failures/blocks: {failed}", file=sys.stderr)
        return 1
    floor = floors_path(set_name)
    if not os.path.exists(floor):
        print(f"[{set_name}] no completed floor rows yet; leaving curve fit pending", file=sys.stderr)
        return 0
    return subprocess.run(
        ["python3.12", f"{TC}/scaling_law.py", "--fit", floor],
        env=os.environ.copy(), pass_fds=inherited_lease_fds(os.environ),
    ).returncode


def run_frontier(label):
    """SERVE-oriented frontier pipeline for a 100B+ model (the real research prize). The doctor does
    NOT fit (f16 2x resident overflows this Studio), so this runs what DOES on streamed shards: the SUBBIT-0
    entropy floor + per-expert sensitivity (MoE) + the serve-fit record. The block-wise condense to a
    serve-fit .tq, the native-serve quality number, and the RAM-cliff tps demo are the Rust serve
    build (read_strand into the serve binary + the per-expert .tq writer) — emitted as gated steps."""
    spec = frontier_by_label(label)
    if not spec:
        print(f"[frontier] unknown {label}; known: {', '.join(frontier_labels())}", file=sys.stderr)
        return 2
    mdir, total, active, bpw, moe, role, hf_id = (
        spec.local_dir, spec.total_b, spec.active_b, spec.serve_bpw, spec.moe, spec.role, spec.hf_id
    )
    artifact = round(spec.artifact_gb(), 1)
    fits = spec.fits_resident(DEFAULT_HARDWARE)
    resident = "RESIDENT (no pager)" if fits else "OVERFLOW (SSD-bound, deep frontier)"
    ncpu = str(os.cpu_count() or 8)
    env = {**os.environ, "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "OMP_NUM_THREADS": ncpu, "MKL_NUM_THREADS": ncpu, "VECLIB_MAXIMUM_THREADS": ncpu}
    print(f"[frontier] {spec.label} ({total}B{f', act {active}B MoE' if moe else ' dense'}, role={role}) "
          f"-> {bpw} bpw = {artifact}GB ({resident} on {DEFAULT_HARDWARE.name}; "
          f"weight budget={DEFAULT_HARDWARE.weight_budget_gb:.0f}GB; source={spec.source_kind})",
          file=sys.stderr)
    if not os.path.isdir(mdir):
        print(f"[frontier] {spec.label} NOT staged at {mdir}. Fastest-SOTA procurement (hf_transfer + hf_xet, "
              f"link-bound): python3.12 {TC}/procure.py {spec.label}  "
              f"(download~{spec.download_gb:.0f}GB; verify current free disk + safety reserve first)",
              file=sys.stderr)
        return 2
    # Auto mode: recommend the bit format + serve regime (RESIDENT / MOE-PAGED / DENSE-OOC) and show
    # the device size ceiling before condensing (the "how big can we pull in" advisor).
    ab = ["python3.12", f"{TC}/auto_bits.py", "--params", str(total), "--label", spec.label]
    if active:
        ab += ["--active", str(active)]
    rc = _checkpointed_call(f"P4_FRONTIER/{spec.label}/auto-bits", ab, env=env)
    if rc != 0:
        return rc
    sf = ["python3.12", f"{TC}/size_frontier.py", str(total), "--bpw", str(bpw)]
    if active:
        sf += ["--active", str(active)]
    rc = _checkpointed_call(f"P4_FRONTIER/{spec.label}/size-frontier", sf, env=env)
    if rc != 0:
        return rc
    # The Doctor registry: the auto-composed recovery chain (L0-L6 + per-expert) for this model/bpw.
    dr = ["python3.12", f"{TC}/doctor.py", "registry", "--select", str(total), str(bpw)]
    if moe:
        dr.append("--moe")
    rc = _checkpointed_call(f"P4_FRONTIER/{spec.label}/doctor-registry", dr, env=env)
    if rc != 0:
        return rc
    # Runs on streamed shards (no full f16 resident): the entropy floor + the MoE expert decision.
    rc = _checkpointed_call(
        f"P4_FRONTIER/{spec.label}/subbit-measure",
        ["python3.12", f"{TC}/subbit.py", "measure", mdir, spec.label], env=env)
    if rc != 0:
        return rc
    # Architecture coverage: real state geometry (Mamba2/RWKV-7 flat state) + which Doctor levers
    # are arch-compatible for this model, so the selector above never wastes a bake on one that isn't.
    rc = _checkpointed_call(
        f"P4_FRONTIER/{spec.label}/arch-coverage",
        ["python3.12", f"{TC}/arch_coverage.py", mdir, spec.label], env=env)
    if rc != 0:
        return rc
    if moe:
        rc = _checkpointed_call(
            f"P4_FRONTIER/{spec.label}/expert-sensitivity",
            ["python3.12", f"{TC}/expert.py", "sensitivity", mdir, "--label", spec.label,
             "--bits", "1,2"], env=env)
        if rc != 0:
            return rc
        # Hot-expert cache policy: simulate hit-rate/blended-tok/s across cache sizes so the OOC
        # pager's cache size is chosen from a measured sweep, not a guess. n_experts best-effort
        # from config; falls back to a documented default sized to this model's active fraction.
        n_experts = 0
        try:
            n_experts = json.load(open(os.path.join(mdir, "config.json"))).get(
                "n_routed_experts") or json.load(open(os.path.join(mdir, "config.json"))).get(
                "num_local_experts") or 0
        except Exception:
            pass
        n_experts = n_experts or max(8, round(total / max(1.0, active or 1.0)) * 4)
        expert_size_b = total / max(1, n_experts)          # params per single expert
        active_k = max(1, round((active or total * 0.05) / max(0.1, expert_size_b)))
        active_gb_tok = (active or total * 0.05) * bpw / 8.0   # TOTAL active bytes/token (all active experts)
        rc = _checkpointed_call(
            f"P4_FRONTIER/{spec.label}/expert-cache",
            ["python3.12", f"{TC}/expert.py", "cache", "--sim", str(n_experts),
             str(active_k), "--active-gb", str(round(active_gb_tok, 3))], env=env)
        if rc != 0:
            return rc
    # the serve-build steps (Rust, gated): block-wise condense + native-serve quality + RAM-cliff tps
    rec = {"model": spec.label, "hf_id": hf_id, "total_b": total, "active_b": active, "moe": moe,
           "role": role, "source_kind": spec.source_kind, "note": spec.note,
           "serve_bpw": bpw, "artifact_gb": artifact, "serve_fits_resident": fits,
           "resident_weight_budget_gb": DEFAULT_HARDWARE.weight_budget_gb,
           "resident_no_pager": fits,
           "condense_cmd": f"# block-wise streamed single-bake (+per-expert if MoE) to {spec.label}.tq @ {bpw}bpw",
           "serve_quality_gated_on": "read_strand wired into hawking-serve binary + native .tq GEMV",
           "ram_cliff_demo": f"serve {spec.label}.tq ({artifact}GB resident) vs Q4_K ({round(total*4.5/8)}GB, overflows->swap)"}
    os.makedirs("reports/condense", exist_ok=True)
    json.dump(rec, open(f"reports/condense/{spec.label}_frontier.json", "w"), indent=2)
    print(f"[frontier] {spec.label} serve-fit recorded; quality+cliff GATED on the native serve build",
          file=sys.stderr)
    return 0


def run_frontier_all():
    """Frontier models are each ~box-filling -> run sequentially (the scheduler would serialize them
    anyway). Skips unstaged. The serve build is the gate on the quality/cliff numbers."""
    failures = {}
    for spec in FRONTIER_MODELS:
        rc = run_frontier(spec.label)
        if rc not in (0, 2):
            failures[spec.label] = rc
    if failures:
        print(f"[frontier] failures: {failures}", file=sys.stderr)
        return 1
    return 0


def plan():
    sys.path.insert(0, TC)
    from ram_scheduler import Scheduler, Job
    print("=" * 78)
    print("CHAINED PER-MODEL PIPELINE (each runs serially inside one scheduler job):")
    process_budget = float(getattr(DEFAULT_HARDWARE, "process_budget_gb",
                                   DEFAULT_HARDWARE.weight_budget_gb))
    for lbl, mdir, params, gb, solo, role in LADDER:
        staged = "staged" if os.path.isdir(mdir) else "NEEDS DOWNLOAD"
        print(f"\n  {lbl} ({params}B, role={role}, doctor~{gb}GB{', SOLO' if solo else ''}, {staged})")
        admission = _full_model_scheduler_admission(gb, process_budget)
        if not admission["ok"]:
            print(f"    FULL-MODEL DEFER: {admission['reason']}")
            print("    SAFE EXECUTION: verified marker + representative-shard SHA -> detached "
                  "frontier_stream_queue VTQ 0.5/0.33/0.25/0.125-bit probes")
            print("    CLAIM LIMIT: reconstruction oracle only; no PPL/Doctor/deployable floor claim")
            continue
        print(f"    1. bake+ppl ladder (audit_ladder, frontier set: AWQ/mixed/residual/outlier + multiwindow ppl)")
        for stage, tool, note in STACK:
            print(f"    2. {stage:13s} {note}")
        print(f"    3. floor-search: lowest eff-bpw at <=+2% ppl AND multi_eval tripwire pass")
        print(f"    4. emit receipt (repro level; 0.5B/1.5B tagged baseline, never set the verdict)")
    print("\n" + "=" * 78)
    print(f"RAM-PACK SCHEDULE ({DEFAULT_HARDWARE.name} -> {process_budget:.0f}GB interactive-safe budget):")
    jobs = [Job(lbl, ["true"], est_gb=gb, solo=solo) for (lbl, _, _, gb, solo, _) in LADDER]
    Scheduler(budget_gb=process_budget).plan(jobs)
    print("\nAfter the last model: scaling_law.py fits floor vs log(N), draws the recovered-vs-PTQ")
    print("band, and extrapolates the 70B/405B floor (T3.1) as a pre-registered prediction.")


SPEC_TARGETS = ["7B", "32B"]   # condensed substrate + capstone to revive spec-decode on


def efficiency_baseline():
    """Write the pre-ladder measurement contract from the computational-efficiency agenda."""
    try:
        from ram_scheduler import resource_snapshot
        resources = resource_snapshot()
    except Exception as exc:
        resources = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    staged = {
        label: {
            "model_dir": model_dir,
            "staged": os.path.isdir(model_dir),
            "estimated_peak_gb": peak_gb,
            "role": role,
        }
        for label, model_dir, _params, peak_gb, _solo, role in LADDER
    }
    out = {
        "schema": "hawking.studio_efficiency_baseline.v1",
        "generated_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "research_agenda": "docs/plans/computational_efficiency_paradigms_2026_07_11.md",
        "objective": {
            "primary": "quality-adjusted utility goodput under p95/p99 SLO constraints",
            "pareto_metrics": [
                "capability_per_joule",
                "capability_per_weighted_byte_moved",
                "capability_per_touched_parameter",
                "capability_per_wall_clock_second",
            ],
            "must_charge": [
                "draft_and_rejected_work",
                "download_and_storage_bytes",
                "checkpoint_and_recompute_cost",
                "quality_and_safety_regressions",
            ],
        },
        "resources": resources,
        "ladder": staged,
        "claim_status": "measurement-contract-only",
    }
    _atomic_json("reports/condense/studio_efficiency_baseline.json", out)
    return 0


def _spec_gate():
    gate_path = pathlib.Path("reports/condense/spec_oracle_gate.json")
    try:
        gate = json.loads(gate_path.read_text())
    except Exception:
        gate = {}
    ok = (
        os.environ.get("HAWKING_ENABLE_SPEC_RESEARCH") == "1"
        and gate.get("status") == "pass"
        and float(gate.get("tau", 0.0)) >= 2.5
        and gate.get("one_pass_verifier") is True
    )
    return ok, gate_path, gate


def run_spec_phase():
    ok, gate_path, gate = _spec_gate()
    if not ok:
        print("[spec] gated: require HAWKING_ENABLE_SPEC_RESEARCH=1 plus a passing "
              f"{gate_path} with tau>=2.5 and one_pass_verifier=true; existing EAGLE/n-gram paths "
              "remain below the resurrection gate", file=sys.stderr)
        return 0
    for lbl in SPEC_TARGETS:
        row = next((r for r in LADDER if r[0] == lbl), None)
        if not row or not os.path.isdir(row[1]) or not _have(lbl, "studio"):
            print(f"[spec] {lbl} lacks a completed condensed substrate — defer", file=sys.stderr)
            continue
        rc = _checkpointed_call(
            f"P3_SPEC/{lbl}",
            ["python3.12", f"{TC}/spec_revive.py", row[1], lbl],
        )
        if rc != 0:
            return rc
    return 0


def run_eval_phase():
    targets = [(l, m, p) for (l, m, p, _g, _s, _r) in LADDER
               if l not in ("0.5B", "1.5B") and _have(l, "studio")]
    commands = (
        ("eval", lambda l, m, p: ["python3.12", f"{TC}/eval_suite.py", "--model", m, "--label", l]),
        ("ctx", lambda l, m, p: ["python3.12", f"{TC}/ctx_extend.py", m, l]),
        ("kv-frontier", lambda l, m, p: ["python3.12", f"{TC}/kv.py", "frontier", m, l, str(p)]),
        ("kv-hybrid", lambda l, m, p: ["python3.12", f"{TC}/kv.py", "hybrid", m, l, str(p)]),
    )
    for lbl, mdir, params in targets:
        for substage, build in commands:
            rc = _checkpointed_call(f"P5_EVAL/{lbl}/{substage}", build(lbl, mdir, params))
            if rc != 0:
                return rc
    return 0


def run_baseline_phase():
    targets = [(l, m) for (l, m, _p, _g, _s, _r) in LADDER
               if l not in ("0.5B", "1.5B") and _have(l, "studio")]
    for lbl, mdir in targets:
        rc = _checkpointed_call(
            f"P6_BASELINE/{lbl}",
            ["python3.12", f"{TC}/bench_baselines.py", "--model", mdir, "--label", lbl,
             "--audit-jsonl", f"reports/cron/studio_{lbl}.jsonl"],
        )
        if rc != 0:
            return rc
    return 0


def run_codec_phase():
    target = next(((l, m) for (l, m, _p, _g, _s, _r) in LADDER
                   if l not in ("0.5B", "1.5B") and _have(l, "studio")), None)
    if not target:
        return 0
    lbl, mdir = target
    return _checkpointed_call(
        f"P8_CODEC/{lbl}",
        ["python3.12", f"{TC}/codec_bakeoff.py", "--model", mdir, "--label", lbl],
    )


def run_synthesis_phase():
    for lane in ("studio", "subbit"):
        floor = floors_path(lane)
        if os.path.exists(floor):
            rc = _checkpointed_call(
                f"P9_SYNTH/fit-{lane}",
                ["python3.12", f"{TC}/scaling_law.py", "--fit", floor],
            )
            if rc != 0:
                return rc
    return _checkpointed_call(
        "P9_SYNTH/scorecard",
        ["python3.12", f"{TC}/scorecard.py"],
    )


def go():
    """Run the Studio program from the first incomplete durable phase checkpoint.

    On the Studio, this is the one command:
        python3.12 tools/condense/studio_run.py go
    A hard power loss can only lose work since the last underlying model/config checkpoint; the
    phase marked `running` is rerun, while `pass` phases are skipped."""
    if _draining():
        print(f"[studio] drain is active at {DRAIN_REQUEST}; use `studio_run.py resume` after relocation",
              file=sys.stderr)
        return 130
    gate_rc = _enforce_launch_gate()
    if gate_rc != 0:
        return gate_rc
    old = _active_run_pid()
    if old is not None:
        print(f"[studio] another run is live pid={old}", file=sys.stderr)
        return 2
    heavy_lease = _try_heavy_lease()
    if heavy_lease is None:
        print(f"[studio] heavy-work lease is held at {HEAVY_LOCK}; launch held", file=sys.stderr)
        return 75
    previous_lease_env = os.environ.get(HEAVY_LEASE_FD_ENV)
    os.environ[HEAVY_LEASE_FD_ENV] = str(heavy_lease.fileno())
    try:
        # Recheck under the lease. This covers an older live GO that predates the
        # flock contract but is still represented by RUN_PID.
        old = _active_run_pid()
        if old is not None:
            print(f"[studio] another run is live pid={old}", file=sys.stderr)
            return 2
        _atomic_json(RUN_PID, {"pid": os.getpid(), "started_at": _now(), "mode": "running",
                               "hardware_profile": DEFAULT_HARDWARE.name,
                               "heavy_lock": str(HEAVY_LOCK)})
        print("=" * 78)
        print(f"HAWKING STUDIO — GO ({DEFAULT_HARDWARE.name}, durable phase resume)")
        print("=" * 78)
        stages = [
            ("P0E_EFFICIENCY", "capability-efficiency baseline/measurement contract", efficiency_baseline),
            ("P0_CODEC", "codec parallelism triage", lambda: _checkpointed_call(
                "P0_CODEC/catalog", ["python3.12", f"{TC}/codec_parallelism.py", "--catalog"])),
            ("P1_CONDENSE", "safe bit-floor ladder", lambda: run_all("studio")),
            ("P2_SUBBIT", "safe sub-bit ladder", lambda: run_all("subbit")),
            ("P3_SPEC", "speculation oracle (gated by tau and one-pass verifier)", run_spec_phase),
            ("P4_FRONTIER", "serve-oriented staged frontier probes", run_frontier_all),
            ("P5_EVAL", "capability, long-context, and state evaluation", run_eval_phase),
            ("P6_BASELINE", "same-box baseline comparison", run_baseline_phase),
            ("P7_CLIFF", "RAM-cliff and energy", lambda: _checkpointed_call(
                "P7_CLIFF/bench", ["python3.12", f"{TC}/ramcliff_bench.py", "--all"])),
            ("P8_CODEC", "codec bakeoff", run_codec_phase),
            ("P9_SYNTH", "curve fit and scorecard", run_synthesis_phase),
        ]
        for key, description, fn in stages:
            print(f"\n### {key} — {description} ###", file=sys.stderr)
            rc = _checkpointed_phase(key, fn)
            if rc != 0:
                print(f"[studio] stopped at {key} rc={rc}; rerun `resume` after fixing/draining",
                      file=sys.stderr)
                return rc
        print("\nGO COMPLETE — SCORECARD at reports/condense/SCORECARD.md; durable state at "
              f"{RUN_STATE}", file=sys.stderr)
        return 0
    finally:
        try:
            if RUN_PID.exists() and json.loads(RUN_PID.read_text()).get("pid") == os.getpid():
                RUN_PID.unlink()
        except Exception:
            pass
        if previous_lease_env is None:
            os.environ.pop(HEAVY_LEASE_FD_ENV, None)
        else:
            os.environ[HEAVY_LEASE_FD_ENV] = previous_lease_env
        _release_heavy_lease(heavy_lease)


def go_plan():
    """Dry overview of the whole GO program (run nothing heavy)."""
    plan()
    print("\n" + "=" * 78); print("FULL GO PROGRAM (studio_run.py go):")
    print("  P0E EFFICIENCY write the Beyond-FLOPS capability/byte/joule/goodput measurement contract")
    print("  P0 CODEC     codec_parallelism.py --catalog -> triage new codec designs before Rust build time")
    print("  P1 CONDENSE  safe full-model parents -> bit_floor_curve.jsonl; 32B gets a budget-bound defer")
    print("  P2 SUBBIT    safe full-model parents -> sub-bit campaign; 32B routes to shard-stream research")
    print("  P3 SPEC      default-skipped; requires tau>=2.5 oracle + measured one-pass verifier")
    print("  P4 FRONTIER  run_frontier_all() -> 100B+ research prize (" + "/".join(frontier_labels()) + ")")
    print("  P5 EVAL      eval_suite.py (capability + NIAH) + ctx_extend.py (YaRN long-ctx + KV-RAM + SSM moat)")
    print("  P6 BASELINE  bench_baselines.py -> wedge gate vs IQ1_S/IQ2/MLX-4bit at matched bpw")
    print("  P7 CLIFF     ramcliff_bench.py --all -> RAM-cliff tok/s + energy J/tok (headline + energy moat)")
    print("  P8 CODEC     codec_bakeoff.py -> STRAND vs QTIP/QuIP#/AQLM (the codec rank map)")
    print("  P9 SCORECARD scorecard.py -> the POPULATED competitive matrix (no WIN cell without a receipt)")
    print("  P3 remains gated until reports/condense/spec_oracle_gate.json records tau>=2.5 and a "
          "one-pass verifier; the existing EAGLE/n-gram paths are not scheduled by default.")


def _other_heavy_processes():
    """Fail-closed process inventory used by launch and unplug decisions."""
    download_probe = _download_activity()
    if not download_probe.get("ok"):
        return {
            "ok": False, "rows": [], "intentional_download_processes": [],
            "intentional_download_rss_gib": None,
            "error": "download PID inventory unavailable: "
                     + "; ".join(download_probe.get("errors", [])),
        }
    processing_activity = [
        row for row in download_probe.get("rows", [])
        if row.get("role") == "processing-queue"
    ]
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pcpu=,rss=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except Exception as exc:
        return {
            "ok": False, "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    parsed = []
    for line in out.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
            cpu, rss_kib = float(parts[2]), int(parts[3])
        except ValueError:
            continue
        parsed.append({
            "pid": pid, "ppid": ppid, "cpu_percent": cpu, "rss_kib": rss_kib,
            "command": parts[4],
        })
    parent_by_pid = {row["pid"]: row["ppid"] for row in parsed}
    intentional_download_roots = {
        int(row["pid"]) for row in download_probe.get("rows", [])
        if row.get("role") in (
            "download-queue", "download-caffeinate", "download-controller", "download-child"
        )
    }

    def has_download_ancestor(pid):
        seen = set()
        while pid > 1 and pid not in seen:
            if pid in intentional_download_roots:
                return True
            seen.add(pid)
            pid = parent_by_pid.get(pid, 0)
        return False

    rows = []
    intentional_downloads = []
    for item in parsed:
        pid = item["pid"]
        cpu = item["cpu_percent"]
        rss_kib = item["rss_kib"]
        command = item["command"]
        if pid == os.getpid() or "studio_run.py" in command or "ram_scheduler.py" in command:
            continue
        if pid in intentional_download_roots:
            intentional_downloads.append({
                "pid": pid,
                "ppid": item["ppid"],
                "cpu_percent": cpu,
                "rss_gib": round(rss_kib / (1024 * 1024), 3),
                "command": command[:300],
            })
            continue
        unregistered_download_descendant = has_download_ancestor(item["ppid"])
        unregistered_download = bool(re.search(
            r"(?:^|\s)(?:(?:\S*/)?hf\s+(?:download|cache\s+verify)|"
            r"(?:\S*/)?procure\.py(?:\s|$))",
            command,
        ))
        if (cpu >= 50.0 or rss_kib >= 4 * 1024 * 1024
                or unregistered_download or unregistered_download_descendant):
            rows.append({
                "pid": pid,
                "cpu_percent": cpu,
                "rss_gib": round(rss_kib / (1024 * 1024), 3),
                "command": command[:300],
                "unregistered_download": unregistered_download,
                "unregistered_download_descendant": unregistered_download_descendant,
            })
    return {
        "ok": True,
        "rows": sorted(
            rows, key=lambda row: (row["cpu_percent"], row["rss_gib"]), reverse=True
        )[:10],
        "intentional_download_processes": sorted(
            intentional_downloads,
            key=lambda row: (row["rss_gib"], row["cpu_percent"]), reverse=True,
        )[:25],
        "intentional_download_rss_gib": round(sum(
            row["rss_gib"] for row in intentional_downloads
        ), 3),
        "processing_activity": processing_activity,
        "error": None,
    }


def _download_activity():
    """Identity-safe inventory of queue -> procure -> HF activity.

    Stale state files are historical evidence, not process authority. A live PID is admitted only
    when ps observes it with the exact role command, its document has the pinned schema/label/path,
    its heartbeat is fresh, and controller/child ancestry is intact.
    """
    rows, errors, seen = [], [], set()
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except Exception as exc:
        return {"ok": False, "rows": [],
                "errors": [f"process inventory failed: {type(exc).__name__}: {exc}"]}
    processes = {}
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid, ppid, pgid = int(fields[0]), int(fields[1]), int(fields[2])
        except ValueError:
            continue
        processes[pid] = {"pid": pid, "ppid": ppid, "pgid": pgid, "command": fields[3]}

    def live_process(raw_pid, source, role):
        if raw_pid is None:
            return None
        try:
            pid = int(raw_pid)
            if pid <= 0:
                raise ValueError("PID must be positive")
        except (TypeError, ValueError) as exc:
            errors.append(f"{source}: invalid {role} PID {raw_pid!r}: {exc}")
            return None
        proc = processes.get(pid)
        if proc is not None:
            return proc
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except OSError as exc:
            errors.append(f"{source}: cannot probe {role} PID {pid}: {exc}")
            return None
        errors.append(f"{source}: live {role} PID {pid} was absent from ps inventory")
        return None

    def add(proc, role, source, label=None, command_pattern=None, expected_parent=None):
        if proc is None:
            return None
        if command_pattern and re.search(command_pattern, proc["command"]) is None:
            errors.append(
                f"{source}: PID {proc['pid']} command does not match {role}: {proc['command'][:300]}"
            )
            return None
        if expected_parent is not None and proc["ppid"] != int(expected_parent):
            errors.append(
                f"{source}: PID {proc['pid']} parent {proc['ppid']} != authorized "
                f"{role} parent {expected_parent}"
            )
            return None
        if proc["pid"] in seen or proc["pid"] == os.getpid():
            return proc
        seen.add(proc["pid"])
        rows.append({
            "pid": proc["pid"], "ppid": proc["ppid"], "pgid": proc["pgid"],
            "role": role, "label": label, "source": str(source),
            "command": proc["command"][:300],
        })
        return proc

    def read_doc(path):
        try:
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict):
                raise ValueError("record must be a JSON object")
            return doc
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            return None

    def heartbeat_fresh(doc, source, report=True):
        try:
            updated = datetime.datetime.fromisoformat(str(doc.get("updated_at")))
            if updated.tzinfo is None:
                raise ValueError("heartbeat has no timezone")
            age_s = (datetime.datetime.now(datetime.timezone.utc)
                     - updated.astimezone(datetime.timezone.utc)).total_seconds()
            max_age_s = max(
                60.0, float(os.environ.get("HAWKING_DOWNLOAD_HEARTBEAT_MAX_S", "180"))
            )
            if age_s < -5.0 or age_s > max_age_s:
                raise ValueError(f"heartbeat age {age_s:.1f}s outside 0..{max_age_s:.1f}s")
            return True
        except Exception as exc:
            if report:
                errors.append(
                    f"{source}: stale/unreadable heartbeat: {type(exc).__name__}: {exc}"
                )
            return False

    def descends_from(proc, root_pid):
        seen_ancestors = set()
        current = proc
        while current["pid"] not in seen_ancestors:
            if current["pid"] == int(root_pid):
                return True
            seen_ancestors.add(current["pid"])
            current = processes.get(current["ppid"])
            if current is None:
                return False
        return False

    queue_doc = read_doc(DOWNLOAD_QUEUE_PID) if DOWNLOAD_QUEUE_PID.exists() else None
    queue_proc = None
    if queue_doc is not None:
        candidate = live_process(queue_doc.get("pid"), DOWNLOAD_QUEUE_PID, "download-queue")
        if candidate is not None:
            queue_pattern = r"(?:^|\s)(?:\S*/)?download_queue\.py\s+run(?:\s|$)"
            if (queue_doc.get("schema") == "hawking.download_queue_pid.v1"
                    and re.search(queue_pattern, candidate["command"])):
                queue_proc = add(
                    candidate, "download-queue", DOWNLOAD_QUEUE_PID,
                    command_pattern=queue_pattern,
                )

    processing_doc = read_doc(PROCESSING_QUEUE_PID) if PROCESSING_QUEUE_PID.exists() else None
    if processing_doc is not None:
        candidate = live_process(
            processing_doc.get("pid"), PROCESSING_QUEUE_PID, "processing-queue"
        )
        if candidate is not None:
            processing_pattern = (
                r"(?:^|\s)(?:\S*/)?processing_queue\.py\s+run(?:\s|$)"
            )
            if (processing_doc.get("schema") == "hawking.processing_queue_pid.v1"
                    and re.search(processing_pattern, candidate["command"])):
                add(candidate, "processing-queue", PROCESSING_QUEUE_PID, "14B",
                    command_pattern=processing_pattern)

    queue_state = read_doc(DOWNLOAD_QUEUE_STATE_PATH) \
        if DOWNLOAD_QUEUE_STATE_PATH.exists() else None
    if queue_proc is not None:
        if queue_state is None:
            errors.append(f"{DOWNLOAD_QUEUE_STATE_PATH}: live queue has no state")
        elif queue_state.get("schema") != "hawking.download_queue.v1":
            errors.append(f"{DOWNLOAD_QUEUE_STATE_PATH}: queue state schema mismatch")
        else:
            heartbeat_fresh(queue_state, DOWNLOAD_QUEUE_STATE_PATH)
        caffeinate_pattern = (
            r"^caffeinate\s+-dimsu\s+.*(?:^|\s)(?:\S*/)?download_queue\.py\s+run(?:\s|$)"
        )
        for candidate in processes.values():
            if (candidate["ppid"] == queue_proc["pid"]
                    and re.search(caffeinate_pattern, candidate["command"])):
                add(candidate, "download-caffeinate", DOWNLOAD_QUEUE_PID,
                    command_pattern=caffeinate_pattern)

    authorized_controllers = {}
    if DOWNLOAD_STATE_DIR.exists():
        for path in sorted(DOWNLOAD_STATE_DIR.glob("*.pid.json")):
            doc = read_doc(path)
            if doc is None:
                continue
            candidate = live_process(doc.get("pid"), path, "detached-download-controller")
            if candidate is None:
                continue
            label = doc.get("label")
            binding = DOWNLOAD_PROCESS_BINDINGS.get(label)
            stored_cmd = doc.get("cmd")
            if (doc.get("schema") != "hawking.frontier_download_pid.v1"
                    or binding is None or not isinstance(stored_cmd, list)
                    or not any(str(part).endswith("procure.py") for part in stored_cmd)
                    or label not in [str(part) for part in stored_cmd]):
                errors.append(f"{path}: live detached controller identity is invalid")
                continue
            detached_pattern = (r"(?:^|\s)(?:\S*/)?procure\.py\s+"
                                + re.escape(label) + r"(?:\s|$)")
            if re.search(detached_pattern, candidate["command"]) is None:
                # Historical PID reuse: this record grants no authority, while the observed process
                # is still independently classified by the whole-machine inventory.
                continue
            proc = add(
                candidate, "unmanaged-download-controller", path, label,
                command_pattern=detached_pattern,
            )
            if proc is not None:
                authorized_controllers[proc["pid"]] = {"process": proc, "managed": False}

        for path in sorted(DOWNLOAD_STATE_DIR.glob("*.state.json")):
            doc = read_doc(path)
            if doc is None:
                continue
            label = doc.get("label")
            binding = DOWNLOAD_PROCESS_BINDINGS.get(label)
            active_status = doc.get("status") in {
                "starting", "running", "downloading", "verifying",
                "terminating_signal", "terminating_disk", "terminating_stall",
            }
            # Inactive/stale historical state is ignored before looking at its numeric PIDs. This
            # prevents PID reuse in old verified/failed/downloading files from granting authority
            # or creating a permanent false blocker.
            if (doc.get("schema") != "hawking.frontier_download_state.v1"
                    or binding is None or not active_status
                    or doc.get("hf_id") != (binding[0] if binding else None)
                    or os.path.abspath(str(doc.get("local_dir", "")))
                       != os.path.abspath(binding[1] if binding else "")):
                continue
            if not heartbeat_fresh(doc, path, report=False):
                continue
            controller = live_process(doc.get("pid"), path, "download-controller")
            child = live_process(doc.get("child_pid"), path, "download-child")
            if controller is None and child is None:
                continue
            managed = False
            if (queue_proc is not None and isinstance(queue_state, dict)
                    and queue_state.get("child_pid") == doc.get("pid")
                    and controller is not None
                    and descends_from(controller, queue_proc["pid"])):
                managed = True
            controller_pattern = (r"(?:^|\s)(?:\S*/)?procure\.py\s+"
                                  + re.escape(label) + r"(?:\s|$)")
            controller = add(
                controller,
                "download-controller" if managed else "unmanaged-download-controller",
                path, label,
                command_pattern=controller_pattern,
            )
            if controller is not None:
                # A foreground controller is authorized only by the active queue ancestry or a
                # validated detached-controller PID record (possibly its caffeinate parent).
                ancestor = controller
                authorized = managed or controller["pid"] in authorized_controllers
                inherited_managed = managed
                while not authorized and ancestor["ppid"] in processes:
                    ancestor = processes[ancestor["ppid"]]
                    authorized = ancestor["pid"] in authorized_controllers
                    if authorized:
                        inherited_managed = bool(
                            authorized_controllers[ancestor["pid"]]["managed"]
                        )
                if not authorized:
                    errors.append(f"{path}: live controller has no authorized queue/detached root")
                    continue
                managed = managed or inherited_managed
                authorized_controllers[controller["pid"]] = {
                    "process": controller, "managed": managed,
                }
            if child is not None:
                if controller is None or controller["pid"] not in authorized_controllers:
                    errors.append(f"{path}: live HF child has no authorized controller")
                    continue
                child_pattern = (
                    r"(?:^|\s)(?:\S*/)?hf\s+(?:download|cache\s+verify)\s+"
                    + re.escape(binding[0]) + r"(?:\s|$)"
                )
                child_managed = authorized_controllers[controller["pid"]]["managed"]
                add(child, "download-child" if child_managed else "unmanaged-download-child",
                    path, label, command_pattern=child_pattern,
                    expected_parent=controller["pid"])

    return {"ok": not errors, "rows": rows, "errors": errors}


def _run_activity():
    """Distinguish a cleanly absent/stale Studio PID from an unreadable activity probe."""
    if not RUN_PID.exists():
        return {"ok": True, "active": False, "pid": None, "error": None}
    try:
        info = json.loads(RUN_PID.read_text())
        if not isinstance(info, dict):
            raise ValueError("run PID record must be a JSON object")
        pid = int(info.get("pid"))
        if pid <= 0:
            raise ValueError("run PID must be positive")
    except Exception as exc:
        return {"ok": False, "active": None, "pid": None,
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"ok": True, "active": False, "pid": pid, "error": None}
    except OSError as exc:
        return {"ok": False, "active": None, "pid": pid,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "active": True, "pid": pid, "error": None}


def _wait_activity():
    """A waiter is not heavy work, but it is live automation that can start work later."""
    if not WAIT_PID.exists():
        return {"ok": True, "active": False, "pid": None, "error": None}
    try:
        info = json.loads(WAIT_PID.read_text())
        if not isinstance(info, dict):
            raise ValueError("wait PID record must be a JSON object")
        if (info.get("schema") != "hawking.studio_wait_pid.v1"
                or info.get("mode") != "waiting-admission"):
            raise ValueError("wait PID schema/mode mismatch")
        pid = int(info.get("pid"))
        if pid <= 0:
            raise ValueError("wait PID must be positive")
    except Exception as exc:
        return {"ok": False, "active": None, "pid": None,
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"ok": True, "active": False, "pid": pid, "error": None}
    except OSError as exc:
        return {"ok": False, "active": None, "pid": pid,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "active": True, "pid": pid, "error": None}


def _safe_to_unplug(resources, run_probe, wait_probe, process_probe, download_probe):
    """Require affirmative telemetry from every inventory before declaring machine-idle."""
    return bool(
        resources.get("ok")
        and run_probe.get("ok") and run_probe.get("active") is False
        and wait_probe.get("ok") and wait_probe.get("active") is False
        and process_probe.get("ok") and not process_probe.get("rows")
        and download_probe.get("ok") and not download_probe.get("rows")
    )


def _launch_gate(resources=None, other_heavy=None):
    """Return a fail-closed whole-machine launch decision for phone/remote operation."""
    if resources is None:
        try:
            from ram_scheduler import resource_snapshot
            resources = resource_snapshot()
        except Exception as exc:
            resources = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if other_heavy is None:
        process_probe = _other_heavy_processes()
    elif isinstance(other_heavy, dict) and "rows" in other_heavy:
        process_probe = other_heavy
    else:
        # Explicit row lists remain useful to deterministic unit tests and callers that already
        # possess a successful process inventory.
        process_probe = {"ok": True, "rows": list(other_heavy), "error": None}
    heavy_rows = process_probe.get("rows", [])

    blockers = []
    if not resources.get("ok"):
        blockers.append(f"resource snapshot unavailable: {resources.get('error', 'unknown error')}")
    else:
        pressure = int(resources.get("pressure_level", 4))
        swap_mb = float(resources.get("swap_used_mb", 0.0))
        usable_disk = float(resources.get("disk_usable_now_gb", 0.0))
        scratch = float(resources.get("scratch_reserve_gb", DEFAULT_HARDWARE.scratch_reserve_gb))
        if pressure > 1:
            blockers.append(f"memory pressure is {resources.get('pressure_name', pressure)}")
        if swap_mb >= 2048.0:
            blockers.append(f"swap use is {swap_mb:.0f} MB")
        if usable_disk < scratch:
            blockers.append(
                f"only {usable_disk:.1f} GB remains after the disk reserve; {scratch:.1f} GB scratch is required"
            )
    if not process_probe.get("ok"):
        blockers.append(
            f"process inventory unavailable: {process_probe.get('error', 'unknown error')}"
        )
    for row in heavy_rows:
        blockers.append(
            f"other heavy pid {row['pid']} ({row['cpu_percent']:.0f}% CPU, "
            f"{row['rss_gib']:.1f} GiB): {row['command']}"
        )
    intentional_download_rss = process_probe.get("intentional_download_rss_gib", 0.0)
    if isinstance(intentional_download_rss, (int, float)) and intentional_download_rss > 0:
        process_budget = float(getattr(
            DEFAULT_HARDWARE, "process_budget_gb", DEFAULT_HARDWARE.weight_budget_gb
        ))
        runnable_peaks = [
            float(gb) for label, model_dir, _params, gb, _solo, _role in LADDER
            if _ladder_parent_admission(label, model_dir)["ok"]
            and _full_model_scheduler_admission(float(gb), process_budget)["ok"]
        ]
        max_runnable_peak = max(runnable_peaks, default=0.0)
        overlap_margin_gb = 2.0
        if (max_runnable_peak + float(intentional_download_rss) + overlap_margin_gb
                > process_budget):
            blockers.append(
                f"monitored downloads use {float(intentional_download_rss):.1f}GiB; "
                f"with {max_runnable_peak:.0f}GB max runnable Studio peak and "
                f"{overlap_margin_gb:.0f}GB overlap margin this exceeds "
                f"the {process_budget:.0f}GB process budget"
            )
    return {
        "schema": "hawking.studio_launch_gate.v1",
        "generated_at": _now(),
        "ok": not blockers,
        "blockers": blockers,
        "resources": resources,
        "process_probe_ok": bool(process_probe.get("ok")),
        "process_probe_error": process_probe.get("error"),
        "other_heavy_processes": heavy_rows,
        "intentional_download_processes": process_probe.get(
            "intentional_download_processes", []
        ),
        "intentional_download_rss_gib": intentional_download_rss,
        "intentional_download_overlap_margin_gb": 2.0,
        "processing_activity": process_probe.get("processing_activity", []),
    }


def _record_launch_gate(gate):
    state = _load_run_state()
    state["launch_gate"] = gate
    state["updated_at"] = _now()
    _atomic_json(RUN_STATE, state)


def _enforce_launch_gate():
    gate = _launch_gate()
    _record_launch_gate(gate)
    if gate["ok"] or os.environ.get("HAWKING_STUDIO_ALLOW_CONCURRENT") == "1":
        return 0
    print("[studio] LAUNCH HELD — whole-machine safety gate is red", file=sys.stderr)
    for blocker in gate["blockers"]:
        print(f"  - {blocker}", file=sys.stderr)
    print("[studio] wait for a green `studio_run.py --status`; deliberate overlap requires "
          "HAWKING_STUDIO_ALLOW_CONCURRENT=1", file=sys.stderr)
    return 75


def status():
    try:
        from ram_scheduler import resource_snapshot
        resources = resource_snapshot()
    except Exception as exc:
        resources = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    state = _load_run_state()
    run_probe = _run_activity()
    wait_probe = _wait_activity()
    running = run_probe.get("active") is True
    waiting = wait_probe.get("active") is True
    active = running or waiting
    process_probe = _other_heavy_processes()
    download_probe = _download_activity()
    other_heavy = process_probe.get("rows", [])
    downloads = download_probe.get("rows", [])
    launch_gate = _launch_gate(resources, process_probe)
    payload = {
        "schema": "hawking.studio_status.v1",
        "generated_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "active": active,
        "pid": run_probe.get("pid") if running else wait_probe.get("pid"),
        "running": running,
        "waiting_admission": waiting,
        "run_probe_ok": bool(run_probe.get("ok")),
        "run_probe_error": run_probe.get("error"),
        "wait_probe_ok": bool(wait_probe.get("ok")),
        "wait_probe_error": wait_probe.get("error"),
        "drain_requested": _draining(),
        "hawking_drained": bool(
            run_probe.get("ok") and not running
            and wait_probe.get("ok") and not waiting
            and download_probe.get("ok") and not downloads
        ),
        "launch_ready": launch_gate["ok"],
        "launch_gate": launch_gate,
        "other_heavy_processes": other_heavy,
        "process_probe_ok": bool(process_probe.get("ok")),
        "process_probe_error": process_probe.get("error"),
        "download_activity": downloads,
        "download_probe_ok": bool(download_probe.get("ok")),
        "download_probe_errors": download_probe.get("errors", []),
        "downloads_drained": bool(download_probe.get("ok") and not downloads),
        "resources": resources,
        "run_state": state,
        "safe_to_unplug": _safe_to_unplug(
            resources, run_probe, wait_probe, process_probe, download_probe
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def drain():
    """Stop new work, let checkpoint-aware children terminate, fsync, then declare move safety."""
    _atomic_json(DRAIN_REQUEST, {"requested_at": _now(), "requested_by_pid": os.getpid()})
    timeout_s = float(os.environ.get("HAWKING_STUDIO_DRAIN_TIMEOUT_S", "300"))
    deadline = time.monotonic() + timeout_s
    first_download_probe = _download_activity()
    for row in first_download_probe.get("rows", []):
        if row["role"] not in ("download-queue", "download-controller", "processing-queue"):
            continue
        try:
            os.killpg(row["pid"], signal.SIGTERM)
        except OSError:
            try:
                os.kill(row["pid"], signal.SIGTERM)
            except OSError:
                pass
    run_probe = _run_activity()
    wait_probe = _wait_activity()
    active_pid = run_probe.get("pid") if run_probe.get("active") is True else None
    waiting_pid = wait_probe.get("pid") if wait_probe.get("active") is True else None
    downloads = []
    download_probe = first_download_probe
    while time.monotonic() < deadline:
        run_probe = _run_activity()
        wait_probe = _wait_activity()
        active_pid = run_probe.get("pid") if run_probe.get("active") is True else None
        waiting_pid = wait_probe.get("pid") if wait_probe.get("active") is True else None
        download_probe = _download_activity()
        downloads = download_probe.get("rows", [])
        if (run_probe.get("ok") and run_probe.get("active") is False
                and wait_probe.get("ok") and wait_probe.get("active") is False
                and download_probe.get("ok") and not downloads):
            break
        time.sleep(2)
    if (not run_probe.get("ok") or run_probe.get("active") is not False
            or not wait_probe.get("ok") or wait_probe.get("active") is not False
            or not download_probe.get("ok") or downloads):
        print(f"[studio] DRAIN TIMEOUT/PROBE FAILURE — Studio pid={active_pid} "
              f"waiter pid={waiting_pid} run_probe={run_probe} wait_probe={wait_probe} "
              f"downloads={downloads} "
              f"download_errors={download_probe.get('errors', [])}; "
              "NOT safe to unplug", file=sys.stderr)
        return 1
    try:
        os.sync()
    except AttributeError:
        subprocess.run(["sync"], check=False)
    process_probe = _other_heavy_processes()
    download_probe = _download_activity()
    other_heavy = process_probe.get("rows", [])
    downloads = download_probe.get("rows", [])
    global_safe = bool(
        process_probe.get("ok") and not other_heavy
        and download_probe.get("ok") and not downloads
    )
    state = _load_run_state()
    state["drain"] = {
        "status": "safe" if global_safe else "hawking-drained-other-work-active",
        "completed_at": _now(),
        "hawking_drained": True,
        "safe_to_unplug": global_safe,
        "other_heavy_processes": other_heavy,
        "process_probe_ok": bool(process_probe.get("ok")),
        "process_probe_error": process_probe.get("error"),
        "download_activity": downloads,
        "download_probe_ok": bool(download_probe.get("ok")),
        "download_probe_errors": download_probe.get("errors", []),
    }
    state["updated_at"] = _now()
    _atomic_json(RUN_STATE, state)
    if not global_safe:
        print("[studio] HAWKING DRAINED, but NOT globally safe to unplug: another heavy process is active",
              file=sys.stderr)
        for row in other_heavy:
            print(f"  pid={row['pid']} cpu={row['cpu_percent']:.0f}% rss={row['rss_gib']:.1f}GiB "
                  f"{row['command']}", file=sys.stderr)
        return 2
    print("[studio] SAFE TO UNPLUG — no Studio/heavy process is active and checkpoints were synced",
          file=sys.stderr)
    return 0


def resume():
    try:
        DRAIN_REQUEST.unlink()
    except FileNotFoundError:
        pass
    state = _load_run_state()
    state["resume"] = {"requested_at": _now(), "previous_drain_cleared": True}
    state["updated_at"] = _now()
    _atomic_json(RUN_STATE, state)
    return go()


def wait_resume():
    """Detached admission waiter: stay alive across a red gate, then run the current ladder.

    The waiter owns only a tiny singleton lock, never the heavy lease. Downloads and the 14B
    processor may therefore advance. A drain makes it exit cleanly; a green gate lets ``go`` take
    the heavy lease and execute. This is the autonomous handoff target after 14B processing.
    """
    WAIT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    wait_lock = open(WAIT_LOCK, "a+")
    try:
        fcntl.flock(wait_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        wait_lock.close()
        print("[studio] another detached admission waiter is active", file=sys.stderr)
        return 0
    try:
        old = _active_run_pid()
        if old is not None:
            print(f"[studio] another run/waiter is live pid={old}", file=sys.stderr)
            return 2
        try:
            DRAIN_REQUEST.unlink()
            _fsync_dir(DRAIN_REQUEST.parent)
        except FileNotFoundError:
            pass
        _atomic_json(WAIT_PID, {
            "schema": "hawking.studio_wait_pid.v1",
            "pid": os.getpid(), "started_at": _now(), "mode": "waiting-admission",
            "hardware_profile": DEFAULT_HARDWARE.name,
        })
        state = _load_run_state()
        state["resume"] = {"requested_at": _now(), "previous_drain_cleared": True,
                           "mode": "waiting-admission"}
        state["updated_at"] = _now()
        _atomic_json(RUN_STATE, state)
        poll_s = max(2.0, float(os.environ.get("HAWKING_STUDIO_WAIT_POLL_S", "30")))
        while True:
            if _draining():
                state = _load_run_state()
                state["waiting_admission"] = {
                    "status": "interrupted", "updated_at": _now(), "reason": "drain requested",
                }
                state["updated_at"] = _now()
                _atomic_json(RUN_STATE, state)
                return 130
            gate = _launch_gate()
            _record_launch_gate(gate)
            if gate["ok"] or os.environ.get("HAWKING_STUDIO_ALLOW_CONCURRENT") == "1":
                state = _load_run_state()
                state["waiting_admission"] = {
                    "status": "admitted", "updated_at": _now(), "gate": gate,
                }
                state["updated_at"] = _now()
                _atomic_json(RUN_STATE, state)
                rc = go()
                if rc not in (2, 75) or _draining():
                    return rc
                # A processor/lease owner may win the narrow race after the green inventory.
                # WAIT_PID remains published, so keep waiting without blocking RUN_PID ownership.
            else:
                state = _load_run_state()
                state["waiting_admission"] = {
                    "status": "waiting", "updated_at": _now(), "gate": gate,
                }
                state["updated_at"] = _now()
                _atomic_json(RUN_STATE, state)
            deadline = time.monotonic() + poll_s
            while time.monotonic() < deadline:
                if _draining():
                    break
                time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    finally:
        try:
            if WAIT_PID.exists() and json.loads(WAIT_PID.read_text()).get("pid") == os.getpid():
                WAIT_PID.unlink()
                _fsync_dir(WAIT_PID.parent)
        except Exception:
            pass
        fcntl.flock(wait_lock.fileno(), fcntl.LOCK_UN)
        wait_lock.close()


def start_background():
    """Start a detached, caffeinated admission waiter that survives app disconnects."""
    try:
        info = json.loads(RUN_PID.read_text())
        os.kill(int(info.get("pid")), 0)
        print(f"[studio] already active pid={info.get('pid')}", file=sys.stderr)
        return 0
    except Exception:
        pass
    wait_probe = _wait_activity()
    if not wait_probe.get("ok"):
        print(f"[studio] waiter state is unreadable: {wait_probe.get('error')}", file=sys.stderr)
        return 75
    if wait_probe.get("active"):
        print(f"[studio] admission waiter already active pid={wait_probe.get('pid')}",
              file=sys.stderr)
        return 0
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(RUN_LOG, "ab", buffering=0)
    python = sys.executable or "python3.12"
    cmd = [python, str(pathlib.Path(__file__).resolve()), "wait-resume"]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-dimsu", *cmd]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            cwd=ROOT)
    log.close()
    print(f"[studio] detached admission waiter pid={proc.pid}; log={RUN_LOG}; "
          "status: studio_run.py --status",
          file=sys.stderr)
    return 0


def selftest():
    global RUN_STATE, RUN_PID, WAIT_PID, DRAIN_REQUEST, LADDER, _BOOT_PHASE_EVIDENCE
    original = (RUN_STATE, RUN_PID, WAIT_PID, DRAIN_REQUEST, LADDER)
    original_ppl_text = os.environ.get("PPL_TEXT")
    try:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            RUN_STATE = root / "state.json"
            RUN_PID = root / "run.pid"
            WAIT_PID = root / "wait.pid"
            DRAIN_REQUEST = root / "drain.json"
            assert _wait_activity()["active"] is False
            _atomic_json(WAIT_PID, {
                "schema": "hawking.studio_wait_pid.v1", "pid": os.getpid(),
                "mode": "waiting-admission",
            })
            assert _wait_activity()["active"] is True
            WAIT_PID.unlink()
            boot_evidence = _freeze_phase_evidence()
            required_helpers = {
                "tools/condense/ram_scheduler.py", "tools/condense/arch_coverage.py",
                "tools/condense/auto_bits.py", "tools/condense/eval_suite.py",
                "tools/condense/ramcliff_bench.py",
                "tools/condense/frontier_common.py",
                "tools/condense/frontier_experiment_runner.py",
                "tools/condense/ladder.py",
            }
            assert required_helpers <= set(boot_evidence)
            _BOOT_PHASE_EVIDENCE = dict(boot_evidence)
            _BOOT_PHASE_EVIDENCE["tools/condense/ram_scheduler.py"] = "0" * 64
            assert not _phase_identity("P0_HELPER_DRIFT")["code_matches_runtime"], (
                "a directly executed helper changing after launch must invalidate runtime identity"
            )
            _BOOT_PHASE_EVIDENCE = boot_evidence
            assert _phase_identity("P0_HELPER_CURRENT")["code_matches_runtime"]
            process_budget = float(getattr(
                DEFAULT_HARDWARE, "process_budget_gb", DEFAULT_HARDWARE.weight_budget_gb
            ))
            previous_over_budget = os.environ.pop("HAWKING_STUDIO_ALLOW_OVER_BUDGET", None)
            try:
                assert not _full_model_scheduler_admission(85.0, process_budget)["ok"], (
                    "32B's estimated full Doctor peak must not silently cross the 78GB budget"
                )
                if (ROOT / "scratch/staging/qwen-32b.partial").is_dir():
                    admitted_32b = _ladder_parent_admission(
                        "32B", "scratch/staging/qwen-32b.partial"
                    )
                    assert admitted_32b["ok"], admitted_32b
                    assert len(admitted_32b["verified_marker_sha256"] or "") == 64
                    assert admitted_32b["model_fingerprint"]
                    assert not _ladder_parent_admission(
                        "32B", "scratch/qwen-7b"
                    )["ok"], "a verified marker must not authorize a different parent path"
                    deferred = root / "32B.defer.json"
                    _atomic_json(deferred, _deferred_model_document(
                        "32B", "studio", 85, process_budget, admitted_32b
                    ))
                    assert _deferred_model_valid("32B", "studio", deferred)
                    drifted_defer = json.loads(deferred.read_text())
                    drifted_defer["parent_admission"]["verified_marker_sha256"] = "0" * 64
                    _atomic_json(deferred, drifted_defer)
                    assert not _deferred_model_valid("32B", "studio", deferred)
            finally:
                if previous_over_budget is not None:
                    os.environ["HAWKING_STUDIO_ALLOW_OVER_BUDGET"] = previous_over_budget
            lease_path = root / "studio_heavy.lock"
            lease = _try_heavy_lease(lease_path)
            assert lease is not None
            assert _try_heavy_lease(lease_path) is None
            _release_heavy_lease(lease)
            lease = _try_heavy_lease(lease_path)
            assert lease is not None
            _release_heavy_lease(lease)
            # GO exports the locked FD and children retain it across exec. Simulate a supervisor
            # SIGKILL by closing only the parent's copy: admission must remain blocked until the
            # real child holder exits.
            lease = _try_heavy_lease(lease_path)
            previous_fd = os.environ.get(HEAVY_LEASE_FD_ENV)
            os.environ[HEAVY_LEASE_FD_ENV] = str(lease.fileno())
            keeper = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(0.25)"],
                pass_fds=inherited_lease_fds(os.environ),
            )
            lease.close()
            assert _try_heavy_lease(lease_path) is None
            keeper.wait(timeout=5)
            reacquired = _try_heavy_lease(lease_path)
            assert reacquired is not None
            _release_heavy_lease(reacquired)
            if previous_fd is None:
                os.environ.pop(HEAVY_LEASE_FD_ENV, None)
            else:
                os.environ[HEAVY_LEASE_FD_ENV] = previous_fd
            _set_phase("P0", "running", command=["synthetic"])
            assert _load_run_state()["phases"]["P0"]["status"] == "running"
            _set_phase("P0", "pass", returncode=0)
            assert _phase_done("P0")
            assert _checkpointed_phase("P0", lambda: 99) == 0
            stale = _load_run_state()
            stale["phases"]["P0"].pop("phase_identity")
            _atomic_json(RUN_STATE, stale)
            assert not _phase_done("P0"), "legacy status-only phase must never skip current code"
            _set_phase("P0", "pass", returncode=0)
            assert _phase_done("P0")
            _set_phase("P0_DRIFT", "running", command=["synthetic"])
            drifted = _load_run_state()
            drifted["phases"]["P0_DRIFT"]["phase_identity"]["identity_sha256"] = "0" * 64
            _atomic_json(RUN_STATE, drifted)
            _set_phase("P0_DRIFT", "pass", returncode=0)
            assert _load_run_state()["phases"]["P0_DRIFT"]["status"] == "failed"
            assert not _phase_done("P0_DRIFT")
            _atomic_json(DRAIN_REQUEST, {"requested_at": _now()})
            assert _checkpointed_phase("P1", lambda: 0) == 130
            assert _load_run_state()["phases"]["P1"]["status"] == "interrupted"
            normal = {
                "ok": True,
                "pressure_level": 1,
                "pressure_name": "normal",
                "swap_used_mb": 0.0,
                "disk_usable_now_gb": 400.0,
                "scratch_reserve_gb": 64.0,
            }
            assert _launch_gate(normal, [])["ok"]
            failed_process_probe = {
                "ok": False, "rows": [], "error": "synthetic ps failure",
            }
            failed_download_probe = {
                "ok": False, "rows": [], "errors": ["synthetic PID-state failure"],
            }
            inactive_run_probe = {
                "ok": True, "active": False, "pid": None, "error": None,
            }
            inactive_wait_probe = dict(inactive_run_probe)
            assert not _launch_gate(normal, failed_process_probe)["ok"]
            assert not _launch_gate(normal, {
                "ok": True, "rows": [], "error": None,
                "intentional_download_processes": [],
                "intentional_download_rss_gib": 100.0,
            })["ok"], "known downloads may overlap only inside the measured RAM budget"
            assert not _safe_to_unplug(
                normal, inactive_run_probe, inactive_wait_probe, failed_process_probe,
                {"ok": True, "rows": [], "errors": []},
            )
            assert not _safe_to_unplug(
                normal, inactive_run_probe, inactive_wait_probe,
                {"ok": True, "rows": [], "error": None}, failed_download_probe,
            )
            assert not _safe_to_unplug(
                normal, inactive_run_probe,
                {"ok": True, "active": True, "pid": 9, "error": None},
                {"ok": True, "rows": [], "error": None},
                {"ok": True, "rows": [], "errors": []},
            )
            assert not _launch_gate({**normal, "pressure_level": 2, "pressure_name": "yellow"}, [])["ok"]
            assert not _launch_gate(normal, [{
                "pid": 7, "cpu_percent": 99.0, "rss_gib": 1.0, "command": "synthetic-heavy"
            }])["ok"]

            audit = root / "studio_synthetic.jsonl"
            coverage = root / "studio_synthetic.coverage.json"
            model_dir = root / "synthetic-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"architectures":["SyntheticForCausalLM"]}\n', encoding="utf-8"
            )
            (model_dir / "tokenizer.json").write_text(
                '{"version":"1.0","model":{"type":"BPE"}}\n', encoding="utf-8"
            )
            eval_text = root / "eval.txt"
            eval_text.write_text("strict synthetic held-out evaluation corpus\n", encoding="utf-8")
            os.environ["PPL_TEXT"] = str(eval_text)
            LADDER = [*LADDER, (
                "synthetic", str(model_dir), 0.001, 1, False, "selftest",
            )]

            evidence_paths = (
                "vendor/strand-quant/target/release/quantize-model",
                "tools/condense/audit_ladder.py",
                "tools/condense/doctor.py",
                "tools/condense/multi_eval.py",
                "tools/condense/adapter_contract.py",
                "tools/condense/tripwire_gate.py",
            ) + (("scratch/calib_corpus.txt",)
                 if os.path.isfile("scratch/calib_corpus.txt") else ())

            def write_synthetic_identity(audit_path, lane):
                identity = {
                    "schema": "hawking.audit_identity.v1",
                    "recipe_version": AUDIT_RECIPE_VERSION,
                    "model": "synthetic",
                    "lane": lane,
                    "model_dir": str(model_dir),
                    "model_fingerprint": _model_stat_fingerprint_for_dir(model_dir),
                    "eval_text_path": str(eval_text),
                    "eval_text_sha256": _sha256_file(eval_text),
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
                        _sha256_file(os.environ["BAKE_ACTMEAN"])
                        if os.environ.get("BAKE_ACTMEAN")
                        and os.path.isfile(os.environ["BAKE_ACTMEAN"]) else None
                    ),
                    "strand_f32_metric": os.environ.get("STRAND_F32_METRIC"),
                    "strand_f32_search": os.environ.get("STRAND_F32_SEARCH"),
                    "doctor_grad_accum": 4,
                    "doctor_kd_topk": 64,
                    "doctor_target_regex": None,
                    "evidence_files": {
                        path: _sha256_file(path) for path in evidence_paths
                    },
                }
                _atomic_json(_audit_identity_path(audit_path), identity)
                return identity

            studio_identity = write_synthetic_identity(audit, "studio")

            def synthetic_doctor(eff_bpw):
                return {
                    "schema": "hawking.doctor_run_evidence.v1",
                    "complete": True,
                    "final": {"stopped_early": False},
                    "adapter_accounting": {
                        "schema": "hawking.doctor_adapter_accounting.v1",
                        "rank": 8,
                        "adapter_bytes": 32768,
                        "quantized_weights": 1048576,
                        "adapter_effective_bpw": 0.25,
                        "base_effective_bpw": eff_bpw - 0.25,
                        "total_effective_bpw": eff_bpw,
                    },
                }

            rows = []
            for config in CORE_RESEARCH_CONFIGS["studio"]:
                row = {"model": "synthetic", "config": config, "eff_bpw": 16.0,
                       "ppl": 10.0, "degr_pct": 0.0}
                if "+dr" in config:
                    row["doctor"] = synthetic_doctor(16.0)
                if config == "2-AWQ+dr":
                    row = {"model": "synthetic", "config": config, "error": "synthetic failure"}
                rows.append(row)
            audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
            assert not _write_core_coverage(
                "synthetic", "studio", audit_jsonl=audit, coverage_path=coverage
            )
            failed_coverage = json.loads(coverage.read_text())
            assert failed_coverage["error_configs"] == ["2-AWQ+dr"]
            with open(audit, "a") as handle:
                handle.write(json.dumps({
                    "model": "synthetic", "config": "2-AWQ+dr", "eff_bpw": 2.5,
                    "ppl": 10.5, "degr_pct": 5.0, "doctor": synthetic_doctor(2.5),
                }) + "\n")
            assert _write_core_coverage(
                "synthetic", "studio", audit_jsonl=audit, coverage_path=coverage
            )
            assert _core_coverage_valid(
                "synthetic", "studio", audit_jsonl=audit, coverage_path=coverage
            )
            with open(audit, "a") as handle:
                handle.write("{}\n")
            assert not _core_coverage_valid(
                "synthetic", "studio", audit_jsonl=audit, coverage_path=coverage
            )

            subbit_audit = root / "subbit_synthetic.jsonl"
            subbit_coverage = root / "subbit_synthetic.coverage.json"
            subbit_rows = [{
                "model": "synthetic", "config": "f16", "eff_bpw": 16.0,
                "ppl": 10.0, "degr_pct": 0.0,
            }]
            for config in CORE_RESEARCH_CONFIGS["subbit"]:
                if config == "f16":
                    continue
                identity = re.fullmatch(
                    r"vtq-k(\d+)-d(\d+)-b(\d+)-(frozen|learned)(?:\+dr-r(\d+))?",
                    config,
                )
                assert identity is not None
                bits, vec_dim, block_len = map(int, identity.group(1, 2, 3))
                rank = int(identity.group(5)) if identity.group(5) else None
                learned = identity.group(4) == "learned"
                quantized_weights = 1048576
                required_lut_bytes = 52 + (1 << (bits + 4)) * vec_dim * 4
                payload_bits = (quantized_weights // vec_dim) * bits
                trellis_side_bits = 100000
                logical_bits = payload_bits + trellis_side_bits + required_lut_bytes * 8
                base_bpw = logical_bits / quantized_weights
                eff_bpw = base_bpw + (0.25 if rank is not None else 0.0)
                accounting = {
                    "quantized_weights": quantized_weights,
                    "payload_bits": payload_bits,
                    "trellis_side_bits": trellis_side_bits,
                    "outlier_side_bits": 0,
                    "required_lut_bytes": required_lut_bytes,
                    "vector_lut_required_tensors": 1,
                    "vector_lut_required_weights": quantized_weights,
                    "learned_lut_selected_tensors": 1 if learned else 0,
                    "learned_lut_selected_weights": quantized_weights if learned else 0,
                    "learned_lut_selected_weight_fraction": 1.0 if learned else 0.0,
                    "logical_stream_bits_including_required_lut": logical_bits,
                    "oracle_effective_bpw": base_bpw,
                    "billing_scope": (
                        "logical_codec_stream_plus_required_lut_not_physical_packed_artifact"
                    ),
                    "method": (
                        "exact encoder payload/trellis-side/OUTL bits + required per-tensor Q12 LUT bytes"
                    ),
                }
                row = {
                    "model": "synthetic", "config": config, "eff_bpw": eff_bpw,
                    "ppl": 20.0, "degr_pct": 100.0,
                    "artifact_class": "reconstruction_oracle", "deployable": False,
                    "oracle": {
                        "schema": "hawking.vtq_reconstruction_oracle.v1",
                        "artifact_class": "reconstruction_oracle", "deployable": False,
                        "packed_artifact": None,
                        "recipe": {
                            "bits": bits, "l_bits": bits + 4,
                            "vec_dim": vec_dim, "block_len": block_len,
                            "learned_codebook": learned, "outlier_pct": 0.0,
                            "awq_alpha": 0.0, "rht": "cols",
                            "trellis_quality": False, "actmean": False,
                            "learned_codebook_iters": 50,
                            "learned_codebook_max_vectors": 16384,
                            "input_dtype": "torch.bfloat16",
                            "encode_workers": 1,
                            "source_fingerprint": _model_stat_fingerprint_for_dir(model_dir),
                            "quantizer_sha256": _sha256_file(
                                "vendor/strand-quant/target/release/quantize-model"
                            ),
                        },
                        "accounting": accounting,
                    },
                }
                row["oracle"]["recipe_sha256"] = hashlib.sha256(
                    json.dumps(
                        row["oracle"]["recipe"], sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                if rank is not None:
                    row["doctor"] = synthetic_doctor(eff_bpw)
                    row["doctor"]["adapter_accounting"]["rank"] = rank
                    accounting["oracle_plus_adapter_effective_bpw"] = eff_bpw
                assert _vtq_oracle_row_valid(config, row)
                no_selection = json.loads(json.dumps(row))
                if learned:
                    no_selection["oracle"]["accounting"]["learned_lut_selected_tensors"] = 0
                    assert not _vtq_oracle_row_valid(config, no_selection)
                else:
                    no_selection["oracle"]["accounting"]["required_lut_bytes"] = 12
                    assert not _vtq_oracle_row_valid(config, no_selection)
                subbit_rows.append(row)
            subbit_audit.write_text("".join(json.dumps(row) + "\n" for row in subbit_rows))
            subbit_identity = write_synthetic_identity(subbit_audit, "subbit")
            assert _write_core_coverage(
                "synthetic", "subbit", audit_jsonl=subbit_audit,
                coverage_path=subbit_coverage,
            )
            campaign = root / "synthetic-subbit-campaign.json"
            _write_subbit_campaign_receipt(
                "synthetic", subbit_audit, subbit_coverage, receipt_path=campaign,
            )
            assert _subbit_campaign_valid(
                campaign, "synthetic", subbit_audit, subbit_coverage,
            )
            # Coverage is hash-bound to the exact source/recipe identity, not merely the JSONL.
            wrong_identity = dict(subbit_identity)
            wrong_identity["lane"] = "studio"
            _atomic_json(_audit_identity_path(subbit_audit), wrong_identity)
            assert not _core_coverage_valid(
                "synthetic", "subbit", audit_jsonl=subbit_audit,
                coverage_path=subbit_coverage,
            )
            _atomic_json(_audit_identity_path(subbit_audit), subbit_identity)
            assert _write_core_coverage(
                "synthetic", "subbit", audit_jsonl=subbit_audit,
                coverage_path=subbit_coverage,
            )
            # A label/recipe mismatch is not coverage, even if its PPL and density are numeric.
            bad = dict(subbit_rows[1])
            bad["oracle"] = json.loads(json.dumps(bad["oracle"]))
            bad["oracle"]["recipe"]["vec_dim"] += 1
            with open(subbit_audit, "a") as handle:
                handle.write(json.dumps(bad) + "\n")
            assert not _write_core_coverage(
                "synthetic", "subbit", audit_jsonl=subbit_audit,
                coverage_path=subbit_coverage,
            )

            studio_receipt = root / "synthetic-studio-floor.json"
            subbit_receipt = root / "synthetic-subbit-floor.json"
            _atomic_json(studio_receipt, {
                "project": "hawking", "source_model": "synthetic (model)",
                "condensed_artifact": f"4-AWQ @ 4.0 eff-bpw ({audit})",
            })
            _atomic_json(subbit_receipt, {
                "project": "hawking", "source_model": "synthetic (model)",
                "condensed_artifact": "subbit @ 1.0 eff-bpw (wrong-lane.jsonl)",
            })
            assert _receipt_matches_lane(
                studio_receipt, "synthetic", "studio", audit_jsonl=audit
            )
            assert not _receipt_matches_lane(
                subbit_receipt, "synthetic", "subbit", audit_jsonl=audit
            )
            assert _lane_receipt_path("7B", "studio") != _lane_receipt_path("7B", "subbit")

            # A completion binds an immutable one-row floor JSONL and the identical unique row in
            # the shared curve. Adding another model must not invalidate this model's proof.
            floor_curve = root / "reports/cron/bit_floor_curve.jsonl"
            floor_row = {
                "schema": FLOOR_POINT_SCHEMA,
                "model": "synthetic",
                "params_b": 0.001,
                "floor_bpw": 4.0,
                "winning_config": "4-AWQ",
                "degr_pct": 1.0,
                "audit_jsonl": str(audit.resolve()),
                "audit_sha256": _sha256_file(audit),
            }
            locked_upsert_floor_row(floor_curve, "synthetic", floor_row)
            binding = create_floor_binding(root, "studio", "synthetic", floor_curve, audit)
            ok, problems = validate_floor_binding(
                binding, root, "studio", "synthetic", floor_curve, audit,
            )
            assert ok, problems
            assert binding["floor_row_sha256"] == canonical_row_sha256(floor_row)
            locked_upsert_floor_row(floor_curve, "other", {
                "schema": FLOOR_POINT_SCHEMA, "model": "other", "floor_bpw": 3.0,
            })
            ok, problems = validate_floor_binding(
                binding, root, "studio", "synthetic", floor_curve, audit,
            )
            assert ok, problems
            bad_binding = json.loads(json.dumps(binding))
            bad_binding["floor_row"]["floor_bpw"] = 0.5
            assert not validate_floor_binding(
                bad_binding, root, "studio", "synthetic", floor_curve, audit,
            )[0]
            point_path = root / binding["floor_jsonl"]
            point_path.write_text(json.dumps(floor_row) + "\n", encoding="utf-8")
            assert not validate_floor_binding(
                binding, root, "studio", "synthetic", floor_curve, audit,
            )[0], "noncanonical or hash-drifted proof bytes must fail"

            prefire = root / "prefire.json"
            _atomic_json(prefire, {
                "schema": "hawking.studio_prefire.v1", "status": "pass",
                "model": "synthetic", "params_b": 14.0, "device": PREFIRE_DEVICE,
                "target_bpw": 3.34,
                "auto_bits": {"advisor_only": True, "device": PREFIRE_DEVICE},
                "size_frontier": {"device": PREFIRE_DEVICE},
                "doctor_plan": {"target_bpw": 3.34},
            })
            assert _prefire_valid("synthetic", 14.0, prefire)
            assert not _prefire_valid("synthetic", 32.0, prefire)
    finally:
        RUN_STATE, RUN_PID, WAIT_PID, DRAIN_REQUEST, LADDER = original
        if original_ppl_text is None:
            os.environ.pop("PPL_TEXT", None)
        else:
            os.environ["PPL_TEXT"] = original_ppl_text
    print("studio_run.py selftest OK")
    return 0


def _standalone_heavy(fn):
    """Give direct --model/--run/frontier entry points the same inherited lease contract as GO."""
    if inherited_lease_fds(os.environ):
        return int(fn() or 0)
    gate_rc = _enforce_launch_gate()
    if gate_rc != 0:
        return gate_rc
    lease = _try_heavy_lease()
    if lease is None:
        print(f"[studio] heavy-work lease is held at {HEAVY_LOCK}; launch held", file=sys.stderr)
        return 75
    previous = os.environ.get(HEAVY_LEASE_FD_ENV)
    os.environ[HEAVY_LEASE_FD_ENV] = str(lease.fileno())
    try:
        return int(fn() or 0)
    finally:
        if previous is None:
            os.environ.pop(HEAVY_LEASE_FD_ENV, None)
        else:
            os.environ[HEAVY_LEASE_FD_ENV] = previous
        _release_heavy_lease(lease)


# Freeze the executable's launch evidence only after all module-level definitions have loaded.
# A detached process must not discover its supposed runtime identity lazily after its source or
# quantizer binary has been replaced on disk.
_freeze_phase_evidence()


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--plan"
    if a == "go":
        sys.exit(go())
    elif a == "resume":
        sys.exit(resume())
    elif a == "wait-resume":
        sys.exit(wait_resume())
    elif a == "start":
        sys.exit(start_background())
    elif a == "drain":
        sys.exit(drain())
    elif a == "--status":
        sys.exit(status())
    elif a == "--selftest":
        sys.exit(selftest())
    elif a == "--lease-selftest-child":
        sys.exit(_standalone_heavy(lambda: 0))
    elif a == "--go-plan":
        go_plan()
    elif a == "--plan":
        plan()
    elif a == "--model":
        # studio_run.py --model <label> [set]   (set = studio | subbit)
        sys.exit(_standalone_heavy(
            lambda: run_model(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "studio")
        ))
    elif a == "--run":
        sys.exit(_standalone_heavy(lambda: run_all("studio")))
    elif a == "--subbit":
        # studio_run.py --subbit <label>   (one model, sub-1-bit lane, with the SUBBIT-0 gate)
        sys.exit(_standalone_heavy(lambda: run_model(sys.argv[2], "subbit")))
    elif a == "--subbit-run":
        sys.exit(_standalone_heavy(lambda: run_all("subbit")))
    elif a == "--frontier":
        sys.exit(_standalone_heavy(lambda: run_frontier(sys.argv[2])))
    elif a == "--frontier-run":
        sys.exit(_standalone_heavy(run_frontier_all))
    else:
        print(__doc__)
