#!/usr/bin/env python3.12
"""procure.py — fastest-SOTA model procurement for the frontier run. Downloads are the ONE tedious
bottleneck left (~6 TB of frontier parents/checkpoints after GLM-5.2 + Kimi-K2), so this forces the
fastest path Hugging Face has and saturates whatever link you are on instead of being software-bottlenecked.

The stack, fastest-first (all already installed on the box: hf 1.13, hf_transfer 0.1.9, hf_xet):
  1. hf_xet          - the Xet content-addressed backend. Chunk-level dedup + parallel range gets;
                        the default for modern large repos, faster and more resumable than plain LFS.
  2. HF_HUB_ENABLE_HF_TRANSFER=1 - the Rust chunked transfer accelerator (parallel byte ranges per
                        file). Saturates a fast link where the pure-Python downloader caps low.
  3. --max-workers N - parallel FILES (shards) in flight at once. A frontier parent is 100s of
                        shards; pulling many concurrently overlaps per-file setup latency.
Together these turn "download bandwidth" from a software cap into a pure link cap. On gigabit fiber
(~125 MB/s) a 1.3 TB 671B lands in ~3 hours; on a sustained 300 MB/s link it lands in ~1.2 hours.
The default single-stream path would take far longer.

The procurement STRATEGY (not just speed):
  - CONDENSE targets (the whole ladder + frontier) prefer the highest-fidelity parent/checkpoint
    available to bake and to measure the floor against the parent baseline. GLM-5.2 is BF16; Kimi-K2.6
    currently stages from its public compressed-tensors checkpoint, so its receipts must say that.
  - The one real reducer for the giant frontier: STREAM-CONDENSE. Because block-wise condense
    processes one shard at a time, you can download shard N, bake it, delete the bf16 shard, fetch
    N+1 - so you never need the full 1.3 TB resident on disk AND the bake compute hides under
    download I/O wait. procure.py --stream emits that fused plan (the pager-free frontier makes this
    clean: the .tq is what stays, the bf16 is transient). This is the advanced path; the serve build
    must support shard-incremental bake for it to run unattended.

Usage:
  procure.py --check                       # confirm hf_transfer + hf_xet are active; print the env
  procure.py --selftest                    # synthetic telemetry checks, no network
  procure.py <label|hf_id> [--dir DIR]     # download one, fastest path (label resolves via FRONTIER)
  procure.py --all-frontier [--link-mbps 1000 | --link-mbs 300] [--efficiency 0.7]
                                            # full-fit view: source parents + .tq outputs + ETA
  procure.py --cycle-frontier [--scratch-gb 200] [--drop-outputs]
                                            # cycle view: keep .tq, release source after each bake
  procure.py --stream <label>              # emit the stream-condense fused plan (download+bake+delete)
  procure.py --cache-status                # list the project-local HF cache used by downloads
  procure.py --cache-prune [--yes]         # dry-run or apply detached-revision cache prune
  procure.py <label> --retries 2 --min-observed-mbs 80 --verify
                                            # resumable retry/backoff + optional hf cache verify
  procure.py <label> --progress-interval-s 60 --stall-timeout-s 900
                                            # live progress samples + stall termination
  procure.py --detach <label> [download options]
                                            # caffeinated, app-independent background download
  procure.py <label> --disk-free-floor-gb 150
                                            # checkpoint + stop before free disk falls below reserve
  procure.py <label> --no-diagnose-on-fail # skip automatic route/HF/network hints on failed attempts

Each real download appends an observed-throughput receipt with live progress samples, stall evidence,
route/HF/network diagnostics for bad attempts, and cache deltas to
reports/condense/frontier_downloads.jsonl. After a bake, use frontier_ops.py
status/ledger/release-source for the guarded lifecycle. This file downloads;
frontier_ops.py summarizes evidence and refuses unsafe source deletion.
An atomic heartbeat lives under reports/condense/download_state/ throughout each attempt; only a
successful download plus any requested cache verification publishes the separate `.verified.json`
completion marker that authorizes the lifecycle to advance from download to bake.
By default HF_HOME / HF_HUB_CACHE / HF_XET_CACHE are pinned under scratch/, so the Studio run does
not silently fill a global ~/.cache/huggingface directory. User-provided HF_* env vars still win.
"""
import sys, os, subprocess, shutil, importlib, json, time, datetime, selectors, tempfile, signal, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "condense"))
from studio_manifest import (
    DEFAULT_HARDWARE,
    FRONTIER_MODELS,
    FrontierModel,
    eta_hours as eta_hours_for,
    fmt_hours,
    frontier_by_label,
    storage_wave_plan,
    total_artifact_gb,
    total_download_gb,
)

# The frontier manifest lives in studio_manifest.py. The doctor ladder parents stay here because they
# are procurement conveniences rather than serve-frontier targets.
LADDER_PARENTS = {
    "14B": ("Qwen/Qwen2.5-14B-Instruct", 28.0, "scratch/qwen-14b"),
    "32B": ("Qwen/Qwen2.5-32B-Instruct", 64.0, "scratch/qwen-32b"),
    "72B": ("Qwen/Qwen2.5-72B-Instruct", 140.0, "scratch/qwen-72b"),
}
# The full turbo env. hf_transfer = Rust chunked accelerator (parallel byte ranges per FILE).
# hf_xet high-performance mode = more concurrent range gets + larger buffers (bigger memory use, more
# throughput). These are the sanctioned "go faster" knobs; an env var an older backend does not know
# is simply ignored, so setting them is harmless. The real ceiling is your PHYSICAL link + router.
HF_HOME_DIR = os.path.abspath(os.environ.get("HF_HOME", os.path.join(ROOT, "scratch", "hf-home")))
HF_HUB_CACHE_DIR = os.path.abspath(os.environ.get("HF_HUB_CACHE", os.path.join(ROOT, "scratch", "hf-cache", "hub")))
HF_XET_CACHE_DIR = os.path.abspath(os.environ.get("HF_XET_CACHE", os.path.join(ROOT, "scratch", "hf-cache", "xet")))

FAST_ENV = {
    "HF_HOME": HF_HOME_DIR,
    "HF_HUB_CACHE": HF_HUB_CACHE_DIR,
    "HF_XET_CACHE": HF_XET_CACHE_DIR,
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_XET_HIGH_PERFORMANCE": "1",              # hf_xet: max concurrency + buffers
    "HF_XET_NUM_CONCURRENT_RANGE_GETS": os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32"),
}
# --max-workers = parallel FILES (shards). A frontier parent is hundreds of shards, so more workers
# overlap per-file TLS/HTTP setup. Capped to keep the router's NAT table + CPU sane (see --check).
MAX_WORKERS = os.environ.get("HF_MAX_WORKERS", str(min(32, (os.cpu_count() or 8) * 4)))
DOWNLOAD_LOG = os.path.join("reports", "condense", "frontier_downloads.jsonl")
DOWNLOAD_STATE_DIR = os.path.join("reports", "condense", "download_state")
CACHE_RESERVE_GB = DEFAULT_HARDWARE.cache_reserve_gb
PROGRESS_INTERVAL_S = float(os.environ.get("HAWKING_PROCURE_PROGRESS_INTERVAL_S", "60"))
STALL_TIMEOUT_S = float(os.environ.get("HAWKING_PROCURE_STALL_TIMEOUT_S", "900"))
STALL_MIN_DELTA_MB = float(os.environ.get("HAWKING_PROCURE_STALL_MIN_DELTA_MB", "64"))
DISK_FREE_FLOOR_GB = float(os.environ.get("HAWKING_PROCURE_DISK_FREE_FLOOR_GB", "150"))
DOWNLOAD_STATE_SCHEMA = "hawking.frontier_download_state.v1"
DOWNLOAD_VERIFIED_SCHEMA = "hawking.frontier_download_verified.v1"


def _ensure_cache_dirs():
    for path in (HF_HOME_DIR, HF_HUB_CACHE_DIR, HF_XET_CACHE_DIR):
        os.makedirs(path, exist_ok=True)


def _cache_sizes_gb():
    return {
        "hf_home_gb": round(_path_size_gb(HF_HOME_DIR), 3),
        "hf_hub_cache_gb": round(_path_size_gb(HF_HUB_CACHE_DIR), 3),
        "hf_xet_cache_gb": round(_path_size_gb(HF_XET_CACHE_DIR), 3),
    }


def _have(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _safe_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(label)).strip("-") or "download"


def _fsync_dir(path):
    """Best-effort directory fsync so an atomic rename survives an abrupt power loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems do not allow directory fsync. The same-directory replace
        # still prevents readers from observing a partially written JSON document.
        pass


def _atomic_write_json(path, row):
    parent = os.path.dirname(os.path.abspath(path))
    parent_existed = os.path.isdir(parent)
    os.makedirs(parent, exist_ok=True)
    if not parent_existed:
        _fsync_dir(os.path.dirname(parent))
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(parent)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _durable_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    _fsync_dir(os.path.dirname(os.path.abspath(path)))


def _checkpoint_paths(label, state_dir=DOWNLOAD_STATE_DIR):
    stem = _safe_label(label)
    return (
        os.path.join(state_dir, f"{stem}.state.json"),
        os.path.join(state_dir, f"{stem}.verified.json"),
    )


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def _disk_free_gb(path=ROOT):
    """Free space on the filesystem that will hold `path`, in decimal GB."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            probe = ROOT
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free / 1e9
    except OSError:
        return shutil.disk_usage(ROOT).free / 1e9


def _verified_marker_valid(marker, *, label=None, hf_id=None, local_dir=None, require_verify=False):
    if not isinstance(marker, dict):
        return False
    if marker.get("schema") != DOWNLOAD_VERIFIED_SCHEMA:
        return False
    if marker.get("status") != "verified" or marker.get("verified_complete") is not True:
        return False
    if marker.get("hf_download_returncode") != 0:
        return False
    verification = marker.get("verification") if isinstance(marker.get("verification"), dict) else {}
    if verification.get("requested") and verification.get("returncode") != 0:
        return False
    if require_verify and (verification.get("requested") is not True or verification.get("returncode") != 0):
        return False
    if label is not None and marker.get("label") != label:
        return False
    if hf_id is not None and marker.get("hf_id") != hf_id:
        return False
    if local_dir is not None and os.path.abspath(str(marker.get("local_dir", ""))) != os.path.abspath(local_dir):
        return False
    return True


def _path_size_gb(path):
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e9
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            p = os.path.join(base, name)
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total / 1e9


def _append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _run_tee(cmd, env):
    """Run a long command while teeing stdout/stderr live and retaining a bounded output tail."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sel = selectors.DefaultSelector()
    streams = {}
    for pipe, stream in ((proc.stdout, sys.stdout.buffer), (proc.stderr, sys.stderr.buffer)):
        if pipe:
            os.set_blocking(pipe.fileno(), False)
            sel.register(pipe, selectors.EVENT_READ)
            streams[pipe] = stream
    tail = bytearray()
    limit = 8192
    while sel.get_map():
        for key, _ in sel.select(timeout=0.25):
            pipe = key.fileobj
            try:
                chunk = os.read(pipe.fileno(), 8192)
            except BlockingIOError:
                continue
            if not chunk:
                sel.unregister(pipe)
                continue
            streams[pipe].write(chunk)
            streams[pipe].flush()
            tail.extend(chunk)
            if len(tail) > limit:
                del tail[:-limit]
    return proc.wait(), tail.decode("utf-8", errors="replace")


def _progress_sample(local_dir, t0, previous=None):
    cache = _cache_sizes_gb()
    local_gb = round(_path_size_gb(local_dir), 3)
    tracked_total_gb = round(
        local_gb
        + cache["hf_home_gb"]
        + cache["hf_hub_cache_gb"]
        + cache["hf_xet_cache_gb"],
        3,
    )
    elapsed_s = max(time.monotonic() - t0, 0.0)
    sample = {
        "elapsed_s": round(elapsed_s, 3),
        "local_dir_gb": local_gb,
        "hf_home_gb": cache["hf_home_gb"],
        "hf_hub_cache_gb": cache["hf_hub_cache_gb"],
        "hf_xet_cache_gb": cache["hf_xet_cache_gb"],
        "tracked_total_gb": tracked_total_gb,
        "disk_free_gb": round(_disk_free_gb(local_dir), 3),
    }
    if previous:
        window_s = max(elapsed_s - float(previous.get("elapsed_s", 0.0)), 0.0)
        delta_gb = tracked_total_gb - float(previous.get("tracked_total_gb", 0.0))
        sample["window_s"] = round(window_s, 3)
        sample["delta_tracked_gb"] = round(delta_gb, 6)
        sample["window_mb_s"] = round((delta_gb * 1000.0 / window_s), 3) if window_s > 0 else None
    else:
        sample["window_s"] = None
        sample["delta_tracked_gb"] = None
        sample["window_mb_s"] = None
    return sample


def _progress_summary(samples, stall_min_delta_mb=STALL_MIN_DELTA_MB, stall_timeout_s=STALL_TIMEOUT_S,
                      terminated_for_stall=False, stall_reason=None,
                      terminated_for_disk=False, disk_reason=None, disk_free_floor_gb=None,
                      terminated_for_signal=False, signal_number=None, signal_reason=None):
    if not samples:
        return {
            "sample_count": 0,
            "stalled": False,
            "terminated_for_stall": bool(terminated_for_stall),
            "stall_reason": stall_reason,
            "terminated_for_disk": bool(terminated_for_disk),
            "disk_reason": disk_reason,
            "disk_free_floor_gb": disk_free_floor_gb,
            "terminated_for_signal": bool(terminated_for_signal),
            "signal_number": signal_number,
            "signal_reason": signal_reason,
        }
    no_progress_s = 0.0
    longest_no_progress_s = 0.0
    for sample in samples[1:]:
        delta_mb = max(float(sample.get("delta_tracked_gb") or 0.0) * 1000.0, 0.0)
        window_s = max(float(sample.get("window_s") or 0.0), 0.0)
        if delta_mb >= float(stall_min_delta_mb or 0.0):
            no_progress_s = 0.0
        else:
            no_progress_s += window_s
            longest_no_progress_s = max(longest_no_progress_s, no_progress_s)
    first = samples[0]
    last = samples[-1]
    duration_s = max(float(last.get("elapsed_s") or 0.0) - float(first.get("elapsed_s") or 0.0), 0.0)
    delta_gb = max(float(last.get("tracked_total_gb") or 0.0) - float(first.get("tracked_total_gb") or 0.0), 0.0)
    stalled = bool(stall_timeout_s and longest_no_progress_s >= float(stall_timeout_s))
    return {
        "sample_count": len(samples),
        "first_elapsed_s": first.get("elapsed_s"),
        "last_elapsed_s": last.get("elapsed_s"),
        "tracked_total_gb_first": first.get("tracked_total_gb"),
        "tracked_total_gb_last": last.get("tracked_total_gb"),
        "delta_tracked_gb": round(delta_gb, 6),
        "average_tracked_mb_s": round((delta_gb * 1000.0 / duration_s), 3) if duration_s > 0 else None,
        "longest_no_progress_s": round(longest_no_progress_s, 3),
        "stall_min_delta_mb": float(stall_min_delta_mb or 0.0),
        "stall_timeout_s": float(stall_timeout_s or 0.0),
        "stalled": stalled,
        "terminated_for_stall": bool(terminated_for_stall),
        "stall_reason": stall_reason,
        "last_window_mb_s": last.get("window_mb_s"),
        "disk_free_gb_first": first.get("disk_free_gb"),
        "disk_free_gb_last": last.get("disk_free_gb"),
        "disk_free_gb_min": min(
            (float(s["disk_free_gb"]) for s in samples if s.get("disk_free_gb") is not None),
            default=None,
        ),
        "disk_free_floor_gb": disk_free_floor_gb,
        "terminated_for_disk": bool(terminated_for_disk),
        "disk_reason": disk_reason,
        "terminated_for_signal": bool(terminated_for_signal),
        "signal_number": signal_number,
        "signal_reason": signal_reason,
    }


def _run_tee_monitored(cmd, env, local_dir, progress_interval_s=PROGRESS_INTERVAL_S,
                       stall_timeout_s=STALL_TIMEOUT_S, stall_min_delta_mb=STALL_MIN_DELTA_MB,
                       disk_free_floor_gb=DISK_FREE_FLOOR_GB, status_callback=None):
    """Run a long command with live output and durable progress/termination callbacks.

    SIGINT/SIGTERM are converted into a graceful child termination so Hugging Face's
    partial files remain resumable. The caller records the final interrupted state.
    """
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sel = selectors.DefaultSelector()
    streams = {}
    for pipe, stream in ((proc.stdout, sys.stdout.buffer), (proc.stderr, sys.stderr.buffer)):
        if pipe:
            os.set_blocking(pipe.fileno(), False)
            sel.register(pipe, selectors.EVENT_READ)
            streams[pipe] = stream

    tail = bytearray()
    limit = 8192
    t0 = time.monotonic()
    samples = []
    last_sample = _progress_sample(local_dir, t0)
    samples.append(last_sample)
    last_sample_at = time.monotonic()
    no_progress_s = 0.0
    terminated_for_stall = False
    stall_reason = None
    terminated_for_disk = False
    disk_reason = None
    terminated_for_signal = False
    signal_number = None
    signal_reason = None
    termination_started_at = None

    def notify(status, sample=None, **extra):
        if status_callback is None:
            return
        try:
            status_callback(status, sample, child_pid=proc.pid, **extra)
        except Exception as e:
            print(f"[procure] warning: download heartbeat update failed: {e}", file=sys.stderr)

    received_signal = None
    old_handlers = {}

    def request_stop(signum, _frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass

    notify("running", last_sample)
    try:
        while sel.get_map():
            for key, _ in sel.select(timeout=0.25):
                pipe = key.fileobj
                try:
                    chunk = os.read(pipe.fileno(), 8192)
                except BlockingIOError:
                    continue
                if not chunk:
                    sel.unregister(pipe)
                    continue
                streams[pipe].write(chunk)
                streams[pipe].flush()
                tail.extend(chunk)
                if len(tail) > limit:
                    del tail[:-limit]

            now = time.monotonic()
            if received_signal is not None and termination_started_at is None and proc.poll() is None:
                terminated_for_signal = True
                signal_number = int(received_signal)
                try:
                    signal_name = signal.Signals(received_signal).name
                except ValueError:
                    signal_name = str(received_signal)
                signal_reason = f"received {signal_name}; partial download is resumable"
                termination_started_at = now
                print(f"[procure] INTERRUPT: {signal_reason}; terminating command", file=sys.stderr)
                notify("terminating_signal", last_sample, reason=signal_reason,
                       signal_number=signal_number)
                proc.terminate()

            sample_interval_s = progress_interval_s if progress_interval_s and progress_interval_s > 0 else 60.0
            if proc.poll() is None and now - last_sample_at >= sample_interval_s:
                sample = _progress_sample(local_dir, t0, last_sample)
                samples.append(sample)
                last_sample = sample
                last_sample_at = now
                delta_mb = max(float(sample.get("delta_tracked_gb") or 0.0) * 1000.0, 0.0)
                window_s = max(float(sample.get("window_s") or 0.0), 0.0)
                if delta_mb >= float(stall_min_delta_mb or 0.0):
                    no_progress_s = 0.0
                else:
                    no_progress_s += window_s
                rate = sample.get("window_mb_s")
                rate_txt = f"{rate:.1f} MB/s" if isinstance(rate, (int, float)) else "n/a"
                print(
                    f"[procure] progress elapsed={sample['elapsed_s']:.0f}s "
                    f"local={sample['local_dir_gb']:.1f}GB "
                    f"tracked={sample['tracked_total_gb']:.1f}GB "
                    f"disk-free={sample['disk_free_gb']:.1f}GB "
                    f"window={rate_txt} no-progress={no_progress_s:.0f}s",
                    file=sys.stderr,
                )
                notify("running", sample)
                if (termination_started_at is None and disk_free_floor_gb
                        and sample["disk_free_gb"] < float(disk_free_floor_gb)):
                    terminated_for_disk = True
                    disk_reason = (
                        f"disk free {sample['disk_free_gb']:.1f} GB below "
                        f"safety floor {float(disk_free_floor_gb):.1f} GB"
                    )
                    termination_started_at = now
                    print(f"[procure] DISK: {disk_reason}; terminating command", file=sys.stderr)
                    notify("terminating_disk", sample, reason=disk_reason)
                    proc.terminate()
                elif (termination_started_at is None and stall_timeout_s and stall_timeout_s > 0
                      and no_progress_s >= stall_timeout_s):
                    terminated_for_stall = True
                    stall_reason = (
                        f"no tracked local/cache growth >= {stall_min_delta_mb:.1f} MB "
                        f"for {no_progress_s:.0f}s"
                    )
                    termination_started_at = now
                    print(f"[procure] STALL: {stall_reason}; terminating command", file=sys.stderr)
                    notify("terminating_stall", sample, reason=stall_reason)
                    proc.terminate()

            if (termination_started_at is not None and proc.poll() is None
                    and now - termination_started_at > 30):
                print("[procure] terminate timed out; killing command", file=sys.stderr)
                proc.kill()
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    if termination_started_at is not None and proc.poll() is None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[procure] child still live after pipe close; killing command", file=sys.stderr)
            proc.kill()
    rc = proc.wait()
    final_sample = _progress_sample(local_dir, t0, last_sample)
    if final_sample["elapsed_s"] != last_sample["elapsed_s"]:
        samples.append(final_sample)
    progress = _progress_summary(
        samples,
        stall_min_delta_mb=stall_min_delta_mb,
        stall_timeout_s=stall_timeout_s,
        terminated_for_stall=terminated_for_stall,
        stall_reason=stall_reason,
        terminated_for_disk=terminated_for_disk,
        disk_reason=disk_reason,
        disk_free_floor_gb=float(disk_free_floor_gb) if disk_free_floor_gb is not None else None,
        terminated_for_signal=terminated_for_signal,
        signal_number=signal_number,
        signal_reason=signal_reason,
    )
    if terminated_for_signal:
        rc = 128 + int(signal_number or signal.SIGTERM)
    elif terminated_for_disk:
        rc = 75
    elif terminated_for_stall:
        rc = 124
    return rc, tail.decode("utf-8", errors="replace"), progress, samples


def _record_download(row, path=DOWNLOAD_LOG):
    _append_jsonl(path, row)
    print(f"[procure] download telemetry -> {path}", file=sys.stderr)


def _workers_for_attempt(base_workers, attempt):
    try:
        base = int(base_workers)
    except Exception:
        base = 1
    return str(max(1, base // (2 ** max(attempt - 1, 0))))


def _link_layer():
    """Best-effort: is the default route over WiFi or ethernet on macOS? WiFi is the #1 reason a
    download undershoots the speedtest (the radio can't sustain the wired link + adds jitter)."""
    try:
        dev = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True).stdout
        iface = next((l.split(":")[1].strip() for l in dev.splitlines() if "interface:" in l), None)
        if not iface:
            return "unknown"
        ports = subprocess.run(["networksetup", "-listallhardwareports"], capture_output=True, text=True).stdout
        blocks = ports.split("Hardware Port:")
        for b in blocks:
            if f"Device: {iface}" in b:
                name = b.splitlines()[0].strip()
                is_wifi = "wi-fi" in name.lower() or "airport" in name.lower()
                return f"{iface} ({name}) -> {'WiFi (fine IF it delivers your full link; measure with a real download, ethernet only matters at gigabit+)' if is_wifi else 'wired ethernet (good)'}"
        return f"{iface} (unknown type)"
    except Exception:
        return "unknown"


def _short_run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout_tail": (p.stdout or "")[-2000:],
            "stderr_tail": (p.stderr or "")[-2000:],
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "returncode": 127,
            "error": f"{type(e).__name__}: {e}",
        }


def _network_probe(url="https://huggingface.co", timeout=8):
    t0 = time.monotonic()
    out = {"url": url, "timeout_s": timeout}
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out.update({
                "ok": True,
                "status": getattr(resp, "status", None),
                "elapsed_s": round(time.monotonic() - t0, 3),
            })
    except Exception as e:
        out.update({
            "ok": False,
            "elapsed_s": round(time.monotonic() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
        })
    return out


def _dns_probe(host="huggingface.co"):
    try:
        import socket

        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        addrs = sorted({item[4][0] for item in infos})
        return {"host": host, "ok": bool(addrs), "addresses": addrs[:8], "address_count": len(addrs)}
    except Exception as e:
        return {"host": host, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _diagnostic_recommendations(reason, route, observed_mb_s, min_observed_mbs, progress_summary,
                                xet_ok, transfer_ok, network_ok):
    recs = []
    route_l = (route or "").lower()
    if "wi-fi" in route_l or "wifi" in route_l:
        recs.append("Use wired ethernet into the router for frontier downloads; WiFi commonly under-runs speed tests.")
    if not xet_ok:
        recs.append("Install/enable hf_xet; modern large repos rely on it for resumable parallel range gets.")
    if not transfer_ok:
        recs.append("Install/enable hf_transfer; it is the Rust chunked transfer path.")
    if network_ok is False:
        recs.append("Hugging Face reachability probe failed; check VPN, DNS, router, and ISP path before retrying.")
    if progress_summary and progress_summary.get("terminated_for_stall"):
        recs.append("Retry resumes the local dir and halves workers; repeated stalls suggest lowering HF_MAX_WORKERS to 8-16.")
    if (observed_mb_s is not None and min_observed_mbs
            and observed_mb_s < min_observed_mbs):
        recs.append("Observed throughput under-ran the floor; retry will use fewer workers to reduce router/NAT pressure.")
    if reason and "returncode" in reason and not recs:
        recs.append("Inspect output_tail and retry once; Hugging Face downloads are resumable from the same local dir.")
    return recs


def _download_diagnostics(reason, workers, observed_mb_s=None, min_observed_mbs=0.0,
                          progress_summary=None, probe_network=True):
    route = _link_layer()
    xet_ok = _have("hf_xet")
    transfer_ok = _have("hf_transfer")
    network = _network_probe() if probe_network else {"ok": None, "skipped": True}
    dns = _dns_probe() if probe_network else {"ok": None, "skipped": True}
    data = {
        "generated_at": _now(),
        "reason": reason,
        "workers": workers,
        "max_workers_config": MAX_WORKERS,
        "route": route,
        "hf_version": _short_run(["hf", "--version"], timeout=10),
        "accelerators": {
            "hf_xet": xet_ok,
            "hf_transfer": transfer_ok,
        },
        "network_probe": network,
        "dns_probe": dns,
        "cache_sizes_gb": _cache_sizes_gb(),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 3),
    }
    data["recommendations"] = _diagnostic_recommendations(
        reason,
        route,
        observed_mb_s,
        min_observed_mbs,
        progress_summary if isinstance(progress_summary, dict) else {},
        xet_ok,
        transfer_ok,
        network.get("ok") if isinstance(network, dict) else None,
    )
    return data


def check():
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    xet, xfer = _have("hf_xet"), _have("hf_transfer")
    _ensure_cache_dirs()
    cache_sizes = _cache_sizes_gb()
    print("--- software path (turbo env) ---", file=sys.stderr)
    print(f"hf CLI          : {hf or 'MISSING (pip install huggingface_hub[cli])'}", file=sys.stderr)
    print(f"hf_xet backend  : {'ACTIVE (Xet dedup + parallel range gets)' if xet else 'MISSING (pip install hf_xet)'}", file=sys.stderr)
    print(f"hf_transfer     : {'ACTIVE (Rust chunked accelerator)' if xfer else 'MISSING (pip install hf_transfer)'}", file=sys.stderr)
    print(f"turbo env       : {' '.join(f'{k}={v}' for k, v in FAST_ENV.items())}  --max-workers {MAX_WORKERS}", file=sys.stderr)
    print(f"cache dirs      : HF_HOME={HF_HOME_DIR} ({cache_sizes['hf_home_gb']:.1f} GB), "
          f"HF_HUB_CACHE={HF_HUB_CACHE_DIR} ({cache_sizes['hf_hub_cache_gb']:.1f} GB), "
          f"HF_XET_CACHE={HF_XET_CACHE_DIR} ({cache_sizes['hf_xet_cache_gb']:.1f} GB)",
          file=sys.stderr)
    ok = bool(hf) and xet and xfer
    print(f"=> software path is {'FASTEST-SOTA (link-bound, not software-bound)' if ok else 'DEGRADED - install the missing piece above'}", file=sys.stderr)
    print("\n--- PHYSICAL layer (where the real gap usually is) ---", file=sys.stderr)
    print(f"default route   : {_link_layer()}", file=sys.stderr)
    print("bandwidth checklist, biggest win first:", file=sys.stderr)
    print("  1. WIRED ETHERNET into the router (the Studio has a 10GbE port) - #1 real-world win over WiFi.", file=sys.stderr)
    print("  2. Plug into the ROUTER, not a switch/extender; short cable; Cat6+.", file=sys.stderr)
    print("  3. No VPN during downloads (adds routing + a single-tunnel bottleneck).", file=sys.stderr)
    print("  4. Nothing else heavy on the link (a 4K stream or a game update steals the pipe).", file=sys.stderr)
    print("  5. If the router is old/cheap it can cap under many connections - see the 'crash' note below.", file=sys.stderr)
    print("# NOTE: --max-workers is capped at 32 on purpose. Hundreds of connections can exhaust a home", file=sys.stderr)
    print("# router's NAT table / CPU and drop OTHER devices (or reboot a cheap router) - that is", file=sys.stderr)
    print("# saturation, not a exploit. 16-32 workers saturates bandwidth without destabilizing the LAN.", file=sys.stderr)
    return ok


def _resolve(label_or_id):
    spec = frontier_by_label(label_or_id)
    if spec:
        return spec
    if label_or_id in LADDER_PARENTS:
        hf_id, gb, dirn = LADDER_PARENTS[label_or_id]
        return FrontierModel(label_or_id, hf_id, dirn, 0.0, None, 0.0, False,
                             "ladder-parent", gb, "bf16 parent")
    # a raw HF id
    dirn = "scratch/" + label_or_id.split("/")[-1].lower()
    return FrontierModel(label_or_id, label_or_id, dirn, 0.0, None, 0.0, False,
                         "raw-hf", 0.0, "unknown")


def download(label_or_id, dir_override=None, dry=False, retries=0, min_observed_mbs=0.0, verify=False,
             progress_interval_s=PROGRESS_INTERVAL_S, stall_timeout_s=STALL_TIMEOUT_S,
             stall_min_delta_mb=STALL_MIN_DELTA_MB, diagnose_on_fail=True, network_diagnose=True,
             disk_free_floor_gb=DISK_FREE_FLOOR_GB, state_dir=DOWNLOAD_STATE_DIR,
             download_log=DOWNLOAD_LOG):
    spec = _resolve(label_or_id)
    dirn = dir_override or spec.local_dir
    _ensure_cache_dirs()
    env = {**os.environ, **FAST_ENV}
    size = f"  (~{spec.download_gb:.0f} GB {spec.source_kind})" if spec.download_gb else ""
    print(f"[procure] {spec.label} -> {dirn}{size}", file=sys.stderr)
    if dry:
        cmd = [
            "hf", "download", spec.hf_id,
            "--local-dir", dirn,
            "--cache-dir", HF_HUB_CACHE_DIR,
            "--max-workers", MAX_WORKERS,
        ]
        print(f"[procure] HF_HUB_ENABLE_HF_TRANSFER=1 {' '.join(cmd)}", file=sys.stderr)
        return 0

    attempts = max(1, int(retries) + 1)
    min_observed_mbs = float(min_observed_mbs or 0.0)
    disk_free_floor_gb = float(disk_free_floor_gb or 0.0)
    state_path, verified_path = _checkpoint_paths(spec.label, state_dir)
    _durable_unlink(verified_path)
    state_doc = {
        "schema": DOWNLOAD_STATE_SCHEMA,
        "label": spec.label,
        "hf_id": spec.hf_id,
        "local_dir": dirn,
        "source_kind": spec.source_kind,
        "manifest_gb": spec.download_gb,
        "started_at": _now(),
        "updated_at": _now(),
        "status": "starting",
        "phase": "download",
        "pid": os.getpid(),
        "attempt_count": attempts,
        "verify_requested": bool(verify),
        "disk_free_floor_gb": disk_free_floor_gb,
        "resumable": True,
        "verified_marker": verified_path,
        "git_commit": _git_commit(),
    }

    def checkpoint(status, **updates):
        state_doc.update(updates)
        state_doc["status"] = status
        state_doc["updated_at"] = _now()
        _atomic_write_json(state_path, state_doc)

    checkpoint("starting", disk_free_gb=round(_disk_free_gb(dirn), 3))
    final_rc = 1
    for attempt in range(1, attempts + 1):
        workers = _workers_for_attempt(MAX_WORKERS, attempt)
        cmd = [
            "hf", "download", spec.hf_id,
            "--local-dir", dirn,
            "--cache-dir", HF_HUB_CACHE_DIR,
            "--max-workers", workers,
        ]
        print(f"[procure] attempt {attempt}/{attempts}: HF_HUB_ENABLE_HF_TRANSFER=1 "
              f"{' '.join(cmd)}", file=sys.stderr)
        already_populated = os.path.isdir(dirn) and os.listdir(dirn)
        if already_populated:
            print(f"[procure] {dirn} already populated - hf resumes/skips complete shards", file=sys.stderr)
        before_gb = _path_size_gb(dirn)
        cache_before = _cache_sizes_gb()
        started_at = _now()
        t0 = time.monotonic()
        progress_summary = None
        progress_samples = []
        checkpoint(
            "downloading",
            phase="download",
            attempt=attempt,
            max_workers=workers,
            cmd=cmd,
            disk_free_gb=round(_disk_free_gb(dirn), 3),
            already_populated_at_start=bool(already_populated),
        )

        def download_heartbeat(status, sample, **extra):
            heartbeat_status = "downloading" if status == "running" else status
            checkpoint(
                heartbeat_status,
                phase="download",
                attempt=attempt,
                progress=sample,
                disk_free_gb=(sample or {}).get("disk_free_gb", round(_disk_free_gb(dirn), 3)),
                **extra,
            )

        disk_at_start = _disk_free_gb(dirn)
        remaining_manifest_gb = max(float(spec.download_gb or 0.0) - before_gb, 0.0)
        required_free_gb = disk_free_floor_gb + remaining_manifest_gb
        if disk_free_floor_gb > 0 and disk_at_start < required_free_gb:
            disk_reason = (
                f"disk free {disk_at_start:.1f} GB below predicted remaining download + safety "
                f"reserve {required_free_gb:.1f} GB ({remaining_manifest_gb:.1f} + "
                f"{disk_free_floor_gb:.1f})"
            )
            sample = _progress_sample(dirn, t0)
            progress_samples = [sample]
            progress_summary = _progress_summary(
                progress_samples,
                stall_min_delta_mb=stall_min_delta_mb,
                stall_timeout_s=stall_timeout_s,
                terminated_for_disk=True,
                disk_reason=disk_reason,
                disk_free_floor_gb=disk_free_floor_gb,
            )
            download_rc, tail = 75, disk_reason
            checkpoint("blocked_disk", phase="download", attempt=attempt, reason=disk_reason,
                       progress=sample, disk_free_gb=round(disk_at_start, 3),
                       remaining_manifest_gb=round(remaining_manifest_gb, 3),
                       required_free_gb=round(required_free_gb, 3))
            print(f"[procure] DISK: {disk_reason}; download not started", file=sys.stderr)
        else:
            try:
                download_rc, tail, progress_summary, progress_samples = _run_tee_monitored(
                    cmd,
                    env,
                    dirn,
                    progress_interval_s=progress_interval_s,
                    stall_timeout_s=stall_timeout_s,
                    stall_min_delta_mb=stall_min_delta_mb,
                    disk_free_floor_gb=disk_free_floor_gb,
                    status_callback=download_heartbeat,
                )
            except Exception as e:
                download_rc, tail = 127, f"{type(e).__name__}: {e}"
                print(f"[procure] download spawn failed: {tail}", file=sys.stderr)
        duration_s = max(time.monotonic() - t0, 0.0)
        ended_at = _now()
        after_gb = _path_size_gb(dirn)
        cache_after = _cache_sizes_gb()
        delta_gb = max(after_gb - before_gb, 0.0)
        observed_mb_s = (delta_gb * 1000.0 / duration_s) if duration_s > 0 else None
        verify_record = None
        final_rc = download_rc
        if verify and download_rc == 0:
            verify_cmd = [
                "hf", "cache", "verify", spec.hf_id,
                "--cache-dir", HF_HUB_CACHE_DIR,
                "--local-dir", dirn,
                "--fail-on-missing-files",
            ]
            vt0 = time.monotonic()
            print(f"[procure] verify attempt {attempt}/{attempts}: {' '.join(verify_cmd)}", file=sys.stderr)
            checkpoint("verifying", phase="verify", attempt=attempt, cmd=verify_cmd,
                       disk_free_gb=round(_disk_free_gb(dirn), 3))

            def verify_heartbeat(status, sample, **extra):
                heartbeat_status = "verifying" if status == "running" else status
                checkpoint(
                    heartbeat_status,
                    phase="verify",
                    attempt=attempt,
                    progress=sample,
                    disk_free_gb=(sample or {}).get("disk_free_gb", round(_disk_free_gb(dirn), 3)),
                    **extra,
                )

            verify_progress = None
            verify_samples = []
            verify_disk_at_start = _disk_free_gb(dirn)
            if disk_free_floor_gb > 0 and verify_disk_at_start < disk_free_floor_gb:
                verify_reason = (
                    f"disk free {verify_disk_at_start:.1f} GB below safety floor "
                    f"{disk_free_floor_gb:.1f} GB before verification"
                )
                verify_rc, verify_tail = 75, verify_reason
                verify_sample = _progress_sample(dirn, vt0)
                verify_samples = [verify_sample]
                verify_progress = _progress_summary(
                    verify_samples,
                    stall_timeout_s=0,
                    terminated_for_disk=True,
                    disk_reason=verify_reason,
                    disk_free_floor_gb=disk_free_floor_gb,
                )
                checkpoint("blocked_disk", phase="verify", attempt=attempt, reason=verify_reason,
                           progress=verify_sample, disk_free_gb=round(verify_disk_at_start, 3))
            else:
                try:
                    verify_rc, verify_tail, verify_progress, verify_samples = _run_tee_monitored(
                        verify_cmd,
                        env,
                        dirn,
                        progress_interval_s=progress_interval_s,
                        stall_timeout_s=0,
                        stall_min_delta_mb=stall_min_delta_mb,
                        disk_free_floor_gb=disk_free_floor_gb,
                        status_callback=verify_heartbeat,
                    )
                except Exception as e:
                    verify_rc, verify_tail = 127, f"{type(e).__name__}: {e}"
                    print(f"[procure] verify spawn failed: {verify_tail}", file=sys.stderr)
            verify_record = {
                "cmd": verify_cmd,
                "returncode": verify_rc,
                "duration_s": round(max(time.monotonic() - vt0, 0.0), 3),
                "output_tail": verify_tail[-4000:],
                "progress": verify_progress,
                "progress_samples": verify_samples,
            }
            final_rc = verify_rc

        termination_summaries = [progress_summary]
        if verify_record:
            termination_summaries.append(verify_record.get("progress"))
        termination_summaries = [p for p in termination_summaries if isinstance(p, dict)]
        terminated_for_disk = any(p.get("terminated_for_disk") for p in termination_summaries)
        terminated_for_signal = any(p.get("terminated_for_signal") for p in termination_summaries)
        terminated_for_stall = any(p.get("terminated_for_stall") for p in termination_summaries)
        retry_reason = None
        if final_rc != 0:
            retry_reason = f"returncode {final_rc}"
            for summary in termination_summaries:
                retry_reason = (
                    summary.get("signal_reason")
                    or summary.get("disk_reason")
                    or summary.get("stall_reason")
                    or retry_reason
                )
        elif (not already_populated and min_observed_mbs > 0
              and observed_mb_s is not None and observed_mb_s < min_observed_mbs):
            retry_reason = f"observed {observed_mb_s:.1f} MB/s below floor {min_observed_mbs:.1f} MB/s"
        will_retry = bool(
            retry_reason and attempt < attempts
            and not terminated_for_disk
            and not terminated_for_signal
        )
        diagnostics = None
        if retry_reason and diagnose_on_fail:
            diagnostics = _download_diagnostics(
                retry_reason,
                workers,
                observed_mb_s=observed_mb_s,
                min_observed_mbs=min_observed_mbs,
                progress_summary=progress_summary,
                probe_network=network_diagnose and not terminated_for_disk and not terminated_for_signal,
            )

        row = {
            "schema": "hawking.frontier_download.v1",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": round(duration_s, 3),
            "returncode": final_rc,
            "hf_download_returncode": download_rc,
            "attempt": attempt,
            "attempt_count": attempts,
            "retry_reason": retry_reason,
            "will_retry": will_retry,
            "label": spec.label,
            "hf_id": spec.hf_id,
            "local_dir": dirn,
            "source_kind": spec.source_kind,
            "manifest_gb": spec.download_gb,
            "local_dir_gb_before": round(before_gb, 3),
            "local_dir_gb_after": round(after_gb, 3),
            "delta_local_dir_gb": round(delta_gb, 3),
            "observed_mb_s_from_delta": round(observed_mb_s, 2) if observed_mb_s is not None else None,
            "min_observed_mbs": min_observed_mbs,
            "already_populated_at_start": bool(already_populated),
            "cmd": cmd,
            "max_workers": workers,
            "fast_env": FAST_ENV,
            "monitor": {
                "progress_interval_s": float(progress_interval_s or 0.0),
                "stall_timeout_s": float(stall_timeout_s or 0.0),
                "stall_min_delta_mb": float(stall_min_delta_mb or 0.0),
                "disk_free_floor_gb": disk_free_floor_gb,
            },
            "progress": progress_summary,
            "progress_samples": progress_samples,
            "cache": {
                "hf_home": HF_HOME_DIR,
                "hf_hub_cache": HF_HUB_CACHE_DIR,
                "hf_xet_cache": HF_XET_CACHE_DIR,
                "before_gb": cache_before,
                "after_gb": cache_after,
                "delta_hf_home_gb": round(cache_after["hf_home_gb"] - cache_before["hf_home_gb"], 3),
                "delta_hf_hub_cache_gb": round(cache_after["hf_hub_cache_gb"] - cache_before["hf_hub_cache_gb"], 3),
                "delta_hf_xet_cache_gb": round(cache_after["hf_xet_cache_gb"] - cache_before["hf_xet_cache_gb"], 3),
            },
            "verify": verify_record,
            "download_state_path": state_path,
            "verified_marker_path": verified_path,
            "diagnostics": diagnostics,
            "git_commit": _git_commit(),
            "output_tail": tail[-4000:],
        }
        if spec.download_gb and observed_mb_s and observed_mb_s > 0:
            remaining_gb = max(spec.download_gb - before_gb, 0.0)
            row["eta_hours_at_observed_rate"] = round((remaining_gb * 1000.0) / observed_mb_s / 3600.0, 3)
        _record_download(row, path=download_log)

        verification_satisfied = not verify or bool(verify_record and verify_record.get("returncode") == 0)
        completion_ok = bool(
            download_rc == 0
            and final_rc == 0
            and verification_satisfied
            and not terminated_for_disk
            and not terminated_for_signal
            and not terminated_for_stall
        )
        if completion_ok and not will_retry:
            marker = {
                "schema": DOWNLOAD_VERIFIED_SCHEMA,
                "status": "verified",
                "verified_complete": True,
                "label": spec.label,
                "hf_id": spec.hf_id,
                "local_dir": dirn,
                "source_kind": spec.source_kind,
                "manifest_gb": spec.download_gb,
                "completed_at": ended_at,
                "attempt": attempt,
                "attempt_count": attempts,
                "hf_download_returncode": download_rc,
                "verification": {
                    "requested": bool(verify),
                    "returncode": verify_record.get("returncode") if verify_record else None,
                },
                "local_dir_gb": round(after_gb, 3),
                "disk_free_gb": round(_disk_free_gb(dirn), 3),
                "download_state_path": state_path,
                "download_telemetry_path": download_log,
                "git_commit": _git_commit(),
            }
            _atomic_write_json(verified_path, marker)
            checkpoint(
                "verified",
                phase="complete",
                attempt=attempt,
                ended_at=ended_at,
                returncode=0,
                hf_download_returncode=download_rc,
                verify_returncode=verify_record.get("returncode") if verify_record else None,
                verified_complete=True,
                verified_marker=verified_path,
                disk_free_gb=marker["disk_free_gb"],
            )
            print(f"[procure] verified download checkpoint -> {verified_path}", file=sys.stderr)

        if completion_ok and not retry_reason:
            break
        if will_retry:
            checkpoint("retry_pending", phase="download", attempt=attempt, reason=retry_reason,
                       returncode=final_rc, next_attempt=attempt + 1,
                       disk_free_gb=round(_disk_free_gb(dirn), 3))
            print(f"[procure] retrying after {retry_reason}; next attempt uses fewer workers", file=sys.stderr)
            continue

        if not completion_ok:
            if terminated_for_signal:
                final_status = "interrupted"
            elif terminated_for_disk:
                final_status = "blocked_disk"
            elif terminated_for_stall:
                final_status = "stalled"
            else:
                final_status = "failed"
            checkpoint(
                final_status,
                phase="complete",
                attempt=attempt,
                ended_at=ended_at,
                returncode=final_rc,
                reason=retry_reason,
                verified_complete=False,
                disk_free_gb=round(_disk_free_gb(dirn), 3),
            )
        break
    return final_rc


def all_frontier(link_mb_s, efficiency):
    tot = total_download_gb()
    artifacts = total_artifact_gb()
    effective = link_mb_s * efficiency
    print(f"[procure] frontier manifest ({len(FRONTIER_MODELS)} parents, "
          f"~{tot/1000:.1f} TB download + ~{artifacts:.0f} GB .tq outputs):", file=sys.stderr)
    for spec in sorted(FRONTIER_MODELS, key=lambda m: m.download_gb):
        _state_path, marker_path = _checkpoint_paths(spec.label)
        populated = os.path.isdir(spec.local_dir) and os.listdir(spec.local_dir)
        verified = populated and _verified_marker_valid(
            _read_json(marker_path), label=spec.label, hf_id=spec.hf_id,
            local_dir=spec.local_dir, require_verify=True
        )
        staged = "VERIFIED" if verified else (
            "PARTIAL — resume+verify" if populated else
            f"~{fmt_hours(eta_hours_for(spec.download_gb, link_mb_s, efficiency))} "
            f"@ {effective:.0f} MB/s effective"
        )
        fit = "resident" if spec.fits_resident(DEFAULT_HARDWARE) else "paged"
        print(f"  {spec.label:16s} {spec.download_gb:>5.0f} GB  {spec.hf_id:42s} -> "
              f"{spec.local_dir:26s} [{staged}; .tq~{spec.artifact_gb():.0f}GB {fit}; "
              f"{spec.source_kind}]", file=sys.stderr)
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    total_with_outputs = tot + artifacts
    disk_ok = total_with_outputs <= DEFAULT_HARDWARE.ssd_gb
    free_ok = total_with_outputs <= free_gb
    print(f"# serial download wall-clock ~{fmt_hours(eta_hours_for(tot, link_mb_s, efficiency))} "
          f"at {link_mb_s:.0f} MB/s physical x {efficiency:.2f} = {effective:.0f} MB/s effective.",
          file=sys.stderr)
    print(f"# disk model: downloads + .tq outputs ~= {total_with_outputs/1000:.1f} TB; "
          f"{'fits' if disk_ok else 'DOES NOT FIT'} on {DEFAULT_HARDWARE.ssd_tb:.0f} TB, "
          f"{'fits current free space' if free_ok else f'needs more than current free space ({free_gb:.0f} GB free)'}",
          file=sys.stderr)
    print(f"# full-fit view keeps every source parent. For the real long run, also check "
          f"`procure.py --cycle-frontier` to model source-release after bake.",
          file=sys.stderr)
    print(f"# download smallest-first / biggest-last so the ladder + condense start immediately on the small ones.",
          file=sys.stderr)
    print(f"# to actually run one: procure.py <label>   (fastest path auto-applied)", file=sys.stderr)


def cycle_frontier(link_mb_s, efficiency,
                   scratch_gb=DEFAULT_HARDWARE.scratch_reserve_gb, keep_outputs=True):
    """Print the disk lifecycle the Studio actually wants: download one source, bake .tq, release source.

    This is deliberately a PLAN, not a deleter. It lets the operator compare full-fit storage
    pressure against the long-running experiment shape where source checkpoints are transient and
    only receipts + .tq outputs accumulate.
    """
    ordered = sorted(FRONTIER_MODELS, key=lambda m: (m.download_gb, m.artifact_gb()))
    tot = total_download_gb()
    artifacts = total_artifact_gb()
    effective = link_mb_s * efficiency
    kept_outputs = 0.0
    peak_gb = 0.0
    print(f"[procure] cycle-frontier plan ({len(ordered)} parents): keep "
          f"{'.tq outputs' if keep_outputs else 'no outputs'}; release source after each bake; "
          f"scratch reserve {scratch_gb:.0f} GB; cache reserve {CACHE_RESERVE_GB:.0f} GB",
          file=sys.stderr)
    print(f"# order: smallest source first for fast early receipts, largest source last for lower peak risk",
          file=sys.stderr)
    for i, spec in enumerate(ordered, 1):
        source = spec.download_gb
        output = spec.artifact_gb()
        before = kept_outputs if keep_outputs else 0.0
        step_peak = before + source + output + scratch_gb + CACHE_RESERVE_GB
        peak_gb = max(peak_gb, step_peak)
        eta = fmt_hours(eta_hours_for(source, link_mb_s, efficiency))
        print(f"  {i:02d}. {spec.label:16s} source {source:>6.0f} GB -> .tq {output:>6.0f} GB  "
              f"step-peak~{step_peak/1000:>4.1f} TB  ETA~{eta}  "
              f"{spec.source_kind}", file=sys.stderr)
        if keep_outputs:
            kept_outputs += output
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    total_with_outputs = tot + artifacts
    print(f"# download-only serial wall-clock ~{fmt_hours(eta_hours_for(tot, link_mb_s, efficiency))} "
          f"at {link_mb_s:.0f} MB/s physical x {efficiency:.2f} = {effective:.0f} MB/s effective.",
          file=sys.stderr)
    print(f"# full-fit storage view: ~{tot/1000:.1f} TB sources + ~{artifacts:.0f} GB .tq "
          f"= ~{total_with_outputs/1000:.1f} TB.", file=sys.stderr)
    print(f"# cycle storage view: peak live disk ~{peak_gb/1000:.1f} TB "
          f"({'.tq outputs retained' if keep_outputs else 'outputs dropped after measurement'}). "
          f"Current free space here: {free_gb:.0f} GB.", file=sys.stderr)
    usable_now = max(0.0, free_gb - DEFAULT_HARDWARE.disk_reserve_gb)
    waves = storage_wave_plan(storage_budget_gb=usable_now, link_mb_s=link_mb_s,
                              efficiency=efficiency, scratch_gb=scratch_gb,
                              cache_reserve_gb=CACHE_RESERVE_GB, keep_outputs=keep_outputs)
    print(f"# storage-wave view against current free space minus "
          f"{DEFAULT_HARDWARE.disk_reserve_gb:.0f} GB: {waves['wave_count']} wave(s), "
          f"peak ~{waves['planned_peak_gb']/1000:.1f} TB, checkpoint target "
          f"<= {waves['max_wave_hours']:.1f} h where possible. "
          f"Use `frontier_ops.py storage-plan` for the exact waves.",
          file=sys.stderr)
    print(f"# operator contract: after each model's bake+receipt+verify, archive receipts and delete/reclaim "
          f"that model's source directory (for example `scratch/glm-5.2`) before starting the next source. "
          f"Do not delete source before receipts pass.",
          file=sys.stderr)


def stream_plan(label):
    spec = _resolve(label)
    print(f"# STREAM-CONDENSE plan for {spec.label} ({spec.download_gb:.0f} GB {spec.source_kind} -> "
          f"a low-bpw .tq): download+bake+delete per", file=sys.stderr)
    print(f"# shard so peak disk ~= one shard + the .tq, and bake compute hides under download I/O wait.", file=sys.stderr)
    print(f"#   for each shard S in {spec.hf_id}:", file=sys.stderr)
    print(f"#     hf download {spec.hf_id} <S> --local-dir {spec.local_dir}   # HF_HUB_ENABLE_HF_TRANSFER=1", file=sys.stderr)
    print(f"#     quantize-model --in {spec.local_dir}/<S> --append-tq {spec.local_dir}.tq   # shard-incremental bake", file=sys.stderr)
    print(f"#     rm {spec.local_dir}/<S>                                # reclaim the source shard", file=sys.stderr)
    print(f"# GATE: needs the baker's shard-incremental --append-tq mode (serve-build item). Until then,", file=sys.stderr)
    print("# whole-parent staging is allowed only when the live disk gate passes; otherwise use external "
          "storage or wait for the real shard-incremental path.", file=sys.stderr)


def cache_status():
    _ensure_cache_dirs()
    print("# Hawking project-local Hugging Face cache", file=sys.stderr)
    print(f"HF_HOME={HF_HOME_DIR}", file=sys.stderr)
    print(f"HF_HUB_CACHE={HF_HUB_CACHE_DIR}", file=sys.stderr)
    print(f"HF_XET_CACHE={HF_XET_CACHE_DIR}", file=sys.stderr)
    sizes = _cache_sizes_gb()
    print(f"# sizes: HF_HOME={sizes['hf_home_gb']:.3f} GB  "
          f"HF_HUB_CACHE={sizes['hf_hub_cache_gb']:.3f} GB  "
          f"HF_XET_CACHE={sizes['hf_xet_cache_gb']:.3f} GB", file=sys.stderr)
    hf = shutil.which("hf")
    if not hf:
        print("# hf CLI missing; cannot list cache entries", file=sys.stderr)
        return 1
    return subprocess.call(["hf", "cache", "list", "--cache-dir", HF_HUB_CACHE_DIR, "--limit", "20"])


def cache_prune(dry=True):
    _ensure_cache_dirs()
    hf = shutil.which("hf")
    if not hf:
        print("# hf CLI missing; cannot prune cache", file=sys.stderr)
        return 1
    cmd = ["hf", "cache", "prune", "--cache-dir", HF_HUB_CACHE_DIR]
    if dry:
        cmd.append("--dry-run")
    else:
        cmd.append("--yes")
    print("[procure] " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


def detach_download():
    """Launch one resumable download independently of the terminal/Codex connection."""
    if len(sys.argv) < 3:
        print("usage: procure.py --detach <label|hf_id> [download options]", file=sys.stderr)
        return 2
    label = sys.argv[2]
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    log_path = os.path.join(ROOT, "reports", "condense", f"download_{safe_label}.log")
    pid_path = os.path.join(DOWNLOAD_STATE_DIR, f"{safe_label}.pid.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), label, *sys.argv[3:]]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-dimsu", *cmd]
    log = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    _atomic_write_json(pid_path, {
        "schema": "hawking.frontier_download_pid.v1",
        "label": label,
        "pid": proc.pid,
        "started_at": _now(),
        "log_path": log_path,
        "cmd": cmd,
    })
    print(f"[procure] detached {label} pid={proc.pid}; log={log_path}", file=sys.stderr)
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory(prefix="procure_telemetry_") as td:
        cmd = [sys.executable, "-c",
               "import sys; sys.stdout.write('stdout-ok\\n'); sys.stderr.write('stderr-ok\\n')"]
        rc, tail = _run_tee(cmd, os.environ.copy())
        check("tee runner return code", rc == 0)
        check("tee runner captures output tail", "stdout-ok" in tail and "stderr-ok" in tail)

        sample_dir = os.path.join(td, "source")
        os.makedirs(sample_dir)
        with open(os.path.join(sample_dir, "part.bin"), "wb") as f:
            f.write(b"x" * 4096)
        check("path size sees synthetic file", _path_size_gb(sample_dir) > 0)
        check("project-local cache default configured", "scratch" in HF_HUB_CACHE_DIR)
        check("worker backoff halves safely", _workers_for_attempt("32", 1) == "32"
              and _workers_for_attempt("32", 2) == "16"
              and _workers_for_attempt("1", 3) == "1")
        moving = [
            {"elapsed_s": 0.0, "tracked_total_gb": 1.0},
            {"elapsed_s": 10.0, "tracked_total_gb": 1.2, "window_s": 10.0, "delta_tracked_gb": 0.2},
        ]
        stalled = [
            {"elapsed_s": 0.0, "tracked_total_gb": 1.0},
            {"elapsed_s": 10.0, "tracked_total_gb": 1.0, "window_s": 10.0, "delta_tracked_gb": 0.0},
            {"elapsed_s": 20.0, "tracked_total_gb": 1.0, "window_s": 10.0, "delta_tracked_gb": 0.0},
        ]
        check("progress summary accepts moving download",
              not _progress_summary(moving, stall_min_delta_mb=64, stall_timeout_s=15)["stalled"])
        check("progress summary flags stalled download",
              _progress_summary(stalled, stall_min_delta_mb=64, stall_timeout_s=15)["stalled"])
        disk_stopped = _progress_summary(
            [{"elapsed_s": 0.0, "tracked_total_gb": 1.0, "disk_free_gb": 149.0}],
            terminated_for_disk=True,
            disk_reason="synthetic disk floor",
            disk_free_floor_gb=150.0,
        )
        check("progress summary records 150GB disk-floor stop",
              disk_stopped["terminated_for_disk"]
              and disk_stopped["disk_free_floor_gb"] == 150.0
              and disk_stopped["disk_free_gb_min"] == 149.0)
        diag = _download_diagnostics(
            "no tracked local/cache growth >= 64.0 MB for 900s",
            "16",
            observed_mb_s=0.0,
            min_observed_mbs=80.0,
            progress_summary={"terminated_for_stall": True},
            probe_network=False,
        )
        check("diagnostics include actionable recommendations",
              bool(diag["recommendations"]) and diag["network_probe"].get("skipped"))

        log_path = os.path.join(td, "downloads.jsonl")
        row = {"schema": "hawking.frontier_download.v1", "label": "synthetic", "returncode": 0}
        _record_download(row, path=log_path)
        rows = [json.loads(line) for line in open(log_path)]
        check("download telemetry JSONL written", rows == [row])

        state_dir = os.path.join(td, "download-state")
        state_path, verified_path = _checkpoint_paths("Synthetic/Model", state_dir)
        heartbeat = {
            "schema": DOWNLOAD_STATE_SCHEMA,
            "label": "Synthetic/Model",
            "hf_id": "example/model",
            "local_dir": sample_dir,
            "status": "downloading",
            "progress": {"local_dir_gb": 0.001},
        }
        _atomic_write_json(state_path, heartbeat)
        check("download heartbeat is atomically readable",
              _read_json(state_path).get("status") == "downloading")
        heartbeat["status"] = "interrupted"
        _atomic_write_json(state_path, heartbeat)
        check("download heartbeat replacement preserves resumable state",
              _read_json(state_path).get("status") == "interrupted")

        incomplete_marker = {
            "schema": DOWNLOAD_VERIFIED_SCHEMA,
            "status": "verified",
            "verified_complete": True,
            "label": "Synthetic/Model",
            "hf_id": "example/model",
            "local_dir": sample_dir,
            "hf_download_returncode": 0,
            "verification": {"requested": True, "returncode": 1},
        }
        _atomic_write_json(verified_path, incomplete_marker)
        check("failed requested verification never validates complete marker",
              not _verified_marker_valid(
                  _read_json(verified_path), label="Synthetic/Model",
                  hf_id="example/model", local_dir=sample_dir))
        complete_marker = dict(incomplete_marker)
        complete_marker["verification"] = {"requested": True, "returncode": 0}
        _atomic_write_json(verified_path, complete_marker)
        check("successful download plus requested verify validates marker",
              _verified_marker_valid(
                  _read_json(verified_path), label="Synthetic/Model",
                  hf_id="example/model", local_dir=sample_dir))
        _durable_unlink(verified_path)
        check("new/resumed attempt can durably invalidate old marker",
              not os.path.exists(verified_path))

        blocked_state_dir = os.path.join(td, "blocked-state")
        blocked_log = os.path.join(td, "blocked-downloads.jsonl")
        blocked_source = os.path.join(td, "blocked-source")
        blocked_rc = download(
            "selftest/blocked",
            dir_override=blocked_source,
            retries=2,
            verify=True,
            diagnose_on_fail=False,
            disk_free_floor_gb=1e12,
            state_dir=blocked_state_dir,
            download_log=blocked_log,
        )
        blocked_state_path, blocked_marker_path = _checkpoint_paths(
            "selftest/blocked", blocked_state_dir
        )
        blocked_rows = [json.loads(line) for line in open(blocked_log)]
        check("disk floor checkpoints before spawning network download",
              blocked_rc == 75
              and _read_json(blocked_state_path).get("status") == "blocked_disk"
              and len(blocked_rows) == 1
              and blocked_rows[0].get("will_retry") is False)
        check("disk-blocked download cannot publish verified marker",
              not os.path.exists(blocked_marker_path))

    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def _arg_float(names, default):
    for name in names:
        if name in sys.argv:
            return float(sys.argv[sys.argv.index(name) + 1])
    return default


def _link_mb_s_from_args():
    if "--link-mbs" in sys.argv or "--link-MBps" in sys.argv:
        return _arg_float(("--link-mbs", "--link-MBps"), 125.0)
    link_mbps = _arg_float(("--link-mbps",), 1000.0)
    return link_mbps / 8.0


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if a == "--check":
        sys.exit(0 if check() else 1)
    elif a == "--selftest":
        sys.exit(0 if selftest() else 1)
    elif a == "--all-frontier":
        all_frontier(_link_mb_s_from_args(), _arg_float(("--efficiency",), 1.0))
    elif a == "--cycle-frontier":
        cycle_frontier(_link_mb_s_from_args(), _arg_float(("--efficiency",), 1.0),
                       scratch_gb=_arg_float(("--scratch-gb",), DEFAULT_HARDWARE.scratch_reserve_gb),
                       keep_outputs="--drop-outputs" not in sys.argv)
    elif a == "--stream":
        stream_plan(sys.argv[2])
    elif a == "--cache-status":
        sys.exit(cache_status())
    elif a == "--cache-prune":
        sys.exit(cache_prune(dry="--yes" not in sys.argv))
    elif a == "--detach":
        sys.exit(detach_download())
    elif a == "--help":
        print(__doc__)
    else:
        d = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else None
        sys.exit(download(
            a,
            d,
            dry="--dry" in sys.argv,
            retries=int(_arg_float(("--retries",), 0)),
            min_observed_mbs=_arg_float(("--min-observed-mbs",), 0.0),
            verify="--verify" in sys.argv,
            progress_interval_s=_arg_float(("--progress-interval-s",), PROGRESS_INTERVAL_S),
            stall_timeout_s=_arg_float(("--stall-timeout-s",), STALL_TIMEOUT_S),
            stall_min_delta_mb=_arg_float(("--stall-min-delta-mb",), STALL_MIN_DELTA_MB),
            diagnose_on_fail="--no-diagnose-on-fail" not in sys.argv,
            network_diagnose="--no-network-diagnose" not in sys.argv,
            disk_free_floor_gb=_arg_float(("--disk-free-floor-gb",), DISK_FREE_FLOOR_GB),
        ))
