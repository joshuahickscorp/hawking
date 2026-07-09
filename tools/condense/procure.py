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
  procure.py <label> --no-diagnose-on-fail # skip automatic route/HF/network hints on failed attempts

Each real download appends an observed-throughput receipt with live progress samples, stall evidence,
route/HF/network diagnostics for bad attempts, and cache deltas to
reports/condense/frontier_downloads.jsonl. After a bake, use frontier_ops.py
status/ledger/release-source for the guarded lifecycle. This file downloads;
frontier_ops.py summarizes evidence and refuses unsafe source deletion.
By default HF_HOME / HF_HUB_CACHE / HF_XET_CACHE are pinned under scratch/, so the Studio run does
not silently fill a global ~/.cache/huggingface directory. User-provided HF_* env vars still win.
"""
import sys, os, subprocess, shutil, importlib, json, time, datetime, selectors, tempfile

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
CACHE_RESERVE_GB = 128.0
PROGRESS_INTERVAL_S = float(os.environ.get("HAWKING_PROCURE_PROGRESS_INTERVAL_S", "60"))
STALL_TIMEOUT_S = float(os.environ.get("HAWKING_PROCURE_STALL_TIMEOUT_S", "900"))
STALL_MIN_DELTA_MB = float(os.environ.get("HAWKING_PROCURE_STALL_MIN_DELTA_MB", "64"))


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
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


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
                      terminated_for_stall=False, stall_reason=None):
    if not samples:
        return {
            "sample_count": 0,
            "stalled": False,
            "terminated_for_stall": bool(terminated_for_stall),
            "stall_reason": stall_reason,
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
    }


def _run_tee_monitored(cmd, env, local_dir, progress_interval_s=PROGRESS_INTERVAL_S,
                       stall_timeout_s=STALL_TIMEOUT_S, stall_min_delta_mb=STALL_MIN_DELTA_MB):
    """Run a long command with live output plus periodic disk/cache progress samples."""
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
        if (proc.poll() is None and progress_interval_s and progress_interval_s > 0
                and now - last_sample_at >= progress_interval_s):
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
                f"window={rate_txt} no-progress={no_progress_s:.0f}s",
                file=sys.stderr,
            )
            if stall_timeout_s and stall_timeout_s > 0 and no_progress_s >= stall_timeout_s:
                terminated_for_stall = True
                stall_reason = (
                    f"no tracked local/cache growth >= {stall_min_delta_mb:.1f} MB "
                    f"for {no_progress_s:.0f}s"
                )
                print(f"[procure] STALL: {stall_reason}; terminating command", file=sys.stderr)
                proc.terminate()

        if terminated_for_stall and proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("[procure] STALL: terminate timed out; killing command", file=sys.stderr)
                proc.kill()
                proc.wait()

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
    )
    if terminated_for_stall:
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
             stall_min_delta_mb=STALL_MIN_DELTA_MB, diagnose_on_fail=True, network_diagnose=True):
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
        try:
            download_rc, tail, progress_summary, progress_samples = _run_tee_monitored(
                cmd,
                env,
                dirn,
                progress_interval_s=progress_interval_s,
                stall_timeout_s=stall_timeout_s,
                stall_min_delta_mb=stall_min_delta_mb,
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
            try:
                verify_rc, verify_tail = _run_tee(verify_cmd, env)
            except Exception as e:
                verify_rc, verify_tail = 127, f"{type(e).__name__}: {e}"
                print(f"[procure] verify spawn failed: {verify_tail}", file=sys.stderr)
            verify_record = {
                "cmd": verify_cmd,
                "returncode": verify_rc,
                "duration_s": round(max(time.monotonic() - vt0, 0.0), 3),
                "output_tail": verify_tail[-4000:],
            }
            final_rc = verify_rc

        retry_reason = None
        if final_rc != 0:
            retry_reason = f"returncode {final_rc}"
            if isinstance(progress_summary, dict) and progress_summary.get("terminated_for_stall"):
                retry_reason = progress_summary.get("stall_reason") or retry_reason
        elif (not already_populated and min_observed_mbs > 0
              and observed_mb_s is not None and observed_mb_s < min_observed_mbs):
            retry_reason = f"observed {observed_mb_s:.1f} MB/s below floor {min_observed_mbs:.1f} MB/s"
        diagnostics = None
        if retry_reason and diagnose_on_fail:
            diagnostics = _download_diagnostics(
                retry_reason,
                workers,
                observed_mb_s=observed_mb_s,
                min_observed_mbs=min_observed_mbs,
                progress_summary=progress_summary,
                probe_network=network_diagnose,
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
            "will_retry": bool(retry_reason and attempt < attempts),
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
            "diagnostics": diagnostics,
            "git_commit": _git_commit(),
            "output_tail": tail[-4000:],
        }
        if spec.download_gb and observed_mb_s and observed_mb_s > 0:
            remaining_gb = max(spec.download_gb - before_gb, 0.0)
            row["eta_hours_at_observed_rate"] = round((remaining_gb * 1000.0) / observed_mb_s / 3600.0, 3)
        _record_download(row)
        if not retry_reason:
            break
        if attempt < attempts:
            print(f"[procure] retrying after {retry_reason}; next attempt uses fewer workers", file=sys.stderr)
    return final_rc


def all_frontier(link_mb_s, efficiency):
    tot = total_download_gb()
    artifacts = total_artifact_gb()
    effective = link_mb_s * efficiency
    print(f"[procure] frontier manifest ({len(FRONTIER_MODELS)} parents, "
          f"~{tot/1000:.1f} TB download + ~{artifacts:.0f} GB .tq outputs):", file=sys.stderr)
    for spec in sorted(FRONTIER_MODELS, key=lambda m: m.download_gb):
        staged = "STAGED" if os.path.isdir(spec.local_dir) and os.listdir(spec.local_dir) else (
            f"~{fmt_hours(eta_hours_for(spec.download_gb, link_mb_s, efficiency))} "
            f"@ {effective:.0f} MB/s effective")
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


def cycle_frontier(link_mb_s, efficiency, scratch_gb=200.0, keep_outputs=True):
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
    waves = storage_wave_plan(storage_budget_gb=DEFAULT_HARDWARE.ssd_gb, link_mb_s=link_mb_s,
                              efficiency=efficiency, scratch_gb=scratch_gb,
                              cache_reserve_gb=CACHE_RESERVE_GB, keep_outputs=keep_outputs)
    print(f"# storage-wave view on an 8 TB target: {waves['wave_count']} wave(s), "
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
    print(f"# download the full parent (8 TB fits) and bake whole. The fused path is the disk+time optimizer.", file=sys.stderr)


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
                       scratch_gb=_arg_float(("--scratch-gb",), 200.0),
                       keep_outputs="--drop-outputs" not in sys.argv)
    elif a == "--stream":
        stream_plan(sys.argv[2])
    elif a == "--cache-status":
        sys.exit(cache_status())
    elif a == "--cache-prune":
        sys.exit(cache_prune(dry="--yes" not in sys.argv))
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
        ))
