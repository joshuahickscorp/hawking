#!/usr/bin/env python3.12
"""preflight.py — run this FIRST on the Mac Studio, before `studio_run.py go`.

Confirms the whole environment is actually ready: Python deps, Rust toolchain, disk space, staged
model parents, launch-time HF refresh/ledger artifacts, and the receipt harness — so `go` never dies
10 minutes in on a missing dependency or stale frontier manifest. Exits 0 (green, safe to `go`) or 1
(red, prints exactly what to fix). Pure stdlib + best-effort imports; never crashes on a missing
optional dep, it just flags it.

Usage:
  python3.12 tools/condense/preflight.py            # full check + signed JSON summary
  python3.12 tools/condense/preflight.py --quiet    # exit code only, minimal output
  python3.12 tools/condense/preflight.py --verify-summary [PATH]
"""
import sys, os, subprocess, shutil, importlib, importlib.metadata, importlib.util, json, datetime, hashlib, socket, time, urllib.request, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)
from studio_manifest import DEFAULT_HARDWARE

QUIET = "--quiet" in sys.argv
REQUIRE_FRONTIER = "--require-frontier" in sys.argv
REQUIRED_PY = ["torch", "transformers", "safetensors", "numpy", "jsonschema"]
MIN_DISK_GB = DEFAULT_HARDWARE.disk_reserve_gb
MIN_RAM_GIB = 90.0      # accept the delivered 96 GiB Studio while rejecting smaller hosts
REFRESH_OUT = "reports/condense/frontier_refresh.preflight.json"
LEDGER_OUT = "reports/condense/frontier_ledger.preflight.json"
LAUNCH_GATE_OUT = "reports/condense/frontier_launch_gate.preflight.json"
SUMMARY_OUT = "reports/condense/studio_preflight_summary.json"
BAKER = "vendor/strand-quant/target/release/quantize-model"


def _say(ok, msg):
    if not QUIET:
        print(f"[{'OK ' if ok else 'FAIL'}] {msg}", file=sys.stderr)
    return ok


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"


def _git_commit():
    rc, out, _ = _run(["git", "rev-parse", "--short", "HEAD"], timeout=10)
    return out.strip() if rc == 0 and out.strip() else "unknown"


def _memory_state():
    """Best-effort live pressure state; values are evidence, not capacity guesses."""
    pressure = None
    rc, out, _ = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], timeout=10)
    if rc == 0:
        try:
            pressure = int(out.strip())
        except ValueError:
            pass
    swap_mb = None
    rc, out, _ = _run(["sysctl", "-n", "vm.swapusage"], timeout=10)
    if rc == 0:
        match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", out)
        if match:
            value, unit = float(match.group(1)), match.group(2)
            swap_mb = value * {"M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}[unit]
    return {
        "pressure_level": pressure,
        "pressure_state": {1: "normal", 2: "warning", 4: "critical"}.get(pressure, "unknown"),
        "swap_used_mb": round(swap_mb, 3) if swap_mb is not None else None,
    }


def _hardware_summary():
    ram_bytes = None
    ram_gib = None
    rc, out, _ = _run(["sysctl", "-n", "hw.memsize"], timeout=10)
    if rc == 0:
        try:
            ram_bytes = int(out.strip())
            ram_gib = ram_bytes / (1024 ** 3)
        except ValueError:
            ram_gib = None
    rc_cpu, cpu_brand, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=10)
    rc_ncpu, ncpu, _ = _run(["sysctl", "-n", "hw.ncpu"], timeout=10)
    rc_model, hw_model, _ = _run(["sysctl", "-n", "hw.model"], timeout=10)
    rc_batt, batt, batt_err = _run(["pmset", "-g", "batt"], timeout=10)
    rc_therm, therm, therm_err = _run(["pmset", "-g", "therm"], timeout=10)
    usage = shutil.disk_usage(ROOT)
    return {
        "profile": DEFAULT_HARDWARE.name,
        "target_ram_gb": MIN_RAM_GIB,  # legacy field consumed by the Studio snapshot UI
        "target_ram_gib": MIN_RAM_GIB,
        "resident_weight_budget_gb": DEFAULT_HARDWARE.weight_budget_gb,
        "interactive_process_budget_gb": DEFAULT_HARDWARE.process_budget_gb,
        "target_free_disk_gb": MIN_DISK_GB,
        "scratch_reserve_gb": DEFAULT_HARDWARE.scratch_reserve_gb,
        "hf_cache_reserve_gb": DEFAULT_HARDWARE.cache_reserve_gb,
        "actual_ram_bytes": ram_bytes,
        "actual_ram_gb": round(ram_bytes / 1e9, 3) if ram_bytes else None,
        "actual_ram_gib": round(ram_gib, 3) if ram_gib else None,
        "actual_cpu_brand": cpu_brand.strip() if rc_cpu == 0 and cpu_brand.strip() else None,
        "actual_cpu_count": int(ncpu.strip()) if rc_ncpu == 0 and ncpu.strip().isdigit() else None,
        "actual_hw_model": hw_model.strip() if rc_model == 0 and hw_model.strip() else None,
        "disk_total_gb": round(usage.total / 1e9, 3),
        "disk_free_gb": round(usage.free / 1e9, 3),
        "disk_free_after_reserve_gb": round(max(0.0, usage.free / 1e9 - MIN_DISK_GB), 3),
        "memory": _memory_state(),
        "power_source": (batt or batt_err).splitlines()[0].strip()
        if (batt or batt_err).strip() else None,
        "thermal_status": (therm or therm_err).strip()[:1000]
        if (therm or therm_err).strip() else None,
    }


def _route_summary(host):
    rc, out, err = _run(["route", "-n", "get", host], timeout=10)
    data = {"ok": rc == 0}
    if rc != 0:
        data["error"] = (err or out).strip()[:500]
        return data
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in ("interface", "gateway", "source", "ifscope"):
            data[key] = value
    return data


def _network_summary():
    host = "huggingface.co"
    api_url = "https://huggingface.co/api/models?limit=1"
    out = {
        "schema": "hawking.studio_network_summary.v1",
        "host": host,
        "api_url": api_url,
        "probe_is_download": False,
        "route": _route_summary(host),
    }
    out["route_interface"] = out["route"].get("interface")
    out["route_gateway"] = out["route"].get("gateway")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        addresses = sorted({info[4][0] for info in infos})
        out["dns_ok"] = True
        out["addresses"] = addresses[:8]
        out["address_count"] = len(addresses)
    except Exception as e:
        out["dns_ok"] = False
        out["dns_error"] = f"{type(e).__name__}: {e}"[:500]
    try:
        req = urllib.request.Request(api_url, method="HEAD", headers={"User-Agent": "hawking-preflight/1"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=10) as response:
            out["hf_api_status"] = getattr(response, "status", None)
            out["hf_api_elapsed_ms"] = round((time.monotonic() - start) * 1000, 1)
            out["hf_api_server"] = response.headers.get("server")
            out["hf_api_cache"] = response.headers.get("x-cache")
        out["hf_api_ok"] = 200 <= int(out.get("hf_api_status") or 0) < 500
    except Exception as e:
        out["hf_api_ok"] = False
        out["hf_api_error"] = f"{type(e).__name__}: {e}"[:500]
    return out


def _parse_version(text):
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(text))
    if not m:
        return None
    return tuple(int(part or 0) for part in m.groups())


def _min_version_from_range(spec):
    m = re.search(r">=\s*(\d+(?:\.\d+){0,2})", str(spec or ""))
    return _parse_version(m.group(1)) if m else None


def _version_gte(actual, minimum):
    return actual is not None and minimum is not None and actual >= minimum


def _node_candidates(pnpm_path):
    candidates = []
    explicit = os.environ.get("HAWKING_NODE") or os.environ.get("NODE_BINARY")
    if explicit:
        candidates.append(explicit)
    if pnpm_path:
        pnpm_real = os.path.realpath(pnpm_path)
        pnpm_bin = os.path.dirname(pnpm_real)
        pnpm_root = os.path.dirname(pnpm_bin)
        candidates.extend([
            os.path.join(pnpm_root, "node", "bin", "node"),
            os.path.join(pnpm_bin, "node"),
        ])
    path_node = shutil.which("node")
    if path_node:
        candidates.append(path_node)
    out = []
    seen = set()
    for cand in candidates:
        if not cand:
            continue
        real = os.path.realpath(cand)
        if real in seen or not os.path.exists(real):
            continue
        seen.add(real)
        out.append(cand)
    return out


def _node_version(path):
    rc, out, err = _run([path, "-v"], timeout=10)
    version = (out or err).strip().splitlines()[0] if (out or err).strip() else None
    return rc, version


def _developer_environment_summary():
    app_package = os.path.join(ROOT, "app", "package.json")
    engine_spec = None
    try:
        with open(app_package) as f:
            engine_spec = (json.load(f).get("engines") or {}).get("node")
    except Exception:
        engine_spec = None
    pnpm_path = shutil.which("pnpm")
    pnpm_rc, pnpm_out, pnpm_err = _run(["pnpm", "-v"], timeout=10) if pnpm_path else (127, "", "pnpm missing")
    pnpm_version = (pnpm_out or pnpm_err).strip().splitlines()[0] if (pnpm_out or pnpm_err).strip() else None
    required = _min_version_from_range(engine_spec)
    checked_nodes = []
    node_path = None
    node_version = None
    actual = None
    for cand in _node_candidates(pnpm_path):
        node_rc, version = _node_version(cand)
        parsed = _parse_version(version or "")
        checked_nodes.append({
            "path": cand,
            "version": version,
            "ok": bool(node_rc == 0 and _version_gte(parsed, required)),
        })
        if node_rc == 0 and _version_gte(parsed, required):
            node_path = cand
            node_version = version
            actual = parsed
            break
        if node_path is None and node_rc == 0:
            node_path = cand
            node_version = version
            actual = parsed
    node_engine_ok = bool(node_path and engine_spec and _version_gte(actual, required))
    return {
        "schema": "hawking.studio_developer_environment.v1",
        "app_package": "app/package.json",
        "node_path": node_path,
        "node_version": node_version,
        "node_engine": engine_spec,
        "node_engine_ok": node_engine_ok,
        "node_required_min": ".".join(str(v) for v in required) if required else None,
        "node_candidates": checked_nodes,
        "pnpm_path": pnpm_path,
        "pnpm_version": pnpm_version,
        "pnpm_ok": bool(pnpm_path and pnpm_rc == 0),
    }


def _artifact_status(path):
    exists = os.path.exists(path)
    out = {"path": path, "exists": exists}
    if exists:
        out["bytes"] = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out


def write_summary(results, preflight_ok):
    try:
        summary = {
            "schema": "hawking.studio_preflight_summary.v1",
            "generated_at": _now(),
            "root": ROOT,
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "preflight_ok_before_summary": bool(preflight_ok),
            "checks": results,
            "hardware": _hardware_summary(),
            "network": _network_summary(),
            "developer_environment": _developer_environment_summary(),
            "artifacts": {
                "frontier_refresh": _artifact_status(REFRESH_OUT),
                "frontier_ledger": _artifact_status(LEDGER_OUT),
                "frontier_launch_gate": _artifact_status(LAUNCH_GATE_OUT),
            },
        }
        canonical = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        summary["signature"] = {
            "algorithm": "sha256-json-v1",
            "digest": hashlib.sha256(canonical).hexdigest(),
        }
        os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
        with open(SUMMARY_OUT, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        return _say(True, f"signed preflight summary written to {SUMMARY_OUT}")
    except Exception as e:
        return _say(False, f"could not write signed preflight summary: {type(e).__name__}: {e}")


def verify_summary(path=SUMMARY_OUT):
    try:
        with open(path) as f:
            data = json.load(f)
        signature = data.pop("signature", {})
        expected = signature.get("digest")
        actual = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        ok = bool(expected) and expected == actual
        return _say(ok, f"preflight summary signature {'valid' if ok else 'INVALID'}: {path}")
    except Exception as e:
        return _say(False, f"could not verify preflight summary {path}: {type(e).__name__}: {e}")


def check_python():
    ok = sys.version_info[:2] == (3, 12)
    _say(ok, f"python3.12 ({sys.version.split()[0]})" if ok else
         f"wrong python: {sys.version.split()[0]} — the whole toolchain assumes python3.12")
    results = [ok]
    for pkg in REQUIRED_PY:
        try:
            m = importlib.import_module(pkg)
            try:
                v = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                v = getattr(m, "__version__", "?")
            results.append(_say(True, f"{pkg} {v}"))
        except ImportError:
            results.append(_say(False, f"{pkg} MISSING — pip install {pkg} (or restore from "
                                        f"docs/plans/studio_pinned_requirements.txt)"))
    return all(results)


def check_rust():
    results = []
    for tool in ("cargo", "rustc"):
        path = shutil.which(tool)
        results.append(_say(bool(path), f"{tool}: {path or 'MISSING — install via rustup/homebrew'}"))
    return all(results)


def check_hardware():
    results = []
    try:
        ram_gib = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                     text=True, check=True).stdout) / (1024 ** 3)
        results.append(_say(ram_gib >= MIN_RAM_GIB,
                            f"RAM {ram_gib:.0f}GiB" + ("" if ram_gib >= MIN_RAM_GIB else
                            f" — below {MIN_RAM_GIB:.0f}GiB minimum for {DEFAULT_HARDWARE.name}")))
    except Exception as e:
        results.append(_say(False, f"could not read RAM: {e}"))
    try:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        results.append(_say(free_gb >= MIN_DISK_GB,
                            f"disk free {free_gb:.0f}GB" + ("" if free_gb >= MIN_DISK_GB else
                            f" — below {MIN_DISK_GB:.0f}GB hard reserve")))
    except Exception as e:
        results.append(_say(False, f"could not read disk: {e}"))
    memory = _memory_state()
    pressure = memory.get("pressure_level")
    swap = memory.get("swap_used_mb")
    pressure_ok = pressure is None or pressure < 2
    swap_ok = swap is None or swap < 2048.0
    results.append(_say(
        pressure_ok and swap_ok,
        f"memory pressure {memory['pressure_state']} (level={pressure}), swap={swap if swap is not None else 'unknown'}MB",
    ))
    return all(results)


def check_compile():
    r = subprocess.run(["bash", "-c", "python3.12 -m py_compile tools/condense/*.py"],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    _say(ok, "all tools/condense/*.py compile clean" if ok else
         f"compile FAILED:\n{r.stderr[-800:]}")
    return ok


def check_cargo():
    r = subprocess.run(["cargo", "check", "--workspace", "--quiet"], capture_output=True, text=True)
    ok = r.returncode == 0
    _say(ok, "cargo check --workspace clean" if ok else
         f"cargo check FAILED (fix before any serve-build work):\n{r.stderr[-800:]}")
    return ok


def check_baker():
    ok = os.path.isfile(BAKER) and os.access(BAKER, os.X_OK)
    if ok:
        return _say(True, f"STRAND baker executable: {BAKER}")
    return _say(False, "STRAND baker missing — build it before P1 with: "
                "nice -n 19 cargo build --release --manifest-path "
                "vendor/strand-quant/Cargo.toml --bin quantize-model -j 2")


def check_staged_models():
    staged, missing = [], []
    for label, mdir in [("0.5B", "scratch/qwen-05b"), ("1.5B", "scratch/qwen-15b"),
                        ("7B", "scratch/qwen-7b"), ("14B", "scratch/qwen-14b"),
                        ("32B", "scratch/qwen-32b")]:
        (staged if os.path.isdir(mdir) else missing).append(label)
    _say(True, f"staged local ladder: {staged or '(none)'}")
    if missing:
        _say(True, f"not yet staged (go will skip them): {missing} — download via "
                    f"docs/RESEARCH.md's manifest when disk allows")
    return True   # informational only; missing parents don't fail preflight (go skips them)


def check_receipt_harness():
    r = subprocess.run(["python3.12", "-m", "tools.condense", "legacy",
                        "receipt_verify",
                        "receipts/official/qwen-05b-tq3.json"], capture_output=True, text=True)
    ok = r.returncode == 0
    _say(ok, "receipt harness verifies" if ok else f"receipt_verify FAILED:\n{r.stderr[-400:]}")
    return ok


def check_procurement():
    """Soft check: the fastest-SOTA download path (hf_transfer + hf_xet). Not a hard fail (you can
    still procure, just slower), but the multi-TB frontier manifest is the one real bottleneck, so
    flag a degraded path loudly. Informational -> always returns True."""
    xfer = importlib.util.find_spec("hf_transfer") is not None
    xet = importlib.util.find_spec("hf_xet") is not None
    _say(xfer, "hf_transfer (Rust chunked accelerator)" if xfer else
         "hf_transfer MISSING - pip install hf_transfer (procurement will be link-bottlenecked)")
    _say(xet, "hf_xet (Xet dedup + parallel range gets)" if xet else
         "hf_xet MISSING - pip install hf_xet (slower on Xet repos)")
    if not (xfer and xet):
        _say(True, "procurement DEGRADED but usable; export HF_HUB_ENABLE_HF_TRANSFER=1")
    else:
        _say(True, "procurement path FASTEST-SOTA (link-bound); use `python -m tools.condense legacy procure`")
    return True


def check_node_engine():
    env = _developer_environment_summary()
    ok = bool(env.get("node_engine_ok") and env.get("pnpm_ok"))
    detail = (
        f"node {env.get('node_version') or 'MISSING'} required {env.get('node_engine') or 'unknown'}; "
        f"pnpm {env.get('pnpm_version') or 'MISSING'}"
    )
    if ok:
        return _say(True, f"HIDE app engine OK: {detail}")
    return _say(False, f"HIDE app engine mismatch: {detail}")


def check_frontier_refresh():
    """Hard check: a launch-time HF refresh ledger exists for candidate/model review."""
    r = subprocess.run(["python3.12", "tools/condense/frontier_ops.py", "refresh", "--out", REFRESH_OUT],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    _say(ok, f"frontier refresh written to {REFRESH_OUT}" if ok else
         f"frontier refresh FAILED:\n{r.stdout[-500:]}{r.stderr[-800:]}")
    return ok


def check_frontier_ledger():
    """Write a refreshed frontier ledger so storage + HF metadata state is captured before launch."""
    r = subprocess.run(["python3.12", "tools/condense/frontier_ops.py", "ledger",
                        "--refresh-hf", "--out", LEDGER_OUT],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    _say(ok, f"frontier ledger written to {LEDGER_OUT}" if ok else
         f"frontier ledger FAILED:\n{r.stderr[-800:]}")
    return ok


def check_frontier_launch_gate():
    """Always materialize the frontier gate; only make it fatal for a frontier launch.

    The already-staged training ladder must not be blocked by an unaccepted giant-model license or a
    source that cannot fit this 1 TB SSD. `--require-frontier` restores hard procurement semantics.
    """
    r = subprocess.run(["python3.12", "tools/condense/frontier_ops.py", "launch-gate",
                        "--phase", "procure", "--require-refresh", REFRESH_OUT,
                        "--out", LAUNCH_GATE_OUT],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    if ok:
        _say(True, f"frontier launch gate green ({LAUNCH_GATE_OUT})")
        return True
    detail = f"frontier launch gate RED ({LAUNCH_GATE_OUT}):\n{r.stdout[-900:]}{r.stderr[-500:]}"
    if REQUIRE_FRONTIER:
        _say(False, detail)
        return False
    _say(True, detail + "\nfrontier gate is isolated: staged training may proceed; downloads remain blocked")
    return True


def main():
    checks = [
        ("Python env", check_python), ("Rust toolchain", check_rust),
        ("Hardware", check_hardware), ("HIDE app engine", check_node_engine),
        ("Tool compile", check_compile),
        ("cargo check", check_cargo), ("STRAND baker", check_baker),
        ("Staged models", check_staged_models),
        ("Procurement path", check_procurement), ("Frontier refresh", check_frontier_refresh),
        ("Frontier ledger", check_frontier_ledger),
        ("Frontier launch gate", check_frontier_launch_gate), ("Receipt harness", check_receipt_harness),
    ]
    results = {}
    for name, fn in checks:
        if not QUIET:
            print(f"\n--- {name} ---", file=sys.stderr)
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = _say(False, f"{name} raised {type(e).__name__}: {e}")
    ok = all(results.values())
    summary_ok = write_summary(results, ok)
    ok = ok and summary_ok
    print(f"\n{'='*60}", file=sys.stderr)
    if ok:
        print("PREFLIGHT GREEN — safe to run: python3.12 -m tools.condense legacy studio_run go", file=sys.stderr)
    else:
        failed = [n for n, v in results.items() if not v]
        if not summary_ok:
            failed.append("Preflight summary")
        print(f"PREFLIGHT RED — fix before `go`: {failed}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--verify-summary" in sys.argv:
        i = sys.argv.index("--verify-summary")
        path = SUMMARY_OUT
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            path = sys.argv[i + 1]
        sys.exit(0 if verify_summary(path) else 1)
    sys.exit(main())
