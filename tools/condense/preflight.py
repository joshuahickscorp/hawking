#!/usr/bin/env python3.12
"""preflight.py — run this FIRST on the Mac Studio, before `studio_run.py go`.

Confirms the whole environment is actually ready: Python deps, Rust toolchain, disk space,
staged model parents, and the receipt harness — so `go` never dies 10 minutes in on a missing
dependency. Exits 0 (green, safe to `go`) or 1 (red, prints exactly what to fix). Pure stdlib +
best-effort imports; never crashes on a missing optional dep, it just flags it.

Usage:
  python3.12 tools/condense/preflight.py            # full check
  python3.12 tools/condense/preflight.py --quiet    # exit code only, minimal output
"""
import sys, os, subprocess, shutil, importlib, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
QUIET = "--quiet" in sys.argv
REQUIRED_PY = ["torch", "transformers", "safetensors", "numpy", "jsonschema"]
MIN_DISK_GB = 100.0     # comfortable headroom for the 0.5B-32B ladder + a first frontier download
MIN_RAM_GB = 64.0       # below this, this isn't the 96GB Studio the plan assumes


def _say(ok, msg):
    if not QUIET:
        print(f"[{'OK ' if ok else 'FAIL'}] {msg}", file=sys.stderr)
    return ok


def check_python():
    ok = sys.version_info[:2] == (3, 12)
    _say(ok, f"python3.12 ({sys.version.split()[0]})" if ok else
         f"wrong python: {sys.version.split()[0]} — the whole toolchain assumes python3.12")
    results = [ok]
    for pkg in REQUIRED_PY:
        try:
            m = importlib.import_module(pkg)
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
        ram_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                    text=True, check=True).stdout) / 1e9
        results.append(_say(ram_gb >= MIN_RAM_GB,
                            f"RAM {ram_gb:.0f}GB" + ("" if ram_gb >= MIN_RAM_GB else
                            f" — below {MIN_RAM_GB:.0f}GB; the plan's RAM budgets assume the 96GB Studio")))
    except Exception as e:
        results.append(_say(False, f"could not read RAM: {e}"))
    try:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        results.append(_say(free_gb >= MIN_DISK_GB,
                            f"disk free {free_gb:.0f}GB" + ("" if free_gb >= MIN_DISK_GB else
                            f" — below {MIN_DISK_GB:.0f}GB minimum")))
    except Exception as e:
        results.append(_say(False, f"could not read disk: {e}"))
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


def check_staged_models():
    staged, missing = [], []
    for label, mdir in [("0.5B", "scratch/qwen-05b"), ("1.5B", "scratch/qwen-15b"),
                        ("7B", "scratch/qwen-7b"), ("14B", "scratch/qwen-14b"),
                        ("32B", "scratch/qwen-32b")]:
        (staged if os.path.isdir(mdir) else missing).append(label)
    _say(True, f"staged local ladder: {staged or '(none)'}")
    if missing:
        _say(True, f"not yet staged (go will skip them): {missing} — download via "
                    f"BASELINES.md's manifest when disk allows")
    return True   # informational only; missing parents don't fail preflight (go skips them)


def check_receipt_harness():
    r = subprocess.run(["python3.12", "tools/condense/receipt_verify.py",
                        "receipts/official/qwen-05b-tq3.json"], capture_output=True, text=True)
    ok = r.returncode == 0
    _say(ok, "receipt harness verifies" if ok else f"receipt_verify FAILED:\n{r.stderr[-400:]}")
    return ok


def check_procurement():
    """Soft check: the fastest-SOTA download path (hf_transfer + hf_xet). Not a hard fail (you can
    still procure, just slower), but the ~4 TB of frontier parents are the one real bottleneck, so
    flag a degraded path loudly. Informational -> always returns True."""
    xfer = importlib.util.find_spec("hf_transfer") is not None
    xet = importlib.util.find_spec("hf_xet") is not None
    _say(xfer, "hf_transfer (Rust chunked accelerator)" if xfer else
         "hf_transfer MISSING - pip install hf_transfer (procurement will be link-bottlenecked)")
    _say(xet, "hf_xet (Xet dedup + parallel range gets)" if xet else
         "hf_xet MISSING - pip install hf_xet (slower on Xet repos)")
    if not (xfer and xet):
        _say(True, "procurement DEGRADED but usable; export HF_HUB_ENABLE_HF_TRANSFER=1 and see procure.py")
    else:
        _say(True, "procurement path FASTEST-SOTA (link-bound); use tools/condense/procure.py")
    return True


def main():
    checks = [
        ("Python env", check_python), ("Rust toolchain", check_rust),
        ("Hardware", check_hardware), ("Tool compile", check_compile),
        ("cargo check", check_cargo), ("Staged models", check_staged_models),
        ("Procurement path", check_procurement), ("Receipt harness", check_receipt_harness),
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
    print(f"\n{'='*60}", file=sys.stderr)
    if ok:
        print("PREFLIGHT GREEN — safe to run: python3.12 tools/condense/studio_run.py go", file=sys.stderr)
    else:
        failed = [n for n, v in results.items() if not v]
        print(f"PREFLIGHT RED — fix before `go`: {failed}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
