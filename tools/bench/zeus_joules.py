#!/usr/bin/env python3
# zeus_joules.py — MEASURED per-domain J/tok for Qwen-3B decode (Wave-6).
#
# Replaces the cpu-encode-fraction PROXY (phase_joules.sh Step 3) with a REAL
# per-domain energy reading from `zeus-apple-silicon`, which subscribes to the
# macOS IOReport "Energy Model" channel group (CPU / GPU / GPU-SRAM / DRAM / ANE,
# 1 mJ resolution, SUDO-FREE — no kext, no password). It wraps exactly one
# steady-state `dismantle generate` decode under the LOCKED fast-path env in a
# begin_window/end_window pair, then divides each domain's mJ by the number of
# tokens the [stats] line reports => measured gpu_mj/tok, dram_mj/tok, etc.
#
# This makes the race-to-idle / energy story MEASURED rather than inferred from
# CPU-encode time. No engine/kernel/Cargo change — pure measurement.
#
#   pip install zeus-apple-silicon     (latest 1.1.0; Python>=3.9; macOS 11+ ARM64)
#
# INTERPRETER NOTE: the rest of tools/bench/ calls /usr/bin/python3 (3.9.6). The
# zeus package must be installed in WHICHEVER interpreter runs THIS file. Default
# is /usr/bin/python3 for suite consistency; override with --python / ZEUS_PYTHON
# (e.g. your Homebrew python3) and install zeus there. On a missing import this
# script prints the exact pip recipe for the SAME interpreter and exits non-zero.
#
# Usage:
#   tools/bench/zeus_joules.py --tokens 256
#   tools/bench/zeus_joules.py --tokens 64 --json
#   ZEUS_PYTHON=/opt/homebrew/bin/python3 tools/bench/zeus_joules.py --tokens 256
#
# Exit codes: 0 ok; 3 zeus not importable (prints install recipe); 4 no [stats]
# line / decode failed; 64 usage error.
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIN = os.environ.get("BIN", str(REPO / "target/release/hawking"))
WEIGHTS = os.environ.get("WEIGHTS", str(REPO / "models/qwen2.5-3b-instruct-q4_k_m.gguf"))
PROFILE = os.environ.get("PROFILE", str(REPO / "profiles/qwen3b-instruct-q4k.m3pro18.json"))

# LOCKED fast-path env — MUST stay byte-identical to measure_joules.sh L42-44 and
# phase_joules.sh L44-46 so the measured energy describes the SHIPPED decode.
BASE_ENV = {
    "HAWKING_QWEN_TCB": "1",
    "HAWKING_QWEN_VOCAB_PRUNE": "32000",
    "HAWKING_QWEN_Q4K_LMHEAD": "1",
    "HAWKING_QWEN_FFN_DOWN_Q4K": "1",
    "HAWKING_QWEN_Q4K_PREDEC": "1",
}

DEFAULT_PROMPT = "fn fibonacci(n: u64) -> u64 {"

# Domain -> tuple of candidate attribute names. The shipped 1.1.0 wheel and the
# RFC/C++ struct disagree on names (gpu_mj vs soc_gpu, dram_mj vs soc_dram, ...);
# we accept either so the harness is correct regardless of which is exposed.
DOMAINS = {
    "gpu":      ("gpu_mj", "soc_gpu"),
    "dram":     ("dram_mj", "soc_dram"),
    "cpu":      ("cpu_total_mj", "cpu_total"),
    "gpu_sram": ("gpu_sram_mj", "soc_gpu_sram"),
    "ane":      ("ane_mj", "ane"),
}


def _get(metrics, names):
    """First non-None value among attribute/dict-key `names`, else None."""
    for n in names:
        v = getattr(metrics, n, None)
        if v is None and isinstance(metrics, dict):
            v = metrics.get(n)
        if v is not None:
            # some builds return a list (per-cluster); sum it for a scalar mJ
            if isinstance(v, (list, tuple)):
                try:
                    return float(sum(x for x in v if x is not None))
                except Exception:
                    continue
            try:
                return float(v)
            except Exception:
                continue
    return None


def import_zeus():
    try:
        from zeus_apple_silicon import AppleEnergyMonitor  # type: ignore
        return AppleEnergyMonitor
    except Exception as e:  # ModuleNotFoundError or load error
        py = sys.executable
        sys.stderr.write(
            "zeus-apple-silicon is not importable from this interpreter.\n"
            f"  interpreter : {py}\n"
            f"  import error: {e!r}\n"
            "Install it INTO THIS interpreter (sudo-free), then re-run:\n"
            f"  {py} -m pip install zeus-apple-silicon\n"
            "If you meant a different python (e.g. Homebrew), re-run with:\n"
            "  ZEUS_PYTHON=/path/to/python3 tools/bench/zeus_joules.py ...\n"
            "  (or  --python /path/to/python3)  and install zeus there.\n"
        )
        sys.exit(3)


def reexec_if_needed(args):
    """If --python/ZEUS_PYTHON names a different interpreter, re-exec under it so
    the zeus import happens in the interpreter the user installed it into."""
    target = args.python or os.environ.get("ZEUS_PYTHON")
    if not target:
        return
    target = str(Path(target))
    if os.path.realpath(target) == os.path.realpath(sys.executable):
        return
    # strip the flag so the child doesn't loop, mark that we've re-execed
    child_argv = [a for a in sys.argv[1:] if a != "--python"]
    child_argv = [a for a in child_argv if a != target]
    env = dict(os.environ)
    env.pop("ZEUS_PYTHON", None)
    env["ZEUS_REEXEC"] = "1"
    os.execve(target, [target, os.path.abspath(__file__), *child_argv], env)


def run_decode(tokens, prompt):
    """Spawn the locked-fast-path decode; return (stdout+stderr text)."""
    if not (os.access(BIN, os.X_OK)):
        sys.stderr.write(f"error: binary not found/executable: {BIN} (cargo build --release?)\n")
        sys.exit(64)
    if not Path(WEIGHTS).is_file():
        sys.stderr.write(f"error: weights not found: {WEIGHTS}\n")
        sys.exit(64)
    env = dict(os.environ)
    env.update(BASE_ENV)
    # match the suite's co-existence wrapper: nice -n 19 taskpolicy -b
    cmd = [
        "nice", "-n", "19", "taskpolicy", "-b", BIN, "generate",
        "--weights", WEIGHTS, "--kernel-profile", PROFILE,
        "--prompt", prompt, "--max-new-tokens", str(tokens),
        "--temperature", "0", "--seed", "0",
    ]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return proc.stdout or ""


def parse_stats(text):
    line = None
    for ln in text.splitlines():
        if "[stats]" in ln:
            line = ln
    if line is None:
        return None
    def grab(pat):
        m = re.search(pat, line)
        return m.group(1) if m else None
    comp = grab(r"completion=(\d+)")
    dec_ms = grab(r"decode_ms=([0-9.]+)")
    dec_tps = grab(r"dec_tps=([0-9.]+)")
    return {
        "completion": int(comp) if comp else None,
        "decode_ms": float(dec_ms) if dec_ms else None,
        "dec_tps": float(dec_tps) if dec_tps else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Measured per-domain J/tok via zeus-apple-silicon (IOReport).")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--python", default=None,
                    help="interpreter that has zeus-apple-silicon installed (re-exec target)")
    ap.add_argument("--json", action="store_true", help="emit a single JSON object on stdout")
    ap.add_argument("--window", default="decode", help="zeus window label")
    args = ap.parse_args()

    if os.environ.get("ZEUS_REEXEC") != "1":
        reexec_if_needed(args)

    AppleEnergyMonitor = import_zeus()

    monitor = AppleEnergyMonitor()
    monitor.begin_window(args.window)
    text = run_decode(args.tokens, args.prompt)
    metrics = monitor.end_window(args.window)

    stats = parse_stats(text)
    if not stats or not stats.get("completion"):
        sys.stderr.write("error: no [stats] line / zero completion tokens. Raw tail:\n")
        sys.stderr.write("\n".join(text.splitlines()[-5:]) + "\n")
        sys.exit(4)
    toks = stats["completion"]

    per_domain_mj = {d: _get(metrics, names) for d, names in DOMAINS.items()}
    per_tok = {d: (v / toks if v is not None else None) for d, v in per_domain_mj.items()}

    # package proxy total for the cross-check vs macmon (sum of available domains)
    avail = [v for v in per_domain_mj.values() if v is not None]
    total_mj = sum(avail) if avail else None

    out = {
        "interpreter": sys.executable,
        "tokens": toks,
        "decode_ms": stats.get("decode_ms"),
        "dec_tps": stats.get("dec_tps"),
        "window": args.window,
        "mj_total": per_domain_mj,
        "mj_per_tok": per_tok,
        "sum_mj": total_mj,
        "sum_mj_per_tok": (total_mj / toks if total_mj is not None else None),
        "source": "zeus-apple-silicon (IOReport Energy Model)",
    }

    if args.json:
        print(json.dumps(out))
        return

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else ("None" if x is None else str(x))
    print("=== zeus_joules — MEASURED per-domain energy (IOReport Energy Model) ===")
    print(f"  interpreter   : {sys.executable}")
    print(f"  dec_tps       : {fmt(stats.get('dec_tps'))}")
    print(f"  tokens        : {toks}")
    print(f"  decode_ms     : {fmt(stats.get('decode_ms'))}")
    print("  --- per-domain mJ / mJ-per-token (None = not exposed on this chip) ---")
    for d in ("gpu", "dram", "gpu_sram", "cpu", "ane"):
        print(f"    {d:9s} : {fmt(per_domain_mj[d])} mJ   ->  {fmt(per_tok[d])} mJ/tok")
    print("    -------------------------------------------------")
    print(f"    SUM       : {fmt(total_mj)} mJ   ->  {fmt(out['sum_mj_per_tok'])} mJ/tok")
    print()
    print("  measured: J/tok_GPU  = gpu_mj / tokens / 1000;  J/tok_DRAM = dram_mj / tokens / 1000")
    print("  (replaces phase_joules.sh Step-3 cpu-encode PROXY with a real per-domain reading)")


if __name__ == "__main__":
    main()

