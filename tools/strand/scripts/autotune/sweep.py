#!/usr/bin/env python3
"""The idle autotuner sweep engine — measures the TUNABLE registry, writes the profile.

Usage:
    python3 tools/autotune/sweep.py [--only NAME] [--reps N] [--out PATH] [--list]

For every ENABLED tunable in tunables.TUNABLES it runs the declared measurement
command per candidate value (kind="sweep") or once for all candidates
(kind="batch"), keeps a sample ONLY if the tunable's guard passes (decode gates
assert bit-identity before printing perf; encode adds a result-invariance
fingerprint across thread counts), takes best-of---reps, and writes
research/tuned-profile.toml with {machine fingerprint, best value, full evidence}.

THE CONTRACT (do not violate):
  * the tuner NEVER changes a default in code — it only writes the profile;
  * consumers OPT IN by reading the profile (tools/autotune/apply.py prints the
    env/flags for each known launcher);
  * all timings are ADVISORY and machine-stamped — the profile is only valid for
    the fingerprint it names, and only as good as how quiet the box was
    (ops/autotune.sh provides the idle gating; running this by hand is safe but
    the numbers inherit whatever contention exists).

Exit codes: 0 = profile written; 1 = a guard/invariance FAILED (profile still
written, failing tunables marked); 2 = nothing runnable (all SKIP, no profile).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunables import TUNABLES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_OUT = os.path.join(ROOT, "research", "tuned-profile.toml")
RUN_TIMEOUT = 600  # seconds per measurement command


# ---------------------------------------------------------------- machine identity

def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def machine_identity():
    ident = {
        "hw_model": _sysctl("hw.model") or platform.machine(),
        "cpu_brand": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count() or 0,
        "mem_bytes": int(_sysctl("hw.memsize") or 0),
        "os": f"{platform.system()} {platform.release()}",
    }
    try:
        ident["rustc"] = subprocess.run(["rustc", "-V"], capture_output=True, text=True,
                                        timeout=10).stdout.strip()
    except Exception:
        ident["rustc"] = "unknown"
    fp = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return ident, fp


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- measurement

_RUN_CACHE = {}  # (argv tuple, env tuple) -> (rc, text, secs) — batch tunables share runs


def run_cmd(argv, env_over, log):
    key = (tuple(argv), tuple(sorted(env_over.items())))
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    env = dict(os.environ, **env_over)
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=RUN_TIMEOUT)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        rc, text = -1, "(timeout)"
    except OSError as e:
        rc, text = -1, f"(spawn failed: {e})"
    secs = time.monotonic() - t0
    log(f"    ran {' '.join(argv)} {dict(env_over)} -> rc={rc} ({secs:.1f}s)")
    _RUN_CACHE[key] = (rc, text, secs)
    return rc, text, secs


def measure_tunable(t, reps, log):
    """Returns a result dict for the profile. status: TUNED|SKIP|FAIL|DISABLED."""
    res = {"name": t["name"], "status": None, "default": t["default"],
           "direction": t["direction"], "metric": t["metric"], "notes": t["notes"],
           "values": t["values"], "evidence": [], "best": None, "guard": ""}

    if t["status"] == "disabled":
        res["status"] = "DISABLED"
        return res

    missing = [r for r in t["requires"] if not os.path.exists(os.path.join(ROOT, r))]
    if missing:
        res["status"] = "SKIP"
        res["guard"] = f"missing: {', '.join(missing)} (sibling-wave bin absent? degrade, not fail)"
        log(f"  SKIP {t['name']}: {res['guard']}")
        return res

    pick = max if t["direction"] == "max" else min
    inv_keys = set()
    samples = {}  # value -> [metric, ...]

    if t["kind"] == "batch":
        argv, env = t["cmd"]()
        for rep in range(reps):
            if rep > 0:
                _RUN_CACHE.pop((tuple(argv), tuple(sorted(env.items()))), None)
            rc, text, _ = run_cmd(argv, env, log)
            ok, why = t["guard"](text)
            if rc != 0 or not ok:
                res["status"] = "FAIL"
                res["guard"] = f"guard failed (rc={rc}): {why}"
                log(f"  FAIL {t['name']}: {res['guard']}")
                return res
            res["guard"] = why
            for v, m in (t["parse"](text) or {}).items():
                samples.setdefault(v, []).append(m)
    else:  # sweep
        for v in t["values"]:
            argv, env = t["cmd"](v)
            for rep in range(reps):
                if rep > 0:
                    _RUN_CACHE.pop((tuple(argv), tuple(sorted(env.items()))), None)
                rc, text, _ = run_cmd(argv, env, log)
                ok, why = t["guard"](text)
                if rc != 0 or not ok:
                    res["status"] = "FAIL"
                    res["guard"] = f"guard failed at value={v} (rc={rc}): {why}"
                    log(f"  FAIL {t['name']}: {res['guard']}")
                    return res
                res["guard"] = why
                if t.get("invariance"):
                    inv_keys.add(t["invariance"](text))
                m = t["parse"](text)
                if m is None:
                    res["status"] = "FAIL"
                    res["guard"] = f"metric unparseable at value={v}"
                    return res
                samples.setdefault(v, []).append(m)

    if t.get("invariance") and len(inv_keys) > 1:
        res["status"] = "FAIL"
        res["guard"] = (f"INVARIANCE BROKEN: {len(inv_keys)} distinct result fingerprints "
                        f"across the sweep — the tunable changes the output; this is a bug, "
                        f"not a tuning point")
        log(f"  FAIL {t['name']}: {res['guard']}")
        return res
    if t.get("invariance") and inv_keys:
        res["guard"] += "; result-invariance OK (1 fingerprint across all values)"

    if not samples:
        res["status"] = "FAIL"
        res["guard"] = "no samples parsed"
        return res

    # best-of-reps per value (max-direction keeps the max sample, min keeps the min:
    # the cleanest estimate under contention), then pick across values.
    summary = {v: pick(ms) for v, ms in samples.items()}
    res["evidence"] = sorted(summary.items())
    res["best"] = pick(summary, key=summary.get)
    res["status"] = "TUNED"
    bm, dm = summary[res["best"]], summary.get(t["default"])
    gain = (f", {bm / dm:.2f}x vs default" if dm and t["direction"] == "max"
            else f", {dm / bm:.2f}x vs default" if dm else "")
    log(f"  TUNED {t['name']}: best={res['best']} ({bm:g}){gain}")
    return res


# ---------------------------------------------------------------- profile writer

def _toml_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_profile(path, ident, fp, results, reps, secs):
    lines = [
        "# research/tuned-profile.toml — written by tools/autotune/sweep.py. DO NOT hand-edit.",
        "# ADVISORY, machine-stamped: valid only for [meta].machine_fingerprint below, and only",
        "# as quiet as the box was. Consumers OPT IN via tools/autotune/apply.py — no default in",
        "# code ever reads this file implicitly.",
        "",
        "[meta]",
        f"generated = {_toml_str(time.strftime('%Y-%m-%dT%H:%M:%S%z'))}",
        f"git = {_toml_str(git_head())}",
        f"machine_fingerprint = {_toml_str(fp)}",
        "advisory = true",
        f"reps = {reps}",
        f"sweep_secs = {int(secs)}",
        "",
        "[machine]",
    ]
    for k, v in ident.items():
        lines.append(f"{k} = {v if isinstance(v, int) else _toml_str(str(v))}")
    for r in results:
        lines += ["", f"[tunable.{r['name']}]", f"status = {_toml_str(r['status'])}"]
        if r["best"] is not None:
            lines.append(f"best = {r['best']}")
        lines += [
            f"default = {r['default']}",
            f"direction = {_toml_str(r['direction'])}",
            f"values = {r['values']}",
            f"metric = {_toml_str(r['metric'])}",
            f"guard = {_toml_str(r['guard'])}",
            "evidence = " + "[" + ", ".join(f"[{v}, {m:g}]" for v, m in r["evidence"]) + "]",
            f"notes = {_toml_str(r['notes'])}",
        ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="run a single tunable by name")
    ap.add_argument("--reps", type=int, default=2, help="samples per point (best-of)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="list the registry and exit")
    args = ap.parse_args()

    def log(msg):
        print(f"[autotune {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.list:
        for t in TUNABLES:
            print(f"{t['name']:<22} {t['status']:<9} kind={t['kind']:<6} "
                  f"values={t['values']} default={t['default']}")
        return 0

    todo = [t for t in TUNABLES if not args.only or t["name"] == args.only]
    if args.only and not todo:
        log(f"no tunable named {args.only!r}")
        return 2

    ident, fp = machine_identity()
    log(f"machine {ident['hw_model']} / {ident['cpu_brand']} / {ident['logical_cpus']} cores"
        f" -> fingerprint {fp}")
    t0 = time.monotonic()
    results = [measure_tunable(t, args.reps, log) for t in todo]
    secs = time.monotonic() - t0

    statuses = [r["status"] for r in results]
    if not any(s == "TUNED" for s in statuses):
        log("nothing tuned (all SKIP/DISABLED/FAIL) — no profile written")
        # honesty rule (replay.sh precedent): "nothing measured" must not look like a profile
        return 2 if not any(s == "FAIL" for s in statuses) else 1

    write_profile(args.out, ident, fp, results, args.reps, secs)
    log(f"profile written: {args.out} ({int(secs)}s)")
    summary = " ".join(f"{r['name']}={r['status']}"
                       + (f"(best={r['best']})" if r["best"] is not None else "")
                       for r in results)
    log(f"summary: {summary}")
    return 1 if any(s == "FAIL" for s in statuses) else 0


if __name__ == "__main__":
    sys.exit(main())
