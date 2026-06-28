#!/usr/bin/env python3.12
"""Independent result VERIFIER — the enforcement PID.

The ladder ranks configs fast on ONE 2048-token window. That single slice is noisy and
overfittable (the doctor trains on a calibration corpus). This verifier independently
REPRODUCES the ship-candidates from scratch and re-scores them on MULTIPLE held-out windows.
A result is only VERIFIED if its reproduced multi-window degradation still holds. Independent
reproduction + multi-window eval is the strongest "this number is real" guarantee — it catches
eval noise, calibration overfit, and non-determinism in one pass.

It NEVER touches the running ladder. It waits until the ladder is idle (lock released, i.e. the
search finished, or an explicit reports/cron/VERIFY_NOW flag) so the two 7B jobs never contend
for RAM. Then it launches a separate audit_ladder run (SETNAME=verify, MULTIWINDOW, own lock/out)
on a curated candidate set and writes a verdict.

Usage (detached):  nohup python3.12 tools/condense/frontier_verifier.py >/dev/null 2>&1 &
"""
import os, sys, re, json, time, subprocess, pathlib

ROOT      = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
SRC_JSONL = "reports/cron/7b_frontier.jsonl"
MAIN_LOCK = "reports/cron/7b_frontier.lock"
VERIFY_NOW = "reports/cron/VERIFY_NOW"           # touch this to force a pass without waiting
OUT       = "reports/cron/7b_verify"
SPEC      = "reports/cron/7b_verify_spec.json"
VERDICT_MD = "reports/cron/7b_frontier_verified.md"
VERDICT_JL = "reports/cron/7b_frontier_verified.jsonl"
MODEL     = str(ROOT / "scratch/qwen-7b")
NWIN      = int(os.environ.get("VERIFY_WINDOWS", "5"))
TOL_PCT   = float(os.environ.get("VERIFY_TOL_PCT", "5.0"))   # allowed degr drift, absolute pts
POLL      = int(os.environ.get("VERIFY_POLL", "300"))


def _mp(a, f):
    return {k: a for k in ("q_proj", "k_proj", "v_proj", "o_proj")} | \
           {k: f for k in ("gate_proj", "up_proj", "down_proj")}


# fixed mixed-precision recipes (name -> (bits, alpha, rung))
_MP = {
    "mp-4a3f": (3, 0.5, _mp(4, 3)), "mp-4a2f": (2, 0.5, _mp(4, 2)),
    "mp-3a2f": (2, 0.5, _mp(3, 2)), "mp-3a1f": (1, 0.5, _mp(3, 1)),
    "mp-2a1f": (1, 0.5, _mp(2, 1)), "mp-1a0.5f": (0.5, 0.5, _mp(1, 0.5)),
}


def _base_recipe(name):
    """name (no +dr suffix) -> (fn_name, [args]) reproducing that base quant, or None if unknown."""
    if name in _MP:
        bits, alpha, rung = _MP[name]
        return "build_awq", [bits, alpha, rung]
    m = re.match(r"^([0-9.]+)-AWQ(?:\.(\d+))?$", name)
    if m:
        bits = float(m.group(1)); alpha = (int(m.group(2)) / 100) if m.group(2) else 0.5
        return "build_awq", [bits, alpha]
    m = re.match(r"^([0-9.]+)-AWQ-o(\d+)$", name)
    if m:
        return "build_awq", [float(m.group(1)), 0.5, None, float(m.group(2))]
    m = re.match(r"^([0-9.]+)-RHT$", name)
    if m:
        return "build_rht", [float(m.group(1))]
    m = re.match(r"^res([0-9.]+)\+([0-9.]+)$", name)
    if m:
        return "build_residual", [float(m.group(1)), float(m.group(2))]
    return None


def _recipe(name):
    """Full recipe for any config name, including +dr doctor variants (params parsed from name)."""
    if name == "f16":
        return "MEASURE_BASE", []
    if "+dr" in name:
        base, suf = name.split("+dr", 1)
        br = _base_recipe(base)
        if not br:
            return None
        _, bargs = br
        # base args: bits[, alpha[, rung[, outlier]]]
        bits = bargs[0]; alpha = bargs[1] if len(bargs) > 1 else 0.5
        rung = bargs[2] if len(bargs) > 2 else None
        outlier = bargs[3] if len(bargs) > 3 else 1.0
        rank = int(re.search(r"-r(\d+)", suf).group(1)) if re.search(r"-r(\d+)", suf) else 64
        steps = int(re.search(r"-(\d+)s", suf).group(1)) if re.search(r"-(\d+)s", suf) else 60
        if "-max" in suf:
            rank, steps = 128, 240
        return "build_recover", [bits, steps, rank, 1e-4, alpha, rung, outlier]
    return _base_recipe(name)


def _claimed():
    """Read the ladder's claimed results: config -> degr_pct (only those with a real ppl)."""
    out = {}
    if not os.path.exists(SRC_JSONL):
        return out
    for ln in open(SRC_JSONL):
        try:
            r = json.loads(ln)
            if "ppl" in r and "degr_pct" in r:
                out[r["config"]] = r
        except Exception:
            pass
    return out


def _pick_candidates(claimed):
    """The set worth verifying: f16 (baseline) + every config that is plausibly shippable —
    any degr below 60% (broken 1/2-bit explorations are not worth reproducing) + all +dr that
    claim to beat their base. Cap to keep a verify pass bounded."""
    cands = {"f16"}
    base_degr = {c: r["degr_pct"] for c, r in claimed.items()}
    for c, r in claimed.items():
        if c == "f16":
            continue
        if "+dr" in c:
            base = c.split("+dr", 1)[0]
            if base in base_degr and r["degr_pct"] < base_degr[base] and _recipe(c):
                cands.add(c)                       # doctor claims a win → must verify
        elif r["degr_pct"] < 60.0 and _recipe(c):
            cands.add(c)                           # plausibly shippable base
    return sorted(cands)


def _ladder_idle():
    if os.path.exists(VERIFY_NOW):
        return True
    if not os.path.exists(MAIN_LOCK):
        return True                                # run finished → safe to do heavy repro
    try:
        pid = int(open(MAIN_LOCK).read().strip())
        os.kill(pid, 0)
        return False                               # ladder alive → wait
    except Exception:
        return True                                # stale lock


def _run_verify(cands):
    spec = []
    for c in cands:
        if c == "f16":
            continue                               # f16 measured as the verify baseline directly
        rec = _recipe(c)
        if not rec:
            continue
        fn, args = rec
        if fn == "MEASURE_BASE":
            continue
        spec.append([c, fn, args])
    json.dump(spec, open(SPEC, "w"))
    env = {**os.environ,
           "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "PYTHONUNBUFFERED": "1", "BAKE_CHUNKS": "8", "BAKE_THREADS": "8",
           "DOCTOR_THREADS": str(os.cpu_count() or 8), "DOCTOR_GRAD_ACCUM": "4",
           "MULTIWINDOW": str(NWIN), "VERIFY_SPEC": str(ROOT / SPEC)}
    # separate lock/out so it never collides with the main run's checkpoint
    cmd = ["python3.12", "tools/condense/audit_ladder.py", MODEL, "7Bverify", "verify", OUT]
    print(f"# verifier: reproducing {len(spec)} candidates on {NWIN} windows", file=sys.stderr)
    subprocess.run(cmd, env=env)


def _verdict(claimed):
    """Compare reproduced multi-window degr vs the ladder's claimed single-window degr."""
    if not os.path.exists(f"{OUT}.jsonl"):
        return
    verified = {}
    for ln in open(f"{OUT}.jsonl"):
        try:
            r = json.loads(ln)
            if "ppl" in r:
                verified[r["config"]] = r
        except Exception:
            pass
    rows, jl = [], []
    for c in sorted(verified):
        v = verified[c]
        claim = claimed.get(c, {}).get("degr_pct")
        vd = v["degr_pct"]
        if claim is None:
            status = "NEW"
        elif vd <= claim + TOL_PCT:
            status = "VERIFIED" if vd <= 5.0 or "+dr" not in c else "VERIFIED"
        else:
            status = "FLAGGED"                      # reproduced much worse → claim was noise/overfit
        rows.append((c, v.get("eff_bpw"), claim, vd, status))
        jl.append({"config": c, "eff_bpw": v.get("eff_bpw"), "claimed_degr": claim,
                   "verified_degr": vd, "windows": NWIN, "status": status})
    with open(VERDICT_JL, "w") as f:
        for r in jl:
            f.write(json.dumps(r) + "\n")
    with open(VERDICT_MD, "w") as f:
        f.write(f"# 7B Frontier — VERIFIED results ({NWIN}-window independent reproduction)\n\n")
        f.write("| config | bpw | claimed degr | verified degr | status |\n|---|--:|--:|--:|---|\n")
        for c, bpw, claim, vd, status in sorted(rows, key=lambda x: (x[3])):
            cs = f"+{claim:.1f}%" if claim is not None else "—"
            f.write(f"| {c} | {bpw} | {cs} | +{vd:.1f}% | {status} |\n")
    print(f"# verifier: wrote {VERDICT_MD} ({len(rows)} verdicts)", file=sys.stderr)


def main():
    print(f"# verifier up pid={os.getpid()} windows={NWIN} tol={TOL_PCT}pts poll={POLL}s", file=sys.stderr)
    while True:
        claimed = _claimed()
        cands = _pick_candidates(claimed)
        if len(cands) <= 1:                         # only f16 → nothing to verify yet
            time.sleep(POLL); continue
        if not _ladder_idle():
            time.sleep(POLL); continue
        print(f"# verifier: ladder idle, {len(cands)} candidates: {cands}", file=sys.stderr)
        _run_verify(cands)
        _verdict(claimed)
        if os.path.exists(VERIFY_NOW):
            os.remove(VERIFY_NOW)
        # if the main run is fully done, one authoritative pass is enough
        if not os.path.exists(MAIN_LOCK):
            print("# verifier: main run finished, verdict written, exiting", file=sys.stderr)
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
