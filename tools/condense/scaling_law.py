#!/usr/bin/env python3.12
"""scaling_law.py — find each model's bit-floor, then fit the floor-vs-scale curve (plan §4 / T3.1).

Two modes:
  --floor <label> <model_ladder.jsonl> <floors_out.jsonl>
      Read a model's measured results (PTQ + recovered), pick its FLOOR = the lowest EFFECTIVE bpw
      whose degradation is <= the +2% (~1:1) gate, append one datapoint to the floors file.
  --fit <floors.jsonl>
      Regress floor vs log10(params) across the ladder, decide H1 (monotone descent) vs H0 (flat),
      and extrapolate the 70B/405B floor as a PRE-REGISTERED prediction (a result only once an
      off-box run confirms it). Writes a markdown curve report next to the jsonl.

Proof discipline: effective bpw only; 0.5B/1.5B are lab points (printed, but the verdict is read
off 7B+); report honestly whether the floor descends or is flat.
"""
import sys, json, math, os, hashlib, subprocess

GATE = float(os.environ.get("FLOOR_GATE_PCT", "2.0"))      # the ~1:1 quality gate
Q4K_BPW = 4.5                                              # the llama Q4_K reference


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _src_hash(model_dir):
    """Stable 64-hex id of a parent: the shard index for sharded models (fast), else model.safetensors."""
    for cand in ("model.safetensors.index.json", "model.safetensors"):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            return _sha256(p)
    return "0" * 64


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _machine():
    try:
        gb = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                      text=True).stdout) / 1e9)
    except Exception:
        gb = 0
    cls = f"Studio-{gb}" if gb >= 64 else f"M-{gb}"
    return f"Apple Silicon, {gb}GB unified", cls


def _suite_hash():
    for p in ("receipts/prompt_suite_v1.sha256", "prompts/frozen/suite_v1.sha256"):
        if os.path.exists(p):
            return open(p).read().strip().split()[0]
    return None


def emit_receipt(label, rec, model_dir, jsonl):
    """Write a schema-valid floor receipt (receipts/official/<label>-floor.json). 7B+ floor points
    are scale-points (R2); labs are baselines (R1, never set the verdict)."""
    params = PARAMS.get(label, 0)
    machine, mclass = _machine()
    is_lab = params < 7.0
    bpw = rec.get("floor_bpw")
    degr = rec.get("degr_pct")
    gate = "pass" if (bpw and degr is not None and degr <= GATE) else ("warn" if bpw else "fail")
    r = {
        "project": "hawking", "receipt_version": "0.2",
        "repro_level": "R1" if is_lab else "R2",
        "claim_type": "baseline" if is_lab else "scale-point",
        "machine": machine, "machine_class": mclass,
        "model_family": "qwen", "source_model": f"{label} ({model_dir})",
        "source_sha256": _src_hash(model_dir), "source_precision": "bf16",
        "condensed_artifact": f"{rec.get('winning_config','none')} @ {bpw} eff-bpw ({jsonl})",
        "artifact_sha256": _sha256(jsonl) if os.path.exists(jsonl) else "0" * 64,
        "effective_bpw": float(bpw) if bpw else 0.0,
        "nominal_bpw": round(float(bpw)) if bpw else 0.0,
        "peak_rss_gb": _peak_rss_gb(label),
        "multiwindow_n": int(os.environ.get("MULTIWINDOW", "4")),
        "quality_gate": gate,
        "hawking_commit": _commit(),
        "commands": [f"python3.12 tools/condense/studio_run.py --model {label}",
                     f"python3.12 tools/condense/scaling_law.py --floor {label} {jsonl} {os.path.basename(jsonl)}"],
        "notes": (f"Bit-floor datapoint for the §4 scale curve. floor = lowest effective bpw at "
                  f"<= +{GATE}% ppl (multiwindow). {'LAB rung - never sets the verdict (§0.5).' if is_lab else ''} "
                  f"beats-Q4_K={'yes' if (bpw and bpw < Q4K_BPW) else 'no'}."),
    }
    sh = _suite_hash()
    if sh:
        r["prompt_suite_hash"] = sh; r["prompt_suite_version"] = "v1"
    out = f"receipts/official/{label}-floor.json"
    os.makedirs("receipts/official", exist_ok=True)
    open(out, "w").write(json.dumps(r, indent=2) + "\n")
    print(f"[receipt] wrote {out} ({r['claim_type']}, {r['quality_gate']})", file=sys.stderr)


def _peak_rss_gb(label):
    """Best-effort: read the scheduler's measured peak for this job if available, else 0.0."""
    p = "reports/cron/ram_actuals.jsonl"
    if os.path.exists(p):
        best = 0.0
        for ln in open(p):
            try:
                r = json.loads(ln)
                if r.get("name") == label:
                    best = max(best, r.get("peak_gb", 0.0))
            except Exception:
                pass
        return round(best, 2)
    return 0.0
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "7B": 7.0, "14B": 14.0, "32B": 32.0,
          "70B": 70.0, "72B": 72.0, "405B": 405.0}


def find_floor(label, jsonl):
    """Lowest effective bpw at <= +GATE% degradation among all measured configs for this model."""
    best = None
    for ln in open(jsonl):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "ppl" not in r or "eff_bpw" not in r or r.get("config") == "f16":
            continue
        bpw, degr = r.get("eff_bpw"), r.get("degr_pct")
        if degr is None or bpw is None:
            continue
        if degr <= GATE and (best is None or bpw < best[0]):
            best = (bpw, r.get("config"), degr)
    return best


def cmd_floor(label, jsonl, out, model_dir=None):
    if not os.path.exists(jsonl):
        print(f"[floor] {label}: no results at {jsonl} yet", file=sys.stderr); return
    fl = find_floor(label, jsonl)
    if not fl:
        print(f"[floor] {label}: no config reached the +{GATE}% gate "
              f"(floor not yet found — recovery stack incomplete?)", file=sys.stderr)
        rec = {"model": label, "params_b": PARAMS.get(label), "floor_bpw": None,
               "gate_pct": GATE, "note": "no config within gate"}
    else:
        bpw, cfg, degr = fl
        rec = {"model": label, "params_b": PARAMS.get(label), "floor_bpw": bpw,
               "winning_config": cfg, "degr_pct": degr, "gate_pct": GATE}
        print(f"[floor] {label}: floor = {bpw:.3f} eff-bpw via {cfg} (+{degr}%)", file=sys.stderr)
    # de-dup: rewrite without any prior row for this label, then append
    rows = []
    if os.path.exists(out):
        rows = [ln for ln in open(out)
                if _safe_model(ln) != label]
    with open(out, "w") as f:
        for ln in rows:
            f.write(ln if ln.endswith("\n") else ln + "\n")
        f.write(json.dumps(rec) + "\n")
    if model_dir:
        try:
            emit_receipt(label, rec, model_dir, jsonl)
        except Exception as e:
            print(f"[receipt] {label}: emit failed ({e})", file=sys.stderr)


def _safe_model(ln):
    try:
        return json.loads(ln).get("model")
    except Exception:
        return None


def _linfit(xs, ys):
    """Least-squares y = m x + b (pure python; no numpy dependency). Returns (m, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx if sxx else 0.0
    b = my - m * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return m, b, r2


def cmd_fit(floors):
    pts = []
    for ln in open(floors):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("floor_bpw") and r.get("params_b"):
            pts.append((r["params_b"], r["floor_bpw"], r["model"]))
    pts.sort()
    if len(pts) < 2:
        print(f"[fit] need >=2 floor points, have {len(pts)} — run more rungs first", file=sys.stderr)
        return
    # verdict is read off 7B+; labs (<7B) shown but excluded from the law (they floor ~3-bit, §0 rule 5)
    big = [(p, f) for (p, f, m) in pts if p >= 7.0]
    xs = [math.log10(p) for p, f in big]
    ys = [f for p, f in big]
    m, b, r2 = _linfit(xs, ys)
    descends = m < -0.05          # meaningful negative slope
    pred = {N: m * math.log10(N) + b for N in (70, 405)}
    md = floors.replace(".jsonl", "_curve.md")
    with open(md, "w") as o:
        o.write("# Bit-floor vs scale (plan §4 / T3.1)\n\n")
        o.write(f"Gate: <= +{GATE}% ppl vs f16 parent, effective bpw, multiwindow.\n\n")
        o.write("| model | params (B) | floor eff-bpw | role |\n|---|--:|--:|---|\n")
        for p, f, mlabel in pts:
            role = "lab (not in fit)" if p < 7.0 else "verdict"
            o.write(f"| {mlabel} | {p} | {f:.3f} | {role} |\n")
        o.write(f"\n**Law (7B+):** floor ~= {m:.3f}*log10(N) + {b:.3f}  (R^2={r2:.3f})\n\n")
        o.write(f"**Verdict:** {'H1 CONFIRMED - floor DESCENDS with scale' if descends else 'H0 - floor ~FLAT, redundancy buys little'} "
                f"(slope {m:.3f} bpw/decade)\n\n")
        o.write("**Pre-registered extrapolation (a PREDICTION until an off-box run confirms it):**\n")
        o.write(f"- 70B  -> ~{pred[70]:.2f} eff-bpw\n")
        o.write(f"- 405B -> ~{pred[405]:.2f} eff-bpw {'(< 1-bit territory!)' if pred[405] < 1 else ''}\n")
    print(f"[fit] {len(big)} verdict points, slope {m:.3f}/decade, R^2 {r2:.3f} -> {md}", file=sys.stderr)
    print(f"[fit] {'H1 (descends)' if descends else 'H0 (flat)'}; 70B~{pred[70]:.2f}bpw 405B~{pred[405]:.2f}bpw",
          file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else ""
    if a == "--floor":
        cmd_floor(sys.argv[2], sys.argv[3], sys.argv[4],
                  sys.argv[5] if len(sys.argv) > 5 else None)
    elif a == "--fit":
        cmd_fit(sys.argv[2])
    else:
        print(__doc__)
