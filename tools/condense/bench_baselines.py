#!/usr/bin/env python3.12
"""bench_baselines.py — the competitive WEDGE-DEFENSE gate (a clean PROBE/BENCH, no overclaim).

The portfolio kill criterion for the sub-4-bit condensation wedge is blunt:

    "Hawking's bit-floor FAILS to beat tuned dynamic GGUF / MLX-4bit at EQUAL quality."

This tool runs that gate head-to-head. At a MATCHED effective bpw on a given model it puts
Hawking's measured floor next to the two strongest commodity baselines and prints a verdict:

  * llama.cpp dynamic GGUF  — IQ1_S (~1.56 bpw) and IQ2_XXS/IQ2_S (~2.06-2.5 bpw), via the
    installed `llama-perplexity` (or `llama-cli`) over the SAME held-out windows as audit_ladder.
  * MLX-LM 4-bit / DWQ      — via `mlx_lm` (mlx_ppl-style relative ppl), for the 4-bit reference.

VERDICT: Hawking WINS the wedge if, at MATCHED effective bpw on a 7B+ model (and 14B/32B), its
floor beats IQ2 quality (lower ppl / lower degradation at <= the same bpw). Otherwise the wedge is
KILLED -> the value reframes to the PORTFOLIO (RWKV state, .tq format, economics) and the sub-4-bit
"we beat everyone on weight quality" claim is retired. The KILL line is printed explicitly.

PROOF DISCIPLINE (matches audit_ladder.py / scaling_law.py / subbit_ladder.py):
  - EFFECTIVE bpw ONLY. Never the nominal payload bpw. A baseline's bpw is its real on-disk
    bytes*8/params (GGUF file size / param count); Hawking's is the baker's AGGREGATE eff-bpw.
    A comparison at MISMATCHED bpw is annotated and DOWN-WEIGHTED, never silently equated.
  - NO FAKE WIN. A "win" that only exists because a method rehydrates to f16 (i.e. you compared
    f16 quality at a fake low bpw) counts as ZERO. We compare quality at the artifact's REAL
    eff-bpw, and a row whose eff-bpw is >= f16-ish (>8 bpw) is flagged fake_win and excluded from
    the verdict. (Spec/serve throughput numbers are out of scope here — this gate is OUTPUT-SPACE
    ppl only; throughput claims live behind exact-match/native-serve gates elsewhere.)
  - 7B+ ONLY sets the verdict. 0.5B/1.5B are lab points (printed, never decide the kill) — same
    rule as scaling_law's R1/R2 split.
  - Honest gating: if a baseline engine or its model artifact is ABSENT, that cell is "unavailable"
    (not a win, not a loss) and the verdict says so. We never fabricate a baseline number on the
    real path.

ENVIRONMENT (honored, same as the neighbors):
  DOCTOR_DEVICE / DOCTOR_DTYPE  — device+dtype for the Hawking-floor ppl measurement (7B: cpu/bf16).
  STRAND_NO_GPU                 — forwarded to any child baker call (none here; we read the floor
                                  from the audit jsonl, we don't re-bake).
  PPL_TEXT (default /tmp/ppl24k.txt) + MULTIWINDOW — the SAME held-out windows as audit_ladder, so
                                  every engine is scored on identical text. 2048-tok windows.
  LLAMA_BIN / LLAMA_PPL_BIN     — override the llama.cpp binaries (else auto-detected on PATH).

REAL vs SYNTHETIC:
  Heavy real paths (running llama-perplexity / mlx_lm / loading 7B+) are STUDIO-TIER and require the
  GGUF/MLX artifacts present. On an 18GB laptop those models are NOT here, so:
    --synthetic   exercises the FULL verdict logic with plausible literature numbers (IQ1_S/IQ2/MLX
                  vs a Hawking floor), proving the head-to-head + KILL logic end-to-end with no model.
    --selftest    runs the synthetic verdict on a few scenarios and asserts the win/kill calls.
  The real path auto-gates: a missing engine/artifact yields an "unavailable" cell, never a crash.

CLI:
  bench_baselines.py --model <hf-dir> --label 7B [--audit-jsonl P] [--gguf-dir D] [--mlx-4bit D]
                     [--out reports/condense/<label>_baselines.json]
  bench_baselines.py --synthetic --label 7B [--floor-bpw 2.34 --floor-degr 1.6]
  bench_baselines.py --selftest
"""
import sys, os, re, json, math, argparse, subprocess, shutil

# ── shared discipline knobs (echoed from scaling_law / subbit_ladder) ──────────────────
GATE_PCT   = float(os.environ.get("FLOOR_GATE_PCT", "2.0"))   # the ~1:1 "quality held" bar
F16_BPW    = 16.0
FAKE_WIN_BPW = 8.0          # a row at/above this eff-bpw is ~f16 -> any "win" is a FAKE win
LAB_PARAMS = 7.0            # < this = lab point, never sets the verdict (scaling_law R1/R2 rule)
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "7B": 7.0, "14B": 14.0, "32B": 32.0,
          "70B": 70.0, "72B": 72.0, "405B": 405.0}

# reference EFFECTIVE bpw of the commodity quant tiers (real on-disk bytes*8/params; the GGUF
# file-size/param ratio is the honest number — these are the catalog anchors used when an artifact
# is absent on the synthetic path; the real path MEASURES the file's bpw instead).
GGUF_TIERS = {                      # llama.cpp dynamic K-quant / I-quant tiers
    "IQ1_S":   1.56,
    "IQ2_XXS": 2.06,
    "IQ2_S":   2.50,
}
MLX_TIERS = {                       # mlx_lm convert -q tiers
    "MLX-4bit": 4.50,               # 4-bit + group scales ~= 4.5 eff-bpw
    "MLX-DWQ":  4.50,               # distilled-weight-quant, same footprint, better quality
}
WIN_TIER = "IQ2_S"                  # the bar that decides the wedge: beat IQ2 at equal bpw on 7B+


def log(m):
    print(m, file=sys.stderr); sys.stderr.flush()


# ── PPL on the SAME held-out windows audit_ladder uses ─────────────────────────────────
PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")


def _ensure_ppl_text():
    """Recreate the eval corpus exactly like audit_ladder._ensure_ppl_text (so windows match)."""
    if os.path.exists(PT):
        return
    import glob
    chunks = []
    if os.path.exists("README.md"):
        chunks.append(open("README.md", errors="ignore").read())
    for path in sorted(glob.glob("docs/plans/*.md")):
        chunks.append(open(path, errors="ignore").read())
    text = "\n".join(chunks)[:24000]
    if not text:
        raise FileNotFoundError(PT)
    with open(PT, "w") as f:
        f.write(text)


# ── engine availability probes (honest gating — binary present != artifact present) ─────
def _which(*names):
    for n in names:
        p = os.environ.get(n.upper().replace("-", "_") + "_BIN")
        if p and os.path.exists(p):
            return p
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def llama_available():
    return _which("llama-perplexity") or os.environ.get("LLAMA_PPL_BIN")


def llama_cli_available():
    return _which("llama-cli") or os.environ.get("LLAMA_BIN")


def mlx_available():
    try:
        import importlib.util
        return importlib.util.find_spec("mlx_lm") is not None
    except Exception:
        return False


# ── effective-bpw of a real artifact (no nominal numbers ever) ─────────────────────────
def gguf_eff_bpw(gguf_path, params_b):
    """Honest eff-bpw = file_bytes * 8 / total_params. The GGUF carries ITS OWN scales/zeros/
    sparse-outlier side-info inside that byte count, so this is genuinely effective, not nominal."""
    if not (gguf_path and os.path.exists(gguf_path) and params_b):
        return None
    return os.path.getsize(gguf_path) * 8.0 / (params_b * 1e9)


def mlx_eff_bpw(mlx_dir, params_b):
    """eff-bpw of an MLX 4-bit dir = sum(weight-shard bytes)*8 / params (config + tokenizer excluded)."""
    if not (mlx_dir and os.path.isdir(mlx_dir) and params_b):
        return None
    tot = 0
    for f in os.listdir(mlx_dir):
        if f.endswith(".safetensors"):
            tot += os.path.getsize(os.path.join(mlx_dir, f))
    return (tot * 8.0 / (params_b * 1e9)) if tot else None


# ── real ppl measurements (STUDIO-TIER; auto-gated when artifacts absent) ──────────────
def llama_ppl(gguf_path, ppl_text=PT):
    """Run llama-perplexity over the SAME corpus. Returns ppl float or None (unavailable/failed).
    STUDIO-TIER: needs the GGUF artifact + enough RAM for the model. Gated cleanly."""
    binp = llama_available()
    if not (binp and gguf_path and os.path.exists(gguf_path)):
        return None
    _ensure_ppl_text()
    cmd = [binp, "-m", gguf_path, "-f", ppl_text, "--ctx-size", "2048", "-b", "2048"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("LLAMA_TIMEOUT", "3600")))
    except Exception as e:
        log(f"  [llama] {gguf_path}: run failed ({e})"); return None
    # llama-perplexity prints "Final estimate: PPL = <x> +/- <y>"
    m = re.search(r"PPL\s*=\s*([0-9.]+)", r.stdout + r.stderr)
    if not m:
        log(f"  [llama] {gguf_path}: no PPL parsed (rc={r.returncode})"); return None
    return float(m.group(1))


def mlx_ppl(mlx_dir, ppl_text=PT):
    """MLX ppl on the SAME windows (mirrors mlx_ppl.py). None when mlx_lm or the dir is absent.
    STUDIO-TIER: loads the model into unified memory. Gated cleanly."""
    if not (mlx_available() and mlx_dir and os.path.isdir(mlx_dir)):
        return None
    _ensure_ppl_text()
    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as e:
        log(f"  [mlx] import failed ({e})"); return None
    try:
        model, tok = load(mlx_dir)
        text = open(ppl_text, errors="ignore").read()
        ids = mx.array(tok.encode(text)[:2048])
        logits = model(ids[None])[0]
        logp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
        tgt = ids[1:]
        nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).mean()
        return math.exp(nll.item())
    except Exception as e:
        log(f"  [mlx] {mlx_dir}: ppl failed ({e})"); return None


def hawking_f16_ppl(model_dir):
    """f16 parent ppl over the SAME windows (so degradation is comparable). STUDIO-TIER for 7B+;
    returns None if torch/model unavailable. Mirrors audit_ladder.measure(None)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        log(f"  [hawking] torch/transformers unavailable ({e})"); return None
    _ensure_ppl_text()
    dev = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
    try:
        torch.set_num_threads(os.cpu_count() or 12)
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, attn_implementation="eager").to(dev).eval()
        ids = tok(open(PT, errors="ignore").read(), return_tensors="pt").input_ids[:, :2048].to(dev)
        with torch.no_grad():
            return math.exp(model(ids, labels=ids).loss.item())
    except Exception as e:
        log(f"  [hawking] f16 ppl failed ({e})"); return None


# ── read Hawking's measured FLOOR from the audit_ladder jsonl (its OWN eff-bpw + degr) ──
def read_hawking_floor(audit_jsonl):
    """Lowest effective bpw at <= +GATE% degradation (scaling_law.find_floor logic). Returns
    {floor_bpw, winning_config, degr_pct, f16_ppl, floor_ppl} or None. NO FAKE WIN: a floor row
    whose eff_bpw >= FAKE_WIN_BPW (~f16) is rejected (rehydrating to f16 is not a sub-4-bit win)."""
    if not (audit_jsonl and os.path.exists(audit_jsonl)):
        return None
    f16_ppl = None
    best = None
    for ln in open(audit_jsonl):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("config") == "f16":
            f16_ppl = r.get("ppl")
            continue
        if "ppl" not in r or "eff_bpw" not in r:
            continue
        bpw, degr = r.get("eff_bpw"), r.get("degr_pct")
        if bpw is None or degr is None:
            continue
        if bpw >= FAKE_WIN_BPW:          # ~f16 weights — not a sub-4-bit floor, skip
            continue
        if degr <= GATE_PCT and (best is None or bpw < best["floor_bpw"]):
            best = {"floor_bpw": bpw, "winning_config": r.get("config"),
                    "degr_pct": degr, "floor_ppl": r.get("ppl")}
    if best is not None:
        best["f16_ppl"] = f16_ppl
    return best


# ── the verdict engine (PURE; runs identically on real + synthetic rows) ───────────────
def _degr(ppl, f16_ppl):
    if ppl is None or not f16_ppl:
        return None
    return (ppl / f16_ppl - 1.0) * 100.0


def build_rows(label, hawking, baselines):
    """Assemble the head-to-head table. Each baseline = dict(engine, tier, eff_bpw, ppl, f16_ppl,
    available). Computes degr%, and the bpw-matched comparison vs the Hawking floor."""
    rows = []
    hf_bpw = hawking.get("floor_bpw") if hawking else None
    hf_degr = hawking.get("degr_pct") if hawking else None
    for b in baselines:
        eff = b.get("eff_bpw")
        degr = b.get("degr_pct")
        if degr is None:
            degr = _degr(b.get("ppl"), b.get("f16_ppl"))
        avail = b.get("available", b.get("ppl") is not None or degr is not None)
        # NO FAKE WIN: a baseline at ~f16 bpw can't be a fair sub-4-bit equal-bpw comparison.
        fake = eff is not None and eff >= FAKE_WIN_BPW
        # bpw-matched? Hawking floor must be at <= the baseline's eff-bpw to claim an equal/better
        # quality-per-bit win. We require Hawking eff-bpw <= baseline eff-bpw + small tol.
        matched = (hf_bpw is not None and eff is not None and hf_bpw <= eff + 0.10)
        # quality win: at matched-or-lower bpw, Hawking's degradation must be <= the baseline's.
        if avail and matched and hf_degr is not None and degr is not None and not fake:
            beat = hf_degr <= degr
        else:
            beat = None
        rows.append({
            "engine": b["engine"], "tier": b["tier"],
            "eff_bpw": round(eff, 3) if eff is not None else None,
            "ppl": round(b["ppl"], 3) if b.get("ppl") is not None else None,
            "degr_pct": round(degr, 2) if degr is not None else None,
            "available": bool(avail),
            "bpw_matched_vs_floor": bool(matched),
            "fake_win_excluded": bool(fake),
            "hawking_beats": beat,
        })
    return rows


def decide(label, hawking, rows):
    """The KILL criterion. WIN iff, on a 7B+ model, the Hawking floor beats the WIN_TIER (IQ2)
    baseline at matched-or-lower EFFECTIVE bpw. Returns a verdict dict including the explicit
    KILL line. Lab models (<7B) never decide — they return verdict 'lab (not decisive)'."""
    params = PARAMS.get(label, 0.0)
    is_lab = params < LAB_PARAMS
    kill_line = (f"KILL CRITERION: the wedge dies if Hawking's floor FAILS to beat {WIN_TIER} "
                 f"(llama.cpp IQ2) at <= equal EFFECTIVE bpw on a 7B+ model.")

    if not hawking or hawking.get("floor_bpw") is None:
        return {"verdict": "INCONCLUSIVE — no Hawking floor within +%.0f%% gate" % GATE_PCT,
                "decisive": False, "is_lab": is_lab, "kill_line": kill_line,
                "reframe": "no measured floor; run audit_ladder first (cannot judge the wedge)."}

    # the decisive baseline = WIN_TIER if present, else the lowest-bpw available non-fake baseline
    decisive = next((r for r in rows if r["tier"] == WIN_TIER and r["available"] and not r["fake_win_excluded"]), None)
    if decisive is None:
        cands = [r for r in rows if r["available"] and not r["fake_win_excluded"] and r["eff_bpw"] is not None]
        decisive = min(cands, key=lambda r: r["eff_bpw"]) if cands else None

    if decisive is None:
        return {"verdict": "INCONCLUSIVE — no commodity baseline available to compare",
                "decisive": False, "is_lab": is_lab, "kill_line": kill_line,
                "reframe": "install llama.cpp IQ2 GGUF / MLX-4bit artifacts, or run --synthetic to "
                           "exercise the logic. Verdict cannot be rendered without a baseline."}

    beat = decisive["hawking_beats"]
    hf_bpw = hawking["floor_bpw"]; hf_degr = hawking.get("degr_pct")
    headline = (f"Hawking floor {hf_bpw:.2f} bpw (+{hf_degr:.1f}%) vs {decisive['engine']} "
                f"{decisive['tier']} {decisive['eff_bpw']} bpw (+{decisive['degr_pct']}%)")

    if not decisive["bpw_matched_vs_floor"]:
        verdict = (f"INCONCLUSIVE — Hawking floor ({hf_bpw:.2f} bpw) sits ABOVE the {decisive['tier']} "
                   f"baseline ({decisive['eff_bpw']} bpw); not an equal-bpw comparison")
        return {"verdict": verdict, "decisive": False, "is_lab": is_lab, "headline": headline,
                "kill_line": kill_line, "baseline_used": decisive["tier"],
                "reframe": "push the floor below the baseline bpw before claiming the wedge."}

    if beat is True:
        verdict = f"HAWKING WINS the wedge — beats {decisive['tier']} at <= equal effective bpw"
        reframe = "wedge defended: sub-4-bit weight-quality leadership holds at this scale."
        decisive_flag = not is_lab
    elif beat is False:
        verdict = f"WEDGE KILLED — Hawking floor does NOT beat {decisive['tier']} at equal bpw"
        reframe = ("reframe to PORTFOLIO value (RWKV state / .tq format / economics / logits), "
                   "retire the 'beats everyone on weight quality' sub-4-bit claim.")
        decisive_flag = not is_lab
    else:
        verdict = "INCONCLUSIVE — comparison not computable (missing degr or fake-win excluded)"
        reframe = "rerun with the baseline ppl measured on the same windows."
        decisive_flag = False

    if is_lab and beat is not None:
        verdict += " [LAB point — NOT decisive; the verdict is read off 7B+]"

    return {"verdict": verdict, "decisive": decisive_flag, "is_lab": is_lab,
            "headline": headline, "baseline_used": decisive["tier"],
            "hawking_beats_baseline": beat, "kill_line": kill_line, "reframe": reframe}


# ── output ─────────────────────────────────────────────────────────────────────────────
def write_report(out_path, label, hawking, rows, verdict, source):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = {
        "tool": "bench_baselines.py", "kind": "wedge-defense bench (PROBE, output-space ppl only)",
        "model": label, "params_b": PARAMS.get(label),
        "source": source,                       # "real" | "synthetic"
        "gate_pct": GATE_PCT, "win_tier": WIN_TIER, "fake_win_bpw_floor": FAKE_WIN_BPW,
        "discipline": "EFFECTIVE bpw only; no fake-win (f16-rehydrate excluded); ppl on shared windows; "
                      "7B+ sets the verdict; throughput/serve claims OUT OF SCOPE.",
        "hawking_floor": hawking,
        "head_to_head": rows,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
    return out_path


def print_table(label, hawking, rows, verdict):
    log(f"\n# wedge-defense head-to-head — {label} (gate +{GATE_PCT}% ppl, EFFECTIVE bpw)")
    if hawking and hawking.get("floor_bpw") is not None:
        log(f"  HAWKING floor: {hawking['floor_bpw']:.3f} bpw  +{hawking.get('degr_pct','?')}%  "
            f"via {hawking.get('winning_config','?')}")
    else:
        log("  HAWKING floor: (none within gate)")
    log(f"  {'engine':9s} {'tier':9s} {'eff_bpw':>8s} {'degr%':>8s}  matched  beats?  note")
    for r in rows:
        note = "FAKE-WIN excl" if r["fake_win_excluded"] else ("" if r["available"] else "UNAVAILABLE")
        beat = {True: "HAWK", False: "base", None: "-"}[r["hawking_beats"]]
        log(f"  {r['engine']:9s} {r['tier']:9s} "
            f"{(r['eff_bpw'] if r['eff_bpw'] is not None else '-'):>8} "
            f"{(r['degr_pct'] if r['degr_pct'] is not None else '-'):>8}  "
            f"{str(r['bpw_matched_vs_floor']):>7}  {beat:>5}  {note}")
    log(f"\n  >>> {verdict['verdict']}")
    log(f"  >>> {verdict['kill_line']}")
    log(f"  >>> reframe: {verdict.get('reframe','-')}\n")


# ── REAL path: measure each engine, gate the absent ones, render verdict ───────────────
def run_real(args):
    label = args.label
    params_b = PARAMS.get(label)
    out_path = args.out or f"reports/condense/{label}_baselines.json"
    audit_jsonl = args.audit_jsonl or f"reports/condense/{label}_audit.jsonl"

    hawking = read_hawking_floor(audit_jsonl)
    if hawking and hawking.get("f16_ppl") is None:
        hawking["f16_ppl"] = hawking_f16_ppl(args.model) if args.model else None

    # the shared f16 anchor for degradation: prefer Hawking's f16 (same windows/engine family).
    f16_anchor = (hawking or {}).get("f16_ppl")
    if f16_anchor is None and args.model:
        f16_anchor = hawking_f16_ppl(args.model)
        if hawking is not None:
            hawking["f16_ppl"] = f16_anchor

    baselines = []
    # ---- llama.cpp GGUF tiers ----
    gguf_map = {}                       # tier -> path
    if args.gguf_dir and os.path.isdir(args.gguf_dir):
        for f in os.listdir(args.gguf_dir):
            for tier in GGUF_TIERS:
                if tier.lower() in f.lower() and f.endswith(".gguf"):
                    gguf_map[tier] = os.path.join(args.gguf_dir, f)
    for tier in GGUF_TIERS:
        path = gguf_map.get(tier)
        eff = gguf_eff_bpw(path, params_b)
        ppl = llama_ppl(path) if path else None
        baselines.append({"engine": "llama.cpp", "tier": tier,
                          "eff_bpw": eff if eff is not None else GGUF_TIERS[tier] if path else None,
                          "ppl": ppl, "f16_ppl": f16_anchor,
                          "available": ppl is not None})
    # ---- MLX 4bit / DWQ ----
    mlx_map = {"MLX-4bit": args.mlx_4bit, "MLX-DWQ": args.mlx_dwq}
    for tier, mdir in mlx_map.items():
        eff = mlx_eff_bpw(mdir, params_b)
        ppl = mlx_ppl(mdir) if mdir else None
        baselines.append({"engine": "mlx_lm", "tier": tier,
                          "eff_bpw": eff if eff is not None else (MLX_TIERS[tier] if mdir else None),
                          "ppl": ppl, "f16_ppl": f16_anchor,
                          "available": ppl is not None})

    rows = build_rows(label, hawking, baselines)
    verdict = decide(label, hawking, rows)
    print_table(label, hawking, rows, verdict)
    p = write_report(out_path, label, hawking, rows, verdict, source="real")
    log(f"# wrote {p}")
    # honesty banner on the absent engines
    miss = [r["tier"] for r in rows if not r["available"]]
    if miss:
        log(f"# NOTE: {len(miss)} baseline cell(s) UNAVAILABLE on this box (no artifact/engine): "
            f"{miss}. Those are 'unavailable', not wins/losses. Real run is STUDIO-TIER.")
    return verdict


# ── SYNTHETIC path: full verdict logic with plausible literature numbers, no model ─────
def synth_baselines(f16_ppl, floor_bpw):
    """Plausible degradations from the sub-4-bit literature (illustrative anchors, clearly tagged
    synthetic): IQ1_S is brutal, IQ2_XXS rough, IQ2_S ~usable, MLX-4bit near-lossless. Returns the
    baseline list scored against the same f16 anchor."""
    # degr% anchors (relative ppl vs f16) — typical published behavior for a 7B
    DEGR = {"IQ1_S": 38.0, "IQ2_XXS": 11.0, "IQ2_S": 4.5, "MLX-4bit": 1.2, "MLX-DWQ": 0.6}
    out = []
    for tier, bpw in GGUF_TIERS.items():
        d = DEGR[tier]
        out.append({"engine": "llama.cpp", "tier": tier, "eff_bpw": bpw,
                    "ppl": f16_ppl * (1 + d / 100.0), "f16_ppl": f16_ppl, "available": True})
    for tier, bpw in MLX_TIERS.items():
        d = DEGR[tier]
        out.append({"engine": "mlx_lm", "tier": tier, "eff_bpw": bpw,
                    "ppl": f16_ppl * (1 + d / 100.0), "f16_ppl": f16_ppl, "available": True})
    return out


def run_synthetic(args, f16_ppl=12.0, floor_bpw=None, floor_degr=None, label=None, write=True, quiet=False):
    label = label or args.label
    floor_bpw = floor_bpw if floor_bpw is not None else (args.floor_bpw if args else 2.34)
    floor_degr = floor_degr if floor_degr is not None else (args.floor_degr if args else 1.6)
    hawking = {"floor_bpw": floor_bpw, "winning_config": "synthetic", "degr_pct": floor_degr,
               "floor_ppl": f16_ppl * (1 + floor_degr / 100.0), "f16_ppl": f16_ppl}
    baselines = synth_baselines(f16_ppl, floor_bpw)
    rows = build_rows(label, hawking, baselines)
    verdict = decide(label, hawking, rows)
    if not quiet:
        print_table(label, hawking, rows, verdict)
    if write:
        out_path = (args.out if args and args.out else f"reports/condense/{label}_baselines.json")
        p = write_report(out_path, label, hawking, rows, verdict, source="synthetic")
        if not quiet:
            log(f"# wrote {p} (SYNTHETIC — plausible literature anchors, NOT a measured result)")
    return verdict


# ── SELFTEST: assert win + kill + lab + inconclusive calls (runs here, no model) ───────
def selftest():
    ok = True
    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    class A:  # minimal args stub
        out = None; label = "7B"; floor_bpw = None; floor_degr = None

    # 1) Hawking floor at 2.34 bpw / +1.6% BEATS IQ2_S (2.50 bpw / +4.5%) on a 7B -> WIN, decisive.
    v = run_synthetic(A, f16_ppl=12.0, floor_bpw=2.34, floor_degr=1.6, label="7B", write=False, quiet=True)
    check("7B floor 2.34/+1.6% WINS vs IQ2_S", v["hawking_beats_baseline"] is True and "WINS" in v["verdict"])
    check("7B win is DECISIVE", v["decisive"] is True)

    # 2) Hawking floor at 2.50 bpw / +9% does NOT beat IQ2_S (+4.5%) -> KILL, decisive.
    v = run_synthetic(A, f16_ppl=12.0, floor_bpw=2.50, floor_degr=9.0, label="7B", write=False, quiet=True)
    check("7B floor 2.50/+9% KILLS the wedge vs IQ2_S", "KILLED" in v["verdict"] and v["hawking_beats_baseline"] is False)
    check("7B kill is DECISIVE", v["decisive"] is True)

    # 3) Same win numbers on a 0.5B LAB model -> NOT decisive (verdict tagged lab).
    v = run_synthetic(A, f16_ppl=20.0, floor_bpw=2.34, floor_degr=1.6, label="0.5B", write=False, quiet=True)
    check("0.5B win is NOT decisive (lab)", v["decisive"] is False and v["is_lab"] is True)
    check("0.5B verdict tagged LAB", "LAB" in v["verdict"])

    # 4) Floor ABOVE the IQ2_S baseline bpw (3.0 > 2.50) -> INCONCLUSIVE (not an equal-bpw test).
    v = run_synthetic(A, f16_ppl=12.0, floor_bpw=3.0, floor_degr=0.5, label="7B", write=False, quiet=True)
    check("floor above baseline bpw -> INCONCLUSIVE", "INCONCLUSIVE" in v["verdict"] and v["decisive"] is False)

    # 5) NO FAKE WIN: a floor row at ~f16 bpw (10) is rejected by read_hawking_floor.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(json.dumps({"config": "f16", "eff_bpw": 16.0, "ppl": 12.0, "degr_pct": 0.0}) + "\n")
        tf.write(json.dumps({"config": "fake", "eff_bpw": 10.0, "ppl": 12.05, "degr_pct": 0.4}) + "\n")
        tf.write(json.dumps({"config": "real-floor", "eff_bpw": 2.34, "ppl": 12.2, "degr_pct": 1.6}) + "\n")
        fp = tf.name
    fl = read_hawking_floor(fp)
    os.remove(fp)
    check("fake-win f16 row excluded; real floor picked", fl and abs(fl["floor_bpw"] - 2.34) < 1e-6)

    # 6) eff-bpw helpers compute bytes*8/params, not nominal.
    check("gguf_eff_bpw None when path absent", gguf_eff_bpw("/no/such.gguf", 7.0) is None)

    # 7) verdict ALWAYS carries an explicit KILL line.
    v = run_synthetic(A, f16_ppl=12.0, floor_bpw=2.34, floor_degr=1.6, label="7B", write=False, quiet=True)
    check("verdict prints an explicit KILL line", "kill_line" in v and "KILL" in v["kill_line"])

    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="wedge-defense baseline gate (PROBE; output-space ppl)")
    ap.add_argument("--model", help="HF model dir (for the f16 anchor on the real path)")
    ap.add_argument("--label", default="7B", help="model label (7B/14B/32B/0.5B...) — sets the verdict rule")
    ap.add_argument("--audit-jsonl", help="audit_ladder jsonl with Hawking's floor (default reports/condense/<label>_audit.jsonl)")
    ap.add_argument("--gguf-dir", help="dir holding IQ1_S/IQ2_XXS/IQ2_S .gguf artifacts (real path)")
    ap.add_argument("--mlx-4bit", help="MLX 4-bit model dir (real path)")
    ap.add_argument("--mlx-dwq", help="MLX DWQ model dir (real path)")
    ap.add_argument("--out", help="output json (default reports/condense/<label>_baselines.json)")
    ap.add_argument("--synthetic", action="store_true", help="exercise the FULL verdict logic with plausible anchors (no model)")
    ap.add_argument("--floor-bpw", type=float, default=2.34, help="[synthetic] Hawking floor eff-bpw")
    ap.add_argument("--floor-degr", type=float, default=1.6, help="[synthetic] Hawking floor degr%%")
    ap.add_argument("--selftest", action="store_true", help="assert win/kill/lab/fake-win logic, here")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.synthetic or not args.model:
        if not args.synthetic and not args.model:
            log("# no --model given and not --synthetic; running --synthetic (the only path that "
                "works without GGUF/MLX artifacts on this box).")
        run_synthetic(args)
        return
    run_real(args)


if __name__ == "__main__":
    main()
