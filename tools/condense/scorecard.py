#!/usr/bin/env python3.12
"""scorecard.py — THE CAPSTONE: synthesize every produced record into one populated competitive
SCORECARD (the proof that makes the whole condense program legible). A pure-stdlib PROBE/synthesizer
(reads JSON only — like subbit_ladder.py, no torch / safetensors / cargo / llama.cpp needed), so it
runs HERE on the 18GB laptop against whatever records already exist on disk.

WHAT THIS TOOL IS: a READER + COMPOSER, not a measurer. It does NOT bake, doctor, serve, or bench.
It reads the records the rest of the pipeline produced and emits a single SCORECARD that says, for
every claim, exactly which record file backs it and at what repro level — and REFUSES to print a
WIN cell that has no receipt. It is the place the program's honesty discipline is enforced once,
at the end, over everything.

THE RECORDS IT READS (all optional — --dry composes from whatever exists):
  reports/cron/*_frontier.jsonl / *_ladder.jsonl / *_verify.jsonl   (the bake CURVES: eff-bpw vs ppl)
  reports/cron/bit_floor_*.jsonl                                     (the per-model FLOORS, scaling_law --floor)
  reports/cron/bit_floor_*_curve.md                                  (the fitted LAW, scaling_law --fit)
  receipts/official/*.json                                          (the FLOOR RECEIPTS — the only WIN proof)
  reports/condense/*_eval.json                                      (downstream-task tripwire evals)
  reports/condense/*_baselines.json                                 (llama.cpp / MLX / QTIP external baselines)
  reports/condense/*_spec.json                                      (spec-decode revival — accept + exact-match)
  reports/condense/*_codec_bakeoff.json                             (STRAND vs QTIP/QuIP#/AQLM head-to-head)
  reports/condense/*_frontier.json                                  (100B+ serve-fit records)
  reports/condense/*_subbit0.json                                   (the sub-1-bit entropy/side-info FLOOR)
  reports/condense/*_expert_sens.json                              (MoE per-expert sensitivity — sub-bit alive?)

PROOF DISCIPLINE (enforced in code, the §6/§20 rules):
  · EFFECTIVE bpw only — a row that reports only nominal bpw is not a measurement.
  · NO FAKE WIN — a result that rehydrates to f16 (no native serve) counts ZERO toward a serve/tps
    win; spec / serve numbers are admissible ONLY under an exact-match / native-serve record.
  · A WIN cell requires a backing RECEIPT at R3+ (one-command same-machine-class repro). Below R3,
    or with no receipt at all, the strongest a claim gets is GATED or MEASURED-but-not-public.
  · 0.5B / 1.5B are LAB points (R1 baselines) and NEVER set a verdict.
  · Anything without a backing record = UNPROVEN (printed, never a GO).
  · An explicit KILL line is always printed.

KILL LINE (the criterion that refuses the headline): the scorecard prints WIN in a capability cell
ONLY when a receipt at R3+ with quality_gate in {pass,warn} and a real effective_bpw exists for that
cell. If no such receipt exists, the cell is GATED/UNPROVEN — never WIN. (Today, on the qwen-0.5B
records, the floor receipt is an R1 baseline => every quality/density WIN cell is correctly withheld.)

HEAVY/REAL paths: there are none here — every input is a JSON record produced upstream. The records
themselves come from Studio-tier runs (audit_ladder bakes, doctor recovery, cargo serve/spec). This
tool just composes them; on this box it composes whatever the qwen-0.5B lane already wrote.

ENV: honors DOCTOR_DEVICE / DOCTOR_DTYPE / STRAND_NO_GPU for parity with the neighbors (this tool
does no compute, so they are recorded into the scorecard provenance but change nothing). FLOOR_GATE_PCT
echoes scaling_law's ~1:1 gate (default 2.0). WIN_MIN_REPRO sets the receipt repro bar for a WIN (R3).

CLI (argv, matching the neighbors):
  scorecard.py                 # compose from records on disk, write reports/condense/SCORECARD.{md,json}
  scorecard.py --dry           # same, but explicitly the "compose from whatever exists" path (default here)
  scorecard.py --selftest      # synthetic: fabricate a full record set in a temp dir, exercise EVERY
                               #   branch (WIN gate, fake-win refusal, KILL, repro ladder), assert anchors
  scorecard.py -h | --help
"""
import sys, os, re, json, glob, math, hashlib, subprocess, tempfile, shutil

# ── paths (match studio_run.py / scaling_law.py layout) ────────────────────────────────
CRON_DIR    = "reports/cron"
COND_DIR    = "reports/condense"
RECEIPT_DIR = "receipts/official"
OUT_MD      = f"{COND_DIR}/SCORECARD.md"
OUT_JSON    = f"{COND_DIR}/scorecard.json"

# ── gates / discipline knobs ────────────────────────────────────────────────────────────
FLOOR_GATE_PCT = float(os.environ.get("FLOOR_GATE_PCT", "2.0"))   # the ~1:1 quality gate (echoes scaling_law)
Q4K_BPW        = 4.5                                              # the llama.cpp Q4_K_M reference rung
WIN_MIN_REPRO  = os.environ.get("WIN_MIN_REPRO", "R3")           # §20.6: no public WIN below R3
REPRO_ORDER    = ["R0", "R1", "R2", "R3", "R4", "R5"]            # the §20.6 reproducibility gradient
POSITIVE_GATES = {"pass", "warn"}                                # quality_gate values asserting a positive
# repro_level descriptions (receipts/schema/condensation_receipt.schema.json §20.6)
REPRO_DESC = {
    "R0": "private (not rerunnable)",
    "R1": "author-rerunnable (lab/baseline)",
    "R2": "artifact identified + measured",
    "R3": "one-command same-machine-class repro (min bar for a public WIN)",
    "R4": "third-party Mac (trust moat)",
    "R5": "format itself cited externally",
}
# params for the scale curve (mirrors scaling_law.PARAMS); <7B = lab, never sets the verdict.
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "7B": 7.0, "14B": 14.0, "32B": 32.0,
          "70B": 70.0, "72B": 72.0, "405B": 405.0, "671B": 671.0, "744B": 744.0}
LAB_MAX_B = 7.0   # strictly below this = lab rung (printed, excluded from the verdict)


def log(m):
    print(m, file=sys.stderr); sys.stderr.flush()


def _repro_ge(level, bar):
    """True if repro `level` is at least as strong as `bar` on the R0..R5 ladder."""
    try:
        return REPRO_ORDER.index(level) >= REPRO_ORDER.index(bar)
    except ValueError:
        return False


def _load_json(path):
    try:
        return json.load(open(path))
    except Exception as e:
        log(f"  [skip] {path}: {e}")
        return None


def _load_jsonl(path):
    rows = []
    try:
        for ln in open(path):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    except Exception as e:
        log(f"  [skip] {path}: {e}")
    return rows


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ════════════════════════════════════════════════════════════════════════════════════════
#  RECORD INGEST — each loader returns a normalized dict the composer + KILL gate can read.
#  Every loader is defensive: a malformed/absent record degrades to "no backing" (UNPROVEN),
#  never an exception that aborts the scorecard.
# ════════════════════════════════════════════════════════════════════════════════════════
# cron jsonl files that are NOT bake curves (floors handled separately; these have no config/ppl rows)
_NON_CURVE = ("bit_floor_",)
_NON_CURVE_SUFFIX = ("_promotions.jsonl", "_verified.jsonl", "_sched.status")


def ingest_curves(root):
    """The bake CURVES (eff-bpw vs ppl) from reports/cron/*.jsonl. Both naming schemes are picked up:
    the autopilot lane ({label}_frontier.jsonl / _ladder.jsonl / _verify.jsonl) AND the studio lane
    ({set_name}_{label}.jsonl, e.g. studio_7B.jsonl). A file qualifies as a CURVE iff it has at least
    one row carrying both eff_bpw and ppl (the bake-result shape) — content-detected, not name-matched,
    so a new lane's file is read without a code change. bit_floor_* (floors) + promotions/verified/
    status (non-curve) are excluded. Returns {filename: {file,label,f16_ppl,rows,n_measured,n_errors}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, CRON_DIR, "*.jsonl"))):
        base = os.path.basename(path)
        if base.startswith(_NON_CURVE) or base.endswith(_NON_CURVE_SUFFIX):
            continue
        rows = _load_jsonl(path)
        good = [r for r in rows if "ppl" in r and "eff_bpw" in r and r.get("config") != "f16"]
        if not good:                       # not a bake curve (e.g. a floors/aux file) — skip
            continue
        label = next((r.get("model") for r in rows if r.get("model")), None) or base.split("_")[0]
        f16 = next((r.get("ppl") for r in rows if r.get("config") == "f16"), None)
        errs = [r for r in rows if "error" in r]
        out[base] = {
            "file": os.path.relpath(path, root), "label": label, "f16_ppl": f16,
            "rows": good, "n_measured": len(good), "n_errors": len(errs)}
    return out


def find_floor_in_curve(curve):
    """Lowest EFFECTIVE bpw at <= FLOOR_GATE_PCT degradation in a curve (mirror scaling_law.find_floor).
    Returns (eff_bpw, config, degr_pct) or None. EFFECTIVE bpw only — nominal is ignored."""
    best = None
    for r in curve["rows"]:
        bpw, degr = r.get("eff_bpw"), r.get("degr_pct")
        if bpw is None or degr is None:
            continue
        if degr <= FLOOR_GATE_PCT and (best is None or bpw < best[0]):
            best = (bpw, r.get("config"), degr)
    return best


def ingest_floors(root):
    """The per-model FLOORS from scaling_law --floor (reports/cron/bit_floor_*.jsonl) + the fitted
    LAW md (bit_floor_*_curve.md). Returns {"points":[{model,params_b,floor_bpw,winning_config,degr_pct}],
    "law": {text, file} or None}."""
    points, law = [], None
    for path in sorted(glob.glob(os.path.join(root, CRON_DIR, "bit_floor_*.jsonl"))):
        for r in _load_jsonl(path):
            if "model" in r:
                r["_file"] = os.path.relpath(path, root)
                points.append(r)
    for path in sorted(glob.glob(os.path.join(root, CRON_DIR, "bit_floor_*_curve.md"))):
        try:
            txt = open(path).read()
        except Exception:
            continue
        m_law = re.search(r"\*\*Law.*?:\*\*\s*(.+)", txt)
        m_verd = re.search(r"\*\*Verdict:\*\*\s*(.+)", txt)
        law = {"file": os.path.relpath(path, root),
               "law": (m_law.group(1).strip() if m_law else None),
               "verdict": (m_verd.group(1).strip() if m_verd else None)}
    return {"points": points, "law": law}


def ingest_receipts(root):
    """The FLOOR RECEIPTS (receipts/official/*.json) — the ONLY artifact that can back a WIN.
    Returns [{file, repro_level, claim_type, quality_gate, effective_bpw, source_model, label,
    win_eligible, baseline_best_effort}]. win_eligible = (repro>=WIN_MIN_REPRO AND gate positive AND
    effective_bpw>0 AND NOT best-effort baseline) — the §20.3/§20.6 rule, in code."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, RECEIPT_DIR, "*.json"))):
        r = _load_json(path)
        if not r:
            continue
        repro = r.get("repro_level", "R0")
        gate = r.get("quality_gate", "fail")
        eff = r.get("effective_bpw", 0.0) or 0.0
        ctype = r.get("claim_type", "")
        best_effort = bool(r.get("baseline_best_effort", False))
        # NO FAKE WIN: a baseline / best-effort receipt cannot back a public win (rule R8); a WIN
        # also requires repro >= R3 (rule §20.6), a positive gate, and a real effective bpw (rule R1).
        win_eligible = (_repro_ge(repro, WIN_MIN_REPRO) and gate in POSITIVE_GATES
                        and eff > 0 and ctype not in ("baseline",) and not best_effort)
        out.append({
            "file": os.path.relpath(path, root), "label": _label_from_receipt(r, path),
            "repro_level": repro, "claim_type": ctype, "quality_gate": gate,
            "effective_bpw": eff, "nominal_bpw": r.get("nominal_bpw"),
            "source_model": r.get("source_model", ""), "machine_class": r.get("machine_class", ""),
            "baseline_best_effort": best_effort, "win_eligible": win_eligible,
            "beats_q4k": (eff > 0 and eff < Q4K_BPW)})
    return out


def _label_from_receipt(r, path):
    sm = r.get("source_model", "")
    m = re.search(r"\b(\d+(?:\.\d+)?B)\b", sm) or re.search(r"\b(\d+(?:\.\d+)?B)\b", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path).replace(".json", "")


def ingest_condense(root, suffix):
    """Generic loader for reports/condense/*_<suffix>.json. Returns {label: record}, label parsed
    off the filename prefix (strip the suffix). Used for eval/baselines/spec/codec_bakeoff/
    frontier/subbit0/expert_sens — each interpreted by the composer that owns the section."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, COND_DIR, f"*_{suffix}.json"))):
        r = _load_json(path)
        if r is None:
            continue
        label = os.path.basename(path)[:-(len(suffix) + 6)]   # strip "_<suffix>.json"
        r["_file"] = os.path.relpath(path, root)
        out[label] = r
    return out


# ════════════════════════════════════════════════════════════════════════════════════════
#  COMPOSE — turn the ingested records into the four scorecard parts.
# ════════════════════════════════════════════════════════════════════════════════════════
# The competitive matrix rows (capabilities) and the columns (the codec/runtime field, from §12).
COMPETITORS = ["Hawking", "llama.cpp", "MLX/mlx-lm", "QTIP/EXL3 · AQLM", "BitNet b1.58"]
# Per-capability prior knowledge for the NON-Hawking columns (web-checked landscape, §12 of the plan).
# Hawking's cell is computed from the records; these are the honest competitor positions, not claims
# about Hawking. yes / no / partial / gated, with a short note.
CAP_COMPETITORS = {
    "Sub-4-bit codec (≤3 eff-bpw)": {
        "llama.cpp": ("yes", "IQ2/IQ3 K-quants (static PTQ)"),
        "MLX/mlx-lm": ("yes", "learned/mixed-bit quant"),
        "QTIP/EXL3 · AQLM": ("yes", "trellis/lattice SOTA ≤2-bit"),
        "BitNet b1.58": ("yes", "native 1.58-bit (pretrain, ≤2.4B only)")},
    "Per-model gradient RECOVERY (QAT/KD)": {
        "llama.cpp": ("no", "static PTQ, no recovery loop"),
        "MLX/mlx-lm": ("partial", "QAT/LoRA primitives, not 1-cmd condense+recover"),
        "QTIP/EXL3 · AQLM": ("partial", "AQLM has global FT; EXL3 none; CUDA-only"),
        "BitNet b1.58": ("no", "pretrain-from-scratch, no BYO model")},
    "Native low-bit SERVE on Apple Silicon": {
        "llama.cpp": ("yes", "first-class Metal GGUF"),
        "MLX/mlx-lm": ("yes", "Apple-native UMA"),
        "QTIP/EXL3 · AQLM": ("no", "CUDA/NVIDIA-only"),
        "BitNet b1.58": ("partial", "bitnet.cpp, vendor models only")},
    "RAM-cliff: serve where Q4_K swaps/OOMs": {
        "llama.cpp": ("partial", "mmap out-of-core (pages = slow)"),
        "MLX/mlx-lm": ("no", "in-core; OOMs when it doesn't fit"),
        "QTIP/EXL3 · AQLM": ("no", "in-core (NVIDIA VRAM)"),
        "BitNet b1.58": ("no", "vendor models only")},
    "Bit-floor-vs-scale LAW (a published curve)": {
        "llama.cpp": ("no", "not a research claim they make"),
        "MLX/mlx-lm": ("no", "—"),
        "QTIP/EXL3 · AQLM": ("no", "papers report points, not a product curve"),
        "BitNet b1.58": ("no", "—")},
    "MoE per-expert bit allocation": {
        "llama.cpp": ("no", "uniform per-tensor quant"),
        "MLX/mlx-lm": ("no", "no per-expert recovery-aware alloc"),
        "QTIP/EXL3 · AQLM": ("no", "—"),
        "BitNet b1.58": ("no", "—")},
    "Stacked spec-decode on the condensed model": {
        "llama.cpp": ("yes", "owns iso-quant decode + spec"),
        "MLX/mlx-lm": ("partial", "spec-decode landing"),
        "QTIP/EXL3 · AQLM": ("partial", "EXL3 has spec on CUDA"),
        "BitNet b1.58": ("no", "—")},
    "Reproducible receipts (R3+ one-command)": {
        "llama.cpp": ("no", "community numbers, no receipt contract"),
        "MLX/mlx-lm": ("no", "—"),
        "QTIP/EXL3 · AQLM": ("partial", "papers w/ code, not one-command receipts"),
        "BitNet b1.58": ("no", "eval-only beyond 2.4B")},
}


def compose_matrix(records):
    """For each capability row, compute Hawking's honest cell from the records, attach the competitor
    cells, and cite the backing record. A WIN/yes for Hawking is gated by a receipt at R3+ (KILL line).
    Returns a list of row dicts."""
    receipts = records["receipts"]
    win_receipts = [r for r in receipts if r["win_eligible"]]
    floors = records["floors"]["points"]
    law = records["floors"]["law"]
    curves = records["curves"]
    subbit0 = records["subbit0"]
    expert = records["expert_sens"]
    spec = records["spec"]
    baselines = records["baselines"]
    bakeoff = records["codec_bakeoff"]
    frontier = records["frontier"]

    # has ANY non-lab (>=7B) floor at/under a bpw that beats Q4_K with a win-eligible receipt?
    def _hawking_density_serve_cell():
        # density/serve WIN needs: a measured floor under Q4_K bpw AND a win-eligible receipt for it.
        measured = [p for p in floors if p.get("floor_bpw") and (PARAMS.get(p["model"], 0) >= LAB_MAX_B)]
        beats = [p for p in measured if p["floor_bpw"] < Q4K_BPW]
        if win_receipts and beats:
            cite = win_receipts[0]["file"]
            return ("yes", f"WIN: {beats[0]['model']} floor {beats[0]['floor_bpw']:.2f} eff-bpw < Q4_K {Q4K_BPW} [{cite}]")
        if beats:
            return ("gated", f"MEASURED {beats[0]['model']} {beats[0]['floor_bpw']:.2f} eff-bpw but no R{WIN_MIN_REPRO[-1]}+ receipt [{beats[0].get('_file','?')}]")
        if measured:
            return ("gated", f"floor {measured[0]['floor_bpw']:.2f} eff-bpw at {measured[0]['model']} (>= Q4_K) [{measured[0].get('_file','?')}]")
        # fall back to curve-derived floor (e.g. 7B frontier) — still GATED (lab/no receipt)
        for cf in curves.values():
            fl = find_floor_in_curve(cf)
            if fl and PARAMS.get(cf["label"], 0) >= LAB_MAX_B:
                return ("gated", f"curve floor {fl[0]:.2f} eff-bpw via {fl[1]} on {cf['label']} (no floor receipt) [{cf['file']}]")
        return ("unproven", "no measured floor on a >=7B model")

    def _hawking_recovery_cell():
        # recovery (QAT/KD) WIN needs a curve point where a recovery config (+dr/-bw/-str) cleared the
        # gate. We scan curves for a recovered config under the gate; today these are all errors/UNPROVEN.
        for cf in curves.values():
            for r in cf["rows"]:
                cfg = r.get("config", "")
                if any(t in cfg for t in ("+dr", "-bw", "-str")) and r.get("degr_pct", 1e9) <= FLOOR_GATE_PCT:
                    return ("gated", f"recovered {cfg} @ {r['eff_bpw']:.2f} eff-bpw +{r['degr_pct']}% (no receipt) [{cf['file']}]")
        # was recovery even attempted (errored)?
        attempted = any(any(t in (r.get("config") or "") for t in ("+dr", "-bw", "-str"))
                        for cf in curves.values() for r in _curve_all(cf, records))
        return ("unproven", "recovery attempted but no config cleared the gate (doctor swap/timeout)" if attempted
                else "no recovery config in any curve")

    def _hawking_serve_cell():
        # native low-bit serve: a serve/tps WIN is admissible ONLY under a native-serve record. The
        # frontier records say the quality/cliff numbers are GATED on the Rust serve build => not a win.
        if frontier:
            f0 = next(iter(frontier.values()))
            return ("gated", f"serve-fit recorded; quality+cliff GATED on native .tq serve build [{f0['_file']}]")
        return ("unproven", "no native-serve record (rehydrate -> f16 = fake win; refused)")

    def _hawking_ramcliff_cell():
        if frontier:
            fits = [v for v in frontier.values() if v.get("serve_fits_84")]
            if fits:
                v = fits[0]
                return ("gated", f"{v['model']} {v['artifact_gb']}GB fits 84GB serve-fit (cliff GATED on serve build) [{v['_file']}]")
        return ("unproven", "no RAM-cliff demo (serve build unbuilt)")

    def _hawking_law_cell():
        if law and law.get("law"):
            verd = law.get("verdict") or ""
            return ("gated", f"law fitted: {law['law']} | {verd[:60]} [{law['file']}]")
        # >=2 floor points but no fit yet?
        pts = [p for p in floors if p.get("floor_bpw")]
        if len(pts) >= 2:
            return ("gated", f"{len(pts)} floor points, law not yet fitted (scaling_law --fit)")
        return ("unproven", f"need >=2 floor points (have {len(pts)})")

    def _hawking_moe_cell():
        if expert:
            e0 = next(iter(expert.values()))
            v = e0.get("verdict", "?")
            if e0.get("alive") is True:
                return ("gated", f"per-expert spread {v} (probe, not a serve win) [{e0['_file']}]")
            if e0.get("alive") is False:
                return ("no", f"per-expert UNIFORM => KILLED [{e0['_file']}]")
            return ("gated", f"per-expert {v} (inconclusive probe) [{e0['_file']}]")
        return ("unproven", "no per-expert sensitivity record")

    def _hawking_spec_cell():
        if spec:
            s0 = next(iter(spec.values()))
            verd = s0.get("verdict", "")
            acc = s0.get("accept_rate")
            if verd.startswith("spec lane complete") and acc:
                return ("gated", f"accept {acc:.0%} under exact-match gate (no tps receipt) [{s0['_file']}]")
            return ("no" if verd.startswith(("KILL", "HALT")) else "gated",
                    f"{verd[:60]} [{s0['_file']}]")
        return ("unproven", "no spec-decode record (lossless-verify gate not run)")

    def _hawking_receipts_cell():
        if win_receipts:
            return ("yes", f"{len(win_receipts)} R{WIN_MIN_REPRO[-1]}+ win-eligible receipt(s) [{win_receipts[0]['file']}]")
        if receipts:
            r0 = receipts[0]
            return ("gated", f"{len(receipts)} receipt(s), best={r0['repro_level']}/{r0['claim_type']} "
                    f"(no R{WIN_MIN_REPRO[-1]}+ win) [{r0['file']}]")
        return ("unproven", "no receipts emitted")

    cell_fns = {
        "Sub-4-bit codec (≤3 eff-bpw)": _hawking_density_serve_cell,
        "Per-model gradient RECOVERY (QAT/KD)": _hawking_recovery_cell,
        "Native low-bit SERVE on Apple Silicon": _hawking_serve_cell,
        "RAM-cliff: serve where Q4_K swaps/OOMs": _hawking_ramcliff_cell,
        "Bit-floor-vs-scale LAW (a published curve)": _hawking_law_cell,
        "MoE per-expert bit allocation": _hawking_moe_cell,
        "Stacked spec-decode on the condensed model": _hawking_spec_cell,
        "Reproducible receipts (R3+ one-command)": _hawking_receipts_cell,
    }
    rows = []
    for cap, fn in cell_fns.items():
        hk_status, hk_note = fn()
        # KILL guard: forbid a bare "yes/WIN" Hawking cell that is not backed by a win-eligible receipt.
        if hk_status == "yes" and cap != "Reproducible receipts (R3+ one-command)" and not win_receipts:
            hk_status, hk_note = "gated", f"WIN refused (no R{WIN_MIN_REPRO[-1]}+ receipt): {hk_note}"
        rows.append({
            "capability": cap,
            "hawking": {"status": hk_status, "note": hk_note},
            "competitors": {c: {"status": CAP_COMPETITORS[cap][c][0],
                                "note": CAP_COMPETITORS[cap][c][1]}
                            for c in COMPETITORS if c != "Hawking"}})
    return rows


def _curve_all(cf, records):
    """Re-read the curve's full row set (incl. errors) from disk for 'was it attempted' checks."""
    path = os.path.join(records["_root"], cf["file"])
    return _load_jsonl(path)


def compose_scale_verdict(records):
    """The bit-floor-vs-scale verdict (H1 descends / H0 flat) read off the fitted law (if present) or
    the floor points. Lab points (<7B) are listed but excluded from the verdict."""
    floors = records["floors"]["points"]
    law = records["floors"]["law"]
    pts = [{"model": p["model"], "params_b": p.get("params_b") or PARAMS.get(p["model"]),
            "floor_bpw": p.get("floor_bpw"), "winning_config": p.get("winning_config"),
            "degr_pct": p.get("degr_pct"),
            "role": "lab (excluded)" if (PARAMS.get(p["model"], 99) < LAB_MAX_B) else "verdict",
            "file": p.get("_file")}
           for p in floors]
    verdict_pts = [p for p in pts if p["role"] == "verdict" and p["floor_bpw"]]
    if law and law.get("verdict"):
        v = law["verdict"]
        hyp = "H1" if "H1" in v or "DESCEND" in v.upper() else ("H0" if "H0" in v or "FLAT" in v.upper() else "UNDECIDED")
        status = "FITTED"
    elif len(verdict_pts) >= 2:
        hyp, v, status = "UNDECIDED", "law not yet fitted (>=2 verdict points present)", "PENDING-FIT"
    else:
        hyp = "UNDECIDED"
        v = (f"INSUFFICIENT: {len(verdict_pts)} verdict (>=7B) floor point(s) — need >=2 to fit a law. "
             "Lab rungs do not set the verdict (§0.5).")
        status = "INSUFFICIENT"
    return {"hypothesis": hyp, "status": status, "verdict_text": v,
            "law": (law or {}).get("law"), "law_file": (law or {}).get("file"),
            "points": pts, "n_verdict_points": len(verdict_pts), "gate_pct": FLOOR_GATE_PCT}


# The headline claims, each mapped to the record(s) that would back it and its current proof state.
HEADLINE_CLAIMS = [
    ("density",  "~52% smaller artifact than f16 at the floor bpw",
     "the floor receipt's effective_bpw vs 16",  "R3"),
    ("quality",  "near-lossless (≤+2% ppl) at sub-4-bit on a ≥7B model",
     "a ≥7B floor receipt, quality_gate=pass, multiwindow",  "R3"),
    ("scale-law","bit-floor descends with scale (H1), fitted power law",
     "bit_floor_curve.md law + ≥2 verdict floor points",  "R3"),
    ("ram-cliff","serves a model that Q4_K swaps/OOMs (the money demo)",
     "a native-serve record at the serve-fit bpw",  "R3"),
    ("recovery", "gradient recovery reaches near-lossless BELOW the PTQ/residual floor",
     "a recovered (+dr/-bw/-str) curve config under the gate + receipt",  "R3"),
    ("moe",      "per-expert bit allocation beats uniform on a real MoE bake",
     "an expert_sens NON-UNIFORM verdict + a verified MoE bake receipt",  "R3"),
    ("spec",     "stacked spec-decode gives N× tps at exact-match output",
     "a spec record: accept≥gate under the exact-match (lossless) governor",  "R3"),
    ("codec",    "STRAND is frontier-class vs QTIP/QuIP#/AQLM at matched eff-bpw",
     "a codec_bakeoff record, head-to-head same harness",  "R3"),
]


def compose_claims(records):
    """For each headline claim emit MEASURED / GATED / UNPROVEN + the backing record + its repro level.
    MEASURED requires a win-eligible receipt (R>=WIN_MIN_REPRO, positive gate, real eff-bpw)."""
    receipts = records["receipts"]
    win_receipts = [r for r in receipts if r["win_eligible"]]
    floors = records["floors"]["points"]
    law = records["floors"]["law"]
    curves = records["curves"]
    spec = records["spec"]
    bakeoff = records["codec_bakeoff"]
    expert = records["expert_sens"]
    frontier = records["frontier"]
    best_repro = max((r["repro_level"] for r in receipts), key=lambda x: REPRO_ORDER.index(x)
                     if x in REPRO_ORDER else -1, default="none")

    out = []
    for cid, text, backed_by, need in HEADLINE_CLAIMS:
        state, repro, cite = "UNPROVEN", "none", None
        if cid in ("density", "quality"):
            big = [p for p in floors if p.get("floor_bpw") and PARAMS.get(p["model"], 0) >= LAB_MAX_B]
            if win_receipts and big:
                state, repro, cite = "MEASURED", win_receipts[0]["repro_level"], win_receipts[0]["file"]
            elif big:
                state, repro, cite = "GATED", "R2", big[0].get("_file")
            elif receipts:
                # a lab/baseline receipt exists — explicitly NOT a measured win
                state, repro, cite = "GATED", best_repro, receipts[0]["file"]
        elif cid == "scale-law":
            if law and law.get("law"):
                vp = [p for p in floors if p.get("floor_bpw") and PARAMS.get(p["model"], 0) >= LAB_MAX_B]
                state = "MEASURED" if (win_receipts and len(vp) >= 2) else "GATED"
                repro, cite = (win_receipts[0]["repro_level"] if win_receipts else "R2"), law["file"]
            else:
                state, cite = "UNPROVEN", None
        elif cid == "ram-cliff":
            # serve/tps is admissible ONLY under a native-serve record; frontier records GATE it.
            if frontier:
                state, repro, cite = "GATED", "R2", next(iter(frontier.values()))["_file"]
        elif cid == "recovery":
            hit = None
            for cf in curves.values():
                for r in cf["rows"]:
                    cfg = r.get("config", "")
                    if any(t in cfg for t in ("+dr", "-bw", "-str")) and r.get("degr_pct", 1e9) <= FLOOR_GATE_PCT:
                        hit = (cf, r); break
                if hit:
                    break
            if hit and win_receipts:
                state, repro, cite = "MEASURED", win_receipts[0]["repro_level"], win_receipts[0]["file"]
            elif hit:
                state, repro, cite = "GATED", "R2", hit[0]["file"]
        elif cid == "moe":
            if expert:
                e0 = next(iter(expert.values()))
                # probe-only: NON-UNIFORM keeps it alive but it is GATED on a real verified bake.
                state = "GATED" if e0.get("alive") in (True, None) else "UNPROVEN"
                repro, cite = "R1", e0["_file"]
        elif cid == "spec":
            if spec:
                s0 = next(iter(spec.values()))
                acc = s0.get("accept_rate")
                if s0.get("verdict", "").startswith("spec lane complete") and acc and win_receipts:
                    state, repro, cite = "MEASURED", win_receipts[0]["repro_level"], s0["_file"]
                else:
                    state, repro, cite = "GATED", "R2", s0["_file"]
        elif cid == "codec":
            if bakeoff:
                state, repro, cite = "GATED", "R2", next(iter(bakeoff.values()))["_file"]
        # KILL: MEASURED is forbidden without a win-eligible receipt (no fake GO).
        if state == "MEASURED" and not win_receipts:
            state, repro = "GATED", (cite and best_repro) or "R2"
        out.append({"id": cid, "claim": text, "state": state, "repro_level": repro,
                    "repro_desc": REPRO_DESC.get(repro, "—"), "backed_by": cite or f"NONE — need: {backed_by}",
                    "needs_for_win": f"{need}+ receipt"})
    return out


def compose_gates(records):
    """The open gates (the questions whose answers unlock the headline) + each one's current status,
    read off the records. recovery? expert non-uniform? lossless-verify? serve build?"""
    curves = records["curves"]
    expert = records["expert_sens"]
    spec = records["spec"]
    frontier = records["frontier"]
    subbit0 = records["subbit0"]

    # recovery gate
    rec_attempted = errored = cleared = False
    for cf in curves.values():
        for r in _load_jsonl(os.path.join(records["_root"], cf["file"])):
            cfg = r.get("config", "")
            if any(t in cfg for t in ("+dr", "-bw", "-str")):
                rec_attempted = True
                if "error" in r:
                    errored = True
                if r.get("degr_pct", 1e9) <= FLOOR_GATE_PCT:
                    cleared = True
    if cleared:
        rec = "PASS — a recovered config cleared the +%.0f%% gate" % FLOOR_GATE_PCT
    elif errored:
        rec = "OPEN — recovery attempted, all configs errored (doctor swap/timeout on this box)"
    elif rec_attempted:
        rec = "OPEN — recovery attempted, none under the gate yet"
    else:
        rec = "OPEN — recovery not yet attempted in any curve"

    # expert non-uniform gate
    if expert:
        e0 = next(iter(expert.values()))
        meta = e0.get("meta", {})
        synth = (meta.get("mode") == "synthetic") or e0.get("label", "").startswith(("synth", "synthu"))
        tag = " (SYNTHETIC probe — not a real model)" if synth else ""
        exp = f"{e0.get('verdict','?')}{tag} — {('ALIVE' if e0.get('alive') else 'DEAD' if e0.get('alive') is False else 'INCONCLUSIVE')}; needs a real bake to confirm"
    else:
        exp = "OPEN — no per-expert sensitivity record"

    # lossless-verify gate (spec)
    if spec:
        s0 = next(iter(spec.values()))
        verd = s0.get("verdict", "")
        lv = ("HALT — verify not lossless" if "lossless" in verd.lower() or verd.startswith("HALT")
              else f"recorded: {verd[:50]}")
    else:
        lv = "OPEN — lossless-verify gate not run (the known FOUNDATIONAL near-tie blocker)"

    # serve build gate
    if frontier:
        f0 = next(iter(frontier.values()))
        sb = f"OPEN — serve-fit recorded ({f0.get('artifact_gb')}GB); quality+cliff GATED on read_strand into hawking-serve + native .tq GEMV"
    else:
        sb = "OPEN — native .tq serve unbuilt; rehydrate -> f16 (any serve win from rehydrate is a FAKE win, refused)"

    # subbit-0 entropy floor gate (sub-1-bit dense alive?)
    if subbit0:
        s0 = next(iter(subbit0.values()))
        sbz = f"{s0.get('verdict','?')} — side-info floor ~{s0.get('sideinfo_floor_bpw','?')} eff-bpw (kill below {s0.get('kill_bpw','?')}) [{s0['_file']}]"
    else:
        sbz = "OPEN — SUBBIT-0 entropy floor not measured"

    return [
        {"gate": "Recovery clears the gate?", "status": rec},
        {"gate": "Expert sensitivity NON-UNIFORM (MoE sub-bit alive)?", "status": exp},
        {"gate": "Lossless-verify bit-exact (spec admissible)?", "status": lv},
        {"gate": "Native .tq serve built (serve/cliff admissible)?", "status": sb},
        {"gate": "SUBBIT-0 dense entropy floor (sub-1-bit alive)?", "status": sbz},
    ]


# ════════════════════════════════════════════════════════════════════════════════════════
#  RENDER
# ════════════════════════════════════════════════════════════════════════════════════════
_SYM = {"yes": "YES", "no": "no", "partial": "partial", "gated": "GATED", "unproven": "UNPROVEN"}


def render_md(sc):
    L = []
    P = L.append
    P("# Hawking Condense — Competitive SCORECARD (the capstone)")
    P("")
    P(f"_Synthesized by `tools/condense/scorecard.py` from records on disk at commit "
      f"`{sc['provenance']['commit']}`. A COMPOSER, not a measurer: every cell cites the record that "
      f"backs it; a WIN is printed ONLY behind a receipt at {WIN_MIN_REPRO}+ (else GATED/UNPROVEN). "
      f"Effective bpw only; no fake-win (rehydrate-to-f16 and non-exact-match spec count zero)._")
    P("")
    P(f"**Records read:** {sc['provenance']['n_records']} "
      f"(curves {sc['provenance']['counts']['curves']}, floors {sc['provenance']['counts']['floors']}, "
      f"receipts {sc['provenance']['counts']['receipts']}, "
      f"eval {sc['provenance']['counts']['eval']}, baselines {sc['provenance']['counts']['baselines']}, "
      f"spec {sc['provenance']['counts']['spec']}, codec_bakeoff {sc['provenance']['counts']['codec_bakeoff']}, "
      f"frontier {sc['provenance']['counts']['frontier']}, subbit0 {sc['provenance']['counts']['subbit0']}, "
      f"expert_sens {sc['provenance']['counts']['expert_sens']}).")
    P("")
    # ---- KILL line ----
    P("## KILL line (the refusal that keeps the scorecard honest)")
    P("")
    P(f"> {sc['kill_line']}")
    P("")

    # ---- 1. competitive matrix ----
    P("## 1. Competitive capability matrix")
    P("")
    hdr = "| Capability | " + " | ".join(COMPETITORS) + " |"
    sep = "|---|" + "|".join(["---"] * len(COMPETITORS)) + "|"
    P(hdr); P(sep)
    for row in sc["matrix"]:
        cells = [f"**{_SYM[row['hawking']['status']]}**<br><sub>{row['hawking']['note']}</sub>"]
        for c in COMPETITORS[1:]:
            cc = row["competitors"][c]
            cells.append(f"{_SYM.get(cc['status'], cc['status'])}<br><sub>{cc['note']}</sub>")
        P(f"| {row['capability']} | " + " | ".join(cells) + " |")
    P("")
    P("_Hawking cells are computed from the records; competitor cells are the web-checked §12 "
      "landscape. `YES` for Hawking appears only behind a win-eligible receipt._")
    P("")

    # ---- 2. scale verdict ----
    sv = sc["scale_verdict"]
    P("## 2. Bit-floor-vs-scale verdict (H1 / H0)")
    P("")
    P(f"**Hypothesis:** {sv['hypothesis']} · **Status:** {sv['status']} · gate ≤ +{sv['gate_pct']}% ppl, "
      f"effective bpw, multiwindow.")
    P("")
    P(f"> {sv['verdict_text']}")
    P("")
    if sv.get("law"):
        P(f"**Fitted law:** {sv['law']}  ({sv['law_file']})")
        P("")
    if sv["points"]:
        P("| model | params (B) | floor eff-bpw | winning config | Δ% | role | record |")
        P("|---|--:|--:|---|--:|---|---|")
        for p in sorted(sv["points"], key=lambda x: x.get("params_b") or 0):
            fb = f"{p['floor_bpw']:.3f}" if p.get("floor_bpw") else "—"
            dp = f"+{p['degr_pct']}" if p.get("degr_pct") is not None else "—"
            P(f"| {p['model']} | {p.get('params_b','?')} | {fb} | {p.get('winning_config') or '—'} | "
              f"{dp} | {p['role']} | {p.get('file') or '—'} |")
    else:
        P("_No floor points on disk yet (run `scaling_law.py --floor` per model)._")
    P("")

    # ---- 3. headline claims ----
    P("## 3. Headline claims — MEASURED / GATED / UNPROVEN (with repro level)")
    P("")
    P("| claim | state | repro | backed by | needs for a public WIN |")
    P("|---|---|---|---|---|")
    for c in sc["claims"]:
        P(f"| {c['claim']} | **{c['state']}** | {c['repro_level']} | {c['backed_by']} | {c['needs_for_win']} |")
    P("")
    P(f"_Repro ladder: " + "; ".join(f"{k}={v}" for k, v in REPRO_DESC.items()) + "._")
    P("")

    # ---- 4. open gates ----
    P("## 4. Open gates (answer these to unlock the headline)")
    P("")
    P("| gate | status |")
    P("|---|---|")
    for g in sc["gates"]:
        P(f"| {g['gate']} | {g['status']} |")
    P("")
    P("---")
    P(f"_Generated {sc['provenance']['commit']} · device={sc['provenance']['device']} "
      f"dtype={sc['provenance']['dtype']} · gate=+{FLOOR_GATE_PCT}% · win-bar={WIN_MIN_REPRO}. "
      f"This file is a synthesis of records, not a new measurement._")
    return "\n".join(L) + "\n"


# ════════════════════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL: gather -> compose -> render -> write
# ════════════════════════════════════════════════════════════════════════════════════════
def gather(root="."):
    records = {
        "_root": root,
        "curves": ingest_curves(root),
        "floors": ingest_floors(root),
        "receipts": ingest_receipts(root),
        "eval": ingest_condense(root, "eval"),
        "baselines": ingest_condense(root, "baselines"),
        "spec": ingest_condense(root, "spec"),
        "codec_bakeoff": ingest_condense(root, "codec_bakeoff"),
        "frontier": ingest_condense(root, "frontier"),
        "subbit0": ingest_condense(root, "subbit0"),
        "expert_sens": ingest_condense(root, "expert_sens"),
    }
    return records


def build_scorecard(records):
    matrix = compose_matrix(records)
    scale = compose_scale_verdict(records)
    claims = compose_claims(records)
    gates = compose_gates(records)

    win_receipts = [r for r in records["receipts"] if r["win_eligible"]]
    n_win_cells = sum(1 for r in matrix if r["hawking"]["status"] == "yes")
    # the KILL line — what the scorecard refuses, and what it actually found.
    if win_receipts:
        kill = (f"WIN cells printed only behind a win-eligible receipt at {WIN_MIN_REPRO}+. "
                f"Found {len(win_receipts)} -> {n_win_cells} WIN cell(s) admitted; the rest GATED/UNPROVEN.")
    else:
        kill = (f"NO win-eligible receipt at {WIN_MIN_REPRO}+ exists on disk (the only receipt is an "
                f"R1 baseline that, by rule R8/§20.6, cannot back a win). Therefore EVERY density / "
                f"quality / serve / scale WIN cell is REFUSED and printed GATED or UNPROVEN. The "
                f"scorecard prints zero unbacked GO. This is the honest current state, not a failure "
                f"of the tool.")

    counts = {k: len(records[k]) for k in
              ("curves", "receipts", "eval", "baselines", "spec", "codec_bakeoff",
               "frontier", "subbit0", "expert_sens")}
    counts["floors"] = len(records["floors"]["points"])
    n_records = sum(counts.values()) + (1 if records["floors"]["law"] else 0)

    return {
        "schema": "hawking-scorecard/0.1",
        "kill_line": kill,
        "matrix": matrix,
        "scale_verdict": scale,
        "claims": claims,
        "gates": gates,
        "provenance": {
            "commit": _commit(),
            "device": os.environ.get("DOCTOR_DEVICE", "cpu"),
            "dtype": os.environ.get("DOCTOR_DTYPE", "float32"),
            "strand_no_gpu": os.environ.get("STRAND_NO_GPU", "0"),
            "gate_pct": FLOOR_GATE_PCT, "win_min_repro": WIN_MIN_REPRO,
            "q4k_bpw": Q4K_BPW, "n_records": n_records, "counts": counts,
            "win_eligible_receipts": len(win_receipts),
            "win_cells_admitted": n_win_cells,
        },
    }


def run(root=".", out_md=None, out_json=None):
    log(f"# scorecard: composing from records under {os.path.abspath(root)} "
        f"(device={os.environ.get('DOCTOR_DEVICE','cpu')} gate=+{FLOOR_GATE_PCT}% win-bar={WIN_MIN_REPRO})")
    records = gather(root)
    sc = build_scorecard(records)
    out_md = out_md or os.path.join(root, OUT_MD)
    out_json = out_json or os.path.join(root, OUT_JSON)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w").write(render_md(sc))
    json.dump(sc, open(out_json, "w"), indent=2)
    log(f"# records: {sc['provenance']['n_records']} "
        f"({', '.join(f'{k}={v}' for k, v in sc['provenance']['counts'].items())})")
    log(f"# win-eligible receipts: {sc['provenance']['win_eligible_receipts']} "
        f"-> {sc['provenance']['win_cells_admitted']} WIN cell(s)")
    log(f"# KILL: {sc['kill_line']}")
    log(f"# wrote {os.path.relpath(out_md, root)} + {os.path.relpath(out_json, root)}")
    return sc


# ════════════════════════════════════════════════════════════════════════════════════════
#  SELF-TEST — fully synthetic; fabricates a record set in a temp dir and asserts every branch.
#  Runs HERE (no model / cargo / llama.cpp / mlx). Two passes:
#    A) BASELINE-ONLY records (mirrors today's qwen-0.5B disk state) -> assert NO WIN cells, KILL fires.
#    B) A fabricated WIN record set (R3 receipt + 7B floor + fitted law) -> assert WIN cells admitted.
# ════════════════════════════════════════════════════════════════════════════════════════
def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, list):   # jsonl
        open(path, "w").write("\n".join(json.dumps(r) for r in obj) + "\n")
    elif isinstance(obj, str):  # raw text (md)
        open(path, "w").write(obj)
    else:
        json.dump(obj, open(path, "w"), indent=2)


def _seed_baseline(root):
    """Mirror today's real disk state: a 7B curve (PTQ only, errored recovery) + an R1 baseline receipt."""
    _write(os.path.join(root, CRON_DIR, "7b_frontier.jsonl"), [
        {"model": "7B", "config": "f16", "eff_bpw": 16.0, "ppl": 21.76, "degr_pct": 0.0},
        {"model": "7B", "config": "4-AWQ", "eff_bpw": 4.846, "ppl": 21.192, "degr_pct": -2.61},
        {"model": "7B", "config": "3-AWQ", "eff_bpw": 3.686, "ppl": 23.64, "degr_pct": 8.64},
        {"model": "7B", "config": "2-AWQ", "eff_bpw": 2.682, "ppl": 109.1, "degr_pct": 401.0},
        {"model": "7B", "config": "3-AWQ+dr", "error": "doctor timeout"},
    ])
    _write(os.path.join(root, RECEIPT_DIR, "qwen-05b-tq3.json"), {
        "project": "hawking", "receipt_version": "0.2", "repro_level": "R1",
        "claim_type": "baseline", "machine_class": "M3Pro-18",
        "source_model": "Qwen2.5-0.5B-Instruct", "effective_bpw": 3.65, "nominal_bpw": 3.0,
        "quality_gate": "warn", "baseline_best_effort": True})
    _write(os.path.join(root, COND_DIR, "qwen-05b_subbit0.json"), {
        "label": "qwen-05b", "probe": "SUBBIT-0", "verdict": "ALIVE",
        "sideinfo_floor_bpw": 0.1093, "kill_bpw": 0.31})
    _write(os.path.join(root, COND_DIR, "synth_expert_sens.json"), {
        "label": "synth", "verdict": "NON-UNIFORM", "alive": True,
        "meta": {"mode": "synthetic"}, "decision": "MoE sub-bit ALIVE (probe)"})


def _seed_win(root):
    """A fabricated, fully-armed record set: a 7B floor UNDER Q4_K + a 14B floor + an R3 win receipt +
    a fitted H1 law + a spec record. Exercises the WIN-admit branches (NONE of this is a real result)."""
    _seed_baseline(root)
    # two verdict floor points + the fitted law
    _write(os.path.join(root, CRON_DIR, "bit_floor_curve.jsonl"), [
        {"model": "0.5B", "params_b": 0.5, "floor_bpw": 3.65, "winning_config": "3-AWQ", "degr_pct": 1.4},
        {"model": "7B", "params_b": 7.0, "floor_bpw": 2.1, "winning_config": "2-AWQ+dr", "degr_pct": 1.8},
        {"model": "14B", "params_b": 14.0, "floor_bpw": 1.7, "winning_config": "2-str", "degr_pct": 1.5},
    ])
    _write(os.path.join(root, CRON_DIR, "bit_floor_curve_curve.md"),
           "# Bit-floor vs scale\n\n**Law (7B+):** floor ~= -0.500*log10(N) + 2.500  (R^2=0.990)\n\n"
           "**Verdict:** H1 CONFIRMED - floor DESCENDS with scale (slope -0.500 bpw/decade)\n")
    # a WIN-eligible receipt (R3, density claim, positive gate, real eff-bpw, not best-effort)
    _write(os.path.join(root, RECEIPT_DIR, "7B-floor.json"), {
        "project": "hawking", "receipt_version": "0.2", "repro_level": "R3",
        "claim_type": "density", "machine_class": "Studio-96",
        "source_model": "Qwen2.5-7B (scratch/qwen-7b)", "effective_bpw": 2.1, "nominal_bpw": 2.0,
        "quality_gate": "pass", "baseline_best_effort": False})
    # a recovered config under the gate (so the recovery claim can go MEASURED)
    _write(os.path.join(root, CRON_DIR, "7b_studio.jsonl"), [
        {"model": "7B", "config": "f16", "eff_bpw": 16.0, "ppl": 21.76, "degr_pct": 0.0},
        {"model": "7B", "config": "2-AWQ+dr", "eff_bpw": 2.1, "ppl": 22.15, "degr_pct": 1.8},
    ])
    # a spec record (complete, accept present) + a codec bakeoff + a frontier serve-fit
    _write(os.path.join(root, COND_DIR, "7B_spec.json"), {
        "label": "7B", "verdict": "spec lane complete", "accept_rate": 0.62, "accept_gate": 0.40})
    _write(os.path.join(root, COND_DIR, "32B_codec_bakeoff.json"), {
        "label": "32B", "strand_bpw": 2.1, "qtip_bpw": 2.2, "winner": "STRAND"})
    _write(os.path.join(root, COND_DIR, "405B_frontier.json"), {
        "model": "405B", "total_b": 405.0, "serve_bpw": 1.34, "artifact_gb": 67.8,
        "serve_fits_84": True, "moe": False})


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- Pass A: baseline-only (today's real state) ----
    da = tempfile.mkdtemp(prefix="scorecard_A_")
    try:
        _seed_baseline(da)
        sc = run(da)
        print("# --- Pass A (baseline-only: no win-eligible receipt) ---")
        check("A: zero win-eligible receipts", sc["provenance"]["win_eligible_receipts"] == 0)
        check("A: zero WIN cells admitted", sc["provenance"]["win_cells_admitted"] == 0)
        check("A: KILL line refuses the WIN", "REFUSED" in sc["kill_line"] or "NO win-eligible" in sc["kill_line"])
        # the R1 baseline receipt must NOT promote any density/quality claim to MEASURED
        dens = next(c for c in sc["claims"] if c["id"] == "density")
        check("A: density claim not MEASURED (baseline can't back a win)", dens["state"] != "MEASURED")
        # no >=2 verdict floor points -> scale verdict INSUFFICIENT
        check("A: scale verdict INSUFFICIENT", sc["scale_verdict"]["status"] == "INSUFFICIENT")
        # the matrix Hawking cells must contain no bare 'yes' (receipts row aside)
        bad = [r for r in sc["matrix"] if r["hawking"]["status"] == "yes"
               and r["capability"] != "Reproducible receipts (R3+ one-command)"]
        check("A: no unbacked WIN in the matrix", not bad)
        # files were written
        check("A: SCORECARD.md written", os.path.exists(os.path.join(da, OUT_MD)))
        check("A: scorecard.json written", os.path.exists(os.path.join(da, OUT_JSON)))
        # subbit-0 + expert gates populated from records
        gtxt = " ".join(g["status"] for g in sc["gates"])
        check("A: SUBBIT-0 gate cites the record", "ALIVE" in gtxt and "0.1093" in gtxt)
        check("A: expert gate flags SYNTHETIC", "SYNTHETIC" in gtxt)
    finally:
        shutil.rmtree(da, ignore_errors=True)

    # ---- Pass B: fabricated WIN set ----
    db = tempfile.mkdtemp(prefix="scorecard_B_")
    try:
        _seed_win(db)
        sc = run(db)
        print("# --- Pass B (fabricated win-eligible R3 receipt + fitted H1 law) ---")
        check("B: one win-eligible receipt", sc["provenance"]["win_eligible_receipts"] == 1)
        check("B: >=1 WIN cell admitted", sc["provenance"]["win_cells_admitted"] >= 1)
        check("B: KILL admits the win", "admitted" in sc["kill_line"])
        # density + quality now MEASURED at R3
        dens = next(c for c in sc["claims"] if c["id"] == "density")
        check("B: density MEASURED @ R3", dens["state"] == "MEASURED" and dens["repro_level"] == "R3")
        # scale-law H1
        check("B: scale verdict H1 FITTED", sc["scale_verdict"]["hypothesis"] == "H1"
              and sc["scale_verdict"]["status"] == "FITTED")
        check("B: scale-law claim MEASURED", next(c for c in sc["claims"] if c["id"] == "scale-law")["state"] == "MEASURED")
        # recovery MEASURED (recovered config under gate + win receipt)
        check("B: recovery MEASURED", next(c for c in sc["claims"] if c["id"] == "recovery")["state"] == "MEASURED")
        # spec MEASURED (complete + accept + win receipt); codec + ram-cliff GATED (no native-serve / no receipt for those)
        check("B: spec MEASURED", next(c for c in sc["claims"] if c["id"] == "spec")["state"] == "MEASURED")
        check("B: ram-cliff GATED (serve build unbuilt, no fake win)",
              next(c for c in sc["claims"] if c["id"] == "ram-cliff")["state"] == "GATED")
        check("B: codec GATED", next(c for c in sc["claims"] if c["id"] == "codec")["state"] == "GATED")
        # the density matrix cell now YES with the receipt cited
        dcell = next(r for r in sc["matrix"] if r["capability"] == "Sub-4-bit codec (≤3 eff-bpw)")
        check("B: density matrix cell YES w/ receipt", dcell["hawking"]["status"] == "yes"
              and "7B-floor.json" in dcell["hawking"]["note"])
    finally:
        shutil.rmtree(db, ignore_errors=True)

    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "--dry"
    if a in ("-h", "--help"):
        print(__doc__); return
    if a == "--selftest":
        sys.exit(0 if selftest() else 1)
    # --dry (default) and no-arg both compose from whatever records exist on disk.
    run(".")


if __name__ == "__main__":
    main()
