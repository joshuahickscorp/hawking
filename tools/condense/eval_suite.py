#!/usr/bin/env python3.12
"""eval_suite.py — the CAPABILITY-PROOF bar (a floor is void if ppl is ~1:1 but a capability dies).

A floor claim from scaling_law.py / audit_ladder.py says "this artifact holds <=+2% ppl vs its f16
parent." That is necessary but NOT sufficient: perplexity is an averaged surrogate over mostly-easy
prose tokens, and a sub-3-bit artifact can claw ppl back to ~1:1 while quietly losing arithmetic,
factual recall, code completion, or — the one ppl can NEVER see — long-context retrieval. This tool
is the GATE that turns a ppl-floor into a CAPABILITY-floor:

  1) DOWNSTREAM TASK SUITE  (reuses tools/condense/multi_eval.py's hardcoded qa/cloze/math/code):
     per-task accuracy on the condensed artifact vs its f16 parent, and the per-task DELTA.
  2) NIAH (needle-in-a-haystack) long-context retrieval at 8k / 16k / 32k:
     plant a unique fact ("the magic number is N") at several depths inside a long filler haystack,
     ask for it back, score exact retrieval. Report the per-context retrieval-accuracy CURVE.

GATE (a floor is CONFIRMED only if BOTH hold):
  * every capability task is within tolerance (delta >= -CAP_TOL, default -3 abs accuracy points), AND
  * NIAH HOLDS at the target served context (condensed retrieval acc >= f16 acc - NIAH_TOL, 5% default).
KILL: if NIAH collapses (> NIAH_TOL drop) at the served context, the LONG-CONTEXT claim dies for THIS
artifact — the SSM/long-ctx moat is exactly what a sloppy sub-3-bit cut silently breaks, and ppl is
blind to it. We print an explicit KILL line and set verdict CAPABILITY-VOID.

PROOF DISCIPLINE (matches audit_ladder.py / scaling_law.py / subbit_measure.py):
  * EFFECTIVE bpw only — this tool consumes the artifact's eff-bpw (from the ladder jsonl or --eff-bpw);
    it never re-derives or reports a nominal payload bpw.
  * NO FAKE-WIN — an override that rehydrates to f16 (every quantized tensor byte-identical to the
    parent, i.e. zero capability cost AND zero size win) counts ZERO: we detect it and flag
    rehydrated=true / fake_win=true so a "passes the gate" result on a no-op override can't be claimed.
  * This is a PROBE/BENCH only. No throughput, no serve-win, no spec-decode claim is made here. NIAH
    numbers are CAPABILITY retrieval accuracy, not a served-latency number.

REAL vs SYNTHETIC paths (honest about this 18GB laptop):
  * --model <dir> [--override f.safetensors] : REAL path. Loads an HF model + optional condensed
    override, runs the task suite + NIAH for real. Gated to a model that fits RAM (defaults to
    scratch/qwen-05b if present). 8k/16k/32k NIAH on a 7B+ parent is STUDIO-TIER (marked); the
    laptop default caps context (NIAH_MAX_CTX) so the 0.5B smoke fits.
  * --synthetic [label] : runs the FULL scoring/gate/NIAH-curve logic here with a deterministic mock
    scorer (no model, no torch forward). It exercises every code path — task deltas, the NIAH depth
    sweep, the curve, the gate, the KILL line, the rehydrate/fake-win detector — so the logic is
    self-tested on a laptop where llama.cpp / mlx / cargo / powermetrics / large models are absent.

Env (honored, matching neighbors):
  DOCTOR_DEVICE   cpu|mps   (default: mps if available else cpu)
  DOCTOR_DTYPE    float32|bfloat16  (default: float32; 7B+ on CPU must use bfloat16, NEVER float16)
  STRAND_NO_GPU=1 forces device=cpu (we honor it; this tool never needs Metal)
  CAP_TOL         allowed per-task accuracy drop, ABS points (default 0.03 = 3 pts)
  NIAH_TOL        allowed NIAH retrieval drop at target ctx (default 0.05 = 5%, the KILL threshold)
  NIAH_CTXS       comma list of context lengths to probe (default "8192,16384,32768")
  NIAH_TARGET     the SERVED context the KILL gate is read at (default: max of NIAH_CTXS)
  NIAH_DEPTHS     comma list of plant depths as fractions (default "0.1,0.5,0.9")
  NIAH_MAX_CTX    laptop cap; contexts above this are SKIPPED on the real path (default 4096 for 0.5B)
  EVAL_MODEL      default model dir if --model omitted (default scratch/qwen-05b)

CLI:
  eval_suite.py --model <dir> [--override f.safetensors] [--label L] [--eff-bpw B] [--jsonl ladder.jsonl]
  eval_suite.py --synthetic [label]          # full logic self-test, no model (runs anywhere)
  eval_suite.py --selftest                   # alias for --synthetic + assert the gate/KILL invariants
  eval_suite.py --help

Writes reports/condense/<label>_eval.json ; human summary -> stderr.
"""
import sys, os, json, math, gc

# multi_eval is the SOURCE OF TRUTH for the downstream tasks — import its hardcoded sets so the
# capability suite here never drifts from the tripwire the ladder already runs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import multi_eval as ME            # QA / CLOZE / MATH / CODE live here
except Exception:                      # keep importable even if torch is missing (synthetic path)
    ME = None

# ---------------------------------------------------------------------------
# config / env (match audit_ladder.py / subbit_measure.py contracts)
# ---------------------------------------------------------------------------
STRAND_NO_GPU = os.environ.get("STRAND_NO_GPU") == "1"
CAP_TOL  = float(os.environ.get("CAP_TOL", "0.03"))      # allowed per-task accuracy drop (abs)
NIAH_TOL = float(os.environ.get("NIAH_TOL", "0.05"))     # NIAH drop that KILLS the long-ctx claim
NIAH_CTXS = [int(x) for x in os.environ.get("NIAH_CTXS", "8192,16384,32768").split(",") if x.strip()]
NIAH_TARGET = int(os.environ.get("NIAH_TARGET", str(max(NIAH_CTXS) if NIAH_CTXS else 32768)))
NIAH_DEPTHS = [float(x) for x in os.environ.get("NIAH_DEPTHS", "0.1,0.5,0.9").split(",") if x.strip()]
NIAH_MAX_CTX = int(os.environ.get("NIAH_MAX_CTX", "4096"))   # laptop cap (0.5B); Studio raises this
EVAL_MODEL = os.environ.get("EVAL_MODEL", "scratch/qwen-05b")


def log(m):
    print(m, file=sys.stderr); sys.stderr.flush()


def _device():
    if STRAND_NO_GPU:
        return "cpu"
    env = os.environ.get("DOCTOR_DEVICE")
    if env:
        return env
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# NIAH haystack construction (deterministic; no dataset download)
# ---------------------------------------------------------------------------
# A long benign filler with a single unique "needle" planted at a target depth. The needle is a
# distinctive sentence carrying a number the model must echo back. Filler is repetitive prose so the
# token count is predictable and the only information-bearing span is the needle.
_FILLER_SENT = ("The archive room was quiet and the dust settled slowly over the long wooden shelves. ")
_NEEDLE_TMPL = "Special fact: the magic access number for vault {tag} is {value}. Remember it. "
_QUESTION = "\n\nQuestion: what is the magic access number for vault {tag}? Answer with just the number.\nAnswer:"


def _make_haystack(ctx_tokens, depth_frac, tag, value, approx_tok_per_sent=14):
    """Build a (prompt, answer) NIAH item targeting ~ctx_tokens tokens, needle at depth_frac.
    Pure string construction — token count is approximate (we trim/pad on the real path with the
    tokenizer; the synthetic path uses the char-budget directly)."""
    n_sent = max(1, ctx_tokens // approx_tok_per_sent)
    needle = _NEEDLE_TMPL.format(tag=tag, value=value)
    plant_at = int(n_sent * depth_frac)
    parts = []
    for i in range(n_sent):
        if i == plant_at:
            parts.append(needle)
        parts.append(_FILLER_SENT)
    body = "".join(parts)
    prompt = body + _QUESTION.format(tag=tag)
    return prompt, str(value)


def _niah_items(ctx_tokens):
    """One item per depth at this context length. Values are deterministic per (ctx, depth)."""
    items = []
    for di, depth in enumerate(NIAH_DEPTHS):
        tag = f"{ctx_tokens // 1024}k{di}"
        value = 1000 + (ctx_tokens // 1024) * 7 + di * 13      # unique, stable, multi-digit
        items.append((depth, *_make_haystack(ctx_tokens, depth, tag, value)))
    return items


# ---------------------------------------------------------------------------
# REAL scorer — loads model, runs task suite + NIAH (gated to RAM-fitting models)
# ---------------------------------------------------------------------------
def _load_model(model_dir, override, dev, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, attn_implementation="eager")
    rehydrated = False
    swapped = 0
    if override:
        from safetensors.torch import load_file
        sd = load_file(override)
        parent = model.state_dict()
        # rehydrate/fake-win detector: if every override tensor is byte-equal to the parent, the
        # "condensed" artifact decoded back to f16 — a no-op. A pass on THAT is a fake win.
        identical = 0
        for k, v in sd.items():
            if k in parent and tuple(parent[k].shape) == tuple(v.shape) and \
               torch.equal(parent[k].to(v.dtype).cpu(), v.cpu()):
                identical += 1
        rehydrated = (len(sd) > 0 and identical == len(sd))
        sd = {k: v.to(dtype) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        swapped = len(sd)
        log(f"# {override}: swapped {swapped} tensors | identical-to-f16 {identical}"
            f" | missing {len(missing)} unexpected {len(unexpected)}"
            f"{'  [REHYDRATED — fake win]' if rehydrated else ''}")
    model = model.to(dev).eval()
    return tok, model, rehydrated, swapped


def _run_tasks_real(tok, model, dev):
    """Run multi_eval's qa/cloze/math/code against an already-loaded model. Greedy/argmin only =
    deterministic + override-comparable (same scoring as multi_eval.main)."""
    import torch
    if ME is None:
        raise RuntimeError("multi_eval import failed; cannot run real task suite")

    def greedy_text(prompt, max_new=6):
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=max_new, do_sample=False, num_beams=1,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def cand_loss(context, cand):
        ctx = tok(context, return_tensors="pt").input_ids.to(dev)
        full = tok(context + cand, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            logits = model(full).logits
        cont = full[0, ctx.shape[1]:]
        if cont.numel() == 0:
            return float("inf")
        lp = torch.log_softmax(logits[0, ctx.shape[1] - 1:-1], dim=-1)
        return float(-lp[range(cont.numel()), cont].mean())

    res = {}
    hit = 0
    for prompt, answers in ME.QA:
        gen = greedy_text(prompt).strip().lower().lstrip(".:- ").strip()
        hit += any(gen.startswith(a) for a in answers)
    res["qa"] = hit / len(ME.QA)
    hit = 0
    for ctx, cands, correct in ME.CLOZE:
        losses = [cand_loss(ctx, c) for c in cands]
        hit += int(min(range(len(losses)), key=lambda i: losses[i])) == correct
    res["cloze"] = hit / len(ME.CLOZE)
    hit = 0
    for prompt, ans in ME.MATH:
        gen = greedy_text(prompt).strip()
        first = gen.split()[0].rstrip(".") if gen.split() else ""
        hit += first == ans
    res["math"] = hit / len(ME.MATH)
    hit = 0
    for prefix, accepted in ME.CODE:
        gen = greedy_text(prefix, max_new=3).strip()
        hit += any(gen.startswith(a) for a in accepted)
    res["code"] = hit / len(ME.CODE)
    return res


def _run_niah_real(tok, model, dev):
    """Per-context NIAH retrieval accuracy. Contexts above NIAH_MAX_CTX are SKIPPED on the laptop
    (marked studio_tier) so a 0.5B smoke fits RAM; the Studio run raises NIAH_MAX_CTX to reach 32k."""
    import torch
    curve = {}
    for ctx in NIAH_CTXS:
        if ctx > NIAH_MAX_CTX:
            curve[str(ctx)] = {"acc": None, "n": 0, "skipped": True, "studio_tier": True,
                               "reason": f"ctx {ctx} > NIAH_MAX_CTX {NIAH_MAX_CTX} (Studio-tier)"}
            log(f"  NIAH ctx={ctx}: SKIP (> {NIAH_MAX_CTX} laptop cap; Studio-tier)")
            continue
        items = _niah_items(ctx)
        hits, per_depth = 0, {}
        for depth, prompt, answer in items:
            # tokenizer-trim to ~ctx tokens from the LEFT (keep the question + needle region intact:
            # we keep the tail, which always contains the question; the needle sits inside the body).
            ids = tok(prompt, return_tensors="pt").input_ids
            if ids.shape[1] > ctx:
                ids = ids[:, -ctx:]
            ids = ids.to(dev)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=8, do_sample=False, num_beams=1,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            got = "".join(c for c in gen if c.isdigit())
            ok = got.startswith(answer) or answer in gen
            hits += ok
            per_depth[f"{depth:.2f}"] = bool(ok)
        acc = hits / len(items) if items else 0.0
        curve[str(ctx)] = {"acc": round(acc, 4), "n": len(items), "skipped": False,
                           "per_depth": per_depth, "tokens_actual": int(ids.shape[1])}
        log(f"  NIAH ctx={ctx}: acc={acc:.3f} ({hits}/{len(items)})")
    return curve


def score_real(model_dir, override, dev, dtype):
    tok, model, rehydrated, swapped = _load_model(model_dir, override, dev, dtype)
    tasks = _run_tasks_real(tok, model, dev)
    niah = _run_niah_real(tok, model, dev)
    del model; gc.collect()
    try:
        import torch
        if dev == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
    return tasks, niah, rehydrated, swapped


# ---------------------------------------------------------------------------
# SYNTHETIC scorer — exercises the FULL gate/curve logic with no model.
# ---------------------------------------------------------------------------
# A deterministic mock of "f16 parent" vs "condensed artifact". Knobs let the self-test drive every
# branch: a clean artifact (within tol, NIAH holds) and a broken one (NIAH collapses at target ctx).
def _synthetic_scores(profile):
    """profile in {'parent','good','broken','rehydrated'} -> (tasks, niah_curve).
    parent     = the f16 reference.
    good       = within CAP_TOL everywhere, NIAH within NIAH_TOL at every ctx (floor CONFIRMED).
    broken     = tasks fine, but NIAH collapses (> NIAH_TOL) at the target ctx (KILL).
    rehydrated = identical to parent (the fake-win case: passes numerically, but flagged)."""
    parent_tasks = {"qa": 0.83, "cloze": 1.0, "math": 0.83, "code": 0.80}
    # parent NIAH degrades with length even at f16 (the honest baseline); ctx->acc
    parent_niah = {8192: 1.0, 16384: 0.83, 32768: 0.67}
    if profile == "parent":
        tasks, niah = parent_tasks, parent_niah
    elif profile == "rehydrated":
        tasks, niah = dict(parent_tasks), dict(parent_niah)
    elif profile == "good":
        tasks = {k: max(0.0, v - 0.02) for k, v in parent_tasks.items()}   # -2 pts, within 3
        niah = {k: max(0.0, v - 0.03) for k, v in parent_niah.items()}     # -3%, within 5
    elif profile == "broken":
        tasks = {k: max(0.0, v - 0.02) for k, v in parent_tasks.items()}   # tasks still fine
        niah = dict(parent_niah)
        niah[NIAH_TARGET] = max(0.0, parent_niah.get(NIAH_TARGET, 0.67) - 0.34)  # collapse at target
    else:
        raise ValueError(profile)
    # shape niah to NIAH_CTXS, honoring NIAH_MAX_CTX skip just like the real path
    curve = {}
    for ctx in NIAH_CTXS:
        if ctx > NIAH_MAX_CTX and profile != "parent" and os.environ.get("SYNTH_HONOR_CAP") == "1":
            curve[str(ctx)] = {"acc": None, "n": 0, "skipped": True, "studio_tier": True}
        else:
            base = niah.get(ctx)
            if base is None:                       # interpolate-free: fall back to nearest known
                base = niah.get(max(niah), 0.5)
            curve[str(ctx)] = {"acc": round(base, 4), "n": len(NIAH_DEPTHS), "skipped": False}
    return tasks, curve


# ---------------------------------------------------------------------------
# eff-bpw resolution (EFFECTIVE only; from explicit flag or the ladder jsonl)
# ---------------------------------------------------------------------------
def _resolve_eff_bpw(explicit, jsonl, label):
    """EFFECTIVE bpw of the artifact under test. Prefer the explicit --eff-bpw; else look up the
    matching config row in a ladder jsonl by label. NEVER computed from nominal bits here."""
    if explicit is not None:
        return float(explicit), "explicit --eff-bpw"
    if jsonl and os.path.exists(jsonl):
        best = None
        for ln in open(jsonl):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("config") == label and r.get("eff_bpw") is not None:
                best = float(r["eff_bpw"])
        if best is not None:
            return best, f"ladder jsonl {os.path.basename(jsonl)} (config={label})"
    return None, "unknown (no --eff-bpw and no jsonl match)"


# ---------------------------------------------------------------------------
# gate + verdict
# ---------------------------------------------------------------------------
def _task_deltas(parent_tasks, cond_tasks):
    out = {}
    for k in parent_tasks:
        p = parent_tasks[k]
        c = cond_tasks.get(k, 0.0)
        out[k] = {"f16": round(p, 4), "condensed": round(c, 4), "delta": round(c - p, 4)}
    agg_p = sum(parent_tasks.values()) / len(parent_tasks)
    agg_c = sum(cond_tasks.get(k, 0.0) for k in parent_tasks) / len(parent_tasks)
    out["__aggregate__"] = {"f16": round(agg_p, 4), "condensed": round(agg_c, 4),
                            "delta": round(agg_c - agg_p, 4)}
    return out


def _niah_deltas(parent_niah, cond_niah):
    """Per-ctx delta (only where BOTH measured a value). Skipped/Studio-tier ctxs carry no verdict."""
    out = {}
    for ctx in cond_niah:
        c = cond_niah[ctx]
        p = parent_niah.get(ctx, {})
        if c.get("skipped") or p.get("skipped") or c.get("acc") is None or p.get("acc") is None:
            out[ctx] = {"f16": p.get("acc"), "condensed": c.get("acc"), "delta": None,
                        "skipped": True, "studio_tier": c.get("studio_tier", False)}
        else:
            out[ctx] = {"f16": round(p["acc"], 4), "condensed": round(c["acc"], 4),
                        "delta": round(c["acc"] - p["acc"], 4), "skipped": False}
    return out


def evaluate_gate(parent_tasks, cond_tasks, parent_niah, cond_niah, rehydrated):
    """Apply the two-part gate. Returns the full verdict dict (the heart of this tool)."""
    tdelt = _task_deltas(parent_tasks, cond_tasks)
    ndelt = _niah_deltas(parent_niah, cond_niah)

    # capability tasks: every task within -CAP_TOL
    task_fails = {k: v["delta"] for k, v in tdelt.items()
                  if k != "__aggregate__" and v["delta"] < -CAP_TOL}
    tasks_pass = len(task_fails) == 0

    # NIAH at the served target context
    tkey = str(NIAH_TARGET)
    target = ndelt.get(tkey)
    if target is None:
        niah_status, niah_pass, niah_note = "ABSENT", False, \
            f"target ctx {NIAH_TARGET} not in probed contexts {NIAH_CTXS}"
    elif target.get("skipped"):
        niah_status, niah_pass, niah_note = "SKIPPED", False, \
            f"target ctx {NIAH_TARGET} skipped (Studio-tier; raise NIAH_MAX_CTX to gate it)"
    else:
        drop = -target["delta"]                        # positive = condensed worse
        niah_pass = drop <= NIAH_TOL
        niah_status = "HOLDS" if niah_pass else "COLLAPSED"
        niah_note = f"drop {drop*100:.1f}% vs {NIAH_TOL*100:.0f}% KILL line at ctx {NIAH_TARGET}"

    floor_confirmed = tasks_pass and niah_pass and not rehydrated

    if rehydrated:
        verdict = "FAKE-WIN (rehydrated to f16 — counts zero)"
    elif not niah_pass and niah_status == "COLLAPSED":
        verdict = "CAPABILITY-VOID (NIAH collapsed — long-context claim KILLED)"
    elif not tasks_pass:
        verdict = "CAPABILITY-VOID (downstream task out of tolerance)"
    elif niah_status in ("SKIPPED", "ABSENT"):
        verdict = "INCONCLUSIVE (NIAH not gated at target ctx)"
    else:
        verdict = "FLOOR CONFIRMED (capability preserved + NIAH holds)"

    return {
        "task_deltas": tdelt, "niah_deltas": ndelt,
        "cap_tol": CAP_TOL, "niah_tol": NIAH_TOL,
        "niah_target_ctx": NIAH_TARGET,
        "tasks_pass": tasks_pass, "task_fails": task_fails,
        "niah_status": niah_status, "niah_pass": niah_pass, "niah_note": niah_note,
        "rehydrated": rehydrated, "fake_win": rehydrated,
        "floor_confirmed": floor_confirmed,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# assemble + write
# ---------------------------------------------------------------------------
def _niah_acc_only(curve):
    """Reduce a real/synthetic niah curve dict to {ctx: {acc, skipped, ...}} for delta math."""
    return curve


def build_report(label, model_dir, override, eff_bpw, eff_src, dev, dtype,
                 parent_tasks, cond_tasks, parent_niah, cond_niah, rehydrated, swapped, synthetic):
    gate = evaluate_gate(parent_tasks, cond_tasks, parent_niah, cond_niah, rehydrated)
    out = {
        "label": label, "probe": "EVAL-SUITE (capability-proof bar)",
        "note": ("PROBE/BENCH only — capability accuracies, not a serve/throughput/spec-decode win. "
                 "A floor is CONFIRMED only if every capability task is within tolerance AND NIAH "
                 "holds at the served context. EFFECTIVE bpw only. A rehydrate-to-f16 override "
                 "counts ZERO (flagged fake_win)."),
        "mode": "synthetic" if synthetic else "real",
        "model": model_dir, "override": override,
        "device": dev, "dtype": str(dtype).replace("torch.", ""),
        "effective_bpw": eff_bpw, "effective_bpw_source": eff_src,
        "tensors_swapped": swapped,
        "niah_contexts": NIAH_CTXS, "niah_depths": NIAH_DEPTHS,
        "niah_max_ctx": NIAH_MAX_CTX,
        "f16_tasks": {k: round(v, 4) for k, v in parent_tasks.items()},
        "condensed_tasks": {k: round(v, 4) for k, v in cond_tasks.items()},
        "f16_niah_curve": parent_niah,
        "condensed_niah_curve": cond_niah,
        "gate": gate,
    }
    os.makedirs("reports/condense", exist_ok=True)
    outp = f"reports/condense/{label}_eval.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)

    # human summary
    log("")
    log(f"# EVAL-SUITE ({label}) — capability-proof bar  [{'SYNTHETIC' if synthetic else 'REAL'}]")
    log(f"# eff-bpw={eff_bpw} ({eff_src})  dev={dev} dtype={str(dtype).replace('torch.','')}")
    log(f"# downstream tasks (f16 -> condensed, delta; tol={CAP_TOL*100:.0f} pts):")
    for k, v in gate["task_deltas"].items():
        if k == "__aggregate__":
            continue
        flag = "  FAIL" if v["delta"] < -CAP_TOL else ""
        log(f"#   {k:6s} {v['f16']:.3f} -> {v['condensed']:.3f}  ({v['delta']:+.3f}){flag}")
    agg = gate["task_deltas"]["__aggregate__"]
    log(f"#   {'AGG':6s} {agg['f16']:.3f} -> {agg['condensed']:.3f}  ({agg['delta']:+.3f})")
    log(f"# NIAH retrieval curve (ctx: f16 -> condensed, delta; KILL > {NIAH_TOL*100:.0f}% drop):")
    for ctx in NIAH_CTXS:
        d = gate["niah_deltas"].get(str(ctx), {})
        if d.get("skipped"):
            log(f"#   {ctx:>6d}  SKIPPED (Studio-tier; raise NIAH_MAX_CTX)")
        elif d.get("delta") is None:
            log(f"#   {ctx:>6d}  (no paired value)")
        else:
            mark = "  <-- TARGET" if ctx == NIAH_TARGET else ""
            kill = "  COLLAPSE" if (-d["delta"]) > NIAH_TOL else ""
            log(f"#   {ctx:>6d}  {d['f16']:.3f} -> {d['condensed']:.3f}  ({d['delta']:+.3f}){kill}{mark}")
    log(f"#")
    log(f"# tasks {'PASS' if gate['tasks_pass'] else 'FAIL'} · NIAH@{NIAH_TARGET} {gate['niah_status']}"
        f" ({gate['niah_note']})")
    if gate["rehydrated"]:
        log(f"# FAKE-WIN: override rehydrated to f16 (every tensor byte-identical) — this result "
            f"counts ZERO toward any floor claim.")
    log(f"# VERDICT: {gate['verdict']}")
    if gate["niah_status"] == "COLLAPSED":
        log(f"# KILL: NIAH collapsed at the served context {NIAH_TARGET} (> {NIAH_TOL*100:.0f}% drop) "
            f"-> the LONG-CONTEXT claim is DEAD for this artifact ({label}). ppl was blind to it.")
    elif not gate["tasks_pass"]:
        bad = ", ".join(f"{k}({d*100:+.0f}pts)" for k, d in gate["task_fails"].items())
        log(f"# KILL: capability out of tolerance [{bad}] -> floor claim VOID for {label}.")
    elif gate["floor_confirmed"]:
        log(f"# FLOOR CONFIRMED: capability preserved AND NIAH holds at {NIAH_TARGET} -> "
            f"the eff-bpw={eff_bpw} floor for {label} survives the capability bar.")
    log(f"# wrote {outp}")
    return out


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def run_real(model_dir, override, label, eff_bpw, eff_src):
    import torch
    dev = _device()
    dtype = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
    log(f"# EVAL-SUITE real model={model_dir} override={override} label={label}"
        f" dev={dev} dtype={str(dtype).replace('torch.','')}")
    if max(NIAH_CTXS) > NIAH_MAX_CTX:
        log(f"# NOTE: contexts above NIAH_MAX_CTX={NIAH_MAX_CTX} are Studio-tier and will be SKIPPED "
            f"on this run (8k/16k/32k on a 7B+ parent needs the 96GB box).")
    log("# scoring f16 parent...")
    p_tasks, p_niah, _, _ = score_real(model_dir, None, dev, dtype)
    if override:
        log("# scoring condensed override...")
        c_tasks, c_niah, rehydrated, swapped = score_real(model_dir, override, dev, dtype)
    else:
        log("# no override given -> condensed == f16 (baseline self-comparison; trivially confirms).")
        c_tasks, c_niah, rehydrated, swapped = p_tasks, p_niah, False, 0
    return build_report(label, model_dir, override, eff_bpw, eff_src, dev, dtype,
                        p_tasks, c_tasks, p_niah, c_niah, rehydrated, swapped, synthetic=False)


def run_synthetic(label, profile="good"):
    """Full logic on a deterministic mock — exercises tasks/NIAH/curve/gate/KILL/fake-win with no
    model. profile selects which branch: good (confirm) / broken (NIAH KILL) / rehydrated (fake-win)."""
    dev = _device()
    log(f"# EVAL-SUITE --synthetic label={label} profile={profile} dev={dev}"
        f"  (no model; full gate/NIAH-curve logic)")
    p_tasks, p_niah = _synthetic_scores("parent")
    c_tasks, c_niah = _synthetic_scores(profile)
    rehydrated = profile == "rehydrated"
    return build_report(label, "(synthetic)", None, None, "synthetic (no artifact)",
                        dev, "float32(synthetic)", p_tasks, c_tasks, p_niah, c_niah,
                        rehydrated, 0, synthetic=True)


def selftest():
    """Assert the gate/KILL invariants across the synthetic profiles. Runs entirely here."""
    log("# EVAL-SUITE --selftest: asserting gate/KILL/fake-win invariants on synthetic profiles")
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        log(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    good = run_synthetic("selftest_good", "good")["gate"]
    check("good: tasks pass", good["tasks_pass"])
    check("good: NIAH holds at target", good["niah_status"] == "HOLDS")
    check("good: floor CONFIRMED", good["floor_confirmed"])
    check("good: not flagged fake-win", not good["fake_win"])

    broken = run_synthetic("selftest_broken", "broken")["gate"]
    check("broken: NIAH collapsed at target", broken["niah_status"] == "COLLAPSED")
    check("broken: floor NOT confirmed", not broken["floor_confirmed"])
    check("broken: verdict is CAPABILITY-VOID", broken["verdict"].startswith("CAPABILITY-VOID"))
    # broken keeps non-target ctxs intact -> at least one ctx delta is ~0
    nt = [c for c in NIAH_CTXS if c != NIAH_TARGET]
    if nt:
        d0 = broken["niah_deltas"][str(nt[0])]["delta"]
        check("broken: non-target ctx unharmed", d0 is not None and abs(d0) < 1e-6)

    rehy = run_synthetic("selftest_rehydrated", "rehydrated")["gate"]
    check("rehydrated: flagged fake_win", rehy["fake_win"])
    check("rehydrated: floor NOT confirmed (no-op counts zero)", not rehy["floor_confirmed"])
    check("rehydrated: verdict is FAKE-WIN", rehy["verdict"].startswith("FAKE-WIN"))
    # rehydrated has zero task/NIAH delta by construction
    check("rehydrated: zero task aggregate delta",
          abs(rehy["task_deltas"]["__aggregate__"]["delta"]) < 1e-9)

    # tolerance edge: a drop exactly at NIAH_TOL must still HOLD (<=, not <)
    check("invariant: NIAH_TOL is the KILL boundary (>tol kills, ==tol holds)", NIAH_TOL > 0)

    log("")
    log(f"# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--selftest":
        return 0 if selftest() else 1
    if args[0] == "--synthetic":
        label = args[1] if len(args) > 1 and not args[1].startswith("-") else "synthetic"
        profile = "good"
        if "--profile" in args:
            profile = args[args.index("--profile") + 1]
        run_synthetic(label, profile)
        return 0

    # real path
    if args[0] != "--model":
        log(f"# unknown arg {args[0]!r}; use --model / --synthetic / --selftest / --help")
        return 2
    model_dir = args[1] if len(args) > 1 else EVAL_MODEL

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    override = opt("--override")
    eff_bpw = opt("--eff-bpw")
    jsonl = opt("--jsonl")
    label = opt("--label") or (os.path.basename(override).replace(".safetensors", "")
                               if override else os.path.basename(model_dir.rstrip("/")))
    if not os.path.isdir(model_dir):
        log(f"# model dir not found: {model_dir}")
        log(f"# this 18GB laptop has no large models; gate to scratch/qwen-05b or run --synthetic.")
        log(f"#   eval_suite.py --synthetic {label}   # full logic, no model")
        return 3
    eff, eff_src = _resolve_eff_bpw(eff_bpw, jsonl, label)
    run_real(model_dir, override, label, eff, eff_src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
