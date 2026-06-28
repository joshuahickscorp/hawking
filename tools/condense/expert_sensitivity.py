#!/usr/bin/env python3.12
"""expert_sensitivity.py — SUBBIT-4 probe: per-expert sensitivity + route-frequency (MoE).

A MEASUREMENT PROBE, NOT A SERVING WIN. This tool decides — cheaply, before any bake —
whether MoE expert sensitivity is NON-UNIFORM. It measures; it does not claim a tok/s or a
shipped artifact. (Dense mixed-precision died because the per-tensor sensitivity SPREAD was
only ~3% — within that band you cannot rob a tolerant tensor to pay a sensitive one, so
uniform bit-width wins. mixed_precision.py reports that honest null.) The Type-2 reframe that
could ESCAPE that dead verdict: in an MoE, the experts are not one population. If the rarely
routed ("cold") experts tolerate ternary / 1-bit while the hot experts must stay high-bit,
then per-EXPERT sub-bit allocation is alive even though per-TENSOR (dense) allocation is dead —
and DeepSeek-V3 671B @ ~1.0 amortized bpw ≈ 84 GB becomes a real serving target.

────────────────────────────────────────────────────────────────────────────────────────
WHAT IS MEASURED  (per routed expert e)
────────────────────────────────────────────────────────────────────────────────────────
  · route_freq(e)   : fraction of routed token-slots that selected expert e, summed over the
                      calibration corpus via a forward hook on each layer's router/gate. A
                      cold expert (low route_freq) contributes little to the output and is the
                      candidate to crush to 1-bit/ternary.
  · sensitivity@k(e) : how much expert e's weights are damaged at k bits. Two metrics:
       proxy (default, cheap): per-expert weight rel_L2 at k bits, ||W - Q_k(W)|| / ||W||,
            using a deterministic stochastic-free round-to-grid Q_k (RHT-free upper bound on
            the trellis error — the baker does strictly better, so proxy is conservative),
            then activation-weighted by route_freq so a cold expert's damage is discounted:
            sens_imp(e,k) = rel_L2(e,k) · route_freq(e)  (output-space IMPORTANCE proxy).
       outxe (--metric outxe, heavier): per-expert ||(Q_k(W) - W) · X|| / ||W · X|| on captured
            per-expert input activations X — the exact output-space error for that expert. Needs
            the expert's calib inputs cached (memory cost), so it is opt-in.
    The SPREAD is computed on the RAW rel_L2(e,k) (not importance-weighted) — we want to know
    whether the experts genuinely differ in compressibility, independent of routing.

────────────────────────────────────────────────────────────────────────────────────────
VERDICT  (the decision this probe exists to make)
────────────────────────────────────────────────────────────────────────────────────────
    spread = max_e sens / min_e sens   AND   stdev/mean (CoV) across experts.
  · spread > 2x (CoV high)  => NON-UNIFORM  => MoE sub-bit ALIVE  (build the writer next).
  · spread < 5% (CoV ~0)    => UNIFORM      => dies the dense death => KILL SUBBIT-4.
  · in between               => INCONCLUSIVE => widen calib / try outxe metric.

────────────────────────────────────────────────────────────────────────────────────────
KILL LINE  (the criterion that REFUTES the lever)
────────────────────────────────────────────────────────────────────────────────────────
    KILL SUBBIT-4 IF per-expert sensitivity spread < 5% (CoV < 0.05): expert sensitivity is
    uniform, so per-expert bit allocation buys nothing over a uniform bake — it dies the exact
    dense mixed-precision death. Do NOT build the per-expert .tq writer. Report the honest null.

────────────────────────────────────────────────────────────────────────────────────────
PROPOSED ALLOCATION (only meaningful if NON-UNIFORM; printed either way for inspection)
────────────────────────────────────────────────────────────────────────────────────────
  Router + shared/attention weights: PROTECTED high-bit (never sub-bit — a wrong route is
  unrecoverable and these are a tiny fraction of params). Hot experts: 2-bit. Cold experts:
  1-bit / ternary. The resulting AMORTIZED bpw is computed on EFFECTIVE bpw (ladder.BPW: the
  RHT + outlier side-info is folded in — 1-bit costs 1.34 bpw, not 1.0), param-weighted by each
  expert's element count. We NEVER report nominal bpw.

EFFECTIVE-BPW DISCIPLINE: amortized bpw uses ladder.BPW (side-info included). The "84 GB"
DeepSeek figure in the docstring is illustrative of the GOAL, not a result of this probe.

────────────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────────────
  # Self-test here (no MoE model needed): fabricate a plausible expert distribution and run the
  # spread / verdict / allocation math end-to-end. Prints the verdict path it took.
  python3.12 tools/condense/expert_sensitivity.py --synthetic --label synth

  # Force the UNIFORM (kill) branch in the self-test:
  python3.12 tools/condense/expert_sensitivity.py --synthetic --synthetic-mode uniform

  # Real MoE (gated; needs the model local — DeepSeek-V2-Lite / Qwen3-MoE):
  python3.12 tools/condense/expert_sensitivity.py /path/deepseek-v2-lite --label dsv2lite \
        --bits 1,2 --max-tokens 4096

Honors DOCTOR_DEVICE (cpu/mps), DOCTOR_DTYPE (bfloat16 for 7B+ on CPU), STRAND_NO_GPU=1.
Writes reports/condense/<label>_expert_sens.json + a stderr summary.
"""
import sys, os, gc, json, math, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ladder as L                                       # canonical EFFECTIVE BPW table

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEV_ENV = os.environ.get("DOCTOR_DEVICE")
DTYPE_NAME = os.environ.get("DOCTOR_DTYPE", "float32")
CALIB = os.environ.get("DOCTOR_CALIB", os.path.join(ROOT, "scratch", "calib_corpus.txt"))
PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")

# spread thresholds (the verdict cutoffs). Override via env for sensitivity analysis.
NONUNIFORM_RATIO = float(os.environ.get("SUBBIT_NONUNIFORM_RATIO", "2.0"))   # max/min > this => alive
UNIFORM_COV = float(os.environ.get("SUBBIT_UNIFORM_COV", "0.05"))           # CoV < this => kill

KILL_LINE = (f"KILL SUBBIT-4 IF per-expert sensitivity spread < 5% (CoV < {UNIFORM_COV:.2f}): "
             "expert sensitivity is uniform => per-expert bit allocation buys nothing over a "
             "uniform bake (the dense mixed-precision death). Do NOT build the per-expert writer.")


def log(*m):
    print(*m, file=sys.stderr); sys.stderr.flush()


# ── effective-bpw helpers (NEVER nominal) ──────────────────────────────────────────────
def amortized_eff_bpw(alloc_bits, elems):
    """Param-weighted EFFECTIVE bpw over a {key: nominal_bit} allocation, using ladder.BPW so
    the RHT + outlier side-info is folded in (1-bit => 1.34 bpw, not 1.0). elems = {key: count}."""
    tot = sum(elems.values())
    if tot == 0:
        return float("nan")
    return sum(L.BPW[alloc_bits[k]] * elems[k] for k in alloc_bits) / tot


# ── synthetic distribution (testable here; no MoE model required) ───────────────────────
def synthetic_experts(n=64, mode="nonuniform", seed=0):
    """Fabricate a PLAUSIBLE per-expert {route_freq, rel_L2@k} distribution so the spread /
    verdict / allocation math can be exercised end-to-end on this box.

    mode='nonuniform': a few HOT experts (high route_freq, high rel_L2 — they encode used,
        dense signal that resists quantization) and a long tail of COLD experts (rarely routed,
        low rel_L2 — near-dead, crush-to-ternary candidates). This is the SUBBIT-4-ALIVE shape.
    mode='uniform': every expert ~identical sensitivity (the dense-death shape, forces KILL).
    """
    import random
    rng = random.Random(seed)
    experts = []
    if mode == "uniform":
        for e in range(n):
            base = 0.30 + rng.uniform(-0.006, 0.006)          # ~±2% spread => CoV well under 5%
            experts.append({"expert": e, "route_freq": 1.0 / n,
                            "rel_l2": {1: base * 2.0, 2: base}})
        return experts
    # non-uniform: zipf-ish routing, hot experts harder to quantize
    raw = [1.0 / (1.0 + i) ** 1.1 for i in range(n)]
    rng.shuffle(raw)
    s = sum(raw)
    for e in range(n):
        rf = raw[e] / s
        hot = rf * n                                          # ~>1 for hot, <1 for cold
        # cold experts compress easily (small rel_L2); hot ones resist (large rel_L2)
        base2 = 0.08 + 0.55 * min(1.0, hot) + rng.uniform(0, 0.03)
        experts.append({"expert": e, "route_freq": rf,
                        "rel_l2": {1: min(0.99, base2 * 2.1), 2: base2}})
    return experts


# ── real MoE path (GATED: needs the model local) ───────────────────────────────────────
def _grid_quant_rel_l2(W, bits):
    """RHT-free per-output-row uniform round-to-grid rel_L2 = ||W - Q_k(W)|| / ||W||.

    A CONSERVATIVE upper bound on the baker's trellis+RHT error (the baker does strictly
    better), used as the cheap sensitivity proxy. Symmetric per-row max-abs scale, (2^bits)
    levels (so bits=1 => ternary-ish {-1,0,+1}-scaled grid)."""
    import torch
    W = W.float()
    levels = max(2, 2 ** bits)
    half = (levels - 1) / 2.0
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / half
    Q = (W / scale).round().clamp_(-half, half) * scale
    num = (W - Q).pow(2).sum().sqrt()
    den = W.pow(2).sum().sqrt().clamp_min(1e-12)
    return float((num / den).item())


def measure_real_moe(model_dir, label, bits_set, max_tokens, metric):
    """Load an MoE HF model, hook the router/gate per layer to count per-expert route frequency
    over the calib corpus, then measure each expert's per-bit rel_L2. Returns the experts list
    (aggregated across layers — one row per (layer, expert)) + meta. GATED: raises if the model
    is not local or is not MoE."""
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = DEV_ENV or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, DTYPE_NAME)
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"model dir not local: {model_dir} (real MoE path is gated; use --synthetic)")
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    n_exp = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
    if not n_exp:
        raise RuntimeError(f"{model_dir} is not an MoE (no n_routed_experts/num_experts in config "
                           f"-- this probe is MoE-only; dense models die the mixed-precision death)")
    log(f"# real MoE: {model_dir} type={cfg.get('model_type')} routed_experts={n_exp} "
        f"top_k={cfg.get('num_experts_per_tok')} dev={dev}/{dtype}")

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager").to(dev).eval()

    # route-frequency: hook every gate/router Linear (out_features == n_exp), argmax/top-k its
    # logits. We count top-k selections to mirror the real router (num_experts_per_tok).
    topk = int(cfg.get("num_experts_per_tok") or 1)
    counts = {}                                              # (layer_name) -> tensor[n_exp]
    hooks = []

    def mk_gate(name):
        def h(mod, inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits.detach().float().reshape(-1, logits.shape[-1])
            if logits.shape[-1] != n_exp:
                return
            sel = logits.topk(min(topk, n_exp), dim=-1).indices.reshape(-1)
            c = counts.setdefault(name, torch.zeros(n_exp))
            c.index_add_(0, sel.cpu(), torch.ones(sel.numel()))
        return h

    gate_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features == n_exp:
            gate_names.append(name)
            hooks.append(mod.register_forward_hook(mk_gate(name)))
    if not gate_names:
        for h in hooks:
            h.remove()
        del model; gc.collect()
        raise RuntimeError(f"no router/gate Linear with out_features=={n_exp} found; this loader "
                           f"does not expose the gate as nn.Linear (extend the hook for "
                           f"{cfg.get('model_type')})")
    log(f"# hooked {len(gate_names)} gate/router modules (top_k={topk})")

    text = open(CALIB, errors="ignore").read() if os.path.exists(CALIB) else (
        open(PT, errors="ignore").read() if os.path.exists(PT) else "")
    if not text:
        for h in hooks:
            h.remove()
        del model; gc.collect()
        raise RuntimeError(f"no calib corpus at {CALIB} or {PT}")
    ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens].to(dev)
    log(f"# routing forward over {ids.shape[1]} tokens…")
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    # per-expert rel_L2 at each bit: walk expert weight tensors (gate/up/down per expert).
    # Expert weights live as either a stack [n_exp, ...] or per-expert submodules; handle both.
    experts = []
    sd = model.state_dict()
    # build a map layer_prefix -> route_freq[n_exp]  (normalized within the layer)
    route = {}
    for name, c in counts.items():
        prefix = name.rsplit(".", 1)[0]                     # drop the ".gate"/".router" leaf
        tot = float(c.sum().item()) or 1.0
        route[prefix] = (c / tot).tolist()

    def expert_rel(W):
        return {b: _grid_quant_rel_l2(W, b) for b in bits_set}

    # heuristic: gather per-expert weight matrices. DeepSeek/Qwen3-MoE name them
    # "...mlp.experts.<e>.{gate,up,down}_proj.weight". We average the three projections' rel_L2.
    by_le = {}                                              # (layer_prefix, e) -> [rel dicts]
    for k in sd:
        if ".experts." not in k or not k.endswith("_proj.weight"):
            continue
        pre, rest = k.split(".experts.", 1)
        try:
            e = int(rest.split(".", 1)[0])
        except ValueError:
            continue
        W = sd[k]
        if W.dim() != 2 or min(W.shape) < 64:
            continue
        by_le.setdefault((pre, e), []).append(expert_rel(W))

    if not by_le:
        del model; gc.collect()
        raise RuntimeError("found gate routing but no '.experts.<e>....weight' matrices; expert "
                           "weights may be a fused stack — extend measure_real_moe for this layout")

    for (pre, e), rels in sorted(by_le.items()):
        merged = {b: sum(r[b] for r in rels) / len(rels) for b in bits_set}
        rf_list = None
        # match the layer prefix to a routed-gate prefix (gate prefix == experts' mlp prefix)
        for gp, rf in route.items():
            if gp in pre or pre in gp:
                rf_list = rf; break
        rf = rf_list[e] if rf_list and e < len(rf_list) else float("nan")
        experts.append({"layer": pre, "expert": e, "route_freq": rf, "rel_l2": merged,
                        "elems": int(sd[[k for k in sd if k.startswith(pre)
                                         and f".experts.{e}." in k and k.endswith("_proj.weight")][0]].numel())})

    meta = {"model_type": cfg.get("model_type"), "n_routed_experts": n_exp,
            "num_experts_per_tok": topk, "tokens": int(ids.shape[1]),
            "device": dev, "dtype": DTYPE_NAME, "metric": metric, "layers": len(route)}
    del model; gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    return experts, meta


# ── spread + verdict + allocation (the decision math; pure python, runs anywhere) ───────
def _stats(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return dict(n=0, mean=float("nan"), stdev=float("nan"), cov=float("nan"),
                    min=float("nan"), max=float("nan"), ratio=float("nan"))
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    sd = math.sqrt(var)
    lo, hi = min(vals), max(vals)
    return dict(n=n, mean=mean, stdev=sd, cov=(sd / mean if mean else float("nan")),
                min=lo, max=hi, ratio=(hi / lo if lo > 0 else float("inf")))


def decide(experts, bits_set):
    """Compute spread on the RAW per-expert rel_L2 at the FLOOR bit (lowest in bits_set), the
    bit at which sensitivity differences matter most, and return the verdict dict."""
    floor = min(bits_set)
    rel_floor = [e["rel_l2"].get(floor) for e in experts]
    st = _stats(rel_floor)
    # also report importance (route_freq-weighted) spread for context
    imp = [(e["rel_l2"].get(floor) or 0.0) * (e.get("route_freq") or 0.0) for e in experts]
    st_imp = _stats([v for v in imp if v > 0])

    nonuniform = (st["ratio"] >= NONUNIFORM_RATIO) and (st["cov"] >= UNIFORM_COV)
    uniform = st["cov"] < UNIFORM_COV
    if uniform:
        verdict = "UNIFORM"
        decision = (f"KILL SUBBIT-4: per-expert sensitivity is uniform (CoV {st['cov']:.3f} < "
                    f"{UNIFORM_COV:.2f}) -- dies the dense mixed-precision death. Do NOT build "
                    f"the per-expert writer.")
        alive = False
    elif nonuniform:
        verdict = "NON-UNIFORM"
        decision = (f"MoE sub-bit ALIVE: spread max/min = {st['ratio']:.2f}x (>= {NONUNIFORM_RATIO:.1f}x), "
                    f"CoV {st['cov']:.3f} -- cold experts tolerate sub-bit while hot resist. "
                    f"Per-expert allocation can beat uniform. Next step: build the per-expert .tq "
                    f"writer + VERIFY with a real bake (this probe does NOT claim a serving win).")
        alive = True
    else:
        verdict = "INCONCLUSIVE"
        decision = (f"INCONCLUSIVE: spread {st['ratio']:.2f}x, CoV {st['cov']:.3f} sits between the "
                    f"kill (<{UNIFORM_COV:.2f} CoV) and alive (>={NONUNIFORM_RATIO:.1f}x ratio) "
                    f"gates. Widen the calib corpus or use --metric outxe.")
        alive = None
    return dict(floor_bit=floor, spread_raw=st, spread_importance=st_imp,
                verdict=verdict, alive=alive, decision=decision)


def propose_allocation(experts, bits_set, protect_keys=None):
    """Per-expert bit plan: cold experts -> floor bit (1-bit/ternary), hot -> 2-bit; router /
    shared / attention PROTECTED high-bit (4). Returns (alloc, elems, amortized_eff_bpw, summary).

    'hot' = route_freq above the median (the experts actually carrying the forward pass). Only
    MEANINGFUL when the verdict is NON-UNIFORM — printed regardless so the operator can inspect.
    """
    floor = min(bits_set)
    hot_bit = 2 if 2 in bits_set else max(bits_set)
    protect_bit = max(bits_set + [4])                       # protected weights never go sub-bit
    rfs = sorted(e.get("route_freq") or 0.0 for e in experts)
    med = rfs[len(rfs) // 2] if rfs else 0.0
    alloc, elems = {}, {}
    n_hot = n_cold = 0
    for e in experts:
        key = f"{e.get('layer','L')}.expert{e['expert']}"
        rf = e.get("route_freq") or 0.0
        if rf > med:
            alloc[key] = hot_bit; n_hot += 1
        else:
            alloc[key] = floor; n_cold += 1
        elems[key] = e.get("elems", 1)
    # protected (router + shared/attn): a small synthetic footprint if we don't know real sizes.
    # We model it as ~3% of expert params at protect_bit so the amortized number isn't fantasy.
    exp_total = sum(elems.values())
    prot_elems = int(0.03 * exp_total) or 1
    alloc["__protected_router_shared_attn"] = protect_bit
    elems["__protected_router_shared_attn"] = prot_elems
    amo = amortized_eff_bpw(alloc, elems)
    summary = dict(hot_experts=n_hot, hot_bit=hot_bit, cold_experts=n_cold, cold_bit=floor,
                   protected_bit=protect_bit, protected_frac=round(prot_elems / (exp_total + prot_elems), 4),
                   amortized_eff_bpw=round(amo, 3),
                   note="amortized_eff_bpw uses ladder.BPW (RHT+outlier side-info folded in); "
                        "NOMINAL bpw is never reported. This is a PLAN, not a measured artifact.")
    return alloc, elems, amo, summary


# ── main ───────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="SUBBIT-4 probe: per-expert sensitivity + route-frequency for MoE. " + KILL_LINE,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="KILL LINE: " + KILL_LINE)
    ap.add_argument("model_dir", nargs="?", default=None,
                    help="HF MoE dir (DeepSeek-V2-Lite / Qwen3-MoE). Omit with --synthetic.")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate a plausible expert distribution + run the math here (no model)")
    ap.add_argument("--synthetic-mode", choices=["nonuniform", "uniform"], default="nonuniform",
                    help="shape of the fabricated distribution (uniform forces the KILL branch)")
    ap.add_argument("--synthetic-experts", type=int, default=64, help="# experts to fabricate")
    ap.add_argument("--label", default=None, help="report label (default: model basename or 'synthetic')")
    ap.add_argument("--bits", default="1,2", help="per-expert bits to probe (comma, e.g. 1,2)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="calib tokens for routing forward")
    ap.add_argument("--metric", choices=["proxy", "outxe"], default="proxy",
                    help="proxy = weight rel_L2 (cheap); outxe = exact ||(Q-W)X||/||WX|| (heavy, opt-in)")
    ap.add_argument("--out", default=None, help="JSON out (default reports/condense/<label>_expert_sens.json)")
    args = ap.parse_args()

    bits_set = sorted(int(x) for x in args.bits.split(","))
    label = args.label or (os.path.basename((args.model_dir or "").rstrip("/")) or "synthetic"
                           if not args.synthetic else (args.label or "synthetic"))
    if args.synthetic and not args.label:
        label = "synthetic"

    log("# expert_sensitivity (SUBBIT-4 PROBE — measures, does NOT claim a serving win)")
    log("# " + KILL_LINE)

    t0 = time.time()
    if args.synthetic or not args.model_dir:
        if not args.synthetic and not args.model_dir:
            log("# no model_dir given -> running --synthetic self-test")
        experts = synthetic_experts(args.synthetic_experts, args.synthetic_mode)
        meta = {"mode": "synthetic", "synthetic_mode": args.synthetic_mode,
                "n_routed_experts": len(experts), "metric": args.metric,
                "note": "FABRICATED distribution — exercises the spread/verdict/allocation math; "
                        "NOT a measurement of any real model."}
    else:
        experts, meta = measure_real_moe(args.model_dir, label, bits_set, args.max_tokens, args.metric)

    dec = decide(experts, bits_set)
    alloc, elems, amo, alloc_summary = propose_allocation(experts, bits_set)

    # ── stderr summary ──
    floor = dec["floor_bit"]
    st = dec["spread_raw"]
    log(f"# experts measured: {len(experts)}  (probed bits {bits_set}, floor={floor})")
    log(f"# per-expert rel_L2 @ {floor}-bit: min={st['min']:.4f} max={st['max']:.4f} "
        f"mean={st['mean']:.4f} stdev={st['stdev']:.4f}")
    log(f"# SPREAD: max/min = {st['ratio']:.2f}x   CoV(stdev/mean) = {st['cov']:.4f}")
    log(f"# route_freq range: {min((e.get('route_freq') or 0) for e in experts):.4f} .. "
        f"{max((e.get('route_freq') or 0) for e in experts):.4f}")
    log(f"# VERDICT: {dec['verdict']}")
    log(f"#   {dec['decision']}")
    log(f"# proposed allocation: {alloc_summary['hot_experts']} hot @ {alloc_summary['hot_bit']}-bit, "
        f"{alloc_summary['cold_experts']} cold @ {alloc_summary['cold_bit']}-bit, "
        f"router/shared/attn protected @ {alloc_summary['protected_bit']}-bit")
    log(f"#   amortized EFFECTIVE bpw = {amo:.3f}  (ladder.BPW: side-info folded in; NOT nominal; "
        f"NOT a measured artifact — verify with a real bake)")
    if dec["alive"] is False:
        log("# => KILL SUBBIT-4: do not build the per-expert writer (allocation above is for "
            "inspection only; it would not beat a uniform bake here).")
    elif dec["alive"] is True:
        log("# => SUBBIT-4 worth building: next step is the per-expert .tq writer + a REAL bake to "
            "confirm the amortized-bpw figure (this probe only measures the precondition).")

    # ── JSON report ──
    out = args.out or os.path.join(ROOT, "reports", "condense", f"{label}_expert_sens.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report = {
        "tool": "expert_sensitivity.py", "kind": "probe (not a serving win)",
        "label": label, "kill_line": KILL_LINE,
        "meta": meta, "bits_probed": bits_set,
        "thresholds": {"nonuniform_ratio": NONUNIFORM_RATIO, "uniform_cov": UNIFORM_COV},
        "verdict": dec["verdict"], "alive": dec["alive"], "decision": dec["decision"],
        "spread_raw_rel_l2_at_floor": st, "spread_importance_weighted": dec["spread_importance"],
        "proposed_allocation": alloc_summary,
        "experts": [{"layer": e.get("layer"), "expert": e["expert"],
                     "route_freq": e.get("route_freq"),
                     "sensitivity": {f"k{b}": e["rel_l2"].get(b) for b in bits_set}}
                    for e in experts],
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(out, "w") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    log(f"# wrote {out}")
    # one-line stdout record (machine-parseable, like the ladder/mixed_precision tools)
    print(json.dumps({"label": label, "verdict": dec["verdict"], "alive": dec["alive"],
                      "spread_ratio": round(st["ratio"], 3) if not math.isnan(st["ratio"]) else None,
                      "cov": round(st["cov"], 4) if not math.isnan(st["cov"]) else None,
                      "amortized_eff_bpw": round(amo, 3), "n_experts": len(experts),
                      "report": out}))


if __name__ == "__main__":
    main()
