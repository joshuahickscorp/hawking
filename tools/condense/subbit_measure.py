#!/usr/bin/env python3.12
"""subbit_measure.py — SUBBIT-0: the entropy/compressibility FLOOR for sub-1-bit dense (plan gate).

A PROBE, not a codec and NOT a serving win. It measures, per 2D linear weight, the ACHIEVABLE
EFFECTIVE-bpw floor of a dense k-bit artifact = the hard wall below which sub-1-bit dense is
physically impossible regardless of how clever the codec or how much recovery training is thrown
at it. It claims no throughput, no quality, no deployment — only "the floor is here."

  floor_eff_bpw(k) = bulk_weight_entropy(k)            # order-0 Shannon bits/weight of the
                                                       #   k-bit symmetric/NF-style quant symbols
                   + scale_sideinfo                    # 8-bit per-256-block scales, their order-0
                                                       #   entropy amortized per weight (NOT 8/256)
                   + outlier_sideinfo                  # gap-coded position-index entropy (rANS-class)
                                                       #   + value bits, amortized per weight

EFFECTIVE bpw discipline (hard rule): we NEVER report the nominal k. The bulk term is the empirical
symbol entropy (always <= k, the wall the trellis coder approaches), and side-info is charged at its
order-0 ENTROPY, not its fixed-width billing — because an entropy coder (vendor sideinfo_rans.rs)
provably recovers the gap. This mirrors hawking's measured q2 artifact: scale_q entropy ~0.041 bpw,
outl_pos gap entropy ~0.078 bpw -> ~0.12 bpw side-info floor (~0.232 bpw of fixed-width waste). This
tool reproduces that CLASS of number from first principles, per-tensor, for k in {1,2,3}.

KILL LINE  (the criterion that REFUTES the sub-1-bit-dense lever):
  If the side-info floor ALONE (scale + outlier, summed, entropy-charged) exceeds ~0.31 bpw, then
  sub-1-bit DENSE is DEAD ON ARRIVAL: you have spent your entire <1-bit budget on bookkeeping before
  a single bit of weight signal is stored. The only survivors are MoE-only (where the active-param
  share dilutes the per-stored-weight side-info) or non-dense formats. The 0.31 threshold is the
  bpw headroom between a 1-bit dense target and the empirically observed ~0.69 bpw bulk floor of a
  1-bit NF symbol stream (1.0 - 0.69 ~= 0.31): if side-info eats more than that, you cannot reach
  1 bpw, let alone go below it.

Env (matches audit_ladder.py / scaling_law.py):
  DOCTOR_DEVICE   cpu|mps  (default: cpu; this tool is CPU-only — no forward pass, just histograms)
  DOCTOR_DTYPE    bfloat16 for 7B+ on CPU (default: float32; only affects the staging read)
  STRAND_NO_GPU=1 honored (we never touch Metal anyway).
  SUBBIT_GROUP    block size for per-group scales (default 256, matching the baker).
  SUBBIT_KILL_BPW side-info KILL threshold in bpw (default 0.31).
  SUBBIT_MAXTENS  cap tensors scanned (default 0 = all); for a fast smoke run set e.g. 8.

Usage:
  subbit_measure.py <model-dir> <label>            # measure a real model dir (streams weights)
  subbit_measure.py --dry <label>                  # synthetic self-test (runs anywhere, no model)
  subbit_measure.py --help                         # prints the KILL line + usage

Writes reports/condense/<label>_subbit0.json ; human summary -> stderr.
"""
import sys, os, json, math, gc
import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# config / env (match neighbors)
# ---------------------------------------------------------------------------
DEV = os.environ.get("DOCTOR_DEVICE", "cpu")            # probe is CPU-only; DEV recorded for provenance
DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
GROUP = int(os.environ.get("SUBBIT_GROUP", "256"))      # per-group scale block size (baker default)
SCALE_BITS = 8                                          # per-block scales quantized to 8 bits (then entropy-coded)
KILL_BPW = float(os.environ.get("SUBBIT_KILL_BPW", "0.31"))
MAXTENS = int(os.environ.get("SUBBIT_MAXTENS", "0"))    # 0 = all tensors
KS = (1, 2, 3)                                          # bit budgets to probe
OUTLIER_PCTS = (0.5, 1.0)                               # % of top-|w| kept as an 8-bit sparse channel
OUTLIER_VALUE_BITS = 8                                  # outlier values stored at 8 bits


def log(m):
    print(m, file=sys.stderr); sys.stderr.flush()


# ---------------------------------------------------------------------------
# entropy primitives (pure first-principles; mirror sideinfo_rans.rs order0_entropy)
# ---------------------------------------------------------------------------
def _order0_entropy_bits(counts):
    """Order-0 Shannon entropy (bits/symbol) from an iterable of nonneg integer counts.
    This is the LOWER BOUND an order-0 entropy coder (the vendor rANS) approaches; the real
    coder is within ~0.2 bit/sym of it (measured), so the floor we report is conservative-honest."""
    counts = [int(c) for c in counts if c > 0]
    n = sum(counts)
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        p = c / n
        h -= p * math.log2(p)
    return h


def _hist_entropy(int_tensor):
    """Order-0 entropy (bits/symbol) of an integer tensor via bincount on its symbol range."""
    t = int_tensor.reshape(-1).to(torch.int64)
    lo = int(t.min().item())
    idx = t - lo
    counts = torch.bincount(idx)
    return _order0_entropy_bits(counts.tolist())


# ---------------------------------------------------------------------------
# quantization symbol models (k-bit symmetric; the trellis coder's symbol alphabet)
# ---------------------------------------------------------------------------
def _quantize_symbols(w2d, k, group):
    """Per-group symmetric k-bit quantize a 2D weight along the last dim.

    Returns (q_symbols int16 [out,in], scale_q int32 [out, n_groups]).
      q_symbols : the k-bit code per weight in {-(2^(k-1)-1) .. +(2^(k-1)-1)} (symmetric).
                  For k=1 this is the sign in {-1,+1} (1 effective level pair) — the degenerate
                  case whose bulk entropy is the headline 1-bit floor.
      scale_q   : the per-group scale, itself quantized to SCALE_BITS bits (the side-info we then
                  entropy-code). Quantizing the scale is what makes the scale stream a finite
                  alphabet with measurable entropy — a float scale has no order-0 model.

    This is a measurement-grade scalar quantizer (round-to-nearest, no trellis search): the trellis
    only LOWERS the achieved bits vs this symbol histogram, so the entropy here is an UPPER bound on
    bulk bits => the reported floor is conservative (we never under-state the wall)."""
    out_f, in_f = w2d.shape
    ng = (in_f + group - 1) // group
    pad = ng * group - in_f
    w = w2d.to(torch.float32)
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    wg = w.reshape(out_f, ng, group)                      # [out, ng, group]
    qmax = (1 << (k - 1)) - 1 if k > 1 else 1             # k=1 -> {-1,+1}; k=2 -> {-1,0,1}; k=3 -> -3..3
    absmax = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)   # [out, ng, 1]
    scale = absmax / qmax                                 # float scale per group
    q = torch.round(wg / scale).clamp(-qmax, qmax).to(torch.int16)
    # quantize the scale itself to SCALE_BITS bits (log domain — scales are positive, span decades)
    # so the scale stream is a finite alphabet whose order-0 entropy we can charge honestly.
    s = scale.reshape(out_f, ng)
    smin, smax = float(s.min()), float(s.max())
    if smax <= smin:
        scale_q = torch.zeros_like(s, dtype=torch.int32)
    else:
        lg = torch.log(s.clamp_min(1e-12))
        lmin, lmax = float(lg.min()), float(lg.max())
        step = (lmax - lmin) / ((1 << SCALE_BITS) - 1)
        scale_q = torch.round((lg - lmin) / step).to(torch.int32)
    q = q[:, :ng * group].reshape(out_f, ng, group)[:, :, :group]  # keep padded shape consistent
    # strip the pad columns from the symbol grid before histogramming (don't count padding zeros)
    q_real = q.reshape(out_f, ng * group)[:, :in_f]
    return q_real, scale_q


def _bulk_entropy_bpw(q_symbols):
    """Order-0 entropy of the k-bit weight symbol stream, in bits/weight. This is the BULK term."""
    return _hist_entropy(q_symbols)


def _scale_sideinfo_bpw(scale_q, n_weights):
    """Side-info from the per-group scales: order-0 entropy of the 8-bit quantized scale stream,
    amortized over EVERY weight (entropy_bits_per_scale * n_groups / n_weights).
    NOT the fixed 8/group billing — the vendor rANS recovers the gap, so we charge the entropy."""
    n_groups = scale_q.numel()
    if n_groups == 0 or n_weights == 0:
        return 0.0, 0.0
    h_per_scale = _hist_entropy(scale_q)                  # bits per scale symbol (order-0)
    return (h_per_scale * n_groups) / n_weights, h_per_scale


def _outlier_sideinfo_bpw(w2d, pct, n_weights):
    """Side-info for keeping the top-`pct`% of |w| as an 8-bit sparse channel.

    Two charges, both amortized per weight:
      position bits = order-0 entropy of the GAP stream between sorted outlier flat-indices
                      (gap-coding is exactly positions_to_gaps in sideinfo_rans.rs) * n_outl
      value bits    = OUTLIER_VALUE_BITS * n_outl
    Position entropy is charged, NOT the fixed log2(n_weights) absolute-index width — the rANS
    gap coder provably beats fixed width (measured ledger: 22.6 -> 7.8 effective bits/pos)."""
    n_outl = max(1, int(round(n_weights * pct / 100.0)))
    if n_outl >= n_weights:
        n_outl = n_weights - 1
    flat = w2d.reshape(-1).abs().to(torch.float32)
    # top-n indices, then SORT ascending so the gap transform matches the codec's contract.
    top = torch.topk(flat, n_outl, largest=True).indices
    pos = torch.sort(top).values.to(torch.int64)
    gaps = pos.clone()
    gaps[1:] = pos[1:] - pos[:-1]                         # first absolute, rest = gap (>=1)
    h_gap = _hist_entropy(gaps)                           # order-0 entropy of the gap symbols
    pos_bits = (h_gap * n_outl) / n_weights
    val_bits = (OUTLIER_VALUE_BITS * n_outl) / n_weights
    return pos_bits + val_bits, {"n_outl": n_outl, "h_gap": h_gap,
                                 "pos_bpw": pos_bits, "val_bpw": val_bits}


# ---------------------------------------------------------------------------
# per-tensor measurement
# ---------------------------------------------------------------------------
# embed/lm_head are 2D and >=256 but the baker passes them through UNQUANTIZED (audit_ladder.py
# weights eff-bpw by the baker's quantized count, not the raw param count); including them would
# dilute the floor with weights that never carry the k-bit/side-info cost. Exclude by name.
_SKIP_SUBSTR = ("embed_tokens", "lm_head", "embeddings", "wte", "shared")


def _is_quantizable(name, shape):
    """Match the baker's is_quantizable_linear: 2D, both dims >= 256, and NOT an embedding/lm_head."""
    if any(s in name for s in _SKIP_SUBSTR):
        return False
    return len(shape) == 2 and min(shape) >= 256


def measure_tensor(name, w2d):
    """Return the per-tensor floor record across all k and outlier_pct."""
    out_f, in_f = w2d.shape
    n_weights = out_f * in_f
    rec = {"tensor": name, "shape": [out_f, in_f], "n_weights": n_weights, "k": {}}
    # outlier side-info does NOT depend on k (it's a position+value model of the weight matrix),
    # so compute it once per pct and reuse across k.
    outl = {}
    for pct in OUTLIER_PCTS:
        bpw, dbg = _outlier_sideinfo_bpw(w2d, pct, n_weights)
        outl[pct] = (bpw, dbg)
    for k in KS:
        q, scale_q = _quantize_symbols(w2d, k, GROUP)
        bulk = _bulk_entropy_bpw(q)
        scale_bpw, h_scale = _scale_sideinfo_bpw(scale_q, n_weights)
        per_pct = {}
        for pct in OUTLIER_PCTS:
            o_bpw, o_dbg = outl[pct]
            sideinfo = scale_bpw + o_bpw
            floor = bulk + sideinfo
            per_pct[f"{pct:.1f}"] = {
                "bulk_bpw": round(bulk, 5),
                "scale_sideinfo_bpw": round(scale_bpw, 5),
                "outlier_sideinfo_bpw": round(o_bpw, 5),
                "sideinfo_floor_bpw": round(sideinfo, 5),
                "floor_eff_bpw": round(floor, 5),
                "_outlier": {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                             for kk, vv in o_dbg.items()},
            }
        rec["k"][str(k)] = {"bulk_bpw": round(bulk, 5),
                            "scale_entropy_bits": round(h_scale, 4),
                            "scale_sideinfo_bpw": round(scale_bpw, 5),
                            "by_outlier_pct": per_pct}
        del q, scale_q
    return rec, n_weights


# ---------------------------------------------------------------------------
# model / synthetic iteration
# ---------------------------------------------------------------------------
def _iter_model_tensors(model_dir):
    """Yield (name, 2D float tensor) for every quantizable linear, streamed one at a time.
    Handles both single-file and sharded safetensors (peak ~ one tensor)."""
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        files = {None: single}
        idx = None
    else:
        idxp = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(idxp):
            raise FileNotFoundError(f"no model.safetensors or index in {model_dir}")
        idx = json.load(open(idxp))["weight_map"]
        files = {sh: os.path.join(model_dir, sh) for sh in set(idx.values())}
    handles = {sh: safe_open(p, framework="pt") for sh, p in files.items()}
    try:
        if idx is None:
            fh = handles[None]
            names = [k for k in fh.keys() if _is_quantizable(k, tuple(fh.get_slice(k).get_shape()))]
            loc = {k: None for k in names}
        else:
            names, loc = [], {}
            for sh, fh in handles.items():
                for k in fh.keys():
                    if _is_quantizable(k, tuple(fh.get_slice(k).get_shape())):
                        names.append(k); loc[k] = sh
            names.sort()
        if MAXTENS:
            names = names[:MAXTENS]
        for k in names:
            yield k, handles[loc[k]].get_tensor(k).to(DTYPE)
    finally:
        for fh in handles.values():
            if hasattr(fh, "close"):
                fh.close()


def _synthetic_tensors():
    """Deterministic synthetic linears resembling real Qwen-class FFN/attn shapes. Gaussian weights
    with a heavy outlier tail (a few super-outlier channels) so the outlier-position model exercises
    the same clustered-gap regime as a real matrix. Runs anywhere — no model on disk required."""
    g = torch.Generator().manual_seed(0x5CA1E000)
    shapes = [("synth.attn.q_proj", 896, 896),
              ("synth.attn.o_proj", 896, 896),
              ("synth.ffn.gate_proj", 4864, 896),
              ("synth.ffn.down_proj", 896, 4864)]
    for name, o, i in shapes:
        w = torch.randn(o, i, generator=g, dtype=torch.float32) * 0.02
        # inject ~10 super-outlier channels at ~20x magnitude (the real activation-outlier regime)
        for ch in range(0, i, max(1, i // 10)):
            w[:, ch] *= 20.0
        yield name, w.to(DTYPE)


# ---------------------------------------------------------------------------
# aggregation + verdict
# ---------------------------------------------------------------------------
def _aggregate(per_tensor, total_weights):
    """Model-wide floor = weight-by-weight average of each per-tensor term, for each (k, pct)."""
    agg = {}
    for k in KS:
        agg[str(k)] = {}
        for pct in OUTLIER_PCTS:
            pk = f"{pct:.1f}"
            acc = {"bulk_bpw": 0.0, "scale_sideinfo_bpw": 0.0,
                   "outlier_sideinfo_bpw": 0.0, "sideinfo_floor_bpw": 0.0, "floor_eff_bpw": 0.0}
            for rec, nw in per_tensor:
                cell = rec["k"][str(k)]["by_outlier_pct"][pk]
                for key in acc:
                    acc[key] += cell[key] * nw
            for key in acc:
                acc[key] = round(acc[key] / total_weights, 5) if total_weights else 0.0
            agg[str(k)][pk] = acc
    return agg


def _verdict(agg):
    """Headline: the side-info floor (scale+outlier, entropy-charged) over k and pct. The lever is
    DEAD if the MINIMUM achievable side-info floor across the grid still exceeds KILL_BPW — because
    that minimum is the most-favorable bookkeeping budget, and even it leaves no room under 1 bit."""
    best = None
    for k in KS:
        for pct in OUTLIER_PCTS:
            si = agg[str(k)][f"{pct:.1f}"]["sideinfo_floor_bpw"]
            if best is None or si < best[0]:
                best = (si, k, pct)
    si_floor, k_at, pct_at = best
    alive = si_floor < KILL_BPW
    return si_floor, k_at, pct_at, alive


def run(per_tensor_iter, label):
    per_tensor, total = [], 0
    for name, w in per_tensor_iter:
        rec, nw = measure_tensor(name, w)
        per_tensor.append((rec, nw)); total += nw
        f1 = rec["k"]["1"]["by_outlier_pct"]["1.0"]["sideinfo_floor_bpw"]
        log(f"  {name:40s} {list(rec['shape'])} sideinfo@1b/1%={f1:.4f} bpw")
        del w; gc.collect()
    if not per_tensor:
        raise RuntimeError("no quantizable 2D linear tensors found (>=256 in both dims)")
    agg = _aggregate(per_tensor, total)
    si_floor, k_at, pct_at, alive = _verdict(agg)
    out = {
        "label": label, "probe": "SUBBIT-0", "note": (
            "PROBE ONLY — measures the entropy/compressibility FLOOR of sub-1-bit DENSE; "
            "claims no serving/throughput/quality win. Effective bpw (entropy-charged side-info), "
            "never nominal. Bulk = order-0 symbol entropy (trellis lowers it further -> conservative)."),
        "device": DEV, "dtype": str(DTYPE).replace("torch.", ""),
        "group": GROUP, "scale_bits": SCALE_BITS, "outlier_value_bits": OUTLIER_VALUE_BITS,
        "ks": list(KS), "outlier_pcts": list(OUTLIER_PCTS),
        "n_tensors": len(per_tensor), "total_weights": total,
        "kill_bpw": KILL_BPW,
        "sideinfo_floor_bpw": round(si_floor, 5),
        "sideinfo_floor_at": {"k": k_at, "outlier_pct": pct_at},
        "verdict": "ALIVE" if alive else "DEAD",
        "aggregate": agg,
        "per_tensor": [r for r, _ in per_tensor],
    }
    os.makedirs("reports/condense", exist_ok=True)
    outp = f"reports/condense/{label}_subbit0.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    # human summary
    log("")
    log(f"# SUBBIT-0 floor  ({label})  — PROBE, not a serving win")
    log(f"# tensors={len(per_tensor)} weights={total:,} group={GROUP} kill={KILL_BPW} bpw")
    log("#  k  outl%   bulk   scale_si  outl_si  SIDEINFO  FLOOR_eff_bpw")
    for k in KS:
        for pct in OUTLIER_PCTS:
            c = agg[str(k)][f"{pct:.1f}"]
            log(f"#  {k}   {pct:>4.1f}   {c['bulk_bpw']:.4f}  {c['scale_sideinfo_bpw']:.4f}   "
                f"{c['outlier_sideinfo_bpw']:.4f}   {c['sideinfo_floor_bpw']:.4f}    {c['floor_eff_bpw']:.4f}")
    log(f"#")
    log(f"# side-info floor = {si_floor:.4f} bpw (best @ k={k_at}, outlier={pct_at}%); "
        f"sub-1-bit dense is {'ALIVE' if alive else 'DEAD'} "
        f"({'floor < ' if alive else 'floor >= '}{KILL_BPW} bpw KILL line)")
    if not alive:
        log(f"# KILL: side-info alone ({si_floor:.4f} bpw) >= {KILL_BPW} bpw -> sub-1-bit DENSE is "
            f"dead on arrival (MoE-only / non-dense survives).")
    log(f"# wrote {outp}")
    return out


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print(f"\nKILL LINE: side-info floor >= {KILL_BPW} bpw => sub-1-bit DENSE dead on arrival.")
        return
    if args[0] == "--dry":
        label = args[1] if len(args) > 1 else "synthetic"
        log(f"# SUBBIT-0 --dry synthetic self-test (no model) label={label}")
        run(_synthetic_tensors(), label)
        return
    model_dir, label = args[0], (args[1] if len(args) > 1 else os.path.basename(args[0].rstrip("/")))
    log(f"# SUBBIT-0 model={model_dir} label={label} dev={DEV} dtype={DTYPE}")
    run(_iter_model_tensors(model_dir), label)


if __name__ == "__main__":
    main()
