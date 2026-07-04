#!/usr/bin/env python3.12
"""subbit.py - merged tool: measure (was subbit_measure.py) + ladder (was subbit_ladder.py) + admm (was subbit_admm.py).

the sub-1-bit SUBBIT lane: measure (SUBBIT-0 entropy floor per model), ladder (sub-1-bit config ladder), admm (NanoQuant ADMM low-rank probe, KILLed on real qwen-05b - do not iterate).

  subbit.py measure <args...>   # was: python3.12 tools/condense/subbit_measure.py <args...>
  subbit.py ladder <args...>   # was: python3.12 tools/condense/subbit_ladder.py <args...>
  subbit.py admm <args...>   # was: python3.12 tools/condense/subbit_admm.py <args...>
"""
import sys

def _run_measure():
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

    main()



def _run_ladder():
    """subbit_ladder.py — the SUB-1-BIT rung math + feasibility table (a PROBE, not a serve win).

    The sibling of tools/condense/ladder.py. Where ladder.py stops at the 1.34-bpw "1-bit" rung
    (the lowest rung with a BUILT GPU bitslice .tq serve path), THIS file extends the rung set DOWN
    into the sub-1-bit territory that the studio_maximization "crazy ladder" sketches — 1.00, 0.75,
    0.50, 0.33 effective bpw — and encodes, as runnable code, the feasibility classes that decide
    which (model, eff-bpw) cells are PRODUCT, RESEARCH, FANTASY, or physically BELOW-FLOOR.

    WHAT THIS TOOL IS: a size/fit/feasibility CALCULATOR. Given any (params, eff-bpw) it computes the
    artifact size, whether it serves on an 84 GB weight budget, and its lane. It does NOT bake, doctor,
    measure ppl, or claim a serving win — the serve paths below 1.34 bpw are UNBUILT and the recovery
    at these rungs is UNPROVEN. The "PRODUCT" lane means "the math + a built codec rung exist", NOT
    "this artifact ships at near-1:1 quality today". Treat every sub-1-bit number as a PROBE.

    EFFECTIVE BPW DISCIPLINE (matches ladder.py / audit_ladder.py): rungs here are EFFECTIVE bpw,
    i.e. they already include side-info (trellis scales + outlier-channel positions + residual-pass
    overhead — the baker's AGGREGATE number). We never report a nominal payload bpw. The hard physical
    fact this tool enforces: a DENSE codec cannot encode a weight matrix below the side-info floor
    (~0.28 eff-bpw measured ~0.25-0.31), because the scales/positions ALONE cost that many bits even
    if every weight were free. A dense rung below that floor is BELOW-FLOOR = impossible. A MoE model's
    AMORTIZED-over-total bpw MAY sit below the floor only because the active experts carry real bits and
    the dormant experts dilute the average — the active slice itself is still floor-bound.

    KILL LINE (the criterion that refutes the sub-1-bit lever):
      If, on a 7B+ model, the lowest EFFECTIVE-bpw config that holds <=+2% ppl (multiwindow, after the
      full doctor stack) lands at or ABOVE ~1.34 bpw — i.e. no measured config ever crosses the 1.34
      line at near-1:1 — then every rung below 1.34 here is FANTASY/RESEARCH theater and the sub-1-bit
      ladder is DEAD as a product axis. This file computes sizes; scaling_law.py --fit decides the kill.

    CLI:
      python subbit_ladder.py                 # summary: rungs x models, lane per cell
      python subbit_ladder.py --tsv           # flat table (model, params, active, per-rung gb + class + lane)
      python subbit_ladder.py --fit <p_b> [--active A]   # footprint per rung + the bpw that just fits 84/70/30GB
      python subbit_ladder.py --dream         # the headline callouts (671B@1.0, 744B@0.33, 235B-A22B@1.34)
      python subbit_ladder.py --selftest      # run --tsv + --dream and assert the anchor numbers
    """
    import sys

    # ── hardware envelope (matches ladder.py) ─────────────────────────────────────────────
    WEIGHT_BUDGET = 84.0          # serve weight budget on the 96 GB box (leave headroom for KV/acts/OS)
    SERVE_COMFY   = 70.0          # <= this = comfortable (room for long-ctx KV)
    SERVE_DREAM   = 30.0          # the "fits on a 32 GB laptop class" callout threshold

    # ── the SUB-1-BIT ladder (EFFECTIVE bpw, side-info included) ───────────────────────────
    # 4.50/3.34/2.34/1.34 are ladder.py's built rungs (BPW dict); below 1.34 is the new frontier.
    # 1.00 = the absolute edge that makes 671B fit 84 GB. 0.33 ~= ternary's MoE-amortized dream.
    RUNGS = [4.50, 3.34, 2.34, 1.34, 1.00, 0.75, 0.50, 0.33]

    # rungs at/above this have a BUILT GPU bitslice .tq serve path (ladder.py serves()=single-bake)
    BUILT_SERVE_FLOOR = 1.34

    # ── the side-info floor — the hard physical wall for DENSE codecs ──────────────────────
    # Measured aggregate side-info (trellis scales + outlier positions + residual overhead) is
    # ~0.25-0.31 eff-bpw across the 7B bakes; we take ~0.28 as the floor. A DENSE rung below this is
    # physically impossible: the scales/positions cost this many bits even at zero weight payload.
    SIDE_INFO_FLOOR = 0.28
    SIDE_INFO_RANGE = (0.25, 0.31)

    # ── the quality gate (echoed from scaling_law.GATE; this file doesn't measure, only cites) ─
    GATE_PCT = 2.0                # <= +2% ppl vs f16 parent = the ~1:1 "floor held" bar

    # ── the models (footprint = TOTAL params; tps ~ ACTIVE params for MoE) ─────────────────
    # (family, name, total_b, active_b or None for dense, note)
    def M(family, name, total_b, active_b=None, note=""):
        return dict(family=family, name=name, total_b=total_b, active_b=active_b, note=note)

    MODELS = [
        M("qwen2.5", "Qwen2.5-32B",      32.5,  None, "dense; unconstrained on the box"),
        M("qwen2.5", "Qwen2.5-72B",      72.7,  None, "dense; serve-tight even at 4.5 bpw"),
        M("llama3",  "Llama3.3-70B",     70.6,  None, "dense; the cross-family 70B point"),
        M("qwen3",   "Qwen3-235B-A22B",  235.0, 22.0, "MoE: 235B footprint, decodes like ~22B"),
        M("llama3",  "Llama3.1-405B",    405.0, None, "dense FRONTIER: needs <=1.0 bpw to even fit"),
        M("deepseek","DeepSeek-V3",      671.0, 37.0, "MoE: 671B footprint @1.0 ~= 84GB = box edge; active 37B"),
        M("glm",     "GLM-744B",         744.0, 32.0, "MoE: largest tail; active ~32B; the 0.33 dream"),
    ]


    # ── size math (artifact_gb matches ladder.py tq_gb: params * bpw / 8) ──────────────────
    def tq_gb(params_b, bpw):
        return params_b * bpw / 8.0


    def serve_class(gb):
        """SERVE-COMFY (<=70GB) / SERVE-TIGHT (70-84) / SERVE-OVERFLOW (>84)."""
        if gb <= SERVE_COMFY:
            return "SERVE-COMFY"
        if gb <= WEIGHT_BUDGET:
            return "SERVE-TIGHT"
        return "SERVE-OVERFLOW"


    def bpw_that_fits(params_b, budget):
        """The HIGHEST EFFECTIVE bpw at which `params_b` still fits `budget` GB (params*bpw/8<=budget).
        Inverse of tq_gb: bpw <= 8*budget/params. None if even the smallest rung overflows."""
        cap = 8.0 * budget / params_b
        for r in RUNGS:                       # rungs are descending; first <= cap is the highest that fits
            if r <= cap:
                return r
        return None                           # even 0.33 overflows -> nothing fits this budget


    def below_floor(bpw, is_moe):
        """A rung is BELOW-FLOOR (physically impossible) when it's under the dense side-info floor AND
        the model is dense. A MoE may amortize below the floor (active experts carry the bits, dormant
        experts dilute the per-total average) — so MoE is NOT auto-killed by the dense floor."""
        return (bpw < SIDE_INFO_FLOOR) and not is_moe


    # ── the lane verdict (the studio_maximization "crazy ladder" classification) ───────────
    # Index each sub-1-bit rung as SUBBIT-N by descending bpw (the studio enumeration):
    #   SUBBIT-0 = 1.34 (built serve), 1 = 1.00, 2 = 0.75, 3 = 0.50, 4 = 0.33  ... plus the studio's
    #   extended-research rungs. The studio verdict: SUBBIT-0/1/5 are PRODUCT-lane (a built/near-built
    #   serve rung + a plausible recovery story), SUBBIT-2/3/4 are RESEARCH-lane (the MDL prune+quant /
    #   codec-native frontier — unbuilt, unproven), SUBBIT-6/7 (and ANY dense rung under the floor) are
    #   FANTASY-lane (below where any measured config has reached, or physically impossible).
    # Concretely on THIS rung set:
    SUBBIT_INDEX = {1.34: 0, 1.00: 1, 0.75: 2, 0.50: 3, 0.33: 4}
    PRODUCT_SUBBIT  = {0, 1, 5}
    RESEARCH_SUBBIT = {2, 3, 4}
    FANTASY_SUBBIT  = {6, 7}


    def lane(bpw, is_moe):
        """PRODUCT / RESEARCH / FANTASY / BELOW-FLOOR for a (rung, model-kind) cell.

        BELOW-FLOOR overrides everything for a DENSE rung under the side-info floor (impossible).
        Rungs at/above 1.34 are PRODUCT (the built/serve tier). Sub-1-bit rungs map by SUBBIT index
        to the studio lanes. A MoE whose per-total rung dips under the floor stays RESEARCH/FANTASY
        by its index (it is not impossible, just unproven) rather than BELOW-FLOOR."""
        if below_floor(bpw, is_moe):
            return "BELOW-FLOOR"
        if bpw >= BUILT_SERVE_FLOOR:          # 1.34, 2.34, 3.34, 4.50 — the built serve rungs
            return "PRODUCT"
        idx = SUBBIT_INDEX.get(bpw)
        if idx is None:
            return "RESEARCH"                 # an off-grid sub-1-bit rung defaults to research
        if idx in PRODUCT_SUBBIT:
            return "PRODUCT"
        if idx in RESEARCH_SUBBIT:
            return "RESEARCH"
        return "FANTASY"


    def serves_today(bpw):
        """Honest: is there a BUILT .tq serve path at this rung? (>=1.34 only.) Everything below is
        a probe — the sub-1-bit serve kernels are UNBUILT, this tool measures size, not a serve win."""
        return bpw >= BUILT_SERVE_FLOOR


    def is_moe(model):
        return model["active_b"] is not None


    # ── per-model cells ────────────────────────────────────────────────────────────────────
    def cells(model):
        p = model["total_b"]
        moe = is_moe(model)
        out = []
        for r in RUNGS:
            gb = tq_gb(p, r)
            out.append(dict(bpw=r, gb=round(gb, 1), serve=serve_class(gb),
                            lane=lane(r, moe), serves_today=serves_today(r)))
        return out


    def _moe_tag(model):
        return f"MoE act {model['active_b']}B" if is_moe(model) else "dense"


    # ── CLI handlers ───────────────────────────────────────────────────────────────────────
    def cmd_tsv():
        cols = "\t".join(f"gb@{r}" for r in RUNGS)
        lanes = "\t".join(f"lane@{r}" for r in RUNGS)
        print(f"family\tmodel\ttotal_b\tactive_b\tkind\t{cols}\tfit84\tfit70\tfit30\t{lanes}")
        for m in MODELS:
            p = m["total_b"]
            gbs = "\t".join(f"{tq_gb(p, r):.1f}" for r in RUNGS)
            lns = "\t".join(lane(r, is_moe(m)) for r in RUNGS)
            f84 = bpw_that_fits(p, WEIGHT_BUDGET)
            f70 = bpw_that_fits(p, SERVE_COMFY)
            f30 = bpw_that_fits(p, SERVE_DREAM)
            print(f"{m['family']}\t{m['name']}\t{p}\t{m['active_b'] or ''}\t{_moe_tag(m)}\t{gbs}\t"
                  f"{f84 if f84 else 'none'}\t{f70 if f70 else 'none'}\t{f30 if f30 else 'none'}\t{lns}")


    def cmd_fit(params_b, active_b=None):
        moe = active_b is not None
        print(f"# subbit footprint for {params_b}B "
              f"({'MoE active '+str(active_b)+'B' if moe else 'dense'}), "
              f"weight budget {WEIGHT_BUDGET:.0f} GB  [PROBE — sub-1.34-bpw serve is UNBUILT]")
        print(f"# artifact_gb = total_params * eff_bpw / 8 ; footprint=total, tps~active")
        for r in RUNGS:
            gb = tq_gb(params_b, r)
            bf = "  BELOW-FLOOR(dense impossible)" if below_floor(r, moe) else ""
            print(f"  {r:4.2f} bpw: {gb:7.1f} GB  {serve_class(gb):13s} {lane(r, moe):11s}"
                  f"{'' if serves_today(r) else ' [no built serve path]'}{bf}")
        for budget, tag in ((WEIGHT_BUDGET, "84GB box"), (SERVE_COMFY, "70GB comfy"), (SERVE_DREAM, "30GB laptop")):
            b = bpw_that_fits(params_b, budget)
            if b is None:
                print(f"  just-fits {tag:12s}: none on this rung set (even {RUNGS[-1]} bpw overflows)")
            else:
                print(f"  just-fits {tag:12s}: <= {b:.2f} eff-bpw  ({tq_gb(params_b, b):.1f} GB){'' if serves_today(b) else '  [probe rung]'}")


    def cmd_dream():
        print("# Headline sub-1-bit callouts (PROBE math — sizes only, no serve-win claim):")
        callouts = [
            ("DeepSeek-V3 671B",      671.0, 1.00, 37.0),
            ("GLM-744B",              744.0, 0.33, 32.0),
            ("Qwen3-235B-A22B",       235.0, 1.34, 22.0),
        ]
        for name, p, bpw, act in callouts:
            gb = tq_gb(p, bpw)
            moe = act is not None
            print(f"  {name:22s} @ {bpw:.2f} eff-bpw = {gb:5.1f} GB  "
                  f"{serve_class(gb):13s} {lane(bpw, moe):11s}  (MoE active {act}B)"
                  f"{'' if serves_today(bpw) else '  [serve path UNBUILT]'}")
        print(f"# Side-info floor: dense codec cannot go below ~{SIDE_INFO_FLOOR} eff-bpw "
              f"(measured {SIDE_INFO_RANGE[0]}-{SIDE_INFO_RANGE[1]}); MoE may amortize lower (active experts carry the bits).")
        print(f"# KILL: if no 7B+ config holds <=+{GATE_PCT}% ppl below {BUILT_SERVE_FLOOR} bpw, every sub-1-bit rung is fantasy.")


    def cmd_summary():
        print(f"# subbit_ladder — sub-1-bit rung math (PROBE; sizes/fit/lane, NOT a serve-win claim)")
        print(f"# budget {WEIGHT_BUDGET:.0f}GB · built serve floor {BUILT_SERVE_FLOOR} bpw · "
              f"dense side-info floor ~{SIDE_INFO_FLOOR} bpw (measured {SIDE_INFO_RANGE[0]}-{SIDE_INFO_RANGE[1]})")
        print(f"# lanes: PRODUCT(built/near) · RESEARCH(unbuilt frontier) · FANTASY(beyond reach) · "
              f"BELOW-FLOOR(dense impossible)")
        hdr = "model".ljust(20) + "  total  " + "  ".join(f"{r:>5.2f}" for r in RUNGS)
        print("\n" + hdr)
        print("-" * len(hdr))
        for m in MODELS:
            p = m["total_b"]
            row = m["name"].ljust(20) + f"  {p:5.0f}B "
            marks = []
            for r in RUNGS:
                gb = tq_gb(p, r)
                ln = lane(r, is_moe(m))
                sym = {"PRODUCT": " P ", "RESEARCH": " R ", "FANTASY": " F ", "BELOW-FLOOR": " x "}[ln]
                # mark serve-overflow rungs with a trailing '!'
                sym = sym if gb <= WEIGHT_BUDGET else sym.rstrip() + "!"
                marks.append(f"{sym:>5s}")
            print(row + "  ".join(marks))
        print("\n# legend: P=product R=research F=fantasy x=below-floor  '!'=overflows 84GB")
        print("# fit-on-84GB (highest eff-bpw that serves):")
        for m in MODELS:
            b = bpw_that_fits(m["total_b"], WEIGHT_BUDGET)
            print(f"    {m['name']:20s} <= {b if b else 'none':>4} bpw   ({_moe_tag(m)})")
        print(f"\n# KILL: if no 7B+ config holds <=+{GATE_PCT}% ppl below {BUILT_SERVE_FLOOR} bpw "
              f"(scaling_law.py --fit), the whole sub-1-bit ladder is dead.")


    def cmd_selftest():
        """Synthetic, runs entirely here (touches no model). Asserts the anchor numbers from the
        prompt: 671@1.0 ~ 83.9GB, 744@0.33 ~ 30.7GB, 235@1.34 ~ 39GB, plus lane/floor invariants."""
        ok = True
        def check(name, got, want, tol=0.15):
            nonlocal ok
            good = abs(got - want) <= tol
            ok = ok and good
            print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.2f}, want ~{want:.2f}")

        check("671B @ 1.00 bpw GB", tq_gb(671.0, 1.00), 83.9)
        check("744B @ 0.33 bpw GB", tq_gb(744.0, 0.33), 30.7)
        check("235B @ 1.34 bpw GB", tq_gb(235.0, 1.34), 39.4, tol=0.5)

        # lane invariants
        inv = [
            ("671B@1.00 serves the box (TIGHT)", serve_class(tq_gb(671.0, 1.00)) == "SERVE-TIGHT"),
            ("671B@1.00 is PRODUCT (SUBBIT-1)", lane(1.00, True) == "PRODUCT"),
            ("0.75 dense rung is RESEARCH", lane(0.75, False) == "RESEARCH"),
            ("0.50 dense rung is RESEARCH", lane(0.50, False) == "RESEARCH"),
            ("0.33 rung NOT below floor (>0.28)", not below_floor(0.33, False)),
            ("0.20 dense rung is BELOW-FLOOR", lane(0.20, False) == "BELOW-FLOOR"),
            ("0.20 MoE rung is NOT below-floor", lane(0.20, True) != "BELOW-FLOOR"),
            ("1.34 is the built serve floor", serves_today(1.34) and not serves_today(1.00)),
            ("405B needs <=1.34 to fit 84GB", (bpw_that_fits(405.0, WEIGHT_BUDGET) or 9) <= 1.34),
            ("32B fits at top rung 4.50", bpw_that_fits(32.5, WEIGHT_BUDGET) == 4.50),
        ]
        for name, cond in inv:
            ok = ok and cond
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

        print(f"\n# --tsv smoke + --dream smoke:")
        cmd_tsv()
        print()
        cmd_dream()
        print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
        return ok


    def main():
        arg = sys.argv[1] if len(sys.argv) > 1 else ""
        if arg in ("-h", "--help"):
            print(__doc__)
            return
        if arg == "--tsv":
            cmd_tsv()
            return
        if arg == "--fit":
            p = float(sys.argv[2])
            act = None
            if "--active" in sys.argv:
                act = float(sys.argv[sys.argv.index("--active") + 1])
            cmd_fit(p, act)
            return
        if arg == "--dream":
            cmd_dream()
            return
        if arg == "--selftest":
            sys.exit(0 if cmd_selftest() else 1)
        cmd_summary()

    main()



def _run_admm():
    """subbit_admm.py — SUBBIT-2: NanoQuant-style low-rank BINARY factorization via ADMM (PROBE).

    WHAT THIS IS (and is NOT). This is a BOUNDED research PROBE, not a serving codec. It MEASURES
    whether a sub-bit binary low-rank approximation generalizes; it makes NO serving-win claim and
    emits no deployable artifact. It exists to RE-TEST a named resurrection risk in the kill ledger:
    post-hoc low-rank (the ASVD / data-aware-SVD family, and L2_lowrank_heal_NOGO) is DEAD because
    its held-out functional error runs ~2x its in-sample error — it overfits the calibration slice.
    NanoQuant is the same shape dressed as binary factors, so it inherits the same kill risk.

    THE LEVER. Approximate a weight matrix W (rows x cols) as

            W  ~=  s * (B1 @ B2)        with  B1 in {-1,+1}^(rows x r),  B2 in {-1,+1}^(r x cols)

    solved by ADMM: alternate (a) binarize each factor by sign with a least-squares scale, (b) a
    dual/residual update that pulls the relaxed real factors back toward {-1,+1}. r is the rank arg.

    EFFECTIVE BPW (honest, side-info included; never nominal). The binary factors cost r*(rows+cols)
    sign bits; the per-matrix scale s is one fp32 number => SCALE_BITS. So

            eff_bpw = ( r*(rows+cols)*1  +  SCALE_BITS ) / (rows*cols)

    We then build the PLAIN-RESIDUAL baseline (two passes of sign+per-row-scale binary quant, b1+b2)
    and TRUNCATE its rank so its effective bpw MATCHES the ADMM probe's eff_bpw — a fair shoot-out at
    identical storage. Both are scored on the SAME held-out activations.

    FUNCTIONAL ERROR (output space, the only honest metric). For input activations X (captured from
    the real model, or synthetic), the error of a reconstruction W_hat is

            err = || (W - W_hat) @ X^T ||_F  /  || W @ X^T ||_F

    measured on a FIT split (the ADMM fit it) and a HELD-OUT split (it did NOT). Held-out is the verdict.

    THE KILL (the criterion that REFUTES the lever — printed in --help and in the JSON):
      KILL if  heldout_err / fit_err > 1.5            (the dead ASVD overfit signature), OR
      KILL if  admm_heldout >= residual_heldout       (loses to plain residual at matched eff bpw).
    On KILL: print  'KILL: NanoQuant is a low-rank resurrection'  and exit 1.
    Only a probe that BOTH generalizes (ratio <= 1.5) AND beats matched-bpw residual on held-out
    survives (exit 0) — and even then it is a probe result, not a serving win.

    Env (matches the condense tools): DOCTOR_DEVICE (cpu/mps), DOCTOR_DTYPE (bfloat16 for 7B+ CPU),
    STRAND_NO_GPU=1 (honored — keeps Metal idle). Stdlib + torch + safetensors only.

    Usage:
      python3.12 tools/condense/subbit_admm.py [--model DIR] [--tensor NAME] [--rank R]
                                               [--iters N] [--calib FILE] [--ctx T]
                                               [--out reports/condense/subbit2_admm.json]
      python3.12 tools/condense/subbit_admm.py --self-test     # synthetic low-rank+noise; checks kill logic
      python3.12 tools/condense/subbit_admm.py --help

    Defaults: model = scratch/qwen-05b (fits 18GB), tensor = model.layers.0.mlp.down_proj.weight.
    If the model is absent, a SYNTHETIC structured matrix is used and the path is gated (--dry runs here).
    """
    import sys, os, json, math, argparse

    # Honor the repo's no-GPU contract before importing torch backends do anything heavy.
    os.environ.setdefault("STRAND_NO_GPU", "1")
    import torch

    SCALE_BITS = 32                       # bits per fp32 scale; a per-ROW scale vector is counted below
    DEFAULT_MODEL = "scratch/qwen-05b"
    DEFAULT_TENSOR = "model.layers.0.mlp.down_proj.weight"
    DEFAULT_OUT = "reports/condense/subbit2_admm.json"
    KILL_LINE = "KILL: NanoQuant is a low-rank resurrection"
    OVERFIT_RATIO = 1.5                   # heldout/fit ratio above which the lever is the dead ASVD signature


    def log(m):
        print(m, file=sys.stderr); sys.stderr.flush()


    def _device():
        dev = os.environ.get("DOCTOR_DEVICE")
        if dev:
            return dev
        return "mps" if torch.backends.mps.is_available() else "cpu"


    def _dtype():
        return getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))


    # ----------------------------------------------------------------------------------------------
    # effective-bpw accounting (side-info included; NEVER nominal)
    # ----------------------------------------------------------------------------------------------
    def eff_bpw_lowrank(rows, cols, rank, scale_bits=SCALE_BITS):
        """Binary low-rank W ~= B1 @ diag(a) @ B2: r*(rows+cols) sign bits + r per-COMPONENT fp32
        scales (the a diagonal), over rows*cols weights. Both are side-info and counted (never
        nominal)."""
        return (rank * (rows + cols) * 1 + rank * scale_bits) / (rows * cols)


    def eff_bpw_residual(rows, cols, r1, r2, scale_bits=SCALE_BITS):
        """Plain 2-pass binary residual: ranks r1 + r2 of sign bits plus (r1+r2) per-component fp32
        scales. Side-info fully counted. Same per-component scale cost as the low-rank probe, so the
        two are compared at matched total storage (we never hide the scale cost to flatter either)."""
        return ((r1 + r2) * (rows + cols) * 1 + (r1 + r2) * scale_bits) / (rows * cols)


    def residual_ranks_for_bpw(rows, cols, target_bpw, scale_bits=SCALE_BITS):
        """Pick r1,r2 for the 2-pass residual so its eff_bpw <= the low-rank probe's target (a fair,
        if anything generous-to-low-rank, match: the residual gets no MORE storage). Per-component
        sign+scale cost is (rows+cols)+scale_bits bits; the budget buys total rank, split across two
        passes."""
        per_comp_bits = (rows + cols) + scale_bits
        total_rank = max(2, int((target_bpw * rows * cols) // per_comp_bits))
        r1 = max(1, total_rank // 2)
        r2 = max(1, total_rank - r1)
        return r1, r2


    # ----------------------------------------------------------------------------------------------
    # functional (output-space) error: || (W - W_hat) X^T ||_F / || W X^T ||_F
    # ----------------------------------------------------------------------------------------------
    def func_err(W, W_hat, X):
        """X is (n_samples, cols). Returns relative Frobenius error of the layer's output."""
        num = torch.linalg.norm((W - W_hat) @ X.T)
        den = torch.linalg.norm(W @ X.T)
        return float(num / den) if float(den) > 0 else float("inf")


    # ----------------------------------------------------------------------------------------------
    # ADMM binary low-rank: W ~= B1 @ diag(a) @ B2,  B1,B2 in {-1,+1},  a = per-component scale
    # ----------------------------------------------------------------------------------------------
    def _joint_scale_ls(W, B1, B2):
        """Solve a = argmin ||W - B1 diag(a) B2||_F over the per-component scales (a linear LS whose
        design columns are the rank-1 binary outer products vec(b1_k outer b2_k))."""
        rank = B1.shape[1]
        design = torch.stack([torch.outer(B1[:, k], B2[k, :]).reshape(-1) for k in range(rank)], dim=1)
        sol = torch.linalg.lstsq(design, W.reshape(-1)).solution
        return sol


    def admm_binary_lowrank(W, rank, iters=40, rho=1.0, polish=3):
        """Solve W ~= B1 @ diag(a) @ B2 with binary factors, scale-LS, and an ADMM consensus polish.

        A single global (or per-row) scalar CANNOT fit a binary product (its entries grow with rank);
        the working NanoQuant form gives each rank component its own scale a_k. Two stages:

          STAGE 1 - greedy rank-1 binary pursuit (the binarize-via-sign step). For each component,
            alternate u <- sign(R v), v <- sign(R^T u) on the residual R (this IS the discrete factor
            update: sign() is the projection onto {-1,+1}); take the LS scalar a_k = <R, u v^T>/||u v^T||^2;
            subtract a_k u v^T from R; repeat for r components. Monotone, stable.

          STAGE 2 - ADMM consensus polish (the dual-coupled refinement). Treat the joint scales `a` as
            the consensus variable and the per-component signs as the local variables. Each polish sweep:
              (1) a <- joint least-squares over all components            (scale least-squares)
              (2) re-binarize each (u_k, v_k) against its residual target  (sign projection)
              (3) dual feedback: damp the change by rho so the sweep is a contraction (bounded dual;
                  an unclamped per-vector dual diverges for rank-1 LS, so the consensus form is used).
          `iters` drives stage-1 inner alternations; `polish` drives stage-2 sweeps.
        Returns (W_hat, a, B1, B2) where a is the length-rank scale vector (side-info, counted)."""
        rows, cols = W.shape
        R = W.clone()
        a = torch.zeros(rank, dtype=W.dtype, device=W.device)
        B1 = torch.empty(rows, rank, dtype=W.dtype, device=W.device)
        B2 = torch.empty(rank, cols, dtype=W.dtype, device=W.device)
        inner = max(4, iters)
        # ---- STAGE 1: greedy rank-1 binary pursuit ----
        for k in range(rank):
            try:                                        # SVD-seed the sign vectors (faster convergence)
                U, S, Vh = torch.linalg.svd(R, full_matrices=False)
                u = torch.sign(U[:, 0]); v = torch.sign(Vh[0])
            except Exception:
                u = torch.sign(R.sum(1)); v = torch.sign(R.sum(0))
            u[u == 0] = 1.0; v[v == 0] = 1.0
            for _ in range(inner):
                u = torch.sign(R @ v); u[u == 0] = 1.0
                v = torch.sign(R.T @ u); v[v == 0] = 1.0
            ok = torch.outer(u, v)
            denom = float((ok * ok).sum())
            sc = float((R * ok).sum() / denom) if denom > 0 else 0.0
            a[k] = sc; B1[:, k] = u; B2[k, :] = v
            R = R - sc * ok
        # ---- STAGE 2: ADMM consensus polish (bounded dual via rho-damping) ----
        for _ in range(max(0, polish)):
            a = _joint_scale_ls(W, B1, B2)
            Wr = B1 @ torch.diag(a) @ B2
            for k in range(rank):
                ok = torch.outer(B1[:, k], B2[k, :])
                Rk = W - (Wr - a[k] * ok)               # residual seen by component k
                u_new = torch.sign(Rk @ B2[k, :]); u_new[u_new == 0] = 1.0
                v_new = torch.sign(Rk.T @ u_new); v_new[v_new == 0] = 1.0
                # rho-damped consensus update: only flip toward the new sign (contraction, bounded)
                if rho >= 1.0:
                    B1[:, k] = u_new; B2[k, :] = v_new
                else:
                    B1[:, k] = torch.where(torch.rand_like(u_new) < rho, u_new, B1[:, k])
                    B2[k, :] = torch.where(torch.rand_like(v_new) < rho, v_new, B2[k, :])
        a = _joint_scale_ls(W, B1, B2)
        W_hat = B1 @ torch.diag(a) @ B2
        return W_hat, a, B1, B2


    # ----------------------------------------------------------------------------------------------
    # baseline: plain 2-pass binary residual, truncated to the SAME sign-bit budget (matched eff bpw)
    # ----------------------------------------------------------------------------------------------
    def plain_residual_matched(W, r1, r2, iters=40, rho=1.0):
        """The honest baseline at MATCHED (or smaller) storage. Plain residual b1+b2 quant is the lever
        that low-rank must beat (the kill ledger entry). It is two binary-low-rank passes: factorize W
        (rank r1), then factorize the residual W - W_hat1 (rank r2). r1,r2 are chosen by
        residual_ranks_for_bpw so its eff_bpw <= the low-rank probe's. Unlike the probe, it does NOT
        whiten by calibration activations — it is data-FREE, the property the kill ledger says is what
        makes residual robust. If the data-aware single-pass low-rank cannot beat this, the lever is
        dead."""
        What1, _, _, _ = admm_binary_lowrank(W, r1, iters=iters, rho=rho)
        What2, _, _, _ = admm_binary_lowrank(W - What1, r2, iters=iters, rho=rho)
        return What1 + What2


    # ----------------------------------------------------------------------------------------------
    # activation capture (real model) and synthetic fallback
    # ----------------------------------------------------------------------------------------------
    def capture_activations(model_dir, tensor_name, calib_path, ctx, dev, dtype):
        """Forward the calib corpus once, hook the linear whose .weight == tensor_name, collect its
        INPUT activations (rows of x, shape cols). Returns (W, X) with X = (n_samples, cols)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch.nn as nn
        if tensor_name.endswith(".weight"):
            mod_name = tensor_name[:-len(".weight")]
        else:
            mod_name = tensor_name
        log(f"# loading {model_dir} on {dev}/{dtype} to capture activations for {mod_name}")
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, attn_implementation="eager").to(dev).eval()
        modules = dict(model.named_modules())
        target = modules.get(mod_name)
        if not isinstance(target, nn.Linear):
            raise RuntimeError(f"{mod_name} is not an nn.Linear (got {type(target).__name__})")
        W = target.weight.detach().to("cpu", torch.float32).clone()
        bucket = []

        def hook(m, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to("cpu", torch.float32)
            bucket.append(x)

        h = target.register_forward_hook(hook)
        text = open(calib_path, errors="ignore").read() if os.path.exists(calib_path) else ""
        if not text:
            raise FileNotFoundError(f"calib corpus not found: {calib_path}")
        ids = tok(text, return_tensors="pt").input_ids[:, :ctx].to(dev)
        with torch.no_grad():
            model(ids)
        h.remove()
        X = torch.cat(bucket, dim=0)
        del model
        if dev == "mps":
            torch.mps.empty_cache()
        log(f"# captured W{tuple(W.shape)}  X{tuple(X.shape)}")
        return W, X


    def synthetic_problem(rows=256, cols=512, true_rank=8, n_act=2048, noise=0.05, seed=0,
                          cov_shift=False):
        """A structured low-rank + noise matrix and matched activations — the case that SHOULD be
        recoverable. Used when no model is present (--dry) and for --self-test.

        cov_shift=True builds X so its FIRST half and SECOND half have DIFFERENT dominant input
        channels (a calibration->held distribution shift). A data-aware (whitened) fit then overfits
        the first half's directions and its held-out functional error blows up — the exact ASVD
        overfit signature the overfit-ratio kill is meant to catch."""
        g = torch.Generator().manual_seed(seed)
        L = torch.randn(rows, true_rank, generator=g)
        R = torch.randn(true_rank, cols, generator=g)
        W = (L @ R) / math.sqrt(true_rank)
        W = W + noise * torch.randn(rows, cols, generator=g)
        if not cov_shift:
            X = torch.randn(n_act, cols, generator=g)
        else:
            half = n_act // 2
            X = torch.randn(n_act, cols, generator=g) * 0.1
            ndom = max(2, cols // 64)
            idx = torch.randperm(cols, generator=g)
            dom_a, dom_b = idx[:ndom], idx[ndom:2 * ndom]     # disjoint dominant channels per half
            X[:half, dom_a] += torch.randn(half, ndom, generator=g) * 12.0
            X[half:, dom_b] += torch.randn(n_act - half, ndom, generator=g) * 12.0
        return W.float(), X.float()


    # ----------------------------------------------------------------------------------------------
    # core measurement: fit ADMM on FIT split, score on FIT + HELD-OUT, compare to matched residual
    # ----------------------------------------------------------------------------------------------
    def run_probe(W, X, rank, iters, fit_frac=0.5, shuffle=True):
        """Split activations into FIT/HELD-OUT, fit the (data-aware) low-rank on FIT, score on both.
        The ASVD failure mode is that a calibration-aware fit overfits the FIT activations' directions;
        we expose it by scoring functional error on the split the fit was tuned against vs a disjoint
        held-out split. shuffle=True (real-model path) removes token-position bias; shuffle=False keeps
        a deliberately distribution-shifted split intact (used by the self-test)."""
        rows, cols = W.shape
        n = X.shape[0]
        if shuffle:
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
            X = X[perm]
        cut = max(1, int(n * fit_frac))
        X_fit, X_held = X[:cut], X[cut:]
        if X_held.shape[0] == 0:
            X_held = X_fit

        # Low-rank probe: fit factors weighted toward the FIT activations (data-aware, the ASVD shape).
        # We whiten W by the FIT-activation second moment so the factorization is calibration-aware —
        # this is exactly the post-hoc-low-rank move the kill ledger says overfits.
        cov = (X_fit.T @ X_fit) / X_fit.shape[0] + 1e-3 * torch.eye(cols)
        d = torch.sqrt(torch.clamp(torch.diagonal(cov), min=1e-8))     # per-input-channel importance
        W_w = W * d.unsqueeze(0)                                       # weight columns by FIT importance
        What_w, a_scales, B1, B2 = admm_binary_lowrank(W_w, rank, iters=iters)
        What_admm = What_w / d.unsqueeze(0)                            # un-whiten back to weight space

        admm_fit = func_err(W, What_admm, X_fit)
        admm_held = func_err(W, What_admm, X_held)

        bpw_admm = eff_bpw_lowrank(rows, cols, rank)
        # Matched-bpw plain residual baseline (no activation whitening — it is NOT data-aware). Its
        # ranks are sized so its eff_bpw <= the probe's; it never gets more storage than the lever.
        r1, r2 = residual_ranks_for_bpw(rows, cols, bpw_admm)
        What_res = plain_residual_matched(W, r1, r2, iters=iters)
        res_fit = func_err(W, What_res, X_fit)
        res_held = func_err(W, What_res, X_held)
        bpw_res = eff_bpw_residual(rows, cols, r1, r2)

        return {
            "rows": rows, "cols": cols, "rank": rank, "iters": iters,
            "residual_r1": r1, "residual_r2": r2,
            "n_act_fit": int(X_fit.shape[0]), "n_act_held": int(X_held.shape[0]),
            "scale_abs_mean": round(float(a_scales.abs().mean()), 6), "scale_is_per_component": True,
            "eff_bpw_admm": round(bpw_admm, 5),
            "eff_bpw_residual": round(bpw_res, 5),
            "eff_bpw_matched": bpw_res <= bpw_admm + 1e-6,   # residual gets no MORE storage than probe
            "admm_fit_err": round(admm_fit, 5),
            "admm_heldout_err": round(admm_held, 5),
            "residual_fit_err": round(res_fit, 5),
            "residual_heldout_err": round(res_held, 5),
            "heldout_over_fit_ratio": round(admm_held / admm_fit, 4) if admm_fit > 0 else float("inf"),
        }


    def verdict(rec):
        """Apply the hard KILL criteria. Returns (killed: bool, reasons: list[str])."""
        reasons = []
        ratio = rec["heldout_over_fit_ratio"]
        if ratio > OVERFIT_RATIO:
            reasons.append(f"overfit: heldout/fit = {ratio:.2f} > {OVERFIT_RATIO} (dead ASVD signature)")
        if rec["admm_heldout_err"] >= rec["residual_heldout_err"]:
            reasons.append(
                f"loses to residual at matched bpw: admm_heldout {rec['admm_heldout_err']:.4f} "
                f">= residual_heldout {rec['residual_heldout_err']:.4f}")
        return (len(reasons) > 0), reasons


    # ----------------------------------------------------------------------------------------------
    # self-test: run on synthetic, confirm the kill logic FIRES correctly on a forced-overfit case
    # ----------------------------------------------------------------------------------------------
    def self_test():
        log("# --self-test: synthetic low-rank+noise; checking kill logic")
        ok = True

        # Case A: genuinely low-rank, generous rank -> probe should generalize (ratio low). The kill
        # may still fire on the residual-comparison leg (that is fine — it is a real, honest result).
        W, X = synthetic_problem(rows=256, cols=512, true_rank=8, n_act=4096, noise=0.02, seed=0)
        recA = run_probe(W, X, rank=16, iters=40, fit_frac=0.5)
        killedA, reasonsA = verdict(recA)
        log(f"#  [A clean low-rank] ratio={recA['heldout_over_fit_ratio']:.3f} "
            f"admm_held={recA['admm_heldout_err']:.4f} res_held={recA['residual_heldout_err']:.4f} "
            f"killed={killedA} reasons={reasonsA}")
        # The functional error should NOT blow up out-of-sample on a truly low-rank matrix.
        if not (recA["heldout_over_fit_ratio"] <= OVERFIT_RATIO):
            log("#  [A] FAIL: clean low-rank tripped the overfit ratio (kill logic too sensitive)")
            ok = False

        # Case B: forced overfit — a distribution SHIFT between the FIT and HELD activation halves
        # (disjoint dominant input channels). The data-aware (whitened) fit latches onto the FIT
        # half's directions, so its held-out functional error blows up => heldout/fit > 1.5 (the dead
        # ASVD signature). shuffle=False preserves the deliberate first-half/second-half split.
        Wb, Xb = synthetic_problem(rows=256, cols=512, true_rank=200, n_act=512, noise=0.8, seed=3,
                                   cov_shift=True)
        recB = run_probe(Wb, Xb, rank=64, iters=20, fit_frac=0.5, shuffle=False)
        killedB, reasonsB = verdict(recB)
        log(f"#  [B forced overfit] ratio={recB['heldout_over_fit_ratio']:.3f} "
            f"admm_held={recB['admm_heldout_err']:.4f} res_held={recB['residual_heldout_err']:.4f} "
            f"killed={killedB} reasons={reasonsB}")
        if not killedB:
            log("#  [B] FAIL: forced-overfit case did NOT trip the kill (kill logic too lax)")
            ok = False

        log(f"# self-test {'PASS' if ok else 'FAIL'}: kill logic "
            f"{'fires correctly' if ok else 'is MISCALIBRATED'}")
        return 0 if ok else 1


    # ----------------------------------------------------------------------------------------------
    # CLI
    # ----------------------------------------------------------------------------------------------
    def main():
        ap = argparse.ArgumentParser(
            description="SUBBIT-2 probe: binary low-rank ADMM (W~=s*B1@B2) vs matched-bpw residual on "
                        "HELD-OUT functional error. PROBE ONLY — measures, never claims a serving win. "
                        f"{KILL_LINE!r} on overfit (heldout/fit>1.5) or loss to residual.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__)
        ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF dir (default {DEFAULT_MODEL})")
        ap.add_argument("--tensor", default=DEFAULT_TENSOR, help=f"weight tensor (default {DEFAULT_TENSOR})")
        ap.add_argument("--calib", default="scratch/calib_corpus.txt", help="calibration corpus file")
        ap.add_argument("--ctx", type=int, default=1024, help="calib tokens to forward")
        ap.add_argument("--rank", type=int, default=32, help="binary low-rank r")
        ap.add_argument("--iters", type=int, default=40, help="ADMM iterations")
        ap.add_argument("--out", default=DEFAULT_OUT, help=f"JSON report (default {DEFAULT_OUT})")
        ap.add_argument("--dry", action="store_true",
                        help="force the synthetic problem (no model load) — runs anywhere")
        ap.add_argument("--self-test", action="store_true",
                        help="run synthetic self-test and report whether the kill logic fires correctly")
        args = ap.parse_args()

        if args.self_test:
            sys.exit(self_test())

        dev, dtype = _device(), _dtype()
        log(f"# subbit_admm PROBE — dev={dev} dtype={dtype} STRAND_NO_GPU={os.environ.get('STRAND_NO_GPU')}")
        log(f"# KILL criteria: heldout/fit > {OVERFIT_RATIO}  OR  admm_heldout >= residual_heldout (matched bpw)")

        used_synthetic = False
        model_present = os.path.exists(os.path.join(args.model, "model.safetensors")) or \
            os.path.exists(os.path.join(args.model, "model.safetensors.index.json"))
        if args.dry or not model_present:
            if not args.dry:
                log(f"# model {args.model} absent — GATED: falling back to synthetic structured matrix")
            W, X = synthetic_problem()
            used_synthetic = True
            source = "synthetic(low-rank+noise)"
        else:
            W, X = capture_activations(args.model, args.tensor, args.calib, args.ctx, dev, dtype)
            source = f"{args.model}::{args.tensor}"

        rec = run_probe(W, X, rank=args.rank, iters=args.iters)
        rec["source"] = source
        rec["synthetic"] = used_synthetic
        rec["probe"] = True
        rec["disclaimer"] = ("PROBE: measures held-out functional error of binary low-rank vs "
                             "matched-bpw residual. NOT a serving win and emits no deployable artifact.")
        rec["kill_criteria"] = {"overfit_ratio_gt": OVERFIT_RATIO,
                                "or_admm_heldout_ge_residual_heldout": True}

        killed, reasons = verdict(rec)
        rec["killed"] = killed
        rec["kill_reasons"] = reasons

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(json.dumps(rec, indent=2) + "\n")
        log(f"# wrote {args.out}")
        log(f"# eff_bpw(admm)={rec['eff_bpw_admm']:.4f} eff_bpw(residual)={rec['eff_bpw_residual']:.4f} "
            f"matched={rec['eff_bpw_matched']}")
        log(f"# admm:     fit_err={rec['admm_fit_err']:.4f}  heldout_err={rec['admm_heldout_err']:.4f}  "
            f"ratio={rec['heldout_over_fit_ratio']:.3f}")
        log(f"# residual: fit_err={rec['residual_fit_err']:.4f}  heldout_err={rec['residual_heldout_err']:.4f}")

        if killed:
            for r in reasons:
                log(f"#   reason: {r}")
            print(KILL_LINE)
            sys.exit(1)
        log("# SURVIVES (probe only): generalizes AND beats matched-bpw residual on held-out. "
            "Still NOT a serving win — schedule independent reproduction before any claim.")
        sys.exit(0)

    main()


if __name__ == "__main__":
    _sub = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if _sub == "measure":
        sys.argv = ["subbit_measure.py"] + sys.argv[2:]
        _run_measure()
    elif _sub == "ladder":
        sys.argv = ["subbit_ladder.py"] + sys.argv[2:]
        _run_ladder()
    elif _sub == "admm":
        sys.argv = ["subbit_admm.py"] + sys.argv[2:]
        _run_admm()
    else:
        print(__doc__)
