#!/usr/bin/env python3.12
"""kv.py — the long-context KV lane, two subcommands (was kv_frontier.py + kv_hybrid.py).

  kv.py frontier <model_dir> [label] [params_b] | kv.py frontier --synthetic [label] [params_b]
  kv.py hybrid   <model_dir> [label] [params_b] | kv.py hybrid   --synthetic [label] [params_b]

frontier: the AGGRESSIVE long-context lane, pushing the usable window as far as the KV cache can be
  compressed/evicted/paged, in four HONEST regimes (R1 LOSSLESS-RAM full attention + quantized KV /
  R2 LOSSLESS-SSD cold KV paged / R3 LOSSY-EVICT fixed live set / R4 SSM O(1) state). Pure projection
  + NIAH gate, no serving-win claim. The context wall at long range is the KV cache, not the weights;
  report the regime tag on every number and never call a lossy policy full context.
hybrid: STKV (Strand-Tiered KV), the Hawking-specific hybrid that spends each KV asset where it is
  strongest (Tier0 sink+recent int8 exact / Tier1 warm STRAND-trellis / Tier2 cold SSD-page or RWKV-7
  SSM), so recent context is EXACT (NIAH must pass) and distant context is unbounded (lossy, reported
  as such). Research scaffold + projection; the tiered serve kernel is the Rust serve build.
"""
import sys, os, json

OUT = "reports/condense"


def kv_bpt(g, bits):
    return 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * (bits / 8.0)


# ============================ frontier (was kv_frontier.py) ============================

def _geom_frontier(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0].lower()
    nh = c.get("num_attention_heads", 32)
    return {"arch": arch, "is_ssm": any(t in arch for t in ("rwkv", "mamba", "ssm")),
            "layers": c.get("num_hidden_layers", 28), "kv_heads": c.get("num_key_value_heads", nh),
            "head_dim": c.get("head_dim") or c.get("hidden_size", 4096) // max(1, nh),
            "params_b": None, "trained_ctx": c.get("max_position_embeddings", 32768)}


def run_frontier(model_dir, label, params_b, ram_gb=96.0, ssd_tb=2.0, synthetic=False):
    if synthetic:
        g = {"arch": "qwen2", "is_ssm": False, "layers": 80, "kv_heads": 8, "head_dim": 128,
             "trained_ctx": 1_000_000}
        params_b = params_b or 70.0
    else:
        g = _geom_frontier(model_dir)
    w2 = params_b * 2 / 8.0                       # 2-bit weights
    free = (ram_gb - w2) * 1e9
    def ctx(bits): return free / kv_bpt(g, bits)
    regimes = {
        "R1_lossless_ram": {
            "int4_Mtok": round(ctx(4) / 1e6, 2), "int2_Mtok": round(ctx(2) / 1e6, 2),
            "trellis1p5_Mtok": round(ctx(1.5) / 1e6, 2),
            "note": "full attention in RAM; int2/trellis NIAH-gated", "loss": "lossless"},
        "R2_lossless_ssd": {
            "int2_Mtok": round((ssd_tb * 1e12) / kv_bpt(g, 2) / 1e6, 0),
            "note": "cold KV paged to SSD; storage-bound, slow tail", "loss": "lossless-slow"},
        "R3_lossy_evict": {
            "effective": "unbounded", "note": "attention-sink/heavy-hitter; evicted middle GONE",
            "loss": "lossy-evicted"},
        "R4_ssm": {
            "effective": "unbounded" if g["is_ssm"] else "n/a (dense; distill->RWKV-7 to unlock)",
            "note": "O(1) state, flat memory, no KV cache", "loss": "lossy-summary"},
    }
    rec = {"model": label, "params_b": params_b, "arch": g["arch"], "is_ssm": g["is_ssm"],
           "ram_gb": ram_gb, "ssd_tb": ssd_tb, "weights_2bit_gb": round(w2, 1),
           "kv_bytes_per_token_f16": int(kv_bpt(g, 16)), "regimes": regimes, "probe": True,
           "usable_window": "min(regime ceiling, model trained/NIAH-validated window)",
           "serve_gated_on": "KV-quant + paging + (R4) SSM serve kernels — the Rust serve build"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{label}_kvfrontier.json", "w"), indent=2)
    r1 = regimes["R1_lossless_ram"]
    print(f"[kv] {label} ({params_b}B, 2-bit W {w2:.1f}GB, {ram_gb:.0f}GB RAM):", file=sys.stderr)
    print(f"  R1 full-attn RAM : int4 {r1['int4_Mtok']}M | int2 {r1['int2_Mtok']}M | "
          f"trellis~1.5b {r1['trellis1p5_Mtok']}M tok  [lossless, NIAH-gated]", file=sys.stderr)
    print(f"  R2 SSD-paged     : {regimes['R2_lossless_ssd']['int2_Mtok']:.0f}M tok  "
          f"[lossless full ctx, SSD-slow tail — the 'bound by your disk' regime]", file=sys.stderr)
    print(f"  R3 evict/sink    : unbounded EFFECTIVE  [LOSSY — evicted middle gone, NIAH there fails]", file=sys.stderr)
    print(f"  R4 RWKV-7 SSM    : {regimes['R4_ssm']['effective']}  [flat O(1) state]", file=sys.stderr)
    print("# KILL: if int2/trellis KV or an eviction policy fails NIAH at the target ctx, that lever "
          "does not deliver FULL context there (fall back to int4 or SSD-paged).", file=sys.stderr)
    return 0


def _main_frontier():
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--synthetic":
        run_frontier(None, sys.argv[2] if len(sys.argv) > 2 else "synth70b",
            float(sys.argv[3]) if len(sys.argv) > 3 else 70.0, synthetic=True)
    elif a == "--help":
        print(__doc__)
    else:
        run_frontier(a, sys.argv[2] if len(sys.argv) > 2 else "model",
            float(sys.argv[3]) if len(sys.argv) > 3 else 7.0)


# ============================ hybrid (was kv_hybrid.py) ============================

def _geom_hybrid(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0].lower()
    nh = c.get("num_attention_heads", 32)
    return {"arch": arch, "layers": c.get("num_hidden_layers", 28),
            "kv_heads": c.get("num_key_value_heads", nh),
            "head_dim": c.get("head_dim") or c.get("hidden_size", 4096) // max(1, nh)}


def run_hybrid(model_dir, label, params_b, ram_gb=96.0, ssd_tb=2.0, sink=4096, recent=131072, synthetic=False):
    g = {"layers": 80, "kv_heads": 8, "head_dim": 128} if synthetic else _geom_hybrid(model_dir)
    if synthetic and not params_b:
        params_b = 70.0
    w2 = params_b * 2 / 8.0
    # Tier 0: sinks + recent window at int8 (exact). Tier 1: the rest of RAM at trellis ~1.5b.
    t0_tokens = sink + recent
    t0_gb = kv_bpt(g, 8) * t0_tokens / 1e9
    ram_for_warm = max(0.0, ram_gb - w2 - t0_gb)
    t1_tokens = ram_for_warm * 1e9 / kv_bpt(g, 1.5)            # trellis-KV warm band
    exact_window = t0_tokens + t1_tokens                       # full-attention, lossless, in RAM
    ssd_tail = (ssd_tb * 1e12) / kv_bpt(g, 1.5)                # Tier 2(a): lossless, SSD-paged
    rec = {
        "model": label, "params_b": params_b, "hybrid": "STKV (Strand-Tiered KV)",
        "ram_gb": ram_gb, "weights_2bit_gb": round(w2, 1),
        "tier0_sink_recent": {"tokens": t0_tokens, "kv": "int8 exact", "ram_gb": round(t0_gb, 2)},
        "tier1_warm_trellis": {"tokens_M": round(t1_tokens / 1e6, 2), "kv": "trellis ~1.5b",
                               "ram_gb": round(ram_for_warm, 1)},
        "exact_window_Mtok": round(exact_window / 1e6, 2),     # lossless full attention, in RAM
        "tier2a_ssd_lossless_Mtok": round(ssd_tail / 1e6, 0),  # lossless, SSD-slow tail
        "tier2b_ssm_tail": "UNBOUNDED effective (RWKV-7 O(1) state; lossy gist)",
        "uses": ["STRAND trellis codec (Tier1)", "sparse-outlier wire (Tier0 sink)",
                 "RWKV-7 SSM (Tier2b)", "SSD paging (Tier2a)"],
        "why_hawking_only": "needs BOTH the dense-attention path AND the RWKV-7 SSM path in one "
                            "engine + the STRAND codec on the cache — no commodity stack has all three",
        "niah_gate": "Tier0/1 must hold NIAH (lossless); Tier2b SSM tail is lossy, measured not assumed",
        "serve_gated_on": "the tiered serve kernel (int8 sink + trellis-KV + SSD pager + SSM tail)",
        "probe": True,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{label}_kvhybrid.json", "w"), indent=2)
    print(f"[stkv] {label} ({params_b}B, 2-bit W {w2:.1f}GB, {ram_gb:.0f}GB RAM, sink {sink} + recent {recent}):",
          file=sys.stderr)
    print(f"  Tier0 sink+recent (int8 exact): {t0_tokens/1e3:.0f}k tok, {t0_gb:.1f}GB", file=sys.stderr)
    print(f"  Tier1 warm (trellis ~1.5b)    : {t1_tokens/1e6:.2f}M tok in {ram_for_warm:.0f}GB", file=sys.stderr)
    print(f"  => EXACT window (lossless RAM) : {exact_window/1e6:.2f}M tok  [NIAH-gated]", file=sys.stderr)
    print(f"  Tier2a SSD-paged (lossless)    : {ssd_tail/1e6:.0f}M tok  [slow tail]", file=sys.stderr)
    print(f"  Tier2b RWKV-7 SSM tail         : UNBOUNDED effective  [lossy gist, flat memory]", file=sys.stderr)
    print(f"# Hawking-only: combines STRAND-trellis-KV + RWKV-7 SSM + outlier-sink + SSD-page in one "
          f"tiered policy. KILL: if Tier0/1 trellis fails NIAH, fall back to int4 warm (smaller exact window).",
          file=sys.stderr)
    return 0


def _main_hybrid():
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--synthetic":
        run_hybrid(None, sys.argv[2] if len(sys.argv) > 2 else "stkv70b",
            float(sys.argv[3]) if len(sys.argv) > 3 else 70.0, synthetic=True)
    elif a == "--help":
        print(__doc__)
    else:
        run_hybrid(a, sys.argv[2] if len(sys.argv) > 2 else "model",
            float(sys.argv[3]) if len(sys.argv) > 3 else 7.0)


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if sub == "frontier":
        sys.argv = ["kv_frontier.py"] + sys.argv[2:]
        _main_frontier()
    elif sub == "hybrid":
        sys.argv = ["kv_hybrid.py"] + sys.argv[2:]
        _main_hybrid()
    else:
        print(__doc__)
