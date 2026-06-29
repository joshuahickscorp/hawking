#!/usr/bin/env python3.12
"""kv_hybrid.py — STKV (Strand-Tiered KV): the Hawking-specific long-context hybrid.

Hawking is the only local engine that ships BOTH a dense-attention Metal path AND an RWKV-7 SSM
path, and BOTH the STRAND trellis codec AND a sparse-outlier wire. STKV consolidates the four KV
levers (trellis-quant / SSD-page / sink-evict / SSM) into ONE tiered policy that spends each asset
where it is strongest, so the recent context is EXACT and the distant context is unbounded:

  Tier 0  SINK + RECENT (exact)   : attention sinks (first s tokens) + the last W tokens, kept at
                                    int8 KV in RAM. Lossless recall where attention actually lands.
  Tier 1  WARM (trellis)          : the middle band, compressed with the STRAND trellis codec on the
                                    KV cache (~1.5 bpw), still in RAM, still full-attention. Hawking's
                                    own codec is the lever no commodity KV-quant (q4/q8) reaches.
  Tier 2  COLD TAIL (choose)      : everything older, EITHER
                                      (a) SSD-paged trellis KV  -> LOSSLESS full context, SSD-slow, or
                                      (b) RWKV-7 SSM summary     -> O(1) state, UNBOUNDED, lossy gist.

The point: exact recall on Tier 0/1 (NIAH must pass there), unbounded reach on Tier 2 (lossy, and
honestly reported as such). It is a research SCAFFOLD + projection here; the tiered serve kernel
(int8 sink + trellis-KV + SSD pager + SSM-tail) is the Rust serve build. Probe only, no serve-win
claim. KILL: if Tier-0/1 trellis KV fails NIAH at the warm-band size, STKV degrades to int4 warm
(smaller exact window) — and if the SSM tail can't summarize past-W recall above chance, Tier 2(b)
is dropped in favor of 2(a) SSD-paging.
"""
import sys, os, json

OUT = "reports/condense"


def _geom(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0].lower()
    nh = c.get("num_attention_heads", 32)
    return {"arch": arch, "layers": c.get("num_hidden_layers", 28),
            "kv_heads": c.get("num_key_value_heads", nh),
            "head_dim": c.get("head_dim") or c.get("hidden_size", 4096) // max(1, nh)}


def kv_bpt(g, bits):
    return 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * (bits / 8.0)


def run(model_dir, label, params_b, ram_gb=96.0, ssd_tb=2.0, sink=4096, recent=131072, synthetic=False):
    g = {"layers": 80, "kv_heads": 8, "head_dim": 128} if synthetic else _geom(model_dir)
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


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--synthetic":
        run(None, sys.argv[2] if len(sys.argv) > 2 else "stkv70b",
            float(sys.argv[3]) if len(sys.argv) > 3 else 70.0, synthetic=True)
    elif a == "--help":
        print(__doc__)
    else:
        run(a, sys.argv[2] if len(sys.argv) > 2 else "model",
            float(sys.argv[3]) if len(sys.argv) > 3 else 7.0)
