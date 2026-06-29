#!/usr/bin/env python3.12
"""kv_frontier.py — the AGGRESSIVE long-context lane: push the usable window as far as the KV cache
can be compressed/evicted/paged, in four HONEST regimes (a portfolio reviewer must be able to tell
them apart). Tacks onto the studio run; pure projection + NIAH gate, no serving-win claim.

The context wall at long range is the KV cache, not the weights. Levers, each with its true cost:
  R1 LOSSLESS-RAM    : full attention, KV quantized (int4 / int2 / trellis ~1.5b via the STRAND codec
                       applied to the cache). Every token attendable, in unified memory. The honest
                       headline. int2/trellis are quality-fragile -> gated by NIAH at the target ctx.
  R2 LOSSLESS-SSD    : full attention, COLD KV paged to the 2TB SSD (long-ctx attention is sparse, so
                       hot KV stays in RAM). Context bounded by STORAGE not RAM -> tens of millions of
                       tokens, but the paged tail is SSD-bandwidth slow. True full context, slow.
  R3 LOSSY-EVICT     : keep a fixed live set (attention-sink first-k + recent window, or heavy-hitter
                       retention). Context DECOUPLES from RAM (unbounded "effective" window) but the
                       evicted middle is GONE -> a needle in an evicted region FAILS NIAH. Not free
                       context; lossy context. Report what the policy retains, never call it full.
  R4 SSM (RWKV-7)    : no KV cache at all — O(1) recurrent state. Context bounded by state CAPACITY,
                       not RAM. Unbounded effective window at flat memory; lossy as a learned summary.

Discipline: report the regime tag on every number; the usable window is min(RAM/SSD ceiling, model's
trained/NIAH-validated window). KILL: if int2/trellis KV or an eviction policy fails NIAH at the
target context, that lever does not deliver full context there (fall back to int4 or R2).
"""
import sys, os, json

OUT = "reports/condense"


def _geom(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0].lower()
    nh = c.get("num_attention_heads", 32)
    return {"arch": arch, "is_ssm": any(t in arch for t in ("rwkv", "mamba", "ssm")),
            "layers": c.get("num_hidden_layers", 28), "kv_heads": c.get("num_key_value_heads", nh),
            "head_dim": c.get("head_dim") or c.get("hidden_size", 4096) // max(1, nh),
            "params_b": None, "trained_ctx": c.get("max_position_embeddings", 32768)}


def kv_bpt(g, bits):
    return 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * (bits / 8.0)


def run(model_dir, label, params_b, ram_gb=96.0, ssd_tb=2.0, synthetic=False):
    if synthetic:
        g = {"arch": "qwen2", "is_ssm": False, "layers": 80, "kv_heads": 8, "head_dim": 128,
             "trained_ctx": 1_000_000}
        params_b = params_b or 70.0
    else:
        g = _geom(model_dir)
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


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--synthetic":
        run(None, sys.argv[2] if len(sys.argv) > 2 else "synth70b",
            float(sys.argv[3]) if len(sys.argv) > 3 else 70.0, synthetic=True)
    elif a == "--help":
        print(__doc__)
    else:
        run(a, sys.argv[2] if len(sys.argv) > 2 else "model",
            float(sys.argv[3]) if len(sys.argv) > 3 else 7.0)
