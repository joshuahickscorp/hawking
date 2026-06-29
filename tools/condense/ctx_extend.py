#!/usr/bin/env python3.12
"""ctx_extend.py — the LONG-CONTEXT lane: how far can we stretch the context window for users,
and what does it cost in RAM, WITHOUT retraining. A Hawking serve-engine concern (HIDE only exposes
it). Three levers, all post-hoc:

  1. RoPE scaling (YaRN/NTK) — rescale the trained RoPE frequencies at serve time to stretch the
     trained window (Qwen2.5 32k -> 64k/128k/256k). Quality is validated by NIAH retrieval, not by
     "it didn't crash". The factor = target_ctx / trained_ctx.
  2. KV-cache RAM — the PRACTICAL wall. kv_bytes/token = 2(K,V) * layers * kv_heads * head_dim *
     bytes. This tool computes the cache at f16 and at KV-quant (4-bit/2-bit) per target context, so
     we know what fits the 96GB box alongside the condensed weights (the two compose: smaller weights
     -> more KV headroom -> longer context).
  3. SSM (RWKV-7) — if the model is an SSM, context is O(1) state (constant RAM, no growing KV); the
     window is bounded by state CAPACITY, not RAM. The structural long-context moat Hawking already has.

This is a PROBE/measurement: it reports the max context that holds NIAH >= the gate, the KV RAM at
each rung, and (for SSM) the flat-memory point. It claims no serving win on its own; the native
extended-context serve (RoPE-scaled kernel + KV-quant) is the Rust serve build. EFFECTIVE discipline:
KV RAM is the real measured cache, never a nominal token count. KILL: if even YaRN cannot hold NIAH
past the trained context on a 7B+ model, that model does not extend post-hoc (needs a long-ctx FT).

Usage:
  ctx_extend.py <model-dir> <label> [--targets 65536,131072,262144]   # real (Studio-tier at long ctx)
  ctx_extend.py --synthetic <label> [--arch dense|ssm]                # full logic, runs anywhere
"""
import sys, os, json, math

os.environ.setdefault("STRAND_NO_GPU", "1")
OUT = "reports/condense"
NIAH_GATE = float(os.environ.get("NIAH_TOL", "0.90"))   # retrieval accuracy floor to call a ctx "held"


def _cfg(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0].lower()
    is_ssm = "rwkv" in arch or "mamba" in arch or "ssm" in arch
    n_heads = c.get("num_attention_heads", 32)
    return {
        "arch": arch, "is_ssm": is_ssm,
        "trained_ctx": c.get("max_position_embeddings", 32768),
        "layers": c.get("num_hidden_layers", 28),
        "kv_heads": c.get("num_key_value_heads", n_heads),
        "head_dim": c.get("head_dim") or (c.get("hidden_size", 4096) // max(1, n_heads)),
        "rope_theta": c.get("rope_theta", 10000.0),
        "rope_scaling": c.get("rope_scaling"),
        "hidden": c.get("hidden_size", 4096),
    }


def kv_bytes_per_token(cfg, bits=16):
    # 2 (K and V) * layers * kv_heads * head_dim * bytes_per_elem
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * (bits / 8.0)


def kv_gb(cfg, ctx, bits=16):
    return kv_bytes_per_token(cfg, bits) * ctx / 1e9


def ssm_state_gb(cfg):
    # RWKV-7 carries a fixed per-layer state (~ hidden^2-ish wkv state); O(1) in context.
    # Approximate the resident recurrent state; the point is it does NOT grow with context.
    per_layer = cfg["hidden"] * cfg["head_dim"] * 2  # wkv + aux, f16
    return cfg["layers"] * per_layer * 2 / 1e9


def yarn_factor(trained, target):
    return round(target / trained, 2)


def _niah_real(model_dir, cfg, ctx):
    """Studio-tier: load with YaRN rope_scaling to `ctx`, run a needle-in-a-haystack retrieval.
    Long contexts (>~16k on a 7B) need the 96GB box; gated here. Returns accuracy or None if skipped."""
    cap = int(os.environ.get("CTX_MAX_REAL", "8192"))
    if ctx > cap:
        return None  # skip on the small box; the Studio raises CTX_MAX_REAL
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        factor = yarn_factor(cfg["trained_ctx"], ctx)
        kw = {}
        if factor > 1.0:
            kw["rope_scaling"] = {"type": "yarn", "factor": factor,
                                  "original_max_position_embeddings": cfg["trained_ctx"]}
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=getattr(torch, os.environ.get("DOCTOR_DTYPE", "bfloat16")),
            attn_implementation="eager", **kw).to(os.environ.get("DOCTOR_DEVICE", "cpu")).eval()
        # minimal NIAH: hide a key-value fact at several depths in `ctx` filler, ask for it.
        hits, trials = 0, 3
        filler = (" the sky is calm and the engine hums along quietly.") * (ctx // 12)
        for d, depth in enumerate((0.1, 0.5, 0.9)):
            needle = f" SPECIAL_CODE_{d} is {1000+d}. "
            pos = int(len(filler) * depth)
            ctxtext = filler[:pos] + needle + filler[pos:]
            prompt = ctxtext[:ctx * 4] + f"\nWhat is SPECIAL_CODE_{d}? Answer with the number."
            ids = tok(prompt, return_tensors="pt").input_ids[:, :ctx].to(m.device)
            with torch.no_grad():
                out = m.generate(ids, max_new_tokens=8, do_sample=False)
            ans = tok.decode(out[0, ids.shape[1]:])
            if str(1000 + d) in ans:
                hits += 1
        del m
        return hits / trials
    except Exception as e:
        print(f"[ctx] real NIAH @ {ctx} failed ({e}); marking skipped", file=sys.stderr)
        return None


def _niah_synth(cfg, ctx):
    # transparent model: YaRN holds well up to ~4x trained, degrades beyond unless FT'd.
    r = ctx / cfg["trained_ctx"]
    base = 0.97 if r <= 1 else (0.93 if r <= 2 else (0.86 if r <= 4 else 0.6 if r <= 8 else 0.35))
    return round(base, 2)


def run(model_dir, label, targets, synthetic=False, arch="dense"):
    if synthetic:
        cfg = {"arch": arch, "is_ssm": arch == "ssm", "trained_ctx": 32768, "layers": 28,
               "kv_heads": 4, "head_dim": 128, "rope_theta": 1e6, "rope_scaling": None, "hidden": 3584}
    else:
        cfg = _cfg(model_dir)
    rungs = []
    held_max = cfg["trained_ctx"]
    for ctx in targets:
        niah = _niah_synth(cfg, ctx) if synthetic else _niah_real(model_dir, cfg, ctx)
        rung = {
            "ctx": ctx, "yarn_factor": yarn_factor(cfg["trained_ctx"], ctx),
            "kv_gb_f16": round(kv_gb(cfg, ctx, 16), 2),
            "kv_gb_q4": round(kv_gb(cfg, ctx, 4), 2),
            "kv_gb_q2": round(kv_gb(cfg, ctx, 2), 2),
            "niah": niah, "held": (niah is not None and niah >= NIAH_GATE),
        }
        if rung["held"]:
            held_max = max(held_max, ctx)
        rungs.append(rung)
    rec = {
        "model": label, "arch": cfg["arch"], "is_ssm": cfg["is_ssm"],
        "trained_ctx": cfg["trained_ctx"], "kv_bytes_per_token_f16": int(kv_bytes_per_token(cfg, 16)),
        "extended_ctx_held": held_max, "extend_factor": round(held_max / cfg["trained_ctx"], 1),
        "rungs": rungs, "niah_gate": NIAH_GATE, "probe": True,
        "serve_gated_on": "RoPE-scaled .tq serve kernel + KV-quant (the Rust serve build)",
    }
    if cfg["is_ssm"]:
        rec["ssm_note"] = (f"SSM (RWKV-7 class): O(1) state ~{ssm_state_gb(cfg):.2f}GB CONSTANT in "
                           f"context — window is state-CAPACITY-bound, not RAM-bound. The long-ctx moat.")
        rec["verdict"] = "SSM long-context: flat memory, context effectively unbounded (quality-limited)"
    else:
        any_held = any(r["held"] and r["ctx"] > cfg["trained_ctx"] for r in rungs)
        measured = any(r["niah"] is not None for r in rungs)
        if not measured:
            rec["verdict"] = "GATED: NIAH at target contexts needs the 96GB box (raise CTX_MAX_REAL)"
        elif any_held:
            rec["verdict"] = f"YaRN extends to {held_max} ({rec['extend_factor']}x trained) holding NIAH>={NIAH_GATE}"
        else:
            rec["verdict"] = "KILL: YaRN cannot hold NIAH past the trained context — needs a long-ctx fine-tune"
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{label}_ctx.json", "w"), indent=2)
    print(f"[ctx] {label}: {rec['verdict']}", file=sys.stderr)
    for r in rungs:
        print(f"  ctx {r['ctx']:>7}  yarn x{r['yarn_factor']:<4}  KV f16 {r['kv_gb_f16']:>5}GB / "
              f"q4 {r['kv_gb_q4']:>5}GB / q2 {r['kv_gb_q2']:>5}GB  NIAH {r['niah']}  {'HELD' if r['held'] else '-'}",
              file=sys.stderr)
    print(f"# KILL: if YaRN can't hold NIAH past trained ctx on 7B+, the model needs a long-ctx FT "
          f"(SSM/RWKV-7 sidesteps this entirely).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    DEF = [65536, 131072, 262144]
    if a == "--synthetic":
        arch = "ssm" if "--arch" in sys.argv and sys.argv[sys.argv.index("--arch") + 1] == "ssm" else "dense"
        run(None, sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "synth",
            DEF, synthetic=True, arch=arch)
    elif a == "--help":
        print(__doc__)
    else:
        tgts = DEF
        if "--targets" in sys.argv:
            tgts = [int(x) for x in sys.argv[sys.argv.index("--targets") + 1].split(",")]
        sys.exit(run(a, sys.argv[2] if len(sys.argv) > 2 else "model", tgts))
