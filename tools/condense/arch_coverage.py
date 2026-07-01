#!/usr/bin/env python3.12
"""arch_coverage.py — the NEW-ARCHITECTURE tack-on: Mamba2 condense-track support.

RWKV-7 is the only SSM the condense/context tooling understands (ctx_extend.py / kv_hybrid.py
approximate its per-layer state). Hawking ALSO ships a real, wired Mamba2 serve engine
(crates/hawking-core/src/model/mamba2.rs, GGUF-native, no KV cache) with ZERO condense-track
coverage — this closes that gap so Mamba2 gets the same flat-memory long-context treatment as
RWKV-7, computed from its ACTUAL state geometry (not RWKV's approximation).

Mamba2 per-layer resident state (SSD - Selective State Space Duality), from the real config
(mamba2.rs Mamba2Config): the SSM state matrix is (n_heads * head_dim * state_size) elements,
plus a short conv state (inner * conv_kernel). BOTH are O(1) in context length - same structural
long-context moat as RWKV-7, computed honestly from Mamba2's own shape instead of reusing RWKV's.

Also: the doctor registry's train-free stack (calib/AWQ/mixed-prec/residual) is architecture-
agnostic (it operates on 2D Linear weights) and applies to Mamba2's in/out/conv projections
unchanged; only the attention-specific and MoE-specific levers (learned_rotation targeting QK,
expert_alloc) don't apply. This tool reports which Doctor methods are ARCH-COMPATIBLE for a model
so the auto-selector (doctor_registry.select) doesn't waste a bake on an inapplicable method.

Usage:
  arch_coverage.py <model-dir> <label>          # real HF/GGUF config -> state geometry + compat
  arch_coverage.py --synthetic <arch> <label>    # arch in {mamba2, rwkv7, dense, moe}; runs anywhere
"""
import sys, os, json

OUT = "reports/condense"
DOCTOR_LEVERS = ["calib", "awq", "mixed_prec", "residual", "outlier_channel",
                 "block_qat", "gptq_hessian", "deep_kd"]
ARCH_INCOMPATIBLE = {           # levers that don't apply per architecture family
    "ssm": {"learned_rotation"},              # no QK attention to rotate
    "moe_only": {"expert_alloc"},             # gate this ON only for MoE, off elsewhere
}


def _detect(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0].lower()
    is_mamba2 = "mamba2" in arch
    is_rwkv = "rwkv" in arch
    is_moe = bool(c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts"))
    return c, arch, is_mamba2, is_rwkv, is_moe


def mamba2_state_gb(n_layers, n_heads, head_dim, state_size, inner, conv_kernel, bytes_elem=2):
    """The REAL Mamba2 per-layer resident state (SSD): ssm state + short conv state. O(1) in ctx."""
    ssm_state = n_heads * head_dim * state_size
    conv_state = inner * conv_kernel
    return n_layers * (ssm_state + conv_state) * bytes_elem / 1e9


def arch_report(label, arch, is_mamba2, is_rwkv, is_moe, state_gb=None, geometry=None):
    is_ssm = is_mamba2 or is_rwkv
    excluded = set()
    if is_ssm:
        excluded |= ARCH_INCOMPATIBLE["ssm"]
    if not is_moe:
        excluded |= ARCH_INCOMPATIBLE["moe_only"]
    compatible = [m for m in DOCTOR_LEVERS if m not in excluded]
    rec = {
        "model": label, "arch": arch, "is_ssm": is_ssm, "is_mamba2": is_mamba2,
        "is_rwkv": is_rwkv, "is_moe": is_moe,
        "state_gb_flat": round(state_gb, 3) if state_gb is not None else None,
        "geometry": geometry,
        "doctor_compatible_levers": compatible, "doctor_excluded_levers": sorted(excluded),
        "long_context_note": ("O(1) recurrent state, flat memory, context bounded by state "
                              "CAPACITY not RAM (the SSM moat)") if is_ssm else
                              "dense attention: KV cache grows with context (see ctx_extend.py/kv_frontier.py)",
        "probe": True,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{label}_arch.json", "w"), indent=2)
    print(f"[arch] {label}: {arch} (SSM={is_ssm}, MoE={is_moe})" +
          (f", flat state {state_gb:.2f}GB" if state_gb is not None else ""), file=sys.stderr)
    print(f"  Doctor-compatible levers: {compatible}", file=sys.stderr)
    if excluded:
        print(f"  excluded (arch-inapplicable): {sorted(excluded)}", file=sys.stderr)
    return rec


def run(model_dir, label):
    c, arch, is_mamba2, is_rwkv, is_moe = _detect(model_dir)
    state_gb, geom = None, None
    if is_mamba2:
        n_layers = c.get("num_hidden_layers", 48)
        n_heads = c.get("num_heads") or c.get("n_heads", 32)
        head_dim = c.get("head_dim", 64)
        state_size = c.get("state_size") or c.get("ssm_state_size", 128)
        inner = c.get("intermediate_size") or c.get("expand", 2) * c.get("hidden_size", 1024)
        conv_kernel = c.get("conv_kernel", 4)
        state_gb = mamba2_state_gb(n_layers, n_heads, head_dim, state_size, inner, conv_kernel)
        geom = {"n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
               "state_size": state_size, "inner": inner, "conv_kernel": conv_kernel}
    return arch_report(label, arch, is_mamba2, is_rwkv, is_moe, state_gb, geom)


def run_synthetic(arch_key, label):
    table = {
        "mamba2": ("mamba2forcausallm", True, False, False,
                  mamba2_state_gb(48, 32, 64, 128, 2048, 4),
                  {"n_layers": 48, "n_heads": 32, "head_dim": 64, "state_size": 128,
                   "inner": 2048, "conv_kernel": 4}),
        "rwkv7": ("rwkv7forcausallm", False, True, False, None, None),
        "dense": ("qwen2forcausallm", False, False, False, None, None),
        "moe": ("deepseekv2forcausallm", False, False, True, None, None),
    }
    if arch_key not in table:
        print(f"[arch] unknown synthetic arch {arch_key}; choices: {list(table)}", file=sys.stderr)
        return None
    arch, is_m2, is_rwkv, is_moe, sgb, geom = table[arch_key]
    return arch_report(label, arch, is_m2, is_rwkv, is_moe, sgb, geom)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--synthetic":
        run_synthetic(sys.argv[2] if len(sys.argv) > 2 else "mamba2",
                      sys.argv[3] if len(sys.argv) > 3 else "synth-arch")
    elif a == "--help":
        print(__doc__)
    else:
        run(a, sys.argv[2] if len(sys.argv) > 2 else "model")
