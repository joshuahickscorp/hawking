#!/usr/bin/env python3
"""G019 three-way accounting: resident bytes, AVERAGE active bytes per token, WORST-CASE.

A single "BPW" hides which bytes are stored, which are moved every token, and which are moved
only sometimes. The three can differ by a lot, and a representation that is small but moves
everything is worse than one that is larger and moves little.

For THIS model the interesting result is that two of the three collapse, and the reason is
already measured rather than assumed: the MLP is NOT conditional (G088 -- a token needs 28-57%
of channels and adjacent-token active-set overlap equals random-pair overlap), and
dense-to-conditional found mid-network unconditional-zero mass of about 0 (G087). So every
weight byte is read every token and average equals worst case on the weight path.

What does NOT collapse is STATE. The architecture is hybrid at full_attention_interval 4:
16 of 64 layers are full attention and carry a growing KV cache, while 48 are DeltaNet and
carry a FIXED-SIZE recurrent state. So state cost is sublinear in depth and linear in context
only on a quarter of the layers -- which changes where the lever is at long context.

Everything is computed from the real config and the real artifact bytes. Nothing is assumed
about a codec that is not on disk.
"""
from __future__ import annotations
import argparse, json, os, sys

RUNS = "workspace/campaign/records/runs/qwen38-27b"
SOURCE_PARAM_COUNT = 26_895_998_464


def cfg():
    c = json.load(open(os.path.join(RUNS, "bf16", "config.json")))
    return c.get("text_config", c)


def artifact_bytes(name):
    root = os.path.join(RUNS, name)
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(root) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="append",
                    default=None)
    ap.add_argument("--contexts", default="4096,32768,131072")
    ap.add_argument("--kv-bytes", type=int, default=4,
                    help="the cache is f32: qwen38_hybrid_decode.rs allocates f32b and "
                         "mha.metal:605 reads device const float*. An earlier run of this "
                         "tool defaulted to 2 (f16) and understated the state share by 2x.")
    a = ap.parse_args()
    arts = a.artifact or ["uniform-q4-v1", "compact-q3attn-r1p2-v1"]
    c = cfg()
    L = c["num_hidden_layers"]
    types = c.get("layer_types") or []
    n_full = sum(1 for t in types if t == "full_attention") or L // c["full_attention_interval"]
    n_lin = L - n_full
    kv_per_tok = (n_full * 2 * c["num_key_value_heads"] * c["head_dim"] * a.kv_bytes)
    # DeltaNet carries a fixed recurrent state per layer: keys x values outer product,
    # plus a short causal conv window. It does NOT grow with context.
    delta_state = n_lin * (c["linear_num_value_heads"] * c["linear_key_head_dim"]
                           * c["linear_value_head_dim"] * 4
                           + c["linear_num_key_heads"] * c["linear_key_head_dim"]
                           * c["linear_conv_kernel_dim"] * 4)
    print(f"{L} layers: {n_full} full_attention (growing KV), {n_lin} linear_attention "
          f"(fixed recurrent state)")
    print(f"KV per context token: {kv_per_tok:,} B   DeltaNet state, context-independent: "
          f"{delta_state:,} B\n")

    print(f"{'artifact':<26}{'resident W B':>16}{'W BPW':>9}{'active W/token':>16}{'avg==worst':>12}")
    rows = {}
    for n in arts:
        b = artifact_bytes(n)
        rows[n] = b
        # every weight byte is read every token: the MLP is measured NOT conditional (G088)
        # and unconditional-zero mass is ~0 (G087), so there is no sometimes-path to discount
        print(f"{n:<26}{b:>16,}{8*b/SOURCE_PARAM_COUNT:>9.4f}{b:>16,}{'yes':>12}")

    print(f"\n{'artifact':<26}{'context':>10}{'KV B':>16}{'state B':>14}"
          f"{'resident total':>16}{'state share':>12}")
    for n, b in rows.items():
        for ctx in (int(x) for x in a.contexts.split(",")):
            kv = kv_per_tok * ctx
            tot = b + kv + delta_state
            print(f"{n:<26}{ctx:>10,}{kv:>16,}{delta_state:>14,}{tot:>16,}"
                  f"{100*(kv+delta_state)/tot:>11.2f}%")

    print("\nwhat this says:")
    print("  average and worst-case ACTIVE weight bytes are equal on this model, so the only")
    print("  lever on information moved per token is total stored weight bytes. There is no")
    print("  conditional path to exploit -- that is measured (G087, G088), not assumed.")
    b = rows[arts[-1]]
    for ctx in (int(x) for x in a.contexts.split(",")):
        kv = kv_per_tok * ctx
        print(f"  at {ctx:>7,} context, state is {100*(kv+delta_state)/(b+kv+delta_state):.2f}% "
              f"of resident on {arts[-1]}; latent-KV would attack that share, weight "
              f"compression cannot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
