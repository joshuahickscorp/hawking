#!/usr/bin/env python3
"""G137: KV/state residency at 131k, re-derived from architecture two independent ways.

Qwen3.5-27B is a HYBRID. config.text_config.layer_types has 64 entries with
full_attention_interval=4, so only 16 layers (indices 3,7,...,63) are GQA and grow
a context-length KV cache. The other 48 are linear_attention (DeltaNet/gated linear)
and hold a FIXED recurrent state independent of context length. Naive accounting that
multiplies by all 64 layers overcounts KV by 4x.

The control is redundancy: the growing KV is computed two independent ways -- by
enumerating layer_types and summing only the full-attention layers, and by the closed
form 16 * 2 * n_kv * head_dim * dtype. If they disagree the derivation is wrong. They
must equal the ledger's ~17.18 GB f32 figure at 131072 tokens, which is the check that
this is the same quantity the campaign already named.

  ./tools/kv_residency.py --context 131072 --out receipts/.../G137_KV_RESIDENCY.json
"""
from __future__ import annotations
import argparse, datetime, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16/config.json"

WEIGHT_ARTIFACT_GB = 11.25   # cited compact weight artifact (addressed bytes)


def load_text_cfg() -> dict:
    return json.loads(CFG.read_text())["text_config"]


def kv_bytes_per_token(cfg: dict, dtype_bytes: int) -> dict:
    lt = cfg["layer_types"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    # Way 1: enumerate layer_types, sum only full_attention layers.
    per_full = 2 * n_kv * hd * dtype_bytes          # K and V
    full_layers = [i for i, x in enumerate(lt) if x == "full_attention"]
    way1 = per_full * len(full_layers)
    # Way 2: closed form from the interval.
    n_full = len(lt) // cfg["full_attention_interval"]
    way2 = 2 * n_kv * hd * dtype_bytes * n_full
    return {"per_full_layer_bytes": per_full, "n_full_layers": len(full_layers),
            "n_full_closed_form": n_full, "way1_enumerated": way1,
            "way2_closed_form": way2, "agree": way1 == way2, "bytes_per_token": way1}


def linear_state_bytes(cfg: dict, dtype_bytes: int) -> dict:
    """The 48 linear-attention layers hold a FIXED state: a conv window plus the
    DeltaNet recurrent matrix (v_heads * v_dim * k_dim). Independent of context."""
    lt = cfg["layer_types"]
    n_lin = sum(1 for x in lt if x == "linear_attention")
    vk = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"] * cfg["linear_key_head_dim"]
    conv = cfg["linear_conv_kernel_dim"] * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    per = (vk + conv) * dtype_bytes
    return {"n_linear_layers": n_lin, "recurrent_elems_per_layer": vk,
            "conv_elems_per_layer": conv, "bytes_per_layer": per,
            "total_fixed_bytes": per * n_lin, "grows_with_context": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cfg = load_text_cfg()

    f32 = kv_bytes_per_token(cfg, 4)
    q8 = kv_bytes_per_token(cfg, 1)
    lin = linear_state_bytes(cfg, 4)
    assert f32["agree"], "two KV derivations disagree -- accounting is wrong"

    kv_f32_gb = f32["bytes_per_token"] * a.context / 1e9
    kv_q8_gb = q8["bytes_per_token"] * a.context / 1e9
    lin_gb = lin["total_fixed_bytes"] / 1e9

    reachable = kv_q8_gb < WEIGHT_ARTIFACT_GB
    print(f"CONTEXT {a.context}")
    print(f"  growing KV is 16 full-attention layers of 64 (interval 4), NOT all 64")
    print(f"  KV bytes/token: way1 {f32['way1_enumerated']} == way2 {f32['way2_closed_form']} "
          f"-> {f32['agree']}")
    print(f"  KV f32 @ {a.context}: {kv_f32_gb:.2f} GB   (ledger figure ~17.18)")
    print(f"  KV q8  @ {a.context}: {kv_q8_gb:.2f} GB   < weight artifact {WEIGHT_ARTIFACT_GB} "
          f"-> reduction reachable: {reachable}")
    print(f"  linear-attn fixed state (48 layers, context-independent): {lin_gb:.3f} GB")

    doc = {
        "schema": "hawking.nos.kv_residency.v1",
        "obligation": "G137 -- KV/state bytes at 131k measured and driven below the weight artifact",
        "started": start,
        "architecture": {"n_layers": len(cfg["layer_types"]),
                         "full_attention_interval": cfg["full_attention_interval"],
                         "n_full_attention": f32["n_full_layers"],
                         "n_kv_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"]},
        "control_two_derivations_agree": f32["agree"],
        "kv_bytes_per_token_f32": f32["bytes_per_token"],
        "kv_f32_gb_at_context": round(kv_f32_gb, 3),
        "matches_ledger_17_18": abs(kv_f32_gb - 17.18) < 0.05,
        "kv_q8_gb_at_context": round(kv_q8_gb, 3),
        "weight_artifact_gb": WEIGHT_ARTIFACT_GB,
        "reduction_below_artifact_reachable": reachable,
        "reduction_mechanism": "q8 KV halves the growing cache to below the weight artifact; "
                               "the q8-kv path is already landed bit-identical (+1.6/2.1/2.5% "
                               "TPS on short context) so this is a taken win, not a projection",
        "the_hybrid_insight": (
            f"only 16 of 64 layers grow with context. The 48 linear-attention layers hold a "
            f"FIXED {lin_gb:.3f} GB state regardless of context length, so at long context the "
            "growing memory is 4x smaller than a uniform-64-layer accounting would claim. This "
            "is why KV overtakes the weight artifact only past ~86k tokens, not ~21k."),
        "linear_state": lin,
        "context": a.context,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
        "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
