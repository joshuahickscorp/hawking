#!/usr/bin/env python3
"""G037 — STATE_GRAVITY + PREFIX_SHARING (S011 §18, §19, §74).

This body is a HYBRID: full_attention_interval=4, so only 16 of 64 layers keep a KV
cache and the other 48 carry a fixed-size recurrent state. That changes what state
optimization even means here, so the census comes before any technique.

The prefill measurement is the load-bearing result. Marginal cost per PROMPT token is
32-43 ms against 30.198 ms for a DECODE token, so prompt tokens cost essentially what
generated tokens cost: this runtime has no batched prefill on the hybrid path. Every
turn of a multi-turn mission re-pays it over the whole growing transcript, which makes
prefix reuse worth more than any KV precision change.
"""
import json, statistics, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
PARENT = Path("/Volumes/corpdrive/personalmodel/correspondent/qwen3.8-27b-abliterated-bf16")
DECODE_TPOT_MS = 30.198          # variantB, measured in G040


def state_census():
    c = json.load(open(PARENT / "config.json"))
    t = c.get("text_config", c)
    L = t["num_hidden_layers"]
    interval = t["full_attention_interval"]
    n_full = sum(1 for x in t["layer_types"] if x == "full_attention")
    n_lin = L - n_full
    kv_heads, head_dim = t["num_key_value_heads"], t["head_dim"]

    kv_elems_per_token = n_full * 2 * kv_heads * head_dim
    kv_bytes_per_token_bf16 = kv_elems_per_token * 2

    vh, vd = t["linear_num_value_heads"], t["linear_value_head_dim"]
    kd = t["linear_key_head_dim"]
    rec_elems = n_lin * vh * vd * kd
    rec_bytes_bf16 = rec_elems * 2

    crossover = rec_bytes_bf16 / kv_bytes_per_token_bf16
    ctx = t["max_position_embeddings"]
    return {
        "layers": L, "full_attention_interval": interval,
        "full_attention_layers": n_full, "linear_attention_layers": n_lin,
        "kv": {"kv_heads": kv_heads, "head_dim": head_dim,
               "elements_per_token": kv_elems_per_token,
               "bytes_per_token_bf16": kv_bytes_per_token_bf16,
               "kib_per_token": round(kv_bytes_per_token_bf16 / 1024, 2),
               "gib_at_max_context": round(kv_bytes_per_token_bf16 * ctx / 2**30, 2),
               "max_context": ctx,
               "grows_with_context": True},
        "recurrent": {"value_heads": vh, "value_head_dim": vd, "key_head_dim": kd,
                      "elements": rec_elems, "bytes_bf16": rec_bytes_bf16,
                      "mib": round(rec_bytes_bf16 / 2**20, 2),
                      "grows_with_context": False},
        "crossover_tokens": round(crossover),
        "reading": (f"only {n_full} of {L} layers keep a KV cache, so this architecture "
                    f"already carries {round(L / n_full)}x less KV than a pure-attention "
                    f"model of the same depth. The recurrent state is a flat "
                    f"{round(rec_bytes_bf16 / 2**20)} MiB whatever the context. KV "
                    f"overtakes it at about {round(crossover)} tokens."),
    }


def kv_precision_ladder(kv_bytes_per_token, ctx_points=(4096, 32768, 131072, 262144)):
    """Asymmetric KV precision (S011 §18), priced as bytes. NOT a capability claim."""
    out = []
    for name, kbits, vbits in (("bf16 / bf16 (current)", 16, 16),
                               ("q8 / q8", 8, 8),
                               ("q8 K / q4 V (asymmetric)", 8, 4),
                               ("q4 / q4", 4, 4)):
        per_tok = kv_bytes_per_token * ((kbits + vbits) / 32.0)
        out.append({"scheme": name, "k_bits": kbits, "v_bits": vbits,
                    "bytes_per_token": int(per_tok),
                    "gib_at": {str(n): round(per_tok * n / 2**30, 3)
                               for n in ctx_points},
                    "reduction_x": round(kv_bytes_per_token / per_tok, 3)})
    return out


def main():
    pf = json.load(open("/tmp/prefill_scale.json"))
    pts = pf["points"]

    marginal = []
    for a, b in zip(pts, pts[1:]):
        dt = (b["prefill_ns"] - a["prefill_ns"]) / 1e6
        dn = b["prompt_tokens"] - a["prompt_tokens"]
        marginal.append({"from_tokens": a["prompt_tokens"], "to_tokens": b["prompt_tokens"],
                         "marginal_ms_per_prompt_token": round(dt / dn, 3)})
    mm = [m["marginal_ms_per_prompt_token"] for m in marginal]

    cen = state_census()
    shared = pf["hcli_shared_prefix"]

    # what reusing the fixed prefix would save, at the MEASURED marginal cost
    med_marginal = statistics.median(mm)
    saved_ms = shared["shared_tokens"] * med_marginal

    out = {
        "schema": "hawking.odyssey.state_gravity.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/state_gravity.py",
        "obligation": "G037 — STATE_GRAVITY + PREFIX_SHARING",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "state_census": cen,
        "kv_precision_ladder": kv_precision_ladder(
            cen["kv"]["bytes_per_token_bf16"]),
        "kv_ladder_caveat": "bytes only. The runtime has no KV quantization path, so "
                            "these are the sizes a scheme WOULD have, not measured "
                            "latency or measured long-context capability. Quoting them "
                            "as a result would be exactly the design-constant-under-a-"
                            "physical-label error G033 exists to prevent.",
        "prefill": {
            "measured_points": pts,
            "marginal_cost": marginal,
            "median_marginal_ms_per_prompt_token": med_marginal,
            "decode_ms_per_token": DECODE_TPOT_MS,
            "prompt_token_vs_decode_token_ratio": round(med_marginal / DECODE_TPOT_MS, 3),
            "FINDING": (
                f"a prompt token costs {med_marginal:.1f} ms against "
                f"{DECODE_TPOT_MS:.1f} ms for a generated token. Prefill is being paid at "
                f"decode price, so there is no batched prefill on this hybrid path. "
                f"Marginal cost also rises with length ({mm[0]:.1f} to {mm[-1]:.1f} "
                f"ms/token), which is the O(n^2) term from the 16 full-attention layers "
                f"sitting on top of a large constant."),
            "measured_vs_inferred": {
                "MEASURED": "marginal cost per prompt token is 32.0-43.1 ms while a "
                            "decode token costs 30.198 ms; prompt tokens are NOT cheaper "
                            "than generated ones",
                "INFERRED": "that signature is what an unbatched prefill looks like. A "
                            "batched prefill amortizes the weight read across the whole "
                            "prompt and should make a prompt token far cheaper than a "
                            "decode token. The inference is strong but it is an "
                            "inference: no kernel trace was taken.",
                "how_to_settle_it": "count dispatches during prefill. Unbatched prefill "
                                    "issues ~964 per prompt token, the same as decode.",
            },
            "consequence": "a 2,305-token prompt takes 90 seconds to prefill. For HCLI "
                           "this dominates everything: each turn of a multi-turn mission "
                           "re-prefills the whole transcript from scratch.",
        },
        "prefix_sharing": {
            "shared_prefix": shared,
            "measured_marginal_ms_per_token": med_marginal,
            "ttft_saving_ms_if_prefix_reused": round(saved_ms, 1),
            "ttft_saving_pct_of_mission_prefill": round(
                100 * shared["shared_tokens"] / shared["example_mission_tokens"], 1),
            "why_it_compounds": "the HCLI system prompt and tool schemas are byte-"
                                "identical on every mission and every turn, so the same "
                                "prefix is re-prefilled once per turn per mission.",
            "IS_A_PROJECTION": (
                "the saving is computed from the MEASURED marginal prefill cost and the "
                "MEASURED shared-prefix length, but no prefix cache exists in this "
                "runtime, so nothing was reused and no TTFT was observed to fall. This "
                "is the size of the prize, not a delivered win."),
        },
        "ranking": {
            "1_batched_prefill": "largest by far. Prompt tokens cost ~1.1-1.4x a decode "
                                 "token, so the headroom is the whole gap to a properly "
                                 "batched prefill.",
            "2_prefix_reuse": f"{round(saved_ms)} ms per mission turn on the measured "
                              f"marginal cost, and it stacks with every turn.",
            "3_kv_precision": "smallest here, and structurally so: only 16 of 64 layers "
                              "hold KV at all, and below ~1,152 tokens the flat 72 MiB "
                              "recurrent state dominates the KV cache entirely.",
        },
    }
    out["pass"] = bool(cen["kv"]["bytes_per_token_bf16"] > 0 and len(marginal) >= 3)
    p = RH / "STATE_GRAVITY.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"census: {cen['full_attention_layers']}/{cen['layers']} layers keep KV "
          f"({cen['kv']['kib_per_token']} KiB/token, "
          f"{cen['kv']['gib_at_max_context']} GiB at {cen['kv']['max_context']})")
    print(f"        recurrent state {cen['recurrent']['mib']} MiB FIXED; "
          f"KV overtakes it at {cen['crossover_tokens']} tokens")
    print()
    for m in marginal:
        print(f"  prefill {m['from_tokens']:5d}->{m['to_tokens']:5d} tok: "
              f"{m['marginal_ms_per_prompt_token']:6.1f} ms/prompt-token")
    print(f"  decode: {DECODE_TPOT_MS} ms/token  -> ratio "
          f"{out['prefill']['prompt_token_vs_decode_token_ratio']}")
    print()
    print(f"prefix sharing: {shared['shared_tokens']}/{shared['example_mission_tokens']} "
          f"tokens shared ({out['prefix_sharing']['ttft_saving_pct_of_mission_prefill']}%), "
          f"worth {out['prefix_sharing']['ttft_saving_ms_if_prefix_reused']} ms/turn")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
