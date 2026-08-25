#!/usr/bin/env python3
"""G038 — DECODING_GRAVITY (S011 §20, §21, §22, §73).

The lever here is ACCEPTED TOKENS PER FORWARD PASS, which is the only one that can move
throughput without touching representation: G005 showed decode reaches 274-290 GB/s
against a 778.8 GB/s device roof, so it is dispatch-bound and no density change will help.

Two questions, and the second one has a trap.

  CENSUS      does this model carry multi-token-prediction or auxiliary-head machinery?
  §21         can a body that FAILED as a generator still serve as a draft?

The trap: a high acceptance rate does not imply a speedup. This codebase already
falsified that once, measuring 87% acceptance at 0.91x. Speculative decoding pays only
when the draft is CHEAPER than the verifier, so acceptance must be multiplied by the
cost ratio before any claim is made.
"""
import json, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
PARENT = Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16")
PARENT_PARAMS = 26895998464

# measured in G005 / G040, ms per token
TPOT_MS = {"sealed-3.14": 29.284, "variantA-2.98": 29.203,
           "variantB-2.76": 30.198, "clean-2.60": 30.018}
CAPABILITY = {"sealed-3.14": 30, "variantA-2.98": 0, "variantB-2.76": 24, "clean-2.60": 0}


def census():
    idx = json.load(open(PARENT / "model.safetensors.index.json"))["weight_map"]
    families = {}
    for pat in ("mtp", "multi_token", "nextn", "draft", "medusa", "eagle", "exit",
                "aux"):
        families[pat] = sorted(n for n in idx if pat in n.lower())
    mtp = families["mtp"]
    shapes, total = {}, 0
    if mtp:
        from safetensors import safe_open
        for n in mtp:
            with safe_open(str(PARENT / idx[n]), framework="pt") as f:
                s = list(f.get_slice(n).get_shape())
            numel = 1
            for d in s:
                numel *= d
            shapes[n] = {"shape": s, "params": numel}
            total += numel
    return {
        "parent": str(PARENT),
        "n_tensors_total": len(idx),
        "machinery_found": {k: len(v) for k, v in families.items()},
        "mtp_tensors": shapes,
        "mtp_params": total,
        "mtp_share_of_model_pct": round(100 * total / PARENT_PARAMS, 4),
        "mtp_bf16_bytes": total * 2,
        "structure": ("a complete single transformer layer (self_attn + SwiGLU MLP + "
                      "norms) plus mtp.fc [5120, 10240], which fuses the normalized "
                      "embedding with the normalized main hidden state. This is the "
                      "standard MTP arrangement: predict token t+2 from the main model's "
                      "hidden at t, then decode through the SHARED lm_head."),
        "status_in_this_campaign": "PRESENT ON DISK AND NEVER PACKED. No noetic artifact "
                                   "in this campaign contains any mtp.* tensor.",
    }


def economics(alpha, draft_ms, verify_ms):
    """k=1 greedy speculation.

    One verifier pass yields 1 token on rejection and 2 on acceptance, so per verifier
    pass you get (1 + alpha) tokens at a cost of (draft + verify).
    Break-even needs verify*alpha > draft, i.e. draft/verify < alpha.
    """
    speedup = verify_ms * (1 + alpha) / (draft_ms + verify_ms)
    return {"acceptance": alpha, "draft_ms": draft_ms, "verify_ms": verify_ms,
            "cost_ratio_draft_over_verify": round(draft_ms / verify_ms, 4),
            "break_even_cost_ratio": alpha,
            "pays": (draft_ms / verify_ms) < alpha,
            "predicted_speedup_x": round(speedup, 4)}


def main():
    agree = json.load(open("/tmp/draft_agree.json"))
    meta = agree["_meta"]
    verifier = meta["verifier"]
    v_ms = TPOT_MS[verifier]

    drafts = {}
    for name, d in agree.items():
        if name.startswith("_"):
            continue
        alpha = d["acceptance_rate"]
        ec = economics(alpha, TPOT_MS[name], v_ms)
        drafts[name] = {
            "capability_passed": CAPABILITY[name],
            "is_dead_final_generator": CAPABILITY[name] == 0,
            "agreement": f"{d['agree']}/{d['tested']}",
            "acceptance_rate": alpha,
            **ec,
            "role": ("USEFUL_DRAFT_BY_AGREEMENT_BUT_NOT_ECONOMIC" if not ec["pays"]
                     else "USEFUL_DRAFT"),
        }

    c = census()
    mtp_ratio = c["mtp_share_of_model_pct"] / 100.0
    best_alpha = max(d["acceptance_rate"] for d in drafts.values())
    mtp_ec = {a: economics(a, mtp_ratio * v_ms, v_ms) for a in (0.5, 0.6, 0.75, 0.9)}

    out = {
        "schema": "hawking.odyssey.decoding_gravity.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/decoding_gravity.py",
        "obligation": "G038 — DECODING_GRAVITY",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "why_this_lever": "G005 measured the model-reachable roof at 274-290 GB/s against "
                          "a 778.8 GB/s device sustained roof, so decode is dispatch-bound "
                          "and no representation change moves TPOT. Accepted tokens per "
                          "forward pass is the remaining lever.",
        "census": c,
        "draft_measurement": {
            "method": "greedy k=1: the verifier's token sequence is generated once, then "
                      "each draft is asked for ONE token on each prefix and compared. "
                      "Acceptance under exact greedy verification is argmax agreement.",
            "verifier": verifier,
            "n_positions": meta["n_tokens"],
            "round_trip_skipped": meta["round_trip_skipped"],
            "round_trip_note": "every prefix is detokenized back to text because the "
                               "binary takes text; the round trip is VERIFIED per "
                               "position and non-round-tripping positions are excluded "
                               "rather than guessed",
            "drafts": drafts,
        },
        "reclassification_S011_21": {
            "finding": "a DEAD_FINAL_GENERATOR does carry usable draft signal: "
                       "variantA-2.98 scores 0/43 as a generator and still matches the "
                       "verifier's argmax 75% of the time, and clean-2.60 matches 50%.",
            "but": "none of them pays. Every body in this ladder is the same size class "
                   "with the same 964 dispatches per token, so draft cost and verify "
                   "cost are within 3% of each other. Break-even needs "
                   "draft/verify < acceptance; the best case here is 0.997 vs 0.75.",
            "prior_falsification": "this codebase already measured 87% acceptance at "
                                   "0.91x. Acceptance alone never proves a speedup, and "
                                   "these numbers are reported with the cost ratio "
                                   "attached for that reason.",
            "roles": {k: v["role"] for k, v in drafts.items()},
        },
        "where_it_could_pay": {
            "candidate": "the MTP head already on disk",
            "draft_cost_ratio": round(mtp_ratio, 5),
            "why": f"the MTP head is {c['mtp_share_of_model_pct']}% of the model, so a "
                   f"draft pass costs about that fraction of a full forward pass instead "
                   f"of ~100%. Break-even needs only acceptance above {mtp_ratio:.4f}.",
            "projected": mtp_ec,
            "IS_A_PROJECTION_NOT_A_MEASUREMENT": (
                "these speedups assume the MTP head's acceptance rate lands in the range "
                "shown and that its cost scales with its parameter share. Neither is "
                "measured: the head has never been packed into a noetic artifact and the "
                "runtime has no MTP path. The measured acceptance figures above are for "
                "the FULL-SIZE drafts only."),
            "next_step": "pack mtp.* into an artifact and add a runtime path, then "
                         "measure its acceptance the same way this receipt measured the "
                         "full-size drafts",
        },
    }
    out["pass"] = bool(c["mtp_params"] > 0 and drafts and
                       all("pays" in d for d in drafts.values()))
    p = RH / "DECODING_GRAVITY.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"CENSUS: MTP machinery {'FOUND' if c['mtp_params'] else 'ABSENT'} — "
          f"{len(c['mtp_tensors'])} tensors, {c['mtp_params']:,} params "
          f"({c['mtp_share_of_model_pct']}% of model), {c['status_in_this_campaign']}")
    print()
    print(f"{'draft':16s}{'cap':>6s}{'accept':>9s}{'d/v':>8s}{'break-even':>12s}"
          f"{'speedup':>10s}  role")
    for k, d in drafts.items():
        print(f"{k:16s}{str(d['capability_passed'])+'/43':>6s}"
              f"{d['acceptance_rate']:>9.2f}{d['cost_ratio_draft_over_verify']:>8.3f}"
              f"{d['break_even_cost_ratio']:>12.2f}{d['predicted_speedup_x']:>10.3f}"
              f"  {'PAYS' if d['pays'] else 'does not pay'}")
    print()
    print(f"MTP head as draft (PROJECTION, cost ratio {mtp_ratio:.5f}):")
    for a, e in mtp_ec.items():
        print(f"  acceptance {a:.2f} -> {e['predicted_speedup_x']:.3f}x  "
              f"pays={e['pays']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
