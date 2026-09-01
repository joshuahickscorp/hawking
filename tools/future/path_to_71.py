#!/usr/bin/env python3
"""Not one promised path to 71. A set of compositions, each with its evidence.

Every component here carries its own evidence class, and the arithmetic keeps
them apart: QUALIFIED means measured on the real resident with parity;
PROSPECTIVE means the byte model is measured but capability is not; SPECULATIVE
means a school is running and no number exists yet.

The honest headline: everything currently QUALIFIED and PROSPECTIVE composed
together does not reach 50 TPS, let alone 71. 71 requires a representation
result that does not exist yet, and this module says so by arithmetic rather
than by hedging.

    python3 tools/future/path_to_71.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "PATH_TO_71.json"

# THE BASELINE IS READ, NOT REMEMBERED. G075 measured the post-widen_f4 body in a
# protected window on a release build: 27.2896 ms, 36.644 TPS, token-identical
# over 48 decode steps. 28.722 was the pre-widen_f4 anchor and is now history.
# S025 §14 and §15: every verified speedup becomes the new parent, and widen_f4
# stays. A rebase is not editing one constant - every row recomputes from here.
# G131/G132 REBASED IT AGAIN, and changed the UNIT. The three levers - widen_f4,
# affine2 bitcast, q4 bitcast - are the SEALED DEFAULT now, measured in a
# protected window at 21.9464 ms GPU / 45.566 TPS with the levers UNSET. And the
# ladder is denominated in GPU ms: every rung below is a GPU-time lever, while
# 27.2896 was a WALL figure. G125 showed the host gap spans 4.7x across windows
# while GPU moves under 0.3%, so subtracting GPU levers from a wall number was
# comparing two different quantities.
ABSOLUTE_REL = "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"

# ALREADY INSIDE THE BASELINE. G126 promoted these three to sealed defaults and
# G131 measured the result at 21.9464 ms, so their saving is spent - subtracting
# them again would count it twice. This is the same double-count
# LEVER_PROMOTION_GATE refuses when it checks that the bitcast lease control was
# widen_f4: once a lever is in the parent, it is not a rung.
PROMOTED_INTO_BASELINE = {
    "deltanet_widen_f4": "G126: HAWKING_QWEN38_DN_STATE=widen_f4 is the default",
    "ba_delta_fusion": "G126: folded by the widen_f4 kernel on the sealed graph",
    "mlp_affine2_bitcast": "G126: HAWKING_AFFINE2_GEO=bitcast is the default",
    "mlp_q4_bitcast": "G126: HAWKING_Q4_UNPACK=bitcast is the default",
}
FALLBACK_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"


def _current_token_ms() -> float:
    p = REPO / ABSOLUTE_REL
    if p.is_file():
        return float(json.loads(p.read_text())["measured"]["gpu_ms_per_token"])
    path = REPO / FALLBACK_REL
    if not path.is_file():
        raise RuntimeError(
            f"neither {ABSOLUTE_REL} nor {FALLBACK_REL} is on disk; the ladder "
            "refuses to compute against a remembered baseline. Run "
            "tools/future/sealed_default_absolute.py --build."
        )
    return float(json.loads(path.read_text())["decode_wall_ms_per_token"])


HISTORICAL_TOKEN_MS = 28.722   # pre-widen_f4, kept so the move is auditable
PREVIOUS_TOKEN_MS = 27.2896    # post-widen_f4 WALL, superseded by G131
TOKEN_MS = _current_token_ms()
# Measured in the same protected window as TOKEN_MS rather than carried from a
# window three rebases ago.
HOST_MS = 0.956
ACTIVE_GB = 9.878901136
MLP_GB = 5.347795776
MLP_MS = 15.541
DELTANET_MS = 8.227

# --- components, each with what is actually known about it -----------------
COMPONENTS: dict[str, dict[str, Any]] = {
    "ba_delta_fusion": {
        "ms_saved": 0.1384, "evidence": "QUALIFIED",
        "parity": "token-identical across 8 runs, 128 tokens, fallbacks 0",
        "source": "receipts/future/BA_DELTA_AB.json",
        "cost": "set one env flag",
    },
    "mlp_dispatch_size": {
        "ms_saved": 1.258, "evidence": "PROSPECTIVE",
        "why_prospective": "the 374.4 GB/s at 337.7 MB/dispatch is measured, but "
                           "no implementation batches the MLP that way yet",
        "parity": "arithmetic is unchanged; a real implementation must prove it",
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
        "cost": "an executor change",
    },
    "aux_group_size_1024": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "the capability screen it was waiting for has now run. A real LS "
               "refit of scale/zero, not an error model: weight-space injury "
               "0.24-0.27 on every screened layer (3, 31, 38, 63), organ output "
               "below the 0.99 cosine bar, and in logit space G=1024 keeps argmax "
               "on only 44% of hold rows with KL 1.23 nats. This rung leaves the "
               "ladder; it was never worth 2.914 ms.",
        "source": "receipts/future/AUX_CAPABILITY_SCREEN.json",
    },
    "aux_group_size_256": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "same screen, same failure mode; MUTUALLY_EXCLUSIVE with "
               "group_size_1024 in any case. Was carried as 2.331 ms.",
        "source": "receipts/future/AUX_CAPABILITY_SCREEN.json",
    },
    "aux_u8": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "it passed its capability screen and then FAILED as an executable. "
               "The native consumer was built and measured: layer 3, "
               "GPUStartTime/GPUEndTime, incumbent 334.5682 GB/s in 249.75 us "
               "against native 269.7453 GB/s in 278.791 us - 29.04 us SLOWER for "
               "8,355,840 FEWER bytes. Decoding a u8 log-scale needs an exp plus "
               "ten int_to_float per group: 2.5 decode-FMA/weight-byte at 4 B/iter "
               "against the incumbent's 1.3333 at 6 B/iter. Billed bytes only it is "
               "+1.5541 ms; billed with aux-decode FLOPs it is -0.3885 ms. A "
               "bytes-only model is not safe on an organ MLP_ISSUE_RATE_LADDER "
               "showed is arithmetic-sensitive. REOPEN_IF a decode exists that adds "
               "no per-group transcendental math - a 256-entry LUT is the obvious "
               "one and is untried.",
        "source": "receipts/future/AUX_U8_NATIVE.json",
    },
    "_aux_u8_superseded": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "placeholder retained so the id count is stable",
        "source": "receipts/future/AUX_U8_NATIVE.json",
        "_orig": "FITTED_HELDOUT",
        "why": "a real u8 encode of the incumbent f16 aux (log scale + linear "
               "bias, 2-bit codes kept), screened in weight, organ AND logit "
               "space. argmax agreement 1.00 AND KL 0.003 - actual parity, which "
               "is why the screen refuses to accept argmax alone. The cheapest "
               "live byte lever left and the first one to survive a capability "
               "screen.",
        "source": "receipts/future/AUX_CAPABILITY_SCREEN.json",
        "cost": "a native consumer, then a dirty A/B",
    },
    "deltanet_widen_f4": {
        "ms_saved": 1.0245, "evidence": "QUALIFIED",
        "parity": "token-id IDENTICAL across 14 runs, 32 tokens, fallbacks 0",
        "why": "MEASURED complete-token A/B under gpu_lane_lock with "
               "MTLCommandBuffer GPUStartTime/GPUEndTime: incumbent 27.4065 ms "
               "against widen_f4 26.3821 ms. Production launches "
               "qwen38_gated_delta_decode_vi_simd_ba_f4 rather than unfused "
               "vi-SIMD plus ba_to_decay. It BEAT its own diagnostic - the "
               "isolated fair cut was 0.7046 ms and the extra 0.32 ms is the "
               "48-launch fold that only appears in the real graph.",
        "source": "receipts/future/DELTANET_WIDEN_AB.json",
        "cost": "landed",
    },
    "_deltanet_widen_f4_old": {
        "ms_saved": 0.0, "evidence": "REFUTED",
        "why": "superseded by the measured A/B on the same id",
        "source": "receipts/future/DELTANET_WIDEN_AB.json",
        "why": "gated-delta layout, measured back-to-back in one process: fused-ba "
               "1.581 ms against widen_f4 0.879 ms on the same work. Production "
               "still launches unfused vi-SIMD, so a 628-graph A/B is required "
               "before this rises above DIRTY_DIAGNOSTIC. Fusing ba_to_decay "
               "alone saves no GPU ms - fused-ba is about equal to unfused delta.",
        "source": "receipts/future/DELTANET_ORGAN_DECOMPOSE.json",
        "cost": "a layout change plus a 628-graph A/B",
    },
    "mlp_entropy_floor": {
        "gb_saved": 0.277697891, "evidence": "PROSPECTIVE",
        "why_prospective": "the 1.87018-of-2-bits floor is measured; an actual "
                           "entropy coder that a GPU can consume in-register does "
                           "not exist and may cost more ALU than it saves bytes",
        "source": "receipts/future/MLP_CODE_INFORMATION.json",
        "cost": "a codec plus a native consumer",
    },
    "mlp_low_rank_program": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "activation-WEIGHTED functional rank, not raw SVD: 10% held-out "
               "output error needs rank 5120 (full) on three of four layers. The "
               "largest rank whose complete ledger is under the incumbent "
               "5,347,795,776 bytes is r=617, and there hold error is 35-84%. "
               "Reaching 10% costs 44.3 GB added.",
        "source": "receipts/future/MLP_FUNCTIONAL_RANK.json",
    },
    "mlp_nonlinear_program": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "six families on the real corpus, all held-out by prompt: "
               "FACTORIZE_THE_FACTORS 0.977, PRODUCT_DICTIONARY 0.997 (0.928 "
               "even with an unfair Y-oracle), NONLINEAR_GENERATOR 0.918, "
               "CONDITIONAL_PROGRAM 0.917, GENERATED_BLOCK 0.943. The MEAN "
               "PREDICTOR on the same hold set is 0.970, so the factorized "
               "program is worse than predicting the mean. Common mechanism: "
               "every one is an r-bottleneck. Reopen is a full-width structured "
               "operator only.",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
    },
    "deltanet_generated_transition": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "the candidate removed 2,139,096,960 bytes and added 4,548,560 - "
               "worth about 5.8 ms/token if the function had held. It does not: "
               "the shared map does not emit this layer's coefficients and the "
               "state diverges at STEP 1, not after a long horizon. "
               "MEASURED_NEGATIVE, economics IMMATERIAL.",
        "source": "receipts/future/DELTANET_GENERATED_TRANSITION.json",
    },
    "mlp_decode_fold_addqx": {
        "ms_saved": 3.9833, "evidence": "DIRTY_DIAGNOSTIC",
        "parity": "token-id IDENTICAL across 7x2 runs (fnv1a64 e04e1b12206475d8, "
                  "128 bytes compared, 0 mismatch) - but layer-0 named-matvec "
                  "output buffers are NOT byte-identical (gate 22309 of 69632 "
                  "bytes differ). APPROX_CANDIDATE, not exact.",
        "why_not_qualified": "faster-but-not-exact. The single-layer probe was "
                             "called bit-identical; at token level it is not. A "
                             "faster resident that answers differently is not the "
                             "same resident, so production default stays "
                             "Affine2Geo::Tpr64 and this is not blended into a "
                             "bit-identical verdict.",
        "measured": "complete-token A/B on the fused 580-graph: incumbent "
                    "26.3026 ms against fold_addqx 22.3192 ms, saved 3.9833 ms. "
                    "The organ cut REACHED the token (3.98 against 3.95 "
                    "isolated). The 1.745 ms one-layer projection did not "
                    "reproduce - it under-predicted by 2.24 ms.",
        "why": "BIT-IDENTICAL on the probed layer. 370.9 GB/s against production "
               "329.2, 1.127x, by folding the affine into integer adds for q.x and "
               "applying s and b once per group: decode FMA/byte 1.3333 -> 0.3333. "
               "Projection over a single-layer probe, not a resident measurement: "
               "MLP 15.541 -> 13.796 ms. It closes only 24.8% of the 329.2->497.4 "
               "gap, because an EIGHTFOLD cut in total FMA/byte bought 12.7% while "
               "ARM A bought 50.9% - FMA count is not the ceiling.",
        "source": "receipts/future/MLP_DECODE_CHEAPEN.json",
        "cost": "a kernel change plus a complete-token A/B",
    },
    "deltanet_q3": {
        "gb_saved": 0.503316480, "evidence": "REFUTED",
        "why": "entropies are 3.465-3.479 of 4 across q/k/v/z and any_supported "
               "is false; no sensitivity measurement licenses the reduction",
        "source": "receipts/future/DELTANET_QKVZ_PRECISION.json",
    },
}

SCHOOLS_RUNNING = (
    "mlp full-width structured operator (Monarch/butterfly/distilled control) - "
    "the only reopen left after six r-bottleneck families died",
    "mlp error budget: what relative output error the model actually tolerates",
    "native consumer for aux_u8, the one byte lever that passed its screen",
    "cheapen the MLP decode arithmetic: 1.33 FMA/byte down to 0.88",
    "the DeviceCompiler that the Odyssey NR-NX path now blocks on",
    "deltanet gated-delta widen_f4 layout, 0.70 ms measured back-to-back",
    "is the MLP kernel ALU-bound or memory-bound",
    "deltanet multi-step authority (the instrument, after the one-step candidate died)",
    "sparse residual concentration curve",
)


# Evidence tiers, weakest first. A composed path is only as strong as its weakest
# component, and this used to be computed by checking for the literal string
# "PROSPECTIVE" - so ANY other non-qualified tier (DIRTY_DIAGNOSTIC,
# FITTED_HELDOUT, NATIVE_UNMEASURED, MEASURED) silently reported as QUALIFIED.
# That is an over-claim mechanism sitting in the campaign's headline artifact, so
# an unknown tier now raises instead of defaulting to the strongest one.
EVIDENCE_ORDER: tuple[str, ...] = (
    "REFUTED",
    "PROSPECTIVE_ECONOMIC",
    "PROSPECTIVE",
    "FUNCTIONAL_ORACLE",
    "FITTED_HELDOUT",
    "NATIVE_UNMEASURED",
    "DIRTY_DIAGNOSTIC",
    "MEASURED",
    "QUALIFIED",
)


class UnknownEvidenceTier(ValueError):
    """A tier nobody ranked must not be silently treated as the strongest."""


def _weakest(tiers: list[str]) -> str:
    if not tiers:
        return "QUALIFIED"
    for t in tiers:
        if t not in EVIDENCE_ORDER:
            raise UnknownEvidenceTier(
                f"{t!r} is not in EVIDENCE_ORDER; rank it rather than letting a "
                f"composed path inherit QUALIFIED from an unranked component"
            )
    return min(tiers, key=EVIDENCE_ORDER.index)


def _apply(ms: float, gb_removed: float, comp_ms: float) -> float:
    """Bytes come off the MLP at the MLP's own measured rate."""
    mlp_after = MLP_MS * (MLP_GB - gb_removed) / MLP_GB
    return ms - (MLP_MS - mlp_after) - comp_ms


def compose(ids: list[str]) -> dict[str, Any]:
    gb = 0.0
    ms = 0.0
    seen: set[str] = set()
    used: list[str] = []
    skipped: list[str] = []
    for cid in ids:
        c = COMPONENTS[cid]
        if cid in PROMOTED_INTO_BASELINE:
            skipped.append(f"{cid}: ALREADY IN THE BASELINE - "
                           f"{PROMOTED_INTO_BASELINE[cid]}")
            continue
        if c["evidence"] == "REFUTED":
            skipped.append(f"{cid}: REFUTED")
            continue
        if any(o in seen for o in c.get("overlaps", [])):
            skipped.append(f"{cid}: overlaps {c['overlaps']}, not additive")
            continue
        seen.add(cid)
        used.append(cid)
        gb += c.get("gb_saved", 0.0)
        ms += c.get("ms_saved", 0.0)
    total = _apply(TOKEN_MS, gb, ms)
    worst = _weakest([COMPONENTS[cid]["evidence"] for cid in used])
    return {
        "components": used,
        "skipped": skipped,
        "gb_removed": round(gb, 4),
        "ms_removed_directly": round(ms, 4),
        "token_ms": round(total, 3),
        "tps": round(1000.0 / total, 2),
        "weakest_evidence": worst,
    }


def paths() -> list[dict[str, Any]]:
    now = {"path": "PATH_00", "label": "measured now",
           "token_ms": TOKEN_MS, "tps": round(1000.0 / TOKEN_MS, 2),
           "weakest_evidence": "MEASURED", "components": [], "skipped": []}
    rows = [now]
    for pid, label, ids in (
        ("PATH_01", "everything QUALIFIED today",
         ["ba_delta_fusion", "deltanet_widen_f4"]),
        ("PATH_02", "qualified + the decode fold (APPROX, not exact)",
         ["ba_delta_fusion", "deltanet_widen_f4", "mlp_decode_fold_addqx"]),
        ("PATH_03", "that + the executor lever that survived",
         ["ba_delta_fusion", "deltanet_widen_f4", "mlp_decode_fold_addqx",
          "mlp_dispatch_size"]),
        ("PATH_04", "everything on record, refuted excluded",
         ["ba_delta_fusion", "deltanet_widen_f4", "mlp_decode_fold_addqx",
          "mlp_dispatch_size", "aux_group_size_1024", "aux_u8",
          "mlp_entropy_floor", "deltanet_q3"]),
    ):
        r = compose(list(ids))
        r["path"] = pid
        r["label"] = label
        rows.append(r)
    return rows


def gap_to_71() -> dict[str, Any]:
    best = max(paths(), key=lambda r: r["tps"])
    need_ms = 1000.0 / 71.0
    # After PATH_04, how much MORE has to go, at current executor efficiency?
    remaining = best["token_ms"] - need_ms
    gpu_after = best["token_ms"] - HOST_MS
    gpu_needed = need_ms - HOST_MS
    return {
        "best_composed_path": best["path"],
        "best_composed_tps": best["tps"],
        "best_composed_token_ms": best["token_ms"],
        "target_token_ms": round(need_ms, 3),
        "still_to_remove_ms": round(remaining, 3),
        "still_to_remove_share_of_gpu": round(1 - gpu_needed / gpu_after, 4),
        "verdict": (
            "Everything on record — every qualified win, every prospective byte "
            f"lever, refuted ones excluded — reaches {best['tps']} TPS. Reaching "
            f"71 needs another {remaining:.2f} ms, which is "
            f"{(1 - gpu_needed / gpu_after) * 100:.0f}% of the GPU time that "
            "would remain. No component on record can supply that. It requires a "
            "representation result that does not exist yet."
        ),
        "what_would_supply_it": (
            "A functional replacement for the MLP or DeltaNet that eliminates "
            "information rather than coding it better. Both organs are at their "
            "entropy floor, so the only remaining source of that magnitude is "
            "weights that need not exist independently."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.path_to_71.v1",
        "version": 1,
        "recorded_by": "tools/future/path_to_71.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "baseline": {"token_ms": TOKEN_MS, "host_ms": HOST_MS,
                     "tps": round(1000.0 / TOKEN_MS, 2), "active_gb": ACTIVE_GB},
        "components": COMPONENTS,
        "paths": paths(),
        "gap_to_71": gap_to_71(),
        "schools_running": list(SCHOOLS_RUNNING),
        "claim_boundary": (
            "Arithmetic over recorded components. QUALIFIED means measured on the "
            "real resident with parity. PROSPECTIVE means the byte model is "
            "measured and capability is NOT — no generate gate has been run on "
            "any auxiliary lever. Overlapping levers are not summed. A composed "
            "path is a prediction, not a result, and every one above PATH_01 "
            "contains at least one component nobody has implemented."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    # AN UNKNOWN FLAG IS A REFUSAL, NOT A NO-OP. This CLI took --record and
    # silently ignored everything else, so `--build` - the verb every other
    # module in tools/future uses - printed the new table, exited 0, and wrote
    # NOTHING. The receipt stayed stale while the terminal showed fresh numbers.
    # A tool that reports success without doing the work is the exact failure
    # this campaign keeps finding in its own checks.
    from _common import require_known_flags
    require_known_flags(["--build", "--record"])
    d = build()
    if "--record" in sys.argv or "--build" in sys.argv:
        print(f"wrote {record()}")
    for r in d["paths"]:
        print(f"  {r['path']}  {r['tps']:6.2f} TPS  {r['token_ms']:7.3f} ms  "
              f"[{r['weakest_evidence']:11s}] {r['label']}")
    g = d["gap_to_71"]
    print(f"\n  gap: {g['still_to_remove_ms']} ms more, "
          f"{g['still_to_remove_share_of_gpu'] * 100:.0f}% of remaining GPU time")
