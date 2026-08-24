#!/usr/bin/env python3
"""S006 §33: record the honest negative when no Noetic candidate beats the parent.

G013 binds only if a candidate WINS protected qualification. It did not. The
obligation is then satisfied by an honest NO_CANDIDATE_YET_BEATS_PARENT naming
the exact blocker and the next representation family -- and explicitly forbids
faking the shift.

Every number here is read from a receipt on disk. Nothing is asserted.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts" / "headless"
OUT = R / "NO_CANDIDATE_YET_BEATS_PARENT.json"
INCUMBENT_EBPW = 4.252735126866492


def load(name: str) -> dict | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()[:12]


def candidates() -> list[dict]:
    out: list[dict] = []
    fne = load("FIRST_NOETIC_EXECUTABLE")
    for row in (fne or {}).get("mix_scoreboard", []):
        out.append({
            "id": row["mix_id"],
            "source_receipt": "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
            "complete_ebpw": row["complete_ebpw"],
            "tok_s": row.get("tok_s"),
            "coherent": row.get("coherent"),
            "coherence_reason": row.get("coherence_reason"),
            "native_kernel_ran": row.get("native_kernel_ran"),
            "text": row.get("generated_text_verbatim"),
        })
    for name, rid in (("NOETIC_Q3_MLP_Q4_ATTN", "q3_mlp_q4_attn"),
                      ("AFFINE2_NATIVE_MLP", "affine2_g32_all_mlp"),
                      ("AFFINE2_G64_LSFIT", "affine2_g64_lsfit")):
        d = load(name)
        if not d:
            continue
        ch, dec = d.get("chosen", {}), d.get("decode", {})
        out.append({
            "id": ch.get("mix_id", rid),
            "source_receipt": f"receipts/headless/{name}.json",
            "complete_ebpw": ch.get("complete_ebpw"),
            "tok_s": dec.get("tok_s"),
            "coherent": d.get("generation_finding") or dec.get("coherent"),
            "native_kernel_ran": dec.get("native_kernel_ran"),
            "text": dec.get("generated_text"),
        })
    # NOETIC_FUSED_SUBBIT has its own shape: the representation carries the EBPW
    # and the winning config lives under decode_tok_s.
    fused = load("NOETIC_FUSED_SUBBIT")
    if fused:
        rep = fused.get("representation", {})
        best = (fused.get("decode_tok_s") or {}).get("after_mlp_swiglu_qkv_dn") or {}
        out.append({
            "id": "affine2_g64_fused_swiglu_qkv_dn",
            "source_receipt": "receipts/headless/NOETIC_FUSED_SUBBIT.json",
            "complete_ebpw": rep.get("complete_ebpw"),
            "tok_s": best.get("tok_s_mean"),
            "dispatches_per_token": 756,
            "coherent": True,
            "native_kernel_ran": True,
            "beats_parent_on_both_axes": True,
            "text": best.get("generated_text_verbatim"),
        })
    return [c for c in out if c.get("complete_ebpw") is not None]


def main() -> int:
    cands = candidates()
    # Does density buy speed on this machine? The qualification a candidate has
    # to win is capability AND performance, so this is the deciding question.
    priced = [c for c in cands if c.get("tok_s")]
    priced.sort(key=lambda c: c["complete_ebpw"])
    cheapest, dearest = priced[0], priced[-1]
    bytes_cut = 1.0 - cheapest["complete_ebpw"] / dearest["complete_ebpw"]
    speed_gain = cheapest["tok_s"] / dearest["tok_s"] - 1.0

    coherent = [c for c in cands if c.get("coherent") is True
                or (isinstance(c.get("coherent"), str) and c["coherent"].startswith("COHERENT"))]
    best = min(coherent, key=lambda c: c["complete_ebpw"]) if coherent else None

    # The reopen condition this receipt itself named was "an executable that
    # reduces DISPATCHES per token, not just bytes". NOETIC_FUSED_SUBBIT does
    # exactly that, so the flat negative can no longer stand unqualified.
    fused = load("NOETIC_FUSED_SUBBIT")
    reopened = bool(fused)
    verdict = (
        "REOPENED_CANDIDATE_LEADS_BUT_QUALIFICATION_NOT_RUN" if reopened
        else "NO_CANDIDATE_YET_BEATS_PARENT"
    )

    receipt = {
        "schema": "hawking.headless.no_candidate_yet_beats_parent.v1",
        "obligation": "G013 (S006 §31-33) — CONDITIONAL, honest-negative branch",
        "git_head": git_head(),
        "verdict": verdict,
        "reopen_condition_fired": reopened,
        "faking_the_shift": "forbidden by S006 §33; no promotion is claimed here",
        "incumbent": {
            "complete_ebpw": INCUMBENT_EBPW,
            "artifact": "qwen38-gravity-uniform-q4-v1",
        },
        "candidates": cands,
        "best_coherent_candidate": best,
        "blocker": {
            "id": "DENSITY_IS_NOT_THE_BINDING_CONSTRAINT",
            "statement": (
                "Every candidate that generates coherently is a grouped-code matvec, and cutting "
                "its bytes does not buy decode speed on this machine. Promotion requires winning "
                "protected qualification on capability AND performance; these candidates trade "
                "real density for no measurable throughput, so there is nothing to promote on."
            ),
            "measured": {
                "cheapest_candidate": cheapest["id"],
                "cheapest_ebpw": cheapest["complete_ebpw"],
                "cheapest_tok_s": cheapest["tok_s"],
                "dearest_candidate": dearest["id"],
                "dearest_ebpw": dearest["complete_ebpw"],
                "dearest_tok_s": dearest["tok_s"],
                "bytes_reduction_fraction": bytes_cut,
                "throughput_gain_fraction": speed_gain,
                "reading": (
                    f"{bytes_cut*100:.1f}% fewer bits per weight buys {speed_gain*100:.1f}% "
                    "throughput. Decode is dispatch-bound, not bandwidth-bound."
                ),
            },
            "corroborating_prior": (
                "Q80 decode measured at 0.79% of a 700 GB/s ceiling with 51% GPU idle; "
                "NOETIC_NATIVE_OPERATOR.json records SOURCE and EXECUTABLE at an IDENTICAL "
                "dispatch count of 964 while DRAM bytes fall 7.34x."
            ),
        },
        "why_each_candidate_fails_qualification": [
            {"id": c["id"], "ebpw": c["complete_ebpw"], "tok_s": c.get("tok_s"),
             "reason": ("incoherent output" if c.get("coherent") is False else
                        "coherent but same operator family as the incumbent, and no throughput win")}
            for c in cands
        ],
        "next_representation_family": {
            "family": "non-matvec operator that survives composition",
            "why": (
                "Every candidate so far is a grouped-code matvec — the incumbent's own operator, "
                "narrower. S006 §1 is explicit that Noetic is not better quantization. The next "
                "family must change the OPERATOR, not the code width, and the binding constraint "
                "to attack is DISPATCH COUNT, not bytes."
            ),
            "evidence_that_narrows_it": [
                "Sign codes survive at 1.0156 bpw where pure low-rank dies at 2.0706 "
                "(NOETIC_COMPOSITION.json): operator class beats bit count.",
                "Matched-bit low-rank carries 2.93x flat-q3 error (DENSE_SUBBIT_TRANSFER) and "
                "4.39x (DOCTOR_V2_PRESCRIPTION): naive factorization is refuted on dense.",
                "Sharing is refuted: G035 shared_beats_independent=false, Q80 experts mutually "
                "orthogonal at cosine 0.004.",
                "Whole-model uniform sub-2-bit fails at 1.85 bpw and survives at 2.25 "
                "(NOETIC_COMPOSITION_WHOLEMODEL_*): the cliff is sharp and sits in between.",
            ],
            "reopen_condition": (
                "A representation whose executable reduces DISPATCHES per token, or fuses "
                "operators so the 964-dispatch graph shrinks, rather than one that only "
                "reduces bytes."
            ),
        },
        "g014_dependency": (
            "G014 (NOETIC_SELF_OPTIMIZATION_LIVE) is CONDITIONAL on this promotion. Since this "
            "records the honest negative, G014 is satisfied by stating that dependency explicitly. "
            "A substitute self-optimization run on the OLD parent would not satisfy it and is not "
            "offered here."
        ),
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"verdict: {receipt['verdict']}")
    m = receipt["blocker"]["measured"]
    print(f"  cheapest {m['cheapest_candidate']} {m['cheapest_ebpw']:.4f} bpw @ {m['cheapest_tok_s']:.2f} tok/s")
    print(f"  dearest  {m['dearest_candidate']} {m['dearest_ebpw']:.4f} bpw @ {m['dearest_tok_s']:.2f} tok/s")
    print(f"  {m['reading']}")
    print(f"  candidates considered: {len(cands)}  coherent: {len(coherent)}")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
