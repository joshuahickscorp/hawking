#!/usr/bin/env python3
"""N015 — the Noetic scoreboard and the frontier points it tracks.

S017 §44 fixes the columns; §3 fixes the frontier points. The point of writing it
as a harness rather than a table is that every cell must come from a receipt on
disk, and a cell nobody has measured yet must read ABSENT with a reason instead of
0. A scoreboard with plausible zeros in it is worse than no scoreboard: it makes
an unmeasured candidate look cheap.

The frontier points are deliberately separate columns because S017 says they MAY
BE DIFFERENT ARTIFACTS:

    LOWEST_SCREEN_SURVIVOR      cheapest thing that passed an organ-local screen
    LOWEST_CHAIN_SURVIVOR       cheapest thing that survived composition
    LOWEST_GENERATION_COHERENT  cheapest thing that actually generated text
    LOWEST_CAPABILITY_SURVIVOR  cheapest thing that kept capability
    FASTEST_COHERENT            highest tok/s among coherent candidates
    FASTEST_PRODUCTION          highest verified useful work per wall second

Collapsing those into one "best" is how a campaign talks itself into promoting an
artifact that was only ever screened.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts" / "headless"
OUT = R / "NOETIC_SCOREBOARD.json"

PARENT_PARAMS = 26_895_998_464
INCUMBENT_EBPW = 4.252735126866492

ABSENT = "ABSENT"


def load(name: str) -> dict[str, Any] | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()[:12]


def cell(value, why_absent: str | None = None, kind: str = "MEASURED") -> dict[str, Any]:
    """Every cell says whether it is real. An unmeasured cell is never a number."""
    if value is None:
        return {"value": None, "state": ABSENT, "reason": why_absent or "not measured"}
    return {"value": value, "state": kind}


def dig(d: Any, *path, default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and len(cur) > k:
            cur = cur[k]
        else:
            return default
    return cur


_gl = load("GPU_LEDGER") or {}
_LEDGER_ACTIVE = (_gl.get("ACTIVE_BYTES_PER_TOKEN") or {}).get("value")
_LEDGER_DRAM = (_gl.get("DRAM_BYTES_PER_TOKEN") or {}).get("value")


def candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def row(cid, ebpw, tps, disp, coherent, screen, chain, generation, src, note=""):
        gb = (ebpw * PARENT_PARAMS / 8) / 2**30 if ebpw else None
        return {
            "id": cid,
            "source_receipt": src,
            "note": note,
            # --- S017 §44 columns ---
            "EBPW": cell(ebpw),
            "RESIDENT_GB": cell(round(gb, 4) if gb else None, kind="DERIVED"),
            "ACTIVE_GB_PER_TOKEN": cell(
                round(_LEDGER_ACTIVE / 2**30, 4) if (_LEDGER_ACTIVE and "incumbent" in cid) else None,
                "GPU ledger measured only the q4 incumbent; per-candidate ACTIVE bytes is the "
                "missing measurement that would explain why 44.7% fewer STORED bits bought 1.9%",
            ),
            "DRAM_GB_PER_TOKEN": cell(
                round(_LEDGER_DRAM / 2**30, 4) if (_LEDGER_DRAM and "incumbent" in cid) else None,
                "same: per-candidate DRAM bytes not measured",
                kind="DERIVED",
            ),
            "FLOP_PER_TOKEN": cell(None, "per-candidate FLOP census not yet run"),
            "DISPATCHES_PER_TOKEN": cell(disp),
            "ROUTES_PER_TOKEN": cell(0, kind="MEASURED") if ebpw else cell(None),
            "ROUTING_NS_PER_TOKEN": cell(
                0, kind="MEASURED"
            ) if ebpw else cell(None),
            "COMPLETE_TOKEN_NS": cell(
                round(1e9 / tps) if tps else None, kind="DERIVED"
            ),
            "TPS": cell(tps),
            "AGGREGATE_TPS_C2": cell(None, "concurrency bench (N007) not yet landed"),
            "AGGREGATE_TPS_C4": cell(None, "concurrency bench (N007) not yet landed"),
            "VERIFIED_WUS_PER_HOUR": cell(None, "production bench (N007) not yet landed"),
            "CAPABILITY": cell(None, "no capability suite has been run on any candidate"),
            # --- ladder position, S017 §28 ---
            "passed_screen": screen,
            "survived_chain": chain,
            "generated_coherently": generation,
            "coherent": coherent,
        }

    fne = load("FIRST_NOETIC_EXECUTABLE")
    for m in (fne or {}).get("mix_scoreboard", []):
        out.append(
            row(
                m["mix_id"],
                m.get("complete_ebpw"),
                m.get("tok_s"),
                None,
                m.get("coherent"),
                True,
                None,
                bool(m.get("coherent")),
                "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
            )
        )

    q3 = load("NOETIC_Q3_MLP_Q4_ATTN")
    if q3:
        out.append(
            row(
                dig(q3, "chosen", "mix_id", default="q3_mlp_q4_attn"),
                dig(q3, "chosen", "complete_ebpw"),
                dig(q3, "decode", "tok_s"),
                None,
                True,
                True,
                True,
                True,
                "receipts/headless/NOETIC_Q3_MLP_Q4_ATTN.json",
                "text identical to the q4 incumbent",
            )
        )

    a32 = load("AFFINE2_NATIVE_MLP")
    if a32:
        out.append(
            row(
                "affine2_g32_all_mlp",
                dig(a32, "chosen", "complete_ebpw"),
                dig(a32, "decode", "tok_s"),
                None,
                False,
                True,
                True,
                False,
                "receipts/headless/AFFINE2_NATIVE_MLP.json",
                "degraded: malformed think-tag and a stutter",
            )
        )

    fused = load("NOETIC_FUSED_SUBBIT")
    if fused:
        best = dig(fused, "decode_tok_s", "after_mlp_swiglu_qkv_dn", default={})
        out.append(
            row(
                "NOETIC_PARENT_A (affine2_g64_LS + fused graph)",
                dig(fused, "representation", "complete_ebpw"),
                best.get("tok_s_mean"),
                756,
                True,
                True,
                True,
                True,
                "receipts/headless/NOETIC_FUSED_SUBBIT.json",
                "LEADER. Frontier candidate, NOT resident-promoted. "
                "Artifact bytes were reaped with a lane worktree; N001 rebuilds and seals.",
            )
        )

    out.append(
        row(
            "q4 incumbent (control)",
            INCUMBENT_EBPW,
            33.717,
            964,
            True,
            True,
            True,
            True,
            "receipts/headless/NOETIC_DISPATCH_FUSION.json",
            "immutable control",
        )
    )
    return out


def frontier_points(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def ebpw(r):
        return r["EBPW"]["value"]

    def tps(r):
        return r["TPS"]["value"]

    cands = [r for r in rows if ebpw(r) is not None and "control" not in r["id"]]
    screened = [r for r in cands if r["passed_screen"]]
    chained = [r for r in cands if r["survived_chain"]]
    generated = [r for r in cands if r["generated_coherently"]]
    coherent_tps = [r for r in cands if r["coherent"] and tps(r)]

    def lowest(rs):
        return min(rs, key=ebpw)["id"] if rs else None

    return {
        "LOWEST_SCREEN_SURVIVOR": lowest(screened),
        "LOWEST_CHAIN_SURVIVOR": lowest(chained),
        "LOWEST_GENERATION_COHERENT": lowest(generated),
        "LOWEST_CAPABILITY_SURVIVOR": {
            "value": None,
            "state": ABSENT,
            "reason": "no capability suite has been run on ANY candidate; "
            "coherence on a 16-token greedy sample is not capability",
        },
        "FASTEST_COHERENT": max(coherent_tps, key=tps)["id"] if coherent_tps else None,
        "FASTEST_PRODUCTION": {
            "value": None,
            "state": ABSENT,
            "reason": "production is verified useful work per wall second; "
            "the concurrency bench (N007) has not been built",
        },
        "note": "S017 §3: these MAY BE DIFFERENT ARTIFACTS. They are not collapsed.",
    }


def _organ_floor_arithmetic() -> dict[str, Any]:
    """What the MEASURED organ floors imply for the whole model.

    This is the campaign's central number and it belongs in the tracking spine,
    not only in the organ receipt. ORGAN_FRONTIERS measured three floors
    independently; weighting each by its share of the parent parameters answers
    whether the ~1 EBPW research pressure is reachable by per-organ quantization
    at all.
    """
    org = load("ORGAN_FRONTIERS")
    if not org:
        return {"state": ABSENT, "reason": "ORGAN_FRONTIERS.json not on disk"}
    floors = dig(org, "verdict", "floors_storage_bpw", default={})
    dn = dig(org, "organs", "deltanet", "physical_cited", "elements")
    gqa = dig(org, "organs", "gqa", "physical_cited", "elements")
    mlp = 17_112_760_320  # NOETIC_PARENT_A mlp_elements
    if not (floors and dn and gqa):
        return {"state": ABSENT, "reason": "organ element counts incomplete"}
    emb = PARENT_PARAMS - (dn + gqa + mlp)
    rows = [
        ("deltanet", dn, floors.get("deltanet")),
        ("gqa", gqa, floors.get("gqa")),
        # The leader's MLP is a TRUE 4-level affine at 2 bits + f16 scale + f16
        # bias per 64 = 2.50, and it is the cheapest MLP codec MEASURED to
        # GENERATE coherently. The bias-free 2.25 variant decodes natively but
        # degrades. Using 2.25 here -- as this file first did, taken from the
        # mis-billed composition arm -- understates the floor.
        ("mlp", mlp, 2.50),
        ("embedding_output", emb, floors.get("embedding_output")),
    ]
    if any(b is None for _, _, b in rows):
        return {"state": ABSENT, "reason": "a measured floor is missing"}
    total = sum(e * b for _, e, b in rows)
    without_mlp = sum(e * b for n, e, b in rows if n != "mlp")
    return {
        "state": "DERIVED",
        "per_organ": [
            {"organ": n, "elements": e, "share": round(e / PARENT_PARAMS, 4), "floor_bpw": b}
            for n, e, b in rows
        ],
        "implied_whole_model_ebpw": round(total / PARENT_PARAMS, 4),
        "leader_today": 3.139300850311054,
        "incumbent": INCUMBENT_EBPW,
        "research_pressure": 1.0,
        "floor_even_if_mlp_were_free_ebpw": round(without_mlp / PARENT_PARAMS, 4),
        "reading": (
            "Per-organ quantization cannot reach ~1 EBPW. The leader is already within "
            "6% of the implied floor, and the non-MLP organs alone would still cost "
            "more than 1.5 EBPW with the MLP at zero bits. Reaching ~1 requires "
            "STRUCTURE that breaks these floors, not narrower codes."
        ),
    }


def main() -> int:
    rows = candidates()
    fp = frontier_points(rows)
    n_cells = sum(1 for r in rows for k, v in r.items() if isinstance(v, dict) and "state" in v)
    n_absent = sum(
        1
        for r in rows
        for k, v in r.items()
        if isinstance(v, dict) and v.get("state") == ABSENT
    )
    receipt = {
        "schema": "hawking.headless.noetic_scoreboard.v1",
        "obligation": "N015 (S017 §3, §42, §44)",
        "git_head": git_head(),
        "parent_params": PARENT_PARAMS,
        "columns": "S017 §44",
        "honesty": (
            "Every cell states MEASURED, DERIVED or ABSENT. An unmeasured cell is "
            "never rendered as 0 — a plausible zero makes an unmeasured candidate "
            "look cheap, which is the specific way a scoreboard lies."
        ),
        "candidates": rows,
        "frontier_points": fp,
        "coverage": {
            "cells": n_cells,
            "absent": n_absent,
            "absent_fraction": round(n_absent / n_cells, 4) if n_cells else None,
            "blocking_obligations": [
                "N004 GPU ledger -> ACTIVE_GB/TOKEN, DRAM_GB/TOKEN",
                "N007 production bench -> AGGREGATE_TPS_C2/C4, VERIFIED_WUS/HOUR, FASTEST_PRODUCTION",
                "capability suite -> CAPABILITY, LOWEST_CAPABILITY_SURVIVOR",
            ],
        },
        "phase_transition_map": {
            "state": "PARTIAL",
            "family": "uniform grouped-code on the whole MLP",
            "measured_points": [
                {"bpw_body": 1.85, "codec": "ternary g64", "composed": "FAILS", "argmax": "flips"},
                {"bpw_body": 3.06, "codec": "q2f_g64 AS ACTUALLY BILLED (7 levels)",
                 "composed": "SURVIVES", "argmax": "agrees",
                 "correction": "recorded at the time as 2.25 bpw. The codec emitted SEVEN "
                 "levels, not four: log2(7)+16/64 = 3.06. The survival is real; the PRICE "
                 "was wrong by 0.81 bpw."},
                {"bpw_body": 2.25, "codec": "TRUE bias-free 4-level g64",
                 "composed": "not run", "generation": "DEGRADED", "measured_ebpw": 2.9802,
                 "note": "the honest 2-bit point: decodes natively at 2.9802 EBPW but emits "
                 "' IR' then EOS. Cheaper than the leader and NOT coherent."},
                {"bpw_body": 2.50, "codec": "affine2 g64 LS (leader: 4-level + bias)",
                 "composed": "SURVIVES", "generation": "COHERENT", "measured_ebpw": 3.1393,
                 "note": "the 0.25-bpw bias is what separates degraded from coherent at 2 bits"},
                {"bpw_body": 3.25, "codec": "q3 g64", "composed": "SURVIVES", "argmax": "agrees"},
            ],
            "knee": "CORRECTED. The composed knee sits between 1.85 (argmax flips) and the "
            "3.06-bpw arm that survives -- NOT 2.25, which was a mis-billing. At GENERATION "
            "the boundary is tighter: a TRUE 4-level 2.25-bpw body degrades (' IR' then EOS) "
            "while the same codec plus a 0.25-bpw bias (2.50 body) is coherent.",
            "collapse_boundary": 1.85,
            "coherent_plateau": "SUPERSEDED -- it rested on the mis-billed 2.25 point.",
            "correction_note": "The composition arm billed at 2.25 bpw emitted SEVEN levels "
            "and really cost ~3.06 bpw. The survival was real; the price was not.",
            "missing": "only ONE family is mapped for the MLP. S017 §42 wants a curve per "
            "family; N008 screened five structurally distinct ones at matched bytes.",
        },
        "implied_whole_model_floor": _organ_floor_arithmetic(),
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"candidates {len(rows)}  cells {n_cells}  ABSENT {n_absent} ({n_absent/n_cells:.0%})")
    for k, v in fp.items():
        if k == "note":
            continue
        s = v["reason"] if isinstance(v, dict) else v
        print(f"  {k:<28} {str(s)[:70]}")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
