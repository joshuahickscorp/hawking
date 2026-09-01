#!/usr/bin/env python3
"""The three TPS explanations that were tested and killed.

Three lanes each went after one step of the bandwidth loss chain, and all
three falsified their own angle. That is successful science, not three failed
lanes, and it is worth more than a confirmation would have been: it moves the
whole campaign off "make the same byte stream stream faster" and onto "why
does this token require these bytes and these operations at all".

The point of writing them down is that a future HCLI must not rediscover them
unchanged. Each entry carries the evidence, the exact scope it holds over, the
falsifier that would reopen it, and — the part that matters — the redirect.

    python3 tools/future/tps_falsifications.py --record

Emits a JSON receipt with the full record, and a JSONL scar feed registered in
tools/future/negative_index.py SEED_SOURCES so `refuse_if_dead` reaches it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402, require_known_flags

RECEIPT = REPO / "receipts" / "future" / "TPS_FALSIFICATIONS.json"
SCARS = REPO / "receipts" / "future" / "TPS_FALSIFICATIONS.jsonl"
SCHEMA = "hawking.future.tps_falsifications.v1"

MACHINE = "M3 Ultra 96GB / 28c — Apple Silicon unified memory"
MODEL = "qwen3.8-27b sealed-3.14 resident (physical_ebpw 3.1393)"

# The loss chain these three sit on. Every number here is from a committed
# receipt, not from this module.
CHAIN: tuple[dict[str, Any], ...] = (
    {"gb_s": 819.0, "what": "published M3 Ultra peak", "share_of_peak": 1.000},
    {"gb_s": 703.5, "what": "single GEMV clean addressing", "share_of_peak": 0.859},
    {"gb_s": 530.7, "what": "production catalog addressing", "share_of_peak": 0.648},
    {"gb_s": 513.0, "what": "production catalog decode", "share_of_peak": 0.626},
    {"gb_s": 337.3, "what": "actual production decode", "share_of_peak": 0.412},
)

ACTIVE_WEIGHT_BYTES_PER_TOKEN = 9_878_901_136

FALSIFICATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "CATALOG_ADDRESSING_NOT_PRIMARY_703_530_CAUSE",
        "lane": "L40",
        "step_attacked": "703.5 -> 530.7 GB/s",
        "hypothesis": (
            "The production catalog's host-side weight addressing indirection "
            "is what costs the 24% between a clean single GEMV and production "
            "catalog addressing."
        ),
        "verdict": "FALSIFIED",
        "evidence": (
            "The 24% did not follow the catalog indirection. Isolating host "
            "catalog lookup off the GPU addr-probe timestamp left the gap in "
            "place. What the 24% tracks instead is dispatch topology and mixed "
            "organ sizes: the marginal cost of a dispatch (~15us for this "
            "host/catalog ceremony class) multiplied by how many the token "
            "issues, plus the fact that organs of very different sizes do not "
            "amortise a dispatch equally."
        ),
        "scope": "this machine, this resident, the production catalog path",
        "tested_shapes": "production catalog decode vs isolated single GEMV",
        "falsifier_that_would_reopen_it": (
            "A catalog change that leaves dispatch count and organ size mix "
            "unchanged and still moves the 703->530 step."
        ),
        "redirect": (
            "Attack dispatch COUNT and the organ size mix, not the addressing "
            "layer. Measure the current exact production dispatch count first "
            "and do not reuse a historical one."
        ),
        "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
        "patch_preserved_unapplied": "receipts/future/patches/l40-catalog-addressing.crate.patch",
    },
    {
        "id": "TESTED_GEOMETRY_NOT_PRIMARY_530_337_CAUSE",
        "lane": "L41",
        "step_attacked": "530.7 -> 337.3 GB/s",
        "hypothesis": (
            "The tpr64/tg128 kernel geometry is mis-tuned for these shapes, "
            "and occupancy/coalescing is what costs the 530->337 step."
        ),
        "verdict": "FALSIFIED",
        "evidence": (
            "Parity holds across K=5120, 6144 and 17408 with occupancy "
            "unchanged; the geometry is not mis-tuned for the shapes it runs "
            "on. The lane's own handoff names two other things: affine2-vs-q4 "
            "ALU cost in the low-bit decode, and the fact that MLP is ~54% of "
            "the bytes, so the shape that dominates is not the one being tuned."
        ),
        "scope": (
            "the tested geometries only. This does NOT say kernel geometry can "
            "never matter — it says the geometries tested at these K values are "
            "not where this step went."
        ),
        "tested_shapes": "K=5120, K=6144, K=17408; tpr64 / tg128",
        "falsifier_that_would_reopen_it": (
            "An untested geometry family, or a shape not covered by K in "
            "{5120, 6144, 17408}, that moves the 530->337 step."
        ),
        "redirect": (
            "Go at low-bit decode ALU cost (affine2 vs q4) and at MLP bytes. "
            "The 54% share means MLP representation, not MLP kernel tuning, is "
            "the larger lever."
        ),
        "source_receipt": "receipts/future/KERNEL_GEOMETRY.json",
        "patch_preserved_unapplied": "receipts/future/patches/l41-kernel-geometry.crate.patch",
    },
    {
        "id": "COMPLETE_VS_DECODE_GAP_PREFILL_ACCOUNTING_NOT_CEREMONY",
        "lane": "L42",
        "step_attacked": "the ~25% between complete-token TPS and decode TPS",
        "hypothesis": (
            "About a quarter of the complete-token cost is runtime ceremony — "
            "command buffer, encoder and synchronization overhead that decode "
            "TPS does not pay."
        ),
        "verdict": "FALSIFIED",
        "evidence": (
            "The gap is an arithmetic artifact of generate_greedy's Instant "
            "identity: complete-token TPS amortises prefill across a short "
            "generation as P/(P+N-1). It is teacher-forced prefill accounting, "
            "not work the runtime performs and could remove. Shortening the "
            "generation widens the 'gap' without changing any runtime cost."
        ),
        "scope": "generate_greedy short-generation measurement, this resident",
        "tested_shapes": "complete-token vs steady decode, short N",
        "falsifier_that_would_reopen_it": (
            "A measurement at large N where the complete-vs-decode gap persists "
            "after P/(P+N-1) is divided out."
        ),
        "redirect": (
            "Steady decode ~35.5 TPS (~28.17 ms/token) is the resident's real "
            "rate and the quantity to minimize. Complete-token TPS on a short "
            "run is not a ceremony measurement and must not be quoted as one."
        ),
        "correction_of_an_earlier_claim": (
            "An earlier reading of this gap called it '24% ceremony'. That was "
            "wrong and this entry supersedes it."
        ),
        "source_receipt": "receipts/future/DISPATCH_CEREMONY.json",
        "patch_preserved_unapplied": "receipts/future/patches/l42-dispatch-ceremony.crate.patch",
    },
)


def scar_rows() -> list[dict[str, Any]]:
    """The JSONL feed negative_index parses (record_id/model/mechanism/...)."""
    rows = []
    for f in FALSIFICATIONS:
        rows.append(
            {
                "record_id": f["id"],
                "model": MODEL,
                "model_geometry": MODEL,
                "machine": MACHINE,
                "mechanism": f["id"].lower(),
                "status": "FALSIFIED",
                "verdict": "FALSIFIED",
                "failure_reason": f["evidence"],
                "claim_boundary": f["scope"],
                "reopen_condition": f["falsifier_that_would_reopen_it"],
                "redirect": f["redirect"],
                "source_receipt": f["source_receipt"],
                "lane": f["lane"],
            }
        )
    return rows


def build() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": 1,
        "recorded_by": "tools/future/tps_falsifications.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "machine": MACHINE,
        "model": MODEL,
        "n_falsifications": len(FALSIFICATIONS),
        "falsifications": list(FALSIFICATIONS),
        "loss_chain": list(CHAIN),
        "active_weight_bytes_per_token": ACTIVE_WEIGHT_BYTES_PER_TOKEN,
        "what_these_three_together_establish": (
            "The remaining levers on this resident are dispatch COUNT and total "
            "BYTES. Addressing, tested geometry and complete-vs-decode "
            "accounting are all spent as explanations."
        ),
        "the_question_that_replaces_them": (
            "Not 'how do we make the same byte stream reach 703 GB/s' but 'why "
            "does this token require these bytes and these physical operations "
            "at all'."
        ),
        "scar_feed": "receipts/future/TPS_FALSIFICATIONS.jsonl",
        "registered_in": "tools/future/negative_index.py SEED_SOURCES",
        "claim_boundary": (
            "Each falsification holds only over the scope named in its own "
            "entry. A falsified hypothesis is not a proven negative for every "
            "member of its family: L41 in particular kills the geometries it "
            "tested, not kernel geometry as a subject. No hardware measurement "
            "is asserted here; the numbers are quoted from the receipts named."
        ),
    }


def record() -> tuple[Path, Path]:
    doc = build()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    SCARS.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in scar_rows()))
    return RECEIPT, SCARS


if __name__ == "__main__":
    from _common import require_known_flags
    require_known_flags(["--build", "--record"])
    if "--record" in sys.argv:
        a, b = record()
        print(f"wrote {a}")
        print(f"wrote {b} ({len(scar_rows())} scars)")
    else:
        print(json.dumps(build(), indent=1))
