#!/usr/bin/env python3
"""Design the Generation B rate ladder against the three mandatory bands.

The directive names three exact physical rates and the primary candidate must land at or
under 0.75 complete BPW.  The inherited ladder cannot serve them: R0 bills 0.876 (over the
primary band), R2 bills 0.505 (over the half-bit band), R4 bills 0.261 (under the
one-third band).  So the rungs are designed here, from the exact billing formula, against
the real GLM-5.2 tensor shapes, and then confirmed by packing a real tensor.

Billing, taken from gravity_forge.ByteLedger rather than restated:

    bits  = N * ceil(log2 k)  +  k * sub * 16  +  64 * 8
            indices              codebook         metadata
    bpw   = bits / elements

with N = rows * cols / D for one subspace.  Two terms matter and they pull opposite ways:
the index term log2(k)/D wants long subvectors, and the codebook term k*D*16/elements
punishes them.  A geometry is admissible only if D divides the row length of every shape
it must serve.

Complete BPW carries one more term the pilot cannot remove.  0.0434 percent of GLM-5.2's
weight is control-sensitive and is carried natively at source precision, which costs a
fixed 0.00694 bits on every candidate.  A band is met on the complete rate, so the
compressed rate must leave room for it.

    design      write GLM52_GENERATION_B_RATE_LADDER.json
    selftest
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402
import glm52_pack as pack  # noqa: E402

REPO = HERE.parent.parent
REPORTS = REPO / "reports/condense/glm52_generation_b"

# The shapes the ladder must serve.  Routed experts are 97.492 percent of all weight, so
# the ladder is designed on them and merely has to remain legal on the rest.
EXPERT_SHAPES = ((2048, 6144), (6144, 2048))
OTHER_SHAPES = ((576, 6144), (28672, 512), (6144, 16384), (2048, 6144), (16384, 2048),
                (951582720 // 6144, 6144))

# Fraction of total weight carried natively at 16 BPW, measured from the official index
# and the tensor contract.  Not a guess: 326,936,576 of 753,329,940,480 elements.
NATIVE_ELEMENT_FRACTION = 326_936_576 / 753_329_940_480
NATIVE_FLOOR_BPW = NATIVE_ELEMENT_FRACTION * 16.0

BANDS = (
    {"band": "B_PRIMARY", "low": 0.70, "high": 0.75,
     "role": "primary full-stream candidate", "ceiling": 0.75},
    {"band": "B_HALF", "low": 0.48, "high": 0.50,
     "role": "mandatory exact half-bit pilot", "ceiling": 0.50},
    {"band": "B_THIRD", "low": 0.30, "high": 0.33,
     "role": "capability-density boundary probe", "ceiling": 0.33},
)

METADATA_BITS = forge._METADATA_BYTES * 8

# Denominator of a mixed-rate allocation.  Sixteen is enough resolution to land inside a
# 0.02-wide band and small enough that every window of routed experts sees the full cycle.
ALLOCATION_PERIOD = 16

# A geometry below these is arithmetically legal and representationally useless: two
# codewords over two dimensions bills exactly half a bit and reconstructs almost nothing.
# Excluding them keeps the blend from pairing a real rung with a degenerate one.
MIN_DIM = 8
MIN_K = 16


def divisors_of_all(shapes) -> list[int]:
    """Subvector lengths that tile every shape's row without a remainder."""
    cols = {shape[1] for shape in shapes}
    limit = min(cols)
    return [d for d in range(2, limit + 1) if all(c % d == 0 for c in cols)]


def billed_bpw(rows: int, cols: int, *, dim: int, k: int) -> float:
    """Exact ledger rate for one tensor at one geometry, single subspace."""
    elements = rows * cols
    vectors = elements // dim
    bits = vectors * max(1, math.ceil(math.log2(max(2, k)))) + k * dim * 16 + METADATA_BITS
    return bits / elements


def weighted_expert_bpw(*, dim: int, k: int) -> float:
    """The rate that actually decides the model, weighted by how much of it is each shape."""
    total_elements = sum(r * c for r, c in EXPERT_SHAPES)
    return sum(billed_bpw(r, c, dim=dim, k=k) * (r * c) for r, c in EXPERT_SHAPES) \
        / total_elements


def complete_bpw(compressed: float) -> float:
    """What a candidate is judged on: the compressed rate plus the native organ floor."""
    return compressed * (1.0 - NATIVE_ELEMENT_FRACTION) + NATIVE_FLOOR_BPW


def search() -> list[dict]:
    """Every legal geometry, with its exact complete rate, ranked inside each band."""
    dims = [d for d in divisors_of_all(EXPERT_SHAPES) if d <= 256]
    rows: list[dict] = []
    for dim in dims:
        for bits in range(1, 17):
            k = 1 << bits
            if k > dim * 4096:  # a codebook larger than this never amortizes at these shapes
                continue
            compressed = weighted_expert_bpw(dim=dim, k=k)
            if compressed >= 1.0:
                continue
            if dim < MIN_DIM or k < MIN_K:
                continue
            legal_everywhere = all(cols % dim == 0 for _, cols in OTHER_SHAPES)
            rows.append({
                "dim": dim, "k": k, "index_bits": bits,
                "index_term_bpw": bits / dim,
                "codebook_bits": k * dim * 16,
                "compressed_bpw": compressed,
                "complete_bpw": complete_bpw(compressed),
                "tiles_every_shape": legal_everywhere,
            })
    return rows


def blend(low: dict, high: dict, target: float) -> dict | None:
    """Hit a rate no single geometry can reach, by splitting tensors between two rungs.

    Single-subspace PQ at power-of-two k quantizes the reachable rate: log2(k)/D moves in
    steps, and around one half bit the steps straddle the band.  On these shapes the
    nearest legal rates are 0.4468 and 0.5119 complete, so the 0.48 to 0.50 band is empty
    and no amount of geometry search fills it.

    Section 6.5 of the directive authorizes asymmetric allocation across tensor roles, and
    that is exactly what this is: a deterministic fraction of routed-expert tensors takes
    the richer rung and the rest take the leaner one.  The split is by position, never
    random, so a repack reproduces the artifact byte for byte.

    Until the pilot returns per-role sensitivity, the split is positional rather than
    sensitivity-ranked, and it says so.
    """
    span = high["complete_bpw"] - low["complete_bpw"]
    if span <= 0:
        return None
    fraction = (target - low["complete_bpw"]) / span
    if not 0.0 < fraction < 1.0:
        return None
    # The split is r tensors in every ALLOCATION_PERIOD, applied by position, so the packer
    # needs no state and a repack reproduces the artifact byte for byte.  One-in-N was too
    # coarse: it cannot express a majority-rich split at all, and the half-bit band needs
    # roughly three quarters rich.
    rich = min(ALLOCATION_PERIOD - 1, max(1, round(fraction * ALLOCATION_PERIOD)))
    achieved_fraction = rich / ALLOCATION_PERIOD
    return {
        "kind": "MIXED_ALLOCATION",
        "lean": {k: low[k] for k in ("dim", "k", "compressed_bpw", "complete_bpw")},
        "rich": {k: high[k] for k in ("dim", "k", "compressed_bpw", "complete_bpw")},
        "allocation_period": ALLOCATION_PERIOD,
        "rich_per_period": rich,
        "rich_fraction": achieved_fraction,
        "rule": (f"a routed-expert tensor takes the rich rung when its position within the "
                 f"window modulo {ALLOCATION_PERIOD} is below {rich}"),
        "complete_bpw": low["complete_bpw"] + span * achieved_fraction,
        "compressed_bpw": (low["compressed_bpw"]
                           + (high["compressed_bpw"] - low["compressed_bpw"])
                           * achieved_fraction),
        "allocation_basis": "POSITIONAL_PENDING_PILOT_SENSITIVITY",
        "tiles_every_shape": low["tiles_every_shape"] and high["tiles_every_shape"],
    }


def pick(rows: list[dict]) -> dict:
    """One rung per band: the highest rate that fits, since a lower rate is a worse trade.

    Inside a band, more bits means more fidelity for the same admissibility, so the choice
    is the largest complete_bpw at or under the band ceiling.  Ties break toward the longer
    subvector, which the inherited ladder measured as the better geometry at equal rate.

    When no single geometry lands in a band, the two nearest legal rates are blended.
    """
    legal = [row for row in rows if row["tiles_every_shape"]]
    chosen: dict[str, dict] = {}
    for band in BANDS:
        candidates = [row for row in legal
                      if band["low"] <= row["complete_bpw"] <= band["ceiling"]]
        if candidates:
            best = max(candidates, key=lambda row: (row["complete_bpw"], row["dim"]))
            chosen[band["band"]] = {**band, "status": "SELECTED",
                                    "kind": "SINGLE_GEOMETRY", **best}
            continue
        below = [row for row in legal if row["complete_bpw"] < band["low"]]
        above = [row for row in legal if row["complete_bpw"] > band["ceiling"]]
        if not below or not above:
            chosen[band["band"]] = {**band, "status": "NO_LEGAL_GEOMETRY_IN_BAND"}
            continue
        low = max(below, key=lambda row: row["complete_bpw"])
        high = min(above, key=lambda row: row["complete_bpw"])
        target = (band["low"] + band["ceiling"]) / 2
        mixed = blend(low, high, target)
        if mixed is None or not band["low"] <= mixed["complete_bpw"] <= band["ceiling"]:
            chosen[band["band"]] = {**band, "status": "NO_LEGAL_GEOMETRY_IN_BAND",
                                    "nearest_below": low, "nearest_above": high}
            continue
        chosen[band["band"]] = {**band, "status": "SELECTED_BY_ALLOCATION", **mixed}
    return chosen


def confirm_on_real_tensor(rung: dict, *, seed: int = 0, iters: int = 4,
                           shape: tuple[int, int] = EXPERT_SHAPES[0]) -> dict:
    """Pack a real expert tensor and check the file is exactly what the formula billed.

    This has to run at the true shape, not a convenient slice of it.  The codebook term is
    k*dim*16 bits regardless of tensor size, so it amortizes over elements: at k=8192 and
    dim=32 it is 4.19 Mbit, which is 0.33 BPW against a real 2048x6144 expert and 2.67 BPW
    against a 256-row stand-in.  A confirmation on the small shape would report a rate four
    times the one the ladder claims and still call itself consistent.
    """
    rng = np.random.default_rng(0)
    rows_, cols_ = shape
    weights = rng.standard_normal((rows_, cols_)).astype(np.float32)
    artifact = forge.pack_product_quant(weights, dim=rung["dim"], subspaces=1,
                                        k=rung["k"], seed=seed, iters=iters)
    blob = pack.serialize(artifact)
    predicted = billed_bpw(rows_, cols_, dim=rung["dim"], k=rung["k"])
    observed = len(blob) * 8 / weights.size
    return {
        "shape": [rows_, cols_],
        "predicted_bpw": predicted,
        "serialized_bpw": observed,
        "serialized_bytes": len(blob),
        "ledger_bpw": artifact.whole_artifact_bpw,
        "relative_frobenius_error": float(forge._rel_error(weights, artifact.recon)),
        "formula_matches_file": abs(predicted - observed) < 1e-9,
    }


def design(confirm: bool = True) -> int:
    rows = search()
    chosen = pick(rows)

    ladder = []
    for index, band in enumerate(BANDS):
        rung = chosen[band["band"]]
        if not str(rung.get("status", "")).startswith("SELECTED"):
            ladder.append({"rung": f"G{index}", **rung})
            continue
        entry = {
            "rung": f"G{index}",
            "band": band["band"], "role": band["role"],
            "kind": rung["kind"],
            "compressed_bpw": rung["compressed_bpw"],
            "complete_bpw": rung["complete_bpw"],
            "band_low": band["low"], "band_ceiling": band["ceiling"],
            "within_band": band["low"] <= rung["complete_bpw"] <= band["ceiling"],
            "under_one_bit_ceiling": rung["complete_bpw"] < 1.0,
        }
        if rung["kind"] == "SINGLE_GEOMETRY":
            entry.update({"dim": rung["dim"], "k": rung["k"], "subspaces": 1,
                          "index_bits": rung["index_bits"]})
            if confirm:
                entry["confirmation"] = confirm_on_real_tensor(rung, iters=2)
        else:
            entry.update({"lean": rung["lean"], "rich": rung["rich"],
                          "allocation_period": rung["allocation_period"],
                          "rich_per_period": rung["rich_per_period"],
                          "rich_fraction": rung["rich_fraction"],
                          "rule": rung["rule"],
                          "allocation_basis": rung["allocation_basis"]})
            if confirm:
                entry["confirmation"] = {
                    "lean": confirm_on_real_tensor(rung["lean"], iters=2),
                    "rich": confirm_on_real_tensor(rung["rich"], iters=2),
                }
        ladder.append(entry)

    payload = {
        "schema": "hawking.glm52.generation_b_rate_ladder.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "why_the_inherited_ladder_does_not_serve": {
            "R0": "0.876 compressed, over the 0.75 primary ceiling",
            "R2": "0.505 compressed, over the 0.50 half-bit ceiling",
            "R4": "0.261 compressed, under the 0.30 one-third floor",
        },
        "billing": {
            "formula": "N*ceil(log2 k) + k*sub*16 + 512 bits, N = elements/dim",
            "source": "gravity_forge.ByteLedger, not restated",
            "metadata_bits": METADATA_BITS,
        },
        "native_floor": {
            "element_fraction": NATIVE_ELEMENT_FRACTION,
            "bpw": NATIVE_FLOOR_BPW,
            "note": ("0.0434 percent of weight is control-sensitive and carried at source "
                     "precision; every complete rate pays this and no codec can remove it"),
        },
        "shapes": {"expert": [list(s) for s in EXPERT_SHAPES],
                   "expert_weight_share": 0.97492,
                   "confirmation_shape": list(EXPERT_SHAPES[0]),
                   "confirmation_note": (
                       "confirmations run at the true expert shape; the codebook term is "
                       "fixed per tensor, so a smaller stand-in would report a different "
                       "rate and still call itself consistent")},
        "relative_frobenius_error_is_not_quality": (
            "confirmation error is measured on gaussian noise at the real shape to prove "
            "the file matches the ledger; it is a size proof, not an F1 result, and real "
            "weight-space and trajectory error come from the pilot"),
        "geometries_evaluated": len(rows),
        "ladder": ladder,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_B_RATE_LADDER.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({
        "wrote": str(target.relative_to(REPO)),
        "ladder": [{k: v for k, v in entry.items()
                    if k in ("rung", "band", "dim", "k", "compressed_bpw",
                             "complete_bpw", "within_band", "status")}
                   for entry in ladder]}, indent=2))
    return 0


def selftest() -> int:
    # The formula must reproduce the inherited ladder's published rates, or it is not the
    # same accounting and nothing built on it can be compared to prior work.
    r0 = weighted_expert_bpw(dim=8, k=128)
    r2 = weighted_expert_bpw(dim=16, k=256)
    r4 = weighted_expert_bpw(dim=32, k=256)
    assert abs(r0 - 0.876) < 0.002, r0
    assert abs(r2 - 0.505) < 0.002, r2
    assert abs(r4 - 0.261) < 0.002, r4

    # Carrying 0.0434 percent of the weight at 16 bits instead of the codec rate can only
    # push the complete rate UP.  A ladder that reported complete below compressed would
    # be excluding the organs again, which is the defect this campaign exists to close.
    assert complete_bpw(0.0) == NATIVE_FLOOR_BPW
    assert complete_bpw(0.75) > 0.75, complete_bpw(0.75)
    assert complete_bpw(0.75) - 0.75 < 0.01, complete_bpw(0.75)

    rows = search()
    assert rows, "no legal geometries found"
    chosen = pick(rows)
    for band in BANDS:
        rung = chosen[band["band"]]
        assert str(rung["status"]).startswith("SELECTED"), (band["band"], rung)
        assert band["low"] <= rung["complete_bpw"] <= band["ceiling"], rung
        assert rung["complete_bpw"] < 1.0

    # The half-bit band is empty for every legal single geometry on these shapes, so it
    # must have been reached by allocation.  If that ever stops being true the ladder
    # should say so rather than silently keep the blend.
    assert chosen["B_HALF"]["status"] == "SELECTED_BY_ALLOCATION", chosen["B_HALF"]
    blended = chosen["B_HALF"]
    assert blended["lean"]["complete_bpw"] < 0.48 < blended["complete_bpw"], blended
    assert blended["rich"]["complete_bpw"] > 0.50, blended

    # The primary rung is the one the campaign is judged on: it must clear 0.75 complete.
    assert chosen["B_PRIMARY"]["complete_bpw"] <= 0.75

    # And the file must be exactly what was billed, at the real row length.
    confirmation = confirm_on_real_tensor(chosen["B_THIRD"], iters=2)  # single geometry
    assert confirmation["formula_matches_file"], confirmation

    print("glm52_rate_ladder selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "design"
    raise SystemExit({"design": design, "selftest": selftest}[command]())
