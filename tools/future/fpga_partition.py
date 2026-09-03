"""Within-organ Apple/FPGA partition: where the split helps, where it cannot.

Nothing in this repo swept a partition ratio. The one partitioner computes a
single deterministic row split from HBM capacity, and the 65/35 Apple/FPGA figure
that appears in the roadmap is stored as inert metadata explicitly marked "not
applied as physics". There was also no transport break-even anywhere: the nearest
things were a bare ratio with no threshold and a per-scenario argmin over
hand-picked nanosecond constants.

THE MODEL IS DIMENSIONLESS ON PURPOSE. tools/future/hwir.py deliberately refuses
to carry an HBM bandwidth in GB/s -- it names the datasheet rate and declines it --
and this module must not smuggle one back in through a side door. So every rate
enters as a RATIO to Apple's effective rate, and the answer is a surface over
those ratios rather than a number. An absolute prediction would require the two
things nobody has: a measured Apple rate for this organ and a measured link rate
for a board that is not here.

    W        bytes of organ weight traffic per token          MEASURED (census)
    A + R    activation out + partial reduction back          MEASURED or DERIVED
    r        FPGA effective rate / Apple effective rate       UNPINNED, swept
    t        link effective rate / Apple effective rate       UNPINNED, swept
    s        fixed per-token setup, in Apple-byte-times       UNPINNED, swept

In units of "the time Apple takes to move one byte", with the domains overlapped:

    apple(f)     = (1 - f) * W
    fpga(f)      = f * W / r
    transport    = (A + R) / t + s          == C, independent of f
    critical(f)  = max(apple(f), fpga(f) + C)
    baseline     = W                        (f = 0, Apple alone)

Balancing the two arms gives the optimum in closed form:

    f* = (1 - C/W) * r/(r+1)        clamped to [0, 1]
    speedup(f*) = 1 / (1 - f*)      when f* > 0

and therefore the whole break-even reduces to ONE inequality:

    THE FPGA HELPS IF AND ONLY IF  C < W

Per-token transport must cost less than the Apple-time of the organ's entire
weight traffic. Everything else -- the optimum split, the ceiling, which regime
you are in -- follows from that. r sets how much of the win you can collect;
C/W decides whether there is a win at all.

No clause here is a hardware measurement. U50_PRESENT is false.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future import decision_value, hwir

REPO = Path(__file__).resolve().parents[2]
CENSUS = REPO / "receipts" / "headless" / "FLASH_ORGAN_CENSUS.json"

#: A conclusion that depends on an unpinned input is an envelope, never a prediction.
DERIVED_FROM_MEASURED = "DERIVED_FROM_MEASURED"
ENVELOPE = "ENVELOPE"


class UnpinnedConclusion(ValueError):
    """Raised when an envelope is asked to behave like a prediction.

    The failure this prevents is the one the campaign keeps finding: a guessed
    constant travels three hops and arrives as a result. r, t and s are not
    measurable without the board, so anything computed from them is a shape, not
    a number, and saying so has to be enforced rather than remembered.
    """


@dataclass(frozen=True)
class Conclusion:
    """A value plus what it is allowed to claim."""
    value: Any
    basis: str
    inputs: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def as_prediction(self) -> Any:
        if self.basis != DERIVED_FROM_MEASURED:
            unpinned = sorted(k for k, v in self.inputs.items()
                              if isinstance(v, Mapping) and not v.get("pinned", True))
            raise UnpinnedConclusion(
                f"{self.note or 'this conclusion'} rests on unpinned inputs "
                f"{unpinned or list(self.inputs)}; it is an {self.basis}. A swept "
                "ratio is not a measurement and must not be reported as one."
            )
        return self.value


# ---------------------------------------------------------------------------
# Measured side: bytes. No ratio, no assumption, no board.
# ---------------------------------------------------------------------------

def census() -> dict[str, Any]:
    """The organ census as measured off the indexed model on disk."""
    doc = json.loads(CENSUS.read_text(encoding="utf-8"))
    return doc


def family_bytes() -> dict[str, int]:
    return {row["family"]: int(row["bytes"]) for row in census()["family_summary"]}


def hbm_capacity_report(hbm_bytes: int | None = None) -> dict[str, Any]:
    """What fraction of the body the scarce HBM can hold. Pure arithmetic.

    This is the one part of the FPGA question that needs no rate at all, so it is
    the one part that can be DERIVED_FROM_MEASURED today.
    """
    doc = census()
    fam = {row["family"]: int(row["bytes"]) for row in doc["family_summary"]}
    total = int(doc["source_parameter_bytes_indexed"])
    profile = hwir.u50_family_profile("u50dd").to_dict()
    cap = int(hbm_bytes if hbm_bytes is not None else profile["hbm_capacity_bytes"])

    ordered = sorted(fam.items(), key=lambda kv: -kv[1])
    # The families a reader would reach for first: everything that is not the
    # MoE body or the n-gram table.
    non_bulk = {k: v for k, v in fam.items() if k not in ("routed_experts", "ngram_embedding")}
    non_bulk_bytes = sum(non_bulk.values())

    # Largest-first greedy is NOT the answer, it is the control: hbm_doctor owns
    # the exact 0/1 knapsack and refuses size ranking for good reason. This only
    # reports how many WHOLE families fit at all.
    whole_fit = [k for k, v in ordered if v <= cap]

    return {
        "basis": DERIVED_FROM_MEASURED,
        "model": doc["model"],
        "source": {
            "receipt": str(CENSUS.relative_to(REPO)),
            "pinned_source": doc.get("pinned_source"),
            "source_index_sha256": doc.get("source_index_sha256"),
            "tensor_count": doc.get("tensor_count"),
            "layer_count_observed": doc.get("layer_count_observed"),
        },
        "device": {
            "variant_id": profile["variant_id"],
            "origin": profile["origin"],
            "hbm_capacity_bytes": cap,
            "hbm_channels": profile["hbm_channels"],
            "declared_not_measured": profile["declared_not_measured"],
        },
        "parameter_bytes": total,
        "hbm_fraction_of_body": cap / total,
        "family_bytes": dict(ordered),
        "families_that_fit_whole": whole_fit,
        "non_bulk_bytes": non_bulk_bytes,
        "non_bulk_families": sorted(non_bulk),
        "non_bulk_fits": non_bulk_bytes <= cap,
        "non_bulk_overflow_bytes": max(0, non_bulk_bytes - cap),
        "finding": (
            "HBM holds a small fraction of the body, and the obvious resident set -- "
            "everything that is not the MoE experts or the n-gram table -- does NOT "
            "fit. Residency is therefore a real selection problem, not an obvious one, "
            "and the selection must be value-ranked rather than size-ranked "
            "(tools/future/hbm_doctor.py owns that knapsack and refuses size ranking)."
        ) if non_bulk_bytes > cap else (
            "The non-bulk families fit whole, so the residency question reduces to "
            "what else to admit with the remaining headroom."
        ),
    }


# ---------------------------------------------------------------------------
# Ratio side: everything below is an ENVELOPE.
# ---------------------------------------------------------------------------

def unpinned_ratio(name: str, reason: str) -> dict[str, Any]:
    return hwir.unpinned_field(f"{name}: {reason}")


def transport_cost(activation_bytes: float, reduction_bytes: float,
                   t: float, setup: float = 0.0) -> float:
    """C, in Apple-byte-times. t is link rate / Apple rate.

    t == 0 is a link that moves nothing, which must cost infinity rather than
    divide by zero -- a zero-bandwidth link that reports a finite cost is how a
    partition model recommends a split across a disconnected card.
    """
    if t <= 0:
        return float("inf")
    payload = max(0.0, float(activation_bytes)) + max(0.0, float(reduction_bytes))
    return payload / float(t) + max(0.0, float(setup))


def critical_path(f: float, W: float, r: float, C: float) -> float:
    """max(Apple arm, FPGA arm + transport), the two domains overlapped."""
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"partition fraction must be in [0, 1], got {f}")
    apple = (1.0 - f) * W
    if f == 0.0:
        return apple                      # nothing crosses the link
    if r <= 0:
        return float("inf")               # an FPGA that computes nothing
    return max(apple, f * W / r + C)


def optimum_fraction(W: float, r: float, C: float) -> float:
    """f* = (1 - C/W) * r/(r+1), clamped. Balances the two arms."""
    if W <= 0 or r <= 0:
        return 0.0
    if C >= W:
        return 0.0                        # transport alone exceeds the whole organ
    raw = (1.0 - C / W) * (r / (r + 1.0))
    return min(1.0, max(0.0, raw))


def speedup(W: float, r: float, C: float) -> float:
    """Baseline W over the critical path at f*. 1.0 means the FPGA bought nothing."""
    f = optimum_fraction(W, r, C)
    if f <= 0.0:
        return 1.0
    return W / critical_path(f, W, r, C)


def helps(W: float, C: float) -> bool:
    """The entire break-even, in one inequality: C < W.

    Per-token transport must cost less than the Apple-time of the organ's whole
    weight traffic. r decides how much of the win is collectable; it cannot
    create one.
    """
    return float(C) < float(W)


def break_even_payload_bytes(W: float, t: float, setup: float = 0.0) -> float:
    """The activation+reduction size at which transport exactly eats the organ.

    Above this, no r and no f make the FPGA worth using for this organ.
    """
    if t <= 0:
        return 0.0
    return max(0.0, (float(W) - float(setup)) * float(t))


PHASES = (
    "APPLE_ONLY",           # transport alone costs more than the organ
    "TRANSPORT_DOMINATED",  # a split exists but the link caps it hard
    "COMPUTE_BALANCED",     # both arms matter; the classic overlapped regime
    "FPGA_LIMITED",         # the link is cheap, the FPGA's own rate is the wall
)


def phase(W: float, r: float, C: float) -> str:
    if not helps(W, C):
        return "APPLE_ONLY"
    f = optimum_fraction(W, r, C)
    share_lost_to_transport = C / W               # in [0, 1) here
    if share_lost_to_transport >= 0.5:
        return "TRANSPORT_DOMINATED"
    return "FPGA_LIMITED" if r < 1.0 else "COMPUTE_BALANCED"


def sweep(W: float, activation_bytes: float, reduction_bytes: float,
          r_grid: Iterable[float], t_grid: Iterable[float],
          setup: float = 0.0) -> list[dict[str, Any]]:
    """The phase dataset. One row per (r, t) cell, machine-queryable."""
    rows = []
    for r in r_grid:
        for t in t_grid:
            C = transport_cost(activation_bytes, reduction_bytes, t, setup)
            f = optimum_fraction(W, r, C)
            rows.append({
                "r_fpga_over_apple": r,
                "t_link_over_apple": t,
                "transport_cost_C": C,
                "C_over_W": C / W if W else float("inf"),
                "optimum_fpga_fraction": f,
                "speedup": speedup(W, r, C),
                "helps": helps(W, C),
                "phase": phase(W, r, C),
            })
    return rows


def envelope(W: float, activation_bytes: float, reduction_bytes: float,
             r_grid: Iterable[float], t_grid: Iterable[float],
             setup: float = 0.0) -> Conclusion:
    """The sweep, wrapped so it cannot be reported as a prediction."""
    rows = sweep(W, activation_bytes, reduction_bytes, r_grid, t_grid, setup)
    useful = [row for row in rows if row["helps"]]
    return Conclusion(
        value={
            "rows": rows,
            "cells": len(rows),
            "cells_where_fpga_helps": len(useful),
            "best": max(rows, key=lambda x: x["speedup"]) if rows else None,
            "phase_counts": {p: sum(1 for x in rows if x["phase"] == p) for p in PHASES},
            "break_even_rule": "the FPGA helps if and only if C < W",
        },
        basis=ENVELOPE,
        inputs={
            "W_bytes": {"pinned": True, "value": W},
            "activation_bytes": {"pinned": True, "value": activation_bytes},
            "reduction_bytes": {"pinned": True, "value": reduction_bytes},
            "r_fpga_over_apple": unpinned_ratio(
                "r", "needs a measured Apple rate for this organ and a board"),
            "t_link_over_apple": unpinned_ratio(
                "t", "needs sustained host<->board bandwidth, which requires the carrier"),
            "setup": unpinned_ratio("s", "needs measured DMA submission and completion cost"),
        },
        note="Apple/FPGA partition envelope",
    )


# ---------------------------------------------------------------------------
# Receipt. The phase dataset is the deliverable: a surface, never a tok/s number.
# ---------------------------------------------------------------------------

#: Decades of ratio either side of parity. The interesting structure is near 1.
R_GRID = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
#: With A+R = 0.002W, TRANSPORT_DOMINATED is exactly 0.002 < t <= 0.004, a band a
#: decade-spaced grid steps straight over. The first grid reported that region as
#: EMPTY when it had simply never been sampled, which is a phase diagram lying by
#: omission. These three points exist to sample the band the model predicts.
T_GRID = (0.001, 0.0025, 0.003, 0.0035, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)


def preboard_link_constant_audit() -> dict[str, Any]:
    """The pre-board link simulator's bandwidth constant has no provenance.

    hcli/agentos/fpga_preboard.py sets bandwidth_gbps=64.0 and latency_ns=900.0 as
    bare defaults. The module contains no provenance vocabulary at all, and never
    says which link 64 Gb/s is meant to be. The device profile meanwhile declares
    PCIe Gen3 x16 with real_carrier UNPINNED, so the constant cannot be reconciled
    with either leg of the declared topology -- not because it is wrong, but
    because nothing says what it refers to.
    """
    src = (REPO / "hcli" / "agentos" / "fpga_preboard.py").read_text(encoding="utf-8")
    # INPUT provenance -- where a number came from -- is the question. Counting
    # "DERIVED" naively answers a different one and says the module is tagged: it
    # defines DERIVED = "[D]" at line 23 and stamps it on 26 emitted dicts as an
    # OUTPUT claim-boundary label. A constant's epistemic status and a result's
    # are not the same field, and conflating them is how an assumed number is
    # mistaken for a sourced one.
    input_vocab = ("PUBLISHED_SPEC", "CALIBRATED", "ASSUMED", "PLACEHOLDER",
                   "provenance", "citation", "document_class", "sourced_field",
                   "unpinned_field", "UNPINNED")
    output_labels = {"DERIVED": src.count("DERIVED"), "SIMULATED": src.count("SIMULATED"),
                     "VERIFIED": src.count("VERIFIED")}
    profile = hwir.u50_family_profile("u50dd").to_dict()
    return {
        "module": "hcli/agentos/fpga_preboard.py",
        "input_provenance_occurrences": {w: src.count(w) for w in input_vocab},
        "has_input_provenance_tagging": any(src.count(w) for w in input_vocab),
        "output_claim_labels_present": output_labels,
        "why_they_are_different": (
            "DERIVED there is DERIVED = \"[D]\" (line 23), stamped on emitted dicts. "
            "It says what the RESULT may claim, never where the INPUT came from."
        ),
        "declared_link_in_device_profile": {
            "pcie_generation": profile["pcie_generation"],
            "pcie_lanes": profile["pcie_lanes"],
            "real_carrier": profile["real_carrier"],
            "real_carrier_note": profile.get("real_carrier_note"),
            "host_device_bytes_per_modelled_cycle": profile["host_device_bytes_per_modelled_cycle"],
        },
        "finding": (
            "The link constant carries no input provenance, and the device profile "
            "expresses host<->device movement in bytes per MODELLED CYCLE precisely so "
            "that no absolute rate is asserted. One module refuses to name a rate and "
            "the other hardcodes one. This model takes the first convention."
        ),
    }


#: Which sealed U50 prediction pins each unpinned model input. hwir already
#: pre-registers the device coefficients with falsifiers; what was missing is the
#: statement of WHICH of them decides the architecture.
PINNED_BY = {
    "t_link_over_apple": (
        "u50.coeff.host_device_bytes_per_modelled_cycle",
        "u50.coeff.host_device_queue_cycles",
    ),
    "setup": ("u50.coeff.host_device_queue_cycles",),
    "r_fpga_over_apple": (
        "u50.coeff.hbm_bytes_per_modelled_cycle",
        "u50.coeff.fabric_bytes_per_modelled_cycle",
    ),
}

#: Both ratios are FPGA-or-link over APPLE. The denominator is measurable on this
#: host today -- it needs no board, only an exclusive GPU lane -- and the board
#: supplies only the numerator. Half of every ratio is already reachable.
DENOMINATOR_IS_LOCAL = (
    "r and t are both <something>/apple_effective_rate. The Apple denominator is a "
    "local measurement gated on an exclusive benchmark lane, not on the U50. "
    "Measuring it before the board arrives halves the remaining unknown in both "
    "ratios and cannot be contaminated by the board's absence."
)


def sensitivity(W: float, activation_bytes: float, reduction_bytes: float,
                r_grid: Iterable[float] = R_GRID, t_grid: Iterable[float] = T_GRID,
                setup: float = 0.0) -> list[dict[str, Any]]:
    """Rank the unknowns by how much the ANSWER moves, not by how unknown they are.

    The ordering rule lives in tools/future/decision_value.py because it is not an
    FPGA fact: a measurement is worth its wall time in proportion to its ability to
    change the DECISION, not the number. This function supplies the domain -- the
    decision is the phase, the magnitude is the speedup -- and annotates the result
    with which sealed prediction pins each input.

    The outcome here is the case that produced the rule: r moves the speedup ~14x
    and is measured LAST, because r does not appear in C and the break-even is
    C < W, so no value of r reaches APPLE_ONLY. The FPGA's own speed sizes a win
    and can never create or destroy one.
    """
    setups = [setup, 0.1 * W, 0.5 * W, W, 2.0 * W]

    def decide(r_fpga_over_apple: float, t_link_over_apple: float, setup_cost: float):
        C = transport_cost(activation_bytes, reduction_bytes, t_link_over_apple, setup_cost)
        return {
            "decision": phase(W, r_fpga_over_apple, C),
            "magnitude": speedup(W, r_fpga_over_apple, C),
        }

    ranking = decision_value.rank_measurements(
        {
            "r_fpga_over_apple": sorted(r_grid),
            "t_link_over_apple": sorted(t_grid),
            "setup_cost": sorted(set(setups)),
        },
        decide,
        # Hold setup at the caller's stated value, not at the median of a grid
        # invented to span 0..2W: that median is 0.5W, an assumption that transport
        # already eats half the organ before any payload moves, and holding the
        # other sweeps there made r look 8x weaker than it is.
        hold_at={"setup_cost": setup},
        notes={
            "r_fpga_over_apple": DENOMINATOR_IS_LOCAL,
            "t_link_over_apple": DENOMINATOR_IS_LOCAL,
            "setup_cost": "needs measured DMA submission and completion cost on the board",
        },
    )

    _PIN = {**PINNED_BY, "setup_cost": PINNED_BY["setup"]}
    rows = []
    for m in ranking.measurements:
        row = m.as_dict()
        row["swept_over"] = row.pop("range_swept")
        row["speedup_min"] = row.pop("magnitude_min")
        row["speedup_max"] = row.pop("magnitude_max")
        row["speedup_swing"] = row.pop("magnitude_swing")
        row["phases_reachable"] = row.pop("decisions_reachable")
        row["changes_phase"] = row.pop("changes_decision")
        row["input"] = row.pop("name")
        row["can_falsify_the_architecture"] = "APPLE_ONLY" in row["phases_reachable"]
        row["pinned_by_sealed_predictions"] = list(_PIN.get(row["input"], ()))
        row["resolvable_without_the_board"] = row["input"] in (
            "r_fpga_over_apple", "t_link_over_apple")
        row["how"] = row.pop("note")
        rows.append(row)
    return rows


def build() -> Path:
    from tools.future._common import write_receipt

    capacity = hbm_capacity_report()
    # One organ's weight traffic per token, measured. mlp_hyperconnection is the
    # FFN family the organ map assigns to within-organ tensor parallelism.
    fam = capacity["family_bytes"]
    W_bytes = float(fam["mlp_hyperconnection"])
    layers = int(capacity["source"]["layer_count_observed"] or 1)
    per_layer_W = W_bytes / layers
    # Activation and partial reduction are two vectors per token, not a weight
    # body. Their size is the hidden width, which the census does not carry, so
    # they stay a swept fraction of W rather than an invented byte count.
    env = envelope(per_layer_W, per_layer_W * 0.001, per_layer_W * 0.001, R_GRID, T_GRID)

    doc = {
        "schema": "hawking.future.fpga_partition.v1",
        "evidence_tier": "COST_MODEL",
        "u50_present": False,
        "hardware_measured": False,
        "not_a_board_result": True,
        "capacity": capacity,
        "partition_envelope": {
            "basis": env.basis,
            "inputs": env.inputs,
            "organ": "mlp_hyperconnection",
            "W_bytes_per_layer": per_layer_W,
            "activation_plus_reduction_as_fraction_of_W": 0.002,
            **env.value,
        },
        "preboard_link_constant_audit": preboard_link_constant_audit(),
        "experiment_pack": {
            "purpose": (
                "The minimal discriminating measurement set, DERIVED from which "
                "unpinned input moves the answer -- not a wishlist. Ordered by what "
                "can falsify the architecture, then by what sizes the win."
            ),
            "ordering_result": (
                "r (FPGA rate / Apple rate) swings the speedup ~14x and t (link rate "
                "/ Apple rate) only ~2x, yet t is measured FIRST: r does not appear "
                "in C, and the break-even is C < W, so no value of r can put the "
                "system into APPLE_ONLY. The FPGA's own speed sizes a win and can "
                "never create or destroy one. Only the link decides whether there is "
                "an architecture to build."
            ),
            "apple_denominator_is_local": DENOMINATOR_IS_LOCAL,
            "rows": sensitivity(per_layer_W, per_layer_W * 0.001, per_layer_W * 0.001),
        },
        "claim_boundary": (
            "COST_MODEL over MEASURED bytes and UNPINNED rate ratios. The capacity "
            "arithmetic is derived from an indexed model on disk and is exact. Every "
            "partition number is a surface over ratios nobody can measure without the "
            "board, and Conductor.as_prediction() raises rather than reporting one as "
            "a rate. No tok/s, no GB/s, no Fmax, no utilisation."
        ).replace("Conductor", "Conclusion"),
    }
    return write_receipt("FPGA_PARTITION_ENVELOPE.json", doc,
                         recorded_by="tools/future/fpga_partition.py")


def main() -> int:
    path = build()
    cap = hbm_capacity_report()
    print(f"HBM {cap['device']['hbm_capacity_bytes']:,} B is "
          f"{cap['hbm_fraction_of_body']*100:.2f}% of {cap['parameter_bytes']:,} B of parameters")
    print(f"non-bulk resident set {cap['non_bulk_bytes']:,} B  fits={cap['non_bulk_fits']}  "
          f"overflow={cap['non_bulk_overflow_bytes']:,} B")
    env = envelope(1.0, 0.001, 0.001, R_GRID, T_GRID)
    print(f"phase cells {env.value['cells']}, fpga helps in {env.value['cells_where_fpga_helps']}")
    for phase_name, n in sorted(env.value["phase_counts"].items()):
        print(f"    {phase_name:22s} {n}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
