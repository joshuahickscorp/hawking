#!/usr/bin/env python3.12
"""What 1,000 true batch-1 tokens per second actually costs, in bytes and in stages.

A tokens-per-second target is two budgets, not one.  There is a TRAFFIC budget, set by how
many bytes a token must move against the measured 736 GB/s, and there is a DEPTH budget,
set by how many serialized launch boundaries fit inside the token's wall-clock at the
measured 215.8 microsecond command-buffer cost.  A design can satisfy either alone and
still be impossible, so this module reports both and lets the tighter one decide.

The depth budget is the one that gets forgotten.  It is independent of how good the kernels
are: 78 layers submitted one command buffer each cost 16.8 ms of pure submission before any
weight is read, which caps the current runtime near 59 tok/s no matter what the kernels do.
That is why a GPU-resident causal loop is a prerequisite for four-digit throughput rather
than an optimization of it.

Everything here is arithmetic over a sealed measured ledger.  It predicts, it does not
measure, and the distinction is carried in the output: every figure is tagged with the
evidence grade the run report defines, and a projected configuration is never reported at
the same grade as a run one.  A lane whose enabling artifact does not exist is marked
BLOCKED with the specific missing input, because an unreachable configuration that looks
reachable on a spreadsheet is the failure mode this file exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

SCHEMA = "hawking.true_batch1.tps_budget.v1"

# Measured on this box.  Sources named so a reader can refuse any of them individually.
MEASURED_BANDWIDTH_BPS = 736e9          # sustained GPU read, best-median, 1 GiB private float4
MEASURED_COMPUTE_FLOPS = 17_703e9       # fp32 FMA, 32 independent accumulators
MEASURED_COMMAND_BUFFER_SECONDS = 215.8e-6
MEASURED_DISPATCH_SECONDS = 0.71e-6
GPU_CORES = 60
THREADGROUP_MEMORY_BYTES = 32_768

# The vendor figure, carried only so a reader can see it is not what was used.
VENDOR_BANDWIDTH_BPS = 819e9

BANDWIDTH_EFFICIENCIES = (1.00, 0.70, 0.50)


@dataclass(frozen=True)
class Organ:
    """One class of tensor traffic in a token, from the sealed active-byte ledger."""

    name: str
    active_bytes: int
    dense_bf16_bytes: int
    macs: int
    # How the lanes act on this organ.  routed_expert scales with the expert count, anything
    # per-layer scales with sequential depth, and lm_head runs once per token whatever else
    # changes, so it is the one term that survives every collapse.
    scales_with_experts: bool = False
    scales_with_depth: bool = True
    compressible: bool = True

    @property
    def stored_bpw(self) -> float:
        """Effective bits per weight as actually stored, from the bytes, never nominal."""
        weights = self.dense_bf16_bytes / 2
        return 0.0 if weights <= 0 else self.active_bytes * 8 / weights


@dataclass
class Config:
    """One candidate architecture/runtime point."""

    name: str
    lane: str
    routed_experts: int = 8
    sequential_stages: int = 78
    compress_protected: bool = False     # bring indexer and router into the compact codec
    command_buffers_per_token: int = 78
    bandwidth_efficiency: float = 0.70
    evidence: str = "PROJECTED"
    blocked_on: str | None = None
    note: str = ""


def load_organs(ledger: dict) -> list[Organ]:
    """Read the sealed per-organ ledger and attach each organ's lane behaviour.

    The behaviour flags are the modelling assumption in this file; they are declared here
    rather than buried in the arithmetic so a reviewer can reject one without rereading the
    maths.  lm_head does not scale with depth because a token passes it exactly once.
    """
    per_expert = {"routed_expert"}
    per_token_once = {"lm_head", "embeddings"}
    organs = []
    for row in ledger["per_organ"]:
        name = row["organ"]
        active = int(row["active_bytes"])
        dense = int(row["dense_bf16_bytes"])
        organs.append(Organ(
            name=name,
            active_bytes=active,
            dense_bf16_bytes=dense,
            macs=int(row.get("dense_equivalent_macs", 0)),
            scales_with_experts=name in per_expert,
            scales_with_depth=name not in per_token_once,
            # An organ already stored below BF16 is compressed; one stored at BF16 is a
            # protected tensor the packer deliberately left alone.
            compressible=active >= dense,
        ))
    return organs


def project(organs: list[Organ], cfg: Config, *, base_experts: int = 8,
            base_stages: int = 78, compact_bpw: float = 0.87633) -> dict:
    """Active bytes per token for one configuration, organ by organ."""
    expert_scale = cfg.routed_experts / base_experts
    depth_scale = cfg.sequential_stages / base_stages

    rows = []
    total = 0
    for organ in organs:
        bytes_ = float(organ.active_bytes)
        if cfg.compress_protected and organ.compressible:
            # A protected tensor brought into the compact codec pays the codec's own rate.
            weights = organ.dense_bf16_bytes / 2
            bytes_ = weights * compact_bpw / 8
        if organ.scales_with_experts:
            bytes_ *= expert_scale
        if organ.scales_with_depth:
            bytes_ *= depth_scale
        rows.append({"organ": organ.name, "bytes": int(bytes_),
                     "share": None, "stored_bpw_before": round(organ.stored_bpw, 5)})
        total += bytes_

    total = int(total)
    for row in rows:
        row["share"] = round(row["bytes"] / total, 5) if total else 0.0

    traffic_seconds = total / (MEASURED_BANDWIDTH_BPS * cfg.bandwidth_efficiency)
    depth_seconds = cfg.command_buffers_per_token * MEASURED_COMMAND_BUFFER_SECONDS
    # The two budgets do not add: submission of the next stage overlaps the current one's
    # execution in a well-built runtime.  The token cannot be faster than either, so the
    # binding constraint is the larger, and which one binds is the actionable output.
    token_seconds = max(traffic_seconds, depth_seconds)
    binds = "DEPTH" if depth_seconds >= traffic_seconds else "TRAFFIC"

    # Labels travel with the row, not with the document, so a projection cannot be quoted out
    # of context as a measurement.  The control carries none of them because it is the only
    # row read straight off a measured ledger.
    labels: list[str] = []
    if cfg.evidence != "MEASURED_LEDGER":
        labels = ["PROJECTED", "NOT_MEASURED",
                  f"ASSUMES_{int(round(cfg.bandwidth_efficiency * 100))}_PERCENT_BANDWIDTH",
                  ("ASSUMES_ONE_COMMAND_GRAPH" if cfg.command_buffers_per_token == 1
                   else f"ASSUMES_{cfg.command_buffers_per_token}_COMMAND_BUFFERS_PER_TOKEN")]
        if cfg.blocked_on:
            labels.append("ENABLING_ARTIFACT_DOES_NOT_EXIST")

    return {
        "config": asdict(cfg),
        "labels": labels,
        "active_bytes_per_token": total,
        "active_mb_per_token": round(total / 1e6, 2),
        "organs": rows,
        "traffic_seconds_per_token": traffic_seconds,
        "depth_seconds_per_token": depth_seconds,
        "token_seconds": token_seconds,
        "tokens_per_second": round(1.0 / token_seconds, 2) if token_seconds else None,
        "traffic_only_tps": round(1.0 / traffic_seconds, 2) if traffic_seconds else None,
        "depth_only_tps": round(1.0 / depth_seconds, 2) if depth_seconds else None,
        "binding_constraint": binds,
        "evidence": cfg.evidence,
        "blocked_on": cfg.blocked_on,
    }


def traffic_budget_table() -> list[dict]:
    """Bytes per token a TPS target allows, at each bandwidth efficiency.  Pure physics."""
    out = []
    for tps in (100, 250, 500, 1000, 2000, 5000):
        row = {"target_tps": tps, "seconds_per_token": 1.0 / tps}
        for eff in BANDWIDTH_EFFICIENCIES:
            budget = MEASURED_BANDWIDTH_BPS * eff / tps
            row[f"max_bytes_at_{int(eff*100)}pct"] = int(budget)
            row[f"max_mb_at_{int(eff*100)}pct"] = round(budget / 1e6, 1)
        out.append(row)
    return out


def depth_budget_table() -> list[dict]:
    """How many command buffers a TPS target can afford, at the measured fixed cost.

    This is the table that rules out submission models outright.  A target whose whole
    wall-clock is smaller than one command buffer cannot be reached by any runtime that
    submits per token, however fast its kernels are.
    """
    out = []
    for tps in (100, 250, 500, 1000, 2000, 5000):
        budget = 1.0 / tps
        affordable = budget / MEASURED_COMMAND_BUFFER_SECONDS
        # Judge on the SHARE one buffer takes, not on how many would nominally fit.  A count
        # of 4.6 sounds comfortable until you notice each one costs 21.6% of the token, which
        # leaves nothing for the work the buffers exist to carry.
        share = MEASURED_COMMAND_BUFFER_SECONDS / budget
        out.append({
            "target_tps": tps,
            "seconds_per_token": budget,
            "command_buffers_affordable": round(affordable, 3),
            "one_cb_share_of_budget": round(share, 4),
            "per_layer_submission_tps_ceiling": round(
                1.0 / (78 * MEASURED_COMMAND_BUFFER_SECONDS), 2),
            "verdict": (
                "IMPOSSIBLE_WITH_PER_TOKEN_SUBMISSION" if share >= 1.0
                else "REQUIRES_ONE_COMMAND_GRAPH" if share >= 0.20
                else "REQUIRES_FEW_COMMAND_BUFFERS" if share >= 0.05
                else "SUBMISSION_NOT_BINDING"),
        })
    return out


def candidate_configs() -> list[Config]:
    """The lane ladder from the moonshot brief, each tagged with what actually blocks it.

    The blocked_on strings are the point of this list.  Lanes A and G are runtime work on
    a sealed artifact and are open today.  Lane B's protected-tensor item is a packing
    decision that belongs to the science stream.  Lanes D, E and F all require training a
    student against teacher state, and the teacher capsules live inside the live campaign
    root with a capture running, so they are blocked on access rather than on difficulty.
    """
    return [
        Config("shipped-today", "control", 8, 78, False, 78, 0.70,
               evidence="MEASURED_LEDGER",
               note="the sealed R0 artifact under per-layer submission"),
        # Lane A alone, kept as its own row precisely because it moves nothing: the kernels get
        # arbitrarily better and the token does not, because submission still binds.
        Config("A-kernels-only", "A", 8, 78, False, 78, 0.70,
               evidence="PROJECTED",
               note="Lane A kernels with the submission model unchanged; the control for "
                    "whether kernel work alone is visible at the token boundary"),
        Config("A+G-few-buffers", "A+G", 8, 78, False, 3, 0.70,
               evidence="PROJECTED",
               note="Lane A plus a partial Lane G: three command buffers per token"),
        Config("A+G-one-graph", "A+G", 8, 78, False, 1, 0.70,
               evidence="PROJECTED",
               note="one command graph per token, no CPU in the hot loop"),
        Config("B1-compress-protected", "B", 8, 78, True, 1, 0.70,
               evidence="PROJECTED",
               blocked_on="indexer and router are CONTROL_SENSITIVE_CANDIDATE, "
                          "protected at source precision by the science stream",
               note="brings the two BF16 organs into the compact codec"),
        Config("D2-top2-experts", "D", 2, 78, True, 1, 0.70,
               evidence="PROJECTED",
               blocked_on="requires a distilled router/expert topology validated against "
                          "teacher trajectory; teacher capsules UNAVAILABLE",
               note="top-8 to top-2 routing with a stronger shared path"),
        Config("D3-top1-experts", "D", 1, 78, True, 1, 0.70,
               evidence="PROJECTED",
               blocked_on="same as D2, plus a shared functional student",
               note="top-1 plus generated correction"),
        Config("E2-depth-20", "D+E", 2, 20, True, 1, 0.70,
               evidence="PROJECTED",
               blocked_on="superblock distillation against teacher post-block trajectory; "
                          "teacher capsules UNAVAILABLE",
               note="78 to 20 stages, top-2 experts"),
        Config("E3-depth-10", "D+E", 2, 10, True, 1, 0.70,
               evidence="PROJECTED",
               blocked_on="same as E2, deeper collapse",
               note="78 to 10 stages, top-2 experts"),
        Config("F-native-functional", "F", 1, 8, True, 1, 0.70,
               evidence="PROJECTED",
               blocked_on="a Hawking-native student trained end to end; teacher capsules "
                          "UNAVAILABLE and this is the science stream's lane",
               note="native compact causal model, 8 stages, top-1"),
    ]


def analyse(ledger_path: Path) -> dict:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    organs = load_organs(ledger)
    projections = [project(organs, cfg) for cfg in candidate_configs()]

    baseline = projections[0]
    for row in projections:
        row["reduction_vs_shipped"] = round(
            baseline["active_bytes_per_token"] / max(1, row["active_bytes_per_token"]), 3)

    # Which milestones any projected configuration reaches, and which none do.
    milestones = {}
    for target in (100, 250, 500, 1000, 2000, 5000):
        reached = [r for r in projections if (r["tokens_per_second"] or 0) >= target]
        open_now = [r for r in reached if not r["blocked_on"]]
        milestones[str(target)] = {
            "any_projection_reaches": bool(reached),
            "reachable_without_a_blocked_lane": bool(open_now),
            "cheapest_reaching_config": reached[0]["config"]["name"] if reached else None,
            "blocked_on": reached[0]["blocked_on"] if reached else None,
        }

    protected = [o for o in organs if o.compressible]
    protected_bytes = sum(o.active_bytes for o in protected)

    return {
        "schema": SCHEMA,
        "evidence_level": "ARITHMETIC_OVER_A_SEALED_MEASURED_LEDGER",
        "not_evidence_of": "measured throughput, quality, or the reachability of any "
                           "configuration whose enabling artifact does not exist",
        "measured_constants": {
            "bandwidth_bytes_per_second": MEASURED_BANDWIDTH_BPS,
            "vendor_bandwidth_not_used": VENDOR_BANDWIDTH_BPS,
            "compute_flops": MEASURED_COMPUTE_FLOPS,
            "command_buffer_seconds": MEASURED_COMMAND_BUFFER_SECONDS,
            "dispatch_seconds": MEASURED_DISPATCH_SECONDS,
            "gpu_cores": GPU_CORES,
            "threadgroup_memory_bytes": THREADGROUP_MEMORY_BYTES,
        },
        "traffic_budget": traffic_budget_table(),
        "depth_budget": depth_budget_table(),
        "protected_organs": {
            "names": [o.name for o in protected],
            "active_bytes": protected_bytes,
            "share_of_token": round(protected_bytes / ledger["totals"]["active_bytes_per_token"], 5),
            "note": "stored at source precision, so they cost 16 bits per weight while the "
                    "routed experts cost 0.876",
        },
        "projections": projections,
        "milestones": milestones,
    }


def selftest() -> int:
    """The arithmetic's own invariants, on a synthetic ledger with known answers."""
    ledger = {
        "totals": {"active_bytes_per_token": 1_000_000},
        "per_organ": [
            {"organ": "routed_expert", "active_bytes": 800_000,
             "dense_bf16_bytes": 16_000_000, "dense_equivalent_macs": 8_000_000},
            {"organ": "attention", "active_bytes": 100_000,
             "dense_bf16_bytes": 2_000_000, "dense_equivalent_macs": 1_000_000},
            {"organ": "indexer", "active_bytes": 100_000,
             "dense_bf16_bytes": 100_000, "dense_equivalent_macs": 50_000},
        ],
    }
    organs = load_organs(ledger)
    by_name = {o.name: o for o in organs}

    # an organ stored at its BF16 size is protected, one stored below it is compressed
    assert by_name["indexer"].compressible, "equal bytes means source precision"
    assert not by_name["routed_expert"].compressible
    assert abs(by_name["routed_expert"].stored_bpw - 0.8) < 1e-9, by_name["routed_expert"].stored_bpw
    assert abs(by_name["indexer"].stored_bpw - 16.0) < 1e-9

    # halving the experts halves only the expert organ
    base = project(organs, Config("base", "x", 8, 78, False, 1, 1.0))
    half = project(organs, Config("half", "x", 4, 78, False, 1, 1.0))
    assert half["active_bytes_per_token"] == base["active_bytes_per_token"] - 400_000, half

    # halving the depth halves the per-layer organs and not lm_head
    ledger2 = dict(ledger)
    ledger2["per_organ"] = ledger["per_organ"] + [
        {"organ": "lm_head", "active_bytes": 200_000, "dense_bf16_bytes": 4_000_000,
         "dense_equivalent_macs": 2_000_000}]
    organs2 = load_organs(ledger2)
    full = project(organs2, Config("full", "x", 8, 78, False, 1, 1.0))
    shallow = project(organs2, Config("shallow", "x", 8, 39, False, 1, 1.0))
    head = next(r for r in shallow["organs"] if r["organ"] == "lm_head")
    assert head["bytes"] == 200_000, "lm_head runs once per token whatever the depth"
    assert shallow["active_bytes_per_token"] < full["active_bytes_per_token"]

    # depth binds when submission costs more than the traffic does
    deep = project(organs, Config("deep", "x", 8, 78, False, 78, 1.0))
    assert deep["binding_constraint"] == "DEPTH", deep["binding_constraint"]
    assert deep["depth_only_tps"] < 60, deep["depth_only_tps"]

    # Traffic binds once submission is collapsed, but only for a token big enough to out-cost
    # one command buffer: 215.8 us of submission buys 158.8 MB of reads at 736 GB/s, so any
    # token lighter than that is submission-bound however few buffers it uses.  The synthetic
    # ledger above is 1 MB and stays depth-bound at one buffer, which is itself the point.
    small_graph = project(organs, Config("small", "x", 8, 78, False, 1, 1.0))
    assert small_graph["binding_constraint"] == "DEPTH", "a 1 MB token cannot outrun one buffer"

    heavy = [{"organ": o["organ"], "active_bytes": o["active_bytes"] * 1000,
              "dense_bf16_bytes": o["dense_bf16_bytes"] * 1000,
              "dense_equivalent_macs": o["dense_equivalent_macs"] * 1000}
             for o in ledger["per_organ"]]
    graph = project(load_organs({"per_organ": heavy}),
                    Config("graph", "x", 8, 78, False, 1, 1.0))
    assert graph["binding_constraint"] == "TRAFFIC", graph["binding_constraint"]

    # the crossover itself, stated as a number rather than left implicit
    crossover = MEASURED_COMMAND_BUFFER_SECONDS * MEASURED_BANDWIDTH_BPS
    assert 1.5e8 < crossover < 1.7e8, crossover

    # 5,000 tok/s is under one command buffer, so no per-token submission reaches it
    depth = {row["target_tps"]: row for row in depth_budget_table()}
    assert depth[5000]["command_buffers_affordable"] < 1.0
    assert depth[5000]["verdict"] == "IMPOSSIBLE_WITH_PER_TOKEN_SUBMISSION"
    assert depth[1000]["verdict"] == "REQUIRES_ONE_COMMAND_GRAPH", depth[1000]["verdict"]

    # the traffic table is pure physics and must not depend on the ledger
    traffic = {row["target_tps"]: row for row in traffic_budget_table()}
    assert traffic[1000]["max_bytes_at_100pct"] == int(736e9 / 1000)

    # every projected row carries its own labels; the measured control carries none
    control = project(organs, Config("c", "control", evidence="MEASURED_LEDGER"))
    assert control["labels"] == [], control["labels"]
    graphed = project(organs, Config("p", "x", command_buffers_per_token=1,
                                     bandwidth_efficiency=0.70))
    assert graphed["labels"] == ["PROJECTED", "NOT_MEASURED",
                                 "ASSUMES_70_PERCENT_BANDWIDTH",
                                 "ASSUMES_ONE_COMMAND_GRAPH"], graphed["labels"]
    # a row that does not assume one graph must not claim it does
    staged = project(organs, Config("p3", "x", command_buffers_per_token=3))
    assert "ASSUMES_3_COMMAND_BUFFERS_PER_TOKEN" in staged["labels"], staged["labels"]
    assert "ASSUMES_ONE_COMMAND_GRAPH" not in staged["labels"]
    blocked = project(organs, Config("b", "x", command_buffers_per_token=1,
                                     blocked_on="teacher capsules"))
    assert "ENABLING_ARTIFACT_DOES_NOT_EXIST" in blocked["labels"]

    # Lane A alone must not move the token: submission still binds at 78 buffers
    kernels_only = project(organs, Config("a", "A", command_buffers_per_token=78))
    assert kernels_only["binding_constraint"] == "DEPTH"
    assert abs(kernels_only["tokens_per_second"] - 59.41) < 0.1, kernels_only["tokens_per_second"]

    print(json.dumps({"selftest": "PASS", "schema": SCHEMA,
                      "per_layer_submission_ceiling_tps":
                          depth[1000]["per_layer_submission_tps_ceiling"],
                      "five_thousand_tps_verdict": depth[5000]["verdict"]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traffic and depth budgets for a TPS target.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--ledger", default="reports/condense/breakthrough/GLM52_ACTIVE_BYTE_LEDGER.json")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    out = analyse(Path(args.ledger))
    text = json.dumps(out, indent=2)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
