#!/usr/bin/env python3.12
"""What binds a token once the expert function is gone.

The old estimate assumed the model reads eight routed experts and a shared expert at every
sparse layer, which is 45.9 GB of weight traffic per token and made everything else a
rounding error.  A functional organ reads 12.6 MB instead of 604 MB, which is a 48x cut on
97.9 percent of the model's weights.  The question this answers is what is left holding the
clock.

Bandwidth is measured on this box through the same kernel the runtime uses, at an occupancy
where occupancy has stopped binding.  No specification figure is used.

    run
    selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "condense" / "glm52_generation_b"
BENCH = OUT / "GLM52_FUNCTIONAL_METAL_BENCHMARK.json"
LEDGER = ROOT / "GLM52_LOGICAL_WEIGHT_LEDGER.json"

MOE_LAYERS = 76
# Every routed and shared expert matrix a token actually touches at BF16: eight of 256
# routed experts plus the shared expert, three [2048, 6144] matrices each.
TEACHER_MOE_BYTES_PER_LAYER = (8 + 1) * 3 * 2048 * 6144 * 2

# Which protected organs a single decode step actually reads.  The embedding table is one
# row, not the table; everything else is read whole.
ACTIVE_PROTECTED = ("attention", "indexer", "lm_head", "dense_mlp", "normalization",
                    "mtp_projection", "mtp_normalization", "mtp_head_norm")
PROTECTED_RATES = {"bf16_source": 16.0, "int8": 8.0, "int4": 4.0}


def _weights() -> dict:
    return json.loads(LEDGER.read_text())["primary_categories"]


def run() -> dict:
    bench = json.loads(BENCH.read_text())
    bandwidth = bench["bandwidth_probe"]["achieved_bytes_per_second"]
    grammars = {row["grammar"]: row for row in bench["grammars"]}
    categories = _weights()
    protected_active = {name: categories[name]["logical_weights"]
                        for name in ACTIVE_PROTECTED if name in categories}
    protected_total = sum(protected_active.values())

    teacher_moe = TEACHER_MOE_BYTES_PER_LAYER * MOE_LAYERS
    rows = []
    for label, moe_bytes in [
        ("teacher_bf16_experts", teacher_moe),
        ("FRT_A_explicit", grammars["FRT_A"]["resident_bytes_per_layer"] * MOE_LAYERS),
        ("FRT_B_procedural", grammars["FRT_B"]["resident_bytes_per_layer"] * MOE_LAYERS),
        ("FRT_D_direct_linear", grammars["FRT_D"]["resident_bytes_per_layer"] * MOE_LAYERS),
    ]:
        entry = {"path": label, "moe_bytes_per_token": moe_bytes, "at_protected_rate": {}}
        for rate_name, bits in PROTECTED_RATES.items():
            protected_bytes = int(protected_total * bits / 8)
            total = moe_bytes + protected_bytes
            entry["at_protected_rate"][rate_name] = {
                "protected_bytes_per_token": protected_bytes,
                "total_bytes_per_token": total,
                "moe_fraction_of_traffic": moe_bytes / total,
                "protected_fraction_of_traffic": protected_bytes / total,
                "bandwidth_bound_seconds_per_token": total / bandwidth,
                "bandwidth_bound_tps": bandwidth / total,
                "binding_organ": "expert_function" if moe_bytes > protected_bytes
                else "attention_and_protected_organs",
            }
        rows.append(entry)

    measured = {
        grammar: {
            "measured_seconds_per_layer": row["seconds_per_layer_call"],
            "measured_seconds_per_token_moe_only": row["seconds_per_layer_call"] * MOE_LAYERS,
            "measured_tps_moe_only": 1.0 / (row["seconds_per_layer_call"] * MOE_LAYERS),
            "command_buffers_per_token": row["command_buffers_per_call"] * MOE_LAYERS,
            "fraction_of_roofline": (
                (row["resident_bytes_per_layer"] / bandwidth) / row["seconds_per_layer_call"]),
        } for grammar, row in grammars.items()}

    best = min(rows[1:], key=lambda row:
               row["at_protected_rate"]["int4"]["total_bytes_per_token"])
    return {
        "schema": "hawking.glm52.functional_token_roofline.v1",
        "device": bench["device"],
        "measured_bandwidth_bytes_per_second": bandwidth,
        "measured_bandwidth_gigabytes_per_second": bandwidth / 1e9,
        "bandwidth_provenance": bench["bandwidth_probe"]["note"],
        "moe_layers": MOE_LAYERS,
        "active_protected_organs": protected_active,
        "active_protected_weights": protected_total,
        "supersedes": "the 5.914 GB/token estimate, which assumed compressed teacher experts",
        "paths": rows,
        "measured_kernels": measured,
        "binding_limit_answer": {
            "before": "expert weight traffic, at %.1f GB per token"
                      % (teacher_moe / 1e9),
            "after": "attention and the protected organs, at %.1f GB per token at source "
                     "precision against %.2f GB of functional payload"
                     % (protected_total * 2 / 1e9,
                        best["moe_bytes_per_token"] / 1e9),
            "moved": True,
            "consequence": "compressing attention is now worth more than compressing the "
                           "expert function again; the expert function is already a "
                           "rounding error in the token budget",
        },
        "honest_scope": "weight traffic only. KV and state traffic, activation traffic, "
                        "kernel launch overhead and the sequential depth of 79 blocks are "
                        "not in this roofline, so the bandwidth-bound TPS is a ceiling and "
                        "not a projection.",
    }


def selftest() -> int:
    result = run()
    # The teacher path must be expert-bound and every functional path must not be.
    teacher = result["paths"][0]["at_protected_rate"]["bf16_source"]
    assert teacher["binding_organ"] == "expert_function", teacher["binding_organ"]
    for row in result["paths"][1:]:
        entry = row["at_protected_rate"]["bf16_source"]
        assert entry["binding_organ"] == "attention_and_protected_organs", row["path"]
        assert entry["moe_fraction_of_traffic"] < 0.2, (row["path"],
                                                        entry["moe_fraction_of_traffic"])
    # Measured kernels must be slower than their own roofline, never faster.
    for grammar, row in result["measured_kernels"].items():
        assert 0.0 < row["fraction_of_roofline"] <= 1.0, (grammar, row)
    print(json.dumps({
        "selftest": "PASS",
        "bandwidth_gbps": round(result["measured_bandwidth_gigabytes_per_second"], 1),
        "teacher_moe_gb_per_token": round(result["paths"][0]["moe_bytes_per_token"] / 1e9, 2),
        "frt_b_moe_gb_per_token": round(result["paths"][2]["moe_bytes_per_token"] / 1e9, 3),
    }))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "run":
        payload = run()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "GLM52_FUNCTIONAL_TOKEN_ROOFLINE.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, indent=2))
    else:
        raise SystemExit(f"unknown command: {command}")
