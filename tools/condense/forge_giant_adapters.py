#!/usr/bin/env python3.12
"""Giant-parent Forge adapter contracts (Section 9), composed from SHARED primitives (Section 19).

Reads the already-bound read-only source authority (no downloads, no giant model execution) and
freezes one adapter contract per giant parent (~685B / ~1T / ~1.6T). Adapters DECLARE tensor
mappings, geometry, and primitive composition - they are not one custom implementation per parent.
Shared primitives are reused wherever semantics truly match; genuine differences (multimodal vision
tower, MTP head, source precision) are respected, not abstracted away.

Nothing here runs or downloads a giant parent. The contracts stabilize the interface the heavy run
will use later; they are launch-gated and carry no capability claim.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

AUTHORITY = "reports/condense/event_horizon_successor/frontier/source_authority.json"
OUT = Path("reports/condense/gravity_forge/giant_adapters")
SCHEMA = "hawking.gravity_forge.giant_adapter_contract.v1"

# audited Gravity sub-bit stress priors (GRAVITY_STATE.json / directive Section 9)
STRESS = {"deepseek-v3.2-685b": "11/20", "kimi-k2.6-1t": "1/3", "deepseek-v4-pro-1.6t": "1/4"}

REQUIRED = ("architecture_primitives", "tensor_taxonomy", "router_and_expert_mapping",
            "shared_expert_handling", "mtp_or_vision_boundaries", "source_precision",
            "streamed_source_interface", "forge_hooks", "doctor_hooks", "runtime_hooks",
            "physical_byte_model", "gravity_stress_rate", "queue_identity")

# shared primitive library (Section 19) - reused across parents
PRIMS = ["dense_transformer_trunk", "moe_router", "shared_expert", "routed_expert",
         "mla_attention", "mtp_head", "vision_tower", "safetensors_source",
         "compressed_tensor_source", "streamed_source", "compact_expert_runtime", "expert_paging"]


def _source_decoder(precision: str) -> str:
    p = precision.lower()
    if "int4" in p or "compressed_tensors" in p:
        return "compressed_tensor_source(int4)"
    if "fp4" in p or "fp8" in p:
        return "compressed_tensor_source(fp4/fp8)"
    return "safetensors_source(bf16)"


def compose(parent: dict[str, Any]) -> dict[str, Any]:
    rid = parent["row_id"]
    mm = bool(parent.get("multimodal"))
    mtp = int(parent.get("mtp_layers") or 0)
    # primitive composition: reuse shared primitives, include vision/mtp only where the geometry has them
    prims = ["dense_transformer_trunk", "mla_attention", "moe_router", "routed_expert",
             "shared_expert" if parent.get("n_shared_experts") else None,
             "mtp_head" if mtp > 0 else None, "vision_tower" if mm else None,
             _source_decoder(parent["source_precision"]).split("(")[0],
             "streamed_source", "compact_expert_runtime", "expert_paging"]
    prims = [p for p in prims if p]
    contract = {
        "schema": SCHEMA, "row_id": rid, "hf_id": parent["hf_id"],
        "architecture": parent["architecture"],
        "architecture_primitives": prims,
        "tensor_taxonomy": {
            "trunk": ["embedding", "final_norm", "unembedding", "attn.norm", "mlp.norm"],
            "attention": ["attn.qkv", "attn.out", "attn.sinks(if_present)"],
            "router": ["mlp.gate.weight", "mlp.gate.bias"],
            "routed_expert": ["mlp.{gate,up,down}"], "shared_expert": ["shared.mlp.*"],
            "mtp": ["mtp.*"] if mtp > 0 else [], "vision": ["vision.*"] if mm else [],
        },
        "router_and_expert_mapping": {
            "n_routed_experts": parent.get("n_routed_experts"),
            "experts_per_tok": parent.get("experts_per_tok"),
            "n_shared_experts": parent.get("n_shared_experts"),
        },
        "shared_expert_handling": ("always-on shared expert added to every token; protected organ "
                                   "(candidate Doctor precision island)"),
        "mtp_or_vision_boundaries": {"mtp_layers": mtp, "multimodal": mm,
                                     "note": "MTP/vision tensors are pass-through unless capability "
                                             "eval requires them; excluded from sub-bit stress first"},
        "source_precision": parent["source_precision"],
        "streamed_source_interface": {
            "decoder": _source_decoder(parent["source_precision"]),
            "access": "bounded source windows (per-expert / per-shard); no full download",
            "n_shards": parent.get("n_shards"), "source_bytes": parent.get("source_bytes"),
        },
        "forge_hooks": {
            "routed_expert": ["transform_pq", "shared_expert_grammar", "repairability_shaped",
                              "ternary_factor"],
            "shared_expert": ["repairability_shaped(protected)"],
            "router": ["pass_through(fp16)"], "trunk": ["pass_through(fp16)"],
        },
        "doctor_hooks": ["protected_precision_islands", "router_protection", "shared_expert_protection",
                         "rare_expert_protection", "low_rank_correction", "structured_sparse_correction"],
        "runtime_hooks": {"expert_paging": True, "compact_expert_runtime": True,
                          "selected_expert_fusion": "planned", "reconstructs_dense_model": False},
        "physical_byte_model": {
            "unit": "whole_artifact_bpw = total_physical_bits / n_weights",
            "counts": ["indices", "codebooks", "factors", "scales", "transform_seeds",
                       "doctor_corrections", "router", "shared_expert", "trunk_passthrough",
                       "metadata", "alignment"],
            "primary_claim": "installed whole-artifact BPW",
        },
        "gravity_stress_rate": STRESS.get(rid, "1/2"),
        "queue_identity": {"parent_label": rid.split("-")[-1].upper(),
                           "row_id": rid, "lane": "hawking-gravity-forge", "launch_gate": "disabled"},
        "launch": "DISABLED (contract only; no giant run, no download)",
    }
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in contract.items() if k != "contract_sha256"},
                   sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return contract


def build() -> dict[str, Any]:
    auth = json.load(open(AUTHORITY))
    OUT.mkdir(parents=True, exist_ok=True)
    contracts, index = [], {}
    for parent in auth["parents"]:
        c = compose(parent)
        missing = [f for f in REQUIRED if f not in c or c[f] in (None, "", [], {})]
        c_valid = not missing
        (OUT / f"{c['row_id']}.json").write_text(json.dumps(c, indent=2, sort_keys=True, default=str))
        index[c["row_id"]] = {"valid": c_valid, "missing": missing,
                              "sha256": c["contract_sha256"][:16], "hf_id": c["hf_id"]}
        contracts.append(c)
    all_valid = all(v["valid"] for v in index.values())
    stable = {"schema": "hawking.gravity_forge.giant_adapters.stable.v1",
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "shared_primitive_library": PRIMS, "required_fields": list(REQUIRED),
              "adapters": index, "all_contracts_valid": all_valid,
              "composed_from": "read-only source authority; no giant download or execution",
              "source_authority_sha256": auth.get("authority_sha256", "")[:16]}
    stable["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in stable.items() if k != "sha256"},
                   sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    (OUT / "STABLE.json").write_text(json.dumps(stable, indent=2, sort_keys=True, default=str))
    return stable


def main(argv: list[str] | None = None) -> int:
    s = build()
    print(f"giant adapter contracts: all_valid={s['all_contracts_valid']}")
    for rid, v in s["adapters"].items():
        print(f"  {'OK ' if v['valid'] else 'BAD'} {rid:24s} {v['hf_id']}" +
              (f"  missing={v['missing']}" if v["missing"] else ""))
    return 0 if s["all_contracts_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
