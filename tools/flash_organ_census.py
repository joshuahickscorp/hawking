#!/usr/bin/env python3
"""Batched, metadata-first Flash organ census for Doctor/NR prioritization.

This deliberately avoids materializing the multi-GB model.  It inventories
all indexed tensors, groups them by semantic organ/layer/expert family, and
reports byte shares and candidate representation targets.  It is a structural
screen, not a parity or capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc"))
    ap.add_argument("--out", type=Path, default=Path("receipts/headless/FLASH_ORGAN_CENSUS.json"))
    a = ap.parse_args()
    started = time.perf_counter_ns()
    root = a.root.resolve()
    index_path = root / "model.safetensors.index.json"
    idx_bytes = index_path.read_bytes()
    idx = json.loads(idx_bytes)
    tensors = idx["weight_map"]
    shards = Counter(tensors.values())
    records = []
    family_bytes = defaultdict(int)
    family_tensors = Counter()
    layers = Counter()
    experts = Counter()
    total = 0
    # Read only safetensors headers; payloads remain untouched.
    for shard, names in shards.items():
        path = root / shard
        with path.open("rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        for name in (n for n, s in tensors.items() if s == shard):
            meta = header[name]
            begin, end = meta["data_offsets"]
            size = end - begin
            total += size
            m = re.search(r"layers\.(\d+)", name)
            layer = int(m.group(1)) if m else None
            if layer is not None:
                layers[layer] += size
            if "ngram_embedding" in name:
                family = "ngram_embedding"
            elif ".experts." in name:
                family = "routed_experts"
                experts[family] += size
            elif ".shared_expert" in name:
                family = "shared_expert"
            elif ".linear_attn." in name or ".attn_hyper_connection" in name:
                family = "linear_attention_hyperconnection"
            elif ".self_attn." in name:
                family = "full_attention"
            elif ".mlp." in name or ".mlp_hyper_connection" in name:
                family = "mlp_hyperconnection"
            elif "embed" in name or "lm_head" in name:
                family = "embedding_lm_head"
            elif "norm" in name:
                family = "norm"
            else:
                family = "other"
            family_bytes[family] += size
            family_tensors[family] += 1
            records.append({"name": name, "layer": layer, "shape": meta.get("shape"), "dtype": meta.get("dtype"), "bytes": size, "family": family, "shard": shard})
    largest = sorted(records, key=lambda r: r["bytes"], reverse=True)[:32]
    doc = {
        "schema": "hawking.flash.organ_census.v1",
        "status": "STRUCTURAL_METADATA_SCREEN",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_source": str(root),
        "source_index_sha256": hashlib.sha256(idx_bytes).hexdigest(),
        "tensor_count": len(records),
        "layer_count_observed": len(layers),
        "source_parameter_bytes_indexed": total,
        "shard_count": len(shards),
        "shard_tensor_count": dict(shards),
        "family_summary": [
            {"family": k, "tensor_count": family_tensors[k], "bytes": v, "fraction": v / total if total else 0.0}
            for k, v in sorted(family_bytes.items(), key=lambda kv: kv[1], reverse=True)
        ],
        "layer_bytes": {str(k): v for k, v in sorted(layers.items())},
        "largest_tensors": largest,
        "doctor_priority": [
            {"family": "routed_experts", "reason": "largest mutable byte family; test shared basis/archetype/sparse residual and route-conditioned generation"},
            {"family": "linear_attention_hyperconnection", "reason": "recurrent state and projection fusion; test state/program sharing across layers"},
            {"family": "full_attention", "reason": "KV-sensitive; test fused QKV/SDPA and compact cache without host ceremony"},
        ],
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": "tools/flash_organ_census.py",
            "machine": "Apple M3 Ultra (metadata/header scan)",
            "rule": "S032 §3 -- structural census timing is not a model performance benchmark",
            "provenance": "No payload execution or throughput claim; index/header scan only",
        },
        "claim_boundary": "Index/header-only census. No reconstruction, native execution, complete-token capability, EBPW, TPS, or promotion claim.",
        "next": "Feed family populations to Doctor Stage A, batch candidate fits, then native-organ qualify only Pareto survivors.",
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "tensor_count": len(records), "bytes": total, "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
