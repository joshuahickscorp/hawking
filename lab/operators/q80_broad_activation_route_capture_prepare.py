#!/usr/bin/env python3
"""Prepare a broad Q80 all-layer activation capture input.

Reuses the sealed Q30 broad corpus token IDs because Qwen3-Coder-30B and
Qwen3-Coder-Next share an identical tokenizer.json (sha256 verified at prepare
time). Does not execute the model. Diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")

DEFAULT_Q30_INPUT = (
    MAIN_HAWKING
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30"
    / "quality-diagnostics/broad-activation-v1/requests"
    / "QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_901a24bdcfc6c1d2.json"
)
DEFAULT_TOKENIZER = (
    MAIN_HAWKING
    / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json"
)
DEFAULT_Q30_TOKENIZER = (
    MAIN_HAWKING
    / "workspace/campaign/records/runs/qwen-30b"
    / "Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-diagnostics/all-layer-activation-v1"
)

INPUT_SCHEMA = "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_input.v1"
PREPARE_SCHEMA = "hawking.ascension.qwen80_broad_activation_capture_prepare.v1"
STATUS = "NEW_DIAGNOSTIC_NOT_HISTORICAL"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q30-input", type=Path, default=DEFAULT_Q30_INPUT)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--q30-tokenizer-json", type=Path, default=DEFAULT_Q30_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-prompts", type=int, default=0)
    args = parser.parse_args()

    q30_path = args.q30_input.expanduser().resolve()
    tok80 = args.tokenizer_json.expanduser().resolve()
    tok30 = args.q30_tokenizer_json.expanduser().resolve()
    if not q30_path.is_file():
        print(f"missing Q30 broad input: {q30_path}", flush=True)
        return 2
    if not tok80.is_file():
        print(f"missing Q80 tokenizer: {tok80}", flush=True)
        return 2

    tok80_sha = sha256_file(tok80)
    tok30_sha = sha256_file(tok30) if tok30.is_file() else None
    if tok30_sha is not None and tok30_sha != tok80_sha:
        print(
            "Q30/Q80 tokenizer sha mismatch; refuse to reuse token ids "
            f"(q30={tok30_sha} q80={tok80_sha})",
            flush=True,
        )
        return 2

    src = json.loads(q30_path.read_text(encoding="utf-8"))
    probes = list(src.get("probes") or [])
    if args.max_prompts > 0:
        probes = probes[: int(args.max_prompts)]
    if len(probes) < 12:
        print("need at least 12 probes for broad schema", flush=True)
        return 2

    domain_counts: dict[str, int] = {}
    directory_counts: dict[str, int] = {}
    for probe in probes:
        domain = str(probe.get("domain") or "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        dkey = str(probe.get("source_dir") or "handwritten")
        directory_counts[dkey] = directory_counts.get(dkey, 0) + 1

    total_tokens = sum(
        int((p.get("source_one_user_native_prompt") or {}).get("token_count") or 0)
        for p in probes
    )
    token_counts = [
        int((p.get("source_one_user_native_prompt") or {}).get("token_count") or 0)
        for p in probes
    ]
    ordered_counts = sorted(token_counts)
    if ordered_counts:
        mid = len(ordered_counts) // 2
        if len(ordered_counts) % 2 == 1:
            median_tokens: float = float(ordered_counts[mid])
        else:
            median_tokens = (ordered_counts[mid - 1] + ordered_counts[mid]) / 2.0
    else:
        median_tokens = 0.0

    document: dict[str, Any] = {
        "schema": INPUT_SCHEMA,
        "status": STATUS,
        "prepared_at": utc_now(),
        "purpose": (
            "broad multi-probe token set for Q80 all-layer router-input "
            "activation capture so constant-mean null can be priced before "
            "surplus-first activation_weighted_svd packing"
        ),
        "claim_boundary": {
            "model_execution_started": False,
            "new_diagnostic_not_historical": True,
            "diagnostic_activation_pricing_only": True,
            "does_not_claim_coherence_hcli_tps_or_capability": True,
            "not_hcli_compiler_trace_bound": True,
            "source_tokenizer_one_user_native_prompts": True,
            "production_server_not_used": True,
            "all_layer_intent": True,
            "reuse_of_q30_token_ids_requires_identical_tokenizer_sha": True,
        },
        "tokenizer_binding": {
            "path": str(tok80),
            "sha256": tok80_sha,
            "identical_to_q30_tokenizer": tok30_sha == tok80_sha if tok30_sha else None,
            "q30_tokenizer_path": str(tok30) if tok30.is_file() else None,
            "q30_tokenizer_sha256": tok30_sha,
        },
        "provenance": {
            "q30_input_path": str(q30_path),
            "q30_input_sha256": sha256_file(q30_path),
            "q30_input_schema": src.get("schema"),
            "reuse_policy": "identical_tokenizer_sha_token_id_reuse",
        },
        "capture_intent": {
            "target_layers": list(range(48)),
            "require_all_layer": True,
            "bounded_storage": {
                "full_route_membership_all_tokens_all_layers": True,
                "raw_hidden_strategy": "stratified_token_subsample",
                "default_max_hidden_tokens_per_layer": 1024,
            },
            "blocked_until": [
                "gqa_full_layer_same_runtime_encode",
                "broad_activation_capture_binary",
            ],
        },
        "corpus_summary": {
            "probe_count": len(probes),
            "domain_counts": domain_counts,
            "directory_counts": directory_counts,
            "total_tokens": total_tokens,
            "min_tokens": min(token_counts) if token_counts else 0,
            "median_tokens": median_tokens,
            "max_tokens": max(token_counts) if token_counts else 0,
            "mean_tokens": round(sum(token_counts) / max(len(token_counts), 1), 2),
        },
        "probes": probes,
    }

    out_dir = args.out_dir.expanduser().resolve()
    req_dir = out_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = sha256_bytes(body.encode("utf-8"))
    out_path = req_dir / f"QWEN80_BROAD_ACTIVATION_ALL_LAYER_ROUTE_CAPTURE_INPUT_{digest[:16]}.json"
    out_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    file_sha = sha256_file(out_path)

    provenance = {
        "schema": PREPARE_SCHEMA,
        "prepared_at": utc_now(),
        "input_path": str(out_path),
        "input_sha256": file_sha,
        "corpus_summary": document["corpus_summary"],
        "tokenizer_binding": document["tokenizer_binding"],
        "provenance": document["provenance"],
        "capture_intent": document["capture_intent"],
        "claim_boundary": document["claim_boundary"],
        "note": (
            "Input only. Capture binary cannot run until GQA full-layer "
            "same-runtime encode is ready. Do not disturb live servers."
        ),
    }
    (out_dir / "PREPARE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "input_path": str(out_path),
                "input_sha256": file_sha,
                **document["corpus_summary"],
                "tokenizer_identical_to_q30": document["tokenizer_binding"][
                    "identical_to_q30_tokenizer"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
