"""Emit the Qwen27 token-budget scaffold from existing static receipts.

The byte atlas is an exact catalog attribution for the sealed Qwen27 control,
not a GPU trace.  This receipt makes the campaign's latency taxonomy explicit
without turning contaminated historical observations into an authoritative
baseline.  Native protected execution is the only writer of actual timing,
resident, active-read, synchronization, or accepted-token fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hcli.persist import atomic_write_json
from tools.accelerator.scoreboard import normalize_receipt


SCHEMA = "hawking.accelerator.qwen27_token_ns_budget.v1"
DEFAULT_OUT = Path("receipts/headless/QWEN27_TOKEN_NS_BUDGET.json")
CONTROL_RECEIPT = Path("receipts/headless/HCLI_PROTECTED_ACCELERATOR_BENCHMARK_AFTER_FLASH.json")
BYTE_ATLAS = Path("receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json")
REQUIRED_METRICS = (
    "total_nx_bytes",
    "resident_bytes",
    "active_representation_bytes_per_token",
    "actual_read_bytes_per_token",
    "transient_bytes_per_token",
    "gpu_ns_per_token",
    "complete_wall_ns_per_accepted_token",
    "dispatches_per_token",
    "sync_ns_per_token",
    "accepted_tps",
    "fallback_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _empty_metric_row(organ: str, *, source_bytes: int | None = None, source_dispatches: int | None = None) -> dict[str, Any]:
    return {
        "organ": organ,
        "source_weight_bytes_per_token": source_bytes,
        "source_dispatches_per_token": source_dispatches,
        "actual": {name: None for name in REQUIRED_METRICS},
        "status": "WAITING_FOR_NATIVE_PROTECTED_EXECUTION",
        "absence_reasons": {
            name: "native provider has not emitted a quiet protected complete-token receipt"
            for name in REQUIRED_METRICS
        },
    }


def _control_observation(repo: Path) -> dict[str, Any]:
    path = repo / CONTROL_RECEIPT
    if not path.is_file():
        return {
            "status": "NOT_FOUND",
            "receipt": str(CONTROL_RECEIPT),
            "claim_boundary": "no control observation was available",
        }
    payload = _load(path)
    normalized = normalize_receipt(path, payload, root=repo)
    return {
        "status": "PROTECTED_CONTROL_NOT_FOR_PROMOTION",
        "receipt": str(CONTROL_RECEIPT),
        "receipt_sha256": _sha256(path),
        "benchmark_class": normalized.get("benchmark_class"),
        "bench_state": normalized.get("bench_state"),
        "metrics": {
            "complete_wall_ns_per_token": normalized.get("wall_ns_per_token"),
            "gpu_ns_per_token": normalized.get("gpu_ns_per_token"),
            "dispatches_per_token": normalized.get("dispatches_per_token"),
            "accepted_tps": normalized.get("accepted_tps"),
        },
        "claim_boundary": "quiet protected control evidence; NOT_FOR_PROMOTION and not a mutation result",
    }


def build_budget(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else REPO
    atlas_path = repo / BYTE_ATLAS
    atlas = _load(atlas_path)
    headline = atlas.get("headline") if isinstance(atlas.get("headline"), Mapping) else {}
    regions = atlas.get("pareto_by_bytes")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Qwen27 byte atlas has no pareto_by_bytes rows")

    source_regions = []
    for row in regions:
        if not isinstance(row, Mapping):
            raise ValueError("Qwen27 byte atlas region rows must be objects")
        source_regions.append({
            "kernel": row.get("kernel"),
            "dispatches_per_token": row.get("dispatches"),
            "weight_bytes_per_token": row.get("weight_bytes"),
            "bytes_per_dispatch": row.get("bytes_per_dispatch"),
            "roles": list(row.get("roles") or []),
            "label": "STATIC_DERIVATION_FROM_CATALOG",
        })

    organs = [
        _empty_metric_row("representation_access"),
        _empty_metric_row("qkv_and_projection"),
        _empty_metric_row("attention"),
        _empty_metric_row("deltanet_and_recurrent_state"),
        _empty_metric_row("mlp"),
        _empty_metric_row("lm_head_and_sampling"),
        _empty_metric_row("dispatch_encode"),
        _empty_metric_row("command_wait_and_synchronization"),
        _empty_metric_row("host_ceremony"),
    ]

    return {
        "schema": SCHEMA,
        "status": "PLANNED_UNTIL_NATIVE_PROTECTED_EXECUTION",
        "label": "DERIVED",
        "model": "qwen3.8-27b-sealed-3.14",
        "baseline": {
            "profile": "hcli/hawking-native.sealed-3.14.json",
            "representation": "native-packed sealed control",
            "byte_atlas": str(BYTE_ATLAS),
            "byte_atlas_sha256": _sha256(atlas_path),
        },
        "source_byte_denominator": {
            "active_weight_bytes_per_token": headline.get("active_weight_bytes_per_token"),
            "active_ebpw_per_token": headline.get("active_ebpw_per_token"),
            "complete_ebpw": headline.get("complete_ebpw"),
            "regions": source_regions,
            "claim_boundary": "catalog-derived weight traffic only; activations, KV, and recurrent state are not included",
        },
        "organs": organs,
        "lifecycle_buckets": {
            name: None
            for name in (
                "cold_load_ns",
                "warm_start_ns",
                "first_token_ns",
                "warm_decode_token_ns",
                "steady_state_decode_token_ns",
                "accepted_complete_token_ns",
            )
        },
        "system_ledger": {name: None for name in REQUIRED_METRICS},
        "control_observation": _control_observation(repo),
        "measurement_protocol": {
            "same_tokenizer_and_output_contract": True,
            "cold_warm_first_warm_steady_state_separated": True,
            "complete_accepted_token_denominator": True,
            "protected_quiescent_before_and_after": True,
            "native_kernel_genome_and_dispatch_trace": True,
            "active_bytes_include_actual_read_and_transient_fields": True,
            "diagnostic_relative_runs_cannot_promote": True,
        },
        "promotion_allowed": False,
        "bench": {
            "state": "UNKNOWN",
            "machine": "planning artifact; this command executes no GPU work",
            "claim_boundary": "static byte derivation and control metadata are not physical qualification",
        },
        "claim_boundary": "No Qwen27 latency, TPS, GPU, active-read, resident, or accepted-token claim is made until a native protected complete-token receipt fills the ledger.",
    }


def emit_budget(*, output: str | Path | None = None, repo_root: str | Path | None = None) -> Path:
    repo = Path(repo_root).expanduser().resolve() if repo_root else REPO
    destination = Path(output).expanduser() if output else repo / DEFAULT_OUT
    if not destination.is_absolute():
        destination = repo / destination
    body = build_budget(repo_root=repo)
    atomic_write_json(destination, body)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--emit", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    destination = emit_budget(output=args.emit, repo_root=args.repo_root)
    body = _load(destination)
    print(json.dumps({
        "status": "PASSED",
        "path": str(destination),
        "schema": body["schema"],
        "source_active_weight_bytes_per_token": body["source_byte_denominator"]["active_weight_bytes_per_token"],
        "claim_boundary": body["claim_boundary"],
    }, sort_keys=True))
    return 0


__all__ = ["DEFAULT_OUT", "REQUIRED_METRICS", "SCHEMA", "build_budget", "emit_budget", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
