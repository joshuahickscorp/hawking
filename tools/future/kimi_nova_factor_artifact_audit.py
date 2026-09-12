#!/usr/bin/env python3
"""Independently replay a persisted Nova factor payload against KIMI_BASE.

This audit is intentionally separate from the training process.  It verifies
that the saved BF16 factor bit patterns, complete-byte accounting, and direct
consumer reproduce the reported held-out organ result within BF16 storage
rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

from tools.future.kimi_representation_native_nova_fanout import _split_corpus  # noqa: E402
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_nova_factor_artifact_audit.v1"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bf16_bits_to_float(array: np.ndarray):
    import torch

    if array.dtype != np.uint16:
        raise RuntimeError(f"factor payload must be uint16 BF16 bits, got {array.dtype}")
    return torch.from_numpy(np.array(array, copy=True)).view(torch.bfloat16).float()


def audit(receipt_path: Path, corpus_path: Path, output: Path) -> dict[str, Any]:
    import torch

    receipt = json.loads(receipt_path.read_text())
    best = min(receipt["results"], key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    digest = _digest(artifact)
    if digest != best["factor_artifact"]["sha256"]:
        raise RuntimeError("factor artifact SHA-256 does not match the training receipt")

    row_rank = int(best["variant"]["row_rank"])
    col_rank = int(best["variant"]["col_rank"])
    residual_rank = int(best["variant"]["residual_rank"])
    names = _select_names(Path(receipt["specimen"]), int(receipt["expert_count"]),
                          receipt["tensor_family"])
    loaded, shards = _load_weights(Path(receipt["specimen"]), names)
    weights = [tensor.float() for _name, tensor, _shard in loaded]
    width = int(weights[0].shape[1])
    train, heldout, counts = _split_corpus(corpus_path, int(receipt["expert_count"]), width)
    del train

    with np.load(artifact, allow_pickle=False) as payload:
        required = {"u", "v"}
        required |= {f"core_{i}" for i in range(len(weights))}
        required |= {f"residual_a_{i}" for i in range(len(weights))}
        required |= {f"residual_b_{i}" for i in range(len(weights))}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"factor artifact missing arrays: {missing}")
        u = _bf16_bits_to_float(payload["u"])
        v = _bf16_bits_to_float(payload["v"])
        cores = [_bf16_bits_to_float(payload[f"core_{i}"]) for i in range(len(weights))]
        a = [_bf16_bits_to_float(payload[f"residual_a_{i}"]) for i in range(len(weights))]
        b = [_bf16_bits_to_float(payload[f"residual_b_{i}"]) for i in range(len(weights))]
        logical_payload_bytes = sum(int(payload[name].nbytes) for name in required)

    expected_shapes = [(int(weights[0].shape[0]), row_rank),
                       (width, col_rank)]
    if [tuple(u.shape), tuple(v.shape)] != expected_shapes:
        raise RuntimeError(f"shared factor shapes {u.shape}, {v.shape} != {expected_shapes}")
    for i, (core, aa, bb) in enumerate(zip(cores, a, b)):
        expected = [(row_rank, col_rank), (int(weights[i].shape[0]), residual_rank),
                    (width, residual_rank)]
        got = [tuple(core.shape), tuple(aa.shape), tuple(bb.shape)]
        if got != expected:
            raise RuntimeError(f"expert {i} factor shapes {got} != {expected}")

    relative: list[float] = []
    cosine: list[float] = []
    with torch.no_grad():
        for weight, core, aa, bb, xs in zip(weights, cores, a, b, heldout):
            for row in xs:
                x = torch.from_numpy(row)
                direct = u.matmul(core.matmul(v.transpose(0, 1).matmul(x)))
                direct = direct + aa.matmul(bb.transpose(0, 1).matmul(x))
                oracle = weight.matmul(x)
                relative.append(float(torch.linalg.vector_norm(direct - oracle).item()
                                     / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
                cosine.append(float(torch.nn.functional.cosine_similarity(
                    direct[None, :], oracle[None, :]).item()))

    accounting = best["representation"]["complete_accounting"]
    logical_components = best["representation"]["shared_output_basis_bytes"]
    logical_components += best["representation"]["shared_input_basis_bytes"]
    logical_components += best["representation"]["expert_core_bytes"]
    logical_components += best["representation"]["expert_local_residual_factor_bytes"]
    logical_components_match = int(logical_components) == logical_payload_bytes
    complete_bytes_match = int(accounting["executable_bytes"]) == logical_payload_bytes + 4096 + 65536
    reported = best["heldout"]
    remeasured = {"mean_relative_l2": statistics.mean(relative),
                  "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)],
                  "max_relative_l2": max(relative),
                  "mean_cosine": statistics.mean(cosine),
                  "total_heldout_rows": len(relative)}
    deltas = {key: remeasured[key] - reported[key]
              for key in ("mean_relative_l2", "p95_relative_l2", "max_relative_l2", "mean_cosine")}
    replay_match = (logical_components_match and complete_bytes_match
                    and abs(deltas["mean_relative_l2"]) <= 0.02
                    and abs(deltas["mean_cosine"]) <= 0.02)
    return {
        "schema": SCHEMA,
        "status": "NOVA_FACTOR_ARTIFACT_AUDIT_PASS" if replay_match else "NOVA_FACTOR_ARTIFACT_AUDIT_FAIL",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "training_receipt": str(receipt_path),
        "source_specimen": receipt["specimen"],
        "tensor_names": names,
        "shards": shards,
        "selected_variant": best["variant"],
        "factor_artifact": {"path": str(artifact), "sha256": digest,
                             "logical_payload_bytes": logical_payload_bytes,
                             "compressed_file_bytes": artifact.stat().st_size},
        "accounting_audit": {
            "factor_components_bytes": int(logical_components),
            "payload_matches_factor_components": logical_components_match,
            "recorded_complete_bytes": accounting["executable_bytes"],
            "expected_complete_bytes": logical_payload_bytes + 4096 + 65536,
            "complete_bytes_match": complete_bytes_match,
            "dense_parent_materialized": False,
        },
        "activation_corpus": {"path": str(corpus_path), "rows": counts,
                              "split": "same interleaved split as training receipt"},
        "reported_heldout": reported,
        "remeasured_heldout": remeasured,
        "metric_deltas_replayed_minus_reported": deltas,
        "claim_boundary": (
            "Independent replay of a saved BF16 factor payload on the bounded KIMI organ. "
            "This verifies artifact/accounting/direct-consumer consistency only; it is not "
            "full-model capability, end-to-end TPS, NX, or promotion evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = audit(args.receipt, args.corpus, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "selected_variant": out["selected_variant"],
                      "remeasured_heldout": out["remeasured_heldout"],
                      "accounting_audit": out["accounting_audit"]}, indent=2))
    return 0 if out["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
