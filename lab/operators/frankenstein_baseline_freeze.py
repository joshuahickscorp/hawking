#!/usr/bin/env python3.12
"""Stage 1 — freeze exact BASE_DSV4F capability / runtime / routing / HCLI baseline.

Records what is measurable from the existing multi-layer BOS forward receipts
and pinned geometries.  Fields that cannot be measured yet are marked PENDING
honestly (never fabricated).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_fusion_op import (
    DEEPSEEK_V4_FLASH,
    TRANSPLANT_POINT_NAMES,
)
from lab.operators.frankenstein_gates import LINEAR_SUBSPACE_INITIALIZATION
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
EVIDENCE_ROOT = WORKSPACE_ROOT / "campaign" / "evidence" / "models" / "frankenstein"
RECEIPTS_DIR = REPO_ROOT / "receipts"

BASELINE_SCHEMA = "hawking.frankenstein.base_dsv4f_baseline_freeze.v1"
DEFAULT_OUT = EVIDENCE_ROOT / "BASE_DSV4F_BASELINE_FREEZE.json"

# Known BOS multi-layer forward receipts (repo-local evidence).
DEFAULT_FORWARD_RECEIPTS: tuple[str, ...] = (
    "dsv4f_multi_layer_gpu_forward_bos_l0_l42_receipt.json",
    "dsv4f_multi_layer_gpu_forward_bos_l0_l42_greedy_receipt.json",
    "dsv4f_multi_layer_gpu_forward_bos_l0_l3_receipt.json",
    "dsv4f_multi_layer_gpu_forward_bos_l0_l2_receipt.json",
    "dsv4f_learned_bias_route_metal_receipt.json",
)


class BaselineFreezeError(RuntimeError):
    """Baseline freeze failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise BaselineFreezeError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise BaselineFreezeError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    _ensure_dir(path.parent)
    if path.exists():
        _regular_file(path, f"existing {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise BaselineFreezeError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return hashlib.sha256(existing).hexdigest()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return hashlib.sha256(encoded).hexdigest()


def _bind_receipt(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "present": False,
            "status": "MISSING",
        }
    _regular_file(path, str(path))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineFreezeError(f"receipt not JSON: {path}: {exc}") from exc
    honesty = doc.get("honesty") if isinstance(doc, Mapping) else None
    metal = doc.get("metal") if isinstance(doc, Mapping) else None
    return {
        "path": str(path.resolve()),
        "present": True,
        "file_sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "final_child_layer": doc.get("final_child_layer") if isinstance(doc, Mapping) else None,
        "honesty": dict(honesty) if isinstance(honesty, Mapping) else None,
        "metal_summary": (
            {
                "command_buffers": metal.get("command_buffers"),
                "fallback": metal.get("fallback"),
            }
            if isinstance(metal, Mapping)
            else None
        ),
        "artifact": doc.get("artifact") if isinstance(doc, Mapping) else None,
    }


def extract_measurable_runtime(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pull runtime facts that receipts actually state; leave the rest PENDING."""

    deepest = None
    full_43 = None
    greedy = None
    present = [b for b in bindings if b.get("present")]
    for b in present:
        honesty = b.get("honesty") or {}
        if honesty.get("deepest_full_layer") is not None:
            deepest = honesty.get("deepest_full_layer")
        if honesty.get("full_43_layer_bos_body") is not None:
            full_43 = honesty.get("full_43_layer_bos_body")
        if honesty.get("greedy_token_produced") is not None:
            greedy = honesty.get("greedy_token_produced")

    return {
        "forward_family": "gravity_deepseek_v4_multi_layer_gpu_forward_bos",
        "num_hidden_layers_pinned": DEEPSEEK_V4_FLASH["num_hidden_layers"],
        "hidden_size_pinned": DEEPSEEK_V4_FLASH["hidden_size"],
        "deepest_full_layer_observed": deepest,
        "full_43_layer_bos_body": full_43,
        "greedy_token_produced": greedy,
        "receipts_bound": len(present),
        "receipts_missing": sum(1 for b in bindings if not b.get("present")),
        # Not present in BOS receipts as suite numbers — mark pending.
        "tps_mean": {"status": "PENDING", "reason": "no sealed TPS suite on BASE yet"},
        "tps_p99": {"status": "PENDING", "reason": "no sealed p99 latency suite on BASE yet"},
        "ttft_ms": {"status": "PENDING", "reason": "no sealed TTFT measurement on BASE yet"},
        "uma_peak_bytes": {
            "status": "PENDING",
            "reason": "no sealed UMA peak accounting bound into this freeze",
        },
    }


def extract_routing_baseline(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Routing geometry is pinned; live load-balance stats need forward hooks."""

    return {
        "n_routed_experts": DEEPSEEK_V4_FLASH["n_routed_experts"],
        "num_experts_per_tok": DEEPSEEK_V4_FLASH["num_experts_per_tok"],
        "n_shared_experts": DEEPSEEK_V4_FLASH["n_shared_experts"],
        "native_routing_preserved": True,
        "glm_router_weights_copied": False,
        "load_balance_stats": {
            "status": "PENDING",
            "reason": "route histogram capture not bound into baseline freeze yet",
        },
        "route_stability_probe": {
            "status": "PENDING",
            "reason": "requires repeated forward on frozen prompts",
        },
        "receipts_consulted": [b.get("path") for b in bindings if b.get("present")],
    }


def extract_hcli_baseline() -> dict[str, Any]:
    return {
        "transplant_points": list(TRANSPLANT_POINT_NAMES),
        "hcli_tool_action_decision_point": "hcli_tool_action_decision",
        "hcli_protocol_scores": {
            "status": "PENDING",
            "reason": "no sealed HCLI capability suite scores for BASE yet",
        },
        "chat_template_admission": {
            "status": "PENDING",
            "reason": "bind DSV4F tokenizer/template admission receipt when freezing live",
        },
    }


def extract_capability_baseline() -> dict[str, Any]:
    """Capability floors — all PENDING until benchmark corpus + forward scores exist."""

    domains = (
        "math_raw",
        "math_long_horizon",
        "coding_and_repository_work",
        "tool_use",
        "agentic_planning",
        "long_context_reasoning",
        "repair_and_critique",
        "hcli_protocols",
        "general_knowledge_conversation",
    )
    return {
        domain: {
            "status": "PENDING",
            "score": None,
            "reason": "REQUIRES_BENCHMARK_CORPUS + student forward measurement",
        }
        for domain in domains
    }


def freeze_base_dsv4f_baseline(
    *,
    receipts_dir: Path | None = None,
    receipt_names: Sequence[str] | None = None,
    out_path: Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Build the sealed BASE_DSV4F baseline freeze descriptor."""

    rdir = Path(receipts_dir) if receipts_dir is not None else RECEIPTS_DIR
    names = list(receipt_names) if receipt_names is not None else list(DEFAULT_FORWARD_RECEIPTS)
    bindings = [_bind_receipt(rdir / name) for name in names]

    document = {
        "schema": BASELINE_SCHEMA,
        "name": "BASE_DSV4F",
        "arm": "A_BASE_DSV4F",
        "recorded_at": _utc_now(),
        "role_of_linear_mapping": LINEAR_SUBSPACE_INITIALIZATION,
        "student_body": {
            "family": DEEPSEEK_V4_FLASH["family"],
            "repository": DEEPSEEK_V4_FLASH["repository"],
            "revision": DEEPSEEK_V4_FLASH["revision"],
            "model_type": DEEPSEEK_V4_FLASH["model_type"],
            "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
            "num_hidden_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
            "n_routed_experts": DEEPSEEK_V4_FLASH["n_routed_experts"],
            "num_experts_per_tok": DEEPSEEK_V4_FLASH["num_experts_per_tok"],
            "n_shared_experts": DEEPSEEK_V4_FLASH["n_shared_experts"],
            "vocab_size": DEEPSEEK_V4_FLASH["vocab_size"],
            "moe_intermediate_size": DEEPSEEK_V4_FLASH["moe_intermediate_size"],
            "source_torch_dtype": DEEPSEEK_V4_FLASH["source_torch_dtype"],
        },
        "forward_receipts": list(bindings),
        "runtime": extract_measurable_runtime(bindings),
        "routing": extract_routing_baseline(bindings),
        "hcli": extract_hcli_baseline(),
        "capability": extract_capability_baseline(),
        "claim_boundary": {
            "fabricated_scores": False,
            "pending_fields_honest": True,
            "full_capability_bench": False,
            "linear_init_is_not_base_delta": True,
            "note": (
                "This freeze records geometry + what BOS forward receipts state. "
                "Capability and TPS numbers are PENDING until a frozen suite runs."
            ),
        },
    }
    sealed = seal(document)
    if write:
        path = Path(out_path) if out_path is not None else DEFAULT_OUT
        _atomic_write_json(path, sealed)
        sealed = dict(sealed)
        sealed["_written_path"] = str(path)
    return sealed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Freeze BASE_DSV4F baseline descriptor (measurable fields only)."
    )
    p.add_argument("--receipts-dir", type=Path, default=RECEIPTS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--write", action="store_true", help="Write sealed JSON to --out")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    doc = freeze_base_dsv4f_baseline(
        receipts_dir=args.receipts_dir,
        out_path=args.out,
        write=bool(args.write),
    )
    # _written_path is a non-sealed runtime annotation; strip before verify.
    check = {k: v for k, v in doc.items() if not str(k).startswith("_")}
    verify(check, label="baseline freeze")
    print(
        json.dumps(
            {
                "status": "OK",
                "schema": doc["schema"],
                "seal_sha256": doc["seal_sha256"],
                "receipts_bound": doc["runtime"]["receipts_bound"],
                "written": doc.get("_written_path"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
