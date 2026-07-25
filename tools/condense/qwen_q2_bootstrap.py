#!/usr/bin/env python3.12
"""Run the first real Qwen Gravity checkpoint directly from immutable HTTP ranges.

Q2 is intentionally bounded: one real token, layer 0, the real top-8 router decision, all selected
real experts, then a 120B-rate Gravity pack of the highest-weight selected expert. It proves the
source/forward/packer seam and starts the target-matched frontier without downloading 438 GiB.
It is not a full-model capability result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import qwen_bpw_budget as B
import qwen_correction_wave as C
import qwen_real_forward as Q

ROOT = Path(__file__).resolve().parents[2]
GF = ROOT / "reports/condense/general_frontier"
DEFAULT_RECEIPT = GF / "QWEN3_235B_Q2_RECEIPT.json"
DEFAULT_STATE = GF / "QWEN3_235B_GRAVITY_STATE.json"
DEFAULT_PLAN = GF / "QWEN3_235B_GRAVITY_BPW_PLAN.json"
DEFAULT_LEASE = GF / "QWEN_Q2/leases/qwen_q2.lease"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _seal(obj: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    out["sha256"] = hashlib.sha256(
        json.dumps(out, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float((aa @ bb) / max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1e-30))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) /
                 max(float(np.linalg.norm(a.astype(np.float64))), 1e-30))


def _phase(state_path: Path, phase: str, **extra: Any) -> None:
    _atomic(state_path, {
        "schema": "hawking.qwen3_235b.gravity_state.v1", "status": "RUNNING",
        "phase": phase, "updated_at": _now(), "final": False,
        "target_whole_artifact_bpw_ceiling": B.TARGET_WHOLE_BPW, **extra,
    })
    print(f"[qwen-q2] {phase}", flush=True)


def run(*, meta_dir: Path, cache_dir: Path, receipt_path: Path, state_path: Path,
        plan_path: Path, lease_path: Path, prompt: str) -> int:
    if lease_path.exists():
        try:
            held = json.loads(lease_path.read_text())
            pid = int(held.get("pid", -1))
            if pid > 0:
                os.kill(pid, 0)
                raise RuntimeError(f"Q2 lease already held by live pid {pid}")
        except ProcessLookupError:
            pass
        except (ValueError, json.JSONDecodeError):
            pass
    _atomic(lease_path, {"owner": "com.hawking.qwen_q2", "pid": os.getpid(),
                         "started_at": _now(), "revision": Q.DEFAULT_REVISION})
    t0 = time.time()
    try:
        config = json.loads((meta_dir / "config.json").read_text())
        index = json.loads((meta_dir / "model.safetensors.index.json").read_text())
        plan = B.build_plan(config, index)
        _atomic(plan_path, plan)
        if not plan["accounting"]["target_met"]:
            raise RuntimeError("whole-model byte plan exceeds the 120B target")

        from tokenizers import Tokenizer  # type: ignore
        tokenizer = Tokenizer.from_file(str(meta_dir / "tokenizer.json"))
        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            raise RuntimeError("tokenizer produced no prompt tokens")
        token_id = int(prompt_ids[-1])
        ids = [token_id]
        _phase(state_path, "REMOTE_LAYER0_PARENT", prompt=prompt, token_id=token_id,
               plan_sha256=plan["sha256"])

        fwd = Q.from_remote(meta_dir, cache_dir=cache_dir)
        # This executes attention + router + all selected experts for a complete real layer-0 step.
        hidden = fwd.logits_for(ids, positions="all", max_blocks=1, progress=True)
        if hidden.shape != (1, fwd.g.hidden) or not np.isfinite(hidden).all():
            raise RuntimeError(f"non-finite or wrong-shape layer output: {hidden.shape}")

        # Recover the exact post-attention MoE input and router decision. Repeated ranges hit the
        # local content-addressed cache, so this does not redownload the attention matrices.
        x0 = fwd.reader.bf16_rows("model.embed_tokens.weight", ids)
        x_attn = x0 + fwd._attention(0, x0)
        h = Q.rmsnorm(x_attn, fwd.reader.bf16("model.layers.0.post_attention_layernorm.weight"),
                      fwd.g.eps)
        router = fwd.reader.bf16("model.layers.0.mlp.gate.weight")
        router_logits = h[0] @ router.T
        selected, routing_weights = Q.route_topk(
            router_logits, fwd.g.top_k, fwd.g.norm_topk_prob
        )
        if len(selected) != 8 or not np.isfinite(routing_weights).all():
            raise RuntimeError("real Qwen router did not produce a finite top-8 decision")

        top_expert = int(selected[0])
        source_expert = fwd._get_expert(0, top_expert, None)
        audit: dict[str, Any] = {}
        hook = C._make_hook(C.CANDIDATES["T3_qwen_organ_alloc"], audit)
        packed_expert = hook(0, top_expert, source_expert)
        budget = C._budget_from_audit(audit)
        if budget is None:
            raise RuntimeError("Gravity pack did not produce a complete three-projection ledger")

        xp = h[0]
        a0 = Q.swiglu(source_expert["gate"] @ xp, source_expert["up"] @ xp)
        a1 = Q.swiglu(packed_expert["gate"] @ xp, packed_expert["up"] @ xp)
        y0 = source_expert["down"] @ a0
        y1 = packed_expert["down"] @ a1
        finite = bool(np.isfinite(y0).all() and np.isfinite(y1).all())
        target_met = bool(budget["per_expert_projection_bpw"] <= B.TARGET_WHOLE_BPW)
        if not finite or not target_met:
            raise RuntimeError("Q2 candidate failed finiteness or the 120B numeric BPW ceiling")

        reader_telem = fwd.reader.telemetry_json()
        receipt = _seal({
            "schema": "hawking.qwen3_235b.q2_router_expert_layer.v1",
            "generated_at": _now(), "gate": "Q2_real_router_expert_layer_and_gravity_ignition",
            "repo": Q.DEFAULT_REPO, "revision": Q.DEFAULT_REVISION,
            "source_mode": "immutable revision-pinned HTTP byte ranges with content-addressed cache",
            "prompt": prompt, "prompt_token_count": len(prompt_ids), "bounded_token_id": token_id,
            "real_layer0": {
                "complete_top8_forward": True, "hidden_shape": list(hidden.shape),
                "hidden_rms": round(float(np.sqrt(np.mean(hidden.astype(np.float64) ** 2))), 8),
                "finite": True, "selected_experts": [int(v) for v in selected],
                "routing_weights": [round(float(v), 8) for v in routing_weights],
                "routing_weight_sum": round(float(routing_weights.sum()), 8),
            },
            "first_gravity_candidate": {
                "candidate": "T3_qwen_organ_alloc", "real_layer": 0,
                "real_selected_expert": top_expert,
                "mapping": {c: C.CANDIDATES["T3_qwen_organ_alloc"][c] for c in C.CLASSES},
                "budget": budget, "target_numeric_bpw_ceiling": B.TARGET_WHOLE_BPW,
                "target_numeric_bpw_met": target_met,
                "real_activation_output_relative_error": round(_rel(y0, y1), 8),
                "real_activation_output_cosine": round(_cosine(y0, y1), 8),
                "finite": finite,
                "quality_scope": "single real expert activation diagnostic; not a capability pass",
            },
            "whole_model_byte_plan": {
                "path": str(plan_path), "sha256": plan["sha256"],
                "projected_whole_artifact_bpw": plan["accounting"]["projected_whole_artifact_bpw"],
                "target_met": plan["accounting"]["target_met"],
                "physical_bytes": plan["accounting"]["total_artifact_bytes"],
            },
            "transport": reader_telem,
            "elapsed_seconds": round(time.time() - t0, 1),
            "verdict": "Q2 PASS; real Qwen layer/router/expert path works and the first asymmetric "
                       "Gravity candidate is below the 120B numeric BPW ceiling. Full-model "
                       "parent-vs-packed capability evaluation remains required.",
        })
        _atomic(receipt_path, receipt)
        _atomic(state_path, {
            "schema": "hawking.qwen3_235b.gravity_state.v1",
            "status": "Q2_SEALED_GRAVITY_STARTED", "phase": "FULL_MODEL_FRONTIER_PENDING",
            "updated_at": _now(), "final": False, "q2_receipt": str(receipt_path),
            "q2_sha256": receipt["sha256"], "whole_model_bpw_plan": str(plan_path),
            "projected_whole_artifact_bpw": plan["accounting"]["projected_whole_artifact_bpw"],
            "target_whole_artifact_bpw_ceiling": B.TARGET_WHOLE_BPW,
            "next": "run full parent-vs-packed Qwen forward ladder; do not promote on Q2 diagnostic",
        })
        print(json.dumps({"status": "Q2_SEALED_GRAVITY_STARTED", "receipt": str(receipt_path),
                          "expert_projection_bpw": budget["per_expert_projection_bpw"],
                          "projected_whole_bpw": plan["accounting"]["projected_whole_artifact_bpw"],
                          "activation_relative_error": round(_rel(y0, y1), 8)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        _atomic(state_path, {
            "schema": "hawking.qwen3_235b.gravity_state.v1", "status": "Q2_FAILED_RETRYABLE",
            "phase": "Q2", "updated_at": _now(), "final": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--lease", type=Path, default=DEFAULT_LEASE)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args(argv)
    return run(meta_dir=args.meta_dir, cache_dir=args.cache_dir, receipt_path=args.receipt,
               state_path=args.state, plan_path=args.plan, lease_path=args.lease,
               prompt=args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
