#!/usr/bin/env python3.12
"""frontier_parity.py - architecture parity plan for every frontier target.

This is a readiness ledger, not a substitute for real parity. It prevents an easy failure mode:
loading a giant checkpoint and then making claims with the wrong router, attention geometry, chat
template, or reference backend.

Usage:
  frontier_parity.py plan [--out reports/condense/frontier_parity_plan.json]
  frontier_parity.py status
  frontier_parity.py selftest

Real parity evidence, when generated later on the Studio, should be recorded as:
  reports/condense/<LABEL>_parity.json

Expected fields in a real parity record:
  model, hf_id, status in {pass, fail}, reference_backend, tokenizer_sha256,
  config_sha256, prompt_count, max_logit_abs_err, greedy_match_tokens, exact commands,
  reference/native trace hashes, architecture adapter receipt, tensor-map hash,
  tokenizer/context contracts, parity test receipts, unsupported-by-design exits, notes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import FRONTIER_MODELS, FrontierModel  # noqa: E402

OUT = pathlib.Path("reports/condense")
PLAN_PATH = OUT / "frontier_parity_plan.json"


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label)


def _sha256(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parity_record_path(model: FrontierModel, root: pathlib.Path = ROOT) -> pathlib.Path:
    return root / OUT / f"{model.label}_parity.json"


def _load_json(path: pathlib.Path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _family_spec(model: FrontierModel) -> dict:
    hf = model.hf_id.lower()
    label = model.label.lower()
    if "deepseek-v4" in hf:
        return {
            "family": "deepseek-v4",
            "reference_backends": ["Transformers trust_remote_code", "vLLM", "SGLang"],
            "required_native_features": [
                "DeepSeek V4 hybrid CSA/HCA attention",
                "DSpark speculative module ignored or handled explicitly",
                "official encoding parser/chat format",
                "MoE router and reasoning-mode tags",
                "1M-context RoPE/YARN settings",
            ],
            "risk": "critical",
        }
    if "deepseek-v3" in hf or label == "671b":
        return {
            "family": "deepseek-v3",
            "reference_backends": ["Transformers trust_remote_code", "vLLM", "SGLang"],
            "required_native_features": [
                "MLA attention geometry",
                "DeepSeek MoE router/top-k",
                "YARN/mscale rope",
                "chat template parity",
            ],
            "risk": "critical",
        }
    if "glm-5" in hf:
        return {
            "family": "glm-5",
            "reference_backends": ["Transformers trust_remote_code", "vLLM"],
            "required_native_features": [
                "GLM MoE router",
                "1M-context rope/scaling",
                "GLM chat/template tokens",
                "custom config fields",
            ],
            "risk": "critical",
        }
    if "kimi-k2" in hf:
        return {
            "family": "kimi-k2",
            "reference_backends": ["Transformers trust_remote_code", "vLLM", "SGLang"],
            "required_native_features": [
                "Kimi K2 MoE router",
                "native INT4/compressed-tensors source handling",
                "custom code/template parity",
                "multimodal fields ignored safely for text-only runs",
            ],
            "risk": "critical",
        }
    if "qwen3" in hf:
        return {
            "family": "qwen3-moe",
            "reference_backends": ["Transformers", "vLLM"],
            "required_native_features": [
                "Qwen3 MoE router",
                "Qwen tokenizer/chat template",
                "expert activation count",
            ],
            "risk": "high",
        }
    if "llama-3.1-405b" in hf:
        return {
            "family": "llama3.1-dense",
            "reference_backends": ["Transformers", "llama.cpp GGUF where available"],
            "required_native_features": [
                "GQA geometry",
                "Llama 3.1 rope",
                "license/gated-access receipt",
                "chat template parity",
            ],
            "risk": "high",
        }
    return {
        "family": "unknown",
        "reference_backends": ["Transformers"],
        "required_native_features": ["manual architecture review"],
        "risk": "critical",
    }


def parity_status(model: FrontierModel, root: pathlib.Path = ROOT) -> dict:
    path = _parity_record_path(model, root)
    rec = _load_json(path)
    status = "missing"
    problems = []
    if rec:
        status = rec.get("status", "unknown")
        if rec.get("model") != model.label:
            problems.append("model label mismatch")
        if rec.get("hf_id") != model.hf_id:
            problems.append("hf_id mismatch")
        if status != "pass":
            problems.append(f"parity status is {status}")
        prompt_count = rec.get("prompt_count")
        if not isinstance(prompt_count, int) or prompt_count < 4:
            problems.append("too few parity prompts")
        if not isinstance(rec.get("max_logit_abs_err"), (int, float)):
            problems.append("missing logit error")
        greedy_match_tokens = rec.get("greedy_match_tokens")
        if not isinstance(greedy_match_tokens, int) or greedy_match_tokens < 16:
            problems.append("greedy match window too short")
    else:
        problems.append("no parity record")
    return {
        "record": str(path),
        "record_exists": bool(rec),
        "status": status,
        "problems": problems,
    }


def plan_row(model: FrontierModel, root: pathlib.Path = ROOT) -> dict:
    family = _family_spec(model)
    local = root / model.local_dir
    config = local / "config.json"
    tokenizer = local / "tokenizer.json"
    parity = parity_status(model, root)
    gates = [
        "Load config and tokenizer from the exact staged source revision.",
        "Select a project-native architecture adapter with an exact tensor-name map.",
        "Run a trusted reference backend on fixed prompts and capture logits/top-k.",
        "Run Hawking/native path on the same prompts after architecture implementation.",
        "Capture reference and Hawking/native trace files and hash them in the receipt.",
        "Require tokenizer/config hash match in the parity record.",
        "Require tokenizer/chat-template and context/KV/position contracts before signing.",
        "Require max logit error and greedy-token match thresholds before quality claims.",
        "Record unsupported-by-design exits for ignored custom-code paths before claims.",
    ]
    receipt_fields = [
        "architecture_adapter",
        "adapter_receipt",
        "tensor_map_receipt",
        "tensor_map_sha256",
        "reference_trace",
        "reference_trace_sha256",
        "hawking_trace or native_trace",
        "hawking_trace_sha256 or native_trace_sha256",
        "tokenizer_contract",
        "context_contract",
        "parity_tests",
        "unsupported_by_design",
    ]
    return {
        "label": model.label,
        "hf_id": model.hf_id,
        "family": family["family"],
        "risk": family["risk"],
        "reference_backends": family["reference_backends"],
        "required_native_features": family["required_native_features"],
        "local_dir": model.local_dir,
        "source_staged": local.is_dir(),
        "config_sha256": _sha256(config),
        "tokenizer_sha256": _sha256(tokenizer),
        "required_gates": gates,
        "required_receipt_fields": receipt_fields,
        "parity": parity,
        "claim_gate": "BLOCK" if parity["status"] != "pass" else "ALLOW",
    }


def build_plan(root: pathlib.Path = ROOT) -> dict:
    rows = [plan_row(model, root) for model in FRONTIER_MODELS]
    return {
        "schema": "hawking.frontier_parity_plan.v1",
        "root": str(root),
        "model_count": len(rows),
        "blocked_claims": sum(1 for r in rows if r["claim_gate"] == "BLOCK"),
        "rows": rows,
        "note": "A frontier model cannot support quality/tok/s claims until its parity row is ALLOW.",
    }


def _write(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def cmd_plan(args) -> int:
    data = build_plan(ROOT)
    path = pathlib.Path(args.out)
    _write(path, data)
    print(f"[frontier-parity] wrote {path}  blocked={data['blocked_claims']}/{data['model_count']}",
          file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    data = build_plan(ROOT)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print(f"# frontier parity status: blocked={data['blocked_claims']}/{data['model_count']}")
    print("label              family            risk      gate   record")
    for row in data["rows"]:
        print(f"{row['label'][:18]:18s} {row['family'][:17]:17s} {row['risk'][:8]:8s} "
              f"{row['claim_gate']:6s} {row['parity']['record']}")
    return 0


def cmd_selftest(args) -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    data = build_plan(ROOT)
    labels = {r["label"] for r in data["rows"]}
    families = {r["family"] for r in data["rows"]}
    check("one row per frontier model", data["model_count"] == len(FRONTIER_MODELS))
    check("DeepSeek V4 family recognized", "deepseek-v4" in families)
    check("Kimi family recognized", "kimi-k2" in families)
    check("GLM family recognized", "glm-5" in families)
    check("V4 Pro present", "DeepSeek-V4-Pro" in labels)
    check("missing parity blocks claims", data["blocked_claims"] >= 1)
    first = data["rows"][0]
    check("adapter/tensor-map fields are required",
          {"architecture_adapter", "tensor_map_sha256"}.issubset(set(first["required_receipt_fields"])))
    check("tokenizer/context contracts are required",
          {"tokenizer_contract", "context_contract"}.issubset(set(first["required_receipt_fields"])))
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Frontier architecture parity readiness ledger.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan")
    p.add_argument("--out", default=str(PLAN_PATH))
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.cmd:
        args = ap.parse_args(["status"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
