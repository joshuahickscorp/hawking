"""Run fresh KIMI transcripts through the resident HCLI Engine boundary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "tools" / "future"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from hcli.engine import Engine
from hcli.workspace import Workspace
from kimi_hcli_live_eval import (
    CASES,
    _prompt_for,
    effective_refusal_count,
    score_reply,
)


class KimiEngineProvider:
    """Minimal legacy-complete adapter for the normal Engine path.

    KIMI's direct HCLI battery is a natural-language answer model, while the
    resident Engine accepts a normalized ``complete`` result.  Keeping that
    normalization at the provider seam lets the actual Engine execute path
    (including its boundary contract) remain the system under test.
    """

    def __init__(self, model: Any, tok: Any, device: str, max_new_tokens: int):
        self.model = model
        self.tok = tok
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.last_text = ""
        self.calls = 0

    @staticmethod
    def _case_prompt(prompt: str) -> str:
        text = str(prompt or "").strip()
        marker = "\nTOOL ACCESS IS CLOSED FOR THIS ROUND."
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
        return text

    def complete(self, prompt: str, evidence=None, compiled=None):
        import torch

        # The Engine's legacy provider seam receives the stable case prompt
        # plus its compact tool-closure instruction.  Do not feed the whole
        # agentic schema packet back to this natural-language KIMI probe.
        case_prompt = self._case_prompt(prompt)
        messages = [{"role": "user", "content": case_prompt}]
        rendered = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tok(rendered, return_tensors="pt").to(self.device)
        limit = max(1, min(self.max_new_tokens, 256))
        with torch.no_grad():
            generated = self.model.generate(
                **ids, max_new_tokens=limit, do_sample=False
            )
        reply = self.tok.decode(
            generated[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        )
        self.last_text = reply
        self.calls += 1
        return {"kind": "answer", "content": reply}


def run(
    spec: Path,
    *,
    device: str,
    dtype: str,
    max_new_tokens: int,
    workspace_root: Path,
) -> dict[str, Any]:
    import dsv3_native_loader as loader
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(spec), trust_remote_code=True, use_fast=False
    )
    model, _cfg = loader.load(
        spec, device=device, dtype=dtype, min_free_gib=12.0
    )
    provider = KimiEngineProvider(model, tok, device, max_new_tokens)
    engine = Engine(
        Workspace(str(workspace_root)),
        model_client=provider,
        model_name="KIMI_BASE",
    )
    rows = []
    old_no_tools = os.environ.get("HCLI_NO_TOOLS")
    os.environ["HCLI_NO_TOOLS"] = "1"
    try:
        for case in CASES:
            prompt = _prompt_for(case)
            result = engine.execute(prompt, evidence=[], compiled={})
            raw_reply = provider.last_text
            engine_reply = str(result.get("content") or "")
            raw_score = score_reply(case, raw_reply)
            engine_score = score_reply(case, engine_reply)
            rows.append({
                "case_id": case.case_id,
                "response_mode": case.response_mode,
                "raw_reply_snippet": raw_reply[:500],
                "engine_reply_snippet": engine_reply[:500],
                "raw_score": raw_score,
                "engine_score": engine_score,
                "boundary_contract": result.get("boundary_contract"),
                "engine_kind": result.get("kind"),
                "engine_status": result.get("status"),
                "engine_error": result.get("error"),
            })
    finally:
        if old_no_tools is None:
            os.environ.pop("HCLI_NO_TOOLS", None)
        else:
            os.environ["HCLI_NO_TOOLS"] = old_no_tools

    raw_rows = [
        {"case_id": row["case_id"], "score": row["raw_score"]}
        for row in rows
    ]
    engine_rows = [
        {"case_id": row["case_id"], "score": row["engine_score"]}
        for row in rows
    ]
    return {
        "schema": "hawking.kimi.hcli_engine_live.v1",
        "status": "ENGINE_LIVE_SCORED",
        "source_model": "KIMI_BASE",
        "specimen": str(spec),
        "device": device,
        "dtype": dtype,
        "engine_path": "hcli.Engine.execute -> provider.complete (in-process)",
        "tools_disabled_for_boundary_battery": True,
        "provider_calls": provider.calls,
        "summary": {
            "cases": len(rows),
            "raw_passed": sum(bool(row["raw_score"].get("passed")) for row in rows),
            "engine_passed": sum(bool(row["engine_score"].get("passed")) for row in rows),
            "raw_effective_refusals": effective_refusal_count(raw_rows),
            "engine_effective_refusals": effective_refusal_count(engine_rows),
            "boundary_contract_repairs": sum(
                bool(row.get("boundary_contract")) for row in rows
            ),
            "all_engine_calls_completed": all(
                row.get("engine_status") == "completed" for row in rows
            ),
        },
        "rows": rows,
        "authority": {
            "weights_written": False,
            "external_writes": False,
            "tool_calls_executed": False,
            "daemon_contacted": False,
        },
        "claim_boundary": (
            "Fresh KIMI_BASE output was passed through the resident HCLI Engine; "
            "the boundary contract may append a lawful alternative only after the "
            "model withholds the unsafe operation. This is deployment-path evidence, "
            "not candidate promotion or authorization expansion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_HCLI_ENGINE_LIVE_20260910.json",
    )
    args = parser.parse_args()
    result = run(
        args.spec,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        workspace_root=args.workspace_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
