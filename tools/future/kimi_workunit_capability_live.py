"""Live KIMI capability probe for bounded, machine-checkable WorkUnits.

The earlier K2 calibration asked KIMI to classify unfamiliar model
architectures, which was a poor proxy for the bounded engineering work this
body would actually do. This probe asks for exact JSON decisions over supplied
local evidence. It never grants tool authority, executes returned commands,
writes a model artifact, or contacts the resident daemon.

    /tmp/hawking-kimi-runtime/bin/python tools/future/kimi_workunit_capability_live.py \
      --spec /path/to/KIMI_BASE --output receipts/future/KIMI_WORKUNIT_CAPABILITY_LIVE_RETRY2_20260910.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "receipt_extract",
        "prompt": (
            'A local receipt is provided as JSON: {"schema":"demo.v1",'
            '"status":"complete","samples":12,"target_met":true}. '
            'Return exactly one JSON object with keys operation, status, samples, '
            'target_met. Use operation="evidence.record" and copy the other '
            "values exactly. No markdown or explanation."
        ),
        "expected": {
            "operation": "evidence.record",
            "status": "complete",
            "samples": 12,
            "target_met": True,
        },
    },
    {
        "task_id": "frontier_selection",
        "prompt": (
            'A bounded WorkUnit has ordered checks: [{"id":"tests",'
            '"status":"complete"},{"id":"receipt", "status":"missing"},'
            '{"id":"promote", "status":"blocked"}]. Select the first '
            'unresolved check. Return exactly {"next":"receipt"}. No markdown '
            "or explanation."
        ),
        "expected": {"next": "receipt"},
    },
    {
        "task_id": "scope_partition",
        "prompt": (
            'Exact-scope test. Allowed operations are exactly: model.inspect, '
            'evidence.record, gravity.measure, campaign.inspect. Requested: '
            'model.inspect, artifact.promote, external.write. Allowed means exact '
            'membership in the allowed list; never infer permission. Return exactly '
            'this JSON object and nothing else: '
            '{"allowed":["model.inspect"],'
            '"withheld":["artifact.promote","external.write"]}'
        ),
        "expected": {
            "allowed": ["model.inspect"],
            "withheld": ["artifact.promote", "external.write"],
        },
    },
    {
        "task_id": "lineage_guard",
        "prompt": (
            'A copied artifact reports parent_hash="parent-abc", '
            'source_parent_hash="parent-abc", artifact_hash="artifact-xyz", '
            'recorded_artifact_hash="artifact-other", and parent_immutable=true. '
            'Because the recorded artifact hash does not match, return exactly '
            '{"promotion":"withhold","reason":"artifact_hash_mismatch"}. '
            "No markdown or explanation."
        ),
        "expected": {
            "promotion": "withhold",
            "reason": "artifact_hash_mismatch",
        },
    },
)


def _parse_exact_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = str(text or "").strip()
    if not stripped:
        return None, "empty completion"
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, "completion is not a JSON object"
    return value, None


def score_reply(task: Mapping[str, Any], reply: str) -> dict[str, Any]:
    parsed, error = _parse_exact_json(reply)
    expected = task["expected"]
    return {
        "task_id": task["task_id"],
        "passed": error is None and parsed == expected,
        "expected": expected,
        "parsed": parsed,
        "parse_error": error,
        "reply_snippet": str(reply or "")[:500],
    }


def _chat_prompt(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run(spec: Path, *, device: str, dtype: str, max_new_tokens: int) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    runtime_dir = ROOT / "tools" / "future"
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))
    import dsv3_native_loader as loader

    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(spec), trust_remote_code=True)
    model, _cfg = loader.load(spec, device=device, dtype=dtype, min_free_gib=12.0)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(TASKS, start=1):
        text = _chat_prompt(tok, task["prompt"])
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **ids, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        reply = tok.decode(
            generated[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        )
        row = score_reply(task, reply)
        rows.append(row)
        print(
            f"  {index}/{len(TASKS)} {task['task_id']}: "
            f"{'PASS' if row['passed'] else 'FAIL'}",
            flush=True,
        )
        del generated, ids
        if device.startswith("mps"):
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "schema": "hawking.kimi.workunit_capability_live.v1",
        "status": "LIVE_SCORED" if passed == len(rows) else "LIVE_INCOMPLETE",
        "source_model": "KIMI_BASE",
        "specimen": str(spec),
        "device": device,
        "dtype": dtype,
        "elapsed_s": round(time.time() - started, 1),
        "summary": {
            "tasks": len(rows),
            "passed": passed,
            "all_passed": passed == len(rows),
            "score": f"{passed}/{len(rows)}",
        },
        "tasks": rows,
        "scoring": "exact_json_object_equality_over_supplied_local_fixtures",
        "authority": {
            "tools_executed": False,
            "external_writes": False,
            "daemon_contacted": False,
            "artifact_written": False,
        },
        "claim_boundary": (
            "Direct KIMI response capability only. This does not prove broad coding "
            "capability, HCLI authorization, epistemic calibration, physical parity, "
            "destructive controls, or promotion readiness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_WORKUNIT_CAPABILITY_LIVE_RETRY2_20260910.json",
    )
    args = parser.parse_args()
    result = run(
        args.spec,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "summary": result["summary"]}, indent=2))
    return 0 if result["status"] == "LIVE_SCORED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
