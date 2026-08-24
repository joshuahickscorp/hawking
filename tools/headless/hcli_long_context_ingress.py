#!/usr/bin/env python3
"""Physically prove a >24K-token root request through the canonical HCLI path.

Not a curl. This spawns HCLI's own RuntimePool, lets the canonical context
authority size the allocation from the measured demand, assembles the prompt
through ``Engine._call_model`` exactly as a real mission would, and records the
runtime identity alongside the token counts.

The regression it closes: the same path previously died with
``llama-server HTTP 400: request (23532 tokens) exceeds the available context
size (11008 tokens)`` because the pool asked for ``--ctx-size 32768 --parallel 3``
and nothing in HCLI knew llama.cpp divides that by the slot count.

    python3 tools/headless/hcli_long_context_ingress.py
    python3 tools/headless/hcli_long_context_ingress.py --target-tokens 26000

Exit 0 only on a real completion whose server-counted prompt exceeds the target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_prompt(target_tokens: int, chars_per_token: float = 3.7) -> str:
    """A harmless, real prompt large enough to exceed the target.

    Built from repository text rather than filler so the model has something
    coherent to answer, which is what makes the completion meaningful rather
    than merely long.
    """
    sources = [
        REPO_ROOT / "tools/haider/hcli/mission.py",
        REPO_ROOT / "tools/haider/hcli/scheduler.py",
        REPO_ROOT / "tools/haider/hcli/context_budget.py",
        REPO_ROOT / "tools/haider/hcli/report_compiler.py",
        REPO_ROOT / "tools/haider/hcli/workunit.py",
        REPO_ROOT / "tools/haider/hcli/engine.py",
        REPO_ROOT / "tools/haider/hcli/ledger.py",
        REPO_ROOT / "tools/haider/hcli/grok_bridge.py",
        REPO_ROOT / "tools/haider/hcli/controller.py",
    ]
    parts: List[str] = [
        "You are reading part of a Python control plane. "
        "Answer the question at the end in one short sentence.\n"
    ]
    want = int(target_tokens * chars_per_token)
    total = 0
    for path in sources:
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        parts.append(f"===== {path.relative_to(REPO_ROOT)} =====\n{body}\n")
        total += len(body)
        if total >= want:
            break
    parts.append(
        "\nQUESTION: name the single module above that defines the mutation "
        "lock's exclusive acquire. Answer with the file path only."
    )
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=int, default=24000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="receipts/headless/HCLI_LONG_CONTEXT_ROOT_INGRESS.json")
    args = ap.parse_args(argv)

    # One sequence for a root ingress: the root prompt must not be divided by a
    # worker slot count. This is the policy the authority encodes; setting it
    # explicitly here makes the run reproducible.
    os.environ.setdefault("HCLI_RESIDENT_RUNTIME_LIMIT", "1")

    from hcli import context_budget as cb
    from hcli.engine import Engine
    from hcli.events import EventBus
    from hcli.models import ModelRegistry
    from hcli.runtime import RuntimePool
    from hcli.workspace import Workspace

    registry = ModelRegistry()
    # ModelRegistry exposes discover()/models, not list().
    found = registry.discover()
    if not found:
        print("no model discovered; cannot spawn a runtime", file=sys.stderr)
        return 2
    first = found[0]
    model_path = getattr(first, "path", None) or (
        first.get("path") if isinstance(first, dict) else first
    )

    prompt = build_prompt(args.target_tokens)

    # Size the pool against the SAME quantity the preflight will check. A raw
    # len(prompt)/3.7 estimate under-counts, because what actually goes on the
    # wire is the system prompt plus the GOAL: wrapper plus the evidence
    # section -- and sizing the runtime from the smaller number is how a root
    # ingress ends up 1173 tokens short of a ceiling it chose for itself.
    os.environ["HCLI_MODEL_TOKENS"] = str(args.max_tokens)
    probe = Engine(workspace=Workspace(str(REPO_ROOT)), model_client=None)
    messages = [
        {"role": "system", "content": getattr(__import__("hcli.engine", fromlist=["_SYSTEM_PROMPT"]), "_SYSTEM_PROMPT", "")},
        {"role": "user", "content": f"GOAL:\n{prompt}\n\nDETERMINISTIC EVIDENCE:\n(none)"},
    ]
    estimate = probe._estimate_prompt_tokens(messages)

    budget = cb.resolve(
        model_path=str(model_path),
        n_parallel=1,
        demand_tokens=estimate,
        repo_root=str(REPO_ROOT),
    )
    print(
        f"budget: total={budget.total_ctx} per_request={budget.per_request_ctx} "
        f"usable={budget.usable_input_tokens} source={budget.source} "
        f"ceiling={budget.model_ceiling}",
        flush=True,
    )
    os.environ["HCLI_CTX_SIZE"] = str(budget.total_ctx)

    pool = RuntimePool(str(model_path), requested_n=1, workspace=str(REPO_ROOT))
    started = time.perf_counter()
    pool.start()
    spawn_s = time.perf_counter() - started
    runtimes = [r for r in pool.runtimes if getattr(r, "active", False)]
    if not runtimes:
        print(f"pool admitted nothing: {pool.refusal_reason}", file=sys.stderr)
        pool.stop()
        return 2
    print(f"pool up in {spawn_s:.1f}s, {len(runtimes)} runtime(s)", flush=True)

    receipt: Dict[str, Any] = {
        "gate": "HCLI_LONG_CONTEXT_ROOT_INGRESS",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": "canonical: RuntimePool -> Engine._call_model -> llama-server",
        "budget": {
            "total_ctx": budget.total_ctx,
            "n_parallel": budget.n_parallel,
            "per_request_ctx": budget.per_request_ctx,
            "usable_input_tokens": budget.usable_input_tokens,
            "source": budget.source,
            "model_ceiling": budget.model_ceiling,
        },
        "prompt_chars": len(prompt),
        "prompt_tokens_estimated": estimate,
        "estimator": "Engine._estimate_prompt_tokens over the assembled messages",
        "spawn_s": round(spawn_s, 2),
    }

    try:
        engine = Engine(
            workspace=Workspace(str(REPO_ROOT)),
            event_bus=EventBus(),
            runtime_provider=lambda: pool,
            runtime_state_provider=lambda: pool,
            runtime_count=1,
            model_name=str(model_path),
        )
        t0 = time.perf_counter()
        raw = engine._call_model(prompt, [], {})
        wall = time.perf_counter() - t0
        calls = getattr(engine, "_model_calls", [])
        receipt["model_calls"] = calls
        receipt["wall_s"] = round(wall, 2)
        receipt["answer_excerpt"] = json.dumps(raw)[:400]
        served = calls[0].get("prompt_tokens") if calls else None
        receipt["prompt_tokens_server_counted"] = served
        receipt["runtime_identity"] = {
            "model_path": str(model_path),
            "endpoint": f"http://127.0.0.1:{runtimes[0].port}",
            "pid": runtimes[0].pid,
            "requested_n": 1,
            "ctx_size_arg": budget.total_ctx,
        }
        ok = bool(served) and int(served) >= args.target_tokens
        receipt["result"] = "PASS" if ok else "FAIL"
        if not ok:
            receipt["reason"] = (
                f"server counted {served} prompt tokens, target was {args.target_tokens}"
            )
    except Exception as exc:
        receipt["result"] = "FAIL"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    finally:
        pool.stop()

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt.get(k) for k in
                      ("result", "prompt_tokens_server_counted", "wall_s", "error")}, indent=1))
    print(f"receipt: {out}")
    return 0 if receipt.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
