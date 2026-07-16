#!/usr/bin/env python3
"""OpenAI-compatible serving concurrency matrix for Dismantle/Hawking.

The harness is inert unless you provide `--base-url` or `--launch-cmd`.
Use `--plan-only` to materialize the request matrix without sending requests.

It intentionally depends only on the Python standard library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD_FILE = ROOT / "tools/bench/workloads.json"
DEFAULT_REPORT = ROOT / "docs/reports/serve_matrix/latest.md"


@dataclass
class RequestPlan:
    request_id: str
    concurrency: int
    prompt_token_target: int
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float
    stream: bool
    seed: int

    def body(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
            "seed": self.seed,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("empty list")
    return result


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_workload(value: str) -> tuple[dict[str, Any], str]:
    path = Path(value)
    if path.exists():
        doc = load_json(path)
        if "profiles" not in doc:
            return doc, str(path)
        name = path.stem
    else:
        name = path.stem
        doc = load_json(DEFAULT_WORKLOAD_FILE)
    profiles = doc.get("profiles", {})
    if name in profiles:
        return profiles[name], f"{DEFAULT_WORKLOAD_FILE}#{name}"
    raise SystemExit(f"unknown workload: {value}")


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def repeated_context(target_tokens: int, seed: int, blocks: list[str]) -> str:
    if not blocks:
        blocks = ["local Apple serving benchmark context"]
    out: list[str] = []
    i = 0
    while approx_tokens("\n".join(out)) < target_tokens:
        block = blocks[(seed + i) % len(blocks)]
        out.append(f"[context-{i:04d}] {block}")
        i += 1
    return "\n".join(out)


def build_messages(workload: dict[str, Any], request_id: int, target_tokens: int) -> list[dict[str, str]]:
    kind = workload.get("kind", "shared_agent")
    blocks = list(workload.get("shared_context") or [])
    template_list = list(workload.get("user_templates") or ["Turn {request_id}: continue."])
    template = template_list[request_id % len(template_list)]

    if kind == "mixed_latency" and request_id % 4 != 0:
        target_tokens = min(target_tokens, 1024)
    elif kind == "cache_miss_taxonomy" and request_id % 3 == 1:
        blocks = blocks + [f"near-prefix mutation {request_id}"]
    elif kind == "cache_miss_taxonomy" and request_id % 3 == 2:
        blocks = [f"divergent-prefix block {request_id}", *blocks]

    system_context = repeated_context(max(target_tokens - 256, 256), request_id, blocks)
    user = template.format(request_id=request_id, prompt_tokens=target_tokens)
    return [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user},
    ]


def build_plan(
    workload: dict[str, Any],
    concurrency_values: list[int],
    prompt_targets: list[int],
    decode_tokens: int,
    requests_per_level: int,
    stream: bool,
) -> list[RequestPlan]:
    temperature = float(workload.get("temperature", 0.0))
    plans: list[RequestPlan] = []
    counter = 0
    for concurrency in concurrency_values:
        for prompt_target in prompt_targets:
            for _ in range(max(requests_per_level, concurrency)):
                counter += 1
                plans.append(RequestPlan(
                    request_id=f"req-{counter:05d}",
                    concurrency=concurrency,
                    prompt_token_target=prompt_target,
                    messages=build_messages(workload, counter, prompt_target),
                    max_tokens=decode_tokens,
                    temperature=temperature,
                    stream=stream,
                    seed=42 + counter,
                ))
    return plans


def post_json(url: str, payload: dict[str, Any], timeout: float, stream: bool) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    bytes_read = 0
    text_parts: list[str] = []
    error: str | None = None
    status = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if stream:
                for raw in resp:
                    now = time.perf_counter()
                    bytes_read += len(raw)
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_text = line[5:].strip()
                    if payload_text == "[DONE]":
                        break
                    try:
                        event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue
                    delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        if first_token_at is None:
                            first_token_at = now
                        text_parts.append(str(delta))
            else:
                raw_body = resp.read()
                bytes_read = len(raw_body)
                body = json.loads(raw_body.decode("utf-8"))
                content = body.get("choices", [{}])[0].get("message", {}).get("content")
                if content is None:
                    content = body.get("choices", [{}])[0].get("text", "")
                text_parts.append(str(content or ""))
    except urllib.error.HTTPError as exc:
        error = f"http {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
    except Exception as exc:
        error = str(exc)

    end = time.perf_counter()
    text = "".join(text_parts)
    return {
        "status": status,
        "error": error,
        "ttft_ms": None if first_token_at is None else round((first_token_at - start) * 1000, 3),
        "wall_ms": round((end - start) * 1000, 3),
        "output_chars": len(text),
        "output_tokens_est": approx_tokens(text) if text else 0,
        "bytes_read": bytes_read,
    }


def wait_healthz(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/healthz"
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if getattr(resp, "status", 200) < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"server did not become healthy at {url}: {last_error}")


def wait_health_url(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if getattr(resp, "status", 200) < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"server did not become healthy at {url}: {last_error}")


def run_level(base_url: str, model: str, plans: list[RequestPlan], timeout: float) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    if not base_url.rstrip("/").endswith("/v1"):
        url = base_url.rstrip("/") + "/v1/chat/completions"
    concurrency = plans[0].concurrency if plans else 1
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(post_json, url, plan.body(model), timeout, plan.stream): plan
            for plan in plans
        }
        for future in concurrent.futures.as_completed(future_map):
            plan = future_map[future]
            result = future.result()
            result.update({
                "request_id": plan.request_id,
                "concurrency": plan.concurrency,
                "prompt_token_target": plan.prompt_token_target,
                "max_tokens": plan.max_tokens,
                "temperature": plan.temperature,
                "stream": plan.stream,
                "completed_at": now_iso(),
            })
            results.append(result)
    return results


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((int(row["concurrency"]), int(row["prompt_token_target"])), []).append(row)
    summaries = []
    for (concurrency, prompt_target), rows in sorted(grouped.items()):
        ok = [row for row in rows if not row.get("error")]
        ttft = [float(row["ttft_ms"]) for row in ok if row.get("ttft_ms") is not None]
        wall = [float(row["wall_ms"]) for row in ok if row.get("wall_ms") is not None]
        out_tokens = sum(float(row.get("output_tokens_est") or 0) for row in ok)
        wall_max = max(wall) if wall else None
        summaries.append({
            "concurrency": concurrency,
            "prompt_token_target": prompt_target,
            "requests": len(rows),
            "ok": len(ok),
            "errors": len(rows) - len(ok),
            "ttft_p50_ms": percentile(ttft, 0.50),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "ttft_p99_ms": percentile(ttft, 0.99),
            "wall_p50_ms": percentile(wall, 0.50),
            "wall_p95_ms": percentile(wall, 0.95),
            "aggregate_tokens_per_second_est": None if not wall_max or wall_max <= 0 else out_tokens / (wall_max / 1000.0),
            "per_user_tokens_per_second_est": None if not wall_max or wall_max <= 0 or concurrency <= 0 else out_tokens / (wall_max / 1000.0) / concurrency,
        })
    return summaries


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_report(meta: dict[str, Any], summaries: list[dict[str, Any]], plan_only: bool) -> str:
    lines = [
        "# Serving Concurrency Matrix",
        "",
        f"- generated_at: `{meta['generated_at']}`",
        f"- engine: `{meta['engine']}`",
        f"- workload: `{meta['workload_name']}`",
        f"- hardware: `{meta['hardware']}`",
        f"- mode: `{'plan-only' if plan_only else 'measured'}`",
        "",
        "| Concurrency | Prompt target | Requests | OK | Errors | TTFT P50 ms | TTFT P95 ms | TTFT P99 ms | Agg t/s est | User t/s est |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not summaries:
        lines.append("| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |")
    for row in summaries:
        lines.append(
            f"| {row['concurrency']} | {row['prompt_token_target']} | {row['requests']} | "
            f"{row['ok']} | {row['errors']} | {fmt(row['ttft_p50_ms'])} | "
            f"{fmt(row['ttft_p95_ms'])} | {fmt(row['ttft_p99_ms'])} | "
            f"{fmt(row['aggregate_tokens_per_second_est'])} | {fmt(row['per_user_tokens_per_second_est'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `*_est` token rates use output text length / 4 as a rough token estimate unless the engine exposes usage.",
        "- Streaming mode is required for true TTFT.",
        "- Record hardware tier explicitly; do not compare 16GB and 96GB Apple machines as one tier.",
        "",
    ])
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def launch_server(cmd: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="dismantle")
    parser.add_argument("--engine-config", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--launch-cmd", default=None)
    parser.add_argument("--health-timeout", type=float, default=120)
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--hardware", default="apple-local")
    parser.add_argument("--workload", default="shared_agent")
    parser.add_argument("--concurrency", type=parse_int_list)
    parser.add_argument("--prompt-tokens", type=parse_int_list)
    parser.add_argument("--decode-tokens", type=int)
    parser.add_argument("--requests-per-level", type=int, default=0, help="0 means max(concurrency)")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out", default="docs/reports/serve_matrix/latest.jsonl")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    engine_config = {}
    if args.engine_config:
        engine_config = load_json(Path(args.engine_config))
    base_url = args.base_url or engine_config.get("base_url")
    launch_cmd = args.launch_cmd or engine_config.get("launch_command")
    health_url = engine_config.get("health_url")
    model = args.model if args.model != "local-model" else engine_config.get("model", args.model)

    workload, workload_source = resolve_workload(args.workload)
    concurrency = args.concurrency or list(workload.get("concurrency") or [1, 2, 4, 8])
    prompt_tokens = args.prompt_tokens or list(workload.get("prompt_token_targets") or [8192])
    decode_tokens = args.decode_tokens or int(workload.get("decode_tokens", 128))
    requests_per_level = args.requests_per_level if args.requests_per_level > 0 else max(concurrency)
    plans = build_plan(
        workload,
        concurrency,
        prompt_tokens,
        decode_tokens,
        requests_per_level,
        args.stream,
    )
    meta = {
        "generated_at": now_iso(),
        "engine": args.engine,
        "workload_name": workload.get("name", Path(workload_source).stem),
        "workload_path": workload_source,
        "hardware": args.hardware,
        "model": model,
    }

    out_path = Path(args.out)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        write_jsonl(out_path, [{"kind": "plan", "meta": meta, **plan.__dict__} for plan in plans])
        report_path.write_text(render_report(meta, [], plan_only=True), encoding="utf-8")
        print(f"wrote plan {out_path}")
        print(f"wrote report {report_path}")
        return 0

    if not base_url and not launch_cmd:
        raise SystemExit("refusing to run: pass --base-url, --launch-cmd, or --plan-only")

    proc = None
    results: list[dict[str, Any]] = []
    try:
        if launch_cmd:
            proc = launch_server(launch_cmd)
        if not base_url:
            base_url = engine_config.get("base_url", "http://127.0.0.1:8080/v1")
        if health_url:
            wait_health_url(str(health_url), args.health_timeout)
        else:
            wait_healthz(base_url.replace("/v1", ""), args.health_timeout)

        grouped: dict[tuple[int, int], list[RequestPlan]] = {}
        for plan in plans:
            grouped.setdefault((plan.concurrency, plan.prompt_token_target), []).append(plan)
        for _key, level_plans in sorted(grouped.items()):
            results.extend(run_level(base_url, model, level_plans, args.timeout))
    finally:
        stop_server(proc)

    summaries = summarize(results)
    write_jsonl(out_path, [{"kind": "result", "meta": meta, **row} for row in results])
    report_path.write_text(render_report(meta, summaries, plan_only=False), encoding="utf-8")
    print(f"wrote results {out_path}")
    print(f"wrote report {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
