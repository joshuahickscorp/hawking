#!/usr/bin/env python3
"""Execution-grounded thesis gate for the hawking serve path.

Answers "can the local model code" with an HONEST, execution-checked number, not a
substring proxy (the frontier read is unanimous that substring / self-judge scoring is
dead and execution is the only real accept signal). For each task it hits the live
OpenAI-compatible endpoint, extracts the code block, executes the model's function
against real asserts in a subprocess with a timeout, and reports pass@1 with a Wilson
95% interval plus a receipt JSON.

House rules: no em or en dashes. This is a LAB harness (R0/R1): a first number on this
box, author-rerunnable, below R3. It does not back a public WIN; it moves the thesis
gate off UNPROVEN.

Usage:
  python3 tools/eval/thesis_gate.py \
      --endpoint http://127.0.0.1:8899 --model qwen \
      --corpus tools/eval/thesis_smoke_corpus_v0.jsonl \
      --out reports/eval/thesis_gate_<label>.json
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.request


def wilson(passes, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


CODE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Pull the first fenced code block; fall back to the whole body."""
    m = CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def call_endpoint(endpoint, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    dt = time.time() - t0
    content = payload["choices"][0]["message"]["content"]
    return content, dt


def run_python_task(code, entry, test, timeout):
    """Exec the model's code + the test in a fresh subprocess. Pass = exit 0."""
    program = code + "\n\n" + test + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0:
        return True, "ok"
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (err[-1] if err else "nonzero exit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8899")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--corpus", default="tools/eval/thesis_smoke_corpus_v0.jsonl")
    ap.add_argument("--out", default="reports/eval/thesis_gate.json")
    ap.add_argument("--label", default="qwen2.5-7b-q4km-debug")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--http-timeout", type=float, default=180.0)
    ap.add_argument("--exec-timeout", type=float, default=10.0)
    args = ap.parse_args()

    tasks = []
    with open(args.corpus) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    results = []
    passes = 0
    gen_s = 0.0
    for i, t in enumerate(tasks, 1):
        try:
            content, dt = call_endpoint(
                args.endpoint, args.model, t["prompt"], args.max_tokens, args.http_timeout)
        except Exception as e:
            results.append({"id": t["id"], "passed": False, "reason": f"http:{e}"})
            print(f"[{i:2d}/{len(tasks)}] {t['id']:20s} HTTP-FAIL {e}", file=sys.stderr)
            continue
        gen_s += dt
        code = extract_code(content)
        if t.get("lang", "python") != "python":
            results.append({"id": t["id"], "passed": False, "reason": "lang-unsupported"})
            continue
        ok, reason = run_python_task(code, t.get("entry", ""), t["test"], args.exec_timeout)
        passes += 1 if ok else 0
        results.append({"id": t["id"], "passed": ok, "reason": reason, "gen_s": round(dt, 2)})
        print(f"[{i:2d}/{len(tasks)}] {t['id']:20s} {'PASS' if ok else 'FAIL':4s} {reason}", file=sys.stderr)

    n = len(tasks)
    lo, hi = wilson(passes, n)
    report = {
        "gate": "thesis_gate_v0_execution_grounded",
        "grade": "MEASURED-LAB (R0/R1: first on-box number, below R3, not a public WIN)",
        "endpoint": args.endpoint,
        "model_label": args.label,
        "corpus": args.corpus,
        "n": n,
        "passes": passes,
        "pass_at_1": round(passes / n, 4) if n else 0.0,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "total_generation_s": round(gen_s, 1),
        "results": results,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== THESIS GATE ===")
    print(f"model     : {args.label}")
    print(f"pass@1    : {passes}/{n} = {report['pass_at_1']:.1%}  (Wilson95 {lo:.1%} to {hi:.1%})")
    print(f"receipt   : {args.out}")


if __name__ == "__main__":
    main()
