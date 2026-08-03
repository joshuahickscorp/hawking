#!/usr/bin/env python3.12
"""Execution-grounded quality gate: generate a function, then run its tests.

Rebuilt to the shape the sealed baselines already use, so
`reports/eval/thesis_gate_qwen7b_q4km.json` (pass@1 0.9333, n=15) stays a
valid comparison. The metric and the corpora are fixed on purpose: widening
either one silently invalidates every earlier receipt.

    python3.12 tools/eval/thesis_gate.py --weights models/X.gguf
    python3.12 tools/eval/thesis_gate.py --weights models/X.gguf --corpus rust
    python3.12 tools/eval/thesis_gate.py --endpoint http://127.0.0.1:8899 --model-label X

Passing requires the generated code to *execute* against the corpus test, not
to look plausible. A model that emits prose, an empty block, or code that does
not compile fails, and the reason is recorded per item.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPORA = {
    "python": ROOT / "tools/eval/thesis_smoke_corpus_v0.jsonl",
    "rust": ROOT / "tools/eval/thesis_rust_corpus_v0.jsonl",
}
GATE_ID = "thesis_gate_v0_execution_grounded"
FENCE = re.compile(r"```(?:[a-zA-Z+]*)\n(.*?)```", re.S)


def wilson95(passes: int, n: int) -> list[float]:
    """Wilson score interval. Report this, not the point estimate: at n=15 a
    one-task difference is not significant."""
    if n == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = passes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - s) / d), 4), round(min(1.0, (c + s) / d), 4)]


def extract_code(text: str) -> str:
    """First fenced block, else the raw text. Models that ignore the fence
    instruction are not punished twice."""
    m = FENCE.search(text)
    return (m.group(1) if m else text).strip()


def generate_cli(weights: Path, prompt: str, max_new: int, profile: str) -> tuple[str, float]:
    cmd = [
        str(ROOT / "target/release/hawking"), "generate",
        "--weights", str(weights), "--prompt", prompt,
        "--max-new-tokens", str(max_new), "--temperature", "0", "--seed", "5",
    ]
    if profile:
        cmd += ["--profile", profile]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    # Drop the runtime's own preamble and stats lines; keep only model output.
    body = [
        ln for ln in r.stdout.splitlines()
        if not ln.startswith(("[hawking]", "[stats]", "[stats-json]"))
    ]
    return "\n".join(body), dt


def generate_http(endpoint: str, prompt: str, max_new: int) -> tuple[str, float]:
    import urllib.request
    payload = json.dumps({
        "model": "hawking", "temperature": 0, "max_tokens": max_new,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        doc = json.loads(resp.read())
    return doc["choices"][0]["message"]["content"], time.time() - t0


def run_python(code: str, entry: str, test: str) -> tuple[bool, str]:
    src = f"{code}\n\n{test}\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "case.py"
        f.write_text(src)
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if r.returncode == 0:
        return True, "ok"
    tail = (r.stderr or r.stdout).strip().splitlines()
    return False, tail[-1][:200] if tail else f"exit {r.returncode}"


def run_rust(code: str, test: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        f, exe = Path(d) / "case.rs", Path(d) / "case"
        f.write_text(f"{code}\n\n{test}\n")
        try:
            c = subprocess.run(["rustc", "-O", "-o", str(exe), str(f)],
                               capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            return False, "rustc absent"
        except subprocess.TimeoutExpired:
            return False, "compile timeout"
        if c.returncode != 0:
            err = [l for l in c.stderr.splitlines() if l.startswith("error")]
            return False, (err[0] if err else "compile failed")[:200]
        try:
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return (True, "ok") if r.returncode == 0 else (False, (r.stderr.strip()[:200] or "assert failed"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", type=Path, help="drive the CLI directly")
    src.add_argument("--endpoint", help="drive a running hawking serve")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="python")
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--profile", default="exact",
                    help="exact keeps decode bit-identical; pass '' for the runtime default")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    corpus = CORPORA[a.corpus]
    cases = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
    label = a.model_label or (a.weights.stem if a.weights else a.endpoint)

    results, passes, total_s = [], 0, 0.0
    for c in cases:
        try:
            if a.weights:
                out, dt = generate_cli(a.weights, c["prompt"], a.max_new_tokens, a.profile)
            else:
                out, dt = generate_http(a.endpoint, c["prompt"], a.max_new_tokens)
        except subprocess.TimeoutExpired:
            results.append({"id": c["id"], "passed": False, "reason": "generation timeout", "gen_s": 0.0})
            continue
        total_s += dt
        code = extract_code(out)
        if not code:
            ok, why = False, "empty generation"
        elif c["lang"] == "rust":
            ok, why = run_rust(code, c["test"])
        else:
            ok, why = run_python(code, c.get("entry", ""), c["test"])
        passes += ok
        results.append({"id": c["id"], "passed": ok, "reason": why, "gen_s": round(dt, 2)})
        print(f"  {'PASS' if ok else 'FAIL'}  {c['id']:<20} {why}", flush=True)

    n = len(cases)
    doc = {
        "gate": GATE_ID,
        "grade": "MEASURED-LAB (R0/R1: on-box number, below R3, not a public WIN)",
        "endpoint": a.endpoint,
        "model_label": label,
        "corpus": str(corpus.relative_to(ROOT)),
        "n": n,
        "passes": passes,
        "pass_at_1": round(passes / n, 4) if n else 0.0,
        "wilson95": wilson95(passes, n),
        "total_generation_s": round(total_s, 1),
        "results": results,
    }
    out = a.out or ROOT / f"workspace/campaign/records/reports/eval/thesis_gate_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n  pass@1 {doc['pass_at_1']} ({passes}/{n})  wilson95 {doc['wilson95']}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
