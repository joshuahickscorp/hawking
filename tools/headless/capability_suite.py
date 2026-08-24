#!/usr/bin/env python3
"""Doctor / Tabula capability gate (directive §12, §13).

    "Every promoted Qwen child must preserve the required capability contract."
    "No 'feels about the same.'"

MLX is currently ELIGIBLE on performance (+47.67% decode over llama.cpp on the
same weights) and cannot be promoted, because nothing here yet asks whether it
still WORKS. This is that check.

DESIGN CONSTRAINT THAT SHAPES EVERYTHING BELOW
----------------------------------------------
A model must never grade itself, and a model must never grade another model,
because then the gate inherits exactly the unreliability it exists to detect.
Every item scores through a deterministic predicate over the response text:
an exact string, a parsed JSON structure, a compiled AST, a executed test.
If an item cannot be scored deterministically it does not belong in the suite.

That rules out most "quality" evals, and deliberately so. What survives is a
contract: strict structured output, tool-protocol adherence, exact mutations,
repository reasoning, self-correction, refusal to hallucinate a path. Those are
the behaviours HCLI actually depends on, and they are all checkable.

Scoring is per-item pass/fail with N repeats, because a capability that works
4 times in 5 is a different thing from one that works every time, and an agent
loop turns a 20% failure into a repair cycle.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))


# --------------------------------------------------------------- predicates

def contains_all(*needles):
    def f(text, _):
        low = (text or "").lower()
        return all(n.lower() in low for n in needles), f"expected all of {needles}"
    return f


def exact_number(value):
    def f(text, _):
        nums = re.findall(r"-?\d[\d,]*", text or "")
        got = [int(n.replace(",", "")) for n in nums]
        return (value in got), f"expected {value} among the numbers emitted, got {got[:8]}"
    return f


def valid_json_obj(required_keys=()):
    def f(text, _):
        obj = extract_json(text)
        if obj is None:
            return False, "no parseable JSON object in the reply"
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return False, f"JSON parsed but missing keys {missing}"
        return True, ""
    return f


def json_matches(pred, describe):
    def f(text, _):
        obj = extract_json(text)
        if obj is None:
            return False, "no parseable JSON object in the reply"
        try:
            return bool(pred(obj)), describe
        except Exception as e:
            return False, f"{describe} (predicate raised {type(e).__name__}: {e})"
    return f


def python_compiles_and(pred=None, describe=""):
    def f(text, _):
        code = extract_code(text)
        if not code:
            return False, "no python code block in the reply"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"emitted python does not parse: {e}"
        if pred is None:
            return True, ""
        try:
            return bool(pred(tree, code)), describe
        except Exception as e:
            return False, f"{describe} (predicate raised {type(e).__name__}: {e})"
    return f


def must_not_contain(*needles):
    def f(text, _):
        low = (text or "").lower()
        hit = [n for n in needles if n.lower() in low]
        return (not hit), f"reply contained forbidden {hit}"
    return f


def extract_json(text):
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    start = -1
    return None


def extract_code(text):
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    return m.group(1) if m else None


# --------------------------------------------------------------- the contract

MUTATION_SCHEMA_KEYS = ("kind", "content", "operations", "tests")

SUITE = [
    # ---- factual floor: if these move, the weights are damaged ----
    dict(id="fact-capital", axis="knowledge", repeats=3,
         prompt="What is the capital of France? Answer with only the city name.",
         check=contains_all("paris")),
    dict(id="fact-arith", axis="reasoning", repeats=3,
         prompt="Compute 17 * 19. Reply with only the number.",
         check=exact_number(323)),
    dict(id="fact-multistep", axis="reasoning", repeats=3,
         prompt=("A repo has 12 modules. 3 are deleted, then 5 are added, then a quarter of "
                 "the total are split in two. How many modules are there at the end? "
                 "Reply with only the final number."),
         check=exact_number(21)),

    # ---- strict structured output: HCLI's entire contract ----
    dict(id="json-answer", axis="structured_output", repeats=5,
         system=("Return exactly one JSON object and nothing else, of the form "
                 '{"kind":"answer","content":"...","operations":[],"tests":[]}'),
         prompt="What does the `os.replace` call guarantee that `os.rename` may not?",
         check=valid_json_obj(MUTATION_SCHEMA_KEYS)),
    dict(id="json-kind-correct", axis="structured_output", repeats=5,
         system=('Return exactly one JSON object: {"kind":"answer"|"mutation","content":"...",'
                 '"operations":[...],"tests":[...]}. Use "answer" when nothing should change.'),
         prompt="Explain what a mutex is. Do not modify any file.",
         check=json_matches(lambda o: o.get("kind") == "answer" and o.get("operations") == [],
                            'a read-only request must yield kind="answer" with no operations')),
    dict(id="json-no-prose", axis="structured_output", repeats=5,
         system='Return exactly one JSON object and nothing else.',
         prompt='Reply with {"kind":"answer","content":"ok","operations":[],"tests":[]}',
         check=valid_json_obj(MUTATION_SCHEMA_KEYS)),

    # ---- exact mutation: the thing that actually edits the repo ----
    dict(id="mutation-anchor-exact", axis="mutation", repeats=5,
         system=('Return exactly one JSON object: {"kind":"mutation","content":"...",'
                 '"operations":[{"op":"replace","path":"...","old_text":"...","new_text":"..."}],'
                 '"tests":["..."]}. old_text must be copied EXACTLY from the file shown.'),
         prompt=("File calc.py contains exactly:\n"
                 "def add(a, b):\n    return a - b\n\n"
                 "Change it so add returns the sum. Emit one replace operation on calc.py."),
         check=json_matches(
             lambda o: (o.get("kind") == "mutation"
                        and len(o.get("operations") or []) >= 1
                        and o["operations"][0].get("path", "").endswith("calc.py")
                        and "a - b" in (o["operations"][0].get("old_text") or "")
                        and "a + b" in (o["operations"][0].get("new_text") or "")),
             "must emit a replace on calc.py whose old_text carries `a - b` and new_text `a + b`")),
    dict(id="mutation-refuses-invention", axis="mutation", repeats=5,
         system=('Return exactly one JSON object. If the requested file was not shown to you, '
                 'return {"kind":"answer",...} explaining that, rather than inventing content.'),
         prompt=("Rename the function in totally_unseen_module.py. You have NOT been shown that "
                 "file's contents."),
         check=json_matches(lambda o: o.get("kind") == "answer" or not (o.get("operations") or []),
                            "must not emit operations against a file it was never shown")),

    # ---- code that actually compiles ----
    dict(id="code-compiles", axis="coding", repeats=3,
         prompt=("Write a Python function `dedupe(xs)` that removes duplicates while preserving "
                 "first-seen order. Reply with a single ```python code block and nothing else."),
         check=python_compiles_and(
             lambda tree, code: any(isinstance(n, ast.FunctionDef) and n.name == "dedupe"
                                    for n in ast.walk(tree)),
             "must define a function named dedupe")),
    dict(id="code-self-correct", axis="self_correction", repeats=3,
         prompt=("This function is wrong:\n```python\ndef mean(xs):\n    return sum(xs) / len(xs) + 1\n```\n"
                 "State the bug in one sentence, then give the corrected function in a single "
                 "```python block."),
         check=python_compiles_and(
             lambda tree, code: "+ 1" not in code.replace(" ", "") .replace("+1", "+ 1")
                                and "sum(xs)" in code and "len(xs)" in code,
             "the corrected function must drop the spurious + 1")),

    # ---- no leaked reasoning: receipts must stay clean ----
    dict(id="no-think-leak", axis="hygiene", repeats=3,
         system="Return exactly one JSON object. Do not include <think> blocks or hidden reasoning.",
         prompt='Reply with {"kind":"answer","content":"ok","operations":[],"tests":[]}',
         check=must_not_contain("<think>", "</think>", "reasoning_content")),
]


# --------------------------------------------------------------- backends

def call_llama(endpoint, system, prompt, max_tokens, no_think, timeout):
    payload = {
        "model": "local",
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{endpoint}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8", "replace"))
    ch = (body.get("choices") or [{}])[0]
    return {
        "text": (ch.get("message") or {}).get("content") or "",
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
        "wall_s": round(time.time() - t0, 3),
    }


MLX_RUNNER = r'''
import json, sys, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
path = sys.argv[1]
items = json.loads(sys.stdin.read())
model, tok = load(path)
sampler = make_sampler(temp=0.0)
out = []
for it in items:
    msgs = ([{"role":"system","content":it["system"]}] if it.get("system") else []) + \
           [{"role":"user","content":it["prompt"]}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    resp = generate(model, tok, prompt=text, max_tokens=it["max_tokens"],
                    sampler=sampler, verbose=False)
    out.append({"id": it["id"], "rep": it["rep"], "text": resp,
                "wall_s": round(time.time()-t0, 3),
                "completion_tokens": len(tok.encode(resp)), "finish_reason": None})
print(json.dumps(out))
'''


def run_mlx(model_path, items, mlx_py, timeout):
    p = subprocess.run([mlx_py, "-c", MLX_RUNNER, model_path],
                       input=json.dumps(items), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"mlx runner exited {p.returncode}: {p.stderr[-1500:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------- scoring

def score(responses):
    """responses: list of {id, rep, text, ...}. Returns per-item and per-axis."""
    by_id = {}
    for r in responses:
        by_id.setdefault(r["id"], []).append(r)
    items = {i["id"]: i for i in SUITE}
    per_item, per_axis = {}, {}
    for iid, rs in by_id.items():
        spec = items[iid]
        results = []
        for r in rs:
            ok, why = spec["check"](r.get("text") or "", r)
            results.append({"rep": r.get("rep"), "pass": bool(ok),
                            "why": "" if ok else why,
                            "finish_reason": r.get("finish_reason"),
                            "completion_tokens": r.get("completion_tokens"),
                            "wall_s": r.get("wall_s"),
                            "reply_head": (r.get("text") or "")[:220]})
        passed = sum(1 for x in results if x["pass"])
        per_item[iid] = {
            "axis": spec["axis"], "repeats": len(results), "passed": passed,
            "rate": round(passed / len(results), 3) if results else 0.0,
            "results": results,
        }
        a = per_axis.setdefault(spec["axis"], {"passed": 0, "total": 0})
        a["passed"] += passed
        a["total"] += len(results)
    for a in per_axis.values():
        a["rate"] = round(a["passed"] / a["total"], 3) if a["total"] else 0.0
    return per_item, per_axis


def build_items():
    out = []
    for spec in SUITE:
        for rep in range(spec.get("repeats", 1)):
            out.append({"id": spec["id"], "rep": rep, "system": spec.get("system"),
                        "prompt": spec["prompt"], "max_tokens": spec.get("max_tokens", 512)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama", "mlx"], required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080", help="llama backend only")
    ap.add_argument("--model-path", help="mlx backend only")
    ap.add_argument("--label", required=True, help="what this run identifies, e.g. 'llamacpp-q5k'")
    ap.add_argument("--mlx-py", default=os.path.expanduser(
        "~/.local/share/uv/tools/mlx-lm/bin/python"))
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = build_items()
    print(f"suite: {len(SUITE)} items, {len(items)} calls, backend={args.backend}", flush=True)

    responses = []
    if args.backend == "mlx":
        if not args.model_path:
            print("FAIL: --model-path required for the mlx backend")
            return 2
        responses = run_mlx(args.model_path, items, args.mlx_py, args.timeout)
    else:
        for it in items:
            try:
                r = call_llama(args.endpoint, it.get("system"), it["prompt"],
                               it["max_tokens"], args.no_think, args.timeout)
            except Exception as e:
                r = {"text": "", "finish_reason": f"ERROR:{type(e).__name__}",
                     "completion_tokens": None, "wall_s": None}
            r.update({"id": it["id"], "rep": it["rep"]})
            responses.append(r)
            print(f"  {it['id']}[{it['rep']}] {r.get('finish_reason')} "
                  f"{r.get('completion_tokens')}tok {r.get('wall_s')}s", flush=True)

    per_item, per_axis = score(responses)
    overall_pass = sum(v["passed"] for v in per_item.values())
    overall_total = sum(v["repeats"] for v in per_item.values())

    doc = {
        "schema": "hawking.headless.capability_suite.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "backend": args.backend,
        "target": args.model_path or args.endpoint,
        "scoring": ("every item scores through a deterministic predicate over the reply text — "
                    "exact string, parsed JSON, compiled AST. No model grades any model, because "
                    "that would make the gate inherit the unreliability it exists to detect."),
        "overall": {"passed": overall_pass, "total": overall_total,
                    "rate": round(overall_pass / overall_total, 4) if overall_total else 0.0},
        "per_axis": per_axis,
        "per_item": per_item,
    }
    out = Path(args.out or (REPO / f"receipts/headless/CAPABILITY_{args.label}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))

    print(f"\n=== CAPABILITY {args.label} ===")
    print(f"  overall {overall_pass}/{overall_total} = {doc['overall']['rate']}")
    for axis, v in sorted(per_axis.items()):
        print(f"    {axis:<20} {v['passed']:>3}/{v['total']:<3} = {v['rate']}")
    weak = [(k, v) for k, v in sorted(per_item.items()) if v["rate"] < 1.0]
    if weak:
        print("  items below 1.0:")
        for k, v in weak:
            first = next((r for r in v["results"] if not r["pass"]), {})
            print(f"    {k:<28} {v['passed']}/{v['repeats']}  {first.get('why','')[:90]}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
