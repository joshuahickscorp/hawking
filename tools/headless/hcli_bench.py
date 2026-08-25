#!/usr/bin/env python3
"""G039 — HCLI_AUTONOMOUS_BENCH + HCLI_SCORE (S011 §26, §27, §59-§61).

The capability suite asks single-shot questions. Production does not: it hands the model
a mission, the model calls tools, the tools answer, verification fails, the model repairs,
and only then is a WorkUnit accepted. This measures that loop.

A WorkUnit is ACCEPTED only when a DETERMINISTIC verifier says so -- code is executed,
JSON is parsed against a predicate, a tool call is matched against a schema. No model
grades any model, because that makes the score inherit the unreliability it exists to
detect.

The headline is VERIFIED ACCEPTED WORKUNITS PER HOUR. It is a rate, so a body that is
correct but slow and a body that is fast but wrong both score badly, which is the point.

Generated code runs in a subprocess with a wall timeout, in a scratch directory, with no
arguments and no network use by the harness. It is the model's own code under test.
"""
import argparse, json, re, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/headless"))

TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.S)
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)

TOOL_SCHEMAS = [
    {"name": "run_python", "description": "Execute a python snippet and return stdout.",
     "args": {"code": "string"}},
    {"name": "read_file", "description": "Return the contents of a file by path.",
     "args": {"path": "string"}},
    {"name": "list_dir", "description": "List entries of a directory.",
     "args": {"path": "string"}},
]

SYSTEM = (
    "You are an autonomous engineering agent. Work in as few turns as possible.\n"
    "To call a tool, emit exactly one line of the form:\n"
    "<tool>{\"name\": \"<tool>\", \"args\": {...}}</tool>\n"
    "Available tools:\n"
    + "\n".join(f"  {t['name']}({', '.join(t['args'])}) - {t['description']}"
                for t in TOOL_SCHEMAS)
    + "\nWhen you are finished, give the final answer directly with no tool call.\n"
      "Put any code in a ```python fenced block."
)


def run_python(code, timeout=15):
    """Execute model-authored code in a scratch dir with a wall timeout."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.py"
        f.write_text(code)
        try:
            p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
            return {"ok": p.returncode == 0, "stdout": p.stdout[-2000:],
                    "stderr": p.stderr[-2000:], "exit_code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"timeout after {timeout}s",
                    "exit_code": -1}


def exec_tool(call):
    name, args = call.get("name"), call.get("args") or {}
    if name == "run_python":
        return run_python(args.get("code", ""))
    if name == "read_file":
        p = Path(args.get("path", ""))
        return ({"ok": True, "content": p.read_text()[:2000]} if p.is_file()
                else {"ok": False, "error": "no such file"})
    if name == "list_dir":
        p = Path(args.get("path", ""))
        return ({"ok": True, "entries": sorted(x.name for x in p.iterdir())[:50]}
                if p.is_dir() else {"ok": False, "error": "no such directory"})
    return {"ok": False, "error": f"unknown tool {name!r}"}


# --------------------------------------------------------------- verifiers

def verify_code(reply, asserts):
    """Extract the model's code, append hidden asserts, execute. Truth is the exit code."""
    m = CODE_RE.search(reply or "")
    if not m:
        return False, "no python code block in the reply"
    r = run_python(m.group(1) + "\n\n" + asserts)
    return r["ok"], ("asserts passed" if r["ok"]
                     else f"execution failed: {(r['stderr'] or '')[-200:]}")


def verify_json(reply, pred):
    for cand in re.findall(r"\{.*\}", reply or "", re.S):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        ok, why = pred(obj)
        if ok:
            return True, why
    return False, "no parseable JSON object satisfying the predicate"


def verify_tool_call(reply, want_name, want_pred):
    m = TOOL_RE.search(reply or "")
    if not m:
        return False, "no tool call emitted"
    try:
        c = json.loads(m.group(1))
    except Exception:
        return False, "tool call is not valid JSON"
    if c.get("name") != want_name:
        return False, f"selected {c.get('name')!r}, expected {want_name!r}"
    return want_pred(c.get("args") or {})


WORKUNITS = [
    {"id": "wu-code-dedupe", "axis": "code_generation", "max_turns": 3,
     "mission": "Write a python function `dedupe(xs)` that returns the items of list "
                "`xs` with duplicates removed, preserving first-appearance order. "
                "Reply with only a ```python block defining it.",
     "verify": lambda r: verify_code(r, "assert dedupe([3,1,3,2,1])==[3,1,2]\n"
                                        "assert dedupe([])==[]\n"
                                        "assert dedupe(['a','a','b'])==['a','b']\n"
                                        "print('ok')")},
    {"id": "wu-code-repair", "axis": "repair", "max_turns": 3,
     "mission": "This function is wrong:\n```python\ndef median(xs):\n"
                "    xs = sorted(xs)\n    return xs[len(xs)//2]\n```\n"
                "It fails on even-length input: median([1,2,3,4]) returns 3 but should "
                "return 2.5. Reply with a corrected ```python block defining `median`.",
     "verify": lambda r: verify_code(r, "assert median([1,2,3,4])==2.5\n"
                                        "assert median([1,3,2])==2\n"
                                        "print('ok')")},
    {"id": "wu-tool-select", "axis": "tool_selection", "max_turns": 2,
     "mission": "You need the contents of the file at /etc/hostname. Emit exactly one "
                "tool call to obtain it, and nothing else.",
     "verify": lambda r: verify_tool_call(
         r, "read_file",
         lambda a: (a.get("path") == "/etc/hostname",
                    f"path={a.get('path')!r}"))},
    {"id": "wu-structured-extract", "axis": "structured_output", "max_turns": 3,
     "mission": "Extract from this text into JSON with exactly the keys "
                "\"name\", \"port\", \"tls\": "
                "'The service auth-gw listens on port 8443 with TLS enabled.' "
                "Reply with only the JSON object.",
     "verify": lambda r: verify_json(
         r, lambda o: ((o.get("name") == "auth-gw" and int(o.get("port", 0)) == 8443
                        and o.get("tls") in (True, "true", "enabled")),
                       f"got {o}"))},
    {"id": "wu-multiturn-mission", "axis": "multi_turn", "max_turns": 4,
     "mission": "Step 1: use the run_python tool to compute the sum of squares of the "
                "integers 1..20. Step 2: once you have the number, reply with the final "
                "line exactly `ANSWER: <number>`.",
     "verify": lambda r: (bool(re.search(r"ANSWER:\s*2870\b", r or "")),
                          "expected ANSWER: 2870")},
    {"id": "wu-self-verify", "axis": "verification", "max_turns": 4,
     "mission": "Someone claims `sorted(set([2,2,1]))` evaluates to [1,2]. Use the "
                "run_python tool to check, then reply with the final line exactly "
                "`VERDICT: TRUE` or `VERDICT: FALSE`.",
     "verify": lambda r: (bool(re.search(r"VERDICT:\s*TRUE\b", r or "")),
                          "expected VERDICT: TRUE")},
]


# --------------------------------------------------------------- backends

_TOK = {}


def _chat(tokenizer_dir, msgs, no_think):
    from transformers import AutoTokenizer
    if "t" not in _TOK:
        _TOK["t"] = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tok = _TOK["t"]
    try:
        return tok, tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=not no_think)
    except TypeError:
        return tok, tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)


def call_noetic(binary, root, tokenizer_dir, msgs, max_tokens, no_think, timeout):
    """Multi-turn against the one-shot CLI: the whole transcript is re-templated each
    turn, which is what a stateless greedy binary requires."""
    tok, text = _chat(tokenizer_dir, msgs, no_think)
    budget = max_tokens if no_think else max_tokens * 3
    n_prompt = len(tok(text)["input_ids"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out = f.name
    cmd = [str(binary), "--artifact-root", str(root),
           "--tokenizer", str(Path(tokenizer_dir) / "tokenizer.json"),
           "--prompt", text, "--max-new-tokens", str(budget),
           "--max-seq-len", str(n_prompt + budget + 16), "--out", out, "--raw-prompt"]
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    body = {}
    try:
        body = json.loads(Path(out).read_text())
    except Exception:
        pass
    Path(out).unlink(missing_ok=True)
    raw = body.get("generated_text") or ""
    # identical rule to the capability suite: a generation that never closes <think>
    # never left the reasoning block and produced no reply
    unterminated = not no_think and "</think>" not in raw and bool(raw.strip())
    reply = raw.split("</think>", 1)[1] if "</think>" in raw else ("" if unterminated else raw)
    return {"text": reply.strip(), "wall_s": round(time.time() - t0, 3),
            "n_new_tokens": len(body.get("new_token_ids") or []),
            "unterminated_think_block": unterminated,
            "exit_code": pr.returncode}


def call_llama(endpoint, msgs, max_tokens, timeout):
    import urllib.request
    payload = {"model": "local", "messages": msgs, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": False}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]["message"]["content"] or ""
    reply = ch.split("</think>", 1)[1] if "</think>" in ch else ch
    return {"text": reply.strip(), "wall_s": round(time.time() - t0, 3),
            "n_new_tokens": d.get("usage", {}).get("completion_tokens", 0),
            "unterminated_think_block": False, "exit_code": 0}


# --------------------------------------------------------------- the loop

def run_workunit(wu, call, max_tokens):
    """One WorkUnit: tool turns, then verification, then repair turns if it failed."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": wu["mission"]}]
    turns, tool_calls, tool_errors, tokens, repairs = 0, 0, 0, 0, 0
    trace, verified, why = [], False, "no turn produced a reply"
    t0 = time.time()

    while turns < wu["max_turns"]:
        turns += 1
        r = call(msgs, max_tokens)
        tokens += r.get("n_new_tokens") or 0
        reply = r["text"]
        trace.append({"turn": turns, "wall_s": r["wall_s"],
                      "n_new_tokens": r.get("n_new_tokens"),
                      "unterminated_think_block": r.get("unterminated_think_block"),
                      "reply_head": reply[:200]})
        msgs = msgs + [{"role": "assistant", "content": reply}]

        m = TOOL_RE.search(reply)
        if m and turns < wu["max_turns"]:
            tool_calls += 1
            try:
                call_obj = json.loads(m.group(1))
            except Exception:
                tool_errors += 1
                msgs.append({"role": "user",
                             "content": "That tool call was not valid JSON. Retry."})
                trace[-1]["tool"] = "INVALID_JSON"
                continue
            res = exec_tool(call_obj)
            if not res.get("ok"):
                tool_errors += 1
            trace[-1]["tool"] = {"call": call_obj, "result_ok": res.get("ok")}
            msgs.append({"role": "user",
                         "content": f"Tool result: {json.dumps(res)[:1500]}"})
            continue

        verified, why = wu["verify"](reply)
        trace[-1]["verify"] = {"verified": verified, "why": why}
        if verified:
            break
        if turns < wu["max_turns"]:
            repairs += 1
            msgs.append({"role": "user",
                         "content": f"That did not pass verification: {why}. "
                                    f"Fix it and reply again."})

    wall = round(time.time() - t0, 3)
    return {"id": wu["id"], "axis": wu["axis"], "verified": verified, "why": why,
            "turns": turns, "tool_calls": tool_calls, "tool_errors": tool_errors,
            "repair_loops": repairs, "wall_s": wall, "model_tokens": tokens,
            "trace": trace}


def main():
    import statistics
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["noetic", "llama"], required=True)
    ap.add_argument("--artifact-root")
    ap.add_argument("--noetic-binary")
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--no-think", action="store_true", default=False)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.backend == "noetic":
        def call(msgs, mt):
            return call_noetic(a.noetic_binary, a.artifact_root,
                               a.tokenizer_dir or a.artifact_root, msgs, mt,
                               a.no_think, a.timeout)
    else:
        def call(msgs, mt):
            return call_llama(a.endpoint, msgs, mt, a.timeout)

    t0 = time.time()
    results = []
    for wu in WORKUNITS:
        r = run_workunit(wu, call, a.max_tokens)
        results.append(r)
        print(f"  {r['id']:24s} verified={str(r['verified']):5s} turns={r['turns']} "
              f"tools={r['tool_calls']} repairs={r['repair_loops']} "
              f"{r['wall_s']:7.1f}s  {r['why'][:54]}", flush=True)
    wall = time.time() - t0

    acc = [r for r in results if r["verified"]]
    lat = sorted(r["wall_s"] for r in results)
    n = len(results)
    score = {
        "verified_accepted_workunits": len(acc),
        "total_workunits": n,
        "acceptance_rate": round(len(acc) / n, 4),
        "total_wall_s": round(wall, 3),
        "VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR": round(len(acc) / (wall / 3600.0), 3),
        "median_workunit_latency_s": round(statistics.median(lat), 3),
        "p95_workunit_latency_s": round(lat[max(0, int(0.95 * n) - 1)], 3),
        "repair_loops_per_workunit": round(sum(r["repair_loops"] for r in results) / n, 3),
        "tool_calls": sum(r["tool_calls"] for r in results),
        "tool_errors": sum(r["tool_errors"] for r in results),
        "tool_reliability": (round(1 - sum(r["tool_errors"] for r in results)
                                   / max(1, sum(r["tool_calls"] for r in results)), 4)),
        "model_tokens": sum(r["model_tokens"] for r in results),
        "tokens_per_accepted_workunit": (round(sum(r["model_tokens"] for r in results)
                                               / len(acc), 1) if acc else None),
    }
    out = {
        "schema": "hawking.headless.hcli_bench.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/hcli_bench.py",
        "obligation": "G039 — HCLI_AUTONOMOUS_BENCH + HCLI_SCORE",
        "hand_authored": False,
        "label": a.label, "backend": a.backend,
        "artifact_root": a.artifact_root,
        "headline": "VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR",
        "scoring": "every WorkUnit is accepted only by a deterministic verifier -- code "
                   "executed against hidden asserts, JSON parsed against a predicate, a "
                   "tool call matched against a schema. No model grades any model.",
        "axes_covered": sorted({w["axis"] for w in WORKUNITS}),
        "score": score, "workunits": results,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n{a.label}: {score['verified_accepted_workunits']}/{n} accepted, "
          f"{score['VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR']} verified WUs/hour, "
          f"median {score['median_workunit_latency_s']}s, "
          f"repairs/WU {score['repair_loops_per_workunit']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
