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

import argparse, hashlib
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Derived from this file, never a hardcoded home path: ~/Downloads/hawking-copy
# exists on this machine, so the old constant silently wrote receipts into a
# DIFFERENT repository whenever --out was omitted.
REPO = Path(__file__).resolve().parents[2]


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
        # An empty reply contains nothing forbidden, so this predicate used to PASS on a
        # model that emitted nothing at all. The 2.5970-EBPW body scored 3/43 that way --
        # every one of the 3 was this axis passing on empty output, while every axis
        # requiring content scored 0. A check a dead model passes is not a check.
        if not (text or "").strip():
            return False, "empty reply: nothing was produced to be clean"
        low = text.lower()
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

_NOETIC_TOK = {}


# A reasoning model spends budget on the reasoning before it writes a word of the answer.
# The llama and mlx baselines answer inside the item's max_tokens because their replies are
# the answer; the noetic backend must emit a whole <think> block first. Capping both at the
# same number measures BUDGET, not capability -- code-compiles failed on the sealed body
# with "unterminated string literal" at exactly 512 tokens, and passed at 544 when given
# room. The multiplier restores the comparison rather than loosening it.
NOETIC_BUDGET_MULTIPLIER = 3


def call_noetic(binary, artifact_root, system, prompt, max_tokens, no_think, timeout,
                tokenizer_dir=None):
    """The rebuilt noetic executable: a greedy CLI over the sealed closure.

    The closure carries its own tokenizer and chat template (that is what makes it
    parent-free), so the template is applied from the ARTIFACT, never from the parent
    directory -- applying the parent's would quietly reintroduce the dependency the
    zero-parent gate exists to remove.
    """
    import subprocess, tempfile
    root = Path(artifact_root)
    # The tokenizer is decoupled from the artifact root on purpose: the SEALED artifact
    # carries no tokenizer state at all (MIX_REPORT.json, catalog, segments -- nothing
    # else), which is the closure gap the rebuild fixed. Scoring it requires pointing at a
    # closure that does have one, and the two are byte-identical.
    tdir = Path(tokenizer_dir or artifact_root)
    if "tok" not in _NOETIC_TOK:
        from transformers import AutoTokenizer
        _NOETIC_TOK["tok"] = AutoTokenizer.from_pretrained(str(tdir))
    _NOETIC_TOK["dir"] = str(tdir)
    tok = _NOETIC_TOK["tok"]
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=not no_think)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    budget = max_tokens if no_think else max_tokens * NOETIC_BUDGET_MULTIPLIER
    n_prompt = len(tok(text)["input_ids"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out_json = f.name
    cmd = [str(binary), "--artifact-root", str(root),
           "--tokenizer", str(tdir / "tokenizer.json"),
           "--prompt", text, "--max-new-tokens", str(budget),
           "--max-seq-len", str(n_prompt + budget + 16),
           "--out", out_json, "--raw-prompt"]
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    body = {}
    try:
        body = json.loads(Path(out_json).read_text())
    except Exception:
        pass
    Path(out_json).unlink(missing_ok=True)
    raw = body.get("generated_text") or ""
    # A serving backend returns the reply, not the reasoning trace. The llama and mlx
    # baselines this run is compared against are scored on post-</think> content, so the
    # raw CLI output has to be cut the same way or the comparison measures the harness.
    #
    # The chat template ends the prompt with an OPEN <think>, so a generation that never
    # emits </think> never left the reasoning block and never produced an answer. Scoring
    # its raw reasoning prose as if it were the reply is what let the 2.5970-EBPW body pass
    # a "must not leak </think>" check: it passed precisely BECAUSE it never finished
    # thinking. An unterminated think block is no reply.
    unterminated = not no_think and "</think>" not in raw and bool(raw.strip())
    reply = raw.split("</think>", 1)[1] if "</think>" in raw else ("" if unterminated else raw)
    return {"text": reply.strip(), "raw_text": raw,
            "token_budget": budget, "budget_multiplier": NOETIC_BUDGET_MULTIPLIER,
            "hit_budget_cap": len(body.get("new_token_ids") or []) >= budget,
            "unterminated_think_block": unterminated,
            "think_block_stripped": "</think>" in raw,
            "wall_s": round(time.time() - t0, 3),
            "exit_code": pr.returncode,
            "n_new_tokens": len(body.get("new_token_ids") or []),
            "stderr_tail": (pr.stderr or "")[-300:] if pr.returncode else ""}


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

def measurement_weight(per_item) -> dict:
    """43 cases are not 43 measurements.

    Decoding is greedy at temperature 0, so every repetition of an item is
    BYTE-IDENTICAL to its siblings -- verified across both arms of the
    template-arm comparison, where not one of eleven items in either receipt
    held more than one distinct (completion_tokens, reply_head, pass) triple.

    So a score out of 43 is eleven measurements carrying weights
    (3,3,3,5,5,5,5,5,3,3,3), and a five-point move is ONE ITEM whose weight
    happens to be five -- not five independent successes. Reporting x/43 without
    this invites reading repetition as evidence. The repeats are not useless:
    they would catch nondeterminism, and this field is what shows they found
    none rather than leaving that unstated.
    """
    per_item = per_item or {}
    determinism = {}
    for iid, blk in per_item.items():
        trips = {(r.get("completion_tokens"), r.get("reply_head"), r.get("pass"))
                 for r in blk.get("results", [])}
        determinism[iid] = len(trips)
    return {
        "distinct_items": len(per_item),
        "cases": sum(b.get("repeats", 0) for b in per_item.values()),
        "weights": {i: b.get("repeats") for i, b in per_item.items()},
        "distinct_outputs_per_item": determinism,
        "every_repeat_identical": all(v <= 1 for v in determinism.values()),
        "how_to_read_it": (
            "a move of N points is a move of however many ITEMS carry weight N, "
            "not N independent cases. Repeats detect nondeterminism and add no "
            "independent evidence when there is none."),
    }


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
                            # finish_reason is a PROCESS EXIT CODE for the noetic
                            # backend (see the runner: "stop" iff exit_code == 0), so
                            # it reads "stop" on a call that was TRUNCATED at its
                            # budget exactly as on one that hit EOS. It cannot report
                            # truncation and the flag beside it says so, because a
                            # field that always says stop is worse than no field.
                            "finish_reason": r.get("finish_reason"),
                            "finish_reason_is_exit_code": r.get("finish_reason_is_exit_code"),
                            # Computed by the runner and previously DROPPED here, which
                            # left no field in any receipt able to distinguish EOS from
                            # a cap hit -- the eight empty open_think replies at
                            # 1135-1536 tokens all read finish_reason "stop".
                            "hit_budget_cap": r.get("hit_budget_cap"),
                            "token_budget": r.get("token_budget"),
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


def build_items(default_system=None):
    out = []
    for spec in SUITE:
        for rep in range(spec.get("repeats", 1)):
            out.append({"id": spec["id"], "rep": rep,
                        "system": spec.get("system") or default_system,
                        "system_was_default": not spec.get("system")
                                              and bool(default_system),
                        "prompt": spec["prompt"], "max_tokens": spec.get("max_tokens", 512)})
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def artifact_identity(args) -> dict:
    """What body was graded, and under which chat template.

    G073: every CAPABILITY_noetic-*.json ever written recorded
    target="http://127.0.0.1:8080" -- the llama ENDPOINT DEFAULT -- because
    `args.model_path or args.endpoint` has no noetic branch, and the noetic
    backend loads from --artifact-root. So the field named a URL for a body
    that was never served over one, and no receipt could say which artifact
    earned its score.

    The template is recorded for the same reason: G067 measured the SAME BODY
    scoring 0/43 with an open <think> and 14/43 with enable_thinking=false.
    A score without its template is not comparable to another score.

    Weight shards are identified by (name, size) rather than content hash:
    a full hash of a multi-GB body costs minutes and this runs before every
    suite. Sizes catch a different artifact; they do NOT catch an edit that
    preserves length, and identity_is_content_hash says so rather than
    letting the field read stronger than it is.
    """
    ident = {
        "backend": args.backend,
        "no_think": bool(args.no_think),
        "chat_template_arm": "pre_closed_think" if args.no_think else "open_think",
        "default_system": args.default_system,
        "identity_is_content_hash": False,
        "identity_note": "weight shards keyed by (name, size); a same-length edit is invisible",
    }
    if args.backend == "noetic":
        root = Path(args.artifact_root)
        ident["artifact_root"] = str(root)
        shards = sorted((f.name, f.stat().st_size) for f in root.rglob("*") if f.is_file())
        ident["artifact_files"] = len(shards)
        ident["artifact_bytes"] = sum(s for _, s in shards)
        ident["artifact_inventory_sha"] = hashlib.sha256(
            repr(shards).encode()).hexdigest()[:16]
        ident["binary"] = args.noetic_binary
        ident["binary_sha256_16"] = _sha(Path(args.noetic_binary))
        tok = Path(args.tokenizer_dir or args.artifact_root)
        ident["tokenizer_dir"] = str(tok)
        for name, key in (("tokenizer.json", "tokenizer_sha256_16"),
                          ("chat_template.jinja", "chat_template_sha256_16")):
            f = tok / name
            ident[key] = _sha(f) if f.is_file() else None
    elif args.backend == "mlx":
        ident["artifact_root"] = args.model_path
    else:
        ident["endpoint"] = args.endpoint
        ident["served_artifact"] = None
        ident["identity_note"] = ("a served endpoint does not disclose its artifact; the body "
                                  "behind this port is NOT established by this receipt")
    return ident


def identity_is_sufficient(ident: dict) -> tuple[bool, str]:
    """Refuse a receipt that cannot name the body it graded.

    A score bound to a hand-typed label and a URL is a claim about a LABEL.
    """
    if ident.get("backend") == "llama" and not ident.get("served_artifact"):
        return False, ("llama backend: the endpoint does not disclose which artifact answered, "
                       "so this score cannot be attributed to a body")
    if not ident.get("artifact_root"):
        return False, "no artifact_root recorded"
    return True, "artifact named"


def machine_state(responses) -> dict:
    """The machine this run happened on, RECORDED rather than left to be inferred.

    Two unpaired capability runs of the same artifact once differed by 19% of wall
    and it took reading per-repetition SPREAD to work out that one arm had run on a
    contended machine (ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json). That inference
    worked and is weaker than a field. Sampled at the START and END of the run only:
    a sample per item would perturb the thing it measures, which is the same reason
    bench.time_arm does not call it either.

    ``worst_repetition_spread_pct`` is the signal that actually caught it -- a graph
    change cannot make a kernel more REPEATABLE, so a run whose repeats scatter was
    sharing the machine. It is derived from the responses already collected and costs
    nothing.
    """
    try:
        sys.path.insert(0, str(REPO / "tools/accelerator"))
        import bench
        before = machine_state._before
        after = bench.machine_quiescence()
    except Exception as e:  # never let a diagnostic field fail a scored run
        return {"recorded": False, "why": f"{type(e).__name__}: {e}"}
    if not before:
        # An absent BEFORE sample is NOT a recorded machine state. Returning
        # recorded=True with a null field is the same failure the quiescence
        # instrument itself refuses: "I could not look" must not read as "I
        # looked and found nothing".
        return {"recorded": False,
                "why": "no quiescence sample was taken before the items ran"}

    by_id: dict[str, list[float]] = {}
    for r in responses:
        w = r.get("wall_s")
        if isinstance(w, (int, float)):
            by_id.setdefault(r["id"], []).append(float(w))
    spreads = {k: round(100 * (max(v) - min(v)) / (sorted(v)[len(v) // 2] or 1), 1)
               for k, v in by_id.items() if len(v) > 1}
    return {
        "recorded": True,
        "quiescence_before": before,
        "quiescence_after": after,
        "quiet_at_both_samples": bool(
            before and after and before.get("quiet") and after.get("quiet")),
        "worst_repetition_spread_pct": max(spreads.values()) if spreads else None,
        "repetition_spread_pct_by_item": dict(sorted(
            spreads.items(), key=lambda kv: -kv[1])),
        "how_to_read_it": (
            "a run whose repeats scatter shared the machine. A graph or model change "
            "cannot make a kernel more REPEATABLE, so a cross-run timing comparison "
            "between one run at 40% worst spread and another at 0.8% is not a "
            "comparison. PASS COUNTS do not drift with load; wall times do."),
    }


def harness_health(responses) -> dict:
    """Separate A BODY THAT ANSWERED BADLY from A HARNESS THAT NEVER ASKED.

    G073 follow-on, found by running this suite under python3.14 (no
    transformers): all 43 calls raised, every reply was empty, and the receipt
    reported overall 0/43 rate 0.0 -- INDISTINGUISHABLE IN A SUMMARY LINE from
    the 2.60 body's real 0/43 under the open-<think> arm. A score of zero is a
    claim about a MODEL; a suite that never reached the model has no claim to
    make, and reporting one is the campaign's probe-crash-labelled-as-the-
    subject's-failure defect in the scoreboard itself.

    Caught only because 0/43 CONTRADICTED a known 14/43 on the same body and
    arm. Nothing in the harness would have said so.
    """
    total = len(responses)
    errs = [r for r in responses if str(r.get("finish_reason", "")).startswith("ERROR:")]
    empties = [r for r in responses if not str(r.get("text", "")).strip()]
    kinds = sorted({str(r["finish_reason"]).split(":", 1)[1].split(":")[0].strip()
                    for r in errs}) if errs else []
    all_errored = bool(total) and len(errs) == total
    return {
        "calls": total,
        "errored": len(errs),
        "empty_replies": len(empties),
        "error_kinds": kinds,
        "every_call_errored": all_errored,
        "scoreable": not all_errored,
        "verdict": ("REFUSED: every call raised, so the suite never reached the model and "
                    f"this is a HARNESS failure, not a capability score. kinds={kinds}"
                    if all_errored else "the suite reached the model"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama", "mlx", "noetic"], required=True)
    ap.add_argument("--artifact-root", help="noetic backend only")
    ap.add_argument("--noetic-binary", help="noetic backend only")
    ap.add_argument("--tokenizer-dir", help="noetic backend only; defaults to --artifact-root")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080", help="llama backend only")
    ap.add_argument("--model-path", help="mlx backend only")
    ap.add_argument("--label", required=True, help="what this run identifies, e.g. 'llamacpp-q5k'")
    ap.add_argument("--mlx-py", default=os.path.expanduser(
        "~/.local/share/uv/tools/mlx-lm/bin/python"))
    ap.add_argument("--no-think", dest="no_think", action="store_true", default=False,
                    help="prefill an empty think block. On Qwen3.8 this makes the model "
                         "emit <|im_end|> immediately, so it is OFF by default; the first "
                         "noetic run scored 14/43 purely because it was on.")
    ap.add_argument("--think", dest="no_think", action="store_false")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    # G042 adversary: some items carry a system prompt and some do not, and G039 showed
    # the code items are decided by deliberation runaway under the no-system-prompt
    # regime. This re-scores the suite with a default system prompt for items that lack
    # one, to test whether the capability ranking is regime-dependent.
    ap.add_argument("--default-system", default=None,
                    help="system prompt applied ONLY to items that define none")
    args = ap.parse_args()

    # The BEFORE sample must be taken before any item runs, not reconstructed after.
    try:
        sys.path.insert(0, str(REPO / "tools/accelerator"))
        import bench as _bench
        machine_state._before = _bench.machine_quiescence()
    except Exception as e:
        machine_state._before = {"quiet": None, "refused": f"{type(e).__name__}: {e}"}

    items = build_items(args.default_system)
    print(f"suite: {len(SUITE)} items, {len(items)} calls, backend={args.backend}", flush=True)

    responses = []
    if args.backend == "mlx":
        if not args.model_path:
            print("FAIL: --model-path required for the mlx backend")
            return 2
        responses = run_mlx(args.model_path, items, args.mlx_py, args.timeout)
    elif args.backend == "noetic":
        if not (args.artifact_root and args.noetic_binary):
            print("FAIL: --artifact-root and --noetic-binary required for the noetic backend")
            return 2
        for it in items:
            try:
                r = call_noetic(args.noetic_binary, args.artifact_root, it.get("system"),
                                it["prompt"], it["max_tokens"], args.no_think, args.timeout,
                                tokenizer_dir=args.tokenizer_dir)
                r["finish_reason"] = "stop" if r["exit_code"] == 0 else f"EXIT{r['exit_code']}"
                r["finish_reason_is_exit_code"] = True
                r["completion_tokens"] = r.get("n_new_tokens")
            except Exception as e:
                r = {"text": "", "finish_reason": f"ERROR:{type(e).__name__}: {e}",
                     "completion_tokens": None, "wall_s": None}
            r.update({"id": it["id"], "rep": it["rep"]})
            responses.append(r)
            print(f"  {it['id']}[{it['rep']}] {r.get('finish_reason')} "
                  f"{r.get('completion_tokens')}tok {r.get('wall_s')}s", flush=True)
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

    _machine = machine_state(responses)
    _health = harness_health(responses)
    _ident = artifact_identity(args)
    _ident_ok, _ident_why = identity_is_sufficient(_ident)
    per_item, per_axis = score(responses)
    overall_pass = sum(v["passed"] for v in per_item.values())
    overall_total = sum(v["repeats"] for v in per_item.values())

    doc = {
        "schema": "hawking.headless.capability_suite.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "backend": args.backend,
        "target": args.model_path or args.artifact_root or args.endpoint,
        "artifact_identity": _ident,
        "identity_sufficient": _ident_ok,
        "identity_verdict": _ident_why,
        "scoring": ("every item scores through a deterministic predicate over the reply text — "
                    "exact string, parsed JSON, compiled AST. No model grades any model, because "
                    "that would make the gate inherit the unreliability it exists to detect."),
        "overall": {"passed": overall_pass, "total": overall_total,
                    # A rate is a claim about the MODEL. If the suite never reached it,
                    # there is no rate -- null, not 0.0, which reads as "answered wrong".
                    "rate": (round(overall_pass / overall_total, 4)
                             if overall_total and _health["scoreable"] else None),
                    "scoreable": _health["scoreable"]},
        "harness_health": _health,
        "machine_state": _machine,
        "per_axis": per_axis,
        "measurement_weight": measurement_weight(per_item),
        "per_item": per_item,
    }
    out = Path(args.out or (REPO / f"receipts/headless/CAPABILITY_{args.label}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))

    print(f"\n=== CAPABILITY {args.label} ===")
    if not _health["scoreable"]:
        print(f"  {_health['verdict']}")
    print(f"  overall {overall_pass}/{overall_total} = {doc['overall']['rate']}")
    print(f"  arm     {_ident['chat_template_arm']}")
    if not _ident_ok:
        print(f"  UNIDENTIFIED: {_ident_why} -- this score names a LABEL, not a body")
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
