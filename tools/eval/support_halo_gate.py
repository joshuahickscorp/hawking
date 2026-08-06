#!/usr/bin/env python3
"""Odyssey G5 support-halo gate — offline-first tournament judge.

Scores a frozen support-halo corpus under frozen scoring rules. Generation
against a live Math-Preserve `.gravity` serve path is optional and deferred
while the GPU is busy; the pure scoring path needs only a completions JSONL.

House rules: no invented scores. Dimensions that cannot be measured are
NOT_MEASURABLE with a reason. Do not touch tools/prometheus/ or the sealed
Math-Preserve artifact from this script.

Usage:
  # Seal hashes / self-check (no model, no Metal):
  python3 tools/eval/support_halo_gate.py self-check

  # Score pre-recorded completions (deterministic offline):
  python3 tools/eval/support_halo_gate.py score-completions \\
      --completions path/to/completions.jsonl \\
      --artifact-sha256 <index_or_file_sha> \\
      --out workspace/campaign/governance/odyssey/program/evaluation/receipts/run.json

  # Compare two sealed reports (tournament rank):
  python3 tools/eval/support_halo_gate.py compare --a a.json --b b.json

  # Live baseline against a serve endpoint (DEFERRED while GPU busy):
  python3 tools/eval/support_halo_gate.py run-baseline \\
      --endpoint http://127.0.0.1:8899 \\
      --model math-preserve \\
      --artifact-sha256 33d40c254eb982d4a495f5f0792a116e9d9810d937f5f3969f4f84742b2364d9 \\
      --out workspace/campaign/governance/odyssey/program/evaluation/SUPPORT_HALO_BASELINE.json

  # Candidate artifact, bound to the server's exact index hash (also deferred):
  python3 tools/eval/support_halo_gate.py run-artifact \\
      --force --artifact /path/to/assembled-artifact \\
      --endpoint http://127.0.0.1:8899 --out /path/to/HALO.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import odyssey_path

EVALUATION_DIR = odyssey_path("evaluation")
RULES_PATH = EVALUATION_DIR / "SUPPORT_HALO_SCORING_RULES.json"
CORPUS_PATH = EVALUATION_DIR / "support_halo_corpus_v0.jsonl"
DIMENSIONS = [
    "technical_language",
    "general_reasoning",
    "coding",
    "retrieval",
    "tools",
    "long_context",
    "self_correction",
]
MIN_MEASURABLE = 4
DEGENERATE_N = 8
REGRESSION_AGGREGATE_EPS = 0.05
REGRESSION_DIMENSION_DROP = 0.15
Z_95 = 1.96
CODE_FENCE = re.compile(r"`{2,}[ \t]*[a-zA-Z0-9_+-]*[ \t]*\n(.*?)`{2,}", re.DOTALL)
INDEX_NAMES = (
    "model.gravity.index.json",
    "model.activation_aware.index.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_index(artifact: Path) -> Path:
    artifact = Path(artifact).expanduser().resolve()
    present = [artifact / name for name in INDEX_NAMES if (artifact / name).is_file()]
    if len(present) != 1:
        raise ValueError(
            f"{artifact}: expected exactly one of {INDEX_NAMES}, found "
            f"{[path.name for path in present]}"
        )
    return present[0]


def validate_runtime_attestation(
    payload: dict,
    *,
    expected_index_sha256: str,
    requested_model: str | None = None,
) -> dict:
    """Refuse an endpoint that cannot prove the exact base artifact and no fallback."""
    if payload.get("status") != "ok":
        raise ValueError("serve health attestation is not healthy")
    if payload.get("runtime") != "base" or payload.get("base_runtime") is not True:
        raise ValueError("serve health attestation is not the base runtime")
    if payload.get("fallback_present") is not False:
        raise ValueError("serve health attestation does not prove fallback_present=false")
    if payload.get("artifact_index_sha256") != expected_index_sha256:
        raise ValueError("serve health artifact index SHA-256 does not match the candidate")
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("serve health attestation has no model_id")
    if requested_model is not None and requested_model != model_id:
        raise ValueError(
            f"requested model {requested_model!r} differs from served model {model_id!r}"
        )
    return {
        "runtime": "base",
        "base_runtime": True,
        "fallback_present": False,
        "artifact_index_sha256": expected_index_sha256,
        "model_id": model_id,
        "architecture": payload.get("architecture"),
    }


def fetch_runtime_attestation(
    endpoint: str,
    *,
    expected_index_sha256: str,
    requested_model: str | None = None,
    timeout: float = 30.0,
) -> dict:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/healthz", timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    return validate_runtime_attestation(
        payload,
        expected_index_sha256=expected_index_sha256,
        requested_model=requested_model,
    )


def wilson(passes: int, n: int, z: float = Z_95):
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_rules():
    return json.loads(RULES_PATH.read_text())


def load_corpus():
    tasks = []
    for i, line in enumerate(CORPUS_PATH.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"corpus line {i}: {e}")
    return tasks


def expand_needle_prompt(task: dict) -> str:
    template = task["prompt_template"]
    needle = task["needle"]
    haystack_chars = int(task.get("haystack_chars", 3500))
    frac = float(task.get("needle_offset_frac", 0.25))
    frac = max(0.0, min(1.0, frac))
    filler = "lorem context pad. "
    hay = ""
    while len(hay) < haystack_chars:
        hay += filler
    hay = hay[:haystack_chars]
    insert_at = min(int(len(hay) * frac), len(hay))
    hay = hay[:insert_at] + f" {needle} " + hay[insert_at:]
    return template.replace("{haystack}", hay)


def task_prompt(task: dict) -> str:
    if task.get("oracle") == "needle":
        return expand_needle_prompt(task)
    return task["prompt"]


def extract_code(text: str) -> str:
    m = CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("`")]
    return "\n".join(lines).strip().strip("`").strip()


def has_degenerate_repetition(text: str, n: int = DEGENERATE_N) -> bool:
    tokens = text.split()
    if len(tokens) < n:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= n:
                return True
        else:
            run = 1
    return False


def score_tool_json(task: dict, completion: str):
    raw = extract_code(completion) if "```" in completion else completion.strip()
    # Prefer first JSON object substring if prose wraps it.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return False, "json parse fail"
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            return False, f"json: {e}"
    if not isinstance(obj, dict):
        return False, "not an object"
    name = obj.get("name", "")
    if name != task.get("tool_name"):
        return False, f"name {name!r} != {task.get('tool_name')!r}"
    args = obj.get("arguments")
    if not isinstance(args, dict):
        return False, "arguments not object"
    for k in task.get("required_argument_keys", []):
        if k not in args:
            return False, f"missing arg {k}"
    for k, want in task.get("argument_equals", {}).items():
        got = args.get(k)
        if got != want and not (
            isinstance(want, (int, float))
            and isinstance(got, str)
            and got.replace(".", "", 1).isdigit()
            and float(got) == float(want)
        ) and not (
            isinstance(got, (int, float))
            and isinstance(want, str)
            and str(got) == want
        ):
            return False, f"arg {k}: {got!r} != {want!r}"
    return True, "ok"


def score_expect_or_exact(task: dict, completion: str):
    text = completion.strip()
    accept = task.get("accept_any_of") or []
    if accept:
        lower = text.lower()
        if any(a.lower() in lower for a in accept):
            return True, "ok"
        return False, "accept_any_of miss"
    oracle = task.get("oracle")
    if oracle in ("exact", "needle"):
        want = task.get("exact") or task.get("needle_answer") or ""
        first = text.splitlines()[0].strip() if text else ""
        tokens = first.split()
        ok = (
            first.lower() == want.lower()
            or text.lower() == want.lower()
            or any(t.lower() == want.lower() for t in tokens)
        )
        return (True, "ok") if ok else (False, f"expected exact {want!r}")
    if oracle == "expect_all":
        ok = all(e in text for e in task.get("expect", []))
        return (True, "ok") if ok else (False, "expect_all miss")
    return False, f"unknown oracle {oracle}"


def run_python_execution(code: str, test: str, timeout: float = 5.0):
    program = code + "\n\n" + test + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0:
        return True, "ok"
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (err[-1] if err else "nonzero exit")


def score_task(task: dict, completion: str, coding_timeout: float = 5.0):
    oracle = task.get("oracle")
    if oracle == "execution":
        code = extract_code(completion)
        return run_python_execution(code, task.get("test", "assert False"), coding_timeout)
    if oracle == "tool_json":
        return score_tool_json(task, completion)
    return score_expect_or_exact(task, completion)


def load_completions_jsonl(path: Path) -> dict:
    """Map task id -> completion text. Lines: {\"id\":..., \"completion\":...}."""
    out = {}
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"completions line {i}: {e}")
        tid = obj.get("id")
        if not tid:
            raise SystemExit(f"completions line {i}: missing id")
        out[tid] = obj.get("completion", "")
    return out


def build_report(
    tasks,
    completions: dict,
    *,
    artifact_sha256: str,
    rules_sha: str,
    corpus_sha: str,
    seed: int = 0,
    no_hidden_fallback: bool = True,
    baseline_rules_sha: str | None = None,
    baseline_corpus_sha: str | None = None,
):
    task_scores = []
    total_chars = 0
    empty = 0
    measured = 0
    dqs = []

    for task in tasks:
        tid = task["id"]
        if tid not in completions:
            task_scores.append(
                {
                    "id": tid,
                    "dimension": task["dimension"],
                    "passed": False,
                    "reason": "missing completion",
                }
            )
            continue
        comp = completions[tid]
        measured += 1
        total_chars += len(comp)
        if not comp.strip():
            empty += 1
        if has_degenerate_repetition(comp):
            dqs.append(
                {
                    "code": "DEGENERATE_REPETITION",
                    "detail": f"task {tid} has >= {DEGENERATE_N} identical consecutive tokens",
                }
            )
        passed, reason = score_task(task, comp)
        task_scores.append(
            {
                "id": tid,
                "dimension": task["dimension"],
                "passed": bool(passed),
                "reason": reason,
            }
        )

    by_dim = {d: [0, 0] for d in DIMENSIONS}
    for s in task_scores:
        d = s["dimension"]
        if d not in by_dim:
            by_dim[d] = [0, 0]
        by_dim[d][1] += 1
        if s["passed"]:
            by_dim[d][0] += 1

    dimensions = []
    measurable_rates = []
    for d in DIMENSIONS:
        passes, total = by_dim.get(d, [0, 0])
        if total == 0:
            dimensions.append(
                {
                    "status": "NOT_MEASURABLE",
                    "dimension": d,
                    "reason": "no tasks measured for this dimension",
                }
            )
        else:
            lo, hi = wilson(passes, total)
            rate = passes / total
            measurable_rates.append(rate)
            dimensions.append(
                {
                    "status": "MEASURED",
                    "dimension": d,
                    "passes": passes,
                    "total": total,
                    "pass_rate": rate,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

    if len(measurable_rates) < MIN_MEASURABLE:
        aggregate = {
            "status": "NOT_MEASURABLE",
            "reason": f"only {len(measurable_rates)} measurable dimensions; need >= {MIN_MEASURABLE}",
        }
    else:
        aggregate = {
            "status": "MEASURED",
            "score": sum(measurable_rates) / len(measurable_rates),
            "measurable_dimensions": len(measurable_rates),
        }

    if not no_hidden_fallback:
        dqs.append(
            {
                "code": "HIDDEN_FALLBACK",
                "detail": "receipt did not assert no_hidden_fallback=true",
            }
        )
    if not artifact_sha256:
        dqs.append({"code": "ARTIFACT_MISMATCH", "detail": "artifact_sha256 is empty"})
    if measured and empty / measured > 0.5:
        dqs.append(
            {
                "code": "EMPTY_GENERATION",
                "detail": f"{empty}/{measured} completions empty",
            }
        )
    if baseline_rules_sha and rules_sha != baseline_rules_sha:
        dqs.append(
            {
                "code": "SCORING_RULES_DRIFT",
                "detail": f"rules {rules_sha} != baseline {baseline_rules_sha}",
            }
        )
    if baseline_corpus_sha and corpus_sha != baseline_corpus_sha:
        dqs.append(
            {
                "code": "CORPUS_MUTATION",
                "detail": f"corpus {corpus_sha} != baseline {baseline_corpus_sha}",
            }
        )

    # de-dupe by code
    seen = set()
    uniq = []
    for d in dqs:
        if d["code"] not in seen:
            seen.add(d["code"])
            uniq.append(d)

    return {
        "schema": "hawking.odyssey.support_halo.report.v0",
        "rules_sha256": rules_sha,
        "corpus_sha256": corpus_sha,
        "artifact_sha256": artifact_sha256,
        "seed": seed,
        "no_hidden_fallback": no_hidden_fallback,
        "task_scores": task_scores,
        "dimensions": dimensions,
        "aggregate": aggregate,
        "disqualifications": uniq,
        "total_completion_chars": total_chars,
    }


def dim_rate(report, name):
    for d in report["dimensions"]:
        if d.get("dimension") == name and d.get("status") == "MEASURED":
            return d["pass_rate"]
    return None


def min_measurable(report):
    rates = [
        d["pass_rate"]
        for d in report["dimensions"]
        if d.get("status") == "MEASURED"
    ]
    return min(rates) if rates else None


def compare_reports(a, b) -> int:
    """Return -1 if a better, 1 if b better, 0 if tied after all breaks."""
    a_dq = bool(a.get("disqualifications"))
    b_dq = bool(b.get("disqualifications"))
    if a_dq and not b_dq:
        return 1
    if b_dq and not a_dq:
        return -1
    if a_dq and b_dq:
        return -1 if a["artifact_sha256"] < b["artifact_sha256"] else (
            1 if a["artifact_sha256"] > b["artifact_sha256"] else 0
        )

    aa, ba = a.get("aggregate", {}), b.get("aggregate", {})
    if aa.get("status") == "MEASURED" and ba.get("status") == "MEASURED":
        if abs(aa["score"] - ba["score"]) > 1e-12:
            return -1 if aa["score"] > ba["score"] else 1
    elif aa.get("status") == "MEASURED":
        return -1
    elif ba.get("status") == "MEASURED":
        return 1

    ma, mb = min_measurable(a), min_measurable(b)
    if ma is not None and mb is not None and abs(ma - mb) > 1e-12:
        return -1 if ma > mb else 1
    if ma is not None and mb is None:
        return -1
    if mb is not None and ma is None:
        return 1

    ca, cb = dim_rate(a, "coding"), dim_rate(b, "coding")
    if ca is not None and cb is not None and abs(ca - cb) > 1e-12:
        return -1 if ca > cb else 1
    if ca is not None and cb is None:
        return -1
    if cb is not None and ca is None:
        return 1

    if a.get("total_completion_chars", 0) != b.get("total_completion_chars", 0):
        return -1 if a["total_completion_chars"] < b["total_completion_chars"] else 1

    if a["artifact_sha256"] < b["artifact_sha256"]:
        return -1
    if a["artifact_sha256"] > b["artifact_sha256"]:
        return 1
    return 0


def regression_vs_baseline(candidate, baseline):
    agg_reg = False
    dim_regs = []
    ca, ba = candidate.get("aggregate", {}), baseline.get("aggregate", {})
    if ca.get("status") == "MEASURED" and ba.get("status") == "MEASURED":
        if ca["score"] < ba["score"] - REGRESSION_AGGREGATE_EPS:
            agg_reg = True
    for dim in DIMENSIONS:
        c = next((d for d in candidate["dimensions"] if d.get("dimension") == dim), None)
        b = next((d for d in baseline["dimensions"] if d.get("dimension") == dim), None)
        if (
            c
            and b
            and c.get("status") == "MEASURED"
            and b.get("status") == "MEASURED"
        ):
            drop = b["pass_rate"] - c["pass_rate"]
            non_overlap = c["ci_high"] < b["ci_low"] or b["ci_high"] < c["ci_low"]
            if drop > REGRESSION_DIMENSION_DROP and non_overlap:
                dim_regs.append(dim)
    blocks = agg_reg or bool(dim_regs) or bool(candidate.get("disqualifications"))
    return {
        "aggregate_regression": agg_reg,
        "dimension_regressions": dim_regs,
        "blocks_t7_winner": blocks,
    }


def call_endpoint(endpoint, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def cmd_self_check(_args):
    rules = load_rules()
    tasks = load_corpus()
    rules_sha = sha256_file(RULES_PATH)
    corpus_sha = sha256_file(CORPUS_PATH)
    dims_in_corpus = {t["dimension"] for t in tasks}
    missing = [d for d in DIMENSIONS if d not in dims_in_corpus]
    # Offline perfect-completion smoke for non-execution tasks + synthetic coding.
    completions = {}
    for t in tasks:
        oid = t["oracle"]
        if oid == "exact":
            completions[t["id"]] = t["exact"]
        elif oid == "needle":
            completions[t["id"]] = t["needle_answer"]
        elif oid == "expect_all":
            if t.get("accept_any_of"):
                completions[t["id"]] = t["accept_any_of"][0]
            else:
                completions[t["id"]] = " ".join(t.get("expect", []))
        elif oid == "tool_json":
            completions[t["id"]] = json.dumps(
                {
                    "name": t["tool_name"],
                    "arguments": t.get("argument_equals", {}),
                }
            )
        elif oid == "execution":
            # Minimal correct stubs for the four coding / self-correction tasks.
            stubs = {
                "cd01_add": "def add(a, b):\n    return a + b\n",
                "cd02_is_prime": (
                    "def is_prime(n):\n"
                    "    if n < 2: return False\n"
                    "    for i in range(2, int(n**0.5)+1):\n"
                    "        if n % i == 0: return False\n"
                    "    return True\n"
                ),
                "cd03_reverse_words": (
                    "def reverse_words(s):\n"
                    "    return ' '.join(s.split()[::-1])\n"
                ),
                "cd04_gcd": (
                    "def gcd(a, b):\n"
                    "    while b: a, b = b, a % b\n"
                    "    return a\n"
                ),
                "sc02_fix_code": (
                    "def clamp(x, lo, hi):\n"
                    "    return max(lo, min(hi, x))\n"
                ),
            }
            completions[t["id"]] = f"```python\n{stubs.get(t['id'], 'pass')}\n```"
        else:
            completions[t["id"]] = ""

    report = build_report(
        tasks,
        completions,
        artifact_sha256="self-check-artifact",
        rules_sha=rules_sha,
        corpus_sha=corpus_sha,
        seed=0,
        no_hidden_fallback=True,
    )
    # Perfect synthetic should measure all 7 dims and pass aggregate.
    ok = (
        report["aggregate"]["status"] == "MEASURED"
        and not report["disqualifications"]
        and not missing
        and rules.get("status") == "FROZEN_PRE_ODYSSEY"
    )
    # Determinism: second build identical
    report2 = build_report(
        tasks,
        completions,
        artifact_sha256="self-check-artifact",
        rules_sha=rules_sha,
        corpus_sha=corpus_sha,
        seed=0,
        no_hidden_fallback=True,
    )
    det = json.dumps(report, sort_keys=True) == json.dumps(report2, sort_keys=True)

    out = OrderedDict(
        [
            ("ok", bool(ok and det)),
            ("rules_sha256", rules_sha),
            ("corpus_sha256", corpus_sha),
            ("n_tasks", len(tasks)),
            ("dimensions_in_corpus", sorted(dims_in_corpus)),
            ("missing_dimensions", missing),
            ("aggregate", report["aggregate"]),
            ("disqualifications", report["disqualifications"]),
            ("deterministic_rescore", det),
            ("schema", rules.get("schema")),
        ]
    )
    print(json.dumps(out, indent=2))
    # Persist sealed hashes next to the corpus for baseline binding.
    seal = {
        "schema": "hawking.odyssey.support_halo.seal.v0",
        "rules_path": str(RULES_PATH.relative_to(ROOT)),
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
        "rules_sha256": rules_sha,
        "corpus_sha256": corpus_sha,
        "n_tasks": len(tasks),
        "status": "FROZEN_PRE_ODYSSEY",
        "baseline_status": "NOT_RUN",
        "note": "Baseline generation on Math-Preserve is deferred (GPU/runtime measurement). Rules and corpus are frozen now so T7 cannot rewrite the judge after seeing candidates.",
    }
    seal_path = EVALUATION_DIR / "SUPPORT_HALO_SEAL.json"
    write_json(seal_path, seal)
    return 0 if out["ok"] else 1


def cmd_score_completions(args):
    tasks = load_corpus()
    completions = load_completions_jsonl(Path(args.completions))
    rules_sha = sha256_file(RULES_PATH)
    corpus_sha = sha256_file(CORPUS_PATH)
    report = build_report(
        tasks,
        completions,
        artifact_sha256=args.artifact_sha256,
        rules_sha=rules_sha,
        corpus_sha=corpus_sha,
        seed=args.seed,
        no_hidden_fallback=not args.allow_hidden_fallback,
        baseline_rules_sha=args.baseline_rules_sha256,
        baseline_corpus_sha=args.baseline_corpus_sha256,
    )
    if args.out:
        write_json(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["disqualifications"] else 2


def cmd_compare(args):
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    cmp = compare_reports(a, b)
    winner = "a" if cmp < 0 else ("b" if cmp > 0 else "tie")
    out = {"winner": winner, "compare": cmp}
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        out["a_vs_baseline"] = regression_vs_baseline(a, base)
        out["b_vs_baseline"] = regression_vs_baseline(b, base)
        # T7 winner must not be a regression vs baseline
        if winner == "a" and out["a_vs_baseline"]["blocks_t7_winner"]:
            out["t7_eligible"] = False
            out["t7_block_reason"] = "a regresses vs baseline"
        elif winner == "b" and out["b_vs_baseline"]["blocks_t7_winner"]:
            out["t7_eligible"] = False
            out["t7_block_reason"] = "b regresses vs baseline"
        elif winner == "tie":
            out["t7_eligible"] = False
            out["t7_block_reason"] = "tie"
        else:
            out["t7_eligible"] = True
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_run_baseline(args):
    """Live generation. Requires endpoint. Deferred when GPU is busy — not invoked by default."""
    if not args.force:
        print(
            "REFUSED: live baseline needs a serve endpoint and model generation.\n"
            "GPU/runtime measurement may be occupying the device. Re-run with\n"
            "  --force --endpoint ... --model ... --artifact-sha256 ...\n"
            "only when the Math-Preserve serve path is free.\n"
            "Rules and corpus are already frozen; this step only seals measured scores.",
            file=sys.stderr,
        )
        return 3

    tasks = load_corpus()
    rules_sha = sha256_file(RULES_PATH)
    corpus_sha = sha256_file(CORPUS_PATH)
    completions = {}
    gen_meta = []
    for t in tasks:
        prompt = task_prompt(t)
        max_tok = int(t.get("max_new_tokens", 128))
        content, dt = call_endpoint(
            args.endpoint, args.model, prompt, max_tok, args.timeout
        )
        completions[t["id"]] = content
        gen_meta.append({"id": t["id"], "gen_s": round(dt, 3)})

    report = build_report(
        tasks,
        completions,
        artifact_sha256=args.artifact_sha256,
        rules_sha=rules_sha,
        corpus_sha=corpus_sha,
        seed=args.seed,
        no_hidden_fallback=True,
    )
    report["generation"] = {
        "endpoint": args.endpoint,
        "model": args.model,
        "tasks": gen_meta,
        "mode": "live_baseline",
    }
    report["substrate"] = {
        "label": "GLM-5.2-H0.98-Math-Preserve.gravity",
        "artifact_sha256": args.artifact_sha256,
    }
    out_path = Path(args.out or EVALUATION_DIR / "SUPPORT_HALO_BASELINE.json")
    write_json(out_path, report)
    # Update seal baseline status
    seal_path = EVALUATION_DIR / "SUPPORT_HALO_SEAL.json"
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text())
        seal["baseline_status"] = "SEALED"
        seal["baseline_path"] = str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path)
        write_json(seal_path, seal)
    print(json.dumps({"wrote": str(out_path), "aggregate": report["aggregate"]}, indent=2))
    return 0 if not report["disqualifications"] else 2


def cmd_run_artifact(args):
    """Live candidate score with exact pre/post runtime identity attestations."""
    if not args.force:
        print(
            "REFUSED: run-artifact executes 26 whole-model generation tasks. "
            "Re-run with --force only in a valid heavy window.",
            file=sys.stderr,
        )
        return 3

    index = artifact_index(Path(args.artifact))
    index_sha = sha256_file(index)
    before = fetch_runtime_attestation(
        args.endpoint,
        expected_index_sha256=index_sha,
        requested_model=args.model,
    )
    model = before["model_id"]
    tasks = load_corpus()
    rules_sha = sha256_file(RULES_PATH)
    corpus_sha = sha256_file(CORPUS_PATH)
    seal_path = EVALUATION_DIR / "SUPPORT_HALO_SEAL.json"
    seal = json.loads(seal_path.read_text())

    completions = {}
    for task in tasks:
        content, _elapsed = call_endpoint(
            args.endpoint,
            model,
            task_prompt(task),
            int(task.get("max_new_tokens", 128)),
            args.timeout,
        )
        completions[task["id"]] = content

    after = fetch_runtime_attestation(
        args.endpoint,
        expected_index_sha256=index_sha,
        requested_model=model,
    )
    report = build_report(
        tasks,
        completions,
        artifact_sha256=index_sha,
        rules_sha=rules_sha,
        corpus_sha=corpus_sha,
        seed=args.seed,
        no_hidden_fallback=(
            before["fallback_present"] is False
            and after["fallback_present"] is False
        ),
        baseline_rules_sha=seal.get("rules_sha256"),
        baseline_corpus_sha=seal.get("corpus_sha256"),
    )
    report["generation"] = {
        "mode": "live_artifact",
        "model": model,
        "temperature": 0,
        "pre_attestation": before,
        "post_attestation": after,
    }
    report["substrate"] = {
        "label": model,
        "address_kind": "model_index_sha256",
        "artifact_index": index.name,
        "artifact_sha256": index_sha,
    }

    out_path = Path(args.out)
    completions_path = out_path.with_name(
        f"{out_path.stem}.completions.jsonl"
    )
    completions_text = "".join(
        json.dumps(
            {"id": task["id"], "completion": completions[task["id"]]},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for task in tasks
    )
    completions_path.parent.mkdir(parents=True, exist_ok=True)
    completions_path.write_text(completions_text)
    report["completion_evidence"] = {
        "sha256": sha256_bytes(completions_text.encode()),
        "tasks": len(tasks),
    }
    write_json(out_path, report)
    print(
        json.dumps(
            {
                "wrote": str(out_path),
                "completions": str(completions_path),
                "aggregate": report["aggregate"],
                "disqualifications": report["disqualifications"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not report["disqualifications"] else 2


def cmd_print_baseline_command(_args):
    """Print the deferred baseline command without running it."""
    sha = "33d40c254eb982d4a495f5f0792a116e9d9810d937f5f3969f4f84742b2364d9"
    cmd = (
        "python3 tools/eval/support_halo_gate.py run-baseline \\\n"
        "  --force \\\n"
        "  --endpoint http://127.0.0.1:8899 \\\n"
        "  --model math-preserve \\\n"
        f"  --artifact-sha256 {sha} \\\n"
        "  --out workspace/campaign/governance/odyssey/program/evaluation/SUPPORT_HALO_BASELINE.json\n"
        "\n"
        "# Prerequisite (live serve path; Math-Preserve base runtime, default-off flag):\n"
        "#   cargo run --release -p hawking -- serve \\\n"
        "#     --gravity \"$HOME/Library/Application Support/Hawking/Models/GLM-5.2/"
        "b4734de4facf877f85769a911abafc5283eab3d9/GLM-5.2-H0.98-Math-Preserve.gravity\" \\\n"
        "#     --addr 127.0.0.1:8899\n"
        "# Equivalent env form: HAWKING_GRAVITY=<dir> cargo run --release -p hawking -- serve "
        "--addr 127.0.0.1:8899\n"
        "# Base runtime is ~0.4 tok/s warm; request timeout defaults to 3600s "
        "(HAWKING_GRAVITY_REQUEST_TIMEOUT_SECS).\n"
    )
    print(cmd)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("self-check", help="Frozen rules/corpus integrity + offline rescore")
    s.set_defaults(func=cmd_self_check)

    s = sub.add_parser("score-completions", help="Score a completions JSONL offline")
    s.add_argument("--completions", required=True)
    s.add_argument("--artifact-sha256", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.add_argument("--allow-hidden-fallback", action="store_true")
    s.add_argument("--baseline-rules-sha256")
    s.add_argument("--baseline-corpus-sha256")
    s.set_defaults(func=cmd_score_completions)

    s = sub.add_parser("compare", help="Tournament compare two reports")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--baseline", help="Optional baseline report for regression gate")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("run-baseline", help="Live baseline (refused without --force)")
    s.add_argument("--force", action="store_true")
    s.add_argument("--endpoint", default="http://127.0.0.1:8899")
    s.add_argument("--model", default="math-preserve")
    s.add_argument("--artifact-sha256", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_run_baseline)

    s = sub.add_parser(
        "run-artifact",
        help="Live candidate gate with exact server/artifact attestation (refused without --force)",
    )
    s.add_argument("--force", action="store_true")
    s.add_argument("--artifact", required=True)
    s.add_argument("--endpoint", default="http://127.0.0.1:8899")
    s.add_argument("--model", help="optional expected model id; defaults to attested model_id")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=float, default=3600.0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_run_artifact)

    s = sub.add_parser(
        "print-baseline-command",
        help="Print the deferred Math-Preserve baseline command",
    )
    s.set_defaults(func=cmd_print_baseline_command)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
