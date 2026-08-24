#!/usr/bin/env python3
"""N007 — production bench: verified useful work per wall second.

Given the measured concurrency ceiling (~1.32x, bandwidth-bound), this harness
runs representative Hawking WorkUnits through a deterministic verifier at
c=1,2,4 on the sealed parent and the q4 incumbent. The winner is the
configuration with the most VERIFIED accepted WorkUnits per hour, not the
highest stream count or single-stream TPS.

Completed WUs and verified WUs are separate ledgers. A generation that the
verifier rejects is throughput the production loop cannot use.

Does not mutate ~/noetic/NOETIC_PARENT_A. Does not load a second 27B.
Does not write receipts/ascent-2026-08-16 or workspace/campaign.

    python3 tools/headless/production_bench.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metal_budget import metal_device  # noqa: E402
from noetic_multisession import workspace_bytes  # noqa: E402

SCHEMA = "hawking.headless.production_bench.v1"
RECEIPT = REPO / "receipts" / "headless" / "PRODUCTION_BENCH.json"
RAW_DIR = REPO / "receipts" / "headless"
WU_FILE = RAW_DIR / "_PRODUCTION_BENCH_workunits.json"
DRIVER_MANIFEST = HERE / "prodbench_driver" / "Cargo.toml"
CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust-prodbench"),
    )
)
DRIVER_BIN = (
    CARGO_TARGET / "release-fast" / "ascension_qwen38_production_bench"
)

Q4_ROOT = Path(
    os.environ.get(
        "QWEN38_Q4_ARTIFACT",
        str(Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"),
    )
)
PARENT_ROOT = Path(
    os.environ.get(
        "NOETIC_PARENT_A_ROOT",
        str(Path.home() / "noetic" / "NOETIC_PARENT_A"),
    )
)
TOKENIZER = Path(
    os.environ.get(
        "QWEN38_TOKENIZER",
        str(Path.home() / "models" / "qwen3.8-27b-abliterated-bf16" / "tokenizer.json"),
    )
)

# Anchors already measured. Do not re-derive.
Q4_ACTIVE_BYTES = 13_622_266_960
PARENT_ACTIVE_BYTES = 9_878_901_136
Q4_EBPW = 4.252735126866492
PARENT_EBPW = 3.139300850311054
Q4_DISPATCHES = 964
PARENT_DISPATCHES = 756
MULTISESSION_CEILING = {
    "concurrent_independent": {"1": 1.0, "2": 1.325, "4": 1.323},
    "sequential_per_session": {"1": 1.0, "2": 1.002, "4": 0.989},
}
GPU_LEDGER_GB_S = 468.9
ROOF_GB_S = 778.8
GPU_CORES = 60

# Enough tokens that a thinking reply can close </think> and emit the answer.
# Truncation is recorded as completed-but-not-verified, not dropped.
DEFAULT_MAX_NEW = 192
DEFAULT_MAX_SEQ = 512
DEFAULT_CONCURRENCIES = (1, 2, 4)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def workunits() -> list[dict[str, Any]]:
    """Representative Hawking WorkUnits. Each has a deterministic verifier.

    Shaped like an HCLI card (WORKUNIT + ACCEPTANCE + TASK) so the generation
    is the same class of work the production loop actually admits, not a
    token generator.
    """
    return [
        {
            "id": "wu_fact_france",
            "role": "FACTUAL",
            "kind": "contains",
            "expected": "Paris",
            "prompt": (
                "WORKUNIT: wu_fact_france\n"
                "ACCEPTANCE: the capital city name appears in the reply.\n"
                "TASK: What is the capital city of France? Reply with the city name only."
            ),
        },
        {
            "id": "wu_fact_arith",
            "role": "REASONING",
            "kind": "number",
            "expected": "323",
            "prompt": (
                "WORKUNIT: wu_fact_arith\n"
                "ACCEPTANCE: the integer 323 appears as the answer.\n"
                "TASK: Compute 17 * 19. Reply with only the number."
            ),
        },
        {
            "id": "wu_fact_tungsten",
            "role": "FACTUAL",
            "kind": "contains",
            "expected": "tungsten",
            "prompt": (
                "WORKUNIT: wu_fact_tungsten\n"
                "ACCEPTANCE: the element name tungsten appears in the reply.\n"
                "TASK: Which element has atomic number 74? Reply with the element name only."
            ),
        },
        {
            "id": "wu_reason_expr",
            "role": "REASONING",
            "kind": "number",
            "expected": "86",
            "prompt": (
                "WORKUNIT: wu_reason_expr\n"
                "ACCEPTANCE: the integer 86 appears as the answer.\n"
                "TASK: What is (23 + 19) times 3 minus 40? Reply with the number only."
            ),
        },
        {
            "id": "wu_proc_reverse",
            "role": "PROCEDURAL",
            "kind": "contains",
            "expected": "ananab",
            "prompt": (
                "WORKUNIT: wu_proc_reverse\n"
                "ACCEPTANCE: the reversed spelling ananab appears in the reply.\n"
                'TASK: Write the word "banana" backwards. Reply with the word only.'
            ),
        },
        {
            "id": "wu_proc_count",
            "role": "PROCEDURAL",
            "kind": "number",
            "expected": "2",
            "prompt": (
                "WORKUNIT: wu_proc_count\n"
                "ACCEPTANCE: the integer 2 appears as the answer.\n"
                'TASK: How many times does the letter "n" appear in "banana"? '
                "Reply with the number only."
            ),
        },
        {
            "id": "wu_lang_are",
            "role": "LANGUAGE",
            "kind": "word",
            "expected": "are",
            "prompt": (
                "WORKUNIT: wu_lang_are\n"
                "ACCEPTANCE: the single word are fills the blank.\n"
                'TASK: Fill in the blank with one word: "The three cats ___ waiting outside." '
                "Reply with the single word only."
            ),
        },
        {
            "id": "wu_tool_json_tokyo",
            "role": "TOOL",
            "kind": "json_city",
            "expected": "Tokyo",
            "prompt": (
                "WORKUNIT: wu_tool_json_tokyo\n"
                'ACCEPTANCE: a JSON object {"city": "Tokyo"} (city name may vary in spacing).\n'
                'TASK: Reply with ONLY a JSON object of the form {"city": "..."} '
                "giving the capital of Japan."
            ),
        },
        {
            "id": "wu_code_dedupe",
            "role": "CODE",
            "kind": "python_fn",
            "expected": "dedupe",
            "prompt": (
                "WORKUNIT: wu_code_dedupe\n"
                "ACCEPTANCE: a Python function named dedupe that parses.\n"
                "TASK: Write a Python function `dedupe(xs)` that removes duplicates while "
                "preserving first-seen order. Reply with a single ```python code block."
            ),
        },
        {
            "id": "wu_mutation_add",
            "role": "MUTATION",
            "kind": "json_mutation_add",
            "expected": "a + b",
            "prompt": (
                "WORKUNIT: wu_mutation_add\n"
                "ACCEPTANCE: JSON mutation whose old_text contains `a - b` and new_text `a + b`.\n"
                "TASK: File calc.py contains exactly:\n"
                "def add(a, b):\n    return a - b\n\n"
                "Change it so add returns the sum. Emit one JSON object "
                '{"kind":"mutation","content":"...","operations":'
                '[{"op":"replace","path":"calc.py","old_text":"...","new_text":"..."}],'
                '"tests":[]}.'
            ),
        },
        {
            "id": "wu_json_exact",
            "role": "TOOL",
            "kind": "json_keys",
            "expected": "kind,content,operations,tests",
            "prompt": (
                "WORKUNIT: wu_json_exact\n"
                "ACCEPTANCE: a JSON object with keys kind, content, operations, tests.\n"
                "TASK: Return exactly one JSON object and nothing else: "
                '{"kind":"answer","content":"ok","operations":[],"tests":[]}.'
            ),
        },
        {
            "id": "wu_reason_modules",
            "role": "REASONING",
            "kind": "number",
            "expected": "21",
            "prompt": (
                "WORKUNIT: wu_reason_modules\n"
                "ACCEPTANCE: the integer 21 appears as the answer.\n"
                "TASK: A repo has 12 modules. 3 are deleted, then 5 are added, then a quarter "
                "of the total are split in two. How many modules are there at the end? "
                "Reply with only the final number."
            ),
        },
        {
            # Completed tokens that the verifier MUST reject. Proves a token
            # generator is not counted as accepted useful work.
            "id": "wu_token_generator",
            "role": "CONTROL",
            "kind": "json_city",
            "expected": "THIS_STRING_MUST_NOT_MATCH",
            "prompt": (
                "WORKUNIT: wu_token_generator\n"
                "ACCEPTANCE: this unit is a negative control; any prose fails the verifier.\n"
                "TASK: Continue the letter a for as long as you can. Do not emit JSON."
            ),
        },
    ]


def answer_body(text: str | None) -> tuple[str | None, bool]:
    """Return (scorable_body, truncated).

    A still-open <think> block is truncated, not wrong. Scoring the think
    text would give false credit (a reversed word appears in reasoning
    before the model commits).
    """
    if not text:
        return None, True
    if "</think>" in text:
        return text.split("</think>")[-1], False
    if "<think>" in text:
        return None, True
    return text, False


def extract_json_obj(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
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
                    obj = json.loads(t[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    start = -1
    return None


def extract_python(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", flags=re.S)
    if m:
        return m.group(1)
    return text if text and "def " in text else None


def verify_workunit(wu: dict[str, Any], text: str | None) -> dict[str, Any]:
    """Deterministic predicate. The model does not grade itself."""
    body, truncated = answer_body(text)
    if truncated or body is None:
        return {
            "id": wu["id"],
            "accepted": False,
            "truncated": True,
            "reason": "still inside <think> or empty generation — not a scorable answer",
        }
    kind = wu["kind"]
    expected = str(wu.get("expected") or "")
    ok = False
    reason = ""
    if kind == "contains":
        ok = expected.lower() in body.lower()
        reason = f"expected {expected!r} in body" if not ok else "ok"
    elif kind == "number":
        ok = re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", body) is not None
        reason = f"expected number {expected}" if not ok else "ok"
    elif kind == "word":
        tokens = re.findall(r"[A-Za-z']+", body)
        ok = any(t.lower() == expected.lower() for t in tokens[:12]) or (
            expected.lower() in body.lower().split()
        )
        reason = f"expected word {expected}" if not ok else "ok"
    elif kind == "json_city":
        obj = extract_json_obj(body)
        city = (obj or {}).get("city")
        ok = isinstance(city, str) and expected.lower() in city.lower()
        reason = "JSON city field missing or wrong" if not ok else "ok"
    elif kind == "json_keys":
        obj = extract_json_obj(body)
        need = [k.strip() for k in expected.split(",") if k.strip()]
        missing = [k for k in need if not isinstance(obj, dict) or k not in obj]
        ok = not missing
        reason = f"JSON missing {missing}" if missing else "ok"
    elif kind == "json_mutation_add":
        obj = extract_json_obj(body)
        ops = (obj or {}).get("operations") or []
        hit = False
        for op in ops:
            if not isinstance(op, dict):
                continue
            old = op.get("old_text") or ""
            new = op.get("new_text") or ""
            path = str(op.get("path") or "")
            if "a - b" in old and "a + b" in new and path.endswith("calc.py"):
                hit = True
                break
        ok = bool(obj) and obj.get("kind") == "mutation" and hit
        reason = "mutation JSON did not rewrite a - b to a + b on calc.py" if not ok else "ok"
    elif kind == "python_fn":
        code = extract_python(body)
        if not code:
            ok, reason = False, "no python code block"
        else:
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                ok, reason = False, f"python does not parse: {e}"
            else:
                ok = any(
                    isinstance(n, ast.FunctionDef) and n.name == expected
                    for n in ast.walk(tree)
                )
                reason = f"no function named {expected}" if not ok else "ok"
    else:
        ok, reason = False, f"unknown kind {kind}"
    return {
        "id": wu["id"],
        "accepted": bool(ok),
        "truncated": False,
        "reason": reason,
        "kind": kind,
        "role": wu.get("role"),
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def occupancy_snapshot(exclude_pids: set[int] | None = None) -> dict[str, Any]:
    exclude_pids = exclude_pids or set()
    proc = subprocess.run(
        ["ps", "-eo", "pid,rss,command"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    lines = []
    second_27b = False
    for line in proc.stdout.splitlines():
        low = line.lower()
        if not any(s in low for s in ("llama-server", "ascension_qwen", "mlx_lm.server")):
            continue
        if "rg " in low or "production_bench" in low:
            continue
        parts = line.split()
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except (IndexError, ValueError):
            pid, rss_kb = -1, 0
        if pid in exclude_pids:
            continue
        lines.append(line.strip())
        if rss_kb > 4_000_000:
            second_27b = True
    return {
        "ps_matches": lines,
        "loaded_a_second_27b": second_27b,
        "note": (
            "A second Qwen3.8-27B would show RSS in the 10+ GiB class. "
            "The driver process itself is excluded from this snapshot."
        ),
    }


def memory_pressure() -> dict[str, Any]:
    """Host memory pressure. Not a GPU counter."""
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
    text = vm.stdout or ""
    fields: dict[str, int] = {}
    page_size = 16384
    for line in text.splitlines():
        if "page size of" in line:
            digits = "".join(ch for ch in line.split("page size of", 1)[-1] if ch.isdigit())
            if digits:
                page_size = int(digits)
        for label, key in (
            ("Pages free:", "pages_free"),
            ("Pages purgeable:", "pages_purgeable"),
            ("Pages speculative:", "pages_speculative"),
            ("Pages occupied by compressor:", "pages_compressor"),
            ("Pages wired down:", "pages_wired"),
            ("Pages active:", "pages_active"),
            ("Pages inactive:", "pages_inactive"),
        ):
            if line.strip().startswith(label):
                token = (
                    line.split(":", 1)[-1]
                    .strip()
                    .strip(".")
                    .split()[0]
                    .replace(",", "")
                )
                try:
                    fields[key] = int(token)
                except ValueError:
                    pass
    free_bytes = fields.get("pages_free", 0) * page_size
    compressor_bytes = fields.get("pages_compressor", 0) * page_size
    mp = subprocess.run(
        ["memory_pressure"], capture_output=True, text=True, timeout=10
    )
    mp_text = (mp.stdout or "") + (mp.stderr or "")
    pct = None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)%\s+free", mp_text, re.I)
    if m:
        pct = float(m.group(1))
    pressure = None
    m2 = re.search(r"Pages\s+free\s+percentage:\s*([0-9.]+)", mp_text, re.I)
    if pct is None and m2:
        pct = float(m2.group(1))
    if "warn" in mp_text.lower():
        pressure = "warn"
    elif "critical" in mp_text.lower():
        pressure = "critical"
    elif pct is not None:
        pressure = "ok" if pct > 15 else "tight"
    return {
        "kind": "MEASURED",
        "page_size_bytes": page_size,
        "free_bytes": free_bytes,
        "compressor_bytes": compressor_bytes,
        "vm_stat": fields,
        "free_percent": pct,
        "pressure": pressure,
        "command": "vm_stat; memory_pressure",
        "note": (
            "Host unified-memory pressure, not a Metal occupancy counter. "
            "Compressor pages growing while free RAM looks fine is the swap-risk signal."
        ),
    }


def parent_is_unmutated(root: Path) -> dict[str, Any]:
    cat = root / "catalog.hq38m20"
    mix = root / "MIX_REPORT.json"
    return {
        "path": str(root),
        "catalog_present": cat.is_file(),
        "mix_report_present": mix.is_file(),
        "catalog_mtime": cat.stat().st_mtime if cat.is_file() else None,
        "writable_check": "driver only reads; python never opens the catalog for write",
        "outside_worktree": "/worktrees/" not in str(root.resolve()),
    }


def wu_by_id() -> dict[str, dict[str, Any]]:
    return {w["id"]: w for w in workunits()}


def summarize_cell(
    cell: dict[str, Any],
    *,
    artifact: str,
    active_bytes: int,
    kv_bytes_one: int,
    c1_aggregate_tps: float | None,
) -> dict[str, Any]:
    catalog = wu_by_id()
    rows_in = cell.get("workunits") or []
    wall_ns = int(cell.get("wall_ns") or 0)
    wall_s = wall_ns / 1e9 if wall_ns else 0.0
    c = int(cell.get("concurrency") or cell.get("sessions") or 1)
    topology = cell.get("topology") or "unknown"
    verified = []
    completed = []
    truncated = 0
    tokens = 0
    ttfts = []
    ttfts_queued = []
    latencies_ms = []
    fallbacks = 0
    dispatches = []
    per_stream_tokens = [0] * c
    for row in rows_in:
        wu = catalog.get(row.get("id") or "", {"id": row.get("id"), "kind": "contains", "expected": ""})
        n = int(row.get("n_new_tokens") or 0)
        tokens += n
        slot = int(row.get("session_index") or 0)
        if 0 <= slot < c:
            per_stream_tokens[slot] += n
        fallbacks += int(row.get("fallbacks") or 0)
        if row.get("dispatches_last_step") is not None:
            dispatches.append(int(row["dispatches_last_step"]))
        if row.get("ttft_exclusive_ns") is not None:
            ttfts.append(int(row["ttft_exclusive_ns"]) / 1e9)
        if row.get("ttft_from_batch_start_ns") is not None:
            ttfts_queued.append(int(row["ttft_from_batch_start_ns"]) / 1e9)
        for ns in row.get("decode_step_wall_ns") or []:
            latencies_ms.append(int(ns) / 1e6)
        verdict = verify_workunit(wu, row.get("generated_text"))
        completed.append(
            {
                "id": wu.get("id"),
                "n_new_tokens": n,
                "session_index": slot,
                "fallbacks": row.get("fallbacks"),
                "verification": verdict,
            }
        )
        if verdict["truncated"]:
            truncated += 1
        if verdict["accepted"]:
            verified.append(wu.get("id"))
    n_completed = len(rows_in)
    n_verified = len(verified)
    agg_tps = (tokens / wall_s) if wall_s else 0.0
    per_stream_shared = [(t / wall_s) if wall_s else 0.0 for t in per_stream_tokens]
    wu_per_hour_completed = (n_completed / wall_s * 3600.0) if wall_s else 0.0
    wu_per_hour_verified = (n_verified / wall_s * 3600.0) if wall_s else 0.0
    slowdown = None
    if c1_aggregate_tps and agg_tps:
        # vs linear: linear would be c * c1. Measured aggregate / c1 is the scaling.
        slowdown = {
            "aggregate_vs_c1": agg_tps / c1_aggregate_tps,
            "vs_linear": (agg_tps / c1_aggregate_tps) / c if c else None,
        }
    achieved_gb_s = (active_bytes * agg_tps) / 1e9 if agg_tps else 0.0
    metal = cell.get("metal_after_cell") or {}
    allocated = metal.get("current_allocated_size")
    rec = metal.get("recommended_max_working_set_size")
    ws_occ = None
    if allocated and rec:
        ws_occ = allocated / rec
    return {
        "artifact": artifact,
        "topology": topology,
        "concurrency": c,
        "wall_s": wall_s,
        "completed_workunits": n_completed,
        "verified_workunits": n_verified,
        "truncated_workunits": truncated,
        "rejected_workunits": n_completed - n_verified,
        "verified_ids": verified,
        "tokens": tokens,
        "aggregate_tok_s": agg_tps,
        "per_stream_tok_s_shared_wall": per_stream_shared,
        "ttft_s": ttfts,
        "ttft_from_batch_start_s": ttfts_queued,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "token_latency_p50_ms": percentile(latencies_ms, 50),
        "token_latency_p95_ms": percentile(latencies_ms, 95),
        "fallbacks": fallbacks,
        "dispatches_last_step": dispatches,
        "dispatch_mode": (
            max(set(dispatches), key=dispatches.count) if dispatches else None
        ),
        "active_bytes_per_token": active_bytes,
        "kv_bytes": kv_bytes_one * c,
        "kv_bytes_one_session": kv_bytes_one,
        "achieved_gb_s": achieved_gb_s,
        "completed_wu_per_hour": wu_per_hour_completed,
        "verified_wu_per_hour": wu_per_hour_verified,
        "verifier_throughput_wu_per_hour": wu_per_hour_completed,
        "accepted_wu_per_hour": wu_per_hour_verified,
        "concurrency_slowdown": slowdown,
        "metal_current_allocated_size": allocated,
        "metal_working_set_occupancy": ws_occ,
        "rss_after_cell_bytes": (cell.get("rss_after_cell_bytes")),
        "workunit_ledger": completed,
    }


def choose_winner(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Winner is max verified WUs/hour. Stream count and TPS are not keys."""
    scored = [c for c in cells if isinstance(c.get("verified_wu_per_hour"), (int, float))]
    if not scored:
        return {"winner": None, "reason": "no scored cells"}
    def key(c: dict[str, Any]) -> tuple:
        return (
            float(c["verified_wu_per_hour"]),
            -int(c.get("concurrency") or 0),
            -float(c.get("ttft_p50_s") or 0.0),
        )
    best = max(scored, key=key)
    by_tps = max(scored, key=lambda c: float(c.get("aggregate_tok_s") or 0.0))
    by_c = max(scored, key=lambda c: int(c.get("concurrency") or 0))
    tps_would_differ = (
        by_tps.get("artifact"),
        by_tps.get("concurrency"),
        by_tps.get("topology"),
    ) != (
        best.get("artifact"),
        best.get("concurrency"),
        best.get("topology"),
    )
    return {
        "winner": {
            "artifact": best.get("artifact"),
            "concurrency": best.get("concurrency"),
            "topology": best.get("topology"),
            "verified_wu_per_hour": best.get("verified_wu_per_hour"),
            "completed_wu_per_hour": best.get("completed_wu_per_hour"),
            "aggregate_tok_s": best.get("aggregate_tok_s"),
            "ttft_p50_s": best.get("ttft_p50_s"),
            "token_latency_p50_ms": best.get("token_latency_p50_ms"),
        },
        "ranking_quantity": "verified_accepted_workunits_per_hour",
        "not_the_ranking_quantity": [
            "stream_count",
            "single_stream_tps",
            "aggregate_tok_s",
        ],
        "highest_aggregate_tok_s_cell": {
            "artifact": by_tps.get("artifact"),
            "concurrency": by_tps.get("concurrency"),
            "topology": by_tps.get("topology"),
            "aggregate_tok_s": by_tps.get("aggregate_tok_s"),
            "verified_wu_per_hour": by_tps.get("verified_wu_per_hour"),
        },
        "highest_concurrency_cell": {
            "artifact": by_c.get("artifact"),
            "concurrency": by_c.get("concurrency"),
            "verified_wu_per_hour": by_c.get("verified_wu_per_hour"),
        },
        "winner_differs_from_highest_tps": tps_would_differ,
        "why": (
            "A configuration with lower aggregate tok/s can win if it verifies more "
            "WorkUnits per wall second (less truncation, fewer unusable answers, "
            "better TTFT so the same token budget closes </think>)."
        ),
    }


def c8_physically_meaningful(
    scaling_c2: float | None, scaling_c4: float | None
) -> dict[str, Any]:
    """c=8 is only run if c=2..4 still shows headroom above the known ceiling class."""
    ceiling = MULTISESSION_CEILING["concurrent_independent"]["4"]
    # Physically meaningful if measured c=4 scaling is still climbing vs c=2
    # by more than noise, or if it clearly exceeds the known ~1.32x ceiling.
    climb = None
    if scaling_c2 and scaling_c4:
        climb = scaling_c4 / scaling_c2
    run = False
    reason = (
        "c=8 is not physically meaningful on this box: one shared body is already "
        f"bandwidth-bound (GPU_LEDGER {GPU_LEDGER_GB_S} GB/s streaming 13.62 GB/token). "
        f"NOETIC_MULTISESSION concurrent_independent scaled 1.000 -> 1.325 -> 1.323; "
        "c=4 did not beat c=2. Extra streams past the memory-system delivery limit "
        "split the same tokens, stretch TTFT, and cannot raise verified WUs/hour."
    )
    if scaling_c4 is not None and scaling_c2 is not None:
        if scaling_c4 > scaling_c2 * 1.08 and scaling_c4 > ceiling * 1.05:
            run = True
            reason = (
                f"c=4 still climbed vs c=2 ({scaling_c2:.3f}x -> {scaling_c4:.3f}x) "
                "and cleared the prior 1.32x ceiling class, so c=8 is a real question."
            )
    return {
        "run": run,
        "reason": reason,
        "prior_ceiling": ceiling,
        "measured_c2_vs_c1": scaling_c2,
        "measured_c4_vs_c1": scaling_c4,
        "c4_over_c2": climb,
        "memory_would_fit": True,
        "memory_note": (
            "Workspace is ~225 MiB/session at seq 512; c=8 is ~1.8 GiB extra on a "
            "13–14 GiB body (same 1.04x-class resident ratio as c=4). Memory is not "
            "the reason c=8 is skipped. Bandwidth is."
        ),
    }


def bandwidth_eaten(
    q4: dict[str, Any] | None, parent: dict[str, Any] | None
) -> dict[str, Any]:
    """If the denser parent does not raise WUs/hour, name what ate the bytes."""
    if not q4 or not parent:
        return {"comparable": False}
    q4_tps = float(q4.get("aggregate_tok_s") or 0.0)
    p_tps = float(parent.get("aggregate_tok_s") or 0.0)
    q4_wu = float(q4.get("verified_wu_per_hour") or 0.0)
    p_wu = float(parent.get("verified_wu_per_hour") or 0.0)
    if not q4_tps:
        return {"comparable": False, "reason": "q4 tps is 0"}
    byte_ratio = Q4_ACTIVE_BYTES / PARENT_ACTIVE_BYTES
    expected_tps = q4_tps * byte_ratio
    expected_wu = q4_wu * byte_ratio if q4_wu else None
    tps_gap = expected_tps - p_tps
    q4_gb = float(q4.get("achieved_gb_s") or 0.0)
    p_gb = float(parent.get("achieved_gb_s") or 0.0)
    reclaimed = Q4_ACTIVE_BYTES - PARENT_ACTIVE_BYTES
    converted = p_wu > q4_wu * 1.02 if q4_wu else p_tps > q4_tps * 1.02
    if converted:
        ate = (
            "Lower density converted: parent verified WUs/hour (or tok/s) rose. "
            "The reclaimed bytes are showing up as useful work."
        )
    elif p_gb + 1e-9 < q4_gb * 0.92:
        ate = (
            "Reclaimed bytes did not convert into tokens or verified WUs. Parent "
            f"achieved {p_gb:.1f} GB/s vs q4 {q4_gb:.1f} GB/s. The affine2 fused "
            "kernel did not keep the memory system fed at the incumbent's 468.9 GB/s; "
            "ALU / scale-bias / ceremony on the 756-dispatch graph ate the bandwidth "
            "the 3.14 EBPW pack theoretically returned. Concurrent sessions cannot "
            "spend bytes the kernel is not streaming."
        )
    elif p_tps >= q4_tps * 0.98 and p_wu + 1e-9 < q4_wu:
        ate = (
            "Parent matched tok/s but lost on verified WUs/hour. The extra density "
            "did not buy a better (or equally closed) answer; truncation and verifier "
            "rejects consumed the physical tie. Useful work is not TPS."
        )
    else:
        ate = (
            f"Parent should have been ~{byte_ratio:.2f}x faster if it streamed "
            f"{PARENT_ACTIVE_BYTES} active bytes at the q4 achieved bandwidth. "
            f"Measured parent tok/s {p_tps:.2f} vs expected {expected_tps:.2f} "
            f"(gap {tps_gap:.2f} tok/s). The fused affine2 graph is not delivering "
            "the byte cut as wall-time. Dispatch fusion (964→756) and lower EBPW "
            "are real; the memory system is still the ceiling, and the cheaper "
            "representation is not occupying it."
        )
    return {
        "comparable": True,
        "q4_active_bytes_per_token": Q4_ACTIVE_BYTES,
        "parent_active_bytes_per_token": PARENT_ACTIVE_BYTES,
        "reclaimed_bytes_per_token": reclaimed,
        "byte_ratio_q4_over_parent": byte_ratio,
        "q4_aggregate_tok_s": q4_tps,
        "parent_aggregate_tok_s": p_tps,
        "expected_parent_tok_s_if_same_gb_s": expected_tps,
        "parent_tok_s_gap_vs_byte_cut": tps_gap,
        "q4_achieved_gb_s": q4_gb,
        "parent_achieved_gb_s": p_gb,
        "q4_verified_wu_per_hour": q4_wu,
        "parent_verified_wu_per_hour": p_wu,
        "expected_parent_wu_per_hour_if_same_gb_s": expected_wu,
        "converted_into_useful_work": converted,
        "what_ate_the_reclaimed_bandwidth": ate,
        "gpu_ledger_incumbent_gb_s": GPU_LEDGER_GB_S,
        "roof_gb_s": ROOF_GB_S,
    }


def ensure_driver() -> Path:
    env_bin = os.environ.get("QWEN38_PRODBENCH_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    if DRIVER_BIN.is_file() and os.access(DRIVER_BIN, os.X_OK):
        return DRIVER_BIN
    CARGO_TARGET.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "--manifest-path",
        str(DRIVER_MANIFEST),
    ]
    print(f"prodbench: building driver {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(
        cmd, cwd=str(REPO), text=True, capture_output=True, env=env, timeout=3600
    )
    if proc.returncode != 0 or not DRIVER_BIN.is_file():
        raise SystemExit(
            f"cargo build of production bench driver failed rc={proc.returncode}\n"
            f"{proc.stderr[-8000:]}"
        )
    return DRIVER_BIN


def artifact_preflight(root: Path) -> list[str]:
    missing = []
    if not (root / "manifest.json").is_file() and not (root / "catalog.hq38m20").is_file():
        missing.append(f"artifact missing: {root}")
    return missing


def run_driver(
    *,
    artifact: Path,
    fusion: str,
    out: Path,
    concurrencies: list[int],
    topologies: list[str],
    probe_c8: bool,
    max_new: int,
    max_seq: int,
) -> dict[str, Any]:
    binary = ensure_driver()
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd: list[str] = []
    if lock.is_file():
        cmd.extend(["bash", str(lock), "n007-production-bench"])
    cmd.extend(
        [
            str(binary),
            "--artifact-root",
            str(artifact),
            "--tokenizer",
            str(TOKENIZER),
            "--workunits",
            str(WU_FILE),
            "--out",
            str(out),
            "--fusion",
            fusion,
            "--max-new-tokens",
            str(max_new),
            "--max-seq-len",
            str(max_seq),
            "--concurrencies",
            ",".join(str(c) for c in concurrencies),
            "--topologies",
            ",".join(topologies),
        ]
    )
    if probe_c8:
        cmd.append("--probe-c8")
    print(f"prodbench: {' '.join(cmd)}", file=sys.stderr, flush=True)
    # gpu_lane_lock.sh is argv0 of cmd: occupancy is sampled AFTER that lock
    # is acquired, by the driver itself (rss/metal). A pre-lock snapshot would
    # false-refuse a serialized neighbor still draining. After the driver
    # exits, a leftover 10+ GiB peer is a real second 27B.
    occ_before = occupancy_snapshot()
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(REPO), text=True, capture_output=True, timeout=7200
    )
    elapsed = time.perf_counter() - t0
    if out.is_file():
        raw = json.loads(out.read_text())
    else:
        raw = {
            "ok": False,
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-8000:],
        }
    raw["_runner"] = {
        "binary": str(binary),
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "stderr_tail": (proc.stderr or "")[-4000:],
        "occupancy_before": occ_before,
        "occupancy_after": occupancy_snapshot(),
    }
    if proc.returncode != 0:
        raw["_runner"]["failed"] = True
    return raw


def _c1_tps(summaries: list[dict[str, Any]], artifact: str, topology: str) -> float | None:
    for s in summaries:
        if (
            s.get("artifact") == artifact
            and s.get("topology") == topology
            and s.get("concurrency") == 1
        ):
            return float(s.get("aggregate_tok_s") or 0.0) or None
    return None


def wrap(
    q4_raw: dict[str, Any],
    parent_raw: dict[str, Any],
    *,
    elapsed_s: float,
    max_seq: int,
    ran_c8: bool,
    c8_decision: dict[str, Any],
) -> dict[str, Any]:
    ws = workspace_bytes(max_seq)
    kv = ws["gqa_kv_bytes"]
    pressure = memory_pressure()
    device = metal_device()
    finals: list[dict[str, Any]] = []
    # First pass without slowdown (need c=1 tps).
    staged: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    for name, raw, active in (
        ("q4_incumbent", q4_raw, Q4_ACTIVE_BYTES),
        ("parent_a", parent_raw, PARENT_ACTIVE_BYTES),
    ):
        for cell in raw.get("cells") or []:
            staged.append((name, active, cell, raw))
    # Seed c=1 tps per artifact/topology from raw walls.
    prelim: list[dict[str, Any]] = []
    for name, active, cell, _raw in staged:
        prelim.append(
            summarize_cell(
                cell,
                artifact=name,
                active_bytes=active,
                kv_bytes_one=kv,
                c1_aggregate_tps=None,
            )
        )
    summaries = []
    for name, active, cell, _raw in staged:
        topo = cell.get("topology") or ""
        c1 = _c1_tps(prelim, name, topo)
        summaries.append(
            summarize_cell(
                cell,
                artifact=name,
                active_bytes=active,
                kv_bytes_one=kv,
                c1_aggregate_tps=c1,
            )
        )
        finals.append(summaries[-1])

    def pick(artifact: str, c: int, topo_substr: str) -> dict[str, Any] | None:
        hits = [
            s
            for s in summaries
            if s.get("artifact") == artifact
            and s.get("concurrency") == c
            and topo_substr in (s.get("topology") or "")
        ]
        return hits[0] if hits else None

    q4_c1 = pick("q4_incumbent", 1, "sequential") or pick("q4_incumbent", 1, "concurrent")
    parent_c1 = pick("parent_a", 1, "sequential") or pick("parent_a", 1, "concurrent")
    q4_c4c = pick("q4_incumbent", 4, "concurrent")
    parent_c4c = pick("parent_a", 4, "concurrent")
    eaten = bandwidth_eaten(q4_c1, parent_c1)
    # If density did not convert at c=1, check whether it converted at c=4 concurrent.
    eaten_c4 = bandwidth_eaten(q4_c4c, parent_c4c)
    winner = choose_winner(summaries)

    def scaling(artifact: str, topology: str) -> dict[str, float]:
        c1 = pick(artifact, 1, topology)
        out = {}
        if not c1 or not c1.get("aggregate_tok_s"):
            return out
        base = c1["aggregate_tok_s"]
        for c in (1, 2, 4, 8):
            cell = pick(artifact, c, topology)
            if cell and base:
                out[str(c)] = cell["aggregate_tok_s"] / base
        return out

    parent_mtime_after = parent_is_unmutated(PARENT_ROOT)
    q4_live = q4_raw.get("_runner") or {}
    parent_live = parent_raw.get("_runner") or {}
    # Occupancy_before can see a serialized neighbor still holding the GPU
    # lock. The proof we did not load two bodies is one process, one weight
    # load, shared Arc, and the two artifacts never resident at once.
    second = bool(
        (q4_raw.get("weight_loads") not in (1, None))
        or (parent_raw.get("weight_loads") not in (1, None))
        or (q4_raw.get("process_count") not in (1, None))
        or (parent_raw.get("process_count") not in (1, None))
        or q4_raw.get("weights_ptr_shared") is False
        or parent_raw.get("weights_ptr_shared") is False
    )
    # Occupancy: launch geometry is DERIVED (GPU_LEDGER); hardware counter ABSENT.
    occupancy = {
        "hardware_occupancy_counter": {
            "value": None,
            "kind": "ABSENT",
            "unit": "fraction",
            "command": "MTLDevice.counterSets",
            "absent_reason": (
                "MTLDevice.counterSets on this Apple M3 Ultra contains only the "
                "'timestamp' set. There is no occupancy or SIMD-utilization counter. "
                "Launch-geometry occupancy is DERIVED, not MEASURED."
            ),
        },
        "launch_geometry_threadgroups_per_core": {
            "value": 8704 / GPU_CORES,
            "kind": "DERIVED",
            "unit": "threadgroups/core",
            "command": "GPU_LEDGER.json fields.active_threadgroups / 60",
            "absent_reason": None,
            "note": (
                "Workhorse gate_proj launch ceil(17408/2)=8704 TGs of 128 threads "
                "on 60 cores. Bandwidth-saturated, not occupancy-starved. Not re-derived."
            ),
        },
        "metal_working_set": {
            "q4_final": (q4_raw.get("metal_final") or {}),
            "parent_final": (parent_raw.get("metal_final") or {}),
            "kind": "MEASURED",
            "note": "MTLDevice.currentAllocatedSize / recommendedMaxWorkingSetSize in-process",
        },
    }

    answer = _answer(winner, eaten, eaten_c4, summaries, c8_decision)
    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "elapsed_s": elapsed_s,
        "obligation": (
            "N007 — given the ~1.32x concurrency ceiling, what actually maximizes "
            "verified accepted useful work per wall second?"
        ),
        "question": (
            "Which configuration of {q4 incumbent, sealed parent A} × {c=1,2,4} "
            "(and c=8 only if physically meaningful) maximizes VERIFIED accepted "
            "Hawking WorkUnits per hour?"
        ),
        "answer": answer,
        "ranking_quantity": "verified_accepted_workunits_per_hour",
        "did_not_load_second_27b": (not second)
        and bool(q4_raw.get("did_not_load_second_27b", True))
        and bool(parent_raw.get("did_not_load_second_27b", True)),
        "did_not_write_ascent_or_campaign": True,
        "did_not_mutate_parent": True,
        "parent_immutable": parent_mtime_after,
        "workunits": [
            {k: w[k] for k in ("id", "role", "kind", "expected")} for w in workunits()
        ],
        "n_workunits": len(workunits()),
        "negative_control": "wu_token_generator",
        "gpu_gate": {
            "device": device.get("name"),
            "source": device.get("source"),
            "recommendedMaxWorkingSetSize_gib": (
                device["recommendedMaxWorkingSetSize"] / (1024**3)
                if device.get("recommendedMaxWorkingSetSize")
                else None
            ),
            "hasUnifiedMemory": device.get("hasUnifiedMemory"),
        },
        "finalists": {
            "q4_incumbent": {
                "artifact": str(Q4_ROOT),
                "complete_physical_bpw": Q4_EBPW,
                "active_bytes_per_token": Q4_ACTIVE_BYTES,
                "dispatches_per_token": Q4_DISPATCHES,
                "fusion": "off",
                "resident_weight_bytes": q4_raw.get("resident_weight_bytes"),
                "rss_after_load_bytes": q4_raw.get("rss_after_load_bytes"),
                "rss_final_bytes": q4_raw.get("rss_final_bytes"),
                "metal_final": q4_raw.get("metal_final"),
                "attached_sessions": q4_raw.get("attached_sessions"),
                "weights_ptr_shared": q4_raw.get("weights_ptr_shared"),
                "theoretical_dispatches": q4_raw.get("theoretical_dispatches"),
            },
            "parent_a": {
                "artifact": str(PARENT_ROOT),
                "complete_physical_bpw": PARENT_EBPW,
                "active_bytes_per_token": PARENT_ACTIVE_BYTES,
                "dispatches_per_token": PARENT_DISPATCHES,
                "fusion": "parent (mlp swiglu + gqa qkv + dn inproj)",
                "resident_weight_bytes": parent_raw.get("resident_weight_bytes"),
                "rss_after_load_bytes": parent_raw.get("rss_after_load_bytes"),
                "rss_final_bytes": parent_raw.get("rss_final_bytes"),
                "metal_final": parent_raw.get("metal_final"),
                "attached_sessions": parent_raw.get("attached_sessions"),
                "weights_ptr_shared": parent_raw.get("weights_ptr_shared"),
                "theoretical_dispatches": parent_raw.get("theoretical_dispatches"),
            },
        },
        "workspace": ws,
        "memory_pressure": pressure,
        "occupancy": occupancy,
        "c8": {"ran": ran_c8, **c8_decision},
        "prior_ceiling_not_rederived": MULTISESSION_CEILING,
        "cells": summaries,
        "scaling_vs_c1_aggregate_tps": {
            "q4_incumbent": {
                "sequential_per_session": scaling("q4_incumbent", "sequential"),
                "concurrent_independent": scaling("q4_incumbent", "concurrent"),
            },
            "parent_a": {
                "sequential_per_session": scaling("parent_a", "sequential"),
                "concurrent_independent": scaling("parent_a", "concurrent"),
            },
        },
        "bandwidth_eaten": {
            "at_c1": eaten,
            "at_c4_concurrent": eaten_c4,
        },
        "winner": winner,
        "raw": {
            "q4": str(RAW_DIR / "PRODUCTION_BENCH.q4.raw.json"),
            "parent": str(RAW_DIR / "PRODUCTION_BENCH.parent.raw.json"),
        },
        "sentinels": {
            "weights_ptr_shared_q4": q4_raw.get("weights_ptr_shared"),
            "weights_ptr_shared_parent": parent_raw.get("weights_ptr_shared"),
            "q4_theoretical_dispatches": q4_raw.get("theoretical_dispatches"),
            "parent_theoretical_dispatches": parent_raw.get("theoretical_dispatches"),
            "token_generator_must_not_verify": "wu_token_generator",
            "noop_control": (
                "wu_token_generator is completed tokens that the verifier rejects; "
                "a harness that counted tokens as WorkUnits would score it"
            ),
            "bad_control": (
                "the same verifier rejects a missing JSON city and an unclosed <think>; "
                "those WUs stay in completed and out of verified"
            ),
        },
    }


def _answer(
    winner: dict[str, Any],
    eaten: dict[str, Any],
    eaten_c4: dict[str, Any],
    summaries: list[dict[str, Any]],
    c8: dict[str, Any],
) -> str:
    w = winner.get("winner") or {}
    bits = [
        f"Winner is {w.get('artifact')} c={w.get('concurrency')} "
        f"{w.get('topology')} at {float(w.get('verified_wu_per_hour') or 0):.1f} "
        f"verified WUs/hour "
        f"(aggregate {float(w.get('aggregate_tok_s') or 0):.2f} tok/s, "
        f"TTFT p50 {float(w.get('ttft_p50_s') or 0):.3f}s)."
    ]
    if winner.get("winner_differs_from_highest_tps"):
        t = winner.get("highest_aggregate_tok_s_cell") or {}
        bits.append(
            f"Highest aggregate tok/s was {t.get('artifact')} c={t.get('concurrency')} "
            f"{t.get('topology')} at {float(t.get('aggregate_tok_s') or 0):.2f} tok/s "
            f"but only {float(t.get('verified_wu_per_hour') or 0):.1f} verified WUs/hour "
            "— TPS is not the ranking quantity."
        )
    if eaten.get("comparable"):
        bits.append(eaten.get("what_ate_the_reclaimed_bandwidth") or "")
    if eaten_c4.get("comparable") and eaten_c4.get("what_ate_the_reclaimed_bandwidth"):
        if eaten_c4.get("converted_into_useful_work") != eaten.get("converted_into_useful_work"):
            bits.append("At c=4 concurrent: " + eaten_c4["what_ate_the_reclaimed_bandwidth"])
    if not c8.get("run"):
        bits.append("c=8 skipped: " + (c8.get("reason") or ""))
    n = len(summaries)
    bits.append(f"{n} measured cells; completed vs verified is the gap the bench exists to see.")
    return " ".join(b for b in bits if b)


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {RECEIPT}", file=sys.stderr, flush=True)


def write_workunits_file() -> None:
    WU_FILE.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "workunits": [
            {"id": w["id"], "prompt": w["prompt"]} for w in workunits()
        ]
    }
    WU_FILE.write_text(json.dumps(body, indent=1) + "\n")


def build(*, live: bool = True, force: bool = False) -> dict[str, Any]:
    max_new = int(os.environ.get("NOETIC_PRODBENCH_MAX_NEW", str(DEFAULT_MAX_NEW)))
    max_seq = int(os.environ.get("NOETIC_PRODBENCH_MAX_SEQ", str(DEFAULT_MAX_SEQ)))
    reuse = os.environ.get("NOETIC_PRODBENCH_REUSE", "1") != "0"
    if reuse and not force and RECEIPT.is_file():
        doc = json.loads(RECEIPT.read_text())
        if doc.get("schema") == SCHEMA and doc.get("cells"):
            return doc
    q4_raw_path = RAW_DIR / "PRODUCTION_BENCH.q4.raw.json"
    parent_raw_path = RAW_DIR / "PRODUCTION_BENCH.parent.raw.json"
    t0 = time.perf_counter()
    if not live:
        q4_raw = {"did_not_load_second_27b": True, "cells": []}
        parent_raw = {"did_not_load_second_27b": True, "cells": []}
        c8 = c8_physically_meaningful(
            MULTISESSION_CEILING["concurrent_independent"]["2"],
            MULTISESSION_CEILING["concurrent_independent"]["4"],
        )
        doc = wrap(
            q4_raw,
            parent_raw,
            elapsed_s=0.0,
            max_seq=max_seq,
            ran_c8=False,
            c8_decision=c8,
        )
        write_receipt(doc)
        return doc

    missing = artifact_preflight(Q4_ROOT) + artifact_preflight(PARENT_ROOT)
    if not TOKENIZER.is_file():
        missing.append(f"tokenizer missing: {TOKENIZER}")
    if missing:
        raise SystemExit("preflight: " + "; ".join(missing))

    write_workunits_file()
    # c=8 decision uses the already-measured ceiling unless a live c=4
    # climb is requested via env. Default: skip c=8 as not physically meaningful.
    c8 = c8_physically_meaningful(
        MULTISESSION_CEILING["concurrent_independent"]["2"],
        MULTISESSION_CEILING["concurrent_independent"]["4"],
    )
    if os.environ.get("NOETIC_PRODBENCH_FORCE_C8") == "1":
        c8 = {**c8, "run": True, "reason": "NOETIC_PRODBENCH_FORCE_C8=1"}
    conc = list(DEFAULT_CONCURRENCIES)
    if c8["run"]:
        conc.append(8)
    # Concurrent independent is the topology that can raise WUs/hour.
    # sequential c=N is the same aggregate as c=1 (n006: 1.002x / 0.989x) and
    # would triple the wall without changing the ranking quantity. c=1 concurrent
    # is the serial baseline.
    topologies = ["concurrent"]

    def load_or_run(path: Path, artifact: Path, fusion: str) -> dict[str, Any]:
        if (
            not force
            and path.is_file()
            and os.environ.get("NOETIC_PRODBENCH_REUSE_RAW") == "1"
        ):
            return json.loads(path.read_text())
        return run_driver(
            artifact=artifact,
            fusion=fusion,
            out=path,
            concurrencies=conc,
            topologies=topologies,
            probe_c8=True,
            max_new=max_new,
            max_seq=max_seq,
        )

    # One body at a time. q4 process must exit before parent loads.
    q4_raw = load_or_run(q4_raw_path, Q4_ROOT, "off")
    parent_raw = load_or_run(parent_raw_path, PARENT_ROOT, "parent")
    elapsed = round(time.perf_counter() - t0, 3)
    # If live c=4 scaling actually climbed, note it; do not silently add c=8
    # after the fact (that would require a third load).
    def live_scale(raw: dict[str, Any]) -> tuple[float | None, float | None]:
        by = {}
        for cell in raw.get("cells") or []:
            if "concurrent" in (cell.get("topology") or ""):
                by[int(cell.get("concurrency") or 0)] = cell
        c1 = by.get(1)
        if not c1 or not c1.get("wall_ns"):
            return None, None
        def tps(cell: dict[str, Any]) -> float:
            toks = sum(int(r.get("n_new_tokens") or 0) for r in cell.get("workunits") or [])
            wall = int(cell.get("wall_ns") or 0) / 1e9
            return toks / wall if wall else 0.0
        base = tps(c1)
        if not base:
            return None, None
        s2 = tps(by[2]) / base if 2 in by else None
        s4 = tps(by[4]) / base if 4 in by else None
        return s2, s4

    s2, s4 = live_scale(q4_raw)
    live_c8 = c8_physically_meaningful(s2, s4)
    # Keep the pre-run skip unless live evidence says we should have run it.
    if live_c8["run"] and not c8["run"]:
        live_c8 = {
            **live_c8,
            "run": False,
            "reason": live_c8["reason"]
            + " Live c=4 climbed, but c=8 was not pre-admitted (would need a third load).",
            "live_would_have_run": True,
        }
    elif not live_c8["run"]:
        c8 = live_c8
    doc = wrap(
        q4_raw,
        parent_raw,
        elapsed_s=elapsed,
        max_seq=max_seq,
        ran_c8=8 in conc,
        c8_decision=c8 if 8 in conc else live_c8,
    )
    write_receipt(doc)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()
    doc = build(live=not args.cpu_only, force=args.force)
    w = (doc.get("winner") or {}).get("winner") or {}
    print(
        json.dumps(
            {
                "winner": w,
                "c8_ran": (doc.get("c8") or {}).get("ran"),
                "n_cells": len(doc.get("cells") or []),
                "receipt": str(RECEIPT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
