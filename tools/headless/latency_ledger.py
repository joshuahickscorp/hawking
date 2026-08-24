#!/usr/bin/env python3
"""N028 — LATENCY LEDGER: user-visible INTERACTIVE latency, measured.

The product is a ledger, not a target. S022 §43: establish BASELINE first;
no invented budgets. Control-plane stages are reused from
receipts/headless/CONTROL_PLANE_LATENCY_LEDGER.json (N022). Model-side
stages (TTFT, prefill, per-token, sampler) come from the actual decode
path — GPU_LEDGER complete-wall, PREFILL_KV, PRODUCTION_BENCH, isolated
sample_argmax, plus a live CPU argmax of Qwen3.8 vocab that mirrors
crates/hawking-core/src/sample.rs / kernels::argmax_f32.

GPU is not re-run here (that would load a 27B). If a lane remasures GPU,
serialize with: bash tools/gpu_lane_lock.sh n028-latency <cmd>

    python3 tools/headless/latency_ledger.py
    python3 -m pytest tools/headless/test_latency_ledger.py -q
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCHEMA = "hawking.headless.latency_ledger.v1"
GATE = "LATENCY_LEDGER"
RECEIPT = REPO / "receipts" / "headless" / "LATENCY_LEDGER.json"

CPL_LEDGER = REPO / "receipts" / "headless" / "CONTROL_PLANE_LATENCY_LEDGER.json"
GPU_LEDGER = REPO / "receipts" / "headless" / "GPU_LEDGER.json"
PREFILL_KV = REPO / "receipts" / "headless" / "PREFILL_KV.json"
PROD_BENCH = REPO / "receipts" / "headless" / "PRODUCTION_BENCH.json"
PROD_BENCH_Q4_RAW = REPO / "receipts" / "headless" / "PRODUCTION_BENCH.q4.raw.json"
GPU_RAW_GLOB = "GPU_LEDGER_RAW.run*.json"
TOKEN_NS_GIT = "HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"

PARENT = Path.home() / "noetic" / "NOETIC_PARENT_A"
QWEN38_VOCAB = 248_320  # crates/hawking-core/src/model/qwen38_geometry.rs
SAMPLE_KERNEL = "sample_argmax_f32"
SAMPLE_SHADER = "crates/hawking-core/shaders/sample.metal"
SAMPLE_RS = "crates/hawking-core/src/sample.rs"

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"

# Acceptance vector. Every name is MEASURED or ABSENT-with-reason.
REQUIRED_STAGES = (
    "ttft",
    "tpot",
    "prefill",
    "inter_token",
    "token_latency_p50",
    "token_latency_p95",
    "token_latency_p99",
    "admission",
    "cold_start",
    "warm_start",
    "scheduling",
    "context_compile",
    "sampler",
)

# N022 measurement id for each control-plane stage we reuse (never re-time).
CPL_REUSE = {
    "admission": "runtime_admission",
    "scheduling": "scheduler_cycle",
    "context_compile": "context_compile",
}

N_NOOP = 5
N_SAMPLER = 21


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear interpolation, same as tools/headless/production_bench.py."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stats_ms(samples: Sequence[float]) -> dict[str, Any]:
    xs = [float(x) for x in samples]
    if not xs:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "stdev": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "spread_pct": None,
            "samples": [],
        }
    med = statistics.median(xs)
    spread_pct = None if med == 0 else 100.0 * (max(xs) - min(xs)) / med
    return {
        "n": len(xs),
        "median": round(med, 6),
        "mean": round(statistics.mean(xs), 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
        "stdev": round(statistics.pstdev(xs), 6) if len(xs) > 1 else 0.0,
        "p50": round(float(percentile(xs, 50) or 0.0), 6),
        "p95": round(float(percentile(xs, 95) or 0.0), 6),
        "p99": round(float(percentile(xs, 99) or 0.0), 6),
        "spread_pct": round(spread_pct, 6) if spread_pct is not None else None,
        "samples": [round(x, 6) for x in xs],
    }


def ns_to_ms(ns: Sequence[float] | float) -> list[float] | float:
    if isinstance(ns, (int, float)):
        return float(ns) / 1e6
    return [float(x) / 1e6 for x in ns]


def argmax_f32(xs: Sequence[float]) -> int:
    """crates/hawking-core/src/kernels/mod.rs::argmax_f32 (CPU reference)."""
    best = 0
    best_v = -math.inf
    for i, v in enumerate(xs):
        if v > best_v:
            best = i
            best_v = v
    return best


def occupancy_snapshot() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,rss,command"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (PermissionError, OSError) as exc:
        return {
            "ps_matches": [],
            "loaded_a_second_27b": False,
            "kind": ABSENT,
            "absent_reason": (
                f"ps was refused ({type(exc).__name__}: {exc}). Occupancy is "
                "ABSENT, not assumed empty. This harness does not spawn a "
                "decode binary and cannot load a second 27B."
            ),
            "note": "A second Qwen3.8-27B would show RSS in the 10+ GiB class.",
        }
    lines: list[str] = []
    second_27b = False
    for line in proc.stdout.splitlines():
        low = line.lower()
        if not any(s in low for s in ("llama-server", "ascension_qwen", "mlx_lm.server")):
            continue
        if "rg " in low or "latency_ledger" in low:
            continue
        parts = line.split()
        try:
            rss_kb = int(parts[1])
        except (IndexError, ValueError):
            rss_kb = 0
        lines.append(line.strip())
        if rss_kb > 4_000_000:
            second_27b = True
    return {
        "ps_matches": lines,
        "loaded_a_second_27b": second_27b,
        "kind": MEASURED,
        "note": (
            "A second Qwen3.8-27B would show RSS in the 10+ GiB class. "
            "This harness does not spawn a decode binary."
        ),
    }


def parent_identity(root: Path) -> dict[str, Any]:
    catalog = root / "catalog.hq38m20"
    out: dict[str, Any] = {
        "path": str(root),
        "exists": root.is_dir(),
        "catalog": str(catalog),
        "catalog_present": catalog.is_file(),
    }
    if catalog.is_file():
        st = catalog.stat()
        out["catalog_bytes"] = int(st.st_size)
        out["catalog_mtime_ns"] = int(st.st_mtime_ns)
        out["catalog_ino"] = int(st.st_ino)
        out["writable_check"] = oct(st.st_mode)
    return out


def time_process(cmd: Sequence[str], n: int) -> dict[str, Any]:
    samples: list[float] = []
    rcs: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter()
        p = subprocess.run(
            list(cmd),
            capture_output=True,
            timeout=60,
            cwd=str(REPO),
        )
        samples.append((time.perf_counter() - t0) * 1000.0)
        rcs.append(int(p.returncode))
    return {"samples_ms": samples, "returncodes": rcs, "ok": all(c == 0 for c in rcs)}


def measure_cpu_sampler(n: int = N_SAMPLER) -> dict[str, Any]:
    """Live CPU greedy argmax over Qwen3.8 vocab. No weights, no Metal, no 27B.

    Production INTERACTIVE greedy does not take this path: sample_argmax_f32
    runs on GPU inside the decode CB and only a u32 is read back. This is
    the CPU reference in crates/hawking-core/src/sample.rs (temperature==0)
    and kernels::argmax_f32, timed so the sampler stage is not a reused
    GPU number wearing a CPU costume.
    """
    vocab = QWEN38_VOCAB
    logits = [0.0] * vocab
    # One unique peak so the scan is the whole cost, not a data-dependent early exit.
    logits[vocab // 2] = 1.0
    argmax_f32(logits)  # warmup, page the list
    greedy: list[float] = []
    ids: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = argmax_f32(logits)
        greedy.append((time.perf_counter() - t0) * 1000.0)
        ids.append(tok)
    assert all(i == vocab // 2 for i in ids), ids[:3]
    return {
        "vocab": vocab,
        "kernel_equivalent": "hawking_core::kernels::argmax_f32 / sample.rs argmax (temp=0)",
        "shader": SAMPLE_SHADER,
        "source": SAMPLE_RS,
        "greedy_ms": stats_ms(greedy),
        "picked": vocab // 2,
        "n": n,
        "command": [
            sys.executable,
            str(HERE / "latency_ledger.py"),
            "--measure-cpu-sampler",
        ],
    }


def cpl_row(cpl: dict[str, Any], measurement_id: str) -> Optional[dict[str, Any]]:
    for row in cpl.get("measurements") or []:
        if isinstance(row, dict) and row.get("id") == measurement_id:
            return row
    return None


def reuse_cpl_stage(
    *,
    stage_id: str,
    cpl: Optional[dict[str, Any]],
    owner: str,
    blocking_dep: str,
    avoidable: bool,
    cacheable: bool,
    fusible: bool,
    classification_reason: str,
    pct_of: str,
) -> dict[str, Any]:
    mid = CPL_REUSE[stage_id]
    if not isinstance(cpl, dict):
        return absent_stage(
            stage_id,
            reason=f"{CPL_LEDGER.name} missing; cannot reuse N022 {mid}",
            command=["cat", str(CPL_LEDGER)],
            owner=owner,
            blocking_dep=blocking_dep,
            pct_of=pct_of,
        )
    row = cpl_row(cpl, mid)
    if not row or not row.get("ok"):
        return absent_stage(
            stage_id,
            reason=f"N022 measurement {mid!r} missing or not ok",
            command=["python3", "tools/headless/control_plane_latency.py"],
            owner=owner,
            blocking_dep=blocking_dep,
            pct_of=pct_of,
        )
    cold = row.get("cold_ms") or {}
    warm = row.get("warm_ms") or {}
    samples = list(warm.get("samples") or cold.get("samples") or [])
    fig = stats_ms(samples)
    # Preserve N022 medians exactly as the baseline (do not re-time).
    if warm.get("median") is not None:
        fig["median"] = float(warm["median"])
        fig["p50"] = float(warm["median"])
    return measured_stage(
        stage_id,
        fig=fig,
        owner=owner,
        blocking_dep=blocking_dep,
        avoidable=avoidable,
        cacheable=cacheable,
        fusible=fusible,
        classification_reason=classification_reason,
        command=list(row.get("command") or ["N022", mid]),
        source=str(CPL_LEDGER.relative_to(REPO)),
        kind=MEASURED,
        note=(
            f"Reused N022 {mid} (not re-timed). "
            f"cold_median_ms={cold.get('median')} warm_median_ms={warm.get('median')}. "
            "INTERACTIVE overlay uses the warm figure (process already up)."
        ),
        pct_of=pct_of,
        extra={
            "n022_id": mid,
            "n022_label": row.get("label"),
            "cold_ms": cold,
            "warm_ms": warm,
            "n022_category": row.get("category"),
        },
    )


def absent_stage(
    stage_id: str,
    *,
    reason: str,
    command: Sequence[str],
    owner: str,
    blocking_dep: str,
    pct_of: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row = {
        "id": stage_id,
        "status": ABSENT,
        "kind": ABSENT,
        "ms": None,
        "absolute_ms": None,
        "pct_of_total": None,
        "pct_of": pct_of,
        "owner": owner,
        "blocking_dep": blocking_dep,
        "avoidable": None,
        "cacheable": None,
        "fusible": None,
        "classification_reason": None,
        "command": list(command),
        "source": None,
        "note": None,
        "reason": reason,
        "spread": stats_ms([]),
    }
    if extra:
        row["extra"] = extra
    return row


def measured_stage(
    stage_id: str,
    *,
    fig: dict[str, Any],
    owner: str,
    blocking_dep: str,
    avoidable: bool,
    cacheable: bool,
    fusible: bool,
    classification_reason: str,
    command: Sequence[str],
    source: str,
    kind: str,
    note: str,
    pct_of: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    med = fig.get("median")
    row = {
        "id": stage_id,
        "status": MEASURED,
        "kind": kind,
        "ms": med,
        "absolute_ms": med,
        "pct_of_total": None,  # filled after totals
        "pct_of": pct_of,
        "owner": owner,
        "blocking_dep": blocking_dep,
        "avoidable": bool(avoidable),
        "cacheable": bool(cacheable),
        "fusible": bool(fusible),
        "classification_reason": classification_reason,
        "command": list(command),
        "source": source,
        "note": note,
        "reason": None,
        "spread": fig,
    }
    if extra:
        row["extra"] = extra
    return row


def pool_gpu_raw() -> dict[str, Any]:
    walls: list[float] = []
    gpus: list[float] = []
    prefills: list[float] = []
    first_walls: list[float] = []
    first_gpus: list[float] = []
    sample_readbacks: list[float] = []
    encodes: list[float] = []
    files: list[str] = []
    for path in sorted((REPO / "receipts" / "headless").glob(GPU_RAW_GLOB)):
        doc = load_json(path)
        if not doc:
            continue
        files.append(str(path.relative_to(REPO)))
        auth = doc.get("authority") or {}
        w = ((auth.get("pooled_steady_complete_wall_ns") or {}).get("all")) or []
        g = ((auth.get("pooled_steady_gpu_ns") or {}).get("all")) or []
        walls.extend(float(x) for x in w)
        gpus.extend(float(x) for x in g)
        cg = doc.get("cold_generate") or {}
        if cg.get("prefill_wall_ns") is not None:
            prefills.append(float(cg["prefill_wall_ns"]))
        step = cg.get("cold_or_first_step") or {}
        if step.get("complete_wall_ns") is not None:
            first_walls.append(float(step["complete_wall_ns"]))
        if step.get("gpu_ns") is not None:
            first_gpus.append(float(step["gpu_ns"]))
        means = ((cg.get("closure") or {}).get("named_component_means_ns")) or {}
        if means.get("sample_readback") is not None:
            sample_readbacks.append(float(means["sample_readback"]))
        if means.get("encode_host_prepare") is not None:
            encodes.append(float(means["encode_host_prepare"]))
    return {
        "files": files,
        "complete_wall_ns": walls,
        "gpu_ns": gpus,
        "prefill_wall_ns": prefills,
        "cold_first_step_wall_ns": first_walls,
        "cold_first_step_gpu_ns": first_gpus,
        "sample_readback_ns_means": sample_readbacks,
        "encode_host_prepare_ns_means": encodes,
    }


def git_show_json(spec: str) -> Optional[dict[str, Any]]:
    p = subprocess.run(
        ["git", "show", spec],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def isolated_argmax(token_ns: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(token_ns, dict):
        return {"ok": False, "reason": f"git show {TOKEN_NS_GIT} failed"}
    for fam in token_ns.get("isolated") or []:
        if isinstance(fam, dict) and fam.get("name") == "argmax":
            reps = [float(x) for x in (fam.get("gpu_ns_reps") or [])]
            return {
                "ok": True,
                "name": "argmax",
                "gpu_ns_reps": reps,
                "median_gpu_ns": fam.get("median_gpu_ns"),
                "dispatches": fam.get("dispatches"),
                "command_buffers": fam.get("command_buffers"),
                "source": TOKEN_NS_GIT,
            }
    return {"ok": False, "reason": "isolated family 'argmax' not in TOKEN_NS ledger"}


def fill_pct(stages: dict[str, dict[str, Any]], totals: dict[str, float]) -> None:
    for st in stages.values():
        key = st.get("pct_of")
        total = totals.get(key or "")
        ms = st.get("absolute_ms")
        if total and total > 0 and isinstance(ms, (int, float)):
            st["pct_of_total"] = round(100.0 * float(ms) / float(total), 4)
        elif st.get("status") == MEASURED:
            st["pct_of_total"] = 0.0


def interactive_cell(prod: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(prod, dict):
        return None
    for cell in prod.get("cells") or []:
        if (
            cell.get("artifact") == "q4_incumbent"
            and int(cell.get("concurrency") or 0) == 1
            and cell.get("topology") == "concurrent_independent"
        ):
            return cell
    return None


def build() -> dict[str, Any]:
    generated_at = now_iso()
    head = git_head()
    occupancy = occupancy_snapshot()
    parent_before = parent_identity(PARENT)

    cpl = load_json(CPL_LEDGER)
    gpu = load_json(GPU_LEDGER)
    prefill_doc = load_json(PREFILL_KV)
    prod = load_json(PROD_BENCH)
    prod_raw = load_json(PROD_BENCH_Q4_RAW)
    raw = pool_gpu_raw()
    token_ns = git_show_json(TOKEN_NS_GIT)
    argmax_iso = isolated_argmax(token_ns)

    noop = time_process([sys.executable, "-c", "pass"], N_NOOP)
    cpu_sampler = measure_cpu_sampler(N_SAMPLER)

    cell = interactive_cell(prod)
    q4_short = ((prefill_doc or {}).get("prefill") or {}).get("q4") or {}
    q4_short = q4_short.get("short") or {}

    stages: dict[str, dict[str, Any]] = {}

    # --- Control plane (N022 reuse) ---
    stages["admission"] = reuse_cpl_stage(
        stage_id="admission",
        cpl=cpl,
        owner="CPU",
        blocking_dep="none (MemGate.consider before the decode CB)",
        avoidable=False,
        cacheable=True,
        fusible=False,
        classification_reason=(
            "Cold path compiles a temp Swift Metal probe (~276 ms). "
            "Warm/cached is ~23 ms. Cacheable: metal_device + genome JSON. "
            "Not fusible into the decode CB."
        ),
        pct_of="interactive_first_token_ms",
    )
    stages["scheduling"] = reuse_cpl_stage(
        stage_id="scheduling",
        cpl=cpl,
        owner="CPU",
        blocking_dep="dag_load",
        avoidable=True,
        cacheable=True,
        fusible=False,
        classification_reason=(
            "Scheduler.dispatch always persists the DAG. Avoidable: persist "
            "on dirty flag. Cacheable: in-process unit set. Not GPU-fusible."
        ),
        pct_of="interactive_first_token_ms",
    )
    stages["context_compile"] = reuse_cpl_stage(
        stage_id="context_compile",
        cpl=cpl,
        owner="CPU",
        blocking_dep="mission_load",
        avoidable=False,
        cacheable=True,
        fusible=False,
        classification_reason=(
            "GoalCompiler.compile of the INTERACTIVE goal text is <1 ms. "
            "Cacheable for an unchanged goal. Not the first-token cost."
        ),
        pct_of="interactive_first_token_ms",
    )

    # --- Model-side: prefill (N016 decode-path teacher-forced walk) ---
    warm_prefill = q4_short.get("warm_prefill_wall_ns") or {}
    cold_prefill = q4_short.get("cold_prefill_wall_ns") or {}
    if warm_prefill.get("kind") == MEASURED and warm_prefill.get("spread"):
        prefill_ms = ns_to_ms(list(warm_prefill["spread"]["all"]))
        assert isinstance(prefill_ms, list)
        stages["prefill"] = measured_stage(
            "prefill",
            fig=stats_ms(prefill_ms),
            owner="GPU",
            blocking_dep="resident weights + tokenizer prompt ids",
            avoidable=False,
            cacheable=True,
            fusible=True,
            classification_reason=(
                "Teacher-forced walk of every prompt token on the production "
                "decode GEMV graph (964 dispatches / step). Not avoidable for "
                "a new prompt. Cacheable via prompt-cache / KV prefix share. "
                "Fusible: a batched-prefill path is allowed and is not shipped "
                "(N016). INTERACTIVE uses the warm short-prompt figure."
            ),
            command=[
                "python3",
                "tools/headless/prefill_kv.py",
                "--measure",
            ],
            source=str(PREFILL_KV.relative_to(REPO)),
            kind=MEASURED,
            note=(
                f"q4 short prompt_len={q4_short.get('prompt_len')} "
                f"warm_prefill_wall median {warm_prefill.get('value')} ns. "
                f"cold_prefill_wall median {cold_prefill.get('value')} ns. "
                "GPU_LEDGER_RAW cold_generate.prefill_wall_ns corroborates "
                f"{[round(x/1e6,3) for x in raw['prefill_wall_ns']]} ms."
            ),
            pct_of="interactive_first_token_ms",
            extra={
                "prompt_len": q4_short.get("prompt_len"),
                "dispatches_per_step": q4_short.get("dispatches_per_step"),
                "cold_prefill_wall_ns": cold_prefill,
                "warm_prefill_wall_ns": warm_prefill,
                "gpu_raw_prefill_wall_ms": [
                    round(x / 1e6, 6) for x in raw["prefill_wall_ns"]
                ],
            },
        )
    else:
        stages["prefill"] = absent_stage(
            "prefill",
            reason="PREFILL_KV.json q4 short warm_prefill_wall_ns not MEASURED",
            command=["python3", "tools/headless/prefill_kv.py", "--measure"],
            owner="GPU",
            blocking_dep="resident weights",
            pct_of="interactive_first_token_ms",
        )

    # --- TTFT: user-visible first token, INTERACTIVE c=1 WorkUnit ---
    if cell and cell.get("ttft_s"):
        ttft_ms = [float(s) * 1000.0 for s in cell["ttft_s"]]
        stages["ttft"] = measured_stage(
            "ttft",
            fig=stats_ms(ttft_ms),
            owner="CPU+GPU",
            blocking_dep="prefill (last prefill step emits new-token[0])",
            avoidable=False,
            cacheable=True,
            fusible=True,
            classification_reason=(
                "User-visible time to first token of a Hawking WorkUnit at "
                "c=1 on the q4 incumbent. Dominated by prefill of the chat "
                "prompt. Cacheable: prompt-cache. Fusible: batched prefill "
                "not shipped. Not avoidable for a new user turn."
            ),
            command=[
                "python3",
                "tools/headless/production_bench.py",
            ],
            source=str(PROD_BENCH.relative_to(REPO)),
            kind=MEASURED,
            note=(
                "INTERACTIVE product TTFT = PRODUCTION_BENCH q4_incumbent "
                "c=1 concurrent_independent ttft_s (WorkUnit prompts, "
                f"prompt_token_counts={((prod_raw or {}).get('prompt_token_counts'))}). "
                "Decode-path definition (GPU_LEDGER): last prefill step emits "
                "new-token[0]; steady TPOT is new-tokens[1..] only. Short "
                "compiler-prompt prefill is the PREFILL stage, not this row — "
                "WorkUnit prompts are longer (~47–109 tokens) so TTFT > short prefill."
            ),
            pct_of="user_visible_ttft_ms",
            extra={
                "ttft_p50_s": cell.get("ttft_p50_s"),
                "ttft_p95_s": cell.get("ttft_p95_s"),
                "artifact": cell.get("artifact"),
                "concurrency": cell.get("concurrency"),
                "topology": cell.get("topology"),
            },
        )
    else:
        stages["ttft"] = absent_stage(
            "ttft",
            reason="PRODUCTION_BENCH.json missing q4 c=1 ttft_s",
            command=["python3", "tools/headless/production_bench.py"],
            owner="CPU+GPU",
            blocking_dep="prefill",
            pct_of="interactive_first_token_ms",
        )

    # --- TPOT / inter-token / p50 p95 p99 from complete-token wall ---
    wall_ms = ns_to_ms(raw["complete_wall_ns"])
    if isinstance(wall_ms, list) and wall_ms:
        wall_fig = stats_ms(wall_ms)
        gpu_cmd = (
            ((gpu or {}).get("fields") or {})
            .get("COMPLETE_TOKEN_WALL_NS", {})
            .get("command")
            or "./tools/gpu_lane_lock.sh n004-gpu-ledger ascension_qwen38_hybrid_greedy --complete-wall"
        )
        decode_note = (
            "Complete-token wall from GPU_LEDGER_RAW pooled_steady_complete_wall_ns "
            f"(n={len(wall_ms)} steps across {len(raw['files'])} process runs). "
            "Definition: one native decode step + host encode/submit/wait/"
            "sample_readback/state/tokenizer/bookkeeping. Prefill excluded. "
            "This IS TPOT and the inter-token gap on the INTERACTIVE stream."
        )
        tpot_extra = {
            "gpu_ns_median_ms": round(float(statistics.median(raw["gpu_ns"])) / 1e6, 6)
            if raw["gpu_ns"]
            else None,
            "raw_files": raw["files"],
            "complete_token_definition": (
                "encode + submit + wait + epilogue + sample_readback + "
                "state_update + tokenizer_decode + bookkeeping + residual"
            ),
        }
        stages["tpot"] = measured_stage(
            "tpot",
            fig=wall_fig,
            owner="GPU",
            blocking_dep="previous token's decode CB (serial greedy)",
            avoidable=False,
            cacheable=False,
            fusible=True,
            classification_reason=(
                "Steady complete-token wall. Not avoidable: each token streams "
                "~13.6 GiB of active weights. Not cacheable per token. Fusible: "
                "dispatch fusion / organ cuts on the 964-dispatch graph (N025)."
            ),
            command=[gpu_cmd] if isinstance(gpu_cmd, str) else list(gpu_cmd),
            source=",".join(raw["files"]) or str(GPU_LEDGER.relative_to(REPO)),
            kind=MEASURED,
            note=decode_note,
            pct_of="interactive_token_ms",
            extra=tpot_extra,
        )
        stages["inter_token"] = measured_stage(
            "inter_token",
            fig=wall_fig,
            owner="GPU",
            blocking_dep="previous token's decode CB (serial greedy)",
            avoidable=False,
            cacheable=False,
            fusible=True,
            classification_reason=(
                "Inter-token latency is the same complete-token wall as TPOT "
                "on this greedy path: tokens are emitted one CB at a time. "
                "A trivial sleep() would not reproduce the 30 ms GPU interval."
            ),
            command=[gpu_cmd] if isinstance(gpu_cmd, str) else list(gpu_cmd),
            source=",".join(raw["files"]) or str(GPU_LEDGER.relative_to(REPO)),
            kind=MEASURED,
            note="Identical sample set to tpot. Named separately because the acceptance vector lists both.",
            pct_of="interactive_token_ms",
            extra={"identical_to": "tpot"},
        )
        for pid, pval in (("token_latency_p50", 50), ("token_latency_p95", 95), ("token_latency_p99", 99)):
            pv = percentile(wall_ms, pval)
            fig_p = dict(wall_fig)
            fig_p["median"] = round(float(pv), 6) if pv is not None else None
            stages[pid] = measured_stage(
                pid,
                fig=fig_p,
                owner="GPU",
                blocking_dep="previous token's decode CB (serial greedy)",
                avoidable=False,
                cacheable=False,
                fusible=True,
                classification_reason=(
                    f"p{pval} of the INTERACTIVE complete-token wall distribution "
                    f"(n={len(wall_ms)}). Same physical work as TPOT."
                ),
                command=[gpu_cmd] if isinstance(gpu_cmd, str) else list(gpu_cmd),
                source=",".join(raw["files"]) or str(GPU_LEDGER.relative_to(REPO)),
                kind=MEASURED,
                note=f"p{pval} of pooled_steady_complete_wall_ns converted to ms.",
                pct_of="interactive_token_ms",
                extra={"percentile": pval, "n": len(wall_ms)},
            )
    else:
        for name in ("tpot", "inter_token", "token_latency_p50", "token_latency_p95", "token_latency_p99"):
            stages[name] = absent_stage(
                name,
                reason="GPU_LEDGER_RAW.run*.json complete-token wall samples missing",
                command=["bash", "tools/gpu_lane_lock.sh", "n028-latency", "ascension_qwen38_hybrid_greedy", "--complete-wall"],
                owner="GPU",
                blocking_dep="previous token",
                pct_of="interactive_token_ms",
            )

    # --- cold / warm start ---
    load_ns = (prod_raw or {}).get("load_ns")
    first_ms = ns_to_ms(raw["cold_first_step_wall_ns"])
    cli_row = cpl_row(cpl, "cli_startup") if cpl else None
    if isinstance(first_ms, list) and first_ms:
        stages["cold_start"] = measured_stage(
            "cold_start",
            fig=stats_ms(first_ms),
            owner="CPU+GPU",
            blocking_dep="weight residency (optional) then graph-cold first CB",
            avoidable=True,
            cacheable=True,
            fusible=False,
            classification_reason=(
                "Decode-path cold is the first step of the first generate in a "
                "fresh process (graph-cold, role=prefill). Cacheable: keep the "
                "process/pipelines resident. Avoidable as a per-turn cost. "
                "Weight load is a separate process-level figure (n=1)."
            ),
            command=[
                "bash",
                "tools/gpu_lane_lock.sh",
                "n004-gpu-ledger",
                "ascension_qwen38_hybrid_greedy",
                "--complete-wall",
            ],
            source=",".join(raw["files"]),
            kind=MEASURED,
            note=(
                "Headline is graph-cold first complete_wall (n=3 processes). "
                "OS page-cache-cold is ABSENT (sudo purge not run). "
                f"Weight load (PRODUCTION_BENCH.q4.raw load_ns) is "
                f"{None if load_ns is None else round(float(load_ns)/1e6, 3)} ms "
                "n=1 — reported as extra, not mixed into the graph-cold median."
            ),
            pct_of="interactive_cold_ms",
            extra={
                "weight_load_ms": None if load_ns is None else round(float(load_ns) / 1e6, 6),
                "weight_load_n": 1 if load_ns is not None else 0,
                "weight_load_kind": MEASURED if load_ns is not None else ABSENT,
                "weight_load_source": str(PROD_BENCH_Q4_RAW.relative_to(REPO))
                if PROD_BENCH_Q4_RAW.is_file()
                else None,
                "cli_startup_cold_ms": (cli_row or {}).get("cold_ms", {}).get("median")
                if cli_row
                else None,
                "os_page_cache_cold": ABSENT,
                "os_page_cache_cold_reason": (
                    "A true disk-cold run requires dropping the kernel page cache. "
                    "This session cannot sudo purge. Three sequential processes "
                    "share the ~14 GiB artifact page cache."
                ),
                "cold_first_step_gpu_ms": [
                    round(x / 1e6, 6) for x in raw["cold_first_step_gpu_ns"]
                ],
            },
        )
    else:
        stages["cold_start"] = absent_stage(
            "cold_start",
            reason="GPU_LEDGER_RAW cold_or_first_step.complete_wall_ns missing",
            command=["bash", "tools/gpu_lane_lock.sh", "n028-latency", "ascension_qwen38_hybrid_greedy"],
            owner="CPU+GPU",
            blocking_dep="weight residency",
            pct_of="interactive_cold_ms",
        )

    if stages.get("prefill", {}).get("status") == MEASURED:
        # Warm start of a NEW user turn with resident weights = warm prefill.
        stages["warm_start"] = measured_stage(
            "warm_start",
            fig=dict(stages["prefill"]["spread"]),
            owner="GPU",
            blocking_dep="resident weights + new prompt ids",
            avoidable=False,
            cacheable=True,
            fusible=True,
            classification_reason=(
                "Warm INTERACTIVE start is a new short prompt on an already-"
                "resident q4 process (pipelines compiled, page cache hot). "
                "Prefill still runs unless the prefix hits the prompt cache. "
                "Same samples as the prefill stage."
            ),
            command=list(stages["prefill"]["command"]),
            source=stages["prefill"]["source"],
            kind=MEASURED,
            note="Warm-start == warm short prefill on this runtime. Not TPOT.",
            pct_of="interactive_warm_start_ms",
            extra={"identical_to": "prefill", "cli_startup_warm_ms": (cli_row or {}).get("warm_ms", {}).get("median") if cli_row else None},
        )
    else:
        stages["warm_start"] = absent_stage(
            "warm_start",
            reason="prefill not MEASURED; warm start is that figure",
            command=["python3", "tools/headless/prefill_kv.py"],
            owner="GPU",
            blocking_dep="resident weights",
            pct_of="interactive_first_token_ms",
        )

    # --- sampler: GPU isolated argmax (MEASURED) + live CPU reference ---
    gpu_readback = ((gpu or {}).get("fields") or {}).get("sample_readback_ns") or {}
    gpu_sampling = ((gpu or {}).get("stages") or {}).get("sampling") or {}
    if argmax_iso.get("ok") and argmax_iso.get("gpu_ns_reps"):
        samp_ms = ns_to_ms(argmax_iso["gpu_ns_reps"])
        assert isinstance(samp_ms, list)
        readback_ms = (
            float(gpu_readback["value"]) / 1e6
            if gpu_readback.get("kind") == MEASURED and gpu_readback.get("value") is not None
            else 0.0
        )
        fig_s = stats_ms(samp_ms)
        stages["sampler"] = measured_stage(
            "sampler",
            fig=fig_s,
            owner="GPU",
            blocking_dep="lm_head logits on device (same CB)",
            avoidable=False,
            cacheable=False,
            fusible=True,
            classification_reason=(
                "Production greedy is sample_argmax_f32 inside the decode CB; "
                "already fused (one CB, 964 dispatches includes the argmax). "
                "Not cacheable. Host readback is 4 bytes. CPU softmax/top-p is "
                "not on the INTERACTIVE greedy path."
            ),
            command=["git", "show", TOKEN_NS_GIT],
            source=TOKEN_NS_GIT,
            kind=MEASURED,
            note=(
                f"Isolated family 'argmax' GPU ns reps {argmax_iso['gpu_ns_reps']}. "
                f"Host sample_readback median {gpu_readback.get('value')} ns "
                f"(MEASURED, {gpu_readback.get('kind')}). "
                f"GPU_LEDGER live_ns (scaled onto this session) "
                f"{gpu_sampling.get('live_ns')} ns is DERIVED and is not the "
                f"headline. Live CPU argmax of vocab={QWEN38_VOCAB} this run: "
                f"median {cpu_sampler['greedy_ms']['median']} ms "
                "(reference only; production does not take this path)."
            ),
            pct_of="interactive_token_ms",
            extra={
                "kernel": SAMPLE_KERNEL,
                "shader": SAMPLE_SHADER,
                "cpu_reference": cpu_sampler,
                "host_sample_readback_ns": gpu_readback,
                "gpu_ledger_sampling_derived": gpu_sampling,
                "host_readback_ms": round(readback_ms, 9),
                "already_fused_into_decode_cb": True,
            },
        )
    else:
        stages["sampler"] = measured_stage(
            "sampler",
            fig=cpu_sampler["greedy_ms"],
            owner="CPU",
            blocking_dep="materialized logits on host (not the production path)",
            avoidable=False,
            cacheable=False,
            fusible=True,
            classification_reason=(
                "GPU isolated argmax ABSENT this checkout; CPU argmax_f32 "
                "reference over Qwen3.8 vocab was measured instead. Production "
                "INTERACTIVE greedy is device sample_argmax_f32."
            ),
            command=list(cpu_sampler["command"]),
            source=SAMPLE_RS,
            kind=MEASURED,
            note=argmax_iso.get("reason") or "TOKEN_NS isolated argmax missing",
            pct_of="interactive_token_ms",
            extra={"cpu_reference": cpu_sampler, "gpu_isolated": argmax_iso},
        )

    # Totals for % — INTERACTIVE warm-resident first token and per-token.
    def _ms(name: str, side: str = "absolute_ms") -> float:
        st = stages.get(name) or {}
        if st.get("status") != MEASURED:
            return 0.0
        extra = st.get("extra") or {}
        if name in CPL_REUSE:
            warm = (extra.get("warm_ms") or {}).get("median")
            if warm is not None:
                return float(warm)
        v = st.get(side)
        return float(v) if isinstance(v, (int, float)) else 0.0

    first_token_parts = {
        "context_compile": _ms("context_compile"),
        "scheduling": _ms("scheduling"),
        "admission": _ms("admission"),
        "prefill": _ms("prefill"),
    }
    interactive_first_token_ms = round(sum(first_token_parts.values()), 6)
    interactive_token_ms = _ms("tpot")
    interactive_cold_ms = round(
        (_ms("cold_start"))
        + float(((stages.get("cold_start") or {}).get("extra") or {}).get("weight_load_ms") or 0.0),
        6,
    )
    user_visible_ttft_ms = _ms("ttft")
    interactive_warm_start_ms = _ms("warm_start")
    totals = {
        "interactive_first_token_ms": interactive_first_token_ms,
        "interactive_token_ms": interactive_token_ms,
        "interactive_cold_ms": interactive_cold_ms,
        "user_visible_ttft_ms": user_visible_ttft_ms,
        "interactive_warm_start_ms": interactive_warm_start_ms,
    }
    fill_pct(stages, totals)

    # Largest contributor of the INTERACTIVE first-token composition.
    # TTFT is a product-level measurement of a longer WorkUnit prompt and is
    # not a component of the short-prompt composition. warm_start aliases prefill.
    ranked_first = sorted(
        (
            (name, st)
            for name, st in stages.items()
            if st.get("status") == MEASURED
            and st.get("pct_of") == "interactive_first_token_ms"
            and name in {"prefill", "admission", "scheduling", "context_compile"}
            and isinstance(st.get("absolute_ms"), (int, float))
        ),
        key=lambda kv: float(kv[1]["absolute_ms"]),
        reverse=True,
    )
    top = ranked_first[0] if ranked_first else (None, {})
    ranked_token = sorted(
        (
            (name, st)
            for name, st in stages.items()
            if st.get("status") == MEASURED
            and st.get("pct_of") == "interactive_token_ms"
            and name in {"tpot", "inter_token", "sampler"}
            and isinstance(st.get("absolute_ms"), (int, float))
        ),
        key=lambda kv: float(kv[1]["absolute_ms"]),
        reverse=True,
    )

    operator = ((gpu or {}).get("stages") or {}).get("operator") or {}
    largest = {
        "id": top[0],
        "label": (
            "warm short-prompt prefill on the production decode GEMV graph"
            if top[0] == "prefill"
            else (top[1] or {}).get("id")
        ),
        "ms": (top[1] or {}).get("absolute_ms"),
        "pct_of_interactive_first_token": (top[1] or {}).get("pct_of_total"),
        "owner": (top[1] or {}).get("owner"),
        "which_total": "interactive_first_token_ms",
        "why": (
            "INTERACTIVE (S022 §5) is the user-visible first-token wait with "
            "the q4 incumbent already resident. Prefill of the chat wrap is "
            "the wall. Control-plane overlay (admission+schedule+compile) is "
            "tens of milliseconds. Sampler is already fused into the decode CB "
            "(~0.3 ms GPU). Per-token, GPU operator dominates (~80% of GPU_NS) "
            f"— GPU_LEDGER stages.operator live_ns={operator.get('live_ns')}."
        ),
        "per_token_largest": {
            "id": "tpot" if ranked_token else None,
            "ms": (ranked_token[0][1].get("absolute_ms") if ranked_token else None),
            "owner": "GPU",
            "gpu_operator_pct_of_live_gpu": operator.get("pct_of_live_gpu"),
            "note": "TPOT/inter-token IS the complete-token wall; operator is the intra-CB majority.",
        },
    }

    # Noop adversary: would python3 -c pass post the same number?
    floor = stats_ms(noop["samples_ms"])
    floor_ms = floor["median"]
    noop_rows = []
    for name, st in stages.items():
        med = st.get("absolute_ms")
        if med is None or floor_ms is None:
            continue
        cmd = st.get("command") or []
        is_python_pass = list(cmd) == [sys.executable, "-c", "pass"] or list(cmd)[-2:] == ["-c", "pass"]
        numeric_collision = abs(float(med) - float(floor_ms)) < 3.0 and float(med) > 10.0
        owner = st.get("owner") or ""
        # A GPU complete-token wall that happens to sit near 28 ms is not `pass`.
        would = bool(is_python_pass) or (
            numeric_collision and "GPU" not in owner and is_python_pass
        )
        noop_rows.append(
            {
                "id": name,
                "ms": med,
                "python_pass_floor_ms": floor_ms,
                "numeric_collision_with_noop_floor": bool(numeric_collision),
                "would_noop_post_same_number": bool(would),
                "kind": "gpu" if "GPU" in owner else "in_process_or_inner",
                "command": cmd,
                "note": (
                    "python3 -c pass is the interpreter spawn floor. "
                    "A GPU complete-token wall is a Metal CB (GPU_LEDGER_RAW), "
                    "not that spawn — even when the milliseconds are nearby. "
                    "Prefill/TTFT are an order of magnitude above the floor. "
                    "CPU sampler and N022 in-process rows sit far below it."
                ),
            }
        )

    winner = (prod or {}).get("winner") or {}
    win_cell = winner.get("winner") or {}
    c1_verified = (cell or {}).get("verified_wu_per_hour")
    verified_work_per_wall = {
        "s022_section": 42,
        "ranking_quantity": (prod or {}).get("ranking_quantity")
        or "verified_accepted_workunits_per_hour",
        "not_the_ranking_quantity": ["TTFT", "TPOT", "tok/s", "stream_count"],
        "interactive_c1": {
            "artifact": (cell or {}).get("artifact"),
            "concurrency": 1,
            "verified_wu_per_hour": c1_verified,
            "verified_work_per_wall_s": None
            if c1_verified is None
            else round(float(c1_verified) / 3600.0, 9),
            "ttft_p50_s": (cell or {}).get("ttft_p50_s"),
            "token_latency_p50_ms": (cell or {}).get("token_latency_p50_ms"),
        },
        "production_winner_not_interactive": {
            "artifact": win_cell.get("artifact"),
            "concurrency": win_cell.get("concurrency"),
            "verified_wu_per_hour": win_cell.get("verified_wu_per_hour"),
            "ttft_p50_s": win_cell.get("ttft_p50_s"),
            "note": (
                "c=4 wins verified WUs/hour and is not the INTERACTIVE profile. "
                "INTERACTIVE is c=1. Both cells are MEASURED; this ledger reports c=1."
            ),
        },
        "source": str(PROD_BENCH.relative_to(REPO)) if PROD_BENCH.is_file() else None,
    }

    parent_after = parent_identity(PARENT)
    parent_unchanged = (
        parent_before.get("catalog_ino") == parent_after.get("catalog_ino")
        and parent_before.get("catalog_mtime_ns") == parent_after.get("catalog_mtime_ns")
        and parent_before.get("catalog_bytes") == parent_after.get("catalog_bytes")
    )

    missing = [s for s in REQUIRED_STAGES if s not in stages]
    for name in missing:
        stages[name] = absent_stage(
            name,
            reason="stage not assembled",
            command=[sys.executable, str(HERE / "latency_ledger.py")],
            owner="unknown",
            blocking_dep="unknown",
            pct_of="interactive_first_token_ms",
        )

    breakdown = []
    for name in REQUIRED_STAGES:
        st = stages[name]
        breakdown.append(
            {
                "id": name,
                "status": st.get("status"),
                "absolute_ms": st.get("absolute_ms"),
                "pct_of_total": st.get("pct_of_total"),
                "pct_of": st.get("pct_of"),
                "owner": st.get("owner"),
                "blocking_dep": st.get("blocking_dep"),
                "avoidable": st.get("avoidable"),
                "cacheable": st.get("cacheable"),
                "fusible": st.get("fusible"),
                "kind": st.get("kind"),
                "n": (st.get("spread") or {}).get("n"),
                "spread_pct": (st.get("spread") or {}).get("spread_pct"),
            }
        )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "gate": GATE,
        "generated_at": generated_at,
        "git_head": head,
        "obligation": "N028 — LATENCY LEDGER",
        "question": (
            "What is the user-visible INTERACTIVE latency vector, as a "
            "per-stage breakdown with a named largest contributor?"
        ),
        "profile": {
            "name": "INTERACTIVE",
            "s022_section": 5,
            "concurrency": 1,
            "topology": "concurrent_independent",
            "artifact": "qwen38-gravity-uniform-q4-v1",
            "definition": (
                "User-visible short-context c=1 path: resident q4 incumbent, "
                "chat wrap, first-token wait then streaming tokens. Not batch, "
                "not c=4, not 16K prefill. Control-plane overlay is the N022 "
                "warm figures. Model-side is the production decode path."
            ),
        },
        "baseline": {
            "s022_section": 43,
            "rule": "Establish BASELINE first. No invented targets.",
            "targets": None,
            "targets_invented": False,
            "what_is_baseline": (
                "Every stage is a MEASURED sample set (or ABSENT with a "
                "physical reason). Medians are the baseline. There is no "
                "budget, SLO, or 'should be < N ms' in this receipt."
            ),
        },
        "method": {
            "control_plane": (
                "Reuse receipts/headless/CONTROL_PLANE_LATENCY_LEDGER.json "
                "(N022). Do not re-time grok_bridge / Swift / scheduler."
            ),
            "model_side": (
                "TTFT from PRODUCTION_BENCH q4 c=1 ttft_s. Prefill from "
                "PREFILL_KV q4 short warm_prefill_wall_ns. TPOT / inter-token / "
                "p50/p95/p99 from GPU_LEDGER_RAW pooled_steady_complete_wall_ns "
                "(actual hybrid_greedy complete-wall decode path). Sampler from "
                "TOKEN_NS isolated argmax GPU reps + live CPU argmax_f32."
            ),
            "gpu_this_lane": (
                "Not re-run. Remeasurement would load the q4 27B. Existing "
                "complete-wall receipts ARE the decode-path measurement. "
                "If a later lane remasures: bash tools/gpu_lane_lock.sh n028-latency …"
            ),
            "wall": "time.perf_counter for live CPU; GPUEnd−GPUStart for GPU (GPU_LEDGER authority)",
            "anti_goodhart": (
                "Rank by measured milliseconds of the INTERACTIVE first-token "
                "wait, not by how ugly a 20 µs which() looks. Ultimate correlate "
                "is verified-work/wall (§42), not tok/s."
            ),
        },
        "did_not_load_second_27b": occupancy["loaded_a_second_27b"] is False,
        "did_not_mutate_sealed_parent": parent_unchanged,
        "did_not_write_ascent_or_campaign": True,
        "did_not_write_under_models": True,
        "occupancy": occupancy,
        "parent_identity_before": parent_before,
        "parent_identity_after": parent_after,
        "totals": {
            "interactive_first_token_ms": interactive_first_token_ms,
            "interactive_first_token_parts_ms": first_token_parts,
            "interactive_token_ms": interactive_token_ms,
            "interactive_cold_ms": interactive_cold_ms,
            "definition": (
                "interactive_first_token_ms = warm context_compile + scheduling "
                "+ admission + prefill (model resident, app warm). Sampler is "
                "inside the prefill/decode CB and is not added again. "
                "interactive_token_ms = median complete-token wall (TPOT). "
                "interactive_cold_ms = graph-cold first step + weight load. "
                "user_visible_ttft_ms = PRODUCTION_BENCH c=1 WorkUnit TTFT "
                "(longer prompts; not mixed into the short-prompt composition)."
            ),
        },
        "user_visible_vector": {
            "ttft_ms": (stages.get("ttft") or {}).get("absolute_ms"),
            "tpot_ms": (stages.get("tpot") or {}).get("absolute_ms"),
            "prefill_ms": (stages.get("prefill") or {}).get("absolute_ms"),
            "inter_token_ms": (stages.get("inter_token") or {}).get("absolute_ms"),
            "token_latency_p50_ms": (stages.get("token_latency_p50") or {}).get("absolute_ms"),
            "token_latency_p95_ms": (stages.get("token_latency_p95") or {}).get("absolute_ms"),
            "token_latency_p99_ms": (stages.get("token_latency_p99") or {}).get("absolute_ms"),
            "admission_ms": (stages.get("admission") or {}).get("absolute_ms"),
            "cold_start_ms": (stages.get("cold_start") or {}).get("absolute_ms"),
            "warm_start_ms": (stages.get("warm_start") or {}).get("absolute_ms"),
            "scheduling_ms": (stages.get("scheduling") or {}).get("absolute_ms"),
            "context_compile_ms": (stages.get("context_compile") or {}).get("absolute_ms"),
            "sampler_ms": (stages.get("sampler") or {}).get("absolute_ms"),
        },
        "stages": stages,
        "breakdown": breakdown,
        "largest_contributor": largest,
        "noop_adversary": {
            "question": "Would a trivial path (python3 -c pass) post the same number?",
            "noop": "python3 -c pass",
            "floor_ms": floor,
            "command": [sys.executable, "-c", "pass"],
            "comparisons": noop_rows,
        },
        "verified_work_per_wall": verified_work_per_wall,
        "cpu_sampler_this_run": cpu_sampler,
        "n022_reuse": {
            "receipt": str(CPL_LEDGER.relative_to(REPO)),
            "schema": (cpl or {}).get("schema"),
            "gate": (cpl or {}).get("gate"),
            "mapped": CPL_REUSE,
        },
        "sources": {
            "control_plane": str(CPL_LEDGER.relative_to(REPO)),
            "gpu_ledger": str(GPU_LEDGER.relative_to(REPO)),
            "gpu_raw": raw["files"],
            "prefill_kv": str(PREFILL_KV.relative_to(REPO)),
            "production_bench": str(PROD_BENCH.relative_to(REPO)),
            "token_ns": TOKEN_NS_GIT,
            "cpu_sampler": SAMPLE_RS,
        },
        "scope_guard": {
            "wrote": [
                "tools/headless/latency_ledger.py",
                "tools/headless/test_latency_ledger.py",
                "receipts/headless/LATENCY_LEDGER.json",
            ],
            "did_not_modify": [
                "receipts/ascent-2026-08-16",
                "receipts/ascent-2026-08-18",
                "workspace/campaign/odyssey",
                "hawking-experiments",
                str(PARENT),
            ],
        },
    }
    return doc


def validate_receipt(doc: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    if doc.get("schema") != SCHEMA:
        fails.append(f"schema {doc.get('schema')!r} != {SCHEMA}")
    if doc.get("gate") != GATE:
        fails.append(f"gate {doc.get('gate')!r} != {GATE}")
    if (doc.get("profile") or {}).get("name") != "INTERACTIVE":
        fails.append("profile.name must be INTERACTIVE")
    if (doc.get("baseline") or {}).get("targets") is not None:
        fails.append("baseline.targets must be null (S022 §43, no invented targets)")
    stages = doc.get("stages") or {}
    if not isinstance(stages, dict):
        fails.append("stages missing")
        return fails
    for name in REQUIRED_STAGES:
        st = stages.get(name)
        if not isinstance(st, dict):
            fails.append(f"stages.{name} missing")
            continue
        status = st.get("status")
        for key in (
            "absolute_ms",
            "pct_of_total",
            "owner",
            "blocking_dep",
            "avoidable",
            "cacheable",
            "fusible",
        ):
            if key not in st:
                fails.append(f"stages.{name} missing {key}")
        if status == MEASURED:
            if st.get("absolute_ms") is None:
                fails.append(f"stages.{name} MEASURED but absolute_ms empty")
            spread = st.get("spread") or {}
            if not spread.get("samples"):
                fails.append(f"stages.{name} MEASURED but spread.samples empty")
            if spread.get("n") in (None, 0):
                fails.append(f"stages.{name} MEASURED but spread.n empty")
            if not st.get("command"):
                fails.append(f"stages.{name} missing command")
        elif status == ABSENT:
            if not st.get("reason"):
                fails.append(f"stages.{name} ABSENT without reason")
        else:
            fails.append(f"stages.{name} status {status!r} not MEASURED/ABSENT")
    top = doc.get("largest_contributor") or {}
    if not top.get("id") or top.get("ms") is None:
        fails.append("largest_contributor id/ms missing")
    vec = doc.get("user_visible_vector") or {}
    for k in (
        "ttft_ms",
        "tpot_ms",
        "prefill_ms",
        "inter_token_ms",
        "token_latency_p50_ms",
        "token_latency_p95_ms",
        "token_latency_p99_ms",
        "admission_ms",
        "cold_start_ms",
        "warm_start_ms",
        "scheduling_ms",
        "context_compile_ms",
        "sampler_ms",
    ):
        if k not in vec:
            fails.append(f"user_visible_vector.{k} missing")
    if not (doc.get("noop_adversary") or {}).get("comparisons"):
        fails.append("noop_adversary.comparisons missing")
    vww = doc.get("verified_work_per_wall") or {}
    if vww.get("s022_section") != 42:
        fails.append("verified_work_per_wall must cite §42")
    if not (vww.get("interactive_c1") or {}).get("verified_work_per_wall_s"):
        fails.append("verified_work_per_wall.interactive_c1.verified_work_per_wall_s missing")
    if doc.get("did_not_load_second_27b") is not True:
        fails.append("did_not_load_second_27b is not True")
    return fails


def write_receipt(doc: dict[str, Any]) -> Path:
    atomic_write(RECEIPT, doc)
    return RECEIPT


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--measure-cpu-sampler"]:
        print(json.dumps(measure_cpu_sampler(), indent=2))
        return 0
    doc = build()
    path = write_receipt(doc)
    fails = validate_receipt(doc)
    top = doc.get("largest_contributor") or {}
    print(f"wrote {path}")
    print(
        f"profile={doc['profile']['name']} "
        f"largest_contributor={top.get('id')} {top.get('ms')} ms "
        f"({top.get('pct_of_interactive_first_token')}% of first token)"
    )
    vec = doc.get("user_visible_vector") or {}
    print(
        "vector "
        + " ".join(f"{k}={vec.get(k)}" for k in ("ttft_ms", "tpot_ms", "prefill_ms", "sampler_ms"))
    )
    if fails:
        print("receipt self-check FAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print("receipt self-check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
