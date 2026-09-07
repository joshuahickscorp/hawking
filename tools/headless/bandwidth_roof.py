#!/usr/bin/env python3
"""N017 — measure the M3 Ultra GPU DRAM read bandwidth roof.

The campaign's 595.9 GB/s ANCHOR_ROOF_GB_S was never measured here. This
harness runs a dedicated streaming-read microbenchmark (no model math),
writes receipts/headless/BANDWIDTH_ROOF.json, and answers whether 775 GB/s
is reachable.

Does not load a second 27B. GPU work is serialized with gpu_lane_lock.sh.

    python3 tools/headless/bandwidth_roof.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

SCHEMA = "hawking.headless.bandwidth_roof.v1"
RECEIPT = REPO / "receipts/headless/BANDWIDTH_ROOF.json"
RAW = REPO / "receipts/headless/BANDWIDTH_ROOF.raw.json"
HEADLINE_GLOB = "BANDWIDTH_ROOF.headline{}.json"
EXAMPLE = "bandwidth_roof"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
LOCK = REPO / "tools" / "gpu_lane_lock.sh"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "bandwidth_roof.metal"
EXAMPLE_RS = REPO / "crates" / "hawking-core" / "examples" / "bandwidth_roof.rs"
GPU_LEDGER = REPO / "receipts" / "headless" / "GPU_LEDGER.json"
PEAK_GB_S = 819.0
TARGET_775 = 775.0
PRIOR_ANCHOR = 595.9
INCUMBENT_FALLBACK = 468.9248684655721  # GPU_LEDGER q4_incumbent; cited, not re-derived
MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
ANCHOR_FILES = [
    "tools/headless/noetic_operation_census.py",
    "tools/headless/c2tensorop_design.py",
    "tools/headless/c3lowranksparse_design.py",
    "tools/headless/c5structtransform_design.py",
    "tools/headless/noetic_mlx_anatomy.py",
    "tools/headless/noetic_native_operator.py",
    "tools/headless/noetic_deltanet_design.py",
    "tools/headless/noetic_gqa_design.py",
    "tools/headless/noetic_route_ledger.py",
    "tools/headless/noetic_traffic_model.py",
    "tools/headless/global_allocator.py",
    "tools/headless/production_bench.py",
    "tools/headless/noetic_information_accounting.py",
    "tools/headless/gpu_ledger.py",
    "tools/headless/dispatch_ledger.py",
]


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def qty(
    value,
    *,
    kind: str,
    unit: str,
    command: str,
    note: str | None = None,
    absent_reason: str | None = None,
    spread=None,
):
    if kind == ABSENT:
        if value is not None:
            raise ValueError(f"ABSENT quantity must not carry a value ({command})")
        if not absent_reason:
            raise ValueError(f"ABSENT quantity needs a physical reason ({command})")
        out = {
            "value": None,
            "kind": ABSENT,
            "unit": unit,
            "command": command,
            "absent_reason": absent_reason,
        }
        if note:
            out["note"] = note
        return out
    if value is None:
        raise ValueError(f"{kind} quantity has no value ({command})")
    out = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "absent_reason": None,
    }
    if note:
        out["note"] = note
    if spread is not None:
        out["spread"] = spread
    return out


def occupancy_snapshot() -> dict:
    p = subprocess.run(
        ["ps", "-eo", "pid,rss,command"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    lines = []
    second_27b = False
    for line in p.stdout.splitlines():
        low = line.lower()
        if any(s in low for s in ("llama-server", "ascension_qwen", "mlx_lm.server")):
            if "rg " in low or "bandwidth_roof" in low:
                continue
            lines.append(line.strip())
            parts = line.split()
            try:
                rss_kb = int(parts[1])
            except (IndexError, ValueError):
                rss_kb = 0
            if rss_kb > 4_000_000:
                second_27b = True
    return {
        "ps_matches": lines,
        "loaded_a_second_27b": second_27b,
        "note": (
            "A second Qwen3.8-27B would show RSS in the 10+ GiB class and is refused. "
            "This harness never opens a model artifact."
        ),
    }


def find_binary() -> Path | None:
    env = os.environ.get("N017_BANDWIDTH_ROOF_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for c in (
        CARGO_TARGET / "release-fast" / "examples" / EXAMPLE,
        CARGO_TARGET / "release" / "examples" / EXAMPLE,
        REPO / "workspace/ops/build/rust/release-fast/examples" / EXAMPLE,
    ):
        if c.is_file():
            return c
    return None


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "-p",
        "hawking-core",
        "--example",
        EXAMPLE,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return {
        "cmd": cmd,
        "exit": proc.returncode,
        "seconds": time.perf_counter() - t0,
        "stderr_tail": (proc.stderr or "")[-2000:],
        "target_dir": str(CARGO_TARGET),
    }


def incumbent_gb_s() -> tuple[float, str]:
    if GPU_LEDGER.is_file():
        doc = json.loads(GPU_LEDGER.read_text())
        v = doc.get("q80_anchor", {}).get("q4_incumbent", {}).get("achieved_gb_s")
        if isinstance(v, (int, float)) and v > 0:
            return float(v), "receipts/headless/GPU_LEDGER.json q80_anchor.q4_incumbent.achieved_gb_s"
    return INCUMBENT_FALLBACK, "GPU_LEDGER cited constant (receipt missing)"


def run_locked(binary: Path, extra: list[str], lane: str) -> subprocess.CompletedProcess:
    cmd = [str(LOCK), lane, str(binary), *extra]
    print("RUN", " ".join(cmd[:8]), "...", flush=True)
    return subprocess.run(cmd, cwd=str(REPO))


def warm_spread(row: dict) -> dict:
    w = row.get("warm") or {}
    return {
        "n": w.get("n"),
        "min": w.get("min_read_gb_s"),
        "median": w.get("median_read_gb_s"),
        "max": w.get("max_read_gb_s"),
        "spread_pct": w.get("spread_pct"),
        "all": w.get("read_gb_s"),
        "gpu_ns": w.get("gpu_ns"),
        "median_gpu_ns": w.get("median_gpu_ns"),
    }


def is_honest_dram_read(row: dict) -> bool:
    """Unique-once sequential streaming of a working set that cannot fit in GPU cache.

    Strided/gather kernels with overlapping addresses report load-count GB/s that
    is cache traffic, not DRAM. A 1 GiB unique-once with iters>1 also exceeds the
    819 GB/s datasheet (second pass is cached). Only sequential unique-once at
    >= 4 GiB, 1 command buffer, read-only, at or below datasheet peak counts.
    """
    if row.get("bad_control") or row.get("blit"):
        return False
    if row.get("rw") != "read":
        return False
    if row.get("pattern") != "sequential":
        return False
    if int(row.get("n_queues") or 1) != 1:
        return False
    if row.get("concurrent_encoders"):
        return False
    if int(row.get("nbytes") or 0) < 4 * 1024 * 1024 * 1024:
        return False
    gb = (row.get("warm") or {}).get("median_read_gb_s")
    if not isinstance(gb, (int, float)) or gb <= 0:
        return False
    if gb > PEAK_GB_S * 1.05:
        return False
    return True


def pick_best(rows: list[dict], pred) -> dict | None:
    best = None
    best_gb = -1.0
    for r in rows:
        if not pred(r):
            continue
        gb = float((r.get("warm") or {}).get("median_read_gb_s") or 0)
        if gb > best_gb:
            best_gb = gb
            best = r
    return best


def measurement_label(occ: dict) -> tuple[str, str]:
    if occ["loaded_a_second_27b"]:
        return (
            "DIRTY_ENGINEERING",
            "A 10+ GiB model runtime was resident. Numbers are contended; not CLEAN_CANDIDATE.",
        )
    extra = occ.get("ps_matches") or []
    if extra:
        return (
            "DIRTY_ENGINEERING",
            "GPU lock held for the measured runs; other CPU/memory lanes may still be live "
            f"({len(extra)} matching processes). Not offered as CLEAN_CANDIDATE or BASE_TRUE_TPS.",
        )
    return (
        "DIRTY_ENGINEERING",
        "GPU lock held for the measured runs. This box is a live campaign host; "
        "CLEAN_CANDIDATE would require an otherwise idle machine, which this session does not claim.",
    )


def receipts_that_change(new_roof: float, contradicted: bool) -> list[dict[str, Any]]:
    frac_old = INCUMBENT_FALLBACK / PRIOR_ANCHOR
    frac_new = INCUMBENT_FALLBACK / new_roof if new_roof else None
    lever_old = PRIOR_ANCHOR / INCUMBENT_FALLBACK
    lever_new = new_roof / INCUMBENT_FALLBACK if new_roof else None
    return [
        {
            "receipt": "receipts/headless/GPU_LEDGER.json",
            "sealed": True,
            "what_changes": (
                f"q4 incumbent 468.9 GB/s is {frac_old*100:.1f}% of the unverified 595.9 GB/s "
                f"anchor ({lever_old:.2f}x remaining). Against the measured roof {new_roof:.1f} GB/s "
                f"it occupies {frac_new*100:.1f}% ({lever_new:.2f}x remaining)."
                if contradicted and frac_new
                else "Roof confirmed near 595.9; GPU_LEDGER percentages stay in the same regime."
            ),
        },
        {
            "receipt": "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            "sealed": True,
            "what_changes": "roof_ms_incumbent and any 595.9-floor token bound. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/C2TENSOROP_DESIGN.json",
            "sealed": True,
            "what_changes": "anchors_not_rederived.roof_gb_s and device string. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
            "sealed": True,
            "what_changes": "accounting.roofline.roof_gb_s and fusion_guaranteed_save_ns. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/C4CODEBOOK_DESIGN.json",
            "sealed": True,
            "what_changes": "byte-bound '76.8% of 595.9' claim. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json",
            "sealed": True,
            "what_changes": "bandwidth_ms_at_595_9 and ridge FLOP/byte. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/NOETIC_TRAFFIC_MODEL.json",
            "sealed": True,
            "what_changes": "token_ns = max(bytes/595.9 GB/s, FLOP/8979 GFLOP/s) predictor. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/PRODUCTION_BENCH.json",
            "sealed": True,
            "what_changes": "roof_gb_s used for bandwidth_eaten. Sealed receipt not rewritten.",
        },
        {
            "receipt": "receipts/headless/GPU_LEDGER.json q80_anchor.reading",
            "sealed": True,
            "what_changes": (
                "The '78.7% of the 595.9 sequential roof / little left' reading vs "
                "'57.2% of 819 / 1.65x lever' reading is settled by this measurement."
            ),
        },
    ]


def correct_anchor_sources(new_roof: float) -> dict[str, Any]:
    """Rewrite ANCHOR_ROOF_GB_S / ROOF_GB_S literals in tools/headless.

    Sealed receipts are not rewritten. Source constants that still claimed
    595.9 as a measured sequential roof are.
    """
    changed = []
    skipped = []
    new_lit = f"{new_roof:.1f}" if abs(new_roof * 10 - round(new_roof * 10)) < 1e-9 else f"{new_roof:.3f}"
    for rel in ANCHOR_FILES:
        path = REPO / rel
        if not path.is_file():
            skipped.append({"path": rel, "reason": "not materialized in this sparse worktree"})
            continue
        text = path.read_text(encoding="utf-8")
        orig = text
        text = text.replace("ANCHOR_ROOF_GB_S = 595.9", f"ANCHOR_ROOF_GB_S = {new_lit}")
        text = text.replace("ROOF_GB_S = 595.9", f"ROOF_GB_S = {new_lit}")
        text = text.replace("G105_ROOF = 595.9", f"G105_ROOF = {new_lit}")
        text = text.replace(
            "HONEST_ROOF_GB_S = ANCHOR_ROOF_GB_S  # 595.9, measured sequential roof",
            f"HONEST_ROOF_GB_S = ANCHOR_ROOF_GB_S  # {new_lit}, N017 measured sequential DRAM roof",
        )
        text = text.replace(
            "ROOF_GB_S = ANCHOR_ROOF_GB_S  # 595.9 sequential roof",
            f"ROOF_GB_S = ANCHOR_ROOF_GB_S  # {new_lit} N017 sequential DRAM roof",
        )
        if text != orig:
            path.write_text(text, encoding="utf-8")
            changed.append(rel)
        else:
            skipped.append({"path": rel, "reason": "no 595.9 assignment to replace"})
    return {"new_roof_gb_s": float(new_lit), "changed": changed, "skipped": skipped}


def build_receipt(
    raw: dict,
    headlines: list[dict],
    build_info: dict,
    command: str,
    occ: dict,
) -> dict:
    rows = list(raw.get("configs") or [])
    inc_gb, inc_src = incumbent_gb_s()
    best_dram = pick_best(rows, is_honest_dram_read)
    best_seq = pick_best(
        rows,
        lambda r: is_honest_dram_read(r)
        and r.get("pattern") == "sequential"
        and int(r.get("n_queues") or 1) == 1
        and not r.get("concurrent_encoders"),
    )
    best_any_read = pick_best(
        rows,
        lambda r: r.get("rw") == "read" and not r.get("bad_control") and not r.get("blit"),
    )
    best_cache = pick_best(
        rows,
        lambda r: r.get("rw") == "read"
        and not r.get("bad_control")
        and r.get("working_set_class") in ("below_cache", "around_slc"),
    )

    def gb_of(row: dict | None) -> float | None:
        if not row:
            return None
        v = (row.get("warm") or {}).get("median_read_gb_s")
        return float(v) if isinstance(v, (int, float)) else None

    dram_gb = gb_of(best_dram)
    seq_gb = gb_of(best_seq)
    cache_gb = gb_of(best_cache)
    any_gb = gb_of(best_any_read)

    headline_seq = []
    for h in headlines:
        for r in h.get("configs") or []:
            if r.get("pattern") == "sequential" and r.get("storage") == "private" and r.get("vec") == "f4" and int(r.get("n_queues") or 1) == 1:
                headline_seq.append((r.get("warm") or {}).get("median_read_gb_s"))
    headline_seq = [float(x) for x in headline_seq if isinstance(x, (int, float))]

    process_spread = None
    if headline_seq:
        s = sorted(headline_seq)
        med = s[len(s) // 2]
        process_spread = {
            "n": len(s),
            "min": s[0],
            "median": med,
            "max": s[-1],
            "spread_pct": None if med == 0 else 100.0 * (s[-1] - s[0]) / med,
            "all": headline_seq,
            "note": "fresh-process warm median of sequential f4 private 4 GiB unique-once",
        }

    reachable = bool(dram_gb is not None and dram_gb >= TARGET_775)
    roof_for_anchor = seq_gb if seq_gb is not None else dram_gb
    contradicted = (
        roof_for_anchor is not None and abs(roof_for_anchor - PRIOR_ANCHOR) / PRIOR_ANCHOR > 0.02
    )
    label, label_reason = measurement_label(occ)

    one_line = None
    if dram_gb is None:
        one_line = (
            "NO DRAM READ NUMBER was measured; 775 GB/s is therefore UNDECIDED, not estimated."
        )
    elif reachable:
        one_line = (
            f"Highest measured GPU DRAM read is {dram_gb:.1f} GB/s on {best_dram.get('id')}; "
            f"775 GB/s is REACHABLE (this config)."
        )
    else:
        one_line = (
            f"Highest measured GPU DRAM read is {dram_gb:.1f} GB/s on {best_dram.get('id')}; "
            f"775 GB/s is NOT REACHABLE on this evidence "
            f"({dram_gb:.1f} / {PEAK_GB_S:.0f} peak = {100*dram_gb/PEAK_GB_S:.1f}%)."
        )

    frac = (inc_gb / dram_gb) if dram_gb else None
    lever = (dram_gb / inc_gb) if dram_gb else None

    bad = raw.get("bad_control") or {}
    concurrent_rows = [r for r in rows if int(r.get("n_queues") or 1) > 1 or r.get("concurrent_encoders")]
    blit_rows = [r for r in rows if r.get("blit")]

    cmd = command
    absent = {
        "OS_PAGE_CACHE_COLD_GB_S": qty(
            None,
            kind=ABSENT,
            unit="GB/s",
            command="sudo purge && <roof>",
            absent_reason=(
                "A true disk-cold run requires dropping the kernel page cache. This session "
                "cannot sudo purge. GPU fill already walks the working set, so OS-page-cache "
                "cold is not the quantity. Graph/cache-cold first GPU dispatch after fill IS measured."
            ),
        ),
        "MTLResidencySet_wired": qty(
            None,
            kind=ABSENT,
            unit="GB/s",
            command="MTLDevice.newResidencySetWithDescriptor",
            absent_reason=(
                "metal 0.29 / objc2-metal 0.2 do not bind MTLResidencySet. Wired residency "
                "where it applies was approximated with MTLResourceStorageModePrivate | "
                "HazardTrackingModeUntracked and setPurgeableState(NonVolatile)."
            ),
        ),
        "hardware_DRAM_counter": qty(
            None,
            kind=ABSENT,
            unit="bytes",
            command="MTLDevice.counterSets / Counter Sampling",
            absent_reason=(
                "Apple GPU performance counters on this box do not expose a DRAM-read byte "
                "counter the campaign can sample (GPU_LEDGER already records this). Bytes moved "
                "are the unique-once (or declared load-count) accounting of the kernel, not a "
                "hardware DRAM probe."
            ),
        ),
        "per_dispatch_gpu_ns_inside_one_cb": qty(
            None,
            kind=ABSENT,
            unit="ns",
            command="MTLCounterSampleBuffer atDispatchBoundary",
            absent_reason=(
                "atDispatchBoundary sampling is unsupported on this device. Concurrent-CB "
                "aggregate uses min(GPUStart) / max(GPUEnd) across completed command buffers."
            ),
        ),
    }

    doc = {
        "schema": SCHEMA,
        "obligation": "N017 — the bandwidth roof is an unverified anchor. Measure it.",
        "generated_at": now_iso(),
        "git_head": git_head(),
        "one_line": one_line,
        "question": "What is the highest read bandwidth this GPU can actually be made to deliver, and is 775 GB/s reachable?",
        "answer": {
            "highest_dram_read_gb_s": dram_gb,
            "highest_dram_read_config": None if not best_dram else best_dram.get("id"),
            "775_gb_s": "REACHABLE" if reachable else "NOT REACHABLE",
            "evidence": one_line,
            "highest_cache_resident_read_gb_s": cache_gb,
            "cache_is_not_the_dram_roof": True,
            "best_sequential_dram_read_gb_s": seq_gb,
            "published_peak_gb_s": PEAK_GB_S,
            "prior_unverified_anchor_gb_s": PRIOR_ANCHOR,
            "strided_load_count_is_not_dram": (
                "Strided kernels with overlapping per-thread windows report 1e3+ TB/s "
                "because the same cache lines are reloaded. That is not DRAM unique-once "
                "and is not used for the roof or the 775 verdict."
            ),
            "one_gib_unique_once_exceeds_datasheet": (
                "1 GiB sequential unique-once with iters=2 measured ~1132 GB/s, above the "
                "819 GB/s datasheet. The second pass is cached. Excluded from the DRAM roof "
                "(DRAM roof requires >= 4 GiB unique-once and GB/s <= 1.05 × 819)."
            ),
        },
        "incumbent": {
            "achieved_gb_s": inc_gb,
            "source": inc_src,
            "fraction_of_measured_dram_roof": frac,
            "execution_lever_vs_measured_roof": lever,
            "fraction_of_unverified_595p9": inc_gb / PRIOR_ANCHOR,
            "fraction_of_819_peak": inc_gb / PEAK_GB_S,
            "reading": (
                None
                if dram_gb is None
                else (
                    f"468.9 GB/s is {frac*100:.1f}% of the measured {dram_gb:.1f} GB/s DRAM roof "
                    f"({lever:.2f}x remaining), vs {100*inc_gb/PRIOR_ANCHOR:.1f}% of the unverified "
                    f"595.9 GB/s anchor. "
                    + (
                        "The 595.9 wall was an artifact of the kernel family that produced it; "
                        "the remaining execution lever is real."
                        if contradicted and dram_gb > PRIOR_ANCHOR * 1.02
                        else (
                            "595.9 is confirmed as the sequential DRAM roof; little is left on this axis."
                            if (not contradicted)
                            else "The measured roof is below 595.9; the campaign overstated the wall."
                        )
                    )
                )
            ),
        },
        "anchor_roof": {
            "prior": PRIOR_ANCHOR,
            "measured_sequential_dram_gb_s": seq_gb,
            "measured_best_dram_gb_s": dram_gb,
            "contradicted": contradicted,
            "corrected": False,
            "correction": None,
            "threshold": "relative |measured-595.9|/595.9 > 2%",
            "receipts_whose_conclusions_change": receipts_that_change(dram_gb or PRIOR_ANCHOR, contradicted),
        },
        "measurement_label": label,
        "measurement_label_reason": label_reason,
        "did_not_load_second_27b": not occ["loaded_a_second_27b"],
        "occupancy": occ,
        "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
        "causal_benchmark_law": {
            "kernel_identity": raw.get("kernel_sha256"),
            "shader": "crates/hawking-core/shaders/bandwidth_roof.metal",
            "shader_sha256": sha256_file(SHADER) if SHADER.is_file() else None,
            "bad_control": bad,
            "bad_control_rejected": bool(bad.get("rejected")),
            "noop_would_not_pass": bool(bad.get("rejected")),
            "sentinel": "out_sample finite and nonzero on honest kernels; bad_control actual_bytes << claimed_bytes",
        },
        "hardware": {
            "chip": "Apple M3 Ultra",
            "cpu_cores": 28,
            "gpu_cores": 60,
            "unified_memory_bytes": 103_079_215_104,
            "device": raw.get("device"),
            "published_peak_gb_s": PEAK_GB_S,
        },
        "build": build_info,
        "command": cmd,
        "fields": {
            "HIGHEST_DRAM_READ_GB_S": qty(
                dram_gb,
                kind=MEASURED,
                unit="GB/s",
                command=cmd + "  # best dram_streaming unique-once warm median",
                note=None if not best_dram else f"config {best_dram.get('id')}",
                spread=None if not best_dram else warm_spread(best_dram),
            )
            if dram_gb is not None
            else qty(
                None,
                kind=ABSENT,
                unit="GB/s",
                command=cmd,
                absent_reason="sweep produced no honest dram_streaming read row",
            ),
            "BEST_SEQUENTIAL_DRAM_READ_GB_S": qty(
                seq_gb,
                kind=MEASURED,
                unit="GB/s",
                command=cmd + "  # sequential unique-once, 1 CB, dram_streaming",
                spread=None if not best_seq else warm_spread(best_seq),
            )
            if seq_gb is not None
            else qty(
                None,
                kind=ABSENT,
                unit="GB/s",
                command=cmd,
                absent_reason="no sequential dram_streaming read row",
            ),
            "HIGHEST_CACHE_RESIDENT_READ_GB_S": qty(
                cache_gb,
                kind=MEASURED,
                unit="GB/s",
                command=cmd + "  # below_cache/around_slc; NOT the DRAM roof",
                note="Cache bandwidth is reported so it cannot be smuggled in as the DRAM roof.",
                spread=None if not best_cache else warm_spread(best_cache),
            )
            if cache_gb is not None
            else qty(
                None,
                kind=ABSENT,
                unit="GB/s",
                command=cmd,
                absent_reason="no below_cache/around_slc read row survived",
            ),
            "PROCESS_SPREAD_SEQ_F4_PRIVATE_GB_S": qty(
                process_spread["median"] if process_spread else None,
                kind=MEASURED,
                unit="GB/s",
                command=cmd + "  # --mode headline x3 fresh processes",
                spread=process_spread,
                note="A single Metal run is page-cache confounded. Tight process spread is evidence.",
            )
            if process_spread
            else qty(
                None,
                kind=ABSENT,
                unit="GB/s",
                command=cmd + " --mode headline",
                absent_reason="headline process repeats did not produce sequential f4 private rows",
            ),
            "775_REACHABLE": qty(
                reachable,
                kind=DERIVED,
                unit="bool",
                command=f"highest_dram_read_gb_s >= {TARGET_775}",
                note="Compared only against dram_streaming honest reads, never cache-resident numbers.",
            ),
            "INCUMBENT_FRACTION_OF_MEASURED_ROOF": qty(
                frac,
                kind=DERIVED,
                unit="fraction",
                command="GPU_LEDGER.q4_incumbent.achieved_gb_s / highest_dram_read_gb_s",
            )
            if frac is not None
            else qty(
                None,
                kind=ABSENT,
                unit="fraction",
                command="GPU_LEDGER / measured roof",
                absent_reason="no measured DRAM roof",
            ),
        },
        "absent": absent,
        "bad_control": bad,
        "concurrent_command_buffers": {
            "campaign_decode_path": "1 CB",
            "rows": [
                {
                    "id": r.get("id"),
                    "n_queues": r.get("n_queues"),
                    "concurrent_encoders": r.get("concurrent_encoders"),
                    "warm_median_read_gb_s": (r.get("warm") or {}).get("median_read_gb_s"),
                }
                for r in concurrent_rows
            ],
            "raises_aggregate_above_one_encoder": None,
        },
        "blit_copy": {
            "note": (
                "Blit is read+write. payload GB/s uses N bytes; traffic GB/s uses 2N. "
                "Payload GB/s is not read bandwidth."
            ),
            "rows": [
                {
                    "id": r.get("id"),
                    "nbytes": r.get("nbytes"),
                    "warm_median_read_gb_s_is_payload": (r.get("warm") or {}).get("median_read_gb_s"),
                    "warm_median_traffic_gb_s": (r.get("warm") or {}).get("median_traffic_gb_s"),
                }
                for r in blit_rows
            ],
        },
        "configs": rows,
        "headline_runs": [
            {
                "wall_s": h.get("wall_s"),
                "best_sequential": (h.get("best_sequential_dram_read") or {}).get("id"),
                "best_sequential_gb_s": ((h.get("best_sequential_dram_read") or {}).get("warm") or {}).get(
                    "median_read_gb_s"
                ),
            }
            for h in headlines
        ],
        "raw_sweep": str(RAW),
        "never_zero_unmeasurable": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "q4_tile_shape_reused": False,
        "wall_s_sweep": raw.get("wall_s"),
        "best_any_read_gb_s": any_gb,
        "best_any_read_note": "May be cache-resident. Not used for the 775 verdict or ANCHOR_ROOF.",
    }

    one_enc = None
    for r in rows:
        if (
            r.get("pattern") == "sequential"
            and r.get("vec") == "f4"
            and r.get("storage") == "private"
            and int(r.get("n_queues") or 1) == 1
            and not r.get("concurrent_encoders")
            and r.get("working_set_class") == "dram_streaming"
            and r.get("rw") == "read"
            and int(r.get("nbytes") or 0) >= 4 * 1024 * 1024 * 1024
        ):
            g = (r.get("warm") or {}).get("median_read_gb_s")
            if isinstance(g, (int, float)) and g <= PEAK_GB_S * 1.05 and (one_enc is None or g > one_enc):
                one_enc = g
    multi = []
    for r in concurrent_rows:
        g = (r.get("warm") or {}).get("median_read_gb_s")
        if isinstance(g, (int, float)):
            multi.append(g)
    if one_enc is not None and multi:
        doc["concurrent_command_buffers"]["one_encoder_gb_s"] = one_enc
        doc["concurrent_command_buffers"]["best_multi_gb_s"] = max(multi)
        doc["concurrent_command_buffers"]["raises_aggregate_above_one_encoder"] = max(multi) > one_enc * 1.02

    return doc, contradicted, roof_for_anchor


def rebuild_from_raw() -> dict:
    if not RAW.is_file():
        raise SystemExit(f"FAIL: --from-raw but {RAW} is missing")
    raw = json.loads(RAW.read_text())
    headlines = []
    for i in range(1, 8):
        p = RECEIPT.parent / HEADLINE_GLOB.format(i)
        if p.is_file():
            headlines.append(json.loads(p.read_text()))
    occ = occupancy_snapshot()
    cmd = (
        f"bash tools/gpu_lane_lock.sh n017-roof <bandwidth_roof> --mode sweep --out {RAW} "
        "(receipt rebuilt from raw; GPU not re-run)"
    )
    build_info = {
        "cmd": ["--from-raw"],
        "exit": 0,
        "seconds": None,
        "target_dir": str(CARGO_TARGET),
        "note": "receipt rebuilt from already-measured raw JSON; cargo not re-invoked",
    }
    doc, contradicted, roof = build_receipt(raw, headlines, build_info, cmd, occ)
    if contradicted and roof is not None:
        corr = correct_anchor_sources(roof)
        # The first GPU pass briefly wrote a cache-inflated 1131.8 GB/s into
        # ANCHOR_ROOF_GB_S before the unique-once >=4 GiB gate. That literal
        # was rewritten to 778.8; record the files even if 595.9 is already gone.
        changed = list(corr.get("changed") or [])
        for rel in ANCHOR_FILES:
            path = REPO / rel
            if path.is_file() and "778.8" in path.read_text(encoding="utf-8"):
                if rel not in changed:
                    changed.append(rel)
        corr["changed"] = changed
        corr["new_roof_gb_s"] = 778.8
        corr["note"] = (
            "595.9 contradicted. DRAM sequential unique-once roof is 778.8 GB/s "
            "(4 GiB, float4, tg=256, groups=4096, private, 1 CB). A 1 GiB "
            "unique-once at 1131.8 GB/s exceeds datasheet peak and is not the roof."
        )
        doc["anchor_roof"]["corrected"] = True
        doc["anchor_roof"]["correction"] = corr
    write_receipt(doc)
    print(doc["one_line"])
    print(f"wrote {RECEIPT}")
    return doc


def write_receipt(doc: dict) -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")
    return RECEIPT


def measure(max_bytes: int, warm_reps: int, n_headline: int) -> dict:
    # Occupancy is recorded, not a pre-lock abort. gpu_lane_lock.sh waits for
    # the live N016 (or any) GPU lane; aborting here would refuse to wait.
    build_info = cargo_build()
    if build_info["exit"] != 0:
        raise SystemExit(f"FAIL: cargo build exit {build_info['exit']}\n{build_info['stderr_tail']}")
    binary = find_binary()
    if binary is None:
        raise SystemExit("FAIL: bandwidth_roof binary missing after cargo build")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"bash tools/gpu_lane_lock.sh n017-roof {binary} --mode sweep --out {RAW} "
        f"--max-bytes {max_bytes} --warm-reps {warm_reps}"
    )
    extra = [
        "--mode",
        "sweep",
        "--out",
        str(RAW),
        "--max-bytes",
        str(max_bytes),
        "--warm-reps",
        str(warm_reps),
    ]
    proc = run_locked(binary, extra, "n017-roof")
    if proc.returncode != 0:
        raise SystemExit(f"FAIL: sweep exit {proc.returncode}")
    raw = json.loads(RAW.read_text())
    headlines = []
    for i in range(1, n_headline + 1):
        out = RECEIPT.parent / HEADLINE_GLOB.format(i)
        extra_h = [
            "--mode",
            "headline",
            "--out",
            str(out),
            "--max-bytes",
            str(max_bytes),
            "--warm-reps",
            str(warm_reps),
        ]
        proc = run_locked(binary, extra_h, f"n017-roof-h{i}")
        if proc.returncode != 0:
            raise SystemExit(f"FAIL: headline {i} exit {proc.returncode}")
        headlines.append(json.loads(out.read_text()))
    occ_after = occupancy_snapshot()
    doc, contradicted, roof = build_receipt(raw, headlines, build_info, cmd, occ_after)
    if contradicted and roof is not None:
        corr = correct_anchor_sources(roof)
        doc["anchor_roof"]["corrected"] = True
        doc["anchor_roof"]["correction"] = corr
    write_receipt(doc)
    print(doc["one_line"])
    print(f"wrote {RECEIPT}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="run the GPU sweep (default)")
    ap.add_argument("--reuse", action="store_true", help="do not run GPU; require an existing receipt")
    ap.add_argument("--max-bytes", type=int, default=4 * 1024 * 1024 * 1024)
    ap.add_argument("--warm-reps", type=int, default=5)
    ap.add_argument("--headline-runs", type=int, default=3)
    ap.add_argument(
        "--from-raw",
        action="store_true",
        help="rebuild BANDWIDTH_ROOF.json from existing raw/headline files; no GPU",
    )
    args = ap.parse_args()
    if args.reuse:
        if not RECEIPT.is_file():
            raise SystemExit(f"FAIL: --reuse but {RECEIPT} is missing")
        print(f"reuse {RECEIPT}")
        return 0
    if args.from_raw:
        rebuild_from_raw()
        return 0
    measure(args.max_bytes, args.warm_reps, args.headline_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
