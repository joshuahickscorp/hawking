#!/usr/bin/env python3
"""mst_gap.py — Metal-System-Trace gap-distribution + two-engine DIFF parser.

This is the analysis half of tools/bench/mst_diff.sh, the single-stream
1.6x-gap decider. It consumes the per-kernel GPU-interval XML that
mst_export.sh produces (one `.trace` -> one or more `<schema>.xml`) for BOTH
dismantle and llama-cli on the *same* model/prompt/seed/temp0/256-tok run, and
emits a side-by-side diff plus the inter-dispatch gap distribution.

WHY (the surviving reframe, adverse prior):
  The "24% idle" framing is DEAD — production decode runs ONE command buffer
  per token, so there is ~0.0 ms intra-token inter-dispatch idle (SplitCbGpu
  was an artifact). The trace's real, narrow job is a PER-KERNEL
  GPU-us/call + GiB-s/call DIFF of llama.cpp vs dismantle for the dominant
  GEMV. That is the named, cheap oracle for the only single-stream reframe
  still alive:
    • per-kernel GPU-us/call: llama < dismantle for the dominant GEMV
        => dismantle's kernel is leaving GPU time on the table at equal bytes
           => Type-2 reframe ALIVE (a faster GEMV closes part of the 1.6x).
    • ~equal GPU-us/call (within noise) at ~equal GiB/s/call
        => the GEMV is at the same per-call efficiency on both engines
           => single-stream decode tps is a CLOSED Type-1 kill (the gap is
              NOT in the dominant kernel's GPU time; it is whole-pipeline /
              dispatch-rate that the 0.0 ms-idle production CB already shows
              dismantle is not losing to intra-token idle).

XML PARSING:
  xctrace export XML uses id/ref dedup — a cell value is written once with
  id="N" and later rows reference it with ref="N". The parser here reuses
  mst_analyze.py's resolve-on-the-fly iterparse approach (so a multi-hundred-MB
  256-token trace streams instead of loading whole). See parse_rows().

GAP MATH:
  Within ONE engine's GPU-interval table, sort intervals by start time. The
  inter-dispatch gap is start[i+1] - end[i] (GPU idle between consecutive
  kernels). We bucket the gaps and report {count, sum, p50, p90, max, #>50us}.
  Two start/end columns are REQUIRED for gap math; --inspect surfaces them and
  you pass --start-col/--end-col (defaults heuristic).

  NOTE: in PRODUCTION single-CB decode the in-trace gap should be ~0. A large
  measured inter-dispatch gap on the dismantle side is itself the finding
  (something is serializing the CB); a large gap on the llama side at HIGHER
  tps would mean its per-kernel work is just denser. Report both; do not
  pre-judge.

Usage:
  # inspect one table to find the columns (do this first per engine/schema):
  tools/bench/mst_gap.py --inspect ENGINE.xml

  # single-engine gap distribution:
  tools/bench/mst_gap.py ENGINE.xml --tokens 256 \
      [--name-col C --dur-col C --start-col C --end-col C] [--dur-unit ns]

  # two-engine diff (the decider):
  tools/bench/mst_gap.py \
      --dismantle DISMANTLE.xml --llama LLAMA.xml \
      --tokens 256 --gemv-match q4_k,gemv,mul_mm,mul_mv [--json]

Notes / honesty:
  • Absolute GPU-us are contaminated if any other GPU workload is open
    (the agent app). Run agent-quit (mst_diff.sh hard-aborts otherwise).
  • Kernel NAMES differ across engines (dismantle: gemm_q4_k_v4_predec_pair;
    llama.cpp Metal: kernel_mul_mm_q4_K_f32 / kernel_mul_mv_*). --gemv-match
    takes a comma-list of case-insensitive substrings; the DOMINANT GEMV per
    engine is auto-selected as the matched kernel with the largest total
    gpu_us, and that is what the per-call diff compares.
"""
import argparse
import collections
import json
import sys
import xml.etree.ElementTree as ET

M3_PRO_PEAK_GBPS = 150.0
# Qwen2.5-3B-Q4_K_M reads ~1.93 GiB of weights/token (bandwidth anchor).
QWEN3B_BYTES_PER_TOKEN = int(1.93 * 1024 ** 3)
DUR_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
GAP_THRESHOLD_US = 50.0  # "#gaps>50us" bucket the task asks for


# ---------------------------------------------------------------------------
# XML row parser — id/ref dedup resolved on the fly (mirrors mst_analyze.py).
# Streams via iterparse so a 256-token trace's GPU-interval table (which can be
# hundreds of MB) does not have to be fully materialized.
# ---------------------------------------------------------------------------
def parse_rows(path):
    """Return (col_names, rows). Each row is a list of resolved cell values;
    a cell value is its text, else its `fmt` attr/child."""
    id_map = {}
    col_names, rows = [], []
    in_schema = False
    cur = None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        tag = el.tag
        if ev == "start":
            if tag == "schema":
                in_schema = True
            elif tag == "row":
                cur = []
        else:  # end
            if tag == "schema":
                in_schema = False
            elif in_schema and tag == "col":
                nm = el.findtext("name") or el.findtext("mnemonic") or f"col{len(col_names)}"
                col_names.append(nm.strip())
            elif tag == "row":
                rows.append(cur)
                cur = None
                el.clear()
            elif cur is not None and tag not in ("fmt",):
                ref = el.get("ref")
                if ref is not None:
                    cur.append(id_map.get(ref))
                else:
                    val = (el.text or "").strip() or el.get("fmt")
                    if val is None:
                        f = el.find("fmt")
                        val = f.text.strip() if f is not None and f.text else None
                    if el.get("id") is not None:
                        id_map[el.get("id")] = val
                    cur.append(val)
    return col_names, rows


def looks_numeric(v):
    if v is None:
        return False
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Column heuristics. name = most-distinct non-numeric col; dur = a col whose
# name hints time, else the widest-spread numeric col; start/end = the two
# numeric cols whose names hint start/end (for the gap math).
# ---------------------------------------------------------------------------
def pick_cols(col_names, rows, name_col, dur_col, start_col, end_col):
    ncols = max((len(r) for r in rows), default=0)
    names = col_names if len(col_names) == ncols else [f"col{i}" for i in range(ncols)]

    def resolve(spec, default):
        if spec is None:
            return default
        if spec.isdigit():
            return int(spec)
        return names.index(spec) if spec in names else default

    n_idx = resolve(name_col, None)
    d_idx = resolve(dur_col, None)
    s_idx = resolve(start_col, None)
    e_idx = resolve(end_col, None)

    if n_idx is None:
        best, bn = -1, 0
        for i in range(ncols):
            vals = [r[i] for r in rows[:500] if i < len(r)]
            distinct = len({v for v in vals if v and not looks_numeric(v)})
            if distinct > bn:
                bn, best = distinct, i
        n_idx = best if best >= 0 else 0

    def first_named(hints):
        for i, nm in enumerate(names):
            if any(h in nm.lower() for h in hints):
                return i
        return None

    if d_idx is None:
        d_idx = first_named(("duration", "dur", "gpu-time", "gputime", "elapsed", "length"))
        if d_idx is None:
            for i in range(ncols):
                col = [r[i] for r in rows[:80] if i < len(r) and r[i]]
                if col and all(looks_numeric(x) for x in col):
                    d_idx = i
                    break
    if s_idx is None:
        s_idx = first_named(("start-time", "starttime", "start", "begin"))
    if e_idx is None:
        e_idx = first_named(("end-time", "endtime", "finish", "stop"))
        if e_idx is not None and e_idx == s_idx:
            e_idx = None
    return names, n_idx, d_idx, s_idx, e_idx


def pctl(sorted_vals, q):
    """Nearest-rank percentile on an already-sorted list (us)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


# ---------------------------------------------------------------------------
# Per-engine analysis: per-kernel {n, gpu_us}, total busy, and the
# inter-dispatch gap distribution from sorted start/end intervals.
# ---------------------------------------------------------------------------
def analyze_engine(path, name_col, dur_col, start_col, end_col, dur_unit, tokens, label):
    col_names, rows = parse_rows(path)
    if not rows:
        sys.exit(f"[{label}] no <row> elements parsed from {path} (wrong table? try --inspect)")
    names, n_idx, d_idx, s_idx, e_idx = pick_cols(
        col_names, rows, name_col, dur_col, start_col, end_col
    )
    ns = DUR_TO_NS[dur_unit]  # converts the duration/start/end columns to ns

    by_k = collections.defaultdict(lambda: {"n": 0, "gpu_us": 0.0})
    intervals = []  # (start_ns, end_ns) for gap math, when start/end available
    n_dur = 0
    for r in rows:
        if d_idx is None or d_idx >= len(r) or not looks_numeric(r[d_idx]):
            # still try start/end-only rows for gaps
            pass
        else:
            k = (r[n_idx] if (n_idx is not None and n_idx < len(r)) else None) or "other"
            dur_us = _num(r[d_idx]) * ns / 1e3
            by_k[k]["n"] += 1
            by_k[k]["gpu_us"] += dur_us
            n_dur += 1
        if s_idx is not None and e_idx is not None and s_idx < len(r) and e_idx < len(r):
            s, e = _num(r[s_idx]), _num(r[e_idx])
            if s is not None and e is not None:
                intervals.append((s * ns, e * ns))

    total_gpu_us = sum(v["gpu_us"] for v in by_k.values())

    # Inter-dispatch gap distribution (sorted by GPU start).
    gap = None
    if len(intervals) >= 2:
        intervals.sort(key=lambda t: t[0])
        gaps_us = []
        for i in range(len(intervals) - 1):
            g_ns = intervals[i + 1][0] - intervals[i][1]
            if g_ns > 0:
                gaps_us.append(g_ns / 1e3)
        span_us = (intervals[-1][1] - intervals[0][0]) / 1e3
        gaps_sorted = sorted(gaps_us)
        gap = {
            "n_intervals": len(intervals),
            "n_gaps": len(gaps_us),
            "sum_us": sum(gaps_us),
            "p50_us": pctl(gaps_sorted, 0.50),
            "p90_us": pctl(gaps_sorted, 0.90),
            "max_us": gaps_sorted[-1] if gaps_sorted else 0.0,
            "n_over_50us": sum(1 for g in gaps_us if g > GAP_THRESHOLD_US),
            "span_us": span_us,
            "busy_us": total_gpu_us,
            "busy_frac_of_span": (total_gpu_us / span_us) if span_us > 0 else None,
        }

    per_tok_gpu_us = (total_gpu_us / tokens) if tokens else None
    eff_gbps = None
    if per_tok_gpu_us and per_tok_gpu_us > 0:
        eff_gbps = QWEN3B_BYTES_PER_TOKEN / (per_tok_gpu_us / 1e6) / 1024 ** 3

    return {
        "label": label,
        "path": path,
        "columns": names,
        "name_col": n_idx,
        "dur_col": d_idx,
        "start_col": s_idx,
        "end_col": e_idx,
        "n_intervals": n_dur,
        "total_gpu_us": total_gpu_us,
        "per_token_gpu_us": per_tok_gpu_us,
        "effective_gibps": eff_gbps,
        "gap": gap,
        "by_kernel": {k: dict(v) for k, v in by_k.items()},
    }


def select_gemv(by_kernel, gemv_match):
    """Return (name, stats) of the matched kernel with the largest total
    gpu_us, or (None, None) if no kernel name matches any substring."""
    subs = [s.strip().lower() for s in gemv_match.split(",") if s.strip()]
    cand = [
        (k, v) for k, v in by_kernel.items()
        if any(s in (k or "").lower() for s in subs)
    ]
    if not cand:
        return None, None
    return max(cand, key=lambda kv: kv[1]["gpu_us"])


def gemv_call_stats(name, stats, tokens):
    """Per-call GPU-us and per-call achieved GiB/s for one matched GEMV.
    GiB/s/call is computed against the per-token weight read divided by the
    GEMV's calls-per-token (an approximation: the dominant GEMV moves the bulk
    of the 1.93 GiB; this is a relative cross-engine comparison, not absolute).
    """
    n = stats["n"]
    if n == 0:
        return None
    us_per_call = stats["gpu_us"] / n
    calls_per_tok = (n / tokens) if tokens else None
    gibps_per_call = None
    if calls_per_tok and us_per_call > 0:
        bytes_per_call = QWEN3B_BYTES_PER_TOKEN / calls_per_tok
        gibps_per_call = bytes_per_call / (us_per_call / 1e6) / 1024 ** 3
    return {
        "name": name,
        "n": n,
        "calls_per_token": calls_per_tok,
        "us_per_call": us_per_call,
        "total_gpu_us": stats["gpu_us"],
        "gibps_per_call": gibps_per_call,
    }


def print_engine(a):
    print(f"--- engine: {a['label']}  ({a['path']}) ---")
    print(f"  columns: {a['columns']}")
    print(f"  cols used: name={a['name_col']} dur={a['dur_col']} "
          f"start={a['start_col']} end={a['end_col']}")
    print(f"  intervals: {a['n_intervals']}   total GPU busy: {a['total_gpu_us']/1000:.2f} ms")
    if a["per_token_gpu_us"] is not None:
        print(f"  per-token GPU busy: {a['per_token_gpu_us']/1000:.3f} ms")
    if a["effective_gibps"] is not None:
        print(f"  effective BW: {a['effective_gibps']:.1f} GiB/s "
              f"({a['effective_gibps']/M3_PRO_PEAK_GBPS*100:.0f}% of {M3_PRO_PEAK_GBPS:.0f} peak)")
    g = a["gap"]
    if g:
        print(f"  inter-dispatch gap: n_gaps={g['n_gaps']}  sum={g['sum_us']/1000:.3f} ms  "
              f"p50={g['p50_us']:.2f}us  p90={g['p90_us']:.2f}us  max={g['max_us']:.1f}us  "
              f"#>50us={g['n_over_50us']}")
        if g["busy_frac_of_span"] is not None:
            print(f"  GPU-busy fraction of trace span: {g['busy_frac_of_span']*100:.1f}% "
                  f"(production single-CB => expect ~100%)")
    else:
        print("  inter-dispatch gap: N/A (no start/end columns; pass --start-col/--end-col)")
    print(f"  top kernels by GPU-us:")
    print(f"    {'kernel':46s} {'n':>7s} {'gpu_ms':>9s} {'us/call':>9s}")
    for k, v in sorted(a["by_kernel"].items(), key=lambda kv: -kv[1]["gpu_us"])[:8]:
        print(f"    {k:46s} {v['n']:7d} {v['gpu_us']/1000:9.2f} {v['gpu_us']/max(v['n'],1):9.1f}")


def decision(dm_g, ll_g):
    """Render the GEMV per-call decision tree. dm_g/ll_g are gemv_call_stats."""
    if dm_g is None or ll_g is None:
        return ("INCONCLUSIVE", [
            "Could not isolate the dominant GEMV on both engines.",
            "Re-run --inspect to find the kernel-name column and widen --gemv-match.",
            f"  (dismantle GEMV found: {dm_g is not None}; llama GEMV found: {ll_g is not None})",
        ])
    dm_us, ll_us = dm_g["us_per_call"], ll_g["us_per_call"]
    # 8% band = quant/measurement noise floor for a per-call GPU-us compare.
    lo, hi = 0.92 * dm_us, 1.08 * dm_us
    lines = [
        f"dominant GEMV per-call GPU time:",
        f"  dismantle  {dm_g['name']:38s} {dm_us:8.2f} us/call  "
        f"({dm_g['gibps_per_call']:.0f} GiB/s/call)" if dm_g["gibps_per_call"] else
        f"  dismantle  {dm_g['name']:38s} {dm_us:8.2f} us/call",
        f"  llama.cpp  {ll_g['name']:38s} {ll_us:8.2f} us/call  "
        f"({ll_g['gibps_per_call']:.0f} GiB/s/call)" if ll_g["gibps_per_call"] else
        f"  llama.cpp  {ll_g['name']:38s} {ll_us:8.2f} us/call",
        f"  ratio dismantle/llama = {dm_us/ll_us:.3f}x" if ll_us > 0 else "",
        "",
    ]
    if ll_us < lo:
        verdict = "TYPE-2 REFRAME ALIVE"
        lines += [
            f"llama's dominant GEMV is FASTER per call ({ll_us:.2f} < {dm_us:.2f} us, "
            f">8% gap).",
            "=> dismantle's GEMV is leaving GPU time on the table at equal bytes.",
            "   A faster GEMV (vectorized uint4 nibble load / better occupancy) is the",
            "   live single-stream lever — it closes part of the 1.6x. Build + re-DIFF.",
        ]
    elif ll_us > hi:
        verdict = "DISMANTLE GEMV ALREADY FASTER"
        lines += [
            f"dismantle's GEMV is FASTER per call ({dm_us:.2f} < {ll_us:.2f} us).",
            "=> the 1.6x tps gap is NOT in the dominant kernel's GPU time on dismantle's",
            "   side. The gap is whole-pipeline (dispatch rate / kernel count / non-GEMV",
            "   work). Single-stream GEMV micro-opt is NOT the lever; look at dispatch",
            "   count/token and the gap distribution above.",
        ]
    else:
        verdict = "SINGLE-STREAM TYPE-1 (GEMV efficiency parity)"
        lines += [
            f"per-call GPU time is ~EQUAL ({dm_us:.2f} vs {ll_us:.2f} us, within 8%).",
            "=> the dominant GEMV runs at the same per-call efficiency on both engines",
            "   at equal bytes. There is no GPU-us to reclaim in the dominant kernel.",
            "   Combined with production's 0.0 ms intra-token inter-dispatch idle, the",
            "   single-stream decode-tps reframe is a CLOSED Type-1 kill: the 1.6x is a",
            "   dispatch-RATE / per-token-overhead property, not the GEMV. Record in",
            "   reports/dead_levers.md; do not re-spawn a single-stream GEMV micro-opt.",
        ]
    return verdict, [l for l in lines if l != ""] or ["(no detail)"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("table_xml", nargs="?", help="single-engine table (gap distribution mode)")
    ap.add_argument("--dismantle", help="dismantle GPU-interval XML (two-engine diff)")
    ap.add_argument("--llama", help="llama.cpp GPU-interval XML (two-engine diff)")
    ap.add_argument("--inspect", action="store_true", help="dump schema + first rows, then exit")
    ap.add_argument("--name-col")
    ap.add_argument("--dur-col")
    ap.add_argument("--start-col")
    ap.add_argument("--end-col")
    ap.add_argument("--dur-unit", choices=sorted(DUR_TO_NS), default="ns",
                    help="unit of the duration AND start/end columns (default ns)")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--gemv-match",
                    default="q4_k,q4_K,gemv,mul_mv,mul_mm,gemm_q4",
                    help="comma-list of case-insensitive substrings naming the GEMV")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # --inspect on whichever single table was given.
    insp_target = args.table_xml or args.dismantle or args.llama
    if args.inspect:
        if not insp_target:
            sys.exit("--inspect needs a table (positional, or --dismantle/--llama)")
        col_names, rows = parse_rows(insp_target)
        if not rows:
            sys.exit(f"no <row> elements parsed from {insp_target}")
        names, n_idx, d_idx, s_idx, e_idx = pick_cols(
            col_names, rows, args.name_col, args.dur_col, args.start_col, args.end_col
        )
        print(f"columns ({len(names)}): {names}")
        print(f"heuristic: name={n_idx} dur={d_idx} start={s_idx} end={e_idx}")
        print("first rows:")
        for r in rows[:6]:
            print("  ", r)
        print("\nPass --name-col/--dur-col/--start-col/--end-col if the heuristic is wrong.")
        print("start+end columns are REQUIRED for the inter-dispatch gap distribution.")
        return

    # Two-engine diff (the decider).
    if args.dismantle and args.llama:
        dm = analyze_engine(args.dismantle, args.name_col, args.dur_col,
                            args.start_col, args.end_col, args.dur_unit,
                            args.tokens, "dismantle")
        ll = analyze_engine(args.llama, args.name_col, args.dur_col,
                            args.start_col, args.end_col, args.dur_unit,
                            args.tokens, "llama.cpp")
        dm_name, dm_stats = select_gemv(dm["by_kernel"], args.gemv_match)
        ll_name, ll_stats = select_gemv(ll["by_kernel"], args.gemv_match)
        dm_g = gemv_call_stats(dm_name, dm_stats, args.tokens) if dm_name else None
        ll_g = gemv_call_stats(ll_name, ll_stats, args.tokens) if ll_name else None
        verdict, detail = decision(dm_g, ll_g)

        if args.json:
            out = {
                "tokens": args.tokens,
                "dismantle": dm, "llama": ll,
                "dismantle_gemv": dm_g, "llama_gemv": ll_g,
                "verdict": verdict, "detail": detail,
            }
            print(json.dumps(out, indent=2, default=str))
            return

        print("=" * 80)
        print("  MST TWO-ENGINE DIFF — single-stream 1.6x-gap decider")
        print("=" * 80)
        print_engine(dm)
        print()
        print_engine(ll)
        print()
        print("=" * 80)
        print(f"  VERDICT: {verdict}")
        print("=" * 80)
        for l in detail:
            print(f"  {l}")
        print()
        # Whole-pipeline context: dispatch-count/token both engines.
        dm_disp = dm["n_intervals"] / args.tokens if args.tokens else 0
        ll_disp = ll["n_intervals"] / args.tokens if args.tokens else 0
        print(f"  context: dispatch-intervals/token  dismantle={dm_disp:.1f}  "
              f"llama={ll_disp:.1f}")
        if dm["per_token_gpu_us"] and ll["per_token_gpu_us"]:
            print(f"           per-token GPU busy        dismantle="
                  f"{dm['per_token_gpu_us']/1000:.3f} ms  "
                  f"llama={ll['per_token_gpu_us']/1000:.3f} ms")
        return

    # Single-engine gap distribution.
    if not args.table_xml:
        sys.exit("give a table (positional) for gap mode, or --dismantle+--llama for the diff")
    a = analyze_engine(args.table_xml, args.name_col, args.dur_col,
                       args.start_col, args.end_col, args.dur_unit,
                       args.tokens, "engine")
    if args.json:
        print(json.dumps(a, indent=2, default=str))
        return
    print_engine(a)


if __name__ == "__main__":
    main()
