#!/usr/bin/env python3
"""0.1 — analyze the Metal-System-Trace XML exported by mst_export.sh into a
per-kernel GPU-occupancy / achieved-bandwidth breakdown, and cross-check the
busy fraction against the homemade §1 tool (analyze_tcb_trace.py --json).

The MST is the un-distorted ground truth: production decode runs ONE command
buffer per token, so unlike the split-CB homemade trace its GPU-busy time is
real. This is the Stage-2 unblock — it tells the next kernel lever where the
actual stall is instead of guessing (the `_4r` dead lever).

xctrace XML uses id/ref dedup: a cell value is written once with id="N" and
later rows reference it with ref="N". This parser resolves that.

FIRST-CAPTURE VALIDATION (no sample trace existed when this was written):
  tools/bench/mst_analyze.py export/<schema>.xml --inspect
prints the schema columns + first rows so you can see which column is the
kernel name and which is the GPU-interval duration, then pass --name-col /
--dur-col (by 0-based index or column name). Defaults use heuristics.

Usage:
  mst_analyze.py TABLE.xml --inspect
  mst_analyze.py TABLE.xml [--name-col C] [--dur-col C] [--dur-unit ns|us|ms]
                 [--tokens N --decode-wall-ms W] [--model qwen3b]
                 [--cross-check tcb_analysis.json] [--json]
"""
import argparse
import collections
import json
import sys
import xml.etree.ElementTree as ET

M3_PRO_PEAK_GBPS = 150.0
MODEL_BYTES = {"qwen3b": int(1.93 * 1024 ** 3), "v2lite": int(1.82 * 1024 ** 3)}
DUR_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}


def parse_rows(path):
    """Return (col_names, rows) where each row is a list of resolved cell
    values, resolving xctrace's id/ref dedup. Cell value = text or fmt attr."""
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
    except ValueError:
        return False


def pick_cols(col_names, rows, name_col, dur_col):
    ncols = max((len(r) for r in rows), default=0)
    names = col_names if len(col_names) == ncols else [f"col{i}" for i in range(ncols)]

    def resolve(spec, default):
        if spec is None:
            return default
        if spec.isdigit():
            return int(spec)
        return names.index(spec) if spec in names else default

    # Heuristic name col = most distinct non-numeric values; dur col = numeric
    # col whose name hints time, else the numeric col with the largest spread.
    n_idx = resolve(name_col, None)
    d_idx = resolve(dur_col, None)
    if n_idx is None:
        best, bn = -1, 0
        for i in range(ncols):
            vals = [r[i] for r in rows[:500] if i < len(r)]
            distinct = len({v for v in vals if v and not looks_numeric(v)})
            if distinct > bn:
                bn, best = distinct, i
        n_idx = best
    if d_idx is None:
        for i, nm in enumerate(names):
            if any(h in nm.lower() for h in ("duration", "dur", "gpu-time", "elapsed")):
                d_idx = i
                break
        if d_idx is None:
            for i in range(ncols):
                if all(looks_numeric(r[i]) for r in rows[:50] if i < len(r) and r[i]):
                    d_idx = i
                    break
    return names, n_idx, d_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table_xml")
    ap.add_argument("--inspect", action="store_true", help="dump schema + first rows, then exit")
    ap.add_argument("--name-col", help="kernel-name column (index or name)")
    ap.add_argument("--dur-col", help="GPU-interval duration column (index or name)")
    ap.add_argument("--dur-unit", choices=sorted(DUR_TO_US), default="ns")
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--decode-wall-ms", type=float)
    ap.add_argument("--model", choices=sorted(MODEL_BYTES), default="qwen3b")
    ap.add_argument("--cross-check", help="analyze_tcb_trace.py --json output to compare busy-fraction")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    col_names, rows = parse_rows(args.table_xml)
    if not rows:
        sys.exit(f"no <row> elements parsed from {args.table_xml} (wrong table? try --inspect on toc)")
    names, n_idx, d_idx = pick_cols(col_names, rows, args.name_col, args.dur_col)

    if args.inspect:
        print(f"columns ({len(names)}): {names}")
        print(f"heuristic name-col = {n_idx} ({names[n_idx] if n_idx is not None else '?'}), "
              f"dur-col = {d_idx} ({names[d_idx] if d_idx is not None else '?'})")
        print("first rows:")
        for r in rows[:5]:
            print("  ", r)
        print("\nIf the heuristic columns are wrong, re-run with --name-col / --dur-col.")
        return
    if n_idx is None or d_idx is None:
        sys.exit("could not identify name/duration columns; use --inspect then --name-col/--dur-col")

    scale = DUR_TO_US[args.dur_unit]
    by_k = collections.defaultdict(lambda: {"n": 0, "gpu_us": 0.0})
    for r in rows:
        if d_idx >= len(r) or n_idx >= len(r) or not looks_numeric(r[d_idx]):
            continue
        k = r[n_idx] or "other"
        by_k[k]["n"] += 1
        by_k[k]["gpu_us"] += float(r[d_idx]) * scale
    total_us = sum(v["gpu_us"] for v in by_k.values())

    busy_frac = None
    if args.tokens and args.decode_wall_ms:
        busy_frac = (total_us / args.tokens) / (args.decode_wall_ms * 1000)
    eff_gbps = None
    if args.tokens:
        per_tok_s = (total_us / args.tokens) / 1e6
        if per_tok_s > 0:
            eff_gbps = MODEL_BYTES[args.model] / per_tok_s / 1024 ** 3

    cross = None
    if args.cross_check:
        tcb = json.load(open(args.cross_check))
        tcb_ptus = tcb.get("per_token_gpu_us")
        tcb_busy = None
        if tcb_ptus and args.decode_wall_ms:
            tcb_busy = tcb_ptus / (args.decode_wall_ms * 1000)
        if busy_frac is not None and tcb_busy:
            delta = abs(busy_frac - tcb_busy) / tcb_busy
            cross = {"mst_busy": busy_frac, "tcb_busy": tcb_busy,
                     "rel_delta": delta, "within_5pct": delta <= 0.05}

    if args.json:
        out = {"table": args.table_xml, "intervals": sum(v["n"] for v in by_k.values()),
               "total_gpu_us": total_us, "tokens": args.tokens,
               "per_token_gpu_us": (total_us / args.tokens) if args.tokens else None,
               "busy_fraction": busy_frac, "effective_bandwidth_gibps": eff_gbps,
               "by_kernel": {k: {"n": v["n"], "gpu_us": round(v["gpu_us"], 1)} for k, v in by_k.items()},
               "cross_check": cross}
        print(json.dumps(out, indent=2))
        if cross and not cross["within_5pct"]:
            sys.exit(2)
        return

    print(f"--- MST table: {args.table_xml} ---")
    print(f"intervals: {sum(v['n'] for v in by_k.values())}   total GPU busy: {total_us/1000:.2f} ms")
    if args.tokens:
        print(f"per-token GPU busy: {total_us/args.tokens/1000:.3f} ms over {args.tokens} tokens")
    print(f"\n{'kernel':45s} {'n':>6s} {'gpu_ms':>10s} {'us/call':>9s} {'% GPU':>7s}")
    print("-" * 82)
    for k, v in sorted(by_k.items(), key=lambda kv: -kv[1]["gpu_us"]):
        pct = v["gpu_us"] / total_us * 100 if total_us else 0
        print(f"{k:45s} {v['n']:6d} {v['gpu_us']/1000:10.2f} {v['gpu_us']/max(v['n'],1):9.1f} {pct:6.2f}%")
    if eff_gbps is not None:
        print(f"\nachieved BW: {eff_gbps:.1f} GiB/s ({eff_gbps/M3_PRO_PEAK_GBPS*100:.0f}% of {M3_PRO_PEAK_GBPS:.0f} peak)")
    if busy_frac is not None:
        print(f"GPU-busy fraction: {busy_frac*100:.1f}% of decode wall (Bible: ~85% ⇒ kernel-bound)")
    if cross:
        verdict = "PASS (within 5%)" if cross["within_5pct"] else "MISMATCH (>5%)"
        print(f"\ncross-check vs §1 TCB tool: MST {cross['mst_busy']*100:.1f}% vs TCB "
              f"{cross['tcb_busy']*100:.1f}% — {verdict}")
        if not cross["within_5pct"]:
            sys.exit(2)


if __name__ == "__main__":
    main()
