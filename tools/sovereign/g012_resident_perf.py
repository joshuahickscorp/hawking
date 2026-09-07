#!/usr/bin/env python3
"""G012 producer: a FRESH complete-token decode measurement, conditions recorded.

Receipt: receipts/sovereign/G012_resident_perf.json

Refuses to emit a partial receipt. Every exit that cannot measure honestly writes
nothing and returns non-zero with the reason.

Two things this deliberately does NOT do:

  * It does not measure several samples in ONE resident process and take their
    median. That resident degrades ~4.5x by its third request (measured: the same
    16K prompt takes 446 s on a fresh process and 1,998 s as the third call on a
    warm one), so a median over a degrading series is not a rate. Every arm here
    runs in a FRESH subprocess, and the drift between arms is reported rather
    than averaged away.
  * It does not divide one call's whole wall by the tokens it produced. That wall
    contains a one-time model load and the prompt prefill, so a short run reports
    a "decode rate" dominated by setup -- measured: 10.4 tok/s at 96 tokens, which
    is setup, not decode. Worse, a 2-stream arm amortises the SAME load over twice
    the tokens, so it would look faster for a reason that has nothing to do with
    saturation. Decode is measured as a MARGINAL rate from two token counts on
    otherwise identical calls, which cancels load and prefill exactly.
  * It does not reuse a historical 34/36/45/62 tok/s figure. `reused_historical_figure`
    is hardcoded False because the number is taken here, now, or not at all.
  * It does not report `vm.swapusage used=` as live swap. That counter is a BOOT
    HIGH-WATER MARK: it never falls, so comparing it to a ceiling latches any gate
    that reads it. The receipt records it under its real name and ALSO records the
    Swapouts delta across the measurement, which is the only figure on macOS that
    can distinguish "swapping now" from "swapped once, hours ago".
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/sovereign/G012_resident_perf.json"
PRODUCED_BY = "tools/sovereign/g012_resident_perf.py"
SCHEMA = "hawking.sovereign.g012_resident_perf.v1"
ENVELOPE = REPO / "ascension_envelope.hawking.json"

sys.path.insert(0, str(REPO))


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERR {exc}"


def _swapouts() -> int:
    """Cumulative swapouts since boot. Only its DELTA means anything."""
    out = _sh("vm_stat")
    m = re.search(r"Swapouts:\s+(\d+)", out)
    return int(m.group(1)) if m else -1


def _free_bytes() -> int:
    out = _sh("vm_stat")
    page = re.search(r"page size of (\d+) bytes", out)
    free = re.search(r"Pages free:\s+(\d+)", out)
    if not page or not free:
        return -1
    return int(page.group(1)) * int(free.group(1))


def _contamination() -> list:
    out = _sh("ps aux | sort -k3 -rn | head -8")
    rows = []
    for line in out.splitlines()[:8]:
        parts = line.split(None, 10)
        if len(parts) >= 11:
            try:
                cpu = float(parts[2])
            except ValueError:
                continue
            rows.append({"cpu_pct": cpu, "command": parts[10][:110]})
    return rows


def _decode_rate(conn, prompt: str, max_tokens: int) -> dict:
    """One complete-token measurement: wall for a known number of decoded tokens."""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    body = conn.complete_payload(payload, timeout=1800)
    wall = time.perf_counter() - t0
    usage = body.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    calls = body.get("model_calls") or []
    prefill_ns = 0
    if calls:
        prefill_ns = int(((calls[0].get("prefill_profile") or {}).get("totals") or {})
                         .get("wall_ns") or 0)
    return {
        "wall_s": wall,
        "completion_tokens": completion,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "prefill_wall_ns": prefill_ns,
        "decode_wall_s": max(wall - prefill_ns / 1e9, 0.0) if prefill_ns else None,
    }


def _run_arm(streams: int, tokens: int) -> dict:
    """One arm, in a FRESH interpreter. Process exit is the only reset available."""
    out = subprocess.run(
        [sys.executable, str(REPO / PRODUCED_BY), "--arm", "--streams", str(streams),
         "--tokens", str(tokens)],
        capture_output=True, text=True, timeout=7200, cwd=str(REPO),
    )
    for line in out.stdout.splitlines():
        if line.startswith("ARM "):
            return json.loads(line[4:])
    raise RuntimeError(f"arm produced no result: {out.stderr[-400:]}")


def _arm_main(args) -> int:
    from hcli.hawking_native import HawkingNativeConnector, config_for_model_path
    cfg = config_for_model_path(str(ENVELOPE))
    conn = HawkingNativeConnector(cfg)
    prompt = ("Count upward from one, one number per line, and do not stop early. "
              "Write nothing else.")
    t0 = time.perf_counter()
    if args.streams <= 1:
        res = [_decode_rate(conn, prompt, args.tokens)]
    else:
        with ThreadPoolExecutor(max_workers=args.streams) as pool:
            futures = [pool.submit(_decode_rate, conn, prompt, args.tokens)
                       for _ in range(args.streams)]
            res = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    toks = sum(r["completion_tokens"] for r in res)
    print("ARM " + json.dumps({
        "streams": args.streams, "wall_s": wall, "completion_tokens": toks,
        "tok_s": (toks / wall) if wall > 0 else 0, "calls": res,
    }), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tokens", type=int, default=128,
                    help="decoded tokens per single-stream sample")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--streams", type=int, default=2,
                    help="concurrent streams for the saturation retest")
    args = ap.parse_args()
    if args.arm:
        return _arm_main(args)

    cfg = None
    try:
        from hcli.hawking_native import config_for_model_path
        cfg = config_for_model_path(str(ENVELOPE))
    except Exception:
        pass

    swap_before = _swapouts()
    conds_before = {
        "free_bytes": _free_bytes(),
        "background_contamination": _contamination(),
        "swap_used_bytes_boot_high_water": _sh("sysctl -n vm.swapusage"),
        "swapouts_cumulative_before": swap_before,
    }

    # Marginal decode: two token counts, otherwise identical calls, each arm in a
    # FRESH process. wall(hi) - wall(lo) contains ONLY the extra decoded tokens --
    # the model load and the prompt prefill are identical in both and cancel.
    LO, HI = 64, 576

    def marginal(streams: int) -> dict:
        lo = _run_arm(streams, LO)
        hi = _run_arm(streams, HI)
        d_tok = hi["completion_tokens"] - lo["completion_tokens"]
        d_wall = hi["wall_s"] - lo["wall_s"]
        return {
            "lo": lo, "hi": hi, "delta_tokens": d_tok, "delta_wall_s": d_wall,
            "tok_s": (d_tok / d_wall) if d_wall > 0 else 0.0,
            "ns_per_token": (d_wall * 1e9 / d_tok) if d_tok > 0 else 0.0,
        }

    single = marginal(1)
    if single["delta_tokens"] <= 0 or single["delta_wall_s"] <= 0:
        print(f"REFUSED: the single-stream arms did not separate "
              f"(delta_tokens={single['delta_tokens']}, "
              f"delta_wall_s={single['delta_wall_s']}); the model stopped early",
              file=sys.stderr)
        return 2
    single_ns = single["ns_per_token"]

    repeat = marginal(1)
    repeat_ns = repeat["ns_per_token"] or None

    agg = marginal(args.streams)

    swap_after = _swapouts()
    doc = {
        "schema": SCHEMA,
        "produced_by": PRODUCED_BY,
        "command": f"python3 {PRODUCED_BY} --tokens {args.tokens} "
                   f"--samples {args.samples} --streams {args.streams}",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "reused_historical_figure": False,
        "resident_identity": getattr(cfg, "resident_identity", None),
        "envelope": str(ENVELOPE.relative_to(REPO)),
        "complete_token_ns": single_ns,
        "complete_token_ns_repeat": repeat_ns,
        "decode_tok_s": 1e9 / single_ns,
        "decode_tok_s_repeat": (1e9 / repeat_ns) if repeat_ns else None,
        "fresh_process_per_arm": True,
        "run_to_run_spread_pct": (
            abs(repeat_ns - single_ns) / min(repeat_ns, single_ns) * 100.0
            if repeat_ns else None
        ),
        "method": "marginal: (wall(576) - wall(64)) / 512 decoded tokens, each arm a "
                  "fresh process; cancels model load and prompt prefill exactly",
        "arms": {"single": single, "single_repeat": repeat, "aggregate": agg},
        "aggregate_vs_single": {
            "single_tok_s": 1e9 / single_ns,
            # _run_arm SUMS completion tokens across streams, so agg["tok_s"] is
            # ALREADY the aggregate rate. Multiplying it by `streams` double-counts
            # and turns a flat result into an apparent 2x. Do not.
            "aggregate_tok_s": agg["tok_s"],
            "aggregate_tok_s_per_stream": agg["tok_s"] / max(1, args.streams),
            "aggregate_scaling_vs_single": agg["tok_s"] / (1e9 / single_ns),
            "streams": args.streams,
            "aggregate_delta_wall_s": agg["delta_wall_s"],
            "aggregate_delta_tokens": agg["delta_tokens"],
            "both_arms_fresh": True,
        },
        "conditions": {
            "free_bytes": conds_before["free_bytes"],
            "background_contamination": conds_before["background_contamination"],
            # Named for what it is. See the module docstring.
            "swap_used_bytes": conds_before["swap_used_bytes_boot_high_water"],
            "swap_used_bytes_is_boot_high_water": True,
            "swapouts_cumulative_before": swap_before,
            "swapouts_cumulative_after": swap_after,
            "swapouts_delta_during_measurement": (
                swap_after - swap_before if swap_before >= 0 and swap_after >= 0 else None
            ),
            "free_bytes_after": _free_bytes(),
        },
        "evidence": {
            "note": "complete-token wall divided by tokens actually produced; "
                    "no kernel time, no synthetic loop",
            "fresh_arms_run": 3,
            "arm_protocol": "single, single_repeat and aggregate each in a FRESH "
                            "interpreter; the resident degrades ~4.5x across requests "
                            "in one process, so arms must not share one",
        },
    }

    for field in ("complete_token_ns", "decode_tok_s"):
        if not isinstance(doc[field], (int, float)) or doc[field] == 0:
            print(f"REFUSED: {field} is not a measurement", file=sys.stderr)
            return 3
    if not doc["aggregate_vs_single"]["aggregate_tok_s"]:
        print("REFUSED: the saturation retest produced no aggregate rate", file=sys.stderr)
        return 4

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT}")
    print(f"  decode_tok_s   {doc['decode_tok_s']:.3f}  "
          f"(fresh repeat {doc['decode_tok_s_repeat']}, "
          f"run-to-run spread {doc['run_to_run_spread_pct']}%)")
    a = doc["aggregate_vs_single"]
    print(f"  single {a['single_tok_s']:.3f} tok/s   aggregate {a['aggregate_tok_s']:.3f} tok/s "
          f"across {a['streams']} streams")
    print(f"  swapouts delta during measurement: "
          f"{doc['conditions']['swapouts_delta_during_measurement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
