"""`hcli report` -- what is this resident actually like to use?

RUNNABLE, NOT TRANSCRIBED. Every performance claim about HCLI so far has been a
number somebody pasted into a document, which stops being true the moment the
resident changes -- and the resident WILL change. This produces those numbers on
demand, against whatever is loaded right now, and writes a receipt that names
the body it measured. A table in a markdown file is a memory; this is a
measurement.

RESIDENT-AGNOSTIC BY CONSTRUCTION. It asks `/health` what it is talking to and
reports that. It never assumes sealed-3.14, never assumes greedy, and never
assumes a native backend -- point it at an MLX specimen or any OpenAI-compatible
endpoint and the same three questions get answered.

THE THREE QUESTIONS A USER ACTUALLY HAS:

  cold    -- how long until the first answer, for a prompt of size N?
  warm    -- does a follow-up turn re-pay the conversation history?
  decode  -- once it starts, how fast does text arrive?

`warm` carries its own CONTROL. A second identical request being fast proves
nothing on its own: a warm process, a warm allocator and a filled page cache all
look the same. So the run also sends a DIFFERENT prompt of the same size, and
the speedup is only reported as prefix reuse when the control stays slow.
Without that comparison the headline number is unearned.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .latency import format_duration_ns, now_ns, seconds_from_ns

DEFAULT_BASE = "http://127.0.0.1:8011"
FILLER = ("The Hawking runtime schedules Metal command buffers across a hybrid "
          "state-space and attention body. ")


def _post(base: str, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ask(base: str, messages: List[Dict[str, str]], *, max_tokens: int,
        timeout: float) -> Dict[str, Any]:
    started_ns = now_ns()
    data = _post(base, {"messages": messages, "max_tokens": max_tokens}, timeout)
    wall_ns = max(0, now_ns() - started_ns)
    wall_s = seconds_from_ns(wall_ns) or 0.0
    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    return {
        "wall_ns": wall_ns,
        # Compatibility view for callers of report.v1. Rates below are
        # calculated from wall_ns, never from this rounded presentation value.
        "wall_s": wall_s,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "text": ((choice.get("message") or {}).get("content") or ""),
    }


def health(base: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=timeout) as r:
            body = json.loads(r.read())
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _prompt_of(target_tokens: int, salt: str = "") -> str:
    # ~16 tokens per repetition of FILLER, close enough to land near the target.
    return (salt + FILLER * max(1, target_tokens // 16)
            + "\nReply with exactly one word: acknowledged.")


def cold_prefill(base: str, sizes: List[int], *, timeout: float) -> List[Dict[str, Any]]:
    """Time to first answer, each prompt distinct so nothing can be cached."""
    rows = []
    for index, size in enumerate(sizes):
        prompt = _prompt_of(size, salt=f"Run {index} of a cold sweep. ")
        got = ask(base, [{"role": "user", "content": prompt}],
                  max_tokens=8, timeout=timeout)
        tokens = got["prompt_tokens"]
        wall_ns = got["wall_ns"]
        wall_s = got["wall_s"]
        rows.append({
            "requested": size,
            "prompt_tokens": tokens,
            "wall_ns": wall_ns,
            "wall_s": round(wall_s, 2),
            "prompt_tokens_per_s": round(tokens * 1_000_000_000 / wall_ns, 1) if wall_ns else None,
            "ms_per_prompt_token": round(wall_ns / 1_000_000 / tokens, 1) if tokens else None,
        })
    return rows


def warm_turns(base: str, *, size: int, turns: int, timeout: float) -> Dict[str, Any]:
    """Does turn 2 re-pay turn 1's history? With the control that earns the claim."""
    base_prompt = _prompt_of(size, salt="A conversation. ")
    messages = [{"role": "user", "content": base_prompt}]
    rows = []
    first = ask(base, messages, max_tokens=8, timeout=timeout)
    rows.append({"turn": 1, "kind": "cold",
                 "prompt_tokens": first["prompt_tokens"],
                 "wall_ns": first["wall_ns"],
                 "wall_s": round(first["wall_s"], 2)})
    reply = first["text"]
    for turn in range(2, turns + 1):
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"Now say the number {turn}."}]
        got = ask(base, messages, max_tokens=8, timeout=timeout)
        reply = got["text"]
        rows.append({"turn": turn, "kind": "extends prefix",
                     "prompt_tokens": got["prompt_tokens"],
                     "wall_ns": got["wall_ns"],
                     "wall_s": round(got["wall_s"], 2)})

    # THE CONTROL. A different prompt of the same size must stay slow, or the
    # speedup above is a warm process rather than a reused prefix.
    control_prompt = _prompt_of(size, salt="An unrelated conversation. ").replace(
        "Metal", "Vulkan").replace("state-space", "systolic")
    control = ask(base, [{"role": "user", "content": control_prompt}],
                  max_tokens=8, timeout=timeout)
    rows.append({"turn": None, "kind": "control: different prompt, same size",
                 "prompt_tokens": control["prompt_tokens"],
                 "wall_ns": control["wall_ns"],
                 "wall_s": round(control["wall_s"], 2)})

    cold_ns = rows[0]["wall_ns"]
    warm = [r["wall_ns"] for r in rows[1:-1]]
    control_ns = rows[-1]["wall_ns"]
    warm_median_ns = int(statistics.median(warm)) if warm else None
    speedup = (cold_ns / warm_median_ns) if warm_median_ns else None
    # The control has to be within 2x of the cold turn for it to be a control at
    # all; otherwise something else changed and the comparison is void.
    control_held = bool(control_ns >= cold_ns * 0.5)
    return {
        "turns": rows,
        "timing_unit": "ns",
        "cold_ns": cold_ns,
        "warm_median_ns": warm_median_ns,
        "control_ns": control_ns,
        # Compatibility aliases for report.v1 consumers.
        "cold_s": seconds_from_ns(cold_ns),
        "warm_median_s": seconds_from_ns(warm_median_ns),
        "control_s": seconds_from_ns(control_ns),
        "speedup": round(speedup, 1) if speedup else None,
        "prefix_reuse": bool(speedup and speedup >= 2.0 and control_held),
        "verdict": (
            "prefix reuse: a follow-up turn does NOT re-pay the history"
            if (speedup and speedup >= 2.0 and control_held) else
            "no prefix reuse: every turn re-pays its history"
            if control_held else
            "UNDECIDED: the control was fast too, so the speedup is not "
            "attributable to prefix reuse"),
    }


def decode_rate(base: str, *, max_tokens: int, timeout: float) -> Dict[str, Any]:
    """Tokens per second once generation has started, on a short prompt."""
    got = ask(base, [{"role": "user", "content":
                      "Write a detailed paragraph about memory bandwidth."}],
              max_tokens=max_tokens, timeout=timeout)
    completion = got["completion_tokens"]
    wall_ns = got["wall_ns"]
    wall_s = got["wall_s"]
    return {
        "completion_tokens": completion,
        "wall_ns": wall_ns,
        "wall_s": round(wall_s, 2),
        "tokens_per_s": round(completion * 1_000_000_000 / wall_ns, 1) if wall_ns else None,
        "ms_per_token": round(wall_ns / 1_000_000 / completion, 1) if completion else None,
    }


def _wall_ns(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("wall_ns")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    legacy = row.get("wall_s")
    if isinstance(legacy, (int, float)) and not isinstance(legacy, bool):
        return max(0, int(round(float(legacy) * 1_000_000_000)))
    return None


def render(report: Dict[str, Any]) -> str:
    out = []
    resident = report.get("resident") or {}
    out.append(f"HCLI REPORT   resident {resident.get('resident', 'unknown')}"
               f"   {report.get('base_url')}")
    if resident.get("sampling"):
        out.append(f"              sampling {resident['sampling']}")
    out.append("")
    cold = report.get("cold_prefill") or []
    if cold:
        out.append("COLD  time to first answer, nothing cached")
        out.append(f"  {'prompt tok':>10}  {'wall':>8}  {'tok/s':>7}  {'ms/tok':>7}")
        for row in cold:
            wall_ns = _wall_ns(row)
            wall = format_duration_ns(wall_ns) if wall_ns is not None else "unknown"
            out.append(f"  {row['prompt_tokens']:>10}  {wall:>8}  "
                       f"{row['prompt_tokens_per_s']:>7}  {row['ms_per_prompt_token']:>7}")
        out.append("")
    warm = report.get("warm_turns")
    if warm:
        out.append("WARM  does a follow-up turn re-pay the history?")
        for row in warm["turns"]:
            label = row["kind"] if row["turn"] is None else f"turn {row['turn']} ({row['kind']})"
            wall_ns = _wall_ns(row)
            wall = format_duration_ns(wall_ns) if wall_ns is not None else "unknown"
            out.append(f"  {label:<44} {row['prompt_tokens']:>6} tok  {wall:>8}")
        if warm.get("speedup"):
            control_ns = warm.get("control_ns")
            control = format_duration_ns(control_ns) if control_ns is not None else "unknown"
            out.append(f"  speedup {warm['speedup']}x   control {control}")
        out.append(f"  -> {warm['verdict']}")
        out.append("")
    decode = report.get("decode")
    if decode:
        wall_ns = _wall_ns(decode)
        wall = format_duration_ns(wall_ns) if wall_ns is not None else "unknown"
        out.append(f"DECODE  {decode['completion_tokens']} tokens in "
                   f"{wall} = {decode['tokens_per_s']} tok/s "
                   f"({decode['ms_per_token']} ms/token)")
        out.append("")
    if report.get("receipt"):
        out.append(f"receipt {report['receipt']}")
    return "\n".join(out)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hcli report",
        description="Measure the resident that is running right now.")
    # POSITIONAL, so `hcli report Qwen3-14B` measures that body: it switches to
    # it first, then runs. Omit it to measure whatever is answering.
    ap.add_argument("model", nargs="?", default=None,
                    help="body to measure; omit for the one already loaded")
    ap.add_argument("--base", default=DEFAULT_BASE, help=argparse.SUPPRESS)
    ap.add_argument("--sizes", default="100,500,2000",
                    help="cold-prefill prompt sizes in tokens, comma separated")
    ap.add_argument("--warm-size", type=int, default=700)
    ap.add_argument("--warm-turns", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--json", action="store_true", help="print the receipt, not the table")
    ap.add_argument("--out", default=None, help="write the receipt here")
    ap.add_argument("--skip", default="", help="comma list of: cold,warm,decode")
    a = ap.parse_args(list(argv or []))

    if a.model:
        # Measuring a body you have to switch to by hand is two commands where
        # one will do, and the switch is the same door `hcli use` knocks on.
        from .use import main as use_main
        if use_main([a.model, "--base", a.base]) != 0:
            return 2
    resident = health(a.base)
    if resident is None:
        print(f"nothing is answering on {a.base}/health.\n"
              f"Start one with:  python -m hcli serve --port "
              f"{a.base.rsplit(':', 1)[-1] or '8011'}", file=sys.stderr)
        return 2

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    report: Dict[str, Any] = {
        "schema": "hcli.report.v2",
        "timing_unit": "ns",
        "base_url": a.base,
        "resident": resident,
        "started_at": time.time(),
    }
    try:
        if "cold" not in skip:
            sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
            report["cold_prefill"] = cold_prefill(a.base, sizes, timeout=a.timeout)
        if "warm" not in skip:
            report["warm_turns"] = warm_turns(
                a.base, size=a.warm_size, turns=a.warm_turns, timeout=a.timeout)
        if "decode" not in skip:
            report["decode"] = decode_rate(
                a.base, max_tokens=a.decode_tokens, timeout=a.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        print(f"the surface refused a request ({exc.code}): {detail}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report["finished_at"] = time.time()

    out = Path(a.out) if a.out else (
        Path.home() / ".hcli" / "reports"
        / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["receipt"] = str(out)

    print(json.dumps(report, indent=2) if a.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
