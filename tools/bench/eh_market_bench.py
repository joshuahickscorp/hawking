#!/usr/bin/env python3
"""Event Horizon proposal-market bench.

Drives the hawking CLI over three decode arms:
  A  plain greedy      HAWKING_QWEN_USER_DRAFT=0  HAWKING_QWEN_EVENT_HORIZON=0
  B  n-gram only       HAWKING_QWEN_USER_DRAFT=1  HAWKING_QWEN_EVENT_HORIZON=0
  C  full free market  HAWKING_QWEN_USER_DRAFT=1  HAWKING_QWEN_EVENT_HORIZON=1

Primary metric: accepted_tps = (accepted_draft_tokens + greedy_tokens) / wall_s
                             = completion_tokens / wall_s  (greedy = always 1 per cycle)
The [stats] line on stderr carries all we need; no extra instrumentation.

SMOKE MODE (--smoke):
  1 short prompt, max_new_tokens=16, 1 trial per arm.
  GPU footprint: 3 * 16 = 48 forward steps, < 1s per run on M3.

LOG FORMAT (from crates/hawking/src/main.rs):
  [stats] reason=... prompt=N completion=N prefill_ms=F decode_ms=F dec_tps=F
          dispatches_per_fwd=N draft_accepted=N draft_rejected=N profile=...
  [stats-json] {JSON}  (also available; we prefer the flat line for robustness)

USAGE
  python3 tools/bench/eh_market_bench.py --smoke --model models/qwen2.5-3b-instruct-q4_k_m.gguf
  python3 tools/bench/eh_market_bench.py --model models/... --trials 3 --max-tokens 32
  python3 tools/bench/eh_market_bench.py --selftest

NOTE: no new pip deps beyond tools/training/requirements.txt (numpy, etc.).
      This file uses only the stdlib + numpy (already required).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo root: two levels up from tools/bench/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
REPORT_OUT = REPORTS_DIR / "eh_market_bench_smoke.md"

# Default binary: prefer pre-built release; caller can override via --bin
DEFAULT_BIN = str(REPO_ROOT / "target" / "release" / "hawking")

# Default kernel profile
DEFAULT_PROFILE = str(REPO_ROOT / "profiles" / "qwen3b-instruct-q4k.m3pro18.json")

# Base env: locked Qwen fast-path (matches ab_lever.sh / user_draft_3arm_bench.sh)
BASE_ENV_VARS: Dict[str, str] = {
    "HAWKING_QWEN_TCB": "1",
    "HAWKING_QWEN_VOCAB_PRUNE": "32000",
    "HAWKING_QWEN_Q4K_LMHEAD": "1",
    "HAWKING_QWEN_FFN_DOWN_Q4K": "1",
    "HAWKING_QWEN_Q4K_PREDEC": "1",
}

# ---------------------------------------------------------------------------
# Arm definitions
ARM_LABELS = {
    "A": "plain greedy",
    "B": "n-gram only",
    "C": "full free market",
}

ARM_EXTRA_ENV: Dict[str, Dict[str, str]] = {
    "A": {"HAWKING_QWEN_USER_DRAFT": "0", "HAWKING_QWEN_EVENT_HORIZON": ""},
    "B": {"HAWKING_QWEN_USER_DRAFT": "1"},
    "C": {"HAWKING_QWEN_USER_DRAFT": "1", "HAWKING_QWEN_EVENT_HORIZON": "1"},
}
# Keys with empty-string values are DELETED from the environment (disable).


# ---------------------------------------------------------------------------
# Embedded mini-prompts (used when no --prompt-file given).
# Three classes: repetitive code (high n-gram hits), chat (medium), prose (low).
MINI_PROMPTS: List[Dict[str, str]] = [
    {
        "tag": "repetitive_code",
        "text": (
            "fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc } "
            "fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc } "
            "fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc } "
        ),
    },
    {
        "tag": "chat",
        "text": "What is the capital of France? Give a one-sentence answer.",
    },
    {
        "tag": "prose",
        "text": (
            "Explain the tradeoffs between optimistic and pessimistic concurrency "
            "control in distributed databases."
        ),
    },
]

SMOKE_PROMPT: Dict[str, str] = {
    "tag": "smoke",
    "text": (
        "fn add(a: u32, b: u32) -> u32 { a + b } "
        "fn add(a: u32, b: u32) -> u32 { a + b } "
        "fn add(a: u32, b: u32) -> u32 {"
    ),
}

# ---------------------------------------------------------------------------
# Stats line parser — matches the [stats] eprintln in crates/hawking/src/main.rs:
#   [stats] reason=X prompt=N completion=N prefill_ms=F decode_ms=F dec_tps=F
#           dispatches_per_fwd=N draft_accepted=N draft_rejected=N profile=X
_STATS_RE = re.compile(
    r"\[stats\]"
    r".*?completion=(?P<completion>\d+)"
    r".*?decode_ms=(?P<decode_ms>[0-9.]+)"
    r".*?dec_tps=(?P<dec_tps>[0-9.]+)"
    r".*?draft_accepted=(?P<draft_accepted>\d+)"
    r".*?draft_rejected=(?P<draft_rejected>\d+)"
)

# [stats-json] fallback
_STATS_JSON_RE = re.compile(r"\[stats-json\]\s+(\{.+\})")


def _parse_stats(stderr: str) -> Optional[Dict]:
    """Extract stats from hawking stderr. Prefers [stats] line; falls back to [stats-json]."""
    for line in reversed(stderr.splitlines()):
        m = _STATS_RE.search(line)
        if m:
            d = m.groupdict()
            return {
                "completion_tokens": int(d["completion"]),
                "decode_ms": float(d["decode_ms"]),
                "dec_tps": float(d["dec_tps"]),
                "draft_accepted": int(d["draft_accepted"]),
                "draft_rejected": int(d["draft_rejected"]),
            }
    # Fallback: [stats-json]
    for line in reversed(stderr.splitlines()):
        m = _STATS_JSON_RE.search(line)
        if m:
            try:
                j = json.loads(m.group(1))
                return {
                    "completion_tokens": int(j.get("completion_tokens", 0)),
                    "decode_ms": float(j.get("decode_ms", 1.0)),
                    "dec_tps": float(j.get("dec_tps", 0.0)),
                    "draft_accepted": int(j.get("draft_accepted", 0)),
                    "draft_rejected": int(j.get("draft_rejected", 0)),
                }
            except (ValueError, KeyError):
                pass
    return None


# ---------------------------------------------------------------------------
def _build_env(arm: str) -> Dict[str, str]:
    """Build subprocess environment for the given arm."""
    env = dict(os.environ)
    env.update(BASE_ENV_VARS)
    for k, v in ARM_EXTRA_ENV.get(arm, {}).items():
        if v == "":
            env.pop(k, None)
        else:
            env[k] = v
    return env


def _run_one(
    bin_path: str,
    model: str,
    kernel_profile: Optional[str],
    prompt: str,
    arm: str,
    max_tokens: int,
    seed: int = 0,
    timeout: int = 120,
) -> Tuple[Optional[Dict], str]:
    """Run one generation trial. Returns (stats_dict_or_None, raw_stderr)."""
    cmd = [
        bin_path, "generate",
        "--weights", model,
        "--prompt", prompt,
        "--max-new-tokens", str(max_tokens),
        "--temperature", "0",
        "--seed", str(seed),
    ]
    if kernel_profile and Path(kernel_profile).is_file():
        cmd += ["--kernel-profile", kernel_profile]

    env = _build_env(arm)
    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        stats = _parse_stats(stderr)
        return stats, stderr
    except subprocess.TimeoutExpired:
        return None, f"[TIMEOUT after {timeout}s]"
    except FileNotFoundError:
        return None, f"[BINARY NOT FOUND: {bin_path}]"


# ---------------------------------------------------------------------------
def _accepted_tps(stats: Dict) -> float:
    """accepted_tps = completion_tokens / wall_s.

    completion_tokens already includes all accepted drafts + greedy bonus tokens.
    wall_s = decode_ms / 1000; decode_ms excludes prefill.
    When draft_accepted=0 (plain greedy), this is just dec_tps — consistent.
    """
    wall_s = stats["decode_ms"] / 1000.0
    if wall_s <= 0:
        return 0.0
    return stats["completion_tokens"] / wall_s


def _mean_accept_rate(stats: Dict) -> float:
    total = stats["draft_accepted"] + stats["draft_rejected"]
    if total == 0:
        return float("nan")
    return stats["draft_accepted"] / total


# ---------------------------------------------------------------------------
def run_bench(
    bin_path: str,
    model: str,
    kernel_profile: Optional[str],
    prompts: List[Dict[str, str]],
    arms: List[str],
    max_tokens: int,
    trials: int,
    verbose: bool = False,
) -> Dict:
    """Run all (arm, prompt, trial) combinations and return aggregated results."""
    import statistics

    results: Dict[str, Dict[str, List[Dict]]] = {arm: {} for arm in arms}

    for prompt_d in prompts:
        tag = prompt_d["tag"]
        text = prompt_d["text"]
        for arm in arms:
            if verbose:
                print(
                    f"  arm={arm} ({ARM_LABELS[arm]}) prompt={tag!r} trials={trials} ...",
                    file=sys.stderr,
                )
            trial_stats = []
            for t in range(trials):
                stats, stderr = _run_one(
                    bin_path, model, kernel_profile, text, arm, max_tokens, seed=t
                )
                if stats is None:
                    if verbose:
                        print(f"    trial {t}: FAILED stderr={stderr[:200]!r}", file=sys.stderr)
                    trial_stats.append(None)
                else:
                    atps = _accepted_tps(stats)
                    mar = _mean_accept_rate(stats)
                    if verbose:
                        print(
                            f"    trial {t}: dec_tps={stats['dec_tps']:.2f}"
                            f" accepted_tps={atps:.2f}"
                            f" accept_rate={mar:.3f}"
                            f" accepted={stats['draft_accepted']}"
                            f" rejected={stats['draft_rejected']}",
                            file=sys.stderr,
                        )
                    trial_stats.append({**stats, "accepted_tps": atps, "mean_accept_rate": mar})
            results[arm][tag] = trial_stats

    # Aggregate: median over valid trials
    agg: Dict[str, Dict[str, Dict]] = {}
    for arm in arms:
        agg[arm] = {}
        for prompt_d in prompts:
            tag = prompt_d["tag"]
            valid = [t for t in results[arm][tag] if t is not None]
            if not valid:
                agg[arm][tag] = {
                    "median_accepted_tps": float("nan"),
                    "median_dec_tps": float("nan"),
                    "mean_accept_rate": float("nan"),
                    "n_valid": 0,
                }
            else:
                atps_vals = sorted(t["accepted_tps"] for t in valid)
                dtps_vals = sorted(t["dec_tps"] for t in valid)
                mar_vals = [t["mean_accept_rate"] for t in valid if not (t["mean_accept_rate"] != t["mean_accept_rate"])]
                agg[arm][tag] = {
                    "median_accepted_tps": _median(atps_vals),
                    "median_dec_tps": _median(dtps_vals),
                    "mean_accept_rate": (sum(mar_vals) / len(mar_vals)) if mar_vals else float("nan"),
                    "n_valid": len(valid),
                }
    return {"raw": results, "agg": agg}


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
def format_table(
    agg: Dict,
    arms: List[str],
    prompts: List[Dict],
    timestamp: str,
) -> str:
    """Render a Markdown table comparing the three arms."""
    prompt_tags = [p["tag"] for p in prompts]
    lines = []
    lines.append(f"# Event Horizon Market Bench — {timestamp}")
    lines.append("")
    lines.append("Primary metric: **accepted_tps** = completion_tokens / decode_s")
    lines.append("(plain greedy has 0 drafts so accepted_tps == dec_tps; ratios vs A are meaningful)")
    lines.append("")

    for tag in prompt_tags:
        lines.append(f"## Prompt: `{tag}`")
        lines.append("")
        lines.append("| Arm | Config | accepted_tps | dec_tps | mean_accept_rate | n_valid |")
        lines.append("|-----|--------|:------------:|:-------:|:----------------:|:-------:|")
        base_atps = None
        for arm in arms:
            d = agg.get(arm, {}).get(tag, {})
            atps = d.get("median_accepted_tps", float("nan"))
            dtps = d.get("median_dec_tps", float("nan"))
            mar = d.get("mean_accept_rate", float("nan"))
            n = d.get("n_valid", 0)

            atps_str = f"{atps:.2f}" if atps == atps else "n/a"
            dtps_str = f"{dtps:.2f}" if dtps == dtps else "n/a"
            mar_str = f"{mar:.3f}" if mar == mar else "n/a"

            if arm == "A":
                base_atps = atps if atps == atps else None
                ratio_str = "(baseline)"
            elif base_atps and base_atps > 0 and atps == atps:
                ratio = atps / base_atps
                ratio_str = f"{atps_str} ({ratio:+.2f}x)"
                atps_str = ratio_str
            else:
                ratio_str = atps_str

            lines.append(
                f"| {arm} | {ARM_LABELS[arm]} | {atps_str} | {dtps_str} | {mar_str} | {n} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(
        "> accepted_tps ratio vs A: >1.0 means speculative decoding is paying off. "
        "Arm B = n-gram only; Arm C = full free market (n-gram + suffix-array via router)."
    )
    lines.append("> mean_accept_rate = draft_accepted / (draft_accepted + draft_rejected). "
                 "nan for arm A (no drafts).")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def selftest() -> bool:
    """Verify the stats parser against known log strings."""
    sample_line = (
        "[stats] reason=max_tokens prompt=12 completion=16 prefill_ms=45.3 decode_ms=310.2 "
        "dec_tps=51.58 dispatches_per_fwd=115 draft_accepted=8 draft_rejected=4 profile=none"
    )
    s = _parse_stats(sample_line)
    assert s is not None, "parser returned None on valid [stats] line"
    assert s["completion_tokens"] == 16, f"completion_tokens={s['completion_tokens']}"
    assert abs(s["decode_ms"] - 310.2) < 1e-6, f"decode_ms={s['decode_ms']}"
    assert abs(s["dec_tps"] - 51.58) < 1e-6, f"dec_tps={s['dec_tps']}"
    assert s["draft_accepted"] == 8, f"draft_accepted={s['draft_accepted']}"
    assert s["draft_rejected"] == 4, f"draft_rejected={s['draft_rejected']}"
    atps = _accepted_tps(s)
    assert atps > 0, f"accepted_tps={atps}"
    # plain greedy: accepted=0, rejected=0 -> accepted_tps == dec_tps
    plain = {"completion_tokens": 32, "decode_ms": 500.0, "dec_tps": 64.0,
             "draft_accepted": 0, "draft_rejected": 0}
    assert abs(_accepted_tps(plain) - 64.0) < 1e-6, f"plain greedy accepted_tps mismatch"
    assert _mean_accept_rate(plain) != _mean_accept_rate(plain), "nan expected for 0 drafts"
    mar = _mean_accept_rate(s)
    assert abs(mar - 8 / 12) < 1e-6, f"mean_accept_rate={mar}"
    print("selftest PASSED")
    return True


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="models/qwen2.5-3b-instruct-q4_k_m.gguf",
                    help="Path to GGUF weights (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)")
    ap.add_argument("--bin", default=DEFAULT_BIN,
                    help="Path to hawking binary (default: target/release/hawking)")
    ap.add_argument("--kernel-profile", default=DEFAULT_PROFILE,
                    help="Kernel profile JSON path")
    ap.add_argument("--prompt-file", default=None,
                    help="JSON file with list of {tag, text} prompt objects. "
                         "If omitted, uses 3 embedded mini-prompts.")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="max_new_tokens per trial (default: 32; smoke forces 16)")
    ap.add_argument("--trials", type=int, default=1,
                    help="Trials per (arm, prompt) combination (default: 1)")
    ap.add_argument("--arms", default="A,B,C",
                    help="Comma-separated arm IDs to run (default: A,B,C)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: 1 prompt, 16 tokens, 1 trial per arm. "
                         "GPU footprint: 3 * 16 = 48 forward steps.")
    ap.add_argument("--out", default=str(REPORT_OUT),
                    help=f"Output Markdown path (default: {REPORT_OUT})")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="Run parser self-test (no GPU) and exit.")
    args = ap.parse_args()

    if args.selftest:
        return 0 if selftest() else 1

    # Resolve paths relative to repo root
    model_path = args.model
    if not Path(model_path).is_absolute():
        model_path = str(REPO_ROOT / model_path)
    bin_path = args.bin
    if not Path(bin_path).is_absolute():
        bin_path = str(REPO_ROOT / bin_path)
    kp = args.kernel_profile
    if kp and not Path(kp).is_absolute():
        kp = str(REPO_ROOT / kp)

    # Validate binary
    if not Path(bin_path).is_file():
        print(
            f"[eh_market_bench] ERROR: binary not found: {bin_path}\n"
            "  Build with: cargo build --release -p hawking",
            file=sys.stderr,
        )
        return 1

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_LABELS:
            print(f"[eh_market_bench] ERROR: unknown arm {a!r}. Valid: A, B, C", file=sys.stderr)
            return 1

    # Smoke overrides
    if args.smoke:
        prompts = [SMOKE_PROMPT]
        max_tokens = 16
        trials = 1
        print(
            "[eh_market_bench] SMOKE: 1 prompt, 16 tokens, 1 trial per arm "
            f"(arms: {', '.join(arms)})",
            file=sys.stderr,
        )
    else:
        max_tokens = args.max_tokens
        trials = args.trials
        if args.prompt_file:
            with open(args.prompt_file) as f:
                prompts = json.load(f)
        else:
            prompts = MINI_PROMPTS

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(
        f"[eh_market_bench] model={model_path}  bin={bin_path}  "
        f"max_tokens={max_tokens}  trials={trials}  arms={arms}",
        file=sys.stderr,
    )

    bench_result = run_bench(
        bin_path=bin_path,
        model=model_path,
        kernel_profile=kp,
        prompts=prompts,
        arms=arms,
        max_tokens=max_tokens,
        trials=trials,
        verbose=args.verbose or args.smoke,
    )

    agg = bench_result["agg"]
    table_md = format_table(agg, arms, prompts, timestamp)

    # Print to stdout
    print(table_md)

    # Write to report file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_md, encoding="utf-8")
    print(f"[eh_market_bench] wrote {out_path}", file=sys.stderr)

    # Also write raw JSON alongside
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        # Make JSON-serializable: replace nan with null
        def _clean(obj):
            if isinstance(obj, float):
                return None if obj != obj else obj
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj
        json.dump(
            {
                "bench": "eh_market_bench",
                "timestamp": timestamp,
                "model": model_path,
                "max_tokens": max_tokens,
                "trials": trials,
                "arms": {a: ARM_LABELS[a] for a in arms},
                "agg": _clean(agg),
            },
            f,
            indent=2,
        )
    print(f"[eh_market_bench] wrote {json_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
