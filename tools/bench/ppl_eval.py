#!/usr/bin/env python3
"""
tools/bench/ppl_eval.py — WikiText-2 perplexity oracle for dismantle.

Path-to-90 Stage 1 / B1. Quality gate for KV-cache and expert-quant
variants (A2 Q8 latent KV, B2 WHT 3-bit KV, B3 hot/cold expert tiering).
Without this oracle, "ΔPPL within +0.5%" can't be measured and those
levers can only be scored by token-identical greedy at 64 tokens
(batch-hash), which is too brittle for any lossy quant change.

The heavy lifting (model forward + log-softmax NLL) lives in the Rust
`dismantle ppl-eval` subcommand; this script handles dataset prep and
delta computation between variant runs.

Subcommands
-----------
  prep    Download WikiText-2 test split, slice a deterministic
          N-sample subset, write JSON-lines to disk. One-time per slice.
  run     Invoke `dismantle ppl-eval` with the given kernel profile,
          parse the JSON-lines output, print PPL summary. Optionally
          accepts a baseline `--diff-baseline` JSONL to compute per-sample
          ΔNLL and corpus ΔPPL.
  diff    Compare two pre-existing ppl-eval JSONL outputs. Prints
          ΔPPL (variant - baseline) and per-sample NLL delta histogram.

Quality bar
-----------
The plan (immutable-jellyfish.md §A2 / §B2) defines passing as:
  - token-identical greedy at 64 tokens vs the FP16 baseline
    (use `dismantle batch-hash --tokens 64` for that half)
  - ΔPPL within +0.5% of the FP16 baseline (this script's domain)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "tests/data/wikitext2_256_samples.jsonl"
DEFAULT_WEIGHTS = REPO_ROOT / "models/deepseek-v2-lite-q4.gguf"
DEFAULT_PROFILE = REPO_ROOT / "profiles/deepseek-v2-lite-q4.m3pro18.json"
DEFAULT_BINARY = REPO_ROOT / "target/release/dismantle"


# -----------------------------------------------------------------------------
# prep — slice WikiText-2 test split deterministically
# -----------------------------------------------------------------------------

def cmd_prep(args: argparse.Namespace) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: `datasets` package not found. Install with\n"
            "  /Library/Frameworks/Python.framework/Versions/3.12/bin/pip install datasets",
            file=sys.stderr,
        )
        return 2

    out = pathlib.Path(args.out)
    if out.exists() and not args.force:
        print(f"[prep] {out} already exists; pass --force to overwrite", file=sys.stderr)
        return 0

    print(f"[prep] loading {args.dataset} / {args.config} split={args.split}", file=sys.stderr)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    print(f"[prep] loaded {len(ds)} rows", file=sys.stderr)

    paragraphs: List[str] = []
    for row in ds:
        t = row["text"]
        # Drop blanks, section headers, and very short rows: those produce
        # noisy NLL averages and don't reflect typical decode workload.
        if not t or t.startswith(" =") or len(t.strip()) < args.min_chars:
            continue
        paragraphs.append(t.rstrip("\n"))
    print(f"[prep] {len(paragraphs)} substantive paragraphs", file=sys.stderr)
    if len(paragraphs) < args.n:
        print(
            f"ERROR: only {len(paragraphs)} paragraphs available; need {args.n}",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    sampled = rng.sample(paragraphs, args.n)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, text in enumerate(sampled):
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")
    chars = sum(len(p) for p in sampled)
    print(f"[prep] wrote {out} ({args.n} samples, {chars} chars)", file=sys.stderr)
    return 0


# -----------------------------------------------------------------------------
# run — invoke dismantle ppl-eval, parse summary, optionally diff baseline
# -----------------------------------------------------------------------------

def _load_jsonl(path: pathlib.Path) -> Tuple[Dict[str, dict], dict]:
    """Return (id -> per-sample row, summary)."""
    per_sample: Dict[str, dict] = {}
    summary: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "summary" in row:
                summary = row["summary"]
                continue
            per_sample[str(row["id"])] = row
    return per_sample, summary


def cmd_run(args: argparse.Namespace) -> int:
    samples = pathlib.Path(args.samples)
    if not samples.exists():
        print(f"ERROR: samples file {samples} not found. Run `ppl_eval.py prep` first.", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    binary = pathlib.Path(args.binary)
    if not binary.exists():
        print(f"ERROR: dismantle binary not found at {binary}. Run `cargo build --release`.", file=sys.stderr)
        return 2

    cmd = [
        str(binary),
        "ppl-eval",
        "--weights", str(args.weights),
        "--samples", str(samples),
        "--max-tokens", str(args.max_tokens),
        "--out", str(out),
    ]
    if args.kernel_profile:
        cmd += ["--kernel-profile", str(args.kernel_profile)]

    print(f"[run] $ {' '.join(cmd)}", file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[run] dismantle ppl-eval exited {rc}", file=sys.stderr)
        return rc

    variant_samples, variant_summary = _load_jsonl(out)
    print(f"\n[run] variant: {out}")
    print(f"  samples       = {variant_summary.get('samples')}")
    print(f"  tokens_scored = {variant_summary.get('tokens_scored')}")
    print(f"  avg_nll       = {variant_summary.get('avg_nll'):.6f}")
    print(f"  PPL           = {variant_summary.get('ppl'):.4f}")
    print(f"  model         = {variant_summary.get('model_id')}")
    print(f"  profile       = {variant_summary.get('profile_id')}")
    print(f"  elapsed_s     = {variant_summary.get('elapsed_s'):.1f}")

    if args.diff_baseline:
        baseline_path = pathlib.Path(args.diff_baseline)
        if not baseline_path.exists():
            print(f"ERROR: baseline {baseline_path} not found", file=sys.stderr)
            return 2
        _diff_runs(baseline_path, out, args.delta_ppl_threshold_pct)

    return 0


# -----------------------------------------------------------------------------
# diff — compare two existing ppl-eval JSONL outputs
# -----------------------------------------------------------------------------

def _diff_runs(
    baseline_path: pathlib.Path,
    variant_path: pathlib.Path,
    delta_threshold_pct: float,
) -> Tuple[float, float, int]:
    """Compute and print ΔPPL + per-sample NLL distribution. Returns (delta_pct, baseline_ppl, ranked_count)."""
    bsamp, bsumm = _load_jsonl(baseline_path)
    vsamp, vsumm = _load_jsonl(variant_path)

    baseline_ppl = bsumm["ppl"]
    variant_ppl = vsumm["ppl"]
    delta_ppl = variant_ppl - baseline_ppl
    delta_pct = 100.0 * delta_ppl / baseline_ppl

    # Sample-level NLL deltas where IDs overlap.
    shared_ids = sorted(set(bsamp.keys()) & set(vsamp.keys()))
    deltas: List[Tuple[str, float, int]] = []
    for sid in shared_ids:
        b = bsamp[sid]
        v = vsamp[sid]
        scored = min(b["tokens_scored"], v["tokens_scored"])
        if scored <= 0:
            continue
        b_avg = b["nll_sum"] / b["tokens_scored"]
        v_avg = v["nll_sum"] / v["tokens_scored"]
        deltas.append((sid, v_avg - b_avg, scored))

    print(f"\n[diff] baseline = {baseline_path}")
    print(f"[diff] variant  = {variant_path}")
    print(f"  baseline PPL    = {baseline_ppl:.4f}")
    print(f"  variant  PPL    = {variant_ppl:.4f}")
    print(f"  ΔPPL            = {delta_ppl:+.4f}  ({delta_pct:+.3f}%)")
    print(f"  threshold       = ±{delta_threshold_pct:.3f}%")
    if abs(delta_pct) <= delta_threshold_pct:
        print(f"  RESULT          = PASS (within ±{delta_threshold_pct:.3f}%)")
    else:
        print(f"  RESULT          = FAIL (exceeds ±{delta_threshold_pct:.3f}%)")

    # Per-sample NLL/token delta distribution.
    if deltas:
        deltas.sort(key=lambda x: x[1])
        per_sample = [d[1] for d in deltas]
        n = len(per_sample)
        print(f"\n  per-sample ΔNLL/tok (n={n} shared samples):")
        print(f"    min   = {per_sample[0]:+.5f}")
        print(f"    p10   = {per_sample[max(0, n // 10)]:+.5f}")
        print(f"    p50   = {per_sample[n // 2]:+.5f}")
        print(f"    p90   = {per_sample[min(n - 1, n - 1 - n // 10)]:+.5f}")
        print(f"    max   = {per_sample[-1]:+.5f}")
        # Worst regressors.
        worst = sorted(deltas, key=lambda x: -x[1])[:5]
        print(f"    worst 5: {[(d[0], round(d[1], 4)) for d in worst]}")

    return (delta_pct, baseline_ppl, len(deltas))


def cmd_diff(args: argparse.Namespace) -> int:
    _diff_runs(
        pathlib.Path(args.baseline),
        pathlib.Path(args.variant),
        args.delta_ppl_threshold_pct,
    )
    return 0


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="Download WikiText-2 and slice deterministic sample subset")
    pp.add_argument("--out", default=str(DEFAULT_SAMPLES))
    pp.add_argument("--dataset", default="wikitext")
    pp.add_argument("--config", default="wikitext-2-raw-v1")
    pp.add_argument("--split", default="test")
    pp.add_argument("--n", type=int, default=256, help="Number of samples")
    pp.add_argument("--seed", type=int, default=20260515)
    pp.add_argument("--min-chars", type=int, default=80, help="Min paragraph length filter")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_prep)

    pr = sub.add_parser("run", help="Invoke dismantle ppl-eval; emit summary; optionally diff baseline")
    pr.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    pr.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    pr.add_argument("--kernel-profile", default=str(DEFAULT_PROFILE))
    pr.add_argument("--max-tokens", type=int, default=128)
    pr.add_argument("--out", required=True, help="Output JSONL path")
    pr.add_argument("--binary", default=str(DEFAULT_BINARY))
    pr.add_argument("--diff-baseline", default=None, help="Optional baseline JSONL to diff against")
    pr.add_argument("--delta-ppl-threshold-pct", type=float, default=0.5)
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("diff", help="Compare two pre-existing ppl-eval JSONL outputs")
    pd.add_argument("--baseline", required=True)
    pd.add_argument("--variant", required=True)
    pd.add_argument("--delta-ppl-threshold-pct", type=float, default=0.5)
    pd.set_defaults(func=cmd_diff)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
