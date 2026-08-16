#!/usr/bin/env python3
"""Multi-prompt greedy-id gate. Seals a baseline, then fails on any drift.

Why this exists: the coherence check used throughout 2026-08-16 was greedy-id
identity over 12 tokens on ONE prompt. That is enough to catch gross breakage,
and it did so repeatedly. It is not enough to certify a change.

The lm_head screen measured 11-13 greedy flips across 384 samples at Q4 - a
change that large in kind could pass a 12-token single-prompt check without
leaving a trace. Anything touching lm_head, embed, sampling, or numerics needs
a wider net than that.

Usage:
    coherence_gate.py seal   --binary B --args "..." --prompts P.txt --out SEAL.json
    coherence_gate.py verify --binary B --args "..." --prompts P.txt --seal SEAL.json

`--args` is passed through verbatim and must contain the artifact/tokenizer
flags for the model under test. The prompt is appended as `--prompt <text>`.

Every generate runs under tools/gpu_lane_lock.sh, because measurement lanes
share this GPU and an unlocked run corrupts both sides.

Exit codes: 0 pass, 1 drift detected, 2 could not run.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "tools" / "gpu_lane_lock.sh"


def generate(binary: str, args: str, prompt: str, lane: str) -> list[int] | None:
    """Run one greedy generate under the GPU lock. Returns token ids, or None."""
    cmd = [str(LOCK), lane, binary, *shlex.split(args), "--prompt", prompt]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.SubprocessError as e:
        print(f"  RUN FAILED: {e}", file=sys.stderr)
        return None
    if r.returncode != 0:
        print(f"  exit {r.returncode}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return None
    for line in r.stdout.splitlines():
        # harnesses print e.g. `generated_token_ids=[1, 2, 3]` or `NEW_TOKENS: [1, 2]`
        if "token_ids=" in line or "NEW_TOKENS" in line:
            frag = line.split("=", 1)[-1] if "=" in line else line.split(":", 1)[-1]
            frag = frag.strip()
            if frag.startswith("["):
                try:
                    return json.loads(frag)
                except json.JSONDecodeError:
                    continue
    print("  no token ids found in stdout", file=sys.stderr)
    return None


def run_all(binary: str, args: str, prompts: list[str], lane: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, p in enumerate(prompts, 1):
        ids = generate(binary, args, p, f"{lane}-{i}")
        if ids is None:
            print(f"  prompt {i}/{len(prompts)}: FAILED TO RUN", file=sys.stderr)
            sys.exit(2)
        out[p] = ids
        print(f"  prompt {i}/{len(prompts)}: {len(ids)} ids")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seal", "verify"])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--prompts", required=True, help="file, one prompt per line")
    ap.add_argument("--seal")
    ap.add_argument("--out")
    ap.add_argument("--lane", default="coherence-gate")
    a = ap.parse_args()

    prompts = [l.strip() for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    if len(prompts) < 2:
        print("REFUSED: a single prompt is what this tool exists to replace", file=sys.stderr)
        return 2
    print(f"{a.mode}: {len(prompts)} prompts")
    got = run_all(a.binary, a.args, prompts, a.lane)

    if a.mode == "seal":
        dest = Path(a.out or "coherence_seal.json")
        dest.write_text(json.dumps({"binary": a.binary, "args": a.args, "ids": got}, indent=2))
        print(f"sealed {len(got)} prompts -> {dest}")
        return 0

    seal = json.loads(Path(a.seal).read_text())["ids"]
    drift = [p for p in got if seal.get(p) != got[p]]
    missing = [p for p in got if p not in seal]
    for p in drift:
        print(f"DRIFT on {p!r}:\n  sealed {seal.get(p)}\n  got    {got[p]}", file=sys.stderr)
    if missing:
        print(f"NOT IN SEAL (cannot certify): {missing}", file=sys.stderr)
    if drift or missing:
        print(f"\nFAIL: {len(drift)} drifted, {len(missing)} unsealed", file=sys.stderr)
        return 1
    print(f"PASS: {len(got)} prompts id-identical to seal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
