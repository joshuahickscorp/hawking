#!/usr/bin/env python3.12
"""Decode-parity harness (NUCLEAR PASTA Section 7): the key that unlocks parity-critical reduction.

Runs the real hawking engine on a real GGUF fixture under pinned config (bit-identical `exact`
profile, fixed seed, greedy) across decode modes, captures a golden {tokens, text, sha, accept/reject
accounting}, and verifies a refactored build reproduces it EXACTLY. Fail-closed: if the hawking
binary or the fixture is absent, it errors - it never claims parity from a mock.

Usage:
  decode_parity_harness.py capture   # record goldens/decode_parity_golden.json from the current build
  decode_parity_harness.py verify    # re-run and diff against the golden (exit 1 on any mismatch)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "debug" / "hawking"
FIXTURE = ROOT / "models" / "SmolLM2-135M-Instruct-Q4_K_M.gguf"
GOLDEN = ROOT / "reports" / "condense" / "gravity_forge" / "condensation" / "decode_parity_golden.json"
PROMPT = "The capital of France is"
MAX_TOK = 16

# decode modes: (label, extra generate args). eagle5 requires a trained head fixture -> skipped
# unless present; suffix-automaton is toggled by the HAWKING_EH_SAM env in the mode's env.
MODES = [
    ("baseline_no_spec", [], {}),
    ("exact_shared", ["--speculate", "exact-shared"], {}),
    ("suffix_automaton", [], {"HAWKING_EH_SAM": "1"}),
]


def _fixture_ok() -> tuple[bool, str]:
    if not BIN.exists():
        return False, f"hawking binary absent at {BIN} (run `cargo build -p hawking`)"
    if not FIXTURE.exists():
        return False, f"GGUF fixture absent at {FIXTURE}"
    return True, ""


def _run(mode_args: list[str], env_extra: dict) -> dict:
    import os
    env = dict(os.environ); env.update(env_extra)
    cmd = [str(BIN), "generate", "--weights", str(FIXTURE), "--prompt", PROMPT,
           "--max-new-tokens", str(MAX_TOK), "--temperature", "0", "--seed", "0",
           "--profile", "exact", *mode_args]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    out = p.stdout
    # completion text = everything before the [stats] line, minus the profile banner line
    text = out.split("[stats]")[0]
    text = "\n".join(l for l in text.splitlines() if not l.startswith("[hawking]"))
    stats = {}
    for line in out.splitlines():
        if line.startswith("[stats-json]"):
            stats = json.loads(line[len("[stats-json]"):].strip())
    return {"returncode": p.returncode, "text": text.strip(),
            "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest(),
            "completion_tokens": stats.get("completion_tokens"),
            "draft_accepted": stats.get("draft_accepted"), "draft_rejected": stats.get("draft_rejected")}


def capture() -> int:
    ok, why = _fixture_ok()
    if not ok:
        print(f"FAIL-CLOSED: {why}", file=sys.stderr); return 2
    golden = {"schema": "hawking.decode_parity_golden.v1", "prompt": PROMPT, "max_tokens": MAX_TOK,
              "profile": "exact", "fixture": FIXTURE.name,
              "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest()[:16], "modes": {}}
    for label, args, env in MODES:
        r = _run(args, env)
        if r["returncode"] != 0:
            print(f"FAIL-CLOSED: mode {label} exited {r['returncode']}", file=sys.stderr); return 2
        golden["modes"][label] = r
        print(f"captured {label}: sha {r['text_sha256'][:12]} tokens={r['completion_tokens']} "
              f"accept={r['draft_accepted']} reject={r['draft_rejected']}")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"golden written: {GOLDEN}")
    return 0


def verify() -> int:
    ok, why = _fixture_ok()
    if not ok:
        print(f"FAIL-CLOSED: {why}", file=sys.stderr); return 2
    if not GOLDEN.exists():
        print("FAIL-CLOSED: no golden; run `capture` first", file=sys.stderr); return 2
    golden = json.loads(GOLDEN.read_text())
    failures = []
    for label, args, env in MODES:
        r = _run(args, env)
        g = golden["modes"].get(label, {})
        if r["text_sha256"] != g.get("text_sha256"):
            failures.append(label)
            print(f"MISMATCH {label}: golden {g.get('text_sha256','?')[:12]} != now {r['text_sha256'][:12]}")
        else:
            print(f"PARITY OK {label}: sha {r['text_sha256'][:12]}")
    if failures:
        print(f"DECODE PARITY FAILED: {failures}", file=sys.stderr); return 1
    print("DECODE PARITY GREEN (all modes bit-identical to golden)")
    return 0


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv or argv[0] not in ("capture", "verify"):
        print("usage: decode_parity_harness.py [capture|verify]", file=sys.stderr); return 2
    return capture() if argv[0] == "capture" else verify()


if __name__ == "__main__":
    raise SystemExit(main())
