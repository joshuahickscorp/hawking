#!/usr/bin/env python3
"""Does the fast artifact actually say anything true?

The runtime A/B measured tok/s on the prompt "Say hi." and never looked at what
came out. That is the shape this repository has already been burned by: an
artifact served at full speed and answered "capital of France" with " combust".
The runtime was fine, the weights were not, and nobody noticed because every
measurement had been a speed measurement.

So this gate asks the only question that makes a tok/s number mean anything:
given two prompts with checkable answers, does each arm produce the answer? A
fast incoherent artifact is a negative result, not a win.

The gate must be able to fail, so `--damage` points the native arm at a
deliberately corrupted catalog and the run must come back FAIL. A correctness
gate never seen to reject anything is the same defect as the speed number it
was built to qualify.

    python3 tools/headless/runtime_correctness_gate.py
    python3 tools/headless/runtime_correctness_gate.py --damage
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "headless"))

# Reuse the A/B's own resolution so this gate qualifies the SAME artifacts and
# the SAME binary that produced the timing. A gate over a different build would
# qualify nothing.
import runtime_experiment as rx  # noqa: E402

# Deliberately boring and unambiguous: one factual completion, one small
# arithmetic. Both have a single right answer that a coherent 27B gets and an
# incoherent one does not, and neither depends on instruction-following style.
# The native path applies the chat template with thinking ON, so the first
# tokens are always a `<think>` block. Twelve tokens never escape it, and an
# answer that has not been reached yet is not a wrong answer -- scoring it as
# FAIL would have reported a coherent artifact as broken. The llama arm runs
# with reasoning off at the server, so it needs far fewer.
NATIVE_MAX_NEW = 160
LLAMA_MAX_NEW = 16

CASES = [
    {
        "id": "factual",
        "prompt": "The capital of France is",
        "accept": [r"\bparis\b"],
    },
    {
        "id": "arithmetic",
        "prompt": "17 * 19 =",
        "accept": [r"\b323\b"],
    },
]


def _passes(text: str, accept: List[str]) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in accept)



def _corrupt(victim: Path, artifact: Path, dst: Path) -> None:
    """Zero the second half of one file, never writing through the hardlink.

    The catalog is hardlinked to save 14 GB of copying, so writing in place
    would corrupt the REAL artifact. The link is broken first and the file
    rebuilt as its own inode; the original's inode, size and mtime are asserted
    unchanged afterwards.
    """
    original = artifact / victim.relative_to(dst)
    before = original.stat()
    size = victim.stat().st_size
    victim.unlink()
    with open(victim, "wb") as fh:
        with open(original, "rb") as src:
            fh.write(src.read(size // 2))
        fh.write(b"\x00" * (size - size // 2))
        fh.truncate(size)
    after = original.stat()
    assert (after.st_ino, after.st_size, after.st_mtime) == (
        before.st_ino, before.st_size, before.st_mtime), (
        f"the real artifact changed at {original}; refusing to continue")


def native_case(case: Dict[str, Any], artifact: Path) -> Dict[str, Any]:
    binp = rx.binary_path()
    with tempfile.TemporaryDirectory(prefix="rcg-") as tmp:
        out = Path(tmp) / "o.json"
        cmd = [
            str(binp),
            "--artifact-root", str(artifact),
            "--tokenizer", str(rx.TOKENIZER),
            "--prompt", case["prompt"],
            "--max-new-tokens", str(NATIVE_MAX_NEW),
            "--max-seq-len", str(rx.MAX_SEQ_LEN),
            "--out", str(out),
        ]
        p = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True,
            env=os.environ.copy(), timeout=1800,
        )
        body: Any = None
        if out.is_file():
            try:
                body = json.loads(out.read_text())
            except Exception:
                body = None
    text = ""
    if isinstance(body, dict):
        for key in ("generated_text", "full_decode", "text", "completion", "output"):
            if isinstance(body.get(key), str):
                text = body[key]
                break
    if not text:
        # Fall back to stdout. Say so, rather than silently scoring "" as a fail
        # -- "the harness could not find the text" and "the model said nothing"
        # are different results and must not share a shape.
        text = (p.stdout or "")[-2000:]
    return {
        "arm": "native_gravity_q4",
        "case": case["id"],
        "prompt": case["prompt"],
        "exit_code": p.returncode,
        "text": text[-400:],
        "text_source": "json" if isinstance(body, dict) else "stdout_tail",
        "pass": bool(p.returncode == 0 and _passes(text, case["accept"])),
    }


def llama_case(case: Dict[str, Any], port: int) -> Dict[str, Any]:
    payload = {
        "prompt": case["prompt"],
        "n_predict": LLAMA_MAX_NEW,
        "temperature": 0.0,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        text = str(body.get("content") or "")
        err = None
    except Exception as e:
        text, err = "", str(e)
    return {
        "arm": "llama_cpp_q5k",
        "case": case["id"],
        "prompt": case["prompt"],
        "error": err,
        "text": text[-400:],
        "pass": bool(err is None and _passes(text, case["accept"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--damage", action="store_true",
        help="point the native arm at a corrupted catalog; the gate MUST fail",
    )
    ap.add_argument(
        "--damage-frac", type=float, default=0.25,
        help="fraction of tensor files to corrupt under --damage",
    )
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    artifact = Path(rx.ARTIFACT)
    damaged_root = None

    if args.damage:
        # Copy the catalog and corrupt one weight file. The original artifact is
        # never touched -- this repo's rule is that model weights are not
        # destroyed to make a point.
        # HARDLINK the catalog rather than copying 14 GB, then replace the
        # victim with a fresh corrupted file. Writing THROUGH a hardlink would
        # corrupt the real artifact, which is why the victim is unlinked first
        # and rebuilt as its own inode. The original is never modified.
        damaged_root = Path(tempfile.mkdtemp(prefix="rcg-damaged-"))
        dst = damaged_root / artifact.name
        shutil.copytree(
            artifact, dst, symlinks=True,
            copy_function=os.link,
        )
        victims = sorted(
            [p for p in dst.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000]
        )
        if not victims:
            print("FAIL: no file large enough to corrupt; the damage control "
                  "cannot run, so this proves nothing")
            return 2
        # Zeroing ONE 16 MB tensor out of a 14 GB catalog did not change the
        # answer -- the model shrugged it off and the gate correctly refused to
        # call that a demonstration. So the damage is a FRACTION of the catalog,
        # and --damage-frac records how much corruption it actually takes before
        # output degrades. That number is worth more than a binary pass: it is
        # the gate's sensitivity.
        frac = max(0.0, min(1.0, args.damage_frac))
        chosen = victims[:: max(1, int(round(1.0 / frac)))] if frac else victims[:1]
        print(f"corrupting {len(chosen)} of {len(victims)} tensor files "
              f"(~{100.0 * len(chosen) / max(1, len(victims)):.0f}%)")
        for victim in chosen:
            _corrupt(victim, artifact, dst)
        victim = chosen[-1]
        size = victim.stat().st_size
        print(f"last victim {victim.relative_to(dst)} ({size} bytes); "
              f"originals verified untouched")
        artifact = dst

    # rx.pick_llama_port takes the foreign-load survey; discovering the port
    # directly is simpler and does not depend on that survey's shape.
    port = None
    try:
        lsof = subprocess.run(
            ["bash", "-lc",
             "lsof -iTCP -sTCP:LISTEN -P -a -c llama-server | "
             "awk 'NR>1{print $9}' | sed 's/.*://' | head -1"],
            capture_output=True, text=True, timeout=30,
        )
        cand = (lsof.stdout or "").strip()
        port = int(cand) if cand.isdigit() else None
    except Exception:
        port = None
    if port is None:
        print("note: no resident llama-server found; the llama arm is NOT_PROBED "
              "and this run qualifies the native arm only")

    for case in CASES:
        n = native_case(case, artifact)
        results.append(n)
        print(f"{'ok  ' if n['pass'] else 'FAIL'} native/{case['id']}: "
              f"{n['text'].strip()[:90]!r}")
        if port and not args.damage:
            l = llama_case(case, port)
            results.append(l)
            print(f"{'ok  ' if l['pass'] else 'FAIL'} llama/{case['id']}: "
                  f"{l['text'].strip()[:90]!r}")

    if damaged_root:
        shutil.rmtree(damaged_root, ignore_errors=True)

    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)
    verdict = {
        arm: ("PASS" if all(r["pass"] for r in rs) else "FAIL")
        for arm, rs in by_arm.items()
    }

    receipt = {
        "gate": "RUNTIME_CORRECTNESS",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "mode": "damage_control" if args.damage else "live",
        "artifact": str(artifact),
        "why": "A tok/s number from an artifact whose output nobody checked "
               "measures how fast something produces text. This gate qualifies "
               "the arms that RUNTIME_EXPERIMENT.json timed.",
        "cases": CASES,
        "results": results,
        "verdict_by_arm": verdict,
        "rule": "If an arm FAILS, its tok/s is not a result. A fast incoherent "
                "artifact is a negative result, not a win.",
        "promotion": {
            "decision": "NOT_PROMOTED",
            "basis": "the paired A/B found no improvement against the historical "
                     "native anchor of 33.03 tok/s; correctness alone does not "
                     "promote anything",
            "owner": "the measuring harness does NOT own promotion; it reports "
                     "and a human decides",
        },
    }
    name = ("RUNTIME_CORRECTNESS_DAMAGED.json" if args.damage
            else "RUNTIME_CORRECTNESS.json")
    out = REPO / "receipts/headless" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nverdict: {verdict}")
    print(f"receipt: {out}")

    if args.damage:
        # Inverted: the damage control PASSES only when the gate rejected.
        ok = verdict.get("native_gravity_q4") == "FAIL"
        print("DAMAGE CONTROL " + ("ok: the gate rejected a corrupted catalog"
                                   if ok else
                                   "FAILED: a corrupted catalog still passed, "
                                   "so this gate proves nothing"))
        return 0 if ok else 1
    return 0 if all(v == "PASS" for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
