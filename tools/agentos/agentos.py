#!/usr/bin/env python3
"""AgentOS callable primitives — the G012 embryo (steer S001/S004, plan W0.T10).

The tournament + Odyssey + V4 Flash need these operations callable, not tribal
(Bible Sec 6). This wraps procedures THIS campaign already ran by hand, over the
existing SUBSTRATE + machine_state — no new ontology, no reimplementation.

  verify_coherence(text)     -> did the model answer the Paris/4/Jupiter needles?
  verify_bit_identity(a, b)  -> token ids identical? (serial-oracle compare)
  substrate_append(record)   -> typed record onto receipts/agentos/SUBSTRATE.jsonl
  base_true_tps_ok()         -> BASE_TRUE_TPS authority: REFUSE unless clean_box_ok
  base_true_tps(steps_us)    -> the ONE legal formula: sum/n INCLUDING drain (RUG001)
  park_dead_worktrees()      -> free the clean box by parking ONLY provably-dead
                                Hawking worktrees; NEVER touch foreign/active ones

stdlib only; read-only except substrate_append (append) and park (rename dead dirs).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/scammermike/Downloads/hawking")
SUBSTRATE = REPO / "receipts" / "agentos" / "SUBSTRATE.jsonl"
WORKTREES = Path.home() / ".claude-grok" / "worktrees"

# The sealed coherence needles this campaign proved (G002/G005).
NEEDLES = [("capital of france", "paris"), ("2 + 2", "4"), ("largest planet", "jupiter")]


def verify_coherence(text: str, needles=None) -> tuple[bool, str]:
    """A completion is coherent on a needle if the expected token appears. Caller
    supplies the matching (prompt-substring, expected) pairs it actually ran."""
    t = (text or "").lower()
    pairs = needles or NEEDLES
    hits = [exp for _, exp in pairs if exp.lower() in t]
    ok = len(hits) == len(pairs)
    return ok, f"{len(hits)}/{len(pairs)} needles present: {hits}"


def verify_bit_identity(ids_a, ids_b) -> tuple[bool, str]:
    ok = list(ids_a) == list(ids_b)
    if ok:
        return True, "token ids identical"
    diff = sum(1 for x, y in zip(ids_a, ids_b) if x != y) + abs(len(ids_a) - len(ids_b))
    return False, f"{diff}/{max(len(ids_a), len(ids_b))} tokens differ"


def substrate_append(record: dict) -> None:
    if "type" not in record:
        raise ValueError("substrate record needs a 'type'")
    with SUBSTRATE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def base_true_tps(steps_us) -> float:
    """The ONLY legal BASE_TRUE_TPS formula (RUG001): n / sum(step_us) INCLUDING the
    final drain step. Never steady-median, never speculative-accept (NS008)."""
    us = [u for u in steps_us if u]
    if not us:
        return 0.0
    return len(us) / (sum(us) / 1e6)


def _machine_state():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ms", REPO / "tools" / "agentos" / "machine_state.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    return ms


def base_true_tps_ok() -> tuple[bool, str]:
    """Authority gate: a BASE_TRUE_TPS measurement may seal ONLY on a clean box
    (NS007). Callers MUST check this before running a timed generate and refuse to
    seal otherwise."""
    ms = _machine_state()
    return ms.clean_box_ok()


def _owner_repo(wt: Path) -> str | None:
    """Resolve the repo owning a git worktree (git-common-dir -> repo root)."""
    p = subprocess.run(["git", "-C", str(wt), "rev-parse", "--git-common-dir"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    common = p.stdout.strip()
    if not os.path.isabs(common):
        common = os.path.join(str(wt), common)
    common = os.path.realpath(common)
    return os.path.dirname(common) if os.path.basename(common) == ".git" else None


def park_dead_worktrees(apply: bool = False) -> list[dict]:
    """Free the clean box by PARKING (renaming out) ONLY provably-dead Hawking
    worktrees — clean porcelain AND HEAD reachable from a main ref. NEVER touches
    foreign-repo worktrees or any dirty/active one (that would break another session).
    Parked dirs move to ~/.claude-grok/parked/ (recoverable), never deleted."""
    out = []
    if not WORKTREES.is_dir():
        return out
    parked = Path.home() / ".claude-grok" / "parked"
    for wt in sorted(p for p in WORKTREES.iterdir() if p.is_dir()):
        owner = _owner_repo(wt)
        rec = {"worktree": wt.name, "owner": owner, "action": "keep", "reason": ""}
        if owner is None or os.path.realpath(owner) != os.path.realpath(REPO):
            rec["action"], rec["reason"] = "skip-foreign", "not Hawking (never touch)"
            out.append(rec); continue
        status = subprocess.run(["git", "-C", str(wt), "status", "--porcelain"],
                                capture_output=True, text=True)
        if status.returncode != 0 or status.stdout.strip():
            rec["action"], rec["reason"] = "skip-dirty", "uncommitted work — never park"
            out.append(rec); continue
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        anc = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", head, "HEAD"],
                             capture_output=True, text=True)
        if anc.returncode != 0:
            rec["action"], rec["reason"] = "skip-unmerged", "HEAD not in main — would lose commits"
            out.append(rec); continue
        rec["action"], rec["reason"] = ("parked" if apply else "would-park"), "dead Hawking lane"
        if apply:
            parked.mkdir(parents=True, exist_ok=True)
            os.rename(str(wt), str(parked / wt.name))
        out.append(rec)
    return out


if __name__ == "__main__":
    ok, why = verify_coherence("The capital of France is Paris. 2 + 2 = 4. Jupiter is largest.")
    assert ok, why
    assert not verify_coherence("garble garble")[0]
    assert verify_bit_identity([1, 2, 3], [1, 2, 3])[0]
    assert not verify_bit_identity([1, 2, 3], [1, 9, 3])[0]
    assert abs(base_true_tps([5000] * 31 + [1_150_000]) - 24.6) < 3  # ~24 with drain
    plan = park_dead_worktrees(apply=False)  # dry-run only in self-check
    cb, cbwhy = base_true_tps_ok()
    print(json.dumps({
        "self_check": "ok",
        "clean_box_ok": cb, "reason": cbwhy,
        "park_plan": {a: sum(1 for r in plan if r["action"] == a)
                      for a in {r["action"] for r in plan}},
    }, indent=2))
