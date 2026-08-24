#!/usr/bin/env python3
"""Protected test for storage reclamation (directive §8, §23).

The obligation is narrow and load-bearing: a protected artifact must be
IMPOSSIBLE to select for deletion, and the guard must still allow reclaiming
something genuinely reproducible — a policy that refuses everything is as useless
as one that refuses nothing.

This exists because a KEEP_LIST written in prose already failed once. The
2026-08-18 G28 receipt named the 51 GB bf16 SOURCE patient and the champion
3.3448 BPW artifacts as keep-forever, and they were reclaimed anyway.

No GPU, no model, no network. Never deletes anything.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("storagemgr", HERE / "storage_manager.py")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


def row(**over):
    r = {
        "path": "/Users/x/.cache/huggingface/hub/models--org--repo/blobs/abc",
        "size_gib": 10.0,
        "classification": "REDOWNLOADABLE",
        "hf_repo_id": "org/repo",
        "atime": "2026-01-01T00:00:00Z",
        "mtime": "2026-01-01T00:00:00Z",
    }
    r.update(over)
    return r


def test_every_protected_class_is_refused():
    for cls in sorted(sm.PROTECTED_CLASSES):
        r = row(classification=cls, path=f"/models/{cls.lower()}/weights.gguf")
        prot, why = sm.is_protected(r)
        check(f"{cls} is refused", prot is True, why)
        try:
            sm.assert_deletable(r)
            check(f"{cls} raises on assert_deletable", False, "no exception raised")
        except sm.ProtectedArtifact:
            check(f"{cls} raises on assert_deletable", True)


def test_the_active_parent_cannot_be_selected_however_large():
    """The headline case: the current qualified Qwen parent is huge and untouched,
    which makes it the most attractive candidate by bytes. It must still be
    impossible to pick."""
    parent = row(path="/Users/x/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf",
                 size_gib=18.19, classification="KEEP_ACTIVE_PARENT",
                 hf_repo_id=None, atime="2020-01-01T00:00:00Z")
    junk = row(size_gib=1.0)
    sel = sm.select([parent, junk], need_gib=500.0)   # desperate for space
    paths = [c["path"] for c in sel["selected"]]
    check("the active parent is not selected even when 500 GiB is demanded",
          parent["path"] not in paths, str(paths))
    check("the refusal is recorded with a reason",
          any(r["path"] == parent["path"] and r["reason"] for r in sel["refused"]),
          str(sel["refused"])[:300])
    check("the selection honestly reports it could not satisfy the need",
          sel["satisfied"] is False, str(sel["satisfied"]))


def test_receipts_and_negative_science_are_never_candidates():
    for p in ("/repo/receipts/headless/GPU_ATTACK.json",
              "/repo/.haider/bootstrap-director-v6/negative-science.jsonl",
              "/repo/receipts/headless/MACHINE_GENOME.json"):
        r = row(path=p, classification="REDOWNLOADABLE", size_gib=50.0)
        prot, why = sm.is_protected(r)
        check(f"evidence path refused: {os.path.basename(p)}", prot is True, why)


def test_redownloadable_without_a_route_is_refused():
    r = row(classification="REDOWNLOADABLE", hf_repo_id=None)
    prot, why = sm.is_protected(r)
    check("REDOWNLOADABLE with no repo id is refused (unproven route)", prot is True, why)


def test_the_policy_can_still_say_yes():
    """A policy that refuses everything reclaims nothing and is not a policy."""
    cands = [row(path=f"/Users/x/.cache/huggingface/hub/models--org--r{i}/blobs/b{i}",
                 size_gib=5.0, atime="2020-01-01T00:00:00Z") for i in range(4)]
    sel = sm.select(cands, need_gib=12.0)
    check("reclaims genuinely reproducible artifacts", len(sel["selected"]) >= 3, str(sel["selected_gib"]))
    check("satisfies a satisfiable need", sel["satisfied"] is True, str(sel))
    for c in sel["selected"]:
        sm.assert_deletable(next(r for r in cands if r["path"] == c["path"]))
    check("everything selected passes assert_deletable", True)


def test_it_takes_only_what_is_needed():
    cands = [row(path=f"/Users/x/.cache/huggingface/hub/models--org--r{i}/blobs/b{i}",
                 size_gib=5.0, atime="2020-01-01T00:00:00Z") for i in range(10)]
    sel = sm.select(cands, need_gib=7.0)
    check("does not over-reclaim (takes ~need, not everything)",
          sel["selected_gib"] <= 12.0 and sel["satisfied"], str(sel["selected_gib"]))


def test_recently_used_ranks_below_cold():
    cold = row(path="/Users/x/.cache/huggingface/hub/models--org--cold/blobs/a",
               atime="2020-01-01T00:00:00Z")
    import time as _t
    hot = row(path="/Users/x/.cache/huggingface/hub/models--org--hot/blobs/b",
              atime=_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()))
    check("a cold artifact outranks an identically sized hot one",
          sm.score(cold)["score"] > sm.score(hot)["score"],
          f"cold={sm.score(cold)} hot={sm.score(hot)}")


def test_real_ledger_protects_the_real_parent():
    """Against the actual ledger on disk, not a fixture."""
    led = HERE.parent.parent / "receipts/headless/ARTIFACT_LEDGER.json"
    if not led.is_file():
        print("SKIP real-ledger check: no ARTIFACT_LEDGER.json (run artifact_census.py)")
        return
    rows = json.loads(led.read_text())["artifacts"]
    sel = sm.select(rows, need_gib=10_000.0)   # absurd demand
    picked = {c["path"] for c in sel["selected"]}
    protected = [r for r in rows if r.get("classification") in sm.PROTECTED_CLASSES]
    leaked = [p["path"] for p in protected if p["path"] in picked]
    check(f"no protected artifact selected from the real ledger "
          f"({len(protected)} protected, {len(picked)} picked)",
          not leaked, str(leaked[:5]))
    q = [r for r in rows if "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf" in r["path"]]
    if q:
        check("the live production GGUF specifically is not selected",
              q[0]["path"] not in picked, q[0]["path"])


def main() -> int:
    for fn in sorted([f for n, f in globals().items() if n.startswith("test_")],
                     key=lambda f: f.__code__.co_firstlineno):
        fn()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for f in FAILS:
            print("  " + f)
        return 1
    print("\nall storage reclamation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
