#!/usr/bin/env python3
"""G134: the protected GPU timing lane, enforced mechanically.

The lane has been policy since the directive was written, and G131 proved the
policy alone does not bind: that experiment ran three Qwen workers concurrently
by invoking the binary DIRECTLY, and nothing anywhere refused. An advisory lock
only protects callers who choose to call it, which makes it convention.

G131 also priced the failure it is protecting against. Three resident workers
inflate per-worker decode from 30.34 to 76.23 ms/token -- 2.51x. A candidate
timed under load looks 2.5x worse than it is, and two candidates timed under
DIFFERENT load rank arbitrarily. That is not a degraded measurement, it is a
measurement of something else.

So enforcement moves to where timing is PRODUCED rather than where the lane is
politely requested:

  A CONTENTION WITNESS IS TAKEN BEFORE AND AFTER THE TIMED REGION.

Both halves are needed. Before-only misses a worker that spawns mid-run;
after-only misses one that exits mid-run. A timing whose witness is not clean at
BOTH ends is marked VOID, and a VOID timing carries that verdict into its own
receipt -- so it cannot be cited later without the citation announcing itself.

That is the mechanical part. It does not stop anyone running the binary by hand;
it stops the resulting number from passing as evidence.

  ./tools/gpu_lane_guard.py selftest
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = "ascension_qwen38_hybrid"
LOCK = ROOT / "tools/gpu_lane_lock.sh"


def witness(pattern: str = PROC) -> dict:
    ps = subprocess.run(["ps", "-axo", "pid=,rss=,comm="], capture_output=True, text=True).stdout
    hits = [l.split(None, 2) for l in ps.splitlines() if pattern in l]
    me = str(os.getpid())
    hits = [h for h in hits if h[0] != me]
    return {"count": len(hits),
            "pids": [int(h[0]) for h in hits],
            "rss_gb": round(sum(int(h[1]) for h in hits) * 1024 / 1e9, 2),
            "clean": len(hits) == 0}


def guard(fn, pattern: str = PROC, label: str = "timing"):
    """Run fn() between two contention witnesses. Returns (result, verdict)."""
    before = witness(pattern)
    t0 = time.monotonic()
    result = fn()
    wall = time.monotonic() - t0
    after = witness(pattern)
    clean = before["clean"] and after["clean"]
    if before["clean"] and not after["clean"]:
        why = (f"a worker APPEARED during the timed region ({after['count']} resident at the "
               "end, none at the start) -- the before-witness alone would have passed this")
    elif after["clean"] and not before["clean"]:
        why = (f"a worker EXITED during the timed region ({before['count']} resident at the "
               "start, none at the end) -- the after-witness alone would have passed this")
    elif not clean:
        why = (f"{before['count']} worker(s) resident throughout, holding "
               f"{before['rss_gb']} GB -- G131 measured this inflating per-worker decode "
               "30.34 -> 76.23 ms/token, 2.51x")
    else:
        why = "no other GPU worker resident at either end of the timed region"
    return result, {"label": label, "verdict": "VALID" if clean else "VOID",
                    "witness_before": before, "witness_after": after,
                    "wall_s": round(wall, 3), "why": why}


def inspect_lock() -> dict:
    """Characterise the pre-existing lane lock by reading it, not by asserting."""
    if not LOCK.exists():
        return {"present": False}
    src = LOCK.read_text()
    # Comments are not implementation. The first cut of this matched the word
    # "flock" inside the comment "Swap for flock if lane count grows" and
    # recorded a flock lock that does not exist.
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    return {"present": True, "bytes": len(src),
            "uses_flock": bool(re.search(r"\bflock\b", code)),
            "uses_mkdir_lock": bool(re.search(r"mkdir\s+\"?\$?\{?LOCK", code, re.I)),
            "comment_mentions_flock_but_code_does_not":
                bool(re.search(r"\bflock\b", src)) and not bool(re.search(r"\bflock\b", code)),
            "callers_must_opt_in": True,
            "bypassable_by_direct_invocation": True,
            "proof_of_bypass": "G131 ran three concurrent workers by invoking the greedy "
                               "binary directly; the lock was never consulted and nothing "
                               "refused"}


def selftest(out: pathlib.Path | None):
    steps = []
    lock = inspect_lock()
    print(f"pre-existing lane lock: present={lock.get('present')} "
          f"flock={lock.get('uses_flock')} -- opt-in, bypassable by direct invocation "
          f"(G131 did exactly that)\n")

    print("STEP 1  clean machine, guard should mark the timing VALID")
    _, v1 = guard(lambda: time.sleep(1.0), label="clean-lane")
    steps.append(v1)
    print(f"  {v1['verdict']}: {v1['why']}")

    print("\nSTEP 2  one REAL resident worker up, guard must mark the timing VOID")
    g = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
    r = ROOT / "workspace/campaign/records/runs/qwen38-27b"
    proc = subprocess.Popen(
        [str(g), "--artifact-root", str(r / "uniform-q4-v1"),
         "--tokenizer", str(r / "bf16/tokenizer.json"),
         "--prompt", "Write a long detailed essay about the history of metallurgy.",
         "--max-new-tokens", "900", "--max-seq-len", "2048",
         "--out", "/tmp/g134_contender.json"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    for _ in range(60):
        time.sleep(1)
        if witness()["count"] >= 1:
            break
    _, v2 = guard(lambda: time.sleep(1.0), label="contended-lane")
    steps.append(v2)
    print(f"  {v2['verdict']}: {v2['why']}")

    print("\nSTEP 3  worker exits DURING the region, after-witness must catch it")
    def kill_midway():
        time.sleep(0.5)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
        time.sleep(1.0)
    _, v3 = guard(kill_midway, label="worker-exits-midway")
    steps.append(v3)
    print(f"  {v3['verdict']}: {v3['why']}")

    print("\nSTEP 4  clean again, guard returns to VALID")
    _, v4 = guard(lambda: time.sleep(1.0), label="clean-again")
    steps.append(v4)
    print(f"  {v4['verdict']}: {v4['why']}")

    voids = [s for s in steps if s["verdict"] == "VOID"]
    valids = [s for s in steps if s["verdict"] == "VALID"]
    ok = v1["verdict"] == "VALID" and v2["verdict"] == "VOID" and \
         v3["verdict"] == "VOID" and v4["verdict"] == "VALID"
    print(f"\n{len(voids)} VOID, {len(valids)} VALID -- guard was watched REFUSING, "
          f"not only permitting: {ok}")

    if out:
        doc = {
            "schema": "hawking.nos.gpu_lane_guard.v1",
            "obligation": "G134 -- protected GPU timing lane enforced MECHANICALLY",
            "why_the_old_lane_was_convention": (
                "an advisory lock only protects callers who choose to call it. G131 ran three "
                "concurrent workers by invoking the greedy binary directly, never consulting "
                "the lock, and nothing refused. That is the proof, and it is a measurement of "
                "this repo rather than an opinion about it."),
            "pre_existing_lock": lock,
            "what_is_being_protected": (
                "G131 measured three resident workers inflating per-worker decode from 30.34 to "
                "76.23 ms/token, 2.51x. A candidate timed under load looks 2.5x worse than it "
                "is; two candidates timed under DIFFERENT load rank arbitrarily. That is not a "
                "degraded measurement, it is a measurement of something else."),
            "mechanism": (
                "a contention witness is taken BEFORE and AFTER the timed region. Both halves "
                "are load-bearing: before-only misses a worker that spawns mid-run, after-only "
                "misses one that exits mid-run. A timing not clean at both ends is VOID, and "
                "the verdict travels in the receipt -- so the number cannot be cited later "
                "without the citation announcing itself."),
            "honest_limit": (
                "this does not stop anyone running the binary by hand. It stops the resulting "
                "number from passing as evidence. Enforcement is at the evidence layer, which "
                "is the layer that actually binds in this campaign, because a ledger entry "
                "requires a receipt."),
            "steps": steps,
            "watched_refusing": ok,
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {out}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest"])
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    sys.exit(0 if selftest(a.out) else 1)


if __name__ == "__main__":
    main()
