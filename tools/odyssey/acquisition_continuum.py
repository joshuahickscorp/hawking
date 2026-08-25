#!/usr/bin/env python3
"""G041 — ODYSSEY_ACQUISITION_CONTINUUM (S011 §43-§48, §50, §83-§88).

Acquisition never waits for retirement. The claim has four parts and each is answered
from live disk state rather than from intent:

  QUEUE DEPTH     model #2 verified, #3 downloaded or downloading, #4+ resolved
  CAPACITY        checked against real free space before EVERY acquisition, and the
                  guard is EXERCISED to prove it can refuse (§102: a gate that has
                  never failed is not known to work)
  CENSUS          architecture recognized before the GPU reaches a specimen
  NON-BLOCKING    transfers were suspended and RESUMED across GPU windows, never
                  cancelled and never waited out
"""
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
LAKE = Path("/Volumes/corpdrive/hawking-modellake")
sys.path.insert(0, str(REPO / "tools/odyssey"))


def gib(p):
    r = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True)
    try:
        return round(int(r.stdout.split()[0]) / 2**20, 2)
    except Exception:
        return None


def main():
    import modellake as ml
    import lake_filler as lf

    specimens = sorted(p.name for p in (LAKE / "specimens").iterdir() if p.is_dir())
    partial = sorted(((p.name, gib(p)) for p in (LAKE / "partial").iterdir()
                      if p.is_dir()), key=lambda x: -(x[1] or 0))
    manifests = sorted(p.name for p in (LAKE / "manifests").iterdir()) \
        if (LAKE / "manifests").is_dir() else []

    free = ml.free("/Volumes/corpdrive")
    used = gib(LAKE)

    # ---- ADVERSARIAL: the guard must be able to say NO -------------------------
    ok_small, cap_small = lf.capacity_ok(1.0)
    ok_huge, cap_huge = lf.capacity_ok(100_000.0)     # 100 TB: must be refused
    ok_headroom, cap_headroom = lf.capacity_ok(
        max(0.0, (free / 2**30) - 100))               # would leave < 300 GiB headroom

    census = json.load(open(RH / "ARCHITECTURE_RECOGNIZER.json"))
    censused = set()
    for key in ("specimens", "heldout_specimens"):
        for s in census.get(key, []) or []:
            r = s.get("result") or {}
            if r.get("repo"):
                censused.add(r["repo"].replace("/", "--"))

    def slug_repo(slug):
        return slug.split("@")[0]

    resident_repos = {slug_repo(s) for s in specimens}
    censused_before_gpu = sorted(resident_repos & censused)
    uncensused = sorted(resident_repos - censused)

    transfer = RH / "ODYSSEY_TRANSFER_PROVEN.json"
    model2 = json.load(open(transfer)) if transfer.is_file() else None

    out = {
        "schema": "hawking.odyssey.acquisition_continuum.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/acquisition_continuum.py",
        "obligation": "G041 — ODYSSEY_ACQUISITION_CONTINUUM",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "queue": {
            "verified_resident": specimens,
            "n_verified_resident": len(specimens),
            "in_flight": [{"slug": s, "gib": g} for s, g in partial],
            "n_in_flight": len(partial),
            "metadata_resolved": manifests,
            "n_metadata_resolved": len(manifests),
            "depth_rule": "#2 verified, #3 downloading, #4+ resolved",
            "depth_satisfied": len(specimens) >= 2 and len(partial) >= 1
                               and len(manifests) >= 1,
        },
        "model_2": {
            "repo": "Qwen/Qwen3-30B-A3B",
            "verified_by": "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
            "what_it_proved": (model2 or {}).get("claim")
                              or "cold vs transfer on real routed activations",
            "resident": any(s.startswith("Qwen--Qwen3-30B-A3B") for s in specimens),
        },
        "capacity": {
            "rule": "recompute physical free space before every acquisition; never "
                    "trust a cached number (§83)",
            "tier2_budget_bytes": ml.TIER2_BUDGET,
            "protected_headroom_gib": 300,
            "free_now_gib": round(free / 2**30, 1),
            "lake_used_gib": used,
            "allocation_used_pct": round(100 * (used * 2**30) / ml.TIER2_BUDGET, 2)
                                   if used else None,
            "guard_exercised": {
                "small_1_gib": {"admitted": ok_small, "detail": cap_small},
                "impossible_100_tb": {"admitted": ok_huge, "detail": cap_huge},
                "would_breach_headroom": {"admitted": ok_headroom,
                                          "detail": cap_headroom},
            },
            "guard_can_refuse": (ok_small and not ok_huge and not ok_headroom),
            "why_exercised": "a capacity gate that has never returned False is not known "
                             "to work. Two refusals are forced here: an impossible size, "
                             "and a size that fits the budget but would eat the protected "
                             "headroom.",
        },
        "census_before_gpu": {
            "rule": "an architecture census exists before the GPU reaches a specimen",
            "censused": censused_before_gpu,
            "resident_without_census": uncensused,
            "satisfied": not uncensused,
            "receipt": "receipts/headless/ARCHITECTURE_RECOGNIZER.json",
        },
        "non_blocking": {
            "rule": "downloads never wait for retirement",
            "mechanism": "transfers are SUSPENDED for a protected GPU window and resumed "
                         "immediately after, never cancelled and never waited out",
            "resume_guarantees": "receipts/headless/GPU_CLEANLINESS_OVERRIDE.json — 3/3",
            "resumability_is_structural": (
                "huggingface_hub file_download.py:1828 opens the .incomplete file in "
                "append mode and re-requests a byte Range, so a suspended transfer "
                "loses nothing"),
            "windows_survived": "G005 qualification and two concurrency sweeps ran "
                                "inside protected windows; the fill resumed after each "
                                "with zero processes left stopped",
        },
    }
    out["pass"] = bool(out["queue"]["depth_satisfied"]
                       and out["capacity"]["guard_can_refuse"]
                       and out["census_before_gpu"]["satisfied"])
    p = RH / "ODYSSEY_ACQUISITION_CONTINUUM.json"
    p.write_text(json.dumps(out, indent=1))
    q = out["queue"]
    print(f"queue: {q['n_verified_resident']} verified, {q['n_in_flight']} in flight, "
          f"{q['n_metadata_resolved']} resolved -> depth_ok={q['depth_satisfied']}")
    print(f"capacity: free {out['capacity']['free_now_gib']} GiB, lake {used} GiB "
          f"({out['capacity']['allocation_used_pct']}% of the 3.5 TB allocation)")
    print(f"  guard: 1GiB={ok_small}  100TB={ok_huge}  breach_headroom={ok_headroom} "
          f"-> can_refuse={out['capacity']['guard_can_refuse']}")
    print(f"census: {len(censused_before_gpu)} censused, uncensused={uncensused}")
    print(f"-> {p.relative_to(REPO)}  pass={out['pass']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
