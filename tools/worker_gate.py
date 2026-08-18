#!/usr/bin/env python3
"""G133: memory-pressure gate consulted BEFORE spawning worker N.

G131 measured the law this gate enforces:

    wired(n) = 4.63 + 15.59n GB     on 103.08 GB total

and, more importantly, measured WHICH RESOURCE BINDS. It is not RSS and it is
not free pages. Wired memory cannot be compressed and cannot be swapped, so
exceeding it is a hard crash rather than a slowdown -- which is why the
five-worker attempt crashed instead of merely thrashing. A gate that watches
RSS (49.4 GB of 103.08 at n=3, comfortable-looking) or free pages (0.13 GB at
n=3, alarming-looking) reads the wrong number in both directions.

Two independent triggers, because a projection and a measurement fail
differently:

  PROJECTED  spawning worker N would leave less than RESERVE non-wired.
  OBSERVED   the machine is ALREADY under pressure -- swap in use or the
             compressor above its floor -- regardless of what the projection
             says. G131 recorded compressed pages flat at 0.10 GB and swap at
             0 through the whole three-worker run, so any movement in either is
             a real change of state, not noise.

The reserve is calibrated from one success and one crash: n=3 worked leaving
51.7 GB non-wired, n=5 crashed leaving 20.5 GB. The true boundary is somewhere
in between and n=4 HAS NOT BEEN TESTED. So the gate reports UNVALIDATED for any
n above the measured maximum instead of implying the projection is a
measurement.

  ./tools/worker_gate.py --sweep 6
"""
from __future__ import annotations
import argparse, json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

BASELINE_WIRED_GB = 4.63      # G131, measured with zero workers
PER_WORKER_WIRED_GB = 15.59   # G131, 40/43 samples at n=3
MEASURED_MAX_N = 3            # G131, held 235s stable
CRASHED_AT_N = 5              # directive S18, "do not repeat the five-worker crash"
RESERVE_GB = 32.0             # between the 51.7 that worked and the 20.5 that crashed
COMPRESSOR_FLOOR_GB = 0.10    # G131, flat through the entire three-worker run
PROC = "ascension_qwen38_hybrid"


def observe() -> dict:
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = int(re.search(r"page size of (\d+)", vm).group(1))
    d = {k.strip(): int(v.strip(" ."))
         for k, v in (l.split(":") for l in vm.splitlines()
                      if ":" in l and l.split(":")[1].strip(" .").isdigit())}
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True).stdout.strip())
    ps = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True).stdout
    rss = [int(l.split()[0]) * 1024 for l in ps.splitlines() if PROC in l]
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", sw)
    return {
        "total_gb": total / 1e9,
        "wired_gb": d.get("Pages wired down", 0) * pg / 1e9,
        "free_gb": d.get("Pages free", 0) * pg / 1e9,
        "inactive_gb": d.get("Pages inactive", 0) * pg / 1e9,
        "compressed_gb": d.get("Pages occupied by compressor", 0) * pg / 1e9,
        "swap_used_mb": float(used.group(1)) if used else 0.0,
        "workers_resident": len(rss),
        "worker_rss_total_gb": sum(rss) / 1e9,
    }


def gate(obs: dict, reserve: float = RESERVE_GB) -> dict:
    n = obs["workers_resident"]
    # Prefer the OBSERVED per-worker cost when workers are resident; the G131
    # calibration is a fallback, not an assumption that overrides the machine.
    if n >= 1:
        per = (obs["wired_gb"] - BASELINE_WIRED_GB) / n
        src = f"observed from {n} resident worker(s)"
    else:
        per = PER_WORKER_WIRED_GB
        src = "G131 calibration (no workers resident to observe)"

    projected = obs["wired_gb"] + per
    headroom = obs["total_gb"] - projected
    target_n = n + 1

    reasons = []
    if obs["swap_used_mb"] > 0:
        reasons.append(f"OBSERVED: swap already in use ({obs['swap_used_mb']:.0f} MB) -- "
                       "the machine is under real pressure now, projection is irrelevant")
    if obs["compressed_gb"] > COMPRESSOR_FLOOR_GB * 1.5:
        reasons.append(f"OBSERVED: compressor at {obs['compressed_gb']:.2f} GB, above its "
                       f"{COMPRESSOR_FLOOR_GB} GB floor -- memory is being reclaimed under duress")
    if headroom < reserve:
        reasons.append(f"PROJECTED: worker {target_n} would wire {projected:.1f} GB and leave "
                       f"{headroom:.1f} GB non-wired, under the {reserve:.0f} GB reserve")

    permitted = not reasons
    unvalidated = permitted and target_n > MEASURED_MAX_N
    return {
        "workers_resident": n, "spawn_target": target_n,
        "per_worker_wired_gb": round(per, 2), "per_worker_source": src,
        "current_wired_gb": round(obs["wired_gb"], 2),
        "projected_wired_gb": round(projected, 2),
        "projected_headroom_gb": round(headroom, 1), "reserve_gb": reserve,
        "decision": "PERMIT" if permitted else "REFUSE",
        "unvalidated": unvalidated,
        "note": (f"PERMITTED BUT UNVALIDATED: n={target_n} exceeds the measured maximum of "
                 f"{MEASURED_MAX_N}. The projection is arithmetic, not a measurement, and "
                 f"n={CRASHED_AT_N} is known to crash." if unvalidated else
                 "within the measured-stable range" if permitted else "; ".join(reasons)),
        "reasons": reasons,
    }


def sweep(upto: int, obs: dict) -> list[dict]:
    """Project the decision for each n from the CURRENT machine state."""
    rows = []
    for n in range(upto):
        o = dict(obs)
        o["workers_resident"] = n
        o["wired_gb"] = BASELINE_WIRED_GB + PER_WORKER_WIRED_GB * n
        rows.append(gate(o))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=6)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    obs = observe()
    print(f"MEASURED NOW: total {obs['total_gb']:.2f} GB, wired {obs['wired_gb']:.2f}, "
          f"free {obs['free_gb']:.2f}, inactive {obs['inactive_gb']:.1f}, "
          f"compressed {obs['compressed_gb']:.2f}, swap {obs['swap_used_mb']:.0f} MB, "
          f"workers {obs['workers_resident']}")
    live = gate(obs)
    print(f"LIVE DECISION for worker {live['spawn_target']}: {live['decision']} "
          f"({live['note']})\n")

    print(f"PROJECTED SWEEP (reserve {RESERVE_GB:.0f} GB non-wired)")
    print(f"  {'spawn':>5} {'wired after':>12} {'headroom':>9}  decision")
    rows = sweep(a.sweep, obs)
    for r in rows:
        tag = "  <- UNVALIDATED" if r["unvalidated"] else ""
        print(f"  {r['spawn_target']:>5} {r['projected_wired_gb']:>11.1f}G "
              f"{r['projected_headroom_gb']:>8.1f}G  {r['decision']}{tag}")

    first_refuse = next((r["spawn_target"] for r in rows if r["decision"] == "REFUSE"), None)
    print(f"\nGATE REFUSES AT n={first_refuse} "
          f"(measured stable {MEASURED_MAX_N}, known crash {CRASHED_AT_N})")

    # Watch it refuse on the OBSERVED trigger too, not only the projection.
    # This tests the gate's arithmetic, not the machine -- labelled as such.
    print("\nOBSERVED-TRIGGER TEST (injected state, tests the gate not the machine)")
    for label, inject in [("swap in use", {"swap_used_mb": 512.0}),
                          ("compressor above floor", {"compressed_gb": 3.4})]:
        o = dict(obs); o.update(inject); o["workers_resident"] = 0
        o["wired_gb"] = BASELINE_WIRED_GB
        g = gate(o)
        print(f"  {label:<24} {g['decision']}: {g['note']}")
        if g["decision"] != "REFUSE":
            print("  FAILED -- gate permitted under real pressure")
            sys.exit(1)

    if first_refuse is None:
        print("FAILED -- gate never refuses, it enforces nothing")
        sys.exit(1)

    if a.out:
        doc = {
            "schema": "hawking.nos.worker_spawn_gate.v1",
            "obligation": "G133 -- memory-pressure gate before spawning worker N",
            "calibration": {"source": "G131", "baseline_wired_gb": BASELINE_WIRED_GB,
                            "per_worker_wired_gb": PER_WORKER_WIRED_GB,
                            "measured_max_n": MEASURED_MAX_N, "crashed_at_n": CRASHED_AT_N,
                            "reserve_gb": RESERVE_GB},
            "why_it_gates_on_WIRED": (
                "wired memory cannot be compressed and cannot be swapped, so exceeding it is a "
                "hard crash rather than a slowdown -- which is why n=5 crashed instead of "
                "thrashing. A gate watching RSS reads 49.4 GB of 103.08 at n=3 and looks "
                "comfortable; a gate watching free pages reads 0.13 GB at the same moment and "
                "looks alarming. Both are the wrong number, in opposite directions."),
            "two_independent_triggers": {
                "projected": "spawning worker N would leave less than the reserve non-wired",
                "observed": "swap in use or compressor above its floor -- real pressure NOW, "
                            "which makes the projection irrelevant. G131 recorded compressed "
                            "flat at 0.10 GB and swap at 0 across the entire three-worker run, "
                            "so movement in either is a genuine state change, not noise."},
            "reserve_is_bracketed_not_derived": (
                f"n=3 worked leaving 51.7 GB non-wired; n=5 crashed leaving 20.5 GB. The true "
                f"boundary lies between and N=4 HAS NOT BEEN TESTED. The reserve of "
                f"{RESERVE_GB:.0f} GB sits inside that bracket, and the gate reports UNVALIDATED "
                "for any n above the measured maximum rather than implying arithmetic is "
                "measurement."),
            "live": live, "sweep": rows, "first_refusal_at_n": first_refuse,
            "observed_trigger_refuses": True,
            "measured_now": {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in obs.items()},
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip(),
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
