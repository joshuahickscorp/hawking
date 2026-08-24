#!/usr/bin/env python3
"""Run a real mixed campaign through HCLI's own scheduler and measure it.

Every WorkUnit here is work that would be worth doing anyway: each one runs a
real check over this repository and is accepted only on that check's own exit
code. Nothing sleeps, nothing greps for a nonce the harness itself wrote, and
no unit is generated to occupy idle capacity. That is the whole point --
`MAX_NO_ARTIFICIAL_WORK` is an obligation, and a qualification built from
synthetic units qualifies nothing. The previous "live mixed max" receipt on this
box was exactly that failure: four units whose verifiers could not fail,
extrapolated from 12.4 seconds into 1164 units/hour.

    python3 tools/headless/hcli_mixed_max_campaign.py --rungs 1,2,4
    python3 tools/headless/hcli_mixed_max_campaign.py --dry-run   # list the work

Per rung it records requested / admitted / actually-active concurrency, verified
and rejected units, failures, retries, queue delay, p50/p95 completion latency,
verifier wait, mutation wait, scheduler overhead and verified units per hour.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

from hcli.mission import Mission  # noqa: E402
from hcli.workunit import WorkUnit  # noqa: E402

PY = sys.executable


def _cpu_unit(uid: str, description: str, command: str, deps: Optional[List[str]] = None) -> WorkUnit:
    """A unit whose acceptance IS the command's exit code.

    The command is anchored to the repository root. A CPU verifier runs with the
    MISSION workspace as its cwd, and these checks import `hcli`,
    so an absolute path alone is not enough -- without the anchor every unit
    fails with ModuleNotFoundError for a reason that has nothing to do with what
    it is checking. Found the hard way while building the self-supplement chain.
    """
    command = f"cd {REPO_ROOT} && {command}"
    return WorkUnit(
        id=uid,
        role="validate",
        description=description,
        dependencies=list(deps or []),
        preferred_backend="cpu",
        resource_class="TEST",
        verifier=command,
    )


def real_work() -> List[WorkUnit]:
    """The campaign's actual work: real checks over this repository.

    Each is a regression gate that this session's changes could plausibly break,
    so running them is useful independently of measuring throughput.
    """
    headless = REPO_ROOT / "tools" / "headless"
    units: List[WorkUnit] = []

    # Headless integration checks. Real scripts, real exit codes.
    for name in (
        "hcli_runtimepool_test.py",
        "hcli_scheduler_test.py",
        "hcli_dag_consolidation_test.py",
        "hcli_command_ingress_test.py",
        "hcli_verifier_pipeline_test.py",
        "hcli_validation_authority_test.py",
        "rollback_integrity_test.py",
        "hcli_callpath_test.py",
        "hcli_containment_test.py",
        "hcli_mission_test.py",
        "hcli_agentos_ledger_test.py",
        "hcli_max_isolation_test.py",
    ):
        if (headless / name).is_file():
            units.append(
                _cpu_unit(
                    f"check.{name.replace('.py','')}",
                    f"headless regression: {name}",
                    f'{PY} {headless / name}',
                )
            )

    # Focused unit-test slices. Split so several can run concurrently rather
    # than one long serial pytest.
    for slice_name, expr in (
        ("ledger", "ledger"),
        ("context", "context or budget"),
        ("grok", "grok"),
        ("verifier", "verif or mutation"),
        ("scheduler", "scheduler or workunit or dag"),
        ("report", "report or compact"),
    ):
        units.append(
            _cpu_unit(
                f"pytest.{slice_name}",
                f"unit tests matching {expr!r}",
                f'{PY} -m pytest {REPO_ROOT / "tools/haider/hcli/tests"} -q -k "{expr}"',
            )
        )

    # Property checks over this session's own claims. Each fails loudly if the
    # corresponding repair regressed.
    units.append(
        _cpu_unit(
            "prop.p0gates",
            "the environment-independent P0 gates are all green",
            # P0-7 compares reported Grok activity against the real process
            # table, so it is INCONCLUSIVE (exit 1) whenever no lane happens to
            # be live. That is right for the gate suite and wrong here: a
            # throughput unit must not fail because of what else is running.
            f'{PY} {headless / "hcli_p0_gates.py"} '
            "--gate P0-1 --gate P0-2 --gate P0-3 --gate P0-4 "
            "--gate P0-6 --gate P0-8 --gate P0-11 --gate P0-12",
        )
    )
    units.append(
        _cpu_unit(
            "prop.context_authority",
            "the context authority still reproduces the 11008 regression arithmetic",
            f'{PY} -c "import sys;'
            'from hcli import context_budget as cb;'
            'assert cb.per_seq_context(32768,3)==11008;'
            'assert cb.solve_parallel(32768,11008)==3;'
            'print(\'ok\')"',
        )
    )
    units.append(
        _cpu_unit(
            "prop.gguf_ceiling",
            "the GGUF parser still reads the model's real context ceiling",
            f'{PY} -c "import sys;'
            'from hcli import context_budget as cb;'
            'v=cb.gguf_context_length(r\'/Users/scammermike/models/qwen3.8-27b-abliterated/'
            'Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf\');'
            'assert v==262144, v; print(\'ok\', v)"',
        )
    )
    # One genuine dependency chain, so ordering is exercised by real work.
    units.append(
        _cpu_unit(
            "chain.compile",
            "imports resolve across the whole hcli package",
            f'{PY} -c "import sys;'
            'import hcli.engine,hcli.mission,hcli.scheduler,hcli.controller,hcli.executors,'
            'hcli.grok_bridge,hcli.ledger,hcli.context_budget;print(\'ok\')"',
        )
    )
    units.append(
        _cpu_unit(
            "chain.suite",
            "full unit suite, gated on imports resolving first",
            f'{PY} -m pytest {REPO_ROOT / "tools/haider/hcli/tests"} -q',
            deps=["chain.compile"],
        )
    )
    return units


class NullEngine:
    """No cognition in this campaign: every unit is CPU-class real work.

    Present because Mission requires an engine. If it is ever asked to run a
    unit that is a bug in unit construction, not a fallback -- so it says so.
    """

    active = False

    def execute_workunit(self, wu: Any, context: Any) -> Dict[str, Any]:
        return {
            "kind": "answer",
            "validation": {
                "ok": False,
                "reason": "NO_COGNITION_BACKEND_IN_THIS_CAMPAIGN",
                "unit": getattr(wu, "id", "?"),
            },
        }


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[idx], 3)


def _limits_for(rung: int):
    """A ResourceLimits whose TEST capacity IS the rung.

    Without this the rung is decoration: `runtime_count` gates GPU_DECODE, while
    TEST class is capped at ncpu, so 22 units ran simultaneously at rung 1, 2
    and 4 alike and the throughput curve was flat by construction.
    """
    from hcli.resources import ResourceLimits

    base = ResourceLimits.resolve(str(REPO_ROOT))
    base.test = max(1, int(rung))
    base.test_authoring = max(1, int(rung))
    return base


def run_rung(rung: int, timeout_s: float) -> Dict[str, Any]:
    units = {u.id: u for u in real_work()}
    engine = NullEngine()
    active_samples: List[int] = []
    stop = {"flag": False}

    with tempfile.TemporaryDirectory(prefix=f"maxcampaign-c{rung}-") as ws:
        mission = Mission(
            ws,
            engine=engine,
            units=units,
            goal="verify the hardening frontier",
            runtime_count=rung,
            quiet=True,
            no_progress_threshold=60,
            limits=_limits_for(rung),
        )

        import threading

        def sampler() -> None:
            while not stop["flag"]:
                running = sum(
                    1 for u in mission.scheduler.units.values() if u.status == "running"
                )
                active_samples.append(running)
                time.sleep(0.05)

        t = threading.Thread(target=sampler, daemon=True)
        t.start()

        deadline = threading.Event()

        def watchdog() -> None:
            if not deadline.wait(timeout_s):
                try:
                    mission.cancel("campaign rung deadline")
                except Exception:
                    pass

        w = threading.Thread(target=watchdog, daemon=True)
        w.start()

        t0 = time.perf_counter()
        try:
            mission.run()
        finally:
            deadline.set()
            stop["flag"] = True
            t.join(timeout=1)
        wall = time.perf_counter() - t0

        final = list(mission.scheduler.units.values())
        last_dispatch = dict(getattr(mission.scheduler, "last_dispatch", {}) or {})

    verified = [u for u in final if u.status == "completed"]
    failed = [u for u in final if u.status == "failed"]
    repairs = [u for u in final if ".repair." in u.id]
    retries = sum(int(getattr(u, "attempts", 0) or 0) for u in final)

    queue_delays = [
        u.running_at - u.ready_at
        for u in final
        if getattr(u, "ready_at", None) and getattr(u, "running_at", None)
    ]
    completions = [
        u.finished_at - u.running_at
        for u in final
        if getattr(u, "running_at", None) and getattr(u, "finished_at", None)
    ]

    return {
        "requested_concurrency": rung,
        "admitted_concurrency": last_dispatch.get("admitted"),
        "peak_actually_active": max(active_samples) if active_samples else 0,
        "mean_actually_active": round(statistics.mean(active_samples), 3) if active_samples else 0,
        "units_total": len(final),
        "units_original": len(units),
        "verified": len(verified),
        "rejected_or_failed": len(failed),
        "repair_units_created": len(repairs),
        "retries": retries,
        "wall_s": round(wall, 3),
        "queue_delay_p50_s": _pct(queue_delays, 0.5),
        "queue_delay_p95_s": _pct(queue_delays, 0.95),
        "completion_p50_s": _pct(completions, 0.5),
        "completion_p95_s": _pct(completions, 0.95),
        "scheduler_overhead_s": last_dispatch.get("overhead_s"),
        "mutation_blocked_at_last_dispatch": last_dispatch.get("mutation_blocked"),
        "verified_units_per_hour": round(len(verified) / wall * 3600, 2) if wall > 0 else None,
        "failed_ids": sorted(u.id for u in failed)[:12],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rungs", default="1,2,4")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="receipts/headless/HCLI_MIXED_MAX.json")
    args = ap.parse_args(argv)

    work = real_work()
    if args.dry_run:
        for u in work:
            print(f"{u.id:<38} {u.resource_class:<6} {u.description}")
            print(f"    verifier: {u.verifier[:150]}")
        print(f"\n{len(work)} real WorkUnits")
        return 0

    # The slowest unit here is the full unit suite at ~125s; the executor's
    # default CPU timeout is 120s, so it was being killed just short of the
    # finish line and then retried into the same wall.
    os.environ.setdefault("HCLI_CPU_TIMEOUT", "600")

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    results = []
    for r in rungs:
        print(f"--- rung c={r} ---", flush=True)
        rec = run_rung(r, args.timeout)
        results.append(rec)
        print(
            f"c={r} verified={rec['verified']}/{rec['units_original']} "
            f"peak_active={rec['peak_actually_active']} wall={rec['wall_s']}s "
            f"WU/h={rec['verified_units_per_hour']}",
            flush=True,
        )

    base = results[0]["verified_units_per_hour"] if results else None
    for rec in results:
        if base and rec["verified_units_per_hour"]:
            rec["throughput_vs_rung1"] = round(rec["verified_units_per_hour"] / base, 4)

    # The useful equilibrium is the KNEE, not the argmax. Taking the highest
    # number rewards a rung that bought 1.3% for twice the concurrency, which
    # is the "largest integer reached" the directive explicitly warns against.
    # A rung only counts as an improvement if it beats its predecessor by more
    # than MIN_GAIN_PCT.
    MIN_GAIN_PCT = 5.0
    best = None
    for rec in results:
        rate = rec.get("verified_units_per_hour")
        if not rate:
            continue
        if best is None:
            best = rec
            continue
        prev = best["verified_units_per_hour"]
        gain = (rate - prev) / prev * 100.0
        rec["gain_over_previous_pct"] = round(gain, 2)
        if gain > MIN_GAIN_PCT:
            best = rec
        else:
            rec["rejected_as_equilibrium"] = (
                f"only {gain:.1f}% over c={best['requested_concurrency']}, "
                f"below the {MIN_GAIN_PCT}% bar"
            )
    receipt = {
        "gates": ["HCLI_GROK_MAX_EQUILIBRIUM", "MAX_NO_ARTIFICIAL_WORK", "HCLI_MIXED_MAX"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "work_is_real": {
            "unit_count": len(work),
            "principle": "every unit runs a real check over this repository and is accepted only on "
            "that check's own exit code; none sleeps, none greps for a nonce the harness wrote, and "
            "no unit exists to occupy capacity",
            "units": [
                {"id": u.id, "class": u.resource_class, "description": u.description,
                 "verifier": u.verifier}
                for u in work
            ],
        },
        "rungs": results,
        "useful_equilibrium": best["requested_concurrency"] if best else None,
        "equilibrium_rule": "the highest rung that beat its predecessor by more than 5% in VERIFIED "
        "units per hour -- the knee, not the argmax, and explicitly not the largest integer reached",
        "min_gain_pct": 5.0,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"useful_equilibrium": receipt["useful_equilibrium"],
                      "rungs": [(r["requested_concurrency"], r["verified"],
                                 r["verified_units_per_hour"]) for r in results]}, indent=1))
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
