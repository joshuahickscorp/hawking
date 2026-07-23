#!/usr/bin/env python3.12
"""Detached heavy lane for the functional-gravity campaign: capture the teacher evidence
the functional student gauntlet needs, one layer-split at a time, restart-safe.

The gauntlet asks for early, middle, late and final sparse layers, a propagation partner
for at least two of them, and replication splits that were never fitted on.  Only layer 38
has that evidence today.  This walks the rest of the matrix.

Restart-safe by construction: the unit of work is one (layer, split) capsule, a capsule
that already exists on disk is skipped rather than recomputed, and the lease file records
the pid so a second controller refuses to start.  Killing this process loses at most the
capsule in flight.

    run          walk the plan, capturing what is missing
    plan         print the plan and what is already done
    status       print the lease, heartbeat and progress
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

SUPPORT = Path(os.environ.get(
    "GLM52_SUPPORT_ROOT",
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity"))
CAPSULES = SUPPORT / "source_fetch" / "teacher" / "capsules_generation_b"
CONTROL = SUPPORT / "control" / "functional"
LEASE = CONTROL / "controller.lease.json"
HEARTBEAT = CONTROL / "controller.heartbeat.json"
PROGRESS = CONTROL / "controller.progress.jsonl"

# Layers that are fully resident and sparse (first_k_dense_replace = 3, so 0-2 are dense
# and 78 is the MTP layer).  The strata are the gauntlet's, not a convenience sample:
# 5 is early, 38 middle, 74 late, 77 final; 39 and 75 exist to carry a perturbation one
# complete layer forward.
STRATA = {5: "early", 38: "middle", 74: "late", 77: "final"}
PROPAGATION = {39: 38, 75: 74}
EXTRA = {11: "early", 41: "middle", 76: "late", 3: "early_boundary", 40: "middle"}

# The fit needs three train splits and one disjoint score split; the replication splits are
# never fitted on and exist to answer "does it hold on documents it has not seen".
FIT_SPLITS = ("teacher_fit", "teacher_router", "teacher_doctor")
SCORE_SPLIT = "teacher_score"
REPLICATION_SPLITS = ("teacher_cv", "teacher_holdout", "teacher_replication",
                      "teacher_protected", "teacher_longctx")

# Free-disk floor.  A capsule is about 1 GiB and the policy would rather stop the lane than
# fill the volume the source stream is also using.
MIN_FREE_BYTES = 60 * 1024 ** 3


def capsule_path(layer: int, split: str) -> Path:
    name = f"L{layer:02d}_L{layer:02d}.npz"
    return CAPSULES / name if split == "teacher_fit" else CAPSULES / split / name


def plan() -> list[tuple[int, str]]:
    """Ordered so the earliest work unblocks the most gates.

    Wave 1 is the four strata plus the two propagation partners: it is what FS1, FS2 and
    FS3 need.  Wave 2 is replication on the layer that already has a fitted student.  Wave
    3 widens the strata once the decisive results exist.
    """
    work: list[tuple[int, str]] = []
    for layer in (39, 5, 77, 74, 75):
        splits = FIT_SPLITS + (SCORE_SPLIT,) if layer in STRATA or layer == 39 \
            else ("teacher_fit", SCORE_SPLIT)
        work += [(layer, split) for split in splits]
    work += [(38, split) for split in REPLICATION_SPLITS]
    work += [(layer, split) for layer in (11, 41, 76, 3, 40)
             for split in ("teacher_fit", SCORE_SPLIT)]
    work += [(layer, split) for layer in (5, 74, 77) for split in REPLICATION_SPLITS[:3]]
    return work


def remaining() -> list[tuple[int, str]]:
    return [item for item in plan() if not capsule_path(*item).exists()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def acquire() -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    if LEASE.exists():
        held = json.loads(LEASE.read_text())
        if held.get("pid") and _alive(int(held["pid"])) and int(held["pid"]) != os.getpid():
            raise SystemExit(f"controller already running: pid {held['pid']}")
    LEASE.write_text(json.dumps({
        "pid": os.getpid(),
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "glm52_functional_teacher_capture",
    }, indent=2))


def beat(**fields) -> None:
    HEARTBEAT.write_text(json.dumps({
        "pid": os.getpid(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "free_bytes": free_bytes(),
        **fields,
    }, indent=2))


def free_bytes() -> int:
    stat = os.statvfs(str(SUPPORT))
    return stat.f_bavail * stat.f_frsize


def record(**fields) -> None:
    with PROGRESS.open("a") as handle:
        handle.write(json.dumps(
            {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields},
            sort_keys=True) + "\n")


def run() -> int:
    acquire()
    import glm52_teacher_capture as capture  # deferred: importing costs seconds

    graph = capture._graph()
    schedule = capture._schedule()
    config = capture.official_config()

    stopping = {"now": False}

    def stop(signum, frame):  # noqa: ARG001
        stopping["now"] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    work = remaining()
    record(event="start", pending=len(work), plan=len(plan()))
    for index, (layer, split) in enumerate(work):
        if stopping["now"]:
            record(event="stopped", at_index=index)
            break
        if free_bytes() < MIN_FREE_BYTES:
            record(event="halt_disk", free_bytes=free_bytes())
            break
        target = capsule_path(layer, split)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        beat(state="capturing", layer=layer, split=split,
             done=index, pending=len(work) - index)
        started = time.time()
        try:
            receipt = capture.capture_layers(
                [layer], split=split, graph=graph, schedule=schedule, config=config,
                capsule_dir=target.parent)
        except Exception as error:  # noqa: BLE001 - a refusal must not kill the lane
            record(event="failed", layer=layer, split=split, error=repr(error)[:400])
            continue
        record(event="captured", layer=layer, split=split,
               seconds=round(time.time() - started, 1),
               bytes=target.stat().st_size if target.exists() else 0,
               capsule=receipt.get("capsule_id"))
    beat(state="idle", pending=len(remaining()))
    record(event="finished", pending=len(remaining()))
    return 0


# Phase 2.  Capture is finished, so the heavy lane's job becomes the science that the
# capsules unblocked: is the residual stream expansive everywhere, or only where the first
# probe happened to look.  Each job is (output artifact, argv); an existing artifact is
# skipped, which is what makes the lane restart-safe.
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "condense" / "glm52_generation_b"
PYTHON = str(ROOT / ".venv" / "glm52" / "bin" / "python")
GAUNTLET = str(Path(__file__).resolve().parent / "glm52_functional_gauntlet.py")
CASCADE = str(Path(__file__).resolve().parent / "glm52_functional_cascade.py")

# Each tool names its own artifact by layer, so a job is (artifact, argv) and nothing is
# renamed.  The first version of this lane renamed a fixed-name output into place, which
# let a probe at one stratum silently consume the artifact of another; the layer suffix in
# the tool is what makes that impossible rather than merely unlikely.
JOBS = [
    # The early stratum probes from layer 3, not 5: a probe needs its successors resident,
    # and layers 6 and 7 are only partially fetched while 4 and 5 are complete.  Layer 3 is
    # the first sparse layer, so this is the earliest probe the source permits.
    ("GLM52_FUNCTIONAL_DEPTH_THRESHOLD_L03.json", [PYTHON, CASCADE, "threshold", "3", "2"]),
    ("GLM52_FUNCTIONAL_DEPTH_PERTURBATION_L03.json",
     [PYTHON, CASCADE, "perturbation", "3", "2"]),
    ("GLM52_FUNCTIONAL_DEPTH_THRESHOLD_L38.json", [PYTHON, CASCADE, "threshold", "38", "2"]),
    ("GLM52_FUNCTIONAL_DEPTH_PERTURBATION_L38.json",
     [PYTHON, CASCADE, "perturbation", "38", "3"]),
    ("GLM52_FUNCTIONAL_DEPTH_CASCADE_L38.json", [PYTHON, CASCADE, "cascade", "38", "3"]),
    ("GLM52_FUNCTIONAL_DEPTH_THRESHOLD_L74.json", [PYTHON, CASCADE, "threshold", "74", "2"]),
    ("GLM52_FUNCTIONAL_DEPTH_PERTURBATION_L74.json",
     [PYTHON, CASCADE, "perturbation", "74", "3"]),
    ("GLM52_FUNCTIONAL_DEPTH_CASCADE_L74.json", [PYTHON, CASCADE, "cascade", "74", "3"]),
]


def analysis() -> int:
    """Walk the phase-2 job list.  An artifact that already exists is skipped."""
    import subprocess
    acquire()
    record(event="analysis_start", jobs=len(JOBS))
    for target, argv in JOBS:
        if (REPORTS / target).exists():
            continue
        beat(state="analysing", target=target)
        started = time.time()
        finished = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
        if finished.returncode != 0:
            record(event="analysis_failed", target=target,
                   stderr=finished.stderr[-500:])
            continue
        record(event="analysed", target=target,
               seconds=round(time.time() - started, 1))
    beat(state="idle", pending=0)
    record(event="analysis_finished",
           done=sum((REPORTS / target).exists() for target, _ in JOBS))
    return 0


def status() -> dict:
    done = [item for item in plan() if capsule_path(*item).exists()]
    return {
        "phase_2_jobs": len(JOBS),
        "phase_2_done": sum((REPORTS / target).exists() for target, _ in JOBS),
        "lease": json.loads(LEASE.read_text()) if LEASE.exists() else None,
        "lease_alive": bool(LEASE.exists()
                            and _alive(int(json.loads(LEASE.read_text())["pid"]))),
        "heartbeat": json.loads(HEARTBEAT.read_text()) if HEARTBEAT.exists() else None,
        "planned": len(plan()),
        "captured": len(done),
        "pending": len(remaining()),
        "free_gib": round(free_bytes() / 1024 ** 3, 1),
    }


def selftest() -> int:
    # The plan must not ask for a capsule twice, and every entry must be a real split.
    items = plan()
    assert len(items) == len(set(items)), "plan repeats a (layer, split)"
    import glm52_capture_program as program
    for layer, split in items:
        assert split in program.SPLIT_PARTITIONS, split
        assert 3 <= layer <= 77, layer
    # The four strata must each be fittable and scorable.
    for layer in STRATA:
        for split in FIT_SPLITS + (SCORE_SPLIT,):
            assert (layer, split) in items or capsule_path(layer, split).exists(), \
                (layer, split)
    # Propagation partners must be the layer immediately after their strata layer.
    for later, earlier in PROPAGATION.items():
        assert later == earlier + 1
    print(json.dumps({"selftest": "PASS", "planned": len(items),
                      "strata": STRATA, "propagation": PROPAGATION}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "run":
        raise SystemExit(run())
    if command == "analysis":
        raise SystemExit(analysis())
    if command == "plan":
        print(json.dumps({"plan": [[a, b] for a, b in plan()],
                          "remaining": [[a, b] for a, b in remaining()]}, indent=2))
    elif command == "status":
        print(json.dumps(status(), indent=2))
    elif command == "selftest":
        raise SystemExit(selftest())
    else:
        raise SystemExit(f"unknown command: {command}")
