"""Measured KIMI Gravity fan-out for the sub-1 complete-EBPW frontier.

The scientific fan-out is broad, but MLX execution is intentionally single-lane:
loading multiple 32 GB-class bodies concurrently would measure contention and
memory pressure instead of the representation. Each candidate is independent,
checkpointed after completion, and carries the evaluator's phase/resource trace.

This is a screen, not promotion. The sparse families currently reconstruct a
dense bf16 tensor inside the evaluator because no production sparse kernel exists;
their capability and byte measurements are useful, but direct execution remains
unearned until a native decoder is implemented and timed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "future"))
import gravity_outlier_eval as G  # noqa: E402


SPECS = (
    # Existing byte-frontier control, rerun with the new phase telemetry.
    "sparse0.05-g128",
    # New materially different discriminators.
    "sparseact0.05percal-g128",
    "sparseact0.0625percal-g128",
    "ternarysparse0.05percal-g128",
    "organsparse0.02u0.10percal-g128",
    "organsparse0.03u0.12percal-g128",
    "organupternary0.02u0.10percal-g128",
    "organdownternary0.02u0.10percal-g128",
    "organupternary0.03u0.12percal-g128",
    "organdownternary0.03u0.12percal-g128",
    # Just over the target: a useful capability-vs-bytes ladder point.
    "organsparse0.04u0.14percal-g128",
    # Compound frontier: sparse MoE expert bodies plus a separately billed
    # 2-bit/group-128 non-expert stream.  These are deliberately distinct from
    # the first screen: they test whether the non-expert floor, rather than the
    # expert codec, is the next byte bottleneck.
    "sparseact0.05percal-g128+o2g128",
    "sparseact0.0625percal-g128+o2g128",
    "ternarysparse0.05percal-g128+o2g128",
    "organsparse0.03u0.12percal-g128+o2g128",
    "organupternary0.03u0.12percal-g128+o2g128",
    "organdownternary0.03u0.12percal-g128+o2g128",
    "organsparse0.04u0.14percal-g128+o2g128",
    "sparseact0.08percal-g128+o2g64",
)

OUT = ROOT / "receipts" / "future" / "KIMI_GRAVITY_FANOUT_20260910.json"


def _atomic_write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def _checkpoint(rows: list[dict], started: float, status: str) -> None:
    _atomic_write(
        OUT,
        {
            "schema": "hawking.future.kimi_gravity_fanout.v1",
            "status": status,
            "specimen": "KIMI_BASE",
            "model_source": G.SNAP,
            "execution_lane": "single MLX physical lane; candidate fan-out is sequential",
            "complete_ebpw_target": 1.0,
            "claim_boundary": (
                "Measured KIMI full-model screen. Complete bytes, capability, phase timing, "
                "and resource snapshots are evidence. Sparse forms have no native decoder, "
                "so direct execution and TPS remain unmeasured. No promotion evidence."
            ),
            "frontier": [
                {"spec": s, "predicted_complete_ebpw": round(G.predict_ebpw(s), 6)}
                for s in SPECS
            ],
            "rows": rows,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "updated_at": time.time(),
        },
    )


def main() -> int:
    if "--worker" in sys.argv:
        code = _worker_main()
        # MLX has occasionally faulted during interpreter teardown after a
        # valid receipt was flushed. The worker owns no persistent state after
        # its atomic write; terminate cleanly at the OS boundary so the parent
        # records scientific results rather than destructor noise.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    started = time.perf_counter()
    prior: dict[str, dict] = {}
    if OUT.exists():
        try:
            prior = {str(r["spec"]): r for r in json.loads(OUT.read_text()).get("rows", [])}
        except (OSError, KeyError, TypeError, ValueError):
            prior = {}
    rows: list[dict] = [prior[s] for s in SPECS if s in prior]
    for index, spec in enumerate(SPECS, 1):
        predicted = G.predict_ebpw(spec)
        old = prior.get(spec)
        if old and old.get("measurement_state") == "MEASURED_FULL_MODEL_SCREEN":
            print(f"[{index}/{len(SPECS)}] {spec} already measured; resuming", flush=True)
            continue
        print(f"[{index}/{len(SPECS)}] {spec} predicted_complete_ebpw={predicted:.6f}", flush=True)
        worker_out = OUT.with_name(f".{OUT.stem}.{index}.{os.getpid()}.worker.json")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", spec,
             "--index", str(index), "--out", str(worker_out)],
            cwd=ROOT,
            check=False,
        )
        if worker_out.exists():
            row = json.loads(worker_out.read_text())
            row["worker_exit_code"] = proc.returncode
            worker_out.unlink()
        else:
            row = {
                "schema": "hawking.future.kimi_gravity_fanout.row.v1",
                "specimen": "KIMI_BASE", "spec": spec, "fanout_index": index,
                "predicted_complete_ebpw": predicted,
                "_evidence": "MEASURED (worker failure)",
                "measurement_state": "RUNNER_FAILURE",
                "error": f"worker exited {proc.returncode} without a receipt",
                "execution_complete": False, "capability_ok": False,
            }
        rows = [r for r in rows if r.get("spec") != spec]
        rows.append(row)
        rows.sort(key=lambda r: int(r.get("fanout_index", 0)))
        print(json.dumps({k: row.get(k) for k in
                          ("spec", "complete_ebpw", "capability_ok",
                           "measurement_state", "wall_s", "worker_exit_code")}), flush=True)
        _checkpoint(rows, started, "ACTIVE_SINGLE_LANE")

    measured_under_1 = [
        r["spec"] for r in rows
        if isinstance(r.get("complete_ebpw"), (int, float))
        and r["complete_ebpw"] <= 1.0
    ]
    _checkpoint(rows, started, "COMPLETE")
    print(json.dumps({"status": "COMPLETE", "measured_under_1": measured_under_1}), flush=True)
    return 0


def _worker_main() -> int:
    """Run exactly one MLX arm in a fresh process and persist its row."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    spec = args.worker
    predicted = G.predict_ebpw(spec)
    try:
        row = G.evaluate(SimpleNamespace(spec=spec, specimen="KIMI_BASE"))
        row.update({"fanout_index": args.index,
                    "predicted_complete_ebpw": predicted,
                    "measurement_state": "MEASURED_FULL_MODEL_SCREEN"})
    except Exception as exc:
        row = {
            "schema": "hawking.future.kimi_gravity_fanout.row.v1",
            "specimen": "KIMI_BASE", "spec": spec, "fanout_index": args.index,
            "predicted_complete_ebpw": predicted,
            "_evidence": "MEASURED (runner failure)",
            "measurement_state": "RUNNER_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "execution_complete": False, "capability_ok": False,
        }
    _atomic_write(args.out, row)
    print(json.dumps({k: row.get(k) for k in
                      ("spec", "complete_ebpw", "capability_ok",
                       "measurement_state", "wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
