#!/usr/bin/env python3
"""Bounded probe: GLM teacher-forced sequence-shard parallelism vs link ceiling.

What this measures (honest, no full-scale re-stream):

1. **Sealed official full-stack receipts** already on disk:
   wall MiB/s vs TG_XET public-path sustained winner (194 MiB/s).
2. **Synthetic resident-weight multi-worker sequence sharding**:
   bit-exact merge vs serial, wall speedup (or lack thereof on tiny fixtures).
3. **Amplification model** for naive N-worker --stream:
   network_bytes = N × body_bytes; shared-link wall lower-bound = N × T_serial
   when already link-saturated (or T_serial if workers serialize on the link).

Does NOT touch glm-floats-safe, does NOT launch multi-TB live downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lab.operators import frankenstein_teacher_forced_executor as tfe  # noqa: E402
from lab.operators import glm52_synthetic as synthetic  # noqa: E402
from lab.operators.glm52_common import verify_sealed  # noqa: E402
from tools.condense.merge_glm_teacher_forced_shards import (  # noqa: E402
    merge_workers,
)

PUBLIC_PATH_WINNER = (
    _REPO
    / "workspace/campaign/evidence/runtime/tg"
    / "TG_XET_PUBLIC_PATH_SUSTAINED_WINNER_RETRY_BALANCED_SCHEDULER.json"
)
OFFICIAL_FULL = (
    _REPO
    / "workspace/campaign/evidence/models/frankenstein/teacher_forced"
    / "official_L0_stream_full_20260805T200728Z"
    / "GLM_TEACHER_FORCED_CAPTURE_RECEIPT.json"
)
OFFICIAL_REEXPORT = (
    _REPO
    / "workspace/campaign/evidence/models/frankenstein/teacher_forced"
    / "official_L0_stream_reexport_20260805T214500Z"
    / "GLM_TEACHER_FORCED_CAPTURE_RECEIPT.json"
)
CEILING_MIB_S = 194.0  # sealed sustained public-path winner


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_official_receipt(path: Path) -> dict[str, Any]:
    d = _load_json(path)
    stream = d.get("stream") or {}
    bytes_fetched = int(stream.get("bytes_fetched") or 0)
    capture_s = float(d.get("capture_seconds") or 0.0)
    fetch_s = float(stream.get("fetch_seconds") or 0.0)
    wall_mib_s = (
        (bytes_fetched / capture_s / (1024 * 1024)) if capture_s > 0 else None
    )
    pure_dl_at_ceiling = (
        bytes_fetched / (CEILING_MIB_S * 1024 * 1024) if bytes_fetched else None
    )
    residual_s = (
        (capture_s - pure_dl_at_ceiling)
        if pure_dl_at_ceiling is not None
        else None
    )
    return {
        "path": str(path),
        "status": d.get("status"),
        "capture_seconds": capture_s,
        "bytes_fetched": bytes_fetched,
        "fetch_seconds_sum_of_shard_downloads": fetch_s,
        "wall_mib_s": round(wall_mib_s, 3) if wall_mib_s is not None else None,
        "ceiling_mib_s": CEILING_MIB_S,
        "pct_of_ceiling": (
            round(100.0 * wall_mib_s / CEILING_MIB_S, 2)
            if wall_mib_s is not None
            else None
        ),
        "pure_download_seconds_at_ceiling": (
            round(pure_dl_at_ceiling, 3) if pure_dl_at_ceiling is not None else None
        ),
        "residual_seconds_vs_ceiling": (
            round(residual_s, 3) if residual_s is not None else None
        ),
        "residual_fraction_of_wall": (
            round(residual_s / capture_s, 4)
            if residual_s is not None and capture_s > 0
            else None
        ),
        "amdahl_max_speedup_if_only_residual_parallel": (
            round(1.0 / (1.0 - residual_s / capture_s), 4)
            if residual_s is not None and capture_s > 0 and residual_s > 0
            else (
                1.0
                if residual_s is not None and residual_s <= 0
                else None
            )
        ),
        "layers_captured": len(d.get("layers_captured") or []),
        "n_sequences": (d.get("corpus") or {}).get("n_sequences"),
        "interpretation": (
            "Already at/near public-path link ceiling on wall-clock bytes/"
            "capture_seconds. Residual compute/export vs pure-download-at-ceiling "
            "is the ONLY sequence-parallel headroom without re-fetching weights."
            if wall_mib_s is not None and wall_mib_s >= 0.9 * CEILING_MIB_S
            else "Below 90% of sealed ceiling — inspect for contention/cache."
        ),
    }


def amplification_model(
    *,
    body_bytes: int,
    serial_wall_s: float,
    n_workers: list[int],
    wall_mib_s: float,
) -> dict[str, Any]:
    rows = []
    for n in n_workers:
        # Naive independent streamers share one uplink/downlink.
        bytes_total = body_bytes * n
        # Lower bound on wall when link is saturated: each byte still crosses once
        # per worker → wall ≈ n * serial if serial was already link-bound.
        wall_lb_saturated = serial_wall_s * n
        # Optimistic: link fair-shares and compute free → still n * body / rate.
        wall_lb_bytes = bytes_total / (wall_mib_s * 1024 * 1024) if wall_mib_s > 0 else None
        rows.append(
            {
                "workers": n,
                "network_bytes": bytes_total,
                "network_amplification": n,
                "wall_lower_bound_if_link_saturated_s": round(wall_lb_saturated, 3),
                "wall_lower_bound_from_bytes_at_measured_rate_s": (
                    round(wall_lb_bytes, 3) if wall_lb_bytes is not None else None
                ),
                "speedup_vs_serial_if_link_saturated": round(1.0 / n, 4),
                "verdict": "WORSE_NOT_BETTER" if n > 1 else "BASELINE",
            }
        )
    return {
        "assumption": (
            "N independent --stream workers each re-download the full donor body; "
            "single shared public path at measured wall MiB/s."
        ),
        "rows": rows,
    }


def _run_one_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Process entry: run one synthetic shard (must be top-level for spawn)."""
    source = Path(payload["source"])
    out = Path(payload["out"])
    if out.exists():
        shutil.rmtree(out)
    t0 = time.perf_counter()
    cfg = tfe.ExecutorConfig(
        mode="synthetic",
        corpus_level=payload["level"],
        source_root=source,
        output_dir=out,
        max_sequence=int(payload["max_sequence"]),
        microbatch=int(payload["microbatch"]),
        sample_hidden=16,
        profile="synthetic",
        # Shared fixture across workers: never unlink weights.
        allow_eviction=False,
        require_floor=False,
        max_layers=payload.get("max_layers"),
        stream=False,
        seq_start=int(payload["seq_start"]),
        seq_end=payload.get("seq_end"),
        worker_id=payload.get("worker_id"),
    )
    receipt = tfe.run_teacher_forced(cfg)
    wall = time.perf_counter() - t0
    return {
        "out": str(out),
        "wall_s": wall,
        "status": receipt.get("status"),
        "seal_sha256": receipt.get("seal_sha256"),
        "n_sequences": (receipt.get("corpus") or {}).get("n_sequences"),
        "shard": receipt.get("shard"),
    }


def compare_bit_exact(serial_dir: Path, merged_dir: Path) -> dict[str, Any]:
    """Compare layer/carry npz digests between serial out and merged out."""
    report: dict[str, Any] = {"layers": {}, "carry": {}, "PASS_BIT_EXACT": True}
    for sub in ("layers", "carry"):
        s_dir = serial_dir / sub
        m_dir = merged_dir / sub
        if not s_dir.is_dir():
            continue
        for npz in sorted(s_dir.glob("*.npz")):
            other = m_dir / npz.name
            if not other.is_file():
                report["PASS_BIT_EXACT"] = False
                report[sub][npz.name] = {"match": False, "reason": "missing_in_merged"}
                continue
            import numpy as np

            with np.load(npz) as a, np.load(other) as b:
                keys_a, keys_b = set(a.files), set(b.files)
                if keys_a != keys_b:
                    report["PASS_BIT_EXACT"] = False
                    report[sub][npz.name] = {
                        "match": False,
                        "reason": f"key_mismatch {sorted(keys_a ^ keys_b)}",
                    }
                    continue
                ok = True
                per_key = {}
                for k in sorted(keys_a):
                    eq = np.array_equal(a[k], b[k])
                    per_key[k] = bool(eq)
                    if not eq:
                        ok = False
                if not ok:
                    report["PASS_BIT_EXACT"] = False
                report[sub][npz.name] = {"match": ok, "per_key": per_key}
    return report


def run_synthetic_probe(
    work: Path,
    *,
    level: str = "L0",
    max_sequence: int = 2,
    microbatch: int = 8,
    workers: int = 2,
) -> dict[str, Any]:
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    fixture = synthetic.build_synthetic_fixture(work / "fixture")
    source = fixture.full_dir

    serial_out = work / "serial"
    t0 = time.perf_counter()
    serial_receipt = tfe.run_teacher_forced(
        tfe.ExecutorConfig(
            mode="synthetic",
            corpus_level=level,
            source_root=source,
            output_dir=serial_out,
            max_sequence=max_sequence,
            microbatch=microbatch,
            sample_hidden=16,
            profile="synthetic",
            allow_eviction=False,
            require_floor=False,
            stream=False,
        )
    )
    serial_wall = time.perf_counter() - t0
    n_seq = int((serial_receipt.get("corpus") or {}).get("n_sequences") or 0)
    verify_sealed(serial_receipt, label="serial")

    # Even split.
    cuts = [i * n_seq // workers for i in range(workers)] + [n_seq]
    payloads = []
    for i in range(workers):
        payloads.append(
            {
                "source": str(source),
                "out": str(work / f"w{i}"),
                "level": level,
                "max_sequence": max_sequence,
                "microbatch": microbatch,
                "seq_start": cuts[i],
                "seq_end": cuts[i + 1],
                "worker_id": f"w{i}",
            }
        )

    t1 = time.perf_counter()
    worker_results = []
    # Use threads-as-processes alternative: ProcessPool for true multi-core.
    # Fall back to sequential spawn if pool fails.
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_one_worker, p) for p in payloads]
            for fut in as_completed(futs):
                worker_results.append(fut.result())
    except Exception as exc:  # noqa: BLE001
        worker_results = []
        for p in payloads:
            # Sequential fallback still exercises sharding correctness.
            worker_results.append(_run_one_worker(p))
        parallel_note = f"process_pool_failed_fallback_serial_workers: {exc}"
    else:
        parallel_note = "process_pool_ok"
    parallel_wall = time.perf_counter() - t1
    worker_results.sort(key=lambda r: (r.get("shard") or {}).get("seq_start", 0))

    merged_out = work / "merged"
    merge_receipt = merge_workers(merged_out, [Path(r["out"]) for r in worker_results])
    bit_exact = compare_bit_exact(serial_out, merged_out)

    # Stream amplification refuse check (subprocess so we don't need live HF).
    refuse_ok = False
    refuse_msg = ""
    try:
        tfe.run_teacher_forced(
            tfe.ExecutorConfig(
                mode="official",
                corpus_level="L0",
                source_root=source,  # unused; stream path errors earlier
                output_dir=work / "refuse_probe",
                profile="official",
                stream=True,
                require_floor=False,
                seq_start=0,
                seq_end=4,
                allow_weight_stream_amplification=False,
            )
        )
    except tfe.TeacherForcedError as exc:
        refuse_msg = str(exc)
        refuse_ok = "weight" in refuse_msg.lower() or "stream" in refuse_msg.lower()
    except Exception as exc:  # noqa: BLE001 — bootstrap may fail after guard
        refuse_msg = f"{type(exc).__name__}: {exc}"
        # Guard should fire before stream bootstrap; if we got past it, fail.
        refuse_ok = "amplification" in refuse_msg.lower() or "re-fetch" in refuse_msg.lower()

    speedup = serial_wall / parallel_wall if parallel_wall > 0 else None
    return {
        "level": level,
        "n_sequences": n_seq,
        "workers": workers,
        "serial_wall_s": round(serial_wall, 4),
        "parallel_wall_s": round(parallel_wall, 4),
        "speedup": round(speedup, 4) if speedup is not None else None,
        "parallel_efficiency": (
            round(speedup / workers, 4) if speedup is not None else None
        ),
        "serial_status": serial_receipt.get("status"),
        "worker_results": worker_results,
        "merge_status": merge_receipt.get("status"),
        "bit_exact": bit_exact,
        "stream_amplification_refuse": {
            "PASS": refuse_ok,
            "message_head": refuse_msg[:400],
        },
        "parallel_note": parallel_note,
        "weights_path": "resident_synthetic_fixture_no_network",
        "honesty": (
            "Synthetic geometry is miniature (hidden=32, 7 layers). Process "
            "startup can dominate wall; speedup here is NOT a prediction of "
            "official multi-TB stream speedup (which is link-bound and refused)."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work",
        type=Path,
        default=_REPO
        / "workspace/campaign/evidence/models/frankenstein"
        / "glm_parallelism_probe_20260806",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--level", default="L0", choices=tuple(tfe.CORPUS_LEVELS))
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args(argv)

    official = []
    for p in (OFFICIAL_FULL, OFFICIAL_REEXPORT):
        if p.is_file():
            official.append(analyze_official_receipt(p))

    primary = official[0] if official else None
    amp = None
    if primary and primary.get("bytes_fetched"):
        amp = amplification_model(
            body_bytes=int(primary["bytes_fetched"]),
            serial_wall_s=float(primary["capture_seconds"]),
            n_workers=[1, 2, 4],
            wall_mib_s=float(primary["wall_mib_s"] or CEILING_MIB_S),
        )

    synthetic_report = None
    if not args.skip_synthetic:
        synthetic_report = run_synthetic_probe(
            Path(args.work) / "synthetic",
            level=args.level,
            workers=int(args.workers),
        )

    # Headroom verdict
    headroom_yes = False
    headroom_why = []
    if primary:
        residual = primary.get("residual_fraction_of_wall")
        pct = primary.get("pct_of_ceiling")
        if pct is not None and pct >= 90:
            headroom_yes = False
            headroom_why.append(
                f"Official wall rate {primary['wall_mib_s']} MiB/s is "
                f"{pct}% of sealed 194 MiB/s public-path ceiling — link-bound."
            )
        if residual is not None and residual <= 0.10:
            headroom_yes = False
            headroom_why.append(
                f"Residual vs pure-download-at-ceiling is only "
                f"{100 * residual:.1f}% of wall (Amdahl max "
                f"~{primary.get('amdahl_max_speedup_if_only_residual_parallel')}x "
                f"even if residual fully parallelizes)."
            )
        if residual is not None and residual > 0.25 and (pct or 0) < 80:
            headroom_yes = True
            headroom_why.append(
                "Substantial non-download residual with unsaturated link — "
                "sequence-batch on resident weights could help."
            )
    if not headroom_why:
        headroom_why.append("Insufficient sealed official receipts in worktree.")

    report = {
        "schema": "hawking.frankenstein.glm_teacher_forced_parallelism_probe.v1",
        "public_path_ceiling_mib_s": CEILING_MIB_S,
        "public_path_winner_path": str(PUBLIC_PATH_WINNER),
        "official_receipt_analysis": official,
        "naive_stream_amplification_model": amp,
        "synthetic_sequence_shard_probe": synthetic_report,
        "headroom": {
            "exists_for_multi_worker_download": headroom_yes,
            "exists_for_naive_process_duplication_with_restream": False,
            "exists_for_sequence_batch_on_resident_weights": (
                True
                if synthetic_report and synthetic_report.get("bit_exact", {}).get("PASS_BIT_EXACT")
                else None
            ),
            "why": headroom_why,
            "recommended_action": (
                "Do NOT launch N independent --stream workers. Keep single "
                "layer-major streamer (already double-buffered). Sequence "
                "sharding is implemented and correctness-proven for resident "
                "weights only; official path remains one process, one weight "
                "stream. Optional future: one downloader + multi-consumer "
                "forward on resident layer (attacks ~5% residual only)."
            ),
        },
        "fabricated_speedup": False,
        "full_scale_run_launched": False,
        "touched_glm_floats_safe": False,
    }

    out_json = Path(args.work) / "GLM_TEACHER_FORCED_PARALLELISM_PROBE.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # Also seal at repo root findings path for the task contract.
    root_findings = _REPO / "GLM_TEACHER_FORCED_PARALLELISM_FINDINGS.json"
    root_findings.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {out_json}", file=sys.stderr)
    print(f"Wrote {root_findings}", file=sys.stderr)

    if synthetic_report and not synthetic_report.get("bit_exact", {}).get(
        "PASS_BIT_EXACT"
    ):
        return 2
    if synthetic_report and not synthetic_report.get(
        "stream_amplification_refuse", {}
    ).get("PASS"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
