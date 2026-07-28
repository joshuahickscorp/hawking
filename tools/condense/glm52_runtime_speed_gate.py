#!/usr/bin/env python3.12
"""Bind a true batch-1 GLM runtime measurement to one exact artifact index.

The default is plan-only. ``--run`` is deliberately required because even the
short profile executes the whole model; ``--sustained`` selects the canonical
2K/8K/32K, 80-token campaign and is therefore much heavier.

    python3.12 tools/condense/glm52_runtime_speed_gate.py --artifact <dir>
    python3.12 tools/condense/glm52_runtime_speed_gate.py --artifact <dir> --run --out SPEED.json
    python3.12 tools/condense/glm52_runtime_speed_gate.py --artifact <dir> \
        --sustained --run --out SUSTAINED.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INDEX_NAMES = (
    "model.gravity.index.json",
    "model.activation_aware.index.json",
)
MILESTONES_MS = (
    ("TG20", 20.0),
    ("TG10", 10.0),
    ("TG5", 5.0),
    ("TG2", 2.0),
    ("TG1", 1.0),
)


class RuntimeSpeedError(RuntimeError):
    """The benchmark could not produce an attributable measurement."""


def artifact_index(artifact: Path) -> Path:
    present = [artifact / name for name in INDEX_NAMES if (artifact / name).is_file()]
    if len(present) != 1:
        raise RuntimeSpeedError(
            f"{artifact}: expected exactly one of {INDEX_NAMES}, found "
            f"{[path.name for path in present]}"
        )
    return present[0]


def achieved_milestones(median_ms: float) -> list[str]:
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeSpeedError(f"invalid median latency {median_ms!r}")
    return [name for name, ceiling in MILESTONES_MS if median_ms <= ceiling]


def benchmark_command(
    artifact: Path,
    contexts: list[int],
    decode: int,
    raw_out: Path,
    *,
    verify_hash: bool,
    sustained: bool,
) -> list[str]:
    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "hawking-core",
        "--example",
        "gravity_glm_tps",
        "--",
        "--dir",
        str(artifact),
    ]
    for context in contexts:
        command.extend(["--context", str(context)])
    command.extend(["--decode", str(decode), "--out", str(raw_out)])
    if sustained:
        command.extend(["--token-curve", "--progress", str(raw_out.with_suffix(".jsonl"))])
    if not verify_hash:
        command.append("--no-verify-hash")
    return command


def validate_measurement(
    raw: dict[str, Any],
    *,
    expected_index: Path,
    expected_contexts: list[int],
    expected_decode: int,
    target_ms: float,
) -> dict[str, Any]:
    if raw.get("schema") != "hawking.gravity.glm_base_tps.v1":
        raise RuntimeSpeedError(f"unexpected benchmark schema {raw.get('schema')!r}")
    if raw.get("scoreboard") != "BASE_TRUE_TPS":
        raise RuntimeSpeedError("benchmark did not identify the BASE_TRUE_TPS scoreboard")
    if raw.get("verify_hash") is not True:
        raise RuntimeSpeedError("artifact verification was disabled")

    bound = raw.get("artifact") or {}
    expected_hash = hashlib.sha256(expected_index.read_bytes()).hexdigest()
    if bound.get("index") != expected_index.name:
        raise RuntimeSpeedError("receipt names a different artifact index")
    if bound.get("index_sha256") != expected_hash:
        raise RuntimeSpeedError("receipt artifact index hash does not match the measured path")

    rows = raw.get("measurements")
    if not isinstance(rows, list) or len(rows) != len(expected_contexts):
        raise RuntimeSpeedError("benchmark returned the wrong number of context rows")
    by_context = {row.get("context_tokens"): row for row in rows}
    if set(by_context) != set(expected_contexts):
        raise RuntimeSpeedError("benchmark context set differs from the requested set")

    graded = []
    for context in expected_contexts:
        row = by_context[context]
        samples = row.get("decode_ms_per_token_all")
        if row.get("decode_tokens") != expected_decode:
            raise RuntimeSpeedError(f"context {context}: decode count differs")
        if not isinstance(samples, list) or len(samples) != expected_decode:
            raise RuntimeSpeedError(f"context {context}: incomplete latency samples")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in samples
        ):
            raise RuntimeSpeedError(f"context {context}: invalid latency sample")
        median = float(row.get("decode_ms_per_token_median", math.nan))
        measured_tps = float(row.get("base_true_decode_tps", math.nan))
        recomputed_tps = expected_decode * 1000.0 / sum(float(v) for v in samples)
        if not math.isclose(measured_tps, recomputed_tps, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeSpeedError(f"context {context}: TPS does not reconcile")
        graded.append(
            {
                "context_tokens": context,
                "median_ms_per_token": median,
                "base_true_decode_tps": measured_tps,
                "achieved_milestones": achieved_milestones(median),
                "target_ms": target_ms,
                "target_met": median <= target_ms,
            }
        )

    return {
        "schema": "hawking.glm52.runtime_speed_gate.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_index": expected_index.name,
        "artifact_index_sha256": expected_hash,
        "scoreboard": "BASE_TRUE_TPS",
        "true_batch_1": True,
        "non_speculative": True,
        "contexts": graded,
        "target_ms": target_ms,
        "target_verdict": "PASS" if all(row["target_met"] for row in graded) else "FAIL",
        "measurement_verdict": "VALID",
        "raw_measurement": raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--context", action="append", type=int)
    parser.add_argument("--decode", type=int)
    parser.add_argument("--target-ms", type=float, default=2.0)
    parser.add_argument(
        "--sustained",
        action="store_true",
        help="select 2K/8K/32K contexts and at least 80 decode tokens",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the heavy whole-model benchmark; default prints its command only",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()

    artifact = args.artifact.expanduser().resolve()
    index = artifact_index(artifact)
    contexts = args.context or ([2048, 8192, 32768] if args.sustained else [8, 16])
    decode = args.decode or (80 if args.sustained else 4)
    if any(context <= 0 for context in contexts):
        raise RuntimeSpeedError("contexts must be positive")
    if decode <= 0:
        raise RuntimeSpeedError("decode must be positive")
    if args.sustained and (set(contexts) != {2048, 8192, 32768} or decode < 80):
        raise RuntimeSpeedError(
            "--sustained requires exactly 2K/8K/32K and at least 80 decode tokens"
        )
    if args.target_ms <= 0 or not math.isfinite(args.target_ms):
        raise RuntimeSpeedError("--target-ms must be finite and positive")
    if args.run and args.no_verify_hash:
        raise RuntimeSpeedError("a speed gate cannot run with artifact verification disabled")

    with tempfile.TemporaryDirectory(prefix="glm52-runtime-speed-") as temporary:
        raw_out = Path(temporary) / "raw.json"
        command = benchmark_command(
            artifact,
            contexts,
            decode,
            raw_out,
            verify_hash=not args.no_verify_hash,
            sustained=args.sustained,
        )
        if not args.run:
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "heavy_execution_started": False,
                        "artifact": str(artifact),
                        "artifact_index": index.name,
                        "artifact_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                        "contexts": contexts,
                        "decode_tokens": decode,
                        "sustained": args.sustained,
                        "target_ms": args.target_ms,
                        "command": command,
                        "note": "rerun with --run only under a valid heavy window",
                    },
                    indent=2,
                )
            )
            return 0

        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeSpeedError(
                f"benchmark failed ({result.returncode}): "
                f"{(result.stderr or result.stdout)[-2000:]}"
            )
        raw = json.loads(raw_out.read_text())
        receipt = validate_measurement(
            raw,
            expected_index=index,
            expected_contexts=contexts,
            expected_decode=decode,
            target_ms=args.target_ms,
        )
        receipt["benchmark_stderr_tail"] = result.stderr[-2000:]
        text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.write_text(text)
        print(text, end="")
        return 0 if receipt["target_verdict"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RuntimeSpeedError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(json.dumps({"measurement_verdict": "ERROR", "error": str(error)}, indent=2))
        raise SystemExit(2) from error
