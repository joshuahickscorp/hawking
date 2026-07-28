#!/usr/bin/env python3.12
"""Post-capability Python-oracle versus Rust/Metal whole-model agreement gate.

This is intentionally separate from ``glm52_capability_gate.py``:

* capability asks whether the artifact itself generates prompt-conditioned output;
* this gate asks whether the production Rust/Metal runtime agrees with that oracle.

The comparison is between two f32 backends, not an FP64 authority, so it is
labelled runtime agreement rather than Numeric Parity V2.1. Discrete decisions
(argmax and ordered top-5) are exact; continuous agreement uses relative L2
and cosine. Without ``--run`` the command only prints the future heavy commands.

    python3.12 tools/condense/glm52_runtime_parity_gate.py --artifact <dir>
    python3.12 tools/condense/glm52_runtime_parity_gate.py --artifact <dir> \
        --suite capability --run --out PARITY.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from glm52_capability_gate import G_LIVE_PROMPTS, G_MATH_TOKENS


ROOT = Path(__file__).resolve().parents[2]
INDEX_NAMES = (
    "model.gravity.index.json",
    "model.activation_aware.index.json",
)
MAX_RELATIVE_L2 = 1e-5
MIN_COSINE = 1.0 - 1e-7


class RuntimeParityError(RuntimeError):
    """The gate could not produce an honest comparison."""


def prompt_cases(
    suite: str,
    tokens: list[int] | None,
) -> list[tuple[str, list[int]]]:
    if suite == "capability":
        if tokens is not None:
            raise RuntimeParityError(
                "--tokens cannot be combined with --suite capability"
            )
        return [
            ("math", list(G_MATH_TOKENS)),
            *[(name, list(prompt)) for name, prompt in G_LIVE_PROMPTS],
        ]
    if suite != "single":
        raise RuntimeParityError(f"unknown prompt suite {suite!r}")
    return [("single", list(tokens or G_MATH_TOKENS))]


def artifact_index(artifact: Path) -> Path:
    present = [artifact / name for name in INDEX_NAMES if (artifact / name).is_file()]
    if len(present) != 1:
        raise RuntimeParityError(
            f"{artifact}: expected exactly one of {INDEX_NAMES}, found "
            f"{[path.name for path in present]}"
        )
    return present[0]


def stable_top(values: np.ndarray, k: int) -> list[int]:
    return [
        int(index)
        for index in np.argsort(-np.asarray(values), kind="stable")[:k]
    ]


def compare_logits(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).ravel()
    candidate = np.asarray(candidate, dtype=np.float32).ravel()
    if reference.shape != candidate.shape or reference.size == 0:
        raise RuntimeParityError(
            f"logit shapes differ or are empty: {reference.shape} vs {candidate.shape}"
        )
    finite = bool(np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate)))
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    diff = candidate64 - reference64
    relative_l2 = float(
        np.linalg.norm(diff) / max(float(np.linalg.norm(reference64)), 1e-30)
    )
    denominator = max(
        float(np.linalg.norm(reference64) * np.linalg.norm(candidate64)), 1e-30
    )
    cosine = float(np.dot(reference64, candidate64) / denominator)
    max_abs = float(np.max(np.abs(diff)))
    reference_top5 = stable_top(reference, 5)
    candidate_top5 = stable_top(candidate, 5)
    argmax_match = reference_top5[0] == candidate_top5[0]
    top5_match = reference_top5 == candidate_top5
    passed = (
        finite
        and relative_l2 <= MAX_RELATIVE_L2
        and cosine >= MIN_COSINE
        and argmax_match
        and top5_match
    )
    failures = []
    if not finite:
        failures.append("non-finite logits")
    if relative_l2 > MAX_RELATIVE_L2:
        failures.append(
            f"relative_l2 {relative_l2} > {MAX_RELATIVE_L2}"
        )
    if cosine < MIN_COSINE:
        failures.append(f"cosine {cosine} < {MIN_COSINE}")
    if not argmax_match:
        failures.append("argmax differs")
    if not top5_match:
        failures.append("ordered top-5 differs")
    return {
        "schema": "hawking.glm52.runtime_agreement.v1",
        "authority": "numpy f32 artifact oracle (not FP64 Numeric Parity V2.1)",
        "n_logits": int(reference.size),
        "all_finite": finite,
        "relative_l2": relative_l2,
        "max_relative_l2": MAX_RELATIVE_L2,
        "cosine": cosine,
        "min_cosine": MIN_COSINE,
        "max_abs": max_abs,
        "reference_argmax": reference_top5[0],
        "candidate_argmax": candidate_top5[0],
        "argmax_exact": argmax_match,
        "reference_top5": reference_top5,
        "candidate_top5": candidate_top5,
        "ordered_top5_exact": top5_match,
        "pass": passed,
        "failures": failures,
    }


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeParityError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{(result.stderr or result.stdout)[-2000:]}"
        )
    return result


def _last_json(stdout: str) -> dict[str, Any]:
    starts = [index for index, char in enumerate(stdout) if char == "{"]
    for start in reversed(starts):
        try:
            return json.loads(stdout[start:])
        except json.JSONDecodeError:
            continue
    raise RuntimeParityError("Rust runtime emitted no parseable JSON object")


def commands(
    artifact: Path,
    tokens: list[int],
    oracle_dump: Path,
    runtime_dump: Path,
    *,
    verify_hash: bool,
) -> tuple[list[str], list[str]]:
    verify_arg = [] if verify_hash else ["--no-verify-hash"]
    oracle = [
        sys.executable,
        str(ROOT / "tools/condense/glm52_flagship_oracle.py"),
        "--dir",
        str(artifact),
        "--tokens",
        *map(str, tokens),
        "--dump",
        str(oracle_dump),
        *verify_arg,
    ]
    runtime = [
        "cargo",
        "run",
        "--release",
        "-p",
        "hawking-core",
        "--example",
        "gravity_glm_flagship",
        "--",
        "--dir",
        str(artifact),
        "--tokens",
        *map(str, tokens),
        "--dump",
        str(runtime_dump),
        "--gpu",
        "--json",
        *verify_arg,
    ]
    return oracle, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tokens", nargs="+", type=int)
    parser.add_argument(
        "--suite",
        choices=("single", "capability"),
        default="single",
        help="capability compares math plus both live prompts; single uses --tokens or G_math",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the heavy whole-model oracle and Metal runtime; default prints commands only",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=4 * 60 * 60)
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()

    artifact = args.artifact.expanduser().resolve()
    index = artifact_index(artifact)
    cases = prompt_cases(args.suite, args.tokens)
    with tempfile.TemporaryDirectory(prefix="glm52-runtime-parity-") as temporary:
        temporary_path = Path(temporary)
        planned = []
        for name, tokens in cases:
            oracle_dump = temporary_path / f"{name}.oracle.f32"
            runtime_dump = temporary_path / f"{name}.runtime.f32"
            oracle_command, runtime_command = commands(
                artifact,
                tokens,
                oracle_dump,
                runtime_dump,
                verify_hash=not args.no_verify_hash,
            )
            planned.append(
                {
                    "name": name,
                    "tokens": tokens,
                    "oracle_dump": oracle_dump,
                    "runtime_dump": runtime_dump,
                    "oracle_command": oracle_command,
                    "runtime_command": runtime_command,
                }
            )
        if not args.run:
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "heavy_execution_started": False,
                        "artifact": str(artifact),
                        "index": str(index),
                        "suite": args.suite,
                        "cases": [
                            {
                                "name": row["name"],
                                "tokens": row["tokens"],
                                "oracle_command": row["oracle_command"],
                                "runtime_command": row["runtime_command"],
                            }
                            for row in planned
                        ],
                        "note": "rerun with --run only under a valid heavy window",
                    },
                    indent=2,
                )
            )
            return 0

        started = time.time()
        case_receipts = []
        for row in planned:
            oracle_result = _run(row["oracle_command"], args.timeout_seconds)
            runtime_result = _run(row["runtime_command"], args.timeout_seconds)
            oracle_logits = np.fromfile(row["oracle_dump"], dtype="<f4")
            runtime_logits = np.fromfile(row["runtime_dump"], dtype="<f4")
            comparison = compare_logits(oracle_logits, runtime_logits)
            case_receipts.append(
                {
                    "name": row["name"],
                    "tokens": row["tokens"],
                    "oracle_stderr_tail": oracle_result.stderr[-1000:],
                    "runtime": _last_json(runtime_result.stdout),
                    "comparison": comparison,
                }
            )
        passed = all(row["comparison"]["pass"] for row in case_receipts)
        receipt = {
            "schema": "hawking.glm52.runtime_parity_gate.v1",
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifact": str(artifact),
            "artifact_index": index.name,
            "artifact_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            "suite": args.suite,
            "cases": case_receipts,
            "elapsed_seconds": time.time() - started,
            "verdict": "PASS" if passed else "FAIL",
            "claims": {
                "runtime_agreement_only": True,
                "fp64_numeric_parity_v2_1": False,
                "quality": False,
                "speed": False,
            },
        }
        if len(case_receipts) == 1:
            receipt["tokens"] = case_receipts[0]["tokens"]
            receipt["runtime"] = case_receipts[0]["runtime"]
            receipt["comparison"] = case_receipts[0]["comparison"]
        text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.write_text(text)
        print(text, end="")
        return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeParityError, OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"verdict": "ERROR", "error": str(error)}, indent=2))
        raise SystemExit(2) from error
