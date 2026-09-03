#!/usr/bin/env python3
"""Narrow, hash-bound provider capability runner for the one-mountain model.

This deliberately exercises only cheap G2/G3 provider checks.  It is not a
substitute for the full G0-G9 contract and never promotes throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CASES = (
    ("G2", "arithmetic_exact", "What is 2 + 2? Reply with only the numeral.", r"\b4\b"),
    (
        "G3",
        "dependent_retrieval",
        "Mira owns the violin. Dev owns the telescope. Who owns the violin? Reply with only the person's name.",
        r"\bMira\b",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-cli", type=Path, default=Path("/opt/homebrew/bin/llama-cli"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    model_sha = sha256(model)
    runs = []
    for gate, case_id, prompt, expected in CASES:
        command = [
            str(args.llama_cli), "-m", str(model), "-ngl", "999", "-c", "128",
            "-b", "16", "-ub", "16", "-n", "16", "--temp", "0", "--seed", "42",
            "-fa", "on", "-no-cnv", "--no-display-prompt", "-p", prompt,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        runs.append({
            "gate": gate,
            "case": case_id,
            "prompt": prompt,
            "expected_regex": expected,
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "pass": completed.returncode == 0 and re.search(expected, completed.stdout) is not None,
        })
    result = {
        "schema": "hawking.one_mountain.provider_capability_subset.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": {"path": str(model), "bytes": model.stat().st_size, "sha256": model_sha},
        "measurement_mode": "shared_load_provider_exercise",
        "runs": runs,
        "gates_passed": sorted({run["gate"] for run in runs if run["pass"]}),
        "unclaimed_gates": ["G0", "G1", "G4", "G5", "G6", "G7", "G8", "G9"],
        "limits": "No throughput, Gravity parity, HIDE, or full capability promotion follows from this subset.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if all(run["pass"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
