#!/usr/bin/env python3
"""Run Flash's smallest honest repeated-accepted-decode gate.

The source-bound Flash executor performs a complete 48-layer stateful session.
It needs a known second reference token in order to check two consecutive
terminal decisions.  This runner therefore has two intentionally distinct
phases:

1. discover the next terminal token from ``prompt + first_accepted_token``;
2. start one *new, explicitly recorded* verification session containing
   ``prompt + first_accepted_token + discovered_token``.

Only phase 2 can pass the repeated-decode gate.  It initializes state once at
the prompt boundary, retains it through both continuation tokens, and records
that it did not re-prefill inside the verification session.  Phase 1 is never
reported as persistence evidence.  The runner refuses a busy Hawking GPU lane
by default, preserving the protected measurement requirement.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
PROMPT_IDS = (5423, 799, 4581, 3817, 13)
FIRST_ACCEPTED_TOKEN = 2972
REPO_ID = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"receipt root must be an object: {path}")
    return payload


def _flash_binary_command(model_root: Path, token_ids: list[int], prompt_len: int, out: Path) -> list[str]:
    return [
        "cargo", "run", "--release", "-q", "-p", "hawking-core",
        "--example", "flash_stateful_complete_token_session", "--",
        "--root", str(model_root),
        "--token-ids", ",".join(str(token) for token in token_ids),
        "--prompt-length", str(prompt_len),
        "--out", str(out),
    ]


def _hawking_gpu_is_busy() -> bool:
    """Detect a live resident provider without treating a reaped child as live."""

    completed = subprocess.run(
        ["ps", "-axo", "stat=,command="], check=True, capture_output=True, text=True,
    )
    for row in completed.stdout.splitlines():
        state, _, command = row.strip().partition(" ")
        # A terminated child can remain visible briefly until its live
        # hawkingd parent reaps it.  It owns no GPU/UMA working set and must
        # not block a protected lane; every non-zombie matching row still does.
        if state.startswith("Z"):
            continue
        if "mlx_vlm.server" in command:
            return True
    return False


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _accepted_reference(doc: dict[str, Any]) -> tuple[int, int]:
    """Return the checked first token and the model-produced next reference."""

    _verify_source_identity(doc)
    checks = doc.get("terminal", {}).get("reference_checks", [])
    if not isinstance(checks, list) or len(checks) != 1:
        raise ValueError("reference-discovery session did not emit exactly one reference check")
    check = checks[0]
    if not isinstance(check, dict) or check.get("accepted") is not True:
        raise ValueError("first accepted reference token was rejected")
    expected = check.get("expected_token_id")
    next_token = doc.get("terminal", {}).get("next_after_reference")
    if not isinstance(expected, int) or not isinstance(next_token, int):
        raise ValueError("reference-discovery receipt omitted integer terminal tokens")
    return expected, next_token


def _verify_repeated(doc: dict[str, Any], references: list[int]) -> None:
    _verify_source_identity(doc)
    if doc.get("status") != "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE":
        raise ValueError(f"repeated session status was {doc.get('status')!r}")
    if doc.get("accepted_generation_tokens") != len(references):
        raise ValueError("repeated session did not accept every reference token")
    execution = doc.get("execution")
    if not isinstance(execution, dict) or execution.get("source_reset_or_reprefill") is not False:
        raise ValueError("verification session did not prove reset/re-prefill boundary")
    memory = execution.get("state_memory")
    if not isinstance(memory, dict) or int(memory.get("total_persistent_bytes", 0)) <= 0:
        raise ValueError("verification session omitted persistent-state census")
    if int(memory.get("growth_bytes_per_additional_token", 0)) <= 0:
        raise ValueError("verification session omitted state-growth census")
    if int(execution.get("source_payload_bytes_read", 0)) <= 0:
        raise ValueError("verification session omitted source-byte census")
    checks = doc.get("terminal", {}).get("reference_checks", [])
    if not isinstance(checks, list) or len(checks) != len(references):
        raise ValueError("verification session omitted independent reference checks")
    for expected, check in zip(references, checks):
        if not isinstance(check, dict) or check.get("expected_token_id") != expected:
            raise ValueError("verification reference order drifted")
        if check.get("predicted_token_id") != expected or check.get("accepted") is not True:
            raise ValueError("verification terminal check failed")


def _verify_source_identity(doc: dict[str, Any]) -> None:
    """Reject a plausible session receipt from an unpinned or wrong body."""

    if doc.get("model") != REPO_ID:
        raise ValueError(f"source model mismatch: {doc.get('model')!r}")
    if doc.get("pinned_revision") != PINNED_REVISION:
        raise ValueError("source revision mismatch")
    if doc.get("execution", {}).get("process_boundary") != "one native process":
        raise ValueError("source receipt did not establish one native process")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "receipts/headless/FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json",
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    model_root = args.root.resolve()
    out = args.out.resolve()
    discovery = out.with_name(f"{out.stem}.reference-discovery.json")
    first_tokens = [*PROMPT_IDS, FIRST_ACCEPTED_TOKEN]
    first_command = _flash_binary_command(model_root, first_tokens, len(PROMPT_IDS), discovery)

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
            "phase_1": first_command,
            "phase_2": "derived from phase_1 terminal.next_after_reference",
            "claim_boundary": "no GPU work, no repeated-decode result",
        }, indent=2))
        return 0
    if not args.allow_busy_gpu and _hawking_gpu_is_busy():
        print(json.dumps({
            "status": "BLOCKED_CLEAN_GPU_LANE",
            "reason": "a Hawking resident/provider is active; refusing an unprotected Flash run",
            "reopen_condition": "stop or otherwise quiesce the resident provider, then rerun this exact command",
        }, indent=2))
        return 3
    if not model_root.is_dir():
        raise FileNotFoundError(f"Flash source root does not exist: {model_root}")

    started_ns = time.time_ns()
    _run(first_command)
    discovered_doc = _json(discovery)
    first_reference, next_reference = _accepted_reference(discovered_doc)
    references = [first_reference, next_reference]
    verification_tokens = [*PROMPT_IDS, *references]
    verification_command = _flash_binary_command(
        model_root, verification_tokens, len(PROMPT_IDS), out,
    )
    _run(verification_command)
    final_doc = _json(out)
    _verify_repeated(final_doc, references)
    run_receipt = {
        "schema": "hawking.flash.repeated_accepted_decode_runner.v1",
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "model_root": str(model_root),
        "prompt_token_ids": list(PROMPT_IDS),
        "reference_generated_token_ids": references,
        "reference_discovery_receipt": str(discovery),
        "verification_receipt": str(out),
        "verification_session": "one source-bound session; prompt prefill occurs once at its start and no token-boundary reset or re-prefill is claimed",
        "elapsed_ns": time.time_ns() - started_ns,
        "claim_boundary": "This runner proves only the verification receipt it validates. It does not time a clean repeated benchmark, establish TPS, capability, EBPW, direct Noetic execution, or residency.",
    }
    manifest = out.with_name(f"{out.stem}.runner.json")
    manifest.write_text(json.dumps(run_receipt, indent=2) + "\n")
    print(json.dumps(run_receipt, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "FAILED_PRECISE_DIAGNOSIS", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
