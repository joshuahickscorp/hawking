#!/usr/bin/env python3.12
"""Deterministic, non-executing experiment matrix for speculative-decode re-entry.

This scaffold deliberately plans evidence cells only. It does not hash model
artifacts, launch inference, train a draft head, acquire a heavy lease, or alter
the active Doctor/ladder queue. Execution remains fail-closed in
``spec_revive.py`` until the TQ verifier parity and checkpointed-runner blockers
are implemented.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import doctor_v5_local_observer
import spec_receipt_contract


ROOT = HERE.parents[1]
SCHEMA = "hawking.spec_reentry_matrix.v1"
RUNTIME_PATHS = ("stored", "compact", "hashed", "computed")
WORKLOADS = ("code", "prose", "tool_json")
FREE_PROPOSERS = ("user_ngram", "suffix_array", "retrieval")
DRAFT_PRECISIONS = ("q4_control", "tq3", "tq2", "tq1")
HEAVY_PATTERNS = doctor_v5_local_observer.HEAVY_COMMAND_PATTERNS
PS_PATH = doctor_v5_local_observer.AUTHORITY_TOOL_PATHS["ps"]


def _cell_id(phase: str, family: str, knobs: dict) -> str:
    canonical = json.dumps(
        {"phase": phase, "family": family, "knobs": knobs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{phase.lower()}-{hashlib.sha256(canonical).hexdigest()[:12]}"


def _cell(
    phase: str,
    family: str,
    knobs: dict,
    *,
    depends_on: tuple[str, ...] = (),
    receipt_schema: str,
    heavy: bool,
) -> dict:
    return {
        "id": _cell_id(phase, family, knobs),
        "phase": phase,
        "family": family,
        "knobs": knobs,
        "depends_on": list(depends_on),
        "receipt_schema": receipt_schema,
        "requires_exclusive_heavy_lease": heavy,
        "state": "deferred",
    }


def build_matrix(label: str) -> dict:
    """Return the additive post-ladder matrix; no filesystem/model work."""
    cells: list[dict] = []
    parity_by_runtime: dict[str, str] = {}
    cost_by_runtime: dict[str, str] = {}

    for runtime in RUNTIME_PATHS:
        parity = _cell(
            "P0",
            "tq_batched_verifier_parity",
            {
                "runtime_path": runtime,
                "batch_sizes": list(range(1, 9)),
                "min_prompts": 20,
                "min_generated_tokens_per_prompt": 256,
                "reference": "tq_single_token_greedy",
                "candidate": "tq_batched_verify",
                "exact_token_match": True,
            },
            receipt_schema="hawking.spec_tq_batched_parity.v1",
            heavy=True,
        )
        cells.append(parity)
        parity_by_runtime[runtime] = parity["id"]

        cost = _cell(
            "P1",
            "verifier_cost_curve",
            {
                "runtime_path": runtime,
                "batch_sizes": list(range(1, 9)),
                "min_trials_per_batch": 5,
                "statistics": ["median", "p95", "ucb"],
                "normalize_to": "same_artifact_single_token_greedy",
                "record": [
                    "wall_ns",
                    "joules",
                    "bytes_moved",
                    "peak_memory_gib",
                    "swap_delta_mb",
                    "thermal_state",
                ],
            },
            depends_on=(parity["id"],),
            receipt_schema="hawking.spec_verifier_curve.v1",
            heavy=True,
        )
        cells.append(cost)
        cost_by_runtime[runtime] = cost["id"]

    oracle_ids: list[str] = []
    for proposer in FREE_PROPOSERS:
        for workload in WORKLOADS:
            oracle = _cell(
                "P2",
                "exact_free_proposer_oracle",
                {
                    "proposer": proposer,
                    "workload": workload,
                    "draft_lengths": list(range(2, 8)),
                    "min_prompts": 10,
                    "min_scored_tokens": 1024,
                    "history": "held_out_real_token_sessions",
                    "charge_lookup_and_miss_cost": True,
                },
                depends_on=(cost_by_runtime["stored"],),
                receipt_schema="hawking.spec_proposer_oracle.v1",
                heavy=False,
            )
            cells.append(oracle)
            oracle_ids.append(oracle["id"])

    draft_ids: list[str] = []
    for precision in DRAFT_PRECISIONS:
        draft = _cell(
            "P3",
            "autoregressive_draft_control",
            {
                "draft_model": "qwen_0_5b_control",
                "precision": precision,
                "workloads": list(WORKLOADS),
                "draft_lengths": list(range(2, 8)),
                "target_distribution": "served_tq_doctor_artifact",
                "tokenizer_identity_required": True,
            },
            depends_on=tuple(cost_by_runtime.values()),
            receipt_schema="hawking.spec_cost_oracle.v1",
            heavy=True,
        )
        cells.append(draft)
        draft_ids.append(draft["id"])

    parallel_ids: list[str] = []
    for precision in DRAFT_PRECISIONS:
        parallel = _cell(
            "P4",
            "parallel_draft_head",
            {
                "architecture": "p_eagle_style_parallel_mtp",
                "precision": precision,
                "positions": list(range(1, 8)),
                "workloads": list(WORKLOADS),
                "single_forward": True,
                "target_feature_taps": ["early", "middle", "last"],
                "train_on": "served_tq_doctor_distribution",
                "placeholder_tokens_forbidden": True,
            },
            depends_on=tuple(draft_ids),
            receipt_schema="hawking.spec_parallel_head.v1",
            heavy=True,
        )
        cells.append(parallel)
        parallel_ids.append(parallel["id"])

        diffusion = _cell(
            "P4",
            "block_diffusion_draft",
            {
                "architecture": "dflash_style_block_diffusion",
                "precision": precision,
                "positions": list(range(1, 9)),
                "workloads": list(WORKLOADS),
                "parallel_denoise_steps": [1, 2],
                "target_feature_conditioning": True,
                "train_on": "served_tq_doctor_distribution",
            },
            depends_on=tuple(draft_ids),
            receipt_schema="hawking.spec_learned_draft.v1",
            heavy=True,
        )
        cells.append(diffusion)
        parallel_ids.append(diffusion["id"])

        block_iterative = _cell(
            "P4",
            "block_iterative_draft",
            {
                "architecture": "specblock_style_path_dependent_blocks",
                "precision": precision,
                "block_widths": [2, 4],
                "max_blocks": [1, 2, 3],
                "workloads": list(WORKLOADS),
                "dynamic_rank_head": True,
                "train_on": "served_tq_doctor_distribution",
            },
            depends_on=tuple(draft_ids),
            receipt_schema="hawking.spec_learned_draft.v1",
            heavy=True,
        )
        cells.append(block_iterative)
        parallel_ids.append(block_iterative["id"])

    tree_ids: list[str] = []
    for width in (2, 4, 8):
        tree = _cell(
            "P5",
            "metal_tree_verify",
            {
                "tree_width": width,
                "max_nodes": 64,
                "ancestor_mask": "explicit",
                "exact_commit": "longest_argmax_confirmed_prefix",
                "cpu_fallback_is_not_performance_evidence": True,
            },
            depends_on=(parity_by_runtime["stored"], *parallel_ids),
            receipt_schema="hawking.spec_tree_verify.v1",
            heavy=True,
        )
        cells.append(tree)
        tree_ids.append(tree["id"])

    for runtime in RUNTIME_PATHS:
        cells.append(
            _cell(
                "P6",
                "target_draft_runtime_composition",
                {
                    "target_runtime_path": runtime,
                    "proposer_selection": "best_admitted_per_workload",
                    "workloads": list(WORKLOADS),
                    "metrics": [
                        "accepted_tokens_per_second_lcb",
                        "p50_p95_p99_latency",
                        "joules_per_accepted_token",
                        "bytes_per_accepted_token",
                        "rejected_work",
                        "peak_unified_memory_gib",
                    ],
                },
                depends_on=(
                    parity_by_runtime[runtime],
                    cost_by_runtime[runtime],
                    *oracle_ids,
                    *draft_ids,
                    *tree_ids,
                ),
                receipt_schema="hawking.spec_composition_gate.v1",
                heavy=True,
            )
        )

    return {
        "schema": SCHEMA,
        "receipt_contract_schema": spec_receipt_contract.CONTRACT_SCHEMA,
        "label": label,
        "execution_supported": False,
        "relationship_to_ladder": "additive_post_ladder_appendage",
        "schedule": {
            "do_not_stop_or_modify_active_ladder": True,
            "start_only_after_active_heavy_owner_exits": True,
            "exclusive_heavy_lease_required_for_gpu_cells": True,
            "checkpoint_each_cell_atomically": True,
            "gated_skip_state": "deferred",
        },
        "global_admission": {
            "exact_token_match": True,
            "minimum_speedup_lcb_per_workload": 1.10,
            "maximum_process_peak_gib": 78.0,
            "maximum_swap_delta_mb": 0.0,
            "maximum_memory_pressure": "normal",
            "artifact_binding": [
                "target_sha256",
                "draft_sha256",
                "parity_receipt_sha256",
                "source_commit",
                "prompt_set_sha256",
                "cell_id",
            ],
        },
        "blockers_before_execution": [
            "TQ-native batched verifier with strict ownership proof",
            "non-skipping parity harness and hash-bound receipt writer",
            "checkpointed runner with heavy-lease and pressure tripwires",
            "real parallel head (the current zero-token stub is inadmissible)",
            "Metal tree verifier before tree speed claims",
        ],
        "cells": cells,
    }


def _owners_from_ps_output(output: str, *, own_pid: int) -> list[dict]:
    owners = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, separator, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        command = command.strip() if separator else ""
        matches = sorted({
            pattern.pattern for pattern in HEAVY_PATTERNS
            if pattern.search(command.lower())
        })
        if pid != own_pid and command and matches:
            owners.append({"pid": pid, "command": command,
                           "matched_patterns": matches})
    return sorted(owners, key=lambda row: row["pid"])


def active_heavy_owners() -> list[dict]:
    """Read-only broad process snapshot that fails closed on probe errors."""
    try:
        process = subprocess.run(
            [str(PS_PATH), "-axo", "pid=,command="], capture_output=True,
            text=True, timeout=10, check=False,
        )
        if process.returncode != 0:
            raise subprocess.SubprocessError(
                f"ps exited with status {process.returncode}"
            )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        return [{
            "pid": 0, "command": "owner-probe-unavailable",
            "matched_patterns": [],
            "probe_error": f"{type(exc).__name__}: {exc}",
        }]
    return _owners_from_ps_output(process.stdout, own_pid=os.getpid())


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _selftest() -> int:
    first = build_matrix("TEST")
    second = build_matrix("TEST")
    assert first == second
    ids = [cell["id"] for cell in first["cells"]]
    assert len(ids) == len(set(ids))
    all_ids = set(ids)
    for cell in first["cells"]:
        assert set(cell["depends_on"]) <= all_ids
        assert cell["state"] == "deferred"
    assert not first["execution_supported"]
    assert {cell["phase"] for cell in first["cells"]} == {
        "P0", "P1", "P2", "P3", "P4", "P5", "P6"
    }
    print("spec_reentry_scaffold.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        return _selftest()
    if len(argv) == 2 and argv[0] == "--plan":
        print(json.dumps(build_matrix(argv[1]), indent=2, sort_keys=True))
        return 0
    if len(argv) == 3 and argv[0] == "--write-plan":
        _atomic_json(pathlib.Path(argv[2]), build_matrix(argv[1]))
        return 0
    if argv == ["--status"]:
        owners = active_heavy_owners()
        print(
            json.dumps(
                {
                    "schema": "hawking.spec_reentry_status.v1",
                    "execution_supported": False,
                    "active_heavy_owners": owners,
                    "safe_to_start_heavy_cells": not owners,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        "usage: spec_reentry_scaffold.py --plan LABEL | --write-plan LABEL PATH "
        "| --status | --selftest",
        file=sys.stderr,
    )
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
