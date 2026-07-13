#!/usr/bin/env python3.12
"""Fail-closed readiness gate for speculative decoding on Hawking ``.tq`` artifacts.

This file deliberately does *not* launch training or inference.  The former
``spec_revive.py`` wired together commands with incompatible CLIs and could
write ``"spec lane complete"`` after failed phases.  That is unsafe for a
detached queue, especially because Qwen's single-token ``.tq`` path and its
batched verifier do not yet execute the same weight format.

The only supported operations are therefore:

``spec_revive.py --status MODEL.tq LABEL``
    Validate the target, a TQ batched-verifier parity receipt, and a cost-aware
    proposal-oracle receipt.  Write an atomic readiness receipt.  Status is
    observational and exits zero even when blocked.

``spec_revive.py MODEL.tq LABEL``
    Apply the same validation as an execution request.  It exits non-zero while
    any evidence is missing and also remains blocked until a real experiment
    runner replaces ``RUNNER_IMPLEMENTED = False`` below.

``spec_revive.py --plan LABEL``
    Print the evidence contract without doing work.

``spec_revive.py --selftest``
    Exercise valid and invalid synthetic receipts without model/GPU work.

No result from this tool changes ``reports/condense/spec_oracle_gate.json`` or
enables P3.  That operator-controlled gate remains separate.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)

READINESS_SCHEMA = "hawking.spec_readiness.v2"
TQ_PARITY_SCHEMA = "hawking.spec_tq_batched_parity.v1"
ORACLE_SCHEMA = "hawking.spec_cost_oracle.v1"
DEVICE_PROFILE = "Studio-M3Ultra-96"
MIN_PARITY_PROMPTS = 20
MIN_PARITY_TOKENS = 256
MIN_ORACLE_PROMPTS = 50
MIN_ORACLE_TOKENS = 8192
MIN_WORKLOAD_PROMPTS = 10
MIN_WORKLOAD_TOKENS = 1024
MIN_SPEEDUP_LCB = 1.10
PROCESS_BUDGET_GIB = 78.0
REQUIRED_WORKLOADS = {"code", "tool_json", "prose"}
REQUIRED_BATCHES = set(range(1, 9))

# There is no valid condensed-distribution capture -> head training -> dual
# runtime experiment runner today.  Turning this on requires implementing that
# runner and adding an end-to-end self-test; evidence receipts alone must never
# cause dormant scaffold code to execute.
RUNNER_IMPLEMENTED = False


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json(path: pathlib.Path) -> tuple[dict, str | None]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
        if not isinstance(value, dict):
            return {}, "JSON root is not an object"
        return value, None
    except FileNotFoundError:
        return {}, "receipt is absent"
    except Exception as exc:
        return {}, f"receipt is unreadable: {type(exc).__name__}: {exc}"


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _receipt_path(label: str, kind: str, root: pathlib.Path) -> pathlib.Path:
    env_name = {
        "parity": "HAWKING_SPEC_TQ_PARITY_RECEIPT",
        "oracle": "HAWKING_SPEC_COST_ORACLE_RECEIPT",
    }[kind]
    override = os.environ.get(env_name)
    if override:
        path = pathlib.Path(override)
        return path if path.is_absolute() else root / path
    suffix = "tq_batched_verify_parity" if kind == "parity" else "spec_cost_oracle"
    return root / "reports" / "condense" / f"{label}_{suffix}.json"


def _check_target(path: pathlib.Path) -> tuple[dict, list[str]]:
    blockers: list[str] = []
    if not path.exists():
        blockers.append("target artifact is absent")
    elif not path.is_file():
        blockers.append("target must be a durable file, not a model directory")
    if path.suffix.lower() != ".tq":
        blockers.append("target must be a Hawking .tq artifact")

    sha = None
    size = None
    if not blockers:
        size = path.stat().st_size
        if size <= 0:
            blockers.append("target artifact is empty")
        else:
            sha = _sha256(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": size,
        "sha256": sha,
        "ok": not blockers,
    }, blockers


def _check_parity(path: pathlib.Path, target_sha: str | None) -> tuple[dict, list[str]]:
    doc, read_error = _read_json(path)
    blockers: list[str] = []
    if read_error:
        blockers.append(read_error)
    if doc.get("schema") != TQ_PARITY_SCHEMA:
        blockers.append(f"schema must be {TQ_PARITY_SCHEMA}")
    if doc.get("status") != "pass":
        blockers.append("status is not pass")
    if not target_sha or doc.get("target_tq_sha256") != target_sha:
        blockers.append("receipt is not bound to the current .tq artifact hash")
    if doc.get("device_profile") != DEVICE_PROFILE:
        blockers.append(f"device_profile must be {DEVICE_PROFILE}")
    if doc.get("distribution_reference") != "tq-single-token-greedy":
        blockers.append("reference distribution must be tq-single-token-greedy")
    if doc.get("batched_verifier_path") != "tq-batched":
        blockers.append("batched verifier must execute the TQ path")
    if doc.get("exact_token_match") is not True:
        blockers.append("exact_token_match must be true")
    prompts_executed = _nonnegative_int(doc.get("prompts_executed"))
    if prompts_executed is None or prompts_executed < MIN_PARITY_PROMPTS:
        blockers.append(f"fewer than {MIN_PARITY_PROMPTS} parity prompts executed")
    max_new_tokens = _nonnegative_int(doc.get("max_new_tokens"))
    if max_new_tokens is None or max_new_tokens < MIN_PARITY_TOKENS:
        blockers.append(f"parity depth is below {MIN_PARITY_TOKENS} tokens")
    skipped_cases = _nonnegative_int(doc.get("skipped_cases"))
    if skipped_cases != 0:
        blockers.append("skipped parity cases are not admissible")

    curve = doc.get("verify_cost_curve")
    curve = curve if isinstance(curve, list) else []
    measured_curve: dict[int, dict] = {}
    for row in curve:
        if not isinstance(row, dict):
            continue
        batch = _nonnegative_int(row.get("batch"))
        median = _finite_float(row.get("median_cost_target_forwards"))
        cost_ucb = _finite_float(row.get("cost_target_forwards_ucb"))
        trials = _nonnegative_int(row.get("trials"))
        if (
            batch not in REQUIRED_BATCHES
            or median is None
            or median <= 0
            or cost_ucb is None
            or cost_ucb < median
            or trials is None
            or trials < 5
        ):
            continue
        previous = measured_curve.get(batch)
        if previous is None or cost_ucb > previous["cost_target_forwards_ucb"]:
            measured_curve[batch] = {
                "batch": batch,
                "median_cost_target_forwards": median,
                "cost_target_forwards_ucb": cost_ucb,
                "trials": trials,
            }
    missing_batches = sorted(REQUIRED_BATCHES - set(measured_curve))
    if missing_batches:
        blockers.append(
            "verify-cost curve lacks finite median/UCB measurements with >=5 trials "
            f"for batches {missing_batches}"
        )
    if not isinstance(doc.get("git_commit"), str) or not doc.get("git_commit"):
        blockers.append("git_commit is absent")

    receipt_sha = _sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "sha256": receipt_sha,
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "measured_batches": sorted(measured_curve),
        "verify_cost_curve": [measured_curve[b] for b in sorted(measured_curve)],
        "prompts_executed": prompts_executed,
        "max_new_tokens": max_new_tokens,
        "skipped_cases": skipped_cases,
        "ok": not blockers,
        "blockers": blockers,
    }, blockers


def _validated_cost_row(row: dict, parity_cost_ucb: dict[int, float]) -> dict | None:
    """Validate and conservatively recompute one workload/proposal cost row."""
    workload = row.get("workload")
    proposal = row.get("proposal")
    k = _nonnegative_int(row.get("k"))
    useful = _finite_float(row.get("useful_tokens_lcb"))
    draft = _finite_float(row.get("draft_cost_target_forwards_ucb"))
    verify = _finite_float(row.get("verify_cost_target_forwards_ucb"))
    fixed = _finite_float(row.get("fixed_target_cost_forwards_ucb", 0.0))
    claimed = _finite_float(row.get("predicted_speedup_lcb"))
    if (
        workload not in REQUIRED_WORKLOADS
        or not isinstance(proposal, str)
        or not proposal.strip()
        or k not in REQUIRED_BATCHES
        or useful is None
        or not 0 < useful <= k + 1
        or draft is None
        or draft < 0
        or verify is None
        or verify <= 0
        or fixed is None
        or fixed < 0
        or claimed is None
        or claimed <= 0
        or k not in parity_cost_ucb
    ):
        return None

    # The proposal oracle may observe a slower verifier than the parity run,
    # but it may not price verification below the parity receipt's measured
    # upper confidence bound.
    effective_verify = max(verify, parity_cost_ucb[k])
    denom = draft + effective_verify + fixed
    speedup = useful / denom
    # A receipt may round down, never materially up.  A >1% optimistic
    # discrepancy is rejected because it could flip a marginal GO.
    if claimed > speedup * 1.01:
        return None
    return {
        "workload": workload,
        "k": k,
        "proposal": proposal,
        "claimed_speedup_lcb": claimed,
        "recomputed_speedup_lcb": round(speedup, 6),
        "effective_verify_cost_target_forwards_ucb": effective_verify,
    }


def _check_oracle(
    path: pathlib.Path,
    target_sha: str | None,
    parity_receipt_sha: str | None,
    parity_curve: list[dict],
) -> tuple[dict, list[str]]:
    doc, read_error = _read_json(path)
    blockers: list[str] = []
    if read_error:
        blockers.append(read_error)
    if doc.get("schema") != ORACLE_SCHEMA:
        blockers.append(f"schema must be {ORACLE_SCHEMA}")
    if doc.get("status") != "pass":
        blockers.append("status is not pass")
    if not target_sha or doc.get("target_tq_sha256") != target_sha:
        blockers.append("oracle is not bound to the current .tq artifact hash")
    if not parity_receipt_sha or doc.get("tq_parity_receipt_sha256") != parity_receipt_sha:
        blockers.append("oracle is not bound to the current TQ parity receipt")
    if doc.get("device_profile") != DEVICE_PROFILE:
        blockers.append(f"device_profile must be {DEVICE_PROFILE}")
    if doc.get("exact_match") is not True:
        blockers.append("oracle must use exact-match acceptance")
    prompt_count = _nonnegative_int(doc.get("prompt_count"))
    if prompt_count is None or prompt_count < MIN_ORACLE_PROMPTS:
        blockers.append(f"oracle has fewer than {MIN_ORACLE_PROMPTS} prompts")
    token_count = _nonnegative_int(doc.get("token_count"))
    if token_count is None or token_count < MIN_ORACLE_TOKENS:
        blockers.append(f"oracle has fewer than {MIN_ORACLE_TOKENS} scored tokens")
    raw_workloads = doc.get("workload_classes")
    workloads = (
        {w for w in raw_workloads if isinstance(w, str)}
        if isinstance(raw_workloads, list)
        else set()
    )
    missing_workloads = sorted(REQUIRED_WORKLOADS - workloads)
    if missing_workloads:
        blockers.append(f"oracle lacks workload classes {missing_workloads}")

    prompt_counts = doc.get("workload_prompt_counts")
    prompt_counts = prompt_counts if isinstance(prompt_counts, dict) else {}
    token_counts = doc.get("workload_token_counts")
    token_counts = token_counts if isinstance(token_counts, dict) else {}
    checked_prompt_counts: dict[str, int | None] = {}
    checked_token_counts: dict[str, int | None] = {}
    for workload in sorted(REQUIRED_WORKLOADS):
        count = _nonnegative_int(prompt_counts.get(workload))
        checked_prompt_counts[workload] = count
        if count is None or count < MIN_WORKLOAD_PROMPTS:
            blockers.append(
                f"{workload} has fewer than {MIN_WORKLOAD_PROMPTS} oracle prompts"
            )
        count = _nonnegative_int(token_counts.get(workload))
        checked_token_counts[workload] = count
        if count is None or count < MIN_WORKLOAD_TOKENS:
            blockers.append(
                f"{workload} has fewer than {MIN_WORKLOAD_TOKENS} scored tokens"
            )

    parity_cost_ucb = {
        int(row["batch"]): float(row["cost_target_forwards_ucb"])
        for row in parity_curve
        if isinstance(row, dict)
        and _nonnegative_int(row.get("batch")) in REQUIRED_BATCHES
        and _finite_float(row.get("cost_target_forwards_ucb")) is not None
    }

    rows = doc.get("rows")
    rows = rows if isinstance(rows, list) else []
    recomputed: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        checked = _validated_cost_row(row, parity_cost_ucb)
        if checked is not None:
            recomputed.append(checked)

    best_by_workload = {
        workload: max(
            (
                row["recomputed_speedup_lcb"]
                for row in recomputed
                if row["workload"] == workload
            ),
            default=0.0,
        )
        for workload in sorted(REQUIRED_WORKLOADS)
    }
    best_speedup = max(best_by_workload.values(), default=0.0)
    if not recomputed:
        blockers.append("oracle has no internally consistent cost row")
    for workload, speedup in best_by_workload.items():
        if speedup < MIN_SPEEDUP_LCB:
            blockers.append(
                f"{workload} best recomputed speedup LCB {speedup:.3f} "
                f"< {MIN_SPEEDUP_LCB:.2f}"
            )

    peak = _finite_float(doc.get("dual_residency_peak_gib"))
    if peak is None or peak <= 0:
        blockers.append("dual-residency peak is absent or non-finite")
    elif peak > PROCESS_BUDGET_GIB:
        blockers.append(
            f"dual-residency peak {peak:g} GiB exceeds {PROCESS_BUDGET_GIB:g} GiB budget"
        )
    swap = _finite_float(doc.get("swap_used_mb"))
    if swap != 0.0:
        blockers.append("oracle dual-residency run used swap")
    if doc.get("memory_pressure_max") != "normal":
        blockers.append("oracle dual-residency run did not remain at normal memory pressure")
    if not isinstance(doc.get("git_commit"), str) or not doc.get("git_commit"):
        blockers.append("git_commit is absent")

    return {
        "path": str(path),
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "prompt_count": prompt_count,
        "token_count": token_count,
        "workload_classes": sorted(workloads),
        "workload_prompt_counts": checked_prompt_counts,
        "workload_token_counts": checked_token_counts,
        "consistent_rows": recomputed,
        "best_recomputed_speedup_lcb_by_workload": best_by_workload,
        "best_recomputed_speedup_lcb": best_speedup,
        "dual_residency_peak_gib": peak,
        "swap_used_mb": swap,
        "ok": not blockers,
        "blockers": blockers,
    }, blockers


def evaluate_readiness(
    model: str | pathlib.Path,
    label: str,
    *,
    root: pathlib.Path = ROOT,
    parity_path: pathlib.Path | None = None,
    oracle_path: pathlib.Path | None = None,
    runner_implemented: bool = RUNNER_IMPLEMENTED,
    write_receipt: bool = True,
) -> dict:
    target_path = pathlib.Path(model)
    if not target_path.is_absolute():
        target_path = root / target_path
    parity_path = parity_path or _receipt_path(label, "parity", root)
    oracle_path = oracle_path or _receipt_path(label, "oracle", root)

    target, target_blockers = _check_target(target_path)
    parity, parity_blockers = _check_parity(parity_path, target.get("sha256"))
    oracle, oracle_blockers = _check_oracle(
        oracle_path,
        target.get("sha256"),
        parity.get("sha256"),
        parity.get("verify_cost_curve", []),
    )
    evidence_ready = target["ok"] and parity["ok"] and oracle["ok"]
    blockers = [f"target: {b}" for b in target_blockers]
    blockers += [f"tq-parity: {b}" for b in parity_blockers]
    blockers += [f"cost-oracle: {b}" for b in oracle_blockers]
    if not runner_implemented:
        blockers.append(
            "runner: no validated condensed capture/train/dual-runtime experiment runner exists"
        )

    receipt = {
        "schema": READINESS_SCHEMA,
        "generated_at": _now(),
        "label": label,
        "device_profile": DEVICE_PROFILE,
        "target": target,
        "tq_batched_parity": parity,
        "cost_oracle": oracle,
        "evidence_ready": evidence_ready,
        "runner_implemented": runner_implemented,
        "ready_for_execution": evidence_ready and runner_implemented,
        "blockers": blockers,
        "operator_gate_modified": False,
        "note": (
            "A one_pass_verifier boolean is not evidence. TQ single-token and batched paths "
            "must execute the same artifact, pass non-skipped 1..8 parity, and carry a measured "
            "cost curve before proposal economics are evaluated."
        ),
    }
    if write_receipt:
        out = root / "reports" / "condense" / f"{label}_spec_readiness.json"
        _atomic_json(out, receipt)
        receipt["receipt_path"] = str(out)
    return receipt


def _plan(label: str) -> int:
    print(f"# Speculative-decoding readiness plan: {label}\n")
    print("1. Materialize a durable Hawking .tq target/draft artifact; temporary audit candidates do not count.")
    print("2. Implement TQ ownership in the batched verifier. The current GGUF batched path is inadmissible.")
    print(
        f"3. Execute >= {MIN_PARITY_PROMPTS} prompts to >= {MIN_PARITY_TOKENS} tokens, "
        "zero skips, exact TQ single-token vs TQ-batched match for B=1..8."
    )
    print(
        "4. Measure the M3 Ultra verify-cost median/UCB curve, draft cost, useful-token LCB, "
        "peak RSS, pressure, and swap."
    )
    print(
        f"5. Admit a research run only if recomputed speedup LCB >= {MIN_SPEEDUP_LCB:.2f} "
        f"across {', '.join(sorted(REQUIRED_WORKLOADS))}; then require live p95/energy gates."
    )
    print("6. Build a real detached experiment runner with atomic per-cell receipts before setting RUNNER_IMPLEMENTED.")
    print("\nP3 remains default-blocked; this plan does not create or modify its operator gate.")
    return 0


def _selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="hawking-spec-readiness-") as td:
        root = pathlib.Path(td)
        target = root / "model.tq"
        target.write_bytes(b"synthetic-tq-artifact")
        target_sha = _sha256(target)
        parity_path = root / "parity.json"
        parity = {
            "schema": TQ_PARITY_SCHEMA,
            "status": "pass",
            "target_tq_sha256": target_sha,
            "device_profile": DEVICE_PROFILE,
            "distribution_reference": "tq-single-token-greedy",
            "batched_verifier_path": "tq-batched",
            "exact_token_match": True,
            "prompts_executed": MIN_PARITY_PROMPTS,
            "max_new_tokens": MIN_PARITY_TOKENS,
            "skipped_cases": 0,
            "git_commit": "selftest",
            "verify_cost_curve": [
                {
                    "batch": b,
                    "median_cost_target_forwards": 1.0 + b / 10,
                    "cost_target_forwards_ucb": 1.1 + b / 10,
                    "trials": 5,
                }
                for b in sorted(REQUIRED_BATCHES)
            ],
        }
        parity_path.write_text(json.dumps(parity), encoding="utf-8")
        oracle_path = root / "oracle.json"
        oracle = {
            "schema": ORACLE_SCHEMA,
            "status": "pass",
            "target_tq_sha256": target_sha,
            "tq_parity_receipt_sha256": _sha256(parity_path),
            "device_profile": DEVICE_PROFILE,
            "exact_match": True,
            "prompt_count": MIN_ORACLE_PROMPTS,
            "token_count": MIN_ORACLE_TOKENS,
            "workload_classes": sorted(REQUIRED_WORKLOADS),
            "workload_prompt_counts": {
                workload: MIN_WORKLOAD_PROMPTS for workload in REQUIRED_WORKLOADS
            },
            "workload_token_counts": {
                workload: MIN_WORKLOAD_TOKENS for workload in REQUIRED_WORKLOADS
            },
            "dual_residency_peak_gib": 12.0,
            "swap_used_mb": 0.0,
            "memory_pressure_max": "normal",
            "git_commit": "selftest",
            "rows": [
                {
                    "workload": workload,
                    "k": 4,
                    "proposal": "synthetic",
                    "useful_tokens_lcb": 4.0,
                    "draft_cost_target_forwards_ucb": 0.5,
                    "verify_cost_target_forwards_ucb": 2.5,
                    "fixed_target_cost_forwards_ucb": 0.0,
                    "predicted_speedup_lcb": 1.33,
                }
                for workload in sorted(REQUIRED_WORKLOADS)
            ],
        }
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")

        good = evaluate_readiness(
            target, "TEST", root=root, parity_path=parity_path, oracle_path=oracle_path,
            runner_implemented=True, write_receipt=False,
        )
        assert good["evidence_ready"] and good["ready_for_execution"]

        parity["skipped_cases"] = 1
        parity_path.write_text(json.dumps(parity), encoding="utf-8")
        skipped = evaluate_readiness(
            target, "TEST", root=root, parity_path=parity_path, oracle_path=oracle_path,
            runner_implemented=True, write_receipt=False,
        )
        assert not skipped["evidence_ready"], "skipped parity must fail closed"

        parity["skipped_cases"] = 0
        parity_path.write_text(json.dumps(parity), encoding="utf-8")
        oracle["tq_parity_receipt_sha256"] = _sha256(parity_path)
        oracle["rows"][0]["predicted_speedup_lcb"] = 9.0
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        optimistic = evaluate_readiness(
            target, "TEST", root=root, parity_path=parity_path, oracle_path=oracle_path,
            runner_implemented=True, write_receipt=False,
        )
        assert not optimistic["evidence_ready"], "optimistic cost arithmetic must fail closed"

        oracle["rows"][0]["predicted_speedup_lcb"] = 1.33
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        blocked_runner = evaluate_readiness(
            target, "TEST", root=root, parity_path=parity_path, oracle_path=oracle_path,
            runner_implemented=False, write_receipt=False,
        )
        assert blocked_runner["evidence_ready"] and not blocked_runner["ready_for_execution"]

        oracle["dual_residency_peak_gib"] = float("nan")
        oracle["rows"][0]["useful_tokens_lcb"] = {"malformed": True}
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        malformed = evaluate_readiness(
            target, "TEST", root=root, parity_path=parity_path, oracle_path=oracle_path,
            runner_implemented=True, write_receipt=False,
        )
        assert not malformed["evidence_ready"], "malformed/non-finite fields must fail closed"

    print("spec_revive.py selftest OK")
    return 0


def _print_status(receipt: dict) -> None:
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selftest":
        return _selftest()
    if argv and argv[0] == "--plan":
        return _plan(argv[1] if len(argv) > 1 else "MODEL")
    if argv and argv[0] == "--status":
        if len(argv) != 3:
            print("usage: spec_revive.py --status MODEL.tq LABEL", file=sys.stderr)
            return 64
        receipt = evaluate_readiness(argv[1], argv[2])
        _print_status(receipt)
        return 0
    if len(argv) == 2:
        receipt = evaluate_readiness(argv[0], argv[1])
        _print_status(receipt)
        if not receipt["ready_for_execution"]:
            print("[spec] HALT: readiness contract is blocked; no work launched", file=sys.stderr)
            return 2
        # Deliberately unreachable while RUNNER_IMPLEMENTED is false.  Keeping
        # an explicit halt here prevents a future flag flip from becoming a
        # silent no-op that a detached queue could mark complete.
        print("[spec] HALT: experiment runner entry point is not implemented", file=sys.stderr)
        return 2
    print(
        "usage: spec_revive.py --plan LABEL | --status MODEL.tq LABEL | "
        "MODEL.tq LABEL | --selftest",
        file=sys.stderr,
    )
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
