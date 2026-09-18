#!/usr/bin/env python3
"""Run Flash's admitted source -> native -> ordered-compare control atomically.

The public launcher re-admits an owner-authorized or local-canonical teacher before entering the
existing Hawking GPU lane.  The leased child rechecks authority, executable
closure, quiet-process state, and all fresh output namespaces before loading
the isolated source adapter.  It then performs exactly one bounded source
capture, one matching native diagnostic, and the CPU-only ordered comparison.

This is a localization transaction, not a full replay or promotion path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawking.persist import atomic_write_json  # noqa: E402
from tools.odyssey import flash_external_greedy_reference_mlx as oracle  # noqa: E402
from tools.odyssey import flash_repeated_accepted_decode as gate  # noqa: E402
from tools.odyssey import flash_source_boundary_capture_mlx as source_capture  # noqa: E402
from tools.odyssey import flash_source_boundary_compare as comparison  # noqa: E402
from tools.odyssey import flash_source_boundary_native as native  # noqa: E402


SCHEMA = "hawking.flash.source_boundary_transaction.v1"
RUNNING_STATUS = "RUNNING_ADMITTED_SOURCE_NATIVE_BOUNDARY_TRANSACTION"
COMPLETE_STATUS = "COMPLETED_BOUNDED_SOURCE_NATIVE_ORDERED_COMPARISON__NO_PROMOTION"
FAILED_STATUS = "FAILED_BOUNDED_SOURCE_NATIVE_ORDERED_COMPARISON__EVIDENCE_PRESERVED"
LANE_NAME = "flash-source-boundary-token0"
LANE_SCRIPT = ROOT / "tools/gpu_lane_lock.sh"
# The predecessor closures remain sealed evidence.  The public defaults name
# the fresh release pair that binds both the full-pairwise boundary diagnostic
# and the attention-only HC causal localizer; otherwise a bare transaction can
# only fail after resolving stale executable identity.
CURRENT_RELEASE_TARGET = ROOT / "target/flash-task029-attention-only-trace-speed-20260914"
DEFAULT_CLOSURE = ROOT / (
    "receipts/future/HAWKING_FLASH_SOURCE_BOUNDARY_EXECUTABLE_CLOSURE_"
    "TASK029_ALL_HC_BF16_STAGES_TRACE_COMPARE_REBIND_20260914.json"
)
DEFAULT_NATIVE_BINARY = (
    CURRENT_RELEASE_TARGET / "release/examples/flash_stateful_complete_token_session"
)
DEFAULT_HAWKING_BINARY = CURRENT_RELEASE_TARGET / "release/hawking"


def _lane_path() -> Path:
    raw = Path(os.environ.get("HAWKING_GPU_LANE_LOCK", "/tmp/hawking-gpu-lane.lock")).expanduser()
    path = Path(os.path.abspath(raw))
    unsafe = {Path("/"), Path.home().resolve(), ROOT.resolve(), ROOT.parent.resolve()}
    if path in unsafe:
        raise ValueError(f"refusing unsafe Hawking GPU lane path: {path}")
    if path.is_symlink():
        raise ValueError(f"Hawking GPU lane path may not be a symlink: {path}")
    return path


def _artifact(path: Path, label: str) -> dict[str, Any]:
    resolved = oracle._regular(path, label)
    return {
        "path": str(resolved),
        "sha256": oracle._sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_declared_file(record: object, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} binding is malformed")
    raw_path = record.get("path")
    expected_sha = gate._require_sha256(record.get("sha256"), f"{label} sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is absent")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    observed = _artifact(path, label)
    if observed["sha256"] != expected_sha:
        raise ValueError(f"{label} bytes differ from the executable-closure receipt")
    return observed


def _verify_declared_paths(
    records: object,
    *,
    expected_paths: tuple[Path, ...],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise ValueError(f"{label} is absent or incomplete")
    verified: list[dict[str, Any]] = []
    for index, (record, expected_relative) in enumerate(zip(records, expected_paths, strict=True)):
        observed = _verify_declared_file(record, label=f"{label} {index}")
        expected = (ROOT / expected_relative).resolve()
        if observed["path"] != str(expected):
            raise ValueError(f"{label} path differs from the canonical transaction surface")
        verified.append(observed)
    return verified


def _verify_executable_closure(
    path: Path,
    *,
    native_binary: Path,
    hawking_binary: Path,
) -> dict[str, Any]:
    closure_path = oracle._regular(path, "source-boundary executable closure")
    document = gate._json(closure_path)
    seal = gate._verify_compact_seal(document, "source-boundary executable closure")
    native_record = document.get("native_diagnostic")
    hawking_record = document.get("hawking_metric_owner")
    if (
        document.get("schema") != "hawking.flash.source_boundary_executable_closure.v1"
        or not isinstance(native_record, dict)
        or native_record.get("executed") is not False
        or not isinstance(hawking_record, dict)
    ):
        raise ValueError("source-boundary executable closure has incompatible identity or scope")

    observed_native = _verify_declared_file(native_record, label="native diagnostic executable")
    observed_hawking = _verify_declared_file(hawking_record, label="hawking metric executable")
    if observed_native["path"] != str(oracle._regular(native_binary, "selected native diagnostic executable")):
        raise ValueError("selected native diagnostic differs from the closure receipt")
    if observed_hawking["path"] != str(oracle._regular(hawking_binary, "selected hawking metric executable")):
        raise ValueError("selected hawking metric executable differs from the closure receipt")
    if not os.access(observed_native["path"], os.X_OK) or not os.access(observed_hawking["path"], os.X_OK):
        raise ValueError("a source-boundary executable is not executable")

    native_sources = native_record.get("source_closure")
    hawking_sources = hawking_record.get("source")
    if not isinstance(native_sources, list) or not native_sources:
        raise ValueError("native source closure is absent")
    if not isinstance(hawking_sources, list) or not hawking_sources:
        raise ValueError("hawking metric source closure is absent")
    verified_native_sources = [
        _verify_declared_file(record, label=f"native source closure {index}")
        for index, record in enumerate(native_sources)
    ]
    verified_hawking_sources = [
        _verify_declared_file(record, label=f"hawking source closure {index}")
        for index, record in enumerate(hawking_sources)
    ]
    wrapper = _verify_declared_file(native_record.get("wrapper"), label="native capture wrapper")
    orchestrator = _verify_declared_file(
        hawking_record.get("orchestrator"), label="ordered comparison wrapper"
    )
    transaction_wrappers = _verify_declared_paths(
        document.get("transaction_wrappers"),
        expected_paths=(
            Path("tools/odyssey/flash_source_boundary_transaction.py"),
            Path("tools/odyssey/flash_source_boundary_capture_mlx.py"),
            Path("tools/odyssey/flash_source_boundary_mlx_worker.py"),
            Path("tools/odyssey/flash_external_greedy_reference_mlx.py"),
            Path("tools/odyssey/flash_external_greedy_reference_mlx_worker.py"),
            Path("tools/odyssey/flash_source_boundary_native.py"),
            Path("tools/odyssey/flash_source_boundary_compare.py"),
            Path("tools/odyssey/flash_source_boundary_preflight.py"),
        ),
        label="leased source-boundary transaction wrapper closure",
    )
    return {
        "receipt": {**_artifact(closure_path, "source-boundary executable closure"), "seal_sha256": seal},
        "native_executable": observed_native,
        "hawking_executable": observed_hawking,
        "native_sources": verified_native_sources,
        "hawking_sources": verified_hawking_sources,
        "native_wrapper": wrapper,
        "comparison_wrapper": orchestrator,
        "transaction_wrappers": transaction_wrappers,
    }


def _output_targets(
    *, source_out_dir: Path, native_out: Path, comparison_out: Path, transaction_out: Path
) -> list[Path]:
    source_dir = source_out_dir.expanduser().resolve()
    native_path = native_out.expanduser().resolve()
    return [
        source_dir,
        native_path,
        native._native_artifact_dir(native_path),
        native_path.with_name(f"{native_path.stem}.runner.json"),
        comparison_out.expanduser().resolve(),
        transaction_out.expanduser().resolve(),
    ]


def _require_fresh_disjoint_outputs(
    *, source_out_dir: Path, native_out: Path, comparison_out: Path, transaction_out: Path
) -> list[str]:
    targets = _output_targets(
        source_out_dir=source_out_dir,
        native_out=native_out,
        comparison_out=comparison_out,
        transaction_out=transaction_out,
    )
    if len(set(targets)) != len(targets):
        raise ValueError("source/native/comparison/transaction outputs collide")
    for index, left in enumerate(targets):
        for right in targets[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("source/native/comparison/transaction outputs must not nest")
    existing = [str(path) for path in targets if os.path.lexists(path)]
    if existing:
        raise ValueError("transaction outputs already exist; refusing reuse: " + ", ".join(existing))
    return [str(path) for path in targets]


def _lease_evidence() -> dict[str, Any]:
    lane = _lane_path()
    if not lane.is_dir():
        raise ValueError("leased execution requires the canonical Hawking GPU lane directory")
    owner_path = lane / "owner"
    pid_path = lane / "pid"
    try:
        owner = owner_path.read_text(encoding="utf-8").strip()
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("canonical Hawking GPU lane ownership is unreadable") from exc
    if owner != LANE_NAME or pid != os.getppid():
        raise ValueError("leased execution is not the direct child of the expected Hawking GPU lane owner")
    return {
        "protocol": "tools/gpu_lane_lock.sh",
        "path": str(lane),
        "owner": owner,
        "owner_pid": pid,
        "direct_child_pid": os.getpid(),
        "exclusive_during_transaction": True,
    }


def _write_progress(path: Path, document: dict[str, Any]) -> None:
    body = dict(document)
    body.pop("seal_sha256", None)
    body["seal_sha256"] = gate._compact_utf8_sorted_seal(body)
    atomic_write_json(path, body)


def _leased_execute(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic_attention_hc_source_bf16_norm = bool(
        getattr(args, "diagnostic_attention_hc_source_bf16_norm", False)
    )
    diagnostic_attention_hc_norm_full_pairwise = bool(
        getattr(args, "diagnostic_attention_hc_norm_full_pairwise", False)
    )
    requested_attention_hc_norm_threadgroup = getattr(
        args, "diagnostic_attention_hc_norm_threadgroup", None
    )
    diagnostic_attention_hc_norm_threadgroup = native._diagnostic_attention_hc_norm_threadgroup(
        diagnostic_attention_hc_source_bf16_norm,
        requested_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise,
    )
    lease = _lease_evidence()
    reference = oracle._regular(args.reference_contract, "admitted V2 reference")
    authorization = (
        oracle._regular(args.owner_authorization, "owner authorization")
        if args.owner_authorization is not None
        else None
    )
    model_root = args.root.expanduser().resolve(strict=True)
    # Re-admit inside the held lane before inspecting outputs or executables.
    authority = gate.load_admitted_reference_for_boundary(
        reference, model_root, owner_authorization_path=authorization
    )
    _, preflight_binding = source_capture._preflight(args.preflight, oracle._sha256(reference))
    outputs = _require_fresh_disjoint_outputs(
        source_out_dir=args.source_out_dir,
        native_out=args.native_out,
        comparison_out=args.comparison_out,
        transaction_out=args.transaction_out,
    )
    closure_before = _verify_executable_closure(
        args.executable_closure,
        native_binary=args.native_binary,
        hawking_binary=args.hawking_binary,
    )
    lane = gate._protected_native_lane()
    if not lane["clean"]:
        raise ValueError(f"protected native/provider lane is not clean inside lease: {lane['matches']}")

    transaction_out = args.transaction_out.expanduser().resolve()
    started = time.time_ns()
    progress: dict[str, Any] = {
        "schema": SCHEMA,
        "status": RUNNING_STATUS,
        "stage": "SOURCE_CAPTURE",
        "started_unix_ns": started,
        "source_reference": {
            "path": str(reference),
            "sha256": oracle._sha256(reference),
            "seal_sha256": authority["seal_sha256"],
            "authorization_binding_sha256": authority["authorization_binding_sha256"],
            "admission_class": authority.get("admission_class"),
            "owner_authorization": authority["owner_authorization"],
        },
        "science_status": "UNEARNED",
        "science_unearned": True,
        "preflight": preflight_binding,
        "lease": lease,
        "protected_lane_preflight": lane,
        "executable_closure_before": closure_before,
        "reserved_fresh_targets": outputs,
        "completed_stages": [],
        "diagnostic_attention_hc_source_bf16_norm": diagnostic_attention_hc_source_bf16_norm,
        "diagnostic_attention_hc_norm_threadgroup": diagnostic_attention_hc_norm_threadgroup,
        "diagnostic_attention_hc_norm_full_pairwise": diagnostic_attention_hc_norm_full_pairwise,
        "full_body_replay_allowed": False,
        "promotion_allowed": False,
        "claim_boundary": "in-flight bounded localization transaction; no scientific or promotion verdict",
    }
    _write_progress(transaction_out, progress)
    try:
        source_result = source_capture.capture(
            model_root=model_root,
            reference=reference,
            owner_authorization=authorization,
            preflight_path=args.preflight,
            out_dir=args.source_out_dir,
        )
        progress["completed_stages"].append({
            "stage": "SOURCE_CAPTURE",
            "status": source_result["status"],
            "receipt": _artifact(args.source_out_dir / "capture.json", "source boundary capture"),
        })
        progress["stage"] = "NATIVE_CAPTURE"
        _write_progress(transaction_out, progress)

        native_result = native.run(
            model_root=model_root,
            reference=reference,
            owner_authorization=authorization,
            preflight_path=args.preflight,
            source_capture_path=args.source_out_dir / "capture.json",
            native_binary=args.native_binary,
            out=args.native_out,
            diagnostic_attention_hc_source_bf16_norm=diagnostic_attention_hc_source_bf16_norm,
            diagnostic_attention_hc_norm_threadgroup=requested_attention_hc_norm_threadgroup,
            diagnostic_attention_hc_norm_full_pairwise=diagnostic_attention_hc_norm_full_pairwise,
        )
        progress["completed_stages"].append({
            "stage": "NATIVE_CAPTURE",
            "status": native_result["status"],
            "receipt": _artifact(args.native_out, "native boundary capture"),
            "runner": _artifact(
                args.native_out.with_name(f"{args.native_out.stem}.runner.json"),
                "native boundary runner",
            ),
        })
        progress["stage"] = "ORDERED_COMPARISON"
        _write_progress(transaction_out, progress)

        compare_result = comparison.compare(
            model_root=model_root,
            reference=reference,
            owner_authorization=authorization,
            preflight_path=args.preflight,
            source_capture_path=args.source_out_dir / "capture.json",
            native_capture_path=args.native_out,
            hawking_binary=args.hawking_binary,
            out=args.comparison_out,
            diagnostic_attention_hc_source_bf16_norm=diagnostic_attention_hc_source_bf16_norm,
            diagnostic_attention_hc_norm_threadgroup=requested_attention_hc_norm_threadgroup,
            diagnostic_attention_hc_norm_full_pairwise=diagnostic_attention_hc_norm_full_pairwise,
        )
        progress["completed_stages"].append({
            "stage": "ORDERED_COMPARISON",
            "status": compare_result["status"],
            "receipt": _artifact(args.comparison_out, "source/native ordered comparison"),
            "first_observed_difference": compare_result["first_observed_difference"],
        })
        closure_after = _verify_executable_closure(
            args.executable_closure,
            native_binary=args.native_binary,
            hawking_binary=args.hawking_binary,
        )
        if closure_after != closure_before:
            raise ValueError("source-boundary executable/source closure changed during transaction")
        progress.update({
            "status": COMPLETE_STATUS,
            "stage": "COMPLETE",
            "finished_unix_ns": time.time_ns(),
            "elapsed_ns": time.time_ns() - started,
            "executable_closure_after": closure_after,
            "first_observed_difference": compare_result["first_observed_difference"],
            "model_execution_scope": "source and native layers 0..4 at token 0 only",
            "full_body_executed": False,
            "native_ple_implemented": False,
            "promotion_allowed": False,
            "claim_boundary": (
                "Completed one admitted bounded source/native localization and ordered comparison. "
                "The native diagnostic injects source post-PLE state. This is not complete inference, source "
                "parity, NR, EBPW/TPS/capability, deployment, Pulsar promotion, or Kimi retirement."
            ),
        })
        _write_progress(transaction_out, progress)
        return progress
    except Exception as exc:
        progress.update({
            "status": FAILED_STATUS,
            "failed_stage": progress.get("stage"),
            "finished_unix_ns": time.time_ns(),
            "elapsed_ns": time.time_ns() - started,
            "failure": {"type": type(exc).__name__, "message": str(exc)[:4000]},
            "scientific_disposition": "none",
            "promotion_allowed": False,
            "claim_boundary": "transaction failed; completed stage evidence is preserved without a parity verdict",
        })
        _write_progress(transaction_out, progress)
        raise


def _command(args: argparse.Namespace) -> list[str]:
    diagnostic_attention_hc_source_bf16_norm = bool(
        getattr(args, "diagnostic_attention_hc_source_bf16_norm", False)
    )
    diagnostic_attention_hc_norm_full_pairwise = bool(
        getattr(args, "diagnostic_attention_hc_norm_full_pairwise", False)
    )
    diagnostic_attention_hc_norm_threadgroup = native._diagnostic_attention_hc_norm_threadgroup(
        diagnostic_attention_hc_source_bf16_norm,
        getattr(args, "diagnostic_attention_hc_norm_threadgroup", None),
        diagnostic_attention_hc_norm_full_pairwise,
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--leased-execute",
        "--root", str(args.root),
        "--reference-contract", str(args.reference_contract),
        *([
            "--owner-authorization", str(args.owner_authorization),
        ] if args.owner_authorization is not None else []),
        "--preflight", str(args.preflight),
        *([
            "--diagnostic-attention-hc-source-bf16-norm",
        ] if diagnostic_attention_hc_source_bf16_norm else []),
        *([
            "--diagnostic-attention-hc-norm-full-pairwise",
        ] if diagnostic_attention_hc_norm_full_pairwise else []),
        *([
            "--diagnostic-attention-hc-norm-threadgroup",
            str(diagnostic_attention_hc_norm_threadgroup),
        ] if diagnostic_attention_hc_norm_threadgroup is not None and not diagnostic_attention_hc_norm_full_pairwise else []),
        "--executable-closure", str(args.executable_closure),
        "--native-binary", str(args.native_binary),
        "--hawking-binary", str(args.hawking_binary),
        "--source-out-dir", str(args.source_out_dir),
        "--native-out", str(args.native_out),
        "--comparison-out", str(args.comparison_out),
        "--transaction-out", str(args.transaction_out),
    ]
    return ["bash", str(LANE_SCRIPT), LANE_NAME, *command]


def _record_release_observation(path: Path, *, completed: subprocess.CompletedProcess[str]) -> None:
    if not path.is_file():
        return
    document = gate._json(path)
    gate._verify_compact_seal(document, "source-boundary transaction")
    document.pop("seal_sha256", None)
    document["lease_release_observation"] = {
        "observed_after_lane_command_exit": True,
        "lane_path_absent": not os.path.lexists(_lane_path()),
        "lane_command_exit_code": completed.returncode,
    }
    _write_progress(path, document)


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    reference = oracle._regular(args.reference_contract, "admitted V2 reference")
    authorization = (
        oracle._regular(args.owner_authorization, "owner authorization")
        if args.owner_authorization is not None
        else None
    )
    model_root = args.root.expanduser().resolve(strict=True)
    # The public hard gate precedes lane acquisition, output inspection, and
    # executable inspection.  The leased child repeats it against fresh bytes.
    gate.load_admitted_reference_for_boundary(
        reference, model_root, owner_authorization_path=authorization
    )
    _, _ = source_capture._preflight(args.preflight, oracle._sha256(reference))
    _require_fresh_disjoint_outputs(
        source_out_dir=args.source_out_dir,
        native_out=args.native_out,
        comparison_out=args.comparison_out,
        transaction_out=args.transaction_out,
    )
    _verify_executable_closure(
        args.executable_closure,
        native_binary=args.native_binary,
        hawking_binary=args.hawking_binary,
    )
    _lane_path()  # Refuse an unsafe inherited lock target before the shell owns it.
    completed = subprocess.run(_command(args), cwd=ROOT, capture_output=True, text=True)
    _record_release_observation(args.transaction_out.expanduser().resolve(), completed=completed)
    if completed.returncode != 0:
        raise RuntimeError(
            "protected source-boundary transaction failed; inspect its preserved receipt: "
            f"{args.transaction_out.expanduser().resolve()}\n{completed.stderr[-4000:]}"
        )
    document = gate._json(args.transaction_out.expanduser().resolve(strict=True))
    gate._verify_compact_seal(document, "completed source-boundary transaction")
    if document.get("status") != COMPLETE_STATUS:
        raise ValueError("protected source-boundary transaction did not reach its complete status")
    return document


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    reference = oracle._regular(args.reference_contract, "V2 reference candidate")
    model_root = args.root.expanduser().resolve(strict=True)
    authorization = (
        oracle._regular(args.owner_authorization, "owner authorization")
        if args.owner_authorization is not None
        else None
    )
    admission: dict[str, Any] | None = None
    admission_error: str | None = None
    try:
        admission = gate.load_admitted_reference_for_boundary(
            reference,
            model_root,
            owner_authorization_path=authorization,
        )
    except (OSError, ValueError) as exc:
        admission_error = f"{type(exc).__name__}: {exc}"
    _, preflight_binding = source_capture._preflight(args.preflight, oracle._sha256(reference))
    targets = _output_targets(
        source_out_dir=args.source_out_dir,
        native_out=args.native_out,
        comparison_out=args.comparison_out,
        transaction_out=args.transaction_out,
    )
    try:
        closure = _verify_executable_closure(
            args.executable_closure,
            native_binary=args.native_binary,
            hawking_binary=args.hawking_binary,
        )
        closure_status = "EXACT_EXECUTABLE_CLOSURE_PRESENT"
    except (OSError, ValueError) as exc:
        closure = {"error": f"{type(exc).__name__}: {exc}"}
        closure_status = "EXECUTABLE_CLOSURE_REFUSED"
    owner_supplied = args.owner_authorization is not None
    status = (
        (
            "DRY_RUN_OWNER_AUTHORIZATION_SUPPLIED_NOT_CONSUMED"
            if owner_supplied
            else "DRY_RUN_LOCAL_CANONICAL_ADMISSION_READY_NOT_CONSUMED"
        )
        if admission is not None
        else "WITHHELD_ADMISSION_AUTHORITY"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "reference_contract": _artifact(reference, "V2 reference candidate"),
        "preflight": preflight_binding,
        "owner_authorization_supplied": owner_supplied,
        "owner_authorization_verified": False,
        "diagnostic_attention_hc_source_bf16_norm": bool(
            getattr(args, "diagnostic_attention_hc_source_bf16_norm", False)
        ),
        "diagnostic_attention_hc_norm_full_pairwise": bool(
            getattr(args, "diagnostic_attention_hc_norm_full_pairwise", False)
        ),
        "diagnostic_attention_hc_norm_threadgroup": native._diagnostic_attention_hc_norm_threadgroup(
            bool(getattr(args, "diagnostic_attention_hc_source_bf16_norm", False)),
            getattr(args, "diagnostic_attention_hc_norm_threadgroup", None),
            bool(getattr(args, "diagnostic_attention_hc_norm_full_pairwise", False)),
        ),
        "admission_class": None if admission is None else admission.get("admission_class"),
        "science_status": "UNEARNED",
        "science_unearned": True,
        "admission_error": admission_error,
        "lane": {
            "protocol": str(LANE_SCRIPT.relative_to(ROOT)),
            "path": str(_lane_path()),
            "currently_free": not os.path.lexists(_lane_path()),
            "acquired": False,
        },
        "protected_native_lane": gate._protected_native_lane(),
        "executable_closure_status": closure_status,
        "executable_closure": closure,
        "outputs_fresh": all(not os.path.lexists(path) for path in targets),
        "output_targets": [str(path) for path in targets],
        "model_or_metal_started": False,
        "full_body_replay_allowed": False,
        "promotion_allowed": False,
        "claim_boundary": (
            "dry transaction preflight only; admission/lease/output reservation/execution not consumed; "
            "science_status=UNEARNED"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--executable-closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--native-binary", type=Path, default=DEFAULT_NATIVE_BINARY)
    parser.add_argument("--hawking-binary", type=Path, default=DEFAULT_HAWKING_BINARY)
    parser.add_argument("--source-out-dir", type=Path, required=True)
    parser.add_argument("--native-out", type=Path, required=True)
    parser.add_argument("--comparison-out", type=Path, required=True)
    parser.add_argument("--transaction-out", type=Path, required=True)
    parser.add_argument("--diagnostic-attention-hc-source-bf16-norm", action="store_true")
    parser.add_argument("--diagnostic-attention-hc-norm-threadgroup", type=int, choices=(128, 256))
    parser.add_argument("--diagnostic-attention-hc-norm-full-pairwise", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--leased-execute", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.leased_execute:
        result = _leased_execute(args)
    elif args.dry_run:
        result = _dry_run(args)
    elif args.execute:
        result = _execute(args)
    else:
        raise ValueError("source-boundary transaction requires explicit --dry-run or --execute")
    print(json.dumps({
        "status": result["status"],
        "first_observed_difference": result.get("first_observed_difference"),
        "seal_sha256": result.get("seal_sha256"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
