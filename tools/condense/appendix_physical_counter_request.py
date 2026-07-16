#!/usr/bin/env python3.12
"""Default-off production builder for one Appendix counter execution request.

The builder removes handwritten JSON from the release path.  It accepts exact
parent-document files, ten signed authority envelopes, and typed device/spec
flags; hashes every workload file; loads the source-pinned authority registry;
then runs the executor's structural, release, corpus, workload, signature, live
host, final-ready, and owner-free validations before writing one immutable
request.  It never opens the heavy lease or starts a collector/probe.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Any, Mapping, Sequence

import appendix_contract
import appendix_physical_counter_authority as authority
import appendix_physical_counter_executor as executor
import physical_counter_attestation
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
OBSERVER = executor.OBSERVER
SCHEMA = "hawking.appendix_physical_counter_request_builder.v1"
STATUS_SCHEMA = "hawking.appendix_physical_counter_request_builder_status.v1"

PARENT_KEYS = (
    "boundary_attestation", "boundary_observation", "corpus_index",
    "corpus_verification", "corpus_prebuild_verification_receipt",
    "corpus_verification_receipt", "source_manifest", "release_build",
)


class RequestBuildError(ValueError):
    """A production request cannot be constructed without weakening a gate."""


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = appendix_contract.canonical_sha256(stamped)
    return stamped


def build_config() -> dict[str, Any]:
    capability = executor.execution_capability_contract()
    registry = authority.load_default_registry()
    return _stamp({
        "schema": SCHEMA,
        "default_off": True,
        "build_cli_exposed": True,
        "execution_capability": False,
        "opens_heavy_lease": False,
        "starts_collector_or_probe": False,
        "mutates_runtime_default": False,
        "request_schema": executor.REQUEST_SCHEMA,
        "execution_capability_sha256": capability["capability_sha256"],
        "authority_registry_sha256": registry["registry_sha256"],
        "required_release_parent_keys": list(PARENT_KEYS),
        "required_authority_keys": sorted(executor.AUTHORITY_SPECS),
        "commands": ["build-device", "build-spec"],
        "invariants": [
            "typed CLI flags construct every workload field",
            "all file identities are measured; no caller-supplied artifact hash",
            "current Doctor final-ready observer and zero owners are required",
            "all release/corpus/source/build/registry parents validate",
            "all ten SSHSIG envelopes verify against the source-pinned trust root",
            "authority host hashes equal the live IOPlatformUUID hash",
            "output request is immutable and grants no execution by itself",
        ],
    }, "config_sha256")


def status() -> dict[str, Any]:
    try:
        observer = authority._load_json(OBSERVER)
    except (OSError, authority.AuthorityError):
        observer = None
    owners = spec_reentry_scaffold.active_heavy_owners()
    authority_status = authority.status()
    blockers: list[str] = []
    if not isinstance(observer, dict) or observer.get("final_interpretation_ready") is not True:
        blockers.append("Doctor final_interpretation_ready is false")
    if owners:
        blockers.append(f"{len(owners)} heavy owner(s) remain")
    if authority_status["registry_valid"] is not True:
        blockers.append("source-pinned authority registry is invalid")
    if authority_status["signing_key_available"] is not True:
        blockers.append("operator signing key is unavailable or does not match the pinned root")
    return _stamp({
        "schema": STATUS_SCHEMA,
        "default_off": True,
        "request_build_ready": not blockers,
        "execution_requested": False,
        "final_interpretation_ready": (
            isinstance(observer, dict) and observer.get("final_interpretation_ready") is True
        ),
        "active_heavy_owner_count": len(owners),
        "authority_registry_valid": authority_status["registry_valid"],
        "signing_key_available": authority_status["signing_key_available"],
        "config_sha256": build_config()["config_sha256"],
        "blockers": blockers,
    }, "status_sha256")


def dry_run(kind: str) -> dict[str, Any]:
    if kind not in {"device", "spec"}:
        raise ValueError("kind must be device or spec")
    return {
        "schema": "hawking.appendix_physical_counter_request_builder_dry_run.v1",
        "kind": kind,
        "would_write_request": False,
        "would_open_heavy_lease": False,
        "would_start_collector_or_probe": False,
        "would_hash_named_workload_files": True,
        "would_validate_live_host_and_signed_authorities": True,
        "execution_capability": False,
        "config_sha256": build_config()["config_sha256"],
    }


def _identity(path: pathlib.Path) -> dict[str, Any]:
    return physical_counter_attestation.file_identity(path)


def _release(parents: Mapping[str, pathlib.Path]) -> dict[str, Any]:
    if set(parents) != set(PARENT_KEYS):
        raise RequestBuildError("release parent path set is incomplete or unexpected")
    value = {key: authority._load_json(parents[key]) for key in PARENT_KEYS}
    value["authority_registry"] = authority.load_default_registry()
    return value


def _authorities(paths: Mapping[str, pathlib.Path]) -> dict[str, Any]:
    if set(paths) != set(executor.AUTHORITY_SPECS):
        raise RequestBuildError("signed authority path set is incomplete or unexpected")
    return {key: authority._load_json(paths[key]) for key in sorted(paths)}


def build_request(
    *, kind: str, parents: Mapping[str, pathlib.Path],
    authority_paths: Mapping[str, pathlib.Path], workload: Mapping[str, Any],
    output_directory: pathlib.Path, scratch_reserve_gb: float,
    additional_parent_receipt_sha256: Sequence[str] = (),
    observer: Mapping[str, Any] | None = None,
    active_owners: Sequence[Mapping[str, Any]] | None = None,
    live_host_sha256: str | None = None,
    signature_verifier: executor.SignatureVerifier = executor._sshsig_verify,
    verify_files: bool = True,
) -> dict[str, Any]:
    if kind not in {"device", "spec"}:
        raise RequestBuildError("request kind must be device or spec")
    release = _release(parents)
    signed = _authorities(authority_paths)
    request = executor._stamp({
        "schema": executor.REQUEST_SCHEMA,
        "kind": kind,
        "release": release,
        "workload": dict(workload),
        "authorities": signed,
        "output_directory": str(output_directory.absolute()),
        "scratch_reserve_gb": scratch_reserve_gb,
        "additional_parent_receipt_sha256": list(additional_parent_receipt_sha256),
        "execution_requested": True,
        "runtime_default_mutation_requested": False,
    }, "request_sha256")
    errors = executor._validate_request_structure(request)
    current_observer = observer
    if current_observer is None:
        current_observer = authority._load_json(OBSERVER)
    errors.extend(executor._release_errors(
        release, observer=current_observer, verify_files=verify_files,
    ))
    errors.extend(executor._workload_errors(request, verify_files=verify_files))
    owners = list(active_owners) if active_owners is not None else spec_reentry_scaffold.active_heavy_owners()
    if owners:
        errors.append("heavy owners remain while constructing the physical request")
    if live_host_sha256 is None:
        live_host_sha256, detail = authority.live_host_hardware_uuid_sha256()
        if live_host_sha256 is None:
            errors.append(detail)
    authority_errors, receipts = executor.validate_authorities(
        signed, release_build_sha256=release["release_build"].get("receipt_sha256", ""),
        now_unix_ns=__import__("time").time_ns(), registry=release["authority_registry"],
        expected_host_hardware_uuid_sha256=live_host_sha256 or "",
        verify_files=verify_files, signature_verifier=signature_verifier,
    )
    errors.extend(authority_errors)
    for key in ("process_joule_capability", "process_joule_attribution"):
        if receipts.get(key, {}).get("binary") != workload.get("probe"):
            errors.append(f"{key} does not bind the exact selected self-sampling release probe")
    if errors:
        raise RequestBuildError("request construction failed: " + "; ".join(errors))
    return request


def device_workload(
    *, release_build: Mapping[str, Any], artifact: pathlib.Path, runtime_path: str,
    cell_id: str, tensor: str, residual_artifact: pathlib.Path | None = None,
    residual_tensor: str | None = None, warmups: int = 3, trials: int = 10,
    label: str = "CORPUS",
) -> dict[str, Any]:
    return {
        "runtime_path": runtime_path,
        "probe": release_build["probes"]["device"],
        "label": label,
        "cell_id": cell_id,
        "artifact": _identity(artifact),
        "tensor": tensor,
        "residual_artifact": _identity(residual_artifact) if residual_artifact is not None else None,
        "residual_tensor": residual_tensor,
        "warmups": warmups,
        "trials": trials,
    }


def spec_workload(
    *, release_build: Mapping[str, Any], weights: pathlib.Path, artifact: pathlib.Path,
    prompts: pathlib.Path, runtime_path: str, generated_tokens: int = 256,
    warmups_per_batch: int = 3, repeats_per_batch: int = 5,
    label: str = "CORPUS",
) -> dict[str, Any]:
    return {
        "runtime_path": runtime_path,
        "probe": release_build["probes"]["spec"],
        "label": label,
        "weights": _identity(weights),
        "artifact": _identity(artifact),
        "prompts": _identity(prompts),
        "generated_tokens": generated_tokens,
        "warmups_per_batch": warmups_per_batch,
        "repeats_per_batch": repeats_per_batch,
    }


def _parse_key_paths(rows: Sequence[str], *, expected: set[str], label: str) -> dict[str, pathlib.Path]:
    output: dict[str, pathlib.Path] = {}
    for row in rows:
        if "=" not in row:
            raise RequestBuildError(f"{label} must use KEY=PATH")
        key, raw = row.split("=", 1)
        if key in output:
            raise RequestBuildError(f"duplicate {label} key: {key}")
        output[key] = pathlib.Path(raw)
    if set(output) != expected:
        missing = sorted(expected - set(output))
        extra = sorted(set(output) - expected)
        raise RequestBuildError(f"{label} key set differs (missing={missing}, extra={extra})")
    return output


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    authority._atomic_json(path, value)


def _selftest() -> int:
    config = build_config()
    assert config["default_off"] is True and config["execution_capability"] is False
    assert dry_run("device")["would_start_collector_or_probe"] is False
    print("appendix_physical_counter_request.py selftest OK")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent", action="append", required=True, metavar="KEY=PATH")
    parser.add_argument("--authority", action="append", required=True, metavar="KEY=PATH")
    parser.add_argument("--physical-output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--scratch-reserve-gb", type=float, default=5.0)
    parser.add_argument("--additional-parent-receipt-sha256", action="append", default=[])
    parser.add_argument("--request-output", type=pathlib.Path, required=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    dry = sub.add_parser("dry-run")
    dry.add_argument("kind", choices=("device", "spec"))
    device = sub.add_parser("build-device")
    _common(device)
    device.add_argument("--artifact", type=pathlib.Path, required=True)
    device.add_argument("--runtime-path", choices=executor.collector.RUNTIME_PATHS, required=True)
    device.add_argument("--cell-id", required=True)
    device.add_argument("--tensor", required=True)
    device.add_argument("--residual-artifact", type=pathlib.Path)
    device.add_argument("--residual-tensor")
    device.add_argument("--warmups", type=int, default=3)
    device.add_argument("--trials", type=int, default=10)
    device.add_argument("--label", default="CORPUS")
    spec = sub.add_parser("build-spec")
    _common(spec)
    spec.add_argument("--weights", type=pathlib.Path, required=True)
    spec.add_argument("--artifact", type=pathlib.Path, required=True)
    spec.add_argument("--prompts", type=pathlib.Path, required=True)
    spec.add_argument("--runtime-path", choices=executor.collector.RUNTIME_PATHS, required=True)
    spec.add_argument("--generated-tokens", type=int, default=256)
    spec.add_argument("--warmups-per-batch", type=int, default=3)
    spec.add_argument("--repeats-per-batch", type=int, default=5)
    spec.add_argument("--label", default="CORPUS")
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.command == "dry-run":
        print(json.dumps(dry_run(args.kind), indent=2, sort_keys=True))
        return 0
    if args.command == "selftest":
        return _selftest()
    try:
        parents = _parse_key_paths(args.parent, expected=set(PARENT_KEYS), label="parent")
        authorities = _parse_key_paths(
            args.authority, expected=set(executor.AUTHORITY_SPECS), label="authority",
        )
        release_build = authority._load_json(parents["release_build"])
        if args.command == "build-device":
            workload = device_workload(
                release_build=release_build, artifact=args.artifact,
                runtime_path=args.runtime_path, cell_id=args.cell_id, tensor=args.tensor,
                residual_artifact=args.residual_artifact, residual_tensor=args.residual_tensor,
                warmups=args.warmups, trials=args.trials, label=args.label,
            )
            kind = "device"
        else:
            workload = spec_workload(
                release_build=release_build, weights=args.weights, artifact=args.artifact,
                prompts=args.prompts, runtime_path=args.runtime_path,
                generated_tokens=args.generated_tokens,
                warmups_per_batch=args.warmups_per_batch,
                repeats_per_batch=args.repeats_per_batch, label=args.label,
            )
            kind = "spec"
        request = build_request(
            kind=kind, parents=parents, authority_paths=authorities, workload=workload,
            output_directory=args.physical_output_directory,
            scratch_reserve_gb=args.scratch_reserve_gb,
            additional_parent_receipt_sha256=args.additional_parent_receipt_sha256,
        )
        _atomic_json(args.request_output, request)
    except (RequestBuildError, authority.AuthorityError, OSError, ValueError) as exc:
        print(f"appendix physical request build blocked: {exc}", file=sys.stderr)
        return 75
    print(json.dumps(request, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
