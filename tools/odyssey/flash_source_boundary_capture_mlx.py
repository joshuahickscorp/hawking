#!/usr/bin/env python3
"""Capture the sealed Flash layers-0..4/token-0 source boundary after admission.

The wrapper fails before output reservation or model load unless the existing
V2 greedy teacher passes owner-authorized or local-canonical admission.  MLX-VLM
remains an isolated reference adapter; the result is source payload evidence,
not production execution or a parity verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from tools.odyssey import flash_source_boundary_preflight as preflight  # noqa: E402


WORKER = Path(__file__).with_name("flash_source_boundary_mlx_worker.py")
SCHEMA = "hawking.flash.source_boundary_capture.v1"
STATUS = "CAPTURED_ADMITTED_SOURCE_LAYERS_0_THROUGH_4_TOKEN_0"
WORKER_SCHEMA = "hawking.flash.source_boundary_mlx_worker.v1"
WORKER_STATUS = "CAPTURED_EXTERNAL_SOURCE_LAYERS_0_THROUGH_4_TOKEN_0"


def _preflight(path: Path, reference_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = oracle._regular(path, "source boundary preflight")
    document = gate._json(path)
    seal = gate._verify_compact_seal(document, "source boundary preflight")
    candidate = document.get("reference_candidate")
    capture = document.get("capture_contract")
    if (
        document.get("schema") != preflight.SCHEMA
        or not isinstance(candidate, dict)
        or candidate.get("sha256") != reference_sha256
        or not isinstance(capture, dict)
        or capture.get("token_index") != 0
        or capture.get("token_id") != gate.PROMPT_IDS[0]
        or capture.get("source_checkpoint_mutation_allowed") is not False
        or capture.get("layer4_cache") != preflight._required_layer4_cache()
        or capture.get("layer0_trace_contract_version") != 5
        or capture.get("layer0_trace") != preflight._required_layer0_trace()
    ):
        raise ValueError("source boundary preflight does not bind the selected V2 reference and token-0 contract")
    seams = capture.get("seams")
    if seams != preflight._required_seams():
        raise ValueError("source boundary preflight seam declaration differs from the canonical contract")
    return document, {**oracle._artifact_binding(path), "seal_sha256": seal}


def _payload(binding: object, *, out_dir: Path, label: str, dtype: str) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} payload binding is absent")
    path = oracle._regular(Path(str(binding.get("path", ""))), label)
    if path.parent != out_dir:
        raise ValueError(f"{label} payload escaped the reserved output directory")
    expected_sha = gate._require_sha256(binding.get("sha256"), f"{label} sha256")
    elements = gate._require_positive_int(binding.get("elements"), f"{label} elements")
    byte_count = gate._require_positive_int(binding.get("bytes"), f"{label} bytes")
    bytes_per_element = 4
    if binding.get("dtype") != dtype or byte_count != elements * bytes_per_element:
        raise ValueError(f"{label} payload dtype/extent differs")
    if path.stat().st_size != byte_count or oracle._sha256(path) != expected_sha:
        raise ValueError(f"{label} payload bytes differ from its binding")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "dtype": dtype,
        "elements": elements,
        "bytes": byte_count,
    }


def _verify_worker_manifest(
    path: Path,
    *,
    out_dir: Path,
    model_root: Path,
    reference_sha256: str,
    authorization_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    path = oracle._regular(path, "source boundary worker manifest")
    document = gate._json(path)
    if (
        document.get("schema") != WORKER_SCHEMA
        or document.get("status") != WORKER_STATUS
        or document.get("root") != str(model_root)
        or document.get("token_index") != 0
        or document.get("token_id") != gate.PROMPT_IDS[0]
        or document.get("reference_contract_sha256") != reference_sha256
        or document.get("owner_authorization_sha256") != authorization_sha256
        or document.get("preflight_sha256") != preflight_sha256
        or document.get("layers_executed") != [0, 1, 2, 3, 4]
        or document.get("ple_layer_ids") != [1]
        or document.get("full_body_executed") is not False
        or document.get("lm_head_executed") is not False
        or document.get("layer0_trace_contract_version") != 5
    ):
        raise ValueError("source boundary worker manifest identity or execution scope differs")
    seams = document.get("seams")
    declarations = preflight._required_seams()
    if not isinstance(seams, list) or len(seams) != len(declarations):
        raise ValueError("source boundary worker did not emit every predeclared seam")
    verified_seams: list[dict[str, Any]] = []
    for declaration, record in zip(declarations, seams, strict=True):
        if not isinstance(record, dict) or any(
            record.get(field) != declaration[field] for field in ("ordinal", "layer", "stage", "shape")
        ):
            raise ValueError("source boundary worker seam order, name, layer, or shape differs")
        verified = dict(record)
        if declaration["ordinal"] == 11:
            ids = record.get("expert_ids")
            weights = record.get("weights")
            if (
                not isinstance(ids, list) or len(ids) != 10 or len(set(ids)) != 10
                or not isinstance(weights, list) or len(weights) != 10
            ):
                raise ValueError("source boundary layer-4 route is malformed")
            verified["ids_payload"] = _payload(
                record.get("ids_payload"), out_dir=out_dir, label="layer-4 route ids", dtype="I32_LE"
            )
            verified["weights_payload"] = _payload(
                record.get("weights_payload"), out_dir=out_dir, label="layer-4 route weights", dtype="F32_LE"
            )
        else:
            verified["payload"] = _payload(
                record.get("payload"), out_dir=out_dir,
                label=f"source seam {declaration['ordinal']}", dtype="F32_LE",
            )
        verified_seams.append(verified)
    trace_declarations = preflight._required_layer0_trace()
    layer0_trace = document.get("layer0_trace")
    if not isinstance(layer0_trace, list) or len(layer0_trace) != len(trace_declarations):
        raise ValueError("source boundary worker did not emit every predeclared layer-0 trace payload")
    verified_trace: list[dict[str, Any]] = []
    for declaration, record in zip(trace_declarations, layer0_trace, strict=True):
        if not isinstance(record, dict) or any(
            record.get(field) != declaration[field] for field in ("ordinal", "layer", "stage", "shape")
        ):
            raise ValueError("source boundary worker layer-0 trace order, name, layer, or shape differs")
        if not isinstance(record.get("original_dtype"), str) or not record["original_dtype"]:
            raise ValueError("source boundary worker layer-0 trace omitted the source runtime dtype")
        verified = dict(record)
        verified["payload"] = _payload(
            record.get("payload"),
            out_dir=out_dir,
            label=f"source layer-0 trace {declaration['ordinal']}",
            dtype="F32_LE",
        )
        if verified["payload"]["elements"] != preflight._shape_elements(declaration["shape"]):
            raise ValueError("source boundary worker layer-0 trace payload extent differs from its declared shape")
        verified_trace.append(verified)
    caches = document.get("layer4_cache")
    if not isinstance(caches, dict) or set(caches) != {"pre_attention", "post_attention"}:
        raise ValueError("source boundary worker omitted pre/post layer-4 cache payloads")
    cache_declarations = preflight._required_layer4_cache()
    verified_cache: dict[str, list[dict[str, Any]]] = {}
    for boundary in ("pre_attention", "post_attention"):
        records = caches[boundary]
        declarations = cache_declarations[boundary]
        if not isinstance(records, list) or len(records) != len(declarations):
            raise ValueError(f"source boundary worker omitted declared {boundary} cache arrays")
        verified_cache[boundary] = []
        for index, (record, declaration) in enumerate(zip(records, declarations, strict=True)):
            if (
                not isinstance(record, dict)
                or any(record.get(field) != declaration[field] for field in ("state_role", "shape", "initialization"))
            ):
                raise ValueError(f"source boundary {boundary} cache record is malformed")
            payload = _payload(
                record.get("payload"),
                out_dir=out_dir,
                label=f"layer-4 {boundary} cache {index}",
                dtype="F32_LE",
            )
            if payload["elements"] != preflight._shape_elements(declaration["shape"]):
                raise ValueError(f"source boundary {boundary} cache payload extent differs from its declared shape")
            verified_cache[boundary].append({
                **record,
                "payload": payload,
            })
    return {
        "manifest": oracle._artifact_binding(path),
        "layer0_trace_contract_version": document["layer0_trace_contract_version"],
        "seams": verified_seams,
        "layer0_trace": verified_trace,
        "layer4_cache": verified_cache,
        "execution_schedule": document.get("execution_schedule"),
        "elapsed_ns": document.get("elapsed_ns"),
        "max_resident_set_size_bytes": document.get("max_resident_set_size_bytes"),
        "expert_source_access": document.get("expert_source_access"),
        "ple_storage_manifest": document.get("ple_storage_manifest"),
    }


def capture(
    *,
    model_root: Path,
    reference: Path,
    owner_authorization: Path | None,
    preflight_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    model_root = model_root.expanduser().resolve(strict=True)
    reference = oracle._regular(reference, "admitted V2 reference")
    if owner_authorization is not None:
        owner_authorization = oracle._regular(owner_authorization, "owner authorization")
    reference_sha256 = oracle._sha256(reference)

    # This is the hard gate.  It precedes runtime probing, output reservation,
    # model construction, and every provider/GPU action.
    authority = gate.load_admitted_reference_for_boundary(
        reference,
        model_root,
        owner_authorization_path=owner_authorization,
    )
    authorization_sha256 = authority["authorization_binding_sha256"]
    _, preflight_binding = _preflight(preflight_path, reference_sha256)
    lane = gate._protected_native_lane()
    if not lane["clean"]:
        raise ValueError(f"protected native/provider lane is not clean: {lane['matches']}")
    runtime = oracle._runtime_lock()
    worker = oracle._regular(WORKER, "source boundary provider worker")
    out_dir = oracle._reserve_output(out_dir)
    implementation = oracle._write_implementation_manifest(out_dir, runtime)
    started = time.time_ns()
    progress = {
        "schema": SCHEMA,
        "status": "RUNNING_ADMITTED_SOURCE_BOUNDARY_CAPTURE",
        "source_reference": {
            "path": str(reference),
            "sha256": reference_sha256,
            "seal_sha256": authority["seal_sha256"],
            "authorization_binding_sha256": authorization_sha256,
            "admission_class": authority.get("admission_class"),
            "owner_authorization": authority["owner_authorization"],
        },
        "science_status": "UNEARNED",
        "science_unearned": True,
        "preflight": preflight_binding,
        "provider": {
            "runtime_lock": {"path": runtime["path"], "sha256": runtime["sha256"]},
            "implementation": implementation,
            "boundary_worker": oracle._artifact_binding(worker),
            "reused_forward_owner": oracle._artifact_binding(oracle.WORKER.resolve()),
        },
        "started_unix_ns": started,
        "claim_boundary": "in-flight bounded source capture; no source-parity verdict",
    }
    atomic_write_json(out_dir / "capture-progress.json", progress)
    command = [
        runtime["python_executable"], str(worker),
        "--root", str(model_root),
        "--out-dir", str(out_dir),
        "--token-id", str(gate.PROMPT_IDS[0]),
        "--reference-sha256", reference_sha256,
        "--authorization-sha256", authorization_sha256,
        "--preflight-sha256", preflight_binding["sha256"],
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        progress.update({
            "status": "FAILED_EXTERNAL_BOUNDARY_PROVIDER__NO_PARITY_RESULT",
            "failure": {
                "exit_code": exc.returncode,
                "stdout_tail": (exc.stdout or "")[-12_000:],
                "stderr_tail": (exc.stderr or "")[-12_000:],
                "scientific_disposition": "none",
            },
            "finished_unix_ns": time.time_ns(),
        })
        atomic_write_json(out_dir / "capture-progress.json", progress)
        raise RuntimeError("source boundary provider failed; terminal runtime evidence is preserved") from exc
    verified = _verify_worker_manifest(
        out_dir / "source-boundary-manifest.json",
        out_dir=out_dir,
        model_root=model_root,
        reference_sha256=reference_sha256,
        authorization_sha256=authorization_sha256,
        preflight_sha256=preflight_binding["sha256"],
    )
    final = {
        **progress,
        "status": STATUS,
        "source_payloads": verified,
        "finished_unix_ns": time.time_ns(),
        "elapsed_ns": time.time_ns() - started,
        "model_loaded": True,
        "full_body_executed": False,
        "native_body_executed": False,
        "parity_compared": False,
        "promotion_allowed": False,
        "admission_class": authority.get("admission_class"),
        "science_status": "UNEARNED",
        "science_unearned": True,
        "claim_boundary": (
            "Admitted (owner-authorized or local-canonical) external source boundary capture only. Native "
            "counterpart and ordered comparison remain required; science_status=UNEARNED; no complete inference, "
            "EBPW, TPS, capability, deployment, promotion, or Kimi retirement."
        ),
    }
    final["seal_sha256"] = gate._compact_utf8_sorted_seal(final)
    atomic_write_json(out_dir / "capture.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    reference = oracle._regular(args.reference_contract, "V2 reference candidate")
    reference_sha256 = oracle._sha256(reference)
    _, preflight_binding = _preflight(args.preflight, reference_sha256)
    lane = gate._protected_native_lane()
    if args.dry_run:
        print(json.dumps({
            "status": (
                "DRY_RUN_OWNER_AUTHORIZATION_SUPPLIED_NOT_CONSUMED"
                if args.owner_authorization is not None
                else "DRY_RUN_LOCAL_CANONICAL_ADMISSION_AVAILABLE_IF_PROVENANCE_COMPLETE"
            ),
            "reference_contract": {"path": str(reference), "sha256": reference_sha256},
            "preflight": preflight_binding,
            "owner_authorization_supplied": args.owner_authorization is not None,
            "protected_native_lane": lane,
            "out_dir": str(args.out_dir.expanduser().resolve()),
            "out_exists": args.out_dir.expanduser().resolve().exists(),
            "model_loaded": False,
            "claim_boundary": "dry preflight only; admission is verified only on explicit execution; science_unearned",
        }, indent=2, sort_keys=True))
        return 0
    if not args.execute:
        raise ValueError("source boundary capture requires explicit --execute")
    result = capture(
        model_root=args.root,
        reference=reference,
        owner_authorization=args.owner_authorization,
        preflight_path=args.preflight,
        out_dir=args.out_dir,
    )
    print(json.dumps({"status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
