#!/usr/bin/env python3.12
"""Fail-closed aggregate gate for Appendix physical evidence.

This module is deliberately validation-only.  It never opens a model, launches
Metal, collects counters, mutates a runtime default, or writes into the Doctor or
Appendix result trees.  A green result means that already-produced evidence is
complete, source/corpus/build bound, one-to-one with the frozen matrices, and
backed by immutable physical-counter attestations.  Missing evidence is a normal
red result; it is never filled with estimates or structural placeholders.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import statistics
import sys
from typing import Any

import appendix_contract
import appendix_corpus
import appendix_device_runner
import physical_counter_attestation
import spec_receipt_contract
import spec_reentry_scaffold
import spec_tq_runner
import tq_receipt_contract
import tq_runtime_matrix


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_physical_evidence_gate.v2"
RELEASE_BOUNDARY_SCHEMA = "hawking.appendix_release_boundary_attestation.v1"
CORPUS_VERIFICATION_SCHEMA = "hawking.appendix_corpus_verification_attestation.v2"
SOURCE_MANIFEST_SCHEMA = "hawking.appendix_critical_source_capsule.v3"
RELEASE_BUILD_SCHEMA = "hawking.appendix_release_build_receipt.v4"
BUILD_CONTEXT_SCHEMA = "hawking.appendix_release_build_context.v1"
DEVICE_COUNTER_SCHEMA = "hawking.tq_runtime_physical_counters.v2"
SPEC_COUNTER_SCHEMA = "hawking.spec_tq_physical_counters.v2"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
RUNTIME_PATHS = ("stored", "compact", "hashed", "computed")
CPU_ERROR_POLICY = {
    "name": "stored_gpu_bitwise_and_cpu_q12_bounded_v1",
    "max_abs_error": 1.0e-4,
    "max_rel_error": 1.0e-4,
}
DETERMINISTIC_BUILD_ENVIRONMENT_KEYS = frozenset({
    "CARGO_HOME", "CARGO_INCREMENTAL", "CARGO_NET_OFFLINE", "CARGO_TARGET_DIR",
    "CARGO_TERM_COLOR", "HAWKING_HEAVY_LEASE_FD", "HOME", "LANG", "LC_ALL",
    "PATH", "RUSTC", "RUSTUP_HOME", "SOURCE_DATE_EPOCH", "TERM", "TMPDIR",
    "TZ", "ZERO_AR_DATE",
})
BASE_REQUIRED_SOURCE_PATHS = {
    "Cargo.toml",
    "Cargo.lock",
    "crates/hawking/Cargo.toml",
    "crates/hawking-core/Cargo.toml",
    "crates/hawking-core/build.rs",
    "crates/hawking-core/src/metal/mod.rs",
    "crates/hawking-core/src/metal/physical_signpost.c",
    "crates/hawking/src/tq_device_probe.rs",
    "crates/hawking/src/tq_spec_probe.rs",
    "crates/hawking/src/process_joule.rs",
    "crates/hawking-core/shaders/strand_bitslice.metal",
    "crates/hawking-core/src/kernels/mod.rs",
    "crates/hawking-core/src/lib.rs",
    "crates/hawking-core/src/tq.rs",
    "crates/hawking-core/src/tq_gpu.rs",
    "crates/hawking-core/src/model/qwen_dense.rs",
    "tools/condense/appendix_device_runner.py",
    "tools/condense/appendix_postrun.py",
    "tools/condense/appendix_corpus.py",
    "tools/condense/appendix_physical_release_packet.py",
    "tools/condense/spec_tq_runner.py",
    "tools/condense/tq_receipt_contract.py",
    "tools/condense/tq_runtime_matrix.py",
    "tools/condense/tq_runtime_probe.py",
    "tools/condense/spec_receipt_contract.py",
    "tools/condense/physical_counter_attestation.py",
    "tools/condense/appendix_physical_counter_collector.py",
    "tools/condense/appendix_physical_counter_authority.py",
    "tools/condense/appendix_physical_counter_executor.py",
    "tools/condense/appendix_physical_counter_normalizer.py",
    "tools/condense/appendix_physical_counter_request.py",
    "tools/condense/appendix_process_joule_collector.py",
    "tools/condense/appendix_xctrace_export_adapter.py",
    "tools/condense/appendix_physical_evidence_gate.py",
    "tools/condense/appendix_physical_release_state.py",
    "docs/plans/appendix_counter_authority_allowed_signers",
    "docs/plans/appendix_counter_authority_registry.json",
}
DEVICE_COUNTER_DOMAINS = (
    "energy", "gpu_time", "physical_bytes", "occupancy", "bandwidth",
)
SPEC_COUNTER_DOMAINS = ("energy", "gpu_time", "physical_bytes")


def _python_dependency_closure(paths: set[str]) -> set[str]:
    """Return the transitive local import closure for in-scope Python sources."""
    output = set(paths)
    pending = sorted(path for path in paths if path.endswith(".py"))
    seen: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = ROOT / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise RuntimeError(f"cannot derive Python dependency closure for {relative}: {exc}") from exc
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
        for module in sorted(modules):
            candidate = f"tools/condense/{module}.py"
            if (ROOT / candidate).is_file() and candidate not in output:
                output.add(candidate)
                pending.append(candidate)
    return output


REQUIRED_SOURCE_PATHS = frozenset(_python_dependency_closure(BASE_REQUIRED_SOURCE_PATHS))


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def stamp(document: dict[str, Any]) -> dict[str, Any]:
    stamped = copy.deepcopy(document)
    stamped.pop("gate_sha256", None)
    stamped["gate_sha256"] = canonical_sha256(stamped)
    return stamped


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (value > 0 if positive else value >= 0)
    )


def _hash_errors(value: Any, *, field: str, hash_field: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{field} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(hash_field, None)
    if not _hex64(claimed) or claimed != canonical_sha256(unstamped):
        return [f"{field}.{hash_field} does not match canonical bytes"]
    return []


def _unwrap_operator_sealed_evidence(
    value: Any, *, kind: str, verify_counter_files: bool,
    validator: Any | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Verify one signed executor envelope before exposing its core evidence.

    The aggregate gate deliberately does not accept a caller-provided signature
    verifier.  ``validator`` exists only as a narrow unit-test seam for this
    helper; :func:`validate_gate` always uses the production SSHSIG verifier in
    ``appendix_physical_counter_executor``.
    """
    if validator is None:
        # Lazy import avoids the executor -> release packet -> aggregate gate
        # import cycle while keeping the production verification implementation
        # single-sourced.
        import appendix_physical_counter_executor as counter_executor

        def validator(row: Any) -> list[str]:
            return counter_executor.validate_sealed_evidence(
                row, verify_files=verify_counter_files,
            )

    try:
        errors = list(validator(value))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return None, [f"sealed {kind} evidence verifier failed closed: {exc}"]
    if not isinstance(value, dict):
        return None, [*errors, f"sealed {kind} evidence must be an object"]
    request = value.get("request")
    attestation = value.get("signed_result_attestation")
    core = value.get("core_evidence")
    if not isinstance(request, dict) or request.get("kind") != kind:
        errors.append(f"sealed evidence request kind must be {kind}")
    signed_kind = (
        attestation.get("attestation", {}).get("kind")
        if isinstance(attestation, dict) else None
    )
    if signed_kind != kind:
        errors.append(f"operator-signed result kind must be {kind}")
    if not isinstance(core, dict):
        errors.append(f"sealed {kind} core_evidence must be an object")
    # Never pass even a plausible core object onward when any envelope,
    # authority-chain, signer, or dynamic-binding check failed.
    return (core if not errors and isinstance(core, dict) else None), errors


def _unwrap_operator_sealed_list(
    value: Any, *, kind: str, verify_counter_files: bool,
    seen_seals: set[str], aggregate_parents: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list):
        return [], [f"{kind}_evidence must be a list of operator-signed sealed evidence"]
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, sealed in enumerate(value):
        prefix = f"{kind}_evidence[{index}]"
        seal_sha = sealed.get("sealed_evidence_sha256") if isinstance(sealed, dict) else None
        if not _hex64(seal_sha):
            errors.append(f"{prefix} sealed evidence hash is invalid")
        elif seal_sha in seen_seals:
            errors.append(f"{prefix} reuses a sealed evidence envelope")
        else:
            seen_seals.add(seal_sha)
        core, seal_errors = _unwrap_operator_sealed_evidence(
            sealed, kind=kind, verify_counter_files=verify_counter_files,
        )
        errors.extend(f"{prefix}: {error}" for error in seal_errors)
        if isinstance(sealed, dict) and aggregate_parents is not None:
            release = sealed.get("request", {}).get("release") \
                if isinstance(sealed.get("request"), dict) else None
            if not isinstance(release, dict):
                errors.append(f"{prefix}: sealed request release parents are absent")
            else:
                for release_field, expected in aggregate_parents.items():
                    if release.get(release_field) != expected:
                        errors.append(
                            f"{prefix}: sealed request {release_field} differs from the "
                            "aggregate packet parent"
                        )
        if core is not None:
            output.append(core)
    return output, errors


def requirements() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "default_off": True,
        "execution_capability": False,
        "required_runtime_paths": list(RUNTIME_PATHS),
        "required_device_matrix_scope": "every deferred cell",
        "required_spec_scope": "one P0 parity and one P1 curve receipt per runtime path",
        "cpu_error_policy": CPU_ERROR_POLICY,
        "required_counter_domains": {
            "device": list(DEVICE_COUNTER_DOMAINS),
            "spec": list(SPEC_COUNTER_DOMAINS),
        },
        "accepted_evidence_container": (
            "operator-signed SSHSIG sealed executor evidence only; raw core evidence is rejected"
        ),
        "invariants": [
            "Doctor final interpretation ready and owner-free release boundary attested",
            "one held-lease prepare-release transaction binds one boundary, source capsule, corpus index, pre-build verification, build, and post-build verification",
            "the complete corpus is retained with a truthful explicit negative/failure/partial census; zero is valid",
            "exact symlink-free critical-source capsule including transitive local Python dependencies is stable across the release build",
            "resolved Cargo/Rustc bytes, verbose versions, host target, unique target directory, exact compiler artifacts, and dep-info source closures are bound",
            "one raw bundle and one receipt per exact matrix cell; no cross-credit or reuse",
            "compact/hashed/computed receipts depend on a stored receipt with identical normalized base/residual projection work",
            "device parity is bit-exact against stored GPU and bounded against CPU Q12",
            "RHT-cols, OUTL, and a real independently bound two-pass residual-accumulate path are explicit",
            "physical numbers are backed by immutable counter attestations, not source labels",
            "spec B=1..8 uses warmed independent paired repeats and phase markers",
            "observed verifier curves are retained without a monotonicity transform",
            "no evidence cell or aggregate gate requests a runtime-default mutation",
        ],
    }


def _validate_release_boundary(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["release_boundary must be an object"]
    expected = {
        "schema", "final_interpretation_ready", "final_packet_sha256",
        "observer_state_sha256", "all_recorded_hashes_verified",
        "active_heavy_owner_count", "owner_snapshot_sha256",
        "ram_swap_guard_healthy", "observed_at_unix_ns", "attestation_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("release_boundary fields are incomplete or unexpected")
    if value.get("schema") != RELEASE_BOUNDARY_SCHEMA:
        errors.append(f"release_boundary.schema must be {RELEASE_BOUNDARY_SCHEMA}")
    if value.get("final_interpretation_ready") is not True:
        errors.append("Doctor final_interpretation_ready is not attested true")
    if value.get("all_recorded_hashes_verified") is not True:
        errors.append("Doctor final packet hashes are not attested verified")
    for field in ("final_packet_sha256", "observer_state_sha256", "owner_snapshot_sha256"):
        if not _hex64(value.get(field)):
            errors.append(f"release_boundary.{field} is invalid")
    if value.get("active_heavy_owner_count") != 0:
        errors.append("release boundary is not owner-free")
    if value.get("ram_swap_guard_healthy") is not True:
        errors.append("release boundary RAM/swap guard is not healthy")
    observed = value.get("observed_at_unix_ns")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        errors.append("release_boundary.observed_at_unix_ns is invalid")
    errors.extend(_hash_errors(value, field="release_boundary", hash_field="attestation_sha256"))
    return errors


def _validate_corpus_index(index: Any) -> list[str]:
    if not isinstance(index, dict):
        return ["corpus_index must be an object"]
    errors: list[str] = []
    if index.get("schema") != "hawking.appendix_corpus_index.v3":
        errors.append("corpus_index schema is invalid")
    if not isinstance(index.get("source_base_commit"), str) \
            or COMMIT.fullmatch(index["source_base_commit"]) is None:
        errors.append("corpus_index source_base_commit is invalid")
    if index.get("source_base_commit_role") != "repository-base-only-not-byte-authority":
        errors.append("corpus index overclaims base-commit byte authority")
    unstamped = copy.deepcopy(index)
    claimed = unstamped.pop("index_sha256", None)
    if not _hex64(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("corpus_index.index_sha256 mismatch")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("corpus index entries must be non-empty")
        return errors
    seen: set[str] = set()
    total = 0
    kinds: dict[str, int] = {}
    for row in entries:
        if not isinstance(row, dict) or set(row) != {
            "path", "kind", "size", "sha256", "semantics",
        }:
            errors.append("corpus index entry is malformed")
            continue
        path = row.get("path")
        if (
            not isinstance(path, str) or not path or path in seen
            or pathlib.PurePosixPath(path).is_absolute()
            or ".." in pathlib.PurePosixPath(path).parts
        ):
            errors.append("corpus index entry path is unsafe or duplicated")
            continue
        seen.add(path)
        if not _hex64(row.get("sha256")):
            errors.append(f"corpus entry {path} has an invalid hash")
        size = row.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"corpus entry {path} has an invalid size")
            continue
        total += size
        kind = row.get("kind")
        if not isinstance(kind, str) or not kind:
            errors.append(f"corpus entry {path} has an invalid kind")
        else:
            kinds[kind] = kinds.get(kind, 0) + 1
        semantics = row.get("semantics")
        if not isinstance(semantics, list) or semantics != sorted(set(semantics)) \
                or any(value not in appendix_corpus.SEMANTIC_CLASSES for value in semantics):
            errors.append(f"corpus entry {path} has invalid explicit semantics")
    if index.get("file_count") != len(entries):
        errors.append("corpus index file_count does not match entries")
    if index.get("total_bytes") != total:
        errors.append("corpus index total_bytes does not match entries")
    if index.get("kind_counts") != dict(sorted(kinds.items())):
        errors.append("corpus index kind_counts do not match entries")
    semantic_counts = {
        semantic: sum(
            isinstance(row, dict) and semantic in row.get("semantics", [])
            for row in entries
        )
        for semantic in appendix_corpus.SEMANTIC_CLASSES
    }
    if index.get("semantic_counts") != semantic_counts:
        errors.append("corpus index semantic_counts do not match entries")
    if index.get("contains_explicit_negative_failure_or_partial_evidence") \
            is not any(semantic_counts.values()):
        errors.append("corpus index negative/failure/partial summary is not truthful")
    if index.get("semantic_census_policy") != "explicit-filename-or-structured-status-v1":
        errors.append("corpus index semantic census policy is invalid")
    return errors


def _validate_corpus_verification(value: Any, *, index: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return ["corpus_verification must be an object"]
    expected = {
        "schema", "index_sha256", "verified_at_unix_ns",
        "active_heavy_owner_count", "file_count", "total_bytes",
        "changed_files", "missing_files", "added_files", "symlinks",
        "semantic_counts", "all_censused_semantics_verified",
        "verification_receipt_sha256",
        "attestation_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("corpus_verification fields are incomplete or unexpected")
    if value.get("schema") != CORPUS_VERIFICATION_SCHEMA:
        errors.append(f"corpus_verification.schema must be {CORPUS_VERIFICATION_SCHEMA}")
    if value.get("index_sha256") != index.get("index_sha256"):
        errors.append("corpus_verification is not bound to the corpus index")
    if value.get("active_heavy_owner_count") != 0:
        errors.append("corpus verification was not owner-free")
    if value.get("file_count") != index.get("file_count") or value.get("total_bytes") != index.get("total_bytes"):
        errors.append("corpus verification counts do not match the index")
    for field in ("changed_files", "missing_files", "added_files", "symlinks"):
        if value.get(field) != 0:
            errors.append(f"corpus_verification.{field} must be zero")
    if value.get("semantic_counts") != index.get("semantic_counts"):
        errors.append("corpus verification semantic counts differ from index")
    if value.get("all_censused_semantics_verified") is not True:
        errors.append("corpus verification did not verify the complete semantic census")
    if not _hex64(value.get("verification_receipt_sha256")):
        errors.append("corpus verification receipt hash is invalid")
    observed = value.get("verified_at_unix_ns")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        errors.append("corpus verification timestamp is invalid")
    errors.extend(_hash_errors(value, field="corpus_verification", hash_field="attestation_sha256"))
    return errors


def _corpus_artifact_binding_errors(
    binding: Any, *, corpus_index: dict[str, Any], label: str,
) -> list[str]:
    if not isinstance(binding, dict):
        return [f"{label} artifact binding is missing"]
    digest = binding.get("sha256")
    size = binding.get("size_bytes")
    entries = corpus_index.get("entries", [])
    exact = [
        row for row in entries
        if isinstance(row, dict)
        and row.get("sha256") == digest
        and row.get("size") == size
    ]
    if len(exact) != 1:
        return [
            f"{label} artifact is not bound exactly once by hash and size in the frozen corpus index"
        ]
    return []


def _validate_source_manifest(value: Any, *, verify_files: bool = False) -> list[str]:
    if not isinstance(value, dict):
        return ["source_manifest must be an object"]
    expected = {
        "schema", "source_base_commit", "source_base_commit_role", "scope",
        "release_boundary_attestation_sha256", "release_boundary_observation_sha256",
        "required_paths_sha256",
        "entry_count", "symlink_count", "entries", "capsule_sha256",
        "manifest_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("source_manifest fields are incomplete or unexpected")
    if value.get("schema") != SOURCE_MANIFEST_SCHEMA:
        errors.append(f"source_manifest.schema must be {SOURCE_MANIFEST_SCHEMA}")
    if not isinstance(value.get("source_base_commit"), str) \
            or not COMMIT.fullmatch(value["source_base_commit"]):
        errors.append("source_manifest.source_base_commit is invalid")
    if value.get("source_base_commit_role") != "repository-base-only-not-byte-authority":
        errors.append("source capsule overclaims base-commit byte authority")
    if value.get("scope") != "isolated-exact-critical-source-capsule":
        errors.append("source capsule scope is invalid")
    for field in (
        "release_boundary_attestation_sha256", "release_boundary_observation_sha256",
    ):
        if not _hex64(value.get(field)):
            errors.append(f"source capsule {field} is invalid")
    if value.get("required_paths_sha256") != canonical_sha256(sorted(REQUIRED_SOURCE_PATHS)):
        errors.append("source capsule required-path set is not the frozen gate set")
    if value.get("symlink_count") != 0:
        errors.append("source capsule contains or permits symlinks")
    entries = value.get("entries")
    paths: set[str] = set()
    if not isinstance(entries, list) or not entries:
        errors.append("source_manifest.entries must be non-empty")
    else:
        for row in entries:
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
                errors.append("source manifest entry is malformed")
                continue
            path = row.get("path")
            if not isinstance(path, str) or not path or path in paths:
                errors.append("source manifest path is empty or duplicated")
                continue
            paths.add(path)
            if not _hex64(row.get("sha256")):
                errors.append(f"source manifest hash is invalid for {path}")
            size = row.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                errors.append(f"source manifest size is invalid for {path}")
            elif verify_files:
                try:
                    observed = physical_counter_attestation.file_identity(ROOT / path)
                except (OSError, ValueError) as exc:
                    errors.append(f"source manifest file cannot be verified for {path}: {exc}")
                else:
                    if observed["sha256"] != row.get("sha256") or observed["size_bytes"] != size:
                        errors.append(f"source manifest file differs from bound identity: {path}")
        missing = sorted(REQUIRED_SOURCE_PATHS - paths)
        extra = sorted(paths - REQUIRED_SOURCE_PATHS)
        if missing or extra:
            errors.append(
                "source capsule must contain exactly the frozen critical sources"
                f" (missing={missing}, extra={extra})"
            )
        if value.get("entry_count") != len(entries):
            errors.append("source capsule entry_count does not match entries")
        if value.get("capsule_sha256") != canonical_sha256(entries):
            errors.append("source capsule byte-set hash does not match entries")
    errors.extend(_hash_errors(value, field="source_manifest", hash_field="manifest_sha256"))
    return errors


def _file_binding_errors(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        return [f"{label} must contain exactly path/sha256/size_bytes"]
    errors: list[str] = []
    if not isinstance(value.get("path"), str) or not pathlib.Path(value["path"]).is_absolute():
        errors.append(f"{label}.path must be absolute")
    if not _hex64(value.get("sha256")):
        errors.append(f"{label}.sha256 is invalid")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        errors.append(f"{label}.size_bytes must be positive")
    return errors


def _toolchain_binding_errors(value: Any, *, label: str, verify_files: bool) -> list[str]:
    expected = {
        "invocation_path", "resolved_binary", "version_verbose",
        "version_verbose_sha256", "selection",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return [f"{label} toolchain binding is malformed"]
    errors: list[str] = []
    invocation = value.get("invocation_path")
    if not isinstance(invocation, str) or not pathlib.Path(invocation).is_absolute():
        errors.append(f"{label}.invocation_path must be absolute")
    errors.extend(_file_binding_errors(value.get("resolved_binary"), label=f"{label}.resolved_binary"))
    version = value.get("version_verbose")
    if not isinstance(version, str) or not version.strip() \
            or value.get("version_verbose_sha256") != canonical_sha256(version):
        errors.append(f"{label} verbose version binding is invalid")
    selection = value.get("selection")
    selection_expected = {
        "mode", "discovered_invocation_path", "discovered_resolved_binary",
        "selection_environment", "selector_argv_sha256", "selected_invocation_path",
        "version_probe_environment", "version_probe_argv_sha256",
    }
    if not isinstance(selection, dict) or set(selection) != selection_expected:
        errors.append(f"{label} toolchain selection binding is malformed")
        selection = {}
    mode = selection.get("mode")
    if mode not in {"direct", "rustup-which-to-direct-binary"}:
        errors.append(f"{label} toolchain selection mode is invalid")
    discovered = selection.get("discovered_invocation_path")
    if not isinstance(discovered, str) or not pathlib.Path(discovered).is_absolute():
        errors.append(f"{label} discovered invocation path is invalid")
    errors.extend(_file_binding_errors(
        selection.get("discovered_resolved_binary"),
        label=f"{label}.selection.discovered_resolved_binary",
    ))
    selection_environment = selection.get("selection_environment")
    expected_selection_keys = {"CARGO_HOME", "HOME", "LANG", "LC_ALL", "PATH", "RUSTUP_HOME", "TZ"}
    if not isinstance(selection_environment, dict) \
            or set(selection_environment) != expected_selection_keys \
            or selection_environment.get("LANG") != "C" \
            or selection_environment.get("LC_ALL") != "C" \
            or selection_environment.get("PATH") != "/usr/bin:/bin" \
            or selection_environment.get("TZ") != "UTC" \
            or any(
                not isinstance(selection_environment.get(name), str)
                or not pathlib.Path(selection_environment[name]).is_absolute()
                for name in ("CARGO_HOME", "HOME", "RUSTUP_HOME")
            ):
        errors.append(f"{label} toolchain selection environment is not exact/minimal")
    selector_sha = selection.get("selector_argv_sha256")
    if mode == "direct" and selector_sha is not None:
        errors.append(f"{label} direct tool selection unexpectedly has selector argv")
    if mode == "rustup-which-to-direct-binary" and not _hex64(selector_sha):
        errors.append(f"{label} rustup selector argv binding is invalid")
    if selection.get("selected_invocation_path") != invocation \
            or invocation != value.get("resolved_binary", {}).get("path"):
        errors.append(f"{label} build invocation is not the selected direct binary")
    version_environment = selection.get("version_probe_environment")
    if version_environment != {
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin", "TZ": "UTC",
    } or not _hex64(selection.get("version_probe_argv_sha256")):
        errors.append(f"{label} version probe environment/argv binding is invalid")
    if verify_files and isinstance(value.get("resolved_binary"), dict):
        binding = value["resolved_binary"]
        try:
            observed = physical_counter_attestation.file_identity(pathlib.Path(binding["path"]))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{label} resolved tool cannot be verified: {exc}")
        else:
            if observed != binding:
                errors.append(f"{label} resolved tool differs from bound bytes")
        if isinstance(invocation, str):
            try:
                resolved_invocation = pathlib.Path(invocation).resolve(strict=True)
            except OSError as exc:
                errors.append(f"{label} invocation path cannot be resolved: {exc}")
            else:
                if str(resolved_invocation) != binding.get("path"):
                    errors.append(f"{label} invocation path no longer resolves to bound tool")
        discovered_binding = selection.get("discovered_resolved_binary")
        if isinstance(discovered, str) and isinstance(discovered_binding, dict):
            try:
                resolved_discovered = pathlib.Path(discovered).resolve(strict=True)
                observed_discovered = physical_counter_attestation.file_identity(resolved_discovered)
            except (OSError, ValueError) as exc:
                errors.append(f"{label} selected-tool discovery path cannot be verified: {exc}")
            else:
                if observed_discovered != discovered_binding:
                    errors.append(f"{label} selected-tool discovery bytes/path changed")
    return errors


def _directory_identity(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"directory is a symlink: {path}")
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"not a directory: {path}")
    return {
        "path": str(path.resolve(strict=True)), "device": value.st_dev,
        "inode": value.st_ino, "mode": stat.S_IMODE(value.st_mode),
    }


def _live_file_identity(path: pathlib.Path) -> dict[str, Any]:
    lexical = pathlib.Path(os.path.abspath(path))
    descriptor = os.open(lexical, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"not a nonempty regular file: {lexical}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(lexical, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise ValueError(f"file changed/replaced while hashing: {lexical}")
    return {"path": str(lexical), "sha256": digest.hexdigest(), "size_bytes": before.st_size}


def _live_path_snapshot(value: str) -> dict[str, Any]:
    directories: list[dict[str, Any]] = []
    for raw in value.split(os.pathsep):
        path = pathlib.Path(raw)
        entries: list[dict[str, Any]] = []
        for entry in sorted(path.iterdir(), key=lambda candidate: candidate.name):
            if not entry.is_file() or not os.access(entry, os.X_OK):
                continue
            entries.append({
                "name": entry.name,
                "link_target": os.readlink(entry) if entry.is_symlink() else None,
                "resolved_binary": _live_file_identity(entry.resolve(strict=True)),
            })
        directories.append({
            **_directory_identity(path),
            "executable_entries": entries,
            "executable_entries_sha256": canonical_sha256(entries),
        })
    return {
        "value": value, "directories": directories,
        "snapshot_sha256": canonical_sha256(directories),
    }


def _configuration_candidates(cargo_home: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    rows: list[tuple[str, pathlib.Path]] = []
    current = ROOT.resolve(strict=True)
    while True:
        rows.extend((
            ("cargo", current / ".cargo" / "config.toml"),
            ("cargo", current / ".cargo" / "config"),
            ("rustup", current / "rust-toolchain.toml"),
            ("rustup", current / "rust-toolchain"),
        ))
        if current.parent == current:
            break
        current = current.parent
    rows.extend((("cargo", cargo_home / "config.toml"), ("cargo", cargo_home / "config")))
    deduplicated = {
        str(pathlib.Path(os.path.abspath(path))): (kind, pathlib.Path(os.path.abspath(path)))
        for kind, path in rows
    }
    return [deduplicated[key] for key in sorted(deduplicated)]


def _live_configuration_snapshot(cargo_home: pathlib.Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for kind, path in _configuration_candidates(cargo_home):
        if path.is_symlink():
            raise ValueError(f"configuration candidate is a symlink: {path}")
        binding = _live_file_identity(path) if path.exists() else None
        entries.append({"kind": kind, "path": str(path), "binding": binding})
    return {"entries": entries, "snapshot_sha256": canonical_sha256(entries)}


def _build_context_errors(
    context: Any, *, environment: Any, toolchain: dict[str, Any], verify_files: bool,
) -> list[str]:
    expected = {
        "schema", "environment_keys", "environment_sha256", "path_snapshot",
        "configuration_snapshot", "cargo_home", "isolated_directories",
        "toolchain_selection_sha256", "ambient_environment_inherited", "context_sha256",
    }
    if not isinstance(context, dict) or set(context) != expected:
        return ["release build execution context is malformed"]
    errors = _hash_errors(context, field="release_build_context", hash_field="context_sha256")
    if context.get("schema") != BUILD_CONTEXT_SCHEMA:
        errors.append("release build execution context schema is invalid")
    if not isinstance(environment, dict) or set(environment) != DETERMINISTIC_BUILD_ENVIRONMENT_KEYS:
        errors.append("release build environment is not the exact allowlist")
        environment = {}
    if context.get("environment_keys") != sorted(environment) \
            or context.get("environment_sha256") != canonical_sha256(environment):
        errors.append("release build context does not bind exact environment bytes")
    if context.get("ambient_environment_inherited") is not False:
        errors.append("release build inherited ambient environment")
    path_snapshot = context.get("path_snapshot")
    configuration_snapshot = context.get("configuration_snapshot")
    cargo_home = context.get("cargo_home")
    isolated = context.get("isolated_directories")
    if not isinstance(path_snapshot, dict) or set(path_snapshot) != {
        "value", "directories", "snapshot_sha256",
    } or path_snapshot.get("value") != environment.get("PATH") \
            or path_snapshot.get("snapshot_sha256") != canonical_sha256(
                path_snapshot.get("directories")
            ):
        errors.append("release build PATH snapshot is malformed")
    if not isinstance(configuration_snapshot, dict) or set(configuration_snapshot) != {
        "entries", "snapshot_sha256",
    } or configuration_snapshot.get("snapshot_sha256") != canonical_sha256(
        configuration_snapshot.get("entries")
    ):
        errors.append("release build configuration snapshot is malformed")
    for label, row in (("CARGO_HOME", cargo_home), *(
        (name, isolated.get(name) if isinstance(isolated, dict) else None)
        for name in ("HOME", "RUSTUP_HOME", "TMPDIR")
    )):
        if not isinstance(row, dict) or set(row) != {"path", "device", "inode", "mode"} \
                or row.get("path") != environment.get(label) \
                or not all(isinstance(row.get(field), int) for field in ("device", "inode", "mode")):
            errors.append(f"release build {label} directory binding is malformed")
    expected_selection_sha = canonical_sha256({
        name: toolchain.get(name, {}).get("selection") for name in ("cargo", "rustc")
    })
    if context.get("toolchain_selection_sha256") != expected_selection_sha:
        errors.append("release build context toolchain selector hash mismatch")
    if verify_files and environment:
        try:
            if path_snapshot != _live_path_snapshot(environment["PATH"]):
                errors.append("release build PATH executable namespace differs from receipt")
            cargo_home_path = pathlib.Path(environment["CARGO_HOME"])
            if configuration_snapshot != _live_configuration_snapshot(cargo_home_path):
                errors.append("release build Cargo/rustup configuration differs from receipt")
            if cargo_home != _directory_identity(cargo_home_path):
                errors.append("release build CARGO_HOME directory differs from receipt")
            for name in ("HOME", "RUSTUP_HOME", "TMPDIR"):
                if not isinstance(isolated, dict) \
                        or isolated.get(name) != _directory_identity(pathlib.Path(environment[name])):
                    errors.append(f"release build {name} directory differs from receipt")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"release build execution context cannot be verified: {exc}")
    return errors


def _compiler_artifact_errors(value: Any, *, label: str) -> list[str]:
    expected = {"target_name", "target_kind", "fresh", "executable"}
    if not isinstance(value, dict) or set(value) != expected:
        return [f"{label} compiler artifact is malformed"]
    errors: list[str] = []
    if value.get("target_name") != label or value.get("target_kind") != ["bin"]:
        errors.append(f"{label} compiler artifact target identity is invalid")
    if value.get("fresh") is not False:
        errors.append(f"{label} compiler artifact must be newly built (fresh=false)")
    errors.extend(_file_binding_errors(value.get("executable"), label=f"{label}.executable"))
    return errors


def _compiled_closure_errors(
    value: Any, *, label: str, executable: dict[str, Any] | None,
    verify_files: bool,
) -> list[str]:
    expected = {"dep_info", "entry_count", "entries", "closure_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        return [f"{label} compiled source closure is malformed"]
    errors: list[str] = []
    errors.extend(_file_binding_errors(value.get("dep_info"), label=f"{label}.dep_info"))
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{label} compiled source closure is empty")
        entries = []
    seen: set[str] = set()
    for row in entries:
        errors.extend(_file_binding_errors(row, label=f"{label}.entry"))
        if isinstance(row, dict):
            path = row.get("path")
            if path in seen:
                errors.append(f"{label} compiled source closure duplicates {path}")
            elif isinstance(path, str):
                seen.add(path)
            if verify_files and isinstance(path, str):
                try:
                    observed = physical_counter_attestation.file_identity(pathlib.Path(path))
                except (OSError, ValueError) as exc:
                    errors.append(f"{label} compiled source cannot be verified: {exc}")
                else:
                    if observed != row:
                        errors.append(f"{label} compiled source differs from bound bytes: {path}")
    if value.get("entry_count") != len(entries) \
            or value.get("closure_sha256") != canonical_sha256(entries):
        errors.append(f"{label} compiled source closure count/hash mismatch")
    if isinstance(executable, dict) and isinstance(executable.get("path"), str):
        expected_dep = str(pathlib.Path(executable["path"]).with_suffix(".d"))
        if value.get("dep_info", {}).get("path") != expected_dep:
            errors.append(f"{label} dep-info is not paired with its compiler artifact")
    return errors


def _validate_release_build(
    value: Any, *, source_manifest: dict[str, Any],
    release_boundary: dict[str, Any] | None = None,
    verify_files: bool = False,
) -> list[str]:
    if not isinstance(value, dict):
        return ["release_build must be an object"]
    expected = {
        "schema", "source_base_commit", "source_base_commit_role",
        "source_manifest_sha256", "source_authority_capsule_sha256", "cargo_lock_sha256",
        "release_boundary_attestation_sha256", "build_argv_sha256", "profile",
        "features", "target_host", "toolchain", "build_environment",
        "build_execution_context",
        "target_directory", "compiler_artifacts", "compiled_source_closures",
        "success", "built_at_unix_ns", "build_log",
        "probes", "runtime_defaults_changed", "receipt_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("release_build fields are incomplete or unexpected")
    if value.get("schema") != RELEASE_BUILD_SCHEMA:
        errors.append(f"release_build.schema must be {RELEASE_BUILD_SCHEMA}")
    if value.get("source_base_commit") != source_manifest.get("source_base_commit"):
        errors.append("release build source base commit does not match source capsule")
    if value.get("source_base_commit_role") != "repository-base-only-not-byte-authority":
        errors.append("release build overclaims base-commit byte authority")
    if value.get("source_manifest_sha256") != source_manifest.get("manifest_sha256"):
        errors.append("release build is not bound to source manifest")
    if value.get("source_authority_capsule_sha256") != source_manifest.get("capsule_sha256"):
        errors.append("release build does not name the critical capsule as byte authority")
    boundary_sha = value.get("release_boundary_attestation_sha256")
    if not _hex64(boundary_sha):
        errors.append("release build boundary attestation hash is invalid")
    if release_boundary is not None:
        if boundary_sha != release_boundary.get("attestation_sha256"):
            errors.append("release build is not bound to the aggregate release boundary")
    for field in ("cargo_lock_sha256", "build_argv_sha256"):
        if not _hex64(value.get(field)):
            errors.append(f"release_build.{field} is invalid")
    if value.get("profile") != "release" or value.get("features") != ["tq"]:
        errors.append("release build must be the exact release/tq profile")
    target_host = value.get("target_host")
    if not isinstance(target_host, str) or not target_host or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ."
        for character in target_host
    ):
        errors.append("release build target_host is invalid")
    toolchain = value.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != {"cargo", "rustc"}:
        errors.append("release build toolchain binding is incomplete")
        toolchain = {}
    for name in ("cargo", "rustc"):
        errors.extend(_toolchain_binding_errors(
            toolchain.get(name), label=f"release_build.toolchain.{name}",
            verify_files=verify_files,
        ))
    rustc_version = toolchain.get("rustc", {}).get("version_verbose", "") \
        if isinstance(toolchain.get("rustc"), dict) else ""
    if isinstance(target_host, str) and f"host: {target_host}" not in rustc_version.splitlines():
        errors.append("release build target_host is not the bound rustc host")
    environment = value.get("build_environment")
    target_directory = value.get("target_directory")
    expected_target_directory = None
    if release_boundary is not None:
        expected_target_directory = str((
            ROOT / "target" / "appendix-release" / canonical_sha256({
                "release_boundary_attestation_sha256": release_boundary.get("attestation_sha256"),
                "source_capsule_sha256": source_manifest.get("capsule_sha256"),
            })
        ).resolve())
    if not isinstance(target_directory, str) or not pathlib.Path(target_directory).is_absolute():
        errors.append("release build target_directory is invalid")
    elif expected_target_directory is not None and target_directory != expected_target_directory:
        errors.append("release build target_directory is not unique to boundary+capsule")
    if not isinstance(environment, dict) or set(environment) != DETERMINISTIC_BUILD_ENVIRONMENT_KEYS:
        errors.append("release build environment is not the exact deterministic allowlist")
        environment = {}
    fixed_environment = {
        "CARGO_INCREMENTAL": "0", "CARGO_NET_OFFLINE": "true",
        "CARGO_TARGET_DIR": target_directory, "CARGO_TERM_COLOR": "never",
        "LANG": "C", "LC_ALL": "C", "SOURCE_DATE_EPOCH": "1",
        "TERM": "dumb", "TZ": "UTC", "ZERO_AR_DATE": "1",
    }
    for name, expected_value in fixed_environment.items():
        if environment.get(name) != expected_value:
            errors.append(f"release build environment {name} is not deterministic")
    rustc_invocation = (
        toolchain.get("rustc", {}).get("invocation_path")
        if isinstance(toolchain.get("rustc"), dict) else None
    )
    if environment.get("RUSTC") != rustc_invocation:
        errors.append("release build environment RUSTC differs from selected binary")
    for name, suffix in (
        ("HOME", ".isolated-home"), ("RUSTUP_HOME", ".isolated-rustup-home"),
        ("TMPDIR", ".isolated-tmp"),
    ):
        expected_path = str(pathlib.Path(target_directory) / suffix) \
            if isinstance(target_directory, str) else None
        if environment.get(name) != expected_path:
            errors.append(f"release build environment {name} is not target-isolated")
    if not isinstance(environment.get("CARGO_HOME"), str) \
            or not pathlib.Path(environment["CARGO_HOME"]).is_absolute():
        errors.append("release build environment CARGO_HOME is invalid")
    expected_path_directories: list[str] = []
    for candidate in (
        pathlib.Path(toolchain.get("cargo", {}).get("invocation_path", "/invalid")).parent,
        pathlib.Path(toolchain.get("rustc", {}).get("invocation_path", "/invalid")).parent,
        pathlib.Path("/usr/bin"), pathlib.Path("/bin"),
    ):
        try:
            rendered = str(candidate.resolve(strict=True))
        except OSError:
            rendered = str(candidate)
        if rendered not in expected_path_directories:
            expected_path_directories.append(rendered)
    if environment.get("PATH") != os.pathsep.join(expected_path_directories):
        errors.append("release build environment PATH is not the exact minimal namespace")
    lease_fd = environment.get("HAWKING_HEAVY_LEASE_FD")
    try:
        if int(lease_fd) < 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("release build inherited lease FD environment binding is invalid")
    errors.extend(_build_context_errors(
        value.get("build_execution_context"), environment=environment,
        toolchain=toolchain, verify_files=verify_files,
    ))
    cargo_invocation = toolchain.get("cargo", {}).get("invocation_path") \
        if isinstance(toolchain.get("cargo"), dict) else None
    if isinstance(cargo_invocation, str) and isinstance(target_host, str):
        expected_argv = [
            cargo_invocation, "build", "--locked", "--release", "--target",
            target_host, "--target-dir", target_directory,
            "-p", "hawking", "--features", "tq", "--bin",
            "hawking-tq-device-probe", "--bin", "hawking-tq-spec-probe",
            "--message-format=json-render-diagnostics",
        ]
        if value.get("build_argv_sha256") != canonical_sha256(expected_argv):
            errors.append("release build argv differs from the pinned toolchain/host command")
    if value.get("success") is not True:
        errors.append("release build did not succeed")
    built = value.get("built_at_unix_ns")
    if isinstance(built, bool) or not isinstance(built, int) or built <= 0:
        errors.append("release build timestamp is invalid")
    elif release_boundary is not None \
            and built < release_boundary.get("observed_at_unix_ns", built + 1):
        errors.append("release build predates its release-boundary attestation")
    errors.extend(_file_binding_errors(value.get("build_log"), label="release_build.build_log"))
    probes = value.get("probes")
    if not isinstance(probes, dict) or set(probes) != {"device", "spec"}:
        errors.append("release build must bind exactly device and spec probes")
    else:
        errors.extend(_file_binding_errors(probes.get("device"), label="release_build.probes.device"))
        errors.extend(_file_binding_errors(probes.get("spec"), label="release_build.probes.spec"))
        if isinstance(probes.get("device"), dict) and isinstance(probes.get("spec"), dict):
            if probes["device"].get("sha256") == probes["spec"].get("sha256"):
                errors.append("device and spec release probes cannot have the same binary hash")
    artifacts = value.get("compiler_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "hawking-tq-device-probe", "hawking-tq-spec-probe",
    }:
        errors.append("release build compiler artifact set is incomplete")
        artifacts = {}
    for target_name in ("hawking-tq-device-probe", "hawking-tq-spec-probe"):
        errors.extend(_compiler_artifact_errors(
            artifacts.get(target_name), label=target_name,
        ))
        artifact = artifacts.get(target_name)
        executable = artifact.get("executable") if isinstance(artifact, dict) else None
        if isinstance(executable, dict) and isinstance(target_host, str) \
                and isinstance(target_directory, str):
            expected_path = str(
                (pathlib.Path(target_directory) / target_host / "release" / target_name).resolve()
            )
            if executable.get("path") != expected_path:
                errors.append(
                    f"{target_name} executable is not the exact pinned host-target Cargo output"
                )
    if isinstance(probes, dict):
        for probe_name, target_name in (
            ("device", "hawking-tq-device-probe"),
            ("spec", "hawking-tq-spec-probe"),
        ):
            artifact = artifacts.get(target_name)
            if not isinstance(artifact, dict) or probes.get(probe_name) != artifact.get("executable"):
                errors.append(f"release build {probe_name} probe is not the executable emitted by Cargo")
    closures = value.get("compiled_source_closures")
    if not isinstance(closures, dict) or set(closures) != {
        "hawking-tq-device-probe", "hawking-tq-spec-probe",
    }:
        errors.append("release build compiled source closure set is incomplete")
        closures = {}
    for target_name, required_source in (
        ("hawking-tq-device-probe", ROOT / "crates/hawking/src/tq_device_probe.rs"),
        ("hawking-tq-spec-probe", ROOT / "crates/hawking/src/tq_spec_probe.rs"),
    ):
        artifact = artifacts.get(target_name)
        executable = artifact.get("executable") if isinstance(artifact, dict) else None
        closure = closures.get(target_name)
        errors.extend(_compiled_closure_errors(
            closure, label=target_name, executable=executable,
            verify_files=verify_files,
        ))
        if isinstance(closure, dict) and str(required_source.resolve()) not in {
            row.get("path") for row in closure.get("entries", []) if isinstance(row, dict)
        }:
            errors.append(f"{target_name} compiled source closure omits its direct probe source")
    if verify_files:
        bindings = [("build_log", value.get("build_log"))]
        if isinstance(probes, dict):
            bindings.extend((f"probe {name}", probes.get(name)) for name in ("device", "spec"))
        for label, binding in bindings:
            if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
                continue
            try:
                observed = physical_counter_attestation.file_identity(pathlib.Path(binding["path"]))
            except (OSError, ValueError) as exc:
                errors.append(f"release build {label} cannot be verified: {exc}")
            else:
                if observed != binding:
                    errors.append(f"release build {label} differs from immutable file")
        log_binding = value.get("build_log")
        if isinstance(log_binding, dict) and isinstance(log_binding.get("path"), str):
            try:
                log_text = pathlib.Path(log_binding["path"]).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                errors.append(f"release build log cannot be parsed: {exc}")
            else:
                logged: dict[str, dict[str, Any]] = {}
                for line in log_text.splitlines():
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(message, dict) or message.get("reason") != "compiler-artifact":
                        continue
                    target = message.get("target")
                    name = target.get("name") if isinstance(target, dict) else None
                    if name not in {"hawking-tq-device-probe", "hawking-tq-spec-probe"} \
                            or target.get("kind") != ["bin"]:
                        continue
                    if name in logged:
                        errors.append(f"release build log duplicates compiler artifact {name}")
                    logged[name] = {
                        "target_name": name,
                        "target_kind": ["bin"],
                        "fresh": message.get("fresh"),
                        "executable_path": message.get("executable"),
                    }
                for name in ("hawking-tq-device-probe", "hawking-tq-spec-probe"):
                    artifact = artifacts.get(name)
                    expected_logged = {
                        "target_name": name,
                        "target_kind": ["bin"],
                        "fresh": artifact.get("fresh") if isinstance(artifact, dict) else None,
                        "executable_path": (
                            artifact.get("executable", {}).get("path")
                            if isinstance(artifact, dict) else None
                        ),
                    }
                    if logged.get(name) != expected_logged:
                        errors.append(f"release build log does not prove exact compiler artifact {name}")
    if value.get("runtime_defaults_changed") is not False:
        errors.append("release build cannot change runtime defaults")
    errors.extend(_hash_errors(value, field="release_build", hash_field="receipt_sha256"))
    return errors


def _receipt_parent_errors(
    receipt: dict[str, Any], *, corpus_sha: str, build_sha: str, source_sha: str,
) -> list[str]:
    errors: list[str] = []
    bindings = receipt.get("bindings")
    if not isinstance(bindings, dict):
        return ["receipt bindings are missing"]
    parents = bindings.get("parent_receipt_sha256")
    if not isinstance(parents, list):
        errors.append("receipt parent bindings are missing")
        parents = []
    for digest, label in ((corpus_sha, "corpus index"), (build_sha, "release build")):
        if digest not in parents:
            errors.append(f"receipt does not bind the {label} parent")
    if bindings.get("corpus_index_sha256") != corpus_sha:
        errors.append("receipt corpus_index_sha256 binding is absent or wrong")
    if bindings.get("release_build_sha256") != build_sha:
        errors.append("receipt release_build_sha256 binding is absent or wrong")
    if bindings.get("source_manifest_sha256") != source_sha:
        errors.append("receipt source_manifest_sha256 binding is absent or wrong")
    return errors


def _validate_execution_and_attestation(
    item: dict[str, Any], *, bundle: dict[str, Any], artifact_sha256: str,
    counter_payload: dict[str, Any], required_domains: tuple[str, ...],
    minimum_samples: int, expected_probe: dict[str, Any], verify_counter_files: bool,
) -> list[str]:
    errors: list[str] = []
    raw = bundle.get("raw_probe") if isinstance(bundle, dict) else None
    raw_sha = canonical_sha256(raw) if isinstance(raw, dict) else ""
    execution = item.get("execution_authority")
    errors.extend(physical_counter_attestation.validate_execution_authority(
        execution, raw_probe_sha256=raw_sha,
    ))
    if isinstance(execution, dict) and execution.get("probe_binary") != expected_probe:
        errors.append("execution authority probe binary does not match release build")
    attestation = item.get("counter_attestation")
    errors.extend(physical_counter_attestation.validate(
        attestation,
        raw_bundle_sha256=bundle.get("raw_bundle_sha256") if isinstance(bundle, dict) else "",
        artifact_sha256=artifact_sha256,
        execution_authority=execution if isinstance(execution, dict) else {},
        counter_payload=counter_payload,
        required_domains=required_domains,
        minimum_samples=minimum_samples,
        verify_files=verify_counter_files,
    ))
    return errors


def _device_counter_errors(payload: Any, *, raw: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return ["device counter_payload must be an object"]
    expected = {
        "schema", "raw_bundle_sha256", "artifact_sha256", "tensor",
        "runtime_path", "phase_markers_sha256", "trials", "summary",
    }
    errors: list[str] = []
    if set(payload) != expected:
        errors.append("device counter payload fields are incomplete or unexpected")
    if payload.get("schema") != DEVICE_COUNTER_SCHEMA:
        errors.append(f"device counter schema must be {DEVICE_COUNTER_SCHEMA}")
    if payload.get("artifact_sha256") != raw.get("artifact", {}).get("sha256"):
        errors.append("device counters are not bound to the artifact")
    if payload.get("tensor") != raw.get("tensor", {}).get("name") or payload.get("runtime_path") != raw.get("runtime_path"):
        errors.append("device counters are not bound to tensor/runtime")
    raw_phase_sha = raw.get("phase_markers", {}).get("phase_markers_sha256")
    if not _hex64(payload.get("phase_markers_sha256")):
        errors.append("device counter phase marker hash is invalid")
    elif payload.get("phase_markers_sha256") != raw_phase_sha:
        errors.append("device counters are not bound to the raw phase-marker manifest")
    trials = payload.get("trials")
    expected_trials = raw.get("benchmark", {}).get("trials")
    if not isinstance(trials, list) or len(trials) != expected_trials:
        errors.append("device counters do not cover every measured candidate trial")
        trials = []
    marker_hashes: set[str] = set()
    expected_markers = raw.get("benchmark", {}).get("trial_phase_marker_sha256", [])
    energy_total = 0.0
    gpu_total = 0
    bytes_total = 0
    occupancies: list[float] = []
    bandwidths: list[float] = []
    for index, row in enumerate(trials):
        if not isinstance(row, dict) or set(row) != {
            "index", "phase_marker_sha256", "energy_j", "gpu_time_ns",
            "physical_bytes", "occupancy_percent", "bandwidth_bytes_per_second",
        }:
            errors.append(f"device counter trial {index} is malformed")
            continue
        if row.get("index") != index:
            errors.append(f"device counter trial {index} index is wrong")
        marker = row.get("phase_marker_sha256")
        if (
            not _hex64(marker) or marker in marker_hashes
            or index >= len(expected_markers) or marker != expected_markers[index]
        ):
            errors.append(f"device counter trial {index} phase marker is invalid or reused")
        else:
            marker_hashes.add(marker)
        if not _finite(row.get("energy_j"), positive=True):
            errors.append(f"device counter trial {index} energy is invalid")
        else:
            energy_total += float(row["energy_j"])
        if not _finite(row.get("gpu_time_ns"), positive=True):
            errors.append(f"device counter trial {index} GPU time is invalid")
        else:
            gpu_total += int(row["gpu_time_ns"])
        if not _finite(row.get("physical_bytes"), positive=True):
            errors.append(f"device counter trial {index} physical bytes are invalid")
        else:
            bytes_total += int(row["physical_bytes"])
        occupancy = row.get("occupancy_percent")
        if not _finite(occupancy) or occupancy > 100:
            errors.append(f"device counter trial {index} occupancy is invalid")
        else:
            occupancies.append(float(occupancy))
        bandwidth = row.get("bandwidth_bytes_per_second")
        if not _finite(bandwidth, positive=True):
            errors.append(f"device counter trial {index} bandwidth is invalid")
        else:
            bandwidths.append(float(bandwidth))
    summary = payload.get("summary")
    if not isinstance(summary, dict) or set(summary) != {
        "energy_j_total", "gpu_time_ns_total", "physical_bytes_total",
        "occupancy_percent_mean", "bandwidth_bytes_per_second_mean",
    }:
        errors.append("device counter summary is malformed")
    elif trials:
        expected_summary = {
            "energy_j_total": energy_total,
            "gpu_time_ns_total": gpu_total,
            "physical_bytes_total": bytes_total,
            "occupancy_percent_mean": statistics.fmean(occupancies),
            "bandwidth_bytes_per_second_mean": statistics.fmean(bandwidths),
        }
        for field, expected_value in expected_summary.items():
            observed = summary.get(field)
            if not _finite(observed) or not math.isclose(float(observed), float(expected_value), rel_tol=1e-12, abs_tol=1e-12):
                errors.append(f"device counter summary {field} does not match trials")
    return errors


def _feature_census_errors(raw: dict[str, Any]) -> list[str]:
    value = raw.get("feature_census")
    expected = {
        "schema", "rht_mode", "rht_blocks", "rht_exercised", "outlier_count",
        "outlier_exercised", "projection_passes", "residual_passes",
        "residual_exercised", "dispatches_per_invocation",
        "dispatch_geometry_sha256", "kernel_sequence", "feature_identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["raw feature_census is incomplete or unexpected"]
    errors: list[str] = []
    if value.get("schema") != appendix_device_runner.FEATURE_CENSUS_SCHEMA:
        errors.append("feature census schema is invalid")
    if value.get("rht_mode") not in {"none", "cols"}:
        errors.append("feature census rht_mode must be none|cols")
    for field in ("rht_blocks", "outlier_count", "residual_passes"):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            errors.append(f"feature census {field} is invalid")
    if value.get("rht_exercised") is not (value.get("rht_mode") == "cols" and value.get("rht_blocks", 0) > 0):
        errors.append("feature census RHT exercised flag is inconsistent")
    if value.get("outlier_exercised") is not (isinstance(value.get("outlier_count"), int) and value.get("outlier_count", 0) > 0):
        errors.append("feature census OUTL exercised flag is inconsistent")
    residual_enabled = raw.get("residual_probe", {}).get("enabled") is True
    if (
        value.get("projection_passes") != 1 + int(residual_enabled)
        or value.get("residual_passes") != int(residual_enabled)
        or value.get("residual_exercised") is not residual_enabled
    ):
        errors.append("feature census residual pass is not backed by the exact two-pass probe")
    dispatches = value.get("dispatches_per_invocation")
    if isinstance(dispatches, bool) or not isinstance(dispatches, int) or dispatches <= 0:
        errors.append("feature census dispatch count must be positive")
    if not _hex64(value.get("dispatch_geometry_sha256")):
        errors.append("feature census dispatch geometry hash is invalid")
    kernels = value.get("kernel_sequence")
    if not isinstance(kernels, list) or not kernels or any(not isinstance(row, str) or not row for row in kernels):
        errors.append("feature census kernel sequence is invalid")
    feature_identity = raw.get("feature_identity", {})
    if value.get("feature_identity_sha256") != feature_identity.get("feature_identity_sha256"):
        errors.append("feature census is not bound to exact feature identity")
    if residual_enabled and "strand_bitslice_reduce_rows_accum" not in (kernels or []):
        errors.append("feature census residual pass omits the accumulate reduction kernel")
    return errors


def _projection_work_identity(raw: Any) -> dict[str, Any]:
    """Normalize only the mathematical projection work shared across runtimes.

    Runtime recipes, kernels, matrix-cell ids, and dispatch details are
    deliberately excluded: a compact/hashed/computed child is expected to
    differ from its stored parent there.  Artifact/tensor geometry and every
    optional residual input are included so a receipt cannot borrow a stored
    parent that performed different work.
    """
    raw = raw if isinstance(raw, dict) else {}
    tensor = raw.get("tensor") if isinstance(raw.get("tensor"), dict) else {}
    census = (
        raw.get("feature_census")
        if isinstance(raw.get("feature_census"), dict) else {}
    )
    feature = (
        raw.get("feature_identity")
        if isinstance(raw.get("feature_identity"), dict) else {}
    )
    artifact = raw.get("artifact") if isinstance(raw.get("artifact"), dict) else {}
    residual = (
        raw.get("residual_probe")
        if isinstance(raw.get("residual_probe"), dict) else {}
    )
    residual_enabled = residual.get("enabled") is True

    def normalized_tensor(
        value: dict[str, Any], *, rht_mode: Any, rht_blocks: Any,
        outlier_count: Any,
    ) -> dict[str, Any]:
        return {
            "name": value.get("name"),
            "rows": value.get("rows"),
            "cols": value.get("cols"),
            "weights": value.get("weights"),
            "blocks": value.get("blocks"),
            "k_bits": value.get("k_bits"),
            "l_bits": value.get("l_bits"),
            "rht_mode": rht_mode,
            "rht_blocks": rht_blocks,
            "outlier_count": outlier_count,
        }

    identity: dict[str, Any] = {
        "projection_recipe": feature.get("projection_recipe"),
        "projection_passes": feature.get("projection_passes"),
        "base": {
            "artifact_sha256": artifact.get("sha256"),
            "tensor": normalized_tensor(
                tensor,
                rht_mode=census.get("rht_mode"),
                rht_blocks=census.get("rht_blocks"),
                outlier_count=census.get("outlier_count"),
            ),
        },
        "residual_enabled": residual_enabled,
        "residual": None,
    }
    if residual_enabled:
        residual_artifact = (
            residual.get("artifact")
            if isinstance(residual.get("artifact"), dict) else {}
        )
        residual_tensor = (
            residual.get("tensor")
            if isinstance(residual.get("tensor"), dict) else {}
        )
        identity["residual"] = {
            "artifact_sha256": residual_artifact.get("sha256"),
            "tensor": normalized_tensor(
                residual_tensor,
                rht_mode=residual_tensor.get("rht_mode"),
                rht_blocks=residual_tensor.get("rht_blocks"),
                outlier_count=residual_tensor.get("outlier_count"),
            ),
        }
    return identity


def _stored_parent_projection_errors(child: Any, stored: Any) -> list[str]:
    if _projection_work_identity(child) != _projection_work_identity(stored):
        return [
            "stored parent does not perform the same normalized projection work "
            "(base/residual artifacts, tensors, geometry, quantization, RHT, or outliers differ)"
        ]
    return []


def _spec_counter_errors(payload: Any, *, raw: dict[str, Any], repeats: int) -> list[str]:
    if not isinstance(payload, dict):
        return ["spec counter_payload must be an object"]
    expected = {
        "schema", "raw_bundle_sha256", "artifact_sha256", "runtime_path",
        "phase_markers_sha256", "batches",
    }
    errors: list[str] = []
    if set(payload) != expected:
        errors.append("spec counter payload fields are incomplete or unexpected")
    if payload.get("schema") != SPEC_COUNTER_SCHEMA:
        errors.append(f"spec counter schema must be {SPEC_COUNTER_SCHEMA}")
    if payload.get("artifact_sha256") != raw.get("artifact", {}).get("sha256"):
        errors.append("spec counters are not bound to the artifact")
    if payload.get("runtime_path") != raw.get("runtime_path"):
        errors.append("spec counters are not bound to the runtime path")
    protocol = raw.get("measurement_protocol", {})
    if payload.get("phase_markers_sha256") != protocol.get("phase_markers_sha256"):
        errors.append("spec counters are not bound to phase markers")
    rows = payload.get("batches")
    if not isinstance(rows, list) or [row.get("b") for row in rows if isinstance(row, dict)] != list(range(1, 9)):
        errors.append("spec counter batches must be ordered B=1..8")
        return errors
    for row in rows:
        batch = row.get("b")
        protocol_rows = {
            value.get("b"): value
            for value in protocol.get("batches", []) if isinstance(value, dict)
        }
        expected_markers = [
            value.get("phase_marker_sha256")
            for value in protocol_rows.get(batch, {}).get("repeats", [])
            if isinstance(value, dict)
        ]
        measurements = row.get("repeats")
        if not isinstance(measurements, list) or len(measurements) != repeats:
            errors.append(f"B={batch} counters do not cover every independent repeat")
            continue
        markers: set[str] = set()
        for repeat, measurement in enumerate(measurements):
            if not isinstance(measurement, dict) or set(measurement) != {
                "repeat", "phase_marker_sha256", "energy_j", "gpu_time_ns", "physical_bytes",
            }:
                errors.append(f"B={batch} counter repeat {repeat} is malformed")
                continue
            if measurement.get("repeat") != repeat:
                errors.append(f"B={batch} counter repeat index is wrong")
            marker = measurement.get("phase_marker_sha256")
            if (
                not _hex64(marker) or marker in markers
                or repeat >= len(expected_markers) or marker != expected_markers[repeat]
            ):
                errors.append(f"B={batch} counter phase marker is invalid or reused")
            else:
                markers.add(marker)
            for field in ("energy_j", "gpu_time_ns", "physical_bytes"):
                if not _finite(measurement.get(field), positive=True):
                    errors.append(f"B={batch} counter {field} is invalid")
    return errors


def _spec_protocol_errors(raw: dict[str, Any], curve: dict[str, Any]) -> tuple[list[str], int]:
    protocol = raw.get("measurement_protocol")
    expected = {
        "warmups_per_batch", "independent_repeats_per_batch",
        "randomized_balanced_batch_order", "paired_interleaved_baseline",
        "baseline_reused_across_batches", "phase_marker_schema",
        "phase_markers_sha256", "monotone_transform_applied", "batches",
    }
    if not isinstance(protocol, dict) or set(protocol) != expected:
        return ["spec measurement_protocol is incomplete or unexpected"], 0
    errors: list[str] = []
    warmups = protocol.get("warmups_per_batch")
    repeats = protocol.get("independent_repeats_per_batch")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 3:
        errors.append("spec measurement protocol requires at least three warmups per batch")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 5:
        errors.append("spec measurement protocol requires at least five independent repeats per batch")
        repeats = 0
    if protocol.get("randomized_balanced_batch_order") is not True:
        errors.append("spec batch order is not randomized and balanced")
    if protocol.get("paired_interleaved_baseline") is not True or protocol.get("baseline_reused_across_batches") is not False:
        errors.append("spec repeats are not independently paired with a same-batch baseline")
    if protocol.get("phase_marker_schema") != "hawking.physical_phase_markers.v1" or not _hex64(protocol.get("phase_markers_sha256")):
        errors.append("spec phase marker binding is invalid")
    if protocol.get("monotone_transform_applied") is not False:
        errors.append("spec verifier curve applied a manufactured monotonicity transform")
    batches = protocol.get("batches")
    if not isinstance(batches, list) or [row.get("b") for row in batches if isinstance(row, dict)] != list(range(1, 9)):
        errors.append("spec measurement protocol batches must be ordered B=1..8")
        return errors, int(repeats or 0)
    ratios: list[float] = []
    curve_rows = curve.get("experiment_payload", {}).get("batches", [])
    curve_by_b = {row.get("b"): row for row in curve_rows if isinstance(row, dict)}
    all_markers: set[str] = set()
    for batch_row in batches:
        batch = batch_row["b"]
        rows = batch_row.get("repeats")
        if not isinstance(rows, list) or len(rows) != repeats:
            errors.append(f"B={batch} does not contain every independent repeat")
            continue
        baseline: list[int] = []
        verifier: list[int] = []
        for repeat, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {
                "repeat", "baseline_wall_ns", "verifier_wall_ns",
                "phase_marker_sha256", "exact_token_match", "mismatches", "skipped",
            }:
                errors.append(f"B={batch} repeat {repeat} is malformed")
                continue
            if row.get("repeat") != repeat:
                errors.append(f"B={batch} repeat index is wrong")
            marker = row.get("phase_marker_sha256")
            if not _hex64(marker) or marker in all_markers:
                errors.append(f"B={batch} repeat phase marker is invalid or reused")
            else:
                all_markers.add(marker)
            if not _finite(row.get("baseline_wall_ns"), positive=True) or not _finite(row.get("verifier_wall_ns"), positive=True):
                errors.append(f"B={batch} repeat timing is invalid")
                continue
            baseline.append(int(row["baseline_wall_ns"]))
            verifier.append(int(row["verifier_wall_ns"]))
            if row.get("exact_token_match") is not True or row.get("mismatches") != 0 or row.get("skipped") != 0:
                errors.append(f"B={batch} independent repeat failed parity or skipped work")
        if len(baseline) != repeats:
            continue
        median = statistics.median(verifier)
        raw_ratio = median / statistics.median(baseline)
        ratios.append(raw_ratio)
        curve_row = curve_by_b.get(batch)
        if not isinstance(curve_row, dict):
            errors.append(f"curve receipt is missing B={batch}")
            continue
        if curve_row.get("trials") != repeats:
            errors.append(f"curve B={batch} trials do not equal independent repeats")
        if not math.isclose(float(curve_row.get("median_ns", -1)), float(median), rel_tol=0, abs_tol=0):
            errors.append(f"curve B={batch} median is not derived from independent repeats")
        for field in ("raw_total_forward_equiv", "total_forward_equiv"):
            if not _finite(curve_row.get(field), positive=True) or not math.isclose(
                float(curve_row[field]), raw_ratio, rel_tol=1e-12, abs_tol=1e-12,
            ):
                errors.append(f"curve B={batch} {field} differs from observed raw ratio")
    if len(ratios) == 8 and any(right < left for left, right in zip(ratios, ratios[1:])):
        errors.append("observed verifier curve is not monotone; no transform is permitted")
    curve_payload = curve.get("experiment_payload", {})
    if curve_payload.get("curve_method") != {
        "warmups_per_batch": warmups,
        "independent_repeats_per_batch": repeats,
        "paired_interleaved_baseline": True,
        "monotone_transform_applied": False,
        "ucb_method": "paired_bootstrap_95",
        "confidence_level": 0.95,
        "phase_markers_sha256": protocol.get("phase_markers_sha256"),
    }:
        errors.append("curve receipt method does not bind the rigorous measurement protocol")
    return errors, int(repeats or 0)


def validate_gate(
    document: Any,
    *,
    matrix: dict[str, Any] | None = None,
    spec_matrix: dict[str, Any] | None = None,
    verify_counter_files: bool = True,
) -> list[str]:
    """Validate a complete physical-evidence packet without executing anything."""
    if not isinstance(document, dict):
        return ["physical evidence packet must be an object"]
    expected_fields = {
        "schema", "release_boundary", "corpus_index", "corpus_verification",
        "source_manifest", "release_build", "cpu_error_policy", "spec_label",
        "device_evidence", "spec_evidence", "default_mutation_requested",
        "gate_sha256",
    }
    errors: list[str] = []
    if set(document) != expected_fields:
        errors.append("physical evidence packet fields are incomplete or unexpected")
    if document.get("schema") != SCHEMA:
        errors.append(f"physical evidence packet schema must be {SCHEMA}")
    unstamped = copy.deepcopy(document)
    claimed = unstamped.pop("gate_sha256", None)
    if not _hex64(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("physical evidence packet gate_sha256 mismatch")
    if document.get("default_mutation_requested") is not False:
        errors.append("aggregate physical evidence cannot request a default mutation")

    boundary = document.get("release_boundary")
    errors.extend(_validate_release_boundary(boundary))
    corpus = document.get("corpus_index")
    errors.extend(_validate_corpus_index(corpus))
    corpus_dict = corpus if isinstance(corpus, dict) else {}
    errors.extend(_validate_corpus_verification(document.get("corpus_verification"), index=corpus_dict))
    source = document.get("source_manifest")
    errors.extend(_validate_source_manifest(source, verify_files=verify_counter_files))
    source_dict = source if isinstance(source, dict) else {}
    if isinstance(boundary, dict) and source_dict.get(
        "release_boundary_attestation_sha256"
    ) != boundary.get("attestation_sha256"):
        errors.append("critical-source capsule is not bound to aggregate release boundary")
    build = document.get("release_build")
    errors.extend(_validate_release_build(
        build, source_manifest=source_dict, release_boundary=boundary if isinstance(boundary, dict) else None,
        verify_files=verify_counter_files,
    ))
    build_dict = build if isinstance(build, dict) else {}
    if corpus_dict.get("source_base_commit") != source_dict.get("source_base_commit"):
        errors.append("corpus index and critical-source capsule have different base commits")
    if document.get("cpu_error_policy") != CPU_ERROR_POLICY:
        errors.append("physical evidence CPU error policy is absent or weaker than the gate policy")

    corpus_sha = corpus_dict.get("index_sha256", "")
    build_sha = build_dict.get("receipt_sha256", "")
    source_sha = source_dict.get("manifest_sha256", "")
    probes = build_dict.get("probes") if isinstance(build_dict.get("probes"), dict) else {}
    device_probe = probes.get("device", {})
    spec_probe = probes.get("spec", {})

    matrix = tq_runtime_matrix.build_matrix() if matrix is None else matrix
    expected_cells = {
        row["id"]: row for row in matrix.get("cells", [])
        if isinstance(row, dict) and row.get("state") == "deferred"
    }
    seen_seals: set[str] = set()
    aggregate_parents = {
        "boundary_attestation": boundary,
        "corpus_index": corpus,
        "corpus_verification": document.get("corpus_verification"),
        "source_manifest": source,
        "release_build": build,
    }
    evidence, sealed_device_errors = _unwrap_operator_sealed_list(
        document.get("device_evidence"), kind="device",
        verify_counter_files=verify_counter_files, seen_seals=seen_seals,
        aggregate_parents=aggregate_parents,
    )
    errors.extend(sealed_device_errors)
    item_by_cell: dict[str, dict[str, Any]] = {}
    receipt_hashes: set[str] = set()
    raw_hashes: set[str] = set()
    attestation_hashes: set[str] = set()
    feature_seen = {"rht": False, "outlier": False, "residual": False}
    for index, item in enumerate(evidence):
        prefix = f"device_evidence[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "cell_id", "raw_bundle", "receipt", "execution_authority",
            "counter_payload", "counter_attestation", "xctrace_export_evidence",
        }:
            errors.append(f"{prefix} is malformed")
            continue
        cell_id = item.get("cell_id")
        cell = expected_cells.get(cell_id)
        if cell is None:
            errors.append(f"{prefix} cell is absent or not deferred in the frozen matrix")
            continue
        if cell_id in item_by_cell:
            errors.append(f"{prefix} duplicates matrix cell {cell_id}")
            continue
        item_by_cell[cell_id] = item
        bundle = item.get("raw_bundle")
        bundle_errors = appendix_device_runner.validate_bundle(bundle)
        errors.extend(f"{prefix}: {error}" for error in bundle_errors)
        raw = bundle.get("raw_probe", {}) if isinstance(bundle, dict) else {}
        if raw.get("source_commit") != build_dict.get("source_base_commit"):
            errors.append(f"{prefix} raw probe repository-base marker differs from release build")
        errors.extend(f"{prefix}: {error}" for error in _corpus_artifact_binding_errors(
            raw.get("artifact") if isinstance(raw, dict) else None,
            corpus_index=corpus_dict,
            label="base",
        ))
        residual_probe = raw.get("residual_probe", {}) if isinstance(raw, dict) else {}
        if residual_probe.get("enabled") is True:
            errors.extend(f"{prefix}: {error}" for error in _corpus_artifact_binding_errors(
                residual_probe.get("artifact"),
                corpus_index=corpus_dict,
                label="residual",
            ))
        raw_bundle_sha = bundle.get("raw_bundle_sha256") if isinstance(bundle, dict) else None
        if not _hex64(raw_bundle_sha) or raw_bundle_sha in raw_hashes:
            errors.append(f"{prefix} raw bundle hash is invalid or reused")
        else:
            raw_hashes.add(raw_bundle_sha)
        tensor = raw.get("tensor", {}) if isinstance(raw, dict) else {}
        identity = raw.get("matrix_identity") if isinstance(raw, dict) else None
        expected_identity = {
            "schema": appendix_device_runner.MATRIX_IDENTITY_SCHEMA,
            "cell_id": cell_id,
            "matrix_cell_sha256": canonical_sha256(cell),
            "model": cell.get("model"),
            "tensor_family": cell.get("tensor_family"),
            "shape": cell.get("shape"),
            "k_bits": cell.get("k_bits"),
            "l_bits": cell.get("l_bits"),
            "runtime_path": cell.get("runtime_path"),
            "artifact_sha256": raw.get("artifact", {}).get("sha256") if isinstance(raw, dict) else None,
            "artifact_tensor_name": tensor.get("name") if isinstance(tensor, dict) else None,
        }
        expected_identity["identity_sha256"] = canonical_sha256(expected_identity)
        if identity != expected_identity:
            errors.append(f"{prefix} exact model/tensor/matrix identity is not proven")
        if tensor.get("name") != cell.get("tensor_family"):
            errors.append(f"{prefix} artifact tensor cannot be cross-credited to the matrix family")
        if (
            {"rows": tensor.get("rows"), "cols": tensor.get("cols")} != cell.get("shape")
            or tensor.get("k_bits") != cell.get("k_bits")
            or tensor.get("l_bits") != cell.get("l_bits")
            or raw.get("runtime_path") != cell.get("runtime_path")
        ):
            errors.append(f"{prefix} raw geometry/runtime does not exactly match matrix cell")
        errors.extend(f"{prefix}: {error}" for error in _feature_census_errors(raw))
        census = raw.get("feature_census", {}) if isinstance(raw, dict) else {}
        feature_counts = raw.get("feature_identity", {}).get("feature_counts", {})
        rht_passes = feature_counts.get("rht_cols_passes")
        outlier_passes = feature_counts.get("outlier_corrected_passes")
        feature_seen["rht"] |= (
            isinstance(rht_passes, int) and not isinstance(rht_passes, bool) and rht_passes > 0
        )
        feature_seen["outlier"] |= (
            isinstance(outlier_passes, int)
            and not isinstance(outlier_passes, bool)
            and outlier_passes > 0
        )
        feature_seen["residual"] |= (
            census.get("residual_exercised") is True
            and raw.get("residual_probe", {}).get("enabled") is True
            and raw.get("feature_identity", {}).get("projection_recipe")
            == "two_pass_residual_accumulate"
        )
        parity = raw.get("parity", {}) if isinstance(raw, dict) else {}
        if (
            parity.get("exact_q12") is not True or parity.get("q12_mismatches") != 0
            or parity.get("exact_fused_vs_stored_gpu") is not True
            or parity.get("fused_bit_mismatches") != 0
        ):
            errors.append(f"{prefix} exact stored/Q12 device parity failed")
        for field, maximum in (
            ("cpu_reference_max_abs_error", CPU_ERROR_POLICY["max_abs_error"]),
            ("cpu_reference_max_rel_error", CPU_ERROR_POLICY["max_rel_error"]),
        ):
            value = parity.get(field)
            if not _finite(value) or float(value) > maximum:
                errors.append(f"{prefix} {field} exceeds the bounded CPU policy")
        receipt = item.get("receipt")
        validation = tq_receipt_contract.validate_receipt(
            receipt, known_cell_ids=set(expected_cells),
        )
        errors.extend(f"{prefix}: {error}" for error in validation)
        if not isinstance(receipt, dict) or receipt.get("cell_id") != cell_id:
            errors.append(f"{prefix} receipt is not bound to its exact cell")
            receipt = {}
        receipt_sha = receipt.get("receipt_sha256")
        if not _hex64(receipt_sha) or receipt_sha in receipt_hashes:
            errors.append(f"{prefix} receipt hash is invalid or reused")
        else:
            receipt_hashes.add(receipt_sha)
        errors.extend(f"{prefix}: {error}" for error in _receipt_parent_errors(
            receipt, corpus_sha=corpus_sha, build_sha=build_sha, source_sha=source_sha,
        ))
        counter_payload = item.get("counter_payload")
        errors.extend(f"{prefix}: {error}" for error in _device_counter_errors(counter_payload, raw=raw))
        if isinstance(counter_payload, dict) and counter_payload.get("raw_bundle_sha256") != raw_bundle_sha:
            errors.append(f"{prefix} counter payload is not bound to raw bundle")
        attestation = item.get("counter_attestation")
        attestation_sha = attestation.get("attestation_sha256") if isinstance(attestation, dict) else None
        if not _hex64(attestation_sha) or attestation_sha in attestation_hashes:
            errors.append(f"{prefix} counter attestation hash is invalid or reused")
        else:
            attestation_hashes.add(attestation_sha)
        trial_count = raw.get("benchmark", {}).get("trials", 0) if isinstance(raw, dict) else 0
        errors.extend(f"{prefix}: {error}" for error in _validate_execution_and_attestation(
            item, bundle=bundle if isinstance(bundle, dict) else {},
            artifact_sha256=raw.get("artifact", {}).get("sha256", "") if isinstance(raw, dict) else "",
            counter_payload=counter_payload if isinstance(counter_payload, dict) else {},
            required_domains=DEVICE_COUNTER_DOMAINS,
            minimum_samples=trial_count if isinstance(trial_count, int) and trial_count > 0 else 1,
            expected_probe=device_probe if isinstance(device_probe, dict) else {},
            verify_counter_files=verify_counter_files,
        ))
        if receipt.get("physical_counter_attestation_sha256") != attestation_sha:
            errors.append(f"{prefix} receipt does not bind counter attestation")
        if receipt.get("physical_counter_payload_sha256") != canonical_sha256(counter_payload):
            errors.append(f"{prefix} receipt does not bind normalized counter payload")
        payload = receipt.get("experiment_payload", {})
        if payload.get("default_change_requested") is not False:
            errors.append(f"{prefix} receipt requests a default change")

    if set(item_by_cell) != set(expected_cells):
        missing = len(set(expected_cells) - set(item_by_cell))
        extra = len(set(item_by_cell) - set(expected_cells))
        errors.append(f"device matrix physical coverage is incomplete (missing={missing}, extra={extra})")
    if {cell.get("runtime_path") for cell in expected_cells.values()} != set(RUNTIME_PATHS):
        errors.append("frozen device matrix does not expose all four runtime paths")
    if {expected_cells[cell_id].get("runtime_path") for cell_id in item_by_cell} != set(RUNTIME_PATHS):
        errors.append("device evidence does not cover all four runtime paths")
    if not all(feature_seen.values()):
        missing_features = sorted(name for name, seen in feature_seen.items() if not seen)
        errors.append("device evidence lacks positive feature coverage: " + ", ".join(missing_features))
    for cell_id, item in item_by_cell.items():
        cell = expected_cells[cell_id]
        if cell.get("runtime_path") == "stored":
            continue
        dependencies = cell.get("depends_on")
        if not isinstance(dependencies, list) or len(dependencies) != 1:
            errors.append(f"device cell {cell_id} lacks one exact stored dependency")
            continue
        stored_item = item_by_cell.get(dependencies[0])
        if stored_item is None:
            errors.append(f"device cell {cell_id} stored dependency is absent")
            continue
        receipt = item.get("receipt", {})
        stored_receipt = stored_item.get("receipt", {})
        parents = receipt.get("bindings", {}).get("parent_receipt_sha256", [])
        if stored_receipt.get("receipt_sha256") not in parents:
            errors.append(f"device cell {cell_id} does not bind its exact stored receipt")
        raw = item.get("raw_bundle", {}).get("raw_probe", {})
        stored_raw = stored_item.get("raw_bundle", {}).get("raw_probe", {})
        errors.extend(
            f"device cell {cell_id} {error}"
            for error in _stored_parent_projection_errors(raw, stored_raw)
        )

    label = document.get("spec_label")
    if not isinstance(label, str) or not label:
        errors.append("spec_label must be non-empty")
        label = "CORPUS"
    spec_matrix = spec_reentry_scaffold.build_matrix(label) if spec_matrix is None else spec_matrix
    parity_cells = {
        row["knobs"]["runtime_path"]: row for row in spec_matrix.get("cells", [])
        if isinstance(row, dict) and row.get("receipt_schema") == "hawking.spec_tq_batched_parity.v1"
    }
    curve_cells = {
        row["knobs"]["runtime_path"]: row for row in spec_matrix.get("cells", [])
        if isinstance(row, dict) and row.get("receipt_schema") == "hawking.spec_verifier_curve.v1"
    }
    known_spec_ids = {row.get("id") for row in spec_matrix.get("cells", []) if isinstance(row, dict)}
    spec_evidence, sealed_spec_errors = _unwrap_operator_sealed_list(
        document.get("spec_evidence"), kind="spec",
        verify_counter_files=verify_counter_files, seen_seals=seen_seals,
        aggregate_parents=aggregate_parents,
    )
    errors.extend(sealed_spec_errors)
    spec_by_runtime: dict[str, dict[str, Any]] = {}
    spec_receipt_hashes: set[str] = set()
    spec_attestation_hashes: set[str] = set()
    for index, item in enumerate(spec_evidence):
        prefix = f"spec_evidence[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "runtime_path", "raw_bundle", "parity_receipt", "curve_receipt",
            "execution_authority", "counter_payload", "counter_attestation",
            "xctrace_export_evidence",
        }:
            errors.append(f"{prefix} is malformed")
            continue
        runtime = item.get("runtime_path")
        if runtime not in RUNTIME_PATHS or runtime in spec_by_runtime:
            errors.append(f"{prefix} runtime path is invalid or duplicated")
            continue
        spec_by_runtime[runtime] = item
        bundle = item.get("raw_bundle")
        errors.extend(f"{prefix}: {error}" for error in spec_tq_runner.validate_bundle(bundle))
        raw = bundle.get("raw_probe", {}) if isinstance(bundle, dict) else {}
        if raw.get("source_commit") != build_dict.get("source_base_commit"):
            errors.append(f"{prefix} raw spec probe repository-base marker differs from release build")
        if raw.get("runtime_path") != runtime:
            errors.append(f"{prefix} raw runtime does not match evidence runtime")
        identity = raw.get("matrix_identity") if isinstance(raw, dict) else None
        expected_identity = {
            "runtime_path": runtime,
            "parity_cell_id": parity_cells.get(runtime, {}).get("id"),
            "curve_cell_id": curve_cells.get(runtime, {}).get("id"),
            "model_sha256": raw.get("model", {}).get("sha256"),
            "artifact_sha256": raw.get("artifact", {}).get("sha256"),
            "tokenizer_sha256": raw.get("tokenizer", {}).get("sha256"),
            "prompt_set_sha256": raw.get("prompt_set", {}).get("sha256"),
        }
        if identity != expected_identity:
            errors.append(f"{prefix} exact speculative matrix identity is not proven")
        parity_receipt = item.get("parity_receipt")
        curve_receipt = item.get("curve_receipt")
        for kind, receipt, expected_cell in (
            ("parity", parity_receipt, parity_cells.get(runtime, {})),
            ("curve", curve_receipt, curve_cells.get(runtime, {})),
        ):
            validation = spec_receipt_contract.validate_receipt(
                receipt, known_cell_ids=known_spec_ids,
            )
            errors.extend(f"{prefix} {kind}: {error}" for error in validation)
            if not isinstance(receipt, dict) or receipt.get("cell_id") != expected_cell.get("id"):
                errors.append(f"{prefix} {kind} receipt is not bound to its exact matrix cell")
                receipt = {}
            receipt_sha = receipt.get("receipt_sha256")
            if not _hex64(receipt_sha) or receipt_sha in spec_receipt_hashes:
                errors.append(f"{prefix} {kind} receipt hash is invalid or reused")
            else:
                spec_receipt_hashes.add(receipt_sha)
            errors.extend(f"{prefix} {kind}: {error}" for error in _receipt_parent_errors(
                receipt, corpus_sha=corpus_sha, build_sha=build_sha, source_sha=source_sha,
            ))
            if receipt.get("experiment_payload", {}).get("default_change_requested") is not False:
                errors.append(f"{prefix} {kind} receipt requests a default change")
        if isinstance(parity_receipt, dict) and isinstance(curve_receipt, dict):
            parents = curve_receipt.get("bindings", {}).get("parent_receipt_sha256", [])
            if parity_receipt.get("receipt_sha256") not in parents:
                errors.append(f"{prefix} curve does not bind its exact parity receipt")
        protocol_errors, repeats = _spec_protocol_errors(
            raw, curve_receipt if isinstance(curve_receipt, dict) else {},
        )
        errors.extend(f"{prefix}: {error}" for error in protocol_errors)
        counter_payload = item.get("counter_payload")
        errors.extend(f"{prefix}: {error}" for error in _spec_counter_errors(
            counter_payload, raw=raw, repeats=repeats,
        ))
        raw_bundle_sha = bundle.get("raw_bundle_sha256") if isinstance(bundle, dict) else None
        if isinstance(counter_payload, dict) and counter_payload.get("raw_bundle_sha256") != raw_bundle_sha:
            errors.append(f"{prefix} counter payload is not bound to raw bundle")
        attestation = item.get("counter_attestation")
        attestation_sha = attestation.get("attestation_sha256") if isinstance(attestation, dict) else None
        if not _hex64(attestation_sha) or attestation_sha in spec_attestation_hashes:
            errors.append(f"{prefix} counter attestation hash is invalid or reused")
        else:
            spec_attestation_hashes.add(attestation_sha)
        errors.extend(f"{prefix}: {error}" for error in _validate_execution_and_attestation(
            item, bundle=bundle if isinstance(bundle, dict) else {},
            artifact_sha256=raw.get("artifact", {}).get("sha256", "") if isinstance(raw, dict) else "",
            counter_payload=counter_payload if isinstance(counter_payload, dict) else {},
            required_domains=SPEC_COUNTER_DOMAINS,
            minimum_samples=max(1, repeats * 8),
            expected_probe=spec_probe if isinstance(spec_probe, dict) else {},
            verify_counter_files=verify_counter_files,
        ))
        for kind, receipt in (("parity", parity_receipt), ("curve", curve_receipt)):
            if not isinstance(receipt, dict):
                continue
            if receipt.get("physical_counter_attestation_sha256") != attestation_sha:
                errors.append(f"{prefix} {kind} receipt does not bind counter attestation")
            if receipt.get("physical_counter_payload_sha256") != canonical_sha256(counter_payload):
                errors.append(f"{prefix} {kind} receipt does not bind normalized counter payload")
    if set(spec_by_runtime) != set(RUNTIME_PATHS):
        errors.append("spec physical evidence must cover stored/compact/hashed/computed exactly once")
    return errors


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest() -> int:
    assert validate_gate({})
    req = requirements()
    assert req["default_off"] is True and req["execution_capability"] is False
    print("appendix_physical_evidence_gate.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", action="store_true")
    parser.add_argument("--validate", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.requirements:
        print(json.dumps(requirements(), indent=2, sort_keys=True))
        return 0
    if args.selftest:
        return _selftest()
    if args.validate is not None:
        errors = validate_gate(_load(args.validate))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    parser.error("choose --requirements, --validate PATH, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
