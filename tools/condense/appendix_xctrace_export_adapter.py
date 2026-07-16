#!/usr/bin/env python3.12
"""Fail-closed xctrace Metal export adapter for Appendix release probes.

The trusted counter normalizer accepts canonical JSON, while ``xctrace record``
produces a ``.trace`` package.  This adapter closes that boundary without
guessing table aliases, columns, units, clocks, or process/device attribution.
Production export requires an immutable operator-reviewed profile tied to one
exact full-Xcode binary/version, Metal System Trace TOC fingerprint, export
XPath, XML/plist schema fingerprint, and exact column selectors.

Synthetic documents may exercise the deterministic parser through Python APIs,
but their receipts are permanently marked ineligible for physical evidence.
No production profile is shipped by this module.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import pathlib
import plistlib
import re
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Sequence

import appendix_contract
import appendix_physical_counter_authority as authority_root
import appendix_physical_counter_collector as counter_collector
import appendix_physical_counter_normalizer as trusted_normalizer
import physical_counter_attestation
import ram_scheduler


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_xctrace_export_adapter.v2"
PROFILE_SCHEMA = "hawking.xctrace_metal_export_profile.v2"
RECEIPT_SCHEMA = "hawking.xctrace_export_adapter_receipt.v2"
CAPTURE_SCHEMA = "hawking.direct_metal_process_counter_capture.v1"
BACKEND_ID = "xctrace-metal-system-trace-direct-process-v1"
FULL_XCODE_XCTRACE = pathlib.Path(
    "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"
)
DEFAULT_PROFILE = ROOT / "tools/condense/profiles/appendix_xctrace_metal_export_profile.json"
PROFILE_SSHSIG_NAMESPACE = "hawking-xctrace-export-profile-v2"
REVIEW_SCHEMA = "hawking.xctrace_profile_review_envelope.v2"
REVIEW_ATTESTATION_SCHEMA = "hawking.xctrace_profile_review_attestation.v1"
MAX_DOCUMENT_BYTES = 512 * 1024 * 1024
MAX_TRACE_BYTES = 512 * 1024 * 1024
MAX_PROFILE_VALIDITY_NS = 7 * 24 * 60 * 60 * 1_000_000_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")

TABLE_COLUMNS = {
    "signposts": {
        "signpost_event_id": ("identifier", "string"),
        "signpost_name": ("identifier", "string"),
        "signpost_payload": ("identifier", "string"),
        "signpost_type": ("identifier", "string"),
        "signpost_id": ("count", "integer"),
        "signpost_timestamp_continuous_ns": ("mach_continuous_nanosecond", "integer"),
        "process_id": ("count", "integer"),
    },
    "command_buffers": {
        "command_buffer_id": ("identifier", "string"),
        "command_buffer_label": ("identifier", "string"),
        "process_id": ("count", "integer"),
        "metal_registry_id": ("identifier", "string"),
    },
    "encoders": {
        "encoder_id": ("identifier", "string"),
        "command_buffer_id": ("identifier", "string"),
        "encoder_label": ("identifier", "string"),
        "process_id": ("count", "integer"),
        "metal_registry_id": ("identifier", "string"),
    },
    "counters": {
        "source_event_id": ("identifier", "string"),
        "command_buffer_id": ("identifier", "string"),
        "encoder_id": ("identifier", "string"),
        "process_id": ("count", "integer"),
        "metal_registry_id": ("identifier", "string"),
        "gpu_time_ns": ("nanosecond", "integer"),
        "physical_bytes": ("byte", "integer"),
        "occupancy_percent": ("percent", "float"),
        "skipped": ("boolean", "boolean"),
    },
}
REQUIRED_TABLES = tuple(TABLE_COLUMNS)
AGGREGATION_RULE = {
    "mode": "phase-multi-event-direct-counter-v1",
    "all_table_rows_must_be_consumed": True,
    "minimum_counter_events_per_interval": 1,
    "gpu_time_ns": "sum-direct-counter-rows",
    "physical_bytes": "sum-direct-counter-rows",
    "occupancy_percent": "gpu-time-weighted-mean-direct-counter-rows",
    "bandwidth_bytes_per_second": (
        "sum-physical-bytes-divided-by-sum-gpu-time-ns-times-1000000000"
    ),
    "source_ids": {
        "signposts": ["signpost_event_id"],
        "command_buffers": ["command_buffer_id"],
        "encoders": ["encoder_id", "command_buffer_id"],
        "counters": ["source_event_id", "encoder_id", "command_buffer_id"],
    },
}
TRACE_BOUND_COLUMNS = {
    "capture_started_at_unix_ns": ("nanosecond_since_unix_epoch", "integer"),
    "capture_ended_at_unix_ns": ("nanosecond_since_unix_epoch", "integer"),
    "capture_started_at_continuous_ns": ("mach_continuous_nanosecond", "integer"),
    "capture_ended_at_continuous_ns": ("mach_continuous_nanosecond", "integer"),
}
PINNED_EXPORT_ENV = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "HOME": "/var/empty",
    "LANG": "C",
    "LC_ALL": "C",
    "TMPDIR": "/tmp",
}
LABEL_COMPONENT = r"[A-Za-z0-9_.-]+"
BASE_LABEL_RE = re.compile(
    r"^hawking\.physical\.v1\|interval_id=(?P<interval_id>[0-9a-f]{64})"
    r"\|run_nonce=(?P<run_nonce>[0-9a-f]{64})"
    rf"\|phase=(?P<phase>{LABEL_COMPONENT})"
    rf"\|role=(?P<role>{LABEL_COMPONENT})"
    r"\|batch=(?P<batch>none|0|[1-9][0-9]*)"
    r"\|iteration=(?P<iteration>0|[1-9][0-9]*)$"
)
COMMAND_LABEL_RE = re.compile(
    BASE_LABEL_RE.pattern[:-1]
    + r"\|kind=command_buffer\|command_index=(?P<command_index>0|[1-9][0-9]*)$"
)
ENCODER_LABEL_RE = re.compile(
    BASE_LABEL_RE.pattern[:-1]
    + r"\|kind=(?P<encoder_kind>compute_encoder|blit_encoder)"
    + r"\|command_index=(?P<command_index>0|[1-9][0-9]*)"
    + r"\|encoder_index=(?P<encoder_index>0|[1-9][0-9]*)"
    + rf"\|kernel=(?P<kernel>{LABEL_COMPONENT})$"
)


class XctraceAdapterError(ValueError):
    """A trace, profile, export, or exact phase join is not admissible."""


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def contract() -> dict[str, Any]:
    return _stamp({
        "schema": SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "receipt_schema": RECEIPT_SCHEMA,
        "capture_schema": CAPTURE_SCHEMA,
        "backend_id": BACKEND_ID,
        "full_xcode_binary": str(FULL_XCODE_XCTRACE),
        "template": "Metal System Trace",
        "production_commands": [
            "xctrace version",
            "xctrace export --input TRACE --toc --output TOC",
            "one xctrace export --input TRACE --xpath SIGNED_TABLE_XPATH --output TABLE per signed table",
        ],
        "required_tables": {
            table: {
                field: {"unit": unit, "value_type": value_type}
                for field, (unit, value_type) in columns.items()
            }
            for table, columns in TABLE_COLUMNS.items()
        },
        "trace_bound_columns": {
            field: {"unit": unit, "value_type": value_type}
            for field, (unit, value_type) in TRACE_BOUND_COLUMNS.items()
        },
        "profile_signature_namespace": PROFILE_SSHSIG_NAMESPACE,
        "pinned_export_environment": PINNED_EXPORT_ENV,
        "invariants": [
            "one unexpired operator-SSHSIG profile; no runtime aliases",
            "exact full-Xcode binary/version/template/TOC/export fingerprints",
            "predeclared interval identity joins exact signpost, command-buffer, encoder, and counter tables",
            "N-event direct aggregation and total row consumption in every signed table",
            "exact probe PID and Metal registry ID on every exported row",
            "no timestamp-enclosure-only attribution, interpolation, estimation, or apportionment",
            "capture bounds come directly from reviewed trace TOC fields",
            "ordered hierarchy, multiplicity, and normalized id/ref edges are fingerprinted",
            "raw trace tree, TOC, export, profile, bundle, and capture are hash-bound",
        ],
        "synthetic_fixture_physical_credit": 0,
        "default_off": True,
        "physical_evidence_claimed": False,
    }, "contract_sha256")


CONTRACT_SHA256 = contract()["contract_sha256"]


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise XctraceAdapterError(f"{label} fields are incomplete or unexpected")
    return value


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise XctraceAdapterError(f"{label} must be lowercase SHA-256")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise XctraceAdapterError(f"{label} must be a non-empty string")
    return value


def _identity_errors(value: Any, label: str) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        return [f"{label} must contain exact path/sha256/size_bytes"]
    errors = []
    if not isinstance(value.get("path"), str) or not pathlib.Path(value["path"]).is_absolute():
        errors.append(f"{label}.path must be absolute")
    if not isinstance(value.get("sha256"), str) or HEX64.fullmatch(value["sha256"]) is None:
        errors.append(f"{label}.sha256 is invalid")
    if isinstance(value.get("size_bytes"), bool) or not isinstance(value.get("size_bytes"), int) \
            or value["size_bytes"] <= 0:
        errors.append(f"{label}.size_bytes must be positive")
    return errors


def _reviewed_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile["profile_id"],
        "production_approved": profile["production_approved"],
        "synthetic_fixture": profile["synthetic_fixture"],
        "xctrace": profile["xctrace"],
        "toc": profile["toc"],
        "tables": profile["tables"],
    }


def _review_signed_payload(review: Mapping[str, Any]) -> dict[str, Any]:
    """Exact expiring attestation covered by SSHSIG (never the profile alone)."""
    return {
        "schema": REVIEW_ATTESTATION_SCHEMA,
        "profile_payload_sha256": review["profile_payload_sha256"],
        "issued_at_unix_ns": review["issued_at_unix_ns"],
        "expires_at_unix_ns": review["expires_at_unix_ns"],
        "signer_identity": review["signer_identity"],
        "signature_namespace": review["signature_namespace"],
        "allowed_signers": review["allowed_signers"],
    }


def _verify_profile_signature(envelope: dict[str, Any], payload: bytes) -> tuple[bool, str]:
    try:
        registry = authority_root.load_default_registry()
        with authority_root.pinned_verification_material(
            envelope, registry,
        ) as (allowed, signature):
            process = subprocess.run(
                [
                    str(authority_root.SSH_KEYGEN), "-Y", "verify", "-f", str(allowed),
                    "-I", authority_root.SIGNER_IDENTITY, "-n", PROFILE_SSHSIG_NAMESPACE,
                    "-s", str(signature),
                ],
                cwd=ROOT, env=dict(PINNED_EXPORT_ENV), input=payload,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
                check=False, shell=False,
            )
    except (OSError, subprocess.TimeoutExpired, authority_root.AuthorityError) as exc:
        return False, f"SSHSIG verifier failed: {exc}"
    detail = (process.stdout or process.stderr).decode("utf-8", "replace").strip()
    return process.returncode == 0, detail[-1000:]


def validate_profile(
    value: Any, *, production: bool, now_unix_ns: int | None = None,
    signature_verifier: Callable[[dict[str, Any], bytes], tuple[bool, str]] = (
        _verify_profile_signature
    ),
) -> dict[str, Any]:
    """Validate the exact immutable profile; never infer missing selectors."""
    profile = dict(_exact_fields(value, {
        "schema", "profile_id", "production_approved", "synthetic_fixture",
        "operator_review", "xctrace", "toc", "tables", "profile_sha256",
    }, "xctrace export profile"))
    if profile["schema"] != PROFILE_SCHEMA:
        raise XctraceAdapterError("xctrace export profile schema is invalid")
    unstamped = copy.deepcopy(profile)
    claimed = unstamped.pop("profile_sha256")
    _hex64(claimed, "profile_sha256")
    if canonical_sha256(unstamped) != claimed:
        raise XctraceAdapterError("xctrace export profile self-hash mismatch")
    _nonempty(profile["profile_id"], "profile_id")
    if not isinstance(profile["production_approved"], bool) \
            or not isinstance(profile["synthetic_fixture"], bool):
        raise XctraceAdapterError("profile production/synthetic flags must be booleans")
    if production and (
        profile["production_approved"] is not True
        or profile["synthetic_fixture"] is not False
    ):
        raise XctraceAdapterError("production export requires a non-synthetic approved profile")

    review = dict(_exact_fields(profile["operator_review"], {
        "schema", "profile_payload_sha256", "issued_at_unix_ns",
        "expires_at_unix_ns", "signer_identity", "signature_namespace",
        "allowed_signers", "detached_signature", "envelope_sha256",
    }, "operator review"))
    unstamped_review = copy.deepcopy(review)
    claimed_review = unstamped_review.pop("envelope_sha256")
    if _hex64(claimed_review, "operator review envelope hash") \
            != canonical_sha256(unstamped_review):
        raise XctraceAdapterError("operator review envelope self-hash mismatch")
    if review["schema"] != REVIEW_SCHEMA \
            or review["signer_identity"] != authority_root.SIGNER_IDENTITY \
            or review["signature_namespace"] != PROFILE_SSHSIG_NAMESPACE:
        raise XctraceAdapterError("operator review signer/schema/namespace is invalid")
    if _hex64(review["profile_payload_sha256"], "reviewed payload hash") \
            != canonical_sha256(_reviewed_payload(profile)):
        raise XctraceAdapterError("operator review does not bind the exact profile payload")
    issued, expires = review["issued_at_unix_ns"], review["expires_at_unix_ns"]
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0 \
            or isinstance(expires, bool) or not isinstance(expires, int) or expires <= issued:
        raise XctraceAdapterError("operator review validity interval is invalid")
    if expires - issued > MAX_PROFILE_VALIDITY_NS:
        raise XctraceAdapterError("operator review validity exceeds seven days")
    errors = _identity_errors(review["allowed_signers"], "operator allowed_signers")
    errors.extend(_identity_errors(review["detached_signature"], "operator signature"))
    if errors:
        raise XctraceAdapterError("; ".join(errors))
    if production:
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        if not issued <= now <= expires:
            raise XctraceAdapterError("operator-reviewed profile is expired or not yet valid")
        try:
            pinned = authority_root.allowed_signers_identity(
                authority_root.load_default_registry()
            )
        except (OSError, ValueError, authority_root.AuthorityError) as exc:
            raise XctraceAdapterError(f"pinned profile trust root is invalid: {exc}") from exc
        if review["allowed_signers"] != pinned:
            raise XctraceAdapterError("profile review does not use the pinned operator trust root")
        ok, detail = signature_verifier(
            review, appendix_contract.canonical_bytes(_review_signed_payload(review)),
        )
        if not ok:
            raise XctraceAdapterError(
                "operator profile SSHSIG verification failed" + (f": {detail}" if detail else "")
            )

    xctrace = _exact_fields(profile["xctrace"], {
        "binary", "version_argv", "version_string", "version_stdout_sha256",
        "template_name", "environment",
    }, "profile.xctrace")
    errors = _identity_errors(xctrace["binary"], "profile.xctrace.binary")
    if errors:
        raise XctraceAdapterError("; ".join(errors))
    binary_path = xctrace["binary"]["path"]
    if xctrace["version_argv"] != [binary_path, "version"]:
        raise XctraceAdapterError("profile version argv is not exact")
    _nonempty(xctrace["version_string"], "profile xctrace version")
    _hex64(xctrace["version_stdout_sha256"], "version stdout hash")
    if xctrace["template_name"] != "Metal System Trace":
        raise XctraceAdapterError("profile template is not Metal System Trace")
    if xctrace["environment"] != PINNED_EXPORT_ENV:
        raise XctraceAdapterError("profile xctrace environment differs from the pinned map")

    toc = _exact_fields(profile["toc"], {
        "document_format", "schema_fingerprint_sha256",
        "fingerprint_value_attributes", "template_xpath", "template_value_source",
        "capture_bounds",
    }, "profile.toc")
    tables = _exact_fields(profile["tables"], {"exports", "aggregation"}, "profile.tables")
    if tables["aggregation"] != AGGREGATION_RULE:
        raise XctraceAdapterError("profile aggregation rule is not the exact N-event rule")
    exports = _exact_fields(
        tables["exports"], set(REQUIRED_TABLES), "profile.tables.exports",
    )
    for table_name in REQUIRED_TABLES:
        _exact_fields(exports[table_name], {
            "document_format", "xpath", "row_xpath", "selector_mode",
            "schema_columns_xpath", "schema_mnemonic_xpath", "schema_unit_xpath",
            "fingerprint_value_attributes", "schema_fingerprint_sha256", "columns",
        }, f"profile table {table_name}")
    for label, row in (("toc", toc), *exports.items()):
        if row["document_format"] not in {"xml", "plist"}:
            raise XctraceAdapterError(f"profile {label} document format is invalid")
        _hex64(row["schema_fingerprint_sha256"], f"profile {label} schema fingerprint")
        attrs = row["fingerprint_value_attributes"]
        if not isinstance(attrs, list) or any(not isinstance(item, str) or not item for item in attrs) \
                or len(set(attrs)) != len(attrs):
            raise XctraceAdapterError(f"profile {label} fingerprint attributes are invalid")
    _nonempty(toc["template_xpath"], "TOC template XPath")
    if toc["document_format"] == "xml" and not {"id", "ref"}.issubset(
        toc["fingerprint_value_attributes"]
    ):
        raise XctraceAdapterError("TOC fingerprint must normalize exact id/ref edges")
    if toc["template_value_source"] != "text" and not (
        isinstance(toc["template_value_source"], str)
        and toc["template_value_source"].startswith("attribute:")
        and toc["template_value_source"].split(":", 1)[1]
    ) and toc["template_value_source"] != "plist-scalar":
        raise XctraceAdapterError("TOC template value source is invalid")
    observed_xpaths: set[str] = set()
    for table_name, export in exports.items():
        xpath = _nonempty(export["xpath"], f"{table_name} xctrace export XPath")
        if not xpath.startswith("/") or xpath in observed_xpaths:
            raise XctraceAdapterError("every table requires one unique absolute xctrace XPath")
        observed_xpaths.add(xpath)
        _nonempty(export["row_xpath"], f"{table_name} row XPath")
        if export["selector_mode"] != "positional-xctrace-xml-v1" \
                or export["document_format"] != "xml":
            raise XctraceAdapterError("table must use exact positional xctrace XML selectors")
        for selector_name in (
            "schema_columns_xpath", "schema_mnemonic_xpath", "schema_unit_xpath",
        ):
            _nonempty(export[selector_name], f"{table_name}.{selector_name}")
        if export["document_format"] == "xml" and not {"id", "ref"}.issubset(
            export["fingerprint_value_attributes"]
        ):
            raise XctraceAdapterError("table fingerprint must normalize exact id/ref edges")
        columns = _exact_fields(
            export["columns"], set(TABLE_COLUMNS[table_name]),
            f"profile table {table_name} columns",
        )
        observed_ids: set[str] = set()
        for field, (unit, value_type) in TABLE_COLUMNS[table_name].items():
            spec = _exact_fields(columns[field], {
                "column_id", "schema_index", "row_child_index", "value_xpath",
                "value_source", "raw_unit", "unit", "value_type",
                "scale_numerator", "scale_denominator",
            }, f"profile {table_name} column {field}")
            column_id = _nonempty(spec["column_id"], f"{table_name}.{field}.column_id")
            if column_id in observed_ids:
                raise XctraceAdapterError(f"profile {table_name} column IDs are reused")
            observed_ids.add(column_id)
            for index_field in ("schema_index", "row_child_index"):
                index = spec[index_field]
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise XctraceAdapterError(f"{table_name}.{field}.{index_field} is invalid")
            if spec["schema_index"] != spec["row_child_index"]:
                raise XctraceAdapterError("schema and row positional indexes must be identical")
            _nonempty(spec["value_xpath"], f"{table_name}.{field}.value_xpath")
            if spec["value_source"] != "text":
                raise XctraceAdapterError(
                    f"{table_name}.{field} must use resolved XML text"
                )
            if spec["unit"] != unit or spec["value_type"] != value_type:
                raise XctraceAdapterError(
                    f"{table_name}.{field} unit/type differs from the pinned contract"
                )
            _nonempty(spec["raw_unit"], f"{table_name}.{field}.raw_unit")
            numerator, denominator = spec["scale_numerator"], spec["scale_denominator"]
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in (numerator, denominator)):
                raise XctraceAdapterError("raw-unit conversion ratio must be positive integers")
            if value_type in {"string", "boolean"} and (numerator, denominator) != (1, 1):
                raise XctraceAdapterError("identifier/boolean columns cannot be numerically scaled")
    bounds = _exact_fields(toc["capture_bounds"], set(TRACE_BOUND_COLUMNS), "TOC capture bounds")
    observed_bound_ids: set[str] = set()
    for field, (unit, value_type) in TRACE_BOUND_COLUMNS.items():
        spec = _exact_fields(bounds[field], {
            "column_id", "xpath", "value_source", "unit", "value_type",
        }, f"TOC bound {field}")
        column_id = _nonempty(spec["column_id"], f"TOC {field}.column_id")
        if column_id in observed_bound_ids:
            raise XctraceAdapterError("TOC capture-bound column IDs are reused")
        observed_bound_ids.add(column_id)
        _nonempty(spec["xpath"], f"TOC {field}.xpath")
        expected_source = "text" if toc["document_format"] == "xml" else "plist-value"
        if spec["value_source"] != expected_source \
                or spec["unit"] != unit or spec["value_type"] != value_type:
            raise XctraceAdapterError(f"TOC {field} selector/unit/type is invalid")
    return copy.deepcopy(profile)


def _read_regular(path: pathlib.Path, *, maximum: int = MAX_DOCUMENT_BYTES) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.absolute(), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > maximum:
            raise XctraceAdapterError(f"unsafe, empty, or oversized input: {path}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum:
                raise XctraceAdapterError(f"input exceeds bounded size: {path}")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or observed != after.st_size:
            raise XctraceAdapterError(f"input changed while reading: {path}")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    return raw, {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _read_json(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, identity = _read_regular(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise XctraceAdapterError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise XctraceAdapterError(f"JSON input must be an object: {path}")
    return value, identity


def seal_profile_draft(
    *, draft_path: pathlib.Path, private_key: pathlib.Path,
    public_key: pathlib.Path, detached_signature_output: pathlib.Path,
    profile_output: pathlib.Path, validity_hours: int = 24,
    now_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Type-check, sign, seal, and re-verify one reviewed production profile."""
    if isinstance(validity_hours, bool) or not isinstance(validity_hours, int) \
            or not 1 <= validity_hours <= 168:
        raise XctraceAdapterError("profile validity must be an integer from 1 through 168 hours")
    draft, _identity = _read_json(draft_path)
    expected_draft_fields = {
        "schema", "profile_id", "production_approved", "synthetic_fixture",
        "xctrace", "toc", "tables",
    }
    if set(draft) != expected_draft_fields or draft.get("schema") != PROFILE_SCHEMA \
            or draft.get("production_approved") is not True \
            or draft.get("synthetic_fixture") is not False:
        raise XctraceAdapterError("profile draft fields/production flags are invalid")
    registry = authority_root.load_default_registry()
    registry_errors = authority_root.validate_registry(
        registry, verify_files=True, require_default=True,
    )
    custody = authority_root.signing_key_custody_status(
        private_key=private_key, public_key=public_key, registry=registry,
    )
    if registry_errors or not custody["signing_key_available"]:
        raise XctraceAdapterError(
            "profile signer custody/registry is invalid: "
            + "; ".join([*registry_errors, *custody["problems"]])
        )
    issued = time.time_ns() if now_unix_ns is None else now_unix_ns
    expires = issued + validity_hours * 60 * 60 * 1_000_000_000
    allowed_signers = authority_root.allowed_signers_identity(registry)
    profile_payload_sha256 = canonical_sha256({
        "profile_id": draft["profile_id"],
        "production_approved": draft["production_approved"],
        "synthetic_fixture": draft["synthetic_fixture"],
        "xctrace": draft["xctrace"],
        "toc": draft["toc"],
        "tables": draft["tables"],
    })
    unsigned_review = {
        "schema": REVIEW_SCHEMA,
        "profile_payload_sha256": profile_payload_sha256,
        "issued_at_unix_ns": issued,
        "expires_at_unix_ns": expires,
        "signer_identity": authority_root.SIGNER_IDENTITY,
        "signature_namespace": PROFILE_SSHSIG_NAMESPACE,
        "allowed_signers": allowed_signers,
        "detached_signature": {
            "path": str(detached_signature_output), "sha256": "0" * 64, "size_bytes": 1,
        },
        "envelope_sha256": "0" * 64,
    }
    provisional = dict(draft)
    provisional["operator_review"] = _stamp(unsigned_review, "envelope_sha256")
    provisional = _stamp(provisional, "profile_sha256")
    validate_profile(provisional, production=False)
    signed_payload = appendix_contract.canonical_bytes(
        _review_signed_payload(unsigned_review),
    )
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hawking-xctrace-profile-sign-") as directory:
        message = pathlib.Path(directory) / "profile-review.canonical.json"
        message.write_bytes(signed_payload)
        process = subprocess.run(
            [
                str(authority_root.SSH_KEYGEN), "-Y", "sign", "-f", str(private_key),
                "-n", PROFILE_SSHSIG_NAMESPACE, str(message),
            ],
            cwd=ROOT, env=dict(PINNED_EXPORT_ENV), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            check=False, shell=False,
        )
        generated = message.with_suffix(message.suffix + ".sig")
        if process.returncode != 0 or not generated.is_file():
            raise XctraceAdapterError(
                f"profile SSHSIG signing failed ({process.returncode}): "
                f"{(process.stderr or process.stdout)[-500:]!r}"
            )
        signature_raw = generated.read_bytes()
    authority_root._atomic_bytes(detached_signature_output, signature_raw, mode=0o444)
    review = dict(unsigned_review)
    review["detached_signature"] = physical_counter_attestation.file_identity(
        detached_signature_output,
    )
    review = _stamp(review, "envelope_sha256")
    profile = dict(draft)
    profile["operator_review"] = review
    profile = _stamp(profile, "profile_sha256")
    validate_profile(profile, production=True, now_unix_ns=issued)
    authority_root._atomic_bytes(
        profile_output,
        (json.dumps(profile, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o444,
    )
    sealed, _sealed_identity = _read_json(profile_output)
    return validate_profile(sealed, production=True, now_unix_ns=issued)


def trace_tree_identity(
    path: pathlib.Path, *, require_immutable: bool = False,
) -> dict[str, Any]:
    """Hash a safe .trace tree without following links or accepting hardlinks."""
    if path.suffix != ".trace" or path.is_symlink() or not path.is_dir():
        raise XctraceAdapterError("raw xctrace input must be a non-symlink .trace directory")
    root_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    if require_immutable and root_mode & 0o222:
        raise XctraceAdapterError("production xctrace root directory remains writable")
    files = []
    total = 0
    for candidate in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix()
        if candidate.is_symlink():
            raise XctraceAdapterError("xctrace package contains a symlink")
        if candidate.is_dir():
            if require_immutable and stat.S_IMODE(
                candidate.stat(follow_symlinks=False).st_mode
            ) & 0o222:
                raise XctraceAdapterError("production xctrace subdirectory remains writable")
            continue
        if not candidate.is_file():
            raise XctraceAdapterError("xctrace package contains a non-file entry")
        metadata = candidate.stat(follow_symlinks=False)
        if metadata.st_nlink != 1:
            raise XctraceAdapterError("xctrace package contains a hard-linked file")
        if require_immutable and stat.S_IMODE(metadata.st_mode) != 0o444:
            raise XctraceAdapterError("production xctrace file is not frozen mode 0444")
        raw, identity = _read_regular(candidate, maximum=MAX_TRACE_BYTES)
        total += len(raw)
        if total > MAX_TRACE_BYTES:
            raise XctraceAdapterError("xctrace package exceeds bounded size")
        files.append({
            "relative_path": relative,
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        })
    if not files:
        raise XctraceAdapterError("xctrace package contains no raw files")
    value = {
        "schema": "hawking.xctrace_trace_tree_identity.v1",
        "path": str(path.absolute()),
        "total_size_bytes": total,
        "files": files,
    }
    value["tree_sha256"] = canonical_sha256(value)
    return value


def _local_name(value: str) -> str:
    if value.startswith("{"):
        raise XctraceAdapterError("namespaced XML requires a different reviewed profile/parser")
    return value


def _load_document(path: pathlib.Path, document_format: str) -> tuple[Any, dict[str, Any]]:
    raw, identity = _read_regular(path)
    try:
        if document_format == "xml":
            if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                raise XctraceAdapterError("DTD/entity-bearing XML is not accepted")
            return ET.fromstring(raw), identity
        if document_format == "plist":
            return plistlib.loads(raw), identity
    except (ET.ParseError, plistlib.InvalidFileException, ValueError) as exc:
        if isinstance(exc, XctraceAdapterError):
            raise
        raise XctraceAdapterError(f"invalid {document_format} export: {exc}") from exc
    raise XctraceAdapterError("unsupported reviewed document format")


def _plist_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "dict", "keys": {key: _plist_shape(value[key]) for key in sorted(value)}}
    if isinstance(value, list):
        return {"type": "list", "items": [_plist_shape(item) for item in value]}
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    if isinstance(value, bytes):
        return "data"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _canonical_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _xml_id_index(document: ET.Element) -> tuple[dict[str, ET.Element], dict[str, int]]:
    identities: dict[str, ET.Element] = {}
    ordinals: dict[str, int] = {}
    for element in document.iter():
        identifier = element.attrib.get("id")
        if identifier is None:
            continue
        if not identifier or identifier in identities:
            raise XctraceAdapterError("XML id is empty or duplicated")
        ordinals[identifier] = len(ordinals)
        identities[identifier] = element
    for element in document.iter():
        reference = element.attrib.get("ref")
        if reference is not None and reference not in identities:
            raise XctraceAdapterError("XML ref does not resolve to an exact id")
    for identifier, element in identities.items():
        seen = {identifier}
        current = element
        while "ref" in current.attrib:
            target = current.attrib["ref"]
            if target in seen:
                raise XctraceAdapterError("XML id/ref graph contains a cycle")
            seen.add(target)
            current = identities[target]
    return identities, ordinals


def _resolved_xml_element(element: ET.Element, identities: Mapping[str, ET.Element]) -> ET.Element:
    current = element
    seen: set[str] = set()
    while "ref" in current.attrib:
        reference = current.attrib["ref"]
        if reference in seen or reference not in identities:
            raise XctraceAdapterError("XML ref is cyclic or unresolved")
        seen.add(reference)
        current = identities[reference]
    if len(current) == 1 and ("ref" in current[0].attrib or "id" in current[0].attrib):
        return _resolved_xml_element(current[0], identities)
    return current


def schema_fingerprint(document: Any, document_format: str, value_attributes: Sequence[str]) -> str:
    """Fingerprint full ordered hierarchy/multiplicity and normalized id/ref edges."""
    if document_format == "plist":
        return canonical_sha256({"format": "plist", "shape": _plist_shape(document)})
    if document_format != "xml" or not isinstance(document, ET.Element):
        raise XctraceAdapterError("schema fingerprint document/format mismatch")
    identities, ordinals = _xml_id_index(document)

    def shape(element: ET.Element) -> dict[str, Any]:
        attributes = {_local_name(key): value for key, value in element.attrib.items()}
        reviewed = {}
        for key in value_attributes:
            if key not in attributes:
                continue
            value = attributes[key]
            if key in {"id", "ref"}:
                value = f"node-{ordinals[value]}"
            reviewed[key] = value
        return {
            "tag": _local_name(element.tag),
            "attribute_names": sorted(attributes),
            "reviewed_attribute_values": reviewed,
            "children": [shape(child) for child in element],
        }
    return canonical_sha256({
        "format": "xml",
        "hierarchy": shape(document),
        "id_ref_edges": [
            {"source_ordinal": index, "target_ordinal": ordinals[element.attrib["ref"]]}
            for index, element in enumerate(document.iter()) if "ref" in element.attrib
        ],
    })


def _plist_path(value: Any, path: str) -> Any:
    current = value
    for component in [item for item in path.split("/") if item]:
        if not isinstance(current, dict) or component not in current:
            raise XctraceAdapterError(f"reviewed plist path is absent: {path}")
        current = current[component]
    return current


def _validate_document(
    document: Any, section: Mapping[str, Any], *, template: str | None = None,
) -> str:
    observed = schema_fingerprint(
        document, section["document_format"], section["fingerprint_value_attributes"],
    )
    if observed != section["schema_fingerprint_sha256"]:
        raise XctraceAdapterError("xctrace document schema fingerprint differs from profile")
    if template is None:
        return observed
    if section["document_format"] == "xml":
        identities, _ordinals = _xml_id_index(document)
        rows = document.findall(section["template_xpath"])
        if len(rows) != 1:
            raise XctraceAdapterError("TOC template XPath is absent or ambiguous")
        selected = _resolved_xml_element(rows[0], identities)
        source = section["template_value_source"]
        if source == "text":
            observed_template = selected.text
        else:
            observed_template = selected.attrib.get(source.split(":", 1)[1])
    else:
        if section["template_value_source"] != "plist-scalar":
            raise XctraceAdapterError("plist TOC requires plist-scalar template source")
        observed_template = _plist_path(document, section["template_xpath"])
    if observed_template != template:
        raise XctraceAdapterError("TOC does not identify the exact reviewed template")
    return observed


def _capture_bounds(document: Any, toc: Mapping[str, Any]) -> dict[str, int]:
    output = {}
    identities = _xml_id_index(document)[0] if toc["document_format"] == "xml" else {}
    for field, spec in toc["capture_bounds"].items():
        if toc["document_format"] == "xml":
            matches = document.findall(spec["xpath"])
            if len(matches) != 1:
                raise XctraceAdapterError(f"TOC exact capture bound is absent or ambiguous: {field}")
            selected = _resolved_xml_element(matches[0], identities)
            unit = matches[0].attrib.get("unit", selected.attrib.get("unit"))
            if unit != spec["unit"]:
                raise XctraceAdapterError(f"TOC capture bound unit differs: {field}")
            raw = selected.text
        else:
            selected = _plist_path(document, spec["xpath"])
            if not isinstance(selected, dict) or set(selected) != {"value", "unit"} \
                    or selected["unit"] != spec["unit"]:
                raise XctraceAdapterError(f"TOC plist capture bound differs: {field}")
            raw = selected["value"]
        output[field] = _convert_scalar(raw, spec["value_type"], f"TOC {field}")
    if not (
        0 < output["capture_started_at_unix_ns"] < output["capture_ended_at_unix_ns"]
        and 0 < output["capture_started_at_continuous_ns"]
        < output["capture_ended_at_continuous_ns"]
    ):
        raise XctraceAdapterError("trace-derived capture bounds are invalid")
    return output


def _convert_scalar(value: Any, value_type: str, label: str) -> Any:
    if value_type == "string":
        return _nonempty(value, label)
    if value_type == "boolean":
        if isinstance(value, bool):
            return value
        if value == "true":
            return True
        if value == "false":
            return False
        raise XctraceAdapterError(f"{label} is not an exact boolean")
    if value_type == "integer":
        if isinstance(value, bool):
            raise XctraceAdapterError(f"{label} is not an integer")
        if isinstance(value, int):
            return value
        if not isinstance(value, str) or INTEGER.fullmatch(value) is None:
            raise XctraceAdapterError(f"{label} is not a canonical integer")
        return int(value)
    if value_type == "float":
        if isinstance(value, bool):
            raise XctraceAdapterError(f"{label} is not finite")
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise XctraceAdapterError(f"{label} is not a canonical real") from exc
        if not parsed.is_finite():
            raise XctraceAdapterError(f"{label} is not finite")
        output = float(parsed)
        if not math.isfinite(output):
            raise XctraceAdapterError(f"{label} overflows finite float")
        return output
    raise XctraceAdapterError(f"{label} uses an unsupported value type")


def _parse_rows(document: Any, export: Mapping[str, Any]) -> list[dict[str, Any]]:
    if export["document_format"] != "xml" \
            or export["selector_mode"] != "positional-xctrace-xml-v1":
        raise XctraceAdapterError("only signed positional xctrace XML tables are accepted")
    identities, _ordinals = _xml_id_index(document)
    rows = document.findall(export["row_xpath"])
    if not rows:
        raise XctraceAdapterError("xctrace export contains no reviewed rows")
    schema_columns = document.findall(export["schema_columns_xpath"])
    if not schema_columns:
        raise XctraceAdapterError("xctrace export lacks its signed positional schema columns")

    def relative(node: ET.Element, selector: str, label: str) -> ET.Element:
        node = _resolved_xml_element(node, identities)
        if selector == ".":
            return node
        matches = node.findall(selector)
        if len(matches) != 1:
            raise XctraceAdapterError(f"{label} is absent or ambiguous")
        return _resolved_xml_element(matches[0], identities)

    resolved_schema = [_resolved_xml_element(node, identities) for node in schema_columns]
    for field, spec in export["columns"].items():
        index = spec["schema_index"]
        if index >= len(resolved_schema):
            raise XctraceAdapterError(f"schema index for {field} is out of range")
        schema_node = resolved_schema[index]
        mnemonic = relative(
            schema_node, export["schema_mnemonic_xpath"], f"schema mnemonic for {field}",
        ).text
        raw_unit = relative(
            schema_node, export["schema_unit_xpath"], f"schema unit for {field}",
        ).text
        if mnemonic != spec["column_id"] or raw_unit != spec["raw_unit"]:
            raise XctraceAdapterError(
                f"schema position/mnemonic/raw unit differs for {field}"
            )
    output = []
    for ordinal, row in enumerate(rows):
        resolved_row = _resolved_xml_element(row, identities)
        cells = list(resolved_row)
        if len(cells) != len(resolved_schema):
            raise XctraceAdapterError(
                f"row {ordinal} positional multiplicity differs from signed schema"
            )
        parsed: dict[str, Any] = {}
        for field, spec in export["columns"].items():
            cell = _resolved_xml_element(cells[spec["row_child_index"]], identities)
            selected = relative(
                cell, spec["value_xpath"], f"row {ordinal} value for {field}",
            )
            value = _convert_scalar(selected.text, spec["value_type"], f"row {ordinal} {field}")
            numerator, denominator = spec["scale_numerator"], spec["scale_denominator"]
            if spec["value_type"] == "integer":
                scaled = value * numerator
                if scaled % denominator:
                    raise XctraceAdapterError(
                        f"row {ordinal} {field} raw-unit conversion is not exact"
                    )
                value = scaled // denominator
            elif spec["value_type"] == "float":
                value = float(Decimal(str(value)) * numerator / denominator)
                if not math.isfinite(value):
                    raise XctraceAdapterError(f"row {ordinal} {field} scaling overflowed")
            parsed[field] = value
        parsed["row_sha256"] = canonical_sha256(parsed)
        output.append(parsed)
    return output


def _bundle_targets(
    bundle: Mapping[str, Any], *, kind: str, run_nonce: str, probe_argv_sha256: str,
) -> list[dict[str, Any]]:
    if kind not in {"device", "spec"}:
        raise XctraceAdapterError("kind must be device or spec")
    claimed = bundle.get("raw_bundle_sha256")
    unstamped = copy.deepcopy(bundle)
    unstamped.pop("raw_bundle_sha256", None)
    if not isinstance(claimed, str) or claimed != canonical_sha256(unstamped):
        raise XctraceAdapterError("raw bundle self-hash mismatch")
    authority = bundle.get("execution_authority", {})
    if authority.get("run_nonce") != run_nonce or authority.get("argv_sha256") != probe_argv_sha256:
        raise XctraceAdapterError("raw bundle differs from exact nonce/probe argv")
    targets, errors = counter_collector._phase_targets(bundle, kind)
    if errors:
        raise XctraceAdapterError("raw bundle trial targets are invalid: " + "; ".join(errors))
    if not targets:
        raise XctraceAdapterError("raw bundle has no exact trial targets")
    pairs = {
        row.get("phase_marker_sha256"): row
        for row in bundle.get("raw_probe", {}).get("phase_markers", {}).get("pairs", [])
        if isinstance(row, dict) and row.get("phase") == "trial"
    }
    id_field = "candidate_interval_id" if kind == "device" else "verifier_interval_id"
    for target in targets:
        interval_id = target.get("interval", {}).get("interval_id")
        pair = pairs.get(target.get("marker"))
        if not isinstance(interval_id, str) or HEX64.fullmatch(interval_id) is None:
            raise XctraceAdapterError("trial interval lacks a stable predeclared interval id")
        if not isinstance(pair, dict) or pair.get(id_field) != interval_id:
            raise XctraceAdapterError("phase marker does not bind its exact stable interval id")
        target["interval_id"] = interval_id
    return targets


def _bundle_interval_population(
    bundle: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    target_by_hash = {target["interval_sha256"]: target for target in targets}
    intervals = bundle.get("raw_probe", {}).get("phase_markers", {}).get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise XctraceAdapterError("raw bundle has no physical interval population")
    output = []
    seen_ids: set[str] = set()
    for ordinal, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise XctraceAdapterError("raw interval population contains a non-object")
        interval_id = interval.get("interval_id")
        if not isinstance(interval_id, str) or HEX64.fullmatch(interval_id) is None \
                or interval_id in seen_ids:
            raise XctraceAdapterError("raw interval stable id is absent, invalid, or reused")
        seen_ids.add(interval_id)
        target = target_by_hash.get(interval.get("interval_sha256"))
        output.append({
            "interval_id": interval_id,
            "interval_sha256": interval.get("interval_sha256"),
            "phase": interval.get("phase"),
            "role": interval.get("role"),
            "batch": interval.get("batch"),
            "iteration": interval.get("iteration"),
            "interval": interval,
            "target": target,
            "ordinal": ordinal,
        })
    if {row["interval_sha256"] for row in output if row["target"] is not None} \
            != set(target_by_hash):
        raise XctraceAdapterError("target subset does not resolve inside full interval population")
    return output


def _parse_trace_label(value: Any, pattern: re.Pattern[str], label: str) -> dict[str, Any]:
    text = _nonempty(value, label)
    match = pattern.fullmatch(text)
    if match is None:
        raise XctraceAdapterError(f"{label} does not use the exact Hawking physical label grammar")
    output: dict[str, Any] = match.groupdict()
    output["iteration"] = int(output["iteration"])
    output["batch"] = None if output["batch"] == "none" else int(output["batch"])
    if "command_index" in output:
        output["command_index"] = int(output["command_index"])
    if "encoder_index" in output:
        output["encoder_index"] = int(output["encoder_index"])
    return output


def _expected_signpost_id(interval_id: str) -> int:
    value = int(interval_id[:16], 16)
    return 1 if value in {0, (1 << 64) - 1} else value


def _label_matches_interval(
    parsed: Mapping[str, Any], interval: Mapping[str, Any], *, run_nonce: str,
) -> bool:
    return parsed.get("interval_id") == interval.get("interval_id") \
        and parsed.get("run_nonce") == run_nonce \
        and parsed.get("phase") == interval.get("phase") \
        and parsed.get("role") == interval.get("role") \
        and parsed.get("batch") == interval.get("batch") \
        and parsed.get("iteration") == interval.get("iteration")


def _join_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    intervals: Sequence[Mapping[str, Any]], *, probe_pid: int,
    metal_registry_id: str, run_nonce: str,
) -> list[dict[str, Any]]:
    if isinstance(probe_pid, bool) or not isinstance(probe_pid, int) or probe_pid <= 0:
        raise XctraceAdapterError("probe PID must be positive")
    _nonempty(metal_registry_id, "Metal registry ID")
    if set(tables) != set(REQUIRED_TABLES):
        raise XctraceAdapterError("exact four exported tables are required")
    intervals_by_id = {interval["interval_id"]: interval for interval in intervals}
    if len(intervals_by_id) != len(intervals):
        raise XctraceAdapterError("raw bundle stable interval id is reused")
    expected_signpost_ids = {
        interval_id: _expected_signpost_id(interval_id) for interval_id in intervals_by_id
    }
    if len(set(expected_signpost_ids.values())) != len(expected_signpost_ids):
        raise XctraceAdapterError("raw interval population collides in os_signpost_id_t space")
    for interval_id, interval in intervals_by_id.items():
        if interval["interval"].get("signpost_id") != expected_signpost_ids[interval_id]:
            raise XctraceAdapterError("raw interval does not bind its exact nonreserved signpost ID")
    grouped = {
        interval_id: {
            "target": target, "signposts": [], "command_buffers": [],
            "encoders": [], "counters": [],
        }
        for interval_id, target in intervals_by_id.items()
    }

    seen_signposts: set[str] = set()
    for ordinal, row in enumerate(tables["signposts"]):
        event_id = _nonempty(row["signpost_event_id"], f"signpost row {ordinal} id")
        if event_id in seen_signposts or row["process_id"] != probe_pid:
            raise XctraceAdapterError("signpost ID is reused or differs from exact probe PID")
        seen_signposts.add(event_id)
        if row["signpost_name"] != "HawkingPhysicalPhase":
            raise XctraceAdapterError("signpost static name differs from the public shim")
        parsed = _parse_trace_label(
            row["signpost_payload"], BASE_LABEL_RE, "public signpost payload",
        )
        target = intervals_by_id.get(parsed["interval_id"])
        if target is None or not _label_matches_interval(parsed, target, run_nonce=run_nonce):
            raise XctraceAdapterError("signpost does not exactly join one raw interval identity")
        if row["signpost_type"] not in {"begin", "end"} \
                or row["signpost_id"] != expected_signpost_ids[parsed["interval_id"]] \
                or not isinstance(row["signpost_timestamp_continuous_ns"], int):
            raise XctraceAdapterError("signpost type/id/timestamp differs from the exact interval")
        grouped[parsed["interval_id"]]["signposts"].append(dict(row))

    command_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for ordinal, row in enumerate(tables["command_buffers"]):
        command_id = _nonempty(row["command_buffer_id"], f"command-buffer row {ordinal} id")
        if command_id in command_by_id or row["process_id"] != probe_pid \
                or row["metal_registry_id"] != metal_registry_id:
            raise XctraceAdapterError("command-buffer ID is reused or differs from exact PID/device")
        parsed = _parse_trace_label(
            row["command_buffer_label"], COMMAND_LABEL_RE, "command-buffer label",
        )
        target = intervals_by_id.get(parsed["interval_id"])
        if target is None or not _label_matches_interval(parsed, target, run_nonce=run_nonce):
            raise XctraceAdapterError("command buffer does not exactly join a raw interval")
        enriched = dict(row)
        enriched["parsed_label"] = parsed
        command_by_id[command_id] = (enriched, target)
        grouped[parsed["interval_id"]]["command_buffers"].append(enriched)

    encoder_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    command_references: set[str] = set()
    for ordinal, row in enumerate(tables["encoders"]):
        encoder_id = _nonempty(row["encoder_id"], f"encoder row {ordinal} id")
        command_id = _nonempty(row["command_buffer_id"], f"encoder row {ordinal} command id")
        if encoder_id in encoder_by_id or row["process_id"] != probe_pid \
                or row["metal_registry_id"] != metal_registry_id:
            raise XctraceAdapterError("encoder ID is reused or differs from exact PID/device")
        command_entry = command_by_id.get(command_id)
        if command_entry is None:
            raise XctraceAdapterError("encoder references an unexported command buffer")
        parsed = _parse_trace_label(row["encoder_label"], ENCODER_LABEL_RE, "encoder label")
        command, target = command_entry
        if not _label_matches_interval(parsed, target, run_nonce=run_nonce) \
                or parsed["command_index"] != command["parsed_label"]["command_index"]:
            raise XctraceAdapterError("encoder label disagrees with its exact command buffer")
        enriched = dict(row)
        enriched["parsed_label"] = parsed
        encoder_by_id[encoder_id] = (enriched, target)
        command_references.add(command_id)
        grouped[parsed["interval_id"]]["encoders"].append(enriched)

    seen_events: set[str] = set()
    encoder_references: set[str] = set()
    for ordinal, row in enumerate(tables["counters"]):
        event_id = _nonempty(row["source_event_id"], f"counter row {ordinal} id")
        encoder_id = _nonempty(row["encoder_id"], f"counter row {ordinal} encoder id")
        command_id = _nonempty(row["command_buffer_id"], f"counter row {ordinal} command id")
        if event_id in seen_events or row["process_id"] != probe_pid \
                or row["metal_registry_id"] != metal_registry_id:
            raise XctraceAdapterError("counter source ID is reused or differs from exact PID/device")
        encoder_entry = encoder_by_id.get(encoder_id)
        if encoder_entry is None or encoder_entry[0]["command_buffer_id"] != command_id:
            raise XctraceAdapterError("counter row does not exactly join its encoder and command buffer")
        if row["skipped"] is not False or not (
            isinstance(row["gpu_time_ns"], int) and row["gpu_time_ns"] > 0
            and isinstance(row["physical_bytes"], int) and row["physical_bytes"] > 0
            and 0 <= row["occupancy_percent"] <= 100
        ):
            raise XctraceAdapterError("counter row is skipped or lacks positive direct measurements")
        seen_events.add(event_id)
        encoder_references.add(encoder_id)
        target = encoder_entry[1]
        grouped[target["interval_id"]]["counters"].append(dict(row))

    if command_references != set(command_by_id):
        raise XctraceAdapterError("one or more command-buffer rows were not consumed exactly once")
    if encoder_references != set(encoder_by_id):
        raise XctraceAdapterError("one or more encoder rows were not consumed by direct counters")
    output = []
    for value in grouped.values():
        signposts = value["signposts"]
        signposts_by_type = {row["signpost_type"]: row for row in signposts}
        interval = value["target"]["interval"]
        if len(signposts) != 2 or set(signposts_by_type) != {"begin", "end"} \
                or signposts_by_type["begin"]["signpost_timestamp_continuous_ns"] \
                >= signposts_by_type["end"]["signpost_timestamp_continuous_ns"] \
                or not (
                    interval["continuous_started_ns"]
                    <= signposts_by_type["begin"]["signpost_timestamp_continuous_ns"]
                    < signposts_by_type["end"]["signpost_timestamp_continuous_ns"]
                    <= interval["continuous_ended_ns"]
                ) \
                or not value["command_buffers"] \
                or not value["encoders"] or not value["counters"]:
            raise XctraceAdapterError(
                "every raw interval requires one exact begin/end signpost pair and Metal rows"
            )
        counters = value["counters"]
        gpu_time_ns = sum(row["gpu_time_ns"] for row in counters)
        physical_bytes = sum(row["physical_bytes"] for row in counters)
        occupancy = sum(
            Decimal(str(row["occupancy_percent"])) * row["gpu_time_ns"]
            for row in counters
        ) / Decimal(gpu_time_ns)
        bandwidth = Decimal(physical_bytes) * Decimal(1_000_000_000) / Decimal(gpu_time_ns)
        value["aggregate"] = {
            "gpu_time_ns": gpu_time_ns,
            "physical_bytes": physical_bytes,
            "occupancy_percent": float(occupancy),
            "bandwidth_bytes_per_second": float(bandwidth),
        }
        output.append(value)
    output.sort(key=lambda row: (
        row["target"]["batch"] or 0, row["target"]["iteration"],
    ))
    return output


def build_capture(
    *, kind: str, raw_bundle_path: pathlib.Path, profile_path: pathlib.Path,
    trace_path: pathlib.Path, toc_path: pathlib.Path,
    export_paths: Mapping[str, pathlib.Path],
    probe_pid: int, run_nonce: str, probe_argv_sha256: str,
    metal_registry_id: str, production: bool, lease: Mapping[str, Any] | None = None,
    xctrace_runtime: Mapping[str, Any] | None = None,
    signature_verifier: Callable[[dict[str, Any], bytes], tuple[bool, str]] = (
        _verify_profile_signature
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join signed, separate xctrace tables into one direct phase capture."""
    _hex64(run_nonce, "run nonce")
    _hex64(probe_argv_sha256, "probe argv hash")
    profile_raw, profile_identity = _read_json(profile_path)
    profile = validate_profile(
        profile_raw, production=production, signature_verifier=signature_verifier,
    )
    bundle, bundle_identity = _read_json(raw_bundle_path)
    targets = _bundle_targets(
        bundle, kind=kind, run_nonce=run_nonce, probe_argv_sha256=probe_argv_sha256,
    )
    intervals = _bundle_interval_population(bundle, targets)
    trace_identity = trace_tree_identity(trace_path, require_immutable=production)
    toc, toc_identity = _load_document(toc_path, profile["toc"]["document_format"])
    toc_fingerprint = _validate_document(
        toc, profile["toc"], template=profile["xctrace"]["template_name"],
    )
    capture_bounds = _capture_bounds(toc, profile["toc"])
    if set(export_paths) != set(REQUIRED_TABLES):
        raise XctraceAdapterError("exact signed export path set is required")
    rows_by_table: dict[str, list[dict[str, Any]]] = {}
    export_identities: dict[str, dict[str, Any]] = {}
    export_fingerprints: dict[str, str] = {}
    for table_name in REQUIRED_TABLES:
        section = profile["tables"]["exports"][table_name]
        exported, identity = _load_document(
            export_paths[table_name], section["document_format"],
        )
        export_identities[table_name] = identity
        export_fingerprints[table_name] = _validate_document(exported, section)
        rows_by_table[table_name] = _parse_rows(exported, section)
    joined = _join_tables(
        rows_by_table, intervals, probe_pid=probe_pid,
        metal_registry_id=metal_registry_id, run_nonce=run_nonce,
    )
    for joined_phase in joined:
        interval = joined_phase["target"]["interval"]
        if not (
            capture_bounds["capture_started_at_unix_ns"]
            <= interval["wall_started_unix_ns"] < interval["wall_ended_unix_ns"]
            <= capture_bounds["capture_ended_at_unix_ns"]
            and capture_bounds["capture_started_at_continuous_ns"]
            <= interval["continuous_started_ns"] < interval["continuous_ended_ns"]
            <= capture_bounds["capture_ended_at_continuous_ns"]
        ):
            raise XctraceAdapterError("trace-derived capture bounds do not enclose trial interval")
    records = []
    for joined_phase in joined:
        interval_descriptor = joined_phase["target"]
        target = interval_descriptor["target"]
        if target is None:
            continue
        interval = target["interval"]
        source_ids = {
            "signpost_event_ids": sorted(
                row["signpost_event_id"] for row in joined_phase["signposts"]
            ),
            "command_buffer_ids": sorted(
                row["command_buffer_id"] for row in joined_phase["command_buffers"]
            ),
            "encoder_ids": sorted(row["encoder_id"] for row in joined_phase["encoders"]),
            "counter_event_ids": sorted(
                row["source_event_id"] for row in joined_phase["counters"]
            ),
        }
        source_sample_id = canonical_sha256({
            "trace_tree_sha256": trace_identity["tree_sha256"],
            "toc_sha256": toc_identity["sha256"],
            "export_sha256s": {
                table: export_identities[table]["sha256"] for table in REQUIRED_TABLES
            },
            "profile_sha256": profile["profile_sha256"],
            "phase_marker_sha256": target["marker"],
            "interval_id": interval_descriptor["interval_id"],
            "source_ids": source_ids,
            "source_row_sha256s": {
                table: sorted(row["row_sha256"] for row in joined_phase[table])
                for table in REQUIRED_TABLES
            },
        })
        aggregate = joined_phase["aggregate"]
        records.append({
            "source_sample_id": source_sample_id,
            "phase_marker_sha256": target["marker"],
            "interval_sha256": target["interval_sha256"],
            "process_id": probe_pid,
            "run_nonce": run_nonce,
            "interval_started_at_unix_ns": interval["wall_started_unix_ns"],
            "interval_ended_at_unix_ns": interval["wall_ended_unix_ns"],
            "interval_started_at_continuous_ns": interval["continuous_started_ns"],
            "interval_ended_at_continuous_ns": interval["continuous_ended_ns"],
            "measurement_scope": "exact-probe-process+exact-metal-registry-id",
            "attribution": "direct-counter",
            "estimated": False,
            "apportioned": False,
            "gpu_time_ns": aggregate["gpu_time_ns"],
            "physical_bytes": aggregate["physical_bytes"],
            "occupancy_percent": aggregate["occupancy_percent"],
            "bandwidth_bytes_per_second": aggregate["bandwidth_bytes_per_second"],
        })
    capture = _stamp({
        "schema": CAPTURE_SCHEMA,
        "backend_id": BACKEND_ID,
        "probe_pid": probe_pid,
        "run_nonce": run_nonce,
        "probe_argv_sha256": probe_argv_sha256,
        "metal_registry_id": metal_registry_id,
        **capture_bounds,
        "records": records,
    }, "capture_sha256")
    synthetic = profile["synthetic_fixture"] is True or not production
    receipt = _stamp({
        "schema": RECEIPT_SCHEMA,
        "adapter_schema": SCHEMA,
        "adapter_contract_sha256": CONTRACT_SHA256,
        "synthetic_fixture": synthetic,
        "physical_evidence_eligible": not synthetic,
        "physical_credit": 0 if synthetic else None,
        "lease": dict(lease or {"inherited": False}),
        "probe_pid": probe_pid,
        "run_nonce": run_nonce,
        "probe_argv_sha256": probe_argv_sha256,
        "metal_registry_id": metal_registry_id,
        "profile_identity": profile_identity,
        "profile_sha256": profile["profile_sha256"],
        "raw_bundle_identity": bundle_identity,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "trace_identity": trace_identity,
        "toc_identity": toc_identity,
        "toc_schema_fingerprint_sha256": toc_fingerprint,
        "trace_capture_bounds": capture_bounds,
        "export_identities": export_identities,
        "export_schema_fingerprint_sha256s": export_fingerprints,
        "xctrace_runtime": dict(xctrace_runtime or {}),
        "record_count": len(records),
        "source_row_counts": {
            table: len(rows_by_table[table]) for table in REQUIRED_TABLES
        },
        "consumed_row_counts": {
            table: len(rows_by_table[table]) for table in REQUIRED_TABLES
        },
        "all_source_rows_consumed": True,
        "aggregation_rule": profile["tables"]["aggregation"],
        "capture_sha256": capture["capture_sha256"],
    }, "receipt_sha256")
    return capture, receipt


def _validate_receipt_bundle(
    receipt: Mapping[str, Any], capture: Mapping[str, Any], *,
    kind: str,
    raw_bundle_path: pathlib.Path, profile_path: pathlib.Path,
    trace_path: pathlib.Path, toc_path: pathlib.Path,
    export_paths: Mapping[str, pathlib.Path], probe_pid: int, run_nonce: str,
    probe_argv_sha256: str, metal_registry_id: str,
    expected_lease: Mapping[str, Any], expected_output_directory: Mapping[str, Any],
    signature_verifier: Callable[[dict[str, Any], bytes], tuple[bool, str]] = (
        _verify_profile_signature
    ),
) -> dict[str, Any]:
    """Re-open every provenance source and verify one production handoff."""
    expected_receipt_fields = {
        "schema", "adapter_schema", "adapter_contract_sha256", "synthetic_fixture",
        "physical_evidence_eligible", "physical_credit", "lease", "probe_pid",
        "run_nonce", "probe_argv_sha256", "metal_registry_id", "profile_identity",
        "profile_sha256", "raw_bundle_identity", "raw_bundle_sha256",
        "trace_identity", "toc_identity", "toc_schema_fingerprint_sha256",
        "trace_capture_bounds", "export_identities",
        "export_schema_fingerprint_sha256s", "xctrace_runtime", "record_count",
        "source_row_counts", "consumed_row_counts", "all_source_rows_consumed",
        "aggregation_rule", "capture_sha256", "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_receipt_fields:
        raise XctraceAdapterError("production adapter receipt fields are incomplete or unexpected")
    unstamped = copy.deepcopy(receipt)
    claimed_receipt = unstamped.pop("receipt_sha256")
    if claimed_receipt != canonical_sha256(unstamped):
        raise XctraceAdapterError("adapter receipt self-hash mismatch")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["adapter_schema"] != SCHEMA \
            or receipt["adapter_contract_sha256"] != CONTRACT_SHA256:
        raise XctraceAdapterError("adapter receipt schema/contract differs")
    if receipt["synthetic_fixture"] is not False \
            or receipt["physical_evidence_eligible"] is not True \
            or receipt["physical_credit"] is not None:
        raise XctraceAdapterError("synthetic or ineligible adapter receipt cannot be handed off")
    exact_lease = {
        "inherited": True,
        "device": expected_lease.get("device"),
        "inode": expected_lease.get("inode"),
    }
    if receipt["lease"] != exact_lease:
        raise XctraceAdapterError("adapter receipt inherited lease differs")
    stable_output_path = pathlib.Path(str(expected_output_directory.get("path", "")))
    try:
        stable_output_stat = stable_output_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise XctraceAdapterError(f"stable published output directory is unavailable: {exc}") from exc
    if expected_output_directory != {
        "held": True, "path": str(stable_output_path),
        "device": stable_output_stat.st_dev, "inode": stable_output_stat.st_ino,
    } or not stat.S_ISDIR(stable_output_stat.st_mode):
        raise XctraceAdapterError("stable published output path/inode binding differs")
    if (
        receipt["probe_pid"] != probe_pid or receipt["run_nonce"] != run_nonce
        or receipt["probe_argv_sha256"] != probe_argv_sha256
        or receipt["metal_registry_id"] != metal_registry_id
    ):
        raise XctraceAdapterError("adapter receipt PID/nonce/argv/device binding differs")

    profile_raw, profile_identity = _read_json(profile_path)
    profile = validate_profile(
        profile_raw, production=True, signature_verifier=signature_verifier,
    )
    bundle, bundle_identity = _read_json(raw_bundle_path)
    targets = _bundle_targets(
        bundle, kind=kind,
        run_nonce=run_nonce, probe_argv_sha256=probe_argv_sha256,
    )
    _bundle_interval_population(bundle, targets)
    if receipt["profile_identity"] != profile_identity \
            or receipt["profile_sha256"] != profile["profile_sha256"]:
        raise XctraceAdapterError("adapter receipt profile identity differs from disk")
    if receipt["raw_bundle_identity"] != bundle_identity \
            or receipt["raw_bundle_sha256"] != bundle.get("raw_bundle_sha256"):
        raise XctraceAdapterError("adapter receipt raw-bundle identity differs from disk")
    if receipt["trace_identity"] != trace_tree_identity(trace_path, require_immutable=True):
        raise XctraceAdapterError("adapter receipt trace-tree identity differs from disk")

    toc, toc_identity = _load_document(toc_path, profile["toc"]["document_format"])
    toc_fingerprint = _validate_document(
        toc, profile["toc"], template=profile["xctrace"]["template_name"],
    )
    bounds = _capture_bounds(toc, profile["toc"])
    if receipt["toc_identity"] != toc_identity \
            or receipt["toc_schema_fingerprint_sha256"] != toc_fingerprint \
            or receipt["trace_capture_bounds"] != bounds:
        raise XctraceAdapterError("adapter receipt TOC identity/fingerprint/bounds differ")
    if set(export_paths) != set(REQUIRED_TABLES):
        raise XctraceAdapterError("receipt verifier requires every exported table path")
    observed_export_identities = {}
    observed_export_fingerprints = {}
    observed_row_counts = {}
    for table in REQUIRED_TABLES:
        section = profile["tables"]["exports"][table]
        document, identity = _load_document(export_paths[table], section["document_format"])
        observed_export_identities[table] = identity
        observed_export_fingerprints[table] = _validate_document(document, section)
        observed_row_counts[table] = len(_parse_rows(document, section))
    if receipt["export_identities"] != observed_export_identities \
            or receipt["export_schema_fingerprint_sha256s"] != observed_export_fingerprints:
        raise XctraceAdapterError("one or more adapter export identities/fingerprints differ")
    if receipt["source_row_counts"] != observed_row_counts \
            or receipt["consumed_row_counts"] != observed_row_counts \
            or receipt["all_source_rows_consumed"] is not True \
            or receipt["aggregation_rule"] != AGGREGATION_RULE:
        raise XctraceAdapterError("adapter receipt total row accounting/aggregation differs")

    capture_value = dict(capture)
    unstamped_capture = copy.deepcopy(capture_value)
    claimed_capture = unstamped_capture.pop("capture_sha256", None)
    if claimed_capture != canonical_sha256(unstamped_capture) \
            or claimed_capture != receipt["capture_sha256"]:
        raise XctraceAdapterError("capture self-hash or receipt binding differs")
    capture_errors = trusted_normalizer._capture_errors(
        capture_value, schema=trusted_normalizer.METAL_CAPTURE_SCHEMA,
        backend=trusted_normalizer.METAL_BACKEND, probe_pid=probe_pid,
        run_nonce=run_nonce, probe_argv_sha256=probe_argv_sha256,
        metal_registry_id=metal_registry_id,
    )
    if capture_errors:
        raise XctraceAdapterError("capture contract is invalid: " + "; ".join(capture_errors))
    if receipt["record_count"] != len(capture_value["records"]):
        raise XctraceAdapterError("adapter receipt record count differs from capture")

    runtime = receipt["xctrace_runtime"]
    output_fd = runtime.get("operational_output_dir_fd") if isinstance(runtime, dict) else None
    if isinstance(output_fd, bool) or not isinstance(output_fd, int) or output_fd < 3:
        raise XctraceAdapterError("adapter runtime lacks its exact operational dirfd number")
    operational_paths = {
        "toc": f"/dev/fd/{output_fd}/{toc_path.name}",
        **{
            table: f"/dev/fd/{output_fd}/{export_paths[table].name}"
            for table in REQUIRED_TABLES
        },
    }
    published_paths = {
        "toc": str(toc_path),
        **{table: str(export_paths[table]) for table in REQUIRED_TABLES},
    }
    binary = profile["xctrace"]["binary"]["path"]
    expected_toc_argv = [
        binary, "export", "--input", str(trace_path), "--toc", "--output",
        operational_paths["toc"],
    ]
    expected_export_argvs = {
        table: [
            binary, "export", "--input", str(trace_path), "--xpath",
            profile["tables"]["exports"][table]["xpath"], "--output",
            operational_paths[table],
        ]
        for table in REQUIRED_TABLES
    }
    expected_runtime_fields = {
        "binary", "version_string", "version_stdout_sha256", "template_name",
        "toc_argv", "toc_argv_sha256", "export_argvs", "export_argv_sha256s",
        "shell", "environment", "environment_sha256", "output_directory",
        "operational_output_dir_fd", "operational_output_paths",
        "published_output_paths",
    }
    if not isinstance(runtime, dict) or set(runtime) != expected_runtime_fields \
            or runtime["binary"] != profile["xctrace"]["binary"] \
            or runtime["version_string"] != profile["xctrace"]["version_string"] \
            or runtime["version_stdout_sha256"] != profile["xctrace"]["version_stdout_sha256"] \
            or runtime["template_name"] != profile["xctrace"]["template_name"] \
            or runtime["toc_argv"] != expected_toc_argv \
            or runtime["toc_argv_sha256"] != canonical_sha256(expected_toc_argv) \
            or runtime["export_argvs"] != expected_export_argvs \
            or runtime["export_argv_sha256s"] != {
                table: canonical_sha256(argv) for table, argv in expected_export_argvs.items()
            } \
            or runtime["shell"] is not False \
            or runtime["environment"] != PINNED_EXPORT_ENV \
            or runtime["environment_sha256"] != canonical_sha256(PINNED_EXPORT_ENV) \
            or runtime["output_directory"] != dict(expected_output_directory) \
            or runtime["operational_output_paths"] != operational_paths \
            or runtime["published_output_paths"] != published_paths:
        raise XctraceAdapterError("adapter receipt sanitized runtime argv/environment differs")
    recomputed_capture, recomputed_receipt = build_capture(
        kind=kind, raw_bundle_path=raw_bundle_path, profile_path=profile_path,
        trace_path=trace_path, toc_path=toc_path, export_paths=export_paths,
        probe_pid=probe_pid, run_nonce=run_nonce,
        probe_argv_sha256=probe_argv_sha256, metal_registry_id=metal_registry_id,
        production=True, lease=receipt["lease"], xctrace_runtime=runtime,
        signature_verifier=signature_verifier,
    )
    if recomputed_capture != capture_value or recomputed_receipt != receipt:
        raise XctraceAdapterError(
            "capture/receipt do not exactly recompute from sealed multi-table provenance"
        )
    return {
        "receipt_sha256": claimed_receipt,
        "capture_sha256": claimed_capture,
        "all_provenance_files_reopened": True,
        "physical_evidence_eligible": True,
    }


def validate_receipt(
    *, receipt_path: pathlib.Path, capture_path: pathlib.Path,
    kind: str,
    raw_bundle_path: pathlib.Path, profile_path: pathlib.Path,
    trace_path: pathlib.Path, toc_path: pathlib.Path,
    export_paths: Mapping[str, pathlib.Path], probe_pid: int, run_nonce: str,
    probe_argv_sha256: str, metal_registry_id: str,
    expected_lease: Mapping[str, Any], expected_output_directory: Mapping[str, Any],
    signature_verifier: Callable[[dict[str, Any], bytes], tuple[bool, str]] = (
        _verify_profile_signature
    ),
) -> dict[str, Any]:
    """File-backed production verifier; in-memory canonical JSON is insufficient."""
    receipt, _receipt_identity = _read_json(receipt_path)
    capture, _capture_identity = _read_json(capture_path)
    return _validate_receipt_bundle(
        receipt, capture, kind=kind, raw_bundle_path=raw_bundle_path,
        profile_path=profile_path,
        trace_path=trace_path, toc_path=toc_path, export_paths=export_paths,
        probe_pid=probe_pid, run_nonce=run_nonce,
        probe_argv_sha256=probe_argv_sha256, metal_registry_id=metal_registry_id,
        expected_lease=expected_lease,
        expected_output_directory=expected_output_directory,
        signature_verifier=signature_verifier,
    )


def _lease_identity(fd: int) -> dict[str, Any]:
    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 3:
        raise XctraceAdapterError("one inherited heavy-lease descriptor is required")
    try:
        row = os.fstat(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise XctraceAdapterError(f"inherited heavy lease is invalid or unlocked: {exc}") from exc
    if not stat.S_ISREG(row.st_mode):
        raise XctraceAdapterError("inherited heavy lease is not a regular file")
    return {"inherited": True, "device": row.st_dev, "inode": row.st_ino}


def _held_output_directory(fd: int) -> dict[str, Any]:
    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 3:
        raise XctraceAdapterError("one held output-directory descriptor is required")
    try:
        row = os.fstat(fd)
    except OSError as exc:
        raise XctraceAdapterError(f"held output directory is invalid: {exc}") from exc
    if not stat.S_ISDIR(row.st_mode):
        raise XctraceAdapterError("held output descriptor is not a directory")
    try:
        raw_path = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 4096)
        stable_path = pathlib.Path(raw_path.split(b"\0", 1)[0].decode("utf-8")).resolve(strict=True)
        stable_stat = stable_path.stat(follow_symlinks=False)
    except (AttributeError, OSError, UnicodeError) as exc:
        raise XctraceAdapterError(f"held output directory lacks a stable F_GETPATH: {exc}") from exc
    if stable_path.is_symlink() or not stable_path.is_dir() \
            or (stable_stat.st_dev, stable_stat.st_ino) != (row.st_dev, row.st_ino):
        raise XctraceAdapterError("stable output path differs from held directory inode")
    return {
        "held": True, "path": str(stable_path),
        "device": row.st_dev, "inode": row.st_ino,
    }


def _safe_leaf(name: str) -> str:
    if not isinstance(name, str) or pathlib.PurePath(name).name != name \
            or name in {"", ".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None:
        raise XctraceAdapterError("adapter output name is not one safe leaf")
    return name


def _operational_output(fd: int, name: str) -> pathlib.Path:
    _held_output_directory(fd)
    return pathlib.Path(f"/dev/fd/{fd}") / _safe_leaf(name)


def _output_absent(fd: int, name: str) -> None:
    name = _safe_leaf(name)
    try:
        os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise XctraceAdapterError(f"cannot inspect adapter output leaf {name}: {exc}") from exc
    raise XctraceAdapterError(f"adapter output already exists: {name}")


def _immutable_json(fd: int, name: str, value: Mapping[str, Any]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(_safe_leaf(name), flags, 0o444, dir_fd=fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short adapter output write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_command(
    argv: Sequence[str], *, lease_fd: int, output_dir_fd: int,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> subprocess.CompletedProcess[bytes]:
    result = runner(
        list(argv), cwd=ROOT, env=dict(PINNED_EXPORT_ENV),
        pass_fds=(lease_fd, output_dir_fd),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=600, check=False, shell=False,
    )
    if result.returncode != 0:
        raise XctraceAdapterError(
            f"xctrace command failed ({result.returncode}): {result.stderr[-1000:]!r}"
        )
    return result


def run_export(
    *, kind: str, trace_path: pathlib.Path, raw_bundle_path: pathlib.Path,
    profile_path: pathlib.Path, probe_pid: int, run_nonce: str,
    probe_argv_sha256: str, metal_registry_id: str, output_dir_fd: int,
    toc_output_name: str, export_output_prefix: str, capture_output_name: str,
    receipt_output_name: str, lease_fd: int,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the exact reviewed xctrace exports and publish immutable JSON outputs."""
    lease = _lease_identity(lease_fd)
    output_directory = _held_output_directory(output_dir_fd)
    profile_raw, _profile_identity = _read_json(profile_path)
    profile = validate_profile(profile_raw, production=True)
    binary = pathlib.Path(profile["xctrace"]["binary"]["path"])
    if binary != FULL_XCODE_XCTRACE or not binary.is_file() or binary.is_symlink():
        raise XctraceAdapterError("exact full-Xcode xctrace binary is absent")
    if physical_counter_attestation.file_identity(binary) != profile["xctrace"]["binary"]:
        raise XctraceAdapterError("full-Xcode binary identity differs from reviewed profile")
    trace_before = trace_tree_identity(trace_path, require_immutable=True)
    export_output_prefix = _safe_leaf(export_output_prefix)
    export_output_names = {
        table: _safe_leaf(
            f"{export_output_prefix}.{table}.{profile['tables']['exports'][table]['document_format']}"
        )
        for table in REQUIRED_TABLES
    }
    for name in (
        toc_output_name, *export_output_names.values(), capture_output_name,
        receipt_output_name,
    ):
        _output_absent(output_dir_fd, name)
    toc_operational = _operational_output(output_dir_fd, toc_output_name)
    export_operational = {
        table: _operational_output(output_dir_fd, name)
        for table, name in export_output_names.items()
    }
    published_directory = pathlib.Path(output_directory["path"])
    toc_output = published_directory / _safe_leaf(toc_output_name)
    export_outputs = {
        table: published_directory / name for table, name in export_output_names.items()
    }
    version = _run_command(
        profile["xctrace"]["version_argv"], lease_fd=lease_fd,
        output_dir_fd=output_dir_fd, runner=runner,
    )
    version_sha256 = hashlib.sha256(version.stdout).hexdigest()
    version_string = version.stdout.decode("utf-8", errors="strict").strip()
    if version_sha256 != profile["xctrace"]["version_stdout_sha256"] \
            or version_string != profile["xctrace"]["version_string"]:
        raise XctraceAdapterError("xctrace runtime version differs from reviewed profile")
    toc_argv = [
        str(binary), "export", "--input", str(trace_path), "--toc",
        "--output", str(toc_operational),
    ]
    export_argvs = {
        table: [
            str(binary), "export", "--input", str(trace_path), "--xpath",
            profile["tables"]["exports"][table]["xpath"], "--output",
            str(export_operational[table]),
        ]
        for table in REQUIRED_TABLES
    }
    _run_command(toc_argv, lease_fd=lease_fd, output_dir_fd=output_dir_fd, runner=runner)
    for table in REQUIRED_TABLES:
        _run_command(
            export_argvs[table], lease_fd=lease_fd,
            output_dir_fd=output_dir_fd, runner=runner,
        )
    if trace_tree_identity(trace_path, require_immutable=True) != trace_before:
        raise XctraceAdapterError("raw xctrace package changed during export")
    runtime = {
        "binary": profile["xctrace"]["binary"],
        "version_string": version_string,
        "version_stdout_sha256": version_sha256,
        "template_name": profile["xctrace"]["template_name"],
        "toc_argv": toc_argv,
        "toc_argv_sha256": canonical_sha256(toc_argv),
        "export_argvs": export_argvs,
        "export_argv_sha256s": {
            table: canonical_sha256(argv) for table, argv in export_argvs.items()
        },
        "shell": False,
        "environment": PINNED_EXPORT_ENV,
        "environment_sha256": canonical_sha256(PINNED_EXPORT_ENV),
        "output_directory": output_directory,
        "operational_output_dir_fd": output_dir_fd,
        "operational_output_paths": {
            "toc": str(toc_operational),
            **{table: str(path) for table, path in export_operational.items()},
        },
        "published_output_paths": {
            "toc": str(toc_output),
            **{table: str(path) for table, path in export_outputs.items()},
        },
    }
    capture, receipt = build_capture(
        kind=kind, raw_bundle_path=raw_bundle_path, profile_path=profile_path,
        trace_path=trace_path, toc_path=toc_output, export_paths=export_outputs,
        probe_pid=probe_pid, run_nonce=run_nonce,
        probe_argv_sha256=probe_argv_sha256, metal_registry_id=metal_registry_id,
        production=True, lease=lease, xctrace_runtime=runtime,
    )
    for path in (toc_output, *export_outputs.values()):
        path.chmod(0o444)
    if physical_counter_attestation.file_identity(toc_output) != receipt["toc_identity"]:
        raise XctraceAdapterError("TOC changed before immutable publication")
    for table, path in export_outputs.items():
        if physical_counter_attestation.file_identity(path) != receipt["export_identities"][table]:
            raise XctraceAdapterError(f"{table} export changed before immutable publication")
    if trace_tree_identity(trace_path, require_immutable=True) != trace_before:
        raise XctraceAdapterError("raw xctrace package changed before immutable publication")
    _immutable_json(output_dir_fd, capture_output_name, capture)
    _immutable_json(output_dir_fd, receipt_output_name, receipt)
    validate_receipt(
        receipt_path=published_directory / _safe_leaf(receipt_output_name),
        capture_path=published_directory / _safe_leaf(capture_output_name),
        kind=kind, raw_bundle_path=raw_bundle_path, profile_path=profile_path,
        trace_path=trace_path, toc_path=toc_output, export_paths=export_outputs,
        probe_pid=probe_pid, run_nonce=run_nonce,
        probe_argv_sha256=probe_argv_sha256, metal_registry_id=metal_registry_id,
        expected_lease=lease, expected_output_directory=output_directory,
    )
    return capture, receipt


def status(profile_path: pathlib.Path = DEFAULT_PROFILE) -> dict[str, Any]:
    blockers = []
    profile = None
    if not profile_path.is_file() or profile_path.is_symlink():
        blockers.append("operator-reviewed production xctrace export profile is absent")
    else:
        try:
            raw, _identity = _read_json(profile_path)
            profile = validate_profile(raw, production=True)
        except (OSError, ValueError, XctraceAdapterError) as exc:
            blockers.append(f"production xctrace export profile is invalid: {exc}")
    if not FULL_XCODE_XCTRACE.is_file() or FULL_XCODE_XCTRACE.is_symlink():
        blockers.append("exact full-Xcode xctrace binary is absent")
    return _stamp({
        "schema": SCHEMA,
        "contract_sha256": CONTRACT_SHA256,
        "default_profile_path": str(profile_path),
        "production_profile_present": profile is not None,
        "production_export_ready": profile is not None and not blockers,
        "collection_or_export_started": False,
        "physical_evidence_claimed": False,
        "synthetic_fixture_physical_credit": 0,
        "blockers": blockers,
    }, "status_sha256")


def _selftest() -> int:
    assert contract()["contract_sha256"] == CONTRACT_SHA256
    assert status()["physical_evidence_claimed"] is False
    assert contract()["synthetic_fixture_physical_credit"] == 0
    print("appendix_xctrace_export_adapter.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--export", action="store_true")
    group.add_argument("--seal-profile-draft", action="store_true")
    parser.add_argument("--kind", choices=("device", "spec"))
    parser.add_argument("--trace", type=pathlib.Path)
    parser.add_argument("--raw-bundle", type=pathlib.Path)
    parser.add_argument("--profile", type=pathlib.Path)
    parser.add_argument("--probe-pid", type=int)
    parser.add_argument("--run-nonce")
    parser.add_argument("--probe-argv-sha256")
    parser.add_argument("--metal-registry-id")
    parser.add_argument("--output-dir-fd", type=int)
    parser.add_argument("--toc-output-name")
    parser.add_argument("--export-output-prefix")
    parser.add_argument("--capture-output-name")
    parser.add_argument("--receipt-output-name")
    parser.add_argument("--draft", type=pathlib.Path)
    parser.add_argument("--private-key", type=pathlib.Path)
    parser.add_argument("--public-key", type=pathlib.Path)
    parser.add_argument("--signature-output", type=pathlib.Path)
    parser.add_argument("--profile-output", type=pathlib.Path)
    parser.add_argument("--validity-hours", type=int, default=24)
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.selftest:
        return _selftest()
    if args.seal_profile_draft:
        if any(value is None for value in (
            args.draft, args.private_key, args.public_key,
            args.signature_output, args.profile_output,
        )):
            parser.error("profile sealing requires draft/key/signature/profile paths")
        try:
            profile = seal_profile_draft(
                draft_path=args.draft, private_key=args.private_key,
                public_key=args.public_key,
                detached_signature_output=args.signature_output,
                profile_output=args.profile_output,
                validity_hours=args.validity_hours,
            )
        except (OSError, UnicodeError, ValueError, XctraceAdapterError) as exc:
            parser.error(str(exc))
        print(json.dumps({
            "profile_output": str(args.profile_output),
            "profile_sha256": profile["profile_sha256"],
            "expires_at_unix_ns": profile["operator_review"]["expires_at_unix_ns"],
        }, indent=2, sort_keys=True))
        return 0
    required = {
        "kind": args.kind, "trace": args.trace, "raw_bundle": args.raw_bundle,
        "profile": args.profile, "probe_pid": args.probe_pid,
        "run_nonce": args.run_nonce, "probe_argv_sha256": args.probe_argv_sha256,
        "metal_registry_id": args.metal_registry_id,
        "output_dir_fd": args.output_dir_fd,
        "toc_output_name": args.toc_output_name,
        "export_output_prefix": args.export_output_prefix,
        "capture_output_name": args.capture_output_name,
        "receipt_output_name": args.receipt_output_name,
    }
    if any(value is None for value in required.values()):
        parser.error("production xctrace export requires every trace/profile/binding/output argument")
    lease_raw = os.environ.get(ram_scheduler.HEAVY_LEASE_FD_ENV)
    try:
        lease_fd = int(lease_raw) if lease_raw is not None else -1
        run_export(
            kind=args.kind, trace_path=args.trace, raw_bundle_path=args.raw_bundle,
            profile_path=args.profile, probe_pid=args.probe_pid,
            run_nonce=args.run_nonce, probe_argv_sha256=args.probe_argv_sha256,
            metal_registry_id=args.metal_registry_id, output_dir_fd=args.output_dir_fd,
            toc_output_name=args.toc_output_name,
            export_output_prefix=args.export_output_prefix,
            capture_output_name=args.capture_output_name,
            receipt_output_name=args.receipt_output_name, lease_fd=lease_fd,
        )
    except (OSError, UnicodeError, ValueError, XctraceAdapterError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
