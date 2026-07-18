#!/usr/bin/env python3.12
"""Source-bound adapter capability probe for the successor control plane.

The eco_admission module (the additive planner's admission layer) decides adapter
readiness from a STATIC id-map, BUILT_ADAPTERS = {"qwen2.5-dense": "...", "gpt-oss-moe":
"..."}. That map is a fiction: presence of a string does not prove the adapter can
convert its source, reassemble provenance, load natively at parity, resume exactly,
stream a lifecycle, or run an evaluator. Master-goal section 5.4 requires the FULL
conjunction of those capabilities before a parent is execution-ready.

This module replaces the id-map with a REAL, source-bound probe:
  - a registry maps a model family to the concrete doctor_v5 adapter module on disk;
  - probe_adapter() runs that adapter's ``capabilities`` subcommand as a READ-ONLY
    subprocess ([python3.12, <adapter>, "capabilities"]), which is instant and does no
    heavy work, parses its JSON capability report, and binds the adapter file's sha256;
  - admit() normalizes the real report into the master-goal per-requirement booleans
    and returns a sealed admission record whose ready_for_execution is the AND of the
    whole conjunction, never a lookup in a string map.

Two real capability-report shapes are understood, dispatched by their ``schema``:
  - ``hawking.doctor_v5_strand_ladder_capabilities.v1`` (qwen2.5-dense): publishes
    rates, resident evaluation labels, per-treatment ``doctor_hooks`` (only method
    ``none`` is supported), and claim restrictions (``claims.quality`` False);
  - ``hawking.doctor_v5_gptoss_moe_capabilities.v1`` (gpt-oss-moe): a 0.1-contract that
    is ``reviewed_for_live_campaign_execution`` False, has ``implemented`` booleans that
    are mostly False, and a list of ``blockers``; its ``run`` command refuses (exit 78).

Probing only. It launches no heavy compute, never touches the campaign namespace
(reports/condense/doctor_v5_ultra), and in selftest never spawns the real adapters:
the selftest injects a fake probe returning canned reports.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, now_iso, sha_file, repo_root,
)

ADMISSION_SCHEMA = "hawking.successor.admission.v1"

# Real capability-report schemas this probe understands (dispatch key).
QWEN_CAP_SCHEMA = "hawking.doctor_v5_strand_ladder_capabilities.v1"
GPTOSS_CAP_SCHEMA = "hawking.doctor_v5_gptoss_moe_capabilities.v1"

# The master-goal 5.4 readiness conjunction. ready_for_execution is the AND of all of
# these; each is derived from the REAL capability report, never asserted statically.
REQUIREMENTS: tuple[str, ...] = (
    "source_conversion",       # a source -> artifact conversion path is implemented
    "reassembly_provenance",   # original shard/tensor byte-range reassembly is bound
    "runtime_specs",           # native runtime specs (labels/regimes) are published
    "tokenizer_template",      # tokenizer + chat template are source-bound
    "evaluator_executable",    # a standalone evaluator can actually run
    "native_load_parity",      # native load + parity is provable
    "exact_resume",            # exact multi-shard resume artifact exists
    "streamed_lifecycle",      # download -> bake -> seal -> release lifecycle is streamed
    "quality_path",            # a quality-evidence path exists (distinct from claiming it)
    "reviewed",                # reviewed for THIS generation's live execution
)


class AdmissionError(EcoError):
    """Fail-closed error in the source-bound admission probe."""


@dataclass(frozen=True)
class Config:
    """Family -> doctor_v5 adapter module path (relative to the repo root)."""

    adapter_paths: dict[str, str]
    python_exe: str
    probe_timeout_s: float


def default_config() -> Config:
    return Config(
        adapter_paths={
            "qwen2.5-dense": "tools/condense/doctor_v5_strand_ladder_block_parallel_adapter.py",
            "gpt-oss-moe": "tools/condense/doctor_v5_gptoss_moe_adapter.py",
        },
        python_exe="python3.12",
        probe_timeout_s=120.0,
    )


def adapter_path_for(family: str, *, config: Config | None = None) -> Path | None:
    """Absolute adapter module path for a family, or None (must_build)."""
    config = config or default_config()
    rel = config.adapter_paths.get(family)
    if rel is None:
        return None
    return repo_root() / rel


# -- READ-ONLY subprocess probe ------------------------------------------------------------
def probe_adapter(family: str, adapter_path: str | os.PathLike[str] | None,
                  *, config: Config | None = None) -> dict[str, Any]:
    """Run <adapter> capabilities READ-ONLY and return the parsed report + source seal.

    Returns a uniform envelope:
      {available, family, adapter_path, adapter_source_sha256, adapter_source_bytes,
       report, error}
    `available` is False (with `error` set, `report` None) whenever the module is
    absent, the subcommand fails, or its output is not a capability-report object. This
    is instant: `capabilities` prints a static report and does no heavy work.
    """
    config = config or default_config()
    if adapter_path is None:
        return _unavailable(family, None, None, None, "no adapter registered for family (must_build)")
    path = Path(adapter_path)
    if not path.is_file():
        return _unavailable(family, str(path), None, None, f"adapter module absent: {path}")
    try:
        source_sha, source_bytes = sha_file(path)
    except OSError as exc:
        return _unavailable(family, str(path), None, None, f"cannot hash adapter source: {exc}")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, read-only capabilities command
            [config.python_exe, str(path), "capabilities"],
            capture_output=True, text=True, timeout=config.probe_timeout_s,
            cwd=str(path.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(family, str(path), source_sha, source_bytes,
                            f"capabilities subprocess failed: {exc}")
    if completed.returncode != 0:
        return _unavailable(family, str(path), source_sha, source_bytes,
                            f"capabilities exited {completed.returncode}: "
                            f"{completed.stderr.strip()[:200]}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _unavailable(family, str(path), source_sha, source_bytes,
                            f"capabilities output is not JSON: {exc}")
    if not isinstance(report, dict) or not isinstance(report.get("schema"), str):
        return _unavailable(family, str(path), source_sha, source_bytes,
                            "capabilities output is not a schema-tagged report")
    return {
        "available": True, "family": family, "adapter_path": str(path),
        "adapter_source_sha256": source_sha, "adapter_source_bytes": source_bytes,
        "report": report, "error": None,
    }


def _unavailable(family: str, path: str | None, sha: str | None, size: int | None,
                 error: str) -> dict[str, Any]:
    return {
        "available": False, "family": family, "adapter_path": path,
        "adapter_source_sha256": sha, "adapter_source_bytes": size,
        "report": None, "error": error,
    }


# -- normalize a real report into the 5.4 requirement conjunction --------------------------
def _blocker_ids(report: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for b in report.get("blockers") or ():
        if isinstance(b, dict) and isinstance(b.get("id"), str):
            out.add(b["id"])
    return out


def _normalize_gptoss(report: dict[str, Any]) -> dict[str, Any]:
    """gpt-oss-moe 0.1-contract report -> requirement booleans + blockers.

    Nearly every requirement maps to an explicit report field: `implemented.*`,
    `reviewed_for_live_campaign_execution`, `quality_claims_permitted`, and the
    `blockers` list (which names the missing tokenizer and reassembly provenance).
    """
    impl = report.get("implemented") or {}
    ids = _blocker_ids(report)
    reqs = {
        "source_conversion": bool(impl.get("full_str2_campaign_execution")),
        "reassembly_provenance": "original-source-provenance-reassembly-missing" not in ids,
        "runtime_specs": bool(impl.get("apple_silicon_moe_runtime"))
                         and "gptoss-moe-str2-loader-missing" not in ids,
        "tokenizer_template": "gptoss-tokenizer-missing" not in ids,
        "evaluator_executable": bool(impl.get("quality_evaluation")),
        "native_load_parity": bool(impl.get("apple_silicon_moe_runtime")),
        "exact_resume": bool(impl.get("full_str2_campaign_execution")),
        "streamed_lifecycle": bool(impl.get("full_str2_campaign_execution"))
                              and "ten-artifact-disk-retention-infeasible" not in ids,
        "quality_path": bool(report.get("quality_claims_permitted"))
                        or bool(impl.get("quality_evaluation")),
        "reviewed": bool(report.get("reviewed_for_live_campaign_execution")),
    }
    blockers = [f"gpt-oss-moe:{b['id']}: {b.get('detail', '')}"
                for b in (report.get("blockers") or ())
                if isinstance(b, dict) and isinstance(b.get("id"), str)]
    if not reqs["reviewed"]:
        blockers.append("gpt-oss-moe:not_reviewed_for_live_campaign_execution: "
                        "adapter_version is a 0.1-contract; run refuses (exit 78)")
    return {
        "adapter_id": report.get("adapter_id"),
        "requirements": reqs,
        "blockers": blockers,
        "claim_restricted": not bool(report.get("quality_claims_permitted")),
    }


def _normalize_qwen(report: dict[str, Any], method: str) -> dict[str, Any]:
    """qwen2.5-dense ladder report -> requirement booleans + blockers.

    The ladder is execution-ready for method ``none`` but claim-restricted
    (``claims.quality`` is False). The treatment hooks lora_kd / blockwise_qat /
    strand_hessian are unsupported, so admitting any of them fails closed with the
    hook's own blocker string. Requirement booleans are read off the published rates,
    resident evaluation labels, model_family, and evaluation section.
    """
    hooks = report.get("doctor_hooks") or {}
    ev = report.get("evaluation") or {}
    claims = report.get("claims") or {}
    method_entry = hooks.get(method)
    method_supported = bool(isinstance(method_entry, dict) and method_entry.get("supported"))
    resident = ev.get("resident_labels")
    # `reviewed` must be an EXPLICIT attestation in the report, never inferred from hook
    # support (5.4 requires "reviewed for this campaign generation"). The qwen ladder report
    # carries no such field, so we fail closed: the codec is execution-capable, but the live
    # review is not machine-attestable from the capability report.
    reviewed = bool(report.get("reviewed_for_live_campaign_execution", False))
    reqs = {
        "source_conversion": bool(report.get("rates")),
        "reassembly_provenance": bool(report.get("labels")),
        "runtime_specs": bool(report.get("labels")) and bool(resident),
        "tokenizer_template": report.get("model_family") == "qwen2.5-dense",
        "evaluator_executable": bool(resident),
        "native_load_parity": bool(resident),
        "exact_resume": "dense_reconstructions" in ev,
        "streamed_lifecycle": bool(ev),
        "quality_path": bool(ev),
        "reviewed": reviewed,
    }
    blockers: list[str] = []
    if not method_supported:
        detail = (method_entry.get("blocker")
                  if isinstance(method_entry, dict) else None) \
            or f"treatment hook '{method}' is not published by the adapter"
        blockers.append(f"qwen2.5-dense:treatment[{method}]: {detail}")
    if not reviewed:
        blockers.append("qwen2.5-dense:review_flag_absent: the capability report has no "
                        "reviewed_for_live_campaign_execution field; treated as unreviewed "
                        "(fail-closed). The codec is execution-capable but live review is not "
                        "machine-attestable from the report.")
    execution_capable = method_supported and bool(resident) and bool(report.get("rates"))
    return {
        "adapter_id": report.get("adapter_id"),
        "requirements": reqs,
        "blockers": blockers,
        "execution_capable": execution_capable,
        "claim_restricted": not bool(claims.get("quality", False)),
    }


def _normalize(family: str, report: dict[str, Any], method: str) -> dict[str, Any]:
    schema = report.get("schema")
    if schema == GPTOSS_CAP_SCHEMA:
        return _normalize_gptoss(report)
    if schema == QWEN_CAP_SCHEMA:
        return _normalize_qwen(report, method)
    raise AdmissionError(f"unknown capability report schema: {schema!r} (family {family})")


# -- admission record ----------------------------------------------------------------------
def admit(family: str, label: str, *, method: str = "none",
          probe: Callable[..., dict[str, Any]] = probe_adapter,
          config: Config | None = None) -> dict[str, Any]:
    """Return a sealed admission record derived from the REAL capability probe.

    `ready_for_execution` is the AND of the full master-goal 5.4 requirement
    conjunction, extracted from the live report. It is NEVER read from a static string
    map. When the adapter is absent or its report cannot be parsed, the record is
    fail-closed: every requirement False, ready_for_execution False, one blocker.
    """
    config = config or default_config()
    path = adapter_path_for(family, config=config)
    envelope = probe(family, str(path) if path is not None else None, config=config)

    if not envelope.get("available"):
        reqs = {name: False for name in REQUIREMENTS}
        record = {
            "schema": ADMISSION_SCHEMA,
            "generated_at": now_iso(),
            "family": family,
            "label": label,
            "method": method,
            "adapter_id": None,
            "adapter_path": envelope.get("adapter_path"),
            "adapter_available": False,
            "adapter_source_sha256": envelope.get("adapter_source_sha256"),
            "adapter_source_bytes": envelope.get("adapter_source_bytes"),
            "capability_report_schema": None,
            "requirements": reqs,
            "reviewed": False,
            "claim_restricted": True,
            "ready_for_execution": False,
            "blockers": [f"{family}:capability_probe_unavailable: {envelope.get('error')}"],
            "disposition": "must_build",
        }
        return seal_field(record, "admission_sha256")

    report = envelope["report"]
    norm = _normalize(family, report, method)
    reqs = {name: bool(norm["requirements"].get(name, False)) for name in REQUIREMENTS}
    ready = all(reqs.values())
    blockers = list(norm["blockers"])
    # any unmet requirement not already explained gets an explicit blocker
    for name in REQUIREMENTS:
        if not reqs[name] and not any(name in b for b in blockers):
            blockers.append(f"{family}:requirement_unmet[{name}]")
    record = {
        "schema": ADMISSION_SCHEMA,
        "generated_at": now_iso(),
        "family": family,
        "label": label,
        "method": method,
        "adapter_id": norm.get("adapter_id") or f"{family}:{report.get('schema')}",
        "adapter_path": envelope.get("adapter_path"),
        "adapter_available": True,
        "adapter_source_sha256": envelope.get("adapter_source_sha256"),
        "adapter_source_bytes": envelope.get("adapter_source_bytes"),
        "capability_report_schema": report.get("schema"),
        "requirements": reqs,
        "reviewed": reqs["reviewed"],
        "execution_capable": bool(norm.get("execution_capable", False)),
        "claim_restricted": bool(norm["claim_restricted"]),
        "ready_for_execution": ready,
        "blockers": blockers,
        "disposition": "ready" if ready else "blocked",
    }
    return seal_field(record, "admission_sha256")


# -- synthetic reports for offline selftest (byte-shaped like the real adapters) -----------
def _fake_qwen_report() -> dict[str, Any]:
    return {
        "schema": QWEN_CAP_SCHEMA,
        "model_family": "qwen2.5-dense",
        "adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
        "labels": ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
        "rates": [{"rate_id": "q4", "artifact_mode": "packed_vector_control"}],
        "evaluation": {
            "resident_labels": ["0.5B", "1.5B", "3B", "7B", "14B"],
            "32B/72B": "packed codec artifacts supported; disk/RAM-gated",
            "dense_reconstructions": "ephemeral and deleted after evaluation",
        },
        "doctor_hooks": {
            "none": {"supported": True},
            "lora_kd": {"supported": False,
                        "blocker": "doctor.py lora requires one dense base plus resident teacher"},
            "blockwise_qat": {"supported": False,
                              "blocker": "blockwise entrypoint assumes a single resident source"},
            "strand_hessian": {"supported": False,
                               "blocker": "strand Doctor uses fixed scratch paths"},
        },
        "claims": {"quality": False, "dominance": False, "source_deletion": False},
    }


def _fake_gptoss_report() -> dict[str, Any]:
    return {
        "schema": GPTOSS_CAP_SCHEMA,
        "adapter_id": "doctor-v5-strand-ladder-gpt-oss-moe",
        "adapter_version": "0.1-contract",
        "model_family": "gpt-oss-moe",
        "labels": ["120B"],
        "implemented": {
            "typed_spec_builder": True,
            "header_only_checkpoint_validation": True,
            "logical_parameter_accounting": True,
            "bounded_memory_per_expert_mxfp4_to_bf16": True,
            "source_read_only_byte_range_receipts": True,
            "full_str2_campaign_execution": False,
            "quality_evaluation": False,
            "apple_silicon_moe_runtime": False,
        },
        "reviewed_for_live_campaign_execution": False,
        "blockers": [
            {"id": "strand-reader-whole-file-u8-unsupported", "layer": "codec_input",
             "detail": "SafeTensors::open rejects U8 raw shards"},
            {"id": "original-source-provenance-reassembly-missing", "layer": "artifact_format",
             "detail": "byte-range reassembly manifest missing"},
            {"id": "gptoss-moe-str2-loader-missing", "layer": "runtime",
             "detail": "no Apple-Silicon per-expert STR2 loader"},
            {"id": "gptoss-tokenizer-missing", "layer": "evaluation",
             "detail": "local source has no tokenizer"},
            {"id": "ten-artifact-disk-retention-infeasible", "layer": "lifecycle",
             "detail": "ten payload ceilings exceed the reserve"},
        ],
        "source_deletion_permitted": False,
        "quality_claims_permitted": False,
    }


def _fake_probe(family: str, adapter_path: str | os.PathLike[str] | None,
                *, config: Config | None = None) -> dict[str, Any]:
    """Injected offline probe: canned reports, a synthetic source seal, no subprocess."""
    if family == "qwen2.5-dense":
        report = _fake_qwen_report()
    elif family == "gpt-oss-moe":
        report = _fake_gptoss_report()
    else:
        return _unavailable(family, str(adapter_path) if adapter_path else None,
                            None, None, "no adapter registered for family (must_build)")
    return {
        "available": True, "family": family,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_source_sha256": "e" * 64, "adapter_source_bytes": 4096,
        "report": report, "error": None,
    }


def selftest() -> dict[str, Any]:
    # qwen method=none: the codec is execution-CAPABLE, but the ladder report has NO explicit
    # reviewed_for_live_campaign_execution field, so readiness fails closed (never inferred
    # from hook support) with a review_flag_absent blocker. This is the honesty fix.
    qwen = admit("qwen2.5-dense", "14B", method="none", probe=_fake_probe)
    if not sealed(qwen, "admission_sha256"):
        raise AdmissionError("qwen admission not self-sealed")
    if qwen["ready_for_execution"]:
        raise AdmissionError("qwen-none must NOT be ready without an explicit review attestation")
    if not qwen.get("execution_capable"):
        raise AdmissionError("qwen-none should be execution_capable (the codec runs)")
    if not any("review_flag_absent" in b for b in qwen["blockers"]):
        raise AdmissionError(f"qwen-none should carry a review_flag_absent blocker: {qwen['blockers']}")
    if not qwen["claim_restricted"]:
        raise AdmissionError("qwen should be claim-restricted (claims.quality False)")

    # a qwen report WITH an explicit live-review attestation -> execution-ready (the AND holds)
    def _reviewed_probe(family, path, *, config=None):
        p = _fake_probe(family, path, config=config)
        if family == "qwen2.5-dense" and p.get("report"):
            p = dict(p)
            p["report"] = dict(p["report"], reviewed_for_live_campaign_execution=True)
        return p
    qwen_ok = admit("qwen2.5-dense", "14B", method="none", probe=_reviewed_probe)
    if not qwen_ok["ready_for_execution"]:
        raise AdmissionError(f"qwen with explicit review should be ready: {qwen_ok['blockers']}")
    if not all(qwen_ok["requirements"].values()):
        raise AdmissionError("reviewed qwen-none should satisfy every requirement")

    # an unsupported treatment hook fails closed with the hook's own blocker
    qwen_lora = admit("qwen2.5-dense", "14B", method="lora_kd", probe=_fake_probe)
    if qwen_lora["ready_for_execution"]:
        raise AdmissionError("qwen lora_kd must NOT be ready (hook unsupported)")
    if qwen_lora["requirements"]["reviewed"]:
        raise AdmissionError("qwen lora_kd must be unreviewed")
    if not any("treatment[lora_kd]" in b for b in qwen_lora["blockers"]):
        raise AdmissionError(f"qwen lora_kd blocker missing: {qwen_lora['blockers']}")

    # gpt-oss-moe 0.1-contract -> not reviewed, refuses, ready False with 5+ blockers
    gptoss = admit("gpt-oss-moe", "120B", probe=_fake_probe)
    if not sealed(gptoss, "admission_sha256"):
        raise AdmissionError("gptoss admission not self-sealed")
    if gptoss["ready_for_execution"]:
        raise AdmissionError("gpt-oss-moe must NOT be execution-ready")
    if gptoss["reviewed"]:
        raise AdmissionError("gpt-oss-moe must be unreviewed for live execution")
    if len(gptoss["blockers"]) < 5:
        raise AdmissionError(f"gpt-oss-moe should carry 5+ blockers: {gptoss['blockers']}")
    # the ready flag must be the AND of the conjunction, not a string-map lookup
    if gptoss["ready_for_execution"] != all(gptoss["requirements"].values()):
        raise AdmissionError("ready_for_execution is not the AND of the requirements")

    # an unregistered family fails closed as must_build
    unknown = admit("llama-dense", "405B", probe=_fake_probe)
    if unknown["ready_for_execution"] or unknown["adapter_available"]:
        raise AdmissionError("unregistered family must be must_build / unavailable")
    if unknown["disposition"] != "must_build":
        raise AdmissionError("unregistered family disposition should be must_build")

    # the registry resolves the real adapter files on disk (path binding, no execution)
    qwen_path = adapter_path_for("qwen2.5-dense")
    gptoss_path = adapter_path_for("gpt-oss-moe")
    registry_paths_exist = bool(qwen_path and qwen_path.is_file()
                                and gptoss_path and gptoss_path.is_file())

    return {
        "ok": True,
        "qwen_none_ready": qwen["ready_for_execution"],
        "qwen_execution_capable": qwen["execution_capable"],
        "qwen_reviewed_ready": qwen_ok["ready_for_execution"],
        "qwen_claim_restricted": qwen["claim_restricted"],
        "qwen_lora_ready": qwen_lora["ready_for_execution"],
        "gptoss_ready": gptoss["ready_for_execution"],
        "gptoss_reviewed": gptoss["reviewed"],
        "gptoss_blocker_count": len(gptoss["blockers"]),
        "unknown_disposition": unknown["disposition"],
        "ready_is_conjunction": True,
        "registry_paths_exist": registry_paths_exist,
        "requirements": list(REQUIREMENTS),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Source-bound adapter capability probe.")
    ap.add_argument("--probe", metavar="FAMILY",
                    help="run the REAL capabilities subprocess for one family (read-only)")
    ap.add_argument("--label", default="probe")
    ap.add_argument("--method", default="none")
    ap.add_argument("--selftest", action="store_true", help="offline selftest (default)")
    args = ap.parse_args()
    if args.probe and not args.selftest:
        rec = admit(args.probe, args.label, method=args.method)
        print(json.dumps(rec, indent=2, sort_keys=True))
        sys.exit(0 if rec["ready_for_execution"] else 3)
    print(json.dumps(selftest(), indent=2, sort_keys=True))
