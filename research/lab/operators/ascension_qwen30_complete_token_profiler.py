"""Fail-closed complete-token profiler for the admitted Qwen30 direct pack.

This is deliberately a *measurement sidecar*, not another runtime or campaign
controller.  It waits for the real direct-packed all-48-layer execution and
the separate production-CB trace stage owned by the native runtime watcher,
then re-admits the exact source/manifest binding before reconciling them.

The runtime-owned trace retains individual completed Metal dispatch samples
instead of only human-friendly kernel aggregation.  That makes the fixed,
audited Qwen30 token graph attributable without inventing a roofline or
treating a component probe as a token measurement.  Metal may overlap
dispatch intervals, so the sidecar sweep-lines the real timestamp intervals:
unambiguous semantic intervals, multi-stage overlap, and idle/command-graph
gaps remain distinct.  Any host time outside the observed device timeline
also remains visible rather than being assigned to a convenient kernel.  A
profile is eligible only when at least 98% of actual complete-token host wall
time has an unambiguous measured stage/command-topology attribution.  It
never publishes TPS.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox" / "physical"
)

SCHEMA = "hawking.ascension.qwen30_direct_packed_complete_token_profiler.v1"
PROFILE_SCHEMA = "hawking.ascension.qwen30_direct_packed_complete_token_profile.v1"
BOTTLENECK_SCHEMA = "hawking.ascension.qwen30_direct_packed_complete_token_bottleneck.v1"
MICROBENCH_RECEIPT_SCHEMA = "hawking.ascension.qwen30_direct_packed_gate_up_pair_component_receipt.v1"
PROFILE_IMPLEMENTATION_REVISION = "interval-sweep-source-bound-runtime-epoch-v3"
FULL_RESULT_SCHEMA = "hawking.ascension.qwen30_complete_native_runtime_result.v1"
FULL_RESULT_STATUS = "EARNED_QWEN30_DIRECT_PACKED_NATIVE_METAL_FULL_TOKEN_EXECUTED_UNQUALIFIED"
RUNTIME_RECEIPT_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
RUNTIME_RECEIPT_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
MICROBENCH_RAW_SCHEMA = "hawking.ascension.qwen30_direct_packed_gate_up_pair_component.v1"
MICROBENCH_RAW_STATUS = "PASS_DIRECT_PACKED_QWEN30_GATE_UP_PAIR_COMPONENT_CPU_PARITY_NOT_MODEL_TPS"
ADMISSION_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"

MAX_DOCUMENT_BYTES = 128 * 1024 * 1024
MIN_WALL_COVERAGE_PERCENT = 98.0


class Qwen30CompleteTokenProfilerError(RuntimeError):
    """The profiler cannot safely continue with the observed physical state."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_DOCUMENT_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_jsonl_once(path: Path, document: Mapping[str, Any], *, record_id: str) -> bool:
    """Append a sealed knowledge row once, including across watcher restarts."""

    with _locked(path.with_name(f".{path.name}.lock")):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(prior, Mapping) and prior.get("record_id") == record_id:
                    return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(document), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o640)
    return True


def _percent(value: int | float, whole: int | float) -> float:
    return 0.0 if whole <= 0 else float(value) * 100.0 / float(whole)


def _required_string(document: Mapping[str, Any], field: str, *, label: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise Qwen30CompleteTokenProfilerError(f"{label} lacks non-empty {field!r}")
    return value


def _required_mapping(document: Mapping[str, Any], field: str, *, label: str) -> Mapping[str, Any]:
    value = document.get(field)
    if not isinstance(value, Mapping):
        raise Qwen30CompleteTokenProfilerError(f"{label} lacks object {field!r}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


SCALAR_GATE_UP_SWIGLU_KERNEL = "three_dispatch_direct_packed_gate_up_swiglu_control"
PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL = (
    "paired_direct_packed_gate_up_swiglu_scalar_order_production_no_parity"
)


def _kernel_plan(
    gate_up_swiglu_kernel: str = SCALAR_GATE_UP_SWIGLU_KERNEL,
) -> list[tuple[str, str, str]]:
    """Return the exact current direct-packed forward graph in trace order.

    The plan is intentionally tied to the real `forward_token_greedy` source,
    not inferred from an aggregate count.  A trace with any different kernel
    name/order is rejected rather than force-fit into the stage ledger.
    """

    if gate_up_swiglu_kernel not in {
        SCALAR_GATE_UP_SWIGLU_KERNEL,
        PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL,
    }:
        raise Qwen30CompleteTokenProfilerError(
            "unsupported gate/up/SwiGLU kernel for exact complete-token trace plan: "
            f"{gate_up_swiglu_kernel!r}"
        )
    plan: list[tuple[str, str, str]] = [
        ("qwen_complete_binary_embedding_lookup", "embedding", "embedding lookup"),
    ]
    for layer in range(48):
        prefix = f"layer_{layer:02d}"
        plan.extend(
            [
                ("qwen_complete_binary_decode_vector", "norm", f"{prefix} input RMSNorm vector decode"),
                ("qwen_complete_binary_decode_vector", "norm", f"{prefix} post-attention RMSNorm vector decode"),
                ("qwen_complete_binary_decode_vector", "qkv_kv", f"{prefix} Q RMSNorm vector decode"),
                ("qwen_complete_binary_decode_vector", "qkv_kv", f"{prefix} K RMSNorm vector decode"),
                ("rmsnorm_f32", "norm", f"{prefix} input RMSNorm"),
                ("qwen_binary_sign_scale_matvec", "qkv_kv", f"{prefix} Q projection"),
                ("qwen_binary_sign_scale_matvec", "qkv_kv", f"{prefix} K projection"),
                ("qwen_binary_sign_scale_matvec", "qkv_kv", f"{prefix} V projection"),
                ("qwen_complete_rmsnorm_rows_f32", "qkv_kv", f"{prefix} Q RMSNorm"),
                ("qwen_complete_rmsnorm_rows_f32", "qkv_kv", f"{prefix} K RMSNorm"),
                ("rope_qk_kv_append_vbias_f32", "qkv_kv", f"{prefix} RoPE and KV append"),
                ("mha_decode_f32", "attention", f"{prefix} GQA decode"),
                ("qwen_binary_sign_scale_matvec", "attention", f"{prefix} O projection"),
                ("add_inplace", "combine", f"{prefix} attention residual"),
                ("rmsnorm_f32", "norm", f"{prefix} post-attention RMSNorm"),
                ("qwen_binary_sign_scale_matvec", "router", f"{prefix} router projection"),
                ("moe_topk_gate", "router", f"{prefix} device top-k router"),
                ("qwen_complete_normalize_route_weights", "router", f"{prefix} route-weight normalization"),
            ]
        )
        for route in range(8):
            if gate_up_swiglu_kernel == SCALAR_GATE_UP_SWIGLU_KERNEL:
                plan.extend(
                    [
                    ("qwen_binary_sign_scale_matvec", "expert_gate_up", f"{prefix} route_{route} gate projection"),
                    ("qwen_binary_sign_scale_matvec", "expert_gate_up", f"{prefix} route_{route} up projection"),
                    ("qwen_complete_silu_mul_offset", "expert_gate_up", f"{prefix} route_{route} SwiGLU"),
                    ("qwen_binary_sign_scale_matvec", "expert_down", f"{prefix} route_{route} down projection"),
                    ]
                )
            else:
                plan.extend(
                    [
                        (
                            "qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate",
                            "expert_gate_up",
                            f"{prefix} route_{route} paired scalar-order gate/up/SwiGLU",
                        ),
                        ("qwen_binary_sign_scale_matvec", "expert_down", f"{prefix} route_{route} down projection"),
                    ]
                )
        plan.append(("qwen_complete_weighted_expert_add", "combine", f"{prefix} routed expert combine"))
    plan.extend(
        [
            ("qwen_complete_binary_decode_vector", "norm", "final RMSNorm vector decode"),
            ("rmsnorm_f32", "norm", "final RMSNorm"),
            ("qwen_binary_sign_scale_matvec", "final_head", "LM head projection"),
            ("qwen_complete_any_nonfinite_f32", "final_head", "final logits finite guard"),
            ("sample_argmax_f32", "sampling", "device greedy sampler"),
        ]
    )
    return plan


EXPECTED_DISPATCHES = len(_kernel_plan())
assert EXPECTED_DISPATCHES == 2454, EXPECTED_DISPATCHES
PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES = len(
    _kernel_plan(PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL)
)
assert PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES == 1686, (
    PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES
)


def _trace_kernel_plan(runtime: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Select only a source-named graph matching the direct runtime binding."""

    observed = runtime.get("gate_up_swiglu_kernel")
    if not isinstance(observed, str):
        raise Qwen30CompleteTokenProfilerError(
            "runtime trace lacks explicit gate_up_swiglu_kernel provenance"
        )
    return _kernel_plan(observed)


class Qwen30CompleteTokenProfiler:
    """One safe reconciliation/trace/profiling cycle."""

    def __init__(self, *, physical_root: Path = DEFAULT_PHYSICAL_ROOT) -> None:
        self.physical_root = Path(physical_root)
        self.qwen_root = self.physical_root / "qwen30"
        self.complete_root = self.qwen_root / "complete-gravity"
        self.runtime_root = self.qwen_root / "complete-runtime"
        self.profile_root = self.qwen_root / "complete-token-profiler"
        self.full_result = self.runtime_root / "QWEN30_DIRECT_PACKED_NATIVE_FULL_TOKEN_RESULT.json"
        self.admission = self.complete_root / "QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
        self.manifest = self.complete_root / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        self.trace_result = self.runtime_root / "QWEN30_DIRECT_PACKED_NATIVE_KERNEL_PROFILE_RESULT.json"
        self.runtime_receipt = self.runtime_root / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
        self.status_path = self.profile_root / "QWEN30_COMPLETE_TOKEN_PROFILER_STATUS.json"
        self.profile_path = self.profile_root / "QWEN30_COMPLETE_TOKEN_PROFILE.json"
        self.bottleneck_path = self.profile_root / "QWEN30_COMPLETE_TOKEN_BOTTLENECK_RECEIPT.json"
        self.profile_history = self.profile_root / "history"
        self.microbench_raw = self.profile_root / "QWEN30_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_RAW_RESULT.json"
        self.microbench_receipt = self.profile_root / "QWEN30_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_RECEIPT.json"
        self.shared_kernel_genome = self.physical_root / "qwen-family" / "dual-gravity" / "ASCENSION_KERNEL_GENOME.jsonl"

    def _binding(self) -> dict[str, Any]:
        raw = _read_json(self.admission)
        if raw is None:
            raise Qwen30CompleteTokenProfilerError(f"missing admission receipt: {self.admission}")
        try:
            admission = verify(raw, label=str(self.admission))
        except Exception as exc:
            raise Qwen30CompleteTokenProfilerError(f"invalid admission receipt: {exc}") from exc
        if admission.get("status") != ADMISSION_STATUS:
            raise Qwen30CompleteTokenProfilerError("admission receipt is not an admitted complete binary artifact")
        manifest_binding = _required_mapping(admission, "complete_manifest", label="admission receipt")
        current = _required_mapping(admission, "current_source_revalidation", label="admission receipt")
        expected_manifest = self.manifest.resolve()
        observed_manifest = Path(_required_string(manifest_binding, "path", label="manifest binding")).resolve()
        if observed_manifest != expected_manifest:
            raise Qwen30CompleteTokenProfilerError("admission receipt does not bind the protected current Qwen30 manifest")
        raw_manifest = _read_json(observed_manifest)
        if raw_manifest is None:
            raise Qwen30CompleteTokenProfilerError("admitted Qwen30 manifest is unreadable")
        try:
            manifest = verify(raw_manifest, label=str(observed_manifest))
        except Exception as exc:
            raise Qwen30CompleteTokenProfilerError(f"invalid admitted manifest: {exc}") from exc
        if manifest.get("status") != MANIFEST_STATUS:
            raise Qwen30CompleteTokenProfilerError("admitted manifest is not a complete binary candidate")
        manifest_seal = _required_string(manifest_binding, "seal_sha256", label="manifest binding")
        if manifest.get("seal_sha256") != manifest_seal:
            raise Qwen30CompleteTokenProfilerError("admission manifest seal differs from the current protected manifest")
        return {
            "admission_path": str(self.admission),
            "admission_seal_sha256": admission.get("seal_sha256"),
            "manifest_path": str(observed_manifest),
            "manifest_seal_sha256": manifest_seal,
            "source_audit_seal_sha256": _required_string(current, "source_audit_seal_sha256", label="source revalidation"),
            "source_revision": _required_string(current, "revision", label="source revalidation"),
        }

    def _full_result(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        document = _read_json(self.full_result)
        if document is None:
            raise Qwen30CompleteTokenProfilerError("real Qwen30 direct-packed full-token result is absent")
        if document.get("schema") != FULL_RESULT_SCHEMA or document.get("status") != FULL_RESULT_STATUS:
            raise Qwen30CompleteTokenProfilerError("full-token result has unexpected schema or status")
        runtime = _required_mapping(document, "runtime_binding", label="full-token result")
        if runtime.get("manifest_seal_sha256") != binding["manifest_seal_sha256"] or runtime.get("source_revision") != binding["source_revision"]:
            raise Qwen30CompleteTokenProfilerError("full-token result is not bound to current admitted source/manifest")
        execution = _required_mapping(document, "execution", label="full-token result")
        step = _required_mapping(execution, "step", label="full-token execution")
        if execution.get("all_48_layers_executed") is not True or execution.get("final_norm_lm_head_device_argmax_executed") is not True:
            raise Qwen30CompleteTokenProfilerError("full-token result does not prove all layers and final head")
        if not isinstance(step.get("elapsed_us_diagnostic_not_tps"), int) or step["elapsed_us_diagnostic_not_tps"] <= 0:
            raise Qwen30CompleteTokenProfilerError("full-token result lacks a positive diagnostic wall duration")
        if not isinstance(execution.get("input_token_id"), int) or not isinstance(step.get("sampled_token_id"), int):
            raise Qwen30CompleteTokenProfilerError("full-token result lacks token identity for trace parity")
        return document

    def _validate_trace_result(self, document: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
        if document.get("schema") != FULL_RESULT_SCHEMA:
            raise Qwen30CompleteTokenProfilerError("runtime-owned profile-token result schema is invalid")
        if document.get("status") != FULL_RESULT_STATUS:
            raise Qwen30CompleteTokenProfilerError(
                f"trace did not earn a production-CB GPU timing result: {document.get('status')!r}"
            )
        runtime = _required_mapping(document, "runtime_binding", label="trace result")
        if runtime.get("manifest_seal_sha256") != binding["manifest_seal_sha256"] or runtime.get("source_revision") != binding["source_revision"]:
            raise Qwen30CompleteTokenProfilerError("trace result is not bound to current source/manifest")
        if runtime.get("packed_matvec_kernel") != "scalar_one_thread_per_row_control":
            raise Qwen30CompleteTokenProfilerError(
                "this control profiler refuses a candidate kernel receipt; candidate-specific parity and trace mapping are required"
            )
        plan = _trace_kernel_plan(runtime)
        expected_dispatches = len(plan)
        execution = _required_mapping(document, "execution", label="trace result")
        if execution.get("all_48_layers_executed") is not True or execution.get("final_norm_lm_head_device_argmax_executed") is not True:
            raise Qwen30CompleteTokenProfilerError("trace did not prove the direct all-layer token")
        profiler = _required_mapping(document, "profiler", label="trace result")
        if profiler.get("tcb_trace_mode_requested") != "gpu_prod":
            raise Qwen30CompleteTokenProfilerError("runtime-owned profile-token result was not executed in gpu_prod mode")
        if profiler.get("expected_complete_token_dispatch_samples") != expected_dispatches:
            raise Qwen30CompleteTokenProfilerError("runtime-owned trace does not declare the exact expected complete-token sample count")
        if profiler.get("gpu_timing_sample_count") != expected_dispatches:
            raise Qwen30CompleteTokenProfilerError("runtime-owned trace has incomplete GPU timing sample coverage")
        if profiler.get("complete_token_gpu_profile_coverage_earned") is not True:
            raise Qwen30CompleteTokenProfilerError("runtime-owned trace did not earn its own complete GPU sample coverage gate")
        samples = profiler.get("ordered_dispatch_samples")
        if not isinstance(samples, list) or len(samples) != expected_dispatches:
            raise Qwen30CompleteTokenProfilerError(
                f"trace must contain exactly {expected_dispatches} raw dispatch samples"
            )
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise Qwen30CompleteTokenProfilerError("trace contains a non-object dispatch sample")
            if not isinstance(sample.get("gpu_us"), int) or sample["gpu_us"] < 0:
                raise Qwen30CompleteTokenProfilerError("trace sample lacks measured GPU duration")
            if not isinstance(sample.get("gpu_start_ns"), int) or not isinstance(sample.get("gpu_end_ns"), int):
                raise Qwen30CompleteTokenProfilerError("trace sample lacks production GPU timestamp bounds")
            if sample["gpu_end_ns"] < sample["gpu_start_ns"]:
                raise Qwen30CompleteTokenProfilerError("trace GPU timestamps run backward")

    def _trace_result(self, binding: Mapping[str, Any]) -> dict[str, Any] | None:
        document = _read_json(self.trace_result)
        if document is None:
            return None
        try:
            self._validate_trace_result(document, binding)
        except Qwen30CompleteTokenProfilerError:
            return None
        return document

    def _trace_validation_error(self, binding: Mapping[str, Any]) -> str | None:
        """Return an exact rejection reason without promoting a partial trace."""

        document = _read_json(self.trace_result)
        if document is None:
            return None
        try:
            self._validate_trace_result(document, binding)
        except Qwen30CompleteTokenProfilerError as exc:
            return str(exc)
        return None

    def _runtime_bound_profile_binding(
        self,
        artifact_binding: Mapping[str, Any],
        trace: Mapping[str, Any],
        full: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind a profile to the active canonical runtime receipt.

        The raw complete-token and trace artifacts predate the public runtime
        receipt in the execution flow.  They therefore do not themselves need
        to duplicate the executable SHA, but the profiler must refuse to turn
        them into an optimization handoff unless the current canonical receipt
        seals *these exact files* and identifies the active executable/kernel.
        This prevents a same-artifact trace from being silently reused after a
        runtime executable transition.
        """

        raw_receipt = _read_json(self.runtime_receipt)
        if raw_receipt is None:
            raise Qwen30CompleteTokenProfilerError(
                f"missing canonical exact runtime receipt: {self.runtime_receipt}"
            )
        try:
            receipt = verify(raw_receipt, label=str(self.runtime_receipt))
        except Exception as exc:
            raise Qwen30CompleteTokenProfilerError(
                f"invalid canonical exact runtime receipt: {exc}"
            ) from exc
        if (
            receipt.get("schema") != RUNTIME_RECEIPT_SCHEMA
            or receipt.get("status") != RUNTIME_RECEIPT_STATUS
        ):
            raise Qwen30CompleteTokenProfilerError(
                "canonical exact runtime receipt is not a current native runtime PASS"
            )
        receipt_binding = _required_mapping(receipt, "binding", label="runtime receipt")
        executable_sha256 = receipt_binding.get("runtime_executable_sha256")
        if not _is_sha256(executable_sha256):
            raise Qwen30CompleteTokenProfilerError(
                "canonical runtime receipt lacks a valid runtime executable SHA-256"
            )
        if receipt_binding.get("complete_manifest_seal_sha256") != artifact_binding.get(
            "manifest_seal_sha256"
        ):
            raise Qwen30CompleteTokenProfilerError(
                "canonical runtime receipt is not bound to the current admitted manifest"
            )

        evidence = _required_mapping(receipt, "evidence", label="runtime receipt")
        expected_evidence = (
            ("direct_full_token", self.full_result, full),
            ("complete_gpu_profile", self.trace_result, trace),
        )
        for field, path, document in expected_evidence:
            observed = _required_mapping(evidence, field, label="runtime receipt evidence")
            if Path(_required_string(observed, "path", label=f"runtime receipt {field}")).resolve() != path.resolve():
                raise Qwen30CompleteTokenProfilerError(
                    f"canonical runtime receipt {field} path is not the current runtime artifact"
                )
            if observed.get("sha256") != _sha256_file(path):
                raise Qwen30CompleteTokenProfilerError(
                    f"canonical runtime receipt {field} digest does not bind the current artifact"
                )
            if observed.get("schema") != document.get("schema") or observed.get("status") != document.get("status"):
                raise Qwen30CompleteTokenProfilerError(
                    f"canonical runtime receipt {field} schema/status differs from the current artifact"
                )

        receipt_runtime = _required_mapping(receipt, "runtime", label="runtime receipt")
        trace_runtime = _required_mapping(trace, "runtime_binding", label="trace result")
        full_runtime = _required_mapping(full, "runtime_binding", label="full-token result")
        kernel = receipt_runtime.get("gate_up_swiglu_kernel")
        if (
            not isinstance(kernel, str)
            or kernel != trace_runtime.get("gate_up_swiglu_kernel")
            or kernel != full_runtime.get("gate_up_swiglu_kernel")
        ):
            raise Qwen30CompleteTokenProfilerError(
                "canonical runtime receipt and raw full-token/trace kernel provenance differ"
            )
        if receipt_runtime.get("custom_kernel_used") is not True:
            raise Qwen30CompleteTokenProfilerError(
                "canonical runtime receipt does not explicitly identify the active custom kernel"
            )
        return {
            **dict(artifact_binding),
            "canonical_runtime_receipt_path": str(self.runtime_receipt),
            "canonical_runtime_receipt_seal_sha256": receipt.get("seal_sha256"),
            "runtime_executable_sha256": executable_sha256,
            "gate_up_swiglu_kernel": kernel,
            "custom_kernel_used": True,
        }

    def _profile_input_fingerprint(
        self, trace: Mapping[str, Any], full: Mapping[str, Any], binding: Mapping[str, Any]
    ) -> str:
        """Identity for an immutable trace/full-result reconciliation.

        A detached poller must not generate a fresh sealed Genome observation
        every heartbeat for unchanged physical evidence.  This fingerprint is
        intentionally bound to the source files and the attribution algorithm
        revision, so a new raw trace or a real profiling-method correction is
        visible while a normal poll is idempotent.
        """

        identity = {
            "implementation_revision": PROFILE_IMPLEMENTATION_REVISION,
            "binding": dict(binding),
            "direct_full_token": {
                "path": str(self.full_result),
                "sha256": _sha256_file(self.full_result),
                "schema": full.get("schema"),
                "status": full.get("status"),
            },
            "production_command_buffer_trace": {
                "path": str(self.trace_result),
                "sha256": _sha256_file(self.trace_result),
                "schema": trace.get("schema"),
                "status": trace.get("status"),
            },
        }
        return hashlib.sha256(_canonical(identity)).hexdigest()

    def _existing_profile(
        self, trace: Mapping[str, Any], full: Mapping[str, Any], binding: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        existing = _read_json(self.profile_path)
        if existing is None:
            return None
        try:
            checked = verify(existing, label=str(self.profile_path))
        except Exception:
            return None
        if checked.get("schema") != PROFILE_SCHEMA:
            return None
        if checked.get("profiler_implementation_revision") != PROFILE_IMPLEMENTATION_REVISION:
            return None
        if checked.get("input_fingerprint_sha256") != self._profile_input_fingerprint(trace, full, binding):
            return None
        return checked

    def _archive_prior_profile_outputs(self, *, replacement_input_fingerprint: str) -> None:
        """Keep an immutable diagnostic history when a binding is tightened.

        Profile and bottleneck paths are mutable current pointers for detached
        consumers.  A new runtime binding must not silently erase the older
        trace reconciliation; archive its sealed documents once before
        replacing the current pointers.
        """

        prior_profile = _read_json(self.profile_path)
        if prior_profile is not None:
            try:
                checked_profile = verify(prior_profile, label=str(self.profile_path))
            except Exception:
                checked_profile = None
            if (
                isinstance(checked_profile, Mapping)
                and checked_profile.get("schema") == PROFILE_SCHEMA
                and checked_profile.get("input_fingerprint_sha256") != replacement_input_fingerprint
                and _is_sha256(checked_profile.get("seal_sha256"))
            ):
                archive = self.profile_history / (
                    "QWEN30_COMPLETE_TOKEN_PROFILE_"
                    f"{checked_profile['seal_sha256']}.json"
                )
                if not archive.exists():
                    _atomic_json(archive, checked_profile)

        prior_bottleneck = _read_json(self.bottleneck_path)
        if prior_bottleneck is not None:
            try:
                checked_bottleneck = verify(prior_bottleneck, label=str(self.bottleneck_path))
            except Exception:
                checked_bottleneck = None
            if (
                isinstance(checked_bottleneck, Mapping)
                and checked_bottleneck.get("schema") == BOTTLENECK_SCHEMA
                and _is_sha256(checked_bottleneck.get("seal_sha256"))
            ):
                prior_profile_seal = checked_bottleneck.get("profile_seal_sha256")
                if prior_profile_seal != replacement_input_fingerprint:
                    archive = self.profile_history / (
                        "QWEN30_COMPLETE_TOKEN_BOTTLENECK_"
                        f"{checked_bottleneck['seal_sha256']}.json"
                    )
                    if not archive.exists():
                        _atomic_json(archive, checked_bottleneck)

    @staticmethod
    def _timeline_attribution(
        samples: Sequence[Mapping[str, Any]], plan: Sequence[tuple[str, str, str]]
    ) -> dict[str, Any]:
        """Partition the observed GPU timestamp envelope without serializing it.

        Command-buffer execution can legitimately overlap on the device.  The
        ordered trace is therefore an execution-order authority, not permission
        to pretend that every end timestamp precedes the next start timestamp.
        This sweep produces an exact partition of the first-to-last timestamp
        envelope.  An interval with one semantic bucket is attributed to that
        bucket; an interval with multiple different buckets stays explicitly
        multi-stage rather than being arbitrarily charged to one of them.
        """

        events: dict[int, dict[str, list[int]]] = {}
        for index, sample in enumerate(samples):
            start = int(sample["gpu_start_ns"])
            end = int(sample["gpu_end_ns"])
            # A counter can legitimately round a sub-microsecond dispatch to
            # a zero-width interval.  It remains in the semantic GPU-work
            # ledger above, but it contributes no wall-time segment here.
            if end == start:
                continue
            events.setdefault(start, {"start": [], "end": []})["start"].append(index)
            events.setdefault(end, {"start": [], "end": []})["end"].append(index)

        ordered_times = sorted(events)
        if not ordered_times:
            raise Qwen30CompleteTokenProfilerError("production trace has no GPU timestamp events")

        timeline_by_bucket_ns: dict[str, int] = {
            bucket: 0
            for bucket in (
                "embedding", "norm", "qkv_kv", "attention", "router", "expert_gate_up",
                "expert_down", "shared_expert", "combine", "final_head", "sampling",
            )
        }
        active: set[int] = set()
        idle_ns = 0
        multi_stage_overlap_ns = 0
        same_stage_overlap_ns = 0
        merged_busy_intervals = 0
        previous: int | None = None

        for timestamp in ordered_times:
            if previous is not None and timestamp > previous:
                duration_ns = timestamp - previous
                if not active:
                    idle_ns += duration_ns
                else:
                    buckets = {plan[index][1] for index in active}
                    if len(buckets) == 1:
                        bucket = next(iter(buckets))
                        timeline_by_bucket_ns[bucket] += duration_ns
                        if len(active) > 1:
                            same_stage_overlap_ns += duration_ns
                    else:
                        multi_stage_overlap_ns += duration_ns

            event = events[timestamp]
            # Intervals are [start, end).  An interval ending at this exact
            # timestamp is absent from the interval that follows; a starting
            # interval is present.  This order preserves touching intervals as
            # non-overlapping without inventing a zero-length gap.
            for index in event["end"]:
                active.discard(index)
            was_idle = not active
            for index in event["start"]:
                active.add(index)
            if was_idle and active:
                merged_busy_intervals += 1
            previous = timestamp

        if active:
            raise Qwen30CompleteTokenProfilerError("production GPU interval sweep did not drain")

        envelope_ns = ordered_times[-1] - ordered_times[0]
        semantic_timeline_ns = sum(timeline_by_bucket_ns.values())
        busy_union_ns = semantic_timeline_ns + multi_stage_overlap_ns
        if semantic_timeline_ns + multi_stage_overlap_ns + idle_ns != envelope_ns:
            raise Qwen30CompleteTokenProfilerError("GPU timeline partition does not equal the observed envelope")
        return {
            "timeline_by_bucket_ns": timeline_by_bucket_ns,
            "semantic_timeline_ns": semantic_timeline_ns,
            "multi_stage_overlap_ns": multi_stage_overlap_ns,
            "same_stage_overlap_ns": same_stage_overlap_ns,
            "idle_ns": idle_ns,
            "busy_union_ns": busy_union_ns,
            "envelope_ns": envelope_ns,
            "merged_busy_intervals": merged_busy_intervals,
            "first_gpu_timestamp_ns": ordered_times[0],
            "last_gpu_timestamp_ns": ordered_times[-1],
        }

    @staticmethod
    def _host_stage_attribution(trace: Mapping[str, Any], wall_us: int) -> dict[str, Any]:
        """Validate an optional source-bound host-wall stage ledger.

        GPU timestamps alone deliberately cannot explain host-side direct-pack
        lookup, command submission, or readback time.  A later runtime may
        attach exact, non-overlapping timer offsets to its *same sealed trace*
        under ``profiler.host_stage_intervals``.  This consumer recomputes the
        coverage rather than trusting a boolean supplied by the runtime.
        """

        profiler = _required_mapping(trace, "profiler", label="trace")
        raw = profiler.get("host_stage_intervals")
        if raw is None:
            return {
                "available": False,
                "valid": True,
                "reason": "no source-bound host-stage interval ledger is present on this immutable trace",
                "by_bucket_us": {},
                "covered_us": 0.0,
                "coverage_percent": 0.0,
                "uncovered_us": float(wall_us),
                "interval_count": 0,
                "timer_origin": None,
                "declared_coverage_earned": None,
            }
        if profiler.get("host_stage_timer_origin") != "complete_token_runtime_start":
            raise Qwen30CompleteTokenProfilerError(
                "host-stage ledger must declare offsets from complete_token_runtime_start"
            )
        if not isinstance(raw, list) or not raw:
            raise Qwen30CompleteTokenProfilerError("host-stage ledger must be a non-empty list")
        declared = profiler.get("host_stage_interval_coverage_earned")
        if not isinstance(declared, bool):
            raise Qwen30CompleteTokenProfilerError(
                "host-stage ledger lacks a boolean host_stage_interval_coverage_earned gate"
            )
        allowed = {
            "embedding", "norm", "qkv_kv", "attention", "router", "expert_gate_up",
            "expert_down", "shared_expert", "combine", "final_head", "sampling",
            "command_graph_transition_gap", "hcli_overhead",
        }
        by_bucket_us = {bucket: 0.0 for bucket in allowed}
        covered_us = 0.0
        previous_end = 0.0
        for ordinal, entry in enumerate(raw):
            if not isinstance(entry, Mapping):
                raise Qwen30CompleteTokenProfilerError("host-stage ledger contains a non-object interval")
            bucket = entry.get("bucket")
            if bucket not in allowed:
                raise Qwen30CompleteTokenProfilerError(
                    f"host-stage interval {ordinal} has unsupported bucket {bucket!r}"
                )
            label = entry.get("label")
            if not isinstance(label, str) or not label:
                raise Qwen30CompleteTokenProfilerError(
                    f"host-stage interval {ordinal} lacks a descriptive label"
                )
            start = entry.get("start_us")
            end = entry.get("end_us")
            if (
                not isinstance(start, (int, float))
                or isinstance(start, bool)
                or not isinstance(end, (int, float))
                or isinstance(end, bool)
            ):
                raise Qwen30CompleteTokenProfilerError(
                    f"host-stage interval {ordinal} lacks numeric start_us/end_us"
                )
            start = float(start)
            end = float(end)
            if start < 0.0 or end < start or end > float(wall_us):
                raise Qwen30CompleteTokenProfilerError(
                    f"host-stage interval {ordinal} lies outside the complete-token wall interval"
                )
            if start < previous_end:
                raise Qwen30CompleteTokenProfilerError(
                    "host-stage intervals overlap; nested timers cannot be promoted as wall attribution"
                )
            duration = end - start
            by_bucket_us[str(bucket)] += duration
            covered_us += duration
            previous_end = end
        coverage = _percent(covered_us, wall_us)
        if declared is not (coverage >= MIN_WALL_COVERAGE_PERCENT):
            raise Qwen30CompleteTokenProfilerError(
                "host-stage ledger's coverage gate disagrees with its raw non-overlapping intervals"
            )
        return {
            "available": True,
            "valid": True,
            "reason": None,
            "by_bucket_us": by_bucket_us,
            "covered_us": covered_us,
            "coverage_percent": coverage,
            "uncovered_us": max(0.0, float(wall_us) - covered_us),
            "interval_count": len(raw),
            "timer_origin": "complete_token_runtime_start",
            "declared_coverage_earned": declared,
        }

    def _attribute(self, trace: Mapping[str, Any], full: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_trace_result(trace, binding)
        samples = list(_required_mapping(trace, "profiler", label="trace")["ordered_dispatch_samples"])
        plan = _trace_kernel_plan(_required_mapping(trace, "runtime_binding", label="trace"))
        buckets: dict[str, dict[str, Any]] = {
            name: {
                "bucket": name,
                "gpu_us": 0,
                "dispatches": 0,
                "execution_status": "EXECUTED_MEASURED_ON_PRODUCTION_COMMAND_BUFFER",
                "timing_authority": "per-dispatch Metal timestamp counter samples",
                "operations": [],
            }
            for name in (
                "embedding", "norm", "qkv_kv", "attention", "router", "expert_gate_up",
                "expert_down", "shared_expert", "combine", "final_head", "sampling",
            )
        }
        buckets["shared_expert"].update(
            {
                "execution_status": "NOT_APPLICABLE_QWEN30_SOURCE_HAS_NO_SHARED_EXPERT",
                "timing_authority": "source catalog + exact runtime graph",
                "note": "No shared-expert tensor/operator exists in Qwen3-Coder-30B-A3B-Instruct's admitted graph.",
            }
        )
        for index, (sample, expected) in enumerate(zip(samples, plan, strict=True)):
            kernel, bucket, label = expected
            observed = sample.get("kernel_name")
            if observed != kernel:
                raise Qwen30CompleteTokenProfilerError(
                    f"trace kernel sequence mismatch at dispatch {index}: expected {kernel!r}, got {observed!r}"
                )
            duration = int(sample["gpu_us"])
            row = buckets[bucket]
            row["gpu_us"] += duration
            row["dispatches"] += 1
            row["operations"].append(label)

        timeline = self._timeline_attribution(samples, plan)
        for bucket, row in buckets.items():
            row["unambiguous_gpu_timeline_us"] = timeline["timeline_by_bucket_ns"][bucket] / 1_000.0
        buckets["command_graph_transition_gap"] = {
            "bucket": "command_graph_transition_gap",
            "gpu_us": 0,
            "unambiguous_gpu_timeline_us": timeline["idle_ns"] / 1_000.0,
            "dispatches": timeline["merged_busy_intervals"] - 1,
            "execution_status": "OBSERVED_GPU_TIMESTAMP_IDLE_OR_COMMAND_TOPOLOGY_GAPS",
            "timing_authority": "sweep of completed production Metal timestamp counter samples; not inferred",
            "operations": ["non-busy intervals inside the first-to-last observed GPU timestamp envelope"],
        }
        buckets["multi_stage_gpu_overlap"] = {
            "bucket": "multi_stage_gpu_overlap",
            "gpu_us": 0,
            "unambiguous_gpu_timeline_us": 0.0,
            "overlapped_gpu_timeline_us": timeline["multi_stage_overlap_ns"] / 1_000.0,
            "dispatches": 0,
            "execution_status": "OBSERVED_CONCURRENT_DIFFERENT_SEMANTIC_STAGES_NOT_ARBITRARILY_ASSIGNED",
            "timing_authority": "sweep of completed production Metal timestamp counter samples; not inferred",
            "operations": ["concurrent intervals whose active dispatches belong to more than one semantic bucket"],
        }
        buckets["hcli_overhead"] = {
            "bucket": "hcli_overhead",
            "gpu_us": 0,
            "unambiguous_gpu_timeline_us": 0.0,
            "dispatches": 0,
            "execution_status": "NOT_EXECUTED_HCLI_ADAPTER_ABSENT",
            "timing_authority": "not available until the native HCLI endpoint exists",
            "operations": [],
        }

        full_step = _required_mapping(_required_mapping(full, "execution", label="full result"), "step", label="full step")
        trace_step = _required_mapping(_required_mapping(trace, "execution", label="trace"), "step", label="trace step")
        wall_us = int(trace_step["elapsed_us_diagnostic_not_tps"])
        try:
            host_ledger = self._host_stage_attribution(trace, wall_us)
        except Qwen30CompleteTokenProfilerError as exc:
            host_ledger = {
                "available": True,
                "valid": False,
                "reason": str(exc),
                "by_bucket_us": {},
                "covered_us": 0.0,
                "coverage_percent": 0.0,
                "uncovered_us": float(wall_us),
                "interval_count": 0,
                "timer_origin": None,
                "declared_coverage_earned": None,
            }
        for bucket, row in buckets.items():
            row["source_bound_host_stage_wall_us"] = float(
                host_ledger["by_bucket_us"].get(bucket, 0.0)
            )
        kernel_us = sum(int(sample["gpu_us"]) for sample in samples)
        timestamp_envelope_us = timeline["envelope_ns"] / 1_000.0
        gpu_busy_union_us = timeline["busy_union_ns"] / 1_000.0
        semantic_timeline_us = timeline["semantic_timeline_ns"] / 1_000.0
        # The timestamp envelope is the only real device-side interval that
        # can be compared to the host wall without summing overlapping work.
        # A second, stricter coverage number says how much of host latency is
        # unambiguously tied to one required semantic stage or a measured
        # command-topology gap.  We do not silently allocate multi-stage
        # overlap or the host time outside the observed timestamp envelope.
        envelope_coverage = _percent(timestamp_envelope_us, wall_us)
        gpu_unambiguous_coverage = _percent(
            semantic_timeline_us + (timeline["idle_ns"] / 1_000.0), wall_us
        )
        host_coverage = float(host_ledger["coverage_percent"])
        effective_coverage = (
            host_coverage
            if host_ledger["available"] and host_ledger["valid"]
            else gpu_unambiguous_coverage
        )
        host_outside_envelope_us = max(0.0, wall_us - timestamp_envelope_us)
        elapsed_disagreement = int(full_step["elapsed_us_diagnostic_not_tps"]) - wall_us
        sampled_token_matches_baseline = trace_step.get("sampled_token_id") == full_step.get("sampled_token_id")
        profile_status = (
            "EARNED_REAL_COMPLETE_TOKEN_STAGE_PROFILE_DIAGNOSTIC_NOT_TPS"
            if sampled_token_matches_baseline
            and host_ledger["valid"]
            and effective_coverage >= MIN_WALL_COVERAGE_PERCENT
            and (host_ledger["available"] or timestamp_envelope_us <= wall_us)
            else (
                "PROFILE_BLOCKED_INVALID_SOURCE_BOUND_HOST_STAGE_LEDGER"
                if host_ledger["available"] and not host_ledger["valid"]
                else "PROFILE_BLOCKED_INSUFFICIENT_REAL_COMPLETE_TOKEN_STAGE_COVERAGE"
            )
        )
        ordered_buckets = sorted(
            buckets.values(),
            key=lambda row: (-int(row["gpu_us"]), str(row["bucket"])),
        )
        dominant = next(
            (row for row in ordered_buckets if row["bucket"] not in {"hcli_overhead", "shared_expert"}),
            None,
        )
        host_bucket_rows = sorted(
            (row for row in buckets.values() if row["bucket"] not in {"hcli_overhead", "shared_expert", "multi_stage_gpu_overlap"}),
            key=lambda row: (-float(row.get("source_bound_host_stage_wall_us", 0.0)), str(row["bucket"])),
        )
        dominant_host = host_bucket_rows[0] if host_bucket_rows and host_bucket_rows[0].get("source_bound_host_stage_wall_us", 0.0) else None
        source_trace = {
            "path": str(self.trace_result),
            "sha256": _sha256_file(self.trace_result),
            "schema": trace.get("schema"),
            "status": trace.get("status"),
        }
        source_full = {
            "path": str(self.full_result),
            "sha256": _sha256_file(self.full_result),
            "schema": full.get("schema"),
            "status": full.get("status"),
        }
        input_fingerprint = self._profile_input_fingerprint(trace, full, binding)
        return seal(
            {
                "schema": PROFILE_SCHEMA,
                "status": profile_status,
                "profiler_implementation_revision": PROFILE_IMPLEMENTATION_REVISION,
                "input_fingerprint_sha256": input_fingerprint,
                "recorded_at": _utc_now(),
                "binding": dict(binding),
                "inputs": {
                    "direct_full_token": source_full,
                    "production_command_buffer_trace": source_trace,
                },
                "execution": {
                    "all_48_layers_executed": True,
                    "final_norm_lm_head_device_argmax_executed": True,
                    "trace_input_token_id": trace.get("execution", {}).get("input_token_id") if isinstance(trace.get("execution"), Mapping) else None,
                    "trace_sampled_token_id": trace_step.get("sampled_token_id"),
                    "baseline_sampled_token_id": full_step.get("sampled_token_id"),
                    "sampled_token_matches_baseline": sampled_token_matches_baseline,
                    "raw_direct_graph_dispatches": len(samples),
                    "runtime_graph_dispatches_excluding_vector_decode": trace_step.get("metal_dispatches"),
                    "command_buffers_committed": _required_mapping(trace, "profiler", label="trace").get("command_buffers_committed"),
                },
                "timing": {
                    "complete_token_host_wall_us": wall_us,
                    "baseline_full_token_host_wall_us": full_step.get("elapsed_us_diagnostic_not_tps"),
                    "baseline_minus_trace_host_wall_us": elapsed_disagreement,
                    "production_gpu_work_sum_us": kernel_us,
                    "production_gpu_busy_union_us": gpu_busy_union_us,
                    "production_gpu_timestamp_envelope_us": timestamp_envelope_us,
                    "production_gpu_timestamp_envelope_first_ns": timeline["first_gpu_timestamp_ns"],
                    "production_gpu_timestamp_envelope_last_ns": timeline["last_gpu_timestamp_ns"],
                    "production_gpu_unambiguous_semantic_timeline_us": semantic_timeline_us,
                    "production_gpu_multi_stage_overlap_us": timeline["multi_stage_overlap_ns"] / 1_000.0,
                    "production_gpu_same_stage_overlap_us": timeline["same_stage_overlap_ns"] / 1_000.0,
                    "production_gpu_idle_or_command_topology_gap_us": timeline["idle_ns"] / 1_000.0,
                    "production_gpu_merged_busy_intervals": timeline["merged_busy_intervals"],
                    "host_wall_outside_gpu_timestamp_envelope_us": host_outside_envelope_us,
                    "complete_token_timestamp_envelope_coverage_percent": envelope_coverage,
                    "complete_token_gpu_unambiguous_stage_coverage_percent": gpu_unambiguous_coverage,
                    "source_bound_host_stage_ledger_available": host_ledger["available"],
                    "source_bound_host_stage_ledger_valid": host_ledger["valid"],
                    "source_bound_host_stage_ledger_reason": host_ledger["reason"],
                    "source_bound_host_stage_timer_origin": host_ledger["timer_origin"],
                    "source_bound_host_stage_interval_count": host_ledger["interval_count"],
                    "source_bound_host_stage_covered_us": host_ledger["covered_us"],
                    "source_bound_host_stage_coverage_percent": host_coverage,
                    "source_bound_host_stage_uncovered_us": host_ledger["uncovered_us"],
                    "source_bound_host_stage_declared_coverage_earned": host_ledger["declared_coverage_earned"],
                    "complete_token_unambiguous_stage_coverage_percent": effective_coverage,
                    # Kept as the public gate name so downstream consumers do
                    # not mistake a sum of overlapping GPU work for wall time.
                    "complete_token_wall_coverage_percent": effective_coverage,
                    "required_minimum_wall_coverage_percent": MIN_WALL_COVERAGE_PERCENT,
                    "timing_authority": {
                        "complete_token_host_wall": "actual runtime Instant around one all-48-layer direct-packed Metal token; diagnostic only",
                        "per_dispatch_gpu": "completed production command-buffer Metal timestamp counter samples",
                        "no_roofline_or_component_rate_substitution": True,
                    },
                },
                "buckets": ordered_buckets,
                "dominant_observed_bucket": {
                    "bucket": dominant.get("bucket") if dominant else None,
                    "gpu_us": dominant.get("gpu_us") if dominant else None,
                    "share_of_gpu_kernel_percent": _percent(dominant.get("gpu_us", 0), kernel_us) if dominant else 0.0,
                },
                "dominant_source_bound_host_stage": {
                    "bucket": dominant_host.get("bucket") if dominant_host else None,
                    "host_stage_wall_us": dominant_host.get("source_bound_host_stage_wall_us") if dominant_host else 0.0,
                    "share_of_host_wall_percent": _percent(
                        dominant_host.get("source_bound_host_stage_wall_us", 0.0), wall_us
                    ) if dominant_host else 0.0,
                },
                "next_optimization_condition": {
                    "profile_eligible": profile_status.startswith("EARNED_"),
                    "if_coverage_below_98": "add source-bound host-stage boundary timing around packed tensor I/O/allocation, route-id readback, command submission/wait, and pre-first/post-last GPU trace time before treating a bucket share as a complete-token latency decision",
                    "first_safe_candidate": "source-bound direct-packed gate/up paired-projection component candidate; CPU parity against the exact admitted binary payload is mandatory",
                },
                "claim_boundary": {
                    "real_native_metal_complete_token_trace": True,
                    "not_clean_sustained_base_true_tps": True,
                    "not_hcli_capability_tg_or_tournament_qualification": True,
                    "no_raw_bf16_or_mps_full_model": True,
                    "hcli_overhead_unavailable_until_actual_endpoint": True,
                },
            }
        )

    def _write_bottleneck(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        existing = _read_json(self.bottleneck_path)
        if existing is not None:
            try:
                checked_existing = verify(existing, label=str(self.bottleneck_path))
            except Exception:
                checked_existing = None
            if (
                checked_existing is not None
                and checked_existing.get("schema") == BOTTLENECK_SCHEMA
                and checked_existing.get("bottleneck_implementation_revision") == PROFILE_IMPLEMENTATION_REVISION
                and checked_existing.get("profile_seal_sha256") == profile.get("seal_sha256")
            ):
                return checked_existing
        timing = _required_mapping(profile, "timing", label="profile")
        dominant = _required_mapping(profile, "dominant_observed_bucket", label="profile")
        dominant_host = _required_mapping(profile, "dominant_source_bound_host_stage", label="profile")
        blocked = profile.get("status") != "EARNED_REAL_COMPLETE_TOKEN_STAGE_PROFILE_DIAGNOSTIC_NOT_TPS"
        record = seal(
            {
                "schema": BOTTLENECK_SCHEMA,
                "status": "SEALED_REAL_TOKEN_PROFILER_BOTTLENECK_REQUIRES_HOST_STAGE_COVERAGE"
                if blocked
                else "SEALED_REAL_TOKEN_PROFILER_BOTTLENECK_READY_FOR_KERNEL_EXPERIMENT",
                "bottleneck_implementation_revision": PROFILE_IMPLEMENTATION_REVISION,
                "recorded_at": _utc_now(),
                "binding": dict(_required_mapping(profile, "binding", label="profile")),
                "profile_path": str(self.profile_path),
                "profile_seal_sha256": profile.get("seal_sha256"),
                "profile_input_fingerprint_sha256": profile.get("input_fingerprint_sha256"),
                "observed": {
                    "dominant_gpu_bucket": dominant,
                    "dominant_source_bound_host_stage": dominant_host,
                    "complete_token_wall_coverage_percent": timing.get("complete_token_wall_coverage_percent"),
                    "complete_token_timestamp_envelope_coverage_percent": timing.get("complete_token_timestamp_envelope_coverage_percent"),
                    "host_wall_outside_gpu_timestamp_envelope_us": timing.get("host_wall_outside_gpu_timestamp_envelope_us"),
                    "command_graph_transition_gap_us": timing.get("production_gpu_idle_or_command_topology_gap_us"),
                    "multi_stage_gpu_overlap_us": timing.get("production_gpu_multi_stage_overlap_us"),
                },
                "next_experiment": {
                    "id": "qwen30-direct-packed-gate-up-pair-command-topology",
                    "hypothesis": "fusing each routed expert's independent gate/up direct-packed projections can remove one dispatch and encoder per route without changing the admitted packed values",
                    "acceptance": "exact CPU parity against the same admitted direct-binary payload; any speed observation remains component-only until integrated and re-profiled on a complete token",
                    "reopen_condition": "the current artifact binding, direct-binary layout, Qwen30 runtime graph, or complete-token trace changes",
                },
                "claim_boundary": {
                    "not_a_tps_or_tg_receipt": True,
                    "not_a_runtime_or_capability_pass": True,
                    "no_roofline_substitution": True,
                },
            }
        )
        _atomic_json(self.bottleneck_path, record)
        row = seal(
            {
                "schema": "hawking.ascension.kernel_genome.v1",
                "record_id": f"qwen30-complete-token-profiler:{profile.get('input_fingerprint_sha256')}",
                "recorded_at": _utc_now(),
                "model": "qwen30",
                "status": record["status"],
                "source": "real_direct_packed_complete_token_trace",
                "profile_path": str(self.profile_path),
                "profile_seal_sha256": profile.get("seal_sha256"),
                "bottleneck_path": str(self.bottleneck_path),
                "bottleneck_seal_sha256": record["seal_sha256"],
                "next_experiment": record["next_experiment"],
                "claim_boundary": "profile/experiment handoff only; no model TPS or qualification claim",
            }
        )
        _append_jsonl_once(self.shared_kernel_genome, row, record_id=str(row["record_id"]))
        return record

    def _reconcile_microbench(
        self, binding: Mapping[str, Any], profile: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Seal a real isolated kernel result without promoting its scope.

        The raw result is produced by the Rust component executable outside this
        watcher so it can be scheduled only in a GPU quiet window.  This method
        never creates it, never substitutes a source BF16 tensor, and refuses
        a result whose admission binding or CPU parity fields are incomplete.
        """

        raw = _read_json(self.microbench_raw)
        if raw is None:
            return None
        if raw.get("schema") != MICROBENCH_RAW_SCHEMA or raw.get("status") != MICROBENCH_RAW_STATUS:
            return None
        observed = _required_mapping(raw, "binding", label="gate/up raw component result")
        for field in ("manifest_seal_sha256", "source_audit_seal_sha256", "source_revision"):
            if observed.get(field) != binding.get(field):
                raise Qwen30CompleteTokenProfilerError(
                    f"gate/up raw component result has mismatched {field}"
                )
        candidate = _required_mapping(raw, "candidate", label="gate/up raw component result")
        baseline = _required_mapping(candidate, "baseline_command_topology", label="gate/up candidate")
        paired = _required_mapping(candidate, "candidate_command_topology", label="gate/up candidate")
        if (
            candidate.get("id") != "qwen30-direct-packed-gate-up-pair-command-topology"
            or baseline.get("compute_dispatches") != 2
            or paired.get("compute_dispatches") != 1
            or candidate.get("direct_packed_layout") != "HQ30G1B1, group_size=128, FP16 scales plus sign bits"
        ):
            raise Qwen30CompleteTokenProfilerError("gate/up component topology/layout contract is invalid")
        parity = _required_mapping(raw, "parity", label="gate/up raw component result")
        if parity.get("baseline_within_tolerance") is not True or parity.get("paired_within_tolerance") is not True:
            raise Qwen30CompleteTokenProfilerError("gate/up component does not have CPU parity over the admitted direct pack")
        timing = _required_mapping(raw, "timing", label="gate/up raw component result")
        baseline_timing = _required_mapping(timing, "baseline_two_dispatch", label="gate/up raw component timing")
        paired_timing = _required_mapping(timing, "candidate_one_dispatch", label="gate/up raw component timing")
        for row in (baseline_timing, paired_timing):
            value = row.get("host_wall_us_p50")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise Qwen30CompleteTokenProfilerError("gate/up component result lacks a positive observed p50")
        if "base_true_tps" in raw or "tokens_per_second" in _canonical(raw).decode("utf-8"):
            raise Qwen30CompleteTokenProfilerError("gate/up component result attempts to make a forbidden model TPS claim")
        raw_sha256 = _sha256_file(self.microbench_raw)
        existing = _read_json(self.microbench_receipt)
        if existing is not None:
            try:
                checked_existing = verify(existing, label=str(self.microbench_receipt))
            except Exception:
                checked_existing = None
            existing_raw = (
                checked_existing.get("raw_result") if isinstance(checked_existing, Mapping) else None
            )
            existing_profile = (
                checked_existing.get("complete_token_profile") if isinstance(checked_existing, Mapping) else None
            )
            if (
                checked_existing is not None
                and checked_existing.get("schema") == MICROBENCH_RECEIPT_SCHEMA
                and checked_existing.get("component_receipt_implementation_revision") == PROFILE_IMPLEMENTATION_REVISION
                and isinstance(existing_raw, Mapping)
                and existing_raw.get("sha256") == raw_sha256
                and isinstance(existing_profile, Mapping)
                and existing_profile.get("seal_sha256") == profile.get("seal_sha256")
            ):
                return checked_existing
        receipt = seal(
            {
                "schema": MICROBENCH_RECEIPT_SCHEMA,
                "status": "EARNED_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_NOT_MODEL_TPS",
                "component_receipt_implementation_revision": PROFILE_IMPLEMENTATION_REVISION,
                "recorded_at": _utc_now(),
                "binding": dict(binding),
                "raw_result": {
                    "path": str(self.microbench_raw),
                    "sha256": raw_sha256,
                    "schema": raw.get("schema"),
                    "status": raw.get("status"),
                },
                "complete_token_profile": {
                    "path": str(self.profile_path),
                    "seal_sha256": profile.get("seal_sha256"),
                    "status": profile.get("status"),
                },
                "candidate": dict(candidate),
                "parity": dict(parity),
                "timing": {
                    "baseline_two_dispatch": dict(baseline_timing),
                    "candidate_one_dispatch": dict(paired_timing),
                    "p50_component_host_wall_delta_us": timing.get("p50_component_host_wall_delta_us"),
                    "p50_component_host_wall_speedup_ratio": timing.get("p50_component_host_wall_speedup_ratio"),
                    "authority": "real completed direct-packed component command buffers only; no token/runtime rate substitution",
                },
                "integration_gate": {
                    "runtime_not_modified_by_this_receipt": True,
                    "requires_runtime_integration_parity": True,
                    "requires_new_all_48_layer_complete_token_profile": True,
                    "requires_clean_hcli_benchmark_before_any_tps_claim": True,
                },
                "claim_boundary": {
                    "cpu_parity_uses_same_admitted_direct_packed_representation_only": True,
                    "not_a_full_layer_full_model_generation_hcli_or_tps_result": True,
                    "not_a_tg_or_tournament_receipt": True,
                    "no_raw_bf16_or_mps_model_path": True,
                },
            }
        )
        _atomic_json(self.microbench_receipt, receipt)
        row = seal(
            {
                "schema": "hawking.ascension.kernel_genome.v1",
                "record_id": f"qwen30-direct-packed-gate-up-pair:{raw_sha256}:{profile.get('seal_sha256')}",
                "recorded_at": _utc_now(),
                "model": "qwen30",
                "status": receipt["status"],
                "component_receipt_path": str(self.microbench_receipt),
                "component_receipt_seal_sha256": receipt["seal_sha256"],
                "candidate": candidate,
                "next_gate": receipt["integration_gate"],
                "claim_boundary": "component candidate only; any full-token effect remains unearned until integration and exact re-profile",
            }
        )
        _append_jsonl_once(self.shared_kernel_genome, row, record_id=str(row["record_id"]))
        return receipt

    def _status(self, phase: str, **fields: Any) -> dict[str, Any]:
        prior = _read_json(self.status_path) or {}
        document = {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(prior.get("heartbeat", 0)) + 1,
            "phase": phase,
            **fields,
            "claim_boundary": {
                "not_a_tps_hcli_tg_capability_or_tournament_result": True,
                "requires_real_direct_packed_full_token_and_production_trace": True,
                "raw_bf16_and_mps_full_model_are_forbidden": True,
            },
        }
        _atomic_json(self.status_path, document)
        return document

    def run_cycle(self) -> dict[str, Any]:
        try:
            binding = self._binding()
            full = self._full_result(binding)
        except Qwen30CompleteTokenProfilerError as exc:
            return self._status("WAITING_FOR_REAL_ADMITTED_DIRECT_PACKED_FULL_TOKEN", error=str(exc))

        trace = self._trace_result(binding)
        if trace is not None:
            try:
                profile_binding = self._runtime_bound_profile_binding(binding, trace, full)
            except Qwen30CompleteTokenProfilerError as exc:
                return self._status(
                    "RUNTIME_OWNED_PRODUCTION_TRACE_RECEIPT_REJECTED_REOPENED",
                    runtime_profile_result_path=str(self.trace_result),
                    canonical_runtime_receipt_path=str(self.runtime_receipt),
                    reason="the exact trace is present but its canonical runtime binding is missing, revoked, or mismatched",
                    trace_rejection=str(exc),
                )
            profile = self._existing_profile(trace, full, profile_binding)
            if profile is None:
                self._archive_prior_profile_outputs(
                    replacement_input_fingerprint=self._profile_input_fingerprint(
                        trace, full, profile_binding
                    )
                )
                profile = self._attribute(trace, full, profile_binding)
                _atomic_json(self.profile_path, profile)
            bottleneck = self._write_bottleneck(profile)
            microbench = self._reconcile_microbench(profile_binding, profile)
            return self._status(
                "COMPLETE_TOKEN_TRACE_RECONCILED",
                profile_path=str(self.profile_path),
                profile_status=profile.get("status"),
                profile_seal_sha256=profile.get("seal_sha256"),
                bottleneck_path=str(self.bottleneck_path),
                bottleneck_status=bottleneck.get("status"),
                microbench_receipt_path=str(self.microbench_receipt) if microbench is not None else None,
                microbench_status=microbench.get("status") if microbench is not None else "WAITING_FOR_ISOLATED_GPU_QUIET_WINDOW_RESULT",
                wall_coverage_percent=profile.get("timing", {}).get("complete_token_wall_coverage_percent") if isinstance(profile.get("timing"), Mapping) else None,
                canonical_runtime_receipt_path=str(self.runtime_receipt),
                canonical_runtime_receipt_seal_sha256=profile_binding.get("canonical_runtime_receipt_seal_sha256"),
                runtime_executable_sha256=profile_binding.get("runtime_executable_sha256"),
                gate_up_swiglu_kernel=profile_binding.get("gate_up_swiglu_kernel"),
            )
        trace_error = self._trace_validation_error(binding)
        return self._status(
            "RUNTIME_OWNED_PRODUCTION_TRACE_RECEIPT_REJECTED_REOPENED"
            if trace_error is not None
            else "WAITING_FOR_RUNTIME_OWNED_PRODUCTION_TRACE_RECEIPT",
            runtime_profile_result_path=str(self.trace_result),
            reason="the runtime watcher owns the gpu_prod trace stage; this sidecar will only consume a source-bound raw-dispatch receipt",
            trace_rejection=trace_error,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("once", "watch"))
    parser.add_argument("--physical-root", type=Path, default=DEFAULT_PHYSICAL_ROOT)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.idle_seconds <= 0:
        raise SystemExit("--idle-seconds must be positive")
    profiler = Qwen30CompleteTokenProfiler(physical_root=args.physical_root)
    if args.command == "once":
        print(json.dumps(profiler.run_cycle(), sort_keys=True))
        return 0
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not _STOP:
        print(json.dumps(profiler.run_cycle(), sort_keys=True), flush=True)
        deadline = time.monotonic() + args.idle_seconds
        while not _STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
