"""Negative controls for DeviceCompiler.

A placeholder that claims compiled identity, a source digest sold as a
metallib, or an organ recorded COMPILED when Metal produced nothing would
let Odyssey lower nothing and report success. These tests make each of
those refusals fire. pytest.skip that actually fires is a P0.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import device_compiler as dc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _cfg() -> dict:
    return {
        "hidden_size": 64,
        "intermediate_size": 128,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "model_type": "qwen3",
    }


def _slot(organ: str, shape: dict | None = None) -> dict:
    return {
        "organ": organ,
        "status": dc.NATIVE_UNMEASURED,
        "occupying": {
            "kind": dc.NATIVE_UNMEASURED,
            "compiled_kernel": None,
            "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
        },
        "specimen_shape": shape,
        "name_is_not_a_compiled_kernel": True,
        "why": dc.NAME_IS_NOT_A_COMPILED_KERNEL,
    }


def _plan(organs: list[tuple[str, dict | None]] | None = None) -> dict:
    if organs is None:
        organs = [
            ("mlp_down", {"rows": 64, "cols": 128, "extents": [64, 128]}),
            ("mlp_gate_up", {"rows": 128, "cols": 64, "extents": [128, 64]}),
        ]
    return {
        "route": "PLAN-THEN-COMPILE",
        "plan": [_slot(name, shape) for name, shape in organs],
        "n_compiled": 0,
        "n_native_unmeasured": len(organs),
    }


def _placeholder(**overrides) -> dict:
    row = {
        "kind": "PLACEHOLDER",
        "value": "deadbeef" * 4,
        "shader_hash": "deadbeef" * 4,
        "entry_point": "organ_mlp_down_gemv",
        "pipeline": {
            "object": dc.PIPELINE_OBJECT,
            "created": True,
            "function_found": True,
        },
        "archive_bytes": 16,
        "source_sha256": "deadbeef" * 4,
    }
    row.update(overrides)
    return row


class LyingBackend:
    """Claims compile success. That is the defect: lower nothing, report compiled."""

    def __init__(self, mode: str = "source_hash") -> None:
        self.mode = mode

    def compile_jobs(self, jobs):
        results = []
        for job in jobs:
            src_h = job.source_sha256
            if self.mode == "source_hash":
                digest = src_h
                archive_path = job.archive_path
                Path(archive_path).write_text(job.source)
            elif self.mode == "no_file":
                digest = "ab" * 32
                archive_path = job.archive_path
            elif self.mode == "absent_kind":
                results.append(
                    {
                        "id": job.organ,
                        "ok": True,
                        "entry_point": job.entry_point,
                        "function_found": True,
                        "pipeline_created": True,
                        "pipeline_object": dc.PIPELINE_OBJECT,
                        "archive_sha256": None,
                        "archive_bytes": None,
                        "archive_path": None,
                        "source_sha256": src_h,
                    }
                )
                continue
            else:
                digest = src_h
                archive_path = job.archive_path
            results.append(
                {
                    "id": job.organ,
                    "ok": True,
                    "entry_point": job.entry_point,
                    "function_found": True,
                    "pipeline_created": True,
                    "pipeline_object": dc.PIPELINE_OBJECT,
                    "archive_sha256": digest,
                    "archive_bytes": 99,
                    "archive_path": archive_path,
                    "source_sha256": src_h,
                }
            )
        return {
            "ok": True,
            "n_devices": 1,
            "device": "liar",
            "results": results,
            "error": None,
            "backend": "lying",
            "created_command_queue": False,
            "dispatched": False,
        }


def test_placeholder_claiming_compiled_identity_is_refused():
    """NEGATIVE CONTROL: the defect that would let Odyssey lower nothing and report success."""
    identity = _placeholder()
    assert dc.is_placeholder_compiled_identity(identity, source_sha256="deadbeef" * 4)
    assert dc.is_genuine_compiled_identity(identity) is False
    with pytest.raises(dc.PlaceholderCompiledIdentity):
        dc.refuse_placeholder(identity, source_sha256="deadbeef" * 4)


def test_kernel_library_absent_identity_is_not_compiled():
    """NEGATIVE CONTROL: KERNEL_LIBRARY's ABSENT metallib_digest is not compiled."""
    identity = {
        "kind": "ABSENT",
        "value": None,
        "unit": "metallib_digest",
        "absent_reason": "NOETIC_PARENT_A records no on-disk .metallib",
    }
    assert dc.is_placeholder_compiled_identity(identity)
    with pytest.raises(dc.PlaceholderCompiledIdentity):
        dc.refuse_placeholder(identity)


def test_source_digest_is_not_a_compiled_identity():
    """NEGATIVE CONTROL: sha256(source) sold as shader_hash is a placeholder."""
    source = "kernel void organ_mlp_down_gemv() { }"
    digest = hashlib.sha256(source.encode()).hexdigest()
    identity = {
        "kind": dc.COMPILED_IDENTITY_KIND,
        "shader_hash": digest,
        "value": digest,
        "entry_point": "organ_mlp_down_gemv",
        "pipeline": {
            "object": dc.PIPELINE_OBJECT,
            "created": True,
            "function_found": True,
        },
        "archive_bytes": 8,
        "source_sha256": digest,
    }
    assert dc.is_placeholder_compiled_identity(
        identity, source_sha256=digest, source=source, entry_point="organ_mlp_down_gemv"
    )
    with pytest.raises(dc.PlaceholderCompiledIdentity) as exc:
        dc.refuse_placeholder(
            identity, source_sha256=digest, source=source, entry_point="organ_mlp_down_gemv"
        )
    assert "source" in str(exc.value).lower()


def test_lying_backend_source_hash_is_not_recorded_compiled():
    """NEGATIVE CONTROL: a backend that hashes the source cannot mint COMPILED."""
    plan = _plan()
    out = dc.lower_plan(
        plan,
        specimen_id="synth",
        family="dense_swiglu_transformer",
        config=_cfg(),
        native_architectures=["qwen2", "qwen3moe"],
        model_type="qwen3",
        backend=LyingBackend("source_hash"),
    )
    assert out["n_compiled"] == 0
    assert out["n_placeholder_refused"] >= 1
    for slot in out["plan"]:
        assert slot["status"] == dc.NATIVE_UNMEASURED
        assert slot["occupying"]["kind"] == dc.NATIVE_UNMEASURED
        assert slot["compiled_identity"] is None
        assert slot["status"] != dc.COMPILED
    compiled, planned = dc.split_compiled_and_planned(out["nx_fragment"])
    assert compiled == []
    assert planned
    assert out["nx_fragment"]["native_kernel"]["status"] != "BOUND"


def test_lying_backend_without_archive_file_is_not_compiled():
    plan = _plan([("mlp_down", {"rows": 64, "cols": 128, "extents": [64, 128]})])
    out = dc.lower_plan(plan, config=_cfg(), backend=LyingBackend("no_file"))
    assert out["n_compiled"] == 0
    assert out["plan"][0]["status"] == dc.NATIVE_UNMEASURED


def test_lying_backend_absent_kind_row_is_not_compiled():
    plan = _plan([("mlp_down", {"rows": 64, "cols": 128, "extents": [64, 128]})])
    out = dc.lower_plan(plan, config=_cfg(), backend=LyingBackend("absent_kind"))
    assert out["n_compiled"] == 0
    assert out["plan"][0]["status"] == dc.NATIVE_UNMEASURED


def test_unavailable_metal_stays_native_unmeasured():
    """Fail closed: no device means no COMPILED organ, not a fake identity."""
    plan = _plan()
    out = dc.lower_plan(
        plan,
        config=_cfg(),
        backend=dc.UnavailableMetalBackend("MTLCreateSystemDefaultDevice returned nil"),
    )
    assert out["n_compiled"] == 0
    assert out["ok"] is False
    for slot in out["plan"]:
        assert slot["status"] == dc.NATIVE_UNMEASURED
        assert slot["compiled_identity"] is None
    compiled, _planned = dc.split_compiled_and_planned(out["nx_fragment"])
    assert compiled == []


def test_unsupported_organ_stays_native_unmeasured_not_a_gemv_placeholder():
    """NEGATIVE CONTROL: deltanet is not lowered by relabelling a GEMV."""
    plan = _plan([("deltanet", {"rows": 64, "cols": 64})])
    out = dc.lower_plan(
        plan, config=_cfg(), backend=dc.UnavailableMetalBackend("unused")
    )
    assert out["n_compiled"] == 0
    assert out["plan"][0]["organ"] == "deltanet"
    assert out["plan"][0]["status"] == dc.NATIVE_UNMEASURED
    assert "placeholder" in out["plan"][0]["why"].lower() or "no honest lowering" in out["plan"][0]["why"]


def test_empty_plan_is_refused_not_compiled():
    out = dc.lower_plan({"route": "PLAN-THEN-COMPILE", "plan": []}, config=_cfg())
    assert out["ok"] is False
    assert out["error"] == "no_plan"
    assert out["n_compiled"] == 0
    assert out["nx_fragment"]["native_kernel"]["status"] == "ABSENT"


def test_emit_shader_carries_real_entry_point_in_source():
    emitted = dc.emit_shader(
        "mlp_down",
        shape={"rows": 64, "cols": 128},
        config=_cfg(),
        specimen_id="synth",
    )
    assert emitted is not None
    assert f"kernel void {emitted['entry_point']}(" in emitted["source"]
    assert "BAKED_ROWS 64" in emitted["source"]
    assert "BAKED_COLS 128" in emitted["source"]


def test_gqa_shader_is_attention_not_a_relabelled_gemv():
    emitted = dc.emit_shader(
        "gqa_attention",
        shape={"q_rows": 64, "kv_rows": 32, "head_dim": 16},
        config=_cfg(),
    )
    assert emitted is not None
    src = emitted["source"]
    assert "organ_gqa_decode" in src
    assert "BAKED_N_Q" in src
    assert "BAKED_N_KV" in src
    assert "rsqrt" in src
    assert "exp(" in src


def test_nx_fragment_distinguishes_compiled_from_planned():
    planned_slot = _slot("embed", {"rows": 256, "cols": 64})
    planned_slot["status"] = dc.NATIVE_UNMEASURED
    planned_slot["compiled_identity"] = None
    # A placeholder must not appear as compiled on the NX even if status is COMPILED.
    fake = _slot("mlp_down", {"rows": 64, "cols": 128})
    fake["status"] = dc.COMPILED
    fake["occupying"] = {"kind": dc.COMPILED, "compiled_kernel": "nope"}
    fake["compiled_identity"] = _placeholder()
    nx = dc.nx_fragment_from_slots(
        [planned_slot, fake], specimen_id="synth", family="dense_swiglu_transformer"
    )
    compiled, planned = dc.split_compiled_and_planned(nx)
    assert compiled == []
    organs = {k["organ"] for k in planned}
    assert organs == {"embed", "mlp_down"}
    assert nx["native_kernel"]["status"] == "ABSENT"


def test_qwen3_dense_is_not_mapped_onto_moe_or_qwen2():
    blocker = dc.qwen3_dense_gguf_blocker(
        ["qwen2", "qwen2.5", "qwen", "qwen2moe", "qwen3moe", "qwen-moe"],
        family="dense_swiglu_transformer",
        model_type="qwen3",
    )
    assert blocker["id"] == dc.QWEN3_DENSE_GGUF_BLOCKER
    assert blocker["holds"] is True
    assert blocker["includes_qwen3_dense"] is False
    assert blocker["includes_qwen3moe"] is True
    assert blocker["qwen3moe_is_different_family"] is True
    assert blocker["did_not_map_dense_onto_moe_arm"] is True
    assert blocker["did_not_map_dense_onto_qwen2_arm"] is True
    assert "qwen3" not in blocker["architectures"]
    assert "qwen3moe" in blocker["architectures"]


def test_qwen3_dense_blocker_travels_with_the_lowering():
    out = dc.lower_plan(
        _plan(),
        family="dense_swiglu_transformer",
        model_type="qwen3",
        native_architectures=["qwen2", "qwen3moe"],
        backend=dc.UnavailableMetalBackend("nil"),
    )
    b = out["qwen3_dense_gguf_blocker"]
    assert b["id"] == dc.QWEN3_DENSE_GGUF_BLOCKER
    assert b["holds"] is True
    assert out["nx_fragment"]["qwen3_dense_gguf_blocker"]["id"] == dc.QWEN3_DENSE_GGUF_BLOCKER


def test_live_metal_compile_is_genuine_or_unavailable():
    """Live path: a real pipeline, or fail closed. Never a placeholder."""
    emitted = dc.emit_shader(
        "mlp_down",
        shape={"rows": 8, "cols": 8},
        config={"hidden_size": 8, "intermediate_size": 8},
        specimen_id="probe",
    )
    assert emitted is not None
    result = dc.compile_source(emitted["source"], emitted["entry_point"], organ="mlp_down")
    if result["ok"] is True:
        identity = result["compiled_identity"]
        assert dc.is_genuine_compiled_identity(
            identity,
            source_sha256=result["source_sha256"],
            source=emitted["source"],
            entry_point=emitted["entry_point"],
        )
        assert identity["kind"] == dc.COMPILED_IDENTITY_KIND
        assert identity["entry_point"] == emitted["entry_point"]
        assert identity["shader_hash"] != result["source_sha256"]
        assert identity["pipeline"]["created"] is True
        assert identity["pipeline"]["object"] == dc.PIPELINE_OBJECT
        assert identity["pipeline"]["function_found"] is True
        assert identity["archive_bytes"] > 0
        assert identity["pipeline"]["created_command_queue"] is False
    else:
        assert result["compiled_identity"] is None
        why = str(result.get("error") or "")
        assert why, "unavailable compile must name why, not mint an identity"


def test_lower_plan_live_never_marks_placeholder_compiled():
    out = dc.lower_plan(
        _plan(),
        specimen_id="synth",
        family="dense_swiglu_transformer",
        config=_cfg(),
        native_architectures=["qwen2", "qwen3moe"],
        model_type="qwen3",
    )
    for slot in out["plan"]:
        if slot["status"] == dc.COMPILED:
            identity = slot["compiled_identity"]
            assert dc.is_genuine_compiled_identity(
                identity,
                source_sha256=identity.get("source_sha256"),
                entry_point=identity.get("entry_point"),
            )
            assert identity["shader_hash"] != identity.get("source_sha256")
        else:
            assert slot["status"] == dc.NATIVE_UNMEASURED
            assert slot["compiled_identity"] is None
    compiled, _planned = dc.split_compiled_and_planned(out["nx_fragment"])
    assert len(compiled) == out["n_compiled"]
    if out["n_compiled"] == 0:
        assert out["ok"] is False
        assert out["nx_fragment"]["native_kernel"]["status"] == "ABSENT"


def test_helper_swift_creates_no_command_queue():
    """Compile is the compiler service. A queue would make this a GPU user."""
    src = dc.HELPER_SWIFT
    assert "makeCommandQueue" not in src
    assert "newCommandQueue" not in src
    assert "makeLibrary" in src
    assert "makeComputePipelineState" in src
    assert "MTLBinaryArchive" in src or "makeBinaryArchive" in src
    assert "created_command_queue" in src


def test_receipt_is_static_only_and_records_the_placeholder_control():
    out = dc.build()
    assert out.name == "DEVICE_COMPILER.json"
    assert out.parent == RECEIPTS
    doc = json.loads(out.read_text())
    assert doc["schema"] == dc.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["is_a_measurement"] is False
    assert doc["created_command_queue"] is False
    assert doc["dispatched"] is False
    assert doc["placeholder_negative_control"]["caught"] is True
    assert doc["placeholder_negative_control"]["source_digest_caught_as_placeholder"] is True
    assert doc["qwen3_dense_gguf_blocker"]["id"] == dc.QWEN3_DENSE_GGUF_BLOCKER
    assert doc["qwen3_dense_gguf_blocker"]["did_not_map_dense_onto_moe_arm"] is True
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    lowering = doc["lowering"]
    for slot in lowering["plan"]:
        if slot["status"] == dc.COMPILED:
            assert slot["compiled_identity"]["kind"] == dc.COMPILED_IDENTITY_KIND
            assert slot["compiled_identity"]["shader_hash"]
            assert slot["compiled_identity"]["entry_point"]
        else:
            assert slot["status"] == dc.NATIVE_UNMEASURED
            assert slot["compiled_identity"] is None


def test_receipt_refuses_hardware_fields():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "DEVICE_COMPILER_SHOULD_NOT_EXIST.json",
            {"schema": "x", "tps": 1.0},
            "tools/future/test_device_compiler.py",
        )
    for field in HARDWARE_FIELDS:
        assert field not in {"schema", "version", "purpose"}


def test_no_pytest_skip_in_this_file():
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            assert name != "skip", "pytest.skip that actually fires is a P0"
