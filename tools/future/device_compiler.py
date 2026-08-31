"""DeviceCompiler — lower a PLAN-THEN-COMPILE plan into compiled Metal pipelines.

Contract
--------
IN:  a KernelPlanner plan (the evidence object with route + plan[] slots).
     Each slot names an organ, an occupying kind (NATIVE_UNMEASURED or
     COMPILED), and the specimen shape the planner derived. Optional:
     specimen_id, family, config, native GGUF match arms.

OUT: a lowering of that plan.
     Every organ is either COMPILED with a genuine compiled identity, or
     NATIVE_UNMEASURED with a reason. The same identities are written onto
     an NX fragment so a later stage can tell a compiled kernel from a
     planned one without trusting the planner's occupying kind.

Genuine compiled identity (all of these, not any one):
     * kind == METAL_PIPELINE
     * entry_point is a `kernel void <name>(` in the shader that was looked
       up in the compiled MTLLibrary (function_found)
     * an MTLComputePipelineState was created from that function
       (pipeline.object, pipeline.created)
     * shader_hash is sha256 of the serialized MTLBinaryArchive bytes, AND
       that digest is not the source digest

A placeholder claiming compiled identity is refused. The organ stays
NATIVE_UNMEASURED and says why. Status COMPILED is never recorded for a
source hash, an ABSENT/PLACEHOLDER kind, a hardcoded digest, or a pipeline
that was not created. That is the defect that would let Odyssey lower
nothing and report success.

This is COMPILE_TIME_SCIENCE_ONLY. Compiling a shader and creating a
pipeline state is the Metal compiler service, not a GPU lease: no command
queue is created and nothing is dispatched. Absence of a Metal device in
this process is recorded; it is not rewritten as "the host has no GPU".
No hardware measurement is claimed.

The qwen3-dense GGUF match-arm gap is a named blocker
(QWEN3_DENSE_GGUF_MATCH_ARM_ABSENT). qwen3moe is a different family. Dense
qwen3 is not mapped onto the moe arm, and is not silently aliased to qwen2
(Qwen3 applies QK-RMSNorm before RoPE; QwenDense does not load those
tensors).

    python3 tools/future/device_compiler.py --build
    python3 -m pytest tools/future/test_device_compiler.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from tools.future._common import RECEIPTS, REPO, write_receipt

RECEIPT = "DEVICE_COMPILER.json"
SCHEMA = "hawking.future.device_compiler.v1"
RECORDED_BY = "tools/future/device_compiler.py"
VERSION = 1

PASSED = "PASSED"
FAILED = "FAILED"
REFUSED = "REFUSED"
BLOCKED = "BLOCKED"

NATIVE_UNMEASURED = "NATIVE_UNMEASURED"
COMPILED = "COMPILED"

COMPILED_IDENTITY_KIND = "METAL_PIPELINE"
PIPELINE_OBJECT = "MTLComputePipelineState"
NAME_IS_NOT_A_COMPILED_KERNEL = (
    "A shared organ name is not a compiled kernel for this body."
)
PLACEHOLDER_REFUSED = "placeholder compiled identity refused"

QWEN3_DENSE_GGUF_BLOCKER = "QWEN3_DENSE_GGUF_MATCH_ARM_ABSENT"

# Organs this compiler can actually lower. Anything else stays NATIVE_UNMEASURED.
LOWABLE_ORGANS = frozenset(
    {
        "mlp_down",
        "mlp_gate_up",
        "lm_head",
        "embed",
        "rmsnorm",
        "gqa_attention",
    }
)

PLACEHOLDER_KINDS = frozenset(
    {
        "ABSENT",
        "PLACEHOLDER",
        "FAKE",
        "SYNTHETIC",
        "SOURCE_HASH",
        "SOURCE",
        "PLANNED",
        "NATIVE_UNMEASURED",
    }
)


class PlaceholderCompiledIdentity(ValueError):
    """A claimed compiled identity that is not a Metal pipeline."""


class DeviceCompilerError(ValueError):
    """The compiler cannot run this plan."""


# ---------------------------------------------------------------------------
# Contract helpers. These are the gate: a later stage asks them, not the
# occupying kind the planner wrote.
# ---------------------------------------------------------------------------


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _int_pos(*candidates: Any) -> int | None:
    for raw in candidates:
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, float) and raw > 0 and raw == int(raw):
            return int(raw)
    return None


def placeholder_reasons(
    identity: Any,
    *,
    source_sha256: str | None = None,
    source: str | None = None,
    entry_point: str | None = None,
) -> list[str]:
    """Why this object is not a compiled Metal pipeline. Empty means genuine."""
    reasons: list[str] = []
    if not isinstance(identity, Mapping):
        return ["compiled_identity is not an object"]
    kind = identity.get("kind")
    if kind in PLACEHOLDER_KINDS or kind is None:
        reasons.append(f"kind={kind!r} is not {COMPILED_IDENTITY_KIND}")
    elif kind != COMPILED_IDENTITY_KIND:
        reasons.append(
            f"kind={kind!r} is not {COMPILED_IDENTITY_KIND}; "
            "a MEASURED digest without a Metal pipeline is not compiled"
        )
    entry = identity.get("entry_point") or entry_point
    if not (isinstance(entry, str) and entry.strip()):
        reasons.append("entry_point missing")
        entry = None
    src = source if isinstance(source, str) else identity.get("source")
    if isinstance(src, str) and entry:
        needle = f"kernel void {entry}("
        if needle not in src:
            reasons.append(
                f"entry_point={entry!r} is not a kernel in the shader source"
            )
    shader_hash = identity.get("shader_hash")
    if shader_hash is None and identity.get("kind") == COMPILED_IDENTITY_KIND:
        shader_hash = identity.get("value")
    if not _is_sha256(shader_hash):
        reasons.append("shader_hash is not a sha256 of compiled archive bytes")
    src_hash = source_sha256 or identity.get("source_sha256")
    if _is_sha256(shader_hash) and _is_sha256(src_hash) and shader_hash == src_hash:
        reasons.append(
            "shader_hash equals source_sha256; a source digest is not a compiled "
            "metallib / MTLBinaryArchive"
        )
    if isinstance(shader_hash, str) and shader_hash.lower() in {
        "deadbeef" * 4,
        "0" * 64,
        "a" * 64,
        "abc123" * 8,
    }:
        reasons.append("shader_hash is a hardcoded placeholder digest")
    pipeline = identity.get("pipeline")
    if not isinstance(pipeline, Mapping):
        reasons.append("pipeline object missing")
    else:
        if pipeline.get("object") != PIPELINE_OBJECT:
            reasons.append(
                f"pipeline.object={pipeline.get('object')!r} is not {PIPELINE_OBJECT}"
            )
        if pipeline.get("created") is not True:
            reasons.append("MTLComputePipelineState was not created")
        if pipeline.get("function_found") is not True:
            reasons.append("entry point was not looked up in the compiled MTLLibrary")
    archive_bytes = identity.get("archive_bytes")
    if not isinstance(archive_bytes, int) or archive_bytes <= 0:
        reasons.append("compiled MTLBinaryArchive is empty or absent")
    if identity.get("kind") == "ABSENT" or identity.get("value") in (None, "", [], {}):
        if "kind=" not in " ".join(reasons):
            reasons.append("compiled_identity value is absent")
    return reasons


def is_placeholder_compiled_identity(
    identity: Any,
    *,
    source_sha256: str | None = None,
    source: str | None = None,
    entry_point: str | None = None,
) -> bool:
    return bool(
        placeholder_reasons(
            identity,
            source_sha256=source_sha256,
            source=source,
            entry_point=entry_point,
        )
    )


def refuse_placeholder(
    identity: Any,
    *,
    source_sha256: str | None = None,
    source: str | None = None,
    entry_point: str | None = None,
) -> Mapping[str, Any]:
    """Raise if `identity` is a placeholder claiming to be compiled."""
    reasons = placeholder_reasons(
        identity,
        source_sha256=source_sha256,
        source=source,
        entry_point=entry_point,
    )
    if reasons:
        raise PlaceholderCompiledIdentity("; ".join(reasons))
    if not isinstance(identity, Mapping):
        raise PlaceholderCompiledIdentity("compiled_identity is not an object")
    return identity


def is_genuine_compiled_identity(
    identity: Any,
    *,
    source_sha256: str | None = None,
    source: str | None = None,
    entry_point: str | None = None,
) -> bool:
    return not is_placeholder_compiled_identity(
        identity,
        source_sha256=source_sha256,
        source=source,
        entry_point=entry_point,
    )


# ---------------------------------------------------------------------------
# qwen3 dense GGUF gap. Named blocker; never a silent alias.
# ---------------------------------------------------------------------------


def qwen3_dense_gguf_blocker(
    architectures: Sequence[str] | None = None,
    *,
    family: Any = None,
    model_type: Any = None,
) -> dict[str, Any]:
    """qwen3 dense is not qwen2 and is not qwen3moe. Do not map it onto either."""
    arms = [str(a) for a in (architectures or [])]
    has_qwen3 = "qwen3" in arms
    has_qwen3moe = "qwen3moe" in arms
    has_qwen2 = "qwen2" in arms or "qwen2.5" in arms or "qwen" in arms
    mt = None if model_type is None else str(model_type)
    fam = None if family is None else str(family)
    specimen_is_dense_qwen3 = mt == "qwen3" or fam in {
        "dense_swiglu_transformer",
        "dense_transformer",
    }
    why = (
        f"{QWEN3_DENSE_GGUF_BLOCKER}: native GGUF match arms={arms!r}. "
        "qwen3 dense is absent (there is no 'qwen3' arm). qwen3moe is a different "
        "family (routed MoE) and was not used as a stand-in. qwen2 is a different "
        "architecture: QwenDense::load reads attn_q/k/v/output plus optional qkv "
        "bias and does not load attn_q_norm/attn_k_norm; Qwen3 dense applies "
        "QK-RMSNorm before RoPE, so aliasing 'qwen3' onto the qwen2 arm would "
        "skip a real operator. The arm is not added from this sidecar (the "
        "native loader is out of write scope); the gap is recorded rather than "
        "silently mapped."
    )
    return {
        "id": QWEN3_DENSE_GGUF_BLOCKER,
        "holds": (not has_qwen3) if specimen_is_dense_qwen3 else (not has_qwen3),
        "applies_to_this_specimen": bool(specimen_is_dense_qwen3),
        "includes_qwen3_dense": has_qwen3,
        "includes_qwen3moe": has_qwen3moe,
        "includes_qwen2": has_qwen2,
        "qwen3moe_is_different_family": True,
        "did_not_map_dense_onto_moe_arm": True,
        "did_not_map_dense_onto_qwen2_arm": True,
        "architectures": arms,
        "family": fam,
        "model_type": mt,
        "why": why,
        "evidence": [
            "crates/hawking-core/src/model/mod.rs match arch.as_str() lists "
            "qwen2|qwen2.5|qwen and qwen2moe|qwen3moe|qwen-moe; there is no qwen3",
            "crates/hawking-core/src/model/qwen_dense.rs loads attn_q.weight and "
            "optional attn_q.bias; it does not load attn_q_norm.weight / "
            "attn_k_norm.weight (Qwen3 QK-RMSNorm)",
            "llama.cpp / GGUF architecture string for dense Qwen3 is 'qwen3'; "
            "qwen3moe is LLM_ARCH_QWEN3MOE, a different family",
        ],
    }


# ---------------------------------------------------------------------------
# Shader emission. Each organ gets a self-contained Metal kernel whose
# operator is that organ's role, specialized to this body's extents.
# ---------------------------------------------------------------------------


def _shape_map(slot: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(slot, Mapping):
        return {}
    shape = slot.get("specimen_shape")
    return dict(shape) if isinstance(shape, Mapping) else {}


def emit_shader(
    organ: str,
    *,
    shape: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    specimen_id: str | None = None,
) -> dict[str, Any] | None:
    """Return {entry_point, source, why} or None if this organ cannot be lowered."""
    cfg = config if isinstance(config, Mapping) else {}
    shp = dict(shape) if isinstance(shape, Mapping) else {}
    sid = specimen_id or "unspecified-body"
    if organ not in LOWABLE_ORGANS:
        return None
    hidden = _int_pos(shp.get("cols"), cfg.get("hidden_size"))
    if organ == "mlp_down":
        rows = _int_pos(shp.get("rows"), cfg.get("hidden_size"))
        cols = _int_pos(shp.get("cols"), cfg.get("intermediate_size"), cfg.get("moe_intermediate_size"))
        if rows is None or cols is None:
            return None
        entry = "organ_mlp_down_gemv"
        source = _gemv_source(entry, organ, sid, rows, cols)
        return {
            "entry_point": entry,
            "source": source,
            "why": f"native f32 GEMV for {organ} at {rows}x{cols}",
            "extents": [rows, cols],
        }
    if organ == "lm_head":
        rows = _int_pos(shp.get("rows"), cfg.get("vocab_size"))
        cols = _int_pos(shp.get("cols"), cfg.get("hidden_size"))
        if rows is None or cols is None:
            return None
        entry = "organ_lm_head_gemv"
        source = _gemv_source(entry, organ, sid, rows, cols)
        return {
            "entry_point": entry,
            "source": source,
            "why": f"native f32 GEMV for {organ} at {rows}x{cols}",
            "extents": [rows, cols],
        }
    if organ == "mlp_gate_up":
        rows = _int_pos(shp.get("rows"), cfg.get("intermediate_size"), cfg.get("moe_intermediate_size"))
        cols = _int_pos(shp.get("cols"), cfg.get("hidden_size"))
        if rows is None or cols is None:
            return None
        entry = "organ_mlp_gate_up_swiglu"
        source = _swiglu_source(entry, organ, sid, rows, cols)
        return {
            "entry_point": entry,
            "source": source,
            "why": f"fused gate-up SwiGLU GEMV for {organ} at {rows}x{cols}",
            "extents": [rows, cols],
        }
    if organ == "rmsnorm":
        n = _int_pos(shp.get("cols"), hidden, cfg.get("hidden_size"))
        if n is None:
            return None
        entry = "organ_rmsnorm"
        source = _rmsnorm_source(entry, organ, sid, n)
        return {
            "entry_point": entry,
            "source": source,
            "why": f"RMSNorm over hidden={n}",
            "extents": [n],
        }
    if organ == "embed":
        rows = _int_pos(shp.get("rows"), cfg.get("vocab_size"))
        cols = _int_pos(shp.get("cols"), cfg.get("hidden_size"))
        if rows is None or cols is None:
            return None
        entry = "organ_embed_gather"
        source = _embed_source(entry, organ, sid, rows, cols)
        return {
            "entry_point": entry,
            "source": source,
            "why": f"embedding gather table {rows}x{cols}",
            "extents": [rows, cols],
        }
    if organ == "gqa_attention":
        head_dim = _int_pos(shp.get("head_dim"), cfg.get("head_dim"))
        n_q = _int_pos(cfg.get("num_attention_heads"))
        n_kv = _int_pos(cfg.get("num_key_value_heads"), n_q)
        if n_q is None and head_dim and _int_pos(shp.get("q_rows")):
            q_rows = _int_pos(shp.get("q_rows"))
            if q_rows and q_rows % head_dim == 0:
                n_q = q_rows // head_dim
        if n_kv is None and head_dim and _int_pos(shp.get("kv_rows")):
            kv_rows = _int_pos(shp.get("kv_rows"))
            if kv_rows and kv_rows % head_dim == 0:
                n_kv = kv_rows // head_dim
        if head_dim is None or n_q is None or n_kv is None:
            return None
        if n_q % n_kv != 0:
            return None
        entry = "organ_gqa_decode"
        source = _gqa_source(entry, organ, sid, n_q, n_kv, head_dim)
        return {
            "entry_point": entry,
            "source": source,
            "why": f"GQA decode n_q={n_q} n_kv={n_kv} head_dim={head_dim}",
            "extents": [n_q, n_kv, head_dim],
        }
    return None


def _header(entry: str, organ: str, specimen_id: str, baked: str) -> str:
    return (
        f"// DeviceCompiler organ={organ} specimen={specimen_id} entry={entry}\n"
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        f"{baked}\n"
    )


def _gemv_source(entry: str, organ: str, specimen_id: str, rows: int, cols: int) -> str:
    baked = f"#define BAKED_ROWS {rows}u\n#define BAKED_COLS {cols}u\n"
    return (
        _header(entry, organ, specimen_id, baked)
        + f"""
kernel void {entry}(
    device const float* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= rows) return;
    float acc = 0.0f;
    device const float* rowp = w + ((ulong)gid * (ulong)cols);
    for (uint k = 0; k < cols; k++) {{
        acc += rowp[k] * x[k];
    }}
    y[gid] = acc;
}}
"""
    )


def _swiglu_source(entry: str, organ: str, specimen_id: str, rows: int, cols: int) -> str:
    baked = f"#define BAKED_ROWS {rows}u\n#define BAKED_COLS {cols}u\n"
    return (
        _header(entry, organ, specimen_id, baked)
        + f"""
kernel void {entry}(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= rows) return;
    float g = 0.0f;
    float u = 0.0f;
    device const float* grow = gate + ((ulong)gid * (ulong)cols);
    device const float* urow = up + ((ulong)gid * (ulong)cols);
    for (uint k = 0; k < cols; k++) {{
        g += grow[k] * x[k];
        u += urow[k] * x[k];
    }}
    float silu = g / (1.0f + exp(-g));
    y[gid] = silu * u;
}}
"""
    )


def _rmsnorm_source(entry: str, organ: str, specimen_id: str, n: int) -> str:
    baked = f"#define BAKED_N {n}u\n"
    return (
        _header(entry, organ, specimen_id, baked)
        + f"""
kernel void {entry}(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid != 0u) return;
    float acc = 0.0f;
    for (uint i = 0; i < n; i++) {{
        acc += x[i] * x[i];
    }}
    float inv = rsqrt(acc / float(n) + eps);
    for (uint i = 0; i < n; i++) {{
        y[i] = x[i] * inv * w[i];
    }}
}}
"""
    )


def _embed_source(entry: str, organ: str, specimen_id: str, vocab: int, hidden: int) -> str:
    baked = f"#define BAKED_VOCAB {vocab}u\n#define BAKED_HIDDEN {hidden}u\n"
    return (
        _header(entry, organ, specimen_id, baked)
        + f"""
kernel void {entry}(
    device const float* table [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& n_tokens [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= n_tokens) return;
    uint tok = tokens[gid];
    device const float* src = table + ((ulong)tok * (ulong)hidden);
    device float* dst = out + ((ulong)gid * (ulong)hidden);
    for (uint i = 0; i < hidden; i++) {{
        dst[i] = src[i];
    }}
}}
"""
    )


def _gqa_source(
    entry: str, organ: str, specimen_id: str, n_q: int, n_kv: int, head_dim: int
) -> str:
    baked = (
        f"#define BAKED_N_Q {n_q}u\n"
        f"#define BAKED_N_KV {n_kv}u\n"
        f"#define BAKED_HEAD_DIM {head_dim}u\n"
    )
    return (
        _header(entry, organ, specimen_id, baked)
        + f"""
kernel void {entry}(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    uint h [[thread_position_in_grid]])
{{
    const uint n_q = BAKED_N_Q;
    const uint n_kv = BAKED_N_KV;
    const uint d = BAKED_HEAD_DIM;
    if (h >= n_q) return;
    uint kv = h / (n_q / n_kv);
    float scale = rsqrt(float(d));
    device const float* qh = q + ((ulong)h * (ulong)d);
    float maxv = -3.402823e+38f;
    for (uint t = 0; t < seq; t++) {{
        device const float* kt = k + ((((ulong)t * (ulong)n_kv) + (ulong)kv) * (ulong)d);
        float dot = 0.0f;
        for (uint i = 0; i < d; i++) {{
            dot += qh[i] * kt[i];
        }}
        dot *= scale;
        maxv = max(maxv, dot);
    }}
    float sum = 0.0f;
    for (uint t = 0; t < seq; t++) {{
        device const float* kt = k + ((((ulong)t * (ulong)n_kv) + (ulong)kv) * (ulong)d);
        float dot = 0.0f;
        for (uint i = 0; i < d; i++) {{
            dot += qh[i] * kt[i];
        }}
        sum += exp(dot * scale - maxv);
    }}
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    device float* oh = out + ((ulong)h * (ulong)d);
    for (uint i = 0; i < d; i++) {{
        float acc = 0.0f;
        for (uint t = 0; t < seq; t++) {{
            device const float* kt = k + ((((ulong)t * (ulong)n_kv) + (ulong)kv) * (ulong)d);
            device const float* vt = v + ((((ulong)t * (ulong)n_kv) + (ulong)kv) * (ulong)d);
            float dot = 0.0f;
            for (uint j = 0; j < d; j++) {{
                dot += qh[j] * kt[j];
            }}
            float wgt = exp(dot * scale - maxv) * inv;
            acc += wgt * vt[i];
        }}
        oh[i] = acc;
    }}
}}
"""
    )


def cannot_lower_why(
    organ: str,
    *,
    shape: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> str:
    if organ not in LOWABLE_ORGANS:
        return (
            f"organ={organ!r} has no honest lowering in this DeviceCompiler; "
            f"it stays {NATIVE_UNMEASURED}. A GEMV labelled as {organ} would be "
            "a placeholder for the wrong operator."
        )
    emitted = emit_shader(organ, shape=shape, config=config)
    if emitted is None:
        return (
            f"organ={organ!r} is lowable in principle but this body's shape/config "
            f"does not supply the extents the kernel needs; stays {NATIVE_UNMEASURED}"
        )
    return emitted["why"]


# ---------------------------------------------------------------------------
# Metal backend. Compile service only: library + function + pipeline + archive.
# No command queue, no dispatch, no GPU lease.
# ---------------------------------------------------------------------------


class CompileJob:
    def __init__(
        self,
        organ: str,
        source: str,
        entry_point: str,
        archive_path: str,
        why: str,
        extents: list[int] | None = None,
    ) -> None:
        self.organ = organ
        self.source = source
        self.entry_point = entry_point
        self.archive_path = archive_path
        self.why = why
        self.extents = list(extents or [])
        self.source_sha256 = _sha256_text(source)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.organ,
            "source": self.source,
            "entry_point": self.entry_point,
            "archive_path": self.archive_path,
        }


class MetalBackend(Protocol):
    def compile_jobs(self, jobs: Sequence[CompileJob]) -> dict[str, Any]:
        ...


HELPER_SWIFT = r'''
import Foundation
import Metal
import CommonCrypto

func sha256Hex(_ data: Data) -> String {
    var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
    data.withUnsafeBytes { buf in
        _ = CC_SHA256(buf.baseAddress, CC_LONG(data.count), &digest)
    }
    return digest.map { String(format: "%02x", $0) }.joined()
}

let inputData = FileHandle.standardInput.readDataToEndOfFile()
guard let root = try JSONSerialization.jsonObject(with: inputData) as? [String: Any] else {
    let err: [String: Any] = ["ok": false, "error": "stdin is not a JSON object", "results": []]
    let out = try! JSONSerialization.data(withJSONObject: err)
    FileHandle.standardOutput.write(out)
    exit(0)
}
let jobs = (root["jobs"] as? [[String: Any]]) ?? []

var payload: [String: Any] = [:]
payload["ok"] = false
payload["n_devices"] = 0
payload["created_command_queue"] = false
payload["dispatched"] = false
payload["results"] = []

let all = MTLCopyAllDevices()
payload["n_devices"] = all.count
payload["devices"] = all.map { $0.name }

guard let device = MTLCreateSystemDefaultDevice() else {
    payload["error"] = "MTLCreateSystemDefaultDevice returned nil; MTLCopyAllDevices n=\(all.count). This process cannot see a Metal device, so it cannot mint an MTLComputePipelineState. That is a process property, not a claim that the host has no GPU."
    payload["system_default"] = NSNull()
    let out = try! JSONSerialization.data(withJSONObject: payload)
    FileHandle.standardOutput.write(out)
    exit(0)
}
payload["system_default"] = device.name
payload["device"] = device.name

var results: [[String: Any]] = []
for job in jobs {
    let organ = (job["id"] as? String) ?? ""
    let source = (job["source"] as? String) ?? ""
    let entry = (job["entry_point"] as? String) ?? ""
    let archivePath = (job["archive_path"] as? String) ?? ""
    var row: [String: Any] = [
        "id": organ,
        "ok": false,
        "entry_point": entry,
        "function_found": false,
        "pipeline_created": false,
        "pipeline_object": NSNull(),
        "source_sha256": sha256Hex(Data(source.utf8)),
        "created_command_queue": false,
        "dispatched": false,
    ]
    if source.isEmpty || entry.isEmpty {
        row["error"] = "job missing source or entry_point"
        results.append(row)
        continue
    }
    do {
        let lib = try device.makeLibrary(source: source, options: nil)
        guard let fn = lib.makeFunction(name: entry) else {
            row["error"] = "function \(entry) not found in compiled MTLLibrary"
            results.append(row)
            continue
        }
        row["function_found"] = true
        let pso = try device.makeComputePipelineState(function: fn)
        row["pipeline_created"] = true
        row["pipeline_object"] = "MTLComputePipelineState"
        row["thread_execution_width"] = pso.threadExecutionWidth
        row["max_total_threads_per_threadgroup"] = pso.maxTotalThreadsPerThreadgroup
        let adesc = MTLBinaryArchiveDescriptor()
        let archive = try device.makeBinaryArchive(descriptor: adesc)
        let pdesc = MTLComputePipelineDescriptor()
        pdesc.computeFunction = fn
        pdesc.label = entry
        try archive.addComputePipelineFunctions(descriptor: pdesc)
        let url = URL(fileURLWithPath: archivePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        try archive.serialize(to: url)
        let bytes = try Data(contentsOf: url)
        row["archive_bytes"] = bytes.count
        row["archive_sha256"] = sha256Hex(bytes)
        row["archive_path"] = archivePath
        row["ok"] = true
    } catch {
        row["error"] = String(describing: error)
    }
    results.append(row)
}
payload["results"] = results
payload["ok"] = results.contains { ($0["ok"] as? Bool) == true }
payload["error"] = NSNull()
if payload["ok"] as? Bool != true && (payload["error"] is NSNull) {
    let first = results.compactMap { $0["error"] as? String }.first
    payload["error"] = first ?? "no job produced an MTLComputePipelineState"
}
let out = try! JSONSerialization.data(withJSONObject: payload)
FileHandle.standardOutput.write(out)
'''


def _helper_binary() -> Path:
    cache = Path(tempfile.gettempdir()) / "hawking-device-compiler-helper"
    digest = _sha256_text(HELPER_SWIFT)[:16]
    binary = cache / f"metal_lower_{digest}"
    return binary


def ensure_metal_helper(*, swiftc: str | None = None) -> tuple[Path | None, str | None]:
    """Build the Metal lower helper once per helper source digest."""
    compiler = swiftc or shutil.which("swiftc")
    if not compiler:
        return None, "swiftc is not on PATH; cannot compile the Metal lower helper"
    binary = _helper_binary()
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary, None
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hawking-dc-swift-") as tmp:
        src = Path(tmp) / "metal_lower.swift"
        src.write_text(HELPER_SWIFT)
        staged = Path(tmp) / "metal_lower"
        proc = subprocess.run(
            [
                compiler,
                "-O",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                str(src),
                "-o",
                str(staged),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0 or not staged.is_file():
            err = (proc.stderr or proc.stdout or "").strip()[:800]
            return None, f"swiftc failed to build Metal lower helper: {err}"
        shutil.copy2(staged, binary)
        binary.chmod(0o755)
    return binary, None


class LiveMetalBackend:
    """Drive MTLDevice.makeLibrary + makeComputePipelineState + MTLBinaryArchive.

    No command queue. Nothing is dispatched. Creating a pipeline is compile.
    """

    def compile_jobs(self, jobs: Sequence[CompileJob]) -> dict[str, Any]:
        if not jobs:
            return {
                "ok": True,
                "n_devices": None,
                "results": [],
                "error": None,
                "created_command_queue": False,
                "dispatched": False,
                "backend": "live_metal",
            }
        binary, err = ensure_metal_helper()
        if binary is None:
            return {
                "ok": False,
                "n_devices": 0,
                "results": [],
                "error": err,
                "created_command_queue": False,
                "dispatched": False,
                "backend": "live_metal",
            }
        payload = {"jobs": [j.to_payload() for j in jobs]}
        proc = subprocess.run(
            [str(binary)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=180,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return {
                "ok": False,
                "n_devices": 0,
                "results": [],
                "error": (
                    f"Metal lower helper wrote no JSON (rc={proc.returncode}): "
                    f"{(proc.stderr or '')[-400:]}"
                ),
                "created_command_queue": False,
                "dispatched": False,
                "backend": "live_metal",
            }
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "n_devices": 0,
                "results": [],
                "error": f"Metal lower helper JSON decode failed: {exc}",
                "created_command_queue": False,
                "dispatched": False,
                "backend": "live_metal",
            }
        if not isinstance(doc, dict):
            return {
                "ok": False,
                "n_devices": 0,
                "results": [],
                "error": "Metal lower helper returned a non-object",
                "created_command_queue": False,
                "dispatched": False,
                "backend": "live_metal",
            }
        doc.setdefault("backend", "live_metal")
        doc.setdefault("created_command_queue", False)
        doc.setdefault("dispatched", False)
        return doc


class UnavailableMetalBackend:
    """Test double: Metal device is not visible to this process."""

    def __init__(self, why: str = "Metal device unavailable (test double)") -> None:
        self.why = why

    def compile_jobs(self, jobs: Sequence[CompileJob]) -> dict[str, Any]:
        return {
            "ok": False,
            "n_devices": 0,
            "results": [],
            "error": self.why,
            "created_command_queue": False,
            "dispatched": False,
            "backend": "unavailable",
        }


def identity_from_compile_row(
    row: Mapping[str, Any],
    *,
    source: str,
    entry_point: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Build a compiled_identity from a Metal helper row, then refuse placeholders."""
    path_raw = row.get("archive_path")
    if not (isinstance(path_raw, str) and path_raw):
        raise PlaceholderCompiledIdentity("compiled archive_path missing")
    archive = Path(path_raw)
    if not archive.is_file():
        raise PlaceholderCompiledIdentity(f"compiled archive is not a file: {path_raw}")
    raw = archive.read_bytes()
    if not raw:
        raise PlaceholderCompiledIdentity("compiled archive file is empty")
    if raw == source.encode("utf-8") or raw.lstrip().startswith(b"#include") or raw.lstrip().startswith(b"//"):
        raise PlaceholderCompiledIdentity(
            "archive bytes are shader source, not an MTLBinaryArchive"
        )
    digest = _sha256_bytes(raw)
    claimed = row.get("archive_sha256")
    if claimed != digest:
        raise PlaceholderCompiledIdentity(
            "archive_sha256 does not match the bytes on archive_path"
        )
    if digest == source_sha256:
        raise PlaceholderCompiledIdentity(
            "archive digest equals source digest; a source hash is not compiled"
        )
    shader_hash = digest
    pipeline = {
        "object": row.get("pipeline_object") or PIPELINE_OBJECT,
        "created": bool(row.get("pipeline_created") is True),
        "function_found": bool(row.get("function_found") is True),
        "thread_execution_width": row.get("thread_execution_width"),
        "max_total_threads_per_threadgroup": row.get(
            "max_total_threads_per_threadgroup"
        ),
        "created_command_queue": False,
        "dispatched": False,
    }
    identity = {
        "kind": COMPILED_IDENTITY_KIND,
        "unit": "mtl_binary_archive_sha256",
        "shader_hash": shader_hash,
        "value": shader_hash,
        "entry_point": entry_point,
        "pipeline": pipeline,
        "archive_bytes": len(raw),
        "archive_path": path_raw,
        "source_sha256": source_sha256,
        "device": row.get("device"),
        "claim_boundary": (
            "MTLComputePipelineState created from a source-compiled MTLLibrary. "
            "shader_hash is sha256 of the serialized MTLBinaryArchive. "
            "No command queue, no dispatch, not a hardware measurement."
        ),
    }
    refuse_placeholder(
        identity, source_sha256=source_sha256, source=source, entry_point=entry_point
    )
    return identity


def _unmeasured_slot(
    organ: str,
    *,
    why: str,
    shape: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "organ": organ,
        "status": NATIVE_UNMEASURED,
        "occupying": {
            "kind": NATIVE_UNMEASURED,
            "compiled_kernel": None,
            "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
        },
        "compiled_identity": None,
        "specimen_shape": None if shape is None else dict(shape),
        "name_is_not_a_compiled_kernel": True,
        "why": why,
    }
    if extra:
        row.update(dict(extra))
    return row


def _compiled_slot(
    organ: str,
    *,
    identity: Mapping[str, Any],
    why: str,
    shape: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    refuse_placeholder(
        identity,
        source_sha256=identity.get("source_sha256") if isinstance(identity, Mapping) else None,
        source=extra.get("source") if extra else None,
        entry_point=identity.get("entry_point") if isinstance(identity, Mapping) else None,
    )
    row: dict[str, Any] = {
        "organ": organ,
        "status": COMPILED,
        "occupying": {
            "kind": COMPILED,
            "compiled_kernel": identity.get("entry_point"),
            "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
        },
        "compiled_identity": dict(identity),
        "specimen_shape": None if shape is None else dict(shape),
        "name_is_not_a_compiled_kernel": False,
        "why": why,
    }
    if extra:
        # source is kept off the slot to avoid dumping shaders into every receipt
        # twice; identity already carries source_sha256.
        extras = {k: v for k, v in dict(extra).items() if k != "source"}
        row.update(extras)
    return row


def _plan_document(kernel_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(kernel_plan, Mapping):
        return {}
    if isinstance(kernel_plan.get("plan"), list):
        return dict(kernel_plan)
    ev = kernel_plan.get("evidence")
    if isinstance(ev, Mapping) and isinstance(ev.get("plan"), list):
        return dict(ev)
    return dict(kernel_plan)


def nx_fragment_from_slots(
    slots: Sequence[Mapping[str, Any]],
    *,
    specimen_id: str | None = None,
    family: Any = None,
    qwen3_blocker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """NX-in-progress. Compiled organs carry identity; planned ones do not.

    This is not a packed NX. source_independent stays False. A later packer
    can tell compiled from planned by status + compiled_identity.
    """
    kernels = []
    n_compiled = 0
    for slot in slots:
        organ = slot.get("organ")
        status = slot.get("status")
        identity = slot.get("compiled_identity")
        genuine = status == COMPILED and is_genuine_compiled_identity(
            identity,
            source_sha256=None if not isinstance(identity, Mapping) else identity.get("source_sha256"),
            entry_point=None if not isinstance(identity, Mapping) else identity.get("entry_point"),
        )
        if status == COMPILED and not genuine:
            # Defence in depth: never put a placeholder on the NX.
            status = NATIVE_UNMEASURED
            identity = None
        if genuine:
            n_compiled += 1
        kernels.append(
            {
                "organ": organ,
                "status": status,
                "occupying": slot.get("occupying"),
                "compiled_identity": None if not genuine else identity,
                "entry_point": None
                if not genuine or not isinstance(identity, Mapping)
                else identity.get("entry_point"),
                "shader_hash": None
                if not genuine or not isinstance(identity, Mapping)
                else identity.get("shader_hash"),
                "why": slot.get("why"),
            }
        )
    compiled = [k for k in kernels if k.get("status") == COMPILED]
    planned = [k for k in kernels if k.get("status") != COMPILED]
    return {
        "status": "COMPILED_KERNELS_NOT_PACKED",
        "source_independent": False,
        "serialized_artifact": None,
        "specimen_id": specimen_id,
        "family": family,
        "physical_program": {
            "kernels": kernels,
            "n_compiled": n_compiled,
            "n_planned": len(planned),
        },
        "native_kernel": {
            "status": "BOUND" if n_compiled else "ABSENT",
            "n_compiled": n_compiled,
            "organs": [k.get("organ") for k in compiled],
        },
        "compiled_organs": [k.get("organ") for k in compiled],
        "planned_organs": [k.get("organ") for k in planned],
        "qwen3_dense_gguf_blocker": None if qwen3_blocker is None else dict(qwen3_blocker),
        "claim_boundary": (
            "compiled Metal pipelines for organs that lowered; not a packed NX; "
            "not source-independent; not a hardware measurement"
        ),
    }


def split_compiled_and_planned(
    nx: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Later-stage helper: tell a compiled kernel from a planned one on the NX."""
    if not isinstance(nx, Mapping):
        return [], []
    kernels = _dot(nx, "physical_program.kernels")
    if not isinstance(kernels, list):
        kernels = nx.get("physical_program", {}).get("kernels") if isinstance(nx.get("physical_program"), Mapping) else []
    compiled: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    for k in kernels or []:
        if not isinstance(k, Mapping):
            continue
        identity = k.get("compiled_identity")
        if k.get("status") == COMPILED and is_genuine_compiled_identity(
            identity,
            source_sha256=None if not isinstance(identity, Mapping) else identity.get("source_sha256"),
            entry_point=None if not isinstance(identity, Mapping) else identity.get("entry_point"),
        ):
            compiled.append(dict(k))
        else:
            planned.append(dict(k))
    return compiled, planned


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur: Any = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def lower_plan(
    kernel_plan: Mapping[str, Any] | None,
    *,
    specimen_id: str | None = None,
    family: Any = None,
    config: Mapping[str, Any] | None = None,
    native_architectures: Sequence[str] | None = None,
    model_type: Any = None,
    backend: MetalBackend | None = None,
) -> dict[str, Any]:
    """Lower a KernelPlanner plan. Never records a placeholder as COMPILED."""
    doc = _plan_document(kernel_plan)
    slots_in = [s for s in (doc.get("plan") or []) if isinstance(s, Mapping) and s.get("organ")]
    blocker = qwen3_dense_gguf_blocker(
        native_architectures, family=family, model_type=model_type
    )
    if not slots_in:
        nx = nx_fragment_from_slots([], specimen_id=specimen_id, family=family, qwen3_blocker=blocker)
        return {
            "ok": False,
            "error": "no_plan",
            "why": "no KernelPlanner plan was handed; a compiled identity is not invented",
            "route": doc.get("route"),
            "n_organs": 0,
            "n_compiled": 0,
            "n_native_unmeasured": 0,
            "n_placeholder_refused": 0,
            "plan": [],
            "nx_fragment": nx,
            "qwen3_dense_gguf_blocker": blocker,
            "metal": None,
            "created_command_queue": False,
            "dispatched": False,
        }

    metal = backend if backend is not None else LiveMetalBackend()
    jobs: list[CompileJob] = []
    pre_slots: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hawking-dc-archives-") as tmp:
        tmp_path = Path(tmp)
        for incoming in slots_in:
            organ = str(incoming["organ"])
            shape = _shape_map(incoming)
            emitted = emit_shader(
                organ, shape=shape, config=config, specimen_id=specimen_id
            )
            if emitted is None:
                pre_slots.append(
                    _unmeasured_slot(
                        organ,
                        why=cannot_lower_why(organ, shape=shape, config=config),
                        shape=shape,
                    )
                )
                continue
            archive = tmp_path / f"{organ}.mtlarchive"
            jobs.append(
                CompileJob(
                    organ=organ,
                    source=str(emitted["source"]),
                    entry_point=str(emitted["entry_point"]),
                    archive_path=str(archive),
                    why=str(emitted["why"]),
                    extents=list(emitted.get("extents") or []),
                )
            )
            pre_slots.append(
                {
                    "organ": organ,
                    "shape": shape,
                    "job": True,
                    "why_emitted": emitted["why"],
                }
            )

        batch = metal.compile_jobs(jobs) if jobs else {
            "ok": True,
            "results": [],
            "error": None,
            "n_devices": None,
            "backend": getattr(metal, "__class__", type(metal)).__name__,
            "created_command_queue": False,
            "dispatched": False,
        }
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in batch.get("results") or []:
            if isinstance(row, Mapping) and row.get("id"):
                by_id[str(row["id"])] = row

        out_slots: list[dict[str, Any]] = []
        n_placeholder_refused = 0
        job_iter = iter(jobs)
        for prepared in pre_slots:
            organ = str(prepared["organ"])
            if not prepared.get("job"):
                out_slots.append(prepared)
                continue
            job = next(job_iter)
            row = by_id.get(organ)
            shape = prepared.get("shape") if isinstance(prepared.get("shape"), Mapping) else None
            if not isinstance(row, Mapping) or row.get("ok") is not True:
                why = (
                    f"{NATIVE_UNMEASURED}: Metal compile did not produce a pipeline "
                    f"for {organ}. "
                    + (
                        str(row.get("error"))
                        if isinstance(row, Mapping) and row.get("error")
                        else str(batch.get("error") or "no compile row")
                    )
                )
                out_slots.append(_unmeasured_slot(organ, why=why, shape=shape))
                continue
            try:
                identity = identity_from_compile_row(
                    row,
                    source=job.source,
                    entry_point=job.entry_point,
                    source_sha256=job.source_sha256,
                )
                # Re-check after construction. A lying backend that set ok=True
                # with a source digest as shader_hash dies here.
                refuse_placeholder(
                    identity,
                    source_sha256=job.source_sha256,
                    source=job.source,
                    entry_point=job.entry_point,
                )
            except PlaceholderCompiledIdentity as exc:
                n_placeholder_refused += 1
                out_slots.append(
                    _unmeasured_slot(
                        organ,
                        why=f"{PLACEHOLDER_REFUSED}: {exc}. organ stays {NATIVE_UNMEASURED}",
                        shape=shape,
                        extra={"placeholder_refused": True},
                    )
                )
                continue
            out_slots.append(
                _compiled_slot(
                    organ,
                    identity=identity,
                    why=(
                        f"{COMPILED}: MTLComputePipelineState created; "
                        f"entry_point={identity.get('entry_point')}; "
                        f"shader_hash={identity.get('shader_hash')}. {job.why}"
                    ),
                    shape=shape,
                    extra={
                        "entry_point": identity.get("entry_point"),
                        "shader_hash": identity.get("shader_hash"),
                        "extents": job.extents,
                    },
                )
            )

    # Defence: if any COMPILED slot is a placeholder, demote it.
    demoted: list[dict[str, Any]] = []
    for slot in out_slots:
        if slot.get("status") != COMPILED:
            demoted.append(slot)
            continue
        identity = slot.get("compiled_identity")
        if is_genuine_compiled_identity(
            identity,
            source_sha256=None if not isinstance(identity, Mapping) else identity.get("source_sha256"),
            entry_point=None if not isinstance(identity, Mapping) else identity.get("entry_point"),
        ):
            demoted.append(slot)
            continue
        n_placeholder_refused += 1
        demoted.append(
            _unmeasured_slot(
                str(slot.get("organ")),
                why=f"{PLACEHOLDER_REFUSED} on a COMPILED slot; demoted to {NATIVE_UNMEASURED}",
                shape=slot.get("specimen_shape") if isinstance(slot.get("specimen_shape"), Mapping) else None,
                extra={"placeholder_refused": True},
            )
        )

    n_compiled = sum(1 for s in demoted if s.get("status") == COMPILED)
    n_unmeasured = sum(1 for s in demoted if s.get("status") == NATIVE_UNMEASURED)
    nx = nx_fragment_from_slots(
        demoted, specimen_id=specimen_id, family=family, qwen3_blocker=blocker
    )
    metal_error = batch.get("error")
    why = (
        f"lowered {n_compiled}/{len(demoted)} organ(s) to {PIPELINE_OBJECT}; "
        f"{n_unmeasured} remain {NATIVE_UNMEASURED}; "
        f"placeholder_refused={n_placeholder_refused}. "
        f"{blocker['id']} holds={blocker['holds']}"
    )
    if n_compiled == 0 and metal_error:
        why = f"{why}; metal={metal_error}"
    return {
        "ok": n_compiled > 0,
        "error": None if n_compiled > 0 else (metal_error or "zero_organs_compiled"),
        "why": why,
        "route": doc.get("route"),
        "specimen_id": specimen_id,
        "family": family,
        "n_organs": len(demoted),
        "n_compiled": n_compiled,
        "n_native_unmeasured": n_unmeasured,
        "n_placeholder_refused": n_placeholder_refused,
        "plan": demoted,
        "nx_fragment": nx,
        "qwen3_dense_gguf_blocker": blocker,
        "metal": {
            "backend": batch.get("backend"),
            "ok": batch.get("ok"),
            "n_devices": batch.get("n_devices"),
            "device": batch.get("device") or batch.get("system_default"),
            "error": batch.get("error"),
            "created_command_queue": bool(batch.get("created_command_queue")),
            "dispatched": bool(batch.get("dispatched")),
        },
        "created_command_queue": False,
        "dispatched": False,
        "claim_boundary": (
            "COMPILE_TIME_SCIENCE_ONLY DeviceCompiler lowering. "
            "NATIVE_UNMEASURED is not a compiled kernel, not a hardware "
            "measurement, not physical EBPW, not a packed NX."
        ),
    }


def compile_source(
    source: str,
    entry_point: str,
    *,
    backend: MetalBackend | None = None,
    organ: str = "probe",
) -> dict[str, Any]:
    """Compile one shader. Used by tests. Refuses a placeholder identity."""
    metal = backend if backend is not None else LiveMetalBackend()
    with tempfile.TemporaryDirectory(prefix="hawking-dc-one-") as tmp:
        job = CompileJob(
            organ=organ,
            source=source,
            entry_point=entry_point,
            archive_path=str(Path(tmp) / f"{organ}.mtlarchive"),
            why="direct compile_source",
        )
        batch = metal.compile_jobs([job])
        rows = [r for r in (batch.get("results") or []) if isinstance(r, Mapping)]
        row = rows[0] if rows else {}
        out: dict[str, Any] = {
            "ok": False,
            "error": batch.get("error") or row.get("error"),
            "source_sha256": job.source_sha256,
            "compiled_identity": None,
            "metal": batch,
        }
        if row.get("ok") is True:
            try:
                identity = identity_from_compile_row(
                    row,
                    source=source,
                    entry_point=entry_point,
                    source_sha256=job.source_sha256,
                )
                out["ok"] = True
                out["error"] = None
                out["compiled_identity"] = identity
            except PlaceholderCompiledIdentity as exc:
                out["error"] = f"{PLACEHOLDER_REFUSED}: {exc}"
        return out


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _synthetic_plan() -> dict[str, Any]:
    """A PLAN-THEN-COMPILE plan with Qwen3-0.6B-like extents. No specimen load."""
    cfg = {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "model_type": "qwen3",
    }
    organs = [
        ("embed", {"rows": 151936, "cols": 1024, "extents": [151936, 1024]}),
        (
            "gqa_attention",
            {
                "q_rows": 2048,
                "kv_rows": 1024,
                "cols": 1024,
                "head_dim": 128,
                "extents": [2048, 1024, 1024],
            },
        ),
        ("lm_head", {"rows": 151936, "cols": 1024, "extents": [151936, 1024]}),
        ("mlp_down", {"rows": 1024, "cols": 3072, "extents": [1024, 3072]}),
        ("mlp_gate_up", {"rows": 3072, "cols": 1024, "extents": [3072, 1024]}),
        ("rmsnorm", {"cols": 1024, "extents": [1024]}),
    ]
    plan = [
        {
            "organ": name,
            "status": NATIVE_UNMEASURED,
            "occupying": {
                "kind": NATIVE_UNMEASURED,
                "compiled_kernel": None,
                "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
            },
            "specimen_shape": shape,
            "n_role_matched": 0,
            "n_compiled_for_this_body": 0,
            "name_is_not_a_compiled_kernel": True,
            "why": NAME_IS_NOT_A_COMPILED_KERNEL,
        }
        for name, shape in organs
    ]
    return {
        "route": "PLAN-THEN-COMPILE",
        "plan": plan,
        "n_compiled": 0,
        "n_native_unmeasured": len(plan),
        "config": cfg,
    }


def assemble() -> dict[str, Any]:
    dummy_plan = _synthetic_plan()
    native_arms = [
        "llama",
        "llama",
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "mistral",
        "deepseek2",
        "deepseek-v2",
        "deepseek2-lite",
        "qwen2",
        "qwen2.5",
        "qwen",
        "qwen2moe",
        "qwen3moe",
        "qwen-moe",
        "rwkv7",
        "rwkv-7",
    ]
    # Negative control: a placeholder that claims compiled identity.
    placeholder = {
        "kind": "PLACEHOLDER",
        "value": "deadbeef" * 4,
        "shader_hash": "deadbeef" * 4,
        "entry_point": "organ_mlp_down_gemv",
        "pipeline": {
            "object": PIPELINE_OBJECT,
            "created": True,
            "function_found": True,
        },
        "archive_bytes": 16,
        "source_sha256": "deadbeef" * 4,
    }
    placeholder_caught = is_placeholder_compiled_identity(
        placeholder, source_sha256="deadbeef" * 4
    )
    placeholder_raise = None
    try:
        refuse_placeholder(placeholder, source_sha256="deadbeef" * 4)
    except PlaceholderCompiledIdentity as exc:
        placeholder_raise = str(exc)

    source_hash_claim = {
        "kind": COMPILED_IDENTITY_KIND,
        "shader_hash": _sha256_text("kernel void organ_mlp_down_gemv() {}"),
        "value": _sha256_text("kernel void organ_mlp_down_gemv() {}"),
        "entry_point": "organ_mlp_down_gemv",
        "pipeline": {
            "object": PIPELINE_OBJECT,
            "created": True,
            "function_found": True,
        },
        "archive_bytes": 4,
        "source_sha256": _sha256_text("kernel void organ_mlp_down_gemv() {}"),
    }
    source_hash_caught = is_placeholder_compiled_identity(
        source_hash_claim,
        source_sha256=source_hash_claim["source_sha256"],
        source="kernel void organ_mlp_down_gemv() {}",
        entry_point="organ_mlp_down_gemv",
    )

    lowering = lower_plan(
        dummy_plan,
        specimen_id="Qwen--Qwen3-0.6B@c1899de289a0",
        family="dense_swiglu_transformer",
        config=dummy_plan["config"],
        native_architectures=native_arms,
        model_type="qwen3",
    )
    compiled, planned = split_compiled_and_planned(lowering["nx_fragment"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Lower a PLAN-THEN-COMPILE KernelPlanner plan into Metal compute "
            "pipelines with a genuine compiled identity (MTLComputePipelineState, "
            "MTLBinaryArchive sha256, entry point), refuse placeholders, and "
            "carry those identities on an NX fragment"
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "contract": {
            "in": "KernelPlanner plan (route + plan[] organs with occupying kind and specimen_shape)",
            "out": (
                "per-organ COMPILED with compiled_identity "
                f"(kind={COMPILED_IDENTITY_KIND}, shader_hash, entry_point, "
                f"{PIPELINE_OBJECT}) or {NATIVE_UNMEASURED} with why; "
                "nx_fragment.physical_program.kernels carries the same"
            ),
            "identity": (
                "shader_hash is sha256(MTLBinaryArchive bytes), not sha256(source); "
                "entry_point was looked up in the compiled MTLLibrary; "
                "pipeline.created means makeComputePipelineState succeeded"
            ),
            "placeholder_policy": (
                "a placeholder claiming compiled identity is refused; the organ "
                f"stays {NATIVE_UNMEASURED} and is never recorded as {COMPILED}"
            ),
        },
        "entry_point": "tools.future.device_compiler.lower_plan",
        "helper": "tools.future.device_compiler.HELPER_SWIFT (swiftc, Metal.framework)",
        "created_command_queue": False,
        "dispatched": False,
        "placeholder_negative_control": {
            "caught": placeholder_caught,
            "raise": placeholder_raise,
            "source_digest_caught_as_placeholder": source_hash_caught,
        },
        "lowering": {
            "ok": lowering.get("ok"),
            "why": lowering.get("why"),
            "error": lowering.get("error"),
            "n_organs": lowering.get("n_organs"),
            "n_compiled": lowering.get("n_compiled"),
            "n_native_unmeasured": lowering.get("n_native_unmeasured"),
            "n_placeholder_refused": lowering.get("n_placeholder_refused"),
            "plan": lowering.get("plan"),
            "metal": lowering.get("metal"),
        },
        "nx_fragment": lowering.get("nx_fragment"),
        "nx_compiled_organs": [k.get("organ") for k in compiled],
        "nx_planned_organs": [k.get("organ") for k in planned],
        "qwen3_dense_gguf_blocker": lowering.get("qwen3_dense_gguf_blocker"),
        "lowable_organs": sorted(LOWABLE_ORGANS),
        "gaps_closed": [
            "a DeviceCompiler callable exists at tools.future.device_compiler.lower_plan",
            "PLAN-THEN-COMPILE organs are lowered to MTLComputePipelineState when a Metal device is visible to this process",
            "compiled identity is shader_hash (MTLBinaryArchive) + entry_point, not a source digest or a role name",
            "a placeholder claiming compiled identity is refused and stays NATIVE_UNMEASURED",
            "the NX fragment carries compiled vs planned so a later stage can tell them apart",
            "qwen3 dense GGUF match-arm absence is a named blocker; dense is not mapped onto qwen3moe or qwen2",
        ],
        "negative_findings": [
            (
                "if this process cannot see a Metal device, every organ stays "
                f"{NATIVE_UNMEASURED} and none are recorded as {COMPILED}"
            ),
            "KERNEL_LIBRARY compiled_identity remains ABSENT; this compiler does not back-fill that library",
            f"{QWEN3_DENSE_GGUF_BLOCKER}: native loader has no qwen3 arm; qwen3moe is a different family",
            "this module does not pack a source-independent NX; NoeticExecutable is a later stage",
            "no physical EBPW, no GPU lease, no dispatch",
        ],
        "what_this_cannot_establish": [
            "protected complete-token performance or physical EBPW",
            "that a packed NX exists for this specimen",
            "that adding a qwen3 GGUF match arm to hawking-core would load this safetensors body",
            "why a given process cannot see a present GPU; sandbox, launch context and build target are not distinguished here",
        ],
        "resident_callable": {
            "entry_point": "tools.future.device_compiler.lower_plan",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_EXECUTION.complete-token",
            "fails_closed": (
                "placeholder compiled identity raises and is not recorded as COMPILED; "
                "an organ that cannot lower stays NATIVE_UNMEASURED; "
                "no command queue; no hardware field"
            ),
        },
    }


def build() -> Path:
    doc = assemble()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
