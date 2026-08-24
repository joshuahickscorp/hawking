#!/usr/bin/env python3
"""Noetic Representation IR: a node that executes, not a schema that describes.

NR (`hawking.nos.noetic_representation`) is what the patient IS and must be
portable. NX (`hawking.nos.noetic_executable_genome`) is how ONE machine runs
it and must not be portable. Both exist. Neither executes. gravity_ir names a
kernel string on every node and accounts bytes; naming is not executing. NVM
binds route kinds to NX kernel names and then shells out to the Rust runtime.
The production Metal path executes grouped-absmax q4 fused, but the graph it
runs is not an IR document.

This module is the missing executing IR. A semantic node says what function a
packed payload is (portable). A machine lowering says how this box would run
it (not portable). Interpreting the semantic node produces y; that y is
compared to the same function evaluated directly. An IR with no executing
node is a schema, not an IR.

Does not load a 27B. Micro-sites are synthetic. Machine fields are refused on
the semantic side (G103 teeth). Stored bytes are computed from payloads and
the shared pool, so a 1 GB planted SharedBasis moves BPW (the G103 hole).

    python3 tools/headless/noetic_ir.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts/headless/NOETIC_IR.json"
SCHEMA = "hawking.headless.noetic_ir.v1"
IR_KIND = "hawking.nos.noetic_representation_ir"

# Same denominator gravity_ir / G103 / the sealed uniform-q4-v1 packer use.
SOURCE_PARAM_COUNT = 26_895_998_464
PLANTED_SHARED_BASIS_BYTES = 1_000_000_000
SEALED_BPW = 4.252735126866492

# Copied from tools/nr_container.py: presence anywhere in SEMANTIC is rejection.
MACHINE_SPECIFIC = {
    "kernel", "kernels", "kernel_name", "shader", "metallib", "threadgroup",
    "threadgroup_size", "tg_size", "grid", "dispatch", "dispatches", "device",
    "device_id", "machine_genome", "gpu", "residency_plan", "cache_plan",
    "schedule", "ps_per_element", "gb_s", "tps", "token_ns", "occupancy",
    "register_pressure", "simd_width", "r_tiling", "k_amortization",
    "implementation",
}

# ---------------------------------------------------------------------------
# What already approximates an IR. Cited, not re-derived.
# ---------------------------------------------------------------------------

ALREADY_APPROXIMATES = [
    {
        "what": "hawking.nos.noetic_representation (NR / G103)",
        "path": "tools/nr_container.py",
        "receipt": "receipts/ascent-2026-08-16/G103_NR_uniform-q4-v1.json",
        "does": (
            "Portable schema of the patient: codec families, kernel REQUIREMENTS "
            "(grouped_absmax_decoder group 64, gated_delta_recurrence). Refuses "
            "machine-specific field names."
        ),
        "does_not": (
            "Execute. complete_bits_per_weight is the packer's declared number. "
            "validate() accepted a 1 GB SharedBasis + TensorTrain family + 50 MB "
            "generated blob; BPW stayed 4.252735126866492 "
            "(receipts/headless/NOETIC_CLOSURE_GAP.json)."
        ),
        "kind": "schema",
    },
    {
        "what": "hawking.nos.noetic_executable_genome (NX / G104)",
        "path": "tools/nx_genome.py",
        "receipt": "receipts/ascent-2026-08-16/G104_NX_SEAL.json",
        "does": (
            "Machine compilation: genome digest, the 38 dispatched kernels of 554 "
            "declared, threadgroup geometry, residency, scheduling. An NX that "
            "could load anywhere has failed; the 40-core refusal test is the teeth."
        ),
        "does_not": (
            "Interpret a representation. Kernel names are a seal of what the "
            "Rust decode path already dispatches, not a graph a VM walks."
        ),
        "kind": "schema",
    },
    {
        "what": "hawking.gravity1.program.v1 (gravity_ir)",
        "path": "tools/gravity_ir.py",
        "receipt": None,
        "does": (
            "Program of QuantTensor/SharedBasis/GeneratedBlock/… nodes. Every "
            "node reports stored_bytes. Shared objects are content-addressed "
            "and counted once. Complete BPW is computed FROM the program. "
            "Every node NAMES a kernel."
        ),
        "does_not": (
            "Execute. Node.kernel is a string. There is no interpreter. "
            "'A representation with no execution path is not a representation, "
            "it is a compression demo' — and the kernel field is still a name."
        ),
        "kind": "schema",
    },
    {
        "what": "hawking.nos.nvm_minimal.v1 (G120)",
        "path": "tools/nvm_minimal.py",
        "receipt": None,
        "does": "Binds G106 route kinds to NX kernel names, then invokes the Rust greedy binary.",
        "does_not": (
            "Dispatch Metal or own the residual stream. Explicitly refuses to "
            "reimplement execution in Python."
        ),
        "kind": "orchestrator",
    },
    {
        "what": "production fused Q4 GEMV",
        "path": "crates/hawking-core/shaders/qwen_uniform_q4.metal",
        "receipt": "receipts/headless/NOETIC_KERNEL_CENSUS.json",
        "does": (
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 unpacks nibbles in "
            "registers and FMAs into y. 401 dispatches/token. W is never written. "
            "NATIVE (NOETIC_NATIVE_OPERATOR.json)."
        ),
        "does_not": "Take an IR document as input. The graph is hard-coded in qwen38_hybrid_decode.rs.",
        "kind": "executing_kernel",
    },
    {
        "what": "operation / kernel censuses",
        "path": "tools/headless/noetic_operation_census.py",
        "receipt": "receipts/headless/NOETIC_OPERATION_CENSUS.json",
        "does": "Enumerate what actually runs (38 dispatched, workhorse grouped-absmax q4).",
        "does_not": "Provide a node type or an interpreter.",
        "kind": "census",
    },
]


# ---------------------------------------------------------------------------
# Node types. A type nothing drove is omitted.
# ---------------------------------------------------------------------------

NODE_TYPES: dict[str, dict[str, Any]] = {
    "grouped_absmax": {
        "executes": True,
        "family": "grouped_absmax_q4",
        "experiment": (
            "uniform-q4-v1 production decode: grouped-absmax q4 group-64 is the "
            "workhorse GEMV (401 dispatches/token of 964)."
        ),
        "receipts": [
            "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            "receipts/headless/NOETIC_OPERATION_CENSUS.json",
            "receipts/headless/NOETIC_NATIVE_OPERATOR.json",
        ],
        "semantic": "signed offset-binary nibbles, fp16 absmax/8 scale, group 64",
        "machine_example": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 tg=128 tpr=64",
    },
    "ternary_group64": {
        "executes": True,
        "family": "ternary_group64",
        "experiment": (
            "noetic_composition requantize_absmax bits=2 (Ternary Weight Networks "
            "threshold 0.7·mean|w|, alpha refit on the kept set). Whole-model "
            "ternary flipped the composition-ladder prediction."
        ),
        "receipts": [
            "receipts/headless/NOETIC_COMPOSITION.json",
            "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
        ],
        "semantic": "per-group ternary {-α,0,+α}, group 64",
        "machine_example": "cpu_interpreter (no Metal ternary kernel on the Qwen3.8 path)",
    },
    "binary_sign_codes": {
        "executes": True,
        "family": "binary_sign_codes",
        "experiment": (
            "HGRAVB01 binary_sign_scale on Q80 gate_proj; composition 1-bit sign "
            "code (α = mean|w|) after bits=1 absmax collapsed to the zero tensor. "
            "test_lowbit_codec locks the sign-code optimum."
        ),
        "receipts": [
            "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            "tools/headless/test_lowbit_codec.py",
        ],
        "semantic": "1-bit sign, fp16 mean-abs scale, group 64",
        "machine_example": "q80_binary_group_matvec_tg256 (Q80 production uses group 128)",
    },
    "low_rank_uv": {
        "executes": True,
        "family": "low_rank_uv",
        "experiment": (
            "HGRAVS01 y=L@(R@x); q80_hgravs01_two_stage_matvec threadgroup mid[160]. "
            "NATIVE vs reconstruct-then-GEMM ORACLE in NOETIC_NATIVE_OPERATOR."
        ),
        "receipts": [
            "receipts/headless/NOETIC_NATIVE_OPERATOR.json",
            "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
        ],
        "semantic": "y = L @ (R @ x); W is not a stored object",
        "machine_example": "q80_hgravs01_two_stage_matvec / two q80_hgravs01_factor_matvec_*",
    },
    "product_quantization": {
        "executes": True,
        "family": "product_quantization",
        "experiment": (
            "gravity_pq_matvec: per subspace, pq_index → codebook entry, fma into "
            "acc, simd_sum to y[row]. EXISTS and REACHABLE; not on uniform-q4-v1. "
            "C4 refused the Qwen3.8 port, not the kernel."
        ),
        "receipts": [
            "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            "receipts/headless/C4CODEBOOK_DESIGN.json",
        ],
        "semantic": "subspace codebooks × packed indices; W never written",
        "machine_example": "gravity_pq_matvec",
    },
    "shared_basis": {
        "executes": False,
        "family": "shared_basis",
        "experiment": (
            "G035 shared column-basis vs independent SVD at matched bits: "
            "shared_beats_independent=false on 3/3 pairs. C1: NOT_WORTH_BUILDING "
            "(fidelity, not reconstruction). Kernel census family "
            "shared_basis_x_coefficients = ABSENT. The node exists so a planted "
            "basis is charged; it has no executor."
        ),
        "receipts": [
            "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            "receipts/headless/NOETIC_CLOSURE_GAP.json",
        ],
        "semantic": "one basis stored once; per-site coefficients",
        "machine_example": None,
    },
}

CANNOT_EXPRESS = [
    {
        "what": "tensor_train / Tucker / tensor-ring as an executing node",
        "why": (
            "G034/C2: 373/373 rows unhealthy, healthy=true count 0 at sub-0.5 BPW. "
            "G096 NEVER BUILT a TT node in NR. Kernel census tensor_contraction = "
            "ABSENT. A node nothing executed, that the experiment closed, is "
            "speculative — left out."
        ),
        "receipts": [
            "receipts/headless/C2TENSOROP_DESIGN.json",
            "receipts/headless/NOETIC_KERNEL_CENSUS.json",
        ],
    },
    {
        "what": "shared_basis as an executing operator",
        "why": (
            "The node is in the IR so accounting cannot go blind, but there is no "
            "executor: zero shaders mention shared_basis. Execute raises "
            "UnexecutableNode. This is the honest form of the G103 hole: we "
            "charge the 1 GB and we refuse to claim it ran."
        ),
        "receipts": ["receipts/headless/C1SHAREDBASIS_DESIGN.json"],
    },
    {
        "what": "UV + sparse as one fused representation",
        "why": (
            "C3: no kernel does UV + sparse in one representation. Mixed decode "
            "assigns low-rank to down_proj and binary+sparse to up_proj — "
            "different organs. Fusion traffic save is 2199.6 ns. NOT_WORTH_BUILDING."
        ),
        "receipts": ["receipts/headless/C3LOWRANKSPARSE_DESIGN.json"],
    },
    {
        "what": "structured Hadamard as a weight-side operator without materialising W",
        "why": (
            "strand_rht_forward_cols transforms x then a bitslice GEMV consumes "
            "tx (feature=tq). G032 Block-diagonal Sylvester-Hadamard was a "
            "reparameterization with GENERATED_BPW_EQUIVALENT=0.0. Kernel census "
            "structured_transform = PARTIAL."
        ),
        "receipts": ["receipts/headless/NOETIC_KERNEL_CENSUS.json"],
    },
    {
        "what": "a sidecar file that never became a node (the remaining G103 hole)",
        "why": (
            "account() sums payload bytes + referenced pool objects. A path "
            "string with no nbytes (NR's basis.bin next to tensors/) is still "
            "invisible. Closing THAT is the executable-closure hash over files "
            "the loader actually opens (NOETIC_EXECUTABLE_CLOSURE.json), not "
            "another schema field. This IR does not walk directories and does "
            "not load a 27B to find stragglers."
        ),
        "receipts": [
            "receipts/headless/NOETIC_CLOSURE_GAP.json",
            "receipts/headless/NOETIC_EXECUTABLE_CLOSURE.json",
        ],
    },
    {
        "what": "whole-model Metal dispatch of a 27B from this document",
        "why": (
            "Do not load a second 27B. The production kernel is cited, not relaunched. "
            "Semantic execution here is a CPU interpreter of the same function on "
            "micro-sites."
        ),
        "receipts": ["receipts/headless/NOETIC_NATIVE_OPERATOR.json"],
    },
]


class UnexecutableNode(RuntimeError):
    """Raised when a node has no interpreter. SharedBasis hits this on purpose."""


class MachineRefusal(RuntimeError):
    """Raised when a machine lowering does not match this box. NX teeth."""


class SemanticContamination(ValueError):
    """Raised when a machine-only field is found on the semantic side. NR teeth."""


# ---------------------------------------------------------------------------
# tiny utils
# ---------------------------------------------------------------------------

def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    return str(x)


def walk_keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, f"{path}.{k}" if path else k
            yield from walk_keys(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_keys(v, f"{path}[{i}]")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def f16_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype="<f2").tobytes()


def f16_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f2")


def f32_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype="<f4").tobytes()


def f32_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


# ---------------------------------------------------------------------------
# shared pool (content-addressed; counted once). Same law as gravity_ir.
# ---------------------------------------------------------------------------

class SharedPool:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def put(self, kind: str, nbytes: int, content_id: str | None = None, **meta) -> str:
        if content_id is None:
            key = {"kind": kind, "nbytes": nbytes, **meta, "_n": len(self.objects)}
            content_id = hashlib.sha256(
                json.dumps(key, sort_keys=True).encode()
            ).hexdigest()[:16]
        if content_id in self.objects:
            if self.objects[content_id]["nbytes"] != nbytes:
                raise ValueError(f"content id {content_id} collides on size")
        else:
            self.objects[content_id] = {"kind": kind, "nbytes": nbytes, **meta}
        return content_id

    def bytes_used(self, refs: set[str]) -> int:
        return sum(self.objects[c]["nbytes"] for c in refs)


# ---------------------------------------------------------------------------
# packers / interpreters  (semantic function of each family)
# ---------------------------------------------------------------------------

def pack_grouped_absmax_q4(W: np.ndarray, group: int = 64) -> dict:
    """Nibble layout matching qwen_uniform_q4_group64_matvec_geo_tpr64_tg128.

    code byte i stores even local in the low nibble and odd local in the high
    nibble; q = nibble - 8 in [-8, 7]; value = float(q) * fp16(absmax/8).
    """
    W = np.asarray(W, dtype=np.float32)
    rows, cols = int(W.shape[0]), int(W.shape[1])
    if cols % group != 0:
        raise ValueError(f"cols {cols} not divisible by group {group}")
    gpr = cols // group
    g = W.reshape(rows * gpr, group)
    bound = np.float32(8.0)
    amax = np.max(np.abs(g), axis=1)
    scales = (amax / bound).astype(np.float16)
    den = np.where(scales.astype(np.float32) > 0.0, scales.astype(np.float32), 1.0)
    q = np.rint(g / den[:, None]).clip(-8, 7).astype(np.int16)
    nib = (q + 8).astype(np.uint8)
    codes = np.empty((g.shape[0], group // 2), dtype=np.uint8)
    codes[:] = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return {
        "rows": rows, "cols": cols, "group": group, "bits": 4,
        "scales": f16_bytes(scales),
        "codes": codes.tobytes(),
    }


def _q4_groups(payload: dict) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    rows, cols, group = payload["rows"], payload["cols"], payload["group"]
    gpr = cols // group
    scales = f16_from_bytes(payload["scales"]).astype(np.float32)
    codes = np.frombuffer(payload["codes"], dtype=np.uint8).reshape(rows * gpr, group // 2)
    q = np.empty((rows * gpr, group), dtype=np.float32)
    q[:, 0::2] = (codes.astype(np.int16) & 0x0F) - 8
    q[:, 1::2] = (codes.astype(np.int16) >> 4) - 8
    return q, scales, rows, cols, group


def execute_grouped_absmax_q4(payload: dict, x: np.ndarray) -> np.ndarray:
    """Fused unpack+FMA. Never writes a (rows × cols) W."""
    q, scales, rows, cols, group = _q4_groups(payload)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != cols:
        raise ValueError("x length != cols")
    gpr = cols // group
    xg = x.reshape(gpr, group)
    q3 = q.reshape(rows, gpr, group)
    sc = scales.reshape(rows, gpr)
    # Metal: acc += float(q) * scale * x[col]. Never writes a (rows × cols) W.
    y = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        acc = np.float32(0.0)
        for g in range(gpr):
            acc += np.dot(q3[r, g] * sc[r, g], xg[g])
        y[r] = acc
    return y


def reconstruct_grouped_absmax_q4(payload: dict) -> np.ndarray:
    q, scales, rows, cols, group = _q4_groups(payload)
    gpr = cols // group
    return (q * scales[:, None]).reshape(rows, cols)


def pack_binary_sign(W: np.ndarray, group: int = 64) -> dict:
    """HGRAVB01 function: fp16 mean-abs scale, LSB-first sign bits."""
    W = np.asarray(W, dtype=np.float32)
    rows, cols = int(W.shape[0]), int(W.shape[1])
    if cols % group != 0:
        raise ValueError("cols not divisible by group")
    gpr = cols // group
    g = W.reshape(rows * gpr, group)
    scales = np.mean(np.abs(g), axis=1).astype(np.float16)
    bits = (g.reshape(-1) >= 0.0).astype(np.uint8)
    packed = np.zeros((bits.size + 7) // 8, dtype=np.uint8)
    idx = np.arange(bits.size)
    np.bitwise_or.at(packed, idx >> 3, (bits << (idx & 7)).astype(np.uint8))
    return {
        "rows": rows, "cols": cols, "group": group, "bits": 1,
        "scales": f16_bytes(scales),
        "codes": packed.tobytes(),
    }


def _unpack_lsb_bits(blob: bytes, nbits: int) -> np.ndarray:
    packed = np.frombuffer(blob, dtype=np.uint8)
    idx = np.arange(nbits)
    return ((packed[idx >> 3] >> (idx & 7)) & 1).astype(np.uint8)


def execute_binary_sign(payload: dict, x: np.ndarray) -> np.ndarray:
    rows, cols, group = payload["rows"], payload["cols"], payload["group"]
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    gpr = cols // group
    scales = f16_from_bytes(payload["scales"]).astype(np.float32).reshape(rows, gpr)
    signs = _unpack_lsb_bits(payload["codes"], rows * cols).reshape(rows, gpr, group)
    pm = np.where(signs, 1.0, -1.0).astype(np.float32)
    xg = x.reshape(gpr, group)
    return ((pm * scales[:, :, None]) * xg[None, :, :]).sum(axis=(1, 2)).astype(np.float32)


def reconstruct_binary_sign(payload: dict) -> np.ndarray:
    rows, cols, group = payload["rows"], payload["cols"], payload["group"]
    gpr = cols // group
    scales = f16_from_bytes(payload["scales"]).astype(np.float32).reshape(rows, gpr, 1)
    signs = _unpack_lsb_bits(payload["codes"], rows * cols).reshape(rows, gpr, group)
    pm = np.where(signs, 1.0, -1.0).astype(np.float32)
    return (pm * scales).reshape(rows, cols)


def pack_ternary_group64(W: np.ndarray, group: int = 64) -> dict:
    """Composition codec at bits=2: t = 0.7·mean|w|, α = mean |w| of the kept set."""
    W = np.asarray(W, dtype=np.float32)
    rows, cols = int(W.shape[0]), int(W.shape[1])
    if cols % group != 0:
        raise ValueError("cols not divisible by group")
    gpr = cols // group
    g = W.reshape(rows * gpr, group)
    absg = np.abs(g)
    thresh = 0.7 * absg.mean(axis=1, keepdims=True)
    keep = absg > thresh
    kept = np.where(keep, absg, 0.0).sum(axis=1, keepdims=True)
    cnt = keep.sum(axis=1, keepdims=True)
    alpha = np.where(cnt > 0, kept / np.maximum(cnt, 1), 0.0).astype(np.float16)
    # codes in {0,1,2} ↔ {-1,0,+1}
    codes = np.where(keep & (g >= 0), 2, np.where(keep & (g < 0), 0, 1)).astype(np.uint8)
    flat = codes.reshape(-1)
    # 2 bits each, LSB-first pairs in a byte
    n = flat.size
    packed = np.zeros((n + 3) // 4, dtype=np.uint8)
    for i, c in enumerate(flat):
        packed[i >> 2] |= (int(c) & 0x3) << ((i & 3) * 2)
    return {
        "rows": rows, "cols": cols, "group": group, "bits": 2,
        "scales": f16_bytes(alpha.reshape(-1)),
        "codes": packed.tobytes(),
    }


def _unpack_ternary_codes(payload: dict) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    rows, cols, group = payload["rows"], payload["cols"], payload["group"]
    gpr = cols // group
    n = rows * cols
    packed = np.frombuffer(payload["codes"], dtype=np.uint8)
    raw = np.empty(n, dtype=np.uint8)
    for i in range(n):
        raw[i] = (int(packed[i >> 2]) >> ((i & 3) * 2)) & 0x3
    mapped = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)[raw]
    scales = f16_from_bytes(payload["scales"]).astype(np.float32)
    return mapped.reshape(rows * gpr, group), scales, rows, cols, group


def execute_ternary_group64(payload: dict, x: np.ndarray) -> np.ndarray:
    q, scales, rows, cols, group = _unpack_ternary_codes(payload)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    gpr = cols // group
    q3 = q.reshape(rows, gpr, group)
    sc = scales.reshape(rows, gpr)
    xg = x.reshape(gpr, group)
    return ((q3 * sc[:, :, None]) * xg[None, :, :]).sum(axis=(1, 2)).astype(np.float32)


def reconstruct_ternary_group64(payload: dict) -> np.ndarray:
    q, scales, rows, cols, group = _unpack_ternary_codes(payload)
    gpr = cols // group
    return (q * scales[:, None]).reshape(rows, cols)


def pack_low_rank_uv(L: np.ndarray, R: np.ndarray) -> dict:
    L = np.asarray(L, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    rows, rank = int(L.shape[0]), int(L.shape[1])
    rank2, cols = int(R.shape[0]), int(R.shape[1])
    if rank != rank2:
        raise ValueError("rank mismatch")
    return {
        "rows": rows, "cols": cols, "rank": rank,
        "L": f32_bytes(L), "R": f32_bytes(R),
    }


def execute_low_rank_uv(payload: dict, x: np.ndarray) -> np.ndarray:
    """y = L @ (R @ x). Mid is rank-wide. W is never formed."""
    rows, cols, rank = payload["rows"], payload["cols"], payload["rank"]
    L = f32_from_bytes(payload["L"]).reshape(rows, rank)
    R = f32_from_bytes(payload["R"]).reshape(rank, cols)
    x = np.asarray(x, dtype=np.float32).reshape(cols)
    mid = R @ x
    return L @ mid


def reconstruct_low_rank_uv(payload: dict) -> np.ndarray:
    rows, cols, rank = payload["rows"], payload["cols"], payload["rank"]
    L = f32_from_bytes(payload["L"]).reshape(rows, rank)
    R = f32_from_bytes(payload["R"]).reshape(rank, cols)
    return L @ R


def pack_pq(codebooks: np.ndarray, indices: np.ndarray, bits: int) -> dict:
    """codebooks: (subspaces, card, sub); indices: (rows, nchunk, subspaces) uint."""
    cb = np.asarray(codebooks, dtype=np.float32)
    idx = np.asarray(indices)
    subspaces, card, sub = (int(x) for x in cb.shape)
    rows, nchunk, s2 = (int(x) for x in idx.shape)
    if s2 != subspaces:
        raise ValueError("index subspaces mismatch")
    cols = nchunk * subspaces * sub
    # pack indices MSB-first, matching gravity_pq.metal pq_index
    flat = idx.reshape(-1).astype(np.uint32)
    n = flat.size
    nbits = n * bits
    packed = np.zeros((nbits + 7) // 8 + 4, dtype=np.uint8)  # +4 tail like the kernel
    for i, v in enumerate(flat):
        bitoff = i * bits
        for b in range(bits):
            bit = (int(v) >> (bits - 1 - b)) & 1
            pos = bitoff + b
            packed[pos >> 3] |= bit << (7 - (pos & 7))
    return {
        "rows": rows, "cols": cols, "subspaces": subspaces, "card": card,
        "sub": sub, "nchunk": nchunk, "bits": bits,
        "codebooks": f32_bytes(cb),
        "codes": packed.tobytes(),
    }


def _pq_index(codes: np.ndarray, i: int, bits: int) -> int:
    bitoff = i * bits
    byte = bitoff >> 3
    shift = bitoff & 7
    word = (
        (int(codes[byte]) << 24) | (int(codes[byte + 1]) << 16)
        | (int(codes[byte + 2]) << 8) | int(codes[byte + 3])
    )
    return (word >> (32 - shift - bits)) & ((1 << bits) - 1)


def execute_pq(payload: dict, x: np.ndarray) -> np.ndarray:
    """Fused dictionary lookup + accumulate. W is never written."""
    rows = payload["rows"]
    subspaces = payload["subspaces"]
    card = payload["card"]
    sub = payload["sub"]
    nchunk = payload["nchunk"]
    bits = payload["bits"]
    dim = subspaces * sub
    cb = f32_from_bytes(payload["codebooks"]).reshape(subspaces, card, sub)
    codes = np.frombuffer(payload["codes"], dtype=np.uint8)
    x = np.asarray(x, dtype=np.float32).reshape(nchunk * dim)
    y = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        acc = np.float32(0.0)
        for s in range(subspaces):
            for c in range(nchunk):
                flat = (row * nchunk + c) * subspaces + s
                entry = cb[s, _pq_index(codes, flat, bits)]
                base = c * dim + s * sub
                acc = acc + np.dot(entry, x[base:base + sub])
        y[row] = acc
    return y


def reconstruct_pq(payload: dict) -> np.ndarray:
    rows = payload["rows"]
    subspaces = payload["subspaces"]
    card = payload["card"]
    sub = payload["sub"]
    nchunk = payload["nchunk"]
    bits = payload["bits"]
    cols = payload["cols"]
    dim = subspaces * sub
    cb = f32_from_bytes(payload["codebooks"]).reshape(subspaces, card, sub)
    codes = np.frombuffer(payload["codes"], dtype=np.uint8)
    W = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for s in range(subspaces):
            for c in range(nchunk):
                flat = (row * nchunk + c) * subspaces + s
                entry = cb[s, _pq_index(codes, flat, bits)]
                base = c * dim + s * sub
                W[row, base:base + sub] = entry
    return W


EXECUTORS = {
    "grouped_absmax": (execute_grouped_absmax_q4, reconstruct_grouped_absmax_q4),
    "ternary_group64": (execute_ternary_group64, reconstruct_ternary_group64),
    "binary_sign_codes": (execute_binary_sign, reconstruct_binary_sign),
    "low_rank_uv": (execute_low_rank_uv, reconstruct_low_rank_uv),
    "product_quantization": (execute_pq, reconstruct_pq),
}


# ---------------------------------------------------------------------------
# IR graph
# ---------------------------------------------------------------------------

def payload_bytes(payload: dict) -> int:
    n = 0
    for k, v in payload.items():
        if isinstance(v, (bytes, bytearray, memoryview)):
            n += len(v)
    return n


def make_node(
    node_id: str,
    kind: str,
    payload: dict | None,
    *,
    lowering: dict,
    shared_refs: list[str] | None = None,
    exclusive_bytes: int = 0,
) -> dict:
    spec = NODE_TYPES[kind]
    semantic = {
        "id": node_id,
        "kind": kind,
        "family": spec["family"],
        "executes": spec["executes"],
        "experiment": spec["experiment"],
        "receipts": list(spec["receipts"]),
        "shape": {k: payload[k] for k in ("rows", "cols", "group", "bits", "rank",
                                          "subspaces", "card", "sub")
                  if payload is not None and k in payload},
        "shared_refs": list(shared_refs or []),
        "exclusive_bytes": exclusive_bytes if payload is None else payload_bytes(payload),
    }
    return {
        "semantic": semantic,
        "machine": dict(lowering),
        "payload": payload,
    }


def validate_semantic(nodes: list[dict]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for node in nodes:
        sem = node["semantic"]
        for k, path in walk_keys(sem):
            if k.lower() in MACHINE_SPECIFIC:
                bad.append(f"MACHINE-SPECIFIC FIELD in semantic: {sem['id']}.{path}")
        if sem["kind"] not in NODE_TYPES:
            bad.append(f"unknown node kind {sem['kind']}")
        if "implementation" in sem or "kernel" in sem:
            bad.append(f"semantic names an implementation: {sem['id']}")
    return (not bad), bad


def this_machine_stub() -> dict:
    """CPU-interpreter genome. Not an NX seal; portable on purpose."""
    return {
        "kind": "semantic_interpreter",
        "portable": True,
        "target": "cpu",
        "note": (
            "This is the SEMANTIC function of the node, not an NX. An NX that "
            "could load anywhere has failed; this interpreter is supposed to."
        ),
    }


def check_lowering(lowering: dict) -> None:
    if lowering.get("kind") == "semantic_interpreter":
        return
    # Metal / NX-shaped lowerings must name a genome and match it.
    want = lowering.get("compiled_for_machine_genome")
    if not want:
        raise MachineRefusal("non-portable lowering has no compiled_for_machine_genome")
    here = lowering.get("required_here")
    if here and here != want:
        raise MachineRefusal(
            f"MACHINE GENOME MISMATCH: lowering built for {want}, this box is {here}"
        )


def execute_node(node: dict, x: np.ndarray) -> np.ndarray:
    kind = node["semantic"]["kind"]
    if kind not in EXECUTORS:
        raise UnexecutableNode(
            f"{node['semantic']['id']}: family {kind} has no interpreter "
            f"(justified by {NODE_TYPES[kind]['experiment'][:80]}…)"
        )
    check_lowering(node["machine"])
    fn, _ = EXECUTORS[kind]
    return fn(node["payload"], x)


def direct_node(node: dict, x: np.ndarray) -> np.ndarray:
    kind = node["semantic"]["kind"]
    if kind not in EXECUTORS:
        raise UnexecutableNode(f"{node['semantic']['id']}: no direct path")
    _, recon = EXECUTORS[kind]
    W = recon(node["payload"])
    return np.asarray(W, dtype=np.float32) @ np.asarray(x, dtype=np.float32)


def account(nodes: list[dict], pool: SharedPool) -> dict:
    exclusive = 0
    refs: set[str] = set()
    for node in nodes:
        exclusive += int(node["semantic"]["exclusive_bytes"])
        refs.update(node["semantic"]["shared_refs"])
    shared = pool.bytes_used(refs)
    total = exclusive + shared
    return {
        "exclusive_bytes": exclusive,
        "shared_bytes": shared,
        "total_bytes": total,
        "source_param_count": SOURCE_PARAM_COUNT,
        "complete_bpw": 8.0 * total / SOURCE_PARAM_COUNT,
        "shared_objects": {c: pool.objects[c] for c in sorted(refs)},
        "law": (
            "complete_bpw = 8 * (payload bytes + referenced pool objects counted "
            "once) / SOURCE_PARAM_COUNT. Declared packer BPW is not consulted."
        ),
    }


def compare_execute_vs_direct(node: dict, x: np.ndarray) -> dict:
    y_ir = execute_node(node, x)
    y_direct = direct_node(node, x)
    return {
        "node_id": node["semantic"]["id"],
        "family": node["semantic"]["family"],
        "kind": node["semantic"]["kind"],
        "executes": True,
        "shape": {"rows": int(y_ir.shape[0]), "cols": int(np.asarray(x).size)},
        "y_from_ir": [float(v) for v in y_ir[:8]],
        "y_direct": [float(v) for v in y_direct[:8]],
        "y_from_ir_all": [float(v) for v in y_ir],
        "y_direct_all": [float(v) for v in y_direct],
        "max_abs_diff": max_abs(y_ir, y_direct),
        "rel_l2": rel_l2(y_ir, y_direct),
        "match_atol_1e5": bool(np.allclose(y_ir, y_direct, rtol=0.0, atol=1e-5)),
        "direct_is": (
            "the same function evaluated by reconstructing the implied matrix "
            "and doing a dense matvec. Difference is floating-point association, "
            "not a different codec."
        ),
        "ir_path": "interpreter of the packed payload; W is not written",
        "machine": node["machine"],
    }


# ---------------------------------------------------------------------------
# build the demonstration graph (synthetic micro-sites, no 27B)
# ---------------------------------------------------------------------------

SEED = 20260823


def _rng() -> np.random.RandomState:
    return np.random.RandomState(SEED)


def build_executing_nodes(rng: np.random.RandomState) -> list[dict]:
    cpu = this_machine_stub()
    nodes: list[dict] = []

    Wq = rng.randn(16, 64).astype(np.float32)
    nodes.append(make_node(
        "site.q4", "grouped_absmax", pack_grouped_absmax_q4(Wq),
        lowering={**cpu, "equivalent_metal_kernel":
                  "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                  "equivalent_metal_kernel_is_not_semantic": True},
    ))
    # stash source W so the receipt can also show quantisation error vs parent
    nodes[-1]["_source_W"] = Wq

    Wt = rng.randn(8, 64).astype(np.float32)
    nodes.append(make_node(
        "site.ternary", "ternary_group64", pack_ternary_group64(Wt),
        lowering=cpu,
    ))

    Wb = rng.randn(8, 64).astype(np.float32)
    nodes.append(make_node(
        "site.binary", "binary_sign_codes", pack_binary_sign(Wb),
        lowering={**cpu, "equivalent_metal_kernel": "q80_binary_group_matvec_tg256",
                  "equivalent_metal_kernel_is_not_semantic": True},
    ))

    rank, rows, cols = 4, 12, 32
    L = rng.randn(rows, rank).astype(np.float32)
    R = rng.randn(rank, cols).astype(np.float32)
    nodes.append(make_node(
        "site.uv", "low_rank_uv", pack_low_rank_uv(L, R),
        lowering={**cpu, "equivalent_metal_kernel": "q80_hgravs01_two_stage_matvec",
                  "equivalent_metal_kernel_is_not_semantic": True},
    ))

    # PQ constructed FROM the codebook so the representation IS the function.
    subspaces, card, sub, rows_pq = 4, 8, 8, 6
    bits = 3
    cb = rng.randn(subspaces, card, sub).astype(np.float32)
    idx = rng.randint(0, card, size=(rows_pq, 1, subspaces))
    nodes.append(make_node(
        "site.pq", "product_quantization", pack_pq(cb, idx, bits),
        lowering={**cpu, "equivalent_metal_kernel": "gravity_pq_matvec",
                  "equivalent_metal_kernel_is_not_semantic": True},
    ))
    return nodes


def planted_basis_node(pool: SharedPool) -> dict:
    cid = pool.put(
        "SharedBasis", nbytes=PLANTED_SHARED_BASIS_BYTES,
        content_id="planted-1gb-shared-basis",
        rank=256, note="injection: 1 GB basis NR.validate() accepted without moving BPW",
    )
    return make_node(
        "planted.shared_basis", "shared_basis", None,
        lowering=this_machine_stub(),
        shared_refs=[cid],
        exclusive_bytes=0,
    )


def run_comparisons(nodes: list[dict], rng: np.random.RandomState) -> list[dict]:
    out = []
    for node in nodes:
        if not node["semantic"]["executes"]:
            continue
        cols = node["semantic"]["shape"]["cols"]
        x = rng.randn(int(cols)).astype(np.float32)
        rec = compare_execute_vs_direct(node, x)
        src = node.get("_source_W")
        if src is not None:
            y_src = src @ x
            rec["y_source_dense"] = [float(v) for v in y_src[:8]]
            rec["quantization_rel_l2_vs_source"] = rel_l2(
                np.array(rec["y_from_ir_all"]), y_src
            )
            rec["source_note"] = (
                "y_source_dense is original W @ x, a DIFFERENT computation "
                "(quantisation error). y_direct is reconstructed-W @ x, the "
                "same computation as the IR interpreter."
            )
        # drop full vectors from the comparison copy used in tests? keep in rec
        out.append(rec)
    return out


def strip_payloads_for_receipt(nodes: list[dict]) -> list[dict]:
    """JSON cannot carry raw bytes; record lengths and sha256 instead."""
    rows = []
    for node in nodes:
        payload = node.get("payload")
        blobs = {}
        if payload:
            for k, v in payload.items():
                if isinstance(v, (bytes, bytearray)):
                    blobs[k] = {
                        "bytes": len(v),
                        "sha256": hashlib.sha256(v).hexdigest(),
                    }
                else:
                    blobs[k] = v
        rows.append({
            "semantic": node["semantic"],
            "machine": node["machine"],
            "payload": blobs,
        })
    return rows


def build() -> dict:
    t0 = time.time()
    rng = _rng()
    pool = SharedPool()
    executing = build_executing_nodes(rng)
    ok, problems = validate_semantic(executing)
    if not ok:
        raise SemanticContamination(problems)

    comparisons = run_comparisons(executing, rng)
    before = account(executing, pool)

    planted = planted_basis_node(pool)
    after_nodes = executing + [planted]
    after = account(after_nodes, pool)
    planted_exec_error = None
    try:
        execute_node(planted, np.zeros(1, dtype=np.float32))
    except UnexecutableNode as exc:
        planted_exec_error = str(exc)

    # NX-shaped refusal: a metal lowering sealed for a 40-core GPU.
    metal_refusal = None
    try:
        fake = {
            "kind": "metal_kernel",
            "portable": False,
            "compiled_for_machine_genome": "gpu_cores=40",
            "required_here": "gpu_cores=60",
        }
        check_lowering(fake)
        metal_refusal = "LOADED -- THE CHECK IS BROKEN"
    except MachineRefusal as exc:
        metal_refusal = f"REFUSED, as required: {exc}"

    executed_families = [c["family"] for c in comparisons]
    flagship = next(c for c in comparisons if c["kind"] == "grouped_absmax")

    # contamination negative test (in-memory, does not mutate the graph)
    dirty = json.loads(json.dumps(jsonable(executing[0]["semantic"])))
    dirty["threadgroup_size"] = 128
    dirty["kernel"] = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"
    dirty_ok, dirty_bad = validate_semantic([{"semantic": dirty, "machine": {}, "payload": None}])

    elapsed = round(time.time() - t0, 4)
    doc = {
        "schema": SCHEMA,
        "ir_kind": IR_KIND,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "elapsed_s": elapsed,
        "question": (
            "Is there an IR with an executing node — semantic representation "
            "versus machine compilation — that produces the same numbers as "
            "the same computation done directly?"
        ),
        "answer": (
            f"Yes. Family {flagship['family']} is expressed as a semantic IR "
            f"node and EXECUTED from it. y_from_ir vs y_direct max_abs_diff="
            f"{flagship['max_abs_diff']:.3e} rel_l2={flagship['rel_l2']:.3e}. "
            "Kernel names live on the machine lowering, not the semantic node. "
            "A 1 GB planted SharedBasis moves complete_bpw and cannot execute."
        ),
        "doctrine": {
            "semantic": (
                "what function this is, portable. grouped_absmax q4 is offset-binary "
                "nibbles with fp16 absmax/8 scale at group 64, regardless of device."
            ),
            "machine": (
                "how it executes here. geo_tpr64_tg128, threadgroup 128, the M3 Ultra "
                "genome — NX. An NX that could load anywhere has failed."
            ),
            "executing_node": (
                "the semantic interpreter: it reads the packed payload and produces y "
                "without writing W. That is the IR. The Metal kernel is a lowering "
                "of the same function, not the function."
            ),
            "nr_nx_split_preserved": True,
        },
        "already_approximates_the_ir": ALREADY_APPROXIMATES,
        "node_types": NODE_TYPES,
        "cannot_express": CANNOT_EXPRESS,
        "executed_families": executed_families,
        "flagship": {
            "family": flagship["family"],
            "node_id": flagship["node_id"],
            "y_from_ir": flagship["y_from_ir"],
            "y_direct": flagship["y_direct"],
            "max_abs_diff": flagship["max_abs_diff"],
            "rel_l2": flagship["rel_l2"],
            "match_atol_1e5": flagship["match_atol_1e5"],
            "y_source_dense": flagship.get("y_source_dense"),
            "quantization_rel_l2_vs_source": flagship.get("quantization_rel_l2_vs_source"),
            "direct_is": flagship["direct_is"],
        },
        "executions": [
            {k: v for k, v in c.items() if k not in ("y_from_ir_all", "y_direct_all")}
            for c in comparisons
        ],
        "graph": {
            "nodes": strip_payloads_for_receipt(executing),
            "n_executing": sum(1 for n in executing if n["semantic"]["executes"]),
            "n_total": len(executing),
        },
        "accounting": {
            "before_plant": before,
            "after_1gb_shared_basis": after,
            "planted_bytes": PLANTED_SHARED_BASIS_BYTES,
            "bpw_moved": after["complete_bpw"] != before["complete_bpw"],
            "bpw_delta": after["complete_bpw"] - before["complete_bpw"],
            "sealed_packer_bpw_was": SEALED_BPW,
            "planted_execute": planted_exec_error,
            "nr_hole": {
                "was": (
                    "NR.validate() returned ok and complete_bits_per_weight stayed "
                    f"{SEALED_BPW} after injecting a 1 GB SharedBasis, a TensorTrain "
                    "family, and a 50 MB generated blob "
                    "(receipts/headless/NOETIC_CLOSURE_GAP.json)."
                ),
                "here": (
                    "account() recomputes BPW from payloads + pool. The 1 GB is "
                    "charged (bpw_moved=true). execute() of the planted node raises "
                    "UnexecutableNode. Description without accounting is refused; "
                    "description without execution is named, not claimed as an IR."
                ),
            },
        },
        "semantic_vs_machine_teeth": {
            "machine_field_on_semantic_rejected": (not dirty_ok),
            "problems": dirty_bad,
            "metal_lowering_genome_mismatch": metal_refusal,
        },
        "did_not_load_27b": True,
        "opened_model_paths": [],
        "self_check": {
            "at_least_one_family_executed": len(executed_families) >= 1,
            "flagship_is_grouped_absmax_q4": flagship["family"] == "grouped_absmax_q4",
            "flagship_ir_matches_direct": flagship["match_atol_1e5"],
            "all_executing_match_direct": all(c["match_atol_1e5"] for c in comparisons),
            "semantic_refuses_kernel_name": not dirty_ok,
            "planted_1gb_moves_bpw": after["complete_bpw"] != before["complete_bpw"],
            "planted_1gb_cannot_execute": planted_exec_error is not None,
            "tensor_train_is_not_a_node": "tensor_train" not in NODE_TYPES,
            "every_node_type_names_an_experiment": all(
                bool(v.get("experiment")) and bool(v.get("receipts"))
                for v in NODE_TYPES.values()
            ),
            "no_27b_loaded": True,
        },
    }
    return jsonable(doc)


def write_receipt(doc: dict, path: Path = RECEIPT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def print_report(doc: dict) -> None:
    print(f"schema {doc['schema']}")
    print(f"ir_kind {doc['ir_kind']}")
    print(f"answer {doc['answer']}")
    print()
    print("executed families:")
    for ex in doc["executions"]:
        print(
            f"  {ex['family']:<24} max_abs {ex['max_abs_diff']:.3e}  "
            f"rel_l2 {ex['rel_l2']:.3e}  match={ex['match_atol_1e5']}"
        )
        print(f"    y_ir     {ex['y_from_ir']}")
        print(f"    y_direct {ex['y_direct']}")
    f = doc["flagship"]
    print()
    print(f"flagship {f['family']}")
    print(f"  y_from_ir {f['y_from_ir']}")
    print(f"  y_direct  {f['y_direct']}")
    print(f"  diff max_abs={f['max_abs_diff']:.3e} rel_l2={f['rel_l2']:.3e}")
    acc = doc["accounting"]
    print()
    print(f"BPW before plant {acc['before_plant']['complete_bpw']:.12f}")
    print(f"BPW after 1 GB   {acc['after_1gb_shared_basis']['complete_bpw']:.12f}")
    print(f"bpw_moved {acc['bpw_moved']}  planted_execute {acc['planted_execute']}")
    print()
    failed = [k for k, v in doc["self_check"].items() if v is not True]
    print("self_check", "PASS" if not failed else f"FAIL {failed}")


def main() -> int:
    doc = build()
    write_receipt(doc)
    print_report(doc)
    print(f"\nwrote {RECEIPT.relative_to(REPO)}")
    failed = [k for k, v in doc["self_check"].items() if v is not True]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
