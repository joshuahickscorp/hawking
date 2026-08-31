"""Adjudicate the six static ABI rows reported by the Claude sidecar.

This is a source-only feedback receipt. It records whether each preflight row
was a real defect or a parser limitation, and it refuses to emit any hardware
metric. The receipt is intentionally separate from the static preflight so a
future verifier can learn from both repaired defects and resolved false alarms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.future._common import REPO, RECEIPTS, git, load_json, write_receipt


RECEIPT = "CLAUDE_SIDECAR_ABI_ADJUDICATION.json"
SCHEMA = "hawking.future.claude_sidecar_abi_adjudication.v1"


def _source_line(relative: str, needle: str) -> str:
    path = REPO / relative
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if needle in line:
            return f"{relative}:{number}"
    raise RuntimeError(f"source needle not found: {relative}: {needle!r}")


def _require_source(relative: str, needle: str) -> None:
    _source_line(relative, needle)


def build() -> Path:
    static_path = RECEIPTS / "STATIC_KERNEL_PREFLIGHT.json"
    static = load_json(static_path)
    static_errors = [
        row for row in static.get("findings", []) if row.get("severity") == "ERROR"
    ]
    if static_errors:
        raise RuntimeError(
            "the current static preflight still has ERROR rows; do not seal an "
            f"adjudication receipt: {static_errors[:2]}"
        )

    attn = "crates/hawking-core/shaders/attn.metal"
    moe = "crates/hawking-core/shaders/moe.metal"
    kernels = "crates/hawking-core/src/kernels/mod.rs"
    verifier = "tools/future/static_kernel_verify.py"

    # These are the exact rows from the six-row Claude preflight. The original
    # host line numbers are retained as provenance; current source locations
    # are resolved below so edits do not silently stale the receipt.
    outcomes = [
        {
            "finding_id": "claude-abi-01-kv-append-q8",
            "preflight": {
                "severity": "ERROR",
                "check": "kernel_existence",
                "kernel": "kv_append_q8_0_f32",
                "host_sites": ["crates/hawking-core/src/kernels/mod.rs:8509"],
            },
            "classification": "REAL_DEFECT",
            "status": "FIXED",
            "reachability": (
                "Public kv_append_q8_0_f32_metal is called by q8_kv_parity; the "
                "entry point was not dead code and the host already selected it."
            ),
            "evidence": [
                _source_line(attn, "kernel void kv_append_q8_0_f32("),
                _source_line(kernels, '"kv_append_q8_0_f32"'),
            ],
            "fix": (
                "Added the Q8_0 kernel with the ArgbufKvAppend contract, fp16 scale "
                "serialization, ties-away-from-zero code rounding, and in-launch "
                "k_pe copy; widened the grid for a longer RoPE tail."
            ),
        },
        {
            "finding_id": "claude-abi-02-qwen-generated-k1",
            "preflight": {
                "severity": "ERROR",
                "check": "kernel_existence",
                "kernel": "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128",
                "host_sites": [
                    "crates/hawking-core/examples/ascension_qwen38_matmul_k_amortization.rs:221",
                    "crates/hawking-core/examples/ascension_qwen38_matmul_k_amortization.rs:351",
                ],
            },
            "classification": "BLOCKED_BY_GENERATION",
            "status": "RESOLVED_BY_MACRO_AWARE_PREFLIGHT",
            "reachability": (
                "The Qwen example references a concrete entry point emitted by "
                "QWEN_UNIFORM_Q4_MATMUL_K(1); it is reachable, but its body is "
                "created by token pasting before Metal compilation."
            ),
            "evidence": [
                _source_line(
                    "crates/hawking-core/shaders/qwen_uniform_q4.metal",
                    "#define QWEN_UNIFORM_Q4_MATMUL_K",
                ),
                _source_line(
                    "crates/hawking-core/shaders/qwen_uniform_q4.metal",
                    "QWEN_UNIFORM_Q4_MATMUL_K(1)",
                ),
                _source_line(verifier, "def generated_kernel_names"),
            ],
            "fix": (
                "Taught the static verifier to recover the K, RK, and binary-plane "
                "macro families. It resolves existence only; generated-body ABI "
                "and PSO checks remain explicitly compiler/runtime work."
            ),
        },
        {
            "finding_id": "claude-abi-03-gemv-attn-command-batch",
            "preflight": {
                "severity": "ERROR",
                "check": "type_width",
                "kernel": "gemv_f32_attn",
                "host_site": "crates/hawking-core/src/kernels/mod.rs:8694",
            },
            "classification": "REAL_DEFECT",
            "status": "FIXED",
            "reachability": "mla_decode_and_o_proj_metal is a live combined attention path.",
            "evidence": [
                _source_line(kernels, "let gemv_args = ArgbufRowsCols"),
                _source_line(attn, "constant ArgbufRowsCols& args [[buffer(3)]]"),
            ],
            "fix": "Packed rows and cols into one contiguous ArgbufRowsCols buffer at buffer(3).",
        },
        {
            "finding_id": "claude-abi-04-gemv-attn-arena",
            "preflight": {
                "severity": "ERROR",
                "check": "type_width",
                "kernel": "gemv_f32_attn",
                "host_site": "crates/hawking-core/src/kernels/mod.rs:8789",
            },
            "classification": "REAL_DEFECT",
            "status": "FIXED",
            "reachability": "mla_decode_and_o_proj_arena_metal is the arena-resident attention path.",
            "evidence": [
                _source_line(kernels, "let gemv_args = ArgbufRowsCols"),
                _source_line(attn, "constant ArgbufRowsCols& args [[buffer(3)]]"),
            ],
            "fix": "Applied the same single-buffer ArgbufRowsCols binding to the arena variant.",
        },
        {
            "finding_id": "claude-abi-05-moe-silu",
            "preflight": {
                "severity": "ERROR",
                "check": "type_width",
                "kernel": "moe_batched_silu_mul",
                "host_site": "crates/hawking-core/src/kernels/mod.rs:9166",
            },
            "classification": "REAL_DEFECT",
            "status": "FIXED",
            "reachability": "moe_block_batched_indexed_metal reaches encode_silu_mul.",
            "evidence": [
                _source_line(kernels, "let args = ArgbufN { n: n_u32 }") ,
                _source_line(moe, "constant ArgbufN& args   [[buffer(3)]]"),
            ],
            "fix": "Packed n into one ArgbufN constant buffer at buffer(3).",
        },
        {
            "finding_id": "claude-abi-06-moe-route-accumulate",
            "preflight": {
                "severity": "ERROR",
                "check": "type_width",
                "kernel": "moe_route_accumulate",
                "host_site": "crates/hawking-core/src/kernels/mod.rs:9193",
            },
            "classification": "REAL_DEFECT",
            "status": "FIXED",
            "reachability": "The non-TCB MoE route accumulation helper is directly callable by the MoE host path.",
            "evidence": [
                _source_line(kernels, "let args = ArgbufRouteAcc"),
                _source_line(moe, "constant ArgbufRouteAcc& args   [[buffer(4)]]"),
            ],
            "fix": "Packed hidden, routes, and has_shared into one ArgbufRouteAcc at buffer(4).",
        },
    ]

    # The source checks above are deliberately followed by current verifier
    # state, so a later partial revert cannot leave a green feedback receipt.
    current_generated = static.get("generated_kernel_names", {})
    target = "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128"
    if target not in current_generated:
        raise RuntimeError("current static verifier did not record the generated Qwen K=1 name")

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": "Claude sidecar six-row static ABI adjudication and verifier feedback.",
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": (
            "Source adjudication only. No Metal device, timing, throughput, BPW, or "
            "physical qualification claim is present."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "source_preflight": {
            "receipt": "receipts/future/STATIC_KERNEL_PREFLIGHT.json",
            "current_blocking_error_count": len(static_errors),
            "current_would_waste_a_protected_window": static.get(
                "would_waste_a_protected_window"
            ),
        },
        "outcomes": outcomes,
        "summary": {
            "reported_findings": 6,
            "real_defects_fixed": 5,
            "blocked_by_generation_resolved": 1,
            "dead_paths": 0,
            "intentional_aliases": 0,
            "outstanding_confirmed_defects": 0,
        },
        "verifier_feedback": {
            "kernel_existence": (
                "Recognize token-pasted entry points only from explicit macro invocation "
                "and preserve generated-body ABI as a compiler/runtime boundary."
            ),
            "constant_struct_abi": (
                "A Metal constant struct at buffer(N) requires one contiguous host "
                "binding with the matching layout; adjacent set_u32 calls at N/N+1 "
                "are not equivalent."
            ),
            "physical_gate": (
                "Static resolution clears a protected-window correctness blocker; it "
                "does not promote Flash or produce qualified physical EBPW."
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/claude_abi_adjudication.py")


def main() -> int:
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
