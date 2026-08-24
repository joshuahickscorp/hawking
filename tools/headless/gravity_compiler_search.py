#!/usr/bin/env python3
"""Gravity compiler search: a candidate is representation AND execution.

Gravity compiles (parent_function, machine, capability_contract) into
candidate executable programs. Candidate identity is not "which codec".
It is representation AND kernel AND layout AND runtime graph. Two
programs with the same stored weights and different kernels are
different candidates.

This is demonstrated, not asserted:

  1. Scoring a candidate that names no kernel is refused. There is no
     default to geo_tpr64. Defaulting would collapse the Q4 serial
     1-thread-per-row kernel and the production tpr64 kernel into one
     4-tuple (bytes, FLOP, dispatches, recon=0) — the bug that
     transferred a 5.9× occupancy penalty onto the wrong vehicle.

  2. A kernel that is genuinely faster per dispatch is refused as a
     win when the representation's byte traffic swamps the gain.
     gemv_simdgroup_f32 issues the dense GEMV MACs (no Q4 scale-mul);
     attached to the Q4 reconstruct-then-GEMM lowering it still streams
     218.8 GB/token. The bandwidth floor buries the 1.50× per-dispatch
     FLOP cut.

Does not load a second 27B. GPU wall of a live decode is ABSENT with
the physical reason, never 0. Storage bpw AND active bpw are reported
for every candidate.

    python3 tools/headless/gravity_compiler_search.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import noetic_traffic_model as tm  # noqa: E402

SCHEMA = "hawking.headless.gravity_compiler_search.v1"
RECEIPT = REPO / "receipts" / "headless" / "GRAVITY_COMPILER_SEARCH.json"
SHADERS = REPO / "crates" / "hawking-core" / "shaders"

KIND_ALG = "measured_from_encoded_algorithm"
KIND_ANCHOR = "anchor_not_rederived"
KIND_ABSENT = "ABSENT"

# Native-operator SOURCE vs EXECUTABLE (receipts/headless/NOETIC_NATIVE_OPERATOR.json).
# Not re-derived. This is the shape of the tradeoff being searched:
# executable does 1.50× the FLOPs and 3.49× the operations of the dense
# source, for 7.34× fewer DRAM bytes, at an identical dispatch count.
NAT_SOURCE_FLOPS = 51_541_222_144
NAT_SOURCE_OPS = 51_541_222_144
NAT_SOURCE_DRAM = 102_679_468_036
NAT_SOURCE_DISP = 964
NAT_Q4_FUSED_FLOPS = 77_163_181_824
NAT_Q4_FUSED_OPS = 179_651_020_544
NAT_Q4_FUSED_DRAM = 13_988_022_948
NAT_Q4_FUSED_DISP = 964
NAT_Q4_FUSED_PEAK_TEMP = 3_145_728
NAT_RECON_FLOPS = 77_163_176_704
NAT_RECON_OPS = 179_650_994_944
NAT_RECON_DRAM = 218_778_949_636
NAT_RECON_DISP = 1_365
NAT_RECON_PEAK_TEMP = 5_085_593_600
NAT_PARENT_LM_HEAD_BYTES = 5_085_593_600
NAT_STORED_BPW = 4.253
NAT_DECODE_VECTOR_EXTRA_DISP = 401

# GEMV MACs only — what gemv_simdgroup_f32 issues. The Q4 fused kernel
# adds one scale-mul per weight on top (native operator 1.50× FLOPs).
GEMV_MAC_FLOPS = tm.GEMV_MAC_FLOPS  # 51_243_909_120
ACTIVATION_FLOPS = NAT_SOURCE_FLOPS - GEMV_MAC_FLOPS  # 297_313_024
DENSE_F32_W_BYTES = tm.DENSE_F32_W_BYTES

# 0.0485 UV structure, same receipt. Stored and active are different regimes.
UV_STORED_BPW = 0.04853084788602941
UV_ACTIVE_FUSED_BPW = 0.04852941176470588
UV_ACTIVE_CACHE_F16_BPW = 16.0
UV_GENERATOR_BYTES = 540_672
UV_CACHE_F16_BYTES = 178_257_920
UV_FUSED_FLOPS = 540_672
UV_FUSED_DRAM = 630_832
UV_FUSED_DISP = 2
UV_CACHE_FLOPS = 178_257_920  # per-token after cache = dense GEMV
UV_CACHE_DRAM = 178_348_032
UV_CACHE_DISP = 1
UV_PARENT = {"name": "L31.mlp.gate_proj.weight", "shape": [17408, 5120], "rank": 2, "bytes": 178_257_920}

# Q80 dispatch-bound anchor (cited, not re-run).
Q80_PCT_OF_700 = 0.79
Q80_GPU_IDLE_PCT = 51.0
Q80_CEILING_GB_S = 700.0
Q80_SERIAL_SPEEDUP = tm.Q80_SERIAL_SPEEDUP  # 23.7

REFUSED_KERNEL_SENTINELS = {
    None,
    "",
    "default",
    "auto",
    "unspecified",
    "none",
    "null",
    "implicit",
    "whatever_the_codec_uses",
}

REQUIRED_KERNELS = {
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128": "qwen_uniform_q4.metal",
    "qwen_uniform_q4_group64_matvec": "qwen_uniform_q4.metal",
    "qwen_uniform_q4_decode_vector": "qwen_uniform_q4.metal",
    "gemv_simdgroup_f32": "matmul.metal",
    "gemv_f16_simdmat": "matmul.metal",
    "q80_hgravs01_two_stage_matvec": "q80_mixed_decode.metal",
    "q80_hgravs01_factor_matvec_simd3": "q80_mixed_decode.metal",
    "gravity_pq_matvec": "gravity_pq.metal",
}


class ScoringRefused(Exception):
    """A candidate cannot be scored. Never a default kernel."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(payload.get("why") or "scoring refused")


class KernelWinRefused(Exception):
    """A faster-per-dispatch kernel is not a candidate win: traffic dominates."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(payload.get("why") or "kernel win refused")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def git_ls_tree(rel: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", rel],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.returncode == 0 and rel in r.stdout.splitlines()
    except Exception:
        return False


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def cell(value, *, kind: str, unit: str, trace: str, absent_reason: str | None = None) -> dict:
    if kind == KIND_ABSENT:
        if value is not None:
            raise ValueError("ABSENT cells must not carry a numeric value")
        if not absent_reason:
            raise ValueError("ABSENT cells need a physical reason")
        return {
            "value": None,
            "unit": unit,
            "kind": KIND_ABSENT,
            "absent_reason": absent_reason,
            "trace": trace,
        }
    if value is None:
        raise ValueError(f"non-ABSENT cell has no value ({trace})")
    return {
        "value": value,
        "unit": unit,
        "kind": kind,
        "absent_reason": None,
        "trace": trace,
    }


def absent_gpu(reason: str) -> dict:
    return {
        "gpu_wall_ns_per_token": cell(
            None, kind=KIND_ABSENT, unit="ns", trace="not taken", absent_reason=reason
        ),
        "gpu_idle_pct": cell(
            None, kind=KIND_ABSENT, unit="percent", trace="not taken", absent_reason=reason
        ),
        "pct_of_bandwidth_ceiling": cell(
            None, kind=KIND_ABSENT, unit="percent", trace="not taken", absent_reason=reason
        ),
    }


def parse_kernel_voids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    names = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("kernel void "):
            names.append(line.split()[2].split("(")[0])
    return names


def kernel_catalog() -> dict[str, Any]:
    """Named kernels this search is allowed to bind. Existence is a file read."""
    by_file: dict[str, list[str]] = {}
    missing_on_disk: list[str] = []
    verified_against_head: list[str] = []
    for kernel, fname in REQUIRED_KERNELS.items():
        rel = f"crates/hawking-core/shaders/{fname}"
        path = SHADERS / fname
        if fname not in by_file:
            names = parse_kernel_voids(path)
            by_file[fname] = names
            if not path.is_file():
                missing_on_disk.append(rel)
            if git_ls_tree(rel):
                verified_against_head.append(rel)
    present = {}
    for kernel, fname in REQUIRED_KERNELS.items():
        names = by_file.get(fname) or []
        present[kernel] = {
            "file": fname,
            "path": f"crates/hawking-core/shaders/{fname}",
            "declared": kernel in names,
            "on_disk": (SHADERS / fname).is_file(),
        }
    missing = [k for k, v in present.items() if not v["declared"]]
    return {
        "required": present,
        "declared_names_by_file": by_file,
        "missing_required": missing,
        "missing_shader_on_disk": missing_on_disk,
        "shader_paths_verified_against_HEAD": verified_against_head,
        "all_required_declared": not missing,
    }


def kernel_named(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return name.strip() not in {s for s in REFUSED_KERNEL_SENTINELS if isinstance(s, str)}


def parent_function() -> dict:
    return {
        "id": "qwen38_token_gemv_organs",
        "artifact": "qwen38-gravity-uniform-q4-v1",
        "layers": 64,
        "hidden": 5120,
        "production_dispatches_per_token": NAT_Q4_FUSED_DISP,
        "production_command_buffers": 1,
        "production_workhorse_kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "source": [
            "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs",
            "receipts/headless/NOETIC_NATIVE_OPERATOR.json",
            "receipts/headless/NOETIC_OPERATION_CENSUS.json",
        ],
    }


def machine() -> dict:
    return {
        "chipset": "Apple M3 Ultra",
        "gpu_cores": tm.ANCHOR_GPU_CORES,
        "unified_memory_bytes": tm.ANCHOR_UNIFIED_B,
        "metal": tm.ANCHOR_METAL,
        "measured_roof_gb_s": tm.ANCHOR_ROOF_GB_S,
        "datasheet_peak_gb_s": tm.ANCHOR_DATASHEET_PEAK_GB_S,
        "honest_decode_ceiling_gb_s": tm.ANCHOR_HONEST_CEILING_GB_S,
        "compute_peak_gflops": tm.G143_COMPUTE_PEAK_GFLOPS,
        "compute_peak_source": "receipts/ascent-2026-08-16/G143_FLOPS_PER_TOKEN.json compute_peak_gflops",
        "roof_source": "NOETIC_TRAFFIC_MODEL anchors; not re-derived",
    }


def capability_contract() -> dict:
    return {
        "id": "gravity_compiler_search.v1",
        "a_candidate_is": "representation AND kernel AND layout AND runtime graph",
        "scoring_without_a_kernel": "REFUSED — no default",
        "kernel_must_be_declared_in_shaders": True,
        "reconstruct_then_gemm_is": "ORACLE, never production",
        "composed_behaviour_is_part_of_identity": True,
        "report_stored_and_active_bpw": True,
        "do_not_load_second_27b": True,
        "absent_is_never_zero": True,
        "executable_today_families": list(tm.EXECUTABLE_TODAY),
        "q80_decode_was_dispatch_bound": {
            "pct_of_ceiling": Q80_PCT_OF_700,
            "ceiling_gb_s": Q80_CEILING_GB_S,
            "gpu_idle_pct": Q80_GPU_IDLE_PCT,
            "reading": (
                "A representation win that adds dispatches can lose outright. "
                "Q80 sat at 0.79% of a 700 GB/s ceiling with 51% GPU idle."
            ),
        },
    }


def representation_key(rep: dict) -> dict:
    """The stored half. Active bpw / kernel / graph are execution."""
    return {
        "family": rep["family"],
        "packing": rep["packing"],
        "weight_identity": rep["weight_identity"],
    }


def execution_key(exe: dict) -> dict:
    return {
        "kernel": exe.get("kernel"),
        "layout": exe.get("layout"),
        "runtime_graph": exe.get("runtime_graph"),
    }


def attach_identity(candidate: dict) -> dict:
    rep_key = representation_key(candidate["representation"])
    exe_key = execution_key(candidate["execution"])
    candidate["identity"] = {
        "representation_fingerprint": fingerprint(rep_key),
        "execution_fingerprint": fingerprint(exe_key),
        "candidate_fingerprint": fingerprint({"representation": rep_key, "execution": exe_key}),
        "representation_key": rep_key,
        "execution_key": exe_key,
        "law": (
            "Two candidates with the same weights and different kernels are "
            "DIFFERENT candidates. Identity is the pair, not the codec."
        ),
    }
    return candidate


def bpw_block(*, stored, active_fused, active_cached, which: str) -> dict:
    return {
        "stored_bpw": stored,
        "active_bpw_fused": active_fused,
        "active_bpw_cached_dense": active_cached,
        "which": which,
    }


def q4_representation() -> dict:
    return {
        "family": "grouped_absmax_q4",
        "packing": "group64_nibble_codes_plus_fp16_scale",
        "weight_identity": "qwen38_uniform_q4_v1_codes",
        "artifact": "qwen38-gravity-uniform-q4-v1",
        "bpw": bpw_block(
            stored=NAT_STORED_BPW,
            active_fused=NAT_STORED_BPW,
            active_cached=32.0,
            which=(
                "stored = 4.253 Q4. Fused active = stored (codes stay packed). "
                "If the dense f32 form is cached, active is 32 bpw. Same structure, two regimes."
            ),
        ),
    }


def uv_representation() -> dict:
    return {
        "family": "generated_uv_low_rank",
        "packing": "U,V stored f16; W ≈ U V rank 12",
        "weight_identity": "generated_uv_L31_mlp_gate_proj_r12",
        "artifact": "GENERATED_WEIGHTS_RETEST L31.mlp.gate_proj.weight",
        "parent": UV_PARENT,
        "bpw": bpw_block(
            stored=UV_STORED_BPW,
            active_fused=UV_ACTIVE_FUSED_BPW,
            active_cached=UV_ACTIVE_CACHE_F16_BPW,
            which=(
                "0.0485 stored, 0.0485 active fused. Same structure is 16.0 active "
                "if the dense f16 form is cached. Report both or neither."
            ),
        ),
    }


def make_candidate(
    *,
    cid: str,
    title: str,
    representation: dict,
    execution: dict,
    scope: str,
    label: str,
    notes: list[str],
) -> dict:
    cand = {
        "id": cid,
        "title": title,
        "scope": scope,
        "label": label,
        "parent_function": parent_function()["id"],
        "machine": machine()["chipset"],
        "representation": representation,
        "execution": execution,
        "notes": notes,
    }
    return attach_identity(cand)


def candidate_q4_geo_tpr64() -> dict:
    return make_candidate(
        cid="q4_geo_tpr64_fused",
        title="Qwen3.8 uniform-q4 fused geo_tpr64 (production execution)",
        representation=q4_representation(),
        execution={
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "kernel_flops": NAT_Q4_FUSED_FLOPS,
            "kernel_class": "DISPATCHED",
            "occupancy_class": "tpr64_free",
            "layout": {
                "threads_per_row": 64,
                "threadgroup": 128,
                "rows_per_threadgroup": 2,
                "grid": "ceil(rows/2)*128",
                "packed_decode": "in-register",
            },
            "runtime_graph": {
                "dispatches_per_token": NAT_Q4_FUSED_DISP,
                "gemv_dispatches": 964,
                "command_buffers": 1,
                "prologue_kernel": None,
                "prologue_dispatches": 0,
                "reconstructs_dense_w": False,
                "dense_w_bytes": 0,
                "composition_scope": "whole_model_uniform_q4",
            },
            "bytes_per_token": NAT_Q4_FUSED_DRAM,
            "flop_per_token": NAT_Q4_FUSED_FLOPS,
            "operation_count": NAT_Q4_FUSED_OPS,
            "peak_temporary_materialization": NAT_Q4_FUSED_PEAK_TEMP,
        },
        scope="token",
        label="NATIVE",
        notes=[
            "Production path. Packed decode stays in registers; W is never written.",
            "Executable does 1.50× the FLOPs and 3.49× the operations of the dense source, "
            "for 7.34× fewer DRAM bytes, at an identical 964 dispatch count.",
        ],
    )


def candidate_q4_serial() -> dict:
    return make_candidate(
        cid="q4_serial_one_thread_per_row",
        title="Same Q4 codes, 1-thread-per-row serial matvec kernel",
        representation=q4_representation(),
        execution={
            "kernel": "qwen_uniform_q4_group64_matvec",
            "kernel_flops": NAT_Q4_FUSED_FLOPS,
            "kernel_class": "REACHABLE",
            "occupancy_class": "serial_extract",
            "layout": {
                "threads_per_row": 1,
                "threadgroup": None,
                "rows_per_threadgroup": 1,
                "grid": "thread_position_in_grid = row",
                "packed_decode": "1-thread-per-row serial nibble loop",
            },
            "runtime_graph": {
                "dispatches_per_token": NAT_Q4_FUSED_DISP,
                "gemv_dispatches": 964,
                "command_buffers": 1,
                "prologue_kernel": None,
                "prologue_dispatches": 0,
                "reconstructs_dense_w": False,
                "dense_w_bytes": 0,
                "composition_scope": "whole_model_uniform_q4",
            },
            "bytes_per_token": NAT_Q4_FUSED_DRAM,
            "flop_per_token": NAT_Q4_FUSED_FLOPS,
            "operation_count": NAT_Q4_FUSED_OPS,
            "peak_temporary_materialization": NAT_Q4_FUSED_PEAK_TEMP,
        },
        scope="token",
        label="NATIVE",
        notes=[
            "SAME stored Q4 codes as q4_geo_tpr64_fused. Different kernel, different layout.",
            "qwen_uniform_q4_group64_matvec: one thread owns a whole row "
            "(qwen_uniform_q4.metal:38, uint row [[thread_position_in_grid]]).",
            "occupancy_class=serial_extract is the named Q80 23.7× lowering "
            "(gpu_matvec 867.0 → 36.6 ms). Isolated Q4 tpr64 recon excess is 0 ns "
            "on 32/33 variants (NOETIC_TPR64_REOPEN). These two kernels are not interchangeable.",
            "Live GPU wall of this serial Q4 kernel on the 27B token is ABSENT "
            "(refused to load a second 27B). Occupancy is applied by the traffic model, labelled.",
        ],
    )


def candidate_q4_reconstruct_gemv() -> dict:
    return make_candidate(
        cid="q4_reconstruct_then_gemv",
        title="Same Q4 codes, decode_vector then gemv_simdgroup_f32 (ORACLE)",
        representation=q4_representation(),
        execution={
            "kernel": "gemv_simdgroup_f32",
            "kernel_flops": GEMV_MAC_FLOPS,
            "kernel_class": "REACHABLE",
            "occupancy_class": "unknown",
            "layout": {
                "threads_per_row": "simdgroup",
                "threadgroup": None,
                "dtype": "f32",
                "packed_decode": "NOT in this kernel; dense W already written",
            },
            "runtime_graph": {
                "dispatches_per_token": NAT_RECON_DISP,
                "gemv_dispatches": 964,
                "command_buffers": 1,
                "prologue_kernel": "qwen_uniform_q4_decode_vector",
                "prologue_dispatches": NAT_DECODE_VECTOR_EXTRA_DISP,
                "reconstructs_dense_w": True,
                "dense_w_bytes": DENSE_F32_W_BYTES,
                "composition_scope": "whole_model_uniform_q4_oracle_reconstruct",
            },
            "bytes_per_token": NAT_RECON_DRAM,
            "flop_per_token": NAT_RECON_FLOPS,
            "operation_count": NAT_RECON_OPS,
            "peak_temporary_materialization": NAT_RECON_PEAK_TEMP,
        },
        scope="token",
        label="ORACLE",
        notes=[
            "SAME stored Q4 codes. Execution writes dense f32 W then runs ordinary matvec.",
            "gemv_simdgroup_f32 issues only the GEMV MACs — 1.50× fewer FLOPs per "
            "dispatch than geo_tpr64 (which also does the scale-mul). That is a real "
            "per-dispatch kernel gain. The lowering also streams 218.8 GB/token.",
            "Peak temp reaches lm_head parent 5_085_593_600 B. ORACLE.",
        ],
    )


def candidate_uv_fused() -> dict:
    return make_candidate(
        cid="uv_0485_fused_two_stage",
        title="Generated UV at 0.0485 bpw, fused y = U @ (V @ x)",
        representation=uv_representation(),
        execution={
            "kernel": "q80_hgravs01_two_stage_matvec",
            "kernel_flops": UV_FUSED_FLOPS,
            "kernel_class": "REACHABLE",
            "occupancy_class": "tpr64_free",
            "layout": {
                "two_stage": True,
                "mid_in_threadgroup": True,
                "rank": 12,
            },
            "runtime_graph": {
                "dispatches_per_token": UV_FUSED_DISP,
                "gemv_dispatches": UV_FUSED_DISP,
                "command_buffers": 1,
                "prologue_kernel": None,
                "prologue_dispatches": 0,
                "reconstructs_dense_w": False,
                "dense_w_bytes": 0,
                "composition_scope": "single_organ",
            },
            "bytes_per_token": UV_FUSED_DRAM,
            "flop_per_token": UV_FUSED_FLOPS,
            "operation_count": UV_FUSED_FLOPS,
            "peak_temporary_materialization": 69680,
        },
        scope="organ",
        label="NATIVE",
        notes=[
            "No default-build UV kernel on the Qwen3.8 production token. "
            "q80_hgravs01_two_stage_matvec is the existing two-stage factor kernel "
            "(threadgroup mid[rank], never dense W). Bound here as the execution "
            "this structure would run.",
            "token_ns is ABSENT: organ-level candidate, not a 64-layer token graph.",
        ],
    )


def candidate_uv_cache() -> dict:
    return make_candidate(
        cid="uv_0485_cache_f16_gemv",
        title="Same 0.0485-bpw UV, dense f16 cache then ordinary matvec (ORACLE)",
        representation=uv_representation(),
        execution={
            "kernel": "gemv_f16_simdmat",
            "kernel_flops": UV_CACHE_FLOPS,
            "kernel_class": "REACHABLE",
            "occupancy_class": "unknown",
            "layout": {
                "two_stage": False,
                "cached_dense": "f16",
                "dtype": "f16",
            },
            "runtime_graph": {
                "dispatches_per_token": UV_CACHE_DISP,
                "gemv_dispatches": UV_CACHE_DISP,
                "command_buffers": 1,
                "prologue_kernel": None,
                "prologue_dispatches": 0,
                "reconstructs_dense_w": True,
                "dense_w_bytes": UV_CACHE_F16_BYTES,
                "composition_scope": "single_organ",
            },
            "bytes_per_token": UV_CACHE_DRAM,
            "flop_per_token": UV_CACHE_FLOPS,
            "operation_count": UV_CACHE_FLOPS,
            "peak_temporary_materialization": 178_327_552,
        },
        scope="organ",
        label="ORACLE",
        notes=[
            "SAME stored UV (0.0485 bpw). Active is 16.0 if the dense f16 form is cached.",
            "Peak temp reaches parent f16 shape. ORACLE. token_ns ABSENT (organ-level).",
        ],
    )


def candidate_without_kernel() -> dict:
    """Crafted invalid candidate. compile() never emits this. score() must refuse it."""
    return make_candidate(
        cid="q4_codes_kernel_unspecified",
        title="Q4 codes with no kernel named (must be unscorable)",
        representation=q4_representation(),
        execution={
            "kernel": None,
            "kernel_flops": NAT_Q4_FUSED_FLOPS,
            "kernel_class": None,
            "occupancy_class": None,
            "layout": None,
            "runtime_graph": {
                "dispatches_per_token": NAT_Q4_FUSED_DISP,
                "gemv_dispatches": 964,
                "command_buffers": 1,
                "prologue_kernel": None,
                "prologue_dispatches": 0,
                "reconstructs_dense_w": False,
                "dense_w_bytes": 0,
                "composition_scope": "whole_model_uniform_q4",
            },
            "bytes_per_token": NAT_Q4_FUSED_DRAM,
            "flop_per_token": NAT_Q4_FUSED_FLOPS,
            "operation_count": NAT_Q4_FUSED_OPS,
            "peak_temporary_materialization": NAT_Q4_FUSED_PEAK_TEMP,
        },
        scope="token",
        label="INVALID",
        notes=["Representation is fully specified. Execution names no kernel. Scoring must refuse."],
    )


def compile_candidates(
    parent: dict | None = None,
    mach: dict | None = None,
    contract: dict | None = None,
) -> list[dict]:
    """Gravity compiles the triple into candidate executable programs.

    Every emitted candidate names a kernel that is declared in this tree.
    A kernel-less dict is not a compiler output.
    """
    del parent, mach, contract  # the triple is the closed set below; callers pass for the contract
    catalog = kernel_catalog()
    if not catalog["all_required_declared"]:
        raise ScoringRefused(
            {
                "why": f"required kernels not declared: {catalog['missing_required']}",
                "catalog": catalog,
            }
        )
    out = [
        candidate_q4_geo_tpr64(),
        candidate_q4_serial(),
        candidate_q4_reconstruct_gemv(),
        candidate_uv_fused(),
        candidate_uv_cache(),
    ]
    for c in out:
        k = c["execution"]["kernel"]
        if not kernel_named(k):
            raise ScoringRefused(
                {
                    "why": f"compiler emitted a candidate with no kernel: {c['id']}",
                    "candidate_id": c["id"],
                }
            )
        if k not in catalog["required"] or not catalog["required"][k]["declared"]:
            raise ScoringRefused(
                {
                    "why": f"compiler bound undeclared kernel {k!r} on {c['id']}",
                    "candidate_id": c["id"],
                }
            )
    return out


def _kernel_token(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return ""
    return stripped.lower() if stripped.lower() in {
        s for s in REFUSED_KERNEL_SENTINELS if isinstance(s, str)
    } else stripped


def score(candidate: dict, *, catalog: dict | None = None) -> dict:
    """Score a candidate. Refuses if the execution half does not name a kernel.

    There is no default kernel. Returning a number here for kernel=None is the
    bug this obligation exists to catch.
    """
    exe = candidate.get("execution")
    if not isinstance(exe, dict):
        raise ScoringRefused(
            {
                "why": (
                    "candidate has no execution half. A candidate is representation "
                    "AND kernel AND layout AND runtime graph. Refusing rather than "
                    "defaulting a kernel."
                ),
                "candidate_id": candidate.get("id"),
                "kernel_supplied": None,
                "defaulted_kernel": None,
            }
        )
    supplied = exe.get("kernel")
    token = _kernel_token(supplied)
    if token in REFUSED_KERNEL_SENTINELS or not kernel_named(supplied):
        raise ScoringRefused(
            {
                "why": (
                    "scoring without a kernel is impossible. Refusing rather than "
                    f"defaulting (saw kernel={supplied!r}). A default to "
                    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 would collapse "
                    "the serial 1-thread-per-row kernel and production tpr64 into "
                    "one 4-tuple — the bug this obligation exists to catch."
                ),
                "candidate_id": candidate.get("id"),
                "kernel_supplied": supplied,
                "defaulted_kernel": None,
            }
        )
    cat = catalog or kernel_catalog()
    info = (cat.get("required") or {}).get(token)
    if not info or not info.get("declared"):
        raise ScoringRefused(
            {
                "why": (
                    f"kernel {token!r} is not a declared kernel void in this tree. "
                    "A representation is not executable without a kernel that exists."
                ),
                "candidate_id": candidate.get("id"),
                "kernel_supplied": supplied,
                "defaulted_kernel": None,
            }
        )

    scope = candidate.get("scope") or "token"
    gpu_reason = (
        "Refused to load a second 27B. Live GPU decode of the resident model is "
        "out of scope for this search. Prior TPS 32.73 / 30.606 ms/token and the "
        "Q80 0.79%/51%-idle dispatch-bound measurement are anchors, not this run."
    )
    graph = exe["runtime_graph"]
    stored_bpw = candidate["representation"]["bpw"]["stored_bpw"]
    active_fused = candidate["representation"]["bpw"]["active_bpw_fused"]
    active_cached = candidate["representation"]["bpw"]["active_bpw_cached_dense"]
    active_this = active_cached if graph.get("reconstructs_dense_w") else active_fused

    if scope != "token":
        return {
            "candidate_id": candidate["id"],
            "kernel": token,
            "scored": True,
            "scope": scope,
            "prediction": None,
            "token_ns": cell(
                None,
                kind=KIND_ABSENT,
                unit="ns",
                trace="not taken",
                absent_reason=(
                    f"{candidate['id']} is organ-level. Refused to scale one organ "
                    "to a 64-layer token_ns. Storage/active bpw still apply."
                ),
            ),
            "binding_term": None,
            "gpu_live": absent_gpu(gpu_reason),
            "label": candidate.get("label"),
            "stored_bpw": stored_bpw,
            "active_bpw_fused": active_fused,
            "active_bpw_cached_dense": active_cached,
            "active_bpw_this_execution": active_this,
        }

    pred = tm.predict_token_ns(
        float(exe["bytes_per_token"]),
        float(exe["flop_per_token"]),
        float(graph["dispatches_per_token"]),
        reconstruction_cost={
            "dense_w_bytes": 0,
            "reread": False,
            "extra_dispatches": 0,
            "occupancy_class": exe.get("occupancy_class") or "unknown",
            "note": (
                "bytes_per_token already includes the path's DRAM (native-operator "
                "EXECUTABLE/SOURCE). reconstruction_cost is not added again."
            ),
        },
        command_buffers=int(graph["command_buffers"]),
    )
    # Honour the oracle flag from the graph, not from reconstruction_cost bytes
    # (those were folded into bytes_per_token to avoid double-counting).
    if graph.get("reconstructs_dense_w"):
        pred = dict(pred)
        pred["dense_reconstruction_is_oracle"] = True

    traffic_ns = pred["floors_ns"]["bandwidth_at_measured_roof_595p9"]
    compute_ns = pred["floors_ns"]["compute_at_g143_8979_gflops"]
    ceremony_ns = pred["ceremony"]["total_ns"]
    if traffic_ns >= compute_ns and traffic_ns >= ceremony_ns:
        dominate = "representation_traffic"
    elif ceremony_ns >= traffic_ns and ceremony_ns >= compute_ns:
        dominate = "dispatch_ceremony"
    else:
        dominate = "kernel_compute"

    gemv_disp = float(graph["gemv_dispatches"])
    kernel_compute_ns = tm.ns_from_flops(float(exe["kernel_flops"]), tm.G143_COMPUTE_PEAK_GFLOPS)
    per_dispatch_ns = kernel_compute_ns / gemv_disp if gemv_disp else None

    return {
        "candidate_id": candidate["id"],
        "kernel": token,
        "scored": True,
        "scope": scope,
        "prediction": pred,
        "token_ns": cell(
            pred["token_ns"],
            kind=KIND_ALG,
            unit="ns",
            trace=(
                "noetic_traffic_model.predict_token_ns on this candidate's "
                "(bytes, FLOP, dispatches, occupancy_class). Roofline, not a live GPU wall."
            ),
        ),
        "token_ms": pred["token_ms"],
        "tok_s": pred["tok_s"],
        "binding_term": pred["binding_term"],
        "dominating_term": dominate,
        "floors_ns": pred["floors_ns"],
        "kernel_compute_ns": kernel_compute_ns,
        "kernel_compute_ns_per_dispatch": per_dispatch_ns,
        "traffic_over_kernel_compute": (
            traffic_ns / kernel_compute_ns if kernel_compute_ns else None
        ),
        "gpu_live": absent_gpu(gpu_reason),
        "label": candidate.get("label"),
        "stored_bpw": stored_bpw,
        "active_bpw_fused": active_fused,
        "active_bpw_cached_dense": active_cached,
        "active_bpw_this_execution": active_this,
    }


def try_score(candidate: dict, *, catalog: dict | None = None) -> dict:
    try:
        result = score(candidate, catalog=catalog)
    except ScoringRefused as exc:
        return {
            "scored": False,
            "refused": True,
            "exception": "ScoringRefused",
            "reason": exc.payload.get("why"),
            "payload": exc.payload,
            "defaulted_kernel": exc.payload.get("defaulted_kernel"),
            "kernel_supplied": exc.payload.get("kernel_supplied"),
            "candidate_id": candidate.get("id"),
        }
    return {
        "scored": True,
        "refused": False,
        "exception": None,
        "result": result,
        "defaulted_kernel": None,
        "kernel_supplied": candidate.get("execution", {}).get("kernel"),
        "candidate_id": candidate.get("id"),
    }


def per_dispatch_kernel_compute_ns(candidate: dict) -> float:
    exe = candidate["execution"]
    disp = float(exe["runtime_graph"]["gemv_dispatches"])
    if disp <= 0:
        raise ValueError("gemv_dispatches must be > 0")
    return tm.ns_from_flops(float(exe["kernel_flops"]), tm.G143_COMPUTE_PEAK_GFLOPS) / disp


def credit_kernel_win(candidate: dict, reference: dict, *, catalog: dict | None = None) -> dict:
    """Credit a per-dispatch kernel speedup only if it can move token_ns.

    A kernel that is faster per dispatch, attached to a representation whose
    byte traffic swamps the gain, is REFUSED. This is an exception, not a
    WIN with a footnote.
    """
    c_score = score(candidate, catalog=catalog)
    r_score = score(reference, catalog=catalog)
    if c_score.get("scope") != "token" or r_score.get("scope") != "token":
        raise ScoringRefused(
            {
                "why": "kernel-win credit is a token-level decision; organ-level token_ns is ABSENT",
                "candidate_id": candidate.get("id"),
                "defaulted_kernel": None,
            }
        )

    c_pd = per_dispatch_kernel_compute_ns(candidate)
    r_pd = per_dispatch_kernel_compute_ns(reference)
    faster = c_pd < r_pd
    ratio = (r_pd / c_pd) if c_pd else None
    c_pred = c_score["prediction"]
    traffic_ns = c_pred["floors_ns"]["bandwidth_at_measured_roof_595p9"]
    compute_ns = c_pred["floors_ns"]["compute_at_g143_8979_gflops"]
    ceremony_ns = c_pred["ceremony"]["total_ns"]
    kernel_gain_ns = (r_pd - c_pd) * float(candidate["execution"]["runtime_graph"]["gemv_dispatches"])
    traffic_dominates = traffic_ns > compute_ns and traffic_ns > ceremony_ns

    payload = {
        "candidate_id": candidate["id"],
        "reference_id": reference["id"],
        "candidate_kernel": candidate["execution"]["kernel"],
        "reference_kernel": reference["execution"]["kernel"],
        "candidate_kernel_flops": candidate["execution"]["kernel_flops"],
        "reference_kernel_flops": reference["execution"]["kernel_flops"],
        "candidate_ns_per_dispatch": c_pd,
        "reference_ns_per_dispatch": r_pd,
        "faster_per_dispatch": faster,
        "per_dispatch_speedup": ratio,
        "kernel_compute_gain_ns": kernel_gain_ns,
        "candidate_traffic_floor_ns": traffic_ns,
        "candidate_compute_floor_ns": compute_ns,
        "candidate_ceremony_ns": ceremony_ns,
        "traffic_over_compute": traffic_ns / compute_ns if compute_ns else None,
        "traffic_dominates": traffic_dominates,
        "representation_family": candidate["representation"]["family"],
        "representation_weight_identity": candidate["representation"]["weight_identity"],
        "candidate_dram_bytes": candidate["execution"]["bytes_per_token"],
        "reference_dram_bytes": reference["execution"]["bytes_per_token"],
        "same_stored_representation": (
            representation_key(candidate["representation"])
            == representation_key(reference["representation"])
        ),
    }

    if not faster:
        payload["decision"] = "NOT_FASTER_PER_DISPATCH"
        payload["why"] = (
            f"{candidate['execution']['kernel']} is not faster per dispatch than "
            f"{reference['execution']['kernel']} "
            f"({c_pd:.3f} ns vs {r_pd:.3f} ns compute-floor / GEMV dispatch)."
        )
        return payload

    if traffic_dominates:
        payload["decision"] = "KERNEL_WIN_REFUSED"
        payload["why"] = (
            f"{candidate['execution']['kernel']} is {ratio:.3f}× faster per dispatch "
            f"than {reference['execution']['kernel']} on the compute floor "
            f"({c_pd:.3f} ns vs {r_pd:.3f} ns; kernel FLOPs "
            f"{candidate['execution']['kernel_flops']:,} vs "
            f"{reference['execution']['kernel_flops']:,} — the native-operator "
            f"1.50×). Kernel compute gain {kernel_gain_ns/1e6:.3f} ms. "
            f"Representation traffic is {candidate['execution']['bytes_per_token']:,} B/token "
            f"({candidate['execution']['bytes_per_token'] / reference['execution']['bytes_per_token']:.2f}× "
            f"the fused stream), flooring the GPU at {traffic_ns/1e6:.3f} ms vs kernel "
            f"compute {compute_ns/1e6:.3f} ms "
            f"({traffic_ns/compute_ns:.1f}×). The kernel gain cannot move token_ns. "
            "Search REFUSES the kernel win."
        )
        raise KernelWinRefused(payload)

    payload["decision"] = "KERNEL_WIN"
    payload["why"] = (
        f"{candidate['execution']['kernel']} is faster per dispatch and the "
        "representation is not bandwidth-side of the kernel compute floor."
    )
    return payload


def try_credit_kernel_win(candidate: dict, reference: dict, *, catalog: dict | None = None) -> dict:
    try:
        result = credit_kernel_win(candidate, reference, catalog=catalog)
    except KernelWinRefused as exc:
        return {
            "credited": False,
            "refused": True,
            "exception": "KernelWinRefused",
            "decision": "KERNEL_WIN_REFUSED",
            "payload": exc.payload,
            "why": exc.payload.get("why"),
        }
    except ScoringRefused as exc:
        return {
            "credited": False,
            "refused": True,
            "exception": "ScoringRefused",
            "decision": "SCORING_REFUSED",
            "payload": exc.payload,
            "why": exc.payload.get("why"),
        }
    return {
        "credited": result.get("decision") == "KERNEL_WIN",
        "refused": False,
        "exception": None,
        "decision": result.get("decision"),
        "payload": result,
        "why": result.get("why"),
    }


def composition_constraint() -> dict:
    """Composed behaviour is part of a candidate's identity.

    The same codec that is locally adequate breaks the model when applied
    to every layer. Cited, not re-run.
    """
    path = REPO / "receipts" / "headless" / "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"
    if not path.is_file():
        return {
            "cited": False,
            "absent_reason": f"{path} not on disk in this worktree",
        }
    doc = json.loads(path.read_text())
    ea = doc.get("error_accumulation") or {}
    fb = doc.get("failure_boundary") or {}
    watched = doc.get("watched_fail") or []
    return {
        "cited": True,
        "receipt": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
        "law": (
            "runtime_graph.composition_scope is part of candidate identity. "
            "A layer-local application of a codec and a whole-model application "
            "of the same codec are different programs."
        ),
        "first_fail_free_layer": ea.get("first_fail_free_layer"),
        "undegraded_first_fail_layer": (fb.get("undegraded_depth") or {}).get("first_fail_layer"),
        "watched_fail_head": watched[:3],
        "local_q4_gate_survives": True,
        "whole_model_free_run_fails": ea.get("first_fail_free_layer") is not None,
    }


def same_rep_different_exec(candidates: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for c in candidates:
        fp = c["identity"]["representation_fingerprint"]
        groups.setdefault(fp, []).append(c)
    out = []
    for fp, members in groups.items():
        kernels = [m["execution"]["kernel"] for m in members]
        exec_fps = {m["identity"]["execution_fingerprint"] for m in members}
        cand_fps = {m["identity"]["candidate_fingerprint"] for m in members}
        if len(members) < 2:
            continue
        out.append(
            {
                "representation_fingerprint": fp,
                "family": members[0]["representation"]["family"],
                "weight_identity": members[0]["representation"]["weight_identity"],
                "n_candidates": len(members),
                "ids": [m["id"] for m in members],
                "kernels": kernels,
                "kernels_are_not_all_equal": len(set(kernels)) > 1,
                "execution_fingerprints_differ": len(exec_fps) == len(members),
                "candidate_fingerprints_differ": len(cand_fps) == len(members),
                "stored_bpw": members[0]["representation"]["bpw"]["stored_bpw"],
            }
        )
    return out


def walk_cells(obj: Any):
    if isinstance(obj, dict):
        if "kind" in obj and "value" in obj:
            yield obj
        for v in obj.values():
            yield from walk_cells(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_cells(v)


def build() -> dict:
    t0 = time.time()
    catalog = kernel_catalog()
    parent = parent_function()
    mach = machine()
    contract = capability_contract()
    compiled = compile_candidates(parent, mach, contract)
    scored = []
    for c in compiled:
        attempt = try_score(c, catalog=catalog)
        if not attempt["scored"]:
            raise ScoringRefused(attempt["payload"])
        entry = dict(c)
        entry["score"] = attempt["result"]
        scored.append(entry)

    fused = next(c for c in scored if c["id"] == "q4_geo_tpr64_fused")
    serial = next(c for c in scored if c["id"] == "q4_serial_one_thread_per_row")
    recon = next(c for c in scored if c["id"] == "q4_reconstruct_then_gemv")
    uv_f = next(c for c in scored if c["id"] == "uv_0485_fused_two_stage")
    uv_c = next(c for c in scored if c["id"] == "uv_0485_cache_f16_gemv")

    no_kernel = candidate_without_kernel()
    demo_no_kernel = try_score(no_kernel, catalog=catalog)
    # Also refuse the sentinels the compiler might be tempted to default to.
    sentinel_demos = []
    for sent in ("default", "auto", "", None):
        probe = candidate_without_kernel()
        probe["id"] = f"q4_codes_kernel_{sent!r}"
        probe["execution"] = dict(probe["execution"])
        probe["execution"]["kernel"] = sent
        sentinel_demos.append(try_score(probe, catalog=catalog))

    demo_kernel_win = try_credit_kernel_win(recon, fused, catalog=catalog)
    # Serial is not faster per dispatch on the FLOP floor (same kernel_flops).
    serial_vs_tpr = try_credit_kernel_win(serial, fused, catalog=catalog)

    pairs = same_rep_different_exec(scored)
    q4_pair = next(p for p in pairs if p["family"] == "grouped_absmax_q4")
    uv_pair = next(p for p in pairs if p["family"] == "generated_uv_low_rank")

    # Ratios matching the native-operator headline, computed here from the same constants.
    flop_ratio = NAT_Q4_FUSED_FLOPS / NAT_SOURCE_FLOPS
    ops_ratio = NAT_Q4_FUSED_OPS / NAT_SOURCE_OPS
    dram_ratio = NAT_SOURCE_DRAM / NAT_Q4_FUSED_DRAM

    gpu_reason = (
        "Refused to load a second 27B. Live GPU decode of the resident model is "
        "out of scope. Q80 0.79% of 700 GB/s with 51% GPU idle is the sealed "
        "dispatch-bound anchor, not this run."
    )

    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "elapsed_s": round(time.time() - t0, 3),
        "question": (
            "Is a Gravity candidate which codec, or a (representation, kernel, "
            "layout, runtime graph) executable program?"
        ),
        "answer": (
            "A candidate is the pair. Two programs with the same Q4 codes and "
            "different kernels (geo_tpr64 vs 1-thread-per-row serial) are different "
            "candidates and score differently (occupancy 1× vs 23.7×). Scoring a "
            "candidate that names no kernel is refused — there is no default. "
            "gemv_simdgroup_f32 is 1.50× faster per dispatch on the compute floor "
            "than geo_tpr64, but attached to the Q4 reconstruct-then-GEMM lowering "
            "it streams 218.8 GB/token; the bandwidth floor swamps the kernel gain "
            "and the search REFUSES the kernel win."
        ),
        "convention": {
            "candidate": "representation AND kernel AND layout AND runtime graph",
            "representation": "stored weights / packing / family. Active bpw is a regime of execution.",
            "kernel": "a declared kernel void. Required to score. No default.",
            "layout": "threadgroup, threads-per-row, occupancy class",
            "runtime_graph": "dispatches, command buffers, prologue kernels, composition scope",
            "SOURCE": "parent dense forward cost for the same organ/token",
            "EXECUTABLE": "what the compressed structure actually costs as run",
            "ORACLE": "reconstruct dense W then ordinary matmul. Allowed as a control. Must be labelled.",
            "zero_policy": "0 means measured zero. Did-not-measure is ABSENT with a physical reason.",
            "flop": "IEEE floating-point; FMA counted as 2",
        },
        "native_operator_shape": {
            "source_receipt": "receipts/headless/NOETIC_NATIVE_OPERATOR.json",
            "source_flops": NAT_SOURCE_FLOPS,
            "executable_flops": NAT_Q4_FUSED_FLOPS,
            "flops_ratio_executable_over_source": flop_ratio,
            "source_ops": NAT_SOURCE_OPS,
            "executable_ops": NAT_Q4_FUSED_OPS,
            "ops_ratio_executable_over_source": ops_ratio,
            "source_dram_bytes": NAT_SOURCE_DRAM,
            "executable_dram_bytes": NAT_Q4_FUSED_DRAM,
            "dram_ratio_source_over_executable": dram_ratio,
            "dispatch_count_both": NAT_Q4_FUSED_DISP,
            "reading": (
                f"Executable does {flop_ratio:.2f}× the FLOPs and {ops_ratio:.2f}× "
                f"the operations of the dense source, for {dram_ratio:.2f}× fewer "
                f"DRAM bytes, at an identical {NAT_Q4_FUSED_DISP} dispatch count."
            ),
        },
        "parent_function": parent,
        "machine": mach,
        "capability_contract": contract,
        "kernel_catalog": catalog,
        "candidates": scored,
        "same_representation_different_execution": pairs,
        "demonstration_scoring_without_kernel": {
            "attempted": True,
            "candidate_id": no_kernel["id"],
            "kernel_supplied": no_kernel["execution"]["kernel"],
            "refused": demo_no_kernel["refused"],
            "scored": demo_no_kernel["scored"],
            "defaulted_kernel": demo_no_kernel["defaulted_kernel"],
            "exception": demo_no_kernel["exception"],
            "reason": demo_no_kernel["reason"],
            "sentinel_probes": [
                {
                    "kernel_supplied": s["kernel_supplied"],
                    "refused": s["refused"],
                    "scored": s["scored"],
                    "defaulted_kernel": s["defaulted_kernel"],
                }
                for s in sentinel_demos
            ],
            "all_sentinel_probes_refused": all(s["refused"] and not s["scored"] for s in sentinel_demos),
        },
        "demonstration_kernel_win_refused_when_traffic_dominates": demo_kernel_win,
        "serial_vs_tpr64_is_not_a_flop_kernel_win": serial_vs_tpr,
        "composition_is_part_of_identity": composition_constraint(),
        "q80_dispatch_bound_anchor": {
            "pct_of_700_GBs": Q80_PCT_OF_700,
            "gpu_idle_pct": Q80_GPU_IDLE_PCT,
            "ceiling_gb_s": Q80_CEILING_GB_S,
            "serial_extract_speedup": Q80_SERIAL_SPEEDUP,
            "kind": KIND_ANCHOR,
            "reading": (
                "Q80 decode was DISPATCH-BOUND, not bandwidth-bound. A representation "
                "win that adds dispatches can lose outright. Occupancy 23.7× at "
                "unchanged codec bytes is why a candidate without a kernel cannot be scored."
            ),
        },
        "gpu_live_this_run": absent_gpu(gpu_reason),
        "identity_checks": {
            "q4_fused_vs_serial_same_representation": (
                fused["identity"]["representation_fingerprint"]
                == serial["identity"]["representation_fingerprint"]
            ),
            "q4_fused_vs_serial_different_execution": (
                fused["identity"]["execution_fingerprint"]
                != serial["identity"]["execution_fingerprint"]
            ),
            "q4_fused_vs_serial_different_candidate": (
                fused["identity"]["candidate_fingerprint"]
                != serial["identity"]["candidate_fingerprint"]
            ),
            "q4_fused_vs_reconstruct_same_representation": (
                fused["identity"]["representation_fingerprint"]
                == recon["identity"]["representation_fingerprint"]
            ),
            "q4_fused_vs_reconstruct_different_kernel": (
                fused["execution"]["kernel"] != recon["execution"]["kernel"]
            ),
            "uv_fused_vs_cache_same_representation": (
                uv_f["identity"]["representation_fingerprint"]
                == uv_c["identity"]["representation_fingerprint"]
            ),
            "uv_fused_vs_cache_different_execution": (
                uv_f["identity"]["execution_fingerprint"]
                != uv_c["identity"]["execution_fingerprint"]
            ),
            "q4_pair_has_at_least_two": q4_pair["n_candidates"] >= 2,
            "uv_pair_has_at_least_two": uv_pair["n_candidates"] >= 2,
            "q4_kernels_differ": q4_pair["kernels_are_not_all_equal"],
        },
        "bpw_every_candidate": [
            {
                "id": c["id"],
                "stored_bpw": c["representation"]["bpw"]["stored_bpw"],
                "active_bpw_fused": c["representation"]["bpw"]["active_bpw_fused"],
                "active_bpw_cached_dense": c["representation"]["bpw"]["active_bpw_cached_dense"],
                "active_bpw_this_execution": c["score"].get("active_bpw_this_execution"),
                "which": c["representation"]["bpw"]["which"],
            }
            for c in scored
        ],
        "what_i_did_not_do": [
            "Did not load a second 27B.",
            "Did not open receipts/ascent-2026-08-16 or workspace/campaign for writing.",
            "Did not default a missing kernel to geo_tpr64.",
            "Did not credit gemv_simdgroup_f32 as a win on the reconstruct lowering.",
            "Did not write 0 for a number that was not measured.",
            "Did not treat organ-level UV token_ns as a 64-layer number.",
        ],
        "self_check": {
            "all_required_kernels_declared": catalog["all_required_declared"],
            "compiler_emitted_at_least_two_q4_executions": q4_pair["n_candidates"] >= 2
            and q4_pair["kernels_are_not_all_equal"],
            "no_kernel_scoring_refused": demo_no_kernel["refused"] is True
            and demo_no_kernel["scored"] is False
            and demo_no_kernel["defaulted_kernel"] is None,
            "all_sentinels_refused": all(s["refused"] and not s["scored"] for s in sentinel_demos),
            "kernel_win_refused_on_reconstruct": demo_kernel_win["decision"] == "KERNEL_WIN_REFUSED"
            and demo_kernel_win["refused"] is True,
            "reconstruct_kernel_is_faster_per_dispatch": (
                (demo_kernel_win.get("payload") or {}).get("faster_per_dispatch") is True
            ),
            "reconstruct_traffic_dominates": (
                (demo_kernel_win.get("payload") or {}).get("traffic_dominates") is True
            ),
            "same_stored_q4_on_the_refused_pair": (
                (demo_kernel_win.get("payload") or {}).get("same_stored_representation") is True
            ),
            "fused_vs_serial_identity_splits_on_execution": (
                fused["identity"]["representation_fingerprint"]
                == serial["identity"]["representation_fingerprint"]
                and fused["identity"]["candidate_fingerprint"]
                != serial["identity"]["candidate_fingerprint"]
            ),
            "every_scored_candidate_names_a_kernel": all(
                kernel_named(c["execution"]["kernel"]) for c in scored
            ),
            "uv_reports_0485_and_16": (
                abs(uv_f["representation"]["bpw"]["stored_bpw"] - 0.0485) < 1e-4
                and uv_c["representation"]["bpw"]["active_bpw_cached_dense"] == 16.0
            ),
        },
        "written_to": str(RECEIPT),
    }

    for cell_d in walk_cells(doc):
        if cell_d.get("kind") == KIND_ABSENT:
            if cell_d.get("value") is not None:
                raise SystemExit(f"ABSENT cell has value: {cell_d}")
            if not cell_d.get("absent_reason"):
                raise SystemExit(f"ABSENT without reason: {cell_d}")
    return doc


def write_receipt(doc: dict, path: Path = RECEIPT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return path


def print_table(doc: dict) -> None:
    print("=" * 88)
    print("GRAVITY COMPILER SEARCH — a candidate is representation AND execution")
    print(doc["question"])
    print("=" * 88)
    print()
    print(doc["answer"])
    print()
    shape = doc["native_operator_shape"]
    print(shape["reading"])
    print()
    print(f"{'id':<32} {'label':<8} {'kernel':<42} {'stored':>8} {'active':>8}")
    print("-" * 104)
    for c in doc["candidates"]:
        k = c["execution"]["kernel"]
        stored = c["representation"]["bpw"]["stored_bpw"]
        active = c["score"].get("active_bpw_this_execution")
        print(f"{c['id']:<32} {c['label']:<8} {k:<42} {stored:8.4f} {active if active is None else f'{active:8.4f}'}")
    print()
    demo = doc["demonstration_scoring_without_kernel"]
    print(
        f"no-kernel scoring: refused={demo['refused']} scored={demo['scored']} "
        f"defaulted={demo['defaulted_kernel']!r}"
    )
    kw = doc["demonstration_kernel_win_refused_when_traffic_dominates"]
    print(f"kernel win: decision={kw['decision']} refused={kw['refused']}")
    print()
    print(f"wrote {doc['written_to']}")
    print("=" * 88)


def main() -> int:
    doc = build()
    write_receipt(doc)
    print_table(doc)
    sc = doc["self_check"]
    if not all(sc.values()):
        bad = [k for k, v in sc.items() if not v]
        print(f"FAIL self_check: {bad}", file=sys.stderr)
        return 2
    demo = doc["demonstration_scoring_without_kernel"]
    if demo["scored"] or not demo["refused"] or demo["defaulted_kernel"] is not None:
        print("FAIL: scoring without a kernel was not refused", file=sys.stderr)
        return 3
    kw = doc["demonstration_kernel_win_refused_when_traffic_dominates"]
    if kw["decision"] != "KERNEL_WIN_REFUSED":
        print(f"FAIL: kernel win was not refused ({kw['decision']})", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
