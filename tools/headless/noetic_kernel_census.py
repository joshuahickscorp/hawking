#!/usr/bin/env python3
"""Census of Metal kernels this machine can actually dispatch.

A new representation is only executable if a kernel can run it. This script
enumerates every `kernel void` declared under crates/hawking-core/shaders,
classifies each one, and records whether the kernels that already run on the
Qwen3.8 decode path reconstruct a dense weight tensor before computing.

Classification (reconciled against G071 / nx_genome.py 38 bound of 554 declared):

  DISPATCHED  string-literal in qwen38_hybrid_decode.rs ∩ declared kernel name
              (nx_genome.bound_kernels; a seal listing all 554 would be a lie)
  REACHABLE   compiled into a Metal library this tree can load, and referenced
              by a quoted identifier in Rust, but not in the decode-literal 38
  DEAD        declared, never referenced as a quoted identifier in Rust
  UNKNOWN     cannot classify without guessing (macro templates, missing compile)

Reconstructs-dense YES means the kernel materializes a (rows × cols) weight
matrix in device memory and would then need an ordinary matmul. That is an
oracle path: it proves correctness and proves nothing about compressed-structure
cost. In-register dequant + FMA into y is NO even when dequant FLOPs dominate
(G043 reconstruction_share ≈ 0.71 is that in-register ALU, not dense W).

  python3 tools/headless/noetic_kernel_census.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "hawking.headless.noetic_kernel_census.v1"
G071_BOUND = 38
G071_DECLARED = 554

REPO = Path(__file__).resolve().parents[2]
DECODE = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SHADERS = REPO / "crates/hawking-core/shaders"
METAL_MOD = REPO / "crates/hawking-core/src/metal/mod.rs"
RECEIPT = REPO / "receipts/headless/NOETIC_KERNEL_CENSUS.json"

# Geometry from crates/hawking-core/src/model/qwen38_geometry.rs (read, not guessed).
QWEN38_LAYERS = 64
QWEN38_DELTANET_LAYERS = 48
QWEN38_GQA_LAYERS = 16
QWEN38_HIDDEN = 5120
QWEN38_INTERMEDIATE = 17408
QWEN38_VOCAB = 248320
QWEN38_QKVZ_ROWS = 16384
QWEN38_BA_ROWS = 96
QWEN38_Q_PROJ_ROWS = 12288
QWEN38_KV_PROJ_ROWS = 1024
QWEN38_O_PROJ_ROWS = 5120
QWEN38_O_PROJ_COLS = 6144
UNIFORM_Q4_GROUP = 64
Q4_BYTES_PER_GROUP = UNIFORM_Q4_GROUP // 2 + 2  # 32 code + 2 scale

# Production token shape (qwen38_token_ns_ledger.rs).
PRODUCTION_DISPATCHES = 964
PRODUCTION_COMMAND_BUFFERS = 1

FAMILY_IDS = [
    "shared_basis_x_coefficients",
    "tensor_contraction",
    "fused_dictionary_lookup_accumulate",
    "low_rank_plus_sparse_correction",
    "structured_transform",
    "routed_group_execution",
    "recurrent_state_operator",
]


def q4_matrix_bytes(rows: int, cols: int, group: int = UNIFORM_Q4_GROUP) -> int:
    gpr = (cols + group - 1) // group
    return rows * gpr * (group // 2 + 2)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN:{exc}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_declared_kernels() -> list[dict]:
    """Same extractor as tools/nx_genome.py: line.startswith('kernel void ')."""
    out = []
    for path in sorted(SHADERS.glob("*.metal")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("kernel void "):
                name = line.split()[2].split("(")[0]
                start = i
                i += 1
                while i < len(lines) and not lines[i].startswith("kernel void "):
                    i += 1
                body = "\n".join(lines[start:i])
                out.append(
                    {
                        "name": name,
                        "file": path.name,
                        "path": f"crates/hawking-core/shaders/{path.name}",
                        "line": start + 1,
                        "n_lines": i - start,
                        "signature": line.strip(),
                        "body": body,
                    }
                )
                continue
            i += 1
    return out


def decode_string_literals(text: str) -> set[str]:
    """Same splitter as tools/nx_genome.py bound_kernels()."""
    lits = set()
    for tok in text.split('"'):
        if tok and all(c.isalnum() or c == "_" for c in tok):
            lits.add(tok)
    return lits


def is_macro_template(name: str) -> bool:
    return ("##" in name) or name == "NAME"


def rust_quoted_kernel_hits(names: set[str]) -> dict[str, dict]:
    """Quoted identifiers in crates/**/*.rs that match a declared kernel name."""
    hits: dict[str, dict] = {
        n: {"files": [], "kinds": Counter()} for n in names
    }
    rust_files = [
        p
        for p in (REPO / "crates").rglob("*.rs")
        if "target" not in p.parts
    ]
    ident_re = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')

    def kind_of(p: Path) -> str:
        s = str(p)
        if "/examples/" in s:
            return "examples"
        if "/tests/" in s:
            return "tests"
        if p.name == "mod.rs" and p.parent.name == "metal":
            return "metal_registry"
        if p.name == "qwen38_hybrid_decode.rs":
            return "qwen38_decode"
        return "src"

    for path in rust_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = set(ident_re.findall(text)) & names
        if not found:
            continue
        rel = str(path.relative_to(REPO))
        knd = kind_of(path)
        for name in found:
            hits[name]["files"].append(rel)
            hits[name]["kinds"][knd] += 1
    return hits


def compiled_shader_files() -> dict:
    """Where each .metal file is compiled from."""
    mod = METAL_MOD.read_text(encoding="utf-8", errors="replace")
    runtime = set(
        re.findall(r'include_str!\("\.\./\.\./shaders/([^"]+\.metal)"\)', mod)
    )
    tq_gated = "strand_bitslice.metal" in runtime
    example_include: dict[str, list[str]] = defaultdict(list)
    example_path: dict[str, list[str]] = defaultdict(list)
    for p in (REPO / "crates").rglob("*.rs"):
        if "target" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(REPO))
        for m in re.finditer(r'include_str!\("([^"]*shaders/([^"]+\.metal))"\)', text):
            example_include[m.group(2)].append(rel)
        for m in re.finditer(r'shaders/([A-Za-z0-9_.-]+\.metal)', text):
            example_path[m.group(1)].append(rel)
    return {
        "runtime_all_shader_sources": sorted(runtime),
        "tq_gated": sorted(["strand_bitslice.metal"] if tq_gated else []),
        "example_include_str": {k: sorted(set(v)) for k, v in example_include.items()},
        "example_path_mention": {k: sorted(set(v)) for k, v in example_path.items()},
    }


def compile_gate_for(fname: str, compiled: dict) -> str | None:
    runtime = set(compiled["runtime_all_shader_sources"])
    if fname == "strand_bitslice.metal":
        return "feature=tq"
    if fname in runtime:
        return None  # default library
    if fname in compiled["example_include_str"] or fname in compiled["example_path_mention"]:
        return "example_only"
    return "not_compiled"


# Hand-verified facts for the 38 DISPATCHED kernels. Evidence is quoted from
# the Metal/Rust source that was read; a missing name here becomes UNKNOWN
# rather than a guessed YES/NO.
DISPATCHED_FACTS: dict[str, dict] = {
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128": {
        "representation_family": "grouped_absmax_q4",
        "role": "production uniform-q4 GEMV (geometry-sweep winner, default Qwen38MatvecKernel)",
        "production_uniform_q4_default": True,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen_uniform_q4.metal:181-211: 'Packed decode stays in registers.' "
            "acc += qwen_uniform_q4_unpack8(packed, scale, input, col); writes only output[row]."
        ),
        "reads_per_token": "every Q4 GEMV on the 964-dispatch token (MLP 3×64, DeltaNet 3×48, GQA 4×16, lm_head 1). Packed codes + fp16 scales + x; never a dense W.",
    },
    "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128": {
        "representation_family": "grouped_absmax_q4",
        "role": "group-128 sibling of geo_tpr64; bound only when cols % 128 == 0",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen_uniform_q4.metal:223-271: same unpack8-into-acc as group-64; "
            "sealed uniform-q4-v1 is group-64 so this is not the production bind."
        ),
        "reads_per_token": "none on uniform-q4-v1 (group 64). Would read codes+scales+x for a group-128 organ.",
    },
    "qwen_uniform_q4_group64_matvec": {
        "representation_family": "grouped_absmax_q4",
        "role": "serial one-thread-per-row Q4 GEMV (association baseline, not default launch)",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen_uniform_q4.metal:38-67: nibble dequant in-register, sum += q * scale * input[col]; "
            "output[row] = sum."
        ),
        "reads_per_token": "alt launch of the same packed Q4 organ. Default session uses geo_tpr64_tg128.",
    },
    "qwen_uniform_q4_group64_matvec_vecgroup": {
        "representation_family": "grouped_absmax_q4",
        "role": "Qwen38MatvecKernel::Vecgroup retarget; same packed Q4, different TG",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "qwen38_hybrid_decode.rs:604-623 retargets launch geometry only; 'does not generate new shaders'.",
        "reads_per_token": "same bytes as geo_tpr64 if selected; default is GeoTpr64Tg128.",
    },
    "qwen_uniform_q4_group64_matvec_vecgroup_r4": {
        "representation_family": "grouped_absmax_q4",
        "role": "Qwen38MatvecKernel::VecgroupR4 retarget",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same packed Q4 consume-in-register family; retarget only (decode.rs:609-623).",
        "reads_per_token": "same bytes as geo_tpr64 if selected.",
    },
    "qwen_uniform_q4_group64_matvec_vecgroup_x64": {
        "representation_family": "grouped_absmax_q4",
        "role": "Qwen38MatvecKernel::VecgroupX64 retarget",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same packed Q4 consume-in-register family; retarget only (decode.rs:609-623).",
        "reads_per_token": "same bytes as geo_tpr64 if selected.",
    },
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe": {
        "representation_family": "diagnostic_probe",
        "role": "address+DRAM load probe; no nibble unpack, no x, no FMA",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": True,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen_uniform_q4.metal:274-317: 'only the addressing + DRAM load of scales and packed codes. "
            "... No nibble unpack, no input-vector load, no FMA.' Sinks into a scalar; does not write W."
        ),
        "reads_per_token": "not on the 964-dispatch token. Diagnostic traffic of codes+scales.",
    },
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe": {
        "representation_family": "diagnostic_probe",
        "role": "address+dequant ALU probe; still no x / FMA / dense W",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": True,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen_uniform_q4.metal:330-370: 'address + dequant, still no input-vector load / FMA. "
            "Difference vs addr_probe is the reconstruction ALU.' Reconstruction here is in-register "
            "nibble unpack into a scalar sink, not a dense tensor. G043's reconstruction_share is this ALU."
        ),
        "reads_per_token": "not on the 964-dispatch token.",
    },
    "qwen_uniform_q4_embedding_lookup": {
        "representation_family": "grouped_absmax_q4",
        "role": "production embedding gather: one packed row → hidden f32",
        "production_uniform_q4_default": True,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen_uniform_q4.metal:639-653: 'Direct packed Q4 embedding lookup — no host f32 embedding table.' "
            "Writes output[id] for hidden elements of one token, not a vocab×hidden W."
        ),
        "reads_per_token": f"one gathered row: {q4_matrix_bytes(1, QWEN38_HIDDEN)} packed bytes (codes+fp16 scales) of embed; full table {q4_matrix_bytes(QWEN38_VOCAB, QWEN38_HIDDEN)} B stays resident and is not streamed.",
    },
    "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128": {
        "representation_family": "grouped_absmax_q3",
        "role": "HGRAVU01 bits=3 geo_tpr64; mixed/recon-fuse factor path",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "q80_mixed_decode.metal:1-5 contract: 'packed bytes are read directly. A value is decoded "
            "in registers and consumed in the matvec in the same kernel. These kernels must never write "
            "a dense (rows × cols) weight reconstruction.'"
        ),
        "reads_per_token": "mixed Uniform HGRAVU01 q3 organs only (qwen38_hgravu01_geo_tpr64_launch bits==3). Not uniform-q4-v1.",
    },
    "qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128": {
        "representation_family": "grouped_absmax_q4",
        "role": "HGRAVU01 bits=4 geo_tpr64 on mixed Uniform factors",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same q80_mixed_decode.metal file contract as q3 sibling; decode-in-registers, no dense W.",
        "reads_per_token": "mixed Uniform HGRAVU01 q4 organs only. Not uniform-q4-v1.",
    },
    "qwen38_hgravu_embedding_lookup": {
        "representation_family": "grouped_absmax_hgravu",
        "role": "mixed-path embedding gather (HGRAVU packed row)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen38_device_activations.metal:503+: lookup of one packed embedding row into hidden; "
            "does not materialize vocab×hidden."
        ),
        "reads_per_token": "one packed embedding row on mixed catalog; unused on uniform-q4-v1.",
    },
    "q80_binary_group_matvec_tg256": {
        "representation_family": "binary_sign_scale",
        "role": "mixed recon-fuse binary GEMV occupancy tile (cols<=2048)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "q80_mixed_decode.metal:1-5 and :256-261: 'A value is still decoded into a register and "
            "consumed in the same FMA. Nothing writes a (rows × cols) reconstruction.'"
        ),
        "reads_per_token": "mixed binary_group organs (typically gate_proj). Sign bits + fp16 group scales + x.",
    },
    "q80_binary_group_matvec_simd_bytes": {
        "representation_family": "binary_sign_scale",
        "role": "mixed recon-fuse binary GEMV (cols>2048)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same binary in-register consume as tg256; decode.rs:1667-1671 selects by cols.",
        "reads_per_token": "mixed binary_group organs with cols>2048.",
    },
    "q80_binary_group_csr_matvec_tg256": {
        "representation_family": "binary_plus_sparse_csr",
        "role": "fused binary GEMV + CSR q1 residual in one dispatch (recon-fuse residual)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "q80_mixed_decode.metal:414-461: binary lane terms then CSR residual add into acc; "
            "file contract forbids dense W. rice expand is bind-time indices, never W (line 182)."
        ),
        "reads_per_token": "mixed up_proj-style binary+rice residual: signs+scales + CSR indices/row_ptr + residual signs + x.",
    },
    "q80_binary_group_csr_matvec_bytes": {
        "representation_family": "binary_plus_sparse_csr",
        "role": "wide-column sibling of fused binary+CSR",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same fused binary+CSR consume as tg256; decode.rs:1693-1702.",
        "reads_per_token": "mixed residual organs with cols>2048.",
    },
    "q80_sparse_q1_apply_csr": {
        "representation_family": "sparse_correction",
        "role": "CSR residual apply; HAWKING_QWEN38_RECON_FUSE=0 split path after binary",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "q80_mixed_decode.metal:154-179: acc += residual_sign * scale * input[col] over CSR; "
            "line 182: bind-time rice expand 'writes uint32 indices, never a dense W'."
        ),
        "reads_per_token": "CSR indices + residual signs + x, added onto an already-written y. Split path only.",
    },
    "q80_hgravs01_factor_matvec_simd": {
        "representation_family": "low_rank_factors",
        "role": "one HGRAVS01 factor (n-bit); two dispatches make y = L@(R@x)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "q80_mixed_decode.metal:24-25: 'execute y = L @ (R @ x); mid[rank] is the only temporary.' "
            ":256-261 never dense W. Host dispatch_hgravs (decode.rs:1761-1797) writes workspace.hgravs_mid "
            "(rank vector), not W = L@R. NS-019: reconstruct-W-then-multiply is refuted."
        ),
        "reads_per_token": "mixed down_proj-style HGRAVS01: two factor bodies (right then left) + x + rank-vector mid.",
    },
    "q80_hgravs01_factor_matvec_simd3": {
        "representation_family": "low_rank_factors",
        "role": "3-bit occupancy tile of one HGRAVS01 factor (default recon-fuse bits==3)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same y=L@(R@x) via two factor dispatches; simd3 is the bits==3 tile (decode.rs:1745-1746).",
        "reads_per_token": "mixed HGRAVS01 3-bit factors; mid[rank] only extra buffer.",
    },
    "q80_uniform8_matvec_simd_bytes": {
        "representation_family": "grouped_absmax_q8",
        "role": "uniform-8 factor tile, narrow cols",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "q80_mixed_decode.metal file contract; decode.rs:1738-1744 bits==8 narrow.",
        "reads_per_token": "mixed uniform-8 organs with cols<2048.",
    },
    "q80_uniform8_matvec_tg256": {
        "representation_family": "grouped_absmax_q8",
        "role": "uniform-8 factor tile, wide cols",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same in-register uniform decode as simd_bytes; decode.rs:1740-1741 cols>=2048.",
        "reads_per_token": "mixed uniform-8 organs with cols>=2048.",
    },
    "qwen38_qkvz_rearrange_conv_l2_f32": {
        "representation_family": "activation_rearrange_conv",
        "role": "DeltaNet qkvz rearrange + conv1d on activations (48 layers)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Operates on f32 activations + small conv weights; does not decode a packed GEMV W.",
        "reads_per_token": f"48× (qkvz activation {QWEN38_QKVZ_ROWS}*4 B + conv1d f32 + conv state rw).",
    },
    "qwen38_fuse_split_qkvz_f32": {
        "representation_family": "activation_glue",
        "role": "concat split in_proj_qkv + in_proj_z into qkvz (unfused pack)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "decode.rs:1842-1855 copies f32 activation slices; no weight decode.",
        "reads_per_token": "only if qkv/z were projected separately. Production pack fuses qkvz as one Q4 GEMV.",
    },
    "qwen38_fuse_split_ba_f32": {
        "representation_family": "activation_glue",
        "role": "concat split in_proj_b + in_proj_a (unfused pack)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "decode.rs:1866-1877 copies f32 activation slices.",
        "reads_per_token": "only if b/a were projected separately. Production pack fuses BA as one Q4 GEMV.",
    },
    "qwen38_gated_delta_decode_vi": {
        "representation_family": "recurrent_state_operator",
        "role": "DeltaNet recurrence, one TG per (head, value_dim); serial reduction",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen38_device_activations.metal:314-363: updates rec_state[head,k,v] in place from q,k,v,decay,beta. "
            "No packed weights. HAWKING_DN_VI_SIMD=0 restores this; default is the simd sibling."
        ),
        "reads_per_token": "48× rec_state rw + q,k,v,decay,beta activations (DeltaNet layers).",
    },
    "qwen38_gated_delta_decode_vi_simd": {
        "representation_family": "recurrent_state_operator",
        "role": "production DeltaNet recurrence (default: deltanet_vi_parallel + HAWKING_DN_VI_SIMD)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "qwen38_device_activations.metal:365-389: same state arithmetic as vi, simd reduction. "
            "decode.rs:2014-2028 default simd=true. Not a weight codec."
        ),
        "reads_per_token": "48× rec_state rw + q,k,v,decay,beta. Token-ns component 'deltanet'.",
    },
    "qwen80_gated_delta_decode_tg": {
        "representation_family": "recurrent_state_operator",
        "role": "Q80-layout DeltaNet recurrence (deltanet_vi_parallel=false)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "decode.rs:2030-2034 fallback when vi_parallel is off. State operator, no packed W.",
        "reads_per_token": "same state as vi; coarser launch. Default session has vi_parallel=true.",
    },
    "qwen80_ba_to_decay_beta_f32": {
        "representation_family": "activation_glue",
        "role": "BA → decay/beta (48 DeltaNet layers)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "qwen80_device_activations.metal:188: f32 activation map, no packed W.",
        "reads_per_token": "48× BA activation (96 f32) + writes decay/beta.",
    },
    "qwen80_deltanet_gated_rmsnorm_f32": {
        "representation_family": "activation_norm",
        "role": "DeltaNet gated RMSNorm, one-thread-per-head (HAWKING_DN_RMSNORM_TG=0)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "f32 residual / z / norm-scale; no packed GEMV W.",
        "reads_per_token": "48× (value-head activations + f32 norm scale) when TG=0.",
    },
    "qwen80_deltanet_gated_rmsnorm_tg": {
        "representation_family": "activation_norm",
        "role": "production DeltaNet gated RMSNorm (default HAWKING_DN_RMSNORM_TG=256)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "decode.rs:2384-2396 unwrap_or(256) selects the tg kernel. Activation RMSNorm.",
        "reads_per_token": "48× rec_out + z + f32 norm scale.",
    },
    "qwen80_residual_rmsnorm_f32": {
        "representation_family": "activation_norm",
        "role": "residual RMSNorm, 256-pinned (HAWKING_RMSNORM_TG=0)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "qwen80_device_activations.metal:24: f32 x * (w+δ) RMSNorm. decode.rs:2990-2994.",
        "reads_per_token": f"hidden f32 + {QWEN38_HIDDEN}*4 B norm scale per dispatch, when TG=0.",
    },
    "qwen80_residual_rmsnorm_tg": {
        "representation_family": "activation_norm",
        "role": "production residual RMSNorm (default HAWKING_RMSNORM_TG=1024)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "decode.rs:2981-2991 unwrap_or(1024). Reads f32 activation + f32 affine, not packed W.",
        "reads_per_token": "2 per layer (input + post-attn) + final = 129 RMSNorms/token on 64L. Each reads hidden f32 + hidden f32 scale.",
    },
    "qwen38_gqa_qk_norm_rope_cache_f32": {
        "representation_family": "activation_rope_cache",
        "role": "GQA Q/K RMSNorm + RoPE + KV cache write (HAWKING_ROPE_TG=0)",
        "production_uniform_q4_default": False,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Activation + f32 q/k norm scales + writes one KV slot. No packed W reconstruct.",
        "reads_per_token": "16 GQA layers when TG=0.",
    },
    "qwen38_gqa_qk_norm_rope_cache_tg": {
        "representation_family": "activation_rope_cache",
        "role": "production GQA Q/K RMSNorm + RoPE + KV cache (default HAWKING_ROPE_TG=256)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "decode.rs:2428-2439 unwrap_or(256). Writes K/V cache; does not unpack a GEMV W.",
        "reads_per_token": "16× (q,k,v activations + q/k f32 norms + one KV slot write, seq_len KV read is in mha_decode_f32).",
    },
    "qwen38_attention_apply_sigmoid_gate": {
        "representation_family": "activation_glue",
        "role": "GQA sigmoid gate on attention output (16 layers)",
        "production_uniform_q4_default": True,
        "mixed_path": True,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "qwen38_device_activations.metal:428: f32 attn ⊙ σ(q_proj slice). No packed W.",
        "reads_per_token": "16× query_dim f32 attn + q_proj.",
    },
    "qwen38_f32_stream_probe": {
        "representation_family": "diagnostic_probe",
        "role": "f32 stream diagnostic",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": True,
        "reconstructs_dense": "NO",
        "evidence": "qwen38_device_activations.metal:524: 10-line probe; not a weight codec.",
        "reads_per_token": "not on the 964-dispatch token.",
    },
    "sample_argmax_f32_pass1": {
        "representation_family": "sampling",
        "role": "two-pass argmax pass1 (HAWKING_ARGMAX_TWO_PASS=1; default OFF)",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": (
            "decode.rs:2513-2526: two-pass is DEFAULT OFF because it does not move token wall. "
            "Default encode_argmax calls sample_argmax_f32_tcb → kernel sample_argmax_f32, which is "
            "NOT in the nx_genome 38 (helper, not a decode.rs literal)."
        ),
        "reads_per_token": "logits f32 vocab when two-pass enabled; default token uses sample_argmax_f32 instead.",
    },
    "sample_argmax_f32_pass2": {
        "representation_family": "sampling",
        "role": "two-pass argmax pass2 (HAWKING_ARGMAX_TWO_PASS=1; default OFF)",
        "production_uniform_q4_default": False,
        "mixed_path": False,
        "diagnostic": False,
        "reconstructs_dense": "NO",
        "evidence": "Same as pass1; default path does not dispatch either pass.",
        "reads_per_token": "partials from pass1 when two-pass enabled.",
    },
}


HELPER_DISPATCHED = [
    {
        "name": "mha_decode_f32",
        "why_not_in_38": "decode.rs imports mha_decode_f32_tcb; the kernel name string lives in kernels/mod.rs:10562, not as a decode.rs literal",
        "reconstructs_dense": "NO",
        "evidence": "mha.metal:602 mha_decode_f32: GQA attention over KV cache. Activations only.",
        "production_uniform_q4_default": True,
        "dispatches_per_token": QWEN38_GQA_LAYERS,
    },
    {
        "name": "qwen_next_add_residual",
        "why_not_in_38": "decode.rs calls qwen_next_add_residual_tcb; kernel name is in kernels/mod.rs:13309",
        "reconstructs_dense": "NO",
        "evidence": "qwen_next.metal:328: x+residual f32. Schedule uses it twice per layer (mixer + mlp).",
        "production_uniform_q4_default": True,
        "dispatches_per_token": QWEN38_LAYERS * 2,
    },
    {
        "name": "sample_argmax_f32",
        "why_not_in_38": "default encode_argmax uses sample_argmax_f32_tcb (kernels/mod.rs:14255); pass1/pass2 ARE literals and ARE in the 38 but default-off",
        "reconstructs_dense": "NO",
        "evidence": "sample.metal:48. Reads logits, writes one id. No weights.",
        "production_uniform_q4_default": True,
        "dispatches_per_token": 1,
    },
    {
        "name": "gk_swiglu_f32",
        "why_not_in_38": "encode_silu dispatches decode_family::swiglu_f32() which is gk_swiglu_f32 (or legacy qwen80_silu_mul_f32); neither string is a decode.rs literal",
        "reconstructs_dense": "NO",
        "evidence": "gk_family.metal gk_swiglu_f32 / qwen80_silu_mul_f32: silu(gate)*up on f32 activations.",
        "production_uniform_q4_default": True,
        "dispatches_per_token": QWEN38_LAYERS,
        "alt_name": "qwen80_silu_mul_f32",
    },
]


def production_byte_budget() -> dict:
    h = QWEN38_HIDDEN
    mid = QWEN38_INTERMEDIATE
    mlp = QWEN38_LAYERS * (
        q4_matrix_bytes(mid, h) + q4_matrix_bytes(mid, h) + q4_matrix_bytes(h, mid)
    )
    qkvz = q4_matrix_bytes(QWEN38_QKVZ_ROWS, h)
    ba = q4_matrix_bytes(QWEN38_BA_ROWS, h)
    dn_out = q4_matrix_bytes(h, QWEN38_O_PROJ_COLS)
    linear = QWEN38_DELTANET_LAYERS * (qkvz + ba + dn_out)
    gqa_q = q4_matrix_bytes(QWEN38_Q_PROJ_ROWS, h)
    gqa_kv = q4_matrix_bytes(QWEN38_KV_PROJ_ROWS, h)
    gqa_o = q4_matrix_bytes(QWEN38_O_PROJ_ROWS, QWEN38_O_PROJ_COLS)
    full = QWEN38_GQA_LAYERS * (gqa_q + gqa_kv + gqa_kv + gqa_o)
    lm_head = q4_matrix_bytes(QWEN38_VOCAB, h)
    embed_row = q4_matrix_bytes(1, h)
    embed_table = q4_matrix_bytes(QWEN38_VOCAB, h)
    # token_ns_ledger.rs theoretical_weight_bytes norms formula
    norms = (
        4 * QWEN38_LAYERS * h * 4
        + h * 4
        + QWEN38_GQA_LAYERS * 2 * 256 * 4
        + QWEN38_DELTANET_LAYERS * 6144 * 4
    )
    active = mlp + linear + full + lm_head + norms + embed_row
    return {
        "formula": "q4 group-64: rows * ceil(cols/64) * 34 bytes (32 code + 2 fp16 scale)",
        "mlp_bytes": mlp,
        "linear_attn_bytes": linear,
        "full_attn_bytes": full,
        "lm_head_bytes": lm_head,
        "norms_bytes": norms,
        "embed_row_bytes": embed_row,
        "embed_table_resident_not_streamed": embed_table,
        "active_weight_bytes_per_token": active,
        "active_budget_bytes_constant": 13_622_264_240,
        "active_budget_minus_payload_geometry": 13_622_264_240 - active,
        "note": (
            "Payload geometry matches qwen38_token_ns_ledger.rs::theoretical_weight_bytes "
            "(mlp 9_091_153_920, linear 2_953_789_440, full 891_289_600, lm_head 675_430_400). "
            "ACTIVE_BUDGET_BYTES 13_622_264_240 is the measured class total: the same test says "
            "it includes per-tensor HQ30UQ4 headers (~40 B) and a few f32 mixer scales, and "
            "allows a <20 MiB gap. Embed table is resident; only one row is read per token."
        ),
        "workhorse_kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "workhorse_dispatches_per_token": (
            QWEN38_LAYERS * 3  # mlp gate,up,down
            + QWEN38_DELTANET_LAYERS * 3  # qkvz, ba, out
            + QWEN38_GQA_LAYERS * 4  # q,k,v,o
            + 1  # lm_head
        ),
        "production_dispatches_per_token": PRODUCTION_DISPATCHES,
        "production_command_buffers": PRODUCTION_COMMAND_BUFFERS,
    }


def family_table(by_name: dict) -> list[dict]:
    """Seven representation families. EXISTS/PARTIAL/ABSENT with a named kernel.

    EXISTS = a kernel in this tree executes that structure without writing dense W.
    PARTIAL = pieces exist (or a sibling on activations / another organ) but not
              the fused family as specified.
    ABSENT = no kernel. Do not promote a dense GEMM or a host SVD to EXISTS.
    """

    def kinfo(name: str) -> dict:
        k = by_name.get(name)
        if not k:
            return {"name": name, "present": False}
        return {
            "name": name,
            "present": True,
            "class": k["class"],
            "file": k["file"],
            "line": k["line"],
            "compile_gate": k.get("compile_gate"),
        }

    return [
        {
            "id": "shared_basis_x_coefficients",
            "spec": "shared basis × per-site coefficients without dense reconstruction",
            "verdict": "ABSENT",
            "kernel": None,
            "why": (
                "Zero shaders mention shared_basis / shared_codebook / joint_basis. "
                "q80_hgravs01_factor_matvec_* is per-tensor U,V — not a basis shared across layers. "
                "G035 G-SHARE measured shared_beats_independent=false. G042 SHARED_BPW is hardcoded 0.0. "
                "NS-010 killed cross-expert shared bases on Q80."
            ),
            "related": [],
        },
        {
            "id": "tensor_contraction",
            "spec": "structured tensor contraction (TT / Tucker / CP / tensor-ring cores) without forming W",
            "verdict": "ABSENT",
            "kernel": None,
            "why": (
                "No kernel void names or bodies contain tensor_train / tt_core / tucker / tensor_ring / hosvd. "
                "Ordinary GEMM in matmul.metal is dense contraction, not this family. "
                "G1/G034 ran TT/Tucker/ring on real Qwen3.8 GEMV tensors: 223 rows with local_bpw<0.5, "
                "healthy=true count 0. G096 NEVER BUILT a TT node in NR."
            ),
            "related": ["gemm_f32", "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"],
        },
        {
            "id": "fused_dictionary_lookup_accumulate",
            "spec": "codebook/PQ lookup fused with accumulate into y; no dense W",
            "verdict": "EXISTS",
            "kernel": kinfo("gravity_pq_matvec"),
            "why": (
                "gravity_pq.metal:399-428 gravity_pq_matvec: per subspace, pq_index(codes) → codebook entry, "
                "fma into acc, simd_sum to y[row]. Codebook is the dictionary; W is never written. "
                "gravity_residual_pq_matvec is additive multi-stage PQ. Not on the Qwen3.8 uniform-q4 path "
                "(gravity GLM/Llama). Porting is bind work, not a new kernel."
            ),
            "related": [
                "gravity_residual_pq_matvec",
                "gravity_pq_matvec_bits8_direct",
                "gravity_glm_expert_table_pq_matvec",
            ],
        },
        {
            "id": "low_rank_plus_sparse_correction",
            "spec": "low-rank factors plus a sparse correction, executed without forming W",
            "verdict": "PARTIAL",
            "kernel": kinfo("q80_hgravs01_factor_matvec_simd3"),
            "why": (
                "Low-rank EXISTS: HGRAVS01 y=L@(R@x) via two factor kernels (DISPATCHED on mixed) or "
                "q80_hgravs01_two_stage_matvec (REACHABLE; threadgroup mid[160], '640 B, not dense W'). "
                "Sparse correction EXISTS separately: q80_sparse_q1_apply_csr (DISPATCHED) and fused "
                "binary+CSR q80_binary_group_csr_matvec_tg256 (DISPATCHED). "
                "qwen30_quality_repack_sparse_gate_up_swiglu is binary-base + sparse residual, not low-rank. "
                "No kernel does UV + sparse in one representation. Mixed decode assigns low-rank to down_proj "
                "and binary+sparse to up_proj — different organs. NNS-015: hybrid low-rank+correction is "
                "Pareto-dominated by q3 as a byte lever."
            ),
            "related": [
                "q80_hgravs01_two_stage_matvec",
                "q80_sparse_q1_apply_csr",
                "q80_binary_group_csr_matvec_tg256",
                "qwen30_quality_repack_sparse_gate_up_swiglu",
            ],
        },
        {
            "id": "structured_transform",
            "spec": "structured transform (Hadamard / FWHT / FFT / Kronecker) as the operator, not as generated W",
            "verdict": "PARTIAL",
            "kernel": kinfo("strand_rht_forward_cols"),
            "why": (
                "strand_rht_forward_cols implements a 256-wide FWHT on an activation vector "
                "(strand_bitslice.metal:826-862). compile_gate=feature=tq; not in the default library. "
                "It transforms x, then a bitslice GEMV consumes tx — it is not a weight-side "
                "y = H diag(s) Hᵀ x kernel. G032 Block-diagonal Sylvester-Hadamard ran as a "
                "reparameterization with GENERATED_BPW_EQUIVALENT=0.0 and mean entropy delta +0.024 bits. "
                "No default-build kernel executes a structured weight transform without materializing W."
            ),
            "related": ["strand_rht_forward_cols_batched"],
        },
        {
            "id": "routed_group_execution",
            "spec": "route-selected group of experts/slots executed without densifying the catalog",
            "verdict": "EXISTS",
            "kernel": kinfo("gk_worklist_fp4"),
            "why": (
                "G023 family: gk_pack_worklist + gk_worklist_fp4(+_simd) + gk_swiglu_bf16_worklist + "
                "gk_combine_bf16. Also moe_topk_gate, gravity_glm_expert_table_pq_matvec, "
                "qwen80_routed_expert_wave / qwen80_all_ten_routed_expert_wave, dsv4f_pack_worklist. "
                "Qwen3.8 is dense (qwen38_geometry.rs refuses num_experts). These kernels run Q80 / DSV4F / GLM, "
                "not the uniform-q4 Qwen3.8 token. ROUTING_FLOPS on that artifact is 0.0 because there is no route."
            ),
            "related": [
                "gk_pack_worklist",
                "moe_topk_gate",
                "gravity_glm_expert_table_pq_matvec",
                "qwen80_all_ten_routed_expert_wave",
                "dsv4f_pack_worklist",
            ],
        },
        {
            "id": "recurrent_state_operator",
            "spec": "recurrent / state operator (DeltaNet, RWKV WKV) with persistent GPU state",
            "verdict": "EXISTS",
            "kernel": kinfo("qwen38_gated_delta_decode_vi_simd"),
            "why": (
                "DISPATCHED on the Qwen3.8 token: qwen38_gated_delta_decode_vi_simd (default) updates "
                "rec_state in place. Siblings: qwen38_gated_delta_decode_vi, qwen80_gated_delta_decode_tg, "
                "qwen_next_gated_delta_decode_single. Separate family: rwkv7_wkv_decode (REACHABLE; "
                "threadgroup-per-head, fixed head×head state, no growing KV). This is the mixer the NR assumes."
            ),
            "related": [
                "qwen38_gated_delta_decode_vi",
                "qwen80_gated_delta_decode_tg",
                "rwkv7_wkv_decode",
                "qwen_next_gated_delta_decode_single",
            ],
        },
    ]


def missing_cost(families: list[dict]) -> list[dict]:
    out = []
    for fam in families:
        if fam["verdict"] == "EXISTS":
            if fam["id"] == "recurrent_state_operator":
                estimate = "Already on the Qwen3.8 production token."
            elif fam["id"] == "routed_group_execution":
                estimate = (
                    "Kernel already exists for Q80/DSV4F/GLM. Qwen3.8 is dense "
                    "(qwen38_geometry.rs refuses num_experts); there is no route to bind. "
                    "Do not invent a router to use these kernels."
                )
            else:
                estimate = (
                    "Kernel already exists. Wiring it onto a Qwen3.8 token is bind/schedule "
                    "work, not a new shader family."
                )
            out.append(
                {
                    "id": fam["id"],
                    "status": "nothing_to_write_for_the_family",
                    "estimate": estimate,
                }
            )
            continue
        if fam["id"] == "shared_basis_x_coefficients":
            out.append(
                {
                    "id": fam["id"],
                    "status": "kernel_cheap_quality_blocked",
                    "estimate": (
                        "GPU: clone q80_hgravs01_two_stage_matvec (~200 lines) to take a shared B "
                        "buffer plus per-site C, y = B@(C@x), mid[rank] in threadgroup. Host: bind B once "
                        "into unified memory, per-layer C like today's factors (~100 lines). NX: name the "
                        "kernel. That is a 1–2 day shader+bind if quality were solved. G035 already refuted "
                        "sharing on this model (shared_beats_independent=false). Do not write the kernel "
                        "to reopen a closed idea."
                    ),
                }
            )
        elif fam["id"] == "tensor_contraction":
            out.append(
                {
                    "id": fam["id"],
                    "status": "kernel_large_quality_blocked",
                    "estimate": (
                        "A TT-core or Tucker kernel is a new family: packer for cores, NX fields, "
                        "and a contraction that never forms W (roughly 400–800 lines Metal + pack path). "
                        "G1 scored 80 tensor_train rows and G034 refuted TT unfolding; 223 sub-0.5 local_bpw "
                        "rows, healthy=true: 0. Do not write until a component is healthy on real X."
                    ),
                }
            )
        elif fam["id"] == "low_rank_plus_sparse_correction":
            out.append(
                {
                    "id": fam["id"],
                    "status": "fuse_the_existing_pieces",
                    "estimate": (
                        "Add the CSR residual loop that q80_binary_group_csr_matvec already has onto "
                        "q80_hgravs01_two_stage_matvec after L consumes mid. ~80–150 lines Metal, ABI for "
                        "indices/row_ptr/residual_signs. Pieces are DISPATCHED separately today. "
                        "NNS-015: hybrid correction is Pareto-dominated by q3 as a byte lever; distillation "
                        "of the MLP function was named as the surviving avenue and has not been run."
                    ),
                }
            )
        elif fam["id"] == "structured_transform":
            out.append(
                {
                    "id": fam["id"],
                    "status": "activation_fwht_exists_weight_side_absent",
                    "estimate": (
                        "Activation-side FWHT is strand_rht_forward_cols under feature=tq. A weight-side "
                        "y=H diag(s) Hᵀ x fused with consume would be ~200 lines if RHT is reused, plus "
                        "default-library compile (drop the tq gate or fork into all_shader_sources). "
                        "G032 measured +0.024 bits entropy and GENERATED_BPW_EQUIVALENT=0.0. Quality, not "
                        "kernel volume, is the blocker."
                    ),
                }
            )
        else:
            out.append({"id": fam["id"], "status": "unspecified", "estimate": "UNKNOWN"})
    return out


def classify(declared: list[dict], compiled: dict) -> list[dict]:
    decode_text = DECODE.read_text(encoding="utf-8", errors="replace")
    lits = decode_string_literals(decode_text)
    names = {k["name"] for k in declared}
    bound = sorted(lits & names)
    hits = rust_quoted_kernel_hits(names)

    runtime = set(compiled["runtime_all_shader_sources"])
    classified = []
    for k in declared:
        name = k["name"]
        gate = compile_gate_for(k["file"], compiled)
        quoted = hits.get(name, {"files": [], "kinds": Counter()})
        nref = len(quoted["files"])
        if name in bound:
            cls = "DISPATCHED"
            reason = "quoted identifier in qwen38_hybrid_decode.rs ∩ declared kernel void (nx_genome)"
        elif is_macro_template(name):
            cls = "UNKNOWN"
            reason = (
                "preprocessor token-paste template (`##` or kernel void NAME); not a Metal function symbol. "
                "nx_genome counts the unexpanded line toward the 554. Expansions are separate runtime names."
            )
        elif nref == 0:
            cls = "DEAD"
            reason = "declared kernel void, never a quoted identifier in crates/**/*.rs"
        elif gate == "not_compiled":
            cls = "UNKNOWN"
            reason = "quoted in Rust but no include_str / all_shader_sources / example compile site found"
        else:
            cls = "REACHABLE"
            reason = (
                "compiled "
                + ("in default Metal library" if gate is None else f"({gate})")
                + " and referenced from Rust; not a decode.rs literal"
            )
        facts = DISPATCHED_FACTS.get(name) if cls == "DISPATCHED" else None
        row = {
            "name": name,
            "file": k["file"],
            "path": k["path"],
            "line": k["line"],
            "n_lines": k["n_lines"],
            "signature": k["signature"],
            "class": cls,
            "reason": reason,
            "compile_gate": gate,
            "in_default_library": k["file"] in runtime and k["file"] != "strand_bitslice.metal",
            "quoted_file_count": nref,
            "quoted_kinds": dict(quoted["kinds"]),
            "quoted_files": sorted(quoted["files"])[:12],
        }
        if cls == "DISPATCHED":
            if facts is None:
                row["reconstructs_dense"] = "UNKNOWN"
                row["representation_family"] = "UNKNOWN"
                row["evidence"] = "not in DISPATCHED_FACTS; refusing to guess"
                row["production_uniform_q4_default"] = False
                row["mixed_path"] = False
                row["diagnostic"] = False
                row["reads_per_token"] = "UNKNOWN"
                row["role"] = "UNKNOWN"
            else:
                row.update(facts)
        classified.append(row)
    return classified, bound


def watched_fail(classified: list[dict], bound: list[str], compiled: dict) -> list[dict]:
    counts = Counter(k["class"] for k in classified)
    unknown_dispatched = [
        k["name"]
        for k in classified
        if k["class"] == "DISPATCHED" and k.get("reconstructs_dense") == "UNKNOWN"
    ]
    macros = [k["name"] for k in classified if is_macro_template(k["name"])]
    dead = [k["name"] for k in classified if k["class"] == "DEAD"]
    not_in_mod = sorted(
        {p.name for p in SHADERS.glob("*.metal")}
        - set(compiled["runtime_all_shader_sources"])
    )
    return [
        {
            "id": 1,
            "what": "HCLI baseline 464 passed / 1 skipped did not reproduce",
            "detail": (
                "tools/haider is not in this sparse checkout (git ls-tree has 54 haider files; "
                "on-disk find returned only the untracked tar extras). pytest tools/headless "
                "collection errors with ModuleNotFoundError: hcli.grok_bridge / hcli.app. "
                "Contract forbids git sparse-checkout add. Kernel census proceeded on materialized "
                "crates/hawking-core which is what this lane enumerates."
            ),
        },
        {
            "id": 2,
            "what": "nx_genome 38 is an underestimate of kernels the decode path actually launches",
            "detail": (
                "qwen38_hybrid_decode.rs calls mha_decode_f32_tcb, qwen_next_add_residual_tcb, "
                "sample_argmax_f32_tcb, and decode_family::swiglu_f32(). Those names are not "
                "string literals in decode.rs, so they are REACHABLE rather than DISPATCHED under "
                "the G071 rule. The 64-layer schedule names them. A seal of the 38 is honest about "
                "literals and silent about helpers. See helper_dispatched_not_in_38."
            ),
        },
        {
            "id": 3,
            "what": "554 declared includes preprocessor templates that are not Metal symbols",
            "detail": (
                f"Unexpanded lines counted as kernel names: {macros}. Real expansions "
                "(qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128, …, "
                "strand_bitslice_gemm_partials_b4/b16/b64) exist at runtime and are REACHABLE "
                "via metal/mod.rs static_kernel_name, but are extra symbols on top of the 554, not inside it."
            ),
        },
        {
            "id": 4,
            "what": "fused two-stage HGRAVS01 exists and the mixed graph still launches two factor dispatches",
            "detail": (
                "q80_hgravs01_two_stage_matvec is REACHABLE (examples + metal registry) and explicitly "
                "refuses to write dense W (mid[160] in threadgroup). decode.rs::dispatch_hgravs still "
                "calls dispatch_factor twice (right then left) through hgravs_mid. NS-030 refuted fusing "
                "by recomputing R in every threadgroup on Q80 down_proj; this is that kernel."
            ),
        },
        {
            "id": 5,
            "what": "G043 reconstruction_share ≈ 0.71 is not reconstructs-dense YES",
            "detail": (
                "The decode_probe kernel isolates nibble-unpack ALU with no x and no FMA. Production "
                "geo_tpr64 does that unpack in-register and FMAs immediately. 71% of physical FLOPs can "
                "be dequant and the kernel still never writes W. Treating reconstruction_share as "
                "oracle-path evidence would relitigate NS-019."
            ),
        },
        {
            "id": 6,
            "what": "five shader files are not in all_shader_sources()",
            "detail": (
                f"{not_in_mod}. They compile from examples; the historical "
                "ascension_qwen30_packed_matvec_exactness producer is receipt-only and retired. "
                "They are not in the default decode library."
            ),
        },
        {
            "id": 7,
            "what": "strand structured-transform / bitslice family is feature=tq",
            "detail": (
                "SHADER_STRAND_BITSLICE is #[cfg(feature = tq)]. Default hawking-core library does not "
                "contain strand_rht_forward_cols. Claiming structured-transform EXISTS without the gate "
                "would overstate what a vanilla decode binary can launch."
            ),
        },
        {
            "id": 8,
            "what": "shared-basis and tensor-contraction families have no kernel, and prior science already failed on quality",
            "detail": (
                "A low number without a health verdict is a trap (223 sub-0.5 local_bpw, healthy=0). "
                "Writing the missing kernels would not reopen G035 or G034. Dictionary lookup EXISTS "
                "as gravity_pq_matvec and is still unused by Qwen3.8 uniform-q4."
            ),
        },
        {
            "id": 9,
            "what": "default two-pass argmax is off, so the 38 lists pass1/pass2 that the token does not run",
            "detail": (
                "HAWKING_ARGMAX_TWO_PASS default false. Production uses sample_argmax_f32 (helper). "
                "pass1/pass2 are DISPATCHED only under the literal rule."
            ),
        },
        {
            "id": 10,
            "what": f"class counts {dict(counts)}; unknown_dispatched={unknown_dispatched}; dead={dead}",
            "detail": "Any DISPATCHED kernel missing from DISPATCHED_FACTS is UNKNOWN reconstructs-dense, never guessed.",
        },
    ]


def build() -> tuple[dict, str]:
    declared_raw = parse_declared_kernels()
    compiled = compiled_shader_files()
    classified, bound = classify(declared_raw, compiled)
    by_name = {k["name"]: k for k in classified}
    families = family_table(by_name)
    # strip bodies from JSON (keep census lean); facts already carry evidence
    for row in classified:
        row.pop("body", None)

    counts = Counter(k["class"] for k in classified)
    dispatched = [k for k in classified if k["class"] == "DISPATCHED"]
    dead = [k for k in classified if k["class"] == "DEAD"]
    unknown = [k for k in classified if k["class"] == "UNKNOWN"]
    reachable = [k for k in classified if k["class"] == "REACHABLE"]

    recon = Counter(k.get("reconstructs_dense", "n/a") for k in dispatched)
    by_file = {}
    for k in classified:
        slot = by_file.setdefault(
            k["file"], {"declared": 0, "DISPATCHED": 0, "REACHABLE": 0, "DEAD": 0, "UNKNOWN": 0}
        )
        slot["declared"] += 1
        slot[k["class"]] += 1

    missing = [k["name"] for k in dispatched if k["name"] not in DISPATCHED_FACTS]
    if missing:
        raise SystemExit(f"DISPATCHED kernels missing facts (refusing to guess): {missing}")

    if len(declared_raw) != G071_DECLARED:
        recon_note = (
            f"declared {len(declared_raw)} != recorded {G071_DECLARED} — "
            "extractor drifted; receipt records live count"
        )
    else:
        recon_note = f"declared {len(declared_raw)} matches recorded {G071_DECLARED}"
    if len(bound) != G071_BOUND:
        bound_note = f"bound {len(bound)} != recorded {G071_BOUND}"
    else:
        bound_note = f"bound {len(bound)} matches recorded {G071_BOUND}"

    shader_roll = hashlib.sha256()
    for p in sorted(SHADERS.glob("*.metal")):
        shader_roll.update(p.name.encode())
        shader_roll.update(sha256_bytes(p.read_bytes()).encode())

    receipt = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "method": {
            "declared": "line.startswith('kernel void ') over crates/hawking-core/shaders/*.metal (nx_genome.py)",
            "dispatched": "alnum/underscore string literals in qwen38_hybrid_decode.rs ∩ declared names",
            "reachable": "quoted in crates/**/*.rs AND compiled (default library or example or feature=tq)",
            "dead": "declared, never quoted in Rust",
            "unknown": "macro template, or quoted with no compile site; never guessed",
            "reconstructs_dense": "YES only if the kernel writes a (rows×cols) W in device memory. In-register dequant+FMA is NO.",
        },
        "reconciliation": {
            "recorded_bound": G071_BOUND,
            "recorded_declared": G071_DECLARED,
            "live_bound": len(bound),
            "live_declared": len(declared_raw),
            "bound_note": bound_note,
            "declared_note": recon_note,
            "unique_declared_names": len({k["name"] for k in declared_raw}),
            "shader_files": len(list(SHADERS.glob("*.metal"))),
            "shader_tree_sha256": shader_roll.hexdigest(),
            "counts": dict(counts),
            "sum_classes": int(sum(counts.values())),
            "dispatched_reconstructs_dense": dict(recon),
        },
        "production_token": production_byte_budget(),
        "helper_dispatched_not_in_38": HELPER_DISPATCHED,
        "families": families,
        "missing_family_cost": missing_cost(families),
        "by_file": by_file,
        "compile": {
            "runtime_shader_files": compiled["runtime_all_shader_sources"],
            "not_in_all_shader_sources": sorted(
                {p.name for p in SHADERS.glob("*.metal")}
                - set(compiled["runtime_all_shader_sources"])
            ),
            "tq_gated": compiled["tq_gated"],
        },
        "dispatched": dispatched,
        "dead": dead,
        "unknown": unknown,
        "reachable": [
            {
                "name": k["name"],
                "file": k["file"],
                "line": k["line"],
                "compile_gate": k["compile_gate"],
                "quoted_file_count": k["quoted_file_count"],
                "quoted_kinds": k["quoted_kinds"],
            }
            for k in reachable
        ],
        "watched_fail": watched_fail(classified, bound, compiled),
        "write_scope": {
            "write": [
                "tools/headless/noetic_kernel_census.py",
                "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            ],
            "deny": ["crates", "workspace", "visionmcp", "app", "lab", "tools/haider"],
            "crates_read_only": True,
        },
        "constraints_from_recovered_science": [
            "223 components measured below 0.5 local BPW, zero healthy — a new low number is not a result without a health verdict",
            "Q80 storage BPW 0.6462 vs ACTIVE 2.518 — storage and execution cost are different quantities",
            "G035 G-SHARE shared_beats_independent=false",
            "GLM 0.167 expert BPW trap; HGRAVS01 0.13 on down_proj ONLY",
            "five information-hiding scenarios currently MISSED by NR/NX (SHARED_BPW 0, route_graph null, ROUTING_FLOPS 0, generated_structures empty, no shader source hashed, set_bytes unhashed)",
            "never evaluate compression on synthetic activations; cosine is scale-invariant; raw activation cosine null ≈ 0.898",
        ],
    }
    return receipt, format_report(receipt)


def format_report(r: dict) -> str:
    rec = r["reconciliation"]
    prod = r["production_token"]
    lines = []
    a = lines.append
    a("=" * 78)
    a("NOETIC KERNEL CENSUS")
    a("=" * 78)
    a(f"schema     {r['schema']}")
    a(f"generated  {r['generated_at']}")
    a(f"head       {r['git_head'][:12]}")
    a(f"receipt    {RECEIPT}")
    a("")
    a("## RECONCILIATION (G071 / nx_genome 38 of 554)")
    a(f"  declared  live={rec['live_declared']}  recorded={rec['recorded_declared']}  {rec['declared_note']}")
    a(f"  bound     live={rec['live_bound']}  recorded={rec['recorded_bound']}  {rec['bound_note']}")
    a(f"  classes   {rec['counts']}  sum={rec['sum_classes']}")
    a(f"  dispatched reconstructs-dense {rec['dispatched_reconstructs_dense']}")
    a(f"  shader files {rec['shader_files']}  tree_sha256 {rec['shader_tree_sha256'][:16]}…")
    a("")
    a("## PRODUCTION TOKEN (uniform-q4 path)")
    a(f"  dispatches/token {prod['production_dispatches_per_token']}  command_buffers {prod['production_command_buffers']}")
    a(f"  workhorse {prod['workhorse_kernel']}  × {prod['workhorse_dispatches_per_token']} GEMVs/token")
    a(
        f"  active payload bytes/token {prod['active_weight_bytes_per_token']:,}  "
        f"(ledger ACTIVE_BUDGET_BYTES {prod['active_budget_bytes_constant']:,}, "
        f"gap {prod['active_budget_minus_payload_geometry']:,} headers/mixer scales)"
    )
    a(f"  mlp {prod['mlp_bytes']:,}  delta {prod['linear_attn_bytes']:,}  gqa {prod['full_attn_bytes']:,}  lm_head {prod['lm_head_bytes']:,}")
    a("")
    a("## DISPATCHED (38) — reconstructs-dense is the deliverable")
    a(f"{'name':<58} {'dense':>5} {'prod':>4} {'mix':>3} {'diag':>4}  family")
    for k in r["dispatched"]:
        a(
            f"{k['name']:<58} {k['reconstructs_dense']:>5} "
            f"{'Y' if k.get('production_uniform_q4_default') else '-':>4} "
            f"{'Y' if k.get('mixed_path') else '-':>3} "
            f"{'Y' if k.get('diagnostic') else '-':>4}  {k.get('representation_family')}"
        )
    a("")
    a("  evidence (each DISPATCHED kernel):")
    for k in r["dispatched"]:
        a(f"    [{k['reconstructs_dense']}] {k['name']}")
        a(f"      {k['file']}:{k['line']}  {k.get('role')}")
        a(f"      {k.get('evidence')}")
        a(f"      reads: {k.get('reads_per_token')}")
    a("")
    a("## HELPER-DISPATCHED ON DECODE PATH, NOT IN THE 38")
    for h in r["helper_dispatched_not_in_38"]:
        a(f"  {h['name']:40s} dense={h['reconstructs_dense']}  prod={h['production_uniform_q4_default']}  n/token={h['dispatches_per_token']}")
        a(f"    why not in 38: {h['why_not_in_38']}")
        a(f"    {h['evidence']}")
    a("")
    a("## SEVEN REPRESENTATION FAMILIES")
    for fam in r["families"]:
        kern = fam.get("kernel")
        kn = None
        if isinstance(kern, dict):
            kn = f"{kern.get('name')} ({kern.get('class')}, {kern.get('file')}:{kern.get('line')})"
        a(f"  {fam['verdict']:<8} {fam['id']}")
        a(f"           kernel: {kn}")
        a(f"           {fam['why']}")
    a("")
    a("## COST OF THE MISSING ONES")
    for c in r["missing_family_cost"]:
        a(f"  {c['id']}: {c['status']}")
        a(f"    {c['estimate']}")
    a("")
    a("## DEAD (declared, never quoted in Rust)")
    if not r["dead"]:
        a("  (none)")
    for k in r["dead"]:
        a(f"  {k['name']:55s} {k['file']}:{k['line']}  gate={k['compile_gate']}")
        a(f"    {k['reason']}")
    a("")
    a("## UNKNOWN (never guessed)")
    if not r["unknown"]:
        a("  (none)")
    for k in r["unknown"]:
        a(f"  {k['name']:55s} {k['file']}:{k['line']}")
        a(f"    {k['reason']}")
    a("")
    a("## REACHABLE summary by file")
    for fname, slot in sorted(r["by_file"].items(), key=lambda kv: -kv[1]["declared"]):
        a(
            f"  {fname:55s} n={slot['declared']:3d}  "
            f"DISP={slot['DISPATCHED']:2d} REACH={slot['REACHABLE']:3d} "
            f"DEAD={slot['DEAD']:2d} UNK={slot['UNKNOWN']:2d}"
        )
    a(f"  reachable named in JSON: {len(r['reachable'])}")
    a(f"  not in all_shader_sources: {r['compile']['not_in_all_shader_sources']}")
    a(f"  tq-gated: {r['compile']['tq_gated']}")
    a("")
    a("## WHAT I WATCHED FAIL")
    for w in r["watched_fail"]:
        a(f"  {w['id']}. {w['what']}")
        a(f"     {w['detail']}")
    a("")
    a("## WRITE SCOPE")
    a(f"  WRITE {r['write_scope']['write']}")
    a(f"  DENY  {r['write_scope']['deny']}  crates_read_only={r['write_scope']['crates_read_only']}")
    a("=" * 78)
    return "\n".join(lines) + "\n"


def main() -> int:
    if not SHADERS.is_dir() or not DECODE.is_file():
        print(
            f"missing decode/shaders under {REPO} (sparse checkout hole?)",
            file=sys.stderr,
        )
        return 2
    receipt, report = build()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2) + "\n"
    RECEIPT.write_text(text)
    sys.stdout.write(report)
    # live checks the operator can see fail
    rec = receipt["reconciliation"]
    problems = []
    if rec["live_declared"] != G071_DECLARED:
        problems.append(f"declared {rec['live_declared']} != {G071_DECLARED}")
    if rec["live_bound"] != G071_BOUND:
        problems.append(f"bound {rec['live_bound']} != {G071_BOUND}")
    if rec["sum_classes"] != rec["live_declared"]:
        problems.append("class sum != declared")
    if rec["dispatched_reconstructs_dense"].get("UNKNOWN"):
        problems.append("a DISPATCHED kernel has reconstructs_dense UNKNOWN")
    yes = rec["dispatched_reconstructs_dense"].get("YES", 0)
    no = rec["dispatched_reconstructs_dense"].get("NO", 0)
    if yes + no != G071_BOUND:
        problems.append(f"YES+NO reconstructs-dense {yes}+{no} != {G071_BOUND}")
    fam_ok = {f["id"]: f["verdict"] for f in receipt["families"]}
    if set(fam_ok) != set(FAMILY_IDS):
        problems.append("family id set drifted")
    for f in receipt["families"]:
        if f["verdict"] not in ("EXISTS", "PARTIAL", "ABSENT"):
            problems.append(f"bad family verdict {f['id']}={f['verdict']}")
        if f["verdict"] == "EXISTS" and not (f.get("kernel") or {}).get("present"):
            problems.append(f"EXISTS family {f['id']} did not name a present kernel")
    if problems:
        print("CENSUS SELF-CHECK FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"wrote {RECEIPT} ({RECEIPT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
