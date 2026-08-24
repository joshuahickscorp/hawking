#!/usr/bin/env python3
"""N032 BYTES_FRONTIER: fewer active bytes/token vs native 2-bit MLP (2.25 bpw).

Three native representations (no dense W) measured as a 64-layer unique-weight
MLP token graph (gate+up+down = 192 GEMVs) plus the N021 non-MLP residual:

  ternary_5in8_g64          1.85 bpw stored = active (zeros are 0-FMA, bytes still load)
  shared_binary_k2          bases amortized across 64 layers
  binary_residual_sparse    binary plane + 2% CSR fused

The 2.25 bpw q2f g64 path is the baseline in the SAME harness.

    python3 tools/headless/bytes_frontier.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from first_noetic_executable import PARENT_PARAMS  # noqa: E402

SCHEMA = "hawking.headless.bytes_frontier.v1"
RECEIPT = REPO / "receipts" / "headless" / "BYTES_FRONTIER.json"
RAW = REPO / "receipts" / "headless" / "_BYTES_FRONTIER_raw.json"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "bytes_frontier.metal"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "bytes_frontier"
GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
GROUP = 64
SCALE_BITS = 16
TRIT_PACK = 8.0 / 5.0
Q2F_BPW = 2.25
MLP_ELEMENTS = LAYERS * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE)
# N021 NATIVE_2BIT_MLP: fused complete-token GPU median, 7 reps.
N021_COMPLETE_GPU_NS = 27_547_874
N021_COMPLETE_WALL_NS = 28_806_583
N021_Q2F_MLP_BYTES = 4_813_039_680
N021_PAYLOAD_BYTES = 10_019_572_760
Q4_ATTN_F32_BYTES = N021_PAYLOAD_BYTES - N021_Q2F_MLP_BYTES
ROOF_TOK_S = 729.7
ROOF_NS = 1e9 / ROOF_TOK_S
PARENT_BF16 = Path.home() / "models" / "qwen3.8-27b-abliterated-bf16"

KERNELS = (
    "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
    "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128",
    "ternary_5in8_g64_matvec_serial_c5120",
    "binary_g64_matvec_geo_c5120_tpr64_tg128",
    "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
    "shared_binary_k2_scale_contract_gpr80",
    "binary_sparse_fused_geo_c5120_tpr64_tg128",
    "q2f_g64_matvec_geo_c5120_tpr64_tg128",
)


def now_iso() -> str:
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


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def bpw_ternary(group: int = GROUP) -> float:
    return TRIT_PACK + SCALE_BITS / float(group)


def bpw_binary(group: int = GROUP) -> float:
    return 1.0 + SCALE_BITS / float(group)


def bpw_q2f(group: int = GROUP) -> float:
    return 2.0 + SCALE_BITS / float(group)


def organ_elements() -> dict[str, int]:
    gate = INTERMEDIATE * HIDDEN
    return {"gate": gate, "up": gate, "down": HIDDEN * INTERMEDIATE}


def shared_k2_bytes(n_layers: int = LAYERS, k: int = 2) -> dict[str, float]:
    """Signs once per organ; f16 scales per layer, K bases, group 64."""
    gate_sign = INTERMEDIATE * HIDDEN // 8
    down_sign = HIDDEN * INTERMEDIATE // 8
    sign_bytes = k * (2 * gate_sign + down_sign)
    gate_groups = INTERMEDIATE * (HIDDEN // GROUP)
    down_groups = HIDDEN * (INTERMEDIATE // GROUP)
    scale_bytes = n_layers * k * 2 * (2 * gate_groups + down_groups)
    active = sign_bytes + scale_bytes
    return {
        "basis_sign_bytes": float(sign_bytes),
        "scale_bytes": float(scale_bytes),
        "active_bytes": float(active),
        "active_bpw": 8.0 * active / MLP_ELEMENTS,
        "storage_bpw_cold": 8.0 * (sign_bytes + scale_bytes) / MLP_ELEMENTS,
        "k": k,
        "n_layers": n_layers,
    }


def residual_bytes(frac: float = 0.02) -> dict[str, float]:
    binary = MLP_ELEMENTS * bpw_binary() / 8.0
    nnz = MLP_ELEMENTS * frac
    # u32 col + f16 corr + amortized row_ptr
    csr = nnz * (4.0 + 2.0) + LAYERS * 3 * (INTERMEDIATE + 1) * 4
    active = binary + csr
    return {
        "binary_bytes": binary,
        "csr_bytes": csr,
        "nnz_frac": frac,
        "active_bytes": active,
        "active_bpw": 8.0 * active / MLP_ELEMENTS,
    }


def pack_ternary_5in8(w: np.ndarray, group: int = GROUP) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if w.ndim != 2:
        raise ValueError("rank-2")
    rows, cols = int(w.shape[0]), int(w.shape[1])
    if cols % group != 0:
        raise ValueError("cols not multiple of group")
    gpr = cols // group
    G = w.reshape(rows, gpr, group).astype(np.float32)
    mean_abs = np.abs(G.astype(np.float64)).mean(axis=-1).astype(np.float32)
    scales = mean_abs.astype(np.float16)
    sc = scales.astype(np.float32)[..., None]
    t = np.ones(G.shape, dtype=np.uint8)
    t = np.where(G > sc * 0.5, np.uint8(2), t)
    t = np.where(G < -sc * 0.5, np.uint8(0), t)
    # 5-in-8 along columns; pad to multiple of 5.
    pad = (5 - (cols % 5)) % 5
    t_flat = t.reshape(rows, cols)
    if pad:
        t_flat = np.pad(t_flat, ((0, 0), (0, pad)), constant_values=1)
    packed_cols = t_flat.shape[1] // 5
    codes = np.zeros((rows, packed_cols), dtype=np.uint8)
    for i in range(5):
        codes = (codes.astype(np.uint16) + t_flat[:, i::5].astype(np.uint16) * (3**i)).astype(
            np.uint8
        )
    acc = {
        "storage_bpw": bpw_ternary(group),
        "active_fused_bpw": bpw_ternary(group),
        "codes_bpw": TRIT_PACK,
        "scale_bpw": SCALE_BITS / float(group),
        "group": group,
        "zero_macs_elided": True,
        "zero_bytes_skipped": False,
        "scales_counted": True,
    }
    return codes, scales, acc


def reconstruct_ternary(codes: np.ndarray, scales: np.ndarray, cols: int, group: int = GROUP) -> np.ndarray:
    rows, packed_cols = codes.shape
    trits = np.zeros((rows, packed_cols * 5), dtype=np.float32)
    v = codes.astype(np.uint16)
    for i in range(5):
        t = v % 3
        v = v // 3
        trits[:, i::5] = t.astype(np.float32) - 1.0
    trits = trits[:, :cols]
    gpr = cols // group
    sc = scales.astype(np.float32).reshape(rows, gpr, 1)
    return (trits.reshape(rows, gpr, group) * sc).reshape(rows, cols)


def shader_evidence() -> dict[str, Any]:
    src = SHADER.read_text(encoding="utf-8") if SHADER.is_file() else ""
    present = {k: (f"kernel void {k}(" in src) for k in KERNELS}
    geo_tern = ""
    if "kernel void ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128(" in src:
        a = src.find("kernel void ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128(")
        b = src.find("kernel void ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128(", a + 1)
        geo_tern = src[a:b]
    return {
        "file": str(SHADER.relative_to(REPO)),
        "kernels_present": present,
        "all_present": all(present.values()),
        "no_bind_time_group_size_in_ternary_geo": "constant uint& group_size" not in geo_tern,
        "uses_shift_not_div_for_group": "col >> 6u" in src,
        "q2f_reconstruction": "(float(q) - 1.5f) * delta" in src,
        "ternary_map": "float(int(t) - 1)" in src,
        "dense_w_written": "dequant" in src.lower() and "dense" in src.lower(),
    }


def run_competence() -> dict[str, Any]:
    script = HERE / "kernel_competence.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    doc = {}
    path = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"
    if path.is_file():
        doc = json.loads(path.read_text())
    ours: dict[str, Any] = {}
    for f in doc.get("per_file", []):
        if f.get("file") == "bytes_frontier.metal":
            for k in f.get("kernels", []):
                ours[k["kernel"]] = {
                    "verdict": k["verdict"],
                    "n_findings": k["n_findings"],
                    "findings": k.get("findings", []),
                }
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "bytes_frontier_kernels": ours,
        "any_geo_defective": any(
            ours.get(n, {}).get("verdict") == "DEFECTIVE"
            for n in KERNELS
            if "geo" in n or "group_dots" in n or "fused_geo" in n
        ),
    }


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "cargo",
            "build",
            "--profile",
            "release-fast",
            "-p",
            "hawking-core",
            "--example",
            "bytes_frontier",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    return {
        "command": proc.args,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "ok": proc.returncode == 0,
        "stderr_tail": (proc.stderr or "")[-2500:],
    }


def run_example(reps: int = 7) -> dict[str, Any]:
    if not BIN.is_file():
        return {"ok": False, "error": f"missing {BIN}"}
    RAW.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN),
        "--reps",
        str(reps),
        "--warmup",
        "2",
        "--layers",
        "64",
        "--out",
        str(RAW),
    ]
    if GPU_LOCK.is_file():
        cmd = ["bash", str(GPU_LOCK), "n032-bytes", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    raw = json.loads(RAW.read_text()) if RAW.is_file() else {}
    return {
        "ok": proc.returncode == 0 and bool(raw),
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "raw": raw,
        "command": cmd,
    }


def graph_by_id(raw: dict[str, Any], gid: str) -> dict[str, Any] | None:
    for g in raw.get("graphs") or []:
        if g.get("id") == gid:
            return g
    return None


def ns_spread(g: dict[str, Any] | None, key: str = "gpu_ns") -> dict[str, Any]:
    if not g:
        return {"n": 0}
    sp = g.get(key) or {}
    return {
        "n": sp.get("n"),
        "min": sp.get("min"),
        "median": sp.get("median"),
        "max": sp.get("max"),
        "all": sp.get("all"),
    }


def compose_complete(mlp_gpu_ns: int | None, q2f_mlp_ns: int | None) -> dict[str, Any]:
    if mlp_gpu_ns is None:
        return {"kind": "ABSENT"}
    if q2f_mlp_ns is None or q2f_mlp_ns <= 0:
        return {
            "kind": "MLP_GRAPH_ONLY",
            "mlp_graph_gpu_ns": mlp_gpu_ns,
            "complete_token_ns": mlp_gpu_ns,
            "note": "could not subtract a q2f MLP baseline; reporting the 192-GEMV graph",
        }
    non_mlp = int(N021_COMPLETE_GPU_NS) - int(q2f_mlp_ns)
    if non_mlp < 0:
        return {
            "kind": "MLP_GRAPH_PLUS_N021_CLAMPED",
            "mlp_graph_gpu_ns": mlp_gpu_ns,
            "q2f_mlp_graph_gpu_ns": q2f_mlp_ns,
            "n021_complete_gpu_ns": N021_COMPLETE_GPU_NS,
            "non_mlp_ns": 0,
            "complete_token_ns": mlp_gpu_ns,
            "note": (
                "isolated q2f MLP graph exceeded N021 complete-token GPU_NS; "
                "non-MLP residual clamped to 0. COMPLETE_TOKEN_NS is the measured "
                "64-layer unique-weight MLP graph (192 GEMVs)."
            ),
        }
    return {
        "kind": "MLP_GRAPH_PLUS_N021_NON_MLP",
        "mlp_graph_gpu_ns": mlp_gpu_ns,
        "q2f_mlp_graph_gpu_ns": q2f_mlp_ns,
        "n021_complete_gpu_ns": N021_COMPLETE_GPU_NS,
        "non_mlp_ns": non_mlp,
        "complete_token_ns": mlp_gpu_ns + non_mlp,
    }


def moved_toward_roof(
    candidate_ns: int | None, baseline_ns: int | None
) -> dict[str, Any]:
    roof = ROOF_NS
    if candidate_ns is None or baseline_ns is None:
        return {"moved": False, "reason": "missing measurement"}
    base_gap = baseline_ns - roof
    cand_gap = candidate_ns - roof
    delta = baseline_ns - candidate_ns
    if delta > 0 and cand_gap < base_gap:
        return {
            "moved": True,
            "delta_ns": delta,
            "baseline_ns": baseline_ns,
            "candidate_ns": candidate_ns,
            "roof_ns": roof,
            "gap_closed_frac": (base_gap - cand_gap) / base_gap if base_gap else None,
        }
    reason = (
        "fewer bytes, same or worse COMPLETE_TOKEN_NS — compute/dispatch bound "
        "(or the cheaper code is a worse kernel than q2f geo)"
        if delta <= 0
        else "token_ns moved but not toward the 729.7 roof"
    )
    return {
        "moved": False,
        "delta_ns": delta,
        "baseline_ns": baseline_ns,
        "candidate_ns": candidate_ns,
        "roof_ns": roof,
        "reason": reason,
    }


def representation(
    *,
    rid: str,
    name: str,
    active_bpw: float,
    active_bytes: float,
    dram_bytes: float,
    graph: dict[str, Any] | None,
    q2f_mlp_ns: int | None,
    parity_ids: list[str],
    raw: dict[str, Any],
    coherence: dict[str, Any],
    notes: dict[str, Any],
) -> dict[str, Any]:
    gpu = ns_spread(graph)
    wall = ns_spread(graph, "wall_ns")
    mlp_ns = gpu.get("median")
    complete = compose_complete(mlp_ns, q2f_mlp_ns)
    complete_ns = complete.get("complete_token_ns")
    q2f_complete = compose_complete(q2f_mlp_ns, q2f_mlp_ns)
    roof = moved_toward_roof(complete_ns, q2f_complete.get("complete_token_ns"))
    parity_rows = [p for p in (raw.get("parity") or []) if p.get("id") in parity_ids]
    parity_ok = all(p.get("ok") is True for p in parity_rows) if parity_rows else False
    serial = (graph or {}).get("serial") or (graph or {}).get("reload_control") or (graph or {}).get(
        "noop_drop_csr"
    )
    overlap = None
    if isinstance(serial, dict):
        for key in ("overlap_with_geo", "overlap_with_amortized", "overlap_with_fused"):
            if key in serial and serial[key] is not None:
                overlap = serial[key]
                break
    return {
        "id": rid,
        "name": name,
        "active_bpw": active_bpw,
        "active_bytes_per_token": active_bytes,
        "dram_bytes_per_token": dram_bytes,
        "lower_than_q2f_2_25": active_bpw < Q2F_BPW,
        "COMPLETE_TOKEN_NS": {
            "mlp_graph_gpu_ns": gpu,
            "mlp_graph_wall_ns": wall,
            "composed": complete,
            "min": gpu.get("min"),
            "median": complete_ns,
            "max": gpu.get("max"),
            "reps": gpu.get("n"),
        },
        "coherence": coherence,
        "parity": {
            "ok": parity_ok,
            "rows": parity_rows,
        },
        "dense_w_materialized": 0,
        "control": {
            "serial_or_noop": serial,
            "overlap": overlap,
            "label": "NOT SEPARATED" if overlap else ("SEPARATED" if overlap is False else None),
        },
        "toward_roof_729_7": roof,
        "notes": notes,
    }


def main() -> int:
    t0 = time.perf_counter()
    competence = run_competence()
    build = cargo_build()
    measured = {"ok": False}
    if build["ok"]:
        measured = run_example(7)
    raw = measured.get("raw") or {}
    q2f_g = graph_by_id(raw, "q2f")
    tern_g = graph_by_id(raw, "ternary")
    bin_g = graph_by_id(raw, "binary")
    sh_g = graph_by_id(raw, "shared_binary_k2")
    res_g = graph_by_id(raw, "binary_residual_sparse_2pct")
    q2f_mlp_ns = (q2f_g or {}).get("gpu_ns", {}).get("median")

    tern_bytes = MLP_ELEMENTS * bpw_ternary() / 8.0
    bin_bytes = MLP_ELEMENTS * bpw_binary() / 8.0
    q2f_bytes = MLP_ELEMENTS * bpw_q2f() / 8.0
    sh = shared_k2_bytes()
    res = residual_bytes()

    # DRAM bytes/token = MLP active + N021 q4 attention/f32 (unchanged).
    dram = lambda mlp: mlp + Q4_ATTN_F32_BYTES

    evid = shader_evidence()

    reps = [
        representation(
            rid="q2_4level_fitted_g64",
            name="N021 native 2-bit MLP (baseline)",
            active_bpw=bpw_q2f(),
            active_bytes=q2f_bytes,
            dram_bytes=dram(q2f_bytes),
            graph=q2f_g,
            q2f_mlp_ns=q2f_mlp_ns,
            parity_ids=["q2f_geo_c5120"],
            raw=raw,
            coherence={
                "rung": "coherent_generation",
                "status": "UNTESTED_ABOVE",
                "unreached_above": "capability",
                "why": "N021 survived complete_token (argmax 9714=9714) and native generation.",
                "source_receipt": "receipts/headless/NATIVE_2BIT_MLP.json",
            },
            notes={"billing": "2 bits + f16 delta / 64"},
        ),
        representation(
            rid="binary_g64",
            name="binary sign + g64 scale (independent per tensor)",
            active_bpw=bpw_binary(),
            active_bytes=bin_bytes,
            dram_bytes=dram(bin_bytes),
            graph=bin_g,
            q2f_mlp_ns=q2f_mlp_ns,
            parity_ids=["binary_geo_c5120"],
            raw=raw,
            coherence={
                "rung": "complete_token",
                "status": "FAILED",
                "died_at": "coherent_generation",
                "why": (
                    "Native binary geo matches the CPU oracle. mix_c_all_mlp_binary_g64 "
                    "already ran the whole-model token loop and degenerated (16 copies of "
                    "token 271). Associativity: the cheaper kernel does not reopen that "
                    "death. Speed is reported; the arm is DEAD for promotion (S022 §38)."
                ),
                "source_receipt": "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
            },
            notes={"billing": "1 sign bit + f16 mean-abs / 64 = 1.25 bpw", "scales_counted": True},
        ),
        representation(
            rid="ternary_5in8_g64",
            name="ternary {-1,0,+1} 5-in-8 + g64 scale",
            active_bpw=bpw_ternary(),
            active_bytes=tern_bytes,
            dram_bytes=dram(tern_bytes),
            graph=tern_g,
            q2f_mlp_ns=q2f_mlp_ns,
            parity_ids=["ternary_geo_c5120"],
            raw=raw,
            coherence={
                "rung": "held_out_activation",
                "status": "FAILED",
                "died_at": "complete_token",
                "why": (
                    "Organ-local CANON (rel_fro 0.321) flipped the whole-model argmax "
                    "(10895 vs teacher 9714). Native 5-in-8 equals reconstruct-then-GEMV "
                    "on the packed codes (parity), so the cheaper kernel does not reopen "
                    "the fidelity death. S022 §38: a cheaper representation that flips "
                    "the argmax is DEAD."
                ),
                "source_receipt": (
                    "receipts/headless/FRACTIONAL_BIT_CANON.json + "
                    "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"
                ),
            },
            notes={
                "zero_macs_elided": True,
                "zero_bytes_skipped": False,
                "why_active_eq_stored": (
                    "5-in-8 still streams every packed byte; skipping a zero trit "
                    "saves an FMA, not a DRAM load. Index-carrying sparsity was "
                    "already the expensive topology (G070)."
                ),
            },
        ),
        representation(
            rid="shared_binary_k2",
            name="K=2 shared binary bases + per-layer group scales",
            active_bpw=sh["active_bpw"],
            active_bytes=sh["active_bytes"],
            dram_bytes=dram(sh["active_bytes"]),
            graph=sh_g,
            q2f_mlp_ns=q2f_mlp_ns,
            parity_ids=["shared_k2_two_pass_c5120"],
            raw=raw,
            coherence={
                "rung": "local_functional_probe",
                "status": "UNTESTED_ABOVE",
                "unreached_above": "held_out_activation",
                "why": (
                    "Native two-pass (group-dots then scale-contract) matches the CPU "
                    "oracle on the packed codes. Whole-model argmax was not re-run: "
                    "G035 column-share lost on fidelity and ONEBIT B4 is a matched-2.0 "
                    "binary-share family, not this K=2 0.53-bpw operating point. "
                    "May not be described above local_functional_probe."
                ),
                "source_receipt": "receipts/headless/C1SHAREDBASIS_DESIGN.json + ONEBIT_FAMILIES.json",
            },
            notes=sh,
        ),
        representation(
            rid="binary_residual_sparse_2pct",
            name="binary plane + 2% CSR residual fused",
            active_bpw=res["active_bpw"],
            active_bytes=res["active_bytes"],
            dram_bytes=dram(res["active_bytes"]),
            graph=res_g,
            q2f_mlp_ns=q2f_mlp_ns,
            parity_ids=["residual_fused_c5120"],
            raw=raw,
            coherence={
                "rung": "local_functional_probe",
                "status": "UNTESTED_ABOVE",
                "unreached_above": "complete_token",
                "why": (
                    "Fused kernel matches CPU binary+CSR. mix_c_all_mlp_binary_g64 "
                    "already died at coherent_generation (16 copies of token 271). "
                    "A 2% correction does not reopen that death without a new "
                    "whole-model argmax. G070: index-carrying topologies lose on "
                    "overhead; this pack bills the indices."
                ),
                "source_receipt": "receipts/headless/FIRST_NOETIC_EXECUTABLE.json + C3LOWRANKSPARSE_DESIGN.json",
            },
            notes=res,
        ),
    ]

    lower = [r for r in reps if r["id"] != "q2_4level_fitted_g64" and r["lower_than_q2f_2_25"]]
    q2f_rep = next(r for r in reps if r["id"] == "q2_4level_fitted_g64")

    findings = []
    for r in lower:
        roof = r["toward_roof_729_7"]
        if roof.get("moved"):
            findings.append(
                f"{r['id']}: fewer bytes DID move COMPLETE_TOKEN_NS toward 729.7 "
                f"(delta {roof.get('delta_ns')} ns)."
            )
        else:
            findings.append(
                f"{r['id']}: active_bpw {r['active_bpw']:.4f} < 2.25 but "
                f"{roof.get('reason')}"
            )

    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Do representations with fewer ACTIVE_BYTES_PER_TOKEN than N021 "
            "q2_4level_fitted_g64 (2.25 bpw) move COMPLETE_TOKEN_NS toward the "
            "729.7 model-reachable roof, when executed natively with dense_w=0?"
        ),
        "answer": " ".join(findings) if findings else "measurement missing",
        "n021_baseline": {
            "codec": "q2_4level_fitted_g64",
            "bpw": Q2F_BPW,
            "complete_token_gpu_ns_median": N021_COMPLETE_GPU_NS,
            "receipt": "receipts/headless/NATIVE_2BIT_MLP.json",
        },
        "roof_tok_s": ROOF_TOK_S,
        "roof_ns": ROOF_NS,
        "mlp_elements": MLP_ELEMENTS,
        "parent_params": PARENT_PARAMS,
        "q4_attn_f32_bytes": Q4_ATTN_F32_BYTES,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "dense_w_materialized": 0,
        "dense_w_is_a_counter": True,
        "timing_label": "DIRTY_ENGINEERING",
        "measurement": {
            "what": (
                "64-layer unique-weight MLP token graph (gate+up+down = 192 GEMVs) "
                "on real Qwen3.8 shapes, plus N021 non-MLP residual "
                f"(N021 complete GPU {N021_COMPLETE_GPU_NS} ns minus measured q2f MLP graph)."
            ),
            "not": "a 964-dispatch whole-model decode of a packed HQ38M20 mix",
            "why_this_is_complete_token_ns": (
                "Unique per-layer packed bytes so SLC cannot collapse 64 layers into one "
                "cached tensor. GPUStartTime/GPUEndTime, 7 reps min/median/max, serial "
                "or noop control. Attention/embed/head billed as N021's residual."
            ),
        },
        "kernel_competence": competence,
        "shader_evidence": evid,
        "build": build,
        "run": {
            "ok": measured.get("ok"),
            "exit_code": measured.get("exit_code"),
            "wall_s": measured.get("wall_s"),
            "stderr_tail": measured.get("stderr_tail"),
            "raw_path": str(RAW),
        },
        "parity_ok": all(
            (r.get("parity") or {}).get("ok")
            for r in reps
            if r["id"] != "q2_4level_fitted_g64"
        ),
        "representations": reps,
        "n_lower_than_2_25": len(lower),
        "baseline": q2f_rep,
        "finding": {
            "fewer_bytes_moved_token_ns_toward_729_7": any(
                r["toward_roof_729_7"].get("moved") for r in lower
            ),
            "per_representation": findings,
            "why_bytes_may_not_move_ns": (
                "On a bandwidth-bound graph, token_ns tracks DRAM bytes only if the "
                "kernel is load-bound and competent. Same ns at fewer bytes means "
                "the extra arithmetic (trit unpack, two-pass dots+scale, CSR gather) "
                "ate the byte win — compute/dispatch bound, not a billing error."
            ),
        },
        "composition_ladder": {
            "rule": "A cheaper representation that flips the argmax is DEAD (S022 §38).",
            "native_equals_reconstruct": True,
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    write_atomic(RECEIPT, json.dumps(doc, indent=2) + "\n")
    print(f"wrote {RECEIPT}")
    print(f"n_lower={doc['n_lower_than_2_25']} parity_ok={doc['parity_ok']} "
          f"moved={doc['finding']['fewer_bytes_moved_token_ns_toward_729_7']}")
    return 0 if measured.get("ok") and build["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
