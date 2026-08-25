#!/usr/bin/env python3
"""N035 SHARED_BASIS_COHERENT: K-sweep on the full 64-layer model + islands.

N033: the fused kernel is competent (K=2, 0.53 bpw, 24.55 ms < q2f 27.55 ms) but
K=2 dies at held_out_activation on 2 layers; K=8 (~2.13 bpw) was the only
healthy point on that pair. This lane finds the coherent operating point on
the WHOLE model, bills complete EBPW (bases+coefficients+islands, no hidden
bits), and measures COMPLETE_TOKEN_NS with the N033 fused kernel.

    python3 tools/headless/shared_basis_coherent.py
    python3 -m pytest tools/headless -q

Does not load a second 27B. Does not write under ~/models. Does not mutate
NOETIC_PARENT_A. Native only, dense_w=0. GPU serialized with
`bash tools/gpu_lane_lock.sh n035-sharedcoherent`.
"""
from __future__ import annotations

import gc
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bytes_frontier import (  # noqa: E402
    GROUP,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    MLP_ELEMENTS,
    N021_COMPLETE_GPU_NS,
    PARENT_PARAMS,
    Q4_ATTN_F32_BYTES,
    compose_complete,
    git_head,
    moved_toward_roof,
    now_iso,
    ns_spread,
    write_atomic,
)
from first_noetic_executable import judge_coherence  # noqa: E402
from kernel_competence import kernel_bodies, params_of, screen_kernel, strip_comments  # noqa: E402
from shared_basis_kernel import fused_bpw, SHADER  # noqa: E402

SCHEMA = "hawking.headless.shared_basis_coherent.v1"
RECEIPT = REPO / "receipts" / "headless" / "SHARED_BASIS_COHERENT.json"
RAW = REPO / "receipts" / "headless" / "_SHARED_BASIS_COHERENT_raw.json"
FIT_DIR = REPO / "artifacts" / "shared-basis-coherent"
CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "shared_basis_coherent"
if not BIN.is_file():
    alt = REPO / "workspace" / "ops" / "build" / "rust" / "release-fast" / "examples" / "shared_basis_coherent"
    if alt.is_file():
        BIN = alt
GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"

K_SWEEP = (2, 4, 8)
K_MAX = 8
# K=16 is measured on the fused kernel (COMPLETE_TOKEN_NS) but not fitted:
# 4.25 bpw already exceeds q2f 2.25, so it cannot be the density-winning
# coherent point. If K=8 fails composition, the island mix is the sub-2.25 path.
Q2F_BPW = 2.25
Q2F_COMPLETE_NS = N021_COMPLETE_GPU_NS
N032_Q2F_MLP_NS = 15_738_249
ORGANS = ("gate_proj", "up_proj", "down_proj")
PROBE_LAYERS = (0, 1, 3, 7, 15, 31, 47, 63)
ROW_SLICE = 1024
SHORT_TOKENS = 256
MAX_PROMPT = 16
MAX_NEW = 16

PRODUCTION_KERNELS = (
    "shared_binary_k2_fused_stream_c5120_tpr64_tg128",
    "shared_binary_k2_fused_stream_c17408_tpr64_tg128",
    "shared_binary_k4_fused_stream_c5120_tpr64_tg128",
    "shared_binary_k4_fused_stream_c17408_tpr64_tg128",
    "shared_binary_k8_fused_stream_c5120_tpr64_tg128",
    "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
    "shared_binary_k16_fused_stream_c5120_tpr64_tg128",
    "shared_binary_k16_fused_stream_c17408_tpr64_tg128",
)

PROMPT = (
    "Explain, in ordinary prose, how a compiler turns a for-loop into "
    "basic blocks and then into machine code."
)


def organ_shape(organ: str) -> tuple[int, int]:
    if organ in ("gate_proj", "up_proj"):
        return INTERMEDIATE, HIDDEN
    return HIDDEN, INTERMEDIATE


def row_block_for(organ: str) -> int:
    return 512 if organ == "down_proj" else 1024


def tensor_name(layer: int, organ: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.{organ}.weight"


def catalog_mlp_name(layer: int, organ: str) -> str:
    return f"language_model.model.layers.{layer}.mlp.{organ}.weight"


def complete_mlp_bytes(
    k: int,
    *,
    n_protected: int = 0,
    protected_bpw: float = Q2F_BPW,
    extra_correction_bytes: float = 0.0,
    n_layers: int = LAYERS,
    group: int = GROUP,
) -> dict[str, float]:
    """Complete MLP payload: shared signs + per-layer scales + islands.

    192 MLP tensors, equal element counts. Protected tensors are billed at
    `protected_bpw` (q2f = 2.25) instead of sharing the global bases' scales;
    the bases themselves stay (they still serve the unprotected tensors).
    """
    bill = fused_bpw(k, n_layers=n_layers, group=group)
    n_tensors = n_layers * 3
    elems_per = MLP_ELEMENTS / n_tensors
    n_prot = max(0, min(int(n_protected), n_tensors))
    n_share = n_tensors - n_prot
    # scales were n_layers * k * 2 * groups_all_three_organs = bill['scale_bytes']
    scale_per_tensor = bill["scale_bytes"] / n_tensors
    scale_bytes = scale_per_tensor * n_share
    prot_bytes = n_prot * elems_per * protected_bpw / 8.0
    active = bill["basis_sign_bytes"] + scale_bytes + prot_bytes + float(extra_correction_bytes)
    return {
        "k": float(k),
        "n_protected_tensors": float(n_prot),
        "n_shared_tensors": float(n_share),
        "basis_sign_bytes": bill["basis_sign_bytes"],
        "scale_bytes": float(scale_bytes),
        "protected_bytes": float(prot_bytes),
        "extra_correction_bytes": float(extra_correction_bytes),
        "active_bytes": float(active),
        "active_bpw": 8.0 * active / MLP_ELEMENTS,
        "dram_bytes_per_token": float(active + Q4_ATTN_F32_BYTES),
        "complete_ebpw": 8.0 * (active + Q4_ATTN_F32_BYTES) / PARENT_PARAMS,
        "protected_codec_bpw": float(protected_bpw),
    }


def shader_autopsy() -> dict[str, Any]:
    src = SHADER.read_text(encoding="utf-8") if SHADER.is_file() else ""
    stripped = strip_comments(src)
    kernels = []
    for name, body in kernel_bodies(stripped):
        r = screen_kernel(name, body, params_of(stripped, name))
        kernels.append(
            {
                "kernel": name,
                "verdict": r["verdict"],
                "n_findings": r["n_findings"],
                "findings": r["findings"],
            }
        )
    production = [k for k in kernels if k["kernel"] in PRODUCTION_KERNELS]
    present = {n: (f"kernel void {n}(" in src) for n in PRODUCTION_KERNELS}
    return {
        "file": str(SHADER.relative_to(REPO)),
        "n_kernels": len(kernels),
        "all_clear": all(k["verdict"] == "CLEAR" for k in kernels),
        "production_all_clear": all(k["verdict"] == "CLEAR" for k in production) and all(present.values()),
        "production_present": present,
        "kernels": kernels,
    }


def pack_pm1(b) -> Any:
    import numpy as np

    bits = (np.asarray(b) >= 0).astype(np.uint8).ravel()
    return np.packbits(bits, bitorder="little")


def unpack_pm1(packed, rows: int, cols: int):
    import numpy as np

    bits = np.unpackbits(np.ascontiguousarray(packed, dtype=np.uint8), bitorder="little")
    bits = bits[: rows * cols]
    return np.where(bits.reshape(rows, cols) > 0, np.float32(1.0), np.float32(-1.0))


def reconstruct_what(signs_k, scales_layer, k: int, rows: int, cols: int, group: int = GROUP):
    """What = sum_k scale[k,r,c//g] * B[k,r,c]. Dense W is a scoring vehicle."""
    import numpy as np

    gpr = cols // group
    what = np.zeros((rows, cols), dtype=np.float32)
    sc = np.asarray(scales_layer, dtype=np.float32)
    if sc.ndim == 3:
        sc = sc[:k, :rows, :]
    else:
        sc = sc.reshape(k, rows, gpr)
    # signs may be a full-organ plane; the leading rows*cols bits are row 0..rows.
    packed = np.ascontiguousarray(signs_k, dtype=np.uint8)
    if packed.ndim == 2:
        # [K, plane_full]; unpack each plane's leading rows.
        for ki in range(k):
            b = unpack_pm1(packed[ki], rows, cols)
            what += (b.reshape(rows, gpr, group) * sc[ki, :, :, None]).reshape(rows, cols)
            del b
        return what
    plane = (rows * cols + 7) // 8
    packed = packed.reshape(-1)
    for ki in range(k):
        b = unpack_pm1(packed[ki * plane : (ki + 1) * plane], rows, cols)
        what += (b.reshape(rows, gpr, group) * sc[ki, :, :, None]).reshape(rows, cols)
        del b
    return what


# ---------------------------------------------------------------------------
# Parent row reader (streamed, no second 27B)
# ---------------------------------------------------------------------------


class ParentReader:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self._hdr: dict[str, tuple[Path, int, dict]] = {}

    def _meta(self, name: str) -> tuple[Path, int, dict]:
        shard = self.weight_map[name]
        if shard not in self._hdr:
            path = self.root / shard
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            self._hdr[shard] = (path, n, header)
        path, n, header = self._hdr[shard]
        return path, n, header[name]

    def load_rows(self, name: str, row0: int, nrows: int):
        import numpy as np

        path, hlen, meta = self._meta(name)
        _rows, cols = meta["shape"]
        start = meta["data_offsets"][0]
        off = 8 + hlen + start + int(row0) * int(cols) * 2
        with open(path, "rb") as f:
            f.seek(off)
            raw = f.read(int(nrows) * int(cols) * 2)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return np.array(f32.reshape(int(nrows), int(cols)), dtype=np.float32, copy=True)

    def load(self, name: str):
        import numpy as np

        path, hlen, meta = self._meta(name)
        start, end = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(8 + hlen + start)
            raw = f.read(end - start)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return np.array(f32.reshape(meta["shape"]), dtype=np.float32, copy=True)


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
            "shared_basis_coherent",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
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
        cmd = ["bash", str(GPU_LOCK), "n035-sharedcoherent", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=2400,
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


# ---------------------------------------------------------------------------
# Fit (row-blocked, 64 layers jointly, K_MAX greedy then prefix joint-LS)
# ---------------------------------------------------------------------------


def _column_energy(cap, layers, fit_idx):
    from fractional_bit_canon import load_X

    import numpy as np

    ds = []
    for L in layers:
        x = load_X(cap, L)
        xf = x[fit_idx]
        d = (xf.astype(np.float64) ** 2).mean(axis=0).astype(np.float32)
        ds.append(d)
        del x, xf
        gc.collect()
    return ds


def fit_organ(parent: ParentReader, organ: str, ds, group: int = GROUP) -> dict[str, Any]:
    """Greedy K_MAX shared bases across 64 layers, then prefix joint-LS for K_SWEEP."""
    import numpy as np

    from onebit_families import _joint_group_ls, fit_shared_binary_bases

    rows, cols = organ_shape(organ)
    block = row_block_for(organ)
    plane = (rows * cols + 7) // 8
    gpr = cols // group
    signs = np.memmap(
        FIT_DIR / f"{organ}_signs_k{K_MAX}.u8",
        dtype=np.uint8,
        mode="w+",
        shape=(K_MAX, plane),
    )
    scales = {
        k: np.memmap(
            FIT_DIR / f"{organ}_scales_k{k}.f16",
            dtype=np.float16,
            mode="w+",
            shape=(LAYERS, k, rows, gpr),
        )
        for k in K_SWEEP
    }
    t0 = time.perf_counter()
    for row0 in range(0, rows, block):
        nrows = min(block, rows - row0)
        print(f"    {organ} rows {row0}:{row0+nrows} ...", flush=True)
        ws = []
        d_use = []
        for L in range(LAYERS):
            sl = parent.load_rows(tensor_name(L, organ), row0, nrows)
            ws.append(sl)
            d = ds[L]
            if d.shape[0] != cols:
                d_use.append(np.ones(cols, dtype=np.float32))
            else:
                d_use.append(d)
        whats, bases, _alphas = fit_shared_binary_bases(ws, d_use, K_MAX, group)
        del whats
        bit0 = row0 * cols
        if bit0 % 8 != 0:
            raise RuntimeError(f"{organ} row0*cols={bit0} not byte-aligned")
        byte0 = bit0 // 8
        for ki, b in enumerate(bases):
            packed = pack_pm1(b)
            signs[ki, byte0 : byte0 + packed.size] = packed
        for k in K_SWEEP:
            prefix = bases[:k]
            for L in range(LAYERS):
                _what, alpha = _joint_group_ls(ws[L], prefix, d_use[L], group)
                # alpha: [nrows, gpr, k]
                scales[k][L, :, row0 : row0 + nrows, :] = np.moveaxis(alpha, -1, 0).astype(
                    np.float16
                )
                del _what
        del ws, bases, d_use
        gc.collect()
    signs.flush()
    for s in scales.values():
        s.flush()
    return {
        "organ": organ,
        "rows": rows,
        "cols": cols,
        "k_max": K_MAX,
        "wall_s": time.perf_counter() - t0,
        "signs_path": str(FIT_DIR / f"{organ}_signs_k{K_MAX}.u8"),
        "scale_paths": {str(k): str(FIT_DIR / f"{organ}_scales_k{k}.f16") for k in K_SWEEP},
    }


def load_fit(organ: str, k: int):
    import numpy as np

    rows, cols = organ_shape(organ)
    plane = (rows * cols + 7) // 8
    gpr = cols // GROUP
    signs = np.memmap(
        FIT_DIR / f"{organ}_signs_k{K_MAX}.u8",
        dtype=np.uint8,
        mode="r",
        shape=(K_MAX, plane),
    )
    scales = np.memmap(
        FIT_DIR / f"{organ}_scales_k{k}.f16",
        dtype=np.float16,
        mode="r",
        shape=(LAYERS, k, rows, gpr),
    )
    return signs, scales


def fit_exists() -> bool:
    return all((FIT_DIR / f"{organ}_signs_k{K_MAX}.u8").is_file() for organ in ORGANS) and all(
        (FIT_DIR / f"{organ}_scales_k{k}.f16").is_file() for organ in ORGANS for k in K_SWEEP
    )


def run_fit(parent: ParentReader, cap, fit_idx) -> dict[str, Any]:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    print("column energy (streamed X, fit split) ...", flush=True)
    ds_gate = _column_energy(cap, range(LAYERS), fit_idx)
    t0 = time.perf_counter()
    organs = {}
    for organ in ORGANS:
        print(f"  fit {organ} K_MAX={K_MAX} across {LAYERS} layers ...", flush=True)
        d_use = ds_gate if organ != "down_proj" else [None] * LAYERS
        if organ == "down_proj":
            import numpy as np

            d_use = [np.ones(INTERMEDIATE, dtype=np.float32) for _ in range(LAYERS)]
        organs[organ] = fit_organ(parent, organ, d_use)
        gc.collect()
    return {"ok": True, "organs": organs, "wall_s": time.perf_counter() - t0, "streamed": True}


# ---------------------------------------------------------------------------
# Composition ladder on real X
# ---------------------------------------------------------------------------


def _healthy(sc: dict, rel_max: float, gain_min: float) -> bool:
    return bool(sc["rel_fro"] <= rel_max and sc["gain"] >= gain_min)


def score_held_out_organ(
    parent: ParentReader,
    cap,
    organ: str,
    k: int,
    layers,
    hold_idx,
    row_slice: int,
    rel_max: float,
    gain_min: float,
) -> dict[str, Any]:
    import numpy as np

    from fractional_bit_canon import score_pair

    signs, scales = load_fit(organ, k)
    rows, cols = organ_shape(organ)
    sl = min(row_slice, rows)
    rows_out = []
    from fractional_bit_canon import load_X as _lx
    from fractional_bit_canon import swiglu_intermediate

    for L in layers:
        if organ == "down_proj":
            x_full = _lx(cap, L)
            xh = x_full[hold_idx][:64]
            del x_full
            wg = parent.load(tensor_name(L, "gate_proj"))
            wu = parent.load(tensor_name(L, "up_proj"))
            mid = swiglu_intermediate(xh, wg, wu)
            w = parent.load_rows(tensor_name(L, organ), 0, sl)
            what = reconstruct_what(signs[:k], scales[L], k, sl, cols)
            y = mid @ w.T
            yh = mid @ what.T
            sc = score_pair(y, yh)
            del wg, wu, mid, w, what, y, yh, xh
        else:
            x_full = _lx(cap, L)
            xh = x_full[hold_idx]
            del x_full
            w = parent.load_rows(tensor_name(L, organ), 0, sl)
            what = reconstruct_what(signs[:k], scales[L], k, sl, cols)
            y = xh @ w.T
            yh = xh @ what.T
            sc = score_pair(y, yh)
            del w, what, y, yh, xh
        healthy = _healthy(sc, rel_max, gain_min)
        rows_out.append(
            {
                "layer": int(L),
                "k": k,
                "organ": organ,
                "row_slice": sl,
                "rel_fro": sc["rel_fro"],
                "cosine": sc["cosine"],
                "gain": sc["gain"],
                "beats_null": sc["beats_null"],
                "healthy": bool(healthy),
            }
        )
        gc.collect()
    rels = [r["rel_fro"] for r in rows_out if "rel_fro" in r]
    gains = [r["gain"] for r in rows_out if "gain" in r]
    import numpy as np

    mean_rel = float(np.mean(rels)) if rels else 1.0
    mean_gain = float(np.mean(gains)) if gains else 0.0
    return {
        "k": k,
        "organ": organ,
        "layers": rows_out,
        "mean_rel_fro": mean_rel,
        "mean_gain": mean_gain,
        "n_healthy": sum(1 for r in rows_out if r.get("healthy")),
        "n": len(rows_out),
        "healthy": bool(mean_rel <= rel_max and mean_gain >= gain_min and rels),
    }


def run_held_out(parent, cap, hold_idx, rel_max, gain_min) -> dict[str, Any]:
    curve = []
    localization = None
    for k in K_SWEEP:
        print(f"  held-out K={k} probe layers {PROBE_LAYERS} ...", flush=True)
        organs = {}
        held_organs = ("gate_proj", "up_proj")
        for organ in held_organs:
            organs[organ] = score_held_out_organ(
                parent, cap, organ, k, PROBE_LAYERS, hold_idx, ROW_SLICE, rel_max, gain_min
            )
        mean_rel = sum(organs[o]["mean_rel_fro"] for o in held_organs) / float(len(held_organs))
        mean_gain = sum(organs[o]["mean_gain"] for o in held_organs) / float(len(held_organs))
        healthy = all(organs[o]["healthy"] for o in held_organs)
        acc = complete_mlp_bytes(k)
        curve.append(
            {
                "k": k,
                "n_layers_fitted": LAYERS,
                "row_slice": ROW_SLICE,
                "group": GROUP,
                "mean_rel_fro": mean_rel,
                "mean_gain": mean_gain,
                "healthy": bool(healthy),
                "active_bpw": acc["active_bpw"],
                "complete_ebpw": acc["complete_ebpw"],
                "organs": organs,
            }
        )
        gc.collect()
    print("  K=2 all-layer gate localization ...", flush=True)
    localization = score_held_out_organ(
        parent, cap, "gate_proj", 2, list(range(LAYERS)), hold_idx, ROW_SLICE, rel_max, gain_min
    )
    first_fail = next((r["layer"] for r in localization["layers"] if not r.get("healthy")), None)
    localization["first_unhealthy_layer"] = first_fail
    return {"curve": curve, "k2_gate_all_layers": localization}


def run_short_chain(parent, cap, hold_idx, k: int, rel_max, gain_min) -> dict[str, Any]:
    """SwiGLU intermediate on layer 0, reconstructed gate+up, real X."""
    import numpy as np

    from fractional_bit_canon import load_X, score_pair, swiglu_intermediate

    print(f"  short_chain K={k} L0 SwiGLU ...", flush=True)
    x = load_X(cap, 0)
    xh = x[hold_idx][:SHORT_TOKENS]
    del x
    wg = parent.load(tensor_name(0, "gate_proj"))
    wu = parent.load(tensor_name(0, "up_proj"))
    sg, scg = load_fit("gate_proj", k)
    su, scu = load_fit("up_proj", k)
    rows, cols = organ_shape("gate_proj")
    what_g = reconstruct_what(sg[:k], scg[0], k, rows, cols)
    what_u = reconstruct_what(su[:k], scu[0], k, rows, cols)
    mid_t = swiglu_intermediate(xh, wg, wu)
    mid_s = swiglu_intermediate(xh, what_g, what_u)
    sc = score_pair(mid_t, mid_s)
    healthy = _healthy(sc, rel_max, gain_min)
    del wg, wu, what_g, what_u, mid_t, mid_s, xh
    gc.collect()
    return {
        "k": k,
        "organ": "swiglu_intermediate_gate_up",
        "layers": [0],
        "n_tokens": SHORT_TOKENS,
        "rel_fro": sc["rel_fro"],
        "cosine": sc["cosine"],
        "gain": sc["gain"],
        "beats_null": sc["beats_null"],
        "healthy": bool(healthy),
    }


def run_complete_organ(parent, cap, hold_idx, k: int, layer: int, rel_max, gain_min) -> dict[str, Any]:
    """Full MLP (gate, up, down) on real X at one layer."""
    from fractional_bit_canon import load_X, score_pair, swiglu_intermediate

    print(f"  complete_organ K={k} L{layer} ...", flush=True)
    x = load_X(cap, layer)
    xh = x[hold_idx][:SHORT_TOKENS]
    del x
    wg = parent.load(tensor_name(layer, "gate_proj"))
    wu = parent.load(tensor_name(layer, "up_proj"))
    wd = parent.load(tensor_name(layer, "down_proj"))
    sg, scg = load_fit("gate_proj", k)
    su, scu = load_fit("up_proj", k)
    sd, scd = load_fit("down_proj", k)
    rg, cg = organ_shape("gate_proj")
    rd, cd = organ_shape("down_proj")
    hg = reconstruct_what(sg[:k], scg[layer], k, rg, cg)
    hu = reconstruct_what(su[:k], scu[layer], k, rg, cg)
    hd = reconstruct_what(sd[:k], scd[layer], k, rd, cd)
    mid_t = swiglu_intermediate(xh, wg, wu)
    mid_s = swiglu_intermediate(xh, hg, hu)
    y_t = mid_t @ wd.T
    y_s = mid_s @ hd.T
    sc = score_pair(y_t, y_s)
    healthy = _healthy(sc, rel_max, gain_min)
    del wg, wu, wd, hg, hu, hd, mid_t, mid_s, y_t, y_s, xh
    gc.collect()
    return {
        "k": k,
        "layer": layer,
        "rel_fro": sc["rel_fro"],
        "cosine": sc["cosine"],
        "gain": sc["gain"],
        "beats_null": sc["beats_null"],
        "healthy": bool(healthy),
    }


def run_complete_token(parent: ParentReader, k: int, islands: set[tuple[int, str]] | None = None) -> dict[str, Any]:
    """64-layer residual + lm_head argmax. MLP = shared-basis (or q2f island)."""
    import numpy as np
    import torch

    from fractional_bit_canon import _fourlevel_fitted
    from noetic_composition import (
        ART,
        SRC,
        ArtifactQ4,
        SourceBF16,
        additive_causal,
        inject_weight,
        load_student_layer,
        load_teacher_layer,
        rmsnorm_delta,
        run_layer_sites,
        score,
        t_np,
        text_config,
        tokenize_prompt,
    )

    islands = islands or set()
    print(f"  complete_token K={k} islands={len(islands)} ...", flush=True)
    t0 = time.perf_counter()
    art = ArtifactQ4(ART)
    src = SourceBF16(SRC)
    token_ids, tok_info = tokenize_prompt()
    if len(token_ids) > MAX_PROMPT:
        token_ids = token_ids[:MAX_PROMPT]
        tok_info = dict(tok_info)
        tok_info["truncated_to"] = MAX_PROMPT
    cfg = text_config()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    t_emb = src.embed_rows(token_ids)
    s_emb = art.embed_rows(token_ids)
    x_teacher = torch.from_numpy(t_emb[None].copy())
    student_h = torch.from_numpy(s_emb[None].copy())
    bsz, seqlen, _ = x_teacher.shape
    pos = torch.arange(seqlen)[None]
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    cos, sin = rope(x_teacher, pos)
    pos_emb = (cos, sin)
    pad = torch.ones(bsz, seqlen, dtype=torch.bool)
    causal = additive_causal(seqlen, torch.float32)

    teacher_h = [x_teacher]
    depth = []
    first_fail_free = None
    fits = {organ: load_fit(organ, k) for organ in ORGANS}
    for L in range(LAYERS):
        t_layer = load_teacher_layer(src, cfg, L)
        sites_t = run_layer_sites(t_layer, teacher_h[-1], pos_emb, causal, pad, want_swiglu=False)
        teacher_h.append(sites_t["x_out"].contiguous())
        del t_layer, sites_t
        s_layer = load_student_layer(art, cfg, L)
        for organ in ORGANS:
            if (L, organ) in islands:
                w = parent.load(tensor_name(L, organ))
                what = _fourlevel_fitted(w, GROUP)
                del w
            else:
                signs, scales = fits[organ]
                rows, cols = organ_shape(organ)
                what = reconstruct_what(signs[:k], scales[L], k, rows, cols)
            inject_weight(s_layer, f"mlp.{organ}.weight", what)
            del what
        local = run_layer_sites(s_layer, teacher_h[L], pos_emb, causal, pad, want_swiglu=False)
        free = run_layer_sites(s_layer, student_h, pos_emb, causal, pad, want_swiglu=False)
        s_local = score(t_np(local["x_out"]), t_np(teacher_h[L + 1]), tag=f"L{L}.local")
        s_free = score(t_np(free["x_out"]), t_np(teacher_h[L + 1]), tag=f"L{L}.free")
        depth.append({"layer": L, "local": s_local, "free": s_free})
        if first_fail_free is None and not s_free["survives"]:
            first_fail_free = L
        student_h = free["x_out"].contiguous()
        del s_layer, local, free
        gc.collect()
        if L % 8 == 7 or L == LAYERS - 1:
            print(
                f"    L{L:02d} free rel_l2={s_free['rel_l2']:.4f} survives={s_free['survives']}",
                flush=True,
            )

    t_final_w = src.load("model.language_model.norm.weight")
    s_final_w = art.load("language_model.model.norm.weight")
    t_normed = rmsnorm_delta(t_np(teacher_h[-1]), t_final_w)
    s_normed = rmsnorm_delta(t_np(student_h), s_final_w)
    hid = score(s_normed, t_normed, tag="final_norm")

    def teacher_logits(hvec: np.ndarray) -> np.ndarray:
        name = "lm_head.weight"
        shard = src.weight_map[name]
        path, hlen, hdr = src._header(shard)
        meta = hdr[name]
        v, width = meta["shape"]
        row_bytes = width * 2
        base = 8 + hlen + meta["data_offsets"][0]
        hvec = np.asarray(hvec, dtype=np.float32).reshape(-1)
        logits = np.empty(v, dtype=np.float32)
        chunk = 4096
        with open(path, "rb") as f:
            for start in range(0, v, chunk):
                n = min(chunk, v - start)
                f.seek(base + start * row_bytes)
                raw = f.read(n * row_bytes)
                u16 = np.frombuffer(raw, dtype=np.uint16)
                w = (u16.astype(np.uint32) << 16).view(np.float32).reshape(n, width)
                logits[start : start + n] = w @ hvec
        return logits

    t_logits = teacher_logits(t_normed[0, -1])
    s_logits = art.lm_head_logits(s_normed[0, -1])
    t_arg = int(np.argmax(t_logits))
    s_arg = int(np.argmax(s_logits))
    logit_score = score(s_logits[None], t_logits[None], tag="lm_head")
    agree = t_arg == s_arg
    survives = bool(hid["survives"] and agree)
    return {
        "k": k,
        "n_islands": len(islands),
        "prompt_token_ids": token_ids,
        "tokenize": tok_info,
        "final_hidden": hid,
        "logits": logit_score,
        "teacher_argmax": t_arg,
        "student_argmax": s_arg,
        "argmax_agree": agree,
        "survives": survives,
        "first_fail_free_layer": first_fail_free,
        "free_rel_l2_path": [d["free"]["rel_l2"] for d in depth],
        "wall_s": time.perf_counter() - t0,
        "streamed_per_layer": True,
        "dense_w_materialized": 0,
        "note": "What reconstructed per layer as a scoring vehicle; executable is the fused kernel",
    }


def run_generation(parent: ParentReader, k: int, prompt_ids: list[int]) -> dict[str, Any]:
    """16 greedy tokens, student-only, re-prefill. Not a second 27B."""
    import numpy as np
    import torch

    from noetic_composition import (
        ART,
        SRC,
        ArtifactQ4,
        additive_causal,
        inject_weight,
        load_student_layer,
        run_layer_sites,
        text_config,
    )

    print(f"  generation K={k} 16 greedy ...", flush=True)
    t0 = time.perf_counter()
    art = ArtifactQ4(ART)
    cfg = text_config()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from tokenizers import Tokenizer

    tz = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    ids = list(prompt_ids)
    new_ids: list[int] = []
    fits = {organ: load_fit(organ, k) for organ in ORGANS}
    for step in range(MAX_NEW):
        s_emb = art.embed_rows(ids)
        h = torch.from_numpy(s_emb[None].copy())
        seqlen = h.shape[1]
        pos = torch.arange(seqlen)[None]
        rope = Qwen3_5TextRotaryEmbedding(cfg)
        cos, sin = rope(h, pos)
        pos_emb = (cos, sin)
        pad = torch.ones(1, seqlen, dtype=torch.bool)
        causal = additive_causal(seqlen, torch.float32)
        for L in range(LAYERS):
            layer = load_student_layer(art, cfg, L)
            for organ in ORGANS:
                signs, scales = fits[organ]
                rows, cols = organ_shape(organ)
                what = reconstruct_what(signs[:k], scales[L], k, rows, cols)
                inject_weight(layer, f"mlp.{organ}.weight", what)
                del what
            sites = run_layer_sites(layer, h, pos_emb, causal, pad, want_swiglu=False)
            h = sites["x_out"].contiguous()
            del layer, sites
            gc.collect()
        from noetic_composition import rmsnorm_delta

        s_final_w = art.load("language_model.model.norm.weight")
        normed = rmsnorm_delta(h.detach().float().cpu().numpy(), s_final_w)
        logits = art.lm_head_logits(normed[0, -1])
        nxt = int(np.argmax(logits))
        new_ids.append(nxt)
        ids.append(nxt)
        print(f"    gen step {step} -> {nxt}", flush=True)
        del h, logits
        gc.collect()
    text = tz.decode(new_ids)
    coh = judge_coherence(text, new_ids)
    return {
        "k": k,
        "n_new": len(new_ids),
        "token_ids": new_ids,
        "text": text,
        "coherence": coh,
        "wall_s": time.perf_counter() - t0,
        "vehicle": "python 64-layer re-prefill greedy; fused kernel is the executable",
    }


def island_curve(held: dict[str, Any]) -> dict[str, Any]:
    """Protect the worst K=2 (layer, organ) with q2f. Marginal EBPW vs health."""
    k2 = next((c for c in held["curve"] if c["k"] == 2), None)
    loc = held.get("k2_gate_all_layers") or {}
    ranked = []
    if k2:
        for organ, od in (k2.get("organs") or {}).items():
            for row in od.get("layers") or []:
                ranked.append(
                    {
                        "layer": row["layer"],
                        "organ": organ,
                        "rel_fro": row.get("rel_fro", 1.0),
                        "healthy": row.get("healthy", False),
                    }
                )
    # all-layer gate fills in the rest of the layers for gate.
    seen = {(r["layer"], r["organ"]) for r in ranked}
    for row in loc.get("layers") or []:
        key = (row["layer"], "gate_proj")
        if key not in seen and "rel_fro" in row:
            ranked.append(
                {
                    "layer": row["layer"],
                    "organ": "gate_proj",
                    "rel_fro": row["rel_fro"],
                    "healthy": row.get("healthy", False),
                }
            )
    ranked.sort(key=lambda r: -float(r["rel_fro"]))
    unhealthy = [r for r in ranked if not r["healthy"]]
    curve = []
    for n_prot in (0, 3, 6, 12, 24, 48, 96):
        bill = complete_mlp_bytes(2, n_protected=n_prot, protected_bpw=Q2F_BPW)
        prot = unhealthy[:n_prot]
        curve.append(
            {
                "n_protected_tensors": n_prot,
                "protected": prot[:12],
                "n_unhealthy_known": len(unhealthy),
                "active_bpw": bill["active_bpw"],
                "complete_ebpw": bill["complete_ebpw"],
                "active_bytes": bill["active_bytes"],
                "below_q2f_bpw": bill["active_bpw"] < Q2F_BPW,
                "note": (
                    "islands are q2f g64 on the worst held-out (layer, organ) at K=2. "
                    "capability-gain is whether mixed held-out / token loop heals; "
                    "n_protected counts tensors, not layers."
                ),
            }
        )
    return {
        "shared_k": 2,
        "island_codec": "q2f_g64_4level",
        "ranked_worst_first": ranked[:24],
        "n_unhealthy_probe": len(unhealthy),
        "earliest_k2_gate_fail_layer": loc.get("first_unhealthy_layer"),
        "marginal": curve,
    }


def climb(held: dict[str, Any], shorts: dict[int, dict], organs: dict[int, dict], token, gen) -> dict[str, Any]:
    first_healthy = next((c for c in held["curve"] if c.get("healthy")), None)
    k2 = next((c for c in held["curve"] if c["k"] == 2), None)
    if k2 and not k2["healthy"]:
        died = "held_out_activation"
        reached = "local_functional_probe"
        # maybe a higher K heals held-out
        if first_healthy:
            k = first_healthy["k"]
            short = shorts.get(k)
            if not short or not short.get("healthy"):
                died = "short_chain" if short else "short_chain"
                reached = "held_out_activation"
            else:
                organ = organs.get(k)
                if not organ or not organ.get("healthy"):
                    died = "complete_organ"
                    reached = "short_chain"
                elif not token or not token.get("survives"):
                    died = "complete_token"
                    reached = "complete_organ"
                elif not gen or not (gen.get("coherence") or {}).get("coherent"):
                    died = "coherent_generation"
                    reached = "complete_token"
                else:
                    died = None
                    reached = "coherent_generation"
        return {
            "rung": reached,
            "status": "FAILED" if died else "UNTESTED_ABOVE",
            "died_at": died,
            "unreached_above": None if died else "capability",
            "first_healthy_k_on_held_out": None if not first_healthy else first_healthy["k"],
            "why": (
                "K=2 on the 64-layer joint fit missed health on held-out real activations. "
                + (
                    f"First healthy K is {first_healthy['k']}."
                    if first_healthy
                    else "No K in {2,4,8} healed held-out on the full 64-layer joint fit. K=16 is 4.25 bpw > q2f 2.25 so it cannot win density."
                )
            ),
        }
    # K=2 healthy on full model (unexpected)
    k = 2
    short = shorts.get(k)
    if not short or not short.get("healthy"):
        return {
            "rung": "held_out_activation",
            "status": "FAILED",
            "died_at": "short_chain",
            "first_healthy_k_on_held_out": 2,
        }
    organ = organs.get(k)
    if not organ or not organ.get("healthy"):
        return {
            "rung": "short_chain",
            "status": "FAILED",
            "died_at": "complete_organ",
            "first_healthy_k_on_held_out": 2,
        }
    if not token or not token.get("survives"):
        return {
            "rung": "complete_organ",
            "status": "FAILED",
            "died_at": "complete_token",
            "first_healthy_k_on_held_out": 2,
        }
    if not gen or not (gen.get("coherence") or {}).get("coherent"):
        return {
            "rung": "complete_token",
            "status": "FAILED",
            "died_at": "coherent_generation",
            "first_healthy_k_on_held_out": 2,
        }
    return {
        "rung": "coherent_generation",
        "status": "UNTESTED_ABOVE",
        "died_at": None,
        "unreached_above": "capability",
        "first_healthy_k_on_held_out": 2,
    }


def main() -> int:
    try:
        from fractional_bit_canon import _ensure_torch

        _ensure_torch()
    except Exception:
        pass
    t0 = time.perf_counter()
    autopsy = shader_autopsy()
    from fractional_bit_canon import (
        GAIN_HEALTHY,
        REL_FRO_LOCAL_MAX,
        find_capture,
        find_parent,
        split_from_manifest,
        load_X,
    )

    parent_path = find_parent()
    cap = find_capture()
    parent = ParentReader(parent_path)
    x0 = load_X(cap, 0)
    fit_idx, hold_idx, _man, split_rule = split_from_manifest(cap, x0.shape[0])
    del x0
    gc.collect()

    FIT_DIR.mkdir(parents=True, exist_ok=True)
    if fit_exists() and os.environ.get("N035_REFIT") != "1":
        print("reusing cached 64-layer shared-basis fit", flush=True)
        fit_info = {"ok": True, "cached": True, "dir": str(FIT_DIR)}
    else:
        print("fitting shared bases on 64 layers (streamed row-blocks) ...", flush=True)
        fit_info = run_fit(parent, cap, fit_idx)

    print("held-out activation sweep ...", flush=True)
    held = run_held_out(parent, cap, hold_idx, REL_FRO_LOCAL_MAX, GAIN_HEALTHY)
    first_healthy = next((c for c in held["curve"] if c.get("healthy")), None)
    op_k = int(first_healthy["k"]) if first_healthy else 8
    for c in held["curve"]:
        print(
            f"    K={c['k']} mean_rel_fro={c['mean_rel_fro']:.4f} "
            f"mean_gain={c['mean_gain']:.4f} healthy={c['healthy']} "
            f"bpw={c['active_bpw']:.4f}",
            flush=True,
        )
    loc = held.get("k2_gate_all_layers") or {}
    print(
        f"    K=2 gate first_unhealthy_layer={loc.get('first_unhealthy_layer')} "
        f"n_healthy={loc.get('n_healthy')}/{loc.get('n')}",
        flush=True,
    )

    shorts: dict[int, dict] = {}
    organs_c: dict[int, dict] = {}
    for k in K_SWEEP:
        if k < op_k and not any(c["k"] == k and c["healthy"] for c in held["curve"]):
            # still run short chain on K=8 always, and on first_healthy
            if k != 8:
                continue
        try:
            shorts[k] = run_short_chain(parent, cap, hold_idx, k, REL_FRO_LOCAL_MAX, GAIN_HEALTHY)
        except Exception as exc:  # noqa: BLE001
            shorts[k] = {"k": k, "healthy": False, "error": str(exc)}
        if shorts[k].get("healthy") or k == op_k:
            try:
                organs_c[k] = run_complete_organ(
                    parent, cap, hold_idx, k, 0, REL_FRO_LOCAL_MAX, GAIN_HEALTHY
                )
            except Exception as exc:  # noqa: BLE001
                organs_c[k] = {"k": k, "healthy": False, "error": str(exc)}

    islands = island_curve(held)

    token = None
    gen = None
    climb_k = op_k
    organ_ok = (organs_c.get(climb_k) or {}).get("healthy")
    short_ok = (shorts.get(climb_k) or {}).get("healthy")
    # Always attempt complete_token at the hypothesized K=8 if anything at K=8
    # is close, or at first healthy K. A miss is a measured death.
    try:
        if short_ok or organ_ok or climb_k == 8:
            token = run_complete_token(parent, climb_k)
            if token.get("argmax_agree"):
                gen = run_generation(parent, climb_k, token["prompt_token_ids"])
    except Exception as exc:  # noqa: BLE001
        token = {"ok": False, "error": str(exc), "survives": False, "k": climb_k}

    # If K=8 (or first healthy) did not compose, try a small island mix: K=2 +
    # q2f on the worst 12 tensors, only if that mix still bills below 2.25.
    island_token = None
    mix12 = complete_mlp_bytes(2, n_protected=12)
    if (not token or not token.get("survives")) and mix12["active_bpw"] < Q2F_BPW:
        worst = (islands.get("ranked_worst_first") or [])[:12]
        prot = {(int(r["layer"]), str(r["organ"])) for r in worst}
        try:
            print("  island complete_token K=2 + 12 q2f ...", flush=True)
            island_token = run_complete_token(parent, 2, islands=prot)
        except Exception as exc:  # noqa: BLE001
            island_token = {"ok": False, "error": str(exc), "survives": False}

    print("kernel autopsy + COMPLETE_TOKEN_NS ...", flush=True)
    build = {"ok": BIN.is_file(), "note": "prebuilt" if BIN.is_file() else "missing"}
    if not RAW.is_file():
        if not BIN.is_file():
            build = cargo_build()
        measured = run_example(7) if (build.get("ok") or BIN.is_file()) else {"ok": False}
    else:
        measured = {
            "ok": True,
            "raw": json.loads(RAW.read_text()),
            "note": "reused existing GPU raw",
        }
    raw = measured.get("raw") or (json.loads(RAW.read_text()) if RAW.is_file() else {})

    k_ns = {}
    for k, gid in ((2, "fused_k2_192"), (4, "fused_k4_192"), (8, "fused_k8_192"), (16, "fused_k16_192")):
        g = graph_by_id(raw, gid)
        mlp = (g or {}).get("gpu_ns", {}).get("median")
        k_ns[k] = {
            "graph": gid,
            "mlp_graph_gpu_ns": ns_spread(g),
            "COMPLETE_TOKEN_NS": {
                "mlp_graph_gpu_ns": ns_spread(g),
                "composed": compose_complete(mlp, N032_Q2F_MLP_NS),
                "min": (g or {}).get("gpu_ns", {}).get("min"),
                "median": compose_complete(mlp, N032_Q2F_MLP_NS).get("complete_token_ns"),
                "max": (g or {}).get("gpu_ns", {}).get("max"),
                "reps": (g or {}).get("gpu_ns", {}).get("n"),
            },
            "dispatches": (g or {}).get("dispatches"),
            "kernels": (g or {}).get("kernels"),
        }

    op_bill = complete_mlp_bytes(climb_k)
    op_ns = k_ns.get(climb_k) or {}
    op_complete = (op_ns.get("COMPLETE_TOKEN_NS") or {}).get("median")
    beat_q2f_ns = isinstance(op_complete, (int, float)) and op_complete < Q2F_COMPLETE_NS
    beat_q2f_bpw = op_bill["active_bpw"] < Q2F_BPW
    coh = climb(held, shorts, organs_c, token, gen)
    coherent = coh.get("rung") == "coherent_generation" and coh.get("died_at") is None
    candidate = bool(coherent and beat_q2f_bpw and beat_q2f_ns)

    g8 = graph_by_id(raw, "fused_k8_192")
    extra = g8 or {}
    overlap_serial = extra.get("overlap_with_serial")
    overlap_noop = extra.get("overlap_with_noop")
    controls = {
        "serial": {
            "gpu_ns": ns_spread(graph_by_id(raw, "fused_k8_serial")),
            "overlap_with_fused": overlap_serial,
        },
        "noop": {
            "gpu_ns": ns_spread(graph_by_id(raw, "fused_k8_noop")),
            "overlap_with_fused": overlap_noop,
        },
        "overlap": bool(overlap_serial) or bool(overlap_noop),
        "label": (
            "NOT SEPARATED"
            if (overlap_serial or overlap_noop)
            else ("SEPARATED" if overlap_serial is False and overlap_noop is False else None)
        ),
    }

    parity_rows = raw.get("parity") or []
    fused_parity = [p for p in parity_rows if p.get("must_match") is True]
    parity_ok = bool(fused_parity) and all(p.get("ok") is True for p in fused_parity)
    noop_row = next((p for p in parity_rows if p.get("must_match") is False), None)
    noop_diverges = noop_row is not None and not noop_row.get("ok")

    token_ids = None
    if gen and gen.get("token_ids"):
        token_ids = {
            "prompt": (token or {}).get("prompt_token_ids"),
            "generated": gen.get("token_ids"),
            "teacher_argmax": (token or {}).get("teacher_argmax"),
            "student_argmax": (token or {}).get("student_argmax"),
        }
    elif token:
        token_ids = {
            "prompt": token.get("prompt_token_ids"),
            "teacher_argmax": token.get("teacher_argmax"),
            "student_argmax": token.get("student_argmax"),
            "argmax_agree": token.get("argmax_agree"),
        }

    if candidate:
        reason = (
            f"COHERENT shared-basis executable exists at K={climb_k}: "
            f"active_bpw={op_bill['active_bpw']:.6f} < {Q2F_BPW}, "
            f"COMPLETE_TOKEN_NS={op_complete} < {Q2F_COMPLETE_NS}. "
            f"Ladder reached {coh.get('rung')}."
        )
    else:
        why_not = []
        if not coherent:
            why_not.append(
                f"not coherent (rung={coh.get('rung')} died_at={coh.get('died_at')})"
            )
        if not beat_q2f_bpw:
            why_not.append(
                f"active_bpw={op_bill['active_bpw']:.6f} is not below q2f {Q2F_BPW}"
            )
        if not beat_q2f_ns:
            why_not.append(
                f"COMPLETE_TOKEN_NS={op_complete} is not below q2f {Q2F_COMPLETE_NS}"
            )
        reason = (
            "No coherent shared-basis point beats q2f on both density and ns. "
            + "; ".join(why_not)
            + ". Density/coherence/ns curve and K=2 divergence localization are in this receipt."
        )

    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": "N035",
        "question": (
            "Does a COHERENT shared-basis executable exist below q2f's 2.25 bpw "
            "that also beats q2f's 27.55 ms on the FULL 64-layer model, or a "
            "measured reason not?"
        ),
        "answer": reason,
        "coherent_shared_basis_beats_q2f": candidate,
        "operating_point": {
            "k": climb_k,
            "first_healthy_k_held_out": None if not first_healthy else first_healthy["k"],
            "active_bpw": op_bill["active_bpw"],
            "active_bytes_per_token": op_bill["active_bytes"],
            "dram_bytes_per_token": op_bill["dram_bytes_per_token"],
            "complete_ebpw": op_bill["complete_ebpw"],
            "COMPLETE_TOKEN_NS": op_ns.get("COMPLETE_TOKEN_NS"),
            "below_q2f_bpw": beat_q2f_bpw,
            "below_q2f_ns": beat_q2f_ns,
            "coherent": coherent,
            "promotable_frontier_candidate": candidate,
            "accounting": op_bill,
            "note": (
                "complete EBPW counts bases + per-layer coefficients + protected "
                "islands + extra corrections. Attention remains q4. dense_w=0."
            ),
        },
        "ACTIVE_BYTES_PER_TOKEN": op_bill["active_bytes"],
        "COMPLETE_TOKEN_NS": op_ns.get("COMPLETE_TOKEN_NS"),
        "k": climb_k,
        "k_sweep": K_SWEEP,
        "k_ns": {str(k): v for k, v in k_ns.items()},
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "dense_w_materialized": 0,
        "dense_w": 0,
        "native_only": True,
        "timing_label": "DIRTY_ENGINEERING",
        "kernel_autopsy": autopsy,
        "competent": bool(autopsy.get("production_all_clear") and autopsy.get("all_clear")),
        "build": build,
        "run": {
            "ok": measured.get("ok"),
            "exit_code": measured.get("exit_code"),
            "wall_s": measured.get("wall_s"),
            "stderr_tail": measured.get("stderr_tail"),
            "raw_path": str(RAW),
        },
        "parity": {
            "ok": parity_ok,
            "noop_diverges": noop_diverges,
            "rows": parity_rows,
        },
        "controls": controls,
        "q2f_baseline": {
            "complete_token_ns": Q2F_COMPLETE_NS,
            "mlp_graph_gpu_ns": N032_Q2F_MLP_NS,
            "bpw": Q2F_BPW,
            "receipt": "receipts/headless/NATIVE_2BIT_MLP.json + BYTES_FRONTIER.json",
        },
        "parent": str(parent_path),
        "capture": str(cap),
        "split_rule": split_rule,
        "streamed_per_tensor": True,
        "not_gaussian": True,
        "fit": fit_info,
        "held_out": held,
        "short_chain": shorts,
        "complete_organ": organs_c,
        "complete_token": token,
        "generation": gen,
        "token_ids": token_ids,
        "protected_islands": islands,
        "island_complete_token": island_token,
        "composition_ladder": coh,
        "toward_roof_729_7": moved_toward_roof(op_complete, Q2F_COMPLETE_NS),
        "finding": {
            "coherent_shared_basis_beats_q2f": candidate,
            "reason": reason,
            "k_where_held_out_heals": None if not first_healthy else first_healthy["k"],
            "k2_first_unhealthy_layer": (held.get("k2_gate_all_layers") or {}).get(
                "first_unhealthy_layer"
            ),
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    def _finite(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _finite(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_finite(v) for v in x]
        if isinstance(x, float):
            import math

            if math.isnan(x) or math.isinf(x):
                return None
        return x

    write_atomic(RECEIPT, json.dumps(_finite(doc), indent=2, allow_nan=False) + "\n")
    print(f"wrote {RECEIPT}")
    print(
        f"k={climb_k} coherent={coherent} bpw={op_bill['active_bpw']:.4f} "
        f"ns={op_complete} candidate={candidate} rung={coh.get('rung')}"
    )
    ok = autopsy.get("all_clear") and (measured.get("ok") or RAW.is_file()) and fit_info.get("ok")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
