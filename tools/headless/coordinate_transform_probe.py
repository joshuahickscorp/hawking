#!/usr/bin/env python3
"""N044 COORDINATE_TRANSFORM_PROBE (S026 §7-11, §78, §93, §117).

Cheap discriminator, CPU only: does a function-preserving coordinate
transform materially move the low-bit MLP COMPOSITION barrier that closed
at 2.25? The 2.25 result was measured in the un-rotated parameterization.
The information floor of a PARAMETERIZATION is not necessarily the floor
of the FUNCTION (S026 §117). A closed floor reopens only if the coordinate
system / family / healing / operator / accounting / objective changes
(S026 §11). This probe is that discriminator, before any GPU reopening.

Transforms (function-preserving: T then T^{-1} absorbed into weights):

  identity          no-op control; must reproduce the un-rotated baseline
  hadamard_b1024    block-diagonal Walsh-Hadamard (G032 tile; 0 stored bytes)
  pca_orth_b1024    learned block-orthogonal from fit-activation PCA
  bad_nonorth_b1024 deliberately non-orthogonal control; must not spuriously help

Codecs re-fit in each coordinate system:

  binary_g64   1.25 bpw (1 + f16 scale / 64) — N036 injured body
  ternary_g64  ~1.58 code bpw (log2(3)); 1.85 packed 5-in-8 + scale
  q2f_g64      2.25 bpw — the closed MLP floor, same fit as the survivor

Gate: held-out-activation composition on REAL post_attn_norm captures
(rel_fro / argmax agreement). A rotation that improves weight-space rel_fro
but not held-out composition does NOT reopen. Absorbed rotations bill 0
runtime bytes (S026 §9, §93); an online rotation would bill its cost.

    python3 tools/headless/coordinate_transform_probe.py
    python3 -m pytest tools/headless -q

Does not load a second 27B. Streams one parent tensor at a time from the
qualified BF16 (the FUNCTION) and reads ~/noetic/NOETIC_PARENT_A catalog
read-only. Does not mutate Parent A. Does not touch the GPU. Does not run
cargo or Metal. Dense W is a counter of materialised reconstructs, not a
candidate.
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bytes_frontier import (  # noqa: E402
    GROUP,
    HIDDEN,
    INTERMEDIATE,
    MLP_ELEMENTS,
    Q2F_BPW,
    git_head,
    now_iso,
    write_atomic,
)
from first_noetic_executable import PARENT_PARAMS  # noqa: E402
from fractional_bit_canon import (  # noqa: E402
    GAIN_HEALTHY,
    LOG2_3,
    REL_FRO_LOCAL_MAX,
    SCALE_AWARE_MARGIN,
    TRIT_PACK_5IN8,
    VISION_PY,
    codec_binary,
    codec_q2_4level,
    codec_ternary,
    find_capture,
    find_parent,
    j,
    load_X,
    load_tensor,
    rel_fro,
    score_pair,
    split_from_manifest,
    swiglu_intermediate,
    tensor_name,
    x_wt,
)

SCHEMA = "hawking.headless.coordinate_transform_probe.v1"
GENERATOR = "tools/headless/coordinate_transform_probe.py"
RECEIPT = REPO / "receipts" / "headless" / "COORDINATE_TRANSFORM_PROBE.json"
PARENT_A = Path(
    os.environ.get("NOETIC_PARENT_A_ROOT", str(Path.home() / "noetic" / "NOETIC_PARENT_A"))
)

BLOCK = 1024
HOLD_TOKENS = 512
FIT_TOKENS_PCA = 1024
PROBE_LAYERS = (0, 31)  # N036 earliest injury is L0; L31 is the canon deep probe
ORGANS = ("gate_proj", "up_proj", "down_proj")
N036_EARLIEST_LAYER = 0
N036_EARLIEST_ORGAN = "up_proj"
N036_WORST_ORGAN = "down_proj"

# G032 Q2 mean_delta_hold — a rotation at this scale did NOT reopen anything.
G032_Q2_DELTA_HOLD = 0.008204946837970573
# Material: larger than G032, and large enough to matter against the 0.50 bar.
MATERIAL_REL_FRO = 0.03
MATERIAL_ARGMAX = 0.05
BINARY_BPW = 1.0 + 16.0 / GROUP  # 1.25
TERNARY_CODE_BPW = LOG2_3  # ~1.58496, the S026 "~1.58"
TERNARY_PACKED_BPW = TRIT_PACK_5IN8 + 16.0 / GROUP  # 1.85
SCALE_TRAP = 0.01
SEED_PCA = 10582654
SEED_BAD = 20260824
FWHT_NORM = float(2.0 ** (-0.5 * int(math.log2(BLOCK))))


def _reexec_vision() -> None:
    if not VISION_PY.is_file():
        return
    try:
        if Path(sys.executable).resolve() == VISION_PY.resolve():
            return
    except OSError:
        return
    os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])


def parent_a_readonly() -> dict[str, Any]:
    """Census of the sealed parent. Read-only; never write, chmod, or decode."""
    cat = PARENT_A / "catalog.hq38m20"
    mix_path = PARENT_A / "MIX_REPORT.json"
    segs = PARENT_A / "segments"
    mix: dict[str, Any] = {}
    if mix_path.is_file():
        mix = json.loads(mix_path.read_text())
    n_seg = 0
    if segs.is_dir():
        n_seg = sum(1 for _ in segs.iterdir())
    st_mode = oct(PARENT_A.stat().st_mode) if PARENT_A.is_dir() else None
    return {
        "path": str(PARENT_A),
        "outside_worktree": "/worktrees/" not in str(PARENT_A),
        "catalog_present": cat.is_file(),
        "catalog_bytes": int(cat.stat().st_size) if cat.is_file() else 0,
        "n_segments": int(n_seg),
        "mix_id": mix.get("mix_id"),
        "mode": "read_only",
        "mutated": False,
        "st_mode": st_mode,
    }


def subsample(idx, n: int, seed: int = 0):
    import numpy as np

    idx = np.asarray(idx)
    if idx.size <= n:
        return idx
    rng = np.random.RandomState(seed)
    pick = np.sort(rng.choice(idx.size, size=n, replace=False))
    return idx[pick]


def argmax_agree(Y, Yh) -> float:
    import numpy as np

    if Y.shape[0] == 0:
        return float("nan")
    return float((np.argmax(Y, axis=1) == np.argmax(Yh, axis=1)).mean())


def survives(sc: dict[str, Any]) -> bool:
    return bool(
        float(sc.get("rel_fro", 1.0)) <= REL_FRO_LOCAL_MAX
        and float(sc.get("gain", 0.0)) >= GAIN_HEALTHY
        and bool(sc.get("beats_null"))
        and float(sc.get("scale_aware", 0.0)) >= SCALE_AWARE_MARGIN
    )


def score_composition(Y, Yh) -> dict[str, Any]:
    sc = score_pair(Y, Yh)
    sc["argmax_agree"] = argmax_agree(Y, Yh)
    sc["n_rows"] = int(Y.shape[0])
    sc["survives"] = survives(sc)
    sc["matches_scale_trap"] = bool(
        abs(float(sc["cosine"]) - 1.0) < 1e-5 and float(sc["gain"]) < 0.05
    )
    return sc


def codec_reconstruct(W, name: str):
    if name == "binary_g64":
        return codec_binary(W, g=GROUP)
    if name == "ternary_g64":
        return codec_ternary(W, g=GROUP)
    if name == "q2f_g64":
        return codec_q2_4level(W, g=GROUP)
    raise ValueError(name)


def codec_bpw(name: str) -> dict[str, float]:
    if name == "binary_g64":
        return {
            "code_bpw": 1.0,
            "storage_bpw": BINARY_BPW,
            "active_fused_bpw": BINARY_BPW,
            "s026_quote_bpw": 1.25,
        }
    if name == "ternary_g64":
        return {
            "code_bpw": float(TERNARY_CODE_BPW),
            "storage_bpw": float(TERNARY_PACKED_BPW),
            "storage_bpw_5in8": float(TERNARY_PACKED_BPW),
            "storage_bpw_entropy": float(TERNARY_CODE_BPW + 16.0 / GROUP),
            "active_fused_bpw": float(TERNARY_PACKED_BPW),
            "s026_quote_bpw": 1.58,
        }
    if name == "q2f_g64":
        return {
            "code_bpw": 2.0,
            "storage_bpw": Q2F_BPW,
            "active_fused_bpw": Q2F_BPW,
            "s026_quote_bpw": 2.25,
        }
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Structured transforms. Block size 1024 tiles hidden=5*1024 and
# intermediate=17*1024 with no padding (G032).
# ---------------------------------------------------------------------------


def assert_tiles(dim: int, block: int = BLOCK) -> None:
    if dim % block != 0:
        raise ValueError(f"dim {dim} does not tile block {block}")


def fwht_last(x):
    """Normalized Walsh-Hadamard on the last axis. Involutive for power-of-two."""
    import numpy as np

    x = np.array(x, dtype=np.float32, copy=True)
    n = int(x.shape[-1])
    if n == 0 or (n & (n - 1)):
        raise ValueError(f"fwht length {n} is not a power of two")
    h = 1
    stages = 0
    while h < n:
        y = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        h *= 2
        stages += 1
    x *= np.float32(2.0 ** (-0.5 * stages))
    return x


def apply_fwht_last_axis(M, block: int = BLOCK):
    import numpy as np

    M = np.ascontiguousarray(M, dtype=np.float32)
    *lead, d = M.shape
    assert_tiles(d, block)
    x = M.reshape(*lead, d // block, block)
    y = fwht_last(x)
    return np.ascontiguousarray(y.reshape(*lead, d))


def apply_fwht_first_axis(M, block: int = BLOCK):
    import numpy as np

    M = np.ascontiguousarray(M, dtype=np.float32)
    d = int(M.shape[0])
    assert_tiles(d, block)
    rest = M.shape[1:]
    x = M.reshape(d // block, block, int(np.prod(rest)) if rest else 1)
    x = np.moveaxis(x, 1, -1)
    y = fwht_last(x)
    y = np.moveaxis(y, -1, 1)
    return np.ascontiguousarray(y.reshape(d, *rest))


def block_rmatmul(M, blocks):
    """Right-multiply by block-diag(blocks): M @ T. Last axis of M is transformed."""
    import numpy as np

    M = np.ascontiguousarray(M, dtype=np.float32)
    b = int(blocks[0].shape[0])
    *lead, d = M.shape
    n_b = d // b
    if n_b != len(blocks) or d % b != 0:
        raise ValueError(f"dim {d} vs {len(blocks)} blocks of {b}")
    x = M.reshape(*lead, n_b, b)
    out = np.empty_like(x)
    for i, T in enumerate(blocks):
        out[..., i, :] = x[..., i, :] @ T
    return np.ascontiguousarray(out.reshape(*lead, d))


def block_lmatmul(M, blocks):
    """Left-multiply by block-diag(blocks): T @ M. First axis of M is transformed."""
    import numpy as np

    M = np.ascontiguousarray(M, dtype=np.float32)
    b = int(blocks[0].shape[0])
    d = int(M.shape[0])
    n_b = d // b
    if n_b != len(blocks) or d % b != 0:
        raise ValueError(f"dim {d} vs {len(blocks)} blocks of {b}")
    rest = M.shape[1:]
    width = int(np.prod(rest)) if rest else 1
    x = M.reshape(n_b, b, width)
    out = np.empty_like(x)
    for i, T in enumerate(blocks):
        out[i] = T @ x[i]
    return np.ascontiguousarray(out.reshape(d, *rest))


def orth_error_blocks(blocks) -> float:
    import numpy as np

    acc = 0.0
    n = 0
    eye = None
    for T in blocks:
        b = T.shape[0]
        if eye is None or eye.shape[0] != b:
            eye = np.eye(b, dtype=np.float32)
        g = T.T @ T
        acc += float(np.linalg.norm(g - eye) ** 2)
        n += b * b
    return math.sqrt(acc / max(n, 1))


def pca_blocks(X, dim: int, block: int = BLOCK, n_fit: int = FIT_TOKENS_PCA):
    """Learned block-orthogonal: eigenvectors of per-block activation covariance."""
    import numpy as np

    assert_tiles(dim, block)
    n_b = dim // block
    if X.shape[1] != dim:
        raise ValueError(f"PCA X dim {X.shape[1]} != {dim}")
    use = X
    if use.shape[0] > n_fit:
        step = max(1, use.shape[0] // n_fit)
        use = use[::step][:n_fit]
    Ts = []
    for i in range(n_b):
        xb = np.ascontiguousarray(use[:, i * block : (i + 1) * block], dtype=np.float32)
        mu = xb.mean(axis=0, keepdims=True)
        xc = xb - mu
        gram = xc.T @ xc
        _w, v = np.linalg.eigh(gram.astype(np.float64))
        # eigh returns ascending; reverse so leading energy is first. Sign
        # is free; flip a column so the max-abs entry is positive (stable).
        v = np.ascontiguousarray(v[:, ::-1], dtype=np.float32)
        peak = np.argmax(np.abs(v), axis=0)
        signs = np.sign(v[peak, np.arange(block)])
        signs = np.where(signs == 0, 1.0, signs).astype(np.float32)
        v = v * signs
        Ts.append(v)
    return Ts


def bad_blocks(dim: int, block: int = BLOCK, seed: int = SEED_BAD):
    """Block-diagonal I + 0.35 N(0,1). Invertible, not orthogonal."""
    import numpy as np

    assert_tiles(dim, block)
    rng = np.random.RandomState(seed + dim)
    n_b = dim // block
    Ts, Tinvs = [], []
    eye = np.eye(block, dtype=np.float32)
    for i in range(n_b):
        A = eye + np.float32(0.35) * rng.randn(block, block).astype(np.float32)
        Ts.append(A)
        Tinvs.append(np.linalg.inv(A.astype(np.float64)).astype(np.float32))
    return Ts, Tinvs


def hadamard_storage_bytes() -> int:
    """Sylvester-Hadamard is generated from its size. Zero stored bytes."""
    return 0


def dense_block_storage_bytes(dim: int, block: int = BLOCK, f16: bool = True) -> int:
    n_b = dim // block
    return n_b * block * block * (2 if f16 else 4)


@dataclass
class Transform:
    name: str
    kind: str
    orthogonal: bool
    structured: bool
    learned: bool
    involutive: bool
    function_preserving: bool
    absorbed_zero_runtime: bool
    control: str  # "candidate" | "noop" | "bad"
    block: int
    orth_error: float
    apply_last: Callable
    apply_last_inv: Callable
    apply_first: Callable
    apply_first_inv: Callable
    storage_bytes_if_not_absorbed: dict
    note: str


def make_identity() -> Transform:
    def ident(M):
        import numpy as np

        return np.ascontiguousarray(M, dtype=np.float32)

    return Transform(
        name="identity",
        kind="identity",
        orthogonal=True,
        structured=True,
        learned=False,
        involutive=True,
        function_preserving=True,
        absorbed_zero_runtime=True,
        control="noop",
        block=BLOCK,
        orth_error=0.0,
        apply_last=ident,
        apply_last_inv=ident,
        apply_first=ident,
        apply_first_inv=ident,
        storage_bytes_if_not_absorbed={"hidden": 0, "intermediate": 0},
        note="No-op control. Quantizing W I on X I must match the un-rotated fit exactly.",
    )


def make_hadamard() -> Transform:
    return Transform(
        name="hadamard_b1024",
        kind="hadamard",
        orthogonal=True,
        structured=True,
        learned=False,
        involutive=True,
        function_preserving=True,
        absorbed_zero_runtime=True,
        control="candidate",
        block=BLOCK,
        orth_error=0.0,
        apply_last=apply_fwht_last_axis,
        apply_last_inv=apply_fwht_last_axis,
        apply_first=apply_fwht_first_axis,
        apply_first_inv=apply_fwht_first_axis,
        storage_bytes_if_not_absorbed={"hidden": 0, "intermediate": 0},
        note=(
            "Block-diagonal Walsh-Hadamard, G032 tile B=1024. Generated from "
            "size, 0 stored bytes. Involutive: H=H^T=H^{-1}. Absorbed into W "
            "at pack time is zero-runtime (S026 §9)."
        ),
    )


def make_pca(X_hidden, X_inter=None) -> Transform:
    import numpy as np

    Th = pca_blocks(X_hidden, HIDDEN, BLOCK)
    Ti = pca_blocks(X_inter, INTERMEDIATE, BLOCK) if X_inter is not None else None
    Th_T = [t.T.copy() for t in Th]
    Ti_T = [t.T.copy() for t in Ti] if Ti is not None else None
    err_h = orth_error_blocks(Th)
    err_i = orth_error_blocks(Ti) if Ti is not None else 0.0

    def last(M):
        d = int(M.shape[-1])
        if d == HIDDEN:
            return block_rmatmul(M, Th)
        if d == INTERMEDIATE:
            if Ti is None:
                raise ValueError("pca transform has no intermediate blocks")
            return block_rmatmul(M, Ti)
        raise ValueError(d)

    def last_inv(M):
        d = int(M.shape[-1])
        if d == HIDDEN:
            return block_rmatmul(M, Th_T)
        if d == INTERMEDIATE:
            if Ti_T is None:
                raise ValueError("pca transform has no intermediate blocks")
            return block_rmatmul(M, Ti_T)
        raise ValueError(d)

    def first(M):
        d = int(M.shape[0])
        if d == HIDDEN:
            return block_lmatmul(M, Th)
        if d == INTERMEDIATE:
            if Ti is None:
                raise ValueError("pca transform has no intermediate blocks")
            return block_lmatmul(M, Ti)
        raise ValueError(d)

    def first_inv(M):
        d = int(M.shape[0])
        if d == HIDDEN:
            return block_lmatmul(M, Th_T)
        if d == INTERMEDIATE:
            if Ti_T is None:
                raise ValueError("pca transform has no intermediate blocks")
            return block_lmatmul(M, Ti_T)
        raise ValueError(d)

    return Transform(
        name="pca_orth_b1024",
        kind="learned_orthogonal",
        orthogonal=True,
        structured=True,
        learned=True,
        involutive=False,
        function_preserving=True,
        absorbed_zero_runtime=True,
        control="candidate",
        block=BLOCK,
        orth_error=float(max(err_h, err_i)),
        apply_last=last,
        apply_last_inv=last_inv,
        apply_first=first,
        apply_first_inv=first_inv,
        storage_bytes_if_not_absorbed={
            "hidden": dense_block_storage_bytes(HIDDEN),
            "intermediate": dense_block_storage_bytes(INTERMEDIATE),
        },
        note=(
            "Learned block-orthogonal: per-1024-block eigenvectors of the "
            "fit-activation covariance (SpinQuant-class, structured). "
            "Absorbed into W (and the neighbouring layer) is zero-runtime; "
            "if T were stored it would bill f16 block matrices under §93."
        ),
    )


def make_bad() -> Transform:
    Th, Th_inv = bad_blocks(HIDDEN, BLOCK, SEED_BAD)
    Ti, Ti_inv = bad_blocks(INTERMEDIATE, BLOCK, SEED_BAD)
    err_h = orth_error_blocks(Th)
    err_i = orth_error_blocks(Ti)
    Th_inv_T = [t.T.copy() for t in Th_inv]
    Ti_inv_T = [t.T.copy() for t in Ti_inv]
    Th_T = [t.T.copy() for t in Th]
    Ti_T = [t.T.copy() for t in Ti]

    def last(M):
        d = int(M.shape[-1])
        return block_rmatmul(M, Th if d == HIDDEN else Ti)

    def last_inv(M):
        # For W' = W T, X' = X T^{-T} so last-axis inv on activations is T^{-T}.
        d = int(M.shape[-1])
        return block_rmatmul(M, Th_inv_T if d == HIDDEN else Ti_inv_T)

    def last_true_inv(M):
        d = int(M.shape[-1])
        return block_rmatmul(M, Th_inv if d == HIDDEN else Ti_inv)

    def first(M):
        d = int(M.shape[0])
        return block_lmatmul(M, Th if d == HIDDEN else Ti)

    def first_inv(M):
        d = int(M.shape[0])
        return block_lmatmul(M, Th_inv if d == HIDDEN else Ti_inv)

    def apply_last_act(M):
        # Activations: X_rot = X @ T^{-T}
        return last_inv(M)

    # Store both: weights use T, activations use T^{-T}.
    t = Transform(
        name="bad_nonorth_b1024",
        kind="non_orthogonal",
        orthogonal=False,
        structured=True,
        learned=False,
        involutive=False,
        function_preserving=True,  # with explicit T^{-1}; healing billed
        absorbed_zero_runtime=False,
        control="bad",
        block=BLOCK,
        orth_error=float(max(err_h, err_i)),
        apply_last=last,
        apply_last_inv=apply_last_act,
        apply_first=first,
        apply_first_inv=first_inv,
        storage_bytes_if_not_absorbed={
            "hidden": 2 * dense_block_storage_bytes(HIDDEN),  # T and T^{-1}
            "intermediate": 2 * dense_block_storage_bytes(INTERMEDIATE),
        },
        note=(
            "Deliberately non-orthogonal I+0.35 N(0,1) blocks. Function-"
            "preserving only with an explicit inverse (healing accounting). "
            "Must not spuriously 'help' the composition scores. T and T^{-1} "
            "bill as stored bytes; they cannot be absorbed as an orthogonal."
        ),
    )
    t.apply_last_weight = last  # type: ignore[attr-defined]
    t.apply_last_true_inv = last_true_inv  # type: ignore[attr-defined]
    t.apply_last_T = last  # type: ignore[attr-defined]
    t.apply_last_T_T = lambda M: block_rmatmul(  # type: ignore[attr-defined]
        M, Th_T if int(M.shape[-1]) == HIDDEN else Ti_T
    )
    return t


def rotate_weight_input(W, T: Transform):
    """W_rot = W @ T  (columns / contraction axis)."""
    return T.apply_last(W)


def rotate_activation_for_weight(X, T: Transform):
    """X_rot such that X_rot @ (W T).T = X @ W.T when T is handled correctly.

    Orthogonal: X_rot = X @ T.
    Non-orthogonal: X_rot = X @ T^{-T}.
    Identity: X_rot = X.
    """
    if T.name == "bad_nonorth_b1024":
        return T.apply_last_inv(X)
    return T.apply_last(X)


def rotate_weight_output(W, T: Transform):
    """W_rot = T^T @ W (orthogonal) so y_rot = (mid @ W.T) @ T.

    Hadamard: H = H^T, so H @ W. Learned: apply_first_inv is T.T @ W.
    Bad control: T.T @ W via (W.T @ T).T; unrotate with T^{-1} on the last axis.
    """
    if T.kind in ("identity", "hadamard"):
        return T.apply_first(W)
    if T.kind == "learned_orthogonal":
        return T.apply_first_inv(W)
    if T.kind == "non_orthogonal":
        return T.apply_last(W.T).T
    return T.apply_first(W)


def unrotate_output(Y_rot, T: Transform):
    """Map y_rot back to the original hidden coordinates for scoring vs teacher."""
    if T.kind == "identity":
        return T.apply_last(Y_rot)
    if T.kind == "hadamard":
        return T.apply_last(Y_rot)
    if T.kind == "learned_orthogonal":
        return T.apply_last_inv(Y_rot)
    if T.kind == "non_orthogonal":
        # y_rot = y @ T  (because W_rot = T.T @ W => y_rot = y @ T)
        # y = y_rot @ T^{-1} = apply last-axis T^{-1}
        return T.apply_last_true_inv(Y_rot)  # type: ignore[attr-defined]
    return T.apply_last_inv(Y_rot)


def rotation_bill(T: Transform, absorbed: bool) -> dict[str, Any]:
    """Complete-EBPW tax of the transform (S026 §93). Absorbed => 0 runtime bytes."""
    hid = int(T.storage_bytes_if_not_absorbed.get("hidden", 0))
    inter = int(T.storage_bytes_if_not_absorbed.get("intermediate", 0))
    stored = 0 if absorbed else (hid + inter)
    # MLP-body extra bpw if T is stored as model-specific bytes.
    extra_bpw_mlp = (8.0 * stored / MLP_ELEMENTS) if stored else 0.0
    extra_ebpw = (8.0 * stored / PARENT_PARAMS) if stored else 0.0
    return {
        "name": T.name,
        "absorbed": bool(absorbed and T.absorbed_zero_runtime),
        "runtime_bytes": 0 if (absorbed and T.absorbed_zero_runtime) else stored,
        "storage_bytes_if_not_absorbed": hid + inter,
        "hidden_bytes_if_stored": hid,
        "intermediate_bytes_if_stored": inter,
        "extra_bpw_on_mlp": extra_bpw_mlp,
        "extra_complete_ebpw": extra_ebpw,
        "online_fwht": T.kind == "hadamard" and not absorbed,
        "note": (
            "Hadamard generated, 0 bytes. Learned T absorbed into W and the "
            "neighbour is 0 runtime bytes; storing T would bill f16 blocks. "
            "Non-orthogonal T cannot be absorbed as an orthogonal — T and "
            "T^{-1} count (S026 §93)."
            if T.kind != "identity"
            else "Identity stores nothing."
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic instruments (also used by tests). No parent tensors.
# ---------------------------------------------------------------------------


def synthetic_W(out: int, inn: int, seed: int = 0):
    import numpy as np

    rng = np.random.RandomState(seed)
    return rng.randn(out, inn).astype(np.float32)


def synthetic_X(n: int, inn: int, seed: int = 1):
    import numpy as np

    rng = np.random.RandomState(seed)
    return rng.randn(n, inn).astype(np.float32)


def function_preserve_error(T: Transform, out: int, inn: int, n: int = 32, seed: int = 3) -> float:
    """rel_fro between Y = X W^T and the T / T^{-1} round-trip, no quantize."""
    W = synthetic_W(out, inn, seed)
    X = synthetic_X(n, inn, seed + 1)
    Y = X @ W.T
    Wr = rotate_weight_input(W, T)
    Xr = rotate_activation_for_weight(X, T)
    Yh = Xr @ Wr.T
    return rel_fro(Y, Yh)


def identity_codec_error(codec: str, out: int = 64, inn: int = 128, n: int = 16) -> float:
    """Identity transform must match the un-rotated codec exactly (Y space)."""
    W = synthetic_W(out, inn, 7)
    X = synthetic_X(n, inn, 8)
    What, _ = codec_reconstruct(W, codec)
    T = make_identity()
    Wr = rotate_weight_input(W, T)
    Wq, _ = codec_reconstruct(Wr, codec)
    Xr = rotate_activation_for_weight(X, T)
    Y0 = X @ What.T
    Y1 = Xr @ Wq.T
    return rel_fro(Y0, Y1)


# ---------------------------------------------------------------------------
# Live probe
# ---------------------------------------------------------------------------


def _quant_pair(W, codec: str):
    return codec_reconstruct(W, codec)


def run_layer(
    parent: Path,
    cap: Path,
    layer: int,
    hold_idx,
    fit_idx,
    transforms: dict[str, Transform],
    codecs: tuple[str, ...],
) -> dict[str, Any]:
    import numpy as np

    print(f"  layer {layer} load X/W ...", flush=True)
    X_all = load_X(cap, layer)
    Xh = np.ascontiguousarray(X_all[hold_idx], dtype=np.float32)
    Xf = np.ascontiguousarray(X_all[fit_idx], dtype=np.float32)
    del X_all
    gc.collect()

    Wg = load_tensor(parent, tensor_name(layer, "gate_proj"))
    Wu = load_tensor(parent, tensor_name(layer, "up_proj"))
    print(f"  layer {layer} teacher SwiGLU hold={Xh.shape[0]} ...", flush=True)
    mid_h = swiglu_intermediate(Xh, Wg, Wu)
    # PCA of the intermediate on a fit subsample (no hold leakage).
    mid_f = swiglu_intermediate(Xf[: min(FIT_TOKENS_PCA, Xf.shape[0])], Wg, Wu)
    Wd = load_tensor(parent, tensor_name(layer, "down_proj"))
    Y_gate = x_wt(Xh, Wg)
    Y_up = x_wt(Xh, Wu)
    Y_down = x_wt(mid_h, Wd)
    Y_mlp = Y_down  # residual-stream contribution of the MLP (post-down)

    # Rebuild PCA with this layer's activations (learned, per-layer).
    transforms = dict(transforms)
    transforms["pca_orth_b1024"] = make_pca(Xf, mid_f)

    organs_w = {"gate_proj": Wg, "up_proj": Wu, "down_proj": Wd}
    organs_x = {"gate_proj": Xh, "up_proj": Xh, "down_proj": mid_h}
    organs_y = {"gate_proj": Y_gate, "up_proj": Y_up, "down_proj": Y_down}

    organ_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []

    for tname, T in transforms.items():
        print(f"    transform {tname} ...", flush=True)
        for organ in ORGANS:
            W = organs_w[organ]
            X = organs_x[organ]
            Yt = organs_y[organ]
            Wr = rotate_weight_input(W, T)
            Xr = rotate_activation_for_weight(X, T)
            # Exact (no quantize) round-trip — instrument, not a candidate.
            Y_exact = x_wt(Xr, Wr)
            preserve = score_composition(Yt, Y_exact)
            for codec in codecs:
                Wq, acc = _quant_pair(Wr, codec)
                Yh = x_wt(Xr, Wq)
                sc = score_composition(Yt, Yh)
                ws = rel_fro(Wr, Wq)  # weight-space in ROTATED coords
                # Q(W T) mapped back: Q(WT) T^{-1} in original coordinates.
                if T.kind == "identity":
                    Wq_orig = Wq
                elif T.kind == "hadamard":
                    Wq_orig = T.apply_last(Wq)
                elif T.kind == "learned_orthogonal":
                    Wq_orig = T.apply_last_inv(Wq)
                else:
                    Wq_orig = T.apply_last_true_inv(Wq)  # type: ignore[attr-defined]
                ws_back = rel_fro(W, Wq_orig)
                bill = codec_bpw(codec)
                rot_bill = rotation_bill(T, absorbed=T.absorbed_zero_runtime)
                organ_rows.append(
                    {
                        "layer": int(layer),
                        "organ": organ,
                        "transform": tname,
                        "codec": codec,
                        "control": T.control,
                        "n_hold": int(Xh.shape[0]),
                        "real_activations": True,
                        "not_gaussian": True,
                        "composition": sc,
                        "function_preserve_no_quant": {
                            "rel_fro": preserve["rel_fro"],
                            "argmax_agree": preserve["argmax_agree"],
                        },
                        "weight_space_rel_fro_rotated": ws,
                        "weight_space_rel_fro_original": ws_back,
                        "weight_space_is_not_the_gate": True,
                        "accounting": {**bill, **acc, "rotation": rot_bill},
                        "n036": {
                            "earliest_layer": organ == N036_EARLIEST_ORGAN
                            and layer == N036_EARLIEST_LAYER,
                            "worst_organ": organ == N036_WORST_ORGAN,
                        },
                    }
                )
                print(
                    f"      {organ:9s} {codec:12s} rel_fro={sc['rel_fro']:.4f} "
                    f"gain={sc['gain']:.3f} argmax={sc['argmax_agree']:.3f} "
                    f"survives={sc['survives']}",
                    flush=True,
                )
                del Wq, Yh, Wq_orig
                gc.collect()
            del Wr, Xr, Y_exact
            gc.collect()

        # Full MLP, absorbed hidden-dim T (zero-runtime if orthogonal).
        # Input rotation on gate/up; output rotation on down; score in original coords.
        for codec in codecs:
            Wg_r = rotate_weight_input(Wg, T)
            Wu_r = rotate_weight_input(Wu, T)
            Wd_r = rotate_weight_output(Wd, T)
            Wg_q, acc_g = _quant_pair(Wg_r, codec)
            Wu_q, acc_u = _quant_pair(Wu_r, codec)
            Wd_q, acc_d = _quant_pair(Wd_r, codec)
            Xr = rotate_activation_for_weight(Xh, T)
            mid_s = swiglu_intermediate(Xr, Wg_q, Wu_q)
            y_rot = x_wt(mid_s, Wd_q)
            y_hat = unrotate_output(y_rot, T)
            sc = score_composition(Y_mlp, y_hat)
            bill = codec_bpw(codec)
            rot_bill = rotation_bill(T, absorbed=T.absorbed_zero_runtime)
            mlp_rows.append(
                {
                    "layer": int(layer),
                    "organ": "mlp_swiglu_down",
                    "transform": tname,
                    "codec": codec,
                    "control": T.control,
                    "n_hold": int(Xh.shape[0]),
                    "real_activations": True,
                    "not_gaussian": True,
                    "absorbed_hidden_T": True,
                    "down_output_rotated": True,
                    "composition": sc,
                    "accounting": {
                        **bill,
                        "rotation": rot_bill,
                        "gate": acc_g,
                        "up": acc_u,
                        "down": acc_d,
                    },
                }
            )
            print(
                f"      MLP       {codec:12s} rel_fro={sc['rel_fro']:.4f} "
                f"gain={sc['gain']:.3f} argmax={sc['argmax_agree']:.3f} "
                f"survives={sc['survives']}",
                flush=True,
            )
            del Wg_r, Wu_r, Wd_r, Wg_q, Wu_q, Wd_q, Xr, mid_s, y_rot, y_hat
            gc.collect()

    # Scale-trap instrument on L0-style hold of gate teacher (once per layer).
    trap = score_composition(Y_gate, SCALE_TRAP * Y_gate)

    out = {
        "layer": int(layer),
        "n_hold": int(Xh.shape[0]),
        "n_fit_pca": int(Xf.shape[0]),
        "organs": organ_rows,
        "mlp": mlp_rows,
        "scale_trap": trap,
        "pca_orth_error": float(transforms["pca_orth_b1024"].orth_error),
        "bad_orth_error": float(transforms["bad_nonorth_b1024"].orth_error),
    }
    del Wg, Wu, Wd, Xh, Xf, mid_h, mid_f, Y_gate, Y_up, Y_down, Y_mlp
    gc.collect()
    return out


def _index_rows(rows: list[dict[str, Any]], *, transform: str, codec: str, organ: str | None = None):
    out = []
    for r in rows:
        if r["transform"] != transform or r["codec"] != codec:
            continue
        if organ is not None and r.get("organ") != organ:
            continue
        out.append(r)
    return out


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    xs = [float(r["composition"][key]) for r in rows]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def delta_row(id_rows, rot_rows, metric: str) -> dict[str, Any]:
    a = mean_metric(id_rows, metric)
    b = mean_metric(rot_rows, metric)
    return {"identity": a, "rotated": b, "delta": b - a}


def material_move(id_rows, rot_rows, q2f_id_rows) -> dict[str, Any]:
    """Composition-gated material-move test. Weight-space is ignored."""
    if not id_rows or not rot_rows:
        return {
            "moves": False,
            "reason": "missing rows",
            "crossed_bar": False,
            "material_rel_fro": False,
            "material_argmax": False,
        }
    id_rf = mean_metric(id_rows, "rel_fro")
    rot_rf = mean_metric(rot_rows, "rel_fro")
    id_am = mean_metric(id_rows, "argmax_agree")
    rot_am = mean_metric(rot_rows, "argmax_agree")
    id_sv = all(r["composition"]["survives"] for r in id_rows)
    rot_sv = all(r["composition"]["survives"] for r in rot_rows)
    d_rf = rot_rf - id_rf  # negative = better
    d_am = rot_am - id_am  # positive = better
    crossed = (not id_sv) and rot_sv
    mat_rf = d_rf <= -MATERIAL_REL_FRO
    mat_am = d_am >= MATERIAL_ARGMAX
    q2f_rf = mean_metric(q2f_id_rows, "rel_fro") if q2f_id_rows else float("nan")
    gap_id = id_rf - q2f_rf
    gap_rot = rot_rf - q2f_rf
    closed = (gap_id == gap_id) and gap_id > 0 and (gap_id - gap_rot) >= MATERIAL_REL_FRO
    # Weight-space-only improvement is explicitly not enough.
    moves = bool(crossed or (mat_rf and (mat_am or d_am >= -1e-6) and closed))
    reason = []
    if crossed:
        reason.append("rotated crosses local_survives while identity does not")
    if mat_rf:
        reason.append(f"rel_fro drop {d_rf:.4f} exceeds material {MATERIAL_REL_FRO}")
    if mat_am:
        reason.append(f"argmax_agree gain {d_am:.4f} exceeds material {MATERIAL_ARGMAX}")
    if closed:
        reason.append("closes a material fraction of the gap to unrotated q2f")
    if not reason:
        reason.append(
            f"rel_fro delta {d_rf:.4f}, argmax delta {d_am:.4f}; "
            f"neither crosses the bar nor exceeds G032-beating material thresholds"
        )
    return {
        "moves": moves,
        "crossed_bar": crossed,
        "material_rel_fro": mat_rf,
        "material_argmax": mat_am,
        "closed_gap_to_q2f": closed,
        "identity_survives": id_sv,
        "rotated_survives": rot_sv,
        "identity_rel_fro": id_rf,
        "rotated_rel_fro": rot_rf,
        "delta_rel_fro": d_rf,
        "identity_argmax_agree": id_am,
        "rotated_argmax_agree": rot_am,
        "delta_argmax_agree": d_am,
        "q2f_identity_rel_fro": q2f_rf,
        "reason": "; ".join(reason),
    }


def decide(layers_out: list[dict[str, Any]]) -> dict[str, Any]:
    mlp_all = [r for L in layers_out for r in L["mlp"]]
    organ_all = [r for L in layers_out for r in L["organs"]]
    candidates = ("hadamard_b1024", "pca_orth_b1024")
    codecs = ("binary_g64", "ternary_g64")
    per: list[dict[str, Any]] = []
    any_move = False
    any_spurious = False
    for codec in codecs:
        q2f_id = _index_rows(mlp_all, transform="identity", codec="q2f_g64")
        id_mlp = _index_rows(mlp_all, transform="identity", codec=codec)
        for tname in candidates:
            rot = _index_rows(mlp_all, transform=tname, codec=codec)
            m = material_move(id_mlp, rot, q2f_id)
            # Organ-local is supporting evidence, not the gate.
            organ_m = material_move(
                _index_rows(organ_all, transform="identity", codec=codec),
                _index_rows(organ_all, transform=tname, codec=codec),
                _index_rows(organ_all, transform="identity", codec="q2f_g64"),
            )
            rec = {
                "transform": tname,
                "codec": codec,
                "mlp_composition": m,
                "organ_local_not_the_gate": organ_m,
                "counts": m["moves"],
            }
            per.append(rec)
            any_move = any_move or m["moves"]
        bad = _index_rows(mlp_all, transform="bad_nonorth_b1024", codec=codec)
        bad_m = material_move(id_mlp, bad, q2f_id)
        per.append(
            {
                "transform": "bad_nonorth_b1024",
                "codec": codec,
                "mlp_composition": bad_m,
                "counts": False,
                "control": "bad",
                "note": "A non-orthogonal help is spurious and does not reopen.",
            }
        )
        if bad_m["moves"] and not any(
            p["counts"] for p in per if p["transform"] in candidates and p["codec"] == codec
        ):
            any_spurious = True

    identity_ok = True
    identity_notes = []
    for codec in ("binary_g64", "ternary_g64", "q2f_g64"):
        # identity vs itself is tautological; check function-preserve rel_fro ~ 0
        id_org = _index_rows(organ_all, transform="identity", codec=codec)
        pres = [
            float(r["function_preserve_no_quant"]["rel_fro"])
            for r in id_org
        ]
        if pres and max(pres) > 1e-5:
            identity_ok = False
            identity_notes.append(f"{codec} identity preserve rel_fro max {max(pres)}")
    if not identity_notes:
        identity_notes.append("identity function-preserve rel_fro <= 1e-5 on every organ")

    def _arm_line(p: dict[str, Any]) -> str:
        m = p["mlp_composition"]
        return (
            f"{p['transform']}/{p['codec']}: "
            f"Δrel_fro={m['delta_rel_fro']:+.4f} "
            f"({m['identity_rel_fro']:.4f}->{m['rotated_rel_fro']:.4f}) "
            f"Δargmax={m['delta_argmax_agree']:+.4f} "
            f"({m['identity_argmax_agree']:.4f}->{m['rotated_argmax_agree']:.4f})"
        )

    cand_lines = [
        _arm_line(p)
        for p in per
        if p["transform"] in candidates
    ]
    moves = bool(any_move) and not any_spurious
    frontier = "QWEN_MLP_ROTATED_TERNARY" if moves else None
    if moves:
        which = [
            f"{p['transform']}/{p['codec']}"
            for p in per
            if p.get("counts")
        ]
        answer = (
            "ROTATION_MOVES_BARRIER=true. A function-preserving rotation "
            f"materially moved held-out MLP composition for {which}. "
            "Measured MLP-composition deltas (L0+L31 mean, real hold activations): "
            + "; ".join(cand_lines)
            + ". Bounded reopening frontier: QWEN_MLP_ROTATED_TERNARY. "
            "QWEN_MLP_2_25 stays closed for the un-rotated family (S026 §11)."
        )
    else:
        answer = (
            "ROTATION_MOVES_BARRIER=false. Hadamard-structured and learned "
            "block-orthogonal rotations do not materially move held-out MLP "
            "composition for ternary (~1.58/1.85) or binary (1.25) versus the "
            "same fit in un-rotated coordinates. Measured MLP-composition "
            "deltas (L0+L31 mean, real hold activations): "
            + "; ".join(cand_lines)
            + f". Material bar is Δrel_fro<=-{MATERIAL_REL_FRO} (and closing "
            f"the gap to q2f) or crossing local_survives; G032's "
            f"{G032_Q2_DELTA_HOLD:.4f} hold delta is not material. "
            "Hadamard can lift organ-local ternary meanabs over the 0.50/0.50 "
            "GEMV bar without moving MLP composition — local is not the gate "
            "(N011). The 2.25 floor is coordinate-robust and stays closed. "
            "No GPU reopening is warranted on this discriminator (S026 §78)."
        )
    return {
        "ROTATION_MOVES_BARRIER": moves,
        "measured_deltas": cand_lines,
        "reopening_frontier": frontier,
        "QWEN_MLP_2_25_stays_closed_for_unrotated_family": True,
        "identity_reproduces_baseline": identity_ok,
        "identity_notes": identity_notes,
        "bad_control_spurious_help": any_spurious,
        "per_arm": per,
        "answer": answer,
        "material_thresholds": {
            "rel_fro_drop": MATERIAL_REL_FRO,
            "argmax_agree_gain": MATERIAL_ARGMAX,
            "g032_q2_delta_hold_not_material": G032_Q2_DELTA_HOLD,
            "note": (
                "Material means larger than G032's 0.008 hold delta AND large "
                "enough to matter against the 0.50 composition bar, with the "
                "gate on held-out composition not weight-space."
            ),
        },
    }


def build_receipt(live: dict[str, Any]) -> dict[str, Any]:
    decision = decide(live["layers"])
    trap = live["layers"][0]["scale_trap"] if live["layers"] else {}
    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": (
            "N044 — COORDINATE_TRANSFORM_PROBE (S026 §7-11, §78, §93, §117). "
            "Cheap discriminator: does a function-preserving coordinate "
            "transform materially move the low-bit COMPOSITION barrier that "
            "closed at 2.25?"
        ),
        "hand_authored": False,
        "python": sys.executable,
        "question": (
            "Does a function-preserving coordinate transform (Hadamard-"
            "structured rotation and one learned/structured orthogonal) "
            "materially move the MLP composition barrier at 2.25 / 1.58-"
            "ternary / 1.25-binary on high-sensitivity blocks, measured as "
            "held-out-activation rel_fro / argmax agreement on REAL "
            "activations versus the same fit in un-rotated coordinates?"
        ),
        "answer": decision["answer"],
        "ROTATION_MOVES_BARRIER": decision["ROTATION_MOVES_BARRIER"],
        "measured_deltas": decision["measured_deltas"],
        "reopening_frontier": decision["reopening_frontier"],
        "QWEN_MLP_2_25_stays_closed_for_unrotated_family": True,
        "s026": ["§7", "§8", "§9", "§10", "§11", "§78", "§93", "§117"],
        "n036": {
            "receipt": "receipts/headless/BINARY_HEALING.json",
            "earliest_layer": N036_EARLIEST_LAYER,
            "earliest_organ": N036_EARLIEST_ORGAN,
            "worst_organ": N036_WORST_ORGAN,
            "uniformly_injured": True,
            "probe_layers": list(PROBE_LAYERS),
            "note": (
                "N036: binary g64 death is UNIFORM across 64 layers, earliest "
                "at L0 up_proj, worst mean at down_proj. This probe takes L0 "
                "and L31 as representative high-sensitivity MLP blocks."
            ),
        },
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "did_not_write_ascent_or_campaign": True,
        "dense_w": 0,
        "dense_w_materialized": 0,
        "dense_w_is_a_counter": True,
        "parent_bf16": str(live["parent"]),
        "parent_a": live["parent_a"],
        "capture": live["capture"],
        "geometry": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "block": BLOCK,
            "hidden_blocks": HIDDEN // BLOCK,
            "intermediate_blocks": INTERMEDIATE // BLOCK,
            "group": GROUP,
        },
        "codecs": {
            "binary_g64": codec_bpw("binary_g64"),
            "ternary_g64": codec_bpw("ternary_g64"),
            "q2f_g64": codec_bpw("q2f_g64"),
        },
        "transforms": live["transform_meta"],
        "controls": {
            "noop": "identity — must reproduce the un-rotated baseline exactly",
            "bad": (
                "bad_nonorth_b1024 — random non-orthogonal block matrix; a "
                "help here is spurious and does not reopen"
            ),
            "identity_reproduces_baseline": decision["identity_reproduces_baseline"],
            "identity_notes": decision["identity_notes"],
            "bad_control_spurious_help": decision["bad_control_spurious_help"],
        },
        "quality_bar": {
            "name": "held_out_activation composition (N011 / S017 §28)",
            "gain_min": GAIN_HEALTHY,
            "rel_fro_max": REL_FRO_LOCAL_MAX,
            "scale_aware_min": SCALE_AWARE_MARGIN,
            "beats_null": True,
            "argmax_agree_reported": True,
            "weight_space_is_not_the_gate": True,
            "scale_trap_rejected": bool(
                trap.get("matches_scale_trap") or (trap.get("gain", 1) < 0.05)
            ),
            "scale_trap": trap,
        },
        "prior_science_not_rederived": {
            "g032_hadamard_q2_mean_delta_hold": G032_Q2_DELTA_HOLD,
            "g032_receipt": "receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q2.json",
            "c5_structured_transform": "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json",
            "c5_verdict": "NOT_WORTH_BUILDING as a GEMV replacement",
            "this_probe_is_not_g032": (
                "G032 scored a Hadamard codec-reparam at Q2/Q3/Q4 hold error. "
                "This probe re-fits ternary and binary in rotated coordinates "
                "and gates on held-out MLP composition, which G032 did not."
            ),
            "fractional_bit_canon": "receipts/headless/FRACTIONAL_BIT_CANON.json",
            "wholemodel_ternary_died": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
            "wholemodel_q2f_survived": "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
        },
        "material_thresholds": decision["material_thresholds"],
        "decision": decision,
        "layers": live["layers"],
        "elapsed_s": live["elapsed_s"],
        "self_check": live["self_check"],
    }


def self_check() -> dict[str, Any]:
    """Tiny synthetic instruments. Must pass before touching parent tensors."""
    import numpy as np

    T_id = make_identity()
    T_h = make_hadamard()
    inn, out = 1024, 64
    Xb = synthetic_X(80, inn, 12)
    err_id = function_preserve_error(T_id, out, inn, n=24, seed=3)
    err_h = function_preserve_error(T_h, out, inn, n=24, seed=3)
    W = synthetic_W(out, inn, 4)
    Ts = pca_blocks(Xb, inn, BLOCK)
    Wr = block_rmatmul(W, Ts)
    Xr = block_rmatmul(Xb[:24], Ts)
    Y = Xb[:24] @ W.T
    Yh = Xr @ Wr.T
    err_p = rel_fro(Y, Yh)
    T_b = make_bad()
    err_id_codec = identity_codec_error("binary_g64", out=32, inn=128, n=8)
    I = np.eye(BLOCK, dtype=np.float32)
    H = fwht_last(I)
    had_err = float(np.linalg.norm(H.T @ H - I) / math.sqrt(BLOCK * BLOCK))
    pca_err = orth_error_blocks(Ts)
    ok = (
        err_id < 1e-6
        and err_h < 1e-5
        and err_p < 1e-4
        and err_id_codec < 1e-6
        and had_err < 1e-5
        and T_b.orth_error > 0.1
        and pca_err < 1e-5
        and T_h.involutive
        and T_id.orth_error == 0.0
    )
    return {
        "ok": bool(ok),
        "identity_preserve_rel_fro": err_id,
        "hadamard_preserve_rel_fro": err_h,
        "pca1024_preserve_rel_fro": err_p,
        "identity_codec_rel_fro": err_id_codec,
        "hadamard_orth_error": had_err,
        "bad_orth_error": T_b.orth_error,
        "pca_orth_error": pca_err,
    }


def main() -> int:
    _reexec_vision()
    t0 = time.perf_counter()
    print("N044 coordinate-transform probe (CPU, no GPU)", flush=True)
    sc = self_check()
    print(f"  self_check ok={sc['ok']} {sc}", flush=True)
    if not sc["ok"]:
        raise SystemExit(f"synthetic self-check failed: {sc}")

    parent = find_parent()
    cap = find_capture()
    X0 = load_X(cap, 0)
    fit_idx, hold_idx, man, split_rule = split_from_manifest(cap, X0.shape[0])
    hold_use = subsample(hold_idx, HOLD_TOKENS, seed=17)
    fit_use = subsample(fit_idx, FIT_TOKENS_PCA, seed=19)
    print(
        f"  parent={parent}\n  capture={cap} n={X0.shape[0]} "
        f"fit={fit_use.size} hold={hold_use.size} split={split_rule}",
        flush=True,
    )
    del X0
    gc.collect()

    pa = parent_a_readonly()
    print(f"  parent_a catalog read-only n_segments={pa['n_segments']}", flush=True)

    # Static transforms; PCA is rebuilt per layer inside run_layer.
    T_id = make_identity()
    T_h = make_hadamard()
    T_b = make_bad()
    transforms_meta = {
        "identity": {
            "kind": T_id.kind,
            "orthogonal": True,
            "control": "noop",
            "absorbed_zero_runtime": True,
            "storage_bytes": 0,
            "note": T_id.note,
        },
        "hadamard_b1024": {
            "kind": T_h.kind,
            "orthogonal": True,
            "control": "candidate",
            "absorbed_zero_runtime": True,
            "storage_bytes": hadamard_storage_bytes(),
            "block": BLOCK,
            "note": T_h.note,
            "bill": rotation_bill(T_h, absorbed=True),
        },
        "pca_orth_b1024": {
            "kind": "learned_orthogonal",
            "orthogonal": True,
            "control": "candidate",
            "absorbed_zero_runtime": True,
            "storage_bytes_if_stored": dense_block_storage_bytes(HIDDEN)
            + dense_block_storage_bytes(INTERMEDIATE),
            "storage_bytes_absorbed": 0,
            "block": BLOCK,
            "note": (
                "Learned block-orthogonal from fit-activation PCA, rebuilt "
                "per layer. Absorbed => 0 runtime bytes."
            ),
        },
        "bad_nonorth_b1024": {
            "kind": T_b.kind,
            "orthogonal": False,
            "control": "bad",
            "absorbed_zero_runtime": False,
            "orth_error": T_b.orth_error,
            "storage_bytes": rotation_bill(T_b, absorbed=False)["runtime_bytes"],
            "note": T_b.note,
            "bill": rotation_bill(T_b, absorbed=False),
        },
    }

    # Placeholder PCA (replaced per layer). Need some X to construct the object
    # so the dict is well-typed; run_layer overwrites it.
    X_dummy = synthetic_X(8, HIDDEN, 0)
    mid_dummy = synthetic_X(8, INTERMEDIATE, 1)
    transforms = {
        "identity": T_id,
        "hadamard_b1024": T_h,
        "pca_orth_b1024": make_pca(X_dummy, mid_dummy),
        "bad_nonorth_b1024": T_b,
    }
    del X_dummy, mid_dummy

    codecs = ("binary_g64", "ternary_g64", "q2f_g64")
    layers_out = []
    for L in PROBE_LAYERS:
        layers_out.append(
            run_layer(parent, cap, L, hold_use, fit_use, transforms, codecs)
        )

    live = {
        "parent": str(parent),
        "parent_a": pa,
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm (real distribution)",
            "n_hold_used": int(hold_use.size),
            "n_fit_pca": int(fit_use.size),
            "hold_cap": HOLD_TOKENS,
            "split_rule": split_rule,
            "not_gaussian": True,
            "layers": list(PROBE_LAYERS),
        },
        "transform_meta": transforms_meta,
        "layers": layers_out,
        "self_check": sc,
        "elapsed_s": time.perf_counter() - t0,
    }
    doc = build_receipt(live)
    write_atomic(RECEIPT, json.dumps(j(doc), indent=2, sort_keys=False) + "\n")
    print(
        f"wrote {RECEIPT} ROTATION_MOVES_BARRIER={doc['ROTATION_MOVES_BARRIER']} "
        f"elapsed={doc['elapsed_s']:.1f}s",
        flush=True,
    )
    print(doc["answer"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
