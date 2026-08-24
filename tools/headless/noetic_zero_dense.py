#!/usr/bin/env python3
"""Zero-dense reconstruction: does the forward ever materialise parent W?

A representation that claims STRUCTURAL compression must never need to
materialise the dense parent tensor to run. If it reconstructs dense W and
then does an ordinary matvec, it is a storage format wearing a compression
costume: its STORAGE_BPW is fractional while its ACTIVE_BPW_CACHED is 16.

This harness decides that by MEASUREMENT, not by reading kernel comments.
For each structured representation that lives in this repo it packs the
structure, runs the forward under a TorchDispatchMode allocation probe, and
records the largest single *new* tensor. If any new allocation's numel
reaches the parent weight count, ZERO_DENSE=false.

Three BPW figures on every structure, never one:

  STORAGE_BPW        packed bytes on disk / parent weights
  ACTIVE_BPW_FUSED   bytes resident when the operator is fused (usually = storage)
  ACTIVE_BPW_CACHED  bytes resident if the dense f16 parent is materialised once (= 16)

Negative controls (both mandatory):

  * ordinary q4 dequant-then-matvec  -> ZERO_DENSE=false
  * a genuinely fused two-GEMM SVD   -> ZERO_DENSE=true

If the first comes back true, the detector is broken.

Does not load a 27B model. Work is per-tensor, synthetic factors at Qwen3.8
mlp.up_proj geometry (17408 x 5120). Torch via ~/.grok-vision/bin/python.

    python3 tools/headless/noetic_zero_dense.py
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
RECEIPT = ROOT / "receipts" / "headless" / "NOETIC_ZERO_DENSE.json"
SCHEMA = "hawking.headless.noetic_zero_dense.v1"

# Qwen3.8 mlp.up_proj.weight as stored: (out, in) = (intermediate, hidden).
ROWS = 17_408
COLS = 5_120
TOKENS = 4
GROUP = 64
Q4_CODE_BYTES = 32
Q4_SCALE_BYTES = 2
Q4_BYTES_PER_GROUP = Q4_CODE_BYTES + Q4_SCALE_BYTES  # 34
F16_BPW = 16.0
CACHE_DTYPE_BPW = 16.0
SVD_RANK = 12  # svd_w_r12_bpw0.05
HGRAVS_RANK = 160
SHARED_RANK = 64
LOW_RANK = 32
TT_INNER = 128
TT_RANKS = (8, 8, 8)  # r1, r2, r3
PQ_SUB = 32
PQ_CARD = 256
PQ_BITS = 8
BINARY_K = 2
HADAMARD_N = 4096
SPARSE_DENSITY = 0.005
SEED = 23
CHUNK_NNZ = 8192


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
        os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])
    sys.exit("torch required (tried sys python and ~/.grok-vision/bin/python)")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}
        if isinstance(x, torch.device):
            return str(x)
        if isinstance(x, torch.dtype):
            return str(x).replace("torch.", "")
    except Exception:
        pass
    try:
        return x.item()
    except Exception:
        return str(x)


def bpw_of(nbytes: int, n_w: int) -> float:
    return 8.0 * float(nbytes) / float(n_w)


def three_bpw(storage_bytes: int, n_w: int, fused_bytes: int | None = None) -> dict:
    stored = int(storage_bytes)
    fused = int(storage_bytes if fused_bytes is None else fused_bytes)
    return {
        "storage_bpw": bpw_of(stored, n_w),
        "active_bpw_fused": bpw_of(fused, n_w),
        "active_bpw_cached": CACHE_DTYPE_BPW,
        "storage_bytes": stored,
        "fused_resident_bytes": fused,
        "cached_f16_parent_bytes": int(n_w * 2),
    }


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _walk_tensors(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_tensors(v)


def _storage_ptr(t) -> int | None:
    try:
        if t.numel() == 0:
            return None
        return int(t.untyped_storage().data_ptr())
    except Exception:
        return None


class AllocProbe:
    """Record NEW storages produced by aten ops. Views of existing storage are
    not allocations. Parent materialisation = any new tensor with numel
    >= parent_numel (the dense weight count), any layout.
    """

    def __init__(self, parent_numel: int, parent_shape: tuple[int, int]):
        from torch.utils._python_dispatch import TorchDispatchMode

        self._mode_cls = TorchDispatchMode
        self.parent_numel = int(parent_numel)
        self.parent_shape = tuple(int(x) for x in parent_shape)
        self.peak_bytes = 0
        self.peak_shape: list[int] | None = None
        self.peak_numel = 0
        self.peak_op = None
        self.n_new = 0
        self.parent_hits: list[dict] = []
        self._mode = None

    def __enter__(self):
        probe = self

        class _Mode(self._mode_cls):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                before: set[int] = set()
                for t in _walk_tensors(args):
                    p = _storage_ptr(t)
                    if p:
                        before.add(p)
                for t in _walk_tensors(kwargs):
                    p = _storage_ptr(t)
                    if p:
                        before.add(p)
                out = func(*args, **kwargs)
                probe._record(out, str(func), before)
                return out

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        if self._mode is not None:
            return self._mode.__exit__(*exc)
        return False

    def _record(self, obj, op: str, before: set[int]) -> None:
        op_name = op.split(".")[-1][:48]
        for t in _walk_tensors(obj):
            ptr = _storage_ptr(t)
            if ptr is not None and ptr in before:
                continue
            nbytes = int(t.numel() * t.element_size())
            shape = [int(s) for s in t.shape]
            self.n_new += 1
            if nbytes > self.peak_bytes or (
                nbytes == self.peak_bytes and t.numel() > self.peak_numel
            ):
                self.peak_bytes = nbytes
                self.peak_shape = shape
                self.peak_numel = int(t.numel())
                self.peak_op = op_name
            if t.numel() >= self.parent_numel:
                self.parent_hits.append(
                    {
                        "shape": shape,
                        "numel": int(t.numel()),
                        "bytes": nbytes,
                        "dtype": str(t.dtype).replace("torch.", ""),
                        "op": op_name,
                        "matches_parent_shape": shape == list(self.parent_shape)
                        or shape == list(self.parent_shape[::-1]),
                    }
                )

    def snapshot(self) -> dict:
        return {
            "zero_dense": len(self.parent_hits) == 0,
            "peak_allocation_bytes": int(self.peak_bytes),
            "peak_allocation_shape": self.peak_shape,
            "peak_allocation_numel": int(self.peak_numel),
            "peak_allocation_op": self.peak_op,
            "n_new_tensors": int(self.n_new),
            "parent_shaped_allocations": self.parent_hits[:8],
            "n_parent_shaped_allocations": len(self.parent_hits),
        }


def measure(fn, parent_numel: int, parent_shape: tuple[int, int]) -> dict:
    import torch

    gc.collect()
    with torch.inference_mode():
        with AllocProbe(parent_numel, parent_shape) as probe:
            y = fn()
            # Keep y alive until the probe exits so a would-be-elided output
            # is still a real allocation. Then drop it.
            _ = y
    snap = probe.snapshot()
    del y
    gc.collect()
    return snap


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


def q4_pack(W, group: int = GROUP):
    """Offset-binary signed Q4, group-64, matching qwen_uniform_q4.metal."""
    import torch

    rows, cols = W.shape
    assert cols % group == 0
    n_groups = cols // group
    gW = W.reshape(rows, n_groups, group)
    absmax = gW.abs().amax(dim=-1).clamp_min(1e-8)
    scale = absmax / 7.0
    q = torch.round(gW / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16) + 8
    lo = q[:, :, 0::2].to(torch.uint8)
    hi = q[:, :, 1::2].to(torch.uint8)
    codes = (lo | (hi << 4)).contiguous()
    return codes, scale.to(torch.float16).contiguous()


def q4_dequant(codes, scales, group: int = GROUP):
    import torch

    rows, n_groups, _ = codes.shape
    lo = (codes & 0x0F).to(torch.float32) - 8.0
    hi = (codes.to(torch.int16) >> 4).to(torch.float32) - 8.0
    W = torch.stack((lo, hi), dim=-1).reshape(rows, n_groups * group)
    W = W * scales.to(torch.float32).unsqueeze(-1).expand(-1, -1, group).reshape(
        rows, n_groups * group
    )
    return W


def q4_fused_matvec(codes, scales, x, group: int = GROUP):
    import torch

    rows, n_groups, _ = codes.shape
    T = x.shape[0]
    y = torch.zeros(T, rows, dtype=x.dtype, device=x.device)
    sc = scales.to(dtype=x.dtype)
    for g in range(n_groups):
        packed = codes[:, g, :]
        lo = (packed & 0x0F).to(x.dtype) - 8
        hi = (packed.to(torch.int16) >> 4).to(x.dtype) - 8
        Wg = torch.stack((lo, hi), dim=-1).reshape(rows, group) * sc[:, g].unsqueeze(1)
        y = y + x[:, g * group : (g + 1) * group] @ Wg.t()
    return y


def svd_fused(x, U, V):
    """y = (x @ V) @ U.T   W ≈ U @ V.T, never formed."""
    return (x @ V) @ U.t()


def svd_reconstruct(x, U, V):
    W = U @ V.t()
    return x @ W.t()


def pq_fused_adc(x, codebooks, codes, sub: int):
    """Lookup-plus-accumulate. dots are (T, card), never (rows, cols)."""
    import torch

    T = x.shape[0]
    rows = codes.shape[0]
    n_sub = codebooks.shape[0]
    y = torch.zeros(T, rows, dtype=x.dtype, device=x.device)
    cb = codebooks.to(dtype=x.dtype)
    for s in range(n_sub):
        xs = x[:, s * sub : (s + 1) * sub]
        dots = xs @ cb[s].t()
        y = y + dots[:, codes[:, s].long()]
    return y


def pq_reconstruct(x, codebooks, codes, sub: int):
    import torch

    rows = codes.shape[0]
    n_sub = codebooks.shape[0]
    s_idx = torch.arange(n_sub, device=codes.device).view(1, n_sub).expand(rows, n_sub)
    gathered = codebooks[s_idx, codes.long()]
    W = gathered.reshape(rows, n_sub * sub).to(dtype=x.dtype)
    return x @ W.t()


def tt_fused(x, c0, c1, c2, c3, I0, I1, J0, J1):
    """4-core TT-GEMV, G1 contraction order, batch T. Never forms W.

    x → (T, J0, J1)
    A[t,j0,r3] = ∑_j1 x[t,j0,j1] c3[r3,j1]
    H[t,r2]    = ∑_{j0,r3} c2[r2,j0,r3] A
    C[t,r1,i1] = ∑_r2 c1[r1,i1,r2] H
    Y[t,i0,i1] = ∑_r1 c0[i0,r1] C
    """
    import torch

    T = x.shape[0]
    xr = x.reshape(T, J0, J1)
    A = xr @ c3.t()
    H = torch.einsum("rjk,tjk->tr", c2, A)
    C = torch.einsum("rik,tk->tri", c1, H)
    Y = torch.einsum("or,trj->toj", c0, C)
    return Y.reshape(T, I0 * I1)


def tt_reconstruct(x, c0, c1, c2, c3, I0, I1, J0, J1):
    import torch

    # c0[I0,r1] c1[r1,I1,r2] -> t01[r2,I0,I1]
    t01 = torch.einsum("or,ria->aoi", c0, c1)
    # c2[r2,J0,r3] c3[r3,J1] -> t23[r2,J0,J1]
    t23 = torch.einsum("rjb,bc->rjc", c2, c3)
    W = torch.einsum("aoi,ajc->oijc", t01, t23).reshape(I0 * I1, J0 * J1)
    return x @ W.t()


def fwht(x):
    """Normalised FWHT along the last dim. Working set is x-shaped, not n×n."""
    import torch

    n = x.shape[-1]
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(x.shape[0], n)
        h *= 2
    return y * (n ** -0.5)


def hadamard_matrix(n, device, dtype):
    import torch

    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        a = H
        H = torch.cat(
            [torch.cat([a, a], dim=1), torch.cat([a, -a], dim=1)],
            dim=0,
        )
    return H * (n ** -0.5)


def binary_fused(x, packed, scales, cols: int, tile: int = 64):
    """k binary planes, decoded in column tiles from packed bits. No parent W."""
    import torch

    k, rows, _nbytes = packed.shape
    T = x.shape[0]
    y = torch.zeros(T, rows, dtype=x.dtype, device=x.device)
    bit_pow = (2 ** torch.arange(8, device=x.device, dtype=torch.int32)).view(1, 1, 8)
    one = torch.ones((), dtype=x.dtype, device=x.device)
    for p in range(k):
        acc = torch.zeros(T, rows, dtype=x.dtype, device=x.device)
        for start in range(0, cols, tile):
            end = min(start + tile, cols)
            width = end - start
            b0 = start // 8
            b1 = (end + 7) // 8
            chunk = packed[p, :, b0:b1]
            mask = (chunk.to(torch.int32).unsqueeze(-1) & bit_pow) != 0
            signs = torch.where(mask, one, -one).reshape(rows, -1)[:, :width]
            acc = acc + x[:, start:end] @ signs.t()
        y = y + scales[p].to(dtype=x.dtype) * acc
    return y


def binary_reconstruct(x, packed, scales, cols: int):
    import torch

    k, rows, nbytes = packed.shape
    bit_pow = (2 ** torch.arange(8, device=x.device, dtype=torch.int32)).view(1, 1, 8)
    one = torch.ones((), dtype=x.dtype, device=x.device)
    W = torch.zeros(rows, nbytes * 8, dtype=x.dtype, device=x.device)
    for p in range(k):
        mask = (packed[p].to(torch.int32).unsqueeze(-1) & bit_pow) != 0
        plane = torch.where(mask, one, -one).reshape(rows, nbytes * 8)
        W = W + scales[p].to(dtype=x.dtype) * plane
    return x @ W[:, :cols].t()


def csr_fused(x, values, cols_idx, rows_idx, rows: int):
    import torch

    T = x.shape[0]
    y = torch.zeros(T, rows, dtype=x.dtype, device=x.device)
    nnz = values.numel()
    for start in range(0, nnz, CHUNK_NNZ):
        sl = slice(start, min(start + CHUNK_NNZ, nnz))
        contrib = x[:, cols_idx[sl]] * values[sl].to(x.dtype)
        y.index_add_(1, rows_idx[sl], contrib)
    return y


def lr_sparse_fused(x, U, V, values, cols_idx, rows_idx, rows: int):
    return svd_fused(x, U, V) + csr_fused(x, values, cols_idx, rows_idx, rows)


def lr_sparse_reconstruct(x, U, V, values, cols_idx, rows_idx):
    W = U @ V.t()
    W.index_put_((rows_idx, cols_idx), W[rows_idx, cols_idx] + values.to(W.dtype), accumulate=False)
    return x @ W.t()


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------


def _pack_record(id_, family, repo_anchor, parent_shape, n_w, storage_bytes,
                 fused_snap, recon_snap, extra=None, fused_bytes=None) -> dict:
    rec = {
        "id": id_,
        "family": family,
        "repo_anchor": repo_anchor,
        "parent_shape": list(parent_shape),
        "parent_numel": int(n_w),
        **three_bpw(storage_bytes, n_w, fused_bytes),
        "fused": fused_snap,
        "reconstruct_then_matvec": recon_snap,
        "zero_dense": fused_snap["zero_dense"],
        "zero_dense_fused": fused_snap["zero_dense"],
        "zero_dense_reconstruct": (
            None if recon_snap is None else recon_snap["zero_dense"]
        ),
        "peak_allocation_bytes_fused": fused_snap["peak_allocation_bytes"],
        "peak_allocation_bytes_reconstruct": (
            None if recon_snap is None else recon_snap["peak_allocation_bytes"]
        ),
    }
    if extra:
        rec.update(extra)
    return rec


def run_all(device) -> list[dict]:
    import torch

    torch.manual_seed(SEED)
    dt = torch.float32
    rows, cols, T = ROWS, COLS, TOKENS
    n_w = rows * cols
    parent = (rows, cols)
    x = torch.randn(T, cols, device=device, dtype=dt)

    out: list[dict] = []

    # ---- 1. uniform Q4 group-64 (production Qwen3.8 path + dequant control)
    print("  packing uniform_q4_group64", flush=True)
    W_src = torch.randn(rows, cols, device=device, dtype=dt)
    codes, scales = q4_pack(W_src)
    del W_src
    storage_q4 = int(codes.numel() + scales.numel() * Q4_SCALE_BYTES)
    print("  measuring uniform_q4_group64", flush=True)
    fused = measure(lambda: q4_fused_matvec(codes, scales, x), n_w, parent)
    recon = measure(lambda: x @ q4_dequant(codes, scales).t(), n_w, parent)
    out.append(
        _pack_record(
            "uniform_q4_group64",
            "grouped_absmax_q4",
            "crates/hawking-core/shaders/qwen_uniform_q4.metal :: qwen_uniform_q4_group64_matvec",
            parent,
            n_w,
            storage_q4,
            fused,
            recon,
            extra={
                "role": "negative_control_pair",
                "fused_is_production_kernel_algebra": True,
                "reconstruct_is_dequant_then_matvec": True,
                "q4_bytes_per_group": Q4_BYTES_PER_GROUP,
                "groups_per_row": cols // GROUP,
            },
        )
    )
    del codes, scales
    gc.collect()

    # ---- 2. svd_w_r12_bpw0.05 (cited DENSE_SUBBIT_TRANSFER structure)
    U = torch.randn(rows, SVD_RANK, device=device, dtype=dt)
    V = torch.randn(cols, SVD_RANK, device=device, dtype=dt)
    storage_svd = int((U.numel() + V.numel()) * 2)  # billed as f16 factors
    print("  measuring svd_w_r12_bpw0.05", flush=True)
    fused = measure(lambda: svd_fused(x, U, V), n_w, parent)
    recon = measure(lambda: svd_reconstruct(x, U, V), n_w, parent)
    out.append(
        _pack_record(
            "svd_w_r12_bpw0.05",
            "low_rank_svd",
            "receipts/headless/DENSE_SUBBIT_TRANSFER.json (activation-aware SVD factors)",
            parent,
            n_w,
            storage_svd,
            fused,
            recon,
            extra={
                "rank": SVD_RANK,
                "factor_bpw_formula": "16*k*(m+n)/(m*n)",
                "cited_storage_bpw": 0.0485,
                "cited_active_bpw_fused": 0.0485,
                "cited_active_bpw_cache_f16": 16.0,
            },
        )
    )
    del U, V
    gc.collect()

    # ---- 3. HGRAVS01 two-stage factor matvec (Q80 production algebra)
    L = torch.randn(rows, HGRAVS_RANK, device=device, dtype=dt)
    R = torch.randn(cols, HGRAVS_RANK, device=device, dtype=dt)
    storage_hg = int((L.numel() + R.numel()) * 2)
    print("  measuring hgravs01_factor_r160", flush=True)
    fused = measure(lambda: svd_fused(x, L, R), n_w, parent)
    recon = measure(lambda: svd_reconstruct(x, L, R), n_w, parent)
    out.append(
        _pack_record(
            "hgravs01_factor_r160",
            "low_rank_plus_sparse_correction",
            "crates/hawking-core/shaders/q80_mixed_decode.metal :: q80_hgravs01_factor_matvec",
            parent,
            n_w,
            storage_hg,
            fused,
            recon,
            extra={
                "rank": HGRAVS_RANK,
                "note": (
                    "Production kernel dequantises 3-bit factors in registers and "
                    "runs L@(R@x). This probe runs the two-GEMM algebra on f16-billed "
                    "factors; forming W = L@R.T is the reconstruct path."
                ),
            },
        )
    )
    del L, R
    gc.collect()

    # ---- 4. gravity_pq / fused ADC
    assert cols % PQ_SUB == 0
    n_sub = cols // PQ_SUB
    codebooks = torch.randn(n_sub, PQ_CARD, PQ_SUB, device=device, dtype=dt)
    pq_codes = torch.randint(0, PQ_CARD, (rows, n_sub), device=device, dtype=torch.int64)
    storage_pq = int(codebooks.numel() * 2 + pq_codes.numel() * (PQ_BITS // 8))
    print("  measuring gravity_pq_adc", flush=True)
    fused = measure(lambda: pq_fused_adc(x, codebooks, pq_codes, PQ_SUB), n_w, parent)
    recon = measure(lambda: pq_reconstruct(x, codebooks, pq_codes, PQ_SUB), n_w, parent)
    out.append(
        _pack_record(
            "gravity_pq_adc",
            "fused_dictionary_lookup_accumulate",
            "crates/hawking-core/shaders/gravity_pq.metal :: gravity_pq_matvec",
            parent,
            n_w,
            storage_pq,
            fused,
            recon,
            extra={
                "sub_dim": PQ_SUB,
                "card": PQ_CARD,
                "bits": PQ_BITS,
                "n_sub": n_sub,
            },
        )
    )
    del codebooks, pq_codes
    gc.collect()

    # ---- 5. tensor train, G1 4-core layout I1=J1=128
    I1 = J1 = TT_INNER
    assert rows % I1 == 0 and cols % J1 == 0
    I0, J0 = rows // I1, cols // J1
    r1, r2, r3 = TT_RANKS
    c0 = torch.randn(I0, r1, device=device, dtype=dt)
    c1 = torch.randn(r1, I1, r2, device=device, dtype=dt)
    c2 = torch.randn(r2, J0, r3, device=device, dtype=dt)
    c3 = torch.randn(r3, J1, device=device, dtype=dt)
    storage_tt = int((c0.numel() + c1.numel() + c2.numel() + c3.numel()) * 2)
    print("  measuring tensor_train_tt4_r8", flush=True)
    fused = measure(
        lambda: tt_fused(x, c0, c1, c2, c3, I0, I1, J0, J1), n_w, parent
    )
    recon = measure(
        lambda: tt_reconstruct(x, c0, c1, c2, c3, I0, I1, J0, J1), n_w, parent
    )
    out.append(
        _pack_record(
            "tensor_train_tt4_r8",
            "tensor_contraction",
            "G1 tt_gemv_f16 / receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json",
            parent,
            n_w,
            storage_tt,
            fused,
            recon,
            extra={
                "I0": I0, "I1": I1, "J0": J0, "J1": J1,
                "ranks": list(TT_RANKS),
                "named_kernel_in_shaders": False,
            },
        )
    )
    del c0, c1, c2, c3
    gc.collect()

    # ---- 6. binary planes k=2
    nbytes = (cols + 7) // 8
    packed = torch.randint(0, 256, (BINARY_K, rows, nbytes), device=device, dtype=torch.uint8)
    bscales = torch.randn(BINARY_K, device=device, dtype=dt)
    storage_bin = int(packed.numel() + bscales.numel() * 2)
    print("  measuring binary_planes_k2", flush=True)
    fused = measure(lambda: binary_fused(x, packed, bscales, cols), n_w, parent)
    recon = measure(lambda: binary_reconstruct(x, packed, bscales, cols), n_w, parent)
    out.append(
        _pack_record(
            "binary_planes_k2",
            "structured_transform",
            "crates/hawking-core/shaders/qwen_uniform_q4.metal :: qwen_binary_planes_k*_matvec",
            parent,
            n_w,
            storage_bin,
            fused,
            recon,
            extra={"k": BINARY_K, "bits_per_weight": BINARY_K},
        )
    )
    del packed, bscales
    gc.collect()

    # ---- 7. implicit FWHT vs materialised Hadamard (own parent n×n)
    nH = HADAMARD_N
    n_w_h = nH * nH
    parent_h = (nH, nH)
    xh = torch.randn(T, nH, device=device, dtype=dt)
    storage_h = 0  # generated / implicit: the transform is the code
    print("  measuring fwht_hadamard_n4096", flush=True)
    fused = measure(lambda: fwht(xh), n_w_h, parent_h)
    recon = measure(lambda: xh @ hadamard_matrix(nH, device, dt).t(), n_w_h, parent_h)
    out.append(
        _pack_record(
            "fwht_hadamard_n4096",
            "structured_transform",
            "tools/gravity_xform_hadamard.py / G032_XFORM_HADAMARD_Q4",
            parent_h,
            n_w_h,
            storage_h,
            fused,
            recon,
            extra={
                "n": nH,
                "generated": True,
                "note": "STORAGE_BPW=0: the operator is the butterfly, not a stored H.",
            },
        )
    )
    del xh
    gc.collect()

    # ---- 8. low-rank + sparse correction (C3 family)
    U = torch.randn(rows, LOW_RANK, device=device, dtype=dt)
    V = torch.randn(cols, LOW_RANK, device=device, dtype=dt)
    nnz = max(1, int(SPARSE_DENSITY * n_w))
    rows_idx = torch.randint(0, rows, (nnz,), device=device)
    cols_idx = torch.randint(0, cols, (nnz,), device=device)
    values = torch.randn(nnz, device=device, dtype=dt)
    storage_lrs = int((U.numel() + V.numel()) * 2 + nnz * (4 + 4 + 4))
    print("  measuring lowrank_r32_plus_sparse_0p5pct", flush=True)
    fused = measure(
        lambda: lr_sparse_fused(x, U, V, values, cols_idx, rows_idx, rows),
        n_w,
        parent,
    )
    recon = measure(
        lambda: lr_sparse_reconstruct(x, U, V, values, cols_idx, rows_idx),
        n_w,
        parent,
    )
    out.append(
        _pack_record(
            "lowrank_r32_plus_sparse_0p5pct",
            "low_rank_plus_sparse_correction",
            "C3LOWRANKSPARSE_DESIGN / q80 rice CSR residual (no dense W)",
            parent,
            n_w,
            storage_lrs,
            fused,
            recon,
            extra={"rank": LOW_RANK, "nnz": nnz, "density": SPARSE_DENSITY},
        )
    )
    del U, V, values, rows_idx, cols_idx
    gc.collect()

    # ---- 9. shared basis × per-layer coefficients (C1)
    V = torch.randn(cols, SHARED_RANK, device=device, dtype=dt)
    U = torch.randn(rows, SHARED_RANK, device=device, dtype=dt)
    n_layers = 64
    storage_one = int((U.numel() + V.numel()) * 2)
    storage_amortised = int((n_layers * U.numel() + V.numel()) * 2)
    n_w_all = n_w * n_layers
    print("  measuring shared_basis_r64", flush=True)
    fused = measure(lambda: svd_fused(x, U, V), n_w, parent)
    recon = measure(lambda: svd_reconstruct(x, U, V), n_w, parent)
    rec = _pack_record(
        "shared_basis_r64",
        "shared_basis_x_coefficients",
        "C1SHAREDBASIS_DESIGN / G035 shared_beats_independent=false",
        parent,
        n_w,
        storage_one,
        fused,
        recon,
        extra={
            "rank": SHARED_RANK,
            "amortised_over_layers": n_layers,
            "storage_bpw_this_tensor": bpw_of(storage_one, n_w),
            "storage_bpw_amortised_64_layers": bpw_of(storage_amortised, n_w_all),
            "note": (
                "Required three-BPW triple is billed on ONE organ (U+V). "
                "Amortised figure is extra and does not replace it."
            ),
        },
    )
    out.append(rec)
    del U, V, x
    gc.collect()
    return out


def detector_failure_mode() -> dict:
    return {
        "what_it_measures": (
            "New aten-visible torch tensors whose numel reaches the parent "
            "weight count, during the forward, under TorchDispatchMode."
        ),
        "would_miss": [
            {
                "id": "preallocated_in_place",
                "what": (
                    "A parent-shaped buffer allocated BEFORE the probe window "
                    "and filled in-place (copy_, index_put_ into an existing W). "
                    "The fill is not a new storage."
                ),
            },
            {
                "id": "custom_metal_scratch",
                "what": (
                    "A Metal/C++ kernel that writes a (rows x cols) reconstruction "
                    "into a device buffer never wrapped as a torch.Tensor. This "
                    "probe is torch-side."
                ),
            },
            {
                "id": "aten_workspace_not_returned",
                "what": (
                    "BLAS/MPS scratch inside mm/addmm that is not returned as a "
                    "tensor. A fused two-GEMM of rank 12 cannot hide a parent-sized "
                    "workspace; a custom kernel could."
                ),
            },
            {
                "id": "tiled_reconstruction_below_parent",
                "what": (
                    "Decoding W in tiles smaller than parent (the fused q4 path "
                    "does this). By the peak-allocation definition that IS "
                    "zero-dense: the parent is never resident. A detector that "
                    "wanted 'never decoded any weight' would need a different "
                    "instrument (ALU counters), and G043 already named that as "
                    "in-register dequant, not dense W."
                ),
            },
            {
                "id": "parent_via_many_concats_into_prealloc",
                "what": (
                    "torch.cat of tiles into a preallocated parent `out=` buffer "
                    "created outside the probe. Same as preallocated_in_place."
                ),
            },
        ],
        "would_false_positive": [
            {
                "id": "huge_batch",
                "what": (
                    "An activation with numel >= parent_numel (T >= min(rows, cols)). "
                    "This run uses T=4; parent is 17408 x 5120."
                ),
            }
        ],
        "not_a_source_read": True,
    }


def controls(structures: list[dict]) -> dict:
    by_id = {s["id"]: s for s in structures}
    q4 = by_id["uniform_q4_group64"]
    svd = by_id["svd_w_r12_bpw0.05"]
    dequant_false = q4["reconstruct_then_matvec"]["zero_dense"] is False
    fused_true = (
        svd["fused"]["zero_dense"] is True
        and q4["fused"]["zero_dense"] is True
    )
    return {
        "dequant_then_matvec": {
            "id": "uniform_q4_group64",
            "path": "reconstruct_then_matvec",
            "expected_zero_dense": False,
            "observed_zero_dense": q4["reconstruct_then_matvec"]["zero_dense"],
            "peak_allocation_bytes": q4["reconstruct_then_matvec"]["peak_allocation_bytes"],
            "ok": dequant_false,
            "meaning": (
                "Ordinary q4 dequant-then-matvec MUST report ZERO_DENSE=false. "
                "If the detector calls it true, the detector is broken."
            ),
        },
        "fused": {
            "id": "svd_w_r12_bpw0.05",
            "path": "fused",
            "expected_zero_dense": True,
            "observed_zero_dense": svd["fused"]["zero_dense"],
            "peak_allocation_bytes": svd["fused"]["peak_allocation_bytes"],
            "also_checked": {
                "id": "uniform_q4_group64",
                "path": "fused",
                "observed_zero_dense": q4["fused"]["zero_dense"],
            },
            "ok": fused_true,
            "meaning": (
                "A genuinely fused two-GEMM (and the production q4 group GEMV) "
                "MUST report ZERO_DENSE=true."
            ),
        },
        "both_ok": bool(dequant_false and fused_true),
    }


def three_bpw_ok(structures: list[dict]) -> dict:
    missing = []
    keys = ("storage_bpw", "active_bpw_fused", "active_bpw_cached")
    for s in structures:
        for k in keys:
            v = s.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
                missing.append({"id": s["id"], "key": k, "value": v})
        if s.get("active_bpw_cached") != CACHE_DTYPE_BPW:
            missing.append(
                {
                    "id": s["id"],
                    "key": "active_bpw_cached",
                    "value": s.get("active_bpw_cached"),
                    "expected": CACHE_DTYPE_BPW,
                }
            )
    return {"ok": not missing, "missing_or_wrong": missing}


def main() -> int:
    _ensure_torch()
    import torch

    t0 = time.time()
    device = pick_device()
    print(f"python:  {sys.executable}", flush=True)
    print(f"torch:   {torch.__version__} mps={torch.backends.mps.is_available()} device={device}", flush=True)
    print(f"parent:  mlp.up_proj {ROWS} x {COLS}  T={TOKENS}  (no 27B load)", flush=True)

    structures = run_all(device)
    ctl = controls(structures)
    bpw_ck = three_bpw_ok(structures)

    pass_ids = [s["id"] for s in structures if s["zero_dense_fused"] is True]
    fail_ids = [s["id"] for s in structures if s["zero_dense_fused"] is False]
    recon_fail = [
        s["id"]
        for s in structures
        if s.get("zero_dense_reconstruct") is False
    ]

    watched = [
        {
            "what": "detector calling q4 dequant-then-matvec ZERO_DENSE=true",
            "happened": not ctl["dequant_then_matvec"]["ok"],
            "why_it_would_matter": "the instrument cannot see a parent-sized tensor",
        },
        {
            "what": "detector calling fused SVD ZERO_DENSE=false",
            "happened": not ctl["fused"]["ok"],
            "why_it_would_matter": "the instrument is counting views, factors, or activations as parent W",
        },
        {
            "what": "a structure with fewer than three BPW figures",
            "happened": not bpw_ck["ok"],
            "why_it_would_matter": "STORAGE vs FUSED vs CACHED collapse back into one costume number",
        },
        {
            "what": "loading a second 27B into memory",
            "happened": False,
            "why_it_would_matter": "the brief forbids it; this run uses synthetic factors at organ geometry",
        },
    ]

    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "python": sys.executable,
        "torch": f"{torch.__version__} mps={torch.backends.mps.is_available()}",
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "claim": (
            "A representation that claims STRUCTURAL compression must never need "
            "to materialise the dense parent tensor to run. If it reconstructs "
            "dense W and then does an ordinary matvec, it is a STORAGE format "
            "wearing a compression costume: storage_bpw is fractional while "
            "active_bpw_cached is 16."
        ),
        "question": (
            "For each structured representation available in this repo, does the "
            "forward pass ever allocate a dense parent-shaped tensor?"
        ),
        "method": {
            "instrument": "torch.utils._python_dispatch.TorchDispatchMode",
            "new_storage_only": True,
            "parent_rule": (
                "ZERO_DENSE=false iff any NEW tensor allocated during the forward "
                "has numel >= parent_numel (the dense weight count). Views of "
                "existing storage do not count. Shape may be 2D parent, transpose, "
                "or a 4D pack of the same cardinality (q4 stack-before-reshape)."
            ),
            "not_decided_by_reading_source": True,
            "parent_weights_loaded": False,
            "synthetic_factors": True,
            "tokens": TOKENS,
            "compute_dtype": "float32",
            "cache_dtype_for_active_bpw_cached": "float16",
        },
        "parent_geometry": {
            "organ": "mlp.up_proj",
            "rows": ROWS,
            "cols": COLS,
            "numel": ROWS * COLS,
            "model": "Qwen3.8-27B",
            "note": "geometry only; no parent shard was opened",
        },
        "cited_prior": {
            "receipt": "receipts/headless/DENSE_SUBBIT_TRANSFER.json",
            "structure": "svd_w_r12_bpw0.05",
            "storage_bpw": 0.0485,
            "active_bpw_fused": 0.0485,
            "active_bpw_cache_f16": 16.0,
            "note": (
                "Same structure, three answers depending on whether you fuse or "
                "materialise. This lane MEASURES the allocation, it does not "
                "re-score cosine."
            ),
        },
        "negative_controls": ctl,
        "structures": structures,
        "summary": {
            "n_structures": len(structures),
            "zero_dense_fused_true": pass_ids,
            "zero_dense_fused_false": fail_ids,
            "reconstruct_then_matvec_zero_dense_false": recon_fail,
            "all_three_bpw": bpw_ck["ok"],
            "peak_allocation_bytes_by_structure": {
                s["id"]: {
                    "fused": s["peak_allocation_bytes_fused"],
                    "reconstruct_then_matvec": s["peak_allocation_bytes_reconstruct"],
                    "storage_bpw": s["storage_bpw"],
                    "active_bpw_fused": s["active_bpw_fused"],
                    "active_bpw_cached": s["active_bpw_cached"],
                    "zero_dense_fused": s["zero_dense_fused"],
                    "zero_dense_reconstruct": s["zero_dense_reconstruct"],
                }
                for s in structures
            },
        },
        "detector_failure_mode": detector_failure_mode(),
        "what_i_watched_fail": watched,
        "self_check": {
            "dequant_then_matvec_is_false": ctl["dequant_then_matvec"]["ok"],
            "fused_is_true": ctl["fused"]["ok"],
            "both_controls_ok": ctl["both_ok"],
            "all_three_bpw": bpw_ck["ok"],
            "did_not_load_27b": True,
            "wrote_receipt": True,
        },
        "wall_s": time.time() - t0,
        "written_to": str(RECEIPT),
        "write_scope": ["tools/headless/noetic_zero_dense.py", "receipts/headless/NOETIC_ZERO_DENSE.json"],
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(j(doc), indent=2) + "\n")

    print()
    print(f"{'structure':36} {'ZD_fused':8} {'ZD_recon':8} {'peak_f':>12} {'peak_r':>12} {'stor':>8} {'fused':>8} {'cache':>8}")
    for s in structures:
        pr = s["peak_allocation_bytes_reconstruct"]
        pr_s = "-" if pr is None else str(pr)
        print(
            f"{s['id']:36} "
            f"{str(s['zero_dense_fused']):8} "
            f"{str(s['zero_dense_reconstruct']):8} "
            f"{s['peak_allocation_bytes_fused']:12d} "
            f"{pr_s:>12} "
            f"{s['storage_bpw']:8.4f} "
            f"{s['active_bpw_fused']:8.4f} "
            f"{s['active_bpw_cached']:8.1f}"
        )
    print()
    print("PASS fused (zero-dense true): ", ", ".join(pass_ids) or "(none)")
    print("FAIL fused (zero-dense false):", ", ".join(fail_ids) or "(none)")
    print(
        "controls:",
        "dequant_then_matvec ZERO_DENSE=false",
        "OK" if ctl["dequant_then_matvec"]["ok"] else "BROKEN",
        "| fused ZERO_DENSE=true",
        "OK" if ctl["fused"]["ok"] else "BROKEN",
    )
    print(f"wrote {RECEIPT}  wall={doc['wall_s']:.2f}s")

    if not ctl["both_ok"] or not bpw_ck["ok"]:
        print("SELF-CHECK FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
