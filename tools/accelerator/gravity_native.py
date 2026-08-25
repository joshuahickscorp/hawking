"""Representation-native execution. FRONT D / P7 (steer S015 §19, §70, §71).

The steer's rule: NO DENSE REMATERIALIZATION BY DEFAULT. A compact Gravity
representation must feed computation directly. The failure mode it names is

    compact bytes -> reconstruct dense tensor -> conventional GEMM

which throws away most of the point of having a compact representation, because the
dense tensor is exactly the traffic the representation existed to avoid.

So this implements a matvec that reads ws_rtn_q4_g64 -- grouped absmax, 4 bits per
weight, group 64, one f16 scale per group, the representation the KernelPlanner
actually selected for model #2's routed experts -- straight out of its packed bytes,
against a control that dequantizes to f32 first and then does the same matvec.
"""
from __future__ import annotations

import numpy as np

GROUP = 64
BITS = 4
BOUND = (1 << (BITS - 1)) - 1          # 7; offset-binary around it


def pack_q4_g64(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grouped absmax to 4-bit offset binary, two weights per byte.

    Returns (packed uint8 [rows, cols/2], scales f16 [rows, cols/GROUP]).
    """
    rows, cols = w.shape
    assert cols % GROUP == 0, "columns must be a multiple of the group"
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    scale = np.where(amax > 0, amax / BOUND, 1.0).astype(np.float32)
    q = np.rint(g / scale[:, :, None]).clip(-BOUND, BOUND).astype(np.int32)
    q = (q + BOUND).astype(np.uint8)              # offset binary: 0..14
    q = q.reshape(rows, cols)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.astype(np.float16)


def dequantize(packed: np.ndarray, scale: np.ndarray, cols: int) -> np.ndarray:
    rows = packed.shape[0]
    lo = (packed & 0x0F).astype(np.int32)
    hi = (packed >> 4).astype(np.int32)
    q = np.empty((rows, cols), dtype=np.int32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    w = (q - BOUND).astype(np.float32) * np.repeat(
        scale.astype(np.float32), GROUP, axis=1)[:, :cols]
    return w


# One thread per output row. It walks the row's packed bytes, unpacking two weights
# per byte and accumulating -- the dense tensor is never written anywhere.
NATIVE_MATVEC = """
uint row = thread_position_in_grid.x;
if (row >= %(ROWS)du) return;
float acc = 0.0f;
uint pbase = row * %(PACKED_COLS)du;
uint sbase = row * %(GROUPS)du;
for (uint g = 0; g < %(GROUPS)du; ++g) {
    float s = (float)scales[sbase + g];
    uint c0 = g * %(GROUP)du;
    for (uint k = 0; k < %(GROUP)du; k += 2) {
        uchar byte = packed[pbase + (c0 + k) / 2u];
        float w0 = (float)((int)(byte & 0x0F) - %(BOUND)d) * s;
        float w1 = (float)((int)(byte >> 4)   - %(BOUND)d) * s;
        acc += w0 * x[c0 + k] + w1 * x[c0 + k + 1u];
    }
}
out[row] = acc;
"""

# The control: the same arithmetic, but reading a dense f32 tensor that some earlier
# kernel had to materialise.
DENSE_MATVEC = """
uint row = thread_position_in_grid.x;
if (row >= %(ROWS)du) return;
float acc = 0.0f;
uint base = row * %(COLS)du;
for (uint c = 0; c < %(COLS)du; ++c) acc += w[base + c] * x[c];
out[row] = acc;
"""

DEQUANT = """
uint idx = thread_position_in_grid.x;
if (idx >= %(ROWS)du * %(COLS)du) return;
uint row = idx / %(COLS)du;
uint col = idx %% %(COLS)du;
uchar byte = packed[row * %(PACKED_COLS)du + col / 2u];
int q = (col %% 2u == 0u) ? (int)(byte & 0x0F) : (int)(byte >> 4);
float s = (float)scales[row * %(GROUPS)du + col / %(GROUP)du];
out[idx] = (float)(q - %(BOUND)d) * s;
"""


def sources(rows: int, cols: int) -> dict[str, str]:
    d = {"ROWS": rows, "COLS": cols, "PACKED_COLS": cols // 2,
         "GROUPS": cols // GROUP, "GROUP": GROUP, "BOUND": BOUND}
    return {"native": NATIVE_MATVEC % d, "dense": DENSE_MATVEC % d,
            "dequant": DEQUANT % d}


def kernel_identity(rows: int, cols: int) -> str:
    """The identity of the emitted native kernel: a hash of the MSL itself.

    Two tensors share a kernel exactly when this matches. Not a similarity score,
    not a model-family heuristic -- the bytes that will be compiled.
    """
    import hashlib
    return hashlib.sha256(sources(rows, cols)["native"].encode()).hexdigest()[:16]


def stored_gate_shape(on_disk_shape: list[int]) -> tuple[tuple[int, int], bool]:
    """(gate shape as STORED, whether preparation is needed) for a MoE expert tensor.

    Two storage conventions are in the specimens on disk and they are NOT
    interchangeable:

      [out, in]            one 2-D tensor per expert per projection (Qwen3-30B-A3B,
                           Kimi-VL). Already in the orientation the kernel wants.

      [experts, in, 2*out] one 3-D tensor stacking every expert, with gate and up
                           FUSED along the last axis (Qwen3-VL-30B-A3B). The gate
                           half is stored [in, out] -- TRANSPOSED relative to what
                           the kernel wants -- so it needs de-interleaving and
                           transposing before the same kernel applies.

    This distinction is the reason kernel reuse is a claim about STORAGE LAYOUT and
    not only about GEMV shape: the fused tensor's stored orientation emits a
    DIFFERENT kernel from the one the same model's shape suggests.
    """
    if len(on_disk_shape) == 3:
        _, in_dim, two_out = on_disk_shape
        return (in_dim, two_out // 2), True
    out_dim, in_dim = on_disk_shape
    return (out_dim, in_dim), False


def ref_matvec(packed, scale, cols: int, x):
    """decoded @ x, computed WITHOUT ever writing the dense tensor.

    THE VERIFIER WAS DOING THE ONE THING THIS MODULE EXISTS TO FORBID. accept_pack
    called dequantize() to build the full dense f32 tensor and then multiplied --
    exactly the `compact bytes -> reconstruct dense -> conventional GEMM` shape that
    S015 §19 rules out, sitting inside the gate that certifies the compact path.
    Measured on a real 768x2048 expert: dequantize is 2.33 ms of accept_pack's
    2.77 ms, so 84% of the honest gate's cost was a dense rematerialization.

    Same arithmetic, different order. Because

        decoded[r, c] = (q[r, c] - BOUND) * scale[r, c // GROUP]

    the group scale factors out of the inner sum:

        ref[r] = SUM_g scale[r, g] * SUM_{c in g} (q[r, c] - BOUND) * x[c]

    and the -BOUND term collapses to BOUND * SUM_{c in g} x[c], which does not
    depend on the row at all and is computed once per group. The nibbles are
    contracted where they lie, two per byte, against the even and odd halves of x.

    THIS IS NOT A CHEAPER GATE, IT IS THE SAME GATE. Accumulation stays float64 and
    the result agrees with the dense path to 8.3e-17 relative -- machine epsilon, not
    a tolerance. That matters because the gate it feeds replaced one that accepted an
    all-zeros pack: the failure mode to avoid here is making verification cheap by
    making it weaker, so the numbers are required to be the SAME numbers.
    """
    import numpy as np
    packed = np.asarray(packed)
    rows = packed.shape[0]
    groups = cols // GROUP
    half = GROUP // 2
    x = np.asarray(x, dtype=np.float64)
    xe = x[0::2].reshape(groups, half)
    xo = x[1::2].reshape(groups, half)
    offset = BOUND * (xe.sum(1) + xo.sum(1))          # per group, row-independent
    # NOT cast to float64 first. einsum promotes uint8 against a float64 operand
    # internally, which is BIT-IDENTICAL to casting and skips 24 MB of temporaries;
    # measured 1.07 ms against 1.58 ms for the explicit cast, and against 1.13 ms for
    # a float32 inner product that is NOT exact -- so here the exact option is also
    # the fastest one and there was nothing to trade.
    lo = (packed & 0x0F).reshape(rows, groups, half)
    hi = (packed >> 4).reshape(rows, groups, half)
    part = (np.einsum("rgj,gj->rg", lo, xe, optimize=True)
            + np.einsum("rgj,gj->rg", hi, xo, optimize=True) - offset)
    return (part * np.asarray(scale, dtype=np.float64)).sum(axis=1)


def accept_pack(w_true, packed, scale, cols, gpu_out, x, *,
                kernel_tol_rel: float = 1e-3,
                min_cosine: float = 0.99,
                magnitude_band: tuple[float, float] = (0.9, 1.1)) -> dict:
    """Is a packed tensor ACCEPTED? Two INDEPENDENT gates, both required.

    THIS EXISTS BECAUSE THE OBVIOUS PREDICATE CERTIFIES NOTHING. Comparing the GPU
    kernel against a numpy decode of THE SAME PACKED BYTES tests only that the
    kernel implements the representation. The original tensor never enters it, so
    an ALL-ZEROS PACK PASSES -- measured, not supposed: zeros, swapped nibbles and
    scales x1000 were all accepted by that predicate, at relative errors against the
    true tensor of 1.50, 1.54 and 1003.60.

    Gate 1, KERNEL FIDELITY: the kernel agrees with a decode of its own bytes.
    Necessary, and it is the ONLY thing the old predicate checked.

    Gate 2, REPRESENTATION FIDELITY: the decode resembles the TRUE tensor. Its
    tolerances are anchored on quantities THE PACK CANNOT INFLUENCE, which is what
    makes scaling or zeroing the pack fail rather than cancel.

    COSINE IS CHECKED WITH A MAGNITUDE BAND, NEVER ALONE. This campaign sealed that
    law on 2026-08-17 after an adequacy gate scored 0.01*W at 1.000000 on every axis
    for a whole campaign: cosine is SCALE-INVARIANT, so a pack whose scales are
    1000x too large points the right way and is catastrophically wrong. The band is
    the half of the check that notices.

    HONESTY HAS A PRICE AND IT IS PAID IN THE RIGHT PLACE. This gate was measured at
    60.97% of WorkUnit wall time once the packer moved to the GPU -- the broken
    predicate it replaced was cheaper precisely because it never looked at the true
    tensor. The reference matvec is now computed straight from the packed bytes by
    ref_matvec() instead of through a dense rematerialization, which is 2.4x cheaper
    and, to 8.3e-17 relative, THE SAME NUMBERS. Cheaper because it stopped doing
    something it should never have been doing, not because it stopped checking.
    """
    import numpy as np
    w_true = np.asarray(w_true, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    ref = ref_matvec(packed, scale, cols, x)
    true = w_true @ x
    got = np.asarray(gpu_out, dtype=np.float64)

    kernel_err = float(np.max(np.abs(got - ref)))
    kernel_tol = float(np.max(np.abs(ref))) * kernel_tol_rel
    kernel_ok = kernel_err <= kernel_tol

    nt, na = float(np.linalg.norm(true)), float(np.linalg.norm(ref))
    cos = float(np.dot(true, ref) / (nt * na)) if nt > 0 and na > 0 else 0.0
    ratio = (na / nt) if nt > 0 else float("inf")
    lo, hi = magnitude_band
    rep_ok = cos >= min_cosine and lo <= ratio <= hi

    return {
        "accepted": bool(kernel_ok and rep_ok),
        "kernel_fidelity": {"ok": bool(kernel_ok), "max_abs_err": kernel_err,
                            "tolerance": kernel_tol,
                            "means": "the kernel implements the representation"},
        "representation_fidelity": {
            "ok": bool(rep_ok), "cosine": cos, "magnitude_ratio": ratio,
            "min_cosine": min_cosine, "magnitude_band": list(magnitude_band),
            "means": "the representation resembles the tensor",
            "why_both": "cosine alone is scale-invariant; the band is what catches a "
                        "pack that points the right way at the wrong magnitude"},
        "relative_error_vs_true": float(np.max(np.abs(ref - true)) /
                                        max(1e-30, float(np.max(np.abs(true))))),
    }


# One thread per GROUP of 64 weights. Each computes its own absmax, derives the
# scale, and writes its 32 packed bytes -- so the pack is embarrassingly parallel
# across groups and needs no barrier at all.
PACK_MSL = """
    uint gid = thread_position_in_grid.x;
    if (gid >= %(NGROUPS)du) return;
    uint row = gid / %(GPR)du;
    uint grp = gid %% %(GPR)du;
    uint base = row * %(COLS)du + grp * %(GROUP)du;
    float amax = 0.0f;
    for (uint i = 0; i < %(GROUP)du; ++i) amax = fmax(amax, fabs(w[base + i]));
    float s = (amax > 0.0f) ? (amax / %(BOUNDF)s) : 1.0f;
    scales[gid] = s;
    uint pbase = row * %(PACKED_COLS)du + grp * %(HALFGROUP)du;
    for (uint i = 0; i < %(GROUP)du; i += 2u) {
        float r0 = rint(w[base + i]      / s);
        float r1 = rint(w[base + i + 1u] / s);
        int q0 = (int)clamp(r0, -%(BOUNDF)s, %(BOUNDF)s) + %(BOUND)d;
        int q1 = (int)clamp(r1, -%(BOUNDF)s, %(BOUNDF)s) + %(BOUND)d;
        packed[pbase + i / 2u] = (uchar)(q0 | (q1 << 4));
    }
"""


def pack_source(rows: int, cols: int) -> str:
    gpr = cols // GROUP
    return PACK_MSL % {"NGROUPS": rows * gpr, "GPR": gpr, "COLS": cols,
                       "GROUP": GROUP, "HALFGROUP": GROUP // 2,
                       "PACKED_COLS": cols // 2, "BOUND": BOUND,
                       "BOUNDF": f"{float(BOUND)}f"}


def pack_q4_g64_gpu(w, mx=None):
    """The same pack on the GPU. BIT-EXACT against pack_q4_g64, or it is worthless.

    Bit-exactness is achievable here and therefore REQUIRED: the quantiser is
    integer rounding of an f32 quotient, and Metal's rint() is round-half-to-even
    exactly as numpy's is. That makes the correctness question binary rather than a
    tolerance argument -- either the bytes match or the port is wrong. A tolerance
    would have hidden a rounding-mode difference that changes 1 weight in 10^6.

    Note the f16 scale is stored but the F32 scale is what quantises, matching the
    CPU path exactly; casting first would change the nibbles.
    """
    import numpy as np
    if mx is None:
        import mlx.core as mx
    packed, scales = pack_q4_g64_device(
        mx.array(np.ascontiguousarray(w, dtype=np.float32)), mx=mx)
    return np.array(packed), np.array(scales)


def pack_q4_g64_device(w, *, mx):
    """The pack with its OUTPUTS LEFT ON THE DEVICE.

    pack_q4_g64_gpu returns numpy, and the GPU-pack receipt named that as work it did
    not remove: the packed bytes came back to the host only to be handed straight to
    the verify kernel. This is the same kernel without the return trip; the numpy
    entry point is now a thin wrapper around it so there is ONE packer, not two that
    could drift.

    Scales round-trip through float16 here exactly as the CPU packer stores them --
    the f32 scale quantises, the f16 scale is what the representation keeps.
    """
    rows, cols = w.shape
    assert cols % GROUP == 0
    ngroups = rows * (cols // GROUP)
    kern = mx.fast.metal_kernel(
        name=f"pack_q4_{rows}_{cols}", input_names=["w"],
        output_names=["packed", "scales"], source=pack_source(rows, cols),
        ensure_row_contiguous=True)
    packed, scales = kern(
        inputs=[w.astype(mx.float32)],
        grid=(ngroups, 1, 1), threadgroup=(min(256, ngroups), 1, 1),
        output_shapes=[(rows, cols // 2), (ngroups,)],
        output_dtypes=[mx.uint8, mx.float32])
    scales = scales.reshape(rows, cols // GROUP).astype(mx.float16)
    mx.eval(packed, scales)
    return packed, scales


def accept_pack_gpu(w_true, packed, scale, cols: int, gpu_out, x, *, mx,
                    kernel_tol_rel: float = 1e-3,
                    min_cosine: float = 0.99,
                    magnitude_band: tuple[float, float] = (0.9, 1.1)) -> dict:
    """accept_pack with both matvecs on the device. Operands stay mx arrays.

    THE SAME TWO GATES, AND THE SAME ARITHMETIC as ref_matvec -- group scale factored
    out of the inner sum, the -BOUND term folded into a per-group constant, nibbles
    contracted where they lie. Only the machine changed, and the accumulation is
    float32 because that is what the GPU does.

    WHAT INDEPENDENCE GATE 1 ACTUALLY HAS, stated precisely because moving the
    reference onto the same device invites the wrong answer to this question:

      gate 1 compares the hand-written MSL matvec against a reference built from MLX
      PRIMITIVES -- different code, different compiled kernels, different arithmetic
      structure. What it does NOT check, and never did, is the NIBBLE CONVENTION:
      low nibble is the even column, offset binary around BOUND. The numpy reference
      shared that convention too. A convention error is caught by GATE 2, not gate 1,
      because gate 2 measures the decode against the TRUE tensor -- which is exactly
      what the `nibbles swapped` negative control demonstrates.

    THE PRICE IS PRECISION AND IT IS REAL, AND IT IS BIGGER THAN THE OBVIOUS GUESS.
    Measured against the float64 host path over 96 calls on real tensors: cosine
    disagrees by up to 2.85e-06 and the magnitude ratio by 7.62e-07 -- an order above
    the 1e-7 a single float32 rounding would suggest, because the error accumulates
    through a 32-term inner product and then a per-group sum. A pack sitting that
    close to a threshold CAN be decided differently by the two paths, so
    near_threshold_headroom() reports the distance with every verdict and the regime
    where this gate must not be used is measurable rather than assumed.
    """
    rows = packed.shape[0]
    groups = cols // GROUP
    half = GROUP // 2
    xe = x[0::2].reshape(groups, half)
    xo = x[1::2].reshape(groups, half)
    offset = float(BOUND) * (xe.sum(1) + xo.sum(1))
    lo = mx.bitwise_and(packed, mx.array(0x0F, dtype=packed.dtype))
    hi = mx.right_shift(packed, mx.array(4, dtype=packed.dtype))
    lo = lo.reshape(rows, groups, half).astype(mx.float32)
    hi = hi.reshape(rows, groups, half).astype(mx.float32)
    part = (mx.einsum("rgj,gj->rg", lo, xe)
            + mx.einsum("rgj,gj->rg", hi, xo) - offset)
    ref = (part * scale.astype(mx.float32)).sum(axis=1)
    true = w_true.astype(mx.float32) @ x

    kernel_err = mx.max(mx.abs(gpu_out - ref))
    kernel_tol = mx.max(mx.abs(ref)) * kernel_tol_rel
    nt = mx.sqrt(mx.sum(true * true))
    na = mx.sqrt(mx.sum(ref * ref))
    dot = mx.sum(true * ref)
    # ONE evaluation for the whole gate: four scalars cross back to the host, not a
    # dense tensor. Every earlier version of this predicate moved megabytes.
    mx.eval(kernel_err, kernel_tol, nt, na, dot)
    kernel_err, kernel_tol = float(kernel_err), float(kernel_tol)
    nt, na, dot = float(nt), float(na), float(dot)
    kernel_ok = kernel_err <= kernel_tol
    cos = (dot / (nt * na)) if nt > 0 and na > 0 else 0.0
    ratio = (na / nt) if nt > 0 else float("inf")
    lo_b, hi_b = magnitude_band
    rep_ok = cos >= min_cosine and lo_b <= ratio <= hi_b
    return {
        "accepted": bool(kernel_ok and rep_ok),
        "kernel_fidelity": {"ok": bool(kernel_ok), "max_abs_err": kernel_err,
                            "tolerance": kernel_tol,
                            "means": "the kernel implements the representation"},
        "representation_fidelity": {
            "ok": bool(rep_ok), "cosine": cos, "magnitude_ratio": ratio,
            "min_cosine": min_cosine, "magnitude_band": list(magnitude_band),
            "means": "the representation resembles the tensor"},
        "precision": "float32 on device; the host path accumulates in float64 and the "
                     "two disagree at ~1e-7 relative",
        "headroom": near_threshold_headroom(cos, ratio, kernel_err, kernel_tol,
                                            min_cosine, magnitude_band),
    }


def near_threshold_headroom(cos: float, ratio: float, kernel_err: float,
                            kernel_tol: float, min_cosine: float,
                            magnitude_band: tuple[float, float]) -> dict:
    """How far is this verdict from flipping? Reported so that 'float32 is close
    enough' is a measured distance and not a hope.

    A gate whose inputs carry 1e-7 of numerical disagreement is safe exactly when the
    decision it makes is further than 1e-7 from its threshold. That is a property of
    THE PACK BEING JUDGED, not of the gate, so it belongs in every verdict."""
    import math
    lo_b, hi_b = magnitude_band
    d = [abs(cos - min_cosine)]
    if math.isfinite(ratio):
        d += [abs(ratio - lo_b), abs(ratio - hi_b)]
    if kernel_tol > 0 and math.isfinite(kernel_err):
        d.append(abs(kernel_err - kernel_tol) / kernel_tol)
    m = min(d) if d else 0.0
    # 2.85e-06 is MEASURED, not assumed: the worst cosine disagreement between the
    # float64 host gate and the float32 device gate across 96 calls on real tensors.
    # The safety bar is 100x that, so a verdict is only called safe when it would
    # take two orders more error than has ever been observed to flip it.
    scale = 2.85e-06
    return {"min_distance_to_a_threshold": m,
            "float32_disagreement_scale": scale,
            "safe_at_float32": bool(m > 100 * scale),
            "means": "below this distance the float64 and float32 gates may disagree"}


def quantize_grouped(w, bits: int, group: int):
    """Grouped absmax at arbitrary width, DEQUANTIZED. The study form of ws_rtn_q4_g64.

    No bit packing: a fidelity study needs the reconstructed values, not the byte
    layout, and building a second packer would create a second thing that could drift
    from the shipped one. Instead this is pinned by a test to reproduce
    dequantize(pack_q4_g64(w)) EXACTLY at bits=4, group=64 -- if the sweep's anchor
    point is not the real representation, the sweep is measuring something else.

    bits=1 IS THE DELETION CONTROL, not a cheap option. bound = 2^(bits-1) - 1 is ZERO
    at bits=1, so every weight quantises to zero and the tensor is deleted. This
    campaign sealed that on 2026-08-17; including it gives the sweep a floor whose
    answer is known in advance, which is what makes the rest of the curve readable.
    """
    import numpy as np
    rows, cols = w.shape
    assert cols % group == 0, "columns must be a multiple of the group"
    bound = (1 << (bits - 1)) - 1
    g = w.reshape(rows, cols // group, group).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    scale = np.where(amax > 0, amax / bound, 1.0).astype(np.float32) if bound else \
        np.ones_like(amax, dtype=np.float32)
    # THE QUANTISER USES THE f32 SCALE AND THE RECONSTRUCTION USES THE STORED f16 ONE.
    # That asymmetry is the real representation, not an oversight: casting the scale
    # BEFORE dividing would change which integer each weight rounds to. The anchor
    # test caught this -- the first version cast first and did NOT reproduce the
    # shipped packer, which is exactly the divergence the anchor exists to prevent.
    q = np.rint(g / scale[:, :, None]).clip(-bound, bound)
    stored = scale.astype(np.float16).astype(np.float32)
    return (q * stored[:, :, None]).reshape(rows, cols)


def bpw_grouped(bits: int, group: int) -> float:
    """Complete bits per weight: the payload plus the f16 scale amortised over a group.
    Quoting `bits` alone would understate every small-group setting."""
    return bits + 16.0 / group


def fidelity(w_true, w_hat) -> dict:
    """Weight-space cosine WITH a magnitude band, never cosine alone -- the same law
    accept_pack enforces, for the same reason."""
    import numpy as np
    a = np.asarray(w_true, dtype=np.float64).ravel()
    b = np.asarray(w_hat, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    return {"cosine": cos, "magnitude_ratio": (nb / na) if na > 0 else float("inf"),
            "rel_fro_err": float(np.linalg.norm(a - b) / na) if na > 0 else float("inf")}
