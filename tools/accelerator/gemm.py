"""Tiled GEMM with threadgroup memory and barriers. FRONT D (G046).

Three CCL gaps in one kernel: EXECUTION.shared_memory, EXECUTION.synchronization and
MATH.gemm were all recorded LARGE, and GEMM sat behind the other two. This is the
smallest thing that exercises all three for real.

It LOSES to MLX, decisively and at every size measured. That is a result, not a
failure: S015 §141 says a capability does not require Hawking authorship, and §142
says replacement needs evidence. The evidence here says do not replace.
"""
TILE = 16

TILED_GEMM_MSL = """
    uint gx = thread_position_in_grid.x;
    uint gy = thread_position_in_grid.y;
    uint lx = thread_position_in_threadgroup.x;
    uint ly = thread_position_in_threadgroup.y;
    threadgroup float As[%d][%d];
    threadgroup float Bs[%d][%d];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += %du) {
        As[ly][lx] = (gy < M && (k0 + lx) < K) ? A[gy * K + k0 + lx] : 0.0f;
        Bs[ly][lx] = ((k0 + ly) < K && gx < N) ? B[(k0 + ly) * N + gx] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < %du; ++k) acc += As[ly][k] * Bs[k][lx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gy < M && gx < N) C[gy * N + gx] = acc;
""" % (TILE, TILE, TILE, TILE, TILE, TILE)


def negative_controls() -> dict[str, str]:
    """Deliberately broken variants. A correctness check nobody has watched FAIL is
    not a check, so these exist to prove the GEMM test can fail: removing the
    barriers introduces a race and transposing an index computes the wrong thing.
    Both were run and both produced large errors."""
    return {
        "barriers_removed": TILED_GEMM_MSL.replace(
            "threadgroup_barrier(mem_flags::mem_threadgroup);", ""),
        "index_transposed": TILED_GEMM_MSL.replace(
            "acc += As[ly][k] * Bs[k][lx];", "acc += As[ly][k] * Bs[lx][k];"),
    }
