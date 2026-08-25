"""C2M pins. FRONT C (G045). The refusals matter more than the successes: a silent
mistranslation is worse than a rejection."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import c2m  # noqa: E402

IDX = "int i = blockIdx.x * blockDim.x + threadIdx.x;"


def k(body, params="const float* a, const float* b, float* c, int n"):
    return f"__global__ void t({params}) {{ {IDX} {body} }}"


def test_translates_the_four_supported_forms():
    for expr, op in [("a[i] + b[i]", "add"), ("a[i] * b[i]", "mul"),
                     ("a[i] - b[i]", "sub"), ("fmaxf(a[i], 0.0f)", "relu")]:
        t = c2m.translate(k(f"if (i < n) c[i] = {expr};"), elements=64)
        assert t.program.ops[0].kind == op
        assert t.tier == "C2M-T0"


@pytest.mark.parametrize("body,token", [
    ("__shared__ float s[4]; c[i] = a[i] + b[i];", "shared memory"),
    ("for (int j=0;j<4;j++) c[i] = a[i] + b[i];", "loops"),
    ("while (n) c[i] = a[i] + b[i];", "loops"),
    ("atomicAdd(&c[i], b[i]);", "atomics"),
    ("__syncthreads(); c[i] = a[i] + b[i];", "block synchronization"),
    ("printf(\"x\"); c[i] = a[i] + b[i];", "device printf"),
])
def test_unsupported_constructs_are_refused_by_name(body, token):
    with pytest.raises(c2m.C2MRefusal) as e:
        c2m.translate(k(body), elements=64)
    assert token in str(e.value)


def test_fp64_is_refused():
    with pytest.raises(c2m.C2MRefusal, match="fp64"):
        c2m.translate(k("c[i] = a[i] + b[i];",
                        params="const double* a, const double* b, double* c, int n"),
                      elements=64)


def test_an_unrecognised_expression_is_refused_not_guessed():
    with pytest.raises(c2m.C2MRefusal, match="outside the C2M-T0 pattern set"):
        c2m.translate(k("c[i] = sinf(a[i]);"), elements=64)


def test_missing_thread_index_is_refused():
    src = "__global__ void t(const float* a, float* c, int n) { c[i] = a[i] + a[i]; }"
    with pytest.raises(c2m.C2MRefusal, match="thread index"):
        c2m.translate(src, elements=64)


def test_multiple_stores_are_refused():
    with pytest.raises(c2m.C2MRefusal, match="more than one store"):
        c2m.translate(k("c[i] = a[i] + b[i]; c[i] = a[i] * b[i];"), elements=64)


def test_a_non_pointer_operand_is_refused():
    src = ("__global__ void t(const float* a, float n, float* c) { " + IDX +
           " c[i] = a[i] + n[i]; }")
    with pytest.raises(c2m.C2MRefusal, match="not a pointer parameter"):
        c2m.translate(src, elements=64)


def test_conformance_never_claims_a_tier_it_did_not_earn():
    empty = c2m.conformance([])
    assert empty["tier_claimed"] == "NONE"
    got = c2m.conformance([{"matches_oracle": True}])
    assert got["tier_claimed"] == "C2M-T0"
    for tier in ("C2M-T1", "C2M-T2", "C2M-T3", "C2M-T4", "C2M-T5"):
        assert got["higher_tiers"][tier].startswith("NOT CLAIMED")


def test_conformance_states_it_is_not_a_cuda_differential():
    c = c2m.conformance([{"matches_oracle": True}])
    assert c["is_a_cuda_differential"] is False
    assert c["oracle"] == "numpy on CPU"


# ------------------------------------------------------------- T1 runtime (G045)

import cuda_runtime as cr  # noqa: E402

_KERNELS = {"vadd": "__global__ void vadd(const float* a, const float* b, float* c, int n){"
                    " int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }"}
_SAFE = """
cudaMalloc((void**)&d_a, N*sizeof(float));
cudaMalloc((void**)&d_b, N*sizeof(float));
cudaMalloc((void**)&d_c, N*sizeof(float));
cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
vadd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
cudaDeviceSynchronize();
cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
"""
_HAZARD = _SAFE.replace("vadd<<<", "h_a[0] = 999.0f;\nvadd<<<")


@pytest.mark.parametrize("bad,frag", [
    ("cudaStreamCreate(&s);", "streams"),
    ("cudaMemcpyAsync(a,b,4,cudaMemcpyHostToDevice,0);", "asynchronous"),
    ("cudaMallocManaged((void**)&p, 4);", "managed memory"),
    ("cudaMemset(d_a, 0, 4);", "memset"),
    ("int x = foo();", "outside the C2M-T1 subset"),
])
def test_t1_refuses_by_name(bad, frag):
    with pytest.raises(c2m.C2MRefusal, match=frag):
        cr.parse_host(bad)


def test_may_delete_copies_names_the_offending_statement():
    """The condition on S015 §9. A host write after the copy makes the alias
    observationally different from the snapshot, so the deletion is refused."""
    assert cr.parse_host(_SAFE).may_delete_copies()[0] is True
    ok, why = cr.parse_host(_HAZARD).may_delete_copies()
    assert ok is False and "h_a" in why and "999.0f" in why


def test_a_write_after_the_launch_does_not_block_the_deletion():
    """The guard must be about the SNAPSHOT WINDOW, not about host writes in
    general -- otherwise it would refuse programs that are perfectly safe."""
    after = _SAFE.replace("cudaDeviceSynchronize();", "cudaDeviceSynchronize();\nh_a[0] = 5.0f;")
    assert cr.parse_host(after).may_delete_copies()[0] is True


def test_both_modes_agree_on_a_safe_program():
    pytest.importorskip("mlx.core")
    import numpy as np
    n = 4096
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(n).astype(np.float32) for _ in range(2))
    arr = lambda: {"h_a": a.copy(), "h_b": b.copy(), "h_c": np.zeros(n, np.float32)}
    f = cr.execute_host(_SAFE, _KERNELS, arr(), elements=n, mode="FAITHFUL")
    u = cr.execute_host(_SAFE, _KERNELS, arr(), elements=n, mode="UNIFIED")
    assert np.array_equal(f["host"]["h_c"], u["host"]["h_c"])
    assert np.max(np.abs(f["host"]["h_c"] - (a + b))) < 1e-5
    assert (f["copies_performed"], u["copies_performed"]) == (3, 0)


def test_the_hazard_is_real_and_the_guard_is_necessary():
    """A refusal nobody has watched be NECESSARY is indistinguishable from one that
    is merely cautious. Bypassing the guard must produce a WRONG answer."""
    pytest.importorskip("mlx.core")
    import numpy as np
    n = 4096
    rng = np.random.default_rng(3)
    a, b = (rng.standard_normal(n).astype(np.float32) for _ in range(2))
    arr = lambda: {"h_a": a.copy(), "h_b": b.copy(), "h_c": np.zeros(n, np.float32)}
    f = cr.execute_host(_HAZARD, _KERNELS, arr(), elements=n, mode="FAITHFUL")
    u = cr.execute_host(_HAZARD, _KERNELS, arr(), elements=n, mode="UNIFIED", _unsafe_ok=True)
    assert abs(f["host"]["h_c"][0] - u["host"]["h_c"][0]) > 100
    assert np.array_equal(f["host"]["h_c"][1:], u["host"]["h_c"][1:])  # one element only
    with pytest.raises(c2m.C2MRefusal):
        cr.execute_host(_HAZARD, _KERNELS, arr(), elements=n, mode="UNIFIED")


def test_uninitialised_device_read_is_refused():
    pytest.importorskip("mlx.core")
    import numpy as np
    src = ("cudaMalloc((void**)&d_a, N*sizeof(float));\n"
           "cudaMalloc((void**)&d_b, N*sizeof(float));\n"
           "cudaMalloc((void**)&d_c, N*sizeof(float));\n"
           "vadd<<<1,256>>>(d_a, d_b, d_c, N);\n")
    with pytest.raises(c2m.C2MRefusal, match="never written"):
        cr.execute_host(src, _KERNELS, {"h_a": np.zeros(4, np.float32)},
                        elements=4, mode="FAITHFUL")


def test_t1_is_claimed_only_when_both_modes_agree():
    assert cr.conformance_t1([])["tier_claimed"] == "C2M-T0"
    assert cr.conformance_t1([{"matches_oracle": True, "both_modes_agree": False}]
                             )["tier_claimed"] == "C2M-T0"
    c = cr.conformance_t1([{"matches_oracle": True, "both_modes_agree": True}])
    assert c["tier_claimed"] == "C2M-T1"
    assert "FRONTEND" in c["higher_tiers"]["C2M-T2"] or "C2M frontend" in c["higher_tiers"]["C2M-T2"]


# ------------------------------------------------------ T2 idiom recognition (G045)

import c2m_idiom as ci  # noqa: E402

_GEMM = """
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int t = 0; t < (K + 15) / 16; ++t) {
        As[ty][tx] = (row < M && t * 16 + tx < K) ? A[row * K + t * 16 + tx] : 0.0f;
        Bs[ty][tx] = (t * 16 + ty < K && col < N) ? B[(t * 16 + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"""
_RED = """
__global__ void reduce_sum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
"""


def test_the_two_textbook_idioms_are_recognised():
    g = ci.recognize(_GEMM)
    assert (g.idiom, g.kernel_name, g.tile) == ("tiled_gemm", "sgemm_tiled", 16)
    r = ci.recognize(_RED)
    assert (r.idiom, r.threadgroup) == ("block_reduce_sum", 256)


# Each near-miss differs from its idiom in EXACTLY ONE respect. Accepting any of
# them would mean recognition is not checking anything, so these ARE the evidence
# that "this is a tiled GEMM" is a claim rather than a guess.
@pytest.mark.parametrize("mutate,frag", [
    (lambda s: s.replace("acc += As[ty][k] * Bs[k][tx]", "acc += As[ty][k] * Bs[tx][k]"),
     "inner product"),
    (lambda s: s.replace("__syncthreads();", "", 1), "__syncthreads"),
    (lambda s: s.replace("float acc = 0.0f", "float acc = 1.0f"), "accumulator"),
    (lambda s: s.replace("C[row * N + col] = acc", "C[col * M + row] = acc"), "store to C"),
    (lambda s: s.replace("__shared__ float Bs[16][16];", "__shared__ float Bs[16][32];"),
     "shared tiles are"),
])
def test_gemm_near_misses_are_refused(mutate, frag):
    with pytest.raises(c2m.C2MRefusal, match=frag):
        ci.recognize(mutate(_GEMM))


@pytest.mark.parametrize("mutate,frag", [
    (lambda s: s.replace("sdata[tid] += sdata[tid + s]", "sdata[tid] *= sdata[tid + s]"),
     "partial SUM"),
    (lambda s: s.replace("(i < n) ? in[i] : 0.0f", "(i < n) ? in[i] : 1.0f"), "zero identity"),
    (lambda s: s.replace("s = blockDim.x / 2; s > 0; s >>= 1", "s = 1; s < blockDim.x; s <<= 1"),
     "halving tree"),
    (lambda s: s.replace("__syncthreads();", "", 1), "__syncthreads"),
    (lambda s: s.replace("sdata[256]", "sdata[100]"), "legal Metal threadgroup"),
])
def test_reduce_near_misses_are_refused(mutate, frag):
    with pytest.raises(c2m.C2MRefusal, match=frag):
        ci.recognize(mutate(_RED))


def test_the_two_doors_do_not_overlap():
    """T0 must refuse what T2 accepts and vice versa. If either door quietly took
    the other's kernels, neither would mean what its tier says."""
    for src in (_GEMM, _RED):
        with pytest.raises(c2m.C2MRefusal):
            c2m.translate(src, elements=64)
    elementwise = ("__global__ void vadd(const float* a, const float* b, float* c, int n){"
                   " int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }")
    with pytest.raises(c2m.C2MRefusal, match="no known idiom"):
        ci.recognize(elementwise)
    assert c2m.translate(elementwise, elements=64).output == "c"


def test_t2_is_refused_unless_every_near_miss_was_refused():
    """A tier claimed on recognition that never rejects anything is a coin flip."""
    good = [{"idiom": "tiled_gemm", "matches_oracle": True}]
    assert ci.conformance_t2(good, [])["tier_claimed"] == "C2M-T1"
    assert ci.conformance_t2(good, [{"refused": False}])["tier_claimed"] == "C2M-T1"
    c = ci.conformance_t2(good, [{"refused": True}])
    assert c["tier_claimed"] == "C2M-T2"
    assert "not general compilation" in c["mechanism"]


def test_recognised_idioms_execute_and_match_numpy():
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    rng = np.random.default_rng(2)
    M, K, N = 48, 80, 32                       # not a multiple of the 16 tile
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    got = np.array(ci.execute_idiom(ci.recognize(_GEMM), {"A": mx.array(A), "B": mx.array(B)},
                                    dims={"M": M, "K": K, "N": N}))
    ref = A.astype(np.float64) @ B.astype(np.float64)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-5
    n = 4194311                                # not a power of two
    x = rng.standard_normal(n).astype(np.float32)
    s = float(np.array(ci.execute_idiom(ci.recognize(_RED), {"in": mx.array(x)},
                                        dims={"n": n}))[0])
    assert abs(s - float(np.sum(x.astype(np.float64)))) / abs(float(np.sum(x.astype(np.float64)))) < 1e-4
