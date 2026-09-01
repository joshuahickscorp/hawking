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


def test_translates_cuda_neutral_zero_literal_without_changing_the_operation():
    """NVIDIA's canonical vectorAdd sample spells add as ``A[i] + B[i] + 0.0f``.

    The trailing literal is an f32 identity, so the frontend may normalize it to
    the already-supported add operation without inventing a new computation.
    """
    t = c2m.translate(k("if (i < n) c[i] = a[i] + b[i] + 0.0f;"), elements=64)
    assert t.program.ops[0].kind == "add"


def test_accepts_cuda_restrict_pointer_qualifier():
    src = ("__global__ void t(const float * __restrict__ a, "
           "const float * __restrict__ b, float * __restrict__ c, int n) { "
           + IDX + " if (i < n) c[i] = a[i] + b[i]; }")
    t = c2m.translate(src, elements=64)
    assert t.program.ops[0].kind == "add"


def test_neutral_literal_near_miss_stays_refused():
    with pytest.raises(c2m.C2MRefusal, match="outside the C2M-T0 pattern set"):
        c2m.translate(k("if (i < n) c[i] = a[i] + b[i] + 1.0f;"), elements=64)


@pytest.mark.parametrize("expr,alpha", [("a[i] * -1", -1.0), ("2.0f * a[i]", 2.0)])
def test_translates_numeric_scalar_elementwise_scale(expr, alpha):
    t = c2m.translate(k(f"if (i < n) c[i] = {expr};",
                        params="const float* a, float* c, int n"), elements=64)
    assert t.program.ops[0].kind == "scale"
    assert t.program.specialization["ALPHA"] == alpha


def test_runtime_scalar_remains_refused_until_scalar_argument_plumbing_exists():
    src = k("if (i < n) c[i] = a[i] * factor;",
            params="const float* a, float* c, float factor, int n")
    with pytest.raises(c2m.C2MRefusal, match="outside the C2M-T0 pattern set"):
        c2m.translate(src, elements=64)


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


def test_the_index_variable_is_discovered_not_assumed():
    """Three kernels in the pinned seed compute EXACTLY a[index] + b[index] -- a
    supported operation -- and were refused because the frontend hardcoded `i`. A
    supported operation rejected over a VARIABLE NAME. See
    ACCELERATOR_C2M_CORPUS_CENSUS.json."""
    for idx in ("i", "id", "index", "gid"):
        src = ("__global__ void vector_add(const float* a, const float* b, float* c) {"
               f"  int {idx} = blockIdx.x * blockDim.x + threadIdx.x;"
               f"  c[{idx}] = a[{idx}] + b[{idx}];"
               "}")
        t = c2m.translate(src, elements=64)
        assert t.program.ops[0].kind == "add"
        assert t.inputs == ["a", "b"]


def test_a_computed_index_that_is_never_named_is_refused():
    """The subset indexes THROUGH the name, so an index computed inline has nothing to
    match against and is refused with that reason rather than mistranslated."""
    src = ("__global__ void k(const float* a, const float* b, float* c) {"
           "  c[blockIdx.x * blockDim.x + threadIdx.x] = a[0] + b[0];"
           "}")
    with pytest.raises(c2m.C2MRefusal, match="not bound to a named variable"):
        c2m.translate(src, elements=64)


def test_the_recogniser_misses_the_real_idioms_on_a_DECLARATION_not_an_algorithm():
    """The seed carries a real tiled GEMM and a real block reduction, and the T2 door
    matched NEITHER -- failing on the first required fragment both times, because the
    real code writes `__shared__ float As[BLOCKSIZE * BLOCKSIZE]` where the recogniser
    demands a 2-D literal, and `extern __shared__ float sdata[]` where it demands a
    sized array. The T2 receipt PREDICTED this brittleness; this measures it."""
    gemm = ("__global__ void sgemm(const float* A, const float* B, float* C, int M,"
            " int N, int K) { __shared__ float As[BLOCKSIZE * BLOCKSIZE];"
            " __shared__ float Bs[BLOCKSIZE * BLOCKSIZE]; __syncthreads(); }")
    with pytest.raises(c2m.C2MRefusal, match="shared tile for A"):
        ci.recognize(gemm)
    red = ("__global__ void reduce(const float* in, float* out, int n) {"
           " extern __shared__ float sdata[]; int tid = threadIdx.x; }")
    with pytest.raises(c2m.C2MRefusal, match="shared buffer"):
        ci.recognize(red)


_REAL_SGEMM = """
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;
    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
"""


def test_the_real_sgemm_is_its_own_idiom_because_alpha_beta_is_a_different_function():
    """Widening tiled_gemm to match the seed's real kernel was the obvious fix and
    would have SUBSTITUTED A DIFFERENT COMPUTATION: this computes alpha*A@B + beta*C.
    The tiled door must still refuse it, and the new door must not claim the tile,
    which is a template parameter the kernel source does not contain."""
    rec = ci.recognize(_REAL_SGEMM)
    assert rec.idiom == "sgemm_alpha_beta"
    assert rec.tile is None
    with pytest.raises(c2m.C2MRefusal, match="shared tile for A"):
        ci._match_all(_REAL_SGEMM, ci._GEMM_FRAGMENTS, "tiled_gemm")
    with pytest.raises(c2m.C2MRefusal, match="TEMPLATE PARAMETER"):
        ci.execute_idiom(rec, {}, dims={"M": 8, "K": 8, "N": 8})


def test_the_naive_widening_would_have_passed_the_seeds_own_test():
    """The seed's main() calls this kernel with alpha=1 and beta=0, where a plain
    matmul agrees EXACTLY. A recogniser loosened to match the shape would have been
    correct on the only inputs the corpus exercises and wrong everywhere else."""
    pytest.importorskip("mlx.core")
    import numpy as _np, air, c2m_idiom
    rng = _np.random.default_rng(3)
    M = K = N = 32
    A = (rng.standard_normal((M, K)) * 0.3).astype(_np.float32)
    B = (rng.standard_normal((K, N)) * 0.3).astype(_np.float32)
    C0 = (rng.standard_normal((M, N)) * 0.3).astype(_np.float32)
    rec = ci.recognize(_REAL_SGEMM)
    plain = _np.asarray(air.execute_matmul(
        air.AirMatmul("plain", M, K, N, tile=16, strategy="tiled"), A, B))
    for alpha, beta, must_agree in ((1.0, 0.0, True), (2.0, 0.5, False)):
        got = _np.asarray(c2m_idiom.execute_idiom(
            rec, {"A": A, "B": B, "C": C0, "alpha": alpha, "beta": beta},
            dims={"M": M, "K": K, "N": N, "tile": 16}))
        ref = alpha * (A.astype(_np.float64) @ B.astype(_np.float64)) + beta * C0
        assert float(_np.max(_np.abs(got - ref)) / _np.max(_np.abs(ref))) < 1e-5
        plain_err = float(_np.max(_np.abs(plain - ref)) / _np.max(_np.abs(ref)))
        assert (plain_err < 1e-5) is must_agree


@pytest.mark.parametrize("label,old,new", [
    ("beta dropped", "alpha * tmp + beta * C[threadRow * N + threadCol]", "alpha * tmp"),
    ("B index transposed", "Bs[dotIdx * BLOCKSIZE + threadCol]",
     "Bs[threadCol * BLOCKSIZE + dotIdx]"),
    ("accumulator starts at one", "float tmp = 0.0f", "float tmp = 1.0f"),
    ("rectangular tile", "As[BLOCKSIZE * BLOCKSIZE]", "As[BLOCKSIZE * OTHERSIZE]"),
])
def test_sgemm_near_misses_are_refused(label, old, new):
    """Widening recognition without near-miss evidence is how a loose match ships."""
    with pytest.raises(c2m.C2MRefusal):
        ci.recognize(_REAL_SGEMM.replace(old, new, 1))


# ---------------------------------------------------------------------------
# The corpus denominator. ACCELERATOR_C2M_CORPUS_DENOMINATOR.json.
# ---------------------------------------------------------------------------

_EMPTY_PROBE = 'extern "C" __global__ void p(const float*, int64_t, float*) {}'


def test_empty_body_is_refused_for_being_empty_not_for_its_parameters():
    """The nine seed stubs were refused with 'cannot parse parameter const float*'.

    True, and the wrong cause: an unnamed parameter is fixable, a kernel with no body
    is not translatable by anything, ever. A refusal naming the wrong cause sends the
    reader to fix a parameter parser for a kernel that computes nothing.
    """
    with pytest.raises(c2m.C2MRefusal) as e:
        c2m.translate(_EMPTY_PROBE, elements=8)
    assert "EMPTY BODY" in str(e.value)
    assert "parameter" not in str(e.value)


def test_commuted_index_product_is_the_same_index():
    """blockDim.x * blockIdx.x is integer multiplication written the other way round.

    The addend order was already accepted and the factor order was not, which is what
    hid it. Two seed kernels spell it this way.
    """
    src = ("__global__ void t(const float* a, const float* b, float* c, int n) {"
           " int id = blockDim.x * blockIdx.x + threadIdx.x;"
           " if (id < n) { c[id] = a[id] + b[id]; } }")
    tk = c2m.translate(src, elements=64)
    assert tk.program.ops[0].kind == "add"


@pytest.mark.parametrize("idx,why", [
    ("int id = blockIdx.x + blockDim.x * threadIdx.x;", "a different index entirely"),
    ("int id = blockDim.y * blockIdx.x + threadIdx.x;", "wrong axis on the factor"),
    ("int id = blockIdx.x * blockDim.x + threadIdx.y;", "wrong axis on the addend"),
    ("int id = blockIdx.x * blockIdx.x + threadIdx.x;", "the same factor twice"),
])
def test_index_near_misses_stay_refused(idx, why):
    """The widening is exactly commutativity. Each of these MEANS something else."""
    src = ("__global__ void t(const float* a, const float* b, float* c, int n) {"
           f" {idx} if (id < n) {{ c[id] = a[id] + b[id]; }} }}")
    with pytest.raises(c2m.C2MRefusal):
        c2m.translate(src, elements=64)


def test_split_kernels_finds_every_kernel_in_one_file():
    """translate()'s greedy `\\{(.*)\\}` swallows every kernel after the first."""
    src = (f"__global__ void one(float* c) {{ {IDX} c[i] = 1.0f; }}\n"
           f"__global__ void two(float* c) {{ {IDX} c[i] = 2.0f; }}\n")
    assert [n for n, _, _ in c2m.split_kernels(src)] == ["one", "two"]


def test_census_keeps_empty_kernels_out_of_the_computing_denominator():
    r = c2m.census({"a.cu": _EMPTY_PROBE + "\n" + k("c[i] = a[i] + b[i];")})
    assert r["kernels"] == 2
    assert r["empty_body"] == 1
    assert r["computing_kernels"] == 1
    assert r["translated"] == 1


def test_census_counts_one_computation_written_two_ways_as_one():
    """Seven of the seed's kernels compute a[idx] + b[idx] under four spellings.

    Counting kernels instead of computations is how a corpus of one trivial map reads
    as coverage.
    """
    a = ("__global__ void one(const float* a, const float* b, float* c, int n)"
         " { int i = blockIdx.x * blockDim.x + threadIdx.x;"
         " if (i < n) { c[i] = a[i] + b[i]; } }")
    b = ("__global__ void two(const float* a, const float* b, float* c, int n)"
         " { int id = blockDim.x * blockIdx.x + threadIdx.x;"
         " if (id < n) { c[id] = a[id] + b[id]; } }")
    r = c2m.census({"a.cu": a, "b.cu": b})
    assert r["elementwise_shaped_upper_bound"] == 2
    assert r["distinct_elementwise_computations"] == 1


def test_elementwise_shaped_is_an_upper_bound_not_a_classification():
    """The cooperative check is a regex over the body, so a shuffle one function call
    away is invisible. The seed's softmax_warp is counted elementwise-shaped and is a
    warp reduction. Pinned as a KNOWN blindness rather than left for a reader to trust
    the count."""
    hidden = ("__global__ void t(const float* a, float* c, int n) {"
              " int i = blockIdx.x * blockDim.x + threadIdx.x;"
              " float v = warp_reduce_sum(a[i]);"
              " if (i < n) { c[i] = v; } }")
    r = c2m.census({"a.cu": hidden})
    assert r["elementwise_shaped_upper_bound"] == 1   # counted, and it is not one
    assert r["translated"] == 0                       # translate is not fooled


# --- the corpus that is not there ------------------------------------------------

def test_census_REFUSES_AN_EMPTY_CORPUS_rather_than_reporting_zero_coverage():
    """census({}) returned kernels 0, translated 0 and an empty refusal histogram --
    which reads EXACTLY like a frontend that translates nothing. It is not; it is a
    corpus that is not there, and the two must not read alike.

    LIVE, NOT HYPOTHETICAL: the pinned cuda-metal clone every C2M coverage number was
    measured over lived in a session scratchpad and has been reaped."""
    with pytest.raises(c2m.EmptyCorpus) as e:
        c2m.census({})
    assert "not there" in str(e.value) and "19a70230" in str(e.value), str(e.value)


def test_census_REFUSES_FILES_THAT_HOLD_NO_KERNELS_and_says_which_case():
    """Files present and no __global__ in them is a SECOND way to reach 0 of 0, and
    it needs its own message -- the first says the corpus is missing, this says the
    path is probably wrong."""
    with pytest.raises(c2m.EmptyCorpus) as e:
        c2m.census({"notes.txt": "int main() { return 0; }"})
    assert "ZERO __global__" in str(e.value) and "0 of 0" in str(e.value)


def test_census_STILL_MEASURES_A_REAL_CORPUS():
    """A refusal that fires on everything proves nothing. One real kernel must still
    produce a census with a nonzero denominator."""
    out = c2m.census({"k.cu": "__global__ void vadd(const float* a, const float* b, "
                              "float* c) { int i = blockIdx.x * blockDim.x + "
                              "threadIdx.x; c[i] = a[i] + b[i]; }"})
    assert out["kernels"] == 1 and out["computing_kernels"] == 1
