"""AIR — the Accelerator IR, and its Metal lowering. FRONT A (G043, steer S015).

AIR is Hawking's internal representation of GPU computation. This is the minimum
that is REAL rather than described: a program is built, lowered to MSL, executed on
the Apple GPU, and graded against a numpy oracle. The steer's proof P3 is exactly
that -- AIR executes a basic operation on Metal -- so nothing here is allowed to be
a schema with no execution behind it.

What AIR represents today: operations, operands with dtype and shape, dependencies
(via SSA names), device requirements, and specialization constants. What it does
NOT yet represent, stated so the gap is not mistaken for coverage: explicit
synchronization primitives, side effects, multi-device placement constraints, and
memory-domain requirements. Those arrive with HUMF (G050) and EGB (G051/G052); AIR
must not need redesign when they do, which is why placement is already a field on
the program rather than an afterthought.

Execution goes through mx.fast.metal_kernel because `xcrun metal` is not present on
this machine (no Metal developer toolchain), so ahead-of-time metallib compilation
is unavailable here. MLX JIT-compiles the MSL we emit. That is a real dependency
and it is recorded as one, not hidden.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

DTYPES = {"f32": "float", "f16": "half"}

# Memory domains an operand may be REQUIRED to live in. AIR carries the requirement;
# HUMF (G050) owns actually satisfying it. Keeping the vocabularies identical means a
# program can be handed to HUMF without translation.
MEMORY_DOMAINS = ("APPLE_UM", "MOCK_EXTERNAL_VRAM", "NVIDIA_VRAM_SIDECAR",
                  "NVIDIA_VRAM_DIRECT", "SSD_COLD", "HOST_LOGICAL", "ANY")

# Synchronization scopes. AIR can REPRESENT these; the current Metal backend can
# execute NONE of them, and lower_to_msl refuses rather than dropping them silently.
SYNC_SCOPES = ("THREADGROUP", "SIMDGROUP", "DEVICE")

# How a matmul is realised. Measured on this machine: simdgroup wins 1.42-1.62x over
# tiled at every size tried, but the two are kept as separate strategies rather than
# one being deleted, because tiled has no shape constraint while simdgroup requires
# every dimension to be a multiple of 8.
MATMUL_STRATEGIES = ("tiled", "simdgroup")


@dataclass(frozen=True)
class AirTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    memory_domain: str = "ANY"          # where this operand must live
    read_only: bool = True              # a side-effect declaration, not a hint

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; known: {sorted(DTYPES)}")
        if self.memory_domain not in MEMORY_DOMAINS:
            raise ValueError(f"unknown memory domain {self.memory_domain!r}; "
                             f"known: {MEMORY_DOMAINS}")
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError(f"bad shape {self.shape!r}")

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# Each op lowers to one elementwise MSL expression over index `i`. Deliberately
# narrow: this is the P3 corpus, not the whole math library the steer names.
# Operands arrive already resolved: a buffer reads as `name[i]`, an SSA
# intermediate as the scalar `name`. The templates must therefore never write
# `[i]` themselves -- doing so indexed a scalar and failed to compile.
ELEMENTWISE = {
    "add": "{a} + {b}",
    "mul": "{a} * {b}",
    "sub": "{a} - {b}",
    "saxpy": "ALPHA * {a} + {b}",
    "relu": "max(({t}){a}, ({t})0)",
    "silu": "{a} / ({t})(1.0 + exp(-(float){a}))",
}


@dataclass(frozen=True)
class AirBarrier:
    """A synchronization point. A silently dropped barrier is a race, which is the
    worst class of bug to ship quietly, so lowering REFUSES rather than drops.

    TWO of the three scopes now have an instruction (see barrier_msl); DEVICE does not
    and cannot, and the refusal says where the construct that DOES express it lives."""
    scope: str

    def __post_init__(self):
        if self.scope not in SYNC_SCOPES:
            raise ValueError(f"unknown sync scope {self.scope!r}; known: {SYNC_SCOPES}")


def barrier_msl(scope: str) -> str:
    """The Metal instruction for a sync scope, or a refusal that names the alternative.

    MEASURED, not assumed, and the two probes together say something sharper than
    either alone:

      WITHIN a simdgroup -- 32 lanes exchanging through threadgroup memory are correct
      with NO fence, with simdgroup_barrier, and with threadgroup_barrier, all three at
      max_abs_err 0.0. An Apple simdgroup runs in lockstep so the neighbour's write is
      already visible. SIMDGROUP IS THEREFORE NOT LOAD-BEARING FOR CORRECTNESS HERE.

      ACROSS simdgroups -- the same exchange at 256 threads reading 32 slots away is
      WRONG BY 9.854 with no fence AND WRONG BY EXACTLY THE SAME 9.854 with
      simdgroup_barrier, while threadgroup_barrier is exact. So the instruction's scope
      is precisely what its name says: no effect inside (unnecessary) and no effect
      outside (out of scope).

    It is still emitted, for two reasons that are not correctness-on-this-chip: it
    constrains the COMPILER's reordering, and lockstep is a HARDWARE PROPERTY that a
    future device need not preserve. But calling SIMDGROUP a closed synchronization gap
    would be overreach and the receipt says so.

    DEVICE has no in-kernel instruction in Metal at all. Device-wide ordering IS a
    command-buffer boundary, and AIR already expresses that -- as an AirGraph EDGE, not
    as a barrier. Refusing while naming that is worth more than refusing.
    """
    if scope == "THREADGROUP":
        return "threadgroup_barrier(mem_flags::mem_threadgroup);"
    if scope == "SIMDGROUP":
        return "simdgroup_barrier(mem_flags::mem_threadgroup);"
    if scope == "DEVICE":
        raise NotImplementedError(
            "DEVICE-scope synchronization has no Metal instruction inside a kernel: "
            "device-wide ordering IS a command-buffer boundary. AIR expresses it as an "
            "AirGraph dependency EDGE between nodes, which the runtime turns into "
            "ordered dispatches. Use AirGraph; a barrier cannot carry this.")
    raise ValueError(f"unknown sync scope {scope!r}")


@dataclass(frozen=True)
class AirOp:
    kind: str
    inputs: tuple[str, ...]
    output: str
    attrs: dict[str, Any] = field(default_factory=dict)
    writes_external: bool = False       # declares a side effect beyond `output`

    def __post_init__(self):
        if self.kind not in ELEMENTWISE:
            raise ValueError(f"unknown AIR op {self.kind!r}; known: {sorted(ELEMENTWISE)}")
        want = 1 if self.kind in ("relu", "silu") else 2
        if len(self.inputs) != want:
            raise ValueError(f"{self.kind} takes {want} input(s), got {len(self.inputs)}")


@dataclass
class AirProgram:
    """A program plus its device requirement. Placement is a field from the start so
    that EGB does not later force an AIR redesign (steer: 'EGB must not force AIR
    redesign')."""
    name: str
    inputs: list[AirTensor]
    ops: list[AirOp]
    output: str
    device: str = "APPLE_GPU_0"
    specialization: dict[str, float] = field(default_factory=dict)
    barriers: list[AirBarrier] = field(default_factory=list)
    output_domain: str = "ANY"

    def has_side_effects(self) -> bool:
        return any(o.writes_external for o in self.ops) or any(
            not t.read_only for t in self.inputs)

    def required_domains(self) -> dict[str, str]:
        d = {t.name: t.memory_domain for t in self.inputs if t.memory_domain != "ANY"}
        if self.output_domain != "ANY":
            d[self.output] = self.output_domain
        return d

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        """What AIR can REPRESENT is deliberately wider than what this backend can
        RUN. Saying which is which is the whole point."""
        if self.barriers:
            # THE ORIGINAL REASON HERE WAS WRONG AND IT TAUGHT THE WRONG LESSON. It
            # said "the Metal backend has no synchronization lowering" -- but AIR's
            # matmul, softmax and attention all emit threadgroup barriers, and
            # barrier_msl() now supplies THREADGROUP and SIMDGROUP instructions. The
            # binding constraint is not the backend, it is THIS PROGRAM: an elementwise
            # AIR program declares no threadgroup allocation and no cross-thread read,
            # so every thread is independent and a barrier of ANY scope has NOTHING TO
            # ORDER. A refusal that names the wrong cause is worse than a terse one.
            scopes = sorted({b.scope for b in self.barriers})
            return False, (f"{len(self.barriers)} barrier(s) declared at {scopes}, but "
                           f"an elementwise program has no threadgroup allocation and "
                           f"no cross-thread reads, so there is nothing for a barrier "
                           f"to order. The constraint is the PROGRAM SHAPE, not the "
                           f"backend: barrier_msl() supplies THREADGROUP and SIMDGROUP "
                           f"instructions and AIR's matmul lowering already emits them. "
                           f"DEVICE scope has no instruction at all -- that ordering is "
                           f"an AirGraph edge, not a barrier.")
        if self.has_side_effects():
            return False, ("the program declares side effects beyond its output; the "
                           "elementwise lowering assumes pure ops")
        foreign = {t.memory_domain for t in self.inputs
                   if t.memory_domain not in ("ANY", "APPLE_UM")}
        if foreign:
            return False, (f"operands require {sorted(foreign)}, which this backend "
                           f"cannot reach; HUMF must materialise them in APPLE_UM first")
        return True, "pure elementwise program over Apple unified memory"

    def validate(self) -> None:
        live = {t.name for t in self.inputs}
        for op in self.ops:
            missing = [n for n in op.inputs if n not in live]
            if missing:
                raise ValueError(f"op {op.kind} reads undefined {missing}")
            if op.output in live:
                raise ValueError(f"{op.output} assigned twice; AIR is SSA")
            live.add(op.output)
        if self.output not in live:
            raise ValueError(f"program output {self.output!r} is never produced")
        shapes = {t.shape for t in self.inputs}
        if len(shapes) != 1:
            raise ValueError(f"elementwise AIR needs one shape, got {shapes}")
        dts = {t.dtype for t in self.inputs}
        if len(dts) != 1:
            raise ValueError(f"mixed dtypes not supported yet: {dts}")

    @property
    def dtype(self) -> str:
        return self.inputs[0].dtype


def lower_to_msl(prog: AirProgram) -> str:
    """AIR -> MSL body. MLX supplies the kernel signature and thread position."""
    prog.validate()
    ok, why = prog.executable_on_metal_backend()
    if not ok:
        raise NotImplementedError(
            f"AIR represents this program but the Metal backend cannot execute it: "
            f"{why}. Refusing to lower rather than emitting something that ignores it.")
    t = DTYPES[prog.dtype]
    lines = ["uint i = thread_position_in_grid.x;",
             f"if (i >= {prog.inputs[0].elements}u) return;"]
    for k, v in prog.specialization.items():
        lines.append(f"const float {k} = {float(v)!r}f;")
    buffers = {x.name for x in prog.inputs}          # indexed with [i]
    scalars: set[str] = set()                        # SSA intermediates
    for op in prog.ops:
        def ref(nm: str) -> str:
            return f"{nm}[i]" if nm in buffers else nm
        rhs = ELEMENTWISE[op.kind].format(
            a=ref(op.inputs[0]),
            b=ref(op.inputs[1]) if len(op.inputs) > 1 else "",
            t=t)
        if op.output == prog.output:
            # only the final op writes the buffer MLX allocated for us
            lines.append(f"out[i] = ({t})({rhs});")
        else:
            lines.append(f"{t} {op.output} = ({t})({rhs});")
            scalars.add(op.output)
    return "\n    ".join(lines)


@dataclass
class AirMatmul:
    """A tiled matmul as an AIR PROGRAM, not a hand-written kernel sitting beside AIR.

    This exists because the GEMM receipt recorded exactly that gap: AIR had no tiling,
    no threadgroup allocation and only barrier REPRESENTATION rather than lowering, so
    the working kernel was MSL I wrote by hand. Now the tile size is a specialization,
    the threadgroup allocation is declared, and the barriers are EMITTED BY THE
    LOWERING -- which is what makes the THREADGROUP scope executable rather than
    refused.
    """
    name: str
    m: int
    k: int
    n: int
    tile: int = 16
    strategy: str = "tiled"          # "tiled" | "simdgroup"
    block: int = 1                   # simdgroup only: NxN grid of 8x8 tiles per simdgroup
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"
    a_domain: str = "ANY"
    b_domain: str = "ANY"

    def validate(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.strategy not in MATMUL_STRATEGIES:
            raise ValueError(f"unknown matmul strategy {self.strategy!r}; "
                             f"known: {MATMUL_STRATEGIES}")
        if self.strategy == "simdgroup":
            if self.dtype != "f32":
                raise ValueError("the simdgroup strategy is f32 only here")
            if self.block not in (1, 2):
                raise ValueError(f"block {self.block} not supported; only 1 and 2 are "
                                 f"implemented")
            step = 8 * self.block
            for d, nm in ((self.m, "m"), (self.k, "k"), (self.n, "n")):
                if d % 8:
                    raise ValueError(f"simdgroup matrices are 8x8, so {nm}={d} must be "
                                     f"a multiple of 8; refusing rather than emitting a "
                                     f"kernel that would read out of bounds")
            for d, nm in ((self.m, "m"), (self.n, "n")):
                if d % step:
                    raise ValueError(f"block={self.block} makes each simdgroup cover "
                                     f"{step}x{step}, so {nm}={d} must be a multiple of "
                                     f"{step}")
        if self.tile <= 0 or self.tile & (self.tile - 1):
            raise ValueError(f"tile {self.tile} must be a positive power of two")
        if self.tile > 32:
            raise ValueError(f"tile {self.tile} exceeds 32; a {self.tile}x{self.tile} "
                             f"threadgroup would exceed Metal's 1024-thread limit")
        for d in (self.a_domain, self.b_domain):
            if d not in MEMORY_DOMAINS:
                raise ValueError(f"unknown memory domain {d!r}")

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        foreign = {d for d in (self.a_domain, self.b_domain)
                   if d not in ("ANY", "APPLE_UM")}
        if foreign:
            return False, (f"operands require {sorted(foreign)}; HUMF must materialise "
                           f"them in APPLE_UM first")
        # THREADGROUP barriers ARE executable now: this lowering emits them itself.
        return True, "tiled matmul with threadgroup tiles and emitted barriers"

    def barrier_scopes_emitted(self) -> list[str]:
        if self.strategy == "simdgroup":
            # a simdgroup executes in lockstep, so no explicit barrier is emitted --
            # the synchronization is implicit in the SIMD width
            return []
        return ["THREADGROUP", "THREADGROUP"]   # one after load, one after accumulate

    def out_shape(self) -> tuple[int, int]:
        return (self.m, self.n)

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Grid and threadgroup for this strategy. They differ, so the caller must not
        guess."""
        if self.strategy == "simdgroup":
            s = 8 * self.block
            return (self.n // s * 32, self.m // s, 1), (32, 1, 1)
        # ROUND UP TO WHOLE THREADGROUPS. The kernel body is fully guarded -- it checks
        # gy < m, gx < n and k0 + lx < k -- so the tile has never been the constraint.
        # The LAUNCH was: returning (n, m, 1) against a (tile, tile) threadgroup
        # under-dispatches whenever m is not a multiple of the tile, and the result is
        # WRONG rather than refused: 8x64x32 at tile 16 was off by 4.65 and the
        # 60x60x60 the simdgroup receipt cited as tiled's own justification was off by
        # 3.84. See ACCELERATOR_GRAPH_COMPOSITION.json.
        t = self.tile
        return (((self.n + t - 1) // t) * t, ((self.m + t - 1) // t) * t, 1), (t, t, 1)


def lower_matmul_to_msl(mm: AirMatmul) -> str:
    """AIR -> tiled MSL. The barriers here are EMITTED BY AIR."""
    mm.validate()
    ok, why = mm.executable_on_metal_backend()
    if not ok:
        raise NotImplementedError(f"AIR represents this matmul but the Metal backend "
                                  f"cannot execute it: {why}")
    T, ty = mm.tile, DTYPES[mm.dtype]
    if mm.strategy == "simdgroup":
        B_, K_, N_ = mm.block, mm.k, mm.n
        s = 8 * B_
        decl = "\n    ".join(
            f"simdgroup_float8x8 acc{i}{j} = make_filled_simdgroup_matrix<float,8,8>(0.0f);"
            for i in range(B_) for j in range(B_))
        loads = "\n        ".join(
            [f"simdgroup_float8x8 a{i};" for i in range(B_)] +
            [f"simdgroup_float8x8 b{j};" for j in range(B_)] +
            [f"simdgroup_load(a{i}, A + (row + {8*i}u) * {K_}u + k, {K_}u);"
             for i in range(B_)] +
            [f"simdgroup_load(b{j}, B + k * {N_}u + col + {8*j}u, {N_}u);"
             for j in range(B_)])
        macs = "\n        ".join(
            f"simdgroup_multiply_accumulate(acc{i}{j}, a{i}, b{j}, acc{i}{j});"
            for i in range(B_) for j in range(B_))
        stores = "\n    ".join(
            f"simdgroup_store(acc{i}{j}, C + (row + {8*i}u) * {N_}u + col + {8*j}u, {N_}u);"
            for i in range(B_) for j in range(B_))
        return f"""
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint row = tg_y * {s}u;
    uint col = tg_x * {s}u;
    {decl}
    for (uint k = 0; k < {K_}u; k += 8u) {{
        {loads}
        {macs}
    }}
    {stores}
"""
    return f"""
    uint gx = thread_position_in_grid.x;
    uint gy = thread_position_in_grid.y;
    uint lx = thread_position_in_threadgroup.x;
    uint ly = thread_position_in_threadgroup.y;
    threadgroup {ty} As[{T}][{T}];
    threadgroup {ty} Bs[{T}][{T}];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < {mm.k}u; k0 += {T}u) {{
        As[ly][lx] = (gy < {mm.m}u && (k0 + lx) < {mm.k}u)
                   ? A[gy * {mm.k}u + k0 + lx] : ({ty})0;
        Bs[ly][lx] = ((k0 + ly) < {mm.k}u && gx < {mm.n}u)
                   ? B[(k0 + ly) * {mm.n}u + gx] : ({ty})0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < {T}u; ++k) acc += (float)As[ly][k] * (float)Bs[k][lx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (gy < {mm.m}u && gx < {mm.n}u) C[gy * {mm.n}u + gx] = ({ty})acc;
"""


def execute_matmul(mm: AirMatmul, a, b):
    """Run an AIR matmul on the Apple GPU."""
    import mlx.core as mx
    src = lower_matmul_to_msl(mm)
    hdr = "#include <metal_simdgroup_matrix>\n" if mm.strategy == "simdgroup" else ""
    kern = mx.fast.metal_kernel(name=f"air_mm_{mm.strategy}_{mm.name}",
                                input_names=["A", "B"], output_names=["C"], source=src,
                                header=hdr, ensure_row_contiguous=True)
    dt = mx.float32 if mm.dtype == "f32" else mx.float16
    grid, tg = mm.launch()
    (c,) = kern(inputs=[mx.array(a, dtype=dt), mx.array(b, dtype=dt)],
                grid=grid, threadgroup=tg,
                output_shapes=[(mm.m, mm.n)], output_dtypes=[dt])
    mx.eval(c)
    return c


REDUCE_OPS = ("sum", "max")


@dataclass
class AirSoftmax:
    """Row-wise softmax as ONE fused kernel. FRONT D / the CCL's attention prerequisite.

    The naive shape is three passes over the row -- max, then sum of exp, then divide --
    each re-reading data the previous pass already touched. This does all three inside
    one threadgroup with two barriers, so the row is read once into registers and the
    reductions happen in the SIMD units. That fusion is only expressible because
    barriers and simd reductions now lower, which is what the reduction work bought.
    """
    name: str
    rows: int
    cols: int
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("softmax is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")

    def barrier_scopes_emitted(self) -> list[str]:
        return ["THREADGROUP", "THREADGROUP"]     # after the max, after the sum

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "fused row softmax with two threadgroup reductions"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.rows * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)


def lower_softmax_to_msl(sm: AirSoftmax) -> str:
    sm.validate()
    lanes = sm.threadgroup // 32
    return f"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    // WRITE-AFTER-READ: the second reduction gets ITS OWN SLOTS rather than reusing
    // the first's. Reusing them needs a barrier between every thread's READ of the
    // first result and the first thread's WRITE of the second, and there was none --
    // found when the intact LayerNorm came out wrong by 0.061 at threadgroup 1024
    // while exact at 64 and 256. Separate slots remove the hazard structurally, which
    // beats a third barrier: there is nothing left to forget.
    threadgroup float red[{2 * lanes}];
    uint base = row * {sm.cols}u;

    // pass 1: row max, for numerical stability
    float m = -INFINITY;
    for (uint c = lid; c < {sm.cols}u; c += {sm.threadgroup}u) m = max(m, x[base + c]);
    float sm_ = simd_max(m);
    if (lane == 0u) red[warp] = sm_;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowmax = -INFINITY;
    for (uint i = 0; i < {lanes}u; ++i) rowmax = max(rowmax, red[i]);

    // pass 2: sum of exp, same threads, no re-launch
    float s = 0.0f;
    for (uint c = lid; c < {sm.cols}u; c += {sm.threadgroup}u) s += exp(x[base + c] - rowmax);
    float ss = simd_sum(s);
    if (lane == 0u) red[{lanes}u + warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowsum = 0.0f;
    for (uint i = 0; i < {lanes}u; ++i) rowsum += red[{lanes}u + i];

    // pass 3: normalise
    float inv = 1.0f / rowsum;
    for (uint c = lid; c < {sm.cols}u; c += {sm.threadgroup}u)
        out[base + c] = exp(x[base + c] - rowmax) * inv;
"""


def execute_softmax(sm: AirSoftmax, x):
    import mlx.core as mx
    src = lower_softmax_to_msl(sm)
    kern = mx.fast.metal_kernel(name=f"air_softmax_{sm.name}", input_names=["x"],
                                output_names=["out"], source=src,
                                ensure_row_contiguous=True)
    g, tg = sm.launch()
    (o,) = kern(inputs=[mx.array(x, dtype=mx.float32)], grid=g, threadgroup=tg,
                output_shapes=[(sm.rows, sm.cols)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o



@dataclass
class AirReduce:
    """A full reduction as an AIR program. FRONT D / CCL MATH.reduction_and_scan.

    Two stage rather than atomic: each threadgroup reduces its slice to one partial
    using simd_sum plus a threadgroup barrier, then the partials are reduced again.
    Atomics would collapse this to one pass but atomics are still a LARGE gap in the
    ledger, and borrowing an unbuilt capability to make this look simpler would be a
    lie about what executes.
    """
    name: str
    n: int
    op: str = "sum"
    threadgroup: int = 256
    strategy: str = "two_stage"      # "two_stage" | "atomic"
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.op not in REDUCE_OPS:
            raise ValueError(f"unknown reduce op {self.op!r}; known: {REDUCE_OPS}")
        if self.dtype != "f32":
            raise ValueError("reductions are f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of the "
                             f"32-wide SIMD group and within Metal's 1024 limit")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.strategy not in ("two_stage", "atomic"):
            raise ValueError(f"unknown reduce strategy {self.strategy!r}")
        if self.strategy == "atomic" and self.op != "sum":
            raise ValueError("the atomic strategy implements sum only; Metal has no "
                             "float atomic max here, so max must use two_stage")

    def partials(self) -> int:
        return (self.n + self.threadgroup - 1) // self.threadgroup

    def barrier_scopes_emitted(self) -> list[str]:
        return ["THREADGROUP"]

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "two-stage reduction with simd_sum and one threadgroup barrier"


def lower_reduce_to_msl(rd: AirReduce) -> str:
    rd.validate()
    if rd.strategy == "atomic":
        lanes = rd.threadgroup // 32
        return f"""
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    threadgroup float red[{lanes}];
    float v = (gid < {rd.n}u) ? x[gid] : 0.0f;
    float s = simd_sum(v);
    if (lane == 0u) red[lid / 32u] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {{
        float acc = 0.0f;
        for (uint i = 0; i < {lanes}u; ++i) acc += red[i];
        atomic_fetch_add_explicit(&out[0], acc, memory_order_relaxed);
    }}
"""
    ident = "0.0f" if rd.op == "sum" else "-INFINITY"
    simd = "simd_sum" if rd.op == "sum" else "simd_max"
    comb = "acc += partial[i];" if rd.op == "sum" else "acc = max(acc, partial[i]);"
    lanes = rd.threadgroup // 32
    return f"""
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    threadgroup float partial[{lanes}];
    float v = (gid < {rd.n}u) ? x[gid] : ({ident});
    float s = {simd}(v);
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    if (lane == 0u) partial[warp] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {{
        float acc = {ident};
        for (uint i = 0; i < {lanes}u; ++i) {{ {comb} }}
        out[tgid] = acc;
    }}
"""


def execute_reduce(rd: AirReduce, x):
    """Stage one on the GPU; the partials are reduced again until one value remains.

    The atomic strategy does it in ONE launch instead, which is what the two-stage
    receipt named as the reason it lost to MLX by 3x.
    """
    import mlx.core as mx
    rd.validate()
    if rd.strategy == "atomic":
        kern = metal_kernel(mx, name=f"air_red_atomic_{rd.name}", input_names=["x"],
                            output_names=["out"],
                            source=lower_reduce_to_msl(rd),
                            ensure_row_contiguous=True, atomic_outputs=True)
        (o,) = kern(inputs=[mx.array(x, dtype=mx.float32)],
                    grid=(rd.partials() * rd.threadgroup, 1, 1),
                    threadgroup=(rd.threadgroup, 1, 1),
                    output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)
        mx.eval(o)
        return o
    cur = mx.array(x, dtype=mx.float32)
    n = rd.n
    while n > 1:
        step = AirReduce(rd.name, n, rd.op, rd.threadgroup)
        k2 = metal_kernel(mx, name=f"air_red_{rd.op}_{n}", input_names=["x"],
                          output_names=["out"],
                          source=lower_reduce_to_msl(step),
                          ensure_row_contiguous=True)
        p = step.partials()
        (cur,) = k2(inputs=[cur], grid=(p * step.threadgroup, 1, 1),
                    threadgroup=(step.threadgroup, 1, 1),
                    output_shapes=[(p,)], output_dtypes=[mx.float32])
        n = p
    mx.eval(cur)      # ONE host round trip for the whole tree, not one per level.
    return cur        # The per-level eval was a defect; see ACCELERATOR_SCAN.json.


@dataclass
class AirAttention:
    """Single-head scaled dot-product attention as ONE fused kernel.

    scores = Q K^T / sqrt(d); P = softmax(scores); O = P V -- computed by one
    threadgroup per query row, with the score row held in threadgroup memory so it is
    never written to device memory at all. The naive shape materialises an
    (seq x seq) score matrix and a second (seq x seq) probability matrix; this writes
    neither.

    THE SCORE ROW LIVES IN THREADGROUP MEMORY, which caps seq_len. That cap is
    enforced rather than assumed: a longer sequence is REFUSED, because the honest
    answer is that this kernel does not do flash-attention's online softmax and
    therefore cannot stream an unbounded sequence.
    """
    name: str
    seq_q: int
    seq_k: int
    head_dim: int
    threadgroup: int = 256
    causal: bool = False
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    # 32 KiB of threadgroup memory is the Metal limit; the score row plus the
    # reduction scratch must fit inside it.
    MAX_THREADGROUP_BYTES: ClassVar[int] = 32 * 1024

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("attention is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        need = self.seq_k * 4 + (self.threadgroup // 32) * 4
        if need > self.MAX_THREADGROUP_BYTES:
            raise ValueError(
                f"seq_k={self.seq_k} needs {need} bytes of threadgroup memory, over the "
                f"{self.MAX_THREADGROUP_BYTES} limit. This kernel holds the whole score "
                f"row in threadgroup memory and does NOT implement flash-attention's "
                f"online softmax, so it cannot stream a longer sequence. Refusing "
                f"rather than silently truncating.")
        if min(self.seq_q, self.seq_k, self.head_dim) <= 0:
            raise ValueError("all dimensions must be positive")

    def barrier_scopes_emitted(self) -> list[str]:
        return ["THREADGROUP"] * 4

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "fused single-head attention, score row in threadgroup memory"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.seq_q * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)

    def materialised_bytes_avoided(self) -> int:
        """What the naive shape would write to device memory and this does not."""
        return 2 * self.seq_q * self.seq_k * 4


def lower_attention_to_msl(at: AirAttention) -> str:
    at.validate()
    lanes = at.threadgroup // 32
    scale = 1.0 / (at.head_dim ** 0.5)
    causal = ("if (j > q) s = -INFINITY;" if at.causal else "")
    return f"""
    uint q = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    threadgroup float scores[{at.seq_k}];
    // WRITE-AFTER-READ: the second reduction gets ITS OWN SLOTS rather than reusing
    // the first's. Reusing them needs a barrier between every thread's READ of the
    // first result and the first thread's WRITE of the second, and there was none --
    // found when the intact LayerNorm came out wrong by 0.061 at threadgroup 1024
    // while exact at 64 and 256. Separate slots remove the hazard structurally, which
    // beats a third barrier: there is nothing left to forget.
    threadgroup float red[{2 * lanes}];

    // scores = Q . K^T * scale, held in threadgroup memory and never written out
    for (uint j = lid; j < {at.seq_k}u; j += {at.threadgroup}u) {{
        float s = 0.0f;
        for (uint d = 0; d < {at.head_dim}u; ++d)
            s += Q[q * {at.head_dim}u + d] * K[j * {at.head_dim}u + d];
        s *= {scale}f;
        {causal}
        scores[j] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    for (uint j = lid; j < {at.seq_k}u; j += {at.threadgroup}u) m = max(m, scores[j]);
    float mm = simd_max(m);
    if (lane == 0u) red[warp] = mm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowmax = -INFINITY;
    for (uint i = 0; i < {lanes}u; ++i) rowmax = max(rowmax, red[i]);

    float acc = 0.0f;
    for (uint j = lid; j < {at.seq_k}u; j += {at.threadgroup}u) {{
        float e = exp(scores[j] - rowmax);
        scores[j] = e;
        acc += e;
    }}
    float ss = simd_sum(acc);
    if (lane == 0u) red[{lanes}u + warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowsum = 0.0f;
    for (uint i = 0; i < {lanes}u; ++i) rowsum += red[{lanes}u + i];
    float inv = 1.0f / rowsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O = P V, one output element per thread stride
    for (uint d = lid; d < {at.head_dim}u; d += {at.threadgroup}u) {{
        float o = 0.0f;
        for (uint j = 0; j < {at.seq_k}u; ++j) o += scores[j] * V[j * {at.head_dim}u + d];
        out[q * {at.head_dim}u + d] = o * inv;
    }}
"""


def execute_attention(at: AirAttention, q, k, v):
    import mlx.core as mx
    src = lower_attention_to_msl(at)
    kern = mx.fast.metal_kernel(name=f"air_attn_{at.name}",
                                input_names=["Q", "K", "V"], output_names=["out"],
                                source=src, ensure_row_contiguous=True)
    g, tg = at.launch()
    (o,) = kern(inputs=[mx.array(q, dtype=mx.float32), mx.array(k, dtype=mx.float32),
                        mx.array(v, dtype=mx.float32)],
                grid=g, threadgroup=tg,
                output_shapes=[(at.seq_q, at.head_dim)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


def machine_identity() -> dict[str, Any]:
    """MachineIdentity for the receipt. The steer: no result without physical
    identity."""
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {
        "soc": sysctl("machdep.cpu.brand_string"),
        "cpu_cores": sysctl("hw.ncpu"),
        "memory_bytes": sysctl("hw.memsize"),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
    }


def execute(prog: AirProgram, arrays: dict[str, Any]):
    """Run the lowered program on the Apple GPU via MLX's JIT."""
    import mlx.core as mx
    prog.validate()
    src = lower_to_msl(prog)
    names = [t.name for t in prog.inputs]
    kern = mx.fast.metal_kernel(
        name=f"air_{prog.name}", input_names=names, output_names=["out"], source=src)
    n = prog.inputs[0].elements
    mxdt = mx.float32 if prog.dtype == "f32" else mx.float16
    (out,) = kern(
        inputs=[mx.array(arrays[k], dtype=mxdt) for k in names],
        grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
        output_shapes=[prog.inputs[0].shape], output_dtypes=[mxdt])
    mx.eval(out)
    return out


SCAN_OPS = ("sum", "max")


@dataclass
class AirScan:
    """A prefix scan as an AIR program. FRONT D / CCL MATH.reduction_and_scan.

    Scan is the piece the reduction receipt named as ABSENT: "SCAN STILL DOES NOT
    EXIST". It is harder than reduction for one structural reason -- a reduction
    throws away all but one value, so a threadgroup can finish independently, while
    a scan needs every earlier block's total before any of its own outputs are
    final. That forces the three-phase shape: scan within blocks, scan the block
    totals, then add the offsets back.

    TWO STRATEGIES, KEPT FOR THE SAME REASON THE MATMUL KEEPS TWO:

      simd_prefix    -- Apple's simd_prefix_inclusive_sum does a 32-wide scan in the
                        SIMD unit with no threadgroup memory and no barrier. Sum
                        ONLY: Metal provides prefix sum and product, and NO prefix
                        max, so op="max" is refused on this strategy rather than
                        silently falling back.
      hillis_steele  -- the textbook CUDA-era shape: the whole block in threadgroup
                        memory, log2(threadgroup) doubling steps, 2 barriers each.
                        Works for ANY associative op including max, and is the
                        honest baseline for asking whether the vendor primitive is
                        worth anything (S015 §141).
    """
    name: str
    n: int
    op: str = "sum"
    inclusive: bool = True
    threadgroup: int = 256
    strategy: str = "simd_prefix"     # "simd_prefix" | "hillis_steele"
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.op not in SCAN_OPS:
            raise ValueError(f"unknown scan op {self.op!r}; known: {SCAN_OPS}")
        if self.dtype != "f32":
            raise ValueError("scans are f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of the "
                             f"32-wide SIMD group and within Metal's 1024 limit")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.strategy not in ("simd_prefix", "hillis_steele"):
            raise ValueError(f"unknown scan strategy {self.strategy!r}")
        if self.strategy == "simd_prefix" and self.op != "sum":
            raise ValueError(
                "the simd_prefix strategy implements sum only. Metal provides "
                "simd_prefix_inclusive_sum and simd_prefix_inclusive_product but NO "
                "prefix max, so op='max' must use hillis_steele. Refusing rather "
                "than falling back silently, because a strategy that quietly becomes "
                "a different strategy makes its own benchmark meaningless.")
        if not self.inclusive and self.op != "sum":
            raise ValueError(
                "exclusive scan is implemented for sum only (it is derived as "
                "inclusive - x). Exclusive max is NOT IMPOSSIBLE, just not built: "
                "it needs a shift rather than a subtraction. Not implemented is "
                "what this says, not unsupported by the hardware.")

    def blocks(self) -> int:
        return (self.n + self.threadgroup - 1) // self.threadgroup

    def barrier_scopes_emitted(self) -> list[str]:
        if self.strategy == "simd_prefix":
            return ["THREADGROUP"] * 2
        # buffer load, then 2 per doubling step
        steps = max(1, (self.threadgroup - 1).bit_length())
        return ["THREADGROUP"] * (1 + 2 * steps)

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, f"three-phase {self.strategy} scan, block totals scanned recursively"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.blocks() * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)


def lower_scan_block_to_msl(sc: AirScan) -> str:
    """Phase one: scan within each block, and write each block's total to sums[]."""
    sc.validate()
    if sc.strategy == "simd_prefix":
        lanes = sc.threadgroup // 32
        return f"""
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    threadgroup float wsum[{lanes}];
    float v = (gid < {sc.n}u) ? x[gid] : 0.0f;
    float incl = simd_prefix_inclusive_sum(v);
    if (lane == 31u) wsum[warp] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {{
        float acc = 0.0f;
        for (uint i = 0; i < {lanes}u; ++i) {{ float t = wsum[i]; wsum[i] = acc; acc += t; }}
        sums[tgid] = acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float res = incl + wsum[warp];
    if (gid < {sc.n}u) out[gid] = res;
"""
    ident = "0.0f" if sc.op == "sum" else "-INFINITY"
    comb = "a + b" if sc.op == "sum" else "max(a, b)"
    steps = max(1, (sc.threadgroup - 1).bit_length())
    return f"""
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    threadgroup float buf[{sc.threadgroup}];
    float v = (gid < {sc.n}u) ? x[gid] : ({ident});
    buf[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < {steps}u; ++s) {{
        uint off = 1u << s;
        float a = buf[lid];
        float b = (lid >= off) ? buf[lid - off] : ({ident});
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[lid] = {comb};
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float res = buf[lid];
    if (lid == {sc.threadgroup - 1}u) sums[tgid] = res;
    if (gid < {sc.n}u) out[gid] = res;
"""


def lower_scan_offset_to_msl(sc: AirScan) -> str:
    """Phase three: fold each block's exclusive prefix back into its elements."""
    comb = "y[gid] + off[tgid]" if sc.op == "sum" else "max(y[gid], off[tgid])"
    return f"""
    uint gid = thread_position_in_grid.x;
    uint tgid = threadgroup_position_in_grid.x;
    if (gid < {sc.n}u) out[gid] = {comb};
"""


_KERNEL_CACHE: dict[tuple, Any] = {}


def metal_kernel(mx, *, name, input_names, output_names, source, **kw):
    """Build a metal_kernel once and reuse it.

    Not a micro-optimization: building the wrapper on every call puts Python object
    construction and MLX's compiled-kernel lookup INSIDE any timing loop that calls
    it, which showed up as 21-24% IQR on an arm whose MLX baseline measured 3%. A
    benchmark that times its own harness is measuring the wrong thing.
    """
    key = (name, source, tuple(input_names), tuple(output_names),
           tuple(sorted(kw.items())))
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=name, input_names=list(input_names), output_names=list(output_names),
            source=source, **kw)
    return k


def _scan_inclusive_gpu(sc: AirScan, arr, mx):
    """Inclusive scan of `arr` (an mx array of length sc.n). Recursive on blocks."""
    nb = sc.blocks()
    blk = metal_kernel(
        mx, name=f"air_scan_{sc.strategy}_{sc.op}_{sc.n}",
        input_names=["x"], output_names=["out", "sums"],
        source=lower_scan_block_to_msl(sc), ensure_row_contiguous=True)
    y, sums = blk(inputs=[arr], grid=(nb * sc.threadgroup, 1, 1),
                  threadgroup=(sc.threadgroup, 1, 1),
                  output_shapes=[(sc.n,), (nb,)],
                  output_dtypes=[mx.float32, mx.float32])
    if nb == 1:
        return y
    inner = AirScan(sc.name, nb, sc.op, True, sc.threadgroup, sc.strategy)
    sums_incl = _scan_inclusive_gpu(inner, sums, mx)
    ident = 0.0 if sc.op == "sum" else float("-inf")
    offs = mx.concatenate([mx.array([ident], dtype=mx.float32), sums_incl[:-1]])
    add = metal_kernel(
        mx, name=f"air_scanoff_{sc.op}_{sc.n}", input_names=["y", "off"],
        output_names=["out"], source=lower_scan_offset_to_msl(sc),
        ensure_row_contiguous=True)
    (z,) = add(inputs=[y, offs], grid=(nb * sc.threadgroup, 1, 1),
               threadgroup=(sc.threadgroup, 1, 1),
               output_shapes=[(sc.n,)], output_dtypes=[mx.float32])
    return z


def execute_scan(sc: AirScan, x):
    """Full-array scan on the Apple GPU.

    Three phases because a scan cannot be block-local: block scan, scan of block
    totals (recursively, so 2^24 elements is 3 levels not a serial pass), then the
    offset fold. Exclusive is derived from inclusive by subtracting the element,
    which is why it is sum-only.
    """
    import mlx.core as mx
    sc.validate()
    arr = mx.array(x, dtype=mx.float32)
    out = _scan_inclusive_gpu(sc, arr, mx)
    if not sc.inclusive:
        out = out - arr
    mx.eval(out)                 # ONE sync for the whole recursion, not one per level
    return out


@dataclass
class AirGraphNode:
    """One dispatch in an AIR graph. `inputs` name either externals or other nodes.

    THE GEOMETRY COMES FROM THE OP, NOT FROM A GUESS. For many blocks this node carried
    only `n` and a scalar threadgroup, and the executor DERIVED a 1-D grid of
    blocks*threadgroup with an output of shape (n,). That is right for an elementwise
    dispatch and WRONG for every other AIR primitive -- and it failed in the worst
    possible way, which is that IT SOMETIMES LOOKED RIGHT. A row-wise norm at
    8x1024 with threadgroup 256 got 32 threadgroups where it needed 8: the extra 24 ran
    OUT OF BOUNDS on both buffers and the answer STILL MATCHED THE ORACLE to 2.6e-07.
    The same defect at 8x64 launched 2 threadgroups for 8 rows and was wrong by 2.62.
    Correct BY ACCIDENT OF SHAPE. See ACCELERATOR_GRAPH_COMPOSITION.json.

    So grid, threadgroup, output shape, parameter names and output name are all FIELDS,
    and node_for() fills them from the op's own launch() rather than re-deriving them.
    """
    name: str
    source: str
    inputs: list[str]
    n: int
    threadgroup: int = 256
    grid: tuple[int, int, int] | None = None
    threadgroup3: tuple[int, int, int] | None = None
    out_shape: tuple[int, ...] | None = None
    param_names: list[str] | None = None
    out_name: str = "out"
    header: str = ""

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if self.grid is not None and self.threadgroup3 is not None:
            return self.grid, self.threadgroup3
        blocks = (self.n + self.threadgroup - 1) // self.threadgroup
        return (blocks * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)

    def shape(self) -> tuple[int, ...]:
        return self.out_shape if self.out_shape is not None else (self.n,)


def node_for(name: str, op: Any, inputs: list[str], *, source: str,
             param_names: list[str], out_name: str = "out", header: str = "") -> AirGraphNode:
    """Build a graph node whose geometry is TAKEN FROM THE OP.

    op must expose launch() and out_shape(). Asking the op is the whole point: the
    executor cannot infer a row-wise or tiled launch from an element count, and a
    guess that is right at one shape and silently wrong at another is worse than a
    refusal.
    """
    grid, tg = op.launch()
    shape = op.out_shape()
    n = 1
    for d in shape:
        n *= d
    return AirGraphNode(name=name, source=source, inputs=inputs, n=n,
                        threadgroup=tg[0], grid=grid, threadgroup3=tg,
                        out_shape=shape, param_names=param_names,
                        out_name=out_name, header=header)


@dataclass
class AirGraph:
    """A recorded DAG of dispatches executed as ONE submission. CCL RUNTIME.graphs.

    This is the Apple-side answer to what CUDA graphs and streams are FOR, and the
    two halves of that question must not be conflated:

      SUBMISSION BATCHING -- many dispatches, one command buffer, one host sync.
        ACCELERATOR_SYNC_CORRECTION.json measured what this is worth: 1.28x to
        2.30x, and it is the ENTIRE effect that was previously misattributed to
        launch count. A graph gets this by construction.

      CONCURRENCY -- independent branches of the DAG running AT THE SAME TIME.
        That is what CUDA streams add on top of ordering, and NOTHING here
        establishes that Metal gives it. serial_depth() and width() exist so the
        question can be asked of a specific graph rather than assumed.

    A graph that batches but serializes is still worth building and is NOT the same
    capability. The receipt says which one was measured.
    """
    name: str
    nodes: list[AirGraphNode] = field(default_factory=list)
    externals: list[str] = field(default_factory=list)

    def validate(self) -> None:
        seen: set[str] = set(self.externals)
        names = [nd.name for nd in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique; SSA is what makes the "
                             "dependency edges unambiguous")
        for nd in self.nodes:
            for i in nd.inputs:
                if i not in seen:
                    raise ValueError(
                        f"node {nd.name!r} reads {i!r}, which is neither an external "
                        f"nor a node defined before it. AIR graphs are recorded in "
                        f"dependency order; a forward reference is either a cycle or "
                        f"a typo and both are refused rather than reordered.")
            seen.add(nd.name)

    def levels(self) -> list[list[str]]:
        """Nodes grouped by depth. Everything in one level is mutually independent."""
        self.validate()
        depth = {e: 0 for e in self.externals}
        out: dict[int, list[str]] = {}
        for nd in self.nodes:
            d = 1 + max(depth[i] for i in nd.inputs)
            depth[nd.name] = d
            out.setdefault(d, []).append(nd.name)
        return [out[d] for d in sorted(out)]

    def serial_depth(self) -> int:
        return len(self.levels())

    def width(self) -> int:
        return max(len(l) for l in self.levels())

    def submissions(self) -> int:
        return 1

    def sync_points(self) -> int:
        return 1


def execute_graph(g: AirGraph, arrays: dict[str, Any], *, eager: bool = False):
    """Run the whole DAG. One submission and one host sync unless eager=True.

    eager=True forces a host round trip after every node -- not a mode anyone should
    use, but the CONTROL that makes the batched number mean something. Without it
    "a graph is fast" is a claim with no baseline.
    """
    import mlx.core as mx
    g.validate()
    env = {k: mx.array(v, dtype=mx.float32) for k, v in arrays.items()}
    for nd in g.nodes:
        kern = metal_kernel(mx, name=f"air_g_{g.name}_{nd.name}",
                            input_names=nd.param_names or nd.inputs,
                            output_names=[nd.out_name], source=nd.source,
                            header=nd.header, ensure_row_contiguous=True)
        grid, tg = nd.launch()
        (env[nd.name],) = kern(inputs=[env[i] for i in nd.inputs], grid=grid,
                               threadgroup=tg, output_shapes=[nd.shape()],
                               output_dtypes=[mx.float32])
        if eager:
            mx.eval(env[nd.name])
    leaves = [nd.name for nd in g.nodes]
    mx.eval(*[env[k] for k in leaves])
    return env


@dataclass
class AirTopKSample:
    """Top-k filtering and categorical sampling: the decode tail of an LLM.

    S015 §3 names TOP-K and SAMPLING in the required corpus and no receipt has ever
    claimed or refused either, which by the ledger's own rule means NOT YET STUDIED.

    THE RANDOMNESS IS AN INPUT, NOT A SIDE EFFECT. `u` carries one uniform per row,
    drawn host-side from a seeded generator, so the kernel is a PURE FUNCTION of
    (logits, u) and can be graded by exact equality against an independent oracle.
    A kernel that generated its own randomness could only ever be graded
    statistically, and a statistical check is far weaker than an exact one -- this
    design choice is what makes the strongest grading layer available at all.

    Three outputs on purpose: the selected values and indices are returned alongside
    the sampled choice so top-k can be graded against numpy's sort WITHOUT the
    sampler in the way. Grading only the final index would confound two mechanisms.

    TIES BREAK TOWARD THE LOWER INDEX, matching a stable argsort. An unspecified
    tie-break would make the exact grading layer unusable on any logit vector with
    repeats -- which includes every masked or clamped distribution in practice.
    """
    name: str
    rows: int
    cols: int
    k: int
    temperature: float = 1.0
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("top-k sampling is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 1 <= self.k <= self.cols:
            raise ValueError(f"k={self.k} must be in 1..cols ({self.cols})")
        if self.k > 64:
            # The selection is k rounds of a full-row argmax with an O(k) membership
            # test per element, so cost grows as k^2. Refusing beyond 64 states the
            # algorithm's domain instead of quietly becoming the slowest way to sort.
            raise ValueError(f"k={self.k} exceeds 64; this is iterative extraction, "
                             f"not a sort, and beyond ~64 a different algorithm is "
                             f"the right answer rather than this one scaled up")
        if not self.temperature > 0:
            raise ValueError("temperature must be positive; temperature 0 is argmax "
                             "and is a DIFFERENT operation, not a limit this kernel takes")
        if not self.name.isidentifier():
            # The name is interpolated into the generated function's SYMBOL, so a dot
            # or a space produces a Metal compile error twenty lines deep in MLX's
            # own header with no hint that a Python string caused it. Found by naming
            # a variant after its temperature. The other AIR lowerings interpolate
            # their names the same way and are NOT guarded -- named, not silently
            # fixed, because a guard nobody has watched fire is worth little.
            raise ValueError(f"name {self.name!r} becomes a Metal symbol and must be a "
                             f"valid identifier; a '.' or ' ' fails deep inside the "
                             f"MLX header with no mention of the name")

    def barrier_scopes_emitted(self) -> list[str]:
        # one per tree-reduction step per extraction round, plus one after each write
        steps = max(1, self.threadgroup.bit_length() - 1)
        return ["THREADGROUP"] * (self.k * (steps + 1))

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "iterative argmax extraction then a serial CDF walk over k"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.rows * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)


def lower_topk_sample_to_msl(ts: AirTopKSample) -> str:
    ts.validate()
    tg = ts.threadgroup
    return f"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup float cv[{tg}];
    threadgroup uint  ci[{tg}];
    threadgroup float sel_v[{ts.k}];
    threadgroup uint  sel_i[{ts.k}];
    uint base = row * {ts.cols}u;

    for (uint t = 0u; t < {ts.k}u; ++t) {{
        float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint c = lid; c < {ts.cols}u; c += {tg}u) {{
            bool taken = false;
            for (uint s = 0u; s < t; ++s) if (sel_i[s] == c) taken = true;
            if (taken) continue;
            float v = x[base + c];
            // TIES TO THE LOWER INDEX, so the answer matches a stable argsort
            if (v > bv || (v == bv && c < bi)) {{ bv = v; bi = c; }}
        }}
        cv[lid] = bv; ci[lid] = bi;
        {barrier_msl("THREADGROUP")}
        // `half` is a METAL TYPE NAME, so the obvious loop variable does not compile
        for (uint span = {tg}u / 2u; span > 0u; span >>= 1u) {{
            if (lid < span) {{
                float ov = cv[lid + span]; uint oi = ci[lid + span];
                if (ov > cv[lid] || (ov == cv[lid] && oi < ci[lid])) {{
                    cv[lid] = ov; ci[lid] = oi;
                }}
            }}
            {barrier_msl("THREADGROUP")}
        }}
        if (lid == 0u) {{ sel_v[t] = cv[0]; sel_i[t] = ci[0]; }}
        {barrier_msl("THREADGROUP")}
    }}

    if (lid == 0u) {{
        // k is small, so the CDF walk is SERIAL IN ONE THREAD on purpose: a
        // parallel scan over k <= 64 would cost more in barriers than it saves.
        float m = -INFINITY;
        for (uint s = 0u; s < {ts.k}u; ++s) m = max(m, sel_v[s]);
        float tot = 0.0f;
        for (uint s = 0u; s < {ts.k}u; ++s) tot += exp((sel_v[s] - m) / {ts.temperature}f);
        float target = u[row] * tot;
        float acc = 0.0f;
        uint pick = sel_i[{ts.k}u - 1u];   // the last bucket absorbs any rounding slack
        for (uint s = 0u; s < {ts.k}u; ++s) {{
            acc += exp((sel_v[s] - m) / {ts.temperature}f);
            if (target < acc) {{ pick = sel_i[s]; break; }}
        }}
        choice[row] = (int)pick;
    }}
    for (uint s = lid; s < {ts.k}u; s += {tg}u) {{
        topk_val[row * {ts.k}u + s] = sel_v[s];
        topk_idx[row * {ts.k}u + s] = (int)sel_i[s];
    }}
"""


def execute_topk_sample(ts: AirTopKSample, x, u):
    import mlx.core as mx
    src = lower_topk_sample_to_msl(ts)
    kern = mx.fast.metal_kernel(
        name=f"air_topk_{ts.name}", input_names=["x", "u"],
        output_names=["choice", "topk_val", "topk_idx"], source=src,
        ensure_row_contiguous=True)
    g, tg = ts.launch()
    choice, tv, ti = kern(
        inputs=[mx.array(x, dtype=mx.float32), mx.array(u, dtype=mx.float32)],
        grid=g, threadgroup=tg,
        output_shapes=[(ts.rows,), (ts.rows, ts.k), (ts.rows, ts.k)],
        output_dtypes=[mx.int32, mx.float32, mx.int32])
    mx.eval(choice, tv, ti)
    return choice, tv, ti


def topk_sample_oracle(x, u, k: int, temperature: float = 1.0):
    """An INDEPENDENT numpy implementation, written from the operation's definition
    rather than from the kernel: a stable argsort for the top-k, and a searchsorted
    over the cumulative distribution for the choice.

    Independent in code, NOT in convention -- both sides agree that ties go to the
    lower index and that the bucket is chosen by `target < cumulative`. A convention
    error would agree with itself here, which is exactly why the distributional layer
    with its negative controls exists rather than this oracle alone.
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(-x, axis=1, kind="stable")[:, :k]
    vals = np.take_along_axis(x, order, axis=1)
    p = np.exp((vals - vals.max(axis=1, keepdims=True)) / temperature)
    cdf = np.cumsum(p, axis=1)
    target = np.asarray(u, dtype=np.float32)[:, None] * cdf[:, -1:]
    pos = (target >= cdf).sum(axis=1)
    pos = np.minimum(pos, k - 1)
    return order[np.arange(len(order)), pos], vals, order


@dataclass
class AirNorm:
    """RMSNorm and LayerNorm as ONE fused kernel per row. S015 §3 names NORMALIZATION
    in the required corpus.

    TWO VARIANCE STRATEGIES ARE KEPT ON PURPOSE, and the reason is numerical, not
    stylistic. TWO_PASS computes the mean, then sums (x - mean)^2. ONE_PASS computes
    sum(x) and sum(x^2) in the SAME loop and takes var = E[x^2] - E[x]^2, which saves a
    pass over the row and a barrier. The second form is a textbook example of
    CATASTROPHIC CANCELLATION: when the mean is large relative to the spread, E[x^2]
    and E[x]^2 are two nearly equal large numbers whose difference is the small answer,
    and f32 has no bits left to express it. ONE_PASS is retained as the DEMONSTRATION
    of that, with its error measured rather than asserted, because a strategy nobody
    has watched fail reads as a strategy that merely lost a style argument.

    RMSNorm has NO mean to subtract, so the cancellation question does not arise there
    at all -- which is worth saying, because it is the reason the modern transformer's
    choice of RMSNorm removes a numerical hazard as well as an arithmetic step.
    """
    name: str
    rows: int
    cols: int
    mode: str = "rms"                 # rms | layer
    variance: str = "two_pass"        # two_pass | one_pass  (layer only)
    eps: float = 1e-5
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("norm is f32 only here")
        if self.mode not in ("rms", "layer"):
            raise ValueError(f"mode {self.mode!r} must be 'rms' or 'layer'")
        if self.variance not in ("two_pass", "one_pass"):
            raise ValueError(f"variance {self.variance!r} must be 'two_pass' or 'one_pass'")
        if self.mode == "rms" and self.variance != "two_pass":
            raise ValueError("RMSNorm subtracts no mean, so there is no one-pass "
                             "variance to choose -- the option does not apply and "
                             "silently ignoring it would hide that")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not self.name.isidentifier():
            raise ValueError(f"name {self.name!r} becomes a Metal symbol and must be a "
                             f"valid identifier")

    def out_shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def barrier_scopes_emitted(self) -> list[str]:
        n = 1 if (self.mode == "rms" or self.variance == "one_pass") else 2
        return ["THREADGROUP"] * n

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, f"fused row {self.mode}norm, {len(self.barrier_scopes_emitted())} barriers"

    def input_names(self) -> list[str]:
        return ["x", "w"] if self.mode == "rms" else ["x", "w", "b"]

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.rows * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)


def lower_norm_to_msl(nm: AirNorm) -> str:
    nm.validate()
    lanes = nm.threadgroup // 32
    tg, cols = nm.threadgroup, nm.cols
    # the one-pass form reduces TWO quantities, so its scratch is twice as wide.
    # This was a real bug for one commit: `head + f"..." .replace(...)` binds the
    # replace to the f-string alone, so the declaration in `head` never changed.
    slots = 2 * lanes if nm.mode == "layer" else lanes
    head = f"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    threadgroup float red[{slots}];
    uint base = row * {cols}u;
"""
    if nm.mode == "rms":
        return head + f"""
    float s = 0.0f;
    for (uint c = lid; c < {cols}u; c += {tg}u) {{ float v = x[base + c]; s += v * v; }}
    float ss = simd_sum(s);
    if (lane == 0u) red[warp] = ss;
    {barrier_msl("THREADGROUP")}
    float tot = 0.0f;
    for (uint i = 0u; i < {lanes}u; ++i) tot += red[i];
    float rstd = rsqrt(tot / {float(cols)}f + {nm.eps}f);
    for (uint c = lid; c < {cols}u; c += {tg}u)
        out[base + c] = x[base + c] * rstd * w[c];
"""
    if nm.variance == "one_pass":
        return head + f"""
    // E[x^2] - E[x]^2 IN ONE PASS. Cheaper by a pass and a barrier, and WRONG when the
    // mean is large relative to the spread -- see ACCELERATOR_NORMALIZATION.json.
    float s = 0.0f, s2 = 0.0f;
    for (uint c = lid; c < {cols}u; c += {tg}u) {{ float v = x[base + c]; s += v; s2 += v * v; }}
    float ss = simd_sum(s), ss2 = simd_sum(s2);
    if (lane == 0u) {{ red[warp] = ss; red[{lanes}u + warp] = ss2; }}
    {barrier_msl("THREADGROUP")}
    float tot = 0.0f, tot2 = 0.0f;
    for (uint i = 0u; i < {lanes}u; ++i) {{ tot += red[i]; tot2 += red[{lanes}u + i]; }}
    float mean = tot / {float(cols)}f;
    float var = tot2 / {float(cols)}f - mean * mean;
    float rstd = rsqrt(var + {nm.eps}f);
    for (uint c = lid; c < {cols}u; c += {tg}u)
        out[base + c] = (x[base + c] - mean) * rstd * w[c] + b[c];
"""
    return head + f"""
    float s = 0.0f;
    for (uint c = lid; c < {cols}u; c += {tg}u) s += x[base + c];
    float ss = simd_sum(s);
    if (lane == 0u) red[warp] = ss;
    {barrier_msl("THREADGROUP")}
    float tot = 0.0f;
    for (uint i = 0u; i < {lanes}u; ++i) tot += red[i];
    float mean = tot / {float(cols)}f;

    float v = 0.0f;
    for (uint c = lid; c < {cols}u; c += {tg}u) {{ float d = x[base + c] - mean; v += d * d; }}
    float sv = simd_sum(v);
    if (lane == 0u) red[{lanes}u + warp] = sv;
    {barrier_msl("THREADGROUP")}
    float tot2 = 0.0f;
    for (uint i = 0u; i < {lanes}u; ++i) tot2 += red[{lanes}u + i];
    float rstd = rsqrt(tot2 / {float(cols)}f + {nm.eps}f);
    for (uint c = lid; c < {cols}u; c += {tg}u)
        out[base + c] = (x[base + c] - mean) * rstd * w[c] + b[c];
"""


def execute_norm(nm: AirNorm, x, w, b=None):
    import mlx.core as mx
    src = lower_norm_to_msl(nm)
    names = nm.input_names()
    kern = mx.fast.metal_kernel(name=f"air_norm_{nm.name}", input_names=names,
                                output_names=["out"], source=src,
                                ensure_row_contiguous=True)
    ins = [mx.array(x, dtype=mx.float32), mx.array(w, dtype=mx.float32)]
    if nm.mode == "layer":
        ins.append(mx.array(b, dtype=mx.float32))
    g, tg = nm.launch()
    (o,) = kern(inputs=ins, grid=g, threadgroup=tg,
                output_shapes=[(nm.rows, nm.cols)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


def norm_oracle(x, w, b, mode: str, eps: float = 1e-5):
    """float64 reference, written from the definition. RMSNorm has no mean."""
    import numpy as np
    xd = np.asarray(x, dtype=np.float64)
    if mode == "rms":
        r = 1.0 / np.sqrt((xd * xd).mean(axis=1, keepdims=True) + eps)
        return xd * r * np.asarray(w, np.float64)
    m = xd.mean(axis=1, keepdims=True)
    v = ((xd - m) ** 2).mean(axis=1, keepdims=True)
    return (xd - m) / np.sqrt(v + eps) * np.asarray(w, np.float64) + np.asarray(b, np.float64)


@dataclass
class AirBatchedMatvec:
    """B independent matvecs, each with its OWN matrix AND its own vector.

    S015 §3 names BATCHED GEMM in the required corpus. The batched case that matters
    for MoE needs a distinction nothing here had drawn:

    A DECODE-TIME EXPERT BATCH IS NOT A BATCHED OPERATION. Every routed expert
    multiplies THE SAME activation, so y_e = W_e @ x for e in 0..B is arithmetically
    identical to stacking the W_e vertically and doing ONE matvec -- verified bit for
    bit, not argued. That case needs no batched kernel and never did.

    THIS KERNEL IS FOR THE CASE THAT DOES NOT COLLAPSE: a different x per batch
    element, which is what prefill and any multi-token batch produce. Keeping the two
    apart is the point -- calling both `batched` would have hidden that the first is
    free and only the second needs anything built.
    """
    name: str
    batch: int
    rows: int
    cols: int
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("batched matvec is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if min(self.batch, self.rows, self.cols) <= 0:
            raise ValueError("batch, rows and cols must be positive")
        if not self.name.isidentifier():
            raise ValueError(f"name {self.name!r} becomes a Metal symbol and must be a "
                             f"valid identifier")

    def barrier_scopes_emitted(self) -> list[str]:
        return []          # one thread owns one output row; nothing to order

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "one thread per (batch, row); no cross-thread reads"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        n = self.batch * self.rows
        return (n, 1, 1), (min(self.threadgroup, n), 1, 1)


def lower_batched_matvec_to_msl(bm: AirBatchedMatvec) -> str:
    bm.validate()
    return f"""
    uint gid = thread_position_in_grid.x;
    if (gid >= {bm.batch * bm.rows}u) return;
    uint b = gid / {bm.rows}u;
    uint r = gid % {bm.rows}u;
    uint wbase = b * {bm.rows * bm.cols}u + r * {bm.cols}u;
    uint xbase = b * {bm.cols}u;
    float acc = 0.0f;
    for (uint c = 0u; c < {bm.cols}u; ++c) acc += w[wbase + c] * x[xbase + c];
    out[gid] = acc;
"""


def execute_batched_matvec(bm: AirBatchedMatvec, w, x):
    import mlx.core as mx
    kern = mx.fast.metal_kernel(name=f"air_bmv_{bm.name}", input_names=["w", "x"],
                                output_names=["out"],
                                source=lower_batched_matvec_to_msl(bm),
                                ensure_row_contiguous=True)
    g, tg = bm.launch()
    (o,) = kern(inputs=[mx.array(w, dtype=mx.float32), mx.array(x, dtype=mx.float32)],
                grid=g, threadgroup=tg,
                output_shapes=[(bm.batch, bm.rows)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


@dataclass
class AirSparseMatvec:
    """CSR sparse matvec. S015 §12 names SPARSE OPERATIONS as a MATH capability.

    WHAT THIS IS NOT FOR, measured rather than assumed: a SPARSE RESIDUAL on top of
    ws_rtn_q4_g64 does NOT pay on real Qwen3 expert weights -- spending the same bits
    on a finer group or one more level buys 2.5-3.9x more cosine per bit. See
    ACCELERATOR_SPARSE.json. The kernel exists because the capability is named and
    because sparsity has other uses (structured pruning, routing masks), NOT because a
    sparse residual earned a place in the representation ladder.

    ONE THREAD PER ROW, so no barrier is needed and none is emitted. That makes the
    load IMBALANCED by construction: a row with 10000 non-zeros and a row with 3 sit in
    the same threadgroup and the whole group waits for the long one. Naming that is the
    difference between a kernel with a known limitation and one with a surprise.
    """
    name: str
    rows: int
    cols: int
    nnz: int
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("sparse matvec is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if min(self.rows, self.cols) <= 0 or self.nnz < 0:
            raise ValueError("rows and cols must be positive and nnz non-negative")
        if self.nnz > self.rows * self.cols:
            raise ValueError(f"nnz {self.nnz} exceeds the {self.rows * self.cols} "
                             f"entries the matrix can hold")
        if not self.name.isidentifier():
            raise ValueError(f"name {self.name!r} becomes a Metal symbol and must be a "
                             f"valid identifier")

    def barrier_scopes_emitted(self) -> list[str]:
        return []

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "one thread per row over CSR; no cross-thread reads"

    def density(self) -> float:
        return self.nnz / float(self.rows * self.cols)

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.rows, 1, 1), (min(self.threadgroup, self.rows), 1, 1)


def lower_sparse_matvec_to_msl(sp: AirSparseMatvec) -> str:
    sp.validate()
    return f"""
    uint r = thread_position_in_grid.x;
    if (r >= {sp.rows}u) return;
    int lo = row_ptr[r];
    int hi = row_ptr[r + 1u];
    float acc = 0.0f;
    for (int p = lo; p < hi; ++p) acc += values[p] * x[col_idx[p]];
    out[r] = acc;
"""


def execute_sparse_matvec(sp: AirSparseMatvec, row_ptr, col_idx, values, x):
    import mlx.core as mx
    kern = mx.fast.metal_kernel(
        name=f"air_spmv_{sp.name}", input_names=["row_ptr", "col_idx", "values", "x"],
        output_names=["out"], source=lower_sparse_matvec_to_msl(sp),
        ensure_row_contiguous=True)
    g, tg = sp.launch()
    (o,) = kern(inputs=[mx.array(row_ptr, dtype=mx.int32), mx.array(col_idx, dtype=mx.int32),
                        mx.array(values, dtype=mx.float32), mx.array(x, dtype=mx.float32)],
                grid=g, threadgroup=tg,
                output_shapes=[(sp.rows,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


def to_csr(a):
    """Dense -> CSR, keeping structurally-zero rows as EMPTY rather than dropping them.
    An empty row is a real case (a fully pruned output) and a kernel that mishandles it
    returns garbage for that row rather than the zero the maths demands."""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    nz = a != 0
    counts = nz.sum(axis=1)
    row_ptr = np.zeros(a.shape[0] + 1, np.int32)
    np.cumsum(counts, out=row_ptr[1:])
    col_idx = np.tile(np.arange(a.shape[1], dtype=np.int32), (a.shape[0], 1))[nz]
    return row_ptr, col_idx.astype(np.int32), a[nz].astype(np.float32)


@dataclass
class AirCausalConv1d:
    """Causal DEPTHWISE conv1d -- the Mamba mixer's convolution.

    S015 §3 says CONVOLUTION WHERE USEFUL, and the qualifier is load-bearing. A census
    of the four specimens on disk answers it rather than an opinion: Falcon-H1-7B holds
    88 conv tensors (weight + bias for each of 44 layers, [3584, 1, 4] bf16), and
    Qwen3-30B-A3B, Qwen3-VL-30B-A3B and Kimi-VL-A3B hold ZERO between them. So this
    exists because ONE REAL SPECIMEN NEEDS IT, in exactly the shape that specimen uses.

    CAUSALITY IS THE CORRECTNESS, NOT A DETAIL. Left-padding by W-1 is what stops
    position t reading position t+1; get it wrong and an autoregressive model sees its
    own future, which changes no norm and shows up as a model that is mysteriously good
    at teacher forcing. A negative control pins that symmetric padding gives a
    DIFFERENT answer.
    """
    name: str
    channels: int
    length: int
    width: int = 4
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("conv1d is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if min(self.channels, self.length, self.width) <= 0:
            raise ValueError("channels, length and width must be positive")
        if self.width > self.length + self.width - 1:
            raise ValueError("width exceeds the padded input")
        if not self.name.isidentifier():
            raise ValueError(f"name {self.name!r} becomes a Metal symbol and must be a "
                             f"valid identifier")

    def barrier_scopes_emitted(self) -> list[str]:
        return []          # one thread owns one output position; nothing to order

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "one thread per (channel, position); depthwise so no cross-channel reads"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        n = self.channels * self.length
        return (n, 1, 1), (min(self.threadgroup, n), 1, 1)


def lower_causal_conv1d_to_msl(cv: AirCausalConv1d) -> str:
    cv.validate()
    return f"""
    uint gid = thread_position_in_grid.x;
    if (gid >= {cv.channels * cv.length}u) return;
    uint c = gid / {cv.length}u;
    uint t = gid % {cv.length}u;
    float acc = bias[c];
    for (uint k = 0u; k < {cv.width}u; ++k) {{
        // CAUSAL: tap k reads t - (W-1) + k, and anything before 0 is ZERO, not
        // wrapped and not clamped. Wrapping would read the END of the sequence.
        int src = (int)t - {cv.width - 1} + (int)k;
        if (src >= 0) acc += w[c * {cv.width}u + k] * x[c * {cv.length}u + (uint)src];
    }}
    out[gid] = acc;
"""


def execute_causal_conv1d(cv: AirCausalConv1d, x, w, b):
    import mlx.core as mx
    kern = mx.fast.metal_kernel(name=f"air_conv1d_{cv.name}", input_names=["x", "w", "bias"],
                                output_names=["out"],
                                source=lower_causal_conv1d_to_msl(cv),
                                ensure_row_contiguous=True)
    g, tg = cv.launch()
    (o,) = kern(inputs=[mx.array(x, dtype=mx.float32), mx.array(w, dtype=mx.float32),
                        mx.array(b, dtype=mx.float32)],
                grid=g, threadgroup=tg,
                output_shapes=[(cv.channels, cv.length)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


def causal_conv1d_oracle(x, w, b, width: int):
    """float64 reference from the definition: left-pad by width-1, then correlate."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(x.shape[0], width)
    pad = np.concatenate([np.zeros((x.shape[0], width - 1)), x], axis=1)
    out = np.empty_like(x)
    for k in range(width):
        term = pad[:, k:k + x.shape[1]] * w[:, k:k + 1]
        out = term if k == 0 else out + term
    return out + np.asarray(b, dtype=np.float64)[:, None]
