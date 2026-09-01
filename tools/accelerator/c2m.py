"""C2M — the CUDA to Metal frontend. FRONT C (G045, steer S015).

This lowers a RESTRICTED subset of CUDA C into AIR, which then executes on the
Apple GPU. The subset is small and its boundary is enforced: anything outside it is
REFUSED with the exact construct named, because the steer requires unsupported
paths to fail explicitly rather than silently producing something plausible.

CONFORMANCE HONESTY. This is C2M-T0 -- trivial elementwise kernels -- and it may
not be described as "CUDA support" without that tier. It is also NOT a CUDA
differential: no NVIDIA hardware exists on this machine, so a translated kernel is
graded against a numpy oracle running on the CPU, not against CUDA. That is a
weaker claim and it is labelled as one everywhere it appears. P2 (a real CUDA
differential corpus) stays open and cannot be closed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from air import AirOp, AirProgram, AirTensor

# Constructs we can see but deliberately do not handle. Naming them is the point:
# a silent mistranslation is worse than a refusal.
UNSUPPORTED = {
    "__shared__": "shared memory",
    "__syncthreads": "block synchronization",
    "atomicAdd": "atomics",
    "atomicCAS": "atomics",
    "for": "loops",
    "while": "loops",
    "__ldg": "read-only cache intrinsics",
    "cooperative_groups": "cooperative groups",
    "printf": "device printf",
    # NOT A FRONTEND GAP AND NOT A FEATURE THIS SUBSET DECLINED TO BUILD.
    # Metal REFUSES the type outright -- the compiler says
    # "'double' is not supported in Metal" -- so this is a HARDWARE refusal in
    # the same class as the FFT one, and the workaround has a MEASURED price:
    # a double-double chain reaches ~42.6 effective mantissa bits against f32's
    # 18.6 on the same 4096-term dot product and f64's 53
    # (receipts/headless/ACCELERATOR_FP64_IS_A_HARDWARE_REFUSAL.json).
    "double": "fp64 (Metal has no double; see the fp64 receipt for the cost)",
    "__syncwarp": "warp synchronization",
    "__shfl": "warp shuffle",
}

# The T0 expression forms. A pattern frontend, not a general C expression parser --
# stated plainly so nobody mistakes its reach.
#
# THE INDEX VARIABLE IS DISCOVERED, NOT ASSUMED. These read `i` for display only;
# patterns_for() rebuilds them around whatever identifier the kernel actually binds to
# the global thread index. The hardcoded `i` REFUSED TWO KERNELS IN THE PINNED SEED
# THAT COMPUTE EXACTLY a[index] + b[index] -- a supported operation rejected over a
# VARIABLE NAME. Measured in ACCELERATOR_C2M_CORPUS_CENSUS.json.
PATTERNS: list[tuple[str, str, int]] = [
    (r"^(\w+)\[i\]\s*\+\s*(\w+)\[i\]$", "add", 2),
    # A common CUDA sample spelling of the same elementwise add.  The literal is
    # an identity in f32, so accepting it does not widen the computation; it only
    # removes a source-level normalization obstacle.
    (r"^(\w+)\[i\]\s*\+\s*(\w+)\[i\]\s*\+\s*0(?:\.0)?f?$", "add", 2),
    (r"^(\w+)\[i\]\s*\*\s*(\w+)\[i\]$", "mul", 2),
    (r"^(\w+)\[i\]\s*-\s*(\w+)\[i\]$", "sub", 2),
    (r"^fmaxf\(\s*(\w+)\[i\]\s*,\s*0(?:\.0)?f?\s*\)$", "relu", 1),
    (r"^fmax\(\s*(\w+)\[i\]\s*,\s*0(?:\.0)?f?\s*\)$", "relu", 1),
]


def patterns_for(idx: str) -> list[tuple[str, str, int]]:
    """The same forms, written around the kernel's OWN index identifier."""
    q = re.escape(idx)
    return [(pat.replace(r"\[i\]", r"\[" + q + r"\]"), op, ar)
            for pat, op, ar in PATTERNS]


# THE GLOBAL THREAD INDEX, IN EVERY SPELLING THAT MEANS THE SAME THING.
#
# The addend order was already accepted; the FACTOR order was not, so
# `blockDim.x * blockIdx.x + threadIdx.x` -- integer multiplication written the other
# way round, identical semantics -- was refused as "no recognised global thread
# index". Two kernels in the pinned seed spell it that way and both compute exactly
# c[id] = a[id] + b[id], the one elementwise computation this frontend already
# handles. Half of the commutation being handled is what hid the other half.
#
# THIS IS THE SECOND SUPPORTED COMPUTATION REJECTED OVER SPELLING: the corpus census
# already found the index VARIABLE NAME hardcoded to `i`. A pattern matcher's misses
# are dominated by spelling, not by expressiveness -- measured in
# ACCELERATOR_C2M_CORPUS_DENOMINATOR.json.
#
# The widening is exactly commutativity and no wider. `blockIdx.x + blockDim.x *
# threadIdx.x` is a DIFFERENT index and stays refused; a test pins that near miss.
_PRODUCT = r"(?:blockIdx\.x\s*\*\s*blockDim\.x|blockDim\.x\s*\*\s*blockIdx\.x)"
_GRID_IDX_SRC = (rf"(?:{_PRODUCT}\s*\+\s*threadIdx\.x"
                 rf"|threadIdx\.x\s*\+\s*{_PRODUCT})")

INDEX_BINDING = re.compile(
    r"\b(?:const\s+)?(?:unsigned\s+)?(?:int|long|size_t|unsigned)\s+(\w+)\s*=\s*[^;]*"
    + _GRID_IDX_SRC)

GRID_IDX = re.compile(_GRID_IDX_SRC)
_SCALAR = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[fF]?"


class C2MRefusal(Exception):
    """Raised when a kernel falls outside the supported subset. Carries the exact
    construct, never a vague 'unsupported kernel'."""


@dataclass
class TranslatedKernel:
    name: str
    params: list[tuple[str, str, bool]]     # (ctype, name, is_pointer)
    inputs: list[str]
    output: str
    program: AirProgram
    tier: str = "C2M-T0"


def _check_unsupported(src: str) -> None:
    for token, human in UNSUPPORTED.items():
        if re.search(r"\b" + re.escape(token), src):
            raise C2MRefusal(
                f"{human} is not in the C2M-T0 subset (found {token!r}); "
                f"refusing rather than mistranslating")


def translate(src: str, *, elements: int) -> TranslatedKernel:
    _check_unsupported(src)

    m = re.search(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{(.*)\}", src, re.S)
    if not m:
        raise C2MRefusal("no __global__ kernel with a body was found")
    name, params_src, body = m.group(1), m.group(2), m.group(3)

    # AN EMPTY BODY IS REFUSED FIRST, AND THE ORDER IS THE POINT. The pinned seed
    # holds nine `extern "C" __global__ void f(const float*, int64_t, ...) {}` stubs
    # that exist to check name mangling and ABI, not to compute. Their parameters are
    # UNNAMED, so parameter parsing reached them first and refused with "cannot parse
    # parameter 'const float*'" -- a true statement about a kernel whose real and
    # permanent disqualification is that IT COMPUTES NOTHING. This program has already
    # sealed that a refusal naming the wrong cause teaches the wrong lesson (the
    # barrier-scopes receipt); here it invited the reader to go fix a parameter parser
    # for a kernel no frontend can ever translate. Measured in
    # ACCELERATOR_C2M_CORPUS_DENOMINATOR.json.
    if not strip_comments(body).strip():
        raise C2MRefusal(f"kernel {name!r} has an EMPTY BODY and computes nothing; "
                         f"it is a signature/ABI probe, not a translatable kernel")

    params: list[tuple[str, str, bool]] = []
    for raw in [p.strip() for p in params_src.split(",") if p.strip()]:
        pm = re.match(
            r"(?:const\s+)?(\w+)\s*(\*?)\s*"
            r"(?:(?:__restrict__|restrict)\s*)?(\w+)$", raw
        )
        if not pm:
            raise C2MRefusal(f"cannot parse parameter {raw!r}")
        ctype, star, pname = pm.group(1), pm.group(2), pm.group(3)
        if ctype not in ("float", "int"):
            raise C2MRefusal(f"parameter type {ctype!r} is outside the subset")
        params.append((ctype, pname, star == "*"))

    if not GRID_IDX.search(body):
        raise C2MRefusal("no recognised global thread index "
                         "(expected blockIdx.x * blockDim.x + threadIdx.x)")
    ib = INDEX_BINDING.search(body)
    if not ib:
        raise C2MRefusal("the global thread index is computed but not bound to a named "
                         "variable; the T0 subset indexes through that name")
    idx = ib.group(1)
    q = re.escape(idx)

    # the single guarded assignment this tier supports
    am = re.search(r"(\w+)\s*\[\s*" + q + r"\s*\]\s*=\s*([^;]+);", body)
    if not am:
        raise C2MRefusal(f"no single assignment indexed by {idx!r} found in the body")
    out_name, expr = am.group(1), am.group(2).strip()

    if len(re.findall(r"\w+\s*\[\s*" + q + r"\s*\]\s*=(?!=)", body)) > 1:
        raise C2MRefusal("more than one store; the T0 subset is a single assignment")

    for pat, op, arity in patterns_for(idx):
        pm = re.match(pat, expr)
        if not pm:
            continue
        srcs = list(pm.groups())[:arity]
        ptr_names = [p[1] for p in params if p[2]]
        for s in srcs:
            if s not in ptr_names:
                raise C2MRefusal(f"{s!r} is not a pointer parameter")
        ins = [AirTensor(s, (elements,), "f32") for s in srcs]
        prog = AirProgram(name, ins, [AirOp(op, tuple(srcs), "out_")], "out_")
        return TranslatedKernel(name, params, srcs, out_name, prog)

    # A scalar multiply is a common CUDA elementwise idiom (including ``x * -1``).
    # It lowers to AIR's specialization-backed scale op: the scalar is a compile-time
    # value, not a fabricated buffer input.  Only numeric literals are accepted here;
    # a runtime scalar remains outside T0 until scalar-argument plumbing exists.
    sm = re.match(
        rf"^(?:(?P<left>{_SCALAR})\s*\*\s*(?P<left_src>\w+)\s*\[\s*{q}\s*\]"
        rf"|(?P<right_src>\w+)\s*\[\s*{q}\s*\]\s*\*\s*(?P<right>{_SCALAR}))$",
        expr,
    )
    if sm:
        scalar = sm.group("left") or sm.group("right")
        source = sm.group("left_src") or sm.group("right_src")
        ptr_names = [p[1] for p in params if p[2]]
        if source not in ptr_names:
            raise C2MRefusal(f"{source!r} is not a pointer parameter")
        alpha = float(scalar.rstrip("fF"))
        ins = [AirTensor(source, (elements,), "f32")]
        prog = AirProgram(
            name, ins, [AirOp("scale", (source,), "out_")], "out_",
            specialization={"ALPHA": alpha},
        )
        return TranslatedKernel(name, params, [source], out_name, prog)

    raise C2MRefusal(f"expression {expr!r} is outside the C2M-T0 pattern set "
                     f"(supported: a+b, a+b+0.0f, a*b, a-b, scalar*a, fmaxf(a,0))")


def conformance(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A tier is claimed only from what actually executed and matched."""
    passed = [r for r in results if r.get("matches_oracle")]
    return {
        "tier_claimed": "C2M-T0" if passed else "NONE",
        "tier_definition": "T0 = trivial elementwise kernels",
        "kernels_translated": len(results),
        "kernels_matching_oracle": len(passed),
        "higher_tiers": {
            "C2M-T1": "NOT CLAIMED: no runtime, memory API or stream semantics exist",
            "C2M-T2": "NOT CLAIMED: no math/ML kernel corpus (no GEMM, no reduction)",
            "C2M-T3": "NOT CLAIMED: a bounded source slice and one translated kernel "
                       "are present, but no project-level CUDA runtime has been run",
            "C2M-T4": "NOT CLAIMED: no AI workload",
            "C2M-T5": "NOT CLAIMED: nothing is production-supported",
        },
        "oracle": "numpy on CPU",
        "is_a_cuda_differential": False,
        "why_not": "no NVIDIA hardware is present, so nothing was compared against "
                   "CUDA itself; P2 remains open",
    }


# --------------------------------------------------------------------------
# PTX front end. The steer names PTX as a C2M input alongside CUDA source, and
# the pinned seed ships a real PTX corpus, so this parses THAT rather than PTX I
# wrote to suit myself. Same discipline as the CUDA path: a tiny supported subset
# and an explicit refusal naming the instruction for everything else.
# --------------------------------------------------------------------------

PTX_ARITH = {"add.f32": "add", "mul.f32": "mul", "sub.f32": "sub"}
PTX_SPECIAL = {"%ctaid.x": "ctaid", "%ntid.x": "ntid", "%tid.x": "tid"}


class PtxRefusal(C2MRefusal):
    """A PTX construct outside the supported subset."""


def _strip(line: str) -> str:
    return re.sub(r"//.*$", "", line).strip()


def translate_ptx(src: str, *, elements: int) -> TranslatedKernel:
    lines = [_strip(l) for l in src.splitlines()]
    lines = [l for l in lines if l and not l.startswith("//")]

    m = re.search(r"\.visible\s+\.entry\s+(\w+)\s*\(", src)
    if not m:
        raise PtxRefusal("no .visible .entry found")
    name = m.group(1)

    params = re.findall(r"\.param\s+\.u64\s+(\w+)", src)
    if not params:
        raise PtxRefusal("no .param .u64 pointer parameters; only pointer kernels "
                         "are in the subset")

    regs: dict[str, Any] = {}          # register -> symbolic value
    loads: dict[str, int] = {}         # value name -> param index
    store: tuple[int, str] | None = None
    counter = 0

    body = src[src.index("{") + 1:src.rindex("}")]
    for raw in [_strip(l) for l in body.splitlines()]:
        if not raw or raw.startswith(".reg") or raw == "ret;":
            continue
        line = raw.rstrip(";")
        op = line.split()[0]

        if op.startswith("ld.param.u64"):
            d, s = re.match(r"ld\.param\.u64\s+(%\w+),\s*\[(\w+)\]", line).groups()
            regs[d] = ("ptr", params.index(s))
        elif op.startswith("mov.u32"):
            d, s = re.match(r"mov\.u32\s+(%\w+),\s*(\S+)", line).groups()
            if s not in PTX_SPECIAL:
                raise PtxRefusal(f"mov.u32 from {s!r} is outside the subset; only "
                                 f"{sorted(PTX_SPECIAL)} are recognised")
            regs[d] = ("special", PTX_SPECIAL[s])
        elif op.startswith("cvta.to.global.u64") or op.startswith("cvta.global.u64"):
            # Address-space cast. Generic -> global is a NO-OP for our purposes
            # because AIR has no address spaces; the pointer identity carries through.
            # This single instruction blocked 43 of 291 corpus files.
            d, s = re.match(r"cvta\.[\w.]+\s+(%\w+),\s*(%\w+)", line).groups()
            if s not in regs:
                raise PtxRefusal(f"cvta from untracked register {s!r}")
            regs[d] = regs[s]
        elif op in ("mul.lo.u32", "add.u32", "cvt.u64.u32", "shl.b64", "add.u64",
                    "mad.lo.u32", "mul.wide.u32", "mul.wide.s32", "cvt.s64.s32"):
            # index arithmetic. The subset assumes the canonical
            # gid = ctaid.x*ntid.x + tid.x then a scaled pointer bump, so these are
            # tracked structurally rather than evaluated.
            parts = re.match(r"\S+\s+(%\w+),\s*(.+)", line)
            if not parts:
                raise PtxRefusal(f"cannot parse index arithmetic {line!r}")
            d = parts.group(1)
            srcs = [s.strip() for s in parts.group(2).split(",")]
            base = next((regs[s] for s in srcs
                         if s in regs and regs[s][0] == "ptr"), None)
            regs[d] = base if base else ("index", None)
        elif op.startswith("ld.global.f32"):
            d, s = re.match(r"ld\.global\.f32\s+(%\w+),\s*\[(%\w+)\]", line).groups()
            if s not in regs or regs[s][0] != "ptr":
                raise PtxRefusal(f"ld.global.f32 from {s!r}, which is not a tracked "
                                 f"parameter pointer")
            vname = f"v{counter}"; counter += 1
            regs[d] = ("value", vname)
            loads[vname] = regs[s][1]
        elif op in PTX_ARITH:
            d, a, b = re.match(r"\S+\s+(%\w+),\s*(%\w+),\s*(%\w+)", line).groups()
            for r in (a, b):
                if r not in regs or regs[r][0] != "value":
                    raise PtxRefusal(f"{op} operand {r!r} is not a loaded value")
            vname = f"v{counter}"; counter += 1
            regs[d] = ("value", vname)
            regs[vname] = ("expr", PTX_ARITH[op], regs[a][1], regs[b][1])
        elif op.startswith("st.global.f32"):
            p, s = re.match(r"st\.global\.f32\s+\[(%\w+)\],\s*(%\w+)", line).groups()
            if p not in regs or regs[p][0] != "ptr":
                raise PtxRefusal(f"st.global.f32 to {p!r}, not a tracked parameter")
            if s not in regs or regs[s][0] != "value":
                raise PtxRefusal(f"st.global.f32 of {s!r}, not a computed value")
            store = (regs[p][1], regs[s][1])
        else:
            raise PtxRefusal(f"PTX instruction {op!r} is outside the C2M-T0 subset")

    if store is None:
        raise PtxRefusal("no st.global.f32; the subset requires exactly one store")

    out_param, out_val = store
    expr = regs.get(out_val)
    if not (isinstance(expr, tuple) and expr[0] == "expr"):
        raise PtxRefusal("the stored value is not a supported arithmetic expression")
    _, kind, lhs, rhs = expr
    if lhs not in loads or rhs not in loads:
        raise PtxRefusal("an operand was not loaded from a parameter pointer")

    in_names = [f"p{loads[lhs]}", f"p{loads[rhs]}"]
    ins = [AirTensor(n, (elements,), "f32") for n in in_names]
    prog = AirProgram(name, ins, [AirOp(kind, tuple(in_names), "out_")], "out_")
    return TranslatedKernel(name, [("float", p, True) for p in params],
                            in_names, f"p{out_param}", prog)


# ---------------------------------------------------------------------------
# CORPUS CENSUS -- what the denominator actually contains.
#
# Every C2M receipt has quoted coverage as a fraction of "35 kernels in the pinned
# seed". Measuring that denominator found two things that make the fraction mean
# something different from what it reads as. Both are in
# ACCELERATOR_C2M_CORPUS_DENOMINATOR.json.
#
#   1. NINE of the 36 __global__ kernels have EMPTY BODIES. No frontend can ever
#      translate a kernel that computes nothing, so nine denominator slots are
#      unreachable by construction and any coverage fraction over 35 is understated.
#
#   2. Of the 27 kernels that DO compute, SEVEN are elementwise maps and ALL SEVEN
#      COMPUTE a[idx] + b[idx]. The corpus holds ONE distinct elementwise
#      computation, written seven times with different index names, guard styles
#      and dtypes. A coverage number over this corpus is measuring repetition.
# ---------------------------------------------------------------------------

# Cooperative constructs disqualify a kernel from being an elementwise map. This is
# a REGEX over the body and is therefore an UPPER BOUND: the seed's softmax_warp
# calls warp_reduce_max(), whose __shfl_xor_sync lives one function call away, so
# this classifies it as elementwise-shaped and it is not. Named rather than fixed --
# resolving it needs a call graph, which is the C compiler this frontend is not.
_COOPERATIVE = ("__shared__", "__syncthreads", "__syncwarp", "__shfl",
                "__ballot", "cg::", "atomic")


def strip_comments(src: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)


def split_kernels(src: str) -> list[tuple[str, str, str]]:
    """Every __global__ kernel in a translation unit as (name, params, body).

    Brace-matched rather than regex-terminated, because the `\\{(.*)\\}` greedy form
    translate() uses swallows every kernel after the first when a file holds several.
    """
    out: list[tuple[str, str, str]] = []
    head = re.compile(r"__global__\s+[\w\s\*]*?\s(\w+)\s*\(([^)]*)\)\s*\{", re.S)
    for m in head.finditer(src):
        depth, end = 0, None
        for j in range(m.end() - 1, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            raise C2MRefusal(f"kernel {m.group(1)!r} has an unterminated body")
        out.append((m.group(1), m.group(2), src[m.end():end]))
    return out


class EmptyCorpus(RuntimeError):
    """The corpus was not there. Distinct from `nothing translated`."""


def census(sources: dict[str, str], *, elements: int = 16) -> dict[str, Any]:
    """The honest denominator for a CUDA corpus. `sources` maps path -> file text.

    RAISES ON AN EMPTY CORPUS, and the message points at the CORPUS rather than the
    frontend. census({}) used to return kernels 0, translated 0 and an empty refusal
    histogram -- which reads EXACTLY like a coverage measurement of a frontend that
    translates nothing, and this program has sealed that shape five times
    (a gate blind to magnitude, a predicate accepting an all-zeros pack, a tie-break
    mutation on dead code, a coverage curve counting absent blockers, and a shape
    sweep reporting 0 wrong of 0 cases). kernel_forge.shape_sweep already raises for
    the same reason and names the shape FILTER; this names the CORPUS.

    IT MATTERS RIGHT NOW AND NOT HYPOTHETICALLY: the pinned Lulzx/cuda-metal clone
    every C2M coverage number was measured over lived in a session scratchpad and
    has been reaped, so calling census over a corpus reader that finds nothing is a
    live possibility rather than a defensive one."""
    if not sources:
        raise EmptyCorpus(
            "census called with ZERO SOURCE FILES. This is not a frontend that "
            "translates nothing -- it is a corpus that is not there. The pinned "
            "corpus is Lulzx/cuda-metal @ 19a702303dd29a1b25d668c6a6eca51302a2323c "
            "and it is NOT vendored into this repo; if the clone has been reaped, "
            "re-clone it before reading any coverage number as a measurement.")
    kernels, empty, ew, translated = [], [], [], []
    refusals: dict[str, int] = {}
    computations: dict[str, list[str]] = {}
    for path, text in sorted(sources.items()):
        for name, params, body in split_kernels(text):
            kernels.append((path, name))
            clean = strip_comments(body)
            if not clean.strip():
                empty.append(name)
                continue
            cooperative = [c for c in _COOPERATIVE if c in clean]
            stores = re.findall(r"(\w+)\s*\[\s*[^\]]+?\s*\]\s*=(?!=)\s*([^;]+);", clean)
            if not cooperative and not re.search(r"\b(for|while)\s*\(", clean) \
                    and len(stores) == 1:
                ew.append(name)
                key = re.sub(r"\s+", "", re.sub(r"\[[^\]]*\]", "[_]", stores[0][1]))
                computations.setdefault(key, []).append(name)
            try:
                translate(f"__global__ void {name}({params}){{{body}}}",
                          elements=elements)
                translated.append(name)
            except C2MRefusal as e:
                refusals[str(e)[:60]] = refusals.get(str(e)[:60], 0) + 1
    if not kernels:
        raise EmptyCorpus(
            f"census read {len(sources)} source file(s) and found ZERO __global__ "
            f"kernels in them. A coverage fraction over zero kernels is 0 of 0, "
            f"which reads identically to 0 of many. Files read: "
            f"{sorted(sources)[:5]}. Check the corpus path before treating any "
            f"number here as a measurement.")
    return {
        "kernels": len(kernels),
        "empty_body": len(empty),
        "empty_body_names": empty,
        "computing_kernels": len(kernels) - len(empty),
        "elementwise_shaped_upper_bound": len(ew),
        "elementwise_shaped_is_an_upper_bound_because":
            "the cooperative check is a regex over the body and is blind to a "
            "shuffle or barrier behind a device-function call",
        "distinct_elementwise_computations": len(computations),
        "elementwise_computations": {k: sorted(v) for k, v in computations.items()},
        "translated": len(translated),
        "refusal_histogram": dict(sorted(refusals.items(), key=lambda kv: -kv[1])),
    }
