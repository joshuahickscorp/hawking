"""Algorithm-level recognition of CUDA kernels. C2M-T2 (G045).

WHY THIS EXISTS AND WHY IT IS NOT A COMPILER. The T1 receipt named the blocker
exactly: AIR has GEMM, reduction, scan, softmax and attention, and NONE of them is
reachable through C2M, because the T0 kernel frontend is a pattern matcher over 4
elementwise expression forms and cannot express a loop, shared memory or a barrier.
Closing that by writing a C compiler is not the work in front of us, and the PTX
receipt already recorded what happens when you chase the most frequent refusal
instead of the binding constraint.

So this takes the other route: recognize that a kernel IS a known algorithm and
hand it to AIR's implementation of that algorithm. That is a legitimate technique --
it is what a pattern-matching optimizer does -- and it has ONE serious hazard which
governs the whole design.

THE HAZARD: RECOGNITION IS A CLAIM ABOUT SEMANTICS. Saying "this is a tiled GEMM"
asserts the kernel computes A@B. If the match is loose, a kernel that is ALMOST the
idiom -- one index transposed, one barrier missing, += changed to *= -- gets
silently replaced by a DIFFERENT computation that runs fast and returns a plausible
wrong answer. And the usual defence does not work here: grading the output against
an oracle derived from the idiom is CIRCULAR, since the oracle was chosen by the
same recognition being tested.

WHAT BREAKS THE CIRCLE is the negative direction. The recognizer is built from
NAMED REQUIRED FRAGMENTS, every one of which must match, and the tests feed it
kernels that differ from the idiom in exactly one respect and require a REFUSAL
naming the fragment that failed. A recognizer that accepts near-misses recognizes
nothing; one that rejects them is making a checkable claim.

TWO DOORS, EACH HONEST ABOUT ITSELF. c2m.translate() is the T0 door: it REFUSES
__shared__ and __syncthreads by name. This is the T2 door: it accepts ONLY exact
idioms. Neither pretends to be general, and a kernel that is neither trivially
elementwise nor an exact known idiom is refused by both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from c2m import C2MRefusal


@dataclass
class RecognizedIdiom:
    idiom: str                      # "tiled_gemm" | "block_reduce_sum"
    kernel_name: str
    tile: int | None
    threadgroup: int | None
    fragments_matched: list[str]
    mechanism: str = "IDIOM RECOGNITION, not general compilation"


def _n(s: str) -> str:
    """Whitespace-normalise so formatting is not mistaken for semantics."""
    return re.sub(r"\s+", " ", s)


# Each entry is (fragment_name, regex). ALL must match. The name is what a refusal
# reports, so "this is not a tiled GEMM" is never the whole answer.
_GEMM_FRAGMENTS = [
    ("shared tile for A", r"__shared__ float As\[(\d+)\]\[(\d+)\]"),
    ("shared tile for B", r"__shared__ float Bs\[(\d+)\]\[(\d+)\]"),
    ("row from blockIdx.y", r"row = blockIdx\.y \* \d+ \+ ty"),
    ("col from blockIdx.x", r"col = blockIdx\.x \* \d+ \+ tx"),
    ("accumulator initialised to zero", r"float acc = 0\.0f"),
    ("loop over k-tiles", r"for \(int t = 0; t < \(K \+ \d+\) / \d+; \+\+t\)"),
    ("A staged row-major into As[ty][tx]", r"As\[ty\]\[tx\] =.*A\[row \* K \+ t \* \d+ \+ tx\]"),
    ("B staged row-major into Bs[ty][tx]", r"Bs\[ty\]\[tx\] =.*B\[\(t \* \d+ \+ ty\) \* N \+ col\]"),
    ("inner product over the tile", r"for \(int k = 0; k < \d+; \+\+k\) acc \+= As\[ty\]\[k\] \* Bs\[k\]\[tx\]"),
    ("store to C row-major", r"C\[row \* N \+ col\] = acc"),
]

# THE SEED'S REAL SGEMM. Widening the tiled_gemm fragments to match it would have been
# the obvious fix and WOULD HAVE SUBSTITUTED A DIFFERENT COMPUTATION: this kernel
# computes alpha*A@B + beta*C, and AIR's matmul computes A@B. The syntactic difference
# (1-D shared tiles with a SYMBOLIC size, flattened indexing) is the visible one; the
# SEMANTIC difference is the one that matters, and it is why this is a SEPARATE idiom
# rather than a looser version of the first. See ACCELERATOR_C2M_SGEMM_IDIOM.json.
_SGEMM_FRAGMENTS = [
    ("shared tile for A, 1-D with a symbolic extent",
     r"__shared__ float As\[(\w+) \* (\w+)\]"),
    ("shared tile for B, 1-D with a symbolic extent",
     r"__shared__ float Bs\[(\w+) \* (\w+)\]"),
    ("output block row from blockIdx.x", r"cRow = blockIdx\.x"),
    ("output block column from blockIdx.y", r"cCol = blockIdx\.y"),
    ("thread column from threadIdx.x %", r"threadCol = threadIdx\.x % \w+"),
    ("thread row from threadIdx.x /", r"threadRow = threadIdx\.x / \w+"),
    ("accumulator initialised to zero", r"float tmp = 0\.0f"),
    ("loop over k-tiles", r"for \(int bkIdx = 0; bkIdx < K; bkIdx \+= \w+\)"),
    ("A staged row-major into As",
     r"As\[threadRow \* \w+ \+ threadCol\] = A\[threadRow \* K \+ threadCol\]"),
    ("B staged row-major into Bs",
     r"Bs\[threadRow \* \w+ \+ threadCol\] = B\[threadRow \* N \+ threadCol\]"),
    ("inner product over the tile",
     r"tmp \+= As\[threadRow \* \w+ \+ dotIdx\] \* Bs\[dotIdx \* \w+ \+ threadCol\]"),
    ("alpha/beta store, which is what makes this NOT a plain matmul",
     r"C\[threadRow \* N \+ threadCol\] = alpha \* tmp \+ beta \* C\[threadRow \* N \+ threadCol\]"),
]

_REDUCE_FRAGMENTS = [
    ("shared buffer", r"__shared__ float sdata\[(\d+)\]"),
    ("thread id", r"int tid = threadIdx\.x"),
    ("global index", r"int i = blockIdx\.x \* blockDim\.x \+ threadIdx\.x"),
    ("guarded load with zero identity", r"sdata\[tid\] = \(i < n\) \? in\[i\] : 0\.0f"),
    ("halving tree loop", r"for \(int s = blockDim\.x / 2; s > 0; s >>= 1\)"),
    ("partial SUM into the lower half", r"if \(tid < s\) sdata\[tid\] \+= sdata\[tid \+ s\]"),
    ("block result written by thread zero", r"if \(tid == 0\) out\[blockIdx\.x\] = sdata\[0\]"),
]

_NAME = re.compile(r"__global__\s+void\s+(\w+)\s*\(")


def _count_syncthreads(src: str) -> int:
    return len(re.findall(r"__syncthreads\s*\(\s*\)", src))


def _match_all(src: str, fragments, idiom: str) -> tuple[list[str], list[re.Match]]:
    flat = _n(src)
    names, matches = [], []
    for name, pat in fragments:
        m = re.search(pat, flat)
        if not m:
            raise C2MRefusal(
                f"not recognised as {idiom}: the required element {name!r} is absent. "
                f"Refusing rather than treating a near-miss as the idiom, because "
                f"substituting AIR's {idiom} for a kernel that is not one would "
                f"compute a DIFFERENT ANSWER and return it quickly.")
        names.append(name)
        matches.append(m)
    return names, matches


def recognize(src: str) -> RecognizedIdiom:
    """Recognise src as a known algorithm, or refuse naming what did not match."""
    nm = _NAME.search(src)
    if not nm:
        raise C2MRefusal("no __global__ kernel signature found")
    name = nm.group(1)
    flat = _n(src)

    if "__shared__ float As" in flat and "__shared__ float Bs" in flat:
        # THE ALPHA/BETA FORM IS TRIED FIRST, because it is the STRICTER claim: a
        # kernel that scales and accumulates is not a plain matmul, and letting the
        # looser idiom match it would be the exact substitution this door refuses.
        if re.search(r"alpha \* tmp \+ beta \* C", _n(src)):
            names, ms = _match_all(src, _SGEMM_FRAGMENTS, "sgemm_alpha_beta")
            for i in (0, 1):
                if ms[i].group(1) != ms[i].group(2):
                    raise C2MRefusal(
                        f"not recognised as sgemm_alpha_beta: the shared tile extent "
                        f"is {ms[i].group(1)} * {ms[i].group(2)}, and a square tile "
                        f"needs the same symbol twice.")
            if _count_syncthreads(src) != 2:
                raise C2MRefusal(
                    f"not recognised as sgemm_alpha_beta: found "
                    f"{_count_syncthreads(src)} __syncthreads() calls and the idiom "
                    f"has exactly 2. A missing one is a RACE on the tiles.")
            # THE TILE SIZE IS NOT IN THE KERNEL. It is a template parameter, so
            # recognition CANNOT supply it and does not pretend to -- execute_idiom
            # requires dims['tile'] and refuses without it.
            return RecognizedIdiom("sgemm_alpha_beta", name, None, None, names)
        names, ms = _match_all(src, _GEMM_FRAGMENTS, "tiled_gemm")
        ta, tb = (int(ms[0].group(1)), int(ms[0].group(2)))
        tc, td = (int(ms[1].group(1)), int(ms[1].group(2)))
        if not ta == tb == tc == td:
            raise C2MRefusal(
                f"not recognised as tiled_gemm: the shared tiles are {ta}x{tb} and "
                f"{tc}x{td}. AIR's tiled matmul assumes ONE square tile size; a "
                f"rectangular or mismatched tiling is a different kernel.")
        if _count_syncthreads(src) != 2:
            raise C2MRefusal(
                f"not recognised as tiled_gemm: found {_count_syncthreads(src)} "
                f"__syncthreads() calls, and the idiom has exactly 2 (after staging, "
                f"after consuming). A different barrier count is a different "
                f"synchronization structure, and a missing one is a RACE.")
        return RecognizedIdiom("tiled_gemm", name, ta, None, names)

    if "__shared__ float sdata" in flat:
        names, ms = _match_all(src, _REDUCE_FRAGMENTS, "block_reduce_sum")
        tg = int(ms[0].group(1))
        if _count_syncthreads(src) != 2:
            raise C2MRefusal(
                f"not recognised as block_reduce_sum: found "
                f"{_count_syncthreads(src)} __syncthreads() calls, and the idiom has "
                f"exactly 2 (after the load, inside the tree loop). A missing one is "
                f"a RACE on sdata.")
        if tg % 32 or not 32 <= tg <= 1024:
            raise C2MRefusal(
                f"not recognised as block_reduce_sum: shared buffer of {tg} does not "
                f"correspond to a legal Metal threadgroup (multiple of 32, at most 1024)")
        return RecognizedIdiom("block_reduce_sum", name, None, tg, names)

    raise C2MRefusal(
        "no known idiom: the kernel declares neither the As/Bs tile pair of a tiled "
        "GEMM nor the sdata buffer of a block reduction. The T2 door recognises "
        "exactly two algorithms and this is neither; the T0 door (c2m.translate) "
        "handles trivially elementwise kernels.")


def execute_idiom(rec: RecognizedIdiom, operands: dict[str, Any], *, dims: dict[str, int]):
    """Run the recognised algorithm using AIR's implementation of it."""
    import air
    if rec.idiom == "tiled_gemm":
        mm = air.AirMatmul(rec.kernel_name, dims["M"], dims["K"], dims["N"],
                           tile=rec.tile, strategy="tiled")
        return air.execute_matmul(mm, operands["A"], operands["B"])
    if rec.idiom == "sgemm_alpha_beta":
        if "tile" not in dims:
            raise C2MRefusal(
                "sgemm_alpha_beta declares its tile as a TEMPLATE PARAMETER, so the "
                "kernel source does not contain it; pass dims['tile'] from the "
                "instantiation rather than letting recognition guess one")
        mm = air.AirMatmul(rec.kernel_name, dims["M"], dims["K"], dims["N"],
                           tile=dims["tile"], strategy="tiled")
        prod = air.execute_matmul(mm, operands["A"], operands["B"])
        # alpha*(A@B) + beta*C. THE SCALING IS PART OF THE ANSWER, not decoration:
        # handing the bare product back would be the wrong computation whenever
        # alpha != 1 or beta != 0.
        import numpy as _np
        return (operands["alpha"] * _np.asarray(prod)
                + operands["beta"] * _np.asarray(operands["C"]))
    if rec.idiom == "block_reduce_sum":
        rd = air.AirReduce(rec.kernel_name, dims["n"], "sum", rec.threadgroup)
        return air.execute_reduce(rd, operands["in"])
    raise C2MRefusal(f"no executor for idiom {rec.idiom!r}")


def conformance_t2(results: list[dict[str, Any]], near_misses: list[dict[str, Any]]
                   ) -> dict[str, Any]:
    """T2 is claimed only if idioms executed AND every near-miss was refused.

    The second half is not decoration. A recognizer that accepts a transposed index
    or a missing barrier is not recognising anything, so acceptance without the
    rejections would be a tier claimed on a coin flip.
    """
    ok = [r for r in results if r.get("matches_oracle")]
    all_refused = bool(near_misses) and all(n.get("refused") for n in near_misses)
    claimed = bool(ok) and all_refused
    return {
        "tier_claimed": "C2M-T2" if claimed else "C2M-T1",
        "tier_definition": "T2 = a math/ML kernel corpus reachable through C2M",
        "mechanism": "IDIOM RECOGNITION, not general compilation. C2M does not "
                     "compile a loop, shared memory or a barrier; it recognises two "
                     "exact algorithms and hands them to AIR's implementation.",
        "idioms_recognised": sorted({r["idiom"] for r in results}),
        "kernels_executed": len(results), "kernels_matching_oracle": len(ok),
        "near_misses_presented": len(near_misses),
        "near_misses_refused": sum(1 for n in near_misses if n.get("refused")),
        "why_the_near_misses_are_the_evidence":
            "Grading a recognised kernel against an oracle derived from the idiom is "
            "CIRCULAR -- the same recognition chose the oracle. The non-circular "
            "evidence is that kernels differing from the idiom in exactly one respect "
            "are REFUSED with that respect named.",
        "higher_tiers": {
            "C2M-T3": "NOT CLAIMED: no real open CUDA project has been run, and "
                      "recognition of 2 idioms will not carry one",
            "C2M-T4": "NOT CLAIMED: no AI workload",
            "C2M-T5": "NOT CLAIMED: nothing is production-supported",
        },
        "oracle": "numpy on CPU",
        "is_a_cuda_differential": False,
    }
