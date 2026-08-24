#!/usr/bin/env python3
"""N003 — a representation cannot be condemned until its native kernel is competent.

This is Doctor law now, not a slogan, and it was earned by measurement: the 2-bit
affine representation looked physically slower than q4 while moving FEWER bytes.
The cause was not the representation. Q4's geo_tpr64 kernel has a compile-time
group of 64, so `col / 64` is a SHIFT. The affine2 kernel took `group_size` as a
bind-time parameter, which put a NON-CONSTANT INTEGER DIVIDE on every 8-wide tile.
Runtime-divide measured 1.37x the specialized body, and specializing it moved the
unfused decode 26.84 -> 32.84 tok/s before any fusion.

So: a terrible kernel is not evidence against a representation. Before any
low-density candidate may be recorded REFUSED on speed, its kernel gets screened
for the defects that produce exactly that illusion.

What this screen is NOT: a proof of competence. Passing means "none of the known
illusion-producing patterns are present in the inner loop", which is a necessary
condition, not a sufficient one. A kernel can pass every check here and still be
slow for a reason nobody has named yet. That limitation is reported, not hidden.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SHADERS = REPO / "crates" / "hawking-core" / "shaders"
OUT = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"

# The measured anchor this law rests on.
ANCHOR = {
    "defect": "non-constant integer divide from a bind-time group_size",
    "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
    "runtime_div_vs_specialized": 1.37,
    "unfused_tok_s_before": 26.84,
    "unfused_tok_s_after": 32.84,
    "source_receipt": "receipts/headless/NOETIC_FUSED_SUBBIT.json",
    "reading": (
        "The representation was unchanged across that measurement. Only the "
        "kernel's index arithmetic changed. Condemning 2-bit on the first number "
        "would have discarded a representation that later beat the incumbent."
    ),
}

# Patterns that produce the illusion. Each names why it matters, because a check
# whose rationale is not written down gets deleted by the next person.
CHECKS: list[dict[str, Any]] = [
    {
        "id": "runtime_integer_divide",
        "why": (
            "A divide or modulo by a value the compiler cannot see is the exact "
            "defect measured at 1.37x. Division by a compile-time constant is a "
            "shift and is fine."
        ),
        # `/` or `%` whose divisor is an identifier DECLARED AS AN INTEGER in this
        # kernel. The first version of this check matched any `/` followed by a
        # name and flagged 238 of 565 kernels on tokens like `float`, `sum` and
        # `rms` -- floating-point division, which is ordinary arithmetic and not
        # the defect. A screen that flags 42% of everything is as useless as one
        # that flags nothing, so the divisor must now be provably integer.
        "pattern": re.compile(r"([/%])\s*(?!\d)([A-Za-z_][A-Za-z0-9_]*)\b"),
        "integer_divisor_only": True,
        "severity": "high",
    },
    {
        "id": "runtime_sized_loop",
        "why": "A loop bound the compiler cannot see blocks unrolling and vectorization.",
        "pattern": re.compile(r"for\s*\([^;]*;\s*[A-Za-z_][A-Za-z0-9_]*\s*<\s*(?!\d)([A-Za-z_][A-Za-z0-9_]*)"),
        "severity": "med",
    },
    {
        "id": "dynamic_branch_in_loop",
        "why": "A data-dependent branch inside the inner loop serializes a SIMD group.",
        "pattern": re.compile(r"\bif\s*\([^)]*\b(tid|lane|gid|idx|col|row)\b[^)]*\)"),
        "severity": "med",
    },
    {
        "id": "bind_time_shape_param",
        "why": (
            "A shape passed as `constant uint&` is invisible to the optimizer. "
            "This is the upstream cause of the divide, not just a co-symptom."
        ),
        "pattern": re.compile(r"constant\s+uint\s*&\s*(group_size|cols|rows|rank|dim|hidden)\b"),
        "specialization_exempt": True,
        "severity": "high",
    },
]

INNER_HINT = re.compile(r"\bfor\s*\(")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()[:12]


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def specialized_names(src: str) -> set[str]:
    """Kernels whose group is a compile-time constant, i.e. already specialized."""
    out: set[str] = set()
    for m in re.finditer(r"kernel\s+void\s+([A-Za-z0-9_]+)", src):
        name = m.group(1)
        if re.search(r"group(32|64|128)|tg\d+|tpr\d+", name):
            out.add(name)
    return out


def params_of(src: str, name: str) -> str:
    m = re.search(r"kernel\s+void\s+" + re.escape(name) + r"\s*\((.*?)\)\s*\{", src, re.S)
    return m.group(1) if m else ""


def kernel_bodies(src: str) -> list[tuple[str, str]]:
    """(name, body) for each `kernel void`, body taken to brace balance."""
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"kernel\s+void\s+([A-Za-z0-9_]+)\s*\(", src):
        name = m.group(1)
        i = src.find("{", m.end())
        if i < 0:
            continue
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((name, src[i : j + 1]))
    return out


INT_DECL = re.compile(
    r"\b(?:const\s+|constant\s+|device\s+|threadgroup\s+)*"
    r"(?:u?int|ushort|short|u?char|size_t|uint2|uint3)\s*&?\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


# `constexpr uint k = 2u;` and `const uint k = 8;` are COMPILE-TIME. The compiler
# turns a divide by these into a shift, so they are not the defect. Missing this
# made the screen flag `kSplit` in the very kernel whose specialization produced
# the 32.84 tok/s the law is built on.
CONST_DECL = re.compile(
    r"\b(?:constexpr|const)\s+(?:u?int|ushort|short|u?char|size_t)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\d"
)

# A bind-time shape used ONLY inside an equality dispatch -- `if (group_size ==
# 32u) { ... } else if (group_size == 64u) { ... }` -- is SPECIALIZED, not
# generic: inside each arm the value is a literal. That IS the measured fix, so a
# screen that calls it a defect contradicts its own anchor.
def specialized_on(body: str, name: str) -> bool:
    return bool(re.search(r"\b" + re.escape(name) + r"\s*==\s*\d+u?\b", body))


def compile_time_names(body: str, params: str) -> set[str]:
    return {m.group(1) for m in CONST_DECL.finditer(params + "\n" + body)}


def integer_names(body: str, params: str) -> set[str]:
    """Names declared with an integer type in this kernel's params or body.

    Only a divide by one of these can be the integer-divide defect. Everything
    else that looks like `a / b` is float math and is none of this screen's
    business.
    """
    return {m.group(1) for m in INT_DECL.finditer(params + "\n" + body)}


def screen_kernel(name: str, body: str, params: str = "") -> dict[str, Any]:
    ints = integer_names(body, params) - compile_time_names(body, params)
    findings = []
    has_loop = bool(INNER_HINT.search(body))
    for chk in CHECKS:
        for m in chk["pattern"].finditer(body):
            if chk.get("integer_divisor_only"):
                tok = m.group(2)
                # Must be a declared integer, and an all-caps name is a
                # compile-time constant, which the compiler turns into a shift.
                if tok not in ints or tok.isupper():
                    continue
                if specialized_on(body, tok):
                    continue  # equality-dispatched to literal arms
            else:
                tok = m.group(1) if m.groups() else m.group(0)
                if chk.get("specialization_exempt") and specialized_on(body, tok):
                    continue
            findings.append(
                {
                    "check": chk["id"],
                    "severity": chk["severity"],
                    "token": tok,
                    "why": chk["why"],
                    "snippet": body[max(0, m.start() - 40) : m.end() + 40].strip()[:140],
                }
            )
            break  # one hit per check is enough to flag
    high = [f for f in findings if f["severity"] == "high"]
    return {
        "kernel": name,
        "has_loop": has_loop,
        "findings": findings,
        "n_findings": len(findings),
        "verdict": "DEFECTIVE" if high else ("SUSPECT" if findings else "CLEAR"),
        "may_condemn_representation": not findings,
    }


def main() -> int:
    files = sorted(SHADERS.glob("*.metal"))
    per_file, all_k = [], []
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        src = strip_comments(raw)
        spec = specialized_names(src)
        ks = []
        for name, body in kernel_bodies(src):
            r = screen_kernel(name, body, params_of(src, name))
            r["file"] = f.name
            r["name_suggests_specialized"] = name in spec
            ks.append(r)
            all_k.append(r)
        per_file.append({"file": f.name, "n_kernels": len(ks), "kernels": ks})

    by_verdict: dict[str, int] = {}
    for k in all_k:
        by_verdict[k["verdict"]] = by_verdict.get(k["verdict"], 0) + 1

    defective = [k for k in all_k if k["verdict"] == "DEFECTIVE"]
    # The kernel the anchor came from: did the specialization actually land?
    affine = [k for k in all_k if "affine" in k["kernel"]]

    receipt = {
        "schema": "hawking.headless.kernel_competence.v1",
        "obligation": "N003 (S017 §5)",
        "git_head": git_head(),
        "law": (
            "A REPRESENTATION CANNOT BE CONDEMNED UNTIL ITS NATIVE KERNEL IS "
            "COMPETENT. A candidate may not be recorded REFUTED on speed until "
            "this screen passes for its kernel."
        ),
        "measured_anchor": ANCHOR,
        "checks": [{"id": c["id"], "severity": c["severity"], "why": c["why"]} for c in CHECKS],
        "counts": {
            "files": len(files),
            "kernels": len(all_k),
            "by_verdict": by_verdict,
        },
        "defective": [
            {k2: d[k2] for k2 in ("file", "kernel", "verdict", "findings")} for d in defective
        ],
        "affine2_family": [
            {k2: a[k2] for k2 in ("file", "kernel", "verdict", "n_findings")} for a in affine
        ],
        "what_this_screen_cannot_do": [
            "It is a NECESSARY condition, not a sufficient one. A kernel can pass "
            "every check and still be slow for a reason nobody has named.",
            "It is STATIC. It cannot see register pressure, occupancy, or whether "
            "loads actually coalesce at runtime.",
            "A divide by an all-caps name is treated as a compile-time constant. "
            "If such a name is in fact a runtime value, this screen misses it.",
        ],
        "per_file": per_file,
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"files {len(files)}  kernels {len(all_k)}  {by_verdict}")
    for d in defective[:12]:
        f0 = d["findings"][0]
        print(f"  DEFECTIVE {d['file']}::{d['kernel']}  {f0['check']} -> {f0['token']}")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
