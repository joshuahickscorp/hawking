"""STATIC_KERNEL_PREFLIGHT — host/shader ABI checker with zero GPU.

Reads the real .metal sources and their Rust hosts and checks the contract
between them. Everything this tool emits is STATIC_ONLY. Static correctness
does NOT prove speed and does NOT substitute for a protected measurement.
This tool exists to stop wasting protected GPU windows on defects that were
detectable from source, not to replace those windows.

    python3 tools/future/static_kernel_verify.py --scan
    python3 tools/future/static_kernel_verify.py --build
    python3 -m pytest tools/future/test_static_kernel_verify.py -q
    python3 tools/future/static_kernel_verify_parity.py --rust-bin PATH

scan() prefers crates/hawking-core's hawking-static-kernel-verify binary when
present and well-formed. HAWKING_SKV_FORCE_PYTHON=1 or a missing binary keeps
the Python analyzer as the default; HCLI must not crash on an absent artifact.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tools.future._common import write_receipt, load_json, REPO, git

import subprocess

RECEIPT = "STATIC_KERNEL_PREFLIGHT.json"
SCHEMA = "hawking.future.static_kernel_verify.v1"
VERSION = 1

# Architectural Apple Silicon compute ceilings. These are device limits from
# the Metal shading language / Apple GPU ISA, not measurements. A pipeline's
# actual maxTotalThreadsPerThreadgroup is a runtime PSO property this checker
# cannot see; 1024 is the hard ceiling we can apply statically.
APPLE_MAX_THREADS_PER_THREADGROUP = 1024
APPLE_SIMDGROUP_WIDTH = 32

SHADER_DIR = Path("crates/hawking-core/shaders")
HOST_ROOT = Path("crates/hawking-core")

# Kind tags for a host bind or a shader parameter.
KIND_DEVICE = "device"
KIND_CONSTANT_U32 = "constant_u32"
KIND_CONSTANT_F32 = "constant_f32"
KIND_CONSTANT_STRUCT = "constant_struct"
KIND_CONSTANT_BYTES = "constant_bytes"
KIND_THREADGROUP = "threadgroup"
KIND_UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Small source utilities (line-preserving)
# ---------------------------------------------------------------------------


def _line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, max(0, pos)) + 1


def _loc(path: str, line: int) -> str:
    return f"{path}:{line}"


def strip_comments_preserve_lines(src: str) -> str:
    """Replace // and /* */ comments with spaces, keep newlines (and so line numbers)."""
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            i += 2
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            out.extend("  ")
            while i < n:
                if src[i] == "*" and i + 1 < n and src[i + 1] == "/":
                    out.extend("  ")
                    i += 2
                    break
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            continue
        if c == '"':
            out.append(c)
            i += 1
            while i < n:
                out.append(src[i])
                if src[i] == "\\" and i + 1 < n:
                    out.append(src[i + 1])
                    i += 2
                    continue
                if src[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _match_paren(src: str, open_pos: int) -> int:
    """Return index of matching closer for src[open_pos] ('(' or '{')."""
    opener = src[open_pos]
    closer = ")" if opener == "(" else "}" if opener == "{" else "]"
    depth = 0
    i = open_pos
    n = len(src)
    in_str = False
    str_ch = ""
    while i < n:
        c = src[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == str_ch:
                in_str = False
            i += 1
            continue
        if c in ('"', "'"):
            in_str = True
            str_ch = c
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i += 2
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_level(src: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = depth_brack = depth_brace = 0
    in_str = False
    str_ch = ""
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if in_str:
            buf.append(c)
            if c == "\\" and i + 1 < n:
                buf.append(src[i + 1])
                i += 2
                continue
            if c == str_ch:
                in_str = False
            i += 1
            continue
        if c in ('"', "'"):
            in_str = True
            str_ch = c
            buf.append(c)
            i += 1
            continue
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
        elif c == "[":
            depth_brack += 1
        elif c == "]":
            depth_brack -= 1
        elif c == "{":
            depth_brace += 1
        elif c == "}":
            depth_brace -= 1
        if c == sep and depth_paren == depth_brack == depth_brace == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    tail = "".join(buf)
    if tail.strip():
        parts.append(tail)
    return parts


def _string_literals(src: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"', src):
        out.append(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Type / layout
# ---------------------------------------------------------------------------

_METAL_PRIM: dict[str, tuple[int, int]] = {
    "bool": (1, 1),
    "uchar": (1, 1),
    "char": (1, 1),
    "ushort": (2, 2),
    "short": (2, 2),
    "half": (2, 2),
    "uint": (4, 4),
    "int": (4, 4),
    "float": (4, 4),
    "ulong": (8, 8),
    "long": (8, 8),
    "double": (8, 8),
    "float2": (8, 8),
    "half2": (4, 4),
    "uint2": (8, 8),
    "int2": (8, 8),
    "uchar2": (2, 2),
    "float3": (16, 16),
    "float4": (16, 16),
    "half4": (8, 8),
    "uint3": (16, 16),
    "uint4": (16, 16),
    "int3": (16, 16),
    "int4": (16, 16),
    "uchar4": (4, 4),
    "ushort4": (8, 8),
    "half3": (8, 8),
}

_RUST_PRIM: dict[str, tuple[int, int]] = {
    "bool": (1, 1),
    "u8": (1, 1),
    "i8": (1, 1),
    "u16": (2, 2),
    "i16": (2, 2),
    "u32": (4, 4),
    "i32": (4, 4),
    "f32": (4, 4),
    "u64": (8, 8),
    "i64": (8, 8),
    "f64": (8, 8),
    "usize": (8, 8),
    "isize": (8, 8),
    "f16": (2, 2),
}


def _align_up(n: int, a: int) -> int:
    if a <= 1:
        return n
    r = n % a
    return n if r == 0 else n + (a - r)


def _metal_field_size_align(type_s: str) -> tuple[int, int] | None:
    t = re.sub(r"\s+", " ", type_s.strip())
    if "*" in t or t.startswith("device ") or t.startswith("constant ") or t.startswith("threadgroup "):
        # device/constant/threadgroup pointer in a struct is a GPU VA (8 bytes).
        if "*" in t:
            return 8, 8
    # array: T name[N] is handled at field parse; here T[N]
    m = re.fullmatch(r"(.+?)\s*\[(\d+)\]", t)
    if m:
        inner = _metal_field_size_align(m.group(1))
        if inner is None:
            return None
        return inner[0] * int(m.group(2)), inner[1]
    vec = _METAL_PRIM.get(t)
    if vec:
        return vec
    base = t.split(" ")[-1]
    return _METAL_PRIM.get(base)


def _rust_field_size_align(type_s: str) -> tuple[int, int] | None:
    t = re.sub(r"\s+", " ", type_s.strip())
    if t.startswith("*"):
        return 8, 8
    m = re.fullmatch(r"\[(.+);\s*(\d+)\]", t)
    if m:
        inner = _rust_field_size_align(m.group(1))
        if inner is None:
            return None
        return inner[0] * int(m.group(2)), inner[1]
    last = t.split("::")[-1]
    return _RUST_PRIM.get(last) or _RUST_PRIM.get(t)


def layout_fields(
    fields: list[tuple[str, str]],
    size_align: Any,
    nested: dict[str, "StructDef"] | None = None,
) -> tuple[list[dict[str, Any]], int, int] | None:
    """Return (field_rows, total_size, align) or None if a field is unresolvable."""
    off = 0
    max_a = 1
    rows: list[dict[str, Any]] = []
    nested = nested or {}
    for name, ty in fields:
        sa = size_align(ty)
        if sa is None:
            key = ty.split("::")[-1].strip()
            if key in nested:
                sa = (nested[key].size, nested[key].align)
            else:
                return None
        sz, al = sa
        off = _align_up(off, al)
        rows.append({"name": name, "type": ty, "offset": off, "size": sz, "align": al})
        off += sz
        max_a = max(max_a, al)
    total = _align_up(off, max_a)
    return rows, total, max_a


# ---------------------------------------------------------------------------
# Parsed records
# ---------------------------------------------------------------------------


@dataclass
class ShaderParam:
    name: str
    type_s: str
    space: str  # device / constant / threadgroup / builtin
    index: int | None
    attr: str
    kind: str


@dataclass
class MetalKernel:
    name: str
    path: str
    line: int
    params: list[ShaderParam]
    buffer_indices: dict[int, ShaderParam] = field(default_factory=dict)
    threadgroup_indices: dict[int, ShaderParam] = field(default_factory=dict)
    builtins: list[str] = field(default_factory=list)

    def buffer_set(self) -> set[int]:
        return set(self.buffer_indices)


@dataclass
class StructField:
    name: str
    type_s: str


@dataclass
class StructDef:
    name: str
    path: str
    line: int
    fields: list[StructField]
    lang: str
    size: int | None = None
    align: int | None = None
    layout: list[dict[str, Any]] | None = None
    layout_status: str = "UNVERIFIABLE"  # computed / nested-unresolved


@dataclass
class HostBind:
    kind: str
    index: int | None
    raw_index: str
    line: int
    hint: str = ""  # e.g. argbuf, handle()


@dataclass
class HostDispatch:
    path: str
    line: int
    method: str
    kernel_expr: str
    resolved: list[str]
    resolve_status: str  # resolved / dual / unverifiable / plumbing
    grid_raw: str
    tg_raw: str
    grid: tuple[int | None, int | None, int | None]
    tg: tuple[int | None, int | None, int | None]
    binds: list[HostBind]
    binds_unverifiable: bool
    feature_cfg: str | None
    receiver: str
    argbuf_layouts: dict[int, list[str]]  # buffer index -> ArgLayout names
    notes: list[str] = field(default_factory=list)


@dataclass
class Finding:
    severity: str
    check: str
    message: str
    host: str | None = None
    shader: str | None = None
    kernel: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
            "host": self.host,
            "shader": self.shader,
            "kernel": self.kernel,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


# ---------------------------------------------------------------------------
# Metal parser
# ---------------------------------------------------------------------------

_KERNEL_RE = re.compile(r"\bkernel\s+void\s+(\w+)\s*\(", re.M)
_STRUCT_RE = re.compile(r"\bstruct\s+(\w+)\s*\{", re.M)
_ATTR_RE = re.compile(r"\[\[([^\]]+)\]\]")
_BUF_IDX_RE = re.compile(r"buffer\((\d+)\)")
_TG_IDX_RE = re.compile(r"threadgroup\((\d+)\)")


def _classify_shader_param(type_s: str, attr: str) -> tuple[str, str, int | None, str]:
    """Return (space, kind, index, attr_head)."""
    space = "builtin"
    index = None
    attr_s = attr.strip()
    head = attr_s.split("(")[0].strip()
    if "buffer(" in attr_s:
        space = "constant" if "constant" in type_s.split("*")[0] else "device"
        # `constant T&` vs `device T*`
        lead = type_s.strip()
        if lead.startswith("constant"):
            space = "constant"
        elif lead.startswith("device") or lead.startswith("const device"):
            space = "device"
        m = _BUF_IDX_RE.search(attr_s)
        index = int(m.group(1)) if m else None
    elif "threadgroup(" in attr_s:
        space = "threadgroup"
        m = _TG_IDX_RE.search(attr_s)
        index = int(m.group(1)) if m else None
    kind = KIND_UNKNOWN
    if space == "threadgroup":
        kind = KIND_THREADGROUP
    elif space == "device":
        kind = KIND_DEVICE
    elif space == "constant":
        t = type_s.replace("constant", "").replace("&", "").replace("const", "").strip()
        t = re.sub(r"\s+", " ", t)
        if t in {"uint", "int"}:
            kind = KIND_CONSTANT_U32
        elif t in {"float", "half"}:
            kind = KIND_CONSTANT_F32
        elif "*" in type_s:
            kind = KIND_DEVICE
        else:
            kind = KIND_CONSTANT_STRUCT
    return space, kind, index, head


def parse_metal(src: str, path: str) -> tuple[list[MetalKernel], list[StructDef]]:
    clean = strip_comments_preserve_lines(src)
    kernels: list[MetalKernel] = []
    for m in _KERNEL_RE.finditer(clean):
        name = m.group(1)
        open_p = m.end() - 1
        close_p = _match_paren(clean, open_p)
        if close_p < 0:
            continue
        body = clean[open_p + 1 : close_p]
        params: list[ShaderParam] = []
        for raw in _split_top_level(body):
            chunk = raw.strip()
            if not chunk:
                continue
            am = _ATTR_RE.search(chunk)
            if not am:
                continue
            attr = am.group(1)
            before = chunk[: am.start()].strip().rstrip(",")
            # last identifier is the parameter name
            nm = re.search(r"([A-Za-z_]\w*)\s*$", before)
            pname = nm.group(1) if nm else ""
            type_s = before[: nm.start()].strip() if nm else before
            space, kind, index, head = _classify_shader_param(type_s, attr)
            params.append(
                ShaderParam(
                    name=pname,
                    type_s=type_s,
                    space=space,
                    index=index,
                    attr=head if space == "builtin" else attr,
                    kind=kind,
                )
            )
        k = MetalKernel(name=name, path=path, line=_line_of(src, m.start()), params=params)
        for p in params:
            if p.space in {"device", "constant"} and p.index is not None:
                k.buffer_indices[p.index] = p
            elif p.space == "threadgroup" and p.index is not None:
                k.threadgroup_indices[p.index] = p
            else:
                k.builtins.append(p.attr)
        kernels.append(k)

    structs: list[StructDef] = []
    for m in _STRUCT_RE.finditer(clean):
        name = m.group(1)
        open_b = m.end() - 1
        close_b = _match_paren(clean, open_b)
        if close_b < 0:
            continue
        body = clean[open_b + 1 : close_b]
        fields: list[StructField] = []
        for raw in body.split(";"):
            chunk = raw.strip()
            if not chunk or chunk.startswith("{"):
                continue
            # T name  OR  T name[N]
            fm = re.search(r"([A-Za-z_]\w*)\s*(?:\[(\d+)\])?\s*$", chunk)
            if not fm:
                continue
            fname = fm.group(1)
            if fname in {"struct", "enum", "if", "for", "return"}:
                continue
            type_s = chunk[: fm.start()].strip()
            if not type_s:
                continue
            if fm.group(2):
                type_s = f"{type_s}[{fm.group(2)}]"
            fields.append(StructField(fname, type_s))
        if not fields:
            continue
        structs.append(
            StructDef(
                name=name,
                path=path,
                line=_line_of(src, m.start()),
                fields=fields,
                lang="metal",
            )
        )
    return kernels, structs


def generated_kernel_names(metal_files: dict[str, str]) -> dict[str, dict[str, str]]:
    """Recover concrete names emitted by the small shader macro families.

    The source preflight intentionally does not run the Metal preprocessor.
    That is the right boundary for host/shader ABI parsing, but a literal
    ``kernel void`` regex cannot see token-pasted entry points. Keep the
    recovery table explicit and source-backed: the checker can close a false
    kernel-existence alarm while still reporting that binding/PSO analysis for
    the generated body is deferred to the compiler/runtime.
    """
    generated: dict[str, dict[str, str]] = {}
    specs = (
        (
            "QWEN_UNIFORM_Q4_MATMUL_K",
            re.compile(r"(?m)^\s*QWEN_UNIFORM_Q4_MATMUL_K\(\s*(\d+)\s*\)"),
            lambda m: f"qwen_uniform_q4_group64_matmul_k{m.group(1)}_geo_tpr64_tg128",
        ),
        (
            "QWEN_UNIFORM_Q4_MATMUL_RK",
            re.compile(
                r"(?m)^\s*QWEN_UNIFORM_Q4_MATMUL_RK\(\s*(\d+)\s*,\s*(\d+)\s*\)"
            ),
            lambda m: (
                f"qwen_uniform_q4_group64_matmul_r{m.group(1)}k{m.group(2)}"
                "_geo_tpr64_tg128"
            ),
        ),
        (
            "QWEN_BINARY_PLANES",
            re.compile(r"(?m)^\s*QWEN_BINARY_PLANES\(\s*(\d+)\s*\)"),
            lambda m: f"qwen_binary_planes_k{m.group(1)}_matvec_geo_tpr64_tg128",
        ),
    )
    for path, src in sorted(metal_files.items()):
        if not path.endswith("qwen_uniform_q4.metal"):
            continue
        for macro, call_re, name_fn in specs:
            for match in call_re.finditer(src):
                name = name_fn(match)
                generated[name] = {
                    "path": path,
                    "macro": macro,
                    "invocation": match.group(0).strip(),
                    "verification": (
                        "NAME_RECOVERED_FROM_MACRO; body binding and PSO geometry remain "
                        "compiler/runtime checks"
                    ),
                }
    return generated


# ---------------------------------------------------------------------------
# Rust host parser
# ---------------------------------------------------------------------------

_DISPATCH_RE = re.compile(
    r"\.dispatch_threads(?:_timed|_in_concurrent_group|_pair_in_one_encoder)?\s*\(",
)
_CFG_FEATURE_RE = re.compile(r'#\[cfg\((?:feature\s*=\s*"([^"]+)"|all\([^]]*feature\s*=\s*"([^"]+)"[^]]*)\)\]')
_CONST_STR_RE = re.compile(
    r"(?:pub(?:\([^)]+\))?\s+)?(?:const|static)\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:'static\s+)?str\s*=\s*\"([A-Za-z_][A-Za-z0-9_]*)\""
)
_CONST_U32_RE = re.compile(
    r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*u32\s*=\s*(\d+)"
)
_ARGLAYOUT_RE = re.compile(
    r"KernelArgBuffer::new\s*\([^,]+,\s*&\[([^\]]+)\]"
)
_REPR_C_STRUCT_RE = re.compile(
    r"#\[repr\(C[^\]]*\)\](?:\s*#\[[^\]]+\])*\s*(?:pub(?:\([^)]+\))?\s+)?struct\s+(\w+)\s*\{",
    re.M,
)
_INCLUDE_STR_SHADER_RE = re.compile(r'include_str!\s*\(\s*(?:concat!\([^)]*\))?\s*[^)]*shaders/([^"\)]+\.metal)')
_PICK_FN_RE = re.compile(
    r"pub fn (\w+)\s*\(\s*\)\s*->\s*&'static str\s*\{[^}]*pick\(\s*([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)",
    re.S,
)


def _enclosing_fn(src: str, pos: int) -> tuple[int, int, str, str | None]:
    """Return (fn_start, fn_end, fn_src, cfg_feature) for the function containing pos."""
    best = -1
    for m in re.finditer(r"(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)", src):
        if m.start() > pos:
            break
        # find opening brace after signature
        brace = src.find("{", m.end())
        if brace < 0 or brace > pos:
            continue
        end = _match_paren(src, brace)
        if end >= pos:
            best = m.start()
            fn_end = end
            fn_src = src[best : fn_end + 1]
            # cfg immediately above
            pre = src[max(0, best - 400) : best]
            cfg = None
            cm = list(_CFG_FEATURE_RE.finditer(pre))
            if cm:
                last = cm[-1]
                cfg = last.group(1) or last.group(2)
            return best, fn_end, fn_src, cfg
    return 0, len(src), src, None


def _parse_triple(expr: str, consts: dict[str, int]) -> tuple[int | None, int | None, int | None]:
    e = expr.strip()
    if not (e.startswith("(") and e.endswith(")")):
        return (None, None, None)
    parts = _split_top_level(e[1:-1])
    if len(parts) != 3:
        return (None, None, None)

    def one(p: str) -> int | None:
        s = p.strip()
        s = re.sub(r"\s+as\s+u32$", "", s)
        if re.fullmatch(r"\d+", s):
            return int(s)
        if s in consts:
            return consts[s]
        # N as u32 already stripped; allow trailing `u32` suffix literals 256u
        m = re.fullmatch(r"(\d+)(?:u32|u64|usize)?", s)
        if m:
            return int(m.group(1))
        return None

    return one(parts[0]), one(parts[1]), one(parts[2])


def _receiver_before(src: str, call_dot: int) -> str:
    i = call_dot - 1
    while i >= 0 and src[i].isspace():
        i -= 1
    end = i + 1
    while i >= 0 and (src[i].isalnum() or src[i] in "_"):
        i -= 1
    return src[i + 1 : end]


def _resolve_kernel_expr(
    expr: str,
    fn_src: str,
    file_const_str: dict[str, str],
    pick_fns: dict[str, tuple[str, str]],
    metal_names: set[str],
) -> tuple[list[str], str]:
    e = expr.strip().rstrip(",")
    m = re.fullmatch(r'"([A-Za-z_][A-Za-z0-9_]*)"', e)
    if m:
        return [m.group(1)], "resolved"
    # decode_family::foo()
    m = re.search(r"decode_family::(\w+)\s*\(\s*\)", e)
    if m:
        fn = m.group(1)
        if fn in pick_fns:
            a, b = pick_fns[fn]
            names = []
            for key in (a, b):
                if key in file_const_str:
                    names.append(file_const_str[key])
                elif key in metal_names:
                    names.append(key)
            # also try the ident as a const in decode_family
            return names or [fn], "dual" if len(names) > 1 else "resolved"
        if fn in file_const_str:
            return [file_const_str[fn]], "resolved"
    if e in {"fn_name", "first_name", "second_name"}:
        return [], "plumbing"
    # const ident
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", e):
        # local const first
        for cm in _CONST_STR_RE.finditer(fn_src):
            if cm.group(1) == e:
                return [cm.group(2)], "resolved"
        if e in file_const_str:
            return [file_const_str[e]], "resolved"
    # identifier: only string literals assigned to THIS ident, not every
    # kernel name mentioned in a large function.
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", e):
        for cm in _CONST_STR_RE.finditer(fn_src):
            if cm.group(1) == e:
                return [cm.group(2)], "resolved"
        assigned: list[str] = []
        for m in re.finditer(
            rf"(?:let\s+(?:mut\s+)?(?:\([^;]*\b{re.escape(e)}\b[^;]*\)|{re.escape(e)})\s*(?::[^=]+)?\s*=|{re.escape(e)}\s*=)",
            fn_src,
        ):
            rest = fn_src[m.end() : m.end() + 2500]
            assigned.extend(s for s in _string_literals(rest[: rest.find(";") + 1 or 2500]) if s in metal_names)
        uniq = sorted(set(assigned))
        if uniq:
            return uniq, "dual" if len(uniq) > 1 else "resolved"
        return [], "unverifiable"
    return [], "unverifiable"


_METHOD_BIND_RE = re.compile(
    r"(?P<recv>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?P<meth>[A-Za-z0-9_]*set_u32|[A-Za-z0-9_]*set_f32|set_buffer|set_bytes|"
    r"set_threadgroup_memory_length)\s*\(\s*(?P<idx>[^,\)]+)"
)
_FREE_BIND_RE = re.compile(
    r"(?<![.\w])(?P<meth>set_u32|set_f32|set_params)\s*\(\s*[^,]+,\s*(?P<idx>[^,\)]+)"
)
_ARGBUF_RECV = frozenset({"ab", "argbuf", "args", "layout", "arg_buf", "arg_buffer"})


def _index_from_raw(raw: str) -> int | None:
    s = raw.strip()
    s = re.sub(r"\s+as\s+u(?:32|64)$", "", s)
    m = re.fullmatch(r"(\d+)(?:u32|u64|usize)?", s)
    return int(m.group(1)) if m else None


def _extract_binds(closure_src: str, path: str, closure_abs_start: int, src: str) -> tuple[list[HostBind], bool, dict[int, list[str]]]:
    binds: list[HostBind] = []
    unverifiable = False

    def add(kind: str, raw: str, pos: int, hint: str = "") -> None:
        nonlocal unverifiable
        idx = _index_from_raw(raw)
        if idx is None:
            unverifiable = True
        line = _line_of(src, closure_abs_start + pos)
        binds.append(HostBind(kind=kind, index=idx, raw_index=raw.strip(), line=line, hint=hint))

    for m in _METHOD_BIND_RE.finditer(closure_src):
        recv = m.group("recv")
        meth = m.group("meth")
        raw = m.group("idx")
        if recv in _ARGBUF_RECV and meth.endswith(("set_u32", "set_f32")):
            # KernelArgBuffer field index, not a Metal buffer slot.
            continue
        if meth == "set_buffer":
            kind = KIND_DEVICE
        elif meth.endswith("set_u32"):
            kind = KIND_CONSTANT_U32
        elif meth.endswith("set_f32"):
            kind = KIND_CONSTANT_F32
        elif meth == "set_bytes":
            kind = KIND_CONSTANT_BYTES
        else:
            kind = KIND_THREADGROUP
        hint = ""
        open_paren = closure_src.find("(", m.start())
        close_paren = _match_paren(closure_src, open_paren) if open_paren >= 0 else -1
        call = closure_src[open_paren : close_paren + 1] if close_paren >= 0 else ""
        if meth == "set_buffer" and "handle()" in call:
            hint = "argbuf"
            kind = KIND_CONSTANT_STRUCT
        add(kind, raw, m.start(), hint)

    for m in _FREE_BIND_RE.finditer(closure_src):
        meth = m.group("meth")
        raw = m.group("idx")
        if meth == "set_u32":
            kind = KIND_CONSTANT_U32
        elif meth == "set_f32":
            kind = KIND_CONSTANT_F32
        else:
            kind = KIND_CONSTANT_BYTES
        add(kind, raw, m.start())

    if not binds:
        # A closure that only forwards to a helper we did not expand is not an
        # empty encode; it is UNVERIFIABLE.
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", closure_src):
            unverifiable = True

    argbuf_at: dict[int, list[str]] = {}
    return binds, unverifiable, argbuf_at


def _arglayouts_in(fn_src: str) -> list[list[str]]:
    out: list[list[str]] = []
    for m in _ARGLAYOUT_RE.finditer(fn_src):
        names = re.findall(r"ArgLayout::(\w+)", m.group(1))
        if names:
            out.append(names)
    return out


def parse_rust_host(
    src: str,
    path: str,
    metal_names: set[str],
    file_const_str: dict[str, str] | None = None,
    pick_fns: dict[str, tuple[str, str]] | None = None,
) -> tuple[list[HostDispatch], list[StructDef], list[str], dict[str, str]]:
    file_const_str = dict(file_const_str or {})
    pick_fns = dict(pick_fns or {})
    for m in _CONST_STR_RE.finditer(src):
        file_const_str[m.group(1)] = m.group(2)
    for m in _PICK_FN_RE.finditer(src):
        pick_fns[m.group(1)] = (m.group(2), m.group(3))

    includes = [m.group(1) for m in _INCLUDE_STR_SHADER_RE.finditer(src)]
    # also catch include_str!("../../shaders/foo.metal")
    includes += re.findall(r'shaders/([A-Za-z0-9_]+\.metal)', src)
    includes = sorted(set(includes))

    structs: list[StructDef] = []
    for m in _REPR_C_STRUCT_RE.finditer(src):
        name = m.group(1)
        open_b = src.find("{", m.end() - 1)
        if open_b < 0:
            continue
        close_b = _match_paren(src, open_b)
        if close_b < 0:
            continue
        body = strip_comments_preserve_lines(src[open_b + 1 : close_b])
        fields: list[StructField] = []
        for raw in _split_top_level(body):
            chunk = raw.strip().rstrip(",")
            if not chunk or chunk.startswith("#") or chunk.startswith("//"):
                continue
            # pub name: Type
            fm = re.match(r"(?:pub(?:\([^)]+\))?\s+)?([A-Za-z_]\w*)\s*:\s*(.+)$", chunk, re.S)
            if not fm:
                continue
            fname, ty = fm.group(1), re.sub(r"\s+", " ", fm.group(2).strip())
            if fname in {"fn", "impl", "const"}:
                continue
            fields.append(StructField(fname, ty))
        if fields:
            structs.append(
                StructDef(
                    name=name,
                    path=path,
                    line=_line_of(src, m.start()),
                    fields=fields,
                    lang="rust",
                )
            )

    dispatches: list[HostDispatch] = []
    for m in _DISPATCH_RE.finditer(src):
        method = m.group(0).lstrip(".").split("(")[0].strip()
        open_p = m.end() - 1
        close_p = _match_paren(src, open_p)
        if close_p < 0:
            continue
        args_src = src[open_p + 1 : close_p]
        args = [a.strip() for a in _split_top_level(args_src)]
        if method.endswith("pair_in_one_encoder"):
            # first_name, grid, tg, encode, barrier, second_name, grid, tg, encode
            groups = []
            if len(args) >= 4:
                groups.append(args[0:4])
            if len(args) >= 9:
                groups.append(args[5:9])
        else:
            groups = [args[:4]] if len(args) >= 3 else []

        fn_start, fn_end, fn_src, cfg = _enclosing_fn(src, m.start())
        local_consts = {k: int(v) for k, v in _CONST_U32_RE.findall(fn_src)}
        # file-level u32 consts visible too
        for k, v in _CONST_U32_RE.findall(src[: fn_start + 1][-4000:]):
            local_consts.setdefault(k, int(v))
        arglayouts = _arglayouts_in(fn_src)

        for g in groups:
            if len(g) < 3:
                continue
            kexpr, grid_raw, tg_raw = g[0], g[1], g[2]
            encode_arg = g[3] if len(g) > 3 else ""
            if "MTLSize" in kexpr:
                continue
            if encode_arg.strip() in {"encode", "first_encode", "second_encode"}:
                # plumbing: the MetalContext/TCB method itself
                continue
            fn_prefix = src[fn_start : m.start()]
            resolved, status = _resolve_kernel_expr(
                kexpr, fn_prefix, file_const_str, pick_fns, metal_names
            )
            if status == "plumbing":
                continue
            # closure body
            closure = encode_arg
            cm = re.search(r"\|[^|]*\|\s*(\{)?", encode_arg)
            binds: list[HostBind] = []
            binds_uv = False
            if cm:
                if cm.group(1):
                    # braced
                    rel = encode_arg.find("{")
                    # find matching in encode_arg
                    abs_open = open_p + 1 + args_src.find(encode_arg) + rel
                    # simpler: parse from encode_arg
                    copen = encode_arg.find("{")
                    cclose = _match_paren(encode_arg, copen) if copen >= 0 else -1
                    body = encode_arg[copen : cclose + 1] if cclose >= 0 else encode_arg
                else:
                    body = encode_arg
                abs_start = src.find(encode_arg, open_p)
                binds, binds_uv, _ = _extract_binds(body, path, abs_start if abs_start >= 0 else m.start(), src)
            else:
                # named helper call: |enc| foo(enc, ...)  OR just foo
                helper = re.search(r"([A-Za-z_]\w+)\s*\(", encode_arg)
                if helper:
                    hname = helper.group(1)
                    hm = re.search(rf"fn\s+{re.escape(hname)}\s*\(", src)
                    if hm:
                        _, _, hsrc, _ = _enclosing_fn(src, hm.start())
                        binds, binds_uv, _ = _extract_binds(hsrc, path, hm.start(), src)
                    else:
                        binds_uv = True
                elif encode_arg.strip():
                    binds_uv = True

            # Associate ArgLayout sequences with argbuf binds in order.
            argbuf_map: dict[int, list[str]] = {}
            ab_binds = [b for b in binds if b.hint == "argbuf" and b.index is not None]
            for layout, b in zip(arglayouts, ab_binds):
                argbuf_map[b.index] = layout  # type: ignore[index]

            rec = _receiver_before(src, m.start())
            dispatches.append(
                HostDispatch(
                    path=path,
                    line=_line_of(src, m.start()),
                    method=method,
                    kernel_expr=kexpr.strip()[:160],
                    resolved=resolved,
                    resolve_status=status,
                    grid_raw=grid_raw.strip()[:160],
                    tg_raw=tg_raw.strip()[:160],
                    grid=_parse_triple(grid_raw, local_consts),
                    tg=_parse_triple(tg_raw, local_consts),
                    binds=binds,
                    binds_unverifiable=binds_uv,
                    feature_cfg=cfg,
                    receiver=rec,
                    argbuf_layouts=argbuf_map,
                )
            )

    return dispatches, structs, includes, file_const_str


def parse_decode_family(src: str) -> tuple[dict[str, str], dict[str, tuple[str, str]], list[str]]:
    consts: dict[str, str] = {}
    for m in _CONST_STR_RE.finditer(src):
        consts[m.group(1)] = m.group(2)
    # also allow lowercase? decode_family uses UPPER.
    picks: dict[str, tuple[str, str]] = {}
    for m in _PICK_FN_RE.finditer(src):
        picks[m.group(1)] = (m.group(2), m.group(3))
    named: list[str] = []
    for m in re.finditer(
        r"(?:FAMILY_KERNELS|Q80_GRAPH_KERNELS|Q80_TILE_KERNELS|Q80_GRAPH_SIMD_KERNELS|"
        r"QWEN38_GRAPH_KERNELS|DSV4F_GRAPH_KERNELS)\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\]",
        src,
        re.S,
    ):
        named.extend(re.findall(r"([A-Z][A-Z0-9_]*)", m.group(1)))
    kernel_names = []
    for n in named:
        if n in consts:
            kernel_names.append(consts[n])
    kernel_names.extend(consts.values())
    return consts, picks, sorted(set(kernel_names))


# ---------------------------------------------------------------------------
# Library membership (control-path availability)
# ---------------------------------------------------------------------------

_SHADER_CONST_RE = re.compile(
    r'(SHADER_[A-Z0-9_]+)\s*:\s*&str\s*=\s*include_str!\s*\(\s*"\.\./\.\./shaders/([^"]+)"',
)


def parse_cfg_modules(lib_src: str) -> dict[str, str]:
    """Map rust module name -> cargo feature from `#[cfg(feature=...)] mod foo`."""
    out: dict[str, str] = {}
    for m in re.finditer(
        r'#\[cfg\((?:feature\s*=\s*"([^"]+)"|all\([^]]*feature\s*=\s*"([^"]+)"[^]]*)\)\]\s*'
        r"(?:pub(?:\([^)]+\))?\s+)?(?:mod|use)\s+([A-Za-z_][A-Za-z0-9_]*)",
        lib_src,
    ):
        out[m.group(3)] = m.group(1) or m.group(2)
    return out


def parse_library_membership(metal_mod_src: str) -> dict[str, str]:
    """Map shader filename -> 'default' | 'tq' based on all_shader_sources."""
    const_to_file = {m.group(1): m.group(2) for m in _SHADER_CONST_RE.finditer(metal_mod_src)}
    # default: every SHADER_* mentioned in all_shader_sources before the tq push
    fn_m = re.search(r"fn all_shader_sources\s*\(\s*\)[^{]*\{", metal_mod_src)
    membership: dict[str, str] = {}
    if not fn_m:
        return {fn: "default" for fn in const_to_file.values()}
    brace = metal_mod_src.find("{", fn_m.end() - 1)
    end = _match_paren(metal_mod_src, brace)
    body = metal_mod_src[brace : end + 1]
    tq_at = body.find('#[cfg(feature = "tq")]')
    default_body = body[: tq_at if tq_at >= 0 else len(body)]
    tq_body = body[tq_at:] if tq_at >= 0 else ""
    for const, fn in const_to_file.items():
        if const in tq_body and const not in default_body:
            membership[fn] = "tq"
        elif const in default_body:
            membership[fn] = "default"
    return membership


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _compatible_kinds(host: str, shader: str) -> bool:
    if host == shader:
        return True
    # Host may bind a small MTLBuffer onto a constant slot (scalar or struct).
    # `set_buffer(7, Some(&srb_buf))` for `constant uint& scale_row_base [[buffer(7)]]`
    # is legal Metal. set_u32 onto a device pointer is not.
    if host == KIND_DEVICE and shader in {
        KIND_CONSTANT_STRUCT,
        KIND_CONSTANT_U32,
        KIND_CONSTANT_F32,
        KIND_CONSTANT_BYTES,
    }:
        return True
    if host == KIND_CONSTANT_STRUCT and shader == KIND_CONSTANT_STRUCT:
        return True
    # set_bytes onto a constant scalar/struct
    if host == KIND_CONSTANT_BYTES and shader in {
        KIND_CONSTANT_U32,
        KIND_CONSTANT_F32,
        KIND_CONSTANT_STRUCT,
    }:
        return True
    # set_u32/set_f32 onto matching constant
    if host == KIND_CONSTANT_U32 and shader == KIND_CONSTANT_U32:
        return True
    if host == KIND_CONSTANT_F32 and shader == KIND_CONSTANT_F32:
        return True
    return False


_ARGLAYOUT_WIDTH = {"U32": (4, 4, "uint"), "F32": (4, 4, "float"), "U64": (8, 8, "ulong")}


def _abi_field_names_compatible(metal_name: str, rust_name: str, metal_ty: str, rust_ty: str) -> bool:
    """Field names may differ when the host stores a GPU VA as u64.

    Anything else at the same slot with a different name is field-order drift.
    """
    if metal_name == rust_name:
        return True
    if metal_name in rust_name or rust_name in metal_name:
        return True
    mt = metal_ty.replace("const", " ")
    if "*" in metal_ty or "device" in mt.split("*")[0]:
        last = rust_ty.split("::")[-1]
        if last in {"u64", "usize"}:
            return True
    return False


def compute_struct_layouts(structs: list[StructDef]) -> None:
    by_lang: dict[str, dict[str, StructDef]] = {"metal": {}, "rust": {}}
    for s in structs:
        by_lang[s.lang][s.name] = s
    # iterate until nested names resolve or we stall
    for lang, sa in (("metal", _metal_field_size_align), ("rust", _rust_field_size_align)):
        pending = list(by_lang[lang].values())
        changed = True
        while changed:
            changed = False
            nested = {n: st for n, st in by_lang[lang].items() if st.size is not None}
            for st in pending:
                if st.size is not None:
                    continue
                pairs = [(f.name, f.type_s) for f in st.fields]
                got = layout_fields(pairs, sa, nested)
                if got is None:
                    continue
                rows, total, al = got
                st.layout = rows
                st.size = total
                st.align = al
                st.layout_status = "computed"
                changed = True
        for st in pending:
            if st.size is None:
                st.layout_status = "UNVERIFIABLE"


def analyze(
    metal_files: dict[str, str],
    rust_files: dict[str, str],
    *,
    library_membership: dict[str, str] | None = None,
    production_host_prefix: str = "crates/hawking-core/src/",
) -> dict[str, Any]:
    """Pure-function preflight. Keys are repo-relative paths; values are source text.

    A result of PASS is only emitted when the checker followed both sides to a
    concrete index/type. Anything it could not follow is UNVERIFIABLE, never PASS.
    """
    findings: list[Finding] = []
    kernels: list[MetalKernel] = []
    metal_structs: list[StructDef] = []
    for path, src in sorted(metal_files.items()):
        ks, sts = parse_metal(src, path)
        kernels.extend(ks)
        metal_structs.extend(sts)

    by_name: dict[str, list[MetalKernel]] = defaultdict(list)
    for k in kernels:
        by_name[k.name].append(k)
    generated_names = generated_kernel_names(metal_files)
    # Generated names are existence-resolved, but intentionally not inserted
    # into `by_name`: without preprocessing there is no trustworthy parameter
    # list to use for host binding/geometry checks.
    metal_names = set(by_name) | set(generated_names)

    for name, ks in sorted(by_name.items()):
        if len(ks) > 1:
            findings.append(
                Finding(
                    "WARNING",
                    "duplicate_kernel_name",
                    f"kernel {name!r} is defined in {len(ks)} shader files",
                    shader=", ".join(_loc(k.path, k.line) for k in ks),
                    kernel=name,
                )
            )

    # decode_family first so pick() maps are available to every host file
    decode_consts: dict[str, str] = {}
    pick_fns: dict[str, tuple[str, str]] = {}
    family_named: list[str] = []
    for path, src in rust_files.items():
        if path.endswith("decode_family.rs"):
            decode_consts, pick_fns, family_named = parse_decode_family(src)
            break

    module_feat: dict[str, str] = {}
    for path, src in rust_files.items():
        if path.endswith("src/lib.rs") or path.endswith("/lib.rs"):
            module_feat.update(parse_cfg_modules(src))
            break

    all_dispatches: list[HostDispatch] = []
    rust_structs: list[StructDef] = []
    includes_by_file: dict[str, list[str]] = {}
    merged_consts = dict(decode_consts)
    for path, src in sorted(rust_files.items()):
        ds, sts, includes, consts = parse_rust_host(
            src, path, metal_names, merged_consts, pick_fns
        )
        merged_consts.update(consts)
        stem = Path(path).stem
        file_feat = module_feat.get(stem)
        if file_feat:
            for d in ds:
                if d.feature_cfg is None:
                    d.feature_cfg = file_feat
        all_dispatches.extend(ds)
        rust_structs.extend(sts)
        includes_by_file[path] = includes

    compute_struct_layouts(metal_structs)
    compute_struct_layouts(rust_structs)

    # ---- 1. kernel existence ----
    referenced: dict[str, list[str]] = defaultdict(list)
    for d in all_dispatches:
        for n in d.resolved:
            referenced[n].append(_loc(d.path, d.line))
    for n in family_named:
        referenced.setdefault(n, []).append("crates/hawking-core/src/decode_family.rs:FAMILY")
    # const strings that match a kernel name
    for path, src in rust_files.items():
        for lit in _string_literals(src):
            if lit in metal_names:
                referenced.setdefault(lit, []).append(path)

    missing_host_refs = sorted(n for n in referenced if n not in metal_names)
    for n in missing_host_refs:
        findings.append(
            Finding(
                "ERROR",
                "kernel_existence",
                f"host references kernel {n!r} but no shader defines `kernel void {n}(`",
                host=referenced[n][0],
                shader=None,
                kernel=n,
                extra={"host_sites": referenced[n][:8]},
            )
        )

    # ---- 2/3/4/5 per-dispatch binding, type, geometry ----
    binding_checked = 0
    geometry_checked = 0
    uv_dispatch = 0
    for d in all_dispatches:
        host_loc = _loc(d.path, d.line)
        if d.resolve_status == "unverifiable" or not d.resolved:
            uv_dispatch += 1
            findings.append(
                Finding(
                    "UNVERIFIABLE",
                    "kernel_name",
                    "dispatch kernel name is not a string literal or resolvable const; "
                    "not scored as PASS",
                    host=host_loc,
                    kernel=d.kernel_expr,
                    extra={"expr": d.kernel_expr, "status": d.resolve_status},
                )
            )
            continue
        for kname in d.resolved:
            ks = by_name.get(kname)
            if not ks:
                # existence already recorded
                continue
            shader = ks[0]
            shader_loc = _loc(shader.path, shader.line)

            if d.binds_unverifiable or any(b.index is None for b in d.binds):
                findings.append(
                    Finding(
                        "UNVERIFIABLE",
                        "binding_index",
                        "one or more host binds use a non-literal index or a helper "
                        "this checker cannot follow; not scored as PASS",
                        host=host_loc,
                        shader=shader_loc,
                        kernel=kname,
                    )
                )
            else:
                host_bufs = {
                    b.index
                    for b in d.binds
                    if b.index is not None and b.kind != KIND_THREADGROUP
                }
                shader_bufs = shader.buffer_set()
                missing = sorted(shader_bufs - host_bufs)
                extra_h = sorted(host_bufs - shader_bufs)
                off_by_one = (
                    len(host_bufs) == len(shader_bufs)
                    and host_bufs
                    and shader_bufs
                    and (
                        {i - 1 for i in host_bufs} == shader_bufs
                        or {i + 1 for i in host_bufs} == shader_bufs
                    )
                )
                if missing or extra_h:
                    msg = (
                        f"host binds buffers {sorted(host_bufs)} but shader declares {sorted(shader_bufs)}"
                    )
                    if off_by_one:
                        msg = "off-by-one buffer index: " + msg
                    # Extra host slots (no missing shader slot) are often a
                    # runtime-path optional bind. A missing shader slot is the
                    # protected-window killer and stays ERROR.
                    extra_only = bool(extra_h) and not missing and not off_by_one
                    sev = "WARNING" if extra_only else "ERROR"
                    findings.append(
                        Finding(
                            sev,
                            "binding_index",
                            msg,
                            host=host_loc,
                            shader=shader_loc,
                            kernel=kname,
                            extra={
                                "host_indices": sorted(host_bufs),
                                "shader_indices": sorted(shader_bufs),
                                "missing_on_host": missing,
                                "extra_on_host": extra_h,
                                "off_by_one": off_by_one,
                            },
                        )
                    )
                    if extra_only:
                        binding_checked += 1
                else:
                    binding_checked += 1
                by_host = {b.index: b for b in d.binds if b.index is not None and b.kind != KIND_THREADGROUP}
                type_ok = True
                for idx, sp in sorted(shader.buffer_indices.items()):
                    hb = by_host.get(idx)
                    if hb is None:
                        continue
                    if not _compatible_kinds(hb.kind, sp.kind):
                        type_ok = False
                        findings.append(
                            Finding(
                                "ERROR",
                                "type_width",
                                f"buffer({idx}): host bind kind {hb.kind} is not compatible "
                                f"with shader {sp.kind} ({sp.type_s!r} {sp.name})",
                                host=_loc(d.path, hb.line),
                                shader=shader_loc,
                                kernel=kname,
                                extra={"index": idx, "host_kind": hb.kind, "shader_kind": sp.kind},
                            )
                        )
                if type_ok and not missing:
                    findings.append(
                        Finding(
                            "INFO",
                            "type_width",
                            f"host bind kinds are compatible with shader {kname}",
                            host=host_loc,
                            shader=shader_loc,
                            kernel=kname,
                        )
                    )
                for idx, layout in d.argbuf_layouts.items():
                    sp = shader.buffer_indices.get(idx)
                    if sp is None or sp.kind != KIND_CONSTANT_STRUCT:
                        continue
                    st_name = re.sub(r"[&*]", "", sp.type_s)
                    st_name = st_name.replace("constant", "").replace("const", "").strip().split()[-1]
                    ms = next((s for s in metal_structs if s.name == st_name), None)
                    if ms is None or ms.layout is None:
                        findings.append(
                            Finding(
                                "UNVERIFIABLE",
                                "host_shader_abi",
                                f"argbuf at buffer({idx}) names shader struct {st_name} "
                                f"whose layout could not be fully resolved",
                                host=host_loc,
                                shader=shader_loc,
                                kernel=kname,
                            )
                        )
                        continue
                    host_w = [_ARGLAYOUT_WIDTH[x][0] for x in layout if x in _ARGLAYOUT_WIDTH]
                    sh_w = [row["size"] for row in ms.layout]
                    if host_w != sh_w:
                        findings.append(
                            Finding(
                                "ERROR",
                                "host_shader_abi",
                                f"KernelArgBuffer layout {layout} widths {host_w} != "
                                f"shader struct {st_name} field widths {sh_w}",
                                host=host_loc,
                                shader=_loc(ms.path, ms.line),
                                kernel=kname,
                            )
                        )
                    else:
                        findings.append(
                            Finding(
                                "INFO",
                                "host_shader_abi",
                                f"KernelArgBuffer {layout} matches {st_name} field widths",
                                host=host_loc,
                                shader=_loc(ms.path, ms.line),
                                kernel=kname,
                            )
                        )

            # threadgroup memory slots
            host_tg_slots = {b.index for b in d.binds if b.kind == KIND_THREADGROUP and b.index is not None}
            shader_tg_slots = set(shader.threadgroup_indices)
            if shader_tg_slots and not d.binds_unverifiable:
                if host_tg_slots != shader_tg_slots and not host_tg_slots.issuperset(shader_tg_slots):
                    # missing tg mem is often conditional (if shmem_bytes > 0); do not ERROR blindly
                    findings.append(
                        Finding(
                            "WARNING",
                            "threadgroup_memory",
                            f"shader threadgroup slots {sorted(shader_tg_slots)} vs host "
                            f"set_threadgroup_memory_length {sorted(host_tg_slots)}",
                            host=host_loc,
                            shader=shader_loc,
                            kernel=kname,
                        )
                    )

            # geometry
            tx, ty, tz = d.tg
            gx, gy, gz = d.grid
            if None not in (tx, ty, tz):
                geometry_checked += 1
                prod = int(tx) * int(ty) * int(tz)  # type: ignore[arg-type]
                if prod == 0:
                    findings.append(
                        Finding(
                            "ERROR",
                            "dispatch_geometry",
                            f"threadgroup {d.tg} has a zero axis",
                            host=host_loc,
                            shader=shader_loc,
                            kernel=kname,
                        )
                    )
                elif prod > APPLE_MAX_THREADS_PER_THREADGROUP:
                    findings.append(
                        Finding(
                            "ERROR",
                            "dispatch_geometry",
                            f"threadgroup {d.tg} product {prod} exceeds Apple Silicon "
                            f"device limit {APPLE_MAX_THREADS_PER_THREADGROUP}",
                            host=host_loc,
                            shader=shader_loc,
                            kernel=kname,
                            extra={"threads_per_threadgroup": prod},
                        )
                    )
                else:
                    if prod % APPLE_SIMDGROUP_WIDTH != 0:
                        findings.append(
                            Finding(
                                "WARNING",
                                "dispatch_geometry",
                                f"threadgroup product {prod} is not a multiple of "
                                f"simdgroup width {APPLE_SIMDGROUP_WIDTH}",
                                host=host_loc,
                                shader=shader_loc,
                                kernel=kname,
                            )
                        )
            else:
                findings.append(
                    Finding(
                        "UNVERIFIABLE",
                        "dispatch_geometry",
                        "threadgroup size is not a literal/const triple; coverage of the "
                        "problem size is not scored as PASS",
                        host=host_loc,
                        shader=shader_loc,
                        kernel=kname,
                        extra={"tg_raw": d.tg_raw, "grid_raw": d.grid_raw},
                    )
                )
            if None not in (gx, gy, gz) and None not in (tx, ty, tz):
                gprod = int(gx) * int(gy) * int(gz)  # type: ignore[arg-type]
                tprod = int(tx) * int(ty) * int(tz)  # type: ignore[arg-type]
                if gprod == 0:
                    findings.append(
                        Finding(
                            "ERROR",
                            "dispatch_geometry",
                            f"grid {d.grid} has a zero axis (would launch no threads)",
                            host=host_loc,
                            shader=shader_loc,
                            kernel=kname,
                        )
                    )
                # dispatch_threads takes a thread count, not a threadgroup count.
                # Metal rounds up to a whole number of threadgroups; not an error.
                _ = tprod  # used only for the zero/limit checks above

    # ---- 5. named struct ABI (repr(C) vs metal struct) ----
    rust_by = {s.name: s for s in rust_structs}
    metal_by = {s.name: s for s in metal_structs}
    paired = 0
    for mname, ms in sorted(metal_by.items()):
        rs = rust_by.get(mname)
        if rs is None:
            # suffix match: GravityDeviceExpertTensorRef vs DeviceExpertTensorRef
            cands = [s for n, s in rust_by.items() if mname.endswith(n) and n]
            if len(cands) == 1:
                rs = cands[0]
        if rs is None:
            continue
        if ms.layout is None or rs.layout is None:
            findings.append(
                Finding(
                    "UNVERIFIABLE",
                    "host_shader_abi",
                    f"struct {mname} exists on both sides but a nested field could not be sized",
                    host=_loc(rs.path, rs.line),
                    shader=_loc(ms.path, ms.line),
                    extra={"rust": rs.name, "metal": ms.name},
                )
            )
            continue
        paired += 1
        mismatches: list[str] = []
        if len(ms.layout) != len(rs.layout):
            mismatches.append(f"field count metal={len(ms.layout)} rust={len(rs.layout)}")
        n = min(len(ms.layout), len(rs.layout))
        for i in range(n):
            mf, rf = ms.layout[i], rs.layout[i]
            if mf["size"] != rf["size"] or mf["offset"] != rf["offset"]:
                mismatches.append(
                    f"field[{i}] metal {mf['name']}:{mf['type']} @{mf['offset']}+{mf['size']} "
                    f"vs rust {rf['name']}:{rf['type']} @{rf['offset']}+{rf['size']}"
                )
            elif not _abi_field_names_compatible(mf["name"], rf["name"], mf["type"], rf["type"]):
                mismatches.append(
                    f"field[{i}] order/name metal {mf['name']!r} vs rust {rf['name']!r} "
                    f"(same width {mf['size']}; field order is ABI)"
                )
        if ms.size != rs.size:
            mismatches.append(f"sizeof metal={ms.size} rust={rs.size}")
        if mismatches:
            findings.append(
                Finding(
                    "ERROR",
                    "host_shader_abi",
                    f"ABI drift in {mname}: " + "; ".join(mismatches[:6]),
                    host=_loc(rs.path, rs.line),
                    shader=_loc(ms.path, ms.line),
                    extra={"mismatches": mismatches[:12]},
                )
            )
        else:
            findings.append(
                Finding(
                    "INFO",
                    "host_shader_abi",
                    f"repr(C) {rs.name} matches metal {ms.name} ({ms.size} bytes, {len(ms.layout)} fields)",
                    host=_loc(rs.path, rs.line),
                    shader=_loc(ms.path, ms.line),
                )
            )

    # ---- 6. feature-gate wiring ----
    tq_kernels = [k for k in kernels if Path(k.path).name == "strand_bitslice.metal"]
    tq_names = {k.name for k in tq_kernels}
    for k in tq_kernels:
        sites = [d for d in all_dispatches if k.name in d.resolved]
        if not sites:
            # still referenced by string elsewhere?
            if k.name in referenced:
                continue
            findings.append(
                Finding(
                    "WARNING",
                    "feature_gate",
                    f"tq kernel {k.name} has no statically resolved host dispatch",
                    shader=_loc(k.path, k.line),
                    kernel=k.name,
                )
            )
            continue
        cfgs = {d.feature_cfg for d in sites}
        off_reachable = any(d.feature_cfg != "tq" for d in sites)
        on_reachable = any(d.feature_cfg == "tq" or d.feature_cfg is None for d in sites)
        # production src host without cfg(feature="tq") would keep it reachable when off
        if off_reachable and any(
            d.path.startswith(production_host_prefix) and d.feature_cfg != "tq" for d in sites
        ):
            findings.append(
                Finding(
                    "ERROR",
                    "feature_gate",
                    f"kernel {k.name} is dispatched from a production host without "
                    f"#[cfg(feature = \"tq\")], so it is still reachable when the flag is off "
                    f"(or would fail pipeline lookup on a default library)",
                    host=_loc(sites[0].path, sites[0].line),
                    shader=_loc(k.path, k.line),
                    kernel=k.name,
                    extra={"cfgs": sorted(c or "<none>" for c in cfgs)},
                )
            )
        else:
            findings.append(
                Finding(
                    "INFO",
                    "feature_gate",
                    f"kernel {k.name} host sites are cfg-gated {sorted(c or '<none>' for c in cfgs)}; "
                    f"reachable_when_on={on_reachable} genuinely_unreachable_when_off={not off_reachable}",
                    host=_loc(sites[0].path, sites[0].line),
                    shader=_loc(k.path, k.line),
                    kernel=k.name,
                )
            )

    # env-var name switch (decode family) is a runtime gate, not a cfg
    if pick_fns:
        findings.append(
            Finding(
                "INFO",
                "feature_gate",
                "HAWKING_DECODE_FAMILY is an env opt-out (default on) that picks family vs "
                "legacy kernel symbols; both names must exist. This is not a Cargo feature.",
                host="crates/hawking-core/src/decode_family.rs",
                extra={"pick_fns": sorted(pick_fns)},
            )
        )
        for fn, (a, b) in sorted(pick_fns.items()):
            for key in (a, b):
                name = decode_consts.get(key, key)
                if name not in metal_names:
                    findings.append(
                        Finding(
                            "ERROR",
                            "feature_gate",
                            f"decode_family::{fn}() can pick {name!r} which has no kernel void",
                            host="crates/hawking-core/src/decode_family.rs",
                            kernel=name,
                        )
                    )

    # ---- 7. control-path / queue identity ----
    check_library = library_membership is not None
    membership = library_membership or {}
    file_of_kernel = {k.name: Path(k.path).name for k in kernels}
    for d in all_dispatches:
        if not d.resolved:
            continue
        if not d.path.startswith(production_host_prefix):
            continue
        for kname in d.resolved:
            fn = file_of_kernel.get(kname)
            if not fn:
                continue
            mem = membership.get(fn)
            if mem == "tq" and d.feature_cfg != "tq":
                findings.append(
                    Finding(
                        "ERROR",
                        "control_path",
                        f"production host dispatches {kname} from {fn} which all_shader_sources "
                        f"only compiles under feature tq, but this site is not cfg-gated tq",
                        host=_loc(d.path, d.line),
                        shader=fn,
                        kernel=kname,
                    )
                )
            elif mem is None and check_library:
                # shader file is not in the MetalContext library at all
                findings.append(
                    Finding(
                        "ERROR",
                        "control_path",
                        f"production host dispatches {kname} from {fn} which is not in "
                        f"MetalContext::all_shader_sources; pipeline() would fail",
                        host=_loc(d.path, d.line),
                        shader=fn,
                        kernel=kname,
                    )
                )

    # queue identity (static)
    queue_identity = {
        "construction": (
            "MetalContext::new builds Device::system_default then "
            "device.new_command_queue(); one queue per context"
        ),
        "per_dispatch_command_buffer": (
            "MetalContext.dispatch_threads creates a new command buffer, encodes one "
            "kernel, commits, and waits"
        ),
        "fused_command_buffer": (
            "TokenCommandBuffer and dispatch_batch encode many kernels onto one "
            "command buffer before commit"
        ),
        "concurrent_group": (
            "dispatch_threads_in_concurrent_group shares one compute encoder"
        ),
        "ordered_encoder": (
            "enable_ordered_encoder fuses serial dispatches onto one encoder"
        ),
        "statically_determinable": (
            "queue construction and the four encode modes are statically visible. "
            "Which MetalContext instance a given call uses at runtime, and whether "
            "an ordered/concurrent encoder is currently open, is UNVERIFIABLE without execution."
        ),
        "gpu_authority": False,
    }

    # ---- coverage honesty ----
    n_pass_like = sum(1 for f in findings if f.severity == "INFO" and f.check in {"type_width", "host_shader_abi"})
    n_error = sum(1 for f in findings if f.severity == "ERROR")
    n_warn = sum(1 for f in findings if f.severity == "WARNING")
    n_uv = sum(1 for f in findings if f.severity == "UNVERIFIABLE")

    # Drop INFO spam: keep ABI/type INFO only as counts, not hundreds of rows,
    # except for tests which look at findings. We keep them; receipt builder trims.

    shaders_not_in_library = []
    if membership:
        on_disk = {Path(p).name for p in metal_files}
        shaders_not_in_library = sorted(on_disk - set(membership))

    return {
        "kernels": kernels,
        "dispatches": all_dispatches,
        "metal_structs": metal_structs,
        "rust_structs": rust_structs,
        "findings": findings,
        "referenced": referenced,
        "metal_names": metal_names,
        "generated_kernel_names": generated_names,
        "family_named": family_named,
        "binding_checked": binding_checked,
        "geometry_checked": geometry_checked,
        "uv_dispatch": uv_dispatch,
        "structs_paired": paired,
        "queue_identity": queue_identity,
        "library_membership": membership,
        "shaders_not_in_library": shaders_not_in_library,
        "includes_by_file": includes_by_file,
        "counts": {
            "ERROR": n_error,
            "WARNING": n_warn,
            "UNVERIFIABLE": n_uv,
            "INFO": n_pass_like,
        },
        "n_pass_like": n_pass_like,
    }


# ---------------------------------------------------------------------------
# Repo scan + receipt
# ---------------------------------------------------------------------------

_STATIC_CORRECTNESS_IS_NOT_SPEED = (
    "Static correctness does NOT prove speed and does NOT substitute for a "
    "protected measurement. A finding of no static ERROR is not a tps number, "
    "not a token_ns, not a complete-token, and not PROTECTED_ABSOLUTE. This "
    "tool exists to stop wasting a protected GPU window on a defect that was "
    "detectable from the .metal sources and their Rust hosts. It produces "
    "neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. Everything here is "
    "STATIC_ONLY with bench state UNKNOWN."
)


def load_repo_sources(
    repo: Path | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    root = Path(repo) if repo is not None else REPO
    metal: dict[str, str] = {}
    shaders = root / SHADER_DIR
    if shaders.is_dir():
        for p in sorted(shaders.glob("*.metal")):
            metal[str(p.relative_to(root))] = p.read_text(errors="replace")
    rust: dict[str, str] = {}
    host = root / HOST_ROOT
    if host.is_dir():
        for p in sorted(host.rglob("*.rs")):
            rust[str(p.relative_to(root))] = p.read_text(errors="replace")
    membership: dict[str, str] = {}
    modp = root / "crates/hawking-core/src/metal/mod.rs"
    if modp.is_file():
        membership = parse_library_membership(modp.read_text(errors="replace"))
    return metal, rust, membership


def _trim_findings(findings: list[Finding]) -> list[dict[str, Any]]:
    """Keep every ERROR/WARNING; cap UNVERIFIABLE/INFO with grouped samples."""
    out: list[dict[str, Any]] = []
    uv_by_check: dict[str, list[Finding]] = defaultdict(list)
    info: list[Finding] = []
    for f in findings:
        if f.severity in {"ERROR", "WARNING"}:
            out.append(f.as_dict())
        elif f.severity == "UNVERIFIABLE":
            uv_by_check[f.check].append(f)
        else:
            info.append(f)
    for check, rows in sorted(uv_by_check.items()):
        sample = [r.as_dict() for r in rows[:12]]
        out.append(
            {
                "severity": "UNVERIFIABLE",
                "check": check,
                "message": f"{len(rows)} UNVERIFIABLE site(s) for {check} (sample of {min(12, len(rows))})",
                "host": None,
                "shader": None,
                "kernel": None,
                "extra": {"count": len(rows), "sample": sample},
            }
        )
    # ABI/type INFO: keep paired struct matches (few) and drop per-dispatch spam
    struct_info = [f for f in info if f.check == "host_shader_abi" and f.kernel is None]
    other_info = [f for f in info if f.check in {"feature_gate"}]
    for f in struct_info + other_info:
        out.append(f.as_dict())
    info_counts = Counter(f.check for f in info)
    if info_counts:
        out.append(
            {
                "severity": "INFO",
                "check": "info_tally",
                "message": "INFO rows tallied rather than dumped (PASS-like rows are not promotions)",
                "host": None,
                "shader": None,
                "kernel": None,
                "extra": dict(info_counts),
            }
        )
    return out


def report_from_analyze(raw: dict[str, Any]) -> dict[str, Any]:
    kernels: list[MetalKernel] = raw["kernels"]
    dispatches: list[HostDispatch] = raw["dispatches"]
    findings: list[Finding] = raw["findings"]
    metal_names: set[str] = raw["metal_names"]
    referenced: dict[str, list[str]] = raw["referenced"]
    generated_names: dict[str, dict[str, str]] = raw["generated_kernel_names"]

    errors = [f for f in findings if f.severity == "ERROR"]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Deterministic host/shader ABI preflight. Reads .metal sources and Rust "
            "hosts. Emits STATIC_ONLY. Never a hardware measurement."
        ),
        "static_correctness_does_not_prove_speed": _STATIC_CORRECTNESS_IS_NOT_SPEED,
        "does_not_substitute_for_protected_measurement": True,
        "evidence_class": "STATIC_ONLY",
        "measurement_states_we_are_not": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "recovered_implementation": {
            "decode_family_kernel_name_tests": (
                "crates/hawking-core/src/decode_family.rs — unit tests that FAMILY_KERNELS "
                "and Q80_TILE_KERNELS appear as `kernel void {name}(` in gk_family.metal / "
                "q80_mixed_decode.metal. Name presence only; no buffer-index or ABI check."
            ),
            "metal_mod_trace_name_tests": (
                "crates/hawking-core/src/metal/mod.rs — static_kernel_name plus per-family "
                "`SHADER_X.contains(kernel void {name}()` tests. Compile/trace-name contract, "
                "not host bind vs [[buffer(N)]]."
            ),
            "argbuf_layout_comments": (
                "crates/hawking-core/src/metal/argbuf.rs — KernelArgBuffer packs U32/F32/U64 "
                "at natural alignment; the shader must declare a packed constant struct at "
                "the bound index. No automated checker compared the two."
            ),
            "megakernel_sizeof_guards": (
                "crates/hawking-core/src/kernels/megakernel.rs — const size_of::<MkLayerArgs>() "
                "== 120 and MkArgs == 20. Compile-time size, not field-order vs the .metal."
            ),
            "dispatch_ledger": (
                "tools/headless/dispatch_ledger.py (git, not this sparse checkout) — GPU "
                "dispatch-count ledger for a sealed 756-dispatch parent. Measurement, not ABI."
            ),
            "accelerator_geometry_tests": (
                "tools/accelerator/test_threadgroup_width.py, test_native_geometry.py, "
                "kernel_forge.py — arithmetic and GPU-backed geometry. Not a host/shader "
                "buffer-index preflight."
            ),
            "frontier_entry": (
                "receipts/future/CLAUDE_GLOBAL_FRONTIER.json F014 — 'No static kernel/ABI "
                "preflight independent of the GPU'. Probe *static_kernel_verify* was absent "
                "when the frontier was sealed."
            ),
            "flash_layer46_dispatch_ledger": (
                "receipts/headless/FLASH_LAYER46_DISPATCH_LEDGER.json was named in the lane "
                "contract. It is not on disk in this sparse checkout and is not in git HEAD "
                "(git cat-file / ls-tree miss). Not treated as evidence it never existed "
                "elsewhere; treated as ABSENT here."
            ),
            "adequate_duplicate": (
                "No existing module performed host set_buffer(N) vs shader [[buffer(N)]] "
                "comparison, struct field-order ABI, or feature-gate reachability as a "
                "STATIC_ONLY receipt. The name-presence tests above are consumed, not forked."
            ),
        },
        "gaps_closed": [
            "kernel existence: every resolved host kernel name must have kernel void in a .metal file",
            "binding count and buffer index, including the classic off-by-one",
            "type/width compatibility between host set_u32/set_f32/set_buffer and shader space",
            "threadgroup product against the Apple Silicon 1024 device ceiling",
            "repr(C) vs metal struct field order, size, and alignment where both exist",
            "KernelArgBuffer ArgLayout vs constant shader struct field widths",
            "feature-gate wiring for strand_bitslice (tq) and HAWKING_DECODE_FAMILY pick()",
            "control-path: production host vs MetalContext::all_shader_sources membership",
            "queue identity described statically (one queue per MetalContext; four encode modes)",
            "UNVERIFIABLE is a first-class result: a bind this checker cannot follow is never PASS",
        ],
        "negative_findings": [
            "No Metal runtime, no PSO, no maxTotalThreadsPerThreadgroup from the driver.",
            "FLASH_LAYER46_DISPATCH_LEDGER.json is absent from git HEAD and from this worktree.",
            "tools/accelerator/* and tools/headless/* are not materialized in this sparse checkout; recovered via git show / git ls-tree only.",
            "Bindings set through helpers in another crate, macros, or function pointers are UNVERIFIABLE.",
            "Grid covering a runtime problem size (rows, seq_len, hidden) is UNVERIFIABLE without those values.",
            "This sidecar produces no DIAGNOSTIC_RELATIVE and no PROTECTED_ABSOLUTE.",
        ],
        "coverage": {
            "metal_files": len({k.path for k in kernels}),
            "metal_kernels": len(metal_names),
            "metal_source_kernels": len(metal_names - set(generated_names)),
            "metal_generated_kernels": len(generated_names),
            "host_files_with_dispatch": len({d.path for d in dispatches}),
            "host_dispatches": len(dispatches),
            "dispatches_resolved": sum(1 for d in dispatches if d.resolved),
            "dispatches_unverifiable_name": sum(1 for d in dispatches if not d.resolved),
            "binding_pairs_with_matching_index_sets": raw["binding_checked"],
            "threadgroup_triples_resolved": raw["geometry_checked"],
            "structs_paired": raw["structs_paired"],
            "referenced_kernel_names": len(referenced),
            "unreferenced_kernel_names": len(metal_names - set(referenced)),
            "honesty": (
                "PASS-like INFO is only recorded when the checker followed both sides to a "
                "concrete index or layout. A bind it could not follow is UNVERIFIABLE, never PASS. "
                "Matching indices are not a speed claim."
            ),
            "shaders_not_in_metalcontext_library": raw["shaders_not_in_library"],
        },
        "counts": {
            "ERROR": raw["counts"]["ERROR"],
            "WARNING": raw["counts"]["WARNING"],
            "UNVERIFIABLE": raw["counts"]["UNVERIFIABLE"],
            "INFO_pass_like": raw["counts"]["INFO"],
        },
        "would_waste_a_protected_window": bool(errors),
        "blocking_defect_count": len(errors),
        "findings": _trim_findings(findings),
        "queue_identity": raw["queue_identity"],
        "library_membership_files": sorted(raw["library_membership"].items()),
        "decode_family_named_kernels": raw["family_named"],
        "generated_kernel_names": generated_names,
        "apple_static_limits": {
            "max_threads_per_threadgroup": APPLE_MAX_THREADS_PER_THREADGROUP,
            "simdgroup_width": APPLE_SIMDGROUP_WIDTH,
            "note": (
                "Device hard ceiling, not a measured PSO limit. A kernel may refuse a "
                "legal-looking (256) threadgroup at pipeline creation; that is runtime."
            ),
        },
    }


def scan(repo: Path | None = None) -> dict[str, Any]:
    root = Path(repo) if repo is not None else REPO
    rust_doc = _scan_via_rust(root)
    if rust_doc is not None:
        return rust_doc
    metal, rust, membership = load_repo_sources(repo)
    raw = analyze(metal, rust, library_membership=membership)
    return report_from_analyze(raw)


def _rust_binary() -> Path | None:
    """Locate hawking-static-kernel-verify. Missing/unexecutable → None (Python path)."""
    if _os.environ.get("HAWKING_SKV_FORCE_PYTHON", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    env = _os.environ.get("HAWKING_STATIC_KERNEL_VERIFY")
    if env is not None:
        env = env.strip()
        if env.lower() in {"", "0", "off", "none", "false"}:
            return None
        p = Path(env)
        return p if p.is_file() and _os.access(p, _os.X_OK) else None
    roots: list[Path] = []
    td = _os.environ.get("CARGO_TARGET_DIR")
    if td:
        roots.append(Path(td))
    roots.extend(
        [
            REPO / "workspace" / "ops" / "build" / "rust",
            REPO / "target",
        ]
    )
    names = (
        "release/hawking-static-kernel-verify",
        "release-fast/hawking-static-kernel-verify",
        "debug/hawking-static-kernel-verify",
    )
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file() and _os.access(p, _os.X_OK):
                return p
    return None


def _scan_via_rust(root: Path) -> dict[str, Any] | None:
    """Run the Rust preflight. Any failure returns None so Python remains the default."""
    binary = _rust_binary()
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [str(binary), "--repo", str(root), "--json"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    stdout = proc.stdout.strip()
    if not stdout:
        return None
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return None
    return doc


def build() -> Path:
    doc = scan()
    return write_receipt(RECEIPT, doc, "tools/future/static_kernel_verify.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="analyze and write the sealed receipt")
    ap.add_argument("--build", action="store_true", help="alias of --scan")
    a = ap.parse_args()
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(
        "STATIC_KERNEL_PREFLIGHT",
        f"kernels={doc['coverage']['metal_kernels']}",
        f"dispatches={doc['coverage']['host_dispatches']}",
        f"ERROR={doc['counts']['ERROR']}",
        f"WARNING={doc['counts']['WARNING']}",
        f"UNVERIFIABLE={doc['counts']['UNVERIFIABLE']}",
        f"would_waste_a_protected_window={doc['would_waste_a_protected_window']}",
    )
    print("static_correctness_does_not_prove_speed: YES — see receipt top-level field")
    _ = a
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
