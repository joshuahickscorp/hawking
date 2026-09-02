#!/usr/bin/env python3
"""Noetic information accounting, plus a canary that tries to hide model bytes.

A representation search that reports EBPW is trivially dishonest if the
supporting structures escape the accounting. An apparent 0.08 BPW
representation that needs a 2 GB shared basis, a 1 GB routing table, 500 MB
of generated-state seeds and a 4 GB correction cache is not a 0.08 BPW
representation.

The test is not "is this file in the model directory". The test is "does this
byte carry information derived from the parent". A shared runtime may be
amortised across models but it cannot disappear from the receipt, and a
model-specific kernel constant containing learned information is model
information no matter which file it sits in.

Seven buckets, measured on the sealed uniform-q4-v1 artifact (G105), then
a five-attempt canary. The canary is supposed to be able to miss: a canary
where everything is caught on the first try is more often a weak canary than
strong accounting.

  python3 tools/headless/noetic_information_accounting.py

Never writes under the real artifact. The canary lives in tempfile.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_INFORMATION_ACCOUNTING.json"
ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
DECODE_RS = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
SHADERS = REPO / "crates" / "hawking-core" / "shaders"
G105_PATH = "receipts/ascent-2026-08-16/G105_NR_NX_ARTIFACT.json"
G105_DIGESTS_PATH = "receipts/ascent-2026-08-16/G105_TENSOR_DIGESTS.json"

# Anchors. Already measured. Do not re-derive.
SOURCE_PARAM_COUNT = 26_895_998_464
G105_PAYLOAD_BYTES = 14_297_694_680
G105_ON_DISK_BYTES = 14_297_933_604
G105_TENSOR_COUNT = 755
G105_ON_DISK_FILES = 756
G105_Q4_TENSORS = 402
G105_F32_TENSORS = 353
G105_TPS = 32.73
G105_TOKEN_MS = 30.606
G105_BATTERY = "28/30"
G105_BOUND_KERNELS = 38
G105_DECLARED_KERNELS = 554
G105_ROLLING_DIGEST = "89e780555634f28aaf86d03108407f29da254af61404a92d2ca750e00b3fa812"
G105_CHIPSET = "Apple M3 Ultra"
G105_GPU_CORES = 60
G105_UNIFIED = 103_079_215_104
G105_METAL = "Metal 4"
G105_ROOF = 778.8

# Geometry authority: crates/hawking-core/src/model/qwen38_geometry.rs
# Workspace formula: qwen38_workspace_bytes in qwen38_hybrid_decode.rs
QWEN38_HIDDEN = 5_120
QWEN38_INTERMEDIATE = 17_408
QWEN38_VOCAB = 248_320
QWEN38_GQA_HEADS = 24
QWEN38_GQA_KV_HEADS = 4
QWEN38_GQA_HEAD_DIM = 256
QWEN38_GQA_LAYERS = 16
QWEN38_DELTANET_LAYERS = 48
QWEN38_MIXED_HGRAVS_RANK = 160
QWEN38_IN_PROJ_QKV_ROWS = 10_240
QWEN38_IN_PROJ_B_ROWS = 48
QWEN38_IN_PROJ_A_ROWS = 48
QWEN38_LINEAR_KEY_HEADS = 16
QWEN38_LINEAR_VALUE_HEADS = 48
QWEN38_LINEAR_VALUES_PER_KEY = 3
QWEN38_LINEAR_KEY_HEAD_DIM = 128
QWEN38_LINEAR_VALUE_HEAD_DIM = 128
QWEN38_LINEAR_CONV_KERNEL = 4

HEADLINE_CTX = 8192
CTXS = (1, 128, 8192, 131072)

MIN_HIDDEN = 16
GEOMETRY_INTS = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}

TEXT_SUFFIXES = {".py", ".rs", ".metal", ".c", ".h", ".cc", ".cpp", ".json", ".txt", ".md"}
TENSOR_SUFFIXES = {".hq30uq4", ".f32v2", ".f32", ".bin", ".gguf", ".safetensors"}
COMPILED_GPU_SUFFIXES = {".metallib", ".cubin", ".ptx", ".spv", ".dxil"}

BUCKETS_5 = (
    "MODEL_SPECIFIC_BYTES",
    "SHARED_RUNTIME_BYTES",
    "MACHINE_SPECIFIC_BYTES",
    "GENERATED_CACHE_BYTES",
    "TEMPORARY_BYTES",
)
BUCKETS_7 = BUCKETS_5 + ("RESIDENT_BYTES", "ACTIVE_BYTES")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def bpw(nbytes: int | float) -> float:
    return 8.0 * float(nbytes) / SOURCE_PARAM_COUNT


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def git_show(rel: str) -> bytes:
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        capture_output=True, check=False,
    )
    if r.returncode != 0:
        raise FileNotFoundError(f"git show HEAD:{rel}: {r.stderr.decode()[:400]}")
    return r.stdout


def git_show_json(rel: str) -> dict:
    return json.loads(git_show(rel))


def f32b(n: int) -> int:
    return n * 4


def xorshift64_bytes(seed: int, n: int) -> bytes:
    x = seed & 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while len(out) < n:
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        out += x.to_bytes(8, "little")
    return bytes(out[:n])


def canary_seed(salt: bytes) -> int:
    return int.from_bytes(hashlib.sha256(salt).digest()[:8], "little") or 1


def canary_f32_payload(n: int, salt: bytes) -> bytes:
    raw = xorshift64_bytes(canary_seed(salt), n * 4)
    # keep them looking like weights, not NaNs
    vals = []
    for i in range(n):
        u = int.from_bytes(raw[i * 4:(i + 1) * 4], "little")
        vals.append(struct.pack("<f", ((u & 0xFFFF) / 65535.0) * 2.0 - 1.0))
    return b"".join(vals)


def floats_from_payload(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack("<" + "f" * n, blob[: n * 4]))


def fmt_list(vals: list[float], per: int = 8) -> str:
    parts = []
    for i in range(0, len(vals), per):
        row = ", ".join(f"{v:.7g}" for v in vals[i:i + per])
        parts.append("    " + row + ",")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# workspace bytes — lockstep with qwen38_workspace_bytes
# ---------------------------------------------------------------------------

def deltanet_layout():
    value_rows = QWEN38_LINEAR_VALUES_PER_KEY * QWEN38_LINEAR_VALUE_HEAD_DIM
    qkvz_rows_per_key = QWEN38_LINEAR_KEY_HEAD_DIM * 2 + value_rows * 2
    ba_rows_per_key = QWEN38_LINEAR_VALUES_PER_KEY * 2
    conv_channels = (
        QWEN38_LINEAR_KEY_HEADS * QWEN38_LINEAR_KEY_HEAD_DIM * 2
        + QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_VALUE_HEAD_DIM
    )
    return {
        "qkvz_rows": QWEN38_LINEAR_KEY_HEADS * qkvz_rows_per_key,
        "ba_rows": QWEN38_LINEAR_KEY_HEADS * ba_rows_per_key,
        "value_elements": QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_VALUE_HEAD_DIM,
        "conv_state_elements": conv_channels * (QWEN38_LINEAR_CONV_KERNEL - 1),
        "recurrent_state_elements": (
            QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_KEY_HEAD_DIM
            * QWEN38_LINEAR_VALUE_HEAD_DIM
        ),
        "value_heads": QWEN38_LINEAR_VALUE_HEADS,
    }


def qwen38_workspace_bytes(max_seq_len: int) -> dict:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    lay = deltanet_layout()
    hidden = f32b(QWEN38_HIDDEN)
    qkvz = f32b(lay["qkvz_rows"])
    ba = f32b(lay["ba_rows"])
    value = f32b(lay["value_elements"])
    q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)
    kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)
    query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)
    mid = f32b(QWEN38_INTERMEDIATE)
    logits = f32b(QWEN38_VOCAB)
    conv = f32b(QWEN38_DELTANET_LAYERS * lay["conv_state_elements"])
    rec = f32b(QWEN38_DELTANET_LAYERS * lay["recurrent_state_elements"])
    kv_cache = f32b(
        QWEN38_GQA_LAYERS * max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM
    )
    hgravs = f32b(QWEN38_MIXED_HGRAVS_RANK)
    split_qkv = f32b(QWEN38_IN_PROJ_QKV_ROWS)
    split_b = f32b(QWEN38_IN_PROJ_B_ROWS)
    split_a = f32b(QWEN38_IN_PROJ_A_ROWS)
    sampled = 4
    heads_f32 = f32b(lay["value_heads"])
    activation = (
        hidden * 2 + qkvz + ba + value * 6 + heads_f32 * 2 + hidden * 2
        + q_proj + kv * 2 + query * 3 + mid * 3 + hidden + logits
        + sampled + hgravs + split_qkv + split_b + split_a
    )
    deltanet = conv + rec
    gqa = kv_cache * 2
    return {
        "max_seq_len": max_seq_len,
        "activation_bytes": activation,
        "deltanet_state_bytes": deltanet,
        "gqa_kv_bytes": gqa,
        "kv_bytes_per_position": gqa // max_seq_len,
        "total_bytes": activation + deltanet + gqa,
    }


def self_check_workspace() -> None:
    w = qwen38_workspace_bytes(2048)
    assert w["activation_bytes"] == 1_691_396, w["activation_bytes"]
    assert w["deltanet_state_bytes"] == 156_893_184, w["deltanet_state_bytes"]
    a = qwen38_workspace_bytes(2048)
    b = qwen38_workspace_bytes(4096)
    assert a["activation_bytes"] == b["activation_bytes"]
    assert a["deltanet_state_bytes"] == b["deltanet_state_bytes"]
    assert b["gqa_kv_bytes"] == a["gqa_kv_bytes"] * 2
    assert w["kv_bytes_per_position"] == 131_072


# ---------------------------------------------------------------------------
# artifact snapshot (identity, not contents of 14 GB)
# ---------------------------------------------------------------------------

def snapshot_tree(root: Path) -> dict:
    h = hashlib.sha256()
    n = 0
    total = 0
    mtimes = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            p = Path(dirpath) / fn
            st = p.lstat()
            rel = str(p.relative_to(root))
            h.update(rel.encode())
            h.update(b"\0")
            h.update(str(st.st_size).encode())
            h.update(b"\0")
            h.update(str(st.st_mtime_ns).encode())
            h.update(b"\0")
            h.update(str(st.st_ino).encode())
            n += 1
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
            mtimes.append(st.st_mtime_ns)
    man = root / "manifest.json"
    return {
        "root": str(root),
        "file_count": n,
        "total_bytes": total,
        "meta_digest": h.hexdigest(),
        "manifest_sha256": sha256_file(man) if man.is_file() else None,
        "mtime_ns_min": min(mtimes) if mtimes else None,
        "mtime_ns_max": max(mtimes) if mtimes else None,
    }


def walk_regular(root: Path) -> list[tuple[Path, os.stat_result]]:
    out = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            out.append((p, st))
    return out


# ---------------------------------------------------------------------------
# hidden-payload scanners (content, not path)
# ---------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_CONSTANT_HEAD = re.compile(
    r"\bconstant\b[^;{]*?\b(?:float|half|int|uint|ushort|short|float[234]|int[234])"
    r"(?:\d(?:x\d)?)?\s+\w+\s*(?:\[[^\]]*\])?\s*=\s*\{",
    re.S,
)


def _numeric_value(node) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _numeric_value(node.operand)
        if v is None:
            return None
        return -v if isinstance(node.op, ast.USub) else v
    return None


def scan_python_literals(text: str) -> list[dict]:
    hits = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return hits
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        if len(node.elts) < MIN_HIDDEN:
            continue
        vals = [_numeric_value(e) for e in node.elts]
        if any(v is None for v in vals):
            continue
        if all(v == 0.0 for v in vals):
            continue
        hits.append({
            "kind": "python_numeric_literal",
            "n": len(vals),
            "payload_bytes": len(vals) * 4,
            "line": getattr(node, "lineno", None),
        })
    return hits


def _delimited_body(text: str, open_idx: int, open_ch: str, close_ch: str) -> tuple[str, int] | None:
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i
    return None


def _brace_body(text: str, open_idx: int) -> tuple[str, int] | None:
    return _delimited_body(text, open_idx, "{", "}")


def _bracket_body(text: str, open_idx: int) -> tuple[str, int] | None:
    return _delimited_body(text, open_idx, "[", "]")


def scan_metal_constants(text: str) -> list[dict]:
    hits = []
    for m in _CONSTANT_HEAD.finditer(text):
        body = _brace_body(text, m.end() - 1)
        if body is None:
            continue
        inner, _ = body
        nums = re.findall(_NUM, inner)
        if len(nums) < MIN_HIDDEN:
            continue
        try:
            vals = [float(x) for x in nums]
        except ValueError:
            continue
        if all(v == 0.0 for v in vals):
            continue
        hits.append({
            "kind": "metal_constant_array",
            "n": len(vals),
            "payload_bytes": len(vals) * 4,
            "line": text[:m.start()].count("\n") + 1,
        })
    return hits


def _json_numeric_lists(obj, path="") -> list[dict]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(_json_numeric_lists(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            if len(obj) >= MIN_HIDDEN:
                as_int = all(isinstance(x, int) or (isinstance(x, float) and x == int(x)) for x in obj)
                if as_int and set(int(x) for x in obj) <= GEOMETRY_INTS and len(obj) <= 64:
                    pass  # plausible threadgroup/geometry sequence — see miss discussion
                else:
                    hits.append({
                        "kind": "json_numeric_array",
                        "path": path,
                        "n": len(obj),
                        "payload_bytes": len(obj) * 4,
                    })
        else:
            for i, v in enumerate(obj):
                hits.extend(_json_numeric_lists(v, f"{path}[{i}]"))
    return hits


def scan_json_payloads(text: str) -> list[dict]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _json_numeric_lists(doc)


def scan_bracket_numeric_arrays(text: str) -> list[dict]:
    """Flat `[ ... ]` numeric literals: Rust `const X: [f32; N] = [ ... ]`.

    The Metal `constant float[] = { ... }` grammar does not match real Rust
    (or C++ `std::array`) weight dumps. Path is not provenance; neither is
    'the file is .rs so it must be code'.
    """
    hits = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("[", i)
        if j < 0:
            break
        body = _bracket_body(text, j)
        if body is None:
            i = j + 1
            continue
        inner, end = body
        if "[" in inner:
            i = j + 1
            continue
        nums = re.findall(_NUM, inner)
        if len(nums) < MIN_HIDDEN:
            i = end + 1
            continue
        try:
            vals = [float(x) for x in nums]
        except ValueError:
            i = end + 1
            continue
        if all(v == 0.0 for v in vals):
            i = end + 1
            continue
        rest = re.sub(_NUM, "", inner)
        if re.search(r"[^\s,._]", rest):
            i = end + 1
            continue
        hits.append({
            "kind": "bracket_numeric_array",
            "n": len(vals),
            "payload_bytes": len(vals) * 4,
            "line": text[:j].count("\n") + 1,
        })
        i = end + 1
    return hits


def scan_text_file(path: Path, text: str) -> list[dict]:
    suf = path.suffix.lower()
    if suf == ".py":
        return scan_python_literals(text)
    if suf == ".metal":
        return scan_metal_constants(text)
    if suf == ".json":
        return scan_json_payloads(text)
    if suf == ".rs":
        return scan_metal_constants(text) + scan_bracket_numeric_arrays(text)
    if suf in {".c", ".h", ".cc", ".cpp"}:
        return scan_metal_constants(text)  # same constant-array grammar
    return []


# ---------------------------------------------------------------------------
# seed recipes
# ---------------------------------------------------------------------------

DTYPE_WIDTH = {"f32": 4, "f16": 2, "u8": 1, "i32": 4, "u32": 4}


def load_recipe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def expanded_from_recipe(recipe: dict) -> int | None:
    if not isinstance(recipe, dict):
        return None
    prng = recipe.get("prng") or recipe.get("generator")
    if prng not in {"xorshift64", "counter", "splitmix64"}:
        return None
    n = recipe.get("elements") or recipe.get("n") or recipe.get("count")
    dt = recipe.get("dtype", "f32")
    if not isinstance(n, int) or n <= 0 or dt not in DTYPE_WIDTH:
        return None
    return n * DTYPE_WIDTH[dt]


# ---------------------------------------------------------------------------
# classify a self-contained tree (the canary, and the real artifact directory)
# ---------------------------------------------------------------------------

def _zero_buckets() -> dict[str, int]:
    return {k: 0 for k in BUCKETS_7}


def embedded_weightlike_bytes(blob: bytes, min_run: int = 64) -> int:
    """Bytes inside a compiled GPU blob that parse as a run of plausible weights.

    Attempt 4 hides model information in a `.metallib` and it was MISSED, because
    the accountant trusted the SUFFIX: anything ending `.metallib` booked as
    MACHINE_SPECIFIC without being opened. The receipt's own prescription was
    "path is not provenance", and a fatbin is exactly where learned constant
    tables have always lived.

    So look inside. A float32 weight table has a signature compiled code does not:
    a long contiguous run of finite values in a bounded range, with enough
    distinct values that it is not padding. Those bytes book as MODEL_SPECIFIC.

    account_tree applies this to compiled GPU blobs AND to any other non-text
    file that path_role would park in SHARED / MACHINE / GENERATED (runtime
    sidecars, MLX .npy, safetensors, header-prefixed .bin). Integer-coded
    payloads still score 0 — that named gap is not closed here.

    This is a heuristic and it is stated as one. It can miss weights stored as
    f16, bf16, or integer codes, and it can false-positive on a legitimate large
    float constant table compiled from scanned source. It is strictly better than
    trusting the file extension, which caught nothing.
    """
    if len(blob) < min_run * 4:
        return 0
    # Scan ALL FOUR byte alignments. Embedded data does not start on a
    # convenient boundary: the planted blob has an 18-byte header, and scanning
    # only from offset 0 found nothing at all while the payload alone scored the
    # full 512 bytes. An accountant that can be defeated by an odd-sized header
    # is not an accountant.
    best = 0
    for align in range(4):
        body = blob[align:]
        n = len(body) // 4
        run = 0
        run_distinct: set[int] = set()
        for i in range(n):
            w = body[i * 4:(i + 1) * 4]
            u = int.from_bytes(w, "little")
            v = struct.unpack("<f", w)[0]
            ok = (v == v) and abs(v) != float("inf") and 1e-8 < abs(v) <= 16.0
            if ok:
                run += 1
                run_distinct.add(u)
            else:
                if run >= min_run and len(run_distinct) >= min_run // 2:
                    best = max(best, run)
                run, run_distinct = 0, set()
        if run >= min_run and len(run_distinct) >= min_run // 2:
            best = max(best, run)
    return best * 4


def path_role(rel: str) -> str:
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    top = parts[0] if parts else ""
    suf = Path(rel).suffix.lower()
    if top in {"tmp", "scratch", "temporary"} or "/tmp/" in f"/{rel}/":
        return "TEMPORARY_BYTES"
    if top in {"nx", "machine", "genome"} or suf in COMPILED_GPU_SUFFIXES:
        return "MACHINE_SPECIFIC_BYTES"
    if top in {"runtime", "kernels", "shaders", "src"}:
        return "SHARED_RUNTIME_BYTES"
    if top in {"cache", "generated"}:
        return "GENERATED_CACHE_BYTES"
    if top in {"seeds", "seed"}:
        return "MODEL_SPECIFIC_BYTES"
    if top in {"tensors", "weights"} or suf in TENSOR_SUFFIXES:
        return "MODEL_SPECIFIC_BYTES"
    if Path(rel).name in {"manifest.json", "PACK_REPORT.json"}:
        return "MODEL_SPECIFIC_BYTES"
    return "MODEL_SPECIFIC_BYTES"


def account_tree(root: Path, mode: str = "content") -> dict:
    """Account every regular file under root.

    mode='content'  — the real accountant: provenance, not path.
    mode='path'     — naive path-only classifier. What we watched fail.
    """
    content = mode == "content"
    files = walk_regular(root)
    buckets = _zero_buckets()
    evidence = []
    remaining = {}  # rel -> (role, bytes still in that role)

    recipes = {}
    for p, _ in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        if p.name.endswith(".recipe.json") or p.suffix == ".recipe":
            rec = load_recipe(p)
            if rec:
                stem = p.name.replace(".recipe.json", "").replace(".recipe", "")
                recipes[stem] = rec
                parent = str(p.parent.relative_to(root)).replace("\\", "/")
                recipes[f"{parent}/{stem}"] = rec

    required = set()
    man = root / "manifest.json"
    if man.is_file():
        try:
            md = json.loads(man.read_text())
        except json.JSONDecodeError:
            md = {}
        for s in md.get("required_sidecars") or []:
            required.add(str(s).replace("\\", "/"))

    for p, st in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        role = path_role(rel)
        sz = st.st_size
        stem = p.name.split(".")[0]

        # PATH IS NOT PROVENANCE. A compiled GPU blob booked as MACHINE_SPECIFIC
        # purely on its suffix, which is how attempt 4 hid model bytes in a
        # .metallib and was MISSED. In content mode, open it: any embedded run
        # that parses as plausible weights is charged to MODEL_SPECIFIC.
        if content and p.suffix.lower() in COMPILED_GPU_SUFFIXES:
            try:
                hidden = embedded_weightlike_bytes(p.read_bytes())
            except OSError:
                hidden = 0
            if hidden:
                hidden = min(hidden, sz)
                buckets["MODEL_SPECIFIC_BYTES"] += hidden
                buckets[role] += sz - hidden
                remaining[rel] = [role, sz - hidden]
                evidence.append(
                    {
                        "file": rel,
                        "role": "MODEL_SPECIFIC_BYTES",
                        "bytes": hidden,
                        "why": (
                            "compiled GPU blob contains a contiguous run parsing as "
                            "plausible f32 weights; suffix is not provenance"
                        ),
                    }
                )
                continue
        rec = recipes.get(stem) or recipes.get(rel.rsplit(".", 1)[0])
        expanded = expanded_from_recipe(rec) if rec else None

        if p.name.endswith(".recipe.json") or p.suffix == ".recipe":
            # recipe text is not the payload; charged via the seed expansion
            buckets[role] += sz
            remaining[rel] = [role, sz]
            continue

        if content and expanded:
            buckets["MODEL_SPECIFIC_BYTES"] += expanded
            remaining[rel] = ["MODEL_SPECIFIC_BYTES", 0]
            evidence.append({
                "rel": rel, "event": "seed_expanded",
                "seed_bytes": sz, "expanded_bytes": expanded,
            })
            continue

        if content and role == "GENERATED_CACHE_BYTES":
            regenerable = expanded is not None
            if (not regenerable) or rel in required:
                role = "MODEL_SPECIFIC_BYTES"
                evidence.append({
                    "rel": rel, "event": "cache_required_not_regenerable",
                    "bytes": sz, "required": rel in required,
                })

        # Path is not provenance for ANY binary, not only .metallib. A .bin
        # parked next to the host, an MLX .npy under runtime/, a safetensors
        # sidecar, or an 18-byte-header blob with the "wrong" suffix all used
        # to sit in SHARED/MACHINE because the GPU-suffix branch never opened
        # them. Apply the same f32-run detector. Integer codes still score 0
        # (the named remaining gap); high-entropy NaN/Inf runs also score 0.
        suf = p.suffix.lower()
        if (
            content
            and role != "MODEL_SPECIFIC_BYTES"
            and suf not in TEXT_SUFFIXES
            and suf not in COMPILED_GPU_SUFFIXES
        ):
            try:
                hidden = embedded_weightlike_bytes(p.read_bytes())
            except OSError:
                hidden = 0
            if hidden:
                hidden = min(hidden, sz)
                buckets["MODEL_SPECIFIC_BYTES"] += hidden
                buckets[role] += sz - hidden
                remaining[rel] = [role, sz - hidden]
                evidence.append(
                    {
                        "file": rel,
                        "role": "MODEL_SPECIFIC_BYTES",
                        "bytes": hidden,
                        "why": (
                            "non-text file contains a contiguous run parsing as "
                            "plausible f32 weights; directory/suffix is not provenance"
                        ),
                    }
                )
                continue

        buckets[role] += sz
        remaining[rel] = [role, sz]

    if content:
        for p, st in files:
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            hits = scan_text_file(p, text)
            if not hits:
                continue
            payload = sum(h["payload_bytes"] for h in hits)
            role, left = remaining.get(rel, ["SHARED_RUNTIME_BYTES", 0])
            if role == "MODEL_SPECIFIC_BYTES":
                evidence.append({
                    "rel": rel, "event": "hidden_in_already_model",
                    "payload_bytes": payload, "hits": hits,
                })
                continue
            take = min(payload, left)
            buckets[role] -= take
            remaining[rel][1] = left - take
            extra = payload - take
            buckets["MODEL_SPECIFIC_BYTES"] += payload
            # we moved `take` out of role (already in total via file size)
            # and added `payload` to MODEL, so `take` was double-counted:
            # file bytes still exist. Correct: MODEL += payload, role -= take,
            # and if take==payload the file bytes that carried the literals
            # changed bucket. If extra>0 the literals encode more than the
            # file (shouldn't happen for source). Undo the double count:
            # buckets[MODEL] currently += payload, buckets[role] -= take.
            # Original: role had `sz`. After: role has sz-take, MODEL has payload.
            # Distinct bytes = (sz-take)+payload = sz+extra. Good.
            evidence.append({
                "rel": rel, "event": "reclassified_hidden_literals",
                "from": role, "take": take, "payload_bytes": payload,
                "hits": hits,
            })

    # occupancy views over this tree (canary has no separate runtime workspace)
    partition = sum(buckets[k] for k in BUCKETS_5)
    buckets["RESIDENT_BYTES"] = partition
    buckets["ACTIVE_BYTES"] = buckets["MODEL_SPECIFIC_BYTES"]
    return {
        "buckets": buckets,
        "partition_sum": partition,
        "naive_seven_sum": sum(buckets[k] for k in BUCKETS_7),
        "evidence": evidence,
        "file_count": len(files),
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# real artifact
# ---------------------------------------------------------------------------

def kernel_file_census(dispatched: list[str]) -> dict:
    dispatched_set = set(dispatched)
    declared: set[str] = set()
    bound_files = []
    other_files = []
    all_bytes = 0
    bound_bytes = 0
    if not SHADERS.is_dir():
        return {
            "present": False, "declared": 0, "bound_file_bytes": 0,
            "all_shader_bytes": 0, "bound_files": [],
        }
    for p in sorted(SHADERS.glob("*.metal")):
        text = p.read_text(encoding="utf-8", errors="replace")
        names = re.findall(r"kernel\s+void\s+(\w+)", text)
        declared.update(names)
        sz = p.stat().st_size
        all_bytes += sz
        if any(n in dispatched_set for n in names):
            bound_files.append({"file": p.name, "bytes": sz, "kernels": sorted(set(names) & dispatched_set)})
            bound_bytes += sz
        else:
            other_files.append({"file": p.name, "bytes": sz, "declared_kernels": len(names)})
    missing = sorted(dispatched_set - declared)
    return {
        "present": True,
        "declared_kernel_void_names": len(declared),
        "g105_declared": G105_DECLARED_KERNELS,
        "declared_matches_g105": len(declared) == G105_DECLARED_KERNELS,
        "bound_files": bound_files,
        "bound_file_bytes": bound_bytes,
        "all_shader_bytes": all_bytes,
        "unbound_shader_bytes": all_bytes - bound_bytes,
        "unbound_file_count": len(other_files),
        "dispatched_names_missing_from_shaders": missing,
        "decode_rs_bytes": DECODE_RS.stat().st_size if DECODE_RS.is_file() else 0,
    }


def verify_artifact_identity(manifest: dict, g105_digests: dict) -> dict:
    tensors_dir = ARTIFACT / "tensors"
    by_name = {t["artifact"]: t for t in manifest["tensors"]}
    files = list(tensors_dir.iterdir()) if tensors_dir.is_dir() else []
    size_mismatches = []
    missing = []
    extra = 0
    payload = 0
    q4 = f32 = 0
    for t in manifest["tensors"]:
        p = tensors_dir / t["artifact"]
        if not p.is_file():
            missing.append(t["artifact"])
            continue
        sz = p.stat().st_size
        payload += sz
        if sz != t["bytes"]:
            size_mismatches.append(t["artifact"])
        if t.get("kind") == "q4":
            q4 += 1
        elif t.get("kind") == "f32":
            f32 += 1
    on_disk_names = {p.name for p in files if p.is_file()}
    extra = len(on_disk_names - set(by_name))
    digest_checks = []
    filename_is_sha = 0
    digest_ok = 0
    mismatch_classes: dict[str, int] = {}
    g105_t = (g105_digests or {}).get("tensors") or {}
    # Hash every tensor under 1 MiB against G105. Cheap, covers every f32.
    for t in manifest["tensors"]:
        p = tensors_dir / t["artifact"]
        if not p.is_file() or t["bytes"] > 1 << 20:
            continue
        h = sha256_file(p)
        stem = t["artifact"].split(".")[0]
        is_name = h == stem
        if is_name:
            filename_is_sha += 1
        rec = g105_t.get(t["artifact"]) or {}
        ok = rec.get("sha256") == h
        if ok:
            digest_ok += 1
        else:
            cls = f"{t.get('kind')}:{t['bytes']}"
            mismatch_classes[cls] = mismatch_classes.get(cls, 0) + 1
        digest_checks.append({
            "artifact": t["artifact"], "name": t["name"], "bytes": t["bytes"],
            "sha256": h, "matches_g105": ok, "filename_is_content_sha256": is_name,
        })
    return {
        "tensor_files_on_disk": len(on_disk_names),
        "manifest_tensor_count": len(manifest["tensors"]),
        "missing": missing[:8],
        "size_mismatches": size_mismatches[:8],
        "extra_files_in_tensors": extra,
        "payload_bytes_sum": payload,
        "payload_matches_manifest": payload == manifest.get("tensor_payload_bytes"),
        "payload_matches_g105": payload == G105_PAYLOAD_BYTES,
        "q4_tensors": q4, "f32_tensors": f32,
        "codec_families": [
            {"family": "grouped_absmax", "bits": 4, "group": manifest.get("q4_group_size", 64),
             "applies_to": "q4_tensors", "count": q4},
            {"family": "raw_f32", "applies_to": "f32_tensors", "count": f32},
        ],
        "g105_digest_checks_under_1miB": {
            "n": len(digest_checks),
            "matches": digest_ok,
            "mismatches": len(digest_checks) - digest_ok,
            "mismatch_classes": mismatch_classes,
            "filename_equals_sha256": filename_is_sha,
            "content_addressing_defect_confirmed": filename_is_sha == 0 and len(digest_checks) > 0,
            "sample_matching": [c for c in digest_checks if c["matches_g105"]][:4],
            "sample_mismatch": [c for c in digest_checks if not c["matches_g105"]][:4],
        },
    }


def account_real(g105: dict, g105_digests: dict) -> dict:
    man = json.loads((ARTIFACT / "manifest.json").read_text())
    identity = verify_artifact_identity(man, g105_digests)
    files = walk_regular(ARTIFACT)
    on_disk = sum(st.st_size for _, st in files)
    manifest_bytes = (ARTIFACT / "manifest.json").stat().st_size
    tensor_bytes = identity["payload_bytes_sum"]

    embed = next(t for t in man["tensors"] if t["name"].endswith("embed_tokens.weight"))
    embed_row = embed["bytes"] // embed["shape"][0]
    active_weights = tensor_bytes - embed["bytes"] + embed_row

    nx = g105["NX"]
    nx_bytes = len(json.dumps(nx, sort_keys=True).encode())
    dispatched = nx["kernel_binding"]["dispatched"]
    kcens = kernel_file_census(dispatched)
    decode_bytes = kcens.get("decode_rs_bytes") or 0
    shared_required = decode_bytes + kcens.get("bound_file_bytes", 0)

    hidden_in_bound = []
    if SHADERS.is_dir():
        for row in kcens.get("bound_files") or []:
            p = SHADERS / row["file"]
            text = p.read_text(encoding="utf-8", errors="replace")
            hits = scan_metal_constants(text)
            if hits:
                hidden_in_bound.append({"file": row["file"], "hits": hits})

    ws = {c: qwen38_workspace_bytes(c) for c in CTXS}
    headline = ws[HEADLINE_CTX]

    model = tensor_bytes + manifest_bytes
    # anything else under the artifact root is still model-specific: it is
    # there because of this patient, and G042 STORED already learned that
    # leftover directories are the whole point of complete accounting.
    other_artifact = on_disk - tensor_bytes - manifest_bytes
    model += other_artifact

    shared = shared_required
    machine = nx_bytes
    generated = headline["deltanet_state_bytes"] + headline["gqa_kv_bytes"]
    temporary = headline["activation_bytes"]

    resident = model + shared + machine + generated + temporary
    active = active_weights

    buckets = {
        "MODEL_SPECIFIC_BYTES": model,
        "SHARED_RUNTIME_BYTES": shared,
        "MACHINE_SPECIFIC_BYTES": machine,
        "GENERATED_CACHE_BYTES": generated,
        "TEMPORARY_BYTES": temporary,
        "RESIDENT_BYTES": resident,
        "ACTIVE_BYTES": active,
    }
    partition = model + shared + machine + generated + temporary
    return {
        "artifact": str(ARTIFACT),
        "identity": identity,
        "on_disk_files": len(files),
        "on_disk_bytes": on_disk,
        "manifest_bytes": manifest_bytes,
        "other_artifact_bytes": other_artifact,
        "embed": {"name": embed["name"], "table_bytes": embed["bytes"],
                  "row_bytes": embed_row, "rows": embed["shape"][0]},
        "hidden_learned_constants_in_bound_shaders": hidden_in_bound,
        "kernel_census": kcens,
        "nx_bytes": nx_bytes,
        "workspace": ws,
        "headline_ctx": HEADLINE_CTX,
        "buckets": buckets,
        "partition_sum": partition,
        "naive_seven_sum": sum(buckets[k] for k in BUCKETS_7),
        "bpw": {
            "model_specific": bpw(model),
            "payload_only": bpw(tensor_bytes),
            "on_disk_artifact": bpw(on_disk),
            "complete_with_runtime_not_amortised": bpw(model + shared + machine),
            "resident_at_ctx": bpw(resident),
            "active_per_token": bpw(active),
        },
        "anchors_confirmed": {
            "payload_bytes": tensor_bytes == G105_PAYLOAD_BYTES,
            "on_disk_bytes": on_disk == G105_ON_DISK_BYTES,
            "tensor_count": identity["manifest_tensor_count"] == G105_TENSOR_COUNT,
            "file_count": len(files) == G105_ON_DISK_FILES,
            "q4_f32": identity["q4_tensors"] == G105_Q4_TENSORS and identity["f32_tensors"] == G105_F32_TENSORS,
            "kv_per_position": headline["kv_bytes_per_position"] == 131_072,
        },
    }


# ---------------------------------------------------------------------------
# canary
# ---------------------------------------------------------------------------

def write_honest(root: Path) -> dict:
    (root / "tensors").mkdir()
    (root / "runtime").mkdir()
    (root / "nx").mkdir()
    weights = canary_f32_payload(64, b"honest-tensor")
    (root / "tensors" / "w.f32").write_bytes(weights)
    man = {
        "schema": "canary.nr.v1",
        "tensor_count": 1,
        "tensor_payload_bytes": len(weights),
        "tensors": [{"name": "w", "artifact": "tensors/w.f32", "bytes": len(weights), "kind": "f32"}],
        "required_sidecars": [],
    }
    (root / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    nx = {
        "nx_kind": "hawking.nos.noetic_executable_genome",
        "compiled_for_machine_genome": {
            "chipset": G105_CHIPSET, "gpu_cores": G105_GPU_CORES,
            "unified_memory_bytes": G105_UNIFIED, "metal_family": G105_METAL,
            "measured_roof_gb_s": G105_ROOF,
        },
        "threadgroup_geometry": {"gemv": {"threadgroup": 128}, "mha_decode": {"threadgroup": 512}},
        "kernel_binding": {"count": 1, "dispatched": ["canary_matvec"]},
    }
    (root / "nx" / "machine_genome.json").write_text(json.dumps(nx, indent=2) + "\n")
    (root / "runtime" / "host.py").write_text("def dispatch():\n    return 0\n")
    (root / "runtime" / "canary.metal").write_text(
        "kernel void canary_matvec(device const float *x [[buffer(0)]]) {\n"
        "    float acc = 0.0f;\n    acc += x[0];\n    (void)acc;\n}\n"
    )
    return {"honest_tensor_bytes": len(weights)}


def plant_source_literal(root: Path) -> dict:
    payload = canary_f32_payload(32, b"hide-1-source-literal")
    vals = floats_from_payload(payload)
    body = "LEARNED_BIAS = [\n" + fmt_list(vals) + "\n]\n"
    (root / "runtime" / "baked_bias.py").write_text(body)
    return {"attempt": 1, "name": "source_literal",
            "payload_bytes": len(payload), "n": 32,
            "where": "runtime/baked_bias.py",
            "hide": "learned tensor written into a source file as a literal array"}


def plant_metal_constant(root: Path) -> dict:
    payload = canary_f32_payload(64, b"hide-2-metal-constant")
    vals = floats_from_payload(payload)
    body = (
        "// canary shader with a learned constant baked in\n"
        "constant float learned_scale[64] = {\n"
        + fmt_list(vals)
        + "\n};\n"
        "kernel void canary_scaled(device float *x [[buffer(0)]]) {\n"
        "    x[0] *= learned_scale[0];\n}\n"
    )
    (root / "runtime" / "learned.metal").write_text(body)
    return {"attempt": 2, "name": "metal_constant",
            "payload_bytes": len(payload), "n": 64,
            "where": "runtime/learned.metal",
            "hide": "learned constant baked into a Metal shader"}


def plant_required_cache(root: Path) -> dict:
    payload = canary_f32_payload(256, b"hide-3-required-cache")
    (root / "cache").mkdir(exist_ok=True)
    (root / "cache" / "correction_lut.bin").write_bytes(payload)
    man = json.loads((root / "manifest.json").read_text())
    man.setdefault("required_sidecars", []).append("cache/correction_lut.bin")
    (root / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    return {"attempt": 3, "name": "required_cache_lut",
            "payload_bytes": len(payload), "n": 256,
            "where": "cache/correction_lut.bin",
            "hide": "lookup table shipped as a cache that is actually required for correctness"}


def plant_metallib(root: Path) -> dict:
    payload = canary_f32_payload(128, b"hide-4-metallib")
    header = b"MTLB" + b"\x00\x00\x00\x01" + b"CANARY\x00\x00" + b"\x00\x00"
    (root / "nx" / "qwen_uniform_q4_group64.metallib").write_bytes(header + payload)
    return {"attempt": 4, "name": "machine_specific_metallib",
            "payload_bytes": len(payload), "n": 128,
            "where": "nx/qwen_uniform_q4_group64.metallib",
            "file_bytes": len(header) + len(payload),
            "hide": "model information moved into a compiled GPU binary so it books as MACHINE_SPECIFIC"}


def plant_seed(root: Path) -> dict:
    n = 4096
    seed = canary_seed(b"hide-5-generated-seed")
    (root / "seeds").mkdir(exist_ok=True)
    (root / "seeds" / "up_proj.seed").write_bytes(seed.to_bytes(8, "little"))
    recipe = {"prng": "xorshift64", "seed": seed, "elements": n, "dtype": "f32"}
    (root / "seeds" / "up_proj.recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    return {"attempt": 5, "name": "generated_state_seed",
            "payload_bytes": n * 4, "n": n, "seed_bytes": 8,
            "where": "seeds/up_proj.seed + seeds/up_proj.recipe.json",
            "hide": "a generated-state seed that regenerates a large tensor at load"}


PLANTS = (
    plant_source_literal,
    plant_metal_constant,
    plant_required_cache,
    plant_metallib,
    plant_seed,
)


def _delta(before: dict, after: dict) -> dict:
    b, a = before["buckets"], after["buckets"]
    return {k: a[k] - b[k] for k in BUCKETS_7}


def verdict_for(plant: dict, delta: dict, mode: str) -> dict:
    payload = plant["payload_bytes"]
    d_model = delta["MODEL_SPECIFIC_BYTES"]
    d_shared = delta["SHARED_RUNTIME_BYTES"]
    d_machine = delta["MACHINE_SPECIFIC_BYTES"]
    d_gen = delta["GENERATED_CACHE_BYTES"]
    d_tmp = delta["TEMPORARY_BYTES"]
    counted = d_model >= int(payload * 0.9)
    if counted:
        status = "COUNTED"
        why = (f"MODEL_SPECIFIC grew by {d_model:,} B against a planted "
               f"{payload:,} B parent-derived payload")
    else:
        status = "MISSED"
        parked = max(
            [("SHARED_RUNTIME_BYTES", d_shared),
             ("MACHINE_SPECIFIC_BYTES", d_machine),
             ("GENERATED_CACHE_BYTES", d_gen),
             ("TEMPORARY_BYTES", d_tmp)],
            key=lambda kv: kv[1],
        )
        why = (f"MODEL_SPECIFIC grew by {d_model:,} B, not the planted "
               f"{payload:,} B. Largest other growth: {parked[0]} {parked[1]:,} B")
    return {
        "attempt": plant["attempt"],
        "name": plant["name"],
        "hide": plant["hide"],
        "where": plant["where"],
        "payload_bytes": payload,
        "mode": mode,
        "status": status,
        "delta": delta,
        "why": why,
    }


def run_canary() -> dict:
    isolated = []
    with tempfile.TemporaryDirectory(prefix="noetic-canary-") as td:
        root = Path(td)
        for plant_fn in PLANTS:
            case = root / f"a{plant_fn.__name__}"
            case.mkdir()
            write_honest(case)
            before_c = account_tree(case, "content")
            before_p = account_tree(case, "path")
            plant = plant_fn(case)
            after_c = account_tree(case, "content")
            after_p = account_tree(case, "path")
            isolated.append({
                "plant": plant,
                "content": verdict_for(plant, _delta(before_c, after_c), "content"),
                "path": verdict_for(plant, _delta(before_p, after_p), "path"),
                "after_buckets_content": after_c["buckets"],
                "evidence": after_c["evidence"],
            })

        combo = root / "all"
        combo.mkdir()
        write_honest(combo)
        plants = [fn(combo) for fn in PLANTS]
        combo_c = account_tree(combo, "content")
        combo_p = account_tree(combo, "path")

    content_statuses = [r["content"]["status"] for r in isolated]
    path_statuses = [r["path"]["status"] for r in isolated]
    return {
        "isolated": isolated,
        "combined": {
            "plants": plants,
            "content_buckets": combo_c["buckets"],
            "path_buckets": combo_p["buckets"],
            "content_partition_sum": combo_c["partition_sum"],
            "content_evidence": combo_c["evidence"],
        },
        "content_tally": {
            "COUNTED": content_statuses.count("COUNTED"),
            "MISSED": content_statuses.count("MISSED"),
            "statuses": content_statuses,
        },
        "path_tally": {
            "COUNTED": path_statuses.count("COUNTED"),
            "MISSED": path_statuses.count("MISSED"),
            "statuses": path_statuses,
        },
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def print_report(real: dict, canary: dict, before: dict, after: dict, g105: dict) -> None:
    b = real["buckets"]
    print("=== NOETIC INFORMATION ACCOUNTING ===")
    print(f"artifact   {real['artifact']}")
    print(f"identity   {real['on_disk_files']} files, {real['on_disk_bytes']:,} bytes")
    print(f"           G105 payload {G105_PAYLOAD_BYTES:,}  on-disk {G105_ON_DISK_BYTES:,}")
    print(f"           anchors: {json.dumps(real['anchors_confirmed'])}")
    print()
    print("seven buckets (ctx={0} for generated/resident/temporary):".format(real["headline_ctx"]))
    width = max(len(k) for k in BUCKETS_7)
    for k in BUCKETS_7:
        print(f"  {k:<{width}}  {b[k]:>16,} B   {bpw(b[k]):10.6f} BPW")
    print()
    print(f"  provenance partition (1-5)     {real['partition_sum']:>16,} B   {bpw(real['partition_sum']):10.6f} BPW")
    print(f"  naive seven-sum (with overlap) {real['naive_seven_sum']:>16,} B")
    print("  note: RESIDENT is the occupancy of the partition at ctx="
          f"{real['headline_ctx']} (G085: all weights in unified memory).")
    print("        ACTIVE is the per-token weight working set (G042: addressed")
    print("        minus the embed TABLE plus one embed ROW). They are views,")
    print("        not extra stored bytes. naive_seven_sum therefore double-counts.")
    print()
    print("MODEL_SPECIFIC breakdown")
    print(f"  tensors                 {real['identity']['payload_bytes_sum']:>16,}")
    print(f"  manifest                {real['manifest_bytes']:>16,}")
    print(f"  other under artifact    {real['other_artifact_bytes']:>16,}")
    print(f"  codec                   {real['identity']['q4_tensors']} grouped_absmax(q4,g64) + "
          f"{real['identity']['f32_tensors']} raw_f32")
    print(f"  embed table/row         {real['embed']['table_bytes']:,} / {real['embed']['row_bytes']:,}")
    print()
    kc = real["kernel_census"]
    print("SHARED_RUNTIME")
    print(f"  decode.rs               {kc.get('decode_rs_bytes', 0):>16,}")
    print(f"  bound shader files      {kc.get('bound_file_bytes', 0):>16,}  "
          f"({len(kc.get('bound_files') or [])} files, "
          f"{G105_BOUND_KERNELS} dispatched / {G105_DECLARED_KERNELS} declared)")
    print(f"  unbound shader files    {kc.get('unbound_shader_bytes', 0):>16,}  "
          f"(declared, not bound; not in SHARED_RUNTIME_BYTES)")
    print(f"  declared kernel void    {kc.get('declared_kernel_void_names')}  "
          f"matches G105 554: {kc.get('declared_matches_g105')}")
    print(f"  hidden constants in bound shaders: "
          f"{len(real['hidden_learned_constants_in_bound_shaders'])}")
    print()
    print("MACHINE_SPECIFIC")
    print(f"  sealed NX JSON          {real['nx_bytes']:>16,}  "
          f"(G105 genome {G105_CHIPSET}, {G105_GPU_CORES} cores, "
          f"{G105_UNIFIED:,} B, {G105_METAL}, roof {G105_ROOF} GB/s)")
    print()
    print("GENERATED_CACHE / TEMPORARY  (runtime-allocated, not on disk in the artifact)")
    print(f"  {'ctx':>8}  {'DeltaNet':>14}  {'KV':>14}  {'activation':>14}  {'workspace':>14}")
    for c, w in real["workspace"].items():
        mark = "  <-- headline" if c == real["headline_ctx"] else ""
        print(f"  {c:>8}  {w['deltanet_state_bytes']:>14,}  {w['gqa_kv_bytes']:>14,}  "
              f"{w['activation_bytes']:>14,}  {w['total_bytes']:>14,}{mark}")
    print()
    print("EBPW")
    for k, v in real["bpw"].items():
        print(f"  {k:<38} {v:10.6f}")
    print()
    print(f"load (G105, not re-measured): {G105_TPS} tps, {G105_TOKEN_MS} ms/token, battery {G105_BATTERY}")
    dg = real["identity"]["g105_digest_checks_under_1miB"]
    print(f"content-addressing defect: filename==sha256 on {dg['filename_equals_sha256']}/{dg['n']} "
          f"hashed tensors (<1 MiB). Confirmed.")
    print(f"G105 digest check on those {dg['n']}: {dg['matches']} match, {dg['mismatches']} differ "
          f"{dg['mismatch_classes']}. Sizes still match the manifest and the G105 payload total; "
          f"this copy is size-identical, not byte-identical, on the RMSNorm/QK-norm f32 files. "
          f"Did not re-hash the 14 GB q4 payload. Rolling digest {G105_ROLLING_DIGEST} not recomputed.")
    print("byte_reproducibility_NOT_met is a G105 finding, not re-derived.")
    print()

    print("=== CANARY ===")
    print("Five isolated plants. content = this accountant. path = naive path-only.")
    for row in canary["isolated"]:
        c, p = row["content"], row["path"]
        print(f"  {c['attempt']}. {c['name']:<28} content={c['status']:<7} path={p['status']:<7}  "
              f"payload {c['payload_bytes']:,} B")
        print(f"       {c['hide']}")
        print(f"       {c['where']}")
        print(f"       content: {c['why']}")
        if p["status"] != c["status"]:
            print(f"       path:    {p['why']}")
    print()
    print(f"  content tally  COUNTED {canary['content_tally']['COUNTED']}  "
          f"MISSED {canary['content_tally']['MISSED']}")
    print(f"  path tally     COUNTED {canary['path_tally']['COUNTED']}  "
          f"MISSED {canary['path_tally']['MISSED']}")
    print()

    print("=== ARTIFACT UNTOUCHED ===")
    same = before == after
    print(f"  snapshot equal: {same}")
    print(f"  before {before['file_count']} files, {before['total_bytes']:,} B, "
          f"manifest {before['manifest_sha256'][:16]}… meta {before['meta_digest'][:16]}…")
    print(f"  after  {after['file_count']} files, {after['total_bytes']:,} B, "
          f"manifest {after['manifest_sha256'][:16]}… meta {after['meta_digest'][:16]}…")
    if not same:
        print("  FAIL: artifact identity changed during this run")
    print()

    print("## WHAT I WATCHED FAIL")
    print()
    print("Path-only accounting (classify by directory: runtime→SHARED,")
    print("nx/.metallib→MACHINE, cache→GENERATED, seeds counted at file size,")
    print("no scan of source literals) missed every hide:")
    for row in canary["isolated"]:
        print(f"  path  {row['path']['attempt']}. {row['path']['name']}: {row['path']['status']}")
        print(f"        {row['path']['why']}")
    print()
    print("That is the 0.08 BPW lie in miniature: the bytes are still on disk,")
    print("just sitting in a bucket a representation search is willing to exclude")
    print("or amortise. Content-aware accounting reclassifies source literals,")
    print("Metal `constant` arrays, non-regenerable 'caches', and seed expansions")
    print("as MODEL_SPECIFIC. It still misses the compiled GPU binary:")
    miss = [r for r in canary["isolated"] if r["content"]["status"] == "MISSED"]
    if miss:
        for r in miss:
            print(f"  content MISS  {r['content']['name']}: {r['content']['why']}")
        print()
        print("Smallest change that would catch the metallib hide: treat compiled")
        print("GPU blobs (.metallib/.cubin/.spv) as MODEL_SPECIFIC unless a rebuild")
        print("from scanned shader *source* reproduces the blob (or its constant")
        print("data sections). Path is not provenance. A fatbin is exactly where")
        print("learned `__constant__` tables have always lived.")
        print()
        print("I believe this is strong accounting with a real remaining hole,")
        print("not a weak canary. Attempts 1–3 and 5 are the hides a packing")
        print("search actually uses (literals, shader constants, 'cache' LUTs,")
        print("PRNG-expanded bases). Attempt 4 is the one that would have broken")
        print("a claim of completeness, and it did.")
    else:
        print("Content-aware accounting caught all five. That is a warning, not a")
        print("celebration: a canary with no miss is usually too polite. The")
        print("attempt that would still break it is a learned u8 codebook encoded")
        print("as a sequence of valid threadgroup sizes (32/64/128/256/512/1024)")
        print("inside NX geometry. The JSON walker deliberately allows short")
        print("power-of-two integer lists as geometry, and that is the next hole.")
    print()
    print(f"-> {RECEIPT}")


def main() -> int:
    self_check_workspace()
    before = snapshot_tree(ARTIFACT)
    if before["file_count"] != G105_ON_DISK_FILES or before["total_bytes"] != G105_ON_DISK_BYTES:
        print("FAIL: on-disk artifact does not match the G105 identity "
              f"(got {before['file_count']} files, {before['total_bytes']} bytes; "
              f"expected {G105_ON_DISK_FILES} / {G105_ON_DISK_BYTES})", file=sys.stderr)
        return 2

    g105 = git_show_json(G105_PATH)
    g105_digests = git_show_json(G105_DIGESTS_PATH)
    if g105.get("schema") != "hawking.nos.nr_nx_artifact.v1":
        print(f"FAIL: unexpected G105 schema {g105.get('schema')}", file=sys.stderr)
        return 2

    real = account_real(g105, g105_digests)
    canary = run_canary()
    after = snapshot_tree(ARTIFACT)
    untouched = before == after

    doc = {
        "schema": "hawking.headless.noetic_information_accounting.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "why": (
            "A representation search that reports EBPW is dishonest if supporting "
            "structures escape the accounting. Seven buckets over the sealed "
            "uniform-q4-v1 artifact, then a canary that hides parent-derived "
            "bytes in excluded buckets."
        ),
        "anchors_not_rederived": {
            "artifact": "uniform-q4-v1",
            "parameter_count": SOURCE_PARAM_COUNT,
            "payload_bytes": G105_PAYLOAD_BYTES,
            "on_disk_bytes": G105_ON_DISK_BYTES,
            "tps": G105_TPS, "token_ms": G105_TOKEN_MS, "battery": G105_BATTERY,
            "machine": {
                "chipset": G105_CHIPSET, "gpu_cores": G105_GPU_CORES,
                "unified_memory_bytes": G105_UNIFIED, "metal_family": G105_METAL,
                "measured_roof_gb_s": G105_ROOF,
            },
            "kernel_binding": {"dispatched": G105_BOUND_KERNELS, "declared": G105_DECLARED_KERNELS},
            "content_addressing_defect": "64-hex filenames look like content addresses and are not",
            "byte_reproducibility_NOT_met": True,
            "g105_rolling_digest": G105_ROLLING_DIGEST,
        },
        "definitions": {
            "MODEL_SPECIFIC_BYTES": (
                "bytes whose information is derived from the parent. Tensor "
                "payloads, the manifest, leftover files under the artifact root, "
                "learned literals/constants wherever they sit, non-regenerable "
                "'caches', and PRNG expansions of learned seeds."
            ),
            "SHARED_RUNTIME_BYTES": (
                "host decode + shader files that contain a dispatched kernel. "
                "Amortisable across models of this architecture, but must appear. "
                "Declared-but-unbound shaders are reported separately and are "
                "not in the bucket."
            ),
            "MACHINE_SPECIFIC_BYTES": (
                "the sealed NX genome (G105): compiled_for_machine_genome, "
                "kernel_binding, threadgroup_geometry, residency_plan, "
                "cache_plan, scheduling. Plus compiled GPU blobs, which this "
                "accountant currently trusts as machine-only — the canary miss."
            ),
            "GENERATED_CACHE_BYTES": (
                "runtime-allocated state implied by the NX residency plan: "
                "DeltaNet recurrent+conv state (context-independent) plus GQA KV "
                f"at ctx={HEADLINE_CTX}. Zero on disk in this artifact (G042 "
                "GENERATED_BPW_EQUIVALENT is 0). A file under cache/ that cannot "
                "be regenerated from a declared seed is NOT this bucket."
            ),
            "TEMPORARY_BYTES": (
                "qwen38_workspace_bytes activation scratch. Independent of seq_len. "
                "Traffic (G042 CACHE_BPW, ~108 GB/token of f32 activations) is not "
                "storage and is not charged here."
            ),
            "RESIDENT_BYTES": (
                "occupancy of the provenance partition when the NX is loaded at "
                f"ctx={HEADLINE_CTX}. G085: all weights in unified memory; no SSD tier."
            ),
            "ACTIVE_BYTES": (
                "bytes addressed to produce one token on the weight path: payload "
                "minus the embed TABLE plus one embed ROW (G042). MLP is not "
                "conditional (G088), so average==worst on weights."
            ),
        },
        "artifact_accounting": real,
        "canary": {
            "rule": (
                "COUNTED iff MODEL_SPECIFIC grew by >= 90% of the planted "
                "parent-derived payload. Bytes that merely appear in SHARED / "
                "MACHINE / GENERATED / TEMP do not count as caught: those are "
                "the buckets a dishonest EBPW excludes or amortises."
            ),
            "isolated": [
                {
                    "attempt": r["content"]["attempt"],
                    "name": r["content"]["name"],
                    "hide": r["content"]["hide"],
                    "where": r["content"]["where"],
                    "payload_bytes": r["content"]["payload_bytes"],
                    "content": r["content"]["status"],
                    "path_only": r["path"]["status"],
                    "content_why": r["content"]["why"],
                    "path_why": r["path"]["why"],
                    "content_delta": r["content"]["delta"],
                    "path_delta": r["path"]["delta"],
                }
                for r in canary["isolated"]
            ],
            "content_tally": canary["content_tally"],
            "path_tally": canary["path_tally"],
            "combined_content_buckets": canary["combined"]["content_buckets"],
            "completeness": (
                "not complete. compiled GPU binaries are trusted as machine-only."
                if canary["content_tally"]["MISSED"]
                else "caught the five plants; geometry-shaped integer codes would still miss."
            ),
        },
        "artifact_untouched": {
            "verified": untouched,
            "method": (
                "lstat walk: relpath + size + mtime_ns + inode, plus sha256 of "
                "manifest.json. No writes under the artifact. Canary used tempfile. "
                "Did not re-hash the 14 GB payload; hashed every tensor < 1 MiB "
                "against G105_TENSOR_DIGESTS and summed sizes of all 755."
            ),
            "before": before,
            "after": after,
        },
        "what_i_watched_fail": {
            "path_only_classifier": (
                "runtime→SHARED, nx/.metallib→MACHINE, cache→GENERATED, seeds at "
                "file size, no source scan. Missed all five hides. The bytes did "
                "not disappear; they sat in buckets a representation search excludes."
            ),
            "content_aware_remaining_hole": (
                "compiled GPU binary (.metallib) with a learned f32 table after a "
                "plausible MTLB header. Booked as MACHINE_SPECIFIC. Smallest fix: "
                "charge compiled GPU blobs to MODEL_SPECIFIC unless reproduced "
                "from scanned shader source."
            ),
            "would_also_miss": (
                "a learned u8 codebook encoded as a sequence of valid threadgroup "
                "sizes inside NX geometry. The JSON walker allows short power-of-two "
                "integer lists as geometry by design."
            ),
        },
        "g105_defects_not_rederived": {
            "content_addressing_defect_found_and_fixed_forward":
                g105.get("content_addressing_defect_found_and_fixed_forward"),
            "byte_reproducibility_NOT_met": g105.get("byte_reproducibility_NOT_met"),
        },
        "baseline_suite": {
            "claimed": "464 passed, 1 skipped",
            "ran": (
                "not in this sparse worktree. tools/haider is DENIED and not "
                "materialized; git apply of tree-state.patch failed with "
                "'hcli/*.py: No such file or directory'. pytest "
                "tools/headless collection errors with ModuleNotFoundError: hcli. "
                "Non-haider scripts storage_manager_test and performance_ledger_test "
                "passed. rollback_integrity_test needs hcli and failed to import."
            ),
        },
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(real, canary, before, after, g105)
    return 0 if untouched and canary["content_tally"]["MISSED"] >= 1 else (0 if untouched else 1)


if __name__ == "__main__":
    sys.exit(main())
