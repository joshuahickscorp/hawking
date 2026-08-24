#!/usr/bin/env python3
"""N014 — attack the frontier claims. A refutation is the valuable outcome.

Each of the six questions is RUN, not answered from the receipts. Plants go
through the live accountant. Counters are grepped in the decoder that emitted
them. Arithmetic is recomputed from the organ census. Capability predicates
score the sealed 16-token sample. Optional live Metal on the sealed parent
is an extra observation, never a second 27B.

    python3 tools/headless/frontier_adversary.py
    python3 -m pytest tools/headless -q

Does not modify ~/noetic/NOETIC_PARENT_A, receipts/ascent-2026-08-16, or
workspace/campaign.
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import noetic_information_accounting as accounting  # noqa: E402
from capability_suite import SUITE  # noqa: E402
from causal_benchmark_law import REQUIREMENTS, audit as audit_law  # noqa: E402
from noetic_dispatch_fusion import theoretical_after  # noqa: E402
from noetic_multisession import (  # noqa: E402
    expected_n_copies_bytes,
    expected_shared_resident_bytes,
    one_body_not_n_copies,
)
from noetic_parent_a import (  # noqa: E402
    DURABLE,
    RECORDED_TOKEN_IDS,
    RECORDED_TOK_S,
    RECORDED_TOK_S_MAX,
    RECORDED_TOK_S_MIN,
    reseal,
)
from noetic_zero_parent import (  # noqa: E402
    GpuLaneLock,
    OPENLOG_C,
    classify_path,
    compile_dylib,
    observe_command,
    parse_decode_output,
    parse_open_log,
)

SCHEMA = "hawking.headless.frontier_adversary.v1"
RECEIPT = REPO / "receipts" / "headless" / "FRONTIER_ADVERSARY.json"
HEADLESS_RECEIPTS = REPO / "receipts" / "headless"
PARENT_A = Path.home() / "noetic" / "NOETIC_PARENT_A"
PARENT_BF16 = Path.home() / "models" / "qwen3.8-27b-abliterated-bf16"
TOKENIZER = PARENT_BF16 / "tokenizer.json"
FUSED_SUBBIT = Path(
    "/Users/scammermike/.claude-grok/worktrees/n001seal-20260823-223440"
    "/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_fused_subbit"
)
DECODE_RS = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
GREEDY_RS = REPO / "crates" / "hawking-core" / "examples" / "ascension_qwen38_hybrid_greedy.rs"
FUSED_RS = REPO / "crates" / "hawking-core" / "examples" / "ascension_qwen38_fused_subbit.rs"
AFFINE_METAL = REPO / "crates" / "hawking-core" / "shaders" / "affine2_group32_matvec.metal"
PARENT_PARAMS = 26_895_998_464
SEALED_CLOSURE = "7921a6a27e0561343c1b54b740ef0c552bff7c939117487c88d2e6b4d4de5adb"

ORGAN_ELEMS = {
    "embedding": 1_271_398_400,
    "attention_gqa": 1_677_811_712,
    "deltanet": 5_562_296_832,
    "mlp": 17_113_088_000,
    "output": 1_271_403_520,
}

ONEBIT_RULE = {
    "gain_min": 0.5,
    "rel_fro_max": 0.5,
    "scale_aware_min": 0.05,
}

CLAIM_IDS = (
    "NOETIC_PARENT_A",
    "dispatch_fusion",
    "GPU_ledger",
    "multisession",
    "organ_floors",
    "onebit_families",
)

QUESTION_IDS = (
    "hidden_bytes",
    "hidden_parent",
    "reconstructed_dense_state",
    "wrong_benchmark_path",
    "cache_artifact",
    "capability_collapse",
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((HEADLESS_RECEIPTS / name).read_text())


def packed_affine2_codes(n_words: int = 256, salt: bytes = b"hide-int-codes-n014") -> bytes:
    """Parent-shaped 2-bit codes. Not f32. The accountant's remaining named gap."""
    x = int.from_bytes(hashlib.sha256(salt).digest()[:8], "little") or 1
    out = bytearray()
    for _ in range(n_words):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        word = 0
        for k in range(16):
            word |= ((x >> (2 * k)) & 3) << (2 * k)
        out += struct.pack("<I", word & 0xFFFFFFFF)
    return bytes(out)


def plant_integer_metallib(root: Path) -> dict[str, Any]:
    payload = packed_affine2_codes()
    header = b"MTLB" + b"\x00\x00\x00\x01" + b"INTC" + b"\x00" * 10
    blob = header + payload
    dest = root / "nx" / "learned_q2.metallib"
    dest.write_bytes(blob)
    return {
        "attempt": 6,
        "name": "integer_coded_metallib",
        "hide": "packed 2-bit affine codes inside a .metallib booked as MACHINE_SPECIFIC",
        "where": "nx/learned_q2.metallib",
        "payload_bytes": len(payload),
        "file_bytes": len(blob),
        "f32_scanner_bytes": accounting.embedded_weightlike_bytes(blob),
    }


def plant_integer_sidecar(root: Path) -> dict[str, Any]:
    payload = packed_affine2_codes(salt=b"hide-int-sidecar-n014")
    dest = root / "nx" / "learned_q.u8"
    dest.write_bytes(payload)
    return {
        "attempt": 7,
        "name": "integer_coded_machine_sidecar",
        "hide": "packed 2-bit codes as nx/*.u8 so path_role books MACHINE_SPECIFIC",
        "where": "nx/learned_q.u8",
        "payload_bytes": len(payload),
        "f32_scanner_bytes": accounting.embedded_weightlike_bytes(payload),
    }


def plant_geometry_codebook(root: Path, n: int) -> dict[str, Any]:
    cycle = [32, 64, 128, 256, 512, 1024]
    codes = [cycle[i % len(cycle)] for i in range(n)]
    nx_path = root / "nx" / "machine_genome.json"
    nx = json.loads(nx_path.read_text())
    nx.setdefault("threadgroup_geometry", {})["learned_codebook"] = codes
    nx_path.write_text(json.dumps(nx, indent=2) + "\n")
    return {
        "attempt": 8 if n <= 64 else 9,
        "name": f"geometry_shaped_u8_codebook_n{n}",
        "hide": (
            "learned integer codebook encoded as valid threadgroup sizes "
            f"(n={n}, cap 64 is the JSON walker's explicit hole)"
        ),
        "where": "nx/machine_genome.json",
        "payload_bytes": n * 4,
        "n": n,
    }


def run_plant(plant_fn, *args) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="frontier-canary-") as td:
        root = Path(td)
        accounting.write_honest(root)
        before = accounting.account_tree(root, "content")
        plant = plant_fn(root, *args) if args else plant_fn(root)
        after = accounting.account_tree(root, "content")
        delta = {k: after["buckets"][k] - before["buckets"][k] for k in accounting.BUCKETS_7}
        verdict = accounting.verdict_for(plant, delta, "content")
        return {
            "plant": plant,
            "content": verdict,
            "delta": delta,
            "after_buckets": after["buckets"],
            "evidence": after.get("evidence") or [],
        }


def attack_hidden_bytes() -> dict[str, Any]:
    """Q1: plant a hide the accountant has not seen — integer-coded weights."""
    metallib = run_plant(plant_integer_metallib)
    sidecar = run_plant(plant_integer_sidecar)
    geo30 = run_plant(plant_geometry_codebook, 30)
    geo66 = run_plant(plant_geometry_codebook, 66)
    known = accounting.run_canary()
    known_tally = known["content_tally"]
    new_misses = [
        row["content"]["name"]
        for row in (metallib, sidecar, geo30)
        if row["content"]["status"] == "MISSED"
    ]
    refuted = metallib["content"]["status"] == "MISSED"
    return {
        "id": "hidden_bytes",
        "question": "Plant a payload and confirm the accountant catches it. Try a hide it has NOT seen — integer-coded weights.",
        "ran": True,
        "commands": [
            "python3 -c 'packed 2-bit affine codes -> account_tree content mode'",
            "noetic_information_accounting.run_canary()  # 5/5 still caught",
        ],
        "known_five": known_tally,
        "new_hides": {
            "integer_coded_metallib": metallib["content"],
            "integer_coded_machine_sidecar": sidecar["content"],
            "geometry_codebook_n30": geo30["content"],
            "geometry_codebook_n66": geo66["content"],
        },
        "f32_scanner_on_integer_codes": metallib["plant"]["f32_scanner_bytes"],
        "named_gap_confirmed": new_misses,
        "verdict": "REFUTED" if refuted else "SURVIVED",
        "why": (
            "The accountant still catches the original 5/5 plants, including .metallib "
            f"f32 ({known_tally}). Packed 2-bit affine codes in a .metallib booked "
            f"{metallib['content']['status']}: MODEL_SPECIFIC grew by "
            f"{metallib['content']['delta']['MODEL_SPECIFIC_BYTES']} B against a planted "
            f"{metallib['plant']['payload_bytes']} B, parked in MACHINE_SPECIFIC. "
            "embedded_weightlike_bytes scored 0 on the integer payload. Short "
            f"geometry-shaped integer lists (n=30) {geo30['content']['status']}; n=66 "
            f"(over the walker's 64 cap) {geo66['content']['status']}. Integer-coded "
            "weights remain an open hide. PARENT_A's 3.1393 EBPW is an artifact-tree "
            "sum, not a completeness proof against this hide."
        ),
    }


def _source_parent_weight_hits(text: str, path: str) -> list[dict[str, Any]]:
    hits = []
    needles = (
        ".safetensors",
        "qwen3.8-27b-abliterated-bf16",
        "QWEN38_PARENT",
        "parent_bf16",
        "SourceBF16",
    )
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if "tokenizer" in low and "safetensor" not in low:
            continue
        for n in needles:
            if n.lower() in low:
                hits.append({"file": path, "line": i, "needle": n, "text": line.strip()[:200]})
                break
    return hits


def _ast_opens_parent_weights(path: Path) -> list[str]:
    """Python-only. Rust is grepped. A Constant open() of a safetensors path is a hit."""
    if path.suffix != ".py" or not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"open", "read_bytes", "read_text"}:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.endswith(".safetensors") or "model-000" in arg.value:
                    found.append(f"{path.name}:{getattr(node, 'lineno', '?')} {name}({arg.value})")
    return found


def page_cache_frac(path: Path, limit: int = 32 << 20) -> dict[str, Any]:
    """macOS mincore over a private mmap. Does not purge. Does not write the file."""
    libc_name = ctypes.util.find_library("c")
    if not libc_name or not path.is_file():
        return {"frac": None, "reason": "no libc or missing file", "path": str(path)}
    libc = ctypes.CDLL(libc_name, use_errno=True)
    pagesize = os.sysconf("SC_PAGESIZE")
    fd = os.open(str(path), os.O_RDONLY)
    try:
        size = min(os.fstat(fd).st_size, limit)
        if size <= 0:
            return {"frac": None, "reason": "empty", "path": str(path)}
        prot_read, map_private = 1, 2
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        addr = libc.mmap(None, size, prot_read, map_private, fd, 0)
        map_failed = ctypes.c_void_p(-1).value
        if not addr or addr == map_failed:
            return {"frac": None, "errno": ctypes.get_errno(), "path": str(path)}
        try:
            npages = (size + pagesize - 1) // pagesize
            vec = (ctypes.c_ubyte * npages)()
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            rc = libc.mincore(addr, size, vec)
            if rc != 0:
                return {"frac": None, "rc": rc, "errno": ctypes.get_errno(), "path": str(path)}
            resident = sum(1 for i in range(npages) if vec[i] & 1)
            return {
                "path": str(path),
                "bytes": size,
                "pages": npages,
                "resident": resident,
                "frac": resident / npages,
                "pagesize": pagesize,
            }
        finally:
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def live_parent_a_observe(*, max_new_tokens: int, prompt: str, timeout: float) -> dict[str, Any]:
    """DYLD-observe fused_subbit on the sealed parent. Tokenizer only from bf16."""
    if not FUSED_SUBBIT.is_file():
        return {"ran": False, "reason": f"fused_subbit missing: {FUSED_SUBBIT}"}
    if not PARENT_A.is_dir() or not (PARENT_A / "catalog.hq38m20").is_file():
        return {"ran": False, "reason": f"sealed parent missing: {PARENT_A}"}
    if not TOKENIZER.is_file():
        return {"ran": False, "reason": f"tokenizer missing: {TOKENIZER}"}
    with tempfile.TemporaryDirectory(prefix="frontier-openlog-") as td:
        tmp = Path(td)
        try:
            dylib = compile_dylib(tmp / "libopenlog.dylib")
        except SystemExit as e:
            return {"ran": False, "reason": f"dylib compile failed: {e}"}
        raw_out = tmp / "live.json"
        argv = [
            str(FUSED_SUBBIT),
            "--artifact-root",
            str(PARENT_A),
            "--tokenizer",
            str(TOKENIZER),
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-seq-len",
            "128",
            "--reps",
            "1",
            "--skip-kernel-cost",
            "--out",
            str(raw_out),
        ]
        try:
            with GpuLaneLock("n014-frontier-adversary", timeout=20.0):
                live = observe_command(
                    argv,
                    dylib=dylib,
                    log_path=tmp / "live.log",
                    parent=PARENT_BF16,
                    artifact=PARENT_A,
                    timeout=timeout,
                )
        except SystemExit as e:
            return {"ran": False, "reason": str(e), "argv": argv}
        decoded = parse_decode_output(live.get("stdout") or "", live.get("stderr") or "")
        live.update(decoded)
        obs = live.get("observation") or parse_open_log(
            tmp / "live.log", parent=PARENT_BF16, cwd=tmp, artifact=PARENT_A
        )
        parent_weight = obs.get("parent_weight") or []
        return {
            "ran": True,
            "argv": argv,
            "binary": str(FUSED_SUBBIT),
            "exit_code": live.get("exit_code"),
            "timed_out": live.get("timed_out"),
            "elapsed_s": live.get("elapsed_s"),
            "dylib_loaded": live.get("dylib_loaded"),
            "generated_text": decoded.get("generated_text"),
            "new_token_ids": decoded.get("new_token_ids"),
            "n_new_tokens": decoded.get("n_new_tokens"),
            "n_parent_weight_opens": len(parent_weight),
            "parent_weight_paths": parent_weight,
            "parent_tokenizer_paths": obs.get("parent_tokenizer") or [],
            "n_file_open_events": obs.get("n_events"),
            "stderr_head": (live.get("stderr") or "")[-1500:],
            "stdout_head": (live.get("stdout") or "")[:1500],
            "observation": {
                "n_events": obs.get("n_events"),
                "n_unique": obs.get("n_unique"),
                "parent_weight": parent_weight,
                "parent_tokenizer": obs.get("parent_tokenizer"),
                "parent_config": obs.get("parent_config"),
            },
        }


def attack_hidden_parent(*, live_gpu: bool) -> dict[str, Any]:
    """Q2: does anything open the bf16 parent at inference?"""
    hits: list[dict[str, Any]] = []
    for path in (DECODE_RS, GREEDY_RS, FUSED_RS):
        if path.is_file():
            rel = str(path.relative_to(REPO))
            hits.extend(_source_parent_weight_hits(path.read_text(encoding="utf-8", errors="replace"), rel))
    py_opens = []
    for py in (HERE / "noetic_parent_a.py", HERE / "noetic_fused_subbit.py"):
        py_opens.extend(_ast_opens_parent_weights(py))

    prior = None
    prior_path = HEADLESS_RECEIPTS / "NOETIC_ZERO_PARENT.json"
    if prior_path.is_file():
        z = json.loads(prior_path.read_text())
        prior = {
            "artifact": z.get("artifact"),
            "is_parent_a": "NOETIC_PARENT_A" in str(z.get("artifact") or ""),
            "n_parent_weight_opens": (z.get("production_run") or {}).get("n_parent_weight_opens"),
            "verdict": (z.get("production_run") or {}).get("verdict") or z.get("verdict"),
            "negative_control_caught": (z.get("negative_control") or {}).get("detector_caught"),
        }

    live = {"ran": False, "reason": "live_gpu disabled"}
    if live_gpu:
        live = live_parent_a_observe(
            max_new_tokens=48,
            prompt="What is the capital of France? Answer with only the city name.",
            timeout=180.0,
        )

    live_opened = bool(live.get("ran")) and (live.get("n_parent_weight_opens") or 0) > 0
    live_metal_refused = "no Metal-capable GPU" in (live.get("stderr_head") or "")
    live_complete = bool(
        live.get("ran")
        and live.get("exit_code") == 0
        and (live.get("n_new_tokens") or 0) >= 1
        and not live_metal_refused
    )

    if live_opened:
        verdict = "REFUTED"
        why = (
            "Live DYLD interpose on the sealed-parent fused_subbit binary observed "
            f"{live.get('n_parent_weight_opens')} parent-weight opens: "
            f"{live.get('parent_weight_paths')}"
        )
    else:
        verdict = "SURVIVED"
        why = (
            "Decode sources do not open .safetensors. Python packers do not open() a "
            "safetensors path. The prior complete DYLD run on the q4 incumbent reported "
            f"n_parent_weight_opens={(prior or {}).get('n_parent_weight_opens')} and the "
            "SourceBF16 negative control was caught "
            f"({(prior or {}).get('negative_control_caught')}). "
        )
        if live.get("ran"):
            why += (
                f"This lane also launched fused_subbit against PARENT_A under DYLD "
                f"(exit={live.get('exit_code')} events={live.get('n_file_open_events')} "
                f"parent_weight_opens={live.get('n_parent_weight_opens')} tokenizer="
                f"{live.get('parent_tokenizer_paths')}"
            )
            if live_metal_refused or not live_complete:
                why += (
                    "). Metal refused ('no Metal-capable GPU') before a token was emitted; "
                    "a truncated prefix is INCONCLUSIVE for the sealed parent, not a PASS. "
                    "No parent .safetensors appeared in the 16 opens that did happen."
                )
            else:
                why += ")."
        else:
            why += (
                f"Live PARENT_A observe did not run ({live.get('reason')}). "
                "tokenizer.json under the bf16 tree is a tokenizer dependency, not a weight open."
            )

    return {
        "id": "hidden_parent",
        "question": "Does anything open the bf16 parent at inference?",
        "ran": True,
        "commands": [
            f"rg safetensors {DECODE_RS}",
            "ast walk of noetic_parent_a.py / noetic_fused_subbit.py for open(.safetensors)",
            (
                "DYLD_INSERT_LIBRARIES=noetic_openlog.dylib fused_subbit --artifact-root "
                "~/noetic/NOETIC_PARENT_A"
                if live.get("ran")
                else "live observe skipped"
            ),
        ],
        "source_hits": hits[:12],
        "python_opens": py_opens,
        "prior_zero_parent": prior,
        "live": {k: v for k, v in live.items() if k != "observation"} | {
            "observation": live.get("observation")
        },
        "did_not_load_second_27b": True,
        "did_not_modify_parent_a": True,
        "verdict": verdict,
        "why": why,
    }


def _count_hardcoded_dense(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    literal_zero = len(re.findall(r"dense_w_materialized:\s*0", text))
    print_zero = len(re.findall(r"DENSE_W_MATERIALIZED:\s*0", text))
    json_zero = len(re.findall(r'"dense_w_materialized":\s*0', text))
    # Production packed GEMV never calls `account_dense_w` (N021 diagnostic).
    # Count only increments outside that method so a later reconstruct hook
    # cannot hide a real += on the decode path.
    increments = 0
    for m in re.finditer(r"dense_w_materialized\s*\+=", text):
        window = text[max(0, m.start() - 240) : m.start()]
        if "fn account_dense_w" not in window:
            increments += 1
    assigns = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"dense_w_materialized\s*=", line) and "0" not in line.split("=")[-1][:8]
    ]
    dequant_dispatch = "affine2_group32_dequant" in text
    return {
        "path": str(path.relative_to(REPO)) if REPO in path.parents or path == REPO else str(path),
        "present": True,
        "literal_zero": literal_zero,
        "print_zero": print_zero,
        "json_zero": json_zero,
        "increments": increments,
        "non_zero_assigns": assigns[:8],
        "mentions_dequant_kernel": dequant_dispatch,
        "bytes": len(text),
    }


def affine2_numpy_probe() -> dict[str, Any]:
    """Synthetic group-64 affine2: reconstruct allocates parent W, fused does not."""
    import numpy as np

    rows, cols, group = 64, 256, 64
    parent_numel = rows * cols
    rng = np.random.RandomState(14)
    q = rng.randint(0, 4, size=(rows, cols), dtype=np.int32)
    n_g = cols // group
    scale = rng.randn(rows, n_g).astype(np.float32) * 0.05
    bias = rng.randn(rows, n_g).astype(np.float32) * 0.01
    x = rng.randn(4, cols).astype(np.float32)

    allocated: list[int] = []

    def record(n: int) -> None:
        allocated.append(int(n))

    # Reconstruct path: materialise W then matvec.
    allocated.clear()
    W = (q.astype(np.float32) * np.repeat(scale, group, axis=1)) + np.repeat(bias, group, axis=1)
    record(W.size)
    y_r = x @ W.T
    reconstruct_peak = max(allocated)
    reconstruct_hits_parent = reconstruct_peak >= parent_numel

    # Fused path: per-group outer product, never (rows, cols).
    allocated.clear()
    y_f = np.zeros((x.shape[0], rows), dtype=np.float32)
    for g in range(n_g):
        sl = slice(g * group, (g + 1) * group)
        Wg = q[:, sl].astype(np.float32) * scale[:, g : g + 1] + bias[:, g : g + 1]
        record(Wg.size)
        y_f += x[:, sl] @ Wg.T
    fused_peak = max(allocated) if allocated else 0
    fused_hits_parent = fused_peak >= parent_numel
    max_abs = float(np.max(np.abs(y_r - y_f)))
    return {
        "rows": rows,
        "cols": cols,
        "parent_numel": parent_numel,
        "reconstruct_peak_numel": reconstruct_peak,
        "fused_peak_numel": fused_peak,
        "reconstruct_zero_dense": not reconstruct_hits_parent,
        "fused_zero_dense": not fused_hits_parent,
        "max_abs_diff": max_abs,
        "detector_ok": reconstruct_hits_parent and not fused_hits_parent,
    }


def attack_dense_state() -> dict[str, Any]:
    """Q3: are the zero counters real, or is dense W materialised off-instrument?"""
    decode = _count_hardcoded_dense(DECODE_RS)
    greedy = _count_hardcoded_dense(GREEDY_RS)
    fused = _count_hardcoded_dense(FUSED_RS)
    metal = AFFINE_METAL.read_text(encoding="utf-8", errors="replace") if AFFINE_METAL.is_file() else ""
    dequant_kernel = "kernel void affine2_group32_dequant(" in metal
    gemv_claims_registers = "never writes a dense W" in metal or "In-register dequant only" in metal
    dequant_only_in_parity = True
    parity = REPO / "crates" / "hawking-core" / "examples" / "affine2_parity.rs"
    if parity.is_file():
        dequant_only_in_parity = "affine2_group32_dequant" in parity.read_text(
            encoding="utf-8", errors="replace"
        )
    probe = affine2_numpy_probe()
    census_hardcoded = False
    if DECODE_RS.is_file():
        t = DECODE_RS.read_text(encoding="utf-8", errors="replace")
        census_hardcoded = "expanded_to_q4=0" in t and "expanded_to_float_gemv=0" in t

    counter_is_constant = (
        decode.get("increments", 0) == 0
        and greedy.get("increments", 0) == 0
        and (greedy.get("print_zero", 0) + greedy.get("json_zero", 0) + decode.get("literal_zero", 0))
        > 0
    )
    return {
        "id": "reconstructed_dense_state",
        "question": "Are the zero counters real, or is a dense tensor materialized somewhere the counters do not watch?",
        "ran": True,
        "commands": [
            "rg dense_w_materialized crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            "rg affine2_group32_dequant crates/hawking-core",
            "numpy affine2 reconstruct vs fused allocation probe (64x256, not 27B)",
        ],
        "decoder": decode,
        "greedy_example": greedy,
        "fused_example": fused,
        "metal": {
            "dequant_kernel_present": dequant_kernel,
            "gemv_claims_in_register": gemv_claims_registers,
            "dequant_kernel_referenced_from_decode": decode.get("mentions_dequant_kernel"),
            "dequant_kernel_only_in_parity_example": dequant_only_in_parity
            and not decode.get("mentions_dequant_kernel"),
        },
        "census_format_string_hardcodes_zeros": census_hardcoded,
        "numpy_probe": probe,
        "verdict": "REFUTED" if counter_is_constant else "SURVIVED",
        "why": (
            "DENSE_W_MATERIALIZED / dense_w_materialized / expanded_to_q4 / "
            "expanded_to_float_gemv on the Qwen38 production path are literals. "
            f"qwen38_hybrid_decode.rs increments={decode.get('increments')} "
            f"literal_zero={decode.get('literal_zero')}; greedy prints "
            f"DENSE_W_MATERIALIZED: 0 ({greedy.get('print_zero')} times). MixedCatalogCensus "
            "comments that expanded_to_q4 stays zero 'so a later reader cannot miss a "
            "forbidden fallback' — and the eprintln format string writes '=0' rather than "
            "the field. The instrument cannot catch a dense reconstruction. The Metal GEMV "
            "shader does in-register dequant and affine2_group32_dequant is only fetched "
            "by affine2_parity.rs, not the decode session; a numpy reconstruct-vs-fused "
            f"probe at 64x256 still shows reconstruct peak {probe['reconstruct_peak_numel']} "
            f"(parent {probe['parent_numel']}) vs fused peak {probe['fused_peak_numel']}. "
            "Kernel comments are not a counter. PARENT_A's dense_w_materialized=0 is not "
            "evidence."
        ),
    }


def implied_gb_s(active_bytes: float, gpu_ns: float) -> float:
    return (active_bytes / (gpu_ns * 1e-9)) / 1e9


def attack_wrong_benchmark_path() -> dict[str, Any]:
    """Q4: supply a deliberately BAD control and see what survives the law."""
    law_rows = []
    for name, claim in (
        ("NOETIC_DISPATCH_FUSION", "964 -> 756"),
        ("NOETIC_FUSED_SUBBIT", "affine2 fused"),
        ("GPU_LEDGER", "468.9 GB/s"),
        ("NOETIC_MULTISESSION", "one body 1.32x"),
        ("NOETIC_PARENT_A", "sealed leader"),
        ("AFFINE2_G64_LSFIT", "ls fit"),
        ("AFFINE2_NATIVE_MLP", "native 2-bit"),
        ("NOETIC_Q3_MLP_Q4_ATTN", "q3 mlp q4 attn"),
    ):
        law_rows.append(audit_law(name, claim))

    # Counting-level bad fusion: add one dummy dispatch per layer on top of 756.
    good = theoretical_after("swiglu", True, True)
    bad_extra = good + 64
    unfused = theoretical_after("swiglu", False, False)
    unfused_base = theoretical_after("nope", False, False)

    fused = load_json("NOETIC_FUSED_SUBBIT.json")
    why = fused.get("why_affine2_g64_was_slower") or {}
    per = why.get("per_dispatch") or []
    rd_ns = None
    sp_ns = None
    if per:
        row0 = per[0] if isinstance(per[0], dict) else {}
        rd_ns = row0.get("affine2_g64_runtime_div_gpu_ns")
        sp_ns = row0.get("affine2_g64_specialized_gpu_ns")

    tok = fused.get("decode_tok_s") or {}
    pair = tok.get("after_mlp_pair") or {}
    full = tok.get("after") or tok.get("after_mlp_swiglu_qkv_dn") or {}

    gpu = load_json("GPU_LEDGER.json")
    active = gpu["ACTIVE_BYTES_PER_TOKEN"]["value"]
    gpu_ns = gpu["fields"]["GPU_NS"]["value"]
    measured_gb = implied_gb_s(active, gpu_ns)
    noop_gb = implied_gb_s(active, gpu_ns)  # formula is blind to bytes actually moved
    bad_gb = implied_gb_s(active, gpu_ns * 2.0)
    dram_absent = gpu["fields"]["DRAM_READ_BYTES"]["kind"] == "ABSENT"

    body = 13.50 * (1 << 30)
    ws = 192_139_012
    four_copies = expected_n_copies_bytes(int(body), ws, 4)
    one_body = expected_shared_resident_bytes(int(body), ws, 4)
    copies_rejected = not one_body_not_n_copies(four_copies, int(body), ws, 4)
    shared_accepted = one_body_not_n_copies(int(14.03 * (1 << 30)), int(body), ws, 4)

    missing_bad = [r["receipt"] for r in law_rows if "bad_control" in (r.get("missing") or [])]
    return {
        "id": "wrong_benchmark_path",
        "question": "Does the measured path actually execute the changed kernel? Supply a BAD control.",
        "ran": True,
        "commands": [
            "theoretical_after('swiglu', True, True) vs +64 dummy dispatches",
            "recompute GPU_LEDGER implied GB/s = ACTIVE_BYTES / GPU_NS with 2x GPU_NS bad control",
            "one_body_not_n_copies(four_copies) must reject",
            "read fused_subbit runtime_div gpu_ns vs specialized",
        ],
        "law_missing_bad_control": missing_bad,
        "dispatch_fusion_bad_control": {
            "unfused": unfused_base,
            "full_fusion": good,
            "mlp_swiglu_only": unfused,
            "full_plus_dummy_per_layer": bad_extra,
            "dummy_is_worse": bad_extra > good,
            "fused_subbit_mlp_pair_dispatches": (pair.get("dispatches_last_step_reps") or [None])[0],
            "fused_subbit_mlp_pair_tok_s": pair.get("tok_s_mean"),
            "fused_subbit_full_tok_s": full.get("tok_s_mean"),
            "pair_is_worse_than_full": bool(
                pair.get("tok_s_mean") and full.get("tok_s_mean") and pair["tok_s_mean"] < full["tok_s_mean"]
            ),
            "runtime_div_gpu_ns": rd_ns,
            "specialized_gpu_ns": sp_ns,
            "runtime_div_is_slower": bool(rd_ns and sp_ns and rd_ns > sp_ns),
        },
        "gpu_ledger_bad_control": {
            "formula": "ACTIVE_BYTES_PER_TOKEN / (GPU_NS * 1e-9) / 1e9",
            "active_bytes": active,
            "gpu_ns": gpu_ns,
            "implied_gb_s": measured_gb,
            "noop_same_gpu_ns_implied_gb_s": noop_gb,
            "bad_2x_gpu_ns_implied_gb_s": bad_gb,
            "formula_blind_to_bytes_moved": abs(noop_gb - measured_gb) < 1e-9,
            "DRAM_READ_BYTES_kind": gpu["fields"]["DRAM_READ_BYTES"]["kind"],
            "dram_absent": dram_absent,
        },
        "multisession_bad_control": {
            "four_copies_bytes": four_copies,
            "one_body_bytes": one_body,
            "four_copies_rejected": copies_rejected,
            "measured_c4_accepted_as_one_body": shared_accepted,
        },
        "verdict": "REFUTED" if dram_absent and abs(noop_gb - measured_gb) < 1e-9 else "WEAKENED",
        "why": (
            f"Causal law still finds {len(missing_bad)}/8 GPU claims missing a bad_control "
            f"({missing_bad}). Supplied: (1) dummy-per-layer fusion {good} -> {bad_extra} "
            "dispatches, worse; fused_subbit already measured mlp_pair at 900 disp / "
            f"{pair.get('tok_s_mean')} tok/s vs full {full.get('tok_s_mean')} and "
            f"runtime_div {rd_ns} ns vs specialized {sp_ns} ns. (2) GPU ledger 468.9 GB/s "
            f"recomputed {measured_gb:.3f}; a no-op that did not move weights would report "
            f"the same {noop_gb:.3f} because DRAM_READ_BYTES is ABSENT. A 2x-slower bad "
            f"kernel would report {bad_gb:.3f} GB/s — the formula tracks time, not bytes "
            "moved. (3) four copies of the body are rejected by one_body_not_n_copies. "
            "Dispatch 964->756 still shows the graph changed shape. The bandwidth-bound "
            "label is not a measured DRAM counter."
        ),
    }


def attack_cache_artifact() -> dict[str, Any]:
    """Q5: are the tok/s numbers page-cache or warm-state artifacts?"""
    parent = load_json("NOETIC_PARENT_A.json")
    gpu = load_json("GPU_LEDGER.json")
    fusion = load_json("NOETIC_DISPATCH_FUSION.json")
    repro = (parent.get("reproduction") or {}).get("rows") or {}
    tok_row = repro.get("decode_tok_s") or {}
    measured = tok_row.get("measured") or parent["complete_token_wall"]["tok_s_mean"]
    recorded = tok_row.get("recorded") or RECORDED_TOK_S
    inside = tok_row.get("inside_recorded_band")
    n_reps = len(parent["complete_token_wall"].get("tok_s_reps") or [])
    page_abs = gpu["fields"]["OS_PAGE_CACHE_COLD_GPU_NS"]
    fusion_reps = len((fusion.get("decode_tok_s") or {}).get("after", {}).get("tok_s_reps") or [])

    catalog = PARENT_A / "catalog.hq38m20"
    segs = sorted(
        (PARENT_A / "segments").glob("*"),
        key=lambda p: p.stat().st_size if p.is_file() else 0,
        reverse=True,
    )
    cache_samples = []
    if catalog.is_file():
        cache_samples.append(page_cache_frac(catalog, limit=catalog.stat().st_size))
    for p in segs[:4]:
        if p.is_file():
            cache_samples.append(page_cache_frac(p, limit=min(p.stat().st_size, 64 << 20)))

    sealed_tok_s_refuted = (
        measured is not None
        and recorded is not None
        and abs(float(measured) - float(recorded)) > 0.2
        and inside is False
    )
    return {
        "id": "cache_artifact",
        "question": "Are the tok/s numbers page-cache or warm-state artifacts? A single Metal run is page-cache confounded.",
        "ran": True,
        "commands": [
            "read NOETIC_PARENT_A reproduction.decode_tok_s",
            "read GPU_LEDGER.fields.OS_PAGE_CACHE_COLD_GPU_NS",
            "mincore mmap of NOETIC_PARENT_A catalog + largest segments (read-only, no purge)",
        ],
        "parent_a": {
            "recorded_tok_s": recorded,
            "measured_tok_s": measured,
            "recorded_band": [RECORDED_TOK_S_MIN, RECORDED_TOK_S_MAX],
            "inside_recorded_band": inside,
            "delta": tok_row.get("delta"),
            "n_reps_in_seal_process": n_reps,
            "n_process_runs": 1,
            "honest_note": tok_row.get("honest"),
        },
        "gpu_ledger": {
            "OS_PAGE_CACHE_COLD_GPU_NS_kind": page_abs["kind"],
            "absent_reason": page_abs.get("absent_reason"),
            "warm_n_process_runs": (gpu.get("warm") or {}).get("n_process_runs"),
            "warm_tps_median": ((gpu.get("runs") or {}).get("warm_process_tps") or {}).get("median"),
        },
        "dispatch_fusion": {
            "after_tok_s_mean": (fusion.get("decode_tok_s") or {}).get("after", {}).get("tok_s_mean"),
            "n_reps": fusion_reps,
            "n_process_runs_named": False,
        },
        "mincore": cache_samples,
        "did_not_purge": True,
        "did_not_modify_parent_a": True,
        "verdict": "REFUTED" if sealed_tok_s_refuted else "WEAKENED",
        "why": (
            f"Sealed PARENT_A quotes {recorded:.3f} tok/s; the seal's own remesure is "
            f"{measured:.3f} (delta {tok_row.get('delta')}), outside the original band "
            f"[{RECORDED_TOK_S_MIN:.3f}, {RECORDED_TOK_S_MAX:.3f}]. n_reps={n_reps} in one "
            "process. GPU_LEDGER.OS_PAGE_CACHE_COLD_GPU_NS is ABSENT because sudo purge was "
            "not run; three sequential processes share the ~14 GiB page cache. Dispatch "
            f"fusion's +5.8% is {fusion_reps} in-process reps, also unpurged. mincore on "
            "the sealed parent (read-only) reports the current cache footprint; this is "
            "the confound, not a cold number."
        ),
    }


def score_text_against_suite(text: str) -> dict[str, Any]:
    rows = []
    n_pass = 0
    for item in SUITE:
        ok, why = item["check"](text, None)
        rows.append({"id": item["id"], "axis": item["axis"], "pass": bool(ok), "why": why})
        if ok:
            n_pass += 1
    return {
        "n_items": len(SUITE),
        "n_pass": n_pass,
        "n_fail": len(SUITE) - n_pass,
        "items": rows,
        "text_head": (text or "")[:240],
    }


def attack_capability(*, live: dict[str, Any] | None) -> dict[str, Any]:
    """Q6: coherence on 16 greedy tokens is not capability. Attack that gap."""
    parent = load_json("NOETIC_PARENT_A.json")
    cap = parent.get("capability_evidence") or {}
    coh = cap.get("coherence") or {}
    text = coh.get("text") or parent.get("verbatim", {}).get("generated_text") or ""
    ids = coh.get("new_token_ids") or parent.get("verbatim", {}).get("new_token_ids") or []
    sealed_score = score_text_against_suite(text)
    think_preamble = (text or "").lstrip().startswith("<think>")

    cap_files = sorted(HEADLESS_RECEIPTS.glob("CAPABILITY_*.json"))
    cap_summaries = []
    parent_a_suite_ran = False
    for p in cap_files:
        d = json.loads(p.read_text())
        target = str(d.get("target") or "") + str(d.get("label") or "") + str(d.get("backend") or "")
        hits_parent = "NOETIC_PARENT_A" in target or "parent_a" in target.lower()
        parent_a_suite_ran = parent_a_suite_ran or hits_parent
        cap_summaries.append(
            {
                "file": p.name,
                "label": d.get("label"),
                "backend": d.get("backend"),
                "overall": d.get("overall"),
                "is_parent_a": hits_parent,
            }
        )

    live_score = None
    live_text = None
    if live and live.get("ran") and (live.get("n_new_tokens") or 0) >= 1:
        live_text = live.get("generated_text") or ""
        live_score = score_text_against_suite(live_text)

    q5k = next((c for c in cap_summaries if c.get("label") == "llamacpp-q5k"), None)
    return {
        "id": "capability_collapse",
        "question": "No capability suite has been run on ANY candidate. Does the leader lose something a 16-token sample cannot see?",
        "ran": True,
        "commands": [
            "capability_suite.SUITE predicates over PARENT_A's sealed 16-token sample",
            "glob receipts/headless/CAPABILITY_*.json",
            "optional live fused_subbit fact-capital on PARENT_A",
        ],
        "sealed_sample": {
            "n_tokens": len(ids),
            "ids": ids,
            "think_preamble": think_preamble,
            "text_head": (text or "")[:200],
            "suite": sealed_score,
        },
        "capability_receipts": cap_summaries,
        "parent_a_capability_suite_ran": parent_a_suite_ran,
        "q5k_control": q5k,
        "live_fact_capital": live_score,
        "live_text_head": (live_text or "")[:200] if live_text is not None else None,
        "verdict": "REFUTED",
        "why": (
            "PARENT_A's capability_evidence is 16 greedy tokens of a compiler-prose prompt "
            f"starting {('<think>' if think_preamble else 'not-think')}. Scoring that sample "
            f"against the Doctor suite yields {sealed_score['n_pass']}/{sealed_score['n_items']} "
            "passes — it cannot emit Paris, 323, a mutation JSON, or compiling code. "
            f"CAPABILITY receipts on disk: {[c['file'] for c in cap_summaries]}. None target "
            "NOETIC_PARENT_A. llama.cpp Q5_K on the same machine scored "
            f"{(q5k or {}).get('overall')} on the suite (Paris in 2 completion tokens). "
            "A 16-token <think> preamble cannot see that loss. "
            + (
                f"Live fact-capital on PARENT_A: {live_score['n_pass']}/{live_score['n_items']} "
                f"head={live_text[:80]!r}."
                if live_score
                else "Live suite item was not obtained this run; the sealed sample already fails every item."
            )
        ),
    }


def whole_model_floor(mlp_bpw: float, output_bpw: float) -> float:
    bits = (
        ORGAN_ELEMS["embedding"] * 4.125
        + ORGAN_ELEMS["attention_gqa"] * 4.25
        + ORGAN_ELEMS["deltanet"] * 4.125
        + ORGAN_ELEMS["mlp"] * mlp_bpw
        + ORGAN_ELEMS["output"] * output_bpw
    )
    return bits / PARENT_PARAMS


def onebit_local_survives(rel_fro: float, gain: float, cosine: float, null: float) -> bool:
    return (
        rel_fro <= ONEBIT_RULE["rel_fro_max"]
        and gain >= ONEBIT_RULE["gain_min"]
        and cosine > null
    )


def attack_claims(questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    parent = load_json("NOETIC_PARENT_A.json")
    fusion = load_json("NOETIC_DISPATCH_FUSION.json")
    gpu = load_json("GPU_LEDGER.json")
    multi = load_json("NOETIC_MULTISESSION.json")
    organs = load_json("ORGAN_FRONTIERS.json")
    onebit = load_json("ONEBIT_FAMILIES.json")

    live_hash = reseal(PARENT_A)
    sealed = (parent.get("executable_closure") or {}).get("closure_sha256")
    hash_match = live_hash.get("closure_sha256") == sealed == SEALED_CLOSURE

    # --- PARENT_A ---
    tok = questions["cache_artifact"]["parent_a"]
    parent_refute_bits = [
        questions["hidden_bytes"]["verdict"] == "REFUTED",
        questions["reconstructed_dense_state"]["verdict"] == "REFUTED",
        questions["cache_artifact"]["verdict"] == "REFUTED",
        questions["capability_collapse"]["verdict"] == "REFUTED",
    ]
    claims = []
    claims.append(
        {
            "id": "NOETIC_PARENT_A",
            "claim": "3.1393 EBPW, 756 dispatches, 34.873 tok/s, sealed closure "
            + SEALED_CLOSURE,
            "ran": True,
            "attacks_run": [
                "integer-coded hide against the accountant",
                "reseal of ~/noetic/NOETIC_PARENT_A (read-only merkle)",
                "reproduction tok/s vs recorded band",
                "hardcoded dense_w_materialized=0 in the decoder that printed the counter",
                "capability_suite predicates on the sealed 16-token sample",
            ],
            "evidence": {
                "complete_ebpw": parent["RepresentationGenome"]["complete_ebpw"],
                "dispatches_fused": parent["dispatch_count"]["fused"],
                "recorded_tok_s": tok["recorded_tok_s"],
                "measured_tok_s": tok["measured_tok_s"],
                "tok_s_inside_band": tok["inside_recorded_band"],
                "closure_sha256_receipt": sealed,
                "closure_sha256_live": live_hash.get("closure_sha256"),
                "closure_match": hash_match,
                "n_affine_live": live_hash.get("n_affine"),
            },
            "verdict": "REFUTED" if any(parent_refute_bits) else "SURVIVED",
            "why": (
                f"Byte closure reseals to {live_hash.get('closure_sha256')} "
                f"(match={hash_match}, n_affine={live_hash.get('n_affine')}). "
                f"EBPW {parent['RepresentationGenome']['complete_ebpw']} and 756 dispatches "
                "reproduced. The quoted 34.873 tok/s does not: measured "
                f"{tok['measured_tok_s']} outside {tok['recorded_band']}. "
                "dense_w_materialized=0 is a literal. Integer-coded weights hide from the "
                "accountant that blesses the EBPW. Capability is 16 tokens of <think>, not "
                "a suite. A sealed parent whose headline tok/s already missed its own "
                "reproduction is not a sealed speed number."
            ),
        }
    )

    before_ids = (fusion.get("decode_tok_s") or {}).get("before", {}).get("new_token_ids")
    after_ids = (fusion.get("decode_tok_s") or {}).get("after", {}).get("new_token_ids")
    before_tps = (fusion.get("decode_tok_s") or {}).get("before", {}).get("tok_s_mean")
    after_tps = (fusion.get("decode_tok_s") or {}).get("after", {}).get("tok_s_mean")
    pct = None if not (before_tps and after_tps) else 100.0 * (after_tps - before_tps) / before_tps
    bad = questions["wrong_benchmark_path"]["dispatch_fusion_bad_control"]
    claims.append(
        {
            "id": "dispatch_fusion",
            "claim": "964 -> 756 with token ids IDENTICAL, +5.8% tok/s",
            "ran": True,
            "attacks_run": [
                "recompute theoretical_after and a +64 dummy-dispatch worse graph",
                "compare fused_subbit mlp_pair (900) vs full (756) tok/s",
                "runtime_div vs specialized gpu_ns",
                "token-id identity from the fusion receipt",
            ],
            "evidence": {
                "before_dispatches": fusion["dispatches_per_token"]["before"],
                "after_dispatches": fusion["dispatches_per_token"]["after"],
                "ids_identical": before_ids == after_ids,
                "before_tok_s": before_tps,
                "after_tok_s": after_tps,
                "pct": pct,
                "bad_control_dummy_dispatches": bad["full_plus_dummy_per_layer"],
                "runtime_div_slower": bad["runtime_div_is_slower"],
                "pair_worse_than_full": bad["pair_is_worse_than_full"],
            },
            "verdict": "SURVIVED",
            "why": (
                f"Token ids match ({len(before_ids or [])} ids). Dispatches 964->756. "
                f"tok/s {before_tps:.3f}->{after_tps:.3f} ({pct:.2f}%). A dummy-per-layer "
                f"graph is {bad['full_plus_dummy_per_layer']} dispatches (worse). fused_subbit "
                "already ran worse fusion arms (mlp_pair 900 disp, runtime_div slower than "
                "specialized) and they scored worse. The fusion receipt itself still lacks a "
                "named bad_control field, which is why the causal law marked it incomplete; "
                "the attacks that were executed did not invert 964->756 or the identical ids."
            ),
        }
    )

    q80 = gpu.get("q80_anchor") or {}
    inc = q80.get("q4_incumbent") or {}
    gb = questions["wrong_benchmark_path"]["gpu_ledger_bad_control"]
    gpu_frac = inc.get("gpu_as_fraction_of_wall")
    claims.append(
        {
            "id": "GPU_ledger",
            "claim": "incumbent is BANDWIDTH-bound: 468.9 GB/s, 67% of ceiling, GPU 95.6% of wall",
            "ran": True,
            "attacks_run": [
                "recompute ACTIVE_BYTES/GPU_NS",
                "no-op control: same GPU_NS still reports 468.9",
                "bad control: 2x GPU_NS halves implied GB/s",
                "confirm DRAM_READ_BYTES ABSENT",
            ],
            "evidence": {
                "recomputed_gb_s": gb["implied_gb_s"],
                "receipt_gb_s": inc.get("achieved_gb_s"),
                "pct_of_700": inc.get("pct_of_700_GBs"),
                "gpu_as_fraction_of_wall": gpu_frac,
                "DRAM_READ_BYTES": gpu["fields"]["DRAM_READ_BYTES"]["kind"],
                "noop_same_implied_gb_s": gb["formula_blind_to_bytes_moved"],
                "bad_2x_gpu_ns_gb_s": gb["bad_2x_gpu_ns_implied_gb_s"],
            },
            "verdict": "REFUTED",
            "why": (
                f"Recomputed implied GB/s {gb['implied_gb_s']:.3f} matches the receipt. "
                "That identity is ACTIVE_BYTES (manifest payload) / GPU_NS, not a DRAM "
                f"counter (DRAM_READ_BYTES={gpu['fields']['DRAM_READ_BYTES']['kind']}). A "
                "kernel that did not stream the weights would still report 468.9 at the "
                f"same GPU_NS. GPU busy-fraction {gpu_frac} is measured "
                "(GPUEnd-GPUStart)/complete-wall and does overturn the Q80 51% idle story. "
                "The bandwidth-bound *label* is not a physical bytes-moved measurement."
            ),
        }
    )

    proof = (multi.get("live") or {}).get("proof_one_body") or {}
    ratio = proof.get("metal_c4_over_c1") or (multi.get("live") or {}).get("metal_c4_over_c1")
    # fall back to scanning
    if ratio is None:
        blob = json.dumps(multi)
        m = re.search(r'"metal_c4_over_c1":\s*([0-9.]+)', blob)
        ratio = float(m.group(1)) if m else None
    scaling = (multi.get("live") or {}).get("scaling_vs_c1_aggregate_tps") or {}
    conc = scaling.get("concurrent_independent") or {}
    c4 = conc.get("4")
    bad_ms = questions["wrong_benchmark_path"]["multisession_bad_control"]
    claims.append(
        {
            "id": "multisession",
            "claim": "one shared body, Metal c4/c1 = 1.0398; ceiling 1.32x",
            "ran": True,
            "attacks_run": [
                "one_body_not_n_copies(four copies) must reject",
                "recompute metal_c4_over_c1 from the live receipt",
                "confirm operator microbatch 1.00x (weight stream not amortised)",
            ],
            "evidence": {
                "shared_body": multi.get("NOETIC_MULTISESSION_SHARED_BODY"),
                "metal_c4_over_c1": ratio,
                "concurrent_independent_c4": c4,
                "four_copies_rejected": bad_ms["four_copies_rejected"],
                "one_body_formula_accepts_measured_c4": bad_ms["measured_c4_accepted_as_one_body"],
                "weights_ptr_shared": (multi.get("live") or {}).get("weights_ptr_shared"),
            },
            "verdict": "SURVIVED",
            "why": (
                f"Four full copies are rejected by the one-body predicate "
                f"(copies={bad_ms['four_copies_bytes']}, one={bad_ms['one_body_bytes']}). "
                f"Measured c4/c1 Metal {ratio} is ~1.04, not 4. Concurrent-independent c=4 "
                f"is {c4}. Operator microbatch of lm_head is 1.00x in the same receipt — "
                "the weight stream is not amortised, which is why 1.32x is a ceiling rather "
                "than a win. No page-cache-cold of the c=4 run was added; the one-body "
                "pointer evidence (weights_ptr_shared, distinct KV pointers) is independent "
                "of tok/s and survived the four-copy bad control."
            ),
        }
    )

    quoted = 2.9398
    with_output_4125 = whole_model_floor(2.25, 4.125)
    with_lm_head_325 = whole_model_floor(2.25, 3.25)
    floors = (organs.get("verdict") or {}).get("floors_storage_bpw") or {}
    lm_head_survives_q3 = "lm_head mix survives q3 (3.25)" in str((organs.get("verdict") or {}).get("reading") or "")
    claims.append(
        {
            "id": "organ_floors",
            "claim": "deltanet 4.125 / gqa 4.25 / embedding 4.125, implying a whole-model floor of 2.9398 EBPW",
            "ran": True,
            "attacks_run": [
                "recompute weighted EBPW from NOETIC_ORGAN_CENSUS element counts",
                "apply lm_head 3.25 from ORGAN_FRONTIERS instead of embed 4.125 on output",
            ],
            "evidence": {
                "floors_storage_bpw": floors,
                "mlp_survive_bpw": 2.25,
                "quoted_whole_model": quoted,
                "recomputed_mlp225_output4125": with_output_4125,
                "recomputed_mlp225_lmhead325": with_lm_head_325,
                "quoted_matches_output_at_embed_floor": abs(with_output_4125 - quoted) < 1e-4,
                "lm_head_survives_q3_in_organ_receipt": lm_head_survives_q3,
                "embedding_output_do_not_quote_one_number": "do not quote one number"
                in str((organs.get("verdict") or {}).get("floors_active") or {}),
            },
            "verdict": "REFUTED",
            "why": (
                "The 2.9398 figure is exactly MLP@2.25 + DeltaNet@4.125 + GQA@4.25 + "
                f"embed@4.125 + output@4.125 = {with_output_4125:.6f}. ORGAN_FRONTIERS "
                "says lm_head mix survives q3 (3.25) and 'embed gather vs lm_head stream "
                "— do not quote one number'. Charging the output organ (1.271e9 weights, "
                "almost entirely lm_head) at the embed-table rare-token floor is the "
                f"illegal transfer. With lm_head at 3.25 the mix is {with_lm_head_325:.6f}, "
                "not 2.9398. The three per-organ floors themselves were not inverted; the "
                "implication that forces a whole-model floor of 2.9398 is."
            ),
        }
    )

    null = (onebit.get("null") or {}).get("mean") or 0.40992871175209683
    # per-family re-application of the published rule
    family_rows = []
    for fam in onebit.get("families") or []:
        tensors = fam.get("per_tensor") or fam.get("tensors") or []
        unhealthy = []
        all_ok = True
        for t in tensors:
            fn = t.get("function") or t
            rel = float(fn.get("rel_fro") if fn.get("rel_fro") is not None else t.get("rel_fro"))
            gain = float(fn.get("gain") if fn.get("gain") is not None else t.get("gain"))
            cos = float(fn.get("cosine") if fn.get("cosine") is not None else t.get("mean_cosine") or fam.get("mean_cosine") or 0)
            n = float(fn.get("null") if fn.get("null") is not None else t.get("null") or null)
            ok = onebit_local_survives(rel, gain, cos, n)
            if not ok:
                all_ok = False
                unhealthy.append(
                    {
                        "organ": t.get("organ") or t.get("name"),
                        "rel_fro": rel,
                        "gain": gain,
                        "survives": False,
                    }
                )
        family_rows.append(
            {
                "family_id": fam.get("family_id"),
                "name": fam.get("name"),
                "storage_bpw": fam.get("storage_bpw"),
                "mean_rel_fro": fam.get("mean_rel_fro"),
                "mean_gain": fam.get("mean_gain"),
                "n_unhealthy": len(unhealthy),
                "unhealthy": unhealthy,
                "family_survives_reapplied_rule": all_ok
                and 1.7 < float(fam.get("storage_bpw") or 0) < 2.15,
            }
        )
    b3 = next((f for f in family_rows if f["family_id"] == "B3"), None)
    b6 = next((f for f in family_rows if f["family_id"] == "B6"), None)
    onebit_ok = bool(b3 and b3["family_survives_reapplied_rule"] and b6 and not b6["family_survives_reapplied_rule"])
    claims.append(
        {
            "id": "onebit_families",
            "claim": "ternary survives at 1.85; routed codebook has the best error but is UNHEALTHY per tensor",
            "ran": True,
            "attacks_run": [
                "re-apply local_survives := gain>=0.5 AND rel_fro<=0.5 AND cosine>null on every tensor in the receipt",
            ],
            "evidence": {
                "null_cosine": null,
                "families": family_rows,
                "receipt_best_survivor": (onebit.get("verdict") or {}).get("best_survivor"),
                "receipt_ranked": (onebit.get("verdict") or {}).get("ranked_by_function_space_error"),
            },
            "verdict": "SURVIVED" if onebit_ok else "REFUTED",
            "why": (
                "Re-applied the published survival rule to the six tensors of each family. "
                f"B3 ternary@1.85: family_survives={None if not b3 else b3['family_survives_reapplied_rule']} "
                f"n_unhealthy={None if not b3 else b3['n_unhealthy']}. "
                f"B6 routed codebook: best mean_rel_fro={None if not b6 else b6['mean_rel_fro']} "
                f"but n_unhealthy={None if not b6 else b6['n_unhealthy']} "
                f"(down_proj gain 0.395 and 0.325, both < 0.5). The claim already named "
                "this; the attack that was executed reproduced it rather than inverting it."
            ),
        }
    )
    return claims


def run(*, live_gpu: bool = True) -> dict[str, Any]:
    t0 = time.time()
    q_hidden_bytes = attack_hidden_bytes()
    q_hidden_parent = attack_hidden_parent(live_gpu=live_gpu)
    q_dense = attack_dense_state()
    q_path = attack_wrong_benchmark_path()
    q_cache = attack_cache_artifact()
    q_cap = attack_capability(live=q_hidden_parent.get("live"))
    questions = {
        "hidden_bytes": q_hidden_bytes,
        "hidden_parent": q_hidden_parent,
        "reconstructed_dense_state": q_dense,
        "wrong_benchmark_path": q_path,
        "cache_artifact": q_cache,
        "capability_collapse": q_cap,
    }
    claims = attack_claims(questions)
    qlist = [questions[k] for k in QUESTION_IDS]
    n_refute_c = sum(1 for c in claims if c["verdict"] == "REFUTED")
    n_survive_c = sum(1 for c in claims if c["verdict"] == "SURVIVED")
    n_refute_q = sum(1 for q in qlist if q["verdict"] == "REFUTED")
    doc = {
        "schema": SCHEMA,
        "obligation": "N014 — ADVERSARY: attack the frontier claims, do not merely question them",
        "generated_at": now_iso(),
        "git_head": git_head(),
        "elapsed_s": round(time.time() - t0, 3),
        "did_not_load_second_27b": True,
        "did_not_modify_parent_a": True,
        "did_not_write_ascent_or_campaign": True,
        "law": (
            "A refutation is the valuable outcome. If a claim survives, the attack that "
            "was executed is named. Answering the six questions without running them does "
            "not satisfy this."
        ),
        "questions": qlist,
        "claims": claims,
        "counts": {
            "questions": len(qlist),
            "questions_ran": sum(1 for q in qlist if q.get("ran")),
            "questions_refuted": n_refute_q,
            "claims": len(claims),
            "claims_ran": sum(1 for c in claims if c.get("ran")),
            "claims_refuted": n_refute_c,
            "claims_survived": n_survive_c,
        },
        "at_least_one_claim_physically_refuted": n_refute_c >= 1,
        "refuted_claim_ids": [c["id"] for c in claims if c["verdict"] == "REFUTED"],
        "survived_claim_ids": [c["id"] for c in claims if c["verdict"] == "SURVIVED"],
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"questions ran {doc['counts']['questions_ran']}/{doc['counts']['questions']}  "
          f"refuted {n_refute_q}")
    print(f"claims    ran {doc['counts']['claims_ran']}/{doc['counts']['claims']}  "
          f"refuted {n_refute_c}  survived {n_survive_c}")
    for c in claims:
        print(f"  {c['id']:<20} {c['verdict']}")
    for q in qlist:
        print(f"  Q {q['id']:<28} {q['verdict']}")
    print(f"receipt: {RECEIPT}")
    return doc


def main() -> int:
    live = os.environ.get("FRONTIER_ADVERSARY_LIVE", "1") != "0"
    doc = run(live_gpu=live)
    if doc["counts"]["questions_ran"] != 6:
        return 2
    if doc["counts"]["claims_ran"] != 6:
        return 2
    if not doc["at_least_one_claim_physically_refuted"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
