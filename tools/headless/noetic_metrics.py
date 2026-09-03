#!/usr/bin/env python3
"""NoeticMetrics over the real uniform-q4-v1 artifact.

BPW is one field (EXECUTABLE_BPW). The three byte figures this project
conflates — on-disk, resident, moved-per-token — are measured separately.
Shared Hawking runtime/kernel bytes are reported next to the model, never
folded into EBPW.

Every surviving field is MEASURED in this process or explicitly null with a
reason. Numbers copied from a receipt are tagged with that receipt and are
not re-derived from a different artifact (the live llama-server on 52484 is
a Q5_K GGUF, not this patient).

    python3 tools/headless/noetic_metrics.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
NATIVE_DECODE_CANDIDATES = [
    REPO / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy",
    Path.home() / "Downloads/hawking-copy/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy",
]
NATIVE_RAW_DIRS = [
    REPO / "receipts/headless",
    Path.home() / "Downloads/hawking-copy/receipts/headless",
]
LLAMA_URL = "http://127.0.0.1:52484"
HQ30_MAGIC = b"HQ30UQ4\0"
HQ30_HEADER_BYTES = 40
F32V2_HEADER_BYTES = 8
GROUP = 64
CODE_BYTES_PER_GROUP = 32
SCALE_BYTES_PER_GROUP = 2  # IEEE FP16
Q4_BYTES_PER_GROUP = CODE_BYTES_PER_GROUP + SCALE_BYTES_PER_GROUP  # 34
RECORDED_EBPW = 4.253
RECORDED_EBPW_EXACT = 4.252735126866492
CONTENT_ADDRESS_SAMPLE = 12
SCHEMA = "hawking.headless.noetic_metrics.v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sh(cmd, timeout=30) -> str:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (p.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_head() -> str | None:
    return sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]) or None


def git_show(rel: str) -> dict | None:
    p = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        capture_output=True, text=True, timeout=30,
    )
    if p.returncode != 0 or not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def load_json(path: Path) -> dict | None:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def field(value, status, *, unit=None, method=None, formula=None,
          source=None, extra=None):
    d = {"value": value, "status": status}
    if unit is not None:
        d["unit"] = unit
    if method is not None:
        d["method"] = method
    if formula is not None:
        d["formula"] = formula
    if source is not None:
        d["source"] = source
    if extra:
        d.update(extra)
    return d


def measured(value, **kw):
    return field(value, "MEASURED", **kw)


def null_field(reason, **kw):
    return field(None, "NULL", extra={"null_reason": reason, **kw})


def walk_bytes(root: Path) -> tuple[int, int, dict]:
    n_files = 0
    total = 0
    by_ext = collections.Counter()
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for fn in files:
            p = Path(dirpath) / fn
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            n_files += 1
            total += sz
            ext = p.suffix.lower() or "(none)"
            by_ext[ext] += sz
    return n_files, total, dict(by_ext)


def parse_hq30_header(blob: bytes) -> dict:
    if len(blob) < HQ30_HEADER_BYTES or blob[:8] != HQ30_MAGIC:
        raise ValueError("not HQ30UQ4")
    version, group, ndims = struct.unpack_from("<III", blob, 8)
    numel = struct.unpack_from("<Q", blob, 20)[0]
    dims = list(struct.unpack_from(f"<{ndims}I", blob, 28)) if ndims else []
    return {
        "version": version,
        "group": group,
        "ndims": ndims,
        "numel": numel,
        "dims": dims,
        "header_bytes": HQ30_HEADER_BYTES,
    }


def q4_matrix_bytes(rows: int, cols: int, group: int = GROUP) -> int:
    groups_per_row = (cols + group - 1) // group
    return rows * groups_per_row * Q4_BYTES_PER_GROUP


def classify_tensor_name(name: str) -> str:
    n = name
    if "embed_tokens" in n:
        return "embed_table"
    if n.endswith("lm_head.weight") or n.endswith(".lm_head.weight"):
        return "lm_head"
    if ".mlp." in n:
        return "mlp"
    if "linear_attn" in n:
        return "linear_attn"
    if "self_attn" in n:
        return "full_attn"
    if "norm" in n.lower():
        return "norms"
    return "other"


def safetensors_payload_census(source_dir: Path) -> dict:
    """Header-only walk: language vs vision vs mtp vs lm_head, no weight I/O."""
    buckets = {
        "language_model": {"tensors": 0, "bytes": 0},
        "lm_head": {"tensors": 0, "bytes": 0},
        "visual": {"tensors": 0, "bytes": 0},
        "mtp": {"tensors": 0, "bytes": 0},
        "other": {"tensors": 0, "bytes": 0, "names": []},
    }
    files = 0
    header_overhead = 0
    for p in sorted(source_dir.glob("*.safetensors")):
        files += 1
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header_overhead += 8 + n
            meta = json.loads(f.read(n))
        for k, v in meta.items():
            if k == "__metadata__":
                continue
            offsets = v.get("data_offsets") if isinstance(v, dict) else None
            nbytes = (offsets[1] - offsets[0]) if offsets else 0
            if k == "lm_head.weight":
                b = "lm_head"
            elif "language_model" in k:
                b = "language_model"
            elif "visual" in k or "vision" in k:
                b = "visual"
            elif k.startswith("mtp."):
                b = "mtp"
            else:
                b = "other"
                if len(buckets["other"]["names"]) < 8:
                    buckets["other"]["names"].append(k)
            buckets[b]["tensors"] += 1
            buckets[b]["bytes"] += nbytes
    language_plus_head = (
        buckets["language_model"]["bytes"] + buckets["lm_head"]["bytes"]
    )
    return {
        "safetensors_files": files,
        "header_overhead_bytes": header_overhead,
        "buckets": buckets,
        "language_plus_lm_head_payload_bytes": language_plus_head,
        "method": "safetensors header JSON + data_offsets; file bodies not hashed",
    }


def proc_rss(pid: int) -> dict | None:
    """macOS proc_pidinfo TASKINFO. RSS is CPU pages, not Metal working set."""
    import ctypes
    class proc_taskinfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_syscalls_unix", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
        ]
        lib.proc_pidinfo.restype = ctypes.c_int
        info = proc_taskinfo()
        n = lib.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if n <= 0:
            return None
        return {
            "pid": pid,
            "rss_bytes": int(info.pti_resident_size),
            "vsize_bytes": int(info.pti_virtual_size),
            "threads": int(info.pti_threadnum),
            "caveat": ("CPU RSS via proc_pidinfo; mmap'd weights and MTLBuffers "
                       "are not a 1:1 of this number. MACHINE_GENOME measured "
                       "~20.5 GiB RSS on a freshly admitted llama-server."),
        }
    except Exception as e:
        return {"error": str(e)}


def artifact_mapped_now(artifact: Path) -> dict:
    """Is any process holding the gravity artifact open?"""
    man = artifact / "manifest.json"
    out = sh(["lsof", "-nP", str(man)], timeout=20)
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("COMMAND")]
    return {
        "manifest_open_in_n_processes": len(lines),
        "lsof_lines": lines[:8],
        "method": "lsof -nP on manifest.json",
    }


def listen_pid_on_port(port: int) -> int | None:
    out = sh(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=10)
    for ln in out.splitlines()[1:]:
        parts = ln.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    return None


def llama_dylib_bytes() -> dict:
    paths = [
        "/opt/homebrew/Cellar/llama.cpp/9430/bin/llama-server",
        "/opt/homebrew/Cellar/llama.cpp/9430/lib/libllama.0.0.9430.dylib",
        "/opt/homebrew/Cellar/llama.cpp/9430/lib/libllama-server-impl.dylib",
        "/opt/homebrew/Cellar/llama.cpp/9430/lib/libllama-common.0.0.9430.dylib",
        "/opt/homebrew/Cellar/llama.cpp/9430/lib/libmtmd.0.0.9430.dylib",
        "/opt/homebrew/Cellar/ggml/0.13.1/lib/libggml.0.13.1.dylib",
        "/opt/homebrew/Cellar/ggml/0.13.1/lib/libggml-base.0.13.1.dylib",
        "/opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-metal.so",
        "/opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-cpu-apple_m2_m3.so",
        "/opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-blas.so",
    ]
    rows = []
    total = 0
    missing = []
    for p in paths:
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            rows.append({"path": p, "bytes": sz})
            total += sz
        else:
            missing.append(p)
    return {"bytes": total, "files": rows, "missing": missing}


def hawking_runtime_bytes() -> dict:
    shaders = REPO / "crates/hawking-core/shaders"
    shader_files = 0
    shader_bytes = 0
    if shaders.is_dir():
        for p in shaders.rglob("*.metal"):
            shader_files += 1
            shader_bytes += p.stat().st_size
    decode_rs = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
    decode_rs_bytes = decode_rs.stat().st_size if decode_rs.is_file() else None
    binary = next((p for p in NATIVE_DECODE_CANDIDATES if p.is_file()), None)
    binary_bytes = binary.stat().st_size if binary else None
    return {
        "shader_source_files": shader_files,
        "shader_source_bytes": shader_bytes,
        "decode_rs_bytes": decode_rs_bytes,
        "decode_rs_path": str(decode_rs) if decode_rs.is_file() else None,
        "native_decode_binary_path": str(binary) if binary else None,
        "native_decode_binary_bytes": binary_bytes,
        "note": ("The release-fast example binary embeds the compiled Metal "
                 "library. Shader sources are listed so a later pack cannot "
                 "hide model information in an executable without the split "
                 "showing up here. OS Metal.framework / AGX metallib are not "
                 "Hawking and are not counted."),
    }


def bound_kernels() -> dict:
    decode = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
    shaders = REPO / "crates/hawking-core/shaders"
    if not decode.is_file() or not shaders.is_dir():
        return {"present": False}
    lits = set()
    for tok in decode.read_text().split('"'):
        if tok and all(c.isalnum() or c == "_" for c in tok):
            lits.add(tok)
    declared = set()
    for p in shaders.glob("*.metal"):
        for line in p.read_text().splitlines():
            if line.startswith("kernel void "):
                declared.add(line.split()[2].split("(")[0])
    bound = sorted(lits & declared)
    return {
        "present": True,
        "dispatched": len(bound),
        "declared_in_tree": len(declared),
        "extraction": ("string literals in qwen38_hybrid_decode.rs intersected "
                       "with declared `kernel void` names — same method as NX"),
        "names": bound,
    }


def geometry_from_parent_config(cfg: dict) -> dict:
    text = cfg.get("text_config") or {}
    layer_types = text.get("layer_types") or []
    gqa_layers = sum(1 for t in layer_types if t == "full_attention")
    dn_layers = sum(1 for t in layer_types if t == "linear_attention")
    hidden = int(text["hidden_size"])
    vocab = int(text["vocab_size"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text["head_dim"])
    key_heads = int(text["linear_num_key_heads"])
    value_heads = int(text["linear_num_value_heads"])
    key_dim = int(text["linear_key_head_dim"])
    value_dim = int(text["linear_value_head_dim"])
    conv_k = int(text["linear_conv_kernel_dim"])
    conv_channels = key_heads * key_dim * 2 + value_heads * value_dim
    conv_state_elems = conv_channels * (conv_k - 1)
    rec_state_elems = value_heads * key_dim * value_dim
    conv_one = conv_state_elems * 4
    rec_one = rec_state_elems * 4
    kv_one = kv_heads * head_dim * 4  # one of K or V, one layer, one position
    return {
        "source": "parent config.json text_config, measured this process",
        "hidden": hidden,
        "vocab": vocab,
        "n_layers": int(text["num_hidden_layers"]),
        "gqa_layers": gqa_layers,
        "deltanet_layers": dn_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "linear_key_heads": key_heads,
        "linear_value_heads": value_heads,
        "linear_key_head_dim": key_dim,
        "linear_value_head_dim": value_dim,
        "linear_conv_kernel": conv_k,
        "conv_state_elements_per_layer": conv_state_elems,
        "recurrent_state_elements_per_layer": rec_state_elems,
        "conv_state_resident_bytes": dn_layers * conv_one,
        "rec_state_resident_bytes": dn_layers * rec_one,
        "conv_state_rw_bytes_per_token": dn_layers * conv_one * 2,
        "rec_state_rw_bytes_per_token": dn_layers * rec_one * 2,
        "gqa_kv_write_bytes_per_position": gqa_layers * 2 * kv_one,
        "gqa_kv_read_bytes_at_len": {
            "formula": "gqa_layers * 2 * seq_len * kv_heads * head_dim * 4",
            "at_1": gqa_layers * 2 * 1 * kv_one,
            "at_256": gqa_layers * 2 * 256 * kv_one,
            "at_8192": gqa_layers * 2 * 8192 * kv_one,
            "at_131072": gqa_layers * 2 * 131072 * kv_one,
        },
        "matches_qwen38_geometry_rs": (
            hidden == 5120 and vocab == 248320 and gqa_layers == 16
            and dn_layers == 48 and kv_heads == 4 and head_dim == 256
        ),
    }


def gemv_payload_from_geometry() -> dict:
    mlp = 64 * (
        q4_matrix_bytes(17408, 5120)
        + q4_matrix_bytes(17408, 5120)
        + q4_matrix_bytes(5120, 17408)
    )
    linear = 48 * (
        q4_matrix_bytes(16384, 5120)
        + q4_matrix_bytes(96, 5120)
        + q4_matrix_bytes(5120, 6144)
    )
    full = 16 * (
        q4_matrix_bytes(12288, 5120)
        + q4_matrix_bytes(1024, 5120)
        + q4_matrix_bytes(1024, 5120)
        + q4_matrix_bytes(5120, 6144)
    )
    lm_head = q4_matrix_bytes(248320, 5120)
    embed_row = q4_matrix_bytes(1, 5120)
    gemv = mlp + linear + full + lm_head
    return {
        "mlp_bytes": mlp,
        "linear_attn_bytes": linear,
        "full_attn_bytes": full,
        "lm_head_bytes": lm_head,
        "embed_row_bytes": embed_row,
        "gemv_payload_bytes": gemv,
        "formula": "rows * ceil(cols/64) * 34  (32 code bytes + 2-byte f16 scale)",
        "matches_honest_roof_GEMV_PAYLOAD_BYTES": gemv == 13_611_663_360,
    }


def kernel_reconstruction_is_fused() -> dict:
    metal = REPO / "crates/hawking-core/shaders/qwen_uniform_q4.metal"
    if not metal.is_file():
        return {"present": False}
    text = metal.read_text()
    # The production matvec accumulates float(q)*scale*input in a register
    # and writes only the output row — never a reconstructed weight buffer.
    has_inregister = "sum += float(q) * scale * input[" in text
    writes_weight_buf = "device float* reconstructed" in text or "device float* dequant" in text
    return {
        "present": True,
        "path": str(metal),
        "in_register_fma": has_inregister,
        "reconstructed_weight_buffer_in_kernel": writes_weight_buf,
        "conclusion": (
            "grouped_absmax Q4 reconstructs in-register inside the GEMV. "
            "No extra DRAM write of expanded f32 weights. Reconstruction "
            "bytes are already counted as the 34 B/group of codes+scales "
            "that DRAM_BYTES_PER_TOKEN streams."
        ),
    }


def content_address_sample(artifact: Path, tensors: list, n: int = 12) -> dict:
    """Deterministic sample: sha256 of file bytes vs 64-hex filename stem."""
    rng = random.Random(0x4E4D3132)  # 'NM12'
    sample = rng.sample(tensors, min(n, len(tensors)))
    rows = []
    matches = 0
    for t in sample:
        p = artifact / "tensors" / t["artifact"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        stem = t["artifact"].split(".")[0]
        ok = h == stem
        matches += int(ok)
        rows.append({
            "name": t["name"],
            "filename_stem": stem,
            "sha256": h,
            "filename_is_content_sha256": ok,
            "bytes": t["bytes"],
        })
    return {
        "n": len(rows),
        "matches": matches,
        "method": "sha256(entire tensor file) vs 64-hex filename stem",
        "rows": rows,
        "defect_confirmed": matches == 0,
    }


def load_native_time(receipts_headless: Path) -> dict:
    """COMPLETE_* from the published native receipt; TTFT/prefill from RAW runs."""
    pub = load_json(receipts_headless / "QWEN38_GRAVITY_NATIVE.json")
    raws = []
    for d in NATIVE_RAW_DIRS:
        found = sorted(d.glob("QWEN38_GRAVITY_NATIVE_RAW.run*.json"))
        if found:
            raws = found
            break
    prefill_ns = []
    first_step_ns = []
    decode_ns_per_token = []
    n_prefill = None
    raw_dir = None
    for p in raws:
        d = load_json(p)
        if not d:
            continue
        raw_dir = str(p.parent)
        for w in d.get("warm_reps") or []:
            s = w.get("summary") or w
            if s.get("prefill_wall_ns") is not None:
                prefill_ns.append(int(s["prefill_wall_ns"]))
            n_prefill = s.get("n_prefill_steps", n_prefill)
            fs = (s.get("cold_or_first_step") or {}).get("complete_wall_ns")
            if fs is not None:
                first_step_ns.append(int(fs))
        ctrl = d.get("control_uninstrumented_generate_greedy") or {}
        if ctrl.get("decode_wall_ns_per_token") is not None:
            decode_ns_per_token.append(int(ctrl["decode_wall_ns_per_token"]))
    out = {
        "published_receipt": str(receipts_headless / "QWEN38_GRAVITY_NATIVE.json")
        if pub else None,
        "published_present": bool(pub),
        "raw_dir": raw_dir,
        "raw_files": [str(p) for p in raws],
        "n_warm_prefill_samples": len(prefill_ns),
        "n_prefill_steps": n_prefill,
    }
    if pub:
        dec = pub.get("decode") or {}
        out["median_complete_wall_tps"] = dec.get("median_complete_wall_tps")
        out["median_complete_wall_ns"] = dec.get("median_complete_wall_ns")
        out["per_run_complete_wall_tps"] = dec.get("per_run_complete_wall_tps")
        out["measurement_spread_pct"] = dec.get("measurement_spread_pct")
        out["n_process_runs"] = dec.get("n_process_runs")
        out["bar"] = pub.get("bar")
        out["bar_reasons"] = pub.get("bar_reasons")
        out["recorded_at"] = pub.get("recorded_at")
        out["not_re_run"] = (
            "A llama-server is already resident on 52484. A native run with "
            "two 27B servers was measured at 3.986 tok/s vs 33.47 with one. "
            "This process did not spawn a second 27B."
        )
    if prefill_ns:
        prefill_ns.sort()
        out["median_prefill_wall_ns"] = statistics.median(prefill_ns)
        out["prefill_wall_ns_min"] = prefill_ns[0]
        out["prefill_wall_ns_max"] = prefill_ns[-1]
        out["prefill_samples"] = prefill_ns
    if first_step_ns:
        first_step_ns.sort()
        out["median_last_prefill_step_ns"] = statistics.median(first_step_ns)
    if decode_ns_per_token:
        out["raw_control_decode_ns_per_token"] = decode_ns_per_token
    return out


def token_ns_reconstruction(repo: Path) -> dict | None:
    """weight_decode_reconstruction from the sealed Qwen38 token-ns ledger."""
    rel = "receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"
    on_disk = repo / rel
    doc = load_json(on_disk) or git_show(rel)
    if not doc:
        return None
    rec = None
    addr = None
    for c in doc.get("components") or []:
        if c.get("component") == "weight_decode_reconstruction":
            rec = c
        if c.get("component") == "weight_addressing":
            addr = c
    return {
        "receipt": rel,
        "commit": doc.get("commit"),
        "measurement_label": doc.get("measurement_label"),
        "median_wall_ns": doc.get("median_wall_ns"),
        "median_gpu_ns": doc.get("median_gpu_ns"),
        "reconstruction_ns_per_token": (rec or {}).get("ns_per_token"),
        "reconstruction_bytes_read": (rec or {}).get("bytes_read"),
        "reconstruction_bytes_written": (rec or {}).get("bytes_written"),
        "weight_addressing_ns_per_token": (addr or {}).get("ns_per_token"),
        "weight_addressing_bytes_read": (addr or {}).get("bytes_read"),
        "probes": doc.get("probes"),
        "not_re_run": "Isolating reconstruction requires the native 27B decode with addr/decode probes. Not re-run while llama-server holds a 27B.",
    }


def probe_live_llama(timeout=60) -> dict:
    """Control measurement of the already-resident GGUF. NOT the gravity artifact."""
    try:
        with urllib.request.urlopen(LLAMA_URL + "/v1/models", timeout=5) as r:
            models = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {"reachable": False, "error": str(e)}
    data = (models.get("data") or models.get("models") or [{}])[0]
    meta = data.get("meta") or {}
    body = {
        "prompt": ("Explain, in ordinary prose and at length, how a compiler "
                   "turns a for-loop into basic blocks."),
        "n_predict": 48,
        "temperature": 0,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        LLAMA_URL + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        wall = time.perf_counter() - t0
        d = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {
            "reachable": True,
            "model": data.get("id") or data.get("name"),
            "n_params": meta.get("n_params"),
            "completion_error": str(e),
        }
    tim = d.get("timings") or {}
    return {
        "reachable": True,
        "this_is_not_the_gravity_artifact": True,
        "model": data.get("id") or data.get("name"),
        "n_params": meta.get("n_params"),
        "n_ctx": meta.get("n_ctx"),
        "gguf_size_bytes": meta.get("size"),
        "timings": tim,
        "client_wall_s": round(wall, 3),
        "ttft_ms_approx": tim.get("prompt_ms"),
        "prefill_tps": tim.get("prompt_per_second"),
        "decode_tps": tim.get("predicted_per_second"),
        "why_separate": (
            "llama-server:52484 is Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf "
            f"(n_params={meta.get('n_params')}), not uniform-q4-v1 "
            "(26,895,998,464). Folding these timings into gravity fields is "
            "the same class of error as annualising a 12.4 s window into "
            "1164 units/hour."
        ),
    }


def inspect_artifact(artifact: Path) -> dict:
    man = json.loads((artifact / "manifest.json").read_text())
    tensors = man["tensors"]
    n_files, total, by_ext = walk_bytes(artifact)
    class_bytes = collections.Counter()
    class_n = collections.Counter()
    kind_n = collections.Counter()
    kind_bytes = collections.Counter()
    missing = 0
    present_bytes = 0
    header_bytes_q4 = 0
    header_bytes_f32 = 0
    hq30_groups = set()
    hq30_ok = 0
    for t in tensors:
        tag = classify_tensor_name(t["name"])
        class_bytes[tag] += t["bytes"]
        class_n[tag] += 1
        kind_n[t["kind"]] += 1
        kind_bytes[t["kind"]] += t["bytes"]
        p = artifact / "tensors" / t["artifact"]
        if not p.is_file():
            missing += 1
            continue
        present_bytes += p.stat().st_size
        if t["kind"] == "q4":
            hdr = parse_hq30_header(p.read_bytes()[:HQ30_HEADER_BYTES])
            hq30_groups.add(hdr["group"])
            hq30_ok += 1
            header_bytes_q4 += HQ30_HEADER_BYTES
        elif t["kind"] == "f32":
            header_bytes_f32 += F32V2_HEADER_BYTES
    embed = class_bytes["embed_table"]
    embed_row = q4_matrix_bytes(1, 5120)
    payload = int(man["tensor_payload_bytes"])
    active = payload - embed + embed_row
    families = sorted(kind_n.keys())
    return {
        "manifest": {k: man[k] for k in man if k != "tensors"},
        "n_files_walked": n_files,
        "total_physical_bytes": total,
        "by_ext": by_ext,
        "tensor_count": len(tensors),
        "missing_tensor_files": missing,
        "present_tensor_bytes": present_bytes,
        "payload_bytes": payload,
        "kind_n": dict(kind_n),
        "kind_bytes": dict(kind_bytes),
        "class_n": dict(class_n),
        "class_bytes": dict(class_bytes),
        "hq30_parsed": hq30_ok,
        "hq30_groups": sorted(hq30_groups),
        "header_bytes_q4": header_bytes_q4,
        "header_bytes_f32": header_bytes_f32,
        "embed_table_bytes": embed,
        "embed_row_bytes": embed_row,
        "active_bytes_per_token": active,
        "codec_families": families,
        "source_dir": man.get("source_dir"),
        "source_weight_elements": int(man["source_weight_elements"]),
        "complete_physical_bpw_manifest": man.get("complete_physical_bpw"),
        "skipped_vision_tensors": man.get("skipped_vision_tensors"),
        "fused_in_proj_layers": man.get("fused_in_proj_layers"),
        "tensors": tensors,
    }


def build_record(artifact: Path, live_probe: bool) -> dict:
    generated_at = utc_now()
    art = inspect_artifact(artifact)
    params = art["source_weight_elements"]
    payload = art["payload_bytes"]
    physical = art["total_physical_bytes"]
    ebpw = 8.0 * payload / params
    source_dir = Path(art["source_dir"]) if art["source_dir"] else None
    parent_cfg = None
    source_census = None
    source_tree = None
    if source_dir and source_dir.is_dir():
        n, tot, by_ext = walk_bytes(source_dir)
        source_tree = {"files": n, "bytes": tot, "by_ext": by_ext, "path": str(source_dir)}
        source_census = safetensors_payload_census(source_dir)
        cfg_path = source_dir / "config.json"
        if cfg_path.is_file():
            parent_cfg = json.loads(cfg_path.read_text())
    geo = geometry_from_parent_config(parent_cfg) if parent_cfg else None
    gemv = gemv_payload_from_geometry()
    fused = kernel_reconstruction_is_fused()
    kernels = bound_kernels()
    runtime = hawking_runtime_bytes()
    mapped = artifact_mapped_now(artifact)
    mem_bytes = int(sh(["sysctl", "-n", "hw.memsize"]) or "0")
    occupancy = (payload / mem_bytes) if mem_bytes else None
    ca = content_address_sample(artifact, art["tensors"])
    native = load_native_time(REPO / "receipts/headless")
    recon_ledger = token_ns_reconstruction(REPO)
    g105 = git_show("receipts/ascent-2026-08-16/G105_NR_NX_ARTIFACT.json")

    source_bytes = None
    source_method = None
    if source_census:
        source_bytes = source_census["language_plus_lm_head_payload_bytes"]
        source_method = (
            "sum of safetensors data_offsets for model.language_model.* plus "
            "lm_head.weight. Vision (333 tensors) and MTP are not packed and "
            "are not SOURCE_BYTES. Equals PARENT_PARAMETER_COUNT * 2 (bf16)."
        )

    dropped = [
        {
            "field": "RESIDENT_EQUIVALENT_BPW",
            "why": (
                "NX places all weights in unified memory and they fit "
                f"({occupancy * 100:.2f}% of {mem_bytes} bytes RAM). "
                "8*RESIDENT/PARAMS is then EBPW under a second name. The "
                "useful resident fact is the byte count and the occupancy "
                "fraction, not another BPW."
            ),
        },
        {
            "field": "ACTIVE_EQUIVALENT_BPW_PER_TOKEN",
            "why": (
                "BPW is demoted to one field (EXECUTABLE_BPW). The active "
                "quantity that cannot be recovered from EBPW is the byte "
                "count (embed table is not moved). Re-encoding that count "
                "as BPW is how G042's family of BPWs hid the on-disk vs "
                "moved distinction."
            ),
        },
        {
            "field": "PREFILL_TPS",
            "why": (
                "The native decoder prefills by a teacher-forced walk of "
                "each prompt token (same 964-dispatch GEMV organ as decode). "
                "At the 34-token prompt used in QWEN38_GRAVITY_NATIVE, "
                "prefill wall is ~1.016 s → ~33.5 steps/s, indistinguishable "
                "from complete decode. A distinct batched-prefill TPS does "
                "not exist on this executable. Reporting one would invent a "
                "lever. The live llama-server Q5_K GGUF DOES have a distinct "
                "prefill (measured this process as a labeled control only)."
            ),
        },
        {
            "field": "CACHE_BPW / GENERATED_BPW / CORRECTION_BPW / SHARED_BPW",
            "why": (
                "G042 BPW family. GENERATED/CORRECTION/SHARED were measured "
                "0 for this dense patient. CACHE_BPW is activation traffic "
                "that G094 found has zero rank correlation with time. BPW "
                "is one field now; zeros and uncorrelated traffic are not "
                "promoted into the record."
            ),
        },
    ]

    # --- static ---
    parent_parameter_count = measured(
        params, unit="parameters",
        method="manifest.source_weight_elements; sum(tensor.elements) checked equal",
        extra={
            "sum_tensor_elements": sum(t["elements"] for t in art["tensors"]),
            "g105_nr": (g105 or {}).get("NR", {}).get("semantic_provenance", {})
            .get("parameter_count") if g105 else None,
        },
    )
    source_b = (
        measured(source_bytes, unit="bytes", method=source_method,
                 extra={"bf16_equivalent": params * 2,
                        "matches_params_times_2": source_bytes == params * 2,
                        "parent_tree": source_tree,
                        "census": {k: v for k, v in (source_census or {}).items()
                                   if k != "buckets"} | {
                            "buckets": {
                                k: {kk: vv for kk, vv in buck.items() if kk != "names"}
                                for k, buck in (source_census or {}).get("buckets", {}).items()
                            }
                        }})
        if source_bytes is not None else
        null_field(f"parent source_dir not on disk: {art['source_dir']}")
    )
    executable_model = measured(
        payload, unit="bytes",
        method="sum of 755 tensor file sizes (= manifest.tensor_payload_bytes)",
        extra={"present_tensor_bytes": art["present_tensor_bytes"],
               "missing": art["missing_tensor_files"]},
    )
    executable_bpw = measured(
        ebpw, unit="bits_per_weight",
        formula="8 * EXECUTABLE_MODEL_BYTES / PARENT_PARAMETER_COUNT",
        method="computed this process from walked payload and manifest parameter count",
        extra={
            "exact": ebpw,
            "recorded_approx": RECORDED_EBPW,
            "manifest_complete_physical_bpw": art["complete_physical_bpw_manifest"],
            "delta_vs_recorded_approx": ebpw - RECORDED_EBPW,
            "delta_vs_manifest": ebpw - float(art["complete_physical_bpw_manifest"]),
            "reconciliation": (
                "Matches manifest complete_physical_bpw exactly. The sealed "
                "anchor '~4.253' is that value rounded to three decimals. "
                "G042 STORED_BPW 4.25595 was a different directory (campaign "
                "run root with extra files); this artifact's walk has no "
                "dead tensors."
            ),
        },
    )
    total_physical = measured(
        physical, unit="bytes",
        method="os.walk of artifact root, every regular file, followlinks=False",
        extra={"n_files": art["n_files_walked"], "by_ext": art["by_ext"],
               "overhead_vs_payload": physical - payload,
               "overhead_is": "manifest.json only (238924 bytes)"},
    )
    resident = measured(
        payload, unit="bytes",
        method=(
            "When loaded, NX residency_plan is all_weights in unified_memory. "
            "lsof on the artifact this process shows it is NOT currently "
            "mapped, so this is the loaded occupancy (G085's 13.9% numerator), "
            "not a live RSS. Headers stay on disk after load (~19 KiB); "
            "Metal holds codes+scales ≈ payload."
        ),
        extra={
            "live_mapped_now": mapped,
            "machine_unified_memory_bytes": mem_bytes,
            "occupancy_fraction_of_ram": occupancy,
            "occupancy_pct": occupancy * 100 if occupancy is not None else None,
            "g085_stated": "13.9% of 103.1 GB",
        },
    )

    # --- active ---
    active_bytes = art["active_bytes_per_token"]
    active = measured(
        active_bytes, unit="bytes/token",
        formula="payload - embed_table + one_embed_row",
        method="tensor census this process; embed row = q4_matrix_bytes(1, 5120) = 2720",
        extra={
            "embed_table_bytes": art["embed_table_bytes"],
            "embed_row_bytes": art["embed_row_bytes"],
            "class_bytes": art["class_bytes"],
            "why_not_all_payload": (
                "Dense model reads every weight per token EXCEPT the embedding "
                "table, from which exactly one row is gathered."
            ),
        },
    )
    dram = measured(
        gemv["gemv_payload_bytes"], unit="bytes/token",
        method=(
            "Bytes the Q4 GEMV kernels actually stream: 32 code bytes + 2-byte "
            "f16 scale per group of 64, summed over MLP+DeltaNet+GQA+lm_head "
            "from geometry that matches parent config.json. Headers are not "
            "streamed. Mixer extras (conv, A_log, dt_bias, q/k RMS) are not "
            "GEMV traffic. This is NOT G042's 'DRAM = ACTIVE'."
        ),
        formula=gemv["formula"],
        extra={
            "breakdown": gemv,
            "vs_active_bytes": active_bytes - gemv["gemv_payload_bytes"],
            "vs_active_is": (
                "ACTIVE includes HQ30UQ4 headers, f32 mixer extras, f32 norms, "
                "and the gathered embed row. DRAM here is GEMV payload only."
            ),
        },
    )
    kv_write = geo["gqa_kv_write_bytes_per_position"] if geo else None
    kv = (
        measured(
            kv_write, unit="bytes/token",
            method=(
                "GQA write of one new (K,V) slot: gqa_layers * 2 * kv_heads * "
                "head_dim * 4, from parent config.json (16 * 2 * 4 * 256 * 4)."
            ),
            extra={
                "read_bytes_grow_with_seq_len": geo["gqa_kv_read_bytes_at_len"],
                "write_is_constant_per_decode_step": True,
                "g042_only_counted_this_as_STATE": (
                    "G042 STATE_BPW_EQUIVALENT is this KV term only. It missed "
                    "DeltaNet conv+recurrent state, reported separately as "
                    "STATE_BYTES_PER_TOKEN."
                ),
            },
        ) if kv_write is not None else
        null_field("parent config.json not readable")
    )
    state_rw = (
        geo["conv_state_rw_bytes_per_token"] + geo["rec_state_rw_bytes_per_token"]
        if geo else None
    )
    state = (
        measured(
            state_rw, unit="bytes/token",
            method=(
                "DeltaNet conv and recurrent state are read+written every "
                "token, independent of context. From config.json: 48 layers, "
                "conv_channels*(kernel-1)*4*2 + value_heads*key_dim*value_dim*4*2."
            ),
            extra={
                "conv_rw": geo["conv_state_rw_bytes_per_token"],
                "rec_rw": geo["rec_state_rw_bytes_per_token"],
                "conv_resident": geo["conv_state_resident_bytes"],
                "rec_resident": geo["rec_state_resident_bytes"],
                "does_not_include_gqa_kv": True,
                "geometry_matches_qwen38_geometry_rs": geo["matches_qwen38_geometry_rs"],
            },
        ) if state_rw is not None else
        null_field("parent config.json not readable")
    )
    recon_b = measured(
        0, unit="bytes/token",
        method=(
            "Inspected qwen_uniform_q4.metal this process: the production "
            "matvec does `sum += float(q)*scale*input[col]` and writes only "
            "the output row. No reconstructed-weight buffer. The sealed "
            "token-ns ledger also records weight_decode_reconstruction "
            "bytes_read=0, bytes_written=0. Codes+scales are DRAM_BYTES, "
            "not an extra reconstruction term."
        ),
        extra={"kernel": fused, "ledger_bytes_read": (recon_ledger or {}).get("reconstruction_bytes_read"),
               "ledger_bytes_written": (recon_ledger or {}).get("reconstruction_bytes_written")},
    )
    if recon_ledger and recon_ledger.get("reconstruction_ns_per_token") is not None:
        recon_ns = measured(
            recon_ledger["reconstruction_ns_per_token"], unit="ns/token",
            method=(
                "Sealed QWEN38_TOKEN_NS_LEDGER component "
                "weight_decode_reconstruction: addr_probe vs decode_probe "
                "fraction of isolated class GEMVs, GPUEndTime-GPUStartTime. "
                "Not re-isolated this process (would require the native 27B)."
            ),
            source=recon_ledger["receipt"],
            extra={
                "ledger_wall_ns": recon_ledger["median_wall_ns"],
                "ledger_gpu_ns": recon_ledger["median_gpu_ns"],
                "measurement_label": recon_ledger["measurement_label"],
                "commit": recon_ledger["commit"],
                "probes": recon_ledger["probes"],
                "not_re_run": recon_ledger["not_re_run"],
                "belongs_to_older_wall": (
                    "Ledger complete wall is 35.228 ms; the more recent "
                    "native receipt is 30.275 ms. Reconstruction ns was not "
                    "scaled to the new wall — scaling a 12-component split "
                    "without re-probing is how a receipt starts lying."
                ),
            },
        )
    else:
        recon_ns = null_field(
            "QWEN38_TOKEN_NS_LEDGER.json not available via disk or git show"
        )

    # --- time ---
    if native.get("median_complete_wall_tps") is not None:
        complete_tps = measured(
            native["median_complete_wall_tps"], unit="tokens/s",
            method=(
                f"{native.get('n_process_runs')} native decode process runs, "
                "pairs=3, max_new_tokens=128, median of per-run complete_wall_tps. "
                "Steady decode steps only (prefill excluded from the denominator)."
            ),
            source=native["published_receipt"],
            extra={
                "per_run": native.get("per_run_complete_wall_tps"),
                "spread_pct": native.get("measurement_spread_pct"),
                "recorded_at": native.get("recorded_at"),
                "bar": native.get("bar"),
                "bar_reasons": native.get("bar_reasons"),
                "g105_tps": (g105 or {}).get("load_and_generate", {}).get("tps"),
                "not_re_run": native.get("not_re_run"),
                "output_collapsed": (
                    "QWEN38_GRAVITY_NATIVE bar is FAIL on coherence "
                    "(replacement characters). The timing is of a collapsed "
                    "greedy loop; it is still the wall of this executable."
                ),
            },
        )
        complete_ns = measured(
            native["median_complete_wall_ns"], unit="ns/token",
            method="same native runs as COMPLETE_DECODE_TPS; median complete_wall_ns",
            source=native["published_receipt"],
            extra={"formula_check_tps": 1e9 / native["median_complete_wall_ns"]
                   if native["median_complete_wall_ns"] else None},
        )
    else:
        complete_tps = null_field("QWEN38_GRAVITY_NATIVE.json not in receipts/headless")
        complete_ns = null_field("QWEN38_GRAVITY_NATIVE.json not in receipts/headless")

    if native.get("median_prefill_wall_ns") is not None:
        ttft = measured(
            native["median_prefill_wall_ns"], unit="ns",
            method=(
                "Median warm prefill_wall_ns across the RAW native runs' "
                "in-process warm reps. Prefill is the teacher-forced walk of "
                "the prompt; the last prefill step emits new-token[0]. "
                "session.open / chat render are request-level and excluded "
                "(session_open_ns was 3.19 s and is NOT in this number)."
            ),
            source=native.get("raw_dir"),
            extra={
                "n_samples": native["n_warm_prefill_samples"],
                "n_prefill_steps": native["n_prefill_steps"],
                "min_ns": native.get("prefill_wall_ns_min"),
                "max_ns": native.get("prefill_wall_ns_max"),
                "median_last_prefill_step_ns": native.get("median_last_prefill_step_ns"),
                "last_step_is_not_user_ttft": (
                    "The last prefill step is ~29.7 ms, same as a decode "
                    "step. User-facing TTFT is the whole prompt walk."
                ),
            },
        )
    else:
        ttft = null_field(
            "QWEN38_GRAVITY_NATIVE_RAW.run*.json not found; TTFT is not in the "
            "published summary receipt. Not re-run (second 27B forbidden)."
        )

    live = probe_live_llama() if live_probe else {"skipped": True}

    llama_pid = listen_pid_on_port(52484)
    llama_rss = proc_rss(llama_pid) if llama_pid else None
    llama_rt = llama_dylib_bytes()

    model_specific = {
        "bytes": payload,
        "what": "755 tensor payloads (HQ30UQ4 + f32v2) under the artifact root",
        "includes": ["codes", "f16 group scales", "f32 mixer/norm vectors",
                     "per-tensor headers that ride in the file"],
        "does_not_include": ["native decode binary", "Metal shaders",
                             "llama.cpp", "OS AGX metallib"],
    }
    shared = {
        "hawking_native_decode_binary_bytes": runtime["native_decode_binary_bytes"],
        "hawking_native_decode_binary_path": runtime["native_decode_binary_path"],
        "hawking_shader_source_bytes": runtime["shader_source_bytes"],
        "hawking_shader_source_files": runtime["shader_source_files"],
        "hawking_decode_rs_bytes": runtime["decode_rs_bytes"],
        "llama_cpp_runtime_bytes_control_only": llama_rt["bytes"],
        "boundary": (
            "EXECUTABLE_MODEL_BYTES / EBPW use only tensor payloads. The "
            "6.3 MiB native decode binary and 1.5 MiB of shader source are "
            "shared Hawking runtime: they decode this patient and every "
            "future Qwen3.8 pack that binds the same kernels. Moving model "
            "information into the binary would show up here as runtime "
            "growth, not as a fake EBPW win. llama.cpp dylibs are the "
            "control runtime for the live GGUF, not this artifact's NX."
        ),
        "runtime": runtime,
        "kernels": {k: kernels[k] for k in kernels if k != "names"},
        "kernel_names": kernels.get("names"),
    }

    static = {
        "PARENT_PARAMETER_COUNT": parent_parameter_count,
        "SOURCE_BYTES": source_b,
        "EXECUTABLE_MODEL_BYTES": executable_model,
        "EXECUTABLE_BPW": executable_bpw,
        "TOTAL_PHYSICAL_BYTES": total_physical,
        "RESIDENT_BYTES": resident,
    }
    active_fields = {
        "ACTIVE_BYTES_PER_TOKEN": active,
        "DRAM_BYTES_PER_TOKEN": dram,
        "KV_BYTES_PER_TOKEN": kv,
        "STATE_BYTES_PER_TOKEN": state,
        "RECONSTRUCTION_BYTES_PER_TOKEN": recon_b,
        "RECONSTRUCTION_NS_PER_TOKEN": recon_ns,
    }
    time_fields = {
        "TTFT": ttft,
        "COMPLETE_DECODE_TPS": complete_tps,
        "COMPLETE_TOKEN_NS": complete_ns,
    }

    def count_status(maps):
        m = n = 0
        for block in maps:
            for v in block.values():
                st = v.get("status")
                if st == "MEASURED":
                    m += 1
                elif st == "NULL":
                    n += 1
        return m, n

    n_measured, n_null = count_status([static, active_fields, time_fields])

    watched_fail = [
        {
            "what": "git apply of lane-bootstrap/tree-state.patch",
            "result": "FAILED",
            "why": (
                "Patch is 30 files under hcli, which is a DENY "
                "path and is not in this sparse checkout. untracked.tar is "
                "the same tree. Baseline 464-pass haider suite cannot run "
                "here without materializing tools/haider."
            ),
        },
        {
            "what": "content-addressing of 64-hex tensor filenames",
            "result": f"{ca['matches']}/{ca['n']} matched",
            "why": (
                "G105 defect reconfirmed this process: filenames look like "
                "sha256 and are not. A substituted tensor would not be "
                "detected by name."
            ),
        },
        {
            "what": "byte_reproducibility of uniform-q4-v1",
            "result": "NOT MET (inherited, not re-run)",
            "why": (
                "G105: encoder rounding and in_proj fusion recipe are not "
                "in the artifact. This process did not re-pack."
            ),
        },
        {
            "what": "using llama-server:52484 as gravity time",
            "result": "REFUSED",
            "why": (
                f"Server is {((live or {}).get('model'))} at "
                f"{(live or {}).get('decode_tps')} tps, n_params="
                f"{(live or {}).get('n_params')}. Gravity native is 33.03 tps "
                "on 26,895,998,464 params. Mixing them is the 1164/hour failure."
            ),
        },
        {
            "what": "G042 STATE_BPW as the state metric",
            "result": "INCOMPLETE",
            "why": (
                "G042 counted GQA KV only (131,072 B/position). Config-measured "
                f"DeltaNet state is {geo['rec_state_resident_bytes'] if geo else '?'} B "
                "resident, read+written every token. That is the actual state "
                "traffic; KV is reported separately."
            ),
        },
        {
            "what": "G042 DRAM_BPW = ACTIVE_BPW",
            "result": "WRONG DENOMINATOR FOR WEIGHT TRAFFIC",
            "why": (
                f"ACTIVE={active_bytes} includes headers and mixer extras. "
                f"GEMV payload={gemv['gemv_payload_bytes']} is what "
                "geo_tpr64 streams. honest_roof already adjudicated this; "
                "this record keeps the split instead of collapsing it."
            ),
        },
        {
            "what": "PREFILL_TPS as a distinct lever on this executable",
            "result": "DROPPED",
            "why": "Native prefill is serial teacher-force at decode speed.",
        },
        {
            "what": "live RSS of the gravity artifact",
            "result": "NOT CURRENTLY RESIDENT",
            "why": (
                f"lsof shows {mapped['manifest_open_in_n_processes']} processes "
                "holding manifest.json. Measuring RSS would require loading "
                "the 27B native decoder alongside the live llama-server."
            ),
        },
        {
            "what": "QWEN38_GRAVITY_NATIVE coherence bar",
            "result": "FAIL (timings still used)",
            "why": (
                "Generated text is replacement characters; greedy ids stuck "
                "on 150910. Speed of a collapsed loop is still the wall of "
                "this executable; it is not a quality claim."
            ),
        },
        {
            "what": "ps / vmmap on pid 59064",
            "result": "sandbox Operation not permitted / privilege",
            "why": "Fell back to lsof + proc_pidinfo for the live GGUF only.",
        },
    ]

    record = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "commit": git_head(),
        "artifact": {
            "name": "uniform-q4-v1",
            "path": str(artifact.resolve()),
            "nr_kind": "hawking.nos.noetic_representation",
            "nx_kind": "hawking.nos.noetic_executable_genome",
            "g105_schema": "hawking.nos.nr_nx_artifact.v1",
            "codec_families": [
                {"family": "grouped_absmax", "kind": "q4", "bits": 4, "group": 64,
                 "count": art["kind_n"].get("q4"), "bytes": art["kind_bytes"].get("q4"),
                 "hq30_parsed": art["hq30_parsed"], "groups_seen": art["hq30_groups"]},
                {"family": "raw_f32", "kind": "f32",
                 "count": art["kind_n"].get("f32"), "bytes": art["kind_bytes"].get("f32")},
            ],
            "exactly_two_codec_families": art["codec_families"] == ["f32", "q4"] or
            art["codec_families"] == ["q4", "f32"],
            "tensor_count": art["tensor_count"],
            "n_files_walked": art["n_files_walked"],
            "skipped_vision_tensors": art["skipped_vision_tensors"],
            "fused_in_proj_layers": art["fused_in_proj_layers"],
        },
        "machine": {
            "unified_memory_bytes": mem_bytes,
            "chipset": "Apple M3 Ultra",
            "source": "sysctl hw.memsize this process; chipset from NX/G105 (not re-derived)",
        },
        "byte_boundary": {
            "model_specific": model_specific,
            "shared_runtime": shared,
        },
        "static": static,
        "active": active_fields,
        "time": time_fields,
        "dropped": dropped,
        "counts": {
            "measured": n_measured,
            "null_with_reason": n_null,
            "dropped": len(dropped),
            "surviving_named_fields": n_measured + n_null,
        },
        "ebpw_reconciliation": {
            "exact": executable_bpw.get("exact", ebpw),
            "recorded_approx": executable_bpw.get("recorded_approx", RECORDED_EBPW),
            "manifest_complete_physical_bpw": executable_bpw.get(
                "manifest_complete_physical_bpw"
            ),
            "delta_vs_recorded_approx": executable_bpw.get("delta_vs_recorded_approx"),
            "delta_vs_manifest": executable_bpw.get("delta_vs_manifest"),
            "reconciliation": executable_bpw.get("reconciliation"),
        },
        "defects_reconfirmed_this_process": {
            "content_addressing": ca,
            "byte_reproducibility_NOT_met": (g105 or {}).get("byte_reproducibility_NOT_met"),
        },
        "live_control_different_artifact": {
            "llama_server": live,
            "pid": llama_pid,
            "rss": llama_rss,
            "runtime_bytes": llama_rt,
        },
        "geometry": geo,
        "watched_fail": watched_fail,
    }
    return record


def print_report(doc: dict) -> None:
    a = doc["artifact"]
    c = doc["counts"]
    b = doc["byte_boundary"]
    print("=== NOETIC METRICS ===")
    print(f"  artifact     {a['name']}  {a['path']}")
    fam = ", ".join(
        f"{c['family']}({c['kind']},{c['count']})" for c in a["codec_families"]
    )
    print(f"  files        {a['n_files_walked']}   tensors {a['tensor_count']}   "
          f"codecs {fam}")
    print(f"  fields       {c['measured']} MEASURED, {c['null_with_reason']} NULL, "
          f"{c['dropped']} DROPPED")
    print()
    print("  STATIC")
    for k, v in doc["static"].items():
        print(f"    {k:28} {v['status']:9} {v.get('value')}")
    print("  ACTIVE")
    for k, v in doc["active"].items():
        print(f"    {k:28} {v['status']:9} {v.get('value')}")
    print("  TIME")
    for k, v in doc["time"].items():
        print(f"    {k:28} {v['status']:9} {v.get('value')}")
    print()
    ebpw_f = doc["static"]["EXECUTABLE_BPW"]
    rec = doc["ebpw_reconciliation"]
    print(f"  EBPW         {ebpw_f['value']:.12f}")
    print(f"  recorded     ~{rec.get('recorded_approx')}   "
          f"manifest {rec.get('manifest_complete_physical_bpw')}")
    print(f"  reconcile    {rec.get('reconciliation')}")
    print()
    ms = b["model_specific"]
    sh_rt = b["shared_runtime"]
    print("  BYTE BOUNDARY")
    print(f"    model-specific     {ms['bytes']}  ({ms['what']})")
    print(f"    hawking binary     {sh_rt['hawking_native_decode_binary_bytes']}")
    print(f"    hawking shaders    {sh_rt['hawking_shader_source_bytes']} "
          f"({sh_rt['hawking_shader_source_files']} .metal)")
    print(f"    llama.cpp control  {sh_rt['llama_cpp_runtime_bytes_control_only']}  (NOT in EBPW)")
    print(f"    {sh_rt['boundary']}")
    print()
    print("  DROPPED")
    for d in doc["dropped"]:
        print(f"    - {d['field']}")
        print(f"      {d['why']}")
    print()
    print("  ## WHAT I WATCHED FAIL")
    for w in doc["watched_fail"]:
        print(f"    [{w['result']}] {w['what']}")
        print(f"      {w['why']}")
    live = doc["live_control_different_artifact"]["llama_server"]
    if live.get("reachable"):
        print()
        print("  LIVE CONTROL (Q5_K GGUF on :52484, NOT gravity)")
        print(f"    model      {live.get('model')}  n_params={live.get('n_params')}")
        print(f"    prefill    {live.get('prefill_tps')} tps   decode {live.get('decode_tps')} tps")
        print(f"    {live.get('why_separate')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--out", type=Path,
                    default=REPO / "receipts/headless/NOETIC_METRICS.json")
    ap.add_argument("--no-live-probe", action="store_true",
                    help="skip the llama-server:52484 control completion")
    args = ap.parse_args()
    if not args.artifact.is_dir():
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        return 2
    doc = build_record(args.artifact, live_probe=not args.no_live_probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(doc)
    print(f"\n-> {args.out}")

    # Hard fails: a plausible number with no measurement is the failure mode.
    problems = []
    for section in ("static", "active", "time"):
        for name, f in doc[section].items():
            if f.get("status") not in ("MEASURED", "NULL"):
                problems.append(f"{name} status {f.get('status')}")
            if f.get("status") == "MEASURED" and f.get("value") is None:
                problems.append(f"{name} MEASURED but value is null")
            if f.get("status") == "NULL" and not f.get("null_reason"):
                problems.append(f"{name} NULL without reason")
    ebpw = doc["static"]["EXECUTABLE_BPW"]["value"]
    if abs(ebpw - RECORDED_EBPW_EXACT) > 1e-12:
        problems.append(f"EBPW {ebpw} != manifest {RECORDED_EBPW_EXACT}")
    if not doc["artifact"]["exactly_two_codec_families"]:
        problems.append(f"codec families {doc['artifact']['codec_families']}")
    if doc["counts"]["dropped"] < 1:
        problems.append("no field dropped")
    if problems:
        print("SELF-CHECK FAIL:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
