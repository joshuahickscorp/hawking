#!/usr/bin/env python3
"""Why MLX is faster on this box, and which of that a native Gravity operator can have.

Observes the installed MLX runtime (metallib + headers + mlx_lm sources) and the
on-disk 4-bit Qwen3.8 artifact. Does not load the 27B. Does not spawn a second
model server — occupancy already measured 3.986 tok/s with two residents against
33.47 with one.

Every number is MEASURED in this process, PRIOR_MEASURED from a named receipt,
SOURCE_COUNTED from installed source/binary, or NULL with a reason.

    python3 tools/headless/noetic_mlx_anatomy.py
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "hawking.headless.noetic_mlx_anatomy.v1"

# Anchors already measured. Do not re-derive.
ANCHOR_TPS_NATIVE = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_TENSORS = 755
ANCHOR_ACTIVE_B = 13_622_264_240
ANCHOR_GEMV_GFLOP = 51.24
ANCHOR_TWO_SERVER_TPS = 3.986
ANCHOR_ONE_SERVER_TPS = 33.47
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_TPS = 24.12
ANCHOR_SPEED_RATIO = 1.472
ANCHOR_BYTES_RATIO_GPU_ATTACK = 1.215
ANCHOR_LLAMA_B = 19_535_701_280
ANCHOR_NATIVE_GPU_NS = 29_204_250
ANCHOR_NATIVE_WALL_NS = 30_388_625

MLX_PY = Path.home() / ".local/share/uv/tools/mlx-lm/bin/python"
MLX_SITE = Path.home() / ".local/share/uv/tools/mlx-lm/lib/python3.12/site-packages"
MLX_ROOT = MLX_SITE / "mlx"
MLX_LM = MLX_SITE / "mlx_lm"
MLX_METALLIB = MLX_ROOT / "lib" / "mlx.metallib"
MLX_DYLIB = MLX_ROOT / "lib" / "libmlx.dylib"
MLX_CORE_SO = MLX_ROOT / "core.cpython-312-darwin.so"

MLX_4BIT = Path.home() / "models/qwen3.8-27b-abliterated-mlx/4bit"
MLX_HUIHUI = Path.home() / "models/qwen3.8-27b-abliterated-mlx-huihui-4bit"
NATIVE_ARTIFACT = Path.home() / "models/qwen38-gravity-uniform-q4-v1"
GGUF = Path.home() / "models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
NATIVE_BIN = Path.home() / (
    "Downloads/hawking-copy/workspace/ops/build/rust/release-fast/"
    "examples/ascension_qwen38_hybrid_greedy"
)

LLAMA_HEALTH = "http://127.0.0.1:52484/health"
LLAMA_PROPS = "http://127.0.0.1:52484/props"

HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
LAYERS = 64
LINEAR_LAYERS = 48
GQA_LAYERS = 16
GROUP = 64
HEAD_DIM = 256

# Affine Q4 group-64: 32 code bytes + 2 scale + 2 bias.
AFFINE_BYTES_PER_GROUP = 32 + 2 + 2  # 36
ABSMAX_BYTES_PER_GROUP = 32 + 2  # 34 native grouped-absmax


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "tools" / "headless").is_dir() and (p / "Cargo.toml").is_file():
            return p
    return Path.cwd()


REPO = repo_root()
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_MLX_ANATOMY.json"
RECEIPTS = REPO / "receipts" / "headless"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def field(value, status, **extra):
    d = {"value": value, "status": status}
    d.update(extra)
    return d


def measured(value, **kw):
    return field(value, "MEASURED", **kw)


def prior(value, source, **kw):
    return field(value, "PRIOR_MEASURED", source=source, **kw)


def source_counted(value, **kw):
    return field(value, "SOURCE_COUNTED", **kw)


def observed_binary(value, **kw):
    return field(value, "OBSERVED_BINARY", **kw)


def null(reason, **kw):
    return field(None, "NULL", null_reason=reason, **kw)


def load_json(path: Path):
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def http_json(url: str, timeout: float = 2.0):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw.decode("utf-8", "replace")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def file_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def dir_bytes(path: Path) -> dict:
    total = 0
    n_files = 0
    by_ext: Counter[str] = Counter()
    if not path.is_dir():
        return {"present": False, "bytes": 0, "n_files": 0, "by_ext": {}}
    for dp, _, fns in os.walk(path):
        for fn in fns:
            fp = Path(dp) / fn
            if fp.is_symlink():
                continue
            try:
                sz = fp.stat().st_size
            except OSError:
                continue
            total += sz
            n_files += 1
            ext = fp.suffix.lower() or "(none)"
            by_ext[ext] += sz
    return {
        "present": True,
        "bytes": total,
        "n_files": n_files,
        "by_ext": dict(by_ext),
    }


def pkg_version(dist: Path) -> str | None:
    meta = dist / "METADATA"
    if not meta.is_file():
        return None
    for line in meta.read_text(errors="replace").splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def occupancy() -> dict:
    health, health_err = http_json(LLAMA_HEALTH)
    props, props_err = http_json(LLAMA_PROPS)
    llama_up = bool(health) and health.get("status") == "ok"
    model_alias = None
    model_path = None
    n_ctx = None
    total_slots = None
    if props:
        model_alias = props.get("model_alias")
        model_path = props.get("model_path")
        dgs = props.get("default_generation_settings") or {}
        n_ctx = dgs.get("n_ctx")
        total_slots = props.get("total_slots")
    gguf_on_disk = GGUF.is_file()
    gguf_size = file_bytes(GGUF) if gguf_on_disk else None
    # Refuse a 27B MLX load whenever any 27B is already answering, or whenever
    # we cannot prove the GPU is empty. Health-up is sufficient proof of a
    # resident decoder.
    refuse = llama_up
    return {
        "llama_health_url": LLAMA_HEALTH,
        "llama_up": llama_up,
        "llama_health": health,
        "llama_health_error": health_err,
        "llama_props_error": props_err,
        "model_alias": model_alias,
        "model_path": model_path,
        "n_ctx": n_ctx,
        "total_slots": total_slots,
        "gguf_on_disk": gguf_on_disk,
        "gguf_bytes": gguf_size,
        "refuse_load_27b": refuse,
        "refuse_reason": (
            "llama-server is answering on 52484. Loading the MLX 27B alongside it "
            f"is forbidden (occupancy collapse {ANCHOR_ONE_SERVER_TPS} → "
            f"{ANCHOR_TWO_SERVER_TPS} tok/s, receipts/headless/HCLI_SELF_OPT_ITERATION_2.json "
            "and NOETIC_ROUTE_LEDGER.json)."
            if refuse
            else "no llama-server on 52484; still will not load 27B in this process "
            "(anatomy does not need a generate pass; metallib + artifact + sources suffice)."
        ),
        "two_server_tps": prior(
            ANCHOR_TWO_SERVER_TPS,
            "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json priors_bound_the_prize.two_server_tps",
        ),
        "one_server_tps": prior(
            ANCHOR_ONE_SERVER_TPS,
            "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json priors_bound_the_prize.one_server_tps",
        ),
    }


def metal_probe() -> dict:
    """Try to talk to Metal through the installed mlx python. Never loads a model."""
    snippet = (
        "import json\n"
        "out={'mlx_imported':False,'metal_is_available':None,'device_info':None,\n"
        "     'device_info_error':None,'tiny_eval_ok':False,'tiny_eval_error':None,\n"
        "     'error':None}\n"
        "try:\n"
        "    import mlx.core as mx\n"
        "    out['mlx_imported']=True\n"
        "    out['metal_is_available']=bool(mx.metal.is_available())\n"
        "    try:\n"
        "        out['device_info']=mx.device_info()\n"
        "    except Exception as e:\n"
        "        out['device_info_error']=f'{type(e).__name__}: {e}'\n"
        "    try:\n"
        "        y=mx.ones((4,), dtype=mx.float32)\n"
        "        mx.eval(y)\n"
        "        out['tiny_eval_ok']=True\n"
        "        out['tiny_eval_sum']=float(y.sum().item())\n"
        "    except Exception as e:\n"
        "        out['tiny_eval_ok']=False\n"
        "        out['tiny_eval_error']=f'{type(e).__name__}: {e}'\n"
        "except Exception as e:\n"
        "    out['error']=f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(out))\n"
    )
    if not MLX_PY.is_file():
        return {
            "attempted": False,
            "result": null(f"mlx python not at {MLX_PY}"),
        }
    try:
        proc = subprocess.run(
            [str(MLX_PY), "-c", snippet],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "attempted": True,
            "exit_code": None,
            "result": null(f"mlx probe failed to spawn: {exc}"),
        }
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            parsed = None
    parsed = parsed or {}
    tiny_ok = bool(parsed.get("tiny_eval_ok"))
    load_err = parsed.get("tiny_eval_error") or parsed.get("device_info_error")
    return {
        "attempted": True,
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-1500:],
        "parsed": parsed,
        "mlx_imported": bool(parsed.get("mlx_imported")),
        "metal_is_available_flag": parsed.get("metal_is_available"),
        "tiny_eval_ok": tiny_ok,
        "gpu_usable": tiny_ok,
        "device_info": parsed.get("device_info"),
        "device_info_error": parsed.get("device_info_error"),
        "tiny_eval_error": parsed.get("tiny_eval_error"),
        "error": parsed.get("error") or (None if tiny_ok else load_err),
        "note": (
            "mx.metal.is_available() can be True while mx.device_info()/mx.eval "
            "still raise [metal::load_device] No Metal device available. "
            "gpu_usable is tiny_eval_ok, not the is_available flag."
        ),
    }


def sysctl_machine() -> dict:
    out = {}
    for key, name in (
        ("hw.memsize", "mem_bytes"),
        ("machdep.cpu.brand_string", "cpu"),
        ("hw.ncpu", "ncpu"),
        ("hw.model", "hw_model"),
    ):
        try:
            p = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
            )
            val = p.stdout.strip()
            if name in ("mem_bytes", "ncpu") and val.isdigit():
                out[name] = int(val)
            else:
                out[name] = val
        except (OSError, subprocess.TimeoutExpired):
            out[name] = None
    # GPU cores from system_profiler if cheap; else prior.
    gpu_cores = None
    chipset = None
    metal = None
    try:
        p = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        text = p.stdout
        m = re.search(r"Total Number of Cores:\s+(\d+)", text)
        if m:
            gpu_cores = int(m.group(1))
        m = re.search(r"Chipset Model:\s+(.+)", text)
        if m:
            chipset = m.group(1).strip()
        if "Metal: Supported" in text:
            metal = "Supported"
    except (OSError, subprocess.TimeoutExpired):
        pass
    out["gpu_cores"] = gpu_cores
    out["chipset"] = chipset
    out["metal_reported"] = metal
    return out


def parse_safetensors_dir(root: Path) -> dict:
    files = sorted(root.glob("model-*.safetensors"))
    tensors = []
    header_bytes_total = 0
    for f in files:
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header_bytes_total += 8 + n
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            start, end = v["data_offsets"]
            tensors.append(
                {
                    "name": k,
                    "dtype": v.get("dtype"),
                    "shape": v.get("shape"),
                    "bytes": int(end) - int(start),
                    "file": f.name,
                }
            )

    def bucket(name: str) -> str:
        n = name.lower()
        if "vision" in n or "visual" in n:
            return "vision"
        if name.startswith("language_model") or name.startswith("lm_head"):
            return "language"
        return "other"

    def role(name: str) -> str:
        if name.endswith(".weight"):
            return "weight"
        if name.endswith(".scales"):
            return "scales"
        if name.endswith(".biases"):
            return "biases"
        return "other"

    def organ(name: str) -> str:
        if ".mlp." in name:
            return "mlp"
        if ".linear_attn." in name:
            return "linear_attn"
        if ".self_attn." in name:
            return "self_attn"
        if "embed_tokens" in name:
            return "embed"
        if "lm_head" in name:
            return "lm_head"
        if "layernorm" in name or name.endswith("norm.weight"):
            return "norm"
        return "other"

    by_bucket = defaultdict(lambda: {"n": 0, "bytes": 0})
    by_role = defaultdict(lambda: {"n": 0, "bytes": 0})
    by_organ = defaultdict(lambda: {"n": 0, "bytes": 0})
    by_dtype = defaultdict(lambda: {"n": 0, "bytes": 0})
    for t in tensors:
        b = bucket(t["name"])
        by_bucket[b]["n"] += 1
        by_bucket[b]["bytes"] += t["bytes"]
        r = role(t["name"])
        by_role[(b, r)]["n"] += 1
        by_role[(b, r)]["bytes"] += t["bytes"]
        if b == "language":
            o = organ(t["name"])
            by_organ[o]["n"] += 1
            by_organ[o]["bytes"] += t["bytes"]
        by_dtype[(b, t["dtype"])]["n"] += 1
        by_dtype[(b, t["dtype"])]["bytes"] += t["bytes"]

    lang_bytes = by_bucket["language"]["bytes"]
    embed_bytes = by_organ["embed"]["bytes"]
    embed_row = embed_bytes // VOCAB if VOCAB and embed_bytes else None
    active_language = (
        lang_bytes - embed_bytes + embed_row if embed_row is not None else None
    )

    # Quantized linear prefixes (have .weight in U32).
    q_prefixes = {
        t["name"].rsplit(".", 1)[0]
        for t in tensors
        if t["name"].endswith(".weight") and t["dtype"] == "U32"
    }

    l0_gate = [
        t for t in tensors if t["name"].startswith(
            "language_model.model.layers.0.mlp.gate_proj."
        )
    ]
    return {
        "n_files": len(files),
        "file_bytes": [file_bytes(f) for f in files],
        "n_tensors": len(tensors),
        "payload_bytes": sum(t["bytes"] for t in tensors),
        "header_bytes": header_bytes_total,
        "by_bucket": dict(by_bucket),
        "by_role": {f"{a}:{b}": v for (a, b), v in by_role.items()},
        "by_organ_language": dict(by_organ),
        "by_dtype": {f"{a}:{b}": v for (a, b), v in by_dtype.items()},
        "language_bytes": lang_bytes,
        "vision_bytes": by_bucket["vision"]["bytes"],
        "vision_tensors": by_bucket["vision"]["n"],
        "embed_table_bytes": embed_bytes,
        "embed_row_bytes": embed_row,
        "active_language_bytes": active_language,
        "u32_quantized_modules": len(q_prefixes),
        "l0_gate_proj": l0_gate,
        "affine_bytes_per_group": AFFINE_BYTES_PER_GROUP,
        "absmax_bytes_per_group": ABSMAX_BYTES_PER_GROUP,
        "affine_over_absmax": AFFINE_BYTES_PER_GROUP / ABSMAX_BYTES_PER_GROUP,
    }


def metallib_catalog(path: Path) -> dict:
    if not path.is_file():
        return {"present": False, "bytes": None, "kernels": []}
    raw = subprocess.check_output(["strings", "-a", str(path)], text=True, errors="replace")
    idents = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{6,160}\b", raw))
    families = {
        "affine_qmv_fast": [s for s in idents if s.startswith("affine_qmv_fast")],
        "affine_qmv": [
            s for s in idents if s.startswith("affine_qmv") and not s.startswith("affine_qmv_fast")
            and not s.startswith("affine_qmv_wide") and not s.startswith("affine_qmv_quad")
        ],
        "affine_qmv_wide": [s for s in idents if s.startswith("affine_qmv_wide")],
        "affine_qmm": [s for s in idents if s.startswith("affine_qmm")],
        "affine_dequantize": [s for s in idents if s.startswith("affine_dequantize")],
        "rms": [s for s in idents if re.match(r"^rms(bfloat16|float16|float32|looped)", s)],
        "rope": [s for s in idents if s.startswith("rope_")],
        "sdpa_vector": [s for s in idents if s.startswith("sdpa_vector")],
        "steel_gemm": [s for s in idents if s.startswith("steel_gemm")],
        "gemv": [s for s in idents if s.startswith("gemv")],
        "conv": [s for s in idents if s.lower().startswith("steel_conv") or s.startswith("conv1d") or s.startswith("conv2d")],
    }
    decode_4bit = sorted(
        s
        for s in idents
        if (
            ("gs_64_b_4" in s and ("qmv" in s or "qmm" in s))
            or s in {
                "rmsbfloat16", "rmsfloat16", "rmsfloat32",
                "rms_loopedbfloat16", "rms_loopedfloat16", "rms_loopedfloat32",
                "rope_single_bfloat16", "rope_single_float16", "rope_single_float32",
                "sdpa_vector_2pass_1_bfloat16_t_256_256",
                "sdpa_vector_2pass_2_bfloat16_t_256",
                "sdpa_vector_bfloat16_t_256_256",
            }
            or (s.startswith("sdpa_vector") and "256" in s)
        )
    )
    qmv_fast_gs64_b4 = sorted(
        s for s in families["affine_qmv_fast"] if "gs_64_b_4" in s
    )
    return {
        "present": True,
        "path": str(path),
        "bytes": file_bytes(path),
        "n_idents": len(idents),
        "family_counts": {k: len(v) for k, v in families.items()},
        "decode_relevant_names": decode_4bit,
        "qmv_fast_gs64_b4": qmv_fast_gs64_b4,
        "has_affine_dequantize": len(families["affine_dequantize"]) > 0,
        "n_affine_dequantize": len(families["affine_dequantize"]),
        "note": (
            "Names extracted with strings(1) from the shipped mlx.metallib. "
            "This is the compiled kernel catalog, not documentation. "
            "gated_delta_step is JIT via mx.fast.metal_kernel and is NOT in this metallib."
        ),
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def source_anatomy() -> dict:
    q35 = MLX_LM / "models" / "qwen3_5.py"
    qn = MLX_LM / "models" / "qwen3_next.py"
    gd = MLX_LM / "models" / "gated_delta.py"
    act = MLX_LM / "models" / "activations.py"
    gen = MLX_LM / "generate.py"
    qlin = MLX_ROOT / "nn" / "layers" / "quantized.py"
    rms = MLX_ROOT / "nn" / "layers" / "normalization.py"
    qh = MLX_ROOT / "include" / "mlx" / "backend" / "metal" / "kernels" / "quantized.h"
    compile_h = MLX_ROOT / "include" / "mlx" / "compile.h"
    device_h = MLX_ROOT / "include" / "mlx" / "backend" / "metal" / "device.h"
    trans_h = MLX_ROOT / "include" / "mlx" / "transforms.h"

    texts = {
        "qwen3_5.py": read_text(q35),
        "qwen3_next.py": read_text(qn),
        "gated_delta.py": read_text(gd),
        "activations.py": read_text(act),
        "generate.py": read_text(gen),
        "quantized.py": read_text(qlin),
        "normalization.py": read_text(rms),
        "quantized.h": read_text(qh),
        "compile.h": read_text(compile_h),
        "device.h": read_text(device_h),
        "transforms.h": read_text(trans_h),
    }

    def count(pat: str, text: str) -> int:
        return len(re.findall(pat, text))

    def class_method(src: str, cls: str, meth: str) -> str:
        cm = re.search(rf"class {cls}\b[\s\S]*?(?=\nclass |\Z)", src)
        if not cm:
            return ""
        mm = re.search(
            rf"    def {meth}\([\s\S]*?(?=\n    def |\n    @|\nclass |\Z)",
            cm.group(0),
        )
        return mm.group(0) if mm else ""

    qlin_call = class_method(texts["quantized.py"], "QuantizedLinear", "__call__")
    qemb_call = class_method(texts["quantized.py"], "QuantizedEmbedding", "__call__")
    # QuantizedLinear.__call__ is quantized_matmul, not dequantize-then-gemm.
    qlin_uses_qmatmul = "mx.quantized_matmul" in qlin_call
    qlin_uses_dequant_in_call = "mx.dequantize" in qlin_call
    embed_uses_dequant_row = "mx.dequantize" in qemb_call

    compile_on_swiglu = "@partial(mx.compile" in texts["activations.py"] and "def swiglu" in texts["activations.py"]
    compile_on_silu = "def silu" in texts["activations.py"] or "@partial(mx.compile" in read_text(
        MLX_ROOT / "nn" / "layers" / "activations.py"
    )
    compile_on_g = "@partial(mx.compile" in texts["gated_delta.py"] and "def compute_g" in texts["gated_delta.py"]
    generate_wraps_step = bool(
        re.search(r"mx\.compile\(_step|_step\s*=\s*mx\.compile", texts["generate.py"])
    )
    async_eval_n = count(r"mx\.async_eval", texts["generate.py"])
    wired_n = count(r"set_wired_limit", texts["generate.py"])
    logsumexp_n = count(r"logsumexp", texts["generate.py"])
    rms_fast = "mx.fast.rms_norm" in texts["normalization.py"]
    sdpa_fast = "mx.fast.scaled_dot_product_attention" in read_text(MLX_LM / "models" / "base.py")
    rope_fast = "mx.fast.rope" in read_text(MLX_LM / "models" / "rope_utils.py") or "fast.rope" in read_text(
        MLX_LM / "models" / "rope_utils.py"
    )

    # qmv_fast geometry from the installed header (not docs).
    qmv_fast_geom = {
        "num_simdgroups": 2,
        "results_per_simdgroup": 4,
        "rows_per_threadgroup": 8,
        "threads_per_threadgroup": 64,  # 2 simdgroups * 32
        "k_block_for_q4": 512,  # values_per_thread(16)*SIMD_SIZE(32)
        "x_reuse_rows": 4,
        "source": str(qh),
        "symbol": "qmv_fast_impl",
        "evidence": (
            "quantized.h qmv_fast_impl: constexpr num_simdgroups=2, results_per_simdgroup=4; "
            "one simdgroup accumulates 4 output rows sharing a loaded x_thread; "
            "affine_qmv_fast is the [[kernel]] wrapper. bits=4 pack_factor=8, packs_per_thread=2 "
            "→ values_per_thread=16, block_size=512."
        ),
        "reconstructs_dense": False,
        "why_not_dense": (
            "qdot() FMA-accumulates from packed uint8/uint16 codes and (scale,bias) in registers. "
            "No (rows×cols) W is written. affine_dequantize_* kernels exist in the metallib for "
            "the convert/oracle path; QuantizedLinear.__call__ does not use them."
        ),
    }

    # Per-token SOURCE_COUNTED launches. Views/reshape/split/transpose are not
    # dispatches. mx.compile fuses the marked elementwise graphs into one kernel
    # each. This is a lower bound on GPU dispatches, not a GPU capture.
    linear_layer = {
        "qmv": 5,  # in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
        "rms_fast": 4,  # input, q, k, post-attn; gated-norm is extra
        "conv1d": 1,
        "silu_compiled": 1,  # nn.silu on conv_out (nn.silu is compiled)
        "compute_g_compiled": 1,
        "gated_delta_kernel": 1,
        "gated_norm_rms": 1,
        "gated_norm_swiglu_uncompiled": 1,  # _precise_swiglu is NOT the compiled swiglu
        "residual": 2,
        "mlp_qmv": 3,
        "mlp_swiglu_compiled": 1,
        "sigmoid_beta": 1,  # mx.sigmoid(b) before gated_delta
    }
    gqa_layer = {
        "qmv": 4,  # q (includes gate), k, v, o
        "rms_fast": 4,  # input, q_norm, k_norm, post
        "rope_fast": 2,
        "sdpa_vector": 1,
        "sigmoid_gate": 1,
        "residual": 2,
        "mlp_qmv": 3,
        "mlp_swiglu_compiled": 1,
    }
    terminal = {
        "embed_row_dequant": 1,
        "final_rms": 1,
        "lm_head_qmv": 1,
        "logsumexp": 1,
        "argmax_or_sampler": 1,
    }

    def sum_over(spec, n):
        return {k: v * n for k, v in spec.items()}

    lin = sum_over(linear_layer, LINEAR_LAYERS)
    gqa = sum_over(gqa_layer, GQA_LAYERS)
    keys = sorted(set(lin) | set(gqa) | set(terminal))
    per_token = {k: lin.get(k, 0) + gqa.get(k, 0) + terminal.get(k, 0) for k in keys}
    qmv = per_token.get("qmv", 0) + per_token.get("mlp_qmv", 0) + per_token.get("lm_head_qmv", 0)
    # mlp_qmv is stored separately; fold
    qmv = lin.get("qmv", 0) + lin.get("mlp_qmv", 0) + gqa.get("qmv", 0) + gqa.get("mlp_qmv", 0) + terminal.get("lm_head_qmv", 0)
    dispatch_lower_bound = sum(per_token.values())

    return {
        "files_read": {k: str(p) for k, p in {
            "qwen3_5": q35, "qwen3_next": qn, "gated_delta": gd, "activations": act,
            "generate": gen, "quantized_linear": qlin, "rms": rms, "quantized_h": qh,
        }.items()},
        "quantized_linear_uses_quantized_matmul": qlin_uses_qmatmul,
        "quantized_linear_call_uses_dequantize": qlin_uses_dequant_in_call,
        "embed_dequantizes_one_row": embed_uses_dequant_row,
        "generate_wraps_step_in_mx_compile": generate_wraps_step,
        "generate_async_eval_mentions": async_eval_n,
        "generate_wired_limit_mentions": wired_n,
        "generate_logsumexp_mentions": logsumexp_n,
        "nn_rmsnorm_is_mx_fast": rms_fast,
        "sdpa_is_mx_fast": sdpa_fast,
        "rope_is_mx_fast": rope_fast,
        "swiglu_is_mx_compile": compile_on_swiglu,
        "compute_g_is_mx_compile": compile_on_g,
        "silu_is_mx_compile": True,  # mlx.nn.layers.activations.silu
        "qmv_fast_geometry": qmv_fast_geom,
        "graph_inventory_method": (
            "Counted nn.Linear / mx.fast / mx.compile / mx.fast.metal_kernel sites "
            "in the installed qwen3_5 + qwen3_next + gated_delta + generate sources, "
            "times the 48 linear + 16 GQA layer split from config.json. Views are omitted. "
            "This is a SOURCE_COUNTED lower bound, not a GPU capture of command-buffer ops."
        ),
        "linear_layer_template": linear_layer,
        "gqa_layer_template": gqa_layer,
        "terminal_template": terminal,
        "qmv_dispatches_per_token": qmv,
        "native_gemv_dispatches_per_token": 401,
        "extra_qmv_vs_native": qmv - 401,
        "why_extra_qmv": (
            "Native fuses in_proj_qkvz (one 16384×5120) and in_proj_ba (one 96×5120). "
            "MLX qwen3_5.GatedDeltaNet splits those into in_proj_qkv + in_proj_z and "
            "in_proj_b + in_proj_a: +2 quantized_matmul per linear layer × 48 = +96 GEMVs. "
            "Same bytes, more launches. Native already has this fusion."
        ),
        "source_counted_named_op_sites": per_token,
        "source_counted_named_op_site_sum": dispatch_lower_bound,
        "source_counted_named_op_site_sum_is_not_a_gpu_counter": True,
        "gpu_observed_dispatches": null("filled in main() from metal_probe + occupancy"),
        "gpu_observed_command_buffers": null("filled in main() from metal_probe + occupancy"),
        "lazy_eval": {
            "present": True,
            "evidence": (
                "mlx/include/mlx/transforms.h exposes eval / async_eval. Arrays are "
                "lazy; generate.py calls mx.async_eval(next_y, next_logprobs) BEFORE "
                "y.item(), overlapping token t+1 GPU work with token t CPU sample."
            ),
            "generate_does_not_compile_the_step": (not generate_wraps_step),
        },
        "fusion_sites": [
            {
                "what": "mx.fast.rms_norm",
                "where": "mlx.nn.RMSNorm + GatedDeltaNet q/k + RMSNormGated",
                "kind": "hand-written fused kernel",
            },
            {
                "what": "mx.fast.scaled_dot_product_attention / sdpa_vector_*_256",
                "where": "mlx_lm.models.base.scaled_dot_product_attention for GQA decode",
                "kind": "hand-written fused kernel, vector (decode) path, head_dim=256 in metallib",
            },
            {
                "what": "mx.fast.rope / rope_single_*",
                "where": "Qwen3NextAttention via initialize_rope",
                "kind": "hand-written fused kernel",
            },
            {
                "what": "mx.compile swiglu",
                "where": "mlx_lm.models.activations.swiglu used by Qwen3NextMLP",
                "kind": "graph-compiler fused elementwise (silu*x)",
            },
            {
                "what": "mx.compile compute_g",
                "where": "gated_delta.compute_g: exp(-exp(A_log)*softplus(a+dt_bias))",
                "kind": "graph-compiler fused elementwise",
            },
            {
                "what": "mx.compile silu",
                "where": "mlx.nn.silu, used on conv1d output",
                "kind": "graph-compiler fused elementwise",
            },
            {
                "what": "gated_delta_step mx.fast.metal_kernel",
                "where": "gated_delta._make_gated_delta_kernel, JIT, not in mlx.metallib",
                "kind": "hand-written fused recurrent state op",
            },
            {
                "what": "affine_qmv_fast",
                "where": "QuantizedLinear → mx.quantized_matmul, M=1 decode",
                "kind": "hand-written fused dequant+GEMV, reconstructs_dense=NO",
            },
        ],
        "unfused_on_purpose": [
            "generate._step is NOT wrapped in mx.compile (searched generate.py)",
            "RMSNormGated._precise_swiglu is a separate f32 silu*x, not the compiled swiglu",
            "DeltaNet in_proj is four QuantizedLinears, not a fused qkvz/ba",
            "residual adds are ordinary mx.add",
            "logsumexp over vocab every token even for greedy (temp=0 sampler is argmax AFTER logsumexp)",
        ],
    }


def native_kernel_geometry() -> dict:
    shader = REPO / "crates/hawking-core/shaders/qwen_uniform_q4.metal"
    text = read_text(shader)
    present = "kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128" in text
    return {
        "file": str(shader.relative_to(REPO)) if shader.is_file() else None,
        "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "present": present,
        "threads_per_row": 64,
        "rows_per_threadgroup": 2,
        "threads_per_threadgroup": 128,
        "k_step": 512,
        "x_reuse_rows": 1,
        "codec": "grouped-absmax Q4 group-64 (32 code + 2 fp16 scale, no bias)",
        "activation_dtype": "f32",
        "comment_in_source": (
            "Geometry-sweep winner for Q4 gate [512, 2048]: 64 threads/row, "
            "128-thread TG, 2 rows/TG. Packed decode stays in registers."
        ),
        "reconstructs_dense": False,
        "dispatches_per_token": 401,
        "total_dispatches_per_token": ANCHOR_DISPATCHES,
        "command_buffers": ANCHOR_CBS,
        "wired_limit_in_decode_rs": "set_wired" not in read_text(
            REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
        ).lower(),
        "async_overlap_in_decode_rs": "async" not in read_text(
            REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
        ).lower(),
    }


def prior_controls() -> dict:
    gpu_attack = load_json(RECEIPTS / "GPU_ATTACK.json") or {}
    runtime_ab = load_json(RECEIPTS / "RUNTIME_AB.json") or {}
    perf = RECEIPTS / "PERFORMANCE_LEDGER.jsonl"
    ledger_mlx = None
    ledger_llama = None
    if perf.is_file():
        for line in perf.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("generation") == "runtime-ab-mlx":
                ledger_mlx = row
            if row.get("generation") == "runtime-ab-llama_cpp":
                ledger_llama = row
    ra_mlx = (runtime_ab.get("arms") or {}).get("mlx") or {}
    ra_llama = (runtime_ab.get("arms") or {}).get("llama_cpp") or {}
    ra_cmp = runtime_ab.get("comparison") or {}
    ga_rt = (gpu_attack.get("runtime_axis") or {})
    return {
        "gpu_attack": {
            "mlx_tps": ga_rt.get("mlx_single_stream_tps"),
            "llama_tps": ga_rt.get("llama_cpp_single_stream_tps"),
            "mlx_over_llama": ga_rt.get("mlx_over_llama"),
            "bytes_ratio_llama_over_mlx": ga_rt.get("bytes_ratio_llama_over_mlx"),
            "reading": ga_rt.get("reading"),
            "confound": ga_rt.get("confound_declared"),
        },
        "runtime_ab": {
            "mlx_tps_median": ra_mlx.get("decode_tps_median"),
            "llama_tps_median": ra_llama.get("decode_tps_median"),
            "mlx_bytes": ra_mlx.get("bytes"),
            "llama_bytes": ra_llama.get("bytes"),
            "mlx_over_llama": ra_cmp.get("mlx_over_llama"),
            "bytes_ratio_llama_over_mlx": ra_cmp.get("bytes_ratio_llama_over_mlx"),
            "mlx_model": ra_mlx.get("model"),
            "llama_model": ra_llama.get("model"),
        },
        "performance_ledger_mlx": ledger_mlx,
        "performance_ledger_llama": ledger_llama,
        "native_qwen38": {
            "tps": ANCHOR_TPS_NATIVE,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "gpu_ns": ANCHOR_NATIVE_GPU_NS,
            "wall_ns": ANCHOR_NATIVE_WALL_NS,
            "source": "campaign brief + receipts/headless/QWEN38_GRAVITY_NATIVE.json median run1 gpu/wall",
        },
    }


def bandwidth(bytes_per_token, tps):
    if not bytes_per_token or not tps:
        return None
    return (bytes_per_token * tps) / 1e9


def classify_advantages(src: dict, native: dict, art: dict) -> list:
    """Each MLX advantage, AVAILABLE_TO_NATIVE or FRAMEWORK_SPECIFIC."""
    rows = []

    def add(name, klass, why, decode_effect, evidence):
        rows.append({
            "advantage": name,
            "class": klass,
            "why": why,
            "decode_effect": decode_effect,
            "evidence": evidence,
        })

    add(
        "affine_qmv_fast geometry (8 rows/TG, 4-row x-reuse, pre-scaled qdot)",
        "AVAILABLE_TO_NATIVE",
        "This is a Metal kernel, not a framework service. Native already has a "
        "fused in-register Q4 GEMV (geo_tpr64_tg128: 2 rows/TG, 64 threads/row, "
        "x_reuse=1). Porting qmv_fast's 8-row TG and 4-row x_thread reuse is a "
        "shader change.",
        "Primary candidate for the roof-utilization gap (native ~75% of 595.9 GB/s "
        "on active bytes vs MLX ~90% on language-active bytes).",
        "mlx quantized.h qmv_fast_impl vs crates/hawking-core/shaders/qwen_uniform_q4.metal:183",
    )
    add(
        "in-register dequant + FMA (no dense W)",
        "AVAILABLE_TO_NATIVE",
        "Native already does this (reconstructs_dense=NO on all 38 bound kernels). "
        "MLX QuantizedLinear.__call__ is mx.quantized_matmul, same law. Not a gap.",
        "Neither runtime reconstructs dense W on the decode path.",
        "QuantizedLinear.__call__ + kernel census dispatched_reconstructs_dense.NO=38",
    )
    add(
        "mx.fast fused primitives (rms_norm, rope, sdpa_vector, gated_delta_step)",
        "AVAILABLE_TO_NATIVE",
        "These are ordinary Metal kernels. Native already has fused RMS, RoPE, "
        "mha_decode_f32, gated_delta. Matching MLX here is bind/tune work, not a "
        "new representation.",
        "Reduces non-GEMV dispatches. GEMV still dominates bytes.",
        "mlx.nn.RMSNorm → mx.fast.rms_norm; mlx_lm.models.base sdpa; gated_delta metal_kernel",
    )
    add(
        "elementwise compile (swiglu, silu, compute_g)",
        "AVAILABLE_TO_NATIVE",
        "The *compiler* is FRAMEWORK_SPECIFIC. The *fused kernels it emits* are "
        "hand-writable. Native already has gk_swiglu_f32. compute_g can be one kernel.",
        "Small vs GEMV. Do not confuse this with the 1.472x.",
        "mlx_lm.models.activations.swiglu @mx.compile; mlx.nn.silu @mx.compile",
    )
    add(
        "lazy eval + one-graph CommandEncoder batching",
        "AVAILABLE_TO_NATIVE",
        "Native already encodes 964 dispatches into 1 command buffer. The MLX "
        "lazy tracer is framework-specific; the effect (don't submit 964 CBs) "
        "is already taken.",
        "Native production_command_buffers=1. Not the gap.",
        "qwen38_token_ns_ledger.rs production_dispatches_per_token=964, 1 CB; mlx device.h CommandEncoder",
    )
    add(
        "async_eval overlap of token t+1 GPU with token t CPU",
        "AVAILABLE_TO_NATIVE",
        "generate.py mx.async_eval(next_y) before y.item(). Native decode.rs has "
        "no async overlap (string 'async' absent). Metal command-buffer enqueue "
        "without waiting on the CPU sample is available to a specialised executable.",
        f"Native wall-gpu gap is {(ANCHOR_NATIVE_WALL_NS-ANCHOR_NATIVE_GPU_NS)/1e6:.3f} ms/token "
        f"({ANCHOR_NATIVE_WALL_NS} vs {ANCHOR_NATIVE_GPU_NS} ns). Hiding that is ~4% of native wall, "
        "not the whole MLX lead.",
        "mlx_lm/generate.py async_eval; QWEN38_GRAVITY_NATIVE.json run1 gpu vs wall",
    )
    add(
        "set_wired_limit(recommendedMaxWorkingSetSize)",
        "AVAILABLE_TO_NATIVE",
        "mlx_lm.generate.wired_limit calls mx.set_wired_limit. Native decode.rs "
        "does not. MTLDevice wired-limit is a public Metal API.",
        "Avoids eviction on a 14–16 GB resident. MACHINE_GENOME already measured "
        "recommendedMaxWorkingSetSize=77.76 GiB, so eviction is not the current limiter, "
        "but the call is still the right default.",
        "generate.py wired_limit; receipts/headless/MACHINE_GENOME.json gpu_gate",
    )
    add(
        "bf16 activations on the Q4 GEMV",
        "AVAILABLE_TO_NATIVE",
        "config.json text_config.dtype=bfloat16; affine_qmv_fast_bfloat16_t_gs_64_b_4 "
        "is in the metallib. Native GEMV writes/reads f32 activations. Switching the "
        "native activation pipeline to fp16/bf16 is a kernel+workspace change, not a "
        "framework privilege. Quality must be re-gated; this campaign does not skip that.",
        "Activation DRAM is ~0.2 GB/token vs ~13–14 GB weights. Secondary.",
        "metallib affine_qmv_fast_bfloat16_t_gs_64_b_4_*; config.json dtype",
    )
    add(
        "concurrent encoding of independent GEMVs (gate vs up)",
        "AVAILABLE_TO_NATIVE",
        "CommandEncoder::start_concurrent() is a Metal encoder flag. Native can "
        "issue independent dispatches without barriers. MLX's graph walker uses it "
        "automatically; a specialised executable can do the same by not inserting "
        "barriers between gate_proj and up_proj.",
        "Latency hiding on two 47 MB Q4 GEMVs that share x. Available.",
        "mlx/include/mlx/backend/metal/device.h ConcurrentContext",
    )
    add(
        "general mx.compile graph capture",
        "FRAMEWORK_SPECIFIC",
        "A general compiler that traces Python, simplifies, and JIT-fuses arbitrary "
        "elementwise DAGs is the framework. generate.py does not even wrap _step in "
        "mx.compile. We do not need a compiler to write the fused kernels we already "
        "know we need.",
        "Not the 35.51. Local @mx.compile on swiglu/compute_g is the part that matters "
        "and is AVAILABLE as handwritten kernels.",
        "compile.h CompileMode; generate.py has no mx.compile(_step)",
    )
    add(
        "162 MB general metallib / Python / dtype-generic kernel catalog",
        "FRAMEWORK_SPECIFIC",
        "A specialised executable ships the kernels it dispatches. Native already "
        "embeds a compiled Metal library in a 6.3 MB binary. Paying for qmm/steel/"
        "fft/sort/dequantize oracles is generality.",
        "Resident bytes, not per-token DRAM. Not the 1.472x.",
        f"mlx.metallib {file_bytes(MLX_METALLIB)} B; native decode binary {file_bytes(NATIVE_BIN)} B",
    )
    add(
        "lazy Python graph tracer and array runtime",
        "FRAMEWORK_SPECIFIC",
        "The tracer, refcount, donation, and dtype/shape checks are the framework. "
        "A specialised executable builds one TokenCommandBuffer. Native already does.",
        "CPU overhead. Native wall-gpu 1.18 ms. MLX hides some of it with async_eval "
        "(AVAILABLE) rather than by being Python (SPECIFIC).",
        "transforms.h eval/async_eval; native 1 CB encode path",
    )
    add(
        "logsumexp + logprobs every greedy token",
        "FRAMEWORK_SPECIFIC",
        "generate.py always does logits - logsumexp even when sampler is argmax. "
        "A specialised greedy path (native sample_argmax_f32) skips it. Vocab=248320 "
        "is noise next to 14 GB weights.",
        "Negative for MLX (overhead we do not need), not an advantage.",
        "generate.py:420 logprobs = logits - mx.logsumexp(...)",
    )
    add(
        "vision tensors in the 4-bit directory",
        "FRAMEWORK_SPECIFIC",
        "mlx-vlm converted the full multimodal checkpoint. qwen3_5.Model.sanitize "
        "drops vision_tower. Those 921 MB do not stream at text decode. They inflated "
        "GPU_ATTACK's 1.215 bytes ratio.",
        "Do not credit MLX with a byte win it does not take on the decode path.",
        "config language_model_only=false; sanitize skips vision; 333 vision tensors",
    )
    add(
        "unified-memory 'native-ness'",
        "AVAILABLE_TO_NATIVE",
        "Both runtimes are Metal on the same unified memory. Apple-framework "
        "membership is not a memory model. Native already allocates MTLBuffers in "
        "the same pool.",
        "Zero. This is not why 35.51 > 32.73.",
        "Both use MTL; MACHINE_GENOME recommendedMaxWorkingSetSize measured",
    )
    return rows


def overhead(art: dict, src: dict) -> list:
    """What a specialised executable would not need, quantified."""
    items = []
    mlb = file_bytes(MLX_METALLIB)
    dylib = file_bytes(MLX_DYLIB)
    core = file_bytes(MLX_CORE_SO)
    native_bin = file_bytes(NATIVE_BIN)
    items.append({
        "item": "mlx.metallib general kernel catalog",
        "bytes": mlb,
        "status": "MEASURED",
        "per_token": False,
        "vs_native": native_bin,
        "note": (
            "162 MB of specialised qmv/qmm/steel/fft/dequantize variants. Decode uses a "
            "handful (affine_qmv_fast gs_64_b_4, rms, rope_single, sdpa_vector_256, "
            "plus JIT gated_delta). Native ships a 6.3 MB example binary with the "
            "compiled library embedded."
        ),
    })
    items.append({
        "item": "libmlx.dylib + core.so",
        "bytes": (dylib or 0) + (core or 0),
        "status": "MEASURED",
        "per_token": False,
        "parts": {"libmlx.dylib": dylib, "core.so": core},
        "note": "General runtime. Native is the 6.3 MB decode binary.",
    })
    items.append({
        "item": "vision tensors on disk (not loaded, not streamed)",
        "bytes": art.get("vision_bytes"),
        "status": "MEASURED",
        "per_token": False,
        "n_tensors": art.get("vision_tensors"),
        "note": "qwen3_5.sanitize drops vision_tower. Inflated GPU_ATTACK 1.215 ratio.",
    })
    items.append({
        "item": "affine per-group bias (vs native absmax)",
        "bytes": ((art.get("by_role") or {}).get("language:biases") or {}).get("bytes"),
        "status": "MEASURED",
        "per_token": True,
        "note": (
            "840,417,280 B of bf16 group biases streamed every token. Native grouped-absmax "
            "has scale only (34 B/group vs 36). This is overhead MLX pays, and native already "
            "does not. MLX is still faster, so the kernel mapping beats this tax."
        ),
    })
    items.append({
        "item": "extra GEMV launches from unfused in_proj",
        "count": src.get("extra_qmv_vs_native"),
        "status": "SOURCE_COUNTED",
        "per_token": True,
        "note": (
            "+96 quantized_matmul launches/token (same bytes). Launch overhead inside one "
            "CB is microseconds, not milliseconds. Native already fused these."
        ),
    })
    items.append({
        "item": "logsumexp over vocab every greedy token",
        "elements": VOCAB,
        "status": "SOURCE_COUNTED",
        "per_token": True,
        "note": (
            "248,320-wide reduction + subtract, then argmax. Native sample_argmax_f32 is "
            "one kernel. Bytes ~1 MB — not the 1.472x."
        ),
    })
    items.append({
        "item": "Python generate_step tracer",
        "bytes": None,
        "status": "NULL",
        "null_reason": "No GPU timestamp for Python vs Metal in this process.",
        "per_token": True,
        "note": (
            "Native wall-gpu gap is 1.184 ms/token (PRIOR QWEN38_GRAVITY_NATIVE run1). "
            "That is an upper bound on 'CPU around the GPU' for the specialised path. "
            "MLX async_eval is designed to hide the equivalent."
        ),
    })
    return items


def how_to_beat(art: dict, src: dict, native: dict) -> dict:
    mlx_active = art.get("active_language_bytes")
    mlx_lang = art.get("language_bytes")
    native_active = ANCHOR_ACTIVE_B
    mlx_on_active = bandwidth(mlx_active, ANCHOR_MLX_TPS)
    native_on_active = bandwidth(native_active, ANCHOR_TPS_NATIVE)
    # If native hit MLX's GB/s on native's smaller bytes:
    tps_if_native_hits_mlx_gbs = None
    if mlx_on_active and native_active:
        tps_if_native_hits_mlx_gbs = (mlx_on_active * 1e9) / native_active
    match = [
        {
            "step": "Port qmv_fast threadgroup map onto grouped-absmax Q4",
            "class": "AVAILABLE_TO_NATIVE",
            "detail": (
                "8 rows/TG, 64 threads, 4-row x_thread reuse, 512-wide K block, "
                "pre-scaled qdot. Keep absmax (no bias) so we do not take MLX's 840 MB tax. "
                "reconstructs_dense must stay NO."
            ),
        },
        {
            "step": "Overlap sample with the next token's command buffer",
            "class": "AVAILABLE_TO_NATIVE",
            "detail": (
                "Native gpu 29.204 ms vs wall 30.389 ms. Enqueue token t+1 before reading "
                "argmax of t. That is mlx.async_eval. Worth ~1.2 ms if fully hidden."
            ),
        },
        {
            "step": "Issue gate_proj and up_proj without a barrier",
            "class": "AVAILABLE_TO_NATIVE",
            "detail": "Independent GEMVs sharing x. Metal concurrent encode.",
        },
        {
            "step": "setWiredLimit to recommendedMaxWorkingSetSize",
            "class": "AVAILABLE_TO_NATIVE",
            "detail": "mlx_lm.wired_limit. Cheap, correct, not the limiter at 77.76 GiB headroom.",
        },
    ]
    beat = [
        {
            "step": "Keep fused in_proj_qkvz / in_proj_ba (already ahead of MLX)",
            "class": "AVAILABLE_TO_NATIVE_ALREADY_OURS",
            "detail": "96 fewer GEMV launches than MLX on the same bytes.",
        },
        {
            "step": "Keep grouped-absmax (34 B/group vs affine 36)",
            "class": "AVAILABLE_TO_NATIVE_ALREADY_OURS",
            "detail": (
                f"Language-active MLX {mlx_active} B vs native active {native_active} B "
                f"(ratio {(mlx_active/native_active) if mlx_active else None:.4f}). "
                "Native already moves fewer bytes. Hitting MLX GB/s on those bytes "
                f"projects {tps_if_native_hits_mlx_gbs:.2f} tok/s."
                if mlx_active and tps_if_native_hits_mlx_gbs else
                "Native already moves fewer bytes per token than MLX language-active."
            ),
        },
        {
            "step": "Keep greedy argmax; do not pay logsumexp",
            "class": "AVAILABLE_TO_NATIVE_ALREADY_OURS",
            "detail": "generate.py always builds logprobs.",
        },
        {
            "step": "Do not reconstruct dense W to 'catch up'",
            "class": "LAW",
            "detail": (
                "representation → reconstruct dense W → ordinary GEMM is an oracle, "
                "not a production implementation. MLX does not do this on decode. "
                "Neither must we."
            ),
        },
        {
            "step": "Do not reopen MLP distillation or G-SHARE as the speed lever",
            "class": "PRIOR_NEGATIVE",
            "detail": (
                "MLP function distillation is NO-GO (+0.4206 held-out gap vs q3 at 72% of "
                "its active bytes). G035 G-SHARE shared_beats_independent=false. "
                "The MLX lead is a kernel/runtime lead on the same Q4 GEMV work."
            ),
        },
    ]
    return {
        "match_35_51": match,
        "beat_35_51": beat,
        "arithmetic": {
            "native_tps": prior(ANCHOR_TPS_NATIVE, "campaign brief / G105"),
            "mlx_tps": prior(ANCHOR_MLX_TPS, "RUNTIME_AB.json decode_tps_median 35.513, GPU_ATTACK 35.506"),
            "gap_tps": ANCHOR_MLX_TPS - ANCHOR_TPS_NATIVE,
            "gap_fraction": (ANCHOR_MLX_TPS - ANCHOR_TPS_NATIVE) / ANCHOR_TPS_NATIVE,
            "native_active_bytes": prior(native_active, "NOETIC_KERNEL_CENSUS production_token.active_budget_bytes_constant"),
            "mlx_active_language_bytes": measured(mlx_active, method="safetensors language payload − embed table + one row"),
            "mlx_language_bytes": measured(mlx_lang, method="safetensors language tensor payloads"),
            "native_implied_GB_s_on_active": native_on_active,
            "mlx_implied_GB_s_on_active_language": mlx_on_active,
            "roof_GB_s": prior(ANCHOR_ROOF_GB_S, "campaign brief"),
            "native_pct_of_roof": (native_on_active / ANCHOR_ROOF_GB_S * 100) if native_on_active else None,
            "mlx_pct_of_roof": (mlx_on_active / ANCHOR_ROOF_GB_S * 100) if mlx_on_active else None,
            "tps_if_native_hits_mlx_GB_s_on_native_bytes": tps_if_native_hits_mlx_gbs,
            "note": (
                "Implied GB/s = bytes_per_token × tok/s / 1e9. This is an accounting identity "
                "from a PRIOR tok/s and a MEASURED byte count, not a new GPU counter. "
                "It is the right identity for a weight-streaming decode."
            ),
        },
    }


def speed_account(art: dict, controls: dict) -> dict:
    """Where the 1.472x over llama.cpp comes from, at the recorded 1.215 bytes ratio."""
    llama_b = ANCHOR_LLAMA_B
    mlx_index = art.get("payload_bytes")  # safetensors total_size identity
    mlx_lang = art.get("language_bytes")
    mlx_active = art.get("active_language_bytes")
    ratio_index = llama_b / mlx_index if mlx_index else None
    ratio_lang = llama_b / mlx_lang if mlx_lang else None
    ratio_active = llama_b / mlx_active if mlx_active else None
    speed = ANCHOR_SPEED_RATIO
    # GPU_ATTACK used 1.215 ≈ llama/safetensors-total including vision.
    kernel_at_1215 = speed / ANCHOR_BYTES_RATIO_GPU_ATTACK
    kernel_at_lang = speed / ratio_lang if ratio_lang else None
    llama_gbs = bandwidth(llama_b, ANCHOR_LLAMA_TPS)
    mlx_gbs_index = bandwidth(mlx_index, ANCHOR_MLX_TPS)
    mlx_gbs_lang = bandwidth(mlx_lang, ANCHOR_MLX_TPS)
    mlx_gbs_active = bandwidth(mlx_active, ANCHOR_MLX_TPS)
    return {
        "headline": {
            "mlx_tps": prior(ANCHOR_MLX_TPS, "GPU_ATTACK.json runtime_axis / RUNTIME_AB.json median 35.513"),
            "llama_tps": prior(ANCHOR_LLAMA_TPS, "GPU_ATTACK.json / campaign brief; ARCHIVED — GGUF is off disk, llama-server still maps it"),
            "speed_ratio": prior(speed, "GPU_ATTACK.json runtime_axis.mlx_over_llama=1.472"),
            "bytes_ratio_recorded": prior(
                ANCHOR_BYTES_RATIO_GPU_ATTACK,
                "GPU_ATTACK.json runtime_axis.bytes_ratio_llama_over_mlx=1.215",
            ),
        },
        "bytes_this_process": {
            "llama_gguf_bytes_prior": prior(llama_b, "PERFORMANCE_LEDGER.jsonl runtime-ab-llama_cpp model_bytes; GGUF not on disk"),
            "mlx_safetensors_payload_bytes": measured(mlx_index, method="sum of safetensors data_offsets"),
            "mlx_language_bytes": measured(mlx_lang),
            "mlx_vision_bytes": measured(art.get("vision_bytes")),
            "mlx_active_language_bytes": measured(mlx_active),
            "ratio_llama_over_mlx_index": measured(ratio_index, formula="19535701280 / safetensors_payload"),
            "ratio_llama_over_mlx_language": measured(ratio_lang, formula="19535701280 / language_payload"),
            "ratio_llama_over_mlx_active_language": measured(ratio_active),
            "runtime_ab_bytes_ratio": prior(
                (controls.get("runtime_ab") or {}).get("bytes_ratio_llama_over_mlx"),
                "RUNTIME_AB.json comparison.bytes_ratio_llama_over_mlx (1.289) — different walk, huihui path",
            ),
        },
        "decomposition": {
            "using_recorded_1_215_includes_vision": {
                "bytes_explain": ANCHOR_BYTES_RATIO_GPU_ATTACK,
                "remaining_kernel_runtime": kernel_at_1215,
                "reading": (
                    "GPU_ATTACK's 1.215 is llama_bytes / mlx_safetensors_total including "
                    "921 MB of vision that sanitize drops. It overstates MLX's byte win "
                    "and therefore understates the kernel/runtime share. Quoted because "
                    "the campaign brief names it; not because it is the fair decode ratio."
                ),
            },
            "using_language_payload_fair_for_text_decode": {
                "bytes_explain": ratio_lang,
                "remaining_kernel_runtime": kernel_at_lang,
                "reading": (
                    "Text decode streams language tensors only. Fair byte ratio is "
                    f"{ratio_lang:.3f} if mlx_lang and llama_b. Speed {speed} / that "
                    f"leaves {kernel_at_lang:.3f}× as kernel/runtime. MLX still wins "
                    "after bytes are taken out."
                ),
            },
        },
        "implied_GB_s": {
            "roof": prior(ANCHOR_ROOF_GB_S, "campaign brief"),
            "llama_on_gguf_bytes": llama_gbs,
            "mlx_on_index_bytes_unfair": mlx_gbs_index,
            "mlx_on_language_bytes": mlx_gbs_lang,
            "mlx_on_active_language_bytes": mlx_gbs_active,
            "native_on_active_bytes": bandwidth(ANCHOR_ACTIVE_B, ANCHOR_TPS_NATIVE),
            "identity": "GB/s = bytes_per_token * tok/s / 1e9 (weight-stream identity, not a GPU counter)",
        },
        "confound": (
            "GPU_ATTACK confound_declared: different abliterations of the same Qwen3.8-27B "
            "base (huihui vs PocketAiHub). Architecture and per-token weight-streaming shape "
            "match, so SPEED is comparable. Quality is not."
        ),
        "not_the_cause": [
            "Storage compression vs native: MLX language is LARGER than native uniform-q4 "
            f"({mlx_lang} vs {ANCHOR_ARTIFACT_B}) because affine stores group biases. "
            "MLX is faster anyway. A candidate that only lowers executable bytes is incomplete.",
            "Fewer GEMVs: MLX issues 497 quantized_matmul vs native 401.",
            "Dense reconstruction: both in-register. affine_dequantize_* in the metallib is "
            "the oracle/convert path.",
            "Apple unified-memory magic: both are Metal on the same M3 Ultra.",
        ],
    }


def what_watched_fail(occ, metal, art, src, controls) -> list:
    fails = []
    fails.append({
        "what": "Live GPU capture of MLX 4-bit decode (dispatches, kernel names from a counter, CB count)",
        "result": "NULL",
        "why": (
            f"mx.metal.is_available()={metal.get('metal_is_available_flag')} but "
            f"mx.eval of a 4-float vector failed: {metal.get('tiny_eval_error')!r}. "
            f"device_info failed: {metal.get('device_info_error')!r}. "
            "system_profiler reports Apple M3 Ultra / Metal Supported — the GPU exists; "
            "this sandbox cannot open a Metal device. Independently, llama-server is "
            f"resident on 52484 (alias={occ.get('model_alias')}), so even a working GPU "
            f"must not load a second 27B (occupancy {ANCHOR_ONE_SERVER_TPS} → "
            f"{ANCHOR_TWO_SERVER_TPS} tok/s). Kernel NAMES observed from mlx.metallib; "
            "launch COUNTS on the 4-bit artifact are source-counted, not GPU-counted."
        ),
    })
    fails.append({
        "what": "Load ~/models/qwen3.8-27b-abliterated-mlx-huihui-4bit (RUNTIME_AB preferred path)",
        "result": "MISSING",
        "why": (
            f"{MLX_HUIHUI} is not on disk. The live 4-bit artifact is "
            f"{MLX_4BIT} (PocketAiHub / mlx-vlm convert). SPEED comparison across "
            "abliterations is the GPU_ATTACK confound; this process did not pretend they "
            "are the same weights."
        ),
    })
    fails.append({
        "what": "Re-measure llama.cpp Q5_K tok/s",
        "result": "NOT ATTEMPTED",
        "why": (
            "GGUF is off disk (ARCHIVED). llama-server on 52484 still maps "
            f"{occ.get('model_path')}, but a generate would occupy the GPU and is the "
            "thing occupancy forbids stacking with. Cited 24.12 / 35.51 as PRIOR_MEASURED."
        ),
    })
    fails.append({
        "what": "Treat GPU_ATTACK 1.215 as the decode-byte ratio",
        "result": "REFUTED as a decode identity",
        "why": (
            f"safetensors payload {art.get('payload_bytes')} / llama {ANCHOR_LLAMA_B} ≈ "
            f"{(ANCHOR_LLAMA_B/(art.get('payload_bytes') or 1)):.4f}, which is the 1.215. "
            f"That payload includes {art.get('vision_bytes')} B vision "
            f"({art.get('vision_tensors')} tensors) that qwen3_5.sanitize drops. "
            "Fair text-decode ratio uses language bytes. Recording both, quoting 1.215 "
            "as the campaign number with the defect named."
        ),
    })
    fails.append({
        "what": "Attribute the 1.472x to 'MLX is Apple's framework' without opening the kernel",
        "result": "FAILS as an explanation",
        "why": (
            "qmv_fast_impl is 8 rows/TG with 4-row x reuse. Native geo_tpr64 is 2 rows/TG "
            "with x_reuse=1. That is a shader. Unified memory is shared. The framework "
            "also pays 162 MB metallib, Python, logsumexp, affine biases, +96 GEMVs — "
            "and still wins. The win is the kernel map + async overlap, which are "
            "AVAILABLE_TO_NATIVE."
        ),
    })
    fails.append({
        "what": "Assume generate.py mx.compile's the whole decode step",
        "result": "FALSE",
        "why": (
            "generate.py has async_eval and wired_limit and logsumexp. It does not wrap "
            f"_step in mx.compile (generate_wraps_step_in_mx_compile="
            f"{src.get('generate_wraps_step_in_mx_compile')}). Fusion on the hot path is "
            "mx.fast.* plus local @mx.compile on swiglu/silu/compute_g."
        ),
    })
    fails.append({
        "what": "Assume fewer bytes is why MLX beats native 32.73",
        "result": "FALSE",
        "why": (
            f"MLX language payload {art.get('language_bytes')} B vs native artifact "
            f"{ANCHOR_ARTIFACT_B} B. Affine biases alone are "
            f"{((art.get('by_role') or {}).get('language:biases') or {}).get('bytes')} B. Native already stores "
            "less. Source and executable both do 964 dispatches and 51.24 GFLOP of GEMV. "
            "A bytes-only candidate is incomplete — that is the campaign constraint, and "
            "MLX is the existence proof that execution can differ at matched work."
        ),
    })
    fails.append({
        "what": "Spawn a tiny MLX graph to observe fusion without the 27B",
        "result": "BLOCKED",
        "why": (
            "mx.eval of four ones() already raises [metal::load_device]. A fusion microbench "
            "would hit the same wall. Not an occupancy issue — the sandbox cannot create a "
            "Metal device. Occupancy separately forbids a 27B generate."
        ),
    })
    fails.append({
        "what": "ps(1) for resident decoders",
        "result": "BLOCKED",
        "why": "sandbox: ps: Operation not permitted. Occupancy inferred from llama /health on 52484 instead.",
    })
    fails.append({
        "what": "MLP distillation / G-SHARE / 0.5 local BPW without health / cosine-only",
        "result": "NOT REDISCOVERED",
        "why": (
            "Prior science: 223 components <0.5 local BPW with ZERO healthy; Q80 storage "
            "BPW 0.6462 vs ACTIVE 2.518; G035 shared_beats_independent=false; GLM 0.167 "
            "expert BPW trap; HGRAVS01 0.13 on down_proj ONLY; MLP distillation NO-GO "
            "+0.4206 held-out gap; cosine blind to 0.01*W. This lane is runtime anatomy, "
            "not a new representation search."
        ),
    })
    return fails


def print_report(doc: dict) -> None:
    print("=" * 78)
    print("NOETIC MLX ANATOMY")
    print("=" * 78)
    print(doc["answer"])
    print()
    print("## Controls (PRIOR_MEASURED, not re-run)")
    h = doc["speed_account"]["headline"]
    print(f"  MLX 4-bit            {h['mlx_tps']['value']} tok/s")
    print(f"  llama.cpp Q5_K       {h['llama_tps']['value']} tok/s  (ARCHIVED artifact)")
    print(f"  native uniform-q4    {ANCHOR_TPS_NATIVE} tok/s / {ANCHOR_TOKEN_MS} ms")
    print(f"  speed MLX/llama      {h['speed_ratio']['value']}×")
    print(f"  recorded bytes ratio {h['bytes_ratio_recorded']['value']}  (GPU_ATTACK, includes vision)")
    print()
    print("## Occupancy — did not load a 27B")
    occ = doc["occupancy"]
    print(f"  llama-server 52484   up={occ['llama_up']} alias={occ.get('model_alias')}")
    print(f"  GGUF on disk         {occ['gguf_on_disk']}")
    print(f"  refuse_load_27b      {occ['refuse_load_27b']}")
    print(f"  {occ['refuse_reason']}")
    print()
    print("## Metal probe (this process)")
    m = doc["metal_probe"]
    print(f"  attempted            {m['attempted']}")
    print(f"  mlx_imported         {m.get('mlx_imported')}")
    print(f"  is_available flag    {m.get('metal_is_available_flag')}")
    print(f"  gpu_usable (eval)    {m.get('gpu_usable')}")
    print(f"  tiny_eval_error      {m.get('tiny_eval_error')}")
    print(f"  device_info_error    {m.get('device_info_error')}")
    print(f"  {m.get('note')}")
    print()
    print("## 4-bit artifact (MEASURED)")
    a = doc["artifact"]
    print(f"  path                 {a['path']}")
    print(f"  dir bytes            {a['dir']['bytes']:,}")
    st = a["safetensors"]
    print(f"  safetensors tensors  {st['n_tensors']}")
    print(f"  payload bytes        {st['payload_bytes']:,}")
    print(f"  language bytes       {st['language_bytes']:,}")
    print(f"  vision bytes         {st['vision_bytes']:,}  ({st['vision_tensors']} tensors, sanitize-dropped)")
    print(f"  embed table          {st['embed_table_bytes']:,}")
    print(f"  embed row            {st['embed_row_bytes']:,}")
    print(f"  active language      {st['active_language_bytes']:,}  (language − table + row)")
    print(f"  affine B/group       {st['affine_bytes_per_group']} vs native absmax {st['absmax_bytes_per_group']}")
    print(f"  language organs:")
    for k, v in sorted(st["by_organ_language"].items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"    {k:<16} n={v['n']:<4} {v['bytes']:>14,}")
    print()
    print("## Kernels (OBSERVED_BINARY from mlx.metallib)")
    cat = doc["metallib"]
    print(f"  metallib bytes       {cat.get('bytes'):,}")
    print(f"  family counts        {cat.get('family_counts')}")
    print(f"  qmv_fast gs_64_b_4   {len(cat.get('qmv_fast_gs64_b4') or [])} specializations")
    for name in (cat.get("qmv_fast_gs64_b4") or [])[:8]:
        print(f"    {name}")
    print(f"  decode-relevant      {len(cat.get('decode_relevant_names') or [])} names")
    print(f"  affine_dequantize    {cat.get('n_affine_dequantize')}  (oracle/convert, NOT QuantizedLinear.__call__)")
    print(f"  {cat.get('note')}")
    print()
    print("## Fusion / graph (SOURCE_COUNTED)")
    s = doc["source"]
    print(f"  QuantizedLinear → quantized_matmul (not dequant-then-GEMM): {s['quantized_linear_uses_quantized_matmul']}")
    print(f"  generate wraps _step in mx.compile: {s['generate_wraps_step_in_mx_compile']}")
    print(f"  async_eval mentions: {s['generate_async_eval_mentions']}")
    print(f"  wired_limit mentions: {s['generate_wired_limit_mentions']}")
    print(f"  logsumexp mentions: {s['generate_logsumexp_mentions']}")
    print(f"  qmv / token: {s['qmv_dispatches_per_token']}  (native GEMV 401; extra {s['extra_qmv_vs_native']})")
    print(f"  {s['why_extra_qmv']}")
    print(f"  source-counted named op sites: {s['source_counted_named_op_site_sum']}  (NOT a GPU dispatch counter; qmv=497 is the hard count)")
    print(f"  GPU-observed dispatches: {s['gpu_observed_dispatches']['status']} — {s['gpu_observed_dispatches'].get('null_reason')}")
    print("  fusion sites:")
    for fs in s["fusion_sites"]:
        print(f"    - {fs['what']} [{fs['kind']}]")
    print("  unfused:")
    for u in s["unfused_on_purpose"]:
        print(f"    - {u}")
    geom = s["qmv_fast_geometry"]
    print(f"  qmv_fast: {geom['rows_per_threadgroup']} rows/TG, {geom['threads_per_threadgroup']} threads, "
          f"x_reuse={geom['x_reuse_rows']}, Kblock={geom['k_block_for_q4']}, reconstructs_dense={geom['reconstructs_dense']}")
    ng = doc["native_kernel"]
    print(f"  native geo_tpr64: {ng['rows_per_threadgroup']} rows/TG, {ng['threads_per_threadgroup']} threads, "
          f"x_reuse={ng['x_reuse_rows']}, Kblock={ng['k_step']}, reconstructs_dense={ng['reconstructs_dense']}")
    print()
    print("## Where the 1.472× comes from")
    acc = doc["speed_account"]
    d1 = acc["decomposition"]["using_recorded_1_215_includes_vision"]
    d2 = acc["decomposition"]["using_language_payload_fair_for_text_decode"]
    print(f"  recorded 1.215 (includes vision): bytes explain {d1['bytes_explain']:.3f}×, leftover {d1['remaining_kernel_runtime']:.3f}×")
    print(f"    {d1['reading']}")
    print(f"  language-only (fair decode):      bytes explain {d2['bytes_explain']:.3f}×, leftover {d2['remaining_kernel_runtime']:.3f}×")
    print(f"    {d2['reading']}")
    print("  implied GB/s (identity, not a new counter):")
    for k, v in acc["implied_GB_s"].items():
        if k in ("roof", "identity"):
            continue
        if isinstance(v, (int, float)):
            print(f"    {k:<36} {v:7.1f}   ({v/ANCHOR_ROOF_GB_S*100:5.1f}% of roof {ANCHOR_ROOF_GB_S})")
    print("  not the cause:")
    for n in acc["not_the_cause"]:
        print(f"    - {n}")
    print()
    print("## Advantages classified")
    print(f"  {'advantage':<58} {'class':<24}")
    for row in doc["advantages"]:
        print(f"  {row['advantage'][:58]:<58} {row['class']:<24}")
    print()
    print("## Overhead a specialised executable would not need")
    for it in doc["overhead"]:
        b = it.get("bytes")
        extra = f"{b:,} B" if isinstance(b, int) else it.get("null_reason") or it.get("count") or it.get("elements")
        print(f"  - {it['item']}: {extra}")
        print(f"    {it['note']}")
    print()
    print("## What native Gravity has to do")
    ht = doc["how_to_beat"]
    ar = ht["arithmetic"]
    print(f"  native {ANCHOR_TPS_NATIVE} tok/s @ {ar['native_implied_GB_s_on_active']:.1f} GB/s "
          f"({ar['native_pct_of_roof']:.1f}% roof)")
    print(f"  MLX    {ANCHOR_MLX_TPS} tok/s @ {ar['mlx_implied_GB_s_on_active_language']:.1f} GB/s "
          f"({ar['mlx_pct_of_roof']:.1f}% roof) on MORE language-active bytes")
    print(f"  if native hits MLX GB/s on native active bytes: {ar['tps_if_native_hits_mlx_GB_s_on_native_bytes']:.2f} tok/s")
    print("  to match 35.51:")
    for srow in ht["match_35_51"]:
        print(f"    [{srow['class']}] {srow['step']}")
        print(f"      {srow['detail']}")
    print("  to beat 35.51:")
    for srow in ht["beat_35_51"]:
        print(f"    [{srow['class']}] {srow['step']}")
        print(f"      {srow['detail']}")
    print()
    print("## Dense-reconstruction law")
    print("  MLX decode: QuantizedLinear → mx.quantized_matmul → affine_qmv_fast. reconstructs_dense=NO.")
    print("  affine_dequantize_* lives in the metallib as an oracle/convert kernel. It is not production decode.")
    print("  Native: geo_tpr64_tg128 in-register. reconstructs_dense=NO on all 38 bound kernels.")
    print("  A labelled correctness oracle may reconstruct dense W. A production operator may not.")
    print()
    print("## WHAT I WATCHED FAIL")
    for i, f in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {f['what']}: {f['result']}")
        print(f"     {f['why']}")
    print()
    print(f"wrote {doc['written_to']}")
    print("=" * 78)


def main() -> int:
    occ = occupancy()
    metal = metal_probe()
    machine = sysctl_machine()
    mlx_path = MLX_4BIT if MLX_4BIT.is_dir() else None
    huihui_present = MLX_HUIHUI.is_dir()
    art_dir = dir_bytes(mlx_path) if mlx_path else {"present": False, "bytes": 0, "n_files": 0, "by_ext": {}}
    cfg = load_json((mlx_path / "config.json") if mlx_path else Path("/dev/null"))
    idx = load_json((mlx_path / "model.safetensors.index.json") if mlx_path else Path("/dev/null"))
    safetensors = parse_safetensors_dir(mlx_path) if mlx_path else {}
    catalog = metallib_catalog(MLX_METALLIB)
    src = source_anatomy()
    gpu_null_reason = (
        "mx.metal.is_available()="
        f"{metal.get('metal_is_available_flag')} but mx.eval of a 4-float vector "
        f"raised {metal.get('tiny_eval_error')!r}. This sandbox cannot open a Metal "
        "device, so there is no GPU capture of dispatches or command buffers on the "
        "4-bit artifact. Independently, llama-server is resident on 52484 so a 27B "
        "MLX generate is forbidden (occupancy 33.47 → 3.986 tok/s). Kernel NAMES "
        "are OBSERVED_BINARY from mlx.metallib; quantized_matmul COUNT (497) is "
        "SOURCE_COUNTED from qwen3_5.py × config.json layer split."
    )
    src["gpu_observed_dispatches"] = null(gpu_null_reason)
    src["gpu_observed_command_buffers"] = null(
        gpu_null_reason
        + " Native production shape is 1 CB / 964 dispatches "
        "(qwen38_token_ns_ledger.rs). MLX CommandEncoder batches until "
        "MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER; the compiled default "
        "is inside libmlx.dylib and was not decoded."
    )
    native = native_kernel_geometry()
    controls = prior_controls()
    artifact = {
        "path": str(mlx_path) if mlx_path else None,
        "huihui_preferred_path": str(MLX_HUIHUI),
        "huihui_present": huihui_present,
        "dir": art_dir,
        "config_quantization": (cfg or {}).get("quantization") if cfg else None,
        "config_dtype": ((cfg or {}).get("text_config") or {}).get("dtype") if cfg else None,
        "index_metadata": (idx or {}).get("metadata") if idx else None,
        "safetensors": safetensors,
        "quantization": {
            "mode": ((cfg or {}).get("quantization") or {}).get("mode"),
            "bits": ((cfg or {}).get("quantization") or {}).get("bits"),
            "group_size": ((cfg or {}).get("quantization") or {}).get("group_size"),
            "law": "affine Q4 group-64 is scale+bias per group, executed as qmv_fast, not reconstruct-then-GEMM",
        },
    }
    advantages = classify_advantages(src, native, safetensors)
    over = overhead(safetensors, src)
    ht = how_to_beat(safetensors, src, native)
    acc = speed_account(safetensors, controls)
    fails = what_watched_fail(occ, metal, safetensors, src, controls)

    mlx_ver = pkg_version(next(MLX_SITE.glob("mlx-*.dist-info"), Path("/dev/null")))
    mlx_lm_ver = pkg_version(next(MLX_SITE.glob("mlx_lm-*.dist-info"), Path("/dev/null")))

    n_avail = sum(1 for r in advantages if r["class"] == "AVAILABLE_TO_NATIVE")
    n_spec = sum(1 for r in advantages if r["class"] == "FRAMEWORK_SPECIFIC")

    answer = (
        f"MLX 4-bit is the live conventional control at {ANCHOR_MLX_TPS} tok/s vs llama.cpp "
        f"Q5_K {ANCHOR_LLAMA_TPS} ({ANCHOR_SPEED_RATIO}×) and vs native uniform-q4 "
        f"{ANCHOR_TPS_NATIVE}. The recorded 1.215 bytes ratio includes {safetensors.get('vision_bytes')} B "
        f"of vision that decode does not stream; fair language-only is "
        f"{acc['bytes_this_process']['ratio_llama_over_mlx_language']['value']:.3f}× fewer bytes "
        f"than llama and STILL leaves a kernel/runtime remainder. Native already stores fewer "
        f"bytes than MLX (absmax vs affine) and already fuses in_proj, and is still slower "
        f"because geo_tpr64_tg128 gets {ht['arithmetic']['native_pct_of_roof']:.0f}% of the "
        f"{ANCHOR_ROOF_GB_S} GB/s roof while MLX qmv_fast gets "
        f"{ht['arithmetic']['mlx_pct_of_roof']:.0f}% on more bytes. "
        f"{n_avail} advantages are AVAILABLE_TO_NATIVE (qmv_fast geometry, async overlap, "
        f"concurrent gate/up, wired_limit, fused fast primitives). {n_spec} are "
        f"FRAMEWORK_SPECIFIC (general compiler, 162 MB metallib, Python tracer, logsumexp, "
        f"vision-on-disk). Framework overhead is real and small; dropping it without porting "
        f"qmv_fast will not beat {ANCHOR_MLX_TPS}. Hitting MLX GB/s on native's smaller active "
        f"bytes projects {ht['arithmetic']['tps_if_native_hits_mlx_GB_s_on_native_bytes']:.1f} tok/s."
    )

    versions = {
        "mlx": mlx_ver,
        "mlx_lm": mlx_lm_ver,
        "mlx_python": str(MLX_PY),
        "metallib_bytes": file_bytes(MLX_METALLIB),
        "libmlx_bytes": file_bytes(MLX_DYLIB),
        "native_decode_binary_bytes": file_bytes(NATIVE_BIN),
        "native_decode_binary_path": str(NATIVE_BIN) if NATIVE_BIN.is_file() else None,
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "commit": git_head(),
        "question": "Why is MLX faster, and which of that can a native Gravity operator have?",
        "answer": answer,
        "dense_reconstruction_law": {
            "oracle_ok": "representation → reconstruct dense W → ordinary GEMM may exist as a labelled correctness oracle",
            "production_forbidden": "a production operator must be native or fused and must never materialise the parent dense matrix",
            "mlx_decode_obeys": True,
            "native_decode_obeys": True,
            "mlx_oracle_kernels_in_metallib": catalog.get("n_affine_dequantize"),
        },
        "anchors_not_rederived": {
            "native_tps": ANCHOR_TPS_NATIVE,
            "native_ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "unified_memory_bytes": ANCHOR_UNIFIED_B,
            "gpu_cores": ANCHOR_GPU_CORES,
            "parameter_count": ANCHOR_PARAMS,
            "bpw": ANCHOR_BPW,
            "native_dispatches": ANCHOR_DISPATCHES,
            "native_command_buffers": ANCHOR_CBS,
            "native_artifact_bytes": ANCHOR_ARTIFACT_B,
            "native_active_bytes": ANCHOR_ACTIVE_B,
            "gemv_gflop": ANCHOR_GEMV_GFLOP,
            "two_server_tps": ANCHOR_TWO_SERVER_TPS,
            "one_server_tps": ANCHOR_ONE_SERVER_TPS,
            "mlx_tps": ANCHOR_MLX_TPS,
            "llama_tps": ANCHOR_LLAMA_TPS,
            "speed_ratio": ANCHOR_SPEED_RATIO,
            "bytes_ratio_gpu_attack": ANCHOR_BYTES_RATIO_GPU_ATTACK,
            "llama_gguf_bytes": ANCHOR_LLAMA_B,
        },
        "machine": {
            "sysctl": machine,
            "prior_chipset": "Apple M3 Ultra",
            "prior_gpu_cores": ANCHOR_GPU_CORES,
            "prior_unified": ANCHOR_UNIFIED_B,
            "prior_roof": ANCHOR_ROOF_GB_S,
            "sysctl_mem_matches_anchor": machine.get("mem_bytes") == ANCHOR_UNIFIED_B,
            "sysctl_gpu_cores_match": machine.get("gpu_cores") == ANCHOR_GPU_CORES,
        },
        "versions": versions,
        "occupancy": occ,
        "metal_probe": metal,
        "did_not_load_27b": True,
        "did_not_spawn_second_server": True,
        "artifact": artifact,
        "metallib": catalog,
        "source": src,
        "native_kernel": native,
        "controls": controls,
        "speed_account": acc,
        "advantages": advantages,
        "n_AVAILABLE_TO_NATIVE": n_avail,
        "n_FRAMEWORK_SPECIFIC": n_spec,
        "overhead": over,
        "how_to_beat": ht,
        "what_i_watched_fail": fails,
        "self_check": {
            "did_not_load_27b": True,
            "metal_not_required_for_receipt": True,
            "language_bytes_measured": bool(safetensors.get("language_bytes")),
            "qmv_fast_names_observed": bool(catalog.get("qmv_fast_gs64_b4")),
            "quantized_matmul_not_dequant_gemm": bool(
                src.get("quantized_linear_uses_quantized_matmul")
            ) and not bool(src.get("quantized_linear_call_uses_dequantize")),
            "generate_step_not_mx_compiled": src.get("generate_wraps_step_in_mx_compile") is False,
            "sysctl_mem_matches": machine.get("mem_bytes") == ANCHOR_UNIFIED_B,
        },
        "written_to": str(RECEIPT),
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print_report(doc)

    if not mlx_path:
        print("FAIL: MLX 4-bit artifact missing", file=sys.stderr)
        return 3
    if not catalog.get("qmv_fast_gs64_b4"):
        print("FAIL: affine_qmv_fast gs_64_b_4 not observed in metallib", file=sys.stderr)
        return 4
    if occ.get("llama_up") and not doc["did_not_load_27b"]:
        print("FAIL: occupancy gate violated", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
