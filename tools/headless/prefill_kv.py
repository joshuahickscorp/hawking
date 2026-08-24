#!/usr/bin/env python3
"""N016 — Prefill and KV: decode-optimal is not production-optimal.

Every prior campaign throughput number is DECODE. Prefill is measured here
separately at short / 4K / 16K / long context for the uniform-q4 incumbent
and the sealed parent. Production footprint is MODEL_BYTES +
SESSION_STATE_BYTES at c=1 and c=4.

Does not mutate ~/noetic/NOETIC_PARENT_A. Does not load a second 27B.
Does not write under receipts/ascent-2026-08-16 or workspace/campaign.

    python3 tools/headless/prefill_kv.py            # reuse receipt / assemble
    python3 tools/headless/prefill_kv.py --measure  # live Metal walks
    python3 -m pytest tools/headless/test_prefill_kv.py -q
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from first_noetic_executable import (  # noqa: E402
    PARENT_PARAMS,
    PROMPT as SHORT_USER_PROMPT,
    Q4_INCUMBENT_EBPW,
    Q4_ROOT,
    TOKENIZER,
    find_decode_binary,
)
from gpu_ledger import (  # noqa: E402
    ABSENT,
    DERIVED,
    MEASURED,
    metal_probe,
    occupancy_snapshot,
    qty,
    spread_of,
)
from noetic_information_accounting import (  # noqa: E402
    qwen38_workspace_bytes,
    self_check_workspace,
)
from noetic_parent_a import DURABLE as PARENT_ROOT  # noqa: E402

SCHEMA = "hawking.headless.prefill_kv.v1"
RECEIPT = REPO / "receipts" / "headless" / "PREFILL_KV.json"
RAW_DIR = REPO / "receipts" / "headless"
LOCK = REPO / "tools" / "gpu_lane_lock.sh"
CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)

# Length buckets. "long" is 32K — AgentOS-relevant, past the 16K mark,
# and the length G118 refused to walk because the quadratic predicted
# ~67 minutes. We walk 16K (MEASURED) and DERIVE 32K from the measured
# per-step slope. A 32K full walk is a choice, not a missing measurement
# disguised as zero.
LENGTHS = ("short", "4k", "16k", "long")
LENGTH_TOKENS = {
    "short": None,  # native compiler-prompt length, measured
    "4k": 4096,
    "16k": 16384,
    "long": 32768,
}
# Admission max_seq_len used for SESSION_STATE_BYTES. Short uses the
# campaign's 256-token workspace (NOETIC_MULTISESSION), not the prompt
# length — production reserves the slot, it does not pack to the prompt.
ADMISSION_SEQ = {
    "short": 256,
    "4k": 4096,
    "16k": 16384,
    "long": 32768,
}

# Multisession validation of the workspace formula (seq=256).
# Metal 13.50 GiB at c=1, 14.03 GiB at c=4. Delta ≈ 3 × 192,139,012.
MULTISESSION_RECEIPT = RAW_DIR / "NOETIC_MULTISESSION.json"
MULTISESSION_SEQ = 256
MULTISESSION_WS = 192_139_012

PARENT_EBPW = 3.139300850311054
PARENT_DISPATCHES = 756
Q4_DISPATCHES = 964
GIB = 1024 ** 3
METAL_BUDGET = 83_494_174_720  # recommendedMaxWorkingSetSize, measured
PEAK_GB_S = 819.0
HONEST_ROOF_GB_S = 595.9

# Chat wrap used by ascension_qwen38_hybrid_greedy unless --raw-prompt.
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

PARENT_FUSION_ENV = {
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
}

TOKENIZERS_SITE = (
    Path.home() / ".grok-vision" / "lib" / "python3.12" / "site-packages"
)

ABSENT_COUNTERS_REASON = (
    "MTLDevice.counterSets on this Apple M3 Ultra contains only the "
    "'timestamp' set with counter ['GPUTimestamp']. "
    "supportsCounterSampling.atDispatchBoundary is false. There is no "
    "DRAM bytes-moved, cache, occupancy, or per-dispatch GPU counter. "
    "Apple GPU does not expose NVIDIA-style Nsight bandwidth counters "
    "via the public Metal API."
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def dir_bytes(root: Path) -> dict[str, int]:
    n_files = 0
    total = 0
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            n_files += 1
            total += st.st_size
    return {"n_files": n_files, "bytes": total}


def parent_identity(root: Path) -> dict[str, Any]:
    cat = root / "catalog.hq38m20"
    mix = root / "MIX_REPORT.json"
    st = cat.lstat() if cat.is_file() else None
    disk = dir_bytes(root)
    payload = None
    mix_id = None
    ebpw = None
    if mix.is_file():
        doc = json.loads(mix.read_text())
        payload = doc.get("payload_bytes")
        mix_id = doc.get("mix_id")
        ebpw = doc.get("complete_ebpw")
    return {
        "path": str(root),
        "catalog": str(cat),
        "catalog_bytes": st.st_size if st else None,
        "catalog_mtime_ns": st.st_mtime_ns if st else None,
        "catalog_ino": st.st_ino if st else None,
        "disk_bytes": disk["bytes"],
        "n_files": disk["n_files"],
        "payload_bytes": payload,
        "mix_id": mix_id,
        "complete_ebpw": ebpw,
        "writable_check": oct(st.st_mode) if st else None,
    }


def identities_match(before: dict[str, Any], after: dict[str, Any]) -> bool:
    keys = ("catalog_bytes", "catalog_mtime_ns", "catalog_ino", "disk_bytes", "n_files")
    return all(before.get(k) == after.get(k) for k in keys)


def q4_model_bytes() -> dict[str, Any]:
    manifest = json.loads((Q4_ROOT / "manifest.json").read_text())
    disk = dir_bytes(Q4_ROOT)
    return {
        "path": str(Q4_ROOT),
        "name": "qwen38-gravity-uniform-q4-v1",
        "payload_bytes": int(manifest["tensor_payload_bytes"]),
        "disk_bytes": disk["bytes"],
        "n_files": disk["n_files"],
        "complete_physical_bpw": float(manifest["complete_physical_bpw"]),
        "parameter_count": int(manifest.get("source_weight_elements") or PARENT_PARAMS),
        "tensor_count": int(manifest["tensor_count"]),
    }


def parent_model_bytes(ident: dict[str, Any]) -> dict[str, Any]:
    payload = int(ident["payload_bytes"] or ident["disk_bytes"])
    return {
        "path": ident["path"],
        "name": "NOETIC_PARENT_A",
        "payload_bytes": payload,
        "disk_bytes": ident["disk_bytes"],
        "n_files": ident["n_files"],
        "complete_physical_bpw": float(ident["complete_ebpw"] or PARENT_EBPW),
        "parameter_count": PARENT_PARAMS,
        "mix_id": ident.get("mix_id"),
        "tensor_count": 755,
    }


# ---------------------------------------------------------------------------
# Workspace / production footprint
# ---------------------------------------------------------------------------

KV_BYTES_PER_POSITION = 131_072  # 16 GQA layers × 4 kv heads × 256 dim × 4 B × 2 (K+V)
DELTANET_STATE_BYTES = 156_893_184
ACTIVATION_BYTES = 1_691_396


def session_state_bytes(max_seq_len: int) -> dict[str, int]:
    """Mirror qwen38_workspace_bytes. A mismatch with the rust formula is a bug."""
    w = qwen38_workspace_bytes(max_seq_len)
    return {
        "max_seq_len": max_seq_len,
        "activation_bytes": w["activation_bytes"],
        "deltanet_state_bytes": w["deltanet_state_bytes"],
        "gqa_kv_bytes": w["gqa_kv_bytes"],
        "kv_bytes_per_position": w["kv_bytes_per_position"],
        "SESSION_STATE_BYTES": w["total_bytes"],
    }


def production_footprint(model_bytes: int, max_seq_len: int, sessions: int) -> dict[str, Any]:
    ss = session_state_bytes(max_seq_len)
    state = ss["SESSION_STATE_BYTES"] * sessions
    total = model_bytes + state
    return {
        "sessions": sessions,
        "max_seq_len": max_seq_len,
        "MODEL_BYTES": model_bytes,
        "SESSION_STATE_BYTES": ss["SESSION_STATE_BYTES"],
        "SESSION_STATE_BYTES_x_c": state,
        "PRODUCTION_FOOTPRINT_BYTES": total,
        "PRODUCTION_FOOTPRINT_GiB": total / GIB,
        "state_exceeds_weights": state > model_bytes,
        "state_share_of_footprint": state / total if total else None,
        "components": {
            "activation_bytes": ss["activation_bytes"],
            "deltanet_state_bytes": ss["deltanet_state_bytes"],
            "gqa_kv_bytes": ss["gqa_kv_bytes"],
            "kv_bytes_per_position": ss["kv_bytes_per_position"],
        },
        "metal_budget_bytes": METAL_BUDGET,
        "fits_metal_working_set": total <= METAL_BUDGET,
        "headroom_bytes": METAL_BUDGET - total,
    }


def crossover_seq(model_bytes: int, sessions: int) -> dict[str, Any]:
    """Seq length at which c × SESSION_STATE exceeds MODEL_BYTES.

    workspace = ACTIVATION + DELTANET + KV_BYTES_PER_POSITION * seq
    c * workspace > model  ⇒  seq > (model/c - ACT - DN) / KV_PER_POS
    """
    per = model_bytes / sessions
    numer = per - ACTIVATION_BYTES - DELTANET_STATE_BYTES
    seq = numer / KV_BYTES_PER_POSITION
    seq_int = int(seq) + 1 if seq > 0 else 0
    return {
        "sessions": sessions,
        "MODEL_BYTES": model_bytes,
        "seq_where_c_state_exceeds_weights": seq_int,
        "formula": (
            "seq = (MODEL_BYTES/c - activation - deltanet) / kv_bytes_per_position"
        ),
        "kv_bytes_per_position": KV_BYTES_PER_POSITION,
        "deltanet_state_bytes_constant": DELTANET_STATE_BYTES,
        "kind": DERIVED,
        "note": (
            "DeltaNet state is independent of seq_len; GQA KV is the only "
            "term that grows. At long context and c>1, state dominates "
            "weights. This is an arithmetic crossover, not a measurement "
            "of Metal currentAllocatedSize."
        ),
    }


def kv_precision_options(max_seq_len: int) -> dict[str, Any]:
    """What lowering KV precision would save — and what it would cost.

    Production GQA KV is f32 (qwen38_workspace_bytes uses size_of::<f32>(),
    encode_mha calls mha_decode_f32_tcb). f16 and int4 MHA kernels exist as
    parity tests and are not wired into Qwen38HybridDecodeSession.
    """
    f32 = KV_BYTES_PER_POSITION * max_seq_len
    return {
        "production_dtype": "f32",
        "production_gqa_kv_bytes": f32,
        "wired_in_qwen38_hybrid_decode": "mha_decode_f32_tcb",
        "candidates": {
            "f16": {
                "gqa_kv_bytes": f32 // 2,
                "saved_bytes": f32 // 2,
                "kernel_exists": True,
                "kernel_path": "crates/hawking-core/tests/mha_decode_f16kv_parity.rs",
                "wired_into_production_session": False,
                "capability_cost": {
                    "kind": ABSENT,
                    "value": None,
                    "unit": "argmax_disagreement_rate",
                    "command": (
                        "crates/hawking-core/tests/mha_decode_f16kv_parity.rs "
                        "(synthetic seq, not production Qwen3.8 greedy)"
                    ),
                    "absent_reason": (
                        "f16 KV is not the production GQA path. Parity tests "
                        "compare a synthetic MHA step against f32, not a "
                        "Qwen3.8 greedy capability suite. Lowering production "
                        "KV to f16 would need that suite; it is not free "
                        "just because a kernel exists."
                    ),
                },
            },
            "int4": {
                "gqa_kv_bytes": f32 // 8,
                "saved_bytes": f32 - f32 // 8,
                "kernel_exists": True,
                "kernel_path": "crates/hawking-core/tests/mha_decode_flash_int4kv_parity.rs",
                "wired_into_production_session": False,
                "capability_cost": {
                    "kind": ABSENT,
                    "value": None,
                    "unit": "argmax_disagreement_rate",
                    "command": (
                        "crates/hawking-core/tests/mha_decode_flash_int4kv_parity.rs"
                    ),
                    "absent_reason": (
                        "int4 KV is a flash-MHA test kernel, not production. "
                        "GQA was measured as the quality floor in "
                        "NOETIC_GQA_DESIGN / ORGAN_FRONTIERS — it cannot "
                        "cheaply go below Q4 *weights*. KV-cache quant is a "
                        "different axis and is unmeasured on this body."
                    ),
                },
            },
        },
        "deltanet_state": {
            "bytes": DELTANET_STATE_BYTES,
            "grows_with_seq": False,
            "dtype": "f32",
            "note": (
                "156,893,184 B of conv+recurrent state, constant in seq_len. "
                "At max_seq_len=256 it already dwarfs GQA KV (33,554,432 B). "
                "At 16K GQA has overtaken it. Compressing DeltaNet state is "
                "an organ-floor question (ORGAN_FRONTIERS), not a free KV tweak."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Tokenizer / prompts
# ---------------------------------------------------------------------------

_TOKENIZER = None


def load_hf_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    site = str(TOKENIZERS_SITE)
    if site not in sys.path:
        sys.path.insert(0, site)
    from tokenizers import Tokenizer  # type: ignore

    _TOKENIZER = Tokenizer.from_file(str(TOKENIZER))
    return _TOKENIZER


def encode_ids(text: str) -> list[int]:
    tok = load_hf_tokenizer()
    return list(tok.encode(text).ids)


def wrap_chat(user_text: str) -> str:
    return f"{CHAT_PREFIX}{user_text}{CHAT_SUFFIX}"


def make_raw_prompt(n_tokens: int) -> str:
    """Raw chat-formatted prompt with exactly n_tokens ids.

    Uses --raw-prompt so the binary does not wrap again. The binary's
    tokenizer is the authority on prompt_len; this aims at the bucket.
    """
    if n_tokens < 8:
        raise ValueError("n_tokens too small for chat wrap")
    unit = (
        "The front-end builds a CFG, the mid-end runs SSA passes, the "
        "back-end emits machine code. "
    )
    # Repeating a unit can BPE-merge across boundaries, so n * len(unit_ids)
    # overestimates. Grow until the wrapped encoding actually meets n_tokens.
    user = unit
    while len(encode_ids(wrap_chat(user))) < n_tokens:
        user += unit * max(1, (n_tokens - len(encode_ids(wrap_chat(user)))) // max(
            len(encode_ids(unit)), 1
        ))
        if len(user) > n_tokens * 32:
            break
    # Binary-search a char cut of the repeated body.
    lo, hi = 0, len(user)
    best = wrap_chat(user)
    while lo < hi:
        mid = (lo + hi) // 2
        cand = wrap_chat(user[:mid])
        n = len(encode_ids(cand))
        if n >= n_tokens:
            best = cand
            hi = mid
        else:
            lo = mid + 1
    user_cut = user[:hi]
    ids = encode_ids(wrap_chat(user_cut))
    # Character-level trim / pad to land on n_tokens. Bounded — the
    # BPE cut is already close.
    guard = 0
    while len(ids) > n_tokens and len(user_cut) > 16 and guard < 64:
        user_cut = user_cut[:-1]
        ids = encode_ids(wrap_chat(user_cut))
        guard += 1
    guard = 0
    while len(ids) < n_tokens and guard < 256:
        user_cut += " x"
        ids = encode_ids(wrap_chat(user_cut))
        guard += 1
    guard = 0
    while len(ids) > n_tokens and len(user_cut) > 16 and guard < 64:
        user_cut = user_cut[:-1]
        ids = encode_ids(wrap_chat(user_cut))
        guard += 1
    return wrap_chat(user_cut)


def short_user_prompt() -> str:
    return SHORT_USER_PROMPT


# ---------------------------------------------------------------------------
# Binary runs
# ---------------------------------------------------------------------------

def decode_binary() -> Path:
    env = os.environ.get("QWEN38_DECODE_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    local = (
        CARGO_TARGET / "release-fast" / "examples" / "ascension_qwen38_hybrid_greedy"
    )
    if local.is_file():
        return local
    return find_decode_binary()


def refuse_second_27b() -> dict[str, Any]:
    """Do not load a second 27B. Wait for a foreign body to leave."""
    deadline = time.time() + float(os.environ.get("PREFILL_KV_GPU_WAIT_S", "5400"))
    snap = occupancy_snapshot()
    while snap.get("loaded_a_second_27b"):
        remain = deadline - time.time()
        if remain <= 0:
            raise SystemExit(
                "REFUSING: a 27B-class runtime is still resident after wait. "
                f"ps={snap.get('ps_matches')}"
            )
        big = [
            m
            for m in (snap.get("ps_matches") or [])
            if any(tok.isdigit() and int(tok) > 4_000_000 for tok in m.split()[:3])
        ]
        print(
            f"waiting for foreign 27B to leave ({remain:.0f}s left): "
            f"{(big or snap.get('ps_matches') or [])[:1]}",
            flush=True,
        )
        time.sleep(15)
        snap = occupancy_snapshot()
    return snap


def run_generate(
    *,
    artifact: Path,
    prompt: str,
    raw_prompt: bool,
    max_new_tokens: int,
    max_seq_len: int,
    out: Path,
    fusion: bool,
    complete_wall: bool,
    pairs: int,
    timeout_s: int,
) -> dict[str, Any]:
    refuse_second_27b()
    binary = decode_binary()
    cmd: list[str] = []
    if LOCK.is_file():
        cmd.extend(["bash", str(LOCK), "n016-prefill-kv"])
    cmd.extend(
        [
            str(binary),
            "--artifact-root",
            str(artifact),
            "--tokenizer",
            str(TOKENIZER),
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-seq-len",
            str(max_seq_len),
            "--out",
            str(out),
        ]
    )
    if raw_prompt:
        cmd.append("--raw-prompt")
    if complete_wall:
        cmd.extend(["--complete-wall", "--pairs", str(pairs)])
    env = os.environ.copy()
    for k in (
        "HAWKING_QWEN38_FUSE_MLP",
        "HAWKING_QWEN38_FUSE_GQA_QKV",
        "HAWKING_QWEN38_FUSE_DN_INPROJ",
    ):
        env.pop(k, None)
    if fusion:
        env.update(PARENT_FUSION_ENV)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    wall_s = time.perf_counter() - t0
    raw = None
    if out.is_file():
        try:
            raw = json.loads(out.read_text())
        except json.JSONDecodeError:
            raw = None
    return {
        "ok": proc.returncode == 0 and raw is not None,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "command": [c if c != prompt else f"<prompt {len(prompt)} chars>" for c in cmd],
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-8000:],
        "out": str(out),
        "raw": raw,
        "fusion": fusion,
        "complete_wall": complete_wall,
        "max_seq_len": max_seq_len,
        "max_new_tokens": max_new_tokens,
        "binary": str(binary),
    }


def summarize_default_raw(raw: dict[str, Any], prompt_len_expected: int | None) -> dict[str, Any]:
    """Prefill is the first prompt_len steps of generate_greedy."""
    prompt_len = int(raw.get("prompt_len") or 0)
    if prompt_len <= 0:
        prompt_len = len(raw.get("prompt_ids") or [])
    gpu = raw.get("gpu_ns_per_step") or []
    wait = raw.get("wait_ns_per_step") or []
    wall_steps = raw.get("wall_ns_per_step") or []
    disp = raw.get("dispatches_per_step") or []
    n_prefill = prompt_len if prompt_len > 0 else len(gpu)
    gpu_prefill = [g for g in gpu[:n_prefill] if g is not None]
    wall_prefill = wall_steps[:n_prefill] if wall_steps else []
    first_gpu = gpu_prefill[0] if gpu_prefill else None
    rest_gpu = gpu_prefill[1:] if len(gpu_prefill) > 1 else []
    last64 = gpu_prefill[-64:] if gpu_prefill else []
    prefill_wall_ns = raw.get("prefill_wall_ns")
    return {
        "prompt_len": prompt_len,
        "prompt_len_expected": prompt_len_expected,
        "prefill_wall_ns": prefill_wall_ns,
        "decode_wall_ns": raw.get("decode_wall_ns"),
        "first_step_wall_ns": raw.get("first_step_wall_ns"),
        "first_step_gpu_ns": first_gpu,
        "n_prefill_gpu": len(gpu_prefill),
        "prefill_gpu_ns_sum": int(sum(gpu_prefill)) if gpu_prefill else None,
        "prefill_gpu_ns_mean": (
            float(sum(gpu_prefill)) / len(gpu_prefill) if gpu_prefill else None
        ),
        "prefill_gpu_ns_last64_median": (
            spread_of(last64)["median"] if last64 else None
        ),
        "rest_gpu": spread_of(rest_gpu) if rest_gpu else None,
        "all_prefill_gpu": spread_of(gpu_prefill) if gpu_prefill else None,
        "wall_steps_prefill": spread_of(wall_prefill) if wall_prefill else None,
        "dispatches": disp[1] if len(disp) > 1 else (disp[0] if disp else None),
        "fallbacks": raw.get("fallbacks"),
        "new_token_ids": raw.get("new_token_ids") or raw.get("generated_token_ids"),
        "generated_text": raw.get("generated_text"),
        "median_gpu_ns_per_token": raw.get("median_gpu_ns_per_token"),
        "gpu_ns_per_step_head": gpu_prefill[:8],
        "gpu_ns_per_step_tail": gpu_prefill[-8:] if gpu_prefill else [],
        "gpu_ns_every_256": gpu_prefill[255::256],
        "wait_ns_head": wait[:4],
    }


def summarize_complete_wall(raw: dict[str, Any]) -> dict[str, Any]:
    cold = raw.get("cold_generate") or {}
    warm_reps = raw.get("warm_reps") or []
    control = raw.get("control_uninstrumented_generate_greedy") or {}

    def pull(summary: dict[str, Any], tag: str) -> dict[str, Any]:
        return {
            "tag": tag,
            "prompt_len": summary.get("prompt_len"),
            "n_prefill_steps": summary.get("n_prefill_steps"),
            "prefill_wall_ns": summary.get("prefill_wall_ns"),
            "n_steady_decode_steps": summary.get("n_steady_decode_steps"),
            "cold_or_first_step": summary.get("cold_or_first_step"),
            "steady_decode_gpu_ns": (summary.get("steady_decode") or {}).get("gpu_ns"),
            "steady_decode_complete_wall_ns": (summary.get("steady_decode") or {}).get(
                "complete_wall_ns"
            ),
            "fallbacks": summary.get("fallbacks"),
            "new_token_ids": summary.get("new_token_ids"),
            "generated_text": summary.get("generated_text"),
            "dispatches": (summary.get("steady_decode") or {}).get("dispatches"),
        }

    warms = []
    for rep in warm_reps:
        s = rep.get("summary") or {}
        warms.append(pull(s, str(rep.get("label") or "warm")))
    return {
        "prompt_len": (raw.get("identity") or {}).get("prompt_len"),
        "cold": pull(cold, "cold"),
        "warm_reps": warms,
        "control": {
            "prefill_wall_ns": control.get("prefill_wall_ns"),
            "decode_wall_ns": control.get("decode_wall_ns"),
            "decode_steps": control.get("decode_steps"),
            "first_step_wall_ns": control.get("first_step_wall_ns"),
            "new_token_ids": control.get("new_token_ids"),
        },
        "authority_decode": raw.get("authority"),
        "definition": raw.get("definition"),
    }


def fit_prefill_slope(gpu_ns_per_step: list[int | float]) -> dict[str, Any]:
    """gpu_ns(i) ≈ a + b * i, dropping the graph-cold first step.

    Total prefill GPU for N tokens ≈ N*a + b*N*(N-1)/2.
    Equivalent to G118's per-token = base + (N/2)*slope when reporting
    the mean token of a walk of length N.
    """
    xs = []
    ys = []
    for i, g in enumerate(gpu_ns_per_step):
        if i == 0:
            continue
        if g is None:
            continue
        xs.append(float(i))
        ys.append(float(g))
    n = len(xs)
    if n < 8:
        return {"ok": False, "n": n, "reason": "need >=8 warm prefill steps"}
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return {"ok": False, "n": n, "reason": "zero x variance"}
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = cov / var_x
    a = mean_y - b * mean_x
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "ok": True,
        "n": n,
        "intercept_ns": a,
        "slope_ns_per_position": b,
        "slope_us_per_position": b / 1e3,
        "r2": r2,
        "mean_gpu_ns": mean_y,
        "first_warm_index": 1,
        "last_index": int(xs[-1]),
        "formula": "gpu_ns(i) = intercept + slope * i  (i is 0-based position)",
        "g118_equivalent_base_ms": a / 1e6,
        "g118_equivalent_slope_us": b / 1e3,
    }


def predicted_prefill_gpu_ns(fit: dict[str, Any], n_tokens: int) -> float | None:
    if not fit.get("ok"):
        return None
    a = float(fit["intercept_ns"])
    b = float(fit["slope_ns_per_position"])
    # Sum_{i=0}^{N-1} (a + b i) = N a + b N (N-1)/2
    return n_tokens * a + b * n_tokens * (n_tokens - 1) / 2.0


# ---------------------------------------------------------------------------
# Assemble labelled quantities
# ---------------------------------------------------------------------------

def labelled_prefill_from_runs(
    runs: list[dict[str, Any]],
    *,
    length_name: str,
    model_name: str,
) -> dict[str, Any]:
    """Build MEASURED/DERIVED/ABSENT prefill block from one or more process runs."""
    cmd = ""
    if runs:
        cmd = " ".join(str(c) for c in (runs[0].get("command") or [])[:8]) + " ..."

    cold_walls = []
    warm_walls = []
    cold_gpu_first = []
    warm_gpu_mean = []
    prompt_lens = []
    dispatches = []
    fallbacks = []
    texts = []
    fits = []
    last64 = []

    for run in runs:
        raw = run.get("raw") or {}
        if run.get("complete_wall"):
            s = summarize_complete_wall(raw)
            prompt_lens.append(s.get("prompt_len") or (s.get("cold") or {}).get("prompt_len"))
            cold = s.get("cold") or {}
            if cold.get("prefill_wall_ns"):
                cold_walls.append(cold["prefill_wall_ns"])
            first = (cold.get("cold_or_first_step") or {}).get("gpu_ns")
            if first:
                cold_gpu_first.append(first)
            for w in s.get("warm_reps") or []:
                if w.get("prefill_wall_ns"):
                    warm_walls.append(w["prefill_wall_ns"])
            if cold.get("dispatches"):
                dispatches.append(cold["dispatches"])
            fallbacks.append(cold.get("fallbacks"))
            texts.append(cold.get("generated_text"))
            if s.get("control") and s["control"].get("new_token_ids"):
                texts.append(str(s["control"]["new_token_ids"]))
        else:
            s = summarize_default_raw(raw, LENGTH_TOKENS.get(length_name))
            prompt_lens.append(s.get("prompt_len"))
            if s.get("prefill_wall_ns"):
                # First process is colder (graph + possible page cache).
                # Subsequent processes are page-warm; still graph-cold on
                # step 0 of a fresh process.
                if not cold_walls:
                    cold_walls.append(s["prefill_wall_ns"])
                else:
                    warm_walls.append(s["prefill_wall_ns"])
            if s.get("first_step_gpu_ns"):
                cold_gpu_first.append(s["first_step_gpu_ns"])
            rest = s.get("rest_gpu") or {}
            if rest.get("median") is not None:
                # Median of steps 1..N-1. Mean is outlier-pulled at 16K.
                warm_gpu_mean.append(rest["median"])
            elif s.get("prefill_gpu_ns_mean"):
                warm_gpu_mean.append(s["prefill_gpu_ns_mean"])
            if s.get("prefill_gpu_ns_last64_median"):
                last64.append(s["prefill_gpu_ns_last64_median"])
            if s.get("dispatches"):
                dispatches.append(s["dispatches"])
            fallbacks.append(s.get("fallbacks"))
            texts.append(s.get("generated_text"))
            gpu_steps = [
                g
                for g in (raw.get("gpu_ns_per_step") or [])[: int(s.get("prompt_len") or 0)]
                if g is not None
            ]
            fit = fit_prefill_slope(gpu_steps)
            if fit.get("ok"):
                fits.append(fit)

    prompt_len = next((p for p in prompt_lens if p), None)
    dispatch = next((d for d in dispatches if d), None)

    def wall_qty(values: list, tag: str) -> dict[str, Any]:
        if not values:
            return qty(
                None,
                kind=ABSENT,
                unit="ns",
                command=cmd or "ascension_qwen38_hybrid_greedy --out RAW",
                absent_reason=(
                    f"No {tag} prefill wall was captured for {model_name}/{length_name}. "
                    "A single Metal run is page-cache confounded; this slot is empty "
                    "rather than filled with a proxy."
                ),
            )
        sp = spread_of(values)
        return qty(
            sp["median"],
            kind=MEASURED,
            unit="ns",
            command=cmd or "ascension_qwen38_hybrid_greedy",
            note=(
                f"{tag} prefill_wall_ns over {sp['n']} process-or-rep samples. "
                "MTLCommandBuffer GPUEnd−GPUStart lives in gpu_ns_per_step; "
                "this wall includes host encode/submit/wait."
            ),
            spread=sp,
        )

    cold_wall = wall_qty(cold_walls, "cold")
    warm_wall = wall_qty(warm_walls, "warm")
    # If we only have one walk, treat it as cold (fresh process) and leave
    # in-process warm ABSENT rather than duplicating the same number.
    if warm_wall["kind"] == ABSENT and cold_wall["kind"] == MEASURED:
        warm_wall = qty(
            None,
            kind=ABSENT,
            unit="ns",
            command=cmd or "ascension_qwen38_hybrid_greedy",
            absent_reason=(
                f"Only one fresh-process {length_name} walk for {model_name}. "
                "In-process warm re-prefill of this length was not repeated "
                "(a second 16K/long walk is another full teacher-forced GEMV "
                "pass). Cold here is graph-cold on step 0 plus page-cache "
                "state of that process; OS_PAGE_CACHE_COLD is separately ABSENT "
                "because this box cannot purge the unified-memory page cache "
                "without disrupting the machine."
            ),
        )

    per_token_warm = None
    if warm_gpu_mean:
        per_token_warm = qty(
            spread_of(warm_gpu_mean)["median"],
            kind=MEASURED,
            unit="ns/token",
            command=cmd,
            note="mean GPU ns over prefill steps 1..N-1 (step 0 excluded as graph-cold)",
            spread=spread_of(warm_gpu_mean),
        )
    elif last64:
        per_token_warm = qty(
            spread_of(last64)["median"],
            kind=MEASURED,
            unit="ns/token",
            command=cmd,
            note="median GPU ns of the last 64 prefill steps of the walk (warm, long prefix)",
            spread=spread_of(last64),
        )
    else:
        if warm_walls and prompt_len:
            per_token_warm = qty(
                spread_of(warm_walls)["median"] / float(prompt_len),
                kind=DERIVED,
                unit="ns/token",
                command=cmd or "ascension_qwen38_hybrid_greedy",
                note=(
                    "warm prefill_wall_ns / prompt_len. Wall, not GPU. "
                    "Used when the raw does not dump gpu_ns_per_step "
                    "(complete-wall summaries)."
                ),
                spread=None,
            )
        else:
            per_token_warm = qty(
                None,
                kind=ABSENT,
                unit="ns/token",
                command=cmd or "ascension_qwen38_hybrid_greedy",
                absent_reason=(
                    "No per-step GPU timestamps were captured for this length "
                    "(complete-wall summaries do not dump the prefill gpu_ns "
                    "vector; default generate does) and no warm wall exists "
                    "to derive a per-token figure."
                ),
            )

    first_gpu = (
        qty(
            spread_of(cold_gpu_first)["median"],
            kind=MEASURED,
            unit="ns",
            command=cmd,
            note="graph-cold first prefill step GPUEnd−GPUStart; never averaged into warm",
            spread=spread_of(cold_gpu_first),
        )
        if cold_gpu_first
        else qty(
            None,
            kind=ABSENT,
            unit="ns",
            command=cmd or "ascension_qwen38_hybrid_greedy",
            absent_reason="first-step gpu_ns missing from raw",
        )
    )

    fit = fits[-1] if fits else {"ok": False}
    return {
        "length": length_name,
        "model": model_name,
        "prompt_len": prompt_len,
        "admission_max_seq_len": ADMISSION_SEQ[length_name],
        "n_process_runs": len(runs),
        "dispatches_per_step": dispatch,
        "fallbacks": fallbacks,
        "generated_text_head": (texts[0][:160] if texts and isinstance(texts[0], str) else texts[:1]),
        "cold_prefill_wall_ns": cold_wall,
        "warm_prefill_wall_ns": warm_wall,
        "cold_first_step_gpu_ns": first_gpu,
        "warm_prefill_gpu_ns_per_token": per_token_warm,
        "quadratic_fit": fit,
        "runs_ok": [bool(r.get("ok")) for r in runs],
        "raw_paths": [r.get("out") for r in runs],
    }


def derive_long_from_fit(
    fit: dict[str, Any],
    n_tokens: int,
    model_name: str,
    source_length: str,
) -> dict[str, Any]:
    pred = predicted_prefill_gpu_ns(fit, n_tokens)
    cmd = (
        f"sum_{{i=0..{n_tokens-1}}} (intercept + slope*i) from MEASURED "
        f"{source_length} gpu_ns_per_step on {model_name}"
    )
    if pred is None:
        return {
            "length": "long",
            "model": model_name,
            "prompt_len": n_tokens,
            "admission_max_seq_len": ADMISSION_SEQ["long"],
            "n_process_runs": 0,
            "cold_prefill_wall_ns": qty(
                None,
                kind=ABSENT,
                unit="ns",
                command=cmd,
                absent_reason=(
                    "No quadratic fit; 32K full walk was not run and no "
                    "shorter walk produced a slope."
                ),
            ),
            "warm_prefill_wall_ns": qty(
                None,
                kind=ABSENT,
                unit="ns",
                command=cmd,
                absent_reason="same as cold: no fit, no 32K walk",
            ),
            "warm_prefill_gpu_ns_per_token": qty(
                None,
                kind=ABSENT,
                unit="ns/token",
                command=cmd,
                absent_reason="no per-step GPU series to evaluate at i=32767",
            ),
            "quadratic_fit": fit,
            "derived_from": source_length,
            "why_not_walked": (
                "G118's quadratic on this architecture predicted ~67 minutes "
                "for a single 32K teacher-forced GEMV walk. We walk 16K "
                "(MEASURED) and DERIVE 32K from that walk's per-step slope."
            ),
        }
    mean_gpu = pred / n_tokens
    # Last-token cost at position n-1.
    last = float(fit["intercept_ns"]) + float(fit["slope_ns_per_position"]) * (n_tokens - 1)
    return {
        "length": "long",
        "model": model_name,
        "prompt_len": n_tokens,
        "admission_max_seq_len": ADMISSION_SEQ["long"],
        "n_process_runs": 0,
        "dispatches_per_step": None,
        "cold_prefill_wall_ns": qty(
            None,
            kind=ABSENT,
            unit="ns",
            command=cmd,
            absent_reason=(
                "32K was not walked. A single Metal run at this length is "
                "also page-cache confounded; we refuse to invent a cold wall. "
                "The DERIVED GPU sum below is the warm-slope integral, not a cold first-touch."
            ),
        ),
        "warm_prefill_wall_ns": qty(
            None,
            kind=ABSENT,
            unit="ns",
            command=cmd,
            absent_reason=(
                "32K in-process warm re-prefill was not run. GPU integral is "
                "DERIVED from the higher-r2 measured slope, not a second walk."
            ),
        ),
        "warm_prefill_gpu_ns_total": qty(
            pred,
            kind=DERIVED,
            unit="ns",
            command=cmd,
            note=(
                f"Integral of MEASURED {source_length} gpu_ns(i)=a+b*i over "
                f"i=0..{n_tokens-1}. Host wall is not in the integral; "
                "GPU is the authority (GPUEnd−GPUStart)."
            ),
        ),
        "warm_prefill_gpu_ns_per_token": qty(
            mean_gpu,
            kind=DERIVED,
            unit="ns/token",
            command=cmd,
            note="mean GPU ns/token of the 32K integral (mean context = 16383.5)",
        ),
        "warm_prefill_gpu_ns_at_last_position": qty(
            last,
            kind=DERIVED,
            unit="ns/token",
            command=cmd,
            note="a + b*(32767): the prefill *step* cost at long context, not the mean",
        ),
        "quadratic_fit": fit,
        "derived_from": source_length,
        "why_not_walked": (
            "A 32K teacher-forced GEMV walk is ~2× the 16K walk we already "
            "ran. The slope is identified by the 16K series (thousands of "
            "points). Repeating the 67-minute G118 choice would not change "
            "the topology conclusion: this path has no batched prefill."
        ),
    }


# ---------------------------------------------------------------------------
# Live measurement
# ---------------------------------------------------------------------------

def raw_path(model: str, length: str, run: int, wall: bool) -> Path:
    tag = "cw" if wall else "gen"
    return RAW_DIR / f"_PREFILL_KV_{model}_{length}_{tag}_run{run}.json"


def measure_model(
    *,
    model_key: str,
    artifact: Path,
    fusion: bool,
    lengths: tuple[str, ...],
    parent_before: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"model": model_key, "artifact": str(artifact), "lengths": {}}
    force = os.environ.get("PREFILL_KV_FORCE", "0") == "1"
    existing = load_existing_raws(model_key)
    def _skip(length: str) -> bool:
        have = existing.get(length) or []
        if have and not force:
            print(
                f"== {model_key} {length} skip ({len(have)} raws on disk) ==",
                flush=True,
            )
            return True
        return False

    # SHORT: complete-wall so cold and warm prefills are in one process,
    # with spread across process runs. One default generate dumps the
    # prefill gpu_ns vector that complete-wall summaries omit.
    if "short" in lengths:
        have = existing.get("short") or []
        have_cw = [p for p in have if "_cw_" in p.name]
        have_gen = [p for p in have if "_gen_" in p.name]
        if have_cw and have_gen and not force:
            print(
                f"== {model_key} short skip ({len(have_cw)} cw + {len(have_gen)} gen) ==",
                flush=True,
            )
        else:
            if not have_cw or force:
                reps = int(os.environ.get("PREFILL_KV_SHORT_REPS", "3"))
                for i in range(1, reps + 1):
                    print(f"== {model_key} short complete-wall run {i}/{reps} ==", flush=True)
                    run = run_generate(
                        artifact=artifact,
                        prompt=short_user_prompt(),
                        raw_prompt=False,
                        max_new_tokens=int(os.environ.get("PREFILL_KV_SHORT_NEW", "8")),
                        max_seq_len=ADMISSION_SEQ["short"],
                        out=raw_path(model_key, "short", i, True),
                        fusion=fusion,
                        complete_wall=True,
                        pairs=1,
                        timeout_s=600,
                    )
                    if not run["ok"]:
                        print(run["stderr_tail"][-1500:], flush=True)
            if not have_gen or force:
                print(f"== {model_key} short generate (gpu_ns_per_step) ==", flush=True)
                run = run_generate(
                    artifact=artifact,
                    prompt=short_user_prompt(),
                    raw_prompt=False,
                    max_new_tokens=1,
                    max_seq_len=ADMISSION_SEQ["short"],
                    out=raw_path(model_key, "short", 0, False),
                    fusion=fusion,
                    complete_wall=False,
                    pairs=1,
                    timeout_s=600,
                )
                if not run["ok"]:
                    print(run["stderr_tail"][-1500:], flush=True)

    if "4k" in lengths and not _skip("4k"):
        reps = int(os.environ.get("PREFILL_KV_4K_REPS", "2"))
        prompt = make_raw_prompt(LENGTH_TOKENS["4k"])
        print(
            f"== {model_key} 4k prompt_ids={len(encode_ids(prompt))} ==",
            flush=True,
        )
        runs = []
        for i in range(1, reps + 1):
            print(f"== {model_key} 4k generate run {i}/{reps} ==", flush=True)
            runs.append(
                run_generate(
                    artifact=artifact,
                    prompt=prompt,
                    raw_prompt=True,
                    max_new_tokens=1,
                    max_seq_len=ADMISSION_SEQ["4k"] + 16,
                    out=raw_path(model_key, "4k", i, False),
                    fusion=fusion,
                    complete_wall=False,
                    pairs=1,
                    timeout_s=1200,
                )
            )
            if not runs[-1]["ok"]:
                print(runs[-1]["stderr_tail"][-1500:], flush=True)
        out["lengths"]["4k"] = labelled_prefill_from_runs(
            runs, length_name="4k", model_name=model_key
        )

    if "16k" in lengths and not _skip("16k"):
        reps = int(os.environ.get("PREFILL_KV_16K_REPS", "1"))
        prompt = make_raw_prompt(LENGTH_TOKENS["16k"])
        print(
            f"== {model_key} 16k prompt_ids={len(encode_ids(prompt))} ==",
            flush=True,
        )
        runs = []
        for i in range(1, reps + 1):
            print(f"== {model_key} 16k generate run {i}/{reps} ==", flush=True)
            runs.append(
                run_generate(
                    artifact=artifact,
                    prompt=prompt,
                    raw_prompt=True,
                    max_new_tokens=1,
                    max_seq_len=ADMISSION_SEQ["16k"] + 16,
                    out=raw_path(model_key, "16k", i, False),
                    fusion=fusion,
                    complete_wall=False,
                    pairs=1,
                    timeout_s=7200,
                )
            )
            if not runs[-1]["ok"]:
                print(runs[-1]["stderr_tail"][-1500:], flush=True)
        out["lengths"]["16k"] = labelled_prefill_from_runs(
            runs, length_name="16k", model_name=model_key
        )

    # long is derived after we have a fit from 16k (preferred) or 4k.
    source = None
    fit = {"ok": False}
    for cand in ("16k", "4k"):
        block = out["lengths"].get(cand) or {}
        f = block.get("quadratic_fit") or {}
        if f.get("ok"):
            source = cand
            fit = f
            break
    if "long" in lengths:
        out["lengths"]["long"] = derive_long_from_fit(
            fit, LENGTH_TOKENS["long"], model_key, source or "none"
        )

    if parent_before is not None:
        after = parent_identity(PARENT_ROOT)
        out["parent_identity_after"] = after
        out["did_not_mutate_sealed_parent"] = identities_match(parent_before, after)
    return out


def load_existing_raws(model_key: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {k: [] for k in LENGTHS}
    for p in sorted(RAW_DIR.glob(f"_PREFILL_KV_{model_key}_*")):
        name = p.name
        for length in ("short", "4k", "16k", "long"):
            if f"_{model_key}_{length}_" in name:
                found[length].append(p)
    return found


def assemble_from_raws(model_key: str, fusion: bool) -> dict[str, Any]:
    found = load_existing_raws(model_key)
    out: dict[str, Any] = {"model": model_key, "lengths": {}, "from_disk_raws": True}
    for length in ("short", "4k", "16k"):
        paths = found.get(length) or []
        if not paths:
            cmd = "ascension_qwen38_hybrid_greedy --out RAW"
            out["lengths"][length] = {
                "length": length,
                "model": model_key,
                "prompt_len": LENGTH_TOKENS[length],
                "admission_max_seq_len": ADMISSION_SEQ[length],
                "n_process_runs": 0,
                "cold_prefill_wall_ns": qty(
                    None, kind=ABSENT, unit="ns", command=cmd,
                    absent_reason=f"no RAW on disk yet for {model_key}/{length}",
                ),
                "warm_prefill_wall_ns": qty(
                    None, kind=ABSENT, unit="ns", command=cmd,
                    absent_reason=f"no RAW on disk yet for {model_key}/{length}",
                ),
                "warm_prefill_gpu_ns_per_token": qty(
                    None, kind=ABSENT, unit="ns/token", command=cmd,
                    absent_reason=f"no RAW on disk yet for {model_key}/{length}",
                ),
                "cold_first_step_gpu_ns": qty(
                    None, kind=ABSENT, unit="ns", command=cmd,
                    absent_reason=f"no RAW on disk yet for {model_key}/{length}",
                ),
                "quadratic_fit": {"ok": False},
            }
            continue
        runs = []
        for p in paths:
            try:
                raw = json.loads(p.read_text())
            except Exception:
                continue
            complete = "complete_token_wall" in (raw.get("schema") or "") or "cold_generate" in raw
            runs.append(
                {
                    "ok": True,
                    "raw": raw,
                    "out": str(p),
                    "complete_wall": complete,
                    "command": [str(p)],
                    "fusion": fusion,
                }
            )
        if runs:
            out["lengths"][length] = labelled_prefill_from_runs(
                runs, length_name=length, model_name=model_key
            )
            out["lengths"][length]["raw_paths"] = [str(p) for p in paths]
    source = None
    fit = {"ok": False}
    # Prefer the higher-r2 fit. The 16K OLS is outlier-pulled
    # (q4 16K r2=0.85, intercept collapses); 4K r2=0.99 is the
    # clean GEMV+linear-GQA model. Last-token at 16K still matches 4K slope.
    for cand in ("4k", "16k"):
        block = out["lengths"].get(cand) or {}
        f = block.get("quadratic_fit") or {}
        if not f.get("ok"):
            continue
        if not fit.get("ok") or float(f.get("r2") or 0) > float(fit.get("r2") or 0):
            source = cand
            fit = f
    out["lengths"]["long"] = derive_long_from_fit(
        fit, LENGTH_TOKENS["long"], model_key, source or "none"
    )
    return out


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def topology_verdict(q4_block: dict, parent_block: dict) -> dict[str, Any]:
    return {
        "production_prefill_path": (
            "Qwen38HybridDecodeSession.generate_greedy teacher-forced "
            "session.step per prompt token. Same 1-CB GEMV graph as decode "
            "(964 dispatches q4 / 756 fused parent). There is no batched "
            "GEMM prefill on this body."
        ),
        "decode_optimal": True,
        "production_optimal": False,
        "separate_prefill_path_allowed": True,
        "separate_path_would_be": (
            "A prompt-parallel GEMM (or flash-prefill) that amortizes the "
            "weight stream across tokens of one prompt, plus a GQA prefill "
            "kernel over the growing prefix. Kernels for batched prefill "
            "exist on the generic engine "
            "(crates/hawking-core/tests/p3_batched_prefill_parity.rs) and "
            "are not wired into Qwen38HybridDecodeSession. Building that "
            "path is explicitly in scope; this receipt does not ship it."
        ),
        "quadratic_mechanism": (
            "encode_mha passes seq=position into mha_decode_f32_tcb. Each "
            "prefill step attends over the prefix built so far, so cost "
            "grows as O(seq) per step and O(N²) for a walk of N. GEMV "
            "weight traffic is ~constant per step and dominates at short "
            "context; GQA is the term that makes long prefill worse than "
            "N × decode."
        ),
        "g117_g118_cited_not_copied": {
            "g117": "HEAD:receipts/ascent-2026-08-16/G117_PREFILL_BASELINE.json",
            "g118": "HEAD:receipts/ascent-2026-08-16/G118_PREFILL_SCALING.json",
            "g117_claimed": "per-token flat near 32.5 ms (11..688 tokens)",
            "g118_corrected": (
                "quadratic; 2641-token point made the slope visible; "
                "32K walk refused (~67 min predicted)"
            ),
            "this_receipt": (
                "Re-measured on the q4 incumbent AND the sealed parent. "
                "G118 numbers are not copied into MEASURED slots."
            ),
        },
    }


def capability_qualification(q4: dict, parent: dict) -> dict[str, Any]:
    def short_text(block: dict) -> str | None:
        b = (block.get("lengths") or {}).get("short") or {}
        t = b.get("generated_text_head")
        if isinstance(t, list) and t:
            return str(t[0])
        return str(t) if t else None

    return {
        "do_not_assume_kv_quant_is_free": True,
        "kv_precision_changed_in_this_measurement": False,
        "short_context_greedy": {
            "q4": short_text(q4),
            "parent": short_text(parent),
            "note": (
                "Short complete-wall generates 8 new tokens (or fewer on EOS). "
                "This is a coherence canary, not the full capability suite. "
                "Parent seal already recorded a 16-token compiler-prompt sample."
            ),
        },
        "long_prefill_capability": {
            "kind": "QUALIFIED",
            "note": (
                "4K/16K walks use max_new_tokens=1. A single greedy id after "
                "a long prefix is not a capability score. Fallbacks must stay "
                "0 (dense_w_materialized stays 0). No KV dtype was changed, "
                "so these walks cannot be read as 'f16 KV is free'."
            ),
        },
        "what_would_qualify_a_kv_compromise": (
            "Wire f16 (or int4) KV into Qwen38HybridDecodeSession, rerun the "
            "short greedy compiler-prompt sample AND a held-out capability "
            "suite, report argmax disagreement / gain / fallbacks against "
            "the f32 KV parent. Until that exists the cost is ABSENT."
        ),
    }


def footprint_table(model_bytes: int, name: str) -> dict[str, Any]:
    by_len = {}
    for length, seq in ADMISSION_SEQ.items():
        by_len[length] = {
            "c=1": production_footprint(model_bytes, seq, 1),
            "c=4": production_footprint(model_bytes, seq, 4),
        }
    return {
        "model": name,
        "MODEL_BYTES": {
            "value": model_bytes,
            "kind": MEASURED,
            "unit": "bytes",
            "command": (
                "stat of artifact tree (payload_bytes from manifest/MIX_REPORT)"
            ),
            "absent_reason": None,
        },
        "by_length": by_len,
        "crossover": {
            "c=1": crossover_seq(model_bytes, 1),
            "c=4": crossover_seq(model_bytes, 4),
        },
        "native_128k_c4": production_footprint(model_bytes, 131072, 4),
        "formula": (
            "PRODUCTION_FOOTPRINT = MODEL_BYTES + c × SESSION_STATE_BYTES(max_seq_len). "
            "SESSION_STATE = activation + DeltaNet(conv+rec) + GQA KV. "
            "DeltaNet is constant in seq; GQA KV = 131072 B × seq. "
            "The runtime allocates the full max_seq_len KV; it does not pack "
            "to the live prompt length."
        ),
        "session_state_kind": DERIVED,
        "session_state_kind_reason": (
            "qwen38_workspace_bytes in crates/hawking-core/src/model/"
            "qwen38_hybrid_decode.rs, mirrored by "
            "tools/headless/noetic_information_accounting.py. Validated "
            "MEASURED at seq=256 by NOETIC_MULTISESSION (Metal 13.50 GiB "
            "c=1 vs 14.03 GiB c=4; delta matches 3 × 192,139,012 B). "
            "Other seqs are the same formula, hence DERIVED not guessed."
        ),
    }


def prefix_sharing_note() -> dict[str, Any]:
    return {
        "valid_when": (
            "N sessions share a byte-identical prefix (system prompt / "
            "tool preamble). Then one GQA KV prefix plus N suffixes is "
            "correct; DeltaNet rec state is NOT prefix-shareable across "
            "diverged suffixes because it is a recurrent summary, not a cache."
        ),
        "wired_into_qwen38_hybrid": False,
        "related_code": "crates/hawking-serve/src/system_kv_bank.rs",
        "kind": ABSENT,
        "value": None,
        "unit": "bytes_saved_at_c4",
        "command": "crates/hawking-serve/src/system_kv_bank.rs",
        "absent_reason": (
            "system_kv_bank is a hawking-serve construct, not attached to "
            "Qwen38HybridDecodeSession. No prefix-sharing bytes were "
            "measured on this body. Claiming the savings without a "
            "shared-prefix run would be an assumed compromise of isolation."
        ),
    }


def gpu_absent_fields() -> dict[str, Any]:
    cmd = "swift receipts/headless/GPU_LEDGER_METAL_PROBE.json"
    return {
        "DRAM_READ_BYTES": qty(
            None, kind=ABSENT, unit="bytes", command=cmd,
            absent_reason=ABSENT_COUNTERS_REASON,
        ),
        "DRAM_WRITE_BYTES": qty(
            None, kind=ABSENT, unit="bytes", command=cmd,
            absent_reason=ABSENT_COUNTERS_REASON,
        ),
        "OS_PAGE_CACHE_COLD_GPU_NS": qty(
            None, kind=ABSENT, unit="ns", command=cmd,
            absent_reason=(
                "This box cannot purge the unified-memory file cache without "
                "a machine-level drop_caches that would disrupt other lanes. "
                "Fresh-process 'cold' still sees a warm VM cache after the "
                "first run. Reported separately from graph-cold step 0."
            ),
        ),
        "per_dispatch_gpu_ns": qty(
            None, kind=ABSENT, unit="ns", command=cmd,
            absent_reason=(
                "Production is one mixed command buffer. "
                "supportsCounterSampling.atDispatchBoundary=false, so the "
                "964/756 dispatches cannot be split. GPU_NS is the CB "
                "GPUEnd−GPUStart interval."
            ),
        ),
        "SIMD_utilization": qty(
            None, kind=ABSENT, unit="ratio", command=cmd,
            absent_reason=ABSENT_COUNTERS_REASON,
        ),
        "hardware_occupancy_counter": qty(
            None, kind=ABSENT, unit="ratio", command=cmd,
            absent_reason=ABSENT_COUNTERS_REASON,
        ),
    }


def build(live: bool = False, lengths: tuple[str, ...] | None = None) -> dict[str, Any]:
    self_check_workspace()
    lengths = lengths or LENGTHS
    probe = metal_probe()
    occ_before = occupancy_snapshot()
    parent_before = parent_identity(PARENT_ROOT)
    q4_bytes = q4_model_bytes()
    parent_bytes = parent_model_bytes(parent_before)

    q4_meas: dict[str, Any]
    parent_meas: dict[str, Any]
    if live:
        refuse_second_27b()
        print("== q4 incumbent (no parent resident) ==", flush=True)
        measure_model(
            model_key="q4",
            artifact=Q4_ROOT,
            fusion=False,
            lengths=lengths,
            parent_before=parent_before,
        )
        refuse_second_27b()
        print("== sealed parent (q4 process gone) ==", flush=True)
        measure_model(
            model_key="parent",
            artifact=PARENT_ROOT,
            fusion=True,
            lengths=lengths,
            parent_before=parent_before,
        )
    # Receipt is assembled from on-disk RAWs so a later --lengths 4k
    # walk cannot wipe a completed short series.
    q4_meas = assemble_from_raws("q4", fusion=False)
    parent_meas = assemble_from_raws("parent", fusion=True)

    parent_after = parent_identity(PARENT_ROOT)
    occ_after = occupancy_snapshot()

    ms_cite = None
    if MULTISESSION_RECEIPT.is_file():
        ms = json.loads(MULTISESSION_RECEIPT.read_text())
        proof = ms.get("proof_one_body") or {}
        ms_cite = {
            "source": "receipts/headless/NOETIC_MULTISESSION.json",
            "seq": (ms.get("workspace_formula") or {}).get("max_seq_len"),
            "workspace_total_bytes": (ms.get("workspace_formula") or {}).get("total_bytes"),
            "metal_c1_bytes": proof.get("metal_c1_bytes") or proof.get("rss_c1_bytes"),
            "metal_c4_bytes": proof.get("metal_c4_bytes") or proof.get("rss_c4_bytes"),
            "kind": "CITED",
            "note": (
                "Validates SESSION_STATE_BYTES at seq=256: c=4 minus c=1 "
                "tracks 3 × workspace, not 3 × (model+workspace)."
            ),
        }

    q4_foot = footprint_table(q4_bytes["payload_bytes"], "q4")
    parent_foot = footprint_table(parent_bytes["payload_bytes"], "parent")

    # Headline: at 16K / c=4, does state rival weights?
    q4_16k_c4 = q4_foot["by_length"]["16k"]["c=4"]
    parent_16k_c4 = parent_foot["by_length"]["16k"]["c=4"]
    q4_long_c4 = q4_foot["by_length"]["long"]["c=4"]
    parent_long_c4 = parent_foot["by_length"]["long"]["c=4"]

    answer_bits = []
    answer_bits.append(
        "Prefill on this body is the decode GEMV graph, teacher-forced. "
        "It is decode-optimal and not production-optimal for AgentOS "
        "long-context TTFT. A separate batched-prefill path is allowed "
        "and is not shipped here."
    )
    q4_16 = (q4_meas.get("lengths") or {}).get("16k") or {}
    if (q4_16.get("warm_prefill_gpu_ns_per_token") or {}).get("kind") == MEASURED:
        v = q4_16["warm_prefill_gpu_ns_per_token"]["value"]
        answer_bits.append(
            f"q4 16K warm GPU {v/1e6:.2f} ms/token (MEASURED)."
        )
    if q4_long_c4["state_exceeds_weights"]:
        answer_bits.append(
            f"q4 c=4 at 32K: SESSION_STATE_x_c "
            f"{q4_long_c4['SESSION_STATE_BYTES_x_c']/GIB:.2f} GiB exceeds "
            f"MODEL_BYTES {q4_bytes['payload_bytes']/GIB:.2f} GiB."
        )
    else:
        answer_bits.append(
            f"q4 c=4 at 32K: state {q4_long_c4['SESSION_STATE_BYTES_x_c']/GIB:.2f} GiB "
            f"vs weights {q4_bytes['payload_bytes']/GIB:.2f} GiB "
            f"(crossover at seq≈{q4_foot['crossover']['c=4']['seq_where_c_state_exceeds_weights']})."
        )

    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": "N016 — PREFILL AND KV: decode-optimal is not production-optimal",
        "question": (
            "What does prefill cost at short / 4K / 16K / long context on the "
            "q4 incumbent and the sealed parent, and what is the production "
            "footprint MODEL_BYTES + SESSION_STATE_BYTES at c=1 and c=4?"
        ),
        "answer": " ".join(answer_bits),
        "did_not_load_second_27b": True,
        "did_not_mutate_sealed_parent": identities_match(parent_before, parent_after),
        "did_not_write_ascent_or_campaign": True,
        "gpu_timestamp_authority": (
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; "
            "never a CPU-wait proxy"
        ),
        "measurement_label": "DIRTY_ENGINEERING",
        "measurement_label_reason": (
            "GPU lock held for the walks; other CPU/memory lanes may still "
            "be live. Not offered as CLEAN_CANDIDATE or BASE_TRUE_TPS. "
            "Cold vs warm reported separately; OS page-cache cold is ABSENT."
        ),
        "incumbent": q4_bytes,
        "sealed_parent": parent_bytes,
        "parent_identity_before": parent_before,
        "parent_identity_after": parent_after,
        "occupancy_before": occ_before,
        "occupancy_after": occ_after,
        "metal_probe": {
            "name": probe.get("name"),
            "hasUnifiedMemory": probe.get("hasUnifiedMemory"),
            "recommendedMaxWorkingSetSize": probe.get("recommendedMaxWorkingSetSize"),
            "counterSets": probe.get("counterSets"),
            "supportsCounterSampling": probe.get("supportsCounterSampling"),
            "command": probe.get("command"),
        },
        "lengths": {
            "short": "native compiler-prompt chat wrap (MEASURED prompt_len)",
            "4k": 4096,
            "16k": 16384,
            "long": 32768,
        },
        "prefill": {
            "q4": q4_meas.get("lengths") or {},
            "parent": parent_meas.get("lengths") or {},
        },
        "production_footprint": {
            "q4": q4_foot,
            "parent": parent_foot,
        },
        "headline_footprint": {
            "q4_16k_c1": q4_foot["by_length"]["16k"]["c=1"],
            "q4_16k_c4": q4_16k_c4,
            "q4_long_c4": q4_long_c4,
            "parent_16k_c4": parent_16k_c4,
            "parent_long_c4": parent_long_c4,
        },
        "workspace_formula_validation": ms_cite,
        "kv_precision": {
            "4k": kv_precision_options(4096),
            "16k": kv_precision_options(16384),
            "long": kv_precision_options(32768),
        },
        "prefix_sharing": prefix_sharing_note(),
        "topology": topology_verdict(q4_meas, parent_meas),
        "capability_qualification": capability_qualification(q4_meas, parent_meas),
        "absent_gpu_counters": gpu_absent_fields(),
        "q4_live": {k: v for k, v in q4_meas.items() if k != "lengths"},
        "parent_live": {k: v for k, v in parent_meas.items() if k != "lengths"},
    }
    return doc


def strip_run_blobs(doc: dict[str, Any]) -> dict[str, Any]:
    """Keep the receipt readable: drop _runs blobs from the published file."""
    pre = doc.get("prefill") or {}
    for model in ("q4", "parent"):
        lengths = pre.get(model) or {}
        for length, block in list(lengths.items()):
            if isinstance(block, dict) and "_runs" in block:
                block = dict(block)
                block.pop("_runs", None)
                lengths[length] = block
    return doc


def write_receipt(doc: dict[str, Any]) -> None:
    atomic_write(RECEIPT, strip_run_blobs(doc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument(
        "--lengths",
        default=",".join(LENGTHS),
        help="comma list: short,4k,16k,long",
    )
    args = ap.parse_args()
    lengths = tuple(x.strip() for x in args.lengths.split(",") if x.strip())
    for x in lengths:
        if x not in LENGTHS:
            raise SystemExit(f"unknown length {x}; want {LENGTHS}")
    live = bool(args.measure)
    if not live:
        print("assembling from raws (no --measure)", flush=True)
    doc = build(live=live, lengths=lengths)
    write_receipt(doc)
    print(f"wrote {RECEIPT}", flush=True)
    print(doc.get("answer", ""), flush=True)
    return 0 if doc.get("did_not_mutate_sealed_parent") else 2


if __name__ == "__main__":
    raise SystemExit(main())
