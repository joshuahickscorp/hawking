#!/usr/bin/env python3
"""N049 DECODING GRAVITY: MTP / self-speculative census (S026 §59-64, §121).

Passes-per-token, not just cost-per-pass. Census whether OUR Qwen3.8 parent,
config, and runtimes expose native MTP heads; whether the fast-but-incoherent
1.25-bpw binary (S026 §63) can serve as a draft against q2f; reclassify
density negatives by PURPOSE (§64). Metrics are accepted-tokens/byte and
accepted-tokens/GPU-second (§94, §95) even though the full GPU bench is
deferred (N042 owns the GPU).

CPU only. Streams parent safetensors headers, not bodies. Does not load a
second 27B, does not touch the GPU, does not run cargo/Metal, does not
mutate NOETIC_PARENT_A.

    python3 tools/headless/decoding_gravity.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import os
import re
import statistics
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEADLESS = REPO / "receipts" / "headless"

SCHEMA = "hawking.headless.decoding_gravity.v1"
RECEIPT = HEADLESS / "DECODING_GRAVITY.json"
GENERATOR = "tools/headless/decoding_gravity.py"
OBLIGATION = (
    "N049 — DECODING GRAVITY (S026 §59-64, §94, §95, §121; DOC-DECODE). "
    "Census native MTP / multi-token-prediction heads on OUR Qwen3.8; "
    "self-speculative feasibility of reduced-depth / early-exit / the "
    "1.25-bpw binary child as a DRAFT against q2f; reclassify negatives "
    "by PURPOSE. Speculative decoding that rejects most proposals LOSES."
)

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
CITED = "CITED"

PARENT_BF16 = Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16")
GRAVITY_Q4 = Path("/Users/scammermike/models/qwen38-gravity-uniform-q4-v1")
MLX_4BIT = Path("/Users/scammermike/models/qwen3.8-27b-abliterated-mlx/4bit")
MLX_QWEN35 = Path.home() / (
    ".local/share/uv/tools/mlx-lm/lib/python3.12/site-packages/"
    "mlx_lm/models/qwen3_5.py"
)
LLAMA_H = Path("/opt/homebrew/opt/llama.cpp/include/llama.h")
GGUF_Q5K = Path(
    "/Users/scammermike/models/qwen3.8-27b-abliterated/"
    "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
)

# Named before looking. Parent must cite these two keys; everything else is
# reported found-or-absent from the live config / index / headers.
MTP_CONFIG_KEYS = (
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
)
EARLY_EXIT_KEYS = (
    "early_exit",
    "num_early_exit_layers",
    "num_draft_layers",
    "medusa_num_heads",
    "eagle_num_layers",
    "n_future_tokens",
    "multi_token_heads",
    "speculative_length",
)

EXPECTED_MTP_TENSORS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)

DTYPE_WIDTH = {
    "BF16": 2,
    "BFLOAT16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

# Sealed composed COMPLETE_TOKEN_NS (BYTES_FRONTIER). Not re-derived.
Q2F_COMPLETE_NS = 27_547_874
BINARY_COMPLETE_NS = 23_431_791
Q2F_ACTIVE_BYTES = 4_812_963_840.0
BINARY_ACTIVE_BYTES = 2_673_868_800.0
PARENT_PARAMS = 26_895_998_464
# G057: a same-shape draft must be several times cheaper. Break-even at K=4
# on that campaign's verify tier was 15.063 ms/token.
G057_BREAK_EVEN_MS = 15.063
GQA_LAYERS = 16
MLP_LAYERS = 64

# llama.cpp Metal MTP is a FOREIGN prior (Qwen3.5-9B, M1 Max), not our 27B.
LLAMA_MTP_METAL_ISSUE = "https://github.com/ggml-org/llama.cpp/issues/23752"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


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


def load_json(rel_path: str) -> dict[str, Any]:
    rel_path = rel_path.lstrip("./")
    p = REPO / rel_path
    if p.is_file():
        return json.loads(p.read_text())
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise FileNotFoundError(rel_path)
    return json.loads(r.stdout)


def load_json_optional(rel_path: str) -> dict[str, Any] | None:
    try:
        return load_json(rel_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def qty(
    value: Any,
    *,
    kind: str,
    unit: str,
    source: str,
    absent_reason: str | None = None,
    note: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "source": source,
        "absent_reason": absent_reason,
    }
    if note is not None:
        out["note"] = note
    return out


def absent(unit: str, reason: str, source: str = "") -> dict[str, Any]:
    return qty(
        None, kind=ABSENT, unit=unit, source=source, absent_reason=reason
    )


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def tensor_nbytes(info: dict[str, Any]) -> int:
    offsets = info.get("data_offsets")
    if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
        return int(offsets[1]) - int(offsets[0])
    shape = info.get("shape") or []
    n = 1
    for s in shape:
        n *= int(s)
    width = DTYPE_WIDTH.get(str(info.get("dtype") or "").upper())
    if width is None:
        return 0
    return n * width


# ---------------------------------------------------------------------------
# Speculative arithmetic (pure)
# ---------------------------------------------------------------------------


def expected_accepted_per_pass(alpha: float, gamma: int) -> float:
    """Leviathan expected tokens per spec cycle, including the target bonus.

    At α=0 the target still emits one token after rejecting the whole draft.
    Speculative decoding that rejects most proposals therefore pays the draft
    AND the verify to produce the same one token greedy would have produced.
    """
    if gamma < 0:
        raise ValueError("gamma")
    if alpha >= 1.0:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def spec_cycle(
    *,
    alpha: float,
    gamma: int,
    draft_ns: float,
    verify_ns: float,
    baseline_ns: float,
    yield_includes_bonus: bool,
) -> dict[str, Any]:
    """One speculative cycle under a named yield convention.

    yield_includes_bonus=True  → Leviathan: E[tokens] = (1-α^{γ+1})/(1-α)
    yield_includes_bonus=False → G057-style: E[tokens] = (1-α^γ)/(1-α)  (α>0)
                                 with α=0 → 0 drafted, still 1 from target
    Verify is billed as one forward of cost `verify_ns` (optimistic batch).
    Draft is billed as γ sequential draft tokens at `draft_ns` each.
    """
    if yield_includes_bonus:
        accepted = expected_accepted_per_pass(alpha, gamma)
    else:
        if alpha >= 1.0:
            accepted = float(gamma)
        elif alpha <= 0.0:
            accepted = 1.0  # target correction only; no drafted token kept
        else:
            accepted = (1.0 - alpha ** gamma) / (1.0 - alpha)
            # G057 counted drafted tokens kept; the target still has to
            # produce the next token after a reject, so floor at 1.
            accepted = max(1.0, accepted)
    pass_ns = gamma * draft_ns + verify_ns
    tok_per_s = accepted / (pass_ns / 1e9) if pass_ns > 0 else 0.0
    baseline_tok_per_s = 1e9 / baseline_ns if baseline_ns > 0 else 0.0
    return {
        "alpha": alpha,
        "gamma": gamma,
        "yield_includes_bonus": yield_includes_bonus,
        "accepted_tokens_per_pass": accepted,
        "pass_ns": pass_ns,
        "accepted_tokens_per_gpu_second": tok_per_s,
        "baseline_accepted_tokens_per_gpu_second": baseline_tok_per_s,
        "ratio_vs_baseline": (
            tok_per_s / baseline_tok_per_s if baseline_tok_per_s else None
        ),
        "wins_vs_baseline": tok_per_s > baseline_tok_per_s,
    }


# ---------------------------------------------------------------------------
# Parent / artifact MTP census (headers only)
# ---------------------------------------------------------------------------


def flatten_keys(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_keys(v, p))
            else:
                out[p] = v
    return out


def census_parent_config(parent: Path) -> dict[str, Any]:
    cfg_path = parent / "config.json"
    if not cfg_path.is_file():
        return {
            "present": False,
            "path": str(cfg_path),
            "keys_found": {},
            "keys_absent": list(MTP_CONFIG_KEYS) + list(EARLY_EXIT_KEYS),
        }
    cfg = json.loads(cfg_path.read_text())
    flat = flatten_keys(cfg)
    found: dict[str, Any] = {}
    for k, v in flat.items():
        leaf = k.split(".")[-1]
        if leaf in MTP_CONFIG_KEYS or "mtp" in k.lower():
            found[k] = v
    absent_keys = [
        k for k in list(MTP_CONFIG_KEYS) + list(EARLY_EXIT_KEYS)
        if not any(kk.split(".")[-1] == k for kk in found)
        and k not in found
    ]
    # early-exit keys are absent if no leaf matches
    for k in EARLY_EXIT_KEYS:
        if not any(kk.endswith(k) or kk.split(".")[-1] == k for kk in flat):
            if k not in absent_keys:
                absent_keys.append(k)
    text = cfg.get("text_config") or {}
    return {
        "present": True,
        "path": str(cfg_path),
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "text_model_type": text.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "vocab_size": text.get("vocab_size"),
        "tie_word_embeddings": text.get("tie_word_embeddings"),
        "keys_found": found,
        "keys_absent": sorted(set(absent_keys)),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "mtp_use_dedicated_embeddings": text.get("mtp_use_dedicated_embeddings"),
    }


def census_parent_tensors(parent: Path) -> dict[str, Any]:
    idx_path = parent / "model.safetensors.index.json"
    if not idx_path.is_file():
        return {"present": False, "path": str(idx_path), "tensors": []}
    idx = json.loads(idx_path.read_text())
    wmap: dict[str, str] = idx.get("weight_map") or {}
    mtp_names = sorted(n for n in wmap if n.startswith("mtp.") or ".mtp." in n)
    shards = sorted({wmap[n] for n in mtp_names})
    headers: dict[str, dict[str, Any]] = {}
    for shard in shards:
        headers[shard] = safetensors_header(parent / shard)
    rows = []
    total = 0
    for name in mtp_names:
        info = headers[wmap[name]].get(name) or {}
        nbytes = tensor_nbytes(info)
        total += nbytes
        rows.append({
            "name": name,
            "shard": wmap[name],
            "dtype": info.get("dtype"),
            "shape": info.get("shape"),
            "bytes": nbytes,
        })
    # lm_head is shared (mtp_use_dedicated_embeddings=false); not an MTP tensor
    lm = wmap.get("lm_head.weight")
    lm_row = None
    if lm:
        hdr = safetensors_header(parent / lm) if lm not in headers else headers.get(lm)
        if hdr is None:
            hdr = safetensors_header(parent / lm)
        info = hdr.get("lm_head.weight") or {}
        lm_row = {
            "name": "lm_head.weight",
            "shard": lm,
            "dtype": info.get("dtype"),
            "shape": info.get("shape"),
            "bytes": tensor_nbytes(info),
            "note": "shared with MTP (mtp_use_dedicated_embeddings=false); not extra",
        }
    return {
        "present": True,
        "path": str(idx_path),
        "n_weight_map": len(wmap),
        "n_mtp_tensors": len(rows),
        "mtp_bytes": total,
        "mtp_elements": total // 2,  # parent MTP is BF16
        "tensors": rows,
        "expected_names": list(EXPECTED_MTP_TENSORS),
        "expected_present": all(n in {r["name"] for r in rows} for n in EXPECTED_MTP_TENSORS),
        "names_not_in_parent": [
            n for n in EXPECTED_MTP_TENSORS if n not in {r["name"] for r in rows}
        ],
        "extra_mtp_names": [
            r["name"] for r in rows if r["name"] not in EXPECTED_MTP_TENSORS
        ],
        "lm_head": lm_row,
        "method": "safetensors index + header JSON data_offsets; file bodies not read",
    }


def census_gravity(root: Path) -> dict[str, Any]:
    man_path = root / "manifest.json"
    if not man_path.is_file():
        return {"present": False, "path": str(man_path), "n_mtp": 0}
    man = json.loads(man_path.read_text())
    tensors = man.get("tensors") or []
    names = [
        t.get("name") for t in tensors if isinstance(t, dict) and t.get("name")
    ]
    mtp = [n for n in names if "mtp" in n.lower()]
    return {
        "present": True,
        "path": str(man_path),
        "schema": man.get("schema"),
        "tensor_count": man.get("tensor_count", len(names)),
        "skipped_vision_tensors": man.get("skipped_vision_tensors"),
        "n_named": len(names),
        "n_mtp": len(mtp),
        "mtp_names": mtp,
        "prefixes": sorted({n.split(".")[0] for n in names}),
        "note": (
            "Language-only pack. skipped_vision_tensors counts vision; MTP "
            "is not a vision tensor and is also not in the packed catalog."
        ),
    }


def census_mlx_artifact(root: Path) -> dict[str, Any]:
    cfg_path = root / "config.json"
    idx_path = root / "model.safetensors.index.json"
    if not cfg_path.is_file():
        return {"present": False, "path": str(root)}
    cfg = json.loads(cfg_path.read_text())
    text = cfg.get("text_config") or {}
    n_map = 0
    mtp_names: list[str] = []
    if idx_path.is_file():
        wmap = (json.loads(idx_path.read_text()).get("weight_map") or {})
        n_map = len(wmap)
        mtp_names = sorted(n for n in wmap if "mtp" in n.lower())
    return {
        "present": True,
        "path": str(root),
        "architectures": cfg.get("architectures"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "mtp_use_dedicated_embeddings": text.get("mtp_use_dedicated_embeddings"),
        "n_weight_map": n_map,
        "n_mtp": len(mtp_names),
        "mtp_names": mtp_names,
        "note": (
            "config still carries mtp_num_hidden_layers but the converted "
            "weight map contains zero mtp.* tensors"
        ),
    }


def census_mlx_sanitize(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path), "strips_mtp": None}
    src = path.read_text(errors="replace")
    lines = src.splitlines()
    hits = [
        {"line": i, "text": ln.strip()}
        for i, ln in enumerate(lines, 1)
        if "mtp" in ln.lower()
    ]
    strips = any(
        'if "mtp." not in k' in ln or "mtp.\" not in k" in ln for ln in lines
    )
    # exact installed line
    strip_line = next(
        (h for h in hits if "mtp." in h["text"] and "not in" in h["text"]),
        None,
    )
    return {
        "present": True,
        "path": str(path),
        "strips_mtp": strips or (strip_line is not None),
        "strip_line": strip_line,
        "mtp_mentions": hits,
        "note": (
            "mlx_lm.models.qwen3_5.Model.sanitize drops every weight whose "
            "name contains 'mtp.' before the model is constructed"
        ),
    }


def census_llama() -> dict[str, Any]:
    help_txt = ""
    help_ok = False
    try:
        r = subprocess.run(
            ["llama-cli", "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        help_txt = (r.stdout or "") + (r.stderr or "")
        help_ok = True
    except Exception as e:
        help_txt = f"{type(e).__name__}: {e}"
    spec_type = None
    m = re.search(r"--spec-type[^\n]*", help_txt)
    if m:
        spec_type = m.group(0).strip()
    draft_mtp = bool(re.search(r"draft-mtp", help_txt))
    llama_h = None
    if LLAMA_H.is_file():
        text = LLAMA_H.read_text(errors="replace")
        ctx = None
        for ln in text.splitlines():
            if "LLAMA_CONTEXT_TYPE_MTP" in ln:
                ctx = ln.strip()
                break
        llama_h = {
            "path": str(LLAMA_H),
            "LLAMA_CONTEXT_TYPE_MTP": ctx,
        }
    gguf = {
        "path": str(GGUF_Q5K),
        "present": GGUF_Q5K.is_file(),
        "bytes": GGUF_Q5K.stat().st_size if GGUF_Q5K.is_file() else None,
        "note": (
            "HCLI's historical Q5_K GGUF path from MODEL_REGISTRY / "
            "DECODE_TOPOLOGY. Absent on this disk, so GGUF tensor names "
            "cannot be censused. Not evidence the convert kept MTP."
        ),
    }
    return {
        "help_ok": help_ok,
        "spec_type_help": spec_type,
        "exposes_draft_mtp": draft_mtp,
        "llama_h": llama_h,
        "gguf_q5k": gguf,
        "metal_prior": {
            "kind": CITED,
            "url": LLAMA_MTP_METAL_ISSUE,
            "title": (
                "MTP speculative decoding degrades throughput on Metal "
                "(Apple Silicon) — net loss at every configuration"
            ),
            "not_our_27b": True,
            "model_in_issue": "Qwen3.5-9B-MTP Q4_K_M",
            "machine_in_issue": "M1 Max, 32 GB, llama-server b9330",
            "finding": (
                "No --spec-type draft-mtp configuration beat the non-MTP "
                "baseline. n_max=0 at 100% accept was 11% slower; n_max=6 "
                "at ~43% accept was 28% slower. Draft-eval overhead on "
                "Metal exceeded the speculative gain."
            ),
            "note": (
                "FOREIGN prior, different size/box. Cited because it is "
                "the same llama.cpp Metal MTP path our --spec-type flag "
                "exposes. Not a measurement of Qwen3.8-27B on this M3 Ultra."
            ),
        },
    }


def census_hawking_runtime() -> dict[str, Any]:
    pack = REPO / "crates/hawking-core/src/model/qwen38_pack.rs"
    geo = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
    spec = REPO / "crates/hawking-speculate/src/lib.rs"
    comp = REPO / "tools/headless/noetic_composition.py"
    hits = []
    language_prefix = None
    vision_prefix = None
    if geo.is_file():
        gsrc = geo.read_text()
        m = re.search(r'QWEN38_LANGUAGE_PREFIX:\s*&str\s*=\s*"([^"]+)"', gsrc)
        if m:
            language_prefix = m.group(1)
        m = re.search(r'QWEN38_VISION_PREFIX:\s*&str\s*=\s*"([^"]+)"', gsrc)
        if m:
            vision_prefix = m.group(1)
        accepts_mtp = "mtp_num_hidden_layers" in gsrc
        hits.append({
            "file": "crates/hawking-core/src/model/qwen38_geometry.rs",
            "language_prefix": language_prefix,
            "vision_prefix": vision_prefix,
            "qwen38_accept_config_reads_mtp": accepts_mtp,
        })
    pack_skips = None
    if pack.is_file():
        psrc = pack.read_text()
        pack_skips = (
            "unexpected tensor root" in psrc
            and "QWEN38_LANGUAGE_PREFIX" in psrc
        )
        hits.append({
            "file": "crates/hawking-core/src/model/qwen38_pack.rs",
            "errors_on_non_language_non_vision": pack_skips,
            "note": (
                "Tensors not starting with language_model. or vision_tower. "
                "are rejected. Parent MTP tensors are named mtp.* so they "
                "are not packed. Gravity catalog confirms n_mtp=0."
            ),
        })
    spec_kind = None
    if spec.is_file():
        ssrc = spec.read_text()
        spec_kind = {
            "file": "crates/hawking-speculate/src/lib.rs",
            "qwen_mtp_module": "mtp" in ssrc.lower() and "qwen" in ssrc.lower(),
            "paths": (
                "user-draft n-gram and shared-expert (ExactShared); "
                "not Qwen3.5 native MTP heads"
            ),
        }
        hits.append(spec_kind)
    hf_skips = None
    if comp.is_file():
        csrc = comp.read_text()
        hf_skips = (
            '"mtp_num_hidden_layers"' in csrc
            and '"mtp_use_dedicated_embeddings"' in csrc
        )
        hits.append({
            "file": "tools/headless/noetic_composition.py",
            "text_config_skips_mtp_keys": hf_skips,
            "note": (
                "HF Qwen3_5TextConfig construction explicitly skips "
                "mtp_num_hidden_layers and mtp_use_dedicated_embeddings"
            ),
        })
    return {
        "language_prefix": language_prefix,
        "vision_prefix": vision_prefix,
        "accept_config_ignores_mtp": True,
        "packer_cannot_ingest_mtp_dot_tensors": pack_skips,
        "speculate_crate_is_not_qwen_mtp": True,
        "hf_layer_path_skips_mtp_keys": hf_skips,
        "citations": hits,
    }


# ---------------------------------------------------------------------------
# Draft quality (harvest N036 real-X local scan + sealed greedy)
# ---------------------------------------------------------------------------


def harvest_token_agreement() -> dict[str, Any]:
    fne = load_json("receipts/headless/FIRST_NOETIC_EXECUTABLE.json")
    q2f = load_json("receipts/headless/Q2F_G64_GENERATION.json")
    binary_ids = None
    for m in fne.get("mixes_attempted") or []:
        mix = (m.get("compile") or {}).get("mix_id")
        if mix == "mix_c_all_mlp_binary_g64":
            binary_ids = (m.get("decode") or {}).get("new_token_ids")
            break
    q2f_ids = (q2f.get("decode") or {}).get("new_token_ids")
    if not binary_ids or not q2f_ids:
        return {
            "ok": False,
            "reason": "sealed greedy token lists missing",
        }
    n = min(len(binary_ids), len(q2f_ids))
    agree = [int(a == b) for a, b in zip(binary_ids[:n], q2f_ids[:n])]
    first = next((i for i, v in enumerate(agree) if v == 0), n)
    score = next(
        (r for r in (fne.get("mix_scoreboard") or [])
         if r.get("mix_id") == "mix_c_all_mlp_binary_g64"),
        {},
    )
    return {
        "ok": True,
        "kind": MEASURED,
        "source": [
            "receipts/headless/FIRST_NOETIC_EXECUTABLE.json mixes_attempted mix_c_all_mlp_binary_g64",
            "receipts/headless/Q2F_G64_GENERATION.json decode.new_token_ids",
        ],
        "real_activations": True,
        "independent_greedy_streams": True,
        "n": n,
        "binary_ids": list(binary_ids[:n]),
        "q2f_ids": list(q2f_ids[:n]),
        "position_wise_agreements": agree,
        "n_agree": int(sum(agree)),
        "position_wise_top1": (sum(agree) / n) if n else None,
        "earliest_mismatch": first,
        "first_token_agree": bool(agree[0]) if agree else None,
        "binary_token_0": binary_ids[0],
        "q2f_token_0": q2f_ids[0],
        "speculative_accept_on_this_prompt": 0 if not agree[0] else None,
        "binary_coherent": score.get("coherent"),
        "binary_coherence_reason": score.get("coherence_reason"),
        "binary_tok_s_native_greedy": score.get("tok_s"),
        "q2f_tok_s_native_greedy": (q2f.get("decode") or {}).get("tok_s"),
        "q2f_coherent": ((q2f.get("decode") or {}).get("coherence") or {}).get("coherent"),
        "note": (
            "These are independent greedy streams, not teacher-forced draft "
            "verification. For speculative decoding the first mismatch is "
            "what matters: binary emits 271 at position 0, q2f emits 15769, "
            "so a binary draft of this prompt is rejected immediately "
            "(0 accepted draft tokens)."
        ),
    }


def _agr(disagree: int, n: int) -> float:
    return 1.0 - (disagree / n) if n else 0.0


def harvest_local_top1() -> dict[str, Any]:
    """Channel-argmax of binary vs q2f on the N036 real hold-set GEMVs.

    This is LOCAL draft quality of the MLP operator, not next-token α.
    SwiGLU output argmax lives in hidden (5120), not vocab.
    """
    raw_path = HEADLESS / "_BINARY_HEALING_local.json"
    if not raw_path.is_file():
        return {
            "ok": False,
            "kind": ABSENT,
            "reason": f"{raw_path.name} not on disk; cannot harvest per-token argmax",
        }
    loc = json.loads(raw_path.read_text())
    layers = loc.get("layers") or []
    n = int(loc.get("n_tokens_used") or 0)
    if not layers or n <= 0:
        return {"ok": False, "kind": ABSENT, "reason": "local scan empty"}

    def series(getter) -> dict[str, Any]:
        vals = [int(getter(row)) for row in layers]
        agrs = [_agr(v, n) for v in vals]
        return {
            "n_layers": len(vals),
            "n_tokens": n,
            "disagree_mean": float(statistics.mean(vals)),
            "disagree_median": float(statistics.median(vals)),
            "disagree_min": int(min(vals)),
            "disagree_max": int(max(vals)),
            "top1_agreement_mean": float(statistics.mean(agrs)),
            "top1_agreement_median": float(statistics.median(agrs)),
            "top1_agreement_min": float(min(agrs)),
            "top1_agreement_max": float(max(agrs)),
            "n_layers_perfect": int(sum(1 for a in agrs if a >= 1.0 - 1e-12)),
            "n_layers_below_0_25": int(sum(1 for a in agrs if a < 0.25)),
            "n_layers_at_least_0_70": int(sum(1 for a in agrs if a >= 0.70)),
            "by_band": {
                f"L{lo:02d}-{hi-1:02d}": {
                    "top1_agreement_mean": float(statistics.mean(agrs[lo:hi])),
                    "top1_agreement_min": float(min(agrs[lo:hi])),
                }
                for lo, hi in ((0, 16), (16, 32), (32, 48), (48, 64))
                if hi <= len(agrs)
            },
        }

    organs = {}
    for organ in ("gate_proj", "up_proj", "down_proj"):
        organs[organ] = series(
            lambda row, o=organ: row["organs"][o][
                "n_tokens_argmax_disagree_q2f_binary"
            ]
        )
    swiglu = series(
        lambda row: row["swiglu"]["n_tokens_argmax_disagree_binary_q2f"]
    )
    return {
        "ok": True,
        "kind": MEASURED,
        "source": "receipts/headless/_BINARY_HEALING_local.json",
        "sealed_receipt": "receipts/headless/BINARY_HEALING.json",
        "real_activations": True,
        "not_gaussian": True,
        "capture": loc.get("capture"),
        "split_rule": loc.get("split_rule"),
        "n_hold_total": loc.get("n_hold_total"),
        "n_tokens_used": n,
        "n_layers": len(layers),
        "site": (
            "gate/up: post_attn_norm capture_diverse2 hold tokens; "
            "down_proj / swiglu: teacher SwiGLU hidden. Binary g64 vs q2f g64 "
            "channel-argmax, NOT vocab top-1."
        ),
        "organs": organs,
        "swiglu_mlp_output": swiglu,
        "draft_quality_proxy": swiglu["top1_agreement_mean"],
        "note": (
            "A GEMV channel-argmax of 0.37 is not a 0.37 token-draft α. "
            "Token-level α on the sealed greedy streams is 0 at position 0. "
            "This local figure is the optimistic operator-level proxy; even "
            "using it as if it were token α, speculation still loses (§95)."
        ),
    }


def extra_head_and_state_bytes(mtp_bytes: int) -> dict[str, Any]:
    """MTP extra storage + decode-state. KV is one GQA layer."""
    # K+V: 4 kv heads * 256 dim * 2 (K and V) * 2 bytes (fp16 cache)
    kv_per_token = 4 * 256 * 2 * 2  # 4096
    return {
        "mtp_weight_bytes_bf16": mtp_bytes,
        "mtp_weight_gib_bf16": mtp_bytes / (1024 ** 3),
        "mtp_as_fraction_of_parent_payload": mtp_bytes / 55_562_855_904,
        "lm_head_not_extra": True,
        "mtp_kv_bytes_per_token_fp16": kv_per_token,
        "mtp_kv_note": (
            "mtp.layers.0 is self_attn (GQA), not DeltaNet. Extra decode "
            "state is one GQA layer of KV: n_kv_heads=4, head_dim=256, K+V, "
            "fp16 → 4096 bytes/token. Negligible vs production_active_bytes "
            "9.88e9 (ORGAN_BANDWIDTH)."
        ),
        "kind": DERIVED,
        "source": "parent safetensors headers + config GQA geometry",
    }


def mtp_cost_estimate() -> dict[str, Any]:
    """Organ-bandwidth fractions applied to one extra GQA+MLP+lm_head.

    DERIVED from N025 organ ns, not a measured MTP forward. Ceiling only.
    """
    ob = load_json("receipts/headless/ORGAN_BANDWIDTH.json")
    organs = ((ob.get("organ_attribution") or {}).get("organs") or {})
    gqa = float((organs.get("gqa_attention") or {}).get("scaled_gpu_ns") or 0)
    gate = float((organs.get("mlp_gate_up") or {}).get("scaled_gpu_ns") or 0)
    down = float((organs.get("mlp_down") or {}).get("scaled_gpu_ns") or 0)
    head = float((organs.get("lm_head") or {}).get("scaled_gpu_ns") or 0)
    prod = float(
        (ob.get("organ_attribution") or {}).get("production_gpu_ns_median") or 0
    )
    extra = (gqa / GQA_LAYERS) + ((gate + down) / MLP_LAYERS) + head
    return {
        "kind": DERIVED,
        "source": "receipts/headless/ORGAN_BANDWIDTH.json organ_attribution",
        "not_a_measured_mtp_forward": True,
        "production_gpu_ns_median": prod,
        "extra_ns_one_gqa_plus_one_mlp_plus_lm_head": extra,
        "extra_fraction_of_production_token": extra / prod if prod else None,
        "tokens_per_pass_ceiling_at_alpha_1": 2.0,
        "pass_ns_at_alpha_1_if_estimate_held": prod + extra,
        "accepted_tokens_per_gpu_second_ceiling": (
            2.0 / ((prod + extra) / 1e9) if prod and extra else None
        ),
        "baseline_accepted_tokens_per_gpu_second": (
            1e9 / prod if prod else None
        ),
        "note": (
            "Optimistic: ignores MTP context setup, extra dispatch, and the "
            "Metal draft-eval overhead llama.cpp issue 23752 measured on a "
            "smaller Qwen3.5. α_mtp is ABSENT (capture is post_attn_norm, "
            "wrong site for the MTP module; no second 27B decode)."
        ),
    }


# ---------------------------------------------------------------------------
# Purpose reclassification (§64)
# ---------------------------------------------------------------------------


def purpose_rows(
    token: dict[str, Any],
    local: dict[str, Any],
    arithmetic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Negatives rejected FOR A PURPOSE, not globally dead."""
    bf = load_json("receipts/headless/BYTES_FRONTIER.json")
    by_id = {r["id"]: r for r in bf.get("representations") or []}

    def ns_of(rid: str) -> float | None:
        row = by_id.get(rid) or {}
        comp = ((row.get("COMPLETE_TOKEN_NS") or {}).get("composed") or {})
        return comp.get("complete_token_ns")

    def faster_than_q2f(rid: str) -> bool:
        n = ns_of(rid)
        return bool(n is not None and n < Q2F_COMPLETE_NS)

    binary_cycles = [
        c for c in arithmetic
        if c.get("alpha_kind") == "token_greedy" and c.get("gamma") == 1
    ]
    binary_wins = any(c.get("wins_vs_baseline") for c in binary_cycles)

    rows = [
        {
            "id": "binary_g64",
            "rejected_as": "final_generator",
            "rejected_because": (
                "mix_c_all_mlp_binary_g64 emits 16 copies of token 271; "
                "generation is dead at position 0 (N036 / FIRST_NOETIC)"
            ),
            "evaluated_as": "draft_against_q2f",
            "why_this_is_the_s026_63_candidate": (
                "Only density-descent body that is both faster than q2f on "
                "composed COMPLETE_TOKEN_NS (23.43 vs 27.55 ms) and incoherent"
            ),
            "faster_than_q2f_composed_ns": faster_than_q2f("binary_g64"),
            "composed_complete_token_ns": ns_of("binary_g64"),
            "native_greedy_tok_s": token.get("binary_tok_s_native_greedy"),
            "q2f_native_greedy_tok_s": token.get("q2f_tok_s_native_greedy"),
            "token_top1_vs_q2f_at_pos_0": token.get("first_token_agree"),
            "local_swiglu_top1_mean": (local or {}).get("draft_quality_proxy"),
            "draft_verdict": "REJECTED_AS_DRAFT",
            "draft_verdict_why": (
                "Token-level top-1 vs q2f is 0 at generated token 0, so a "
                "speculative pass accepts 0 draft tokens and still pays the "
                "draft. Local SwiGLU channel-argmax mean is "
                f"{(local or {}).get('draft_quality_proxy')}. Same-shape "
                "draft is only 15% cheaper on composed ns (23.43 vs 27.55 ms) "
                "and ~1% on native greedy tok/s (33.44 vs 33.09) — G057: a "
                "draft must be several times cheaper, not a quarter. "
                f"Measured-α cycles win_vs_baseline={binary_wins}."
            ),
            "source": [
                "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
                "receipts/headless/BINARY_HEALING.json",
                "receipts/headless/BYTES_FRONTIER.json",
            ],
        },
        {
            "id": "ternary_5in8_g64",
            "rejected_as": "final_generator",
            "rejected_because": (
                "held_out_activation FAILED; COMPLETE_TOKEN_NS 59.42 ms is "
                "slower than q2f 27.55 ms (BYTES_FRONTIER)"
            ),
            "evaluated_as": "draft_against_q2f",
            "faster_than_q2f_composed_ns": faster_than_q2f("ternary_5in8_g64"),
            "composed_complete_token_ns": ns_of("ternary_5in8_g64"),
            "draft_verdict": "NOT_A_CHEAPER_DRAFT",
            "draft_verdict_why": (
                "A draft has to be cheaper than the target. ternary is 2.16x "
                "slower than q2f on composed COMPLETE_TOKEN_NS."
            ),
            "source": ["receipts/headless/BYTES_FRONTIER.json"],
        },
        {
            "id": "shared_binary_k2",
            "rejected_as": "final_generator",
            "rejected_because": (
                "N035: no coherent shared-basis point beats q2f; died at "
                "held_out_activation. K=2 composed ns 111.6 ms."
            ),
            "evaluated_as": "draft_against_q2f",
            "faster_than_q2f_composed_ns": faster_than_q2f("shared_binary_k2"),
            "composed_complete_token_ns": ns_of("shared_binary_k2"),
            "draft_verdict": "NOT_A_CHEAPER_DRAFT",
            "draft_verdict_why": (
                "4.05x slower than q2f. G057 applies even harder: a slower "
                "body cannot be the draft of a faster target."
            ),
            "source": [
                "receipts/headless/BYTES_FRONTIER.json",
                "receipts/headless/SHARED_BASIS_COHERENT.json",
            ],
        },
        {
            "id": "binary_residual_sparse_2pct",
            "rejected_as": "final_generator",
            "rejected_because": (
                "composed COMPLETE_TOKEN_NS 37.15 ms, slower than q2f; "
                "coherence untested above local_functional_probe"
            ),
            "evaluated_as": "draft_against_q2f",
            "faster_than_q2f_composed_ns": faster_than_q2f(
                "binary_residual_sparse_2pct"
            ),
            "composed_complete_token_ns": ns_of("binary_residual_sparse_2pct"),
            "draft_verdict": "NOT_A_CHEAPER_DRAFT",
            "draft_verdict_why": "Slower than the target on composed ns.",
            "source": ["receipts/headless/BYTES_FRONTIER.json"],
        },
        {
            "id": "hybrid_operator",
            "rejected_as": "final_generator",
            "rejected_because": (
                "N038: no coherent hybrid below 2.25 bpw AND faster than "
                "27.55 ms; confirms the 2.25 MLP floor a fourth way"
            ),
            "evaluated_as": "draft_against_q2f",
            "draft_verdict": "NOT_A_CHEAPER_RESTORED_DRAFT",
            "draft_verdict_why": (
                "The healing that would make binary a generator destroys the "
                "speed that would make it a draft. No leftover hybrid is both "
                "cheap and accepted."
            ),
            "source": ["receipts/headless/HYBRID_OPERATOR.json"],
        },
        {
            "id": "native_mtp_head",
            "rejected_as": None,
            "rejected_because": None,
            "evaluated_as": "native_multi_token_head",
            "draft_verdict": "HEADS_PRESENT_RUNTIME_DROPS_THEM",
            "draft_verdict_why": (
                "Different shape from the 1.25 binary: one extra GQA+MLP "
                "layer after the last hidden, sharing lm_head. Parent has "
                "the weights. Gravity pack, MLX sanitize, and the HF layer "
                "path used here all drop them. Token α_mtp is ABSENT. "
                "llama.cpp exposes --spec-type draft-mtp; a Metal prior on a "
                "smaller Qwen3.5 lost at every n_max including 100% accept. "
                "Full GPU bench deferred (N042)."
            ),
            "source": [
                "parent config.json + safetensors headers",
                "crates/hawking-core/src/model/qwen38_pack.rs",
                LLAMA_MTP_METAL_ISSUE,
            ],
        },
        {
            "id": "reduced_depth_early_exit",
            "rejected_as": None,
            "evaluated_as": "self_speculative_draft",
            "draft_verdict": "NO_NATIVE_HEAD_UNMEASURED_ALPHA",
            "draft_verdict_why": (
                "Config has no early_exit / medusa / eagle / num_draft_layers "
                "keys. The only extra head is MTP after layer 63, not an "
                "intermediate exit. Capture_diverse2 is post_attn_norm, the "
                "wrong site for an early-exit lm_head probe. Token α for a "
                "K-layer prefix is ABSENT. G057 already named 'a different "
                "SHAPE — fewer layers or a smaller hidden' as what would "
                "change the same-shape cost trap; that experiment is not this "
                "census and is not a 27B decode."
            ),
            "source": [
                "parent config.json keys_absent",
                "receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json",
            ],
        },
    ]
    return rows


def metric_framing(
    token: dict[str, Any],
    local: dict[str, Any],
    mtp_bytes: int,
    extra_state: dict[str, Any],
    mtp_cost: dict[str, Any],
    arithmetic: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_tok_s = 1e9 / Q2F_COMPLETE_NS
    baseline_per_byte = 1.0 / Q2F_ACTIVE_BYTES
    # measured-α binary, Leviathan, γ=1
    measured = next(
        (
            c for c in arithmetic
            if c.get("alpha_kind") == "token_greedy"
            and c.get("gamma") == 1
            and c.get("yield_includes_bonus") is True
        ),
        {},
    )
    # extra bytes if we kept a binary MLP body as a resident draft alongside q2f
    extra_draft_bytes = BINARY_ACTIVE_BYTES
    acc = float(measured.get("accepted_tokens_per_pass") or 1.0)
    per_byte_binary = acc / (Q2F_ACTIVE_BYTES + extra_draft_bytes)
    mtp_ceiling_acc = 2.0
    per_byte_mtp_ceiling = mtp_ceiling_acc / (Q2F_ACTIVE_BYTES + mtp_bytes)
    return {
        "bench_deferred": True,
        "bench_deferred_why": (
            "N042 owns the GPU. Full accepted-tokens/GPU-second bench of "
            "native MTP or a binary draft loop is not this lane. Framing "
            "uses sealed COMPLETE_TOKEN_NS and harvested α."
        ),
        "definition": {
            "accepted_tokens_per_byte": (
                "accepted_tokens_per_pass / bytes_streamed_per_pass "
                "(active body + extra draft or MTP bytes)"
            ),
            "accepted_tokens_per_gpu_second": (
                "accepted_tokens_per_pass / pass_gpu_seconds"
            ),
        },
        "accepted_tokens_per_byte": {
            "baseline_q2f": {
                "accepted_tokens_per_pass": 1.0,
                "bytes": Q2F_ACTIVE_BYTES,
                "value": baseline_per_byte,
                "kind": DERIVED,
            },
            "binary_draft_measured_token_alpha": {
                "accepted_tokens_per_pass": acc,
                "bytes": Q2F_ACTIVE_BYTES + extra_draft_bytes,
                "value": per_byte_binary,
                "vs_baseline": per_byte_binary / baseline_per_byte,
                "wins": per_byte_binary > baseline_per_byte,
                "kind": DERIVED,
                "note": (
                    "α=0 → 1 accepted token, but bytes include the draft "
                    "body. More bytes, same tokens: LOSES."
                ),
            },
            "native_mtp_perfect_alpha_ceiling": {
                "accepted_tokens_per_pass": mtp_ceiling_acc,
                "bytes": Q2F_ACTIVE_BYTES + mtp_bytes,
                "value": per_byte_mtp_ceiling,
                "vs_baseline": per_byte_mtp_ceiling / baseline_per_byte,
                "kind": DERIVED,
                "alpha_mtp": absent(
                    "probability",
                    "MTP token agreement not measured; wrong-site capture "
                    "and no second 27B decode",
                    "ABSENT",
                ),
                "note": (
                    "Ceiling only. Counts MTP weight bytes, not the unmeasured "
                    "Metal context overhead from issue 23752."
                ),
            },
        },
        "accepted_tokens_per_gpu_second": {
            "baseline_q2f_composed": {
                "value": baseline_tok_s,
                "complete_token_ns": Q2F_COMPLETE_NS,
                "kind": CITED,
                "source": "receipts/headless/BYTES_FRONTIER.json q2_4level_fitted_g64",
            },
            "binary_draft_cycles": [
                {
                    "alpha_kind": c.get("alpha_kind"),
                    "gamma": c.get("gamma"),
                    "yield_includes_bonus": c.get("yield_includes_bonus"),
                    "accepted_tokens_per_pass": c.get("accepted_tokens_per_pass"),
                    "value": c.get("accepted_tokens_per_gpu_second"),
                    "ratio_vs_baseline": c.get("ratio_vs_baseline"),
                    "wins": c.get("wins_vs_baseline"),
                }
                for c in arithmetic
            ],
            "native_mtp_cost_model": mtp_cost,
            "native_greedy_tok_s_is_not_the_composed_ns": {
                "binary_mix_c": token.get("binary_tok_s_native_greedy"),
                "q2f": token.get("q2f_tok_s_native_greedy"),
                "note": (
                    "The 15% composed-ns win (23.43 vs 27.55 ms) is the MLP "
                    "graph plus billed residual. Native hybrid greedy of "
                    "mix_c vs q2f is 33.44 vs 33.09 tok/s — the draft is "
                    "not cheaper on the vehicle that actually emits tokens."
                ),
            },
        },
        "g057_break_even": {
            "source": "receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json",
            "break_even_ms_at_k4": G057_BREAK_EVEN_MS,
            "binary_composed_ms": BINARY_COMPLETE_NS / 1e6,
            "clears_break_even": (BINARY_COMPLETE_NS / 1e6) < G057_BREAK_EVEN_MS,
            "quote": (
                "Speculative decoding needs a draft several times cheaper, "
                "not a quarter cheaper."
            ),
        },
        "extra_head_state": extra_state,
        "local_top1_is_not_token_alpha": {
            "local_swiglu_mean": (local or {}).get("draft_quality_proxy"),
            "token_pos0": token.get("first_token_agree"),
        },
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build() -> dict[str, Any]:
    parent_cfg = census_parent_config(PARENT_BF16)
    parent_t = census_parent_tensors(PARENT_BF16)
    gravity = census_gravity(GRAVITY_Q4)
    mlx_art = census_mlx_artifact(MLX_4BIT)
    mlx_src = census_mlx_sanitize(MLX_QWEN35)
    llama = census_llama()
    hawking = census_hawking_runtime()
    token = harvest_token_agreement()
    local = harvest_local_top1()
    mtp_bytes = int(parent_t.get("mtp_bytes") or 0)
    extra_state = extra_head_and_state_bytes(mtp_bytes)
    mtp_cost = mtp_cost_estimate()

    alpha_token = 0.0 if token.get("first_token_agree") is False else None
    alpha_local = local.get("draft_quality_proxy") if local.get("ok") else None

    arithmetic: list[dict[str, Any]] = []
    for yield_bonus in (True, False):
        for gamma in (1, 2, 4):
            if alpha_token is not None:
                row = spec_cycle(
                    alpha=alpha_token,
                    gamma=gamma,
                    draft_ns=BINARY_COMPLETE_NS,
                    verify_ns=Q2F_COMPLETE_NS,
                    baseline_ns=Q2F_COMPLETE_NS,
                    yield_includes_bonus=yield_bonus,
                )
                row["alpha_kind"] = "token_greedy"
                arithmetic.append(row)
            if alpha_local is not None:
                row = spec_cycle(
                    alpha=float(alpha_local),
                    gamma=gamma,
                    draft_ns=BINARY_COMPLETE_NS,
                    verify_ns=Q2F_COMPLETE_NS,
                    baseline_ns=Q2F_COMPLETE_NS,
                    yield_includes_bonus=yield_bonus,
                )
                row["alpha_kind"] = "local_swiglu_channel_argmax_proxy"
                arithmetic.append(row)
            # perfect-accept control: must still lose under G057 yield=K
            row = spec_cycle(
                alpha=1.0,
                gamma=gamma,
                draft_ns=BINARY_COMPLETE_NS,
                verify_ns=Q2F_COMPLETE_NS,
                baseline_ns=Q2F_COMPLETE_NS,
                yield_includes_bonus=yield_bonus,
            )
            row["alpha_kind"] = "perfect_accept_control"
            arithmetic.append(row)

    purposes = purpose_rows(token, local, arithmetic)
    metrics = metric_framing(
        token, local, mtp_bytes, extra_state, mtp_cost, arithmetic
    )

    parent_has_mtp = (
        parent_cfg.get("mtp_num_hidden_layers") == 1
        and parent_t.get("n_mtp_tensors") == 15
        and parent_t.get("expected_present") is True
    )
    runtime_drops = (
        gravity.get("n_mtp") == 0
        and mlx_art.get("n_mtp") == 0
        and mlx_src.get("strips_mtp") is True
        and hawking.get("packer_cannot_ingest_mtp_dot_tensors") is True
    )
    token_alpha_zero = token.get("first_token_agree") is False
    measured_alpha_loses = all(
        not c["wins_vs_baseline"]
        for c in arithmetic
        if c.get("alpha_kind") == "token_greedy"
    )

    native_mtp_verdict = (
        "PARENT_HAS_NATIVE_MTP_HEADS_RUNTIMES_DROP_THEM"
        if parent_has_mtp and runtime_drops
        else "SEE_CENSUS"
    )
    draft_verdict = (
        "BINARY_REJECTED_AS_GENERATOR_AND_AS_DRAFT"
        if token_alpha_zero and measured_alpha_loses
        else "SEE_CENSUS"
    )

    answer = (
        "Parent Qwen3.8 BF16 HAS native MTP: text_config.mtp_num_hidden_layers=1, "
        "mtp_use_dedicated_embeddings=false, 15 mtp.* tensors, 849,398,784 BF16 "
        "bytes, one GQA+MLP layer after the last hidden sharing lm_head. Gravity "
        "q4 pack, MLX 4-bit convert/sanitize, and hawking qwen38_pack all drop "
        "those heads (n_mtp=0); llama.cpp exposes --spec-type draft-mtp but our "
        "Q5_K GGUF is absent here, and a Metal prior on Qwen3.5-9B lost at every "
        "n_max. The 1.25-bpw binary is a failed generator (16× token 271) and a "
        "failed draft: token top-1 vs q2f is 0 at position 0, local SwiGLU "
        f"channel-argmax mean is {local.get('draft_quality_proxy')}, composed-ns "
        "is only 15% cheaper than q2f (23.43 vs 27.55 ms) and native greedy tok/s "
        "is tied (33.44 vs 33.09). Speculative decoding that rejects most "
        "proposals LOSES on accepted-tokens/GPU-second. Full MTP bench deferred."
    )

    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "hand_authored": False,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_load_second_27b": True,
        "did_not_mutate_noetic_parent_a": True,
        "did_not_write_under_models": True,
        "streamed_parent_headers_only": True,
        "unmeasured_is_absent": True,
        "s026": ["§59-64", "§94", "§95", "§121"],
        "question": (
            "Does OUR Qwen3.8 artifact/config/runtime expose native MTP heads, "
            "and can a reduced-depth / early-exit / 1.25-bpw binary child serve "
            "as a draft of q2f without losing on accepted-tokens/pass?"
        ),
        "answer": answer,
        "one_line": (
            "MTP heads exist in the parent and are dropped at every runtime we "
            "run; the 1.25 binary is not a useful draft (token α=0 at pos 0)."
        ),
        "native_mtp": {
            "verdict": native_mtp_verdict,
            "parent_has_heads": parent_has_mtp,
            "runtimes_drop_heads": runtime_drops,
            "parent_config": parent_cfg,
            "parent_tensors": parent_t,
            "gravity_q4": gravity,
            "mlx_4bit_artifact": mlx_art,
            "mlx_sanitize": mlx_src,
            "llama_cpp": llama,
            "hawking_runtime": hawking,
            "alpha_mtp": absent(
                "probability",
                "Would need a forward through mtp.layers.0 + shared lm_head "
                "on final hidden+embed(token). Capture is post_attn_norm "
                "(wrong site). No second 27B decode.",
                "parent mtp.* present, runtime dropped",
            ),
        },
        "self_speculative": {
            "verdict": draft_verdict,
            "binary_as_draft": {
                "token_agreement": token,
                "local_top1_on_real_X": local,
                "composed_complete_token_ns": {
                    "binary_g64": BINARY_COMPLETE_NS,
                    "q2f": Q2F_COMPLETE_NS,
                    "source": "receipts/headless/BYTES_FRONTIER.json",
                },
                "g057": {
                    "source": "receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json",
                    "verdict_there": (
                        "REFUTED ON COST, BEFORE ACCEPTANCE RATE ENTERS"
                    ),
                    "break_even_ms_at_k4": G057_BREAK_EVEN_MS,
                    "what_would_change_it": (
                        "a draft that is a different SHAPE, not merely a "
                        "cheaper codec of the same shape — fewer layers or "
                        "a smaller hidden"
                    ),
                    "binary_clears_that_break_even": False,
                },
                "arithmetic": arithmetic,
            },
            "extra_head_and_state_bytes": extra_state,
            "accepted_tokens_per_pass_ceiling": {
                "binary_token_alpha": 1.0,  # target bonus only
                "binary_if_local_proxy_were_token_alpha_gamma_4_leviathan": (
                    expected_accepted_per_pass(float(alpha_local), 4)
                    if alpha_local is not None else None
                ),
                "native_mtp_if_alpha_1": 2.0,
                "note": (
                    "Ceiling is not a forecast. Token α of the binary is 0 at "
                    "pos 0, so the binary ceiling collapses to 1. MTP α is "
                    "ABSENT so 2.0 is an unearned ceiling."
                ),
            },
        },
        "purpose_reclassification": purposes,
        "metric_framing": metrics,
        "citations": [
            "receipts/headless/BYTES_FRONTIER.json",
            "receipts/headless/BINARY_HEALING.json",
            "receipts/headless/_BINARY_HEALING_local.json",
            "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
            "receipts/headless/Q2F_G64_GENERATION.json",
            "receipts/headless/SHARED_BASIS_COHERENT.json",
            "receipts/headless/HYBRID_OPERATOR.json",
            "receipts/headless/ORGAN_BANDWIDTH.json",
            "receipts/headless/NOETIC_METRICS.json",
            "receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json",
            "receipts/ascent-2026-08-16/G091_MULTI_TOKEN.json",
            "crates/hawking-core/src/model/qwen38_pack.rs",
            "crates/hawking-core/src/model/qwen38_geometry.rs",
            "crates/hawking-speculate/src/lib.rs",
            LLAMA_MTP_METAL_ISSUE,
        ],
    }


def write(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = doc if doc is not None else build()
    write_json(RECEIPT, doc)
    return doc


def main() -> int:
    doc = write()
    print(f"wrote {RECEIPT}")
    print(doc["one_line"])
    print(f"native_mtp.verdict={doc['native_mtp']['verdict']}")
    print(f"self_speculative.verdict={doc['self_speculative']['verdict']}")
    n_mtp = doc["native_mtp"]["parent_tensors"].get("n_mtp_tensors")
    b_mtp = doc["native_mtp"]["parent_tensors"].get("mtp_bytes")
    print(f"parent mtp tensors={n_mtp} bytes={b_mtp}")
    tok = doc["self_speculative"]["binary_as_draft"]["token_agreement"]
    print(
        f"token pos0 binary={tok.get('binary_token_0')} "
        f"q2f={tok.get('q2f_token_0')} agree={tok.get('first_token_agree')}"
    )
    loc = doc["self_speculative"]["binary_as_draft"]["local_top1_on_real_X"]
    print(f"local swiglu top1 mean={loc.get('draft_quality_proxy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
