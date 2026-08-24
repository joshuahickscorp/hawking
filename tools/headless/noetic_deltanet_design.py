#!/usr/bin/env python3
"""DeltaNet / recurrent state: can state replace static weight?

Qwen3.8 already carries a recurrent organ. This tool measures what that
organ does per token today, names every static tensor on the path against
the state it sits next to, and asks whether a cheaper STATE + transition
operator can take over work currently encoded in those weights.

It does not load the gravity artifact onto the GPU (two resident model
servers measured 3.986 tok/s against 33.47 with one). It does not spawn a
second 27B. GPU timestamps come from the sealed TOKEN_NS ledger; this run
recomputes the arithmetic and measures bytes, geometry, and source against
the live tree.

Write: receipts/headless/NOETIC_DELTANET_DESIGN.json
Run:   python3 tools/headless/noetic_deltanet_design.py
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "hawking.headless.noetic_deltanet_design.v1"

# Anchors. Do not re-derive. Native complete-wall and machine roof.
ANCHOR_TPS = 32.73
ANCHOR_MS_PER_TOKEN = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_PARAM_COUNT = 26_895_998_464
ANCHOR_ARTIFACT_BYTES = 14_297_933_604
ANCHOR_ARTIFACT_FILES = 756
ANCHOR_TENSORS = 755
ANCHOR_BPW = 4.252735126866492
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_GFLOP = 51.24
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_Q5K_TPS = 24.12
ANCHOR_TWO_SERVERS_TPS = 3.986
ANCHOR_ONE_SERVER_TPS = 33.47
# Historical TOKEN_NS figure in the obligation. Reconciled below, not re-timed.
HISTORICAL_DELTANET_SHARE = 0.106
# MLP distillation NO-GO (contract constraint; confirmed against the probe
# receipt if that file is reachable). I'=2560 vs q3, L31 hold rel-fro.
MLP_DISTILL_GAP = 0.4206
MLP_DISTILL_BYTE_RATIO = 0.724
NULL_COSINE = 0.898

ASCENT = "receipts/ascent-2026-08-16"
TOKEN_NS_REL = f"{ASCENT}/QWEN38_TOKEN_NS_LEDGER.json"
G024_REL = f"{ASCENT}/G024_QWEN38_TOKEN_NS.json"
G042_REL = f"{ASCENT}/G042_BPW_FAMILY.json"
G043_REL = f"{ASCENT}/G043_FLOP_FAMILY.json"
G060_REL = f"{ASCENT}/G060_LATENT_KV_VERDICT.json"
G061_REL = f"{ASCENT}/G061_JOINT_STATE_WEIGHT.json"
G089_REL = f"{ASCENT}/G089_LATENT_RUNTIME_SPAN.json"
G110_REL = f"{ASCENT}/G110_NONGEMV_CENSUS.json"
G035_REL = f"{ASCENT}/G035_CROSSLAYER_SHARE.json"
Q80_STATE_CODEC_REL = (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80/state-kv/"
    "QWEN80_DELTANET_STATE_CODEC_RECEIPT.json"
)
Q80_STATE_TRAFFIC_REL = (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80/state-kv/"
    "QWEN80_DELTANET_RECURRENT_STATE_TRAFFIC_RECEIPT.json"
)
N1ARCH_REL = ".lane-bootstrap/census/n1arch.md"
N15NEG_REL = ".lane-bootstrap/census/n15neg.md"
N16CLOS_REL = ".lane-bootstrap/census/n16clos.md"
ORGAN_REL = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
KERNEL_REL = "receipts/headless/NOETIC_KERNEL_CENSUS.json"
OP_REL = "receipts/headless/NOETIC_OPERATION_CENSUS.json"
INFO_REL = "receipts/headless/NOETIC_INFORMATION_ACCOUNTING.json"
MLP_PROBE_REL = "receipts/headless/NOETIC_MLP_DISTILL_PROBE.json"

DECODE_RS = "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SCHEDULE_RS = "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs"
GEOMETRY_RS = "crates/hawking-core/src/model/qwen38_geometry.rs"
LEDGER_RS = "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
Q38_METAL = "crates/hawking-core/shaders/qwen38_device_activations.metal"
Q80_METAL = "crates/hawking-core/shaders/qwen80_device_activations.metal"

ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
HQ30UQ4_MAGIC = b"HQ30UQ4\0"
Q4_GROUP = 64
Q4_BYTES_PER_GROUP = Q4_GROUP // 2 + 2  # 32 code + 2 fp16 scale


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_repo() -> Path:
    env = os.environ.get("HAWKING_REPO")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "Cargo.toml").exists() and (p / "tools" / "headless").is_dir():
            return p
    return Path.cwd()


REPO = find_repo()


def extra_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HAWKING_COPY"),
        os.environ.get("HAWKING_ROOT"),
        str(Path.home() / "Downloads" / "hawking-copy"),
        "/Users/scammermike/Downloads/hawking-copy",
        str(Path.home() / "Downloads" / "hawking"),
        "/Users/scammermike/.grok/noetic_archaeology",
        "/Users/scammermike/.claude-grok/worktrees/n16mlp-20260823-142304",
    ):
        if not raw:
            continue
        p = Path(raw)
        if p.exists() and p not in roots and p != REPO:
            roots.append(p)
    return roots


def git_head() -> str:
    r = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return (r.stdout or "").strip() or "UNKNOWN"


def git_show(rel: str) -> bytes | None:
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        capture_output=True,
    )
    if r.returncode == 0 and r.stdout:
        return r.stdout
    return None


def locate(rel: str) -> dict[str, Any]:
    """Resolve a receipt. Sparse checkout is not evidence of absence."""
    tried: list[str] = []
    on_disk = REPO / rel
    tried.append(f"disk:{on_disk}")
    if on_disk.is_file():
        return {
            "rel": rel,
            "found": True,
            "how": "disk",
            "path": str(on_disk),
            "bytes": on_disk.stat().st_size,
        }
    blob = git_show(rel)
    tried.append(f"git:HEAD:{rel}")
    if blob is not None:
        return {
            "rel": rel,
            "found": True,
            "how": "git",
            "path": f"HEAD:{rel}",
            "bytes": len(blob),
        }
    name = Path(rel).name
    for root in extra_roots():
        for cand in (
            root / rel,
            root / name,
            root / "receipts" / "headless" / name,
            root / ".lane-bootstrap" / "census" / name,
        ):
            tried.append(f"disk:{cand}")
            if cand.is_file():
                return {
                    "rel": rel,
                    "found": True,
                    "how": "copy",
                    "path": str(cand),
                    "bytes": cand.stat().st_size,
                }
    return {"rel": rel, "found": False, "how": None, "path": None, "tried": tried}


def load_bytes(rel: str) -> tuple[bytes | None, dict[str, Any]]:
    loc = locate(rel)
    if not loc["found"]:
        return None, loc
    if loc["how"] == "git":
        return git_show(rel), loc
    return Path(loc["path"]).read_bytes(), loc


def load_json(rel: str) -> tuple[Any, dict[str, Any]]:
    blob, loc = load_bytes(rel)
    if blob is None:
        return None, loc
    return json.loads(blob.decode("utf-8")), loc


def load_text(rel: str) -> tuple[str | None, dict[str, Any]]:
    blob, loc = load_bytes(rel)
    if blob is None:
        return None, loc
    return blob.decode("utf-8", errors="replace"), loc


def numbered(rel: str) -> tuple[list[str], dict[str, Any]]:
    text, loc = load_text(rel)
    if text is None:
        return [], loc
    return text.splitlines(), loc


def find_line(lines: list[str], pattern: str, start: int = 0) -> int | None:
    rx = re.compile(pattern)
    for i, line in enumerate(lines[start:], start=start + 1):
        if rx.search(line):
            return i
    return None


def find_all_lines(lines: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return [i for i, line in enumerate(lines, start=1) if rx.search(line)]


def snippet(lines: list[str], lineno: int | None, width: int = 160) -> str | None:
    if lineno is None or not (1 <= lineno <= len(lines)):
        return None
    return lines[lineno - 1].strip()[:width]


def q4_matrix_bytes(rows: int, cols: int) -> int:
    groups = (cols + Q4_GROUP - 1) // Q4_GROUP
    return rows * groups * Q4_BYTES_PER_GROUP


def fidelity(a: list[float], b: list[float]) -> dict[str, Any]:
    """Scale-aware. Cosine alone accepts 0.01·x and is not the gate."""
    if len(a) != len(b):
        raise ValueError(f"fidelity length {len(a)} vs {len(b)}")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    dot = sum(x * y for x, y in zip(a, b))
    denom = max(na * nb, 1e-30)
    cosine = dot / denom
    scale = nb / max(na, 1e-30)
    scale_match = min(scale, 1.0 / scale) if scale > 0 else 0.0
    rel_l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / max(na, 1e-30)
    return {
        "cosine": cosine,
        "relative_l2": rel_l2,
        "scale_ratio": scale,
        "scale_match": scale_match,
        "scale_aware": cosine * scale_match,
        "norm_true": na,
        "norm_pred": nb,
        "rejects_perfect_cosine_as_sufficient": True,
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def parse_const_usize(text: str, name: str) -> int | None:
    m = re.search(rf"pub const {name}: usize = ([0-9_]+);", text)
    if not m:
        return None
    return int(m.group(1).replace("_", ""))


def load_f32v2(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"f32v2 too small: {path}")
    n = struct.unpack_from("<Q", data, 0)[0]
    need = 8 + n * 4
    if len(data) != need:
        raise ValueError(f"f32v2 size {len(data)} != 8+4*{n} at {path}")
    return list(struct.unpack_from(f"<{n}f", data, 8))


def pairwise_scale_aware(rows: list[list[float]]) -> dict[str, Any]:
    n = len(rows)
    if n < 2:
        return {"status": "NULL", "reason": f"n={n} < 2"}
    vals: list[float] = []
    cosines: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            f = fidelity(rows[i], rows[j])
            vals.append(f["scale_aware"])
            cosines.append(f["cosine"])
    scaled = [0.01 * x for x in rows[0]]
    trap = fidelity(rows[0], scaled)
    return {
        "status": "MEASURED",
        "n": n,
        "n_pairs": len(vals),
        "mean_pairwise_scale_aware": mean(vals),
        "min_pairwise_scale_aware": min(vals),
        "mean_pairwise_cosine": mean(cosines),
        "scale_trap_0p01": trap,
        "note": "cosine is reported as a contrast; the gate is scale_aware",
    }


def git_grep_names(patterns: list[str], globs: list[str]) -> dict[str, Any]:
    """Name-only search over HEAD. Sparse checkout does not hide tracked files."""
    cmd = ["git", "-C", str(REPO), "grep", "-l", "-i", "-I"]
    for p in patterns:
        cmd.extend(["-e", p])
    cmd.append("--")
    cmd.extend(globs)
    r = subprocess.run(cmd, capture_output=True, text=True)
    names = [ln for ln in (r.stdout or "").splitlines() if ln]
    return {
        "patterns": patterns,
        "globs": globs,
        "exit": r.returncode,
        "n_files": len(names),
        "files_head": names[:40],
        "files_truncated": max(0, len(names) - 40),
    }


# ---------------------------------------------------------------------------
# geometry from source (live), then cross-check against receipts
# ---------------------------------------------------------------------------

def measure_geometry() -> dict[str, Any]:
    text, loc = load_text(GEOMETRY_RS)
    watched: list[str] = []
    if text is None:
        return {"status": "NULL", "reason": "geometry.rs missing", "locate": loc}
    names = {
        "layers": "QWEN38_LAYERS",
        "dn_layers": "QWEN38_DELTANET_LAYERS",
        "gqa_layers": "QWEN38_GQA_LAYERS",
        "hidden": "QWEN38_HIDDEN",
        "intermediate": "QWEN38_INTERMEDIATE",
        "vocab": "QWEN38_VOCAB",
        "key_heads": "QWEN38_LINEAR_KEY_HEADS",
        "value_heads": "QWEN38_LINEAR_VALUE_HEADS",
        "vpk": "QWEN38_LINEAR_VALUES_PER_KEY",
        "key_dim": "QWEN38_LINEAR_KEY_HEAD_DIM",
        "value_dim": "QWEN38_LINEAR_VALUE_HEAD_DIM",
        "conv_k": "QWEN38_LINEAR_CONV_KERNEL",
        "gqa_heads": "QWEN38_GQA_HEADS",
        "gqa_kv_heads": "QWEN38_GQA_KV_HEADS",
        "gqa_head_dim": "QWEN38_GQA_HEAD_DIM",
    }
    g: dict[str, Any] = {k: parse_const_usize(text, v) for k, v in names.items()}
    if any(v is None for v in g.values()):
        watched.append(f"failed to parse some geometry constants: {g}")
    kh, vh, vpk = g["key_heads"], g["value_heads"], g["vpk"]
    kd, vd, ck = g["key_dim"], g["value_dim"], g["conv_k"]
    value_rows = vpk * vd
    qkvz_rows = kh * (kd * 2 + value_rows * 2)
    ba_rows = kh * (vpk * 2)
    conv_channels = kh * kd * 2 + vh * vd
    rec_elems_layer = vh * kd * vd
    conv_elems_layer = conv_channels * (ck - 1)
    dn = g["dn_layers"]
    hidden = g["hidden"]
    rec_elems = dn * rec_elems_layer
    conv_elems = dn * conv_elems_layer
    rec_resident = rec_elems * 4
    conv_resident = conv_elems * 4
    qkvz_q4 = q4_matrix_bytes(qkvz_rows, hidden)
    ba_q4 = q4_matrix_bytes(ba_rows, hidden)
    out_q4 = q4_matrix_bytes(hidden, vh * vd)  # 5120 x 6144
    linear_q4 = dn * (qkvz_q4 + ba_q4 + out_q4)
    qkvz_mac = dn * qkvz_rows * hidden * 2
    ba_mac = dn * ba_rows * hidden * 2
    out_mac = dn * hidden * (vh * vd) * 2
    gemv_mac = qkvz_mac + ba_mac + out_mac
    state_update_flops = rec_elems * 3.0  # G043: decay, k*delta, query product
    gqa_kv_per_pos = (
        g["gqa_layers"] * 2 * g["gqa_kv_heads"] * g["gqa_head_dim"] * 4
    )
    return {
        "status": "MEASURED",
        "origin": f"{GEOMETRY_RS} parsed this run",
        "locate": loc,
        "constants": g,
        "qkvz_rows": qkvz_rows,
        "ba_rows": ba_rows,
        "out_rows": hidden,
        "out_cols": vh * vd,
        "conv_channels": conv_channels,
        "rec_elems_per_layer": rec_elems_layer,
        "conv_elems_per_layer": conv_elems_layer,
        "rec_elems_total": rec_elems,
        "conv_elems_total": conv_elems,
        "rec_resident_bytes": rec_resident,
        "conv_resident_bytes": conv_resident,
        "rec_rw_bytes": rec_resident * 2,
        "conv_rw_bytes": conv_resident * 2,
        "q4_qkvz_per_layer": qkvz_q4,
        "q4_ba_per_layer": ba_q4,
        "q4_out_per_layer": out_q4,
        "q4_linear_attn_payload": linear_q4,
        "gemv_mac_flops_per_token": gemv_mac,
        "qkvz_mac_flops_per_token": qkvz_mac,
        "ba_mac_flops_per_token": ba_mac,
        "out_mac_flops_per_token": out_mac,
        "state_update_flops_per_token": state_update_flops,
        "gqa_kv_bytes_per_position": gqa_kv_per_pos,
        "watched": watched,
    }


def measure_schedule() -> dict[str, Any]:
    text, loc = load_text(SCHEDULE_RS)
    if text is None:
        return {"status": "NULL", "reason": "schedule.rs missing", "locate": loc}
    prefix = parse_const_usize(text, "QWEN38_MIXER_PREFIX_DISPATCHES")
    mlp = parse_const_usize(text, "QWEN38_DENSE_MLP_SUFFIX_DISPATCHES")
    lines = text.splitlines()
    block = []
    capturing = False
    for line in lines:
        if "QWEN38_DELTANET_MIXER_PREFIX_KERNELS" in line:
            capturing = True
        if capturing:
            block.append(line.strip())
            if line.strip().endswith("];"):
                break
    names = re.findall(r'"([^"]+)"', "\n".join(block))
    return {
        "status": "MEASURED",
        "origin": f"{SCHEDULE_RS} parsed this run",
        "locate": loc,
        "mixer_prefix_dispatches": prefix,
        "mlp_suffix_dispatches": mlp,
        "full_layer_dispatches": (prefix or 0) + (mlp or 0),
        "dn_mixer_prefix_kernels": names,
        "dn_mixer_prefix_count": len(names),
        "formula_total_dispatches": 1 + 64 * ((prefix or 0) + (mlp or 0)) + 3,
    }


def measure_source_sites() -> dict[str, Any]:
    decode, dloc = numbered(DECODE_RS)
    metal, mloc = numbered(Q38_METAL)
    q80, qloc = numbered(Q80_METAL)
    ledger, lloc = numbered(LEDGER_RS)
    sites: dict[str, Any] = {}

    def pin(
        key: str,
        lines: list[str],
        pat: str,
        rel: str,
        loc: dict[str, Any],
        start: int = 0,
    ) -> None:
        ln = find_line(lines, pat, start=start)
        sites[key] = {
            "file": rel,
            "line": ln,
            "pattern": pat,
            "text": snippet(lines, ln),
            "locate": loc["how"],
        }

    pin("encode_deltanet", decode, r"fn encode_deltanet\(", DECODE_RS, dloc)
    dn0 = (sites["encode_deltanet"]["line"] or 1) - 1
    pin("in_proj_qkvz_bind", decode, r'linear_attn\.in_proj_qkvz\.weight"', DECODE_RS, dloc, start=dn0)
    pin("in_proj_ba_bind", decode, r'linear_attn\.in_proj_ba\.weight"', DECODE_RS, dloc, start=dn0)
    pin("out_proj_bind", decode, r'linear_attn\.out_proj\.weight"', DECODE_RS, dloc, start=dn0)
    pin("conv_weight_bind", decode, r'linear_attn\.conv1d\.weight"', DECODE_RS, dloc, start=dn0)
    pin("a_log_bind", decode, r'linear_attn\.A_log"', DECODE_RS, dloc, start=dn0)
    pin("dt_bias_bind", decode, r'linear_attn\.dt_bias"', DECODE_RS, dloc, start=dn0)
    pin("norm_bind", decode, r'linear_attn\.norm\.weight"', DECODE_RS, dloc, start=dn0)
    pin("encode_gated_delta", decode, r"fn encode_gated_delta\(", DECODE_RS, dloc)
    pin("rec_state_set_buffer", decode, r"set_buffer\(0, Some\(&self\.workspace\.rec_state\)", DECODE_RS, dloc)
    pin("attach_zero_rec", decode, r"zero_buffer\(&workspace\.rec_state\)", DECODE_RS, dloc)
    pin("reset_fn", decode, r"pub fn reset\(&mut self\)", DECODE_RS, dloc)
    pin("reset_zero_rec", decode, r"zero_buffer\(&self\.workspace\.rec_state\)", DECODE_RS, dloc)
    pin("step_fn", decode, r"pub fn step\(&mut self, token: u32\)", DECODE_RS, dloc)
    step0 = (sites["step_fn"]["line"] or 1) - 1
    pin("step_position_inc", decode, r"self\.position = self\.position\.saturating_add\(1\)", DECODE_RS, dloc, start=step0)
    pin("stream_probe", decode, r"pub fn measure_f32_stream\(", DECODE_RS, dloc)
    pin("rec_alloc", decode, r"f32b\(48 \* layout\.recurrent_state_elements", DECODE_RS, dloc)
    pin("causal_conv", metal, r"inline float qwen38_causal_conv_update_f32", Q38_METAL, mloc)
    pin("conv_state_shift", metal, r"conv_state\[state_base \+ tap\] = conv_state", Q38_METAL, mloc)
    pin("gated_delta_vi", metal, r"kernel void qwen38_gated_delta_decode_vi\(", Q38_METAL, mloc)
    pin("gated_delta_simd", metal, r"kernel void qwen38_gated_delta_decode_vi_simd\(", Q38_METAL, mloc)
    pin("state_decay_mul", metal, r"const float decayed = state\[index\] \* d", Q38_METAL, mloc)
    pin("state_rank1_update", metal, r"state\[index\] \+= key\[key_base \+ ki\] \* delta", Q38_METAL, mloc)
    pin("state_query_readout", metal, r"state\[index\] \* query\[key_base \+ ki\]", Q38_METAL, mloc)
    pin("ba_to_decay", q80, r"kernel void qwen80_ba_to_decay_beta_f32", Q80_METAL, qloc)
    pin("decay_from_a_log", q80, r"const float g = -exp\(a_log\[value_head\]\)", Q80_METAL, qloc)
    pin("gated_rmsnorm", q80, r"kernel void qwen80_deltanet_gated_rmsnorm_tg", Q80_METAL, qloc)
    pin("theoretical_state_bytes", ledger, r"pub fn theoretical_state_bytes", LEDGER_RS, lloc)
    pin("seal_deltanet_arith", ledger, r"let deltanet = \(rearrange as f64 - kv_conv\)", LEDGER_RS, lloc)
    pin("kv_rec_cap", ledger, r"let kv_rec = \(rec_stream as f64\)\.min\(gated as f64\)", LEDGER_RS, lloc)

    # step() body must not zero rec_state. Measure, don't assume.
    step_ln = sites["step_fn"]["line"]
    reset_ln = sites["reset_fn"]["line"]
    step_zeros = []
    if step_ln and reset_ln:
        end = reset_ln if reset_ln > step_ln else min(step_ln + 40, len(decode) + 1)
        # step is near the end of the file; take 30 lines after the def
        body = decode[step_ln - 1 : step_ln + 29]
        step_zeros = [step_ln + i for i, ln in enumerate(body) if "zero_buffer" in ln and "rec_state" in ln]
    sites["step_zeros_rec_state"] = {
        "file": DECODE_RS,
        "line": None,
        "hits": step_zeros,
        "text": "no zero_buffer(rec_state) in step() body" if not step_zeros else "FOUND",
        "locate": dloc["how"],
    }
    return {
        "status": "MEASURED",
        "sites": sites,
        "decode_lines": len(decode),
        "metal_lines": len(metal),
    }


# ---------------------------------------------------------------------------
# artifact walk — live bytes, not a paraphrase of the organ census
# ---------------------------------------------------------------------------

def measure_artifact() -> dict[str, Any]:
    if not ARTIFACT.is_dir():
        return {
            "status": "NULL",
            "reason": f"artifact missing: {ARTIFACT}",
            "path": str(ARTIFACT),
        }
    man_path = ARTIFACT / "manifest.json"
    man = json.loads(man_path.read_text())
    by_suffix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    payload = 0
    files = 0
    q4_n = 0
    f32_n = 0
    q4_magic_ok = 0
    q4_magic_bad = 0
    for t in man["tensors"]:
        name = t["name"]
        if ".linear_attn." not in name:
            continue
        suf = name.split("linear_attn.", 1)[1]
        rec = {
            "name": name,
            "kind": t["kind"],
            "bytes": t["bytes"],
            "elements": t["elements"],
            "shape": t["shape"],
            "artifact": t["artifact"],
        }
        by_suffix[suf].append(rec)
        payload += t["bytes"]
        files += 1
        if t["kind"] == "q4":
            q4_n += 1
            p = ARTIFACT / "tensors" / t["artifact"]
            if p.is_file():
                magic = p.read_bytes()[:8]
                if magic == HQ30UQ4_MAGIC:
                    q4_magic_ok += 1
                else:
                    q4_magic_bad += 1
        else:
            f32_n += 1
    summary = {}
    for suf, rows in sorted(by_suffix.items()):
        summary[suf] = {
            "n": len(rows),
            "kind": rows[0]["kind"],
            "shape": rows[0]["shape"],
            "bytes_one": rows[0]["bytes"],
            "elements_one": rows[0]["elements"],
            "bytes_all": sum(r["bytes"] for r in rows),
            "elements_all": sum(r["elements"] for r in rows),
        }
    # Live A_log / dt_bias contents (tiny f32; this is a real load, not a cite).
    a_logs: list[list[float]] = []
    dt_bias: list[list[float]] = []
    a_load_err = None
    try:
        for row in by_suffix["A_log"]:
            a_logs.append(load_f32v2(ARTIFACT / "tensors" / row["artifact"]))
        for row in by_suffix["dt_bias"]:
            dt_bias.append(load_f32v2(ARTIFACT / "tensors" / row["artifact"]))
    except (KeyError, ValueError, OSError) as e:
        a_load_err = str(e)
    a_stats = pairwise_scale_aware(a_logs) if a_logs and a_load_err is None else {
        "status": "NULL",
        "reason": a_load_err or "no A_log",
    }
    dt_stats = pairwise_scale_aware(dt_bias) if dt_bias and a_load_err is None else {
        "status": "NULL",
        "reason": a_load_err or "no dt_bias",
    }
    a_flat = [x for row in a_logs for x in row]
    return {
        "status": "MEASURED",
        "origin": f"walk {ARTIFACT}/manifest.json + tensor files this run",
        "path": str(ARTIFACT),
        "present": True,
        "manifest_schema": man.get("schema"),
        "manifest_tensor_count": man.get("tensor_count"),
        "manifest_source_weight_elements": man.get("source_weight_elements"),
        "linear_attn_files": files,
        "linear_attn_on_disk_bytes": payload,
        "q4_tensors": q4_n,
        "f32_tensors": f32_n,
        "q4_magic_ok": q4_magic_ok,
        "q4_magic_bad": q4_magic_bad,
        "by_suffix": summary,
        "a_log_loaded": {
            "n_vectors": len(a_logs),
            "len": len(a_logs[0]) if a_logs else 0,
            "min": min(a_flat) if a_flat else None,
            "max": max(a_flat) if a_flat else None,
            "mean": mean(a_flat) if a_flat else None,
            "pairwise": a_stats,
        },
        "dt_bias_pairwise": dt_stats,
        "header_note": (
            "manifest bytes include the 40 B HQ30UQ4 header on every Q4 file "
            "and the 8 B count header on every f32v2 file. TOKEN_NS "
            "linear_attn_bytes is payload geometry without those headers."
        ),
    }


def measure_token_ns(geo: dict[str, Any]) -> dict[str, Any]:
    ledger, loc = load_json(TOKEN_NS_REL)
    if ledger is None:
        return {"status": "NULL", "reason": "TOKEN_NS ledger missing", "locate": loc}
    comps = {c["component"]: c for c in ledger["components"]}
    iso = {r["name"]: r for r in ledger["isolated"]}
    dn = comps["deltanet"]
    kv = comps["kv_state"]
    wall = ledger["median_wall_ns"]
    share = dn["ns_per_token"] / wall
    organ_ns = iso["dn_full_probe"]["median_gpu_ns"] + dn["ns_per_token"]
    organ_share = organ_ns / wall
    # Recompute seal_components deltanet arithmetic from isolated medians.
    rearrange = iso["rearrange_48"]["median_gpu_ns"]
    ba = iso["ba_to_decay_48"]["median_gpu_ns"]
    gated = iso["gated_delta_48"]["median_gpu_ns"]
    gated_n = iso["gated_rmsnorm_48"]["median_gpu_ns"]
    mixer_res = iso["mixer_residual_64"]["median_gpu_ns"]
    rec_stream = iso["stream_rec_state"]["median_gpu_ns"]
    conv_stream = iso["stream_conv_state"]["median_gpu_ns"]
    dn_full = iso["dn_full_probe"]["median_gpu_ns"]
    probe = next(p for p in ledger["probes"] if p["class"] == "dn")
    dn_fma = (1.0 - probe["addr_frac_of_full"] - probe["decode_minus_addr_frac"]) * iso["dn_gemvs"]["median_gpu_ns"]
    kv_rec = min(rec_stream, gated)
    kv_conv = min(conv_stream, rearrange)
    recomputed = (
        max(rearrange - kv_conv, 0.0)
        + ba
        + max(gated - kv_rec, 0.0)
        + gated_n
        + dn_fma
        + mixer_res * (48.0 / 64.0)
    )
    rec_res = geo["rec_resident_bytes"]
    conv_res = geo["conv_resident_bytes"]
    gqa_write = ledger["state_bytes"]["gqa_kv_write_bytes"]
    gqa_read = ledger["state_bytes"]["gqa_kv_read_bytes_at_pos"]
    kv_read_expect = rec_res + conv_res + gqa_read
    kv_write_expect = rec_res + conv_res + gqa_write
    native_share_if_same_ns = dn["ns_per_token"] / (ANCHOR_MS_PER_TOKEN * 1e6)
    return {
        "status": "MEASURED",
        "origin": f"CITED {TOKEN_NS_REL} (GPU timestamps not re-run); arithmetic recomputed this run",
        "locate": loc,
        "schema": ledger["schema"],
        "measurement_label": ledger["measurement_label"],
        "gpu_timestamp_authority": ledger["gpu_timestamp_authority"],
        "commit_of_ledger": ledger.get("commit"),
        "median_wall_ns": wall,
        "median_gpu_ns": ledger["median_gpu_ns"],
        "weight_bytes": ledger["weight_bytes"],
        "state_bytes": ledger["state_bytes"],
        "dispatches": ledger["dispatches"],
        "component_deltanet": {
            "ns_per_token": dn["ns_per_token"],
            "pct_of_token_wall": dn["pct_of_token_wall"],
            "share": share,
            "bytes_read": dn["bytes_read"],
            "bytes_written": dn["bytes_written"],
            "dispatches": dn["dispatches"],
            "method": dn["method"],
            "effective_gb_s": dn["effective_gb_s"],
            "resource_class": dn["resource_class"],
        },
        "component_kv_state": {
            "ns_per_token": kv["ns_per_token"],
            "pct_of_token_wall": kv["pct_of_token_wall"],
            "bytes_read": kv["bytes_read"],
            "bytes_written": kv["bytes_written"],
            "dispatches": kv["dispatches"],
            "method": kv["method"],
            "effective_gb_s": kv["effective_gb_s"],
        },
        "isolated": {
            "dn_full_probe_ns": dn_full,
            "dn_addr_probe_ns": iso["dn_addr_probe"]["median_gpu_ns"],
            "dn_decode_probe_ns": iso["dn_decode_probe"]["median_gpu_ns"],
            "dn_gemvs_ns": iso["dn_gemvs"]["median_gpu_ns"],
            "dn_gemvs_dispatches": iso["dn_gemvs"]["dispatches"],
            "rearrange_48_ns": rearrange,
            "ba_to_decay_48_ns": ba,
            "gated_delta_48_ns": gated,
            "gated_rmsnorm_48_ns": gated_n,
            "stream_rec_state_ns": rec_stream,
            "stream_conv_state_ns": conv_stream,
            "mixer_residual_64_ns": mixer_res,
        },
        "dn_probe": probe,
        "recomputed_deltanet_ns": recomputed,
        "recomputed_matches_component": abs(recomputed - dn["ns_per_token"]) < 1.0,
        "recomputed_abs_delta_ns": recomputed - dn["ns_per_token"],
        "organ_aggregation": {
            "formula": "isolated_dn_full_probe + TOKEN_NS component deltanet",
            "ns_per_token": organ_ns,
            "share_of_token_ns_wall": organ_share,
            "note": (
                "This is the organ-census aggregation (GEMV + non-GEMV). It is "
                "NOT the TOKEN_NS component named deltanet. Mixing them is how "
                "10.6% and 26.3% look like a contradiction."
            ),
        },
        "historical_10p6_reconciliation": {
            "historical_share_quoted": HISTORICAL_DELTANET_SHARE,
            "token_ns_component_share": share,
            "matches_historical": abs(share - HISTORICAL_DELTANET_SHARE) < 0.001,
            "organ_census_share": organ_share,
            "why_they_differ": (
                "seal_components() puts DN Q4 GEMVs in weight_addressing / "
                "weight_decode_reconstruction (addr_probe 90.51% of isolated DN "
                "GEMV) and attributes rec/conv/GQA streams to kv_state. The "
                "component named deltanet is rearrange+ba+gated_delta+"
                "gated_rmsnorm+dn FMA remainder+48/64 mixer residual, MINUS "
                "those streams. 10.6% is that component. 26.3% adds the GEMV "
                "back. Same ledger, two sums."
            ),
            "do_not_use_native_30p606_as_this_denominator": (
                f"TOKEN_NS wall is {wall} ns (DIRTY_ENGINEERING, "
                f"{ledger.get('commit')}). Native complete-wall "
                f"{ANCHOR_MS_PER_TOKEN} ms is a later gravity run. "
                f"component_ns / native_wall would be {native_share_if_same_ns:.6f}; "
                "that mixed denominator is not a measurement and is not used."
            ),
        },
        "kv_state_byte_identity": {
            "bytes_read_ledger": kv["bytes_read"],
            "bytes_read_from_geometry": kv_read_expect,
            "read_match": kv["bytes_read"] == kv_read_expect,
            "bytes_written_ledger": kv["bytes_written"],
            "bytes_written_from_geometry": kv_write_expect,
            "write_match": kv["bytes_written"] == kv_write_expect,
            "formula": "rec_resident + conv_resident + gqa_{read,write}",
        },
        "stream_as_fraction_of_gated_delta": rec_stream / gated if gated else None,
        "dn_addr_frac_of_full": probe["addr_frac_of_full"],
    }


def g042_state_omission(geo: dict[str, Any]) -> dict[str, Any]:
    g042, loc = load_json(G042_REL)
    if g042 is None:
        return {"status": "NULL", "reason": "G042 missing", "locate": loc}
    definition = g042["definitions"]["STATE_BPW_EQUIVALENT"]
    uni = next(c for c in g042["candidates"] if c["candidate"] == "uniform-q4-v1")
    stated = uni["STATE_BPW_EQUIVALENT"]
    rec_bpw = geo["rec_resident_bytes"] * 8 / ANCHOR_PARAM_COUNT
    conv_bpw = geo["conv_resident_bytes"] * 8 / ANCHOR_PARAM_COUNT
    gqa_128 = stated["128"]
    return {
        "status": "MEASURED",
        "origin": f"CITED {G042_REL}; rec/conv BPW computed from geometry this run",
        "locate": loc,
        "definition_quotes_only_gqa": True,
        "definition": definition,
        "uniform_q4_v1_stated_state_bpw": stated,
        "gqa_ctx128_bpw": gqa_128,
        "deltanet_rec_resident_bpw": rec_bpw,
        "deltanet_conv_resident_bpw": conv_bpw,
        "deltanet_state_bpw_omitted": rec_bpw + conv_bpw,
        "rec_over_gqa_at_ctx128": (rec_bpw + conv_bpw) / gqa_128 if gqa_128 else None,
        "generated_bpw_equivalent": uni["GENERATED_BPW_EQUIVALENT"],
        "shared_bpw": uni["SHARED_BPW"],
        "omission": (
            "G042 STATE_BPW_EQUIVALENT charges 16 GQA layers × 131072 B/position "
            "and does not charge DeltaNet rec_state 150994944 B or conv_state "
            "5898240 B, both context-independent. At ctx=128 the omitted DN "
            "state is ~9× the GQA term the receipt does report."
        ),
    }


def measure_mlp_distill_anchor() -> dict[str, Any]:
    probe, loc = load_json(MLP_PROBE_REL)
    if probe is None:
        return {
            "status": "CITED_CONTRACT_NOT_RECONFIRMED_IN_THIS_TREE",
            "locate": loc,
            "decision": "NO-GO",
            "gap": MLP_DISTILL_GAP,
            "byte_ratio": MLP_DISTILL_BYTE_RATIO,
            "reason": (
                "NOETIC_MLP_DISTILL_PROBE.json is not in this checkout or HEAD. "
                "The obligation forbids re-deriving the probe. Numbers are the "
                "contract anchors: +0.4206 held-out gap vs q3 at 72% of its "
                "active bytes."
            ),
        }
    v = probe["verdict"]
    # Confirm the contract numbers against the receipt.
    gap = v["deciding_number"]
    # I'=2560 is the headline width; its fused-active byte ratio is 0.724.
    ratio = None
    for c in v.get("per_candidate") or []:
        if c.get("width") == 2560 or c.get("I_prime") == 2560:
            ratio = c.get("byte_ratio_vs_q3_fused_active")
            break
    if ratio is None:
        # walk layers_out
        try:
            ratio = probe["layers_out"][0]["widths"]["2560"]["byte_ratio_vs_q3_fused_active"]
        except (KeyError, IndexError, TypeError):
            ratio = None
    return {
        "status": "MEASURED",
        "origin": loc["path"],
        "locate": loc,
        "decision": v["decision"],
        "deciding_number": gap,
        "matches_contract_0p4206": abs(gap - 0.4206) < 5e-4,
        "byte_ratio_I2560": ratio,
        "matches_contract_0p72": (
            abs(ratio - MLP_DISTILL_BYTE_RATIO) < 5e-3 if ratio is not None else None
        ),
        "reopen_condition": v.get("reopen_condition"),
    }


# ---------------------------------------------------------------------------
# prior science search (live)
# ---------------------------------------------------------------------------

def prior_science_search() -> dict[str, Any]:
    watched: list[str] = []
    grep = git_grep_names(
        [
            "recurrent_state",
            "joint state",
            "STATE_BPW",
            "gated_delta",
            "deltanet",
            "state_transition",
            "G061",
            "G060",
            "G089",
        ],
        [
            "receipts",
            "tools/headless",
            "tools/gravity_joint_state_weight.py",
            "tools/foundry",
            ".lane-bootstrap",
        ],
    )
    n1, n1loc = load_text(N1ARCH_REL)
    n15, n15loc = load_text(N15NEG_REL)
    n16, n16loc = load_text(N16CLOS_REL)
    arch_idx = Path("/Users/scammermike/.grok/noetic_archaeology/NOETIC_ARCHAEOLOGY_INDEX.json")
    arch = None
    arch_how = None
    if arch_idx.is_file():
        arch = json.loads(arch_idx.read_text())
        arch_how = str(arch_idx)
    else:
        watched.append("archaeology index not at ~/.grok/noetic_archaeology")

    def n1_blob() -> str:
        return n1 or ""

    mechanisms_wanted = [
        "G071",
        "G061",
        "G060",
        "G089",
        "G042",
        "G035",
        "G064",
        "G034",
        "StateTransition",
    ]
    n1_hits = {}
    blob = n1_blob()
    for key in mechanisms_wanted:
        n1_hits[key] = blob.find(key) >= 0 if blob else False
    if n1 is None:
        watched.append("n1arch.md not in this sparse checkout; extra_roots used")

    kcen, kloc = load_json(KERNEL_REL)
    rec_fam = None
    if kcen:
        rec_fam = next(
            (f for f in kcen.get("families", []) if f.get("id") == "recurrent_state_operator"),
            None,
        )
    g061, g061loc = load_json(G061_REL)
    g060, g060loc = load_json(G060_REL)
    g089, g089loc = load_json(G089_REL)
    g043, g043loc = load_json(G043_REL)
    g035, g035loc = load_json(G035_REL)
    codec, codec_loc = load_json(Q80_STATE_CODEC_REL)
    traffic, traffic_loc = load_json(Q80_STATE_TRAFFIC_REL)

    g061_summary = None
    if g061:
        g061_summary = {
            "obligation": g061.get("obligation"),
            "organ": "GQA K at layer 63 (NOT DeltaNet rec_state)",
            "joint_ties_with_independent": g061.get("joint_ties_with_independent"),
            "tokens": g061.get("tokens"),
            "layer": g061.get("layer"),
            "aggregate_rank64": next(
                (r for r in g061.get("aggregate", []) if r["rank"] == 64), None
            ),
            "kv_effective_rank_99pct_mean_of_256": (
                g061.get("what_it_means_for_G060") or {}
            ).get("kv_effective_rank_99pct_mean_of_256"),
            "limitation": g061.get("limitation"),
            "locate": g061loc,
        }
    g060_summary = None
    if g060:
        g060_summary = {
            "obligation": g060.get("obligation"),
            "organ": "GQA attention path (NOT DeltaNet)",
            "ps_per_element_ratio": g060["the_attention_path_in_the_currency_that_binds"][
                "ps_per_element_ratio"
            ],
            "verdict_corrected": g060.get("verdict_corrected"),
            "locate": g060loc,
        }
    g089_summary = None
    if g089:
        g089_summary = {
            "obligation": g089.get("obligation"),
            "verdict_keys": list((g089.get("verdict") or {}).keys()),
            "out_of_distribution": (g089.get("verdict") or {}).get("out_of_distribution_it_does_not"),
            "locate": g089loc,
        }
    g043_flops = None
    if g043:
        g043_flops = {
            "state_update_basis": g043.get("state_update_basis"),
            "STATE_UPDATE_FLOPS": (g043.get("candidates") or [{}])[0].get("STATE_UPDATE_FLOPS")
            if isinstance(g043.get("candidates"), list)
            else None,
            "locate": g043loc,
        }
        if g043_flops["STATE_UPDATE_FLOPS"] is None:
            # candidates may be a dict keyed by name
            cands = g043.get("candidates")
            if isinstance(cands, dict):
                first = next(iter(cands.values()))
                if isinstance(first, dict):
                    g043_flops["STATE_UPDATE_FLOPS"] = first.get("STATE_UPDATE_FLOPS")
            # brute: walk
            if g043_flops["STATE_UPDATE_FLOPS"] is None:
                raw = json.dumps(g043)
                m = re.search(r'"STATE_UPDATE_FLOPS":\s*([0-9.]+)', raw)
                if m:
                    g043_flops["STATE_UPDATE_FLOPS"] = float(m.group(1))

    codec_summary = None
    if codec:
        results = []
        for row in codec.get("codec_results") or []:
            rec = row.get("reconstruction") or {}
            results.append(
                {
                    "codec": row.get("codec"),
                    "bits": row.get("bits"),
                    "cosine": rec.get("cosine"),
                    "relative_l2": rec.get("relative_l2"),
                    "physical_bytes": (row.get("physical_artifact") or {}).get("bytes"),
                }
            )
        codec_summary = {
            "model": codec.get("model"),
            "claim_boundary": codec.get("claim_boundary"),
            "results": results,
            "locate": codec_loc,
            "not_qwen38": codec.get("model") != "qwen38",
            "not_real_tokens": (codec.get("claim_boundary") or {}).get(
                "deterministic_component_vectors_are_not_prompts_or_tokens"
            ),
        }

    n16_state_transition = None
    if n16 and "StateTransition" in n16:
        n16_state_transition = {
            "present": True,
            "excerpt": "StateTransition schema-change; none in NR (n16clos)",
            "locate": n16loc,
        }

    return {
        "status": "MEASURED",
        "git_grep": grep,
        "census": {
            "n1arch": n1loc,
            "n15neg": n15loc,
            "n16clos": n16loc,
            "n1_hits": n1_hits,
        },
        "archaeology_index": arch_how,
        "kernel_census_recurrent_state_operator": rec_fam,
        "kernel_census_locate": kloc,
        "G061_joint_state_weight": g061_summary,
        "G060_latent_kv": g060_summary,
        "G089_latent_runtime": g089_summary,
        "G043_state_update_flops": g043_flops,
        "G035_shared_beats_independent": (
            None
            if not g035
            else next(
                (
                    p.get("shared_beats_independent")
                    for p in (g035.get("pairs") or [])
                    if "shared_beats_independent" in p
                ),
                g035.get("corrected_verdict"),
            )
        ),
        "G035_corrected_verdict": (g035 or {}).get("corrected_verdict") if g035 else None,
        "G035_locate": g035loc,
        "Q80_deltanet_state_codec": codec_summary,
        "Q80_traffic_locate": traffic_loc,
        "n16_state_transition": n16_state_transition,
        "traffic_present": bool(traffic),
        "watched": watched,
        "reading": [
            "A recurrent_state_operator already EXISTS and is DISPATCHED on the Qwen3.8 token (qwen38_gated_delta_decode_vi_simd).",
            "G061 is a GQA-K subspace study at L63, not a DeltaNet rec_state study. Archaeology bundled it with G060 as 'joint state+weight factorisation'. Do not treat that row as a DeltaNet result.",
            "G060 latent KV is the GQA attention path (253× per-element vs GEMV). Different organ.",
            "G089: compiling d_model away holds in-distribution and fails out of distribution. That is the cheap-encoder generalisation bound.",
            "G042 STATE_BPW charges GQA KV only; DN rec_state is omitted (measured in g042_state_omission).",
            "Q80 state codec compresses rec_state (fp16/q8/q4) on deterministic component vectors, on Q80, with a claim_boundary that forbids TPS and capability claims. It does not replace static in_proj.",
            "n16clos: StateTransition is a schema-change family with no NR field. Generated-state cache is one of the five MISSED hiding scenarios.",
            "G035 shared_beats_independent=false. Cross-layer sharing of the static maps around DeltaNet is closed.",
            "223 tensor-operator rows <0.5 local BPW, healthy=true: 0 (G1/G034). A cheap in_proj is the same class of claim.",
            "MLP function distillation NO-GO: +0.4206 held-out gap vs q3 at 72% of its active bytes. The surviving 'distill the function' avenue on MLP is closed; it is not a licence to distill in_proj instead without a new measurement.",
        ],
    }


# ---------------------------------------------------------------------------
# duplication + reuse + candidate
# ---------------------------------------------------------------------------

def duplication_sites(src: dict[str, Any], art: dict[str, Any], geo: dict[str, Any]) -> list[dict[str, Any]]:
    s = src["sites"]
    suf = (art.get("by_suffix") or {}) if art.get("status") == "MEASURED" else {}

    def bytes_all(name: str) -> int | None:
        row = suf.get(name)
        return row["bytes_all"] if row else None

    rec_b = geo["rec_resident_bytes"]
    conv_b = geo["conv_resident_bytes"]
    return [
        {
            "tensor": "linear_attn.in_proj_qkvz.weight",
            "role": "static map hidden → (q,k,v,z) for the CURRENT token",
            "file": DECODE_RS,
            "line": s["in_proj_qkvz_bind"]["line"],
            "text": s["in_proj_qkvz_bind"]["text"],
            "on_disk_bytes": bytes_all("in_proj_qkvz.weight"),
            "duplicates_state": False,
            "why": (
                "Produces the rank-1 factors and the query of THIS token. rec_state "
                f"({rec_b} B) stores the decayed SUM of past k v^T "
                f"({Q38_METAL}:{s['state_rank1_update']['line']}: state += k * delta). "
                "That map is many-to-one: k_t, v_t, q_t are not recoverable from S_t. "
                "S cannot hold this matrix "
                f"({bytes_all('in_proj_qkvz.weight')} B on disk vs {rec_b} B resident S)."
            ),
        },
        {
            "tensor": "linear_attn.in_proj_ba.weight",
            "role": "static map hidden → (beta, a) that parameterise THIS token's decay/beta",
            "file": DECODE_RS,
            "line": s["in_proj_ba_bind"]["line"],
            "text": s["in_proj_ba_bind"]["text"],
            "on_disk_bytes": bytes_all("in_proj_ba.weight"),
            "duplicates_state": False,
            "why": (
                "ba is an activation of the current hidden. decay = exp(-exp(A_log) * "
                f"softplus(a+dt_bias)) at {Q80_METAL}:{s['decay_from_a_log']['line']}. "
                "S is multiplied by that decay; S does not store the ba map."
            ),
        },
        {
            "tensor": "linear_attn.conv1d.weight",
            "role": "FIR taps (k=4) on the qkvz channels",
            "file": Q38_METAL,
            "line": s["causal_conv"]["line"],
            "text": s["causal_conv"]["text"],
            "on_disk_bytes": bytes_all("conv1d.weight"),
            "duplicates_state": False,
            "why": (
                "conv_state is the delay line of the last (k-1) activations per channel "
                f"({Q38_METAL}:{s['conv_state_shift']['line']} shifts history and writes "
                f"current). conv1d.weight is the 4 taps. Delay line ≠ taps. "
                f"conv_state resident {conv_b} B; conv weights "
                f"{bytes_all('conv1d.weight')} B on disk."
            ),
        },
        {
            "tensor": "linear_attn.A_log",
            "role": "per-head static timescale of the transition",
            "file": Q80_METAL,
            "line": s["decay_from_a_log"]["line"],
            "text": s["decay_from_a_log"]["text"],
            "on_disk_bytes": bytes_all("A_log"),
            "duplicates_state": False,
            "why": (
                "48 f32 per layer. Already a parameter of the transition operator, not "
                "content of S. Could be folded into the kernel as a constant; that saves "
                "9.6 kB, not 2.95 GB."
            ),
        },
        {
            "tensor": "linear_attn.dt_bias",
            "role": "per-head static bias on the a-logit before softplus",
            "file": DECODE_RS,
            "line": s["dt_bias_bind"]["line"],
            "text": s["dt_bias_bind"]["text"],
            "on_disk_bytes": bytes_all("dt_bias"),
            "duplicates_state": False,
            "why": "Same class as A_log: a transition parameter, 9.6 kB, not S.",
        },
        {
            "tensor": "linear_attn.norm.weight",
            "role": "gated RMSNorm affine on rec_out (the S@q readout), not on S",
            "file": DECODE_RS,
            "line": s["norm_bind"]["line"],
            "text": s["norm_bind"]["text"],
            "on_disk_bytes": bytes_all("norm.weight"),
            "duplicates_state": False,
            "why": (
                "Kernel binds rec_out + z + norm.weight "
                f"({DECODE_RS}:{s['norm_bind']['line']}). Affine is 128 f32. "
                "Not a copy of the 786432-element S."
            ),
        },
        {
            "tensor": "linear_attn.out_proj.weight",
            "role": "static map rec_out (6144) → hidden (5120)",
            "file": DECODE_RS,
            "line": s["out_proj_bind"]["line"],
            "text": s["out_proj_bind"]["text"],
            "on_disk_bytes": bytes_all("out_proj.weight"),
            "duplicates_state": False,
            "why": (
                "Mixes a READOUT of S into residual space. The matrix is not in S. "
                "Composing out_proj_L with in_proj_{L+1} produces a larger map "
                "(16384×6144) and still leaves the residual path W@h to do, so it "
                "does not save bytes."
            ),
        },
    ]


def state_reuse(src: dict[str, Any], tns: dict[str, Any], geo: dict[str, Any]) -> dict[str, Any]:
    s = src["sites"]
    iso = tns["isolated"]
    gated = iso["gated_delta_48_ns"]
    stream = iso["stream_rec_state_ns"]
    return {
        "content_persists_across_tokens": {
            "value": True,
            "origin": "MEASURED from source",
            "evidence": (
                f"rec_state is allocated once ({DECODE_RS}:{s['rec_alloc']['line']}, "
                f"{geo['rec_resident_bytes']} B), zeroed in attach "
                f"({DECODE_RS}:{s['attach_zero_rec']['line']}) and reset "
                f"({DECODE_RS}:{s['reset_zero_rec']['line']}), and NOT zeroed in step() "
                f"(hits={s['step_zeros_rec_state']['hits']}). step() increments position "
                f"({DECODE_RS}:{s['step_position_inc']['line']}) and re-encodes against "
                "the same buffer."
            ),
        },
        "bytes_reread_from_device_memory_every_token": {
            "value": True,
            "origin": "MEASURED from source + TOKEN_NS isolated stream",
            "evidence": (
                f"encode_gated_delta binds rec_state every dispatch "
                f"({DECODE_RS}:{s['rec_state_set_buffer']['line']}). The kernel loads "
                f"state[index] from device ({Q38_METAL}:{s['state_decay_mul']['line']}). "
                f"Isolated stream_rec_state = {stream} ns for a sequential f32 copy of "
                f"the resident tensor; isolated gated_delta_48 = {gated} ns. Stream is "
                f"{stream / gated:.4f} of gated_delta isolated time. There is no "
                "cross-token SRAM/register residency: each token is a new CB with 964 "
                "dispatches; threadgroup scratch is per-dispatch."
            ),
        },
        "rebuilt_from_static_weights_every_token": {
            "value": False,
            "origin": "MEASURED from source",
            "evidence": "S_{t} = decay ⊙ S_{t-1} + rank-1; the buffer is the history.",
        },
        "sram_persistence_across_tokens": {
            "value": False,
            "origin": "MEASURED from kernel + production shape",
            "evidence": (
                "Production shape is 1 CB / 964 dispatches (TOKEN_NS). gated_delta "
                "threadgroup scratch is 128 floats and dies at dispatch end. Resident "
                "S is 3.15 MiB per layer; it does not fit in a threadgroup."
            ),
        },
        "verdict": (
            "State reuse of CONTENT is exploited (the organ is recurrent). State "
            "reuse of TRAFFIC is not free: the 151 MiB rec_state is re-read and "
            "re-written from unified memory every token. TOKEN_NS puts that traffic "
            "in kv_state (stream floor 0.467 ms rec + 0.019 ms conv), not in the "
            "10.6% deltanet component."
        ),
    }


def candidate_operator(
    geo: dict[str, Any],
    art: dict[str, Any],
    tns: dict[str, Any],
    sched: dict[str, Any],
    src: dict[str, Any],
) -> dict[str, Any]:
    suf = art.get("by_suffix") or {}
    f32_keep = sum(
        suf[k]["bytes_all"]
        for k in ("A_log", "dt_bias", "conv1d.weight", "norm.weight")
        if k in suf
    )
    q4_drop = sum(
        suf[k]["bytes_all"]
        for k in ("in_proj_qkvz.weight", "in_proj_ba.weight", "out_proj.weight")
        if k in suf
    )
    iso = tns["isolated"]
    # Hypothetical arithmetic ONLY, labeled. Isolated DN GEMV vs TOKEN_NS wall
    # and vs native wall. Not a candidate measurement — the operator does not exist.
    dn_gemv_ns = iso["dn_full_probe_ns"]
    wall = tns["median_wall_ns"]
    hyp_wall = wall - dn_gemv_ns
    hyp_tps_on_token_ns_wall = 1e9 / hyp_wall if hyp_wall > 0 else None
    hyp_native_ms = ANCHOR_MS_PER_TOKEN - dn_gemv_ns / 1e6
    hyp_tps_on_native = 1000.0 / hyp_native_ms if hyp_native_ms > 0 else None
    return {
        "name": "state_replaces_static_inproj",
        "question_it_answers": (
            "Can function currently encoded in linear_attn in_proj/out_proj be "
            "represented as rec_state plus a cheaper transition, so those GEMVs "
            "are not streamed?"
        ),
        "dense_reconstruction_oracle_only": False,
        "native_operator": {
            "exists_today": False,
            "closest_existing": {
                "name": "qwen38_gated_delta_decode_vi_simd",
                "file": Q38_METAL,
                "line": src["sites"]["gated_delta_simd"]["line"],
                "class": "DISPATCHED",
                "what_it_already_does": (
                    "decay S, rank-1 update from current k,v,beta, query readout. "
                    "It consumes activations that in_proj just produced. It does "
                    "not replace in_proj."
                ),
            },
            "would_need": (
                "A kernel that emits q,k,v,z,b,a (or a sufficient statistic) from "
                "rec_state plus a cheap feature of h, without streaming the Q4 "
                "in_proj/out_proj. No such kernel is in the 38 dispatched, the 4 "
                "helper-dispatched, or the REACHABLE set under recurrent_state_operator."
            ),
            "production_eligibility_today": (
                "A representation is executable TODAY only if it is grouped-absmax Q4, "
                "binary±CSR, HGRAVS01 factors, PQ codebook lookup, MoE worklists, or "
                "a recurrent state op. The recurrent state op exists and already runs. "
                "Replacing in_proj is a new representation and is not executable today."
            ),
        },
        "incumbent": {
            "q4_linear_attn_payload_bytes": geo["q4_linear_attn_payload"],
            "q4_on_disk_bytes": q4_drop,
            "f32_transition_on_disk_bytes": f32_keep,
            "rec_state_rw_bytes": geo["rec_rw_bytes"],
            "conv_state_rw_bytes": geo["conv_rw_bytes"],
            "dn_gemv_dispatches": 144,
            "dn_transition_dispatches": 192,
            "dn_mixer_prefix_dispatches": 48 * (sched.get("mixer_prefix_dispatches") or 9),
            "gemv_mac_flops": geo["gemv_mac_flops_per_token"],
            "state_update_flops": geo["state_update_flops_per_token"],
            "isolated_dn_full_probe_ns": iso["dn_full_probe_ns"],
            "isolated_gated_delta_ns": iso["gated_delta_48_ns"],
            "isolated_stream_rec_ns": iso["stream_rec_state_ns"],
            "token_ns_deltanet_component_ns": tns["component_deltanet"]["ns_per_token"],
            "origin": "geometry + artifact walk + TOKEN_NS isolated/component",
        },
        "candidate_if_it_existed_and_preserved_function": {
            "q4_linear_attn_payload_bytes": 0,
            "f32_transition_on_disk_bytes": f32_keep,
            "rec_state_rw_bytes": geo["rec_rw_bytes"],
            "conv_state_rw_bytes": geo["conv_rw_bytes"],
            "dn_gemv_dispatches": 0,
            "dn_transition_dispatches": 192,
            "gemv_mac_flops": 0,
            "state_update_flops": geo["state_update_flops_per_token"],
            "quality": {
                "status": "NULL",
                "reason": (
                    "No such operator. Prior science that would have to go the other "
                    "way first: G034/Phase-B activation-aware low-rank, G089 "
                    "out-of-distribution d_model compilation, MLP distill NO-GO "
                    "+0.4206 vs q3 at 72% active bytes, organ-census rank_99 of "
                    "DeltaNet post-LN hidden = 2082 of 5120. None of those is a "
                    "green light to skip in_proj."
                ),
            },
            "tok_s": {
                "status": "NULL",
                "reason": "operator does not exist; no generate was run",
            },
            "hypothetical_arithmetic_NOT_a_measurement": {
                "drop_isolated_dn_gemv_from_token_ns_wall_tps": hyp_tps_on_token_ns_wall,
                "drop_isolated_dn_gemv_from_native_30p606ms_tps": hyp_tps_on_native,
                "warning": (
                    "Isolated DN GEMV is a separate CB. Native 30.606 ms is a different "
                    "run from TOKEN_NS 35.228 ms. Subtracting across those clocks is "
                    "not a forecast. Even this optimistic subtraction on the TOKEN_NS "
                    "wall lands below the LIVE MLX 4-bit control "
                    f"({ANCHOR_MLX_TPS} tok/s) if you use that wall; on the native wall "
                    "it would look like it beats MLX. That disagreement is why this "
                    "block is labeled hypothetical and is not a result."
                ),
            },
        },
        "secondary_not_the_question": {
            "name": "fuse_rearrange_ba_delta_rmsnorm",
            "what": (
                "Four existing native kernels, 192 dispatches, already the 10.6% "
                "component. Fusing them does not drop the 2.95 GB Q4 stream."
            ),
            "bytes_saved": 0,
            "dispatches_192_to_48": True,
            "expected_ns": {
                "status": "NULL",
                "reason": "not timed; G024 said GEMV addressing is the lever, not fusion of isolated CBs",
            },
            "verdict": "NOT_THE_QUESTION",
        },
        "q80_state_codec_is_a_different_lever": (
            "Compressing rec_state (Q80 q8/q4 group-64) shrinks the 151 MiB that "
            "TOKEN_NS already split into kv_state. It does not touch in_proj. The "
            "Q80 receipt's claim_boundary forbids TPS and used deterministic "
            "vectors, not real tokens."
        ),
    }


def verdict_block(
    sites: list[dict[str, Any]],
    reuse: dict[str, Any],
    cand: dict[str, Any],
    tns: dict[str, Any],
    geo: dict[str, Any],
) -> dict[str, Any]:
    n_dup = sum(1 for s in sites if s["duplicates_state"])
    gemv_over_state = geo["gemv_mac_flops_per_token"] / geo["state_update_flops_per_token"]
    return {
        "decision": "NOT_WORTH_BUILDING",
        "decision_applies_to": "replacing DeltaNet static in_proj/out_proj with rec_state + a cheaper transition, on this model, as a production representation",
        "n_static_tensors_that_duplicate_state": n_dup,
        "why": [
            f"{n_dup} of {len(sites)} static tensors duplicate information S already carries. S stores the image of W, not W.",
            f"The organ's work is the projections: {geo['gemv_mac_flops_per_token']} MAC-FLOP vs {geo['state_update_flops_per_token']} state-update FLOP ({gemv_over_state:.2f}×). Isolated DN GEMV is 90.51% addressing.",
            "Capacity: rec_state 150994944 B cannot store in_proj_qkvz 2.14 GB. Rank-1 update is many-to-one (shader algebra).",
            "TOKEN_NS already has a native recurrent op. The 10.6% component is that op plus helpers, after GEMV and state-stream have been attributed out. Building another transition does not stop the 2.95 GB stream.",
            "Prior science that would have to reverse first: G034/Phase-B, G035, G089 OOD, MLP distill NO-GO, 223 sub-0.5 local BPW with 0 healthy.",
            "State CONTENT is reused; state TRAFFIC is re-read every token (stream_rec_state 0.467 ms). 'Reuse is free' is false on this machine.",
        ],
        "what_would_reopen": (
            "A native operator, scored on REAL held-out activations with a scale-aware "
            "gate (not cosine), that matches the incumbent q4 in_proj/out_proj on "
            "Doctor/function-space AND drops active bytes AND drops GEMV addressing "
            "time. Reconstruction of dense W as an oracle is not that operator."
        ),
        "beats_mlx_4bit": {
            "status": "NULL",
            "reason": "no candidate generate; incumbent 32.73 tok/s vs LIVE MLX 35.51",
        },
        "beats_llamacpp_q5k": {
            "status": "NULL",
            "reason": "ARCHIVED control 24.12 tok/s; artifact off disk; incumbent already above it",
        },
    }


def watched_fail(
    geo: dict[str, Any],
    tns: dict[str, Any],
    art: dict[str, Any],
    g042: dict[str, Any],
    prior: dict[str, Any],
    src: dict[str, Any],
    mlp: dict[str, Any],
) -> list[dict[str, Any]]:
    a_trap = ((art.get("a_log_loaded") or {}).get("pairwise") or {}).get("scale_trap_0p01") or {}
    items = [
        {
            "n": 1,
            "what": "10.6% and 26.3% look like two measurements of the same organ",
            "evidence": (
                f"TOKEN_NS component deltanet share={tns['component_deltanet']['share']:.6f} "
                f"on wall {tns['median_wall_ns']} ns. Organ aggregation "
                f"(dn_full_probe + component) share="
                f"{tns['organ_aggregation']['share_of_token_ns_wall']:.6f}. "
                f"seal_components at {LEDGER_RS}:{src['sites']['seal_deltanet_arith']['line']} "
                "subtracts state streams and parks GEMV in weight_addressing. "
                f"Recomputed component ns={tns['recomputed_deltanet_ns']:.3f} vs "
                f"ledger {tns['component_deltanet']['ns_per_token']:.3f}, "
                f"match={tns['recomputed_matches_component']}."
            ),
        },
        {
            "n": 2,
            "what": "G042 STATE_BPW_EQUIVALENT presents itself as KV/state and omits DeltaNet state",
            "evidence": (
                f"Stated ctx=128 BPW={g042.get('gqa_ctx128_bpw')}. Rec+conv BPW="
                f"{g042.get('deltanet_state_bpw_omitted')} "
                f"({g042.get('rec_over_gqa_at_ctx128')}× the GQA term). Definition "
                "names 16 GQA layers and 131072 B/position only."
            ),
        },
        {
            "n": 3,
            "what": "n1arch bundles G060/G061 as one REFUTED 'joint state+weight' row",
            "evidence": (
                "G061 measured GQA K at L63 (1024 tokens, holdout). "
                "joint_ties_with_independent=false; state_pca beats weight_svd at "
                "matched cache bytes. That is a latent-KV subspace result, not "
                "'DeltaNet state replaces in_proj'. Using the archaeology adjective "
                "as a DeltaNet closure would be the same class of error as treating "
                "G098 POSITIVE as 'sharing works'."
            ),
        },
        {
            "n": 4,
            "what": "Q80 rec_state q4/q8 cosine looks like a win for 'state as representation'",
            "evidence": (
                f"codec receipt model={((prior.get('Q80_deltanet_state_codec') or {}).get('model'))}, "
                "claim_boundary.deterministic_component_vectors_are_not_prompts_or_tokens="
                f"{((prior.get('Q80_deltanet_state_codec') or {}).get('not_real_tokens'))}, "
                "no_tps_measurement=true. Compressing S is not replacing W. Cosine "
                "on a reconstruction of S is the scale-blind metric the campaign already burned on."
            ),
        },
        {
            "n": 5,
            "what": "cosine(A_log, 0.01*A_log) on a REAL gravity tensor",
            "evidence": (
                f"scale_trap_0p01={a_trap}. Cosine accepts a 100× magnitude miss; "
                "scale_aware rejects it. This is the 0.01*W trap, measured on "
                "linear_attn.A_log from disk, not on a gaussian proxy."
            ),
        },
        {
            "n": 6,
            "what": ".lane-bootstrap/census is not in this sparse checkout",
            "evidence": (
                f"n1arch locate={prior['census']['n1arch']}. git ls-tree has the "
                "ascent receipts; the census markdown lives in hawking-copy extra_roots. "
                "Absence here is not absence in HEAD."
            ),
        },
        {
            "n": 7,
            "what": "organ-census deltanet physical.bytes counts 48 input_layernorm files",
            "evidence": (
                f"manifest linear_attn files={art.get('linear_attn_files')} "
                f"(336 = 48×7). Organ census tensor_count=384 = 336+48. TOKEN_NS "
                "puts those RMSNorms in 'normalization' (129 dispatches), not in "
                "the deltanet component. Another aggregation trap."
            ),
        },
        {
            "n": 8,
            "what": "search for a kernel that reads rec_state and skips in_proj",
            "evidence": (
                "gated_delta binds rec_state, repeated_q/k, conv_v, decay, beta, rec_out. "
                "in_proj is a separate Q4 GEMV onto workspace.qkvz / workspace.ba. "
                "No dispatched or helper kernel takes S to qkvz. recurrent_state_operator "
                "EXISTS and is already the incumbent transition, not a missing family."
            ),
        },
        {
            "n": 9,
            "what": "MLP distill NO-GO is a different organ; it is not a DeltaNet measurement",
            "evidence": (
                f"mlp_distill status={mlp.get('status')} "
                f"gap={mlp.get('deciding_number')} "
                f"I2560_byte_ratio={mlp.get('byte_ratio_I2560')} "
                f"locate={mlp.get('locate')}. Cited so this lane does not reopen "
                "'distill the function into a cheaper map' on in_proj without a new "
                "held-out run. It is not evidence about rec_state."
            ),
        },
        {
            "n": 10,
            "what": "two-server occupancy is not free; this process did not load gravity on GPU",
            "evidence": (
                f"anchor two_servers_tps={ANCHOR_TWO_SERVERS_TPS}, "
                f"one_server_tps={ANCHOR_ONE_SERVER_TPS}. GPU numbers in this receipt "
                "are cited from TOKEN_NS, not re-generated."
            ),
        },
    ]
    return items


def build() -> dict[str, Any]:
    t0 = time.perf_counter()
    geo = measure_geometry()
    sched = measure_schedule()
    src = measure_source_sites()
    art = measure_artifact()
    tns = measure_token_ns(geo) if geo.get("status") == "MEASURED" else {
        "status": "NULL",
        "reason": "geometry failed",
    }
    g042 = g042_state_omission(geo) if geo.get("status") == "MEASURED" else {
        "status": "NULL",
        "reason": "geometry failed",
    }
    mlp = measure_mlp_distill_anchor()
    prior = prior_science_search()
    sites = duplication_sites(src, art, geo) if geo.get("status") == "MEASURED" else []
    reuse = (
        state_reuse(src, tns, geo)
        if tns.get("status") == "MEASURED"
        else {"status": "NULL", "reason": "token_ns failed"}
    )
    cand = (
        candidate_operator(geo, art, tns, sched, src)
        if tns.get("status") == "MEASURED"
        else {"status": "NULL", "reason": "token_ns failed"}
    )
    verd = (
        verdict_block(sites, reuse, cand, tns, geo)
        if tns.get("status") == "MEASURED"
        else {"decision": "BLOCKED", "reason": "measurement failed"}
    )
    fail = watched_fail(geo, tns, art, g042, prior, src, mlp)

    # Identity checks this run.
    confirms = []

    def confirm(field: str, observed: Any, expected: Any, ok: bool) -> None:
        confirms.append(
            {"field": field, "observed": observed, "expected": expected, "confirmed": bool(ok)}
        )

    if geo.get("status") == "MEASURED":
        confirm("rec_resident_bytes", geo["rec_resident_bytes"], 150_994_944, geo["rec_resident_bytes"] == 150_994_944)
        confirm("conv_resident_bytes", geo["conv_resident_bytes"], 5_898_240, geo["conv_resident_bytes"] == 5_898_240)
        confirm("q4_linear_attn_payload", geo["q4_linear_attn_payload"], 2_953_789_440, geo["q4_linear_attn_payload"] == 2_953_789_440)
        confirm("state_update_flops", geo["state_update_flops_per_token"], 113_246_208.0, geo["state_update_flops_per_token"] == 113_246_208.0)
        confirm("qkvz_rows", geo["qkvz_rows"], 16384, geo["qkvz_rows"] == 16384)
        confirm("ba_rows", geo["ba_rows"], 96, geo["ba_rows"] == 96)
    if sched.get("status") == "MEASURED":
        confirm("formula_dispatches", sched["formula_total_dispatches"], ANCHOR_DISPATCHES, sched["formula_total_dispatches"] == ANCHOR_DISPATCHES)
        confirm("dn_prefix_len", sched["dn_mixer_prefix_count"], 9, sched["dn_mixer_prefix_count"] == 9)
    if tns.get("status") == "MEASURED":
        confirm("historical_10p6", tns["component_deltanet"]["share"], HISTORICAL_DELTANET_SHARE, tns["historical_10p6_reconciliation"]["matches_historical"])
        confirm("recomputed_deltanet", tns["recomputed_matches_component"], True, tns["recomputed_matches_component"])
        confirm("kv_read_identity", tns["kv_state_byte_identity"]["read_match"], True, tns["kv_state_byte_identity"]["read_match"])
        confirm("kv_write_identity", tns["kv_state_byte_identity"]["write_match"], True, tns["kv_state_byte_identity"]["write_match"])
        confirm("linear_attn_bytes_ledger", tns["weight_bytes"]["linear_attn_bytes"], geo["q4_linear_attn_payload"], tns["weight_bytes"]["linear_attn_bytes"] == geo["q4_linear_attn_payload"])
    if art.get("status") == "MEASURED":
        confirm("linear_attn_files", art["linear_attn_files"], 336, art["linear_attn_files"] == 336)
        confirm("q4_magic", art["q4_magic_bad"], 0, art["q4_magic_bad"] == 0)
        trap = (art.get("a_log_loaded") or {}).get("pairwise") or {}
        st = trap.get("scale_trap_0p01") or {}
        confirm("a_log_0p01_cosine_near_1", st.get("cosine"), 1.0, st.get("cosine") is not None and abs(st["cosine"] - 1.0) < 1e-9)
        confirm("a_log_0p01_scale_aware_near_0p01", st.get("scale_aware"), 0.01, st.get("scale_aware") is not None and abs(st["scale_aware"] - 0.01) < 1e-6)
    n_dup = sum(1 for s in sites if s["duplicates_state"])
    confirm("duplication_count_is_zero", n_dup, 0, n_dup == 0)

    organ, organ_loc = load_json(ORGAN_REL)
    organ_share = None
    organ_ns = None
    if organ and "deltanet" in organ.get("organs", {}):
        organ_share = organ["organs"]["deltanet"]["token_ns"]["share_of_token_ns"]
        organ_ns = organ["organs"]["deltanet"]["token_ns"]["ns_per_token"]
        if tns.get("status") == "MEASURED":
            confirm(
                "organ_census_share_matches_reaggregation",
                organ_share,
                tns["organ_aggregation"]["share_of_token_ns_wall"],
                abs(organ_share - tns["organ_aggregation"]["share_of_token_ns_wall"]) < 1e-6,
            )

    numbers = {
        "native_tps": {"value": ANCHOR_TPS, "origin": "ANCHOR_NOT_REDERIVED"},
        "native_ms_per_token": {"value": ANCHOR_MS_PER_TOKEN, "origin": "ANCHOR_NOT_REDERIVED"},
        "mlx_4bit_tps": {"value": ANCHOR_MLX_TPS, "origin": "ANCHOR_NOT_REDERIVED LIVE control"},
        "llamacpp_q5k_tps": {"value": ANCHOR_LLAMA_Q5K_TPS, "origin": "ANCHOR_NOT_REDERIVED ARCHIVED; artifact off disk"},
        "token_ns_wall_ns": {"value": tns.get("median_wall_ns"), "origin": f"CITED {TOKEN_NS_REL}"},
        "deltanet_component_ns": {"value": (tns.get("component_deltanet") or {}).get("ns_per_token"), "origin": f"CITED {TOKEN_NS_REL}"},
        "deltanet_component_share": {"value": (tns.get("component_deltanet") or {}).get("share"), "origin": "ARITHMETIC_ON_CITED (ns/wall)"},
        "organ_share_26p3": {"value": organ_share, "origin": f"CITED {ORGAN_REL}" if organ_share is not None else "NULL"},
        "linear_attn_q4_payload": {"value": geo.get("q4_linear_attn_payload"), "origin": "MEASURED geometry q4_matrix_bytes this run"},
        "rec_resident_bytes": {"value": geo.get("rec_resident_bytes"), "origin": "MEASURED geometry this run"},
        "conv_resident_bytes": {"value": geo.get("conv_resident_bytes"), "origin": "MEASURED geometry this run"},
        "dn_gemv_dispatches": {"value": 144, "origin": "MEASURED 48 layers × 3 Q4 GEMVs (schedule + isolated dn_gemvs)"},
        "dn_transition_dispatches": {"value": 192, "origin": "MEASURED TOKEN_NS component deltanet.dispatches = 48×4"},
        "production_dispatches": {"value": sched.get("formula_total_dispatches"), "origin": "MEASURED 1+64*(9+6)+3 from schedule.rs"},
        "state_update_flops": {"value": geo.get("state_update_flops_per_token"), "origin": "MEASURED 3 ops × rec_elems; matches G043"},
        "gemv_mac_flops": {"value": geo.get("gemv_mac_flops_per_token"), "origin": "MEASURED 2*rows*cols*48 this run"},
        "mlp_distill_gap": {"value": mlp.get("deciding_number", MLP_DISTILL_GAP), "origin": mlp.get("status")},
        "candidate_tok_s": {"value": None, "origin": "NULL operator does not exist"},
    }

    elapsed = time.perf_counter() - t0
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "repo": str(REPO),
        "elapsed_s": elapsed,
        "question": "Can function currently encoded redundantly in static DeltaNet weights be represented more cheaply in STATE plus a transition operator?",
        "verdict": verd,
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_MS_PER_TOKEN,
            "roof_GB_s": ANCHOR_ROOF_GB_S,
            "parameter_count": ANCHOR_PARAM_COUNT,
            "artifact_bytes": ANCHOR_ARTIFACT_BYTES,
            "artifact_files": ANCHOR_ARTIFACT_FILES,
            "tensors": ANCHOR_TENSORS,
            "complete_physical_bpw": ANCHOR_BPW,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers": ANCHOR_CBS,
            "gemv_gflop": ANCHOR_GFLOP,
            "mlx_4bit_tps": ANCHOR_MLX_TPS,
            "llamacpp_q5k_tps_archived": ANCHOR_LLAMA_Q5K_TPS,
            "two_servers_tps": ANCHOR_TWO_SERVERS_TPS,
            "one_server_tps": ANCHOR_ONE_SERVER_TPS,
            "null_cosine": NULL_COSINE,
            "mlp_distill_nogo_gap": MLP_DISTILL_GAP,
            "mlp_distill_byte_ratio": MLP_DISTILL_BYTE_RATIO,
            "historical_deltanet_share": HISTORICAL_DELTANET_SHARE,
            "dense_reconstruction_is_oracle_not_production": True,
            "source_and_executable_both_dispatch_964": True,
        },
        "geometry": geo,
        "schedule": sched,
        "source_sites": src,
        "artifact": art,
        "token_ns": tns,
        "g042_state_omission": g042,
        "mlp_distill": mlp,
        "organ_census_deltanet_token_ns": {
            "locate": organ_loc,
            "ns_per_token": organ_ns,
            "share_of_token_ns": organ_share,
        },
        "per_token_cost": {
            "bytes_weight_q4_payload": geo.get("q4_linear_attn_payload"),
            "bytes_weight_on_disk": art.get("linear_attn_on_disk_bytes"),
            "bytes_state_rec_resident": geo.get("rec_resident_bytes"),
            "bytes_state_conv_resident": geo.get("conv_resident_bytes"),
            "bytes_state_rec_rw": geo.get("rec_rw_bytes"),
            "bytes_state_conv_rw": geo.get("conv_rw_bytes"),
            "dispatches_component_deltanet": (tns.get("component_deltanet") or {}).get("dispatches"),
            "dispatches_dn_gemv": 144,
            "dispatches_dn_mixer_prefix": 48 * (sched.get("mixer_prefix_dispatches") or 9),
            "share_token_ns_component": (tns.get("component_deltanet") or {}).get("share"),
            "share_organ_aggregation": (tns.get("organ_aggregation") or {}).get("share_of_token_ns_wall"),
            "ns_component": (tns.get("component_deltanet") or {}).get("ns_per_token"),
            "ns_organ_aggregation": (tns.get("organ_aggregation") or {}).get("ns_per_token"),
        },
        "duplication_sites": sites,
        "n_duplication_sites_true": n_dup,
        "state_reuse": reuse,
        "candidate": cand,
        "prior_science": prior,
        "numbers": numbers,
        "confirms": confirms,
        "all_pinned_numbers_confirmed": all(c["confirmed"] for c in confirms) if confirms else False,
        "what_i_watched_fail": fail,
        "write_scope": {
            "write": [
                "tools/headless/noetic_deltanet_design.py",
                "receipts/headless/NOETIC_DELTANET_DESIGN.json",
            ],
            "deny": ["workspace", "crates", "visionmcp", "app", "lab", "tools/haider", "ramanujan"],
        },
        "did_not": [
            "load gravity onto the GPU",
            "spawn a second 27B",
            "evaluate on synthetic activations",
            "present reconstruct-dense as production",
            "re-time TOKEN_NS (cited; arithmetic recomputed)",
            "re-run MLP distillation",
            "re-run G034/G035/G061",
        ],
    }


def print_report(r: dict[str, Any]) -> None:
    v = r["verdict"]
    tns = r.get("token_ns") or {}
    geo = r.get("geometry") or {}
    art = r.get("artifact") or {}
    print("=" * 78)
    print("NOETIC DELTANET DESIGN — can state replace static weight?")
    print("=" * 78)
    print(f"schema     {r['schema']}")
    print(f"generated  {r['generated_at']}")
    print(f"head       {r['git_head']}")
    print(f"elapsed_s  {r['elapsed_s']:.3f}")
    print(f"verdict    {v.get('decision')}")
    print(f"question   {r['question']}")
    print()

    print("## 1. DELTANET PER-TOKEN COST (MEASURED / CITED)")
    pt = r["per_token_cost"]
    print(f"  Q4 linear_attn payload bytes     {pt['bytes_weight_q4_payload']}")
    print(f"  linear_attn on-disk bytes        {pt['bytes_weight_on_disk']}")
    print(f"  rec_state resident / R+W bytes   {pt['bytes_state_rec_resident']} / {pt['bytes_state_rec_rw']}")
    print(f"  conv_state resident / R+W bytes  {pt['bytes_state_conv_resident']} / {pt['bytes_state_conv_rw']}")
    print(f"  dispatches (component deltanet)  {pt['dispatches_component_deltanet']}   # 48×(rearrange,ba,gated_delta,gated_rmsnorm)")
    print(f"  dispatches (DN GEMV)             {pt['dispatches_dn_gemv']}   # 48×(qkvz,ba,out) inside the 401 Q4 matvecs")
    print(f"  dispatches (DN mixer prefix)     {pt['dispatches_dn_mixer_prefix']}   # 48×9 including rmsnorm+residual")
    print(f"  TOKEN_NS component ns / share    {pt['ns_component']} / {pt['share_token_ns_component']}")
    print(f"  organ aggregation ns / share     {pt['ns_organ_aggregation']} / {pt['share_organ_aggregation']}")
    rec = (tns.get("historical_10p6_reconciliation") or {})
    print(f"  historical ~10.6%                {rec.get('historical_share_quoted')}  matches={rec.get('matches_historical')}")
    print(f"  why 10.6% ≠ 26.3%                {rec.get('why_they_differ')}")
    print(f"  TOKEN_NS wall ns                 {tns.get('median_wall_ns')}  (do not mix with native {ANCHOR_MS_PER_TOKEN} ms)")
    iso = tns.get("isolated") or {}
    print(f"  isolated dn_full_probe ns        {iso.get('dn_full_probe_ns')}  addr_frac={tns.get('dn_addr_frac_of_full')}")
    print(f"  isolated gated_delta_48 ns       {iso.get('gated_delta_48_ns')}")
    print(f"  isolated stream_rec_state ns     {iso.get('stream_rec_state_ns')}")
    print(f"  GEMV MAC-FLOP / state-update     {geo.get('gemv_mac_flops_per_token')} / {geo.get('state_update_flops_per_token')}")
    kv = tns.get("kv_state_byte_identity") or {}
    print(f"  kv_state byte identity read      {kv.get('read_match')}  write={kv.get('write_match')}")
    print(f"  recomputed deltanet matches      {tns.get('recomputed_matches_component')}  delta_ns={tns.get('recomputed_abs_delta_ns')}")
    print()

    print("## 2. STATIC WEIGHT vs STATE (every site, file:line)")
    print(f"  n_duplicates={r['n_duplication_sites_true']} of {len(r['duplication_sites'])}")
    for site in r["duplication_sites"]:
        mark = "DUPLICATES" if site["duplicates_state"] else "does_not_duplicate"
        print(f"  [{mark}] {site['tensor']}")
        print(f"           {site['file']}:{site['line']}")
        print(f"           {site['why']}")
    print()

    print("## 3. CANDIDATE OPERATOR")
    c = r.get("candidate") or {}
    print(f"  name        {c.get('name')}")
    print(f"  native_op   exists_today={((c.get('native_operator') or {}).get('exists_today'))}")
    inc = c.get("incumbent") or {}
    cand = c.get("candidate_if_it_existed_and_preserved_function") or {}
    print("  incumbent bytes/ops/dispatches:")
    print(f"           q4_payload={inc.get('q4_linear_attn_payload_bytes')}  rec_rw={inc.get('rec_state_rw_bytes')}  gemv_disp={inc.get('dn_gemv_dispatches')}  trans_disp={inc.get('dn_transition_dispatches')}  gemv_mac={inc.get('gemv_mac_flops')}")
    print("  candidate (if it existed and preserved function):")
    print(f"           q4_payload={cand.get('q4_linear_attn_payload_bytes')}  rec_rw={cand.get('rec_state_rw_bytes')}  gemv_disp={cand.get('dn_gemv_dispatches')}  trans_disp={cand.get('dn_transition_dispatches')}  gemv_mac={cand.get('gemv_mac_flops')}")
    print(f"           quality={((cand.get('quality') or {}).get('status'))}  tok_s={((cand.get('tok_s') or {}).get('status'))}")
    print(f"  {((cand.get('tok_s') or {}).get('reason'))}")
    print(f"  secondary (not the question): {((c.get('secondary_not_the_question') or {}).get('name'))} → {((c.get('secondary_not_the_question') or {}).get('verdict'))}")
    print()

    print("## 4. STATE REUSE: CONTENT vs TRAFFIC")
    ru = r.get("state_reuse") or {}
    for key in (
        "content_persists_across_tokens",
        "bytes_reread_from_device_memory_every_token",
        "rebuilt_from_static_weights_every_token",
        "sram_persistence_across_tokens",
    ):
        row = ru.get(key) or {}
        print(f"  {key}: {row.get('value')}")
        print(f"           {row.get('evidence')}")
    print(f"  verdict: {ru.get('verdict')}")
    print()

    print("## 5. VERDICT")
    print(f"  {v.get('decision')} — {v.get('decision_applies_to')}")
    for line in v.get("why") or []:
        print(f"  - {line}")
    print(f"  reopen: {v.get('what_would_reopen')}")
    print()

    print("## 6. PRIOR SCIENCE SEARCH")
    pr = r.get("prior_science") or {}
    gg = pr.get("git_grep") or {}
    print(f"  git grep files: {gg.get('n_files')}  (showing {len(gg.get('files_head') or [])})")
    print(f"  n1arch locate:  {pr.get('census', {}).get('n1arch')}")
    rec_op = pr.get("kernel_census_recurrent_state_operator") or {}
    print(f"  recurrent_state_operator: {rec_op.get('verdict')} kernel={((rec_op.get('kernel') or {}).get('name'))}")
    g061 = pr.get("G061_joint_state_weight") or {}
    print(f"  G061 organ: {g061.get('organ')} joint_ties={g061.get('joint_ties_with_independent')}")
    for line in pr.get("reading") or []:
        print(f"  - {line}")
    print()

    print("## 7. NUMBER ORIGINS")
    for k, row in (r.get("numbers") or {}).items():
        print(f"  {k:32s} {row.get('value')!s:24s}  {row.get('origin')}")
    print()
    print("## CONFIRMS")
    for crow in r.get("confirms") or []:
        mark = "OK" if crow["confirmed"] else "FAIL"
        print(f"  [{mark}] {crow['field']}: observed={crow['observed']} expected={crow['expected']}")
    print(f"  all_pinned={r.get('all_pinned_numbers_confirmed')}")
    print()

    print("## WHAT I WATCHED FAIL")
    for w in r.get("what_i_watched_fail") or []:
        print(f"  {w['n']}. {w['what']}")
        print(f"     {w['evidence']}")
    print()
    a = (art.get("a_log_loaded") or {}).get("pairwise") or {}
    print("## A_log LIVE LOAD (gravity artifact, this run)")
    print(f"  n={((r.get('artifact') or {}).get('a_log_loaded') or {}).get('n_vectors')} "
          f"min={((r.get('artifact') or {}).get('a_log_loaded') or {}).get('min')} "
          f"max={((r.get('artifact') or {}).get('a_log_loaded') or {}).get('max')} "
          f"mean={((r.get('artifact') or {}).get('a_log_loaded') or {}).get('mean')}")
    print(f"  pairwise scale_aware mean={a.get('mean_pairwise_scale_aware')} min={a.get('min_pairwise_scale_aware')}")
    print(f"  0.01 trap cosine={((a.get('scale_trap_0p01') or {}).get('cosine'))} "
          f"scale_aware={((a.get('scale_trap_0p01') or {}).get('scale_aware'))}")
    g042 = r.get("g042_state_omission") or {}
    print()
    print("## G042 STATE ACCOUNTING")
    print(f"  GQA ctx=128 BPW={g042.get('gqa_ctx128_bpw')}  DN rec+conv BPW={g042.get('deltanet_state_bpw_omitted')}  ratio={g042.get('rec_over_gqa_at_ctx128')}")
    print(f"  {g042.get('omission')}")
    print("=" * 78)


def write_receipt(r: dict[str, Any]) -> Path:
    dest = REPO / "receipts" / "headless" / "NOETIC_DELTANET_DESIGN.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r, indent=2, default=str) + "\n")
    return dest


def main() -> int:
    receipt = build()
    try:
        path = write_receipt(receipt)
        receipt["wrote_to"] = str(path)
        path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    except OSError as e:
        receipt["wrote_to"] = f"WRITE_FAILED: {e}"
        print_report(receipt)
        print(f"WRITE_FAILED {e}", file=sys.stderr)
        return 2
    print_report(receipt)
    if receipt.get("verdict", {}).get("decision") == "BLOCKED":
        return 2
    if not receipt.get("all_pinned_numbers_confirmed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
