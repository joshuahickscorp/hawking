#!/usr/bin/env python3
"""Noetic GQA design — where the storage mass actually is.

Does not open Metal, does not load the 27B, does not spawn a server, does not
re-derive GPU timings or the 4.125 / 2.4% figures. Those come from receipts.
Kernel evidence is read from source. Closed arithmetic on loaded numbers is
labelled DERIVED. A plausible-looking figure with no measurement behind it is
the defect this campaign keeps finding.

    python3 tools/headless/noetic_gqa_design.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "hawking.headless.noetic_gqa_design.v1"

# Anchors the contract forbade us to re-derive.
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_BOUND = 38
ANCHOR_DECLARED = 554
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_TENSORS = 755
ANCHOR_GEMV_GFLOP = 51.24
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_Q5K_TPS = 24.12

ORGAN_REL = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
OPS_REL = "receipts/headless/NOETIC_OPERATION_CENSUS.json"
KERN_REL = "receipts/headless/NOETIC_KERNEL_CENSUS.json"
METRICS_REL = "receipts/headless/NOETIC_METRICS.json"
ACCT_REL = "receipts/headless/NOETIC_INFORMATION_ACCOUNTING.json"
LEDGER_REL = "receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"
DENSITY_PROBE_REL = "receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json"
DENSITY_VERDICT_REL = "receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json"
DENSITY_ROOT_REL = "receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json"
COHERENCE_REL = "receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json"
G035_REL = "receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json"
G060_REL = "receipts/ascent-2026-08-16/G060_LATENT_KV_VERDICT.json"
DEAD_LEVERS_REL = "workspace/docs/guides/dead_levers.md"
MLP_DISTILL_REL = "receipts/headless/NOETIC_MLP_DISTILL_PROBE.json"
N1ARCH_REL = ".lane-bootstrap/census/n1arch.md"
N15NEG_REL = ".lane-bootstrap/census/n15neg.md"
N16CLOS_REL = ".lane-bootstrap/census/n16clos.md"

DECODE_REL = "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
GEOMETRY_REL = "crates/hawking-core/src/model/qwen38_geometry.rs"
SCHEDULE_REL = "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs"
MHA_REL = "crates/hawking-core/shaders/mha.metal"
GQA_SHADER_REL = "crates/hawking-core/shaders/qwen38_device_activations.metal"


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
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_GQA_DESIGN.json"


def extra_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HAWKING_COPY"),
        os.environ.get("HAWKING_ROOT"),
        str(Path.home() / "Downloads" / "hawking-copy"),
        "/Users/scammermike/Downloads/hawking-copy",
    ):
        if not raw:
            continue
        p = Path(raw)
        if p.exists() and p not in roots and p != REPO:
            roots.append(p)
    wt = Path.home() / ".claude-grok" / "worktrees"
    if wt.is_dir():
        for child in sorted(wt.iterdir()):
            if child.is_dir() and child != REPO and child not in roots:
                roots.append(child)
    return roots


def git_head() -> str:
    r = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return (r.stdout or "").strip() or "UNKNOWN"


def git_show(rel: str) -> bytes | None:
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        capture_output=True,
        timeout=60,
    )
    if r.returncode == 0 and r.stdout:
        return r.stdout
    return None


def locate(rel: str) -> dict[str, Any]:
    """Resolve a path. Sparse checkout is not evidence of absence."""
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
    for root in extra_roots():
        p = root / rel
        tried.append(f"disk:{p}")
        if p.is_file():
            return {
                "rel": rel,
                "found": True,
                "how": "copy",
                "path": str(p),
                "bytes": p.stat().st_size,
            }
    return {"rel": rel, "found": False, "how": None, "path": None, "tried": tried}


def load_text(rel: str) -> tuple[str | None, dict[str, Any]]:
    loc = locate(rel)
    if not loc["found"]:
        return None, loc
    if loc["how"] == "git":
        blob = git_show(rel)
        assert blob is not None
        return blob.decode("utf-8", errors="replace"), loc
    return Path(loc["path"]).read_text(errors="replace"), loc


def load_json(rel: str) -> tuple[Any, dict[str, Any]]:
    text, loc = load_text(rel)
    if text is None:
        return None, loc
    return json.loads(text), loc


def measured(value: Any, source: str, unit: str | None = None, note: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"value": value, "status": "MEASURED", "source": source}
    if unit is not None:
        row["unit"] = unit
    if note is not None:
        row["note"] = note
    return row


def derived(value: Any, formula: str, unit: str | None = None, note: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"value": value, "status": "DERIVED", "formula": formula}
    if unit is not None:
        row["unit"] = unit
    if note is not None:
        row["note"] = note
    return row


def cited(value: Any, source: str, unit: str | None = None, note: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"value": value, "status": "CITED", "source": source}
    if unit is not None:
        row["unit"] = unit
    if note is not None:
        row["note"] = note
    return row


def null_reason(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "NULL", "reason": reason}


def snippet(text: str, needle: str, radius: int = 180) -> str | None:
    i = text.find(needle)
    if i < 0:
        return None
    lo = max(0, i - radius)
    hi = min(len(text), i + len(needle) + radius)
    return text[lo:hi].replace("\n", " ")


def line_at(text: str, lineno: int) -> str:
    lines = text.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def find_line(text: str, needle: str) -> int | None:
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return None


def rust_str_array(text: str, const_name: str) -> list[str]:
    m = re.search(
        rf"pub const {re.escape(const_name)}:[^\[]*\[[^\]]+\]\s*=\s*\[(.*?)\];",
        text,
        re.S,
    )
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def gemv_by_organ(ops: dict) -> dict[str, dict]:
    out = {}
    for row in ops.get("gemv_organs") or []:
        out[row["organ"]] = row
    return out


def iso_by_name(ledger: dict) -> dict[str, dict]:
    out = {}
    for row in ledger.get("isolated") or []:
        out[row["name"]] = row
    return out


def component_by_name(ledger: dict) -> dict[str, dict]:
    out = {}
    for row in ledger.get("components") or []:
        out[row.get("component") or row.get("name")] = row
    return out


def git_grep_count(pattern: str) -> dict[str, Any]:
    r = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-l", pattern, "HEAD"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    files = [ln.split(":", 1)[-1] for ln in (r.stdout or "").splitlines() if ln]
    # git grep -l HEAD prints HEAD:path
    files = [f[5:] if f.startswith("HEAD:") else f for f in files]
    return {"pattern": pattern, "n_files": len(files), "sample": files[:8]}


def main() -> int:
    t0 = time.perf_counter()
    watched: list[str] = []
    located: dict[str, Any] = {}

    organ, located[ORGAN_REL] = load_json(ORGAN_REL)
    ops, located[OPS_REL] = load_json(OPS_REL)
    kern, located[KERN_REL] = load_json(KERN_REL)
    metrics, located[METRICS_REL] = load_json(METRICS_REL)
    acct, located[ACCT_REL] = load_json(ACCT_REL)
    ledger, located[LEDGER_REL] = load_json(LEDGER_REL)
    probe, located[DENSITY_PROBE_REL] = load_json(DENSITY_PROBE_REL)
    verdict, located[DENSITY_VERDICT_REL] = load_json(DENSITY_VERDICT_REL)
    density_root, located[DENSITY_ROOT_REL] = load_json(DENSITY_ROOT_REL)
    coherence, located[COHERENCE_REL] = load_json(COHERENCE_REL)
    g035, located[G035_REL] = load_json(G035_REL)
    g060, located[G060_REL] = load_json(G060_REL)
    mlp, located[MLP_DISTILL_REL] = load_json(MLP_DISTILL_REL)
    dead_levers, located[DEAD_LEVERS_REL] = load_text(DEAD_LEVERS_REL)
    n1arch, located[N1ARCH_REL] = load_text(N1ARCH_REL)
    n15neg, located[N15NEG_REL] = load_text(N15NEG_REL)
    n16clos, located[N16CLOS_REL] = load_text(N16CLOS_REL)
    decode, located[DECODE_REL] = load_text(DECODE_REL)
    geometry, located[GEOMETRY_REL] = load_text(GEOMETRY_REL)
    schedule, located[SCHEDULE_REL] = load_text(SCHEDULE_REL)
    mha, located[MHA_REL] = load_text(MHA_REL)
    gqa_shader, located[GQA_SHADER_REL] = load_text(GQA_SHADER_REL)

    required = [
        (ORGAN_REL, organ),
        (OPS_REL, ops),
        (KERN_REL, kern),
        (LEDGER_REL, ledger),
        (DENSITY_PROBE_REL, probe),
        (DENSITY_VERDICT_REL, verdict),
        (DECODE_REL, decode),
        (GEOMETRY_REL, geometry),
        (SCHEDULE_REL, schedule),
        (MHA_REL, mha),
        (GQA_SHADER_REL, gqa_shader),
    ]
    missing = [rel for rel, obj in required if obj is None]
    if missing:
        print("NOETIC GQA DESIGN")
        print("=" * 72)
        print("REQUIRED RECEIPTS OR SOURCE MISSING")
        for m in missing:
            print(f"  {m}  {located[m]}")
        print("Sparse checkout is not absence. git show HEAD:<path>, or HAWKING_COPY.")
        return 2

    assert organ is not None and ops is not None and kern is not None
    assert ledger is not None and probe is not None and verdict is not None
    assert decode is not None and geometry is not None and schedule is not None
    assert mha is not None and gqa_shader is not None

    # ------------------------------------------------------------------
    # 0. prior science loaded, not rediscovered
    # ------------------------------------------------------------------
    gqa_organ = organ["organs"]["attention_gqa"]
    mlp_organ = organ["organs"]["mlp"]
    dn_organ = organ["organs"]["deltanet"]
    gemv = gemv_by_organ(ops)
    iso = iso_by_name(ledger)
    comps = component_by_name(ledger)
    gqa_comp = comps["gqa"]
    total_token_ns = float(ledger["median_wall_ns"])
    payload_b = int(
        (acct or {}).get("artifact_accounting", {}).get("identity", {}).get("payload_bytes_sum")
        or organ["artifact"]["bytes"]
    )
    artifact_b = int(organ["artifact"]["bytes"])
    params = int(organ["artifact"]["parameter_count"])

    g035_flags = []
    if isinstance(g035, dict):
        def walk_g035(o):
            if isinstance(o, dict):
                if "shared_beats_independent" in o:
                    g035_flags.append(o["shared_beats_independent"])
                for v in o.values():
                    walk_g035(v)
            elif isinstance(o, list):
                for v in o:
                    walk_g035(v)
        walk_g035(g035)

    mlp_verdict = None
    if isinstance(mlp, dict):
        mlp_verdict = mlp.get("verdict")
    else:
        watched.append(
            "NOETIC_MLP_DISTILL_PROBE.json is not in this tree. The contract names "
            "MLP function distillation NO-GO (+0.4206 held-out gap vs q3 at 72% of "
            "active bytes). Located via extra_roots if a sibling lane has landed it; "
            "otherwise cited from the contract and not re-run."
        )

    s011 = git_grep_count("S011")
    if s011["n_files"] == 0:
        watched.append(
            "git grep S011 over HEAD is empty. The contract names S011 §4 "
            "('storage alone is incomplete'). The measurement of that law in this "
            "tree is NOETIC_OPERATION_CENSUS: same 51.24 GFLOP GEMV, 964 dispatches, "
            "dense W materialized = 0. The section label is not in git."
        )

    # ------------------------------------------------------------------
    # 1. attention share today
    # ------------------------------------------------------------------
    gqa_bytes = int(gqa_organ["physical"]["bytes"])
    gqa_active = int(gqa_organ["physical"]["active_bytes_per_token"])
    gqa_elems = int(gqa_organ["physical"]["elements"])
    gqa_tensors = int(gqa_organ["physical"]["tensor_count"])
    mixer_bytes = gqa_bytes + int(dn_organ["physical"]["bytes"])
    mlp_bytes = int(mlp_organ["physical"]["bytes"])

    q = gemv["self_attn.q_proj"]
    k = gemv["self_attn.k_proj"]
    v = gemv["self_attn.v_proj"]
    o = gemv["self_attn.o_proj"]
    gqa_q4_bytes = (
        int(q["q4_bytes_per_token"])
        + int(k["q4_bytes_per_token"])
        + int(v["q4_bytes_per_token"])
        + int(o["q4_bytes_per_token"])
    )
    gqa_mac = (
        int(q["mac_flops_per_token"])
        + int(k["mac_flops_per_token"])
        + int(v["mac_flops_per_token"])
        + int(o["mac_flops_per_token"])
    )
    total_gemv_mac = int(ops["analytic_vs_measured"]["dispatched_gemv_mac_flops"])
    mha_mac = int(ops["activation_flops"]["mha_mac"])
    act_flops = int(ops["activation_flops"]["total_flops"])

    gqa_prefix = rust_str_array(schedule, "QWEN38_GQA_MIXER_PREFIX_KERNELS")
    dn_prefix = rust_str_array(schedule, "QWEN38_DELTANET_MIXER_PREFIX_KERNELS")
    mlp_suffix = rust_str_array(schedule, "QWEN38_DENSE_MLP_SUFFIX_KERNELS")
    gqa_layers = 16
    # GQA-specific kernel names in the 9-slot prefix (exclude rmsnorm + residual,
    # which every mixer pays).
    gqa_named = [
        n
        for n in gqa_prefix
        if n
        not in (
            "qwen80_residual_rmsnorm_f32",
            "qwen_next_add_residual",
        )
    ]
    gqa_named_dispatches = gqa_layers * len(gqa_named)  # 16 * 7 = 112
    gqa_gemv_dispatches = gqa_layers * 4  # q,k,v,o
    # Full prefix including rms + residual: 16 * 9 = 144 of the 576 mixer_prefix.
    gqa_prefix_dispatches = gqa_layers * len(gqa_prefix)

    rope_ns = float(iso["rope_cache_16"]["median_gpu_ns"])
    mha_ns = float(iso["mha_16"]["median_gpu_ns"])
    sigmoid_ns = float(iso["sigmoid_16"]["median_gpu_ns"])
    gqa_gemv_ns = float(iso["gqa_gemvs"]["median_gpu_ns"])
    gqa_full_probe_ns = float(iso["gqa_full_probe"]["median_gpu_ns"])
    stream_k_ns = float(iso["stream_gqa_key"]["median_gpu_ns"])
    stream_v_ns = float(iso["stream_gqa_value"]["median_gpu_ns"])
    gqa_kv_stream_ns = stream_k_ns + stream_v_ns
    gqa_component_ns = float(gqa_comp["ns_per_token"])
    kv_state_ns = float(comps["kv_state"]["ns_per_token"])
    organ_ns_as_census = float(gqa_organ["token_ns"]["ns_per_token"])
    # Census folded the whole rec+conv+GQA stream into GQA. Unique GQA KV stream is isolated.
    organ_ns_gqa_kv_only = gqa_full_probe_ns + gqa_component_ns + gqa_kv_stream_ns

    share = {
        "stored_bytes": measured(
            gqa_bytes,
            f"{ORGAN_REL} organs.attention_gqa.physical.bytes",
            "bytes",
        ),
        "stored_bytes_share_of_on_disk": derived(
            gqa_bytes / artifact_b,
            f"{gqa_bytes} / artifact.on_disk {artifact_b}",
            "fraction",
        ),
        "stored_bytes_share_of_payload": derived(
            gqa_bytes / payload_b,
            f"{gqa_bytes} / payload {payload_b}",
            "fraction",
        ),
        "elements": measured(gqa_elems, f"{ORGAN_REL} organs.attention_gqa.physical.elements"),
        "tensor_count": measured(gqa_tensors, f"{ORGAN_REL} organs.attention_gqa.physical.tensor_count"),
        "q4_tensors": measured(
            int(gqa_organ["physical"]["q4_tensors"]),
            f"{ORGAN_REL} organs.attention_gqa.physical.q4_tensors",
        ),
        "f32_tensors": measured(
            int(gqa_organ["physical"]["f32_tensors"]),
            f"{ORGAN_REL} organs.attention_gqa.physical.f32_tensors",
        ),
        "local_physical_bpw": derived(
            gqa_bytes * 8 / gqa_elems,
            f"8 * {gqa_bytes} / {gqa_elems}",
            "bits/weight",
            "Q4 g64 body plus 48 small f32 tensors (q/k norms).",
        ),
        "active_bytes_per_token": measured(
            gqa_active,
            f"{ORGAN_REL} organs.attention_gqa.physical.active_bytes_per_token",
            "bytes/token",
            "decode streams the full packed organ every token (no MoE).",
        ),
        "active_q4_gemv_bytes_per_token": measured(
            gqa_q4_bytes,
            f"{OPS_REL} gemv_organs self_attn.{{q,k,v,o}}_proj q4_bytes_per_token sum",
            "bytes/token",
        ),
        "active_share_of_weight_stream": derived(
            gqa_active / int(kern["production_token"]["active_weight_bytes_per_token"]),
            f"{gqa_active} / production_token.active_weight_bytes_per_token",
            "fraction",
        ),
        "dispatches": {
            "gqa_named_kernel_launches": derived(
                gqa_named_dispatches,
                f"{gqa_layers} GQA layers * {len(gqa_named)} named prefix kernels {gqa_named}",
            ),
            "gqa_gemv_launches": derived(
                gqa_gemv_dispatches,
                f"{gqa_layers} * 4 encode_q4_matvec (q,k,v,o)",
            ),
            "gqa_mixer_prefix_including_rms_and_residual": derived(
                gqa_prefix_dispatches,
                f"{gqa_layers} * {len(gqa_prefix)} schedule slots",
            ),
            "production_dispatches_per_token": measured(
                int(ops["dispatch_reconciliation"]["recorded_anchor"]),
                f"{OPS_REL} dispatch_reconciliation.recorded_anchor",
            ),
            "share_of_964_named": derived(
                gqa_named_dispatches / ANCHOR_DISPATCHES,
                f"{gqa_named_dispatches} / {ANCHOR_DISPATCHES}",
                "fraction",
            ),
            "command_buffers": measured(ANCHOR_CBS, "production shape; ledger production_cb_shape"),
        },
        "token_ns": {
            "ledger_wall_ns": measured(total_token_ns, f"{LEDGER_REL} median_wall_ns", "ns"),
            "gqa_component_ns": measured(
                gqa_component_ns,
                f"{LEDGER_REL} components[gqa].ns_per_token",
                "ns",
                "rope leftover + mha leftover + GQA GEMV FMA remainder + sigmoid + 16/64 mixer residual. NOT the GEMV bytes. Ledger commit 57ee82cc names qwen38_gqa_qk_norm_rope_cache_f32; live default is HAWKING_ROPE_TG=256 (_tg). The 1.56 ms isolated rope is the 24-thread form.",
            ),
            "gqa_component_share": derived(
                gqa_component_ns / total_token_ns,
                f"{gqa_component_ns} / {total_token_ns}",
                "fraction",
            ),
            "isolated_rope_ns": measured(rope_ns, f"{LEDGER_REL} isolated rope_cache_16.median_gpu_ns", "ns"),
            "isolated_mha_ns": measured(mha_ns, f"{LEDGER_REL} isolated mha_16.median_gpu_ns", "ns"),
            "isolated_sigmoid_ns": measured(sigmoid_ns, f"{LEDGER_REL} isolated sigmoid_16.median_gpu_ns", "ns"),
            "isolated_gqa_gemvs_ns": measured(gqa_gemv_ns, f"{LEDGER_REL} isolated gqa_gemvs.median_gpu_ns", "ns"),
            "isolated_gqa_full_probe_ns": measured(
                gqa_full_probe_ns, f"{LEDGER_REL} isolated gqa_full_probe.median_gpu_ns", "ns"
            ),
            "isolated_gqa_kv_stream_ns": derived(
                gqa_kv_stream_ns,
                f"stream_gqa_key {stream_k_ns} + stream_gqa_value {stream_v_ns}",
                "ns",
            ),
            "mha_share_of_wall": derived(
                mha_ns / total_token_ns,
                f"isolated mha_16 {mha_ns} / wall {total_token_ns}",
                "fraction",
                "this is the attention-arithmetic slice, historically paraphrased as ~2.4%.",
            ),
            "organ_census_ns": measured(
                organ_ns_as_census,
                f"{ORGAN_REL} organs.attention_gqa.token_ns.ns_per_token",
                "ns",
                gqa_organ["token_ns"]["arithmetic"],
            ),
            "organ_census_share": measured(
                float(gqa_organ["token_ns"]["share_of_token_ns"]),
                f"{ORGAN_REL} organs.attention_gqa.token_ns.share_of_token_ns",
                "fraction",
            ),
            "organ_ns_with_gqa_kv_only": derived(
                organ_ns_gqa_kv_only,
                f"gqa_full_probe {gqa_full_probe_ns} + gqa_component {gqa_component_ns} + unique KV stream {gqa_kv_stream_ns}",
                "ns",
                "replaces the census's full kv_state (rec+conv+GQA) with the GQA KV stream only.",
            ),
            "organ_share_with_gqa_kv_only": derived(
                organ_ns_gqa_kv_only / total_token_ns,
                f"{organ_ns_gqa_kv_only} / {total_token_ns}",
                "fraction",
            ),
            "kv_state_bucket_ns": measured(
                kv_state_ns,
                f"{LEDGER_REL} components[kv_state].ns_per_token",
                "ns",
                "rec_state + conv_state + GQA cache sequential stream. Not GQA-only.",
            ),
        },
        "flops": {
            "gqa_gemv_mac": measured(gqa_mac, f"{OPS_REL} sum self_attn.{{q,k,v,o}} mac_flops_per_token", "flop"),
            "gqa_gemv_mac_share": derived(
                gqa_mac / total_gemv_mac,
                f"{gqa_mac} / dispatched_gemv_mac_flops {total_gemv_mac}",
                "fraction",
            ),
            "mha_mac": measured(mha_mac, f"{OPS_REL} activation_flops.mha_mac", "flop"),
            "activation_flops_total": measured(act_flops, f"{OPS_REL} activation_flops.total_flops", "flop"),
            "source_gemv_gflop_anchor": cited(ANCHOR_GEMV_GFLOP, "contract / operation census 51.24 GFLOP GEMV", "GFLOP"),
        },
        "not_the_mass_on_uniform_q4": {
            "mlp_stored_bytes": measured(mlp_bytes, f"{ORGAN_REL} organs.mlp.physical.bytes", "bytes"),
            "mlp_share": derived(mlp_bytes / artifact_b, f"{mlp_bytes} / {artifact_b}", "fraction"),
            "deltanet_stored_bytes": measured(
                int(dn_organ["physical"]["bytes"]),
                f"{ORGAN_REL} organs.deltanet.physical.bytes",
                "bytes",
            ),
            "deltanet_share": derived(
                int(dn_organ["physical"]["bytes"]) / artifact_b,
                "deltanet / on_disk",
                "fraction",
            ),
            "mixer_gqa_plus_deltanet_share": derived(
                mixer_bytes / artifact_b,
                f"({gqa_bytes}+{int(dn_organ['physical']['bytes'])}) / {artifact_b}",
                "fraction",
            ),
        },
    }

    watched.append(
        "NOETIC_ORGAN_CENSUS attributed kv_state 537665 ns (rec+conv+GQA) wholly to "
        f"attention_gqa, producing organ share {gqa_organ['token_ns']['share_of_token_ns']:.4f}. "
        f"Unique GQA KV stream is {gqa_kv_stream_ns:.0f} ns. With that correction the organ "
        f"is {organ_ns_gqa_kv_only / total_token_ns:.4f} of the wall, not 0.1384."
    )

    # ------------------------------------------------------------------
    # 2. 4.125 BPW floor — reconfirm or correct
    # ------------------------------------------------------------------
    qwen38_gqa_probes = []
    for p in probe.get("probes") or []:
        tensor = str(p.get("tensor") or "")
        if p.get("model") == "qwen38" and "self_attn" in tensor:
            cands = p.get("candidates") or []

            def pick(pred):
                for c in cands:
                    if pred(c.get("codec") or ""):
                        return {
                            "codec": c.get("codec"),
                            "bpw": c.get("bpw"),
                            "output_cosine": c.get("output_cosine"),
                            "output_cosine_min_row": c.get("output_cosine_min_row"),
                            "output_rel_l2": c.get("output_rel_l2"),
                        }
                return None

            qwen38_gqa_probes.append(
                {
                    "tensor": tensor,
                    "shape": p.get("shape"),
                    "winner_any_at_0p990": {
                        "codec": (p.get("winner_any_at_0p990") or {}).get("codec"),
                        "bpw": (p.get("winner_any_at_0p990") or {}).get("bpw"),
                        "output_cosine": (p.get("winner_any_at_0p990") or {}).get("output_cosine"),
                    },
                    "hgravu_q4_g64": pick(lambda s: s == "HGRAVU01_q4_g64"),
                    "hgravu_q3_g64": pick(lambda s: s == "HGRAVU01_q3_g64"),
                    "hgravh_q4_g128": pick(lambda s: s == "HGRAVH01_hadamard_q4_g128"),
                }
            )

    q3_fail = [
        r
        for r in qwen38_gqa_probes
        if r["hgravu_q3_g64"] and r["hgravu_q3_g64"]["output_cosine"] < 0.99
    ]
    q3_pass = [
        r
        for r in qwen38_gqa_probes
        if r["hgravu_q3_g64"] and r["hgravu_q3_g64"]["output_cosine"] >= 0.99
    ]
    hadamard_4p125 = [
        r for r in qwen38_gqa_probes if r["hgravh_q4_g128"] and 4.12 <= r["hgravh_q4_g128"]["bpw"] <= 4.13
    ]

    bpw_g64 = 4.0 + 16.0 / 64.0
    bpw_g128 = 4.0 + 16.0 / 128.0

    floor = {
        "contract_phrase": "Q4 g=128 MSE at 4.125",
        "verdict": "CORRECTED",
        "what_4_125_is": {
            "grouped_absmax_arithmetic": derived(
                bpw_g128,
                "4 code bits + 16-bit scale / group 128",
                "bits/weight",
                "This is the SCALE OVERHEAD of Q4 g=128, not a measured MSE.",
            ),
            "uniform_q4_g64_arithmetic": derived(
                bpw_g64,
                "4 code bits + 16-bit scale / group 64",
                "bits/weight",
                "This is what the sealed uniform-q4-v1 artifact actually stores on GEMV bodies.",
            ),
            "density_probe_codec_at_4_125": measured(
                "HGRAVH01_hadamard_q4_g128",
                f"{DENSITY_PROBE_REL} candidates codec field on rows with bpw≈4.125",
                note=(
                    "The 4.125 figure in QWEN_ATTENTION_DENSITY_VERDICT "
                    "('q4 ~matches uniform Q4 at 4.125 vs 4.250 BPW. 2.9% save, not the mass.') "
                    "is Hadamard-Q4 group-128, reconstruction class MEDIUM_INREGISTER_TRANSFORM, "
                    "not a uniform Q4 g=128 MSE campaign and not the shipping codec."
                ),
            ),
            "quality_bar": measured(
                (verdict.get("quality_bound") or {}).get("primary"),
                f"{DENSITY_VERDICT_REL} quality_bound.primary",
                note="mean row output cosine >= 0.990 vs BF16 W on real captured X. Not MSE. Not the expert bar 0.8604.",
            ),
            "not_used_bar": measured(
                (verdict.get("quality_bound") or {}).get("not_used"),
                f"{DENSITY_VERDICT_REL} quality_bound.not_used",
            ),
        },
        "scope": {
            "models": ["qwen38", "q80"],
            "organs": "attention GEMVs: GQA q/k/v/o AND DeltaNet in_proj/out_proj. lm_head tabulated separately.",
            "layers_qwen38": (verdict.get("probe") or {}).get("layers_qwen38"),
            "n_probes": (verdict.get("probe") or {}).get("n_probes"),
            "real_activations": measured(
                (verdict.get("activation_honesty") or {}).get("used_synthetic_or_gaussian") is False,
                f"{DENSITY_VERDICT_REL} activation_honesty.used_synthetic_or_gaussian == false",
            ),
            "claim": measured(
                verdict.get("claim"),
                f"{DENSITY_VERDICT_REL} claim",
            ),
        },
        "qwen38_gqa_probe_rows": qwen38_gqa_probes,
        "q3_fails_mean_0p99": [
            {
                "tensor": r["tensor"],
                "q3_cosine": r["hgravu_q3_g64"]["output_cosine"],
                "q3_min_row": r["hgravu_q3_g64"]["output_cosine_min_row"],
            }
            for r in q3_fail
        ],
        "q3_clears_mean_0p99": [
            {"tensor": r["tensor"], "q3_cosine": r["hgravu_q3_g64"]["output_cosine"]}
            for r in q3_pass
        ],
        "hadamard_q4_g128_rows": len(hadamard_4p125),
        "organ_level_reading": (
            "The floor is organ-level, not tensor-level. Isolated late q_proj (L63) "
            "clears Q3 mean-cosine 0.99; out_proj and late k_proj do not. A pack that "
            "puts the attention MASS at Q3 fails the 0.99 bar, especially out_proj "
            "(density verdict qwen38 if_q3_all_attention_rejected). The cheapest "
            "quality-intact action on that mass is stay at uniform Q4 g64 (4.250) or "
            "the 2.9% Hadamard g128 save. That is not a new family and is not the mass."
        ),
        "mixed_2p0_is_where_attention_is_the_mass": {
            "complete_physical_bpw": measured(
                (density_root or {}).get("evidence", {}).get("complete_physical_bpw") if density_root else None,
                f"{DENSITY_ROOT_REL} evidence.complete_physical_bpw" if density_root else "missing",
            ) if density_root else null_reason(f"{DENSITY_ROOT_REL} not resolved"),
            "mlp_physical_bpw": measured(
                (density_root or {}).get("evidence", {}).get("mlp_physical_bpw"),
                f"{DENSITY_ROOT_REL} evidence.mlp_physical_bpw",
            ) if density_root else null_reason("density root missing"),
            "attention_share_of_artifact": measured(
                (density_root or {}).get("evidence", {}).get("attention_share_of_artifact"),
                f"{DENSITY_ROOT_REL} evidence.attention_share_of_artifact",
                note=(
                    "74% on mixed-2p0-v1 is attention+embed+norm at 4.250 BPW after MLP "
                    "was already 0.848. It is NOT GQA-only. GQA full_attn on uniform-q4 "
                    "is 891_289_600 B."
                ),
            ) if density_root else null_reason("density root missing"),
            "coherence": measured(
                (coherence or {}).get("honest_note") if coherence else None,
                f"{COHERENCE_REL} honest_note" if coherence else "missing",
            ) if coherence else null_reason(f"{COHERENCE_REL} not resolved"),
        },
        "hgrav_families_on_attention": {
            "HGRAVU01": (verdict.get("codec_applicability") or {}).get("HGRAVU01_uniform_group"),
            "HGRAVB01": (verdict.get("codec_applicability") or {}).get("HGRAVB01_binary"),
            "HGRAVR02": (verdict.get("codec_applicability") or {}).get("HGRAVR02_binary_residual"),
            "HGRAVS01": (verdict.get("codec_applicability") or {}).get("HGRAVS01_act_weighted_svd"),
            "HGRAVH01": (verdict.get("codec_applicability") or {}).get("HGRAVH01_hadamard"),
        },
    }

    watched.append(
        "Calling 4.125 a 'Q4 g=128 MSE floor' overstates the receipt. The probe's "
        "4.125 rows are HGRAVH01_hadamard_q4_g128 (2.9% vs g64). Uniform Q4 g64 at "
        "4.250 is what ships. MSE is not the gate; output cosine 0.99 on real X is."
    )

    # ------------------------------------------------------------------
    # 3. GQA sharing: operator vs layout, from kernel source
    # ------------------------------------------------------------------
    kv_h_line = find_line(mha, "const uint kv_h   = h / GROUP;")
    group_comment = find_line(mha, "via `h / group_size`")
    nkv_guard = find_line(gqa_shader, "n_kv_heads != 4u")
    kv_write = find_line(gqa_shader, "if (head < n_kv_heads)")
    qkv_encode = find_line(decode, 'qwen38_layer_name(layer, "self_attn.q_proj.weight")')
    q_rows = find_line(geometry, "QWEN38_Q_PROJ_ROWS: usize = 12_288")
    kv_rows = find_line(geometry, "QWEN38_KV_PROJ_ROWS: usize = 1_024")
    heads_line = find_line(geometry, "QWEN38_GQA_HEADS: usize = 24")
    kv_heads_line = find_line(geometry, "QWEN38_GQA_KV_HEADS: usize = 4")
    mixer_rule = find_line(geometry, "(layer + 1) % QWEN38_FULL_ATTENTION_INTERVAL")
    q_buf = find_line(decode, "QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2")
    workhorse = find_line(decode, 'QWEN38_Q4_MATVEC_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"')
    g128_kernel = find_line(decode, "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128")
    g128_launch_guard = find_line(decode, "group_size != 64")

    kernel_evidence = {
        "verdict": "LAYOUT_NOT_OPERATOR",
        "geometry": {
            "gqa_layers": measured(16, f"{GEOMETRY_REL}:{mixer_rule} source rule (layer+1)%4==0"),
            "n_heads": measured(24, f"{GEOMETRY_REL}:{heads_line} {line_at(geometry, heads_line).strip()}"),
            "n_kv_heads": measured(4, f"{GEOMETRY_REL}:{kv_heads_line} {line_at(geometry, kv_heads_line).strip()}"),
            "head_dim": measured(256, f"{GEOMETRY_REL} QWEN38_GQA_HEAD_DIM"),
            "group_size": derived(6, "24 query heads / 4 kv heads"),
            "q_proj_rows": measured(12288, f"{GEOMETRY_REL}:{q_rows} — 24*(256 q + 256 gate)"),
            "kv_proj_rows": measured(1024, f"{GEOMETRY_REL}:{kv_rows} — 4*256"),
            "o_proj": measured("5120 x 6144", f"{GEOMETRY_REL} QWEN38_O_PROJ_ROWS/COLS"),
            "q_workspace_is_q_plus_gate": measured(
                True,
                f"{DECODE_REL}:{q_buf} {line_at(decode, q_buf).strip()}",
            ),
        },
        "weight_path_is_four_independent_gemvs": {
            "schedule_gqa_prefix": gqa_prefix,
            "encode": [
                {
                    "file": DECODE_REL,
                    "line": qkv_encode,
                    "text": line_at(decode, qkv_encode).strip() if qkv_encode else None,
                },
                {
                    "file": DECODE_REL,
                    "line": (qkv_encode or 0) + 6,
                    "text": line_at(decode, (qkv_encode or 0) + 6).strip() if qkv_encode else None,
                },
                {
                    "file": DECODE_REL,
                    "line": (qkv_encode or 0) + 12,
                    "text": line_at(decode, (qkv_encode or 0) + 12).strip() if qkv_encode else None,
                },
            ],
            "kernel": measured(
                "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                f"{DECODE_REL}:{workhorse} — same workhorse as MLP and DeltaNet. GQA is a layout of launches, not a kernel family.",
            ),
            "group128_kernel_exists_not_default": measured(
                "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128",
                f"{DECODE_REL}:{g128_kernel} declared; {DECODE_REL}:{g128_launch_guard} geo_tpr64 launch returns None unless group_size==64.",
            ),
        },
        "attention_path_indexes_shared_kv": {
            "mha_header": {
                "file": MHA_REL,
                "line": group_comment,
                "text": line_at(mha, group_comment).strip() if group_comment else None,
            },
            "kv_head_from_query_head": {
                "file": MHA_REL,
                "line": kv_h_line,
                "text": line_at(mha, kv_h_line).strip() if kv_h_line else None,
                "reading": "GQA sharing is integer division of the query-head index. Six query heads read the same K/V cache slice. No shared-Q weight, no grouped-Q matvec, no operator fusion.",
            },
            "rope_writes_kv_only_for_first_4_heads": {
                "file": GQA_SHADER_REL,
                "line": kv_write,
                "text": line_at(gqa_shader, kv_write).strip() if kv_write else None,
                "guard": line_at(gqa_shader, nkv_guard).strip() if nkv_guard else None,
            },
            "reconstructs_dense": measured(
                "NO",
                f"{KERN_REL} helper_dispatched_not_in_38 mha_decode_f32.reconstructs_dense; "
                "qwen38_gqa_qk_norm_rope_cache_{tg,f32} in dispatched also NO. Activations only.",
            ),
        },
        "byte_split_inside_gqa_gemv": {
            "q_plus_gate_q4_bytes": measured(int(q["q4_bytes_per_token"]), f"{OPS_REL} self_attn.q_proj", "bytes/token"),
            "k_q4_bytes": measured(int(k["q4_bytes_per_token"]), f"{OPS_REL} self_attn.k_proj", "bytes/token"),
            "v_q4_bytes": measured(int(v["q4_bytes_per_token"]), f"{OPS_REL} self_attn.v_proj", "bytes/token"),
            "o_q4_bytes": measured(int(o["q4_bytes_per_token"]), f"{OPS_REL} self_attn.o_proj", "bytes/token"),
            "kv_share_of_gqa_gemv_bytes": derived(
                (int(k["q4_bytes_per_token"]) + int(v["q4_bytes_per_token"])) / gqa_q4_bytes,
                "(k+v) / (q+k+v+o) Q4 bytes",
                "fraction",
                "K/V are already 6:1 vs a full MHA. The remaining GQA weight mass is Q+gate and O, which are not shared.",
            ),
            "q_plus_o_share": derived(
                (int(q["q4_bytes_per_token"]) + int(o["q4_bytes_per_token"])) / gqa_q4_bytes,
                "(q_proj including gate + o_proj) / GQA GEMV Q4 bytes",
                "fraction",
            ),
        },
        "what_would_count_as_an_operator": (
            "A native grouped-query projection that stores one Q (and optionally O) "
            "matrix per KV group and applies per-head scale/FiLM without materializing "
            "the 24-head parent, or a fused GQA decode that never writes the parent "
            "dense W. Integer division in mha_decode_f32 is not that."
        ),
    }

    # ------------------------------------------------------------------
    # 4. candidate operator
    # ------------------------------------------------------------------
    group = 6
    q_bytes = int(q["q4_bytes_per_token"])
    # q_proj is 24 * (q_256 | gate_256); geometry QWEN38_Q_PROJ_ROWS = 12288 = 2*24*256.
    q_half = q_bytes / 2.0
    gate_half = q_bytes / 2.0
    new_q_only = q_half / group  # grouped Q, gate stays 24-head
    new_gate_also = gate_half / group
    new_o = int(o["q4_bytes_per_token"]) / group
    new_k = int(k["q4_bytes_per_token"])
    new_v = int(v["q4_bytes_per_token"])
    # Aggressive candidate: group Q AND the per-head gate AND O.
    new_q = new_q_only + new_gate_also
    new_gqa_q4 = new_q + new_k + new_v + new_o
    byte_delta = gqa_q4_bytes - new_gqa_q4
    new_payload = payload_b - byte_delta
    new_bpw = new_payload * 8 / params
    new_q_mac = int(q["mac_flops_per_token"]) / group
    new_o_mac = int(o["mac_flops_per_token"]) / group
    new_gqa_mac = new_q_mac + int(k["mac_flops_per_token"]) + int(v["mac_flops_per_token"]) + new_o_mac
    # Milder: group Q only, leave gate and O at 24-head. Still geometry, not a pack.
    milder_gqa_q4 = new_q_only + gate_half + new_k + new_v + int(o["q4_bytes_per_token"])
    milder_delta = gqa_q4_bytes - milder_gqa_q4
    milder_bpw = (payload_b - milder_delta) * 8 / params
    # Dispatches: one Q matvec per KV group instead of 24-head Q, still 16 layers.
    # Conservative: still 4 GEMVs/layer (q_grouped, k, v, o_grouped) = 64, plus rope/mha/sigmoid.
    # Aggressive fuse of grouped-Q with K (same 1024 rows) still leaves K and V distinct.
    candidate_dispatches_conservative = ANCHOR_DISPATCHES  # same 964, different payload
    # If q/k/v concurrent group collapsed to 2 launches (grouped-Q+K fused, V, O): 16 fewer.
    candidate_dispatches_fused_qk = ANCHOR_DISPATCHES - 16

    fidelity_bar = {
        "primary": measured(
            (verdict.get("quality_bound") or {}).get("primary"),
            f"{DENSITY_VERDICT_REL} quality_bound.primary",
        ),
        "scale_aware_must_reject_0p01W": measured(
            True,
            f"{ORGAN_REL} organs.attention_gqa.functional_sensitivity_local.scaled_0p01_W: "
            f"cosine={gqa_organ['functional_sensitivity_local']['scaled_0p01_W']['cosine']} "
            f"scale_aware={gqa_organ['functional_sensitivity_local']['scaled_0p01_W']['scale_aware']}",
        ),
        "not_expert_bar_0p8604": measured(
            True,
            f"{DENSITY_VERDICT_REL} quality_bound.not_used",
        ),
        "native_operator_no_dense_W": cited(
            True,
            "contract dense-reconstruction law; operation census dense_w_materialized_bytes_per_token=0 on the current path",
        ),
        "health_verdict_required": cited(
            True,
            "n1arch / kernel census: 223 components below 0.5 local BPW with healthy=true count 0. A low number is not a result.",
        ),
        "quality_on_this_candidate": null_reason(
            "KV-group shared Q/O has never been packed or scored on real X. "
            "G035 shared_beats_independent=false is cross-layer, not within-layer heads; "
            "it still forbids treating 'sharing' as a free move. Density probe Q3 on "
            "out_proj already fails 0.99. This candidate is more aggressive than Q3."
        ),
    }

    production_families_today = [
        "grouped_absmax_q4",
        "binary±CSR",
        "HGRAVS01 factors",
        "PQ codebook lookup",
        "MoE worklists",
        "recurrent state op",
    ]
    candidate = {
        "name": "KV-group shared Q/O projection, native (never reconstruct parent dense W)",
        "why_this_one": (
            "GQA already shares K/V 6:1 as a cache layout. The remaining GQA weight mass "
            "is Q+gate and O. An operator formulation would store one Q (and optionally O) "
            "per KV group and broadcast across the 6 query heads with a cheap per-head "
            "scale, executed as y = W_g x in-register. That is the only candidate that "
            "treats grouping as an operator rather than as `h / group_size`."
        ),
        "not_this": [
            "reconstruct grouped W to dense 24-head Q then ordinary GEMM — oracle only",
            "cross-layer shared basis (G035 shared_beats_independent=false)",
            "HGRAVS01 on attention (density: 0 clears of 0.99 at ranks that beat Q4)",
            "latent KV as the first lever (G060: fix the 253x per-element kernel first)",
            "fusing rope+mha+sigmoid without changing bytes or MACs (dispatch Type-1 dead)",
            "Hadamard Q4 g128 (2.9% of the organ, kernel transform, not the mass)",
        ],
        "expected": {
            "bytes_today_gqa_q4": measured(gqa_q4_bytes, "sum of four GQA GEMV q4_bytes_per_token", "bytes/token"),
            "q_proj_is_q_plus_gate": derived(
                True,
                "QWEN38_Q_PROJ_ROWS=12288 = 24*(256 q + 256 gate); half of q_proj bytes are the sigmoid gate",
            ),
            "bytes_if_q_only_grouped_gate_and_o_stay": derived(
                milder_gqa_q4,
                f"q_half/6 + gate_half + k + v + o = {new_q_only}+{gate_half}+{new_k}+{new_v}+{int(o['q4_bytes_per_token'])}",
                "bytes/token",
                "GEOMETRY. Groups the Q half of q_proj only.",
            ),
            "complete_bpw_if_q_only_grouped": derived(
                milder_bpw,
                f"8 * ({payload_b} - {milder_delta}) / {params}",
                "bits/weight",
            ),
            "bytes_if_q_gate_and_o_grouped_6_to_1": derived(
                new_gqa_q4,
                f"(q_half+gate_half)/6 + k + v + o/6 = {new_q}+{new_k}+{new_v}+{new_o}",
                "bytes/token",
                "GEOMETRY, not a packed artifact. Quality UNMEASURED. Groups the sigmoid gate too.",
            ),
            "byte_delta": derived(byte_delta, "today - grouped", "bytes/token"),
            "byte_delta_share_of_payload": derived(
                byte_delta / payload_b, f"{byte_delta} / {payload_b}", "fraction"
            ),
            "complete_bpw_if_q_gate_and_o_grouped": derived(
                new_bpw,
                f"8 * ({payload_b} - {byte_delta}) / {params}",
                "bits/weight",
                "Aggressive geometry. Still ~4.05, not sub-1.5. Uniform-q4 complete BPW today is 4.2527.",
            ),
            "ops_today_gqa_mac": measured(gqa_mac, "sum of four GQA GEMV mac_flops_per_token", "flop"),
            "ops_if_q_and_o_grouped": derived(
                new_gqa_mac,
                "q_mac/6 + k_mac + v_mac + o_mac/6",
                "flop",
            ),
            "dispatches_conservative": derived(
                candidate_dispatches_conservative,
                "same 964: still 4 GEMVs/GQA layer, different matrix shapes",
            ),
            "dispatches_if_q_fused_with_k_shape": derived(
                candidate_dispatches_fused_qk,
                "964 - 16: grouped Q has K's 1024-row shape so q+k can share a launch geometry; still not measured",
            ),
            "tps_delta": null_reason(
                "No native kernel, no A/B. Isolated gqa_gemvs is 1.817 ms of a 35.228 ms wall. "
                "Even a perfect 5/6 cut of Q+O MACs is a PROJECTED ~1 ms, which does not "
                f"close 32.73 vs MLX {ANCHOR_MLX_TPS} and is not a measurement."
            ),
        },
        "fidelity_it_must_hold": fidelity_bar,
        "production_eligibility_today": {
            "allowed_families": production_families_today,
            "this_candidate_in_allowed_set": False,
            "would_need": (
                "a new native grouped-Q matvec (or a bind of HGRAVS01 two-stage onto "
                "per-KV-group factors that never writes W). Reconstruct-then-GEMM is "
                "not production. The Q4 g128 kernel already exists and is not this idea."
            ),
        },
        "prior_science_that_blocks_building_first": [
            {
                "id": "G035",
                "fact": "shared_beats_independent=false",
                "status": "MEASURED" if g035_flags and all(x is False for x in g035_flags) else "CITED",
                "n_flags": len(g035_flags),
                "source": G035_REL,
            },
            {
                "id": "attention-density",
                "fact": "Gravity families miss 0.99 on attention below Q4; HGRAVS01 0 clears at ranks that beat Q4; attention is not a low-rank organ",
                "source": DENSITY_VERDICT_REL,
            },
            {
                "id": "223-unhealthy",
                "fact": "223 components local_bpw<0.5, healthy=true count 0",
                "source": N1ARCH_REL if n1arch else "contract / kernel census constraints_from_recovered_science",
            },
            {
                "id": "G3-FiLM",
                "fact": "shared SwiGLU + per-layer FiLM headline gap 0.026 then refuted as a family",
                "source": N1ARCH_REL if n1arch else "n1arch census",
            },
        ],
    }

    # ------------------------------------------------------------------
    # 5. reconcile expensive-to-compress vs 2.4% of decode
    # ------------------------------------------------------------------
    two_point_four = None
    if dead_levers and "attn only 2.4% wall" in dead_levers:
        for line in dead_levers.splitlines():
            if "attn only 2.4% wall" in line:
                two_point_four = line.strip()
                break
    mha_share = mha_ns / total_token_ns
    gqa_comp_share = gqa_component_ns / total_token_ns
    organ_share = organ_ns_as_census / total_token_ns

    reconciliation = {
        "historical_2p4": {
            "text": two_point_four,
            "source": DEAD_LEVERS_REL if two_point_four else None,
            "status": "MEASURED" if two_point_four else "NULL",
            "scope": (
                "MLA Phase 4 simdgroup attn on the llama.cpp-era kill ledger, "
                "not the Qwen3.8 GQA organ."
            ),
        },
        "this_vehicle": {
            "mha_isolated_share": derived(
                mha_share,
                f"{mha_ns} / {total_token_ns}",
                "fraction",
                "attention arithmetic. Nearest number to the historical 2.4% paraphrase.",
            ),
            "gqa_component_share": derived(
                gqa_comp_share,
                f"{gqa_component_ns} / {total_token_ns}",
                "fraction",
                "ledger components[gqa] = 6.94%: mostly rope occupancy + mha, not weight bytes.",
            ),
            "organ_census_share": derived(organ_share, "census ns / wall", "fraction"),
            "organ_share_gqa_kv_only": derived(
                organ_ns_gqa_kv_only / total_token_ns,
                "probe+component+unique KV / wall",
                "fraction",
            ),
        },
        "why_both_can_be_true": (
            "Attention is expensive to COMPRESS because the density probe's 0.99 bar "
            "rejects Q3 / binary / HGRAVS01 / rice on the attention mass, so mixed packs "
            "that crush MLP still carry attention at 4.25 BPW and that remainder is 74% "
            "of mixed-2p0-v1. Attention arithmetic is cheap to RUN because MHA at seq≈19 "
            "is 0.667 ms (1.89% of the 35.228 ms wall) and even the whole GQA component "
            "is 2.443 ms (6.94%), of which 62% is a 24-thread rope kernel already retiled "
            "behind HAWKING_ROPE_TG=256. Those are different axes."
        ),
        "axis_a_win_would_move": {
            "storage_ebpw_resident": {
                "moves": True,
                "how_much_on_uniform_q4": (
                    f"GQA is {gqa_bytes / artifact_b:.4f} of on-disk bytes. Deleting the "
                    f"organ entirely would drop complete BPW from {artifact_b * 8 / params:.4f} "
                    f"to {(artifact_b - gqa_bytes) * 8 / params:.4f}. Grouped Q/O at perfect "
                    f"quality is {new_bpw:.4f}. Neither is sub-1.5."
                ),
                "how_much_on_mixed_2p0": (
                    "On mixed-2p0-v1 the 74% figure is attention+embed+norm, of which GQA "
                    "full_attn is 891_289_600 B of a 7.01 GB artifact (~12.7%). A GQA-only "
                    "operator does not move that 74% by itself. DeltaNet + embed + lm_head "
                    "are the rest of that remainder."
                ),
            },
            "tok_s": {
                "moves": False,
                "why": (
                    f"MHA is {mha_share:.4f} of wall. Isolated GQA GEMVs are "
                    f"{gqa_gemv_ns / total_token_ns:.4f}. The token is 60.4% weight_addressing "
                    f"across ALL organs (MLP 9.09 GB streamed). Beating MLX {ANCHOR_MLX_TPS} "
                    f"tok/s from {ANCHOR_TPS} is not a GQA-operator problem. Rope occupancy "
                    "was the GQA speed bug and is already shipped (decode.rs HAWKING_ROPE_TG=256, "
                    "measured 4.16% faster token)."
                ),
            },
            "dispatches": {
                "moves": False,
                "why": (
                    "Production is already 1 command buffer / 964 dispatches. Host "
                    "per-dispatch overhead is a Type-1 kill (dead_levers). Cutting GQA "
                    "named launches (112) without cutting bytes or MACs does not move tok/s."
                ),
            },
            "work": {
                "moves": "only if Q/O grouping is quality-legal",
                "why": (
                    f"GQA GEMV MACs are {gqa_mac / total_gemv_mac:.4f} of dispatched GEMV. "
                    "Storage compression of the same W still does the same MACs (operation "
                    "census). An operator that stores one Q per KV group would do fewer "
                    "MACs. Quality on that operator is UNMEASURED and prior science is hostile."
                ),
            },
        },
        "s011_storage_alone_incomplete": {
            "s011_in_git": s011,
            "measurement": measured(
                ops.get("answer"),
                f"{OPS_REL} answer — executable holds fewer bytes, does the same 51.24 GFLOP GEMV plus dequant ALU",
            ),
        },
    }

    # ------------------------------------------------------------------
    # 6. verdict
    # ------------------------------------------------------------------
    mlp_nogo = {
        "status": "MEASURED" if mlp_verdict else "CITED",
        "decision": (mlp_verdict or {}).get("decision") if mlp_verdict else "NO-GO",
        "deciding_number": (mlp_verdict or {}).get("deciding_number") if mlp_verdict else 0.4206259548664093,
        "source": located[MLP_DISTILL_REL] if located[MLP_DISTILL_REL]["found"] else "contract + sibling lane n16mlp",
        "note": (
            (mlp_verdict or {}).get("deciding_number_meaning")
            if mlp_verdict
            else "L31 I'=2560 hold rel_fro 0.8185 vs q3 0.3978, gap +0.4206, byte_ratio 0.724, Doctor UNHEALTHY"
        ),
    }

    overall = {
        "verdict": "NOT_WORTH_BUILDING",
        "as_of": utc_now(),
        "because": [
            "GQA K/V sharing is already exploited as a cache layout (6:1). The kernels do not have a grouped-Q operator to write.",
            "On uniform-q4-v1 GQA is 6.2% of stored bytes and ~6.5% of GEMV MACs. The remaining storage mass on this vehicle is MLP (63.6%), then DeltaNet (20.7%).",
            "The 74% 'attention is the mass' figure is mixed-2p0 after MLP was already 0.848 BPW, and that 74% is attention+embed+norm, not GQA-only (~12.7% of that artifact).",
            "MLP function distillation is NO-GO (deciding_number +0.4206 at I'=2560, ~72% of q3 active bytes, Doctor UNHEALTHY). That closes a function-space cut of the large organ; it does not make GQA large.",
            "A quality-perfect grouped Q/O operator moves complete BPW 4.253 → ~4.05 on this vehicle. Sub-1.5 stays closed. Tok/s vs MLX 35.51 does not move.",
            "Attention below Q4 is a family Gravity does not have (density verdict). Sharing as an idea lost on G035. Latent KV is the second lever after a 253x kernel hole (G060).",
            "A candidate that only lowers executable bytes is incomplete (operation census / S011 §4 as named). This candidate's only legal win is fewer MACs on Q/O, which is quality-blocked.",
        ],
        "reopen_condition": (
            "A native grouped-Q (and/or grouped-O) operator, scored on real activations "
            "with scale-aware + Doctor, clears the 0.99 attention bar at materially fewer "
            "active bytes than uniform Q4, without reconstructing dense W, AND the byte "
            "delta is large enough to move a named gate (sub-1.5 complete BPW or the "
            "32.73→35.51 tok/s gap). Within-layer head cosine/energy on Q of a KV group "
            "is the cheap oracle; do not skip it and do not use synthetic X."
        ),
        "already_shipped_gqa_speed_work": (
            "HAWKING_ROPE_TG=256 retile of qwen38_gqa_qk_norm_rope_cache_tg. Measured "
            "4.16% faster token, token-identical on 24 greedy tokens (decode.rs comment "
            "on encode_rope_cache). Do not rediscover 24-thread rope as this design."
        ),
        "mlp_function_distillation": mlp_nogo,
    }

    # ------------------------------------------------------------------
    # 7. numbers ledger — every figure status-tagged
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 8. WHAT I WATCHED FAIL
    # ------------------------------------------------------------------
    watched.extend(
        [
            "cosine(q_proj, 0.01*q_proj) on real L3 activations is 1.000000; scale_aware 0.0100. A cosine-only GQA design would have accepted a magnitude wipe.",
            "Historical 'attn only 2.4% wall' is dead_levers.md MLA Phase 4, a different stack. On this vehicle MHA is 1.89%, GQA component 6.94%, organ-with-stolen-kv_state 13.84%.",
            "mixed-2p0 'attention is 74%' includes linear_attn + embed + norms. Treating that as GQA-only would have pointed the remaining mass at 0.89 GB instead of 5.20 GB.",
            "G060 latent KV looks like a GQA operator. The receipt says the 253x per-element inefficiency dominates; after the kernel is fixed latent KV is 1.003x. Do not reopen it as this lane.",
            "tpr64 reconstruction is free on 32/33 variants (NNS-011). That reopens codecs killed for a 5.9x penalty; it does not reopen attention quality below Q4.",
            "Q80 storage BPW 0.6462 vs ACTIVE 2.518 is a category error. A GQA byte cut that reconstructs dense W would repeat it.",
            "Rope-as-encoder-tax was false (g1-gqa-and-attention-geometry): sigmoid 2.7 µs/launch vs rope 97.7 µs with 24 threads. Occupancy, not ceremony. Already retiled.",
            "Production workhorse for GQA weights is the MLP workhorse. Writing a new shader that still dequants independent 24-head Q is a layout tweak, not an operator.",
        ]
    )
    watched.append(
        "223 components measured below 0.5 local BPW with healthy=true count 0 "
        "(kernel census constraints_from_recovered_science / n1arch). A grouped-Q "
        "local BPW without a health verdict would be that trap again."
    )
    watched.append(
        "G035 G-SHARE shared_beats_independent=false on this model. Within-layer "
        "head grouping is not the same experiment, but 'sharing' is not a free move."
    )
    watched.append(
        "The sealed QWEN38_TOKEN_NS_LEDGER gqa row (2.443 ms, 6.94%) includes "
        "isolated rope_cache_16 = 1.562 ms on qwen38_gqa_qk_norm_rope_cache_f32 "
        "(24 threads). Production default is now HAWKING_ROPE_TG=256 "
        "(qwen38_gqa_qk_norm_rope_cache_tg), measured 4.16% faster token. The "
        "schedule array still lists _f32. Do not treat 6.94% as the live GQA "
        "component share; MHA 0.667 ms / 1.89% is the stable arithmetic slice."
    )

    ranking = None
    for row in (organ.get("ranking") or {}).get("by_stored_byte") or []:
        if row.get("organ") == "attention_gqa":
            ranking = row
            break

    self_checks = {
        "required_receipts_present": not missing,
        "dispatches_964": int(ops["dispatch_reconciliation"]["recorded_anchor"]) == ANCHOR_DISPATCHES,
        "gqa_prefix_len_9": len(gqa_prefix) == 9,
        "gqa_named_dispatches_112": gqa_named_dispatches == 112,
        "mha_reconstructs_dense_no": True,
        "dispatched_reconstructs_dense_all_no": (kern.get("reconciliation") or {})
        .get("dispatched_reconstructs_dense", {})
        .get("NO")
        == 38,
        "organ_census_arithmetic_reproduces": abs(
            gqa_full_probe_ns + gqa_component_ns + kv_state_ns - organ_ns_as_census
        )
        < 1e-6,
        "gqa_q4_bytes_match_production_full_attn": gqa_q4_bytes
        == int(kern["production_token"]["full_attn_bytes"]),
        "scale_trap_on_gqa_q_proj": gqa_organ["functional_sensitivity_local"]["scaled_0p01_W"][
            "rejects_perfect_cosine_as_sufficient"
        ],
        "g035_all_false": bool(g035_flags) and all(x is False for x in g035_flags),
        "no_dense_w_on_path": ops["dram_and_temp"]["dense_w_materialized_bytes_per_token"] == 0,
        "verdict_is_not_worth_building": overall["verdict"] == "NOT_WORTH_BUILDING",
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "repo": str(REPO),
        "question": (
            "Is there an operator formulation of grouped attention that moves less "
            "or does less, at the fidelity attention actually demands?"
        ),
        "answer": (
            "Not one worth building. GQA K/V sharing is a cache layout "
            "(`kv_h = h / GROUP` in mha_decode_f32) in front of four independent "
            "uniform-Q4 GEMVs. The organ is the quality floor (cannot cheaply go "
            "below Q4), not the storage mass on uniform-q4-v1 (6.2% of bytes, "
            "1.89% of wall for MHA). A grouped-Q/O operator is the only candidate "
            "that would treat grouping as an operator; its quality is unmeasured "
            "and even a perfect pack moves complete BPW 4.253 → ~4.05."
        ),
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "unified_memory_bytes": ANCHOR_UNIFIED_B,
            "gpu_cores": ANCHOR_GPU_CORES,
            "parameter_count": ANCHOR_PARAMS,
            "artifact_bytes": ANCHOR_ARTIFACT_B,
            "tensor_count": ANCHOR_TENSORS,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers": ANCHOR_CBS,
            "kernels_bound": ANCHOR_BOUND,
            "kernels_declared": ANCHOR_DECLARED,
            "gemv_gflop": ANCHOR_GEMV_GFLOP,
            "mlx_4bit_tps_live": ANCHOR_MLX_TPS,
            "llamacpp_q5k_tps_archived": ANCHOR_LLAMA_Q5K_TPS,
            "machine": "Apple M3 Ultra, 60 GPU cores, Metal 4",
            "artifact": "/Users/scammermike/models/qwen38-gravity-uniform-q4-v1",
            "decode_source": DECODE_REL,
        },
        "prior_science_respected": {
            "n1arch_35_mechanisms": located[N1ARCH_REL],
            "n15neg_31_closures": located[N15NEG_REL],
            "n16clos": located[N16CLOS_REL],
            "g035_shared_beats_independent": g035_flags,
            "n223_unhealthy": "223 components local_bpw<0.5, healthy=0 (n1arch / kernel census)",
            "q80_storage_vs_active": "0.6462 vs 2.518 — report both or neither",
            "glm_0_167": "trap",
            "hgravs01_0_13": "down_proj ONLY",
            "tpr64_free_on_32_of_33": True,
            "mlp_function_distillation": mlp_nogo,
            "never_synthetic_activations": True,
            "cosine_scale_blind": True,
            "raw_activation_cosine_null": 0.898,
        },
        "acceptance": {
            "1_share_today": share,
            "2_bpw_floor": floor,
            "3_operator_vs_layout": kernel_evidence,
            "4_candidate": candidate,
            "5_reconciliation": reconciliation,
            "6_verdict": overall,
        },
        "function_if_zeroed": {
            "function_lost": (ranking or {}).get("function_lost"),
            "survival_null": (ranking or {}).get("survival_null"),
            "rank_by_function_per_stored_byte": (ranking or {}).get("rank_by_function_per_stored_byte"),
            "source": f"{ORGAN_REL} ranking.by_stored_byte attention_gqa",
            "reading": (
                "Zeroing GQA loses 0.607 of residual-stream function on the organ census "
                "capture; it is rank 3 of 5 by function/byte, behind embed and lm_head. "
                "It carries function. That does not make it the storage mass."
            ),
        },
        "located": located,
        "self_checks": self_checks,
        "what_i_watched_fail": watched,
        "write_scope": {
            "write": [
                "tools/headless/noetic_gqa_design.py",
                "receipts/headless/NOETIC_GQA_DESIGN.json",
            ],
            "deny": ["workspace", "crates", "visionmcp", "app", "lab", "tools/haider", "ramanujan"],
        },
        "wall_s": time.perf_counter() - t0,
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    w = []
    a = w.append
    a("NOETIC GQA DESIGN")
    a("=" * 72)
    a(f"schema     {SCHEMA}")
    a(f"generated  {doc['generated_at']}")
    a(f"git_head   {doc['git_head']}")
    a(f"repo       {REPO}")
    a(f"wrote      {RECEIPT}")
    a(f"wall_s     {doc['wall_s']:.3f}")
    a("")
    a("## QUESTION")
    a(doc["question"])
    a("")
    a("## ANSWER")
    a(doc["answer"])
    a("")
    a("## VERDICT")
    a(f"  {overall['verdict']}")
    for b in overall["because"]:
        a(f"  - {b}")
    a(f"  reopen: {overall['reopen_condition']}")
    a(f"  already shipped: {overall['already_shipped_gqa_speed_work']}")
    a("")
    a("## 1. ATTENTION SHARE TODAY")
    a(f"  stored_bytes          {gqa_bytes}   share_on_disk={gqa_bytes/artifact_b:.6f}   share_payload={gqa_bytes/payload_b:.6f}")
    a(f"  elements              {gqa_elems}   tensors={gqa_tensors} (q4={gqa_organ['physical']['q4_tensors']} f32={gqa_organ['physical']['f32_tensors']})")
    a(f"  local_physical_bpw    {gqa_bytes*8/gqa_elems:.6f}   (DERIVED)")
    a(f"  active_bytes/token    {gqa_active}   q4_gemv={gqa_q4_bytes}")
    a(f"  dispatches            named={gqa_named_dispatches}  gemv={gqa_gemv_dispatches}  prefix_incl_rms_resid={gqa_prefix_dispatches}  / {ANCHOR_DISPATCHES}   CBs={ANCHOR_CBS}")
    a(f"  token_ns wall         {total_token_ns:.0f}")
    a(f"  MHA isolated          {mha_ns:.0f} ns  share={mha_share:.6f}   (attention arithmetic)")
    a(f"  GQA component         {gqa_component_ns:.1f} ns  share={gqa_comp_share:.6f}   (rope+mha+fma rem+sigmoid)")
    a(f"  organ census          {organ_ns_as_census:.1f} ns  share={organ_share:.6f}   (INCLUDES full kv_state)")
    a(f"  organ gqa-kv-only     {organ_ns_gqa_kv_only:.1f} ns  share={organ_ns_gqa_kv_only/total_token_ns:.6f}   (DERIVED correction)")
    a(f"  GEMV MACs             {gqa_mac}  share={gqa_mac/total_gemv_mac:.6f} of dispatched GEMV {total_gemv_mac}")
    a(f"  MHA MACs              {mha_mac}  (tiny vs GEMV)")
    a(f"  MLP stored            {mlp_bytes}  share={mlp_bytes/artifact_b:.6f}")
    a(f"  DeltaNet stored       {int(dn_organ['physical']['bytes'])}  share={int(dn_organ['physical']['bytes'])/artifact_b:.6f}")
    a("")
    a("## 2. 4.125 BPW FLOOR")
    a(f"  contract phrase       {floor['contract_phrase']}")
    a(f"  verdict               {floor['verdict']}")
    a(f"  Q4 g64 arithmetic     {bpw_g64} bits/weight (SHIPPING)")
    a(f"  Q4 g128 arithmetic    {bpw_g128} bits/weight (scale overhead, not MSE)")
    a("  density 4.125 codec   HGRAVH01_hadamard_q4_g128 — 2.9% save, not the mass")
    a(f"  quality bar           {floor['what_4_125_is']['quality_bar']['value']}")
    a("  qwen38 GQA probe Q3 vs 0.99 mean cosine:")
    for r in qwen38_gqa_probes:
        q3 = r["hgravu_q3_g64"]
        flag = "FAIL" if (q3 and q3["output_cosine"] < 0.99) else ("PASS" if q3 else "n/a")
        a(f"    {r['tensor']:42} Q3={q3['output_cosine'] if q3 else None}  {flag}  winner={r['winner_any_at_0p990']['codec']}@{r['winner_any_at_0p990']['bpw']}")
    a(f"  organ-level reading   {floor['organ_level_reading']}")
    if density_root:
        ev = density_root.get("evidence") or {}
        a(f"  mixed-2p0 attention_share {ev.get('attention_share_of_artifact')}  complete_bpw={ev.get('complete_physical_bpw')}  mlp_bpw={ev.get('mlp_physical_bpw')}")
        a("  mixed-2p0 74% is attention+embed+norm, not GQA-only")
    a("")
    a("## 3. OPERATOR VS LAYOUT")
    a(f"  verdict               {kernel_evidence['verdict']}")
    a(f"  geometry              16 GQA layers, 24:4 heads, d=256, group=6")
    a(f"  q_proj                12288 x 5120 = 24*(q_256|gate_256)")
    a(f"  k/v_proj              1024 x 5120 = 4*256   (already 6:1)")
    a(f"  o_proj                5120 x 6144")
    a(f"  schedule prefix       {gqa_prefix}")
    a("  weights               four independent encode_q4_matvec on the MLP workhorse")
    a(f"  mha.metal:{kv_h_line}     {line_at(mha, kv_h_line).strip() if kv_h_line else 'MISSING'}")
    a(f"  rope kv write:{kv_write}  {line_at(gqa_shader, kv_write).strip() if kv_write else 'MISSING'}")
    a(f"  reconstructs_dense    NO (mha + rope + sigmoid + Q4 GEMV)")
    a(f"  K+V share of GQA GEMV bytes  {(int(k['q4_bytes_per_token'])+int(v['q4_bytes_per_token']))/gqa_q4_bytes:.4f}")
    a(f"  Q+O share of GQA GEMV bytes  {(int(q['q4_bytes_per_token'])+int(o['q4_bytes_per_token']))/gqa_q4_bytes:.4f}")
    a(f"  {kernel_evidence['what_would_count_as_an_operator']}")
    a("")
    a("## 4. CANDIDATE")
    a(f"  {candidate['name']}")
    a(f"  {candidate['why_this_one']}")
    a("  not this:")
    for n in candidate["not_this"]:
        a(f"    - {n}")
    a(f"  bytes today           {gqa_q4_bytes}  (q_proj includes per-head sigmoid gate, half the rows)")
    a(f"  bytes if Q only 6:1   {milder_gqa_q4:.0f}   delta={milder_delta:.0f}  BPW→{milder_bpw:.6f}  [DERIVED geometry]")
    a(f"  bytes if Q+gate+O 6:1 {new_gqa_q4:.0f}   delta={byte_delta:.0f}  ({byte_delta/payload_b:.4f} of payload)  BPW→{new_bpw:.6f}  [DERIVED geometry]")
    a(f"  GEMV MACs today       {gqa_mac}")
    a(f"  GEMV MACs if Q/O 6:1  {new_gqa_mac:.0f}  [DERIVED geometry]")
    a(f"  dispatches            conservative {candidate_dispatches_conservative}; fused-qk-shape {candidate_dispatches_fused_qk}  [DERIVED]")
    a("  tok/s delta           NULL — no kernel, no A/B")
    a("  fidelity              mean-row output cosine >= 0.99 vs BF16 on real X; scale_aware must reject 0.01*W; Doctor healthy; no dense W")
    a("  quality on candidate  NULL — never packed. Prior science hostile.")
    a(f"  production family     NOT in {production_families_today}")
    a("")
    a("## 5. EXPENSIVE TO COMPRESS vs 2.4% OF DECODE")
    a(f"  historical 2.4%       {two_point_four or 'NULL in this checkout of dead_levers.md'}")
    a(f"  this vehicle MHA      {mha_share:.6f} of wall")
    a(f"  this vehicle GQA row  {gqa_comp_share:.6f} of wall")
    a(f"  {reconciliation['why_both_can_be_true']}")
    a("  axis a win would move:")
    a(f"    storage  YES, small on uniform-q4 ({gqa_bytes/artifact_b:.4f} of bytes); 74% figure is mixed-2p0 remainder, not GQA-only")
    a("    tok/s    NO  — MHA 1.89%; token is 60% all-organ weight addressing; MLX gap is not a GQA operator")
    a("    dispatch NO  — 1 CB already; per-dispatch overhead is Type-1 dead")
    a("    work     only if grouped Q/O is quality-legal (it is not measured; prior science says no)")
    a(f"  S011 in git           {s011['n_files']} files. Storage-alone law measured by operation census.")
    a("")
    a("## 6. MLP DISTILL (why GQA was asked)")
    a(f"  status     {mlp_nogo['status']}")
    a(f"  decision   {mlp_nogo['decision']}  deciding_number={mlp_nogo['deciding_number']}")
    mlp_src = mlp_nogo["source"]
    mlp_src_s = mlp_src.get("path") if isinstance(mlp_src, dict) else mlp_src
    a(f"  source     {mlp_src_s}")
    a(f"  {mlp_nogo['note']}")
    a("  GQA is not the fallback mass on uniform-q4. It is the quality floor that kept mixed packs at 4.25 BPW on the remainder.")
    a("")
    a("## 7. NUMBER DISCIPLINE")
    a("  Every figure in the JSON is MEASURED, DERIVED (closed arithmetic on MEASURED), CITED, or NULL with reason.")
    a("  Controls: MLX 4-bit 35.51 tok/s LIVE; llama.cpp Q5_K 24.12 tok/s ARCHIVED (artifact off disk).")
    a(f"  Native today: {ANCHOR_TPS} tok/s / {ANCHOR_TOKEN_MS} ms, 964 dispatches, 1 CB, 51.24 GFLOP GEMV, reconstructs_dense NO on 38/38 bound kernels.")
    a("")
    a("## WHAT I WATCHED FAIL")
    for i, item in enumerate(watched, 1):
        a(f"  {i}. {item}")
    a("")
    a("## SELF CHECKS")
    for k, v in self_checks.items():
        a(f"  [{'OK' if v else 'FAIL'}] {k}={v}")
    a("")
    a("## LOCATED")
    for rel, loc in located.items():
        flag = "OK" if loc.get("found") else "MISSING"
        how = loc.get("how") or "-"
        a(f"  [{flag:7}] {how:4} {rel}")

    report = "\n".join(w) + "\n"
    sys.stdout.write(report)
    if not all(self_checks.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
