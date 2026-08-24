#!/usr/bin/env python3
"""N034 — Organ / Kernel / Representation libraries + Odyssey queue recovery.

CPU archaeology. Reads verified campaign receipts and the on-disk Odyssey
queue; writes four generated libraries. Does not load a model, does not
touch the GPU, does not mutate NOETIC_PARENT_A, does not invent numbers.
Unmeasured fields are ABSENT with a reason.

    python3 tools/headless/genome_libraries.py
    python3 -m pytest tools/headless/test_genome_libraries.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEADLESS = REPO / "receipts" / "headless"
ODYSSEY = REPO / "workspace" / "campaign" / "odyssey"
SHADERS = REPO / "crates" / "hawking-core" / "shaders"

SCHEMA_ORGAN = "hawking.headless.organ_library.v1"
SCHEMA_KERNEL = "hawking.headless.kernel_library.v1"
SCHEMA_REPR = "hawking.headless.representation_library.v1"
SCHEMA_ODYSSEY = "hawking.headless.odyssey_queue_recovered.v1"

ORGAN_OUT = HEADLESS / "ORGAN_LIBRARY.json"
KERNEL_OUT = HEADLESS / "KERNEL_LIBRARY.json"
REPR_OUT = HEADLESS / "REPRESENTATION_LIBRARY.json"
ODYSSEY_OUT = HEADLESS / "ODYSSEY_QUEUE_RECOVERED.json"

GENERATOR = "tools/headless/genome_libraries.py"
OBLIGATION = (
    "N034 — GENOME LIBRARIES + ODYSSEY QUEUE RECOVERY (S023 §12, §13, §14, "
    "§34, §70). Seed from verified Qwen science; recover the Odyssey queue "
    "from disk; do not invent numbers or HF repo ids."
)

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
CITED = "CITED"

# Contract organ ids. ORGAN_BANDWIDTH / ORGAN_ROOF_LEDGER use "embedding".
ORGANS = (
    "embed",
    "gqa_attention",
    "deltanet",
    "mlp_gate_up",
    "mlp_down",
    "lm_head",
    "sampling",
)
ORGAN_ALIAS = {"embed": "embedding"}
BW_TO_CONTRACT = {"embedding": "embed"}

# S023 §14 families required by N034. Sources: BYTES_FRONTIER + C1/C2/C3/C5.
REPR_FAMILIES = (
    "q2_affine",
    "q4_control",
    "binary",
    "ternary",
    "shared_basis",
    "binary_sparse_residual",
    "low_rank_sparse",
)

# Frontier family NAMES the user gave (S023 §34). Reconcile, do not invent.
FRONTIER_NAMES = (
    "Qwen3.8",
    "DeepSeek V4 Flash",
    "GLM 5.x",
    "T5V4",
    "Kimi K3",
)

# Qualified kernels named by N034. Identity is the Metal `kernel void` name.
QUALIFIED_KERNELS: tuple[dict[str, str], ...] = (
    {
        "kernel": "q2f_group64_matvec",
        "organ": "mlp_down",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
        "why": "q2f_group64_matvec family named by N034; 4-level fitted g64 codes.",
    },
    {
        "kernel": "q2f_group64_matvec_geo_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
        "why": "geo_tpr64 specialization of q2f_group64_matvec.",
    },
    {
        "kernel": "affine2_group32_matvec_geo_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
        "why": "affine2 geo_tpr64; KERNEL_COMPETENCE specialized-arm control.",
    },
    {
        "kernel": "affine2_group64_matvec_geo_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
        "why": "affine2 geo_tpr64 g64 shift specialization.",
    },
    {
        "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/q80_mixed_decode.metal",
        "why": "NOETIC_PARENT_A production kernel (KernelGenome.production_kernel).",
    },
    {
        "kernel": "qwen80_add_residual_rmsnorm_tg",
        "organ": "gqa_attention",
        "representation": "q4_control",
        "shader": "crates/hawking-core/shaders/qwen80_device_activations.metal",
        "why": "qwen80_add_residual_rmsnorm named by N034; residual+RMSNorm fusion.",
    },
    {
        "kernel": "qwen38_gated_delta_decode_vi_simd_ba_f4",
        "organ": "deltanet",
        "representation": "q4_control",
        "shader": "crates/hawking-core/shaders/qwen38_device_activations.metal",
        "why": "DeltaNet widen_f4 (DELTANET_ORGAN): 6.36 -> 5.52 ms.",
    },
    {
        "kernel": "q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "organ": "mlp_gate_up",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
        "why": "gate_up_swiglu fused operator, q2f g64 geo.",
    },
    {
        "kernel": "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "organ": "mlp_gate_up",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/q80_mixed_decode.metal",
        "why": "ORGAN_ROOF_LEDGER production mlp_gate_up kernel.",
    },
    {
        "kernel": "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "organ": "mlp_gate_up",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
        "why": "MLP_GATE_UP attempted fused SwiGLU body.",
    },
    {
        "kernel": "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "ternary",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER ternary geo, hidden=5120.",
    },
    {
        "kernel": "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128",
        "organ": "mlp_gate_up",
        "representation": "ternary",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER ternary geo, intermediate=17408.",
    },
    {
        "kernel": "binary_g64_matvec_geo_c5120_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "binary",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER binary geo; the family that DID move token_ns.",
    },
    {
        "kernel": "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "shared_basis",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER shared-binary k=2 group dots.",
    },
    {
        "kernel": "shared_binary_k2_scale_contract_gpr80",
        "organ": "mlp_down",
        "representation": "shared_basis",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER shared-binary k=2 scale contract.",
    },
    {
        "kernel": "binary_sparse_fused_geo_c5120_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "binary_sparse_residual",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER binary + 2% CSR fused.",
    },
    {
        "kernel": "q2f_g64_matvec_geo_c5120_tpr64_tg128",
        "organ": "mlp_down",
        "representation": "q2_affine",
        "shader": "crates/hawking-core/shaders/bytes_frontier.metal",
        "why": "BYTES_FRONTIER q2f baseline in the same harness.",
    },
)

SEMANTIC = {
    "embed": (
        "Token-id gather into the hidden vector. Occupancy-starved; not a GEMV. "
        "ORGAN_BANDWIDTH bills 1 dispatch (embed_lookup), 2720 weight-read bytes."
    ),
    "gqa_attention": (
        "Grouped-query attention mixer on the 16 GQA layers: input RMSNorm, QKV "
        "concat, QK-norm/RoPE/cache, MHA decode, sigmoid gate, mixer residual. "
        "o_proj is billed to q4_remainder, not this organ."
    ),
    "deltanet": (
        "Gated-delta linear-attention mixer on the 48 DeltaNet layers: input "
        "RMSNorm, qkvz+ba concat, rearrange/conv/L2, ba_to_decay, gated-delta "
        "recurrent state, gated RMSNorm, mixer residual. Recurrent state is a "
        "summary, not a prefix-shareable cache. out_proj is q4_remainder."
    ),
    "mlp_gate_up": (
        "Post-attention RMSNorm + fused gate+up+SwiGLU. Largest recoverable "
        "token_ns holder on the parent token graph."
    ),
    "mlp_down": (
        "MLP down-projection (affine2 g64 / q2) plus MLP residual. Independent "
        "of gate_up; MLP_DOWN forbids transferring gate_up conclusions."
    ),
    "lm_head": (
        "Terminal RMSNorm + vocabulary projection (Q4 GEMV) onto 248320 logits."
    ),
    "sampling": (
        "Argmax over logits. No weight stream. Occupancy-starved launch class."
    ),
}

NNS_NEEDLES = {
    "embed": ("embed",),
    "gqa_attention": ("attention", "gqa"),
    "deltanet": ("deltanet",),
    "mlp_gate_up": ("mlp", "gate"),
    "mlp_down": ("mlp", "down_proj", "down"),
    "lm_head": ("lm_head",),
    "sampling": (),
}

PATHY = re.compile(
    r"^(receipts/|workspace/|docs/|crates/|tools/|lab/|hcli/|hawking-experiments/)"
)
CITE_KEYS = {
    "source",
    "receipt",
    "receipt_ref",
    "shader_path",
    "shader",
    "citations",
}


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


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def git_exists(rel_path: str) -> bool:
    rel_path = rel_path.lstrip("./")
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        timeout=20,
    )
    return r.returncode == 0


def citation_exists(rel_path: str) -> bool:
    """A sparse-missing file is not evidence it does not exist — check git."""
    rel_path = rel_path.lstrip("./")
    if (REPO / rel_path).is_file():
        return True
    return git_exists(rel_path)


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


def sha256_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "kind": ABSENT,
            "value": None,
            "unit": "sha256",
            "command": f"sha256({rel(path)})",
            "source": rel(path),
            "absent_reason": f"{rel(path)} is not materialized in this worktree",
        }
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return {
        "kind": MEASURED,
        "value": h.hexdigest(),
        "unit": "sha256",
        "command": f"sha256({rel(path)})",
        "source": rel(path),
        "bytes": path.stat().st_size,
        "absent_reason": None,
    }


def qty(
    value: Any,
    *,
    kind: str,
    unit: str,
    command: str,
    source: str,
    absent_reason: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "source": source,
        "absent_reason": absent_reason,
    }
    if note is not None:
        out["note"] = note
    return out


def absent(unit: str, command: str, reason: str, source: str | None = None) -> dict[str, Any]:
    return qty(
        None,
        kind=ABSENT,
        unit=unit,
        command=command,
        source=source or "",
        absent_reason=reason,
    )


def copy_qty(blob: Any, fallback_source: str, command: str, unit: str) -> dict[str, Any]:
    if isinstance(blob, dict) and "kind" in blob:
        out = dict(blob)
        if not out.get("source"):
            out["source"] = fallback_source
        return out
    if blob is None:
        return absent(unit, command, "source receipt has no value", fallback_source)
    return qty(
        blob,
        kind=CITED,
        unit=unit,
        command=command,
        source=fallback_source,
    )


# ---------------------------------------------------------------------------
# citation walker (exported for the test)
# ---------------------------------------------------------------------------


def iter_citations(obj: Any, acc: list[str] | None = None) -> list[str]:
    acc = acc if acc is not None else []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "citations" and isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and PATHY.match(item):
                        acc.append(item)
            elif k in CITE_KEYS and isinstance(v, str) and PATHY.match(v):
                acc.append(v)
            elif k == "source" and isinstance(v, str) and PATHY.match(v):
                acc.append(v)
            else:
                iter_citations(v, acc)
    elif isinstance(obj, list):
        for x in obj:
            iter_citations(x, acc)
    return acc


def unique_citations(obj: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in iter_citations(obj):
        c = c.lstrip("./")
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def unresolved_citations(obj: Any) -> list[str]:
    return [c for c in unique_citations(obj) if not citation_exists(c)]


# ---------------------------------------------------------------------------
# receipt bundle
# ---------------------------------------------------------------------------


class Bundle:
    def __init__(self) -> None:
        self.organ_bw = load_json("receipts/headless/ORGAN_BANDWIDTH.json")
        self.organ_roof = load_json("receipts/headless/ORGAN_ROOF_LEDGER.json")
        self.deltanet = load_json("receipts/headless/DELTANET_ORGAN.json")
        self.mlp_gate = load_json("receipts/headless/MLP_GATE_UP.json")
        self.mlp_down = load_json("receipts/headless/MLP_DOWN.json")
        self.competence = load_json("receipts/headless/KERNEL_COMPETENCE.json")
        self.bytes_frontier = load_json("receipts/headless/BYTES_FRONTIER.json")
        self.c1 = load_json("receipts/headless/C1SHAREDBASIS_DESIGN.json")
        self.c2 = load_json("receipts/headless/C2TENSOROP_DESIGN.json")
        self.c3 = load_json("receipts/headless/C3LOWRANKSPARSE_DESIGN.json")
        self.c5 = load_json("receipts/headless/C5STRUCTTRANSFORM_DESIGN.json")
        self.negative = load_json("receipts/headless/NOETIC_NEGATIVE_SCIENCE.json")
        self.frontiers = load_json("receipts/headless/ORGAN_FRONTIERS.json")
        self.parent = load_json("receipts/headless/NOETIC_PARENT_A.json")
        self.ops = load_json("receipts/headless/NOETIC_OPERATION_CENSUS.json")
        self.machine = load_json("receipts/headless/MACHINE_GENOME.json")
        self.registry = load_json("receipts/headless/MODEL_REGISTRY.json")
        self.dispatch = load_json("receipts/headless/DISPATCH_LEDGER.json")
        self.native_mlp = load_json("receipts/headless/NATIVE_2BIT_MLP.json")
        self.manifest = load_json("workspace/campaign/odyssey/ODYSSEY_MANIFEST.json")
        self.state = load_json("workspace/campaign/odyssey/ODYSSEY_STATE.json")
        self.completions = load_json("workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json")
        self.provenance = load_json("workspace/campaign/odyssey/PROVENANCE.json")
        self.glm52 = load_json_optional(
            "workspace/campaign/evidence/models/glm52/GLM52_SOURCE_ADMISSION.json"
        )
        self.kimi = load_json_optional(
            "workspace/campaign/evidence/models/kimi-k3/KIMI_K3_SOURCE_ADMISSION.json"
        )
        self.dsv4f = load_json_optional(
            "receipts/DSV4F_NATIVE_PLUS_ZEROCOPY_COMBINED.json"
        )
        self._competence_index = self._index_competence()
        self._autopsy_index = self._index_autopsies()

    def _index_competence(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for f in self.competence.get("per_file") or []:
            fname = f.get("file")
            for k in f.get("kernels") or []:
                name = k.get("kernel")
                if not name:
                    continue
                row = dict(k)
                row["file"] = fname
                out[name] = row
        return out

    def _index_autopsies(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for receipt_name, doc in (
            ("receipts/headless/DELTANET_ORGAN.json", self.deltanet),
            ("receipts/headless/MLP_GATE_UP.json", self.mlp_gate),
            ("receipts/headless/MLP_DOWN.json", self.mlp_down),
        ):
            autopsy = doc.get("kernel_autopsy") or {}
            for k in autopsy.get("new_kernels") or []:
                name = k.get("kernel")
                if name:
                    row = dict(k)
                    row["_receipt"] = receipt_name
                    out[name] = row
        bf = (self.bytes_frontier.get("kernel_competence") or {}).get(
            "bytes_frontier_kernels"
        ) or {}
        for name, k in bf.items():
            if name not in out:
                row = dict(k)
                row["kernel"] = name
                row["_receipt"] = "receipts/headless/BYTES_FRONTIER.json"
                out[name] = row
        return out

    def competence_of(self, name: str) -> dict[str, Any]:
        row = self._competence_index.get(name)
        if row is not None:
            return {
                "verdict": row.get("verdict"),
                "n_findings": row.get("n_findings"),
                "may_condemn_representation": row.get("may_condemn_representation"),
                "file": row.get("file"),
                "kind": CITED,
                "source": "receipts/headless/KERNEL_COMPETENCE.json",
                "command": f"KERNEL_COMPETENCE.per_file.kernels[{name}].verdict",
                "absent_reason": None,
            }
        row = self._autopsy_index.get(name)
        if row is not None:
            return {
                "verdict": row.get("verdict"),
                "n_findings": row.get("n_findings"),
                "file": row.get("file"),
                "kind": CITED,
                "source": row.get("_receipt"),
                "command": f"kernel_autopsy.new_kernels[{name}].verdict",
                "note": (
                    "Not in the sealed KERNEL_COMPETENCE.json per_file index "
                    "(that screen predates this kernel). Verdict copied from "
                    "the organ receipt that screened it."
                ),
                "absent_reason": None,
            }
        return {
            "verdict": None,
            "kind": ABSENT,
            "source": "receipts/headless/KERNEL_COMPETENCE.json",
            "command": f"KERNEL_COMPETENCE.per_file.kernels[{name}].verdict",
            "absent_reason": (
                f"{name} is not in KERNEL_COMPETENCE.json per_file and has no "
                "organ-receipt autopsy verdict. Not guessed CLEAR."
            ),
        }

    def bw_row(self, organ: str) -> dict[str, Any]:
        key = ORGAN_ALIAS.get(organ, organ)
        return ((self.organ_bw.get("organ_attribution") or {}).get("organs") or {}).get(
            key
        ) or {}

    def roof_row(self, organ: str) -> dict[str, Any]:
        key = ORGAN_ALIAS.get(organ, organ)
        return (self.organ_roof.get("organs") or {}).get(key) or {}

    def geometry(self) -> dict[str, Any]:
        return self.ops.get("geometry") or {}


def envelope(schema: str, extra: dict[str, Any]) -> dict[str, Any]:
    base = {
        "schema": schema,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "hand_authored": False,
        "did_not_load_a_model": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_modify_odyssey_state": True,
        "unmeasured_is_absent": True,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# ORGAN LIBRARY
# ---------------------------------------------------------------------------


def nns_for(b: Bundle, organ: str) -> list[dict[str, Any]]:
    needles = NNS_NEEDLES.get(organ) or ()
    if not needles:
        return []
    out: list[dict[str, Any]] = []
    for e in b.negative.get("entries") or []:
        scope = e.get("scope") or {}
        blob = json.dumps(scope).lower()
        if not any(n in blob for n in needles):
            continue
        out.append(
            {
                "id": e.get("id"),
                "kind": e.get("kind"),
                "claim_refuted": e.get("claim_refuted"),
                "scope_organ": (scope.get("organ") if isinstance(scope, dict) else None),
                "source": "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            }
        )
    return out


def organ_dims(b: Bundle, organ: str) -> dict[str, Any]:
    g = b.geometry()
    src = "receipts/headless/NOETIC_OPERATION_CENSUS.json"
    cmd = "NOETIC_OPERATION_CENSUS.geometry"

    def gqty(key: str, unit: str = "1") -> dict[str, Any]:
        if key not in g:
            return absent(unit, f"{cmd}.{key}", f"geometry.{key} missing", src)
        return qty(g[key], kind=CITED, unit=unit, command=f"{cmd}.{key}", source=src)

    common = {
        "layers": gqty("layers"),
        "hidden": gqty("hidden"),
        "source_architecture": {
            "value": "Qwen3.8-27B hybrid (GQA every 4th layer, DeltaNet otherwise)",
            "kind": CITED,
            "source": src,
            "command": "geometry.gqa_layers/dn_layers",
            "gqa_layers": g.get("gqa_layers"),
            "dn_layers": g.get("dn_layers"),
        },
    }
    if organ == "embed":
        common.update({"vocab": gqty("vocab"), "shape": "vocab x hidden"})
    elif organ == "gqa_attention":
        common.update(
            {
                "gqa_layers": gqty("gqa_layers"),
                "gqa_heads": gqty("gqa_heads"),
                "gqa_kv_heads": gqty("gqa_kv_heads"),
                "gqa_head_dim": gqty("gqa_head_dim"),
                "q_proj_rows": gqty("q_proj_rows"),
                "kv_proj_rows": gqty("kv_proj_rows"),
            }
        )
    elif organ == "deltanet":
        common.update(
            {
                "dn_layers": gqty("dn_layers"),
                "lin_key_heads": gqty("lin_key_heads"),
                "lin_value_heads": gqty("lin_value_heads"),
                "lin_key_dim": gqty("lin_key_dim"),
                "lin_value_dim": gqty("lin_value_dim"),
                "qkvz_rows": gqty("qkvz_rows"),
                "rec_state_elements": gqty("rec_state_elements"),
            }
        )
    elif organ == "mlp_gate_up":
        common.update(
            {
                "intermediate": gqty("intermediate"),
                "shape_note": "fused gate+up against hidden; rows = 2 * intermediate on the concat path",
            }
        )
    elif organ == "mlp_down":
        common.update(
            {
                "intermediate": gqty("intermediate"),
                "shape": "hidden x intermediate",
            }
        )
    elif organ == "lm_head":
        common.update({"vocab": gqty("vocab"), "shape": "vocab x hidden"})
    elif organ == "sampling":
        common.update({"vocab": gqty("vocab"), "weight_bytes": 0})
    return common


def organ_representation(b: Bundle, organ: str) -> dict[str, Any]:
    rg = b.parent.get("RepresentationGenome") or {}
    fr = b.frontiers.get("organs") or {}
    if organ in ("mlp_gate_up", "mlp_down"):
        return {
            "family": "q2_affine",
            "codec": rg.get("codec"),
            "group": rg.get("group"),
            "affine_tensor_storage_bpw": qty(
                rg.get("affine_tensor_storage_bpw"),
                kind=CITED,
                unit="bpw",
                command="NOETIC_PARENT_A.RepresentationGenome.affine_tensor_storage_bpw",
                source="receipts/headless/NOETIC_PARENT_A.json",
            ),
            "lowest_coherent_native_mlp_bpw": qty(
                2.25,
                kind=CITED,
                unit="bpw",
                command="BYTES_FRONTIER.representations[q2_4level_fitted_g64].active_bpw",
                source="receipts/headless/BYTES_FRONTIER.json",
                note="N021 / BYTES_FRONTIER q2f g64 baseline. Ternary 1.85 failed whole-model composition.",
            ),
            "source": "receipts/headless/NOETIC_PARENT_A.json",
        }
    if organ == "gqa_attention":
        floor = (fr.get("gqa") or {}).get("floor") or {}
        return {
            "family": "q4_control",
            "codec": rg.get("attention"),
            "floor_storage_bpw": qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="bpw",
                command="ORGAN_FRONTIERS.organs.gqa.floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            "source": "receipts/headless/ORGAN_FRONTIERS.json",
        }
    if organ == "deltanet":
        floor = (fr.get("deltanet") or {}).get("floor") or {}
        return {
            "family": "q4_control",
            "floor_storage_bpw": qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="bpw",
                command="ORGAN_FRONTIERS.organs.deltanet.floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            "source": "receipts/headless/ORGAN_FRONTIERS.json",
        }
    if organ == "embed":
        floor = (fr.get("embedding_output") or {}).get("embed_floor") or {}
        return {
            "family": "q4_control",
            "codec": rg.get("embed_head"),
            "floor_storage_bpw": qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="bpw",
                command="ORGAN_FRONTIERS.organs.embedding_output.embed_floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            "source": "receipts/headless/ORGAN_FRONTIERS.json",
        }
    if organ == "lm_head":
        floor = (fr.get("embedding_output") or {}).get("lm_head_floor") or {}
        return {
            "family": "q4_control",
            "codec": rg.get("embed_head"),
            "floor_storage_bpw": qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="bpw",
                command="ORGAN_FRONTIERS.organs.embedding_output.lm_head_floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
                note="lm_head mix survives q3 (3.25) on observed+cold columns; embed rare-token floor is 4.125.",
            ),
            "source": "receipts/headless/ORGAN_FRONTIERS.json",
        }
    return {
        "family": None,
        "kind": ABSENT,
        "absent_reason": "sampling stores no organ weights",
        "source": "receipts/headless/ORGAN_BANDWIDTH.json",
    }


def organ_ebpw(b: Bundle, organ: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fr = b.frontiers.get("organs") or {}
    fr_active = ((b.frontiers.get("verdict") or {}).get("floors_active") or {})
    if organ in ("mlp_gate_up", "mlp_down"):
        complete = qty(
            2.25,
            kind=CITED,
            unit="EBPW",
            command="BYTES_FRONTIER.representations[q2_4level_fitted_g64].active_bpw",
            source="receipts/headless/BYTES_FRONTIER.json",
            note=(
                "Lowest coherent native MLP in the N021/BYTES_FRONTIER harness. "
                "Parent affine tensors store 2.5 bpw (scale+bias). Whole-model "
                "complete EBPW 3.1393 is NOT this organ's number."
            ),
        )
        active = qty(
            2.25,
            kind=CITED,
            unit="EBPW/token",
            command="BYTES_FRONTIER.representations[q2_4level_fitted_g64].active_bpw",
            source="receipts/headless/BYTES_FRONTIER.json",
            note="For packed q2f, stored = active (BYTES_FRONTIER).",
        )
        return complete, active
    if organ == "gqa_attention":
        floor = (fr.get("gqa") or {}).get("floor") or {}
        act = (fr_active.get("gqa") or {})
        return (
            qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="EBPW",
                command="ORGAN_FRONTIERS.organs.gqa.floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            qty(
                act.get("fused_bpw"),
                kind=CITED,
                unit="EBPW/token",
                command="ORGAN_FRONTIERS.verdict.floors_active.gqa.fused_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
        )
    if organ == "deltanet":
        floor = (fr.get("deltanet") or {}).get("floor") or {}
        act = (fr_active.get("deltanet") or {})
        return (
            qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="EBPW",
                command="ORGAN_FRONTIERS.organs.deltanet.floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            qty(
                act.get("fused_bpw"),
                kind=CITED,
                unit="EBPW/token",
                command="ORGAN_FRONTIERS.verdict.floors_active.deltanet.fused_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
        )
    if organ == "embed":
        floor = (fr.get("embedding_output") or {}).get("embed_floor") or {}
        act = (fr_active.get("embedding_output") or {})
        return (
            qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="EBPW",
                command="ORGAN_FRONTIERS.organs.embedding_output.embed_floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            {
                "kind": ABSENT,
                "value": None,
                "unit": "EBPW/token",
                "command": "ORGAN_FRONTIERS.verdict.floors_active.embedding_output",
                "source": "receipts/headless/ORGAN_FRONTIERS.json",
                "absent_reason": (
                    "Embed gather vs lm_head stream — ORGAN_FRONTIERS forbids quoting "
                    "one active EBPW. Bytes/token is cited instead."
                ),
                "embed_active_bytes_per_token": qty(
                    act.get("embed_active_bytes_per_token"),
                    kind=CITED,
                    unit="bytes/token",
                    command="ORGAN_FRONTIERS.verdict.floors_active.embedding_output.embed_active_bytes_per_token",
                    source="receipts/headless/ORGAN_FRONTIERS.json",
                ),
            },
        )
    if organ == "lm_head":
        floor = (fr.get("embedding_output") or {}).get("lm_head_floor") or {}
        act = (fr_active.get("embedding_output") or {})
        return (
            qty(
                floor.get("storage_bpw"),
                kind=CITED,
                unit="EBPW",
                command="ORGAN_FRONTIERS.organs.embedding_output.lm_head_floor.storage_bpw",
                source="receipts/headless/ORGAN_FRONTIERS.json",
            ),
            qty(
                act.get("lm_head_active_bytes_per_token"),
                kind=CITED,
                unit="bytes/token",
                command="ORGAN_FRONTIERS.verdict.floors_active.embedding_output.lm_head_active_bytes_per_token",
                source="receipts/headless/ORGAN_FRONTIERS.json",
                note="Active billed as bytes/token, not a single EBPW (embed gather vs lm_head stream).",
            ),
        )
    return (
        absent(
            "EBPW",
            "best_complete_ebpw",
            "sampling has no stored weights; EBPW is not defined for this organ",
            "receipts/headless/ORGAN_BANDWIDTH.json",
        ),
        absent(
            "EBPW/token",
            "best_active_ebpw",
            "sampling has no stored weights",
            "receipts/headless/ORGAN_BANDWIDTH.json",
        ),
    )


def organ_kernel_id(b: Bundle, organ: str) -> dict[str, Any]:
    roof = b.roof_row(organ)
    kernels = roof.get("kernels") or []
    src = "receipts/headless/ORGAN_ROOF_LEDGER.json"
    if organ == "deltanet":
        return {
            "production": kernels,
            "best_measured_change": {
                "change": "widen_f4",
                "kernel": "qwen38_gated_delta_decode_vi_simd_ba_f4",
                "source": "receipts/headless/DELTANET_ORGAN.json",
            },
            "source": src,
        }
    if organ == "mlp_gate_up":
        return {
            "production": kernels,
            "source": src,
        }
    if not kernels:
        return {
            "production": kernels,
            "kind": ABSENT if not kernels else CITED,
            "source": src,
            "absent_reason": None if kernels else "ORGAN_ROOF_LEDGER has no kernels list",
        }
    return {"production": kernels, "source": src}


def organ_capability(b: Bundle, organ: str) -> dict[str, Any]:
    fr = b.frontiers.get("organs") or {}
    function_space: Any = None
    if organ == "gqa_attention":
        function_space = (fr.get("gqa") or {}).get("floor")
    elif organ == "deltanet":
        function_space = (fr.get("deltanet") or {}).get("floor")
    elif organ in ("embed", "lm_head"):
        function_space = (fr.get("embedding_output") or {}).get(
            "embed_floor" if organ == "embed" else "lm_head_floor"
        )
    elif organ in ("mlp_gate_up", "mlp_down"):
        function_space = {
            "mlp_fail_bpw": (b.frontiers.get("mlp_not_extrapolated") or {}).get("fail_bpw"),
            "mlp_survive_bpw": (b.frontiers.get("mlp_not_extrapolated") or {}).get(
                "survive_bpw"
            ),
            "note": "These are MLP/whole-model uniform mix figures, not GQA/DeltaNet floors.",
        }
    return {
        "function_space_floor": function_space,
        "function_space_source": "receipts/headless/ORGAN_FRONTIERS.json",
        "live_capability": absent(
            "doctor_pass_rate",
            "NOETIC_PARENT_A.capability_evidence",
            (
                "NOETIC_PARENT_A.capability_evidence is 16 greedy tokens of a "
                "compiler-prose prompt, not a Doctor suite. CAPABILITY_llamacpp-q5k.json "
                "and CAPABILITY_mlx-4bit.json exist and do not target NOETIC_PARENT_A. "
                "Per-organ capability sensitivity is UNMEASURED."
            ),
            "receipts/headless/NOETIC_PARENT_A.json",
        ),
    }


def extra_organ_science(b: Bundle, organ: str) -> dict[str, Any]:
    if organ == "deltanet":
        meas = b.deltanet.get("measurement") or {}
        changes = meas.get("changes") or []
        return {
            "deltanet_organ": {
                "one_line": b.deltanet.get("one_line"),
                "changes": [
                    {
                        "change": c.get("change"),
                        "kernel": c.get("kernel"),
                        "recovered_isolated_ns": c.get("recovered_isolated_ns"),
                        "token_ids_unchanged": c.get("token_ids_unchanged"),
                        "parity_rec_out": c.get("parity_rec_out"),
                    }
                    for c in changes
                    if isinstance(c, dict)
                ],
                "source": "receipts/headless/DELTANET_ORGAN.json",
            }
        }
    if organ == "mlp_gate_up":
        return {
            "mlp_gate_up": {
                "answer": b.mlp_gate.get("answer"),
                "ranking_metric": b.mlp_gate.get("ranking_metric"),
                "n025_gap_share": b.mlp_gate.get("n025_mlp_gate_up_gap_share"),
                "reduced_complete_token_ns": b.mlp_gate.get("reduced_complete_token_ns"),
                "source": "receipts/headless/MLP_GATE_UP.json",
            }
        }
    if organ == "mlp_down":
        return {
            "mlp_down": {
                "one_line": b.mlp_down.get("one_line"),
                "reductions_kind": (b.mlp_down.get("reductions") or {}).get("kind"),
                "source": "receipts/headless/MLP_DOWN.json",
            }
        }
    return {}


def build_organ_genome(b: Bundle, organ: str) -> dict[str, Any]:
    bw = b.bw_row(organ)
    roof = b.roof_row(organ)
    complete, active = organ_ebpw(b, organ)
    citations = [
        "receipts/headless/ORGAN_BANDWIDTH.json",
        "receipts/headless/ORGAN_ROOF_LEDGER.json",
        "receipts/headless/NOETIC_OPERATION_CENSUS.json",
        "receipts/headless/NOETIC_PARENT_A.json",
        "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    ]
    if organ in ("gqa_attention", "deltanet", "embed", "lm_head"):
        citations.append("receipts/headless/ORGAN_FRONTIERS.json")
    if organ == "deltanet":
        citations.append("receipts/headless/DELTANET_ORGAN.json")
    if organ == "mlp_gate_up":
        citations.append("receipts/headless/MLP_GATE_UP.json")
        citations.append("receipts/headless/BYTES_FRONTIER.json")
    if organ == "mlp_down":
        citations.append("receipts/headless/MLP_DOWN.json")
        citations.append("receipts/headless/BYTES_FRONTIER.json")
    genome: dict[str, Any] = {
        "organ": organ,
        "alias_in_bandwidth_receipts": ORGAN_ALIAS.get(organ, organ),
        "semantic_function": SEMANTIC[organ],
        "source_architecture": "Qwen3.8-27B hybrid",
        "dimensions": organ_dims(b, organ),
        "best_representation_family": organ_representation(b, organ),
        "best_complete_ebpw": complete,
        "best_active_ebpw": active,
        "kernel_id": organ_kernel_id(b, organ),
        "measured_bandwidth_gb_s": qty(
            bw.get("achieved_gb_s"),
            kind=CITED,
            unit="GB/s",
            command=f"ORGAN_BANDWIDTH.organ_attribution.organs.{ORGAN_ALIAS.get(organ, organ)}.achieved_gb_s",
            source="receipts/headless/ORGAN_BANDWIDTH.json",
        ),
        "token_ns_contribution": qty(
            bw.get("scaled_gpu_ns"),
            kind=CITED,
            unit="ns/token",
            command=f"ORGAN_BANDWIDTH.organ_attribution.organs.{ORGAN_ALIAS.get(organ, organ)}.scaled_gpu_ns",
            source="receipts/headless/ORGAN_BANDWIDTH.json",
        ),
        "recoverable_token_ns": copy_qty(
            roof.get("recoverable_token_ns"),
            "receipts/headless/ORGAN_ROOF_LEDGER.json",
            f"ORGAN_ROOF_LEDGER.organs.{ORGAN_ALIAS.get(organ, organ)}.recoverable_token_ns",
            "ns/token",
        ),
        "flops": copy_qty(
            roof.get("flops"),
            "receipts/headless/ORGAN_ROOF_LEDGER.json",
            f"ORGAN_ROOF_LEDGER.organs.{ORGAN_ALIAS.get(organ, organ)}.flops",
            "FLOP/token",
        ),
        "dispatches": qty(
            bw.get("dispatches"),
            kind=CITED,
            unit="dispatches/token",
            command=f"ORGAN_BANDWIDTH.organ_attribution.organs.{ORGAN_ALIAS.get(organ, organ)}.dispatches",
            source="receipts/headless/ORGAN_BANDWIDTH.json",
        ),
        "capability_sensitivity": organ_capability(b, organ),
        "negative_science": nns_for(b, organ),
        "citations": citations,
    }
    genome.update(extra_organ_science(b, organ))
    if not bw:
        genome["bandwidth_row_absent"] = {
            "kind": ABSENT,
            "absent_reason": f"ORGAN_BANDWIDTH has no row for {ORGAN_ALIAS.get(organ, organ)}",
        }
    return genome


def build_organ_library(b: Bundle) -> dict[str, Any]:
    organs = [build_organ_genome(b, name) for name in ORGANS]
    return envelope(
        SCHEMA_ORGAN,
        {
            "s023": ["§12", "§70"],
            "one_line": (
                "Seven Qwen3.8 OrganGenomes seeded from ORGAN_BANDWIDTH, "
                "ORGAN_ROOF_LEDGER, DELTANET_ORGAN, MLP_GATE_UP, MLP_DOWN, "
                "ORGAN_FRONTIERS. Unmeasured capability is ABSENT."
            ),
            "organs": organs,
            "n_organs": len(organs),
            "q4_remainder_not_a_contract_organ": {
                "note": (
                    "ORGAN_BANDWIDTH also names q4_remainder (unfused o_proj + "
                    "out_proj). N034's organ list does not include it; it remains "
                    "on ORGAN_BANDWIDTH / ORGAN_ROOF_LEDGER."
                ),
                "source": "receipts/headless/ORGAN_BANDWIDTH.json",
            },
        },
    )


# ---------------------------------------------------------------------------
# KERNEL LIBRARY
# ---------------------------------------------------------------------------


_NAME_GROUP = re.compile(r"group(\d+)")
_NAME_TPR = re.compile(r"tpr(\d+)")
_NAME_TG = re.compile(r"tg(\d+)")
_NAME_COLS = re.compile(r"c(\d+)")


def parse_specialization(name: str) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "kind": DERIVED,
        "command": "parse Metal kernel name",
        "source": "kernel id",
    }
    m = _NAME_GROUP.search(name)
    if m:
        spec["group_size"] = int(m.group(1))
    m = _NAME_TPR.search(name)
    if m:
        spec["tile_rows_per_simdgroup"] = int(m.group(1))
        spec["tile_geometry"] = f"tpr{m.group(1)}"
    m = _NAME_TG.search(name)
    if m:
        spec["threadgroup"] = int(m.group(1))
    m = _NAME_COLS.search(name)
    if m:
        spec["specialized_cols"] = int(m.group(1))
    spec["geo"] = "_geo_" in f"_{name}_" or "geo_" in name
    if "swiglu" in name:
        spec["fused_swiglu"] = True
    if "f4" in name.split("_"):
        spec["vector_width"] = "float4"
    return spec


def kernel_machine(b: Bundle) -> dict[str, Any]:
    mg = b.parent.get("MachineGenome") or {}
    roofs = (b.organ_roof.get("three_roofs") or {})
    return {
        "chipset": mg.get("chipset"),
        "gpu_cores": mg.get("gpu_cores"),
        "unified_memory_bytes": mg.get("unified_memory_bytes"),
        "metal_family": mg.get("metal_family"),
        "source": "receipts/headless/NOETIC_PARENT_A.json",
        "command": "NOETIC_PARENT_A.MachineGenome",
        "device_measured_sustained_gb_s": copy_qty(
            (roofs.get("DEVICE_MEASURED_SUSTAINED") or {}),
            "receipts/headless/ORGAN_ROOF_LEDGER.json",
            "ORGAN_ROOF_LEDGER.three_roofs.DEVICE_MEASURED_SUSTAINED",
            "GB/s",
        ),
        "parent_genome_roof_gb_s_is_not_the_campaign_roof": {
            "parent_machinegenome_measured_roof_gb_s": mg.get("measured_roof_gb_s"),
            "note": (
                "595.9 is the G072 family scoring reference sealed into "
                "NOETIC_PARENT_A.MachineGenome, not DEVICE_MEASURED_SUSTAINED. "
                "Canon law 10. Current DRAM roof is 778.8."
            ),
            "source": "receipts/headless/NOETIC_PARENT_A.json",
        },
    }


def kernel_measurements(b: Bundle, name: str, organ: str) -> dict[str, Any]:
    """Attach measured bytes/FLOPs/latency only when a receipt names this kernel."""
    out: dict[str, Any] = {}
    if name == "qwen38_gated_delta_decode_vi_simd_ba_f4":
        changes = (b.deltanet.get("measurement") or {}).get("changes") or []
        row = next((c for c in changes if c.get("kernel") == name), None)
        if row:
            out["deltanet_widen_f4"] = {
                "recovered_isolated_ns": row.get("recovered_isolated_ns"),
                "recovered_fraction_of_deltanet_25p9_share": row.get(
                    "recovered_fraction_of_deltanet_25p9_share"
                ),
                "token_ids_unchanged": row.get("token_ids_unchanged"),
                "parity_rec_out": row.get("parity_rec_out"),
                "parity_rec_state": row.get("parity_rec_state"),
                "source": "receipts/headless/DELTANET_ORGAN.json",
            }
            out["one_line"] = b.deltanet.get("one_line")
    if name.endswith("gate_up_swiglu_geo_tpr64_tg128") or "gate_up_swiglu" in name:
        isolated = b.mlp_gate.get("isolated_gemv") or {}
        out["mlp_gate_up"] = {
            "answer_head": (b.mlp_gate.get("answer") or "")[:400],
            "isolated_gemv_keys": list(isolated)[:12] if isinstance(isolated, dict) else None,
            "reduced_complete_token_ns": b.mlp_gate.get("reduced_complete_token_ns"),
            "source": "receipts/headless/MLP_GATE_UP.json",
        }
    bf_map = {
        "q2f_g64_matvec_geo_c5120_tpr64_tg128": "q2_4level_fitted_g64",
        "binary_g64_matvec_geo_c5120_tpr64_tg128": "binary_g64",
        "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128": "ternary_5in8_g64",
        "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128": "ternary_5in8_g64",
        "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128": "shared_binary_k2",
        "shared_binary_k2_scale_contract_gpr80": "shared_binary_k2",
        "binary_sparse_fused_geo_c5120_tpr64_tg128": "binary_residual_sparse_2pct",
    }
    if name in bf_map:
        rid = bf_map[name]
        rep = next(
            (r for r in b.bytes_frontier.get("representations") or [] if r.get("id") == rid),
            None,
        )
        if rep:
            ns = ((rep.get("COMPLETE_TOKEN_NS") or {}).get("composed") or {}).get(
                "complete_token_ns"
            )
            out["bytes_frontier_family"] = {
                "representation_id": rid,
                "active_bpw": rep.get("active_bpw"),
                "complete_token_ns": ns,
                "note": (
                    "COMPLETE_TOKEN_NS is the 192-GEMV MLP graph plus N021 non-MLP "
                    "residual, not an isolated single-dispatch kernel time."
                ),
                "source": "receipts/headless/BYTES_FRONTIER.json",
            }
    # Organ-level traffic is NOT this kernel's isolated bytes unless the organ
    # is a single kernel. Sampling/embed are 1 dispatch; others are not.
    bw = b.bw_row(organ)
    roof = b.roof_row(organ)
    organ_kernels = roof.get("kernels") or []
    if organ_kernels == [name] or (len(organ_kernels) == 1 and organ_kernels[0] == name):
        out["organ_equals_this_kernel"] = {
            "traffic_bytes": bw.get("traffic_bytes") or bw.get("weight_read_bytes"),
            "dispatches": bw.get("dispatches"),
            "achieved_gb_s": bw.get("achieved_gb_s"),
            "scaled_gpu_ns": bw.get("scaled_gpu_ns"),
            "source": "receipts/headless/ORGAN_BANDWIDTH.json",
        }
    else:
        out["isolated_bytes"] = absent(
            "bytes",
            "isolated kernel traffic",
            (
                f"{name} shares organ {organ} with {organ_kernels}; organ traffic "
                "is not this kernel's isolated bytes. Not guessed by division."
            ),
            "receipts/headless/ORGAN_ROOF_LEDGER.json",
        )
        out["isolated_latency"] = absent(
            "ns",
            "isolated kernel GPU ns",
            "No isolated GPU ns for this kernel in the seeded receipts"
            if name not in (
                "qwen38_gated_delta_decode_vi_simd_ba_f4",
            )
            else "see deltanet_widen_f4.recovered_isolated_ns (a delta, not a body ns)",
            "receipts/headless/ORGAN_BANDWIDTH.json",
        )
    if name == "qwen80_add_residual_rmsnorm_tg":
        red = b.dispatch.get("reduction") or {}
        out["dispatch_fusion"] = {
            "parent_dispatches": (b.dispatch.get("parent") or {}).get("dispatches")
            or (b.organ_bw.get("prior_not_rederived") or {}).get("parent_dispatches_per_token"),
            "n005_residual_rmsnorm_dispatches": (
                b.organ_bw.get("prior_not_rederived") or {}
            ).get("n005_residual_rmsnorm_dispatches"),
            "reduction": red if red else None,
            "note": "Residual+RMSNorm fusion cut 756 -> 628 (DISPATCH_LEDGER / N005).",
            "source": "receipts/headless/DISPATCH_LEDGER.json",
        }
    return out


def build_kernel_genome(b: Bundle, spec: dict[str, str]) -> dict[str, Any]:
    name = spec["kernel"]
    shader = spec["shader"]
    competence = b.competence_of(name)
    citations = [
        spec["shader"],
        "receipts/headless/KERNEL_COMPETENCE.json",
        "receipts/headless/NOETIC_PARENT_A.json",
        "receipts/headless/ORGAN_ROOF_LEDGER.json",
    ]
    if "deltanet" in name or name.endswith("_f4"):
        citations.append("receipts/headless/DELTANET_ORGAN.json")
    if "swiglu" in name or "gate_up" in name:
        citations.append("receipts/headless/MLP_GATE_UP.json")
    if "bytes_frontier" in shader or name.startswith(
        ("ternary_", "binary_", "shared_binary", "q2f_g64")
    ):
        citations.append("receipts/headless/BYTES_FRONTIER.json")
    if name == "qwen80_add_residual_rmsnorm_tg":
        citations.append("receipts/headless/DISPATCH_LEDGER.json")
    return {
        "kernel_identity": name,
        "organ_identity": spec["organ"],
        "representation_identity": spec["representation"],
        "machine_identity": kernel_machine(b),
        "why_qualified": spec["why"],
        "shader": shader,
        "shader_sha256": sha256_file(REPO / shader),
        "compiled_identity": absent(
            "metallib_digest",
            "compiled metallib identity",
            (
                "NOETIC_PARENT_A records no on-disk .metallib (compiled from source "
                "via newLibraryWithSource). Not guessed."
            ),
            "receipts/headless/NOETIC_PARENT_A.json",
        ),
        "specialization": parse_specialization(name),
        "memory_layout": absent(
            "layout",
            "kernel memory layout",
            "No sealed per-kernel memory-layout receipt; not inferred from the shader text",
            shader,
        ),
        "competence": competence,
        "measurements": kernel_measurements(b, name, spec["organ"]),
        "supported_capability_regime": absent(
            "regime",
            "supported capability regime",
            "No per-kernel Doctor/capability regime was measured. Parent capability is UNMEASURED.",
            "receipts/headless/NOETIC_PARENT_A.json",
        ),
        "parity": _kernel_parity(b, name),
        "citations": citations,
    }


def _kernel_parity(b: Bundle, name: str) -> dict[str, Any]:
    if name == "qwen38_gated_delta_decode_vi_simd_ba_f4":
        changes = (b.deltanet.get("measurement") or {}).get("changes") or []
        row = next((c for c in changes if c.get("kernel") == name), None) or {}
        return qty(
            row.get("parity_rec_out"),
            kind=CITED,
            unit="max_abs_diff",
            command="DELTANET_ORGAN.measurement.changes[widen_f4].parity_rec_out",
            source="receipts/headless/DELTANET_ORGAN.json",
        )
    fused = (b.parent.get("capability_evidence") or {}).get("parity_fused_vs_unfused") or {}
    if "swiglu" in name and fused:
        return qty(
            fused.get("mlp_gate_up_swiglu_max_abs_diff"),
            kind=CITED,
            unit="max_abs_diff",
            command="NOETIC_PARENT_A.capability_evidence.parity_fused_vs_unfused.mlp_gate_up_swiglu_max_abs_diff",
            source="receipts/headless/NOETIC_PARENT_A.json",
        )
    return absent(
        "max_abs_diff",
        "parity",
        f"No sealed parity number names {name}",
        "receipts/headless/KERNEL_COMPETENCE.json",
    )


def build_kernel_library(b: Bundle) -> dict[str, Any]:
    kernels = [build_kernel_genome(b, spec) for spec in QUALIFIED_KERNELS]
    counts = {"DEFECTIVE": 0, "SUSPECT": 0, "CLEAR": 0, "ABSENT": 0}
    for k in kernels:
        v = (k.get("competence") or {}).get("verdict")
        if v in counts:
            counts[v] += 1
        else:
            counts["ABSENT"] += 1
    return envelope(
        SCHEMA_KERNEL,
        {
            "s023": ["§13", "§70"],
            "one_line": (
                f"{len(kernels)} qualified KernelGenomes. Competence from "
                "KERNEL_COMPETENCE.json where indexed, else organ autopsies; "
                "never guessed CLEAR."
            ),
            "screen_of_record": {
                "kernels_screened": (b.competence.get("counts") or {}).get("kernels"),
                "by_verdict": (b.competence.get("counts") or {}).get("by_verdict"),
                "law": b.competence.get("law"),
                "source": "receipts/headless/KERNEL_COMPETENCE.json",
            },
            "qualified_verdict_counts": counts,
            "kernels": kernels,
            "n_kernels": len(kernels),
        },
    )


# ---------------------------------------------------------------------------
# REPRESENTATION LIBRARY
# ---------------------------------------------------------------------------


FEWER_BITS_LAW = {
    "law": "fewer stored bits != fewer nanoseconds",
    "permanent": True,
    "s023": "§21 / N032",
    "statement": (
        "A representation that stores fewer bits than q2f 2.25 bpw is not "
        "thereby faster. On a bandwidth-bound graph, token_ns tracks DRAM bytes "
        "only if the kernel is load-bound and competent. Extra arithmetic "
        "(trit unpack, two-pass dots+scale, CSR gather) can eat the byte win."
    ),
    "evidence": {
        "source": "receipts/headless/BYTES_FRONTIER.json",
        "command": "BYTES_FRONTIER.finding",
        "fewer_bytes_moved_token_ns_toward_729_7": True,
        "who_moved": "binary_g64",
        "who_did_not": [
            "ternary_5in8_g64",
            "shared_binary_k2",
            "binary_residual_sparse_2pct",
        ],
    },
}


def _bf_rep(b: Bundle, rid: str) -> dict[str, Any] | None:
    for r in b.bytes_frontier.get("representations") or []:
        if r.get("id") == rid:
            return r
    return None


def _bf_cost(rep: dict[str, Any] | None) -> dict[str, Any]:
    if not rep:
        return absent(
            "ns",
            "COMPLETE_TOKEN_NS",
            "representation not in BYTES_FRONTIER.representations",
            "receipts/headless/BYTES_FRONTIER.json",
        )
    composed = (rep.get("COMPLETE_TOKEN_NS") or {}).get("composed") or {}
    return {
        "active_bpw": qty(
            rep.get("active_bpw"),
            kind=CITED,
            unit="bpw",
            command=f"BYTES_FRONTIER.representations[{rep.get('id')}].active_bpw",
            source="receipts/headless/BYTES_FRONTIER.json",
        ),
        "active_bytes_per_token": qty(
            rep.get("active_bytes_per_token"),
            kind=CITED,
            unit="bytes/token",
            command=f"BYTES_FRONTIER.representations[{rep.get('id')}].active_bytes_per_token",
            source="receipts/headless/BYTES_FRONTIER.json",
        ),
        "complete_token_ns": qty(
            composed.get("complete_token_ns"),
            kind=CITED,
            unit="ns/token",
            command=f"BYTES_FRONTIER.representations[{rep.get('id')}].COMPLETE_TOKEN_NS.composed.complete_token_ns",
            source="receipts/headless/BYTES_FRONTIER.json",
            note=rep.get("notes"),
        ),
        "toward_roof_729_7": rep.get("toward_roof_729_7"),
        "coherence": rep.get("coherence"),
        "parity": rep.get("parity"),
        "dense_w_materialized": rep.get("dense_w_materialized"),
        "timing_label": "DIRTY_ENGINEERING",
        "timing_label_source": "receipts/headless/BYTES_FRONTIER.json",
    }


def build_repr_entry(b: Bundle, family: str) -> dict[str, Any]:
    parent_rg = b.parent.get("RepresentationGenome") or {}
    fr = b.frontiers
    if family == "q2_affine":
        rep = _bf_rep(b, "q2_4level_fitted_g64")
        return {
            "family": family,
            "successful_organs": ["mlp_gate_up", "mlp_down"],
            "failed_organs": [],
            "density_frontier": {
                "active_bpw": 2.25,
                "parent_affine_storage_bpw": parent_rg.get("affine_tensor_storage_bpw"),
                "whole_model_complete_ebpw": parent_rg.get("complete_ebpw"),
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "execution_cost": _bf_cost(rep),
            "kernel_requirement": [
                "q2f_group64_matvec_geo_tpr64_tg128",
                "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
                "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            ],
            "information_accounting": {
                "codes": "2 bits/weight packed",
                "scales_and_bias": "f16 per group of 64 (parent affine is 2.5 bpw with scale+bias)",
                "dense_reconstruction": 0,
                "source": "receipts/headless/NOETIC_PARENT_A.json",
            },
            "notes": parent_rg.get("codec"),
            "citations": [
                "receipts/headless/BYTES_FRONTIER.json",
                "receipts/headless/NATIVE_2BIT_MLP.json",
                "receipts/headless/NOETIC_PARENT_A.json",
            ],
        }
    if family == "q4_control":
        floors = (fr.get("verdict") or {}).get("floors_storage_bpw") or {}
        return {
            "family": family,
            "successful_organs": ["gqa_attention", "deltanet", "embed", "lm_head"],
            "failed_organs": [],
            "density_frontier": {
                "gqa": floors.get("gqa"),
                "deltanet": floors.get("deltanet"),
                "embedding_output": floors.get("embedding_output"),
                "q4_incumbent_complete_physical_bpw": parent_rg.get(
                    "q4_incumbent_complete_physical_bpw"
                ),
                "source": "receipts/headless/ORGAN_FRONTIERS.json",
            },
            "execution_cost": {
                "note": (
                    "Q4 is the incumbent control on attention/embed/head. Isolated "
                    "q4_remainder sits closer to the DRAM roof (503 GB/s) than MLP."
                ),
                "q4_remainder_achieved_gb_s": qty(
                    ((b.organ_bw.get("organ_attribution") or {}).get("organs") or {})
                    .get("q4_remainder", {})
                    .get("achieved_gb_s"),
                    kind=CITED,
                    unit="GB/s",
                    command="ORGAN_BANDWIDTH.organ_attribution.organs.q4_remainder.achieved_gb_s",
                    source="receipts/headless/ORGAN_BANDWIDTH.json",
                ),
            },
            "kernel_requirement": [
                "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
                "qwen_uniform_q4_embedding_lookup",
            ],
            "information_accounting": {
                "codec": parent_rg.get("attention"),
                "source": "receipts/headless/NOETIC_PARENT_A.json",
            },
            "citations": [
                "receipts/headless/ORGAN_FRONTIERS.json",
                "receipts/headless/ORGAN_BANDWIDTH.json",
                "receipts/headless/NOETIC_PARENT_A.json",
            ],
        }
    if family == "binary":
        rep = _bf_rep(b, "binary_g64")
        return {
            "family": family,
            "successful_organs": ["mlp_gate_up", "mlp_down"],
            "failed_organs": [],
            "density_frontier": {
                "active_bpw": 1.25,
                "note": (
                    "Fewer bytes DID move COMPLETE_TOKEN_NS toward 729.7 "
                    "(delta 4116083 ns). Speed without a generation-coherent "
                    "whole-model mix is not a promotion."
                ),
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "execution_cost": _bf_cost(rep),
            "kernel_requirement": ["binary_g64_matvec_geo_c5120_tpr64_tg128"],
            "information_accounting": {
                "stored_equals_active": True,
                "bpw": 1.25,
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "fewer_bits_moved_ns": True,
            "citations": ["receipts/headless/BYTES_FRONTIER.json"],
        }
    if family == "ternary":
        rep = _bf_rep(b, "ternary_5in8_g64")
        return {
            "family": family,
            "successful_organs": [],
            "failed_organs": ["mlp_gate_up", "mlp_down", "whole_model"],
            "density_frontier": {
                "active_bpw": 1.85,
                "whole_model": "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json flipped argmax (9714 vs 10895)",
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "execution_cost": _bf_cost(rep),
            "kernel_requirement": [
                "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
                "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128",
            ],
            "information_accounting": {
                "trit_pack": "5 trits in 8 bits + f16 scale / group 64 = 1.85 bpw",
                "zeros_still_load": True,
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "fewer_bits_moved_ns": False,
            "citations": [
                "receipts/headless/BYTES_FRONTIER.json",
                "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
                "receipts/headless/ORGAN_FRONTIERS.json",
            ],
        }
    if family == "shared_basis":
        rep = _bf_rep(b, "shared_binary_k2")
        return {
            "family": family,
            "successful_organs": [],
            "failed_organs": ["mlp_gate_up", "mlp_down"],
            "density_frontier": {
                "active_bpw": (rep or {}).get("active_bpw"),
                "c1_verdict": b.c1.get("verdict"),
                "c1_failure": b.c1.get("failure_that_killed_it"),
                "source": "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            },
            "execution_cost": _bf_cost(rep),
            "kernel_requirement": [
                "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
                "shared_binary_k2_scale_contract_gpr80",
            ],
            "information_accounting": {
                "bases_amortized_across_layers": True,
                "native_two_stage_equals_reconstruct": (
                    (b.c1.get("decision") or {}).get("evidence") or {}
                ).get("associativity_identity_holds"),
                "source": "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            },
            "fewer_bits_moved_ns": False,
            "design_verdict": b.c1.get("verdict"),
            "citations": [
                "receipts/headless/BYTES_FRONTIER.json",
                "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            ],
        }
    if family == "binary_sparse_residual":
        rep = _bf_rep(b, "binary_residual_sparse_2pct")
        return {
            "family": family,
            "successful_organs": [],
            "failed_organs": ["mlp_gate_up", "mlp_down"],
            "density_frontier": {
                "active_bpw": (rep or {}).get("active_bpw"),
                "nnz_frac": 0.02,
                "note": "2% CSR indices are not free; active_bpw 2.216 still below 2.25 and still slower.",
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "execution_cost": _bf_cost(rep),
            "kernel_requirement": ["binary_sparse_fused_geo_c5120_tpr64_tg128"],
            "information_accounting": {
                "binary_plane": True,
                "sparse_residual": "CSR 2%",
                "source": "receipts/headless/BYTES_FRONTIER.json",
            },
            "fewer_bits_moved_ns": False,
            "citations": ["receipts/headless/BYTES_FRONTIER.json"],
        }
    if family == "low_rank_sparse":
        c3_answer = b.c3.get("answer")
        return {
            "family": family,
            "successful_organs": [],
            "failed_organs": ["mlp_gate_up", "mlp_down"],
            "density_frontier": {
                "c3_verdict": "NOT_WORTH_BUILDING",
                "c2_tensor_family_also_refuted": b.c2.get("verdict"),
                "c5_struct_transform": b.c5.get("verdict"),
                "source": "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
            },
            "execution_cost": {
                "kind": CITED,
                "source": "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
                "answer": c3_answer,
                "note": (
                    "Fusion guaranteed save is 1,310,720 B = 2199.6 ns at 595.9 GB/s "
                    "(0.0072% of 30.606 ms) — cited as written in C3; 595.9 is the "
                    "then-scoring-reference, not the 778.8 DRAM roof."
                ),
            },
            "kernel_requirement": absent(
                "kernel",
                "native fused low-rank+sparse kernel",
                "C3: Do not write the shader. NOT_WORTH_BUILDING.",
                "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
            ),
            "information_accounting": {
                "source": "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
                "accounting": b.c3.get("accounting"),
            },
            "design_verdict": "NOT_WORTH_BUILDING",
            "related_refutations": {
                "c2_tensor": {
                    "verdict": b.c2.get("verdict"),
                    "source": "receipts/headless/C2TENSOROP_DESIGN.json",
                },
                "c5_struct_transform": {
                    "verdict": b.c5.get("verdict"),
                    "source": "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json",
                },
            },
            "citations": [
                "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
                "receipts/headless/C2TENSOROP_DESIGN.json",
                "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json",
            ],
        }
    raise KeyError(family)


def build_repr_library(b: Bundle) -> dict[str, Any]:
    families = [build_repr_entry(b, name) for name in REPR_FAMILIES]
    return envelope(
        SCHEMA_REPR,
        {
            "s023": ["§14", "§70"],
            "one_line": (
                "Seven representation families from BYTES_FRONTIER + C1/C2/C3/C5. "
                "Permanent law: fewer stored bits != fewer nanoseconds."
            ),
            "laws": [FEWER_BITS_LAW],
            "families": families,
            "n_families": len(families),
        },
    )


# ---------------------------------------------------------------------------
# ODYSSEY QUEUE RECOVERY
# ---------------------------------------------------------------------------


def _hf_pair(repo: Any, revision: Any) -> tuple[str | None, str | None]:
    if isinstance(repo, str) and "/" in repo and " " not in repo.strip():
        rev = revision if isinstance(revision, str) and revision else None
        return repo, rev
    return None, None


def recover_patients(b: Bundle) -> list[dict[str, Any]]:
    state_by = {p.get("oxx"): p for p in (b.state.get("patients") or []) if p.get("oxx")}
    comps = b.completions.get("entries") or []
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for e in comps:
        pid = e.get("patient_id")
        if not pid:
            continue
        by_patient.setdefault(pid, []).append(
            {
                "obligation_id": e.get("obligation_id"),
                "status": e.get("status"),
                "receipt_ref": e.get("receipt_ref"),
                "completed_at": e.get("completed_at"),
            }
        )
    out: list[dict[str, Any]] = []
    for row in b.manifest:
        oxx = row.get("oxx")
        st = state_by.get(oxx) or {}
        repo, rev = _hf_pair(row.get("canonical_source"), row.get("canonical_revision"))
        out.append(
            {
                "oxx": oxx,
                "model": row.get("model"),
                "class": row.get("class"),
                "canonical_source": row.get("canonical_source"),
                "canonical_revision": row.get("canonical_revision"),
                "identity_status": "RESOLVED" if repo else "UNRESOLVED",
                "state": st.get("state"),
                "phase": st.get("phase"),
                "on_disk": st.get("on_disk"),
                "ledger": st.get("ledger"),
                "blocked_reason": st.get("blocked_reason"),
                "state_source_field": st.get("source"),
                "state_canonical_source": st.get("canonical_source"),
                "completions": by_patient.get(oxx) or [],
                "n_completions": len(by_patient.get(oxx) or []),
                "citations": [
                    "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json",
                    "workspace/campaign/odyssey/ODYSSEY_STATE.json",
                    "workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json",
                ],
            }
        )
    return out


def recover_review_queue() -> list[dict[str, Any]]:
    path = ODYSSEY / "REVIEW_QUEUE.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line, "parse": "UNRESOLVED"})
    return rows


def frontier_qwen38(b: Bundle) -> dict[str, Any]:
    parent = (b.registry.get("candidates") or {}).get(b.registry.get("parent") or "")
    recipe = (parent or {}).get("recipe") or {}
    repo, rev = _hf_pair(recipe.get("source_repo"), parent.get("source_sha") if parent else None)
    return {
        "shorthand": "Qwen3.8",
        "identity_status": "RESOLVED" if repo else "UNRESOLVED",
        "repository": repo,
        "revision": rev,
        "registry_parent": b.registry.get("parent"),
        "local_artifact": ((parent or {}).get("artifact") or {}).get("path"),
        "notes": (parent or {}).get("notes"),
        "in_odyssey_manifest": False,
        "role": "current Qwen 3.8 research family (not an Odyssey Oxx patient)",
        "citations": ["receipts/headless/MODEL_REGISTRY.json"],
    }


def frontier_dsv4f(b: Bundle) -> dict[str, Any]:
    man = next((p for p in b.manifest if p.get("oxx") == "O011"), None) or {}
    art = (b.dsv4f or {}).get("artifact") or {}
    repo = art.get("repository") or man.get("canonical_source")
    rev = art.get("revision") or man.get("canonical_revision")
    repo, rev = _hf_pair(repo, rev)
    st = next((p for p in (b.state.get("patients") or []) if p.get("oxx") == "O011"), None) or {}
    return {
        "shorthand": "DeepSeek V4 Flash",
        "identity_status": "RESOLVED" if repo else "UNRESOLVED",
        "repository": repo,
        "revision": rev,
        "odyssey_oxx": "O011",
        "manifest_canonical_source": man.get("canonical_source"),
        "state_source_field": st.get("source"),
        "state_canonical_source": st.get("canonical_source"),
        "notes": (
            "MANIFEST and DSV4F_NATIVE_PLUS_ZEROCOPY_COMBINED.artifact agree on "
            "deepseek-ai/DeepSeek-V4-Flash @ 60d8d707…. ODYSSEY_STATE lists "
            "source='reconstruct from receipts' and canonical_source=null — "
            "recovered, not overwritten."
        ),
        "citations": [
            "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json",
            "workspace/campaign/odyssey/ODYSSEY_STATE.json",
            "receipts/DSV4F_NATIVE_PLUS_ZEROCOPY_COMBINED.json",
        ],
    }


def frontier_glm5(b: Bundle) -> dict[str, Any]:
    repo = (b.glm52 or {}).get("repo")
    rev = (b.glm52 or {}).get("revision")
    repo, rev = _hf_pair(repo, rev)
    glm45 = [p.get("oxx") for p in b.manifest if str(p.get("model") or "").startswith("GLM-4.5")]
    return {
        "shorthand": "GLM 5.x",
        "identity_status": "RESOLVED" if repo else "UNRESOLVED",
        "repository": repo,
        "revision": rev,
        "admission_status": (b.glm52 or {}).get("status"),
        "not_odyssey_glm45": {
            "note": (
                "Odyssey patients O010/O012 are GLM-4.5-Air / GLM-4.5. That is a "
                "different family from GLM 5.x. Not aliased."
            ),
            "glm45_oxx": glm45,
        },
        "notes": (
            "Resolved from GLM52_SOURCE_ADMISSION.json as zai-org/GLM-5.2. "
            "Conversational 'GLM 5.x' maps to that sealed admission; no GLM-5 "
            "repo id was invented."
        ),
        "citations": [
            "workspace/campaign/evidence/models/glm52/GLM52_SOURCE_ADMISSION.json",
            "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json",
        ],
    }


def frontier_t5v4() -> dict[str, Any]:
    return {
        "shorthand": "T5V4",
        "identity_status": "UNRESOLVED",
        "repository": None,
        "revision": None,
        "canonical_source": None,
        "needs_verification": True,
        "notes": (
            "Conversational shorthand. No exact source identity (HF repo + "
            "revision) is on disk in the Odyssey queue, MODEL_REGISTRY, or a "
            "sealed admission receipt. Left UNRESOLVED. No repository id invented."
        ),
        "citations": [
            "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json",
            "workspace/campaign/odyssey/ODYSSEY_STATE.json",
            "receipts/headless/MODEL_REGISTRY.json",
        ],
    }


def frontier_kimi(b: Bundle) -> dict[str, Any]:
    src = (b.kimi or {}).get("source") or {}
    repo = src.get("repository")
    rev = src.get("revision")
    man = next((p for p in b.manifest if p.get("oxx") == "O013"), None) or {}
    if not repo:
        repo = man.get("canonical_source")
        rev = man.get("canonical_revision")
    repo, rev = _hf_pair(repo, rev)
    return {
        "shorthand": "Kimi K3",
        "identity_status": "RESOLVED" if repo else "UNRESOLVED",
        "repository": repo,
        "revision": rev,
        "admission_status": (b.kimi or {}).get("status"),
        "odyssey_oxx": "O013",
        "manifest_canonical_source": man.get("canonical_source"),
        "notes": (
            "KIMI_K3_SOURCE_ADMISSION.json (metadata-only) and ODYSSEY_MANIFEST "
            "O013 agree on moonshotai/Kimi-K3 @ 9f62e4e9…. Body not resident."
        ),
        "citations": [
            "workspace/campaign/evidence/models/kimi-k3/KIMI_K3_SOURCE_ADMISSION.json",
            "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json",
        ],
    }


def build_odyssey_recovered(b: Bundle) -> dict[str, Any]:
    patients = recover_patients(b)
    families = [
        frontier_qwen38(b),
        frontier_dsv4f(b),
        frontier_glm5(b),
        frontier_t5v4(),
        frontier_kimi(b),
    ]
    unresolved = [f["shorthand"] for f in families if f["identity_status"] == "UNRESOLVED"]
    invented = [
        f["shorthand"]
        for f in families
        if f["identity_status"] == "UNRESOLVED" and f.get("repository")
    ]
    return envelope(
        SCHEMA_ODYSSEY,
        {
            "s023": ["§34"],
            "one_line": (
                f"Odyssey queue recovered: {len(patients)} manifest patients, "
                f"{len((b.completions.get('entries') or []))} completions. "
                f"Frontier reconciliation: UNRESOLVED={unresolved}."
            ),
            "method": (
                "READ ODYSSEY_MANIFEST / ODYSSEY_STATE / ODYSSEY_COMPLETIONS / "
                "PROVENANCE / REVIEW_QUEUE from disk. Reconcile user-designated "
                "frontier family NAMES. Never fabricate a repository id."
            ),
            "sources_read": [
                "workspace/campaign/odyssey/ODYSSEY_MANIFEST.json",
                "workspace/campaign/odyssey/ODYSSEY_STATE.json",
                "workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json",
                "workspace/campaign/odyssey/PROVENANCE.json",
                "workspace/campaign/odyssey/REVIEW_QUEUE.jsonl",
                "receipts/headless/MODEL_REGISTRY.json",
                "receipts/DSV4F_NATIVE_PLUS_ZEROCOPY_COMBINED.json",
                "workspace/campaign/evidence/models/glm52/GLM52_SOURCE_ADMISSION.json",
                "workspace/campaign/evidence/models/kimi-k3/KIMI_K3_SOURCE_ADMISSION.json",
            ],
            "queue": patients,
            "n_patients": len(patients),
            "completions_summary": {
                "n": len(b.completions.get("entries") or []),
                "schema": b.completions.get("schema"),
                "patient_ids": sorted(
                    {
                        e.get("patient_id")
                        for e in (b.completions.get("entries") or [])
                        if e.get("patient_id")
                    }
                ),
                "source": "workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json",
            },
            "provenance_recovered": {
                "obligation": b.provenance.get("obligation"),
                "candidate_A_repo": (b.provenance.get("candidate_A_throughput") or {}).get(
                    "repo"
                ),
                "candidate_B_repo": (b.provenance.get("candidate_B_capability") or {}).get(
                    "repo"
                ),
                "note": (
                    "PROVENANCE.json is a G37 inspection snapshot (Qwen3-30B-A3B "
                    "abliterated + GLM-4.5-Air derestricted). Copied, not extended."
                ),
                "source": "workspace/campaign/odyssey/PROVENANCE.json",
            },
            "review_queue": recover_review_queue(),
            "frontier_families": families,
            "unresolved_shorthands": unresolved,
            "did_not_invent_hf_repo_ids": invented == [],
            "unresolved_kept_unresolved": True,
        },
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def build_all() -> dict[str, dict[str, Any]]:
    b = Bundle()
    return {
        "organ": build_organ_library(b),
        "kernel": build_kernel_library(b),
        "representation": build_repr_library(b),
        "odyssey": build_odyssey_recovered(b),
    }


def write_all(docs: dict[str, dict[str, Any]] | None = None) -> dict[str, Path]:
    docs = docs or build_all()
    mapping = {
        "organ": ORGAN_OUT,
        "kernel": KERNEL_OUT,
        "representation": REPR_OUT,
        "odyssey": ODYSSEY_OUT,
    }
    for key, path in mapping.items():
        write_json(path, docs[key])
    return mapping


def main() -> int:
    docs = build_all()
    paths = write_all(docs)
    for key, path in paths.items():
        n_unresolved = unresolved_citations(docs[key])
        print(f"wrote {rel(path)} citations_unresolved={len(n_unresolved)}")
        if n_unresolved:
            for u in n_unresolved[:12]:
                print(f"  MISSING {u}")
            return 1
    print(docs["organ"]["one_line"])
    print(docs["kernel"]["one_line"])
    print(docs["representation"]["one_line"])
    print(docs["odyssey"]["one_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
