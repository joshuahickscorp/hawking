#!/usr/bin/env python3.12
"""Wire trained B_LINEAR_SUBSPACE_INITIALIZATION weights into MultiAdapterHub.

Honest scope
------------
Arm B is the framework baseline: a single ``LinearSubspaceInit`` residual
(low-rank A + bias), no teacher projector, no student observer, never pays
``L_latent``.  Its checkpoint loss is reconstruction / self-consistency, not
held-out math quality.

This module:
  1. Loads ``BEST_BALANCED.pt`` from the real ep8 retrain.
  2. Builds per-site MultiAdapterHub instances at the 8 named V0 sites, each
     carrying an independent weight clone of the trained residual (hub catalog
     attachment points).  Training semantics apply the residual **once**; the
     smoke path that matches training uses the master single-bridge hub.
  3. Activates the hub via env/config override only — never flips module-level
     ``HUB_ACTIVE`` default.
  4. Runs offline smoke against real DSV4F host-export activations from
     ``gravity_deepseek_v4_fullseq_capture`` (not production Rust/Metal serve).

Capability claim: always false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from lab.layout import EVIDENCE_ROOT, REPO_ROOT
from lab.operators import frankenstein_adapter_hub as hub
from lab.operators import frankenstein_latent_v0 as lv
from lab.operators.frankenstein_bridges import V0_BRIDGE_SITES
from lab.receipts import seal

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

ARM_B = lv.ARM_B  # B_LINEAR_SUBSPACE_INITIALIZATION
BRIDGE_NAME = "LINEAR_SUBSPACE_INITIALIZATION"

DEFAULT_CKPT = (
    EVIDENCE_ROOT
    / "models"
    / "frankenstein"
    / "bridge_train_real_fix_ep8"
    / "checkpoints"
    / "real_B_LINEAR_SUBSPACE_INITIALIZATION"
    / "BEST_BALANCED.pt"
)
DEFAULT_DSV4F_EXPORT = (
    REPO_ROOT / "receipts" / "dsv4f_fullseq_capture_L0_frozen_export"
)
DEFAULT_FROZEN_CORPUS = (
    EVIDENCE_ROOT
    / "models"
    / "frankenstein"
    / "teacher_forced"
    / "official_L0_stream_full_20260805T200728Z"
    / "FROZEN_CORPUS_L0.json"
)
DEFAULT_SITE_MAP_SOURCE = (
    EVIDENCE_ROOT
    / "models"
    / "frankenstein"
    / "bridge_train_real_fix_ep8"
    / "BRIDGE_TRAIN_REAL_FIRST_RUN.json"
)
DEFAULT_BODY = (
    REPO_ROOT
    / "workspace"
    / "campaign"
    / "records"
    / "runs"
    / "deepseek-v4"
    / "full-43-layer-stream.gravity"
)
DEFAULT_EVIDENCE_OUT = (
    EVIDENCE_ROOT / "models" / "frankenstein" / "proto-v0-desktop-artifact"
)
DEFAULT_DESKTOP = Path.home() / "Desktop" / "hawking-frankenstein" / "proto-frankenstein"

# Site → DSV4F layer from the real ep8 retrain site_map (not re-derived here).
DEFAULT_SITE_LAYERS: dict[str, int] = {
    "GLM_EARLY_CONTEXT_BRIDGE": 7,
    "GLM_METHOD_BRIDGE": 7,
    "GLM_DECOMPOSITION_BRIDGE": 15,
    "GLM_PRE_ROUTER_BRIDGE": 16,
    "GLM_POST_MOE_BRIDGE": 16,
    "GLM_FORMALIZATION_BRIDGE": 25,
    "GLM_REPAIR_BRIDGE": 34,
    "GLM_LATE_CONSOLIDATION_BRIDGE": 37,
}

SCHEMA_RECEIPT = "hawking.frankenstein.b_linear_hub_desktop_artifact.v1"
SCHEMA_SMOKE = "hawking.frankenstein.b_linear_hub_smoke.v1"
SCHEMA_WIRE = "hawking.frankenstein.b_linear_hub_wire.v1"

# Norm blow-up threshold relative to input (smoke, not a quality gate).
MAX_NORM_RATIO = 50.0


class BLinearHubError(RuntimeError):
    """Fail-closed error for B-linear hub wiring."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_stats(t: torch.Tensor) -> dict[str, Any]:
    arr = t.detach().float()
    finite = bool(torch.isfinite(arr).all().item())
    flat = arr.reshape(-1)
    return {
        "shape": list(arr.shape),
        "finite": finite,
        "has_nan": bool(torch.isnan(arr).any().item()),
        "has_inf": bool(torch.isinf(arr).any().item()),
        "norm_l2": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
        "norm_rms": float(torch.sqrt(torch.mean(flat * flat)).item()) if flat.numel() else 0.0,
        "abs_max": float(torch.max(torch.abs(flat)).item()) if flat.numel() else 0.0,
        "mean": float(torch.mean(flat).item()) if flat.numel() else 0.0,
    }


# ---------------------------------------------------------------------------
# Checkpoint load
# ---------------------------------------------------------------------------


@dataclass
class LoadedBLinear:
    """Loaded arm-B stack + hub wiring metadata."""

    checkpoint_path: Path
    checkpoint_sha256: str
    stack: lv.LatentV0Stack
    blob_metrics: dict[str, Any]
    blob_spec: dict[str, Any]
    load_info: dict[str, Any]


def load_b_linear_checkpoint(
    path: str | Path | None = None,
    *,
    slot_key: str = "runtime_state_dict",
) -> LoadedBLinear:
    """Load BEST_BALANCED (or other slot file) into an arm-B LatentV0Stack."""

    ckpt = Path(path) if path is not None else DEFAULT_CKPT
    if not ckpt.is_file():
        raise BLinearHubError(f"checkpoint missing: {ckpt}")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise BLinearHubError(f"checkpoint is not a dict: {type(blob)}")
    if blob.get("capability_claim") is True:
        raise BLinearHubError("refusing checkpoint with capability_claim=true")
    if slot_key not in blob:
        raise BLinearHubError(f"checkpoint missing {slot_key!r}")

    spec = dict(blob.get("spec") or {})
    d_student = int(spec.get("d_student") or lv.DSV4F_HIDDEN)
    d_teacher = int(spec.get("d_teacher") or lv.GLM_HIDDEN)
    d_latent = int(spec.get("d_latent") or 128)
    rank = int(spec.get("rank") or 16)
    d_hidden = int(spec.get("d_hidden") or 128)

    stack = lv.build_stack_for_arm(
        ARM_B,
        d_teacher=d_teacher,
        d_student=d_student,
        d_latent=d_latent,
        rank=rank,
        d_hidden=d_hidden,
        n_experts=16,
        n_methods=len(lv.METHOD_CLASSES),
        n_actions=12,
    )
    state = blob[slot_key]
    missing, unexpected = stack.load_state_dict(state, strict=False)
    if stack.linear_init is None:
        raise BLinearHubError("arm-B stack has no linear_init after load")
    stack.mark_trained()
    return LoadedBLinear(
        checkpoint_path=ckpt.resolve(),
        checkpoint_sha256=_sha256_file(ckpt),
        stack=stack,
        blob_metrics=dict(blob.get("metrics") or {}),
        blob_spec=spec,
        load_info={
            "slot": blob.get("slot"),
            "phase": blob.get("phase"),
            "schema": blob.get("schema"),
            "capability_claim": bool(blob.get("capability_claim", False)),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "state_key": slot_key,
            "n_state_tensors": len(state),
            "linear_init_param_count": sum(
                int(p.numel()) for p in stack.linear_init.parameters()
            ),
        },
    )


def clone_linear_init(src: lv.LinearSubspaceInit) -> lv.LinearSubspaceInit:
    """Independent weight clone (site hubs cannot share one nn.Module parent)."""

    dst = lv.LinearSubspaceInit(
        d_model=src.d_model,
        rank=src.rank,
        scale=src.scale,
        init_std=0.0,
    )
    dst.load_state_dict(src.state_dict())
    dst.meta = dict(src.meta)
    return dst


# ---------------------------------------------------------------------------
# Hub construction (env/config activation — never module default flip)
# ---------------------------------------------------------------------------


def build_master_hub(
    linear: lv.LinearSubspaceInit,
    *,
    hub_active: bool,
    selection_mode: str = "top1",
    d_model: int | None = None,
) -> hub.MultiAdapterHub:
    """Single-bridge hub matching train semantics (apply residual once)."""

    d = int(d_model if d_model is not None else linear.d_model)
    cfg = hub.HubConfig(
        hub_active=bool(hub_active),
        selection_mode=selection_mode,
        d_model=d,
        gate_hidden=max(8, d // 128),
        include_bypass=True,
        always_on_mode="sequential",
        gate_pool="token",
        claim_boundary={
            "live_v0_switched": False,
            "capability_claim": False,
            "production_validated": False,
            "status": "B_LINEAR_WIRED_OFFLINE_ONLY",
            "arm": ARM_B,
            "validation_scope": "offline_fullseq_host_export_harness",
        },
    )
    h = hub.MultiAdapterHub(
        {BRIDGE_NAME: clone_linear_init(linear)},
        config=cfg,
        site_id="arm_b_master_once",
        d_model=d,
    )
    h.meta["arm"] = ARM_B
    h.meta["application_semantics"] = "once_matching_train"
    h.meta["v0_sites_catalog"] = list(V0_BRIDGE_SITES)
    h.meta["capability_claim"] = False
    h.meta["production_validated"] = False
    return h


def build_site_hubs(
    linear: lv.LinearSubspaceInit,
    *,
    hub_active: bool,
    selection_mode: str = "top1",
    sites: Sequence[str] | None = None,
    site_layers: Mapping[str, int] | None = None,
) -> dict[str, hub.MultiAdapterHub]:
    """One hub per V0 site with weight-cloned B residual + bypass."""

    names = list(sites) if sites is not None else list(V0_BRIDGE_SITES)
    layers = dict(site_layers or DEFAULT_SITE_LAYERS)
    out: dict[str, hub.MultiAdapterHub] = {}
    d = int(linear.d_model)
    for name in names:
        cfg = hub.HubConfig(
            hub_active=bool(hub_active),
            selection_mode=selection_mode,
            d_model=d,
            gate_hidden=max(8, d // 128),
            include_bypass=True,
            always_on_mode="sequential",
            gate_pool="token",
            claim_boundary={
                "live_v0_switched": False,
                "capability_claim": False,
                "production_validated": False,
                "status": "B_LINEAR_SITE_HUB_OFFLINE_ONLY",
                "arm": ARM_B,
                "site": name,
            },
        )
        # Register under the catalog site name so hub.choice_names includes it.
        site_hub = hub.MultiAdapterHub(
            {name: clone_linear_init(linear)},
            config=cfg,
            site_id=f"{name}@L{layers.get(name, '?')}",
            d_model=d,
        )
        site_hub.meta["arm"] = ARM_B
        site_hub.meta["dsv4f_layer"] = layers.get(name)
        site_hub.meta["capability_claim"] = False
        out[name] = site_hub
    return out


def force_hub_bypass(h: hub.MultiAdapterHub) -> None:
    hub.force_router_one_hot(h, hub.BYPASS_CHOICE, logit_scale=50.0)


def force_hub_bridge(h: hub.MultiAdapterHub, bridge_name: str | None = None) -> None:
    name = bridge_name or (h._bridge_order[0] if h._bridge_order else BRIDGE_NAME)
    hub.force_router_one_hot(h, name, logit_scale=50.0)


def resolve_hub_active_via_env_or_config(
    *,
    env_value: str | None = "1",
    config_hub_active: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Activate hub through documented env/config path (not module default).

    Temporarily sets ``FRANKENSTEIN_HUB_ACTIVE`` when ``env_value`` is not None.
    Returns (resolved_active, provenance).
    """

    prev = os.environ.get("FRANKENSTEIN_HUB_ACTIVE")
    provenance: dict[str, Any] = {
        "module_default_HUB_ACTIVE": bool(hub.HUB_ACTIVE),
        "env_key": "FRANKENSTEIN_HUB_ACTIVE",
        "env_previous": prev,
        "env_set_to": env_value,
        "config_hub_active": bool(config_hub_active),
    }
    if env_value is not None:
        os.environ["FRANKENSTEIN_HUB_ACTIVE"] = str(env_value)
    cfg = hub.HubConfig(hub_active=bool(config_hub_active), d_model=4096)
    active = cfg.resolved_active()
    provenance["resolved_active"] = bool(active)
    provenance["note"] = (
        "Module-level HUB_ACTIVE left False. Activation via env/config only."
    )
    if hub.HUB_ACTIVE is not False:
        raise BLinearHubError(
            "module-level HUB_ACTIVE was mutated; fail-closed default must stay False"
        )
    return bool(active), provenance


# ---------------------------------------------------------------------------
# Real offline activations + prompts
# ---------------------------------------------------------------------------


def load_site_layer_map(
    path: str | Path | None = None,
) -> dict[str, int]:
    p = Path(path) if path is not None else DEFAULT_SITE_MAP_SOURCE
    if not p.is_file():
        return dict(DEFAULT_SITE_LAYERS)
    doc = json.loads(p.read_text(encoding="utf-8"))
    sm = doc.get("site_map") or {}
    out: dict[str, int] = {}
    for site, row in sm.items():
        if isinstance(row, Mapping) and "dsv4f_layer" in row:
            out[str(site)] = int(row["dsv4f_layer"])
    return out or dict(DEFAULT_SITE_LAYERS)


def load_dsv4f_layer_activations(
    export_dir: str | Path,
    layer: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load real host-export [N, 4096] activations for one DSV4F layer."""

    root = Path(export_dir)
    npy = root / "activations" / f"L{int(layer):02d}.npy"
    meta_path = root / "activations" / f"L{int(layer):02d}.export.json"
    if not npy.is_file():
        raise BLinearHubError(f"activation npy missing: {npy}")
    arr = np.load(npy)
    if arr.ndim != 2 or int(arr.shape[-1]) != int(lv.DSV4F_HIDDEN):
        raise BLinearHubError(
            f"unexpected activation shape {arr.shape} at {npy}"
        )
    meta: dict[str, Any] = {
        "path": str(npy.resolve()),
        "sha256": _sha256_file(npy),
        "layer": int(layer),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "host_activation_export_is_diagnostic_only": True,
        "source": "gravity_deepseek_v4_fullseq_capture host export",
    }
    if meta_path.is_file():
        meta["export_json"] = json.loads(meta_path.read_text(encoding="utf-8"))
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return t, meta


def load_example_ids(export_dir: str | Path) -> list[str]:
    p = Path(export_dir) / "activations" / "example_ids.json"
    if not p.is_file():
        # Fall back to corpus order from receipt.
        rec = Path(export_dir) / "DSV4F_FULLSEQ_CAPTURE_RECEIPT.json"
        if rec.is_file():
            doc = json.loads(rec.read_text(encoding="utf-8"))
            ids = (doc.get("corpus_provenance") or {}).get("example_ids") or []
            return [str(x) for x in ids]
        raise BLinearHubError(f"example_ids missing under {export_dir}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict) and "example_ids" in raw:
        return [str(x) for x in raw["example_ids"]]
    raise BLinearHubError(f"unrecognized example_ids format: {p}")


def load_prompt_for_example(
    example_id: str,
    *,
    export_dir: str | Path,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve real prompt text for an example_id from traces or frozen corpus."""

    export = Path(export_dir)
    # Trace filenames use the example_id with filesystem-safe chars.
    traces = export / "traces"
    candidates = []
    if traces.is_dir():
        # exact and partial
        exact = traces / f"{example_id}.json"
        if exact.is_file():
            candidates.append(exact)
        else:
            for p in traces.glob("*.json"):
                if example_id in p.name or p.stem == example_id:
                    candidates.append(p)
                    break
    for p in candidates:
        doc = json.loads(p.read_text(encoding="utf-8"))
        text = doc.get("prompt_text")
        if not text and isinstance(doc.get("decoded_spans"), list):
            for span in doc["decoded_spans"]:
                if isinstance(span, dict) and span.get("role") == "prompt":
                    text = span.get("text")
                    break
        if text:
            return {
                "example_id": example_id,
                "prompt_text": str(text),
                "source": str(p.resolve()),
                "domain": doc.get("method_family_choice") or doc.get("domain"),
            }

    corpus = Path(corpus_path) if corpus_path else DEFAULT_FROZEN_CORPUS
    if corpus.is_file():
        doc = json.loads(corpus.read_text(encoding="utf-8"))
        for seq in doc.get("sequences") or []:
            if str(seq.get("example_id")) == example_id:
                return {
                    "example_id": example_id,
                    "prompt_text": str(seq.get("prompt_text") or ""),
                    "source": str(corpus.resolve()),
                    "domain": seq.get("domain"),
                }
    return {
        "example_id": example_id,
        "prompt_text": "",
        "source": None,
        "domain": None,
        "note": "prompt text not found; example_id retained",
    }


def pick_smoke_indices(
    example_ids: Sequence[str],
    *,
    n: int = 5,
    prefer_substrings: Sequence[str] = (
        "two_plus_two",
        "power",
        "t01_add",
        "t09_flatten",
        "gcd",
    ),
    activation_rows: torch.Tensor | None = None,
) -> list[int]:
    """Pick n real sequence indices, preferring diverse known prompts.

    When ``activation_rows`` is provided, skip indices whose hidden vector is
    byte-identical to one already chosen (the L0 fullseq host export collapses
    some sequences to shared late-layer vectors — report that honestly).
    """

    def _row_key(i: int) -> bytes | int:
        if activation_rows is None:
            return i
        return activation_rows[i].detach().cpu().contiguous().numpy().tobytes()

    idxs: list[int] = []
    seen_keys: set[bytes | int] = set()

    def _try_add(i: int) -> bool:
        if i in idxs:
            return False
        key = _row_key(i)
        if key in seen_keys:
            return False
        idxs.append(i)
        seen_keys.add(key)
        return True

    for sub in prefer_substrings:
        for i, eid in enumerate(example_ids):
            if sub in eid and _try_add(i):
                break
        if len(idxs) >= n:
            return idxs[:n]
    for i in range(len(example_ids)):
        _try_add(i)
        if len(idxs) >= n:
            break
    # If uniqueness exhausted, fill remaining with preferred/order indices.
    if len(idxs) < n:
        for i in range(len(example_ids)):
            if i not in idxs:
                idxs.append(i)
            if len(idxs) >= n:
                break
    if len(idxs) < n:
        raise BLinearHubError(
            f"need ≥{n} sequences, have {len(example_ids)}"
        )
    return idxs[:n]


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------


@torch.no_grad()
def bypass_identity_check(
    h: hub.MultiAdapterHub,
    x: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Hub active + forced bypass must reproduce plain identity (plain DSV4F path)."""

    if not h.is_active:
        raise BLinearHubError("bypass identity requires active hub")
    force_hub_bypass(h)
    y, aux = h(x, return_aux=True)
    y2, r, info = h.apply_with_residual(x)
    max_abs = float(torch.max(torch.abs(y - x)).item())
    max_r = float(torch.max(torch.abs(r)).item())
    ok = bool(torch.allclose(y, x, atol=atol, rtol=0.0)) and bool(
        torch.allclose(r, torch.zeros_like(x), atol=atol, rtol=0.0)
    )
    return {
        "pass": ok,
        "max_abs_diff_vs_input": max_abs,
        "max_abs_residual": max_r,
        "atol": atol,
        "output_stats": _finite_stats(y),
        "aux_path": aux.get("path"),
        "hub_active": bool(aux.get("hub_active")),
        "apply_with_residual_path": info.get("path"),
        "y_apply_matches_forward": bool(torch.allclose(y, y2, atol=atol)),
        "plain_dsv4f_path_definition": (
            "identity on captured student_hidden (host export); "
            "no residual applied"
        ),
    }


@torch.no_grad()
def bridge_on_forward(
    h: hub.MultiAdapterHub,
    x: torch.Tensor,
    *,
    bridge_name: str | None = None,
) -> dict[str, Any]:
    if not h.is_active:
        raise BLinearHubError("bridge-on requires active hub")
    force_hub_bridge(h, bridge_name)
    y, aux = h(x, return_aux=True)
    y2, r, info = h.apply_with_residual(x)
    # Reference residual from first bridge module
    name = bridge_name or h._bridge_order[0]
    r_ref = hub.module_residual(h.bridges[name], x)
    return {
        "output_stats": _finite_stats(y),
        "residual_stats": _finite_stats(r),
        "input_stats": _finite_stats(x),
        "matches_module_residual": bool(torch.allclose(r, r_ref, atol=1e-5)),
        "max_abs_residual_vs_ref": float(torch.max(torch.abs(r - r_ref)).item()),
        "aux_path": aux.get("path"),
        "hub_active": bool(aux.get("hub_active")),
        "selected": aux.get("selected"),
        "y_equals_x_plus_r": bool(torch.allclose(y2, x + r, atol=1e-5)),
        "differs_from_input": bool(not torch.allclose(y, x, atol=1e-6)),
    }


@torch.no_grad()
def compare_off_on(
    h: hub.MultiAdapterHub,
    x: torch.Tensor,
    *,
    bridge_name: str | None = None,
    action_proj: nn.Linear | None = None,
) -> dict[str, Any]:
    """Side-by-side bridge-off (bypass) vs bridge-on on one hidden row/batch."""

    force_hub_bypass(h)
    y_off = h(x)
    force_hub_bridge(h, bridge_name)
    y_on = h(x)
    delta = y_on - y_off
    row = {
        "bridge_off_stats": _finite_stats(y_off),
        "bridge_on_stats": _finite_stats(y_on),
        "delta_stats": _finite_stats(delta),
        "max_abs_delta": float(torch.max(torch.abs(delta)).item()),
        "cosine_off_on": float(
            torch.nn.functional.cosine_similarity(
                y_off.reshape(1, -1), y_on.reshape(1, -1)
            ).item()
        )
        if y_off.numel() and y_on.numel()
        else None,
        "norm_ratio_on_over_off": (
            float(
                torch.linalg.vector_norm(y_on.reshape(-1)).item()
                / max(torch.linalg.vector_norm(y_off.reshape(-1)).item(), 1e-12)
            )
        ),
        "identical": bool(torch.equal(y_off, y_on)),
        "near_identical_1e6": bool(torch.allclose(y_off, y_on, atol=1e-6)),
    }
    # Pseudo next-class via stack action_proj (NOT lm_head token prediction).
    # lm_head logits were not captured in the fullseq host export.
    if action_proj is not None:

        def _pool(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 1:
                return t.unsqueeze(0)
            if t.dim() == 2:
                return t  # [B, D] — keep batch
            if t.dim() == 3:
                return t.mean(dim=1)  # [B, S, D] → [B, D]
            return t.reshape(1, -1)

        po = action_proj(_pool(y_off))
        pn = action_proj(_pool(y_on))
        row["action_proj_note"] = (
            "NOT vocabulary lm_head next-token. Host export has logits_captured=false. "
            "This is arm-B action_proj (12-way) over pooled hidden — diagnostic only."
        )
        row["action_logits_off_top3"] = _topk_logits(po[0], k=3)
        row["action_logits_on_top3"] = _topk_logits(pn[0], k=3)
        row["action_argmax_off"] = int(torch.argmax(po[0]).item())
        row["action_argmax_on"] = int(torch.argmax(pn[0]).item())
        row["action_argmax_changed"] = row["action_argmax_off"] != row["action_argmax_on"]
    return row


def _topk_logits(logits: torch.Tensor, k: int = 3) -> list[dict[str, Any]]:
    k = min(k, int(logits.numel()))
    vals, idxs = torch.topk(logits.float(), k=k)
    return [
        {"index": int(idxs[i].item()), "logit": float(vals[i].item())}
        for i in range(k)
    ]


@torch.no_grad()
def run_offline_smoke(
    loaded: LoadedBLinear,
    *,
    export_dir: str | Path = DEFAULT_DSV4F_EXPORT,
    corpus_path: str | Path | None = None,
    n_prompts: int = 5,
    site_layers: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Real offline harness smoke: bypass identity + bridge on/off on real prompts."""

    export = Path(export_dir)
    layers = dict(site_layers or load_site_layer_map())
    example_ids = load_example_ids(export)

    # Activate via env override (documented path).
    active, act_prov = resolve_hub_active_via_env_or_config(
        env_value="1", config_hub_active=True
    )
    if not active:
        raise BLinearHubError("hub failed to resolve active via env/config")

    linear = loaded.stack.linear_init
    assert linear is not None
    master = build_master_hub(linear, hub_active=True, selection_mode="top1")
    site_hubs = build_site_hubs(
        linear, hub_active=True, selection_mode="top1", site_layers=layers
    )

    # Use late consolidation layer as the master smoke surface (final_hidden band).
    master_layer = int(layers.get("GLM_LATE_CONSOLIDATION_BRIDGE", 37))
    acts, act_meta = load_dsv4f_layer_activations(export, master_layer)
    # [N,D] → treat each row as a token-free pooled hidden (matches train seq_len=1)
    x_all = acts  # [N, 4096]
    indices = pick_smoke_indices(
        example_ids, n=n_prompts, activation_rows=x_all
    )
    # Record host-export uniqueness (observed: many sequences share identical rows).
    row_hashes = [
        hashlib.sha256(x_all[i].numpy().tobytes()).hexdigest()[:16]
        for i in range(int(x_all.shape[0]))
    ]
    act_meta["n_unique_rows"] = len(set(row_hashes))
    act_meta["n_rows"] = len(row_hashes)
    act_meta["unique_row_note"] = (
        "Host-export late_hidden rows are not all unique across the 32-sequence "
        "L0 corpus (shared vectors across some example_ids). Smoke prefers "
        "distinct rows when available."
    )

    # Bypass identity on full batch
    bypass = bypass_identity_check(master, x_all)
    bridge_on = bridge_on_forward(master, x_all)

    # Per-prompt comparisons
    prompt_rows: list[dict[str, Any]] = []
    for idx in indices:
        eid = example_ids[idx]
        prompt = load_prompt_for_example(
            eid, export_dir=export, corpus_path=corpus_path
        )
        x = x_all[idx : idx + 1]  # [1, D]
        cmp = compare_off_on(
            master, x, action_proj=loaded.stack.action_proj
        )
        # Also apply each site hub on its own layer's activation for this index
        site_rows: dict[str, Any] = {}
        for site, sh in site_hubs.items():
            L = int(layers[site])
            site_acts, _ = load_dsv4f_layer_activations(export, L)
            xs = site_acts[idx : idx + 1]
            force_hub_bypass(sh)
            y_off = sh(xs)
            force_hub_bridge(sh, site)
            y_on = sh(xs)
            site_rows[site] = {
                "dsv4f_layer": L,
                "max_abs_delta": float(torch.max(torch.abs(y_on - y_off)).item()),
                "off_finite": bool(torch.isfinite(y_off).all().item()),
                "on_finite": bool(torch.isfinite(y_on).all().item()),
                "off_rms": float(torch.sqrt(torch.mean(y_off * y_off)).item()),
                "on_rms": float(torch.sqrt(torch.mean(y_on * y_on)).item()),
                "norm_ratio": float(
                    torch.linalg.vector_norm(y_on.reshape(-1)).item()
                    / max(torch.linalg.vector_norm(y_off.reshape(-1)).item(), 1e-12)
                ),
            }
        prompt_rows.append(
            {
                "index": idx,
                "example_id": eid,
                "prompt_text": prompt.get("prompt_text"),
                "prompt_source": prompt.get("source"),
                "domain": prompt.get("domain"),
                "master_layer": master_layer,
                "comparison": cmp,
                "per_site": site_rows,
            }
        )

    # Site hub bypass identity (one site, full batch of its layer)
    site_bypass: dict[str, Any] = {}
    for site, sh in list(site_hubs.items())[:1]:
        L = int(layers[site])
        sa, _ = load_dsv4f_layer_activations(export, L)
        site_bypass[site] = bypass_identity_check(sh, sa)

    # Aggregate health
    all_finite = bool(bypass["output_stats"]["finite"] and bridge_on["output_stats"]["finite"])
    norms_bounded = True
    for row in prompt_rows:
        if float(row["comparison"]["norm_ratio_on_over_off"]) > MAX_NORM_RATIO:
            norms_bounded = False
        for srow in row["per_site"].values():
            if float(srow["norm_ratio"]) > MAX_NORM_RATIO:
                norms_bounded = False
            if not (srow["off_finite"] and srow["on_finite"]):
                all_finite = False

    # Restore env after smoke (leave process clean)
    env_prev = act_prov.get("env_previous")
    if env_prev is None:
        os.environ.pop("FRANKENSTEIN_HUB_ACTIVE", None)
    else:
        os.environ["FRANKENSTEIN_HUB_ACTIVE"] = str(env_prev)

    return {
        "schema": SCHEMA_SMOKE,
        "capability_claim": False,
        "fabricated": False,
        "recorded_at": _utc_now(),
        "validation_scope": (
            "offline harness only — real DSV4F host-export activations from "
            "gravity_deepseek_v4_fullseq_capture; NOT production Rust/Metal "
            "token-by-token serving; lm_head logits were not captured "
            "(logits_captured=false in fullseq receipt)"
        ),
        "hub_activation": act_prov,
        "module_HUB_ACTIVE_unchanged": hub.HUB_ACTIVE is False,
        "checkpoint": {
            "path": str(loaded.checkpoint_path),
            "sha256": loaded.checkpoint_sha256,
            "load_info": loaded.load_info,
            "metrics": loaded.blob_metrics,
        },
        "activations": act_meta,
        "master_layer": master_layer,
        "site_layers": layers,
        "bypass_identity": bypass,
        "bridge_on_batch": bridge_on,
        "site_bypass_sample": site_bypass,
        "n_prompts": len(prompt_rows),
        "prompts": prompt_rows,
        "health": {
            "no_exception": True,
            "all_finite": all_finite,
            "bypass_identity_pass": bool(bypass["pass"]),
            "norms_bounded": norms_bounded,
            "max_norm_ratio_threshold": MAX_NORM_RATIO,
            "bridge_on_differs_from_off": any(
                not r["comparison"]["near_identical_1e6"] for r in prompt_rows
            ),
        },
        "action_proj_disclaimer": (
            "action_argmax fields are 12-way action_proj diagnostics on pooled "
            "hidden states — not vocabulary next-token predictions. No real "
            "held-out math capability eval is performed here."
        ),
    }


# ---------------------------------------------------------------------------
# Desktop seal
# ---------------------------------------------------------------------------


def body_reference(body_path: str | Path | None = None) -> dict[str, Any]:
    body = Path(body_path) if body_path is not None else DEFAULT_BODY
    body = body.resolve() if body.exists() else body
    try:
        rel = str(body.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        rel = None
    ref: dict[str, Any] = {
        "path": str(body),
        "repo_relative_path": rel
        or "workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity",
        "exists": body.exists(),
        "is_dir": body.is_dir(),
        "note": (
            "Sealed DeepSeek-V4-Flash Gravity body — referenced only, never "
            "copied or mutated by this artifact. Resolve repo_relative_path "
            "against any hawking checkout (not a Grok worktree)."
        ),
    }
    manifest = body / "manifest.json"
    if manifest.is_file():
        ref["manifest_path"] = str(manifest)
        ref["manifest_sha256"] = _sha256_file(manifest)
        ref["manifest_bytes"] = manifest.stat().st_size
    restart = body / "restart-receipt.json"
    if restart.is_file():
        try:
            doc = json.loads(restart.read_text(encoding="utf-8"))
            ref["restart_receipt_seal_sha256"] = doc.get("seal_sha256")
            ref["restart_receipt_schema"] = doc.get("schema")
        except json.JSONDecodeError:
            ref["restart_receipt_seal_sha256"] = None
    return ref


def write_standalone_loader(path: Path) -> Path:
    """Minimal torch-only loader so Desktop artifact is reconstructable offline."""

    src = r'''#!/usr/bin/env python3
"""Standalone loader for the B_LINEAR hub Desktop artifact.

Requires only: Python 3 + torch. Reconstructs the residual and applies
bypass vs bridge-on without the Grok worktree or full hawking lab package.

Usage:
  python load_hub.py --artifact-dir . --hidden-npy /path/to/L37.npy
  python load_hub.py --artifact-dir . --self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSubspaceInit(nn.Module):
    def __init__(self, d_model: int = 4096, rank: int = 16, scale: float = 0.05):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.scale = float(scale)
        self.a_lo = nn.Linear(d_model, rank, bias=False)
        self.a_hi = nn.Linear(rank, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (self.a_hi(self.a_lo(x)) + self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class TinyRouter(nn.Module):
    def __init__(self, d_model: int, n_choices: int, gate_hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, gate_hidden)
        self.fc2 = nn.Linear(gate_hidden, n_choices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean-pool if sequence
        h = x.mean(dim=-2) if x.dim() >= 3 else x
        return self.fc2(F.silu(self.fc1(h)))


def load_linear_from_checkpoint(ckpt_path: Path) -> LinearSubspaceInit:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert blob.get("capability_claim") is not True
    state = blob["runtime_state_dict"]
    spec = blob.get("spec") or {}
    mod = LinearSubspaceInit(
        d_model=int(spec.get("d_student") or 4096),
        rank=int(spec.get("rank") or 16),
        scale=0.05,
    )
    # Map linear_init.* keys
    mapped = {}
    for k, v in state.items():
        if k.startswith("linear_init."):
            mapped[k[len("linear_init.") :]] = v
    missing, unexpected = mod.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected
    return mod


def force_one_hot(router: TinyRouter, idx: int, scale: float = 50.0) -> None:
    with torch.no_grad():
        router.fc2.weight.zero_()
        router.fc2.bias.zero_()
        router.fc2.bias[idx] = float(scale)


@torch.no_grad()
def apply(mod: LinearSubspaceInit, x: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "bypass":
        return x
    if mode == "bridge_on":
        return mod(x)
    raise SystemExit(f"unknown mode {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=Path("."))
    ap.add_argument("--hidden-npy", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    ckpt = args.artifact_dir / "BEST_BALANCED.pt"
    receipt = args.artifact_dir / "ARTIFACT_RECEIPT.json"
    if receipt.is_file():
        rec = json.loads(receipt.read_text())
        assert rec.get("capability_claim") is not True
        print("receipt_arm", rec.get("arm"))
        print("receipt_capability_claim", rec.get("capability_claim"))
    mod = load_linear_from_checkpoint(ckpt)
    print("loaded_linear", mod.d_model, mod.rank, "scale", mod.scale)
    if args.self_test:
        x = torch.randn(4, mod.d_model)
        y_off = apply(mod, x, mode="bypass")
        y_on = apply(mod, x, mode="bridge_on")
        assert torch.equal(y_off, x)
        assert torch.isfinite(y_on).all()
        print("self_test_ok", "max_abs_delta", float((y_on - y_off).abs().max()))
        return
    if args.hidden_npy is None:
        raise SystemExit("pass --hidden-npy or --self-test")
    import numpy as np
    arr = np.load(args.hidden_npy)
    x = torch.from_numpy(arr.astype("float32"))
    y_off = apply(mod, x, mode="bypass")
    y_on = apply(mod, x, mode="bridge_on")
    print("shape", list(x.shape))
    print("bypass_identity", bool(torch.equal(y_off, x)))
    print("bridge_on_finite", bool(torch.isfinite(y_on).all()))
    print("max_abs_delta", float((y_on - y_off).abs().max()))


if __name__ == "__main__":
    main()
'''
    path.write_text(src, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def seal_desktop_artifact(
    loaded: LoadedBLinear,
    smoke: Mapping[str, Any],
    *,
    desktop_dir: str | Path = DEFAULT_DESKTOP,
    evidence_dir: str | Path = DEFAULT_EVIDENCE_OUT,
    body_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write loadable Desktop artifact + mirror evidence receipt."""

    desk = Path(desktop_dir)
    # Keep retracted files; write into a dedicated subdir.
    out = desk / "b-linear-hub-artifact"
    out.mkdir(parents=True, exist_ok=True)
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint into artifact (self-contained load).
    ckpt_dst = out / "BEST_BALANCED.pt"
    shutil.copy2(loaded.checkpoint_path, ckpt_dst)

    body_ref = body_reference(body_path)
    write_standalone_loader(out / "load_hub.py")

    # Hub wire config (hub_active true for this artifact's intended offline use;
    # module default elsewhere remains False).
    wire_cfg = hub.HubConfig(
        hub_active=True,
        selection_mode="top1",
        d_model=int(loaded.blob_spec.get("d_student") or 4096),
        gate_hidden=32,
        include_bypass=True,
        always_on_mode="sequential",
        gate_pool="token",
        claim_boundary={
            "capability_claim": False,
            "production_validated": False,
            "live_v0_switched": False,
            "status": "B_LINEAR_DESKTOP_ARTIFACT",
        },
    )
    hub_cfg_path = out / "HUB_CONFIG.json"
    hub_cfg_path.write_text(
        json.dumps(wire_cfg.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    site_layers = smoke.get("site_layers") or DEFAULT_SITE_LAYERS
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "name": "PROTO_FRANKENSTEIN_B_LINEAR_HUB_DESKTOP",
        "arm": ARM_B,
        "capability_claim": False,
        "fabricated": False,
        "recorded_at": _utc_now(),
        "status": "SEALED_OFFLINE_SMOKE_ONLY",
        "what_this_is": (
            "DeepSeek-V4-Flash sealed Gravity body (referenced by path+hash, not "
            "copied) + honestly-trained B_LINEAR_SUBSPACE_INITIALIZATION residual "
            "weights wired into MultiAdapterHub attachment points at the 8 named "
            "V0 bridge sites. Activation is via env FRANKENSTEIN_HUB_ACTIVE=1 / "
            "config hub_active=true — module default HUB_ACTIVE remains False."
        ),
        "what_this_is_not": [
            "Not G_COMPLETE_V0 (full composed arm remains HOLD / not part of this artifact)",
            "Not a held-out math capability result",
            "Not production Rust/Metal serve-path integration",
            "Not Kimi stage 2",
            "Not 30B/Qwen work",
        ],
        "g_complete_v0_status": {
            "arm": "G_COMPLETE_V0",
            "verdict": "HOLD",
            "included_in_this_artifact": False,
            "reference": (
                "workspace/campaign/evidence/models/frankenstein/"
                "proto-v0-real-fix/DIAGNOSIS.json"
            ),
            "note": (
                "complete_v0_beats_linear_init failed on real retrain (ep2 and ep8); "
                "owner closed the GLM stage with B_LINEAR as the honest working piece."
            ),
        },
        "student_body": body_ref,
        "bridge_weights": {
            "arm": ARM_B,
            "checkpoint_slot": "BEST_BALANCED",
            "checkpoint_path_in_artifact": "BEST_BALANCED.pt",
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "source_checkpoint": str(loaded.checkpoint_path),
            "metrics_from_checkpoint": loaded.blob_metrics,
            "final_eval_loss_note": (
                "Checkpoint-embedded metrics are slot proxies; the campaign "
                "final_eval_loss for arm B at ep8 is 0.011643… from "
                "BEST_BALANCED reeval phase E (DIAGNOSIS.json). That number is "
                "reconstruction/self-consistency, not held-out math."
            ),
            "ep8_final_eval_loss_from_diagnosis": 0.011643319390714169,
            "modules": {
                "linear_init": True,
                "teacher_projector": False,
                "student_observer": False,
                "interventions": [],
                "behavior_heads": False,
                "route_residual": False,
            },
            "application_semantics": (
                "Training applies LinearSubspaceInit once. Master hub mirrors "
                "that. Per-site hubs hold weight clones at the 8 named sites as "
                "attachment points; do not cascade all 8 on one hidden vector "
                "unless intentionally measuring multi-site injection."
            ),
        },
        "hub": {
            "module": "lab.operators.frankenstein_adapter_hub.MultiAdapterHub",
            "module_default_HUB_ACTIVE": False,
            "activation": {
                "env": "FRANKENSTEIN_HUB_ACTIVE=1",
                "config_field": "hub_active=true",
                "config_file": "HUB_CONFIG.json",
            },
            "sites": list(V0_BRIDGE_SITES),
            "site_layers": site_layers,
            "include_bypass": True,
            "selection_mode": "top1",
            "native_moe_router_touched": False,
        },
        "validation": {
            "scope": smoke.get("validation_scope"),
            "bypass_identity_pass": (smoke.get("health") or {}).get(
                "bypass_identity_pass"
            ),
            "all_finite": (smoke.get("health") or {}).get("all_finite"),
            "norms_bounded": (smoke.get("health") or {}).get("norms_bounded"),
            "n_prompts": smoke.get("n_prompts"),
            "smoke_schema": smoke.get("schema"),
            "production_rust_metal_serving": False,
            "lm_head_logits_available": False,
        },
        "load_instructions": {
            "standalone": "python load_hub.py --artifact-dir . --self-test",
            "with_repo": (
                "python -m lab.operators.frankenstein_b_linear_hub_wire "
                "load --checkpoint BEST_BALANCED.pt"
            ),
            "reconstruct_hub": (
                "load_b_linear_checkpoint → build_master_hub / build_site_hubs "
                "with FRANKENSTEIN_HUB_ACTIVE=1"
            ),
            "place_on_desktop": (
                "workspace/campaign/evidence/models/frankenstein/"
                "proto-v0-desktop-artifact/place_on_desktop.sh"
            ),
        },
        "desktop_placement": {
            "intended_path": str(
                Path.home()
                / "Desktop"
                / "hawking-frankenstein"
                / "proto-frankenstein"
                / "b-linear-hub-artifact"
            ),
            "note": (
                "Agent sandbox blocked writes under ~/Desktop (macOS TCC / "
                "seatbelt Operation not permitted). Artifact is fully sealed "
                "under evidence desktop-staging and is byte-identical to the "
                "intended Desktop layout; run place_on_desktop.sh outside the "
                "sandbox to land it on Desktop."
            ),
        },
        "claim_boundary": {
            "capability_claim": False,
            "is_inheritance": False,
            "is_proto_frankenstein_complete": False,
            "promotion": "not claimed",
            "forbidden_labels": [
                "PROTO_FRANKENSTEIN_COMPLETE",
                "MATH_INHERITANCE_COMPLETE",
                "FUNCTIONAL_TRANSFER_COMPLETE",
                "G_COMPLETE_V0_ACCEPT",
            ],
        },
    }
    sealed = seal(receipt)
    (out / "ARTIFACT_RECEIPT.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "SMOKE_TEST.json").write_text(
        json.dumps(dict(smoke), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    # Evidence mirror
    (evidence / "ARTIFACT_RECEIPT.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "SMOKE_TEST.json").write_text(
        json.dumps(dict(smoke), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    # Lightweight wire doc
    wire_doc = {
        "schema": SCHEMA_WIRE,
        "capability_claim": False,
        "recorded_at": _utc_now(),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "desktop_dir": str(out),
        "module_HUB_ACTIVE_default": False,
        "sites": list(V0_BRIDGE_SITES),
    }
    (evidence / "WIRE.json").write_text(
        json.dumps(seal(wire_doc), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    listing = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            listing.append(
                {
                    "path": str(p.relative_to(out)),
                    "bytes": p.stat().st_size,
                    "sha256": _sha256_file(p)
                    if p.suffix in {".json", ".pt", ".py"} and p.stat().st_size < 5_000_000
                    else (
                        _sha256_file(p) if p.name == "BEST_BALANCED.pt" else None
                    ),
                }
            )
    # Always hash checkpoint
    for row in listing:
        if row["path"] == "BEST_BALANCED.pt" and row["sha256"] is None:
            row["sha256"] = _sha256_file(out / "BEST_BALANCED.pt")

    result = {
        "desktop_dir": str(out),
        "evidence_dir": str(evidence),
        "receipt_seal_sha256": sealed.get("seal_sha256"),
        "listing": listing,
        "capability_claim": False,
    }
    (evidence / "DESKTOP_SEAL.json").write_text(
        json.dumps(seal(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "LISTING.json").write_text(
        json.dumps(listing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_load(args: argparse.Namespace) -> int:
    loaded = load_b_linear_checkpoint(args.checkpoint)
    print(json.dumps(
        {
            "ok": True,
            "path": str(loaded.checkpoint_path),
            "sha256": loaded.checkpoint_sha256,
            "load_info": loaded.load_info,
            "metrics": loaded.blob_metrics,
            "capability_claim": False,
            "module_HUB_ACTIVE": hub.HUB_ACTIVE,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    loaded = load_b_linear_checkpoint(args.checkpoint)
    smoke = run_offline_smoke(
        loaded,
        export_dir=args.export_dir,
        n_prompts=args.n_prompts,
    )
    out = Path(args.out) if args.out else None
    text = json.dumps(smoke, indent=2, sort_keys=True, default=str) + "\n"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    health = smoke.get("health") or {}
    return 0 if health.get("bypass_identity_pass") and health.get("all_finite") else 2


def _cmd_seal(args: argparse.Namespace) -> int:
    loaded = load_b_linear_checkpoint(args.checkpoint)
    smoke = run_offline_smoke(
        loaded,
        export_dir=args.export_dir,
        n_prompts=args.n_prompts,
    )
    result = seal_desktop_artifact(
        loaded,
        smoke,
        desktop_dir=args.desktop_dir,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    health = smoke.get("health") or {}
    return 0 if health.get("bypass_identity_pass") and health.get("all_finite") else 2


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("load", help="Load BEST_BALANCED and print metadata")
    pl.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    pl.set_defaults(func=_cmd_load)

    ps = sub.add_parser("smoke", help="Offline bypass + bridge-on smoke")
    ps.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ps.add_argument("--export-dir", type=Path, default=DEFAULT_DSV4F_EXPORT)
    ps.add_argument("--n-prompts", type=int, default=5)
    ps.add_argument("--out", type=Path, default=None)
    ps.set_defaults(func=_cmd_smoke)

    pe = sub.add_parser("seal", help="Smoke + seal Desktop artifact")
    pe.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    pe.add_argument("--export-dir", type=Path, default=DEFAULT_DSV4F_EXPORT)
    pe.add_argument("--n-prompts", type=int, default=5)
    pe.add_argument("--desktop-dir", type=Path, default=DEFAULT_DESKTOP)
    pe.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_OUT)
    pe.set_defaults(func=_cmd_seal)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
