#!/usr/bin/env python3.12
"""Conditional multi-adapter hub — Rank-1 upgrade, inactive by default.

Research verdict (FRANKENSTEIN_ARCHITECTURE_OPTIONS.md Rank 1): a tiny
*external* router selecting among residual bridges is the one real upgrade
over always-on residual addition.  It addresses:

  (a) always-on residual interference when a bridge fires on irrelevant inputs
  (b) GLM × Kimi co-location at shared transplant sites (see
      KIMI_STRATEGIC_BRIDGE_CONTRACT.json ``site_overlap`` notes)

Formula (when active)::

    Hub(x) = x + Σ_i gate_i(x) · residual_i(x)

where ``{A_i}`` are the existing named bridges (GLM_METHOD_BRIDGE, … plus
Kimi strategic children), and ``gate_i`` is a *separate* cheap classifier
over the local hidden state — **not** DeepSeek's native 256-expert top-6
router.

Default::

    HUB_ACTIVE = False

When inactive, forward is **byte-identical** to the current always-on
sequential residual path.  Live V0 bridges keep running as always-on.
Flip is a one-line / one-field config change after the activation trigger
is earned (secondary regression beyond tolerance, or confirmed site
collision).  Fail-closed until then.

New-file lane: wraps ReversibleBridge / LowRankAdapter / StudentIntervention
interfaces; does not reimplement them and does not touch the live
recapture-fixed-corpus worktree.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.operators.frankenstein_adapter_modules import (
    LowRankAdapter,
    ReversibleBridge,
    param_bytes,
    param_count,
    tensor_content_hash,
)
from lab.operators.frankenstein_bridges import V0_BRIDGE_SITES as GLM_V0_BRIDGE_SITES
from lab.operators.frankenstein_promotion_gate import SECONDARY_TOLERANCE
from lab.receipts import seal


# ---------------------------------------------------------------------------
# Schemas / catalog
# ---------------------------------------------------------------------------

HUB_MODULE_SCHEMA = "hawking.frankenstein.multi_adapter_hub.v1"
HUB_CONFIG_SCHEMA = "hawking.frankenstein.multi_adapter_hub_config.v1"
HUB_DESIGN_SCHEMA = "hawking.frankenstein.multi_adapter_hub_design.v1"
HUB_ACTIVATION_GATE = "REQUIRES_HUB_ACTIVATION_SIGNAL"

# Module-level master switch.  Default false — live V0 path stays always-on.
# One-line flip when the documented eval signal is earned:
#   HUB_ACTIVE = True
# or set env FRANKENSTEIN_HUB_ACTIVE=1 / config hub_active=true.
HUB_ACTIVE: bool = False

# Bypass is always choice index 0 in the external router.
BYPASS_CHOICE = "A_BYPASS"
BYPASS_INDEX = 0

# Named bridge catalog (architecture-agnostic; usable by GLM and Kimi passes).
# Source: DSV4F latent V0 sites + KIMI_STRATEGIC_BRIDGE_CONTRACT children.
KIMI_NAMED_BRIDGES: tuple[str, ...] = (
    "KIMI_PLANNING_BRIDGE",
    "KIMI_TOOL_POLICY_BRIDGE",
    "KIMI_LONG_HORIZON_BRIDGE",
    "KIMI_CRITIQUE_BRIDGE",
    "KIMI_CONTEXT_MGMT_BRIDGE",
)

# Parent label preserved for stage-2 handoff; not a residual site itself.
KIMI_PARENT_BRIDGE = "KIMI_STRATEGIC_BRIDGE"

NAMED_BRIDGE_CATALOG: tuple[str, ...] = tuple(GLM_V0_BRIDGE_SITES) + KIMI_NAMED_BRIDGES

# Documented co-location risks (from KIMI_STRATEGIC_BRIDGE_CONTRACT site_overlap).
DEFAULT_COLLISION_SITES: tuple[dict[str, Any], ...] = (
    {
        "transplant_point": "post_attention_hidden_state",
        "student_layers_hint": list(range(14, 23)),
        "contenders": ["GLM_DECOMPOSITION_BRIDGE", "KIMI_PLANNING_BRIDGE"],
        "legacy_workaround": "disjoint block_ids, do not overwrite",
        "hub_resolution": "gate selects EITHER (top-1) or a learned soft mix — never forced dual-add",
    },
    {
        "transplant_point": "pre_router_hidden_state",
        "student_layers_hint": "shared pre_router layers when measured",
        "contenders": [
            "GLM_PRE_ROUTER_BRIDGE",
            "GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL",
            "KIMI_TOOL_POLICY_BRIDGE",
        ],
        "legacy_workaround": "distinct block_ids; no shared parameter tensors",
        "hub_resolution": "gate selects among contenders + bypass; exclusive top-1 available",
    },
)

# Default evidence path for the sealed design + optional live config.
DEFAULT_EVIDENCE_DIR = Path(
    "workspace/campaign/evidence/models/frankenstein"
)
DEFAULT_HUB_CONFIG_NAME = "MULTI_ADAPTER_HUB_CONFIG.json"
DEFAULT_HUB_DESIGN_NAME = "MULTI_ADAPTER_HUB_DESIGN.json"

SELECTION_MODES: tuple[str, ...] = ("soft", "top1")


class HubError(RuntimeError):
    """Multi-adapter hub error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_hub_active_override() -> bool | None:
    raw = os.environ.get("FRANKENSTEIN_HUB_ACTIVE")
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise HubError(
        f"FRANKENSTEIN_HUB_ACTIVE must be boolean-ish, got {raw!r}"
    )


# ---------------------------------------------------------------------------
# Config (one-field flip)
# ---------------------------------------------------------------------------


@dataclass
class HubConfig:
    """Runtime config for the multi-adapter hub.

    ``hub_active`` defaults **false**.  Changing only this field is the
    intended activation path after the eval signal is earned.
    """

    hub_active: bool = False
    selection_mode: str = "soft"  # soft | top1
    gate_hidden: int = 32
    temperature: float = 1.0
    include_bypass: bool = True
    # When inactive: sequential residual apply (matches AdapterBank / bridges).
    always_on_mode: str = "sequential"  # sequential | parallel_sum
    # Pool tokens before gate ("token" = per-token logits; "mean" = sequence).
    gate_pool: str = "token"
    d_model: int = 64
    claim_boundary: dict[str, Any] = field(
        default_factory=lambda: {
            "live_v0_switched": False,
            "capability_claim": False,
            "production_validated": False,
            "status": "READY_INACTIVE_SCAFFOLD",
        }
    )

    def __post_init__(self) -> None:
        mode = str(self.selection_mode).lower()
        if mode not in SELECTION_MODES:
            raise HubError(
                f"selection_mode must be one of {SELECTION_MODES}, got {mode!r}"
            )
        self.selection_mode = mode
        aom = str(self.always_on_mode).lower()
        if aom not in {"sequential", "parallel_sum"}:
            raise HubError(f"always_on_mode invalid: {aom!r}")
        self.always_on_mode = aom
        pool = str(self.gate_pool).lower()
        if pool not in {"token", "mean"}:
            raise HubError(f"gate_pool invalid: {pool!r}")
        self.gate_pool = pool
        if float(self.temperature) <= 0.0:
            raise HubError("temperature must be > 0")

    def resolved_active(self) -> bool:
        """Effective active flag: env override > instance > module default."""

        env = _env_hub_active_override()
        if env is not None:
            return env
        return bool(self.hub_active)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HUB_CONFIG_SCHEMA,
            **asdict(self),
            "module_default_HUB_ACTIVE": bool(HUB_ACTIVE),
            "env_override_key": "FRANKENSTEIN_HUB_ACTIVE",
            "flip_instructions": (
                "Set hub_active=true in this config (or HUB_ACTIVE=True in "
                "frankenstein_adapter_hub.py, or FRANKENSTEIN_HUB_ACTIVE=1) "
                "ONLY after activation_trigger evaluates true.  Do not flip "
                "before A–G ablation / auto-bench shows secondary regression "
                "beyond tolerance or a confirmed transplant-site collision."
            ),
        }


def default_hub_config(*, d_model: int = 64, hub_active: bool | None = None) -> HubConfig:
    """Build default config.  ``hub_active`` defaults to module ``HUB_ACTIVE``."""

    active = bool(HUB_ACTIVE) if hub_active is None else bool(hub_active)
    return HubConfig(hub_active=active, d_model=int(d_model))


def load_hub_config(path: str | Path | None = None, *, d_model: int = 64) -> HubConfig:
    """Load hub config from JSON, or return defaults if absent.

    Missing file is not an error — fail-open to inactive defaults so the
    live always-on path is never blocked by missing hub scaffolding.
    """

    if path is None:
        return default_hub_config(d_model=d_model)
    p = Path(path)
    if not p.is_file():
        return default_hub_config(d_model=d_model)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise HubError(f"hub config at {p} is not a JSON object")
    known = {
        "hub_active",
        "selection_mode",
        "gate_hidden",
        "temperature",
        "include_bypass",
        "always_on_mode",
        "gate_pool",
        "d_model",
        "claim_boundary",
    }
    kwargs = {k: doc[k] for k in known if k in doc}
    if "d_model" not in kwargs:
        kwargs["d_model"] = d_model
    return HubConfig(**kwargs)


def write_default_hub_config(path: str | Path, *, cfg: HubConfig | None = None) -> Path:
    """Write a one-field-flip config JSON (hub_active defaults false)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = cfg or default_hub_config()
    doc = c.to_dict()
    doc["recorded_at"] = _utc_now()
    sealed = seal(doc)
    p.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Residual extraction (duck-type over existing bridge modules)
# ---------------------------------------------------------------------------


def module_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract residual r such that forward ≈ x + r for known bridge types.

    Supports:
      - ReversibleBridge / StudentIntervention-like: ``.residual(x)``
      - LowRankAdapter: ``scale * delta(x)``
      - generic: ``forward(x) - x`` (last resort)
    """

    if hasattr(module, "residual") and callable(getattr(module, "residual")):
        return module.residual(x)  # type: ignore[operator]
    if isinstance(module, LowRankAdapter):
        return module.scale * module.delta(x)
    if hasattr(module, "delta") and callable(getattr(module, "delta")):
        scale = float(getattr(module, "scale", 1.0))
        return scale * module.delta(x)  # type: ignore[operator]
    # Generic residual form.
    y = module(x)
    return y - x


def always_on_apply(
    x: torch.Tensor,
    bridges: Mapping[str, nn.Module],
    *,
    mode: str = "sequential",
    order: Sequence[str] | None = None,
) -> torch.Tensor:
    """Current campaign always-on residual path (reference for inactive hub).

    sequential: y = A_n(...A_1(x)...) — matches AdapterBank.apply_residual
    parallel_sum: y = x + Σ residual_i(x) — matches composition_law sum form
    """

    names = list(order) if order is not None else list(bridges.keys())
    if mode == "sequential":
        y = x
        for name in names:
            y = bridges[name](y)
        return y
    if mode == "parallel_sum":
        y = x
        for name in names:
            y = y + module_residual(bridges[name], x)
        return y
    raise HubError(f"unknown always_on mode: {mode!r}")


# ---------------------------------------------------------------------------
# Tiny external router (NOT the native 256-expert MoE router)
# ---------------------------------------------------------------------------


class TinyExternalRouter(nn.Module):
    """Cheap classifier over local hidden state → logits over {bypass, A_i…}.

    Deliberately tiny (gate_hidden << d_model).  Does not touch DeepSeek's
    native router weights or expert count.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_choices: int,
        gate_hidden: int = 32,
        pool: str = "token",
    ) -> None:
        super().__init__()
        if n_choices < 1:
            raise HubError("n_choices must be >= 1")
        self.d_model = int(d_model)
        self.n_choices = int(n_choices)
        self.gate_hidden = int(gate_hidden)
        self.pool = str(pool)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, self.gate_hidden, bias=True)
        self.fc2 = nn.Linear(self.gate_hidden, self.n_choices, bias=True)
        # Near-uniform init so early soft mix is almost always-on-equal / bypass-friendly.
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean" and x.dim() >= 3:
            return x.mean(dim=-2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise HubError(
                f"router expected last dim {self.d_model}, got {tuple(x.shape)}"
            )
        h = self._pool(x)
        h = self.norm(h)
        h = F.gelu(self.fc1(h))
        return self.fc2(h)  # [..., n_choices]


# ---------------------------------------------------------------------------
# Multi-adapter hub
# ---------------------------------------------------------------------------


class MultiAdapterHub(nn.Module):
    """Conditional multi-bridge hub wrapping existing residual modules.

    Parameters
    ----------
    bridges:
        Ordered mapping name → residual module (ReversibleBridge, LowRankAdapter,
        StudentIntervention, or any module with residual/forward).
    config:
        HubConfig; ``hub_active`` defaults false.
    site_id:
        Optional transplant-site label (e.g. post_attention_hidden_state@L16)
        used for collision bookkeeping.
    """

    def __init__(
        self,
        bridges: Mapping[str, nn.Module] | None = None,
        *,
        config: HubConfig | None = None,
        site_id: str | None = None,
        d_model: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or default_hub_config(
            d_model=int(d_model) if d_model is not None else 64
        )
        if d_model is not None:
            self.config.d_model = int(d_model)
        self.site_id = site_id
        self._bridge_order: list[str] = []
        self.bridges = nn.ModuleDict()
        if bridges:
            for name, mod in bridges.items():
                self.register_bridge(name, mod)

        n_choices = self._n_choices()
        self.router = TinyExternalRouter(
            d_model=self.config.d_model,
            n_choices=n_choices,
            gate_hidden=self.config.gate_hidden,
            pool=self.config.gate_pool,
        )
        self.meta: dict[str, Any] = {
            "schema": HUB_MODULE_SCHEMA,
            "hub_active_default": bool(HUB_ACTIVE),
            "native_moe_router_touched": False,
            "ablatable": True,
            "reversible": True,
            "gravity_accounted": True,
            "live_v0_switched": False,
            "production_validated": False,
            "trained_router": False,
        }

    # -- registration --------------------------------------------------------

    def register_bridge(self, name: str, module: nn.Module) -> None:
        if name in self.bridges:
            raise HubError(f"bridge already registered: {name!r}")
        if name == BYPASS_CHOICE:
            raise HubError(f"{BYPASS_CHOICE!r} is reserved for the bypass channel")
        self.bridges[name] = module
        self._bridge_order.append(name)
        # Rebuild router head if choice count changed after first construction.
        if hasattr(self, "router") and self.router is not None:
            self._resize_router()

    def _n_choices(self) -> int:
        n = len(self._bridge_order)
        return n + 1 if self.config.include_bypass else max(n, 1)

    def choice_names(self) -> list[str]:
        names: list[str] = []
        if self.config.include_bypass:
            names.append(BYPASS_CHOICE)
        names.extend(self._bridge_order)
        return names

    def _resize_router(self) -> None:
        n = self._n_choices()
        if self.router.n_choices == n and self.router.d_model == self.config.d_model:
            return
        self.router = TinyExternalRouter(
            d_model=self.config.d_model,
            n_choices=n,
            gate_hidden=self.config.gate_hidden,
            pool=self.config.gate_pool,
        )

    # -- activity ------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.config.resolved_active()

    def set_active(self, active: bool) -> None:
        """Runtime toggle (still not a production validation claim)."""

        self.config.hub_active = bool(active)

    # -- gate / residual mix -------------------------------------------------

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        self._resize_router()
        return self.router(x)

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate_logits(x) / float(self.config.temperature)
        return F.softmax(logits, dim=-1)

    def _broadcast_weights(
        self,
        weights: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast gate weights [..., C] onto residual [..., H] → [..., C, 1]."""

        # weights may be mean-pooled to drop sequence dim.
        while weights.dim() < residual.dim():
            # Insert missing leading dims just before the choice dim? No —
            # residual is [B,S,H] or [B,H]; weights is [B,S,C] or [B,C] or [B,C]
            # after mean pool: [B,C] vs residual [B,S,H] → unsqueeze seq.
            if weights.dim() + 1 == residual.dim():
                # Insert sequence dim at -2 for residual's sequence.
                weights = weights.unsqueeze(-2).expand(
                    *residual.shape[:-1], weights.shape[-1]
                )
            else:
                weights = weights.unsqueeze(0)
        return weights.unsqueeze(-1)  # [..., C, 1]

    def mixed_residual(
        self,
        x: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Σ_i gate_i · residual_i  (bypass contributes 0)."""

        if not self._bridge_order:
            zero = torch.zeros_like(x)
            return zero, {
                "choice_names": self.choice_names(),
                "weights": None,
                "mode": self.config.selection_mode,
                "selected": BYPASS_CHOICE if self.config.include_bypass else None,
            }

        if weights is None:
            if self.config.selection_mode == "top1":
                logits = self.gate_logits(x) / float(self.config.temperature)
                # Softmax weights for reporting; hard selection for residual.
                weights = F.softmax(logits, dim=-1)
                idx = torch.argmax(logits, dim=-1)  # [...]
                residual = self._top1_residual(x, idx)
                return residual, {
                    "choice_names": self.choice_names(),
                    "weights": weights,
                    "mode": "top1",
                    "selected_index": idx,
                    "selected": self._decode_selection(idx),
                }
            weights = self.gate_weights(x)

        # Soft mix.
        residuals: list[torch.Tensor] = []
        # Align weight channels with [bypass?, bridges...].
        offset = 1 if self.config.include_bypass else 0
        for i, name in enumerate(self._bridge_order):
            residuals.append(module_residual(self.bridges[name], x))
        # Stack → [..., C_bridges, H]
        stacked = torch.stack(residuals, dim=-2)
        w = self._broadcast_weights(weights, x)
        # w: [..., C_total, 1]; take bridge slice
        w_bridges = w[..., offset : offset + len(self._bridge_order), :]
        # If mean-pool made weight seq broadcast already, shapes should match.
        mixed = (w_bridges * stacked).sum(dim=-2)
        return mixed, {
            "choice_names": self.choice_names(),
            "weights": weights,
            "mode": "soft",
            "selected": "soft_mix",
        }

    def _top1_residual(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Per-position top-1 residual; bypass index → zeros."""

        # Build full residual stack with optional leading zero channel for bypass.
        parts: list[torch.Tensor] = []
        if self.config.include_bypass:
            parts.append(torch.zeros_like(x))
        for name in self._bridge_order:
            parts.append(module_residual(self.bridges[name], x))
        stacked = torch.stack(parts, dim=-2)  # [..., C, H]
        # idx: [...] → gather
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, x.shape[-1])
        chosen = torch.gather(stacked, dim=-2, index=gather_idx).squeeze(-2)
        return chosen

    def _decode_selection(self, idx: torch.Tensor) -> Any:
        names = self.choice_names()
        flat = idx.detach().reshape(-1)
        if flat.numel() == 0:
            return None
        # Report dominant selection for diagnostics.
        vals, counts = torch.unique(flat, return_counts=True)
        dom = int(vals[int(torch.argmax(counts))].item())
        return names[dom] if 0 <= dom < len(names) else None

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x.shape[-1] != self.config.d_model:
            raise HubError(
                f"hub expected last dim {self.config.d_model}, got {tuple(x.shape)}"
            )

        if not self.is_active:
            # Inactive: exact always-on path (regression identity).
            y = always_on_apply(
                x,
                self.bridges,
                mode=self.config.always_on_mode,
                order=self._bridge_order,
            )
            if return_aux:
                return y, {
                    "hub_active": False,
                    "path": "always_on",
                    "always_on_mode": self.config.always_on_mode,
                    "applied": list(self._bridge_order),
                    "site_id": self.site_id,
                }
            return y

        residual, gate_info = self.mixed_residual(x)
        y = x + residual
        if return_aux:
            return y, {
                "hub_active": True,
                "path": "gated",
                "site_id": self.site_id,
                **gate_info,
            }
        return y

    def apply_with_residual(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply and return (y, r, aux) so reverse is exact: x = y - r."""

        if not self.is_active:
            y = always_on_apply(
                x,
                self.bridges,
                mode=self.config.always_on_mode,
                order=self._bridge_order,
            )
            r = y - x
            return y, r, {
                "hub_active": False,
                "path": "always_on",
                "exact_reverse": True,
            }
        r, gate_info = self.mixed_residual(x)
        y = x + r
        return y, r, {"hub_active": True, "path": "gated", **gate_info}

    @torch.no_grad()
    def revert_exact(self, y: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return y - residual

    @torch.no_grad()
    def revert(
        self,
        y: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
        n_iters: int = 12,
        atol: float = 1e-5,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        y_arr = y.detach()
        if residual is not None:
            x = self.revert_exact(y_arr, residual)
            step_err = float(torch.max(torch.abs((x + residual) - y_arr)).item())
            return x, {
                "mode": "stored_residual",
                "exact": step_err < atol,
                "stored_step_error": step_err,
            }
        # Fixed-point fallback for inactive sequential path without stored r.
        x = y_arr.clone()
        last_err = None
        for _ in range(int(n_iters)):
            y_hat = self.forward(x)
            # For residual form y = x + r(x), x' = y - (y_hat - x) = 2x - y_hat? No.
            # y_hat = forward(x) ≈ x + r(x). Want y = x + r(x) ⇒ x_next = y - r(x)
            # r(x) = forward(x) - x.
            r = y_hat - x
            x_next = y_arr - r
            last_err = float(torch.max(torch.abs(x_next - x)).item())
            x = x_next
            if last_err < atol:
                break
        recon = float(torch.max(torch.abs(self.forward(x) - y_arr)).item())
        return x, {
            "mode": "fixed_point",
            "iterations": int(n_iters),
            "last_step_delta": last_err,
            "recon_error": recon,
            "exact": recon < atol,
        }

    # -- accounting / seal helpers -------------------------------------------

    def content_hash(self) -> str:
        return tensor_content_hash({k: v for k, v in self.state_dict().items()})

    def gravity_accounting(self, *, bytes_per_param: int = 4) -> dict[str, Any]:
        bridge_parts: dict[str, Any] = {}
        for name in self._bridge_order:
            mod = self.bridges[name]
            if hasattr(mod, "gravity_accounting"):
                bridge_parts[name] = mod.gravity_accounting(  # type: ignore[operator]
                    bytes_per_param=bytes_per_param
                )
            else:
                bridge_parts[name] = {
                    "name": name,
                    "parameter_count": param_count(mod),
                    "parameter_bytes": param_bytes(mod, bytes_per_param=bytes_per_param),
                }
        router_bytes = param_bytes(self.router, bytes_per_param=bytes_per_param)
        bridge_bytes = sum(int(p.get("parameter_bytes", 0)) for p in bridge_parts.values())
        return {
            "name": "MultiAdapterHub",
            "site_id": self.site_id,
            "hub_active": self.is_active,
            "bridges": bridge_parts,
            "router": {
                "parameter_count": param_count(self.router),
                "parameter_bytes": router_bytes,
                "gate_hidden": self.config.gate_hidden,
                "n_choices": self._n_choices(),
                "native_moe_router_touched": False,
            },
            "total_parameter_bytes": bridge_bytes + router_bytes,
            "total_parameter_count": param_count(self),
            "hash_bound": self.content_hash(),
            "ablatable": True,
            "reversible": True,
            "bytes_per_param": int(bytes_per_param),
        }

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": HUB_MODULE_SCHEMA,
            "site_id": self.site_id,
            "hub_active": self.is_active,
            "config": self.config.to_dict(),
            "bridge_order": list(self._bridge_order),
            "choice_names": self.choice_names(),
            "parameter_count": param_count(self),
            "content_hash": self.content_hash(),
            "gravity": self.gravity_accounting(),
            "meta": dict(self.meta),
            "native_moe_router_touched": False,
            "claim_boundary": dict(self.config.claim_boundary),
        }


# ---------------------------------------------------------------------------
# Fixture builders (collision / routing tests)
# ---------------------------------------------------------------------------


def build_synthetic_site_hub(
    *,
    d_model: int = 32,
    d_hidden: int = 16,
    rank: int = 4,
    bridge_names: Sequence[str] | None = None,
    hub_active: bool = False,
    selection_mode: str = "soft",
    site_id: str = "fixture_post_attention_hidden_state@L16",
    seed: int = 0,
) -> MultiAdapterHub:
    """Build a hub with synthetic ReversibleBridge modules (no real weights)."""

    torch.manual_seed(int(seed))
    names = list(bridge_names) if bridge_names is not None else [
        "GLM_DECOMPOSITION_BRIDGE",
        "KIMI_PLANNING_BRIDGE",
    ]
    bridges: dict[str, nn.Module] = {}
    for i, name in enumerate(names):
        bridges[name] = ReversibleBridge(
            name=name,
            d_model=d_model,
            d_hidden=d_hidden,
            rank=rank,
            scale=0.05 + 0.01 * i,
        )
    cfg = HubConfig(
        hub_active=hub_active,
        selection_mode=selection_mode,
        d_model=d_model,
        gate_hidden=max(8, d_model // 4),
        include_bypass=True,
        always_on_mode="sequential",
        gate_pool="token",
    )
    return MultiAdapterHub(bridges, config=cfg, site_id=site_id, d_model=d_model)


def force_router_one_hot(
    hub: MultiAdapterHub,
    choice_name: str,
    *,
    logit_scale: float = 20.0,
) -> None:
    """Hard-wire the tiny router to prefer a single named choice (test helper).

    Sets fc2 bias so ``choice_name`` dominates softmax / top-1.  Zeroes fc2
    weight so the decision is input-independent (deterministic fixture).
    """

    names = hub.choice_names()
    if choice_name not in names:
        raise HubError(f"{choice_name!r} not in {names}")
    idx = names.index(choice_name)
    with torch.no_grad():
        hub.router.fc2.weight.zero_()
        hub.router.fc2.bias.zero_()
        hub.router.fc2.bias[idx] = float(logit_scale)


def force_router_input_linear(
    hub: MultiAdapterHub,
    *,
    feature_index: int = 0,
    positive_choice: str,
    negative_choice: str,
    scale: float = 10.0,
) -> None:
    """Route by sign of x[..., feature_index] between two choices (fixture)."""

    names = hub.choice_names()
    for c in (positive_choice, negative_choice):
        if c not in names:
            raise HubError(f"{c!r} not in {names}")
    pos = names.index(positive_choice)
    neg = names.index(negative_choice)
    d = hub.config.d_model
    h = hub.router.gate_hidden
    n = hub.router.n_choices
    with torch.no_grad():
        # fc1: project feature_index → hidden unit 0
        hub.router.fc1.weight.zero_()
        hub.router.fc1.bias.zero_()
        hub.router.fc1.weight[0, feature_index] = float(scale)
        # fc2: hidden unit 0 → +pos / -neg (gelu-preserving for positive path;
        # use bias split so both directions work via pre-bias feature).
        hub.router.fc2.weight.zero_()
        hub.router.fc2.bias.zero_()
        # After gelu(fc1): roughly gelu(scale * x_f). For x_f > 0 → positive;
        # for x_f < 0 gelu≈0 so we need a second unit for negative.
        hub.router.fc1.weight[1, feature_index] = float(-scale)
        hub.router.fc2.weight[pos, 0] = float(scale)
        hub.router.fc2.weight[neg, 1] = float(scale)
        # Suppress other choices.
        for i in range(n):
            if i not in (pos, neg):
                hub.router.fc2.bias[i] = -float(scale) * 2.0


# ---------------------------------------------------------------------------
# Activation trigger (documented, not fired)
# ---------------------------------------------------------------------------


def evaluate_activation_trigger(
    *,
    secondary_scores: Mapping[str, Any] | None = None,
    site_collision: Mapping[str, Any] | None = None,
    secondary_tolerance: float = SECONDARY_TOLERANCE,
    always_on_ablation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether HUB_ACTIVE may be flipped — fail-closed by default.

    Triggers (either is sufficient, both require real evidence):

    1. **Secondary regression beyond tolerance** — any secondary axis from
       the promotion gate suite drops more than ``secondary_tolerance``
       (default 0.02) relative to the frozen baseline under always-on bridges.
    2. **Confirmed site collision** — measured evidence that GLM and Kimi
       bridges at the same transplant site interfere when both fire additively
       (not merely co-listed in contracts).

    Until A–G ablation / auto-bench is run with real correspondence data,
    this returns ``activate=False`` honestly.
    """

    reasons: list[str] = []
    signals: dict[str, Any] = {
        "secondary_regression": False,
        "confirmed_site_collision": False,
        "ablation_evidence": False,
    }

    if secondary_scores:
        regressions: list[dict[str, Any]] = []
        for axis, row in secondary_scores.items():
            if not isinstance(row, Mapping):
                continue
            # Expect {baseline, always_on, delta?} or {delta, gate}
            if "delta" in row:
                delta = float(row["delta"])
            elif "baseline" in row and "always_on" in row:
                delta = float(row["always_on"]) - float(row["baseline"])
            else:
                continue
            # Regression = negative delta beyond tolerance.
            if delta < -float(secondary_tolerance):
                regressions.append(
                    {
                        "axis": axis,
                        "delta": delta,
                        "tolerance": float(secondary_tolerance),
                    }
                )
        if regressions:
            signals["secondary_regression"] = True
            reasons.append(
                f"secondary regression beyond ±{secondary_tolerance}: "
                + ", ".join(r["axis"] for r in regressions)
            )
            signals["regressions"] = regressions

    if site_collision and site_collision.get("confirmed") is True:
        # Require explicit measured evidence flag — not just contract notes.
        evidence = site_collision.get("evidence") or site_collision.get("measurement")
        if evidence:
            signals["confirmed_site_collision"] = True
            reasons.append(
                "confirmed transplant-site collision: "
                + str(site_collision.get("site_id") or site_collision.get("transplant_point"))
            )
        else:
            reasons.append(
                "site_collision.confirmed set but no evidence/measurement payload — ignored"
            )

    if always_on_ablation and always_on_ablation.get("shows_interference") is True:
        signals["ablation_evidence"] = True
        reasons.append("always-on ablation shows interference")

    activate = bool(
        signals["secondary_regression"]
        or signals["confirmed_site_collision"]
        or signals["ablation_evidence"]
    )

    return {
        "schema": "hawking.frankenstein.multi_adapter_hub_activation_trigger.v1",
        "recorded_at": _utc_now(),
        "activate": activate,
        "hub_active_recommended": activate,
        "currently_default_HUB_ACTIVE": bool(HUB_ACTIVE),
        "secondary_tolerance": float(secondary_tolerance),
        "signals": signals,
        "reasons": reasons,
        "gate": HUB_ACTIVATION_GATE if not activate else "HUB_ACTIVATION_SIGNAL_PRESENT",
        "status": (
            "ACTIVATE_PERMITTED"
            if activate
            else "FAIL_CLOSED_NO_ACTIVATION_SIGNAL"
        ),
        "note": (
            "Scaffold only.  Activation is not a capability claim and does not "
            "switch live V0 bridges until an operator flips hub_active after "
            "this trigger returns activate=true."
        ),
        "claim_boundary": {
            "production_validated": False,
            "live_v0_switched": False,
            "capability_claim": False,
        },
    }


def compose_with_promotion_gate() -> dict[str, Any]:
    """How the hub composes with frankenstein_promotion_gate (doc only)."""

    return {
        "promotion_gate_schema": "hawking.frankenstein.functional_transfer_promotion_gate.v1",
        "secondary_tolerance_absolute": float(SECONDARY_TOLERANCE),
        "relationship": (
            "HUB_ACTIVE is independent of promotion ACCEPT/REJECT.  The hub may "
            "be flipped as a *mitigation* when always-on bridges cause secondary "
            "regression beyond promotion-gate tolerance, or when site collision "
            "is measured.  Promotion still requires held-out math gains, gap "
            "recovery, provenance, gravity accounting, and stable routing."
        ),
        "order_of_operations": [
            "1. Run always-on V0 bridges (HUB_ACTIVE=false) through A–G ablation / auto-bench",
            "2. If secondary regression > tolerance OR confirmed site collision → permit hub flip",
            "3. Train/fit tiny external router on real paired activations (REQUIRES_TRAINING_LOOP)",
            "4. Re-run ablation with HUB_ACTIVE=true; verify bypass identity + no dual-add collision",
            "5. Promotion gate evaluates the chosen configuration (always-on or hub) independently",
        ],
        "reversible_ablatable_gravity": (
            "Hub is independently ablatable (set hub_active=false or remove hub), "
            "bypass is exact identity, residual reverse via stored r, and "
            "gravity_accounting() reports router + bridge bytes like every other module."
        ),
        "does_not": [
            "touch_native_256_expert_router",
            "auto_flip_on_contract_site_overlap_notes_alone",
            "claim_production_validation",
            "replace_promotion_ACCEPT",
        ],
    }


# ---------------------------------------------------------------------------
# Design seal
# ---------------------------------------------------------------------------


def hub_design_document() -> dict[str, Any]:
    """Sealed design record: when to flip, how it composes, claim boundary."""

    return {
        "schema": HUB_DESIGN_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "READY_INACTIVE_SCAFFOLD",
        "rank": 1,
        "source_research": (
            "workspace/campaign/evidence/models/frankenstein/"
            "FRANKENSTEIN_ARCHITECTURE_OPTIONS.md"
        ),
        "objective": (
            "Tiny external router selecting among residual bridges instead of "
            "always-on addition — fixes irrelevant-input interference and "
            "GLM×Kimi co-location without native MoE surgery."
        ),
        "formula": "Hub(x) = x + sum_i gate_i(x) * residual_i(x)",
        "bypass": {
            "choice": BYPASS_CHOICE,
            "index": BYPASS_INDEX,
            "property": "exact identity: Hub(x) = x when bypass is selected (top-1) "
            "or when all bridge gates are zero (soft)",
        },
        "module_default": {
            "HUB_ACTIVE": bool(HUB_ACTIVE),
            "config_field": "hub_active",
            "env_override": "FRANKENSTEIN_HUB_ACTIVE",
            "one_line_flip": "Set hub_active=true in MULTI_ADAPTER_HUB_CONFIG.json "
            "or HUB_ACTIVE=True in frankenstein_adapter_hub.py",
        },
        "named_bridge_catalog": {
            "glm_v0_sites": list(GLM_V0_BRIDGE_SITES),
            "kimi_children": list(KIMI_NAMED_BRIDGES),
            "kimi_parent": KIMI_PARENT_BRIDGE,
            "full": list(NAMED_BRIDGE_CATALOG),
            "note": (
                "Hub wraps existing modules from frankenstein_bridges / "
                "frankenstein_adapter_modules / frankenstein_latent_v0 — "
                "does not reimplement residual architectures."
            ),
        },
        "external_router": {
            "kind": "TinyExternalRouter",
            "not": "DeepSeek native 256-expert top-6 router",
            "input": "local hidden state at transplant site",
            "architecture": "LayerNorm → Linear(d, gate_hidden) → GELU → Linear(gate_hidden, n_choices)",
            "selection_modes": list(SELECTION_MODES),
            "choices": f"{{{BYPASS_CHOICE}, A_glm_i, A_kimi_i, ...}}",
        },
        "inactive_path": {
            "when": "hub_active=false (default)",
            "behavior": "byte-identical to always_on_apply (sequential residual by default)",
            "live_v0": "unchanged — current always-on bridges keep running",
        },
        "collision_resolution": {
            "problem": (
                "GLM and Kimi both want a bridge at the same transplant site; "
                "legacy workaround was disjoint block_ids / do-not-overwrite."
            ),
            "hub_fix": (
                "At a shared site, register both bridges on one MultiAdapterHub; "
                "gate selects EITHER (top-1) or a learned soft mix — not forced "
                "dual additive fire."
            ),
            "default_collision_sites": list(DEFAULT_COLLISION_SITES),
        },
        "activation_trigger": {
            "default": "FAIL_CLOSED_NO_ACTIVATION_SIGNAL",
            "signals": [
                {
                    "name": "secondary_regression_beyond_tolerance",
                    "tolerance_source": "frankenstein_promotion_gate.SECONDARY_TOLERANCE",
                    "tolerance_value": float(SECONDARY_TOLERANCE),
                    "detail": (
                        "Any secondary axis drops more than tolerance under always-on "
                        "bridges relative to baseline (A–G ablation / auto-bench)."
                    ),
                },
                {
                    "name": "confirmed_site_collision",
                    "detail": (
                        "Measured interference when two donor bridges at one "
                        "transplant site fire additively.  Contract site_overlap "
                        "notes alone are NOT sufficient."
                    ),
                },
            ],
            "not_yet_run": (
                "A–G ablation / auto-bench still waiting on real correspondence data. "
                "This scaffold is ready, not production-validated."
            ),
            "evaluator": "lab.operators.frankenstein_adapter_hub.evaluate_activation_trigger",
        },
        "composition_with_promotion_gate": compose_with_promotion_gate(),
        "properties": {
            "reversible": True,
            "ablatable": True,
            "gravity_accounted": True,
            "hash_bound": True,
            "bypass_exact_identity": True,
            "native_router_untouched": True,
        },
        "implementation": {
            "module": "lab/operators/frankenstein_adapter_hub.py",
            "tests": "tools/condense/tests/test_frankenstein_adapter_hub.py",
            "config": f"evidence/models/frankenstein/{DEFAULT_HUB_CONFIG_NAME}",
        },
        "non_goals": [
            "Do not activate for live V0 run",
            "No GPU / no live recapture-fixed-corpus edits",
            "No Kimi/Odyssey execution in this scaffold pass",
            "No fabricated 'this fixed X' production claim",
        ],
        "claim_boundary": {
            "ready_inactive_infrastructure": True,
            "live_v0_switched": False,
            "production_validated": False,
            "capability_claim": False,
            "fabricated_fix_claim": False,
        },
    }


def seal_hub_design(path: str | Path | None = None) -> dict[str, Any]:
    """Seal and optionally write MULTI_ADAPTER_HUB_DESIGN.json."""

    doc = hub_design_document()
    sealed = seal(doc)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sealed


# ---------------------------------------------------------------------------
# CLI (scaffold hygiene)
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-adapter hub scaffold utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_seal = sub.add_parser("seal-design", help="Write MULTI_ADAPTER_HUB_DESIGN.json")
    p_seal.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR / DEFAULT_HUB_DESIGN_NAME,
    )

    p_cfg = sub.add_parser("write-config", help="Write default inactive hub config")
    p_cfg.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR / DEFAULT_HUB_CONFIG_NAME,
    )

    p_trig = sub.add_parser(
        "eval-trigger",
        help="Evaluate activation trigger (fail-closed without signals)",
    )
    p_trig.add_argument("--secondary-json", type=Path, default=None)
    p_trig.add_argument("--collision-json", type=Path, default=None)

    p_smoke = sub.add_parser("smoke", help="CPU smoke: inactive identity + active bypass")
    p_smoke.add_argument("--d-model", type=int, default=32)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "seal-design":
        sealed = seal_hub_design(args.out)
        print(json.dumps({"wrote": str(args.out), "seal_sha256": sealed["seal_sha256"]}))
        return 0

    if args.cmd == "write-config":
        path = write_default_hub_config(args.out)
        print(json.dumps({"wrote": str(path), "hub_active": False}))
        return 0

    if args.cmd == "eval-trigger":
        secondary = None
        collision = None
        if args.secondary_json and args.secondary_json.is_file():
            secondary = json.loads(args.secondary_json.read_text(encoding="utf-8"))
        if args.collision_json and args.collision_json.is_file():
            collision = json.loads(args.collision_json.read_text(encoding="utf-8"))
        result = evaluate_activation_trigger(
            secondary_scores=secondary,
            site_collision=collision,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["activate"] else 0

    if args.cmd == "smoke":
        hub = build_synthetic_site_hub(d_model=args.d_model, hub_active=False)
        x = torch.randn(2, 4, args.d_model)
        y_ref = always_on_apply(x, hub.bridges, mode="sequential", order=hub._bridge_order)
        y_hub = hub(x)
        assert torch.equal(y_hub, y_ref), "inactive path not identical"
        hub.set_active(True)
        force_router_one_hot(hub, BYPASS_CHOICE)
        y_b, aux = hub(x, return_aux=True)
        assert torch.allclose(y_b, x, atol=1e-6), "bypass not identity"
        print(
            json.dumps(
                {
                    "inactive_identical": True,
                    "bypass_identity": True,
                    "hub_active_default": bool(HUB_ACTIVE),
                    "aux_path": aux.get("path"),
                }
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
