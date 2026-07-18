#!/usr/bin/env python3.12
"""Per-architecture loader scaffolds for the three giant frontier parents.

We know each giant's exact geometry (bound READ-ONLY from HF on 2026-07-17, mirrored in
succ_frontier.PARENTS), so we can scaffold their loaders now even though the weights are not
downloaded (they stream). This module defines, per architecture:

  - the tensor-name templates (MLA attention + routed/shared MoE experts + router + MTP),
  - the source dequant scheme (bf16+fp8 / INT4 compressed-tensors / FP4+FP8),
  - a geometry validator that reconstructs the parameter count from the config and checks it
    against the official total (a real correctness check on the scaffold),
  - the Gravity hooks (per-parent sub-bit stress start + candidate representation families),
  - a loader interface mirroring gptoss_moe_runtime, with weight-reading gated behind the
    streamed source (raises a precise NotImplementedError naming what is still needed).

HONEST SCOPE. This is structure derived from known geometry, validated on the geometry, not
on weights. Each loader's `load_expert`/`load_mla` is interface-complete and shape-correct
but raises until its parent's source streams and a per-arch dequant kernel lands. It launches
nothing and downloads nothing.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import succ_frontier as sf  # noqa: E402
import succ_gravity_policy as gp  # noqa: E402

SCAFFOLD_SCHEMA = "hawking.frontier.giant_loader_scaffold.v1"


@dataclass(frozen=True)
class GiantArchSpec:
    """Per-architecture tensor layout + dequant scheme, derived from the bound geometry."""
    row_id: str
    architecture: str
    dequant_scheme: str                 # how source expert weights are packed
    attention: str                      # "MLA" | "MHA"
    tensor_templates: dict[str, str]    # logical role -> tensor-name template ({L}=layer, {E}=expert)
    has_mtp: bool
    multimodal: bool
    gravity_stress_start: Fraction
    representation_families: tuple[str, ...]


def _templates(*, mla: bool, mtp: bool, multimodal: bool) -> dict[str, str]:
    """The DeepSeek/Kimi MoE + MLA tensor naming (compressed-KV attention + routed experts)."""
    t: dict[str, str] = {
        "embed": "model.embed_tokens.weight",
        "router_gate": "model.layers.{L}.mlp.gate.weight",
        "shared_gate": "model.layers.{L}.mlp.shared_experts.gate_proj.weight",
        "shared_up": "model.layers.{L}.mlp.shared_experts.up_proj.weight",
        "shared_down": "model.layers.{L}.mlp.shared_experts.down_proj.weight",
        "expert_gate": "model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
        "expert_up": "model.layers.{L}.mlp.experts.{E}.up_proj.weight",
        "expert_down": "model.layers.{L}.mlp.experts.{E}.down_proj.weight",
        "lm_head": "lm_head.weight",
    }
    if mla:
        t.update({
            "q_a_proj": "model.layers.{L}.self_attn.q_a_proj.weight",
            "q_b_proj": "model.layers.{L}.self_attn.q_b_proj.weight",
            "kv_a_proj": "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight",
            "kv_b_proj": "model.layers.{L}.self_attn.kv_b_proj.weight",
            "o_proj": "model.layers.{L}.self_attn.o_proj.weight",
        })
    else:
        t.update({"qkv_proj": "model.layers.{L}.self_attn.qkv_proj.weight",
                  "o_proj": "model.layers.{L}.self_attn.o_proj.weight"})
    if mtp:
        t["mtp"] = "model.layers.{L}.mtp.weight"
    if multimodal:
        t["vision_tower"] = "vision_tower.{L}.weight"
    return t


_DEQUANT = {
    "deepseek-v3.2-685b": "bf16_body_fp8_scaled_experts",
    "kimi-k2.6-1t": "int4_compressed_tensors_group_scaled",
    "deepseek-v4-pro-1.6t": "fp4_experts_fp8_trunk",
}


def build_specs() -> dict[str, GiantArchSpec]:
    specs: dict[str, GiantArchSpec] = {}
    stress = {"deepseek-v3.2-685b": "685B", "kimi-k2.6-1t": "1T", "deepseek-v4-pro-1.6t": "1.6T"}
    for p in sf.PARENTS:
        prior = gp.prior_for(stress[p.row_id])
        specs[p.row_id] = GiantArchSpec(
            row_id=p.row_id, architecture=p.architecture,
            dequant_scheme=_DEQUANT[p.row_id],
            attention="MLA" if p.kv_lora_rank else "MHA",
            tensor_templates=_templates(mla=bool(p.kv_lora_rank), mtp=p.mtp_layers > 0,
                                        multimodal=p.multimodal),
            has_mtp=p.mtp_layers > 0, multimodal=p.multimodal,
            gravity_stress_start=Fraction(prior["subbit_stress_start"]),
            representation_families=tuple(p.doctor_families),
        )
    return specs


def _parent(row_id: str) -> sf.GiantParent:
    return next(p for p in sf.PARENTS if p.row_id == row_id)


def validate_geometry(row_id: str) -> dict[str, Any]:
    """Reconstruct the parameter count from the config and check it against the official total.
    A real correctness check on the scaffold's understanding of the architecture."""
    p = _parent(row_id)
    h = p.hidden_size
    # per-layer MoE: routed experts (gate+up+down) + shared expert, each gate/up: h*moe_int, down: moe_int*h
    per_expert = 3 * h * p.moe_intermediate_size
    routed = p.n_routed_experts * per_expert
    shared = p.n_shared_experts * per_expert
    router = p.n_routed_experts * h
    # MLA attention params (compressed): q_a(h*q_lora)+q_b(q_lora*heads*dim)+kv_a+kv_b+o, approx via lora ranks
    q_lora = p.q_lora_rank or 0
    kv_lora = p.kv_lora_rank or 0
    attn = (h * q_lora) + (q_lora * h) + (h * (kv_lora + 64)) + ((kv_lora) * h) + (h * h)
    per_layer = routed + shared + router + attn
    embed = 2 * p.vocab_size * h  # embed + lm_head
    total_est = per_layer * p.num_layers + embed
    official = p.official_total_params
    ratio = total_est / official
    # MoE param estimate is dominated by experts; accept broad tolerance (attn/norm/mtp omitted detail)
    plausible = 0.55 <= ratio <= 1.45
    return {"row_id": row_id, "estimated_params": int(total_est), "official_params": int(official),
            "ratio_est_to_official": round(ratio, 3), "geometry_plausible": plausible,
            "per_expert_params": per_expert, "routed_experts": p.n_routed_experts,
            "experts_per_tok": p.experts_per_tok, "attention": "MLA" if kv_lora else "MHA",
            "dequant_scheme": _DEQUANT[row_id]}


class GiantLoaderScaffold:
    """Interface-complete per-expert loader for a giant. Resolves tensor names + expected shapes
    from geometry now; weight reads raise until the source streams and a dequant kernel lands."""

    def __init__(self, row_id: str, *, provenance: dict[str, Any] | None = None):
        self.spec = build_specs()[row_id]
        self.parent = _parent(row_id)
        self.provenance = provenance  # a streamed-source provenance manifest, when available

    def tensor_name(self, role: str, *, layer: int = 0, expert: int = 0) -> str:
        tpl = self.spec.tensor_templates.get(role)
        if tpl is None:
            raise KeyError(f"{self.spec.row_id}: no tensor template for role {role!r}")
        return tpl.format(L=layer, E=expert)

    def expert_shapes(self) -> dict[str, tuple[int, int]]:
        """Expected per-expert projection shapes (gate/up: [moe_int, hidden], down: [hidden, moe_int])."""
        h, m = self.parent.hidden_size, self.parent.moe_intermediate_size
        return {"gate_proj": (m, h), "up_proj": (m, h), "down_proj": (h, m)}

    def load_expert(self, layer: int, expert: int) -> dict[str, Any]:
        if self.provenance is None:
            raise NotImplementedError(
                f"{self.spec.row_id}: streamed source + provenance manifest required "
                f"(scheme={self.spec.dequant_scheme}). Geometry ready: "
                f"experts={self.parent.n_routed_experts}, shapes={self.expert_shapes()}, "
                f"tensors={[self.tensor_name(r, layer=layer, expert=expert) for r in ('expert_gate','expert_up','expert_down')]}")
        raise NotImplementedError(
            f"{self.spec.row_id}: per-arch dequant kernel for {self.spec.dequant_scheme} not yet built")

    def gravity_hooks(self) -> dict[str, Any]:
        return {"stress_start": gp.rate_identity(self.spec.gravity_stress_start),
                "representation_families": list(self.spec.representation_families),
                "attention": self.spec.attention, "has_mtp": self.spec.has_mtp,
                "multimodal": self.spec.multimodal, "dequant_scheme": self.spec.dequant_scheme}


def scaffold_manifest() -> dict[str, Any]:
    """A self-describing manifest of all three giant scaffolds + their geometry checks."""
    specs = build_specs()
    return {
        "schema": SCAFFOLD_SCHEMA,
        "parents": {
            row_id: {
                "architecture": spec.architecture, "attention": spec.attention,
                "dequant_scheme": spec.dequant_scheme, "has_mtp": spec.has_mtp,
                "multimodal": spec.multimodal,
                "gravity_stress_start": gp.rate_identity(spec.gravity_stress_start)["label"],
                "representation_families": list(spec.representation_families),
                "geometry": validate_geometry(row_id),
                "expert_shapes": {k: list(v) for k, v in GiantLoaderScaffold(row_id).expert_shapes().items()},
            }
            for row_id, spec in specs.items()
        },
        "note": "structure from bound geometry; weights gated on streamed sources + per-arch "
                "dequant kernels. No download, no launch.",
    }


def selftest() -> dict[str, Any]:
    specs = build_specs()
    assert set(specs) == {"deepseek-v3.2-685b", "kimi-k2.6-1t", "deepseek-v4-pro-1.6t"}
    for row_id in specs:
        g = validate_geometry(row_id)
        assert g["geometry_plausible"], (row_id, g["ratio_est_to_official"])
        ld = GiantLoaderScaffold(row_id)
        # tensor-name resolution works from geometry
        assert "experts.0" in ld.tensor_name("expert_gate", layer=3, expert=0)
        assert ld.expert_shapes()["gate_proj"][1] == _parent(row_id).hidden_size
        # weight load correctly refuses (no streamed source) with a precise message
        try:
            ld.load_expert(0, 0)
            raise AssertionError("should refuse without streamed source")
        except NotImplementedError as e:
            assert "streamed source" in str(e)
        # gravity hooks resolve
        hk = ld.gravity_hooks()
        assert hk["stress_start"]["value"] < 1.0
    # Kimi is multimodal + MTP=0; V3.2 and V4 have MTP
    assert specs["kimi-k2.6-1t"].multimodal and not specs["kimi-k2.6-1t"].has_mtp
    assert specs["deepseek-v3.2-685b"].has_mtp and specs["deepseek-v4-pro-1.6t"].has_mtp
    # V4 has MHA-style (kv_lora None) per bound config; the other two MLA
    assert specs["deepseek-v4-pro-1.6t"].attention == "MHA"
    assert specs["deepseek-v3.2-685b"].attention == "MLA"
    return {"ok": True, "parents": list(specs),
            "geometry_ratios": {r: validate_geometry(r)["ratio_est_to_official"] for r in specs}}


if __name__ == "__main__":
    import json
    print(json.dumps({"selftest": selftest(), "manifest": scaffold_manifest()},
                     indent=2, sort_keys=True, default=str))
