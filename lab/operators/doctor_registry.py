"""Doctor recovery registry + chain selector (absorbed from tools/condense/doctor.py).

The Doctor is not one script: it is a pluggable L0–L6 method catalog and an
auto-composer that returns an ordered recovery chain for (model, target-bpw,
device). Pure stdlib. No baker, no HF, no campaign CLI scaffolding.

Absorbed: RecoveryMethod, REGISTRY, select, list_methods, emit_chain.
Dropped: blockwise/strand/qat/lora training CLIs (retired studio campaigns,
transformers + safetensors write paths); residual as a default chain member
(rate-additive as implemented — see EXCLUDED note); audit_ladder bake driver.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Effective-bpw ladder used by the original registry (RHT+outlier folded).
# These are planning points for chain composition, not a license to exceed
# the live one-bit complete-BPW ceiling at seal time.
LADDER: tuple[tuple[int, float], ...] = (
    (1, 1.34),
    (2, 2.34),
    (3, 3.34),
    (4, 4.5),
)

# residual.py is rate-ADDITIVE (base + residual pass). Including it by default
# would push complete BPW above the campaign law. Live consumer is orphaned.
EXCLUDED_FROM_DEFAULT_CHAIN: frozenset[str] = frozenset({"residual"})


@dataclass(frozen=True)
class RecoveryMethod:
    name: str
    layer: int  # L0..L6 leverage rank (cheapest/most-general first)
    stage: str  # 'local' | 'studio'
    train_free: bool
    sensitivity: str  # 'global' | 'per_tensor' | 'per_expert'
    tool: str
    provides_serve: bool
    min_params_b: float = 0.0
    max_params_b: Optional[float] = None
    status: str = "MEASURED"  # MEASURED | GATED | UNPROVEN | DEAD | EXCLUDED
    note: str = ""
    live_entry: str = ""  # lab.operators module that carries the method, if any
    build_fn: Optional[Callable[[dict[str, Any]], Any]] = field(default=None, repr=False)


REGISTRY: dict[str, RecoveryMethod] = {}


def register(**kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        m = RecoveryMethod(build_fn=fn, **kw)
        REGISTRY[m.name] = m
        return fn

    return deco


def _emit(name: str, builder: str, args: tuple[Any, ...] = ()) -> tuple[str, str, tuple[Any, ...]]:
    return (name, builder, args)


@register(
    name="calib",
    layer=0,
    stage="local",
    train_free=True,
    sensitivity="global",
    tool="domain calib corpus",
    provides_serve=True,
    status="MEASURED",
    note="domain-matched calibration; multiplies every layer below",
    live_entry="",
)
def _calib(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit("calib", "build_calib")


@register(
    name="awq",
    layer=1,
    stage="local",
    train_free=True,
    sensitivity="per_tensor",
    tool="activation-aware pre-scale",
    provides_serve=True,
    status="MEASURED",
    note="alpha=0.5 activation-aware pre-scale; halves the raw gap at 3-4bit",
    live_entry="lab.operators.ascension_qwen30_activation_weighted_svd_repack",
)
def _awq(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(f"{ctx['bits']}-AWQ", "build_awq", (ctx["bits"], 0.5))


@register(
    name="learned_rotation",
    layer=1,
    stage="local",
    train_free=True,
    sensitivity="per_tensor",
    tool="rotation_search (TODO)",
    provides_serve=True,
    status="UNPROVEN",
    note="QuaRot/SpinQuant learned orthogonal rotation before the cut; ~0 serve bpw",
)
def _rot(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit("rot", "TODO")


@register(
    name="mixed_prec",
    layer=2,
    stage="local",
    train_free=True,
    sensitivity="per_tensor",
    tool="lab.operators.mixed_precision_alloc",
    provides_serve=True,
    status="MEASURED",
    note="output-sensitivity water-fill: sensitive tensors get depth, tolerant get starved",
    live_entry="lab.operators.mixed_precision_alloc",
)
def _mp(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(
        "mp-waterfill",
        "mixed_precision_alloc.allocate",
        (ctx["bits"], ctx.get("target_bpw")),
    )


@register(
    name="expert_alloc",
    layer=2,
    stage="local",
    train_free=True,
    sensitivity="per_expert",
    tool="lab.operators.expert_alloc",
    provides_serve=False,
    min_params_b=1.0,
    status="MEASURED",
    note="MoE per-expert bit allocation: router/shared high-bit, hot 2-bit, cold 1-bit/ternary",
    live_entry="lab.operators.expert_alloc",
)
def _expert(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit("expert-alloc", "expert_alloc.propose_allocation", (ctx["bits"],))


@register(
    name="residual",
    layer=3,
    stage="local",
    train_free=True,
    sensitivity="per_tensor",
    tool="residual.py (EXCLUDED)",
    provides_serve=True,
    status="EXCLUDED",
    note=(
        "W ~= code(W)+code(residual) is rate-ADDITIVE as implemented and violates "
        "complete BPW <= 1/1. Not in default chain. Absorb only if made rate-neutral."
    ),
)
def _res(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(f"res{ctx['bits']}+1", "EXCLUDED_rate_additive", (ctx["bits"], 1))


@register(
    name="outlier_channel",
    layer=3,
    stage="local",
    train_free=True,
    sensitivity="per_tensor",
    tool="sparse high-bit outlier channel",
    provides_serve=True,
    status="MEASURED",
    note="keep top-|w| 5-10% at 8-bit sparse channel; train-free sub-3-bit rescue",
)
def _outl(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(f"{ctx['bits']}-AWQ-o5", "build_awq_outlier", (ctx["bits"], 0.5, 5.0))


@register(
    name="block_qat",
    layer=4,
    stage="studio",
    train_free=False,
    sensitivity="per_tensor",
    tool="lab.operators.lowbit_qat (STE primitives)",
    provides_serve=True,
    status="GATED",
    note="BRECQ-lite full-rank per-linear QAT; STE primitives live in lowbit_qat",
    live_entry="lab.operators.lowbit_qat",
)
def _bw(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(f"{ctx['bits']}-bw", "build_blockwise", (ctx["bits"],))


@register(
    name="gptq_hessian",
    layer=5,
    stage="studio",
    train_free=False,
    sensitivity="per_tensor",
    tool="codec-native Hessian error-feedback",
    provides_serve=True,
    status="UNPROVEN",
    note="codec-native sequential error-feedback (uniform STE path is DEAD)",
)
def _str(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(f"{ctx['bits']}-str", "build_strand", (ctx["bits"],))


@register(
    name="deep_kd",
    layer=6,
    stage="studio",
    train_free=False,
    sensitivity="global",
    tool="KD polish",
    provides_serve=True,
    status="GATED",
    note="logit/feature/attn KD polish on the full-rank base",
)
def _kd(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit(f"{ctx['bits']}-kd", "build_recover", (ctx["bits"],))


@register(
    name="big_teacher_kd",
    layer=6,
    stage="studio",
    train_free=False,
    sensitivity="global",
    tool="larger-teacher KD (TODO)",
    provides_serve=True,
    status="UNPROVEN",
    note="a larger model distilling the condensed one; needs both resident",
)
def _bigkd(ctx: dict[str, Any]) -> tuple[str, str, tuple[Any, ...]]:
    return _emit("bigkd", "TODO")


def floor_bits(params_b: float, entropy_floor: float | None = None) -> int:
    if entropy_floor is not None:
        for b, bpw in LADDER:
            if bpw >= entropy_floor:
                return b
    raw = 3.6 - 0.9 * math.log10(max(0.5, params_b))
    for b, bpw in LADDER:
        if bpw >= raw:
            return b
    return 4


def select(
    params_b: float,
    target_bpw: float | None = None,
    *,
    is_moe: bool = False,
    entropy_floor: float | None = None,
    device: str = "studio-m2max",
    include_excluded: bool = False,
) -> tuple[int, list[str], dict[str, Any]]:
    """Compose the recovery chain. Cheapest-leverage-first train-free stack;
    training layers only when aggressive; per-expert for MoE.

    residual is omitted unless include_excluded=True (it is rate-additive).
    """
    del device  # reserved for stage filtering; catalog is device-agnostic today
    bits: int | None = None
    if target_bpw is not None:
        for b, bpw in LADDER:
            if abs(bpw - target_bpw) < 0.5:
                bits = b
                break
    bits = bits or floor_bits(params_b, entropy_floor)
    ctx: dict[str, Any] = {
        "params_b": params_b,
        "bits": bits,
        "is_moe": is_moe,
        "target_bpw": target_bpw,
    }
    chain: list[str] = ["calib", "awq", "mixed_prec"]
    if include_excluded:
        chain.append("residual")
    if bits <= 2:
        chain.append("outlier_channel")
    if is_moe:
        chain.insert(1, "expert_alloc")
    if bits <= 2:
        chain += (
            ["block_qat", "gptq_hessian", "deep_kd"]
            if params_b >= 7
            else ["block_qat", "deep_kd"]
        )
    elif bits == 3:
        chain.append("deep_kd")

    seen: set[str] = set()
    ordered: list[str] = []
    for n in chain:
        if n not in REGISTRY or n in seen:
            continue
        if n in EXCLUDED_FROM_DEFAULT_CHAIN and not include_excluded:
            continue
        ordered.append(n)
        seen.add(n)
    ordered.sort(key=lambda n: REGISTRY[n].layer)
    return bits, ordered, ctx


def plan(
    params_b: float,
    target_bpw: float | None = None,
    *,
    is_moe: bool = False,
    entropy_floor: float | None = None,
    device: str = "studio-m2max",
) -> dict[str, Any]:
    bits, chain, ctx = select(
        params_b, target_bpw, is_moe=is_moe, entropy_floor=entropy_floor, device=device
    )
    return {
        "schema": "hawking.doctor.recovery_plan.v1",
        "params_b": params_b,
        "target_bits": bits,
        "ladder_eff_bpw": dict(LADDER)[bits],
        "target_bpw": target_bpw,
        "is_moe": is_moe,
        "device": device,
        "excluded_from_default": sorted(EXCLUDED_FROM_DEFAULT_CHAIN),
        "chain": [
            {
                "method": n,
                "layer": REGISTRY[n].layer,
                "stage": REGISTRY[n].stage,
                "train_free": REGISTRY[n].train_free,
                "status": REGISTRY[n].status,
                "live_entry": REGISTRY[n].live_entry,
                "note": REGISTRY[n].note,
            }
            for n in chain
        ],
        "ctx": ctx,
    }


def list_methods() -> list[dict[str, Any]]:
    rows = []
    for m in sorted(REGISTRY.values(), key=lambda x: (x.layer, x.name)):
        rows.append(
            {
                "method": m.name,
                "layer": m.layer,
                "stage": m.stage,
                "train_free": m.train_free,
                "serve": m.provides_serve,
                "status": m.status,
                "sensitivity": m.sensitivity,
                "live_entry": m.live_entry,
                "note": m.note,
            }
        )
    return rows


def emit_chain(
    params_b: float,
    target_bpw: float | None = None,
    *,
    is_moe: bool = False,
) -> list[tuple[str, str, tuple[Any, ...]]]:
    bits, chain, ctx = select(params_b, target_bpw, is_moe=is_moe)
    del bits
    rows = []
    for n in chain:
        fn = REGISTRY[n].build_fn
        if fn is None:
            continue
        spec = fn(ctx)
        if spec and spec[1] != "TODO" and not str(spec[1]).startswith("EXCLUDED"):
            rows.append(spec)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Doctor recovery registry (live lab operator)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="catalog of recovery methods")

    p_sel = sub.add_parser("select", help="compose chain for model/target")
    p_sel.add_argument("params_b", type=float)
    p_sel.add_argument("target_bpw", type=float, nargs="?", default=None)
    p_sel.add_argument("--moe", action="store_true")
    p_sel.add_argument("--floor", type=float, default=None)
    p_sel.add_argument("--device", default="studio-m2max")

    p_emit = sub.add_parser("emit-set", help="emit build specs for the chain")
    p_emit.add_argument("params_b", type=float)
    p_emit.add_argument("target_bpw", type=float, nargs="?", default=None)
    p_emit.add_argument("--moe", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "list":
        print(f"{'method':16s} {'L':2s} {'stage':7s} {'train-free':10s} {'status':9s} live")
        for m in list_methods():
            print(
                f"{m['method']:16s} {m['layer']:<2d} {m['stage']:7s} "
                f"{str(m['train_free']):10s} {m['status']:9s} {m['live_entry'] or '-'}"
            )
        print(f"\n{len(REGISTRY)} recovery methods registered.")
        return 0
    if args.cmd == "select":
        rec = plan(
            args.params_b,
            args.target_bpw,
            is_moe=args.moe,
            entropy_floor=args.floor,
            device=args.device,
        )
        print(json.dumps(rec, indent=2))
        return 0
    if args.cmd == "emit-set":
        rows = emit_chain(args.params_b, args.target_bpw, is_moe=args.moe)
        print(json.dumps(rows, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
