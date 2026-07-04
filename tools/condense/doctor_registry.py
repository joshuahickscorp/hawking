#!/usr/bin/env python3.12
"""doctor_registry.py — THE DOCTOR, expanded. "The doctor" was never one function; it is the name
for the whole program of restoring quality at low bits. This makes that explicit: a pluggable
REGISTRY of recovery methods (L0..L6 + extensions), an auto-SELECTOR that composes the right chain
for a (model, target-bpw, device), and a leverage-ordered catalog. New recovery methods register
themselves with a decorator and immediately become available to the selector and the ledger — no
edit to studio_run/audit_ladder needed. Advisor/orchestration only (pure stdlib): a method's
build_fn returns a BakeSpec (baker argv + optional shadow-weights step + eff-bpw hooks); the driver
runs it, so composition/ordering lives in one place.

CLI:
  doctor_registry.py --list                          # the catalog (layer, stage, train-free, serve)
  doctor_registry.py --select <params_b> <bpw> [--moe] [--floor F] [--device studio-m2max]
                                                     # the recommended ordered recovery chain
  doctor_registry.py --emit-set <params_b> <bpw>     # emit the chain as audit_ladder-style config rows
"""
import sys, os, math, json
from dataclasses import dataclass, field
from typing import Callable, Optional

LADDER = [(1, 1.34), (2, 2.34), (3, 3.34), (4, 4.5)]
OUT = "reports/condense"


@dataclass
class RecoveryMethod:
    name: str
    layer: int                       # L0..L6 leverage rank (cheapest/most-general first)
    stage: str                       # 'local' | 'studio'
    train_free: bool
    sensitivity: str                 # 'global' | 'per_tensor' | 'per_expert'
    tool: str                        # the script/CLI that implements it
    provides_serve: bool             # does its artifact serve natively (vs needing more build)
    min_params_b: float = 0.0
    max_params_b: Optional[float] = None
    status: str = "MEASURED"         # MEASURED | GATED | UNPROVEN | DEAD
    note: str = ""
    build_fn: Optional[Callable] = field(default=None, repr=False)


REGISTRY: dict = {}


def register(**kw):
    def deco(fn):
        m = RecoveryMethod(build_fn=fn, **kw)
        REGISTRY[m.name] = m
        return fn
    return deco


# ---- the catalog: existing levers (thin build_fns emit an audit_ladder-style (name, fn, args) spec) ----
@register(name="calib", layer=0, stage="local", train_free=True, sensitivity="global",
          tool="calib_build.py", provides_serve=True, status="MEASURED",
          note="domain-matched calibration; multiplies every layer below")
def _calib(ctx): return ("calib", "build_calib", ())

@register(name="awq", layer=1, stage="local", train_free=True, sensitivity="per_tensor",
          tool="awq_bake.py", provides_serve=True, status="MEASURED",
          note="alpha=0.5 activation-aware pre-scale; halves the raw gap at 3-4bit")
def _awq(ctx): return (f"{ctx['bits']}-AWQ", "build_awq", (ctx["bits"], 0.5))

@register(name="mixed_prec", layer=2, stage="local", train_free=True, sensitivity="per_tensor",
          tool="mixed_precision.py", provides_serve=True, status="MEASURED",
          note="output-sensitivity water-fill: sensitive tensors get depth, tolerant get starved. Biggest NEW density lever.")
def _mp(ctx): return ("mp-4a3f", "build_awq", (3, 0.5, {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
                                                         "gate_proj": 3, "up_proj": 3, "down_proj": 3}))

@register(name="residual", layer=3, stage="local", train_free=True, sensitivity="per_tensor",
          tool="residual.py bake", provides_serve=True, status="MEASURED",
          note="W ~= STRAND(W)+STRAND(residual); train-free ~1:1; two-part serve (parity-green)")
def _res(ctx): return (f"res{ctx['bits']}+1", "build_residual", (ctx["bits"], 1))

@register(name="outlier_channel", layer=3, stage="local", train_free=True, sensitivity="per_tensor",
          tool="audit_ladder.build_awq(outlier_pct)", provides_serve=True, status="MEASURED",
          note="keep top-|w| 5-10% at 8-bit sparse channel (OUTL wire); train-free sub-3-bit rescue")
def _outl(ctx): return (f"{ctx['bits']}-AWQ-o5", "build_awq", (ctx["bits"], 0.5, None, 5.0))

@register(name="block_qat", layer=4, stage="studio", train_free=False, sensitivity="per_tensor",
          tool="doctor_blockwise.py", provides_serve=True, min_params_b=0.0, status="GATED",
          note="BRECQ-lite full-rank per-linear QAT; the LoRA-plateau fix; studio at 7B+")
def _bw(ctx): return (f"{ctx['bits']}-bw", "build_blockwise", (ctx["bits"],))

@register(name="gptq_hessian", layer=5, stage="studio", train_free=False, sensitivity="per_tensor",
          tool="doctor_strand.py", provides_serve=True, status="UNPROVEN",
          note="codec-native sequential error-feedback (NO uniform STE — that path is DEAD); sub-residual edge")
def _str(ctx): return (f"{ctx['bits']}-str", "build_strand", (ctx["bits"],))

@register(name="deep_kd", layer=6, stage="studio", train_free=False, sensitivity="global",
          tool="doctor_lora.py", provides_serve=True, status="GATED",
          note="logit/feature/attn KD polish on the full-rank base; recovers near the base bpw")
def _kd(ctx): return (f"{ctx['bits']}-AWQ+dr", "build_recover", (ctx["bits"],))

@register(name="expert_alloc", layer=2, stage="studio", train_free=True, sensitivity="per_expert",
          tool="expert.py sensitivity", provides_serve=False, min_params_b=100.0, status="GATED",
          note="MoE per-expert bit allocation: router/shared high-bit, hot 2-bit, cold 1-bit/ternary")
def _expert(ctx): return ("expert-alloc", "per_expert", (ctx["bits"],))

# ---- extension slots the audit named as MISSING (registered as UNPROVEN so the selector knows they exist) ----
@register(name="learned_rotation", layer=1, stage="local", train_free=True, sensitivity="per_tensor",
          tool="rotation_search.py (TODO)", provides_serve=True, status="UNPROVEN",
          note="QuaRot/SpinQuant learned orthogonal rotation before the cut; ~0 serve bpw; may beat RHT")
def _rot(ctx): return ("rot", "TODO", ())

@register(name="big_teacher_kd", layer=6, stage="studio", train_free=False, sensitivity="global",
          tool="doctor_lora.py --teacher (TODO)", provides_serve=True, status="UNPROVEN",
          note="a larger model distilling the condensed one; needs both resident; high upside")
def _bigkd(ctx): return ("bigkd", "TODO", ())


def _floor_bits(params_b, entropy_floor=None):
    if entropy_floor:
        for b, bpw in LADDER:
            if bpw >= entropy_floor:
                return b
    raw = 3.6 - 0.9 * math.log10(max(0.5, params_b))
    for b, bpw in LADDER:
        if bpw >= raw:
            return b
    return 4


def select(params_b, target_bpw=None, is_moe=False, entropy_floor=None, device="studio-m2max"):
    """Compose the recovery chain for this model. Cheapest-leverage-first train-free stack always;
    add training layers only when needed and size-appropriate; per-expert for MoE."""
    bits = None
    for b, bpw in LADDER:
        if target_bpw and abs(bpw - target_bpw) < 0.5:
            bits = b
    bits = bits or _floor_bits(params_b, entropy_floor)
    ctx = {"params_b": params_b, "bits": bits, "is_moe": is_moe}
    chain = []
    # 1) always the train-free stack (L0-L3) + outlier at sub-3-bit
    for name in ("calib", "awq", "mixed_prec", "residual"):
        chain.append(name)
    if bits <= 2:
        chain.append("outlier_channel")
    # 2) MoE: per-expert allocation before the bake
    if is_moe:
        chain.insert(1, "expert_alloc")
    # 3) training recovery only if the target is aggressive (train-free likely leaves a gap)
    if bits <= 2:
        chain += (["block_qat", "gptq_hessian", "deep_kd"] if params_b >= 7 else ["block_qat", "deep_kd"])
    elif bits == 3:
        chain.append("deep_kd")
    # de-dup preserve order; annotate stage feasibility
    seen, ordered = set(), []
    for n in chain:
        if n in REGISTRY and n not in seen:
            ordered.append(n); seen.add(n)
    ordered.sort(key=lambda n: REGISTRY[n].layer)
    return bits, ordered, ctx


def cmd_select(params_b, target_bpw, is_moe, floor, device):
    bits, chain, ctx = select(params_b, target_bpw, is_moe, floor, device)
    rec = {"params_b": params_b, "target_bits": bits, "target_bpw": dict(LADDER)[bits],
           "is_moe": is_moe, "chain": [
               {"method": n, "layer": REGISTRY[n].layer, "stage": REGISTRY[n].stage,
                "train_free": REGISTRY[n].train_free, "status": REGISTRY[n].status} for n in chain]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/doctor_plan_{int(params_b)}b.json", "w"), indent=2)
    print(f"[doctor] {params_b}B{' MoE' if is_moe else ''} -> target {bits}-bit ({dict(LADDER)[bits]} eff-bpw)", file=sys.stderr)
    print(f"[doctor] recovery chain (leverage-ordered):", file=sys.stderr)
    for n in chain:
        m = REGISTRY[n]
        print(f"   L{m.layer} {n:16s} [{m.stage:6s} {'train-free' if m.train_free else 'TRAINING ':10s} {m.status:9s}] {m.note}", file=sys.stderr)
    print("# fallback: this is the STARTING chain; the floor-search confirms/steps up (2->3->4) on the +2% gate.",
          file=sys.stderr)


def cmd_list():
    print(f"{'method':16s} {'L':2s} {'stage':7s} {'train-free':10s} {'serve':6s} {'status':9s} sens")
    for m in sorted(REGISTRY.values(), key=lambda x: (x.layer, x.name)):
        print(f"{m.name:16s} {m.layer:<2d} {m.stage:7s} {str(m.train_free):10s} "
              f"{str(m.provides_serve):6s} {m.status:9s} {m.sensitivity}")
    print(f"\n{len(REGISTRY)} recovery methods registered. The Doctor = this whole registry, not one script.")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--list"
    if a == "--list":
        cmd_list()
    elif a == "--select":
        cmd_select(float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None,
                   "--moe" in sys.argv,
                   float(sys.argv[sys.argv.index("--floor")+1]) if "--floor" in sys.argv else None,
                   sys.argv[sys.argv.index("--device")+1] if "--device" in sys.argv else "studio-m2max")
    elif a == "--emit-set":
        _, chain, ctx = select(float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None)
        rows = [REGISTRY[n].build_fn(ctx) for n in chain if REGISTRY[n].build_fn]
        print(json.dumps([r for r in rows if r and r[1] != "TODO"], default=str))
    else:
        print(__doc__)
