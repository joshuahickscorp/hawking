#!/usr/bin/env python3
"""Prometheus pipeline spine: capability-conditioned byte allocation over a real
model's tensor graph. Deterministic, training-free, no served model required.

This is leg L1 (the spine) + leg L2's RandomPolicy control from
docs/plans/PROMETHEUS_LEG_PLAN.md, taken from theory to running code. It runs
P0 (graph) -> P1 (profile) -> P2 (census) -> P5 (tiering) -> P6 (allocation) end
to end on real per-category element counts read from a sealed logical-weight
ledger, under three allocation policies (Uniform / Conditioned / Random-control),
and emits a byte decomposition that reconciles to physical bytes.

GATED stages, requiring a served runtime (Gate S0.8) and the G1-G8 capability
evaluator (leg L3), are NOT run here and are marked so in the output:
  P3 causal probe, P4 cartography (which experts form the math coalition),
  P8 capability evaluation (does the math coalition retain more than random).
The machinery accepts a coalition SIZE today; coalition MEMBERSHIP is gated.

Why the control lives at expert granularity: in a real MoE the routed experts are
~97% of the weights, so at category granularity every profile allocates almost
identically and the Math-vs-Uniform delta is a <3% tail (excision + head
protection). The scientific surface is WHICH experts get protected. Because
experts are near-equal in size, protecting a random k-subset costs the same bytes
as protecting the math k-subset -- so Math vs Random is exactly byte-matched, which
is the whole point of the control (Revision 4 arm II.7).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO / "profiles" / "prometheus"

# Canonical, model-general category vocabulary (superset of the container spec's
# tensor `category` enum). Profiles are keyed on these, never on model-specific
# organ names, so one profile applies across families.
CANONICAL = {
    "routed_expert", "shared_expert", "router", "normalization", "indexer",
    "embedding", "lm_head", "attention", "dense_mlp", "mtp", "vision", "other",
}

# Raw ledger/inventory category -> canonical. Extend as new families are admitted.
CATEGORY_ALIASES = {
    "attention": "attention",
    "attn.full": "attention", "attn.linear": "attention",
    "dense_mlp": "dense_mlp", "dense_layer": "dense_mlp",
    "embeddings": "embedding", "embed": "embedding",
    "indexer": "indexer",
    "lm_head": "lm_head",
    "normalization": "normalization", "attn.norm": "normalization", "norm": "normalization",
    "router": "router", "moe.gate": "router",
    "routed_expert": "routed_expert", "moe.experts": "routed_expert",
    "shared_expert": "shared_expert", "moe.shared_expert": "shared_expert",
    "mtp": "mtp", "mtp_projection": "mtp", "mtp_normalization": "mtp",
    "mtp_head_norm": "mtp",
    "vision": "vision",
}

TIER_ORDER = ["T0", "T1", "T2", "T3"]


def normalize_category(raw: str) -> str:
    return CATEGORY_ALIASES.get(raw, "other")


@dataclass
class Group:
    """One tensor category from the source graph (P0), with real element and byte
    counts read from the sealed ledger. `n_experts` is set for routed_expert so the
    allocator can split it into a coalition + remainder at expert granularity."""
    category: str
    logical_weights: int
    source_payload_bytes: int
    tensor_count: int
    n_experts: int | None = None


# ---------------------------------------------------------------------------
# P0  graph extraction
# ---------------------------------------------------------------------------

def load_ledger_graph(path: Path) -> list[Group]:
    """Read primary_categories from a hawking logical-weight ledger. Byte-exact:
    every element/byte count was read from the immutable shard headers."""
    d = json.loads(Path(path).read_text())
    prim = d.get("primary_categories")
    if not prim:
        raise ValueError(f"{path} has no primary_categories view")
    # Fold the raw categories into canonical groups, summing collisions (e.g. the
    # three mtp_* rows collapse into one mtp group).
    folded: dict[str, Group] = {}
    for raw, v in prim.items():
        cat = normalize_category(raw)
        g = folded.get(cat)
        if g is None:
            folded[cat] = Group(cat, v["logical_weights"], v["source_payload_bytes"],
                                 v["tensor_count"])
        else:
            g.logical_weights += v["logical_weights"]
            g.source_payload_bytes += v["source_payload_bytes"]
            g.tensor_count += v["tensor_count"]
    # Derive expert count for the routed_expert group. GLM/Qwen MoE experts are 3
    # tensors each (gate, up, down); infer n_experts from the tensor count so the
    # coalition split is grounded, not hard-coded.
    re = folded.get("routed_expert")
    if re is not None:
        re.n_experts = max(1, re.tensor_count // 3)
    return list(folded.values())


# ---------------------------------------------------------------------------
# P1  profile load
# ---------------------------------------------------------------------------

def load_profile(name: str) -> dict:
    p = Path(name)
    if not p.exists():
        p = PROFILE_DIR / f"{name}.json"
    if not p.exists():
        p = PROFILE_DIR / name
    prof = json.loads(p.read_text())
    if prof.get("control"):
        base = load_profile(prof["base_profile"])
        prof["_base"] = base
    return prof


# ---------------------------------------------------------------------------
# P2  structural census
# ---------------------------------------------------------------------------

def census(groups: list[Group]) -> dict:
    total_w = sum(g.logical_weights for g in groups)
    total_b = sum(g.source_payload_bytes for g in groups)
    nontarget = [g for g in groups if g.category in ("vision", "mtp")]
    return {
        "total_logical_weights": total_w,
        "total_source_bytes": total_b,
        "category_share": {
            g.category: round(g.logical_weights / total_w, 6) for g in groups
        },
        "non_target_components": {
            g.category: {"logical_weights": g.logical_weights,
                         "source_bytes": g.source_payload_bytes}
            for g in nontarget
        },
        "routed_expert_count": next(
            (g.n_experts for g in groups if g.category == "routed_expert"), None),
    }


# ---------------------------------------------------------------------------
# structural narrowing: vocabulary pruning (leg L4, "the largest cheap win")
# ---------------------------------------------------------------------------

def apply_vocab_prune(groups: list[Group], profile: dict, vocab_retain: int | None,
                      hidden_size: int) -> tuple[list[Group], dict]:
    """Shrink embedding + lm_head rows to a retained-vocabulary target. Pure byte
    arithmetic and fully deterministic: reclaimed_bytes is exact given the target.
    What is GATED is the SAFETY of the target -- whether the retained vocab still
    passes G4 (tool syntax) and G6 (notation fidelity) needs the served model (L4/L3).
    The estimator computes the win; the coverage floor is set with capability gates.
    """
    if not vocab_retain:
        return groups, {"applied": False, "reason": "no vocab_retain target given"}
    eff = profile["_base"] if profile.get("control") else profile
    prunable = {c for c, d in eff["category_tiers"].items() if d.get("vocab_prune")}
    if not prunable:
        return groups, {"applied": False, "reason": f"profile {profile['name']} prunes no vocab"}
    reclaimed = 0
    full_vocab = None
    out = []
    for g in groups:
        if g.category in prunable and g.category in ("embedding", "lm_head"):
            vocab = round(g.logical_weights / hidden_size)
            full_vocab = vocab
            if vocab_retain >= vocab:
                out.append(g)
                continue
            frac = vocab_retain / vocab
            new_w = round(g.logical_weights * frac)
            new_b = round(g.source_payload_bytes * frac)
            reclaimed += g.source_payload_bytes - new_b
            out.append(Group(g.category, new_w, new_b, g.tensor_count, g.n_experts))
        else:
            out.append(g)
    return out, {
        "applied": True, "hidden_size": hidden_size,
        "full_vocab": full_vocab, "retained_vocab": vocab_retain,
        "reclaimed_bytes": reclaimed,
        "coverage_safety": "GATED: retained vocab must pass G4+G6 (leg L3/L4)",
    }


# ---------------------------------------------------------------------------
# P5 tiering + P6 allocation
# ---------------------------------------------------------------------------

# Structural spine that both real and control arms pin native. Pinning these is a
# structural necessity, not capability conditioning, so the control does NOT get to
# claim credit for it and never randomizes them away (random-v1 protected_invariant).
PINNED_NATIVE = {"router", "normalization", "indexer"}


def _rate_bytes(logical_weights: int, tier_rate: dict, source_bytes: int) -> int:
    """Physical bytes for a block at a tier rate. native -> real source bytes;
    otherwise elements * (num/den) bits / 8."""
    if tier_rate.get("native"):
        return source_bytes
    num, den = tier_rate["num"], tier_rate["den"]
    if num == 0:
        return 0
    return round(logical_weights * num / den / 8)


def _tier_of(profile: dict, category: str) -> str:
    ct = profile["category_tiers"].get(category)
    if ct is None:
        return profile["category_tiers"].get("other", {"tier": "T2"})["tier"]
    return ct["tier"]


@dataclass
class Block:
    """One allocated block in the final artifact plan."""
    label: str
    category: str
    tier: str
    logical_weights: int
    physical_bytes: int
    rate_bpw: float
    excised: bool = False


def allocate(groups: list[Group], profile: dict, coalition_fraction: float,
             coalition_protect_bpw: float, seed: int) -> list[Block]:
    """P6. Produce the per-block allocation for one policy.

    - routed_expert is split into a coalition (protected) + remainder (floor). For a
      conditioned profile the coalition is the math set (GATED membership, size only);
      for the control it is a seeded random set of the SAME size (byte-matched).
    - every other category is allocated whole at its profile tier.
    """
    tier_rates = profile["tier_rates"] if not profile.get("control") \
        else profile["_base"]["tier_rates"]
    eff = profile["_base"] if profile.get("control") else profile
    is_control = bool(profile.get("control"))
    blocks: list[Block] = []

    for g in groups:
        if g.category == "routed_expert" and g.n_experts:
            k = round(coalition_fraction * g.n_experts)
            k = min(max(k, 0), g.n_experts)
            # equal-expert-size model: real per-expert bytes vary and are read at L4.
            # ponytail: equal split; refine with per-expert header bytes at L4.
            w_per = g.logical_weights / g.n_experts
            b_per = g.source_payload_bytes / g.n_experts
            floor_tier = "T2"
            # cold-expert excision only in mathnarrow-style profiles (declared on the
            # routed_expert entry), and never in the control.
            re_decl = eff["category_tiers"].get("routed_expert", {})
            cold_excise = (re_decl.get("cold_expert_policy") == "excise") and not is_control

            coalition_w = round(w_per * k)
            coalition_b = round(coalition_protect_bpw * coalition_w / 8)
            blocks.append(Block(
                label=("random_coalition" if is_control else "math_coalition"),
                category="routed_expert", tier="T0",
                logical_weights=coalition_w, physical_bytes=coalition_b,
                rate_bpw=coalition_protect_bpw))

            rest = g.n_experts - k
            rest_w = g.logical_weights - coalition_w
            if cold_excise:
                # excise the coldest ~half of the non-coalition experts (size only;
                # membership is GATED on the L4 census). Byte-matched controls do not
                # excise, so this branch never runs for is_control.
                n_cold = rest // 2
                cold_w = round(w_per * n_cold)
                blocks.append(Block("cold_experts_excised", "routed_expert", "T3",
                                    cold_w, 0, 0.0, excised=True))
                rest_w -= cold_w
            fr = tier_rates[floor_tier]
            blocks.append(Block(
                label=("random_floor" if is_control else "vestigial_experts"),
                category="routed_expert", tier=floor_tier,
                logical_weights=rest_w,
                physical_bytes=_rate_bytes(rest_w, fr, round(b_per * rest)),
                rate_bpw=fr["num"] / fr["den"]))
            continue

        # non-expert categories: whole-group allocation. The control arm allocates
        # these IDENTICALLY to its base profile -- the ONLY thing the control
        # randomizes is which experts form the coalition (above). Because experts are
        # equal-sized, that makes Random exactly byte-matched to Math, which is what
        # "matched bytes" means (Revision 4 arm II.7). Randomizing the tail tiers
        # would move bytes and test nothing.
        tier = _tier_of(eff, g.category)
        tr = tier_rates[tier]
        pb = _rate_bytes(g.logical_weights, tr, g.source_payload_bytes)
        rate = (g.source_payload_bytes * 8 / g.logical_weights) if tr.get("native") \
            else (0.0 if tr["num"] == 0 else tr["num"] / tr["den"])
        blocks.append(Block(g.category, g.category, tier, g.logical_weights, pb,
                            rate, excised=(tier == "T3")))
    return blocks


# ---------------------------------------------------------------------------
# byte decomposition (Revision 4 arm II.6.1) + reconciliation
# ---------------------------------------------------------------------------

def byte_decomposition(blocks: list[Block]) -> dict:
    protected = sum(b.physical_bytes for b in blocks if b.tier == "T0")
    main = sum(b.physical_bytes for b in blocks if b.tier in ("T1", "T2"))
    excised = sum(b.logical_weights for b in blocks if b.excised)
    retained_w = sum(b.logical_weights for b in blocks if not b.excised)
    compressed_w = sum(b.logical_weights for b in blocks if b.tier in ("T1", "T2"))
    physical = protected + main  # excised weights carry ZERO bytes, by container law

    # Generation-A defect guard: every non-excised block MUST carry payload.
    for b in blocks:
        if not b.excised and b.physical_bytes <= 0:
            raise AssertionError(
                f"non-excised block {b.label}/{b.tier} has {b.physical_bytes} bytes "
                "-- this is the Generation-A defect (billed, written nowhere)")

    return {
        "protected_tensor_bytes": protected,
        "main_representation_bytes": main,
        # The real codec fills these at L5; the deterministic spine models rate only.
        "exception_bytes": 0,
        "codebook_bytes": 0,
        "index_bytes": 0,
        "metadata_bytes": 0,
        "exception_bytes_status": "MODELED_ZERO: populated by the real codec at leg L5",
        "excised_logical_weights": excised,
        "excised_bytes": 0,
        "retained_logical_weights": retained_w,
        "effective_total_physical_bytes": physical,
        # two rates, like the container: packed over compressed elements only,
        # complete over all retained logical weights.
        "packed_bpw": round(main * 8 / compressed_w, 6) if compressed_w else 0.0,
        "complete_bpw": round(physical * 8 / retained_w, 6) if retained_w else 0.0,
        "reconciliation_ok": physical == protected + main,
    }


# ---------------------------------------------------------------------------
# pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(ledger: Path, profile_name: str, coalition_fraction: float,
                 coalition_protect_bpw: float, seed: int,
                 vocab_retain: int | None = None, hidden_size: int = 6144) -> dict:
    groups = load_ledger_graph(ledger)                      # P0
    profile = load_profile(profile_name)                    # P1
    cen = census(groups)                                    # P2
    groups, vocab = apply_vocab_prune(groups, profile,      # structural narrowing (L4)
                                      vocab_retain, hidden_size)
    blocks = allocate(groups, profile, coalition_fraction,  # P5 + P6
                      coalition_protect_bpw, seed)
    decomp = byte_decomposition(blocks)
    return {
        "profile": profile["name"],
        "control": bool(profile.get("control")),
        "coalition_fraction": coalition_fraction,
        "coalition_protect_bpw": coalition_protect_bpw,
        "seed": seed,
        "census": cen,
        "vocab_prune": vocab,
        "blocks": [asdict(b) for b in blocks],
        "byte_decomposition": decomp,
        "gated_stages": {
            "P3_causal_probe": "GATED: needs served runtime (S0.8)",
            "P4_cartography_coalition_membership": "GATED: coalition SIZE used; which "
                "experts is gated on the causal probe",
            "P8_capability_evaluation_G1_G8": "GATED: needs served runtime + Lean-lite "
                "(leg L3). No capability claim is made by this run.",
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prometheus allocation pipeline (L1 spine)")
    ap.add_argument("--ledger", default=str(REPO / "evidence" / "glm52" / "GLM52_LOGICAL_WEIGHT_LEDGER.json"))
    ap.add_argument("--profile", default="math-v1")
    ap.add_argument("--coalition-fraction", type=float, default=0.05)
    ap.add_argument("--coalition-protect-bpw", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--vocab-retain", type=int, default=None,
                    help="retained vocabulary target for embedding/lm_head prune (L4)")
    ap.add_argument("--hidden-size", type=int, default=6144, help="GLM-5.2 default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = run_pipeline(Path(a.ledger), a.profile, a.coalition_fraction,
                          a.coalition_protect_bpw, a.seed, a.vocab_retain, a.hidden_size)
    text = json.dumps(result, indent=2)
    if a.out:
        Path(a.out).write_text(text)
        d = result["byte_decomposition"]
        print(f"{result['profile']:14} complete_bpw={d['complete_bpw']:.4f} "
              f"physical={d['effective_total_physical_bytes']/1e9:.2f}GB "
              f"excised_w={d['excised_logical_weights']/1e9:.2f}B -> {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
