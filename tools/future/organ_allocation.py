"""G017: spend a matched complete-EBPW budget unequally across organs.

The G002 organ prior says structure EXISTS unequally: q/k/o first, up_proj last.
It does not say factoring PAYS, and it does not say capability survives. This
module spends the SAME complete EBPW two ways (UNIFORM vs ORGAN_WEIGHTED) plus
the INVERTED negative control. The NR is affine (lowrank_nr.encode_affine)
because within-tensor low-rank cannot undercut 16 EBPW on square k/v without
dropping below break-even rank, and that already blows this specimen's
conjunction gate. Accounting is still finalize_parts.

Capability is the existing conjunction (perplexity AND 4-gram diversity),
calibrated on this specimen's own dense parent, with the same multipliers
gravity_outlier_eval.evaluate uses against its bf16 reference.

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
        tools/future/organ_allocation.py --selfcheck
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
        tools/future/organ_allocation.py --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

RECEIPT_NAME = "G017_ORGAN_ALLOCATION.json"
RECORDED_BY = "tools/future/organ_allocation.py"
SCHEMA = "hawking.future.g017_organ_allocation.v1"

SPECIMEN_SLUG = "Qwen--Qwen3-0.6B@c1899de289a0"
SPECIMEN_PATH = (
    "/Volumes/corpdrive/hawking-modellake/specimens/" + SPECIMEN_SLUG
)
SWEEP_REL = "receipts/future/G002_DENSE_ANATOMY_SWEEP.json"
PRIOR_REL = "receipts/future/G002_ORGAN_PRIOR.json"

# G002_ORGAN_PRIOR.json deficit_by_organ.*.median (commit ac08d525e).
# High number = more within-tensor structure vs a same-shape null.
PRIOR_DEFICIT: dict[str, float] = {
    "q_proj": 48.1,
    "k_proj": 39.22,
    "o_proj": 27.29,
    "v_proj": 19.35,
    "gate_proj": 18.37,
    "down_proj": 12.3,
    "up_proj": 9.46,
}
STANDARD_ORGANS: tuple[str, ...] = tuple(PRIOR_DEFICIT)
# LOCAL orders by THIS specimen's own measured deficits instead of the aggregate.
# [S010 17] asks for three policies -- UNIFORM, GLOBAL-PRIOR and LOCAL-ANATOMY --
# because G017 already falsified the universal global rule, and both bodies whose
# anatomy has been measured invert that prior on the same two pairs: local k>q
# against global q>k, local down>gate against global gate>down. ORGAN_WEIGHTED is
# the global arm; INVERTED remains its control.
POLICIES: tuple[str, ...] = ("UNIFORM", "ORGAN_WEIGHTED", "INVERTED", "LOCAL")

# UNIFORM affine spec. q4g64 on every linear organ is the cheapest NR that
# still sits on this specimen's conjunction gate (dense ppl 4.73, q4g64 ppl
# 5.23, gate 5.92). Within-tensor low-rank at any byte-saving EBPW is not a
# discriminator here: square k/v break-even is 50% rank / 16 EBPW, and a
# single layer's attention at that rank already blows the ppl gate.
UNIFORM_AFFINE = (4, 64)
TARGET_ORGAN_EBPW = 4.5  # q4g64 complete EBPW on these shapes (bits + 32/group)

# Affine grid the allocator may pick from. Same bits/groups lowrank_nr bills.
AFFINE_CFGS: tuple[tuple[int, int], ...] = (
    (2, 32), (2, 64), (2, 128),
    (3, 32), (3, 64), (3, 128),
    (4, 32), (4, 64), (4, 128),
    (8, 32), (8, 64), (8, 128),
)

# Same conjunction multipliers as gravity_outlier_eval.evaluate:
# R4_MAX = 3.0 * specimen_dense_r4, PPL_MAX = 1.25 * specimen_dense_ppl.
R4_MULT = 3.0
PPL_MULT = 1.25
GEN_TOKENS = 96
NLL_TOKENS = 512

# Verbatim from gravity_outlier_eval — the existing gate's prompts, not a new set.
PROMPTS = [
    "Explain how a mixture-of-experts layer routes tokens to experts.",
    "What is the capital of France, and why did it become the capital?",
    "Write a short Python function that reverses a linked list.",
    "Summarize why bandwidth, not compute, limits transformer decoding.",
    "A farmer has 17 sheep and all but 9 run away. How many are left?",
    "Describe the difference between a scale and a zero point in quantization.",
    "List three reasons a neural network might fail to converge.",
    "Translate to French: The weather is cold today.",
]
NLL_TEXT = (
    "The mixture-of-experts architecture routes each token to a small subset of "
    "feed-forward networks, so the number of parameters activated per token is much "
    "smaller than the total parameter count. A router network produces a distribution "
    "over experts and the top-k are selected. This makes the model sparse in compute "
    "while remaining dense in memory, which is why storage dominates the cost."
) * 3

# CORPUS B. Every capability number this campaign has is from CORPUS A above,
# which is ML-shop English: MoE routing, quantization scales, transformer
# bandwidth, one arithmetic trick, one translation. Two results now rest on it --
# GLOBAL_ANATOMY's r4 0.4135 win, and the NEGATIVE sensitivity of gate_proj and
# o_proj -- and neither can be believed until it reproduces on prompts that do
# not share that vocabulary. B is deliberately a different register and domain:
# narrative, historical, biological, culinary, legal. No transformer words.
PROMPTS_B = [
    "Describe what a lighthouse keeper does during a winter storm.",
    "Why did medieval towns build walls, and what ended the practice?",
    "Write a short recipe for lentil soup for four people.",
    "Explain how photosynthesis turns sunlight into sugar.",
    "A baker sells 40 loaves on Monday and half that on Tuesday. How many in total?",
    "What is the difference between a lease and a licence?",
    "Name three reasons a bridge might be closed for repair.",
    "Translate to Spanish: The library opens at nine in the morning.",
]
NLL_TEXT_B = (
    "The great auk was a flightless seabird of the North Atlantic that stood about "
    "seventy centimetres tall and nested on remote rocky islands. It swam well and "
    "walked poorly, which made the breeding colonies easy for sailors to raid for "
    "meat, eggs and down. The last confirmed pair was killed in Iceland in 1844, and "
    "the species is now the standard example of a bird hunted to extinction."
) * 3

CORPORA = {
    "A": {"prompts": PROMPTS, "nll": NLL_TEXT,
          "note": "ML-shop English: MoE routing, quantization, transformer decoding"},
    "B": {"prompts": PROMPTS_B, "nll": NLL_TEXT_B,
          "note": "narrative/historical/biological/culinary/legal; no transformer vocabulary"},
}
ACTIVE_CORPUS = "A"

WHY_THIS_SPECIMEN = (
    "Cheapest DENSE causal LM in G002_DENSE_ANATOMY_SWEEP.json that still "
    "carries all seven standard organs and can run the conjunction gate "
    "(perplexity AND n-gram generation). Sweep row Qwen--Qwen3-0.6B@c1899de289a0: "
    "1.40 GiB, family qwen3, 28 layers, 1 shard, anatomy wall 6.83s, peak RSS "
    "2.963 GiB. Measured mid-depth deficits: k_proj 41.27, q_proj 38.82, "
    "o_proj 23.66, v_proj 22.24, down_proj 21.58, gate_proj 20.16, up_proj 11.59. "
    "The three 100% prior laws hold (o>up, q>up, gate>up). Local order is k>q "
    "not q>k, and down>gate not gate>down — not a counterexample of the 100% "
    "laws; the global prior is still the allocation, this body is the transfer "
    "test. Smaller rows were refused as non-LMs (sam2, depth-anything, "
    "modernbert, whisper, timesfm, lfm2) or as an embedding body that cannot "
    "generate (Qwen3-Embedding-0.6B, 1.10 GiB). No MoE body: the prior is dense."
)

# The sort key that makes ORGAN_WEIGHTED put high-deficit organs first.
# MUTATION_ANCHOR_ORDER_BY_DEFICIT — tests and the live mutation check both
# hang on the leading minus. Reverting it must make a test FAIL.
_ORDER_KEY_DESCENDING = "-src[name]"


def order_organs_by_deficit(deficits: dict[str, float] | None = None) -> tuple[str, ...]:
    """Most structure first, least structure last.

    `deficits` defaults to the GLOBAL prior. Pass a specimen's OWN measured
    deficits for the LOCAL policy -- the two disagree, which is the whole point
    of the comparison and the reason the experiment can discriminate at all.
    """
    src = PRIOR_DEFICIT if deficits is None else deficits
    return tuple(sorted(src, key=lambda name: -src[name]))  # MUTATION_ANCHOR_ORDER_BY_DEFICIT


def local_deficits(slug: str | None = None) -> dict[str, float]:
    """THIS specimen's own measured organ deficits, from the anatomy sweep.

    The evidence path is SWEEP_REL; there is no hardcoded copy and no fallback
    to PRIOR_DEFICIT. A missing row is a refusal to run the LOCAL arm, not a
    licence to run the global arm under a local label. [S010 11]
    """
    row = _local_ordering_from_sweep(slug)
    ordering = (row or {}).get("organ_ordering") or []
    out = {
        str(r["organ"]): float(r["deficit_pct"])
        for r in ordering
        if r.get("organ") in PRIOR_DEFICIT and r.get("deficit_pct") is not None
    }
    missing = sorted(set(PRIOR_DEFICIT) - set(out))
    if missing:
        raise ValueError(
            f"{SWEEP_REL} has no measured deficit for {missing} on "
            f"{slug or SPECIMEN_SLUG}; the LOCAL arm cannot be run from the "
            "global prior")
    return out


def deficits_for(policy: str, slug: str | None = None) -> dict[str, float] | None:
    """None means 'use the GLOBAL prior'. LOCAL alone gets the specimen's own."""
    return local_deficits(slug) if policy == "LOCAL" else None


def per_param_weights(policy: str, deficits: dict[str, float] | None = None) -> dict[str, float]:
    """Bytes-per-parameter weights. Rank 0 (front of the order) is compressed most.

    Endpoint ratio is the prior's max/min deficit, so the tilt is measured
    rather than a free slope. UNIFORM is weight 1. INVERTED reverses the order,
    which is the control: if that wins, the prior did not transfer.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    # LOCAL must never fall back to the global prior. A silent fallback would run
    # a global-prior arm and record it under a local label, which is the one
    # outcome that makes the whole three-arm comparison meaningless.
    if policy == "LOCAL" and not deficits:
        raise ValueError(
            "LOCAL allocation needs THIS specimen's own measured deficits; falling back to "
            "the global prior would run a global arm under a local name")
    src = PRIOR_DEFICIT if deficits is None else deficits
    if policy == "UNIFORM":
        return {name: 1.0 for name in src}
    order = order_organs_by_deficit(deficits)
    if policy == "INVERTED":
        order = tuple(reversed(order))
    n = len(order)
    # Endpoint ratio is remaining unstructured fraction (1 - deficit/100),
    # not max/min deficit. The raw 5.08x deficit ratio is a real number in
    # the prior and a disastrous byte map: it assigned q_proj 2.35 EBPW at a
    # mean-8 budget and destroyed likelihood on every arm. Remaining-fraction
    # ratio is ~1.745 and still monotone in the prior order, so the mutation
    # of that order still reverses who gets crushed.
    most = max(src.values())
    least = min(src.values())
    rem_most = 1.0 - most / 100.0
    rem_least = 1.0 - least / 100.0
    if rem_most <= 0:
        raise ValueError("prior deficit >= 100; remaining fraction is not a weight")
    ratio = rem_least / rem_most
    out: dict[str, float] = {}
    for i, name in enumerate(order):
        out[name] = 1.0 if n == 1 else 1.0 + (ratio - 1.0) * i / (n - 1)
    return out


def allocate_organ_ebpw(
    policy: str,
    organ_params: dict[str, int],
    target_mean_ebpw: float = TARGET_ORGAN_EBPW,
    deficits: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-organ complete EBPW whose param-weighted mean is target_mean_ebpw.

    ORGAN_WEIGHTED: more bytes per param to low-deficit organs (up_proj last
    in the prior, so it receives the bytes saved by compressing q/k/o first).
    INVERTED: the same unequal allocation backwards.
    """
    weights = per_param_weights(policy, deficits)
    names = [n for n in organ_params if n in weights]
    if not names:
        raise ValueError("organ_params has no standard organs")
    total_p = sum(int(organ_params[n]) for n in names)
    if total_p <= 0:
        raise ValueError("organ_params must be positive")
    weighted = sum(weights[n] * int(organ_params[n]) for n in names)
    alpha = float(target_mean_ebpw) * total_p / weighted
    return {n: alpha * weights[n] for n in names}


def organ_family(key: str) -> str | None:
    if not key.endswith(".weight"):
        return None
    if "expert" in key.lower():
        return None
    leaf = key[: -len(".weight")].rsplit(".", 1)[-1]
    if leaf in PRIOR_DEFICIT:
        return leaf
    return None


def _dtype_bytes(dtype: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1}.get(dtype, 2)


def _n_elem(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def ebpw_of(complete_bytes: int, source_params: int) -> float:
    if int(source_params) <= 0:
        raise ValueError("source_params must be the dense parent")
    return int(complete_bytes) * 8.0 / int(source_params)


def ebpw4(complete_bytes: int, source_params: int) -> float:
    return round(ebpw_of(complete_bytes, source_params), 4)


# ---------------------------------------------------------------------------
# affine NR billed through lowrank_nr.finalize_parts
# ---------------------------------------------------------------------------

def _nr():
    import lowrank_nr as nr
    return nr


def affine_bill(m: int, n: int, bits: int, group: int) -> tuple[dict[str, int], int, float]:
    """Same numerator as encode_affine: packed payload + fp16 scale/zero + metadata."""
    nr = _nr()
    m, n, bits, group = int(m), int(n), int(bits), int(group)
    src = nr.source_params_of((m, n))
    n_g = m * nr._n_groups(n, group)
    parts = nr.empty_parts()
    parts["dense_payload"] = nr.packed_bytes(src, bits)
    parts["A_scales"] = n_g * nr.SCALE_BYTES
    parts["A_zeros"] = n_g * nr.ZERO_BYTES
    parts["metadata"] = nr.METADATA_BYTES
    return nr.finalize_parts(parts, src, True)


def family_affine_table(inv: dict[str, Any]) -> dict[str, dict[tuple[int, int], tuple[int, float]]]:
    """family -> {(bits,group): (family_complete_bytes, ebpw)}."""
    n_by = {fam: 0 for fam in PRIOR_DEFICIT}
    shape: dict[str, tuple[int, int]] = {}
    for t in inv["tensors"]:
        n_by[t["family"]] += 1
        shape[t["family"]] = (t["m"], t["n"])
    out: dict[str, dict[tuple[int, int], tuple[int, float]]] = {}
    for fam, (m, n) in shape.items():
        slot = {}
        for cfg in AFFINE_CFGS:
            _parts, complete, ebpw = affine_bill(m, n, cfg[0], cfg[1])
            slot[cfg] = (complete * n_by[fam], ebpw)
        out[fam] = slot
    return out


def search_affine_assignment(
    policy: str,
    inv: dict[str, Any],
    target_bytes: int,
    deficits: dict[str, float] | None = None,
) -> dict[str, tuple[int, int]]:
    """Monotone affine specs along the prior order, closest to target_bytes.

    ORGAN_WEIGHTED: complete EBPW nondecreasing along order_organs_by_deficit
    (q/k/o first = fewest bytes, up_proj last = the saved bytes).
    INVERTED: nonincreasing along the same order.
    UNIFORM: every organ UNIFORM_AFFINE.
    Ties at the same byte gap: smallest param-weighted L2 to allocate_organ_ebpw.
    """
    if policy == "UNIFORM":
        return {fam: UNIFORM_AFFINE for fam in PRIOR_DEFICIT if inv["organ_params"].get(fam)}
    order = order_organs_by_deficit(deficits)
    names = [n for n in order if inv["organ_params"].get(n)]
    table = family_affine_table(inv)
    targets = allocate_organ_ebpw(
        policy, inv["organ_params"], TARGET_ORGAN_EBPW, deficits)
    # LOCAL is the same tilt as the global arm -- most structure compressed
    # first -- read off a different ordering. Only INVERTED reverses it.
    monotone_up = policy in ("ORGAN_WEIGHTED", "LOCAL")
    best: tuple[float, float, list[tuple[str, tuple[int, int]]]] | None = None

    def rec(i: int, last_e: float | None, chosen: list, tot: int) -> None:
        nonlocal best
        if i == len(names):
            gap = abs(tot - target_bytes)
            dist = 0.0
            for fam, cfg in chosen:
                ebpw = table[fam][cfg][1]
                dist += inv["organ_params"][fam] * (ebpw - targets[fam]) ** 2
            cand = (float(gap), dist, list(chosen))
            if best is None or cand[:2] < best[:2]:
                best = cand
            return
        fam = names[i]
        for cfg in AFFINE_CFGS:
            ebpw = table[fam][cfg][1]
            if last_e is not None:
                if monotone_up and ebpw + 1e-9 < last_e:
                    continue
                if (not monotone_up) and ebpw - 1e-9 > last_e:
                    continue
            rec(i + 1, ebpw, chosen + [(fam, cfg)], tot + table[fam][cfg][0])

    rec(0, None, [], 0)
    if best is None:
        raise RuntimeError(f"no monotone affine assignment for {policy}")
    return {fam: cfg for fam, cfg in best[2]}


def read_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        n = struct.unpack("<Q", raw)[0]
        return {k: v for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"}


def inventory(snapshot: str) -> dict[str, Any]:
    shards = sorted(Path(snapshot).glob("**/*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {snapshot}")
    tensors: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    source_params = 0
    remainder_bytes = 0
    organ_params: dict[str, int] = {n: 0 for n in PRIOR_DEFICIT}
    shard_of: dict[str, str] = {}
    for shard in shards:
        hdr = read_header(str(shard))
        for key, entry in hdr.items():
            shape = [int(x) for x in entry["shape"]]
            n = _n_elem(shape)
            source_params += n
            fam = organ_family(key)
            dtype = str(entry.get("dtype") or "BF16")
            shard_of[key] = str(shard)
            if fam is not None and len(shape) == 2:
                tensors.append({
                    "key": key, "family": fam, "m": shape[0], "n": shape[1],
                    "params": n, "dtype": dtype,
                    "data_offsets": list(entry["data_offsets"]),
                    "shard": str(shard),
                })
                organ_params[fam] += n
            else:
                b = n * _dtype_bytes(dtype)
                remainder.append({"key": key, "shape": shape, "params": n,
                                  "dtype": dtype, "bytes": b})
                remainder_bytes += b
    tensors.sort(key=lambda t: t["key"])
    return {
        "snapshot": snapshot,
        "shards": [str(s) for s in shards],
        "tensors": tensors,
        "remainder": remainder,
        "organ_params": organ_params,
        "organ_params_total": sum(organ_params.values()),
        "remainder_bytes": remainder_bytes,
        "remainder_params": sum(r["params"] for r in remainder),
        "source_params": source_params,
        "n_organs": len(tensors),
        "shard_of": shard_of,
    }


def plan_arm(
    inv: dict[str, Any],
    policy: str,
    target_bytes: int | None = None,
    deficits: dict[str, float] | None = None,
    assignment: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """One arm's plan. `assignment` overrides the policy search entirely.

    The override exists because the monotone search is CONFOUNDED as a test of
    ordering: each policy gets whatever bit-depth multiset best hits the byte
    budget under its own monotonicity constraint, and those multisets differ
    between arms. Measured on Qwen3-0.6B, minimum bit depth separates the
    perplexities almost perfectly -- every arm that reached q2 landed at 6.76 to
    12.84, every arm that stopped at q3 or above landed at 5.23 to 6.17 -- so a
    difference between two policy arms is mostly a difference in how deep the
    ramp went, not in which organ sat where. An explicit assignment lets a
    caller hold the multiset fixed and vary only the organ it is attached to.
    """
    organ_ebpw = allocate_organ_ebpw(
        policy if policy in POLICIES else "UNIFORM",
        inv["organ_params"], TARGET_ORGAN_EBPW,
        deficits if policy in POLICIES else None)
    if assignment is not None:
        assignment = {fam: tuple(cfg) for fam, cfg in assignment.items()}
        missing = sorted(set(inv["organ_params"]) - set(assignment))
        if missing:
            raise ValueError(f"explicit assignment is missing {missing}")
    elif target_bytes is None:
        # UNIFORM defines the budget.
        assignment = {fam: UNIFORM_AFFINE for fam in inv["organ_params"]}
    else:
        assignment = search_affine_assignment(policy, inv, target_bytes, deficits)
    specs: dict[str, tuple[int, int]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    parts_sum = None
    organ_bytes = 0
    for t in inv["tensors"]:
        bits, group = assignment[t["family"]]
        specs[t["key"]] = (bits, group)
        parts, complete, _ebpw = affine_bill(t["m"], t["n"], bits, group)
        organ_bytes += complete
        if parts_sum is None:
            parts_sum = {k: 0 for k in parts}
        for k, v in parts.items():
            parts_sum[k] += int(v)
        fam = t["family"]
        slot = by_family.setdefault(fam, {
            "n_tensors": 0, "source_params": 0, "complete_bytes": 0,
            "bits": bits, "group": group, "target_ebpw": organ_ebpw[fam],
        })
        slot["n_tensors"] += 1
        slot["source_params"] += t["params"]
        slot["complete_bytes"] += complete
    for fam, slot in by_family.items():
        slot["complete_ebpw"] = ebpw4(slot["complete_bytes"], slot["source_params"])
        slot["spec"] = f"q{slot['bits']}-g{slot['group']}"
    complete_bytes = organ_bytes + int(inv["remainder_bytes"])
    src = int(inv["source_params"])
    return {
        "policy": policy,
        "target_mean_organ_ebpw": TARGET_ORGAN_EBPW,
        "organ_target_ebpw": {k: round(v, 6) for k, v in organ_ebpw.items()},
        "per_organ": by_family,
        "organ_bytes": organ_bytes,
        "remainder_bytes": int(inv["remainder_bytes"]),
        "complete_bytes": complete_bytes,
        "source_params": src,
        "complete_ebpw": ebpw4(complete_bytes, src),
        "complete_ebpw_full": ebpw_of(complete_bytes, src),
        "parts": parts_sum,
        "specs": specs,
        "assignment": {fam: list(cfg) for fam, cfg in assignment.items()},
        # A mirror arm carries an explicit assignment and no policy weights;
        # reporting the UNIFORM ramp for it would be a fabricated field.
        "per_param_weights": (per_param_weights(policy, deficits)
                              if policy in POLICIES else None),
    }


# Organs with IDENTICAL parameter counts. Exchanging two specs inside one of
# these groups changes NO bytes and NO bit-depth multiset -- only which organ is
# protected. That is the one comparison the monotone search cannot make.
# Measured on Qwen--Qwen3-0.6B: {q,o} 58,720,256 each; {k,v} 29,360,128 each;
# {down,gate,up} 88,080,384 each.
def equal_size_groups(inv: dict[str, Any]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for fam, n in inv["organ_params"].items():
        groups.setdefault(int(n), []).append(fam)
    return {n: sorted(v) for n, v in groups.items() if len(v) > 1}


def mirror_pairs(inv: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for _n, fams in sorted(equal_size_groups(inv).items()):
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                out.append((fams[i], fams[j]))
    return out


def plan_mirror_arms(
    inv: dict[str, Any],
    cheap: tuple[int, int] = (3, 64),
    rich: tuple[int, int] = (8, 64),
    base: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """BASE plus, for each equal-size pair, the two mirrored assignments.

    For pair (A, B): one arm gives A the cheap spec and B the rich spec, the
    other gives A the rich spec and B the cheap one. Because A and B hold the
    same number of parameters, the two arms are byte-identical and share a
    bit-depth multiset. Any capability difference between them is attributable
    to the organ, and to nothing else -- no budget difference, no depth
    difference, no ordering heuristic in between.
    """
    base = base or UNIFORM_AFFINE
    arms: dict[str, Any] = {}
    base_assign = {fam: base for fam in inv["organ_params"]}
    arms["BASE"] = plan_arm(inv, "UNIFORM", assignment=base_assign)
    pairs = mirror_pairs(inv)
    for a, b in pairs:
        # Key by the PAIR, not the organ: down/gate/up each sit in two pairs, so
        # keying by organ silently overwrote arms and dropped 3 of 10.
        for first, second, tag in ((a, b, f"{a}|{b}:{a}_cheap"),
                                   (b, a, f"{a}|{b}:{b}_cheap")):
            assign = dict(base_assign)
            assign[first] = cheap
            assign[second] = rich
            arms[tag] = plan_arm(inv, tag, assignment=assign)
            arms[tag]["pair"] = [a, b]
            arms[tag]["cheap_organ"] = first
            arms[tag]["rich_organ"] = second
    arms["_pairs"] = [list(p) for p in pairs]
    arms["_cheap"] = list(cheap)
    arms["_rich"] = list(rich)
    return arms


def plan_three_arms(
    inv: dict[str, Any],
    target_mean_ebpw: float = TARGET_ORGAN_EBPW,
    slug: str | None = None,
) -> dict[str, Any]:
    """One plan per POLICIES entry, all at the byte budget UNIFORM defines.

    G017 asked UNIFORM vs GLOBAL-PRIOR with INVERTED as the control. G021 adds
    LOCAL -- the same tilt read off this specimen's own measured deficits --
    because the global prior and both measured bodies disagree on q/k and
    gate/down. Kept the name; the arm count is POLICIES, not the title.
    """
    del target_mean_ebpw  # budget is UNIFORM_AFFINE, not a free mean
    uniform = plan_arm(inv, "UNIFORM")
    target = int(uniform["organ_bytes"])
    out: dict[str, Any] = {"UNIFORM": uniform, "matched_organ_bytes": target}
    for policy in POLICIES:
        if policy == "UNIFORM":
            continue
        out[policy] = plan_arm(
            inv, policy, target_bytes=target, deficits=deficits_for(policy, slug))
    residual = abs(out["ORGAN_WEIGHTED"]["complete_ebpw_full"] - uniform["complete_ebpw_full"])
    residual_inv = abs(out["INVERTED"]["complete_ebpw_full"] - uniform["complete_ebpw_full"])
    out["ebpw_residual_weighted_vs_uniform"] = residual
    out["ebpw_residual_inverted_vs_uniform"] = residual_inv
    out["ebpw_residual_local_vs_uniform"] = abs(
        out["LOCAL"]["complete_ebpw_full"] - uniform["complete_ebpw_full"])
    return out


# ---------------------------------------------------------------------------
# mutation check — revert the order line, confirm a probe FAILS, restore
# ---------------------------------------------------------------------------

_PROBE = r"""
import sys
sys.path.insert(0, %r)
from organ_allocation import allocate_organ_ebpw, PRIOR_DEFICIT, order_organs_by_deficit
params = {n: 1000 for n in PRIOR_DEFICIT}
ebpw = allocate_organ_ebpw("ORGAN_WEIGHTED", params, 8.0)
order = order_organs_by_deficit()
print("ORDER", ",".join(order))
print("Q", ebpw["q_proj"], "UP", ebpw["up_proj"])
assert ebpw["up_proj"] > ebpw["q_proj"], (ebpw["up_proj"], ebpw["q_proj"])
print("PROBE_OK")
"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_mutation_check() -> dict[str, Any]:
    """Revert the deficit-descending sort, confirm the allocator probe FAILS, restore.

    A mutation that did not apply is not a mutation test: we print the anchor
    line and a before/after sha256, and refuse to report success unless the
    hash actually moved and the probe actually failed, then moved back.
    """
    path = Path(__file__).resolve()
    original = path.read_bytes()
    before = _sha256_bytes(original)
    text = original.decode("utf-8")
    # Assembled so this helper does not itself contain the unique live line.
    _tag = "MUTATION_ANCHOR_ORDER_BY_DEFICIT"
    # The anchor moved from PRIOR_DEFICIT to `src` when order_organs_by_deficit
    # learned to take a specimen's OWN deficits for the LOCAL policy. Both this
    # checker and the test pinned the old text, so the live negative control
    # would have raised "anchor not unique: count=0" and the ordering would have
    # had NO mutation test at all -- the guard drifting off its target, which
    # this campaign has already been bitten by once.
    old = "key=lambda name: -" + "src[name]))  # " + _tag
    new = "key=lambda name: " + "src[name]))  # " + _tag
    if text.count(old) != 1:
        raise RuntimeError(
            f"mutation anchor not unique: count={text.count(old)} for {old!r}"
        )
    anchor_line = next(ln for ln in text.splitlines() if old in ln)
    mutated_text = text.replace(old, new, 1)
    mutated = mutated_text.encode("utf-8")
    after_mut = _sha256_bytes(mutated)
    if after_mut == before:
        raise RuntimeError("mutation did not apply: sha256 unchanged")
    probe = _PROBE % str(_HERE)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_HERE)
    try:
        path.write_bytes(mutated)
        applied = _sha256_bytes(path.read_bytes())
        if applied != after_mut:
            raise RuntimeError("mutated bytes did not land on disk")
        failed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        fail_ok = failed.returncode != 0 and "PROBE_OK" not in failed.stdout
        if not fail_ok:
            raise RuntimeError(
                "mutated allocator did not fail the probe: "
                f"rc={failed.returncode} stdout={failed.stdout!r} stderr={failed.stderr!r}"
            )
    finally:
        path.write_bytes(original)
    restored = _sha256_bytes(path.read_bytes())
    if restored != before:
        raise RuntimeError("restore failed: sha256 does not match the pre-mutation file")
    live = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if live.returncode != 0 or "PROBE_OK" not in live.stdout:
        raise RuntimeError(
            f"restored allocator failed its own probe: rc={live.returncode} "
            f"stdout={live.stdout!r} stderr={live.stderr!r}"
        )
    return {
        "anchor_line": anchor_line.strip(),
        "old": old,
        "new": new,
        "sha256_before": before,
        "sha256_mutated": after_mut,
        "sha256_restored": restored,
        "mutation_applied": after_mut != before,
        "mutated_probe_returncode": failed.returncode,
        "mutated_probe_failed_as_required": True,
        "mutated_probe_stdout": failed.stdout[-500:],
        "mutated_probe_stderr": failed.stderr[-500:],
        "restored_probe_returncode": live.returncode,
        "restored_probe_stdout": live.stdout[-500:],
        "no_mutation_left_live": restored == before and old in path.read_text(),
    }


# ---------------------------------------------------------------------------
# payload load / reconstruct (NR rematerializes dense W_hat)
# ---------------------------------------------------------------------------

def _load_bf16_payloads(inv: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    nr = _nr()
    out: dict[str, Any] = {}
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for t in inv["tensors"]:
        by_shard.setdefault(t["shard"], []).append(t)
    for shard, items in by_shard.items():
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            payload_off = 8 + n
            for t in items:
                e = hdr[t["key"]]
                start, end = e["data_offsets"]
                fh.seek(payload_off + int(start))
                raw = fh.read(int(end) - int(start))
                u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
                W = np.ascontiguousarray(arr.reshape(t["m"], t["n"]), dtype=np.float32)
                out[t["key"]] = W
        # round-trip through the same bf16 decoder lowrank_nr uses, so the
        # parent the NR sees is the parent on disk, not a float32 promotion.
        for t in items:
            out[t["key"]] = nr.round_bf16(out[t["key"]])
    return out


def reconstruct_arm(
    inv: dict[str, Any],
    weights: dict[str, Any],
    specs: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Rematerialize W_hat via encode_affine. Accounting is billed separately."""
    nr = _nr()
    recs: dict[str, Any] = {}
    for t in inv["tensors"]:
        bits, group = specs[t["key"]]
        arm = nr.encode_affine(weights[t["key"]], int(bits), int(group))
        recs[t["key"]] = arm.reconstruction
    return recs


# ---------------------------------------------------------------------------
# capability: existing conjunction, this specimen's dense parent as reference
# ---------------------------------------------------------------------------

def _fourgram_repeat(token_ids: list[int]) -> float:
    grams = [tuple(token_ids[i:i + 4]) for i in range(max(0, len(token_ids) - 3))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _distinct_ratio(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def _max_run(token_ids: list[int]) -> int:
    run = cur = 1
    for i in range(1, len(token_ids)):
        cur = cur + 1 if token_ids[i] == token_ids[i - 1] else 1
        run = max(run, cur)
    return run


def measure_capability(model, tokenizer, *, device: str = "cpu") -> dict[str, Any]:
    """Same axes as gravity_outlier_eval.evaluate: NLL ppl + greedy 4-gram r4."""
    import torch
    model.eval()
    corpus = CORPORA[ACTIVE_CORPUS]
    nll_ids = tokenizer.encode(corpus["nll"])[:NLL_TOKENS]
    if len(nll_ids) < 2:
        raise RuntimeError("NLL text tokenised to <2 tokens")
    x = torch.tensor([nll_ids], device=device)
    with torch.inference_mode():
        logits = model(x[:, :-1]).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        nll = float(-logp.gather(-1, x[:, 1:, None]).squeeze(-1).mean())
    ppl = float(2.718281828459045 ** nll)

    r4s: list[float] = []
    dis: list[float] = []
    worst = 1
    generations: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for prompt in corpus["prompts"]:
        ids = tokenizer(prompt, return_tensors="pt")
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.inference_mode():
            out = model.generate(
                **ids,
                max_new_tokens=GEN_TOKENS,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
        tt = out[0].tolist()
        r4s.append(_fourgram_repeat(tt))
        dis.append(_distinct_ratio(tt))
        worst = max(worst, _max_run(tt))
        generations.append({
            "prompt": prompt,
            "n_tokens": len(tt),
            "n_new": len(tt) - int(ids["input_ids"].shape[1]),
            "r4": round(r4s[-1], 6),
            "distinct": round(dis[-1], 6),
            "text": tokenizer.decode(tt, skip_special_tokens=True)[:240],
        })
    r4s.sort()
    dis.sort()
    med_r4 = r4s[len(r4s) // 2]
    med_dis = dis[len(dis) // 2]
    return {
        "ppl": round(ppl, 4),
        "nll": round(nll, 5),
        "median_4gram_repeat": round(med_r4, 4),
        "median_distinct_ratio": round(med_dis, 4),
        "worst_max_repeat_run": int(worst),
        "n_prompts": len(corpus["prompts"]),
        "corpus": ACTIVE_CORPUS,
        "corpus_note": corpus["note"],
        "nll_tokens": len(nll_ids),
        "gen_tokens": GEN_TOKENS,
        "r4s": [round(x, 6) for x in r4s],
        "distincts": [round(x, 6) for x in dis],
        "generations": generations,
        "ppl_full": ppl,
        "r4_full": med_r4,
    }


# median_4gram_repeat is a FRACTION of 4-grams that repeat, so its range is
# [0, 1]. `R4_MULT x dense` is unbounded: for any parent repeating more than a
# third of its 4-grams the bar lands ABOVE the highest value the metric can take,
# and the diversity axis cannot reject anything. Qwen3-0.6B's dense median is
# 0.3619, its bar was 1.0857, and G017_ORGAN_ALLOCATION.json records the
# deliberately INVERTED control -- 82% of 4-grams repeating -- as
# diversity_ok: TRUE. The conjunction had quietly become perplexity alone, which
# is the single-proxy failure the gate exists to prevent.
#
# THE CAP, and where it comes from: a bar may not sit more than halfway from the
# reference to total collapse. Both fixed points already belong to the problem --
# the parent the specimen is judged against, and the ceiling the metric cannot
# exceed -- so this is derived, not chosen to produce a wanted verdict. For any
# parent below dense_r4 = 1/3 it changes NOTHING: O003's bar stays 0.1614 and
# every verdict in the capability cliff is untouched.
R4_CEILING = 1.0


def r4_bar(dense_r4: float) -> float:
    """The diversity bar, capped so it stays inside the metric it measures."""
    return min(R4_MULT * float(dense_r4), (float(dense_r4) + R4_CEILING) / 2.0)


def gate_from_reference(cap: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    r4_max = r4_bar(float(ref["r4_full"]))
    ppl_max = PPL_MULT * float(ref["ppl_full"])
    ppl_ok = float(cap["ppl_full"]) <= ppl_max
    r4_ok = float(cap["r4_full"]) <= r4_max
    ok = bool(ppl_ok and r4_ok)
    return {
        "capability_ok": ok,
        "capability_status": "CANDIDATE_PASS" if ok else "CAPABILITY_LOSS",
        "ppl_ok": ppl_ok,
        "diversity_ok": r4_ok,
        "conjunction": "perplexity AND n-gram diversity",
        "gate_r4_max": round(r4_max, 4),
        "gate_ppl_max": round(ppl_max, 4),
        "gate_r4_mult": R4_MULT,
        "gate_r4_capped": bool(R4_MULT * float(ref["r4_full"]) > r4_max),
        "gate_r4_uncapped": round(R4_MULT * float(ref["r4_full"]), 4),
        "gate_r4_cap_rule": "min(mult x dense, midpoint(dense, 1.0)) -- a bar may not exceed the metric's range",
        "gate_ppl_mult": PPL_MULT,
        "evaluator": "tools/future/organ_allocation.py::measure_capability",
        "evaluator_copied_from": "tools/future/gravity_outlier_eval.py::evaluate",
        "reference": "specimen dense parent (bf16 values, fp32 compute)",
    }


def _apply_recs(model, recs: dict[str, Any]) -> int:
    import torch
    n = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in recs:
                param.copy_(torch.from_numpy(recs[name]))
                n += 1
    return n


def _load_model(snapshot: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def _public_arm(plan: dict[str, Any], cap: dict[str, Any] | None, gate: dict[str, Any] | None) -> dict[str, Any]:
    out = {
        "policy": plan["policy"],
        "complete_ebpw": plan["complete_ebpw"],
        "complete_ebpw_full": plan["complete_ebpw_full"],
        "complete_bytes": plan["complete_bytes"],
        "organ_bytes": plan["organ_bytes"],
        "remainder_bytes": plan["remainder_bytes"],
        "source_params": plan["source_params"],
        "organ_target_ebpw": plan["organ_target_ebpw"],
        "per_organ": plan["per_organ"],
        "parts": plan["parts"],
        "per_param_weights": plan["per_param_weights"],
        "representation": "affine NR via lowrank_nr.encode_affine; UNIFORM is q4g64; "
                          "ORGAN_WEIGHTED/INVERTED are monotone bit/group assignments "
                          "along the prior order at the same complete organ bytes",
        "execution_complete": False,
        "execution_note": "NR rematerializes dense W_hat; there is no low-rank kernel",
    }
    # A mirror arm declares WHICH comparison it belongs to. Without this the
    # receipt reads as one eleven-way comparison and an auditor correctly calls
    # a sound design confounded, because arms from different pairs hold organs
    # of different sizes and are not meant to be compared at all.
    for key in ("pair", "cheap_organ", "rich_organ"):
        if plan.get(key) is not None:
            out[key] = plan[key]
    if cap is not None:
        out["capability"] = {
            "ppl": cap["ppl"],
            "nll": cap["nll"],
            "median_4gram_repeat": cap["median_4gram_repeat"],
            "median_distinct_ratio": cap["median_distinct_ratio"],
            "worst_max_repeat_run": cap["worst_max_repeat_run"],
            "n_prompts": cap["n_prompts"],
            "nll_tokens": cap["nll_tokens"],
            "gen_tokens": cap["gen_tokens"],
            "r4s": cap["r4s"],
            "distincts": cap["distincts"],
            "generations": cap["generations"],
            "ppl_full": cap["ppl_full"],
            "r4_full": cap["r4_full"],
        }
    if gate is not None:
        out["capability_ok"] = gate["capability_ok"]
        out["capability_status"] = gate["capability_status"]
        out["ppl_ok"] = gate["ppl_ok"]
        out["diversity_ok"] = gate["diversity_ok"]
        out["gate"] = gate
    return out


def _pareto_better(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Lower ppl is better, lower r4 is better. mixed / tie / a / b."""
    pa, ra = a["capability"]["ppl_full"], a["capability"]["r4_full"]
    pb, rb = b["capability"]["ppl_full"], b["capability"]["r4_full"]
    ppl_eps = 1e-4
    r4_eps = 1e-4
    ppl_a = pa < pb - ppl_eps
    ppl_b = pb < pa - ppl_eps
    r4_a = ra < rb - r4_eps
    r4_b = rb < ra - r4_eps
    a_not_worse = (not ppl_b) and (not r4_b)
    b_not_worse = (not ppl_a) and (not r4_a)
    a_better_some = ppl_a or r4_a
    b_better_some = ppl_b or r4_b
    if a_better_some and a_not_worse:
        return "a"
    if b_better_some and b_not_worse:
        return "b"
    if (not a_better_some) and (not b_better_some):
        return "tie"
    return "mixed"


def _g021_verdict(arms: dict[str, Any], orders: dict[str, list]) -> dict[str, Any]:
    """UNIFORM vs GLOBAL-PRIOR vs LOCAL-ANATOMY at matched complete EBPW. [S010 11]

    Three arms, one question: does allocating by THIS body's own measured
    anatomy preserve capability better than allocating by the campaign-wide
    prior, or better than not allocating at all? INVERTED stays as the global
    arm's control and is reported, not compared here.

    Capability decides first. Any arm that fails the conjunction gate cannot
    win on continuous axes alone -- that is the G017 lesson, and dropping it
    would let a less-collapsed-but-broken arm be called a winner.
    """
    names = ("UNIFORM", "ORGAN_WEIGHTED", "LOCAL")
    pairs = {
        "LOCAL_vs_UNIFORM": _pareto_better(arms["LOCAL"], arms["UNIFORM"]),
        "LOCAL_vs_GLOBAL": _pareto_better(arms["LOCAL"], arms["ORGAN_WEIGHTED"]),
        "GLOBAL_vs_UNIFORM": _pareto_better(arms["ORGAN_WEIGHTED"], arms["UNIFORM"]),
    }
    passing = [n for n in names if bool(arms[n].get("capability_ok"))]
    # A winner must pass the gate AND not be Pareto-dominated by another passer.
    winners = []
    for n in passing:
        dominated = any(
            m != n and _pareto_better(arms[m], arms[n]) == "a" for m in passing)
        if not dominated:
            winners.append(n)
    if len(winners) == 1:
        outcome = winners[0]
    elif not passing:
        outcome = "NONE_PASSES_GATE"
    else:
        outcome = "UNDECIDED"
    rows = {
        n: {
            "ppl": arms[n]["capability"]["ppl_full"],
            "r4": arms[n]["capability"]["r4_full"],
            "capability_ok": bool(arms[n].get("capability_ok")),
            "complete_ebpw": arms[n]["complete_ebpw"],
        }
        for n in names
    }
    if outcome == "LOCAL":
        prior_note = (
            "LOCAL anatomy is the stronger allocation prior on this body; "
            "promote local measurement over the global median where measured")
    elif outcome == "ORGAN_WEIGHTED":
        prior_note = (
            "the GLOBAL prior beat this body's own anatomy; the local "
            "structural metric does not map directly to compressibility")
    elif outcome == "UNIFORM":
        prior_note = (
            "structure-only allocation is too weak on this body: neither "
            "ordering beat spending the same bytes equally")
    elif outcome == "NONE_PASSES_GATE":
        prior_note = (
            "no arm survived the conjunction gate at this budget; the "
            "comparison is unresolved, not a UNIFORM win")
    else:
        prior_note = (
            "two or more arms pass the gate and none Pareto-dominates the "
            "others; the discriminator did not separate them at this budget")
    sentence = (
        f"G021 {outcome}: at matched complete EBPW "
        f"{arms['UNIFORM']['complete_ebpw']:.4f}, "
        + "; ".join(
            f"{n} ppl {rows[n]['ppl']:.4f} r4 {rows[n]['r4']:.4f} "
            f"gate {'PASS' if rows[n]['capability_ok'] else 'FAIL'}"
            for n in names)
        + f". LOCAL order {','.join(orders.get('local') or [])} vs GLOBAL "
        f"{','.join(orders.get('global') or [])}. {prior_note}."
    )
    return {
        "question": "UNIFORM vs GLOBAL-PRIOR vs LOCAL-ANATOMY at matched complete EBPW",
        "outcome": outcome,
        "winners": winners,
        "passing_gate": passing,
        "pareto": pairs,
        "arms": rows,
        "orders": orders,
        "prior_update": prior_note,
        "inverted_control": {
            "ppl": arms["INVERTED"]["capability"]["ppl_full"],
            "r4": arms["INVERTED"]["capability"]["r4_full"],
            "capability_ok": bool(arms["INVERTED"].get("capability_ok")),
        },
        "verdict_sentence": sentence,
    }


def _verdict(arms: dict[str, Any]) -> dict[str, Any]:
    uni, ow, inv = arms["UNIFORM"], arms["ORGAN_WEIGHTED"], arms["INVERTED"]
    vs_uni = _pareto_better(ow, uni)
    vs_inv = _pareto_better(ow, inv)
    ppl_delta = uni["capability"]["ppl_full"] - ow["capability"]["ppl_full"]
    r4_delta = uni["capability"]["r4_full"] - ow["capability"]["r4_full"]
    # Transfer: ORGAN_WEIGHTED must beat UNIFORM, and INVERTED must not win or tie it.
    transferred = vs_uni == "a" and vs_inv == "a" and bool(ow.get("capability_ok"))
    if vs_inv in ("b", "tie"):
        transferred = False
        transfer_note = (
            "INVERTED won or tied ORGAN_WEIGHTED; the prior did not transfer"
        )
    elif vs_uni != "a":
        transfer_note = (
            "ORGAN_WEIGHTED did not Pareto-beat UNIFORM at matched EBPW"
        )
    elif not ow.get("capability_ok"):
        transferred = False
        transfer_note = (
            "ORGAN_WEIGHTED is less collapsed on the continuous axes but fails "
            "the conjunction gate; that is not a capability win"
        )
    else:
        transfer_note = (
            "ORGAN_WEIGHTED Pareto-beats UNIFORM, beats INVERTED, and passes "
            "the conjunction; the prior transferred"
        )
    beat_uniform = vs_uni == "a" and bool(ow.get("capability_ok"))
    sentence = (
        f"{'YES' if beat_uniform else 'NO'}: unequal ORGAN_WEIGHTED "
        f"{'beats' if beat_uniform else 'does not beat'} UNIFORM at matched "
        f"complete EBPW {uni['complete_ebpw']:.4f} — ppl "
        f"{ow['capability']['ppl']:.4f} vs {uni['capability']['ppl']:.4f} "
        f"(delta {ppl_delta:+.4f}, negative means weighted worse), "
        f"median_4gram_repeat {ow['capability']['median_4gram_repeat']:.4f} vs "
        f"{uni['capability']['median_4gram_repeat']:.4f} (delta {r4_delta:+.4f}). "
        f"INVERTED ppl {inv['capability']['ppl']:.4f} r4 "
        f"{inv['capability']['median_4gram_repeat']:.4f} "
        f"({transfer_note})."
    )
    return {
        "unequal_beats_uniform": beat_uniform,
        "pareto_weighted_vs_uniform": vs_uni,
        "pareto_weighted_vs_inverted": vs_inv,
        "prior_transferred": transferred,
        "ppl_delta_uniform_minus_weighted": round(ppl_delta, 6),
        "r4_delta_uniform_minus_weighted": round(r4_delta, 6),
        "transfer_note": transfer_note,
        "sentence": sentence,
    }


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def _local_ordering_from_sweep(slug: str | None = None) -> dict[str, Any] | None:
    p = _REPO / SWEEP_REL
    if not p.is_file():
        return None
    want = slug or SPECIMEN_SLUG
    sweep = json.loads(p.read_text())
    for row in sweep.get("rows") or []:
        if row.get("slug") == want:
            return {
                "slug": row["slug"],
                "gib": row.get("gib"),
                "family": row.get("family"),
                "n_layers": row.get("n_layers"),
                "organ_ordering": row.get("organ_ordering"),
                "wall_s": row.get("wall_s"),
                "peak_rss_gib": row.get("peak_rss_gib"),
            }
    return None


def run_experiment(
    *,
    skip_capability: bool = False,
    snapshot: str | None = None,
    slug: str | None = None,
    why: str | None = None,
    expected_gb: float = 8.0,
    skip_mutation: bool = False,
) -> dict[str, Any]:
    from campaign_memory_guard import require_ok, resource_cost, sample

    snapshot = snapshot or SPECIMEN_PATH
    slug = slug or SPECIMEN_SLUG
    why = why or WHY_THIS_SPECIMEN
    t0 = time.perf_counter()
    guard = sample(expected_gb=expected_gb)
    if guard.state == "STOP":
        return {
            "schema": SCHEMA,
            "obligation": "G017",
            "status": "GUARD_STOP",
            "specimen": slug,
            "guard": guard.as_dict(),
            "why_this_specimen": why,
            "verdict_sentence": (
                "NO RESULT: campaign_memory_guard STOP before load; "
                f"{'; '.join(guard.reasons)}"
            ),
        }
    require_ok("G017 organ allocation", expected_gb=expected_gb)

    inv = inventory(snapshot)
    plans = plan_three_arms(inv, TARGET_ORGAN_EBPW, slug)
    mutation = None if skip_mutation else run_mutation_check()
    sweep_row = _local_ordering_from_sweep(slug)
    local_order = [r["organ"] for r in (sweep_row or {}).get("organ_ordering") or []]
    global_order = list(order_organs_by_deficit())

    dense_cap = None
    arm_caps: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    applied = {}
    with resource_cost(interval_s=2.0, label="G017") as cost:
        if skip_capability:
            cost["skipped"] = True
        else:
            weights = _load_bf16_payloads(inv)
            recs_by_arm = {}
            for name in POLICIES:
                recs_by_arm[name] = reconstruct_arm(inv, weights, plans[name]["specs"])
            del weights
            model, tok = _load_model(snapshot)
            n_named = sum(1 for n, _ in model.named_parameters() if n in recs_by_arm["UNIFORM"])
            applied["named_parameters_matched"] = n_named
            applied["n_organ_tensors"] = inv["n_organs"]
            if n_named != inv["n_organs"]:
                raise RuntimeError(
                    f"HF named_parameters matched {n_named} of {inv['n_organs']} organs"
                )
            dense_cap = measure_capability(model, tok)
            for name in POLICIES:
                n_set = _apply_recs(model, recs_by_arm[name])
                applied[name] = n_set
                arm_caps[name] = measure_capability(model, tok)
                gates[name] = gate_from_reference(arm_caps[name], dense_cap)
            del model
            del recs_by_arm

    arms_pub = {}
    for name in POLICIES:
        plan = {k: v for k, v in plans[name].items() if k != "specs"}
        arms_pub[name] = _public_arm(plan, arm_caps.get(name), gates.get(name))

    verdict = None
    g021 = None
    if arm_caps:
        verdict = _verdict(arms_pub)
        g021 = _g021_verdict(
            arms_pub, {"local": local_order, "global": global_order})

    residual = plans["ebpw_residual_weighted_vs_uniform"]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "obligation": "G017",
        "status": "MEASURED" if arm_caps else "ALLOC_ONLY",
        "specimen": slug,
        "specimen_path": snapshot,
        "why_this_specimen": why,
        "prior_source": PRIOR_REL,
        "prior_deficit": dict(PRIOR_DEFICIT),
        "prior_order": global_order,
        "local_order": local_order,
        "local_matches_100pct_laws": {
            "o_proj > up_proj": True,
            "q_proj > up_proj": True,
            "gate_proj > up_proj": True,
        },
        "local_vs_global": (
            "local k>q (global q>k) and local down>gate (global gate>down); "
            "the three 100% laws hold. G021 runs BOTH orderings as arms: "
            "ORGAN_WEIGHTED is the global prior, LOCAL is this body's own "
            "measured deficits, at the same matched byte budget"
        ),
        "local_deficit": local_deficits(slug) if local_order else None,
        "sweep_row": sweep_row,
        "target_mean_organ_ebpw": TARGET_ORGAN_EBPW,
        "uniform_spec": {"bits": UNIFORM_AFFINE[0], "group": UNIFORM_AFFINE[1],
                         "note": "UNIFORM defines the matched byte budget; every "
                                 "arm is held to its organ_bytes"},
        "source_params": inv["source_params"],
        "organ_params": inv["organ_params"],
        "n_organ_tensors": inv["n_organs"],
        "remainder_params": inv["remainder_params"],
        "remainder_bytes": inv["remainder_bytes"],
        "accounting": {
            "definition": "complete_ebpw = stored_bytes * 8 / source_parameters",
            "source_parameters": "every persistent parameter tensor in the parent "
                                 "safetensors (embed and lm_head both stored despite "
                                 "tie_word_embeddings; both billed)",
            "numerator": "affine packed payload + fp16 scale + fp16 zero-point + "
                         "24-byte metadata per organ + dense bf16 remainder; "
                         "scales live in the numerator",
            "accountant": "tools/future/lowrank_nr.py::finalize_parts / encode_affine",
            "matched_organ_bytes": plans["matched_organ_bytes"],
            "ebpw_residual_weighted_vs_uniform": residual,
            "ebpw_residual_inverted_vs_uniform": plans["ebpw_residual_inverted_vs_uniform"],
            "ebpw_residual_weighted_vs_uniform_4dp": round(residual, 4),
        },
        "dense_parent_capability": dense_cap,
        "arms": arms_pub,
        "matched_ebpw": {
            **{n: arms_pub[n]["complete_ebpw"] for n in POLICIES},
            "residual_weighted_vs_uniform": round(residual, 6),
            "residual_local_vs_uniform": round(
                plans["ebpw_residual_local_vs_uniform"], 6),
        },
        "g021_verdict": g021,
        "mutation_check": mutation,
        "applied": applied,
        "guard_at_start": guard.as_dict(),
        "resource": cost,
        "device": "cpu",
        "metal": "unused; mlx_lm refused Metal in this session, torch MPS unavailable",
        "nr_not_nx": True,
        "weaker_than_it_looks": [
            "Within-tensor low-rank was tried first and abandoned as a "
            "discriminator: square k/v break-even is 16 EBPW / 50% rank, and "
            "layer-0 attention at that rank already blows the ppl gate. Affine "
            "q4g64 is the cheapest NR that still has a live conjunction here.",
            "ORGAN_WEIGHTED/INVERTED are the closest monotone affine assignments "
            "on the q2/q3/q4/q8 × g32/64/128 grid, not a rate-distortion optimum.",
            "LOCAL and GLOBAL differ by two adjacent transpositions (k/q, "
            "down/gate) and by tilt endpoint ratio (1.505 local vs 1.745 "
            "global). That is the whole size of the G021 effect: a small "
            "reordering of a monotone ramp, not a different representation.",
            "Capability is 8 greedy prompts x 96 tokens plus 512-token NLL on "
            "CPU fp32 — the existing gate's budget, not a large eval.",
            "The parent is bf16 on disk; compute is fp32. Gate multipliers "
            "3.0 / 1.25 are gravity_outlier_eval's, calibrated here against "
            "THIS specimen's dense parent, not O003's absolute R4_MAX/PPL_MAX.",
            "NR rematerializes dense W_hat; execution_complete is False.",
            "Integer rank plus 24-byte metadata leaves a residual EBPW gap; "
            "it is reported, not hidden.",
            "Qwen3-0.6B is the cheapest dense LM, not a demonstration body; "
            "transfer to larger dense bodies is untested.",
            "Embed, lm_head and RMSNorms are dense bf16 in every arm; only "
            "the seven linear organs are allocated.",
        ],
        "wall_s_total": round(time.perf_counter() - t0, 3),
        "recorded_by": RECORDED_BY,
    }
    if verdict:
        doc["verdict"] = verdict
        doc["verdict_sentence"] = verdict["sentence"]
        doc["inverted_control"] = {
            "ran": True,
            "complete_ebpw": arms_pub["INVERTED"]["complete_ebpw"],
            "capability_ok": arms_pub["INVERTED"].get("capability_ok"),
            "ppl": arms_pub["INVERTED"]["capability"]["ppl"],
            "median_4gram_repeat": arms_pub["INVERTED"]["capability"]["median_4gram_repeat"],
            "ppl_ok": arms_pub["INVERTED"].get("ppl_ok"),
            "diversity_ok": arms_pub["INVERTED"].get("diversity_ok"),
            "vs_organ_weighted": verdict["pareto_weighted_vs_inverted"],
        }
    return doc


def write_doc(doc: dict[str, Any], name: str | None = None) -> Path:
    from _common import write_receipt
    return write_receipt(name or RECEIPT_NAME, doc, RECORDED_BY)


def selfcheck() -> dict[str, Any]:
    """Allocator invariants on the real specimen header, no payload load."""
    inv = inventory(SPECIMEN_PATH)
    plans = plan_three_arms(inv, TARGET_ORGAN_EBPW)
    uni, ow, invp = plans["UNIFORM"], plans["ORGAN_WEIGHTED"], plans["INVERTED"]
    ow_e = allocate_organ_ebpw("ORGAN_WEIGHTED", inv["organ_params"])
    inv_e = allocate_organ_ebpw("INVERTED", inv["organ_params"])
    uni_e = allocate_organ_ebpw("UNIFORM", inv["organ_params"])
    assert all(abs(uni_e[n] - TARGET_ORGAN_EBPW) < 1e-12 for n in uni_e)
    assert ow_e["up_proj"] > ow_e["q_proj"]
    assert ow_e["up_proj"] > ow_e["k_proj"]
    assert ow_e["up_proj"] > ow_e["o_proj"]
    assert inv_e["q_proj"] > inv_e["up_proj"]
    assert order_organs_by_deficit()[0] == "q_proj"
    assert order_organs_by_deficit()[-1] == "up_proj"
    assert ow["per_organ"]["up_proj"]["complete_ebpw"] > ow["per_organ"]["q_proj"]["complete_ebpw"]
    assert invp["per_organ"]["q_proj"]["complete_ebpw"] > invp["per_organ"]["up_proj"]["complete_ebpw"]
    residual = plans["ebpw_residual_weighted_vs_uniform"]
    assert residual < 1e-4, residual
    assert uni["complete_bytes"] == ow["complete_bytes"] == invp["complete_bytes"]
    mean_u = (
        sum(uni_e[n] * inv["organ_params"][n] for n in uni_e)
        / sum(inv["organ_params"].values())
    )
    assert abs(mean_u - TARGET_ORGAN_EBPW) < 1e-12
    return {
        "source_params": inv["source_params"],
        "organ_params": inv["organ_params"],
        "UNIFORM_complete_ebpw": uni["complete_ebpw"],
        "ORGAN_WEIGHTED_complete_ebpw": ow["complete_ebpw"],
        "INVERTED_complete_ebpw": invp["complete_ebpw"],
        "residual": residual,
        "organ_target_UNIFORM": uni["organ_target_ebpw"],
        "organ_target_ORGAN_WEIGHTED": ow["organ_target_ebpw"],
        "organ_target_INVERTED": invp["organ_target_ebpw"],
        "per_organ_UNIFORM": uni["per_organ"],
        "per_organ_ORGAN_WEIGHTED": ow["per_organ"],
        "per_organ_INVERTED": invp["per_organ"],
    }


def ingest_g017_into_ledger(
    ledger_path: str = "receipts/future/G034_ODYSSEY_LEDGER.json",
    receipt_path: str = "receipts/future/G017_ORGAN_ALLOCATION.json",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fold G017 into the Odyssey ledger. One NR candidate, explicitly not an NX."""
    from odyssey_ledger import measured, refused, progress
    led = json.loads(Path(ledger_path).read_text())
    rec = json.loads(Path(receipt_path).read_text())
    slug = rec["specimen"]
    row = next((r for r in led["specimens"] if r["slug"] == slug), None)
    if row is None:
        raise RuntimeError(f"{slug} is not in the ledger")
    # capability_ok is the CONJUNCTION (perplexity AND 4-gram diversity). The one
    # row written by hand recorded ppl_ok here instead, so a UNIFORM arm that
    # FAILS diversity was published as a capability pass -- the ledger's only
    # nr_candidate row contradicting its own receipt. Read the conjunction, and
    # emit matched_complete_ebpw under the name the ledger actually stores.
    value = {
        "form": "organ-unequal affine NR",
        "matched_complete_ebpw": rec["arms"]["UNIFORM"]["complete_ebpw"],
        "complete_ebpw": rec["arms"]["UNIFORM"]["complete_ebpw"],
        "unequal_beats_uniform": rec["verdict"]["unequal_beats_uniform"],
        "uniform_capability_ok": rec["arms"]["UNIFORM"]["capability_ok"],
        "organ_weighted_capability_ok": rec["arms"]["ORGAN_WEIGHTED"]["capability_ok"],
        "inverted_capability_ok": rec["arms"]["INVERTED"]["capability_ok"],
        "uniform_ppl": rec["arms"]["UNIFORM"]["capability"]["ppl"],
        "organ_weighted_ppl": rec["arms"]["ORGAN_WEIGHTED"]["capability"]["ppl"],
        "inverted_ppl": rec["arms"]["INVERTED"]["capability"]["ppl"],
        "verdict": rec["verdict_sentence"],
    }
    actions = []
    if row["axes"]["nr_candidate"]["state"] == "OWED":
        measured(row, "nr_candidate", value, receipt_path)
        actions.append("nr_candidate MEASURED")
    else:
        actions.append(f"nr_candidate already {row['axes']['nr_candidate']['state']}")
    if row["axes"]["nx_disposition"]["state"] == "OWED":
        refused(
            row,
            "nx_disposition",
            "G017 arms are NR, not NX: encode_affine rematerializes dense W_hat "
            "(execution_complete is false; there is no organ-unequal affine kernel). "
            "Reopen when a kernel consumes the packed affine codes directly at the "
            "matched 9.2619 complete EBPW without a dense parent.",
        )
        actions.append("nx_disposition REFUSED")
    else:
        actions.append(f"nx_disposition already {row['axes']['nx_disposition']['state']}")
    if not dry_run:
        Path(ledger_path).write_text(json.dumps(led, indent=1) + "\n")
    return {"slug": slug, "actions": actions, "progress": progress(led)}


ISOLATE_SCHEMA = "hawking.future.g022_group_isolation.v1"
ISOLATE_RECEIPT = "G022_GROUP_ISOLATION.json"


def run_isolation_experiment(
    *,
    snapshot: str | None = None,
    slug: str | None = None,
    expected_gb: float = 8.0,
) -> dict[str, Any]:
    """WHICH organ group makes the winning allocation win?

    GLOBAL_ANATOMY is the only arm to pass the conjunction on two corpora, but
    the mechanism is unattributed: GLOBAL and ANTI_SENSITIVITY agree on {q,o}
    and still differ by 0.21 r4, so the win is not one organ's doing. Each arm
    here changes GLOBAL in EXACTLY ONE equal-parameter group and leaves the
    other two identical, so every arm is byte-exact and depth-exact against the
    base and the capability delta is that group's contribution and nothing else.
    """
    from campaign_memory_guard import require_ok, resource_cost, sample

    snapshot = snapshot or SPECIMEN_PATH
    slug = slug or SPECIMEN_SLUG
    guard = sample(expected_gb=expected_gb)
    if guard.state == "STOP":
        return {"schema": ISOLATE_SCHEMA, "status": "GUARD_STOP",
                "guard": guard.as_dict(),
                "verdict_sentence": "NO RESULT: campaign_memory_guard STOP before load"}
    require_ok("G022 group isolation", expected_gb=expected_gb)

    C, M, R = ALLOC_CHEAP, ALLOC_MID, ALLOC_RICH
    base = {"q_proj": C, "o_proj": M, "k_proj": C, "v_proj": M,
            "gate_proj": C, "down_proj": M, "up_proj": R}
    arms: dict[str, dict[str, tuple[int, int]]] = {"GLOBAL_BASE": dict(base)}
    # One group changed per arm. Nothing else moves.
    a = dict(base); a["q_proj"], a["o_proj"] = M, C
    arms["SWAP_qo"] = a
    a = dict(base); a["k_proj"], a["v_proj"] = M, C
    arms["SWAP_kv"] = a
    a = dict(base); a["gate_proj"], a["down_proj"] = M, C     # crush down instead of gate
    arms["MLP_crush_down"] = a
    a = dict(base); a["gate_proj"], a["up_proj"] = R, C       # crush up, protect gate
    arms["MLP_crush_up"] = a

    inv = inventory(snapshot)
    plans = {n: plan_arm(inv, n, assignment=asg) for n, asg in arms.items()}
    names = list(plans)
    caps: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    applied: dict[str, Any] = {}
    with resource_cost(interval_s=2.0, label="G022_ISO") as cost:
        weights = _load_bf16_payloads(inv)
        model, tok = _load_model(snapshot)
        dense_cap = measure_capability(model, tok)
        for nm in names:
            rec = reconstruct_arm(inv, weights, plans[nm]["specs"])
            applied[nm] = _apply_recs(model, rec)
            del rec
            caps[nm] = measure_capability(model, tok)
            gates[nm] = gate_from_reference(caps[nm], dense_cap)
        del model, weights

    pub = {n: _public_arm({k: v for k, v in plans[n].items() if k != "specs"},
                          caps[n], gates[n]) for n in names}
    b_ppl = pub["GLOBAL_BASE"]["capability"]["ppl_full"]
    b_r4 = pub["GLOBAL_BASE"]["capability"]["r4_full"]
    rows = []
    for n in names:
        if n == "GLOBAL_BASE":
            continue
        changed = sorted(k for k in base if arms[n][k] != base[k])
        rows.append({
            "arm": n, "group_changed": changed,
            "delta_ppl": round(pub[n]["capability"]["ppl_full"] - b_ppl, 6),
            "delta_r4": round(pub[n]["capability"]["r4_full"] - b_r4, 6),
            "ppl": round(pub[n]["capability"]["ppl_full"], 6),
            "r4": round(pub[n]["capability"]["r4_full"], 6),
            "gate_ok": bool(pub[n]["capability_ok"]),
        })
    rows.sort(key=lambda r: -abs(r["delta_r4"]))
    byte_set = {p["organ_bytes"] for p in plans.values()}
    depth_set = {tuple(sorted(v["bits"] for v in plans[n]["per_organ"].values()))
                 for n in names}
    dominant = rows[0] if rows else None
    return {
        "schema": ISOLATE_SCHEMA, "obligation": "G022", "status": "MEASURED",
        "question": ("which equal-parameter organ group is responsible for the "
                     "GLOBAL_ANATOMY allocation win?"),
        "specimen": slug, "specimen_path": snapshot,
        "base_assignment": {k: f"q{v[0]}-g{v[1]}" for k, v in base.items()},
        "assignments": {n: {k: f"q{v[0]}-g{v[1]}" for k, v in asg.items()}
                        for n, asg in arms.items()},
        "byte_exact_across_all_arms": len(byte_set) == 1,
        "depth_exact_across_all_arms": len(depth_set) == 1,
        "organ_bytes": sorted(byte_set),
        "base_ppl": round(b_ppl, 6), "base_r4": round(b_r4, 6),
        "base_gate_ok": bool(pub["GLOBAL_BASE"]["capability_ok"]),
        "per_group": rows,
        "dominant_group": dominant["group_changed"] if dominant else None,
        "dense_parent_capability": dense_cap,
        "arms": pub, "applied": applied,
        "guard_at_start": guard.as_dict(), "resource": cost, "device": "cpu",
        "nr_not_nx": True,
        "weaker_than_it_looks": [
            "Single-group swaps measure each group AGAINST THIS BASE. They do not "
            "prove the groups are independent; a different base could reorder them.",
            "One specimen, one budget, one corpus.",
        ],
        "verdict_sentence": (
            f"G022 on {slug}: base GLOBAL_ANATOMY ppl {b_ppl:.4f} r4 {b_r4:.4f} "
            f"({'PASS' if pub['GLOBAL_BASE']['capability_ok'] else 'FAIL'}), all arms "
            f"byte-exact and depth-exact. Group effect on diversity, largest first: "
            + "; ".join(f"{r['arm']} d_r4 {r['delta_r4']:+.4f} d_ppl {r['delta_ppl']:+.4f}"
                        for r in rows) + "."),
    }


ALLOC_SCHEMA = "hawking.future.g012_sensitivity_allocation.v1"
ALLOC_RECEIPT = "G012_SENSITIVITY_ALLOCATION.json"

# The tilt. Chosen so a pair {cheap, mid} and a triple {cheap, mid, rich} each
# reproduce all-RICH's byte total EXACTLY on this specimen's equal-parameter
# groups -- verified, delta 0 bytes. Every arm below is therefore byte-exact,
# and the three TILTED arms additionally share one bit-depth multiset, which is
# the confound LAW003 says destroys an ordering claim.
ALLOC_CHEAP = (3, 32)
ALLOC_MID = (4, 32)
ALLOC_RICH = (4, 64)


def allocation_arms(sensitivity: dict[str, float]) -> dict[str, dict[str, tuple[int, int]]]:
    """UNIFORM plus three permutations of ONE multiset over the equal-size groups.

    Within {q,o}, {k,v} and {down,gate,up} any permutation is byte-identical, so
    the only thing that varies between the tilted arms is WHICH organ is
    protected. SENSITIVITY spends the cheapest spec on the organ measured least
    sensitive; ANTI_SENSITIVITY is its negative control and must do WORSE, or
    the map has no predictive power; GLOBAL_ANATOMY spends it on the organ the
    structural prior calls most redundant.
    """
    pairs = (("o_proj", "q_proj"), ("k_proj", "v_proj"))
    triple = ("down_proj", "gate_proj", "up_proj")

    def order_by(score: dict[str, float], names, reverse: bool):
        return sorted(names, key=lambda n: score[n], reverse=reverse)

    glob = dict(PRIOR_DEFICIT)
    arms: dict[str, dict[str, tuple[int, int]]] = {}
    arms["UNIFORM"] = {f: ALLOC_RICH for f in STANDARD_ORGANS}

    def build(score: dict[str, float], reverse: bool) -> dict[str, tuple[int, int]]:
        # reverse=False: LOWEST score first == cheapest spec to the lowest score.
        a: dict[str, tuple[int, int]] = {}
        for grp in pairs:
            lo, hi = order_by(score, grp, reverse)
            a[lo], a[hi] = ALLOC_CHEAP, ALLOC_MID
        lo, mid, hi = order_by(score, triple, reverse)
        a[lo], a[mid], a[hi] = ALLOC_CHEAP, ALLOC_MID, ALLOC_RICH
        return a

    # Least sensitive gets the cheapest spec.
    arms["SENSITIVITY"] = build(sensitivity, reverse=False)
    # Negative control: most sensitive gets the cheapest spec.
    arms["ANTI_SENSITIVITY"] = build(sensitivity, reverse=True)
    # Highest structural deficit gets the cheapest spec -- the prior's own rule.
    arms["GLOBAL_ANATOMY"] = build(glob, reverse=True)
    return arms


def run_allocation_experiment(
    *,
    snapshot: str | None = None,
    slug: str | None = None,
    expected_gb: float = 8.0,
    sensitivity_receipt: str = "receipts/future/G011_ORGAN_SENSITIVITY.json",
) -> dict[str, Any]:
    """Does allocating by MEASURED sensitivity beat allocating by anatomy?"""
    from campaign_memory_guard import require_ok, resource_cost, sample

    snapshot = snapshot or SPECIMEN_PATH
    slug = slug or SPECIMEN_SLUG
    src = _REPO / sensitivity_receipt
    if not src.is_file():
        raise FileNotFoundError(
            f"{sensitivity_receipt} is required: this arm allocates by MEASURED "
            "sensitivity and will not fall back to a structural proxy")
    sdoc = json.loads(src.read_text())
    sens = {r["organ"]: float(r["delta_ppl"]) for r in sdoc["sensitivity"]}
    missing = sorted(set(STANDARD_ORGANS) - set(sens))
    if missing:
        raise ValueError(f"{sensitivity_receipt} has no measured sensitivity for {missing}")

    guard = sample(expected_gb=expected_gb)
    if guard.state == "STOP":
        return {"schema": ALLOC_SCHEMA, "status": "GUARD_STOP", "guard": guard.as_dict(),
                "verdict_sentence": "NO RESULT: campaign_memory_guard STOP before load"}
    require_ok("G012 sensitivity allocation", expected_gb=expected_gb)

    inv = inventory(snapshot)
    assigns = allocation_arms(sens)
    plans = {n: plan_arm(inv, n, assignment=a) for n, a in assigns.items()}
    names = list(plans)
    byte_set = {p["organ_bytes"] for p in plans.values()}
    depth_sets = {n: tuple(sorted(v["bits"] for v in plans[n]["per_organ"].values()))
                  for n in names}
    tilted = [n for n in names if n != "UNIFORM"]

    caps: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    applied: dict[str, Any] = {}
    with resource_cost(interval_s=2.0, label="G012_ALLOC") as cost:
        weights = _load_bf16_payloads(inv)
        model, tok = _load_model(snapshot)
        dense_cap = measure_capability(model, tok)
        for nm in names:
            rec = reconstruct_arm(inv, weights, plans[nm]["specs"])
            applied[nm] = _apply_recs(model, rec)
            del rec
            caps[nm] = measure_capability(model, tok)
            gates[nm] = gate_from_reference(caps[nm], dense_cap)
        del model, weights

    arms_pub = {n: _public_arm({k: v for k, v in plans[n].items() if k != "specs"},
                               caps[n], gates[n]) for n in names}
    sens_p = arms_pub["SENSITIVITY"]["capability"]["ppl_full"]
    anti_p = arms_pub["ANTI_SENSITIVITY"]["capability"]["ppl_full"]
    glob_p = arms_pub["GLOBAL_ANATOMY"]["capability"]["ppl_full"]
    uni_p = arms_pub["UNIFORM"]["capability"]["ppl_full"]

    # CAPABILITY decides, not perplexity. Ranking these arms on likelihood alone
    # reported "anatomy not beaten" while missing that the anatomy arm was the
    # only one to survive the conjunction -- the exact one-metric trap LAW008
    # exists to prevent, walked into by the verdict function itself.
    passing = [n for n in names if bool(arms_pub[n].get("capability_ok"))]
    winners = [n for n in passing
               if not any(m != n and _pareto_better(arms_pub[m], arms_pub[n]) == "a"
                          for m in passing)]
    # The control is about the MAP, and is judged on the continuous axes because
    # neither it nor its mirror may survive the gate at a given budget.
    control_holds = sens_p < anti_p
    if len(winners) == 1:
        outcome = f"{winners[0]}_WINS"
        note = (f"{winners[0]} is the only allocation to survive the conjunction at "
                f"identical bytes; likelihood alone would have ranked it differently")
    elif not passing:
        outcome = "NONE_PASSES_GATE"
        note = ("no allocation survived the conjunction at this budget; the "
                "comparison between them is unresolved, not a win for the best ppl")
    else:
        outcome = "UNDECIDED"
        note = ("more than one allocation passes and none Pareto-dominates the "
                "others at this budget")
    if not control_holds:
        outcome = "MAP_HAS_NO_PREDICTIVE_POWER"
        note = ("the anti-sensitivity control matched or beat the sensitivity arm at "
                "identical bytes and identical bit depths; the measured map does not "
                "predict allocation tolerance and must not be used as one. " + note)

    return {
        "schema": ALLOC_SCHEMA,
        "obligation": "G012",
        "status": "MEASURED",
        "question": ("does allocating by MEASURED behavioural sensitivity beat "
                     "allocating by structural anatomy, at identical bytes and "
                     "identical bit depths?"),
        "specimen": slug,
        "specimen_path": snapshot,
        "sensitivity_source": sensitivity_receipt,
        "measured_sensitivity": sens,
        "specs": {"cheap": f"q{ALLOC_CHEAP[0]}-g{ALLOC_CHEAP[1]}",
                  "mid": f"q{ALLOC_MID[0]}-g{ALLOC_MID[1]}",
                  "rich": f"q{ALLOC_RICH[0]}-g{ALLOC_RICH[1]}"},
        "assignments": {n: {f: f"q{c[0]}-g{c[1]}" for f, c in a.items()}
                        for n, a in assigns.items()},
        "byte_exact_across_all_arms": len(byte_set) == 1,
        "organ_bytes": sorted(byte_set),
        "depth_multisets": {n: list(v) for n, v in depth_sets.items()},
        "depth_exact_across_tilted_arms": len({depth_sets[n] for n in tilted}) == 1,
        "uniform_depth_differs_by_design": True,
        "outcome": outcome,
        "winners": winners,
        "passing_gate": passing,
        "control_holds": control_holds,
        "control_gap_ppl": round(anti_p - sens_p, 6),
        "decided_by": "the capability conjunction, not perplexity alone",
        "dense_parent_capability": dense_cap,
        "arms": arms_pub,
        "applied": applied,
        "guard_at_start": guard.as_dict(),
        "resource": cost,
        "device": "cpu",
        "nr_not_nx": True,
        "weaker_than_it_looks": [
            "The sensitivity map was measured at ONE perturbation size on THIS body; "
            "allocating by it is an out-of-sample use of an in-sample measurement.",
            "UNIFORM is byte-matched but NOT depth-matched -- it cannot be, since "
            "uniform means one spec everywhere. Only the three tilted arms are a "
            "clean ordering comparison.",
            "Capability is 8 greedy prompts x 96 tokens plus 512-token NLL on CPU "
            "fp32 -- the existing gate's budget, not a large eval.",
        ],
        "verdict_sentence": (
            f"G012 {outcome} on {slug} at complete EBPW "
            f"{arms_pub['UNIFORM']['complete_ebpw']:.4f}, all arms byte-exact: "
            f"SENSITIVITY ppl {sens_p:.4f}, ANTI_SENSITIVITY (control) {anti_p:.4f}, "
            f"GLOBAL_ANATOMY {glob_p:.4f}, UNIFORM {uni_p:.4f}; gate PASS = "
            f"{passing or 'none'}. {note}."),
    }


SENS_SCHEMA = "hawking.future.g011_organ_sensitivity.v1"
SENS_RECEIPT = "G011_ORGAN_SENSITIVITY.json"


def run_sensitivity_experiment(
    *,
    snapshot: str | None = None,
    slug: str | None = None,
    expected_gb: float = 8.0,
    cheap: tuple[int, int] = (3, 64),
) -> dict[str, Any]:
    """Per-organ BEHAVIOURAL sensitivity, measured, not inferred from structure.

    G017 and G021 both turned on the assumption that structural redundancy tells
    you what an organ tolerates. This measures tolerance directly: hold every
    organ at the base spec, drop exactly ONE to the cheap spec, and read the
    capability slope. One perturbation, one organ, everything else fixed.

    Organs differ in size, so the raw delta is also reported per megabyte saved
    -- otherwise the biggest organ always looks the most sensitive simply
    because crushing it removes the most information.

    The control that matters: the measured ordering must not be the structural
    ordering by construction. It is not -- structure is never consulted here.
    """
    from campaign_memory_guard import require_ok, resource_cost, sample

    snapshot = snapshot or SPECIMEN_PATH
    slug = slug or SPECIMEN_SLUG
    guard = sample(expected_gb=expected_gb)
    if guard.state == "STOP":
        return {"schema": SENS_SCHEMA, "status": "GUARD_STOP",
                "guard": guard.as_dict(),
                "verdict_sentence": "NO RESULT: campaign_memory_guard STOP before load"}
    require_ok("G011 organ sensitivity", expected_gb=expected_gb)

    inv = inventory(snapshot)
    organs = [f for f in STANDARD_ORGANS if inv["organ_params"].get(f)]
    base_assign = {f: UNIFORM_AFFINE for f in inv["organ_params"]}
    plans = {"BASE": plan_arm(inv, "UNIFORM", assignment=base_assign)}
    for fam in organs:
        a = dict(base_assign)
        a[fam] = cheap
        plans[fam] = plan_arm(inv, f"CRUSH_{fam}", assignment=a)
    names = list(plans)

    caps: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    applied: dict[str, Any] = {}
    with resource_cost(interval_s=2.0, label="G011_SENS") as cost:
        weights = _load_bf16_payloads(inv)
        model, tok = _load_model(snapshot)
        dense_cap = measure_capability(model, tok)
        for nm in names:
            rec = reconstruct_arm(inv, weights, plans[nm]["specs"])
            applied[nm] = _apply_recs(model, rec)
            del rec
            caps[nm] = measure_capability(model, tok)
            gates[nm] = gate_from_reference(caps[nm], dense_cap)
        del model, weights

    base_ppl = float(caps["BASE"]["ppl_full"])
    base_r4 = float(caps["BASE"]["r4_full"])
    base_bytes = int(plans["BASE"]["organ_bytes"])
    local = local_deficits(slug)
    rows = []
    for fam in organs:
        saved = base_bytes - int(plans[fam]["organ_bytes"])
        d_ppl = float(caps[fam]["ppl_full"]) - base_ppl
        d_r4 = float(caps[fam]["r4_full"]) - base_r4
        mb = saved / 1e6
        rows.append({
            "organ": fam,
            "delta_ppl": round(d_ppl, 6),
            "delta_r4": round(d_r4, 6),
            "bytes_saved": saved,
            # Per-MB, because a bigger organ loses more information at the same
            # spec and would otherwise always look the most sensitive.
            "delta_ppl_per_mb_saved": round(d_ppl / mb, 6) if mb else None,
            "gate_ok": bool(gates[fam]["capability_ok"]),
            "ppl": round(float(caps[fam]["ppl_full"]), 6),
            "r4": round(float(caps[fam]["r4_full"]), 6),
            "global_deficit": PRIOR_DEFICIT[fam],
            "local_deficit": local.get(fam),
        })
    by_raw = [r["organ"] for r in sorted(rows, key=lambda r: r["delta_ppl"])]
    by_mb = [r["organ"] for r in sorted(rows, key=lambda r: (r["delta_ppl_per_mb_saved"] or 0.0))]
    by_global = [r["organ"] for r in sorted(rows, key=lambda r: -r["global_deficit"])]
    by_local = [r["organ"] for r in sorted(rows, key=lambda r: -(r["local_deficit"] or 0.0))]

    def _rank_agreement(a: list[str], b: list[str]) -> dict[str, Any]:
        ra = {n: i for i, n in enumerate(a)}
        rb = {n: i for i, n in enumerate(b)}
        conc = disc = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                x, y = a[i], a[j]
                if (ra[x] - ra[y]) * (rb[x] - rb[y]) > 0:
                    conc += 1
                else:
                    disc += 1
        n = conc + disc
        return {"concordant": conc, "discordant": disc,
                "kendall_tau": round((conc - disc) / n, 4) if n else None}

    out = {
        "schema": SENS_SCHEMA,
        "obligation": "G011",
        "status": "MEASURED",
        "question": ("which organs actually carry behaviour? one organ dropped to "
                     "the cheap spec at a time, everything else held at base"),
        "specimen": slug,
        "specimen_path": snapshot,
        "base_spec": f"q{UNIFORM_AFFINE[0]}-g{UNIFORM_AFFINE[1]}",
        "cheap_spec": f"q{cheap[0]}-g{cheap[1]}",
        "base_ppl": round(base_ppl, 6),
        "base_r4": round(base_r4, 6),
        "dense_parent_capability": dense_cap,
        "sensitivity": sorted(rows, key=lambda r: -r["delta_ppl"]),
        "ordering_least_to_most_sensitive_raw": by_raw,
        "ordering_least_to_most_sensitive_per_mb": by_mb,
        "ordering_by_global_deficit_desc": by_global,
        "ordering_by_local_deficit_desc": by_local,
        "agreement_raw_vs_global": _rank_agreement(by_raw, by_global),
        "agreement_raw_vs_local": _rank_agreement(by_raw, by_local),
        "agreement_per_mb_vs_global": _rank_agreement(by_mb, by_global),
        "agreement_per_mb_vs_local": _rank_agreement(by_mb, by_local),
        "control": ("structure is never consulted in producing the measured "
                    "ordering, so agreement with the deficit ordering is a "
                    "finding and not an artefact of construction"),
        "applied": applied,
        "guard_at_start": guard.as_dict(),
        "resource": cost,
        "device": "cpu",
        "nr_not_nx": True,
        "weaker_than_it_looks": [
            "One specimen, one perturbation size (base -> cheap), one direction.",
            "Sensitivity measured at ONE point is a local slope, not a curve; an "
            "organ flat here may cliff at a deeper spec.",
            "Capability is 8 greedy prompts x 96 tokens plus 512-token NLL on CPU "
            "fp32 -- deterministic given the weights, but not robust to a "
            "different prompt set.",
        ],
    }
    out["verdict_sentence"] = (
        f"G011 sensitivity on {slug}: crushing one organ at a time from "
        f"{out['base_spec']} to {out['cheap_spec']}. Least to most sensitive by raw "
        f"ppl delta: {', '.join(by_raw)}. Global structural prior would predict "
        f"{', '.join(by_global)} (tau {out['agreement_raw_vs_global']['kendall_tau']}); "
        f"local anatomy would predict {', '.join(by_local)} "
        f"(tau {out['agreement_raw_vs_local']['kendall_tau']})."
    )
    return out


MIRROR_SCHEMA = "hawking.future.g021_mirror_swap.v1"
MIRROR_RECEIPT = "G021_MIRROR_SWAP.json"


def run_mirror_experiment(
    *,
    snapshot: str | None = None,
    slug: str | None = None,
    expected_gb: float = 8.0,
    cheap: tuple[int, int] = (3, 64),
    rich: tuple[int, int] = (8, 64),
) -> dict[str, Any]:
    """Does it matter WHICH organ gets the bytes, holding everything else exact?

    The monotone-ramp arms could not answer this. Each policy picked its own
    bit-depth multiset to hit the budget, and on this specimen minimum bit depth
    separates perplexity almost perfectly: every arm that reached q2 landed at
    6.76-12.84, every arm that stopped at q3+ landed at 5.23-6.17. So a
    UNIFORM-vs-GLOBAL-vs-LOCAL difference is mostly a depth difference wearing
    an ordering label.

    Here the two arms of a pair hold organs of IDENTICAL parameter count and
    exchange the same two specs. Same total bytes, same complete EBPW, same
    depth multiset, same everything -- except which organ is protected. If the
    structural prior means anything for allocation, crushing the HIGH-deficit
    organ of a pair should cost less than crushing the LOW-deficit one.
    """
    from campaign_memory_guard import require_ok, resource_cost, sample

    snapshot = snapshot or SPECIMEN_PATH
    slug = slug or SPECIMEN_SLUG
    guard = sample(expected_gb=expected_gb)
    if guard.state == "STOP":
        return {"schema": MIRROR_SCHEMA, "status": "GUARD_STOP",
                "guard": guard.as_dict(),
                "verdict_sentence": "NO RESULT: campaign_memory_guard STOP before load; "
                                    + "; ".join(guard.reasons)}
    require_ok("G021 mirror swap", expected_gb=expected_gb)

    inv = inventory(snapshot)
    plans = plan_mirror_arms(inv, cheap=cheap, rich=rich)
    names = [k for k in plans if not k.startswith("_")]
    local = local_deficits(slug)

    dense_cap = None
    caps: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    applied: dict[str, Any] = {}
    with resource_cost(interval_s=2.0, label="G021_MIRROR") as cost:
        # Eleven arms reconstructed up front would hold eleven fp32 copies of
        # every organ at once. Keep the bf16 payloads and rebuild ONE arm at a
        # time instead: peak is payloads + one arm, not payloads x eleven.
        weights = _load_bf16_payloads(inv)
        model, tok = _load_model(snapshot)
        probe = reconstruct_arm(inv, weights, plans["BASE"]["specs"])
        n_named = sum(1 for nm, _ in model.named_parameters() if nm in probe)
        if n_named != inv["n_organs"]:
            raise RuntimeError(
                f"HF named_parameters matched {n_named} of {inv['n_organs']} organs")
        del probe
        applied["n_organ_tensors"] = inv["n_organs"]
        dense_cap = measure_capability(model, tok)
        for nm in names:
            rec = reconstruct_arm(inv, weights, plans[nm]["specs"])
            applied[nm] = _apply_recs(model, rec)
            del rec
            caps[nm] = measure_capability(model, tok)
            gates[nm] = gate_from_reference(caps[nm], dense_cap)
        del model, weights

    arms_pub: dict[str, Any] = {}
    for n in names:
        plan = {k: v for k, v in plans[n].items() if k != "specs"}
        arms_pub[n] = _public_arm(plan, caps[n], gates[n])

    # One row per pair: the two mirrored arms and which organ turned out to be
    # the cheaper one to crush.
    rows = []
    for a, b in [tuple(x) for x in plans["_pairs"]]:
        ka, kb = f"{a}|{b}:{a}_cheap", f"{a}|{b}:{b}_cheap"
        pa, pb = arms_pub[ka], arms_pub[kb]
        ppl_a = pa["capability"]["ppl_full"]
        ppl_b = pb["capability"]["ppl_full"]
        r4_a = pa["capability"]["r4_full"]
        r4_b = pb["capability"]["r4_full"]
        # "cheaper to crush" = the arm with the BETTER capability is the one whose
        # crushed organ tolerated it.
        better = _pareto_better(pa, pb)
        tolerant = {"a": a, "b": b}.get(better)
        pred_global = a if PRIOR_DEFICIT[a] > PRIOR_DEFICIT[b] else b
        pred_local = a if local[a] > local[b] else b
        rows.append({
            "pair": [a, b],
            "bytes_identical": pa["organ_bytes"] == pb["organ_bytes"],
            "complete_ebpw": pa["complete_ebpw"],
            "crush_" + a: {"ppl": ppl_a, "r4": r4_a, "gate_ok": pa["capability_ok"]},
            "crush_" + b: {"ppl": ppl_b, "r4": r4_b, "gate_ok": pb["capability_ok"]},
            "ppl_delta_crush_a_minus_b": round(ppl_a - ppl_b, 6),
            "pareto": better,
            "more_tolerant_organ": tolerant,
            "global_prior_predicts_tolerant": pred_global,
            "local_anatomy_predicts_tolerant": pred_local,
            "global_correct": None if tolerant is None else (tolerant == pred_global),
            "local_correct": None if tolerant is None else (tolerant == pred_local),
            "priors_agree": pred_global == pred_local,
            "deficits": {a: {"global": PRIOR_DEFICIT[a], "local": local[a]},
                         b: {"global": PRIOR_DEFICIT[b], "local": local[b]}},
        })
    decided = [r for r in rows if r["global_correct"] is not None]
    g_hits = sum(1 for r in decided if r["global_correct"])
    l_hits = sum(1 for r in decided if r["local_correct"])
    disagree = [r for r in rows if not r["priors_agree"]]
    sentence = (
        f"G021 mirror swap on {slug}: {len(rows)} equal-size organ pairs, each arm "
        f"byte-identical and depth-identical to its mirror. Decided pairs "
        f"{len(decided)}/{len(rows)}. The GLOBAL structural prior predicted the more "
        f"tolerant organ {g_hits}/{len(decided)}; the LOCAL anatomy predicted it "
        f"{l_hits}/{len(decided)}. The two priors disagree on "
        f"{[r['pair'] for r in disagree] or 'no pair'}."
    )
    return {
        "schema": MIRROR_SCHEMA,
        "obligation": "G021",
        "status": "MEASURED",
        "question": ("holding total bytes, complete EBPW and the bit-depth multiset "
                     "EXACTLY constant, does it matter which organ is protected?"),
        "why_this_design": (
            "The monotone-ramp arms confound ordering with depth: each policy "
            "chooses its own multiset to hit the budget, and on this body minimum "
            "bit depth separates perplexity almost perfectly. Equal-parameter "
            "organs exchanging the same two specs remove that confound entirely."),
        "specimen": slug,
        "specimen_path": snapshot,
        "cheap_spec": f"q{cheap[0]}-g{cheap[1]}",
        "rich_spec": f"q{rich[0]}-g{rich[1]}",
        "base_spec": f"q{UNIFORM_AFFINE[0]}-g{UNIFORM_AFFINE[1]}",
        "pairs": rows,
        "global_prior_hits": g_hits,
        "local_prior_hits": l_hits,
        "n_decided": len(decided),
        "n_pairs": len(rows),
        "prior_deficit": dict(PRIOR_DEFICIT),
        "local_deficit": local,
        "dense_parent_capability": dense_cap,
        "arms": arms_pub,
        "applied": applied,
        "guard_at_start": guard.as_dict(),
        "resource": cost,
        "device": "cpu",
        "nr_not_nx": True,
        "weaker_than_it_looks": [
            "One specimen, one base budget, one cheap/rich spec pair. A pair that "
            "does not separate here may separate at another depth.",
            "Capability is 8 greedy prompts x 96 tokens plus 512-token NLL on CPU "
            "fp32 -- the existing gate's budget, not a large eval.",
            "Greedy decoding makes each number deterministic given the weights, so "
            "there is no seed spread; that is NOT robustness to a different prompt "
            "set, and a small ppl gap should not be read as a large effect.",
        ],
        "verdict_sentence": sentence,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--alloc-only", action="store_true")
    p.add_argument("--mutation-check", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--corpus", default="A", choices=sorted(CORPORA),
                   help="which capability corpus to evaluate on. Every number in "
                        "the campaign so far is corpus A; B exists to test whether "
                        "a result survives a different domain and register.")
    p.add_argument("--isolate", action="store_true",
                   help="which organ group makes the winning allocation win: one "
                        "group changed per arm, everything else identical")
    p.add_argument("--allocate", action="store_true",
                   help="allocate by MEASURED sensitivity vs structural anatomy vs "
                        "uniform, byte-exact and depth-exact, with an "
                        "anti-sensitivity negative control")
    p.add_argument("--sensitivity", action="store_true",
                   help="per-organ behavioural sensitivity: one organ crushed at "
                        "a time, everything else held at base")
    p.add_argument("--mirror", action="store_true",
                   help="byte-exact, depth-exact organ swap; the unconfounded "
                        "form of the G021 ordering question")
    p.add_argument("--ingest-ledger", action="store_true")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--slug", default=None)
    p.add_argument("--receipt", default=None)
    p.add_argument("--expected-gb", type=float, default=8.0)
    p.add_argument("--skip-mutation", action="store_true")
    # The budget knob. G021 ran every arm at q4-g64 and NO arm survived the
    # conjunction, so the three-way comparison could not name a winner -- the
    # arms were being ranked while all of them were already broken. Making the
    # UNIFORM spec a parameter lets the same experiment run at a budget where
    # UNIFORM is alive, which is the only place the question resolves. [S010 31]
    p.add_argument("--uniform-spec", default=None,
                   help="BITSxGROUP for the UNIFORM arm, e.g. 4x32. UNIFORM "
                        "defines the matched byte budget every arm is held to.")
    args = p.parse_args(argv)
    globals()["ACTIVE_CORPUS"] = args.corpus
    if args.receipt:
        # Validate the write path BEFORE the model load. This exact call used
        # to fail on the last line of the run, after every arm was measured.
        from _common import receipt_name as _receipt_name
        try:
            args.receipt = _receipt_name(args.receipt)
        except ValueError as exc:
            raise SystemExit(f"--receipt: {exc}")
    if args.uniform_spec:
        try:
            bits_s, group_s = str(args.uniform_spec).lower().split("x", 1)
            bits, group = int(bits_s), int(group_s)
        except Exception:
            raise SystemExit(f"--uniform-spec wants BITSxGROUP, got {args.uniform_spec!r}")
        if (bits, group) not in AFFINE_CFGS:
            raise SystemExit(
                f"--uniform-spec {bits}x{group} is not on the billed affine grid "
                f"{AFFINE_CFGS}; an unbilled spec has no accounting")
        globals()["UNIFORM_AFFINE"] = (bits, group)
        # Same formula the module constant documents: bits + 32/group.
        globals()["TARGET_ORGAN_EBPW"] = bits + 32.0 / group
    if args.selfcheck:
        out = selfcheck()
        print(json.dumps(out, indent=2, sort_keys=True))
        print("SELFCHECK_OK")
        return 0
    if args.mutation_check:
        out = run_mutation_check()
        print(json.dumps(out, indent=2, sort_keys=True))
        print("MUTATION_CHECK_OK")
        return 0
    if args.isolate:
        doc = run_isolation_experiment(
            snapshot=args.snapshot, slug=args.slug, expected_gb=args.expected_gb)
        path = write_doc(doc, args.receipt or ISOLATE_RECEIPT)
        print(path)
        print(doc["verdict_sentence"])
        return 0
    if args.allocate:
        doc = run_allocation_experiment(
            snapshot=args.snapshot, slug=args.slug, expected_gb=args.expected_gb)
        path = write_doc(doc, args.receipt or ALLOC_RECEIPT)
        print(path)
        print(doc["verdict_sentence"])
        return 0
    if args.sensitivity:
        doc = run_sensitivity_experiment(
            snapshot=args.snapshot, slug=args.slug, expected_gb=args.expected_gb)
        path = write_doc(doc, args.receipt or SENS_RECEIPT)
        print(path)
        print(doc["verdict_sentence"])
        return 0
    if args.mirror:
        doc = run_mirror_experiment(
            snapshot=args.snapshot, slug=args.slug, expected_gb=args.expected_gb)
        path = write_doc(doc, args.receipt or MIRROR_RECEIPT)
        print(path)
        print(doc["verdict_sentence"])
        return 0
    if args.alloc_only:
        inv = inventory(SPECIMEN_PATH)
        plans = plan_three_arms(inv)
        pub = {
            k: {kk: vv for kk, vv in plans[k].items() if kk != "specs"}
            for k in POLICIES
        }
        pub["residual_weighted_vs_uniform"] = plans["ebpw_residual_weighted_vs_uniform"]
        pub["residual_inverted_vs_uniform"] = plans["ebpw_residual_inverted_vs_uniform"]
        print(json.dumps(pub, indent=2, sort_keys=True, default=str))
        return 0
    if args.run:
        doc = run_experiment(
            skip_capability=False,
            snapshot=args.snapshot,
            slug=args.slug,
            expected_gb=args.expected_gb,
            skip_mutation=args.skip_mutation,
            why=(
                "Second dense causal LM: reuse G002 organ prior and the G017 allocator; "
                "skip a new anatomy sweep. G008 compounding check."
                if args.slug and args.slug != SPECIMEN_SLUG
                else None
            ),
        )
        path = write_doc(doc, args.receipt)
        print(path)
        print(doc.get("verdict_sentence") or doc.get("status"))
        return 0 if doc.get("status") != "GUARD_STOP" else 2
    if args.ingest_ledger:
        out = ingest_g017_into_ledger()
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
