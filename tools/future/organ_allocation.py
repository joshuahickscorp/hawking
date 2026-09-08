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
POLICIES: tuple[str, ...] = ("UNIFORM", "ORGAN_WEIGHTED", "INVERTED")

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
_ORDER_KEY_DESCENDING = "-PRIOR_DEFICIT[name]"


def order_organs_by_deficit() -> tuple[str, ...]:
    """Most structure first (q/k/o), least structure last (up_proj)."""
    return tuple(sorted(PRIOR_DEFICIT, key=lambda name: -PRIOR_DEFICIT[name]))  # MUTATION_ANCHOR_ORDER_BY_DEFICIT


def per_param_weights(policy: str) -> dict[str, float]:
    """Bytes-per-parameter weights. Rank 0 (front of the order) is compressed most.

    Endpoint ratio is the prior's max/min deficit, so the tilt is measured
    rather than a free slope. UNIFORM is weight 1. INVERTED reverses the order,
    which is the control: if that wins, the prior did not transfer.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if policy == "UNIFORM":
        return {name: 1.0 for name in PRIOR_DEFICIT}
    order = order_organs_by_deficit()
    if policy == "INVERTED":
        order = tuple(reversed(order))
    n = len(order)
    # Endpoint ratio is remaining unstructured fraction (1 - deficit/100),
    # not max/min deficit. The raw 5.08x deficit ratio is a real number in
    # the prior and a disastrous byte map: it assigned q_proj 2.35 EBPW at a
    # mean-8 budget and destroyed likelihood on every arm. Remaining-fraction
    # ratio is ~1.745 and still monotone in the prior order, so the mutation
    # of that order still reverses who gets crushed.
    most = max(PRIOR_DEFICIT.values())
    least = min(PRIOR_DEFICIT.values())
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
) -> dict[str, float]:
    """Per-organ complete EBPW whose param-weighted mean is target_mean_ebpw.

    ORGAN_WEIGHTED: more bytes per param to low-deficit organs (up_proj last
    in the prior, so it receives the bytes saved by compressing q/k/o first).
    INVERTED: the same unequal allocation backwards.
    """
    weights = per_param_weights(policy)
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
    order = order_organs_by_deficit()
    names = [n for n in order if inv["organ_params"].get(n)]
    table = family_affine_table(inv)
    targets = allocate_organ_ebpw(policy, inv["organ_params"], TARGET_ORGAN_EBPW)
    monotone_up = policy == "ORGAN_WEIGHTED"
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
) -> dict[str, Any]:
    organ_ebpw = allocate_organ_ebpw(policy, inv["organ_params"], TARGET_ORGAN_EBPW)
    if target_bytes is None:
        # UNIFORM defines the budget.
        assignment = {fam: UNIFORM_AFFINE for fam in inv["organ_params"]}
    else:
        assignment = search_affine_assignment(policy, inv, target_bytes)
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
        "per_param_weights": per_param_weights(policy),
    }


def plan_three_arms(inv: dict[str, Any], target_mean_ebpw: float = TARGET_ORGAN_EBPW) -> dict[str, Any]:
    del target_mean_ebpw  # budget is UNIFORM_AFFINE, not a free mean
    uniform = plan_arm(inv, "UNIFORM")
    target = int(uniform["organ_bytes"])
    weighted = plan_arm(inv, "ORGAN_WEIGHTED", target_bytes=target)
    inverted = plan_arm(inv, "INVERTED", target_bytes=target)
    residual = abs(weighted["complete_ebpw_full"] - uniform["complete_ebpw_full"])
    residual_inv = abs(inverted["complete_ebpw_full"] - uniform["complete_ebpw_full"])
    return {
        "UNIFORM": uniform,
        "ORGAN_WEIGHTED": weighted,
        "INVERTED": inverted,
        "matched_organ_bytes": target,
        "ebpw_residual_weighted_vs_uniform": residual,
        "ebpw_residual_inverted_vs_uniform": residual_inv,
    }


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
    old = "key=lambda name: -" + "PRIOR_DEFICIT[name]))  # " + _tag
    new = "key=lambda name: " + "PRIOR_DEFICIT[name]))  # " + _tag
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
    nll_ids = tokenizer.encode(NLL_TEXT)[:NLL_TOKENS]
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
    for prompt in PROMPTS:
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
        "n_prompts": len(PROMPTS),
        "nll_tokens": len(nll_ids),
        "gen_tokens": GEN_TOKENS,
        "r4s": [round(x, 6) for x in r4s],
        "distincts": [round(x, 6) for x in dis],
        "generations": generations,
        "ppl_full": ppl,
        "r4_full": med_r4,
    }


def gate_from_reference(cap: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    r4_max = R4_MULT * float(ref["r4_full"])
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
    plans = plan_three_arms(inv, TARGET_ORGAN_EBPW)
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
    if arm_caps:
        verdict = _verdict(arms_pub)

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
            "the three 100% laws hold; allocation uses the GLOBAL prior"
        ),
        "sweep_row": sweep_row,
        "target_mean_organ_ebpw": TARGET_ORGAN_EBPW,
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
            "UNIFORM": arms_pub["UNIFORM"]["complete_ebpw"],
            "ORGAN_WEIGHTED": arms_pub["ORGAN_WEIGHTED"]["complete_ebpw"],
            "INVERTED": arms_pub["INVERTED"]["complete_ebpw"],
            "residual_weighted_vs_uniform": round(residual, 6),
        },
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
            "Allocation uses the GLOBAL prior, not this body's local order "
            "(local is k>q; global is q>k).",
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
    value = {
        "form": "organ-unequal affine NR",
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--alloc-only", action="store_true")
    p.add_argument("--mutation-check", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--ingest-ledger", action="store_true")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--slug", default=None)
    p.add_argument("--receipt", default=None)
    p.add_argument("--expected-gb", type=float, default=8.0)
    p.add_argument("--skip-mutation", action="store_true")
    args = p.parse_args(argv)
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
