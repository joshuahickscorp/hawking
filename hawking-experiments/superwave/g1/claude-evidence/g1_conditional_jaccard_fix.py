#!/usr/bin/env python3
"""Recompute token-id Jaccard with honest different-context pairs.

The 5th prompt is 10 tokens. Prompts 0-3 share a 45-token system prefix.
The first pass used prefix_len=10, so positions 10-44 of prompts 0-3 were
falsely treated as different context.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

CAP = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
PREV = "/tmp/g1_conditional_repr.json"
OUT = "/tmp/g1_conditional_jaccard_fix.json"

N_TOK = 256
HIDDEN = 5120
K = 32


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pairwise_mean(sets):
    if len(sets) < 2:
        return float("nan")
    acc = n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            acc += jaccard(sets[i], sets[j])
            n += 1
    return acc / n if n else float("nan")


receipt = json.load(open(f"{CAP}/capture-result.json"))
# slices
slices = []
t0 = 0
ids = []
for i, pr in enumerate(receipt["prompts"]):
    n = int(pr["n_tokens"])
    slices.append((i, t0, t0 + n, list(pr["ids"])))
    ids.extend(pr["ids"])
    t0 += n
ids = np.array(ids, dtype=np.int64)

# common prefix of first 4 prompts
seqs4 = [s[3] for s in slices[:4]]
m = min(len(s) for s in seqs4)
pref4 = 0
while pref4 < m and all(s[pref4] == seqs4[0][pref4] for s in seqs4):
    pref4 += 1
pref5 = 0
seqs5 = [s[3] for s in slices]
m5 = min(len(s) for s in seqs5)
while pref5 < m5 and all(s[pref5] == seqs5[0][pref5] for s in seqs5):
    pref5 += 1

print("prefix_all5", pref5, "prefix_first4", pref4)

# token meta
t2 = {}
for pi, a, b, pids in slices:
    for off, t in enumerate(range(a, b)):
        t2[t] = (pi, off, int(ids[t]))

# load hidden needed layers
layers = [0, 6, 7, 32, 63]
X = {}
for L in layers:
    X[L] = np.fromfile(f"{CAP}/hidden/L{L:02d}.f32", dtype=np.float32).reshape(N_TOK, HIDDEN)

occ = defaultdict(list)
for t, tid in enumerate(ids.tolist()):
    occ[int(tid)].append(t)

rng = np.random.default_rng(3994)
out = {"prefix_all5": pref5, "prefix_first4": pref4, "layers": {}}

for L in layers:
    e = X[L].astype(np.float64) ** 2
    tops = []
    for t in range(N_TOK):
        idx = np.argpartition(e[t], -K)[-K:]
        tops.append(set(int(i) for i in idx))

    # ALL same-id pairs (contaminated)
    same_all = []
    for ts in occ.values():
        if len(ts) >= 2:
            same_all.append(pairwise_mean([tops[t] for t in ts]))

    # honest different-context:
    # drop tokens that sit in the shared first-4 prefix (off < pref4) except keep ONE copy
    # also, two tokens are "same context class" if they have the same offset < pref4
    same_diff = []
    n_ids = 0
    pair_count = 0
    for tid, ts in occ.items():
        # partition: prefix-aligned copies vs truly other occurrences
        by_off = defaultdict(list)
        others = []
        for t in ts:
            pi, off, _ = t2[t]
            if pi < 4 and off < pref4:
                by_off[off].append(t)
            else:
                others.append(t)
        # one representative per shared-prefix offset + all others
        pool = [v[0] for v in by_off.values()] + others
        if len(pool) >= 2:
            same_diff.append(pairwise_mean([tops[t] for t in pool]))
            n_ids += 1
            pair_count += len(pool) * (len(pool) - 1) // 2

    # even stricter: ONLY occurrences outside the 45-token shared prefix
    same_outside = []
    n_out = 0
    for tid, ts in occ.items():
        outside = []
        for t in ts:
            pi, off, _ = t2[t]
            if not (pi < 4 and off < pref4):
                outside.append(t)
        if len(outside) >= 2:
            same_outside.append(pairwise_mean([tops[t] for t in outside]))
            n_out += 1

    # random different ids
    rand = []
    for _ in range(400):
        i, j = rng.integers(0, N_TOK, size=2)
        if i != j and ids[i] != ids[j]:
            rand.append(jaccard(tops[i], tops[j]))

    # consecutive inside prompts
    bounds = {s[2] - 1 for s in slices}
    consec = [jaccard(tops[t], tops[t + 1]) for t in range(N_TOK - 1) if t not in bounds]

    # shared prefix same offset among first 4 (should be ~1)
    pref_js = []
    for off in range(pref4):
        sets_off = [tops[slices[pi][1] + off] for pi in range(4)]
        pref_js.append(pairwise_mean(sets_off))

    rec = {
        "same_id_all_repeats_mean": float(np.nanmean(same_all)),
        "same_id_one_copy_of_shared_prefix_mean": float(np.nanmean(same_diff)) if same_diff else None,
        "n_ids_one_copy": n_ids,
        "same_id_outside_shared45_mean": float(np.nanmean(same_outside)) if same_outside else None,
        "n_ids_outside_shared45": n_out,
        "random_diff_id_mean": float(np.mean(rand)),
        "consecutive_in_prompt_mean": float(np.mean(consec)),
        "shared45_same_offset_mean": float(np.mean(pref_js)),
    }
    out["layers"][str(L)] = rec
    print(L, json.dumps(rec))

# also: which token ids actually have outside-prefix repeats?
print("\nids with >=2 occurrences outside shared45:")
for tid, ts in sorted(occ.items(), key=lambda kv: -len(kv[1])):
    outside = [t for t in ts if not (t2[t][0] < 4 and t2[t][1] < pref4)]
    if len(outside) >= 2:
        print(f"  id={tid} n_all={len(ts)} n_out={len(outside)} toks={outside[:12]}")

json.dump(out, open(OUT, "w"), indent=2)
print("WROTE", OUT)
