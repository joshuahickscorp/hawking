I'll start by reading the frontier document in full, then design the per-component descent search and the roll-up math against its amendment.The amendment is clear: push every component to a 0.01–0.1 BPW floor, then roll those budgets into a total `complete_bpw`. Next I’ll pull prior Hawking science and the existing search machinery so the descent design sits on what’s already there.I’ll pull the novelty contract, existing sensitivity maps, and the Q80 mixed-representation receipts so the descent math is grounded in Hawking’s measured numbers, not generic mixed-precision folklore.Policy already names the descent loop; I’ll read the rulebase, candidate families, and existing search so the algorithm is an executable spec, not a restatement of the amendment.Existing Gravity allocators and rate-distortion tools are the prior art I must not rediscover. I’ll read those next so the descent spec extends them instead of reinventing the ledger.I’ll check whether this lane has a written contract with a required completion-report format before I lock the design.O005’s mass fractions flip the problem: experts dominate stored sub-1, but the 5% always-on tail (especially `lm_head` and attention) is what blocks stored sub-0.2. The search below is built around that, not around a uniform bit ladder.

# Opportunistic per-component descent — search spec

Lane: **deterministic opportunistic descent**. Not a codec. A search that pushes every component to a Doctor-gated floor in `{1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, ~0}` and rolls the heterogeneous budgets into one `complete_bpw`.

This extends machinery that already exists: `gravity_allocator.descend` (damage-per-byte greedy), `gravity_ir.Program` (complete bytes + `SharedPool`), G070/G074 (matched-budget correction topology + FUNCTIONAL vs MAGNITUDE), `gravity_endpoint_alloc` (per-row two-tier), Q80 identity `complete_bpw = 0.97032·expert + 0.02968·nonexpert`, and `ODYSSEY_POLICY.aggressive_ladder.opportunistic_descent`. It does **not** cartesian-expand `candidate_families.json`.

---

## 0. Search invariants (guardrails as code)

| ID | Invariant | Fail mode if violated |
|---|---|---|
| G1 | `complete_bpw = 8 · B_total / N_source`. Denominator is **source** param count, never candidate DOF (`gravity_bpw`, `gravity_ir`). | Fake density. |
| G2 | `B_total` includes payload + scales + biases + tables/codebooks + offsets + correction + index + tier/router metadata + alignment + container + mandatory reconstruction state. Shared objects counted **once** (`SharedPool`). | Sharing looks free or worthless. |
| G3 | Every discriminator uses **real** forward-pass activations / held-out behaviour. Gaussian/PQ-proxy cosine is not a gate (Qwen3.8 sub-bit collapse). | Proxy trap. |
| G4 | Doctor is authority. A D1 cosine win that fails fast-Doctor is a **kill**, not a floor. | Goodhart. |
| G5 | A component that must expand to dense before compute is not a sub-0.1 win. Count decode scratch + FLOPs/token + name the kernel. | Fake native. |
| G6 | Report `stored_complete_bpw` and `active_bytes/token` as **different** wins. `selected/full` is selection, not cost. | Fake active density. |
| G7 | Query `NEGATIVE_SCIENCE.json` **before** an expensive rung. Predicates, not blacklists. | Rediscovery. |

Rungs are **budgets**, not bit-widths. Naive per-weight bits floor at ~1–2 (`NS-raw-weight-pq-vq-at-one-bit`, `NS-uniform-subbit-allocation`). Sub-0.1 is structural: shared index, low-rank factor, drop-to-constant + sparse repair, procedural seed.

---

## 1. Why cartesian search dies

O005 expert tensors ≈ `48 × 128 × 3 = 18 432`. Policy rungs `R=12`, families `F≈6`. Cartesian `(R·F)^C` is not a search.

`odyssey_candgen.py` already cartesian-expands family grids. That is the thing this lane replaces at the sub-1 frontier.

**Reduction law (deterministic):**

1. **Cluster, don’t enumerate.** Descent units are clusters, not tensors.
2. **Independent geometric descent**, not joint grid. Cost `O(C_cluster · log R)`, not `O(R^C)`.
3. **Compose once**, then repair the interaction (error-chain), instead of searching the product.
4. **Cheap discriminator first.** Full Doctor only at the boundary (policy line 69).
5. **Split a cluster only when measured within-cluster variance demands it** (opportunism).

Target `C_cluster`: organ-class × layer-band ≈ `6 × 4 = 24` on O005, not 18 432.

---

## 2. Component ontology (coarse → fine)

| Level | Unit | When to split | Prior |
|---|---|---|---|
| L0 | Organ class: `embed, attn, router, expert.gate, expert.up, expert.down, lm_head, norm` | Always start here | Organ inversion (gate/up sensitive, down slack); Q80 mixed |
| L1 | Layer-band: `{L0}, {L1–L7}, mid, last}` | If layer-wise D1 or `q_inject` span > ~3× | `R-layer0-different-source`; `gravity_allocator` q_inject 16.1× L0→L63 on Qwen3.8 |
| L2 | Expert-family / index | Iff per-expert D1 variance high **or** routing not near-uniform | `R-heterogeneous-expert-allocate`; `R-uniform-routing-no-cold-compress`. O005: 0 cold, entropy 6.16/7; **do not** start at L2 |
| L3 | Channel-group / row-tier | Endpoint tables; or FUNCTIONAL residual concentrated on ≪10% of channels | AWQ salient channels; SpQR/SqueezeLLM outliers; `gravity_endpoint_alloc` |
| L4 | Single tensor | Only as failure localization after joint compose breaks | Anti-retreat: protect smallest responsible component |

**O005 zero-ablation is too coarse for ranking.** Every organ zeroed → `delta_hits = -10`. A **single hot expert** (id 49, all layers) zeroed → `delta_hits = 0` (MEASURED). Organ-zero cannot rank crushability. Use activation-weighted D1, not organ-zero.

**Crush-order (EV, not sensitivity-first protection):**

```
rank_i = (n_i / N) / (s_i + ε)     # mass recovered per unit sensitivity
```

Descend **high rank first** (big mass, low sensitivity). That is the opportunistic reading of “don’t leave a component conservative if it might tolerate far less.” Sensitivity still **weights** the damage; it does not freeze the component at q4.

---

## 3. Sensitivity (activation-aware, Hessian-class, cheap)

Do **not** rank by `|W|` (G074: MAGNITUDE loses to FUNCTIONAL at equal exception budget).

Per cluster `i`, on held-out captured activations `X` (fit/hold split by prompt, never row-shuffle — G081):

| Symbol | Definition | Source |
|---|---|---|
| `d_j` | `E[X_j²]` per input channel, fit-half only | Wanda / G070 / G074 |
| `H ≈ XᵀX` | Empirical Fisher / OBC Hessian | OBC, GPTQ |
| `q_inject(ℓ)` | Residual-stream gain of a unit error at layer ℓ | `gravity_error_chain` (16.1× on Qwen3.8; **UNKNOWN on Odyssey patients** until measured) |
| `Δy` | `‖X Wᵀ − X Ŵᵀ‖_F / ‖X Wᵀ‖_F` | G069 output-rel-fro |
| `s_i` | `q_inject(ℓ_i) · E[‖Δy_i‖²] / n_i` | damage per element, depth-weighted |

OBC connection: at a fixed budget the cluster subproblem is

```
min_Ŵ  ‖W X − Ŵ X‖_F²    s.t.    complete_bytes(Ŵ) ≤ budget
```

GPTQ/OBC solve this sequentially (Hessian update). We use it **inside** a cluster at a rung (which channels to keep exact, which to crush), not as the outer search. Outer search is geometric descent on the **budget**.

If no capture exists for a site (Qwen3.8 `down_proj` was probe-only in `gravity_allocator`): **flag PROBE-ONLY, do not mix with activation-conditioned scores.** Same honesty constraint.

---

## 4. Algorithm — three phases, no product grid

Rung set (policy):

```
R = [8, 4, 3, 2, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0]
```

`0` means structural collapse (shared index / factor / constant+repair / generated), not “zero bits of a scalar quantizer.”

### Phase A — independent floor finding (opportunistic)

For each cluster `i`, others frozen at the conventional anchor (O005: `q3-g32-experts` / protected q4–q8):

```
lo, hi = 0, anchor_budget(i)
while hi/lo > 1+γ and rungs remain:          # geometric, not linear
    r = next_rung_below(hi)                  # 4→3→2→1→0.5→… not binary on ℝ
    Ŵ = best_codec(i, budget=r)              # candgen ∩ gravity_ir, NS-filtered
    v = discriminate(i, Ŵ, max_level=D2)     # cheap first
    if v.fail:
        Ŵ ← repair(i, Ŵ, leftover=hi-r)      # §6; bytes counted
        v = discriminate(i, Ŵ, max_level=D2)
    if v.fail:
        hi = r                               # this rung is dead for i
        break to boundary protocol
    else:
        lo = r                               # opportunistic: keep pushing
floor_i = last_pass_rung
```

Geometric (rung list) not real-valued binary search: codecs are discrete; a 0.37-bit “width” is not a representation.

**`best_codec(i, r)` menu** (NS-gated):

| Budget r | Admissible | Killed / parked |
|---|---|---|
| 8–3 | Affine grouped (native mlx) | — |
| 2 | Affine grouped; mixed; scale×codec (G129) | — |
| ~1 | Binary/ternary + sparse residual; 1-bit PTQ with FUNCTIONAL islands | Uniform raw-weight PQ/VQ (`NS-raw-weight-pq-vq-at-one-bit`) |
| 0.5–0.25 | Binary + G070 topology; activation-weighted low-rank residual (Q80 `hgravs01_r160`); product-quant only on a **transformed** source | Entropy of Lloyd PQ indices (`NS-entropy-coded-pq-indices`); ternary-vs-VQ (`NS-ternary-factorization`) |
| 0.1–0.05 | Shared codebook **index** + per-site coeff; prune-to-constant + sparse FUNCTIONAL repair | Cross-expert tying unless mean pairwise cosine ≥ 0.10 (`NS-inter-expert-redundancy`, `NS-cross-expert-and-cross-layer-tying`) |
| 0.01–0 | Shared index + sparse delta; dropped-to-constant; generated block (`gravity_ir.generated_block`) | Kronecker except L0 (`NS-kronecker-factorisation`) |

**Opportunistic split.** After a cluster floor is found, if D1 error std/mean across members `> τ` (τ = 0.3 is a prior, not a fact): split into quartiles by `s_i`, re-descend the **tolerant** quartile. This is how a few experts stay rich and most collapse near-zero. Do **not** split if variance is low (O005 routing-uniform prior says this is the default).

### Phase B — joint compose + error-chain localization

Independence is false. Residual-stream error accumulates (`q_inject`).

```
apply all floor_i simultaneously
run D3 fast-Doctor
if pass: go Phase C
else:
    measure error-chain: per-layer residual KL / ‖Δstream‖ on a few real tokens
    localize band ℓ* where error explodes
    promote the smallest sub-component in ℓ* (not the whole organ)
    re-compose
```

Cost: `O(C) independent D1 + O(L) chain probes + k≤8 D3 repairs`. No combinatorial joint grid.

This is the Hawking anti-retreat law (`ODYSSEY_POLICY.failure_policy`): global bit retreat is last resort.

### Phase C — roll-up knapsack (only if total still above target **or** Doctor still fails)

Once every cluster has a **damage curve** `{rung → (complete_bytes, D1, D3)}`, allocation is the same greedy already in `gravity_allocator.descend`, inverted when we need quality back:

```
# surplus: still have budget to give back for Doctor
while Doctor fail:
    give +1 rung to argmax_i  ΔDoctor_proxy_i / Δbytes_i

# deficit: total complete_bpw above target, Doctor holds
while complete_bpw > target and Doctor holds:
    take -1 rung from argmin_i  Δdamage_i / Δbytes_i
```

This is HAWQ-V3’s ILP/knapsack with a greedy substitute (deterministic, no RL/DNAS). Use ILP only if `C_cluster` stays ≲ 64 and damage curves are monotone; otherwise greedy. **Monotonicity is an assumption to verify**, not a theorem (G129: MSE-optimal scale can raise coded bits).

---

## 5. Cheap-discriminator ladder

Escalate only on pass. A fail is a cheap kill.

| Lvl | Probe | Cost class | Can declare a floor? |
|---|---|---|---|
| D0 | Data-free: pairwise expert cosine, Van Loan spectrum, row-norm, kurtosis, NS predicates | Seconds, no forward | **No.** Only skip/kill a family |
| D1 | Single-cluster `output_rel_fro` + cosine on **held-out real** `X` | One GEMM on capture | **No.** Proxy. Required to be real-activation (GLM 0.755 @ 0.167 BPW was real; Qwen3.8 0.5–0.8 BPW “wins” were not) |
| D2 | Organ-local replace; residual-stream drift / token-KL, few tokens | Cheap forward | **No.** Localization |
| D3 | Fast-Doctor battery (existing 12 + refusal) | One generate pass | **Provisional floor** |
| D4 | Full Doctor, held-out capability, not the gate set | Expensive | **Declared floor** |
| D5 | Native kernel + clean-room TPS / `active_bytes/token` | Exclusive GPU | **Executable win** |

**False-win gates at D1:** if capture `X` is gaussian/synthetic, D1 is void. If D1 cosine ≥ 0.75 and D3 `delta_hits ≤ -2`, record `PROXY_TRAP` and kill that codec family on this cluster (Qwen3.8 lesson).

**Doctor-near-boundary rule:** D3 only when geometric search is within one rung of failure, and on the joint compose. D4 once per patient on the best joint candidate, not per cluster.

---

## 6. Boundary repair (matched complete bytes)

When rung `r` fails and `r_prev` held, do **not** step back until these are tried, **in order**, each at leftover budget `bytes(r_prev)−bytes(r)`:

| # | Repair | Selector | Bytes | Prior |
|---|---|---|---|---|
| 1 | Exact island on FUNCTIONAL outliers | `d_j · (W−q(W))²` (G074), not `\|W\|` | `n_exc · (value_bits + index_bits)` — index usually dominates (`gravity_ir.sparse_correction`) | SpQR, SqueezeLLM, OWQ |
| 2 | Topology swap at **equal** bit budget | scattered / row / col / block / low-rank / plane (G070) | Must match leftover; never compare at equal *count* | G070 |
| 3 | Salient-channel protect (AWQ-class) | channels with high `d_j` kept one rung up | channel-index + dual scale stream, counted | AWQ, SmoothQuant |
| 4 | Scale×codec re-opt at fixed **coded** bits | G129 α-grid | scale stream already in complete_bpw | G129, absmax-vs-MSE law |
| 5 | Low-rank residual of `R = W−q(W)` | activation-weighted SVD: `SVD(R D^{1/2})` (G070) | `r·(rows+cols)·16`, no index | LQER, CALDERA, Q80 `hgravs01_r160` |
| 6 | Step back one rung | — | — | last resort |

**Dominance test:** if `complete_bytes(r + repair) ≥ complete_bytes(r_prev)`, repair is dominated. Record and step back. This is how the system is “smart enough to stop.”

---

## 7. MATH — per-component → total `complete_bpw`

### 7.1 Bytes (the only honest identity)

Let `N` = source parameter count.

```
B_private  = Σ_i  B_i^exclusive          # payload, private scales, private index, private correction
B_shared   = Σ_{c ∈ used(SharedPool)} bytes(c)     # counted ONCE
B_meta     = container + catalog + router-meta + alignment + mandatory recon state
B_total    = B_private + B_shared + B_meta
complete_bpw_stored = 8 · B_total / N
```

Per-site node constructors already exist (`gravity_ir`): `quant_tensor` (ceil groups, not floor), `shared_basis`, `sparse_correction`, `exact_island`, `generated_block`.

**Affine complete bits/elem** (counted):

```
bpw_affine(b, g, s=16) = b + s/g + 8·header / n
```

Partial last group still pays a scale (`gravity_ir` ceil).

**Shared-index complete bits/elem** (the actual 0.01 path):

```
bpw_shared(K, d, n_sites, n_i) =
    (n_i · log2(K)) / n_i          + 8 · B_codebook / (N)     [codebook in the SHARED pool]
    = log2(K)   +   8 B_codebook / N
```

If `8 B_codebook / N > 0.1`, a “0.01 component” is a fake win — the mass moved to the pool. **Always report pool share separately.**

**Low-rank complete bits/elem:**

```
bpw_lr(r, m, n) = 16 · r · (m + n) / (m · n) + header
```

For O005 expert `down_proj` `2048×768`, `r=16` → `16·16·(2048+768)/(2048·768) ≈ 0.286` BPW **before** any residual. `r=4` → `0.072`. `r=1` → `0.018`. Sub-0.01 needs `r=0` plus a cheaper term (constant + sparse, or shared factor across sites).

**Sparse repair complete bits/elem** at density `p`, value `v` bits, index `⌈log2(n)⌉`:

```
bpw_sparse(p, v, n) = p · (v + ⌈log2 n⌉)
```

O005 expert tensor `2048×768 = 1.57e6` → `⌈log2 n⌉ = 21`. At `p=0.01`, `v=8`: `0.01·29 = 0.29` BPW. **Index, not values, sets the floor.** G070’s row/block topologies exist because they amortize index. At 0.01 BPW you cannot afford scattered exceptions unless `p ≲ 0.01/29 ≈ 3.4e-4`.

### 7.2 Mass-fraction roll-up (planning identity, not a substitute for G1)

```
complete_bpw ≈ Σ_c  f_c · bpw_c^complete  +  bpw_pool  +  bpw_meta
f_c = n_c / N
```

This is the Q80 identity generalized. **Use it to bound, then confirm with `Program.complete_bpw()`.**

### 7.3 Worked envelopes (DERIVED from MEASURED census, not claimed artifacts)

**O005** (`N = 30 532 122 624`, MEASURED):

| Organ | `f_c` | Notes |
|---|---|---|
| expert | 0.94953 | 95% of stored body |
| attn | 0.02967 | always-on |
| embed | 0.01019 | untied |
| lm_head | 0.01019 | untied; **full GEMV every token** |
| router | 0.00041 | cheap to protect |
| norm | 6.5e-6 | ignore for BPW, protect for quality |

Protected mass `f_prot = 0.05047`. If `bpw_prot = 8`:

```
complete ≈ 0.94953 · e + 0.4038
```

| Expert complete bpw `e` | Stored complete (prot@8) | Clears sub-1? | Clears sub-0.2? |
|---|---|---|---|
| 1.00 | 1.353 | no | no |
| 0.63 | 1.00 | boundary | no |
| 0.50 | 0.879 | **yes** | no |
| 0.10 | 0.499 | yes | no |
| 0.01 | 0.413 | yes | **no** |
| 0 | 0.404 | yes | **no** |

**Stored sub-0.2 on O005 is a protected-organ problem.** Experts at 0 still leave ~0.40 BPW in embed+attn+lm_head@8. Solving `0.94953·e + 0.05047·p ≤ 0.2`:

- `e = 0.01` ⇒ `p ≤ 3.77` bits complete on the 5% tail
- `e = 0` ⇒ `p ≤ 3.96`
- `e = 0.10` ⇒ `p ≤ 2.08`

Sub-0.2 stored **requires** attn/embed/lm_head well below 4-bit complete, not just crushed experts. Endpoint per-row allocation (`gravity_endpoint_alloc`) and attn mixed-precision are mandatory lanes, not polish.

**Q80** (MEASURED identity): `complete = 0.97032·e + 0.02968·p`. At `p=8`, sub-1.5 needs `e ≤ 1.3012` (already the Q80 operating point). Sub-0.2 stored: `0.97032·e + 0.237 ≤ 0.2` ⇒ **`e` would need to be negative** unless `p` also collapses. Same tail bottleneck, worse (only 3% non-expert, but those 3% at 8-bit already cost 0.237).

**GLM 0.755 cos @ 0.167 BPW** is the existence proof that a *component* can sit at ~1/6 bit on **real** activations. It is **not** a total-body complete_bpw claim. Treat as a per-cluster floor prior, not a rolled-up win.

### 7.4 Active roll-up (separate axis)

```
B_active(t) = B_always_on_touched(t) + Σ_{e ∈ top-k(t)} B_e^{in-kernel}
            + gather_index + decode_scratch
active_bytes/token = <REDACTED> B_active(t) ]
```

O005: `top-k/N = 8/128 = 0.0625` (16× **selection**). Census: 3.353B active / 30.5B = 11%.

Per-token touched mass (bf16 census 6.71 GB/token):

| Touched | Approx share of active bytes |
|---|---|
| 8 experts | ~54% |
| attn | ~27% |
| lm_head (full vocab) | ~9% |
| embed (one row) | ~0 |
| router/norm | ~0 |

So **active** sub-1 is *not* automatic from expert 0.1:

```
active_bpw_vs_n_active ≈ 0.54·e + 0.27·a + 0.09·h + …
```

`e=0.1, a=4, h=8` → `0.054+1.08+0.72 ≈ 1.85` (still >1). Active sub-1 needs attn also ≲ 2 complete **and** an in-kernel decode (G5). Active sub-0.2 needs attn+lm_head crushed too.

If the kernel dequantizes selected experts to fp16 **before** GEMM, `B_e^{in-kernel}` is the dense expansion — **not a sub-0.1 win** (G5). Name the kernel; measure DRAM traffic.

Stored vs active are different certificates. A stored 0.50 / active 0.9 NX is a real MoE win even if stored never sees 0.2.

### 7.5 Weighting for the knapsack (Phase C)

Damage used in the greedy step, **not** in the BPW identity:

```
ΔD_i(r → r') = q_inject(ℓ_i) · (dmg_i(r') − dmg_i(r)) · n_i
cost_i        = ΔD_i / max(ΔB_i, 1)
```

BPW weights are mass `f_i`. Quality weights are `q_inject · n_i`. Mixing them is how uniform allocation dies (`NS-uniform-subbit-allocation`).

---

## 8. When to stop

**Per cluster**

1. Floor rung `0.01` or structural-`0` **and** D3 holds, repair dominated.
2. Repair `complete_bytes ≥ r_prev` (dominated).
3. `f_i · (bpw_now − 0.01) < ε_total` with `ε_total = 0.005` BPW prior — further descent cannot move the total.
4. Native kernel missing at this rung: park as `HYPOTHESIS`, **do not** write a floor.
5. NS predicate matches (`reopen_if` false).
6. D1 saturated (next rung’s `Δdmg / Δbits` below noise on hold-out).

**Joint**

7. `complete_bpw ≤ target` **and** D4 holds **and** D5 kernel exists (or is explicitly deferred with reconstruction FLOPs accounted).
8. Error-chain promotions exhausted (`k_max = 8`) and Doctor still fails — **frontier is the last holding compose**. Failure **is** data: write `failure_localization` (organ, layer-band, expert-family, channel-tier).
9. Source-pass budget (policy 512) hit.

**Never stop because** a conventional q3/q4 passed (`CONVENTIONAL_ANCHOR ≠ frontier`). Never stop because uniform-q2 failed (`failure_policy.search_around_failures`).

---

## 9. Cheapest per-component falsifiers

| Cluster | Cheapest kill | What it would prove |
|---|---|---|
| expert.gate | D1 real-X: binary-g vs q2 at matched complete_bpw; if cosine collapse **and** D3 `delta_hits≤-2` | Gate cannot leave 2-bit without repair (organ inversion) |
| expert.up | Same + G074 FUNCTIONAL 1% island vs MAGNITUDE 1% at equal bytes | Whether activation-weighted outliers are the residual |
| expert.down | D1: `r=16` activation-weighted SVD vs binary at matched bpw (Q80 down cleared bar at r160 / 1.27 BPW; **UNKNOWN if r≪16 holds on O005**) | Low-rank residual is/isn’t the down_proj floor |
| expert family sharing | D0: mean pairwise cosine. If `< 0.10`, **skip** shared-basis (NS already killed this on gpt-oss and qwen3-235b). If `≥ 0.10`, reopen | Sharing is dead or live **on this parent** |
| single expert prune | Already MEASURED on O005: zero expert 49, all layers, `delta_hits=0`. Next: zero a **random 8-expert set** (one token’s worth) vs 32 vs 64, D3 | How many experts are jointly droppable before Doctor moves |
| attn | D1 per-proj (q/k/v/o); D3 mixed-q2q4-attn vs uniform-q4. If mixed doesn’t beat uniform, attn is not an allocation lever | Sensitivity-driven attn is dead here |
| embed / lm_head | `gravity_endpoint_alloc` recipe: rank rows by fit-half logit influence; top-f% q4, rest q2; score **held-out** argmax-keep + winner-class cosine. Uniform q3 already failed EOS on Qwen3.8 | Per-row is/isn’t how the 5% tail drops below 4 |
| router | D3: q8 vs q4 vs q2. Router is 0.03 GB on O005 — a pass at q2 is almost free; a fail is cheap protection | `R-protect-router-if-sensitive` |
| L0 vs mid | D1 Shannon-gap / non-Gaussianity. If flat, `R-layer0-different-source` is N/A on this patient | Don’t pay for per-layer codecs |
| joint compose | Apply all independent floors, D3 once. If fail, error-chain — **this is the real falsifier of Phase A** | Independence assumption |

Calibration: routing-frequency allocation needs **≥1000** tokens disjoint from holdout (`NS-calibration-88-tokens`). O005 specimen is near-uniform so that lever is currently N/A (`R-uniform-routing-no-cold-compress`).

---

## 10. Mechanism proposals (required schema)

### M1 · OPDESCENT-COORD — coordinate geometric floors + compose + chain-repair

**One-line.** Independent geometric descent per organ×layer-band cluster, joint compose, error-chain promotion. The search itself.

**complete_byte_accounting.** Each cluster emits a `gravity_ir` site (exclusive bytes) + pool refs. Total = `Program.complete_bpw()`. No cluster may omit scale/index/repair.

**stored_bpw / active_bpw.** Stored = rolled-up complete. Active = gather of the same IR (`Node.active_bytes`). UNKNOWN until a compose exists.

**expected_reachable_bpw.** O005 stored **0.40–0.90** if experts 0.01–0.50 and prot@8 (DERIVED envelope). Stored **sub-0.2 not reachable** without crushing the 5% tail. Active sub-1 **possible** if attn also ≲2 and decode is in-kernel. Sub-0.2 active UNKNOWN.

**quality_risk / limiter.** Joint error accumulation (`q_inject` unmeasured on Odyssey patients). Limiter: attn+lm_head for sub-0.2; expert.gate for sub-1.

**cheapest_falsifier.** One joint compose of independent floors vs anchor, D3. If `delta_hits≤-2` and error-chain cannot localize to ≤2 clusters, independence is dead and we fall back to Phase C knapsack from the **anchor**, not from the floors.

**execution_path.** Native for rungs ≥2 (mlx affine grouped). Sub-1 rungs need Hawking kernels (binary+sparse, shared-basis, generated). Reconstruction FLOPs/elem on `generated_block` / low-rank must be reported. Kernel-bound caveat (`R-sparse-active-expert-gather`): fewer bytes only help if decode is not the wall.

**applicability.** All classes. MoE gets expert clusters + active axis. Dense: no 95% subsidy — every organ is on the critical path (O004). Hybrid: protect SSM state (O001 H2 REFUTED: KV beats SSM past ~1557 tok).

**confidence.** High on search **shape** (allocator + IR + G070/G074 already exist). Medium on O005 stored sub-1 via expert ≲0.63. Low on stored sub-0.2. Label: HYPOTHESIS for numbers not DERIVED from census.

**transfer.** Search transfers; floors do not. Re-measure `s_i`, pairwise cosine, `q_inject` per patient. Two Qwens ≠ universal (`TRANSFER_MATRIX` law).

---

### M2 · OPDESCENT-KNAPSACK — Hessian/sensitivity knapsack on measured curves

**One-line.** After Phase A curves exist, greedy (or ILP) damage-per-byte allocation to a total complete_bpw target — HAWQ-V3 / `gravity_allocator` at Odyssey rungs.

**complete_byte_accounting.** Same IR; objective uses **complete** bytes, not nominal bits (allocator already does this via `quant_tensor`).

**stored / active.** Stored from IR. Active: reweight `ΔB_i` by `P(touched)` for MoE (route map ≥1000 tok, not 88).

**expected_reachable_bpw.** Same envelope as M1; knapsack cannot beat the prot@8 floor ~0.40 on O005.

**quality_risk / limiter.** Non-monotone rungs (G129). Probe-only tensors silently mixed with activation scores (allocator already flags this).

**cheapest_falsifier.** Compare knapsack@target vs uniform at same complete_bpw, D3. If knapsack does not win, sensitivity is not an allocation lever (`sensitivity_driven_alloc` family falsifier, already in `candidate_families.json`).

**execution_path.** Assignment is a `quant_predicate` (native mixed ≥2-bit) plus non-native nodes for sub-1 clusters.

**applicability.** Needs D1 curves. Skip clusters with no capture.

**confidence.** High as an algorithm (exists). Unknown payoff on Odyssey patients (O005 mixed-q2q3 not yet the sub-1 rung).

**transfer.** Curves are patient-specific. The greedy rule transfers.

---

### M3 · EXACT-ISLAND-THEN-CRUSH — FUNCTIONAL outliers held, remainder to 0.01–0.1

**One-line.** At each cluster, peel a FUNCTIONAL exact island (G074), dump the complement to binary / shared-index / constant.

**complete_byte_accounting.** `exact_island` + crushed payload + index. Scattered index `⌈log2 n⌉≈21` on an O005 expert tensor ⇒ scattered `p=0.01` already costs ~0.29 BPW. Must use row/block topology or `p≲3e-4`.

**stored / active.** Island is always-on in the kernel (gather those rows/blocks). Active ≥ stored_island + crushed_selected.

**expected_reachable_bpw.** Per expert cluster 0.05–0.3 if topology amortizes index; **UNKNOWN** until G070-style matched-budget run on O005 real X. GLM 0.167 is the optimistic prior for the crushed complement, not the island.

**quality_risk / limiter.** MAGNITUDE selection (dead here by 10–143× on Qwen3.8 tables). Index cost. Doctor-fail on gate even with islands (F1 `dominant_failure_organ=gate`).

**cheapest_falsifier.** G074 recipe on one O005 `gate_proj` at leftover 0.25 BPW: FUNCTIONAL vs RESIDUAL vs MAGNITUDE, D1 then D3. If FUNCTIONAL does not beat MAGNITUDE, this mechanism is dead on this parent.

**execution_path.** SpQR-style mixed kernel: dense-crushed GEMM + sparse add. If the island is expanded to dense, G5 fail.

**applicability.** All GEMV organs. Endpoints: per-row island (`gravity_endpoint_alloc`) not scattered.

**confidence.** Medium. G074 already showed FUNCTIONAL > MAGNITUDE on Qwen3.8 capture; Odyssey transfer UNKNOWN.

**transfer.** Selector (`d_j · resid²`) transfers; `p` and topology do not.

---

### M4 · SHARED-INDEX-COLLAPSE — component → codebook id (the ~0 path)

**One-line.** If D0 pairwise cosine ≥ 0.10, one shared codebook + per-site indices; site payload → `log2(K)` bits/elem plus **one** pool cost.

**complete_byte_accounting.** `SharedPool.put(codebook)` once; per-site `shared_basis`. Report `bpw_pool = 8 B_codebook / N` separately. Q80 cross-expert: pairwise cosine ~0.004 (gate) / ~0 (up) — **mechanism is N/A on Q80 and likely N/A on O005 until D0 says otherwise**.

**stored / active.** Stored can look ~0.01 if `K` is small and pool is amortized across 128·48 sites. Active still fetches indices + codebook rows touched (codebook can dominate DRAM).

**expected_reachable_bpw.** If cosine gate fails: **0 extra** (skip). If it passes: UNKNOWN. Do not quote 0.01 until pool bytes are in the ledger.

**quality_risk / limiter.** Experts mutually orthogonal (`NS-inter-expert-redundancy`). Shared template energy 0.2513 vs orthogonal null 0.2500 on qwen3-235b.

**cheapest_falsifier.** D0 mean pairwise cosine on O005 expert `gate` at one mid layer. If `< 0.10`, **stop**. Cost: headers + a few tensors, no Doctor.

**execution_path.** Index-gather kernel against a resident codebook. Codebook in SRAM/VRAM working set must be in `B_active`.

**applicability.** Only parents with measured sharing. Dense intra-layer channel tying is a different D0 (channel cosine, not expert cosine).

**confidence.** High that it is **dead on current MoE patients** (O005/O003/O006 0-cold + Q80 structure negative). Live as a **predicate**, not a codec.

**transfer.** The predicate transfers; a positive on one parent does not reopen a negative on another without `reopen_if`.

---

### M5 · DROP-TO-CONSTANT + SPARSE REPAIR

**One-line.** Replace cluster with a broadcast constant (or per-row mean) plus FUNCTIONAL sparse delta. Floor can be ~0 for the payload.

**complete_byte_accounting.** `n_rows · 16` (row mean) + `p·(v+index)` delta. Constant-per-tensor is 16 bits **total**, not per-elem — but the delta index will dominate unless `p` is tiny.

**stored / active.** Stored ≈ delta. Active ≈ delta + the constant (free).

**expected_reachable_bpw.** Per cluster ~`p·29` on expert tensors. To hit 0.01 need `p≲3.4e-4` scattered or a block topology. Joint: O005 single-expert zero already D3-stable (MEASURED) → **some** experts may take `p=0` (true drop). All-expert drop is D3-dead (`delta_hits=-10`).

**quality_risk / limiter.** How many experts can be jointly dropped. Limiter = gate organ of **kept** experts.

**cheapest_falsifier.** Geometric drop of random expert subsets size `{1,8,16,32,64}` (already have size 1). D3. Localize the size where battery moves. Cost: in-place mlx zero, no pack.

**execution_path.** Native: skip GEMM for dropped experts (MoE gather just doesn’t fetch them). True NX win on the active axis even if stored still holds the others.

**applicability.** MoE expert clusters. Dense: drop-to-constant is a prune; needs repair or Doctor dies.

**confidence.** High that **some** experts are droppable on O005 (MEASURED n=1). UNKNOWN n_max. Stored-body ~0 for the whole expert organ is REFUTED by organ-zero.

**transfer.** Re-measure; 0-cold ≠ 0-droppable.

---

### M6 · BINARY BASE + ACTIVATION-WEIGHTED LOW-RANK RESIDUAL

**One-line.** 1-bit (or ternary) base + `SVD(R D^{1/2})` residual at rank chosen to the rung. Q80 mixed family, pushed to 0.01–0.1 rungs.

**complete_byte_accounting.** Binary payload `1 + 16/g` + `16 r (m+n)/(m n)` + any sparse leftover. Q80 numbers (MEASURED screen, **not** packed): gate binary 1.13, up binary+rice-q1@2% 1.29, down r160 1.27, mixed expert 1.23, complete 1.43 at 8-bit non-expert. Those are ~1.2 not 0.1 — the **rung below** is smaller `r` and/or sparser rice.

**stored / active.** Stored as above. Active: binary GEMM + low-rank two-GEMM (`X U`, then `V`). Low-rank **is** native-plausible (two thin matmuls), not an expand-to-dense fake, **iff** we do not materialize `UV`.

**expected_reachable_bpw.** Per down_proj: `r=4` ~0.07 + binary ~1.13 ≈ 1.2 if you **add** to binary; **or** replace binary with low-rank only → ~0.07. Replacement vs additive is the search question. Sub-0.1 almost certainly **replacement** (low-rank *is* the representation), not residual-on-binary.

**quality_risk / limiter.** Down_proj tolerated Q80 r160 at cosine 0.89; gate may not. Additive residual at r≪16 UNKNOWN. Generation not tested (Q80 claim_boundary: packed=false, decode kernel=false).

**cheapest_falsifier.** On one O005 down_proj, real routed X: r∈{1,2,4,8,16} replacement vs binary, D1 cosine vs Q80 bar 0.86 (that bar is Q80-specific; O005 bar UNKNOWN — use relative to q3). If r=4 replacement < q3 D1, low-rank-as-body is dead at 0.07.

**execution_path.** Fused binary GEMM (BitNet-class) or two-GEMM low-rank. Count FLOPs: `r(m+n)` vs `m n`. At r=4, `2048×768` expert, FLOPs drop is real. At r=160, not a win vs binary GEMM.

**applicability.** down_proj first (organ inversion slack). gate/up only with D1 permission. Dense MLP same split.

**confidence.** Medium on down_proj slack. Low on 0.01. Q80 screen is organ-cosine, not Doctor (explicit claim_boundary).

**transfer.** Organ inversion transferred F0→F1; still HYPOTHESIS on Odyssey (`R-organ-inversion` supporting_patients empty).

---

### M7 · TAIL-CRUSH — endpoints + attn, the actual sub-0.2 gate

**One-line.** Per-row two-tier embed/lm_head + mixed attn, because prot@8 already costs 0.40 BPW on O005 and 0.24 on Q80 — stored sub-0.2 is this organ, not experts.

**complete_byte_accounting.** `gravity_endpoint_alloc.bits_per_elem`: `f·(rich+s/g) + (1-f)·(poor+s/g) + 1/H` (the **row-tier bit is counted**). Attn: per-proj affine or shared-q/k.

**stored / active.** Stored: this **is** the 5%. Active: lm_head is a full vocab GEMM (~9% of O005 active bytes) — crushing it helps **both** axes. Embed active is one row: stored win, not active.

**expected_reachable_bpw.** Qwen3.8: uniform q3 tables failed Doctor on EOS; uniform q4 is the floor of the **uniform** ladder. Two-tier was the open question. Odyssey: UNKNOWN. Envelope: if tail goes 8→3 complete, O005 stored drops by `0.05047·5 ≈ 0.25` BPW — enough to put an expert-0.01 body at ~0.16, i.e. **into sub-0.2**. That is the only DERIVED path to stored sub-0.2 on O005.

**quality_risk / limiter.** EOS / winner-rows (Doctor-protected classes). Untied embed+head (Q80 note: sub-bit embeddings catastrophic on GLM R0). Attn error accumulates with depth (`q_inject`).

**cheapest_falsifier.** Endpoint two-tier `f∈{0.02,0.05,0.10}` rich q4/poor q2 on O005 lm_head, D3 argmax-keep + battery. Attn: mixed q2 body / q4 O-proj vs uniform q4, D3.

**execution_path.** Native mixed affine ≥2-bit. Sub-2 on tables is non-native (row-gather two-tier kernel).

**applicability.** Any untied tables. Tied embed/head: count once (`SharedPool`).

**confidence.** High that this is **necessary** for stored sub-0.2 (arithmetic). UNKNOWN whether Doctor allows it.

**transfer.** Table sensitivity has been catastrophic before (GLM R0, Qwen3.8 EOS). Do not assume O005 is kinder.

---

### M8 · ACTIVE-GATHER NX — stored may stay >1, touched/token <1

**One-line.** Same floors, executed as expert-gather. Stored complete may be 1.2–2; active_bytes/token the real NX objective (`arch_objective.moe`).

**complete_byte_accounting.** Stored: full IR. Active: only selected experts’ **in-kernel** bytes + always-on. `selected/full=0.0625` is **not** `active_bpw`.

**stored / active.** Distinct certificates. O005 gravity q3-g32-experts: stored 4.03, active 4.23 (MEASURED) — gather not yet in that number. NX-gather 16× is the lever (`R-sparse-active-expert-gather`). Runtime often **kernel-bound** (A3B tps_caveat): byte cut without a gather kernel will not move TPS.

**expected_reachable_bpw.** Active sub-1: needs gather kernel + expert ≲0.5 **and** attn ≲2 (DERIVED). Active sub-0.2: UNKNOWN, likely needs M6/M7 on always-on too.

**quality_risk / limiter.** Kernel, not representation. Limiter = decode of crushed experts + attn.

**cheapest_falsifier.** Measure `active_bytes/token` (DRAM) on current q3-g32-experts vs a gather implementation. If they match, gather isn’t wired. If DRAM drops 16× and TPS doesn’t, kernel-bound (reopen_if on R-sparse-active).

**execution_path.** Must be native gather. mlx specimen 29.3 TPS ~dense is the negative control.

**applicability.** MoE with `experts_per_tok ≪ experts`. Not dense.

**confidence.** High on the lever’s existence (MEASURED selected/full). Medium on combining it with sub-1 expert codecs.

**transfer.** MoE-universal sparse path (O005/O003/O006). Cold-expert variant does **not** transfer (0 cold on all three).

---

## 11. Prior art map (what we steal, what we refuse)

| Family | Steal | Refuse |
|---|---|---|
| **HAWQ / HAWQ-V2 / V3** (Dong, Yao, Gholami) | Hessian-trace rank; ILP/knapsack under a **complete** cost | Treating Hessian as Doctor |
| **HAQ** (Wang et al.) / **DNAS** | — | RL / differentiable NAS. Search must be deterministic |
| **OBC** (Frantar & Alistarh) / **GPTQ** | Layer-wise `‖WX−ŴX‖_F²` solver as **inner** codec at a rung | Uniform 3/4-bit as the outer search |
| **SparseGPT / Wanda** | `\|W_ij\|·‖X_j‖` as a cheap D0/D1 cousin of FUNCTIONAL | Magnitude-only prune |
| **AWQ / SmoothQuant** | Protect high-`d_j` channels; migrate scale | Assuming activation scale transfer without capture |
| **SpQR / SqueezeLLM / OWQ** | Isolated high-sensitivity outliers as exact islands | Uncounted outlier stream |
| **LoSparse / LQER / CALDERA** | `W≈UV+S` and low-rank **residual of quant error** | Unweighted SVD (G070: use `R D^{1/2}`) |
| **BitNet / 1-bit PTQ** | Binary kernel as a native path at ~1 BPW | Claiming 1.58-bit *is* complete_bpw without scales |
| **QTIP / AQLM / QuIP#** | Incoherence processing + codebook as a **rung-1** codec | Raw-weight PQ at ≤1 BPW (`NS-raw-weight-pq-vq-at-one-bit`) |
| **Product quantization** | On a transformed / residual source | Lloyd indices + entropy (`NS-entropy-coded-pq-indices`) |
| **Hawking** | `gravity_allocator`, `gravity_ir.SharedPool`, G070/G074/G129, Q80 identity, organ inversion, GLM 0.167 real-activation existence | Organ-zero as a sensitivity rank; cartesian `candgen` at 18k tensors |

---

## 12. What the deterministic engine should emit

Not a Grok-in-the-loop. One new family in `candidate_families.json` plus a runner mode:

```
opdescent-{cluster}-{rung}[+repair-{topology}{p}][+protect-{organ}]
```

Receipt fields (machine memory):

- `floor_i`, `codec_i`, `B_i^exclusive`, `pool_ids`, `complete_bpw`, `nominal_bits`
- `stored_complete_bpw`, `active_bytes_per_token`, `selected_full` (not a cost)
- `discriminator_level`, `doctor_delta`, `failure_localization`
- `repair_tried`, `repair_dominated`
- `q_inject_used` (MEASURED / INFERRED / skipped)
- `ns_predicates_fired`
- `kernel`, `decode_flops_per_elem`, `expanded_to_dense: bool`

`candgen.generate` must **stop cartesian-expanding** this family. It calls Phase A–C. `prune` remains Pareto on `(complete_bpw, doctor_delta, active_bytes)`.

Cycle-1 cost model is empty (`ODYSSEY_COST_MODEL` UNKNOWN). Until ≥2 patients of compile-economics exist, cap Phase A at `C_cluster≤32`, D3 at boundaries only, D4 once.

---

## 13. Honest bottom line

| Target | O005 stored | O005 active | What the search must crush |
|---|---|---|---|
| sub-1 | **Plausible** if experts ≲ 0.63 complete and prot stay 8-bit (DERIVED) | Plausible only with gather + attn ≲ 2 | expert.gate limiter; independence of floors |
| sub-0.2 | **Not** from experts. Needs the 5% tail ≲ 4 complete, probably ≲ 3 (DERIVED). Doctor permission UNKNOWN | Needs attn+lm_head crushed and in-kernel decode | M7 is mandatory, not optional |
| per-component 0.01–0.1 | **Some** clusters (down_proj, droppable experts, maybe mid-layer up) | Only if G5 kernel exists | Index cost of scattered repair; NS kills on sharing/PQ |

Push every component. Stop when repair is dominated, the mass-weighted leftover cannot move total BPW by 0.005, or Doctor + error-chain say the limiter is a protected organ. Record the limiter. That record is the frontier.
