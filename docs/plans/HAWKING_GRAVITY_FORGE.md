# Hawking Gravity Forge

> Gravity is the law. Forge creates the forms that can survive it. Doctor repairs the damage.
> Event Horizon proves the lowest complete physical rate where capability remains alive.

Sub-bit-first is non-negotiable. This program builds the missing capability-preserving machinery
**before** any codebase-condensation work. It supersedes the naive weight-space packer as the
authoritative sub-bit line. It does not enable Gravity, emit an Event-Horizon claim, or authorize
any escape — every one of those stays gated.

Status: **B — machinery being built; no passing sub-bit artifact yet, no escape.** (Section 20.)

---

## 0. Audited live truth (2026-07-17, read directly; stale state not trusted)

Full sealed record: [`reports/condense/gravity_forge/FORGE_AUDIT.json`](../../reports/condense/gravity_forge/FORGE_AUDIT.json).

- **Repo**: `main @ 088aa661`, clean except three untracked files (the naive packer suite). One
  worktree. **Zero open PRs** — #23 and #25 are already merged.
- **Processes**: **no hawking worker of any kind is running** — no successor controller, no Doctor,
  and **the old CPU F1 diagnostic is not running** (it already exited; its per-rate checkpoints are
  preserved on disk). The only CPU load is the **separate MoP project** (`mop-final-mechanic-*`,
  ~6 procs at 100%), not hawking.
- **Leases**: `watcher.lease` pid 4333 is **dead/stale**; the **heavy lease is free**.
- **Stale claim reconciled**: `GRAVITY_FRONTIER_ARMED.json` says the heavy lease is *"held by
  doctor_v5_disk25_successor"*. Audited truth: **false** — no such process exists. The real
  remaining launch gate is *no sub-1-bit deployable packer* (now being built) + gravity default-off
  + admission.
- **Resources**: M3 Ultra, 96 GiB RAM, 28 cores (20P/8E), **527 GiB free**, swap 1.25 GB, memory 94%
  free, MPS available (torch 2.6.0).
- **Gravity**: `gravity_enabled=false` (correct). Live parent `120B = openai/gpt-oss-120b`. All five
  parents `GRAVITY_UNINITIALIZED`; the **only representation family present is `scalar_trellis_tqv2`**
  — confirming the directive's complaint that one lineage was being reused. Stress priors: 120B 4/5,
  685B 11/20, 1T 1/3, 1.6T 1/4.
- **Successor queue**: 120B `waiting_adapter`, checkpoint `WAIT_OLD_RELEASE`; blocked on the U8/STR2
  reader, provenance reassembly, the GPT-OSS MoE STR2 loader, the **missing tokenizer**, disk
  retention, and a 0.1-contract adapter that refuses (exit 78).
- **120B source**: **present and verified** — `scratch/staging/gpt-oss-120b.partial/original`, 7/7
  shards, 65.25 GiB, native MXFP4, revision `b5c939de`. The per-expert loader works
  (`gptoss_moe_runtime.load_expert` dequantizes MXFP4→fp32) and a CPU MoE reference forward exists.
  **Tokenizer is absent.**

### Stale-job disposition (Section 0)

The old CPU F1 diagnostic is not running. Its partial checkpoints
(`reports/condense/subbit_frontier/gravity_120b_run/rate_*.json`) are **preserved**. It is
**superseded** by the Forge foundry because weight-space reconstruction is a proxy, not the
capability contract. It is **not** reinterpreted as a final scientific result. No active worker was
interfered with (none exists).

---

## 1. Preserved foundations (classification)

| Item | Class |
|---|---|
| 120B source (7/7 shards, verified) + provenance manifest (543 tensors) | `verified_and_integrated` |
| Per-expert MXFP4→fp32 loader + CPU MoE reference forward (`gptoss_moe_runtime.py`) | `verified_and_integrated` |
| Reduced disk reserve (50 GB) — safe at 527 GiB free | `verified_and_integrated` |
| Gravity policy + Escape-Receipt enforcement (`succ_gravity*`) | `verified_and_integrated` (default-off) |
| Naive sub-bit RVQ packer (`gptoss_subbit_packer.py`) | `experimental_baseline` |
| Naive F1 tournament (`gptoss_gravity_run.py`) + `gravity_120b_run/rate_*.json` | `experimental_baseline` |
| `frontier_giant_scaffold.py` | `experimental_baseline` |
| 120B tokenizer + chat template | `blocked` (absent) |
| GPT-OSS MoE STR2 loader / whole-file U8 reader | `blocked` |

The three untracked baseline files are committed on `codex/gravity-forge` (not pushed) so critical
work is not left uncommitted. None of it is merged or activated.

---

## 2. Corrected scientific interpretation (BASELINE NEGATIVE)

Sealed: [`FORGE_BASELINE_NEGATIVE.json`](../../reports/condense/gravity_forge/FORGE_BASELINE_NEGATIVE.json).

> **BASELINE NEGATIVE**: naive weight-space RVQ and simple shared-codebook expert genome do not
> preserve weight reconstruction sufficiently at sub-bit rates on the sampled GPT-OSS 120B expert
> matrices.

The metric is **weight-space relative Frobenius error — a proxy**. Evidence (real experts, blocks
0/35): 4/5 → collapse 0.63, 1/1 → collapse 0.58, 2/1 → degraded 0.36. This is **not** the 120B
Event Horizon; it does **not** authorize a Gravity escape; **no** Telegram Event-Horizon claim was
ever emitted (the run stopped at per-rate checkpoints). It does not prove capability failure, nor
that activation-aware / learned-transform / structured-binary / shared-grammar / progressive-slice /
sparse-exception / QAT / generated-weight representations fail.

The adversarial review (Section 16) found the superseded baseline `gptoss_gravity_run.py` still had
an **armed-by-default** path that would have emitted an "Event Horizon 2.0 BPW" Telegram from this
proxy (it did not fire only because the run was interrupted). That path is now **neutralized**:
proxy-explicit field names, no `gravity_event_horizon` emit (F1-proxy `gravity_feasibility_completed`
only), requires true `survives` not `degraded`, notifications off by default.

---

## 3. Codebase-condensation program: FROZEN

Deferred behind the Forge readiness gate
([`FORGE_READINESS.json`](../../reports/condense/gravity_forge/FORGE_READINESS.json)). Permitted
before the gate passes: census, no-touch manifests, read-only dependency graphs, duplication
analysis, doc classification, condensation *planning*, cleanup of clearly-generated artifacts. **No
structural repository condensation.**

---

## 4. Sub-bit target (whole-artifact)

Sub-bit ⇔ `R_whole_artifact < 1.0`, where the complete physical rate counts **everything**: base
representation, codebooks, factors, expert coefficients, Doctor corrections, protected islands,
sparse exceptions, pass-through tensors, routers, lexical tensors, shared experts, indices,
metadata, alignment, packaging, mandatory runtime tables. Nominal body BPW is never the claim. For
giant sparse models we track installed / resident whole-artifact BPW, active bytes/token, bytes
moved/token, generated working-set bytes; the **primary Gravity claim is installed whole-artifact
BPW**. The foundry's `ByteLedger` enforces this: it itemizes every component and always adds a
non-zero metadata charge, so no family can hide overhead in "free" metadata.

---

## 5. The Forge foundry (`tools/condense/gravity_forge.py`)

Materially-distinct sub-bit-capable lineages, each with exact whole-artifact accounting, on MPS,
reading the real experts:

- **A. `transform_pq`** — seeded randomized-Hadamard incoherence rotation (regenerable from an
  8-byte seed → the rotation is free beyond the billed seed) + product quantization. Sub-bit when
  `dim > subspaces·log2(k)`.
- **B. `shared_expert_grammar`** — one additive codebook fit on the pooled vectors of an expert
  *cluster* (billed once, amortized) + per-expert indices + optional per-expert low-rank correction.
  The MoE amortization lever; a test proves a larger cluster lowers whole-artifact BPW.
- **C. `repairability_shaped`** — deliberately cheap base (coarse VQ) + structured Doctor correction
  (low-rank residual + sparse outlier rows), billing `R_base` and `R_Doctor` separately.
- Baselines **`naive_rvq`** and **`low_rank`** re-implemented for matched-byte comparison.

Not yet implemented (next lineages, Section 5 of the directive): learned (billed) rotations,
binary latent factorization with shared bases, progressive semantic slices, generated-weight
decoders. Design intent recorded here; do not call an RVQ variant a new family.

### First tournament results (real GPT-OSS-120B experts, blocks 0/35, expert 0)

Sealed: [`FORGE_FRONTIER.json`](../../reports/condense/gravity_forge/FORGE_FRONTIER.json).

At matched deep sub-bit whole-artifact BPW (0.5 and 0.8), **every family collapses in weight space**
(rel-err 0.67–0.94). `repairability_shaped` reaches **0.18–0.37 whole-BPW** by amortizing
corrections but still collapses in weight reconstruction. The best sub-bit weight error is
`shared_expert_grammar @ 0.78 BPW ≈ 0.67`. **Output-space** divergence (proto-F2, reference forward,
the routed experts actually exercised, synthetic activations) for `transform_pq @ ~0.75 BPW` is
**~0.61 mean** — consistent with the weight-space collapse.

Reading: naive **and** first-wave advanced weight-space families collapse sub-bit on these
high-entropy experts. Per the doctrine this is negative evidence about *weight-space* objectives,
not proof of impossibility. The lever now moves to **optimization against output/activation/capability
objectives** (Section 6) and **Doctor rescue at F2** — both of which need runtime pieces that are
partially blocked (tokenizer, execute-from-compact path).

---

## 6–8. Objectives, F2 runtime, Doctor (build order)

- **6. Objectives**: weight-space is diagnostic only. Built now: output-space divergence via the
  reference forward. Next: activation-aware loss (needs representative activations → tokenizer),
  output/logit/KL/top-k/router-stability, causal-capability probes, and bounded source-streamed
  distillation/QAT (reconstruction/output/activation/router/blockwise/codec-native/Doctor-fit).
- **7. F2 runtime**: the numpy reference forward already runs packed experts (output divergence).
  Missing for real F2: execute-from-compact Metal kernels (transform, codebook lookup, binary-factor
  GEMV, additive reconstruction, sparse-exception application, Doctor correction), the tokenizer +
  chat template, and end-to-end short-context parity. **No F1-only result authorizes escape.**
- **8. Doctor**: `repairability_shaped` is the first real treatment (executable adapter, bytes
  counted). It is not "real" per Section 8 until it re-runs at F2. Registry-only treatments stay
  blocked from scheduling.

---

## 9. Gravity law (unchanged, not weakened)

Sub-bit-first search stays mandatory. The naive packer result must not request an Escape Receipt.
For 120B, mandatory coverage now requires ≥2 advanced Forge families at F2, ≥1 real Doctor rescue at
sub-bit, whole-artifact accounting, a capability evaluation, and a structural diagnosis. Escalation
order: same rate → stronger representation → better transform → different byte allocation → Doctor
treatment → only then a higher rate. Giant parents start at their configured sub-bit stress rates;
priors (120B 4/5, 685B 11/20, 1T 1/3, 1.6T 1/4) recompute after real fixed overhead + Doctor reserve
are known.

---

## 10–11. 120B role & giant parents

120B is the **calibration and machinery parent**, not the final product. Giant sparse parents
(685B DeepSeek-V3.2, 1T Kimi-K2.6, 1.6T DeepSeek-V4-Pro) are the primary product tests; their Forge
adapters are scaffolded from bound source authority + synthetic twins (bounded source windows only,
no full downloads), architecture-compositional, not one script per model. **Not yet started.**

---

## 12. M3 Ultra harness

Atlas: [`FORGE_RESOURCE_ATLAS.json`](../../reports/condense/gravity_forge/FORGE_RESOURCE_ATLAS.json).
Measured: k-means/assign/Hadamard run on MPS; **`linalg_svd` falls back to CPU** (unsupported on MPS
in torch 2.6.0) so the SVD-heavy families are CPU-bound; expert dequant is numpy/CPU. Honest gaps:
GPU utilization % not yet instrumented; no pipeline overlap yet. Note: the CPU is currently contended
by the separate MoP project.

---

## 13. Authoritative run

Contract: [`FORGE_RUN.json`](../../reports/condense/gravity_forge/FORGE_RUN.json). The tournament ran
**synchronously and bounded** (`gravity_forge_run.py`, sealed cells) — not a daemon, no admission
bypass. The next step is to own it as a real successor-queue row (event-sourced, lease-controlled,
resumable, Telegram-visible, exact-rate + family + source-manifest + Doctor-policy bound,
resource-admitted). Sequence: 120B F2 machinery → 685B → 1T → 1.6T.

---

## 17. Readiness gate (blocks condensation)

`hawking.gravity_forge.readiness.v1` — currently **FALSE**:

| condition | value |
|---|---|
| `subbit_packer_real` | ✅ (foundry packs+round-trips real experts, exact accounting) |
| `packed_runtime_real` | ❌ (no execute-from-compact Metal path) |
| `f2_parent_bound_evidence` | ❌ (output divergence uses synthetic activations) |
| `doctor_treatment_real` | ❌ (applies+bills at F1; F2-rerun pending) |
| `gravity_controller_integrated` | ❌ (not yet a successor row) |
| `giant_adapter_contracts_stable` | ❌ |
| `active_campaign_safe` | ✅ |

**Codebase condensation remains BLOCKED.**

---

## Exact next command

```
# widen the Forge tournament + add activation-aware objective once a tokenizer is bound:
python3.12 tools/condense/gravity_forge_run.py --blocks 0,12,24,35 --experts 0,1 --cluster-size 8
```

Then, in priority order: (1) bind the 120B tokenizer + chat template to unlock real-activation F2;
(2) implement execute-from-compact Metal kernels for `transform_pq`/`shared_expert_grammar`;
(3) wire `gravity_forge_run` as a successor-queue row; (4) add binary-latent-factorization and
learned-rotation families; (5) scaffold the 685B Forge adapter from bounded source windows.

---

## CLEAN SLATE Stage A — COMPLETE (pre-run readiness DERIVES to PASS)

Program: `HAWKING CLEAN SLATE.md` (Stage A prep + freeze; Stage B condensation gated). Fresh §0
audit: no heavy hawking process (only the separate MoP project on CPU); Forge `verified_but_local`
on `codex/gravity-forge`. **§4 contradiction resolved**: the 120B tokenizer is **present and valid**
(`scratch/staging/gpt-oss-120b.partial/tokenizer.json`, vocab 200019 o200k_harmony, round-trips,
`chat_template.jinja` present) — the earlier "absent" claim was stale.

Stage-A deliverables (this session):
- **§5 four materially-distinct families**: added `ternary_factor` (ternary latent factorization,
  billed 2 bits/elem, conservative) to `transform_pq` / `shared_expert_grammar` /
  `repairability_shaped` (+ naive_rvq / low_rank controls). `families_available = 4`.
- **§6 real-token F2 fixture** (`forge_f2_fixture.py`): tokenizes real prompts, embeds with the
  model's `embedding.weight`, runs the reference MoE with original vs packed experts. Real inputs
  route to **74 experts** (vs 11 synthetic) and sub-bit `transform_pq` gives **mean output
  divergence 1.26** (max 1.97) — far worse than the synthetic 0.61. Honest boundary: real
  token-embedding activations, a pre-attention proxy; true residual-stream F2 needs the block
  attention layer (tracked, not hidden).
- **§8 controller integration** (`succ_gravity.materialize_forge_program` +
  `forge_controller_integration.py`): the ONE controller materializes a sealed, source-bound Forge
  program (`forge_subbit`, `gravity_forge:transform_pq`), registered in `GRAVITY_STATE.json`, and
  **refuses to launch it** (default-off + admission not passed). Launch disabled.
- **§9 giant adapters** (`forge_giant_adapters.py`): 685B / 1T / 1.6T adapter contracts composed
  from shared primitives out of the read-only source authority (no downloads); all contracts valid.
- **§10 AUTO-DERIVED pre-run readiness gate** (`forge_pre_run_readiness.py`,
  `hawking.gravity_forge.pre_run_readiness.v1`): computes all 12 conditions from live probes (not
  static JSON). **Result: PASS (12/12), blocking: []** -> authorizes codebase condensation (Stage B).

Honesty guard: `compact_runtime_fixture_green` / `doctor_fixture_green` certify the measurement
apparatus RUNS (bounded, deterministic, finite) - NOT that sub-bit packing passes a capability bar.
The real science remains negative (severe output divergence). This gate authorizes **condensation**,
not the heavy run (§27 heavy-run readiness is separate and not yet derived). 48 tests green.

Stage B (repository condensation, §11-27) is now **authorized but not begun**: it is a multi-week
descent (175k->50k) whose every checkpoint needs a commit+tag, and the standing house rule forbids
commits without explicit approval. Held for the go signal.

### Stage-A hardening (after committing Stage A @ `0eceb10a`)

- **True residual-stream F2** (`gptoss_block.py`): built the block-0 attention forward (GQA 64/8,
  RoPE theta 150000, per-head attention sinks, causal, then residual + mlp.norm) so F2 sees the
  genuine post-attention MoE input instead of raw embeddings. **This corrected the science**: on the
  true residual stream sub-bit `transform_pq` output divergence is **0.69** (max 0.92) - vs the
  token-embedding proxy's *pessimistic* 1.26 and the synthetic 0.61. The gate's
  `compact_runtime_fixture` now runs this faithful residual fixture. Honesty: from-config attention,
  not HF-parity-validated; the approximations largely cancel in the relative orig-vs-packed divergence.
- **Forge program wired as a live successor queue row** (`forge_controller_integration.register_queue_row`):
  the sealed Forge program is now a candidate on the live 120B successor row (drain/resume via the one
  controller), still launch-disabled and refused.

Still negative science, still gated: 0.69 output divergence at ~0.75 BPW is a large capability
perturbation. No escape, no Event Horizon, heavy-run readiness (§27) not derived.
