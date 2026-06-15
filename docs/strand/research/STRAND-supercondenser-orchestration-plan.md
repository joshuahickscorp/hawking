# STRAND Supercondenser — Custom-Solution Orchestration Plan

_Authored 2026-06-13 for the executor/orchestrator chat. This is a HANDOFF artifact: act on it cold._
_Grounded in three parallel research legs (codebase cartography + quantizer/rotation SOTA + training/distillation SOTA, 2026-06-13) and the live sprint doc (`STRAND-supercondenser-sprint.md`)._
_Owner standing order honored: every claim is labeled **[proven]** / **[measured-live]** / **[modeled]** / **[pending]** / **[lit]** (from literature). No soft-positives._

---

## §0 — Rules of engagement (invariants EVERY wave respects)

1. **The moat: bit-identical, float-free integer/LUT decode.** A lever is admissible ONLY if decode stays integer/LUT and identical across devices. Encoder-side / training-side learning is unconstrained **iff** it freezes to integers (trellis indices, Q12 LUT, scale ints) used at decode. Float rotations at decode, runtime-data-dependent codebooks, FP16/f32 value generation in the decode kernel → **moat-dead, auto-reject.** [proven: decode float-free confirmed — `decode.rs:33-97` pure integer; only float is `decode_tensor:289-294` final `q*Q12_TO_F32`]
2. **Ownership / anti-collision.** Two chats share one tree/pod/PV. The **executor chat owns**: all `lib.rs` edits, the shared section-walkers, cloud-arming, and the pod. Any **concurrent file edits run in an isolated git worktree** (separate `target/` so a rebuild can't collide with the live PV requant cadence). Edits to shared walker/provenance files are **serialized**, never parallel.
3. **Box / PV co-tenancy (the freeze trap).** PV (pid 28690, MPS, ~ends Jun-14) **OWNS the Apple GPU**. No second MPS job — co-tenancy hard-rebooted the box twice. While PV runs: local work is **CPU-only, low-RAM, `-j4`**, and any Rust rebuild is **worktree-isolated**. Single-box ⇒ all MPS A/Bs are a **serial queue**, not concurrent.
4. **Cloud gating — no premature spend.** Cloud selective-PV is self-gated (`cloud-selective-pv.sh require_promoted`, lines 199-207): refuses unless `$GATES/pv-kl-routed.json` has `promotion.state == PROMOTE_CLOUD`. The gate flips only when the local PV beats the **26.77** floor (`promote.py:PRIOR_PV_FLOOR`, PROMOTE iff ppl≤30 AND <26.77). **Do not bypass.**
5. **Density ≠ quality. Never report a combined "smaller AND better" from one arm.** The KL-route buys 2-bit QUALITY at 3-bit DENSITY (it lifts v/up/gate to 4-bit). Density (→2.40 bpw) comes from C2/DBIA **separately**. Two headlines, two numbers, always.

---

## §1 — The thesis: what the custom-better STRAND solution actually is

**The research convergence (3 independent legs agree):** STRAND already owns the *structural* substrate that the best 2-bit systems use — a bitshift-style **trellis** (QTIP-class), **RHT** incoherence, a **frozen Q12 LUT**, an outlier channel. On the two structural axes STRAND is **already near-optimal and should NOT be touched**:

- **Rotation is not the bottleneck.** RHT is an orthogonal similarity transform; a *learned* rotation (ButterflyQuant/SpinQuant) only helps when paired with a *weak* rounder (their 2-bit Llama2-7B is 15.4/16.4 PPL — RTN-regime), and both apply **float rotations to activations at decode → moat-dead.** [lit: arXiv 2509.09679, 2405.16406]
- **Codebook geometry is not the bottleneck.** On the post-RHT Gaussian source, scalar **trellis MSE 0.069 already beats E8-lattice 0.089** and sits within ~10% of the R-D bound 0.063. A vector/E8 codebook would *regress*. [lit: QTIP §2]

**So the entire 0.31→≤0.15 loss-tax gap lives in two things STRAND is currently NOT doing well:**

| Gap | Evidence | STRAND-native fix |
|---|---|---|
| **STE rounding stalls at 2-bit** | PV-Tuning proves STE's `(1/L)∇φ` update is often *smaller than one code increment* ⇒ the optimizer never crosses a code boundary ⇒ stalls. PV-tuning's discrete step → **2-bit tax 0.104** (best in field). [lit: 2405.14852] STRAND's PV is STE-style today (`strand-qat.py` delta-forward, `QuantLinear.forward:79-85`). | **STRAND-PVT** (lever Q1): port PV-Tuning's V-step to STRAND's **trellis paths** (not AQLM additive codebooks) — periodically re-pick the Viterbi path of the top-τ highest-gradient weights against the teacher, interleaved with continuous scale/LUT P-steps. Freezes to existing integer indices. **Novel:** PV-tuning was built for additive VQ; the trellis port is STRAND's own. |
| **Bit budget mis-allocated** | KL-Lens proves **`tax = ln(PPLq/PPLbf16) = D_KL(p‖q)` exactly** ⇒ routing bits to minimize output-KL *is* minimizing the loss-tax (not a proxy). [lit: 2604.13440] | **KL-route** (Q2, already built `rung-kl.py`): STRAND measured the real per-class KL — **up_proj 0.0099 / v_proj 0.0092 / gate_proj 0.0066 highest; down_proj LOWEST 0.0028** [measured-live `rung-kl-dp_d4_r2.json`]. This **inverts the naive "protect down_proj" prior**. `cloud-selective-pv.sh:96` already routes `v|up|gate`. |

**The custom solution = STRAND's near-optimal trellis/RHT/LUT substrate + the training-side machinery it's missing, all freezing to the float-free integer container.** That's the differentiator from every reference method (which tolerate float codebooks/rotations the moat forbids). Concretely, the stack is **STRAND-PVT (Q1) ⊕ KL-route (Q2) ⊕ EfficientQAT scale-refine (Q3) ⊕ 2-bit relation-distill (Q4, A/B) ⊕ WSD cooldown (Q5)**, with **function-projection (Q6) gated on a cheap falsification**, and **density (C2/DBIA) run as a decoupled parallel track**.

---

## §2 — Lever ledger

Expected-Δtax is a **[modeled]** estimate unless noted. "Side" = where it hooks. "Moat" = decode/attestation risk.

### Quality levers (close the loss-tax gap)

| ID | Lever | Side | Hook (path:line) | Moat | Exp. Δtax | Gate / kill |
|---|---|---|---|---|---|---|
| **Q1** | **STRAND-PVT** discrete V-step over trellis paths | training | `strand-qat.py` step loop + optimizer `:576`; calls encoder Viterbi `encode.rs:viterbi_path_buf:1149` on top-τ weights | **SAFE** (freezes to int indices) | **0.31→~0.15-0.18** (largest) | 0.5B A/B vs STE-only PV; promote if beats 26.77 floor |
| **Q2** | **KL-routed** mixed-precision allocation | encode/alloc | `rung-kl.py`→`configs/mp-kl-routed.json`→`strand-qat.py --pv-tensors:443`; gate `gates/10-pv-kl-routed.sh` | **SAFE** (per-tensor int decode; SDSC records per-tensor bits) | large (main cloud arm) | already in gate chain; ⚠ lifts deploy to 3.2-3.8 bpw |
| **Q3** | **EfficientQAT scale-refine** (freeze ints, train only scales E2E under LM loss) | training | `strand-qat.py` — new final phase, only scale params `requires_grad` | **SAFE** | ~0.5 PPL (cheap, stacks) | 0.5B A/B; [lit: 2407.11062 worth ~0.5 PPL, no KD] |
| **Q4** | **2-bit relation-distillation** (MiniLM Q-Q/K-K/V-V relation KL, last layer→last L/4) | training | `strand-qat.py` KD block `:612-635` (**greenfield** — logit-KL only today); add teacher+student fwd hooks | **SAFE** | **uncertain — A/B** | A0/A1/A2 + A3 neg-control (FitNet MSE, expect-to-hurt). KILL if A1 ≯ A0 by ≥0.02 tax |
| **Q5** | **WSD cooldown** (`--cooldown-frac 0.2` linear; freeze codes after cooldown) | training | `strand-qat.py:393-397` (flag exists); `pv-recipe.sh` sets it | **SAFE** | 2-5% final-loss [lit]; A/B for QAT | the existing pv-recipe variable |
| **Q6** | **Function-projection** (Wolf-Ziv): quantize `Ŵ=P_F·W`, error→loss null-space | encode-only | `encode.rs:gen_step_dist!:1248-1249` (the weight-MSE `diff*diff` objective) + `choose_scale_q:833`/`choose_sub_scales:863` | **SAFE** iff P_F folded encode-time, NOT at decode | **FALSIFY FIRST** | see §2.1 |

**§2.1 — Q6 falsification (Wave 0, CPU, ~1 day) — the rotation fear is RESOLVED, so the probe is worth running:**
- **Rotation does NOT kill the lever.** RHT is orthogonal similarity `H̃=VᵀHV` ⇒ **eigenvalues/rank exactly preserved**; whitening rotates eigenvectors, not the spectrum. Low-rank stays low-rank. [lit: QuIP# Alg.3, verified linear algebra]
- **The REAL risk is over-sampling N** (empirical Fisher `F̂=(1/N)Σggᵀ` is rank≤N; too many tokens ⇒ rank climbs to identity ⇒ lever dead).
- **Protocol:** per-256-block **TRUE Fisher** (sample labels `ŷ∼p_θ`, NOT grad² on observed labels — empirical Fisher is a biased curvature proxy and is exactly what likely sank Hessian-Viterbi). Eigendecompose (h=256, cheap). `r99 = min{k: Σλ≤k ≥ 0.99Σλ}`.
- **KILL if** median `r99 > h/2` **OR** `r99` grows ~linearly with N (sampling artifact, not curvature). **Correctness check:** post-RHT `r99` must be unchanged (similarity-invariance).
- **Why it's NOT doomed by the Hessian-Viterbi backfire (+1.1%):** that lever reweighted *coordinate-aligned* curvature, which RHT flattens; Q6 operates in the *eigen-basis* (rotation-invariant) and is immune to that specific failure — provided true Fisher + rank test pass. [lit synthesis]

### Density levers (decoupled ship-side track → 2.40 bpw; moat/attestation discipline required)

| ID | Lever | Side | Moat | Notes |
|---|---|---|---|---|
| **D1** | **C2/SDSQ** whole-model sideinfo coder (entropy-code scale_q + sub_scale) | encode | **SAFE** (recon byte-identical; decode losslessly overwrites `BlockMeta.scale_q`) | SDSQ wired (`append_sdsq:~1191`). Stronger **`c2_final.rs` is ORPHAN** (not in `lib.rs`). **Must concatenate model-wide** — per-tensor 256-block recovers only 0.053; whole-model amortizes the table → **2.665→~2.43** [measured: 87,381-stream byte-exact cert] |
| **D2** | **DBIA** de-bias deploy (sealed `.strand` section) | **attestation + DECODE** | **HIGHEST RISK** | −28.7% q2 / −10.9% mixed [measured-live `debias-ppl-dp_d4_r3-ab.json`: 19.08→16.99, LOCAL_PASS]. Today a **JSON sidecar applied only by the Python gate harness — UNSEALED (attestation hole) + not in shipping decode.** Strict order in §2.2 |

**§2.2 — D2 (DBIA) deploy order — DO IN THIS SEQUENCE or it silently breaks the moat:**
1. **SEAL FIRST.** `provenance.rs:descriptor_digest:122` add `dbia:&[u8;32]` slot + a `dbia_digest()` beside `outlier_digest:97`; `provenance_io.rs:live_descriptor_digest:26` + `verify_archive_with:520` read+check DBIA (mirror OUTL exactly). _Else a tampered de-bias add passes `verify_archive`._
2. **Teach ALL walkers + the registry together.** `outlier_wire.rs:read_outl_bytes:440` (steps over only {SPRV,SDSQ} today), `sideinfo_wire.rs:read_sdsq_bytes`, and `selfdesc` `chain_scan` must step over `DBIA`; add `DBIA` (and `SDSQ`) to `format.rs:section_tag:170-175`. _Else OUTL→DBIA→SPRV order makes `read_outl_bytes` halt on the DBIA magic → silent outlier-drop at decode._
3. Wire `debias_wire.rs` into `lib.rs` (orphan→live).
4. Producer: `append_dbia` in `quantize-model.rs` **between OUTL (:1168) and SPRV (:1212)** (all data sections before the SPRV seal; **RSLT stays outermost**). Bump **SDSC** in `selfdesc.rs` for the layout change.
5. Deploy-apply: `loader.rs:from_mmap:51` read DBIA; apply `c` in `outlier_mac.rs` MAC epilogue (`matvec_rht:121`). **Un-ignore** `decode-kernel/tests/debias_decode_apply.rs` (it's the API blueprint, sketch at :537-546) and `sprint_section_chain_audit` DBIA arm.

### Infra / enabling levers

| ID | Lever | Side | Why | Hook |
|---|---|---|---|---|
| **I1** | **bf16 shadow + 8-bit Adam** | training | **GATES the 32B leg** — fp32 shadow = ~238 GB @ 32B → OOM (same class as the 70B encode OOM) | `strand-qat.py:QuantLinear.weight:65` fp32→bf16; `AdamW:576`→bitsandbytes `Adam8bit`. **Both greenfield** (no bf16 shadow, no bitsandbytes import) |
| **I2** | **CUDA block-batched encode** (validate-only) | pod | unblocks >32B/70B/405B GPU encode (per-thread host levels 30 GB→16.8 MB @ batch512) | kernels exist, `#[ignore]`d; CPU stays canonical; parity oracle = **f32 CPU reference** |
| **I3** | **Staged-tile bitslice decode** | decode/GPU | speed (≥90% mem-BW), NOT quality | GPU-blocked until PV exits; out of this plan's scope, listed for non-collision |

---

## §3 — Wave plan (parallel where the box allows; critical path = the PV clock)

```
CRITICAL PATH:  PV (pid 28690, MPS) ──► pv-dp.json ──► gate-10 KL-routed PV ──► promote.py (PROMOTE_CLOUD iff <26.77) ──► cloud 7B ──► cloud 32B(after I1)
                └─ everything MPS queues behind this ─┘
```

**WAVE 0 — NOW. CPU-only, zero-MPS, zero-shared-tree. Runs concurrently with the live PV.**
- **0a. Q6 Fisher falsification** (§2.1) — pure CPU analysis, reads grads, touches nothing shared. Output: PASS/KILL for the function-projection lever.
- **0b. Density-track de-risk in an ISOLATED WORKTREE** (separate `target/`): wire `c2_final.rs` + `debias_wire.rs` into `lib.rs`, build, run the orphan tests (`c2_final_harness`, `debias_determinism`) live for the first time. Proves the orphan→live edits compile + pass **before** they touch the main tree. **Worktree-isolated so the rebuild cannot collide with PV's requant cadence.**
- **0c. STRAND-PVT + relation-distill + EfficientQAT-phase implementation specs** authored as new-file design docs / a feature branch — **do NOT edit the `strand-qat.py` the running PV loaded.** Ready-to-merge when the box frees.

**WAVE 1 — Quality recipe assembly. MPS. Fires when the current PV drains (~Jun-14). SERIAL on-box queue.**
Each is a 0.5B A/B isolating one variable; `promote.py` scores it. Winners compose the cloud recipe. Order by leverage:
1. **Q1 STRAND-PVT** vs STE-only PV ← the big one; if it beats 26.77, it also flips the cloud gate.
2. **Q2 KL-routed PV** (`gate-10`) — consumes Q1; the existing promote path.
3. **Q3 scale-refine** A/B (stacks on the winner).
4. **Q4 relation-distill** A0/A1/A2/A3 (kill if A1 ≯ A0 by ≥0.02).
5. **Q5 cooldown** A/B (the pv-recipe variable).
> Honest constraint: single MPS ⇒ these are **sequential**, ~hours each. Parallelism here is in the *analysis* agents, not concurrent training.

**WAVE 2 — Density deploy. HOT Rust edits. After PV exits (so rebuilds are safe); overlaps Wave-1's MPS queue (CPU/build work).**
- **2a. D1 C2/SDSQ** whole-model concat (wire `c2_final`, add `append_c2`/`read_c2` mirroring `append_sdsq`, SDSC bump, step-over sets, `format.rs:section_tag`). → 2.665→~2.43.
- **2b. D2 DBIA** deploy in the strict §2.2 order.
> 2a and 2b **both edit the shared walkers** (`read_outl_bytes`, `read_sdsq_bytes`, `section_tag`) — **serialize them**, single owner, run `sprint_section_chain_audit` after each step.

**WAVE 3 — Cloud. Gated on `PROMOTE_CLOUD`. No premature spend.**
- **3a. Pre-flight** (partly done): rebuild on-pod `quantize-model` (was 937-byte broken), stage 7B shards. **Land I1 (bf16 shadow + 8-bit Adam) before the 32B leg.**
- **3b. 7B selective-PV** with the assembled recipe (KL-route ⊕ STRAND-PVT ⊕ relation-distill-if-passed ⊕ cooldown ⊕ DBIA). **CPU-canonical** (CUDA validate-only). Success = `ppl_selpv_7b.json` tax **≤0.15-0.18** [modeled].
- **3c. 32B selective-PV** — ONLY after I1. `touch /workspace/SKIP-32B` until then.

**WAVE 4 — Speed (GPU-blocked, after PV exits; independent track).** I3 bitslice staged decode + I2 CUDA encode validate. Listed for completeness / non-collision; not on the quality/density critical path.

---

## §4 — Agentic execution shape (how to fan this out)

| Wave | Parallelism | Agent roles | Isolation |
|---|---|---|---|
| 0 | **3 parallel** | (a) Fisher-probe [CPU], (b) density-worktree wirer [build+test], (c) spec-author [new files] | (b) **worktree**; (a)(c) read-only/new-file |
| 1 | **serial queue + parallel analysis** | 1 box-conductor (runs the MPS A/Bs in sequence, self-guards on `pgrep`), 1-2 analyst agents scoring via `promote.py` as each lands | none (single box) |
| 2 | **serial on shared files** | 1 section-chain owner (the HOT edits in §2.2 order), 1 auditor running `sprint_section_chain_audit` + `verify_archive` after each step | **worktree** per editor; merge in strict order |
| 3 | **gated, sequential** | 1 cloud-conductor honoring `require_promoted`; 7B before 32B | pod |
| 4 | parallel after PV exits | bitslice + CUDA validators | pod / GPU |

**Collision protocol restated:** executor chat owns `lib.rs` + walkers + cloud + pod. All concurrent edits → worktree. Shared walker/provenance files → serialize, never parallel. Rebuilds while PV runs → worktree `target/` only.

---

## §5 — Decision gates, promotion, kill conditions (consolidated)

- **Promotion grammar** (`promote.py:43-48,95,120`): `PV_PROMOTE_PPL=30`, `PV_WEAK_PPL=36`, `PRIOR_PV_FLOOR=26.77`, `DEBIAS_ADOPT=0.995`. `gate_pv` → `PROMOTE_CLOUD` iff `ppl≤30 AND <26.77`. `gate_debias` is LOCAL-only (never auto-promotes cloud alone).
- **Loss-tax targets:** 2-bit **≤0.15** (now ~0.31 [measured ~0.28-0.31 PTQ at 7B/32B]); 3-bit **≤0.05** (now 0.056). Density: ship bpw **2.665→2.40** (Shannon side-info floor ~2.35).
- **Kills:** Q6 if median `r99>h/2` or grows with N. Q4 if A1 ≯ A0 by ≥0.02 tax. Any density lever if `sprint_section_chain_audit` or `verify_archive` fails. Any lever that forces float into the decode kernel.

---

## §6 — Risk ledger / landmines

1. **32B fp32-shadow OOM (~238 GB)** → land **I1** before the 32B leg. 7B unaffected.
2. **DBIA unsealed = attestation hole** → **SEAL FIRST** (§2.2 step 1) before any chain-walk wiring.
3. **Chain-walk silent-drop** → teach ALL walkers + `format.rs:section_tag` **together**; partial wiring drops sections silently.
4. **RSLT must stay outermost; every data section appends before SPRV** (`quantize-model.rs:1212`). **SDSC must bump** (`selfdesc.rs`) for C2/DBIA layout.
5. **896-dim RHT not bit-exact (~1e-6)** for odd-power widths → 256-align tensors or document the IEEE-754-no-FMA dependence. [proven limit, commit 1e5c520]
6. **Rebuild-during-PV collision** with the requant cadence → worktree `target/` until PV exits.
7. **Empirical-Fisher miscalibration** (the likely Hessian-Viterbi backfire cause) → Q6 MUST use **true Fisher** (model-sampled labels).
8. **KL-route density cost** → never report combined "smaller AND better"; density is C2/DBIA's separate headline.
9. **Two-chat collision** on tree/pod/PV → ownership + worktree discipline (§0.2).

---

## §7 — Discarded — do NOT revisit (saves agent-time)

- **E8/Leech/vector codebooks** — scalar trellis already beats E8 (0.069 vs 0.089 MSE) on the post-RHT Gaussian; a regression. [lit: QTIP §2]
- **SpinQuant / ButterflyQuant / QuaRot rotations as-is** — float rotations on activations at decode → moat-dead. The **frozen-integer butterfly** (learn angles, snap to a ±1/π/4 grid, freeze to add/sub/shift) is the only salvage, but **RANK 2, ~0.01-0.03 tax, HIGH impl-risk** — pursue ONLY if STRAND-PVT plateaus above target. The rotation is not the bottleneck. [lit: 2509.09679, 2405.16406]
- **QTIP 1MAD/3INST computed codes** — FP16 add / f32 divide in decode, not bit-exact across devices. STRAND's frozen Q12 LUT is the correct moat-safe substitute (already done). [lit: QTIP `bitshift.py`]
- **FitNet-style absolute-feature MSE in KD** — *hurts* LLMs (LLM-QAT: "attention/hidden-layer distillation hampers performance" at ≤4-bit). Only **relation/attention-map matching** (MiniLM/AT, scale-free) flips positive at ≤2-bit. Q4 uses relations + a FitNet negative control. [lit: 2305.17888 vs 2510.13998]
- **Hessian-Viterbi curvature reweight** — backfired +1.1% (coordinate-curvature flattened by RHT + empirical-Fisher miscalibration). Q6 differs (eigen-basis, true Fisher); the rest stays dead.
- Entropy-coding post-RHT **indices** (max-entropy/uniform), SeedLM, protein-folding/analog/adiabatic decode, rank-sweep eigen water-fill, per-RHT-block MDL routing — all banked-dead (sprint doc §"Discarded").

---

## Appendix — consolidated hook points

| Symbol | path:line | role |
|---|---|---|
| `QuantLinear.weight` (fp32 shadow) | `scripts/strand-qat.py:65` | I1 → bf16 |
| `AdamW` | `scripts/strand-qat.py:576` | I1 → 8-bit Adam |
| `--pv-tensors` regex | `scripts/strand-qat.py:443` | Q2 selective freeze |
| KD loss block (logit-KL only) | `scripts/strand-qat.py:612-635` | Q4 greenfield hooks |
| `--cooldown-frac` | `scripts/strand-qat.py:393-397` | Q5 |
| Viterbi path search | `crates/strand-quant/src/encode.rs:viterbi_path_buf:1149` | Q1 V-step callee |
| weight-MSE objective `gen_step_dist!` | `crates/strand-quant/src/encode.rs:1248-1249` | Q6 target swap |
| producer section hook (OUTL/SDSQ/SPRV) | `crates/strand-quant/src/bin/quantize-model.rs:1122-1243` (OUTL :1168, SDSQ :~1191, SPRV :1213) | D1/D2 append point |
| `read_outl_bytes` step-over {SPRV,SDSQ} | `crates/strand-quant/src/outlier_wire.rs:440` | D2 hazard — add DBIA |
| `section_tag` registry (missing SDSQ/DBIA) | `crates/strand-quant/src/format.rs:170-175` | D1/D2 |
| `descriptor_digest` (OUTL slot, no DBIA) | `crates/strand-quant/src/provenance.rs:112-141` | D2 seal |
| `verify_archive_with` (reads OUTL) | `crates/strand-quant/src/provenance_io.rs:507-531` | D2 seal |
| `from_mmap` (reads only OUTL) | `crates/strand-decode-kernel/src/loader.rs:51` | D2 deploy-apply |
| `matvec_rht` MAC epilogue | `crates/strand-decode-kernel/src/outlier_mac.rs:121` | D2 deploy-apply |
| `c2_final.rs` / `debias_wire.rs` | ORPHANS — not in `lib.rs` | D1/D2 wire-in first |
| `require_promoted` | `scripts/cloud-selective-pv.sh:199-207` (PV_TENSORS :96) | cloud gate |
| promote thresholds | `scripts/promote.py:43-48,95,120` | all gates |
| KL scout | `scripts/rung-kl.py` → `rung-kl-dp_d4_r2.json` | Q2 |

**Key citations:** PV-Tuning 2405.14852 · KL-Lens 2604.13440 · QTIP 2406.11235 · QuIP# 2402.04396 · EfficientQAT 2407.11062 · BitNet-Distillation 2510.13998 · LLM-QAT 2305.17888 · ButterflyQuant 2509.09679 · Null-Space-PTQ 2506.11044.
