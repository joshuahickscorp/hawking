# SUB-1 / SUB-0.2 BPW FRONTIER — SYNTHESIS (10-lane Grok superwave, Opus-arbitrated)
Guardrails applied: complete accounting, activation-aware only, Doctor authority, native execution,
stored-vs-ACTIVE distinct. Evidence: `evidence/sub1/*.md`. Date 2026-08-19.

## HEADLINE VERDICT (honest)
- **STORED sub-1 (complete_bpw < 1) on the MoE expert body: CREDIBLY REACHABLE NOW.** Multiple
  independent lanes converge. The governing identity (DERIVED, O005): `stored_complete ≈ 0.9495·expert_bpw
  + 0.4038` (the 0.4038 = protected tail embed+attn+lm_head+router @8-bit). Drive experts to 0.01-0.1
  complete → **O005 stored ≈ 0.41-0.50** → first honest sub-1. Paths: activation-aware low-rank / shared
  PCA basis / sign-or-binary base + sparse residual / structural expert-drop.
- **STORED sub-0.2 (1/5): BLOCKED BY THE PROTECTED TAIL, not the experts.** embed+attn+lm_head@8 already
  cost ~0.40 BPW. Sub-0.2 stored needs the tail crushed to sub-2-bit (per-row two-tier embed/lm_head +
  attn~2) — HIGH Doctor risk (Qwen3.8 uniform-q3 tables FAILED on EOS). The only MEASURED sub-0.2-class
  complete rate is the GLM procedural student (0.0946 @int4) and it FAILS_UNDER_DEPTH (not capability).
  So sub-0.2 is a tail + function-coding problem; UNKNOWN-but-probeable, not free.
- **ACTIVE sub-1: expert-path YES, all-touched NO.** Geometric wall: q4 attn alone ≈510 MB/tok > the
  380 MB/tok all-touched-1.0 budget → all-touched active-sub-1 impossible while attn ≥4-bit. Expert-path
  active can reach ~0.01 BPW (16× selection × extreme per-expert). The 16× is a MOVEMENT lever, NOT a cost
  cut (counting 0.0625·stored as active is illegal).
- **THE REAL BOTTLENECK IS NATIVE KERNELS, not the representation.** QwenMoE::load is Unimplemented;
  mlx gather_qmm keeps the FULL 14.5GB expert body resident; LUT-GEMM / QTIP-trellis are Type-1 DEAD on
  Apple GPU; runtime is kernel-bound (1.16× fewer bytes → only 1.05× TPS on A3B). So Odyssey-I can PROVE
  reachability + localize the frontier via SPECIMEN measurement; realizing it physically needs new Hawking
  kernels (skinny-GEMV, binary-GEMM, MoE gather-skip) — bible §53 minimal-primitive → Odyssey-II/super-kernel.

## SURVIVORS (pass all guardrails; ranked)
1. **Activation-aware low-rank (shared-PCA / factor-quant)** — organ complete 0.09-0.46; whole-O005 stored
   0.49-0.68 @n=8. NATIVE: `y=(xV)Uᵀ` skinny GEMV, no codebook, ~90× fewer FLOPs, never materialize W. The
   GLM 0.167 precedent. STRONGEST native sub-1 path.
2. **Structural expert-drop + drop-to-constant/mean + sparse repair** — organ 0.008-0.03; MEASURED near-zero
   at expert-index granularity (zeroing hot expert 49 across layers = delta_hits 0). NATIVE = gather-SKIP
   (~0 FLOPs) — but needs mlx to stop residencing the dropped holes.
3. **Sign-block / binary base + sparse rice-q1 residual** — expert 0.30-0.70, O005 stored 0.69-1.07 (first
   honest stored sub-1). Native credible (register-resident 256-entry table, NOT large LUT / not BitNet-fake)
   — decode kernel UNKNOWN.
4. **Cross-component low-rank r=1** — 0.029 BPW, native cheaper (~500× fewer FLOPs) but r=1 quality UNKNOWN.
5. **Fail-mass heterogeneous dispatch** (the framing): sub-0.2 is a gate/up function-coding problem — passing
   organs → skinny-GEMM, failing organs stay high-precision native. Mixed dispatch = the Q80 pattern.

## KILLED (fake / dead — do NOT spend cycles)
uniform sub-1 raw-weight codec (dead ~1 bit) · cross-layer/cross-expert tying+merge (inter-expert cos 1e-4)
· LUT-gather + QTIP Metal decode (Type-1 dead on Apple GPU) · 0.0625·stored counted as active (illegal) ·
any expand-to-dense-before-compute · 2-byte index as 0.01 without charging the shared table · mlx 4-bit
round-8 on already-4-bit weights (no-op).

## FIRST EXPERIMENTS ON O005 (cheapest, SPECIMEN, activation-aware)
1. **Activation-aware low-rank on down_proj** (tolerant organ): organ complete_bpw + Doctor at r∈{64,128}, b=2.
2. **Structural expert-drop descent**: use the sensitivity map; drop lowest-sensitivity experts, measure Doctor delta + stored/active.
3. **Shared PCA basis per layer** (one V per layer, per-expert skinny maps): organ complete + Doctor.
4. **Opportunistic per-component descent (OPDESCENT-COORD)**: geometric per-organ descent to the Doctor floor {1,0.5,0.25,0.1,0.05,0.02,0.01}, joint-compose → total complete_bpw.
All measure COMPLETE bytes + Doctor; a mechanism that only executes by expanding to dense is NOT counted.

## → CANDIDATE FAMILIES (added to candidate_families.json for deterministic descent)
aa-lowrank-organ · shared-pca-basis · sign/binary-base+rice-residual · expert-drop-structural ·
crosscomp-lowrank-r1 · failmass-hetero-dispatch · tail-crush-two-tier (sub-0.2 gate). Each carries its
native-viability + complete-accounting + cheapest-falsifier so the engine Doctor-qualifies per component.
