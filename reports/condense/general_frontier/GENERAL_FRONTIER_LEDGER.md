# GENERAL FRONTIER LEDGER

Durable campaign ledger (goal Part 45). Human-readable running log of
`GENERAL_FRONTIER_STATE.json` (schema `hawking.general_frontier.state.v1`).
Resume from here without re-planning the campaign.

- Goal: HAWKING GENERAL FRONTIER - CONTINUOUS HANDOFF (`~/Desktop/HAWKING_GENERAL_FRONTIER_ULTRACODE_CONTINUOUS_HANDOFF_GOAL.md`)
- Doctrine: One Gravity law. Many scales. Two hardware worlds. One empirical frontier. Sub-bit-first + quality + whole-artifact accounting + backend portability are non-negotiable.
- State snapshot generated_at: `2026-07-19T02:40:31Z`
- State sha256: `4566aa660a28d9a6e5d118dd570662d66ecdc103b1e0b39bb452405c83208e77`
- NOTE: this is a point-in-time snapshot; an active `/goal` loop is advancing the Gate-F ladder in parallel. Statuses below are measured, not remembered.

---

## Authoritative commit
- Accepted handoff anchor `main` = `e2609f9452bcc9c97cfa960d6ef362567a048727` (`e2609f94`); was HEAD=origin/main at handoff.
- Working branch: `codex/general-frontier-gate-f`, HEAD `f648d6be` ("General Frontier: Gate-F transition scaffold (precheck, parents, Atlas, G0/G1, CUDA plan)"), a direct child of `e2609f94`, pushed to origin.
- Second-Light baseline: `b465588384cf9cf7107f3144df1d6bf558bf3581` tag `hawking-second-light-baseline`. Geometry-search result committed at `e2609f94`. Pre-frontier analysis `7f237ed3`.
- Preserved scaffolds `codex/packs-absorption-scaffold` + `codex/cli-consolidation-scaffold` -> do NOT merge into the run-critical frontier. Later valid work must not be reset.

## Live controller
- Status: NONE YET (Gate-F pending). No durable General-Frontier / Gate-F singleton controller with a frozen immutable generation + byte-stable program hash exists, even though the Gate-F branch/scaffold (`f648d6be`) is committed.
- Part XIV/50: exactly ONE authoritative controller will own the next generation; this General Frontier will own it. The Gate-F ladder so far ran via bounded reproduction SCRIPTS, not the durable controller.

## Live process truth (measured, point-in-time; campaign actively advancing)
- gravity_frontier 32-trial geometry search: COMPLETED (32/32 sealed).
- G0 reproduction proc (pid 27566, `gravity_frontier_controller.py run --max-rows 4`): COMPLETED (dead); sealed `GENERAL_FRONTIER_RESULTS/GATE_F_G0_RESULT.json`.
- G1 reproduction proc (pid 28670, `gravity_frontier_g1.py`): COMPLETED (~157s); sealed `GENERAL_FRONTIER_RESULTS/GATE_F_G1_RESULT.json`.
- At final tick: 0 live heavy processes. On-disk `gravity_frontier.lease` is STALE (names dead pid 27566) -> clear it (Part 49).
- MoP CPU load: unrelated, coexists (PQ lane is MPS-bound).

## Active parent
- 120B = `openai/gpt-oss-120b` @ `b5c939de`, source PRESENT (`models/gpt-oss-120b`, ~60.77 GiB), tokenizer + Harmony chat template (vocab 201088). Only present source; the real compute lane.
- Architecture: 36 layers x 128 experts x top-k4, hidden/intermediate 2880, GQA 64/8. Role: representation foundry (F0 anchor, broadest search).

## Active backend
- Primary: `apple_mps` (M3 Ultra 28-core; design center). CPU reference is authoritative tie-break; `pq_cpu_metal_parity` green.
- CUDA: BLOCKED (no sealed budget, no credentials; Part 28) -> implement/CI-fixture + provisioning plan only, no paid launch.

## Program (Gate-F pending)
- Milestone: GPT-OSS-120B GATE-F QUALITY GENERATION. Immutable byte-stable program hash: NOT yet frozen (current program_sha256 embeds `generated_at`).
- Candidate set - mlp1 (up-gate): A1 pq_doctor_lowrank / A2 product_quant / A3 pq_protected_islands / A4 equal-byte control. mlp2 (down-proj): B1 pq_protected_islands / B2 naive_rvq / B3 plain PQ / B4 equal-byte control. Router + non-expert: protected/source-native controls first.
- Scope ladder (G0-G4 are gates; only G5 is the complete run):
  - G0 reproduction: **SEALED - PASS** (deterministic reproduction from `e2609f94`; budgets + functional metric verified; winners confirmed).
  - G1 larger expert reproduction: **SEALED - reproduces (still PROXY)**. Layer-0 experts, cal(16)/val(12) split, CPU/Metal parity green (assignment_agreement 1.0, ranking_match). mlp1 robust to all PQ (~0.0053 val); mlp2 pq_protected_islands wins decisively (val 0.149 vs rvq 0.447, plain PQ 0.610). capability_parity False.
  - G2 complete layer: PENDING.
  - G3 cross-layer transfer (early/mid/late + holdout): PENDING.
  - G4 short end-to-end hybrid (Harmony prompts, real logits, top-k, PPL/NLL, deterministic gen): PENDING.
  - G5 complete 120B frontier generation (whole-artifact BPW, Apple + CUDA-when-authorized, lower-rate boundary): PENDING.

## Completed experiments
- SECOND-LIGHT PQ BASELINE: 183/183 sealed, 0 failed, realized whole-artifact BPW 0.76976, 10.47 GiB (from 60.77 GiB). Quality NEGATIVE (true-residual output divergence 0.68792, weight rel-err 0.554, no capability pass). Commit `b4655883`, tag `hawking-second-light-baseline`. Canonical negative baseline; NOT rerun.
- 32-TRIAL GEOMETRY SEARCH: 32/32 sealed, 0 over-budget (8 geometries x 2 expert classes x 2 rates 3/4 & 1/1; geometry-before-rate). Bounded representation PRIOR only; reruns limited to the short G0 check.

## Frontier (per-class winners, PROXY - NOT champions)
- expert_mlp1 (up-gate): `pq_doctor_lowrank`, functional-div proxy 0.00832, whole-artifact 0.87608 BPW, row t0012.
- expert_mlp2 (down-proj): `pq_protected_islands`, functional-div proxy 0.184, whole-artifact 0.913191 BPW, row t0011.
- Baseline functional divergence 0.688. G1 larger reproduction confirms the DIRECTION (cal/val + CPU/Metal parity) but capability_parity is still False (mlp1 all-PQ statistically indistinguishable ~0.005; mlp2 islands decisive at 0.149).
- HONESTY: synthetic-activation output-divergence PROXY, not a verified capability pass, no Event Horizon. Second-Light artifact and these winners are NOT auto-promoted champions (Part 68).

## Quality gates
- Frontier quality contract SEALED (`GPT_OSS_120B_FRONTIER_QUALITY_CONTRACT.json`, sha `2c1405f9...`).
- Promote threshold functional divergence 0.6; ranking = FUNCTIONAL output divergence (real reference forward); capability tier requires HF-validated forward.
- Gate law: do not lower thresholds after seeing results. Data separation: calibration / validation / holdout (never tuned).
- Readiness: gravity_frontier 25/25 apparatus-green; gates are apparatus, not capability -> NO capability pass claimed.

## Cloud spend
- 0 USD. BLOCKED: no sealed `HAWKING_CLOUD_BUDGET`, no CUDA/RunPod credentials. Provisioning plan only; budget NOT inferred from key presence (Part 28).

## Resource state
- M3 Ultra 28-core / MPS, 96 GB unified memory, ~559 GB disk free, swap present.
- One-Apple-heavy-lease POLICY enforced; live heavy count fluctuated during audit (G0 then G1), 0 live at final tick; stale `gravity_frontier.lease` (dead pid 27566) present -> clear.

## Current blockers
1. Giants source-absent: DeepSeek-V3.2 685B, Kimi-K2.6 1T, optional DeepSeek-V4-Pro 1.6T ABSENT locally (contracts exist) -> source-authority/adapter/metadata PREP ONLY.
2. CUDA budget-absent: no sealed budget / credentials -> CUDA lane implement + provisioning plan only, no paid launch.
3. Gate-F durable controller + byte-stable immutable program hash not built yet (branch/scaffold committed; G0/G1 ran as bounded scripts).
4. Winners remain PROXY (capability_parity False); true-residual + real Harmony activations + holdout (G3/G4) required before any WIN claim.
5. Stale on-disk lease (dead pid 27566) to be cleared.

## Next exact edits
1. Gate-F branch: **DONE** (`codex/general-frontier-gate-f`, `f648d6be`, pushed). Remaining Generation-0 binding PENDING: lock frozen quality contract + exact Forge/Doctor/Metal provider commits + IMMUTABLE byte-stable Gate-F program hash.
2. G0 reproduction: **DONE (SEALED G0 PASS)**.
3. G1 larger expert reproduction: **DONE (SEALED, reproduces, still PROXY)**.
4. Adopt a byte-stable Gate-F program identity (hash program CONTENT excluding `generated_at`).
5. Clear the stale `gravity_frontier.lease` (dead pid 27566).
6. Materialize the durable General-Frontier (Gate-F) singleton controller (Part XIV/50); do NOT launch a second heavy controller while a live heavy Apple experiment runs. Then advance G2 complete-layer (router + all expert paths + weighted combine + residual, layer 0, full byte accounting), then G3 cross-layer + holdout, then G4 short end-to-end.
7. When a sealed cloud budget appears: activate the CUDA lane per the provisioning plan.

---

## Operating-mode ladder + idle-state enum (reference only)
Per Part XV these are compact policies / state fields / services inside the ONE frontier controller.
Do NOT build a separate scheduler, daemon, database, queue, dashboard backend, or evidence format.

- Operating modes: `ALWAYS_ON`, `LOCAL_ONLY`, `LOCAL_PLUS_CLOUD`, `BUDGETED`, `DRAIN_AFTER_ROW`, `DRAIN_AFTER_PARENT`, `PAUSED`, `MAINTENANCE`.
- Idle states: `NO_RUNNABLE_WORK`, `WAITING_SOURCE`, `WAITING_EVIDENCE`, `WAITING_QUALITY_GATE`, `WAITING_RESOURCE`, `WAITING_THERMAL`, `WAITING_COST_AUTHORIZATION`, `WAITING_HUMAN_AUTHORIZATION`, `MAINTENANCE`, `CONTROLLER_FAULT`, `FRONTIER_STABLE`.
- Current effective mode: `LOCAL_ONLY` (cloud blocked). Current idle reason: transient gap after G0/G1 sealed; durable Gate-F controller/auto-dispatcher not yet materialized (nearest enum `WAITING_HUMAN_AUTHORIZATION`).
- No-idle law: if a runnable admitted task fits the resource + cost envelope and gates permit, dispatch within two controller heartbeats.
- One-controller law: exactly one authoritative controller owns programs/queue/leases/evidence/cost/frontiers/transitions; no per-model or per-backend controller.

## Parent DAG (blocked node must not block independent prep)
- F0 120B: active (Gate-F pending), source present.
- F1 intermediate: transfer-calibration; only canonical parents already in queue; do not invent.
- F2 685B (DeepSeek-V3.2): prep-only, source ABSENT. Eligible when 120B has one full-rank family + one Doctor path through G3 + frozen contract + Apple exec + CUDA parity-or-blocked-receipt + measured whole-artifact projection.
- F3 1T (Kimi-K2.6): prep-only, source ABSENT. Eligible when 120B reaches G5-or-boundary, 685B reaches parent-bound F2 + transfer, Kimi source/adapter/claim separation green. Claims separated: `K2.6_TEXT_CORE` vs `K2.6_FULL_MULTIMODAL`.
- F4 1.6T (DeepSeek-V4-Pro): optional post-frontier; requires sealed admission + cost/info justification; NOT activated.
