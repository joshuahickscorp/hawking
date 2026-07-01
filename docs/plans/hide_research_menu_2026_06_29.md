# HIDE Agentic-Frontier Research Menu (2026-06-29)

Parallel deep-research across the agentic frontier for Hawking / HIDE. Each sub-topic is researched
(web fan-out + primary sources), **adversarially verified** (refute the strongest claims to avoid
over-promising), then synthesized per category and tied to the four moats (M1 state / M2 economics /
M3 .tq format / M4 logits).

**Excluded by direction:** token-level speculative decoding (EAGLE-3 / draft models / decode kernels) —
already extensively worked and in the Studio pipeline. Note the deliberate split: *loop-level* speculation
(pre-running tool calls, speculative branches/preview) is NOT token spec-decode and is kept as category C.

**Framing for "does it depend on tokens/sec?":** decode tps governs only the generation slice. Agent
wall-clock is dominated by (1) re-prefill of growing context each step — *constant for RWKV-7, the M1
win*; (2) tool-wait (~45-61% of agent time, PASTE) — attacked by category C; (3) number of steps —
attacked by category D. Raw tps is necessary, not sufficient.

## Categories x sub-topics (35 parallel units)

- **A. Telepathic inter-agent comms (M1)** — A1 latent state handoff (LatentMAS/DroidSpeak/Cache-to-Cache);
  A2 merging divergent agent states; A3 state composition/arithmetic (PICASO); A4 auditing latent channels.
- **B. Latent / non-token reasoning (M1+M4)** — B1 continuous-latent CoT (Coconut); B2 latent reasoning in
  recurrent/SSM models; B3 training recipes to elicit it.
- **C. Agent-loop wall-clock & loop-level speculation (M4+M2)** — C1 speculative tool execution (PASTE);
  C2 parallel tool scheduling; C3 cross-step cache discipline (free for constant state?); C4 grammar-guaranteed
  tool calls (XGrammar/AST grammars).
- **D. Smarter decisions (M2+M4)** — D1 verify-from-tests loops; D2 best-of-N + inference-time scaling;
  D3 search over plans (ToT/LATS/GoT); D4 critics/reflexion/PRMs; D5 termination/convergence/escalation.
- **E. Memory & context quality (M1+M3)** — E1 measured compaction + recall oracles; E2 agentic memory
  architectures (MemGPT/Letta/knowledge-graph); E3 hybrid long-context + RAG for code; E4 recall-fidelity
  probing for recurrent state.
- **F. Model intelligence baked in (M3/training)** — F1 coding eval for local SSMs (hawking-eval);
  F2 agentic RL / execution-feedback post-training (RLEF); F3 on-device personalization flywheel;
  F4 hybrid recall architectures (RWKV-X / Mamba-2-hybrid).
- **G. Economics-as-UX (M2)** — G1 free-fleet orchestration + merge; G2 energy / tokens-per-watt scheduling;
  G3 RAM/state scheduling for many concurrent agents.
- **H. Interaction layer / DX (M2+M1)** — H1 command grammars + predictive palettes; H2 streaming inline
  diffs / apply-in-editor; H3 local voice-to-code + mid-run steering; H4 ambient background-agent UX;
  H5 learned macros / agent-authored commands.
- **I. Trust & verifiability (M2+local)** — I1 provable no-egress privacy; I2 provenance/replay/undo;
  I3 risk-tiered approval gates.

## Status

- [x] Research complete (workflow `wamhigl8n`, 79 agents, 4.3M tokens, 35/35 sub-topics, 9/9 briefs).
  Each sub-topic web-researched + adversarially verified; per-category briefs in `docs/plans/research/`.

## Per-category briefs

- [A. Telepathic state passing](research/a_telepathic_state_passing.md)
- [B. Latent reasoning](research/b_latent_reasoning.md)
- [C. Loop-level speculation](research/c_loop_speculation.md)
- [D. Reasoning harness](research/d_reasoning_harness.md)
- [E. Memory & context](research/e_memory_context.md)
- [F. Model training / intelligence](research/f_model_training.md)
- [G. Economics-as-UX](research/g_economics_ux.md)
- [H. Interaction / DX](research/h_interaction_dx.md)
- [I. Trust & verifiability](research/i_trust_verifiability.md)

## Top-level synthesis

**The single cross-cutting finding (every category converged on it independently):** the moat is real
but the honest framing is **"constant-size semantic handoff with bounded recall," not "lossless / infinite
memory."** Fixed-state SSM recall is theorem-bounded — hard needles break ~4K tokens, passkey ~28-35K @2.9B
(E, A). Consequence: **retrieval is a first-class requirement, not a fallback.** Marketing "perfect recall"
gets us caught.

**M1 (pass state) is the through-line** — it shows up as the differentiator in 8 of 9 categories:
copy-not-merge handoff (A), exact-and-free cross-step caching + cheap state forking (C), memory-free
parallel sampling / tree branching (D), hot zero-latency memory layer (E), personalization surface (F),
~16 MB state vs 0.5-4 GB KV (G), portable checkpoint for command-ranking / voice barge-in rollback (H),
O(1) checkpoint/replay (I).

**M4 (logits) is the most underexploited high-confidence bet** — grammar-constrained decoding gives
structurally-valid tool calls + syntax-valid streaming apply + CRANE-style grammar switching +
entropy-triggered tools. Lower risk than the latent-reasoning bets.

**M2 (economics) is a positioning win, NOT an efficiency claim.** Cloud (B200 / SambaNova) beats Apple
Silicon 1.6-7x on J/token (G). The win is zero-marginal-cost + free fan-out, not "greenest/fastest per
watt." Real concurrency is throughput-capped at ~4-6 simultaneous agents (RAM parks thousands) (G).

**Over-promise traps the verifiers flagged (de-weight these):**
- "Lossless cross-architecture latent handoff" — every method is same-family only; cross-family unvalidated (A).
- "Latent in-state reasoning off the shelf" — RWKV-7 does NOT; near-term is distill+GRPO-LoRA, not Coconut (B).
- "Full tool speculation" — collapses task accuracy -30pp without a verify step; verification is non-optional (C).
- GPT-4-era harness numbers attributed to a local 7B — contamination / metric-conflation / cost-omission (D).
- "Coding parity" / "baked-in coding intelligence" — **no published RWKV/SSM coding-agent score exists** (F).
- Source headline magnitudes (LatentMAS 6.6x, Interlat 24x, state 62.5x, ZK "thousands x") — outliers or
  mis-scoped; report as best-case only (A, I).

**Honest-novelty we could be FIRST to publish:** a state-fidelity(age) curve + a compaction recall-gate (E),
INT8 recurrent-state quant measurement (E/G), and a covert-channel audit layer for state handoff (nobody
has deployed one — A).

## Recommended next dives (deeper, parallel)

1. **F1 hawking-eval first** — the loaded gun. Coding capability is unmeasured; several bets sit on it. [de-risk]
2. **Honest-context architecture** — RWKV state as hot memory + lexical-first retrieval + the state-fidelity
   curve + compaction recall-gate (E). Operationalizes the moat without the over-promise.
3. **A1 intra-fleet telepathic handoff** (same-model, copy-not-merge) — research says it's real *within family*,
   which is exactly Hawking's one-local-model case. Headline moat, green light with honest framing.
4. **C + M4** — grammar-guaranteed tool calls + cross-step cache discipline. Low-risk wall-clock wins.
5. **D execution-feedback loops** — grounded test signals beat self-reflection; highest-leverage "smarter."

## Round-2 deep dive — results + build order

**Status:** complete (workflow `wv3w98njq`, 45 agents, 20/20 sub-questions, 5/5 build-specs). Deep docs:
[F1 hawking-eval](research/f1_hawking_eval_deep.md) · [F2 honest-context](research/f2_honest_context_deep.md) ·
[F3 telepathic handoff](research/f3_telepathic_handoff_deep.md) · [F4 grammar+cache](research/f4_grammar_cache_deep.md) ·
[F5 execution-feedback](research/f5_execution_feedback_deep.md).

**Headline (verified against the tree):** ~70% of the Spine A/B infrastructure the executor plan v2 lists as
"NEW code required" is ALREADY BUILT in the uncommitted tree — oracle ladder (`hide-kernel/src/verify/`),
repair/replan loop + minimal-repair `Failure` context, `RecallOracle`/recall@k + `CompactionEvent`
depth/recall fields, `.tq` multiplier reader (`tq_metadata.rs`), `GET /v1/hawking/context` + Engine context
accessors, per-step `projection_patch` with watermarks, `RecallFidelityProbe` trait (stub), `Interrupt::Steer`,
the `hawking-index` code index (tree-sitter + RRF + reranker), the orch grammar FSM (`GrammarSpec`/`JsonObjectFsm`),
the fleet `TournamentSelector`. **The plan is stale — reconcile against the tree before building.**

**Verified seam corrections:**
- **F3 hint WRONG:** `kv_handoff`/`copy_kv_prefix_to_slot` is a TRANSFORMER KV-cache path; it never touches
  `RwkvState`. Reuse the protocol *shape* (`KvShareGroup` lifecycle), not the copy path; build the RWKV handoff
  from the CPU state + the `prefill_disk.rs` (`DSPRFKV2`) format template.
- **`RwkvState` (`rwkv7.rs:195`) does NOT derive `Clone`** and has no `to_bytes`/`from_bytes` (the line-59 derive
  is a different struct — verified by grep). B1 must ADD the derive + serialization.
- `Engine::{save,load}_checkpoint` do not exist; **GPU→CPU state readback does not exist** (hidden dependency
  for serializing GPU-prefilled states).
- `hawking-eval` does NOT exist (`hawking-bench` is throughput-only). `CompactionRollback` event MISSING.
  Typed `Lesson` MISSING (`lessons: Vec<String>` lives in `hide-kernel`, not `hide-backend`). Convergence/stall
  detection MISSING. Sampler entropy/varentropy MISSING. **Fleet isolation is git-worktree, NOT RWKV state-fork (R5).**

**Cross-frontier build order (de-risked, parity-gated):**
- **Wave 0 (measure/de-risk):** `hawking-eval` crate (proxy gate + Aider-Polyglot via existing
  `/v1/chat/completions`, $0 local) + the RWKV-7 fidelity(age) curve. Every capability number gates on these.
- **The M1 atom:** `RwkvState` `Clone` + `to_bytes`/`from_bytes` (DSSSMV1, mirror `prefill_disk.rs`) +
  `Engine::{save_checkpoint,load_checkpoint,fork_state}`. ~100 LOC, parity-gated (bit-identical round-trip +
  two forks → bit-identical logits). Shared by F3/F4/F5.
- **Wave 1 (cheap honest wins, all small):** F5 convergence/stall detect · F4 grammar-FSM bridge into engine
  decode + CRANE delimiter gating · F2 wire fidelity probe into the live stream + fix `compiler.rs:439`
  `ctx_len_effective` · `CompactionRollback` event · typed `Lesson`.
- **Wave 2 (capability, on the atom):** F3 sequential planner→coder→reviewer handoff (copy-only) + text-decode
  audit tap · F5 state-forked Best-of-N wired to `TournamentSelector` (closes R5) · F4 AST identifier allowlist.
- **Wave 3 (gated on measurement):** int8 `wkv`-state quant · RWKV-X sparse-attention hybrid · bounded
  tree-search escalation · logit-probe tool-readiness prediction.

**Universal guardrails:** parity-gate before merge to `main` (re-run in `main`, gate on exit code not `--watch`);
never market lossless/infinite/perfect-recall (bounded recall is the honest frame); retrieval stays first-class;
no new on-screen widget (capability under the hood, quiet Context-Stack lines only).
