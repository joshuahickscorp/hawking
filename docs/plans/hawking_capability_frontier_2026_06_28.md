# Hawking IDE — Capability Frontier & Build Roadmap

**Date:** 2026-06-28
**Status:** RESEARCH OUTPUT (this doc) → feeds a separate build/integration session (handoff prompt at end)
**Author session:** research-only. No code was changed.

---

## 0. The one-sentence thesis

> **Because we own the whole stack — the RWKV-7 SSM architecture (a constant-size, serializable recurrent state), the `.tq` trellis quant format (the derivation recipe), the Rust engine, and direct logit access — we can ship four classes of capability that metered/closed competitors are *structurally* barred from: (1) pass *state*, not text; (2) own the *economics* (zero marginal cost); (3) own the *format* (lossless adapters, instant load, multi-tier SKUs); (4) own the *logits* (guaranteed-valid tool calls).**

The entire field's bleeding edge (LatentMAS, DroidSpeak, Cache-to-Cache, EAGLE-3, S-LoRA, PICASO) is converging on "pass internal state instead of text" and "reuse the KV cache." **For transformers that means wrestling a quadratically-growing KV cache. We have a constant-size recurrent state that is the thing they are approximating.** A multi-agent handoff that costs everyone else a full re-prefill costs us a `memcpy`.

---

## 1. The honest edge — read this before believing the pitch

Every capability below is gated by a caveat the research surfaced. Internalize these or we will over-promise and get caught.

| # | Hard truth | Consequence for product |
|---|------------|-------------------------|
| H1 | **Constant *throughput* at long context ≠ constant *quality*.** SSM recall is bounded by a theorem (Based/Zoology, ICML'24): a fixed-size state needs Ω(N) bits for associative recall. RWKV-7's *own* paper shows passkey degrades ~20–50K tokens; hard needles (number/UUID-in-haystack) break at **~4K**. The RWKV team itself shipped a hybrid (RWKV-X, ~25% sparse attention) to fix recall. | **Never market "infinite/perfect memory."** Market *"your whole project, always loaded, instantly resumed, never billed twice, never truncated."* Route exact recall through **RAG** (we already have `hawking-index`). |
| H2 | **A *captured* state ≠ a *trained* state.** RWKV state-tuning (+7 MMLU, +8 GSM8K) requires *training* the initial state. Naively running a repo through once and saving the state to "remember" it **degrades quality** — it only saves recompute. | State checkpointing = an *instant-resume / no-re-prefill* feature, **not** a memory-quality feature. |
| H3 | **No public RWKV-7 coding score exists.** Constant memory is worthless if the model can't code. | **Build a coding eval (`hawking-eval`) and run it on RWKV-7 *before* betting the product on it.** This de-risks the whole thesis. |
| H4 | **Spec-decode is a low-batch, memory-bound win.** EAGLE-3 hits 6.47× on HumanEval at batch=1 but *degrades to negative* at high request rate. RWKV-7 can *draft* for Qwen, but can't easily be the *verified* target (recurrent state overwrites → needs rollback). | Gate spec-decode on batch occupancy. RWKV drafts, Qwen verifies — not the reverse. |
| H5 | **On-device training is LoRA/reward-filtered-SFT/small-N DPO only.** Full PPO/RLEF is datacenter-only. Use **MLX, not PyTorch-MPS** (MPS has silent CPU fallbacks + backward-pass bugs). Catastrophic forgetting worsens with size. | Personalization = nightly LoRA + tiny-data DPO on accepted/rejected diffs (Zed Zeta proved ~150 pairs works). **Mandatory guardrail:** keep adapter swappable (don't merge), mix a general-coding replay set, gate deploy on a held-out eval. |
| H6 | **2-bit needs recovery; report *effective* bpw, not nominal; perplexity ≠ usable generation.** "2-bit" methods are often ~2.5–3.6 real bpw; some pass PPL but generate garbage. | Ship a **3-bit "lossless" SKU + 2-bit "recovered" SKU** with honest labels. Always test real generation. |
| H7 | **Latent/telepathic handoffs are unauditable** (CoT-monitorability position paper, 40+ co-signers). | Keep an optional **text-decode "tap"** on any state passed between agents, for debugging/audit. |
| H8 | **Local models genuinely trail frontier on the hardest ~20% of reasoning.** Best local 7B is 10–20 pts below GPT-5.5-class. | Keep a **frontier-optional BYO-key escape hatch** for the hardest tasks; everything else local + free. |
| H9 | **The blocking technical gap is native `.tq` serving (Condense "Stage 7 / RUN").** The `.tq` decoder is test-only; `hawking generate` cannot serve `.tq` yet (`qwen_dense.rs::load` reads GGUF bytes, needs a `.tq` branch). | **This is Phase 0. Almost nothing ships without it.** |

**Market verdict from competitive research:** the field converged — parallel agents, repo context, MCP, subagents are now *table stakes*. The war moved to **cost and trust**. June 2026's "Tokenpocalypse" (Cursor/Copilot/Devin all flipped to usage-based metering; bills spiked up to 25×; ~30% of devs hit limits mid-task) cracked the door wide open for *"free forever, your machine, no meter."*

---

## 2. The structural moats (what ownership uniquely unlocks)

Four asymmetries. Every feature in §3 derives from one of these.

- **M1 — Pass state, not text.** RWKV-7's recurrent state is `n_layer × {wkv, att_shift, ffn_shift}`, **constant size regardless of context** (~6 MB @ 0.4B, ~16 MB @ 7B), trivially cloneable (it's `Vec<Vec<f32>>`). This is the serializable "memory object" the whole latent-comms field is manufacturing out of KV caches. → state fork, telepathic handoff, instant resume, state skill-seeds.
- **M2 — Own the economics.** Zero marginal cost per token on already-paid Apple Silicon. The Nth parallel agent costs only its *divergent* tokens. Cloud *cannot* match "free fleets" without going bankrupt. → free agent swarms as the default UX, unlimited iteration, speculative branches.
- **M3 — Own the format.** `.tq` (QTIP-class trellis, on the winning side of the format war). Lossless in-grid LoRA re-bake (QA-LoRA), embedded LoRA/grammar/tokenizer slots, mmap instant load, one file → multiple quality tiers via a sensitivity map. → personalization that compounds, instant model-as-a-file, multi-SKU.
- **M4 — Own the logits.** Direct mask access → grammar-guaranteed tool calls (XGrammar-class, <40µs/tok, can be *net-faster*), AST-derived grammars (model *cannot* emit a nonexistent symbol), entropy-triggered tool use, custom samplers. → first-try-valid structured output, a small model punching above its weight.

---

## 3. The feature catalog (widest scope, each gated honestly)

Ranked within each moat by impact-for-effort. "Seam" = where it attaches in the existing tree (verify line numbers during build).

### M1 — State as a first-class object
| Feature | What the user sees | Effort | Seam | Caveat |
|---|---|---|---|---|
| **State serialize / save / load** | Open a project and the agent is *instantly warm* — no re-prefill. Close & reopen mid-task, resume exactly. | **Low** | `RwkvState` (`rwkv7.rs` ~204–236): add `to_bytes`/`from_bytes`; new `Engine::save_checkpoint`/`load_checkpoint`; `prefill_slot` (~1186) loads from checkpoint | H2 — recompute saver, not memory-quality |
| **State checkpoint / undo** | Time-travel a long agent session; roll back a bad step with no re-prefill. | **Low** | same + `hide-backend` time-travel scrub/fork (already exists) | — |
| **State fork (clone)** | "Try 5 approaches, keep the best" — fork the analyzed-repo state into N branches free. | **Low–Med** | `state.clone()` (trivial); wrap as engine primitive; `RwkvMultiState.slots` already independent | M2 economics make this cheap |
| **Telepathic agent handoff** (see M1+multi-agent) | Planner reads repo once, hands *state* (not text) to coder→reviewer. ~4× faster handoffs, ~70–84% fewer tokens (LatentMAS). | **Med** | `hide-personalize::kv_handoff` + `KvShareGroup` seam → `copy_kv_prefix_to_slot`; for RWKV it's a state memcpy | H7 — keep text-decode tap |
| **State skill-seeds** (see M3) | Per-mode "personality" states (refactor / test-writing / this-repo-style), ~10 MB each, swap instantly, **zero decode cost**. | **Med** | RWKV state-tuning (trained initial state); load as slot init | H2 — must be *trained*, not captured |

### M2 — Own the economics
| Feature | What the user sees | Effort | Seam | Caveat |
|---|---|---|---|---|
| **Free agent fleets as default** | "Spawn 5, keep the best" is the *default*, not a paid tier. Cloud must meter this. | **Med** | `hide-fleet` (worktrees, FleetGovernor, 3-way merge — already real) + continuous batch=8 | Fan-out wins on exploration/search; tightly-coupled edits parallelize less |
| **Speculative agent branches** | Background agents pre-explore likely next steps while you read. | **Med** | state fork (M1) + `hide-fleet` scheduler | idempotent tool design |
| **Unlimited iteration** | Retry 10× cheaply beats one expensive perfect shot. | Free | inherent | — |
| **Energy-aware scheduling** | "Run this fleet within a 30W envelope" / battery-aware throttle. Empty niche. | **Med** | `hawking-orch` energy/thermal admission (exists) + IOKit | report ~2–4× tokens/watt, **not** the unverified 23× |

### M3 — Own the format
| Feature | What the user sees | Effort | Seam | Caveat |
|---|---|---|---|---|
| **Native `.tq` serving** | The 32B fits in ~9–18 GB where Q4_K swaps → the RAM-cliff throughput win. **Unblocks everything.** | **High (PHASE 0)** | `qwen_dense.rs::load` (~780) needs `.tq` branch; Stage A (F16 fallback) then Stage B (native bitslice GEMV); `strand_bitslice.metal` is ~90% built, parity-green | H9 |
| **mmap instant load** | Model-as-a-file; warm load ≈ instant; RAM ≈ model size (not 2×). | **Low** | comes with native `.tq`; ensure 16KB-aligned zero-copy to Metal | — |
| **Tiny-data personalization flywheel** | The model gets better at *your* code weekly from accepted/rejected diffs. | **Med** | `hide-personalize` (RLEF reward derivation + dataset already real) + MLX LoRA + small-N DPO | H5 guardrails |
| **QA-LoRA in-grid lossless re-bake** | Fold the personal adapter into the `.tq` with no quant round-trip loss. *The technical reason to own the format.* | **Med** | Condense doctor (`doctor_lora.py`) + `.tq` writer (`tq_bake`) | matches our prior "fp16 head vs Q4 runtime" mismatch lesson |
| **Multi-tier SKU from one file** | One `.tq` → 3-bit "lossless" + 2-bit "recovered" via embedded sensitivity map + mixed bpw. | **Med** | Condense allocate/encode (`ladder.py` partial) + format header | H6 — honest labels, test generation |
| **One-button "doctor" recovery** | KL-distillation pass recovers low-bit quality, data-free. | **Med** | `doctor_qat.py`/`doctor_lora.py` (KD is "best lever so far") | distillation > QAT at low bits |
| **INT8 RWKV state quant** | 2× cheaper per-doc state cache → even longer cheap context. **Unclaimed in the literature — we own it.** | **Med–High** | RWKV state path | state error compounds recurrently; **measure, don't assume** |

### M4 — Own the logits
| Feature | What the user sees | Effort | Seam | Caveat |
|---|---|---|---|---|
| **Grammar-guaranteed tool calls** | Tool calls / JSON patches valid on the first try — zero repair retries. | **Low** | `json_constrain.rs::JsonConstraint::mask_logits` (exists); harden + fused on-GPU mask | validity ~93–96% on hard schemas → keep fallback |
| **AST-derived grammars** | Model *cannot* emit a nonexistent function/symbol — grammar built from the open file's AST. | **Med** | `hawking-index` tree-sitter defs/refs → grammar | per-turn grammar recompile (XGrammar-2 ~10ms) |
| **Entropy-triggered tools** | When the model is *uncertain in code* (likely unknown API), auto-grep/read-docs instead of hallucinating. | **Med** | sampler reads logit entropy/varentropy | research-grade; prototype |
| **EAGLE-3 head + RWKV-drafts-Qwen** | Faster decode; code workloads accept highest (fixed templates). | **Med–High** | existing Eagle5 head → EAGLE-3 recipe + data scaling; RWKV-7 as flat-cost long-ctx drafter | H4 — gate on occupancy |
| **Speculative + parallel tool execution** | Hide the 45–61% of agent time spent waiting on tools; predict & pre-run likely calls. | **Med** | `hide-kernel` tool dispatcher + `hide-tools` | idempotent tools |
| **Custom samplers** | min-p brainstorm mode, contrastive decoding (Qwen − draft, +8 GSM8K), token healing for mid-identifier completion. | **Low–Med** | sampler | — |

### Cross-cutting positioning (M2 + local)
- **Verifiable no-egress privacy** — air-gap mode, provable with a packet capture (not a policy page). Copilot now trains on individuals by default (opt-out, from 2026-04-24); Cursor still transits code + stores codebase embeddings in a third-party vector DB. Continue.dev (the leading OSS local-first assistant) was *acquired by Cursor* June 2026 — **the independent local-agentic space is wide open.**
- **Frontier-optional escape hatch** — BYO-key cloud routing for the hardest ~20%; local-default for everything else (Aider's architect-mode pattern).

---

## 4. The build roadmap (phased, dependency-ordered)

Respects the constitution's **"shell-first, moonshots second"** and the **Thesis Gate**.

**Phase 0 — Unblock & de-risk** *(must come first)*
1. Native `.tq` serving: `qwen_dense.rs::load` `.tq` branch → Stage A (F16 dequant-on-load) → Stage B (native bitslice GEMV). [H9]
2. `hawking-eval` coding benchmark; run RWKV-7 + Qwen `.tq` against the Thesis Gate (≥15 tok/s @32B, ≥40% task success @7B, <10 min wall-clock). [H3]
   - **Gate:** GO → proceed. CONDITIONAL/KILL → fix model/quant before building features on it.

**Phase 1 — State primitives (the M1 foundation; mostly low-effort)**
3. `RwkvState::{to_bytes, from_bytes, clone}` + `Engine::{save_checkpoint, load_checkpoint}`.
4. State fork as an engine primitive; checkpoint/undo wired to `hide-backend` time-travel.
5. **Ship:** instant-resume, session undo, "fork & try N."

**Phase 2 — Recall (make context actually usable) [H1]**
6. Wire `hawking-index` RAG (tree-sitter + embeddings + RRF + rerank) as the *exact-recall* path for the agent.
7. Honest context product: whole-project-loaded + RAG retrieval; never market "perfect recall."
8. (If RULER/NIAH gaps remain after RAG) evaluate a small **sliding-window** attention hybrid — keeps the memory moat bounded.

**Phase 3 — Telepathic agents (the headline moat) [H7]**
9. State-passing handoff: planner→coder→reviewer pass RWKV state (memcpy), not text. Build on `kv_handoff`/`KvShareGroup` → `copy_kv_prefix_to_slot`.
10. Mandatory text-decode "tap" for auditability.
11. **Ship:** multi-agent flows ~4× faster handoffs, ~70–84% fewer tokens (LatentMAS-class).

**Phase 4 — Speed (agent-loop wall-clock) [H4]**
12. Grammar-guaranteed tool calls (harden `json_constrain` + AST grammars). Kills JSON-repair loops.
13. Prefix-cache discipline across agent steps (system-prompt-only caching; 41–80% cost / 13–31% TTFT).
14. EAGLE-3 head upgrade; RWKV-7 drafts Qwen on long contexts.
15. Speculative + parallel tool execution.

**Phase 5 — Personalization flywheel (compounding moat) [H5]**
16. Tiny-data DPO/SFT from accepted/rejected diffs (Zed Zeta recipe; `hide-personalize` already has reward derivation + dataset). MLX, not MPS.
17. RWKV state-tuning skill-seeds (per-mode, ~10 MB, zero decode cost).
18. QA-LoRA in-grid lossless re-bake into `.tq`. Guardrails: swappable adapter + replay set + held-out eval gate.

**Phase 6 — Format control plane & SKUs [H6]**
19. Multi-tier SKU from one `.tq` (sensitivity map + mixed bpw); 3-bit lossless + 2-bit recovered, honest labels.
20. One-button KL "doctor" recovery pass.
21. (Research lever) INT8 RWKV state quant — measure error compounding first.
22. Air-gap verifiable no-egress mode as a product guarantee.

---

## 5. Source appendix (load-bearing, verified)

- **State-space recall limits:** Based/Zoology (arXiv 2402.18668), Repeat-After-Me (2402.01032), RWKV-7 Goose passkey §7.5 (2503.14456), RWKV-X hybrid (2504.21463), RULER (2404.06654), NVIDIA Mamba-2-Hybrid 4/56 attn (2406.07887).
- **Telepathic / latent comms:** LatentMAS (2511.20639, 4–4.3× / −70–84% tokens / +14.6%), DroidSpeak (2411.02820, ~3.1× prefill), Cache-to-Cache (2510.03215, 2.5× / +6.4–14.2%), Coconut (2412.06769), CoT-Monitorability safety (2507.11473), EmbeddingRWKV (2601.07861).
- **State as memory object:** RWKV state-tuning (2504.05097, +7 MMLU/+8 GSM8K, *trained* not captured), PICASO state composition (2502.17605, ~5.4×), web-rwkv/Ai00 state files (shipping).
- **Speed:** EAGLE-3 (2503.01840, HumanEval 6.47×), XGrammar (2411.15100, <40µs/tok, ~100% validity), "Don't Break the Cache" (2601.06007, 41–80% cost), Mamba Drafters (2506.01206, SSM drafts flat at 8k), PASTE speculative tools (2603.18897, +43.5% task time).
- **On-device / format:** RLEF (2410.02089), Zed Zeta/Zeta2 (~150 DPO pairs → +30% accept), MLX-LM LoRA, S-LoRA (2311.03285, 4×), zFLoRA (2510.25784, fused adapter ~0 overhead), QA-LoRA (2309.14717, lossless in-grid merge), QTIP (2406.11235, lookup-free trellis), KVQuant (2401.18079), justine.lol/mmap (100× load), HF Xet content-addressed blocks.
- **Market:** June 2026 "Tokenpocalypse" metering shift (Cursor/Copilot/Devin usage-based); Copilot trains on individuals by default from 2026-04-24; Cursor acquired Continue.dev June 2026.

---

*End of research output. The build/integration session prompt follows separately.*
