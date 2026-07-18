# Build-session handoff prompt (paste into the integrated/coding session)

> Copy everything below the line into the new session. It is self-contained.

---

You are continuing the Hawking IDE (HIDE) project at `/Users/scammermike/Downloads/hawking`. A research session produced a capability frontier + build roadmap at **`docs/plans/hawking_capability_frontier_2026_06_28.md`** — read it in full first; it has the thesis, the honest caveats (H1–H9), the moats (M1–M4), the feature catalog with code seams, and the phased roadmap. Also read `docs/hide-bible/SCAFFOLD_STATUS.md` and `docs/hide-bible/00-vision-and-constitution.md` (shell-first, Thesis Gate, parity gate are load-bearing).

**Goal:** start *building* the differentiating capabilities that ownership of the RWKV-7 SSM + `.tq` trellis format + Rust engine + logit access unlocks — exploiting the one idea the whole frontier is converging on: **pass state, not text.** Maximize real, shippable features; respect the honest caveats; do not over-promise.

**Non-negotiable guardrails (from the research):**
- **H9 / Phase 0 first:** native `.tq` serving is the blocking gap. `qwen_dense.rs::load` (~line 780) reads GGUF bytes and needs a `.tq` branch → Stage A (F16 dequant-on-load), then Stage B (native bitslice GEMV; `strand_bitslice.metal` is ~90% built, parity-green). Almost nothing ships without this.
- **H3:** build `hawking-eval` coding benchmark and run RWKV-7 + Qwen `.tq` against the Thesis Gate (≥15 tok/s @32B, ≥40% task success @7B, <10 min wall-clock) *before* betting features on the model. No public RWKV-7 coding score exists.
- **H1:** SSM long-context = constant *throughput*, NOT constant *recall* (theorem-bounded; RWKV-7 hard-needle recall breaks ~4K, passkey ~20–50K). Route exact recall through `hawking-index` RAG. Market "whole project always loaded, instant resume, never truncated, never billed twice" — never "infinite/perfect memory."
- **H2:** a captured state ≠ a trained state. State checkpointing is an instant-resume feature, not a memory-quality feature. State skill-seeds must be *trained* (RWKV state-tuning), not captured.
- **H4:** spec-decode is a low-batch win — gate on occupancy. RWKV drafts, Qwen verifies, not the reverse.
- **H5:** on-device training = MLX (not PyTorch-MPS), LoRA + reward-filtered-SFT + small-N DPO only (no PPO). Keep adapters swappable, mix a general-coding replay set, gate deploy on a held-out eval (catastrophic forgetting).
- **H6:** report *effective* bpw; 2-bit needs recovery; test real generation, not just perplexity.
- **H7:** any state passed between agents must keep an optional text-decode "tap" for audit/debug.
- **Parity gate is sacred:** GPU `.tq` decode must stay bit-identical to the CPU oracle.
- **Git:** never add AI attribution to commits or PRs (no `Co-Authored-By`, no "Generated with" — house rule). Branch off `main`; commit/push only when asked. Verify real `main` and re-run parity yourself before merging any worktree lane.

**Suggested execution order (full detail in the roadmap §4):**
1. **Phase 0** — native `.tq` serving (Stage A→B) + `hawking-eval` Thesis-Gate run. STOP and report the gate verdict before continuing.
2. **Phase 1** — RWKV state primitives: `RwkvState::{to_bytes, from_bytes, clone}`, `Engine::{save_checkpoint, load_checkpoint}`, state fork, checkpoint/undo wired to `hide-backend` time-travel. Ship: instant-resume, session undo, "fork & try N." (Seams: `rwkv7.rs` ~204–236, ~1186 `prefill_slot`, ~1246 `copy_cpu_state_to_gpu_slot`.)
3. **Phase 2** — wire `hawking-index` RAG as the exact-recall path; honest context product.
4. **Phase 3** — telepathic agent handoff (planner→coder→reviewer pass RWKV state via `kv_handoff`/`KvShareGroup` → `copy_kv_prefix_to_slot`) + text-decode tap.
5. **Phase 4** — grammar-guaranteed tool calls (`json_constrain.rs` + AST grammars), prefix-cache discipline, EAGLE-3 head, speculative/parallel tool exec.
6. **Phase 5** — personalization flywheel (tiny-data DPO from accepted/rejected diffs via `hide-personalize`; RWKV state-tuning skill-seeds; QA-LoRA in-grid re-bake into `.tq`).
7. **Phase 6** — multi-tier `.tq` SKUs, one-button "doctor" recovery, INT8 RWKV-state quant (measure first), air-gap no-egress mode.

**Working style:** Start by confirming the current state of Phase 0 in the code (don't trust the line numbers blindly — verify the seams). Propose the smallest Phase-0 slice that gets a `.tq` serving end-to-end, build it, prove it with the parity gate + a tok/s measurement, then move on. Build → measure on M-series hardware → audit-to-improve. Surface honest results (if a gate fails, say so with the numbers). Each phase should land as parity-green, tested commits.

First action: read the two docs above, audit the actual state of native `.tq` serving in `qwen_dense.rs` + `vendor/strand-quant` + `crates/hawking-core/shaders/strand_bitslice.metal`, and come back with a concrete Phase-0 build plan and the smallest first PR.
