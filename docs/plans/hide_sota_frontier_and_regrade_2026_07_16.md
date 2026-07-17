# HIDE SOTA frontier read + regraded ladder (condenser pass)

Date: 2026-07-16
Branch judged: main @ 1c525380.
Companion to: docs/plans/hide_deep_audit_2026_07_16.md (the Zed-anchor audit). This doc re-grades the capability facets against the bleeding-edge frontier (10 = published SOTA as of mid-2026, not Zed), then lists the ripped/adapted build ladder.
Method: 81-agent research workflow. 8 domain scouts (each mapped to an audit capability facet) swept arxiv/web for 2024-2026 SOTA, deep-read the top 5 papers each, tagged every method rip/adapt/skip/watch against our exact baseline, then every load-bearing method was adversarially verified for whether its gain survives at batch=1 on Apple Silicon Metal with our small Qwen/RWKV models. A meta-synthesis produced the cross-domain themes, the ranked build list, and the regrade.

## The headline

The audit said HIDE is good parts, unwired. The frontier read sharpens it into a strategic verdict:

1. **We are simultaneously over-built and under-wired, which is the worst of both.** The 12-phase kernel FSM is a rigid *workflow* for a task class (coding) the frontier explicitly classifies as the open-ended *agent* case where you must not hardcode the path. Every scaffold that scores on SWE-bench Verified (mini-swe-agent >74%, OpenHands ~66-72%, Agentless fixed 3-phase) is a flat ReAct loop where our phases are *emergent from observation-as-next-turn*, not states. Meanwhile the shipped product is a single-shot 256-token no-tools completion, which is the opposite failure (under-built, no environment ground truth). So the fix is not to wire the FSM; it is to collapse it into one flat execution-grounded loop.

2. **The frontier just validated our one real moat.** The SOTA open-weight local coder is now a hybrid recurrent-state MoE (Qwen3-Coder-Next, Gated-DeltaNet, ~70.6% Verified on a 64GB Mac), and the systems frontier (FlashRT / Execution-State Capsules, RadixAttention, Marconi) has converged on byte-exact single-stream state checkpoint/restore, which is *exactly* our built-but-unrouted RWKV-7 memcpy state fork. "Pass state not text" is no longer a bet, it is the frontier. Our edge (O(1) fork/restore, near-free best-of-N branching) is real but invisible because it has no HTTP route.

3. **Everything downstream is gated on an eval that has never run.** Every capability build is unfalsifiable until one OpenAI-compatible endpoint serves batch=1 greedy and produces a single number. This is the true first task.

## The regraded ladder (10 = bleeding-edge frontier)

The audit graded against Zed/Linear craft. Re-anchored to the published frontier, the capability facets drop, as you predicted. This is not a worse product than yesterday; it is an honest measurement against a higher bar, which is the bar we build to.

| Facet | Zed-anchor (audit) | SOTA-frontier (this pass) | Why the frontier bar moves it |
|---|---|---|---|
| Agent-work UX | 2.5 | **1.5** | Frontier = working agentic loop + FIM tab + next-edit rewrite + execution-gated done. We ship none; every steering control is placebo. |
| Context and memory | 2.5 | **1.5** | Frontier assembles a curated turn under a tokenizer-true budget with wired memory. We discard the compiler output to a blake3 hash, count chars/4, pass the raw last message, have zero turn-to-turn memory. |
| Hawking serving seam | 3.0 | **2.5** | Frontier ships constrained decode + speculation + prefix-cache/state reuse as production defaults. Our batch lanes apply none; json_object dropped; stop strings unhonored; re-prefill from 0 every turn. |
| Agent architecture | 4.5 | **3.0** | Wrong shape (rigid workflow) *and* unwired. The FSM craft is real but it is the wrong artifact by frontier doctrine. |
| Agentic tool system | 4.5 | **3.0** | Parser/ToolLoop/MCP/apply_patch unit-tested but unwired; the standard OpenAI round-trip 400s on turn 2; no constrained decode. |
| Product differentiation | 3.0 | **3.0** (held) | The only facet that does not drop. Our pass-state-not-text moat was validated at the frontier; the primitive exists, it is just unexposed. Ceiling rose, floor held. |

Non-capability facets keep their audit grades (visual-doctrine 5.5, frontend-eng 5.5, backend-eng 6, interaction-ux 4, ship-readiness 2.5, testing-ci 3, docs 4), with one correction the research forced: **security 4.5 is optimistic once the loop is live.** The audit graded a plane where the model never executes anything. Every top build runs model-emitted bash and cargo/git; the moment the flat loop ships, this is arbitrary code execution on the user's machine and security must be re-graded against a live-exec threat model (see strategic gap 1).

## The five frontier moves we are missing

All five confirmed by more than one domain and surviving batch=1 verification:

1. **One flat execution-grounded loop, not a phase graph.** Seed `[system, instance]`; loop: model completion -> parse one action -> execute -> append `(returncode, stdout)` as the next turn -> stop on an objective signal + hard iteration cap. Observation-as-next-turn *is* the repair/replan mechanism. Optimize the Agent-Computer Interface (poka-yoke args, absolute paths, text-close formats), not phase transitions.

2. **Execution is the accept signal; substring and self-judge are dead.** Hybrid verify (learned shortlist + execution tie-break, R2E-Gym +7-8pt), fail-to-pass test cogeneration (a passing agent-authored reproduction test implies P(correct)~0.87), deterministic read-only pre-dispatch gates (+12.4pp tau2). LLM-judge self-preference inflates ~10% and is worst exactly in our low-distinguishability small-model regime.

3. **Curated context assembly + tokenizer-true budget + wired memory.** RepoMap in ~1k tokens (deterministic, no model call), line-level tool-output pruning (SWE-Pruner/Squeez, 23-54% cut), Mem0-style extract-consolidate-retrieve with hybrid semantic+BM25+entity retrieval (LoCoMo 92-94). Reserve target set *below* the measured context-rot knee, not at the model's max.

4. **Constrained decode + draft-free speculation are table stakes that compose.** XGrammar-2 TagDispatch / llguidance grammar-mask with jump-forward (mask cost <0.25% of our ~33ms/token decode), SuffixDecoding and file-as-draft (peak *at batch=1* precisely because our decode is bandwidth-bound). They stack. Our dormant ToolSpec/SpecGovernor is this exact lane with zero consumers.

5. **A standing eval gate is the precondition for everything.** BFCL v3/v4, EvalPlus, Aider Polyglot, tau2-bench, SWE-bench Verified all run batch=1 greedy against one OpenAI-compatible endpoint. We have never produced a single coding or tool number.

## The counter-theme: where we are actually ahead

The frontier converged onto our thesis. Qwen3-Coder-Next (hybrid Gated-DeltaNet recurrent MoE) is the SOTA local coder; FlashRT productizes byte-exact single-stream state capsules. Our RWKV-7 memcpy state fork/restore is that primitive, already built and parity-tested. At batch=1 on unified memory, forking a recurrent state is the cheapest possible best-of-N and sub-agent-isolation primitive on the planet, and it is O(1) fixed-size, unlike a KV cache that grows with context. We are one HTTP route away from a capability the datacenter frontier pays 8x A100 to approximate. This is the build that converts the moat to a number (Build 12).

## The ripped build ladder

Ranked by (expected gain x fit to our unique position) / effort, each tied to the audited gap it closes. Only methods that survived batch=1 verification are included. Effort: S = hours to a day, M = a few days, L = a week+.

**Tier 0 (unblockers, all S, do first, no dependencies):**
- **B1. OpenAI tool round-trip fix + lenient parser + honor stop strings + single-tokenize** (tool-calling). Make hawking-serve complete the standard round-trip (assistant `tool_calls` with `content:null`, then `role=tool` continues), wire the unit-tested parser+ToolLoop into the batched lane, canonicalize xLAM-style emit shapes, stop tokenizing twice under the lock. Unblocks the entire tool + loop + spec-decode-for-tools stack. Moves: agentic-tool-system, hawking-serving-seam.
- **B2. RWKV-7 state fork/save/restore HTTP route + batch-1 prefix-cache fix + lift max_seq_len** (serving). Expose the built state save/fork/restore behind a session abstraction keyed by token-prefix hash (FlashRT capsule model: contiguous static buffers, byte-exact, O(1) memcpy restore). Move the transformer prefix-reuse lookup into `admit` so batch=1 stops re-prefilling from 0. Lift the hardcoded 4096. Moves: hawking-serving-seam, product-differentiation, context-and-memory.
- **B3. Execution-grounded accept gate: wire the real cargo/git oracle into the live verify-ladder** (verification). Replace substring heuristics with the real cargo build/test + clippy + git oracle (today live only inside the FSM that never runs). One outcome-label API (pass/fail + feedback + regression-set delta). Unblocks best-of-N, BRT gates, and the first eval number. Our oracle is *stronger* than the Python frontier's. Moves: agent-work-ux, agent-architecture.
- **B4. FIM tab completion via Qwen2.5-Coder native tokens** (capabilities). Splice the FIM sentinels already in our tokenizer (151659 prefix / 151661 suffix / 151660 middle), bounded low-temp middle-only decode with the correct stop set. Use BASE weights not Instruct. Highest capability-per-effort item on the board; adds the entire missing tab-completion surface at near-zero marginal decode cost. Moves: agent-work-ux, product-differentiation.

**Tier 1 (the working agent, M):**
- **B5. Flat ReAct agent loop replacing the 12-phase FSM and the single-shot turn** (orchestration). Reimplement the mini-swe-agent loop inside the existing ToolLoop: seed `[system, instance]`; loop completion -> parse one action (fenced bash or tool-call) -> exec via sandboxed Command with per-command approval -> append result -> stop on sentinel + oracle + iteration cap. Do NOT inherit the 74%; expect ~20% Verified at 7B and *measure it*. Turns the product from a text completer into a working agent. Depends on B1. Moves: agent-architecture, agent-work-ux.
- **B6. Wire the context compiler + real tokenizer + rot-calibrated budget** (context). Stop discarding the reserve-then-fill output; feed it as the actual prompt. Replace chars/4 with the real .tq tokenizer so thresholds stop firing wrong. Reserve below the context-rot knee. Depends on nothing; unblocks the context stack. Moves: context-and-memory.
- **B7. Tiered eval gate and the first-ever coding/tool number** (capabilities). Expose OpenAI-compatible /v1, run batch=1 greedy in order: EvalPlus HumanEval+/MBPP+ pass@1 (week one), Aider Polyglot for edit-format compliance, BFCL offline subset, then the SWE-bench tier. Converts unwired-and-unmeasured into a live dashboard. Depends on B1 (FC-mode BFCL), B5 (SWE tier). Moves: agent-work-ux, product-differentiation.
- **B8. Constrained decode + jump-forward on the batch path (XGrammar-2 TagDispatch / llguidance)** (tool-calling). Vendor the Rust MIT llguidance crate (or FFI xgrammar), apply the per-step bitmask on a background CPU thread overlapping the Metal forward, emit ff_tokens for deterministic scaffold spans. Use TagDispatch to avoid the Tool-Suppression trap (naive masking silently emits zero tool calls). Overhead <0.25% of decode. Depends on B1. Moves: agentic-tool-system, hawking-serving-seam.
- **B9. Deterministic read-only verify gates + BRT fail-to-pass cogeneration** (verification). Read-only pre-dispatch gates at the ToolLoop boundary; for repair, emit fix + reproduction test in one pass and gate on the test flipping. +12.4pp tau2; P(correct)~0.87 when the cogenerated test passes. Depends on B3. Moves: agent-architecture, agentic-tool-system.

**Tier 2 (speed + the moat conversion, M/L):**
- **B10. Bit-exact Metal chain-verify gate** (speed). The one genuinely new Metal kernel: a narrow-chain verification forward returning per-position greedy argmax, longest-prefix acceptance, proven bit-identical to sequential greedy. No standalone speedup; it is the trust gate that makes the lossless lane real for B11 and B13. Reuse existing batched prefill as the verify pass. Moves: hawking-serving-seam.
- **B11. File-as-draft apply path (fast-apply / Predicted Outputs contract)** (speed). Feed a window of K original-file tokens as speculation each step, one batched forward, bulk-accept the longest greedy-matching prefix, fall back at the first mismatch. 3-13x on localized edits (collapses to 1x on heavy rewrites). Relative win is *larger* for us than a datacenter because our decode is bandwidth-bound at batch=1. Depends on B10. Moves: agent-work-ux, hawking-serving-seam.
- **B12. Hybrid best-of-N over RWKV state-forks (learned shortlist + execution tie-break)** (verification). Sample N trajectories, forking the RWKV state by memcpy for near-free branches; shortlist by a learned execution-free verifier, tie-break by the B3 execution signal. +7-8pt over either verifier alone, ~+16pt over pass@1. This is where the moat converts directly to accuracy. Depends on B2, B3, and a learned verifier trained on our Rust cargo/git labels (heavy). Moves: agent-work-ux, product-differentiation.
- **B13. SuffixDecoding drafter + narrow Metal chain-verify** (speed). Draft-free CPU suffix-tree drafter in Rust (global + per-request tree, Ukkonen online build, ~15-20us lookup, zero unified-memory bandwidth), feeding a narrow verify window. ~1.2-1.7x on the repetitive edit/repair slice at our sizes. Depends on B10. Moves: hawking-serving-seam.

**Tier 3 (context depth + memory, M/L):**
- **B14. Rust RepoMap: tree-sitter tags -> ref/def graph -> personalized PageRank -> budget-packed elided views** (context). Port Aider's repomap over the tree-sitter we already vendor. Whole-repo grounding in ~1k tokens, deterministic, no model call. Depends on B6. Moves: context-and-memory.
- **B15. Co-resident .tq tool-output pruner + structural code-protection guard** (context). Rip SWE-Pruner 0.6B (MIT) or Squeez 150M (Apache-2.0) as the compression stage the compiler discards; force line-index/span output so retained bytes are identical and prefix-cache keys stay stable. 23-54% token reduction, up to 26% fewer rounds. Depends on B6, B14. Moves: context-and-memory.
- **B16. Semantic text memory + hybrid retrieval (Mem0 extract-consolidate-retrieve)** (memory). Port Mem0's two prompts into SqliteMemoryStore + sqlite-vec with a non-cwd root, semantic+BM25+entity fusion, outcome-governed write ratchet. LoCoMo-class recall, ~90% token cut vs full context. Depends on B6, a local embedding model. Moves: context-and-memory.

## Strategic forks the research surfaced (owner rulings needed)

The completeness critic flagged six things the 8 domains under-owned that gate *our* deployment specifically:

1. **Security of local code execution.** The moment B5 ships, the agent runs model-emitted bash and cargo/git on the user's machine. The frontier treats Docker-per-task as too heavy for a local IDE but proposed no replacement. We need a sandbox model (macOS seatbelt/sandbox-exec, which hide-tools *already* uses for shell.run but not for the terminal RunCommand path), a per-command approval UX, path/network confinement, and tool-output injection defense. This is a shipping blocker that the audit's security grade does not yet reflect. **Rule on: sandbox strategy before B5 lands, not after.**

2. **Pure-RWKV vs hybrid, the model-track fork.** The SOTA local coder keeps 1-in-4 full-attention layers; pure linear attention has a hard multi-key exact-recall cliff (Gated DeltaNet 0.14 at 32k needle). Our pure RWKV-7 bet may need to become a hybrid (periodic full-attention or a bounded exact cache) to top out on retrieval-heavy fault localization. This gates the whole moat. **Rule on: is the RWKV track pure or hybrid.**

3. **The interactive latency budget.** Best-of-N, multi-sample verify, and next-edit rewrite trade wall-clock for accuracy, and at batch=1 N samples is literally Nx wall-clock. FIM tab completion needs a sub-100ms ceiling. Without a stated TTFT / total-turn budget the ranking cannot trade accuracy against felt responsiveness. **Rule on: the latency envelope for interactive vs background work.**

4. **The .tq quantization quality tax as a first-class workstream.** FIM infill, tool-format adherence, verifier ranking margin, and constrained-decode acceptance are all quant-sensitive. Nobody scoped measuring quant regression per capability as a standing gate; it can silently erode a narrow verifier margin. **Fold into B7 as a per-capability quant-delta column.**

5. **The Rust corpus / trajectory pipeline.** Every eval, learned verifier, and post-training lever needs Rust-labeled trajectories, and all released checkpoints are Python/SWE-bench and non-transferable. Harvesting our own flat-loop rollouts over real Rust repos, labeled by the cargo/git oracle, is a hard dependency for B12 and B16. **Scope as its own build once B3+B5 exist to generate rollouts.**

6. **Multi-file edit transactionality and observability.** FIM and file-as-draft handle single-file edits; whole-repo edit-format compliance (Aider Polyglot percent-well-formed), transactional multi-file apply/rollback, and surfacing the loop's steps to the user determine whether an edit is trustworthy and whether the user can follow and stop the agent. Maps to interaction-ux and ship-readiness. **Track alongside B5/B11.**

## Provenance

81 agents: 8 domain scouts + 40 deep-reads + 24 adversarial batch=1 verifications + meta-synthesis. Sources are arxiv 2024-2026 plus official repos and engineering writeups, cited per method in the full structured output (session scratchpad). Every rip/adapt verdict passed a batch=1-on-Metal skeptic before inclusion. Headline benchmark numbers are model-bound and explicitly NOT our expected numbers; the first real number comes from B7.
