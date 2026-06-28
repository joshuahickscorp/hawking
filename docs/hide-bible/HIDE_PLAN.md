# HIDE: The Plan

> The single authoritative HIDE document. This consolidates the strategy, the roadmap, the binding design doctrine, and the product/contract reference into one source of truth, replacing the scattered front-end chapters. The backend is built and tested (11 Rust crates, a real Planner-Executor-Verifier agent loop, a runnable headless `BackendHost`): see [`SCAFFOLD_STATUS.md`](SCAFFOLD_STATUS.md). The old numbered chapters are archived under [`archive/`](archive/). When this document and any archived chapter disagree, this document wins.

---

## Table of contents

**Part A: Strategy**
- [A1. The spine](#a1-the-spine)
- [A2. Where we stand today](#a2-where-we-stand-today)
- [A3. The wedge, the MLP, positioning](#a3-the-wedge-the-mlp-positioning)
- [A4. The proof demo: five futures, one read](#a4-the-proof-demo-five-futures-one-read)
- [A5. Risks and the honest edge](#a5-risks-and-the-honest-edge)
- [A6. Improvements and refinements](#a6-improvements-and-refinements)
- [A7. The immediate next 3 actions](#a7-the-immediate-next-3-actions)

**Part B: The Roadmap**
- [B1. The unified M0-M8 milestone table](#b1-the-unified-m0-m8-milestone-table)
- [B2. The critical path, and what parallelizes now](#b2-the-critical-path-and-what-parallelizes-now)
- [B3. The design to capability synergy map (S1-S11)](#b3-the-design-to-capability-synergy-map-s1-s11)

**Part C: The Design Doctrine**
- [C0. The spine (design)](#c0-the-spine-design)
- [C1. North star / ethos](#c1-north-star--ethos)
- [C2. Brand and identity](#c2-brand-and-identity)
- [C3. Color and theme](#c3-color-and-theme)
- [C4. Typography](#c4-typography)
- [C5. Density and layout](#c5-density-and-layout)
- [C6. The three surfaces and how they relate](#c6-the-three-surfaces-and-how-they-relate)
- [C7. The Context Stack](#c7-the-context-stack)
- [C8. Agent presence and aliveness](#c8-agent-presence-and-aliveness)
- [C9. Parallel and overnight agents](#c9-parallel-and-overnight-agents)
- [C10. Interaction model](#c10-interaction-model)
- [C11. Motion and micro-interactions](#c11-motion-and-micro-interactions)
- [C12. Voice and copy](#c12-voice-and-copy)
- [C13. Distinctiveness: the one thing](#c13-distinctiveness-the-one-thing)
- [C14. Hard constraints](#c14-hard-constraints)
- [C15. Self-check: tells that we are failing the doctrine](#c15-self-check-tells-that-we-are-failing-the-doctrine)

**Part D: Product & Contract Reference**
- [D1. The three surfaces (consolidated)](#d1-the-three-surfaces-consolidated)
- [D2. The backend contract](#d2-the-backend-contract)
- [D3. The OSS harvest map](#d3-the-oss-harvest-map)
- [D4. The front-end build steps and milestones](#d4-the-front-end-build-steps-and-milestones)

---
---

## Part A: Strategy

## A1. The spine

A black hole is the ultimate black box: nothing escapes, you cannot see in. Hawking proved that is false. Black holes radiate. That single idea is the whole product, and it is the same idea told three times, at three layers.

At the **story** layer it is the brand: Hawking Condense compresses (drives matter toward singularity density, the ultimate compressor), HIDE radiates (makes the agent's work escape and become visible). Two faces of one physics. The dual face resolves the ironic name: HIDE hides you from the cloud (local, nothing leaves your machine) and hides nothing from you (the Context Stack). Privacy outward, transparency inward.

At the **pixels** layer it is the doctrine: an observatory, not a cockpit. Near-black material surfaces wearing one luminous warm-gold rim-light, with confident Cormorant Garamond display over Geist Mono chrome. The gold edge is the box radiating, made into a consistent visual device: it sits on the active agent, the approval gate, the streaming edge, the mark, the live stratum of the Context Stack. When a thin gold glow sits on a dark recessed panel, that is HIDE and nothing else.

At the **architecture** layer it is the capability moats: pass state not text (M1), free local fleets (M2), radical transparency (the observability seam), grammar-guaranteed tool calls (M4). These are structural advantages of owning a constant-size recurrent state on-device, not features bolted onto a chat box.

The load-bearing insight that makes this a plan and not a wish: **the design already assumes the capability.** The doctrine's "fork and try N", its "instant resume", the overnight agent board with its morning digest, the calm no-jank tool calls: each is literally a moat. "Fork and try N" is M1 state clone plus M2 free fleets. "Instant resume" is M1 state serialize. The calm tool feed is M4 first-try-valid masking. So design and capability are not two things to integrate later. They are one object seen from two sides, and the only real planning question is sequencing: land each capability just in time for the UI that expresses it, never earlier (do not gold-plate a primitive with no UI) and never later (do not ship a surface that lies about a stub).

## A2. Where we stand today

The **backend is done** and honest about it. Eleven crates, ~410 tests passing, the agent loop real (`SCAFFOLD_STATUS.md`): `hide-kernel` is an audited-genuine Planner-Executor-Verifier FSM with deterministic oracles that shell to real cargo and git; `hide-backend` is a runnable host with `RuntimeSupervisor`, `CommandRouter`, push `UiEvent` bus, session registry, time-travel scrub and fork; `hide-fleet` has real worktrees, a `FleetGovernor`, and a 3-way merge funnel; `hawking-index` is a built RAG path (tree-sitter, merkle, FTS5, embeddings, RRF, rerank).

The **design is locked** (Part C): tokens, type, motion, voice, the three surfaces, the Self-check ship gate. The **front end is specced** but not started: contract types frozen, OSS harvest mapped, build sequencing written.

The **one blocker is native `.tq` serving** (caveat H9): `qwen_dense.rs::load` has no `.tq` branch, so there is no live model behind the host. Nothing streams a real token until that lands. Every downstream moat, and the GO/KILL verdict on whether the local model can even code, sits behind that single seam.

## A3. The wedge, the MLP, positioning

### The wedge (the one thing)

**"Fork and try 5, keep the best, for free, and watch all five radiate."** A dev highlights a function, hits one key, five agents fan out down five approaches at once. The whole repo is already warmed into each one (instant, no re-prefill), so they start thinking immediately, not loading. Five gold cards breathe; each streams its real moves. You take the winner's diff hunk by hunk. No bill, no limit, no spinner.

Why it converts: every metered competitor charges per agent per token, so "spawn 5 and throw 4 away" is the exact thing their pricing punishes. We make the punished action the default verb. A Cursor/Devin user feeling the bill spike sees this and the math is instant and visceral. Moat tie: M1 (the warm state forks free via `state.clone()` memcpy) plus M2 (zero marginal cost makes "keep the best" the default, not a paid tier). Both are structural: cloud cannot match free fleets without going bankrupt. And it IS the doctrine's headline surface, the Workstation board of cards lit at the rim.

### The MLP (minimum lovable product)

The smallest daily-usable slice that proves a real moat AND the doctrine. Not a demo, a thing you reach for.

| IN | OUT (deliberately, for the MLP) |
|---|---|
| Native `.tq` serving (Phase 0): the unblock, 32B local and free | Telepathic state handoff between roles (M5): deepest moat, needs the tap, wrong debut |
| Instant resume + state save/load (M1, low effort) | Personalization flywheel / LoRA (M7): compounding, slow to show, MLX-gated |
| Fork and try N (the wedge): one-key fan-out, board of gold cards | Multi-tier SKUs / doctor / INT8 state (M8) |
| Identical hunk-by-hunk diff review (j/k/a/r) in IDE and merge | EAGLE-3 / spec-decode (the internal "Event Horizon" lane): invisible, batch-gated |
| Context Stack as the persistent spine, with pin/unpin minimally real | Overnight fleet + morning digest: gorgeous, but the second chapter not the wedge |
| The shell, doctrine-correct (rim-light, Cormorant + Geist Mono, no spinners) | Light mode, plugin/WASM host, cross-machine cluster, frontier BYO-key |
| Verifiable no-egress: air-gap toggle with packet-capture proof | |

### Positioning

30-second pitch: "Cursor and Copilot just turned into taxi meters: bills spiked and a third of devs hit their limit mid-task. Hawking runs a real coding model entirely on your Mac, so it is free forever and nothing leaves the machine. And because we own the model's state, you can fork your whole warmed-up project five ways, run five agents down five approaches at once for zero marginal cost, and watch every one work in plain sight. Most AI tools are a chat box bolted onto a black box. We are the box, opened, lit at the rim, and yours alone."

Taglines (per doctrine): **"Open the box."** (the thesis in three words). **"Nothing leaves your machine. Nothing's hidden from you."** (the dual face). **"Free fleets. Full daylight."** (the wedge).

The anti-metering, anti-black-box sentence: "They put your code behind a meter and an agent you cannot see into; Hawking runs the whole thing on your machine, for free, with every move radiating in plain sight."

A naming note: do not lean on "Event Horizon" publicly. In this repo it is already the internal codename for the speculative-decode proposal engine (`crates/hawking-core/src/speculate/*`). Keep it internal. The brand mines the same physics one level up: the product is HIDE (the radiating box), the maker is Hawking, the mark is the event-horizon ring. Use the event horizon as glyph and story, not as a shipped-feature name.

## A4. The proof demo: five futures, one read

**"Five futures, one read."** One ~75-second continuous screen recording that proves story plus pixels plus architecture in a single take. The viewer comes away thinking: this is the box that radiates, and it just did something the cloud structurally cannot.

**Setup (t=0):** HIDE open on the Workstation front door, dark observatory, gold ring mark top-left. One session already warm: the planner has read the whole project once, the Context Stack rail is lit, one stratum breathing gold. Status pill: `ready`.

1. **t=0-8s. One prompt against a warm state.** User types: "Try five different fixes for the flaky auth retry, keep the one that passes." (`SubmitTurn`.) The lit rail is the proof the repo is loaded, not re-read. Narration in dry mono: "State warm. Forking."
2. **t=8-15s. The fork (M1 + M2).** One control: **Fork x5** (`ForkSession` x5 on the live state, then `fleet_run` per branch). The warm card splits into five, each with the same lit lineage. The beat that sells the architecture: a thin gold radiation edge travels from parent to each child at the moment of fork. That gold edge **is the state memcpy**, rendered. No five spinners, no "prefilling 5/5". Cormorant number resolves: **"5 branches. 0 re-reads."**
3. **t=15-50s. Five futures run free, in parallel, legibly.** Five cards breathing gold, each scrolling one real-work line ("Patching retry.ts", "Running 12 tests", "3 failed, fixing"). No meter anywhere. One card hits a tool call: the `ToolProgress` chip appears and the JSON lands valid first try (M4/M6), so the feed stays calm, no red repair churn.
4. **t=50-65s. The handoff tap (M1 + H7).** On the winning branch, planner hands to reviewer. The viewer clicks the gold handoff edge: it expands the text-decode tap stratum, showing in plain text what state was passed (not re-typed). The single most uncopyable frame: a competitor passes text and pays a full re-prefill; HIDE passes state and shows it to you.
5. **t=65-75s. Resolution, the observatory at dawn.** Four branches quiet to amber/green; one settles green ("tests passed"). The digest number resolves: **"5 ran. 1 passed. 4 archived. 0 billed."** User accepts the winning diff (`AcceptDiff`, same j/k/a/r gesture); it absorbs cleanly. The rail dims from breathing gold to steady. Quiet completion, no fireworks.

**Moats exercised at once:** M1 (fork at t=8, handoff tap at t=50), M2 (free five-way fleet, "0 billed"), M4 (first-try valid tool call), the Context Stack lit gold throughout (transparency as the constant spine). The gold rim carries every one, so brand color and architecture are the same pixel.

**Minimum to film it:** Phase 0 native `.tq` (real free decode), M3 `RwkvState` clone/serialize (the load-bearing build, makes the fork a genuine memcpy), M5 `copy_kv_prefix_to_slot` (the tap). The rest is already real (`hide-fleet`, `ToolProgress`, `context.compile`, `AcceptDiff`, the durable log). **The one new visual primitive to build:** the gold radiation-edge animation on fork and on handoff. It is the single asset that renders the memcpy; everything else is composition of built parts.

**Why metered/closed competitors cannot copy it:** the fork is free and instant only because we own the state (a transformer cloud must re-prefill or wrestle a quadratic KV cache for each branch, five times the bill, a claim their architecture cannot make). The handoff passes state not text and proves it on screen (their passed state is server-side, in their black box, unexposable). And the calm itself is the moat: five agents, no meter, one read is only calm because the work is free (M2) and the tool calls are valid first try (M4). The serenity is structurally local.

## A5. Risks and the honest edge

Almost every risk traces to one of two things: one untested load-bearing assumption (RWKV-7 can code, H3) and one unbuilt load-bearing artifact (native `.tq` serving, H9). Ranked by likelihood times impact.

| # | Risk | Likelihood | Impact | Mitigation | Owns |
|---|---|---|---|---|---|
| R1 | Native `.tq` slips; blocks gate AND all moats (SPOF) | High | Critical | Split Stage A (F16-on-load) / Stage B (bitslice); gate needs only Stage A; timebox Stage A to 2 wks | Phase 0 |
| R2 | Thesis gate returns KILL/CONDITIONAL (H3) | Medium | Critical | Run gate ASAP on Stage A; eval RWKV-7 AND Qwen `.tq` so CONDITIONAL has a fallback model, not a dead end | Phase 0 |
| R3 | FE over-built on an unproven model premise | Medium | High | Build only model-independent FE pre-gate; hold moat-claim surfaces | FE M1 |
| R4 | "Instant memory / perfect recall" over-promise (H1/H2) | High | High | Copy says "resumed, not re-read"; pull RAG (M4) earlier; never market infinite memory | M3->M4 |
| R5 | Harvested modules ship as patchwork (Monaco looks like VS Code) | Medium | High | Doctrine Self-check as CI gate from FE commit 1; budget Monaco re-skin explicitly | FE M1 |
| R6 | Latent handoff ships without audit tap (brand contradiction, H7) | Medium | High | Tap is a ship-blocker for M5, same tier as the moat | M5 |
| R7 | "Free fleets" browns out the laptop (RAM/thermal) | Medium | Medium | Energy/thermal admission gates fan-out; market "cheap" not "free"; cap default N | M3->M4 |
| R8 | Personalization makes garbage adapters via MPS (H5) | Medium | Medium | Hard-ban MPS for training; MLX only; held-out eval gate on every deploy | M7 |
| R9 | "Grammar-guaranteed" is a literal lie on hard schemas (M4) | Medium | Medium | Soften copy to "first-try valid"; keep repair fallback | M6 |
| R10 | Scope creep down the feature catalog | High | Low-Med | Treat the catalog's lower rows as menu, not backlog; nothing past M6 until M0-M4 hold | All |
| R11 | Energy/tps marketing numbers inflate (23x) | Medium | Medium | Single source-of-truth measured figures; report ~2-4x tokens/watt; ban unverified multipliers | Cross-cutting |

**The honest edge (the H-caveats, kept visible):** H1 SSM recall breaks on hard needles, route exact recall through RAG, never say "perfect memory". H2 warm state saves recompute, it is not memory quality. H3 the load-bearing assumption is that RWKV-7 can code; the thesis gate tests it. H4 spec-decode is occupancy-gated and can go negative under the very fan-out the wedge runs in. H5 personalization is LoRA/small-N DPO, not full RLEF, and must be MLX-only and eval-gated. H6 perplexity is not generation quality, gate on real generation. H7 latent handoffs are unauditable without the decode tap. H8 do not let a frontier BYO-key escape hatch quietly become the thing that makes demos work, or the local pitch collapses. H9 native `.tq` serving is the one unbuilt load-bearing artifact behind everything.

**Wow-vs-prove recommendation: prove-first on the critical path, with a deliberately throwaway GGUF wow-spike running strictly in parallel, capped at one engineer-week.** The telepathy/state-fork moats are worth nothing if the thesis gate returns KILL; building the headline UX on a model that cannot code, secretly propped by a frontier escape hatch, is exactly the failure the product exists to reject. But six weeks of plumbing with nothing to show drains belief, so the GGUF spike keeps morale alive and validates the "fork and try N" UI feel early. It must be labeled internally as **mechanism proof, not technical derisking** (it rides the wrong serving path, so its code is mostly thrown away). It must never delay the gate.

## A6. Improvements and refinements

This is the section the inputs did not write on their own: where the unified plan is sharper than its parts. Opinionated.

**A6.1 The single highest-leverage move: split native `.tq` into Stage A and Stage B, and gate ONLY on Stage A.** Native `.tq` is the SPOF and the thesis gate is the scariest decision, but treated together they compound into one six-week bet that resolves late. Decoupled, they become a fast cheap experiment. Stage A (F16 dequant-on-load) is low-risk wiring into `qwen_dense.rs::load` and answers the only question that matters first: can the local model code? Stage B (native bitslice GEMV) is the RAM-cliff perf win and the "32B-on-18GB" headline, but it does **not** gate the GO/KILL decision. A Stage B slip cannot block the verdict. This turns R1 and R2, the two Critical risks, into a week-3 yes/no instead of a multi-week gamble. Do this first.

**A6.2 Pull RAG (M4) forward, ahead of the state primitives it currently sits behind.** The roadmap sequences recall after M3, but M4 is the actual fix for the biggest honesty risk in the whole pitch (H1). If instant resume (M3) ships first and users infer the warm state remembers their repo, you manufacture a false-memory expectation before the machinery that satisfies real recall exists. `hawking-index` is already built, so wiring it as the agent's recall path is integration, not invention. Land RAG concurrently with or even slightly before the instant-resume surface, and let M3's copy be strictly "resumed, not re-read". This is a sequencing inversion, not just a copy fix.

**A6.3 Make the gold radiation-edge primitive the first real FE asset, not a late-demo polish.** The proof demo identifies exactly one new visual primitive (the radiation edge that renders the memcpy on fork and handoff). It is the single pixel where story, design, and architecture become literally the same thing. Build it early, on the stubbed `hide-serve` path, against replayed event logs, so the headline motion is locked and battle-tested long before M3/M5 make it real. This de-risks the most brand-load-bearing frame in the product while the backend critical path runs.

**A6.4 Cut the overnight digest and telepathic handoff from v1, but keep their containers.** Both are gorgeous and both tempt as the debut. The digest is a second-session feature with no daytime proof loop, and it is just the fleet board resolved at dawn (so the wedge already exercises its primitives). Telepathic handoff is the deepest moat but invisible to a first-time viewer and brand-dangerous without the audit tap. Build the containers now (the board is a grid of cards, the digest is a Cormorant number) because they are pure design and risk-free, but do not wire either to a capability claim in v1. Ship "fork and try N" first; both of these are chapter two.

**A6.5 Make the doctrine Self-check a CI gate from FE commit one, and budget the Monaco re-skin as real work.** Under schedule pressure Monaco ships looking like Monaco (VS Code blue, sans, flat) and trips the Self-check's "could be mistaken for VS Code." The fix is process, not vigilance: wire the Self-check tells (blue/purple on screen, flat surfaces, a spinner standing in for real work, a visible harvested seam) into review/CI from the first commit, and put the Monaco-to-mono/near-black/gold re-skin on the schedule as a budgeted task, not an assumed polish pass.

**A6.6 A synergy neither source fully saw: the diff-accept gesture is the personalization corpus, so instrument it on day one.** S10 logs accept/reject and M7 consumes them, but they are described as separate milestones far apart. They are one flywheel with a long fill time. The accepted/rejected diff stream (which the MLP already ships, because the hunk-by-hunk gesture is in v1) is exactly the DPO corpus M7 needs. If the corpus is logged cleanly and scrub-on-write from the very first diff review, M7 has months of real signal the moment it starts, instead of starting cold. The cost is near zero (the logging seam exists in `hide-personalize`); the payoff is that the slowest-compounding moat starts its clock at MLP launch, not at Phase 5. Wire the corpus capture now even though the training stays deferred.

**Two smaller cuts.** Drop the EAGLE-3/spec-decode lane entirely from any user-facing roadmap conversation: it is batch-occupancy-gated (H4) and can go negative under the fan-out the wedge runs in, so it is the one optimization that actively fights the headline. Keep it internal-only. And resist the feature catalog as a backlog: entropy-triggered tools, custom samplers, INT8 state quant, AST grammars are a menu of research bets, not commitments; nothing past M6 starts until M0-M4 ship and hold.

## A7. The immediate next 3 actions

1. **Land native `.tq` Stage A in `qwen_dense.rs::load`** (F16 dequant-on-load), timeboxed to two weeks, and in parallel scaffold the `hawking-eval` coding bench. This is the critical path and the SPOF; everything waits on it. Stage B (bitslice GEMV) starts only after the gate is GO.
2. **Start `hide-serve` (axum HTTP/WS adapter wrapping `BackendHost`) plus the FE web shell** (Vite, `wire.ts`/`ipc.ts`, EventRouter, Zustand stores, three empty surface frames, the Context Stack rail skeleton) against a stubbed serve. Zero model risk, fully parallel, and it includes building the gold radiation-edge primitive (A6.3) early.
3. **Run the thesis gate on Stage A against both RWKV-7 and Qwen `.tq` as soon as Stage A streams a token.** This is the GO/CONDITIONAL/KILL decision and it must arrive at week three, not week six. Branch all downstream work on its verdict; a CONDITIONAL routes the weak axis (likely recall, H1) into the pulled-forward RAG wire-up (A6.2), not into a dead end.

---
---

## Part B: The Roadmap

## B1. The unified M0-M8 milestone table

The interlocked milestones. Each row is one backend capability deliverable (Track A) married to its front-end deliverable (Track B), the thing the user actually sees ship, the hard dependency, and the honest caveat. The ordering is the spine made operational: capability just in time under a frozen UI.

| Milestone | Backend / capability (Track A) | Front-end (Track B) | What the USER sees ship | Hard dependency | Honest caveat |
|---|---|---|---|---|---|
| **M0 Today** | 11 crates done; agent loop real; host with router and bus; `.tq` decode test-only | Design locked (Part C); OSS harvest map done (D3) | Nothing yet, internal | none, baseline | Blocker is native `.tq` serving (H9); no live model behind the host |
| **M1 Shell lights up** | `hide-serve` axum HTTP/WS adapter wrapping `BackendHost`; runtime on Qwen `.tq` Stage A (F16 dequant-on-load) | Vite web shell, `ipc.ts` client, EventRouter, Zustand stores, 3 surface frames, Context Stack rail skeleton | A running IDE: editor, terminal, chat composer streaming tokens, live Context Stack | `.tq` Stage A (Phase 0.1) | Stage A is F16 fallback, not the RAM-cliff win; quality unproven (no eval yet) |
| **M2 Thesis gate** | Native bitslice GEMV (`.tq` Stage B); `hawking-eval` coding bench; run gate (>=15 tok/s @32B, >=40% @7B) | Status pill + degraded banners bound to `RuntimeStatus`; runtime-not-ready gating on composer | "32B running on 18GB" plus an honest GO/KILL verdict on coding | Stage A (M1), eval harness | GO/KILL is real; CONDITIONAL means re-quant before features. PPL is not generation (H6) |
| **M3 State primitives** | `RwkvState::{to_bytes,from_bytes,clone}`; `Engine::{save,load}_checkpoint`; fork as engine primitive | Timeline scrub/fork UI (`ScrubToEvent`/`ForkSession`); "fork and try N" affordance | Instant resume (no re-prefill), session undo, fork a branch | thesis GO (M2); RWKV-7 path | State is a recompute saver, not memory quality (H2) |
| **M4 Recall** | Wire `hawking-index` RAG as exact-recall path; optional sliding-window hybrid | Search surface (IDE), provenance peek, retrieval rail panel | Whole project loaded plus reliable retrieval, "it found the right file" | M3; `hawking-index` (built) | Never market "perfect memory"; route exact recall through RAG (H1) |
| **M5 Telepathic agents** | State-passing handoff (memcpy) planner to coder to reviewer via `copy_kv_prefix_to_slot`; text-decode audit tap | Workstation fleet board: state pills, handoff viz, parallel runs; `fleet_run` wired | Multi-agent flows, ~4x faster handoffs, far fewer tokens | M3 (state fork) | Handoffs unauditable without the tap (H7); fan-out wins exploration, not coupled edits |
| **M6 Speed** | Harden `json_constrain` + AST grammars; prefix-cache discipline; EAGLE-3 head; speculative tool exec | No-jank tool-call chips, grammar-valid diff chips, parallel-tool progress | First-try-valid tool calls, snappier loop, no JSON-repair stalls | M4 (index to AST grammar); M2 (logit access) | Grammar validity ~93-96% on hard schemas (keep fallback); spec-decode gated on occupancy (H4) |
| **M7 Personalization** | Tiny-data DPO/SFT on accepted/rejected diffs (MLX); RWKV state skill-seeds; QA-LoRA re-bake | `personalization` views; "model learned your style"; mode switch | Model gets better at your code weekly; per-mode states | M5 (handoff corpus) + `hide-personalize` (built) | LoRA/small-N DPO only, not full RLEF (H5); keep adapter swappable, eval-gated |
| **M8 Format SKUs** | Multi-tier `.tq` (sensitivity map + mixed bpw): 3-bit lossless + 2-bit recovered; "doctor" KD; INT8 state; air-gap | SKU picker, packet-capture air-gap proof surface | Pick your quality/RAM tier; verifiable no-egress privacy | M2 (format proven) | Report effective bpw, honest labels, test real generation (H6) |

## B2. The critical path, and what parallelizes now

The longest hard-dependency chain (the spine):

```
native .tq Stage A (M1) -> native .tq Stage B / bitslice GEMV (M2)
  -> hawking-eval coding bench -> THESIS GATE (GO) (M2)
    -> RwkvState save/load/clone + state fork (M3)
      -> telepathic handoff via copy_kv_prefix_to_slot (M5)
        -> personalization flywheel: handoff corpus -> DPO/QA-LoRA re-bake (M7)
```

**Nothing downstream of the thesis gate is worth building until the gate is GO.** A KILL or CONDITIONAL verdict on coding quality re-routes effort back into Condense/quant, not into features. Building the telepathy UX on a model that cannot code is the exact "chat box on a black box" pattern the product exists to reject.

**Fully parallel-buildable today, zero dependency on the `.tq` blocker:**

| Stream | Why it is unblocked |
|---|---|
| `hide-serve` axum HTTP/WS adapter | Wraps the already-built `BackendHost`; needs no live model. The single highest-leverage unblocked task. Mirror `hawking-serve`. |
| The entire FE web shell | Vite app, `wire.ts` (TS mirrors of `api.rs`), `ipc.ts`, EventRouter, Zustand stores, 3 empty surface frames, Context Stack rail. Contract types frozen; develop against a stub `hide-serve` or replayed event logs. |
| `hawking-eval` harness | Only needs the model at run time, not build time. Write the coding bench now. |
| OSS module harvest / re-skin | Ripping Monaco diff review, xterm, explorer into designed slots is pure FE. |

**Truly blocked, and on what:** any live decode (the thesis verdict, every M1+ capability feature) is blocked on native `.tq` Stage A. All state primitives, handoff, and personalization are blocked on the thesis GATE. The RAM-cliff "32B-on-18GB" headline is blocked on Stage B bitslice GEMV (Stage A F16 fallback does not deliver it).

**Practical consequence:** one stream on Phase 0 (`.tq` Stage A to B + eval), and the whole FE shell plus `hide-serve` as a fully parallel stream. They converge at M1, where the lit shell first talks to a live runtime.

## B3. The design to capability synergy map (S1-S11)

The design is not waving at vague future power. It pins to four specific capabilities (M1 fork, M1 handoff, M1 instant-resume, M3/M7 flywheel) plus the universal Phase 0 unblock. Each doctrine element below is an expression of a moat, with its exact contract seam and its just-in-time debt (the capability the pixels assume but that is not yet built). Ship the pixels of each only when its moat lands, or the screen lies.

| # | Design element | Moat / feature it IS | Pixels | Contract seam | JIT debt |
|---|---|---|---|---|---|
| S1 | Gold rim-light = agent active/streaming | M1 state decoded; the box radiating | Breathing `--radiation` glow; gold leading-edge cursor | `TokenBatch{stream_id,text}` on Wire-B | No. Needs only a real `.tq` behind the stub (Phase 0). |
| S2 | The Context Stack = transparency made visible | Observability moat; "show the model's mind" | Always-visible right rail of strata, one live stratum in gold | `context.compile` -> `{prompt, manifest}`; `ProjectionPatch` | Partial. Compile is real; "what it saw when it decided THAT" rewind needs M3 checkpoints. |
| S3 | "Let me touch it": pin/evict/mute/inject | Transparency-as-control | Pin/x/drag-handle per stratum | `Custom{pin_span/unpin_span}`, `resolve_conflict` | No for pin/drop. Yes for inject-a-skill-seed-state (M7). |
| S4 | "Fork and try N" | **M1 state clone + M2 free fleets** | Board of cards spawned from one warm state | `ForkSession{at_event}` + `fork_session`; `fleet_run` | **YES, headline debt.** `fork_session` rebuilds a projection today; true free fork is `state.clone()` memcpy (M3). |
| S5 | Agent board (cards, not swarm) + glow status | **M2 free fleets as default** | Calm grid, per-card live line, breathing-gold/green/amber | `hide-fleet` (real) + `ProjectionPatch{fleet/run/merge}` | No. Fabric is built. Debt is only the per-agent decode being free (Phase 0 + batch=8). |
| S6 | Morning digest / observatory at dawn | M2 overnight swarms | One editorial Cormorant big-number | Durable event log replay; `fleet_run` terminal status | Partial. Log + fleet real; assumes M2 economics and M3 resume. |
| S7 | "Progress without anxiety", real work not spinner | **M4 grammar-valid tool calls** + transparency | Live-feed stratum scrolls real moves; no % bar | `ToolProgress{call_id,message}`; `json_constrain::mask_logits` | Partial. `ToolProgress` real; the calm (no repair churn) needs M6 first-try-valid. |
| S8 | Telepathic handoff legibility / the audit tap | **M1 state handoff (pass state not text)** | Handoff edge planner->coder->reviewer; expandable decode tap | `kv_handoff` + `KvShareGroup` -> `copy_kv_prefix_to_slot` | **YES.** Documented deferred seam. For RWKV it is a memcpy (M5). |
| S9 | The "gate" verb: lit approval control | M4 consequential action + HITL | Bright-gold rim-lit button, plain statement | `SecurityGate{gate,message}` -> `Custom{approve_gate}` | No. Permission engine real. Pure FE. |
| S10 | Diff lands cleanly, identical gesture everywhere | M7 flywheel (accept/reject is the signal) | Per-hunk j/k/a/r; same gesture in merge | `AcceptDiff`/`RejectDiff` -> `personalization.records.append` | Partial. Logging real; compounding needs M7 re-bake. |
| S11 | Instant resume, agent instantly warm | **M1 state serialize/save/load** | Front door shows warm sessions; reopen resumes exactly | `RwkvState::{to_bytes,from_bytes}`; `prefill_slot` | **YES.** Not built (M3). "Instant" in the front door is a direct debt (H2: recompute-saver, not memory). |

Operationally: **Phase 0 (native `.tq`) lights up the gold rim and the cards; M3 (`RwkvState` clone/checkpoint) lights up "fork and try N" plus instant resume; M5 (state handoff) lights up the telepathic edge; M7 lights up the flywheel.**

---
---

## Part C: The Design Doctrine

> This is the **binding design system**: the look, feel, tokens, type, motion, voice, and interaction rules every surface and every harvested module conforms to. When this doctrine and a harvested module disagree, the doctrine wins, and the module gets re-housed and re-skinned. The Self-check ([C15](#c15-self-check-tells-that-we-are-failing-the-doctrine)) is the ship gate.

> One line: HIDE is the IDE named after the man who proved the black box leaks. A black hole is the ultimate black box, nothing escapes, you cannot see in. Hawking proved that is false: black holes radiate. HIDE makes the agent's black box radiate everything it sees and does. The whole product is that one idea expressed in pixels.

> The dual face (resolves the ironic name): HIDE hides *you* from the cloud (offline, local, nothing leaves your machine) and hides *nothing* from you (the Context Stack). Privacy outward, transparency inward. Same object, two faces, exactly like a black hole.

## C0. The spine (design)

Read this part first. Everything below is downstream of it.

**The feel:** an observatory, not a cockpit. You sit in the dark and watch enormous work happen with serene clarity. The danger with "show everything" is that it becomes a cockpit: alarms, gauges, white knuckles. We reject that. The agent does vast work; you observe it with the calm of an astronomer, not the stress of a fighter pilot. Calm, powerful, instrument-grade.

**The one adjective we optimize the whole design for: legible.** Not "clean," not "minimal," not "fast," not "powerful" (everyone in this space claims those). Legible. You can read what the agent is doing at a glance, the type is legible, the layout is legible, every state is legible. Radical transparency only works if it is legible, otherwise it is just noise, which is the failure mode of every "show the logs" tool. Legibility is also the craft value that separates material brutalism from vibecoded mush. Two supporting qualities under it: **calm** (so transparency never becomes a cockpit) and **material** (so it never becomes cheap flat).

**The through-line, the one thing that makes HIDE instantly HIDE:** a single luminous warm gold rim-light on near-black material surfaces. Everything alive or important wears a thin gold edge against deep near-black: the active agent, the approval control, the streaming edge, the mark itself, the live stratum of the Context Stack. It is the black box that radiates, made into a consistent visual device. It continues your CX anodized-capsule language (dark recessed surfaces, inset rim, glowing label) directly. When you see a thin gold glow on the edge of a dark recessed panel, that is HIDE and nothing else.

**The brand family insight (use this everywhere):** Hawking Condense and HIDE are the two phenomena of a black hole. Condense *compresses* (the model-maker drives matter toward singularity density, the ultimate compressor). HIDE *radiates* (it makes the agent's work escape and become visible). Compression and radiation. One family, two faces of the same physics. That is a tight, honest, ownable story and it is grounded in what the two products literally do.

## C1. North star / ethos

**Feel:** a calm, powerful instrument. Specifically an observatory. Not a friendly pair-programmer (too soft, too chatty, undersells the power and the transparency), not a fast dense cockpit (the thing we must actively design against, because "show everything" tends to drift there).

**Channel (look and feel to borrow from):**
- **Linear.** The gold standard for dev-tool craft: dark, fast, keyboard-first, command palette, immaculate micro-interactions, zero jank. We borrow the craft floor and the speed. We do not borrow the purple.
- **Things 3.** Calm, restraint, generous breathing room, opinionated curated views, delight that never gets loud. We borrow the calm and the opinionation.
- **Teenage Engineering (OP-1 / field gear).** The instrument: every control labeled, tactile, precise, functional-technical, material aluminum. This is the soul of "show everything and let me touch it." We borrow the labeled-instrument feel and the tactility.
- **032c.** Editorial typographic confidence: big assured display type, the big-number-as-statement moment. We borrow the confidence to set huge serif headings and editorial numbers in a tool that "should" be all small sans.
- **Aesop.** Muted premium restraint, warm neutrals, material, lots of negative space, considered. We borrow the restraint and the warmth.

One non-software reference worth keeping pinned: the Event Horizon Telescope image of M87\*. Black core, warm gold photon ring. That image *is* the brand.

**Avoid (deliberately):**
- **VS Code.** The clone trap. Generic chrome, the blue, infinitely-dockable patchwork, no point of view. This is the gravity well we are escaping.
- **Cursor.** A chat panel bolted onto VS Code. It is exactly the "chat box on a black box" pattern the product exists to reject. We must not look like it.
- The category to avoid as a whole: the vibecoded AI-startup look (purple-blue gradients, glassmorphism, rounded-everything, Inter plus a gradient logo, the ChatGPT-wrapper aesthetic). And its dark cousin, the fake-hacker neon-green terminal.

## C2. Brand and identity

**How "Hawking" reads visually:** cosmic and black-hole, yes, but as a *conceptual spine expressed minimally*, never as literal galaxy wallpaper or starfields. Material brutalism, not a screensaver. The cosmos lives in the mark, the single gold accent, and the radiation language. The chrome stays disciplined near-black.

**The mark:** the event horizon. A precise circle, a black disk with a thin luminous gold rim, abstracted from the EHT photon ring. The mark literally encodes the thesis: a black core (the unknown, the black box) that radiates light at its edge (the information escaping, the transparency). This is the CX rim-light treatment applied to a circle. It scales from a 16px favicon to a hero. Optional refinement: a faint asymmetry in the rim brightness (the accretion-disk lean) so it reads as observed light, not a UI ring.

**Wordmark:** Geist, logo only, per your standing rule. This is the single place Geist appears anywhere. "HIDE" as the product wordmark, "Hawking" as the maker line, the event-horizon ring as the family glyph that sits before the word. Geist nowhere else, ever.

**Shared identity with Condense:** yes, one family. Same ring glyph, same Geist maker wordmark, differentiated only by product name (and, if you want, a slightly different rim hue: Condense could rim cooler/whiter, HIDE rims gold). The family story from C0 (compression and radiation) is the connective tissue. Lead with it whenever the two products appear together.

**Tagline:** carry two registers.
- Primary, the thesis in three words: **Open the box.** It plays directly against "black box" and states the whole product.
- Long form, the dual face: **Nothing leaves your machine. Nothing's hidden from you.**
- Bench/honest alternative if you want the local-lavish angle foregrounded: **Local. Legible. Lavish.** (on-device, transparent, free to spend compute without limit).

## C3. Color and theme

**Dark-first, dark primary, dark is the soul.** Not a preference, a function: an IDE you watch all night, organized around staring into the agent's work, is an observatory and observatories are dark. Light mode is a later accommodation, never the marketing hero, possibly a restrained "paper" mode in v2. Build dark, and build it properly.

**Material, not flat.** The base is not VS Code's flat #1e1e1e grey. It is your #060606 near-black with material depth: subtle two-layer gradients on surfaces, inset shadows for recessed channels, hairline rims, real elevation. Anodized, not painted. This is your CX capsule language applied to the whole shell.

**On the cliche:** "near-black with a single bright accent" is a known AI-default look. We are not it, and here is why: the dark base is mandated by the doctrine (not a lazy free choice), the accent is *derived from the subject* (the gold photon ring and Hawking radiation, not a generic acid-green or vermilion), and the surfaces are *material* (rim-lit, recessed, anodized) rather than flat. The serif-plus-mono type ([C4](#c4-typography)) seals it. If any screen starts to read as "flat dark with a neon dot," it has failed the doctrine.

**Tokens (near-black material ramp):**

| token | value | use |
|---|---|---|
| `--void` | `#060606` | base background (your established base) |
| `--surface-0` | `#0B0B0C` | raised panel |
| `--surface-1` | `#111113` | card |
| `--surface-2` | `#18181B` | elevated / hover |
| `--rim` | `rgba(255,255,255,0.06)` | hairline border, the recessed-channel edge |
| `--rim-strong` | `rgba(255,255,255,0.10)` | emphasized edge |
| `--text-hi` | `#F2F0EC` | primary text, warm off-white, AA on `--void` |
| `--text-mid` | `#A8A6A1` | secondary text |
| `--text-low` | `#7C7A75` | metadata (keep at AA, see constraints) |

**Signature (the radiation, the brand life-color):**

| token | value | use |
|---|---|---|
| `--radiation` | `#F0B95B` | agent active / streaming, the breathing glow |
| `--radiation-bright` | `#FFD888` | needs your approval, peak glow, lit controls |
| `--radiation-bloom` | `rgba(240,185,91,0.32)` | the rim-light bloom / glow shadow |

**Semantic palette:**

| meaning | token | value | notes |
|---|---|---|---|
| agent active / streaming | `--radiation` | `#F0B95B` | soft, breathing, never a spinner |
| needs your approval | `--radiation-bright` | `#FFD888` | steady, rim-lit control, plus icon and label |
| success / tests passed | `--success` | `#6FBF8B` | muted jade, never neon, never terminal-green |
| error / danger | `--danger` | `#E5635E` | refined signal red, a calm nod to 032c red |
| warning | `--warning` | `#E08A3C` | orange, deliberately separated from the gold |
| diff added | `--diff-add` | fg `#8FD0A6`, bg `rgba(111,191,139,0.10)` | plus a `+` marker, not color alone |
| diff removed | `--diff-del` | fg `#E58B86`, bg `rgba(229,99,94,0.10)` | plus a `-` marker, not color alone |

**Off-limits:**
- No blue and no purple, anywhere, at all. This is a hard rule and it is a feature: the absence of blue separates us from VS Code (blue) and the absence of purple separates us from Linear and Cursor and every AI wrapper. Our accent system is neutrals plus gold plus green/red/orange. Nothing cool.
- No true black `#000`. Use `#060606`. True black kills the material depth and is harsh under long sessions.
- No glassmorphism / frosted translucency as a primary device.
- No neon / acid saturation (the cheap-hacker tell).
- Color is never the only signal (see accessibility in [C14](#c14-hard-constraints)).

**Mood:** muted desaturated pro across roughly 95 percent of the surface (Aesop restraint), with the gold glow as the one place allowed to be luminous and alive (bounded maximalism: a calm near-monochrome field, one radiating accent). That contrast *is* the design: the dark observatory, and the glowing thing you are observing.

## C4. Typography

The typographic signature is a serif display paired with a mono everything-else. No IDE does this. That alone is half your distinctiveness.

**Display: Cormorant Garamond.** Big, confident, editorial (032c energy). An elegant high-contrast serif inside a coding IDE is a strong, deliberate statement: this is a considered instrument, not a generic tool. Used strictly for large display: surface titles, the overnight digest's hero number, empty-state hero lines, section heads. Strictly display sizes (28 to 72px); Cormorant gets spindly at body sizes, so it never drops below display.

**Everything functional: Geist Mono.** Not just code. Labels, buttons, status, metadata, the file tree, the Context Stack, the agent narration, the diffs. All of it in mono. This makes HIDE feel like a precision instrument where every control is labeled (the OP-1 / field-gear feel), and it is radically distinctive: no IDE runs mono-everything UI. Mono also *helps* legibility in dense lists (alignment, scanning), which is why terminals use it, which serves the Context Stack directly. Set it at comfortable sizes and line-heights for UI (11 to 14px, line-height 1.5 to 1.6), this is a legibility requirement, not an aesthetic afterthought.

**Logo: Geist.** Wordmark only. Never in UI text. (Standing rule.)

**Type scale:** high contrast on purpose. Huge confident Cormorant display against small precise Geist Mono chrome. The drama is the gap between elegant-large-serif and small-technical-mono. Not flat-utilitarian, the editorial confidence is part of the brand.

One allowance: if a genuinely prose-heavy agent explanation in Chat starts to feel tiring in mono, Cormorant (at a reading size, not display) is the only fallback. Default stays mono, which reads correctly as transcript/log and fits the terse voice.

## C5. Density and layout

**Differentiated density, not uniform.** The editor and the Context Stack are dense (pro IDE, information-rich, the density of a Bloomberg terminal with the craft of Linear) because that is where you work and observe. The Chat and the overview/"between" spaces breathe (Things 3 calm, generous spacing) because that is where you think and converse. You correctly intuited this split; we formalize it.

**What keeps it from feeling like two different apps:** a strict grid and one consistent spacing scale (4px base, 8px rhythm) everywhere, so even the dense surfaces read as composed, never cramped. Density through information, never through abandoning spacing discipline. The instrument model: a control panel is dense, but every control has alignment and breathing room. Dense but never cramped is the rule.

**Layout stance: opinionated and curated, not infinitely dockable.** VS Code's everything-is-draggable is precisely the identity-less, patchwork-enabling choice we are rejecting, and it is the thing that would let harvested modules stay a patchwork. Instead: a small number of designed, named, canonical layouts per surface (the Things 3 / Linear stance of fixed-but-perfect). Splits resize (drag to widen the Context Stack or the editor), but panels do not freely rearrange or dock anywhere. The opinionation *is* the design system that unifies the borrowed parts. This is the single most important structural decision for not shipping a patchwork: every harvested component gets re-housed into a fixed, designed slot, never dropped in as-is.

## C6. The three surfaces and how they relate

**One unified shell, three modes, with the Context Stack as the constant spine across all three.** You never "leave" one space for another. You change what is on the main stage while the Context Stack (the agent's live state) persists at the edge. That persistent rail is the thread that makes the three surfaces feel like one product: wherever you are, you are watching the agent.

**Center of gravity: observation-first.** Not editor-first (that is Cursor, the clone trap) and not purely chat-first. The product is organized around *watching the agent*, and the three surfaces are three lenses on that single activity:
- **AI IDE** = watch and touch the code the agent changes.
- **AI Chat** = watch and steer the agent's reasoning.
- **AI Workstation** = watch and manage many agents at once.

This is the honest expression of your actual differentiator (transparency plus lavish local parallelism), and "observation-first" is genuinely novel: nobody else is organized this way.

**The front door is the Workstation.** When you open HIDE you land on the overview: your agents, your runs, what happened overnight, project state at a glance. You dive *into* the IDE or Chat for focused work, the way Linear opens to your issues and you dive into one. Putting the most novel, most only-local surface at the front door is the right strategic framing. The IDE must still be a genuinely excellent editor (not an afterthought), it is just not the landing.

**Moving between surfaces:** a persistent mode switcher (three named modes as a left icon-rail or segmented control, the glanceable spatial anchor) plus a command palette (Cmd+K for everything, the keyboard-first power-user spine, Linear/Things style). Not a sea of tabs (VS Code tab-hell is out).

## C7. The Context Stack

The signature feature.

**Prominence: an always-visible side rail by default, expandable into a full inspector on demand.** This is how radical transparency stays empowering instead of overwhelming: the default is a calm, glanceable summary (legible at a glance, there is the adjective again), and you drill into depth only when curious. Progressive disclosure is the entire trick. Always present, never shouting.

**The metaphor stays literal: a stack.** A vertical column of strata, each a layer of what the agent is holding right now, top to bottom:
- retrieved files and symbols
- tools called
- memory in play
- tests and state
- current action (this stratum is a live feed, the now, streaming the agent's actual moves)

Scan the column to take in the whole state; expand any stratum for depth. Stable structure (the strata) with one live-feed stratum (the present). We deliberately did *not* choose a graph or a swarm or a galaxy visualization for this: a graph is the overwhelming choice you were worried about, and it would betray the "calm" rule. The cosmos lives in the brand and the glow, not in forcing the context into a busy diagram. Keep the literal metaphor literal.

**"Let me touch it" is the differentiator, and the design must make touch obvious.** Every stratum is editable: pin or unpin a file from context, evict a memory, mute a tool, inject a note. Every stratum carries its affordances visibly (a pin, an x, a drag handle). This is where the CX physical-control language pays off hardest: the strata should feel like tactile hardware modules you toggle, like an OP-1's labeled controls or a patch bay. Touch should feel material. That tactility is what turns "transparency" into "control," which is the whole promise.

Mission-control in spirit (you are monitoring), expressed as a clean stack, never as a busy dashboard.

## C8. Agent presence and aliveness

**Motion philosophy: restrained, but alive. The light is the heartbeat.** No anthropomorphic mascot, no bouncy character animation. Aliveness is carried by the gold radiation: when the agent thinks, a soft breathing pulse on the relevant element; when it streams, text arrives with a subtle gold leading edge (the radiation leaking out). Aliveness equals the box radiating. This ties the agent's life directly to the brand color and the black-hole concept, which is exactly the kind of coherence we want.

**Token streaming:** text streams with a refined leading cursor, a subtle gold block or glow at the streaming edge. No jank, no flicker, calm cadence. The edge is where the radiation is.

**Progress without anxiety, the most important rule in this section:** show the actual work, not abstract progress theater. Avoid spinners and percentage bars, they manufacture false precision and create the cockpit anxiety we are designing against. Convey "working" through (1) the breathing gold glow on the active stratum, (2) the live-feed stratum scrolling the agent's real actions (Reading auth.ts, Running 12 tests), and (3) slow ambient motion. Seeing the real action is calmer and more trustworthy than a mystery spinner. The anti-anxiety move *is* the product thesis: transparency reduces anxiety, so aliveness and transparency reinforce each other instead of fighting. Restrained motion, one living color, real-work-as-progress.

## C9. Parallel and overnight agents

**"Many agents working" is a board of cards, not a swarm.** A chaotic swarm visualization looks cool in a demo and reads as pure noise in use (the cockpit/overwhelming trap, again). Instead, a calm board/grid where each agent is a card showing: what it is working on, one line of its live feed (current action), and its status by glow (breathing gold = active, green = done, amber = waiting on you). A fleet you take in at a glance, like a Linear board or a Things project list where each row is a live agent. Calm, scannable, legible.

**Walking back after agents ran all night: a morning digest, not 50 notifications.** Fifty notifications is anxiety. One composed overnight view is the observatory at dawn: you sit down and the night's observations are composed into a legible report. What ran, what needs your review (ranked first, in gold/amber), what is ready to merge, what failed. Open it with a single ambient editorial number in big Cormorant Garamond ("7 agents ran. 312 files changed. 4 need you."), the 032c big-number-as-statement moment, which is also a genuine delight and entirely on-brand. Reading the morning's findings, calm and complete.

**Merge review of parallel outputs:** a calm queue, each agent's diff reviewed with the *same* hunk-by-hunk accept/reject as the IDE. Consistency of the diff-review interaction across every surface is essential given we are assembling harvested modules, the review gesture must be identical everywhere or the seams show.

## C10. Interaction model

**Keyboard-first, command-palette-centric, mouse-rich where things are physical.** Cmd+K is the spine (jump anywhere, do anything, Linear-style), comprehensive shortcuts throughout, this fits the developer / LocalLLaMA audience. But not vim-modal-mandatory: vim is an *option* in the editor for those who want it, never forced, because forcing it narrows the audience. Mouse stays rich where the interaction is spatial or tactile: diff review, the Context Stack touch-interactions, the agent board. Linear's balance is the model.

**Steering and approving, three verbs: see, steer, gate.** This maps exactly to "show everything and let me touch it."
- **See:** the Context Stack (you always know what the agent is doing).
- **Steer:** a persistent steering input, always available, to redirect the agent mid-flight ("actually, use X"). The agent is interruptible and steerable, never fire-and-forget.
- **Gate:** an approval gate for consequential or irreversible actions (a destructive command, etc.). Clear and calm, unmistakable but not a trap. The CX glowing-button language is perfect here: the approval is a lit control waiting to be pressed, `--radiation-bright`, with an icon and a plain-language statement of what will happen.

**Diff review:** inline in the editor by default (changes shown in place, green/red blocks, accept or reject per hunk via keyboard: move with j/k, accept/reject with a/r, a calm flow), with side-by-side available on toggle for large or complex diffs. Line-by-line granularity available. Inline keeps you in context and stays calmer; side-by-side is there for careful comparison. Identical interaction in the Workstation merge review.

## C11. Motion and micro-interactions

**Polished but restrained: weighted calm.** Every motion is purposeful and refined (Linear-grade easing), never gratuitous, but there *are* signature moments. Bounded maximalism: a disciplined field with a few perfect motions. Things have mass; nothing snaps or bounces cheaply (cheap snap-bounce is the vibecoded tell).

Signature motions:
- **The radiation pulse.** The gold breathing glow on active elements, the agent's heartbeat, the through-line motion of the whole product.
- **How a diff lands.** Accept a hunk and the red/green settles cleanly into normal code with a brief, satisfying absorption (the change is taken in). Reject and it dissolves back out. The acceptance should feel like a clean physical action, a key resolving on the OP-1, your material language.
- **How a finished run resolves.** Breathing gold (active) transitions to a steady state (green done, or amber needs-you) with a brief calm settling: the radiation quiets. No fireworks (noise/anxiety), quiet completion. Calm closure. The overnight version of this is the morning digest.
- **How panels appear.** The Context Stack expanding to inspector slides and grows with weight and ease, never pops. Material, with mass.

Respect `prefers-reduced-motion` everywhere: the breathing glow and every transition need a static fallback. Given how central motion is, this is a real requirement, not a checkbox.

## C12. Voice and copy

**Telemetry, not chatter. Terse, technical, dry, precise.** The voice of a flight log or mission control. Not playful or cute (no "Oops!", no emoji, that is the cheap startup voice and it would undercut the instrument), not warmly chatty. It states what is happening, plainly, in the interface's own voice.

- **Empty states:** calm and matter-of-fact, with one elegant Cormorant hero line allowed as the editorial touch. An empty screen is an invitation to act, not a mood.
- **Errors:** direct, specific, blame-free, actionable. They never apologize and are never vague. "Couldn't reach the local engine. It may not be running." Not "Something went wrong."
- **Labels:** name things by what the user controls and recognizes, never by how the engine is built. A control says exactly what it does, and keeps the same word through the whole flow (the button that says Approve produces a state that says Approved).

**Agent personality:** competence and transparency, expressed by being specific and undramatic, like a good senior engineer thinking out loud in short phrases. Present-tense, specific narration: "Reading auth.ts", "Running 12 tests", "3 failed, fixing." Never "Sure! Let me take a look at that for you!" The agent earns trust by being precise and legible, not by being friendly. A flash of dry wit is allowed in non-critical copy (empty states); never cutesy, never in errors.

## C13. Distinctiveness: the one thing

If forced to one through-line, it is the look from [C0](#c0-the-spine-design): **near-black material surfaces wearing a single luminous warm-gold rim-light, paired with confident Cormorant Garamond display type.** The rim-light is the device that repeats across every surface (the active agent, the approval control, the streaming edge, the mark, the live stratum). It is the black box that radiates, and it is unmistakably not VS Code or Cursor (flat, blue, sans, no material, no serif). When a thin gold glow sits on the edge of a dark recessed panel, that is HIDE.

The functional twin of that visual signature is the **Context Stack as a persistent spine** (observation-first, the agent always legible at the edge). The visual and the structural say the same thing: the box radiates, and you are always watching it.

If you want a one-sentence positioning to hang it all on: *most AI tools are a chat box on a black box; HIDE is the box, opened, lit at the rim, and yours alone.*

## C14. Hard constraints

**Standing rules (must-haves, non-negotiable):**
- No em dashes or en dashes anywhere in UI copy or content. Commas, colons, parentheses, restructured sentences.
- No middot as a separator. Use alternatives.
- Geist is logo only. Never in UI text.
- Geist Mono is the monospace and UI font.
- Dark brutalist aesthetic governs. Material brutalism, not flat vibecoded gradients or glassmorphism.
- Temperature/cooking references (if any ever appear in copy or examples): both Fahrenheit and Celsius.

**Absolute nevers (design):**
- Not a VS Code clone. No VS Code chrome, no VS Code blue, no infinitely-dockable patchwork. Every harvested module gets re-housed into a designed fixed slot.
- No purple or blue anywhere. The accent system is neutrals plus gold plus green/red/orange.
- No true black `#000`. Use `#060606`.
- No glassmorphism / frosted glass as a primary device.
- No neon / acid / fake-hacker terminal green.
- No cutesy or emoji voice.
- No gratuitous bounce or snap motion.

**Accessibility (treat as hard requirements):**
- The muted aesthetic must not cost text contrast. Body and label text meet WCAG AA against `--void` (`#060606`). Verify `--text-low` specifically, it is the one at risk. The gold accent must be legible where it carries text.
- Color is never the sole signal. Needs-approval, success, error, and especially diff-added/removed must also carry an icon, a shape, or a `+`/`-` marker, for red-green color blindness.
- Keyboard-navigable everything (it is keyboard-first anyway). Visible focus states.
- Respect `prefers-reduced-motion`: static fallbacks for the breathing glow and all transitions.
- Mono-everything must be set at comfortable size and line-height; do not let the instrument look cramp legibility.

**References to match in spirit:** Linear (craft, speed, keyboard, command palette, motion polish), Things 3 (calm, restraint, opinionated curated layouts), Teenage Engineering OP-1 (labeled tactile instrument, mono controls, material), 032c (editorial display type, the big-number moment), Aesop (muted premium restraint, warmth), the EHT M87\* image (the gold photon ring, the mark and the glow). Your own CX capsules (the material rim-light language) and the TailorAI base (`#060606`, Cormorant plus Geist Mono) for direct continuity.

**References to avoid literally:** VS Code, Cursor, generic vibecoded AI-wrapper aesthetics, neon-hacker terminals.

## C15. Self-check: tells that we are failing the doctrine

Run this against any screen before it ships. If any are true, it has drifted.

- It could be mistaken for VS Code or Cursor.
- There is blue or purple on screen.
- The dark surfaces are flat (no rim, no recess, no material depth), so it reads as "flat dark with a neon dot."
- A spinner or a percentage bar is standing in for showing the agent's real work.
- The Context Stack is a busy graph or a swarm instead of a legible stack.
- The transparency feels like a cockpit (alarms, noise, anxiety) instead of an observatory (calm watching).
- The voice apologized, used an emoji, or got cute in an error.
- A harvested module was dropped in as-is instead of re-housed into a designed slot, and you can see the seam.
- Geist appears somewhere other than the logo.
- An em dash, en dash, or middot made it into copy.
- Motion bounced or snapped cheaply.
- Body or label text fails AA contrast on `#060606`.

If none are true: the box is radiating, at the rim, in gold, and it is legible. That is HIDE.

---
---

## Part D: Product & Contract Reference

> HIDE (Hawking IDE) is a **local-first agentic coding IDE**. The model runs on your own Apple Silicon GPU via `hawking serve`: no API calls, no telemetry, no subscription, zero marginal cost per decode. The local runtime *is* the product, not a fallback. The front end is the developer-facing surface wrapped around a runtime (`hawking serve`, supervised) and an agent layer (`hide-kernel`'s real Planner-Executor-Verifier loop) that already exist.

**Exceed, not rival.** HIDE does **not** try to match cloud IDEs on raw frontier-model capability (that axis is structurally closed to a local product). It **exceeds** them on the axes a cloud provider *cannot* reach because they bill per token and assemble the context window server-side in a black box. The FE thesis, in one line: a cloud agent is a chat box bolted to a black box (you type, it works where you can't see, it hands back a result). Because HIDE's runtime is ours and local, the FE does the one thing they can't: **show everything and let the user edit it live.** Concretely, the FE surfaces and makes *editable*: the exact tokens the model sees (the `ContextManifest`, rendered as the Context Stack right-rail), the KV/context budget and what got dropped to fit it (drag-to-pin it back), every retrieved file/symbol/tool-call/memory-injection as it happens, the plan before and during execution (approve, edit, reorder), the run as a scrubbable replayable timeline backed by the durable event log, and the model's own logit-derived confidence on demand. This "observability + live steering" is the differentiator.

## D1. The three surfaces (consolidated)

The product front end is **three surfaces over one backend**, plus the Context Stack rail that threads through all of them:

| Surface | What it is | Primary job |
|---|---|---|
| **AI IDE** | Editor (Monaco) + per-hunk Diff Review + File Explorer + integrated Terminal (xterm/PTY) | Edit code; review and accept/reject agent edits; run commands. The workbench body. |
| **AI Chat** | Streaming conversation with the agent: plan cards, tool-call chips, inline diff chips | Talk to the agent; approve/steer/interrupt a run; selection-to-chat. |
| **AI Workstation** | Parallel-agents dashboard: many sessions/runs at once with state pills, progress, pause/stop/open | Fan out N agents, watch them, triage overnight runs. "Spend lavishly, locally." |
| **Context Stack** *(rail, not a surface)* | Live `ContextManifest` render in the right rail | The observability moat: show the model's mind, live, on every surface. |

The shell is **not a hard-coded screen**: it ships a layout engine, an event router, and a store fabric. Panels-as-extensions is the long-term design; the v1 skeleton wires the three surfaces plus rail directly.

### D1.1 The workbench shell (six regions)

The frame all three surfaces live in: a **six-region workbench** on a splittable pane model, defaulting to a calm three-column layout that opens into full observability on demand.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ TITLE BAR   hide ▸ project ▸ branch   ·  [runtime ● Ready 41 tps]  · ⌘K palette│
├──┬────────────────────┬──────────────────────────────────┬────────────────────┤
│A │ PRIMARY SIDEBAR    │  EDITOR GROUP(S) (tabbed, split)  │  CONTEXT STACK      │
│C │ (swappable viewlet)│  ┌────────────────────────────┐  │  (RIGHT RAIL — the  │
│T │  ▸ Explorer        │  │ auth.rs ✎ │ pool.rs │ +Diff │  │   differentiator)   │
│I │  ▸ Search          │  ├────────────────────────────┤  │  ▸ Model            │
│V │  ▸ Source Control  │  │   Monaco editor / diff     │  │  ▸ Budget ▓▓▓▓▓░    │
│I │  ▸ Agent Runs      │  │   (ghost-text, inline-edit,│  │  ▸ Retrieved (6) ▸  │
│T │  ▸ Memory*         │  │    hunk gutters)           │  │  ▸ Symbols / Tools  │
│Y │  ▸ Chat (or → R)   │  │                            │  │  ▸ Dropped (12) ▸   │
│  │                    │  └────────────────────────────┘  │  ──────────────     │
│BAR                    │                                  │  CHAT (default dock)│
├──┴────────────────────┴──────────────────────────────────┴────────────────────┤
│ BOTTOM PANEL:  Terminal │ Problems │ Test Output │ Agent Timeline │ Output      │
├──────────────────────────────────────────────────────────────────────────────┤
│ STATUS BAR:  ⎇ branch · ⚠2 ●0 · Ln42 Col8 · Rust · [agent: planning…⏸] · 41tps │
└──────────────────────────────────────────────────────────────────────────────┘
```

| # | Region | Default | Toggle |
|---|---|---|---|
| 1 | **Activity Bar** | left edge | viewlet switchers + badges (run count, problems, notifications) |
| 2 | **Primary Sidebar** | left ~260px | `Cmd+B`; one active viewlet; remembers width per viewlet |
| 3 | **Editor Group(s)** | center, fills | `Cmd+\` split; drag-tab to split; grid of tab groups |
| 4 | **Context Stack** (right rail) | right ~320px, **on by default** | `Cmd+Alt+B`; Chat docks beneath it |
| 5 | **Bottom Panel** | bottom ~30% | `Cmd+J`; tabbed; Agent Timeline lives here |
| 6 | **Status Bar** | bottom edge | always; agent-state pill is a global steer affordance |

The three *surfaces* are **arrangements of these regions**, not separate windows: **AI IDE** = Explorer + Editor + Bottom Panel; **AI Chat** = Chat dock foregrounded; **AI Workstation** = Agent Runs viewlet + Agent Timeline + fleetview. They share one shell, one store fabric, one event stream. The **Context Stack** is present in all three.

**Layout store.** `layoutStore` owns the pane model (`WorkspaceLayout`). It is **view-state, not log-state**: it persists to `<workspace>/.hide/` (next to the host's state), NOT through the event log. **Decision FE-4: layout never round-trips through Wire-A/B.** Named presets (`Focus`, `Code`, `Agent`, `Review`) are commands. Tear-off uses the wrapper's multi-window support (a deferred packaging concern); a new window opens its own `WS /v1/hide/events` connection and re-subscribes. In a plain browser this degrades to a single window or a second tab.

**Command palette + keyboard model.** Keyboard-first: every action is a command with a palette entry and a re-bindable key. `commandStore` holds the registry (read-only, not event-fed). The palette (`Cmd+Shift+P`) fuzzy-searches commands; Quick Open (`Cmd+P`) takes mode prefixes in one box: `>` commands, `@` file symbols, `#` workspace symbols (via `code_index` connector), `:` go-to-line, **`§` agent actions** (HIDE-specific: "scrub to event…", "fork session here…", "re-run step…", "switch profile…"). The `§` namespace is where Timeline/Context-Stack actions become keyboard-reachable.

**Panel inventory.** v1 shell (build first): Editor, Chat, Agent-Run Timeline, Diff Review, Context Stack, Terminal, File Explorer, Search, Command Palette, Status Bar, Problems, Notifications. Later (designed, not v1): Memory viewlet/editor, Test Explorer tree, Model Lab (gated on HF distribution / 32B `.tq`), Research tab, multiplayer presence, voice composer, energy dashboard.

**Theming.** One token set (CSS variables, the Part C tokens) drives the shell, Monaco (its theme API, synced), and xterm (its theme object, synced) so they never drift. Ships dark (default) + light + high-contrast. Accessibility first-class: keyboard-complete, Monaco accessible mode, ARIA live regions that announce on `token_batch`/turn boundaries (not per token), `prefers-reduced-motion` honored, diff/plan/confidence use shape+label not color alone.

**Seed keymap:**

| Action | Key (`Cmd`=⌘) | `when` |
|---|---|---|
| Command Palette / Quick Open / Agent actions | `Cmd+Shift+P` / `Cmd+P` / `Cmd+Shift+§` | always |
| Toggle Sidebar / Context Stack / Bottom Panel | `Cmd+B` / `Cmd+Alt+B` / `Cmd+J` | always |
| Focus Chat / Selection→Chat | `Cmd+Shift+L` / `Cmd+L` | always / editor |
| Inline Edit | `Cmd+K` | editor selection |
| Accept ghost-text / word | `Tab` / `Cmd+Right` | suggestion visible |
| Accept hunk / Reject hunk | `Cmd+Enter` / `Cmd+Backspace` | diff focus |
| Approve plan | `Cmd+Enter` | plan focus |
| Queue turn / Override turn | `Cmd+Enter` / `Shift+Cmd+Enter` | composer && running |
| Interrupt agent | `Esc Esc` | running |
| Why? (provenance) | `Cmd+Alt+/` | editor |
| Split editor / Save / Close tab | `Cmd+\` / `Cmd+S` / `Cmd+W` | editor |
| Timeline scrub back/fwd/live/fork | `Cmd+[` / `Cmd+]` / `Cmd+End` / `Cmd+Shift+F` | timeline |

### D1.2 AI IDE (editor / diff / files / terminal)

Monaco-centered editing with agent affordances, plus Diff Review, Explorer, Terminal, Search, Status Bar.

**Stores owned:** `editorStore` (open Monaco models, decorations, `ghostText`, `inlineEditWidget`, `confidenceHeat`; fed by `projection_patch:{diff, file_external}` and `token_batch`), `diffStore` (`diffs{ diff_id -> { path, hunks[], status } }`; fed by `projection_patch:diff`), `sourceControlStore` (review aggregate / multibuffer, checkpoint list, git status), `fileTreeStore` (tree, `touchedByRun`, git badges; fed by `projection_patch:{file_external,diff}` + `tool_progress`), `terminalStore` (xterm instances, `agentSessionId`), plus `searchStore`, `diagnosticsStore`, `statusStore`.

**Diff Review.** The agent's edits arrive as `projection_patch:diff` (a `diff_id` with hunks). Per-hunk **Accept** -> `AcceptDiff{run_id,diff_id}` (the host applies; a follow-up `projection_patch:diff` flips status to `applied`). **Reject** -> `RejectDiff`. **Undo** -> `Custom:revert_diff` (a compensating event upstream). Multibuffer review aggregates every `diff_id` for a `run_id` into one scroll. The modified side is editable before accept; the edit becomes a user-authored revision. The `Stale` state handles the file changing under a pending diff (`projection_patch:file_external`): never apply onto drifted content. Checkpoints are the events themselves (the host's CAS post-image); reverts are compensating events, and because shell tool calls are also recorded, a revert can honestly surface "these terminal commands also ran after this point."

**Inline-edit / ghost-text.** Ghost-text is pre-commit/local (cheap completions stay off the log): `Idle -> GhostPending -> GhostShown`, accept on `Tab` (insert, no event), dismiss on `Esc`. The `Cmd+K` path is **agentic**: `EditPrompt -> EditGenerating -> EditOverlay`, it produces a `projection_patch:diff` that is reviewable/undoable; abort flips the host's `abort` flag via `CancelRun`.

**Terminal.** Human input over a dedicated `hide-serve` PTY WebSocket (not Wire-A); `RunCommand{argv,cwd}` for agent/explicit commands (host routes to `shell.run`, recorded as a `tool.call`/`tool.result` pair and mirrored back as `tool_progress`). **Explorer/Search:** `OpenFile`; Search queries the `code_index` connector directly.

### D1.3 AI Chat (conversation / plan / steer)

Default docked beneath the Context Stack; can dock to the sidebar.

**Stores:** `chatStore` (messages, streaming buffers per `stream_id`, composer, queued turns, plan cards), `runStore` (the active-run FSM). **Streaming:** assistant tokens arrive as `token_batch{stream_id,text}`; the FE keys the streaming buffer on `stream_id` and appends, and a render-rate governor flushes one React commit per animation frame (`requestAnimationFrame`) so a 120 tok/s stream never thrashes the UI (the host already coalesced upstream; this is purely a paint optimization).

**Turn model:** a user message -> `SubmitTurn{session_id, text, attachments}` -> `IntentAck{event_seq}`; the assistant turn streams back as `token_batch` + `projection_patch:{turn,plan,tool}`. `@`-mentions add a pinned context source (a `Custom:pin_span` alongside the turn); `/`-slash-commands dispatch their command's intent. **Steer:** `PauseRun`/`ResumeRun`/`CancelRun` are real enum variants; redirect/edit-plan/queue are `Custom` (FE-1). `SecurityGate` renders inline as an approval prompt.

**Plan-steer FSM:** `Planning -> PlanReady -> [autonomy=suggest-only] AwaitingApproval` (Approve -> Executing; Edit/Reorder -> PlanReady; Reject -> Idle) or `[autonomy=auto] -> Executing` directly. Within Executing: step active/done renders from `projection_patch:plan`; Pause (`PauseRun`) -> Paused -> Resume (`ResumeRun`); Redirect (`Custom:redirect_run`) -> Replanning; Stop (`CancelRun`) -> Stopped; `SecurityGate` -> AwaitingApproval. **Decision FE-6: the run FSM is the single source of the agent-state pill** that the status bar, notifications, and fleetview all read.

### D1.4 AI Workstation (parallel agents)

The Agent-Run Timeline, fleetview (multi-run grid), parallel-agent orchestration, and merge-review.

**Agent-Run Timeline.** Turns the host's append-only event log into a visual, scrubbable filmstrip. Step-card kinds map to `UiEvent`s by `(kind, projection)`: **Turn/Plan** <- `projection_patch:{turn,plan}`; **Thinking** <- `token_batch`; **Tool** <- `tool_progress`; **Diff** <- `projection_patch:diff`; **Test/Build** <- `projection_patch:{test,build}`; **Context** <- `projection_patch:context_manifest`; **Runtime** <- `runtime_status`; **Error** <- `error`. The Timeline is the **universal consumer**: it subscribes to every `kind`. Cards thread by `cause` (the host's `Event.cause`) so a card shows "<- caused by Plan step 3 <- Turn 1" (the provenance spine behind "Why?", `Cmd+Alt+/`).

- **Scrub** -> `ScrubToEvent{session_id, event_id}` -> host `scrub_to_event(seq)` returns a read-only `SessionProjection` folded to that seq; the FE rebuilds editor buffers, plan tree, diffs, and the Context Stack as they were at that seq. A "replay mode" banner makes it read-only; effects never re-fire (the host fold is pure).
- **Resume from here** -> `ResumeRun` -> live execution re-attaches and appends new events from `seq` onward.
- **Fork from here** -> `ForkSession{session_id, at_event}` -> host `fork_session(at_seq)` returns a `(SessionId, SessionProjection)` child; original intact. **Edit-then-fork** (scrub -> edit plan/pin via `Custom` -> fork) re-executes deterministically from the edited state: the signature demo.

Scrub beyond the in-memory window requests a log range via the **pull** API `ui_events(session, after_seq, limit)`; live tail auto-scrolls to head, detaches into review on scroll-back with a "Jump to live".

**Fleetview (multi-run grid).** The home for many parallel agents. Launched via `Custom:fleet_run{objective, pattern?}` -> host `fleet_run()` schedules a parallel kernel run under the FleetGovernor (worktree-isolated, admission-gated). Each cell is a run with a state pill, active plan step k/n, elapsed, last event, resource band, and quick pause/stop/open. The orchestration *patterns* (single, fan-out/map-reduce, pipeline, tournament/best-of-N, planner-workers-merger, speculative) are chosen by the planner or the user; the FE renders the chosen pattern and per-run status, it does **not** decide the pattern (backend `hide-fleet`). `fleetStore` is fed by `projection_patch:{fleet, run, merge}`. Per-run control reuses `PauseRun`/`CancelRun`. Fleetview surfaces the per-run resource band (RAM/thermal headroom) the Governor reports, so the user sees why a run is queued vs admitted.

**Merge-review.** When parallel runs finish, results funnel through an **integration branch** (fan-out: combine all disjoint-footprint results; tournament: oracle-first select one winner). Conflicts that reach the human are shown in the **Diff Review** UI (the same Monaco-diff component, fed two candidate diffs). The user branch only ever receives a fully-integrated, full-suite-green result. `merge.*` arrive as `projection_patch:merge`; resolution choices go back as `Custom:resolve_conflict{by}`. The promote bar is the ONLY effect-commit to the user branch, enabled only when the suite is green.

**Remote (later).** The headline thin-client story (laptop drives a Mac-Studio agent server) reuses the **same** intent-in/events-out model over the wire (carrying `UiEvent` verbatim; server-authoritative, reconnect resumes from `seq`). Because the local transport is *already* HTTP/WS to `hide-serve` on `127.0.0.1`, going remote is just pointing the same client at a non-loopback `hide-serve` address: only `VITE_HIDE_BASE` (and auth/transport hardening) change. The host's `remote_protocol` capability is currently `false`; FE work here waits on it. **Decision FE-7: surfaces bind to the HTTP/WS client interface (`sendIntent`/`onUiEvent`/`callConnector`), never to a wrapper or a hard-coded host, so the remote swap is a one-file change.**

### D1.5 The Context Stack right-rail (the differentiator)

Renders the live context manifest verbatim, every turn: the exact answer to "what is the model looking at, why, and what did it leave out." No cloud agent can show this; we own the window assembly (the `context` connector). The manifest arrives as `projection_patch:context_manifest` (the host's `context` connector compiles it; `hawking-context` produces the budgeted manifest).

| Section | Renders | Interaction -> Intent |
|---|---|---|
| **Model** | model id/arch/ctx, profile, greedy-vs-sampled, seed; live `runtime_status` | click -> `Custom:switch_profile` |
| **Budget bar** | `{total, used, free, reservations}` stacked bar, colored by source | hover segment -> tokens + source |
| **Retrieved files** | code spans + `code_index` hits (path:range, relevance) | click -> `OpenFile`; drag -> `Custom:pin_span` |
| **Symbols** | symbol spans + provenance | click -> go-to-definition (`code_index`) |
| **Tools called** | tool spans + `tool_progress` (name, ok/fail, bytes) | click -> expand output, link to Timeline step |
| **Memory injected** | memory spans (fact, confidence, provenance) | click -> Memory editor (later) |
| **KV / tiers** | prefix-reuse tokens, bank hit, tiers touched | read-only |
| **Confidence** (opt-in) | per-token logprob from `token_batch` confidence path | toggle -> `Custom:toggle_confidence` (heat in editor/chat) |
| **Dropped** | dropped candidates `{title, would_be_tokens, reason}` | **one-click pin** -> `Custom:pin_span` (forces into next turn) |
| **Conflicts** | surfaced contradictions | inline resolve -> `Custom:resolve_conflict` |

```
┌─ CONTEXT STACK ─────────────────────────────┐
│ MODEL  qwen ▸ profile: Standard ▾  16k/32k  │
│ BUDGET ▓▓▓▓▓▓▓▓▓▓▓▓░  14,210 / 16,384        │
│        [sys 1.2k][code 6.1k][tools 3.4k]…    │
│ ▾ RETRIEVED (6)  • auth.rs:42-88 .91 ⊙pin    │
│ ▾ SYMBOLS (12)   ▾ TOOLS (3) ✓read ✓grep ✗   │
│ ▾ MEMORY (2)     • "DB uses sqlx" 1.0 📌      │
│ ▾ KV  bank HIT · reuse 1,200 tok · gpu·ram   │
│ ▾ DROPPED (12) ▸ why?  • cargo log 4.2k ⊕pin │
│ ⚠ CONFLICT: pinned arch fact vs new scan ▸   │
└──────────────────────────────────────────────┘
```

`contextStore` holds `currentManifest`, a `manifestRing` (recent manifests indexed by turn/seq, for scrub coupling), and `lastAppliedSeq`. Steering writes (pin/unpin/resolve/switch-profile) emit `Custom` intents -> the host appends events -> the `context` connector's compiler honors them next turn. **Decision FE-8: `contextStore.manifestRing` is keyed by the same seq the Timeline scrubs to**, so when the Timeline scrubs to seq N the rail renders `manifestRing[N]` ("what did it see when it made *that* decision"). The manifest must be published per turn, not just the latest.

### D1.6 The Research tab (secondary, later)

A first-class tab beside the IDE/Chat/Workstation surfaces, but **post-shell**: the research engine is usable headless via the `research` connector (`runs.list`/`runs.append`) before this UI exists. Panels: Library, Graph (interactive KG), Research Runs, Reports (cited, every sentence a provenance chip), Lit Maps, Experiments, Notes/Canvas, Review queue. UX laws: provenance always one click away; measured vs inferred vs speculative visually distinct; contradictions shown, never hidden. Store: `researchStore`, fed by the `research` connector + `projection_patch:research`. Not v1; do not let it gate the three core surfaces.

## D2. The backend contract

This is the load-bearing section. The backend is built; the FE binds to **these exact types and methods**, not to any earlier design sketch. Source of truth: `crates/hide-core/src/api.rs` (the wire types) and `crates/hide-backend/src/{host,commands,ui_bus,connectors}.rs` (the host surface).

### D2.1 The two wires + the `hide-serve` HTTP/WS adapter

Two directions, carried over **localhost HTTP + WebSocket** by `hide-serve`: a thin axum server wrapping `BackendHost`, mirroring `crates/hawking-serve`. `hide-serve` constructs `BackendHost::open_workspace`, binds `127.0.0.1`, and exposes the endpoints below; it (de)serializes JSON and otherwise does nothing, so the contract types and behavior are unchanged.

```
  ┌─────────────────────────── RUST HOST (hide-backend::BackendHost) ─────────────────────────┐
  │                                                                                            │
  │   hawking serve (HTTP/SSE) ──┐                                                              │
  │   OS / tools / files ────────┼─▶ event log (single writer, seq) ─▶ projections ─▶ UiEvent  │
  │                              │                                                  │          │
  │   CommandRouter::handle(Intent) ─▶ validate ─▶ append user.intent.* ─▶ IntentAck│          │
  │           ▲                                                            UiEventBus│ (publish │
  │           │ (Wire-A)                                            subscribe()──────┘  + coalesce)
  └───────────┼──────────────────────────────────────────────────────────────┬─────────────────┘
   hide-serve │  POST /v1/hide/intent                          WS /v1/hide/events │  (axum, 127.0.0.1)
  ┌───────────┼──────────────────────────────────────────────────────────────▼─────────────────┐
  │  WEB APP (React + TS + Vite — browser-renderable, no Tauri dependency)                        │
  │   user action ─▶ sendIntent(intent) ─▶ fetch POST /v1/hide/intent ─▶ IntentAck               │
  │   ws.onmessage(UiEvent) ─▶ EventRouter ─▶ route by kind ─▶ Zustand stores ─▶ render           │
  │   callConnector(id,method,params) ─▶ fetch POST /v1/hide/connector ─▶ Value                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Wire-A (FE -> host): `POST /v1/hide/intent`** (body = `Intent` JSON, response = `IntentAck` JSON). The `hide-serve` handler deserializes the `Intent`, calls `CommandRouter::handle(intent)` (which `BackendHost::handle_intent` delegates to), and serializes the `IntentAck`. **The host validates and can reject:** an empty `SubmitTurn`, empty-argv `RunCommand`, or blank-name `Custom` returns `IntentAck { accepted: false, message: Some(reason) }` (HTTP 200 with `accepted:false`) and logs nothing. The FE must surface a rejected ack, not assume success.
- **Wire-B (host -> FE): `WebSocket /v1/hide/events`** is a stream of `UiEvent` JSON frames. The FE opens one ordered WebSocket per client; the handler forwards everything from `BackendHost::subscribe_ui()` (a `broadcast::Receiver<UiEvent>` off the `UiEventBus`) onto the socket. The bus does **render-coalescing** (consecutive `TokenBatch`es for one stream merge before publish) and has **bounded backpressure** (a slow subscriber gets a `Lagged` drop-oldest signal, never stalls the host). For reconnect, **`GET /v1/hide/events?after_seq=N`** is the **pull** catch-up (backed by `BackendHost::ui_events`): fetch the gap, then resume the live socket. The FE adds its own rAF render-governor on top.

### D2.2 Wire-A: the `Intent` enum (every variant)

`Intent` (`api.rs`, `#[serde(tag="type", content="data", rename_all="snake_case")]`). Each `handle` returns `IntentAck { accepted: bool, event_seq: Option<u64>, message: Option<String> }`.

| Intent variant | Payload fields | UI action that sends it | Host behavior |
|---|---|---|---|
| `SubmitTurn` | `session_id`, `text`, `attachments: Vec<BlobRef>` | Chat composer submit; selection-to-chat | **Rejected if `text` is blank.** Logs `user.intent.submit_turn`; kicks the agent turn. |
| `CancelRun` | `run_id` | Steer bar Stop; Workstation per-run stop | Signals `Interrupt::Abort` on the `InterruptHub` for that run, then logs. |
| `PauseRun` | `run_id` | Steer bar Pause; status-bar agent pill | Signals `Interrupt::Pause`; logs. |
| `ResumeRun` | `run_id` | Resume a paused run | Clears the buffered pause; logs. |
| `AcceptDiff` | `run_id`, `diff_id` | Diff Review: accept hunk/file | **Rejected if `diff_id` blank.** Logs `accept_diff`. |
| `RejectDiff` | `run_id`, `diff_id` | Diff Review: reject hunk/file | **Rejected if `diff_id` blank.** Logs `reject_diff`. |
| `ScrubToEvent` | `session_id`, `event_id: EventId` | Timeline scrub slider | Logs `scrub_to_event`; pairs with `BackendHost::scrub_to_event(seq)` to rebuild the read-only past projection. |
| `ForkSession` | `session_id`, `at_event: EventId` | Timeline "fork session here…" | Logs `fork_session`; pairs with `BackendHost::fork_session(at_seq)`. |
| `OpenFile` | `path`, `line: Option<u32>` | Explorer click; provenance peek; go-to-def | **Rejected if `path` blank.** Logs `open_file`. |
| `RunCommand` | `argv: Vec<String>`, `cwd: Option<String>` | Terminal command; palette "run…" | **Rejected if `argv` empty.** Logs `run_command`; pairs with `BackendHost::run_command`. |
| `Custom` | `name`, `payload: Value` | Extension/HIDE-specific actions (profile switch, pin span, re-run step…) | **Rejected if `name` blank.** Logs `custom.<name>`. **This is the escape hatch** for FE actions without a dedicated variant. |

> **Note on time-travel naming.** The built API uses `event_id`/`at_event` (typed `EventId`) on `ScrubToEvent`/`ForkSession`. The host methods `scrub_to_event(seq)` / `fork_session(at_seq)` operate on the numeric `seq`. The FE carries the `EventId` in the intent; the host resolves it. Don't invent an "at_seq" intent field, it doesn't exist.

### D2.3 Wire-B: `UiEvent` and the `UiEventKind` variants (every kind)

`UiEvent { seq: u64, session_id: Option<SessionId>, kind: UiEventKind }`. The FE routes by `kind` (and filters by `session_id` per surface). `seq` is the cursor each store tracks as `last_applied_seq` for replay-on-reconnect.

| `UiEventKind` | Payload | What the FE renders/does |
|---|---|---|
| `ProjectionPatch` | `projection: String`, `patch: Value` | A state-diff for a named panel/projection. Route by `projection` name to the owning store, apply the patch. The general-purpose state-sync path (plan tree, diff state, context manifest, etc.). |
| `TokenBatch` | `stream_id: String`, `text: String` | Coalesced streamed tokens for a stream/session. Append to the chat/run buffer keyed by `stream_id`; the FE rAF-governor commits once per frame. |
| `RuntimeStatus` | `status: String`, `detail: Option<String>` | Serve up/down/degraded. Drives the status-bar runtime pill + a banner on `down`/`degraded`. `status` mirrors the supervisor states: `down`/`booting`/`ready`/`degraded`/`failed`. |
| `ToolProgress` | `call_id: String`, `message: String` | Live tool-call progress chip (in chat + timeline). The host publishes one per dispatched tool result. |
| `SecurityGate` | `gate: String`, `message: String` | An approval is needed (sandbox/permission gate). FE shows an approval prompt; the user's decision goes back as an intent (`Custom` or an Accept/Reject). |
| `Error` | `code: String`, `message: String` | Route to the notification + status stores; non-fatal inline, fatal as a banner. |
| `Custom(Value)` | free `Value` | Extension-defined events; route by an agreed discriminator inside the value. |

### D2.4 The `BackendHost` method surface (what the `hide-serve` endpoints wrap)

`hide-serve` exposes these (already real on `host.rs`) as HTTP/WS endpoints, or uses them internally:

| Host method | Signature (abbrev.) | FE use |
|---|---|---|
| `open_workspace(root)` | `-> Result<Self>` | App boot: `hide-serve` opens the project at startup. |
| `subscribe_ui()` | `-> broadcast::Receiver<UiEvent>` | Forwarded onto `WebSocket /v1/hide/events` (Wire-B). |
| `handle_intent(Intent)` | `-> Result<IntentAck>` | Backs `POST /v1/hide/intent` (Wire-A). |
| `call_connector(id, method, params)` | `(&str,&str,Value) -> Result<Value>` | Backs `POST /v1/hide/connector` -> `callConnector`. |
| `fleet_run(session, objective)` | `-> Result<String>` | Workstation: schedule a parallel kernel run; returns terminal status. |
| `generate_and_publish(session, base_url, prompt)` | `-> Result<String>` | Drives generation through the runtime client; publishes `TokenBatch`es onto Wire-B. |
| `scrub_to_event(session, seq)` | `-> Result<SessionProjection>` | Timeline scrub (read-only past view). |
| `fork_session(from, at_seq)` | `-> Result<(SessionId, SessionProjection)>` | Timeline fork. |
| `run_agent_to_terminal(session, objective, max_steps)` | `-> Result<AgentState>` | Drive a run to a terminal phase (Chat/IDE turn). |
| `run_command(session, argv, cwd)` | `-> Result<ToolResult>` | Terminal/command execution (shell.run tool). |
| `status()` | `-> BackendStatus` | Boot/settings: workspace root, capabilities, connector statuses, tool specs, model roles. |
| `health()` | `-> HealthReport` | Health panel: per-component Ok/Degraded/Failed checks. |
| `ui_events(session, after_seq, limit)` | `-> Result<Vec<UiEvent>>` | **Pull** catch-up/replay (the durable-log path), exposed as `GET /v1/hide/events?after_seq=N`, used on reconnect to fill the gap before the live `WebSocket /v1/hide/events` resumes. |

### D2.5 The connectors (`call_connector(id, method, params)`)

Connectors are the typed RPC surface for non-intent data the FE needs (search, context, roles…). Registered in `connectors.rs`; all reachable via `BackendHost::call_connector`. Methods take/return `serde_json::Value`.

| Connector `id` | Methods | What it powers in the FE |
|---|---|---|
| `runtime` | `roles.list`, `route` | Model role list (Context Stack "Model" panel, settings); routing decision preview (greedy/sampled, grammar). `route` is read-only (a routing-decision preview, not a mutating setter). |
| `code_index` | `search`, `definition`, `references`, `file.add_text`, `file.index`, `health` | Search surface (IDE); go-to-def / find-refs; provenance/index health. |
| `context` | `compile` (-> `{ prompt, manifest }`) | The Context Stack: compile a prompt + `ContextManifest` for a task; the manifest is the rail's data source. Params: `task`, `max_input_tokens`, `search_limit`, optional `role`. |
| `personalization` | `records.list`, `records.append`, `records.by_task` | Logging accepted/rejected diffs (the flywheel corpus); personalization views. |
| `research` | `runs.list`, `runs.latest`, `runs.append`, `runs.by_state` | Research Lab surfaces (post-shell, but the connector is live). |

### D2.6 The supervisor / runtime-status surface

`hawking serve` is booted and supervised by the `RuntimeSupervisor` inside the host. Its state machine is `Down -> Booting -> Ready -> Degraded -> Failed` (with restart/backoff). The FE never talks to the supervisor directly: it **observes** it via `RuntimeStatus` UiEvents (Wire-B) and the `status()`/`health()` snapshots. The FE responsibilities: a **status-bar runtime pill** bound to the latest `RuntimeStatus.status` (click-through to detail); **degraded/down banners** (on `degraded`/`failed`/`down`, show a non-modal banner; the host auto-restarts, so the banner clears on the next `ready`); and **gate the composer** (while `status != ready`, `SubmitTurn` may be rejected upstream, so reflect "runtime not ready" in the UI rather than spinning).

### D2.7 The HTTP/WS client surface

The FE's single seam to the host: a concrete TS client over `fetch` + `WebSocket`. Everything else (stores, router) sits on top of this. Types are TS mirrors of the Rust `serde` wire shapes in `api.rs`.

```ts
// wire.ts — TS mirrors of crates/hide-core/src/api.rs
export type SessionId = string; export type RunId = string; export type EventId = string;
export type BlobRef = { /* mirror hide-core types::BlobRef */ };

export type Intent =
  | { type: "submit_turn";   data: { session_id: SessionId; text: string; attachments: BlobRef[] } }
  | { type: "cancel_run";    data: { run_id: RunId } }
  | { type: "pause_run";     data: { run_id: RunId } }
  | { type: "resume_run";    data: { run_id: RunId } }
  | { type: "accept_diff";   data: { run_id: RunId; diff_id: string } }
  | { type: "reject_diff";   data: { run_id: RunId; diff_id: string } }
  | { type: "scrub_to_event";data: { session_id: SessionId; event_id: EventId } }
  | { type: "fork_session";  data: { session_id: SessionId; at_event: EventId } }
  | { type: "open_file";     data: { path: string; line: number | null } }
  | { type: "run_command";   data: { argv: string[]; cwd: string | null } }
  | { type: "custom";        data: { name: string; payload: unknown } };

export type IntentAck = { accepted: boolean; event_seq: number | null; message: string | null };

export type UiEventKind =
  | { type: "projection_patch"; data: { projection: string; patch: unknown } }
  | { type: "token_batch";      data: { stream_id: string; text: string } }
  | { type: "runtime_status";   data: { status: string; detail: string | null } }
  | { type: "tool_progress";    data: { call_id: string; message: string } }
  | { type: "security_gate";    data: { gate: string; message: string } }
  | { type: "error";            data: { code: string; message: string } }
  | { type: "custom";           data: unknown };

export type UiEvent = { seq: number; session_id: SessionId | null; kind: UiEventKind };
```

```ts
// ipc.ts — the ONLY module that touches the HTTP/WS transport (fetch + WebSocket).
const BASE = import.meta.env.VITE_HIDE_BASE ?? "http://127.0.0.1:8744"; // hide-serve
const WS_BASE = BASE.replace(/^http/, "ws");

/** Wire-A: POST the intent, get the host's ack (which may be a rejection). */
export async function sendIntent(intent: Intent): Promise<IntentAck> {
  const r = await fetch(`${BASE}/v1/hide/intent`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(intent),
  });
  return (await r.json()) as IntentAck;   // accepted:false is a 200 body, not an HTTP error
}

/** Wire-B: subscribe to the ordered UiEvent stream over a WebSocket. Returns an unsubscribe fn.
 *  On (re)connect, the caller first pulls ui_events(after_seq) (catchUpUiEvents) to fill any gap,
 *  then resumes the live socket. */
export function onUiEvent(handler: (ev: UiEvent) => void): () => void {
  const ws = new WebSocket(`${WS_BASE}/v1/hide/events`); // ordered; host-side coalesced + backpressured
  ws.onmessage = (e) => handler(JSON.parse(e.data) as UiEvent);
  return () => ws.close();
}

/** Pull catch-up/replay for reconnect: GET the durable UiEvents after a seq cursor. */
export async function catchUpUiEvents(afterSeq: number): Promise<UiEvent[]> {
  const r = await fetch(`${BASE}/v1/hide/events?after_seq=${afterSeq}`);
  return (await r.json()) as UiEvent[];
}

/** Typed RPC to a backend connector (runtime/code_index/context/personalization/research). */
export async function callConnector<T = unknown>(
  id: "runtime" | "code_index" | "context" | "personalization" | "research",
  method: string,
  params: unknown,
): Promise<T> {
  const r = await fetch(`${BASE}/v1/hide/connector`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id, method, params }),
  });
  return (await r.json()) as T;
}
```

> The endpoint paths (`POST /v1/hide/intent`, `WS|GET /v1/hide/events`, `POST /v1/hide/connector`) are the FE-host contract `hide-serve` must expose; they're the only strings hard-coded outside `ipc.ts`. Everything above the client (router, stores, components) imports only `sendIntent`/`onUiEvent`/`catchUpUiEvents`/`callConnector`, so the transport (and any desktop wrapper) is swappable without touching UI code.

### D2.8 The Custom-name registry (canonical)

`Intent::Custom{name, payload}` is the escape hatch for every steer/observe action the built `Intent` enum has no dedicated variant for, and `ProjectionPatch{projection}` is the named state-diff for every panel slice. Because both are string-keyed, **the host and FE must agree on the exact string for each logical action / slice.** This is the canonical registry. **Don't add a `Custom` name or a `projection` discriminator anywhere without adding it to this table.**

**`Intent::Custom{name}` values:**

| `name` | Sent by (surface) | Action |
|---|---|---|
| `save_file` | AI IDE (Editor) | persist an editor buffer |
| `inline_edit` | AI IDE (Editor) | Cmd+K agentic inline edit |
| `mention_in_chat` | AI IDE (Explorer) | add a file as a chat context source |
| `pty_input` / `pty_resize` | AI IDE (Terminal) | terminal stdin / resize over the PTY mirror |
| `run_search` | AI IDE (Search) | issue a search query |
| `quick_fix` | AI IDE (Problems) | apply a diagnostic quick-fix |
| `revert_diff` | AI IDE (Diff Review) | undo an applied diff (compensating event) |
| `edit_hunk` | AI IDE (Diff Review) | edit the modified side before accept |
| `queue_turn` | AI Chat (Composer) | append a turn to the prompt queue |
| `redirect_run` | AI Chat (Composer/steer) | redirect a running turn |
| `approve_plan` | AI Chat (PlanCard) | approve a proposed plan |
| `edit_plan_step` | AI Chat (PlanCard) | edit a plan step |
| `reorder_plan` | AI Chat (PlanCard) | reorder plan steps |
| `rerun_step` | AI Workstation (Timeline) | re-run a timeline step |
| `fleet_run` | AI Workstation (Fleetview) | schedule a parallel kernel run (-> `fleet_run()`) |
| `resolve_conflict` | AI Workstation (Merge-review) | choose a merge-conflict resolution |
| `pin_span` / `unpin_span` | Context Stack | pin / unpin a context span into the next turn |
| `switch_profile` | Context Stack | change the model profile |
| `toggle_confidence` | Context Stack | toggle per-token confidence heat |
| `resolve_conflict` | Context Stack | resolve a context contradiction |
| `approve_gate` | Notifications / any panel | approve a `SecurityGate` (when not an Accept/Reject) |
| `focus_run` / `dismiss` | Notifications | focus / dismiss a notification |

**`ProjectionPatch{projection}` discriminators** (the panel-slice names the FE routes on after `kind`): `turn`, `plan`, `tool`, `diff_chip` (chat); `diff`, `file_external`, `editor` (IDE); `context_manifest`, `retrieval`, `memory` (Context Stack); `timeline` (Agent-Run Timeline, universal); `build`, `test`, `diagnostics` (Problems); `sourcecontrol` (checkpoints); `fleet`, `run`, `merge` (Workstation); `turn_ended`, `plan_waiting` (Notifications); `status` (Status Bar). The set is owned jointly by host + FE.

### D2.9 How the three surfaces map onto the contract

Each surface is a composition of intents it sends, UiEvent kinds it consumes, and connectors it calls.

| | **AI IDE** | **AI Chat** | **AI Workstation** |
|---|---|---|---|
| **Sends (Intent)** | `OpenFile`, `RunCommand`, `AcceptDiff`/`RejectDiff`, `Custom`(inline-edit, ghost-text) | `SubmitTurn`, `PauseRun`/`ResumeRun`/`CancelRun`, `ScrubToEvent`/`ForkSession`, `Custom`(redirect, edit-plan) | `SubmitTurn` (fan-out objectives), `PauseRun`/`ResumeRun`/`CancelRun` per run |
| **Consumes (UiEventKind)** | `ProjectionPatch`(diff/editor/files), `ToolProgress`, `RuntimeStatus`, `Error` | `TokenBatch`, `ProjectionPatch`(plan/chat), `ToolProgress`, `SecurityGate`, `Error` | `ProjectionPatch`(run state across sessions), `RuntimeStatus`, `Error` |
| **Calls (connector)** | `code_index`(`search`/`definition`/`references`), `runtime` | `context`(`compile`), `runtime`(`route`), `personalization`(log accept/reject) | `runtime`(`roles.list`), (fleet via `BackendHost::fleet_run`) |
| **Host methods** | `run_command`, `scrub_to_event` | `generate_and_publish`, `run_agent_to_terminal`, `fork_session` | `fleet_run`, `status` |

**Context Stack (rail, all surfaces):** consumes `ProjectionPatch{projection:"context*"}` for the live `ContextManifest`; calls `context.compile` to (re)build it; `Custom` intents for pin/drop/profile-switch; reads `runtime.roles.list` for the Model panel. On a Timeline scrub (`ScrubToEvent`), the rail rewinds to that event's manifest.

**Cross-cutting (every surface):** `RuntimeStatus` -> status pill + banner; `Error` -> notifications; every `UiEvent.seq` advances the owning store's `last_applied_seq` so reconnect replays cleanly (open the `/v1/hide/events` WebSocket, request `GET /v1/hide/events?after_seq=N` catch-up, resume live).

### D2.10 Packaging / desktop wrapper (deferred)

The UI is a **pure web app** talking to `hide-serve` over localhost HTTP/WS. Whether it's eventually shipped inside a native desktop shell is a late, reversible packaging choice that does not touch UI code: the wrapper just hosts the same web client and (optionally) launches `hide-serve` as a child process. **Electron** is the safe default (what VS Code, Cursor, and Void ship, so the harvested UIs assume it). **Tauri** is an option, not the architecture (a small macOS-only binary). **Plain browser / PWA** is fine for dev (the default during skeleton development). Because the contract is transport-agnostic, choosing or changing the wrapper is a build/packaging decision, deferred until the surfaces work end-to-end against `hide-serve`.

### D2.11 Decisions every doc must stay consistent with

1. **Wire types are fixed by `api.rs`:** `Intent`/`IntentAck`/`UiEvent`/`UiEventKind` exactly as enumerated (snake_case `type`/`data` tagging; time-travel uses `event_id`/`at_event: EventId`, not `at_seq`). Don't introduce new variants; use `Custom{name,payload}` for FE-specific actions. **(FE-1)**
2. **Three `hide-serve` endpoints only:** `POST /v1/hide/intent` (Wire-A), `WS /v1/hide/events` (Wire-B; `GET /v1/hide/events?after_seq=N` is its pull twin for reconnect), `POST /v1/hide/connector` (connector RPC: `{id, method, params}`). The HTTP/WS client (`ipc.ts`) is the sole module that touches `fetch`/`WebSocket`. **(FE-3 / FE-7)**
3. **The host owns truth; stores are derived caches** keyed by `last_applied_seq`. Reconnect = open the WebSocket + `GET /v1/hide/events?after_seq=N` pull catch-up + live resume.
4. **Two streaming layers:** host-side `UiEventBus` coalescing (per `stream_id`) and an FE rAF render-governor. Don't render per token. **(FE-4: layout is view-state on disk, never through the log.)**
5. **Connectors are the non-intent RPC surface** (`runtime`/`code_index`/`context`/`personalization`/`research`); the Context Stack's data comes from `context.compile` -> `{prompt, manifest}`. **(FE-2: route on `kind` first, then for `projection_patch` on `data.projection`.)**
6. **Runtime is observed, not controlled:** `RuntimeStatus` states are `down/booting/ready/degraded/failed`; gate the composer on `ready`. **(FE-5: `SecurityGate` is a blocking prompt, never auto-dismissed. FE-6: the run FSM is the single source of the agent-state pill. FE-8: `manifestRing` is keyed by the scrubbed seq.)**

## D3. The OSS harvest map

The front-end build team's map of which open-source AI-IDE UI/interaction components to harvest, what to merely study, and the license rules that gate any port into shipped `app/` code. Where it binds to the backend, it binds to the **real, shipped contract** (`crates/hide-core/src/api.rs`; `crates/hide-backend` `BackendHost`), not the old design sketch.

### D3.1 The license rule (read first, it gates everything below)

This is a hard CI-enforced rule. Treat it as load-bearing, not advisory.

| Bucket | Licenses | What you may do |
|---|---|---|
| **PORT-OK** | **MIT, Apache-2.0** | Adapt/incorporate source into shipped `app/` (the React/TS/Vite web app) and Rust host code (`hide-serve`). Copyright header in every ported file + a `THIRD_PARTY_NOTICES.md` entry. |
| **STUDY-ONLY** | **AGPL-3.0** (Zed) | Read published docs/blogs/behavior only. **Never** copy, port, link, paste, or "reimplement from a peek at" the source: AGPL would force HIDE's proprietary FE open. No exceptions regardless of snippet size. |
| **NEVER-TOUCH** | Proprietary (Cursor, Copilot/Copilot Workspace, Claude Code) | No source exists / is closed. Study observable UX only. No lifting. |

Mechanics the FE build must honor: `THIRD_PARTY_NOTICES.md` at repo root is the canonical attribution file, generated at build time from a structured `harvest.toml` manifest (component -> license -> version -> list of FE source files that incorporate it); Apache-2.0 ports additionally require a "modifications stated" line per file. A **CI license gate** fails the build if any FE source file under a harvest path lacks the required origin-header comment, or if `harvest.toml` references a license outside PORT-OK, or if a file's structure resembles Zed/GPUI. Any PR adding harvested UI must confirm it does not draw on AGPL (Zed) or proprietary (Cursor/Copilot) source. If in doubt: **port only MIT/Apache-2.0; everything else is inspiration you re-implement clean.**

### D3.2 Consolidated harvest table

One row per source. "FE component to harvest" is strictly front-end (UI/interaction); backend ports (diff-apply algorithm, repo-map, MCP client, event schema) are **already built** and live in the backend crates per `SCAFFOLD_STATUS.md`, listed here only where the *UI shape* travels with them, and tagged **(backend-done)**. "Target HIDE FE module" names the `app/` module the FE team creates.

| Source | License | FE component to harvest | Mode | Target HIDE FE module / surface |
|---|---|---|---|---|
| **Void** (Void Editor Contributors) | Apache-2.0 | Monaco `DiffEditor` wrapper; hunk-level accept/reject controls; ghost-text rendering for streaming suggestions; collapsible dock/panel layout with stored widths + split-view config (*the closest existing UI to ours*) | **Port** | `app/src/diff/DiffView.tsx`, `app/src/layout/` → **AI IDE** |
| **Cline** (Cline Bot, Inc.) | MIT | Plan↔Act mode toggle UX; per-step approval controls; auto-approve category toggles (read-only / write / terminal / MCP); MCP server list + tool-discovery panel UX. *(Tiered diff/apply matcher itself = backend-done.)* | **Port (UI) / study** | `app/src/chat/PlanActBar.tsx`, `app/src/settings/AutoApprove.tsx`, `app/src/mcp/McpPanel.tsx` → **AI Chat** + **AI IDE** |
| **OpenHands** (All Hands AI) | MIT | Event-stream timeline + replay observability rendering (the visual filmstrip over an append-only event log); action/observation card pairing with `cause` threading. *(Event schema port itself = backend-done in `hide-core/src/event.rs`.)* | **Port (UI) / study** | `app/src/timeline/AgentRunTimeline.tsx` → **AI Workstation** + **AI IDE** |
| **Continue** (Continue Dev, Inc.) | Apache-2.0 | Context-provider UX glue: `@`-mention pickers (`@file`/`@symbol`/`@docs`/`@terminal`) in the composer; per-message "what's in context" affordance. *(Retrieval algorithm = backend-done.)* | **Study → small port** | `app/src/chat/MentionPicker.tsx` → **AI Chat** + **Context Stack** |
| **Aider** (Paul Gauthier) | Apache-2.0 | Repo-map *view* (ranked symbol map rendering) + architect/editor two-mode selector UX. *(Repo-map ranking algorithm = backend-done.)* | **Study** | `app/src/context/RepoMapView.tsx`, mode selector in **AI Chat** |
| **Goose** (Block, Inc.) | Apache-2.0 | Desktop-agent UX shape; MCP-client UX (install/configure an MCP server, browse its tools, see per-tool status). *(`rmcp` client itself = backend-done.)* | **Study** | feeds `app/src/mcp/McpPanel.tsx` → **AI IDE** |
| **Kilo Code** (Kilo Code Contributors) | Apache-2.0 | Checkpoint/undo *shadow-git* UX: per-run snapshot list, one-click "revert to checkpoint," snapshot→event-range index rendering. *(Checkpoint engine = backend-done in `hide-tools`.)* | **Study → Port (UI)** | `app/src/sourcecontrol/CheckpointList.tsx` → **AI IDE** |
| **OpenCode** (OpenCode Contributors) | MIT | Plan/act step-through interaction model + session list / resume / export UX (numbered steps, step-level confirm). | **Study** | `app/src/sessions/SessionBrowser.tsx` → **AI Chat** + **AI Workstation** |
| **Zed** (Zed Industries) | **AGPL-3.0** | Multibuffer diff-review bar (aggregate many-file edits into one scroll, per-hunk Keep/Reject, editable unified diff); "agent following" cursor; context-window-usage display. | **STUDY ONLY** (never copy) | inspiration for `app/src/diff/MultibufferReview.tsx` (clean reimpl) → **AI IDE** |
| **Cursor** (Anysphere) | Proprietary | Three-mode model (Tab / Cmd+K inline-edit / Composer-agent); "functional minimalism, editor-grade not chat-grade" visual language; instant-apply concept. | **NEVER lift** (study UX) | inspiration only, all surfaces |
| **GitHub Copilot / Workspace** (Microsoft) | Proprietary | Issue/plan/PR flow; plan-then-implement legible-checkpoint UX. | **NEVER lift** (study UX) | inspiration for **AI Workstation** batch flow |

### D3.3 Per-source binding notes

Three FE conventions every harvested component must follow (they reconcile the harvested UX with the built backend):

1. **Panels render from `ProjectionPatch`, not typed events.** Each harvested panel owns one named projection (`"diff"`, `"plan"`, `"timeline"`, `"context"`, `"editor"`, `"filetree"`, `"sourcecontrol"`, `"status"`). The IPC client applies the JSON `patch` into the matching store slice; the component is a pure render of that slice. (Streamed tokens are the exception: `TokenBatch{stream_id,text}` coalesced into the chat/timeline directly.)
2. **Steering = `Intent::Custom`.** Only the eleven intents above are first-class. Pin/unpin, edit/reorder/approve plan step, redirect mid-run, switch profile, revert-to-checkpoint all ride `Intent::Custom{name,payload}`. A harvested UI that assumed a typed intent must be rewired to `Custom`. Every intent returns `IntentAck{accepted, event_seq?, message?}`: render the rejection `message`.
3. **Connectors for non-turn data.** Search, definition/references, context-manifest compile, runtime `roles.list`/`route` (the latter a read-only routing preview), personalization, research all go through `BackendHost::call_connector(id, method, params)`, not a turn.

Source-specific notes:
- **Void** (Apache-2.0, PORT-OK): port the `DiffEditor` wrapper + hunk controls (side-by-side by default, flip to inline when width drops, Monaco `useInlineViewWhenSpaceIsLimited`) and the dock/panel layout. A proposed edit arrives as `ProjectionPatch{projection:"diff"}`; accept/reject emit `AcceptDiff`/`RejectDiff`; layout is FE-local view state.
- **Cline / Roo Code** (MIT, PORT-OK UI): Plan↔Act toggle, per-step approval + auto-approve categories, MCP panel. Plan cards render from `ProjectionPatch{projection:"plan"}`; approve/edit/reorder emit `Custom`; MCP status via `ToolProgress`; approvals via `SecurityGate`.
- **OpenHands** (MIT, PORT-OK UI): the timeline rendering (lane of step cards in `seq` order, action paired with observation, threaded by `cause`). Subscribes to `ProjectionPatch{projection:"timeline"}` + `TokenBatch` + `ToolProgress` + `RuntimeStatus` + `Error`; scrub issues `ScrubToEvent`, fork issues `ForkSession`, resume uses `ResumeRun`. Effects are never re-fired.
- **Continue** (Apache-2.0, PORT-OK small): the `@`-mention pickers and the "what's in context" affordance. A mention emits `Custom:pin_span`; candidate fetch uses `call_connector("code_index", "search"|"definition")`.
- **Aider** (Apache-2.0, study): repo-map view + architect/editor mode selector. The mode selector is **not** a routing setter (`runtime.route` is a read-only preview): mode switch emits `Custom:switch_profile`; the host honors it next turn.
- **Goose** (Apache-2.0, study): MCP client UX. Same binding as the Cline MCP panel; HIDE gates every MCP tool through the host permission model, so the panel must render the `SecurityGate`.
- **Kilo Code** (Apache-2.0, study -> port UI): checkpoint UX, **terminal-aware** (the revert UI must surface the tool/terminal side-effects that ran after a checkpoint). Renders from `ProjectionPatch{projection:"sourcecontrol"|"timeline"}`; revert issues `Custom:revert_to_checkpoint` or the deep `ScrubToEvent` + `ForkSession` pair.
- **OpenCode** (MIT, study): session browser + plan/act step-through. Session resume issues `ResumeRun`; fork uses `ForkSession`; plan/act shares the Cline plan-card binding.
- **Zed** (AGPL-3.0, STUDY ONLY): the multibuffer review pattern (clean reimpl as `MultibufferReview.tsx`). No source contact, reference only published docs/behavior; a PR resembling GPUI structure must be rejected and rewritten. Binds the same as Void (`AcceptDiff`/`RejectDiff` + `ProjectionPatch{projection:"diff"}`).
- **Cursor & GitHub Copilot/Workspace** (proprietary, NEVER lift): study observable UX only. Cursor's three-mode division and "editor-grade functional-minimalism" inform AI IDE; Copilot Workspace's issue->plan->PR flow informs the AI Workstation batch UX. No code, no asset, no lifted layout.

### D3.4 Build-order note

Harvest priority follows the surface build order. The minimum to stand up the **AI IDE** + **AI Chat** skeleton: **Void** (diff + layout), **Cline** (plan/act + approval), **OpenHands** (timeline). **Continue/Aider** (context glue, repo-map view) and **Goose/Cline** (MCP panel) follow with the **Context Stack**. **Kilo Code** (checkpoints) and **OpenCode** (session browser) land with the **AI Workstation**. **Zed** is studied throughout but never blocks a build (clean reimpl). Every port lands with its `THIRD_PARTY_NOTICES.md` row and header in the same PR, or the CI license gate fails it.

## D4. The front-end build steps and milestones

The actionable, ordered plan to build the UI on top of the backend: the React/TS/Vite web app plus the one piece of new Rust glue (`hide-serve`, a thin axum HTTP/WS server wrapping `BackendHost`, mirroring `crates/hawking-serve`). It does **not** re-plan any backend work; that is done.

**The guiding rule (inherited from the backend constitution):** *the headless path is the truth; the UI is a projection, never the source of correctness.* Every panel renders `UiEvent`s and emits `Intent`s. No panel mutates authoritative state locally. The store is a cache folded from the event stream, and the only way to change anything is to send an `Intent`.

### D4.1 Where we start from (the only thing not yet built)

Everything below the wire is finished: `BackendHost` boots, supervises `hawking serve`, runs the Planner-Executor-Verifier loop, persists an event log, scrubs/forks sessions, and exposes a transport-agnostic command surface. What does **not** exist yet:

1. The **`hide-serve` crate**: a small axum HTTP/WS server that constructs `BackendHost::open_workspace` and exposes the `/v1/hide/*` endpoints (Wire-A `POST /v1/hide/intent`, Wire-B `WS /v1/hide/events` + `GET ?after_seq=N`, `POST /v1/hide/connector`). It mirrors `crates/hawking-serve` (the already-proven axum + SSE server). This is the transport seam recorded as a deliberately deferred item in `SCAFFOLD_STATUS.md`.
2. The **React/TS/Vite web app** (`app/src`): the typed HTTP/WS client, the store, and the panels.

### D4.2 Scaffold the web app + `hide-serve`

**Goal:** a web app that talks to `hide-serve`, which boots `BackendHost`, can receive one `Intent` over HTTP, and can stream one `UiEvent` over a WebSocket. No panels yet.

`hide-serve` (the Rust HTTP/WS server) is a thin axum server owning one `BackendHost`, exposing the intent endpoint, the connector endpoint, and the event WebSocket (+ its pull twin), and forwarding the bus. It binds to `BackendHost::open_workspace(root)` (constructs the whole service graph incl. the `RuntimeSupervisor`), `subscribe_ui()` (the bus tap; the `UiEventBus` already coalesces `publish_token` so the FE receives coalesced `TokenBatch`), `handle_intent(Intent)` (literally `CommandRouter::handle` behind the host, validation/rejection/interrupt-signalling already live there, we add zero logic), and `ui_events(session, after_seq, limit)` (backs `GET /v1/hide/events?after_seq=N`). It binds `127.0.0.1` (loopback only, air-gap-safe). Pin CI to `macos-14`. The PTY is hosted here via `portable-pty` and streamed over its own WebSocket.

`app/src` is a standard Vite React-TS template, `pnpm`, browser-renderable, no desktop-shell dependency. Lock now: the **store fabric** (Zustand-style slices, one per surface, one root store) and the **HTTP/WS client module** (`src/ipc/`, the *only* place `fetch`/`WebSocket` against `hide-serve` is touched).

**Done when:** `pnpm build` exits 0 and `hide-serve` builds on `macos-14`; with `hide-serve` running, the web app loads in a browser; a dev-only button calls `sendIntent({ type:'custom', data:{ name:'ping', payload:{} } })` and logs the returned `IntentAck`.

### D4.3 The HTTP/WS client layer + store wiring

The typed TS client surface is the §D2.7 client (`sendIntent`/`onUiEvent`/`catchUpUiEvents`/`callConnector`). `contract.ts` is the hand-mirrored (or `ts-rs`-generated) TypeScript of `hide-core::api`; **keep this file in lockstep with `crates/hide-core/src/api.rs`**, and a CI check that diffs the generated TS against the Rust is cheap insurance.

One reducer routes on `UiEvent.kind.type`; each slice owns one concern. The mapping is fixed:

| `UiEventKind` | Drives slice | What it does |
|---|---|---|
| `ProjectionPatch{projection, patch}` | the panel named by `projection` | merge `patch` (a state-diff) into that panel's slice: editor buffers, plan tree, diff sets, file tree, and the Context Stack all update this way |
| `TokenBatch{stream_id, text}` | `chatStore` / `timelineStore` | append coalesced text to the open stream for `stream_id` |
| `RuntimeStatus{status, detail}` | `runtimeStore` | serve up/down/degraded -> Status Bar chip color + tps |
| `ToolProgress{call_id, message}` | `timelineStore` | live tool-call row updates |
| `SecurityGate{gate, message}` | `gateStore` | raise an approval modal/toast |
| `Error{code, message}` | `notifyStore` | toast |
| `Custom(Value)` | `notifyStore` (default) | forward to whichever panel registered for it |

Slice list: `chatStore`, `editorStore`, `diffStore`, `fileTreeStore`, `terminalStore`, `contextStore`, `timelineStore`, `runtimeStore`, `gateStore`, `notifyStore`. Every write to an authoritative field comes from a `UiEvent`; every user action goes out as an `Intent`. There is no third path.

Connector access (`POST /v1/hide/connector`): real connectors are `runtime` (`roles.list`, `route`), `code_index` (`search`, `definition`, `references`, `file.add_text`, `file.index`, `health`), `context` (`compile` -> prompt + manifest), `personalization` (`records.list`/`records.append`/`records.by_task`), `research` (`runs.list`/`runs.latest`/`runs.append`/`runs.by_state`). Panels call these; they do **not** reach into crates directly.

**Done when:** `onUiEvent` receives a live `RuntimeStatus` after `BackendHost` boots serve; a `sendIntent({type:'open_file', data:{path,line}})` returns `accepted:true`; `callConnector('code_index','search',{q})` returns hits.

### D4.4 Skeleton panels in priority order

Build order is chosen so the **earliest possible end-to-end demo is chat streaming from serve**, then each subsequent panel exercises one more slice of the contract.

| # | Panel | Harvest from D3 | Intents it sends | UiEvents it consumes |
|---|---|---|---|---|
| 1 | **Chat** | message-list + composer patterns; SSE/stream render | `SubmitTurn`, `CancelRun`, `PauseRun`, `ResumeRun` | `TokenBatch`, `ProjectionPatch(chat)`, `RuntimeStatus` |
| 2 | **Editor** | **Monaco** (`monaco-editor`, MIT) | `OpenFile` | `ProjectionPatch(editor)` |
| 3 | **Diff Review** | Monaco `createDiffEditor` + Cline/Void hunk-UX (Apache-2.0, *reference only*) | `AcceptDiff`, `RejectDiff` | `ProjectionPatch(diff)`, `SecurityGate` |
| 4 | **File Tree** | tree-view component; file reads via `hide-serve` / `code_index` connector | `OpenFile` | `ProjectionPatch(file_tree)` |
| 5 | **Terminal** | **xterm.js** (MIT) over `portable-pty` (MIT) | `RunCommand` | dedicated PTY WebSocket + `ProjectionPatch(terminal)` |
| 6 | **Context Stack** | original; renders `ContextManifest` | `ScrubToEvent`, `Custom{pin/unpin/switch_profile}` | `ProjectionPatch(context)`, `RuntimeStatus` |
| 7 | **Agent Timeline** | OpenHands event-model *idea* (MIT, *reference only*) | `ScrubToEvent`, `ForkSession`, `CancelRun` | `ProjectionPatch(timeline)`, `ToolProgress`, `TokenBatch` |
| 8 | **Workstation** | original; grid of timeline cards | `SubmitTurn`(per lane), `CancelRun` | per-session `ProjectionPatch`, `RuntimeStatus` |

- **Chat (first, the walking-skeleton spine):** user types -> `SubmitTurn` -> the kernel generates against serve -> host publishes coalesced `TokenBatch` -> chat appends. Cancel/pause/resume map to the real variants against the active `run_id` (learned from the run's first `ProjectionPatch`). This single panel proves the entire vertical: command in, bus out, store fold, render.
- **Editor (Monaco):** `OpenFile` on file-tree or Context-Stack click; the host streams buffer contents back as `ProjectionPatch(editor)`. Read-mostly in the skeleton; Monaco is the substrate for Diff Review so it is built before #3.
- **Diff Review:** the host emits `ProjectionPatch(diff)` describing the hunks; render in Monaco `createDiffEditor`; per-hunk Accept -> `AcceptDiff`, Reject -> `RejectDiff`. **All apply/revert logic stays in the backend** (`hide-tools` tiered edit); the panel only sends the verdict and re-renders. A `SecurityGate` may precede the apply.
- **File Tree:** the workspace root is listed by the host (through `hide-serve` / `code_index`, not a direct browser FS API); click -> `OpenFile`; live external-change decorations arrive as `ProjectionPatch(file_tree)`.
- **Terminal:** xterm.js front; `portable-pty` in `hide-serve`; keystrokes -> PTY stdin over a dedicated WebSocket; agent-initiated commands route through `RunCommand` / `run_agent_to_terminal`.
- **Context Stack:** the right-rail, on by default, ~320px, renders the `ContextManifest` verbatim and live; arrives as `ProjectionPatch(context)` each turn; `contextStore` keeps a **ring of recent manifests** so scrubbing re-renders the manifest as it was at that turn; steering goes out as `Custom`; profile data and manifest compilation come from `callConnector('context', ...)`.
- **Agent Timeline:** a vertical run timeline (`Idle -> Planning -> Step 1..N -> Done/Failed/Repair`); fed by `ProjectionPatch(timeline)` + `ToolProgress`; the scrub slider maps to `seq` and issues `ScrubToEvent` (the backend replays the projection and pushes rebuilt state; editor, plan tree, diffs, and the Context Stack all rewind together); a node can `ForkSession` to branch.
- **Workstation:** a grid where each cell is a compact Agent Timeline bound to a different `session_id`; each lane sends its own `SubmitTurn`/`CancelRun`; the store keys all slices by `session_id`. No new contract: N timelines plus a layout. Last skeleton panel because it depends on every prior slice keying cleanly by session.

### D4.5 Front-end milestones (M-FE0..M-FE3)

Each milestone has one binary done-when. These are the FE counterparts to the backend's M0/M1 sequencing, but the backend is done, so they are purely UI + host glue.

- **M-FE0, Walking skeleton.** The web app + the `hide-serve` server, with chat streaming over the WebSocket. **Done when:** in the running web app talking to `hide-serve`, typing a message sends `SubmitTurn` (a `fetch` to `POST /v1/hide/intent`), and coalesced `TokenBatch` events arriving on the `WS /v1/hide/events` socket render incrementally in the chat panel, against a stubbed serve or a live one. The `IntentAck.accepted` and the streamed `stream_id` are observable in devtools.
- **M-FE1, IDE surface.** Add Editor, Diff Review, File Tree, Terminal. **Done when:** the agent proposes an `edit_file`; the Diff Review panel shows the hunks in Monaco; the user accepts one hunk (`AcceptDiff`) and rejects another (`RejectDiff`); the file on disk reflects only the accepted hunk; the terminal runs `cargo build` and streams output live.
- **M-FE2, Context Stack + Timeline.** Add the Context Stack and Agent Timeline, including scrub/replay. **Done when:** during a multi-step run the Context Stack updates live (retrieved files appear as the agent searches; the budget bar fills); the Timeline shows each step; dragging the scrub slider issues `ScrubToEvent` and the Context Stack rewinds to that turn's manifest from the ring.
- **M-FE3, Workstation.** Add the parallel-agent grid. **Done when:** two sessions run concurrently in two lanes; each lane's timeline and stream update independently and correctly (no cross-talk, verified by distinct `session_id` on every event); cancelling one lane's `run_id` leaves the other running.

### D4.6 License / harvest CI gate + `THIRD_PARTY_NOTICES`

Track licenses from the first FE commit, not at ship time. FE-relevant bundled/invoked components and obligations:

| Component | License | Usage | Obligation |
|---|---|---|---|
| Monaco Editor | MIT | editor + diff (bundled) | MIT notice |
| xterm.js | MIT | terminal (bundled) | MIT notice |
| portable-pty | MIT | PTY host (`hide-serve`) | MIT notice |
| axum | MIT | HTTP/WS server (`hide-serve`) | MIT notice |
| Zustand | MIT | store | MIT notice |
| OpenHands event model | MIT | timeline design *reference only* | no code copied -> no obligation |
| Cline / Void diff UX | Apache-2.0 | diff design *reference only* | no code copied -> no obligation |

The gate (CI, every commit): (1) `cargo deny check licenses` over `crates/hide-serve` (allow MIT / Apache-2.0 / MPL-2.0; reject GPL / AGPL / BUSL); (2) a JS license scan over `app/src` `node_modules` with the same allow-list; (3) `THIRD_PARTY_NOTICES.md` regenerated by `tools/gen_notices.sh` (walking both Rust host deps and bundled npm deps); (4) `license-header-check`: any *copied* (not npm-installed) third-party file must carry its original copyright header; components marked *reference only* must contain **no** copied lines. **Done when:** all four steps exit 0 in CI and `THIRD_PARTY_NOTICES.md` lists every bundled MIT component above.

### D4.7 Build-order summary

```
§D4.2 hide-serve server + web-app scaffold ─┐
                                             ├─► M-FE0  chat walking skeleton
§D4.3 HTTP/WS client + store slices ────────┘     (SubmitTurn → TokenBatch over WS → render)
                                                 │
§D4.4 editor/diff/tree/terminal ────────────────► M-FE1  IDE surface
                                                 │
§D4.4 Context Stack + timeline ─────────────────► M-FE2  observability + replay
                                                 │
§D4.4 workstation grid ─────────────────────────► M-FE3  parallel agents
                                                 │
§D4.6 license gate + NOTICES ── runs from commit 1, blocks every release
```

The FE talks to the backend through **exactly three** `hide-serve` endpoints; `ProjectionPatch` is the universal state-update mechanism, `TokenBatch` is the only streaming text path, the seven `UiEventKind`s map 1:1 to the slices; the store is a fold of the event stream and the only outbound mutation channel is `Intent`; panel priority is chat -> editor -> diff -> tree -> terminal -> context -> timeline -> workstation; scrub/replay rewinds all panels via `ScrubToEvent`, and the Context Stack keeps a manifest ring to support it.
