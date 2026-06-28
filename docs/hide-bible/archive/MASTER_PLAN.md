# HIDE Master Plan

### One product, three layers: the box that radiates

> Date: 2026-06-28. This is the unifying plan. It folds the design doctrine, the front-end build, and the
> capability frontier into a single sequenced story, then sharpens that story beyond its inputs. It supersedes the
> three tracks read in isolation: there are not three tracks, there is one product expressed at three layers.
>
> Source inputs: `docs/plans/hawking_capability_frontier_2026_06_28.md` (moats M1-M4, caveats H1-H9, the 6-phase
> roadmap), `docs/hide-bible/frontend/00-vision-and-backend-contract.md` through `04-design-doctrine.md`,
> `docs/hide-bible/SCAFFOLD_STATUS.md` (11 crates done, the agent loop real).

---

## 1. The spine

A black hole is the ultimate black box: nothing escapes, you cannot see in. Hawking proved that is false. Black holes
radiate. That single idea is the whole product, and it is the same idea told three times, at three layers.

At the **story** layer it is the brand: Hawking Condense compresses (drives matter toward singularity density, the
ultimate compressor), HIDE radiates (makes the agent's work escape and become visible). Two faces of one physics. The
dual face resolves the ironic name: HIDE hides you from the cloud (local, nothing leaves your machine) and hides
nothing from you (the Context Stack). Privacy outward, transparency inward.

At the **pixels** layer it is the doctrine: an observatory, not a cockpit. Near-black material surfaces wearing one
luminous warm-gold rim-light, with confident Cormorant Garamond display over Geist Mono chrome. The gold edge is the
box radiating, made into a consistent visual device: it sits on the active agent, the approval gate, the streaming
edge, the mark, the live stratum of the Context Stack. When a thin gold glow sits on a dark recessed panel, that is
HIDE and nothing else.

At the **architecture** layer it is the capability moats: pass state not text (M1), free local fleets (M2), radical
transparency (the observability seam), grammar-guaranteed tool calls (M4). These are structural advantages of owning a
constant-size recurrent state on-device, not features bolted onto a chat box.

The load-bearing insight that makes this a plan and not a wish: **the design already assumes the capability.** The
doctrine's "fork and try N", its "instant resume", the overnight agent board with its morning digest, the calm
no-jank tool calls: each is literally a moat. "Fork and try N" is M1 state clone plus M2 free fleets. "Instant resume"
is M1 state serialize. The calm tool feed is M4 first-try-valid masking. So design and capability are not two things to
integrate later. They are one object seen from two sides, and the only real planning question is sequencing: land each
capability just in time for the UI that expresses it, never earlier (do not gold-plate a primitive with no UI) and
never later (do not ship a surface that lies about a stub).

---

## 2. Where we stand today

The **backend is done** and honest about it. Eleven crates, ~410 tests passing, the agent loop real
(`SCAFFOLD_STATUS.md`): `hide-kernel` is an audited-genuine Planner-Executor-Verifier FSM with deterministic oracles
that shell to real cargo and git; `hide-backend` is a runnable host with `RuntimeSupervisor`, `CommandRouter`,
push `UiEvent` bus, session registry, time-travel scrub and fork; `hide-fleet` has real worktrees, a `FleetGovernor`,
and a 3-way merge funnel; `hawking-index` is a built RAG path (tree-sitter, merkle, FTS5, embeddings, RRF, rerank).
The **design is locked** (`04-design-doctrine.md`): tokens, type, motion, voice, the three surfaces, the Self-check ship
gate. The **front end is specced** but not started: contract types frozen, OSS harvest mapped, build sequencing written.
The **one blocker is native `.tq` serving** (caveat H9): `qwen_dense.rs::load` has no `.tq` branch, so there is no live
model behind the host. Nothing streams a real token until that lands. Every downstream moat, and the GO/KILL verdict on
whether the local model can even code, sits behind that single seam.

---

## 3. The unified roadmap

The interlocked milestones. Each row is one Track-A capability deliverable married to its Track-B front-end deliverable,
the thing the user actually sees ship, the hard dependency, and the honest caveat. The ordering is the spine made
operational: capability just in time under a frozen UI.

| Milestone | Backend / capability (Track A) | Front-end (Track B) | What the USER sees ship | Hard dependency | Honest caveat |
|---|---|---|---|---|---|
| **M0 Today** | 11 crates done; agent loop real; host with router and bus; `.tq` decode test-only | Design locked (04); OSS harvest map done (02) | Nothing yet, internal | none, baseline | Blocker is native `.tq` serving (H9); no live model behind the host |
| **M1 Shell lights up** | `hide-serve` axum HTTP/WS adapter wrapping `BackendHost`; runtime on Qwen `.tq` Stage A (F16 dequant-on-load) | Vite web shell, `ipc.ts` client, EventRouter, Zustand stores, 3 surface frames, Context Stack rail skeleton | A running IDE: editor, terminal, chat composer streaming tokens, live Context Stack | `.tq` Stage A (Phase 0.1) | Stage A is F16 fallback, not the RAM-cliff win; quality unproven (no eval yet) |
| **M2 Thesis gate** | Native bitslice GEMV (`.tq` Stage B); `hawking-eval` coding bench; run gate (>=15 tok/s @32B, >=40% @7B) | Status pill + degraded banners bound to `RuntimeStatus`; runtime-not-ready gating on composer | "32B running on 18GB" plus an honest GO/KILL verdict on coding | Stage A (M1), eval harness | GO/KILL is real; CONDITIONAL means re-quant before features. PPL is not generation (H6) |
| **M3 State primitives** | `RwkvState::{to_bytes,from_bytes,clone}`; `Engine::{save,load}_checkpoint`; fork as engine primitive | Timeline scrub/fork UI (`ScrubToEvent`/`ForkSession`); "fork and try N" affordance | Instant resume (no re-prefill), session undo, fork a branch | thesis GO (M2); RWKV-7 path | State is a recompute saver, not memory quality (H2) |
| **M4 Recall** | Wire `hawking-index` RAG as exact-recall path; optional sliding-window hybrid | Search surface (IDE), provenance peek, retrieval rail panel | Whole project loaded plus reliable retrieval, "it found the right file" | M3; `hawking-index` (built) | Never market "perfect memory"; route exact recall through RAG (H1) |
| **M5 Telepathic agents** | State-passing handoff (memcpy) planner to coder to reviewer via `copy_kv_prefix_to_slot`; text-decode audit tap | Workstation fleet board: state pills, handoff viz, parallel runs; `fleet_run` wired | Multi-agent flows, ~4x faster handoffs, far fewer tokens | M3 (state fork) | Handoffs unauditable without the tap (H7); fan-out wins exploration, not coupled edits |
| **M6 Speed** | Harden `json_constrain` + AST grammars; prefix-cache discipline; EAGLE-3 head; speculative tool exec | No-jank tool-call chips, grammar-valid diff chips, parallel-tool progress | First-try-valid tool calls, snappier loop, no JSON-repair stalls | M4 (index to AST grammar); M2 (logit access) | Grammar validity ~93-96% on hard schemas (keep fallback); spec-decode gated on occupancy (H4) |
| **M7 Personalization** | Tiny-data DPO/SFT on accepted/rejected diffs (MLX); RWKV state skill-seeds; QA-LoRA re-bake | `personalization` views; "model learned your style"; mode switch | Model gets better at your code weekly; per-mode states | M5 (handoff corpus) + `hide-personalize` (built) | LoRA/small-N DPO only, not full RLEF (H5); keep adapter swappable, eval-gated |
| **M8 Format SKUs** | Multi-tier `.tq` (sensitivity map + mixed bpw): 3-bit lossless + 2-bit recovered; "doctor" KD; INT8 state; air-gap | SKU picker, packet-capture air-gap proof surface | Pick your quality/RAM tier; verifiable no-egress privacy | M2 (format proven) | Report effective bpw, honest labels, test real generation (H6) |

---

## 4. The critical path, and what parallelizes now

The longest hard-dependency chain (the spine):

```
native .tq Stage A (M1) -> native .tq Stage B / bitslice GEMV (M2)
  -> hawking-eval coding bench -> THESIS GATE (GO) (M2)
    -> RwkvState save/load/clone + state fork (M3)
      -> telepathic handoff via copy_kv_prefix_to_slot (M5)
        -> personalization flywheel: handoff corpus -> DPO/QA-LoRA re-bake (M7)
```

**Nothing downstream of the thesis gate is worth building until the gate is GO.** A KILL or CONDITIONAL verdict on
coding quality re-routes effort back into Condense/quant, not into features. Building the telepathy UX on a model that
cannot code is the exact "chat box on a black box" pattern the product exists to reject.

**Fully parallel-buildable today, zero dependency on the `.tq` blocker:**

| Stream | Why it is unblocked |
|---|---|
| `hide-serve` axum HTTP/WS adapter | Wraps the already-built `BackendHost`; needs no live model. The single highest-leverage unblocked task. Mirror `hawking-serve`. |
| The entire FE web shell | Vite app, `wire.ts` (TS mirrors of `api.rs`), `ipc.ts`, EventRouter, Zustand stores, 3 empty surface frames, Context Stack rail. Contract types frozen; develop against a stub `hide-serve` or replayed event logs. |
| `hawking-eval` harness | Only needs the model at run time, not build time. Write the coding bench now. |
| OSS module harvest / re-skin | Ripping Monaco diff review, xterm, explorer into designed slots is pure FE. |

**Truly blocked, and on what:** any live decode (the thesis verdict, every M1+ capability feature) is blocked on native
`.tq` Stage A. All state primitives, handoff, and personalization are blocked on the thesis GATE. The RAM-cliff
"32B-on-18GB" headline is blocked on Stage B bitslice GEMV (Stage A F16 fallback does not deliver it).

**Practical consequence:** one stream on Phase 0 (`.tq` Stage A to B + eval), and the whole FE shell plus `hide-serve`
as a fully parallel stream. They converge at M1, where the lit shell first talks to a live runtime.

---

## 5. The design to capability synergy map

The design is not waving at vague future power. It pins to four specific capabilities (M1 fork, M1 handoff, M1
instant-resume, M3/M7 flywheel) plus the universal Phase 0 unblock. Each doctrine element below is an expression of a
moat, with its exact contract seam and its just-in-time debt (the capability the pixels assume but that is not yet
built). Ship the pixels of each only when its moat lands, or the screen lies.

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

Operationally: **Phase 0 (native `.tq`) lights up the gold rim and the cards; M3 (`RwkvState` clone/checkpoint) lights
up "fork and try N" plus instant resume; M5 (state handoff) lights up the telepathic edge; M7 lights up the flywheel.**

---

## 6. The proof demo

**"Five futures, one read."** One ~75-second continuous screen recording that proves story plus pixels plus
architecture in a single take. The viewer comes away thinking: this is the box that radiates, and it just did something
the cloud structurally cannot.

**Setup (t=0):** HIDE open on the Workstation front door, dark observatory, gold ring mark top-left. One session already
warm: the planner has read the whole project once, the Context Stack rail is lit, one stratum breathing gold. Status
pill: `ready`.

1. **t=0-8s. One prompt against a warm state.** User types: "Try five different fixes for the flaky auth retry, keep the
   one that passes." (`SubmitTurn`.) The lit rail is the proof the repo is loaded, not re-read. Narration in dry mono:
   "State warm. Forking."
2. **t=8-15s. The fork (M1 + M2).** One control: **Fork x5** (`ForkSession` x5 on the live state, then `fleet_run` per
   branch). The warm card splits into five, each with the same lit lineage. The beat that sells the architecture: a thin
   gold radiation edge travels from parent to each child at the moment of fork. That gold edge **is the state memcpy**,
   rendered. No five spinners, no "prefilling 5/5". Cormorant number resolves: **"5 branches. 0 re-reads."**
3. **t=15-50s. Five futures run free, in parallel, legibly.** Five cards breathing gold, each scrolling one real-work
   line ("Patching retry.ts", "Running 12 tests", "3 failed, fixing"). No meter anywhere. One card hits a tool call: the
   `ToolProgress` chip appears and the JSON lands valid first try (M4/M6), so the feed stays calm, no red repair churn.
4. **t=50-65s. The handoff tap (M1 + H7).** On the winning branch, planner hands to reviewer. The viewer clicks the gold
   handoff edge: it expands the text-decode tap stratum, showing in plain text what state was passed (not re-typed).
   The single most uncopyable frame: a competitor passes text and pays a full re-prefill; HIDE passes state and shows
   it to you.
5. **t=65-75s. Resolution, the observatory at dawn.** Four branches quiet to amber/green; one settles green ("tests
   passed"). The digest number resolves: **"5 ran. 1 passed. 4 archived. 0 billed."** User accepts the winning diff
   (`AcceptDiff`, same j/k/a/r gesture); it absorbs cleanly. The rail dims from breathing gold to steady. Quiet
   completion, no fireworks.

**Moats exercised at once:** M1 (fork at t=8, handoff tap at t=50), M2 (free five-way fleet, "0 billed"), M4 (first-try
valid tool call), the Context Stack lit gold throughout (transparency as the constant spine). The gold rim carries every
one, so brand color and architecture are the same pixel.

**Minimum to film it:** Phase 0 native `.tq` (real free decode), M3 `RwkvState` clone/serialize (the load-bearing build,
makes the fork a genuine memcpy), M5 `copy_kv_prefix_to_slot` (the tap). The rest is already real (`hide-fleet`,
`ToolProgress`, `context.compile`, `AcceptDiff`, the durable log). **The one new visual primitive to build:** the gold
radiation-edge animation on fork and on handoff. It is the single asset that renders the memcpy; everything else is
composition of built parts.

**Why metered/closed competitors cannot copy it:** the fork is free and instant only because we own the state (a
transformer cloud must re-prefill or wrestle a quadratic KV cache for each branch, five times the bill, a claim their
architecture cannot make). The handoff passes state not text and proves it on screen (their passed state is server-side,
in their black box, unexposable). And the calm itself is the moat: five agents, no meter, one read is only calm because
the work is free (M2) and the tool calls are valid first try (M4). The serenity is structurally local.

---

## 7. The wedge, the MLP, positioning

### The wedge (the one thing)

**"Fork and try 5, keep the best, for free, and watch all five radiate."** A dev highlights a function, hits one key,
five agents fan out down five approaches at once. The whole repo is already warmed into each one (instant, no
re-prefill), so they start thinking immediately, not loading. Five gold cards breathe; each streams its real moves. You
take the winner's diff hunk by hunk. No bill, no limit, no spinner.

Why it converts: every metered competitor charges per agent per token, so "spawn 5 and throw 4 away" is the exact thing
their pricing punishes. We make the punished action the default verb. A Cursor/Devin user feeling the bill spike sees
this and the math is instant and visceral. Moat tie: M1 (the warm state forks free via `state.clone()` memcpy) plus M2
(zero marginal cost makes "keep the best" the default, not a paid tier). Both are structural: cloud cannot match free
fleets without going bankrupt. And it IS the doctrine's headline surface, the Workstation board of cards lit at the rim.

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

30-second pitch: "Cursor and Copilot just turned into taxi meters: bills spiked and a third of devs hit their limit
mid-task. Hawking runs a real coding model entirely on your Mac, so it is free forever and nothing leaves the machine.
And because we own the model's state, you can fork your whole warmed-up project five ways, run five agents down five
approaches at once for zero marginal cost, and watch every one work in plain sight. Most AI tools are a chat box bolted
onto a black box. We are the box, opened, lit at the rim, and yours alone."

Taglines (per doctrine): **"Open the box."** (the thesis in three words). **"Nothing leaves your machine. Nothing's
hidden from you."** (the dual face). **"Free fleets. Full daylight."** (the wedge).

The anti-metering, anti-black-box sentence: "They put your code behind a meter and an agent you cannot see into; Hawking
runs the whole thing on your machine, for free, with every move radiating in plain sight."

A naming note: do not lean on "Event Horizon" publicly. In this repo it is already the internal codename for the
speculative-decode proposal engine (`crates/hawking-core/src/speculate/*`). Keep it internal. The brand mines the same
physics one level up: the product is HIDE (the radiating box), the maker is Hawking, the mark is the event-horizon ring.
Use the event horizon as glyph and story, not as a shipped-feature name.

---

## 8. Risks and the honest edge

Almost every risk traces to one of two things: one untested load-bearing assumption (RWKV-7 can code, H3) and one
unbuilt load-bearing artifact (native `.tq` serving, H9). Ranked by likelihood times impact.

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

**The honest edge (the H-caveats, kept visible):** H1 SSM recall breaks on hard needles, route exact recall through RAG,
never say "perfect memory". H2 warm state saves recompute, it is not memory quality. H4 spec-decode is occupancy-gated
and can go negative under the very fan-out the wedge runs in. H6 perplexity is not generation quality, gate on real
generation. H7 latent handoffs are unauditable without the decode tap. H8 do not let a frontier BYO-key escape hatch
quietly become the thing that makes demos work, or the local pitch collapses.

**Wow-vs-prove recommendation: prove-first on the critical path, with a deliberately throwaway GGUF wow-spike running
strictly in parallel, capped at one engineer-week.** The telepathy/state-fork moats are worth nothing if the thesis gate
returns KILL; building the headline UX on a model that cannot code, secretly propped by a frontier escape hatch, is
exactly the failure the product exists to reject. But six weeks of plumbing with nothing to show drains belief, so the
GGUF spike keeps morale alive and validates the "fork and try N" UI feel early. It must be labeled internally as
**mechanism proof, not technical derisking** (it rides the wrong serving path, so its code is mostly thrown away). It
must never delay the gate.

---

## 9. Improvements and refinements

This is the section the four inputs did not write: where the unified plan is sharper than its parts. Opinionated.

**9.1 The single highest-leverage move: split native `.tq` into Stage A and Stage B, and gate ONLY on Stage A.** Every
input names native `.tq` as the SPOF and the thesis gate as the scariest decision, but treated together they compound
into one six-week bet that resolves late. Decoupled, they become a fast cheap experiment. Stage A (F16 dequant-on-load)
is low-risk wiring into `qwen_dense.rs::load` and answers the only question that matters first: can the local model
code? Stage B (native bitslice GEMV) is the RAM-cliff perf win and the "32B-on-18GB" headline, but it does **not** gate
the GO/KILL decision. A Stage B slip cannot block the verdict. This turns R1 and R2, the two Critical risks, into a
week-3 yes/no instead of a multi-week gamble. Do this first.

**9.2 Pull RAG (M4) forward, ahead of the state primitives it currently sits behind.** The roadmap sequences recall
after M3, but M4 is the actual fix for the biggest honesty risk in the whole pitch (H1). If instant resume (M3) ships
first and users infer the warm state remembers their repo, you manufacture a false-memory expectation before the
machinery that satisfies real recall exists. `hawking-index` is already built, so wiring it as the agent's recall path
is integration, not invention. Land RAG concurrently with or even slightly before the instant-resume surface, and let
M3's copy be strictly "resumed, not re-read". Neither the roadmap-interlock nor the red-team lens pushed this hard
enough: it is a sequencing inversion, not just a copy fix.

**9.3 Make the gold radiation-edge primitive the first real FE asset, not a late-demo polish.** The proof demo identifies
exactly one new visual primitive (the radiation edge that renders the memcpy on fork and handoff). It is the single
pixel where story, design, and architecture become literally the same thing. Build it early, on the stubbed `hide-serve`
path, against replayed event logs, so the headline motion is locked and battle-tested long before M3/M5 make it real.
This de-risks the most brand-load-bearing frame in the product while the backend critical path runs.

**9.4 Cut the overnight digest and telepathic handoff from v1, but keep their containers.** Both are gorgeous and both
tempt as the debut. The digest is a second-session feature with no daytime proof loop, and it is just the fleet board
resolved at dawn (so the wedge already exercises its primitives). Telepathic handoff is the deepest moat but invisible
to a first-time viewer and brand-dangerous without the audit tap. Build the containers now (the board is a grid of
cards, the digest is a Cormorant number) because they are pure design and risk-free, but do not wire either to a
capability claim in v1. Ship "fork and try N" first; both of these are chapter two.

**9.5 Make the doctrine Self-check a CI gate from FE commit one, and budget the Monaco re-skin as real work.** Both the
design lens and the red-team lens flag patchwork risk, but neither makes it enforceable. Under schedule pressure Monaco
ships looking like Monaco (VS Code blue, sans, flat) and trips the Self-check's "could be mistaken for VS Code." The
fix is process, not vigilance: wire the Self-check tells (blue/purple on screen, flat surfaces, a spinner standing in
for real work, a visible harvested seam) into review/CI from the first commit, and put the Monaco-to-mono/near-black/gold
re-skin on the schedule as a budgeted task, not an assumed polish pass.

**9.6 A synergy neither source fully saw: the diff-accept gesture is the personalization corpus, so instrument it on day
one.** S10 logs accept/reject and M7 consumes them, but they are described as separate milestones far apart. They are
one flywheel with a long fill time. The accepted/rejected diff stream (which the MLP already ships, because the
hunk-by-hunk gesture is in v1) is exactly the DPO corpus M7 needs. If the corpus is logged cleanly and scrub-on-write
from the very first diff review, M7 has months of real signal the moment it starts, instead of starting cold. The cost
is near zero (the logging seam exists in `hide-personalize`); the payoff is that the slowest-compounding moat starts its
clock at MLP launch, not at Phase 5. Wire the corpus capture now even though the training stays deferred.

**Two smaller cuts.** Drop the EAGLE-3/spec-decode lane entirely from any user-facing roadmap conversation: it is
batch-occupancy-gated (H4) and can go negative under the fan-out the wedge runs in, so it is the one optimization that
actively fights the headline. Keep it internal-only. And resist the feature catalog as a backlog: entropy-triggered
tools, custom samplers, INT8 state quant, AST grammars are a menu of research bets, not commitments; nothing past M6
starts until M0-M4 ship and hold.

---

## 10. The immediate next 3 actions (this week)

1. **Land native `.tq` Stage A in `qwen_dense.rs::load`** (F16 dequant-on-load), timeboxed to two weeks, and in parallel
   scaffold the `hawking-eval` coding bench. This is the critical path and the SPOF; everything waits on it. Stage B
   (bitslice GEMV) starts only after the gate is GO.
2. **Start `hide-serve` (axum HTTP/WS adapter wrapping `BackendHost`) plus the FE web shell** (Vite, `wire.ts`/`ipc.ts`,
   EventRouter, Zustand stores, three empty surface frames, the Context Stack rail skeleton) against a stubbed serve.
   Zero model risk, fully parallel, and it includes building the gold radiation-edge primitive (9.3) early.
3. **Run the thesis gate on Stage A against both RWKV-7 and Qwen `.tq` as soon as Stage A streams a token.** This is the
   GO/CONDITIONAL/KILL decision and it must arrive at week three, not week six. Branch all downstream work on its verdict;
   a CONDITIONAL routes the weak axis (likely recall, H1) into the pulled-forward RAG wire-up (9.2), not into a dead end.
