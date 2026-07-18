# HIDE deep audit and facet ladder

Date: 2026-07-16
Branch judged: main @ 1c525380 (clean tree). The in-progress dense refactor is out of scope by owner instruction.
Method: 82-agent workflow. 14 facet auditors each returned a graded sub-ladder with file:line evidence, every blocker/high finding was handed to an independent adversarial verifier told to refute it, a completeness critic swept for missed facets and grade inconsistency, and 3 critical gap-fill audits ran. All grades are current-state on main, 0 to 10 where 10 is Linear/Vercel/Zed craft. Run-report and roadmap claims were treated as unverified; each was checked against source.

## TL;DR: one root cause explains almost everything

HIDE is two products wearing one skin. The **library layer** (kernel FSM, tool parser, tool loop, apply_patch guards, context compiler, MCP client, RWKV state-fork, redactor, at-rest encryption) is genuinely good, security-reviewed, and densely unit-tested. The **shipped product** cannot do the thing it looks like it does, because that library layer has almost no production callers.

The load-bearing fact, confirmed by three independent auditors and their verifiers:

> The live user turn (`POST /v1/hide/intent` -> `handle_intent` -> `spawn_submit_turn_generation` -> `generate_submit_turn`, crates/hide-backend/src/host.rs:222,402,801) is a **single-shot `runtime.generate` of the raw last message**, `messages: Vec::new()`, no system prompt, no history, no files, no tool dispatch, no FSM, `max_output_tokens` hardcoded to **256**. The fully wired `AgentKernel::builder` exists only in tests (hide-kernel/tests/full_run.rs:141, hide-fleet/tests/kernel_integration.rs:55). The production host builds `AgentKernel::new` = StubPlanner + runtime None + dispatcher None (host.rs:75, lib.rs:116-128). `run_agent_to_terminal` and `fleet_run` have zero production callers.

So on main today: the agent kernel never runs, tools never dispatch on a chat turn, the context compiler's output is thrown away (only its blake3 hash is kept), assistant text is never persisted, and every agent-work affordance in the UI (plan card, tool rows, diff chips, fleet cards, scrub/fork, pause/resume/cancel, steer) is a placebo wired to an intent the host logs and drops. This is a **wiring crisis over good parts**, not a parts crisis, which is why almost every facet's ceiling sits 3 to 5 points above its current grade.

Separately and independently: a DMG built from main is a **dead app for a stranger** (no workspace root passed at Finder launch kills the sidecar in milliseconds; no model is bundled or configured anywhere), and **CI has been red on main for a month** and does not gate merges.

## The facet ladder

Grade = current state on main. Ceiling = reachable with the known, mostly-mechanical fixes. Where the completeness critic flagged a grade as inconsistent with its own findings, the calibrated number is shown and the auditor number in parentheses.

| Facet | Grade | Ceiling | One-line verdict |
|---|---|---|---|
| **Agent and product** | | | |
| Agent-work UX | **2.5** | 7 | Looks like an agent IDE; the live path is a 256-token text demo. Every steering control is placebo. |
| Context and memory | **2.5** | 7.5 | The model receives only the raw last message. Strong context compiler exists, reaches the model in zero paths. |
| Hawking serving seam | **3.0** | 7.5 | Clean, well-tested seam, but no user can reach a live model; stateless untemplated 256-token turns; prefix reuse never fires. |
| Agent architecture | **4.5** | 8 | Kernel FSM is real craft; unreachable from the product. Assistant output never persisted; cancel dead end to end. |
| Agentic tool system | **4.5** | 8.5 | Parser/guards/MCP excellent in tests; on the 5 plan gaps: 1 half, 2 half, 3 no, 4 no, 5 one-third. |
| Product differentiation | **3.0** | 7.5 | Moat legs are engine primitives with no product surface. Thesis gate still never run. FIM absent. |
| **Craft (in isolation)** | | | |
| Visual doctrine | **5.5** (6) | 8.5 | Type/color/grid landed well; app now violates three doctrine absolute-nevers (glass, context bar, VS Code shell) undocumented. |
| Frontend engineering | **5.5** (6) | 8.5 | Good store and transport seam; controls that lie on the wire, mock leaks into prod build, dead flagship surface. |
| Backend engineering | **6.0** (7) | 8.5 | Excellent crate craft; graded in isolation from the product it fails to power; second unsandboxed exec path. |
| Interaction UX | **4.0** | 7.5 | Focus ring and Monaco split real; typing-eats-diff-hunks bug live; no keymap, no Escape stack, no model story. |
| Security | **4.5** (5) | 8.5 | Materially hardened vs 07-05; unsandboxed terminal exec, fs.read traversal, WS has no Origin check, LAN-exposed serve CLI. |
| **Process and ship** | | | |
| Ship-readiness | **2.5** | 7 | DMG from main is a dead app for a stranger: sidecar dies at launch, no model, no dead-child detection. |
| Testing and CI | **3.0** | 7.5 | Suites are real (~340 Rust + 101 FE, green locally); as a gate it is nonfunctional, red a month, merges on red. |
| Docs freshness | **4.0** | 8.5 | Canonical doctrine bans what the app is built from; HIDE is undiscoverable from repo-root docs. |
| **Gap-fills (critic-surfaced)** | | | |
| Crash durability | **2.0** | 7.5 | One torn write to events.jsonl permanently bricks the workspace. Reproduced live. No single-writer guard. |
| Privacy at rest | **2.0** | 8.5 | Every session byte is plaintext in a committable `.hide`; redactor and AES layer built, wired to nobody. |
| WCAG AA / a11y | **3.0** | 8 | Scaffolding real; core loop silent to screen readers; two ship-blocking keyboard hazards; false AA contrast claims. |

**Reading the ladder:** craft-in-isolation clusters at 5.5 to 6, product-as-experienced clusters at 2.5 to 3, and the gap between them is the unwired spine plus the ship-blockers. Move the spine and roughly ten facets rise together.

## Verified blockers

Each was confirmed by an adversarial verifier against main source (a minority were downgraded on verification and are not listed here).

1. **The agent never runs.** Live chat is a single-shot 256-token raw-text completion; the kernel, tool loop, and planner have no production callers. (agent-ux, agent-architecture, agentic-tools, backend-eng) host.rs:75,222,402,801; lib.rs:116-128
2. **The model gets no context.** Prompt = raw last message only. No system prompt, history, files, or tool results. Zero conversation memory turn to turn. (context-memory, hawking-integration) host.rs:837-846
3. **ContextCompiler output never reaches the model.** Only its blake3 hash is kept; the compiled prompt is discarded at the two call sites. (context-memory) lib.rs:72-92, driver.rs:200-201
4. **Assistant output is never persisted.** Tokens stream over an in-memory broadcast bus and are dropped; `open_session` replay loses the model's half of every conversation and turns user messages into notice spam. (agent-ux, agent-architecture) host.rs:848-855,882,971
5. **Live turns never end in the UI.** No turn projection on the real path, so the transcript sticks forever-streaming and the SteerBar never leaves. (agent-ux) host.rs:907-968
6. **Secret auto-compaction is a no-op loop.** The intent fires on schedule and lands in `_ => {}`. (context-memory) autocompact.ts:73-81, host.rs:262
7. **Typing eats diff hunks.** HunkReview installs global window-level j/k/a/r keydown with no target guard, always active; typing into the composer or Monaco silently accepts/rejects hunks. (ux-interaction, wcag) HunkReview.tsx:72-110
8. **Sidecar dies at every Finder launch.** No workspace root passed, cwd is `/`, `create_dir_all("/.hide")` fails EROFS on sealed macOS volume; `.spawn().ok()` hides the dead child. (ship-readiness, hawking-integration) app/src-tauri/src/main.rs:22-27, services.rs:143-144, project.rs:47
9. **No model in the shipped artifact.** `HIDE_MODEL_WEIGHTS` is read but set by nothing; no weights bundled or downloadable; hawking resolved by bare PATH name. Time-to-first-token is never. (ship-readiness, hawking-integration, product) host.rs:104
10. **CI red on main for a month, merges on red, branch unprotected.** (testing-ci) last green 2026-06-18; PRs 17-20 merged failing
11. **Frontend CI job has never once run.** `pnpm-workspace.yaml` lacks the `packages` field, so pnpm 9 dies at Install, so the voice gate and mock-in-dist tripwire have never executed. (testing-ci) pnpm-workspace.yaml, ci.yml:71-73
12. **Torn event-log write bricks the workspace.** One truncated JSON line makes hide-serve exit 1 at boot with no repair path. Reproduced. (crash-durability) event.rs:506-526

## The context-window contract: your core ask, both halves

You asked for "context window as long as possible and efficient, providing all the things a local user needs, using our format," and noted HIDE already sent asks to hawking about this. Here is the state of that contract on main.

### HIDE side (the consumer): the contract is written but the wire is cut

The good half: `hawking-context` contains a genuinely strong context compiler (reserve-then-fill, observation masking, recall-gated compaction, deterministic manifests) and it is tested. The honest-context manifest ceiling is even threaded into the model popover UI end to end.

The broken half: none of it reaches the model. The compiled prompt is discarded (blocker 3), the model receives raw text (blocker 2), compaction is a no-op (blocker 6), token accounting is `chars/4` everywhere live, the recall fidelity shown to the user is a `LinearFidelity` stub fed the wrong quantity, and the FE/BE manifest shapes do not match (`retrieved/tools/memory/dropped` on the FE vs `retained/dropped DroppedContextSpan` on the wire), so the ContextStack strata are dead and the Dropped stratum can crash. The `ContextStack` surface itself, the claimed differentiator, is **unmounted dead code** whose snapshot/skill/fork buttons fake success notices.

Net: HIDE's own context machinery is 80 percent built and 0 percent wired. Closing blockers 2, 3, and 6 is the single highest-leverage move for your stated goal, and it is wiring, not invention.

### Hawking side (the provider): Spine A delivered, the moat unexposed

hawking-serve delivered the Spine A contract well: `GET /v1/hawking/context` with native and effective ceilings, a measured `.tq` multiplier read from real artifact metadata (honest 8x cap), `recurrent_state_bytes`, and slot occupancy; the engine trait accessors landed with RWKV-7 overrides. That part is real.

What HIDE needs from hawking production that is missing today, in priority order (this is the concrete "if you see something, say something" list for the serving side):

1. **Session/state affinity on the native route.** Every `/v1/hawking/generate` is a cold, stateless prefill. This forfeits the entire measured SSM long-context moat and forces HIDE to resend everything as text (which it does not even do yet). Slot pinning keyed by session id, or explicit state save/load, is the unlock.
2. **HTTP exposure of the RWKV state primitives that already exist and pass parity tests** (`RwkvState::to_bytes/from_bytes/fork`, `StateShareGroup`, `Engine::save_checkpoint/load_checkpoint/fork_state`, rwkv7.rs:286-358, engine.rs:335-354). There is no `/v1/hawking/state` route. This is the single highest-leverage serve change: it makes the "pass state not text" moat and the F3 telepathic handoff real, and it makes the ContextStack fork button implementable instead of fake.
3. **Prefix reuse on the direct-admit path.** The Track 5.1/5.2 reuse machinery (`copy_kv_prefix_to_slot`, `SystemPromptKvBank`, lib.rs:862-948) only runs on the waiter-drain branch, reachable only when all slots are busy. The common local case is `max_batch=1`, which always takes direct-admit and re-prefills from position 0 every request. Moving the lookup into `admit` makes the measured ~84 percent prefix win real on the serve path.
4. **`--max-seq-len` on the Serve subcommand.** `max_seq_len` is hardcoded 4096 in serve's EngineConfig (lib.rs:509); `generate` has the flag, `serve` does not, so every serve context is capped at 4096 regardless of model or KV levers. This directly caps "as long as possible."
5. **A `tq_active` handshake.** The supervisor sets `HAWKING_QWEN_TQ=1` when a `.tq` sidecar exists, but it is a silent no-op unless serve was built `--features tq` (off by default). The host can believe TQ is live while serve fell back to Q4_K, and the multiplier still inflates `/v1/hawking/context`. Have `/v1/hawking/context` echo whether TQ actually engaged.
6. **A measured `fidelity(age)` sidecar per `.tq` SKU** to feed the already-built `SplineFidelity` evaluator, replacing the `LinearFidelity` stub currently shown to users as "recall N%".
7. **An rwkv7 branch in `render_chat`** (http.rs:451-457): RWKV-World models currently fall to the generic `<|role|>` template, not their trained format.

## Hawking production-side flags (speed and correctness, seen in passing)

You said speed is mostly your production concern but to flag anything. These are serving-side, file:line on main, not HIDE-plane defects:

- **Stop strings are unhonored by the batch scheduler** (http.rs:663-666 admits it), so every batched generation runs to `max_tokens`. This hits tool calls and native generate. Combined with the 256/512 hardcaps upstream, real coding answers truncate mid-function.
- **`response_format: json_object` is accepted then silently dropped on every serve lane.** The batch lanes never read `json_mode`; `forward_multiseq_*` apply no constraint mask; only the single-seq `forward()` honors it and serve never takes that path. Constrained tool-call decode cannot land on the batching path HIDE would use.
- **Spec-decode and `SpecGovernor` are dormant on the batched path.** `spec_gov.rs` ships but nothing in the batch driver invokes it; spec-decode remains single-sequence only. This is the load-bearing gap for the tool-latency thesis and the ToolSpec ~4x differentiator (which is entirely unconsumed library code in `tool_spec_decode.rs`).
- **Serve tokenizes the prompt twice per request** (http.rs:899-905) and holds the engine lock during tokenization, so a long prefill blocks new admits.
- **Decode loop busy-polls with 1ms sleeps when idle** (lib.rs:830-837): measurable idle CPU burn for a resident desktop sidecar.
- **HIDE adds a blocking `GET /v1/hawking/context` round-trip on the time-to-first-token path** and builds three `reqwest::Client`s per turn (host.rs:831,860,931).
- **The `hawking serve` CLI still defaults `--addr` to `0.0.0.0:8080`** (crates/hawking/src/main.rs:134) with no auth middleware. The library default and HIDE's explicit bind are loopback, so HIDE is safe, but anyone running the serve binary directly is LAN-exposed. Flip the clap default.
- **Chat SSE route emits no usage/stats object** (http.rs:576-644), so OpenAI-compatible clients get no tok/s telemetry; the native route does.

## What is genuinely good (do not rebuild)

- The **kernel FSM**: 12-phase governed machine, acyclic-plan guards, budgeted repair/replan, stall detection, blake3-chained events, a real integration test against real cargo/git oracles.
- The **tool library**: parser, ToolLoop (parse-lint-dedup-dispatch-feedback), apply_patch guards (empty-hunk, out-of-order, unverified-minus-line, envelope injection, git option-injection all verifiably closed), MCP client, all densely unit-tested.
- The **context compiler** in hawking-context, and the delivered Spine A serve contract.
- The **frontend spine**: one zustand store folding a typed UiEvent stream, a single transport seam with a prod-defaults-live design, hand-mirrored wire types that currently match `api.rs`, zero `any`, strict tsc green, Monaco properly code-split (677KB main vs 3.8MB lazy).
- The **crate layering** and typed error taxonomy across ~31k LOC of hide-* with almost zero non-test unwraps.
- The **security hardening actually landed** vs the 07-05 baseline: CORS origin-lock with a deny test, loopback binds, server-side approval round-trip, no-persist bypass, git option-injection closed, model-step dispatch deny-by-default allowlisted to read-only tools.
- The **visual craft that landed**: tokenized type scale, warm concrete ramp, shared 760px gridline, two-tier digest, working voice gate. Waves A-D were real.
- Latent but built and tested: the **redactor**, **AES-256-GCM at-rest storage**, and the **RWKV state-fork** primitive. All three need wiring, not authoring.

## Doctrine drift to rule on

The app has quietly forked from `docs/hide-bible/DESIGN_DOCTRINE.md` in three places the doctrine calls absolute-nevers, none written back into the canonical doc:

1. A full **Liquid Glass translucency system** (backdrop-filter blur, translucent tints, grain, specular sheen) on chrome and floating layers. Doctrine:389 bans "translucency or glassmorphism"; the Part VI ship-check fails on it.
2. A **context occupancy fill bar** in the model popover (SideBar.tsx:100-104). Your house rules make NO budget meter a non-negotiable; doctrine bans a context percentage on any surface in three places.
3. A retained **VS Code edge-to-edge shell** structure (theme.css:7 declares the hybrid; doctrine bans the packed edge-to-edge / VS Code clone).

These are not bugs; they may be good decisions. But the canonical doc no longer describes the product, so either amend the doctrine to sanction them or pull them. This is a fork you should rule on, not an execution defect.

## Grade calibration notes (transparency)

The completeness critic challenged six grades as inconsistent with their own findings. The calibrated numbers above already fold these in; the reasoning:

- **backend-eng 7 -> 6**: the identical "kernel unwired" defect is a blocker in three sibling facets; a crate collection whose flagship machinery cannot power the product it ships in is not a 7 against a Zed anchor. Unit craft is real, but it was graded in isolation from the product.
- **security 5 -> 4.5**: the README-to-RCE closure it credits describes a model step that never runs, so the mitigation is vacuous today; four highs including unsandboxed exec and a LAN-exposed CLI fit 4.5.
- **frontend-eng 6 -> 5.5**: wire-honesty (controls sending semantically wrong intents) is a correctness defect, not UX polish, and its own wire-honesty subfacet is a 3.
- **ux-doctrine 6 -> 5.5**: a currently-shipping violation of a stated non-negotiable (the context bar) should floor that subfacet, not read as style drift.
- **docs-freshness run-report-honesty 9 -> ~6.5**: the sample of 8 claims verified clean, but it missed three claims that are false in effect (the FE CI "blocker fix" has never executed, first-run onboarding claimed done is absent, Code-tab hover prefetch does not exist). Process integrity is high on what was checked; "CI now guards X" was false.
- **testing-ci test-quality subfacet**: 8.5 is hard to hold for a suite with a 50 percent flaky ULID test on the CI path and unexecuted integration targets; 7 to 7.5 is defensible.

## Recommended sequence

Ordered by leverage. The first block is the spine; nothing about the product is real until it lands.

**Block 0 (make it real, roughly one focused week):**
- Wire `SubmitTurn` through the real `AgentKernel::builder` path (runtime + dispatcher + planner) instead of `generate_submit_turn`. This alone converts blockers 1, 5, and most of agent-ux from placebo to real.
- Feed `ContextCompiler.prompt` into the InferenceRequest; assemble system prompt + history + files + tool results. Un-hardcode the 256 cap. (blockers 2, 3)
- Persist assistant tokens to the event log so replay and compaction have something to act on. (blocker 4)
- Pass a workspace root to the sidecar and add dead-child detection + a healthz probe; give the app a model story (bundle or download-on-first-run, set `HIDE_MODEL_WEIGHTS`). (blockers 8, 9)
- Fix `pnpm-workspace.yaml`, get CI green, protect main, fix the flaky ULID test and `cargo fmt`, make the mock tripwire minification-proof. (blockers 10, 11)
- Quarantine-or-truncate an unparseable log tail on open; take a workspace pid-lock. (blocker 12)

**Block 1 (make the moat touchable):** expose RWKV state fork/save/restore over HTTP; add session/state affinity and direct-admit prefix reuse on the serve side; wire the state-fork into a real ContextStack fork button. This is what turns "pass state not text" from a memory into a demo.

**Block 2 (make it honest and safe):** sandbox the terminal exec path to match `shell.run`; canonicalize the fs.read scope check; add an Origin check to the WS handshake; seed `.hide/.gitignore` and wire the redactor into the append path. Amend or enforce the three doctrine forks. Add live regions for the transcript and fix the two a11y keyboard hazards.

**Block 3 (make it defensible):** run the thesis gate (build the hawking-eval binary + a real corpus and point it at a real model through the real app); ship FIM tab completion; ship the NETWORK_LEDGER + air-gap toggle + CI egress test; register the MCP client so a user can add a server.

## Provenance

82 agents, 14 facet audits + completeness critic + 3 gap-fills, every blocker/high adversarially verified against main source before inclusion. Full structured output retained in the session scratchpad. Verified test counts observed this run: FE 101 vitest, hide-backend 70, hide-serve 7, all green locally; CI red since 2026-06-18.
