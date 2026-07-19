# HIDE Supremacy Thesis

Run date: 2026-07-19 · Grounding: `HIDE_LIVE_ARCHAEOLOGY.md`, `HIDE_USER_LOVE_PAIN_MAP.md`, `HIDE_STATE_CAPSULE_ABI.md`, `HIDE_PARITY_GAP_MATRIX.json`.
Discipline: every superiority claim is labeled **structural** (Claude Code cannot do it by architecture) or **economic** (Claude Code could but the meter discourages it), and gated on the specific build item that makes it real. Nothing here is claimed as shipped; the archaeology shows the mechanisms exist as primitives, mostly real-but-unwired.

## 0. The thesis in one paragraph

> Reproduce the Claude Code workflow developers love (BOOK: parity), read their projects verbatim so switching is free (config compatibility), then win on the six things a metered, cloud, remotely-mutable runtime cannot structurally provide: no meter, no egress, no silent regression, warm-state forks and best-of-N, a resident shared-state daemon, and ACP-native local hosting. The differentiators are not features Claude Code lacks; they are properties of *where the model runs*.

Parity is not something to apologize for. The love/pain map shows captivity is almost entirely habit and config, so **HIDE's entry ticket is behavioral parity plus verbatim config migration**; the supremacy is what makes users stay.

## 1. Why parity must come first (the Apple-vs-Samsung gate)

The UX genome ranks the love: it concentrates in *steerable autonomy over a genuinely-understood repo* (interrupt-and-keep, plan gate, reversibility, legibility). These are table-stakes genes; if any is visibly rougher than Claude Code, HIDE loses regardless of its structural wins. The archaeology is encouraging here: the harness that produces this behavior (index + reserve-then-fill compiler + plan-as-data kernel + typed tools + deterministic oracles) **already exists** in the packed backend; parity is a reconnection, not an invention. So the thesis is credible: HIDE can be *as polished on day one* and *structurally better on day two*.

## 2. The six structural advantages

Each: the claim, the mechanism + readiness (from the archaeology), whether structural or economic, the build item it gates on, and the measurement that would prove it (see `HIDE_CAPABILITY_DENSITY_EVAL.md`).

### 2.1 No usage meter, no weekly wall (economic, closes the largest tolerated-pain cluster)

- Claim: HIDE removes the entire metering/anxiety cluster (§2 of the love/pain map): 5-hour windows, weekly caps, `/usage` local-session-only, per-token cost. `/usage` becomes performance telemetry, not a countdown.
- Mechanism: local inference. Readiness: **real-and-wired** at the doctrine level (the FE already forbids a budget meter; `hawking-serve` runs locally). It is economic, not structural: Anthropic *could* stop metering, but its business cannot.
- Proof: no measurement needed for the fact; measure the *behavioral* effect (users run best-of-N and agent teams by default, which the ~7x-token warning suppresses on Claude Code).

### 2.2 No code egress, true offline / air-gap (structural for inference; whole-run air-gap gated on egress enforcement)

- Claim: source never leaves the machine for *inference*; with egress enforcement in place, the entire "approved-domain exfiltration" class does not exist; regulated/air-gapped teams get a first-class tool, not a workaround. Honest boundary: only inference is egress-free by construction today; the tool layer (shell, the `hide-tools` MCP client which speaks Streamable HTTP, and serve binding `0.0.0.0` with no auth per G10) can still reach the network until Seatbelt egress enforcement lands, which is a build item, not a shipped guarantee.
- Mechanism: local inference + egress-default-off. Readiness: **structural today** for inference (no `api.anthropic.com` to reach); the *enforcement* (OS egress proxy) in `hide-security` is a seam (partial). Claude Code's own containment writeup names its network proxy as its biggest failure source; HIDE removes the component.
- Gate: `HIDE_SECURITY_CONSTITUTION.md` Article III (egress default-off) + wire Seatbelt enforcement.
- Proof: a verifiable no-egress mode (network disabled, show what attempted to reach the network). [Bible §50]

### 2.3 No silent server-side regression (structural)

- Claim: a version-pinned local runtime cannot be remotely degraded; "my tool got worse overnight" is impossible by construction.
- Mechanism: local weights + harness + prompts, user-controlled updates. Readiness: **structural**. The March-April 2026 Claude Code regression [DOCUMENTED] is the proof-by-counterexample: three product-layer harness changes degraded quality for weeks with no user opt-out. That cannot happen to a pinned local runtime.
- Proof: reproducibility, byte-identical runs with a fixed seed + resident weights (Hawking's determinism discipline: the exact-match lossless gate, `.tq` byte-identity).

### 2.4 Warm-state forks and best-of-N (structural, the signature advantage)

- Claim: fork one warm session into N isolated agents at ~pointer-copy cost, run different approaches, verify all, merge the winner, with no per-token cloud bill and no re-prefill.
- Mechanism: the state capsule ABI. Readiness (honest, from the archaeology): **real-but-unwired** on the RWKV lane. `RwkvState::fork` is a byte-exact memcpy, unit-tested (`rwkv7.rs:376-378`), but no HTTP/CLI caller exists; `save_checkpoint/load_checkpoint` are unrouted; the warm-state `.sstate` store has no caller. Claude Code's own "fork" is only a prompt-cache hit; Cursor's `/best-of-n` pays for N cloud VMs + re-tokenization.
- Gates (all in `HIDE_EXPERIMENT_MENU.md`): (a) GPU->CPU recurrent readback so live-GPU capture is byte-exact (**missing**, the first hard gate); (b) `/v1/hawking/state/{save,load,fork}` + session->slot affinity (**missing**); (c) the transformer/Hybrid capsule for a Qwen3-Coder-Next-class coder (**missing**, `blocked_on_model`). **Today the structural claim holds only on the RWKV lane and only after (a)+(b).** This honesty is the difference between a demo of unwired primitives and a shipped moat.
- Proof: fork latency, prefill avoided, verified gain per second vs one stronger model, merge success (`HIDE_CAPABILITY_DENSITY_EVAL.md`).

### 2.5 Resident shared-state daemon = one warm context across every surface (structural)

- Claim: Chat, IDE, SDK, and remote clients attach to ONE resident session; a surface switch or a new client is a pointer to the same warm state, not a transcript re-read; the IDE bridge has zero network round-trip.
- Mechanism: the two-surface architecture over the session core + state capsules. Readiness: **partial** (the FE already shares one store + wire across both chambers; the backend seam is packed). Claude Code shares a JSONL history and keeps *separate* histories across web/desktop/CLI [verifier]. Qwen Code's `qwen serve` daemon is the closest competitor pattern; Hawking goes further by holding resident KV/recurrent capsules.
- Gate: the shippable "shared session" version (one durable session, both surfaces read the same store/transcript) needs only the Phase 0/1 spine reconnect (`/v1/hide/*`). The differentiating "one warm SLOT, no re-prefill" version additionally needs session->slot affinity (archaeology G2, missing; slots are anonymous `u32`) plus the state routes (2.4b), so the no-reprefill claim is gated with the state moat, not reachable by reconnection alone.
- Proof: cross-surface handoff latency; resume fidelity with no re-prefill (the warm-slot version).

### 2.6 ACP-native local hosting (structural + interop)

- Claim: HIDE speaks ACP so it appears as a first-class agent inside Zed and JetBrains alongside Claude Code/Codex/Gemini, with no auth/subscription handshake and no egress, and warm-state resume the cloud agents cannot match.
- Mechanism: an ACP server. Readiness: **missing** (a build item), but low-cost and high-leverage. The competitor matrix shows ACP is "the neutral protocol" and Claude Code is itself only a *listed ACP agent*, not a host.
- Proof: HIDE runs in Zed as a local agent; connect latency and offline operation.

## 3. Capability density: the whole-team argument

Beyond the six, HIDE's lexicographic objective (capability density first) gives a distinct axis: verified software-engineering ability per active parameter / resident GB / joule / human intervention (`HIDE_CAPABILITY_DENSITY_EVAL.md`). The mechanisms:

- **Model-role topology** (`HIDE_LOCAL_MODEL_TOPOLOGY.md`): a Reflex + Local-agent + Escalation team, measured as a whole, beats a single model on density; the absorbed provider/capability registry is the substrate (real-but-unwired).
- **First-try-valid tool calls** (`HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md`): the packed `hawking-orch` tool-spec-decode (schema jump-forward + prompt-lookup) plus the exact-match lossless verifier remove tool round-trips losslessly. Structural: HIDE owns the runtime; a cloud API cannot mask logits per token for you.
- **`.tq` sub-4-bit native serving**: density via the RAM cliff, feature-gated today.
- **Context-not-tokens**: the reserve-then-fill compiler + living index put a small exact working set in front of the model instead of stuffing attention, and the warm capsule amortizes the system/instruction prefix to ~free after the first fork.

## 4. What is NOT a HIDE advantage (honesty ledger)

- **Model judgment.** The loved "takes initiative / pushes back" quality is mostly the model. HIDE inherits a *local* model whose judgment is gated on a capable coder (Qwen3-Coder-Next-class, `blocked_on_model`). HIDE does not claim to out-reason Opus 4.8; it claims to match the *workflow* and win on the six structural axes while a capable-enough local coder makes the judgment good-enough.
- **Ecosystem maturity.** Claude Code has marketplaces, a large skill/plugin ecosystem, and years of polish. HIDE's counter is verbatim config/skill/MCP compatibility (import the ecosystem) plus local-only capabilities cloud plugins cannot ship.
- **Best-of-N on transformers today.** Gated on the transformer/Hybrid capsule (missing). The RWKV lane ships first.
- **Any claim of "fastest/most-capable/densest"** is unearned until the reintegrated `hawking-eval` harness produces a receipt on the real app path (`HIDE_CAPABILITY_DENSITY_EVAL.md`).

## 5. Dependency spine of the thesis

The six advantages are not independent; four of them route through the same two build items:

```text
                          reconnect the spine (Phase 0/1)
                     (hide-core + hide-serve + context + index + kernel + tools)
                                     |
              +----------------------+----------------------+
              |                      |                      |
   2.5 shared SESSION       2.4 state moat prerequisites   parity (BOOK IV)
   (shared transcript                |
    ships at the spine)              |
                     +---------------+---------------+
                     |                               |
        state.gpu_readback (2.4a)        state.http_routes + session->slot affinity (2.4b)
                     |                               |
              2.4 warm-state best-of-N (RWKV lane)   |
                     |            2.5 warm-SLOT daemon (no re-prefill) also needs 2.4b affinity
                                     |
                     2.4c transformer/Hybrid capsule (blocked_on_model)
```

**Read this as: reconnect the spine (parity and a shared-transcript session ship here), then unlock the state moat (GPU readback + state routes + session->slot affinity give warm-slot no-reprefill and best-of-N on the RWKV lane), then generalize it to a local coder (transformer/Hybrid capsule).** That is exactly the build ladder (`HIDE_PRIORITIZED_BUILD_LADDER.md`).

## 6. The one-sentence claim HIDE can defend

> On day one HIDE reproduces the Claude Code workflow developers love and imports their projects unchanged; on day two it removes the meter, the egress, and the remote-regression risk that are structural to a cloud runtime; and on the path beyond, its warm-state forks turn "try five approaches and keep the verified winner" into the default rather than a metered premium, on hardware the user already owns.
