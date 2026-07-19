# HIDE Local Model Topology

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` §3.1, §3.6, §5 (code-verified); `HIDE_TWO_SURFACE_ARCHITECTURE.md` §2 (model plane); dossier `hawking_ide_frontier_2026_07_19.md` §4.3, §5.3, §5.3.1.
Status: specification for the model plane named in `HIDE_TWO_SURFACE_ARCHITECTURE.md`. Readiness of every Hawking mechanism is labeled real-and-wired / real-but-unwired / partial / stub / missing. PARITY (reproduce Claude Code) is separated from SUPREMACY (what a local runtime does better), and every supremacy claim is gated on a named build item.

## 1. Thesis: the objective is team capability density, not a main model

The unit of capability is not one model; it is the whole team of local lanes plus the router plus the verifier, measured as success@time and success@cost under a quality floor (dossier §5.3.1, INFERRED). A slow, expensive escalation model that avoids a failed exploration loop can be cheaper end-to-end than a fast model that thrashes; a deterministic ranker can beat a small model at retrieval. HIDE therefore specifies a **model plane of cooperating lanes**, and defers to `HIDE_CAPABILITY_DENSITY_EVAL.md` for how the ensemble is scored (never the main model alone).

This document specifies: the verified live model set and its readiness; the three lanes; the two orthogonal axes (model choice vs effort); phase-aware routing driven by trajectory evidence; the routing substrate Hawking already carries (real-but-unwired); and Qwen3-Coder-Next as the first architecture-fit target with an honest build ledger and the Hybrid-capsule caveat.

## 2. The live model plane (VERIFIED REPO)

Seven architecture readers exist under `crates/hawking-core/src/model/`. Readiness is engine-level unless noted; the continuous-batching serve loop is wired end-to-end for QwenDense on Metal (`HIDE_LIVE_ARCHAEOLOGY.md` §3.2), so non-Qwen archs reach generation through single-sequence CLI `generate()`, not the batched HTTP path.

| Arch (`model/*.rs`) | Family | State | Engine readiness | Serve-loop readiness | Evidence |
|---|---|---|---|---|---|
| `qwen_dense` | dense | KV (not serializable) | real-and-wired | real-and-wired (Metal continuous batching) | archaeology §3.1-3.2 |
| `llama` | dense | KV (not serializable) | real-and-wired | partial (batched loop is QwenDense-specific; CLI path only) | archaeology §3.1 |
| `deepseek_v2` | MoE (routed + shared experts, MLA) | KV/MLA (not serializable) | real-and-wired (`n_routed_experts`, `n_shared_experts`, `top_k_routed`, MLA KV at `deepseek_v2.rs:36-38,123`) | partial (not on the QwenDense batched loop) | archaeology §3.1; `deepseek_v2.rs:33-123` |
| `rwkv7` | recurrent (SSM) | **serializable, forkable** | real-and-wired; **the only engine carrying serializable state** | wired for generate/serve; **state ops unrouted** | archaeology §3.1; `rwkv7.rs:292-378` |
| `qwen_moe` | MoE (generic) | n/a | **stub** (`Error::Unimplemented`, "lands in Phase 3 (DeepSeek-V2-Lite ships first)") | missing | `qwen_moe.rs:18-19` (46 lines) |
| `gemma2`, `phi3`, `mamba2`, `mixtral`, `olmoe` | mixed | n/a | **packed** (extracted to `hawking-adapters-extra`; hydrate + re-add to load) | missing | `model/mod.rs:14-16` |

Cross-cutting weight format:

- **`.tq` sub-4-bit native serving is partial: feature+env gated, off by default.** Both `qwen_dense` and `rwkv7` read `.tq`, but only under the non-default `tq` cargo feature plus `HAWKING_*_TQ` env flags; the GPU bitslice GEMV is staged and the CPU RHT matvec is the parity oracle (archaeology §3.1, lever ledger line 162). Treat `.tq` as a real-but-gated capability, not a shipping default. Weight quantization, KV/state quantization, and context support are separate axes and must be reported separately (dossier §5.5, INFERRED); see `HIDE_SPEED_FRONTIER.md`.

**Topology consequence:** today the plane can *serve* one dense lane well (QwenDense on Metal), *run* three more archs through the slow single-seq path, and *fork state* on exactly one arch (RWKV-7). Everything below is the target the reintegration builds toward, tagged by the gap to this baseline.

## 3. Two orthogonal axes: model choice and effort

Model choice and effort are **distinct** and must not be collapsed into one dial (dossier §4.3, INFERRED).

| Axis | Controls | Set by |
|---|---|---|
| **Model choice** | which lane/arch handles a step (Reflex / Local agent / Escalation) | router policy (§5), phase (§6), trajectory evidence (§7), per-subagent `model` field |
| **Effort** | files read, tests run, tools invoked, alternative hypotheses, verification depth, best-of-N width | permission mode + phase + kernel budget, independent of which model runs |

A larger model at low effort can lose to a smaller model at high effort, and vice versa; the router optimizes the product, not either factor. This mirrors the flat effort-grounded inner loop in `HIDE_AGENT_KERNEL_OPTIONS.md` (planning and verification live *inside* the loop, not as a rigid twelve-stage FSM; dossier §4.4).

**Parity note.** Claude Code exposes model choice per subagent (`model` with `inherit` default) [parity: subagents.file_defined, DOCUMENTED], a separate fast model for the completion evaluator (default Haiku) [parity: goal.evaluator_loop, DOCUMENTED], and a separate classifier model for auto mode [parity: perm.auto_mode, DOCUMENTED, MEASURED ~83% catch / 0.4% FP]. HIDE's lane topology must reproduce "different model for different role" as a first-class concept, and the per-message transcript must show which model produced each message [parity: loop.collapsed_tools, DOCUMENTED].

## 4. The three lanes

| Lane | Purpose | Target policy | Live Hawking substrate | Readiness |
|---|---|---|---|---|
| **Reflex** | tab/FIM completion, classification, retrieval ranking, small transforms | smallest local model **or a deterministic algorithm** meeting the quality floor | deterministic ranking/RRF/PageRank retriever is in `hawking-index` (packed); FIM would run on a small local model (unassigned) | **missing/packed** (no reflex model wired; deterministic ranker unwired) |
| **Local agent** | interactive tool use, edits, tests, repository work | capability-dense local coding model on Hawking | today = `qwen_dense` serve path, but the live turn is single-shot 256-token `generate` with `StubPlanner` (no kernel loop, archaeology S1-S2) | **partial** (dense serving real; agentic loop packed/unwired; capability-dense coder is a build target, §8) |
| **Escalation** | ambiguous architecture, hard debugging, high-risk review | strongest permitted local **or** cloud model | serve is local-only; cloud escalation needs a provider/MCP adapter; `hide-tools` MCP client is packed | **missing** (no escalation route; cloud egress is default-off by doctrine, §9) |

Design rules:

- The Reflex lane must accept a **deterministic algorithm** as a valid "model" for tab-complete ranking and retrieval scoring; a small model is not required where a merkle/FTS5/PageRank/RRF index answers exactly (dossier §5.12 "small deterministic repository maps remain a serious baseline"; the index is `hawking-index`, packed, `HIDE_LIVE_ARCHAEOLOGY.md` §3.5).
- The Local agent lane is the flagship and is where Qwen3-Coder-Next is evaluated (§8).
- The Escalation lane is **permission-gated**: "strongest permitted" means the trust gate and egress policy (`HIDE_SECURITY_CONSTITUTION.md`) decide whether a cloud model is reachable at all; local-only is the default posture.

## 5. Routing substrate that already exists (real-but-unwired)

The absorbed **provider/capability registry** in `hawking-seed-c/src/providers/` is the natural substrate for role routing: one registry answers, per capability, which provider supplies it, why it was selected, its Seed-ABI compat, its LOC/bytes, its source, its tests, and its rollback (`providers/registry.rs:1-11`, VERIFIED REPO). It is integration-tested under real Seed authority but **ACTIVE_UNEXPOSED**: no CLI verb or HTTP route activates it (archaeology §3.6).

Two honest constraints:

1. **The registry cannot yet drive live model-role routing.** The arch adapters in `providers/adapters.rs` are **declarative descriptors, not a runtime engine**: they emit the same `crate::ir` op sequence as the real builder but from declared metadata "purely to produce an evidence plan-summary ... never used to execute a model" (`providers/adapters.rs:13-18`, VERIFIED REPO). Real execution still runs through `crate::adapter::build_plan` (`adapter.rs:72`). So the registry can *describe and select* a provider; it cannot *admit and run* one. Wiring it into a live router is a build item, not a shipping capability.
2. **Start with transparent rules; learn a router only after outcome-labelled trajectories** (dossier §4.3, §5.3.1). The registry gives a clean place to encode transparent selection rules first (declarative, inspectable, with a stated `reason`), which is exactly the "transparent policy + intentionally small model pool" the routing literature recommends before training anything.

## 6. Phase-aware routing

Route at meaningful phases, not once at the initial prompt (dossier §4.3). Each phase has a default lane and effort profile; these are transparent rules, overridable per-repo and per-subagent.

| Phase | Default lane | Effort profile | Primary route signals |
|---|---|---|---|
| triage | Reflex (or Local agent low) | minimal reads, no writes | prompt intent, repo-map hit, ambiguity estimate |
| exploration | Local agent (cheap-first) | wide reads, retrieval-heavy, no writes | index coverage, retrieval confidence (SWE-Router: cheap model explores first) |
| planning | Local agent, escalate on ambiguity | plan-as-data, acceptance oracles declared up front | architectural uncertainty, blast radius |
| patch generation | Local agent | edits + FIM; verifying edit applier | file scope, base-hash concurrency, diff size |
| test diagnosis | Local agent, escalate on repeated failure | run tests, read failures as data | repeated-failure count, flakiness |
| review | Escalation (or best-of-N judges) | read-only, high scrutiny | security risk, high-risk file touch |
| final explanation | Reflex or Local agent | summarize from evidence | length, audience |

Effort escalates within a phase on repeated failure before model choice escalates; model choice escalates when uncertainty, security risk, or blast radius crosses a threshold. The kernel that owns these transitions is specified in `HIDE_AGENT_KERNEL_OPTIONS.md`.

## 7. Route from trajectory evidence, not only the initial prompt

PRIMARY SOURCE (dossier §5.3.1):

- RouteLLM is a useful strong/weak baseline but its results are query-level and workload-specific.
- LLMRouterBench finds real model complementarity yet also finds complex routers often fail to beat simple baselines; pool selection can matter more than the router.
- TwinRouterBench routes from partial agent trajectories, tool logs, and diffs and reports a **53% spend reduction at matched resolution in its limited study** (MEASURED, preprint; reproduce before borrowing).
- SWE-Router argues a cheap model should explore first and routing should use files, tests, and trajectory evidence (recent preprint, not a production standard).

INFERENCE for HIDE:

- Route separately per phase (exploration, patching, diagnosis, review, explanation) using **trajectory evidence**: tests run, uncertainty, repeated failures, security risk, cache affinity, queue time, model complementarity.
- Start with a transparent policy and an intentionally small model pool; **collect counterfactual traces before training a router**. The trace fields required to do this are specified in the dossier §8.3 and consumed by `HIDE_CAPABILITY_DENSITY_EVAL.md`.
- Optimize success@time and success@cost under quality floors, not "cheapest-call percentage."

This is why the registry-as-transparent-rules step (§5) precedes any learned router: HIDE has **no outcome-labelled trajectory corpus yet** (the eval harness `hawking-eval` is packed/unwired, archaeology lever ledger line 164), so a learned router would be premature.

## 8. Qwen3-Coder-Next: first architecture-fit target for the Local agent lane

PRIMARY SOURCE (dossier §5.3): Qwen3-Coder-Next is an open-weight coding-agent model with 80B total / 3B activated per token, 48 layers, a repeating hybrid layout (three Gated DeltaNet + MoE blocks then one gated-attention + MoE block), 512 experts (10 activated + 1 shared), native 262,144-token context, FIM support, a dedicated tool-call format with tokenizer requirements, and training with executable environments.

**Why it is the first target (INFERENCE, not a support claim):** it combines low active compute, sparse experts, recurrent/linear-attention state, periodic exact attention, native coding/tool/FIM behavior, and a total footprint that can plausibly fit high-memory Apple systems after quantization. Architecture fit matters more than its leaderboard number, because public coding benchmarks are currently compromised by contamination and broken tasks (dossier §5.3).

### 8.1 Build ledger: what Hawking must build to serve it

Each item is currently **missing** in the active tree unless noted. This is isolated from the vertical-slice ship path (dossier Phase 1, item 9).

| Build item | What it requires | Nearest existing asset | Gap |
|---|---|---|---|
| Gated DeltaNet kernels + state semantics | Metal forward/decode kernels and a serializable linear-attention state | RWKV-7 SSM kernels + `DSSSMV1` state pattern (`rwkv7.rs:292-378`) | new arch; DeltaNet != RWKV WKV; state format is new |
| Exact sparse MoE route | 512-expert top-10 + 1 shared route, bit-matching the reference | `deepseek_v2` routed+shared expert path + `moe.rs` | different expert count/topology; route must be exact |
| Periodic attention + KV layout | the one-in-four gated-attention block's KV cache + position handling | `qwen_dense`/`deepseek_v2` KV + MLA | hybrid interleave layout is new; KV is **not serializable** anywhere today |
| Tokenizer / special tokens / chat template / tool parser / FIM contract | exact tokenizer, tool-call format, FIM markers | `tool_calls.rs` (Hermes/Qwen preamble), tokenizer in `hawking-core` | model-specific tool + FIM contract unbuilt |
| Quantized weight + state formats | `.tq`-class weight quant and a quantized hybrid state | `.tq` (partial, gated) | `.tq` off by default; hybrid-state quant unbuilt |
| Reference-parity + Apple measurement | next-token and long-continuation parity vs a reference runtime; prefill/decode/memory/power on Apple | `hawking-bench` (perf, wired); `hawking-eval` (packed) | eval harness unwired; no reference run |

### 8.2 The Hybrid-capsule caveat (honest, load-bearing)

Qwen3-Coder-Next's execution state is a **Hybrid capsule**: recurrent (Gated DeltaNet) plus periodic transformer KV. Per `HIDE_STATE_CAPSULE_ABI.md` §6, the `Hybrid { recurrent, periodic_kv }` payload is **missing** because the transformer half requires `KvCache` serialization and the checkpoint seam overridden on a transformer arch, neither of which exists (transformer `KvCache` is not serializable, archaeology §3.1). Therefore:

> **State-fork supremacy does NOT ship first on Qwen3-Coder-Next.** It ships first on the RWKV-7 lane (the only lossless, forkable, serializable state today), and Qwen3-Coder-Next becomes state-forkable only after the transformer/periodic-KV capsule half is built. Do not present the flagship coder as state-fork-capable until that build item lands.

### 8.3 Status label

Qwen3-Coder-Next is a **research bet with kill criteria** (dossier §9): pursue it as the first serious architecture-fit *study*, kill it if Apple performance or quant quality fails the local-agent envelope. It is **not** a settled "best target" (open owner decision, dossier §11.2: whether to evaluate another open coding model alongside it before kernel work begins) and **not** an immediate support claim.

## 9. Adversarial checks

### 9.1 "Qwen3-Coder-Next is the best local target"

Steelman: hybrid recurrent MoE, 3B active on 80B total, 262k native context, FIM + tool format, low active compute, plausibly fits high-memory Apple after quantization. This is the strongest capability-density architecture lead found (dossier §5.3).

Adversarial:

- **Its benchmark is not evidence.** Vendor-reported, on public coding benchmarks that are contaminated and broken (dossier §5.3). Architecture fit is a hypothesis about density, not a demonstrated capability.
- **The build cost is large and entirely ahead of us.** Every row in §8.1 is missing: new Gated DeltaNet kernels, a new exact MoE route, a new hybrid KV layout, a model-specific tool/FIM contract, hybrid-state quant, and reference-parity plus Apple power/quality runs. This is multi-week kernel work, correctly isolated from the ship path.
- **80B total is a real resident-footprint bet.** Even at sub-4-bit, `.tq` serving is feature+env gated off with a staged GPU GEMV (archaeology §3.1). "Fits after quantization" is an INFERENCE pending measurement, not a MEASURED fact.
- **Its moat half is unbuilt.** The Hybrid capsule's transformer side does not exist (§8.2), so the single biggest local advantage (state fork) does not apply to it yet.

Verdict: best **first architecture-fit study to investigate** for the Local agent lane, gated by kill criteria. Not "best target," not "support." A parallel open coding model should be considered before kernel work begins (dossier §11.2).

### 9.2 "Pure RWKV dominates"

Steelman: RWKV-7 is the only serializable, memcpy-forkable, byte-exact-restorable state engine in the tree (`rwkv7.rs:292-378`, archaeology §3.1); prior Hawking benchmarks show flat long-context decode where transformer decode decays (MEASURED, auto-memory `rwkv7_ssm_moat_measured`, ~14x at 8k, not re-verified here); fork is a pointer-copy with no re-prefill. The state-fork moat ships first on RWKV (`HIDE_STATE_CAPSULE_ABI.md` §6).

Adversarial:

- **Fixed-size recurrent state is lossy on exact long-range retrieval.** The dossier states it directly: "Pure RWKV remains valuable as a fast fixed-state lane. It should not be assumed to dominate a hybrid model on exact retrieval-heavy repository work" (§5.3). Repository coding is retrieval-heavy: exact symbol, def, and ref recall across a large tree. A fixed state cannot losslessly hold that; periodic exact attention exists precisely to recover it.
- **Dominance is per-lane, not global.** RWKV wins the Reflex/fast fixed-state lane and is the state-fork demonstrator. It is not established as the capability-dense Local-agent coder.

Verdict: RWKV-7 dominates the **fast fixed-state lane and the state-fork moat today**, and it is the correct first state-capsule vehicle. It does not dominate exact retrieval-heavy repository work; the hybrid lane exists for that. The topology wants both, which is the whole point of §1.

## 10. PARITY vs SUPREMACY

| Claim | Type | Gated on | Basis |
|---|---|---|---|
| Named model roles; per-subagent model choice; per-message model shown | PARITY | lane router (§4-5) + kernel loop | [parity: subagents.file_defined, loop.collapsed_tools], DOCUMENTED |
| Separate cheap model for the completion evaluator and the auto-mode classifier | PARITY | Reflex lane + fork exposure | [parity: goal.evaluator_loop, perm.auto_mode], DOCUMENTED |
| Phase-aware, trajectory-evidence routing over a small transparent pool | PARITY+ | transparent rules on the registry (§5); trace corpus for a learned router (§7) | TwinRouterBench / SWE-Router, MEASURED/preprint |
| Evaluator and classifier as warm local forks at zero marginal cost | SUPREMACY | RWKV state-fork exposure (`HIDE_STATE_CAPSULE_ABI.md` §8 build items) | fork is memcpy, `rwkv7.rs:376-378`, VERIFIED REPO |
| Best-of-N candidate plans executed speculatively in isolated local forks | SUPREMACY | state-fork exposure + kernel loop | [parity: perm.plan_mode superiority], gated |
| Team-of-models is a default, not a rationed premium (no metered quota) | SUPREMACY | fleet reintegration (`hide-fleet`, packed) | [parity: teams.coordinated superiority], no dollar meter (FE doctrine) |
| Capability-dense local coder (Qwen3-Coder-Next) on the Local-agent lane | SUPREMACY | the entire §8.1 build ledger + kill criteria | dossier §5.3/§9, research bet |
| Qwen3-Coder-Next state fork (Hybrid capsule) | SUPREMACY | transformer/periodic-KV capsule half (unbuilt, §8.2) | `HIDE_STATE_CAPSULE_ABI.md` §6, MISSING |

No supremacy row is a shipping capability today; each names its build item. Every parity row is achievable by reintegration plus routing, not new model research.

## 11. Build gates (ordered)

1. **Expose the registry as transparent routing rules** (§5): a lane selector reading declarative, inspectable rules with a stated `reason`, before any learned router. Cheap; unblocks phase-aware routing.
2. **Assign the Reflex lane**: wire the deterministic index ranker (`hawking-index`, packed) as the retrieval "model," and choose a smallest-model FIM policy for tab-complete. Do not require a model where a deterministic index answers exactly.
3. **Make the Local-agent lane real**: replace the 256-token single-shot turn with the flat kernel loop over `qwen_dense` first (the vertical slice, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §7), so a capability-dense coder can drop in later without re-plumbing.
4. **Stand up the trace corpus**: reintegrate `hawking-eval` and log the dossier §8.3 trace fields, so `HIDE_CAPABILITY_DENSITY_EVAL.md` can score the team and a router can eventually be learned from counterfactual traces.
5. **Start the Qwen3-Coder-Next feasibility branch** (§8.1), isolated from the ship path, gated by the §8.3 kill criteria.
6. **Escalation lane last**: only after the trust gate and egress policy (`HIDE_SECURITY_CONSTITUTION.md`) are wired, since "strongest permitted" is a security decision and cloud egress is default-off.

## 12. Cross-references

- `HIDE_TWO_SURFACE_ARCHITECTURE.md`: the model plane's place under both surfaces; the shared session core the router serves.
- `HIDE_STATE_CAPSULE_ABI.md`: the state each lane can (and cannot) fork; the Hybrid-capsule caveat for Qwen3-Coder-Next.
- `HIDE_AGENT_KERNEL_OPTIONS.md`: the flat inner loop that owns phase transitions and the effort axis.
- `HIDE_CAPABILITY_DENSITY_EVAL.md`: how the whole team is scored (success@time / success@cost under quality floors), not the main model.
- `HIDE_SPEED_FRONTIER.md`: `.tq` weight quant, KV/state quant, prompt ABI, batching and prefix/state caches that the lanes ride on.
- `HIDE_SECURITY_CONSTITUTION.md`: trust gate and egress policy that define "strongest permitted" for the Escalation lane and isolate caches by trust domain.
- `HIDE_LIVE_ARCHAEOLOGY.md`: the verified model plane (§3.1), the provider registry (§3.6), and the lever ledger (§5).
