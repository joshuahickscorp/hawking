# HIDE State Capsule ABI

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` §3.1, §5 (code-verified)
Status: specification for a Hawking-native mechanism; every claim is tagged by the readiness of the primitive it depends on.

## 1. Why this is the load-bearing moat

Claude Code cannot structurally offer near-zero-cost session forking or "pass state, not text" handoff, because its model state lives on Anthropic's servers behind a stateless-prompt API: every resumed or forked conversation re-sends text and re-prefills, metered per token. HIDE runs the model locally, so the model's *execution state* is an object HIDE owns and can serialize, clone, and pin.

The catch, verified in the archaeology: **HIDE has the atom but not the capsule, and the atom is not exposed.** This ABI defines the object that turns the real-but-unwired RWKV state primitives into a shipping capability, and states honestly what is lossless, what is lossy, and what is only a cache.

## 2. What "state" actually is in Hawking today (verified)

| State component | Serializable? | Forkable? | Live-capture exact? | Evidence |
|---|---|---|---|---|
| RWKV-7 recurrent state (`RwkvState`, `DSSSMV1`) | **yes, byte-exact** | **yes (memcpy)** | only CPU-only or fresh-prefill-boundary | `rwkv7.rs:292-378`, test `:611-643` |
| RWKV-7 int8 packaged state (`DSSSMI8`) | yes | yes | lossy on wkv plane | `rwkv7.rs:395-483` |
| Transformer KV cache (`KvCache`) | **no** | no | n/a | `cache/mod.rs` (no serialization) |
| Tokenizer / position / boundary metadata | not bound to state | n/a | n/a | `DSSSMV1` header carries only shape + `fresh` |
| GPU-resident recurrent arena | no readback path | n/a | **not implemented** | `rwkv7.rs:1720-1723` |

**Conclusion:** the only serializable execution-state atom is the RWKV recurrent state. There is no transformer capsule and no unified (KV + recurrent + metadata) object. A capsule ABI must therefore (a) generalize beyond RWKV, and (b) bind the identity metadata that the raw blob omits.

## 3. Capsule anatomy

A `StateCapsule` is a content-addressed, self-describing object with three layers:

```text
StateCapsule {
  identity:   IdentityBinding      // what this state is valid against (Section 4)
  boundary:   CommitBoundary       // where in the token stream this was captured (Section 5)
  payload:    ExecutionState       // the actual reusable compute state (Section 6)
  provenance: Provenance           // how it was made, trust domain, audit hash
}
```

The capsule is the unit of save, load, fork, and handoff. It is **not** the transcript - the transcript is a separate durable projection. A capsule without its identity binding is a correctness hazard and must be refused.

## 4. IdentityBinding - what a capsule is valid against

Silent cross-version state reuse is a correctness bug (first-pass dossier §4.2; confirmed by the bare `DSSSMV1` header carrying no identity). A capsule binds, cryptographically (BLAKE3, reusing the pattern already in `hide-security` and `hawking-index` merkle), to:

```text
IdentityBinding {
  model_weights_id     // content hash of the loaded weights (incl. quant format, e.g. .tq STR2 id)
  arch_id              // qwen_dense | deepseek_v2 | llama | rwkv7 ...
  tokenizer_id         // tokenizer + special tokens + chat template hash
  prompt_abi_version   // byte-serialization contract of the prompt (see HIDE_SPEED_FRONTIER)
  tool_registry_id     // hash of the active tool namespace + schemas
  engine_build_id      // engine version + state-format version (DSSSMV1 / capsule vN)
  security_domain      // workspace + trust-domain id; caps do not cross domains
}
```

**Rule:** `load(capsule)` and `fork(capsule)` MUST verify every identity field against the live engine and refuse on mismatch with a typed error (reuse the typed-`Tamper` refusal pattern from `hawking-seed-c/providers/verify.rs`, never a string scrape). A mismatch is not degraded silently.

## 5. CommitBoundary - where capture is legal

Capture is only exact at a **committed token boundary** - a point where the engine's state fully reflects all emitted tokens and no partial GPU work is outstanding. Verified constraint: on the shipping macOS GPU decode path, `self.state` (CPU oracle) is stale relative to the live GPU arena, so a mid-stream capture is **not** exact until GPU→CPU readback exists.

```text
CommitBoundary {
  token_position      // absolute position in the sequence
  kind                // FreshPrefill | PostToken | Compaction | Fork
  gpu_synced: bool    // true only after a GPU->CPU readback (currently only CPU-path or fresh-prefill)
}
```

**Gate G-CAP-1 (hard):** until a GPU→CPU recurrent readback is implemented (`rwkv7.rs:1949-1965` currently returns logits without writing `self.state`; `copy_cpu_state_to_gpu_slot` exists only in the forward direction), live-GPU capsules MUST set `gpu_synced=false` and are valid only for CPU-path continuation or must be recomputed from a fresh prefill. The ABI does not pretend otherwise. Measuring GPU→CPU capture cost is the first experiment (`HIDE_EXPERIMENT_MENU`).

## 6. ExecutionState - the payload, per architecture

```text
ExecutionState =
  | Recurrent { rwkv_state: DSSSMV1_blob }                       // exact, forkable now
  | RecurrentPacked { rwkv_state: DSSSMI8_blob, kl_receipt }     // lossy, gated by a KL floor
  | Transformer { kv: <UNBUILT> }                                // requires KvCache serialization
  | Hybrid { recurrent, periodic_kv }                            // Qwen3-Coder-Next-class; requires both
```

Honest envelope:

- **Lossless:** `Recurrent` (RWKV `DSSSMV1`) fork and restore - proven byte-exact in the model-gated parity test.
- **Lossy (must carry a receipt):** `RecurrentPacked` int8 - never labeled lossless; only usable behind a KL-divergence floor, matching the existing model-gated gate.
- **Missing:** `Transformer` and `Hybrid` - require `KvCache` serialization + the checkpoint seam overridden on `qwen_dense`/`deepseek_v2`. This is a build item, and Qwen3-Coder-Next (the first-pass architecture-fit target) is a `Hybrid`, so **the flagship local coder needs the transformer/periodic-KV half built before its state is capsulable.** State-fork supremacy ships first on the RWKV lane.

## 7. The four operations

```text
save(session)          -> Capsule            // capture at a committed boundary
load(capsule)          -> session            // identity-verified restore, no re-prefill
fork(session|capsule)  -> session'           // memcpy clone; independent sibling; O(state bytes), no re-prefill
handoff(capsule, role) -> session@role       // load into a different model-role slot IF identity compatible
```

- `fork` is copy-not-merge by construction (`rwkv7.rs:376-378`): there is deliberately no state-blend inverse. Best-of-N branches are independent; they are reconciled by **execution evidence** (tests/build), never by averaging states.
- `handoff` is only valid when the target role shares `model_weights_id`/`arch_id`/`tokenizer_id`; across incompatible roles, handoff degrades to a typed structured checkpoint (text + evidence packet), not a state transfer. This is the honest boundary of "state, not text": within a role it is state; across roles it is a typed summary.

## 8. Exposure - the missing wire (build items, not inventions)

The primitives are unwired. To make the ABI load-bearing, three things land, in order:

1. **`hawking-serve` state routes:** `POST /v1/hawking/state/save`, `/load`, `/fork` returning/accepting capsule handles (not raw bytes over the wire unless requested); the shell-side `HttpKvStore` client already targets this shape (`hawking-context/kv.rs`, marked `[RUNTIME-SIDE - LATER]`).
2. **Session→slot affinity:** a client-stable session id pinned to a warm slot, so a fork/handoff targets a real warmed arena (today slots are anonymous `u32`).
3. **Warm-state persistence:** wire `SstateDiskCache` (`cache/sstate_disk.rs`, built + tested, zero callers) behind `save`/`load` for instant resume across process restarts.

## 9. Invalidation

A capsule is invalidated (and MUST be refused for `load`/`fork`) when:

- any `IdentityBinding` field mismatches the live engine;
- `gpu_synced=false` and the target is a GPU continuation;
- the repository snapshot bound to the session has diverged in a way that contradicts the capsule's assumptions (the capsule references, but does not contain, the repo snapshot id - edits invalidate context, not the recurrent state itself);
- the security domain differs;
- a partially-applied or aborted state (mirror the `hawking-seed-c` verify discipline: aborted work is not a valid artifact).

## 10. What this buys, stated conservatively

| Claim | Status | Basis |
|---|---|---|
| Fork a warm RWKV session in O(state bytes), no re-prefill | supported once exposed | `rwkv7.rs:376-378` memcpy, tested |
| Restore an RWKV session byte-exact | supported once exposed | parity test, CPU/fresh-boundary |
| Best-of-N from one warm state at ~zero marginal model cost | supported for RWKV; needs measurement | fork is cheap; verification cost is the real budget |
| Fork a transformer / Qwen3-Coder-Next session | **not yet** | KvCache/Hybrid capsule unbuilt |
| Exact mid-stream GPU capture | **not yet** | GPU→CPU readback unimplemented (G-CAP-1) |
| Cross-model telepathic handoff | bounded | only within an identity-compatible role; else typed checkpoint |

The supremacy thesis (`HIDE_SUPREMACY_THESIS.md`) may claim state-fork best-of-N as a **structural** advantage over Claude Code, but only on the RWKV lane today, and only after the three exposure build items land. Every stronger claim is gated on a measured experiment, not asserted.
