# Q30 command-graph transition gap (S bucket) — mechanism table

**Profile seal:** `b22b4cc650ba58021f1d4ce6dacbbf7f1eadf2d5b13fb0fb98c8600686ab877f`  
**Status:** `EARNED_REAL_COMPLETE_TOKEN_STAGE_PROFILE_DIAGNOSTIC_NOT_TPS`  
**Live decode:** ~3.38 tok/s → TOKEN_NS ~296e6; 100 TPS needs ~10e6 ns/token (~29.6×).

## Sealed arithmetic (do not re-measure to “rediscover”)

| Quantity | Value | Note |
|---|---:|---|
| complete_token_host_wall_us | 842,003 | Instant around one all-48-layer token |
| production_gpu_busy_union_us | 372,654 | GPU busy union |
| production_gpu_work_sum_us | 398,731 | Sum of per-kernel GPU us |
| **production_gpu_idle_or_command_topology_gap_us (true S)** | **427,350** | GPU idle inside timestamp envelope |
| host_wall_outside_gpu_timestamp_envelope_us | 41,999 | Pure host outside GPU envelope |
| host-stage bucket `command_graph_transition_gap` | 609,865 (72.43%) | Includes prepare+submit+**wait** of multi-kernel CBs (GPU work + S) |
| command_buffers_committed | 291 | Cold first token |
| runtime_graph_dispatches (excl. vector decode) | 1,493 | Structural graph |
| expert_gate_up GPU share | 235,663 us / 59.1% of **GPU kernels** | Smaller half of wall — not the primary target |

### Cold-token CB identity

```
291 CBs = 193 cold vector-decode CBs + 98 structural graph CBs
193     = 48 layers × 4 RMSNorm vectors + final norm
98      = 1 embed + 48×(attn/router) + 48×(experts) + 1 final_head
```

Warm structural floor (before eliminating host route-id wait): **98 CBs / token**, still **48 host route-id readbacks**.

### Why expert_gate_up is not the primary target

- GPU work sum ≈ 399 ms; host wall ≈ 842 ms.
- Expert gate/up is 59% of GPU kernels ≈ 236 ms ≈ **28% of host wall**.
- True S (topology idle) alone is **427 ms ≈ 51% of host wall**.
- Recovering all of expert_gate_up still leaves the token >400 ms; recovering S is the only single bucket that can open a path toward the 10 ms/token rung.

Paired gate/up SwiGLU (`paired_direct_packed_gate_up_swiglu_scalar_order_production_no_parity`) is **already live** — do not re-propose as new.

---

## Ranked mechanisms (different KIND)

### M1 — Eliminate host route-id roundtrip (device-indexed expert tables / argument buffers)

| | |
|---|---|
| **Kind** | Binding topology / device-side resource indexing |
| **Bucket** | **S** (forced GPU→CPU→GPU mid-layer), residual **H** (host bind) |
| **Expected recoverable** | **200–400 ms / token** of the 427 ms topology idle. Reasoning: warm path still does 48 hard waits for `route_ids()` before expert encode; each wait drops the GPU out of the envelope. Removing the wait allows attn→expert→next-layer to stay on-device in one (or few) CBs. Upper bound is almost all of S; lower bound assumes residual encoder/driver gaps remain. |
| **Cheapest falsifier (no exclusive lease / no clean TPS)** | (a) Count `host_route_id_readbacks` on a warm token (already reported as `layers=48`). (b) Component: mock 48× `commit_and_wait` + tiny shared-memory read vs one CB with no mid wait on a micro graph — if mid-wait cost ≈ 0, M1 dies. (c) After an argbuf prototype: structural dispatch trace must show **0** host route readbacks and `command_buffers ≤ 2` per token with bit-identical route ids vs control. |
| **Must survive** | “Serial MoE route dispatch was slower” (different lever — serial *expert math*, not host bind). “ICB dead because encode is 0.5%” (encode ≠ wait topology). “Gap is not simply dispatch count” (M1 attacks **host visibility of route**, not dispatch shaving). |
| **Status this worktree** | **Designed, not implemented.** Requires per-layer argument buffers (or equivalent) holding all 128 expert gate/up/down buffer refs and kernels that index by device `route_ids`. |

### M2 — Collapse command buffers / single CB discipline (Q80 L1 shape)

| | |
|---|---|
| **Kind** | Submission topology (commit+fence count) |
| **Bucket** | **S**, secondary **H** |
| **Expected recoverable** | **50–200 ms** alone on warm path (98 → 1–2 CBs) **if and only if M1 lands**; without M1 the warm floor is **≥98 CBs** because of route wait. Cold path without prewarm was 291 CBs — vector-decode fold (implemented) recovers the **193 extra** CB waits on first token only. |
| **Cheapest falsifier** | Warm-token `command_buffers` must equal **98** after prewarm (or after first token cache fill). If already ~98 and topology idle still ~427 ms, pure CB-count reduction without M1 cannot explain S. Component topology test: `multi_cb` vs `one_cb` wall for N identical kernels (see unit test). |
| **Must survive** | “Host per-dispatch overhead dead Type-1” (that kill was encode share, not multi-CB wait). “CPU+GPU pipelining dead” (M2 is fewer fences, not async host overlap). |
| **Status this worktree** | **Partial.** Cold vector-decode no longer opens per-vector CBs on the production token path (folded into layer/final TCB). `prewarm_static_decoded_vectors()` collapses all 193 into **1 CB**. Full single-CB token still blocked on M1. |

### M3 — Serial multi-dispatch encoder (one encoder per wave)

| | |
|---|---|
| **Kind** | Encoder boundary topology (within a CB) |
| **Bucket** | **S** (inter-dispatch GPU idle inside a CB) |
| **Expected recoverable** | **50–250 ms** of the 427 ms idle if encoder end/begin is a dominant gap source. Sealed profile has **~1496 merged busy intervals** ≈ one island per dispatch — consistent with per-dispatch encoders. Serial group keeps Metal serial dispatch type (order-preserving WAW/RAW). |
| **Cheapest falsifier** | Unit test `component_command_topology_serial_vs_split_encoder_and_multi_cb` (component-only, no exclusive lease). If `serial_us ≉ multi_encoder_us` and both ≪ `multi_cb_us`, encoder boundaries matter less than CB fences. If `serial_us ≈ multi_encoder_us ≈ multi_cb_us`, M3 is dead for this GPU. Opt out: `HAWKING_QWEN30_SERIAL_ENCODER=0`. |
| **Must survive** | Megakernel/fusion 4.4× slower (M3 does **not** fuse kernels; it only co-encodes). Concurrent multi-CQ dead (M3 is single queue, ordered). |
| **Status this worktree** | **Implemented, default ON** for Off/CpuEncode TCB modes. No-op under `HAWKING_TCB_TRACE=gpu*`. |

### M4 — Residency / argument-buffer batching of `use_resource` / binds

| | |
|---|---|
| **Kind** | Host bind path |
| **Bucket** | **H** (and a thin slice of S if binds stall encode) |
| **Expected recoverable** | **≪ 20 ms** at 2.62 µs vs 4.5 µs prior measurement × O(10³) binds. Cannot close a 427 ms topology gap alone. |
| **Cheapest falsifier** | Component bind-only loop (no GPU wait) counting `set_buffer` vs `use_resource` wall; if delta < 1 ms at production bind counts, M4 is non-dominant. |
| **Must survive** | Prior positive micro on residency batching (keep as secondary). |
| **Status this worktree** | **Rejected as primary.** Keep as polish after M1–M3. |

### M5 — ICB pre-encode of the per-token graph

| | |
|---|---|
| **Kind** | Pre-encoded command replay |
| **Bucket** | **H** (encode), weak **S** |
| **Expected recoverable** | **≪ 5–10 ms** if prior dense kill still holds (CPU encode 0.22–0.51% wall). Dynamic top-8 of 128 experts forces rebind or full expert ICB tables → either rebuild cost or M1-class infrastructure. |
| **Cheapest falsifier** | CpuEncode TCB trace share on one warm token; if encode < 1% of wall, ICB stays dead. |
| **Must survive** | `dead_levers.md` ICB Type-1 kill; ICB cannot capture `set_bytes` without argbuf scalars. |
| **Status this worktree** | **Rejected as primary** unless M1-class argbufs exist and encode share reopens. |

### Explicitly not reopened

| Lever | Why |
|---|---|
| CPU+GPU pipelining | Type-1 dead; residual stream + route wait serialize host |
| Megakernel / 8-layer fusion | Measured 4.4× slower |
| Per-dispatch shaving | Prior win was elimination/ICB, not µs polish; gap ≠ dispatch count alone |
| Re-ship paired gate/up SwiGLU | Already production |

---

## What this worktree implements

1. **Fold cold RMSNorm vector decode into the parent layer/final TCB** — eliminates 193 dedicated CBs on cold first token.
2. **`prewarm_static_decoded_vectors()`** — one CB / one fence for all 193 static vectors before any measured token.
3. **Serial encoder groups (default ON)** on embed, per-layer attn/router, expert wave, final head. Opt-out: `HAWKING_QWEN30_SERIAL_ENCODER=0`.
4. **Component topology unit test** with printed multi_cb / multi_encoder / serial walls (component-only).
5. This mechanism table.

Parity invariants unchanged by design: packed values, device top-k routes, `fallback_count 0`, `all_layers_executed true`. Serial encoder uses Metal serial dispatch type (order-preserving).

---

## First serialized experiment for the human (exclusive / clean timing)

Run **after** no other GPU contender, against the admitted Q30 body. Prefer a **warm** token (call `prewarm_static_decoded_vectors` or discard first token) so cold decode does not re-contaminate CB counts.

```bash
# From the hawking worktree that has this commit, with the usual Q30
# admission env the production server already uses. Do NOT start a second
# server on :18430. Example shape — use your existing profiler entrypoint:

# 1) Component-only topology (no exclusive lease required; safe anytime):
cargo test -p hawking-core \
  component_command_topology_serial_vs_split_encoder_and_multi_cb \
  -- --nocapture

# 2) FIRST clean complete-token re-profile (serialized — you run this):
#    Baseline (serial ON, default) vs A/B serial OFF.
#    Compare: production_gpu_idle_or_command_topology_gap_us,
#             command_buffers_committed, host_route_id_readbacks,
#             complete_token_host_wall_us, sampled token id parity.
#
#    A: default (serial ON)
#    B: HAWKING_QWEN30_SERIAL_ENCODER=0
#
#    Exact operator (match your sealed profiler recipe; do not invent TPS):
HAWKING_TCB_TRACE=gpu_prod \
  python3 -m lab.operators.ascension_qwen30_complete_token_profiler \
  --manifest <admitted_qwen30_manifest.json> \
  --admission <admission_receipt.json>
# Then repeat with HAWKING_QWEN30_SERIAL_ENCODER=0 for the falsifier A/B.
```

**Decision rule for the first serialized run**

| Observation | Conclusion |
|---|---|
| Serial ON cuts topology gap by ≥10% and preserves token id / parity | Keep M3; proceed to M1 design |
| Serial ON ≈ serial OFF on topology gap | M3 falsified for production path; S is dominated by CB/route waits (M1/M2) |
| Warm `command_buffers` still ≫ 98 | Cold decode or diagnostic parity path still injecting CBs — fix instrumentation first |
| Warm `command_buffers` = 98 and gap still ~400 ms | M1 is the only large remaining S lever |

**Do not** run clean TPS / restart pid 95951 / take exclusive GPU lease from this agent path — those stay serialized under human control.
