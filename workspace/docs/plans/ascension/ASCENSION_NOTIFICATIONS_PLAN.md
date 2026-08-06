# Ascension Notifications Plan

**Bible:** HAWKING_ASCENSION_BIBLE.md §28  
**Status:** plan + scaffold (gated on Proto-Frankenstein offload)  
**Scaffold:** `workspace/ops/ascension/notifications.py`  
**Tests:** `workspace/ops/ascension/tests/test_notifications.py`

---

## What tonight's code already proves

| Live file | Proven capability | Scaffold mapping |
|-----------|-------------------|------------------|
| `workspace/campaign/records/runs/frankenstein/v0_notifier.py` | Telegram via macOS Keychain (`security find-generic-password`) | Transport remains live-only; scaffold is event vocabulary + authority gates |
| same | Lane start/finish from `grok-run status` | `NotificationKind.LANE_STATE` |
| same | Capture WINDOW/LAYER/SHARD milestones | `NotificationKind.CAPTURE_MILESTONE` |
| same | 5-min heartbeat (running lanes, free disk, GPU%) | `NotificationKind.HEARTBEAT` |
| same | `DISK LOW` floor alerts | `DISK_FLOOR` + `MEMORY_DISK_PRESSURE` |
| same | Proto sealed only when receipt `endpoint == PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED` | `PROTO_SEALED` under `AuthoritySource.SEALED_RECEIPT` |
| same | Independent A–G bench after seal (`frankenstein_v0_seal verify`) | `BENCHMARK_COMPLETE` under `AuthoritySource.INDEPENDENT_HARNESS` |

**Do not edit the live notifier.** It is the production Telegram path for the current V0 campaign. The scaffold generalizes the **event vocabulary** and adds the §28 completion-authority rule so a future bus can replace ad-hoc `tg("...")` strings.

---

## Section 28 event set

| Kind | When | Typical authority |
|------|------|-------------------|
| `tg_rung_candidate` | Self-TG gauntlet proposes a rung | supervisor + sealed gauntlet receipt |
| `tg3_review_required` | TG3 stop-for-review rule fires | supervisor |
| `parity_rejection` | Parity harness fails | independent harness |
| `reviewer_disagreement` | Executor vs reviewer diverge | supervisor |
| `repeated_failure` | Same failure N times | supervisor |
| `memory_disk_pressure` | Pressure governor YELLOW+ | pressure_governor |
| `new_model_admitted` | Model passes admission | sealed receipt / human |
| `benchmark_complete` | Benchmark finished with artifact | independent_harness / sealed_receipt |
| `human_decision_required` | Escalation needs owner | supervisor |

Operational extensions retained from `v0_notifier` (not completion authority):  
`heartbeat`, `lane_state`, `capture_milestone`, `disk_floor`, `proto_sealed`.

---

## Hard rule (bible §28)

> **No notification may declare completion solely because a sandbox model said so.**

### Implementation

Completion-shaped kinds:

```text
benchmark_complete
new_model_admitted
tg_rung_candidate
proto_sealed
```

Allowed authorities for those kinds:

```text
human
sealed_receipt
independent_harness
supervisor
```

**Forbidden sole authority:** `sandbox_model`

Additionally, non-human completion-shaped events require `evidence_paths` (receipt / verify JSON / gauntlet artifact).

`build_notification(...)` sets `may_send=False` and a `refuse_reason` when the rule fails. `NotificationBus` records refused events separately — tests assert sandbox claims never ship.

---

## Mapping from v0_notifier string events

```text
"✓ {lane} done"                          → LANE_STATE
"{label} {n}/{tot} · free …G"            → CAPTURE_MILESTONE
"running: … · free …G · GPU …%"          → HEARTBEAT
"DISK LOW: {fg}G free (floor …G)"        → DISK_FLOOR / MEMORY_DISK_PRESSURE
"PROTO_FRANKENSTEIN_V0_…_SEALED …"       → PROTO_SEALED  (receipt endpoint only)
"BENCH DONE — verdict: …"                → BENCHMARK_COMPLETE
                                           authority=INDEPENDENT_HARNESS
                                           evidence=PROTO_…_INDEPENDENT_VERIFY.json
```

New surface not yet in v0_notifier (ascension programme):

```text
tg_rung_candidate
tg3_review_required
parity_rejection
reviewer_disagreement
repeated_failure
new_model_admitted
human_decision_required
```

---

## Transport plan (later)

1. Keep Keychain-backed Telegram send in a thin adapter (extract from `v0_notifier.tg`, do not rewrite live daemon in place until cutover)  
2. `NotificationBus` → adapter: only `may_send` events  
3. Log sink always on (today: `v0_notifier.log`)  
4. Optional second channel later — same event objects  

---

## Integration with sibling scaffolds

| Producer | Event |
|----------|-------|
| Pressure governor level change YELLOW+ | `memory_disk_pressure` |
| Pressure CRITICAL | `memory_disk_pressure` (critical) + maybe `human_decision_required` |
| Garbage cleanup receipt sealed | optional info heartbeat / ops note |
| TG gauntlet (future) | `tg_rung_candidate` / `tg3_review_required` |
| Parity harness | `parity_rejection` |
| Supervisor retry budget | `repeated_failure` |

---

## Non-goals (this scaffold)

- No Telegram API calls  
- No Keychain reads  
- No detached poll loop  
- No edit to `v0_notifier.py`  
- No completion message that trusts a sandbox model utterance  

---

## Remaining work

- [ ] Extract `tg()` + Keychain helpers into a shared transport used by both live notifier and future bus  
- [ ] Cutover: v0_notifier builds `NotificationEvent` objects before send  
- [ ] Dedup / rate-limit (today: floor warns bucketed by `fg//5`) as bus policy  
- [ ] Persist refused completion attempts as campaign evidence (sandbox overclaim ledger)  
