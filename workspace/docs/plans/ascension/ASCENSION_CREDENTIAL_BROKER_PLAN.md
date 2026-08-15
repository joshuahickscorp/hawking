# Ascension Credential Broker Plan

**Status:** plan + scaffold only (no live Qwen acquisition)  
**Bible:** HAWKING_ASCENSION_BIBLE §7  
**Gate:** Proto-Frankenstein offload before real Qwen streaming  
**Scaffold:** `lab/operators/credential_broker/`

---

## 1. Purpose

Install one reusable credential broker so every future model acquisition
(Qwen3-Coder-30B, Qwen3-Coder-Next 80B, and anything after) reuses the same
**stream → verify → Gravity transform → seal → evict source** discipline instead
of hand-rolling another family streamer.

Bible law:

> The models never receive the Hugging Face token.

Broker capabilities:

```text
search public model metadata
resolve official repository
pin immutable revision
list exact files and sizes
download approved ranges
resume
verify
evict
```

Before acquisition (all mandatory):

```text
official source
license
revision
hash inventory
architecture classification
storage forecast
runtime-memory forecast
Gravity plan
scientific purpose
disk-floor proof
```

Lifecycle:

```text
stream -> verify -> Gravity transform -> seal -> evict source
```

Never accumulate full source models and duplicate intermediate artifacts.

---

## 2. Existing patterns reused (tonight's real implementations)

This plan does **not** invent acquisition discipline. It generalises code that
already ran successfully:

| Pattern | Reference | What the broker reuses |
|---------|-----------|------------------------|
| Metadata-only official admission | `lab/operators/kimi_k3_source_admission.py` + `KIMI_K3_SOURCE_ADMISSION.json` | Pin 40-char commit; LICENSE + config + index SHA; LFS weight identities; architecture facts; **no weight body**; claim boundary; 15 GiB floor; `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; isolated HF/Xet caches; seal |
| Bounded header plan + floor + eviction assertion | `lab/operators/deepseek_v4_stream_executor.py` | `build_plan` / `validate_plan`; `assert_floor` before/during/after; `assert_source_evicted` on declared retention paths; header-only kind gates; seal receipts; execution_boundary honesty |
| Full-shard stream + verify + pack + multi-condition evict | `lab/operators/glm52_source_fetch.py` | Manifest sizes; SHA-256 verify; VERIFIED ledger survives body eviction; disk floor; six-condition `_evict`; deferred refusals; `token=False` on public `hf_hub_download` |
| Source-only reclaim | `lab/operators/glm52_layer_stream.py` (`evict`) | Unlink only `stream_root` shards; never control assets; `bytes_reclaimed` accounting |
| Rotation controller (N-1 seal/evict before N) | `lab/operators/condense_controller.py` (`GravityController`) | Phase machine; one heavy lease; `seal_and_evict`; no double source windows |
| Operational reserve / live disk sample | `lab/operators/glm52_grounding.py` | free/used disk + RAM/swap floors; refuse when reserve violated |
| Range restream gate | `lab/operators/glm52_range_stream_executor.py` + tools CLI | Owner-gated schedule/policy; Xet high-performance env; sealed terminal receipts |
| Reclaim after cloud seal | `lab/operators/frankenstein_v0_seal.py` | `reclaim_may_evict_superseded` only after confirmed cloud seal |

Thin CLI wrappers that already point at these bodies:

- `tools/condense/deepseek_v4_stream_executor.py`
- `tools/condense/glm52_range_stream_executor.py`
- `tools/condense/kimi_k3_source_admission.py`

---

## 3. Design

### 3.1 Module layout (scaffolded)

```text
lab/operators/credential_broker/
  __init__.py          public exports
  types.py             OfficialSource, ImmutableRevision, HashInventory,
                       FileEntry, ArchitectureClassification, StorageForecast,
                       RuntimeMemoryForecast, GravityPlanSummary,
                       ScientificPurpose, RangeRequest
  secrets.py           CredentialBroker, TokenHandle — token never on handle,
                       never in receipts; ambient HF_TOKEN refused on public path
  floor.py             assert_disk_floor / FloorProof (15 GiB minimum)
  preflight.py         AcquisitionPreflight + validate_preflight
  lifecycle.py         SourceLifecycle phases: PENDING → PREFLIGHT_SEALED →
                       STREAMING → VERIFIED → TRANSFORMING → TRANSFORMED →
                       SEALED → EVICTED (or FAILED)
```

### 3.2 Secret isolation

| Rule | Enforcement |
|------|-------------|
| Models never see `HF_TOKEN` | Token material lives only in `CredentialBroker._token_material`; `TokenHandle.as_dict()` always reports `token_material_present_on_handle: false` |
| Public path | `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, transport contract `token=False`; refuse `apply_public_environment` if ambient token is set |
| Gated path | `mint_gated_session(token, repository)` once; later calls take only `TokenHandle`; `authorization_header` is broker-owned transport only |
| Receipts | `assert_no_token_in_mapping` refuses keys/values that look like tokens |
| Revocation | `revoke` / `revoke_all` drop material without logging secrets |

### 3.3 Preflight (hard gate)

`AcquisitionPreflight` requires:

1. `OfficialSource` (repo + license + immutable revision)
2. `HashInventory` (exact files/sizes/hashes, same commit as source)
3. `ArchitectureClassification` (from official config metadata)
4. `StorageForecast` with `no_full_source_accumulation=True`
5. `RuntimeMemoryForecast`
6. `GravityPlanSummary` (plan pointer, not the transform)
7. `ScientificPurpose` (programme + success metric)
8. `FloorProof` with `status=PASS` matching storage floor
9. Absolute `source_retention_paths` (DeepSeek eviction-audit pattern)

Preflight does **not** download. Live transport is a later executor.

### 3.4 Lifecycle laws

Illegal without exception:

- stream without sealed preflight
- transform without verify
- seal without transform complete
- **evict before seal** (source-only reclaim only after durable seal)
- lower floor below 15 GiB
- accumulate full source + full intermediates (`StorageForecast` constructor)

Legal path matches Bible §7 and tonight's GLM/DeepSeek/Kimi practice.

### 3.5 Future live transport (explicitly out of scope now)

When Proto-Frankenstein is offloaded and a Qwen programme is authorised:

1. Metadata resolve (Kimi-style `HfApi().model_info`, public or gated via handle)
2. Pin commit; build/seal `AcquisitionPreflight`
3. Schedule ranges (DeepSeek header or GLM shard schedule)
4. Broker-owned download with resume; floor before/during/after each range
5. Verify size + sha256; append VERIFIED ledger row
6. Gravity transform (family-specific packer)
7. Seal artifact + receipt
8. Source-only evict; assert retention paths empty or scaffolding-only
9. Revoke gated session

**No Qwen acquisition in this task.** Scaffold only.

### 3.6 What not to rebuild

Do not re-copy GLM/DeepSeek/Kimi streamers per model. New programmes should:

1. Author a model-specific **schedule/policy** + Gravity transform adapter
2. Call the shared broker preflight + lifecycle
3. Plug transport into broker-owned download (when live path lands)

Family streamers remain valuable as **adapters**; the broker is the shared spine.

---

## 4. Integration points (later)

| System | Integration |
|--------|-------------|
| `lab.receipts.seal/verify` | Seal preflight + lifecycle snapshots |
| `lab.science_registry` | Register `credential_broker` as sealed operator |
| `GravityController` | Map window tasks onto `SourceLifecycle` or embed broker phases |
| GLM/DeepSeek executors | Optionally thin-wrap onto broker types without rewriting live paths now |
| HCLI / models | **Never** import `CredentialBroker.authorization_header` |

---

## 5. Tests (scaffold)

`lab/tests/test_credential_broker.py` — offline:

- revision pin / inventory / range types
- floor pass/fail / minimum floor
- ambient token refused on public path
- gated token not on handle; revoke works
- receipt token embed refused
- preflight retention + happy path
- full lifecycle + refuse early evict / early transform
- storage forecast forbids accumulation flag false

---

## 6. Remaining work (execution, not this task)

1. Live Hub metadata client behind `CredentialBroker` (public + gated)
2. Range download + resume with Xet/HTTP, floor sampling hooks
3. Wire VERIFIED ledger + source-only reclaim utilities into shared module
4. Qwen-30B / Qwen-80B programme schedules (after Proto-Frankenstein offload)
5. Science-registry registration + sealed preflight receipts in campaign evidence
6. Optional: adapt existing GLM/DeepSeek CLIs to emit broker-shaped preflights

---

## 7. Non-goals

- No live Qwen download
- No mutation of `ramanujan/` write paths
- No changes to live Frankenstein / GLM restream executors
- No push/PR/remote, no detached daemons, no venv commit
