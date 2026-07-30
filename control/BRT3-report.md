# B-RT3 rev4 — restore user-owned blackbox matrix

**Rung:** B-RT3 (Core B accel research leaf elimination)
**Revision:** 4 (controller rev3 promotion rejection: user-owned matrix)
**Base:** `eada2080e0110a60a28292907490252246bee009`
**Base active LOC:** 436,895
**Verdict:** `READY_FOR_INDEPENDENT_REVIEW_PENDING_F1_AND_COREC_R1_PROJECTION`
**BRT1/BRT2:** BLOCKED (not landed); this cut assumes neither architecture.
**Commit:** NONE

## Product decision

| ID | Decision |
|----|----------|
| **BC-ACCEL-009** | Event Horizon / EAGLE5 runtime+CLI are a **product-released peripheral** surface: no invocable inputs, no observable outputs, no side effects (`side_effects: []`), no mock/fallback head. Record retained for historical traceability and rollback evidence only. |
| **BC-GENERATION-017** | Satisfied by **exactly one** historical comment in `crates/hawking/src/main.rs` containing the legacy identifiers `eagle5_head`, `--eagle5-head`, `speculate eagle5`, and `HAWKING_SPEC_DECODE=eagle5`. No live CLI field, mode, handler, or capture-help advertisement remains. |
| **Not released** | User-draft authority (BC-ACCEL-001/002/003), BASE_TRUE vs accelerated separation (BC-ACCEL-011), draft-token durable-sink rules (BC-SECURITY-001/015). Training entrypoints under `tools/training/**` remain pending Core C / C-HIST-R1. |

This is an honest product release of a net-negative research path. No mock facade, rejection stub, runtime fallback switch, empty handler, or feature-flagged dead body remains for the released surface.

## rev3 → rev4 repairs (binding controller findings)

1. **`REBUILD_BLACKBOX_TEST_MATRIX.json` was modified.** That path is one of the six user-owned dirty paths on main and is forbidden to every rebuild lane. Restored **byte-for-byte to base `eada2080`** (`sha256=680088865f075419cac61665c2e9209db7d48dc0db825b7d3f84cbcf15e9ec46`). Not present in the projection path set. Rev3's runnable matrix rewrite for BC-ACCEL-009 is **not** reintroduced or merged elsewhere.
2. **BC-ACCEL-009 side effects schema.** Constitution now records `side_effects: []` (empty array = honest "no side effects"); direct reproduction asserts `b['side_effects']==[]` rather than `['none']`.
3. **BC-ACCEL-009 verification path.** Constitution release reproduction runs as a **separate direct gate**. Committed blackbox matrix is the restored baseline only (`pass=86 fail=0 skip=124`).
4. **Evidence validation hardened.** `tools/verify/capability_manifest.py` requires `evidence` to be a nonempty string or nonempty list/object with meaningful values. Rejects empty string/list/object and values that only pass via `str(...)` equaling `"{}"`/`"[]"`. Negative selfchecks cover empty shapes and fake invocation (`true` / `lab --help`).

Rev3 runtime deletion, constitution product-release statement, live docs/CLI closure, capability released row, F1 identities (14 / `68b942ec05678825`), and user-draft/security preservation are retained.

## Old authority (deleted)

### Production modules (3567 active LOC)

| File | LOC |
|------|----:|
| `crates/hawking-speculate/src/eagle5.rs` | 1036 |
| `crates/hawking-speculate/src/eagle5_forward.rs` | 493 |
| `crates/hawking-speculate/src/eagle_proposer.rs` | 175 |
| `crates/hawking-speculate/src/retrieval.rs` | 239 |
| `crates/hawking-speculate/src/suffix_array.rs` | 172 |
| `crates/hawking-speculate/src/suffix_automaton.rs` | 395 |
| `crates/hawking-speculate/src/replay_oracle.rs` | 342 |
| `crates/hawking-speculate/src/tree.rs` | 315 |
| `crates/hawking-speculate/src/parallel_draft.rs` | 169 |
| `crates/hawking-speculate/src/safetensors_io.rs` | 231 |

### Assertion-accounted subject tests (740 whole-file active LOC + 2 identities in surviving e2e)

| File | LOC |
|------|----:|
| `crates/hawking-core/tests/qwen_eagle5_speculate.rs` | 193 |
| `crates/hawking-core/tests/eagle5_forward_parity.rs` | 182 |
| `crates/hawking-core/tests/eagle5_spec_parity.rs` | 102 |
| `crates/hawking-core/tests/eagle5_trained_head_load.rs` | 41 |
| `crates/hawking-core/tests/event_horizon_parity_prop.rs` | 162 |
| `crates/hawking/tests/spec_oracle_cli.rs` | 60 |

Whole-file subject release LOC remains **740**. The two EH e2e tests (and helper) are F1 `subject_deleted` identities; their physical lines land in rewritten residual on the surviving e2e file (−42).

Fixtures deleted with the subject tests (JSON, not LOC-active):
`eagle5_parity_q1p5.json`, `eagle5_parity_q3b.json`.

## Consumer / contract closure

| Consumer | Disposition |
|----------|-------------|
| `hawking-speculate` lib exports | Ten research modules removed; crate retains user-draft + security surface only. |
| `qwen_dense.rs` | Eagle5/EH runtime paths removed (rev2). Comments neutralized to user-draft terms (rev3). |
| `deepseek_v2.rs` | Eagle5 head/load/generate deleted (rev2). ExactShared preserved. |
| `engine.rs` | `SpeculateMode::Eagle5` and `eagle5_head_path` deleted. `from_cli` accepts only `exact-shared` / `off`. |
| `hawking` CLI | Live EH/EAGLE help removed; one historical comment only. |
| `docs/env_flags.md` | Live Spec-decode: user-draft / ExactShared / governor. EH/EAGLE names historical-only. |
| Constitution BC-ACCEL-009 | Released peripheral record; `side_effects: []`; direct reproduction PASS. |
| Capability manifest | `disposition: released` row for `runtime:eagle5_event_horizon` with product decision + rollback evidence. |
| Blackbox matrix | **Unchanged from base** (user-owned; excluded from projection). |
| Training/bench under `tools/` | **Still present** in capability inventory. Owned by Core C / C-HIST-R1. |

**Installed process/service requirement:** none. Default generate path never required EAGLE5/EH (default-off, net-negative).

## Preserved user-draft / security authority

Modules: `user_ngram`, `kv_dual`, `metrics_sep`, `governor`, `suspension`, `token_boundary`, `durable`, `verifier`, `policy`, `router`, `proposal`, `shared`, `cross_tokenizer`.

Surviving `user_draft_parity_e2e` tests: `user_draft_is_bit_identical`, `user_draft_bit_identical_fast_pruned_q4k`, `user_draft_bit_identical_full_fast_env`, `user_draft_propose_first_bit_identical_default`, `user_draft_propose_first_bit_identical_pruned_q4k`, `user_draft_propose_first_lossless_long`.

Gates: `cargo test -p hawking-speculate --lib` (56/56), compile of `user_draft_parity_e2e` and `e3_user_draft_gate_rule`, committed blackbox `pass=86 fail=0`, BC-ACCEL-009 direct constitution reproduction PASS, BC-SECURITY-015 PASS in blackbox.

## F1 assertion projection

Concurrent F1+F2 is on older base `ea33af24`. This branch does **not** import F1 apparatus.

- Identities enumerated: **14** Rust `#[test]` functions.
- Fingerprint algorithm: `sha256(newline_joined_identity_strings_document_order + trailing_newline)[:16]`
- Fingerprint: `68b942ec05678825`
- Receipt: `control/BRT3-f1-identity-receipt.json`
- Every identity: disposition `subject_deleted`, product decision `BC-ACCEL-009 release`, replacement `null`.
- The two additional identities vs rev2: `user_draft_parity_e2e.rs::event_horizon_bit_identical_default` and `…::event_horizon_bit_identical_fast_pruned_q4k`.
- Module-internal unit tests in the ten production modules and `spec_oracle_cli` are deleted with their subjects; inventory logical-cases drop (−66) is fully accounted for by those deletions plus the two EH e2e tests and is not a silent assertion loss.
- Surviving user-draft/security assertions are **not** deleted.

## Core C R1 dependency (exact)

Capability inventory (candidate) still lists these Python entrypoints:

- `tools/training/eagle5_quantize.py`
- `tools/training/eagle5_tau_eval.py`
- `tools/training/eagle5_train.py`
- `tools/training/eh_eagle_tau_sweep.py`

Core C recovery R1 / C-HIST-R1 is scheduled to release `tools/training/**`. B-RT3 does **not** call those entrypoints absent; it reports the dependency and stays pending both F1 and Core C R1 projection. Training was not edited.

## Six-bucket ledger (official full-tree)

Physical-line authority: `python3.12 tools/loc/hawking_loc.py`.

| Bucket | Value |
|--------|------:|
| Eliminated production | −3567 |
| Eliminated subject tests (whole files) | −740 |
| Rewritten residual (counted source only) | −1470 |
| Generated | 0 |
| Relocated | 0 |
| Facade | 0 |
| Added apparatus (`control/BRT3-report.md`) | +237 |

`Cargo.toml` / `Cargo.lock` / JSON receipts are not in the LOC language set (not counted). Markdown report is counted under shared. Comment-only rewrites are not claimed as product-elimination credit; they still appear in physical rewritten residual where line counts change. Matrix is restored to base and is not a projection path (JSON not LOC-counted regardless).

Monorepo: base **436895** → candidate **431355** → Δ **−5540**.

Reconcile: −3567 + −740 + −1470 + 0 + 0 + 0 + 237 = **−5540** (equals measured Δ).

Rewritten residual includes rev4 evidence-hardening growth in `capability_manifest.py` (166 → 303, +137) versus rev3's +74.

## Topology

| Dimension | Base | Candidate | Δ |
|-----------|-----:|----------:|--:|
| directories_all | 137 | 137 | 0 |
| directories_leaf | 109 | 109 | 0 |
| source_files | 1197 | 1182 | −15 |
| public_symbols | 9531 | 9465 | −66 |
| functions | 14675 | 14476 | −199 |
| files_over_1500_lines | 26 | 26 | 0 |
| tiny_forwarders | 13 | 13 | 0 |
| single_file_directories | 24 | 24 | 0 |

Official topology diff: **all dimensions improved or held**.

## Gates (exact exit codes)

| Gate | Result | Exit |
|------|--------|-----:|
| `cargo check -p hawking-speculate -p hawking-core -p hawking` | PASS | 0 |
| `cargo test -p hawking-speculate --lib` | 56/56 PASS | 0 |
| `cargo test -p hawking-core --test user_draft_parity_e2e --no-run` | PASS compile | 0 |
| `cargo test -p hawking-core --test e3_user_draft_gate_rule --no-run` | PASS compile | 0 |
| constitution BC-ACCEL-009 release reproduction (direct) | PASS (`side_effects=[]`) | 0 |
| live-reference scan (docs Spec-decode live table + main help) | PASS (deleted flags not advertised live) | 0 |
| `python3.12 tools/verify/blackbox.py` | pass=86 fail=0 skip=124 (baseline_runnable=86) | 0 |
| `python3.12 tools/verify/capability_manifest.py --selfcheck` | PASS (empty evidence + fake invocation negatives) | 0 |
| `python3.12 tools/loc/hawking_generation_audit.py --gate` | 0 earned / 0 reclassified | 0 |
| topology base→cand | all improved or held | 0 |
| `python3.12 tools/loc/hawking_inventory.py --gate …` | caps OK; tests −66 explained by subject release | **1** |
| `python3.12 tools/verify/capability_manifest.py --gate …` | 1 invocable, 1 released, 0 unaccounted | 0 |
| pure format-only diff hunks (`*.rs`) | 0 | — |
| `git diff --check` (code) | clean | 0 |
| Runtime EAGLE symbols in production Rust | 0 | — |
| `USER_DIRTY_PATH_DIFF` (incl. restored matrix) | none | — |

Inventory exit 1 is **expected** until F1 projects the 14 subject identities (and module unit tests go with deleted subjects). It is not an unaccounted assertion loss.

## Performance / TPS

Paired TPS is **not risk-connected**. Only unreachable default-off research branches and their consumer/contract closure were deleted. The default generate hot path (greedy / ExactShared / user-draft when explicitly enabled) is not reworked for throughput. No TPS claim; no paired bench required.

## Anti-gaming

No minification, comment stripping for LOC credit, line packing, extension games, archives, generated relocation, disabled code, or test-only deletion without subject release. Subject tests deleted only because their subject is explicitly released under BC-ACCEL-009. No whole-file rustfmt on legacy giants. Capability release row has evidence and null invocation (no `true` / `lab --help` facade). User-owned blackbox matrix is not rewritten by this lane.

## Rollback

Restore the ten modules, subject tests/fixtures, CLI test, reverse glue edits, restore constitution/docs/manifest/comment surfaces from base `eada2080`. Matrix already matches base. No data migration.

## Final measured numbers

| Metric | Value |
|--------|------:|
| Base active LOC | 436,895 |
| Candidate complete monorepo active | **431,355** |
| Δ active LOC | **−5,540** |
| Production release LOC | 3,567 |
| Subject test LOC (whole-file) | 740 |
| Eliminated (prod+tests) | −4,307 |
| Rewritten residual (signed) | −1,470 |
| Added apparatus | +237 |
| Six-bucket sum | −5,540 (reconciles) |
| Generated / Relocated / Facade | 0 / 0 / 0 |

F1 identities: **14** — receipt `control/BRT3-f1-identity-receipt.json`, fingerprint `68b942ec05678825`, disposition `subject_deleted` / BC-ACCEL-009.

External mirror: `/private/tmp/HAWKING_COREB_BRT3_ACCEL_RELEASE_REV4_20260729.md` (outside monorepo LOC).

---

## Controller receipt (rev4)

```text
VERDICT: READY_FOR_INDEPENDENT_REVIEW_PENDING_F1_AND_COREC_R1_PROJECTION
BASE: eada2080e0110a60a28292907490252246bee009
BASE_ACTIVE_LOC: 436895
CANDIDATE_ACTIVE_LOC: 431355
DELTA_ACTIVE_LOC: -5540
PRODUCTION_RELEASE_LOC: 3567
SUBJECT_TEST_LOC: 740
ADDED_APPARATUS_LOC: 237
F1_IDENTITIES: 14+68b942ec05678825
COREC_R1_DEPENDENCY: tools/training/** still lists eagle5_quantize.py,eagle5_tau_eval.py,eagle5_train.py,eh_eagle_tau_sweep.py (Core C R1 / C-HIST-R1 owns tools/training/**; not edited)
BC_ACCEL_009: peripheral product-released; inputs=none; outputs=none; side_effects=[]; failure=no deterministic mock; verified_by_run; direct constitution reproduction PASS
LIVE_REFERENCE_CLOSURE: docs Spec-decode live table has no EH/EAGLE flags; historical archaeology note only; main.rs HAWKING_QWEN_EAGLE5_CAPTURE=0; single historical BC-GENERATION-017 comment with four legacy strings
USER_DIRTY_PATH_DIFF: none
USER_DRAFT_AUTHORITY: preserved (user_ngram+governor+propose-first/bonus-first; BC-ACCEL-001/002/003; six e2e tests retained)
SECURITY: preserved (BC-SECURITY-001/015; durable/token_boundary/kv_dual/metrics_sep; blackbox BC-SECURITY-015 PASS)
RUNTIME_EAGLE_REFERENCES: 0
FORMATTING_ONLY_DIFF: 0
TOPOLOGY: directories_all 137->137; directories_leaf 109->109; source_files 1197->1182; public_symbols 9531->9465; functions 14675->14476; files_over_1500 26->26; tiny_forwarders 13->13; single_file_directories 24->24
TESTS: cargo test -p hawking-speculate --lib 56/56 exit=0; user_draft_parity_e2e --no-run exit=0; e3_user_draft_gate_rule --no-run exit=0
BLACKBOX: pass=86 fail=0 skip=124 exit=0 (committed baseline restored)
INVENTORY_CAPABILITY_MIGRATION: inventory --gate exit=1 (tests -66 accounted, caps OK); capability_manifest --gate exit=0 (1 invocable + 1 released); migration sample paths present
PERFORMANCE: N/A — default-off net-negative research release; paired TPS not risk-connected; default hot path not reworked
GENERATED: 0
RELOCATED: 0
FACADE: 0
SIX_BUCKETS: -3567 + -740 + -1470 + 0 + 0 + 0 + 237 = -5540 = measured delta
COMMIT: NONE
FIRST_FAILURE: none (inventory exit 1 is expected pending F1 projection, not unaccounted loss)
REPORT: /private/tmp/HAWKING_COREB_BRT3_ACCEL_RELEASE_REV4_20260729.md
```
