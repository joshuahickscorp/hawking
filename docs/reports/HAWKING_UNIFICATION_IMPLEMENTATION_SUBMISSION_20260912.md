# Hawking Unification — implementation submission

- **Snapshot:** 2026-09-12
- **Branch:** codex/hawking-unification-implementation
- **Base:** 8f10e8bb8ad4b1888c024a22db67b24472cf30f8
- **Implementation tip (last behavior-changing commit):** cc9e8c6c6
- **Worktree:** /Users/scammermike/Downloads/hawking/.worktrees/hawking-unification-implementation

This branch implements bounded ownership consolidations from the Hawking
Unification campaign. It is ready for review and integration planning. It has
not been merged, deployed, or used to restart a resident, service, ModelLake
watcher, GPU/ANE workload, or model process.

## Current completion estimate

The estimate is intentionally scoped rather than a LOC claim:

| Scope | Estimate | Why |
|---|---:|---|
| Whole Hawking Unification campaign | **~30–35%** | The highest-confidence ownership families are implemented, but the larger action-graph, Rust/Python ownership, runtime-policy, and active-Flash integration work remains. |
| Initial safe implementation tranche | **~85%** | HTTP lifecycle, resolved action identity, effective Rust serve policy, Gravity automatic-method execution, Rust test feedback, two Rust duplicate families, both causality consumer families, and shared optional file and value identity helpers are complete. Several larger and less uniform families remain. |

This is not a claim that 25% of source lines, runtime paths, or scientific
qualification has been unified. It is a planning estimate based on completed
authority migrations and the remaining known high-impact families.

## Implemented commits

| Commit | Result |
|---|---|
| e90e6dcd3 | One HTTP child-lifecycle implementation for Llama and MLX-backed server paths. |
| 998209267 | One complete resolved-action identity carried across Rust selection, catalog resolution, terminal action, serve, HTTP switching, and Web reuse. |
| 59aeae8c6 | Preserves resolved-action boundaries where the first migration exposed an edge case. |
| 7f5d0960b | Records the resolved-action evidence and explicit scope limit. |
| c39c320ff | Centralizes effective Rust serving-policy precedence into a typed, pure owner. |
| 3b21ea830 | Consolidates Rust test compilation and provides a fast source-edit feedback lane. |
| ef1cc9b90 | Makes the existing automatic Flash accounting method path bind fresh execution and verifier evidence. |
| f17e6a232 | Centralizes Llama Q4_K/Q6_K packed-weight window validation. |
| 962fbf61b | Centralizes Qwen30/Qwen80 Safetensors source-header parsing. |
| c6d795930 | Centralizes AgentOS gate-causality stamping across eight gates. |
| 68bd177f4 | Removes three identical Future causality fallback/predicate shim sets while retaining their distinct domain verdict guards. |
| 4c8edb82a | Centralizes 22 optional JSON-object readers and 14 optional streaming SHA-256 helpers in the stdlib-only persistence leaf. |
| cc9e8c6c6 | Centralizes nine JSON-compatible-copy and eleven in-memory SHA-256 helpers in the same leaf. |

## Canonical-owner map

| Family | Canonical owner | What was removed or migrated | Evidence |
|---|---|---|---|
| HTTP child lifecycle | hcli.backends._HttpChildLifecycle | Duplicated endpoint, log-tail, readiness, and teardown bodies in Llama/MLX server backends | [receipt](../../receipts/future/HAWKING_UNIFICATION_HTTP_CHILD_LIFECYCLE_20260912.json) |
| Admitted action identity | hcli.catalog resolved-action contract plus typed Rust transport validation | Path-only action handoff across selector, resolver, serve, switching, and Web reuse | [receipt](../../receipts/future/HAWKING_UNIFICATION_RESOLVED_ACTION_IDENTITY_20260912.json) |
| Rust serving policy | crates/hawking-serve/src/policy.rs | Repeated precedence/default interpretation in Rust serving callers | [receipt](../../receipts/future/HAWKING_EFFECTIVE_RUST_SERVE_POLICY_20260912.json) |
| Gravity automatic method execution | tools/foundry/query_gravity_methods.py typed execution outcome | The one automatic Flash accounting path now binds selection, staging, fresh output, verifier identity, and promotion | [receipt](../../receipts/future/HAWKING_GRAVITY_METHOD_EXECUTION_20260912.json) |
| Llama packed windows | crates/hawking-core/src/kernels/mod.rs checked_llama_b9430_weight_window | Repeated Q4_K/Q6_K byte-window validation in packed kernel wrappers | [receipt](../../receipts/future/HAWKING_RUST_WINDOW_UNIFICATION_20260912.json) |
| Qwen source headers | crates/hawking-core/src/model/source_safetensors.rs | Identical Qwen30/Qwen80 header parsing and source-index setup | [receipt](../../receipts/future/HAWKING_RUST_SOURCE_IO_UNIFICATION_20260912.json) |
| AgentOS gate causality | tools/verify/status_causality.py stamp_gate | Eight exact local fallback, predicate, and stamping helper sets | [receipt](../../receipts/future/HAWKING_AGENTOS_GATE_CAUSALITY_UNIFICATION_20260912.json) |
| Future causality consumers | tools/verify/status_causality.py emit and records_five_fields | Three exact cold fallback/predicate shim sets; their law-store, preflight, and contamination guards remain independent | [receipt](../../receipts/future/HAWKING_FUTURE_CAUSALITY_SHIM_UNIFICATION_20260912.json) |
| Optional file identity helpers | hcli/persist.py read_json_object_or_none and sha256_file_or_none | 22 exact optional JSON-object readers and 14 exact optional 1 MiB streaming SHA-256 helpers; local private names remain direct aliases | [receipt](../../receipts/future/HAWKING_PERSIST_IO_HELPER_UNIFICATION_20260912.json) |
| Value serialization and digest helpers | hcli/persist.py json_compatible_copy and sha256_bytes | Nine JSON-compatible copies and eleven in-memory SHA-256 helpers; local names and two compatibility imports remain direct aliases | [receipt](../../receipts/future/HAWKING_PERSIST_VALUE_HELPER_UNIFICATION_20260912.json) |

The migrations deliberately retain narrow adapters where a process, language, or
provider boundary needs one. Each receipt names the removed competing body and
the behavior deliberately left outside its contract.

## Test turnaround

The Rust fast-test lane now aggregates 135 direct integration sources across
10 packages in a single Cargo invocation. For a real comment-only Core edit,
the check lane completed in 3.03 seconds against a historical 36.42-second
generic compile reference: **12.02× faster**.

That is a practical iteration win, not a claim that every suite or every
feature configuration is 12× faster. The initial 20× stretch target is not
yet established; Core compiler front-end work remains the measured constraint.

See the [test-turnaround receipt](../../receipts/future/HAWKING_RUST_TEST_TURNAROUND_20260912.json).

The Future shim tranche removes 213 production lines. Its hermetic regression and
receipt add 274 support lines, so it is an ownership reduction rather than a
whole-commit LOC reduction; the ledger records both values explicitly.

## Validation summary

Each tranche has its own receipt and focused validation. Material examples:

- HTTP lifecycle: 99 focused tests plus 28 subtests, then 34 surface tests.
- Resolved action identity: 45 focused tests; 144 broader affected tests with
  two documented baseline failures outside the tranche.
- Effective Rust serving policy: 11 policy tests and 43 Rust CLI tests.
- Gravity automatic-method execution: 48 focused Python tests plus synthetic
  negative cases for stale output, absent output, verifier mismatch, malformed
  output, timeout, and promotion races.
- Llama packed windows: 2 window tests and 7 static kernel-verifier tests.
- Qwen source I/O: 6 parser tests, including real synthetic index consumers.
- AgentOS gate causality: 31 hermetic focused tests; exact pre/post output
  parity for eight gates under both supplied and absent evidence; static
  coverage lists all eight migrated gates as causality consumers.
- Future causality consumers: 48 combined hermetic focused tests; exact
  pre/post output parity for three wrappers, including the distinct
  law-store OVERREACHING edge and protected-field guards.
- Optional file identity helpers: 38 focused temporary-file tests, 22 existing
  persistence-authority tests, and exact pre/post parity across 210 helper
  calls covering valid, malformed, unreadable, absent, and over-1 MiB inputs.
- Value serialization and digest helpers: 61 focused helper tests; 89 combined
  focused tests including core persistence, tool dispatch, and three synthetic
  Flash source-boundary cases; exact pre/post parity across 69 helper calls.

The Python ownership changes also pass compilation, JSON validation, focused
contract tests, and whitespace checks. No live qualification outcome was used
as proof for a static consolidation.

## What this branch does not claim

The branch does not establish:

- full CLI, menu, Web, and automation generation from one action graph;
- full Rust/Python semantic unification or a tree-wide deletion percentage;
- unified Qwen38 or Gravity Safetensors admission behavior;
- full Gravity experiment, historical-receipt, Noetic, or gauntlet execution
  unification;
- model capability, numerical parity, GPU/ANE behavior, latency/throughput,
  daemon behavior, deployment, or ModelLake state;
- an independently rehashed legacy KIMI aggregate revision;
- a 20× all-suite Rust source-edit turnaround.

The remaining large work should continue as independently tested owner
migrations, not as a broad rename or an unreviewable cleanup.

## Integration and rollback

Review this branch as the ordered range from 8f10e8bb to cc9e8c6c6. Preserve
the main Flash worktree and integrate in a disposable worktree after comparing
the current main tip and its dirty frontier. Re-run the affected focused suites
against that exact integration base.

No live-state migration is included. Source rollback is a reverse-order revert
of the commits above. Any future model, service, or protected-hardware
acceptance belongs to the main Flash owner after integration.

## Immediate next work

The larger action graph, remaining Rust/Python policy overlap, and
Flash-adjacent runtime owners require separate scope and integration evidence.
The capped, replacement-decoding, prefix-hash, and raising file-I/O variants
remain deliberately separate; the new persistence receipt names their contract
boundaries. Provider/native telemetry copy helpers and the Flash pinned-manifest
admission path also remain separate pending focused contract review.
