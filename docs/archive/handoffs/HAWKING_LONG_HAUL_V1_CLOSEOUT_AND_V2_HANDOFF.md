# Hawking Long-Haul V1 Closeout and V2 Handoff

## 1. Old Root

- Root: `GOAL-B33C69F193`
- Disposition: `CLOSED_WITH_CARRY_FORWARD`
- Replacement programme: `HAWKING_MASTER_LONG_HAUL_V2`
- Source revision: `6a86a5cd49fbe9211ec11e133e9d70a4e2044238`
- Governing-objective digest: `ddbed4ef3f253418b2bd92d518ee2aa3fbdde8aad760351f22d4d46e4bcca1a1`

V1 is superseded because its work decomposition and supervision loop became stale. Its accepted evidence remains valid; its unresolved children remain evidence, not erased history.

## 2. Accepted Architecture

- Hawking owns Goal, WorkUnit, authority, effects, tests, Git evidence, and continuation.
- Providers are cognition workers: mutation proposals are validated and executed by the canonical `repo.edit` owner.
- Child `OUTPUT_UNUSABLE` / provider failures do not make the root failed.
- Read-only grounded prose can be normalized only after real read-tool evidence; mutation authority is not weakened.
- Frontier reconciliation, worker leases, and event-driven supervision remain canonical Hawking seams.

## 3. Accepted Source Changes

- `hawking/remote_cognition.py`: read-only evidence-backed result normalization and provider-independent mutation continuation.
- `hawking/workunit_owner.py`: child dead-lettering, provider-output classification, authority-gated mutation flow, and parent-liveness isolation.
- `hawking/goal_surface.py`: dependency-aware frontier reconciliation.
- `hawking/supervisor_watcher.py`: objective-digest caching, compact state fingerprints, and delta-only supervision decisions.
- `hawking/auto_orchestration.py`: bounded work-class concurrency recommendation, with admission still authoritative.
- `hawking/perception/document_extract_contract.py` and `hawking/perception/tools.py`: bounded artifact-backed `document.extract` contract and tool.

## 4. Test / Compile Baseline

- Current-source Hawking test baseline: `2000 passed, 20 skipped, 28 subtests passed` in 62.36 seconds.
- `git diff --check`: passed at closeout inspection.
- One known warning remains: `hawking.perception.terminal` uses `os.fork()` from a multithreaded process.

## 5. Known Provider Failure Classes

- `PROVIDER_OUTPUT_UNUSABLE`: the provider made read calls but did not emit required terminal structure.
- Provider tool-theater / prose-only continuation after admitted read tools.
- Exact exhausted children are preserved in `.hawking/provider-dead-letters/`; they must not be blindly replayed.

## 6. Unresolved Work

- `E_MACOS_CAPTURE_OCR`: carry-forward debt. Text `document.extract` foundation exists and deterministic tests pass; no provider-produced qualified live extraction receipt was accepted.
- `P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR`: carry-forward provider-contract debt.
- Auto/frontier planning needs V2 re-decomposition into larger coherent tranches, not more micro-audits.

## 7. Protected Resources

Flash, Pulsar, ModelLake canonical artifacts, Odyssey sealed receipts, protected model artifacts, credentials, and unrelated dirty work were not altered during closeout.

## 8. Budget / Spend History

- Authorized V1 ceiling: `$55.00`
- Reconciled spend: `$0.664715971`
- Remaining: `$54.335284029`
- Closeout remote spend: `$0.00`

## 9. Behavioral Lessons Now Canonical

- Failed child ≠ failed root; retain dead letters and continue dependency-valid work.
- Repeated known provider-format failures are deferred, not blindly retried.
- Use event-driven watcher decisions, objective digest caching, and delta-only reporting.
- Compute a runnable frontier; never let one `next_workunit` serialize the whole programme.
- Use bounded sprint envelopes and adaptive concurrency by work class, while retaining one-writer protection for overlapping mutation scope.
- Prefer coherent architectural tranches and batched review over patch-sized supervision loops.

## 10. Items That Must Not Be Retried Blindly

- `E_MACOS_CAPTURE_OCR-20260918-REOPEN`
- `E_MACOS_CAPTURE_OCR-20260918-BOUND`
- `F_AUTO_FRONTIER-20260918-AUDIT`
- Existing provider dead letters without a materially changed contract, runtime, or decomposition hypothesis.

## 11. Source / Repo Revision at Handoff

`6a86a5cd49fbe9211ec11e133e9d70a4e2044238` on the existing dirty `main` worktree. Preserve unrelated changes; do not reset as part of V2 import.

## 12. Recommended V2 Imports

Import the root metadata, three accepted WorkUnits, all dead letters, the provider-independent mutation machinery, document-extraction contracts, watcher/autotune evidence, current test baseline, and the protected-resource boundary. Create one new V2 root, construct a dependency graph first, then admit only large dependency-valid tranches under bounded sprint budgets.
