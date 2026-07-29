# GLM-5.2 rare-route basis pilot — Revision 13 executable closure

Revise the refused Revision-12 candidate in place. Preserve the exact-four
delivery surface, fake-only/source-body-free policy, no-real-acquisition
disposition, immutable-audit protocol, and every false authorization/MOP/TG/
HIDE/capable-provider fence.

The predecessor base HEAD is
`83bb2dcc2c30a0840afe19ff33ef253357e1b428`.
The only authorized Revision-12 predecessor files are:

- pilot Git blob `e85b63e1f360a5744d5a0b7629a7bf0447521936`,
  SHA-256
  `0e1754de07e9b620864b95a04720a399136cadd4ac6755a89fd505b22a3eb5cb`;
- focused test Git blob `bea8baf070f8e1a831dee1383814232ea4267c7e`,
  SHA-256
  `b0d516850cfa6b8c54a98e5d36ae9acbfbe126d089d0531cc6c14fa761540706`;
- JSON receipt Git blob `2bc5faa1e36798f3ac034a908409ec0c062eb09a`,
  SHA-256
  `35a7e8de00e25819e901df41d4956edcfef0be768c0742be1ce13415a9e9df28`;
- Markdown receipt Git blob `4dee63667f4a33303e8759c4a45946f55d816ada`,
  SHA-256
  `57beb722f06dfd2c80ca1cd57be3c5cc3c2b432c218fd64b40b702e92b55fd3c`.

Refuse if any Git blob differs or any full SHA-256 fails the report. Remove
`.serena`; modify exactly those four paths and nothing else.

## 1. Freeze Revision-12 refusal

Revision 12 is P0 refused. Its 141-pass suite did not prove the public behavior:

- arbitrary caller-supplied precomputed `Z` with a self-consistent hash was
  accepted and used without deriving `Z = SwiGLU(XW_gate, XW_up)` from
  authenticated source witnesses;
- reader, release, tensor, capsule, peer, and sidecar mutations returned `[]`
  at applicable direct or aggregate-local entrypoints;
- public aggregate tests did not call `aggregate()`, combined unrelated error
  lists, and allowed one layer to mask another;
- duplicate top-level test definitions were silently shadowed;
- evidence normalization erased exact 64-hex identities and large integers;
- R9/R10 lifecycle runners used source scans, constants, callability, and
  state-name counts instead of process-death/replay fixtures;
- two-root and exact-four gates compared literals instead of materialized
  immutable trees.

Do not commit or label the Revision-12 candidate immutable.

## 2. One-mutation, exact-reason science matrix

Every negative case starts from one production-shaped valid fake fixture,
mutates exactly one field, reseals only allowed enclosing objects, satisfies
all unrelated row-count/fit/device constraints, and asserts the exact intended
error token independently at:

1. the direct partial validator;
2. each applicable aggregate-local independent validator;
3. the actual public `aggregate()` entrypoint.

Never accept unioned `partial_errors + aggregate_errors`, any nonempty error,
or an unrelated earlier refusal as proof.

Required rows include release byte/identity mismatch; extra, missing, duplicate,
or reordered event/tensor/capsule/peer/reader fields; missing
run/operation/shard/authority/self-seals; wrong call counts; empty, extra, or
reordered authority inventories; authority-unlisted tensor/capsule; unknown
expert/peer; cross-operation/cross-capsule/cross-member seals; device
`987654`; sidecar dtype, raw/normalized shapes, rows, offsets/ranges, hashes,
equality claim, capsule/member IDs, and cross-seals.

Also execute the exact reproduced attacks: delete the sole release while
claiming zero live; delete `final_live_bytes`; delete or replace top-level
`reader_evidence_sha256`; use `reader_audit={}`; use an empty capsule-member
witness; stream `{expert_id:999}`; duplicate reader calls; use a wrong call
count; and supply authority-unlisted tensor or capsule witnesses.

Independently recompute retain-to-release one-to-one identity, bytes, and legal
order; the live-byte trace, peak, final zero objects and zero bytes; and the
absence of residual tensor, capsule, and peer-Z ownership.

Schemas are recursively closed and versioned. Exact authority-ordered
inventories and exact call/order/counter equality are required at every
validator, not merely nonempty intersections.

## 3. Derive precomputed Z independently

The proof binds exact shard/layer/capsule/member/source tensor identities,
names, hashes, dtype, shapes, offsets/ranges, authority and reader seals,
filtered-row identities/order, Z dtype/shape/hash, and canonical derivation
seal. It also binds and recomputes the target-route-union seal,
counterfactual-reserve seal and cap, authority peer order, member
contributions, and both raw and normalized Z relations. Ordinary and
precomputed paths emit the same complete canonical contribution schema.

Both partial and aggregate-local validators independently:

- read only authenticated fake source witnesses through the governed reader;
- derive the exact retained rows and raw/normalized inputs;
- recompute gate/up products and `Z = SwiGLU(XW_gate, XW_up)`;
- compare caller Z bits, shape, dtype, order, hashes, exclusions, and seals;
- reject arbitrary self-consistent Z, including a constant Z fixture with at
  least 32 valid post-exclusion rows that satisfies unrelated hidden-fit gates.

Caller claims, caller tensor hashes, synthetic producer IDs, or self-hashes are
not derivation authority.

Partial and aggregate implementations may share strict parsing and canonical
hashing only. They may not share semantic validator results, semantic helpers,
mutable objects, or error lists.

## 4. Actual public aggregate proof

Use isolated fake filesystem/authority fixtures and spies around the real
public `aggregate()` call. Prove each independent reader, sidecar, device,
filtered-row, peer-Z, capsule, and mutation checker executes exactly once in
canonical order and returns its distinct `aggregate_independent_*` error.

Repeat with the partial validator forced to `[]` and to one unrelated error;
the aggregate-local checker must still execute and refuse for its own reason.
All refusal cases prove zero score, release, unlink/delete, cache, ledger, or
receipt side effects.

## 5. Real lifecycle crash/replay matrix

Freeze these exact states and order:

- A0 `ACQUISITION_INTENT`, A1 `STAGING_CREATED`, A2
  `DOWNLOAD_COMPLETE`, A3 `BODY_VERIFIED`, A4
  `PUBLICATION_AUTHORIZED`, A5 `FINAL_PUBLISHED`, A6
  `INVARIANT_VERIFIED`, A7 `RECEIPT_PUBLISHED`, A8
  `LEDGER_COMMITTED`;
- R0 `PREPARED_FINAL`, R1 `PENDING_RENAMED`, R2 `PROBES_COMPLETE`, R3
  `UNLINK_AUTHORIZED`, R4 `UNLINKED`, R5 `COMPLETE_INTENT`, R6
  `COMPLETE_RECEIPT`, R7 `LEDGER_COMMITTED`;
- C0 `CACHE_IDENTITY_BOUND`, C1 `STAGING_COPY_DURABLE`, C2
  `SNAPSHOT_UNLINK_AUTHORIZED`, C3 `SNAPSHOT_UNLINKED`, C4
  `BLOB_UNLINK_AUTHORIZED`, C5 `BLOB_UNLINKED`, C6
  `CACHE_ABSENCE_VERIFIED`.

Recovery from A0 deterministically resumes the same operation and epoch; it
may not silently quarantine or start a new operation. Cleanup begins only
after A3 body verification. C0 through C6 run after A3 and before A4. P0 at C6
has the exact C0..C6 prefix and no `CACHE_CLEANUP_OK`; P1 appends the exact
event; A4 is forbidden until that event is durable. C6 alone is never terminal
or fully complete.

Each row uses three real fresh PIDs over one persistent temporary filesystem:

- P0 enters the actual public production entrypoint and `os._exit(91)` at the
  named production hook;
- the parent verifies exit code, hook transcript, and exact
  lstat/hash/inode/generation/mode/seal/Merkle inventory;
- P1 fresh-imports and resumes to the exact terminal;
- P2 fresh-imports again and returns a byte-identical terminal with zero writes,
  a byte/inode-identical filesystem, and exactly one matching event for this
  epoch, operation, and kind. Every preexisting ledger row remains
  byte-identical and the total count is the baseline plus one.

Production kill hooks are compiled and reachable only under the explicit fake
harness and must be unreachable in real-source mode.

After P0 for state index `i`, the immutable chain is exactly
`states[0:i+1]`, with no successor, alternate operation, or alternate epoch.
After P1 it is the exact full chain; after P2 it is byte-identical. A
prepopulated or parallel terminal chain refuses.

Expand every state and event publication into short-write, during-write,
pre/post file-fsync, pre/post no-replace publish, pre/post directory-fsync,
rename/unlink before/after, and post-mutation directory-fsync cuts. Include the
acquire rename gap, release unlink gap, cleanup unlink gaps,
absence-before-authority refusal, and post-authority absence without a second
unlink.

The exact per-state filesystem outcomes are normative:

- A0 has intent only: no staging T, final F, receipt, or event. A1 has one
  empty bound T. A2 has complete hashed T and recovery performs no redownload.
  A3 has verified T and any byte/inode change refuses. A4 executes both
  before-rename and after-rename-before-A5 cuts and refuses an unrelated
  same-byte F. A5 has exact F and no receipt/event. A6 has the invariant and
  P0 still has no receipt/event; recovery may publish only the deterministic
  receipt/event. A7 has receipt and no event. A8 has exact F, receipt, and one
  matching event for its epoch/operation/kind; missing, replaced, consumed, or
  duplicate authority refuses.
- R0 has exact F and the original preservation inventory. R1 has exact pending.
  R2 has two typed probes. R3 executes before-unlink and
  after-unlink-before-R4 cuts. R4 has an absence proof and never re-unlinks.
  R5 has intent only. R6 has receipt and no event. R7 has the exact terminal
  and one event; resident, stale, cross-operation, missing-intent, or
  receipt/event mismatch refuses.
- C0 has source S, blob B, and no copy. C1 has the verified copy. C2 executes
  before and after S unlink. C3 has S absent plus B and copy. C4 executes before
  and after B unlink with a fresh reference rescan. C5 has S/B absent plus
  copy. C6 P0 has the absence scan and no cleanup event; P1 appends exactly one
  matching `CACHE_CLEANUP_OK` for its epoch/operation/kind while all earlier
  ledger rows remain byte-identical.

Use exact closed LF-terminated canonical JSON, mode 100644, trusted-dirfd
access, and top-level fields:

`schema,kind,state,lifecycle_epoch,operation_id,shard,sequence,run_id,root_id,prior_terminal_sha256,predecessor_state_sha256,evidence,record_sha256`.

Evidence is recursively state-indexed and closed. Same-state replay requires
byte-identical destination; semantic equality or resealing is insufficient.
Mismatch returns stable `LIFECYCLE_REPLAY_MISMATCH`, performs zero writes, and
leaves the filesystem Merkle manifest unchanged.

The state-indexed schema manifest requires cumulative typed evidence:

- acquisition authority/admission, intended basenames and size/hash, staging
  and final descriptor identity, verification, publication, cleanup,
  invariant, intended/actual receipt, and event;
- release original-preservation inventory, exact final/pending/source,
  partial/authority seals, two typed probes, rename/unlink/absence, intent,
  receipt, and event;
- cleanup acquisition/lock, exact S/B/reference set, verified copy, separate
  unlink authorities, absence/replacement/link/process revalidations, terminal
  scan, and event.

A file identity is exactly relative basename, device, inode, kernel generation,
mode, size, SHA-256, and link count. On Darwin, generation is
`fstat().st_gen`; unavailable generation refuses rather than synthesizing a
counter.

The ledger event has exactly these nonnullable fields:

`schema,event_kind,sequence,lifecycle_epoch,operation_id,shard,run_id,root_id,predecessor_ledger_sha256,terminal_state_sha256,body_or_artifact_identity_sha256,receipt_or_intent_sha256,event_sha256`.

An artifact identity has exactly
`relative_basename,byte_length,file_sha256,semantic_sha256,record_sha256`.
The ledger uses a strict per-root-and-shard monotonic sequence and a nonempty
authority-bound genesis predecessor. Every lifecycle, ledger, artifact, and
nested evidence field gets delete, wrong-type, wrong-value, extra, alias,
reorder, resealed, and cross-operation attacks at direct and public
entrypoints; each must return the stable field-specific refusal with an
unchanged filesystem.

All canonical parsers refuse duplicate keys, nonfinite numbers, bool-as-int,
alternate case or spelling, unknown or out-of-order fields, extra whitespace,
missing final LF, trailing bytes/streams, wrong mode, symlinks, and nonregular
files.

Exercise separate real rows for lock-owner death, lock replacement/reuse, and
cross-epoch reuse, plus synchronized hardlink/path replacement/stale terminal
at every probe, authorization, and mutation boundary. Never swallow recovery
exceptions. Derive registry counts only from completed fixture rows, never
state-name tuples.

## 6. Test-integrity and evidence meta-gates

AST gates reject duplicate top-level test names, registry IDs, helper bindings,
skips/xfail/deselection, swallowed exceptions, `pass`, source-only checks,
constant/state-count/callability attacks, unused fixtures, and vacuous
any-error or merged-error assertions. Each meta-gate includes an injected
mutant that must fail collection or the governing gate.

Require
`expected_case_ids == collected_case_ids == executed_case_ids == passed_case_ids == preflight_reported_case_ids`.
Every case ID binds a unique fixture hash, production hook, three PID
transcripts, pre/post manifests, expected and observed result, and evidence
hash. Reused evidence hashes refuse. Include explicit mutants for arbitrary-Z
acceptance and merged partial/aggregate error masking.

Evidence hashes use typed canonical structured bytes. They may normalize only
fresh root paths, PIDs, and explicitly non-authoritative timestamps. They must
preserve every cryptographic identity, operation ID, byte count, size,
generation, row/call count, offset, and device value. Receipts differing in
only one 64-hex identity or one large integer must hash differently.
Normalization is a structural transformation of declared non-authoritative
fields before canonical serialization, never a regular-expression replacement;
raw transcripts still bind the actual PIDs.

Mutant-kill gates must fail if a validator returns `[]` on corruption, a
classifier returns constant fully-complete, replay ignores evidence or
predecessor, absence is accepted early, a public entrypoint becomes a no-op,
or a subprocess fixture is replaced by same-process execution.

The inherited local gate is explicit: the full focused suite has zero
skip/xfail/xpass/deselection; selftest and preflight pass; pack-v2 executes 28
cases; route-census executes 25; `py_compile`, strict JSON/parser, AST
integrity, and `git diff --check` pass.

## 7. Real two-root and immutable exact-four proof

Fresh-process generation must run under two unrelated materialized roots and
produce byte-identical JSON and Markdown receipts while preserving all
authoritative hashes and integers. Comparing logical strings or tuples is not
evidence. Each root must regenerate the reports through the public command;
copying a report is forbidden. During each run, the mutable/main worktree is
made unavailable; every authority/support input must resolve from the detached
candidate tree or an explicitly sealed content identity. A hard-coded shared
main path refuses even if it would produce equal bytes.

The mutable candidate exits only as `AWAITING_IMMUTABLE_AUDIT`. After local
tests, the supervisor—not the candidate—must:

1. verify exact four paths and no `.serena`;
2. create one immutable candidate commit/tree using only those paths and prove
   set-equal `git diff-tree` paths with regular mode 100644;
3. materialize at least two fresh detached roots from that commit;
4. rerun the full suite, external named counterexamples, exact-four status,
   and byte comparison in both roots;
5. require empty porcelain in both detached roots after the gates;
6. obtain independent lifecycle, science, and test-integrity acceptance.

## 8. No real acquisition

Current free space is below the admission rule
`free_bytes - sealed_shard_size >= 75,000,000,000`. Real acquisition,
rehydration, model-body reads, deletion, cleanup, or cache mutation remain
forbidden. The next command is `REFUSE_REAL_ACQUISITION`.

Return full Git blobs/SHA-256 identities, exact test/registry rows, executable
counterexample outputs, fake filesystem manifests, and a false-fence report.
