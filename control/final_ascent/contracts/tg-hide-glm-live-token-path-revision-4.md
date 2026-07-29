# Temporal Gravity → HIDE live-token path — Revision 4 production closure

Revise the refused Revision-3 candidate in place. Preserve every earlier
typed-decision, durable-terminal, poison/reset, monotonic-timing,
source-body-free, default-off, no-MOP, false-fence, and no-promotion
requirement.

The only authorized predecessor has base
`35cbd3046aea21abae2a4e8f4046c3b823ec6ff5`. Its canonical manifest over every
changed or untracked non-`.serena` path as
`path<TAB>git-blob-id<LF>`, sorted under `LC_ALL=C`, has 29 paths and SHA-256:

`5953cef5b1196c6f3414f435cd6b2fd3c1eac9abd771da58449efe507e435601`.

Refuse before editing if it differs. Remove `.serena`.

The authority file remains byte-exact:

- `control/final_ascent/contracts/tg-hide-glm-live-token-path.md`;
- Git blob `13f18f0afb6278d0783d410b5d11514355a4da99`;
- SHA-256
  `5e11f2f78b9a327955fde2b33bf3c9041063a8dfb0530522f42dbd0b42905427`.

Its modified status relative to the isolated base is not an authority-content
change. Return the base blob and successor blob separately so the report cannot
confuse worktree status with content identity.

## 1. Freeze the Revision-3 refusal

Revision 3 is `REVISE`. Its green helper tests did not close the public paths:

- `run_turn_core` bypasses `resolve_hide_journal_root` and creates
  `hide-gen-journals-turn-*` under the process temp directory;
- serve emits typed Done, then legacy stats, then `[DONE]`, while HIDE treats
  typed Done and legacy stats as duplicate terminals;
- `max_tokens=0` and other HTTP paths discard journal/fsync errors with
  `let _ =`;
- any plausible nonempty hash can authorize a HIDE Success and
  `agent.message` without opening the serve terminal chain;
- decode poison wires scheduler and `SystemPromptKvBank` but the engine call is
  a no-op;
- frame deserialization accepts unknown fields and omits mandatory
  model/executable/provider identities;
- the “fresh process” test runs `/usr/bin/true`, four public HTTP tests are
  ignored, and the HIDE SSE fixture does not match the live serve order;
- timing stops at the serve terminal and does not bind HIDE terminal or message
  persistence;
- the trailing-buffer SSE Done path does not run the same hash gate.

Do not weaken the contract, keep ignored tests, or label helper coverage as
public/fresh-process evidence.

## 2. Stable durable roots and exclusive ownership

Every production serve and HIDE path uses the common root resolvers. Missing,
implicit, process-temporary, PID/time/turn-scoped, symlinked, nonregular, or
untrusted roots refuse before Start and before any response/message.

The public `run_turn_core`, HTTP/SSE routes, scheduler/deferred tasks, recovery,
and supervisor receive the same explicit durable root and run/request/session
identity. A fresh process discovers the same incomplete chain and obtains
exclusive ownership; it cannot create a second success chain under another
root.

All create/write/truncate/rename/terminal operations check file fsync and parent
directory fsync. No production journal append, heal, abort, or terminal result
may be discarded with `let _ =`, ignored `Result`, best-effort logging, or an
empty terminal hash. Any failure creates a checked refusal/abort if possible
and emits no success-shaped transport or message.

## 3. Closed canonical frames

Serve and HIDE envelopes and payloads use recursively closed versioned schemas.
Rust types use `deny_unknown_fields`, but serde attributes alone are
insufficient: parse raw canonical JSON and refuse duplicate keys, nonfinite
numbers, bool-as-int, extra/missing/reordered fields, alternate spelling/case,
trailing streams/bytes, extra whitespace, and missing final delimiter.

Every frame binds as first-class nonoptional fields:

- magic, schema/version, kind, sequence, predecessor frame hash;
- run, request, session, slot, epoch, model, executable, provider, tokenizer,
  and build identities;
- canonical payload hash and frame hash;
- stream/token/output identity and exact terminal kind;
- serve terminal hash in the HIDE terminal.

Recompute every hash over the exact canonical domain. Refuse cross-identity,
reseeded/resealed, duplicate terminal, post-terminal token/stats, success after
abort/refusal, terminal without Start, corrupt middle/tail, and unknown states.
Only an incomplete physical tail may be truncated after the complete prefix is
validated and both truncation fsyncs succeed.

## 4. One live terminal protocol

Freeze one protocol shared by serve and HIDE:

1. zero or more typed token frames;
2. exactly one typed terminal frame containing terminal reason, final stats,
   output identity, and the already durable serve terminal hash;
3. optional bare `[DONE]` only as an inert transport sentinel.

No separate stats/token frame follows the typed terminal. `[DONE]` never
authorizes success and is ignored only after one valid typed terminal. A bare
or early `[DONE]`, legacy stats-as-terminal, duplicate typed terminal, token or
stats after terminal, channel close, client disconnect, deferred error, or
scheduler error becomes refusal/abort.

All streaming and buffered/trailing-buffer parser paths use the identical
schema, predecessor, identity, and terminal-hash gate. Delete or correct
fixtures that claim a different live serve order.

The durable serve terminal is fsynced before its typed SSE terminal is sent.
The matching durable HIDE terminal is fsynced before `agent.message` is
persisted.

## 5. Live serve-chain authorization

HIDE Success requires opening the explicit shared serve journal root,
validating the complete canonical serve chain through the public
`authorize_success_terminal` authority, and matching exact run/request/model/
executable/provider/output/terminal identities and hashes.

String prefixes, nonempty values, caller-supplied hashes, environment values,
reconstructed receipts, counters, or stubs are never authority. Remove or
hard-refuse `StubInferenceClient::with_serve_terminal` and any
production-shaped model-free/local/test success that persists
`agent.message`. Model-free and stub paths may test refusal only.

Revalidate the serve chain immediately before HIDE terminal append and again
before message persistence. Any TOCTOU, replacement, truncation, alternate
root, or cross-run terminal poisons/refuses with no message.

## 6. Complete slot/epoch poison

On decode, scheduler, transport, journal, or persistence failure, invoke a real
engine poison API for the exact slot/epoch; a downcast placeholder or
`let _ = engine.as_mut()` is forbidden.

Atomically poison/remove the slot/epoch from engine KV and sequence ownership,
scheduler ready/prefill/retry/deferred/free/active/copy/prefix indexes,
`SystemPromptKvBank`, prefix/KV caches, pending senders, transport tasks,
request evidence, journal ownership, and reusable handles. Close pending
senders with a typed error. Neighbor slots/epochs remain healthy. Reuse requires
checked device completion plus verified wipe/reset and a new epoch.

Exercise failure at every boundary and prove no partial residual, KV, sequence,
token, terminal, stats, trace, or message publication.

## 7. One monotonic causal receipt

Bind raw timestamps in one monotonic clock domain and exact derived arithmetic:

`enqueue <= prefill <= first decision <= every decode <= SSE enqueue <= SSE publish <= durable serve terminal <= durable HIDE terminal <= message persistence`.

Require finite, nonnegative, bounded phases and exact
`total_wall = named_disjoint + gaps` within the frozen tolerance. Missing or
zero phases never synthesize qualification. Include queue, prefill, decode
samples, transport, both file and directory fsyncs, HIDE validation/terminal,
and persistence. Execute publish-before-enqueue, `1e12`, cross-clock,
overlap/double-count, negative, NaN/infinity, and missing-phase attacks through
the public path.

## 8. Real public fresh-process matrix

Use real child processes and one explicit shared durable root. `/usr/bin/true`,
same-process reopen, module reload, private helper-only invocation,
source-string checks, ignored tests, or synthetic state labels are not
evidence.

For serve and HIDE, kill a production test hook before/after every write,
file-fsync, publish, directory-fsync, token, terminal, poison, and message
boundary. Fresh P1 recovers/refuses to the exact terminal; fresh P2 repeats
with byte/inode-identical journals and zero extra success/message events.

The public matrix includes deferred engine error, channel close, client
disconnect, scheduler error, fsync/write/permission failure, duplicate and
cross-identity frames, forged/early/bare Done, legacy stats, missing terminal,
model-free hash, serve-chain replacement, message ordering, every poison
surface, verified reset, and neighbor isolation.

Run all HTTP/SSE production tests with zero ignored/skipped/filtered cases.
Assert exact frame order, journal bytes, terminal/message state, child PIDs and
exit codes, and stable refusal reasons.

## 9. Exit

Return the exact successor manifest, every Git blob and SHA-256, raw public
fresh-process matrix, complete test counts including zero ignored, parser
mutation table, poison-surface table, and timing reconciliation. Keep real
weights, heavy benchmarks, MOP, `HIDE_KERNEL_TURN`, TG/HIDE promotion,
capable-provider status, and every authorization transition false/default-off.
