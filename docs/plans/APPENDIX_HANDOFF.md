# Appendix handoff and incorporation contract

This file is the shortest safe entry point for another Hawking work session.
The Appendix remains additive to the compression/Doctor ladder, and the live
ladder remains the primary corpus. Do not stop it, mutate its queue, or run GPU,
model, hashing, or high-bandwidth cells while a heavy owner is active.

The structural maximization phase is closed. Subsequent sessions should monitor
the live run and execute the incorporation order below at the release boundary;
they should not reopen a generic optimization audit or add new scaffolding
unless a specific required gate fails.

## One-command orientation

```sh
python3.12 tools/condense/appendix_handoff.py --audit
```

That command rebuilds the master plan, 25-sector catalog, speculative matrix,
and static TQ probe in memory; hashes every Appendix source file; and fails if a
file, dependency, safety rule, or fingerprint drifted. It reads no model artifact
and launches no inference.

The durable same-workspace packet is:

```text
reports/appendix/appendix_handoff.json
```

Verify it with:

```sh
python3.12 tools/condense/appendix_handoff.py \
  --verify reports/appendix/appendix_handoff.json
```

## What is already implemented

- Stored, compact-metadata, hashed-quantile, and computed-Acklam interpretations
  of identical TQ bytes, with CPU exactness gates, decode-only Metal oracles,
  single-vector fused kernels, and batch-major B=1..8 fused kernels.
- Orthogonal `TqRuntimeRecipe` accounting for metadata and codebook axes,
  including non-executable compact+hashed and compact+computed research cells.
- Machine-readable fused-GEMV admission reasons. The static census exposes that
  Qwen-0.5B FFN-up projections with `cols=896` need a ragged-row design or CPU
  fallback; no silent GPU claim is allowed.
- A fail-closed speculative re-entry matrix and measured-cost router core.
- A TQ-native Qwen batched projection route that follows the greedy ownership
  map, including RHT-cols, OUTL, residual passes, and strict all-linear proof
  coverage.
- A fail-closed post-run bridge that maps the useful existing vendor Metal gates
  without conflating them with Hawking-core artifact-bound receipts.
- Two compiled raw probes plus heavy-lease runners: one for artifact-bound TQ
  decode/GEMV evidence and one for non-skipping TQ verifier parity/cost evidence.
- An explicit deferred release-build gate for both probes (`cargo build --release
  -p hawking --features tq` with both probe binaries). It is part of the ordered
  post-run plan but is not run while Doctor or another heavy owner exists.
- A 25-sector catalog, strict canonical receipt validator, deterministic sample
  rollup, analytic TQ probe, evidence ledger, and post-run immutable corpus
  indexer. Corpus hashing/verification refuses while a heavy owner exists.

## Generated cheap evidence

`reports/appendix/tq_runtime_static_probe.json` contains the deterministic cross
product of 37 projection families spanning Q/K/V/O, FFN, and LM-head geometry
for current 0.5B-72B Qwen configurations, a GPT-OSS-120B projection proxy,
TQ1-TQ4, and seven current/future runtime recipes. Per-layer multiplicities also
produce model-level logical-byte rollups. Its paired `.receipt.json` binds the data to the source commit,
explicitly records that no model/prompt/device was consumed, and makes no speed
claim. These reports are deliberately outside the active Doctor result tree.

`reports/appendix/tq_runtime_device_matrix.json` is the derived execution queue:
496 implemented/eligible cells await the post-run lease, 96 implemented cells
are retained as geometry-blocked evidence, and 444 compound/repacked cells remain
design-deferred without fake runtime or receipt claims.

`reports/appendix/appendix_postrun_plan.json` connects that queue to speculative
re-entry. It maps six existing compile/vendor gates and preserves their limits:
self-contained or vendor shaders, synthetic/padded shapes, non-contract timing,
and no durable device receipt. It also maps both strict Hawking adapters. Their
source and host compilation are green; device execution and receipts remain
deferred while the corpus owns the machine.

## Incorporation order

1. Run the handoff audit and cheap tests from its packet.
2. Preserve all fail-closed defaults and the current four executable runtime
   names. The compound/repacked recipes are accounting cells, not kernels yet.
3. After the heavy run, freeze a hash-bound corpus index including negative
   evidence; verification rejects both changed files and additions after freeze.
4. Compile Metal and establish exact stored-path parity first; then compare the
   four implemented TQ modes and only afterward build compound/repacked paths.
5. Design the ragged-row fused path from the real tensor census instead of
   weakening admission.
6. Run the implemented TQ-native B=1..8 verifier probe and require its parity
   and physical-counter-bound cost receipts before enabling any proposer.
7. Write every result through `appendix_contract.py`; never promote from an
   unstamped ad-hoc benchmark.

The default-off physical release controller and collector status surfaces are:

```sh
python3.12 tools/condense/appendix_physical_release_state.py --status
python3.12 tools/condense/appendix_physical_counter_collector.py --status
python3.12 tools/condense/appendix_physical_counter_authority.py status
python3.12 tools/condense/appendix_physical_counter_executor.py --capability
python3.12 tools/condense/appendix_physical_counter_executor.py --status
python3.12 tools/condense/appendix_physical_counter_request.py status
python3.12 tools/condense/appendix_physical_release_packet.py status
python3.12 tools/condense/appendix_physical_release_packet.py dry-run assemble
python3.12 tools/condense/appendix_xctrace_export_adapter.py --status
```

Neither command opens the heavy lease or exposes collection. The release
controller remains at its staged CAS state until Doctor final readiness, zero
heavy owners, the durable audits, the frozen corpus, release-build receipts,
and the aggregate physical-evidence gate all pass. The collector emits only
authoritative phase-attributed v2 payloads; privilege, capability, and exact
wall/continuous capture receipts remain mandatory. Collection activation is a
separate release-gated executor; the collector's own
`collection_cli_exposed=false` invariant is not weakened.

The Metal counter boundary is deliberately two-stage. Full Xcode records one
raw `.trace` package under the inherited lease. The pinned export adapter then
requires an unexpired operator-SSHSIG profile, freezes and reopens the trace,
exports exact positional signpost/command-buffer/encoder/counter tables, joins
one begin/end signpost pair plus all Metal rows to every predeclared interval,
and deterministically rebuilds its canonical capture and receipt. Only that
verified canonical JSON enters `appendix_physical_counter_normalizer.py`; the
raw trace and any archive are never supplied as JSON. Synthetic profiles and
fixtures permanently receive zero physical credit.

The counter authority has an independent source-sealed trust root:

- `appendix_counter_authority_registry.json` pins the SSHSIG namespace,
  signer identity, allowed-signers byte hash, and Ed25519 public-key hash.
- `appendix_counter_authority_allowed_signers` contains only public material.
  An envelope cannot supply its own trust root.
- The matching operator private key remains at
  `~/.ssh/id_ed25519`, outside this repository, with mode `0600`. The status
  command verifies its public companion against the pinned key without reading
  private-key bytes. No private bytes are copied into a registry, request,
  receipt, handoff, or report.
- `appendix_physical_counter_authority.py receipt ...` constructs a canonical,
  live-host-bound receipt from typed flags. `... sign --receipt RECEIPT
  --private-key ~/.ssh/id_ed25519 --signature-output RECEIPT.sig
  --envelope-output ENVELOPE.json` first derives the supplied key's public key
  and refuses signing unless it exactly matches the pinned root.

After all ten capability/privilege/attribution envelopes exist, construct the
physical request through typed flags, not handwritten JSON:

```sh
python3.12 tools/condense/appendix_physical_counter_request.py \
  build-device --help
python3.12 tools/condense/appendix_physical_counter_request.py \
  build-spec --help
```

The builder requires the seven exact release-parent files plus ten
`KEY=ENVELOPE` authority arguments, measures workload file identities, checks
the live IOPlatformUUID hash, verifies every SSHSIG against the pinned root,
and writes one immutable request. It never opens the heavy lease or starts a
collector/probe. The subsequent executor invocation is intentionally explicit:

```sh
python3.12 tools/condense/appendix_physical_counter_executor.py \
  --execute REQUEST.json --acknowledge-request-sha256 REQUEST_SHA256
```

Before any collector starts, that command rechecks the final Doctor observer,
zero owners, green RAM/swap/thermal/disk, exact source/build/corpus parents, the
live host UUID, signed authority expiry, and an inherited canonical lease. A
separately opened nonblocking flock must fail, proving the inherited descriptor
already owns exclusivity. The Rust probes currently have no native barrier, so
the executor's child waits on a pipe and retains its PID across `execve`; the
full-Xcode Metal System Trace collector must be ready before the barrier byte is
released. Process energy is not a second external stream: the exact probe PID
self-samples Darwin `proc_pid_rusage(..., RUSAGE_INFO_V6)` immediately before
and after each operation, while its elapsed and continuous-clock interval covers
the operation only. The snapshots therefore bracket the measured operation
without charging the sampling syscalls to that operation. Missing full Xcode,
signed live libproc provenance, parser receipts, or a pinned trusted normalizer
returns exit 75 without starting a collector/probe. `powermetrics` energy-impact
is never accepted as joules.

The raw `.trace` package is not JSON and is never passed directly, or through an
archive, to the JSON normalizer. After recording, a pinned xctrace export adapter
must verify the full-Xcode binary/build/template identity, the immutable
operator-reviewed export profile, exact TOC/schema fingerprints and units, the
trace-tree identity, probe PID, run nonce, argv hash, Metal registry ID, and the
complete phase-marker manifest. It requires one exact begin/end signpost pair,
collision-free predeclared signpost IDs, and total command-buffer, encoder, and
counter-row consumption for every raw interval. It exports through fixed
no-shell `xctrace export` argv and aggregates all direct events into one canonical
direct-Metal record per exact trial marker.
Missing, duplicate, ambiguous, cross-interval, interpolated, apportioned, or
estimated events are rejected. The retained raw trace tree, TOC, every exported
table, profile, canonical JSON, and adapter receipt must all be sealed and bound
into the execution receipt. Synthetic export fixtures remain zero-credit.

The separately sealed owner-safe extension report is verified with:

```sh
python3.12 tools/condense/appendix_cheap_gates.py --verify-release-packet \
  reports/appendix/appendix_release_packet_cheap_gates.json
```

That exact six-gate extension runs only Python compilation, four default-off
builder/executor self-tests, and synthetic/temporary-directory tests. It explicitly
records no Cargo, model, GPU, active-corpus hashing, corpus mutation, or runtime-
default mutation. It is separate from the earlier Appendix report so the sealed
older receipt is not retroactively credited with newly added tests.

The release-packet builder is the only production path for the aggregate
physical packet. Its non-status commands acquire `reports/cron/studio_heavy.lock`
and recheck Doctor final readiness, zero heavy owners, and a green RAM/swap
sample under that lease. It then verifies every file identity recorded by the
Doctor final packet before issuing the release boundary. The exact source claim
is an isolated, symlink-free critical-source capsule, not a claim that unrelated
user work is absent: it binds the frozen required-path set, `Cargo.lock`, current
base commit, and stable file bytes, and requires the identical capsule before and
after the locked two-probe release build. It never stages, commits, cleans, or
otherwise mutates the user's Git state.

The release build does not trust a PATH-level `cargo` name or a pre-existing
`target/release` binary. It binds the resolved Cargo and Rustc bytes, their
verbose version output, Rustc host triple, `Cargo.lock`, and an exact sanitized
environment. `CARGO_TARGET_DIR` is a new directory derived from the exact
release-boundary attestation and critical-source capsule; any existing target
at that identity is rejected. Wrapper and Rust flags are cleared, Rustc is
pinned explicitly, and Cargo is invoked for the bound host target with JSON
compiler-artifact messages. The receipt accepts only the two executable paths
named by that invocation under the unique target and binds each compiler
dep-info file plus its hashed source closure. Both compiler-artifact messages
must report `fresh=false`; a cache hit is rejected instead of being relabeled as
a release build. No destructive `cargo clean` is performed automatically.

After the release boundary opens, the production order is:

```sh
python3.12 tools/condense/appendix_physical_release_packet.py prepare-release \
  --root reports/condense/doctor_v5_ultra/results \
  --output-dir reports/appendix/physical_release/prepared/GENERATION
```

`prepare-release` is the only release-parent issuance path. One held lease and
one boundary/observation bind the exact source capsule, the complete corpus
index, a pre-build corpus verification, a fresh unique-target release build,
and a post-build re-verification of the same index. The corpus semantic census
retains explicit negative, failure, and partial evidence truthfully; zero such
items is valid and is recorded as zero rather than synthesized. All JSON parents
and the build log are installed as one immutable group only after every phase is
green. Standalone boundary, capsule, build, and freeze issuance is disabled.

Evidence manifests and `assemble` follow only after the device and speculative
runners have produced their immutable per-cell files. Assembly must receive
both the prepared pre-build receipt and its exact post-build child. It sorts
cells deterministically, rejects reused evidence bytes, validates all parents
and physical counter files, and writes nothing unless
`appendix_physical_evidence_gate` is fully green.

The final ten-facet Doctor aggregate is a separate operator boundary. A raw or
merely self-hashed Doctor packet always scores zero. After all exact facet and
Appendix children validate, use the explicit draft -> SSHSIG sign -> seal ->
verify workflow:

```sh
python3.12 tools/condense/doctor_v5_physical_result_authority.py \
  --draft-result CORE_PACKET.json --output RESULT_DRAFT.json
python3.12 tools/condense/doctor_v5_physical_result_authority.py \
  --sign-result RESULT_DRAFT.json --private-key ~/.ssh/id_ed25519 \
  --signature-output RESULT.sig --envelope-output RESULT_ENVELOPE.json
python3.12 tools/condense/doctor_v5_physical_result_authority.py \
  --seal-result RESULT_ENVELOPE.json --packet CORE_PACKET.json \
  --output OPERATOR_SEALED_PACKET.json
python3.12 tools/condense/doctor_v5_physical_result_authority.py \
  --verify OPERATOR_SEALED_PACKET.json
```

The verifier pins the signer, result-only namespace, trust root, expiry, exact
packet/plan/source/release-boundary hashes, and ten distinct facet receipt
hashes in source. The controller unwraps only this verified sealed form before
running its deeper physical validators.

The first post-run status command is:

```sh
python3.12 tools/condense/appendix_postrun.py --status
```

It does not execute a gate. Exit 75 means an owner still exists or a required
release probe is absent. Hawking uses runtime Metal source compilation; the
optional offline `xcrun metal` utility is reported but is not an execution
prerequisite.

The owner inventory is broad and fail closed. It shares the trusted local
Doctor patterns and explicitly includes the detached Doctor supervisor and
children, MOP generation campaign, cognitive-corpus workers, quantizers, and
Appendix probes, plus narrowly identified vLLM-Metal paths and vLLM server
entrypoints. Generic Python or MLX processes are not guessed to be heavy.
Failure of the canonical `/bin/ps` probe is itself represented
as an owner, so a missing process inventory can never open the device gate.

## Deferred means unmeasured

Runtime Metal compilation, device parity, occupancy, realized bandwidth, energy,
thermals, native TQ batched-verifier parity, learned draft training, and
end-to-end speed remain deferred. The local command-line environment cannot
locate the optional offline `metal` compiler, and the active corpus still owns
the machine. Neither condition weakens the eventual gates.
