# Gravity Rust repatriation

Hawking's release-facing Gravity path is Rust-first. Python remains useful for
cheap science, data exploration, and one-off discriminators, but it is not the
long-term owner of deterministic artifact admission, representation execution,
or performance-critical scheduling.

## Hard migration budget

The refreshed September 11 source census found 486,424 Rust lines in 730 files
and 565,605 Python lines in 1,444 files (ignored files excluded). Rust is
therefore 46.237% of the measured source. Reaching the required 75% at constant
total source requires migrating at least 302,598 Python lines into verified Rust
owners. The `hcli/` and `tools/` pool alone contains 459,928 lines, so the target
is attainable without counting documentation or pretending thin Python wrappers
are a migration.

Odyssey is active again while the first production Python owners continue to
move through persistent native services. Rust repatriation is a parallel
measured front, not a reason to suspend the ModelLake campaign. The ledger is
`receipts/future/GRAVITY_RUST_REPATRIATION_BASELINE_20260911.json`.

## Current foundation

`hawking-core` now owns a growing set of generic deterministic primitives:

- `gravity_manifest`: complete persisted/active byte accounting that refuses
  unattested payload-only EBPW and turns a closed representation into policy
  input.
- `gravity_policy`: integer, input-order-independent, fail-closed selection of
  measured representation candidates.
- `safetensors_inventory`: header-only source inventory that validates shard,
  tensor, dtype, shape, offsets, and exact byte closure without reading payloads.

The latter has closed the 131-shard Flash-Next source exactly. Its command-line
surface is `gravity_safetensors_inventory`; it is accounting/admission support,
not execution.

`gravity_repo_search` is the first high-latency Python replacement candidate:
it keeps a bounded active-code text index resident. A real Hawking-tree probe
has measured a 79,667 ns median identifier query after index construction. Its
5,998,278,292 ns cold build is intentionally not a tool-call path: `hawkingd` must own
asynchronous build/rebuild before HCLI routes requests to it.

The bridge now exists as `hawking-gravityd`, a narrow JSONL child service with
health, bounded search/list, catalog, tool-catalog, tool-dispatch-admission,
rebuild, and shutdown operations. `hcli.repo_context` uses one such child when
the daemon sets `HCLI_NATIVE_GRAVITY=1`. A real Python-to-
native round trip measured 278,208 ns median (441,625 ns p95) after initial build.
The release binaries are content-addressed and bundled beside the stamped HCLI
package: both `hawking-gravityd` and the Rust `hcli-rust` process authority are
part of the immutable snapshot. `hcli serve` prewarms its supervised child while the resident body is
loading, so the first retrieval does not intentionally trigger the multi-second
index build. The currently running resident is not silently restarted; the
next controlled `hawkingd` start adopts this snapshot and its direct native
child.

`model_lake_catalog` is a native query owner. It reads only the durable
catalog, validates its schema and every queried slug, and never falls back to a
ModelLake walk or payload read. The resident `catalog` operation in
`hawking-gravityd` holds that catalog in memory: a real 1,000-query release
probe measured a 250 ns median in-process lookup. This measures catalog
lookup, not process startup, catalog construction, source admission, or model
execution.

HCLI's admitted `lake.catalog` tool now routes to that native owner whenever
the daemon has its Gravity child. Its compatibility fallback reads the same
catalog file and refuses schema mismatch; neither lane is allowed to replace a
missing catalog with a ModelLake walk. This is deliberate runtime adoption,
while `lake.census` remains a separate expensive science operation.

`gravity_tool_catalog` is an adopted hot primitive. HCLI supplies one
immutable snapshot of canonical tool names and search text; the native child
returns ranked canonical names only. Python retains schemas, permission
checks, aliases, and dispatch. A real four-focus parity probe matched the
legacy ranking, while 1,000 resident native rankings measured a 12,000 ns
median. This is tool identification only, not tool execution.

`gravity_tool_dispatch` now owns an earlier, fail-closed admission fence for
those same immutable tool contracts. It validates canonical name, declared
mutation class, and granted permission before Python performs schema validation
or invokes a handler. A native denial is final; unavailable native service
preserves the existing Python check. The 1,000-call resident probe measured a
334 ns median admission check. This is deliberately not a claim about handler
execution or JSONL transport; it establishes a fast deterministic boundary
that prevents an unadmitted mutation from reaching either.

The resident bridge retains that immutable snapshot by identity, so normal
invocations do not rebuild or re-hash every registered contract. A 1,000-call
warm read-only fixture measured 106,125 ns median and 150,042 ns p95 for the
complete Python registry invocation, including native JSONL admission, Python
schema validation, and the trivial handler. This is the real current control
plane floor; work that materially exceeds it should be profiled before being
ported or fused.

Filesystem discovery now has the same native boundary. `fs.list` performs
contained, deterministic discovery in Rust, keeps listing separate from file
content, and preserves the bounded Python path as a differential fallback.
Against the current Hawking checkout, a 1,000-query warm `*.py` probe measured
122,583 ns median and 143,291 ns p95 from Python through the resident JSONL
child; the native operation reported 95,520 ns median. Its first call in that
probe was 1,413,819,500 ns wall time because a fresh child also had to build
the resident text index; this startup cost remains an asynchronous/prewarm
obligation, not a per-list latency claim. The old Python recursive-list
baseline was approximately 28.1 s for the unbounded path; the native default
is one level and content-free.

`fs.search` now uses the resident index for contained recursive text search
when the native owner is available. The query preserves case-sensitive line
matching, basename globbing, result caps, and per-file caps; Python remains
the fallback for an unavailable native owner or a non-contained read root. A
1,000-call warm `ToolRegistry`-independent probe for `ToolRegistry` in `*.py`
measured 392,458 ns median and 442,833 ns p95 before freshness validation; the
native operation reported 368,208 ns median. With the source-freshness guard
enabled, the current 200-call warm probe measured 8,296,937 ns median and
8,687,375 ns p95 from Python through JSONL; the native operation reported
8,241,249 ns median. Its fresh-child first call was 1,456,532,375 ns wall time
including index construction. The historical Python whole-tree search baseline
was approximately 23.5 s. A later release probe narrowed validation to the
requested scope and matching glob, then used bounded parallel metadata checks:
2,065,833 ns median and 2,299,667 ns p95 Python through JSONL, with
1,908,354 ns native-reported median and 2,107,417 ns p95. Its fresh-child first
call was 1,427,927,167 ns wall time including index construction. Directory
metadata still covers additions/removals while matching file metadata covers
content changes; unrelated file edits do not force a rebuild. This is a
measured roughly 75% reduction from the first honest freshness-validated path,
while retaining current-source evidence rather than caching a stale answer.

The complete `ToolRegistry.invoke("fs.search", ...)` path was then measured
warm at 523,708 ns median and 572,833 ns p95 before freshness validation. Under
the current guard, the complete path measured 8,488,208 ns median and
8,871,916 ns p95 for the same two-result, one-hit-per-file query; its native
operation reported 8,099,916 ns median. Its first-call wall cost was
1,305,292,667 ns before the guarded rebuild path was measured. The current
release path, after scoped parallel validation, measured 2,332,291 ns median
and 2,535,875 ns p95; the native operation reported 1,901,896 ns median and
2,057,292 ns p95, with a 1,299,780,500 ns fresh-child first call. These are
control-plane plus read-only handler measurements, not model latency.

Repository path orientation now has a resident path-only index containing
10,615 candidate files. A 200-call release probe for the real path-shaped
question `compare hcli/serve.py and stream handling` uses an exact-path fast
lane: the native operation reported 394,375 ns median and 409,166 ns p95, with
495,875 ns median and 525,709 ns p95 through Python and JSONL. `RepoContext`
path selection measured 484,729 ns median and 501,708 ns p95; the complete
`volatile_block` path, including the selected file read, measured 517,791 ns
median and 532,208 ns p95. Every result was `hcli/serve.py`, `complete=true`,
and `files_seen=10,615`. The prior live-tree native rank was 72,434,791 ns
median round trip, so the accepted resident/exact-path route is approximately
99.3% lower on this repeated query. Its first cold request was 1,478,874,666 ns
wall time while the child built its resident text/path indexes; that startup
work remains a prewarm/lifecycle obligation and is not a hot-query claim.
The exact lane validates the named relative file with containment and regular
file checks; non-exact path queries still validate indexed directory metadata
before ranking, and content-oriented questions retain the existing fallback.

Content-oriented orientation now has a single native multi-term request. On
the same objective query, the native operation measured 2,296,688 ns median and
2,362,667 ns p95; Python through JSONL measured 2,394,312 ns median and
2,469,416 ns p95; and `RepoContext.paths_for` measured 2,449,312 ns median and
2,720,875 ns p95. The first cold request was 1,479,569,833 ns while the child
built its indexes. This replaces the prior 583,104,917 ns warm path, a roughly
99.6% reduction. The native owner uses the resident exact-token index and the
same bounded rare-term weighting, HCLI scope, and test bias; the `rg`
compatibility fallback remains available when native Gravity is unavailable
and may differ in lower-ranked paths because it uses substring matching.

The content scorer then received measured hot-path reductions. The previous
release implementation cloned and compared `PathBuf` values while accumulating
each term's score; the first repair scored stable file indexes and cloned paths
only for the bounded final response. On the same resident release service, 200
calls using the six-term `hcli/` probe measured 1,581,896 ns median and
1,609,792 ns p95 before that change, versus 394,917 ns median and 408,291 ns p95:
a 75.04% native ranking reduction. A second repair replaced the ordered map
with a dense score vector plus a touched-index list, preserving the final sort
and scope fallback. The fresh release probe measured 367,250 ns median and
403,333 ns p95, a further 7.01% median reduction. The Python/JSONL adapter
measured 461,625 ns median and 505,708 ns p95 in the same probe. This is bounded
resident repository ranking only; it excludes index construction, freshness
scans, file reads, and model execution. The parity suite remained green with
24 focused tests.

A third scorer pass changed the resident exact-term index from an ordered map to
hash lookup and precomputed stable relative path text for scope/tie handling.
The same 200-call release probe measured 233,687 ns native median and 239,208 ns
p95, versus 367,250 ns and 403,333 ns: a further 36.37% median and 40.69% p95
reduction. Python through JSONL measured 325,812 ns median and 335,584 ns p95,
down 29.42% and 33.64% from the prior adapter probe. Result paths, six-term
scope behavior, and deterministic final ordering remained unchanged. This is
still ranking bookkeeping only; it does not claim model or end-to-end HCLI TPS.

A fourth scorer pass reuses one bounded match buffer across terms and avoids
allocating the full candidate-index range on the non-indexable fallback. Three
fresh sequential 200-call release probes measured a median-of-run-medians of
219,750 ns native and 225,000 ns p95, down 5.96% and 5.94% from the third pass.
The end-to-end Python/JSONL adapter measured 233,458 ns median and 240,208 ns
p95 in the same three-run summary. The six-term result paths and deterministic
ordering remained unchanged. This remains ranking bookkeeping only and does
not claim a model, tool-body, or end-to-end HCLI TPS improvement.

A fifth scorer pass precomputes the immutable context-code and test-path flags
at index-build time, removing repeated extension and filename classification
from every ranked query. Three fresh sequential 200-call release probes
measured a median-of-run-medians of 23,292 ns native and 23,625 ns p95, down
89.40% and 89.50% from the fourth pass. The end-to-end Python/JSONL adapter
measured 36,000 ns median and 39,584 ns p95, down 84.58% and 83.52% from the
fourth adapter probe. Result paths and deterministic ordering remained
unchanged. This is still a repository-ranking bookkeeping result, not a model
or end-to-end HCLI TPS claim.

A sixth scorer pass keeps a daemon-owned scratch arena with generation marks:
the score, touched-index, match, and selected-result buffers are reused across
queries, while stale scores are invalidated by epoch rather than by clearing a
full file-count vector. The same six-term query and three fresh sequential
200-call release probes measured a median-of-run-medians of 21,792 ns native
and 22,125 ns p95, down 6.44% and 6.35% from the fifth pass. The four result
paths, 1,563 files seen, scope behavior, and deterministic ordering remained
unchanged. The scratch lock is explicit because the current JSONL child is
request-serial; it preserves a safe boundary if a future transport adds
concurrency. This remains a repository-ranking bookkeeping result, not a
model or end-to-end HCLI TPS claim.

A seventh scorer pass adds a second immutable exact-term index containing only
context-code files. This removes the per-hit context classification branch from
the hot identifier-term lane while preserving the general search index for
ordinary substring/search semantics. Three fresh sequential 200-call probes of
the scoped six-term query measured a median-of-run-medians of 20,250 ns and a
median p95 of 20,625 ns, down 7.07% and 6.78% from the reusable-scratch pass.
The four `hcli/` result paths, 1,563-file census, scope fallback, and ordering
remained identical. A follow-up direct-indexing/bounds-check candidate measured
20,542 ns median and 20,959 ns p95 and was rejected as a regression against
that baseline. The accepted change is still ranking bookkeeping only; it does
not claim model, tool-body, or end-to-end HCLI TPS improvement.

An eighth scorer pass makes the scoped hot path rank only its bounded prefix:
for the normal four-result request, `select_nth_unstable_by` partitions the
scoped candidates and a deterministic final sort orders those four rows. A
full sort remains for callers whose requested limit covers the candidate set.
Three fresh sequential 200-call release probes measured 11,500 ns native
median and 11,750 ns p95, while the complete JSONL round trip measured 19,604.5
ns median and 23,417 ns p95. The same four paths, 1,563-file census, scope
fallback, and ordering remained unchanged. This is still repository-ranking
bookkeeping; it is not a model or end-to-end HCLI TPS claim.

An attempted lowercase-allocation removal was rejected: five fresh probes
measured 11,459 ns native median and 11,750 ns p95, statistically flat to the
accepted top-K pass and slightly worse at the tail. The source was reverted.

A ninth pass removes the intermediate `serde_json::Value` from the native
`rank_content_paths` response path and serializes the validated response
struct directly. Five fresh sequential 200-call release probes kept the
internal scorer at 11,459 ns median / 11,750 ns p95 and measured the complete
JSONL round trip at 19,042 ns median / 22,500 ns p95, a 2.87% wall-median and
3.91% wall-p95 reduction against the eighth-pass round-trip baseline. Result
paths, fields, schema, and failure behavior remained unchanged. This is a
transport allocation reduction only; it does not claim model, tool-body, or
end-to-end HCLI TPS improvement.

A tenth scorer pass borrows the immutable posting lists for indexable terms
instead of copying up to 1,000 file indexes into scratch before immediately
iterating them. The non-indexable substring path still uses the reusable
buffer. Five fresh sequential 200-call release probes measured 10,104.5 ns
native median / 10,375 ns p95 and 17,833 ns complete JSONL median / 20,375 ns
p95, reductions of 11.82%, 11.70%, 6.35%, and 9.44% against the ninth-pass
baselines respectively. The same result paths, 1,563-file census, scope
fallback, and deterministic ordering remained unchanged. This is repository
ranking and transport bookkeeping only; it is not a model or end-to-end HCLI
TPS claim.

An attempted Python byte-mode JSONL bridge was rejected after five adapter
probes rose to 153,208 ns median and 165,250 ns p95 from the prior approximately
97,000 ns adapter median. Native result parity held, but the bridge was slower;
the source was reverted and the shipped adapter remains text-mode JSONL.

The next adapter pass keeps text-mode pipes but uses compact JSON separators
for the small request envelope. Five fresh sequential 200-call adapter probes
measured 95,791.5 ns median / 101,750 ns p95, down 1.67% / 3.13% from the
restored text-mode baseline, with the same native result paths and validation.
This is a transport-envelope reduction only; it does not claim model or
end-to-end HCLI TPS.

The following adapter pass keeps the native boundary fail-closed but validates
the native owner's canonical relative code paths with string operations rather
than constructing a `Path` for every returned row. Absolute paths, traversal,
backslash-separated paths, dot segments, empty names, and non-code suffixes
are rejected; valid native paths are returned unchanged. Five fresh sequential
200-call `RepoContext.paths_for` probes measured 123,791.5 ns median / 130,750
ns p95, down from 128,145.5 ns / 131,000 ns on the same question and resident
protocol. The focused retrieval suite passed 26 tests including explicit malformed
path cases. This is a Python boundary-validation reduction only; it does not
claim model or end-to-end HCLI TPS.

The next adapter pass caches successful `hawking-gravityd` binary resolution
by repository root and explicit override, rechecks a cached executable before
reuse, and does not cache negative results. Binary discovery had measured at
about 70,750 ns per call and was dominating the native adapter. Five fresh
sequential 200-call probes using the exact six-term resident query measured
35,084 ns adapter median / 39,125 ns p95, down from 95,791.5 ns / 101,750 ns.
The complete `RepoContext.paths_for` probe measured 54,145.5 ns median /
60,500 ns p95, down from 128,145.5 ns / 131,000 ns on the same question;
result paths and safety checks remained unchanged. This is a binary-resolution
and adapter-boundary reduction only; it does not claim model or end-to-end HCLI
TPS.

The native `fs.search` freshness path now reuses the already computed,
scope-and-glob-filtered candidate file indexes for the coherent generation
instead of recomputing that scan inside the search helper. A rebuild still
publishes a new index and re-queries it, preserving the source-freshness
boundary. Because `fs.search` accepts arbitrary case-sensitive substrings, the
resident index uses a three-byte token-gram candidate superset for
identifier-like needles (and keeps the bounded full scan for shorter or
punctuated needles); final line matching remains unchanged. Five fresh
sequential 100-call probes of the contained `hawkingd` / `*.py` query measured
969,270.5 ns median / 994,958 ns p95, down from 1,042,937.5 ns / 1,059,375 ns:
7.06% median and 6.08% p95 reduction. The query returned the same bounded
20-match prefix, `files_seen=448`, and `truncated=true`; the focused native
search suite passed 15 tests. This is a resident filesystem-search
bookkeeping reduction only; it does not claim model or end-to-end HCLI TPS.

The repeated scope/glob scan is now cached by the resident service using the
index generation, relative scope, and glob as the key. Rebuild publishes the
new index and clears the cache while holding the same service write boundary,
so a source edit cannot reuse candidate indexes from an older generation. On
the same five fresh sequential 100-call probes, this reduced the complete
Python-to-native `fs.search` path to 896,354 ns median / 914,750 ns p95. That
is a further 7.52% / 8.06% reduction from the trigram-prefilter candidate and
14.05% / 13.65% below the original 1,042,937.5 ns / 1,059,375 ns control.
The installed `build-20260911-154929` snapshot measured 889,271 ns median /
967,125 ns p95 in the same probe, or 14.73% / 8.71% below that original
control. Freshness metadata validation remains on every request; this is a
generation-safe candidate-discovery reduction only, not a model or end-to-end
HCLI TPS claim.

The next physical pass kept the positive native dispatch admission cache and
lowered the metadata-validation serial threshold from 512 to 128 stamps, then
used at most eight bounded workers with 128 stamps per worker. All workers
must agree, so the freshness result remains deterministic and fail-closed. A
fresh 10×100 release probe of the full `ToolRegistry.invoke/fs.search` path
measured 967,156.5 ns median-of-run-medians and 1,002,683.65 ns p95, versus
1,232,875 ns / 1,261,429.15 ns before the two changes: 21.55% / 20.51% lower.
The resident Rust operation itself measured 619,666.5 ns median-of-run-medians
and 644,877.1 ns p95; the Python-to-native adapter wall was 696,197.75 ns /
724,237.15 ns. The same 20-match truncated result and focused 749-pass Rust
core suite were preserved. This is bounded filesystem freshness and dispatch
transport work only; it is not a model or end-to-end HCLI TPS claim.

The accepted source was packaged as the immutable
`/Users/scammermike/.local/share/hcli/build-20260911-164607` snapshot with
HCLI digest `1b5525ba2eb37e9adbdf6f3d8bb1dc68c98efb61365f0bba1731c79030dc1c13`
and bundled `hawking-gravityd` digest
`1ef21197043fc804f568c05e0b83a32b06de8cac7f08740b38948e8721893c35`. The
live daemon was deliberately not restarted, so this is packaged adoption—not
yet live-serving adoption.

The Python native connector and resident gate now use the same ns-first rule:
each duration is sampled once from `perf_counter_ns`, canonical receipts carry
`timing_unit: "ns"` with `wall_ns`/`elapsed_ns`, and `wall_ms`,
`generation_wall_s`, or `elapsed_s` are compatibility projections only. The
connector contract suite passed 38 tests; this improves measurement fidelity
without claiming a model-TPS gain.

The native qualification ladder now follows the same rule for its stage and
total durations: `elapsed_ns` is canonical, `timing_unit` is explicit, and the
older `elapsed_s` stage view is derived only for compatibility. A separate
read-mostly adapter pass removes repeated executable validation and creation
lock acquisition once the Rust service is resident. On the same eight-term
`rank_content_paths` request, an eight-probe A/B harness measured 104,291 ns
median / 112,125 ns p95 for the prior lookup path and 24,781 ns / 27,416 ns
for the new path: 76.24% median and 75.55% p95 lower. The native result,
generation, dead-child fallback, and prewarm recovery behavior remained
unchanged. This is recorded in
`receipts/future/GRAVITY_RUST_NATIVE_SERVICE_LOOKUP_NS_20260911.json`; it is
repository adapter latency only, not a model-TPS or end-to-end HCLI claim.

The accelerator regression audit now follows the same ns-first boundary at both
levels: the one live request records `wall_ns`, and the whole audit records
`run_started_ns` plus `elapsed_ns`; Unix start/finish fields remain provenance
and `elapsed_s` is derived only for compatibility. The change is packaged in
`/Users/scammermike/.local/share/hcli/build-20260911-165131` and recorded in
`receipts/future/GRAVITY_RUST_ACCELERATOR_REGRESSION_NS_20260911.json`. This is
measurement precision only; no model-TPS, capability, or Odyssey claim is
attached.

An attempted native `fs.read` migration was measured and withheld from the
ordinary production path. For `hcli/serve.py` (66,674 bytes), the Python
handler alone measured 108,729 ns median and 118,250 ns p95; the equivalent
Rust core read reported about 80,000 ns internally, but returning the body over
the JSONL boundary made the native handler 466,666 ns median and 495,959 ns
p95. At the complete ToolRegistry boundary, Python fallback measured 2,935,458
ns median and 3,013,959 ns p95, while native admission plus body transport
measured 3,354,354 ns median and 3,420,958 ns p95. The native body route is
therefore explicitly lab-only (`HCLI_NATIVE_GRAVITY_READ=1`) until a direct or
shared-buffer ABI removes the serialization cost. The Rust reader remains
tested as a future execution primitive; the default HCLI path does not regress.

Process observation is now a resident Rust service as well. The canonical
classifier compiles its role patterns once per service lifetime instead of once
per host row, and the default physical-footprint path batches all classified
PIDs into one `footprint` invocation. The Python compatibility skin creates the
service only when `hawkingd` sets `HCLI_NATIVE_PROCESS_SERVER=1`; standalone
callers keep the one-shot fallback, and the server remains a supervised child
under the sovereign `hawkingd` root.

Against the live host's four classified Hawking processes, the uncached
release-binary baseline measured 28,282,125 ns median and 29,053,500 ns p95
for warm no-footprint observations; the warm footprint path measured
107,371,167 ns median and 107,523,167 ns p95. The daemon-owned server now
keeps a bounded 250,000,000 ns diagnostic snapshot cache for repeated polls.
The cached probe measured 24,709 ns median and 40,125 ns p95 without footprint,
and 21,000 ns median and 26,542 ns p95 with footprint. Fresh cache fills were
293,952,375 ns and 111,530,208 ns respectively, including server startup and
host inspection. Expiry at 250 ms was verified; `orphaned` and `reap` bypass
the cache and inspect afresh. Compared with the prior one-shot debug median of
2,938,228,604 ns, this is a measured diagnostic-poll reduction, not a model
TPS claim. Measurements are in nanoseconds throughout; no one-PID fiction is
introduced: one sovereign process tree remains the invariant, not one literal
OS PID.

The live loopback HTTP control surface is already below 1 ms when warm: twelve
`/health` requests measured 387,250 ns median and 474,833 ns p95, while twelve
`/v1/models` requests measured 633,083 ns median and 677,500 ns p95. These are
control-plane measurements only; chat completion, provider decode, tool
execution, and model TPS remain separate obligations.

The live Engine telemetry and resident report benchmark now use the same
nanosecond authority. `hcli.latency` measures durations with a monotonic clock;
new model-call, heartbeat, tool-call, executor-wrapper, ToolResult, and
`hcli report` records write integer `elapsed_ns`/`wall_ns` fields. Unix `started_at` and
`finished_at` values remain wall-clock provenance timestamps. Old `elapsed_s`
and `wall_s` values are accepted only as derived compatibility reads, so the
existing event stream, cycle verdict, and report consumers continue to read
historical receipts without making seconds the new write format. The report's
TPS and prefix-reuse calculations now operate on nanoseconds before any
human-readable conversion. This is a measurement-integrity change; it is not
itself a model-TPS claim.

The shared Rust `hawking-core::startup_timing` owner now stores phase and
process durations as integer nanoseconds and emits `hawking.startup_timing.v2`
with `timing_unit: "ns"` and `elapsed_ns`. Its old `time_ms*`, `record_ms`, and
`duration_ms` names remain compatibility entry points, but timed closures no
longer round sub-millisecond work away. Metal library/pipeline timings and the
activation-weighted payload buckets now call the ns APIs directly. The remaining
active Hawking-core admission, streamed-forward, token-graph, and complete-
binary timing call sites were migrated to the explicit `time_ns*` APIs as well;
derived milliseconds remain presentation aliases only. The full Hawking-core
library regression suite passed 748 tests with 8 ignored and 0 failures after
the migration.

The inference `hawking-core::GenStats` boundary is now ns-first too. Native
Llama, Qwen, DeepSeek, Mixtral, RWKV, and Gravity engine producers sample
prefill, aggregate decode, and measured per-forward durations once as integer
nanoseconds. `stats_json`, CLI receipts, and benchmark suites expose
`timing_unit: "ns"`, `prefill_ns`, `decode_ns`, and `decode_token_ns`; the old
millisecond values remain derived compatibility projections. Complete-forward
TPS calculations use the integer decode duration, so sub-millisecond and
fractional-millisecond runs are not rounded before throughput is computed.
This changes measurement authority and precision; it is not itself a claim of
model-TPS acceleration.

The reachability index's active collection receipt is ns-first as well.
`IndexStats.elapsed_ns` is sampled once from the monotonic clock and the old
`elapsed_ms` field is derived from that integer value for compatibility, so a
sub-millisecond parse/reuse pass is no longer represented as zero or timed
twice. The reachability crate regression suite passed 78 tests with no
failures.

VMCP's active PTY and local tool-doctor receipts now emit `timing_unit: "ns"`
with integer `performance_ns`/`elapsed_ns` fields and PTY event boundaries use
`t_ns`. Old millisecond input remains accepted only at the receipt adapter
compatibility boundary; new producers use the shared monotonic nanosecond
authority. Refusal, network, dangerous-command, and real-subprocess behavior
were unchanged, and the focused VMCP suite passed 14 tests.

The HCLI Gravity gauntlet now follows the same duration authority. Candidate
observations and resumable search rows write integer `wall_ns` fields, while
legacy `wall_s`/`verification_wall_s` receipts are converted on read. Resume
normalizes historical cost totals once; each subsequent candidate adds its
measured cost instead of rescanning all prior observations. Atomic full-state
checkpointing remains deliberate for crash recovery, so only the in-memory
bookkeeping scan was removed.

The active public connectivity probe is ns-first as well: it writes integer
monotonic `elapsed_ns` plus `timing_unit: "ns"` and no longer makes a new
`elapsed_ms` measurement. This is a telemetry-contract correction, not a claim
about network or model latency.

## Migration rule

Port a Python path only when it has all three:

1. a stable mathematical/behavioral contract;
2. a small regression fixture or oracle; and
3. a measured reason to put it on a production or repeated experiment path.

The Rust implementation then becomes the canonical owner. The Python version
is retained only as a differential oracle until parity is demonstrated, then
classified as historical/research-only rather than kept as a second runtime.

## Ordered work

1. Artifact inventory and complete-byte accounting: Rust canonical owner.
2. Representation manifest, cost ledger, and noetic decoder ABI: Rust canonical
   owner, including every persistent and active byte.
3. Stable representation codecs and residual allocation: Rust reference path
   plus parity fixtures; Python stays for hypothesis fanout only.
4. Routing/layout, direct compact decode, and Metal kernels: Rust/Metal hot
   path, promoted only with end-to-end latency and capability evidence.
5. `hawkingd` supervision and worker lifecycle: a compact Rust control-plane
   core may replace repeated Python process management after the existing
   process-tree behavior has parity tests. HCLI/Web remains an adapter rather
   than a second daemon owner.
6. Runtime telemetry and benchmark timing: integer nanoseconds are canonical;
   historical seconds fields remain read-only compatibility data.
7. Model-specific Python experiments: migrate prospectively as their results
   become reusable Gravity machinery. Do not mechanically port dead probes.

## Acceptance law

No port is considered a speed improvement merely because it is Rust. Every
hot-path migration must record correctness/parity, complete byte accounting,
and the relevant wall-time measurement. Release policy remains dynamic through
explicit inputs, but every selected artifact must still be direct,
source-independent, and verified when those policy requirements are enabled.
