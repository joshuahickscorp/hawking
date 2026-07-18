# Doctor V5 unbound GPT-OSS and higher-tier acceleration handoff

The consolidated single-device CPU/PGO/I/O/elastic/cache/Metal sprint contract
is in `docs/plans/DOCTOR_V5_SINGLE_DEVICE_SPRINT.md`.

The GPT-OSS/higher-tier scaffold remains separate from the live 320-cell Doctor
campaign and declares execution false. The Qwen campaign now has a separately
audited pending-only acceleration generation; it did not change the campaign
plan, completed evidence, runtime defaults, or parent sources.

## Active pending-Qwen acceleration

Generation `808d8533d8658222d2f730fd540cc61b023abacac6a6beb131a0153ec3d4c34d`
promoted 192 nonterminal Qwen specs to the frozen block-parallel binary while
sealing all 88 terminal cells. Exact gates include the full 10-rate x 4-branch
configuration matrix and a real 4,194,304-weight BF16 Qwen tensor canary. The
real canary was byte-identical, measured 3.491x end-to-end speedup, and peaked
at 16,016,097,280 bytes RSS. The detached supervisor, source seals, active
marker, LaunchAgent, and rollback generation are hash-bound.

Do not copy a point-in-time ETA into this handoff. Regenerate the evidence-bound
projection from the live production logs with
`tools/condense/doctor_v5_production_eta.py` and read
`reports/condense/doctor_v5_ultra/staged_acceleration/production_calibrated_eta.json`.
The through-120B and Appendix dates remain scenarios, not execution claims.

## Unbound Qwen two-shard window

`tools/condense/doctor_v5_qwen_shard_window.py` implements the future bounded
window in which the canonical serial finalization of shard N may overlap the
prepared read/RHT/block work for shard N+1. It never derives a production thread
count from logical CPUs. A production-gated request must load the exact measured
winner for its tier/rate from the hash-bound vendor 8/12/16/20 contract, rehash
the profile, runtime binary, receipts, aggressive overlay, and controller, and
bind the same overlay across thread selection and admission. Missing, drifted,
or unqualified evidence is rejected. A one-thread local fallback and synthetic
two-slot test profile are explicitly non-production.

The coordinator uses unique temporary paths, validates child output/receipt
authority, commits strictly in shard order, adopts valid crash-complete work,
cancels later work after an earlier failure, and preserves parent sources. Every
admission transition and decisive sample is persisted before a launch or denial.
Production execution remains hard-disabled pending owner-free real-artifact
serial/window parity, lifecycle admission, and rollback review.

## GPT-OSS 120B work graph

`tools/condense/doctor_v5_gptoss_parallel_scaffold.py` converts the sealed MXFP4
inventory into:

- 576 independent eight-expert source units (36 layers x 16 batches);
- 36 independent dense-layer units;
- one embedding, one output-head, and one streamed lossless-sidecar unit; and
- 10 independent rate outputs per source unit: 615 source traversals and 6,150
  content-addressed outputs in total.

Each source traversal has an exact original shard/tensor/byte-range binding.
Traversal receipts add a digest for every range and bind the bounded staging
artifact. Output receipts bind that traversal, rate archive, and STR2 attestation.
Canonical merge refuses missing, duplicate, or extra outputs. The structural MoE
index maps `(rate, layer, expert)` to its canonical batch archive and tensor names,
but does not claim numerical runtime parity.

The exact 10-rate x 4-branch packet binds all 40 existing campaign cells and is
strictly pending: it writes no runtime spec and proposes no live registry entry.
Build and verify it with:

```sh
python3.12 tools/condense/doctor_v5_gptoss_parallel_scaffold.py build
python3.12 tools/condense/doctor_v5_gptoss_parallel_scaffold.py verify
```

Generated outputs are under
`reports/condense/doctor_v5_unbound/gptoss_120b_parallel/`.

`tools/condense/doctor_v5_gptoss_reuse_fanout.py` adds the exact four-branch
fanout layer, yielding 24,600 planned output jobs. Every job
has a unique evidence namespace, artifact instance, method receipt, attestation,
and exact prior-branch receipt dependencies. Canonical merge rejects evidence or
artifact aliasing. That structural fanout does not by itself prove that decoded,
outlier-zeroed, or RHT numerical buffers are reusable.

```sh
python3.12 tools/condense/doctor_v5_gptoss_reuse_fanout.py build
python3.12 tools/condense/doctor_v5_gptoss_reuse_fanout.py verify
```

## Bounded shared preprocessing

`tools/condense/doctor_v5_shared_preprocess_cache.py` narrows reuse to the exact
mathematical boundary. A bounded immutable source-range read may be shared. An
F32 decode and original-value rank/statistics cache may be shared only after an
independent decoder-parity qualification. The current Qwen encoder removes the
rate/branch-selected outliers before RHT, so zeroed bulk and RHT output may be
shared only when source, decoder, outlier percentage, RHT orientation, tensor
seed, and preprocessing implementation all match. Trellis geometry, codebook,
adaptive scales, Viterbi symbols, reconstruction, side information, attestation,
and scientific evidence remain unique to each rate/branch.

The concrete unbound Qwen builder expands a hash-sealed tensor inventory into
the exact 10 x 4 matrix. It never infers preprocessing from branch names: every
qualified cell must carry exact adapter, runtime-spec, and recipe authorities
plus all preprocessing fields; an absent/incomplete matrix blocks derived reuse.
The concrete GPT-OSS builder persists both its 24,600
consumer manifest and 615-unit cache plan; because the numerical decoder and
rate/branch recipes remain unqualified, all 24,600 derived-preprocess routes are
blocked and only source-range-read reuse is currently planned.

The cache admits at most one tensor/expert-batch unit, charges conservative
source/decode/rank/RHT workspace, preserves a 64 GB disk reserve, forbids a
whole-model cache, and recomputes the exact aggressive-controller transition
from its sealed prior state and fresh resource sample before admission. Candidate components and outputs must
match independently materialized serial-oracle artifacts byte-for-byte, with a
separate serial program, invocation, and semantic receipt. Crash adoption is
hash-bound. Ephemeral cache GC becomes eligible only after exact no-skip output
coverage and refcounts validate; parent sources and evidence are never deleted.
Execution and runtime-default mutation remain false.

```sh
python3.12 tools/condense/doctor_v5_shared_preprocess_cache.py requirements
python3.12 tools/condense/doctor_v5_shared_preprocess_cache.py build-gptoss-plan
python3.12 tools/condense/doctor_v5_shared_preprocess_cache.py \
  build-qwen-plan --inventory /absolute/path/to/unbound_qwen_inventory.json
```

## Source reading and tokenizer boundary

`tools/condense/doctor_v5_streaming_source.py` opens immutable regular files with
`O_NOFOLLOW`, checks inode/device/size/mtime identity, limits materializing reads
to 64 MiB, streams larger ranges with `pread`, and emits full-file or range hash
receipts without whole-file materialization.

The four tokenizer/chat-template assets are now pinned to repository revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. The offline gate proves identical
IDs across `tokenizers` and `transformers`, ID round-trip idempotence, and a
pinned chat-template vector. Its receipt SHA is
`24f3e8de6288ebf4935f99820686b578f2ca1e67d232339fe048d5861e86a0ee`.
Promotion remains quality-disabled until that evidence receives explicit review;
the missing-asset blocker itself is closed.

## Models beyond 120B

`tools/condense/doctor_v5_higher_tier_scaffold.py` is architecture-neutral. It
accepts no model without a sealed manifest containing exact parameters, immutable
local or range-addressable sources, per-range hashes, unique tensor keys, bounded
work units, estimated peak RAM, and thread declarations. Its deterministic wave
planner computes RAM and CPU lane caps while remaining non-executable until
measured peak, swap, pressure, thermal, source-range, and bit-exact canary gates
pass.

Its prior zero-swap placeholder has been removed. Both the requirements packet
and every unbound admission plan now bind the reviewed aggressive-controller
artifact and exact soft/hard/emergency growth/rate policy, require a swap
baseline sealed at quiescent promotion, forbid baseline ratcheting, and remain
non-executable until the overlay, baseline, and controller state are supplied
and revalidated.

## Optional second-host transport

`tools/condense/doctor_v5_distributed_transport.py` implements the transport
contract without opening a socket or asserting that another host exists:

- immutable content-addressed chunk manifests with exact range coverage;
- bounded chunk hashing, receiver CAS, resume, and dedup;
- canonical full-source reassembly verification;
- expiring signed host capabilities and coordinator leases;
- exact tool-hash, parent-plan, source-manifest, and work-unit bindings;
- host-signed result receipts followed by independent coordinator-signed artifact
  acceptance;
- signed host resource attestations bound to the same aggressive-controller
  hash, policy hash, coordinator-signed host/instance baseline authority,
  signed prior/next controller states, normal pressure, and the exact reviewed
  hysteresis/cooldown decision; the authority digest is carried through the
  transport plan and lease, and bounded nonzero swap is allowed only in the
  reviewed green/soft-throttle envelope;
- overlap detection for active leases, deterministic conflict rejection, retry
  receipts, verified-chunk retention, and unverified-partial deletion; and
- local-only fallback with no runtime-default change.

Contract signatures use out-of-band 256-bit-or-larger HMAC keys. Deployment must
also use mutually authenticated TLS; no secret is stored in a manifest. The
higher-tier wave plan now records this transport as optional and disabled by
default. Generate its no-host requirements packet with:

```sh
python3.12 tools/condense/doctor_v5_distributed_transport.py
```

At a nominal 1 Gbps, transferring the existing 65.25 GB GPT-OSS source has a
522-second wire floor. The contract reports roughly 580-696 seconds at 90%-75%
payload efficiency, before setup, verification, contention, or retries. This is
a planning envelope, not a measured speed claim. Resume/dedup can reduce later
attempts, while full receiver verification still consumes local I/O.

Generate the generic contract (which admits no model) with:

```sh
python3.12 tools/condense/doctor_v5_higher_tier_scaffold.py requirements
```

## Remaining real blockers

- Numerical Apple-Silicon GPT-OSS STR2 MoE execution and parity are not present.
- The tokenizer parity gate exists but is not yet marked promotion-reviewed.
- GPT-OSS-specific encoder parity/speed and measured per-unit RAM receipts have
  not run; the Qwen canaries do not substitute for MoE numerical evidence.
- The four adapters and 40 runtime specs are not reviewed or registered.
- No second host, authenticated channel, host key, capability receipt, or measured
  network throughput currently exists.
- Promotion must wait for the campaign's quiescent checkpoint and all existing
  observer, disk, lifecycle, rollback, and structural-readiness gates.

These are blockers, not negative scientific outcomes.
