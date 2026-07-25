# Handoff request to the live GLM-5.2 science stream

From: the inference campaign (`campaign/hawking-true-batch1-1000tps`).
To: the live GLM-5.2 stream (`campaign/glm52-bf16-xet-gravity`), which owns source fetching, teacher
capture, packing, eviction and scientific representation selection.

Nothing in this request has been acted on. The inference campaign has not read the live capture directory,
has not touched a BF16 shard, and opens `.gravity` shards read-only and only when their mtime is more than
two hours old. This document exists because the two items below are the science stream's to decide, and the
inference campaign's projected ceiling stops without them.

## Why this is being asked now

The sealed budget arithmetic (`HAWKING_1000TPS_ROOFLINE.json`,
`HAWKING_MULTI_THOUSAND_ROADMAP.md`) separates two independent limits and finds:

- Runtime work alone, with the current artifact, projects **102.8 tok/s**. That is the whole open lane.
- Every projected configuration above **230 tok/s** requires training a student against teacher state.
- The first configuration that projects past 1,000 tok/s requires depth 78 to 10 **and** top-8 routing to
  top-2 **and** the protected organs compressed, simultaneously.

So the inference campaign can reach roughly 100 tok/s on its own and then stops. Both requests below are
prerequisites for anything beyond that, and neither is difficulty. They are ownership.

## Request 1: an immutable teacher-capsule export

Requested as a sealed, read-only snapshot outside the live capture path, so the inference campaign never
reads a capture in flight:

- teacher capsules for **layers 3 to 15** (chosen because all 256 experts for those layers are already
  packed and present in sealed shards, so no further packing is implied by this request)
- membership hashes
- source revision, to pin against `b4734de4facf877f85769a911abafc5283eab3d9`
- tensor lineage
- pre-router states
- router traces
- weighted MoE outputs
- post-block states

What each is for, so the request can be trimmed rather than refused wholesale:

| item | lane it unblocks | what it decides |
|---|---|---|
| pre-router states, router traces | D | whether top-8 can become top-2 without losing the routing decision |
| weighted MoE outputs | D | whether a shared functional path can carry what the dropped experts carried |
| post-block states | E | whether 2, 4 or 8 teacher blocks can be fused into one compact block |
| membership hashes, lineage, revision | all | that a student was trained against the same teacher the artifact came from |

If only part of this is available, the useful minimum is **post-block states for layers 3 to 15**: that
alone opens the depth-collapse pilot, which is the larger of the two multipliers in the ladder.

## Request 2: a ruling on protected-organ compression

Four organ classes are carried at source precision in the sealed artifacts and are marked
`CONTROL_SENSITIVE_CANDIDATE`:

| organ | MB/token | share of token | stored |
|---|---:|---:|---:|
| indexer | 394 | 7.9% | 16.000 bpw |
| router | 236 | 4.7% | 16.000 bpw |
| normalization | 2.3 | 0.05% | 16.000 bpw |
| router_control | 0.08 | 0.00% | 16.000 bpw |
| **total** | **632** | **12.6%** | |

This is why the effective active rate is **0.995 bits per active weight** rather than the artifact's
headline 0.876. Compressing these at R0 is worth **1.14x** on the token, which is real but is not a
milestone-class change, so this is the lesser of the two requests and should not be prioritized over the
capsules.

The question is narrow: **may any of these organs be compressed, and if so which?** The inference campaign
has not evaluated why they are protected and is not asking to override that judgment. A ruling of "no" is a
complete answer and will be recorded as a structural floor on the active-byte lane rather than revisited.

## What the inference campaign will do either way

- Lane G, the persistent causal engine, proceeds now and needs nothing from the science stream. It removes
  a submission floor of 2.21 tok/s that currently caps the token regardless of kernel quality.
- Lane A kernel selection proceeds and is measured against real sealed artifacts.
- Neither lane will claim a measured tokens-per-second figure until a complete token executes. No complete
  token has executed yet.

## What is explicitly NOT requested

- No change to eviction policy, source windows, the controller, the lease, or any live ledger.
- No pause to teacher capture.
- No re-pack. The shared-codebook configuration that would make lookup-linear far cheaper is measured as
  absent from disk (60 of 60 codebooks distinct on one shard), and a re-pack to share codebooks across
  experts is a science decision this campaign is not requesting, only noting as measured.
