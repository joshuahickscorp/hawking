# hawking.gravity.container.v1

The frozen on-disk container for Gravity-compressed models. The `.gravity` extension is
permanent. This ABI is frozen: readers written against v1 keep working. Representation
IDs are versioned separately and are expected to churn, because the container's job is to
describe and locate payloads, never to interpret them.

Status: FROZEN as of Generation B. Changes require a new `format_version` and a
compatibility entry in `GRAVITY_CONTAINER_COMPATIBILITY.json`.

## Why a container at all

A `.gravity` shard is self-describing. Given the file alone and nothing else, a runtime
can enumerate every tensor, learn the architecture and tokenizer it belongs to, verify
integrity at two levels, and decode any single tensor by seeking straight to it. No
original repository, no index sidecar, no whole-file read.

## Layout

```text
magic            8 bytes    b"GRAVITY\x00"
format_version   u32 LE     1
header_length    u64 LE
header           UTF-8 JSON, exactly header_length bytes
body             concatenated tensor payloads, each at its declared offset
```

The binary prefix is fixed at 20 bytes so a reader can locate the header without parsing
anything. Everything variable lives in the JSON header.

Offsets are assigned by the writer, never trusted from callers, so a descriptor cannot
disagree with where its bytes actually landed.

## Header

```text
schema           "hawking.gravity.shard_header.v1"
format_version   1
model            repo, revision, source_shard
architecture     type, hidden_layers, routed_experts, shared_experts, hidden_size
tokenizer        kind, source
compression      see below
shard            source, of
integrity        body_sha256, tensor_count
tensors          list of descriptors, see below
```

## Tensor descriptor

```text
name             the official source tensor name, unchanged
category         routed_expert | shared_expert | router | router_control |
                 normalization | indexer | embedding | lm_head | attention | other
layer            int or null
expert           int or null
shape            list[int], the source shape
elements         int, the product of shape
codec            representation id, or native.<dtype> for a natively carried tensor
terminal_state   PACKED_IN_CORE_ARTIFACT | PROTECTED_SOURCE_NATIVE
bpw              measured bits per weight for this tensor: bytes * 8 / elements
offset           byte offset into the body
bytes            payload length, always > 0
sha256           digest of this tensor's payload
```

`bytes` is never zero. A descriptor with no payload is malformed, not merely unusual.
This is the single rule that Generation A violated: it billed protected tensors into the
rate and listed them as descriptors while handing only compressed payloads to the writer,
so routers, router controls, normalizations and indexer tensors were accounted for at
16 BPW and physically written nowhere. The artifact then read as proof that the BF16
source could be evicted.

## The two rates

Both live in `compression` and both must reconcile against physical bytes.

```text
packed_bpw     compressed payload bits / compressed elements
               what the codec achieved on the tensors it compressed

complete_bpw   all payload bits / all elements
               what the shard actually costs, native organs included
```

`complete_bpw` is the only rate a candidate may be judged on. A campaign target such as
"<= 0.75" always refers to `complete_bpw`. `packed_bpw` exists to attribute a result to
the codec and is never a headline.

`verify()` recomputes both from the body and rejects the file if either claim is off by
more than 1e-6, or if any descriptor carries no payload.

## Native tensors

A tensor whose `codec` starts with `native.` carries its exact source bytes. It is not an
approximation and must round-trip bit-identically. Three reasons a tensor takes this path:

```text
PROTECTED_BUDGET_CLASS      the contract classified it as control-sensitive
NON_BF16_CONTROL_TENSOR     not a ladder candidate, still declared source weight
NO_ADMISSIBLE_LADDER_RUNG   no rung met the ceiling, so protect rather than exceed it
```

Native tensors are billed at their real width, in `complete_bpw`, from the same bytes
that are stored. Billing and storing are the same act.

## Integrity

Two levels, on purpose:

```text
body_sha256      covers every payload byte; catches truncation and corruption on open
tensor.sha256    per tensor; identifies which tensor is bad instead of condemning the shard
```

A hash check alone was never sufficient, and the Generation A artifacts prove it: they
passed both levels while missing eleven organs per organ-bearing shard. Coverage is a
separate property from integrity and is checked separately.

## Representation IDs

The container does not imply that any codec is permanent. Current IDs:

```text
glm52.pq.r0.v1                product quantization, the Generation A control family
glm52.functional.block.v1     native functional block student
glm52.indexshare.student.v1   IndexShare-aware attention student
glm52.hybrid.doctor.v1        native base plus a serialized Doctor correction
native.bf16                   exact source bytes, bfloat16
native.f32                    exact source bytes, float32
```

New IDs may be added at any time without touching the container version. A reader that
does not know an ID can still enumerate, locate, hash-verify and skip the tensor.

## What v1 promises

```text
the 20-byte prefix and its field order
the header is UTF-8 JSON at a declared length
every tensor is locatable by name without reading the body
every tensor carries a non-empty payload and its own digest
both rates reconcile against physical bytes
native.<dtype> payloads are exact source bytes
```

## What v1 does not promise

```text
any particular codec, rung, or geometry
that a given representation id will still be produced
tensor ordering beyond the declared offsets
that the header carries any field not listed above
```
