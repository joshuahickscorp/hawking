# `.strand` v3 — the provenance layer (spec)

_Status: the hashing layer is **SHIPPED** (`crates/strand-quant/src/provenance.rs`, 8/8 tests
green, KATs cross-pinned against an independent hashlib implementation). The archive
extension (§6) is **DESIGN (doc only)** — it does not change `format.rs`, v1, v2, the
quantizer, or the integer decode contract. `provenance.rs` is this spec's reference
implementation; this doc and that module's constants must agree byte-for-byte._

---

## 1. Why this layer exists (the one-paragraph gate)

Every quantized-model format can hash its **file bytes**. None of the float-codebook
formats can hash its **weights** — the values the model actually computes with — because
their dequantized outputs drift per platform, compiler, SIMD width, and thread count.
STRAND's decode is integer-only and **byte-identical everywhere** (proven GPU==CPU on the
M3; `decode_lean` pinned against `decode_tensor_fixed` on every input), so for STRAND, and
only for formats in STRAND's determinism class, *"the SHA-256 of the decoded model"* is a
well-defined constant. The provenance layer makes that constant canonical: one 32-byte
**model root** that any party on any hardware can recompute and match, plus per-block
leaves cheap enough to spot-check without a full decode. This is not a feature bolted onto
the moat — it is the moat made *checkable by strangers*.

What it is **not**: a proof that a given inference *response* used these weights (§5.2
states the honest boundary), and not a substitute for a file-at-rest hash (§3.1 states
exactly what the roots bind and what they deliberately do not).

## 2. Canonical serialization (wire level)

Conventions, shared by every layout below:

| rule | value |
|---|---|
| hash | FIPS 180-4 SHA-256 (`crate::sha256::sha256`, pinned to the NIST vectors) |
| integers | little-endian, fixed width, no varints |
| counts | always explicit (every layout is injective — no concatenation ambiguity) |
| domain tags | 8 ASCII bytes, first bytes of the preimage; the three spaces are disjoint |
| hashed object | **decoded Q12 `i32`s** from the reference decode, never wire bytes |

The decoded stream is `decode_tensor_fixed_with_lut(enc, cfg, lut)` — RHT-space Q12
integers, in output order. `cfg` is the tensor's `TrellisConfig` (`l_bits`, `k_bits`,
`vec_dim`, `block_len`); `lut` is the same explicit state-major `[2^L * d]` Q12 table the
decode takes. For the shipped scalar path the LUT is the frozen `codebook_lut(l_bits)` —
pinned by `l_bits` alone, compiled into every conformant decoder. A learned-LUT tensor
must supply its table exactly as it already must for decode; the provenance layer adds no
new LUT channel.

### 2.1 Block leaf

```text
leaf = sha256(
    "SPV3.BLK"                  [u8; 8]   DOMAIN_BLOCK
    block_index                 u64 LE    index into enc.blocks
    n                           u32 LE    weights in this block (== BlockMeta::n)
    q12[0] .. q12[n-1]          i32 LE    the block's decoded Q12 values, output order
)
```

`block_index` binds position (two identical blocks at different indices hash differently);
explicit `n` plus fixed-width values make the preimage unambiguous. The leaf binds **every
bit that influences the decoded weights**: payload bits, `scale_q`, `sub_scales`,
`min_base_q`/`mins`, `init_state`, the wire flags (`tail_biting`, `has_affine_min`), the
trellis geometry, and the LUT — all of it flows through the decode into the Q12 output.
Conversely it deliberately does **not** bind the encoding itself: two different wire
layouts (v1 vs v2) or even two different bit streams that decode to the same Q12 values
hash identically. Layout-independence is a feature — the root names the *weights*, not the
container.

### 2.2 Tensor root

```text
tensor_root = sha256(
    "SPV3.TNS"                  [u8; 8]   DOMAIN_TENSOR
    n_blocks                    u64 LE
    leaf[0] .. leaf[n_blocks-1] [u8; 32]  ordered block leaves
)
```

A flat ordered hash list, not a binary tree — §3 gives the trade and the upgrade path.

### 2.3 Model root

```text
model_root = sha256(
    "SPV3.MDL"                  [u8; 8]   DOMAIN_MODEL
    n_tensors                   u64 LE
    per tensor, in canonical order:
        name_len                u32 LE
        name                    [u8; name_len]   raw UTF-8 (same encoding as the STR2 descriptor)
        tensor_root             [u8; 32]
)
```

Order-sensitive by design (§3.3). Duplicate tensor names are forbidden at the spec level.
Per-tensor `cfg`/`lut` may differ within one model (mixed-precision: down_proj@4, rest@3 —
one model, two LUT geometries); each tensor's leaves use its own.

### 2.4 Domain tags, versioning, and the pinned KATs

`SPV3.*` is frozen. **Any** change to any layout above — a field, a width, an order —
mints a new tag family (`SPV4.*`); the old tags are never reused with new semantics. The
fourth tag, `"SPV3.SEL"`, feeds only the self-test-vector *selection* (§4.2) and never an
expected hash.

Known-answer pins (unit-tested in `provenance.rs::tests::kat_canonical_serialization_pinned`,
cross-computed with an independent python-hashlib implementation of the layouts above):

| input | sha256 |
|---|---|
| `block_hash(0, [])` | `2234d6fcd138505967fb0e109d23cc8452e9498895621e3b30bdb943fd72296f` |
| `block_hash(1, [1, -2, 1000])` | `3990f085d2669dd68b30e9adcfe5a9e69ddcd7ccb2c5d703326aa5c1df7f30c8` |
| `tensor_root([])` | `c6f7a00b2d83315a5315962a057a525886a321b5b2c76200fb9694d6db476a53` |
| `tensor_root([k1, k2])` | `1492f395a50b7c64691f10646c707c6966e09ae4f017fb023ba37356dfb19edf` |
| `model_root([])` | `b482c517817d47c0d95c6808b6ef48b1d86e14fa7c02b780969b467ea23026a7` |
| `model_root([("alpha", k3), ("beta", k4)])` | `d390075de4f8486e6d3d6e975ec5bea8dc5e5858694817e4d635ec8545508666` |

If any of these moves, the wire format moved — mint a new tag or revert.

## 3. The Merkle structure

Two levels, wide nodes:

```text
model_root
 ├─ ("model.layers.0.q_proj",   tensor_root) ── leaf[0] leaf[1] … leaf[B-1]
 ├─ ("model.layers.0.k_proj",   tensor_root) ── …
 └─ …                                            one leaf per 256-weight block
```

**Why a flat hash list instead of a binary Merkle tree:** the dominant verification modes
either need every leaf anyway (full re-decode, §5.1) or hold the leaf list locally
(archive section §6, serving setups §5.2), where recomputing a root over `B` 32-byte
hashes is microseconds. The flat list is one hash invocation with zero padding/duplication
edge cases (the duplicate-leaf mutability class of naive binary trees is structurally
absent), and simplest-possible means fewest places for nondeterminism. The cost, stated
honestly: an *inclusion proof* for one block ships the whole leaf list — `32 B` per block
= **exactly 1.0 bpw at `block_len = 256`** (`32·8/256`) — instead of a binary tree's
`32·log2(B)` bytes. If a remote-light-verifier use case ever matters, a binary inner-node
rule is a tag-mint away (`SPV4`); the `SPV3.BLK` leaves would not change.

### 3.1 What the roots bind — and what they do not

| bound by the roots | NOT bound (and where it lives instead) |
|---|---|
| decoded Q12 stream of every block (payload + all side-info + flags + geometry + LUT, via the decode) | `rht_seed` (descriptor; changes recon, not Q12) |
| block order, block sizes | tensor `shape` (descriptor) |
| tensor names, tensor order | archive layout, padding, offsets |
| | `source_sha256` (the *input* model's hash) |

The roots name the deterministic integer artifact. Archive metadata around it is already
covered at-rest by hashing the file bytes — any v2 consumer can do that today. A publisher
who wants both identities signs the pair `(model_root, sha256(file))`: the file hash gives
at-rest integrity of one particular container; the model root gives platform-independent
identity of the computed-with weights across *any* container (v1, v2, a future v3 — same
root). The two are complementary; neither substitutes for the other.

### 3.2 Sensitivity

Blocks decode independently (per-block `init_state` + side-info, prefix-sum bit offsets;
a tail-bitten block's start state is a function of its **own** trailing symbols). So:
flipping one consumed payload bit changes **exactly the owning block's leaf**, which
changes the tensor root, which changes the model root — unit-tested
(`payload_bit_flip_flips_exactly_one_leaf_and_all_roots`). Index, value, sign, order, and
length sensitivity of the leaf serialization are pinned by `leaf_serialization_sensitivity`.

### 3.3 Canonical order

The model root hashes `(name, tensor_root)` pairs **in the artifact's tensor order** — for
a v2 archive, descriptor order in the file; for an in-memory model, the iteration order the
caller fixes. Order binds (tested); there is no sort-by-name normalization, deliberately:
the artifact is the canon, and re-serializing the same tensors in a different order is a
different artifact. Publishers MUST publish roots computed over the artifact they ship.

## 4. Spot-check verification (self-test vectors + inclusion)

Full trust = full re-decode (§5.1). Everything in this section buys *cheap* partial trust:
verify `m` of `B` blocks for `m/B` of the decode cost. Inclusion-style checks against a
bare tensor root ship the ordered leaf list (§3: flat list — the whole list or nothing)
and recompute the root over it; the embedded-vector path below needs no leaf list at all.

### 4.1 The vector record

```text
ProvenanceVector = {
    block_index   u64 LE
    block_hash    [u8; 32]    expected leaf (§2.1) of that block
}                              — 40 bytes on the wire
```

### 4.2 Content-derived selection (no RNG, anywhere)

`make_test_vectors(enc, cfg, lut, k)` picks `k` distinct block indices as a **pure
function of the encoded artifact** — producer and verifier agree with zero shared state,
and the selection cannot be steered at verify time:

```text
seed      = sha256("SPV3.SEL" || n_blocks u64 LE || total u64 LE || enc.bits)
cand(j)   = u64_le(sha256(seed || j u32 LE)[0..8]) mod n_blocks      j = 0, 1, 2, …
```

Collisions resolve by upward linear probe (mod `n_blocks`); `k` is clamped to `n_blocks`
(so termination is guaranteed: a free slot always exists); output is sorted ascending
(canonical, and locality-friendly for the re-decode). The `mod` bias is ≤ `n_blocks/2^64`
— irrelevant for coverage sampling; this is tamper-evident *coverage*, not a cryptographic
commitment scheme. Note the split of duties: the **selection** binds the payload (changing
`bits` reshuffles which blocks are chosen); the **hashes** bind the decoded content.
Expected hashes come from `decode_block_q12` (§4.3), so creating vectors exercises the
same single-block path verification uses.

### 4.3 Verification (`verify_test_vectors`)

For each vector: bounds-check the index (out-of-range ⇒ `false`, never a panic on hostile
input), re-decode **only that block** via `decode_block_q12` — per-block bit offset is the
prefix sum `Σ num_steps(n_b)·k` (the same cursor the full decoders walk and v2's
`BlockOffsetRecord::bit_offset` stores), side-info is the block's own — recompute the leaf,
compare. All match ⇒ `true`; the empty vector set is vacuously `true`.

`decode_block_q12` is a new decode path, so it carries the standing determinism
obligation: **bit-identical to `decode_tensor_fixed_with_lut` / `decode_lean_with_lut` on
every input**, asserted by `block_decode_is_bit_identical_to_reference` across
`k ∈ {2,3,4}`, all four wire-lever combinations (tail-biting × affine-min), lengths hitting
the structural edges (single weight, sub-block tail, exact block, short final block with
`n·k < L`, multi-block), a small-`L` geometry, and the `d=2` vector trellis. A single
counterexample = the layer hashes something other than what the model computes with — a
release blocker, same class as a `decode_lean` divergence.

## 5. The three use cases

### 5.1 Trustless distribution

The publisher computes `model_root` once and publishes/signs 32 bytes. Any consumer pulls
the artifact from **any** untrusted channel — mirror, torrent, CDN, a stranger's USB stick
— decodes on whatever hardware they have, recomputes the root, and compares. Match ⇔ the
decoded weights are byte-identical to the publisher's, full stop; no trusted mirror, no
per-platform "expected outputs" table. Float-codebook formats cannot offer this — their
dequantized values are not platform-constants, so there is nothing canonical to sign.
Determinism is the precondition; the root is the 32-byte interface to it.

### 5.2 Verifiable inference claims (the challenge protocol)

A provider claims "we serve model `R`" (a published root). A verifier holding `R` (and the
artifact, or just its leaf lists) challenges: *send the decoded Q12 slice (or leaf) of
block `i` of tensor `t`* — indices chosen by the verifier, or by the §4.2 content-derived
selection when the parties want a fixed audit set. The provider answers from its live
copy; the verifier checks the leaf against `R` (re-decoding locally only the challenged
blocks — `m/B` cost, milliseconds for `m` in the tens). Random challenges make holding a
different model while answering correctly require holding the real one too — at which
point the claim is true. **Honest boundary:** this proves the provider *possesses and can
decode* the committed weights at challenge time; it does **not** prove any particular
inference response was computed with them. Binding responses to weights needs heavier
machinery (attestation, zkML) and is explicitly out of scope for v3.

### 5.3 Self-verifying artifacts

Embed roots + `m` self-test vectors in the archive itself (§6). A loader then verifies at
mmap time, before serving: structural checks (free) → `verify_test_vectors` on the
embedded set (`m` block decodes, milliseconds) → optionally the full root recompute (a
complete decode — the §5.1 check, run locally). Bit-rot, truncation, a corrupted mirror,
or a tampered payload is caught at load with a cost knob (`m`) the deployment chooses, not
at inference time as silent garbage. Cheap-first laddering, applied to trust.

## 6. v3 archive extension — OPTIONAL trailing section on STR2 v2 (DESIGN)

**Constraint honored: `format.rs` is not modified.** The v2 wire format (`STR2`,
version 2, the 56-byte file header, descriptors, page-aligned regions) is byte-frozen.
The provenance section is **append-only and optional**: a v2 file with the section is a
valid v2 file to every existing reader, because (a) all v2 internal offsets are absolute
from file start, and (b) `read_strand_v2_header` / `read_strand_v2` validate regions with
overrun checks only — trailing bytes after the last region are never touched (verified
against the parser as shipped). Retrofitting provenance onto an already-baked archive is
therefore a pure append — no offset rewrite, no header patch.

Layout (all integers LE; one new page after the last side-info region's padding):

```text
… last v2 region, padded to PAGE …
PROV HEADER  (page-aligned, 64 bytes)
    magic            [u8; 4]   = "SPRV"
    version          u32       = 1
    n_tensors        u32         must equal the file header's n_tensors
    flags            u32         bit 0 = full leaf lists present; others reserved 0
    model_root       [u8; 32]    §2.3, over the descriptors in file order (§3.3)
    reserved         [u8; 12]    zero
PER-TENSOR RECORDS  (descriptor order, back-to-back)
    tensor_root      [u8; 32]   §2.2
    n_blocks         u64         must equal the descriptor's n_blocks
    n_vectors        u32         m (recommended default 8; 0 = none)
    reserved         u32         zero
    vectors          [40 B × n_vectors]    §4.1 records, block_index ascending
    leaf_list        [32 B × n_blocks]     only if flags bit 0 (see sizing — default OFF)
TRAILER  (the last 16 bytes of the file — self-locating from EOF)
    prov_offset      u64         absolute byte offset of PROV HEADER (page-aligned)
    prov_bytes       u32         PROV HEADER + records, excluding this trailer
    magic            [u8; 4]   = "SPRV"
```

Discovery: read the last 16 bytes; if they do not end in `"SPRV"` the section is absent.
(A pre-extension v2 file ends in zero page-padding except in the measure-zero case where
a region exactly fills its final page and its last data bytes happen to spell the magic —
so the trailing magic is the cheap filter, never the verdict.) If the magic matches,
validate the chain: `prov_offset % 4096 == 0`, `prov_offset + prov_bytes + 16 ==
file_len`, header magic/version at `prov_offset`, `n_tensors` matches the file header,
per-tensor `n_blocks` matches each descriptor. Any mismatch ⇒ the section is treated as
absent (it is optional) unless the caller asked for strict verification, in which case it
is an error. Verification ladder = §5.3.

Per-tensor decode parameters (`l_bits`, `k_bits`, `vec_dim`, `block_len`, flags) come from
the v2 descriptor the verifier already parses; the section adds **no** new decode inputs.

Rejected alternative, recorded: repurposing the file header's `reserved u32` (offset 52)
as a section pointer. It would work (u32 page index covers 16 TiB) but changes the frozen
56-byte header's semantics and makes retrofit an in-place patch instead of an append.
The EOF trailer costs 16 bytes and touches nothing.

Sizing, billed honestly (will.md §5.11 — overhead claims that hide bytes are lies):

| component | cost | example: Qwen2.5-0.5B (168 proj tensors, 357.9 M weights ⇒ ~1.40 M blocks @ 256 w) |
|---|---|---|
| header + trailer | 80 B | 80 B |
| per-tensor record, m=8 | 48 + 320 B | 168 × 368 B ≈ 60 KiB ≈ **0.001 bpw** — free |
| full leaf lists (flags bit 0) | **exactly 1.0 bpw** (32 B / 256 w) | ~45 MiB — **+43 % on the 2.34-bpw 2-bit rung** |

Hence the default ships roots + vectors only; leaf lists are for serving deployments that
answer §5.2 challenges from the archive, and they pay for what they use.

## 7. Reference implementation map

`crates/strand-quant/src/provenance.rs` (`pub mod provenance` in lib.rs):

| spec | code |
|---|---|
| §2.1 leaf | `DOMAIN_BLOCK`, `block_hash`, `block_hashes` |
| §2.2 tensor root | `DOMAIN_TENSOR`, `tensor_root_from_hashes`, `tensor_root` |
| §2.3 model root | `DOMAIN_MODEL`, `model_root_from_tensor_roots`, `model_root` |
| §4.1 record | `ProvenanceVector` |
| §4.2 selection | `DOMAIN_SELECT`, `make_test_vectors` |
| §4.3 single-block verify | `block_bit_offset`, `decode_block_q12`, `verify_test_vectors` |

Tests (all in-module; `nice -n 19 cargo test -p strand-quant --lib provenance`):
`kat_canonical_serialization_pinned` (§2.4 KATs, independently cross-computed) ·
`block_decode_is_bit_identical_to_reference` (THE determinism contract, §4.3) ·
`hashes_and_roots_are_deterministic` · `payload_bit_flip_flips_exactly_one_leaf_and_all_roots`
(§3.2) · `leaf_serialization_sensitivity` · `model_root_is_name_and_order_sensitive` (§3.3)
· `test_vectors_verify_and_catch_corruption` (pass + 3 tamper modes + edge `k`) ·
`empty_tensor_is_well_defined`.
