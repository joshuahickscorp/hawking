# STRAND deterministic swarm inference — Tier 3a design

**Status: DESIGN ONLY (2026-06-10, frontier-wave, rev 2).** Nothing in this document is a
running swarm. Rev 1 predated the provenance layer and placed speculative requirements on a
then-unwritten spec; rev 2 is rebuilt on the **landed** SPV3 implementation
(`crates/strand-quant/src/provenance.rs`, 8/8 tests green in this worktree, including the
bit-identity contract test and the known-answer pins). Every claim is tagged:
*proven* = a test or measured number that exists; *designed* = specified here, unbuilt;
*estimate* = math, not measurement — replaced by measurements at the demo gate (§8).

Hash source of truth: `docs/STRAND-v3-provenance-spec.md` (the v3 provenance spec;
`provenance.rs` is its reference implementation — the doc and the module's constants must
agree byte-for-byte). **This design adds NO hash scheme.** Where it needs something the
spec does not yet cover, it records a requirement (R1–R4, §3.2) for the spec author —
not a parallel construction. Companions: `docs/STRAND-format-v2-spec.md` (wire),
`docs/STRAND-dismantle-wiring.md` (runtime seam). will.md is king; this doc is a reference.

---

## 0. The claim, stated exactly

A set of heterogeneous devices (aarch64 Mac, x86_64 server, WASM page, MCU) each hold
disjoint shards of one `.strand` model and serve them. Because STRAND's decode is pure
integer (`state → frozen LUT → (scale_q × q) >> 16`), every device reconstructs its
shard's Q12 weight plane **byte-identically**, under **any** thread/SIMD/GPU schedule
(*proven*: `gemv_par.rs:599 par_and_simd_decode_are_bit_identical`, `:669
par_simd_bit_identical_large`, GPU==CPU `metal.rs:530 gpu_q12_matches_cpu_decode_lean`).
Identity, not tolerance, is therefore the correctness test for a shard — which makes
shards **attestable**: a 32-byte SPV3 digest either matches or the node is wrong, and any
third device re-derives the truth mechanically. The canonical digests of that plane
already exist and are KAT-pinned (*proven*, §3.1).

**The honest boundary, up front:** what is attestable today is the *weights plane* —
the decoded Q12 stream and, through it, the wire bytes' semantic content — NOT the
inference outputs. Activation math (RHT of `x`, the f32 GEMV MAC, layernorm, softmax,
logits, sampled tokens) is float and is not bit-exact across heterogeneous kernels by
construction. §2 is the precise ledger. A swarm that claims "verified inference" with
today's stack would be lying; the claim this design supports is **"verified model,
verified shards, verified decode — best-effort reproducible inference."** The upgrade
path to attestable layer outputs is an integer activation path (§2.4), which does not
exist and is gated on a quality measurement.

Float tensor-parallel systems cannot offer even the weights-plane claim, for a structural
reason (§2.3): their correctness tests are tolerance bands, and a tolerance band is a
place for a byzantine node to hide. Ours is equality.

---

## 1. Sharding unit

### 1.1 The candidates

| unit | granularity (Qwen2.5-7B q2) | decode-independent? | inference-useful? | verdict |
|---|---|---|---|---|
| whole model | 1 | — | — | not sharding |
| tensor | ~196 proj tensors | yes (per-tensor descriptor/seed/cfg) | yes (a node serves whole projections) | assignment unit for small/RAGGED/vector tensors |
| **row-aligned block range** | rows: 3584–18944 per tensor | yes (STRICT ⇒ rows = whole blocks) | **yes — complete output rows ⇒ cross-node combine is concatenation, not float addition (§5)** | **the assignment unit** |
| single block (256 w) | ~25M blocks | **yes — the proven atom** | no (a block is a row fragment) | **the verification unit** |
| byte range | arbitrary | n/a | no | the *transfer* encoding (page-aligned range-GET), not a semantic unit |

### 1.2 Why any block partition decodes independently (the load-bearing fact)

Not designed — already proven twice over, in two crates:

- Each `BlockMeta` owns its `init_state`, `sub_scales`, `mins`, `min_base_q`, `n`
  (gemv_par.rs:7). The only cross-block state in the serial decode is the bitstream
  cursor and the output cursor — both pure prefix sums of `n` (`block_plans`,
  gemv_par.rs:67). Given `(start_bit, out_off)`, a block decodes with **zero data
  dependency** on any other block.
- Tail-biting does not break this: the start-state re-derivation reads only the block's
  *own* `n_steps` symbols from its own bit range (gemv_par.rs:230–241; same gate
  `n·k ≥ L` in `provenance.rs::decode_block_q12`). Block-local.
- The v2 wire format materializes the prefix sums on disk: every block has a 16-byte
  `BlockOffsetRecord { bit_offset, init_state, scale_q }` (format.rs:242), so a consumer
  seeks to any block with zero scan.
- **The single-block decode primitive now exists** (*proven*):
  `provenance.rs::decode_block_q12` decodes exactly one block from `(enc, cfg, lut,
  block_index)`, touching only that block's bit range and side-info, and carries the
  standing determinism obligation — `block_decode_is_bit_identical_to_reference` asserts
  byte-identity against BOTH reference decoders across k∈{2,3,4}, all four wire-lever
  combinations, structural edge lengths, small-L, and the d=2 vector trellis.

One pleasant arithmetic fact: a full block is 256 weights ⇒ `256·k` bits ⇒ `32·k` bytes
— **every full-block boundary is byte-aligned** (64/96/128 B at k=2/3/4; the shipped
vector config too: `ceil(256/d)·k` at d=2,k=4 is 64 B). Only the final short block of a
tensor can end mid-byte, and nothing follows it. So a shard's payload is an exact byte
slice; a sliced `EncodedTensor { bits: payload[byte_base..], blocks: blocks[b0..b1] }`
re-bases automatically (prefix sums over the truncated block list are slice-relative) and
the existing `decode_q12_par` runs verbatim on it. The missing piece is one loader
function, `encoded_block_range(name, b0, b1)` — the side-info walk
`StrandModel::encoded_tensor` (loader.rs:160) already does, truncated (~80 LOC, §8/M1).

### 1.3 The decision

- **Verification unit = the block (256 weights).** Independently addressable
  (`BlockOffsetRecord`), independently decodable (`decode_block_q12`, contract-tested),
  already the leaf unit of the provenance layer (§3.1) — the sharding design and the hash
  design meet at the same atom, which is not a coincidence: both exist because of
  per-block `init_state` + prefix-sum offsets.
- **Assignment unit = a row-aligned contiguous block range of one tensor.** STRICT
  tensors (`in_features % 256 == 0`; true for Qwen2.5-7B: 3584 = 14·256, 18944 = 74·256;
  Llama-2-7B: 4096 = 16·256, 11008 = 43·256) have `blocks_per_row = in/256`, so
  row-aligned means `b0, b1 ≡ 0 (mod blocks_per_row)`. Row alignment is what makes a
  shard useful for inference (§5): the node emits complete `y` rows.
- **RAGGED tensors (the 0.5B's 896-dim) and vector-trellis tensors: whole-tensor shards
  only** in v0. RAGGED rows straddle blocks (the linear `(row, block)` map does not
  hold); a block range is still *decodable* and *verifiable* but not row-meaningful.
- **Transfer encoding = byte ranges of the page-aligned v2 regions.** `PAGE = 4096`
  alignment of table/payload/side-info (format.rs:227) means a shard fetch is plain HTTP
  range-GETs / `dd`-style offsets — no re-framing, no re-encode. And because the decoded
  plane is the canonical object (§3.1), **fetched bytes are never trusted: every received
  range is verified by decoding it and checking leaves against the manifest** (§3.4).

### 1.4 Shard descriptor (what every attestation message binds)

```
ShardDesc {
  model_root        [u8;32]   // SPV3 model root (§3.1) — pins the artifact version
  tensor_root       [u8;32]   // SPV3 tensor root for tensor_name
  tensor_name       (u32 len ‖ bytes)        // injective, mirrors the spec's encoding
  b0, b1            u64 LE                    // block index range [b0, b1)
  // interpretation metadata — MUST be bound here because the Q12 plane does not pin it (§3.2):
  shape             (u32 ndim ‖ u64 dims…)
  rht_seed          u64 LE                    // 0 = no RHT
  l_bits, k_bits, vec_dim   u8
  flags             u32 LE                    // tail_biting, affine_min (format.rs:26)
}
```

A shard ships: its `BlockOffsetRecord[b0..b1]` slice, its payload byte slice, its
side-info slices (`sub_scales` 6 B/block; if affine: `min_base_q` 4 B + `mins` 6 B per
block), and the descriptor. Everything a node needs to decode is in the shard; the frozen
LUT is compiled into the binary (`lut_tables.rs`, L∈[4,14]).

**Two wire-format gaps, named honestly (v0 scope excludes both; v3 requirements R3/R4):**
1. **Learned vector LUTs are not stored in the v2 file** — vec-trellis tensors (the
   ~1.5-bit rung) cannot be shard-decoded from the artifact alone today. Until v3 adds
   the LUT region, swarm v0 = scalar tensors (the entire 4/3/2-bit product line —
   q2_l12_out1's trellis plane, mp_light, q4 — is scalar ✓).
2. **The outlier channel is not in the v2 file** (`--outlier-channel` lives in the
   quantize-model pipeline as a sparse integer (idx, val) side channel) — and it is **not
   covered by SPV3 leaves either** (leaves hash the trellis-decoded Q12; the outlier
   merge happens after). q2_l12_out1 *needs* it for reconstruction. v3 must serialize it
   per-tensor and define its digest (a new domain tag per the spec's minting rule — owned
   by the spec, not here). It is integer data, so once defined it inherits the same
   equality-attestation as the trellis plane; a shard carries the (idx, val) entries
   whose flat index falls in `[b0·256, b1·256)`.

### 1.5 Per-block cost table (wire vs verification vs plane)

| item | bytes/block (k=2 / 3 / 4) |
|---|---|
| payload | 64 / 96 / 128 |
| BlockOffsetRecord | 16 |
| sub_scales | 6 |
| affine (min_base_q + mins) | 10 (when on) |
| **wire total** | **86–160** |
| SPV3 leaf (manifest side) | 32 |
| decoded Q12 plane (256 × i32 LE) | 1024 |

Qwen2.5-7B at the 2-bit rung ≈ 6.5e9 proj weights ≈ 25.4M blocks ≈ 1.9 GB wire
(0.292 B/w canon density) vs ~26 GB decoded plane; the full leaf list ≈ 813 MB
(verifier/coordinator-side only — nodes hold wire bytes, §3.3). The 0.5B: ~1.4M blocks,
~45 MB leaf list. *(Estimate-class; M0 measures the real artifacts.)*

---

## 2. What is and is NOT bit-exact (the ledger — read before claiming anything)

### 2.1 The plane-by-plane truth table

| plane | arithmetic | bit-exact across heterogeneous devices? | evidence / reason |
|---|---|---|---|
| artifact bytes at rest / on wire | — | **YES** | byte equality; page-aligned ranges transfer verbatim |
| v2 header parse | integer | **YES** | single canonical parser `read_strand_v2_header` (format.rs:556) |
| block decode → Q12 | pure integer | **YES, under ANY schedule** | `decode_lean` == fast == par == SIMD == Metal == `decode_block_q12`, all asserted bit-for-bit; GPU==CPU proven on M3 |
| SPV3 digests of the Q12 plane | integer serialization + SHA-256 | **YES** | *proven*: KAT-pinned vs independent hashlib; determinism tests; hashes have no float problem — hardware SHA (SHA-NI / ARMv8-crypto) is fine |
| outlier channel apply (integer idx/val) | integer | **YES** (once v3 serializes + covers it) | integer merge, order-free |
| RHT of activations `x` | f32 butterfly | **NO** (pinned-impl reproducible only) | gemv.rs:12–17: RHT(x) is the caller's float job |
| GEMV MAC (Q12·x) | f32 | **NO** by construction across kernels | reduction order; FMA contraction; SIMD lane sums |
| layernorm / softmax / rotary / KV cache | f32/bf16 | **NO** | transcendental libm variance + reduction order |
| logits / sampled tokens | f32 + RNG | **NO** | downstream of all of the above |

### 2.2 The precise distinction (this is the whole design, in two sentences)

Integer decode is **schedule-invariant**: any partition, any thread count, any lane
width, any device produces the same bytes — equality survives parallelism. Pinned-order
float is merely **reproducible**: Rust scalar f32 with one fixed accumulation order does
match across mainstream IEEE-754 targets in practice (the e2e tests compare exact f32
bits *on one machine*; cross-arch equality of the pinned scalar MAC is plausible but
**unmeasured** — demo gate G4, §8, measures it instead of assuming it) — but ONE kernel
upgrade, ONE SIMD reduction, ONE FMA contraction breaks it silently. Reproducible-by-
discipline is not attestable-by-construction. The swarm's hard guarantees live strictly
on the integer side of this line; everything float is "cohort-pinned best effort" (§5.3).

### 2.3 Why float tensor-parallelism cannot have this

A float-TP system (sharded sum-reductions across nodes) computes `Σ partial_i` whose
value depends on reduction order and on each node's kernel. Two honest float-TP nodes
**legitimately disagree** in the low bits, so the system's only correctness test is a
tolerance band — and a tolerance band cannot distinguish a benign kernel difference from
a byzantine node injecting bias just under the threshold. Equality testing is
*unavailable* to them even in principle (non-associativity), and the verified research in
`docs/STRAND-decode-parallelism-research.md` established the mirror statement: bit-exact
parallel decode is mathematically available only in integer arithmetic. The moat is the
precondition for attestation, not a tax. There is also no stable object for them to hash:
fused dequant-into-matmul backends never materialize a canonical weight tensor (§7).

### 2.4 The future unlock: integer activations (designed-not-built; a gate, not a promise)

If activations were quantized to Q8 and the MAC accumulated in i64
(`Σ q12·q8`: |w|≤2^15, |x|≤2^7, 18944 terms < 2^15 ⇒ |Σ| < 2^37 — i32 overflows, i64 is
comfortable), per-layer outputs would become integer ⇒ associative ⇒ schedule-invariant ⇒
**attestable layer-by-layer** with the same leaf machinery, and the swarm could attest
logits end-to-end. That is the real "deterministic swarm inference." It does not exist in
STRAND; its PPL cost is unmeasured; per will.md §5.1 it gets a smallest-decisive-
experiment (one layer, 0.5B, PPL delta) before any further design. Until that gate
passes, this document's claims stop at the weights plane.

---

## 3. Attestation protocol (built ON the provenance layer)

### 3.1 The hash scheme — implemented, not designed here

**Every canonical digest in this protocol is an SPV3 object from
`docs/STRAND-v3-provenance-spec.md`, reference-implemented in `provenance.rs`** on the
FIPS 180-4 SHA-256 already in the codebase (`sha256.rs`, pinned to NIST vectors). The
scheme hashes the **decoded Q12 stream** — deliberately: the wire already has
`source_sha256` ancestry (input `.safetensors`, format.rs:326), and what no float
quantizer can offer is a canonical digest of *the weights the model actually computes
with*. STRAND's decode is integer and byte-identical everywhere, so "the SHA-256 of the
decoded model" is a well-defined constant. The objects (*proven*, KAT-pinned):

| object | preimage | role in the swarm |
|---|---|---|
| **block leaf** | `SHA256("SPV3.BLK" ‖ block_index u64 LE ‖ n u32 LE ‖ q12[i] i32 LE …)` | the per-block truth; what challenges check |
| **tensor root** | `SHA256("SPV3.TNS" ‖ n_blocks u64 LE ‖ leaf[0] ‖ …)` | flat list over ALL leaves — full-tensor verification |
| **model root** | `SHA256("SPV3.MDL" ‖ n_tensors u64 LE ‖ (name_len u32 ‖ name ‖ tensor_root) …)` | the artifact version pin; equivocation kill |
| **self-test vectors** | k blocks, content-derived selection (`"SPV3.SEL"`), expected leaf each | zero-state onboarding smoke (§3.3 — NOT byzantine sampling) |

Properties this design leans on, all asserted by the module's tests: domain tags make the
preimage spaces disjoint; counts + explicit lengths make serialization injective; a
single flipped payload bit flips exactly the owning block's leaf and propagates to both
roots (`payload_bit_flip_flips_exactly_one_leaf_and_all_roots`); leaves bind index,
value, sign, order, and length; the model root binds names and tensor order.

Two structural facts the protocol must respect rather than redesign:
- **The tensor root is a flat hash list, not a Merkle tree** — recomputing it needs every
  leaf; single-block checks therefore check against the *leaf list*, and the leaf list's
  integrity comes from the signed manifest (§3.3). A Merkle tree would give O(log n)
  inclusion proofs; the spec chose the flat list, this doc does not relitigate it (the
  leaf list is ≤43% of wire size at k=2, less at higher k, and it lives verifier-side).
- **A leaf binds its block position but not its tensor** (the preimage has no name).
  Cross-tensor leaf substitution is closed by the manifest structure (leaves live under
  their tensor) and, in challenges, by `ShardDesc` in the envelope (§3.4).

### 3.2 What the Q12 plane does and does not pin — and the v3 requirements

**Does pin (semantic content):** any change to payload bits, side-info, `init_state` that
a decoder consumes, l/k/vec_dim geometry, affine/tail flags — all flow through
`decode_block_q12` into different Q12 bytes ⇒ different leaf. **Wire-byte integrity is
therefore checkable *through decode*: decode IS the checksum.** Two wire encodings that
decode to the same plane are semantically interchangeable for inference; file-byte
identity (mirror dedup, CDN resumption) may use ordinary file digests operationally, but
they carry no weight in the trust story and this doc defines none.

**Does NOT pin (interpretation metadata):** `rht_seed` and `shape` never enter the
decode. Two artifacts with identical Q12 planes but different `rht_seed` have the SAME
SPV3 roots yet compute **different effective weights** (the runtime RHTs `x` by the
stored seed — gemv.rs:12–17); a swapped `shape` reinterprets the same flat plane as a
different matrix. This is the one place the plane-first scheme under-covers, and the
protocol closes it at its own layer: **every attestation message binds the full
`ShardDesc` (§1.4), which carries seed/shape/geometry.** The durable fix belongs to the
spec:

| req | on `docs/STRAND-v3-provenance-spec.md` / the v3 archive | why |
|---|---|---|
| **R1** | the archive carries the per-tensor **leaf lists** as an mmap-able section, plus tensor roots + model root in a trailer | verifiers get expected leaves without a full reference decode; nodes self-check at load |
| **R2** | bind interpretation metadata — per-tensor `(shape, rht_seed, l, k, d, flags)` — under the signed root (e.g. in the model-root preimage or a covered descriptor table; spec's call, new domain tag if the layout changes) | the plane alone does not pin meaning (above) |
| **R3** | serialize + digest the **outlier channel** (§1.4 gap 2) | q2_l12_out1 reconstruction; integer ⇒ same equality story |
| **R4** | serialize + digest the **learned vector LUT** (§1.4 gap 1) | the 1.5-bit rung; the LUT is the decode — an unsigned LUT is an unsigned model |

Until R1–R4 land, the demo manifest (§8/M0) is a sidecar file marked DEMO-ONLY: same
SPV3 objects, provisional packaging. Migration is packaging, not re-hashing.

### 3.3 The reference manifest (bake time)

A `strand-swarm-manifest` post-pass over the baked v2 artifact emits, via the existing
library API (`block_hashes` / `tensor_root` / `model_root` / `make_test_vectors`):

- the **model root** (32 B) and per-tensor **roots + descriptors** (~tens of KB),
- the per-tensor **leaf lists** (32 B/block: ~813 MB for 7B@2-bit, ~45 MB for the 0.5B),
- k **self-test vectors** per tensor.

The manifest is signed by the producer (any detached signature; out of scope). The
producer computes it with the reference decode — and because decode is deterministic,
**the producer's leaf is *the* leaf**; that is the entire reason a reference manifest can
exist. Coordinator and verifiers hold it (roots always; leaf lists in full or fetched per
tensor); serving nodes hold shards and need none of it, though holding their own leaf
slice lets them self-audit at load.

**Self-test vectors are an onboarding smoke, not byzantine sampling**: the selection is
content-derived, hence *predictable* — an adversary keeps exactly those blocks correct
and corrupts the rest. They are for catching honest faults (bit rot, broken port) with
zero verifier state. Adversarial coverage comes from verifier-chosen uniform challenges
(§3.4/§4).

### 3.4 The challenges

Canonical truth = SPV3 objects only. Challenges wrap them in an **envelope** —
`SHA256(envelope_tag ‖ ShardDesc ‖ nonce ‖ body)` — which is transport framing for
freshness, explicitly non-canonical, never stored, and namespaced away from SPV3 tags
(proposed: `"SWRM.CH1"` / `"SWRM.CH2"`; 8-byte ASCII like the spec's, minted here as a
*protocol* constant, not a provenance object).

| id | proves | challenge | response body | verifier needs | cost (node) |
|---|---|---|---|---|---|
| **A2 decode** (primary) | the plane the node serves is the producer's, for the sampled blocks | `(nonce, tensor, [b0,b1))` uniform over the node's shards | the envelope over `leaf[b0] ‖ … ‖ leaf[b1−1]`, leaves computed with **global** indices via `decode_block_q12` (or cached) | the manifest leaf slice — nothing from the node, no second copy of the data | decode 256 w ≈ 0.34 µs/block at the *proven* 756 Mw/s single-thread + SHA-256 of 32 B/block |
| **A1 possession** (secondary) | the node holds the exact wire bytes now | same form | the envelope over `wire_bytes[b0,b1)` | a reference copy of those bytes (origin or replica) — deployment-dependent | hash of 86–160 B/block |
| **A3 activation** *(future)* | integer layer output | `(nonce, layer, x_q8)` | digest of integer output | gated on §2.4 | — |

Honest notes, in decreasing order of importance:
- **Every byte transfer in the swarm is verify-on-receipt**: a fetching node decodes each
  received range and checks leaves against the manifest before serving it. Corrupt bytes
  cannot propagate; bytes are never trusted, planes are. This — not A1 — is the load-
  bearing integrity mechanism, and it needs no reference copy.
- **A2 proves correct *output*, not *work*. ** A node may cache leaves (32 B/block) or the
  Q12 plane and answer without re-decoding — its choice; this is data-correctness
  attestation, not proof-of-storage-or-effort. The complementary risk (a node stores
  *only* leaves and serves garbage bytes) is closed by verify-on-receipt at every
  consumer of the bytes, and by A1 where a reference copy exists.
- The nonce kills replay; `ShardDesc` in the envelope kills cross-shard and cross-tensor
  substitution (a response for tensor T blocks [a,b) cannot answer T′ or [a′,b′)) and
  binds the interpretation metadata (§3.2); `model_root` inside `ShardDesc` kills
  stale-version answers.

### 3.5 Verification economics (the part that makes spot-checking real)

Per challenged block, the verifier holds the expected leaf (manifest) and needs nothing
else; the prover's cost is microseconds (table above). If the verifier wants to *localize*
a mismatch it fetches the block's wire bytes (≤160 B, page-aligned range), runs
`decode_block_q12`, and diffs 256 i32s — naming the first wrong weight. Full-shard audit
= decode at the *proven* 4.5 Gw/s rayon rate (a 1/8th-of-7B shard ≈ 0.8e9 w ≈ 0.2 s) +
leaf hashing (~3.3 GB of Q12 at ~2 GB/s hardware SHA ≈ 1.6 s) — *estimate-class*, M0
measures it; hashing can piggyback on a decode the node was doing anyway (§6). There is
no SNARK here and none is claimed: cheapness comes from determinism + block independence,
not succinct proofs.

### 3.6 Dispute resolution is mechanical

If node N's A2 response mismatches the manifest, ANY third device re-derives the truth:
fetch the block's bytes, `decode_block_q12`, `block_hash`, compare. No quorum, no
tolerance adjudication, no "run it three times and vote" — the leaf either matches or it
does not, and the re-derivation is the proof. A false accusation is equally impossible:
an honest node's leaf is reproducible by accuser, accused, and bystander, byte-for-byte
(`verify_test_vectors` is exactly this loop, *proven* against planted payload tampering,
tampered expected hashes, and hostile out-of-range indices — it returns false, never
panics). This collapse of dispute → re-execution is the single biggest practical
difference from float swarms (§7).

---

## 4. Failure and byzantine model

### 4.1 Trust model, stated honestly

This is **verifiable outsourcing with an honest verifier set**, not trustless consensus.
The producer (whoever ran the bake and signed the manifest) is trusted for model
*content* — attestation proves "you are serving the model the producer baked", it cannot
prove the producer baked a good model. Coordinators/verifiers are assumed honest-or-
auditable (anyone holding the manifest can audit them; the roots are 32 B each). There is
no token, no slashing economics, no sybil resistance designed here; those are policy
layers a deployment may add. What the protocol makes *impossible* is undetected shard
corruption — accidental or malicious — which is exactly the part float swarms cannot
test for.

### 4.2 Failure classes and their detectors

| failure | example | detected by | localization |
|---|---|---|---|
| bit rot / truncated fetch | flipped payload byte | verify-on-receipt; A2 sampling; self-test vectors at load | exact block (leaf), exact weight (256-i32 diff) |
| wrong decoder (stale binary, broken SIMD port, endianness bug) | a NEON port drifts | A2 vs manifest — the contract test pattern, in production | exact block; first wrong Q12 named |
| malicious weight substitution | biased weights under a real name | A2 (any Q12 change ⇒ leaf flip, p ≈ 1−2^−256) | exact block |
| **metadata substitution** | same plane, tampered `rht_seed`/`shape` | `ShardDesc` binding in every envelope (§3.2) — and nothing else, until R2 lands in v3 | immediate (desc mismatch) |
| outlier-channel tampering | (idx,val) edits | **NOT COVERED until R3** — named gap; v0 ships trellis-plane tensors or carries outliers under the demo manifest's sidecar digest | — |
| shard withholding / dead node | timeout | liveness timer, not a digest | n/a → reassign (§4.4) |
| equivocation (different bytes to different peers) | V1 to A, V2 to B | both verify against ONE model root; at most one version matches | immediate |
| stale artifact version | yesterday's bake | `model_root` in every `ShardDesc` | immediate |
| coordinator compromise | assigns all shards to colluders | out of scope (§4.1); mitigable: the manifest is public, any party can verify | — |
| **activation-plane fault** | garbage `y` rows | **NOT attestable today** (§2) — only cohort redundancy (§5.3) or downstream sanity | weak — honest gap |

### 4.3 Why a bad shard cannot hide

The byzantine node's fundamental problem: its output space is **discrete and
pre-committed**. Every block has exactly one correct 1 KiB Q12 string, whose leaf the
producer committed at bake time. There is no epsilon-ball to hide in (contrast: a
float-TP node can return `y + δ` with `δ` inside the comparison tolerance — undetectable
by design). Any deviation, in any single weight of any single block, flips the leaf.

Sampling math (standard, for sizing): a node corrupting fraction `f` of its blocks
survives `s` uniform A2 challenges with probability `(1−f)^s`; `s = 300` catches `f = 1%`
with ~95% probability per epoch. Challenges cost microseconds (§3.4), so `s` can be
large; full-shard audits (~2 s, §3.5) are affordable at onboarding and on any suspicion.

### 4.4 Reassignment

Shards are **stateless** with respect to the artifact: reassignment = the replacement
node range-GETs the byte ranges (page-aligned, resumable, from origin or any replica —
the source does not need to be trusted, verify-on-receipt covers it), passes its
self-test vectors, answers one A2 challenge, and is live. Nothing needs migrating from
the failed node. The honest cost: any **KV cache** the failed node held for in-flight
sequences is float, non-attestable, and simply lost — affected sequences replay their
prefix through the replacement (recompute = the prefix forward pass for that stage only).
Designs that "migrate" KV caches across heterogeneous devices would import the
float-equality problem; we decline.

---

## 5. Using the shards: inference topology (the non-attestable tier, designed for least damage)

### 5.1 Row-parallel within a tensor: concatenation, never reduction

`matvec_named_par` (gemv_par.rs:298) already computes `y` rows independently, each row
the same left-to-right scalar MAC as the single-threaded path (bit-identical `y` proven
on one machine, `matvec_named_par_matches_fast_via_v2`). Sharding a tensor by **rows**
(row-aligned block ranges, §1.3) means each node computes its `y` slice from the full
`x`, and the combiner **concatenates** — the swarm adds **zero new float reassociation**:
every `y[o]` is one node's pinned scalar MAC, the same op sequence a single machine runs.
Column-sharding (splitting `in_features`) would require a cross-node float **sum** —
order-dependent, kernel-dependent — and is rejected for exactly that reason. This is the
design's second load-bearing choice (after the block unit): *the only cross-node
combining operations in the swarm are integer equality and byte concatenation.*

Each row-shard node needs the full `x` for that projection and applies RHT(x) locally
when `rht_seed != 0` (gemv.rs:12–17 — the activation-side transform is the caller's job;
duplicated per node; O(n log n) f32 butterfly). Float, ergo unattested, ergo §5.3.

### 5.2 Layer-pipeline across tensors

Above the tensor level the swarm is a conventional pipeline (a node owns the full
projection set of a contiguous layer range — attention math, norms, residuals stay
node-local), optionally row-splitting the fat tensors (`down_proj`, `gate/up`) across
sub-nodes. Nothing novel is claimed for the pipeline tier; it is what petals/exo already
do, minus their scale maturity (§7).

### 5.3 Cohort pinning (the honest knob for float reproducibility)

Nodes advertise a `kernel_id = (crate version, target triple, scalar-MAC path id)`.
Within a cohort of equal `kernel_id`, pinned scalar f32 inference is *expected* to be
bit-reproducible (unmeasured across arches — demo gate G4 turns this into a measurement;
will.md §5.1, measure don't model). Redundant execution + equality inside a cohort is
then a cheap *fault* detector for the activation plane — but it is NOT byzantine-proof
(a malicious node lies about its cohort or colludes) and is NOT attestation. It is
labeled in every API as `reproducibility: cohort-pinned`, never `attested`.

---

## 6. dismantle integration seam

dismantle is the first consumer (will.md §1: STRAND is THE format for it), and it is
untouchable until its uncommitted work settles — so the seam is designed to live entirely
on our side of the line:

1. **Shared artifact, shared loader.** A swarm node embeds the same `StrandModel` mmap
   loader + `decode_q12_par` the dismantle wiring recipe uses
   (`STRAND-dismantle-wiring.md` steps 3/6/7). A shard-serving node is `StrandModel` over
   a *partial* file (its fetched page ranges) + the `encoded_block_range` addition
   (§1.2/M1). No dismantle code is needed to BE a node.
2. **New crate, additive:** `crates/strand-swarm` (manifest tool + challenge client +
   daemon), depending only on `strand-quant` (which now exports `provenance`) +
   `strand-decode-kernel`. dismantle never links it; they meet only at artifact bytes.
3. **The seam itself:** dismantle's `ensure_strand_cache` (wiring step 6) today resolves
   a local `.strand` path. The swarm-aware version resolves *(model_root → byte ranges)*
   from peers instead of disk, verifying each range on receipt (§3.4) — i.e. the seam is
   **"mmap a file" becomes "mmap a verified, range-fetched file"**, one function
   boundary, zero changes to decode/GEMV. Verification piggybacks on work the engine does
   anyway: the first decode of a fetched tensor emits leaves nearly free (the marginal
   cost over the decode it already runs is the SHA-256 of the Q12 stream).
4. KV/scheduler/pipeline integration inside dismantle is explicitly NOT designed here
   (engine territory, owner's call). Delivery follows the house pattern: a ready-to-apply
   recipe appended to `STRAND-dismantle-wiring.md` once Tier 3a has a measured demo — not
   before.

---

## 7. petals / exo / llama.cpp-RPC — what they cannot do, what we cannot do

| capability | petals | exo | llama.cpp RPC | STRAND swarm (this design) |
|---|---|---|---|---|
| serve 70B+ across volunteer/LAN devices TODAY | **yes** | **yes (LAN)** | yes | **no — design + demo plan** |
| schedulers, rebalancing, fault-tolerance at scale | **yes** | partial | partial | no (designed, unbuilt) |
| GPU-class throughput | **yes** | yes | yes | no (CPU decode + f32 GEMV; GPU decode measured ALU-bound and parked) |
| canonical weight plane two devices can compare | no¹ | no¹ | no¹ | **yes — Q12, schedule-invariant, proven; SPV3 digests implemented + KAT-pinned** |
| equality-testable shard correctness (no tolerance band) | no | no | no | **yes** |
| per-block (256-weight) independently verifiable units | no (layer-granular blobs) | no (layer/memory-fraction partitions) | no | **yes (`BlockOffsetRecord` + `decode_block_q12` + SPV3 leaf)** |
| mechanical dispute resolution (any third device re-derives) | no | no | no | **yes** |
| provenance wired into format + library | no | no | no | **yes (v2 `source_sha256` ancestry; SPV3 decoded-plane layer landed; archive packaging = R1–R4, pending)** |
| float-free node (MCU/WASM as verifier or shard host) | no | no | no | **yes (decode is integer; WASM build exists)** |
| attest final logits/tokens | no | no | no | **no — same as them; honesty §2** |

¹ Not carelessness: their backends (cloud-GPU/MLX/tinygrad/Metal) fuse dequant-into-matmul and
never materialize a canonical weight tensor; dequant scales are f16/f32; different
backends legitimately produce different bits. There is no stable object to hash even if
they wanted one. Petals' published trust posture is redundancy + reputation with
tolerance comparison; exo targets cooperative LANs and does not claim adversarial
robustness. **The gap is structural (float, fused), not engineering they forgot.**
Conversely: the first three rows are real systems engineering they have and we do not —
v0 does not compete on serving; it competes on *what can be proven about the bytes being
served*.

The honest niche: deployments where "which weights, exactly, is this fleet running?"
must have a checkable answer — regulated/audited inference, reproducible research fleets,
mixed-hardware edge fleets, untrusted shard suppliers. That niche is narrower than
"decentralized inference" and the doc says so.

---

## 8. Minimal 2-node demo (Mac + RunPod) — plan and effort

**Goal: the smallest experiment that decides the fork** (will.md §5.1): do two unrelated
architectures (M3 Pro aarch64-darwin, RunPod 3090 host x86_64-linux) produce
byte-identical SPV3 manifests from disjoint shards of one artifact, and does a planted
fault localize to the exact block? No daemon, no networking stack — transport is the
existing direct SSH (`-p 40078 root@213.192.2.110`; port drifts on pod restart) + `scp`.
Constraints honored: all local work `nice -n 19`; no long benchmarks (the box runs the
marathon); pod work stays in `/workspace`; this branch does not commit.

The verification core **already exists** (`decode_block_q12`, `block_hashes`,
`tensor_root`, `model_root`, `make_test_vectors`, `verify_test_vectors` — tests green),
which is why the estimate is down from rev 1's 2.5–4 days:

| milestone | contents | size | effort |
|---|---|---|---|
| **M0** `strand-swarm-manifest` bin | walk a `.strand` v2 (`StrandModel::encoded_tensor` per tensor) → emit model root, tensor roots + descriptors, leaf lists, self-test vectors. Sidecar files marked DEMO-ONLY until R1/R2 land in the v3 archive | ~150 LOC (the library does the hashing) | 0.5 d |
| **M1** `encoded_block_range(name, b0, b1)` in loader.rs | the §1.2 enabler: truncated `EncodedTensor` over a byte-sliced payload + bit-identity test (range decode == full-decode slice; k∈{2,3,4}, tail/affine, RAGGED, short-final-block) | ~80 LOC + test | 0.5 d |
| **M2** two-node manifest run | artifacts: the e2e synthetic STRICT multi-tensor model (e2e.rs builder — seconds, both nodes) + the real 0.5B v2 archive (RAGGED → tensor-unit shards). Mac takes shard set S_A, pod S_B; exchange manifests over scp; compare | ~120 LOC scripts | 0.5 d |
| **M3** challenge drill | nonce'd A2 envelopes over ssh round-trips; planted faults on B's copy: flip 1 payload byte → exact (tensor, block) named by both sides independently; flip 1 side-info byte → same; flip `rht_seed` in the descriptor → caught ONLY by ShardDesc binding (demonstrates §3.2/R2 in the flesh) | ~100 LOC | 0.5 d |
| **(stretch) G4** | row-split `matvec_named_par` on both nodes, concatenate, compare f32 bits vs single-node reference — the §2.2/§5.3 cross-arch reproducibility *measurement* (explicitly allowed to fail; a failure is the lesson, recorded in §2.1) | ~80 LOC | 0.5 d |

**Total: ~2–3 days** focused, all inside a new `strand-swarm` crate + one loader
addition; zero changes to encode/decode/format/provenance; zero dismantle changes.

Demo gates:
- **G1** SPV3 manifests for the same shards byte-identical across aarch64/x86_64 — the
  moat demonstrated cross-vendor over a real network boundary for the first time.
- **G2** planted fault → exact block named by both sides independently; metadata flip →
  caught by descriptor binding.
- **G3** A2 nonce round-trip; stale-nonce replay rejected; out-of-range index rejected
  without panic (the `verify_test_vectors` hostile-input behavior, exercised remotely).
- **G4** (stretch) cross-arch f32 pinned-MAC equality: measured yes/no.

Dependency: the v3 provenance spec doc is being written in this same wave; M0's sidecar
packaging migrates to its archive layout when R1–R4 land (same objects, same hashes —
packaging only). The v2-forked-3-ways lesson applies (will.md §5.8): one hand reconciles
this doc's R1–R4 against the spec at merge.

---

## 9. Open questions (ranked)

1. **R1–R4 reconciliation** with `docs/STRAND-v3-provenance-spec.md` (§3.2) — one owner
   merges the requirements; R2 (metadata binding) is the only one that touches the trust
   story, the rest are packaging/coverage.
2. **Integer activation gate (§2.4)** — the one experiment that upgrades "attested
   weights" to "attested inference": Q8 activations + i64 MAC on one 0.5B layer, PPL
   delta. Cheap, decisive, unscheduled.
3. **Outlier channel + vector LUT in the wire (R3/R4)** — blocks the 2-bit flagship
   (q2_l12_out1) and the 1.5-bit rung from swarm serving until v3 closes them; the
   trellis plane of every rung is serveable today.
4. **Does anyone need this?** Honest market question. The attestable-fleet niche (§7) is
   real but narrow; the demo is cheap enough (≤3 days) to build before answering, and the
   artifact — cross-arch determinism proven over a network boundary — strengthens the
   core product story even if no swarm ships.
5. KV-replay cost model for reassignment under long contexts (§4.4) — measure when a
   pipeline exists, not before.
