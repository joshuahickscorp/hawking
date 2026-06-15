# `.strand` v2 — the GPU-random-access deploy format (spec)

_Status: DESIGN (doc only). v1 is shipped and stays the reference / round-trip format.
v2 is **additive** to `crates/strand-quant/src/format.rs` — it does not change v1, the
quantizer, or the integer decode contract. The encoder already produces every field this
format needs (`EncodedTensor` + `BlockMeta`); v2 is a **layout change**, not new math._

---

## 1. Why v2 exists (the one-paragraph gate)

`.strand` v1 is a **sequential per-tensor stream**: each tensor stores `bits` (one
LSB-first concatenation of every block's `k`-bit symbols) followed by a vector of
`BlockMeta`. To decode the tensor you walk it front-to-back, advancing a single
`bit_cursor` across all blocks. That is ideal for CPU round-trip and `read_strand`, and
**useless for a Metal GEMV kernel**, which must jump directly to *any* `(row, block)` —
a threadgroup owns one output row and splits that row's blocks across its threads
(`docs/STRAND-metal-decode-gate.md` §"Metal GEMV kernel design"). Without a precomputed
offset, the kernel would have to re-scan every prior block just to find a block's first
bit, which serialises the parallel-over-blocks design and re-introduces the exact
compute-bound stall that killed dismantle's Q3_K kernel.

**v2 = v1's data + a per-tensor BLOCK-OFFSET TABLE.** One fixed-size record per block —
`{ bit_offset: u64, init_state: u32, scale_q: i32 }` — laid out contiguously and
**page-aligned (4 KiB)** so the table and the bitstream payload are both `mmap`-ready and
a kernel computes a block's byte address with pure arithmetic (no scan, no pointer chase).
This is the "bake the layout as the zero-cost default archive" that dismantle's
`paradigmshift.md` Part IV asks for.

What v2 deliberately does **not** change:

* **The bitstream is byte-identical to v1's** `enc.bits` per tensor. v2 only adds an index
  *over* it and pads it; it never re-encodes a symbol. `decode_tensor_fixed` on a tensor
  reconstructed from v2 returns the same `Vec<i32>` as from v1.
* **`init_state` semantics under tail-biting are unchanged.** The table still carries
  `init_state` for every block (the encoder always records it in `BlockMeta`, even when it
  is not on v1's wire). For a tail-bitten block the *seek* uses the table's `bit_offset`,
  and the decoder recovers the start state from the block's own trailing symbols exactly as
  in v1 — so the stored `init_state` is **advisory / non-load-bearing** for tail-bitten
  blocks (see §6.2). It is authoritative for non-tail-bitten blocks.

---

## 2. Conventions

* **Endianness:** all multi-byte scalars are **little-endian** (matches v1).
* **Bit order inside the payload:** symbols are packed **LSB-first** within each byte,
  identical to v1 / `trellis::push_bits` / `read_bits`. A `bit_offset` is an offset in
  **bits** from the start of *that tensor's* payload region (not from file start) — see
  §5.3 for why this keeps the table tensor-relocatable.
* **Block:** a contiguous run of `cfg.block_len` weights sharing one `BlockMeta`. With the
  shipped `for_bpw` configs `block_len == 256`; v2 does **not** hardcode 256 — it stores
  `block_len` per tensor and derives the table length from it. The final block of a tensor
  may be short (`BlockMeta::n < block_len`); its record is still present.
* **"row" / "block within row":** weights are flattened **row-major** `[out_features,
  in_features]` (encode.rs:340: _"weight `[out, in]` flattened row-major … `in` … columns"_;
  `strand-decode-kernel::matvec` reads `w[o*in_features .. (o+1)*in_features]`). Global
  weight index `g = row * in_features + col`. When `in_features % block_len == 0` (the
  deploy invariant — see §7), block `(row, b)` is global block index
  `row * (in_features / block_len) + b`. This linear map is the whole point of the table.
* **Alignment primitives:** `PAGE = 4096`. `align_up(x, a) = (x + a - 1) & !(a - 1)`.
  Padding bytes are **zero**.

---

## 3. File layout (top level)

```text
┌────────────────────────────────────────────────────────────────────┐
│ FILE HEADER            (fixed 56 bytes, then padded to PAGE)        │
│   magic            [u8;4] = b"STR2"                                 │
│   version          u32 = 2                                          │
│   header_bytes     u32        # bytes from file start to end of the │
│                               #   tensor-descriptor array (i.e. to  │
│                               #   the first padding byte before the │
│                               #   first table). Lets a reader skip  │
│                               #   straight to payload region 0.     │
│   n_tensors        u32                                              │
│   flags            u32        # file-level flags (bit0 = all tensors│
│                               #   satisfy the deploy invariant §7)  │
│   source_sha256    [u8;32]    # provenance, copied from v1 intent   │
│   reserved         [u8;4] = 0                                       │
├────────────────────────────────────────────────────────────────────┤
│ TENSOR DESCRIPTOR ARRAY   (n_tensors × TensorDescriptorV2,          │
│                            variable-length because of the name)     │
│   …descriptor 0…                                                    │
│   …descriptor 1…                                                    │
│   …                                                                 │
│  [pad with 0x00 up to the next PAGE boundary]                       │
├════════════════════════════════════════════════════════════════════┤
│ TENSOR 0 REGION            (PAGE-aligned start)                     │
│   OFFSET TABLE 0    n_blocks_0 × BlockOffsetRecord (16 B each)      │
│     [pad 0x00 to PAGE]                                              │
│   PAYLOAD 0         the v1 bitstream bytes for tensor 0, verbatim   │
│     [pad 0x00 to PAGE]            (so SIDE-INFO / next table align)  │
│   SIDE-INFO 0       per-block sub_scales + mins blobs (see §5.4)    │
│     [pad 0x00 to PAGE]                                              │
├────────────────────────────────────────────────────────────────────┤
│ TENSOR 1 REGION            (PAGE-aligned start)                     │
│   OFFSET TABLE 1 … PAYLOAD 1 … SIDE-INFO 1 …                        │
├────────────────────────────────────────────────────────────────────┤
│ … TENSOR n-1 REGION …                                              │
└────────────────────────────────────────────────────────────────────┘
```

Every **bold boundary** above (`╪`/`═`) is a 4 KiB page boundary. The two things a kernel
touches in its hot loop — **OFFSET TABLE** and **PAYLOAD** — therefore each begin on their
own page, so an `mmap` hands the GPU a pointer it can use without re-copy or re-alignment.

---

## 4. The block-offset table (the core of v2)

### 4.1 `BlockOffsetRecord` — 16 bytes, fixed, naturally aligned

```text
struct BlockOffsetRecord {            // 16 bytes
    bit_offset : u64,   // LSB-first bit position of this block's FIRST symbol,
                        //   measured from the start of this tensor's PAYLOAD region.
    init_state : u32,   // BlockMeta.init_state (start state). Authoritative for
                        //   non-tail-bitten blocks; advisory for tail-bitten (§6.2).
    scale_q    : i32,   // BlockMeta.scale_q (Q16 super-scale) hoisted here so the
                        //   kernel reads it without parsing SIDE-INFO. The eff per-
                        //   sub-block scale is still (scale_q * (code+1)) >> 6.
}
```

* **Why 16 bytes, in this order:** `8 + 4 + 4` packs with no internal padding and the whole
  record is 16-byte aligned when the table starts on a page — so a Metal thread loads it as
  a single `uint4` / `device const BlockOffsetRecord*` index, one coalesced 16-byte read.
* **Why `bit_offset` is `u64`:** a single 7B tensor row stream is small, but the *payload of
  one tensor* (e.g. a 4.3 M-weight matrix at 3 bits) is ~1.6 MB ≈ 13 M bits — fits in u32,
  but `u64` removes any ceiling for fat MoE/embedding tensors and keeps the record a clean
  power-of-two size. The high bits are simply zero on small tensors.
* **`bit_offset` is monotonic** in block order and equals the running symbol-bit count:
  for block `b`, `bit_offset[b] = Σ_{j<b} symbol_bits(block_j)` where
  `symbol_bits(block) = num_steps(block.n) * k`, and `num_steps(n) = ceil(n / vec_dim)`
  (`= n` for the scalar trellis `vec_dim == 1`; `ceil(n/d)` for the vector trellis). This is
  exactly the `bit_cursor` v1's decoder accumulates — v2 just **precomputes and stores it**.
  `bit_offset[0] = 0`. A sentinel `bit_offset[n_blocks]` (= total payload bits) is **not**
  stored; the last block's length comes from its `n` (§4.3).

### 4.2 Table placement & size

For tensor `t` with `n_blocks` blocks:

```
table_bytes(t)      = n_blocks * 16
table_padded(t)     = align_up(table_bytes(t), PAGE)
```

The table starts at the tensor region's page-aligned base; the payload starts at
`region_base + table_padded(t)` (also page-aligned, because `table_padded` is a PAGE
multiple). A reader/kernel locates record `b` at
`region_base + b * 16` and reads its first symbol at payload byte
`payload_base + (rec.bit_offset >> 3)`, bit `(rec.bit_offset & 7)`.

### 4.3 Deriving a block's symbol count without a sentinel

The kernel needs to know how many symbols (hence how many weights) a block emits. It does
**not** read this from the table; it computes it:

* All blocks except the last in a tensor have `n == block_len` (a stored per-tensor field),
  so `num_steps = ceil(block_len / vec_dim)` symbols.
* The **last** block has `n_last = total_weights - (n_blocks - 1) * block_len`, so
  `num_steps_last = ceil(n_last / vec_dim)`.

Both are pure arithmetic from `(total_weights, block_len, vec_dim, b)`, all of which are in
the tensor descriptor (§5). Keeping `n` out of the per-block record is what holds the record
at a tight 16 bytes; the only block whose `n` differs is the last, and it is recomputed.

---

## 5. Tensor descriptor (v2)

### 5.1 `TensorDescriptorV2` byte layout

```text
struct TensorDescriptorV2 {
    // ---- identity (same fields as v1's TensorEntry, kept for round-trip parity) ----
    name_len        u32
    name            [u8; name_len]      // UTF-8
    shape_ndim      u32
    shape           [u64; shape_ndim]   // row-major; shape[0]=out_features, shape[1]=in_features (2-D)
    rht_seed        u64                 // 0 = no RHT (applied to the ACTIVATION at GEMV time)
    trellis_l       u8                  // L (state bits); num_states = 2^L
    trellis_k       u8                  // k (bits/symbol)
    vec_dim         u8                  // d; 1 = scalar trellis
    flags           u8                  // bit0 TAIL_BITING, bit1 AFFINE_MIN, bit2 HAS_RHT (v1 flag bits)

    // ---- v2 layout fields (new) ----
    block_len       u32                 // weights per block (256 for shipped configs)
    total_weights   u64                 // == EncodedTensor.total; gives n_last and n_blocks
    n_blocks        u64                 // == EncodedTensor.blocks.len() (redundant w/ total_weights+block_len, stored for O(1) seek)
    reserved        u32 = 0
    reserved2       u32 = 0

    // ---- region pointers (absolute file byte offsets; all PAGE-aligned) ----
    table_offset    u64                 // file offset of OFFSET TABLE t (page-aligned)
    payload_offset  u64                 // file offset of PAYLOAD t      (page-aligned)
    payload_bytes   u64                 // ceil(total_payload_bits / 8) == enc.bits.len()
    sideinfo_offset u64                 // file offset of SIDE-INFO t    (page-aligned; 0 if none)
    sideinfo_bytes  u64                 // 0 if the tensor has neither sub-scales nor mins
}
```

All descriptors are concatenated in the header (§3). They are variable-length only because
of `name`; a reader walks them sequentially once at load to build an in-memory index, then
uses the **absolute** `*_offset` fields to `mmap`/seek each region directly.

### 5.2 Field provenance (everything comes from the existing encoder)

| descriptor field | source in v1 types |
|---|---|
| `name, shape, rht_seed, trellis_l/k, vec_dim, flags` | `PackedTensor` (identical to v1) |
| `block_len` | `cfg.block_len` (carried alongside the encode; see §8 note) |
| `total_weights` | `EncodedTensor.total` |
| `n_blocks` | `EncodedTensor.blocks.len()` |
| `payload_bytes` | `EncodedTensor.bits.len()` |
| per record `bit_offset` | running `Σ num_steps(n)*k` over `blocks` |
| per record `init_state` | `BlockMeta.init_state` |
| per record `scale_q` | `BlockMeta.scale_q` |
| SIDE-INFO | `BlockMeta.sub_scales` + `BlockMeta.mins` (§5.4) |

### 5.3 Why `bit_offset` is tensor-relative, not file-absolute

Records store `bit_offset` relative to `payload_offset` (the tensor's payload base), and the
descriptor stores `payload_offset` as an absolute file offset. The kernel adds them once:
`addr = payload_offset + (bit_offset >> 3)`. Keeping the per-block value tensor-relative
means (a) it fits in fewer bits / never depends on where the tensor lands in the file, and
(b) a future tool can relocate or stream a single tensor's `{table, payload}` pair without
rewriting `n_blocks` u64s. The single absolute `payload_offset` per tensor is the only thing
that changes on relocation.

### 5.4 SIDE-INFO region (sub-scales + affine-min codes)

The per-sub-block 6-bit `sub_scales` and (optionally) `mins` are **not** needed to *seek*,
but the GEMV does need them to form the effective scale `eff = (scale_q*(code+1))>>6` and the
offset `eff_min_q(min_base_q, code)`. v2 stores them in a third, page-aligned region so the
kernel can read them with the same coalesced-access discipline:

```text
SIDE-INFO t  (only present if flags has AFFINE_MIN, or any sub_scales are non-unity):
  SUB-SCALES half:
    for b in 0..n_blocks:
        sub_scales[b]   : ceil(6 * n_sub(b) / 8) bytes   # n_sub(b)=ceil(n_b/SUB_BLOCK), SUB_BLOCK=32
  [if AFFINE_MIN] MINS half (begins page-/8-aligned right after the sub-scales half):
    for b in 0..n_blocks:
        min_base_q[b]   : i32            # BlockMeta.min_base_q is PER-BLOCK (encode.rs:486/613)
    for b in 0..n_blocks:
        mins[b]         : ceil(6 * n_sub(b) / 8) bytes
```

Because every full block has the same `n_sub = block_len/32` (= 8 for block_len 256), the
sub-scale stride is **constant** for all but the last block: a kernel indexes block `b`'s
sub-scale bytes at `b * stride` (`stride = ceil(6 * (block_len/32) / 8)`), and the last block
is the tail. The `mins` half (if present) begins at `sideinfo_offset + align_up(n_blocks*stride, 4)`
and stores **per-block** `min_base_q` (an `i32` each — it is computed per block by
`choose_affine_min`, NOT constant per tensor) followed by the packed 6-bit `mins` codes, so
the offset `eff_min_q(min_base_q[b], code)` uses the right base for block `b`. (For the
**3-bit deploy default**, `AFFINE_MIN` is OFF — see the project verdict — so most deploy
tensors carry only the sub-scales half; the whole `mins`/`min_base_q` region is absent and
the GEMV skips the offset add entirely.)

---

## 6. Worked example: how the Metal kernel indexes `(row, block) → bit_offset`

Take an attention output projection `W ∈ [out=896, in=896]`, `block_len=256`, scalar trellis
(`vec_dim=1`, `k=3` for the 3-bit deploy default). `in_features=896`, **but** `896 % 256 ≠ 0`
— so this tensor would be rejected by the strict deploy invariant (§7) and either repacked
with `block_len=128` (896 = 7·128) or marked `ROW_RAGGED`. Use instead a clean case
`in=1024`, `block_len=256` ⇒ `blocks_per_row = 1024/256 = 4`, and (illustratively) `out=2048`
rows ⇒ `n_blocks = 2048*4 = 8192`.

The GEMV computes `y[o] = ⟨W_rht[o], x_rht⟩` (RHT folded into the activation once per GEMV,
so no per-row inverse-RHT — see the gate doc §"RHT moves to the activation"). One threadgroup
owns row `o = threadgroup_position_in_grid`; its threads split the row's 4 blocks.

```c
// ---- host-side, once: from the TensorDescriptorV2 ----
uint  blocks_per_row = in_features / block_len;          // = 4
device const BlockOffsetRecord* TBL =                    // page-aligned table base
        (device const BlockOffsetRecord*)(base + desc.table_offset);
device const uint8_t*           PAY = base + desc.payload_offset;   // page-aligned payload
device const uint8_t*           SUB = base + desc.sideinfo_offset;  // page-aligned side-info
uint  sub_stride = ceil_div(6 * (block_len/32), 8);      // bytes of sub_scales per full block

// ---- per (row, block) in the kernel ----
uint row = tg_pos;                                       // 0..out_features
for (uint b = lane; b < blocks_per_row; b += threads_per_tg) {
    uint  gblock   = row * blocks_per_row + b;           // <-- the linear (row,block) map
    BlockOffsetRecord rec = TBL[gblock];                 // ONE coalesced 16-byte load
    ulong bitpos   = rec.bit_offset;                     // bits, relative to PAY
    device const uint8_t* p = PAY + (bitpos >> 3);       // byte address of first symbol
    uint  shift    = (uint)(bitpos & 7);                 // intra-byte LSB-first start
    uint  state    = rec.init_state;                     // start state (tail-biting: re-derive)
    int   scale_q  = rec.scale_q;                         // Q16 super-scale

    uint  col0     = b * block_len;                      // first activation column for this block
    // sub-scale code stream for this block (6-bit packed), constant stride for full blocks:
    device const uint8_t* sub = SUB + (ulong)gblock * sub_stride;

    // ... aligned 32-bit-word symbol reads from p, state machine, shmem LUT lookup,
    //     eff = (long)scale_q * ((code+1)) >> 6, w = ((long)eff * q) >> 16  (native 32x32->64),
    //     partial += w * x_rht[col0 + j];  // per the lean inner loop in the gate doc
}
// shmem-reduce partial across the threadgroup -> y[row]
```

The entire address computation for any `(row, block)` is
`TBL[row*blocks_per_row + b]` → `PAY + (bit_offset>>3)`: **two adds, one shift, one 16-byte
load — no scan.** That is the property v1 could not give and the reason v2 exists.

**Tail-biting note for the kernel:** if `flags & TAIL_BITING`, the kernel ignores
`rec.init_state` and recovers the start state by pre-scanning the block's `num_steps`
trailing symbols into the register (the v1 rule, `decode.rs:155–166`), then replays. The
seek itself is unaffected — `bit_offset` still points at the block's first symbol.

---

## 7. Deploy invariant & validation

The clean `(row, block)` arithmetic in §6 holds **iff** `in_features % block_len == 0`
(each row is an integer number of blocks and no block straddles a row boundary). The writer
**must** enforce or record this:

* **STRICT (default for deploy):** require `in_features % block_len == 0` for every 2-D
  tensor. If a tensor fails, the writer returns an error naming the tensor; the caller may
  re-quantize that tensor with a `block_len` that divides `in_features` (e.g. 128 for
  896-dim Qwen tensors: 896 = 7·128). 1-D tensors (norms, biases) and tensors quantized
  whole are single-stream and exempt (their table is `blocks_per_row`-agnostic — the kernel
  treats the whole tensor as "row 0").
* **RAGGED (opt-in, file flag bit clear):** allow `in_features % block_len != 0`. Then the
  `(row, block)` map is **not** linear and the kernel must instead read a small per-row
  `row_block_base[out_features]` prefix array (block index of each row's first block),
  stored as an extra page-aligned region. This is documented for completeness but the 3-bit
  deploy path should prefer STRICT + a divisor `block_len`, so the hot kernel stays
  branch-free. (The file-level `flags` bit0 in §3 records which mode the archive is in.)

A v2 reader **validates** on load: magic `STR2`, version `2`, every `*_offset` is
page-aligned and within file bounds, `table_offset + n_blocks*16 ≤ payload_offset`,
`payload_offset + payload_bytes ≤ sideinfo_offset` (or EOF/next region), and the
last record's `bit_offset + num_steps_last*k == total_payload_bits`. Any failure ⇒ `Err`.

---

## 8. Rust signatures (additive to `format.rs` — do NOT change v1)

```rust
// ---- new constants (alongside the existing MAGIC/VERSION) ----
pub const MAGIC_V2:   &[u8; 4] = b"STR2";
pub const VERSION_V2: u32      = 2;
pub const PAGE:       usize    = 4096;

/// File-level flags for the v2 header.
pub mod flags_v2 {
    /// All 2-D tensors satisfy `in_features % block_len == 0` (the STRICT deploy
    /// invariant); the `(row, block)` map is linear and the kernel needs no per-row
    /// base array. Clear ⇒ at least one RAGGED tensor (see §7).
    pub const ALL_STRICT: u32 = 1 << 0;
}

/// 16-byte per-block seek record. `#[repr(C)]` + field order give a padding-free,
/// 16-byte-aligned record the Metal kernel reads as one coalesced load. All fields
/// little-endian on disk.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockOffsetRecord {
    /// LSB-first bit position of this block's first symbol, relative to the tensor's
    /// payload base. Monotonic; equals the running `Σ num_steps(n)*k`.
    pub bit_offset: u64,
    /// `BlockMeta::init_state`. Authoritative unless the tensor is tail-bitten.
    pub init_state: u32,
    /// `BlockMeta::scale_q` (Q16) hoisted out of side-info for one-load access.
    pub scale_q: i32,
}

/// What the writer needs per tensor: the same borrow as `PackedTensor`, plus the
/// `block_len` used to encode it (so the table length and the STRICT check are exact).
/// `block_len` is otherwise not on `PackedTensor`; the caller has it from the `TrellisConfig`.
pub struct PackedTensorV2<'a> {
    pub base: PackedTensor<'a>,
    pub block_len: u32,
}

/// Serialize tensors into a `.strand` **v2** archive: v1's data + per-tensor
/// page-aligned block-offset table (`STR2`). This is the deploy/mmap format the
/// Metal GEMV kernel seeks into; `write_strand` (v1) remains the reference format.
///
/// `strict` selects the deploy invariant (§7): `true` ⇒ error if any 2-D tensor has
/// `in_features % block_len != 0`; `false` ⇒ emit RAGGED with per-row base arrays.
/// Returns the archive bytes (the caller writes them to disk), or an `Err` naming the
/// first tensor that violates STRICT.
pub fn write_strand_v2(
    tensors: &[PackedTensorV2],
    source_sha256: [u8; 32],
    strict: bool,
) -> Result<Vec<u8>, String>;

/// One tensor read back from a v2 archive. Mirrors `OwnedTensor` and additionally
/// exposes the reconstructed offset table, so a host can hand `(table, payload,
/// sideinfo)` slices straight to the GPU. The `enc`/`shape`/seed/`l`,`k`,`vec_dim`
/// fields decode bit-identically to the v1 `OwnedTensor` for the same tensor.
#[derive(Clone, Debug)]
pub struct OwnedTensorV2 {
    pub base: OwnedTensor,         // name, shape, rht_seed, l/k/vec_dim, enc (round-trips to v1)
    pub block_len: u32,
    pub table: Vec<BlockOffsetRecord>,   // n_blocks records, reconstructed from the file
}

/// Read a `.strand` v2 archive. Validates magic/version/alignment/bounds (§7) and
/// reconstructs each tensor's `EncodedTensor` (so `decode_tensor_fixed` on the result
/// equals decoding the v1 archive) **and** its `Vec<BlockOffsetRecord>`. Round-trip
/// guarantee: `read_strand_v2(write_strand_v2(ts)).enc == ts.enc` for every tensor,
/// and the recomputed `bit_offset[b]` matches the running cursor `read_strand` walks.
pub fn read_strand_v2(buf: &[u8]) -> Result<Vec<OwnedTensorV2>, String>;

/// (Convenience, optional) Transcode a shipped v1 archive to v2 without touching the
/// quantizer: read with `read_strand`, recompute offsets, write `STR2`. Lets the
/// deploy artifact be produced from an existing `.strand` v1 file.
pub fn strand_v1_to_v2(
    v1: &[u8],
    block_lens: &[u32],            // block_len per tensor, in file order (from the encode config)
    strict: bool,
) -> Result<Vec<u8>, String>;
```

### 8.1 Writer algorithm (per tensor), in words

1. Emit the FILE HEADER (with a placeholder `header_bytes`/region offsets), then every
   `TensorDescriptorV2` with placeholder `table_offset`/`payload_offset`/`sideinfo_offset`.
2. Pad to PAGE. For each tensor, in order:
   a. record `table_offset = current_len`; walk `enc.blocks` accumulating
      `bit_offset` (start 0, add `num_steps(n)*k` after each block) and write a
      `BlockOffsetRecord{ bit_offset, init_state, scale_q }`; pad to PAGE.
   b. record `payload_offset = current_len`; write `enc.bits` verbatim; set
      `payload_bytes = enc.bits.len()`; pad to PAGE.
   c. if any sub-scales/mins: record `sideinfo_offset = current_len`; write all blocks'
      `sub_scales` then (if affine-min) all blocks' `mins`; set `sideinfo_bytes`; pad to
      PAGE. Else `sideinfo_offset = 0, sideinfo_bytes = 0`.
3. Back-patch the descriptor `*_offset` fields and the header `header_bytes`/`flags`.

Because step 2a's accumulation is identical to the `bit_cursor` v1's `read_strand` /
`decode_tensor_fixed` advance, the table is correct **by construction** — a round-trip test
asserts `table[b].bit_offset` equals the cursor position after decoding `b` blocks of the
v1 form, and that `decode_tensor_fixed(v2.enc) == decode_tensor_fixed(v1.enc)`.

---

## 9. Sizing sanity (the table is cheap)

The offset table costs **16 bytes per block**. At `block_len=256`, that is `16/256 = 0.0625`
bytes/weight = **0.5 bits/weight** of index overhead — on top of a ~3.34 bpw payload, ≈ +1.5%
size for full random access. Page padding adds at most `3 * 4096` bytes per tensor (three
regions), negligible against multi-MB tensors. The trade is explicitly worth it: it converts
the GEMV from "re-scan to find a block" (compute-bound, the Q3_K trap) to "one 16-byte load
to seek" (keeps the decode bandwidth-bound per `STRAND-metal-decode-gate.md`).

If 0.5 bpw of index is ever too much, a documented future shrink (NOT in this spec) is to
drop `bit_offset` from the record and reconstruct it on the GPU from `num_steps*k` via a
threadgroup prefix-sum (only legal when every block in a row is full, i.e. STRICT) — trading
0.5 bpw for a per-row scan. v2 keeps the explicit offset: it is the simplest thing that makes
the kernel branch-free and the measurement honest, and 1.5% is well inside the density moat.

---

## 10. Relationship to v1 and to the build order

* **v1 (`STRQ`, version 1) is unchanged and remains the reference/round-trip format.**
  `write_strand`/`read_strand` and all 346 tests stay as-is. v2 lives next to them.
* **v2 (`STR2`, version 2) is the deploy/mmap format.** Same bits, plus a page-aligned
  per-block seek table and hoisted `scale_q`/`init_state`, so a Metal kernel jumps to any
  `(row, block)` with pure arithmetic.
* **Build order** (from the gate doc): (1) `decode_lean` + bit-identity test [correctness
  scaffold], (2) **this `write_strand_v2` + table** [pure CPU/Rust, cheap], (3) the Metal
  GEMV kernel + M3 bandwidth measurement [the one real GPU experiment]. v2 is step 2 and is
  a prerequisite for step 3 — the kernel cannot seek without the table.
