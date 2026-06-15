# `.strand` self-describing artifact — the `SDSC` section (spec)

_Status: **SHIPPED + PROVEN** (2026-06-11). Reference implementation:
`crates/strand-quant/src/selfdesc.rs` (emit / append / read / the SDSC-driven
reference decoder, 6 unit tests). The proof of the property:
`tools/selfdesc-interpreter.py`, a from-scratch stdlib-only interpreter that
reads ONLY the file — gate results in §8. This doc and that module's constants
must agree byte-for-byte._

---

## 1. The eternal-format property (why this section exists)

Every format dies when its last decoder does. A `.strand` archive already
carries deterministic *data* (the payload bits, the side-info, the OUTL
channel, the SPRV commitments) — but the decode *semantics* lived only in the
Rust source. The `SDSC` section closes that gap: it embeds, inside the file,

1. the **frozen quantile LUT** for every geometry the archive uses (the raw
   Q12 `i32` tables — they were always just deterministic data),
2. the **decode arithmetic** as straight-line programs in a tiny, fully
   enumerated integer algebra (§3) — NOT a turing-complete bytecode,
3. the **layout constants and rule ids** (§4) that pin the structural decode
   loop (§5) to this published spec.

A future runtime holding ONLY the file can reconstruct the bulk Q12 plane —
the exact integer object the SPV3 provenance roots hash — by implementing
this document once and taking every number from the file. The property is not
asserted, it is **demonstrated**: an interpreter written from scratch against
this spec (no Rust knowledge, python stdlib only) reproduces the Q12 stream
of the real shipped artifact byte-for-byte (§8).

What it is **not**: SDSC carries data and formulas, not commitments —
tamper-evidence stays SPRV's job. And SDSC v1 deliberately stops at the Q12
plane boundary (§7).

## 2. Wire layout

`SDSC` is an EOF-chained, page-aligned, append-only trailing section in the
family `provenance_io.rs` established (`SPRV`) and `outlier_wire.rs` followed
(`OUTL`). All integers little-endian; `PAGE = 4096`.

```text
… v2 regions, padded to PAGE …
SDSC HEADER  (page-aligned, 48 bytes)
    magic            [u8; 4]   = "SDSC"
    version          u32       = 1
    flags            u32         reserved, 0 (nonzero ⇒ chain-invalid)
    n_consts         u32
    n_exprs          u32
    n_luts           u32
    reserved         [u8; 24]    zero
CONSTS   n_consts × { const_id u32, value i64 }            ids strictly ascending
EXPRS    n_exprs  × { expr_id u32, n_slots u32,
                      prog_len u32, prog [prog_len] }      ids strictly ascending
LUTS     n_luts   × { l_bits u32, vec_dim u32,
                      n_entries u32, reserved u32 = 0,
                      entries i32 × n_entries }            (l,d) strictly ascending;
                                                           n_entries == 2^l_bits · vec_dim
PADDING  zeros to one page before the trailer (end page-aligned)
TRAILER  (last 16 bytes of the padded region)
    sdsc_offset      u64         absolute offset of SDSC HEADER (page-aligned)
    sdsc_bytes       u32         header + consts + exprs + luts (excl. padding/trailer)
    magic            [u8; 4]   = "SDSC"
```

### 2.1 Stacking order and discovery

**SDSC is the INNERMOST chained section: v2 regions, then SDSC, then OUTL,
then SPRV (outermost).** The shipped OUTL/SPRV chain walkers descend only
through trailer magics they know, so SDSC must sit below them. Discovery
walks the 16-byte trailers from EOF: an `SPRV` or `OUTL` trailer descends to
its stored offset (the page-aligned end of everything beneath it); an `SDSC`
trailer is the find; anything else means absent. Absent ⇒ not an error (the
section is optional); present-but-corrupt ⇒ error in strict mode, treated as
absent in tolerant mode — the same contract as SPRV/OUTL.

`append_sdsc` is **restack-aware**: on a file already carrying OUTL/SPRV it
captures both via their public readers (strict), truncates to the chain base,
appends SDSC, and re-appends OUTL then SPRV through their own writers. The
section *bodies* are byte-identical after the restack (only trailer offsets
move), the v2 bytes are never touched, and the SPRV self-verifying loop still
closes (unit-tested, and proven on the real artifact: the restacked copy's
recomputed model root equals the canonical `6e1b0e4e…be54`). Double-append is
rejected.

### 2.2 Versioning / forward-compat

`SDSC` version 1 freezes: this layout, the op set (§3.1), the expression slot
assignments (§3.2), the const ids (§4), and the structural rules (§5). Any
change mints version 2; a version-1 reader MUST reject other versions (error,
never fallback — same stance as SPRV v2). Within v1, *new rule-id values*
(e.g. a future `SIDEINFO_LAYOUT = 2`) are the forward-compat channel: a
reader that meets a rule id it does not know MUST refuse to decode rather
than guess (enforced in both reference implementations). Sanity caps:
n_consts ≤ 256, n_exprs ≤ 64, n_luts ≤ 32, prog ≤ 4096 B, n_slots ≤ 16.

## 3. The expression algebra

A stack machine over **wrapping two's-complement `i64`**, straight-line only:
no branches, no loops, no memory — every program is a finite expression tree,
so termination and worst-case cost are syntactically evident (this is the
"NOT turing-complete" design constraint, on purpose). A program is validated
structurally at parse time (operands in-bounds, `LOAD` slots < `n_slots`,
terminates with `END` exactly at the buffer end) and leaves exactly one value.

### 3.1 The ops (complete enumeration; anything else is malformed)

| opcode | name | operands | semantics (stack: … pops right-to-left) |
|---|---|---|---|
| 0x00 | END | — | program ends; the single remaining value is the result |
| 0x01 | IMM | i64 LE | push immediate |
| 0x02 | LOAD | u8 slot | push input slot value |
| 0x10 | ADD | — | pop rhs, pop lhs, push `lhs + rhs` (wrap64) |
| 0x11 | SUB | — | `lhs − rhs` (wrap64) |
| 0x12 | MUL | — | `lhs · rhs` (wrap64) |
| 0x13 | TDIV | — | truncating `lhs / rhs` (toward zero); `rhs = 0 ⇒ 0`; `MIN/−1` wraps |
| 0x14 | NEG | — | pop x, push `−x` (wrap64) |
| 0x15 | ABS | — | pop x, push `|x|` (wrap64: `|MIN| = MIN`) |
| 0x16 | WRAP32 | — | pop x, push sign-extended low 32 bits |
| 0x18 | SHL | — | pop shift, pop x, push `x << (shift & 63)` (wrap64) |
| 0x19 | ASR | — | pop shift, pop x, push arithmetic `x >> (shift & 63)` |
| 0x1C | AND | — | bitwise and |
| 0x1D | OR | — | bitwise or |
| 0x1E | XOR | — | bitwise xor |
| 0x20 | CLAMP | — | pop hi, pop lo, pop x, push `min(max(x, lo), hi)` |

Determinism notes: every op is total (no traps); division is the *truncating*
kind (Rust `i64::wrapping_div`), NOT floor division — a python implementation
must not use `//` directly (the interpreter's `tdiv` helper is the model).

### 3.2 The v1 expressions (ids + frozen slot assignments)

| id | name | slots | the program computes |
|---|---|---|---|
| 1 | ADVANCE | 0 state, 1 sym, 2 k_bits, 3 state_mask | `((state << k) \| sym) & mask` — the bitshift-trellis step |
| 2 | EFF_SCALE | 0 scale_q, 1 code | `(scale_q · ((code & 63) + 1)) >> 6`, wrap32 — effective sub-block scale (Q16) |
| 3 | OFFSET | 0 min_base_q, 1 code | `((\|min_base_q\| · (code & 31)) · (((code >> 5) & 1)·2 − 1)) /₍trunc₎ 31`, wrap32 — affine-min offset (Q12; mag 0 ⇒ 0 for either sign, so the branch in the Rust source is pure arithmetic here) |
| 4 | RECON | 0 eff_scale, 1 lut_q, 2 off | `wrap32((eff · lut_q) >> 16) + off`, wrap32 — the weight (Q12) |

These four programs ARE the decode arithmetic: the reference decoder
(`decode_q12_with_sdsc`) and the python interpreter run them for every state
advance, sub-scale, offset, and weight; neither contains the formulas in code.
Unit test `sdsc_exprs_match_native_arithmetic` pins each program to its native
Rust twin (`next_state`, `eff_scale_q`, `eff_min_q`, `reconstruct_q`) over
dense input sweeps including the sign/zero edges.

## 4. The constants (ids frozen for v1)

| id | name | v1 value | meaning |
|---|---|---|---|
| 1 | SCALE_SHIFT | 16 | informational (folded into RECON's immediate) |
| 2 | SUB_SCALE_SHIFT | 6 | informational (folded into EFF_SCALE) |
| 3 | SUB_BLOCK | 32 | weights per sub-scale sub-block |
| 4 | SUBSCALE_CODE_BITS | 6 | width of one packed sub-scale / mins code |
| 5 | QUANTILE_SHIFT | 12 | LUT entries are Q12 |
| 6 | PAYLOAD_BIT_ORDER | 0 | 0 = LSB-first within bytes (only defined value) |
| 7 | SIDEINFO_LAYOUT | 1 | the v2 side-info arrangement (§5.3) |
| 8 | TAILBITE_RULE | 1 | the start-state rule (§5.4) |
| 9 | OUTL_PATCH_RULE | 1 | weight-space replacement (§6) |
| 10 | RHT_RULE | 0 | 0 = RHT NOT self-described — the declared v1 gap (§7) |

Ids 1–2 are informational duplicates (the operative shifts live inside the
programs); 3–8 are *load-bearing* for the structural loop; 9–10 declare the
weight-space semantics level. A reader MUST check ids 6/7/8 against the
values it implements and refuse otherwise.

## 5. The structural decode loop (what the algebra cannot carry)

The block/section *structure* is loop-shaped, so it is pinned by rule ids +
this prose instead of programs. With the descriptor fields (`l_bits`,
`k_bits`, `vec_dim`, `block_len`, `total`, `n_blocks`, flags, region offsets)
from the v2 header — v2 itself is the container spec, identified by
`STR2`/version 2 — and `S = SUB_BLOCK`, `C = SUBSCALE_CODE_BITS`:

1. **Blocks.** Block `b` has `n_b = block_len` weights except the last,
   which has `total − (n_blocks−1)·block_len`. Per-block `(bit_offset,
   init_state, scale_q)` come from the offset-table region (16-byte records:
   u64, u32, i32).
2. **Payload.** One contiguous LSB-first bitstream; symbol `i` of block `b`
   is the `k`-bit field at bit `bit_offset_b + i·k` (bit `j` of the stream is
   byte `j/8`'s bit `j mod 8`).
3. **Side-info (`SIDEINFO_LAYOUT = 1`).** From `sideinfo_offset`: every
   block's packed sub-scale codes back-to-back (block `b` contributes
   `ceil(C·ceil(n_b/S)/8)` bytes; codes are C-bit LSB-first within the
   block's slice). If the descriptor's affine-min flag is set: align the
   cursor to 4, then `n_blocks` i32 `min_base_q` values, then every block's
   packed mins codes (same per-block sizes).
4. **Start state (`TAILBITE_RULE = 1`).** If the tensor's tail-biting flag is
   set AND `n_b·k ≥ L`: prescan the block's own `n_b` symbols from state 0
   through ADVANCE; the final state is the start state. Otherwise:
   `init_state & state_mask`.
5. **Weights.** For `i = 0..n_b`: `state = ADVANCE(state, sym_i, k, mask)`;
   `q12_i = RECON(EFF_SCALE(scale_q, scode[i/S]), lut[state],
   affine ? OFFSET(min_base_q, mcode[i/S]) : 0)`. The LUT is the SDSC table
   for the descriptor's `(l_bits, vec_dim)`.

SDSC v1 covers `vec_dim = 1` (the shipped scalar frozen-LUT path); a
vector-trellis tensor (`d > 1`, learned LUT) is a v1 error — same stance as
`provenance_io::default_lut_provider`.

## 6. The OUTL patch rule (`OUTL_PATCH_RULE = 1`)

The sparse-outlier channel (the `OUTL` section, spec'd in `outlier_wire.rs`)
is applied in **weight space, after the inverse RHT**, as REPLACEMENT (not
addition): `w[idx] = (code / levels) · omax` with `levels = 2^(val_bits−1) − 1`,
`omax = f32::from_bits(omax_bits)`, evaluated in f32 **in exactly that order**
(division first — pinned to `quantize-model`'s recon arithmetic;
`outlier_wire.rs::dequant_vals`). This rule rides on the Q12 plane's float
side and is stated here for completeness; the Q12 plane itself (and hence the
provenance roots) is outlier-free by construction.

## 7. The honest boundary (what v1 proves, and the named gap)

**Proven self-describing in v1: the bulk Q12 plane** — the deterministic
integer artifact, the exact object the SPV3 roots hash and the moat's
bit-identity contract covers. The full weight-space semantics stack is:

```text
Q12  --(· 2^-12, exact)-->  f32 (RHT space)
     --(inverse row-aware RHT, descriptor rht_seed)-->  f32 (weight space)
     --(OUTL replacement, §6)-->  the computed-with weights
```

The first arrow is trivial (one exact power-of-two scale) and the third is
§6. The second — the row-aware randomized Hadamard transform (per-row
Rademacher signs drawn FNV-1a-style from the global flat index, normalized
FWHT over `HADAMARD_BLOCK = 256` segments, see `rht.rs` and the
`outlier_mac.rs` per-row-rotation analysis) — is **not expression-encodable
in the §3 algebra**: it is float arithmetic (`1/√n` normalization) with a
loop structure (FWHT butterflies) the deliberately-not-turing-complete
algebra cannot and should not carry. Per the honesty-over-completeness rule,
v1 declares it: `RHT_RULE = 0` = "not self-described; see this spec's §7".
Closing it is the v2 design problem (candidates: a fixed-point FWHT rule id +
a sign-stream generator spec'd as bit-exact integer ops, or accepting the
float recon as a *documented* — not embedded — semantics layer, as the
provenance layer already does for `rht_seed`).

Also out of v1 scope, recorded: vector-trellis learned LUTs (§5 note — the
table would simply be embedded as an `(l, d)` LUT, but no shipped artifact
uses one), and the SPV3 hashing layer itself (the interpreter implements it
from `STRAND-v3-provenance-spec.md` to *verify* its decode; the hash domains
are verification harness, not decode semantics).

## 8. The proof (gate results, 2026-06-11)

Setup: `scratch/artifacts/qwen05b-pv2-2bit.strand` (the canonical honest-rebake
trained-2-bit artifact: 134,493,904 B, 0.3759 B/w billed, 168 tensors, L=12
k=2, OUTL+SPRV v2 stacked, stored model_root `6e1b0e4e…be54`) **copied** to
`qwen05b-pv2-2bit.sdsc.strand`; `append_sdsc` restacked the copy to
v2|SDSC|OUTL|SPRV (+20,480 B: one 4096-entry Q12 LUT + 4 programs + 10 consts
= 16,757 section bytes, page-padded; 0.00006 B/w). The original artifact was
never touched.

| check | result |
|---|---|
| `attest-strand --roots` on the restacked copy (all 168 tensors decoded) | SPRV self-verify (Vectors) **PASS**; recomputed model_root == stored == canonical `6e1b0e4e…be54`; spot fixed==lean determinism gate OK — the restack is byte-faithful |
| in-crate: `decode_q12_with_sdsc` vs `decode_lean` | bit-identical on every wire-lever combination (plain / tail-biting / affine-min / both, ragged blocks) — `sdsc_driven_decode_is_bit_identical` |
| algebra: SDSC programs vs native arithmetic | exact over dense sweeps — `sdsc_exprs_match_native_arithmetic` |
| **THE THEOREM:** `tools/selfdesc-interpreter.py` (stdlib-only, reads ONLY the file) on the spot tensors (first / middle / last — the same set attest-strand spot-decodes) | all 3 computed SPV3 tensor_roots **byte-equal** attest-strand's roots AND the SPRV-stored roots; all 24 stored self-test vectors (8/tensor) match; exit 0 |

Interpreter gate transcript (the decisive lines, run of 2026-06-11 ~17:35;
~8.9 M weights decoded through the SDSC expression programs in pure python,
incl. two 4.36 M-weight tail-biting down_proj tensors):

```text
tensor_root 07e73aca4c0fc407c43b19cf75094702cd7e1413790a50bb4234ba4da160e173  model.layers.0.mlp.down_proj.weight  (n=4358144, 17024 blocks)  vs SPRV: root == stored, 8 self-test vectors match  [PASS]
tensor_root 382ed9cc40b4c549c46aaeb888e63240fd4baa4a245e7204e0c4bd36b37e6bc8  model.layers.2.mlp.down_proj.weight  (n=4358144, 17024 blocks)  vs SPRV: root == stored, 8 self-test vectors match  [PASS]
tensor_root 819232b001a68adf76a22ae1b654f9c398028fba2ea335193384d3d8f776ac41  model.layers.9.self_attn.v_proj.weight  (n=114688, 448 blocks)  vs SPRV: root == stored, 8 self-test vectors match  [PASS]
VERDICT  PASS — interpreter Q12 byte-equals the committed plane (SHA-256 equality over the decoded stream)
```

attest-strand's independently printed roots for the same three tensors are
identical (`scratch/artifacts/sdsc-attest-roots.log`, lines 1 / 85 / 168 of
the per-tensor root list); the interpreter transcript is banked at
`scratch/artifacts/sdsc-interp-spot.log`.

SHA-256 equality over the serialized Q12 stream IS byte equality of the
decoded weights. A from-scratch implementation, holding only the file,
decodes the artifact: the self-description property holds at the Q12 level,
with the §7 boundary stated.

## 9. Reference implementation map

| spec | code |
|---|---|
| §2 wire / append / read | `selfdesc.rs`: `SDSC_*` consts, `sdsc_section_bytes`, `append_sdsc` (restack-aware), `read_sdsc_bytes`, `read_sdsc` |
| §3 algebra | `selfdesc.rs::op`, `eval_expr` (the reference machine), `validate_prog` |
| §3.2 programs | `prog_advance` / `prog_eff_scale` / `prog_offset` / `prog_recon`, `default_exprs` |
| §4 constants | `selfdesc.rs::const_id`, `default_consts` |
| §5 structural loop | `decode_q12_with_sdsc` (Rust), `decode_tensor_q12_leaves` (python) |
| emit entry points | `emit_sdsc(cfg, lut)`, `build_sdsc_for_archive` |
| §8 proof | `tools/selfdesc-interpreter.py`; ignored test `append_sdsc_to_env_target` (`STRAND_SDSC_TARGET=<copy> cargo test … -- --ignored`) |

Tests: `nice -n 19 cargo test -p strand-quant --lib selfdesc` — 6 green
(algebra sweep, hostile programs, bit-identity, restack-under-OUTL+SPRV with
full SPRV re-verify, plain append + emit equivalence, corruption handling).
