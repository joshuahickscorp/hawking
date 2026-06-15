# STRAND on CPU — the deterministic, dense, float-free format that ships TODAY

_Option B of the post-GPU-gate rescue, 2026-06-09._

This is the deployment path that is **not** blocked by the Metal bandwidth gate. The GPU
trellis-GEMV is compute/latency-bound on the M3 (the gate verdict, commit `1ebcab6`: 8–23 % of
peak streaming BW), and worse, the **GPU-side** `BlockEntry` (52 B per 256-weight block) inflates
the GPU's bytes/weight to **0.578 > Q4_K's 0.5625** — so even a hypothetical bandwidth-bound GPU
kernel would not beat Q4_K on bytes/token. **None of that touches the CPU path.** On CPU, STRAND:

- decodes straight from the **lean 16-B-per-block on-disk side-info** (not the 52-B GPU bake
  struct), so the realised on-disk density is **0.4175 B/weight at the 3-bit deploy point — ~26 %
  fewer bytes than Q4_K**;
- decodes with **zero floating-point** (the determinism moat: bit-identical weights on every
  device, compiler, and thread count);
- needs **no GPU and no float unit** — it runs on a phone, a browser (WASM), a microcontroller, or
  an FPGA;
- now decodes **~1.9× faster** than the reference scalar decode via `decode_q12_fast`, with the
  speedup verified bit-identical to the reference.

The honest framing: **STRAND-on-CPU is a shipping product today; STRAND-on-GPU is a research
gate that has not cleared.** This doc is the lock-in of the former.

---

## 1. What ships

| component | where | status |
|---|---|---|
| Integer-deterministic decode (reference) | `strand_quant::decode::decode_lean` / `decode_tensor_fixed` | shipped, the float-free moat |
| **Fast decode (this work)** | `strand_decode_kernel::gemv::decode_q12_fast` (+ `decode_tensor_q12_fast`, `matvec_named_fast`) | **bit-identical to the reference, ~1.9× faster** |
| mmap'd `.strand` v2 loader | `strand_decode_kernel::loader::StrandModel` | shipped |
| CPU decode→GEMV runtime | `strand_decode_kernel::gemv::{matvec_named, matvec_named_fast}` | shipped |
| Throughput + density gate bin | `strand-decode-kernel/src/bin/gate-cpu-fastpath.rs` | this work |

All of it compiles on **every** target (the Metal module is `#[cfg(target_os = "macos")]`-only; the
loader/decode/GEMV build with no GPU). The decode is the correctness reference the GPU kernel is
*required* to reproduce bit-for-bit — so even if the GPU path later clears its gate, the CPU path
remains the canonical, portable artifact.

---

## 2. The determinism moat (why this is not "just a slow quantizer")

The per-weight reconstruction is an **exact function of integer state**:

```
recon_q = (scale_q * quantile_q) >> 16          // i64 product, arithmetic shift; Q12 i32
```

No float arithmetic anywhere in the decode — only shifts, masks, an `i64` multiply, and a frozen
`i32` quantile-LUT lookup (`strand_quant/src/decode.rs`). The only float touch in the whole
pipeline is the final `Q12 * 2^-12` cast (a power-of-two scale: lossless, no rounding) and the
GEMV MAC against the activation — the same float status as **every** GGUF GEMV. The **weights are
bit-identical across devices**; GGUF dequantises through floats and drifts. This is STRAND's
uncontested axis: a reproducible-by-construction, cryptographically-checkable artifact — 70B on a
laptop, 7B on a phone, 1B on a watch, *same bytes, same behaviour, no float in the decode*.

`decode_q12_fast` does **not** weaken this. It is a strict drop-in: every integer operation
(symbol pop, `next_state`, `reconstruct_q`, `eff_scale_q`, `eff_min_q`) is byte-for-byte the one
`decode_lean` performs; only the loop *shape* and allocation change. The contract is enforced by a
test (`gemv::fast_decode_is_bit_identical`) that sweeps the **full** lever matrix — `k ∈ {2,3,4}`,
both scale-fold branches, tail-biting on/off, affine-min on/off, over lengths hitting the three
structural edges (short final block `n·k < L`, sub-block tail `n % 32 ≠ 0`, `u32`-unaligned total
bit length) — i.e. the same matrix as `strand_quant`'s own `decode_lean_is_bit_identical`. A single
counterexample is a release blocker. (`gate-cpu-fastpath` re-asserts it on each benched shape
*before* reporting any timing.)

---

## 3. The 3-bit density (beats Q4_K on bytes)

Measured on real encoded tensors at the 3-bit deploy point (`k=3, L=7`, 256-weight blocks), the
**on-disk** bytes/weight from the actual `EncodedTensor` (payload + ALL side info: super-scale,
packed 6-bit sub-scales, `init_state`, the one tensor-level length — `enc.total_bpw`):

| metric | STRAND 3-bit (CPU/on-disk) | Q4_K (iso-ish quality) |
|---|---|---|
| **bytes/weight** | **0.4175** | 0.5625 |
| 7B weight footprint | ~2.6 GB | ~3.5 GB |
| determinism | bit-identical (integer decode) | float dequant, drifts |

**0.4175 vs 0.5625 ⇒ ~26 % fewer bytes than Q4_K**, the headline density win, realised on the path
that actually ships. (Contrast the **GPU** path's 0.578 B/weight — the 52-B `BlockEntry` table
inflates it *above* Q4_K. The CPU path never pays that table: it reads the lean 16-B
`BlockOffsetRecord` side-info and expands `eff[]` on the fly. The density moat lives on CPU.)

> Quality caveat (priced in, not hidden): the standalone sub-4-bit *PPL* race against Q4_K is lost
> on the merits (the GGUF "smaller AND better" thesis is disproven 4×; see `MEMORY.md` /
> `research/STRAND-quant-findings-and-playbook.md`). STRAND's CPU product does **not** claim to beat
> Q4_K on perplexity. It claims **fewer bytes at a usable 3-bit quality, plus determinism, plus
> runs-anywhere** — the trinity Q4_K does not have all three of.

---

## 4. Measured CPU decode throughput

`cargo run -p strand-decode-kernel --release --bin gate-cpu-fastpath`

Representative Qwen2.5-7B GEMV shapes, full 7B *columns* (so bytes/weight + per-row decode cost are
exact), 256 rows (decode throughput is per-weight, so the row count does not bias Mweights/s):

| shape | decode ref (`decode_lean`) | **decode fast** (`decode_q12_fast`) | speedup | B/weight | decode→GEMV (fast) |
|---|---|---|---|---|---|
| attn_o  (cols=3584)  | 430 Mweights/s | **820 Mweights/s** | **1.91×** | 0.4175 | 523 Mweights/s |
| ffn_up  (cols=18944) | 442 Mweights/s | **816 Mweights/s** | **1.85×** | 0.4175 | 516 Mweights/s |
| ffn_down (cols=3584) | 431 Mweights/s | **816 Mweights/s** | **1.89×** | 0.4175 | 515 Mweights/s |

> **Advisory numbers.** These were captured while four GPU-variant agents shared the 12-core CPU,
> so the *absolute* Mweights/s understate a serial run. The **relative ~1.9× speedup** and the
> **0.4175 B/weight** are contention-independent. Re-run serially for headline absolute throughput.

### What made it ~1.9× faster (the additive levers)

`decode_q12_fast` (`strand-decode-kernel/src/gemv.rs`) keeps `decode_lean`'s arithmetic verbatim and
only removes per-weight overhead:

1. **Per-sub-block scale-fold hoisted out of the inner loop.** A 256-block has ≤ 8 sub-blocks, so
   the effective scales `eff[]` (and affine-min `off[]`) live in fixed `[i32; …]` **stack** arrays —
   no per-block heap `Vec` alloc — and the inner loop is **split per 32-weight sub-block** so the
   `eff[sb]` lookup and the `i / SUB_BLOCK` division are lifted entirely out of the hot stride.
2. **Aligned 32-bit-word symbol reads** (a local `WordBitReader`: one running `u64` accumulator
   instead of a per-bit byte walk — the same lever the Metal kernel uses, with the same
   read-past-end-as-`0` contract).
3. **Unchecked LUT / scale indexing** on the hot path: `state` is masked to `< num_states ==
   lut.len()` and the sub-block index is bounded by construction, so the bounds checks are provably
   redundant and elided.

For the small-`L` fold regime it folds each sub-block's `eff` into the LUT once (`reconstruct_q(es,
lut[s])`) — bit-exact, same `i64` product — turning the per-weight reconstruct into a load, exactly
as `decode_lean` does. Both branches are covered by `fast_decode_is_bit_identical`.

---

## 5. Where it fits vs GGUF

| axis | STRAND 3-bit on CPU | GGUF Q4_K on CPU (llama.cpp) |
|---|---|---|
| **bytes/weight** | **0.4175** (~26 % fewer) | 0.5625 |
| **determinism** | **bit-identical** (integer decode, no float) | float dequant → device/thread drift |
| **runs anywhere** | **CPU / WASM / phone / MCU / FPGA**, no float unit needed for decode | CPU/GPU; dequant assumes a float unit |
| perplexity (sub-4-bit) | loses to Q4_K (priced in) | the yardstick |
| decode throughput | ~0.8 Gweights/s (this work, contended) | mature, hand-tuned SIMD |

**Net:** for a deployment that values **smallest footprint + provable cross-device reproducibility +
no-float-unit portability** — an on-device assistant that must produce identical output on a phone,
a laptop, and a server; a verifiable/auditable model artifact; a target without a usable float path
— STRAND-on-CPU is the format Q4_K cannot match on those three axes simultaneously. For a pure
"fastest tokens/sec on one GPU" deployment where determinism and footprint are secondary, mature
GGUF GPU kernels remain ahead (and STRAND's own GPU gate has not cleared).

---

## 6. API

```rust
use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::gemv::{decode_tensor_q12_fast, matvec_named_fast};

let model = StrandModel::open("model.strand")?;           // mmap, zero-copy

// fast integer decode to Q12 (bit-identical to decode_tensor_q12 / decode_lean):
let q12: Vec<i32> = decode_tensor_q12_fast(&model, "blk.0.attn_q.weight").unwrap();

// or the fused decode→GEMV (RHT-space activation when the tensor was RHT-encoded —
// see shaders/README.md §RHT for the host's once-per-GEMV FWHT responsibility):
let y: Vec<f32> = matvec_named_fast(&model, "blk.0.ffn_up.weight", &x_rht).unwrap();
```

Both have reference twins (`decode_tensor_q12`, `matvec_named`) that produce identical results; the
`_fast` variants are pure speedups. For an explicit custom codebook (LEVER B3 `--dist`) use
`gemv::decode_q12_fast_with_lut`, the fast analog of `decode_lean_with_lut`.

---

## 7. Verify

```
cargo test  -p strand-decode-kernel                       # 16 tests incl. fast_decode_is_bit_identical
cargo run   -p strand-decode-kernel --release --bin gate-cpu-fastpath   # density + throughput
```

The bench runs the correctness gate (`decode_q12_fast == decode_lean`, bit-for-bit) before it
reports any timing — a fast wrong decode is rejected, not measured.
