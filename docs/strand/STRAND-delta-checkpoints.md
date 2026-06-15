# STRAND delta checkpoints — block-level diff/patch between encoded models (Tier 1b)

_Tool: `crates/strand-quant/src/bin/strand-delta.rs` (binary `strand-delta`). Status: **diff/measure
built + smoke-verified 2026-06-10**; the patch writer/applier is the follow-up, gated on the PV
ratio measurement below. This doc defines the identity contract, the patch format sketch, the
measurement protocol for the marathon's artifacts, and the distribution story._

---

## 1. The thesis

The PV training loop requants every 75 steps. Each requant is a full re-encode of the shadow
weights through the real Rust encoder — but between adjacent requants the optimizer moves most
weights a little, not a lot, and the STRAND encode is **block-local**:

- blocks are 256-weight runs of the flat tensor; each block's path bits + `BlockMeta`
  (`scale_q`, `sub_scales`, `min_base_q`, `mins`, `init_state`, `n`) are a pure function of that
  block's post-RHT weights;
- the RHT is **row-aware** (per-row Hadamard, seed = FNV-1a of the tensor NAME — identical for
  both checkpoints by construction), so a weight change cannot smear outside its row;
- a changed row therefore dirties at most `ceil(in_features/256)` blocks (+1 when a block
  straddles a row boundary, e.g. the 896-dim 0.5B).

So checkpoint N+1's encoding should share most blocks **byte-identically** with checkpoint N once
training settles — ship only the changed blocks ("git for quantized models"). `strand-delta diff`
proves or disproves that per pair, with wire-byte counts.

**The one locality breaker is the outlier channel** (`--outlier-channel`): selection is global
top-|w| over the whole tensor, so drift can churn the membership margin and un-zero/zero bulk
positions in otherwise-untouched rows, dirtying their blocks. The smoke (§6) constructs the worst
case deliberately; the real PV churn is unknown until measured (§4).

## 2. Identity contract (what "identical block" means)

A block is identical iff **both**:

1. its packed path-bit slice is byte-equal (block `i` occupies bits `[Σ_{j<i} n_j·k, +n_i·k)` of
   `EncodedTensor.bits`; at `block_len=256` every block starts byte-aligned), and
2. its `BlockMeta` is field-equal (`derive(PartialEq, Eq)` over pure integer/byte fields).

**No float math on the comparison path.** Floats exist only on the encode side (identical to
`quantize-model` — offline preprocessing). The outlier channel is compared as integers: the
`f32::to_bits()` pattern of the absmax, plus `(index, code)` pairs canonicalized ascending-index.

**No hashes are defined here.** Content identity, provenance, and the base/target binding fields
in the patch header are owned by `docs/STRAND-v3-provenance-spec.md`. `strand-delta` decides
equality by byte comparison only; a patch file *carries* provenance-spec identities, it never
invents its own.

**Encode pinning.** The encode is the float side, so byte-stability is per backend/build. The tool
pins the CPU SIMD encode (`STRAND_NO_GPU=1`, removable with `--gpu-encode`): the Metal encode
SIGKILLs on 7B-wide tensors, 12-way CPU beats the serialized GPU ~8× anyway (will.md §7), and the
patch/base pair must come from one encoder. In-process re-encode determinism is asserted by smoke
case 0 (A vs A → 16/16 identical). Cross-machine patch production is **unverified** — produce base
and patch on the same box+build, or gate cross-box use on a measured A/B first.

## 3. Patch format sketch — `SDP1`

Exactly what the tool bills today (every byte counted, ideology §5.11). Apply is integer-only:
copy base records, splice listed blocks, swap the outlier channel, verify the target identity per
the provenance spec, spot-check with `decode_lean` bit-identity.

```
file header (84 B):
  magic "SDP1" (4) | version u32 (4)
  config echo (8):  k u8 | l u8 | flags u8 (bit0 rht, bit1 tail_biting, bit2 affine_min)
                    | outlier_ppm u32 | outlier_bits u8
  base identity (32) | target identity (32)        — fields per STRAND-v3-provenance-spec.md
  n_tensors u32 (4)

per-tensor records (target header order):
  status 0/1/3 prefix: name_len u32 | name | status u8
  status 2     prefix: status u8 only               (the verbatim v1 record below carries
                                                     its own name — never billed twice)
  status 0 UNCHANGED : n_blocks u32                 (sanity echo; 9+name B total)
  status 1 PATCHED   : n_blocks u32 | n_changed u32
      n_changed × changed-block record:
        block_idx u32
        scale_q i32 | min_base_q i32 | init_state u32 | n u32
        sub_len u32 | sub_scales[] | mins_len u32 | mins[]     (v1 BlockMeta wire, verbatim)
        payload ceil(n·k/8) B                       (the block's bit-run, re-aligned to a byte)
      outlier_flag u8
        | if 1: count u64 | omax_f32_bits u32
                | packed (idx,val) stream at (ceil(log2 n_w)+outlier_bits) bits/entry
                  (the channel ships whole — per-tensor side info, not per-block)
  status 2 REPLACED/ADDED : verbatim v1 (STRQ) tensor record + packed outlier channel.
        A patcher emits this whenever the per-block form would cost more than 1+full
        (the tool bills min(patched, 1+full) — the 4 B/block index overhead loses when
        nearly everything changed).
  status 3 REMOVED   : nothing further (tombstone).
```

"Full" baseline for ratios = the v1 STRQ archive of the **target** checkpoint (12 B file header +
per-tensor records + packed outlier channel), i.e. what you would ship without deltas.

Out of scope v1 of the format: vector trellis / learned per-tensor LUTs (the LUT is side info that
needs its own diff treatment — the CLI rejects `--vec-dim`), and `--mp-config` (per-tensor k would
go in the per-tensor record; trivial extension, not wired).

## 4. Expected ratios for PV checkpoints — UNKNOWN; here is the measurement

**Do not guess; measure (ideology §5.1).** The identical-block fraction between adjacent requants
is exactly the quantity the delta thesis lives or dies on, and it depends on how much AdamW drift
at the PV learning rate crosses encode decision boundaries (scale re-pick, sub-scale code flips,
Viterbi path divergence). Two regimes are plausible — early segments (large drift, most blocks
change, patch ≈ full) vs late convergence (tiny drift, most blocks identical) — and tonight's
runs will not settle which without numbers.

**Protocol (run AFTER the marathon finishes — the box and `scratch/qwen-05b/strand-pv` are the
live run's; do not touch them while it is in flight):**

1. Identify adjacent requant-boundary shadow checkpoints among the PV artifacts under the live
   repo's `scratch/qwen-05b/strand-pv/` (the segmented arms save per-boundary shadow safetensors;
   confirm exact filenames from the run log once it drains — if only `.pt` shadows exist, convert
   with the harness's save path first).
2. For each adjacent pair (N, N+1), from this worktree:

   ```sh
   nice -n 19 ./target/release/strand-delta diff \
       <shadow_N.safetensors> <shadow_N+1.safetensors> \
       --bits 2 --l 12 --outlier-channel 1 --threads 4 \
       --json scratch/delta-pv-N.json
   ```

   Flags mirror the PV requant config exactly (`quantize-model --bits 2 --l 12
   --outlier-channel 1`); the tool re-encodes both checkpoints through the same path, so the
   comparison is over the real wire artifacts. Cost ≈ 2× one requant per pair (~30 min at 4
   threads on the 0.5B) — batch it when the box is idle.
3. Also run one early pair and one late pair `--outlier-channel 0` vs `1` to isolate the
   channel's churn contribution (the smoke's +11/11 is the adversarial ceiling, not a forecast).

**Decision gate for building the patch writer:** identical-block fraction ≥ ~50% on late-segment
pairs ⇒ patch ≤ ~55% of full including overhead ⇒ build `patch`/`apply`. Fraction < ~30% ⇒ the
per-block delta wins ≤ 1.4× — park the writer and bank the negative honestly.

## 5. Distribution story

Once ratios justify it: a model repository stores **rev 0 as a full `.strand` archive + a chain of
`SDP1` patches** (rev N → N+1, one per requant/release).

- **Publish:** each release ships its patch; every M revs (or whenever cumulative patch bytes
  exceed ~1 full archive) publish a full snapshot to bound chain length and re-anchor.
- **Fetch:** a client at rev N downloads only `patch_N→N+1 … patch_{M-1}→M`. Applying is integer
  byte-splicing (no encoder needed client-side), then identity verification per
  `STRAND-v3-provenance-spec.md` (base identity checked before apply, target identity after).
- **Why STRAND can do this and float formats cannot cheaply:** the artifact is deterministic —
  same weights + config ⇒ byte-identical blocks — so "unchanged" is decidable by byte equality
  with zero tolerance questions, and a patched archive is **bit-identical** to a from-scratch
  encode of the target (verify with `decode_lean` on spot tensors). The determinism moat is what
  makes the diff well-defined at all.
- This composes with the format tiers: patches target the v1 STRQ record layout; a v2 (STR2)
  deploy archive is re-baked locally from the patched v1 state (or a v2-native patch reuses the
  same changed-block list against the v2 block-offset table — `BlockOffsetRecord` is 16 B/block,
  same identity contract).

## 6. Smoke results (measured 2026-06-10, M3 Pro, CPU encode, bits=2 L=12)

Fixture: `q_proj [8,256]` (1 block/row) + `down_proj [4,512]` (2 blocks/row); checkpoint B
re-rolls q row 3 and down rows 1–2 at 1.5× amplitude. `strand-delta smoke` asserts all of this
and exits non-zero on any mismatch (also runs as `cargo test -p strand-quant --bin strand-delta`).

| case | config | identical | changed sets | patch vs full |
|---|---|---|---|---|
| 0: A vs A | no outlier | **16/16** | — | 175 B vs 1701 B (UNCHANGED markers only) |
| 1: A vs B | no outlier | **11/16** | q={3}, down={2,3,4,5} — **exactly the changed rows** | **675 B vs 1701 B = 39.7% (2.5× smaller)** |
| 2: A vs B | +1% outlier | 0/16 | superset held; **+11 churn blocks** | 1873 B vs 1821 B (no win) |

Readings: case 0 = in-process encode determinism (the foundation); case 1 = block locality is
exact — changed rows map to precisely the predicted blocks, nothing else moves; case 2 = the
outlier channel's global coupling at its adversarial worst (the 1.5× rows capture essentially the
whole top-1%, churning zeroed positions in every row — and at 2 tensors the 84 B patch header
dominates anyway). Real PV drift between requants is incremental, not a 1.5× row re-roll; where
between 39.7% and "no win" a real pair lands is exactly §4's measurement.

## 7. What this is NOT (yet)

- **Not a ratio claim.** No PV delta ratio is asserted anywhere until §4 runs on real artifacts.
- **Not a hashing/provenance system** — that is `STRAND-v3-provenance-spec.md`'s job; this tool
  compares bytes it holds in memory and never persists an identity.
- **Not a patch writer.** `diff` measures and bills the patch exactly; emitting/applying `SDP1`
  is deliberately gated on the §4 verdict.
- **Not wired for** the vector trellis (`--vec-dim` rejected — per-tensor learned LUT side info),
  `--mp-config`, or cross-machine patch production (unverified; pin one box+build).
