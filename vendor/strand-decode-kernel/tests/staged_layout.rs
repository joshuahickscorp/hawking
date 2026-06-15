// Integration test (NEW file) — CPU-only proof of the staged-decode flush contract used by
// shaders/strand_bitslice_staged.metal. Uses only the crate's PUBLIC API
// (block_walk::block_plans), so it needs no edit to lib.rs.
//
// The staged kernel flushes a threadgroup tile to out_q12[obase0 + s] for s in [0, span),
// where obase0 = first block's out_off and the tile lays block `local` at slot local*256.
// This test reproduces that mapping on the CPU and asserts:
//   (1) tile slot (local*256 + j) maps to the block's TRUE global output index for every
//       covered (local, j) — i.e. no holes, no double-writes;
//   (2) the whole output [0, total) is written exactly once;
//   (3) span never exceeds the tile capacity bg*256 (no out-of-tile read).
//
// If these hold, the kernel's coalesced flush is correct for every layout `block_plans`
// produces (full 256-blocks + an optional partial LAST block — the deploy invariant).

use strand_decode_kernel::block_walk::block_plans;
use strand_quant::encode::encode_tensor;
use strand_quant::TrellisConfig;

fn pick_bg(l_bits: u32) -> usize {
    let lut_bytes = (1usize << l_bits) * 4;
    let tile_budget = (32 * 1024usize).saturating_sub(lut_bytes);
    let max_blocks = tile_budget / (256 * 4);
    let mut bg = max_blocks.min(32);
    if bg >= 8 {
        bg -= bg % 8;
    }
    bg.max(1)
}

#[derive(Clone, Copy)]
struct TgFlush {
    base_block: usize,
    obase0: usize,
    span: usize,
}

fn simulate(enc: &strand_quant::encode::EncodedTensor, k: usize, bg: usize) -> Vec<TgFlush> {
    let plans = block_plans(enc, k);
    let nb = plans.len();
    let mut out = Vec::new();
    let mut base = 0;
    while base < nb {
        let last = (base + bg - 1).min(nb - 1);
        let obase0 = plans[base].out_off;
        let span = (plans[last].out_off + plans[last].n) - obase0;
        out.push(TgFlush { base_block: base, obase0, span });
        base += bg;
    }
    out
}

#[test]
fn staged_tile_mapping_is_a_bijection_onto_the_output() {
    for cfg in [
        TrellisConfig::for_bpw(3.0),
        TrellisConfig::for_bpw_l(2.0, 12),
        TrellisConfig::for_bpw(2.0),
        TrellisConfig::for_bpw(4.0),
    ] {
        let k = cfg.k_bits as usize;
        let bg = pick_bg(cfg.l_bits);
        for &total in &[
            256usize,
            512,
            768,
            700,
            1000,
            4097,
            65537,
            bg * 256,
            bg * 256 + 1,
            bg * 256 * 3 + 137,
        ] {
            let w: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.0137).sin() * 0.5).collect();
            let enc = encode_tensor(&w, &cfg);
            let plans = block_plans(&enc, k);
            let nb = plans.len();
            let mut written = vec![0u8; total];

            for f in simulate(&enc, k, bg) {
                assert!(
                    f.span <= bg * 256,
                    "span {} > tile capacity {} (k={k} L={} total={total} base={})",
                    f.span,
                    bg * 256,
                    cfg.l_bits,
                    f.base_block
                );
                for local in 0..bg {
                    let blk = f.base_block + local;
                    if blk >= nb {
                        break;
                    }
                    let n = plans[blk].n;
                    for j in 0..n {
                        let tile_slot = local * 256 + j;
                        assert!(
                            tile_slot < f.span,
                            "tile_slot {tile_slot} >= span {} (block {blk} j={j})",
                            f.span
                        );
                        let via_tile = f.obase0 + tile_slot;
                        let truth = plans[blk].out_off + j;
                        assert_eq!(
                            via_tile, truth,
                            "MAPPING HOLE k={k} L={} total={total}: block {blk} j={j} \
                             tile->global {via_tile} != true {truth}",
                            cfg.l_bits
                        );
                        assert_eq!(
                            written[truth], 0,
                            "double-write at {truth} (block {blk} j={j})"
                        );
                        written[truth] = 1;
                    }
                }
            }
            assert!(
                written.iter().all(|&b| b == 1),
                "not every output written: k={k} L={} total={total} first-miss={:?}",
                cfg.l_bits,
                written.iter().position(|&b| b == 0)
            );
        }
    }
}
