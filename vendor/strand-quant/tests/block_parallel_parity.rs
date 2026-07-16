#![cfg(feature = "block-parallel")]

use strand_quant::encode::{
    encode_tensor_with_block_parallel, encode_tensor_with_lut, encode_tensor_with_lut_block_parallel, vector_lut_from_scalar, BlockParallelConfig, BlockParallelError, EncodeOpts,
};
use strand_quant::{CodebookMode, TrellisConfig};

fn weights(n: usize, mut state: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            let shaped = (unit * 2.0 - 1.0) * (1.0 + (i % 31) as f32 / 16.0);
            if i % 257 == 0 {
                shaped * 8.0
            } else {
                shaped
            }
        })
        .collect()
}

fn parallel(threads: usize) -> BlockParallelConfig {
    BlockParallelConfig::new(threads).unwrap().with_min_blocks(1).with_scratch_budget_bytes(256 * 1024 * 1024)
}

#[test]
fn scalar_matrix_is_byte_identical_in_canonical_order() {
    let option_matrix = [
        EncodeOpts::default(),
        EncodeOpts { adaptive: false, ..EncodeOpts::default() },
        EncodeOpts { tail_biting: true, ..EncodeOpts::default() },
        EncodeOpts { affine_min: true, ..EncodeOpts::default() },
        EncodeOpts { tail_biting: true, affine_min: true, silence_bonus: 0.125, ..EncodeOpts::default() },
        EncodeOpts { silence_bonus: 0.05, entropy_bonus_scale: 0.2, entropy_bonus_two_pass: true, ..EncodeOpts::default() },
    ];
    let mut case = 0u64;
    let configs = [(6, 2, 64, CodebookMode::StoredLut), (8, 3, 256, CodebookMode::HashedQuantile), (10, 4, 512, CodebookMode::ComputedAcklam)];
    for &(l, k, block_len, mode) in &configs {
        let cfg = TrellisConfig::new(l, k, block_len).with_codebook_mode(mode);
        let lengths = [0, 1, block_len - 1, block_len + 1, block_len * 3 + 17];
        for &n in &lengths {
            let w = weights(n, 0xB10C_0000 + case);
            for opts in &option_matrix {
                let lut = cfg.codebook();
                let serial = encode_tensor_with_lut(&w, &cfg, opts, &lut);
                let accelerated = encode_tensor_with_lut_block_parallel(&w, &cfg, opts, &lut, parallel(4)).unwrap();
                assert_eq!(accelerated, serial, "case={case} L={l} k={k} n={n}");
                case += 1;
            }
        }
    }
}

#[test]
fn vector_and_custom_lut_paths_are_byte_identical() {
    for &(d, l, k, n) in &[(2, 6, 2, 2065), (4, 8, 3, 4113)] {
        let cfg = TrellisConfig::new(l, k, 256).with_vec_dim(d);
        let scalar = cfg.codebook();
        let mut lut = vector_lut_from_scalar(&scalar, d as usize);
        // Exercise a deterministic custom/learned-LUT-shaped input rather than
        // only the broadcast table used by the convenience API.
        for (i, value) in lut.iter_mut().enumerate() {
            if i % 7 == 0 {
                *value = value.saturating_add((i % 3) as i32 - 1);
            }
        }
        let w = weights(n, 0xC057_0000 + d as u64);
        for opts in [EncodeOpts::default(), EncodeOpts { tail_biting: true, affine_min: true, ..EncodeOpts::default() }] {
            let serial = encode_tensor_with_lut(&w, &cfg, &opts, &lut);
            let accelerated = encode_tensor_with_lut_block_parallel(&w, &cfg, &opts, &lut, parallel(4)).unwrap();
            assert_eq!(accelerated, serial, "d={d} L={l} k={k}");
        }
    }
}

#[test]
fn dependency_and_configuration_errors_fail_closed() {
    assert_eq!(BlockParallelConfig::new(0), Err(BlockParallelError::ZeroThreads));
    let cfg = TrellisConfig::new(6, 2, 64);
    let w = weights(1024, 0xFA11_C105ED);
    let rolling = EncodeOpts { entropy_bonus_scale: 0.1, entropy_bonus_two_pass: false, ..EncodeOpts::default() };
    assert_eq!(encode_tensor_with_block_parallel(&w, &cfg, &rolling, parallel(4)), Err(BlockParallelError::RollingEntropyDependency));
}

#[test]
fn scratch_budget_reduces_workers_without_changing_bytes() {
    let cfg = TrellisConfig::new(10, 4, 256);
    let w = weights(256 * 12 + 3, 0xB0D6_E7);
    let lut = cfg.codebook();
    let serial = encode_tensor_with_lut(&w, &cfg, &EncodeOpts::default(), &lut);
    let one_worker_budget = BlockParallelConfig::new(16).unwrap().with_min_blocks(1).with_scratch_budget_bytes(1);
    let guarded = encode_tensor_with_lut_block_parallel(&w, &cfg, &EncodeOpts::default(), &lut, one_worker_budget).unwrap();
    assert_eq!(guarded, serial);
}
