#![cfg(feature = "ordered-pipeline")]

use std::sync::{Arc, Mutex};

use strand_quant::encode::{encode_tensor_with_lut, encode_tensor_with_lut_block_parallel, BlockParallelConfig, EncodeOpts, EncodedTensor};
use strand_quant::ordered_pipeline::{run_ordered_pipeline, Accounted, PipelineConfig, PipelineStage};
use strand_quant::rht::{rht_forward_rows, RhtConfig};
use strand_quant::TrellisConfig;

struct RawTensor {
    bytes: Vec<u8>,
    in_features: usize,
    seed: u64,
}

struct PreparedTensor {
    values: Vec<f32>,
}

fn generated_raw(n: usize, mut state: u64, in_features: usize) -> RawTensor {
    let mut bytes = Vec::with_capacity(n * 4);
    for i in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        let value = (unit * 2.0 - 1.0) * (1.0 + (i % 17) as f32 / 8.0);
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    RawTensor { bytes, in_features, seed: state }
}

fn read_prepare(raw: RawTensor) -> PreparedTensor {
    let values: Vec<f32> = raw.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    PreparedTensor { values: rht_forward_rows(&values, &RhtConfig::from_seed(raw.seed), raw.in_features) }
}

fn encoded_wire(enc: &EncodedTensor) -> Vec<u8> {
    let mut wire = Vec::new();
    wire.extend_from_slice(&(enc.total as u64).to_le_bytes());
    wire.extend_from_slice(&(enc.bits.len() as u64).to_le_bytes());
    wire.extend_from_slice(&enc.bits);
    wire.extend_from_slice(&(enc.blocks.len() as u64).to_le_bytes());
    for block in &enc.blocks {
        wire.extend_from_slice(&block.scale_q.to_le_bytes());
        wire.extend_from_slice(&block.min_base_q.to_le_bytes());
        wire.extend_from_slice(&block.init_state.to_le_bytes());
        wire.extend_from_slice(&block.n.to_le_bytes());
        wire.extend_from_slice(&(block.sub_scales.len() as u64).to_le_bytes());
        wire.extend_from_slice(&block.sub_scales);
        wire.extend_from_slice(&(block.mins.len() as u64).to_le_bytes());
        wire.extend_from_slice(&block.mins);
    }
    wire
}

#[test]
fn read_rht_encode_write_pipeline_is_exact_and_ordered() {
    let cfg = TrellisConfig::new(8, 3, 256);
    let lut = cfg.codebook();
    let opts = EncodeOpts { tail_biting: true, affine_min: true, ..EncodeOpts::default() };
    let make_inputs = || (0..7).map(|i| generated_raw(2048 + i * 256, 0xA11C_E000 + i as u64, 256)).collect::<Vec<_>>();

    let serial = make_inputs()
        .into_iter()
        .map(|raw| {
            let prepared = read_prepare(raw);
            encoded_wire(&encode_tensor_with_lut(&prepared.values, &cfg, &opts, &lut))
        })
        .collect::<Vec<_>>();

    let written = Arc::new(Mutex::new(Vec::<(usize, Vec<u8>)>::new()));
    let written_by_sink = Arc::clone(&written);
    let block = BlockParallelConfig::new(4).unwrap().with_min_blocks(1).with_scratch_budget_bytes(64 * 1024 * 1024);
    let stats = run_ordered_pipeline(
        make_inputs(),
        PipelineConfig::new(1, 64 * 1024 * 1024, 8 * 1024 * 1024).unwrap(),
        |_index, raw| {
            let prepared = read_prepare(raw);
            let bytes = prepared.values.len() * std::mem::size_of::<f32>();
            Ok::<_, &'static str>(Accounted::new(prepared, bytes))
        },
        |_index, prepared| {
            let enc = encode_tensor_with_lut_block_parallel(&prepared.values, &cfg, &opts, &lut, block).map_err(|_| "block encode failed")?;
            let wire = encoded_wire(&enc);
            let bytes = wire.len();
            Ok::<_, &'static str>(Accounted::new(wire, bytes))
        },
        move |index, wire| {
            written_by_sink.lock().unwrap().push((index, wire));
            Ok::<_, &'static str>(())
        },
    )
    .unwrap();

    let got = written.lock().unwrap();
    assert_eq!(got.len(), serial.len());
    for (expected_index, ((index, wire), serial_wire)) in got.iter().zip(&serial).enumerate() {
        assert_eq!(*index, expected_index);
        assert_eq!(wire, serial_wire);
    }
    assert_eq!(stats.records_read_prepared, serial.len());
    assert_eq!(stats.records_encoded, serial.len());
    assert_eq!(stats.records_written, serial.len());
    assert!((1..=3).contains(&stats.max_prepared_resident_records));
    assert!((1..=3).contains(&stats.max_encoded_resident_records));
}

#[test]
fn resident_budget_violation_fails_closed() {
    let error = run_ordered_pipeline(
        vec![vec![0u8; 16]],
        PipelineConfig::new(1, 9, 9).unwrap(),
        |_index, bytes| Ok::<_, &'static str>(Accounted::new(bytes, 4)),
        |_index, bytes| Ok::<_, &'static str>(Accounted::new(bytes, 4)),
        |_index, _bytes| Ok::<_, &'static str>(()),
    )
    .unwrap_err();
    assert_eq!(error.stage, PipelineStage::ReadPrepare);
    assert!(error.message.contains("per-slot limit"));
}

#[test]
fn writer_error_is_reported_without_deadlock() {
    let error = run_ordered_pipeline(
        vec![0usize, 1, 2, 3],
        PipelineConfig::new(1, 1024, 1024).unwrap(),
        |_index, value| Ok::<_, &'static str>(Accounted::new(value, 8)),
        |_index, value| Ok::<_, &'static str>(Accounted::new(value, 8)),
        |index, _value| {
            if index == 2 {
                Err("synthetic writer failure")
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert_eq!(error.stage, PipelineStage::Write);
    assert_eq!(error.index, Some(2));
}
