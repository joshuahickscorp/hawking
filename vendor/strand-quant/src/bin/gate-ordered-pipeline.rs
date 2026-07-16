#![forbid(unsafe_code)]

use std::fs;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use strand_quant::encode::{encode_tensor_with_lut_block_parallel, BlockParallelConfig, EncodeOpts, EncodedTensor};
use strand_quant::ordered_pipeline::{run_ordered_pipeline, Accounted, PipelineConfig};
use strand_quant::rht::{rht_forward_rows, RhtConfig};
use strand_quant::sha256::sha256;
use strand_quant::TrellisConfig;

const SCHEMA: &str = "hawking.strand.ordered-pipeline-parity.v1";

struct Args {
    work_dir: PathBuf,
    receipt: PathBuf,
    threads: usize,
    depth: usize,
    prepared_budget_bytes: usize,
    encoded_budget_bytes: usize,
}

fn parse_args() -> Args {
    let mut work_dir = None;
    let mut receipt = None;
    let mut threads = 8usize;
    let mut depth = 1usize;
    let mut prepared_budget_bytes = 64 * 1024 * 1024usize;
    let mut encoded_budget_bytes = 16 * 1024 * 1024usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--work-dir" => work_dir = Some(args.next().expect("--work-dir needs DIR").into()),
            "--receipt" => receipt = Some(args.next().expect("--receipt needs PATH").into()),
            "--threads" => threads = args.next().expect("--threads needs N").parse().expect("threads usize"),
            "--depth" => depth = args.next().expect("--depth needs N").parse().expect("depth usize"),
            "--prepared-budget-bytes" => prepared_budget_bytes = args.next().expect("--prepared-budget-bytes needs BYTES").parse().expect("prepared budget usize"),
            "--encoded-budget-bytes" => encoded_budget_bytes = args.next().expect("--encoded-budget-bytes needs BYTES").parse().expect("encoded budget usize"),
            "-h" | "--help" => {
                println!(
                    "gate-ordered-pipeline --work-dir DIR --receipt PATH [--threads N] \
                     [--depth N] [--prepared-budget-bytes BYTES] \
                     [--encoded-budget-bytes BYTES]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(threads > 0, "--threads must be greater than zero");
    Args { work_dir: work_dir.expect("--work-dir is required"), receipt: receipt.expect("--receipt is required"), threads, depth, prepared_budget_bytes, encoded_budget_bytes }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

fn file_sha256(path: &Path) -> String {
    hex(&sha256(&fs::read(path).expect("read file for sha256")))
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn generated_bytes(n: usize, mut state: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * 4);
    for i in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        let value = (unit * 2.0 - 1.0) * (1.0 + (i % 29) as f32 / 16.0);
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
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

fn write_record(writer: &mut impl std::io::Write, record: &[u8]) -> std::io::Result<()> {
    writer.write_all(&(record.len() as u64).to_le_bytes())?;
    writer.write_all(record)
}

#[derive(Clone)]
struct Source {
    path: PathBuf,
    seed: u64,
    in_features: usize,
}

struct Prepared {
    values: Vec<f32>,
}

fn prepare(source: &Source) -> std::io::Result<Prepared> {
    let bytes = fs::read(&source.path)?;
    let values = decode_f32(&bytes);
    Ok(Prepared { values: rht_forward_rows(&values, &RhtConfig::from_seed(source.seed), source.in_features) })
}

fn main() {
    let args = parse_args();
    fs::create_dir_all(&args.work_dir).expect("create work dir");
    let sources = (0..8)
        .map(|index| {
            let path = args.work_dir.join(format!("source-{index:02}.f32le"));
            fs::write(&path, generated_bytes(4096 + index * 256, 0x51A6_E000 + index as u64)).expect("write synthetic source");
            Source { path, seed: 0xA117_0000 + index as u64, in_features: 256 }
        })
        .collect::<Vec<_>>();

    let cfg = TrellisConfig::new(8, 3, 256);
    let lut = cfg.codebook();
    let opts = EncodeOpts { tail_biting: true, affine_min: true, ..EncodeOpts::default() };
    let block = BlockParallelConfig::new(args.threads).unwrap().with_min_blocks(1).with_scratch_budget_bytes(256 * 1024 * 1024);
    let serial_path = args.work_dir.join("serial.records");
    let serial_start = Instant::now();
    {
        let mut writer = BufWriter::new(fs::File::create(&serial_path).expect("serial output"));
        for source in &sources {
            let prepared = prepare(source).expect("serial read/prepare");
            let encoded = encode_tensor_with_lut_block_parallel(&prepared.values, &cfg, &opts, &lut, block).expect("serial staged block encode");
            write_record(&mut writer, &encoded_wire(&encoded)).expect("serial write");
        }
        writer.flush().expect("flush serial output");
    }
    let serial_ns = serial_start.elapsed().as_nanos();

    let pipeline_path = args.work_dir.join("pipeline.records");
    let writer = fs::File::create(&pipeline_path).expect("pipeline output");
    let pipeline_start = Instant::now();
    let mut writer = writer;
    let stats = run_ordered_pipeline(
        sources.clone(),
        PipelineConfig::new(args.depth, args.prepared_budget_bytes, args.encoded_budget_bytes).expect("pipeline config"),
        |_index, source| {
            let prepared = prepare(&source)?;
            let bytes = prepared.values.len() * std::mem::size_of::<f32>();
            Ok::<_, std::io::Error>(Accounted::new(prepared, bytes))
        },
        |_index, prepared| {
            let encoded = encode_tensor_with_lut_block_parallel(&prepared.values, &cfg, &opts, &lut, block).map_err(std::io::Error::other)?;
            let wire = encoded_wire(&encoded);
            let bytes = wire.len();
            Ok::<_, std::io::Error>(Accounted::new(wire, bytes))
        },
        move |_index, record| write_record(&mut writer, &record),
    )
    .expect("ordered pipeline");
    let pipeline_ns = pipeline_start.elapsed().as_nanos();

    let serial_sha256 = file_sha256(&serial_path);
    let pipeline_sha256 = file_sha256(&pipeline_path);
    assert_eq!(pipeline_sha256, serial_sha256, "pipeline output must be exact");
    assert_eq!(stats.records_written, sources.len());
    assert!(stats.max_prepared_resident_records <= args.depth + 2);
    assert!(stats.max_encoded_resident_records <= args.depth + 2);

    let binary = std::env::current_exe().expect("current exe");
    let binary_sha256 = file_sha256(&binary);
    let mut source_manifest = Vec::new();
    for source in &sources {
        source_manifest.extend_from_slice(file_sha256(&source.path).as_bytes());
    }
    let source_manifest_sha256 = hex(&sha256(&source_manifest));
    let speedup = serial_ns as f64 / pipeline_ns.max(1) as f64;
    let generated_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock before epoch").as_nanos();
    let receipt = format!(
        "{{\n  \"schema\": \"{SCHEMA}\",\n  \"status\": \"pass\",\n  \"scope\": \"synthetic_only\",\n  \"generated_unix_ns\": {generated_unix_ns},\n  \"binary\": \"{}\",\n  \"binary_sha256\": \"{binary_sha256}\",\n  \"source_manifest_sha256\": \"{source_manifest_sha256}\",\n  \"threads\": {},\n  \"depth\": {},\n  \"prepared_resident_budget_bytes\": {},\n  \"encoded_resident_budget_bytes\": {},\n  \"records\": {},\n  \"max_prepared_resident_records\": {},\n  \"max_encoded_resident_records\": {},\n  \"serial_output_sha256\": \"{serial_sha256}\",\n  \"pipeline_output_sha256\": \"{pipeline_sha256}\",\n  \"exact_output\": true,\n  \"canonical_order\": true,\n  \"serial_ns\": {serial_ns},\n  \"pipeline_ns\": {pipeline_ns},\n  \"exploratory_speedup\": {speedup:.6}\n}}\n",
        json_escape(&binary.display().to_string()),
        args.threads,
        args.depth,
        args.prepared_budget_bytes,
        args.encoded_budget_bytes,
        sources.len(),
        stats.max_prepared_resident_records,
        stats.max_encoded_resident_records,
    );
    fs::write(&args.receipt, &receipt).expect("write receipt");
    print!("{receipt}");
}
