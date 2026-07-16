#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use strand_quant::encode::{encode_tensor_with_lut, encode_tensor_with_lut_block_parallel, BlockParallelConfig, EncodeOpts, EncodedTensor};
use strand_quant::sha256::sha256;
use strand_quant::TrellisConfig;

const SCHEMA: &str = "hawking.strand.block-parallel-parity.v1";

struct Args {
    threads: usize,
    scratch_budget_bytes: usize,
    receipt: Option<PathBuf>,
    benchmark_weights: usize,
    benchmark_iters: usize,
}

fn parse_args() -> Args {
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut scratch_budget_bytes = 256 * 1024 * 1024usize;
    let mut receipt = None;
    let mut benchmark_weights = 131_072usize;
    let mut benchmark_iters = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => threads = args.next().expect("--threads needs N").parse().expect("threads usize"),
            "--scratch-budget-bytes" => scratch_budget_bytes = args.next().expect("--scratch-budget-bytes needs BYTES").parse().expect("scratch-budget-bytes usize"),
            "--receipt" => receipt = Some(args.next().expect("--receipt needs PATH").into()),
            "--benchmark-weights" => benchmark_weights = args.next().expect("--benchmark-weights needs N").parse().expect("benchmark-weights usize"),
            "--benchmark-iters" => benchmark_iters = args.next().expect("--benchmark-iters needs N").parse().expect("benchmark-iters usize"),
            "-h" | "--help" => {
                println!(
                    "gate-block-parallel [--threads N] [--scratch-budget-bytes BYTES] \
                     [--benchmark-weights N] [--benchmark-iters N] [--receipt PATH]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(threads > 0, "--threads must be greater than zero");
    assert!(scratch_budget_bytes > 0, "--scratch-budget-bytes must be greater than zero");
    assert!(benchmark_weights > 0, "--benchmark-weights must be greater than zero");
    assert!(benchmark_iters > 0, "--benchmark-iters must be greater than zero");
    Args { threads, scratch_budget_bytes, receipt, benchmark_weights, benchmark_iters }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

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

fn hash_weights(values: &[f32]) -> String {
    let mut wire = Vec::with_capacity(values.len() * 4);
    for value in values {
        wire.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    hex(&sha256(&wire))
}

fn hash_encoded(enc: &EncodedTensor) -> String {
    let mut wire = Vec::new();
    wire.extend_from_slice(&(enc.total as u64).to_le_bytes());
    wire.push(enc.has_rht_seed as u8);
    wire.push(enc.tail_biting as u8);
    wire.push(enc.has_affine_min as u8);
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
    hex(&sha256(&wire))
}

struct CaseReceipt {
    name: String,
    input_sha256: String,
    serial_sha256: String,
    parallel_sha256: String,
    serial_ns: u128,
    parallel_ns: u128,
}

fn run_case(name: &str, values: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts, parallel: BlockParallelConfig, iters: usize) -> CaseReceipt {
    let serial_start = Instant::now();
    let mut serial = None;
    for _ in 0..iters {
        let lut = cfg.codebook();
        serial = Some(encode_tensor_with_lut(values, cfg, opts, &lut));
    }
    let serial_ns = serial_start.elapsed().as_nanos();
    let serial = serial.unwrap();

    let parallel_start = Instant::now();
    let mut accelerated = None;
    for _ in 0..iters {
        let lut = cfg.codebook();
        accelerated = Some(encode_tensor_with_lut_block_parallel(values, cfg, opts, &lut, parallel).unwrap_or_else(|e| panic!("{name}: block-parallel rejected: {e}")));
    }
    let parallel_ns = parallel_start.elapsed().as_nanos();
    let accelerated = accelerated.unwrap();

    let serial_sha256 = hash_encoded(&serial);
    let parallel_sha256 = hash_encoded(&accelerated);
    assert_eq!(accelerated, serial, "{name}: canonical bytes differ");
    assert_eq!(parallel_sha256, serial_sha256, "{name}: receipt digest differs");
    CaseReceipt { name: name.into(), input_sha256: hash_weights(values), serial_sha256, parallel_sha256, serial_ns, parallel_ns }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let args = parse_args();
    let parallel = BlockParallelConfig::new(args.threads).unwrap().with_min_blocks(1).with_scratch_budget_bytes(args.scratch_budget_bytes);

    let cases = [
        ("q2_l8_default", TrellisConfig::new(8, 2, 256), EncodeOpts::default(), 0xB10C_0001),
        ("q3_l9_tail_affine", TrellisConfig::new(9, 3, 256), EncodeOpts { tail_biting: true, affine_min: true, ..EncodeOpts::default() }, 0xB10C_0002),
        ("q4_l10_two_pass_psi", TrellisConfig::new(10, 4, 256), EncodeOpts { silence_bonus: 0.05, entropy_bonus_scale: 0.2, entropy_bonus_two_pass: true, ..EncodeOpts::default() }, 0xB10C_0003),
    ];
    let mut receipts = Vec::new();
    for (name, cfg, opts, seed) in cases {
        let values = weights(args.benchmark_weights + 17, seed);
        receipts.push(run_case(name, &values, &cfg, &opts, parallel, args.benchmark_iters));
    }

    let exe = std::env::current_exe().expect("current executable");
    let exe_sha256 = hex(&sha256(&fs::read(&exe).expect("read current executable")));
    let mut payload = Vec::new();
    payload.extend_from_slice(SCHEMA.as_bytes());
    payload.extend_from_slice(exe_sha256.as_bytes());
    payload.extend_from_slice(&(args.threads as u64).to_le_bytes());
    payload.extend_from_slice(&(args.scratch_budget_bytes as u64).to_le_bytes());
    for case in &receipts {
        payload.extend_from_slice(case.name.as_bytes());
        payload.extend_from_slice(case.input_sha256.as_bytes());
        payload.extend_from_slice(case.serial_sha256.as_bytes());
        payload.extend_from_slice(case.parallel_sha256.as_bytes());
    }
    let payload_sha256 = hex(&sha256(&payload));
    let generated_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before epoch").as_nanos();

    let mut case_json = Vec::new();
    for case in &receipts {
        let speedup = case.serial_ns as f64 / case.parallel_ns.max(1) as f64;
        case_json.push(format!(
            "{{\"name\":\"{}\",\"input_sha256\":\"{}\",\"serial_sha256\":\"{}\",\"parallel_sha256\":\"{}\",\"exact_match\":true,\"serial_ns\":{},\"parallel_ns\":{},\"speedup\":{:.6}}}",
            json_escape(&case.name),
            case.input_sha256,
            case.serial_sha256,
            case.parallel_sha256,
            case.serial_ns,
            case.parallel_ns,
            speedup,
        ));
    }
    let receipt = format!(
        "{{\n  \"schema\": \"{SCHEMA}\",\n  \"status\": \"pass\",\n  \"generated_unix_ns\": {generated_unix_ns},\n  \"feature\": \"block-parallel\",\n  \"binary_path\": \"{}\",\n  \"binary_sha256\": \"{exe_sha256}\",\n  \"threads\": {},\n  \"scratch_budget_bytes\": {},\n  \"benchmark_weights\": {},\n  \"benchmark_iters\": {},\n  \"case_count\": {},\n  \"cases\": [{}],\n  \"canonical_payload_sha256\": \"{payload_sha256}\"\n}}\n",
        json_escape(&exe.display().to_string()),
        args.threads,
        args.scratch_budget_bytes,
        args.benchmark_weights,
        args.benchmark_iters,
        receipts.len(),
        case_json.join(","),
    );
    if let Some(path) = args.receipt {
        fs::write(&path, &receipt).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
    print!("{receipt}");
}
