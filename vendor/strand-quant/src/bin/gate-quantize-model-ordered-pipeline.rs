#![forbid(unsafe_code)]

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use strand_quant::format::read_strand_v2;
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::sha256::sha256;

const SCHEMA: &str = "hawking.strand.quantize-model-ordered-pipeline-parity.v1";
const Q_NAME: &str = "model.layers.0.self_attn.q_proj.weight";
const DOWN_NAME: &str = "model.layers.0.mlp.down_proj.weight";
const NORM_NAME: &str = "model.layers.0.input_layernorm.weight";

struct Args {
    serial_binary: PathBuf,
    pipeline_binary: PathBuf,
    work_dir: PathBuf,
    receipt: PathBuf,
    block_threads: usize,
    scratch_budget_bytes: usize,
    depth: usize,
    prepared_budget_bytes: usize,
    encoded_budget_bytes: usize,
    pipeline_native_io: Option<String>,
}

fn parse_args() -> Args {
    let mut serial_binary = None;
    let mut pipeline_binary = None;
    let mut work_dir = None;
    let mut receipt = None;
    let mut block_threads = 4usize;
    let mut scratch_budget_bytes = 256 * 1024 * 1024usize;
    let mut depth = 1usize;
    let mut prepared_budget_bytes = 64 * 1024 * 1024usize;
    let mut encoded_budget_bytes = 64 * 1024 * 1024usize;
    let mut pipeline_native_io = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--serial-binary" => {
                serial_binary = Some(args.next().expect("--serial-binary needs PATH").into())
            }
            "--pipeline-binary" => {
                pipeline_binary = Some(args.next().expect("--pipeline-binary needs PATH").into())
            }
            "--work-dir" => work_dir = Some(args.next().expect("--work-dir needs DIR").into()),
            "--receipt" => receipt = Some(args.next().expect("--receipt needs PATH").into()),
            "--block-threads" => {
                block_threads = args
                    .next()
                    .expect("--block-threads needs N")
                    .parse()
                    .expect("block threads usize")
            }
            "--scratch-budget-bytes" => {
                scratch_budget_bytes = args
                    .next()
                    .expect("--scratch-budget-bytes needs BYTES")
                    .parse()
                    .expect("scratch budget usize")
            }
            "--depth" => {
                depth = args
                    .next()
                    .expect("--depth needs N")
                    .parse()
                    .expect("depth usize")
            }
            "--prepared-budget-bytes" => {
                prepared_budget_bytes = args
                    .next()
                    .expect("--prepared-budget-bytes needs BYTES")
                    .parse()
                    .expect("prepared budget usize")
            }
            "--encoded-budget-bytes" => {
                encoded_budget_bytes = args
                    .next()
                    .expect("--encoded-budget-bytes needs BYTES")
                    .parse()
                    .expect("encoded budget usize")
            }
            "--pipeline-native-io" => {
                let mode = args.next().expect("--pipeline-native-io needs MODE");
                assert!(
                    matches!(mode.as_str(), "preallocated" | "mmap"),
                    "pipeline native I/O must be preallocated or mmap"
                );
                pipeline_native_io = Some(mode);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(block_threads > 0 && depth > 0);
    assert!(scratch_budget_bytes > 0 && prepared_budget_bytes > 0 && encoded_budget_bytes > 0);
    Args {
        serial_binary: serial_binary.expect("--serial-binary is required"),
        pipeline_binary: pipeline_binary.expect("--pipeline-binary is required"),
        work_dir: work_dir.expect("--work-dir is required"),
        receipt: receipt.expect("--receipt is required"),
        block_threads,
        scratch_budget_bytes,
        depth,
        prepared_budget_bytes,
        encoded_budget_bytes,
        pipeline_native_io,
    }
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
    hex(&sha256(&fs::read(path).unwrap_or_else(|error| {
        panic!("read {}: {error}", path.display())
    })))
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn generated_weights(n: usize, mut state: u64) -> Vec<f32> {
    (0..n)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * (1.0 + (index % 23) as f32 / 16.0)
        })
        .collect()
}

fn write_fixture(path: &Path) {
    let q = generated_weights(32 * 256, 0x0ADE_0001);
    let down = generated_weights(16 * 256, 0x0ADE_0002);
    let norm = generated_weights(256, 0x0ADE_0003);
    let q_bytes = q.len() * 4;
    let down_bytes = down.len() * 4;
    let norm_bytes = norm.len() * 4;
    let mut header = format!(
        "{{\"{Q_NAME}\":{{\"dtype\":\"F32\",\"shape\":[32,256],\"data_offsets\":[0,{q_bytes}]}},\"{DOWN_NAME}\":{{\"dtype\":\"F32\",\"shape\":[16,256],\"data_offsets\":[{q_bytes},{}]}},\"{NORM_NAME}\":{{\"dtype\":\"F32\",\"shape\":[256],\"data_offsets\":[{},{}]}}}}",
        q_bytes + down_bytes,
        q_bytes + down_bytes,
        q_bytes + down_bytes + norm_bytes,
    );
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }
    let mut file = fs::File::create(path).expect("create fixture");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for value in q.iter().chain(&down).chain(&norm) {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
}

fn run(binary: &Path, argv: &[String]) -> u128 {
    let started = Instant::now();
    let output = Command::new(binary)
        .args(argv)
        .env("STRAND_NO_GPU", "1")
        .output()
        .unwrap_or_else(|error| panic!("launch {}: {error}", binary.display()));
    if !output.status.success() {
        panic!(
            "{} failed with {}\nstdout:\n{}\nstderr:\n{}",
            binary.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    started.elapsed().as_nanos()
}

struct Outputs {
    dense: PathBuf,
    sidecar: PathBuf,
    archive: PathBuf,
    dense_wall_ns: u128,
    archive_wall_ns: u128,
}

fn run_variant(binary: &Path, fixture: &Path, args: &Args, label: &str, pipeline: bool) -> Outputs {
    let partial = pipeline && args.pipeline_native_io.is_some();
    let dense = args.work_dir.join(if partial {
        format!("{label}.safetensors.partial")
    } else {
        format!("{label}.safetensors")
    });
    let sidecar = PathBuf::from(format!("{}.json", dense.display()));
    let archive = args.work_dir.join(if partial {
        format!("{label}.strand.partial")
    } else {
        format!("{label}.strand")
    });
    let mut common = vec![
        "--input".into(),
        fixture.display().to_string(),
        "--bits".into(),
        "2".into(),
        "--l".into(),
        "8".into(),
        "--threads".into(),
        "1".into(),
        "--block-threads".into(),
        args.block_threads.to_string(),
        "--block-scratch-budget-bytes".into(),
        args.scratch_budget_bytes.to_string(),
    ];
    if pipeline {
        common.extend([
            "--ordered-pipeline-depth".into(),
            args.depth.to_string(),
            "--ordered-pipeline-prepared-budget-bytes".into(),
            args.prepared_budget_bytes.to_string(),
            "--ordered-pipeline-encoded-budget-bytes".into(),
            args.encoded_budget_bytes.to_string(),
        ]);
        if let Some(mode) = &args.pipeline_native_io {
            common.extend(["--native-io".into(), mode.clone()]);
        }
    }
    let mut dense_args = common.clone();
    dense_args.extend(["--output".into(), dense.display().to_string()]);
    let dense_wall_ns = run(binary, &dense_args);
    common.extend(["--packed-v2-out".into(), archive.display().to_string()]);
    let archive_wall_ns = run(binary, &common);
    Outputs {
        dense,
        sidecar,
        archive,
        dense_wall_ns,
        archive_wall_ns,
    }
}

fn main() {
    let args = parse_args();
    assert!(args.serial_binary.is_file());
    assert!(args.pipeline_binary.is_file());
    fs::create_dir_all(&args.work_dir).expect("create work dir");
    let fixture = args.work_dir.join("ordered-pipeline-canary.safetensors");
    write_fixture(&fixture);

    let serial = run_variant(&args.serial_binary, &fixture, &args, "serial", false);
    let pipeline = run_variant(&args.pipeline_binary, &fixture, &args, "pipeline", true);
    let dense_sha = file_sha256(&serial.dense);
    let sidecar_sha = file_sha256(&serial.sidecar);
    let archive_sha = file_sha256(&serial.archive);
    assert_eq!(file_sha256(&pipeline.dense), dense_sha, "dense mismatch");
    assert_eq!(
        file_sha256(&pipeline.sidecar),
        sidecar_sha,
        "sidecar mismatch"
    );
    assert_eq!(file_sha256(&pipeline.archive), archive_sha, "STR2 mismatch");

    let dense_order = SafeTensors::open(pipeline.dense.to_str().unwrap())
        .expect("read pipeline dense")
        .order;
    assert_eq!(dense_order, [Q_NAME, DOWN_NAME, NORM_NAME]);
    let archive_order = read_strand_v2(&fs::read(&pipeline.archive).expect("read pipeline STR2"))
        .expect("parse pipeline STR2")
        .into_iter()
        .map(|tensor| tensor.base.name)
        .collect::<Vec<_>>();
    assert_eq!(archive_order, [Q_NAME, DOWN_NAME]);

    let mut output_bundle = Vec::new();
    for value in [&dense_sha, &sidecar_sha, &archive_sha] {
        output_bundle.extend_from_slice(value.as_bytes());
        output_bundle.push(0);
    }
    let output_bundle_sha = hex(&sha256(&output_bundle));

    let generated_unix_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    let receipt = format!(
        "{{\n  \"schema\": \"{SCHEMA}\",\n  \"status\": \"pass\",\n  \"scope\": \"synthetic_only\",\n  \"generated_unix_ns\": {generated_unix_ns},\n  \"serial_binary\": \"{}\",\n  \"serial_binary_sha256\": \"{}\",\n  \"pipeline_binary\": \"{}\",\n  \"pipeline_binary_sha256\": \"{}\",\n  \"fixture_sha256\": \"{}\",\n  \"pipeline_native_io\": \"{}\",\n  \"invocation_identity\": \"STRAND_NO_GPU=1;bits=2;l=8;rht=rows;outer_threads=1;block_threads={};scratch={};depth={}\",\n  \"depth\": {},\n  \"prepared_budget_bytes\": {},\n  \"encoded_budget_bytes\": {},\n  \"scratch_budget_bytes\": {},\n  \"dense_output_sha256\": \"{dense_sha}\",\n  \"dense_exact_match\": true,\n  \"sidecar_sha256\": \"{sidecar_sha}\",\n  \"sidecar_exact_match\": true,\n  \"packed_v2_archive_sha256\": \"{archive_sha}\",\n  \"packed_v2_exact_match\": true,\n  \"canonical_order\": true,\n  \"exact_output\": true,\n  \"measurements\": {{\"serial_dense_wall_ns\": {}, \"pipeline_dense_wall_ns\": {}, \"serial_archive_wall_ns\": {}, \"pipeline_archive_wall_ns\": {}, \"read_decode_ns\": null, \"rht_preprocess_ns\": null, \"encode_ns\": null, \"finalize_write_ns\": null, \"cpu_time_ns\": null, \"gpu_time_ns\": null, \"peak_rss_bytes\": null, \"swap_delta_bytes\": null, \"scratch_peak_bytes\": null, \"disk_read_bytes\": null, \"disk_write_bytes\": null, \"thermal_start\": null, \"thermal_end\": null, \"phase_instrumentation_complete\": false}},\n  \"input_bundle_sha256\": \"{}\",\n  \"output_bundle_sha256\": \"{output_bundle_sha}\",\n  \"scientific_receipt_bundle_sha256\": null,\n  \"component_speedup_is_eta_evidence\": false,\n  \"production_promotion_allowed\": false\n}}\n",
        json_escape(&args.serial_binary.display().to_string()),
        file_sha256(&args.serial_binary),
        json_escape(&args.pipeline_binary.display().to_string()),
        file_sha256(&args.pipeline_binary),
        file_sha256(&fixture),
        args.pipeline_native_io.as_deref().unwrap_or("standard"),
        args.block_threads,
        args.scratch_budget_bytes,
        args.depth,
        args.depth,
        args.prepared_budget_bytes,
        args.encoded_budget_bytes,
        args.scratch_budget_bytes,
        serial.dense_wall_ns,
        pipeline.dense_wall_ns,
        serial.archive_wall_ns,
        pipeline.archive_wall_ns,
        file_sha256(&fixture),
    );
    fs::write(&args.receipt, &receipt).expect("write receipt");
    print!("{receipt}");
}
