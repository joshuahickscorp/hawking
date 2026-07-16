#![forbid(unsafe_code)]

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use strand_quant::sha256::sha256;

const SCHEMA: &str = "hawking.strand.quantize-model-block-parallel-parity.v1";

struct Args {
    canonical_binary: PathBuf,
    parallel_binary: PathBuf,
    work_dir: PathBuf,
    receipt: PathBuf,
    block_threads: usize,
    scratch_budget_bytes: usize,
}

fn parse_args() -> Args {
    let mut canonical_binary = None;
    let mut parallel_binary = None;
    let mut work_dir = None;
    let mut receipt = None;
    let mut block_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut scratch_budget_bytes = 256 * 1024 * 1024usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--canonical-binary" => canonical_binary = Some(args.next().expect("--canonical-binary needs PATH").into()),
            "--parallel-binary" => parallel_binary = Some(args.next().expect("--parallel-binary needs PATH").into()),
            "--work-dir" => work_dir = Some(args.next().expect("--work-dir needs PATH").into()),
            "--receipt" => receipt = Some(args.next().expect("--receipt needs PATH").into()),
            "--block-threads" => block_threads = args.next().expect("--block-threads needs N").parse().expect("block-threads usize"),
            "--scratch-budget-bytes" => scratch_budget_bytes = args.next().expect("--scratch-budget-bytes needs BYTES").parse().expect("scratch-budget-bytes usize"),
            "-h" | "--help" => {
                println!(
                    "gate-quantize-model-block-parallel --canonical-binary PATH \
                     --parallel-binary PATH --work-dir DIR --receipt PATH \
                     [--block-threads N] [--scratch-budget-bytes BYTES]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(block_threads > 0, "--block-threads must be greater than zero");
    assert!(scratch_budget_bytes > 0, "--scratch-budget-bytes must be greater than zero");
    Args {
        canonical_binary: canonical_binary.expect("--canonical-binary is required"),
        parallel_binary: parallel_binary.expect("--parallel-binary is required"),
        work_dir: work_dir.expect("--work-dir is required"),
        receipt: receipt.expect("--receipt is required"),
        block_threads,
        scratch_budget_bytes,
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
    hex(&sha256(&fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))))
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn generated_weights(n: usize, mut state: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            let value = (unit * 2.0 - 1.0) * (1.0 + (i % 23) as f32 / 16.0);
            if i % 257 == 0 {
                value * 4.0
            } else {
                value
            }
        })
        .collect()
}

fn write_fixture(path: &Path) {
    let q = generated_weights(32 * 256, 0xCA11_0001);
    let down = generated_weights(16 * 256, 0xCA11_0002);
    let norm = generated_weights(256, 0xCA11_0003);
    let q_bytes = q.len() * 4;
    let down_bytes = down.len() * 4;
    let norm_bytes = norm.len() * 4;
    let mut header = format!(
        "{{\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"F32\",\"shape\":[32,256],\"data_offsets\":[0,{q_bytes}]}},\"model.layers.0.mlp.down_proj.weight\":{{\"dtype\":\"F32\",\"shape\":[16,256],\"data_offsets\":[{q_bytes},{}]}},\"model.layers.0.input_layernorm.weight\":{{\"dtype\":\"F32\",\"shape\":[256],\"data_offsets\":[{},{}]}}}}",
        q_bytes + down_bytes,
        q_bytes + down_bytes,
        q_bytes + down_bytes + norm_bytes,
    );
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }
    let mut file = fs::File::create(path).unwrap_or_else(|e| panic!("create fixture {}: {e}", path.display()));
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for value in q.iter().chain(&down).chain(&norm) {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
}

fn run(binary: &Path, argv: &[String]) {
    let output = Command::new(binary).args(argv).env("STRAND_NO_GPU", "1").output().unwrap_or_else(|e| panic!("launch {}: {e}", binary.display()));
    if !output.status.success() {
        panic!("{} failed with {}\nstdout:\n{}\nstderr:\n{}", binary.display(), output.status, String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr),);
    }
}

struct Outputs {
    dense: PathBuf,
    sidecar: PathBuf,
    archive: PathBuf,
}

fn run_variant(binary: &Path, fixture: &Path, work_dir: &Path, label: &str, parallel: Option<(usize, usize)>) -> Outputs {
    let dense = work_dir.join(format!("{label}.safetensors"));
    let sidecar = PathBuf::from(format!("{}.json", dense.display()));
    let archive = work_dir.join(format!("{label}.strand"));
    let mut common = vec!["--input".into(), fixture.display().to_string(), "--bits".into(), "2".into(), "--l".into(), "8".into(), "--no-rht".into(), "--threads".into(), "1".into()];
    if let Some((threads, budget)) = parallel {
        common.extend(["--block-threads".into(), threads.to_string(), "--block-scratch-budget-bytes".into(), budget.to_string()]);
    }
    let mut dense_argv = common.clone();
    dense_argv.extend(["--output".into(), dense.display().to_string()]);
    run(binary, &dense_argv);

    let mut archive_argv = common;
    archive_argv.extend(["--packed-v2-out".into(), archive.display().to_string()]);
    run(binary, &archive_argv);
    Outputs { dense, sidecar, archive }
}

fn main() {
    let args = parse_args();
    assert!(args.canonical_binary.is_file(), "canonical binary is not a file");
    assert!(args.parallel_binary.is_file(), "parallel binary is not a file");
    fs::create_dir_all(&args.work_dir).unwrap_or_else(|e| panic!("create {}: {e}", args.work_dir.display()));
    let fixture = args.work_dir.join("block-parallel-canary.safetensors");
    write_fixture(&fixture);

    let canonical = run_variant(&args.canonical_binary, &fixture, &args.work_dir, "canonical", None);
    let parallel = run_variant(&args.parallel_binary, &fixture, &args.work_dir, "block-parallel", Some((args.block_threads, args.scratch_budget_bytes)));
    let canonical_dense = file_sha256(&canonical.dense);
    let parallel_dense = file_sha256(&parallel.dense);
    let canonical_sidecar = file_sha256(&canonical.sidecar);
    let parallel_sidecar = file_sha256(&parallel.sidecar);
    let canonical_archive = file_sha256(&canonical.archive);
    let parallel_archive = file_sha256(&parallel.archive);
    assert_eq!(parallel_dense, canonical_dense, "dense output SHA mismatch");
    assert_eq!(parallel_sidecar, canonical_sidecar, "sidecar SHA mismatch");
    assert_eq!(parallel_archive, canonical_archive, "packed-v2 SHA mismatch");

    let canonical_binary_sha256 = file_sha256(&args.canonical_binary);
    let parallel_binary_sha256 = file_sha256(&args.parallel_binary);
    let fixture_sha256 = file_sha256(&fixture);
    let invocation =
        format!("STRAND_NO_GPU=1;bits=2;l=8;rht=off;tensor_scope=linear_default;outer_threads=1;block_threads={};block_scratch_budget_bytes={}", args.block_threads, args.scratch_budget_bytes,);
    let mut payload = Vec::new();
    for value in [SCHEMA, &canonical_binary_sha256, &parallel_binary_sha256, &fixture_sha256, &canonical_dense, &canonical_sidecar, &canonical_archive, &invocation] {
        payload.extend_from_slice(value.as_bytes());
        payload.push(0);
    }
    let payload_sha256 = hex(&sha256(&payload));
    let generated_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before epoch").as_nanos();
    let receipt = format!(
        "{{\n  \"schema\": \"{SCHEMA}\",\n  \"status\": \"pass\",\n  \"generated_unix_ns\": {generated_unix_ns},\n  \"canonical_binary\": \"{}\",\n  \"canonical_binary_sha256\": \"{canonical_binary_sha256}\",\n  \"parallel_binary\": \"{}\",\n  \"parallel_binary_sha256\": \"{parallel_binary_sha256}\",\n  \"fixture\": \"{}\",\n  \"fixture_sha256\": \"{fixture_sha256}\",\n  \"invocation_contract\": \"{}\",\n  \"dense_output_sha256\": \"{canonical_dense}\",\n  \"dense_exact_match\": true,\n  \"sidecar_sha256\": \"{canonical_sidecar}\",\n  \"sidecar_exact_match\": true,\n  \"packed_v2_archive_sha256\": \"{canonical_archive}\",\n  \"packed_v2_exact_match\": true,\n  \"canonical_payload_sha256\": \"{payload_sha256}\"\n}}\n",
        json_escape(&args.canonical_binary.display().to_string()),
        json_escape(&args.parallel_binary.display().to_string()),
        json_escape(&fixture.display().to_string()),
        json_escape(&invocation),
    );
    fs::write(&args.receipt, &receipt).unwrap_or_else(|e| panic!("write {}: {e}", args.receipt.display()));
    print!("{receipt}");
}
