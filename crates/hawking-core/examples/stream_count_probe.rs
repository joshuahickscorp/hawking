//! Stream-count ladder for the MLP ARM A ceiling.
//!
//! Hold bytes per thread-iteration at 38, strip arithmetic, vary only how
//! those bytes are split into concurrent address streams. Alignment and
//! per-thread stride run in the same process as a second discriminator.
//! Does not mutate the production decode path.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example stream_count_probe
//! ./tools/gpu_lane_lock.sh z2stream \
//!   workspace/ops/build/rust/release-fast/examples/stream_count_probe \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_MLP_STREAM_COUNT_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: stream_count_probe --artifact-root DIR \
        [--layer N] [--warmup N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("stream_count_probe: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    layer: usize,
    warmup: usize,
    reps: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut layer = 0usize;
    let mut warmup = 5usize;
    let mut reps = 11usize;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--layer" => {
                layer = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--layer"));
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--warmup"));
            }
            "--reps" => {
                reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--reps"));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if reps == 0 {
        fail("--reps must be > 0");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        layer,
        warmup,
        reps,
        out,
    }
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["--no-optional-locks", "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn cmd_stdout(args: &[&str]) -> String {
    std::process::Command::new(args[0])
        .args(&args[1..])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
}

fn concurrent_load() -> Value {
    let loadavg = cmd_stdout(&["sysctl", "-n", "vm.loadavg"])
        .trim()
        .to_string();
    let uptime = cmd_stdout(&["uptime"]).trim().to_string();
    let ps = cmd_stdout(&["ps", "-axo", "pid=,pcpu=,rss=,comm="]);
    let mut rows: Vec<(f64, i64, i32, String)> = Vec::new();
    for line in ps.lines() {
        let mut it = line.split_whitespace();
        let pid = it.next().and_then(|s| s.parse::<i32>().ok());
        let cpu = it.next().and_then(|s| s.parse::<f64>().ok());
        let rss = it.next().and_then(|s| s.parse::<i64>().ok());
        let comm = it.collect::<Vec<_>>().join(" ");
        if let (Some(pid), Some(cpu), Some(rss)) = (pid, cpu, rss) {
            if cpu >= 3.0 {
                rows.push((cpu, rss, pid, comm));
            }
        }
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(12);
    json!({
        "loadavg": loadavg,
        "uptime": uptime,
        "note": "absolute GB/s is measured-under-load; verdict uses back-to-back ratios",
        "top_cpu": rows.iter().map(|(cpu, rss, pid, comm)| json!({
            "pid": pid,
            "cpu_pct": cpu,
            "rss_kib": rss,
            "comm": comm,
        })).collect::<Vec<_>>(),
    })
}

/// Registered before any GPU timestamp is taken. The sidecar judges the curve
/// against these rules; it does not invent a rule after seeing the numbers.
fn pre_registered_interpretation() -> Value {
    json!({
        "registered_before_measurement": true,
        "bytes_per_thread_iteration_held": 38,
        "arithmetic": "stripped (XOR/add sink); this is streaming, not ALU",
        "rules": [
            "GB/s rises monotonically as streams are merged (4 -> 3 -> 2 -> 1) and 2+2+2 differs from 4+2 -> STREAM_COUNT_BOUND. The MLP's 497 is a packing property. An interleaved affine2 layout (6 B operand record) is the fix; cost is a one-time catalog bake that expands 20 B/group to 48 B tight or 64 B with 8 B records.",
            "2+2+2 and 4+2 measure the same -> stream count is NOT the mechanism. Then the cheapest remaining candidate that ran in this process (alignment 2/4/16, then per-thread code stride) decides ALIGNMENT_BOUND vs NOT_STREAM_COUNT.",
            "2+2+2 differs from 4+2 but 2+4+32 tracks 2+2+2 (the 4-byte operand is the lift, not the count) -> ALIGNMENT_BOUND.",
            "neither curve -> MIXED. Do not force a verdict."
        ],
        "same_ratio_bar": 1.08,
        "monotone_slack": 0.97,
        "align_lift_bar": 1.12,
    })
}

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) -> Value {
    fail("stream_count_probe is Metal-only")
}

#[cfg(target_os = "macos")]
fn run(args: Args) -> Value {
    macos::run(args)
}

fn main() {
    let args = parse_args();
    let out = args.out.clone();
    let doc = run(args);
    let text = serde_json::to_string_pretty(&doc).unwrap_or_else(|e| fail(e));
    println!("{text}");
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).unwrap_or_else(|e| fail(e));
            }
        }
        fs::write(&path, format!("{text}\n")).unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", path.display());
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use hawking_core::model::qwen38_geometry::qwen38_layer_name;
    use hawking_core::model::qwen38_hybrid_decode::{
        QWEN38_MIXED_CATALOG_MAGIC, QWEN38_MIXED_CATALOG_NAME, QWEN38_MIXED_CATALOG_VERSION,
        QWEN38_MIXED_RECORD_SIZE,
    };
    use hawking_core::model::qwen_complete_binary::{parse_affine_container, MAGIC_AFFINE};
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    const SHADER: &str = include_str!("stream_count_probe.metal");
    const TG: u64 = 128;
    const GPU_CORES: u32 = 60;
    const BYTES_PER_ITER: u64 = 38;

    struct CatalogRow {
        name: String,
        codec: u8,
        shape: Vec<usize>,
        segment_path: PathBuf,
        offset: u64,
        nbytes: u64,
    }

    fn read_u16(raw: &[u8], off: usize) -> Result<u16, String> {
        let s = raw.get(off..off + 2).ok_or("catalog truncated at u16")?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn read_u32(raw: &[u8], off: usize) -> Result<u32, String> {
        let s = raw.get(off..off + 4).ok_or("catalog truncated at u32")?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn read_u64(raw: &[u8], off: usize) -> Result<u64, String> {
        let s = raw.get(off..off + 8).ok_or("catalog truncated at u64")?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    fn parse_catalog(root: &Path) -> Result<HashMap<String, CatalogRow>, String> {
        let path = root.join(QWEN38_MIXED_CATALOG_NAME);
        let raw = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if raw.len() < 32 || raw[..8] != QWEN38_MIXED_CATALOG_MAGIC {
            return Err("catalog magic is not HQ38M20".into());
        }
        let version = read_u32(&raw, 8)?;
        if version != QWEN38_MIXED_CATALOG_VERSION {
            return Err(format!("unsupported catalog version {version}"));
        }
        let n_tensors = read_u32(&raw, 12)? as usize;
        let n_segments = read_u32(&raw, 16)? as usize;
        let name_blob_bytes = read_u32(&raw, 24)? as usize;
        let mut cursor = 32usize;
        let mut by_id: HashMap<u16, PathBuf> = HashMap::new();
        for _ in 0..n_segments {
            let id = read_u16(&raw, cursor)?;
            let name_len = read_u16(&raw, cursor + 2)? as usize;
            cursor += 44;
            let filename = raw
                .get(cursor..cursor + name_len)
                .ok_or("segment name truncated")?;
            let filename = std::str::from_utf8(filename)
                .map_err(|_| "segment name is not utf-8")?
                .to_owned();
            cursor += name_len;
            let p = Path::new(&filename);
            let resolved = if p.is_absolute() {
                p.to_path_buf()
            } else {
                root.join("segments").join(&filename)
            };
            by_id.insert(id, resolved);
        }
        let table_bytes = n_tensors
            .checked_mul(QWEN38_MIXED_RECORD_SIZE)
            .ok_or("catalog table size overflow")?;
        let table = raw
            .get(cursor..cursor + table_bytes)
            .ok_or("catalog tensor table truncated")?;
        cursor += table_bytes;
        let name_blob = raw
            .get(cursor..cursor + name_blob_bytes)
            .ok_or("catalog name blob truncated")?;
        let mut out = HashMap::new();
        for index in 0..n_tensors {
            let rec = &table[index * QWEN38_MIXED_RECORD_SIZE..(index + 1) * QWEN38_MIXED_RECORD_SIZE];
            let name_off = read_u32(rec, 0)? as usize;
            let name_len = read_u16(rec, 4)? as usize;
            let codec = rec[6];
            let ndim = rec[8] as usize;
            if ndim > 4 {
                return Err("catalog ndim exceeds 4".into());
            }
            let mut shape = Vec::with_capacity(ndim);
            for dim in 0..ndim {
                shape.push(read_u32(rec, 12 + dim * 4)? as usize);
            }
            let segment_id = read_u16(rec, 36)?;
            let offset = read_u64(rec, 40)?;
            let nbytes = read_u64(rec, 48)?;
            let name = name_blob
                .get(name_off..name_off + name_len)
                .ok_or("tensor name out of blob")?;
            let name = std::str::from_utf8(name)
                .map_err(|_| "tensor name is not utf-8")?
                .to_owned();
            let segment_path = by_id
                .get(&segment_id)
                .cloned()
                .ok_or_else(|| format!("unknown segment_id {segment_id}"))?;
            out.insert(
                name.clone(),
                CatalogRow {
                    name,
                    codec,
                    shape,
                    segment_path,
                    offset,
                    nbytes,
                },
            );
        }
        Ok(out)
    }

    fn read_payload(row: &CatalogRow) -> Result<Vec<u8>, String> {
        let mut file = fs::File::open(&row.segment_path)
            .map_err(|e| format!("open {}: {e}", row.segment_path.display()))?;
        file.seek(SeekFrom::Start(row.offset))
            .map_err(|e| format!("seek {}: {e}", row.name))?;
        let n = usize::try_from(row.nbytes).map_err(|_| "payload exceeds usize")?;
        let mut payload = vec![0u8; n];
        file.read_exact(&mut payload)
            .map_err(|e| format!("read {}: {e}", row.name))?;
        Ok(payload)
    }

    fn as_bytes_u8(v: &[u8]) -> &[u8] {
        v
    }
    fn as_bytes_u16(v: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
    }
    fn as_bytes_u32(v: &[u32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }
    fn as_bytes_f32(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }
    fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        enc.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
    fn gpu_ns(cmd: &metal::CommandBufferRef) -> Option<u64> {
        let start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        if end > start && start > 0.0 {
            Some(((end - start) * 1e9) as u64)
        } else {
            None
        }
    }
    fn median_u64(mut v: Vec<u64>) -> Option<u64> {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }
    fn fill_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i % 17) as f32 * 0.125 - 1.0).collect()
    }
    fn buf_bytes(device: &Device, bytes: &[u8]) -> Buffer {
        device.new_buffer_with_data(
            as_bytes_u8(bytes).as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    fn buf_u16(device: &Device, v: &[u16]) -> Buffer {
        device.new_buffer_with_data(
            as_bytes_u16(v).as_ptr() as *const _,
            as_bytes_u16(v).len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    fn buf_u32(device: &Device, v: &[u32]) -> Buffer {
        device.new_buffer_with_data(
            as_bytes_u32(v).as_ptr() as *const _,
            as_bytes_u32(v).len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    fn buf_f32(device: &Device, v: &[f32]) -> Buffer {
        device.new_buffer_with_data(
            as_bytes_f32(v).as_ptr() as *const _,
            as_bytes_f32(v).len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    struct Pipe {
        name: &'static str,
        state: ComputePipelineState,
        max_threads: u64,
        exec_width: u64,
    }

    const KERNELS: &[&str] = &[
        "stream_count_mlp_2_2_2_32",
        "stream_count_dn_4_2_32",
        "stream_count_mid_2_4_32",
        "stream_count_pack_6_32",
        "stream_count_pack_38",
        "stream_count_align_4",
        "stream_count_align_16",
        "stream_count_stride_contig",
        "stream_count_zero",
    ];

    fn compile(device: &Device) -> Result<HashMap<&'static str, Pipe>, String> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("shader compile: {e}"))?;
        eprintln!(
            "stream_count_probe: compiled shaders in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        let mut out = HashMap::new();
        for name in KERNELS {
            let f = lib
                .get_function(name, None)
                .map_err(|e| format!("{name}: {e}"))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))?;
            out.insert(
                *name,
                Pipe {
                    name,
                    max_threads: p.max_total_threads_per_threadgroup(),
                    exec_width: p.thread_execution_width(),
                    state: p,
                },
            );
        }
        Ok(out)
    }

    fn occupancy(pipe: &Pipe, rows: u32) -> Value {
        let threadgroups = u64::from(rows.div_ceil(2).max(1));
        json!({
            "threads_per_threadgroup": TG,
            "max_total_threads_per_threadgroup": pipe.max_threads,
            "thread_execution_width": pipe.exec_width,
            "occupancy_of_max_threads": if pipe.max_threads == 0 {
                Value::Null
            } else {
                json!(TG as f64 / pipe.max_threads as f64)
            },
            "threadgroups": threadgroups,
            "gpu_cores": GPU_CORES,
            "threadgroups_per_core": threadgroups as f64 / f64::from(GPU_CORES),
            "registers_per_thread": Value::Null,
            "registers_note": "Metal pipeline state does not report register count on this toolchain; xcrun metal is not on PATH",
        })
    }

    struct AffineProj {
        name: String,
        rows: u32,
        cols: u32,
        group_size: u32,
        codes_host: Vec<u8>,
        scales_host: Vec<u16>,
        biases_host: Vec<u16>,
        input_host: Vec<f32>,
        codes: Buffer,
        scales: Buffer,
        biases: Buffer,
        input: Buffer,
        output: Buffer,
        weight_bytes: u64,
        code_bytes: u64,
        scale_bytes: u64,
        bias_bytes: u64,
    }

    fn load_affine(
        device: &Device,
        catalog: &HashMap<String, CatalogRow>,
        name: &str,
    ) -> Result<AffineProj, String> {
        let row = catalog
            .get(name)
            .ok_or_else(|| format!("missing catalog tensor {name}"))?;
        let payload = read_payload(row)?;
        if payload.len() < 8 || payload[..8] != MAGIC_AFFINE {
            return Err(format!(
                "{name} codec={} magic {:?} is not HGRAVF01",
                row.codec,
                payload.get(..8)
            ));
        }
        let packed = parse_affine_container(&payload).map_err(|e| e.to_string())?;
        if packed.is_q2f() {
            return Err(format!("{name} is q2f (no bias); probe needs affine2"));
        }
        let input = fill_f32(packed.cols);
        let output = vec![0f32; packed.rows];
        let code_bytes = packed.codes.len() as u64;
        let scale_bytes = (packed.scales_f16.len() * 2) as u64;
        let bias_bytes = (packed.biases_f16.len() * 2) as u64;
        Ok(AffineProj {
            name: name.to_string(),
            rows: packed.rows as u32,
            cols: packed.cols as u32,
            group_size: packed.group_size as u32,
            codes: buf_bytes(device, &packed.codes),
            scales: buf_u16(device, &packed.scales_f16),
            biases: buf_u16(device, &packed.biases_f16),
            input: buf_f32(device, &input),
            output: buf_f32(device, &output),
            weight_bytes: code_bytes + scale_bytes + bias_bytes,
            code_bytes,
            scale_bytes,
            bias_bytes,
            codes_host: packed.codes,
            scales_host: packed.scales_f16,
            biases_host: packed.biases_f16,
            input_host: input,
        })
    }

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(TG, 1, 1));
    }

    fn encode_mlp(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        proj: &AffineProj,
        work_cols: u32,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        set_u32(enc, 7, work_cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_zero(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, proj: &AffineProj) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_dn(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        packed4: &Buffer,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(packed4), 0);
        enc.set_buffer(1, Some(&proj.biases), 0);
        enc.set_buffer(2, Some(&proj.input), 0);
        enc.set_buffer(3, Some(&proj.output), 0);
        set_u32(enc, 4, proj.rows);
        set_u32(enc, 5, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_mid(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        packed_sb: &Buffer,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(packed_sb), 0);
        enc.set_buffer(2, Some(&proj.input), 0);
        enc.set_buffer(3, Some(&proj.output), 0);
        set_u32(enc, 4, proj.rows);
        set_u32(enc, 5, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_pack6(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        packed6: &Buffer,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(packed6), 0);
        enc.set_buffer(1, Some(&proj.input), 0);
        enc.set_buffer(2, Some(&proj.output), 0);
        set_u32(enc, 3, proj.rows);
        set_u32(enc, 4, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_pack38(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        packed38: &Buffer,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(packed38), 0);
        enc.set_buffer(1, Some(&proj.output), 0);
        set_u32(enc, 2, proj.rows);
        set_u32(enc, 3, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_align(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        codes: &Buffer,
        scales: &Buffer,
        biases: &Buffer,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(codes), 0);
        enc.set_buffer(1, Some(scales), 0);
        enc.set_buffer(2, Some(biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_contig(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        codes_c: &Buffer,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(codes_c), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        dispatch_groups(enc, proj.rows);
    }

    fn time_cb<F>(queue: &metal::CommandQueue, n: usize, mut fill: F) -> Vec<u64>
    where
        F: FnMut(&metal::ComputeCommandEncoderRef),
    {
        let mut gpu = Vec::new();
        for _ in 0..n {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            fill(&enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if let Some(ns) = gpu_ns(cmd) {
                gpu.push(ns);
            }
        }
        gpu
    }

    fn time_all<F>(queue: &metal::CommandQueue, n_proj: usize, n: usize, mut fill: F) -> Vec<u64>
    where
        F: FnMut(&metal::ComputeCommandEncoderRef, usize),
    {
        time_cb(queue, n, |enc| {
            for i in 0..n_proj {
                fill(enc, i);
            }
        })
    }

    fn rung_json(
        id: &str,
        kernel: &str,
        stream_count: u32,
        bytes_per_stream: &[u64],
        weight_bytes: u64,
        gpu: Vec<u64>,
        occupancy: &Value,
        access: Value,
        family: &str,
    ) -> Value {
        let counted: u64 = bytes_per_stream.iter().sum();
        if counted != BYTES_PER_ITER {
            fail(format!(
                "{id}: bytes_per_stream {bytes_per_stream:?} sum to {counted}, not {BYTES_PER_ITER}"
            ));
        }
        let med = median_u64(gpu.clone()).unwrap_or(0);
        let gb_s = if med == 0 {
            0.0
        } else {
            weight_bytes as f64 / med as f64
        };
        let us = med as f64 / 1e3;
        json!({
            "id": id,
            "kernel": kernel,
            "family": family,
            "stream_count": stream_count,
            "bytes_per_stream": bytes_per_stream,
            "bytes_per_thread_iteration": BYTES_PER_ITER,
            "weight_bytes": weight_bytes,
            "gpu_ns_median": med,
            "gpu_ns_reps": gpu,
            "gpu_us_median": us,
            "effective_gb_s": gb_s,
            "dispatches": 3,
            "encoders": 1,
            "command_buffers": 1,
            "threads_per_threadgroup": TG,
            "occupancy": occupancy,
            "access_pattern": access,
        })
    }

    fn groups_of(p: &AffineProj) -> usize {
        (p.rows as usize) * ((p.cols as usize) / 64)
    }

    fn build_packed4(p: &AffineProj) -> Vec<u32> {
        let groups = groups_of(p);
        let mut out = vec![0u32; groups * 8];
        for rgb in 0..groups {
            let scale = p.scales_host[rgb] as u32;
            let base = rgb * 16;
            for slot in 0..8 {
                let code = u16::from_le_bytes([p.codes_host[base + slot * 2], p.codes_host[base + slot * 2 + 1]])
                    as u32;
                out[rgb * 8 + slot] = code | (scale << 16);
            }
        }
        out
    }

    fn build_packed_sb(p: &AffineProj) -> Vec<u32> {
        let groups = groups_of(p);
        let mut out = vec![0u32; groups];
        for rgb in 0..groups {
            out[rgb] = p.scales_host[rgb] as u32 | ((p.biases_host[rgb] as u32) << 16);
        }
        out
    }

    fn build_packed6(p: &AffineProj) -> Vec<u8> {
        let groups = groups_of(p);
        let mut out = vec![0u8; groups * 8 * 8];
        for rgb in 0..groups {
            let scale = p.scales_host[rgb];
            let bias = p.biases_host[rgb];
            let base = rgb * 16;
            for slot in 0..8 {
                let rec = rgb * 8 + slot;
                let off = rec * 8;
                let code = u16::from_le_bytes([p.codes_host[base + slot * 2], p.codes_host[base + slot * 2 + 1]]);
                out[off..off + 4].copy_from_slice(&(code as u32 | ((scale as u32) << 16)).to_le_bytes());
                out[off + 4..off + 6].copy_from_slice(&bias.to_le_bytes());
            }
        }
        out
    }

    fn build_packed38(p: &AffineProj) -> Vec<u8> {
        let rows = p.rows as usize;
        let cols = p.cols as usize;
        let gpr = cols / 64;
        let trips = cols / 512;
        let nslots = rows * 64 * trips;
        let mut out = vec![0u8; nslots * 40];
        for row in 0..rows {
            for lane in 0..64 {
                let mut col = lane * 8;
                let mut trip = 0usize;
                while col + 8 <= cols {
                    let group = col / 64;
                    let local = col % 64;
                    let rgb = row * gpr + group;
                    let code_off = rgb * 16 + (local / 4);
                    let code = u16::from_le_bytes([p.codes_host[code_off], p.codes_host[code_off + 1]]);
                    let slot = row * 64 * trips + trip * 64 + lane;
                    let rec = slot * 40;
                    out[rec..rec + 4].copy_from_slice(
                        &(code as u32 | ((p.scales_host[rgb] as u32) << 16)).to_le_bytes(),
                    );
                    out[rec + 4..rec + 6].copy_from_slice(&p.biases_host[rgb].to_le_bytes());
                    for i in 0..8 {
                        out[rec + 8 + i * 4..rec + 12 + i * 4]
                            .copy_from_slice(&p.input_host[col + i].to_le_bytes());
                    }
                    col += 512;
                    trip += 1;
                }
            }
        }
        out
    }

    fn build_align(p: &AffineProj, slot: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let groups = groups_of(p);
        let mut codes = vec![0u8; groups * 8 * slot];
        let mut scales = vec![0u8; groups * slot];
        let mut biases = vec![0u8; groups * slot];
        for rgb in 0..groups {
            let base = rgb * 16;
            for i in 0..8 {
                let dst = rgb * 8 * slot + i * slot;
                codes[dst] = p.codes_host[base + i * 2];
                codes[dst + 1] = p.codes_host[base + i * 2 + 1];
            }
            let s = p.scales_host[rgb].to_le_bytes();
            let b = p.biases_host[rgb].to_le_bytes();
            scales[rgb * slot] = s[0];
            scales[rgb * slot + 1] = s[1];
            biases[rgb * slot] = b[0];
            biases[rgb * slot + 1] = b[1];
        }
        (codes, scales, biases)
    }

    fn build_contig_codes(p: &AffineProj) -> Vec<u16> {
        let rows = p.rows as usize;
        let cols = p.cols as usize;
        let gpr = cols / 64;
        let trips = cols / 512;
        let mut out = vec![0u16; rows * 64 * trips];
        for row in 0..rows {
            for lane in 0..64 {
                let mut col = lane * 8;
                let mut trip = 0usize;
                while col + 8 <= cols {
                    let group = col / 64;
                    let local = col % 64;
                    let rgb = row * gpr + group;
                    let code_off = rgb * 16 + (local / 4);
                    out[row * 64 * trips + lane * trips + trip] =
                        u16::from_le_bytes([p.codes_host[code_off], p.codes_host[code_off + 1]]);
                    col += 512;
                    trip += 1;
                }
            }
        }
        out
    }

    fn access(
        alignment: u32,
        col_stride: u32,
        contig_codes: bool,
        x_reused: bool,
        scale_broadcast: bool,
    ) -> Value {
        json!({
            "operand_alignment_bytes": alignment,
            "x_alignment_bytes": 4,
            "per_thread_col_stride": col_stride,
            "codes_contiguous_per_thread": contig_codes,
            "scale_bias_group_broadcast": scale_broadcast,
            "activation_reused_across_rows": x_reused,
            "coalesced_across_simd_codes": !contig_codes,
        })
    }

    pub fn run(args: Args) -> Value {
        let interpretation = pre_registered_interpretation();
        let concurrent_start = concurrent_load();
        eprintln!(
            "stream_count_probe opening {} layer={} warmup={} reps={}",
            args.artifact_root.display(),
            args.layer,
            args.warmup,
            args.reps
        );
        eprintln!(
            "  interpretation registered before measurement: {}",
            interpretation["registered_before_measurement"]
        );
        let catalog = parse_catalog(&args.artifact_root).unwrap_or_else(|e| fail(e));
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal-capable GPU"));
        let queue = device.new_command_queue();
        let pipes = compile(&device).unwrap_or_else(|e| fail(e));

        let gate_name = qwen38_layer_name(args.layer, "mlp.gate_proj.weight");
        let up_name = qwen38_layer_name(args.layer, "mlp.up_proj.weight");
        let down_name = qwen38_layer_name(args.layer, "mlp.down_proj.weight");
        eprintln!("  loading {gate_name}");
        let gate = load_affine(&device, &catalog, &gate_name).unwrap_or_else(|e| fail(e));
        eprintln!("  loading {up_name}");
        let up = load_affine(&device, &catalog, &up_name).unwrap_or_else(|e| fail(e));
        eprintln!("  loading {down_name}");
        let down = load_affine(&device, &catalog, &down_name).unwrap_or_else(|e| fail(e));
        if gate.group_size != 64 || up.group_size != 64 || down.group_size != 64 {
            fail(format!(
                "probe needs group 64; got gate={} up={} down={}",
                gate.group_size, up.group_size, down.group_size
            ));
        }
        let mlp = [gate, up, down];
        let mlp_bytes: u64 = mlp.iter().map(|p| p.weight_bytes).sum();
        let n_proj = mlp.len();
        let occ = occupancy(pipes.get("stream_count_mlp_2_2_2_32").unwrap(), mlp[0].rows);

        let mut rungs: Vec<Value> = Vec::new();
        let mut alignment: Vec<Value> = Vec::new();

        let k_mlp = pipes.get("stream_count_mlp_2_2_2_32").unwrap();
        let k_dn = pipes.get("stream_count_dn_4_2_32").unwrap();
        let k_mid = pipes.get("stream_count_mid_2_4_32").unwrap();
        let k_p6 = pipes.get("stream_count_pack_6_32").unwrap();
        let k_p38 = pipes.get("stream_count_pack_38").unwrap();
        let k_a4 = pipes.get("stream_count_align_4").unwrap();
        let k_a16 = pipes.get("stream_count_align_16").unwrap();
        let k_sc = pipes.get("stream_count_stride_contig").unwrap();
        let k_zero = pipes.get("stream_count_zero").unwrap();

        eprintln!("  mlp_2_2_2_32 (4 streams, production addressing)");
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| encode_mlp(enc, k_mlp, &mlp[i], 0));
        let mlp_ns = time_all(&queue, n_proj, args.reps, |enc, i| encode_mlp(enc, k_mlp, &mlp[i], 0));
        rungs.push(rung_json(
            "mlp_2_2_2_32",
            k_mlp.name,
            4,
            &[2, 2, 2, 32],
            mlp_bytes,
            mlp_ns.clone(),
            &occ,
            access(2, 512, false, true, true),
            "stream_ladder",
        ));
        alignment.push(rung_json(
            "align_2",
            k_mlp.name,
            4,
            &[2, 2, 2, 32],
            mlp_bytes,
            mlp_ns.clone(),
            &occ,
            access(2, 512, false, true, true),
            "alignment",
        ));

        eprintln!("  zero_load");
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| encode_zero(enc, k_zero, &mlp[i]));
        let zero_ns = time_all(&queue, n_proj, args.reps, |enc, i| encode_zero(enc, k_zero, &mlp[i]));
        let zero = rung_json(
            "zero_load",
            k_zero.name,
            0,
            &[0, 0, 0, 38],
            mlp_bytes,
            zero_ns,
            &occ,
            json!({"role": "launch+reduction floor; no weight/x loads", "bytes_per_thread_iteration_note": "numerator is the organ's unique payload so GB/s is huge by construction"}),
            "dce",
        );
        // zero uses a dummy 38-split so the rung helper accepts it; overwrite counted identity.
        // The sidecar treats zero_load separately and does not demand it as a ladder rung.

        eprintln!("  mlp_2_2_2_32 half-K (DCE / loads-survived)");
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            let half = mlp[i].cols / 2;
            encode_mlp(enc, k_mlp, &mlp[i], half);
        });
        let half_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            let half = mlp[i].cols / 2;
            encode_mlp(enc, k_mlp, &mlp[i], half);
        });
        let half = rung_json(
            "mlp_2_2_2_32_halfk",
            k_mlp.name,
            4,
            &[2, 2, 2, 32],
            mlp_bytes / 2,
            half_ns,
            &occ,
            access(2, 512, false, true, true),
            "dce",
        );

        eprintln!("  packing dn_4_2_32");
        let packed4: Vec<Buffer> = mlp.iter().map(|p| buf_u32(&device, &build_packed4(p))).collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| encode_dn(enc, k_dn, &packed4[i], &mlp[i]));
        let dn_ns = time_all(&queue, n_proj, args.reps, |enc, i| encode_dn(enc, k_dn, &packed4[i], &mlp[i]));
        rungs.push(rung_json(
            "dn_4_2_32",
            k_dn.name,
            3,
            &[4, 2, 32],
            mlp_bytes,
            dn_ns,
            &occ,
            access(4, 512, false, true, true),
            "stream_ladder",
        ));
        drop(packed4);

        eprintln!("  packing mid_2_4_32");
        let packed_sb: Vec<Buffer> = mlp
            .iter()
            .map(|p| buf_u32(&device, &build_packed_sb(p)))
            .collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_mid(enc, k_mid, &packed_sb[i], &mlp[i])
        });
        let mid_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_mid(enc, k_mid, &packed_sb[i], &mlp[i])
        });
        rungs.push(rung_json(
            "mid_2_4_32",
            k_mid.name,
            3,
            &[2, 4, 32],
            mlp_bytes,
            mid_ns,
            &occ,
            access(2, 512, false, true, true),
            "stream_ladder",
        ));
        drop(packed_sb);

        eprintln!("  packing pack_6_32");
        let packed6: Vec<Buffer> = mlp
            .iter()
            .map(|p| buf_bytes(&device, &build_packed6(p)))
            .collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_pack6(enc, k_p6, &packed6[i], &mlp[i])
        });
        let p6_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_pack6(enc, k_p6, &packed6[i], &mlp[i])
        });
        rungs.push(rung_json(
            "pack_6_32",
            k_p6.name,
            2,
            &[6, 32],
            mlp_bytes,
            p6_ns,
            &occ,
            json!({
                "operand_alignment_bytes": 4,
                "x_alignment_bytes": 4,
                "per_thread_col_stride": 512,
                "codes_contiguous_per_thread": false,
                "scale_bias_group_broadcast": false,
                "scale_bias_replicated_per_tile": true,
                "activation_reused_across_rows": true,
                "storage_bytes_per_group": 64,
                "payload_bytes_per_group": 48,
                "production_bytes_per_group": 20,
            }),
            "stream_ladder",
        ));
        drop(packed6);

        eprintln!("  packing pack_38 (copies x into the record; no row reuse)");
        let packed38: Vec<Buffer> = mlp
            .iter()
            .map(|p| {
                eprintln!("    {} {} rows", p.name, p.rows);
                buf_bytes(&device, &build_packed38(p))
            })
            .collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_pack38(enc, k_p38, &packed38[i], &mlp[i])
        });
        let p38_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_pack38(enc, k_p38, &packed38[i], &mlp[i])
        });
        rungs.push(rung_json(
            "pack_38",
            k_p38.name,
            1,
            &[38],
            mlp_bytes,
            p38_ns,
            &occ,
            json!({
                "operand_alignment_bytes": 4,
                "x_alignment_bytes": 4,
                "per_thread_col_stride": 512,
                "codes_contiguous_per_thread": false,
                "scale_bias_group_broadcast": false,
                "activation_reused_across_rows": false,
                "record_bytes": 40,
                "payload_bytes": 38,
                "note": "x is copied into each row's records; unique DRAM traffic grows, counted payload stays 38",
            }),
            "stream_ladder",
        ));
        drop(packed38);

        eprintln!("  align_4");
        let a4: Vec<(Buffer, Buffer, Buffer)> = mlp
            .iter()
            .map(|p| {
                let (c, s, b) = build_align(p, 4);
                (buf_bytes(&device, &c), buf_bytes(&device, &s), buf_bytes(&device, &b))
            })
            .collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_align(enc, k_a4, &a4[i].0, &a4[i].1, &a4[i].2, &mlp[i])
        });
        let a4_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_align(enc, k_a4, &a4[i].0, &a4[i].1, &a4[i].2, &mlp[i])
        });
        alignment.push(rung_json(
            "align_4",
            k_a4.name,
            4,
            &[2, 2, 2, 32],
            mlp_bytes,
            a4_ns,
            &occ,
            access(4, 512, false, true, true),
            "alignment",
        ));
        drop(a4);

        eprintln!("  align_16");
        let a16: Vec<(Buffer, Buffer, Buffer)> = mlp
            .iter()
            .map(|p| {
                let (c, s, b) = build_align(p, 16);
                (buf_bytes(&device, &c), buf_bytes(&device, &s), buf_bytes(&device, &b))
            })
            .collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_align(enc, k_a16, &a16[i].0, &a16[i].1, &a16[i].2, &mlp[i])
        });
        let a16_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_align(enc, k_a16, &a16[i].0, &a16[i].1, &a16[i].2, &mlp[i])
        });
        alignment.push(rung_json(
            "align_16",
            k_a16.name,
            4,
            &[2, 2, 2, 32],
            mlp_bytes,
            a16_ns,
            &occ,
            access(16, 512, false, true, true),
            "alignment",
        ));
        drop(a16);

        eprintln!("  stride_contig (codes only; scale/bias stay group-broadcast)");
        let contig: Vec<Buffer> = mlp
            .iter()
            .map(|p| buf_u16(&device, &build_contig_codes(p)))
            .collect();
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_contig(enc, k_sc, &contig[i], &mlp[i])
        });
        let sc_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_contig(enc, k_sc, &contig[i], &mlp[i])
        });
        alignment.push(rung_json(
            "stride_contig",
            k_sc.name,
            4,
            &[2, 2, 2, 32],
            mlp_bytes,
            sc_ns,
            &occ,
            access(2, 1, true, true, true),
            "stride",
        ));
        drop(contig);

        let concurrent_end = concurrent_load();
        let trips_gate = mlp[0].cols / 512;
        let trips_down = mlp[2].cols / 512;

        json!({
            "schema": "hawking.future.mlp_stream_count.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "absolute_gb_s_are_measured_under_load": true,
            "bytes_per_thread_iteration_held": BYTES_PER_ITER,
            "pre_registered_interpretation": interpretation,
            "concurrent_load_start": concurrent_start,
            "concurrent_load": concurrent_end,
            "organ": "mlp",
            "codec": "HGRAVF01 affine2 q2 group64",
            "geometry": "geo_tpr64_tg128",
            "projections": mlp.iter().map(|p| json!({
                "name": p.name,
                "rows": p.rows,
                "cols": p.cols,
                "group_size": p.group_size,
                "weight_bytes": p.weight_bytes,
                "code_bytes": p.code_bytes,
                "scale_bytes": p.scale_bytes,
                "bias_bytes": p.bias_bytes,
            })).collect::<Vec<_>>(),
            "weight_bytes": mlp_bytes,
            "inner_loop_trips_gate_up": trips_gate,
            "inner_loop_trips_down": trips_down,
            "dispatches": 3,
            "rungs": rungs,
            "alignment": alignment,
            "zero_load": zero,
            "halfk": half,
        })
    }
}
