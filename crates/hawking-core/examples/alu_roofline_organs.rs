//! Matched-pair ALU vs memory-system probe for one MLP layer and DeltaNet's
//! dominant Q4 kernel, plus `--mode deltanet-decompose`: per-kernel GPU
//! times of the DeltaNet organ as executed, summing to the organ total.
//! Does not mutate the production decode path.
//!
//! ARM A: same bytes and access pattern, decode+dequant+FMA stripped.
//! ARM B: same per-code arithmetic, first half of K only.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example alu_roofline_organs
//! ./tools/gpu_lane_lock.sh u1alu \
//!   workspace/ops/build/rust/release-fast/examples/alu_roofline_organs \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_MLP_ALU_ROOFLINE_raw.json
//! ./tools/gpu_lane_lock.sh w2dnresid \
//!   workspace/ops/build/rust/release-fast/examples/alu_roofline_organs \
//!   --mode deltanet-decompose \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_DELTANET_ORGAN_DECOMPOSE_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: alu_roofline_organs --artifact-root DIR \
        [--mode alu|deltanet-decompose] [--layer N] [--warmup N] [--reps N] \
        [--session-warmup N] [--session-reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("alu_roofline_organs: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    mode: String,
    layer: usize,
    warmup: usize,
    reps: usize,
    session_warmup: usize,
    session_reps: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut mode = "alu".to_string();
    let mut layer = 0usize;
    let mut warmup = 5usize;
    let mut reps = 11usize;
    let mut session_warmup = 2usize;
    let mut session_reps = 7usize;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--mode" => {
                mode = args.next().unwrap_or_else(|| fail(usage()));
                if mode != "alu" && mode != "deltanet-decompose" {
                    fail("--mode must be alu or deltanet-decompose");
                }
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
            "--session-warmup" => {
                session_warmup = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--session-warmup"));
            }
            "--session-reps" => {
                session_reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--session-reps"));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if reps == 0 || session_reps == 0 {
        fail("--reps and --session-reps must be > 0");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        mode,
        layer,
        warmup,
        reps,
        session_warmup,
        session_reps,
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

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) -> Value {
    fail("alu_roofline_organs is Metal-only")
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
    use hawking_core::metal::CommandBufferTiming;
    use hawking_core::model::qwen38_hybrid_decode::{
        Qwen38DeltaNetStateKernel, Qwen38HybridDecodeSession, Qwen38MlpFusion,
        QWEN38_MIXED_CATALOG_MAGIC, QWEN38_MIXED_CATALOG_NAME, QWEN38_MIXED_CATALOG_VERSION,
        QWEN38_MIXED_RECORD_SIZE,
    };
    use hawking_core::model::qwen_complete_binary::{
        parse_affine_container, parse_uniform_q4_header, MAGIC_AFFINE, UNIFORM_Q4_MAGIC,
    };
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    const SHADER: &str = include_str!("alu_roofline_organs.metal");
    const TG: u64 = 128;
    const GPU_CORES: u32 = 60;

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
    fn buf_u8(device: &Device, bytes: &[u8]) -> Buffer {
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

    fn compile(device: &Device) -> Result<HashMap<&'static str, Pipe>, String> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("shader compile: {e}"))?;
        eprintln!(
            "alu_roofline_organs: compiled shaders in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        let names = [
            "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            "alu_roofline_affine_q2_geo_tpr64_tg128_stripped",
            "alu_roofline_affine_q2_geo_tpr64_tg128_halfk",
            "alu_roofline_affine_q2_geo_tpr64_tg128_zero",
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "alu_roofline_q4_geo_tpr64_tg128_stripped",
            "alu_roofline_q4_geo_tpr64_tg128_halfk",
            "alu_roofline_q4_geo_tpr64_tg128_zero",
        ];
        let mut out = HashMap::new();
        for name in names {
            let f = lib
                .get_function(name, None)
                .map_err(|e| format!("{name}: {e}"))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))?;
            out.insert(
                name,
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

    struct Q4Proj {
        name: String,
        rows: u32,
        cols: u32,
        group_size: u32,
        groups_per_row: u32,
        codes: Buffer,
        scales: Buffer,
        input: Buffer,
        output: Buffer,
        weight_bytes: u64,
        code_bytes: u64,
        scale_bytes: u64,
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
            codes: buf_u8(device, &packed.codes),
            scales: buf_u16(device, &packed.scales_f16),
            biases: buf_u16(device, &packed.biases_f16),
            input: buf_f32(device, &input),
            output: buf_f32(device, &output),
            weight_bytes: code_bytes + scale_bytes + bias_bytes,
            code_bytes,
            scale_bytes,
            bias_bytes,
        })
    }

    fn load_q4(
        device: &Device,
        catalog: &HashMap<String, CatalogRow>,
        name: &str,
    ) -> Result<Q4Proj, String> {
        let row = catalog
            .get(name)
            .ok_or_else(|| format!("missing catalog tensor {name}"))?;
        let payload = read_payload(row)?;
        if payload.len() < 8 || payload[..8] != UNIFORM_Q4_MAGIC {
            return Err(format!(
                "{name} codec={} magic {:?} is not HQ30UQ4 (shape {:?})",
                row.codec,
                payload.get(..8),
                row.shape
            ));
        }
        let header = parse_uniform_q4_header(&payload).map_err(|e| e.to_string())?;
        let (rows, cols) = match header.shape.as_slice() {
            [r, c] => (*r as u32, *c as u32),
            other => return Err(format!("{name} Q4 rank {other:?} is not a matrix")),
        };
        let scales = &payload[header.scale_offset..header.sign_offset];
        let codes = &payload[header.sign_offset..header.payload_bytes];
        let input = fill_f32(cols as usize);
        let output = vec![0f32; rows as usize];
        let groups_per_row = cols.div_ceil(header.group_size as u32);
        Ok(Q4Proj {
            name: name.to_string(),
            rows,
            cols,
            group_size: header.group_size as u32,
            groups_per_row,
            codes: buf_u8(device, codes),
            scales: buf_u8(device, scales),
            input: buf_f32(device, &input),
            output: buf_f32(device, &output),
            weight_bytes: (codes.len() + scales.len()) as u64,
            code_bytes: codes.len() as u64,
            scale_bytes: scales.len() as u64,
        })
    }

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(TG, 1, 1));
    }

    enum ArmKind {
        AffineProd,
        AffineStripped { work_cols: u32 },
        AffineHalfk { work_cols: u32 },
        AffineZero,
        Q4Prod,
        Q4Stripped { work_cols: u32 },
        Q4Halfk { work_cols: u32 },
        Q4Zero,
    }

    fn encode_affine(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        proj: &AffineProj,
        kind: &ArmKind,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        set_u32(enc, 7, proj.group_size);
        match kind {
            ArmKind::AffineStripped { work_cols } | ArmKind::AffineHalfk { work_cols } => {
                set_u32(enc, 8, *work_cols);
            }
            _ => {}
        }
        dispatch_groups(enc, proj.rows);
    }

    fn encode_q4(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        proj: &Q4Proj,
        kind: &ArmKind,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.input), 0);
        enc.set_buffer(3, Some(&proj.output), 0);
        set_u32(enc, 4, proj.rows);
        set_u32(enc, 5, proj.cols);
        set_u32(enc, 6, proj.groups_per_row);
        match kind {
            ArmKind::Q4Stripped { work_cols } | ArmKind::Q4Halfk { work_cols } => {
                set_u32(enc, 7, *work_cols);
            }
            _ => {}
        }
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

    fn arm_json(
        label: &str,
        kernel: &str,
        weight_bytes: u64,
        gpu: Vec<u64>,
        dispatches: u64,
        occupancy: &Value,
        extra: Value,
    ) -> Value {
        let med = median_u64(gpu.clone()).unwrap_or(0);
        let gb_s = if med == 0 {
            0.0
        } else {
            weight_bytes as f64 / med as f64
        };
        let mut v = json!({
            "label": label,
            "kernel": kernel,
            "weight_bytes": weight_bytes,
            "gpu_ns_median": med,
            "gpu_ns_reps": gpu,
            "dispatches": dispatches,
            "encoders": 1,
            "command_buffers": 1,
            "effective_gb_s": gb_s,
            "occupancy": occupancy,
        });
        if let Value::Object(extra) = extra {
            if let Value::Object(obj) = &mut v {
                obj.extend(extra);
            }
        }
        v
    }

    fn family_arm(
        label: &str,
        kernel: &str,
        bytes: u64,
        gpu: Vec<u64>,
        dispatches: u64,
        extra: Value,
    ) -> Value {
        arm_json(label, kernel, bytes, gpu, dispatches, &json!({}), extra)
    }

    fn time_session_family(
        session: &Qwen38HybridDecodeSession,
        family: &str,
        warmup: usize,
        reps: usize,
    ) -> Result<(Vec<u64>, u64, u64), String> {
        eprintln!("  family {family} warmup={warmup} reps={reps}");
        for i in 0..warmup {
            let t = session
                .measure_isolated_family(family)
                .map_err(|e| format!("{family} warmup {i}: {e}"))?;
            eprintln!(
                "    warmup{i} gpu={:?} disp={}",
                t.gpu_ns, t.dispatches
            );
        }
        let mut gpu = Vec::with_capacity(reps);
        let mut disp = 0u64;
        let mut encoders = 0u64;
        for i in 0..reps {
            let t = session
                .measure_isolated_family(family)
                .map_err(|e| format!("{family} rep {i}: {e}"))?;
            let g = t.gpu_ns.ok_or_else(|| {
                format!("{family} rep {i}: driver did not expose GPUEndTime-GPUStartTime")
            })?;
            eprintln!("    rep{i} gpu={g} wait={} disp={}", t.wait_ns, t.dispatches);
            gpu.push(g);
            disp = t.dispatches;
            encoders = t.encoder_count;
        }
        Ok((gpu, disp, encoders))
    }

    fn time_session_custom(
        label: &str,
        warmup: usize,
        reps: usize,
        mut once: impl FnMut() -> Result<CommandBufferTiming, String>,
    ) -> Result<(Vec<u64>, u64, u64), String> {
        eprintln!("  {label} warmup={warmup} reps={reps}");
        for i in 0..warmup {
            let t = once()?;
            eprintln!("    warmup{i} gpu={:?} disp={}", t.gpu_ns, t.dispatches);
        }
        let mut gpu = Vec::with_capacity(reps);
        let mut disp = 0u64;
        let mut encoders = 0u64;
        for i in 0..reps {
            let t = once()?;
            let g = t
                .gpu_ns
                .ok_or_else(|| format!("{label} rep {i}: no GPUEndTime-GPUStartTime"))?;
            eprintln!("    rep{i} gpu={g} wait={} disp={}", t.wait_ns, t.dispatches);
            gpu.push(g);
            disp = t.dispatches;
            encoders = t.encoder_count;
        }
        Ok((gpu, disp, encoders))
    }

    fn time_q4_arms(
        queue: &metal::CommandQueue,
        warmup: usize,
        reps: usize,
        prod: &Pipe,
        strip: &Pipe,
        half: &Pipe,
        zero: &Pipe,
        proj: &Q4Proj,
        encode: impl Fn(&metal::ComputeCommandEncoderRef, &Pipe, &ArmKind),
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
        let work_half = proj.cols / 2;
        let time = |n: usize, kind: ArmKind, pipe: &Pipe| {
            time_cb(queue, n, |enc| encode(enc, pipe, &kind))
        };
        let _ = time(warmup, ArmKind::Q4Prod, prod);
        let prod_ns = time(reps, ArmKind::Q4Prod, prod);
        let _ = time(warmup, ArmKind::Q4Stripped { work_cols: 0 }, strip);
        let a_ns = time(reps, ArmKind::Q4Stripped { work_cols: 0 }, strip);
        let _ = time(
            warmup,
            ArmKind::Q4Halfk {
                work_cols: work_half,
            },
            half,
        );
        let b_ns = time(
            reps,
            ArmKind::Q4Halfk {
                work_cols: work_half,
            },
            half,
        );
        let _ = time(warmup, ArmKind::Q4Zero, zero);
        let zero_ns = time(reps, ArmKind::Q4Zero, zero);
        let _ = time(
            warmup,
            ArmKind::Q4Stripped {
                work_cols: work_half,
            },
            strip,
        );
        let a_half_ns = time(
            reps,
            ArmKind::Q4Stripped {
                work_cols: work_half,
            },
            strip,
        );
        (prod_ns, a_ns, b_ns, zero_ns, a_half_ns)
    }

    fn q4_organ_json(
        organ: &str,
        proj: &Q4Proj,
        occ: &Value,
        prod: &Pipe,
        strip: &Pipe,
        half: &Pipe,
        zero: &Pipe,
        prod_ns: Vec<u64>,
        a_ns: Vec<u64>,
        b_ns: Vec<u64>,
        zero_ns: Vec<u64>,
        a_half_ns: Vec<u64>,
    ) -> Value {
        let half_bytes = proj.weight_bytes / 2;
        json!({
            "organ": organ,
            "kernel": prod.name,
            "codec": "HQ30UQ4 group64",
            "projection": {
                "name": proj.name,
                "rows": proj.rows,
                "cols": proj.cols,
                "group_size": proj.group_size,
                "groups_per_row": proj.groups_per_row,
                "weight_bytes": proj.weight_bytes,
                "code_bytes": proj.code_bytes,
                "scale_bytes": proj.scale_bytes,
            },
            "threads_per_threadgroup": TG,
            "production": arm_json("production", prod.name, proj.weight_bytes, prod_ns, 1, occ, json!({"cols": proj.cols})),
            "arm_a_stripped": arm_json("arm_a_stripped", strip.name, proj.weight_bytes, a_ns, 1, occ, json!({"arithmetic": "xor/add sink of codes+scales+x; same access pattern"})),
            "arm_b_halfk": arm_json("arm_b_halfk", half.name, half_bytes, b_ns, 1, occ, json!({"work_cols": proj.cols / 2})),
            "arm_a_halfk": arm_json("arm_a_halfk", strip.name, half_bytes, a_half_ns, 1, occ, json!({"role": "DCE proof: stripped arithmetic, half the bytes"})),
            "zero_load": arm_json("zero_load", zero.name, proj.weight_bytes, zero_ns, 1, occ, json!({"role": "launch+reduction floor; no weight/x loads"})),
        })
    }

    fn run_deltanet_decompose(args: &Args) -> Value {
        let concurrent_start = concurrent_load();
        eprintln!(
            "alu_roofline_organs decompose opening {} layer={} alu_warmup={} alu_reps={} session_warmup={} session_reps={}",
            args.artifact_root.display(),
            args.layer,
            args.warmup,
            args.reps,
            args.session_warmup,
            args.session_reps
        );
        let catalog = parse_catalog(&args.artifact_root).unwrap_or_else(|e| fail(e));
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal-capable GPU"));
        let queue = device.new_command_queue();
        let pipes = compile(&device).unwrap_or_else(|e| fail(e));
        let q4_prod = pipes
            .get("qwen_uniform_q4_group64_matvec_geo_tpr64_tg128")
            .unwrap();
        let q4_strip = pipes.get("alu_roofline_q4_geo_tpr64_tg128_stripped").unwrap();
        let q4_half = pipes.get("alu_roofline_q4_geo_tpr64_tg128_halfk").unwrap();
        let q4_zero = pipes.get("alu_roofline_q4_geo_tpr64_tg128_zero").unwrap();

        let qkvz_name = qwen38_layer_name(args.layer, "linear_attn.in_proj_qkvz.weight");
        let ba_name = qwen38_layer_name(args.layer, "linear_attn.in_proj_ba.weight");
        let out_name = qwen38_layer_name(args.layer, "linear_attn.out_proj.weight");
        eprintln!("  loading {qkvz_name}");
        let qkvz = load_q4(&device, &catalog, &qkvz_name).unwrap_or_else(|e| fail(e));
        eprintln!("  loading {out_name}");
        let outp = load_q4(&device, &catalog, &out_name).unwrap_or_else(|e| fail(e));
        eprintln!("  loading {ba_name}");
        let ba = load_q4(&device, &catalog, &ba_name).unwrap_or_else(|e| fail(e));
        if qkvz.cols % 2 != 0 || outp.cols % 2 != 0 || ba.cols % 2 != 0 {
            fail("Q4 cols must be even for ARM B");
        }

        let encode_one = |enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, kind: &ArmKind, proj: &Q4Proj| {
            encode_q4(enc, pipe, proj, kind);
        };

        eprintln!("  ALU qkvz / out_proj / ba back-to-back (same process)");
        let qkvz_occ = occupancy(q4_prod, qkvz.rows);
        let out_occ = occupancy(q4_prod, outp.rows);
        let ba_occ = occupancy(q4_prod, ba.rows);
        let (q_prod, q_a, q_b, q_z, q_ah) = time_q4_arms(
            &queue, args.warmup, args.reps, q4_prod, q4_strip, q4_half, q4_zero, &qkvz,
            |enc, pipe, kind| encode_one(enc, pipe, kind, &qkvz),
        );
        let (o_prod, o_a, o_b, o_z, o_ah) = time_q4_arms(
            &queue, args.warmup, args.reps, q4_prod, q4_strip, q4_half, q4_zero, &outp,
            |enc, pipe, kind| encode_one(enc, pipe, kind, &outp),
        );
        let (b_prod, b_a, b_b, b_z, b_ah) = time_q4_arms(
            &queue, args.warmup, args.reps, q4_prod, q4_strip, q4_half, q4_zero, &ba,
            |enc, pipe, kind| encode_one(enc, pipe, kind, &ba),
        );

        let alu = json!({
            "in_proj_qkvz": q4_organ_json("in_proj_qkvz", &qkvz, &qkvz_occ, q4_prod, q4_strip, q4_half, q4_zero, q_prod, q_a, q_b, q_z, q_ah),
            "out_proj": q4_organ_json("out_proj", &outp, &out_occ, q4_prod, q4_strip, q4_half, q4_zero, o_prod, o_a, o_b, o_z, o_ah),
            "in_proj_ba": q4_organ_json("in_proj_ba", &ba, &ba_occ, q4_prod, q4_strip, q4_half, q4_zero, b_prod, b_a, b_b, b_z, b_ah),
        });

        drop(qkvz);
        drop(outp);
        drop(ba);
        drop(pipes);
        drop(queue);

        eprintln!("  opening hybrid session (production 628-graph fusions)");
        std::env::set_var("HAWKING_QWEN_RESIDENCY", "1");
        std::env::remove_var("HAWKING_TCB_TRACE");
        std::env::remove_var("HAWKING_QWEN38_FUSE_MLP");
        std::env::remove_var("HAWKING_QWEN38_FUSE_GQA_QKV");
        std::env::remove_var("HAWKING_QWEN38_FUSE_DN_INPROJ");
        std::env::remove_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM");
        std::env::remove_var("HAWKING_QWEN38_FUSE_BA_DELTA");
        std::env::remove_var("HAWKING_QWEN38_DN_STATE");
        let open_started = Instant::now();
        let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, 64)
            .unwrap_or_else(|e| fail(e));
        let session_open_s = open_started.elapsed().as_secs_f64();
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
        session.set_fuse_add_rmsnorm(true, false);
        session.set_fuse_ba_delta(false, false);
        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
        eprintln!(
            "  session open {session_open_s:.3}s dense_w={} theoretical_disp={}",
            session.dense_w_materialized,
            session.theoretical_dispatches()
        );

        let concurrent_session = concurrent_load();
        let sw = args.session_warmup;
        let sr = args.session_reps;

        eprintln!("  structural names of as-executed organ");
        let (named_t, kernel_names) = session
            .measure_dn_as_executed_named()
            .unwrap_or_else(|e| fail(e));
        let mut kernel_hist: HashMap<String, u64> = HashMap::new();
        for n in &kernel_names {
            *kernel_hist.entry(n.clone()).or_insert(0) += 1;
        }
        let mut kernel_hist_rows: Vec<(String, u64)> =
            kernel_hist.into_iter().collect();
        kernel_hist_rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut families = serde_json::Map::new();
        let push_family = |map: &mut serde_json::Map<String, Value>,
                           name: &str,
                           kernel: &str,
                           bytes: u64,
                           gpu: Vec<u64>,
                           disp: u64,
                           extra: Value| {
            map.insert(
                name.to_string(),
                family_arm(name, kernel, bytes, gpu, disp, extra),
            );
        };

        let run_fam = |session: &Qwen38HybridDecodeSession, name: &str| {
            time_session_family(session, name, sw, sr).unwrap_or_else(|e| fail(e))
        };

        {
            eprintln!("  noop_empty warmup={sw} reps={sr}");
            let mut gpu = Vec::new();
            let mut disp = 0u64;
            for i in 0..(sw + sr) {
                let t = session
                    .measure_isolated_organ("noop_empty")
                    .unwrap_or_else(|e| fail(e));
                let g = t.gpu_ns.unwrap_or(0);
                eprintln!("    i{i} gpu={g} wait={} disp={}", t.wait_ns, t.dispatches);
                if i >= sw {
                    gpu.push(g);
                    disp = t.dispatches;
                }
            }
            push_family(
                &mut families,
                "noop_empty",
                "empty_cb",
                0,
                gpu,
                disp,
                json!({"role": "empty command-buffer floor; missing GPU timestamps recorded as 0"}),
            );
        }

        let partition = [
            ("dn_as_executed", "encode_deltanet x 48", 2_961_659_904u64),
            ("dn_inproj", "pair_concat qkvz+ba x 48", 2_151_632_640u64),
            ("dn_qkvz", "qkvz Q4 x 48", 2_139_096_960u64),
            ("dn_ba", "ba Q4 x 48", 12_535_680u64),
            ("dn_out_proj", "out_proj Q4 x 48", 802_162_560u64),
            ("rearrange_48", "qwen38_qkvz_rearrange_conv_l2_f32 x 48", 7_864_704u64),
            ("ba_to_decay_48", "qwen80_ba_to_decay_beta_f32 x 48", 19_200u64),
            ("gated_rmsnorm_48", "qwen80_deltanet_gated_rmsnorm_tg x 48", 24_960u64),
            ("dn_residual_rmsnorm", "add_residual_rmsnorm x 48 DN", 983_040u64),
            ("dn_input_rmsnorm", "layer0 input_layernorm", 20_480u64),
        ];
        for (name, kernel, bytes) in partition {
            let (g, d, enc) = run_fam(&session, name);
            push_family(
                &mut families,
                name,
                kernel,
                bytes,
                g,
                d,
                json!({"encoders": enc, "production_628_graph": true}),
            );
        }

        let (g, d, enc) = time_session_custom("gated_delta_unfused", sw, sr, || {
            session
                .measure_isolated_gated_delta()
                .map_err(|e| e.to_string())
        })
        .unwrap_or_else(|e| fail(e));
        push_family(
            &mut families,
            "gated_delta_unfused",
            "qwen38_gated_delta_decode_vi_simd x 48",
            301_989_888,
            g,
            d,
            json!({"encoders": enc, "traffic": "rec_state R+W", "production": true}),
        );

        session.set_fuse_ba_delta(true, false);
        let (g, d, enc) = time_session_custom("gated_delta_fused_ba", sw, sr, || {
            session
                .measure_isolated_dn_state_update()
                .map_err(|e| e.to_string())
        })
        .unwrap_or_else(|e| fail(e));
        push_family(
            &mut families,
            "gated_delta_fused_ba",
            "qwen38_gated_delta_decode_vi_simd_ba x 48",
            301_989_888,
            g,
            d,
            json!({"encoders": enc, "role": "N026 comparator; not on the 628 graph"}),
        );

        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
        let (g, d, enc) = time_session_custom("gated_delta_widen_f4", sw, sr, || {
            session
                .measure_isolated_dn_state_update()
                .map_err(|e| e.to_string())
        })
        .unwrap_or_else(|e| fail(e));
        push_family(
            &mut families,
            "gated_delta_widen_f4",
            "qwen38_gated_delta_decode_vi_simd_ba_f4 x 48",
            301_989_888,
            g,
            d,
            json!({"encoders": enc, "role": "layout/ALU probe of the recurrent update"}),
        );
        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
        session.set_fuse_ba_delta(false, false);

        let rec_elems = session.rec_state_f32_count();
        let rec_bytes = (rec_elems * 4) as u64;
        let dest = session
            .alloc_profile_buffer(rec_bytes as usize)
            .unwrap_or_else(|e| fail(e));
        let (g, d, enc) = time_session_custom("rec_state_f32_stream", sw, sr, || {
            session
                .measure_f32_stream("rec_state", &dest)
                .map_err(|e| e.to_string())
        })
        .unwrap_or_else(|e| fail(e));
        push_family(
            &mut families,
            "rec_state_f32_stream",
            "qwen38_f32_stream_probe",
            rec_bytes,
            g,
            d,
            json!({
                "encoders": enc,
                "role": "contiguous copy of rec_state; bandwidth floor for the recurrent buffer",
                "f32_count": rec_elems,
            }),
        );

        let (g, d, enc) = time_session_custom("organ_incomplete_missing_out_proj", sw, sr, || {
            session
                .measure_isolated_organ("deltanet")
                .map_err(|e| e.to_string())
        })
        .unwrap_or_else(|e| fail(e));
        push_family(
            &mut families,
            "organ_incomplete_missing_out_proj",
            "encode_organ_dn_compute (no out_proj)",
            2_151_632_640,
            g,
            d,
            json!({"encoders": enc, "role": "N026 isolated organ; 288 launches, out_proj billed elsewhere"}),
        );

        json!({
            "schema": "hawking.future.deltanet_organ_decompose.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "session_warmup": args.session_warmup,
            "session_reps": args.session_reps,
            "session_open_s": session_open_s,
            "dense_w_materialized": session.dense_w_materialized,
            "theoretical_dispatches_628_graph": session.theoretical_dispatches(),
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "concurrent_load_start": concurrent_start,
            "concurrent_load": concurrent_session,
            "absolute_gb_s_are_measured_under_load": true,
            "production_fusions": {
                "mlp": "GateUpSwiglu",
                "fuse_gqa_qkv": true,
                "fuse_dn_inproj": true,
                "fuse_add_rmsnorm": true,
                "fuse_ba_delta": false,
                "dn_state_kernel": "baseline",
            },
            "as_executed_named": {
                "gpu_ns": named_t.gpu_ns,
                "dispatches": named_t.dispatches,
                "encoder_count": named_t.encoder_count,
                "command_buffers": named_t.command_buffers,
                "kernel_names_in_order": kernel_names,
                "kernel_histogram": kernel_hist_rows.iter().map(|(k, n)| json!({"kernel": k, "count": n})).collect::<Vec<_>>(),
            },
            "alu_matched_pair": alu,
            "families": families,
        })
    }

    pub fn run(args: Args) -> Value {
        if args.mode == "deltanet-decompose" {
            return run_deltanet_decompose(&args);
        }
        let concurrent = concurrent_load();
        eprintln!(
            "alu_roofline_organs opening {} layer={} warmup={} reps={}",
            args.artifact_root.display(),
            args.layer,
            args.warmup,
            args.reps
        );
        let catalog = parse_catalog(&args.artifact_root).unwrap_or_else(|e| fail(e));
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal-capable GPU"));
        let queue = device.new_command_queue();
        let pipes = compile(&device).unwrap_or_else(|e| fail(e));

        let gate_name = qwen38_layer_name(args.layer, "mlp.gate_proj.weight");
        let up_name = qwen38_layer_name(args.layer, "mlp.up_proj.weight");
        let down_name = qwen38_layer_name(args.layer, "mlp.down_proj.weight");
        let qkvz_name = qwen38_layer_name(args.layer, "linear_attn.in_proj_qkvz.weight");

        eprintln!("  loading {gate_name}");
        let gate = load_affine(&device, &catalog, &gate_name).unwrap_or_else(|e| fail(e));
        eprintln!("  loading {up_name}");
        let up = load_affine(&device, &catalog, &up_name).unwrap_or_else(|e| fail(e));
        eprintln!("  loading {down_name}");
        let down = load_affine(&device, &catalog, &down_name).unwrap_or_else(|e| fail(e));
        let mlp = [gate, up, down];
        let mlp_bytes: u64 = mlp.iter().map(|p| p.weight_bytes).sum();
        let mlp_rows_gate = mlp[0].rows;
        let mlp_cols = mlp[0].cols;
        let mlp_gs = mlp[0].group_size;
        if mlp_cols % 2 != 0 {
            fail("MLP cols must be even for ARM B");
        }
        let mlp_work_half = mlp_cols / 2;
        // down_proj has transposed K; half of THAT cols.
        let down_half = mlp[2].cols / 2;

        eprintln!("  loading {qkvz_name}");
        let qkvz = load_q4(&device, &catalog, &qkvz_name).unwrap_or_else(|e| fail(e));
        if qkvz.cols % 2 != 0 {
            fail("qkvz cols must be even for ARM B");
        }
        let qkvz_half = qkvz.cols / 2;

        let affine_prod = pipes
            .get("qwen_affine_q2_group32_matvec_geo_tpr64_tg128")
            .unwrap();
        let affine_strip = pipes
            .get("alu_roofline_affine_q2_geo_tpr64_tg128_stripped")
            .unwrap();
        let affine_half = pipes
            .get("alu_roofline_affine_q2_geo_tpr64_tg128_halfk")
            .unwrap();
        let affine_zero = pipes
            .get("alu_roofline_affine_q2_geo_tpr64_tg128_zero")
            .unwrap();
        let q4_prod = pipes
            .get("qwen_uniform_q4_group64_matvec_geo_tpr64_tg128")
            .unwrap();
        let q4_strip = pipes.get("alu_roofline_q4_geo_tpr64_tg128_stripped").unwrap();
        let q4_half = pipes.get("alu_roofline_q4_geo_tpr64_tg128_halfk").unwrap();
        let q4_zero = pipes.get("alu_roofline_q4_geo_tpr64_tg128_zero").unwrap();

        let mlp_occ = occupancy(affine_prod, mlp_rows_gate);
        let q4_occ = occupancy(q4_prod, qkvz.rows);

        let time_mlp = |n: usize, kind: ArmKind, pipe: &Pipe| {
            time_cb(&queue, n, |enc| {
                for (i, p) in mlp.iter().enumerate() {
                    let k = match &kind {
                        ArmKind::AffineHalfk { .. } if i == 2 => ArmKind::AffineHalfk {
                            work_cols: down_half,
                        },
                        ArmKind::AffineStripped { work_cols } if i == 2 && *work_cols != 0 => {
                            ArmKind::AffineStripped {
                                work_cols: down_half,
                            }
                        }
                        other => match other {
                            ArmKind::AffineProd => ArmKind::AffineProd,
                            ArmKind::AffineStripped { work_cols } => ArmKind::AffineStripped {
                                work_cols: *work_cols,
                            },
                            ArmKind::AffineHalfk { work_cols } => ArmKind::AffineHalfk {
                                work_cols: *work_cols,
                            },
                            ArmKind::AffineZero => ArmKind::AffineZero,
                            _ => ArmKind::AffineProd,
                        },
                    };
                    encode_affine(enc, pipe, p, &k);
                }
            })
        };

        eprintln!("  MLP warmup+measure (4 arms back-to-back)");
        let _ = time_mlp(args.warmup, ArmKind::AffineProd, affine_prod);
        let mlp_prod_ns = time_mlp(args.reps, ArmKind::AffineProd, affine_prod);
        let _ = time_mlp(
            args.warmup,
            ArmKind::AffineStripped { work_cols: 0 },
            affine_strip,
        );
        let mlp_a_ns = time_mlp(
            args.reps,
            ArmKind::AffineStripped { work_cols: 0 },
            affine_strip,
        );
        let _ = time_mlp(
            args.warmup,
            ArmKind::AffineHalfk {
                work_cols: mlp_work_half,
            },
            affine_half,
        );
        let mlp_b_ns = time_mlp(
            args.reps,
            ArmKind::AffineHalfk {
                work_cols: mlp_work_half,
            },
            affine_half,
        );
        let _ = time_mlp(args.warmup, ArmKind::AffineZero, affine_zero);
        let mlp_zero_ns = time_mlp(args.reps, ArmKind::AffineZero, affine_zero);
        let _ = time_mlp(
            args.warmup,
            ArmKind::AffineStripped {
                work_cols: mlp_work_half,
            },
            affine_strip,
        );
        let mlp_a_half_ns = time_mlp(
            args.reps,
            ArmKind::AffineStripped {
                work_cols: mlp_work_half,
            },
            affine_strip,
        );

        let mlp_b_bytes = mlp[0].weight_bytes / 2 + mlp[1].weight_bytes / 2 + mlp[2].weight_bytes / 2;
        let mlp_a_half_bytes = mlp_b_bytes;

        let time_q4 = |n: usize, kind: ArmKind, pipe: &Pipe| {
            time_cb(&queue, n, |enc| encode_q4(enc, pipe, &qkvz, &kind))
        };

        eprintln!("  DeltaNet qkvz warmup+measure (4 arms back-to-back)");
        let _ = time_q4(args.warmup, ArmKind::Q4Prod, q4_prod);
        let dn_prod_ns = time_q4(args.reps, ArmKind::Q4Prod, q4_prod);
        let _ = time_q4(args.warmup, ArmKind::Q4Stripped { work_cols: 0 }, q4_strip);
        let dn_a_ns = time_q4(args.reps, ArmKind::Q4Stripped { work_cols: 0 }, q4_strip);
        let _ = time_q4(
            args.warmup,
            ArmKind::Q4Halfk {
                work_cols: qkvz_half,
            },
            q4_half,
        );
        let dn_b_ns = time_q4(
            args.reps,
            ArmKind::Q4Halfk {
                work_cols: qkvz_half,
            },
            q4_half,
        );
        let _ = time_q4(args.warmup, ArmKind::Q4Zero, q4_zero);
        let dn_zero_ns = time_q4(args.reps, ArmKind::Q4Zero, q4_zero);
        let _ = time_q4(
            args.warmup,
            ArmKind::Q4Stripped {
                work_cols: qkvz_half,
            },
            q4_strip,
        );
        let dn_a_half_ns = time_q4(
            args.reps,
            ArmKind::Q4Stripped {
                work_cols: qkvz_half,
            },
            q4_strip,
        );

        let trips_mlp = mlp_cols / 512;
        let trips_dn = qkvz.cols / 512;
        let bytes_per_iter_affine = 2u64 + 2 + 2 + 32; // ushort + half + half + 8 f32
        let bytes_per_iter_q4 = 4u64 + 2 + 32; // uint + half + 8 f32

        json!({
            "schema": "hawking.future.mlp_alu_roofline.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "concurrent_load": concurrent,
            "absolute_gb_s_are_measured_under_load": true,
            "mlp": {
                "organ": "mlp",
                "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
                "codec": "HGRAVF01 affine2 q2 group64",
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
                "threads_per_threadgroup": TG,
                "bytes_per_thread_iteration": bytes_per_iter_affine,
                "inner_loop_trips_gate": trips_mlp,
                "production": arm_json(
                    "production",
                    affine_prod.name,
                    mlp_bytes,
                    mlp_prod_ns,
                    3,
                    &mlp_occ,
                    json!({"group_size": mlp_gs, "cols": mlp_cols}),
                ),
                "arm_a_stripped": arm_json(
                    "arm_a_stripped",
                    affine_strip.name,
                    mlp_bytes,
                    mlp_a_ns,
                    3,
                    &mlp_occ,
                    json!({"arithmetic": "xor/add sink of codes+scales+biases+x; same access pattern"}),
                ),
                "arm_b_halfk": arm_json(
                    "arm_b_halfk",
                    affine_half.name,
                    mlp_b_bytes,
                    mlp_b_ns,
                    3,
                    &mlp_occ,
                    json!({"work_cols_gate_up": mlp_work_half, "work_cols_down": down_half}),
                ),
                "arm_a_halfk": arm_json(
                    "arm_a_halfk",
                    affine_strip.name,
                    mlp_a_half_bytes,
                    mlp_a_half_ns,
                    3,
                    &mlp_occ,
                    json!({"role": "DCE proof: stripped arithmetic, half the bytes"}),
                ),
                "zero_load": arm_json(
                    "zero_load",
                    affine_zero.name,
                    mlp_bytes,
                    mlp_zero_ns,
                    3,
                    &mlp_occ,
                    json!({"role": "launch+reduction floor; no weight/x loads"}),
                ),
            },
            "deltanet": {
                "organ": "deltanet",
                "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                "codec": "HQ30UQ4 group64",
                "projection": {
                    "name": qkvz.name,
                    "rows": qkvz.rows,
                    "cols": qkvz.cols,
                    "group_size": qkvz.group_size,
                    "groups_per_row": qkvz.groups_per_row,
                    "weight_bytes": qkvz.weight_bytes,
                    "code_bytes": qkvz.code_bytes,
                    "scale_bytes": qkvz.scale_bytes,
                },
                "threads_per_threadgroup": TG,
                "bytes_per_thread_iteration": bytes_per_iter_q4,
                "inner_loop_trips": trips_dn,
                "production": arm_json(
                    "production",
                    q4_prod.name,
                    qkvz.weight_bytes,
                    dn_prod_ns,
                    1,
                    &q4_occ,
                    json!({"cols": qkvz.cols}),
                ),
                "arm_a_stripped": arm_json(
                    "arm_a_stripped",
                    q4_strip.name,
                    qkvz.weight_bytes,
                    dn_a_ns,
                    1,
                    &q4_occ,
                    json!({"arithmetic": "xor/add sink of codes+scales+x; same access pattern"}),
                ),
                "arm_b_halfk": arm_json(
                    "arm_b_halfk",
                    q4_half.name,
                    qkvz.weight_bytes / 2,
                    dn_b_ns,
                    1,
                    &q4_occ,
                    json!({"work_cols": qkvz_half}),
                ),
                "arm_a_halfk": arm_json(
                    "arm_a_halfk",
                    q4_strip.name,
                    qkvz.weight_bytes / 2,
                    dn_a_half_ns,
                    1,
                    &q4_occ,
                    json!({"role": "DCE proof: stripped arithmetic, half the bytes"}),
                ),
                "zero_load": arm_json(
                    "zero_load",
                    q4_zero.name,
                    qkvz.weight_bytes,
                    dn_zero_ns,
                    1,
                    &q4_occ,
                    json!({"role": "launch+reduction floor; no weight/x loads"}),
                ),
            },
        })
    }
}
