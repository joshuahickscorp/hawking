//! Per-stream marginal cost of a unique byte on the MLP organ.
//!
//! Hold geometry, trip count, occupancy and the other two streams fixed.
//! Drop a controlled keep-fraction of ONE stream's loads and measure the
//! GPU time delta as a paired A/B (treat vs baseline, alternating order)
//! so a GPU-frequency flip cannot fake a stream delta. That paired dt /
//! unique-bytes-dropped is the calibration executable_economics never
//! had: unique aux bytes are broadcast and cache-served, unique code
//! bytes are per-thread and binding.
//!
//! Does not mutate the production decode path.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example stream_criticality_probe
//! ./tools/gpu_lane_lock.sh z2econ \
//!   workspace/ops/build/rust/release-fast/examples/stream_criticality_probe \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_ECONOMICS_CALIBRATION_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: stream_criticality_probe --artifact-root DIR \
        [--layer N] [--warmup N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("stream_criticality_probe: {message}");
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
    let mut layer = 3usize;
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
        "note": "absolute GPU ns is measured-under-load; A/B is back-to-back in one process",
        "top_cpu": rows.iter().map(|(cpu, rss, pid, comm)| json!({
            "pid": pid,
            "cpu_pct": cpu,
            "rss_kib": rss,
            "comm": comm,
        })).collect::<Vec<_>>(),
    })
}

/// Frozen before any GPU timestamp is taken.
fn pre_registered_interpretation() -> Value {
    json!({
        "registered_before_measurement": true,
        "geometry": "geo_tpr64_tg128, trip count held, occupancy held",
        "arithmetic": "stripped (XOR/add sink); this is streaming, not ALU",
        "mechanism": (
            "Drop a keep-fraction of ONE stream's loads. Unique bytes dropped \
             are the catalog bytes of that stream times (1 - keep). Time delta \
             over unique bytes is the stream's marginal cost. Defaulting a \
             candidate to the organ-average GB/s is how the aux-u8 overcredit \
             stayed invisible."
        ),
        "classes": {
            "weight_codes": "2 B/iter, per-thread-unique, the binding stream",
            "broadcast_aux": "2+2 B/iter scale+bias, per-group, many threads share",
            "activation": "32 B/iter x, reused across rows of the same tile",
        },
        "rules": [
            "If dropping unique code bytes saves time at a rate comparable to the organ average, codes are on the critical path.",
            "If dropping unique aux bytes saves ~0 time (inside replicate noise), aux is not on the critical path and must not be billed at the organ average.",
            "Activation is measured, not assumed. Unique x bytes are small; the rate is still a unique-byte marginal because that is what candidates declare.",
            "A candidate that does not declare a stream class is refused.",
        ],
        "keep_percents": [0, 25, 50, 75, 100],
        "noise_rule": "a |dt| below 2*MAD of the baseline reps is treated as 0 when the sidecar fits the slope",
    })
}

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) -> Value {
    fail("stream_criticality_probe is Metal-only")
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

    const SHADER: &str = include_str!("stream_criticality_probe.metal");
    const TG: u64 = 128;
    const GPU_CORES: u32 = 60;
    const KEEP_PCTS: [u32; 5] = [100, 75, 50, 25, 0];

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
    fn mad_u64(v: &[u64], med: u64) -> u64 {
        if v.is_empty() {
            return 0;
        }
        let mut d: Vec<u64> = v.iter().map(|x| x.abs_diff(med)).collect();
        d.sort_unstable();
        d[d.len() / 2]
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

    const KERNELS: &[&str] = &["stream_crit_gated", "stream_crit_empty"];

    fn compile(device: &Device) -> Result<HashMap<&'static str, Pipe>, String> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("shader compile: {e}"))?;
        eprintln!(
            "stream_criticality_probe: compiled shaders in {:.2}s",
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
        codes: Buffer,
        scales: Buffer,
        biases: Buffer,
        input: Buffer,
        output: Buffer,
        weight_bytes: u64,
        code_bytes: u64,
        scale_bytes: u64,
        bias_bytes: u64,
        x_bytes: u64,
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
        let x_bytes = (packed.cols * 4) as u64;
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
            x_bytes,
        })
    }

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(TG, 1, 1));
    }

    fn encode_gated(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        proj: &AffineProj,
        code_limit: u32,
        aux_limit: u32,
        x_limit: u32,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        set_u32(enc, 7, code_limit);
        set_u32(enc, 8, aux_limit);
        set_u32(enc, 9, x_limit);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_empty(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, proj: &AffineProj) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.output), 0);
        set_u32(enc, 1, proj.rows);
        dispatch_groups(enc, proj.rows);
    }

    fn time_one<F>(queue: &metal::CommandQueue, n_proj: usize, mut fill: F) -> Option<u64>
    where
        F: FnMut(&metal::ComputeCommandEncoderRef, usize),
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for i in 0..n_proj {
            fill(&enc, i);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        gpu_ns(cmd)
    }

    fn time_all<F>(queue: &metal::CommandQueue, n_proj: usize, n: usize, mut fill: F) -> Vec<u64>
    where
        F: FnMut(&metal::ComputeCommandEncoderRef, usize),
    {
        let mut gpu = Vec::new();
        for _ in 0..n {
            if let Some(ns) = time_one(queue, n_proj, |enc, i| fill(enc, i)) {
                gpu.push(ns);
            }
        }
        gpu
    }

    /// Paired A/B under concurrent load. Alternate order so a slow/fast
    /// GPU-frequency flip cannot fake a stream delta. dt = treat - base.
    fn time_paired<F, G>(
        queue: &metal::CommandQueue,
        n_proj: usize,
        warmup: usize,
        reps: usize,
        mut fill_base: F,
        mut fill_treat: G,
    ) -> (Vec<u64>, Vec<u64>, Vec<i64>)
    where
        F: FnMut(&metal::ComputeCommandEncoderRef, usize),
        G: FnMut(&metal::ComputeCommandEncoderRef, usize),
    {
        for _ in 0..warmup {
            let _ = time_one(queue, n_proj, |enc, i| fill_base(enc, i));
            let _ = time_one(queue, n_proj, |enc, i| fill_treat(enc, i));
        }
        let mut bases = Vec::new();
        let mut treats = Vec::new();
        let mut dts = Vec::new();
        for i in 0..reps {
            let (a, b) = if i % 2 == 0 {
                let a = time_one(queue, n_proj, |enc, i| fill_base(enc, i));
                let b = time_one(queue, n_proj, |enc, i| fill_treat(enc, i));
                (a, b)
            } else {
                let b = time_one(queue, n_proj, |enc, i| fill_treat(enc, i));
                let a = time_one(queue, n_proj, |enc, i| fill_base(enc, i));
                (a, b)
            };
            if let (Some(a), Some(b)) = (a, b) {
                bases.push(a);
                treats.push(b);
                dts.push(b as i64 - a as i64);
            }
        }
        (bases, treats, dts)
    }

    fn median_i64(mut v: Vec<i64>) -> i64 {
        if v.is_empty() {
            return 0;
        }
        v.sort_unstable();
        v[v.len() / 2]
    }

    fn mad_i64(v: &[i64], med: i64) -> i64 {
        if v.is_empty() {
            return 0;
        }
        let mut d: Vec<i64> = v.iter().map(|x| (*x - med).abs()).collect();
        d.sort_unstable();
        d[d.len() / 2]
    }

    fn keep_limit(cols: u32, keep_pct: u32) -> u32 {
        if keep_pct >= 100 {
            cols
        } else if keep_pct == 0 {
            0
        } else {
            ((u64::from(cols) * u64::from(keep_pct)) / 100) as u32
        }
    }

    fn unique_kept(full: u64, keep_pct: u32) -> u64 {
        if keep_pct >= 100 {
            full
        } else if keep_pct == 0 {
            0
        } else {
            (full * u64::from(keep_pct)) / 100
        }
    }

    pub fn run(args: Args) -> Value {
        let interpretation = pre_registered_interpretation();
        let concurrent_start = concurrent_load();
        eprintln!(
            "stream_criticality_probe opening {} layer={} warmup={} reps={}",
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
        let device = Device::system_default().unwrap_or_else(|| {
            fail(
                "no Metal-capable GPU: MTLCreateSystemDefaultDevice returned nil. \
                 This probe needs the gate profile (unsandboxed) so the process can \
                 see a Metal device. Refusing to fabricate a hardware number.",
            )
        });
        eprintln!(
            "  Metal device: {}",
            device.name().to_string()
        );
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
        let n_proj = mlp.len();
        let k_gated = pipes.get("stream_crit_gated").unwrap();
        let k_empty = pipes.get("stream_crit_empty").unwrap();
        let occ = occupancy(k_gated, mlp[0].rows);

        let code_bytes: u64 = mlp.iter().map(|p| p.code_bytes).sum();
        let aux_bytes: u64 = mlp.iter().map(|p| p.scale_bytes + p.bias_bytes).sum();
        let x_bytes: u64 = mlp.iter().map(|p| p.x_bytes).sum();
        let weight_bytes: u64 = mlp.iter().map(|p| p.weight_bytes).sum();

        let mut rungs: Vec<Value> = Vec::new();

        let fill_base = |enc: &metal::ComputeCommandEncoderRef, i: usize| {
            encode_gated(
                enc,
                k_gated,
                &mlp[i],
                keep_limit(mlp[i].cols, 100),
                keep_limit(mlp[i].cols, 100),
                keep_limit(mlp[i].cols, 100),
            );
        };

        eprintln!("  baseline (unpaired, for absolute us)");
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| fill_base(enc, i));
        let base_ns = time_all(&queue, n_proj, args.reps, |enc, i| fill_base(enc, i));
        let base_med = median_u64(base_ns.clone()).unwrap_or(0);
        let base_mad = mad_u64(&base_ns, base_med);
        rungs.push(json!({
            "id": "baseline",
            "kernel": "stream_crit_gated",
            "stream_class": "baseline",
            "keep_pct": 100,
            "drop_pct": 0,
            "unique_bytes_dropped": 0,
            "gpu_ns_median": base_med,
            "gpu_ns_reps": base_ns,
            "gpu_us_median": base_med as f64 / 1e3,
            "dt_ns_vs_baseline": 0,
            "dt_us_vs_baseline": 0.0,
            "paired": false,
            "dispatches": 3,
            "encoders": 1,
            "command_buffers": 1,
            "threads_per_threadgroup": TG,
            "occupancy": occ,
        }));

        let dropped_of = |stream: &str, keep_pct: u32| -> u64 {
            match stream {
                "weight_codes" => code_bytes - unique_kept(code_bytes, keep_pct),
                "broadcast_aux" => aux_bytes - unique_kept(aux_bytes, keep_pct),
                "activation" => x_bytes - unique_kept(x_bytes, keep_pct),
                "zero_load" => code_bytes + aux_bytes + x_bytes,
                other => fail(format!("unknown stream {other}")),
            }
        };

        let mut run_paired = |id: &str,
                              stream: &str,
                              keep_pct: u32,
                              code_keep: u32,
                              aux_keep: u32,
                              x_keep: u32| {
            eprintln!("  {id} paired vs baseline (stream={stream} keep={keep_pct})");
            let (bases, treats, dts) = time_paired(
                &queue,
                n_proj,
                args.warmup,
                args.reps,
                |enc, i| fill_base(enc, i),
                |enc, i| {
                    encode_gated(
                        enc,
                        k_gated,
                        &mlp[i],
                        keep_limit(mlp[i].cols, code_keep),
                        keep_limit(mlp[i].cols, aux_keep),
                        keep_limit(mlp[i].cols, x_keep),
                    );
                },
            );
            let treat_med = median_u64(treats.clone()).unwrap_or(0);
            let dt_med = median_i64(dts.clone());
            let dt_mad = mad_i64(&dts, dt_med);
            let dropped = dropped_of(stream, keep_pct);
            rungs.push(json!({
                "id": id,
                "kernel": "stream_crit_gated",
                "stream_class": stream,
                "keep_pct": keep_pct,
                "drop_pct": 100u32.saturating_sub(keep_pct),
                "unique_bytes_dropped": dropped,
                "gpu_ns_median": treat_med,
                "gpu_ns_reps": treats,
                "gpu_us_median": treat_med as f64 / 1e3,
                "paired": true,
                "paired_baseline_gpu_ns_reps": bases,
                "paired_dt_ns_reps": dts,
                "dt_ns_vs_baseline": dt_med,
                "dt_us_vs_baseline": dt_med as f64 / 1e3,
                "paired_dt_ns_mad": dt_mad,
                "dispatches": 3,
                "encoders": 1,
                "command_buffers": 1,
                "threads_per_threadgroup": TG,
                "occupancy": occ,
            }));
        };

        for keep in KEEP_PCTS {
            if keep == 100 {
                continue;
            }
            let id = format!("codes_keep_{keep}");
            run_paired(&id, "weight_codes", keep, keep, 100, 100);
        }
        for keep in KEEP_PCTS {
            if keep == 100 {
                continue;
            }
            let id = format!("aux_keep_{keep}");
            run_paired(&id, "broadcast_aux", keep, 100, keep, 100);
        }
        for keep in KEEP_PCTS {
            if keep == 100 {
                continue;
            }
            let id = format!("x_keep_{keep}");
            run_paired(&id, "activation", keep, 100, 100, keep);
        }

        eprintln!("  zero_load paired vs baseline");
        run_paired("zero_load", "zero_load", 0, 0, 0, 0);

        eprintln!("  empty (launch+reduction, no loop; unpaired floor)");
        let _ = time_all(&queue, n_proj, args.warmup, |enc, i| {
            encode_empty(enc, k_empty, &mlp[i])
        });
        let empty_ns = time_all(&queue, n_proj, args.reps, |enc, i| {
            encode_empty(enc, k_empty, &mlp[i])
        });
        let empty_med = median_u64(empty_ns.clone()).unwrap_or(0);
        rungs.push(json!({
            "id": "empty",
            "kernel": "stream_crit_empty",
            "stream_class": "none",
            "keep_pct": 0,
            "drop_pct": 100,
            "unique_bytes_dropped": 0,
            "gpu_ns_median": empty_med,
            "gpu_ns_reps": empty_ns,
            "gpu_us_median": empty_med as f64 / 1e3,
            "dt_ns_vs_baseline": Value::Null,
            "dt_us_vs_baseline": Value::Null,
            "paired": false,
            "dispatches": 3,
            "encoders": 1,
            "command_buffers": 1,
            "threads_per_threadgroup": TG,
            "occupancy": occ,
            "role": "launch+reduction floor; no inner loop",
        }));

        let concurrent_end = concurrent_load();
        json!({
            "schema": "hawking.future.stream_criticality.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "metal_device": device.name().to_string(),
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "absolute_gpu_ns_are_measured_under_load": true,
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
                "x_bytes": p.x_bytes,
            })).collect::<Vec<_>>(),
            "census": {
                "code_bytes": code_bytes,
                "aux_bytes": aux_bytes,
                "x_bytes": x_bytes,
                "weight_bytes": weight_bytes,
                "note": "unique payload of the launched layer (gate+up+down). Organ-scale rates multiply unique bytes of a candidate, not this layer's bytes.",
            },
            "baseline_gpu_ns_median": base_med,
            "baseline_gpu_ns_mad": base_mad,
            "noise_floor_ns": 2 * base_mad,
            "dispatches": 3,
            "rungs": rungs,
        })
    }
}
