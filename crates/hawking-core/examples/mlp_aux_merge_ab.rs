//! Matched A/B of the production MLP matvec vs merged 4-byte scale+bias aux.
//!
//! ARM A is the production kernel body (two half planes). ARM B is the same
//! arithmetic with one half2 group record, interleaved in memory from the
//! existing planes. Same layer, same geometry, both orderings, warmup 60.
//! Does not mutate the production decode path and does not rewrite artifacts.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example mlp_aux_merge_ab
//! ./tools/gpu_lane_lock.sh mlpaux \
//!   workspace/ops/build/rust/release-fast/examples/mlp_aux_merge_ab \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_MLP_AUX_MERGE_AB_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: mlp_aux_merge_ab --artifact-root DIR \
        [--layer N] [--warmup N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("mlp_aux_merge_ab: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    layer: usize,
    warmup: usize,
    reps: usize,
    out: Option<PathBuf>,
}

const MIN_WARMUP: usize = 60;
const MIN_REPS: usize = 7;
const STEADY_STATE_MAX_SPREAD: f64 = 1.10;
const BYTES_PER_ITER: u64 = 38;

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut layer = 0usize;
    let mut warmup = MIN_WARMUP;
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
    if warmup < MIN_WARMUP {
        fail(format!(
            "--warmup {warmup} < {MIN_WARMUP}; WARMUP_5_LEAVES_THE_FIRST_MEASURED_ARM_OUTSIDE_STEADY_STATE"
        ));
    }
    if reps < MIN_REPS {
        fail(format!("--reps {reps} < {MIN_REPS}"));
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

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) -> Value {
    fail("mlp_aux_merge_ab is Metal-only")
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

    const SHADER: &str = include_str!("mlp_aux_merge_ab.metal");
    const TG: u64 = 128;
    const GPU_CORES: u32 = 60;
    const INCUMBENT_KERNEL: &str = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128";
    const MERGED_KERNEL: &str = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_aux_merge";

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
    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
    }
    fn zero_f32(buf: &Buffer, n: usize) {
        unsafe {
            std::slice::from_raw_parts_mut(buf.contents() as *mut f32, n).fill(0.0);
        }
    }

    /// Interleave per-group f16 scale and f16 bias into one 4-byte record.
    /// Host-side only; the on-disk planes are not rewritten.
    fn interleave_half2(scales: &[u16], biases: &[u16]) -> Result<Vec<u16>, String> {
        if scales.len() != biases.len() {
            return Err(format!(
                "scale/bias length mismatch {} vs {}",
                scales.len(),
                biases.len()
            ));
        }
        let mut out = Vec::with_capacity(scales.len() * 2);
        for i in 0..scales.len() {
            out.push(scales[i]);
            out.push(biases[i]);
        }
        Ok(out)
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
            "mlp_aux_merge_ab: compiled shaders in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        let mut out = HashMap::new();
        for name in [INCUMBENT_KERNEL, MERGED_KERNEL] {
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
        aux_merged: Buffer,
        input: Buffer,
        output: Buffer,
        weight_bytes: u64,
        code_bytes: u64,
        scale_bytes: u64,
        bias_bytes: u64,
        merged_aux_bytes: u64,
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
        let merged = interleave_half2(&packed.scales_f16, &packed.biases_f16)?;
        let input = fill_f32(packed.cols);
        let output = vec![0f32; packed.rows];
        let code_bytes = packed.codes.len() as u64;
        let scale_bytes = (packed.scales_f16.len() * 2) as u64;
        let bias_bytes = (packed.biases_f16.len() * 2) as u64;
        let merged_aux_bytes = (merged.len() * 2) as u64;
        if merged_aux_bytes != scale_bytes + bias_bytes {
            return Err(format!(
                "{name}: interleaved aux {merged_aux_bytes} B != scale+bias {}",
                scale_bytes + bias_bytes
            ));
        }
        Ok(AffineProj {
            name: name.to_string(),
            rows: packed.rows as u32,
            cols: packed.cols as u32,
            group_size: packed.group_size as u32,
            codes: buf_u8(device, &packed.codes),
            scales: buf_u16(device, &packed.scales_f16),
            biases: buf_u16(device, &packed.biases_f16),
            aux_merged: buf_u16(device, &merged),
            input: buf_f32(device, &input),
            output: buf_f32(device, &output),
            weight_bytes: code_bytes + scale_bytes + bias_bytes,
            code_bytes,
            scale_bytes,
            bias_bytes,
            merged_aux_bytes,
        })
    }

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(TG, 1, 1));
    }

    fn encode_incumbent(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, proj: &AffineProj) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales), 0);
        enc.set_buffer(2, Some(&proj.biases), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        set_u32(enc, 7, proj.group_size);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_merged(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, proj: &AffineProj) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.aux_merged), 0);
        enc.set_buffer(2, Some(&proj.input), 0);
        enc.set_buffer(3, Some(&proj.output), 0);
        set_u32(enc, 4, proj.rows);
        set_u32(enc, 5, proj.cols);
        set_u32(enc, 6, proj.group_size);
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

    fn snapshot_outputs(projs: &[AffineProj]) -> Vec<Vec<f32>> {
        projs
            .iter()
            .map(|p| read_f32(&p.output, p.rows as usize))
            .collect()
    }

    fn zero_outputs(projs: &[AffineProj]) {
        for p in projs {
            zero_f32(&p.output, p.rows as usize);
        }
    }

    fn compare_out(reference: &[f32], variant: &[f32]) -> Value {
        let n = reference.len().min(variant.len());
        let mut num = 0f64;
        let mut den = 0f64;
        let mut max_abs = 0f64;
        let mut n_exact = 0u64;
        for i in 0..n {
            let a = reference[i] as f64;
            let b = variant[i] as f64;
            let d = (a - b).abs();
            if d == 0.0 {
                n_exact += 1;
            }
            if d > max_abs {
                max_abs = d;
            }
            num += d * d;
            den += a * a;
        }
        json!({
            "n_compared": n,
            "n_bit_exact": n_exact,
            "max_abs_err": max_abs,
            "rel_fro": if den > 0.0 { (num / den).sqrt() } else { 0.0 },
            "bit_identical": n_exact as usize == n,
        })
    }

    fn compare_layer(a: &[Vec<f32>], b: &[Vec<f32>], names: &[&str]) -> Value {
        let mut n = 0u64;
        let mut n_exact = 0u64;
        let mut max_abs = 0f64;
        let mut num = 0f64;
        let mut den = 0f64;
        let mut per = Vec::new();
        for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            let row = compare_out(ra, rb);
            n += row["n_compared"].as_u64().unwrap_or(0);
            n_exact += row["n_bit_exact"].as_u64().unwrap_or(0);
            let ma = row["max_abs_err"].as_f64().unwrap_or(0.0);
            if ma > max_abs {
                max_abs = ma;
            }
            for j in 0..ra.len().min(rb.len()) {
                let av = ra[j] as f64;
                let d = (av - rb[j] as f64).abs();
                num += d * d;
                den += av * av;
            }
            let mut obj = row;
            if let Value::Object(map) = &mut obj {
                map.insert("name".into(), json!(names.get(i).copied().unwrap_or("")));
            }
            per.push(obj);
        }
        json!({
            "n_compared": n,
            "n_bit_exact": n_exact,
            "max_abs_err": max_abs,
            "rel_fro": if den > 0.0 { (num / den).sqrt() } else { 0.0 },
            "bit_identical": n_exact == n && n > 0,
            "per_projection": per,
        })
    }

    fn arm_json(
        id: &str,
        kernel: &str,
        bytes_per_stream: &[u64],
        unique_payload_bytes: u64,
        gpu: Vec<u64>,
        occupancy: &Value,
    ) -> Value {
        let med = median_u64(gpu.clone()).unwrap_or(0);
        let gb_s = if med == 0 {
            0.0
        } else {
            unique_payload_bytes as f64 / med as f64
        };
        let (lo, hi) = gpu
            .iter()
            .fold((u64::MAX, 0u64), |(l, h), &n| (l.min(n), h.max(n)));
        let spread = if lo == 0 || lo == u64::MAX {
            0.0
        } else {
            hi as f64 / lo as f64
        };
        let bytes_iter: u64 = bytes_per_stream.iter().sum();
        json!({
            "id": id,
            "kernel": kernel,
            "stream_count": bytes_per_stream.len(),
            "bytes_per_stream": bytes_per_stream,
            "bytes_per_thread_iteration": bytes_iter,
            "unique_payload_bytes": unique_payload_bytes,
            "weight_bytes": unique_payload_bytes,
            "gpu_ns_median": med,
            "gpu_ns_min": if lo == u64::MAX { 0 } else { lo },
            "gpu_ns_max": hi,
            "gpu_ns_reps": gpu,
            "gpu_us_median": med as f64 / 1e3,
            "rep_spread": spread,
            "steady_state": spread <= STEADY_STATE_MAX_SPREAD && !gpu.is_empty(),
            "effective_gb_s": gb_s,
            "dispatches": 3,
            "encoders": 1,
            "command_buffers": 1,
            "threads_per_threadgroup": TG,
            "occupancy": occupancy,
        })
    }

    pub fn run(args: Args) -> Value {
        let concurrent_start = concurrent_load();
        eprintln!(
            "mlp_aux_merge_ab opening {} layer={} warmup={} reps={}",
            args.artifact_root.display(),
            args.layer,
            args.warmup,
            args.reps
        );
        let catalog = parse_catalog(&args.artifact_root).unwrap_or_else(|e| fail(e));
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal-capable GPU"));
        let queue = device.new_command_queue();
        let pipes = compile(&device).unwrap_or_else(|e| fail(e));
        let incumbent = pipes
            .get(INCUMBENT_KERNEL)
            .unwrap_or_else(|| fail(INCUMBENT_KERNEL));
        let merged = pipes
            .get(MERGED_KERNEL)
            .unwrap_or_else(|| fail(MERGED_KERNEL));

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
        let merged_aux_bytes: u64 = mlp.iter().map(|p| p.merged_aux_bytes).sum();
        let scale_bias_bytes: u64 = mlp.iter().map(|p| p.scale_bytes + p.bias_bytes).sum();
        if merged_aux_bytes != scale_bias_bytes {
            fail(format!(
                "merged aux unique bytes {merged_aux_bytes} != scale+bias {scale_bias_bytes}"
            ));
        }
        let occ_a = occupancy(incumbent, mlp[0].rows);
        let occ_b = occupancy(merged, mlp[0].rows);

        let time_a = |n: usize| {
            time_cb(&queue, n, |enc| {
                for p in mlp.iter() {
                    encode_incumbent(enc, incumbent, p);
                }
            })
        };
        let time_b = |n: usize| {
            time_cb(&queue, n, |enc| {
                for p in mlp.iter() {
                    encode_merged(enc, merged, p);
                }
            })
        };

        eprintln!("  interleaved warmup A/B x{}", args.warmup);
        for _ in 0..args.warmup {
            let _ = time_a(1);
            let _ = time_b(1);
        }

        eprintln!("  output compare (one dispatch each, after warmup)");
        zero_outputs(&mlp);
        let _ = time_a(1);
        let out_a = snapshot_outputs(&mlp);
        zero_outputs(&mlp);
        let _ = time_b(1);
        let out_b = snapshot_outputs(&mlp);
        let names = [mlp[0].name.as_str(), mlp[1].name.as_str(), mlp[2].name.as_str()];
        let output_compare = compare_layer(&out_a, &out_b, &names);
        eprintln!(
            "    n_compared={} n_bit_exact={} max_abs_err={} rel_fro={} bit_identical={}",
            output_compare["n_compared"],
            output_compare["n_bit_exact"],
            output_compare["max_abs_err"],
            output_compare["rel_fro"],
            output_compare["bit_identical"]
        );

        eprintln!("  order AB (incumbent then merged) reps={}", args.reps);
        let ab_a = time_a(args.reps);
        let ab_b = time_b(args.reps);
        let order_ab_a = arm_json(
            "incumbent",
            INCUMBENT_KERNEL,
            &[2, 2, 2, 32],
            mlp_bytes,
            ab_a,
            &occ_a,
        );
        let order_ab_b = arm_json(
            "merged",
            MERGED_KERNEL,
            &[2, 4, 32],
            mlp_bytes,
            ab_b,
            &occ_b,
        );

        eprintln!("  order BA (merged then incumbent) reps={}", args.reps);
        let ba_b = time_b(args.reps);
        let ba_a = time_a(args.reps);
        let order_ba_a = arm_json(
            "incumbent",
            INCUMBENT_KERNEL,
            &[2, 2, 2, 32],
            mlp_bytes,
            ba_a,
            &occ_a,
        );
        let order_ba_b = arm_json(
            "merged",
            MERGED_KERNEL,
            &[2, 4, 32],
            mlp_bytes,
            ba_b,
            &occ_b,
        );

        let mut pooled_a = Vec::new();
        let mut pooled_b = Vec::new();
        if let Value::Array(v) = &order_ab_a["gpu_ns_reps"] {
            for x in v {
                if let Some(n) = x.as_u64() {
                    pooled_a.push(n);
                }
            }
        }
        if let Value::Array(v) = &order_ba_a["gpu_ns_reps"] {
            for x in v {
                if let Some(n) = x.as_u64() {
                    pooled_a.push(n);
                }
            }
        }
        if let Value::Array(v) = &order_ab_b["gpu_ns_reps"] {
            for x in v {
                if let Some(n) = x.as_u64() {
                    pooled_b.push(n);
                }
            }
        }
        if let Value::Array(v) = &order_ba_b["gpu_ns_reps"] {
            for x in v {
                if let Some(n) = x.as_u64() {
                    pooled_b.push(n);
                }
            }
        }
        let pooled_incumbent = arm_json(
            "incumbent",
            INCUMBENT_KERNEL,
            &[2, 2, 2, 32],
            mlp_bytes,
            pooled_a,
            &occ_a,
        );
        let pooled_merged = arm_json(
            "merged",
            MERGED_KERNEL,
            &[2, 4, 32],
            mlp_bytes,
            pooled_b,
            &occ_b,
        );

        let ratio = |a: &Value, b: &Value| -> Value {
            let ga = a["effective_gb_s"].as_f64().unwrap_or(0.0);
            let gb = b["effective_gb_s"].as_f64().unwrap_or(0.0);
            if ga <= 0.0 {
                Value::Null
            } else {
                json!(gb / ga)
            }
        };
        let r_ab = ratio(&order_ab_a, &order_ab_b);
        let r_ba = ratio(&order_ba_a, &order_ba_b);
        let r_pooled = ratio(&pooled_incumbent, &pooled_merged);
        eprintln!(
            "    AB incumbent {:.1} GB/s merged {:.1} GB/s ratio {:?}",
            order_ab_a["effective_gb_s"], order_ab_b["effective_gb_s"], r_ab
        );
        eprintln!(
            "    BA incumbent {:.1} GB/s merged {:.1} GB/s ratio {:?}",
            order_ba_a["effective_gb_s"], order_ba_b["effective_gb_s"], r_ba
        );
        eprintln!(
            "    pooled incumbent {:.1} GB/s merged {:.1} GB/s ratio {:?}",
            pooled_incumbent["effective_gb_s"], pooled_merged["effective_gb_s"], r_pooled
        );

        let concurrent_end = concurrent_load();
        json!({
            "schema": "hawking.future.mlp_aux_merge_ab.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "fast_math": false,
            "geometry": "geo_tpr64_tg128",
            "codec": "HGRAVF01 affine2 q2 group64",
            "organ": "mlp",
            "dispatches": 3,
            "absolute_gb_s_are_measured_under_load": true,
            "bytes_per_thread_iteration_held": BYTES_PER_ITER,
            "steady_state_max_spread": STEADY_STATE_MAX_SPREAD,
            "aux_repack_on_disk": false,
            "does_not_promote": true,
            "production_shader_untouched": true,
            "concurrent_load": concurrent_start,
            "concurrent_load_end": concurrent_end,
            "projections": mlp.iter().map(|p| json!({
                "name": p.name,
                "rows": p.rows,
                "cols": p.cols,
                "group_size": p.group_size,
                "weight_bytes": p.weight_bytes,
                "code_bytes": p.code_bytes,
                "scale_bytes": p.scale_bytes,
                "bias_bytes": p.bias_bytes,
                "merged_aux_bytes": p.merged_aux_bytes,
            })).collect::<Vec<_>>(),
            "weight_bytes": mlp_bytes,
            "unique_payload_bytes": mlp_bytes,
            "merged_aux_bytes": merged_aux_bytes,
            "scale_bias_bytes": scale_bias_bytes,
            "output_compare": output_compare,
            "order_ab": {
                "sequence": ["incumbent", "merged"],
                "incumbent": order_ab_a,
                "merged": order_ab_b,
                "ratio_merged_over_incumbent": r_ab,
            },
            "order_ba": {
                "sequence": ["merged", "incumbent"],
                "incumbent": order_ba_a,
                "merged": order_ba_b,
                "ratio_merged_over_incumbent": r_ba,
            },
            "pooled": {
                "incumbent": pooled_incumbent,
                "merged": pooled_merged,
                "ratio_merged_over_incumbent": r_pooled,
            },
        })
    }
}
