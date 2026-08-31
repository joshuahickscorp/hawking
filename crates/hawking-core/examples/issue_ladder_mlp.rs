//! Monotone total-ops/byte ladder + C1/C2 discriminators for one MLP layer.
//!
//! Same layer, same buffers, same geometry, same GPU timestamps as
//! decode_cheapen_mlp.rs and alu_roofline_organs.rs. Not a third measurement
//! architecture. Production decode shaders are not modified.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example issue_ladder_mlp
//! ./tools/gpu_lane_lock.sh z1b \
//!   workspace/ops/build/rust/release-fast/examples/issue_ladder_mlp \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_MLP_ISSUE_RATE_LADDER_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: issue_ladder_mlp --artifact-root DIR \
        [--layer N] [--warmup N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("issue_ladder_mlp: {message}");
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
        "note": "absolute GB/s is measured-under-load; ratio to production in this process is the robust number",
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
    fail("issue_ladder_mlp is Metal-only")
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

    const SHADER: &str = include_str!("issue_ladder_mlp.metal");
    const TG: u64 = 128;
    const GPU_CORES: u32 = 60;
    const PRODUCTION_KERNEL: &str = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128";

    struct VariantMeta {
        id: &'static str,
        kernel: &'static str,
        family: &'static str,
        n_accumulators: u32,
        n_live_floats: u32,
        threads_per_threadgroup: u64,
        keep_k: Option<u32>,
        note: &'static str,
    }

    const LADDER: &[VariantMeta] = &[
        VariantMeta {
            id: "production",
            kernel: PRODUCTION_KERNEL,
            family: "ladder",
            n_accumulators: 1,
            n_live_floats: 1,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production geo_tpr64 body; 8 dequant FMA + 8 MAC FMA per 6 B",
        },
        VariantMeta {
            id: "k6",
            kernel: "issue_ladder_k6",
            family: "ladder",
            n_accumulators: 1,
            n_live_floats: 1,
            threads_per_threadgroup: TG,
            keep_k: Some(6),
            note: "6 production slots + 2 xor-sink x loads",
        },
        VariantMeta {
            id: "k4",
            kernel: "issue_ladder_k4",
            family: "ladder",
            n_accumulators: 1,
            n_live_floats: 1,
            threads_per_threadgroup: TG,
            keep_k: Some(4),
            note: "4 production slots + 4 xor-sink x loads",
        },
        VariantMeta {
            id: "k2",
            kernel: "issue_ladder_k2",
            family: "ladder",
            n_accumulators: 1,
            n_live_floats: 1,
            threads_per_threadgroup: TG,
            keep_k: Some(2),
            note: "2 production slots + 6 xor-sink x loads",
        },
        VariantMeta {
            id: "arm_a",
            kernel: "issue_ladder_arm_a",
            family: "ladder",
            n_accumulators: 1,
            n_live_floats: 1,
            threads_per_threadgroup: TG,
            keep_k: Some(0),
            note: "ARM A: XOR/add sink of the same loads; ~0 FMA",
        },
    ];

    const ILP: &[VariantMeta] = &[
        VariantMeta {
            id: "ilp2",
            kernel: "issue_ladder_ilp2",
            family: "ilp",
            n_accumulators: 2,
            n_live_floats: 2,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production 16 FMA split into 2 independent acc chains",
        },
        VariantMeta {
            id: "ilp4",
            kernel: "issue_ladder_ilp4",
            family: "ilp",
            n_accumulators: 4,
            n_live_floats: 4,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production 16 FMA split into 4 independent acc chains",
        },
        VariantMeta {
            id: "ilp8",
            kernel: "issue_ladder_ilp8",
            family: "ilp",
            n_accumulators: 8,
            n_live_floats: 8,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production 16 FMA split into 8 independent acc chains",
        },
    ];

    const WS: &[VariantMeta] = &[
        VariantMeta {
            id: "ws8",
            kernel: "issue_ladder_ws8",
            family: "register_pressure",
            n_accumulators: 1,
            n_live_floats: 8,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production arith + 8 live floats rotated through the tile loop",
        },
        VariantMeta {
            id: "ws16",
            kernel: "issue_ladder_ws16",
            family: "register_pressure",
            n_accumulators: 1,
            n_live_floats: 16,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production arith + 16 live floats rotated through the tile loop",
        },
        VariantMeta {
            id: "ws32",
            kernel: "issue_ladder_ws32",
            family: "register_pressure",
            n_accumulators: 1,
            n_live_floats: 32,
            threads_per_threadgroup: TG,
            keep_k: Some(8),
            note: "production arith + 32 live floats rotated through the tile loop",
        },
    ];

    const TG_SWEEP: &[u64] = &[64, 128, 256, 512, 1024];

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
            "issue_ladder_mlp: compiled shaders in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        let mut names: Vec<&'static str> = Vec::new();
        names.push(PRODUCTION_KERNEL);
        for v in LADDER.iter().chain(ILP.iter()).chain(WS.iter()) {
            if !names.contains(&v.kernel) {
                names.push(v.kernel);
            }
        }
        names.push("issue_ladder_tg");
        let mut out = HashMap::new();
        for name in names {
            if out.contains_key(name) {
                continue;
            }
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

    fn occupancy(pipe: &Pipe, rows: u32, tg: u64) -> Value {
        let threadgroups = u64::from(rows.div_ceil(2).max(1));
        json!({
            "threads_per_threadgroup": tg,
            "max_total_threads_per_threadgroup": pipe.max_threads,
            "thread_execution_width": pipe.exec_width,
            "occupancy_of_max_threads": if pipe.max_threads == 0 {
                Value::Null
            } else {
                json!(tg as f64 / pipe.max_threads as f64)
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

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32, tg: u64) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }

    fn encode_affine(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        proj: &AffineProj,
        tg: u64,
        pass_tg: bool,
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
        if pass_tg {
            set_u32(enc, 8, tg as u32);
        }
        dispatch_groups(enc, proj.rows, tg);
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

    fn fnv1a_bits(vs: &[Vec<f32>]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for v in vs {
            for x in v {
                h ^= u64::from(x.to_bits());
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    fn byte_compare(prod: &[Vec<f32>], var: &[Vec<f32>]) -> Value {
        if prod.len() != var.len() {
            return json!({
                "n_bytes_compared": 0,
                "n_mismatch_bytes": 0,
                "n_float_mismatch": 0,
                "error": "projection count mismatch",
            });
        }
        let mut n_floats = 0u64;
        let mut n_float_mismatch = 0u64;
        let mut n_mismatch_bytes = 0u64;
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut diff2 = 0.0f64;
        let mut first_mismatch: Option<u64> = None;
        for (a_row, b_row) in prod.iter().zip(var.iter()) {
            if a_row.len() != b_row.len() {
                return json!({
                    "n_bytes_compared": 0,
                    "n_mismatch_bytes": 0,
                    "n_float_mismatch": 0,
                    "error": "row length mismatch",
                });
            }
            for (a, b) in a_row.iter().zip(b_row.iter()) {
                let idx = n_floats;
                n_floats += 1;
                let ab = a.to_bits().to_le_bytes();
                let bb = b.to_bits().to_le_bytes();
                for i in 0..4 {
                    if ab[i] != bb[i] {
                        n_mismatch_bytes += 1;
                    }
                }
                if a.to_bits() != b.to_bits() {
                    n_float_mismatch += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some(idx);
                    }
                }
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let af = f64::from(*a);
                let bf = f64::from(*b);
                dot += af * bf;
                na += af * af;
                nb += bf * bf;
                let dd = af - bf;
                diff2 += dd * dd;
            }
        }
        let cosine = if na > 0.0 && nb > 0.0 {
            dot / (na.sqrt() * nb.sqrt())
        } else {
            0.0
        };
        let rel_fro = if na > 0.0 { diff2.sqrt() / na.sqrt() } else { 0.0 };
        json!({
            "n_bytes_compared": n_floats * 4,
            "n_floats_compared": n_floats,
            "n_mismatch_bytes": n_mismatch_bytes,
            "n_float_mismatch": n_float_mismatch,
            "first_mismatch_index": first_mismatch,
            "max_abs_err": max_abs,
            "cosine": cosine,
            "rel_fro": rel_fro,
            "production_fnv1a64": format!("{:016x}", fnv1a_bits(prod)),
            "variant_fnv1a64": format!("{:016x}", fnv1a_bits(var)),
            "compared_against": "production kernel output buffers after the timed command buffer",
        })
    }

    fn variant_json(
        meta: &VariantMeta,
        gpu: Vec<u64>,
        mlp_bytes: u64,
        compare: Value,
        occ: &Value,
    ) -> Value {
        let med = median_u64(gpu.clone()).unwrap_or(0);
        let gb_s = if med == 0 {
            0.0
        } else {
            mlp_bytes as f64 / med as f64
        };
        let us = med as f64 / 1e3;
        eprintln!(
            "    {:>7.1} GB/s  {:>7.1} us  bytes={}",
            gb_s, us, mlp_bytes
        );
        json!({
            "id": meta.id,
            "kernel": meta.kernel,
            "family": meta.family,
            "note": meta.note,
            "keep_k": meta.keep_k,
            "n_accumulators": meta.n_accumulators,
            "n_live_floats": meta.n_live_floats,
            "weight_bytes": mlp_bytes,
            "gpu_ns_median": med,
            "gpu_ns_reps": gpu,
            "gpu_us_median": us,
            "effective_gb_s": gb_s,
            "dispatches": 3,
            "encoders": 1,
            "command_buffers": 1,
            "threads_per_threadgroup": meta.threads_per_threadgroup,
            "occupancy": occ,
            "byte_compare": compare,
        })
    }

    pub fn run(args: Args) -> Value {
        let concurrent_start = concurrent_load();
        eprintln!(
            "issue_ladder_mlp opening {} layer={} warmup={} reps={}",
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
        let mlp_rows_gate = mlp[0].rows;
        let mlp_cols = mlp[0].cols;

        let time_mlp = |n: usize, pipe: &Pipe, tg: u64, pass_tg: bool| {
            time_cb(&queue, n, |enc| {
                for p in mlp.iter() {
                    encode_affine(enc, pipe, p, tg, pass_tg);
                }
            })
        };

        let mut production_out: Option<Vec<Vec<f32>>> = None;
        let mut production_ns: Option<u64> = None;

        let measure = |meta: &VariantMeta,
                       production_out: &mut Option<Vec<Vec<f32>>>,
                       production_ns: &mut Option<u64>|
         -> Value {
            let pipe = pipes.get(meta.kernel).unwrap_or_else(|| fail(meta.kernel));
            let occ = occupancy(pipe, mlp_rows_gate, meta.threads_per_threadgroup);
            eprintln!("  {} ({})", meta.id, meta.kernel);
            let _ = time_mlp(args.warmup, pipe, meta.threads_per_threadgroup, false);
            let gpu = time_mlp(args.reps, pipe, meta.threads_per_threadgroup, false);
            let outs = snapshot_outputs(&mlp);
            let compare = if meta.id == "production" {
                *production_out = Some(outs);
                *production_ns = Some(median_u64(gpu.clone()).unwrap_or(0));
                json!({
                    "n_bytes_compared": mlp.iter().map(|p| p.rows as u64 * 4).sum::<u64>(),
                    "n_floats_compared": mlp.iter().map(|p| p.rows as u64).sum::<u64>(),
                    "n_mismatch_bytes": 0,
                    "n_float_mismatch": 0,
                    "first_mismatch_index": Value::Null,
                    "max_abs_err": 0.0,
                    "cosine": 1.0,
                    "rel_fro": 0.0,
                    "production_fnv1a64": format!("{:016x}", fnv1a_bits(production_out.as_ref().unwrap())),
                    "variant_fnv1a64": format!("{:016x}", fnv1a_bits(production_out.as_ref().unwrap())),
                    "compared_against": "self (production control)",
                })
            } else {
                let prod = production_out
                    .as_ref()
                    .unwrap_or_else(|| fail("production must run first"));
                byte_compare(prod, &outs)
            };
            variant_json(meta, gpu, mlp_bytes, compare, &occ)
        };

        eprintln!("  LADDER back-to-back");
        let mut ladder_json = Vec::new();
        for meta in LADDER {
            ladder_json.push(measure(meta, &mut production_out, &mut production_ns));
        }

        eprintln!("  ILP discriminator back-to-back");
        let mut ilp_json = Vec::new();
        // production is chains=1 at constant op count; measured already.
        for meta in ILP {
            ilp_json.push(measure(meta, &mut production_out, &mut production_ns));
        }

        eprintln!("  REGISTER-PRESSURE discriminator back-to-back");
        let mut ws_json = Vec::new();
        for meta in WS {
            ws_json.push(measure(meta, &mut production_out, &mut production_ns));
        }

        eprintln!("  TG sweep back-to-back");
        let tg_pipe = pipes
            .get("issue_ladder_tg")
            .unwrap_or_else(|| fail("issue_ladder_tg"));
        let mut tg_json = Vec::new();
        let prod_ref = production_out
            .as_ref()
            .unwrap_or_else(|| fail("production must run first"))
            .clone();
        for &tg in TG_SWEEP {
            if tg > tg_pipe.max_threads {
                eprintln!(
                    "  tg{} skipped (pipeline max_threads={})",
                    tg, tg_pipe.max_threads
                );
                continue;
            }
            let occ = occupancy(tg_pipe, mlp_rows_gate, tg);
            eprintln!("  tg{tg} (issue_ladder_tg)");
            let _ = time_mlp(args.warmup, tg_pipe, tg, true);
            let gpu = time_mlp(args.reps, tg_pipe, tg, true);
            let outs = snapshot_outputs(&mlp);
            let compare = byte_compare(&prod_ref, &outs);
            let med = median_u64(gpu.clone()).unwrap_or(0);
            let gb_s = if med == 0 {
                0.0
            } else {
                mlp_bytes as f64 / med as f64
            };
            let us = med as f64 / 1e3;
            eprintln!(
                "    {:>7.1} GB/s  {:>7.1} us  bytes={} tg={}",
                gb_s, us, mlp_bytes, tg
            );
            tg_json.push(json!({
                "id": format!("tg{tg}"),
                "kernel": "issue_ladder_tg",
                "family": "threadgroup",
                "note": "production arith at varying threads per threadgroup; 2 rows/TG",
                "keep_k": 8,
                "n_accumulators": 1,
                "n_live_floats": 1,
                "weight_bytes": mlp_bytes,
                "gpu_ns_median": med,
                "gpu_ns_reps": gpu,
                "gpu_us_median": us,
                "effective_gb_s": gb_s,
                "dispatches": 3,
                "encoders": 1,
                "command_buffers": 1,
                "threads_per_threadgroup": tg,
                "occupancy": occ,
                "byte_compare": compare,
            }));
        }

        let concurrent_end = concurrent_load();

        json!({
            "schema": "hawking.future.mlp_issue_rate_ladder.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "concurrent_load": concurrent_start,
            "concurrent_load_end": concurrent_end,
            "absolute_gb_s_are_measured_under_load": true,
            "fast_math": false,
            "organ": "mlp",
            "codec": "HGRAVF01 affine2 q2 group64",
            "geometry": "geo_tpr64_tg128",
            "production_kernel": PRODUCTION_KERNEL,
            "production_gpu_ns_median": production_ns,
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
            "inner_loop_trips_gate_up": mlp_cols / 512,
            "bytes_per_thread_iteration": 2 + 2 + 2 + 32,
            "ladder": ladder_json,
            "ilp": ilp_json,
            "register_pressure": ws_json,
            "threadgroup": tg_json,
        })
    }
}
