//! Back-to-back A/B: incumbent f16-aux affine-Q2 vs native in-register u8 aux.
//!
//! One representative MLP layer of sealed-3.14. Same geometry, same codes,
//! same process. Native kernel binds uchar scale/bias and decodes them in
//! registers (log-scale exp + linear bias). It does not expand u8 back to
//! an f16 aux buffer. Does not mutate the production decode path.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example aux_u8_ab
//! ./tools/gpu_lane_lock.sh y1auxnative \
//!   workspace/ops/build/rust/release-fast/examples/aux_u8_ab \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_AUX_U8_NATIVE_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: aux_u8_ab --artifact-root DIR [--layer N] [--warmup N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("aux_u8_ab: {message}");
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
        "note": "absolute GB/s is measured-under-load; A/B is back-to-back in one process",
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
    fail("aux_u8_ab is Metal-only")
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

    const SHADER: &str = include_str!("aux_u8_ab.metal");
    const TG: u64 = 128;
    const GPU_CORES: u32 = 60;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AuxU8Endpoints {
        scale_lmin: f32,
        scale_span: f32,
        bias_min: f32,
        bias_span: f32,
    }

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
            "aux_u8_ab: compiled shaders in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        let names = [
            "aux_u8_incumbent_affine_q2_geo_tpr64_tg128",
            "aux_u8_native_affine_q2_geo_tpr64_tg128",
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
            "registers_note": "Metal pipeline state does not report register count on this toolchain",
        })
    }

    fn f16bits_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    fn u8_log_encode(values: &[f32]) -> (Vec<u8>, f64, f64) {
        let mut ls = Vec::with_capacity(values.len());
        let mut lmin = f64::INFINITY;
        let mut lmax = f64::NEG_INFINITY;
        for &v in values {
            let x = (v as f64).max(1e-12).ln();
            ls.push(x);
            if x < lmin {
                lmin = x;
            }
            if x > lmax {
                lmax = x;
            }
        }
        if !lmin.is_finite() || !lmax.is_finite() {
            fail("u8 log encode: non-finite scale");
        }
        let mut q = vec![0u8; values.len()];
        if lmax <= lmin {
            return (q, lmin, lmax);
        }
        let span = (lmax - lmin) / 255.0;
        for (i, x) in ls.into_iter().enumerate() {
            let t = ((x - lmin) / span).round().clamp(0.0, 255.0);
            q[i] = t as u8;
        }
        (q, lmin, lmax)
    }

    fn u8_linear_encode(values: &[f32]) -> (Vec<u8>, f64, f64) {
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        let xs: Vec<f64> = values
            .iter()
            .map(|&v| {
                let x = v as f64;
                if x < vmin {
                    vmin = x;
                }
                if x > vmax {
                    vmax = x;
                }
                x
            })
            .collect();
        if !vmin.is_finite() || !vmax.is_finite() {
            fail("u8 linear encode: non-finite bias");
        }
        let mut q = vec![0u8; values.len()];
        if vmax <= vmin {
            return (q, vmin, vmax);
        }
        let span = (vmax - vmin) / 255.0;
        for (i, x) in xs.into_iter().enumerate() {
            let t = ((x - vmin) / span).round().clamp(0.0, 255.0);
            q[i] = t as u8;
        }
        (q, vmin, vmax)
    }

    fn endpoints_from(lmin: f64, lmax: f64, bmin: f64, bmax: f64) -> AuxU8Endpoints {
        let scale_span = if lmax <= lmin {
            0.0
        } else {
            ((lmax - lmin) / 255.0) as f32
        };
        let bias_span = if bmax <= bmin {
            0.0
        } else {
            ((bmax - bmin) / 255.0) as f32
        };
        AuxU8Endpoints {
            scale_lmin: lmin as f32,
            scale_span,
            bias_min: bmin as f32,
            bias_span,
        }
    }

    struct AffineProj {
        name: String,
        rows: u32,
        cols: u32,
        group_size: u32,
        codes: Buffer,
        scales_f16: Buffer,
        biases_f16: Buffer,
        scales_u8: Buffer,
        biases_u8: Buffer,
        endpoints: AuxU8Endpoints,
        input: Buffer,
        output_inc: Buffer,
        output_nat: Buffer,
        code_bytes: u64,
        scale_f16_bytes: u64,
        bias_f16_bytes: u64,
        scale_u8_bytes: u64,
        bias_u8_bytes: u64,
        endpoint_bytes: u64,
        groups: u64,
        scale_lmin: f64,
        scale_lmax: f64,
        bias_min: f64,
        bias_max: f64,
    }

    impl AffineProj {
        fn incumbent_weight_bytes(&self) -> u64 {
            self.code_bytes + self.scale_f16_bytes + self.bias_f16_bytes
        }
        fn native_weight_bytes(&self) -> u64 {
            self.code_bytes + self.scale_u8_bytes + self.bias_u8_bytes
        }
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
        if packed.group_size != 64 {
            return Err(format!(
                "{name} group_size {} != 64",
                packed.group_size
            ));
        }
        let scales_f32: Vec<f32> = packed
            .scales_f16
            .iter()
            .copied()
            .map(f16bits_to_f32)
            .collect();
        let biases_f32: Vec<f32> = packed
            .biases_f16
            .iter()
            .copied()
            .map(f16bits_to_f32)
            .collect();
        let (scale_u8, slmin, slmax) = u8_log_encode(&scales_f32);
        let (bias_u8, bmin, bmax) = u8_linear_encode(&biases_f32);
        if scale_u8.len() != packed.scales_f16.len() || bias_u8.len() != packed.biases_f16.len() {
            return Err(format!("{name} u8 aux length disagrees with f16 aux"));
        }
        let input = fill_f32(packed.cols);
        let output = vec![0f32; packed.rows];
        let code_bytes = packed.codes.len() as u64;
        let scale_f16_bytes = (packed.scales_f16.len() * 2) as u64;
        let bias_f16_bytes = (packed.biases_f16.len() * 2) as u64;
        let scale_u8_bytes = scale_u8.len() as u64;
        let bias_u8_bytes = bias_u8.len() as u64;
        Ok(AffineProj {
            name: name.to_string(),
            rows: packed.rows as u32,
            cols: packed.cols as u32,
            group_size: packed.group_size as u32,
            codes: buf_u8(device, &packed.codes),
            scales_f16: buf_u16(device, &packed.scales_f16),
            biases_f16: buf_u16(device, &packed.biases_f16),
            scales_u8: buf_u8(device, &scale_u8),
            biases_u8: buf_u8(device, &bias_u8),
            endpoints: endpoints_from(slmin, slmax, bmin, bmax),
            input: buf_f32(device, &input),
            output_inc: buf_f32(device, &output),
            output_nat: buf_f32(device, &output),
            code_bytes,
            scale_f16_bytes,
            bias_f16_bytes,
            scale_u8_bytes,
            bias_u8_bytes,
            endpoint_bytes: 16,
            groups: packed.groups as u64,
            scale_lmin: slmin,
            scale_lmax: slmax,
            bias_min: bmin,
            bias_max: bmax,
        })
    }

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(TG, 1, 1));
    }

    fn encode_incumbent(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        proj: &AffineProj,
    ) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales_f16), 0);
        enc.set_buffer(2, Some(&proj.biases_f16), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output_inc), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        set_u32(enc, 7, proj.group_size);
        dispatch_groups(enc, proj.rows);
    }

    fn encode_native(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, proj: &AffineProj) {
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(&proj.codes), 0);
        enc.set_buffer(1, Some(&proj.scales_u8), 0);
        enc.set_buffer(2, Some(&proj.biases_u8), 0);
        enc.set_buffer(3, Some(&proj.input), 0);
        enc.set_buffer(4, Some(&proj.output_nat), 0);
        set_u32(enc, 5, proj.rows);
        set_u32(enc, 6, proj.cols);
        set_u32(enc, 7, proj.group_size);
        enc.set_bytes(
            8,
            std::mem::size_of::<AuxU8Endpoints>() as u64,
            &proj.endpoints as *const AuxU8Endpoints as *const _,
        );
        dispatch_groups(enc, proj.rows);
    }

    fn encode_layer(
        enc: &metal::ComputeCommandEncoderRef,
        pipe: &Pipe,
        projs: &[AffineProj],
        native: bool,
    ) {
        for p in projs {
            if native {
                encode_native(enc, pipe, p);
            } else {
                encode_incumbent(enc, pipe, p);
            }
        }
    }

    fn time_one<F>(queue: &metal::CommandQueue, fill: F) -> Option<u64>
    where
        F: Fn(&metal::ComputeCommandEncoderRef),
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        fill(&enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        gpu_ns(cmd)
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return f64::NAN;
        }
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..a.len() {
            let x = a[i] as f64;
            let y = b[i] as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let den = na.sqrt() * nb.sqrt();
        if den == 0.0 {
            f64::NAN
        } else {
            dot / den
        }
    }

    fn relfro(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return f64::NAN;
        }
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..a.len() {
            let d = a[i] as f64 - b[i] as f64;
            num += d * d;
            den += (b[i] as f64) * (b[i] as f64);
        }
        if den == 0.0 {
            if num == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            num.sqrt() / den.sqrt()
        }
    }

    fn arm_json(
        label: &str,
        kernel: &str,
        weight_bytes: u64,
        gpu: &[u64],
        dispatches: u64,
        extra: Value,
    ) -> Value {
        let med = median_u64(gpu.to_vec()).unwrap_or(0);
        let gb_s = if med == 0 {
            0.0
        } else {
            weight_bytes as f64 / med as f64
        };
        let us = med as f64 / 1e3;
        let mut v = json!({
            "label": label,
            "kernel": kernel,
            "weight_bytes": weight_bytes,
            "gpu_ns_median": med,
            "gpu_us_median": us,
            "gpu_ns_reps": gpu,
            "dispatches": dispatches,
            "encoders": 1,
            "command_buffers": 1,
            "effective_gb_s": gb_s,
        });
        if let Value::Object(extra) = extra {
            if let Value::Object(obj) = &mut v {
                obj.extend(extra);
            }
        }
        v
    }

    pub fn run(args: Args) -> Value {
        let concurrent_start = concurrent_load();
        eprintln!(
            "aux_u8_ab opening {} layer={} warmup={} reps={}",
            args.artifact_root.display(),
            args.layer,
            args.warmup,
            args.reps
        );
        let catalog = parse_catalog(&args.artifact_root).unwrap_or_else(|e| fail(e));
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal-capable GPU"));
        let queue = device.new_command_queue();
        let pipes = compile(&device).unwrap_or_else(|e| fail(e));
        let inc_pipe = pipes
            .get("aux_u8_incumbent_affine_q2_geo_tpr64_tg128")
            .unwrap();
        let nat_pipe = pipes
            .get("aux_u8_native_affine_q2_geo_tpr64_tg128")
            .unwrap();

        let names = [
            qwen38_layer_name(args.layer, "mlp.gate_proj.weight"),
            qwen38_layer_name(args.layer, "mlp.up_proj.weight"),
            qwen38_layer_name(args.layer, "mlp.down_proj.weight"),
        ];
        let mut mlp: Vec<AffineProj> = Vec::with_capacity(3);
        for name in &names {
            eprintln!("  loading {name}");
            mlp.push(load_affine(&device, &catalog, name).unwrap_or_else(|e| fail(e)));
        }

        let inc_bytes: u64 = mlp.iter().map(|p| p.incumbent_weight_bytes()).sum();
        let nat_bytes: u64 = mlp.iter().map(|p| p.native_weight_bytes()).sum();
        let endpoint_bytes: u64 = mlp.iter().map(|p| p.endpoint_bytes).sum();
        let occ = occupancy(inc_pipe, mlp[0].rows);

        eprintln!("  warmup incumbent/native back-to-back");
        for i in 0..args.warmup {
            let _ = time_one(&queue, |enc| encode_layer(enc, inc_pipe, &mlp, false));
            let _ = time_one(&queue, |enc| encode_layer(enc, nat_pipe, &mlp, true));
            eprintln!("    warmup{i}");
        }

        eprintln!("  reps interleaved (incumbent, native) same process");
        let mut inc_ns = Vec::with_capacity(args.reps);
        let mut nat_ns = Vec::with_capacity(args.reps);
        let mut pairs = Vec::with_capacity(args.reps);
        for i in 0..args.reps {
            let a = time_one(&queue, |enc| encode_layer(enc, inc_pipe, &mlp, false))
                .unwrap_or_else(|| fail(format!("rep {i} incumbent: no GPUEndTime-GPUStartTime")));
            let b = time_one(&queue, |enc| encode_layer(enc, nat_pipe, &mlp, true))
                .unwrap_or_else(|| fail(format!("rep {i} native: no GPUEndTime-GPUStartTime")));
            eprintln!("    rep{i} incumbent_ns={a} native_ns={b}");
            inc_ns.push(a);
            nat_ns.push(b);
            pairs.push(json!({
                "incumbent_gpu_ns": a,
                "native_gpu_ns": b,
                "delta_gpu_ns": (b as i64) - (a as i64),
            }));
        }

        // Output cosine after the last pair (both outputs live).
        let mut proj_out = Vec::new();
        let mut cosines = Vec::new();
        for p in &mlp {
            let yi = read_f32(&p.output_inc, p.rows as usize);
            let yn = read_f32(&p.output_nat, p.rows as usize);
            let c = cosine(&yn, &yi);
            let r = relfro(&yn, &yi);
            cosines.push(c);
            proj_out.push(json!({
                "name": p.name,
                "rows": p.rows,
                "cols": p.cols,
                "group_size": p.group_size,
                "groups": p.groups,
                "code_bytes": p.code_bytes,
                "incumbent_aux_bytes": p.scale_f16_bytes + p.bias_f16_bytes,
                "native_aux_bytes": p.scale_u8_bytes + p.bias_u8_bytes,
                "endpoint_bytes": p.endpoint_bytes,
                "incumbent_weight_bytes": p.incumbent_weight_bytes(),
                "native_weight_bytes": p.native_weight_bytes(),
                "scale_lmin": p.scale_lmin,
                "scale_lmax": p.scale_lmax,
                "bias_min": p.bias_min,
                "bias_max": p.bias_max,
                "output_cosine_vs_incumbent": c,
                "output_relfro_vs_incumbent": r,
                "native_binds": "device const uchar* scales_u8 / biases_u8",
                "native_materializes_f16_aux": false,
            }));
        }
        let mean_cos = if cosines.iter().any(|c| !c.is_finite()) {
            f64::NAN
        } else {
            cosines.iter().sum::<f64>() / cosines.len() as f64
        };

        let concurrent_end = concurrent_load();
        json!({
            "schema": "hawking.future.aux_u8_native.raw.v1",
            "example": "aux_u8_ab",
            "layer": args.layer,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "absolute_gb_s_are_measured_under_load": true,
            "artifact_root": args.artifact_root.display().to_string(),
            "git_head": git_head(),
            "concurrent_load": concurrent_start,
            "concurrent_load_end": concurrent_end,
            "occupancy": occ,
            "native_consumer": {
                "materializes_f16_aux": false,
                "scale_buffer": "u8",
                "bias_buffer": "u8",
                "decode": "in-register log-scale exp + linear bias",
                "codes": "incumbent 2-bit kept",
                "kernel": nat_pipe.name,
                "forbidden_shape": "expand u8 aux to f16 then feed the ordinary kernel",
            },
            "incumbent": arm_json(
                "incumbent_f16_aux",
                inc_pipe.name,
                inc_bytes,
                &inc_ns,
                3,
                json!({
                    "aux_dtype": "f16 scale + f16 bias",
                    "code_bytes": mlp.iter().map(|p| p.code_bytes).sum::<u64>(),
                    "aux_bytes": mlp.iter().map(|p| p.scale_f16_bytes + p.bias_f16_bytes).sum::<u64>(),
                }),
            ),
            "native_u8": arm_json(
                "native_u8_aux",
                nat_pipe.name,
                nat_bytes,
                &nat_ns,
                3,
                json!({
                    "aux_dtype": "u8 log-scale + u8 linear bias",
                    "code_bytes": mlp.iter().map(|p| p.code_bytes).sum::<u64>(),
                    "aux_bytes": mlp.iter().map(|p| p.scale_u8_bytes + p.bias_u8_bytes).sum::<u64>(),
                    "endpoint_bytes": endpoint_bytes,
                    "endpoint_note": "4 f32 constants per tensor, not a streamed aux array",
                    "materializes_f16_aux": false,
                }),
            ),
            "paired_reps": pairs,
            "projections": proj_out,
            "output_cosine_mean": mean_cos,
            "bytes_removed_this_layer": inc_bytes as i64 - nat_bytes as i64,
            "shader": "crates/hawking-core/examples/aux_u8_ab.metal",
        })
    }
}
