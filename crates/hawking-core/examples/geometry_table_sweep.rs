//! G025: geometry is a TABLE the compiler consults, not one hard-coded winner.
//!
//! Sweeps threadgroup size and rows-per-threadgroup across the sealed-3.14
//! resident's REAL HOT DIMENSIONS (MLP gate/up/down, DeltaNet in_proj/out_proj,
//! GQA, LM head). Stream packing is only production 2+2+2+32 vs mid_2_4_32 on
//! affine2 organs — merging further is REFUTED. Accumulator-chain and
//! working-set discriminators are REFUTED and are not re-run.
//!
//! Production shaders are not modified. Diagnostic kernels live in this file.
//!
//! ```text
//! CARGO_TARGET_DIR=workspace/ops/build/rust cargo build --profile release-fast \
//!   -p hawking-core --example geometry_table_sweep
//! ./tools/gpu_lane_lock.sh g025-geometry-table \
//!   workspace/ops/build/rust/release-fast/examples/geometry_table_sweep \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/future/_GEOMETRY_TABLE_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> &'static str {
    "usage: geometry_table_sweep --artifact-root DIR \
        [--warmup N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("geometry_table_sweep: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    warmup: usize,
    reps: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut warmup = 3usize;
    let mut reps = 7usize;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
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

fn iso8601_now() -> String {
    let raw = cmd_stdout(&["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]);
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "UNKNOWN".to_string()
    } else {
        trimmed.to_string()
    }
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
        "note": "absolute GB/s is measured-under-load; back-to-back ratios in this process are the robust number",
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
    fail("geometry_table_sweep is Metal-only")
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
    use hawking_core::model::qwen38_geometry::{
        qwen38_layer_name, qwen38_lm_head_name, QWEN38_BA_ROWS, QWEN38_HIDDEN, QWEN38_INTERMEDIATE,
        QWEN38_KV_PROJ_ROWS, QWEN38_O_PROJ_COLS, QWEN38_O_PROJ_ROWS, QWEN38_QKVZ_ROWS,
        QWEN38_Q_PROJ_ROWS, QWEN38_VOCAB,
    };
    use hawking_core::model::qwen38_hybrid_decode::{
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

    // Diagnostic kernels only. Production shaders are not included and not edited.
    // Launch geometry is parameterized. Packing is only 2+2+2+32 vs mid_2_4_32.
    // pack_6_32 / pack_38 / ILP / working-set are intentionally absent.
    const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float affine_q2_unpack8(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        const float w = float(q) * scale + bias;
        sum += w * x[col + i];
    }
    return sum;
}

static inline float q4_unpack8(
    uint packed, float scale, device const float* x, uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        sum += float(int(byte & 0x0fu) - 8) * scale * x[col + 2u * i];
        sum += float(int(byte >> 4u) - 8) * scale * x[col + 2u * i + 1u];
    }
    return sum;
}

static inline uint xor8(device const float* xp) {
    return as_type<uint>(xp[0]) ^ as_type<uint>(xp[1])
         ^ as_type<uint>(xp[2]) ^ as_type<uint>(xp[3])
         ^ as_type<uint>(xp[4]) ^ as_type<uint>(xp[5])
         ^ as_type<uint>(xp[6]) ^ as_type<uint>(xp[7]);
}

kernel void geo_table_affine2(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant uint& tg_threads       [[buffer(8)]],
    constant uint& rows_per_tg      [[buffer(9)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[32];
    if (rows_per_tg == 0u || (tg_threads % 32u) != 0u || (tg_threads % rows_per_tg) != 0u) {
        return;
    }
    const uint threads_per_row = tg_threads / rows_per_tg;
    if (threads_per_row < 32u || (threads_per_row % 32u) != 0u) {
        return;
    }
    const uint simd_per_row = threads_per_row / 32u;
    const uint team = simd_id / simd_per_row;
    const uint split = simd_id % simd_per_row;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint stride = threads_per_row * 8u;
    const uint row = group_id * rows_per_tg + team;
    float acc = 0.0f;
    if (row < rows && group_size == 64u && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += stride) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const float scale = float(scales[rgb]);
            const float bias = float(biases[rgb]);
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            acc += affine_q2_unpack8(packed16, scale, bias, input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        float s = 0.0f;
        for (uint i = 0u; i < simd_per_row; ++i) {
            s += red[team * simd_per_row + i];
        }
        output[row] = s;
    }
}

kernel void geo_table_q4(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    constant uint& tg_threads       [[buffer(7)]],
    constant uint& rows_per_tg      [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[32];
    if (rows_per_tg == 0u || (tg_threads % 32u) != 0u || (tg_threads % rows_per_tg) != 0u) {
        return;
    }
    const uint threads_per_row = tg_threads / rows_per_tg;
    if (threads_per_row < 32u || (threads_per_row % 32u) != 0u) {
        return;
    }
    const uint simd_per_row = threads_per_row / 32u;
    const uint team = simd_id / simd_per_row;
    const uint split = simd_id % simd_per_row;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint stride = threads_per_row * 8u;
    const uint row = group_id * rows_per_tg + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint rgb0 = row * groups_per_row;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += stride) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = rgb0 + group;
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * 32u + (local >> 1u)));
            acc += q4_unpack8(packed, scale, input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        float s = 0.0f;
        for (uint i = 0u; i < simd_per_row; ++i) {
            s += red[team * simd_per_row + i];
        }
        output[row] = s;
    }
}

// Packing arms at production occupancy (tg=128, 2 rows). Arithmetic stripped
// so this isolates stream addressing. mid_2_4_32 is the MLP_STREAM_COUNT peak;
// pack_6_32 / pack_38 are not present.

kernel void geo_table_pack_22232(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint row = group_id * 2u + (simd_id / 2u);
    const uint lane = (simd_id % 2u) * 32u + simd_lane;
    float acc = 0.0f;
    if (row < rows) {
        const uint gpr = cols >> 6u;
        uint csink = 0u;
        uint xsink = 0u;
        float ssink = 0.0f;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * gpr + group;
            csink ^= uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            ssink += float(scales[rgb]) + float(biases[rgb]);
            xsink ^= xor8(input + col);
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((simd_id % 2u) == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[(simd_id / 2u) * 2u] + red[(simd_id / 2u) * 2u + 1u];
    }
}

kernel void geo_table_pack_mid2432(
    device const uchar* codes       [[buffer(0)]],
    device const uint*  packed_sb   [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint row = group_id * 2u + (simd_id / 2u);
    const uint lane = (simd_id % 2u) * 32u + simd_lane;
    float acc = 0.0f;
    if (row < rows) {
        const uint gpr = cols >> 6u;
        uint csink = 0u;
        uint xsink = 0u;
        uint ssink = 0u;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * gpr + group;
            csink ^= uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            ssink ^= packed_sb[rgb];
            xsink ^= xor8(input + col);
        }
        acc = float(csink) + float(ssink) + as_type<float>(xsink);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((simd_id % 2u) == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[(simd_id / 2u) * 2u] + red[(simd_id / 2u) * 2u + 1u];
    }
}
"#;

    // GPU core count is CITED from qwen38_geometry.rs / KERNEL_GEOMETRY, not measured here.
    const GPU_CORES_CITED: u32 = 60;
    const AFFINE_KERNEL: &str = "geo_table_affine2";
    const Q4_KERNEL: &str = "geo_table_q4";
    const PACK_22232: &str = "geo_table_pack_22232";
    const PACK_MID: &str = "geo_table_pack_mid2432";

    struct LaunchGeo {
        id: &'static str,
        tg: u64,
        rows_per_tg: u32,
    }

    const LAUNCH_GEOS: &[LaunchGeo] = &[
        // Production occupancy first so later geos can compare outputs against it.
        LaunchGeo { id: "tg128_r2", tg: 128, rows_per_tg: 2 },
        LaunchGeo { id: "tg64_r1", tg: 64, rows_per_tg: 1 },
        LaunchGeo { id: "tg64_r2", tg: 64, rows_per_tg: 2 },
        LaunchGeo { id: "tg128_r1", tg: 128, rows_per_tg: 1 },
        LaunchGeo { id: "tg256_r2", tg: 256, rows_per_tg: 2 },
        LaunchGeo { id: "tg256_r4", tg: 256, rows_per_tg: 4 },
        LaunchGeo { id: "tg512_r2", tg: 512, rows_per_tg: 2 },
        LaunchGeo { id: "tg1024_r2", tg: 1024, rows_per_tg: 2 },
    ];

    struct Hot {
        organ: &'static str,
        family: &'static str,
        dtype: &'static str,
        expected_rows: usize,
        expected_cols: usize,
        layer: usize,
        suffix: Option<&'static str>,
        affine: bool,
    }

    fn hots() -> [Hot; 11] {
        [
            Hot {
                organ: "mlp.gate_proj",
                family: "mlp",
                dtype: "affine2_q2",
                expected_rows: QWEN38_INTERMEDIATE,
                expected_cols: QWEN38_HIDDEN,
                layer: 0,
                suffix: Some("mlp.gate_proj.weight"),
                affine: true,
            },
            Hot {
                organ: "mlp.up_proj",
                family: "mlp",
                dtype: "affine2_q2",
                expected_rows: QWEN38_INTERMEDIATE,
                expected_cols: QWEN38_HIDDEN,
                layer: 0,
                suffix: Some("mlp.up_proj.weight"),
                affine: true,
            },
            Hot {
                organ: "mlp.down_proj",
                family: "mlp",
                dtype: "affine2_q2",
                expected_rows: QWEN38_HIDDEN,
                expected_cols: QWEN38_INTERMEDIATE,
                layer: 0,
                suffix: Some("mlp.down_proj.weight"),
                affine: true,
            },
            Hot {
                organ: "linear_attn.in_proj_qkvz",
                family: "deltanet",
                dtype: "uniform_q4",
                expected_rows: QWEN38_QKVZ_ROWS,
                expected_cols: QWEN38_HIDDEN,
                layer: 0,
                suffix: Some("linear_attn.in_proj_qkvz.weight"),
                affine: false,
            },
            Hot {
                organ: "linear_attn.in_proj_ba",
                family: "deltanet",
                dtype: "uniform_q4",
                expected_rows: QWEN38_BA_ROWS,
                expected_cols: QWEN38_HIDDEN,
                layer: 0,
                suffix: Some("linear_attn.in_proj_ba.weight"),
                affine: false,
            },
            Hot {
                organ: "linear_attn.out_proj",
                family: "deltanet",
                dtype: "uniform_q4",
                expected_rows: QWEN38_O_PROJ_ROWS,
                expected_cols: QWEN38_O_PROJ_COLS,
                layer: 0,
                suffix: Some("linear_attn.out_proj.weight"),
                affine: false,
            },
            Hot {
                organ: "self_attn.q_proj",
                family: "gqa",
                dtype: "uniform_q4",
                expected_rows: QWEN38_Q_PROJ_ROWS,
                expected_cols: QWEN38_HIDDEN,
                layer: 3,
                suffix: Some("self_attn.q_proj.weight"),
                affine: false,
            },
            Hot {
                organ: "self_attn.k_proj",
                family: "gqa",
                dtype: "uniform_q4",
                expected_rows: QWEN38_KV_PROJ_ROWS,
                expected_cols: QWEN38_HIDDEN,
                layer: 3,
                suffix: Some("self_attn.k_proj.weight"),
                affine: false,
            },
            Hot {
                organ: "self_attn.v_proj",
                family: "gqa",
                dtype: "uniform_q4",
                expected_rows: QWEN38_KV_PROJ_ROWS,
                expected_cols: QWEN38_HIDDEN,
                layer: 3,
                suffix: Some("self_attn.v_proj.weight"),
                affine: false,
            },
            Hot {
                organ: "self_attn.o_proj",
                family: "gqa",
                dtype: "uniform_q4",
                expected_rows: QWEN38_O_PROJ_ROWS,
                expected_cols: QWEN38_O_PROJ_COLS,
                layer: 3,
                suffix: Some("self_attn.o_proj.weight"),
                affine: false,
            },
            Hot {
                organ: "lm_head",
                family: "lm_head",
                dtype: "uniform_q4",
                expected_rows: QWEN38_VOCAB,
                expected_cols: QWEN38_HIDDEN,
                layer: 0,
                suffix: None,
                affine: false,
            },
        ]
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
            "geometry_table_sweep: compiled diagnostic shaders in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        let mut out = HashMap::new();
        for name in [AFFINE_KERNEL, Q4_KERNEL, PACK_22232, PACK_MID] {
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

    enum Tensor {
        Affine {
            name: String,
            rows: u32,
            cols: u32,
            group_size: u32,
            codes: Buffer,
            scales: Buffer,
            biases: Buffer,
            packed_sb: Buffer,
            input: Buffer,
            output: Buffer,
            weight_bytes: u64,
        },
        Q4 {
            name: String,
            rows: u32,
            cols: u32,
            groups_per_row: u32,
            codes: Buffer,
            scales: Buffer,
            input: Buffer,
            output: Buffer,
            weight_bytes: u64,
        },
    }

    impl Tensor {
        fn rows(&self) -> u32 {
            match self {
                Self::Affine { rows, .. } | Self::Q4 { rows, .. } => *rows,
            }
        }
        fn cols(&self) -> u32 {
            match self {
                Self::Affine { cols, .. } | Self::Q4 { cols, .. } => *cols,
            }
        }
        fn weight_bytes(&self) -> u64 {
            match self {
                Self::Affine { weight_bytes, .. } | Self::Q4 { weight_bytes, .. } => *weight_bytes,
            }
        }
        fn name(&self) -> &str {
            match self {
                Self::Affine { name, .. } | Self::Q4 { name, .. } => name,
            }
        }
        fn output(&self) -> &Buffer {
            match self {
                Self::Affine { output, .. } | Self::Q4 { output, .. } => output,
            }
        }
    }

    fn load_affine(
        device: &Device,
        catalog: &HashMap<String, CatalogRow>,
        name: &str,
    ) -> Result<Tensor, String> {
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
            return Err(format!("{name} is q2f (no bias); affine2 packing needs bias"));
        }
        if packed.group_size != 64 {
            return Err(format!("{name} group_size={} is not 64", packed.group_size));
        }
        let mut packed_sb = vec![0u32; packed.scales_f16.len()];
        for (i, (s, b)) in packed
            .scales_f16
            .iter()
            .zip(packed.biases_f16.iter())
            .enumerate()
        {
            packed_sb[i] = u32::from(*s) | (u32::from(*b) << 16);
        }
        let input = fill_f32(packed.cols);
        let output = vec![0f32; packed.rows];
        let code_bytes = packed.codes.len() as u64;
        let scale_bytes = (packed.scales_f16.len() * 2) as u64;
        let bias_bytes = (packed.biases_f16.len() * 2) as u64;
        Ok(Tensor::Affine {
            name: name.to_string(),
            rows: packed.rows as u32,
            cols: packed.cols as u32,
            group_size: packed.group_size as u32,
            codes: buf_u8(device, &packed.codes),
            scales: buf_u16(device, &packed.scales_f16),
            biases: buf_u16(device, &packed.biases_f16),
            packed_sb: buf_u32(device, &packed_sb),
            input: buf_f32(device, &input),
            output: buf_f32(device, &output),
            weight_bytes: code_bytes + scale_bytes + bias_bytes,
        })
    }

    fn load_q4(
        device: &Device,
        catalog: &HashMap<String, CatalogRow>,
        name: &str,
    ) -> Result<Tensor, String> {
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
        Ok(Tensor::Q4 {
            name: name.to_string(),
            rows,
            cols,
            groups_per_row,
            codes: buf_u8(device, codes),
            scales: buf_u8(device, scales),
            input: buf_f32(device, &input),
            output: buf_f32(device, &output),
            weight_bytes: (codes.len() + scales.len()) as u64,
        })
    }

    fn occupancy(pipe: &Pipe, rows: u32, tg: u64, rows_per_tg: u32) -> Value {
        let threadgroups = u64::from(rows.div_ceil(rows_per_tg).max(1));
        json!({
            "threads_per_threadgroup": tg,
            "rows_per_threadgroup": rows_per_tg,
            "threads_per_row": tg / u64::from(rows_per_tg.max(1)),
            "max_total_threads_per_threadgroup": pipe.max_threads,
            "thread_execution_width": pipe.exec_width,
            "occupancy_of_max_threads": if pipe.max_threads == 0 {
                Value::Null
            } else {
                json!(tg as f64 / pipe.max_threads as f64)
            },
            "threadgroups": threadgroups,
            "gpu_cores_cited": GPU_CORES_CITED,
            "gpu_cores_provenance": "qwen38_geometry.rs ARGMAX_GROUPS comment + KERNEL_GEOMETRY occupancy_class; not a this-run hardware occupancy counter",
            "threadgroups_per_core_cited": threadgroups as f64 / f64::from(GPU_CORES_CITED),
            "registers_per_thread": Value::Null,
        })
    }

    fn dispatch_groups(enc: &metal::ComputeCommandEncoderRef, rows: u32, tg: u64, rows_per_tg: u32) {
        let groups = u64::from(rows.div_ceil(rows_per_tg).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }

    fn encode_launch(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, tensor: &Tensor, geo: &LaunchGeo) {
        enc.set_compute_pipeline_state(&pipe.state);
        match tensor {
            Tensor::Affine {
                rows,
                cols,
                group_size,
                codes,
                scales,
                biases,
                input,
                output,
                ..
            } => {
                enc.set_buffer(0, Some(codes), 0);
                enc.set_buffer(1, Some(scales), 0);
                enc.set_buffer(2, Some(biases), 0);
                enc.set_buffer(3, Some(input), 0);
                enc.set_buffer(4, Some(output), 0);
                set_u32(enc, 5, *rows);
                set_u32(enc, 6, *cols);
                set_u32(enc, 7, *group_size);
                set_u32(enc, 8, geo.tg as u32);
                set_u32(enc, 9, geo.rows_per_tg);
                dispatch_groups(enc, *rows, geo.tg, geo.rows_per_tg);
            }
            Tensor::Q4 {
                rows,
                cols,
                groups_per_row,
                codes,
                scales,
                input,
                output,
                ..
            } => {
                enc.set_buffer(0, Some(codes), 0);
                enc.set_buffer(1, Some(scales), 0);
                enc.set_buffer(2, Some(input), 0);
                enc.set_buffer(3, Some(output), 0);
                set_u32(enc, 4, *rows);
                set_u32(enc, 5, *cols);
                set_u32(enc, 6, *groups_per_row);
                set_u32(enc, 7, geo.tg as u32);
                set_u32(enc, 8, geo.rows_per_tg);
                dispatch_groups(enc, *rows, geo.tg, geo.rows_per_tg);
            }
        }
    }

    fn encode_pack_22232(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, tensor: &Tensor) {
        let Tensor::Affine {
            rows,
            cols,
            codes,
            scales,
            biases,
            input,
            output,
            ..
        } = tensor
        else {
            fail("pack_22232 is affine2-only");
        };
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(codes), 0);
        enc.set_buffer(1, Some(scales), 0);
        enc.set_buffer(2, Some(biases), 0);
        enc.set_buffer(3, Some(input), 0);
        enc.set_buffer(4, Some(output), 0);
        set_u32(enc, 5, *rows);
        set_u32(enc, 6, *cols);
        dispatch_groups(enc, *rows, 128, 2);
    }

    fn encode_pack_mid(enc: &metal::ComputeCommandEncoderRef, pipe: &Pipe, tensor: &Tensor) {
        let Tensor::Affine {
            rows,
            cols,
            codes,
            packed_sb,
            input,
            output,
            ..
        } = tensor
        else {
            fail("pack_mid2432 is affine2-only");
        };
        enc.set_compute_pipeline_state(&pipe.state);
        enc.set_buffer(0, Some(codes), 0);
        enc.set_buffer(1, Some(packed_sb), 0);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(output), 0);
        set_u32(enc, 4, *rows);
        set_u32(enc, 5, *cols);
        dispatch_groups(enc, *rows, 128, 2);
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
            match gpu_ns(cmd) {
                Some(ns) => gpu.push(ns),
                None => fail("no GPU timestamp (MTLCommandBuffer GPUStartTime/GPUEndTime)"),
            }
        }
        gpu
    }

    fn compare(prod: &[f32], var: &[f32]) -> Value {
        if prod.len() != var.len() {
            return json!({"error": "length mismatch", "n_floats_compared": 0});
        }
        let mut n_mismatch = 0u64;
        let mut max_abs = 0.0f32;
        for (a, b) in prod.iter().zip(var.iter()) {
            if a.to_bits() != b.to_bits() {
                n_mismatch += 1;
            }
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        json!({
            "n_floats_compared": prod.len(),
            "n_float_mismatch": n_mismatch,
            "max_abs_err": max_abs,
            "bit_identical": n_mismatch == 0,
            "compared_against": "tg128_r2 production-occupancy output on the same tensor",
        })
    }

    fn point_json(
        id: &str,
        kernel: &str,
        family: &str,
        tg: u64,
        rows_per_tg: u32,
        packing: Option<&str>,
        gpu: Vec<u64>,
        weight_bytes: u64,
        occ: &Value,
        compare: Value,
        note: &str,
    ) -> Value {
        let med = median_u64(gpu.clone()).unwrap_or(0);
        let gb_s = if med == 0 {
            0.0
        } else {
            weight_bytes as f64 / med as f64
        };
        json!({
            "id": id,
            "kernel": kernel,
            "family": family,
            "threads_per_threadgroup": tg,
            "rows_per_threadgroup": rows_per_tg,
            "threads_per_row": tg / u64::from(rows_per_tg.max(1)),
            "stream_packing": packing,
            "weight_bytes": weight_bytes,
            "gpu_ns_median": med,
            "gpu_ns_reps": gpu,
            "gpu_us_median": med as f64 / 1e3,
            "effective_gb_s": gb_s,
            "occupancy": occ,
            "bit_identical_vs_production_geo": compare.get("bit_identical").cloned(),
            "byte_compare": compare,
            "note": note,
        })
    }

    pub fn run(args: Args) -> Value {
        let measured_at = iso8601_now();
        let concurrent_start = concurrent_load();
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal device"));
        let queue = device.new_command_queue();
        let pipes = compile(&device).unwrap_or_else(|e| fail(e));
        let catalog = parse_catalog(&args.artifact_root).unwrap_or_else(|e| fail(e));

        let mut shapes = Vec::new();
        for hot in hots() {
            let tensor_name = match hot.suffix {
                Some(suffix) => qwen38_layer_name(hot.layer, suffix),
                None => qwen38_lm_head_name().to_string(),
            };
            eprintln!(
                "  {} {} (expect {}x{})",
                hot.organ, tensor_name, hot.expected_rows, hot.expected_cols
            );
            let tensor = if hot.affine {
                load_affine(&device, &catalog, &tensor_name)
            } else {
                load_q4(&device, &catalog, &tensor_name)
            }
            .unwrap_or_else(|e| fail(e));
            if tensor.rows() as usize != hot.expected_rows || tensor.cols() as usize != hot.expected_cols {
                fail(format!(
                    "{} catalog shape {}x{} != geometry authority {}x{}",
                    hot.organ,
                    tensor.rows(),
                    tensor.cols(),
                    hot.expected_rows,
                    hot.expected_cols
                ));
            }

            let pipe = if hot.affine {
                pipes.get(AFFINE_KERNEL).unwrap_or_else(|| fail(AFFINE_KERNEL))
            } else {
                pipes.get(Q4_KERNEL).unwrap_or_else(|| fail(Q4_KERNEL))
            };

            let mut launch_json = Vec::new();
            let mut production_out: Option<Vec<f32>> = None;
            for geo in LAUNCH_GEOS {
                if geo.tg > pipe.max_threads {
                    eprintln!(
                        "    {} skipped (pipeline max_threads={})",
                        geo.id, pipe.max_threads
                    );
                    continue;
                }
                if geo.tg % u64::from(geo.rows_per_tg) != 0 {
                    continue;
                }
                let tpr = geo.tg / u64::from(geo.rows_per_tg);
                if tpr < 32 || tpr % 32 != 0 {
                    continue;
                }
                let occ = occupancy(pipe, tensor.rows(), geo.tg, geo.rows_per_tg);
                let _ = time_cb(&queue, args.warmup, |enc| encode_launch(enc, pipe, &tensor, geo));
                let gpu = time_cb(&queue, args.reps, |enc| encode_launch(enc, pipe, &tensor, geo));
                let out = read_f32(tensor.output(), tensor.rows() as usize);
                let cmp = if geo.id == "tg128_r2" {
                    production_out = Some(out.clone());
                    json!({
                        "n_floats_compared": out.len(),
                        "n_float_mismatch": 0,
                        "max_abs_err": 0.0,
                        "bit_identical": true,
                        "compared_against": "self (production occupancy control)",
                    })
                } else {
                    match &production_out {
                        Some(prod) => compare(prod, &out),
                        None => json!({"error": "production geo must run before other geos"}),
                    }
                };
                let med = median_u64(gpu.clone()).unwrap_or(0);
                let gb_s = if med == 0 {
                    0.0
                } else {
                    tensor.weight_bytes() as f64 / med as f64
                };
                eprintln!(
                    "    {:>10}  {:>7.1} GB/s  {:>7.1} us  tg={} rpt={}",
                    geo.id,
                    gb_s,
                    med as f64 / 1e3,
                    geo.tg,
                    geo.rows_per_tg
                );
                launch_json.push(point_json(
                    geo.id,
                    pipe.name,
                    "launch",
                    geo.tg,
                    geo.rows_per_tg,
                    None,
                    gpu,
                    tensor.weight_bytes(),
                    &occ,
                    cmp,
                    "production arithmetic; unique payload bytes held",
                ));
            }

            let mut packing_json = Vec::new();
            if hot.affine {
                let p222 = pipes.get(PACK_22232).unwrap_or_else(|| fail(PACK_22232));
                let pmid = pipes.get(PACK_MID).unwrap_or_else(|| fail(PACK_MID));
                let occ = occupancy(p222, tensor.rows(), 128, 2);
                let _ = time_cb(&queue, args.warmup, |enc| encode_pack_22232(enc, p222, &tensor));
                let gpu = time_cb(&queue, args.reps, |enc| encode_pack_22232(enc, p222, &tensor));
                packing_json.push(point_json(
                    "mlp_2_2_2_32",
                    PACK_22232,
                    "packing",
                    128,
                    2,
                    Some("mlp_2_2_2_32"),
                    gpu,
                    tensor.weight_bytes(),
                    &occ,
                    json!({"note": "stripped packing arm; not compared for bit-identity"}),
                    "stripped 2+2+2+32 at production occupancy; not a merge past mid_2_4_32",
                ));
                let occ = occupancy(pmid, tensor.rows(), 128, 2);
                let _ = time_cb(&queue, args.warmup, |enc| encode_pack_mid(enc, pmid, &tensor));
                let gpu = time_cb(&queue, args.reps, |enc| encode_pack_mid(enc, pmid, &tensor));
                packing_json.push(point_json(
                    "mid_2_4_32",
                    PACK_MID,
                    "packing",
                    128,
                    2,
                    Some("mid_2_4_32"),
                    gpu,
                    tensor.weight_bytes(),
                    &occ,
                    json!({"note": "stripped packing arm; not compared for bit-identity"}),
                    "stripped 2+4+32 (MLP_STREAM_COUNT peak). pack_6_32/pack_38 are not run.",
                ));
            }

            shapes.push(json!({
                "organ": hot.organ,
                "family": hot.family,
                "dtype": hot.dtype,
                "rows": tensor.rows(),
                "cols": tensor.cols(),
                "catalog_tensor": tensor.name(),
                "layer": hot.layer,
                "weight_bytes": tensor.weight_bytes(),
                "launch": launch_json,
                "packing": packing_json,
                "refused_this_shape": [
                    "accumulator_chain",
                    "working_set",
                    "pack_6_32",
                    "pack_38",
                ],
            }));
        }

        let concurrent_end = concurrent_load();
        json!({
            "schema": "hawking.future.geometry_table.raw.v1",
            "git_head": git_head(),
            "artifact_root": args.artifact_root.display().to_string(),
            "measured_at": measured_at,
            "gpu_lane_lock_held": true,
            "warmup": args.warmup,
            "reps": args.reps,
            "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "concurrent_load": concurrent_start,
            "concurrent_load_end": concurrent_end,
            "absolute_gb_s_are_measured_under_load": true,
            "fast_math": false,
            "does_not_edit_production_shaders": true,
            "refused_discriminators": [
                "accumulator_chain (span 1.062, MLP_ISSUE_RATE_LADDER)",
                "working_set (span 1.078 at occupancy span 1.0, MLP_ISSUE_RATE_LADDER)",
                "pack_6_32 / pack_38 (MLP_STREAM_COUNT: merging further than mid_2_4_32 hurt)",
            ],
            "launch_geos": LAUNCH_GEOS.iter().map(|g| json!({
                "id": g.id, "threads_per_threadgroup": g.tg, "rows_per_threadgroup": g.rows_per_tg
            })).collect::<Vec<_>>(),
            "gpu_cores_cited": GPU_CORES_CITED,
            "gpu_cores_provenance": "qwen38_geometry.rs ARGMAX_GROUPS comment + KERNEL_GEOMETRY occupancy_class; not a this-run hardware occupancy counter",
            "shapes": shapes,
        })
    }
}
