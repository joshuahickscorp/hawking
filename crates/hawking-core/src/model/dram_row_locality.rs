//! DRAM row locality: derived address-stream model and execution-order packs.
//!
//! DRAM is not flat. A miss is ACTIVATE → column read → PRECHARGE. A hit on
//! the open row is a column read only. This module does **not** read Apple's
//! memory-controller counters; every row/page figure is derived from the
//! address stream under an 8 KiB LPDDR5 row model. It is a model, not a
//! hardware counter.
//!
//! Standing pack rule: physical layout follows the order the token graph
//! consumes. For MoE that means an expert's gate/up/down (w1/w3/w2) triplet
//! and that organ's scales beside its codes — not consecutive expert numbers
//! and not "all scales then all codes".
//!
//! Layout transforms here are value-preserving: identical decoded elements,
//! identical BPW (same payload bytes). They do not rewrite a live catalog
//! and they are not the default bind path (DSV4F no-copy mmap would be
//! destroyed by a bind-time memcpy; that is a pack-time lever).

use crate::model::qwen_complete_binary::{
    BinaryGroupPacked, UNIFORM_Q4_CODE_BYTES_PER_GROUP, UNIFORM_Q4_GROUP_SIZE,
};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// Derived LPDDR5 row size. Not a published Apple figure.
pub const DRAM_ROW_BYTES_MODEL: u64 = 8192;
pub const PAGE_4K: u64 = 4096;
pub const PAGE_16K: u64 = 16384;
/// Published M3 Ultra peak. Used only as a denominator.
pub const CEILING_GBPS: f64 = 819.0;

pub const Q80_BINARY_GROUP: usize = 128;
pub const Q80_GATE_ROWS: usize = 512;
pub const Q80_GATE_COLS: usize = 2048;
pub const Q80_DOWN_ROWS: usize = 2048;
pub const Q80_DOWN_COLS: usize = 512;
pub const Q80_LAYERS: u64 = 48;
pub const Q80_TOP_K: u64 = 10;
pub const Q80_EXPERTS: u32 = 512;

pub const DSV4F_HIDDEN: usize = 4096;
pub const DSV4F_INTER: usize = 2048;
pub const DSV4F_FP4_BLOCK: usize = 32;
pub const DSV4F_LAYERS: u64 = 43;
pub const DSV4F_TOP_K: u64 = 6;
pub const DSV4F_W1_PACKED: usize = DSV4F_INTER * (DSV4F_HIDDEN / 2);
pub const DSV4F_W1_SCALES: usize = DSV4F_INTER * (DSV4F_HIDDEN / DSV4F_FP4_BLOCK);
pub const DSV4F_W2_PACKED: usize = DSV4F_HIDDEN * (DSV4F_INTER / 2);
pub const DSV4F_W2_SCALES: usize = DSV4F_HIDDEN * (DSV4F_INTER / DSV4F_FP4_BLOCK);

/// One kernel-issued read. Addresses are in a synthetic process space:
/// each named region is assigned a disjoint 1 GiB window so a jump between
/// buffers is visible as a multi-megabyte stride.
#[derive(Clone, Copy, Debug)]
pub struct AddressTouch {
    pub addr: u64,
    pub bytes: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StreamStats {
    pub name: String,
    pub layout: String,
    pub touches: usize,
    pub bytes: u64,
    pub sequential_runs: usize,
    pub mean_run_bytes: f64,
    pub p50_run_bytes: u64,
    pub p90_run_bytes: u64,
    pub max_run_bytes: u64,
    pub stride_ge_4k: usize,
    pub stride_ge_row: usize,
    pub dram_row_transitions: usize,
    pub unique_pages_4k: usize,
    pub unique_pages_16k: usize,
    pub unique_dram_rows: usize,
    pub min_possible_dram_rows: usize,
    pub scatter_ratio: f64,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct RankedStream {
    pub stats: StreamStats,
    pub bytes_x_scatter: f64,
}

fn locality_error(detail: impl Into<String>) -> Error {
    Error::Model(format!("dram-row-locality: {}", detail.into()))
}

/// Analyse one address stream. Touches should be in issue order.
pub fn analyze_stream(name: &str, layout: &str, touches: &[AddressTouch], note: &str) -> StreamStats {
    let mut bytes = 0u64;
    let mut runs: Vec<u64> = Vec::new();
    let mut run_bytes = 0u64;
    let mut stride_ge_4k = 0usize;
    let mut stride_ge_row = 0usize;
    let mut dram_row_transitions = 0usize;
    let mut pages_4k = BTreeMap::<u64, ()>::new();
    let mut pages_16k = BTreeMap::<u64, ()>::new();
    let mut rows = BTreeMap::<u64, ()>::new();

    let mut prev_end: Option<u64> = None;
    let mut prev_row: Option<u64> = None;
    for touch in touches {
        let n = u64::from(touch.bytes);
        bytes = bytes.saturating_add(n);
        let start = touch.addr;
        let end = start.saturating_add(n);
        let mut a = start;
        while a < end {
            pages_4k.insert(a / PAGE_4K, ());
            pages_16k.insert(a / PAGE_16K, ());
            rows.insert(a / DRAM_ROW_BYTES_MODEL, ());
            let next = ((a / PAGE_4K) + 1).saturating_mul(PAGE_4K);
            a = next.min(end);
        }
        let this_row = start / DRAM_ROW_BYTES_MODEL;
        if let Some(prev) = prev_row {
            if this_row != prev {
                dram_row_transitions += 1;
            }
        }
        prev_row = Some(this_row);
        if let Some(prev) = prev_end {
            if start == prev {
                run_bytes = run_bytes.saturating_add(n);
            } else {
                if run_bytes > 0 {
                    runs.push(run_bytes);
                }
                run_bytes = n;
                let stride = start.abs_diff(prev);
                if stride >= PAGE_4K {
                    stride_ge_4k += 1;
                }
                if stride >= DRAM_ROW_BYTES_MODEL {
                    stride_ge_row += 1;
                }
            }
        } else {
            run_bytes = n;
        }
        prev_end = Some(end);
    }
    if run_bytes > 0 {
        runs.push(run_bytes);
    }
    let mut sorted = runs.clone();
    sorted.sort_unstable();
    let n_runs = sorted.len();
    let mean_run_bytes = if n_runs == 0 {
        0.0
    } else {
        runs.iter().sum::<u64>() as f64 / n_runs as f64
    };
    let p50 = if n_runs == 0 {
        0
    } else {
        sorted[n_runs / 2]
    };
    let p90 = if n_runs == 0 {
        0
    } else {
        sorted[(n_runs * 9) / 10]
    };
    let max_run = sorted.last().copied().unwrap_or(0);
    let min_rows = ((bytes + DRAM_ROW_BYTES_MODEL - 1) / DRAM_ROW_BYTES_MODEL).max(1) as usize;
    let unique_rows = rows.len().max(1);
    StreamStats {
        name: name.to_owned(),
        layout: layout.to_owned(),
        touches: touches.len(),
        bytes,
        sequential_runs: n_runs,
        mean_run_bytes,
        p50_run_bytes: p50,
        p90_run_bytes: p90,
        max_run_bytes: max_run,
        stride_ge_4k,
        stride_ge_row,
        dram_row_transitions,
        unique_pages_4k: pages_4k.len(),
        unique_pages_16k: pages_16k.len(),
        unique_dram_rows: unique_rows,
        min_possible_dram_rows: min_rows,
        scatter_ratio: unique_rows as f64 / min_rows as f64,
        note: note.to_owned(),
    }
}

pub fn rank_streams(stats: &[StreamStats]) -> Vec<RankedStream> {
    let mut ranked: Vec<RankedStream> = stats
        .iter()
        .cloned()
        .map(|s| {
            let bytes_x_scatter = s.bytes as f64
                * s.scatter_ratio
                * (1.0 + s.dram_row_transitions as f64 / s.bytes.max(1) as f64);
            RankedStream {
                stats: s,
                bytes_x_scatter,
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.bytes_x_scatter
            .partial_cmp(&a.bytes_x_scatter)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

fn region(id: u64) -> u64 {
    id.saturating_mul(1 << 30)
}

fn push_run(out: &mut Vec<AddressTouch>, addr: u64, bytes: u32) {
    if bytes == 0 {
        return;
    }
    if let Some(last) = out.last_mut() {
        if last.addr.saturating_add(u64::from(last.bytes)) == addr {
            last.bytes = last.bytes.saturating_add(bytes);
            return;
        }
    }
    out.push(AddressTouch { addr, bytes });
}

/// Current Q80 mixed / Q4 bind: two Metal buffers, scales then codes per group.
pub fn q80_binary_split_stream(rows: usize, cols: usize, group: usize) -> Vec<AddressTouch> {
    let groups = cols / group;
    let sign_bytes_per_row = cols / 8;
    let scale_base = region(1);
    let sign_base = region(2);
    let mut out = Vec::with_capacity(rows * groups * 2);
    for row in 0..rows {
        for g in 0..groups {
            push_run(
                &mut out,
                scale_base + ((row * groups + g) * 2) as u64,
                2,
            );
            push_run(
                &mut out,
                sign_base + (row * sign_bytes_per_row + g * (group / 8)) as u64,
                (group / 8) as u32,
            );
        }
    }
    out
}

/// Execution-order binary: per group `[fp16 scale | sign bytes]`.
pub fn q80_binary_interleaved_stream(rows: usize, cols: usize, group: usize) -> Vec<AddressTouch> {
    let groups = cols / group;
    let stride = 2 + group / 8;
    let base = region(3);
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        push_run(
            &mut out,
            base + (row * groups * stride) as u64,
            (groups * stride) as u32,
        );
    }
    out
}

pub fn q80_q4_split_stream(rows: usize, cols: usize) -> Vec<AddressTouch> {
    let groups = cols / UNIFORM_Q4_GROUP_SIZE;
    let scale_base = region(4);
    let code_base = region(5);
    let mut out = Vec::with_capacity(rows * groups * 2);
    for row in 0..rows {
        for g in 0..groups {
            push_run(
                &mut out,
                scale_base + ((row * groups + g) * 2) as u64,
                2,
            );
            push_run(
                &mut out,
                code_base
                    + ((row * groups + g) * UNIFORM_Q4_CODE_BYTES_PER_GROUP) as u64,
                UNIFORM_Q4_CODE_BYTES_PER_GROUP as u32,
            );
        }
    }
    out
}

pub fn q80_q4_interleaved_stream(rows: usize, cols: usize) -> Vec<AddressTouch> {
    let groups = cols / UNIFORM_Q4_GROUP_SIZE;
    let stride = 2 + UNIFORM_Q4_CODE_BYTES_PER_GROUP;
    let base = region(6);
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        push_run(
            &mut out,
            base + (row * groups * stride) as u64,
            (groups * stride) as u32,
        );
    }
    out
}

/// DSV4F worklist FP4: packed weights and E8M0 scales in two buffers.
/// One thread owns (slot, row); we emit one expert's one projection.
pub fn dsv4f_fp4_split_stream(rows: usize, packed_cols: usize, scale_cols: usize) -> Vec<AddressTouch> {
    let packed_base = region(7);
    let scale_base = region(8);
    let mut out = Vec::with_capacity(rows * scale_cols * 2);
    for row in 0..rows {
        for block in 0..scale_cols {
            push_run(
                &mut out,
                packed_base + (row * packed_cols + block * (DSV4F_FP4_BLOCK / 2)) as u64,
                (DSV4F_FP4_BLOCK / 2) as u32,
            );
            push_run(
                &mut out,
                scale_base + (row * scale_cols + block) as u64,
                1,
            );
        }
    }
    out
}

pub fn dsv4f_fp4_interleaved_stream(
    rows: usize,
    packed_cols: usize,
    scale_cols: usize,
) -> Vec<AddressTouch> {
    let _ = packed_cols;
    let stride = 1 + DSV4F_FP4_BLOCK / 2;
    let base = region(9);
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        push_run(
            &mut out,
            base + (row * scale_cols * stride) as u64,
            (scale_cols * stride) as u32,
        );
    }
    out
}

/// Six separate MTLBuffers / gravity chunks for one expert.
pub fn dsv4f_expert_six_chunk_stream() -> Vec<AddressTouch> {
    let sizes = [
        DSV4F_W1_PACKED,
        DSV4F_W1_SCALES,
        DSV4F_W1_PACKED,
        DSV4F_W1_SCALES,
        DSV4F_W2_PACKED,
        DSV4F_W2_SCALES,
    ];
    let mut out = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        // SHA-named chunks live in unrelated 1 GiB windows.
        push_run(&mut out, region(20 + i as u64), n as u32);
    }
    out
}

/// One blob, execution order w1 → w3 → w2, each organ interleaved.
pub fn dsv4f_expert_colocated_stream() -> Vec<AddressTouch> {
    let w1 = DSV4F_W1_PACKED + DSV4F_W1_SCALES;
    let w2 = DSV4F_W2_PACKED + DSV4F_W2_SCALES;
    let total = w1 + w1 + w2;
    vec![AddressTouch {
        addr: region(30),
        bytes: total as u32,
    }]
}

/// Q80 Q4 expert: six SHA-named files (gate/up/down × codes/scales).
pub fn q80_expert_six_file_stream() -> Vec<AddressTouch> {
    let groups = (Q80_GATE_ROWS * Q80_GATE_COLS) / UNIFORM_Q4_GROUP_SIZE;
    let codes = groups * UNIFORM_Q4_CODE_BYTES_PER_GROUP;
    let scales = groups * 2;
    let mut out = Vec::new();
    for i in 0..6u64 {
        let n = if i % 2 == 0 { codes } else { scales };
        push_run(&mut out, region(40 + i), n as u32);
    }
    out
}

pub fn q80_expert_colocated_stream() -> Vec<AddressTouch> {
    let groups = (Q80_GATE_ROWS * Q80_GATE_COLS) / UNIFORM_Q4_GROUP_SIZE;
    let one = groups * (2 + UNIFORM_Q4_CODE_BYTES_PER_GROUP);
    vec![AddressTouch {
        addr: region(50),
        bytes: (one * 3) as u32,
    }]
}

pub fn scale_stream_to_token(stats: &StreamStats, copies: u64) -> StreamStats {
    let mut scaled = stats.clone();
    scaled.bytes = stats.bytes.saturating_mul(copies);
    scaled.touches = stats.touches.saturating_mul(copies as usize);
    scaled.sequential_runs = stats.sequential_runs.saturating_mul(copies as usize);
    scaled.stride_ge_4k = stats.stride_ge_4k.saturating_mul(copies as usize);
    scaled.stride_ge_row = stats.stride_ge_row.saturating_mul(copies as usize);
    scaled.dram_row_transitions = stats
        .dram_row_transitions
        .saturating_mul(copies as usize)
        .saturating_add(copies.saturating_sub(1) as usize);
    scaled.unique_pages_4k = stats.unique_pages_4k.saturating_mul(copies as usize);
    scaled.unique_pages_16k = stats.unique_pages_16k.saturating_mul(copies as usize);
    scaled.unique_dram_rows = stats.unique_dram_rows.saturating_mul(copies as usize);
    scaled.min_possible_dram_rows = ((scaled.bytes + DRAM_ROW_BYTES_MODEL - 1)
        / DRAM_ROW_BYTES_MODEL)
        .max(1) as usize;
    scaled.scatter_ratio =
        scaled.unique_dram_rows as f64 / scaled.min_possible_dram_rows.max(1) as f64;
    scaled
}

// ── value-preserving interleave ──────────────────────────────────────────

pub const Q4_INTERLEAVED_STRIDE: usize = 2 + UNIFORM_Q4_CODE_BYTES_PER_GROUP;
pub const BINARY_INTERLEAVED_STRIDE_G128: usize = 2 + Q80_BINARY_GROUP / 8;
pub const FP4_INTERLEAVED_STRIDE: usize = 1 + DSV4F_FP4_BLOCK / 2;

/// `[fp16 scale | 32 code bytes]` per group of 64. Same bytes as split.
pub fn interleave_q4_groups(scales_f16: &[u16], codes: &[u8]) -> Result<Vec<u8>> {
    if scales_f16.len().saturating_mul(UNIFORM_Q4_CODE_BYTES_PER_GROUP) != codes.len() {
        return Err(locality_error("q4 scale/code length disagree"));
    }
    let mut out = vec![0u8; scales_f16.len() * Q4_INTERLEAVED_STRIDE];
    for (group, &scale) in scales_f16.iter().enumerate() {
        let rec = group * Q4_INTERLEAVED_STRIDE;
        out[rec..rec + 2].copy_from_slice(&scale.to_le_bytes());
        let src = group * UNIFORM_Q4_CODE_BYTES_PER_GROUP;
        out[rec + 2..rec + Q4_INTERLEAVED_STRIDE]
            .copy_from_slice(&codes[src..src + UNIFORM_Q4_CODE_BYTES_PER_GROUP]);
    }
    Ok(out)
}

pub fn deinterleave_q4_groups(body: &[u8]) -> Result<(Vec<u16>, Vec<u8>)> {
    if body.len() % Q4_INTERLEAVED_STRIDE != 0 {
        return Err(locality_error("q4 interleaved body is not a whole number of groups"));
    }
    let groups = body.len() / Q4_INTERLEAVED_STRIDE;
    let mut scales = vec![0u16; groups];
    let mut codes = vec![0u8; groups * UNIFORM_Q4_CODE_BYTES_PER_GROUP];
    for group in 0..groups {
        let rec = group * Q4_INTERLEAVED_STRIDE;
        scales[group] = u16::from_le_bytes([body[rec], body[rec + 1]]);
        let dst = group * UNIFORM_Q4_CODE_BYTES_PER_GROUP;
        codes[dst..dst + UNIFORM_Q4_CODE_BYTES_PER_GROUP]
            .copy_from_slice(&body[rec + 2..rec + Q4_INTERLEAVED_STRIDE]);
    }
    Ok((scales, codes))
}

pub fn q4_weight_from_split(
    scales_f16: &[u16],
    codes: &[u8],
    cols: usize,
    row: usize,
    col: usize,
) -> f32 {
    let group = (row * cols + col) / UNIFORM_Q4_GROUP_SIZE;
    let local = (row * cols + col) % UNIFORM_Q4_GROUP_SIZE;
    let scale = half::f16::from_bits(scales_f16[group]).to_f32();
    let packed = codes[group * UNIFORM_Q4_CODE_BYTES_PER_GROUP + local / 2];
    let nibble = if local & 1 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    (nibble as i32 - 8) as f32 * scale
}

pub fn q4_weight_from_interleaved(body: &[u8], cols: usize, row: usize, col: usize) -> f32 {
    let group = (row * cols + col) / UNIFORM_Q4_GROUP_SIZE;
    let local = (row * cols + col) % UNIFORM_Q4_GROUP_SIZE;
    let rec = group * Q4_INTERLEAVED_STRIDE;
    let scale = half::f16::from_bits(u16::from_le_bytes([body[rec], body[rec + 1]])).to_f32();
    let packed = body[rec + 2 + local / 2];
    let nibble = if local & 1 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    (nibble as i32 - 8) as f32 * scale
}

/// Per-group `[fp16 scale | sign bytes]`. Same bytes as the split body.
pub fn interleave_binary_group(packed: &BinaryGroupPacked) -> Result<Vec<u8>> {
    if packed.group_size % 8 != 0 {
        return Err(locality_error("binary group_size must be a multiple of 8"));
    }
    let sign_bytes = packed.group_size / 8;
    let stride = 2 + sign_bytes;
    let groups = packed.rows * packed.groups_per_row;
    if packed.scales_f16.len() != groups {
        return Err(locality_error("binary scale count disagrees with geometry"));
    }
    let mut out = vec![0u8; groups * stride];
    for group in 0..groups {
        let rec = group * stride;
        out[rec..rec + 2].copy_from_slice(&packed.scales_f16[group].to_le_bytes());
        let src = group * sign_bytes;
        out[rec + 2..rec + stride].copy_from_slice(&packed.signs[src..src + sign_bytes]);
    }
    Ok(out)
}

pub fn binary_weight_from_interleaved(
    body: &[u8],
    cols: usize,
    group_size: usize,
    row: usize,
    col: usize,
) -> f32 {
    let groups_per_row = cols / group_size;
    let group = row * groups_per_row + col / group_size;
    let local = col % group_size;
    let sign_bytes = group_size / 8;
    let stride = 2 + sign_bytes;
    let rec = group * stride;
    let scale = half::f16::from_bits(u16::from_le_bytes([body[rec], body[rec + 1]])).to_f32();
    let byte = body[rec + 2 + local / 8];
    let positive = ((byte >> (local & 7)) & 1) != 0;
    if positive {
        scale
    } else {
        -scale
    }
}

/// Per FP4 block of 32 logical weights: `[e8m0 scale | 16 packed bytes]`.
pub fn interleave_fp4_blocks(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    packed_cols: usize,
    scale_cols: usize,
) -> Result<Vec<u8>> {
    if packed.len() != rows * packed_cols || scales.len() != rows * scale_cols {
        return Err(locality_error("fp4 packed/scale length disagrees with geometry"));
    }
    if packed_cols * 2 != scale_cols * DSV4F_FP4_BLOCK {
        return Err(locality_error("fp4 packed_cols and scale_cols are not a 32-block pair"));
    }
    let packed_per_block = DSV4F_FP4_BLOCK / 2;
    let mut out = vec![0u8; rows * scale_cols * FP4_INTERLEAVED_STRIDE];
    for row in 0..rows {
        for block in 0..scale_cols {
            let rec = (row * scale_cols + block) * FP4_INTERLEAVED_STRIDE;
            out[rec] = scales[row * scale_cols + block];
            let src = row * packed_cols + block * packed_per_block;
            out[rec + 1..rec + FP4_INTERLEAVED_STRIDE]
                .copy_from_slice(&packed[src..src + packed_per_block]);
        }
    }
    Ok(out)
}

pub fn deinterleave_fp4_blocks(
    body: &[u8],
    rows: usize,
    packed_cols: usize,
    scale_cols: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if body.len() != rows * scale_cols * FP4_INTERLEAVED_STRIDE {
        return Err(locality_error("fp4 interleaved length disagrees with geometry"));
    }
    let packed_per_block = DSV4F_FP4_BLOCK / 2;
    let mut packed = vec![0u8; rows * packed_cols];
    let mut scales = vec![0u8; rows * scale_cols];
    for row in 0..rows {
        for block in 0..scale_cols {
            let rec = (row * scale_cols + block) * FP4_INTERLEAVED_STRIDE;
            scales[row * scale_cols + block] = body[rec];
            let dst = row * packed_cols + block * packed_per_block;
            packed[dst..dst + packed_per_block]
                .copy_from_slice(&body[rec + 1..rec + FP4_INTERLEAVED_STRIDE]);
        }
    }
    Ok((packed, scales))
}

/// Execution-order blob: gate, then up, then down. Length-prefixed so a
/// kernel table can bind three subranges of one allocation.
pub fn pack_triplet_blob(gate: &[u8], up: &[u8], down: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + gate.len() + up.len() + down.len());
    out.extend_from_slice(&(gate.len() as u64).to_le_bytes());
    out.extend_from_slice(&(up.len() as u64).to_le_bytes());
    out.extend_from_slice(&(down.len() as u64).to_le_bytes());
    out.extend_from_slice(gate);
    out.extend_from_slice(up);
    out.extend_from_slice(down);
    out
}

pub fn unpack_triplet_blob(body: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if body.len() < 24 {
        return Err(locality_error("triplet blob shorter than the length prefix"));
    }
    let g = u64::from_le_bytes(body[0..8].try_into().unwrap()) as usize;
    let u = u64::from_le_bytes(body[8..16].try_into().unwrap()) as usize;
    let d = u64::from_le_bytes(body[16..24].try_into().unwrap()) as usize;
    let expect = 24usize
        .checked_add(g)
        .and_then(|n| n.checked_add(u))
        .and_then(|n| n.checked_add(d))
        .ok_or_else(|| locality_error("triplet blob length overflow"))?;
    if body.len() != expect {
        return Err(locality_error(format!(
            "triplet blob {} bytes != 24+{g}+{u}+{d}",
            body.len()
        )));
    }
    Ok((
        body[24..24 + g].to_vec(),
        body[24 + g..24 + g + u].to_vec(),
        body[24 + g + u..].to_vec(),
    ))
}

/// Cheap npy v1/v2 reader. Used for the existing capture-index route table.
pub fn read_npy_i32(path: &std::path::Path) -> Result<Vec<i32>> {
    let raw = std::fs::read(path).map_err(|e| locality_error(format!("read {}: {e}", path.display())))?;
    let (_header, data) = split_npy(&raw)?;
    if data.len() % 4 != 0 {
        return Err(locality_error("npy i32 payload is not a multiple of 4"));
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn read_npy_i16(path: &std::path::Path) -> Result<Vec<i16>> {
    let raw = std::fs::read(path).map_err(|e| locality_error(format!("read {}: {e}", path.display())))?;
    let (_header, data) = split_npy(&raw)?;
    if data.len() % 2 != 0 {
        return Err(locality_error("npy i16 payload is not a multiple of 2"));
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn split_npy(raw: &[u8]) -> Result<(&str, &[u8])> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err(locality_error("not a npy file"));
    }
    let ver = raw[6];
    let (hlen, header_off) = if ver == 1 {
        (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10usize)
    } else {
        if raw.len() < 12 {
            return Err(locality_error("npy v2 header truncated"));
        }
        (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12usize,
        )
    };
    let header_end = header_off
        .checked_add(hlen)
        .ok_or_else(|| locality_error("npy header overflow"))?;
    if header_end > raw.len() {
        return Err(locality_error("npy header exceeds file"));
    }
    let header = std::str::from_utf8(&raw[header_off..header_end])
        .map_err(|_| locality_error("npy header is not utf-8"))?;
    Ok((header, &raw[header_end..]))
}

/// Greedy placement: hottest expert first, then the unplaced expert with
/// the largest co-route count against the last placed expert. Never-routed
/// experts (freq 0) fall to the tail in original id order.
pub fn greedy_coreoute_order(freq: &[u32], pair: &[u32], n: usize) -> Vec<u32> {
    debug_assert_eq!(freq.len(), n);
    debug_assert_eq!(pair.len(), n * n);
    let mut remaining: Vec<u32> = (0..n as u32).collect();
    remaining.sort_by(|&a, &b| {
        freq[b as usize]
            .cmp(&freq[a as usize])
            .then(a.cmp(&b))
    });
    let mut order = Vec::with_capacity(n);
    let mut used = vec![false; n];
    if remaining.is_empty() {
        return order;
    }
    let first = remaining[0];
    order.push(first);
    used[first as usize] = true;
    while order.len() < n {
        let last = order[order.len() - 1] as usize;
        let mut best = None;
        let mut best_score = 0u32;
        let mut best_freq = 0u32;
        for cand in 0..n {
            if used[cand] {
                continue;
            }
            let score = pair[last * n + cand];
            let f = freq[cand];
            let better = match best {
                None => true,
                Some(_) => {
                    score > best_score || (score == best_score && (f > best_freq || (f == best_freq && cand < best.unwrap())))
                }
            };
            if better {
                best = Some(cand);
                best_score = score;
                best_freq = f;
            }
        }
        let pick = best.unwrap() as u32;
        used[pick as usize] = true;
        order.push(pick);
    }
    order
}

pub fn pair_mean_abs_distance(pairs: &[(u32, u32, u32)], map: &[u32]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for &(a, b, c) in pairs {
        if c == 0 {
            continue;
        }
        let da = map[a as usize] as i64;
        let db = map[b as usize] as i64;
        num += (da - db).unsigned_abs() as f64 * f64::from(c);
        den += f64::from(c);
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen_complete_binary::{
        binary_group_weight, deterministic_matrix, pack_binary_group, pack_uniform_q4_group64,
        parse_uniform_q4_header,
    };

    #[test]
    fn interleaved_q4_is_bit_identical() {
        let rows = 8;
        let cols = 64;
        let values = deterministic_matrix(rows, cols, 9);
        let (payload, quality) = pack_uniform_q4_group64(&values, &[rows, cols]).unwrap();
        assert!((quality.codec_bpw - 4.25).abs() < 1e-12);
        let header = parse_uniform_q4_header(&payload).unwrap();
        let mut scales = Vec::new();
        for g in 0..header.groups {
            scales.push(u16::from_le_bytes([
                payload[header.scale_offset + g * 2],
                payload[header.scale_offset + g * 2 + 1],
            ]));
        }
        let codes = payload[header.sign_offset..].to_vec();
        let interleaved = interleave_q4_groups(&scales, &codes).unwrap();
        let (back_s, back_c) = deinterleave_q4_groups(&interleaved).unwrap();
        assert_eq!(back_s, scales);
        assert_eq!(back_c, codes);
        for row in 0..rows {
            for col in 0..cols {
                let a = q4_weight_from_split(&scales, &codes, cols, row, col);
                let b = q4_weight_from_interleaved(&interleaved, cols, row, col);
                assert_eq!(a.to_bits(), b.to_bits(), "row {row} col {col}");
            }
        }
        assert_eq!(interleaved.len(), scales.len() * Q4_INTERLEAVED_STRIDE);
    }

    #[test]
    fn interleaved_binary_is_bit_identical() {
        let rows = 4;
        let cols = 256;
        let w = deterministic_matrix(rows, cols, 3);
        let packed = pack_binary_group(&w, rows, cols, Q80_BINARY_GROUP).unwrap();
        let body = interleave_binary_group(&packed).unwrap();
        for row in 0..rows {
            for col in 0..cols {
                let a = binary_group_weight(&packed, row, col);
                let b = binary_weight_from_interleaved(&body, cols, Q80_BINARY_GROUP, row, col);
                assert_eq!(a.to_bits(), b.to_bits(), "row {row} col {col}");
            }
        }
    }

    #[test]
    fn interleaved_fp4_roundtrip_bytes() {
        let rows = 4;
        let packed_cols = 64;
        let scale_cols = 4;
        let packed: Vec<u8> = (0..rows * packed_cols).map(|i| (i * 17) as u8).collect();
        let scales: Vec<u8> = (0..rows * scale_cols).map(|i| (i * 3 + 1) as u8).collect();
        let body = interleave_fp4_blocks(&packed, &scales, rows, packed_cols, scale_cols).unwrap();
        let (p2, s2) = deinterleave_fp4_blocks(&body, rows, packed_cols, scale_cols).unwrap();
        assert_eq!(p2, packed);
        assert_eq!(s2, scales);
    }

    #[test]
    fn triplet_blob_roundtrip() {
        let blob = pack_triplet_blob(b"gate", b"upup", b"down!");
        let (g, u, d) = unpack_triplet_blob(&blob).unwrap();
        assert_eq!(g, b"gate");
        assert_eq!(u, b"upup");
        assert_eq!(d, b"down!");
    }

    #[test]
    fn interleaved_collapses_row_transitions() {
        let split = analyze_stream(
            "q4",
            "split",
            &q80_q4_split_stream(Q80_GATE_ROWS, Q80_GATE_COLS),
            "test",
        );
        let inter = analyze_stream(
            "q4",
            "interleaved",
            &q80_q4_interleaved_stream(Q80_GATE_ROWS, Q80_GATE_COLS),
            "test",
        );
        assert!(
            inter.dram_row_transitions < split.dram_row_transitions,
            "inter {} split {}",
            inter.dram_row_transitions,
            split.dram_row_transitions
        );
        assert!(inter.mean_run_bytes > split.mean_run_bytes);
        assert_eq!(split.bytes, inter.bytes);
    }

    #[test]
    fn greedy_order_puts_never_routed_last_when_zero() {
        let n = 4;
        let freq = vec![10, 0, 8, 0];
        let mut pair = vec![0u32; n * n];
        pair[0 * n + 2] = 5;
        pair[2 * n + 0] = 5;
        let order = greedy_coreoute_order(&freq, &pair, n);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 2);
        assert!(order[2] == 1 || order[2] == 3);
        assert!(order.contains(&1) && order.contains(&3));
    }
}
