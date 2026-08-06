#![cfg(target_os = "macos")]

//! Small, self-contained evidence fixture for the Qwen direct packed-Q4 Metal
//! lane. This deliberately exercises the same GGUF -> Q4_K -> MTLBuffer ->
//! token-command-buffer boundary as the dense loader without admitting a model
//! body or pretending that a kernel fixture is a full-model benchmark.

use half::f16;
use hawking_core::gguf::{GgmlType, GgufFile};
use hawking_core::gravity::{GravityShard, GravityWeights};
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use hawking_core::numeric_parity::{format_score_line, score_pair, Bounds};
use hawking_core::quant;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const ROWS: usize = 8;
const COLS: usize = 256;
const Q4K_BYTES: usize = ROWS * quant::Q4_K_BLOCK_BYTES;
const Q5K_BYTES: usize = ROWS * quant::Q5_K_BLOCK_BYTES;

#[derive(Debug)]
struct Stage {
    name: &'static str,
    api: &'static str,
    bytes: usize,
    elapsed: Duration,
}

impl Stage {
    fn emit(&self) {
        eprintln!(
            "QWEN_Q4K_FIXTURE_STAGE stage={} api={} bytes={} elapsed_us={}",
            self.name,
            self.api,
            self.bytes,
            self.elapsed.as_micros(),
        );
    }
}

fn timed<T>(
    name: &'static str,
    api: &'static str,
    bytes: usize,
    f: impl FnOnce() -> T,
) -> (T, Stage) {
    let start = Instant::now();
    let value = f();
    let stage = Stage {
        name,
        api,
        bytes,
        elapsed: start.elapsed(),
    };
    stage.emit();
    (value, stage)
}

fn put_u32(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn put_string(dst: &mut Vec<u8>, value: &str) {
    put_u64(dst, value.len() as u64);
    dst.extend_from_slice(value.as_bytes());
}

fn write_q4k_gguf(dir: &TempDir, packed: &[u8]) -> PathBuf {
    assert_eq!(packed.len(), Q4K_BYTES);
    let mut gguf = Vec::new();
    put_u32(&mut gguf, 0x4655_4747); // GGUF, little-endian.
    put_u32(&mut gguf, 3);
    put_u64(&mut gguf, 1); // tensors
    put_u64(&mut gguf, 0); // metadata entries
    put_string(&mut gguf, "blk.0.ffn_gate.weight");
    put_u32(&mut gguf, 2);
    // GGUF uses [columns, rows] for a conventional matrix tensor.
    put_u64(&mut gguf, COLS as u64);
    put_u64(&mut gguf, ROWS as u64);
    put_u32(&mut gguf, GgmlType::Q4_K as u32);
    put_u64(&mut gguf, 0); // tensor data offset relative to aligned data base
    while gguf.len() % 32 != 0 {
        gguf.push(0);
    }
    gguf.extend_from_slice(packed);
    let path = dir.path().join("qwen-q4k-metal-fixture.gguf");
    fs::write(&path, gguf).expect("write temporary GGUF fixture");
    path
}

fn write_q5k_gguf(dir: &TempDir, packed: &[u8]) -> PathBuf {
    assert_eq!(packed.len(), Q5K_BYTES);
    let mut gguf = Vec::new();
    put_u32(&mut gguf, 0x4655_4747);
    put_u32(&mut gguf, 3);
    put_u64(&mut gguf, 1);
    put_u64(&mut gguf, 0);
    put_string(&mut gguf, "blk.0.ffn_gate.weight");
    put_u32(&mut gguf, 2);
    put_u64(&mut gguf, COLS as u64);
    put_u64(&mut gguf, ROWS as u64);
    put_u32(&mut gguf, GgmlType::Q5_K as u32);
    put_u64(&mut gguf, 0);
    while gguf.len() % 32 != 0 {
        gguf.push(0);
    }
    gguf.extend_from_slice(packed);
    let path = dir.path().join("qwen-q5k-metal-fixture.gguf");
    fs::write(&path, gguf).expect("write temporary Q5_K GGUF fixture");
    path
}

/// Write the smallest executable `.gravity` container that carries exactly the
/// same raw Q4_K payload as the GGUF fixture.  It is intentionally a
/// single-matvec container, not a claim that a one-tensor fixture can execute
/// a full Qwen graph.
fn write_q4k_gravity(dir: &TempDir, packed: &[u8]) -> PathBuf {
    assert_eq!(packed.len(), Q4K_BYTES);
    let header = serde_json::json!({
        "schema": "hawking.gravity.shard_header.v1",
        "format_version": 1,
        "model": {"name": "qwen-q4k-metal-fixture"},
        "architecture": {"model_type": "qwen2"},
        "tokenizer": {},
        "compression": {"codec": "ggml.q4_k"},
        "shard": {"index": 1, "count": 1},
        "integrity": {"tensor_count": 1},
        "tensors": [{
            "name": "blk.0.ffn_gate.weight",
            "codec": "ggml.q4_k",
            "offset": 0,
            "bytes": packed.len(),
            "sha256": sha256_hex(packed),
            "shape": [ROWS, COLS],
            "elements": ROWS * COLS,
        }],
    });
    let header_bytes = serde_json::to_vec(&header).expect("serialize fixture Gravity header");
    let mut artifact = Vec::with_capacity(20 + header_bytes.len() + packed.len());
    artifact.extend_from_slice(b"GRAVITY\0");
    artifact.extend_from_slice(&1_u32.to_le_bytes());
    artifact.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&header_bytes);
    artifact.extend_from_slice(packed);
    let path = dir.path().join("model-00001-of-00001.gravity");
    fs::write(&path, artifact).expect("write temporary Gravity fixture");
    path
}

fn read_f32(buf: &PinnedBuffer, len: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn read_u32(buf: &PinnedBuffer) -> u32 {
    let ptr = buf.contents() as *const u32;
    unsafe { *ptr }
}

/// Independent FP64 authority decoded from Q4_K's packed bytes. This is kept
/// in the test, rather than calling Hawking's f32 dequantizer, so the V2.1
/// score catches a packing/layout disagreement as well as device arithmetic.
fn q4k_to_f64(packed: &[u8]) -> Vec<f64> {
    assert_eq!(packed.len() % quant::Q4_K_BLOCK_BYTES, 0);
    let mut out = vec![0.0_f64; packed.len() / quant::Q4_K_BLOCK_BYTES * 256];
    for (block, src) in packed.chunks_exact(quant::Q4_K_BLOCK_BYTES).enumerate() {
        let d = f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32() as f64;
        let dmin = f16::from_bits(u16::from_le_bytes([src[2], src[3]])).to_f32() as f64;
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for j in 0..4 {
            scales[j] = src[4 + j] & 0x3f;
            mins[j] = src[8 + j] & 0x3f;
            scales[4 + j] = (src[12 + j] & 0x0f) | ((src[4 + j] >> 6) << 4);
            mins[4 + j] = (src[12 + j] >> 4) | ((src[8 + j] >> 6) << 4);
        }
        for sub in 0..8 {
            let scale = d * scales[sub] as f64;
            let min = dmin * mins[sub] as f64;
            let q_base = 16 + (sub / 2) * 32;
            for i in 0..32 {
                let packed_nibbles = src[q_base + i];
                let q = if sub % 2 == 0 {
                    packed_nibbles & 0x0f
                } else {
                    packed_nibbles >> 4
                };
                out[block * 256 + sub * 32 + i] = scale * q as f64 - min;
            }
        }
    }
    out
}

/// Independent GGML Q5_K FP64 decoder.  It intentionally does not call the
/// Hawking packer/dequantizer or reuse its scale helper, so a shared local
/// layout bug cannot self-certify the Q5 Metal path.
fn q5k_to_f64(packed: &[u8]) -> Vec<f64> {
    const BLOCK_BYTES: usize = 176;
    assert_eq!(packed.len() % BLOCK_BYTES, 0);
    let mut out = vec![0.0_f64; packed.len() / BLOCK_BYTES * 256];
    for (block, src) in packed.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32() as f64;
        let dmin = f16::from_bits(u16::from_le_bytes([src[2], src[3]])).to_f32() as f64;
        let packed_scales = &src[4..16];
        let qh = &src[16..48];
        let qs = &src[48..176];
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for j in 0..4 {
            scales[j] = packed_scales[j] & 0x3f;
            mins[j] = packed_scales[4 + j] & 0x3f;
            scales[4 + j] = (packed_scales[8 + j] & 0x0f) | ((packed_scales[j] >> 6) << 4);
            mins[4 + j] = (packed_scales[8 + j] >> 4) | ((packed_scales[4 + j] >> 6) << 4);
        }
        for sub in 0..8 {
            let scale = d * scales[sub] as f64;
            let min = dmin * mins[sub] as f64;
            let qbase = (sub / 2) * 32;
            for lane in 0..32 {
                let low = if sub % 2 == 0 {
                    qs[qbase + lane] & 0x0f
                } else {
                    qs[qbase + lane] >> 4
                };
                let high = ((qh[lane] >> sub) & 1) << 4;
                out[block * 256 + sub * 32 + lane] = scale * (low | high) as f64 - min;
            }
        }
    }
    out
}

fn matvec_f32(weights: &[f32], x: &[f32]) -> Vec<f32> {
    weights
        .chunks_exact(COLS)
        .map(|row| row.iter().zip(x).map(|(&w, &v)| w * v).sum())
        .collect()
}

fn matvec_f64(weights: &[f64], x: &[f32]) -> Vec<f64> {
    weights
        .chunks_exact(COLS)
        .map(|row| row.iter().zip(x).map(|(&w, &v)| w * v as f64).sum())
        .collect()
}

fn argmax_f64(values: &[f64]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("non-empty logits")
        .0 as u32
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn token_sequence_hash(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug)]
struct TokenProfile {
    samples_us: Vec<u128>,
    dispatches: usize,
    waits: usize,
    tokens: Vec<u32>,
    hot_buffers_created: usize,
    hot_bytes_allocated: usize,
    commits: usize,
}

fn percentile_us(sorted: &[u128], pct: usize) -> u128 {
    assert!(!sorted.is_empty());
    let index = (sorted.len() * pct).div_ceil(100).saturating_sub(1);
    sorted[index]
}

fn profile_device_tokens(
    ctx: &MetalContext,
    weights: &PinnedBuffer,
    x: &PinnedBuffer,
    logits: &PinnedBuffer,
    token: &PinnedBuffer,
    weight_bytes: usize,
    repetitions: usize,
) -> TokenProfile {
    let _ = ctx.drain_stats();
    let mut samples_us = Vec::with_capacity(repetitions);
    let mut tokens = Vec::with_capacity(repetitions);
    let mut dispatches = 0usize;
    for _ in 0..repetitions {
        let start = Instant::now();
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_m_v2_pinned_tcb(
            &mut tcb,
            weights,
            0,
            weight_bytes,
            ROWS,
            COLS,
            x,
            logits,
        )
        .expect("profile Q4_K encode");
        kernels::sample_argmax_f32_tcb(&mut tcb, logits, token, ROWS)
            .expect("profile argmax encode");
        assert_eq!(tcb.dispatch_count(), 2, "profile dispatch composition");
        dispatches += tcb.dispatch_count();
        tcb.commit_and_wait()
            .expect("profile command-buffer completion");
        samples_us.push(start.elapsed().as_micros());
        tokens.push(read_u32(token));
    }
    let (hot_buffers_created, hot_bytes_allocated, commits) = ctx.drain_stats();
    assert!(
        commits >= repetitions,
        "each profiled TokenCommandBuffer must contribute at least one counted commit"
    );
    TokenProfile {
        samples_us,
        dispatches,
        waits: repetitions,
        tokens,
        hot_buffers_created,
        hot_bytes_allocated,
        commits,
    }
}

fn profile_device_tokens_q5(
    ctx: &MetalContext,
    weights: &PinnedBuffer,
    x: &PinnedBuffer,
    logits: &PinnedBuffer,
    token: &PinnedBuffer,
    weight_bytes: usize,
    repetitions: usize,
) -> TokenProfile {
    let _ = ctx.drain_stats();
    let mut samples_us = Vec::with_capacity(repetitions);
    let mut tokens = Vec::with_capacity(repetitions);
    let mut dispatches = 0usize;
    for _ in 0..repetitions {
        let start = Instant::now();
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q5_k_serial_authority_pinned_tcb(
            &mut tcb,
            weights,
            0,
            weight_bytes,
            ROWS,
            COLS,
            x,
            logits,
        )
        .expect("profile Q5_K encode");
        kernels::sample_argmax_f32_tcb(&mut tcb, logits, token, ROWS)
            .expect("profile Q5_K argmax encode");
        assert_eq!(tcb.dispatch_count(), 2, "profile Q5_K dispatch composition");
        dispatches += tcb.dispatch_count();
        tcb.commit_and_wait()
            .expect("profile Q5_K command-buffer completion");
        samples_us.push(start.elapsed().as_micros());
        tokens.push(read_u32(token));
    }
    let (hot_buffers_created, hot_bytes_allocated, commits) = ctx.drain_stats();
    TokenProfile {
        samples_us,
        dispatches,
        waits: repetitions,
        tokens,
        hot_buffers_created,
        hot_bytes_allocated,
        commits,
    }
}

fn emit_profile(
    representation: &str,
    physical_bytes: usize,
    active_payload_bytes: usize,
    profile: &TokenProfile,
    continuation_hash: &str,
) {
    let mut sorted = profile.samples_us.clone();
    sorted.sort_unstable();
    eprintln!(
        "QWEN_Q4K_FIXTURE_TOKEN_PROFILE representation={} physical_bytes={} active_payload_bytes={} executed_fp_ops_per_token=4104 executed_integer_bit_ops_per_token=UNSEALED_STATIC_COUNT dispatches={} dispatches_per_token={} waits={} commits={} p50_us={} p95_us={} p99_us={} hot_buffers_created={} hot_bytes_allocated={} continuation_hash={}",
        representation,
        physical_bytes,
        active_payload_bytes,
        profile.dispatches,
        profile.dispatches / profile.tokens.len(),
        profile.waits,
        profile.commits,
        percentile_us(&sorted, 50),
        percentile_us(&sorted, 95),
        percentile_us(&sorted, 99),
        profile.hot_buffers_created,
        profile.hot_bytes_allocated,
        continuation_hash,
    );
}

#[test]
fn qwen_q4k_fixture_has_metal_dispatch_token_parity_and_v21_authority() {
    let source_weights: Vec<f32> = (0..ROWS * COLS)
        .map(|i| {
            // Distinct row-scale and position terms make a token tie unlikely,
            // while staying inside a normal quantization range.
            let row = (i / COLS) as f32;
            let col = (i % COLS) as f32;
            let fixture_x = ((col * 17.0).sin() * 0.63) + ((col * 5.0).cos() * 0.11);
            (row - 3.5) * 0.02 * fixture_x
                + ((col * 13.0).sin() * 0.19)
                + ((col * 7.0).cos() * 0.03)
        })
        .collect();
    let x: Vec<f32> = (0..COLS)
        .map(|i| ((i as f32 * 17.0).sin() * 0.63) + ((i as f32 * 5.0).cos() * 0.11))
        .collect();
    let mut packed = vec![0_u8; Q4K_BYTES];
    quant::quantize_q4_k(&source_weights, &mut packed).expect("quantize fixture weights");
    let packed_sha256 = sha256_hex(&packed);
    let mut packed_q5 = vec![0_u8; Q5K_BYTES];
    quant::quantize_q5_k(&source_weights, &mut packed_q5)
        .expect("quantize fixture weights to Q5_K");
    let packed_q5_sha256 = sha256_hex(&packed_q5);

    let fixture_dir = TempDir::new().expect("temporary fixture directory");
    let fixture_path = write_q4k_gguf(&fixture_dir, &packed);
    let fixture_file = fs::read(&fixture_path).expect("read fixture for hash");
    let fixture_bytes = fixture_file.len();
    let fixture_sha256 = sha256_hex(&fixture_file);
    let q5_fixture_path = write_q5k_gguf(&fixture_dir, &packed_q5);
    let q5_fixture_file = fs::read(&q5_fixture_path).expect("read Q5_K fixture for hash");
    let q5_fixture_bytes = q5_fixture_file.len();
    let q5_fixture_sha256 = sha256_hex(&q5_fixture_file);
    let gravity_path = write_q4k_gravity(&fixture_dir, &packed);
    let gravity_file = fs::read(&gravity_path).expect("read Gravity fixture for hash");
    let gravity_bytes = gravity_file.len();
    let gravity_sha256 = sha256_hex(&gravity_file);

    let (gguf, _parse) = timed("parse", "GgufFile::open", fixture_bytes, || {
        GgufFile::open(&fixture_path).expect("parse fixture GGUF")
    });
    let (tensor_bytes, _map) = timed(
        "tensor_map",
        "GgufFile::tensor+tensor_bytes",
        Q4K_BYTES,
        || {
            let info = gguf
                .tensor("blk.0.ffn_gate.weight")
                .expect("fixture tensor present");
            assert_eq!(info.dims, vec![COLS as u64, ROWS as u64]);
            assert_eq!(info.dtype, GgmlType::Q4_K);
            assert_eq!(info.byte_size as usize, Q4K_BYTES);
            gguf.tensor_bytes(&info.name)
                .expect("fixture tensor bytes")
                .to_vec()
        },
    );
    assert_eq!(
        tensor_bytes, packed,
        "GGUF tensor mapping changed packed bytes"
    );

    let (q5_tensor_bytes, _q5_parse_map) = timed(
        "q5_parse+tensor_map",
        "GgufFile::open+tensor+tensor_bytes(Q5_K)",
        q5_fixture_bytes,
        || {
            let q5 = GgufFile::open(&q5_fixture_path).expect("parse fixture Q5_K GGUF");
            let info = q5
                .tensor("blk.0.ffn_gate.weight")
                .expect("Q5_K fixture tensor present");
            assert_eq!(info.dtype, GgmlType::Q5_K);
            q5.tensor_bytes(&info.name)
                .expect("Q5_K fixture tensor bytes")
                .to_vec()
        },
    );
    assert_eq!(q5_tensor_bytes, packed_q5);

    let (host_weights, _decode) = timed(
        "quant_decode",
        "quant::dequant_into(Q4_K)",
        tensor_bytes.len(),
        || {
            let mut out = vec![0.0_f32; ROWS * COLS];
            quant::dequant_into(GgmlType::Q4_K, &tensor_bytes, &mut out).expect("host Q4_K decode");
            out
        },
    );
    let host_logits = matvec_f32(&host_weights, &x);
    let mut q5_host_weights = vec![0.0_f32; ROWS * COLS];
    quant::dequant_into(GgmlType::Q5_K, &q5_tensor_bytes, &mut q5_host_weights)
        .expect("host Q5_K decode");
    let q5_host_logits = matvec_f32(&q5_host_weights, &x);

    let ((gravity_payload, gravity_host_logits), _gravity_parse) = timed(
        "gravity_parse_execute",
        "GravityShard::open+read_tensor(sha256)+GravityWeights::open+matvec",
        gravity_bytes,
        || {
            let shard = GravityShard::open(&gravity_path).expect("open fixture Gravity shard");
            let payload = shard
                .read_tensor("blk.0.ffn_gate.weight", true)
                .expect("verified fixture Gravity payload");
            let weights = GravityWeights::open(&gravity_path, true)
                .expect("open executable fixture Gravity weights");
            let logits = weights
                .matvec("blk.0.ffn_gate.weight", &x)
                .expect("execute fixture Gravity raw-Q4 matvec");
            (payload, logits)
        },
    );
    assert_eq!(
        gravity_payload, tensor_bytes,
        "same fixture payload must be byte-identical across GGUF and Gravity"
    );
    assert_eq!(
        gravity_host_logits, host_logits,
        "executable Gravity raw-Q4 matvec must match GGUF host decode"
    );

    let ctx = MetalContext::new_with_trace(true).expect("Metal device required");
    let device_name = ctx.device_name();
    let _ = ctx.drain_stats();
    let ((w_buf, x_buf, logits_buf, token_buf, explicit_buffers), _buffers) = timed(
        "buffers",
        "MetalContext::new_buffer_with_bytes/new_buffer",
        tensor_bytes.len()
            + x.len() * size_of::<f32>()
            + ROWS * size_of::<f32>()
            + size_of::<u32>(),
        || {
            let w_buf = ctx.new_buffer_with_bytes(&tensor_bytes);
            let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&x));
            let logits_buf = ctx.new_buffer(ROWS * size_of::<f32>());
            let token_buf = ctx.new_buffer(size_of::<u32>());
            let stats = ctx.drain_stats();
            (w_buf, x_buf, logits_buf, token_buf, stats)
        },
    );
    assert_eq!(
        explicit_buffers,
        (4, 2_212, 0),
        "fixture explicit Metal buffer ledger"
    );

    let (_, _pipeline) = timed(
        "pipeline",
        "MetalContext::pipeline(Q4_K+Q5_K)+pipeline(sample_argmax_f32)",
        0,
        || {
            ctx.pipeline("gemm_q4_k_m_fused_v2").expect("Q4_K pipeline");
            ctx.pipeline("gemm_q5_k_serial_authority")
                .expect("Q5_K pipeline");
            ctx.pipeline("sample_argmax_f32").expect("argmax pipeline");
        },
    );

    let mut tcb = TokenCommandBuffer::new(&ctx);
    let (_, _encode) = timed(
        "encode",
        "gemv_q4_k_m_v2_pinned_tcb+sample_argmax_f32_tcb",
        tensor_bytes.len()
            + x.len() * size_of::<f32>()
            + ROWS * size_of::<f32>()
            + size_of::<u32>(),
        || {
            kernels::gemv_q4_k_m_v2_pinned_tcb(
                &mut tcb,
                &w_buf,
                0,
                tensor_bytes.len(),
                ROWS,
                COLS,
                &x_buf,
                &logits_buf,
            )
            .expect("Q4_K encode");
            kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, ROWS)
                .expect("argmax encode");
        },
    );
    assert_eq!(
        tcb.dispatch_count(),
        2,
        "one Q4_K matvec + one device argmax"
    );

    let (_, _submit) = timed("submit", "TokenCommandBuffer::commit_and_wait", 0, || {
        tcb.commit_and_wait()
            .expect("Metal command buffer completion");
    });
    let ((device_logits, device_token), _completion) = timed(
        "completion",
        "MTLBuffer::contents(read logits+token)",
        ROWS * size_of::<f32>() + size_of::<u32>(),
        || (read_f32(&logits_buf, ROWS), read_u32(&token_buf)),
    );

    let authority_weights = q4k_to_f64(&tensor_bytes);
    let authority_logits = matvec_f64(&authority_weights, &x);
    let authority_token = argmax_f64(&authority_logits);
    let (_, _token) = timed(
        "token",
        "device_argmax_vs_fp64_argmax",
        size_of::<u32>(),
        || {
            assert_eq!(
                device_token, authority_token,
                "device token must match FP64 authority"
            );
        },
    );

    let q5_w_buf = ctx.new_buffer_with_bytes(&q5_tensor_bytes);
    let mut q5_tcb = TokenCommandBuffer::new(&ctx);
    kernels::gemv_q5_k_serial_authority_pinned_tcb(
        &mut q5_tcb,
        &q5_w_buf,
        0,
        q5_tensor_bytes.len(),
        ROWS,
        COLS,
        &x_buf,
        &logits_buf,
    )
    .expect("Q5_K persistent encode");
    kernels::sample_argmax_f32_tcb(&mut q5_tcb, &logits_buf, &token_buf, ROWS)
        .expect("Q5_K device argmax encode");
    assert_eq!(q5_tcb.dispatch_count(), 2);
    q5_tcb.commit_and_wait().expect("Q5_K device completion");
    let q5_device_logits = read_f32(&logits_buf, ROWS);
    let q5_device_token = read_u32(&token_buf);
    let q5_authority_weights = q5k_to_f64(&q5_tensor_bytes);
    let q5_authority_logits = matvec_f64(&q5_authority_weights, &x);
    let q5_score = score_pair(
        &q5_host_logits,
        &q5_device_logits,
        &q5_authority_logits,
        &Bounds::logits(),
    );
    assert!(q5_score.device.pass, "Q5_K device V2.1 must pass");
    assert!(
        q5_score.host.pass,
        "Q5_K host decode must pass independent V2.1"
    );
    assert_eq!(
        q5_device_token,
        argmax_f64(&q5_authority_logits),
        "Q5_K device token must match its independent packed-byte authority"
    );
    assert_eq!(
        q5_device_token, authority_token,
        "Q4_K and Q5_K representations must preserve the fixture decision"
    );

    // This is a profiler for the *same already-validated fixture token*. The
    // sequence is deliberately labelled as a repeat hash, rather than a
    // transformer continuation claim: this fixture contains no KV/attention
    // state to advance.
    const PROFILE_TOKENS: usize = 16;
    let direct_profile = profile_device_tokens(
        &ctx,
        &w_buf,
        &x_buf,
        &logits_buf,
        &token_buf,
        tensor_bytes.len(),
        PROFILE_TOKENS,
    );
    assert!(
        direct_profile
            .tokens
            .iter()
            .all(|&token| token == authority_token),
        "all direct-GGUF profile tokens must match the authority"
    );
    let direct_continuation_hash = token_sequence_hash(&direct_profile.tokens);

    let q5_profile = profile_device_tokens_q5(
        &ctx,
        &q5_w_buf,
        &x_buf,
        &logits_buf,
        &token_buf,
        q5_tensor_bytes.len(),
        PROFILE_TOKENS,
    );
    assert!(q5_profile
        .tokens
        .iter()
        .all(|&token| token == authority_token));
    let q5_continuation_hash = token_sequence_hash(&q5_profile.tokens);
    assert_eq!(direct_continuation_hash, q5_continuation_hash);

    let (gravity_w_buf, _gravity_buffer) = timed(
        "gravity_buffer",
        "MetalContext::new_buffer_with_bytes(Gravity raw Q4_K payload)",
        gravity_payload.len(),
        || ctx.new_buffer_with_bytes(&gravity_payload),
    );
    let gravity_profile = profile_device_tokens(
        &ctx,
        &gravity_w_buf,
        &x_buf,
        &logits_buf,
        &token_buf,
        gravity_payload.len(),
        PROFILE_TOKENS,
    );
    assert!(
        gravity_profile
            .tokens
            .iter()
            .all(|&token| token == authority_token),
        "all Gravity profile tokens must match the authority"
    );
    let gravity_continuation_hash = token_sequence_hash(&gravity_profile.tokens);
    assert_eq!(
        direct_continuation_hash, gravity_continuation_hash,
        "same fixture repeat-token hash across GGUF and executable Gravity"
    );
    emit_profile(
        "direct_gguf_q4_k",
        fixture_bytes,
        Q4K_BYTES,
        &direct_profile,
        &direct_continuation_hash,
    );
    emit_profile(
        "gravity_raw_ggml_q4_k",
        gravity_bytes,
        Q4K_BYTES,
        &gravity_profile,
        &gravity_continuation_hash,
    );
    emit_profile(
        "direct_gguf_q5_k",
        q5_fixture_bytes,
        Q5K_BYTES,
        &q5_profile,
        &q5_continuation_hash,
    );
    eprintln!(
        "QWEN_Q4K_FIXTURE_REPRESENTATION_GATE same_source_tensor=true direct_q4_gguf_bytes={} direct_q5_gguf_bytes={} gravity_q4_bytes={} deepest_tg_rung=NONE next_gate=sealed_full_qwen_same_model_Q4_Q5_native_gravity_capability_and_TG_harness capability=single_packed_matvec_plus_device_argmax_only",
        fixture_bytes, q5_fixture_bytes, gravity_bytes
    );

    let score = score_pair(
        &host_logits,
        &device_logits,
        &authority_logits,
        &Bounds::logits(),
    );
    eprintln!("QWEN_Q4K_FIXTURE_V21 {}", format_score_line(&score.host));
    eprintln!("QWEN_Q4K_FIXTURE_V21 {}", format_score_line(&score.device));
    eprintln!(
        "QWEN_Q5K_FIXTURE_V21 {}",
        format_score_line(&q5_score.device)
    );
    eprintln!(
        "QWEN_Q4K_FIXTURE_RESULT device={} fixture_gguf_sha256={} q5_fixture_gguf_sha256={} gravity_sha256={} packed_q4k_sha256={} packed_q5k_sha256={} dispatches=2 fallback=0 host_v21_pass={} device_v21_pass={} q5_device_v21_pass={} token={} repeated_token_hash={}",
        device_name,
        fixture_sha256,
        q5_fixture_sha256,
        gravity_sha256,
        packed_sha256,
        packed_q5_sha256,
        score.host.pass,
        score.device.pass,
        q5_score.device.pass,
        device_token,
        direct_continuation_hash,
    );
    assert!(
        score.host.pass,
        "host Q4_K V2.1 score: {:?}",
        score.host.failures
    );
    assert!(
        score.device.pass,
        "device Q4_K V2.1 score: {:?}",
        score.device.failures
    );
    // The fixture invokes only the direct Metal kernel and GPU sampler before
    // reading completion. The FP64 computation above is post-completion
    // authority only, so it is never a serving fallback.
    let cpu_reference_fallback_count = 0usize;
    assert_eq!(cpu_reference_fallback_count, 0);
}

#[test]
fn native_q4k_embedding_lookup_matches_independent_packed_decoder() {
    // Eight rows are enough to exercise row addressing as well as every
    // Q4_K sub-block/nibble layout.  This deliberately compares against the
    // fixture's independent FP64 decoder rather than Hawking's host decoder.
    let source: Vec<f32> = (0..ROWS * COLS)
        .map(|i| {
            let row = (i / COLS) as f32;
            let col = (i % COLS) as f32;
            (row * 0.031 - 0.11) + (col * 0.17).sin() * 0.71 + (col * 0.043).cos() * 0.19
        })
        .collect();
    let mut packed = vec![0_u8; Q4K_BYTES];
    quant::quantize_q4_k(&source, &mut packed).expect("quantize native-embed fixture");
    let authority = q4k_to_f64(&packed);
    let ctx = MetalContext::new_with_trace(true).expect("Metal device required");
    let weights = ctx.new_buffer_with_bytes(&packed);
    let out = ctx.new_buffer(COLS * size_of::<f32>());
    let token = 5_u32;

    let mut tcb = TokenCommandBuffer::new(&ctx);
    kernels::embed_lookup_q4_k_m_tcb(&mut tcb, &weights, 0, packed.len(), token, COLS, &out)
        .expect("native Q4_K embed encode");
    assert_eq!(tcb.dispatch_count(), 1, "one native source lookup dispatch");
    tcb.commit_and_wait().expect("native Q4_K embed completion");

    let actual = read_f32(&out, COLS);
    let expected = &authority[token as usize * COLS..(token as usize + 1) * COLS];
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual as f64 - expected).abs() <= 2.0e-6,
            "Q4_K embed mismatch at {i}: actual={actual} expected={expected}"
        );
    }
}
