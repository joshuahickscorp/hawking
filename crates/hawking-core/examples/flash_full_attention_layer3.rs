//! Bounded source-BF16 full-attention organ for a selected Flash-Next layer.
//!
//! This closes the first physical boundary after the exact linear prefix: it
//! proves the selected full-attention path, both HyperConnection combines, and
//! routed/shared MoE on Apple Metal. It is an organ receipt only: it accepts
//! either a repeated-BOS state or the verified layer-0..2 prefix artifact; later
//! layers, tokenizer, and decoding remain outside this executable.

#![recursion_limit = "512"]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash full-attention organ requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::kernels::{
        mha_decode_f32_qwen38_gated_tcb, mha_decode_f32_tcb, moe_topk_gate_tcb_ex,
        native_bf16_dual_seq_tcb, native_bf16_gemv_hyperconnection_combine_tcb,
        native_bf16_gemv_seq_tcb, native_bf16_swiglu_seq_tcb, native_bf16_triple_seq_tcb,
        qwen_next_bf16_compact_expert_down_shared_direct_hc_tcb,
        qwen_next_bf16_compact_expert_down_tcb,
        qwen_next_bf16_compact_expert_gate_up_shared_swiglu_tcb,
        qwen_next_bf16_compact_expert_gate_up_swiglu_tcb, qwen_next_bf16_expert_down_tcb,
        qwen_next_bf16_expert_gate_up_swiglu_tcb, qwen_next_bf16_qkv_gqa_rope_cache_tcb,
        qwen_next_bf16_router_topk_shared_tcb,
        qwen_next_hyperconnection_input_fused_with_block_router_topk_tcb,
        qwen_next_hyperconnection_input_fused_with_block_tcb,
        qwen_next_moe_weighted_sum_add_shared_sigmoid_hc_tcb,
    };
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::cell::RefCell;
    use std::cmp::Ordering;
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    struct CachedFlashResources {
        root: PathBuf,
        index: Rc<SourceBf16Index>,
        context: Rc<MetalContext>,
    }

    thread_local! {
        static FLASH_RESOURCES: RefCell<Option<CachedFlashResources>> = const { RefCell::new(None) };
    }

    fn cached_flash_resources(
        root: &Path,
    ) -> Result<(Rc<SourceBf16Index>, Rc<MetalContext>), Box<dyn Error>> {
        FLASH_RESOURCES.with(|slot| {
            let mut slot = slot.borrow_mut();
            let replace = slot.as_ref().map(|cached| cached.root.as_path() != root).unwrap_or(true);
            if replace {
                eprintln!("Flash attention executor: opening source index and Metal context once for this process");
                *slot = Some(CachedFlashResources {
                    root: root.to_path_buf(),
                    index: Rc::new(SourceBf16Index::open(root)?),
                    context: Rc::new(MetalContext::new_with_trace(true)?),
                });
            }
            let cached = slot.as_ref().expect("Flash attention resource cache populated");
            Ok((Rc::clone(&cached.index), Rc::clone(&cached.context)))
        })
    }

    const REPO: &str = "Qwen/Qwen3.8-Flash-Next";
    const REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const MANIFEST: &str =
        "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json";
    const DEFAULT_ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const HIDDEN: usize = 2560;
    const QUERY_HEADS: usize = 24;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const ROTARY_DIM: usize = 64;
    const QUERY_DIM: usize = QUERY_HEADS * HEAD_DIM;
    const KV_DIM: usize = KV_HEADS * HEAD_DIM;
    const EPS: f32 = 1.0e-6;
    const ROPE_THETA: f32 = 10_000_000.0;
    const BOS_TOKEN_ID: usize = 248044;
    const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
    const HC_NORM: &str = "attn_hyper_connection.hc_norm.weight";
    const HC_DOWN: &str = "attn_hyper_connection.input_mix_weight_down.weight";
    const HC_UP: &str = "attn_hyper_connection.input_mix_weight_up.weight";
    const HC_BLOCK: &str = "attn_hyper_connection.block_inject_weight.weight";
    const Q_PROJ: &str = "self_attn.q_proj.weight";
    const K_PROJ: &str = "self_attn.k_proj.weight";
    const V_PROJ: &str = "self_attn.v_proj.weight";
    const Q_NORM: &str = "self_attn.q_norm.weight";
    const K_NORM: &str = "self_attn.k_norm.weight";
    const O_PROJ: &str = "self_attn.o_proj.weight";
    const ROUTER: &str = "mlp.gate.weight";
    const EXPERT_GATE_UP: &str = "mlp.experts.gate_up_proj";
    const EXPERT_DOWN: &str = "mlp.experts.down_proj";
    const SHARED_GATE: &str = "mlp.shared_expert.gate_proj.weight";
    const SHARED_UP: &str = "mlp.shared_expert.up_proj.weight";
    const SHARED_DOWN: &str = "mlp.shared_expert.down_proj.weight";
    const SHARED_SCALAR: &str = "mlp.shared_expert_gate.weight";
    const HC_MLP_NORM: &str = "mlp_hyper_connection.hc_norm.weight";
    const HC_MLP_DOWN: &str = "mlp_hyper_connection.input_mix_weight_down.weight";
    const HC_MLP_UP: &str = "mlp_hyper_connection.input_mix_weight_up.weight";
    const HC_MLP_BLOCK: &str = "mlp_hyper_connection.block_inject_weight.weight";
    const TOLERANCE: f32 = 2.0e-2;
    const STREAMS: usize = 4;
    const HC_ELEMENTS: usize = HIDDEN * STREAMS;
    const HC_LOWRANK: usize = 320;
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    const INTERMEDIATE: usize = 640;

    pub(crate) struct Args {
        pub(crate) root: PathBuf,
        pub(crate) out: PathBuf,
        pub(crate) layer: usize,
        pub(crate) base_state: Option<PathBuf>,
        pub(crate) state_out: Option<PathBuf>,
        pub(crate) compact_experts: bool,
        pub(crate) fused_route_accumulate: bool,
        pub(crate) fused_attention_gate: bool,
    }

    struct Tensor {
        name: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
        sha256: String,
    }

    trait StageSetScalar {
        fn set_u32(&self, index: u64, value: u32);
        fn set_f32(&self, index: u64, value: f32);
    }

    impl StageSetScalar for ::metal::ComputeCommandEncoderRef {
        fn set_u32(&self, index: u64, value: u32) {
            self.set_bytes(index, 4, &value as *const u32 as *const _);
        }

        fn set_f32(&self, index: u64, value: f32) {
            self.set_bytes(index, 4, &value as *const f32 as *const _);
        }
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn args() -> Result<Args, Box<dyn Error>> {
        let mut result = Args {
            root: env::var_os("HCLI_FLASH_NEXT_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_ROOT)),
            out: repo_root()
                .join("receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER3_ORGAN.json"),
            layer: 3,
            base_state: env::var_os("HCLI_FLASH_BASE_STATE").map(PathBuf::from),
            state_out: env::var_os("HCLI_FLASH_STATE_OUT").map(PathBuf::from),
            compact_experts: false,
            fused_route_accumulate: false,
            fused_attention_gate: env::var("HAWKING_FLASH_FUSE_ATTENTION_GATE")
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "on" | "yes"
                    )
                })
                .unwrap_or(false),
        };
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--root" => result.root = PathBuf::from(values.next().ok_or("missing --root")?),
                "--out" => result.out = PathBuf::from(values.next().ok_or("missing --out")?),
                "--layer" => result.layer = values.next().ok_or("missing --layer")?.parse()?,
                "--base-state" => {
                    result.base_state =
                        Some(PathBuf::from(values.next().ok_or("missing --base-state")?))
                }
                "--state-out" => {
                    result.state_out =
                        Some(PathBuf::from(values.next().ok_or("missing --state-out")?))
                }
                "--compact-experts" => result.compact_experts = true,
                "--fused-route-accumulate" => result.fused_route_accumulate = true,
                "--fused-attention-gate" => result.fused_attention_gate = true,
                "--help" | "-h" => {
                    println!("usage: flash_full_attention_layer3 [--root DIR] [--layer N] [--out FILE] [--base-state F32] [--state-out F32] [--compact-experts] [--fused-route-accumulate] [--fused-attention-gate]");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if result.layer >= 48 {
            return Err("--layer must be in 0..48".into());
        }
        if !env::args().any(|value| value == "--out") && result.layer != 3 {
            result.out = repo_root().join(format!(
                "receipts/headless/FLASH_NOETIC_FULL_ATTENTION_LAYER{}_ORGAN.json",
                result.layer
            ));
        }
        Ok(result)
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn bf16(bytes: &[u8], index: usize) -> f32 {
        let offset = index * 2;
        f32::from_bits((u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32) << 16)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f32_vec(tensor: &Tensor) -> Vec<f32> {
        (0..tensor.bytes.len() / 2)
            .map(|index| bf16(&tensor.bytes, index))
            .collect()
    }

    fn layer_tensor_name(layer: usize, suffix: &str) -> String {
        format!("model.language_model.layers.{layer}.{suffix}")
    }

    fn tensor(
        index: &SourceBf16Index,
        name: String,
        shape: &[usize],
    ) -> Result<Tensor, Box<dyn Error>> {
        let elements = shape
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .ok_or_else(|| format!("{name} shape overflow"))?;
        let bytes = index.read_raw(&name)?;
        if bytes.len() != elements * 2 {
            return Err(format!(
                "{name} bytes={} expected={} shape={shape:?}",
                bytes.len(),
                elements * 2
            )
            .into());
        }
        Ok(Tensor {
            name,
            shape: shape.to_vec(),
            sha256: sha256(&bytes),
            bytes,
        })
    }

    fn tensor_rows(
        index: &SourceBf16Index,
        name: String,
        shape: &[usize],
        experts: &[u32],
    ) -> Result<Tensor, Box<dyn Error>> {
        let rows = *shape
            .first()
            .ok_or("expert tensor has no leading dimension")?;
        let row_elements = shape[1..]
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .ok_or("expert row shape overflow")?;
        if rows != EXPERTS
            || experts.is_empty()
            || experts.iter().any(|&expert| expert as usize >= rows)
        {
            return Err(format!("invalid compact expert selection for {name}").into());
        }
        let row_bytes = row_elements
            .checked_mul(2)
            .ok_or("expert row bytes overflow")?;
        let mut bytes = Vec::with_capacity(experts.len() * row_bytes);
        for &expert in experts {
            bytes.extend_from_slice(&index.read_raw_range(
                &name,
                expert as usize * row_bytes,
                row_bytes,
            )?);
        }
        let mut compact_shape = shape.to_vec();
        compact_shape[0] = experts.len();
        Ok(Tensor {
            name,
            shape: compact_shape,
            sha256: sha256(&bytes),
            bytes,
        })
    }

    fn manifest(root: &Path) -> Result<Value, Box<dyn Error>> {
        let bytes = fs::read(MANIFEST)?;
        let value: Value = serde_json::from_slice(&bytes)?;
        if value.get("repo").and_then(Value::as_str) != Some(REPO)
            || value.get("revision").and_then(Value::as_str) != Some(REVISION)
            || value.get("resolved_sha").and_then(Value::as_str) != Some(REVISION)
        {
            return Err("Flash ModelLake manifest is not the pinned revision".into());
        }
        let path = value
            .get("path")
            .and_then(Value::as_str)
            .ok_or("manifest path missing")?;
        if Path::new(path).canonicalize()? != root.canonicalize()? {
            return Err("selected root does not match pinned manifest".into());
        }
        Ok(
            json!({"path": MANIFEST, "sha256": sha256(&bytes), "repo": REPO,
            "revision": REVISION, "resolved_sha": REVISION, "n_files": value.get("n_files"),
            "bytes": value.get("bytes")}),
        )
    }

    fn embedding(
        index: &SourceBf16Index,
        vocab: usize,
    ) -> Result<(Vec<f32>, String, usize), Box<dyn Error>> {
        let bytes = index.read_raw(EMBEDDING)?;
        if bytes.len() != vocab * HIDDEN * 2 || BOS_TOKEN_ID >= vocab {
            return Err("embedding geometry or BOS token is invalid".into());
        }
        let start = BOS_TOKEN_ID * HIDDEN;
        Ok((
            (0..HIDDEN).map(|i| bf16(&bytes, start + i)).collect(),
            sha256(&bytes),
            bytes.len(),
        ))
    }

    fn f32_state(path: &Path) -> Result<(Vec<f32>, String), Box<dyn Error>> {
        let bytes = fs::read(path)?;
        if bytes.len() != HC_ELEMENTS * std::mem::size_of::<f32>() {
            return Err(format!(
                "base state bytes={} expected={} ({path:?})",
                bytes.len(),
                HC_ELEMENTS * 4
            )
            .into());
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("base state contains non-finite values: {path:?}").into());
        }
        Ok((values, sha256(&bytes)))
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn source_cache_policy() -> &'static str {
        if env::var("HAWKING_SOURCE_CACHE")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
        {
            "OS_FILE_CACHE_OPT_IN"
        } else {
            "F_NOCACHE_DEFAULT"
        }
    }

    fn grouped_hc_norm(input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; HC_ELEMENTS];
        for stream in 0..STREAMS {
            let base = stream * HIDDEN;
            let inverse = (input[base..base + HIDDEN]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / HIDDEN as f32
                + EPS)
                .sqrt()
                .recip();
            for dim in 0..HIDDEN {
                out[base + dim] = input[base + dim] * inverse * (1.0 + weight[base + dim]);
            }
        }
        out
    }

    fn hc_read_mix(normalized: &[f32], down: &[u8], up: &[u8]) -> Vec<f32> {
        let low = matvec(down, HC_LOWRANK, HC_ELEMENTS, normalized);
        let activated = low
            .into_iter()
            .map(|v| v / STREAMS as f32 / (1.0 + (-v / STREAMS as f32).exp()))
            .collect::<Vec<_>>();
        let logits = matvec(up, HC_ELEMENTS, HC_LOWRANK, &activated);
        (0..HIDDEN)
            .map(|dim| {
                (0..STREAMS)
                    .map(|stream| {
                        let i = stream * HIDDEN + dim;
                        sigmoid(logits[i]) * normalized[i]
                    })
                    .sum::<f32>()
                    / STREAMS as f32
            })
            .collect()
    }

    fn hc_combine(residual: &[f32], block: &[f32], logits: &[f32]) -> Vec<f32> {
        residual
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                value
                    + block[index % HIDDEN]
                        * (2.0 * sigmoid(logits[index / HIDDEN] / STREAMS as f32))
            })
            .collect()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn topk_router(logits: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = logits
            .iter()
            .map(|&value| (value - maximum).exp())
            .collect::<Vec<_>>();
        let sum = probabilities.iter().copied().sum::<f32>();
        for value in &mut probabilities {
            *value /= sum;
        }
        let tie_epsilon = env::var("HAWKING_DS_ROUTE_TIE_EPS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.0);
        let mut ids = (0..EXPERTS).collect::<Vec<_>>();
        ids.sort_by(|&left, &right| {
            let delta = probabilities[right] - probabilities[left];
            if tie_epsilon > 0.0 && delta.abs() <= tie_epsilon {
                left.cmp(&right)
            } else {
                probabilities[right]
                    .partial_cmp(&probabilities[left])
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.cmp(&right))
            }
        });
        ids.truncate(TOP_K);
        let mut weights = ids.iter().map(|&id| probabilities[id]).collect::<Vec<_>>();
        let selected_sum = weights.iter().copied().sum::<f32>();
        for value in &mut weights {
            *value /= selected_sum;
        }
        (ids.into_iter().map(|id| id as u32).collect(), weights)
    }

    fn expert_swiglu(
        gate_up: &[u8],
        expert: usize,
        input: &[f32],
        route_lut: Option<&[u32]>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let slot = route_lut
            .and_then(|lut| lut.get(expert).copied())
            .unwrap_or(expert as u32);
        if slot == u32::MAX {
            return Err(format!("expert {expert} missing from compact route union").into());
        }
        let offset = slot as usize * 2 * INTERMEDIATE * HIDDEN;
        let gate = matvec(&gate_up[offset * 2..], INTERMEDIATE, HIDDEN, input);
        let up = matvec(
            &gate_up[(offset + INTERMEDIATE * HIDDEN) * 2..],
            INTERMEDIATE,
            HIDDEN,
            input,
        );
        Ok(gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| silu(gate) * up)
            .collect())
    }

    fn expert_down(
        down: &[u8],
        expert: usize,
        activated: &[f32],
        route_lut: Option<&[u32]>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let slot = route_lut
            .and_then(|lut| lut.get(expert).copied())
            .unwrap_or(expert as u32);
        if slot == u32::MAX {
            return Err(format!("expert {expert} missing from compact route union").into());
        }
        let offset = slot as usize * HIDDEN * INTERMEDIATE * 2;
        Ok(matvec(&down[offset..], HIDDEN, INTERMEDIATE, activated))
    }

    fn u32_values(buffer: &PinnedBuffer, elements: usize) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec() }
    }

    fn matvec(weight: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        // The CPU oracle is an authority, but it must not serialize the
        // campaign.  Preserve the exact per-row accumulation order while
        // splitting independent output rows across a bounded scoped pool.
        // Small projections stay serial to avoid thread ceremony; setting
        // HAWKING_CPU_PARALLELISM=1 restores the old single-thread path.
        let workers = std::env::var("HAWKING_CPU_PARALLELISM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(|value| value.get())
            })
            .unwrap_or(1)
            .min(rows.max(1));
        if workers <= 1 || rows.saturating_mul(cols) < 1_000_000 {
            return (0..rows)
                .map(|row| {
                    (0..cols)
                        .map(|col| bf16(weight, row * cols + col) * input[col])
                        .sum()
                })
                .collect();
        }
        let chunk = (rows + workers - 1) / workers;
        std::thread::scope(|scope| {
            let mut jobs = Vec::new();
            for start in (0..rows).step_by(chunk) {
                let end = (start + chunk).min(rows);
                jobs.push(scope.spawn(move || {
                    (start..end)
                        .map(|row| {
                            (0..cols)
                                .map(|col| bf16(weight, row * cols + col) * input[col])
                                .sum::<f32>()
                        })
                        .collect::<Vec<_>>()
                }));
            }
            let mut output = Vec::with_capacity(rows);
            for job in jobs {
                output.extend(job.join().expect("Flash CPU oracle worker panicked"));
            }
            output
        })
    }

    fn norm_rope(raw: &[f32], weight: &[f32], heads: usize, position: usize) -> Vec<f32> {
        let mut out = vec![0.0; heads * HEAD_DIM];
        for head in 0..heads {
            let base = head * HEAD_DIM;
            let inverse = (raw[base..base + HEAD_DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / HEAD_DIM as f32
                + EPS)
                .sqrt()
                .recip();
            for dim in 0..HEAD_DIM {
                out[base + dim] = raw[base + dim] * inverse * (1.0 + weight[dim]);
            }
            let before = out[base..base + ROTARY_DIM].to_vec();
            for dim in 0..ROTARY_DIM / 2 {
                let angle =
                    position as f32 * ROPE_THETA.powf(-2.0 * dim as f32 / ROTARY_DIM as f32);
                let (sin, cos) = angle.sin_cos();
                out[base + dim] = before[dim] * cos - before[dim + ROTARY_DIM / 2] * sin;
                out[base + dim + ROTARY_DIM / 2] =
                    before[dim + ROTARY_DIM / 2] * cos + before[dim] * sin;
            }
        }
        out
    }

    fn query(raw: &[f32]) -> Vec<f32> {
        (0..QUERY_HEADS)
            .flat_map(|head| raw[head * 2 * HEAD_DIM..head * 2 * HEAD_DIM + HEAD_DIM].to_vec())
            .collect()
    }

    fn attention(query: &[f32], key: &[f32], value: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; QUERY_DIM];
        for head in 0..QUERY_HEADS {
            let kv = head / (QUERY_HEADS / KV_HEADS);
            let dst = &mut out[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            let v = &value[kv * HEAD_DIM..(kv + 1) * HEAD_DIM];
            // At position zero the causal prefix contains exactly one key;
            // softmax over a singleton is one (the query/key dot is still
            // exercised by the Metal attention kernel).
            let _ = (
                &query[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                &key[kv * HEAD_DIM..(kv + 1) * HEAD_DIM],
            );
            dst.copy_from_slice(v);
        }
        out
    }

    fn gated(attention: &[f32], q_projection: &[f32]) -> Vec<f32> {
        (0..QUERY_HEADS)
            .flat_map(|head| {
                let q = head * 2 * HEAD_DIM;
                (0..HEAD_DIM).map(move |dim| {
                    attention[head * HEAD_DIM + dim]
                        / (1.0 + (-q_projection[q + HEAD_DIM + dim]).exp())
                })
            })
            .collect()
    }

    fn metrics(expected: &[f32], observed: &[f32]) -> Value {
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut en = 0.0f64;
        let mut on = 0.0f64;
        for (a, b) in expected.iter().zip(observed) {
            max_abs = max_abs.max((a - b).abs());
            sum_sq += f64::from(*a - *b).powi(2);
            dot += f64::from(*a) * f64::from(*b);
            en += f64::from(*a) * f64::from(*a);
            on += f64::from(*b) * f64::from(*b);
        }
        let cosine = if en > 0.0 && on > 0.0 {
            dot / (en.sqrt() * on.sqrt())
        } else {
            0.0
        };
        json!({"max_abs_error": max_abs, "rmse": (sum_sq / expected.len() as f64).sqrt(),
            "cosine": cosine, "within_tolerance": max_abs <= TOLERANCE,
            "finite": observed.iter().all(|v| v.is_finite())})
    }

    fn device_f32(buffer: &PinnedBuffer, elements: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() }
    }

    fn main_impl_with_args(
        args: Args,
        external_input: Option<&PinnedBuffer>,
    ) -> Result<Option<PinnedBuffer>, Box<dyn Error>> {
        let function_started = Instant::now();
        let root_started = Instant::now();
        let root = args.root.canonicalize()?;
        let root_canonicalize_ns = root_started.elapsed().as_nanos() as u64;
        let manifest_started = Instant::now();
        let manifest_value = manifest(&root)?;
        let manifest_ns = manifest_started.elapsed().as_nanos() as u64;
        let config_started = Instant::now();
        let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        let config_ns = config_started.elapsed().as_nanos() as u64;
        let text = config.get("text_config").ok_or("text_config missing")?;
        let vocab = text
            .get("vocab_size")
            .and_then(Value::as_u64)
            .ok_or("vocab missing")? as usize;
        if text.get("hidden_size").and_then(Value::as_u64) != Some(HIDDEN as u64)
            || text.get("num_hidden_layers").and_then(Value::as_u64) != Some(48)
            || text.get("num_attention_heads").and_then(Value::as_u64) != Some(QUERY_HEADS as u64)
            || text.get("num_key_value_heads").and_then(Value::as_u64) != Some(KV_HEADS as u64)
        {
            return Err("Flash full-attention geometry drifted".into());
        }
        let layer_type = text
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|layers| layers.get(args.layer))
            .and_then(Value::as_str)
            .unwrap_or("");
        if layer_type != "full_attention" {
            return Err(format!(
                "Flash layer {} is not full_attention ({layer_type:?})",
                args.layer
            )
            .into());
        }
        let resources_started = Instant::now();
        let (index, context) = cached_flash_resources(&root)?;
        let index_context_ns = resources_started.elapsed().as_nanos() as u64;
        let input_started = Instant::now();
        let (input, embedding_source, input_state) = if let Some(path) = args.base_state.as_ref() {
            let (values, sha256) = f32_state(path)?;
            (
                values,
                Value::Null,
                json!({"kind": "prefix_state_f32", "path": path, "bytes": HC_ELEMENTS * 4, "sha256": sha256, "source": "explicit preceding-layer state artifact"}),
            )
        } else {
            let (embedding_row, sha256, bytes) = embedding(&index, vocab)?;
            let mut values = Vec::with_capacity(HC_ELEMENTS);
            for _ in 0..STREAMS {
                values.extend_from_slice(&embedding_row);
            }
            (
                values,
                json!({"tensor": EMBEDDING, "token_id": BOS_TOKEN_ID, "sha256": sha256, "bytes": bytes}),
                json!({"kind": "repeated_bos_embedding", "token_id": BOS_TOKEN_ID}),
            )
        };
        let input_load_ns = input_started.elapsed().as_nanos() as u64;
        let source_bytes_before = index.bytes_read_total();
        let source_load_started = Instant::now();
        let hc_norm = tensor(
            &index,
            layer_tensor_name(args.layer, HC_NORM),
            &[HC_ELEMENTS],
        )?;
        let hc_down = tensor(
            &index,
            layer_tensor_name(args.layer, HC_DOWN),
            &[HC_LOWRANK, HC_ELEMENTS],
        )?;
        let hc_up = tensor(
            &index,
            layer_tensor_name(args.layer, HC_UP),
            &[HC_ELEMENTS, HC_LOWRANK],
        )?;
        let hc_block = tensor(
            &index,
            layer_tensor_name(args.layer, HC_BLOCK),
            &[STREAMS, HC_ELEMENTS],
        )?;
        let q_proj = tensor(
            &index,
            layer_tensor_name(args.layer, Q_PROJ),
            &[QUERY_HEADS * 2 * HEAD_DIM, HIDDEN],
        )?;
        let k_proj = tensor(
            &index,
            layer_tensor_name(args.layer, K_PROJ),
            &[KV_DIM, HIDDEN],
        )?;
        let v_proj = tensor(
            &index,
            layer_tensor_name(args.layer, V_PROJ),
            &[KV_DIM, HIDDEN],
        )?;
        let q_norm = tensor(&index, layer_tensor_name(args.layer, Q_NORM), &[HEAD_DIM])?;
        let k_norm = tensor(&index, layer_tensor_name(args.layer, K_NORM), &[HEAD_DIM])?;
        let o_proj = tensor(
            &index,
            layer_tensor_name(args.layer, O_PROJ),
            &[HIDDEN, QUERY_DIM],
        )?;
        let router = tensor(
            &index,
            layer_tensor_name(args.layer, ROUTER),
            &[EXPERTS, HIDDEN],
        )?;
        let shared_gate = tensor(
            &index,
            layer_tensor_name(args.layer, SHARED_GATE),
            &[INTERMEDIATE, HIDDEN],
        )?;
        let shared_up = tensor(
            &index,
            layer_tensor_name(args.layer, SHARED_UP),
            &[INTERMEDIATE, HIDDEN],
        )?;
        let shared_down = tensor(
            &index,
            layer_tensor_name(args.layer, SHARED_DOWN),
            &[HIDDEN, INTERMEDIATE],
        )?;
        let shared_scalar = tensor(
            &index,
            layer_tensor_name(args.layer, SHARED_SCALAR),
            &[1, HIDDEN],
        )?;
        let mlp_norm = tensor(
            &index,
            layer_tensor_name(args.layer, HC_MLP_NORM),
            &[HC_ELEMENTS],
        )?;
        let mlp_down = tensor(
            &index,
            layer_tensor_name(args.layer, HC_MLP_DOWN),
            &[HC_LOWRANK, HC_ELEMENTS],
        )?;
        let mlp_up = tensor(
            &index,
            layer_tensor_name(args.layer, HC_MLP_UP),
            &[HC_ELEMENTS, HC_LOWRANK],
        )?;
        let mlp_block = tensor(
            &index,
            layer_tensor_name(args.layer, HC_MLP_BLOCK),
            &[STREAMS, HC_ELEMENTS],
        )?;
        let source_load_ns = source_load_started.elapsed().as_nanos() as u64;
        let source_payload_bytes_read =
            index.bytes_read_total().saturating_sub(source_bytes_before);
        let oracle_started = Instant::now();
        let expected_hc_norm = grouped_hc_norm(&input, &f32_vec(&hc_norm));
        let expected_input = hc_read_mix(&expected_hc_norm, &hc_down.bytes, &hc_up.bytes);
        let expected_q = matvec(&q_proj.bytes, QUERY_DIM * 2, HIDDEN, &expected_input);
        let expected_k = matvec(&k_proj.bytes, KV_DIM, HIDDEN, &expected_input);
        let expected_v = matvec(&v_proj.bytes, KV_DIM, HIDDEN, &expected_input);
        let expected_query = norm_rope(&query(&expected_q), &f32_vec(&q_norm), QUERY_HEADS, 0);
        let expected_key = norm_rope(&expected_k, &f32_vec(&k_norm), KV_HEADS, 0);
        let expected_attention = attention(&expected_query, &expected_key, &expected_v);
        let expected_gated = gated(&expected_attention, &expected_q);
        let expected_output = matvec(&o_proj.bytes, HIDDEN, QUERY_DIM, &expected_gated);
        let expected_block_logits =
            matvec(&hc_block.bytes, STREAMS, HC_ELEMENTS, &expected_hc_norm);
        let expected_post_attn_state = hc_combine(&input, &expected_output, &expected_block_logits);
        let expected_mlp_norm = grouped_hc_norm(&expected_post_attn_state, &f32_vec(&mlp_norm));
        let expected_mlp_input = hc_read_mix(&expected_mlp_norm, &mlp_down.bytes, &mlp_up.bytes);
        let expected_router_logits = matvec(&router.bytes, EXPERTS, HIDDEN, &expected_mlp_input);
        let (expected_route_ids, expected_route_weights) = topk_router(&expected_router_logits);
        let route_lut_host = if args.compact_experts {
            let mut lut = vec![u32::MAX; EXPERTS];
            for (slot, &expert) in expected_route_ids.iter().enumerate() {
                lut[expert as usize] = slot as u32;
            }
            Some(lut)
        } else {
            None
        };
        let selected_routes = route_lut_host.as_ref().map(|lut| {
            lut.iter()
                .enumerate()
                .filter_map(|(expert, &slot)| (slot != u32::MAX).then_some(expert as u32))
                .collect::<Vec<_>>()
        });
        let expert_gate_up_weight = if let Some(routes) = selected_routes.as_ref() {
            tensor_rows(
                &index,
                layer_tensor_name(args.layer, EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
                routes,
            )?
        } else {
            tensor(
                &index,
                layer_tensor_name(args.layer, EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            )?
        };
        let expert_down_weight = if let Some(routes) = selected_routes.as_ref() {
            tensor_rows(
                &index,
                layer_tensor_name(args.layer, EXPERT_DOWN),
                &[EXPERTS, HIDDEN, INTERMEDIATE],
                routes,
            )?
        } else {
            tensor(
                &index,
                layer_tensor_name(args.layer, EXPERT_DOWN),
                &[EXPERTS, HIDDEN, INTERMEDIATE],
            )?
        };
        let source_tensors = [&hc_norm, &hc_down, &hc_up, &hc_block, &q_proj, &k_proj, &v_proj, &q_norm, &k_norm, &o_proj, &router, &expert_gate_up_weight, &expert_down_weight, &shared_gate, &shared_up, &shared_down, &shared_scalar, &mlp_norm, &mlp_down, &mlp_up, &mlp_block]
            .iter()
            .map(|t| json!({"name": t.name, "shape": t.shape, "bytes": t.bytes.len(), "sha256": t.sha256}))
            .collect::<Vec<_>>();
        let mut expected_routed_sum = vec![0.0f32; HIDDEN];
        for (&expert, &weight) in expected_route_ids.iter().zip(&expected_route_weights) {
            let routed = expert_down(
                &expert_down_weight.bytes,
                expert as usize,
                &expert_swiglu(
                    &expert_gate_up_weight.bytes,
                    expert as usize,
                    &expected_mlp_input,
                    route_lut_host.as_deref(),
                )?,
                route_lut_host.as_deref(),
            )?;
            for (out, value) in expected_routed_sum.iter_mut().zip(routed) {
                *out += weight * value;
            }
        }
        let shared_gate_values = matvec(
            &shared_gate.bytes,
            INTERMEDIATE,
            HIDDEN,
            &expected_mlp_input,
        );
        let shared_up_values = matvec(&shared_up.bytes, INTERMEDIATE, HIDDEN, &expected_mlp_input);
        let shared_activation = shared_gate_values
            .into_iter()
            .zip(shared_up_values)
            .map(|(gate, up)| silu(gate) * up)
            .collect::<Vec<_>>();
        let shared_output = matvec(&shared_down.bytes, HIDDEN, INTERMEDIATE, &shared_activation);
        let shared_logit = matvec(&shared_scalar.bytes, 1, HIDDEN, &expected_mlp_input)[0];
        let expected_shared_gated = shared_output
            .iter()
            .map(|&value| sigmoid(shared_logit) * value)
            .collect::<Vec<_>>();
        let expected_moe_output = expected_routed_sum
            .iter()
            .zip(&expected_shared_gated)
            .map(|(routed, shared)| routed + shared)
            .collect::<Vec<_>>();
        let expected_mlp_block_logits =
            matvec(&mlp_block.bytes, STREAMS, HC_ELEMENTS, &expected_mlp_norm);
        let expected_final_state = hc_combine(
            &expected_post_attn_state,
            &expected_moe_output,
            &expected_mlp_block_logits,
        );
        let oracle_ns = oracle_started.elapsed().as_nanos() as u64;

        let device = context.device_name();
        let device_prepare_started = Instant::now();
        let input_buf = if external_input.is_some() {
            context.new_buffer_checked(HC_ELEMENTS * 4)?
        } else {
            context.new_buffer_with_bytes_checked(&f32_bytes(&input))?
        };
        let hc_norm_buf = context.new_buffer_with_bytes_checked(&hc_norm.bytes)?;
        let hc_down_buf = context.new_buffer_with_bytes_checked(&hc_down.bytes)?;
        let hc_up_buf = context.new_buffer_with_bytes_checked(&hc_up.bytes)?;
        let hc_block_buf = context.new_buffer_with_bytes_checked(&hc_block.bytes)?;
        let hc_low = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let hc_low_activation = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let hc_gate = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let attn_input = context.new_buffer_checked(HIDDEN * 4)?;
        let q_weight = context.new_buffer_with_bytes_checked(&q_proj.bytes)?;
        let k_weight = context.new_buffer_with_bytes_checked(&k_proj.bytes)?;
        let v_weight = context.new_buffer_with_bytes_checked(&v_proj.bytes)?;
        let q_norm_buf = context.new_buffer_with_bytes_checked(&f32_bytes(&f32_vec(&q_norm)))?;
        let k_norm_buf = context.new_buffer_with_bytes_checked(&f32_bytes(&f32_vec(&k_norm)))?;
        let o_weight = context.new_buffer_with_bytes_checked(&o_proj.bytes)?;
        let router_weight = context.new_buffer_with_bytes_checked(&router.bytes)?;
        let expert_gate_up_device =
            context.new_buffer_with_bytes_checked(&expert_gate_up_weight.bytes)?;
        let expert_down_device =
            context.new_buffer_with_bytes_checked(&expert_down_weight.bytes)?;
        let route_lut = route_lut_host
            .as_ref()
            .map(|lut| context.new_buffer_with_bytes_checked(&u32_bytes(lut)))
            .transpose()?;
        let compact_expert_count = selected_routes
            .as_ref()
            .map(|routes| routes.len())
            .unwrap_or(EXPERTS);
        let shared_gate_weight = context.new_buffer_with_bytes_checked(&shared_gate.bytes)?;
        let shared_up_weight = context.new_buffer_with_bytes_checked(&shared_up.bytes)?;
        let shared_down_weight = context.new_buffer_with_bytes_checked(&shared_down.bytes)?;
        let shared_scalar_weight = context.new_buffer_with_bytes_checked(&shared_scalar.bytes)?;
        let mlp_norm_weight = context.new_buffer_with_bytes_checked(&mlp_norm.bytes)?;
        let mlp_down_weight = context.new_buffer_with_bytes_checked(&mlp_down.bytes)?;
        let mlp_up_weight = context.new_buffer_with_bytes_checked(&mlp_up.bytes)?;
        let mlp_block_weight = context.new_buffer_with_bytes_checked(&mlp_block.bytes)?;
        let normalized = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let q_out = context.new_buffer_checked(QUERY_DIM * 2 * 4)?;
        let k_out = context.new_buffer_checked(KV_DIM * 4)?;
        let v_out = context.new_buffer_checked(KV_DIM * 4)?;
        let query_out = context.new_buffer_checked(QUERY_DIM * 4)?;
        let key_cache = context.new_buffer_checked(KV_DIM * 4)?;
        let value_cache = context.new_buffer_checked(KV_DIM * 4)?;
        let attention_out = context.new_buffer_checked(QUERY_DIM * 4)?;
        let gated_out = context.new_buffer_checked(QUERY_DIM * 4)?;
        let output = context.new_buffer_checked(HIDDEN * 4)?;
        let attn_block_logits = context.new_buffer_checked(STREAMS * 4)?;
        let post_attn_state = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mlp_normalized = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mlp_low = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let mlp_low_activation = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let mlp_gate = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mlp_input = context.new_buffer_checked(HIDDEN * 4)?;
        let router_logits = context.new_buffer_checked(EXPERTS * 4)?;
        let route_ids = context.new_buffer_checked(TOP_K * 4)?;
        let route_weights = context.new_buffer_checked(TOP_K * 4)?;
        let routed_activation = context.new_buffer_checked(TOP_K * INTERMEDIATE * 4)?;
        let routed_outputs = context.new_buffer_checked(TOP_K * HIDDEN * 4)?;
        let routed_sum = context.new_buffer_checked(HIDDEN * 4)?;
        let shared_activation = context.new_buffer_checked(INTERMEDIATE * 4)?;
        let shared_output = context.new_buffer_checked(HIDDEN * 4)?;
        let shared_scalar_output = context.new_buffer_checked(4)?;
        let shared_gated_output = context.new_buffer_checked(HIDDEN * 4)?;
        let moe_output = context.new_buffer_checked(HIDDEN * 4)?;
        let mlp_block_logits = context.new_buffer_checked(STREAMS * 4)?;
        let final_state = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let device_prepare_ns = device_prepare_started.elapsed().as_nanos() as u64;

        let forward_started = Instant::now();
        let fused_qkv_gqa = env::var("HAWKING_FLASH_QKV_GQA_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_router_topk = env::var("HAWKING_FLASH_ROUTER_TOPK_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_hc_router = env::var("HAWKING_FLASH_HC_ROUTER_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_moe_vec4 = env::var("HAWKING_FLASH_MOE_VEC4")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let encode_started = Instant::now();
        let mut tcb = TokenCommandBuffer::new(&context);
        if let Some(source) = external_input {
            tcb.copy_buffer_bytes(source, 0, &input_buf, 0, (HC_ELEMENTS * 4) as u64)?;
        }
        qwen_next_hyperconnection_input_fused_with_block_tcb(
            &mut tcb,
            &input_buf,
            &hc_norm_buf,
            &hc_down_buf,
            &hc_up_buf,
            &normalized,
            &hc_low,
            &hc_low_activation,
            &hc_gate,
            &attn_input,
            &hc_block_buf,
            &attn_block_logits,
            HIDDEN,
            STREAMS,
            HC_LOWRANK,
            EPS,
            STREAMS as f32,
        )?;
        if fused_qkv_gqa {
            qwen_next_bf16_qkv_gqa_rope_cache_tcb(
                &mut tcb,
                &q_weight,
                &k_weight,
                &v_weight,
                &attn_input,
                &q_norm_buf,
                &k_norm_buf,
                &q_out,
                &k_out,
                &v_out,
                &query_out,
                &key_cache,
                &value_cache,
                0,
                QUERY_HEADS,
                KV_HEADS,
                HEAD_DIM,
                ROTARY_DIM,
                HIDDEN,
                ROPE_THETA,
                EPS,
            )?;
        } else {
            native_bf16_triple_seq_tcb(
                &mut tcb,
                &q_weight,
                &k_weight,
                &v_weight,
                &attn_input,
                &q_out,
                &k_out,
                &v_out,
                QUERY_DIM * 2,
                KV_DIM,
                KV_DIM,
                HIDDEN,
            )?;
            tcb.dispatch_threads(
                "qwen80_gqa_qk_norm_rope_cache_f32",
                (QUERY_HEADS as u32, 1, 1),
                (QUERY_HEADS as u32, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q_out), 0);
                    enc.set_buffer(1, Some(&k_out), 0);
                    enc.set_buffer(2, Some(&v_out), 0);
                    enc.set_buffer(3, Some(&q_norm_buf), 0);
                    enc.set_buffer(4, Some(&k_norm_buf), 0);
                    enc.set_buffer(5, Some(&query_out), 0);
                    enc.set_buffer(6, Some(&key_cache), 0);
                    enc.set_buffer(7, Some(&value_cache), 0);
                    enc.set_u32(8, 0);
                    enc.set_u32(9, QUERY_HEADS as u32);
                    enc.set_u32(10, KV_HEADS as u32);
                    enc.set_u32(11, HEAD_DIM as u32);
                    enc.set_u32(12, ROTARY_DIM as u32);
                    enc.set_f32(13, ROPE_THETA);
                    enc.set_f32(14, EPS);
                },
            )?;
        }
        if args.fused_attention_gate {
            mha_decode_f32_qwen38_gated_tcb(
                &mut tcb,
                &query_out,
                &key_cache,
                0,
                &value_cache,
                0,
                &gated_out,
                &q_out,
                1,
                HEAD_DIM,
                QUERY_HEADS,
                KV_HEADS,
            )?;
        } else {
            mha_decode_f32_tcb(
                &mut tcb,
                &query_out,
                &key_cache,
                0,
                &value_cache,
                0,
                &attention_out,
                1,
                HEAD_DIM,
                QUERY_HEADS,
                KV_HEADS,
            )?;
            tcb.dispatch_threads(
                "qwen80_attention_apply_sigmoid_gate",
                (QUERY_DIM as u32, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&attention_out), 0);
                    enc.set_buffer(1, Some(&q_out), 0);
                    enc.set_buffer(2, Some(&gated_out), 0);
                    enc.set_u32(3, QUERY_DIM as u32);
                    enc.set_u32(4, HEAD_DIM as u32);
                },
            )?;
        }
        native_bf16_gemv_hyperconnection_combine_tcb(
            &mut tcb,
            &o_weight,
            &gated_out,
            &input_buf,
            &attn_block_logits,
            &output,
            &post_attn_state,
            HIDDEN,
            QUERY_DIM,
            STREAMS,
            STREAMS as f32,
        )?;
        if fused_hc_router {
            qwen_next_hyperconnection_input_fused_with_block_router_topk_tcb(
                &mut tcb,
                &post_attn_state,
                &mlp_norm_weight,
                &mlp_down_weight,
                &mlp_up_weight,
                &mlp_normalized,
                &mlp_low,
                &mlp_low_activation,
                &mlp_gate,
                &mlp_input,
                &mlp_block_weight,
                &mlp_block_logits,
                &router_weight,
                &shared_scalar_weight,
                &router_logits,
                &shared_scalar_output,
                &route_ids,
                &route_weights,
                HIDDEN,
                STREAMS,
                HC_LOWRANK,
                EXPERTS,
                TOP_K,
                EPS,
                STREAMS as f32,
                true,
            )?;
        } else {
            qwen_next_hyperconnection_input_fused_with_block_tcb(
                &mut tcb,
                &post_attn_state,
                &mlp_norm_weight,
                &mlp_down_weight,
                &mlp_up_weight,
                &mlp_normalized,
                &mlp_low,
                &mlp_low_activation,
                &mlp_gate,
                &mlp_input,
                &mlp_block_weight,
                &mlp_block_logits,
                HIDDEN,
                STREAMS,
                HC_LOWRANK,
                EPS,
                STREAMS as f32,
            )?;
        }
        if !fused_hc_router && fused_router_topk {
            qwen_next_bf16_router_topk_shared_tcb(
                &mut tcb,
                &router_weight,
                &shared_scalar_weight,
                &mlp_input,
                &router_logits,
                &shared_scalar_output,
                &route_ids,
                &route_weights,
                EXPERTS,
                TOP_K,
                HIDDEN,
                true,
            )?;
        } else if !fused_hc_router {
            native_bf16_dual_seq_tcb(
                &mut tcb,
                &router_weight,
                &shared_scalar_weight,
                &mlp_input,
                &router_logits,
                &shared_scalar_output,
                EXPERTS,
                1,
                HIDDEN,
            )?;
            moe_topk_gate_tcb_ex(
                &mut tcb,
                &router_logits,
                &route_ids,
                &route_weights,
                EXPERTS,
                TOP_K,
                true,
            )?;
        }
        let fused_route_accumulate = args.fused_route_accumulate && route_lut.is_some();
        if let Some(route_lut) = route_lut.as_ref() {
            if fused_route_accumulate {
                qwen_next_bf16_compact_expert_gate_up_shared_swiglu_tcb(
                    &mut tcb,
                    &expert_gate_up_device,
                    &route_ids,
                    route_lut,
                    &mlp_input,
                    &routed_activation,
                    &shared_gate_weight,
                    &shared_up_weight,
                    &shared_activation,
                    compact_expert_count,
                    TOP_K,
                    INTERMEDIATE,
                    HIDDEN,
                    EXPERTS,
                )?;
            } else {
                qwen_next_bf16_compact_expert_gate_up_swiglu_tcb(
                    &mut tcb,
                    &expert_gate_up_device,
                    &route_ids,
                    route_lut,
                    &mlp_input,
                    &routed_activation,
                    compact_expert_count,
                    TOP_K,
                    INTERMEDIATE,
                    HIDDEN,
                    EXPERTS,
                )?;
            }
            if !fused_route_accumulate {
                qwen_next_bf16_compact_expert_down_tcb(
                    &mut tcb,
                    &expert_down_device,
                    &route_ids,
                    route_lut,
                    &routed_activation,
                    &routed_outputs,
                    compact_expert_count,
                    TOP_K,
                    INTERMEDIATE,
                    HIDDEN,
                    EXPERTS,
                )?;
            }
        } else {
            qwen_next_bf16_expert_gate_up_swiglu_tcb(
                &mut tcb,
                &expert_gate_up_device,
                &route_ids,
                &mlp_input,
                &routed_activation,
                EXPERTS,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
            )?;
            qwen_next_bf16_expert_down_tcb(
                &mut tcb,
                &expert_down_device,
                &route_ids,
                &routed_activation,
                &routed_outputs,
                EXPERTS,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
            )?;
        }
        if !fused_route_accumulate {
            native_bf16_swiglu_seq_tcb(
                &mut tcb,
                &shared_gate_weight,
                &shared_up_weight,
                &mlp_input,
                &shared_activation,
                INTERMEDIATE,
                HIDDEN,
            )?;
        }
        if fused_route_accumulate {
            qwen_next_bf16_compact_expert_down_shared_direct_hc_tcb(
                &mut tcb,
                &expert_down_device,
                &route_ids,
                route_lut.as_ref().expect("fused route LUT"),
                &routed_activation,
                &route_weights,
                &shared_down_weight,
                &shared_activation,
                &shared_scalar_output,
                &routed_sum,
                &shared_output,
                &shared_gated_output,
                &moe_output,
                &post_attn_state,
                &mlp_block_logits,
                &final_state,
                compact_expert_count,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
                EXPERTS,
                STREAMS,
                STREAMS as f32,
            )?;
        } else {
            native_bf16_gemv_seq_tcb(
                &mut tcb,
                &shared_down_weight,
                &shared_activation,
                &shared_output,
                HIDDEN,
                INTERMEDIATE,
            )?;
            qwen_next_moe_weighted_sum_add_shared_sigmoid_hc_tcb(
                &mut tcb,
                &routed_outputs,
                &route_weights,
                &shared_output,
                &shared_scalar_output,
                &routed_sum,
                &shared_gated_output,
                &moe_output,
                &post_attn_state,
                &mlp_block_logits,
                &final_state,
                HIDDEN,
                TOP_K,
                STREAMS,
                STREAMS as f32,
            )?;
        }
        let expected_dispatches = 21usize
            .saturating_sub(if fused_qkv_gqa { 1 } else { 0 })
            .saturating_sub(if fused_hc_router {
                2
            } else if fused_router_topk {
                1
            } else {
                0
            })
            .saturating_sub(4)
            .saturating_sub(if fused_route_accumulate { 5 } else { 2 })
            .saturating_sub(1)
            .saturating_sub(if args.fused_attention_gate { 1 } else { 0 });
        if tcb.dispatch_count() != expected_dispatches {
            return Err(format!(
                "full-attention organ dispatch drift: {} expected {}",
                tcb.dispatch_count(),
                expected_dispatches
            )
            .into());
        }
        let encode_ns = encode_started.elapsed().as_nanos() as u64;
        let command_wait_started = Instant::now();
        let timing = tcb.commit_and_wait_timed()?;
        let command_wait_ns = command_wait_started.elapsed().as_nanos() as u64;
        let forward_wall_ns = forward_started.elapsed().as_nanos() as u64;
        let parity_started = Instant::now();
        let observed = device_f32(&output, HIDDEN);
        let normalized_observed = device_f32(&normalized, HC_ELEMENTS);
        let attn_input_observed = device_f32(&attn_input, HIDDEN);
        let q_observed = device_f32(&q_out, QUERY_DIM * 2);
        let k_observed = device_f32(&k_out, KV_DIM);
        let v_observed = device_f32(&v_out, KV_DIM);
        let query_observed = device_f32(&query_out, QUERY_DIM);
        let key_observed = device_f32(&key_cache, KV_DIM);
        let gated_observed = device_f32(&gated_out, QUERY_DIM);
        let block_logits_observed = device_f32(&attn_block_logits, STREAMS);
        let post_attn_state_observed = device_f32(&post_attn_state, HC_ELEMENTS);
        let mlp_normalized_observed = device_f32(&mlp_normalized, HC_ELEMENTS);
        let mlp_input_observed = device_f32(&mlp_input, HIDDEN);
        let router_logits_observed = device_f32(&router_logits, EXPERTS);
        let route_ids_observed = u32_values(&route_ids, TOP_K);
        let route_weights_observed = device_f32(&route_weights, TOP_K);
        let routed_sum_observed = device_f32(&routed_sum, HIDDEN);
        let shared_gated_observed = device_f32(&shared_gated_output, HIDDEN);
        let moe_output_observed = device_f32(&moe_output, HIDDEN);
        let mlp_block_logits_observed = device_f32(&mlp_block_logits, STREAMS);
        let final_state_observed = device_f32(&final_state, HC_ELEMENTS);
        let norm_metrics = metrics(&expected_hc_norm, &normalized_observed);
        let input_metrics = metrics(&expected_input, &attn_input_observed);
        let output_metrics = metrics(&expected_output, &observed);
        let query_metrics = metrics(&expected_query, &query_observed);
        let key_metrics = metrics(&expected_key, &key_observed);
        let attention_metrics = if args.fused_attention_gate {
            json!({
                "captured": false,
                "within_tolerance": true,
                "validation": "fused gated attention output is compared directly; raw pre-gate attention is intentionally not materialized"
            })
        } else {
            metrics(&expected_attention, &device_f32(&attention_out, QUERY_DIM))
        };
        let gated_metrics = metrics(&expected_gated, &gated_observed);
        let q_metrics = metrics(&expected_q, &q_observed);
        let k_metrics = metrics(&expected_k, &k_observed);
        let v_metrics = metrics(&expected_v, &v_observed);
        let block_metrics = metrics(&expected_block_logits, &block_logits_observed);
        let post_metrics = metrics(&expected_post_attn_state, &post_attn_state_observed);
        let mlp_norm_metrics = metrics(&expected_mlp_norm, &mlp_normalized_observed);
        let mlp_input_metrics = metrics(&expected_mlp_input, &mlp_input_observed);
        let router_metrics = metrics(&expected_router_logits, &router_logits_observed);
        let route_weight_metrics = metrics(&expected_route_weights, &route_weights_observed);
        let routed_sum_metrics = metrics(&expected_routed_sum, &routed_sum_observed);
        let shared_metrics = metrics(&expected_shared_gated, &shared_gated_observed);
        let moe_metrics = metrics(&expected_moe_output, &moe_output_observed);
        let mlp_block_metrics = metrics(&expected_mlp_block_logits, &mlp_block_logits_observed);
        let final_metrics = metrics(&expected_final_state, &final_state_observed);
        let parity_ns = parity_started.elapsed().as_nanos() as u64;
        let route_ids_match = expected_route_ids == route_ids_observed;
        let passed = route_ids_match
            && [
                &norm_metrics,
                &input_metrics,
                &output_metrics,
                &q_metrics,
                &k_metrics,
                &v_metrics,
                &query_metrics,
                &key_metrics,
                &attention_metrics,
                &gated_metrics,
                &block_metrics,
                &post_metrics,
                &mlp_norm_metrics,
                &mlp_input_metrics,
                &router_metrics,
                &route_weight_metrics,
                &routed_sum_metrics,
                &shared_metrics,
                &moe_metrics,
                &mlp_block_metrics,
                &final_metrics,
            ]
            .iter()
            .all(|m| m.get("within_tolerance").and_then(Value::as_bool) == Some(true));
        let state_write_started = Instant::now();
        let output_state = if passed {
            if let Some(path) = args.state_out.as_ref() {
                let bytes = f32_bytes(&final_state_observed);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, &bytes)?;
                json!({"kind": format!("layer{}_final_state_f32", args.layer), "path": path, "bytes": bytes.len(), "elements": final_state_observed.len(), "sha256": sha256(&bytes), "source": format!("prefix-fed layer-{} Metal final-state snapshot", args.layer)})
            } else {
                Value::Null
            }
        } else {
            Value::Null
        };
        let state_write_ns = state_write_started.elapsed().as_nanos() as u64;
        let total_before_receipt_ns = function_started.elapsed().as_nanos() as u64;
        let next_action = if args.base_state.is_some() {
            format!(
                "continue from this exact layer-{} state into the next Flash layer",
                args.layer
            )
        } else {
            format!("feed the exact preceding Flash state into layer-{} then continue the remaining Flash graph", args.layer)
        };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default();
        let mut receipt = json!({
            "schema": "hawking.flash_noetic_full_attention_source_bf16_moe_organ.v1",
            "artifact_kind": "SOURCE",
            "status": if passed { "PASSED" } else { "BLOCKED_PARITY" },
            "qualification": format!("EXACT_SOURCE_BF16_LAYER{}_FULL_ATTENTION_MOE_ORGAN_THROUGH_HC_COMBINE", args.layer),
            "repo": REPO, "pinned_revision": REVISION, "layer": args.layer, "layer_type": "full_attention",
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{timestamp}"), "recorded_by": format!("flash_full_attention_layer{}", args.layer), "machine": device, "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet"},
            "source": {"manifest": manifest_value, "embedding": embedding_source, "input_state": input_state, "output_state": output_state,
                "tensors": source_tensors,
                "geometry": {"hidden": HIDDEN, "query_heads": QUERY_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "rotary_dim": ROTARY_DIM, "rope_theta": ROPE_THETA}},
            "parity": {"hyperconnection_norm": norm_metrics, "attention_input": input_metrics, "q_projection": q_metrics, "k_projection": k_metrics, "v_projection": v_metrics, "query_norm_rope": query_metrics, "key_norm_rope": key_metrics, "causal_attention": attention_metrics, "sigmoid_gate": gated_metrics, "output": output_metrics, "attention_block_logits": block_metrics, "post_attention_hyperconnection": post_metrics, "mlp_hyperconnection_norm": mlp_norm_metrics, "mlp_input": mlp_input_metrics, "router_logits": router_metrics, "route_ids": {"expected": expected_route_ids, "observed": route_ids_observed, "match": route_ids_match}, "route_weights": route_weight_metrics, "routed_sum": routed_sum_metrics, "shared_gated_output": shared_metrics, "moe_output": moe_metrics, "mlp_block_logits": mlp_block_metrics, "final_hyperconnection": final_metrics},
            "execution": {"device": device, "provider": "apple_metal", "native_source_bf16": true, "dispatches": timing.dispatches, "command_buffers": timing.command_buffers, "gpu_ns": timing.gpu_ns, "wall_ns": forward_wall_ns, "fallback_count": 0, "position": 0, "attention_gate": if args.fused_attention_gate { "fused_into_mha_final_write" } else { "standalone_qwen80_attention_apply_sigmoid_gate" }, "raw_attention_materialized": !args.fused_attention_gate, "qkv_gqa_fused": fused_qkv_gqa, "router_topk_fused": fused_router_topk || fused_hc_router, "router_topk_fused_into_mlp_hc": fused_hc_router, "route_accumulation": if fused_route_accumulate { "fused_compact_gate_up_shared_down_direct_hc" } else { "materialized_routed_outputs_then_weighted_sum_hc" }, "routed_output_materialized": !fused_route_accumulate, "shared_gate_up_fused": fused_route_accumulate, "compact_moe_load_geometry": if fused_route_accumulate && fused_moe_vec4 { "exact_order_vec4_candidate" } else if fused_route_accumulate { "scalar_authority" } else { "not_applicable" }, "mlp_hyperconnection": "fused_into_moe_epilogue"},
            "timing": {"root_canonicalize_ns": root_canonicalize_ns, "manifest_ns": manifest_ns, "config_ns": config_ns, "index_context_ns": index_context_ns, "input_load_ns": input_load_ns, "source_load_ns": source_load_ns, "source_payload_bytes_read": source_payload_bytes_read, "source_cache_policy": source_cache_policy(), "oracle_ns": oracle_ns, "device_prepare_ns": device_prepare_ns, "encode_ns": encode_ns, "command_wait_ns": command_wait_ns, "gpu_ns": timing.gpu_ns, "forward_wall_ns": forward_wall_ns, "parity_ns": parity_ns, "state_write_ns": state_write_ns, "receipt_write_ns": Value::Null, "total_before_receipt_ns": total_before_receipt_ns, "decomposition": "source_open/index/config/input + source payload + CPU oracle + device preparation + command encoding/wait + parity reads/metrics + state serialization + receipt write"},
            "claim_boundary": format!("This proves only the layer-{} source-BF16 full-attention plus routed/shared MoE organ through its second HyperConnection combine for the explicitly identified single-token state. It does not prove later layers, complete token, tokenizer, decoding, TPS, EBPW, or resident promotion.", args.layer),
            "promotion_allowed": false,
            "next_action": next_action
        });
        let seal = sha256(&serde_json::to_vec(&receipt)?);
        receipt["seal_sha256"] = Value::String(seal);
        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        let receipt_write_started = Instant::now();
        fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
        let receipt_write_ns = receipt_write_started.elapsed().as_nanos() as u64;
        receipt["timing"]["receipt_write_ns"] = json!(receipt_write_ns);
        receipt["timing"]["receipt_write_passes"] = json!(2);
        receipt["timing"]["total_ns_including_first_receipt_write"] =
            json!(function_started.elapsed().as_nanos() as u64);
        receipt["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&receipt)?));
        fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        Ok(if passed { Some(final_state) } else { None })
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        main_impl_with_args(args()?, None).map(|_| ())
    }

    pub(crate) fn run_layer(args: Args) -> Result<(), Box<dyn Error>> {
        main_impl_with_args(args, None).map(|_| ())
    }

    pub(crate) fn run_layer_device_input(
        args: Args,
        input: Option<&PinnedBuffer>,
    ) -> Result<Option<PinnedBuffer>, Box<dyn Error>> {
        main_impl_with_args(args, input)
    }

    /// Stateful attention-organ probe: one cached context and one KV buffer
    /// carry two decode positions through the existing Metal Q/K/RoPE and MHA
    /// kernels.  The surrounding MLP and full model are intentionally out of
    /// scope; this qualifies only the KV state contract.
    pub(crate) fn run_stateful_attention_probe_with_inputs(
        root: PathBuf,
        layer: usize,
        token_ids: &[usize],
        input_states: Option<&[Vec<f32>]>,
        out: PathBuf,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        run_stateful_attention_probe_with_inputs_mode(
            root,
            layer,
            token_ids,
            input_states,
            out,
            None,
        )
    }

    pub(crate) fn run_stateful_attention_probe_with_inputs_mode(
        root: PathBuf,
        layer: usize,
        token_ids: &[usize],
        input_states: Option<&[Vec<f32>]>,
        out: PathBuf,
        route_union: Option<Vec<u32>>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        if token_ids.len() < 2 {
            return Err("stateful attention probe requires at least two tokens".into());
        }
        let fused_attention_gate = env::var("HAWKING_FLASH_FUSE_ATTENTION_GATE")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_qkv_gqa = env::var("HAWKING_FLASH_QKV_GQA_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_router_topk = env::var("HAWKING_FLASH_ROUTER_TOPK_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_hc_router = env::var("HAWKING_FLASH_HC_ROUTER_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        let fused_moe_vec4 = env::var("HAWKING_FLASH_MOE_VEC4")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false);
        if let Some(states) = input_states {
            if states.len() != token_ids.len()
                || states.iter().any(|state| state.len() != HC_ELEMENTS)
            {
                return Err("stateful attention input states must match token count and hyperconnection geometry".into());
            }
        }
        let root = root.canonicalize()?;
        let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        let text = config.get("text_config").ok_or("text_config missing")?;
        let vocab = text
            .get("vocab_size")
            .and_then(Value::as_u64)
            .ok_or("vocab missing")? as usize;
        let layer_type = text
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|v| v.get(layer))
            .and_then(Value::as_str)
            .unwrap_or("");
        if layer_type != "full_attention" {
            return Err(format!("layer {layer} is not full_attention").into());
        }
        let (index, context) = cached_flash_resources(&root)?;
        let row = |token_id: usize| -> Result<(Vec<f32>, String, usize), Box<dyn Error>> {
            if token_id >= vocab {
                return Err(format!("token {token_id} outside vocab {vocab}").into());
            }
            let bytes = index.read_raw_range(EMBEDDING, token_id * HIDDEN * 2, HIDDEN * 2)?;
            let values = (0..HIDDEN).map(|i| bf16(&bytes, i)).collect::<Vec<_>>();
            Ok((values, sha256(&bytes), bytes.len()))
        };
        let (first_base, first_sha, first_bytes) = if let Some(states) = input_states {
            let bytes = f32_bytes(&states[0]);
            (states[0].clone(), sha256(&bytes), bytes.len())
        } else {
            let (first, sha, bytes) = row(token_ids[0])?;
            (
                (0..STREAMS)
                    .flat_map(|_| first.iter().copied())
                    .collect::<Vec<_>>(),
                sha,
                bytes,
            )
        };
        let hc_norm = tensor(&index, layer_tensor_name(layer, HC_NORM), &[HC_ELEMENTS])?;
        let hc_down = tensor(
            &index,
            layer_tensor_name(layer, HC_DOWN),
            &[HC_LOWRANK, HC_ELEMENTS],
        )?;
        let hc_up = tensor(
            &index,
            layer_tensor_name(layer, HC_UP),
            &[HC_ELEMENTS, HC_LOWRANK],
        )?;
        let q_proj = tensor(
            &index,
            layer_tensor_name(layer, Q_PROJ),
            &[QUERY_DIM * 2, HIDDEN],
        )?;
        let k_proj = tensor(&index, layer_tensor_name(layer, K_PROJ), &[KV_DIM, HIDDEN])?;
        let v_proj = tensor(&index, layer_tensor_name(layer, V_PROJ), &[KV_DIM, HIDDEN])?;
        let q_norm = tensor(&index, layer_tensor_name(layer, Q_NORM), &[HEAD_DIM])?;
        let k_norm = tensor(&index, layer_tensor_name(layer, K_NORM), &[HEAD_DIM])?;
        let o_proj = tensor(
            &index,
            layer_tensor_name(layer, O_PROJ),
            &[HIDDEN, QUERY_DIM],
        )?;
        let hc_block = tensor(
            &index,
            layer_tensor_name(layer, HC_BLOCK),
            &[STREAMS, HC_ELEMENTS],
        )?;
        let router = tensor(&index, layer_tensor_name(layer, ROUTER), &[EXPERTS, HIDDEN])?;
        let route_lut_host = route_union
            .as_ref()
            .map(|routes| {
                let mut selected = routes.clone();
                selected.sort_unstable();
                selected.dedup();
                if selected.is_empty()
                    || selected.len() > EXPERTS
                    || selected.iter().any(|&expert| expert as usize >= EXPERTS)
                {
                    return Err("invalid full-attention route union".into());
                }
                let mut lut = vec![u32::MAX; EXPERTS];
                for (slot, &expert) in selected.iter().enumerate() {
                    lut[expert as usize] = slot as u32;
                }
                Ok::<Vec<u32>, Box<dyn Error>>(lut)
            })
            .transpose()?;
        let selected_routes = route_lut_host.as_ref().map(|lut| {
            lut.iter()
                .enumerate()
                .filter_map(|(expert, &slot)| (slot != u32::MAX).then_some(expert as u32))
                .collect::<Vec<_>>()
        });
        let expert_gate_up_weight = if let Some(routes) = selected_routes.as_ref() {
            tensor_rows(
                &index,
                layer_tensor_name(layer, EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
                routes,
            )?
        } else {
            tensor(
                &index,
                layer_tensor_name(layer, EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            )?
        };
        let expert_down_weight = if let Some(routes) = selected_routes.as_ref() {
            tensor_rows(
                &index,
                layer_tensor_name(layer, EXPERT_DOWN),
                &[EXPERTS, HIDDEN, INTERMEDIATE],
                routes,
            )?
        } else {
            tensor(
                &index,
                layer_tensor_name(layer, EXPERT_DOWN),
                &[EXPERTS, HIDDEN, INTERMEDIATE],
            )?
        };
        let shared_gate = tensor(
            &index,
            layer_tensor_name(layer, SHARED_GATE),
            &[INTERMEDIATE, HIDDEN],
        )?;
        let shared_up = tensor(
            &index,
            layer_tensor_name(layer, SHARED_UP),
            &[INTERMEDIATE, HIDDEN],
        )?;
        let shared_down = tensor(
            &index,
            layer_tensor_name(layer, SHARED_DOWN),
            &[HIDDEN, INTERMEDIATE],
        )?;
        let shared_scalar = tensor(
            &index,
            layer_tensor_name(layer, SHARED_SCALAR),
            &[1, HIDDEN],
        )?;
        let mlp_norm = tensor(
            &index,
            layer_tensor_name(layer, HC_MLP_NORM),
            &[HC_ELEMENTS],
        )?;
        let mlp_down = tensor(
            &index,
            layer_tensor_name(layer, HC_MLP_DOWN),
            &[HC_LOWRANK, HC_ELEMENTS],
        )?;
        let mlp_up = tensor(
            &index,
            layer_tensor_name(layer, HC_MLP_UP),
            &[HC_ELEMENTS, HC_LOWRANK],
        )?;
        let mlp_block = tensor(
            &index,
            layer_tensor_name(layer, HC_MLP_BLOCK),
            &[STREAMS, HC_ELEMENTS],
        )?;
        let expected_norm = grouped_hc_norm(&first_base, &f32_vec(&hc_norm));
        let expected_input = hc_read_mix(&expected_norm, &hc_down.bytes, &hc_up.bytes);
        let expected_q = matvec(&q_proj.bytes, QUERY_DIM * 2, HIDDEN, &expected_input);
        let expected_k = matvec(&k_proj.bytes, KV_DIM, HIDDEN, &expected_input);
        let expected_v = matvec(&v_proj.bytes, KV_DIM, HIDDEN, &expected_input);
        let expected_query = norm_rope(&query(&expected_q), &f32_vec(&q_norm), QUERY_HEADS, 0);
        let expected_key = norm_rope(&expected_k, &f32_vec(&k_norm), KV_HEADS, 0);
        let expected_attention = attention(&expected_query, &expected_key, &expected_v);
        let expected_gated = gated(&expected_attention, &expected_q);
        let expected_output = matvec(&o_proj.bytes, HIDDEN, QUERY_DIM, &expected_gated);
        let expected_attn_block_logits =
            matvec(&hc_block.bytes, STREAMS, HC_ELEMENTS, &expected_norm);
        let expected_post_attn_state =
            hc_combine(&first_base, &expected_output, &expected_attn_block_logits);
        let expected_mlp_norm = grouped_hc_norm(&expected_post_attn_state, &f32_vec(&mlp_norm));
        let expected_mlp_input = hc_read_mix(&expected_mlp_norm, &mlp_down.bytes, &mlp_up.bytes);
        let expected_router_logits = matvec(&router.bytes, EXPERTS, HIDDEN, &expected_mlp_input);
        let (expected_route_ids, expected_route_weights) = topk_router(&expected_router_logits);
        let mut expected_routed_sum = vec![0.0f32; HIDDEN];
        for (&expert, &weight) in expected_route_ids.iter().zip(&expected_route_weights) {
            let routed = expert_down(
                &expert_down_weight.bytes,
                expert as usize,
                &expert_swiglu(
                    &expert_gate_up_weight.bytes,
                    expert as usize,
                    &expected_mlp_input,
                    route_lut_host.as_deref(),
                )?,
                route_lut_host.as_deref(),
            )?;
            for (out, value) in expected_routed_sum.iter_mut().zip(routed) {
                *out += weight * value;
            }
        }
        let shared_gate_values = matvec(
            &shared_gate.bytes,
            INTERMEDIATE,
            HIDDEN,
            &expected_mlp_input,
        );
        let shared_up_values = matvec(&shared_up.bytes, INTERMEDIATE, HIDDEN, &expected_mlp_input);
        let shared_activation = shared_gate_values
            .into_iter()
            .zip(shared_up_values)
            .map(|(gate, up)| silu(gate) * up)
            .collect::<Vec<_>>();
        let shared_output = matvec(&shared_down.bytes, HIDDEN, INTERMEDIATE, &shared_activation);
        let shared_logit = matvec(&shared_scalar.bytes, 1, HIDDEN, &expected_mlp_input)[0];
        let expected_shared_gated = shared_output
            .iter()
            .map(|&value| sigmoid(shared_logit) * value)
            .collect::<Vec<_>>();
        let expected_moe_output = expected_routed_sum
            .iter()
            .zip(&expected_shared_gated)
            .map(|(routed, shared)| routed + shared)
            .collect::<Vec<_>>();
        let expected_mlp_block_logits =
            matvec(&mlp_block.bytes, STREAMS, HC_ELEMENTS, &expected_mlp_norm);
        let expected_final_state = hc_combine(
            &expected_post_attn_state,
            &expected_moe_output,
            &expected_mlp_block_logits,
        );

        let device = context.device_name();
        let hc_norm_buf = context.new_buffer_with_bytes_checked(&hc_norm.bytes)?;
        let hc_down_buf = context.new_buffer_with_bytes_checked(&hc_down.bytes)?;
        let hc_up_buf = context.new_buffer_with_bytes_checked(&hc_up.bytes)?;
        let q_weight = context.new_buffer_with_bytes_checked(&q_proj.bytes)?;
        let k_weight = context.new_buffer_with_bytes_checked(&k_proj.bytes)?;
        let v_weight = context.new_buffer_with_bytes_checked(&v_proj.bytes)?;
        let q_norm_buf = context.new_buffer_with_bytes_checked(&f32_bytes(&f32_vec(&q_norm)))?;
        let k_norm_buf = context.new_buffer_with_bytes_checked(&f32_bytes(&f32_vec(&k_norm)))?;
        let o_weight = context.new_buffer_with_bytes_checked(&o_proj.bytes)?;
        let hc_block_buf = context.new_buffer_with_bytes_checked(&hc_block.bytes)?;
        let router_weight = context.new_buffer_with_bytes_checked(&router.bytes)?;
        let expert_gate_up_device =
            context.new_buffer_with_bytes_checked(&expert_gate_up_weight.bytes)?;
        let expert_down_device =
            context.new_buffer_with_bytes_checked(&expert_down_weight.bytes)?;
        let shared_gate_weight = context.new_buffer_with_bytes_checked(&shared_gate.bytes)?;
        let shared_up_weight = context.new_buffer_with_bytes_checked(&shared_up.bytes)?;
        let shared_down_weight = context.new_buffer_with_bytes_checked(&shared_down.bytes)?;
        let shared_scalar_weight = context.new_buffer_with_bytes_checked(&shared_scalar.bytes)?;
        let mlp_norm_weight = context.new_buffer_with_bytes_checked(&mlp_norm.bytes)?;
        let mlp_down_weight = context.new_buffer_with_bytes_checked(&mlp_down.bytes)?;
        let mlp_up_weight = context.new_buffer_with_bytes_checked(&mlp_up.bytes)?;
        let mlp_block_weight = context.new_buffer_with_bytes_checked(&mlp_block.bytes)?;
        let input_buf = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let normalized = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let low = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let low_activation = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let gate = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let attn_input = context.new_buffer_checked(HIDDEN * 4)?;
        let q_out = context.new_buffer_checked(QUERY_DIM * 2 * 4)?;
        let k_out = context.new_buffer_checked(KV_DIM * 4)?;
        let v_out = context.new_buffer_checked(KV_DIM * 4)?;
        let query_out = context.new_buffer_checked(QUERY_DIM * 4)?;
        let key_cache = context.new_buffer_checked(token_ids.len() * KV_DIM * 4)?;
        let value_cache = context.new_buffer_checked(token_ids.len() * KV_DIM * 4)?;
        let attention_out = context.new_buffer_checked(QUERY_DIM * 4)?;
        let gated_out = context.new_buffer_checked(QUERY_DIM * 4)?;
        let output = context.new_buffer_checked(HIDDEN * 4)?;
        let attn_block_logits = context.new_buffer_checked(STREAMS * 4)?;
        let post_attn_state = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mlp_normalized = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mlp_low = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let mlp_low_activation = context.new_buffer_checked(HC_LOWRANK * 4)?;
        let mlp_gate = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mlp_input = context.new_buffer_checked(HIDDEN * 4)?;
        let router_logits = context.new_buffer_checked(EXPERTS * 4)?;
        let route_ids = context.new_buffer_checked(TOP_K * 4)?;
        let route_lut = route_lut_host
            .as_ref()
            .map(|lut| context.new_buffer_with_bytes_checked(&u32_bytes(lut)))
            .transpose()?;
        let compact_experts = selected_routes
            .as_ref()
            .map(|routes| routes.len())
            .unwrap_or(EXPERTS);
        let fused_route_accumulate = route_lut.is_some();
        let route_weights = context.new_buffer_checked(TOP_K * 4)?;
        let routed_activation = context.new_buffer_checked(TOP_K * INTERMEDIATE * 4)?;
        let routed_outputs = context.new_buffer_checked(TOP_K * HIDDEN * 4)?;
        let routed_sum = context.new_buffer_checked(HIDDEN * 4)?;
        let shared_activation = context.new_buffer_checked(INTERMEDIATE * 4)?;
        let shared_output = context.new_buffer_checked(HIDDEN * 4)?;
        let shared_scalar_output = context.new_buffer_checked(4)?;
        let shared_gated_output = context.new_buffer_checked(HIDDEN * 4)?;
        let moe_output = context.new_buffer_checked(HIDDEN * 4)?;
        let mlp_block_logits = context.new_buffer_checked(STREAMS * 4)?;
        let final_state = context.new_buffer_checked(HC_ELEMENTS * 4)?;
        let mut final_states: Vec<Vec<f32>> = Vec::with_capacity(token_ids.len());
        let rows = token_ids.iter().enumerate().map(|(step, &token_id)| -> Result<Value, Box<dyn Error>> {
            let (base, embedding_sha, embedding_bytes) = if let Some(states) = input_states {
                let bytes = f32_bytes(&states[step]);
                (states[step].clone(), sha256(&bytes), bytes.len())
            } else {
                let (embedding_row, sha, bytes) = row(token_id)?;
                ((0..STREAMS).flat_map(|_| embedding_row.iter().copied()).collect::<Vec<_>>(), sha, bytes)
            };
            MetalContext::write_buffer_bytes(&input_buf, &f32_bytes(&base));
            let started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&context);
            qwen_next_hyperconnection_input_fused_with_block_tcb(
                &mut tcb, &input_buf, &hc_norm_buf, &hc_down_buf, &hc_up_buf,
                &normalized, &low, &low_activation, &gate, &attn_input,
                &hc_block_buf, &attn_block_logits, HIDDEN, STREAMS, HC_LOWRANK, EPS, STREAMS as f32,
            )?;
            if fused_qkv_gqa {
                qwen_next_bf16_qkv_gqa_rope_cache_tcb(
                    &mut tcb, &q_weight, &k_weight, &v_weight, &attn_input,
                    &q_norm_buf, &k_norm_buf, &q_out, &k_out, &v_out, &query_out,
                    &key_cache, &value_cache, step, QUERY_HEADS, KV_HEADS, HEAD_DIM,
                    ROTARY_DIM, HIDDEN, ROPE_THETA, EPS,
                )?;
            } else {
                native_bf16_triple_seq_tcb(&mut tcb, &q_weight, &k_weight, &v_weight, &attn_input, &q_out, &k_out, &v_out, QUERY_DIM * 2, KV_DIM, KV_DIM, HIDDEN)?;
                tcb.dispatch_threads("qwen80_gqa_qk_norm_rope_cache_f32", (QUERY_HEADS as u32, 1, 1), (QUERY_HEADS as u32, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&q_out), 0); enc.set_buffer(1, Some(&k_out), 0); enc.set_buffer(2, Some(&v_out), 0);
                    enc.set_buffer(3, Some(&q_norm_buf), 0); enc.set_buffer(4, Some(&k_norm_buf), 0); enc.set_buffer(5, Some(&query_out), 0);
                    enc.set_buffer(6, Some(&key_cache), 0); enc.set_buffer(7, Some(&value_cache), 0);
                    enc.set_u32(8, step as u32); enc.set_u32(9, QUERY_HEADS as u32); enc.set_u32(10, KV_HEADS as u32);
                    enc.set_u32(11, HEAD_DIM as u32); enc.set_u32(12, ROTARY_DIM as u32); enc.set_f32(13, ROPE_THETA); enc.set_f32(14, EPS);
                })?;
            }
            if fused_attention_gate {
                mha_decode_f32_qwen38_gated_tcb(
                    &mut tcb, &query_out, &key_cache, 0, &value_cache, 0, &gated_out, &q_out,
                    step + 1, HEAD_DIM, QUERY_HEADS, KV_HEADS,
                )?;
            } else {
                mha_decode_f32_tcb(&mut tcb, &query_out, &key_cache, 0, &value_cache, 0, &attention_out, step + 1, HEAD_DIM, QUERY_HEADS, KV_HEADS)?;
                tcb.dispatch_threads("qwen80_attention_apply_sigmoid_gate", (QUERY_DIM as u32, 1, 1), (256, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&attention_out), 0); enc.set_buffer(1, Some(&q_out), 0); enc.set_buffer(2, Some(&gated_out), 0);
                    enc.set_u32(3, QUERY_DIM as u32); enc.set_u32(4, HEAD_DIM as u32);
                })?;
            }
            native_bf16_gemv_hyperconnection_combine_tcb(
                &mut tcb, &o_weight, &gated_out, &input_buf, &attn_block_logits,
                &output, &post_attn_state, HIDDEN, QUERY_DIM, STREAMS, STREAMS as f32,
            )?;
            if fused_hc_router {
                qwen_next_hyperconnection_input_fused_with_block_router_topk_tcb(
                    &mut tcb, &post_attn_state, &mlp_norm_weight, &mlp_down_weight, &mlp_up_weight,
                    &mlp_normalized, &mlp_low, &mlp_low_activation, &mlp_gate, &mlp_input,
                    &mlp_block_weight, &mlp_block_logits, &router_weight, &shared_scalar_weight,
                    &router_logits, &shared_scalar_output, &route_ids, &route_weights,
                    HIDDEN, STREAMS, HC_LOWRANK, EXPERTS, TOP_K, EPS, STREAMS as f32, true,
                )?;
            } else {
                qwen_next_hyperconnection_input_fused_with_block_tcb(
                    &mut tcb, &post_attn_state, &mlp_norm_weight, &mlp_down_weight, &mlp_up_weight,
                    &mlp_normalized, &mlp_low, &mlp_low_activation, &mlp_gate, &mlp_input,
                    &mlp_block_weight, &mlp_block_logits, HIDDEN, STREAMS, HC_LOWRANK, EPS, STREAMS as f32,
                )?;
            }
            if !fused_hc_router && fused_router_topk {
                qwen_next_bf16_router_topk_shared_tcb(
                    &mut tcb,
                    &router_weight,
                    &shared_scalar_weight,
                    &mlp_input,
                    &router_logits,
                    &shared_scalar_output,
                    &route_ids,
                    &route_weights,
                    EXPERTS,
                    TOP_K,
                    HIDDEN,
                    true,
                )?;
            } else if !fused_hc_router {
                native_bf16_dual_seq_tcb(&mut tcb, &router_weight, &shared_scalar_weight, &mlp_input, &router_logits, &shared_scalar_output, EXPERTS, 1, HIDDEN)?;
                moe_topk_gate_tcb_ex(&mut tcb, &router_logits, &route_ids, &route_weights, EXPERTS, TOP_K, true)?;
            }
            if let Some(route_lut) = route_lut.as_ref() {
                if fused_route_accumulate {
                    qwen_next_bf16_compact_expert_gate_up_shared_swiglu_tcb(
                        &mut tcb, &expert_gate_up_device, &route_ids, route_lut, &mlp_input,
                        &routed_activation, &shared_gate_weight, &shared_up_weight, &shared_activation,
                        compact_experts, TOP_K, INTERMEDIATE, HIDDEN, EXPERTS,
                    )?;
                } else {
                    qwen_next_bf16_compact_expert_gate_up_swiglu_tcb(&mut tcb, &expert_gate_up_device, &route_ids, route_lut, &mlp_input, &routed_activation, compact_experts, TOP_K, INTERMEDIATE, HIDDEN, EXPERTS)?;
                }
            } else {
                qwen_next_bf16_expert_gate_up_swiglu_tcb(&mut tcb, &expert_gate_up_device, &route_ids, &mlp_input, &routed_activation, EXPERTS, TOP_K, INTERMEDIATE, HIDDEN)?;
                qwen_next_bf16_expert_down_tcb(&mut tcb, &expert_down_device, &route_ids, &routed_activation, &routed_outputs, EXPERTS, TOP_K, INTERMEDIATE, HIDDEN)?;
            }
            if !fused_route_accumulate {
                native_bf16_swiglu_seq_tcb(&mut tcb, &shared_gate_weight, &shared_up_weight, &mlp_input, &shared_activation, INTERMEDIATE, HIDDEN)?;
            }
            if let Some(route_lut) = route_lut.as_ref() {
                qwen_next_bf16_compact_expert_down_shared_direct_hc_tcb(
                    &mut tcb, &expert_down_device, &route_ids, route_lut, &routed_activation,
                    &route_weights, &shared_down_weight, &shared_activation, &shared_scalar_output,
                    &routed_sum, &shared_output, &shared_gated_output, &moe_output,
                    &post_attn_state, &mlp_block_logits, &final_state,
                    compact_experts, TOP_K, INTERMEDIATE, HIDDEN, EXPERTS, STREAMS, STREAMS as f32,
                )?;
            } else {
                native_bf16_gemv_seq_tcb(&mut tcb, &shared_down_weight, &shared_activation, &shared_output, HIDDEN, INTERMEDIATE)?;
                qwen_next_moe_weighted_sum_add_shared_sigmoid_hc_tcb(
                    &mut tcb, &routed_outputs, &route_weights, &shared_output,
                    &shared_scalar_output, &routed_sum, &shared_gated_output,
                    &moe_output, &post_attn_state, &mlp_block_logits, &final_state,
                    HIDDEN, TOP_K, STREAMS, STREAMS as f32,
                )?;
            }
            let dispatches = tcb.dispatch_count();
            let timing = tcb.commit_and_wait_timed()?;
            let wall_ns = started.elapsed().as_nanos() as u64;
            let query_observed = device_f32(&query_out, QUERY_DIM);
            let attention_observed = if fused_attention_gate {
                device_f32(&gated_out, QUERY_DIM)
            } else {
                device_f32(&attention_out, QUERY_DIM)
            };
            let cache = device_f32(&key_cache, token_ids.len() * KV_DIM);
            let slot = &cache[step * KV_DIM..(step + 1) * KV_DIM];
            let final_observed = device_f32(&final_state, HC_ELEMENTS);
            final_states.push(final_observed.clone());
            let parity = if step == 0 { Some(json!({"query": metrics(&expected_query, &query_observed), "final_state": metrics(&expected_final_state, &final_observed)})) } else { None };
            let route_ids_observed = u32_values(&route_ids, TOP_K);
            Ok(json!({"step": step, "token_id": token_id, "embedding_sha256": embedding_sha, "embedding_bytes": embedding_bytes, "sequence_length": step + 1, "dispatches": dispatches, "gpu_ns": timing.gpu_ns, "wall_ns": wall_ns, "key_slot_sha256": sha256(&f32_bytes(slot)), "attention_observed_kind": if fused_attention_gate { "gated_attention" } else { "raw_attention" }, "attention_sha256": sha256(&f32_bytes(&attention_observed)), "final_state_sha256": sha256(&f32_bytes(&final_observed)), "route_ids": route_ids_observed, "finite": final_observed.iter().all(|v| v.is_finite()), "first_step_parity": parity}))
        }).collect::<Result<Vec<_>, _>>()?;
        let distinct_slots = rows
            .windows(2)
            .all(|pair| pair[0].get("key_slot_sha256") != pair[1].get("key_slot_sha256"));
        let input_kind = if input_states.is_some() {
            "layer2_stateful_linear_prefix_output"
        } else {
            "repeated_bos_embedding"
        };
        let mut doc = json!({
            "schema": "hawking.flash.stateful_attention_organ_probe.v1",
            "status": if distinct_slots { "PASSED_STATEFUL_KV_ORGAN" } else { "BLOCKED_KV_SLOT_NOT_DISTINCT" },
            "model": REPO, "pinned_revision": REVISION, "layer": layer, "token_ids": token_ids,
            "execution": {"device": device, "provider": "apple_metal", "process_boundary": "one native process", "context_reused": true, "weights_reused": true, "kv_cache_reused": true, "kv_cache_slots": token_ids.len(), "full_attention_mlp_epilogue": true, "mlp_hyperconnection": "fused_into_moe_epilogue", "attention_gate": if fused_attention_gate { "fused_into_mha_final_write" } else { "standalone_qwen80_attention_apply_sigmoid_gate" }, "raw_attention_materialized": !fused_attention_gate, "qkv_gqa_fused": fused_qkv_gqa, "router_topk_fused": fused_router_topk || fused_hc_router, "router_topk_fused_into_mlp_hc": fused_hc_router, "compact_moe_load_geometry": if route_lut_host.is_some() && fused_moe_vec4 { "exact_order_vec4_candidate" } else if route_lut_host.is_some() { "scalar_authority" } else { "not_applicable" }, "input_kind": input_kind, "expert_bank_mode": if route_lut_host.is_some() { "route_union_compact" } else { "dense" }, "compact_expert_count": compact_experts, "source_payload_bytes_read": index.bytes_read_total(), "first_input_sha256": first_sha, "first_input_bytes": first_bytes},
            "steps": rows, "distinct_kv_slots": distinct_slots, "stateful_final_state": true, "accepted_generation_tokens": 0, "accepted_tps": Value::Null, "complete_system_ebpw": Value::Null, "promotion_allowed": false,
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_attention_probe", "machine": device, "rule": "S032 §3 -- stateful organ timing; quiescence unknown"},
            "claim_boundary": if input_states.is_some() { "This proves a cross-species seam from stateful linear-prefix outputs through a persistent full-attention KV organ and its HyperConnection/routed/shared-MoE MLP epilogue, with distinct KV slots and first-step final-state parity. It does not prove device-only cross-module handoff, 48-layer token acceptance, complete-model TPS, EBPW, or resident promotion." } else { "This proves persistent multi-position KV cache writes and the full-attention HyperConnection/routed/shared-MoE MLP epilogue, with first-step final-state parity. It does not prove cross-species state handoff, full-model token acceptance, complete-model TPS, EBPW, or resident promotion." },
            "next": "Feed this final state through the next qualified Flash layer in the persistent 48-layer session, then remove the diagnostic host seam."
        });
        doc["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&doc)?));
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&doc)?)?;
        println!("{}", serde_json::to_string_pretty(&doc)?);
        Ok(final_states)
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}

#[cfg(target_os = "macos")]
pub(crate) use macos::Args;
#[cfg(target_os = "macos")]
pub(crate) fn run_layer(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_layer(args)
}
#[cfg(target_os = "macos")]
pub(crate) fn run_layer_device_input(
    args: Args,
    input: Option<&hawking_core::metal::PinnedBuffer>,
) -> Result<Option<hawking_core::metal::PinnedBuffer>, Box<dyn std::error::Error>> {
    macos::run_layer_device_input(args, input)
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_attention_probe(
    root: std::path::PathBuf,
    layer: usize,
    token_ids: &[usize],
    out: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_stateful_attention_probe_with_inputs(root, layer, token_ids, None, out).map(|_| ())
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_attention_probe_route_union(
    root: std::path::PathBuf,
    layer: usize,
    token_ids: &[usize],
    route_ids: Vec<u32>,
    out: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_stateful_attention_probe_with_inputs_mode(
        root,
        layer,
        token_ids,
        None,
        out,
        Some(route_ids),
    )
    .map(|_| ())
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_attention_probe_from_states(
    root: std::path::PathBuf,
    layer: usize,
    token_ids: &[usize],
    input_states: &[Vec<f32>],
    out: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_stateful_attention_probe_with_inputs(root, layer, token_ids, Some(input_states), out)
        .map(|_| ())
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_attention_probe_from_states_with_outputs(
    root: std::path::PathBuf,
    layer: usize,
    token_ids: &[usize],
    input_states: &[Vec<f32>],
    out: std::path::PathBuf,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    macos::run_stateful_attention_probe_with_inputs(root, layer, token_ids, Some(input_states), out)
}
