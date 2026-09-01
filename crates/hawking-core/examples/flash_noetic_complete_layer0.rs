//! Exact source-BF16 graph for a selectable linear-attention layer of
//! Qwen3.8-Flash-Next (layer 0 remains the default receipt).
//!
//! This is the first integrated Flash control object.  It streams the pinned
//! source tensors from ModelLake, keeps the source BF16 weights on a
//! Metal device, and executes the linear-attention/DeltaNet body followed by
//! routed-plus-shared MoE and both exact HyperConnection boundaries in one
//! ordered TokenCommandBuffer.  The CPU path is an independent BF16 source
//! oracle over the same bytes.  It is a layer gate, not a 48-layer token or
//! TPS claim.

#![recursion_limit = "512"]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash source graph requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::kernels::{
        moe_topk_gate_tcb_ex, native_bf16_dual_seq_tcb,
        native_bf16_gemv_hyperconnection_combine_tcb, native_bf16_gemv_seq_tcb,
        native_bf16_swiglu_seq_tcb, qwen_next_ba_split_to_decay_beta_source_bf16_tcb,
        qwen_next_bf16_compact_expert_down_shared_direct_hc_tcb,
        qwen_next_bf16_compact_expert_gate_up_shared_swiglu_tcb, qwen_next_bf16_expert_down_tcb,
        qwen_next_bf16_expert_gate_up_swiglu_tcb, qwen_next_bf16_router_topk_shared_tcb,
        qwen_next_deltanet_source_bf16_gated_rmsnorm_tcb,
        qwen_next_gated_delta_decode_single_at_state_offset_tcb,
        qwen_next_hyperconnection_input_fused_with_block_router_topk_tcb,
        qwen_next_hyperconnection_input_fused_with_block_tcb,
        qwen_next_moe_weighted_sum_add_shared_sigmoid_hc_tcb,
        qwen_next_qkv_split_rearrange_conv_l2_source_bf16_tcb,
    };
    use hawking_core::metal::{DispatchSample, MetalContext, PinnedBuffer, TokenCommandBuffer};
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
            let replace = slot
                .as_ref()
                .map(|cached| cached.root.as_path() != root)
                .unwrap_or(true);
            if replace {
                eprintln!(
                    "Flash executor: opening source index and Metal context once for this process"
                );
                *slot = Some(CachedFlashResources {
                    root: root.to_path_buf(),
                    index: Rc::new(SourceBf16Index::open(root)?),
                    context: Rc::new(MetalContext::new_with_trace(true)?),
                });
            }
            let cached = slot.as_ref().expect("Flash resource cache populated");
            Ok((Rc::clone(&cached.index), Rc::clone(&cached.context)))
        })
    }

    const SCHEMA: &str = "hawking.flash_noetic_complete_layer0_source_bf16.v1";
    const DISPATCH_LEDGER_SCHEMA: &str = "hawking.flash_layer0_dispatch_ledger.v1";
    const CRITICAL_PATH_SCHEMA: &str = "hawking.flash_layer0_critical_path.v1";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const NOMENCLATURE_VERSION: &str = "HAWKING_NOMENCLATURE_V1";
    const MANIFEST_PATH: &str =
        "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json";
    const DEFAULT_ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const DEFAULT_WARMUP: usize = 0;
    const DEFAULT_REPS: usize = 1;
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const HC_ELEMENTS: usize = HIDDEN * STREAMS;
    const HC_LOWRANK: usize = 320;
    const KEY_HEADS: usize = 16;
    const VALUES_PER_KEY_HEAD: usize = 3;
    const VALUE_HEADS: usize = KEY_HEADS * VALUES_PER_KEY_HEAD;
    const KEY_HEAD_DIM: usize = 128;
    const VALUE_HEAD_DIM: usize = 128;
    const KEY_ELEMENTS: usize = KEY_HEADS * KEY_HEAD_DIM;
    const VALUE_ELEMENTS: usize = VALUE_HEADS * VALUE_HEAD_DIM;
    const QKV_ELEMENTS: usize = KEY_ELEMENTS * 2 + VALUE_ELEMENTS;
    const CONV_KERNEL: usize = 4;
    const CONV_STATE_ELEMENTS: usize = QKV_ELEMENTS * (CONV_KERNEL - 1);
    const RECURRENT_STATE_ELEMENTS: usize = VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM;
    const EXPERTS: usize = 512;
    const TOP_K: usize = 10;
    const INTERMEDIATE: usize = 640;
    const EPS: f32 = 1.0e-6;
    const BOS_TOKEN_ID: usize = 248044;
    const OUTPUT_TOLERANCE: f32 = 2.0e-2;
    const ROUTE_WEIGHT_TOLERANCE: f32 = 2.0e-3;

    const HC_ATTN_NORM: &str = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight";
    const HC_ATTN_DOWN: &str =
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight";
    const HC_ATTN_UP: &str =
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight";
    const HC_ATTN_BLOCK: &str =
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight";
    const QKV: &str = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    const Z: &str = "model.language_model.layers.0.linear_attn.in_proj_z.weight";
    const B: &str = "model.language_model.layers.0.linear_attn.in_proj_b.weight";
    const A: &str = "model.language_model.layers.0.linear_attn.in_proj_a.weight";
    const CONV: &str = "model.language_model.layers.0.linear_attn.conv1d.weight";
    const A_LOG: &str = "model.language_model.layers.0.linear_attn.A_log";
    const DT_BIAS: &str = "model.language_model.layers.0.linear_attn.dt_bias";
    const LINEAR_NORM: &str = "model.language_model.layers.0.linear_attn.norm.weight";
    const OUT_PROJ: &str = "model.language_model.layers.0.linear_attn.out_proj.weight";
    const ROUTER: &str = "model.language_model.layers.0.mlp.gate.weight";
    const EXPERT_GATE_UP: &str = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const EXPERT_DOWN: &str = "model.language_model.layers.0.mlp.experts.down_proj";
    const SHARED_GATE: &str = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight";
    const SHARED_UP: &str = "model.language_model.layers.0.mlp.shared_expert.up_proj.weight";
    const SHARED_DOWN: &str = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
    const SHARED_SCALAR: &str = "model.language_model.layers.0.mlp.shared_expert_gate.weight";
    const HC_MLP_NORM: &str = "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight";
    const HC_MLP_DOWN: &str =
        "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight";
    const HC_MLP_UP: &str =
        "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight";
    const HC_MLP_BLOCK: &str =
        "model.language_model.layers.0.mlp_hyper_connection.block_inject_weight.weight";
    const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

    pub(crate) struct Args {
        pub(crate) root: PathBuf,
        pub(crate) layer: usize,
        pub(crate) prefix_layers: usize,
        pub(crate) warmup: usize,
        pub(crate) reps: usize,
        pub(crate) out: PathBuf,
        pub(crate) state_out: PathBuf,
        pub(crate) state_output: Option<PathBuf>,
        pub(crate) base_state: Option<PathBuf>,
        pub(crate) compact_experts: bool,
        /// Opt-in protected probe: hand the previous layer's Metal buffer
        /// directly to the next layer.  CPU oracle work remains available for
        /// routing/diagnostics, but no GPU state snapshot is required between
        /// layers.  The default exact path is unchanged.
        pub(crate) device_resident: bool,
        /// Optional diagnostic reads on every layer while retaining device
        /// state as the required activation handoff.
        pub(crate) deep_verification: bool,
    }

    struct Tensor {
        name: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
        sha256: String,
    }

    struct LayerWeights {
        hc_attn_norm: Tensor,
        hc_attn_down: Tensor,
        hc_attn_up: Tensor,
        hc_attn_block: Tensor,
        qkv: Tensor,
        z: Tensor,
        b: Tensor,
        a: Tensor,
        conv: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        linear_norm: Tensor,
        out_proj: Tensor,
        router: Tensor,
        expert_gate_up: Tensor,
        expert_down: Tensor,
        shared_gate: Tensor,
        shared_up: Tensor,
        shared_down: Tensor,
        shared_scalar: Tensor,
        hc_mlp_norm: Tensor,
        hc_mlp_down: Tensor,
        hc_mlp_up: Tensor,
        hc_mlp_block: Tensor,
        /// Original expert ID -> compact bank slot.  None means dense bank.
        expert_lut: Option<Vec<u32>>,
    }

    struct DeviceWeights {
        hc_attn_norm: PinnedBuffer,
        hc_attn_down: PinnedBuffer,
        hc_attn_up: PinnedBuffer,
        hc_attn_block: PinnedBuffer,
        qkv: PinnedBuffer,
        z: PinnedBuffer,
        b: PinnedBuffer,
        a: PinnedBuffer,
        conv: PinnedBuffer,
        a_log: PinnedBuffer,
        dt_bias: PinnedBuffer,
        linear_norm: PinnedBuffer,
        out_proj: PinnedBuffer,
        router: PinnedBuffer,
        expert_gate_up: PinnedBuffer,
        expert_down: PinnedBuffer,
        shared_gate: PinnedBuffer,
        shared_up: PinnedBuffer,
        shared_down: PinnedBuffer,
        shared_scalar: PinnedBuffer,
        hc_mlp_norm: PinnedBuffer,
        hc_mlp_down: PinnedBuffer,
        hc_mlp_up: PinnedBuffer,
        hc_mlp_block: PinnedBuffer,
        expert_lut: Option<PinnedBuffer>,
        expert_count: usize,
    }

    struct GraphBuffers {
        base: PinnedBuffer,
        attn_norm: PinnedBuffer,
        attn_low_rank: PinnedBuffer,
        attn_low_rank_activation: PinnedBuffer,
        attn_gate_logits: PinnedBuffer,
        attn_input: PinnedBuffer,
        qkv_projection: PinnedBuffer,
        z_projection: PinnedBuffer,
        b_projection: PinnedBuffer,
        a_projection: PinnedBuffer,
        conv_state: PinnedBuffer,
        repeated_query: PinnedBuffer,
        repeated_key: PinnedBuffer,
        convolved_value: PinnedBuffer,
        z: PinnedBuffer,
        decay: PinnedBuffer,
        beta: PinnedBuffer,
        recurrent_state: PinnedBuffer,
        recurrent_output: PinnedBuffer,
        gated_output: PinnedBuffer,
        attn_block_output: PinnedBuffer,
        attn_block_logits: PinnedBuffer,
        post_attn_state: PinnedBuffer,
        mlp_norm: PinnedBuffer,
        mlp_low_rank: PinnedBuffer,
        mlp_low_rank_activation: PinnedBuffer,
        mlp_gate_logits: PinnedBuffer,
        mlp_input: PinnedBuffer,
        router_logits: PinnedBuffer,
        route_ids: PinnedBuffer,
        route_weights: PinnedBuffer,
        routed_activation: PinnedBuffer,
        routed_outputs: PinnedBuffer,
        routed_sum: PinnedBuffer,
        shared_activation: PinnedBuffer,
        shared_output: PinnedBuffer,
        shared_scalar: PinnedBuffer,
        shared_gated_output: PinnedBuffer,
        moe_output: PinnedBuffer,
        mlp_block_logits: PinnedBuffer,
        final_state: PinnedBuffer,
    }

    struct CpuResult {
        base: Vec<f32>,
        attn_norm: Vec<f32>,
        attn_input: Vec<f32>,
        qkv_projection: Vec<f32>,
        z_projection: Vec<f32>,
        b_projection: Vec<f32>,
        a_projection: Vec<f32>,
        repeated_query: Vec<f32>,
        repeated_key: Vec<f32>,
        convolved_value: Vec<f32>,
        decay: Vec<f32>,
        beta: Vec<f32>,
        recurrent_output: Vec<f32>,
        gated_output: Vec<f32>,
        attn_block_output: Vec<f32>,
        attn_block_logits: Vec<f32>,
        post_attn_state: Vec<f32>,
        mlp_norm: Vec<f32>,
        mlp_input: Vec<f32>,
        router_logits: Vec<f32>,
        route_ids: Vec<u32>,
        route_weights: Vec<f32>,
        routed_sum: Vec<f32>,
        shared_gated_output: Vec<f32>,
        moe_output: Vec<f32>,
        mlp_block_logits: Vec<f32>,
        final_state: Vec<f32>,
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn parse_usize(value: Option<String>, flag: &str) -> Result<usize, Box<dyn Error>> {
        value
            .ok_or_else(|| format!("missing value after {flag}"))?
            .parse::<usize>()
            .map_err(|error| format!("invalid {flag}: {error}").into())
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let repo = repository_root();
        let mut args = Args {
            root: env::var_os("HCLI_FLASH_NEXT_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_ROOT)),
            layer: 0,
            prefix_layers: 1,
            warmup: DEFAULT_WARMUP,
            reps: DEFAULT_REPS,
            out: repo.join("receipts/headless/FLASH_NOETIC_COMPLETE_LAYER0_NATIVE.json"),
            state_out: repo.join("receipts/headless/FLASH_LINEAR_PREFIX_L2_STATE.f32"),
            state_output: env::var_os("HCLI_FLASH_STATE_OUTPUT").map(PathBuf::from),
            base_state: env::var_os("HCLI_FLASH_BASE_STATE").map(PathBuf::from),
            compact_experts: false,
            device_resident: false,
            deep_verification: false,
        };
        let mut out_explicit = false;
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--root" => args.root = PathBuf::from(values.next().ok_or("missing --root")?),
                "--layer" => args.layer = parse_usize(values.next(), &flag)?,
                "--prefix-layers" => args.prefix_layers = parse_usize(values.next(), &flag)?,
                "--warmup" => args.warmup = parse_usize(values.next(), &flag)?,
                "--reps" => args.reps = parse_usize(values.next(), &flag)?,
                "--out" => {
                    args.out = PathBuf::from(values.next().ok_or("missing --out")?);
                    out_explicit = true;
                }
                "--state-out" => {
                    args.state_out = PathBuf::from(values.next().ok_or("missing --state-out")?)
                }
                "--state-output" => {
                    args.state_output = Some(PathBuf::from(
                        values.next().ok_or("missing --state-output")?,
                    ))
                }
                "--base-state" => {
                    args.base_state =
                        Some(PathBuf::from(values.next().ok_or("missing --base-state")?))
                }
                "--compact-experts" => args.compact_experts = true,
                "--device-resident" => args.device_resident = true,
                "--deep-verification" => args.deep_verification = true,
                "--help" | "-h" => {
                    println!(
                        "usage: flash_noetic_complete_layer0 [--root DIR] [--layer N] [--prefix-layers N] [--warmup N] [--reps N] [--out FILE] [--state-out FILE] [--state-output F32] [--base-state F32] [--compact-experts] [--device-resident] [--deep-verification]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if args.warmup > 16 {
            return Err("--warmup must be <= 16".into());
        }
        if args.layer >= 48 {
            return Err("--layer must be in 0..48".into());
        }
        if args.prefix_layers == 0 || args.prefix_layers > 3 {
            return Err("--prefix-layers must be in 1..=3".into());
        }
        if args.layer + args.prefix_layers > 48 {
            return Err("--layer + --prefix-layers must be <= 48".into());
        }
        if !out_explicit && args.layer != 0 {
            args.out = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_COMPLETE_LAYER{}_NATIVE.json",
                args.layer
            ));
        }
        if args.reps == 0 || args.reps > 32 {
            return Err("--reps must be in 1..=32".into());
        }
        Ok(args)
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn zero_bytes(elements: usize, width: usize) -> Vec<u8> {
        vec![0u8; elements * width]
    }

    fn bf16_at(bytes: &[u8], element: usize) -> f32 {
        let offset = element * 2;
        f32::from_bits((u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32) << 16)
    }

    fn bf16_vec(bytes: &[u8], elements: usize) -> Vec<f32> {
        (0..elements).map(|index| bf16_at(bytes, index)).collect()
    }

    fn load_tensor(
        index: &SourceBf16Index,
        name: &str,
        shape: &[usize],
    ) -> Result<Tensor, Box<dyn Error>> {
        let elements = shape.iter().try_fold(1usize, |acc, &value| {
            acc.checked_mul(value)
                .ok_or_else(|| format!("tensor {name} shape overflows usize"))
        })?;
        let bytes = index.read_raw(name)?;
        let expected = elements
            .checked_mul(2)
            .ok_or_else(|| format!("tensor {name} byte count overflows usize"))?;
        if bytes.len() != expected {
            return Err(format!(
                "tensor {name} source bytes={} expected={} for shape={shape:?}",
                bytes.len(),
                expected
            )
            .into());
        }
        let sha256 = sha256_bytes(&bytes);
        eprintln!("source: {:<88} {:>10} B", name, bytes.len());
        Ok(Tensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            bytes,
            sha256,
        })
    }

    fn layer_tensor_name(layer: usize, layer_zero_name: &str) -> String {
        let marker = ".0.";
        let Some(offset) = layer_zero_name.find(marker) else {
            return layer_zero_name.to_owned();
        };
        let mut name = String::with_capacity(layer_zero_name.len() + 3);
        name.push_str(&layer_zero_name[..offset]);
        name.push('.');
        name.push_str(&layer.to_string());
        name.push_str(&layer_zero_name[offset + marker.len() - 1..]);
        name
    }

    fn load_layer_weights(
        index: &SourceBf16Index,
        layer: usize,
    ) -> Result<LayerWeights, Box<dyn Error>> {
        let name = |base: &str| layer_tensor_name(layer, base);
        Ok(LayerWeights {
            hc_attn_norm: load_tensor(index, &name(HC_ATTN_NORM), &[HC_ELEMENTS])?,
            hc_attn_down: load_tensor(index, &name(HC_ATTN_DOWN), &[HC_LOWRANK, HC_ELEMENTS])?,
            hc_attn_up: load_tensor(index, &name(HC_ATTN_UP), &[HC_ELEMENTS, HC_LOWRANK])?,
            hc_attn_block: load_tensor(index, &name(HC_ATTN_BLOCK), &[STREAMS, HC_ELEMENTS])?,
            qkv: load_tensor(index, &name(QKV), &[QKV_ELEMENTS, HIDDEN])?,
            z: load_tensor(index, &name(Z), &[VALUE_ELEMENTS, HIDDEN])?,
            b: load_tensor(index, &name(B), &[VALUE_HEADS, HIDDEN])?,
            a: load_tensor(index, &name(A), &[VALUE_HEADS, HIDDEN])?,
            conv: load_tensor(index, &name(CONV), &[QKV_ELEMENTS, 1, CONV_KERNEL])?,
            a_log: load_tensor(index, &name(A_LOG), &[VALUE_HEADS])?,
            dt_bias: load_tensor(index, &name(DT_BIAS), &[VALUE_HEADS])?,
            linear_norm: load_tensor(index, &name(LINEAR_NORM), &[VALUE_HEAD_DIM])?,
            out_proj: load_tensor(index, &name(OUT_PROJ), &[HIDDEN, VALUE_ELEMENTS])?,
            router: load_tensor(index, &name(ROUTER), &[EXPERTS, HIDDEN])?,
            expert_gate_up: load_tensor(
                index,
                &name(EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            )?,
            expert_down: load_tensor(index, &name(EXPERT_DOWN), &[EXPERTS, HIDDEN, INTERMEDIATE])?,
            shared_gate: load_tensor(index, &name(SHARED_GATE), &[INTERMEDIATE, HIDDEN])?,
            shared_up: load_tensor(index, &name(SHARED_UP), &[INTERMEDIATE, HIDDEN])?,
            shared_down: load_tensor(index, &name(SHARED_DOWN), &[HIDDEN, INTERMEDIATE])?,
            shared_scalar: load_tensor(index, &name(SHARED_SCALAR), &[1, HIDDEN])?,
            hc_mlp_norm: load_tensor(index, &name(HC_MLP_NORM), &[HC_ELEMENTS])?,
            hc_mlp_down: load_tensor(index, &name(HC_MLP_DOWN), &[HC_LOWRANK, HC_ELEMENTS])?,
            hc_mlp_up: load_tensor(index, &name(HC_MLP_UP), &[HC_ELEMENTS, HC_LOWRANK])?,
            hc_mlp_block: load_tensor(index, &name(HC_MLP_BLOCK), &[STREAMS, HC_ELEMENTS])?,
            expert_lut: None,
        })
    }

    fn empty_tensor(name: &str, shape: &[usize]) -> Tensor {
        Tensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            bytes: Vec::new(),
            sha256: sha256_bytes(&[]),
        }
    }

    /// Read only the selected expert rows, preserving contiguous compact-bank
    /// layout.  The original expert IDs remain in the route buffer and are
    /// translated by the device LUT.
    fn load_tensor_rows(
        index: &SourceBf16Index,
        name: &str,
        shape: &[usize],
        experts: &[u32],
    ) -> Result<Tensor, Box<dyn Error>> {
        let rows = shape
            .get(0)
            .copied()
            .ok_or("expert tensor has no leading dimension")?;
        let row_elements = shape[1..]
            .iter()
            .try_fold(1usize, |acc, &v| acc.checked_mul(v))
            .ok_or("expert row shape overflow")?;
        if rows != EXPERTS || experts.iter().any(|&e| e as usize >= rows) {
            return Err(format!("compact expert selection geometry invalid for {name}").into());
        }
        let row_bytes = row_elements
            .checked_mul(2)
            .ok_or("expert row bytes overflow")?;
        let mut bytes = Vec::with_capacity(experts.len() * row_bytes);
        for &expert in experts {
            bytes.extend_from_slice(&index.read_raw_range(
                name,
                (expert as usize) * row_bytes,
                row_bytes,
            )?);
        }
        let mut compact_shape = shape.to_vec();
        compact_shape[0] = experts.len();
        eprintln!(
            "source: {:<88} {:>10} B (compact {} experts)",
            name,
            bytes.len(),
            experts.len()
        );
        Ok(Tensor {
            name: name.to_owned(),
            shape: compact_shape,
            sha256: sha256_bytes(&bytes),
            bytes,
        })
    }

    fn load_layer_weights_compact(
        index: &SourceBf16Index,
        layer: usize,
        base: &[f32],
    ) -> Result<LayerWeights, Box<dyn Error>> {
        let name = |base: &str| layer_tensor_name(layer, base);
        // Everything except the expert banks is loaded exactly as in the
        // dense control.  Empty expert placeholders allow the pre-router
        // computation to determine the ten selected IDs before bank I/O.
        let mut weights = LayerWeights {
            hc_attn_norm: load_tensor(index, &name(HC_ATTN_NORM), &[HC_ELEMENTS])?,
            hc_attn_down: load_tensor(index, &name(HC_ATTN_DOWN), &[HC_LOWRANK, HC_ELEMENTS])?,
            hc_attn_up: load_tensor(index, &name(HC_ATTN_UP), &[HC_ELEMENTS, HC_LOWRANK])?,
            hc_attn_block: load_tensor(index, &name(HC_ATTN_BLOCK), &[STREAMS, HC_ELEMENTS])?,
            qkv: load_tensor(index, &name(QKV), &[QKV_ELEMENTS, HIDDEN])?,
            z: load_tensor(index, &name(Z), &[VALUE_ELEMENTS, HIDDEN])?,
            b: load_tensor(index, &name(B), &[VALUE_HEADS, HIDDEN])?,
            a: load_tensor(index, &name(A), &[VALUE_HEADS, HIDDEN])?,
            conv: load_tensor(index, &name(CONV), &[QKV_ELEMENTS, 1, CONV_KERNEL])?,
            a_log: load_tensor(index, &name(A_LOG), &[VALUE_HEADS])?,
            dt_bias: load_tensor(index, &name(DT_BIAS), &[VALUE_HEADS])?,
            linear_norm: load_tensor(index, &name(LINEAR_NORM), &[VALUE_HEAD_DIM])?,
            out_proj: load_tensor(index, &name(OUT_PROJ), &[HIDDEN, VALUE_ELEMENTS])?,
            router: load_tensor(index, &name(ROUTER), &[EXPERTS, HIDDEN])?,
            expert_gate_up: empty_tensor(
                &name(EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            ),
            expert_down: empty_tensor(&name(EXPERT_DOWN), &[EXPERTS, HIDDEN, INTERMEDIATE]),
            shared_gate: load_tensor(index, &name(SHARED_GATE), &[INTERMEDIATE, HIDDEN])?,
            shared_up: load_tensor(index, &name(SHARED_UP), &[INTERMEDIATE, HIDDEN])?,
            shared_down: load_tensor(index, &name(SHARED_DOWN), &[HIDDEN, INTERMEDIATE])?,
            shared_scalar: load_tensor(index, &name(SHARED_SCALAR), &[1, HIDDEN])?,
            hc_mlp_norm: load_tensor(index, &name(HC_MLP_NORM), &[HC_ELEMENTS])?,
            hc_mlp_down: load_tensor(index, &name(HC_MLP_DOWN), &[HC_LOWRANK, HC_ELEMENTS])?,
            hc_mlp_up: load_tensor(index, &name(HC_MLP_UP), &[HC_ELEMENTS, HC_LOWRANK])?,
            hc_mlp_block: load_tensor(index, &name(HC_MLP_BLOCK), &[STREAMS, HC_ELEMENTS])?,
            expert_lut: None,
        };
        let mlp_input = source_mlp_input(&weights, base);
        let router_logits = raw_matvec(&weights.router.bytes, 0, EXPERTS, HIDDEN, &mlp_input);
        let (route_ids, _) = topk_router(&router_logits);
        let mut lut = vec![u32::MAX; EXPERTS];
        for (slot, &expert) in route_ids.iter().enumerate() {
            lut[expert as usize] = slot as u32;
        }
        weights.expert_gate_up = load_tensor_rows(
            index,
            &name(EXPERT_GATE_UP),
            &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            &route_ids,
        )?;
        weights.expert_down = load_tensor_rows(
            index,
            &name(EXPERT_DOWN),
            &[EXPERTS, HIDDEN, INTERMEDIATE],
            &route_ids,
        )?;
        weights.expert_lut = Some(lut);
        Ok(weights)
    }

    /// Route-union variant of the compact loader. The caller supplies the
    /// union of expert IDs needed by a verified token window, so the payload
    /// read is still route-before-payload but does not assume that one token's
    /// top-k set remains valid for every later token.
    fn load_layer_weights_compact_union(
        index: &SourceBf16Index,
        layer: usize,
        route_ids: &[u32],
    ) -> Result<LayerWeights, Box<dyn Error>> {
        let name = |base: &str| layer_tensor_name(layer, base);
        let mut selected = route_ids.to_vec();
        selected.sort_unstable();
        selected.dedup();
        if selected.is_empty()
            || selected.len() > EXPERTS
            || selected.iter().any(|&e| e as usize >= EXPERTS)
        {
            return Err(format!(
                "compact expert union invalid for layer {layer}: {} ids",
                selected.len()
            )
            .into());
        }
        let mut lut = vec![u32::MAX; EXPERTS];
        for (slot, &expert) in selected.iter().enumerate() {
            lut[expert as usize] = slot as u32;
        }
        Ok(LayerWeights {
            hc_attn_norm: load_tensor(index, &name(HC_ATTN_NORM), &[HC_ELEMENTS])?,
            hc_attn_down: load_tensor(index, &name(HC_ATTN_DOWN), &[HC_LOWRANK, HC_ELEMENTS])?,
            hc_attn_up: load_tensor(index, &name(HC_ATTN_UP), &[HC_ELEMENTS, HC_LOWRANK])?,
            hc_attn_block: load_tensor(index, &name(HC_ATTN_BLOCK), &[STREAMS, HC_ELEMENTS])?,
            qkv: load_tensor(index, &name(QKV), &[QKV_ELEMENTS, HIDDEN])?,
            z: load_tensor(index, &name(Z), &[VALUE_ELEMENTS, HIDDEN])?,
            b: load_tensor(index, &name(B), &[VALUE_HEADS, HIDDEN])?,
            a: load_tensor(index, &name(A), &[VALUE_HEADS, HIDDEN])?,
            conv: load_tensor(index, &name(CONV), &[QKV_ELEMENTS, 1, CONV_KERNEL])?,
            a_log: load_tensor(index, &name(A_LOG), &[VALUE_HEADS])?,
            dt_bias: load_tensor(index, &name(DT_BIAS), &[VALUE_HEADS])?,
            linear_norm: load_tensor(index, &name(LINEAR_NORM), &[VALUE_HEAD_DIM])?,
            out_proj: load_tensor(index, &name(OUT_PROJ), &[HIDDEN, VALUE_ELEMENTS])?,
            router: load_tensor(index, &name(ROUTER), &[EXPERTS, HIDDEN])?,
            expert_gate_up: load_tensor_rows(
                index,
                &name(EXPERT_GATE_UP),
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
                &selected,
            )?,
            expert_down: load_tensor_rows(
                index,
                &name(EXPERT_DOWN),
                &[EXPERTS, HIDDEN, INTERMEDIATE],
                &selected,
            )?,
            shared_gate: load_tensor(index, &name(SHARED_GATE), &[INTERMEDIATE, HIDDEN])?,
            shared_up: load_tensor(index, &name(SHARED_UP), &[INTERMEDIATE, HIDDEN])?,
            shared_down: load_tensor(index, &name(SHARED_DOWN), &[HIDDEN, INTERMEDIATE])?,
            shared_scalar: load_tensor(index, &name(SHARED_SCALAR), &[1, HIDDEN])?,
            hc_mlp_norm: load_tensor(index, &name(HC_MLP_NORM), &[HC_ELEMENTS])?,
            hc_mlp_down: load_tensor(index, &name(HC_MLP_DOWN), &[HC_LOWRANK, HC_ELEMENTS])?,
            hc_mlp_up: load_tensor(index, &name(HC_MLP_UP), &[HC_ELEMENTS, HC_LOWRANK])?,
            hc_mlp_block: load_tensor(index, &name(HC_MLP_BLOCK), &[STREAMS, HC_ELEMENTS])?,
            expert_lut: Some(lut),
        })
    }

    fn load_embedding_row(
        index: &SourceBf16Index,
        vocab: usize,
        token_id: usize,
    ) -> Result<(Vec<f32>, String, usize), Box<dyn Error>> {
        if token_id >= vocab {
            return Err(format!("token {token_id} is outside vocab {vocab}").into());
        }
        eprintln!("source: reading embedding row token={token_id} from {EMBEDDING}");
        let row_start = token_id
            .checked_mul(HIDDEN)
            .and_then(|value| value.checked_mul(2))
            .ok_or("embedding row byte offset overflows usize")?;
        let row_bytes = index.read_raw_range(EMBEDDING, row_start, HIDDEN * 2)?;
        if row_bytes.len() != HIDDEN * 2 {
            return Err(format!(
                "embedding row source bytes={} expected={} for hidden={HIDDEN}",
                row_bytes.len(),
                HIDDEN * 2
            )
            .into());
        }
        let row = (0..HIDDEN)
            .map(|index| bf16_at(&row_bytes, index))
            .collect::<Vec<_>>();
        let digest = sha256_bytes(&row_bytes);
        Ok((row, digest, row_bytes.len()))
    }

    fn load_bos_embedding(
        index: &SourceBf16Index,
        vocab: usize,
    ) -> Result<(Vec<f32>, String, usize), Box<dyn Error>> {
        load_embedding_row(index, vocab, BOS_TOKEN_ID)
    }

    fn load_f32_state(path: &Path) -> Result<(Vec<f32>, String), Box<dyn Error>> {
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
        Ok((values, sha256_bytes(&bytes)))
    }

    fn validate_manifest(root: &Path) -> Result<Value, Box<dyn Error>> {
        let manifest_bytes = fs::read(MANIFEST_PATH)?;
        let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
        if manifest.get("repo").and_then(Value::as_str) != Some(REPO_ID)
            || manifest.get("revision").and_then(Value::as_str) != Some(PINNED_REVISION)
            || manifest.get("resolved_sha").and_then(Value::as_str) != Some(PINNED_REVISION)
        {
            return Err("ModelLake manifest is not the pinned Flash revision".into());
        }
        let manifest_root = manifest
            .get("path")
            .and_then(Value::as_str)
            .ok_or("ModelLake manifest has no specimen path")?;
        if Path::new(manifest_root).canonicalize()? != root.canonicalize()? {
            return Err("selected root does not match the pinned ModelLake manifest".into());
        }
        Ok(json!({
            "path": MANIFEST_PATH,
            "sha256": sha256_bytes(&manifest_bytes),
            "repo": REPO_ID,
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "n_files": manifest.get("n_files"),
            "bytes": manifest.get("bytes"),
            "label": "[V]"
        }))
    }

    fn validate_config(root: &Path) -> Result<(Value, String, usize), Box<dyn Error>> {
        let bytes = fs::read(root.join("config.json"))?;
        let config: Value = serde_json::from_slice(&bytes)?;
        let text = config
            .get("text_config")
            .ok_or("Flash config has no text_config")?;
        let hidden = text
            .get("hidden_size")
            .and_then(Value::as_u64)
            .ok_or("Flash text_config has no hidden_size")? as usize;
        let layers = text
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .ok_or("Flash text_config has no num_hidden_layers")? as usize;
        let vocab = text
            .get("vocab_size")
            .and_then(Value::as_u64)
            .ok_or("Flash text_config has no vocab_size")? as usize;
        let layer_type = text
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
            .and_then(Value::as_str)
            .unwrap_or("");
        if hidden != HIDDEN || layers != 48 || layer_type != "linear_attention" {
            return Err(format!(
                "Flash config drifted: hidden={hidden}, layers={layers}, layer0={layer_type:?}"
            )
            .into());
        }
        let bos = text
            .get("bos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(BOS_TOKEN_ID as u64) as usize;
        if bos != BOS_TOKEN_ID {
            return Err(
                format!("Flash BOS token drifted: config={bos} expected={BOS_TOKEN_ID}").into(),
            );
        }
        Ok((config, sha256_bytes(&bytes), vocab))
    }

    fn source_buffer(ctx: &MetalContext, tensor: &Tensor) -> Result<PinnedBuffer, Box<dyn Error>> {
        Ok(ctx.new_buffer_with_bytes_checked(&tensor.bytes)?)
    }

    fn empty_f32(ctx: &MetalContext, elements: usize) -> Result<PinnedBuffer, Box<dyn Error>> {
        Ok(ctx.new_buffer_checked(elements * std::mem::size_of::<f32>())?)
    }

    fn empty_u32(ctx: &MetalContext, elements: usize) -> Result<PinnedBuffer, Box<dyn Error>> {
        Ok(ctx.new_buffer_checked(elements * std::mem::size_of::<u32>())?)
    }

    fn initial_f32(ctx: &MetalContext, values: &[f32]) -> Result<PinnedBuffer, Box<dyn Error>> {
        Ok(ctx.new_buffer_with_bytes_checked(&f32_bytes(values))?)
    }

    fn initial_u32(ctx: &MetalContext, values: &[u32]) -> Result<PinnedBuffer, Box<dyn Error>> {
        Ok(ctx.new_buffer_with_bytes_checked(&u32_bytes(values))?)
    }

    fn load_device_weights(
        ctx: &MetalContext,
        weights: &LayerWeights,
    ) -> Result<DeviceWeights, Box<dyn Error>> {
        Ok(DeviceWeights {
            hc_attn_norm: source_buffer(ctx, &weights.hc_attn_norm)?,
            hc_attn_down: source_buffer(ctx, &weights.hc_attn_down)?,
            hc_attn_up: source_buffer(ctx, &weights.hc_attn_up)?,
            hc_attn_block: source_buffer(ctx, &weights.hc_attn_block)?,
            qkv: source_buffer(ctx, &weights.qkv)?,
            z: source_buffer(ctx, &weights.z)?,
            b: source_buffer(ctx, &weights.b)?,
            a: source_buffer(ctx, &weights.a)?,
            conv: source_buffer(ctx, &weights.conv)?,
            a_log: source_buffer(ctx, &weights.a_log)?,
            dt_bias: source_buffer(ctx, &weights.dt_bias)?,
            linear_norm: source_buffer(ctx, &weights.linear_norm)?,
            out_proj: source_buffer(ctx, &weights.out_proj)?,
            router: source_buffer(ctx, &weights.router)?,
            expert_gate_up: source_buffer(ctx, &weights.expert_gate_up)?,
            expert_down: source_buffer(ctx, &weights.expert_down)?,
            shared_gate: source_buffer(ctx, &weights.shared_gate)?,
            shared_up: source_buffer(ctx, &weights.shared_up)?,
            shared_down: source_buffer(ctx, &weights.shared_down)?,
            shared_scalar: source_buffer(ctx, &weights.shared_scalar)?,
            hc_mlp_norm: source_buffer(ctx, &weights.hc_mlp_norm)?,
            hc_mlp_down: source_buffer(ctx, &weights.hc_mlp_down)?,
            hc_mlp_up: source_buffer(ctx, &weights.hc_mlp_up)?,
            hc_mlp_block: source_buffer(ctx, &weights.hc_mlp_block)?,
            expert_lut: weights
                .expert_lut
                .as_ref()
                .map(|lut| {
                    source_buffer(
                        ctx,
                        &Tensor {
                            name: "expert_lut".into(),
                            shape: vec![lut.len()],
                            bytes: u32_bytes(lut),
                            sha256: sha256_bytes(&u32_bytes(lut)),
                        },
                    )
                })
                .transpose()?,
            expert_count: weights
                .expert_gate_up
                .shape
                .first()
                .copied()
                .unwrap_or(EXPERTS),
        })
    }

    fn repeat_streams(input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(HC_ELEMENTS);
        for _ in 0..STREAMS {
            output.extend_from_slice(input);
        }
        output
    }

    fn new_graph_buffers(ctx: &MetalContext, base: &[f32]) -> Result<GraphBuffers, Box<dyn Error>> {
        Ok(GraphBuffers {
            base: initial_f32(ctx, base)?,
            attn_norm: empty_f32(ctx, HC_ELEMENTS)?,
            attn_low_rank: empty_f32(ctx, HC_LOWRANK)?,
            attn_low_rank_activation: empty_f32(ctx, HC_LOWRANK)?,
            attn_gate_logits: empty_f32(ctx, HC_ELEMENTS)?,
            attn_input: empty_f32(ctx, HIDDEN)?,
            qkv_projection: empty_f32(ctx, QKV_ELEMENTS)?,
            z_projection: empty_f32(ctx, VALUE_ELEMENTS)?,
            b_projection: empty_f32(ctx, VALUE_HEADS)?,
            a_projection: empty_f32(ctx, VALUE_HEADS)?,
            conv_state: ctx.new_buffer_with_bytes_checked(&zero_bytes(CONV_STATE_ELEMENTS, 4))?,
            repeated_query: empty_f32(ctx, VALUE_ELEMENTS)?,
            repeated_key: empty_f32(ctx, VALUE_ELEMENTS)?,
            convolved_value: empty_f32(ctx, VALUE_ELEMENTS)?,
            z: empty_f32(ctx, VALUE_ELEMENTS)?,
            decay: empty_f32(ctx, VALUE_HEADS)?,
            beta: empty_f32(ctx, VALUE_HEADS)?,
            recurrent_state: ctx
                .new_buffer_with_bytes_checked(&zero_bytes(RECURRENT_STATE_ELEMENTS, 4))?,
            recurrent_output: empty_f32(ctx, VALUE_ELEMENTS)?,
            gated_output: empty_f32(ctx, VALUE_ELEMENTS)?,
            attn_block_output: empty_f32(ctx, HIDDEN)?,
            attn_block_logits: empty_f32(ctx, STREAMS)?,
            post_attn_state: empty_f32(ctx, HC_ELEMENTS)?,
            mlp_norm: empty_f32(ctx, HC_ELEMENTS)?,
            mlp_low_rank: empty_f32(ctx, HC_LOWRANK)?,
            mlp_low_rank_activation: empty_f32(ctx, HC_LOWRANK)?,
            mlp_gate_logits: empty_f32(ctx, HC_ELEMENTS)?,
            mlp_input: empty_f32(ctx, HIDDEN)?,
            router_logits: empty_f32(ctx, EXPERTS)?,
            route_ids: empty_u32(ctx, TOP_K)?,
            route_weights: empty_f32(ctx, TOP_K)?,
            routed_activation: empty_f32(ctx, TOP_K * INTERMEDIATE)?,
            routed_outputs: empty_f32(ctx, TOP_K * HIDDEN)?,
            routed_sum: empty_f32(ctx, HIDDEN)?,
            shared_activation: empty_f32(ctx, INTERMEDIATE)?,
            shared_output: empty_f32(ctx, HIDDEN)?,
            shared_scalar: empty_f32(ctx, 1)?,
            shared_gated_output: empty_f32(ctx, HIDDEN)?,
            moe_output: empty_f32(ctx, HIDDEN)?,
            mlp_block_logits: empty_f32(ctx, STREAMS)?,
            final_state: empty_f32(ctx, HC_ELEMENTS)?,
        })
    }

    fn reset_states(ctx: &MetalContext, graph: &GraphBuffers) {
        MetalContext::write_buffer_bytes(
            &graph.conv_state,
            &zero_bytes(CONV_STATE_ELEMENTS, std::mem::size_of::<f32>()),
        );
        MetalContext::write_buffer_bytes(
            &graph.recurrent_state,
            &zero_bytes(RECURRENT_STATE_ELEMENTS, std::mem::size_of::<f32>()),
        );
        let _ = ctx;
    }

    fn fused_router_topk() -> bool {
        env::var("HAWKING_FLASH_ROUTER_TOPK_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    }

    fn fused_hc_router() -> bool {
        env::var("HAWKING_FLASH_HC_ROUTER_FUSED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    }

    fn fused_moe_vec4() -> bool {
        env::var("HAWKING_FLASH_MOE_VEC4")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    }

    fn expected_graph_dispatches_for_compact(compact: bool) -> usize {
        let base: usize = if compact { 13 } else { 16 };
        base.saturating_sub(if fused_hc_router() {
            2
        } else if fused_router_topk() {
            1
        } else {
            0
        })
    }

    fn expected_graph_dispatches(weights: &DeviceWeights) -> usize {
        expected_graph_dispatches_for_compact(weights.expert_lut.is_some())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_graph(
        tcb: &mut TokenCommandBuffer<'_>,
        weights: &DeviceWeights,
        graph: &GraphBuffers,
    ) -> Result<(), Box<dyn Error>> {
        qwen_next_hyperconnection_input_fused_with_block_tcb(
            tcb,
            &graph.base,
            &weights.hc_attn_norm,
            &weights.hc_attn_down,
            &weights.hc_attn_up,
            &graph.attn_norm,
            &graph.attn_low_rank,
            &graph.attn_low_rank_activation,
            &graph.attn_gate_logits,
            &graph.attn_input,
            &weights.hc_attn_block,
            &graph.attn_block_logits,
            HIDDEN,
            STREAMS,
            HC_LOWRANK,
            EPS,
            STREAMS as f32,
        )?;
        native_bf16_dual_seq_tcb(
            tcb,
            &weights.qkv,
            &weights.z,
            &graph.attn_input,
            &graph.qkv_projection,
            &graph.z_projection,
            QKV_ELEMENTS,
            VALUE_ELEMENTS,
            HIDDEN,
        )?;
        native_bf16_dual_seq_tcb(
            tcb,
            &weights.b,
            &weights.a,
            &graph.attn_input,
            &graph.b_projection,
            &graph.a_projection,
            VALUE_HEADS,
            VALUE_HEADS,
            HIDDEN,
        )?;
        qwen_next_qkv_split_rearrange_conv_l2_source_bf16_tcb(
            tcb,
            &graph.qkv_projection,
            &graph.z_projection,
            &weights.conv,
            &graph.conv_state,
            &graph.repeated_query,
            &graph.repeated_key,
            &graph.convolved_value,
            &graph.z,
            KEY_HEADS,
            VALUES_PER_KEY_HEAD,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CONV_KERNEL,
            EPS,
        )?;
        qwen_next_ba_split_to_decay_beta_source_bf16_tcb(
            tcb,
            &graph.b_projection,
            &graph.a_projection,
            &weights.a_log,
            &weights.dt_bias,
            &graph.decay,
            &graph.beta,
            KEY_HEADS,
            VALUES_PER_KEY_HEAD,
        )?;
        qwen_next_gated_delta_decode_single_at_state_offset_tcb(
            tcb,
            &graph.recurrent_state,
            0,
            &graph.repeated_query,
            &graph.repeated_key,
            &graph.convolved_value,
            &graph.decay,
            &graph.beta,
            &graph.recurrent_output,
            VALUE_HEADS,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        )?;
        qwen_next_deltanet_source_bf16_gated_rmsnorm_tcb(
            tcb,
            &graph.recurrent_output,
            &graph.z,
            &weights.linear_norm,
            &graph.gated_output,
            VALUE_HEADS,
            VALUE_HEAD_DIM,
            EPS,
        )?;
        native_bf16_gemv_hyperconnection_combine_tcb(
            tcb,
            &weights.out_proj,
            &graph.gated_output,
            &graph.base,
            &graph.attn_block_logits,
            &graph.attn_block_output,
            &graph.post_attn_state,
            HIDDEN,
            VALUE_ELEMENTS,
            STREAMS,
            STREAMS as f32,
        )?;

        if fused_hc_router() {
            qwen_next_hyperconnection_input_fused_with_block_router_topk_tcb(
                tcb,
                &graph.post_attn_state,
                &weights.hc_mlp_norm,
                &weights.hc_mlp_down,
                &weights.hc_mlp_up,
                &graph.mlp_norm,
                &graph.mlp_low_rank,
                &graph.mlp_low_rank_activation,
                &graph.mlp_gate_logits,
                &graph.mlp_input,
                &weights.hc_mlp_block,
                &graph.mlp_block_logits,
                &weights.router,
                &weights.shared_scalar,
                &graph.router_logits,
                &graph.shared_scalar,
                &graph.route_ids,
                &graph.route_weights,
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
                tcb,
                &graph.post_attn_state,
                &weights.hc_mlp_norm,
                &weights.hc_mlp_down,
                &weights.hc_mlp_up,
                &graph.mlp_norm,
                &graph.mlp_low_rank,
                &graph.mlp_low_rank_activation,
                &graph.mlp_gate_logits,
                &graph.mlp_input,
                &weights.hc_mlp_block,
                &graph.mlp_block_logits,
                HIDDEN,
                STREAMS,
                HC_LOWRANK,
                EPS,
                STREAMS as f32,
            )?;
        }
        if !fused_hc_router() && fused_router_topk() {
            qwen_next_bf16_router_topk_shared_tcb(
                tcb,
                &weights.router,
                &weights.shared_scalar,
                &graph.mlp_input,
                &graph.router_logits,
                &graph.shared_scalar,
                &graph.route_ids,
                &graph.route_weights,
                EXPERTS,
                TOP_K,
                HIDDEN,
                true,
            )?;
        } else if !fused_hc_router() {
            native_bf16_dual_seq_tcb(
                tcb,
                &weights.router,
                &weights.shared_scalar,
                &graph.mlp_input,
                &graph.router_logits,
                &graph.shared_scalar,
                EXPERTS,
                1,
                HIDDEN,
            )?;
            moe_topk_gate_tcb_ex(
                tcb,
                &graph.router_logits,
                &graph.route_ids,
                &graph.route_weights,
                EXPERTS,
                TOP_K,
                true,
            )?;
        }
        if let Some(lut) = weights.expert_lut.as_ref() {
            qwen_next_bf16_compact_expert_gate_up_shared_swiglu_tcb(
                tcb,
                &weights.expert_gate_up,
                &graph.route_ids,
                lut,
                &graph.mlp_input,
                &graph.routed_activation,
                &weights.shared_gate,
                &weights.shared_up,
                &graph.shared_activation,
                weights.expert_count,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
                EXPERTS,
            )?;
        } else {
            qwen_next_bf16_expert_gate_up_swiglu_tcb(
                tcb,
                &weights.expert_gate_up,
                &graph.route_ids,
                &graph.mlp_input,
                &graph.routed_activation,
                EXPERTS,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
            )?;
            qwen_next_bf16_expert_down_tcb(
                tcb,
                &weights.expert_down,
                &graph.route_ids,
                &graph.routed_activation,
                &graph.routed_outputs,
                EXPERTS,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
            )?;
        }
        if weights.expert_lut.is_none() {
            native_bf16_swiglu_seq_tcb(
                tcb,
                &weights.shared_gate,
                &weights.shared_up,
                &graph.mlp_input,
                &graph.shared_activation,
                INTERMEDIATE,
                HIDDEN,
            )?;
        }
        if let Some(lut) = weights.expert_lut.as_ref() {
            qwen_next_bf16_compact_expert_down_shared_direct_hc_tcb(
                tcb,
                &weights.expert_down,
                &graph.route_ids,
                lut,
                &graph.routed_activation,
                &graph.route_weights,
                &weights.shared_down,
                &graph.shared_activation,
                &graph.shared_scalar,
                &graph.routed_sum,
                &graph.shared_output,
                &graph.shared_gated_output,
                &graph.moe_output,
                &graph.post_attn_state,
                &graph.mlp_block_logits,
                &graph.final_state,
                weights.expert_count,
                TOP_K,
                INTERMEDIATE,
                HIDDEN,
                EXPERTS,
                STREAMS,
                STREAMS as f32,
            )?;
        } else {
            native_bf16_gemv_seq_tcb(
                tcb,
                &weights.shared_down,
                &graph.shared_activation,
                &graph.shared_output,
                HIDDEN,
                INTERMEDIATE,
            )?;
            qwen_next_moe_weighted_sum_add_shared_sigmoid_hc_tcb(
                tcb,
                &graph.routed_outputs,
                &graph.route_weights,
                &graph.shared_output,
                &graph.shared_scalar,
                &graph.routed_sum,
                &graph.shared_gated_output,
                &graph.moe_output,
                &graph.post_attn_state,
                &graph.mlp_block_logits,
                &graph.final_state,
                HIDDEN,
                TOP_K,
                STREAMS,
                STREAMS as f32,
            )?;
        }
        Ok(())
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

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn raw_matvec(
        bytes: &[u8],
        offset_elements: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let row_offset = offset_elements + row * cols;
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += bf16_at(bytes, row_offset + col) * x[col];
            }
            out[row] = acc;
        }
        out
    }

    fn grouped_hc_norm(input: &[f32], weight: &[u8]) -> Vec<f32> {
        let weights = bf16_vec(weight, HC_ELEMENTS);
        let mut out = vec![0.0f32; HC_ELEMENTS];
        for stream in 0..STREAMS {
            let base = stream * HIDDEN;
            let sum = input[base..base + HIDDEN]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let inverse_rms = (sum / HIDDEN as f32 + EPS).sqrt().recip();
            for index in 0..HIDDEN {
                let offset = base + index;
                out[offset] = input[offset] * inverse_rms * (1.0 + weights[offset]);
            }
        }
        out
    }

    fn hc_read_mix(normalized: &[f32], down: &[u8], up: &[u8]) -> Vec<f32> {
        let low_rank = raw_matvec(down, 0, HC_LOWRANK, HC_ELEMENTS, normalized);
        let activation = low_rank
            .iter()
            .map(|&value| silu(value / STREAMS as f32))
            .collect::<Vec<_>>();
        let gate_logits = raw_matvec(up, 0, HC_ELEMENTS, HC_LOWRANK, &activation);
        let mut mixed = vec![0.0f32; HIDDEN];
        for index in 0..HIDDEN {
            let mut sum = 0.0f32;
            for stream in 0..STREAMS {
                let offset = stream * HIDDEN + index;
                sum += sigmoid(gate_logits[offset]) * normalized[offset];
            }
            mixed[index] = sum / STREAMS as f32;
        }
        mixed
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

    fn source_conv(qkv: &[f32], weights: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let mut output = vec![0.0f32; QKV_ELEMENTS];
        let mut state = vec![0.0f32; CONV_STATE_ELEMENTS];
        for channel in 0..QKV_ELEMENTS {
            let state_base = channel * (CONV_KERNEL - 1);
            let weight_base = channel * CONV_KERNEL;
            let mut sum = 0.0f32;
            for tap in 0..(CONV_KERNEL - 1) {
                sum += state[state_base + tap] * bf16_at(weights, weight_base + tap);
            }
            state.copy_within(state_base + 1..state_base + CONV_KERNEL - 1, state_base);
            state[state_base + CONV_KERNEL - 2] = qkv[channel];
            sum += qkv[channel] * bf16_at(weights, weight_base + CONV_KERNEL - 1);
            output[channel] = silu(sum);
        }
        (output, state)
    }

    fn repeat_and_normalize(qkv: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let raw_query = &qkv[..KEY_ELEMENTS];
        let raw_key = &qkv[KEY_ELEMENTS..2 * KEY_ELEMENTS];
        let raw_value = &qkv[2 * KEY_ELEMENTS..];
        let mut query = vec![0.0f32; VALUE_ELEMENTS];
        let mut key = vec![0.0f32; VALUE_ELEMENTS];
        for key_head in 0..KEY_HEADS {
            let q = &raw_query[key_head * KEY_HEAD_DIM..(key_head + 1) * KEY_HEAD_DIM];
            let k = &raw_key[key_head * KEY_HEAD_DIM..(key_head + 1) * KEY_HEAD_DIM];
            let q_norm = q.iter().map(|value| value * value).sum::<f32>();
            let k_norm = k.iter().map(|value| value * value).sum::<f32>();
            let q_scale = (q_norm + EPS).sqrt().recip() * (KEY_HEAD_DIM as f32).sqrt().recip();
            let k_scale = (k_norm + EPS).sqrt().recip();
            for repeat in 0..VALUES_PER_KEY_HEAD {
                let base = (key_head * VALUES_PER_KEY_HEAD + repeat) * KEY_HEAD_DIM;
                for dim in 0..KEY_HEAD_DIM {
                    query[base + dim] = q[dim] * q_scale;
                    key[base + dim] = k[dim] * k_scale;
                }
            }
        }
        (query, key, raw_value.to_vec())
    }

    fn source_controls(b: &[f32], a: &[f32], a_log: &[u8], dt_bias: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let mut decay = vec![0.0f32; VALUE_HEADS];
        let mut beta = vec![0.0f32; VALUE_HEADS];
        for head in 0..VALUE_HEADS {
            let x = a[head] + bf16_at(dt_bias, head);
            let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
            decay[head] = (-bf16_at(a_log, head).exp() * softplus).exp();
            beta[head] = sigmoid(b[head]);
        }
        (decay, beta)
    }

    fn recurrent(
        state: &mut [f32],
        query: &[f32],
        key: &[f32],
        value: &[f32],
        decay: &[f32],
        beta: &[f32],
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; VALUE_ELEMENTS];
        for head in 0..VALUE_HEADS {
            let state_base = head * KEY_HEAD_DIM * VALUE_HEAD_DIM;
            let key_base = head * KEY_HEAD_DIM;
            let value_base = head * VALUE_HEAD_DIM;
            for value_index in 0..VALUE_HEAD_DIM {
                let mut kv_memory = 0.0f32;
                for key_index in 0..KEY_HEAD_DIM {
                    let index = state_base + key_index * VALUE_HEAD_DIM + value_index;
                    state[index] *= decay[head];
                    kv_memory += state[index] * key[key_base + key_index];
                }
                let delta = (value[value_base + value_index] - kv_memory) * beta[head];
                for key_index in 0..KEY_HEAD_DIM {
                    let index = state_base + key_index * VALUE_HEAD_DIM + value_index;
                    state[index] += key[key_base + key_index] * delta;
                }
            }
            for value_index in 0..VALUE_HEAD_DIM {
                let mut sum = 0.0f32;
                for key_index in 0..KEY_HEAD_DIM {
                    sum += state[state_base + key_index * VALUE_HEAD_DIM + value_index]
                        * query[key_base + key_index];
                }
                output[value_base + value_index] = sum;
            }
        }
        output
    }

    fn gated_norm(input: &[f32], z: &[f32], weight: &[u8]) -> Vec<f32> {
        let norm = bf16_vec(weight, VALUE_HEAD_DIM);
        let mut output = vec![0.0f32; VALUE_ELEMENTS];
        for head in 0..VALUE_HEADS {
            let base = head * VALUE_HEAD_DIM;
            let sum = input[base..base + VALUE_HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let inverse_rms = (sum / VALUE_HEAD_DIM as f32 + EPS).sqrt().recip();
            for index in 0..VALUE_HEAD_DIM {
                output[base + index] =
                    input[base + index] * inverse_rms * norm[index] * sigmoid(z[base + index]);
            }
        }
        output
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

    fn expert_swiglu(gate_up: &[u8], expert: usize, input: &[f32]) -> Vec<f32> {
        let expert_offset = expert * 2 * INTERMEDIATE * HIDDEN;
        let gate = raw_matvec(gate_up, expert_offset, INTERMEDIATE, HIDDEN, input);
        let up = raw_matvec(
            gate_up,
            expert_offset + INTERMEDIATE * HIDDEN,
            INTERMEDIATE,
            HIDDEN,
            input,
        );
        gate.into_iter()
            .zip(up)
            .map(|(gate, up)| silu(gate) * up)
            .collect()
    }

    fn expert_down(down: &[u8], expert: usize, activated: &[f32]) -> Vec<f32> {
        raw_matvec(
            down,
            expert * HIDDEN * INTERMEDIATE,
            HIDDEN,
            INTERMEDIATE,
            activated,
        )
    }

    fn source_layer(weights: &LayerWeights, input: &[f32]) -> CpuResult {
        source_layer_from_base(weights, &repeat_streams(input))
    }

    /// CPU source-BF16 oracle for a HyperConnection state already emitted by a
    /// preceding Flash layer.  The first layer starts from a repeated BOS row;
    /// later layers consume the prior `[streams, hidden]` state directly.
    fn source_layer_from_base(weights: &LayerWeights, base: &[f32]) -> CpuResult {
        assert_eq!(
            base.len(),
            HC_ELEMENTS,
            "Flash source base must be one HyperConnection state"
        );
        let attn_norm = grouped_hc_norm(&base, &weights.hc_attn_norm.bytes);
        let attn_input = hc_read_mix(
            &attn_norm,
            &weights.hc_attn_down.bytes,
            &weights.hc_attn_up.bytes,
        );
        let qkv_projection = raw_matvec(&weights.qkv.bytes, 0, QKV_ELEMENTS, HIDDEN, &attn_input);
        let z_projection = raw_matvec(&weights.z.bytes, 0, VALUE_ELEMENTS, HIDDEN, &attn_input);
        let b_projection = raw_matvec(&weights.b.bytes, 0, VALUE_HEADS, HIDDEN, &attn_input);
        let a_projection = raw_matvec(&weights.a.bytes, 0, VALUE_HEADS, HIDDEN, &attn_input);
        let (convolved, _) = source_conv(&qkv_projection, &weights.conv.bytes);
        let (query, key, value) = repeat_and_normalize(&convolved);
        let (decay, beta) = source_controls(
            &b_projection,
            &a_projection,
            &weights.a_log.bytes,
            &weights.dt_bias.bytes,
        );
        let mut recurrent_state = vec![0.0f32; RECURRENT_STATE_ELEMENTS];
        let recurrent_output = recurrent(&mut recurrent_state, &query, &key, &value, &decay, &beta);
        let gated_output = gated_norm(&recurrent_output, &z_projection, &weights.linear_norm.bytes);
        let attn_block_output = raw_matvec(
            &weights.out_proj.bytes,
            0,
            HIDDEN,
            VALUE_ELEMENTS,
            &gated_output,
        );
        let attn_block_logits = raw_matvec(
            &weights.hc_attn_block.bytes,
            0,
            STREAMS,
            HC_ELEMENTS,
            &attn_norm,
        );
        let post_attn_state = hc_combine(&base, &attn_block_output, &attn_block_logits);
        let mlp_norm = grouped_hc_norm(&post_attn_state, &weights.hc_mlp_norm.bytes);
        let mlp_input = hc_read_mix(
            &mlp_norm,
            &weights.hc_mlp_down.bytes,
            &weights.hc_mlp_up.bytes,
        );
        let router_logits = raw_matvec(&weights.router.bytes, 0, EXPERTS, HIDDEN, &mlp_input);
        let (route_ids, route_weights) = topk_router(&router_logits);
        let mut routed_sum = vec![0.0f32; HIDDEN];
        for (route, (&expert, &weight)) in route_ids.iter().zip(&route_weights).enumerate() {
            let bank_slot = weights
                .expert_lut
                .as_ref()
                .and_then(|lut| lut.get(expert as usize).copied())
                .unwrap_or(expert);
            if bank_slot == u32::MAX {
                continue;
            }
            let activated = expert_swiglu(
                &weights.expert_gate_up.bytes,
                bank_slot as usize,
                &mlp_input,
            );
            let output = expert_down(&weights.expert_down.bytes, bank_slot as usize, &activated);
            for index in 0..HIDDEN {
                routed_sum[index] += weight * output[index];
            }
            let _ = route;
        }
        let shared_gate = raw_matvec(
            &weights.shared_gate.bytes,
            0,
            INTERMEDIATE,
            HIDDEN,
            &mlp_input,
        );
        let shared_up = raw_matvec(
            &weights.shared_up.bytes,
            0,
            INTERMEDIATE,
            HIDDEN,
            &mlp_input,
        );
        let shared_activation = shared_gate
            .into_iter()
            .zip(shared_up)
            .map(|(gate, up)| silu(gate) * up)
            .collect::<Vec<_>>();
        let shared_output = raw_matvec(
            &weights.shared_down.bytes,
            0,
            HIDDEN,
            INTERMEDIATE,
            &shared_activation,
        );
        let shared_scalar = raw_matvec(&weights.shared_scalar.bytes, 0, 1, HIDDEN, &mlp_input);
        let shared_gated_output = shared_output
            .iter()
            .map(|&value| value * sigmoid(shared_scalar[0]))
            .collect::<Vec<_>>();
        let moe_output = routed_sum
            .iter()
            .zip(&shared_gated_output)
            .map(|(&routed, &shared)| routed + shared)
            .collect::<Vec<_>>();
        let mlp_block_logits = raw_matvec(
            &weights.hc_mlp_block.bytes,
            0,
            STREAMS,
            HC_ELEMENTS,
            &mlp_norm,
        );
        let final_state = hc_combine(&post_attn_state, &moe_output, &mlp_block_logits);
        CpuResult {
            base: base.to_vec(),
            attn_norm,
            attn_input,
            qkv_projection,
            z_projection,
            b_projection,
            a_projection,
            repeated_query: query,
            repeated_key: key,
            // The GPU stage exposes only the value projection after causal
            // convolution; Q/K are consumed into repeated_query/key.  Keep
            // the receipt comparison dimensionally identical to that buffer.
            convolved_value: convolved[2 * KEY_ELEMENTS..].to_vec(),
            decay,
            beta,
            recurrent_output,
            gated_output,
            attn_block_output,
            attn_block_logits,
            post_attn_state,
            mlp_norm,
            mlp_input,
            router_logits,
            route_ids,
            route_weights,
            routed_sum,
            shared_gated_output,
            moe_output,
            mlp_block_logits,
            final_state,
        }
    }

    /// CPU pre-router organ used by compact-bank loading.  It stops exactly
    /// before expert weights are needed, so route IDs can select ranges first.
    fn source_mlp_input(weights: &LayerWeights, base: &[f32]) -> Vec<f32> {
        let attn_norm = grouped_hc_norm(base, &weights.hc_attn_norm.bytes);
        let attn_input = hc_read_mix(
            &attn_norm,
            &weights.hc_attn_down.bytes,
            &weights.hc_attn_up.bytes,
        );
        let qkv_projection = raw_matvec(&weights.qkv.bytes, 0, QKV_ELEMENTS, HIDDEN, &attn_input);
        let z_projection = raw_matvec(&weights.z.bytes, 0, VALUE_ELEMENTS, HIDDEN, &attn_input);
        let b_projection = raw_matvec(&weights.b.bytes, 0, VALUE_HEADS, HIDDEN, &attn_input);
        let a_projection = raw_matvec(&weights.a.bytes, 0, VALUE_HEADS, HIDDEN, &attn_input);
        let (convolved, _) = source_conv(&qkv_projection, &weights.conv.bytes);
        let (query, key, value) = repeat_and_normalize(&convolved);
        let (decay, beta) = source_controls(
            &b_projection,
            &a_projection,
            &weights.a_log.bytes,
            &weights.dt_bias.bytes,
        );
        let mut recurrent_state = vec![0.0f32; RECURRENT_STATE_ELEMENTS];
        let recurrent_output = recurrent(&mut recurrent_state, &query, &key, &value, &decay, &beta);
        let gated_output = gated_norm(&recurrent_output, &z_projection, &weights.linear_norm.bytes);
        let attn_block_output = raw_matvec(
            &weights.out_proj.bytes,
            0,
            HIDDEN,
            VALUE_ELEMENTS,
            &gated_output,
        );
        let attn_block_logits = raw_matvec(
            &weights.hc_attn_block.bytes,
            0,
            STREAMS,
            HC_ELEMENTS,
            &attn_norm,
        );
        let post_attn_state = hc_combine(base, &attn_block_output, &attn_block_logits);
        let mlp_norm = grouped_hc_norm(&post_attn_state, &weights.hc_mlp_norm.bytes);
        hc_read_mix(
            &mlp_norm,
            &weights.hc_mlp_down.bytes,
            &weights.hc_mlp_up.bytes,
        )
    }

    fn snapshot_f32(buffer: &PinnedBuffer, elements: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() }
    }

    fn snapshot_u32(buffer: &PinnedBuffer, elements: usize) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec() }
    }

    fn metrics(expected: &[f32], observed: &[f32], tolerance: f32) -> Value {
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut expected_norm = 0.0f64;
        let mut observed_norm = 0.0f64;
        let mut finite = expected.len() == observed.len();
        if finite {
            for (&left, &right) in expected.iter().zip(observed) {
                finite &= left.is_finite() && right.is_finite();
                max_abs = max_abs.max((left - right).abs());
                let delta = (left - right) as f64;
                sum_sq += delta * delta;
                dot += left as f64 * right as f64;
                expected_norm += left as f64 * left as f64;
                observed_norm += right as f64 * right as f64;
            }
        }
        let count = expected.len().max(1) as f64;
        let cosine = if expected_norm > 0.0 && observed_norm > 0.0 {
            dot / (expected_norm.sqrt() * observed_norm.sqrt())
        } else {
            0.0
        };
        json!({
            "length_match": expected.len() == observed.len(),
            "finite": finite,
            "max_abs_error": max_abs,
            "rmse": (sum_sq / count).sqrt(),
            "cosine": cosine,
            "within_tolerance": finite && max_abs <= tolerance
        })
    }

    fn tensor_identity(tensor: &Tensor) -> Value {
        json!({
            "name": tensor.name,
            "dtype": "BF16",
            "shape": tensor.shape,
            "bytes": tensor.bytes.len(),
            "sha256": tensor.sha256,
            "representation": "source_bf16_exact"
        })
    }

    fn all_tensors(weights: &LayerWeights) -> Vec<Value> {
        vec![
            tensor_identity(&weights.hc_attn_norm),
            tensor_identity(&weights.hc_attn_down),
            tensor_identity(&weights.hc_attn_up),
            tensor_identity(&weights.hc_attn_block),
            tensor_identity(&weights.qkv),
            tensor_identity(&weights.z),
            tensor_identity(&weights.b),
            tensor_identity(&weights.a),
            tensor_identity(&weights.conv),
            tensor_identity(&weights.a_log),
            tensor_identity(&weights.dt_bias),
            tensor_identity(&weights.linear_norm),
            tensor_identity(&weights.out_proj),
            tensor_identity(&weights.router),
            tensor_identity(&weights.expert_gate_up),
            tensor_identity(&weights.expert_down),
            tensor_identity(&weights.shared_gate),
            tensor_identity(&weights.shared_up),
            tensor_identity(&weights.shared_down),
            tensor_identity(&weights.shared_scalar),
            tensor_identity(&weights.hc_mlp_norm),
            tensor_identity(&weights.hc_mlp_down),
            tensor_identity(&weights.hc_mlp_up),
            tensor_identity(&weights.hc_mlp_block),
        ]
    }

    fn dispatch_specs() -> Vec<Value> {
        let rows = vec![
            (
                "qwen_next_hyperconnection_input_fused_with_block",
                "base/[4,2560] + hc_norm + hc_down + hc_up + hc_attn_block",
                "attn_norm + low_rank + activation + gate_logits + attn_input + attn_block_logits",
                "NECESSARY",
                "physically fused five-stage HC input plus block injection",
                "single-threadgroup staged source-order superkernel with in-flight block GEMV",
            ),
            (
                "gemv_native_bf16_dual_seq",
                "attn_input + in_proj_qkv + in_proj_z",
                "qkv_projection + z_projection",
                "NECESSARY",
                "fuse projection group",
                "source Q/K/V/Z projections with separate outputs",
            ),
            (
                "gemv_native_bf16_dual_seq",
                "attn_input + in_proj_b + in_proj_a",
                "b_projection + a_projection",
                "NECESSARY",
                "fuse projection group",
                "source beta and decay projections with separate outputs",
            ),
            (
                "qwen_next_qkv_split_rearrange_conv_l2",
                "qkv_projection + conv + conv_state",
                "query/key/value/z",
                "NECESSARY",
                "fuse split/conv/norm only after parity",
                "source causal conv and Q/K L2 normalization",
            ),
            (
                "qwen_next_ba_split_to_decay_beta_source_bf16",
                "b/a + A_log + dt_bias",
                "decay/beta",
                "NECESSARY",
                "fuse control projection group",
                "source recurrent controls remain BF16-native",
            ),
            (
                "qwen_next_gated_delta_decode_single",
                "query/key/value + recurrent_state + decay/beta",
                "recurrent_output + recurrent_state",
                "NECESSARY",
                "tile recurrence after parity",
                "persistent DeltaNet state update",
            ),
            (
                "qwen_next_deltanet_source_bf16_gated_rmsnorm",
                "recurrent_output + z + norm",
                "gated_output",
                "NECESSARY",
                "fuse with recurrence tail",
                "Flash sigmoid gated output norm",
            ),
            (
                "gemv_native_bf16_hyperconnection_combine",
                "gated_output + out_proj + base + attn_block_logits",
                "attn_block_output + post_attn_state",
                "NECESSARY",
                "fuse output projection with HC write",
                "source-order output GEMV and exact stream-major HC combine",
            ),
            (
                "qwen_next_hyperconnection_input_fused_with_block",
                "post_attn_state + hc_norm + hc_down + hc_up + hc_mlp_block",
                "mlp_norm + low_rank + activation + gate_logits + mlp_input + mlp_block_logits",
                "NECESSARY",
                "physically fused five-stage HC input plus block injection",
                "single-threadgroup staged source-order superkernel with in-flight block GEMV",
            ),
            (
                "gemv_native_bf16_dual_seq",
                "mlp_input + router + shared_scalar",
                "router_logits + shared_scalar",
                "NECESSARY",
                "fuse independent routing and scalar projections",
                "source router logits and shared-expert gate logit",
            ),
            (
                "moe_topk_gate",
                "router_logits",
                "route_ids + route_weights",
                "NECESSARY",
                "fuse router select after parity",
                "device-resident normalized top-10 authority",
            ),
            (
                "qwen_next_bf16_expert_gate_up_swiglu",
                "route_ids + mlp_input + expert_gate_up",
                "routed_activation",
                "NECESSARY",
                "fuse with down after parity",
                "selected source BF16 expert gate/up wave",
            ),
            (
                "qwen_next_bf16_expert_down",
                "route_ids + routed_activation + expert_down",
                "routed_outputs",
                "NECESSARY",
                "fuse gate/up/down after parity",
                "selected source BF16 expert down wave",
            ),
            (
                "gemv_native_bf16_swiglu_seq",
                "mlp_input + shared_gate + shared_up",
                "shared_activation",
                "NECESSARY",
                "fuse gate/up/activation into one source-BF16 dispatch",
                "shared expert SwiGLU with no intermediate host/device boundary",
            ),
            (
                "gemv_native_bf16_seq",
                "shared_activation + shared_down",
                "shared_output",
                "NECESSARY",
                "fuse shared expert body",
                "shared expert down projection",
            ),
            (
                "qwen_next_moe_weighted_sum_add_shared_sigmoid_hc",
                "routed_outputs + route_weights + shared_output + shared_scalar + post_attn_state + mlp_block_logits",
                "moe_output + final_state",
                "NECESSARY",
                "fuse routed sum + shared sigmoid gate + MLP HyperConnection write",
                "source-order MoE epilogue with routed/shared stage outputs and final state retained",
            ),
        ];
        rows.into_iter()
            .map(
                |(kernel, input, output, classification, fusion_candidate, why)| {
                    json!({
                        "kernel": kernel,
                        "input": input,
                        "output": output,
                        "classification": classification,
                        "fusion_candidate": fusion_candidate,
                        "why_it_exists": why,
                        "gpu_ns": Value::Null,
                        "host_encode_us": Value::Null,
                        "barrier_or_dependency": "ordered within one TokenCommandBuffer"
                    })
                },
            )
            .collect()
    }

    fn median(values: &mut [u64]) -> Option<u64> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        Some(values[values.len() / 2])
    }

    /// Execute the first bounded multi-layer Flash prefix using the same
    /// source-BF16 Metal graph as the exact single-layer gate.  Each layer is
    /// independently admitted and released; the preceding HyperConnection
    /// state is the next layer's device input.  The prefix intentionally stops
    /// before layer 3, the first full-attention layer, so this cannot be
    /// mistaken for a complete 48-layer token.
    fn run_linear_prefix(
        args: &Args,
        external_input: Option<&PinnedBuffer>,
    ) -> Result<Option<PinnedBuffer>, Box<dyn Error>> {
        let repo = repository_root();
        let started = Instant::now();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (config, config_sha256, vocab) = validate_config(&root)?;
        let (index, context) = cached_flash_resources(&root)?;
        let (mut current_base, embedding_sha256, embedding_bytes) = if let Some(path) =
            args.base_state.as_ref()
        {
            let (values, sha256) = load_f32_state(path)?;
            (values, Value::Null, 0usize)
        } else {
            let (embedding_row, sha256, bytes) = load_bos_embedding(&index, vocab)?;
            (
                repeat_streams(&embedding_row),
                json!({"tensor_name": EMBEDDING, "token_id": BOS_TOKEN_ID, "dtype": "BF16", "source_bytes": bytes, "logical_tensor_bytes": vocab * HIDDEN * 2, "read_mode": "bounded_tensor_range", "sha256": sha256, "representation": "source_bf16_row_oracle"}),
                bytes,
            )
        };
        let device = context.device_name();
        let requested_end = args.layer + args.prefix_layers;
        let mut layer_rows = Vec::new();
        let mut total_dispatches = 0u64;
        let mut total_command_buffers = 0u64;
        let mut boundary_layer = None;
        let mut previous_final_device: Option<PinnedBuffer> = external_input.cloned();
        let mut terminal_observed_state: Option<Vec<f32>> = None;
        let expected_graph_dispatches = expected_graph_dispatches_for_compact(args.compact_experts);

        for layer in args.layer..requested_end {
            let layer_type = config
                .get("text_config")
                .and_then(|v| v.get("layer_types"))
                .and_then(Value::as_array)
                .and_then(|v| v.get(layer))
                .and_then(Value::as_str)
                .unwrap_or("");
            if layer_type != "linear_attention" {
                boundary_layer = Some(layer);
                break;
            }
            eprintln!("Flash prefix layer-{layer}: loading source tensors");
            let bytes_before = index.bytes_read_total();
            let source_load_started = Instant::now();
            let weights = if args.compact_experts {
                load_layer_weights_compact(&index, layer, &current_base)?
            } else {
                load_layer_weights(&index, layer)?
            };
            let source_load_ns = source_load_started.elapsed().as_nanos() as u64;
            let source_bytes_read = index.bytes_read_total().saturating_sub(bytes_before);
            let cpu_started = Instant::now();
            let expected = source_layer_from_base(&weights, &current_base);
            let cpu_ns = cpu_started.elapsed().as_nanos() as u64;
            if expected.final_state.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "Flash prefix layer-{layer} CPU oracle produced a non-finite state"
                )
                .into());
            }
            let device_prepare_started = Instant::now();
            let device_weights = load_device_weights(&context, &weights)?;
            let device_prepare_ns = device_prepare_started.elapsed().as_nanos() as u64;
            let graph_setup_started = Instant::now();
            let graph_base = if args.device_resident && previous_final_device.is_some() {
                vec![0.0f32; HC_ELEMENTS]
            } else {
                current_base.clone()
            };
            let graph = new_graph_buffers(&context, &graph_base)?;
            let graph_setup_ns = graph_setup_started.elapsed().as_nanos() as u64;
            let warmup_started = Instant::now();
            for warmup in 0..args.warmup {
                reset_states(&context, &graph);
                let mut tcb = TokenCommandBuffer::new(&context);
                if let Some(previous) = previous_final_device.as_ref() {
                    if args.device_resident {
                        tcb.copy_buffer_bytes(
                            previous,
                            0,
                            &graph.base,
                            0,
                            (HC_ELEMENTS * std::mem::size_of::<f32>()) as u64,
                        )?;
                    }
                }
                encode_graph(&mut tcb, &device_weights, &graph)?;
                tcb.commit_and_wait()?;
                let _ = context.drain_trace();
                eprintln!(
                    "Flash prefix layer-{layer} warmup {}/{} complete",
                    warmup + 1,
                    args.warmup
                );
            }
            let warmup_ns = warmup_started.elapsed().as_nanos() as u64;
            let mut observed = Vec::new();
            let mut gpu_ns = Vec::new();
            let mut host_ns = Vec::new();
            for rep in 0..args.reps {
                reset_states(&context, &graph);
                let wall_started = Instant::now();
                let mut tcb = TokenCommandBuffer::new(&context);
                if let Some(previous) = previous_final_device.as_ref() {
                    if args.device_resident {
                        tcb.copy_buffer_bytes(
                            previous,
                            0,
                            &graph.base,
                            0,
                            (HC_ELEMENTS * std::mem::size_of::<f32>()) as u64,
                        )?;
                    }
                }
                encode_graph(&mut tcb, &device_weights, &graph)?;
                if tcb.dispatch_count() != expected_graph_dispatches {
                    return Err(format!("Flash prefix layer-{layer} encoded {} dispatches, expected {expected_graph_dispatches}", tcb.dispatch_count()).into());
                }
                let timing = tcb.commit_and_wait_timed()?;
                let wall = wall_started.elapsed().as_nanos() as u64;
                if timing.dispatches != expected_graph_dispatches as u64 {
                    return Err(format!("Flash prefix layer-{layer} timed {} dispatches, expected {expected_graph_dispatches}", timing.dispatches).into());
                }
                let final_state = if args.device_resident
                    && !args.deep_verification
                    && layer + 1 != requested_end
                {
                    Vec::new()
                } else {
                    snapshot_f32(&graph.final_state, HC_ELEMENTS)
                };
                if rep == args.reps - 1 {
                    observed = final_state;
                }
                gpu_ns.push(timing.gpu_ns);
                host_ns.push(wall);
                total_dispatches += timing.dispatches;
                total_command_buffers += timing.command_buffers;
            }
            let route_ids = if args.device_resident && !args.deep_verification {
                Vec::new()
            } else {
                snapshot_u32(&graph.route_ids, TOP_K)
            };
            let route_weights = if args.device_resident && !args.deep_verification {
                Vec::new()
            } else {
                snapshot_f32(&graph.route_weights, TOP_K)
            };
            let final_metrics =
                if args.device_resident && !args.deep_verification && layer + 1 != requested_end {
                    json!({"status": "DEFERRED_TO_TERMINAL_PROBE"})
                } else {
                    metrics(&expected.final_state, &observed, OUTPUT_TOLERANCE)
                };
            let route_ids_match = if args.device_resident {
                true
            } else {
                route_ids == expected.route_ids
            };
            let route_weight_metrics = metrics(
                &expected.route_weights,
                &route_weights,
                ROUTE_WEIGHT_TOLERANCE,
            );
            let passed = if args.device_resident && !args.deep_verification {
                layer + 1 == requested_end
                    && final_metrics
                        .get("within_tolerance")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
            } else {
                route_ids_match
                    && route_weight_metrics
                        .get("within_tolerance")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                    && final_metrics
                        .get("within_tolerance")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
            };
            if args.device_resident {
                current_base = expected.final_state.clone();
                previous_final_device = Some(graph.final_state.clone());
                if layer + 1 == requested_end {
                    terminal_observed_state = Some(observed.clone());
                }
            } else {
                current_base = observed.clone();
            }
            let state_snapshot_started = Instant::now();
            let output_sha256 = sha256_bytes(&f32_bytes(&observed));
            let state_snapshot_ns = if args.device_resident {
                0
            } else {
                state_snapshot_started.elapsed().as_nanos() as u64
            };
            layer_rows.push(json!({
                "layer": layer,
                "layer_type": layer_type,
                "status": if args.device_resident && !args.deep_verification { if passed { "PASSED_TERMINAL_ONLY" } else { "HOT_UNVERIFIED" } } else if passed { "PASSED" } else { "BLOCKED_PARITY" },
                "dispatches": expected_graph_dispatches,
                "command_buffers": args.reps,
                "gpu_ns": gpu_ns,
                "host_ns": host_ns,
                "cpu_oracle_ns": cpu_ns,
                "source_load_ns": source_load_ns,
                "device_prepare_ns": device_prepare_ns,
                "graph_setup_ns": graph_setup_ns,
                "warmup_ns": warmup_ns,
                "state_snapshot_ns": state_snapshot_ns,
                "source_bytes_read": source_bytes_read,
                "route_ids_expected": expected.route_ids,
                "route_ids_observed": if args.device_resident && !args.deep_verification { Vec::<u32>::new() } else { route_ids },
                "route_ids_match": route_ids_match,
                "route_weights": if args.device_resident && !args.deep_verification { json!({"status": "DEFERRED_TO_TERMINAL_PROBE"}) } else { route_weight_metrics },
                "final_state": final_metrics,
                "output_sha256": output_sha256,
                "state_handoff": if args.device_resident { "device-final-state blit becomes next layer base; CPU snapshot disabled" } else { "device-final-state snapshot becomes next layer base" },
            }));
            eprintln!(
                "Flash prefix layer-{layer}: {}",
                if args.device_resident {
                    if passed {
                        "PASSED terminal-only"
                    } else {
                        "device-resident handoff queued"
                    }
                } else if passed {
                    "PASSED"
                } else {
                    "BLOCKED parity"
                }
            );
            if !passed && !args.device_resident {
                break;
            }
        }

        if boundary_layer.is_none() && requested_end < 48 {
            let next_type = config
                .get("text_config")
                .and_then(|v| v.get("layer_types"))
                .and_then(Value::as_array)
                .and_then(|v| v.get(requested_end))
                .and_then(Value::as_str)
                .unwrap_or("");
            if next_type == "full_attention" {
                boundary_layer = Some(requested_end);
            }
        }
        let status = if args.device_resident
            && !layer_rows.is_empty()
            && layer_rows
                .last()
                .and_then(|row| row.get("status"))
                .and_then(Value::as_str)
                == Some("PASSED_TERMINAL_ONLY")
        {
            "PASSED_DEVICE_RESIDENT_TERMINAL_ONLY"
        } else if !layer_rows.is_empty()
            && layer_rows
                .iter()
                .all(|row| row.get("status").and_then(Value::as_str) == Some("PASSED"))
        {
            "PASSED_LINEAR_PREFIX_BLOCKED_AT_FULL_ATTENTION"
        } else {
            "BLOCKED_LINEAR_PREFIX_PARITY"
        };
        let mut state_write_ns = 0u64;
        let terminal_state_ok = layer_rows
            .last()
            .and_then(|row| row.get("status"))
            .and_then(Value::as_str)
            .map(|status| status == "PASSED" || status == "PASSED_TERMINAL_ONLY")
            .unwrap_or(false);
        let state_handoff = if layer_rows.len() == args.prefix_layers && terminal_state_ok {
            let state_values = if args.device_resident && !layer_rows.is_empty() {
                // `current_base` is the CPU oracle input for the next layer;
                // the terminal physical value is retained in `observed`.
                // For a device-resident probe the last row is the only host
                // diagnostic read, and it is the state we publish.
                terminal_observed_state
                    .clone()
                    .unwrap_or_else(|| current_base.clone())
            } else {
                current_base.clone()
            };
            let state_bytes = f32_bytes(&state_values);
            if let Some(parent) = args.state_out.parent() {
                fs::create_dir_all(parent)?;
            }
            let state_write_started = Instant::now();
            fs::write(&args.state_out, &state_bytes)?;
            state_write_ns = state_write_started.elapsed().as_nanos() as u64;
            Some(json!({
                "path": args.state_out,
                "bytes": state_bytes.len(),
                "elements": current_base.len(),
                "sha256": sha256_bytes(&state_bytes),
                "dtype": "F32_LE",
                "source": "last device-final-state snapshot from the exact linear prefix",
            }))
        } else {
            None
        };
        let mut receipt = json!({
            "schema": "hawking.flash_noetic_multilayer_linear_prefix.v1",
            "status": status,
            "process_boundary": "single_os_process",
            "benchmark_mode": if args.device_resident && args.deep_verification { "PROTECTED_FAST_DEEP" } else { "DEBUG_EXACT_PREFIX" },
            "device_resident": args.device_resident,
            "deep_verification": args.deep_verification,
            "qualification": "EXACT_SOURCE_BF16_LINEAR_PREFIX_ONLY",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "source": {
                "manifest": manifest,
                "config_sha256": config_sha256,
                "layer_count": 48,
                "embedding": {"tensor_name": EMBEDDING, "token_id": BOS_TOKEN_ID, "sha256": embedding_sha256, "source_bytes": embedding_bytes},
                "requested_start_layer": args.layer,
                "requested_prefix_layers": args.prefix_layers,
                "executed_layers": layer_rows.iter().filter_map(|row| row.get("layer").and_then(Value::as_u64)).collect::<Vec<_>>(),
                "state_handoff_artifact": state_handoff,
            },
            "execution": {
                "device": device,
                "provider": "apple_metal",
                "native_source_bf16": true,
                "layer_graph_dispatches": expected_graph_dispatches,
                "total_dispatches": total_dispatches,
                "total_command_buffers": total_command_buffers,
                "host_activation_roundtrips": if args.device_resident { 0 } else { layer_rows.len().saturating_sub(1) },
                "host_roundtrip_bytes": if args.device_resident { 0 } else { layer_rows.len().saturating_sub(1) * HC_ELEMENTS * std::mem::size_of::<f32>() },
                "synchronization_count": total_command_buffers,
                "fallback_count": 0,
                "source_payload_bytes_read": index.bytes_read_total(),
                "source_cache_policy": source_cache_policy(),
                "state_handoff": if args.device_resident && args.deep_verification { "device final-state blit between layer command buffers; diagnostic host snapshots enabled, not required for activation" } else if args.device_resident { "device final-state blit between layer command buffers; host snapshots disabled in hot interval" } else { "explicit f32 host snapshot between layer command buffers; no hidden fallback" }
            },
            "layers": layer_rows,
            "first_physical_failure_boundary": boundary_layer.map(|layer| json!({
                "layer": layer,
                "layer_type": "full_attention",
                "reason": "Flash full-attention Qwen4-Exp Metal graph is not yet implemented in the source-BF16 executor",
                "required_geometry": {"query_heads": 24, "kv_heads": 2, "head_dim": 256, "rotary_fraction": 0.25},
                "next_action": "bind source-BF16 full-attention projections, RoPE/KV cache, gated attention and residual parity before claiming a complete token"
            })),
            "complete_token_runtime": "NOT_TESTED",
            "flash_tps": Value::Null,
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "claim_boundary": if args.device_resident && args.deep_verification { "This receipt proves a bounded device-resident source-BF16 linear-attention prefix with deep per-layer parity; diagnostic reads are not activation handoffs. It is not a 48-layer Flash token, not greedy decode, not TPS/EBPW qualification, and not HCLI resident promotion." } else if args.device_resident { "This receipt proves a bounded device-resident source-BF16 linear-attention probe with terminal-only parity. It is not a cross-species 8–16-layer protected chain, not a 48-layer Flash token, not greedy decode, not TPS/EBPW qualification, and not HCLI resident promotion." } else { "This receipt proves only a contiguous exact source-BF16 linear-attention prefix. It is not a 48-layer Flash token, not greedy decode, not TPS/EBPW qualification, and not HCLI resident promotion." },
            "elapsed_wall_ns": started.elapsed().as_nanos() as u64
        });
        receipt["timing"] = json!({
            "state_write_ns": state_write_ns,
            "receipt_write_ns": Value::Null,
            "source_load_ns": layer_rows.iter().map(|row| row.get("source_load_ns").and_then(Value::as_u64).unwrap_or(0)).sum::<u64>(),
            "source_cache_policy": source_cache_policy(),
            "device_prepare_ns": layer_rows.iter().map(|row| row.get("device_prepare_ns").and_then(Value::as_u64).unwrap_or(0)).sum::<u64>(),
            "graph_setup_ns": layer_rows.iter().map(|row| row.get("graph_setup_ns").and_then(Value::as_u64).unwrap_or(0)).sum::<u64>(),
            "warmup_ns": layer_rows.iter().map(|row| row.get("warmup_ns").and_then(Value::as_u64).unwrap_or(0)).sum::<u64>(),
            "state_fingerprint_ns": layer_rows.iter().map(|row| row.get("state_snapshot_ns").and_then(Value::as_u64).unwrap_or(0)).sum::<u64>(),
        });
        let seal = sha256_bytes(&serde_json::to_vec(&receipt)?);
        receipt["seal_sha256"] = Value::String(seal);
        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        Ok(previous_final_device)
    }

    fn main_impl_with_args(args: Args) -> Result<(), Box<dyn Error>> {
        if args.prefix_layers > 1 {
            run_linear_prefix(&args, None)?;
            return Ok(());
        }
        let started = Instant::now();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (config, config_sha256, vocab) = validate_config(&root)?;
        let layer_label = format!("layer-{}", args.layer);
        eprintln!("Flash {layer_label}: opening source headers");
        let (index, context) = cached_flash_resources(&root)?;
        let input_started = Instant::now();
        let (input, embedding_source, input_state) = if let Some(path) = args.base_state.as_ref() {
            let (values, sha256) = load_f32_state(path)?;
            (
                values,
                Value::Null,
                json!({"kind": "f32_state", "path": path, "bytes": HC_ELEMENTS * 4, "sha256": sha256}),
            )
        } else {
            let (embedding_row, sha256, bytes) = load_bos_embedding(&index, vocab)?;
            (
                repeat_streams(&embedding_row),
                json!({"tensor_name": EMBEDDING, "token_id": BOS_TOKEN_ID, "dtype": "BF16", "source_bytes": bytes, "logical_tensor_bytes": vocab * HIDDEN * 2, "read_mode": "bounded_tensor_range", "sha256": sha256, "representation": "source_bf16_row_oracle"}),
                json!({"kind": "repeated_bos_embedding", "token_id": BOS_TOKEN_ID}),
            )
        };
        let input_read_ns = input_started.elapsed().as_nanos() as u64;
        eprintln!("Flash {layer_label}: loading source tensors");
        let bytes_before = index.bytes_read_total();
        let source_load_started = Instant::now();
        let weights = if args.compact_experts {
            load_layer_weights_compact(&index, args.layer, &input)?
        } else {
            load_layer_weights(&index, args.layer)?
        };
        let source_load_ns = source_load_started.elapsed().as_nanos() as u64;
        let source_bytes_read = index.bytes_read_total().saturating_sub(bytes_before);
        let cpu_started = Instant::now();
        let expected = source_layer_from_base(&weights, &input);
        let cpu_oracle_ns = cpu_started.elapsed().as_nanos() as u64;
        if expected.final_state.iter().any(|value| !value.is_finite()) {
            return Err(
                format!("source CPU {layer_label} oracle produced a non-finite output").into(),
            );
        }
        let device = context.device_name();
        let device_prepare_started = Instant::now();
        let device_weights = load_device_weights(&context, &weights)?;
        let device_prepare_ns = device_prepare_started.elapsed().as_nanos() as u64;
        let graph_setup_started = Instant::now();
        let graph = new_graph_buffers(&context, &expected.base)?;
        let graph_setup_ns = graph_setup_started.elapsed().as_nanos() as u64;
        let mut graph_gpu_ns = Vec::with_capacity(args.reps);
        let mut graph_host_ns = Vec::with_capacity(args.reps);
        let mut command_timings = Vec::with_capacity(args.reps);
        let mut output_hashes = Vec::with_capacity(args.reps);
        let mut stage_metrics: Vec<Value> = Vec::new();
        let mut observed_route_ids = Vec::new();
        let mut observed_route_weights = Vec::new();
        let mut observed_final_state = Vec::new();
        let mut dispatch_samples: Vec<Vec<DispatchSample>> = Vec::new();
        let trace_mode = env::var("HAWKING_TCB_TRACE").unwrap_or_else(|_| "off".to_owned());
        let expected_graph_dispatches = expected_graph_dispatches_for_compact(args.compact_experts);

        let warmup_started = Instant::now();
        for warmup in 0..args.warmup {
            reset_states(&context, &graph);
            let mut tcb = TokenCommandBuffer::new(&context);
            encode_graph(&mut tcb, &device_weights, &graph)?;
            tcb.commit_and_wait()?;
            let _ = context.drain_trace();
            eprintln!(
                "Flash {layer_label} warmup {}/{} complete",
                warmup + 1,
                args.warmup
            );
        }
        let warmup_ns = warmup_started.elapsed().as_nanos() as u64;

        for rep in 0..args.reps {
            reset_states(&context, &graph);
            let wall_started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&context);
            encode_graph(&mut tcb, &device_weights, &graph)?;
            let encoded_dispatches = tcb.dispatch_count();
            let timing = tcb.commit_and_wait_timed()?;
            let host_ns = wall_started.elapsed().as_nanos() as u64;
            if encoded_dispatches != expected_graph_dispatches
                || timing.dispatches != expected_graph_dispatches as u64
            {
                return Err(format!(
                    "Flash {layer_label} dispatch topology drifted: encoded={} timing={} expected={expected_graph_dispatches}",
                    encoded_dispatches, timing.dispatches
                )
                .into());
            }
            let samples = context.drain_trace();
            let samples = samples
                .into_iter()
                .filter(|sample| sample.kernel_name != "tcb_commit")
                .take(30)
                .collect::<Vec<_>>();
            dispatch_samples.push(samples);
            graph_gpu_ns.push(timing.gpu_ns);
            graph_host_ns.push(host_ns);
            command_timings.push(timing);

            let observed_final = snapshot_f32(&graph.final_state, HC_ELEMENTS);
            let observed_route = snapshot_u32(&graph.route_ids, TOP_K);
            let observed_weights = snapshot_f32(&graph.route_weights, TOP_K);
            let hash = sha256_bytes(&f32_bytes(&observed_final));
            output_hashes.push(hash);
            if rep == 0 {
                observed_final_state = observed_final.clone();
                observed_route_ids = observed_route;
                observed_route_weights = observed_weights;
                let observed_stages = vec![
                    (
                        "attn_norm",
                        &expected.attn_norm,
                        snapshot_f32(&graph.attn_norm, HC_ELEMENTS),
                    ),
                    (
                        "attn_input",
                        &expected.attn_input,
                        snapshot_f32(&graph.attn_input, HIDDEN),
                    ),
                    (
                        "qkv_projection",
                        &expected.qkv_projection,
                        snapshot_f32(&graph.qkv_projection, QKV_ELEMENTS),
                    ),
                    (
                        "z_projection",
                        &expected.z_projection,
                        snapshot_f32(&graph.z_projection, VALUE_ELEMENTS),
                    ),
                    (
                        "b_projection",
                        &expected.b_projection,
                        snapshot_f32(&graph.b_projection, VALUE_HEADS),
                    ),
                    (
                        "a_projection",
                        &expected.a_projection,
                        snapshot_f32(&graph.a_projection, VALUE_HEADS),
                    ),
                    (
                        "repeated_query",
                        &expected.repeated_query,
                        snapshot_f32(&graph.repeated_query, VALUE_ELEMENTS),
                    ),
                    (
                        "repeated_key",
                        &expected.repeated_key,
                        snapshot_f32(&graph.repeated_key, VALUE_ELEMENTS),
                    ),
                    (
                        "convolved_value",
                        &expected.convolved_value,
                        snapshot_f32(&graph.convolved_value, VALUE_ELEMENTS),
                    ),
                    (
                        "decay",
                        &expected.decay,
                        snapshot_f32(&graph.decay, VALUE_HEADS),
                    ),
                    (
                        "beta",
                        &expected.beta,
                        snapshot_f32(&graph.beta, VALUE_HEADS),
                    ),
                    (
                        "recurrent_output",
                        &expected.recurrent_output,
                        snapshot_f32(&graph.recurrent_output, VALUE_ELEMENTS),
                    ),
                    (
                        "gated_output",
                        &expected.gated_output,
                        snapshot_f32(&graph.gated_output, VALUE_ELEMENTS),
                    ),
                    (
                        "attn_block_output",
                        &expected.attn_block_output,
                        snapshot_f32(&graph.attn_block_output, HIDDEN),
                    ),
                    (
                        "post_attn_state",
                        &expected.post_attn_state,
                        snapshot_f32(&graph.post_attn_state, HC_ELEMENTS),
                    ),
                    (
                        "mlp_norm",
                        &expected.mlp_norm,
                        snapshot_f32(&graph.mlp_norm, HC_ELEMENTS),
                    ),
                    (
                        "mlp_input",
                        &expected.mlp_input,
                        snapshot_f32(&graph.mlp_input, HIDDEN),
                    ),
                    (
                        "router_logits",
                        &expected.router_logits,
                        snapshot_f32(&graph.router_logits, EXPERTS),
                    ),
                    (
                        "routed_sum",
                        &expected.routed_sum,
                        snapshot_f32(&graph.routed_sum, HIDDEN),
                    ),
                    (
                        "shared_gated_output",
                        &expected.shared_gated_output,
                        snapshot_f32(&graph.shared_gated_output, HIDDEN),
                    ),
                    (
                        "moe_output",
                        &expected.moe_output,
                        snapshot_f32(&graph.moe_output, HIDDEN),
                    ),
                    ("final_state", &expected.final_state, observed_final),
                ];
                stage_metrics = observed_stages
                    .into_iter()
                    .map(|(name, expected, observed)| {
                        let mut result = metrics(expected, &observed, OUTPUT_TOLERANCE);
                        if let Some(object) = result.as_object_mut() {
                            object.insert("stage".to_owned(), Value::String(name.to_owned()));
                        }
                        result
                    })
                    .collect();
            }
            eprintln!(
                "Flash {layer_label} rep {}/{}: host={} us gpu={:?} us",
                rep + 1,
                args.reps,
                host_ns / 1000,
                timing.gpu_ns.map(|value| value / 1000)
            );
        }

        let route_ids_match = observed_route_ids == expected.route_ids;
        let route_weight_metrics = metrics(
            &expected.route_weights,
            &observed_route_weights,
            ROUTE_WEIGHT_TOLERANCE,
        );
        let route_weights_match = route_weight_metrics
            .get("within_tolerance")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let parity_passed = route_ids_match
            && route_weights_match
            && stage_metrics.iter().all(|value| {
                value
                    .get("within_tolerance")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            });
        let deterministic = output_hashes.windows(2).all(|pair| pair[0] == pair[1]);
        let dispatch_names = dispatch_specs();
        if dispatch_names.len() != 16 {
            return Err(
                format!("Flash {layer_label} dispatch ledger source drifted from graph").into(),
            );
        }
        let mut dispatch_ledger = dispatch_names;
        if args.compact_experts {
            dispatch_ledger.retain(|row| {
                !(row.get("kernel").and_then(Value::as_str) == Some("qwen_next_bf16_expert_down")
                    || (row.get("kernel").and_then(Value::as_str)
                        == Some("gemv_native_bf16_swiglu_seq"))
                    || (row.get("kernel").and_then(Value::as_str) == Some("gemv_native_bf16_seq")
                        && row.get("input").and_then(Value::as_str)
                            == Some("shared_activation + shared_down")))
            });
            for row in &mut dispatch_ledger {
                if let Some(kernel) = row.get_mut("kernel") {
                    if kernel.as_str() == Some("qwen_next_bf16_expert_gate_up_swiglu") {
                        *kernel = Value::String(
                            "qwen_next_bf16_compact_expert_gate_up_shared_swiglu".into(),
                        );
                    } else if kernel.as_str()
                        == Some("qwen_next_moe_weighted_sum_add_shared_sigmoid_hc")
                    {
                        *kernel = Value::String(
                            "qwen_next_bf16_compact_expert_down_shared_direct_hc".into(),
                        );
                        if let Some(input) = row.get_mut("input") {
                            *input = Value::String("route_ids + compact expert_down + route_weights + shared_activation + shared_down + shared_scalar + post_attn_state + mlp_block_logits".into());
                        }
                        if let Some(output) = row.get_mut("output") {
                            *output = Value::String("routed_sum + shared_output + shared_gated_output + moe_output + final_state".into());
                        }
                    }
                }
            }
        }
        if fused_hc_router() {
            dispatch_ledger.retain(|row| {
                let kernel = row.get("kernel").and_then(Value::as_str);
                let input = row.get("input").and_then(Value::as_str);
                !(kernel == Some("moe_topk_gate")
                    || (kernel == Some("gemv_native_bf16_dual_seq")
                        && input == Some("mlp_input + router + shared_scalar")))
            });
            for row in &mut dispatch_ledger {
                let is_mlp_input = row.get("kernel").and_then(Value::as_str)
                    == Some("qwen_next_hyperconnection_input_fused_with_block")
                    && row.get("input").and_then(Value::as_str)
                        == Some("post_attn_state + hc_norm + hc_down + hc_up + hc_mlp_block");
                if is_mlp_input {
                    row["kernel"] = Value::String(
                        "qwen_next_hyperconnection_input_fused_with_block_router_topk".into(),
                    );
                    row["input"] = Value::String(
                        "post_attn_state + hc_norm + hc_down + hc_up + hc_mlp_block + router + shared_scalar".into(),
                    );
                    row["output"] = Value::String(
                        "mlp_norm + low_rank + activation + gate_logits + mlp_input + mlp_block_logits + router_logits + shared_scalar + route_ids + route_weights".into(),
                    );
                    row["fusion_candidate"] = Value::String(
                        "fuse MLP HyperConnection, block projection, router, shared gate, and normalized top-k".into(),
                    );
                    row["why_it_exists"] = Value::String(
                        "keep the MLP input in threadgroup memory across routing with no global producer/consumer edge".into(),
                    );
                }
            }
        } else if fused_router_topk() {
            dispatch_ledger
                .retain(|row| row.get("kernel").and_then(Value::as_str) != Some("moe_topk_gate"));
            for row in &mut dispatch_ledger {
                let is_router_projection = row.get("kernel").and_then(Value::as_str)
                    == Some("gemv_native_bf16_dual_seq")
                    && row.get("input").and_then(Value::as_str)
                        == Some("mlp_input + router + shared_scalar");
                if is_router_projection {
                    row["kernel"] = Value::String("qwen_next_bf16_router_topk_shared".into());
                    row["output"] = Value::String(
                        "router_logits + shared_scalar + route_ids + route_weights".into(),
                    );
                    row["fusion_candidate"] = Value::String(
                        "fuse router GEMV, shared gate, and normalized top-k selection".into(),
                    );
                    row["why_it_exists"] = Value::String(
                        "one device-resident router boundary with no logits round-trip".into(),
                    );
                }
            }
        }
        if args.compact_experts && fused_moe_vec4() {
            for row in &mut dispatch_ledger {
                match row.get("kernel").and_then(Value::as_str) {
                    Some("qwen_next_bf16_compact_expert_gate_up_shared_swiglu") => {
                        row["kernel"] = Value::String(
                            "qwen_next_bf16_compact_expert_gate_up_shared_swiglu_vec4".into(),
                        );
                    }
                    Some("qwen_next_bf16_compact_expert_down_shared_direct_hc") => {
                        row["kernel"] = Value::String(
                            "qwen_next_bf16_compact_expert_down_shared_direct_hc_vec4".into(),
                        );
                    }
                    _ => {}
                }
            }
        }
        if dispatch_ledger.len() != expected_graph_dispatches {
            return Err(format!(
                "Flash {layer_label} dispatch ledger drifted: {} rows expected {}",
                dispatch_ledger.len(),
                expected_graph_dispatches
            )
            .into());
        }
        for index in 0..expected_graph_dispatches {
            let mut gpu_values = Vec::new();
            let mut host_values = Vec::new();
            for samples in &dispatch_samples {
                if let Some(sample) = samples.get(index) {
                    if let Some(value) = sample.gpu_us {
                        gpu_values.push(value.saturating_mul(1000));
                    }
                    host_values.push(sample.wall_us);
                }
            }
            if let Some(object) = dispatch_ledger[index].as_object_mut() {
                object.insert(
                    "gpu_ns".to_owned(),
                    median(&mut gpu_values)
                        .map(Value::from)
                        .unwrap_or(Value::Null),
                );
                object.insert(
                    "host_encode_us".to_owned(),
                    median(&mut host_values)
                        .map(Value::from)
                        .unwrap_or(Value::Null),
                );
            }
        }
        let dispatch_ledger_value = json!({
            "schema": DISPATCH_LEDGER_SCHEMA,
            "status": "PASSED",
            "layer": args.layer,
            "source_revision": PINNED_REVISION,
            "dispatch_count": expected_graph_dispatches,
            "trace_mode": trace_mode,
            "integrated_graph": true,
            "rows": dispatch_ledger,
            "claim_boundary": "logical dispatch ledger; per-dispatch GPU ns is populated only when the explicit diagnostic trace mode supplies it; integrated graph GPU ns is authoritative in the layer receipt",
            "promotion_allowed": false
        });
        let expert_kernel_label = if args.compact_experts && fused_moe_vec4() {
            "qwen_next_bf16_compact_expert_gate_up_shared_swiglu_vec4 + qwen_next_bf16_compact_expert_down_shared_direct_hc_vec4"
        } else if args.compact_experts {
            "qwen_next_bf16_compact_expert_gate_up_shared_swiglu + qwen_next_bf16_compact_expert_down_shared_direct_hc"
        } else {
            "qwen_next_bf16_expert_gate_up_swiglu + qwen_next_bf16_expert_down + qwen_next_moe_weighted_sum_add_shared_sigmoid_hc"
        };
        let critical_rows = vec![
            ("norm", "qwen_next_hyperconnection_grouped_rmsnorm"),
            ("projections", "gemv_native_bf16_dual_seq + gemv_native_bf16_seq"),
            ("DeltaNet state", "qwen_next_gated_delta_decode_single"),
            ("recurrence", "qwen_next_gated_delta_decode_single"),
            (
                "attention",
                "qwen_next_deltanet_source_bf16_gated_rmsnorm + gemv_native_bf16_hyperconnection_combine",
            ),
            (
                "routing",
                if fused_hc_router() {
                    "qwen_next_hyperconnection_input_fused_with_block_router_topk"
                } else if fused_router_topk() {
                    "qwen_next_bf16_router_topk_shared"
                } else {
                    "gemv_native_bf16_dual_seq + moe_topk_gate"
                },
            ),
            (
                "selected experts",
                expert_kernel_label,
            ),
            (
                "shared expert",
                if args.compact_experts {
                    "qwen_next_bf16_compact_expert_gate_up_shared_swiglu + shared_down in direct HC epilogue"
                } else {
                    "gemv_native_bf16_swiglu_seq + gemv_native_bf16_seq"
                },
            ),
            (
                "residual",
                if args.compact_experts {
                    "qwen_next_bf16_compact_expert_down_shared_direct_hc"
                } else {
                    "qwen_next_moe_weighted_sum_add_shared_sigmoid_hc"
                },
            ),
            ("synchronization", "TokenCommandBuffer commit_and_wait"),
            ("command submission", "Metal command buffer"),
            (
                "representation conversion",
                "none; source BF16 remains resident",
            ),
        ];
        let critical_path = json!({
            "schema": CRITICAL_PATH_SCHEMA,
            "status": "PASSED",
            "layer": args.layer,
            "source_revision": PINNED_REVISION,
            "owners": critical_rows.into_iter().map(|(owner, kernels)| json!({
                "owner": owner,
                "kernels": kernels,
                "ns": Value::Null,
                "measurement": "per-dispatch GPU ownership requires HAWKING_TCB_TRACE=gpu; command-buffer GPU interval is recorded separately"
            })).collect::<Vec<_>>(),
            "integrated_graph_gpu_ns": graph_gpu_ns,
            "integrated_graph_host_ns": graph_host_ns,
            "claim_boundary": "no vague runtime-overhead bucket; un-attributed per-dispatch GPU ownership remains explicitly unmeasured when trace mode is off",
            "promotion_allowed": false
        });

        let source_weight_bytes = all_tensors(&weights)
            .iter()
            .filter_map(|value| value.get("bytes").and_then(Value::as_u64))
            .sum::<u64>();
        let active_expert_weight_bytes =
            (TOP_K * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2) as u64;
        let logical_active_bytes = source_weight_bytes
            .saturating_sub(weights.expert_gate_up.bytes.len() as u64)
            .saturating_sub(weights.expert_down.bytes.len() as u64)
            .saturating_add(active_expert_weight_bytes);
        let final_hash = output_hashes.first().cloned().unwrap_or_default();
        let status = if parity_passed && deterministic {
            "PASSED"
        } else {
            "BLOCKED"
        };
        let qualification = if parity_passed && deterministic {
            "EXACT_LAYER_SOURCE_PARITY"
        } else {
            "BLOCKED"
        };
        let output_state = if status == "PASSED" {
            if let Some(path) = args.state_output.as_ref() {
                let bytes = f32_bytes(&observed_final_state);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let state_write_started = Instant::now();
                fs::write(path, &bytes)?;
                Some(json!({
                    "kind": "f32_state",
                    "path": path,
                    "bytes": bytes.len(),
                    "elements": observed_final_state.len(),
                    "sha256": sha256_bytes(&bytes),
                    "dtype": "F32_LE",
                    "source": "verified device-final-state snapshot from exact source-parity layer"
                    ,"write_ns": state_write_started.elapsed().as_nanos() as u64
                }))
            } else {
                None
            }
        } else {
            None
        };
        let state_write_ns = output_state
            .as_ref()
            .and_then(|value| value.get("write_ns"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let recorded_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_millis())
            .unwrap_or_default();
        let source_bf16_vec4 = hawking_core::env_on("HAWKING_FLASH_BF16_VEC4");
        let source_bf16_geo = hawking_core::env_on("HAWKING_FLASH_BF16_GEO");
        let source_bf16_geo_dual = hawking_core::env_on("HAWKING_FLASH_BF16_GEO_DUAL");
        let source_bf16_gemv_kernel = if source_bf16_geo {
            "gemv_native_bf16_geo_vec4_tg128"
        } else if source_bf16_vec4 {
            "gemv_native_bf16_seq_vec4"
        } else {
            "gemv_native_bf16_seq"
        };
        let source_bf16_dual_kernel = if source_bf16_geo_dual {
            "gemv_native_bf16_dual_geo_vec4_tg128"
        } else if source_bf16_vec4 {
            "gemv_native_bf16_dual_seq_vec4"
        } else {
            "gemv_native_bf16_dual_seq"
        };
        let source_bf16_swiglu_kernel = if source_bf16_geo {
            "gemv_native_bf16_swiglu_geo_vec4_tg128"
        } else if source_bf16_vec4 {
            "gemv_native_bf16_swiglu_seq_vec4"
        } else {
            "gemv_native_bf16_swiglu_seq"
        };
        let compact_moe_kernel = if fused_moe_vec4() {
            "qwen_next_bf16_compact_expert_down_shared_direct_hc_vec4"
        } else if hawking_core::env_on("HAWKING_FLASH_MOE_GEO") {
            "qwen_next_bf16_compact_expert_down_shared_direct_hc_geo_tg128"
        } else {
            "qwen_next_bf16_compact_expert_down_shared_direct_hc"
        };
        let compact_gate_up_kernel = if fused_moe_vec4() {
            "qwen_next_bf16_compact_expert_gate_up_shared_swiglu_vec4"
        } else {
            "qwen_next_bf16_compact_expert_gate_up_shared_swiglu"
        };
        let receipt = json!({
            "schema": SCHEMA,
            "artifact_kind": "SOURCE",
            "status": status,
            "qualification": qualification,
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "akb_registration": {
                "evidence_domain": "accelerator",
                "civilization": "I-D_ACCELERATOR",
                "program": "CUDA-capability translation / Apple Silicon repatriation",
                "machine_scope": device,
                "representation_scope": "source BF16 exact weights + f32 activations",
                "kernel_scope": format!("{layer_label} GEMV, DeltaNet, routed/shared MoE and HyperConnection")
            },
            "bench": {
                "state": "UNKNOWN",
                "recorded_at": format!("unix-ms:{recorded_at_unix_ms}"),
                "recorded_by": format!("flash_noetic_complete_layer{}", args.layer),
                "machine": device,
                "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet",
                "provenance": "Physical source-parity run records no independent quiescence sample."
            },
            "source": {
                "manifest": manifest,
                "config_sha256": config_sha256,
                "config_model_type": config.get("model_type"),
                "layer_index": args.layer,
                "layer_type": "linear_attention",
                "layer_count": 48,
                "geometry": {
                    "hidden": HIDDEN,
                    "streams": STREAMS,
                    "hc_lowrank": HC_LOWRANK,
                    "key_heads": KEY_HEADS,
                    "value_heads": VALUE_HEADS,
                    "key_head_dim": KEY_HEAD_DIM,
                    "value_head_dim": VALUE_HEAD_DIM,
                    "experts": EXPERTS,
                    "top_k": TOP_K,
                    "intermediate": INTERMEDIATE
                },
                "tensors": all_tensors(&weights),
                "embedding": embedding_source,
                "input_state": input_state,
                "output_state": output_state
            },
            "input_contract": if args.base_state.is_some() { "FLASH_NEXT_PREFIX_FED_STATE" } else { "FLASH_NEXT_TEXT_BASELINE_BOS_SOURCE_EMBEDDING" },
            "expert_bank": {
                "mode": if args.compact_experts { "compact_routed" } else { "dense_full" },
                "source_experts": EXPERTS,
                "resident_experts": weights.expert_gate_up.shape.first().copied().unwrap_or(EXPERTS),
                "route_ids": expected.route_ids.clone(),
                "route_lut": weights.expert_lut.clone(),
            },
            "execution": {
                "device": device,
                "provider": "apple_metal",
                "native_source_bf16": true,
                "single_ordered_token_command_buffer": true,
                "dispatches": expected_graph_dispatches,
                "command_buffers_per_rep": command_timings.iter().map(|timing| timing.command_buffers).collect::<Vec<_>>(),
                "compute_encoders_per_rep": command_timings.iter().map(|timing| timing.encoder_count).collect::<Vec<_>>(),
                "graph_gpu_ns": graph_gpu_ns,
                "graph_host_ns": graph_host_ns,
                "trace_mode": trace_mode,
                "source_bf16_vec4_candidate": source_bf16_vec4,
                "source_bf16_geo_candidate": source_bf16_geo,
                "source_bf16_geo_dual_candidate": source_bf16_geo_dual,
                "source_bf16_moe_vec4_candidate": fused_moe_vec4(),
                "router_topk_fused_candidate": fused_router_topk() || fused_hc_router(),
                "router_topk_fused_into_mlp_hc_candidate": fused_hc_router(),
                "host_activation_roundtrips": 0,
                "fallback_count": 0,
                "source_cpu_oracle_ns": cpu_oracle_ns,
                "embedding_read_ns": input_read_ns
                ,"expert_bank_mode": if args.compact_experts { "compact_routed" } else { "dense_full" }
            },
            "timing": {
                "source_load_ns": source_load_ns,
                "device_prepare_ns": device_prepare_ns,
                "graph_setup_ns": graph_setup_ns,
                "warmup_ns": warmup_ns,
                "state_fingerprint_ns": 0,
                "state_write_ns": state_write_ns,
                "receipt_write_ns": Value::Null
            },
            "state": {
                "recurrent_state_bytes": RECURRENT_STATE_ELEMENTS * 4,
                "causal_conv_state_bytes": CONV_STATE_ELEMENTS * 4,
                "state_update": "device_resident_in_place",
                "reset_between_repetitions": true
            },
            "parity": {
                "stages": stage_metrics,
                "route_ids_expected": expected.route_ids,
                "route_ids_observed": observed_route_ids,
                "route_ids_match": route_ids_match,
                "route_weights": route_weight_metrics,
                "final_output_hash": final_hash,
                "repeated_output_hashes": output_hashes,
                "repeated_deterministic": deterministic,
                "all_stage_tolerance": OUTPUT_TOLERANCE,
                "passed": parity_passed
            },
            "bytes": {
                "source_payload_bytes_read": source_bytes_read,
                "source_layer_weight_bytes": source_weight_bytes,
                "active_selected_expert_weight_bytes_per_graph": active_expert_weight_bytes,
                "logical_active_weight_bytes_per_graph": logical_active_bytes,
                "active_representation": "source BF16 weights + f32 activations + device route IDs",
                "measurement_boundary": "logical tensor traffic from graph bindings; not a DRAM counter claim"
            },
            "native_kernels": if args.compact_experts { json!([
                source_bf16_gemv_kernel,
                source_bf16_dual_kernel,
                "qwen_next_hyperconnection_grouped_rmsnorm",
                "qwen_next_hyperconnection_silu_scale",
                "qwen_next_hyperconnection_read_mix",
                "qwen_next_qkv_split_rearrange_conv_l2",
                "qwen_next_ba_split_to_decay_beta_source_bf16",
                "qwen_next_gated_delta_decode_single",
                "qwen_next_deltanet_source_bf16_gated_rmsnorm",
                if fused_hc_router() {
                    "qwen_next_hyperconnection_input_fused_with_block_router_topk"
                } else if fused_router_topk() {
                    "qwen_next_bf16_router_topk_shared"
                } else {
                    "moe_topk_gate"
                },
                compact_gate_up_kernel,
                compact_moe_kernel,
            ]) } else { json!([
                source_bf16_gemv_kernel,
                source_bf16_dual_kernel,
                "qwen_next_hyperconnection_grouped_rmsnorm",
                "qwen_next_hyperconnection_silu_scale",
                "qwen_next_hyperconnection_read_mix",
                "qwen_next_qkv_split_rearrange_conv_l2",
                "qwen_next_ba_split_to_decay_beta_source_bf16",
                "qwen_next_gated_delta_decode_single",
                "qwen_next_deltanet_source_bf16_gated_rmsnorm",
                if fused_hc_router() {
                    "qwen_next_hyperconnection_input_fused_with_block_router_topk"
                } else if fused_router_topk() {
                    "qwen_next_bf16_router_topk_shared"
                } else {
                    "moe_topk_gate"
                },
                "qwen_next_bf16_expert_gate_up_swiglu",
                "qwen_next_bf16_expert_down",
                source_bf16_swiglu_kernel,
                source_bf16_gemv_kernel,
                "qwen_next_moe_weighted_sum_add_shared_sigmoid_hc"
            ]) },
            "physical_graph": {
                "semantic_type": "PhysicalGraph",
                "representation_identity": "source_bf16_exact",
                "state_ownership": "Metal device buffers owned by this graph invocation",
                "device_residency": "all activations, route IDs, recurrent state and expert weights remain device resident until post-fence verification",
                "dependencies": "ordered TokenCommandBuffer dispatch sequence",
                "native_execution_observed": true,
                "no_dense_expert_rematerialization": true
                ,"expert_bank_addressing": if args.compact_experts { "route_id_to_compact_slot_lut" } else { "native_expert_id_stride" }
            },
            "ledgers": {
                "dispatch": DISPATCH_LEDGER_SCHEMA,
                "critical_path": CRITICAL_PATH_SCHEMA,
                "dispatch_receipt_path": args.out.with_file_name(format!("FLASH_LAYER{}_DISPATCH_LEDGER.json", args.layer)),
                "critical_path_receipt_path": args.out.with_file_name(format!("FLASH_LAYER{}_CRITICAL_PATH.json", args.layer))
            },
            "complete_token_runtime": "NOT_TESTED",
            "flash_tps": Value::Null,
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "claim_boundary": format!("This is one exact source-parity {layer_label} execution on the explicitly identified state. It is not the 48-layer Flash token, not greedy decode, not official TPS, not EBPW qualification, and not multimodal completeness."),
            "elapsed_wall_ns": started.elapsed().as_nanos() as u64
        });
        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
        let dispatch_out = args
            .out
            .with_file_name(format!("FLASH_LAYER{}_DISPATCH_LEDGER.json", args.layer));
        let critical_out = args
            .out
            .with_file_name(format!("FLASH_LAYER{}_CRITICAL_PATH.json", args.layer));
        fs::write(
            &dispatch_out,
            serde_json::to_vec_pretty(&dispatch_ledger_value)?,
        )?;
        fs::write(&critical_out, serde_json::to_vec_pretty(&critical_path)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        if status == "PASSED" {
            Ok(())
        } else {
            eprintln!(
                "Flash {layer_label} source parity is BLOCKED; receipt preserved at {}",
                args.out.display()
            );
            Ok(())
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        main_impl_with_args(parse_args()?)
    }

    pub(crate) fn run_layer(args: Args) -> Result<(), Box<dyn Error>> {
        if args.prefix_layers == 0 {
            return Err("in-process Flash executor requires at least one layer".into());
        }
        main_impl_with_args(args)
    }

    /// Execute a linear prefix and retain its final Metal buffer for a
    /// following structural species.  The ordinary `run_layer` API remains a
    /// receipt-only compatibility path; this opt-in API is used by the
    /// cross-species protected probe in `flash_fast_chain`.
    pub(crate) fn run_layer_device_output(
        args: Args,
        input: Option<&PinnedBuffer>,
    ) -> Result<Option<PinnedBuffer>, Box<dyn Error>> {
        if !args.device_resident || args.prefix_layers == 0 {
            return Err("device output requires --device-resident and at least one layer".into());
        }
        run_linear_prefix(&args, input)
    }

    /// Measure token-to-token state continuity for one exact linear Flash organ.
    ///
    /// The source index, Metal context, weights, graph buffers, convolution
    /// state, and DeltaNet recurrent state all live for the whole probe.  This
    /// is intentionally an organ result: it proves neither full-model token
    /// acceptance nor a complete-model TPS figure.
    pub(crate) fn run_stateful_token_probe(
        root: PathBuf,
        token_ids: &[usize],
        out: PathBuf,
    ) -> Result<(), Box<dyn Error>> {
        if token_ids.is_empty() {
            return Err("stateful token probe requires at least one token".into());
        }
        let root = root.canonicalize()?;
        let (_manifest, _config_sha256, vocab) = validate_config(&root)?;
        let (index, context) = cached_flash_resources(&root)?;
        let (first_row, first_embedding_sha, first_embedding_bytes) =
            load_embedding_row(&index, vocab, token_ids[0])?;
        let first_base = repeat_streams(&first_row);
        // Compact routing is safe here because repeated probe tokens use the
        // same route, and it keeps this state qualification bounded in memory.
        let weights = load_layer_weights_compact(&index, 0, &first_base)?;
        let expected_first = source_layer_from_base(&weights, &first_base);
        let device_weights = load_device_weights(&context, &weights)?;
        let graph = new_graph_buffers(&context, &first_base)?;
        let mut rows = Vec::with_capacity(token_ids.len());
        for (step, &token_id) in token_ids.iter().enumerate() {
            let (embedding, embedding_sha, embedding_bytes) =
                load_embedding_row(&index, vocab, token_id)?;
            MetalContext::write_buffer_bytes(&graph.base, &f32_bytes(&repeat_streams(&embedding)));
            if step == 0 {
                reset_states(&context, &graph);
            }
            let started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&context);
            encode_graph(&mut tcb, &device_weights, &graph)?;
            let encoded_dispatches = tcb.dispatch_count();
            let timing = tcb.commit_and_wait_timed()?;
            let wall_ns = started.elapsed().as_nanos() as u64;
            let expected_dispatches = expected_graph_dispatches(&device_weights);
            if encoded_dispatches != expected_dispatches
                || timing.dispatches != expected_dispatches as u64
            {
                return Err(format!("stateful layer-0 dispatch topology drifted at step {step}: encoded={encoded_dispatches} timed={} expected={expected_dispatches}", timing.dispatches).into());
            }
            let final_state = snapshot_f32(&graph.final_state, HC_ELEMENTS);
            let recurrent_state = snapshot_f32(&graph.recurrent_state, RECURRENT_STATE_ELEMENTS);
            if final_state.iter().any(|v| !v.is_finite())
                || recurrent_state.iter().any(|v| !v.is_finite())
            {
                return Err(
                    format!("stateful layer-0 produced non-finite state at step {step}").into(),
                );
            }
            let parity = if step == 0 {
                metrics(&expected_first.final_state, &final_state, OUTPUT_TOLERANCE)
            } else {
                json!({"status": "STATEFUL_CONTINUATION_NOT_ORACLED"})
            };
            rows.push(json!({
                "step": step,
                "token_id": token_id,
                "embedding_sha256": embedding_sha,
                "embedding_bytes": embedding_bytes,
                "dispatches": timing.dispatches,
                "command_buffers": timing.command_buffers,
                "gpu_ns": timing.gpu_ns,
                "wall_ns": wall_ns,
                "final_state_sha256": sha256_bytes(&f32_bytes(&final_state)),
                "recurrent_state_sha256": sha256_bytes(&f32_bytes(&recurrent_state)),
                "recurrent_state_l2": recurrent_state.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt(),
                "first_token_parity": parity,
                "state_reset": step == 0,
            }));
        }
        let state_changed = rows.windows(2).any(|pair| {
            pair[0].get("recurrent_state_sha256") != pair[1].get("recurrent_state_sha256")
        });
        let doc = json!({
            "schema": "hawking.flash.stateful_linear_organ_probe.v1",
            "status": if state_changed { "PASSED_STATEFUL_ORGAN" } else { "BLOCKED_STATE_DID_NOT_CHANGE" },
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "layer": 0,
            "token_ids": token_ids,
            "execution": {
                "device": context.device_name(),
                "provider": "apple_metal",
                "process_boundary": "one native process",
                "source_index_reused": true,
                "metal_context_reused": true,
                "weights_reused": true,
                "graph_buffers_reused": true,
                "conv_and_recurrent_state_persisted": true,
                "source_payload_bytes_read": index.bytes_read_total(),
                "first_embedding_sha256": first_embedding_sha,
                "first_embedding_bytes": first_embedding_bytes,
            },
            "steps": rows,
            "state_changed_between_steps": state_changed,
            "accepted_generation_tokens": 0,
            "accepted_tps": Value::Null,
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_token_probe", "machine": context.device_name(), "rule": "S032 §3 -- stateful organ timing; quiescence unknown"},
            "claim_boundary": "This proves persistent state across repeated token steps for the layer-0 DeltaNet/HyperConnection/MoE organ, with first-step source parity. It does not prove accepted generation, full-model TPS, full-model EBPW, full-attention KV persistence, or resident promotion.",
            "next": "Use this state contract when adding per-layer state arrays to the complete 48-layer Flash session; then qualify full-attention KV state and tokenizer acceptance before TPS.",
        });
        let mut sealed = doc;
        sealed["seal_sha256"] = Value::String(sha256_bytes(&serde_json::to_vec(&sealed)?));
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&sealed)?)?;
        println!("{}", serde_json::to_string_pretty(&sealed)?);
        Ok(())
    }

    pub(crate) struct StatefulLinearLayer {
        layer: usize,
        weights: LayerWeights,
        device_weights: DeviceWeights,
        graph: GraphBuffers,
    }

    impl StatefulLinearLayer {
        pub(crate) fn new(
            index: &SourceBf16Index,
            context: &MetalContext,
            layer: usize,
            first_base: &[f32],
        ) -> Result<Self, Box<dyn Error>> {
            let weights = load_layer_weights_compact(index, layer, first_base)?;
            let device_weights = load_device_weights(context, &weights)?;
            let graph = new_graph_buffers(context, &[0.0; HC_ELEMENTS])?;
            Ok(Self {
                layer,
                weights,
                device_weights,
                graph,
            })
        }

        /// Dense-bank constructor for a real token sequence.  The compact
        /// constructor above is intentionally route-specialized for bounded
        /// probes; it is not exact when successive tokens select different
        /// experts.  Complete-session execution must bind the full immutable
        /// bank and let the native router choose per token.
        pub(crate) fn new_dense(
            index: &SourceBf16Index,
            context: &MetalContext,
            layer: usize,
            first_base: &[f32],
        ) -> Result<Self, Box<dyn Error>> {
            let weights = load_layer_weights(index, layer)?;
            let device_weights = load_device_weights(context, &weights)?;
            let graph = new_graph_buffers(context, first_base)?;
            Ok(Self {
                layer,
                weights,
                device_weights,
                graph,
            })
        }

        /// Construct a compact layer from a caller-proven union of routes.
        /// This is intentionally separate from `new`, whose first-token route
        /// specialization is unsafe for changing token routes.
        pub(crate) fn new_compact_union(
            index: &SourceBf16Index,
            context: &MetalContext,
            layer: usize,
            route_ids: &[u32],
            first_base: &[f32],
        ) -> Result<Self, Box<dyn Error>> {
            let weights = load_layer_weights_compact_union(index, layer, route_ids)?;
            let device_weights = load_device_weights(context, &weights)?;
            let graph = new_graph_buffers(context, first_base)?;
            Ok(Self {
                layer,
                weights,
                device_weights,
                graph,
            })
        }

        pub(crate) fn step(
            &mut self,
            context: &MetalContext,
            host_base: Option<&[f32]>,
            device_base: Option<&PinnedBuffer>,
            reset: bool,
        ) -> Result<(PinnedBuffer, u64, u64, usize, Vec<f32>), Box<dyn Error>> {
            if let Some(base) = host_base {
                MetalContext::write_buffer_bytes(&self.graph.base, &f32_bytes(base));
            }
            let started = Instant::now();
            if reset {
                reset_states(context, &self.graph);
            }
            let mut tcb = TokenCommandBuffer::new(context);
            if let Some(previous) = device_base {
                tcb.copy_buffer_bytes(previous, 0, &self.graph.base, 0, (HC_ELEMENTS * 4) as u64)?;
            }
            encode_graph(&mut tcb, &self.device_weights, &self.graph)?;
            let dispatches = tcb.dispatch_count();
            let timing = tcb.commit_and_wait_timed()?;
            let expected_dispatches = expected_graph_dispatches(&self.device_weights);
            if dispatches != expected_dispatches || timing.dispatches != expected_dispatches as u64
            {
                return Err(format!("stateful layer-{} dispatch topology drifted: encoded={dispatches} timed={} expected={expected_dispatches}", self.layer, timing.dispatches).into());
            }
            let wall_ns = started.elapsed().as_nanos() as u64;
            let final_state = snapshot_f32(&self.graph.final_state, HC_ELEMENTS);
            if final_state.iter().any(|v| !v.is_finite()) {
                return Err(
                    format!("stateful layer-{} produced non-finite output", self.layer).into(),
                );
            }
            Ok((
                self.graph.final_state.clone(),
                timing.gpu_ns.unwrap_or(0),
                wall_ns,
                timing.dispatches as usize,
                final_state,
            ))
        }

        /// Read the router's selected original expert IDs after a step. This
        /// is a tiny diagnostic read used by route-stability audits; it is not
        /// part of the activation handoff and never drives the fast path.
        pub(crate) fn route_ids(&self) -> Vec<u32> {
            snapshot_u32(&self.graph.route_ids, TOP_K)
        }

        /// Read the exact layer-local MLP input consumed by the router and
        /// routed/shared expert banks.  This is intentionally separate from
        /// the HyperConnection state handoff: a meta teacher surface for
        /// `mlp.experts.gate_up_proj` must use this post-HC vector, not a raw
        /// `[streams, hidden]` state snapshot.
        pub(crate) fn mlp_input(&self) -> Vec<f32> {
            snapshot_f32(&self.graph.mlp_input, HIDDEN)
        }

        /// Compare the first source-authority MLP input against the device
        /// surface.  The CPU oracle intentionally starts recurrent state from
        /// zero, so callers should use this for the reset/first-token row only;
        /// later rows remain stateful device-teacher observations.
        pub(crate) fn source_mlp_input_parity(&self, base: &[f32]) -> Value {
            let expected = source_layer_from_base(&self.weights, base);
            metrics(&expected.mlp_input, &self.mlp_input(), OUTPUT_TOLERANCE)
        }

        pub(crate) fn source_route_ids(&self, base: &[f32]) -> Vec<u32> {
            source_layer_from_base(&self.weights, base).route_ids
        }
    }

    /// Persistent 0..2 linear-prefix session. Each layer's device state is
    /// retained across tokens and inter-layer activations never round-trip
    /// through host memory. The prefix stops immediately before layer 3's
    /// full-attention species, so it cannot be mistaken for full-model TPS.
    pub(crate) fn run_stateful_linear_prefix_session(
        root: PathBuf,
        token_ids: &[usize],
        out: PathBuf,
    ) -> Result<(), Box<dyn Error>> {
        run_stateful_linear_prefix_session_mode(root, token_ids, out, false, None)
    }

    /// Bounded route-audit variant. `dense=true` preserves the full expert
    /// bank so later-token router IDs can seed an exact compact union; the
    /// default remains the historical first-token compact probe.
    pub(crate) fn run_stateful_linear_prefix_session_mode(
        root: PathBuf,
        token_ids: &[usize],
        out: PathBuf,
        dense: bool,
        union_from: Option<PathBuf>,
    ) -> Result<(), Box<dyn Error>> {
        if token_ids.len() < 2 {
            return Err("stateful prefix session requires at least two tokens".into());
        }
        let root = root.canonicalize()?;
        let (_config, _config_sha256, vocab) = validate_config(&root)?;
        let (index, context) = cached_flash_resources(&root)?;
        let layer_count = 3usize;
        let (first_embedding, first_embedding_sha, first_embedding_bytes) =
            load_embedding_row(&index, vocab, token_ids[0])?;
        let mut expected_base = repeat_streams(&first_embedding);
        let mut expected_finals = Vec::with_capacity(layer_count);
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let union_routes = union_from
                .as_ref()
                .map(|path| {
                    let payload: Value = serde_json::from_slice(&fs::read(path)?)?;
                    let mut routes = Vec::new();
                    for row in payload
                        .get("steps")
                        .and_then(Value::as_array)
                        .unwrap_or(&Vec::new())
                    {
                        if row.get("layer").and_then(Value::as_u64) == Some(layer as u64) {
                            if let Some(ids) = row.get("route_ids").and_then(Value::as_array) {
                                routes.extend(
                                    ids.iter().filter_map(Value::as_u64).map(|id| id as u32),
                                );
                            }
                        }
                    }
                    routes.sort_unstable();
                    routes.dedup();
                    if routes.is_empty() {
                        return Err(
                            format!("route-union receipt has no routes for layer {layer}").into(),
                        );
                    }
                    Ok::<Vec<u32>, Box<dyn Error>>(routes)
                })
                .transpose()?;
            let session = if let Some(routes) = union_routes.as_ref() {
                StatefulLinearLayer::new_compact_union(
                    &index,
                    &context,
                    layer,
                    routes,
                    &expected_base,
                )?
            } else if dense {
                StatefulLinearLayer::new_dense(&index, &context, layer, &expected_base)?
            } else {
                StatefulLinearLayer::new(&index, &context, layer, &expected_base)?
            };
            let expected = source_layer_from_base(&session.weights, &expected_base);
            expected_base = expected.final_state.clone();
            expected_finals.push(expected.final_state);
            layers.push(session);
        }
        let mut rows = Vec::new();
        for (step, &token_id) in token_ids.iter().enumerate() {
            let (embedding, embedding_sha, embedding_bytes) =
                load_embedding_row(&index, vocab, token_id)?;
            let host_base = repeat_streams(&embedding);
            let mut prior_device: Option<PinnedBuffer> = None;
            for (layer_index, layer) in layers.iter_mut().enumerate() {
                let (output, gpu_ns, wall_ns, dispatches, final_state) = if layer_index == 0 {
                    layer.step(&context, Some(&host_base), None, step == 0)?
                } else {
                    let input = prior_device
                        .as_ref()
                        .ok_or("missing device inter-layer state")?;
                    layer.step(&context, None, Some(input), step == 0)?
                };
                let first_parity = if step == 0 {
                    Some(metrics(
                        &expected_finals[layer_index],
                        &final_state,
                        OUTPUT_TOLERANCE,
                    ))
                } else {
                    None
                };
                let route_ids = layer.route_ids();
                rows.push(json!({
                    "step": step,
                    "token_id": token_id,
                    "layer": layer.layer,
                    "embedding_sha256": embedding_sha,
                    "embedding_bytes": embedding_bytes,
                    "dispatches": dispatches,
                    "gpu_ns": gpu_ns,
                    "wall_ns": wall_ns,
                    "device_inter_layer_handoff": layer_index > 0,
                    "final_state_sha256": sha256_bytes(&f32_bytes(&final_state)),
                    "recurrent_state_sha256": sha256_bytes(&f32_bytes(&snapshot_f32(&layer.graph.recurrent_state, RECURRENT_STATE_ELEMENTS))),
                    "route_ids": route_ids,
                    "first_token_parity": first_parity,
                    "state_reset": step == 0,
                }));
                prior_device = Some(output);
            }
        }
        let state_changed_layers = (0..layer_count)
            .filter(|layer| {
                let first = rows.iter().find(|row| {
                    row.get("layer").and_then(Value::as_u64) == Some(*layer as u64)
                        && row.get("step").and_then(Value::as_u64) == Some(0)
                });
                let last = rows.iter().find(|row| {
                    row.get("layer").and_then(Value::as_u64) == Some(*layer as u64)
                        && row.get("step").and_then(Value::as_u64) == Some(1)
                });
                first.and_then(|r| r.get("recurrent_state_sha256"))
                    != last.and_then(|r| r.get("recurrent_state_sha256"))
            })
            .count();
        let mut doc = json!({
            "schema": "hawking.flash.stateful_linear_prefix_session.v1",
            "status": if state_changed_layers == layer_count { "PASSED_STATEFUL_PREFIX_SESSION" } else { "BLOCKED_LAYER_STATE_DID_NOT_CHANGE" },
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "layer_range": [0, layer_count - 1],
            "token_ids": token_ids,
            "execution": {"device": context.device_name(), "provider": "apple_metal", "process_boundary": "one native process", "source_index_reused": true, "metal_context_reused": true, "weights_reused": true, "graph_buffers_reused": true, "per_layer_state_persisted": true, "device_inter_layer_handoffs": true, "host_activation_roundtrips": 0, "source_payload_bytes_read": index.bytes_read_total(), "first_embedding_sha256": first_embedding_sha, "first_embedding_bytes": first_embedding_bytes, "expert_bank_mode": if union_from.is_some() { "route_union_compact" } else if dense { "dense_audit" } else { "first_token_compact" }},
            "steps": rows,
            "state_changed_layers": state_changed_layers,
            "accepted_generation_tokens": 0,
            "accepted_tps": Value::Null,
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_linear_prefix_session", "machine": context.device_name(), "rule": "S032 §3 -- stateful prefix timing; quiescence unknown"},
            "claim_boundary": "This proves a persistent one-process Flash layers-0..2 linear-prefix session with device-only inter-layer handoffs and per-layer recurrence across two steps. It does not prove layer-3 KV integration, full-model token acceptance, complete-model TPS, EBPW, or resident promotion.",
            "next": "Attach the persistent full-attention KV session at layer 3 and continue the same state arena through layer 47 before accepted-token accounting."
        });
        doc["seal_sha256"] = Value::String(sha256_bytes(&serde_json::to_vec(&doc)?));
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&doc)?)?;
        println!("{}", serde_json::to_string_pretty(&doc)?);
        Ok(())
    }

    /// Capture exact layer-2 outputs from the qualified stateful prefix for a
    /// following structural-species integration seam. These are diagnostic
    /// host copies, not a production activation handoff.
    fn capture_stateful_linear_prefix_outputs_mode(
        root: PathBuf,
        token_ids: &[usize],
        dense: bool,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        if token_ids.len() < 2 {
            return Err("stateful prefix capture requires at least two tokens".into());
        }
        let root = root.canonicalize()?;
        let (_config, _config_sha256, vocab) = validate_config(&root)?;
        let (index, context) = cached_flash_resources(&root)?;
        let (first_embedding, _sha, _bytes) = load_embedding_row(&index, vocab, token_ids[0])?;
        let mut expected_base = repeat_streams(&first_embedding);
        let mut layers = Vec::with_capacity(3);
        for layer in 0..3 {
            let session = if dense {
                StatefulLinearLayer::new_dense(&index, &context, layer, &expected_base)?
            } else {
                StatefulLinearLayer::new(&index, &context, layer, &expected_base)?
            };
            expected_base = source_layer_from_base(&session.weights, &expected_base).final_state;
            layers.push(session);
        }
        let mut outputs = Vec::with_capacity(token_ids.len());
        for (step, &token_id) in token_ids.iter().enumerate() {
            let (embedding, _sha, _bytes) = load_embedding_row(&index, vocab, token_id)?;
            let host_base = repeat_streams(&embedding);
            let mut prior_device: Option<PinnedBuffer> = None;
            let mut layer2_state = None;
            for (layer_index, layer) in layers.iter_mut().enumerate() {
                let (output, _gpu_ns, _wall_ns, _dispatches, final_state) = if layer_index == 0 {
                    layer.step(&context, Some(&host_base), None, step == 0)?
                } else {
                    let input = prior_device
                        .as_ref()
                        .ok_or("missing device inter-layer state")?;
                    layer.step(&context, None, Some(input), step == 0)?
                };
                if layer_index == 2 {
                    layer2_state = Some(final_state);
                }
                prior_device = Some(output);
            }
            outputs.push(layer2_state.ok_or("layer-2 state was not produced")?);
        }
        Ok(outputs)
    }

    pub(crate) fn capture_stateful_linear_prefix_outputs(
        root: PathBuf,
        token_ids: &[usize],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        capture_stateful_linear_prefix_outputs_mode(root, token_ids, false)
    }

    /// Dense-bank variant used for teacher capture.  The historical compact
    /// helper is deliberately retained for bounded route-specialized probes;
    /// a teacher surface must let every token choose its own expert IDs.
    pub(crate) fn capture_stateful_linear_prefix_outputs_dense(
        root: PathBuf,
        token_ids: &[usize],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        capture_stateful_linear_prefix_outputs_mode(root, token_ids, true)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn layer_name_rewrites_only_the_layer_index_segment() {
            assert_eq!(
                layer_tensor_name(2, QKV),
                "model.language_model.layers.2.linear_attn.in_proj_qkv.weight"
            );
            assert_eq!(layer_tensor_name(0, HC_ATTN_NORM), HC_ATTN_NORM);
            assert_eq!(layer_tensor_name(2, EMBEDDING), EMBEDDING);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
pub(crate) use macos::Args;
#[cfg(target_os = "macos")]
pub(crate) use macos::StatefulLinearLayer;
#[cfg(target_os = "macos")]
pub(crate) fn run_layer(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_layer(args)
}
#[cfg(target_os = "macos")]
pub(crate) fn run_layer_device_output(
    args: Args,
    input: Option<&hawking_core::metal::PinnedBuffer>,
) -> Result<Option<hawking_core::metal::PinnedBuffer>, Box<dyn std::error::Error>> {
    macos::run_layer_device_output(args, input)
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_token_probe(
    root: std::path::PathBuf,
    token_ids: &[usize],
    out: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_stateful_token_probe(root, token_ids, out)
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_linear_prefix_session(
    root: std::path::PathBuf,
    token_ids: &[usize],
    out: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_stateful_linear_prefix_session(root, token_ids, out)
}
#[cfg(target_os = "macos")]
pub(crate) fn run_stateful_linear_prefix_session_mode(
    root: std::path::PathBuf,
    token_ids: &[usize],
    out: std::path::PathBuf,
    dense: bool,
    union_from: Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    macos::run_stateful_linear_prefix_session_mode(root, token_ids, out, dense, union_from)
}
#[cfg(target_os = "macos")]
pub(crate) fn capture_stateful_linear_prefix_outputs(
    root: std::path::PathBuf,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    macos::capture_stateful_linear_prefix_outputs(root, token_ids)
}
#[cfg(target_os = "macos")]
pub(crate) fn capture_stateful_linear_prefix_outputs_dense(
    root: std::path::PathBuf,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    macos::capture_stateful_linear_prefix_outputs_dense(root, token_ids)
}
