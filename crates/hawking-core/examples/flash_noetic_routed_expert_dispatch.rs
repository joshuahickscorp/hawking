//! Bounded native Flash-Next Noetic routed-expert body dispatch.
//!
//! This executable consumes the native router-selection receipt and the
//! persisted source-independent body/kernel receipts for the selected
//! experts.  It dispatches one bounded Q4/G64 body window per selected expert
//! through the existing Metal matvec kernel, then performs the selected-weight
//! gather on the host.  The gather is deliberately reported as host-side
//! composition: this is not a fused expert, complete expert, token, TPS, or
//! EBPW measurement.  The `--gate-up-swiglu` mode consumes full selected-expert
//! bodies and exercises the existing native gate+up/SwiGLU kernel, while still
//! stopping before down projection and the complete token graph.  The
//! `--expert-composition` mode chains that native gate+up/SwiGLU output directly
//! into independently persisted down-projection bodies, proving only the
//! bounded selected-expert composition and never a complete Flash token graph.
//! `--shared-expert-composition` mode runs the complete layer-0 shared-expert
//! candidate path (gate/up/SwiGLU -> down -> scalar sigmoid gate) with all
//! intermediates retained on-device; it still stops before routed/MoE combine,
//! residuals, attention/state, tokens, TPS, or EBPW.
//! `--shared-residual-composition` extends that bounded shared-expert output
//! into the layer-0 MLP hyperconnection candidate: device-side stream
//! injection -> low-rank down/up -> block-inject gated residual mix.  It is a
//! source-layout candidate graph only; `hc_norm`, routed/MoE semantics,
//! complete layers, tokens, TPS, and EBPW remain outside the claim.
//! `--exact-hyperconnection-composition` closes the layer-0 candidate boundary:
//! exact HyperConnection read -> native top-10 routed expert bodies plus the
//! sigmoid-gated shared expert -> device-resident MoE weighted sum/add -> exact
//! HyperConnection write. Quantized candidate weights and the current router
//! selection parity boundary remain explicit in the receipt.

#![recursion_limit = "512"]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash Noetic routed dispatch requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::metal::{MetalContext, MetalDispatchTiming};
    use metal::Buffer;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.flash_noetic_routed_expert_dispatch_native.v1";
    const GATE_UP_SCHEMA: &str = "hawking.flash_noetic_routed_expert_gate_up_swiglu_native.v1";
    const COMPOSITION_SCHEMA: &str = "hawking.flash_noetic_routed_expert_composition_native.v1";
    const SHARED_EXPERT_SCHEMA: &str = "hawking.flash_noetic_shared_expert_composition_native.v1";
    const SHARED_RESIDUAL_SCHEMA: &str =
        "hawking.flash_noetic_shared_residual_hyperconnection_native.v1";
    const EXACT_HYPERCONNECTION_SCHEMA: &str =
        "hawking.flash_noetic_exact_hyperconnection_native.v1";
    const BODY_SCHEMA: &str = "hcli.agentos.flash_noetic_component_body.v1";
    const VECTOR_BODY_SCHEMA: &str = "hcli.agentos.flash_noetic_vector_body.v1";
    const KERNEL_SCHEMA: &str = "hawking.flash_noetic_q4_kernel_parity.v1";
    const ROUTER_SCHEMA: &str = "hawking.flash_noetic_router_selection_native.v1";
    const CAMPAIGN_SCHEMA: &str = "hcli.agentos.flash_noetic_component_campaign.v1";
    const NOMENCLATURE_VERSION: &str = "HAWKING_NOMENCLATURE_V1";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const TENSOR_NAME: &str = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const DOWN_TENSOR_NAME: &str = "model.language_model.layers.0.mlp.experts.down_proj";
    const GROUP_SIZE: usize = 64;
    const CODE_BYTES_PER_GROUP: usize = GROUP_SIZE / 2;
    const KERNEL_NAME: &str = "qwen_uniform_q4_group64_matvec";
    const GATE_UP_KERNEL_NAME: &str =
        "qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128";
    const REFERENCE_MULTIPLIER: usize = 71;
    const REFERENCE_MODULUS: usize = 509;
    const REFERENCE_OFFSET: f32 = 254.0;
    const DEFAULT_WARMUP: usize = 2;
    const DEFAULT_REPS: usize = 7;
    const OUTPUT_ERROR_TOLERANCE: f32 = 2.0e-3;
    const GATE_UP_BODY_ROWS: usize = 1280;
    const GATE_UP_ROWS: usize = 640;
    const DOWN_BODY_ROWS: usize = 2560;
    const DOWN_COLUMNS: usize = 640;
    const SHARED_GATE_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight";
    const SHARED_UP_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight";
    const SHARED_DOWN_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
    const SHARED_SCALAR_GATE_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp.shared_expert_gate.weight";
    const SHARED_SIGMOID_GATE_KERNEL_NAME: &str = "qwen_next_shared_expert_sigmoid_gate";
    const SHARED_HIDDEN: usize = 2560;
    const SHARED_INTERMEDIATE: usize = 640;
    const HYPER_STATE_HIDDEN: usize = 2560;
    const HYPER_STATE_STREAMS: usize = 4;
    const HYPER_STATE_ELEMENTS: usize = HYPER_STATE_HIDDEN * HYPER_STATE_STREAMS;
    const HYPER_LOWRANK: usize = 320;
    const HYPER_INPUT_DOWN_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight";
    const HYPER_INPUT_UP_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight";
    const HYPER_BLOCK_INJECT_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp_hyper_connection.block_inject_weight.weight";
    const HYPER_HC_NORM_TENSOR_NAME: &str =
        "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight";
    const HYPER_EXPAND_KERNEL_NAME: &str = "qwen_next_expand_shared_to_hyper_state";
    const HYPER_RESIDUAL_MIX_KERNEL_NAME: &str = "qwen_next_hyperconnection_residual_mix_candidate";
    const HYPER_NORM_KERNEL_NAME: &str = "qwen_next_hyperconnection_grouped_rmsnorm";
    const HYPER_SILU_SCALE_KERNEL_NAME: &str = "qwen_next_hyperconnection_silu_scale";
    const HYPER_READ_MIX_KERNEL_NAME: &str = "qwen_next_hyperconnection_read_mix";
    const HYPER_COMBINE_KERNEL_NAME: &str = "qwen_next_hyperconnection_combine";
    const MOE_WEIGHTED_SUM_KERNEL_NAME: &str = "qwen_next_moe_weighted_sum";
    const MOE_ADD_SHARED_KERNEL_NAME: &str = "qwen_next_moe_add_shared";
    const ROUTED_TOP_K: usize = 10;
    const ROUTED_EXPERT_COUNT: usize = 512;
    const MANIFEST_PATH: &str =
        "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json";

    struct Args {
        root: PathBuf,
        router_receipt: PathBuf,
        campaign_receipt: PathBuf,
        warmup: usize,
        reps: usize,
        out: PathBuf,
        gate_up_swiglu: bool,
        expert_composition: bool,
        shared_expert_composition: bool,
        shared_residual_composition: bool,
        exact_hyperconnection_composition: bool,
    }

    struct PackedBody {
        path: PathBuf,
        receipt_sha256: String,
        body_sha256: String,
        expert_index: usize,
        row_start: usize,
        rows: usize,
        columns: usize,
        code_bytes: usize,
        scale_bytes: usize,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    struct RawVectorBody {
        path: PathBuf,
        receipt_sha256: String,
        body_sha256: String,
        elements: usize,
        bytes: Vec<u8>,
    }

    struct LoadedBody {
        body: PackedBody,
        body_receipt: Value,
        body_receipt_path: PathBuf,
        kernel_receipt: Value,
        kernel_receipt_path: PathBuf,
        kernel_receipt_sha256: String,
        codes: Buffer,
        scales: Buffer,
        output: Buffer,
        expected: Vec<f32>,
    }

    struct LoadedGateUp {
        body: PackedBody,
        gate: PackedBody,
        up: PackedBody,
        body_receipt: Value,
        body_receipt_path: PathBuf,
        kernel_receipt: Value,
        kernel_receipt_path: PathBuf,
        kernel_receipt_sha256: String,
        gate_codes: Buffer,
        gate_scales: Buffer,
        up_codes: Buffer,
        up_scales: Buffer,
        output: Buffer,
        expected: Vec<f32>,
    }

    struct LoadedComposition {
        gate_up_body: PackedBody,
        gate: PackedBody,
        gate_up_body_receipt: Value,
        gate_up_body_receipt_path: PathBuf,
        gate_up_kernel_receipt: Value,
        gate_up_kernel_receipt_path: PathBuf,
        gate_up_kernel_receipt_sha256: String,
        down: PackedBody,
        down_body_receipt: Value,
        down_body_receipt_path: PathBuf,
        down_kernel_receipt: Value,
        down_kernel_receipt_path: PathBuf,
        down_kernel_receipt_sha256: String,
        gate_codes: Buffer,
        gate_scales: Buffer,
        up_codes: Buffer,
        up_scales: Buffer,
        gate_up_output: Buffer,
        down_codes: Buffer,
        down_scales: Buffer,
        down_output: Buffer,
        expected_gate_up: Vec<f32>,
        expected_down: Vec<f32>,
    }

    struct LoadedSharedExpertComposition {
        gate: PackedBody,
        up: PackedBody,
        down: PackedBody,
        scalar_gate: PackedBody,
        gate_receipt: Value,
        gate_receipt_path: PathBuf,
        gate_receipt_sha256: String,
        up_receipt: Value,
        up_receipt_path: PathBuf,
        up_receipt_sha256: String,
        down_receipt: Value,
        down_receipt_path: PathBuf,
        down_receipt_sha256: String,
        scalar_gate_receipt: Value,
        scalar_gate_receipt_path: PathBuf,
        scalar_gate_receipt_sha256: String,
        gate_codes: Buffer,
        gate_scales: Buffer,
        up_codes: Buffer,
        up_scales: Buffer,
        gate_up_output: Buffer,
        down_codes: Buffer,
        down_scales: Buffer,
        down_output: Buffer,
        scalar_gate_codes: Buffer,
        scalar_gate_scales: Buffer,
        scalar_gate_output: Buffer,
        gated_output: Buffer,
        expected_gate_up: Vec<f32>,
        expected_down: Vec<f32>,
        expected_scalar_gate: Vec<f32>,
        expected_gated_output: Vec<f32>,
    }

    struct RoutedExpertSpec {
        expert_index: usize,
        gate_up_body: PackedBody,
        gate: PackedBody,
        up: PackedBody,
        down: PackedBody,
        gate_up_receipt: Value,
        gate_up_receipt_path: PathBuf,
        gate_up_kernel_receipt: Value,
        gate_up_kernel_receipt_path: PathBuf,
        gate_up_kernel_receipt_sha256: String,
        down_receipt: Value,
        down_receipt_path: PathBuf,
        down_kernel_receipt: Value,
        down_kernel_receipt_path: PathBuf,
        down_kernel_receipt_sha256: String,
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn default_lake_root() -> PathBuf {
        if let Some(value) = env::var_os("HCLI_FLASH_NEXT_ROOT") {
            return PathBuf::from(value);
        }
        PathBuf::from(
            "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc",
        )
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
            root: default_lake_root(),
            router_receipt: repo
                .join("receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json"),
            campaign_receipt: repo
                .join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json"),
            warmup: DEFAULT_WARMUP,
            reps: DEFAULT_REPS,
            out: repo.join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json"),
            gate_up_swiglu: false,
            expert_composition: false,
            shared_expert_composition: false,
            shared_residual_composition: false,
            exact_hyperconnection_composition: false,
        };
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--root" => args.root = PathBuf::from(values.next().ok_or("missing --root")?),
                "--router-receipt" => {
                    args.router_receipt =
                        PathBuf::from(values.next().ok_or("missing --router-receipt")?)
                }
                "--campaign-receipt" => {
                    args.campaign_receipt =
                        PathBuf::from(values.next().ok_or("missing --campaign-receipt")?)
                }
                "--warmup" => args.warmup = parse_usize(values.next(), &flag)?,
                "--reps" => args.reps = parse_usize(values.next(), &flag)?,
                "--out" => args.out = PathBuf::from(values.next().ok_or("missing --out")?),
                "--gate-up-swiglu" => args.gate_up_swiglu = true,
                "--expert-composition" => args.expert_composition = true,
                "--shared-expert-composition" => args.shared_expert_composition = true,
                "--shared-residual-composition" => args.shared_residual_composition = true,
                "--exact-hyperconnection-composition" => {
                    args.exact_hyperconnection_composition = true
                }
                "--help" | "-h" => {
                    println!(
                        "usage: flash_noetic_routed_expert_dispatch [--root DIR] \
                         [--router-receipt FILE] [--campaign-receipt FILE] \
                         [--warmup N] [--reps N] [--out FILE] [--gate-up-swiglu] \
                         [--expert-composition] [--shared-expert-composition] \
                         [--shared-residual-composition] [--exact-hyperconnection-composition]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if args.warmup > 32 {
            return Err("--warmup must be <= 32".into());
        }
        if args.reps == 0 || args.reps > 128 {
            return Err("--reps must be in 1..=128".into());
        }
        if [
            args.gate_up_swiglu,
            args.expert_composition,
            args.shared_expert_composition,
            args.shared_residual_composition,
            args.exact_hyperconnection_composition,
        ]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
            > 1
        {
            return Err(
                "--gate-up-swiglu, --expert-composition, --shared-expert-composition, --shared-residual-composition, and --exact-hyperconnection-composition are mutually exclusive"
                    .into(),
            );
        }
        let default_out =
            repo.join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json");
        if args.out == default_out {
            if args.shared_residual_composition {
                args.out = repo.join(
                    "receipts/headless/FLASH_NOETIC_SHARED_RESIDUAL_HYPERCONNECTION_NATIVE.json",
                );
            } else if args.exact_hyperconnection_composition {
                args.out =
                    repo.join("receipts/headless/FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json");
            } else if args.shared_expert_composition {
                args.out = repo
                    .join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_COMPOSITION_NATIVE.json");
            } else if args.expert_composition {
                args.out = repo
                    .join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPOSITION_NATIVE.json");
            } else if args.gate_up_swiglu {
                args.out = repo.join(
                    "receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_GATE_UP_SWIGLU_NATIVE.json",
                );
            }
        }
        Ok(args)
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn read_json(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let canonical = path.canonicalize()?;
        let bytes = fs::read(canonical)?;
        let value = serde_json::from_slice(&bytes)?;
        Ok((value, sha256_bytes(&bytes)))
    }

    fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("receipt is missing string field {field}").into())
    }

    fn usize_field(value: &Value, field: &str) -> Result<usize, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| format!("receipt is missing integer field {field}").into())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn validate_manifest(root: &Path) -> Result<Value, Box<dyn Error>> {
        let manifest_path = Path::new(MANIFEST_PATH);
        let (manifest, digest) = read_json(manifest_path)?;
        if string_field(&manifest, "repo")? != REPO_ID
            || string_field(&manifest, "revision")? != PINNED_REVISION
            || string_field(&manifest, "resolved_sha")? != PINNED_REVISION
        {
            return Err("ModelLake manifest is not the exact pinned Flash target".into());
        }
        let manifest_root = manifest
            .get("path")
            .and_then(Value::as_str)
            .ok_or("ModelLake manifest has no specimen path")?;
        if Path::new(manifest_root).canonicalize()? != root.canonicalize()? {
            return Err("selected specimen root does not match the pinned manifest".into());
        }
        Ok(json!({
            "path": manifest_path,
            "sha256": digest,
            "repo": REPO_ID,
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "n_files": manifest.get("n_files"),
            "bytes": manifest.get("bytes"),
            "label": "[V]",
        }))
    }

    fn validate_router(
        path: &Path,
    ) -> Result<(Value, String, Vec<usize>, Vec<f32>), Box<dyn Error>> {
        let (receipt, digest) = read_json(path)?;
        if string_field(&receipt, "schema")? != ROUTER_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("router receipt is not a PASSED pinned native Noetic receipt".into());
        }
        if receipt
            .get("native_selection_execution_observed")
            .and_then(Value::as_bool)
            != Some(true)
            || receipt.get("promotion_allowed").and_then(Value::as_bool) != Some(false)
        {
            return Err("router receipt fails native-observation or promotion guards".into());
        }
        let loader = receipt
            .get("native_loader")
            .ok_or("router receipt has no native_loader")?;
        if loader
            .get("source_independent_execution")
            .and_then(Value::as_bool)
            != Some(true)
            || loader
                .get("source_tensor_read_for_execution")
                .and_then(Value::as_bool)
                != Some(false)
        {
            return Err("router receipt is not source-independent".into());
        }
        let selection = receipt
            .get("selection")
            .ok_or("router receipt has no selection")?;
        let ids = selection
            .get("expert_ids")
            .and_then(Value::as_array)
            .ok_or("router receipt has no selection expert ids")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .ok_or_else(|| "router receipt contains an invalid expert id".into())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let weights = selection
            .get("selected_weights")
            .and_then(Value::as_array)
            .ok_or("router receipt has no selected weights")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|number| number as f32)
                    .filter(|number| number.is_finite() && *number >= 0.0)
                    .ok_or_else(|| "router receipt contains an invalid selected weight".into())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        if ids.is_empty()
            || ids.len() != weights.len()
            || ids.len() > 64
            || ids.iter().any(|expert| *expert >= ROUTED_EXPERT_COUNT)
        {
            return Err("router receipt selection shape is outside bounded dispatch limits".into());
        }
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("router receipt contains duplicate selected experts".into());
        }
        Ok((receipt, digest, ids, weights))
    }

    fn validate_campaign(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let (campaign, digest) = read_json(path)?;
        if string_field(&campaign, "schema")? != CAMPAIGN_SCHEMA
            || string_field(&campaign, "status")? != "PASSED"
            || string_field(&campaign, "repo")? != REPO_ID
            || string_field(&campaign, "pinned_revision")? != PINNED_REVISION
            || string_field(&campaign, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("routed-expert campaign is not a PASSED pinned Noetic campaign".into());
        }
        if campaign
            .get("source_independent_execution")
            .and_then(Value::as_bool)
            != Some(true)
            || campaign
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || campaign.get("promotion_allowed").and_then(Value::as_bool) != Some(false)
        {
            return Err("routed-expert campaign fails source-independent guards".into());
        }
        Ok((campaign, digest))
    }

    fn validate_body_for(
        path: &Path,
        expected_tensor: &str,
        expected_shape: [usize; 3],
        max_rows: usize,
    ) -> Result<(PackedBody, Value), Box<dyn Error>> {
        let canonical_receipt = path.canonicalize()?;
        let (receipt, receipt_sha256) = read_json(&canonical_receipt)?;
        if string_field(&receipt, "schema")? != BODY_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("expert body receipt is not a PASSED pinned Noetic body".into());
        }
        if receipt.get("source_independent").and_then(Value::as_bool) != Some(true)
            || receipt
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("expert body receipt fails source-independent execution guards".into());
        }
        let source = receipt
            .get("source_block")
            .ok_or("expert body receipt has no source_block")?;
        if string_field(source, "tensor_name")? != expected_tensor
            || string_field(source, "dtype")?.to_uppercase() != "BF16"
        {
            return Err(format!(
                "expert body source block is not the pinned {expected_tensor} tensor"
            )
            .into());
        }
        let shape = source
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("expert body source block has no shape")?;
        if shape.len() != 3
            || shape
                .iter()
                .zip(expected_shape)
                .any(|(value, expected)| value.as_u64() != Some(expected as u64))
        {
            return Err(format!(
                "expert body source shape is not [{},{},{}]",
                expected_shape[0], expected_shape[1], expected_shape[2]
            )
            .into());
        }
        let expert_index = usize_field(source, "expert_index")?;
        let row_start = usize_field(source, "row_start")?;
        let rows = usize_field(source, "row_count")?;
        let columns = shape[2]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("expert body column dimension is invalid")?;
        if expert_index >= 512 || rows == 0 || rows > max_rows || row_start + rows > max_rows {
            return Err("expert body window is outside bounded dispatch limits".into());
        }
        if usize_field(source, "bytes")? != rows * columns * 2 {
            return Err("expert body source block byte count is invalid".into());
        }
        let body_record = receipt
            .get("body")
            .ok_or("expert body receipt has no body")?;
        let body_path = Path::new(string_field(body_record, "path")?).canonicalize()?;
        let body_bytes = fs::read(&body_path)?;
        let body_sha256 = sha256_bytes(&body_bytes);
        if body_sha256 != string_field(body_record, "sha256")? {
            return Err("expert body bytes do not match the body receipt hash".into());
        }
        let code_bytes = usize_field(body_record, "code_bytes")?;
        let scale_bytes = usize_field(body_record, "scale_bytes")?;
        let expected_code_bytes = rows * (columns / 2);
        let expected_scale_bytes = rows * (columns / GROUP_SIZE) * 2;
        if code_bytes != expected_code_bytes
            || scale_bytes != expected_scale_bytes
            || body_bytes.len() != code_bytes + scale_bytes
        {
            return Err("expert body bytes do not match the Q4/G64 window".into());
        }
        Ok((
            PackedBody {
                path: body_path,
                receipt_sha256,
                body_sha256,
                expert_index,
                row_start,
                rows,
                columns,
                code_bytes,
                scale_bytes,
                codes: body_bytes[..code_bytes].to_vec(),
                scales: body_bytes[code_bytes..].to_vec(),
            },
            receipt,
        ))
    }

    fn validate_body(path: &Path) -> Result<(PackedBody, Value), Box<dyn Error>> {
        validate_body_for(
            path,
            TENSOR_NAME,
            [512, GATE_UP_BODY_ROWS, 2560],
            GATE_UP_BODY_ROWS,
        )
    }

    fn validate_matrix_body_for(
        path: &Path,
        expected_tensor: &str,
        expected_shape: [usize; 2],
    ) -> Result<(PackedBody, Value), Box<dyn Error>> {
        let canonical_receipt = path.canonicalize()?;
        let (receipt, receipt_sha256) = read_json(&canonical_receipt)?;
        if string_field(&receipt, "schema")? != BODY_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("matrix body receipt is not a PASSED pinned Noetic body".into());
        }
        if receipt.get("source_independent").and_then(Value::as_bool) != Some(true)
            || receipt
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("matrix body receipt fails source-independent execution guards".into());
        }
        let source = receipt
            .get("source_block")
            .ok_or("matrix body receipt has no source_block")?;
        if string_field(source, "tensor_name")? != expected_tensor
            || string_field(source, "dtype")?.to_uppercase() != "BF16"
        {
            return Err(format!(
                "matrix body source block is not the pinned {expected_tensor} tensor"
            )
            .into());
        }
        let shape = source
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("matrix body source block has no shape")?;
        if shape.len() != 2
            || shape
                .iter()
                .zip(expected_shape)
                .any(|(value, expected)| value.as_u64() != Some(expected as u64))
        {
            return Err(format!(
                "matrix body source shape is not [{},{}]",
                expected_shape[0], expected_shape[1]
            )
            .into());
        }
        let row_start = usize_field(source, "row_start")?;
        let rows = usize_field(source, "row_count")?;
        let columns = expected_shape[1];
        if rows == 0
            || rows > expected_shape[0]
            || row_start > expected_shape[0].saturating_sub(rows)
        {
            return Err("matrix body window is outside the expected source shape".into());
        }
        if usize_field(source, "bytes")? != rows * columns * 2 {
            return Err("matrix body source block byte count is invalid".into());
        }
        let body_record = receipt
            .get("body")
            .ok_or("matrix body receipt has no body")?;
        let body_path = Path::new(string_field(body_record, "path")?).canonicalize()?;
        let body_bytes = fs::read(&body_path)?;
        let body_sha256 = sha256_bytes(&body_bytes);
        if body_sha256 != string_field(body_record, "sha256")? {
            return Err("matrix body bytes do not match the body receipt hash".into());
        }
        let code_bytes = usize_field(body_record, "code_bytes")?;
        let scale_bytes = usize_field(body_record, "scale_bytes")?;
        let expected_code_bytes = rows * (columns / 2);
        let expected_scale_bytes = rows * (columns / GROUP_SIZE) * 2;
        if code_bytes != expected_code_bytes
            || scale_bytes != expected_scale_bytes
            || body_bytes.len() != code_bytes + scale_bytes
        {
            return Err("matrix body bytes do not match the Q4/G64 window".into());
        }
        Ok((
            PackedBody {
                path: body_path,
                receipt_sha256,
                body_sha256,
                expert_index: 0,
                row_start,
                rows,
                columns,
                code_bytes,
                scale_bytes,
                codes: body_bytes[..code_bytes].to_vec(),
                scales: body_bytes[code_bytes..].to_vec(),
            },
            receipt,
        ))
    }

    fn validate_vector_body_for(
        path: &Path,
        expected_tensor: &str,
        expected_elements: usize,
    ) -> Result<(RawVectorBody, Value), Box<dyn Error>> {
        let canonical_receipt = path.canonicalize()?;
        let (receipt, receipt_sha256) = read_json(&canonical_receipt)?;
        if string_field(&receipt, "schema")? != VECTOR_BODY_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("vector body receipt is not a PASSED pinned Noetic body".into());
        }
        if receipt.get("source_independent").and_then(Value::as_bool) != Some(true)
            || receipt
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || receipt.get("exact_source_payload").and_then(Value::as_bool) != Some(true)
            || receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("vector body receipt fails exact source-payload guards".into());
        }
        let source = receipt
            .get("source_block")
            .ok_or("vector body receipt has no source_block")?;
        if string_field(source, "tensor_name")? != expected_tensor
            || string_field(source, "dtype")?.to_uppercase() != "BF16"
        {
            return Err(format!(
                "vector body source block is not the pinned {expected_tensor} BF16 tensor"
            )
            .into());
        }
        let shape = source
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("vector body source block has no shape")?;
        if shape.len() != 1 || shape[0].as_u64() != Some(expected_elements as u64) {
            return Err(format!("vector body source shape is not [{expected_elements}]").into());
        }
        let expected_bytes = expected_elements
            .checked_mul(2)
            .ok_or("vector body byte count overflow")?;
        if usize_field(source, "bytes")? != expected_bytes {
            return Err("vector body source payload byte count is invalid".into());
        }
        let body_record = receipt
            .get("body")
            .ok_or("vector body receipt has no body")?;
        if string_field(body_record, "dtype")?.to_uppercase() != "BF16"
            || usize_field(body_record, "elements")? != expected_elements
            || string_field(body_record, "format")?
                != "little-endian BF16 vector payload; one value per source element"
        {
            return Err("vector body storage format is not the exact BF16 vector contract".into());
        }
        let body_path = Path::new(string_field(body_record, "path")?).canonicalize()?;
        let body_bytes = fs::read(&body_path)?;
        let body_sha256 = sha256_bytes(&body_bytes);
        if body_sha256 != string_field(body_record, "sha256")?
            || body_bytes.len() != expected_bytes
            || usize_field(body_record, "bytes")? != expected_bytes
        {
            return Err("vector body bytes do not match the exact BF16 payload".into());
        }
        if string_field(source, "payload_sha256")? != body_sha256 {
            return Err("vector body does not preserve the source payload hash".into());
        }
        Ok((
            RawVectorBody {
                path: body_path,
                receipt_sha256,
                body_sha256,
                elements: expected_elements,
                bytes: body_bytes,
            },
            receipt,
        ))
    }

    fn validate_down_body(path: &Path) -> Result<(PackedBody, Value), Box<dyn Error>> {
        validate_body_for(
            path,
            DOWN_TENSOR_NAME,
            [512, DOWN_BODY_ROWS, DOWN_COLUMNS],
            DOWN_BODY_ROWS,
        )
    }

    fn split_gate_up(body: &PackedBody) -> Result<(PackedBody, PackedBody), Box<dyn Error>> {
        if body.row_start != 0
            || body.rows != GATE_UP_BODY_ROWS
            || GATE_UP_BODY_ROWS != GATE_UP_ROWS * 2
        {
            return Err("gate/up activation requires a full 1280-row fused body".into());
        }
        let row_code_bytes = body.columns / 2;
        let row_scale_bytes = (body.columns / GROUP_SIZE) * 2;
        let half_code_bytes = GATE_UP_ROWS * row_code_bytes;
        let half_scale_bytes = GATE_UP_ROWS * row_scale_bytes;
        if body.code_bytes != GATE_UP_BODY_ROWS * row_code_bytes
            || body.scale_bytes != GATE_UP_BODY_ROWS * row_scale_bytes
            || body.codes.len() != body.code_bytes
            || body.scales.len() != body.scale_bytes
        {
            return Err("full gate/up body has an invalid packed layout".into());
        }
        let make_half = |codes: Vec<u8>, scales: Vec<u8>| PackedBody {
            path: body.path.clone(),
            receipt_sha256: body.receipt_sha256.clone(),
            body_sha256: body.body_sha256.clone(),
            expert_index: body.expert_index,
            row_start: 0,
            rows: GATE_UP_ROWS,
            columns: body.columns,
            code_bytes: half_code_bytes,
            scale_bytes: half_scale_bytes,
            codes,
            scales,
        };
        Ok((
            make_half(
                body.codes[..half_code_bytes].to_vec(),
                body.scales[..half_scale_bytes].to_vec(),
            ),
            make_half(
                body.codes[half_code_bytes..].to_vec(),
                body.scales[half_scale_bytes..].to_vec(),
            ),
        ))
    }

    fn validate_kernel_for(
        path: &Path,
        body: &PackedBody,
        expected_tensor: &str,
    ) -> Result<(Value, String), Box<dyn Error>> {
        let (receipt, digest) = read_json(path)?;
        if string_field(&receipt, "schema")? != KERNEL_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("expert kernel receipt is not a PASSED pinned parity receipt".into());
        }
        if receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("expert kernel receipt fails mutation/model-load guards".into());
        }
        let source = receipt
            .get("source_tensor")
            .ok_or("expert kernel receipt has no source_tensor")?;
        if string_field(source, "tensor_name")? != expected_tensor
            || usize_field(source, "selected_expert")? != body.expert_index
            || usize_field(source, "selected_row_start")? != body.row_start
            || usize_field(source, "selected_row_count")? != body.rows
            || usize_field(source, "selected_block_bytes")? != body.rows * body.columns * 2
        {
            return Err(format!(
                "expert kernel source window does not match the {expected_tensor} body receipt"
            )
            .into());
        }
        let loader = receipt
            .get("native_loader")
            .ok_or("expert kernel receipt has no native_loader")?;
        if string_field(loader, "status")? != "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD"
            || loader
                .get("source_independent_execution")
                .and_then(Value::as_bool)
                != Some(true)
            || loader
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err("expert kernel receipt does not prove persisted-body execution".into());
        }
        let native_kernel = receipt
            .get("native_kernel")
            .ok_or("expert kernel receipt has no native_kernel")?;
        if native_kernel
            .get("kernel_registered")
            .and_then(Value::as_bool)
            != Some(true)
            || string_field(native_kernel, "kernel")? != KERNEL_NAME
        {
            return Err("expert kernel receipt does not identify the expected Metal kernel".into());
        }
        if receipt
            .get("parity")
            .and_then(|value| value.get("within_tolerance"))
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err("expert kernel parity receipt is not within tolerance".into());
        }
        let candidate = receipt
            .get("candidate_body")
            .ok_or("expert kernel receipt has no candidate_body")?;
        if Path::new(string_field(candidate, "path")?).canonicalize()? != body.path
            || string_field(candidate, "sha256")? != body.body_sha256
            || usize_field(candidate, "bytes")? != body.code_bytes + body.scale_bytes
        {
            return Err("expert kernel candidate body does not match the persisted body".into());
        }
        Ok((receipt, digest))
    }

    fn validate_kernel(path: &Path, body: &PackedBody) -> Result<(Value, String), Box<dyn Error>> {
        validate_kernel_for(path, body, TENSOR_NAME)
    }

    fn validate_down_kernel(
        path: &Path,
        body: &PackedBody,
    ) -> Result<(Value, String), Box<dyn Error>> {
        validate_kernel_for(path, body, DOWN_TENSOR_NAME)
    }

    fn validate_routed_expert_specs(
        repo: &Path,
        selected_ids: &[usize],
    ) -> Result<Vec<RoutedExpertSpec>, Box<dyn Error>> {
        if selected_ids.len() != ROUTED_TOP_K {
            return Err(format!(
                "exact layer-0 Flash MoE requires exactly {ROUTED_TOP_K} selected routed experts"
            )
            .into());
        }
        selected_ids
            .iter()
            .copied()
            .map(|expert_index| {
                let gate_up_receipt_path = repo.join(format!(
                    "receipts/headless/FLASH_NOETIC_GATE_UP_BODY_E{expert_index}_R0_1280.json"
                ));
                let gate_up_kernel_receipt_path = repo.join(format!(
                    "receipts/headless/FLASH_NOETIC_GATE_UP_KERNEL_E{expert_index}_R0_1280_PARITY.json"
                ));
                let down_receipt_path = repo.join(format!(
                    "receipts/headless/FLASH_NOETIC_DOWN_BODY_E{expert_index}_R0_2560.json"
                ));
                let down_kernel_receipt_path = repo.join(format!(
                    "receipts/headless/FLASH_NOETIC_DOWN_KERNEL_E{expert_index}_R0_2560_PARITY.json"
                ));
                let (gate_up_body, gate_up_receipt) = validate_body(&gate_up_receipt_path)?;
                if gate_up_body.expert_index != expert_index
                    || gate_up_body.row_start != 0
                    || gate_up_body.rows != GATE_UP_BODY_ROWS
                {
                    return Err(format!(
                        "routed gate/up body for expert {expert_index} is not the full 1280-row body"
                    )
                    .into());
                }
                let (gate, up) = split_gate_up(&gate_up_body)?;
                let (gate_up_kernel_receipt, gate_up_kernel_receipt_sha256) =
                    validate_kernel(&gate_up_kernel_receipt_path, &gate_up_body)?;
                let (down, down_receipt) = validate_down_body(&down_receipt_path)?;
                if down.expert_index != expert_index
                    || down.row_start != 0
                    || down.rows != DOWN_BODY_ROWS
                {
                    return Err(format!(
                        "routed down body for expert {expert_index} is not the full 2560-row body"
                    )
                    .into());
                }
                let (down_kernel_receipt, down_kernel_receipt_sha256) =
                    validate_down_kernel(&down_kernel_receipt_path, &down)?;
                Ok(RoutedExpertSpec {
                    expert_index,
                    gate_up_body,
                    gate,
                    up,
                    down,
                    gate_up_receipt,
                    gate_up_receipt_path,
                    gate_up_kernel_receipt,
                    gate_up_kernel_receipt_path,
                    gate_up_kernel_receipt_sha256,
                    down_receipt,
                    down_receipt_path,
                    down_kernel_receipt,
                    down_kernel_receipt_path,
                    down_kernel_receipt_sha256,
                })
            })
            .collect()
    }

    fn f32_bytes_for_hash(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn deterministic_input(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| {
                ((index * REFERENCE_MULTIPLIER % REFERENCE_MODULUS) as f32 - REFERENCE_OFFSET)
                    / REFERENCE_MODULUS as f32
            })
            .collect()
    }

    fn cpu_matvec(body: &PackedBody, input: &[f32]) -> Vec<f32> {
        let groups_per_row = body.columns / GROUP_SIZE;
        let mut output = vec![0.0f32; body.rows];
        for row in 0..body.rows {
            for group in 0..groups_per_row {
                let group_base = row * groups_per_row + group;
                let scale_offset = group_base * 2;
                let scale = f16::from_bits(u16::from_le_bytes([
                    body.scales[scale_offset],
                    body.scales[scale_offset + 1],
                ]))
                .to_f32();
                let code_base = group_base * CODE_BYTES_PER_GROUP;
                for local in 0..GROUP_SIZE {
                    let byte = body.codes[code_base + local / 2];
                    let nibble = if local % 2 == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    output[row] +=
                        (nibble as i32 - 8) as f32 * scale * input[group * GROUP_SIZE + local];
                }
            }
        }
        output
    }

    fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
        let pointer = buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec()
    }

    fn dispatch(
        context: &MetalContext,
        body: &PackedBody,
        codes: &Buffer,
        scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        dispatch_with_output_offset(context, body, codes, scales, input, output, 0)
    }

    fn dispatch_with_output_offset(
        context: &MetalContext,
        body: &PackedBody,
        codes: &Buffer,
        scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
        output_offset: u64,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let rows = body.rows as u32;
        let columns = body.columns as u32;
        let groups = (body.columns / GROUP_SIZE) as u32;
        Ok(
            context.dispatch_threads_timed(KERNEL_NAME, (rows, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(codes), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), output_offset);
                encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(5, 4, &columns as *const u32 as *const _);
                encoder.set_bytes(6, 4, &groups as *const u32 as *const _);
            })?,
        )
    }

    fn dispatch_gate_up_swiglu(
        context: &MetalContext,
        gate: &PackedBody,
        gate_codes: &Buffer,
        gate_scales: &Buffer,
        up_codes: &Buffer,
        up_scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let rows = gate.rows as u32;
        let columns = gate.columns as u32;
        let groups = (gate.columns / GROUP_SIZE) as u32;
        let threadgroup = 128u32;
        let grid = rows
            .div_ceil(2)
            .saturating_mul(threadgroup)
            .max(threadgroup);
        Ok(context.dispatch_threads_timed(
            GATE_UP_KERNEL_NAME,
            (grid, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate_codes), 0);
                encoder.set_buffer(1, Some(gate_scales), 0);
                encoder.set_buffer(2, Some(up_codes), 0);
                encoder.set_buffer(3, Some(up_scales), 0);
                encoder.set_buffer(4, Some(input), 0);
                encoder.set_buffer(5, Some(output), 0);
                encoder.set_bytes(6, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(7, 4, &columns as *const u32 as *const _);
                encoder.set_bytes(8, 4, &groups as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_moe_weighted_sum(
        context: &MetalContext,
        routed_outputs: &Buffer,
        selected_weights: &Buffer,
        output: &Buffer,
        expert_count: usize,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let hidden = SHARED_HIDDEN as u32;
        let experts = u32::try_from(expert_count)?;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            MOE_WEIGHTED_SUM_KERNEL_NAME,
            (hidden, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(routed_outputs), 0);
                encoder.set_buffer(1, Some(selected_weights), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &experts as *const u32 as *const _);
                encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_moe_add_shared(
        context: &MetalContext,
        routed_output: &Buffer,
        shared_output: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let elements = SHARED_HIDDEN as u32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            MOE_ADD_SHARED_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(routed_output), 0);
                encoder.set_buffer(1, Some(shared_output), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &elements as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_shared_expert_sigmoid_gate(
        context: &MetalContext,
        shared_output: &Buffer,
        gate_logit: &Buffer,
        gated_output: &Buffer,
        elements: usize,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let elements = u32::try_from(elements)?;
        let threadgroup = 64u32;
        Ok(context.dispatch_threads_timed(
            SHARED_SIGMOID_GATE_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(shared_output), 0);
                encoder.set_buffer(1, Some(gate_logit), 0);
                encoder.set_buffer(2, Some(gated_output), 0);
                encoder.set_bytes(3, 4, &elements as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_expand_shared_to_hyper_state(
        context: &MetalContext,
        base_state: &Buffer,
        shared_output: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let hidden = HYPER_STATE_HIDDEN as u32;
        let streams = HYPER_STATE_STREAMS as u32;
        let injected_stream = 1u32;
        let elements = HYPER_STATE_ELEMENTS as u32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            HYPER_EXPAND_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(base_state), 0);
                encoder.set_buffer(1, Some(shared_output), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(4, 4, &streams as *const u32 as *const _);
                encoder.set_bytes(5, 4, &injected_stream as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_hyperconnection_residual_mix(
        context: &MetalContext,
        state: &Buffer,
        correction: &Buffer,
        block_logits: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let hidden = HYPER_STATE_HIDDEN as u32;
        let streams = HYPER_STATE_STREAMS as u32;
        let elements = HYPER_STATE_ELEMENTS as u32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            HYPER_RESIDUAL_MIX_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(state), 0);
                encoder.set_buffer(1, Some(correction), 0);
                encoder.set_buffer(2, Some(block_logits), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(5, 4, &streams as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_hyperconnection_grouped_rmsnorm(
        context: &MetalContext,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let hidden = HYPER_STATE_HIDDEN as u32;
        let streams = HYPER_STATE_STREAMS as u32;
        let elements = HYPER_STATE_ELEMENTS as u32;
        let eps = 1.0e-6f32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            HYPER_NORM_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(weight), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(4, 4, &streams as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            },
        )?)
    }

    fn dispatch_hyperconnection_silu_scale(
        context: &MetalContext,
        input: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let elements = HYPER_LOWRANK as u32;
        let divisor = HYPER_STATE_STREAMS as f32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            HYPER_SILU_SCALE_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(2, 4, &elements as *const u32 as *const _);
                encoder.set_bytes(3, 4, &divisor as *const f32 as *const _);
            },
        )?)
    }

    fn dispatch_hyperconnection_read_mix(
        context: &MetalContext,
        normalized: &Buffer,
        gate_logits: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let hidden = HYPER_STATE_HIDDEN as u32;
        let streams = HYPER_STATE_STREAMS as u32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            HYPER_READ_MIX_KERNEL_NAME,
            (hidden, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(normalized), 0);
                encoder.set_buffer(1, Some(gate_logits), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(4, 4, &streams as *const u32 as *const _);
            },
        )?)
    }

    fn dispatch_hyperconnection_combine(
        context: &MetalContext,
        residual: &Buffer,
        block_output: &Buffer,
        block_logits: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let hidden = HYPER_STATE_HIDDEN as u32;
        let streams = HYPER_STATE_STREAMS as u32;
        let divisor = HYPER_STATE_STREAMS as f32;
        let elements = HYPER_STATE_ELEMENTS as u32;
        let threadgroup = 128u32;
        Ok(context.dispatch_threads_timed(
            HYPER_COMBINE_KERNEL_NAME,
            (elements, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(residual), 0);
                encoder.set_buffer(1, Some(block_output), 0);
                encoder.set_buffer(2, Some(block_logits), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(5, 4, &streams as *const u32 as *const _);
                encoder.set_bytes(6, 4, &divisor as *const f32 as *const _);
            },
        )?)
    }

    fn cpu_gate_up_swiglu(gate: &PackedBody, up: &PackedBody, input: &[f32]) -> Vec<f32> {
        let gate_values = cpu_matvec(gate, input);
        let up_values = cpu_matvec(up, input);
        gate_values
            .into_iter()
            .zip(up_values)
            .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect()
    }

    fn sigmoid(value: f32) -> f32 {
        1.0f32 / (1.0f32 + (-value).exp())
    }

    fn silu(value: f32) -> f32 {
        value / (1.0f32 + (-value).exp())
    }

    fn raw_bf16_values(body: &RawVectorBody) -> Vec<f32> {
        body.bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                f32::from_bits(bits << 16)
            })
            .collect()
    }

    fn cpu_grouped_rmsnorm(
        input: &[f32],
        weight: &RawVectorBody,
        hidden: usize,
        streams: usize,
        eps: f32,
    ) -> Vec<f32> {
        let weights = raw_bf16_values(weight);
        let mut output = vec![0.0f32; input.len()];
        for stream in 0..streams {
            let start = stream * hidden;
            let sum = input[start..start + hidden]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let inverse_rms = (sum / hidden as f32 + eps).sqrt().recip();
            for index in 0..hidden {
                let offset = start + index;
                output[offset] = input[offset] * inverse_rms * (1.0 + weights[offset]);
            }
        }
        output
    }

    fn cpu_hyper_read_mix(
        normalized: &[f32],
        input_down: &PackedBody,
        input_up: &PackedBody,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let low_rank_down = cpu_matvec(input_down, normalized);
        let scaled_silu = low_rank_down
            .iter()
            .map(|value| silu(*value / HYPER_STATE_STREAMS as f32))
            .collect::<Vec<_>>();
        let gate_logits = cpu_matvec(input_up, &scaled_silu);
        let mut mixed = vec![0.0f32; HYPER_STATE_HIDDEN];
        for hidden in 0..HYPER_STATE_HIDDEN {
            let mut sum = 0.0f32;
            for stream in 0..HYPER_STATE_STREAMS {
                let offset = stream * HYPER_STATE_HIDDEN + hidden;
                sum += sigmoid(gate_logits[offset]) * normalized[offset];
            }
            mixed[hidden] = sum / HYPER_STATE_STREAMS as f32;
        }
        (low_rank_down, scaled_silu, gate_logits, mixed)
    }

    fn cpu_hyper_combine(residual: &[f32], block_output: &[f32], block_logits: &[f32]) -> Vec<f32> {
        residual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let stream = index / HYPER_STATE_HIDDEN;
                *value
                    + block_output[index % HYPER_STATE_HIDDEN]
                        * (2.0 * sigmoid(block_logits[stream] / HYPER_STATE_STREAMS as f32))
            })
            .collect()
    }

    fn run_exact_hyperconnection_composition(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let (campaign, campaign_sha256) = validate_campaign(&args.campaign_receipt)?;
        let selected_weight_sum = selected_weights.iter().copied().sum::<f32>();
        if !selected_weight_sum.is_finite() || (selected_weight_sum - 1.0).abs() > 2.0e-3 {
            return Err(format!(
                "exact layer-0 Flash MoE selected weights are not normalized: {selected_weight_sum}"
            )
            .into());
        }
        let routed_specs = validate_routed_expert_specs(&repo, &selected_ids)?;

        let hc_norm_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_HC_NORM_BODY_L0.json");
        let (hc_norm, hc_norm_receipt) = validate_vector_body_for(
            &hc_norm_receipt_path,
            HYPER_HC_NORM_TENSOR_NAME,
            HYPER_STATE_ELEMENTS,
        )?;
        let input_down_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_INPUT_DOWN_BODY_L0_R0_320.json");
        let input_up_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_INPUT_UP_BODY_L0_R0_10240.json");
        let block_inject_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_BLOCK_INJECT_BODY_L0_R0_4.json");
        let gate_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_GATE_BODY_L0_R0_640.json");
        let up_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_UP_BODY_L0_R0_640.json");
        let down_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_DOWN_BODY_L0_R0_2560.json");
        let scalar_gate_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_SCALAR_GATE_BODY_L0_R0_1.json");
        let (input_down, input_down_receipt) = validate_matrix_body_for(
            &input_down_receipt_path,
            HYPER_INPUT_DOWN_TENSOR_NAME,
            [HYPER_LOWRANK, HYPER_STATE_ELEMENTS],
        )?;
        let (input_up, input_up_receipt) = validate_matrix_body_for(
            &input_up_receipt_path,
            HYPER_INPUT_UP_TENSOR_NAME,
            [HYPER_STATE_ELEMENTS, HYPER_LOWRANK],
        )?;
        let (block_inject, block_inject_receipt) = validate_matrix_body_for(
            &block_inject_receipt_path,
            HYPER_BLOCK_INJECT_TENSOR_NAME,
            [HYPER_STATE_STREAMS, HYPER_STATE_ELEMENTS],
        )?;
        let (gate, gate_receipt) = validate_matrix_body_for(
            &gate_receipt_path,
            SHARED_GATE_TENSOR_NAME,
            [SHARED_INTERMEDIATE, SHARED_HIDDEN],
        )?;
        let (up, up_receipt) = validate_matrix_body_for(
            &up_receipt_path,
            SHARED_UP_TENSOR_NAME,
            [SHARED_INTERMEDIATE, SHARED_HIDDEN],
        )?;
        let (down, down_receipt) = validate_matrix_body_for(
            &down_receipt_path,
            SHARED_DOWN_TENSOR_NAME,
            [SHARED_HIDDEN, SHARED_INTERMEDIATE],
        )?;
        let (scalar_gate, scalar_gate_receipt) = validate_matrix_body_for(
            &scalar_gate_receipt_path,
            SHARED_SCALAR_GATE_TENSOR_NAME,
            [1, SHARED_HIDDEN],
        )?;
        if input_down.row_start != 0
            || input_down.rows != HYPER_LOWRANK
            || input_up.row_start != 0
            || input_up.rows != HYPER_STATE_ELEMENTS
            || block_inject.row_start != 0
            || block_inject.rows != HYPER_STATE_STREAMS
            || gate.row_start != 0
            || gate.rows != SHARED_INTERMEDIATE
            || up.row_start != 0
            || up.rows != SHARED_INTERMEDIATE
            || down.row_start != 0
            || down.rows != SHARED_HIDDEN
            || scalar_gate.row_start != 0
            || scalar_gate.rows != 1
        {
            return Err("exact hyperconnection bodies are not complete layer-0 windows".into());
        }

        let base_state = deterministic_input(HYPER_STATE_ELEMENTS);
        let base_state_sha256 = sha256_bytes(&f32_bytes(&base_state));
        let expected_normalized = cpu_grouped_rmsnorm(
            &base_state,
            &hc_norm,
            HYPER_STATE_HIDDEN,
            HYPER_STATE_STREAMS,
            1.0e-6,
        );
        let (
            expected_low_rank_down,
            expected_read_activation,
            expected_read_gate_logits,
            expected_mixed_input,
        ) = cpu_hyper_read_mix(&expected_normalized, &input_down, &input_up);
        let expected_gate_up = cpu_gate_up_swiglu(&gate, &up, &expected_mixed_input);
        let expected_shared_down = cpu_matvec(&down, &expected_gate_up);
        let expected_scalar_gate = cpu_matvec(&scalar_gate, &expected_mixed_input);
        if expected_scalar_gate.len() != 1 {
            return Err("exact hyperconnection shared gate did not produce one logit".into());
        }
        let expected_block_output: Vec<f32> = expected_shared_down
            .iter()
            .map(|value| value * sigmoid(expected_scalar_gate[0]))
            .collect();
        let expected_routed_gate_up: Vec<Vec<f32>> = routed_specs
            .iter()
            .map(|spec| cpu_gate_up_swiglu(&spec.gate, &spec.up, &expected_mixed_input))
            .collect();
        let expected_routed_outputs: Vec<Vec<f32>> = routed_specs
            .iter()
            .zip(&expected_routed_gate_up)
            .map(|(spec, gate_up)| cpu_matvec(&spec.down, gate_up))
            .collect();
        let mut expected_routed_mix = vec![0.0f32; SHARED_HIDDEN];
        for (expert, output) in expected_routed_outputs.iter().enumerate() {
            for (index, value) in output.iter().copied().enumerate() {
                expected_routed_mix[index] += selected_weights[expert] * value;
            }
        }
        let expected_moe_output: Vec<f32> = expected_block_output
            .iter()
            .zip(&expected_routed_mix)
            .map(|(shared, routed)| shared + routed)
            .collect();
        let expected_block_logits = cpu_matvec(&block_inject, &expected_normalized);
        if expected_block_logits.len() != HYPER_STATE_STREAMS {
            return Err("exact hyperconnection block inject did not produce four logits".into());
        }
        let expected_combined =
            cpu_hyper_combine(&base_state, &expected_moe_output, &expected_block_logits);

        let context = MetalContext::new_with_trace(true)?;
        let device_memory = context.device_memory_limits();
        let base_state_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&base_state))?;
        let hc_norm_buffer = context.new_buffer_with_bytes_checked(&hc_norm.bytes)?;
        let normalized_buffer =
            context.new_buffer_checked(HYPER_STATE_ELEMENTS * std::mem::size_of::<f32>())?;
        let input_down_codes = context.new_buffer_with_bytes_checked(&input_down.codes)?;
        let input_down_scales = context.new_buffer_with_bytes_checked(&input_down.scales)?;
        let low_rank_down_buffer =
            context.new_buffer_checked(HYPER_LOWRANK * std::mem::size_of::<f32>())?;
        let read_activation_buffer =
            context.new_buffer_checked(HYPER_LOWRANK * std::mem::size_of::<f32>())?;
        let input_up_codes = context.new_buffer_with_bytes_checked(&input_up.codes)?;
        let input_up_scales = context.new_buffer_with_bytes_checked(&input_up.scales)?;
        let read_gate_logits_buffer =
            context.new_buffer_checked(HYPER_STATE_ELEMENTS * std::mem::size_of::<f32>())?;
        let mixed_input_buffer =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;
        let gate_codes = context.new_buffer_with_bytes_checked(&gate.codes)?;
        let gate_scales = context.new_buffer_with_bytes_checked(&gate.scales)?;
        let up_codes = context.new_buffer_with_bytes_checked(&up.codes)?;
        let up_scales = context.new_buffer_with_bytes_checked(&up.scales)?;
        let gate_up_buffer =
            context.new_buffer_checked(SHARED_INTERMEDIATE * std::mem::size_of::<f32>())?;
        let down_codes = context.new_buffer_with_bytes_checked(&down.codes)?;
        let down_scales = context.new_buffer_with_bytes_checked(&down.scales)?;
        let shared_down_buffer =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;
        let scalar_gate_codes = context.new_buffer_with_bytes_checked(&scalar_gate.codes)?;
        let scalar_gate_scales = context.new_buffer_with_bytes_checked(&scalar_gate.scales)?;
        let scalar_gate_buffer = context.new_buffer_checked(std::mem::size_of::<f32>())?;
        let block_output_buffer =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;
        let block_inject_codes = context.new_buffer_with_bytes_checked(&block_inject.codes)?;
        let block_inject_scales = context.new_buffer_with_bytes_checked(&block_inject.scales)?;
        let block_logits_buffer =
            context.new_buffer_checked(HYPER_STATE_STREAMS * std::mem::size_of::<f32>())?;
        let combined_buffer =
            context.new_buffer_checked(HYPER_STATE_ELEMENTS * std::mem::size_of::<f32>())?;
        let selected_weights_buffer =
            context.new_buffer_with_bytes_checked(&f32_bytes(&selected_weights))?;
        let routed_gate_codes = routed_specs
            .iter()
            .map(|spec| context.new_buffer_with_bytes_checked(&spec.gate.codes))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_gate_scales = routed_specs
            .iter()
            .map(|spec| context.new_buffer_with_bytes_checked(&spec.gate.scales))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_up_codes = routed_specs
            .iter()
            .map(|spec| context.new_buffer_with_bytes_checked(&spec.up.codes))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_up_scales = routed_specs
            .iter()
            .map(|spec| context.new_buffer_with_bytes_checked(&spec.up.scales))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_down_codes = routed_specs
            .iter()
            .map(|spec| context.new_buffer_with_bytes_checked(&spec.down.codes))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_down_scales = routed_specs
            .iter()
            .map(|spec| context.new_buffer_with_bytes_checked(&spec.down.scales))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_gate_up_buffers = (0..routed_specs.len())
            .map(|_| context.new_buffer_checked(SHARED_INTERMEDIATE * std::mem::size_of::<f32>()))
            .collect::<Result<Vec<_>, _>>()?;
        let routed_outputs_buffer = context
            .new_buffer_checked(ROUTED_TOP_K * SHARED_HIDDEN * std::mem::size_of::<f32>())?;
        let routed_mix_buffer =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;
        let moe_output_buffer =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;

        let dispatch_graph = || -> Result<Vec<MetalDispatchTiming>, Box<dyn Error>> {
            let mut timings = Vec::with_capacity(33);
            timings.push(dispatch_hyperconnection_grouped_rmsnorm(
                &context,
                &base_state_buffer,
                &hc_norm_buffer,
                &normalized_buffer,
            )?);
            timings.push(dispatch(
                &context,
                &input_down,
                &input_down_codes,
                &input_down_scales,
                &normalized_buffer,
                &low_rank_down_buffer,
            )?);
            timings.push(dispatch_hyperconnection_silu_scale(
                &context,
                &low_rank_down_buffer,
                &read_activation_buffer,
            )?);
            timings.push(dispatch(
                &context,
                &input_up,
                &input_up_codes,
                &input_up_scales,
                &read_activation_buffer,
                &read_gate_logits_buffer,
            )?);
            timings.push(dispatch_hyperconnection_read_mix(
                &context,
                &normalized_buffer,
                &read_gate_logits_buffer,
                &mixed_input_buffer,
            )?);
            timings.push(dispatch_gate_up_swiglu(
                &context,
                &gate,
                &gate_codes,
                &gate_scales,
                &up_codes,
                &up_scales,
                &mixed_input_buffer,
                &gate_up_buffer,
            )?);
            timings.push(dispatch(
                &context,
                &down,
                &down_codes,
                &down_scales,
                &gate_up_buffer,
                &shared_down_buffer,
            )?);
            timings.push(dispatch(
                &context,
                &scalar_gate,
                &scalar_gate_codes,
                &scalar_gate_scales,
                &mixed_input_buffer,
                &scalar_gate_buffer,
            )?);
            timings.push(dispatch_shared_expert_sigmoid_gate(
                &context,
                &shared_down_buffer,
                &scalar_gate_buffer,
                &block_output_buffer,
                SHARED_HIDDEN,
            )?);
            for (index, spec) in routed_specs.iter().enumerate() {
                timings.push(dispatch_gate_up_swiglu(
                    &context,
                    &spec.gate,
                    &routed_gate_codes[index],
                    &routed_gate_scales[index],
                    &routed_up_codes[index],
                    &routed_up_scales[index],
                    &mixed_input_buffer,
                    &routed_gate_up_buffers[index],
                )?);
                let output_offset = (index * SHARED_HIDDEN * std::mem::size_of::<f32>()) as u64;
                timings.push(dispatch_with_output_offset(
                    &context,
                    &spec.down,
                    &routed_down_codes[index],
                    &routed_down_scales[index],
                    &routed_gate_up_buffers[index],
                    &routed_outputs_buffer,
                    output_offset,
                )?);
            }
            timings.push(dispatch_moe_weighted_sum(
                &context,
                &routed_outputs_buffer,
                &selected_weights_buffer,
                &routed_mix_buffer,
                routed_specs.len(),
            )?);
            timings.push(dispatch_moe_add_shared(
                &context,
                &routed_mix_buffer,
                &block_output_buffer,
                &moe_output_buffer,
            )?);
            timings.push(dispatch(
                &context,
                &block_inject,
                &block_inject_codes,
                &block_inject_scales,
                &normalized_buffer,
                &block_logits_buffer,
            )?);
            timings.push(dispatch_hyperconnection_combine(
                &context,
                &base_state_buffer,
                &moe_output_buffer,
                &block_logits_buffer,
                &combined_buffer,
            )?);
            Ok(timings)
        };
        let sum_gpu_ns = |timings: &[MetalDispatchTiming]| -> Result<u64, Box<dyn Error>> {
            timings.iter().try_fold(0u64, |total, timing| {
                Ok::<u64, Box<dyn Error>>(total.saturating_add(gpu_ns(*timing)?))
            })
        };
        let sum_host_ns = |timings: &[MetalDispatchTiming]| -> u64 {
            timings
                .iter()
                .map(|timing| timing.host_wall_us.saturating_mul(1000))
                .fold(0u64, u64::saturating_add)
        };
        let mut warmup_graph_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            warmup_graph_gpu_ns.push(sum_gpu_ns(&dispatch_graph()?)?);
        }
        let measured_stage_count = 9 + routed_specs.len() * 2 + 4;
        let mut stage_gpu_ns: Vec<Vec<u64>> = (0..measured_stage_count)
            .map(|_| Vec::with_capacity(args.reps))
            .collect();
        let mut graph_gpu_ns = Vec::with_capacity(args.reps);
        let mut graph_host_ns = Vec::with_capacity(args.reps);
        let mut stage_hashes: Vec<Vec<String>> = (0..measured_stage_count)
            .map(|_| Vec::with_capacity(args.reps))
            .collect();
        let mut parity: Vec<Value> = (0..measured_stage_count).map(|_| json!({})).collect();
        for _ in 0..args.reps {
            let timings = dispatch_graph()?;
            let observed_normalized = read_f32(&normalized_buffer, HYPER_STATE_ELEMENTS);
            let observed_low_rank_down = read_f32(&low_rank_down_buffer, HYPER_LOWRANK);
            let observed_read_activation = read_f32(&read_activation_buffer, HYPER_LOWRANK);
            let observed_read_gate_logits =
                read_f32(&read_gate_logits_buffer, HYPER_STATE_ELEMENTS);
            let observed_mixed_input = read_f32(&mixed_input_buffer, SHARED_HIDDEN);
            let observed_gate_up = read_f32(&gate_up_buffer, SHARED_INTERMEDIATE);
            let observed_shared_down = read_f32(&shared_down_buffer, SHARED_HIDDEN);
            let observed_scalar_gate = read_f32(&scalar_gate_buffer, 1);
            let observed_block_output = read_f32(&block_output_buffer, SHARED_HIDDEN);
            let observed_routed_gate_up: Vec<Vec<f32>> = routed_gate_up_buffers
                .iter()
                .map(|buffer| read_f32(buffer, SHARED_INTERMEDIATE))
                .collect();
            let observed_routed_outputs_flat =
                read_f32(&routed_outputs_buffer, ROUTED_TOP_K * SHARED_HIDDEN);
            let observed_routed_outputs: Vec<Vec<f32>> = observed_routed_outputs_flat
                .chunks_exact(SHARED_HIDDEN)
                .map(ToOwned::to_owned)
                .collect();
            let observed_routed_mix = read_f32(&routed_mix_buffer, SHARED_HIDDEN);
            let observed_moe_output = read_f32(&moe_output_buffer, SHARED_HIDDEN);
            let observed_block_logits = read_f32(&block_logits_buffer, HYPER_STATE_STREAMS);
            let observed_combined = read_f32(&combined_buffer, HYPER_STATE_ELEMENTS);
            let mut expected_stages: Vec<&[f32]> = vec![
                &expected_normalized,
                &expected_low_rank_down,
                &expected_read_activation,
                &expected_read_gate_logits,
                &expected_mixed_input,
                &expected_gate_up,
                &expected_shared_down,
                &expected_scalar_gate,
                &expected_block_output,
            ];
            let mut observed_stages: Vec<&[f32]> = vec![
                &observed_normalized,
                &observed_low_rank_down,
                &observed_read_activation,
                &observed_read_gate_logits,
                &observed_mixed_input,
                &observed_gate_up,
                &observed_shared_down,
                &observed_scalar_gate,
                &observed_block_output,
            ];
            for expert in 0..routed_specs.len() {
                expected_stages.push(&expected_routed_gate_up[expert]);
                expected_stages.push(&expected_routed_outputs[expert]);
                observed_stages.push(&observed_routed_gate_up[expert]);
                observed_stages.push(&observed_routed_outputs[expert]);
            }
            expected_stages.push(&expected_routed_mix);
            expected_stages.push(&expected_moe_output);
            expected_stages.push(&expected_block_logits);
            expected_stages.push(&expected_combined);
            observed_stages.push(&observed_routed_mix);
            observed_stages.push(&observed_moe_output);
            observed_stages.push(&observed_block_logits);
            observed_stages.push(&observed_combined);
            for (index, (expected, observed)) in
                expected_stages.iter().zip(observed_stages).enumerate()
            {
                if !observed.iter().all(|value| value.is_finite()) {
                    return Err(format!(
                        "exact hyperconnection stage {index} produced a non-finite value"
                    )
                    .into());
                }
                parity[index] = output_metrics(expected, observed);
                if parity[index]
                    .get("within_tolerance")
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    return Err(format!(
                        "exact hyperconnection stage {index} parity failed: {}",
                        parity[index]
                    )
                    .into());
                }
                stage_hashes[index].push(sha256_bytes(&f32_bytes_for_hash(observed)));
            }
            let gpu_values: Vec<u64> = timings
                .iter()
                .map(|timing| gpu_ns(*timing))
                .collect::<Result<_, _>>()?;
            for (index, value) in gpu_values.iter().copied().enumerate() {
                stage_gpu_ns[index].push(value);
            }
            graph_gpu_ns.push(gpu_values.iter().copied().fold(0u64, u64::saturating_add));
            graph_host_ns.push(sum_host_ns(&timings));
        }
        for hashes in &stage_hashes {
            if hashes.windows(2).any(|pair| pair[0] != pair[1]) {
                return Err(
                    "exact hyperconnection outputs changed across repeated executions".into(),
                );
            }
        }
        let body_ref = |path: &Path, receipt: &Value| -> Result<Value, Box<dyn Error>> {
            let digest = sha256_bytes(&fs::read(path)?);
            Ok(component_ref(path, receipt, Some(&digest)))
        };
        let hc_norm_receipt_digest = sha256_bytes(&fs::read(&hc_norm_receipt_path)?);
        let routed_dependency_refs: Vec<Value> = routed_specs
            .iter()
            .map(|spec| {
                Ok::<Value, Box<dyn Error>>(json!({
                    "expert_index": spec.expert_index,
                    "gate_up_body_receipt": body_ref(&spec.gate_up_body.path, &spec.gate_up_receipt)?,
                    "gate_up_kernel_receipt": component_ref(
                        &spec.gate_up_kernel_receipt_path,
                        &spec.gate_up_kernel_receipt,
                        Some(&spec.gate_up_kernel_receipt_sha256),
                    ),
                    "down_body_receipt": body_ref(&spec.down_receipt_path, &spec.down_receipt)?,
                    "down_kernel_receipt": component_ref(
                        &spec.down_kernel_receipt_path,
                        &spec.down_kernel_receipt,
                        Some(&spec.down_kernel_receipt_sha256),
                    ),
                }))
            })
            .collect::<Result<_, _>>()?;
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "layer-0 exact Qwen3.8-Flash-Next HyperConnection read/write equations around the complete selected routed-plus-shared MoE candidate block",
            "execution_provider": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "nodes": [
                {"id": "residual_input", "shape": [HYPER_STATE_ELEMENTS], "dtype": "F32", "device_resident": true},
                {"id": "hc_norm_weight", "shape": [HYPER_STATE_ELEMENTS], "dtype": "BF16", "representation": "source_bf16_exact"},
                {"id": "hc_norm", "shape": [HYPER_STATE_ELEMENTS], "kernel": HYPER_NORM_KERNEL_NAME, "device_resident": true},
                {"id": "read_low_rank_down", "shape": [HYPER_LOWRANK], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "read_silu_scaled", "shape": [HYPER_LOWRANK], "kernel": HYPER_SILU_SCALE_KERNEL_NAME, "device_resident": true},
                {"id": "read_gate_logits", "shape": [HYPER_STATE_ELEMENTS], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "mixed_block_input", "shape": [SHARED_HIDDEN], "kernel": HYPER_READ_MIX_KERNEL_NAME, "device_resident": true},
                {"id": "shared_gate_up_swiglu", "shape": [SHARED_INTERMEDIATE], "kernel": GATE_UP_KERNEL_NAME, "device_resident": true},
                {"id": "shared_down", "shape": [SHARED_HIDDEN], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "shared_block_output", "shape": [SHARED_HIDDEN], "kernel": SHARED_SIGMOID_GATE_KERNEL_NAME, "device_resident": true},
                {"id": "routed_gate_up_outputs", "shape": [ROUTED_TOP_K, SHARED_INTERMEDIATE], "kernel": GATE_UP_KERNEL_NAME, "device_resident": true},
                {"id": "routed_down_outputs", "shape": [ROUTED_TOP_K, SHARED_HIDDEN], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "selected_weights", "shape": [ROUTED_TOP_K], "dtype": "F32", "device_resident": true},
                {"id": "routed_weighted_sum", "shape": [SHARED_HIDDEN], "kernel": MOE_WEIGHTED_SUM_KERNEL_NAME, "device_resident": true},
                {"id": "moe_output", "shape": [SHARED_HIDDEN], "kernel": MOE_ADD_SHARED_KERNEL_NAME, "device_resident": true},
                {"id": "block_inject_logits", "shape": [HYPER_STATE_STREAMS], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "combined_residual", "shape": [HYPER_STATE_ELEMENTS], "kernel": HYPER_COMBINE_KERNEL_NAME, "device_resident": true}
            ],
            "edges": [
                ["residual_input", "hc_norm"], ["hc_norm_weight", "hc_norm"],
                ["hc_norm", "read_low_rank_down"], ["read_low_rank_down", "read_silu_scaled"],
                ["read_silu_scaled", "read_gate_logits"], ["hc_norm", "mixed_block_input"],
                ["read_gate_logits", "mixed_block_input"], ["mixed_block_input", "shared_gate_up_swiglu"],
                ["shared_gate_up_swiglu", "shared_down"], ["mixed_block_input", "shared_down"],
                ["shared_down", "shared_block_output"], ["mixed_block_input", "shared_block_output"],
                ["mixed_block_input", "routed_gate_up_outputs"], ["routed_gate_up_outputs", "routed_down_outputs"],
                ["selected_weights", "routed_weighted_sum"], ["routed_down_outputs", "routed_weighted_sum"],
                ["routed_weighted_sum", "moe_output"], ["shared_block_output", "moe_output"],
                ["hc_norm", "block_inject_logits"], ["residual_input", "combined_residual"],
                ["moe_output", "combined_residual"], ["block_inject_logits", "combined_residual"]
            ],
            "native_kernels": [HYPER_NORM_KERNEL_NAME, KERNEL_NAME, HYPER_SILU_SCALE_KERNEL_NAME, HYPER_READ_MIX_KERNEL_NAME, GATE_UP_KERNEL_NAME, SHARED_SIGMOID_GATE_KERNEL_NAME, MOE_WEIGHTED_SUM_KERNEL_NAME, MOE_ADD_SHARED_KERNEL_NAME, HYPER_COMBINE_KERNEL_NAME],
            "native_kernel_execution_observed": true,
            "routed_expert_count": routed_specs.len(),
            "routed_expert_ids": selected_ids,
            "dispatches_per_graph": measured_stage_count,
            "device_intermediate_no_host_roundtrip": true,
            "verification_reads_after_graph": true,
            "promotion_allowed": false
        });
        physical_graph["fingerprint"] =
            Value::String(sha256_bytes(&serde_json::to_vec(&physical_graph)?));
        let routed_stage_start = 9usize;
        let routed_weighted_sum_stage = routed_stage_start + routed_specs.len() * 2;
        let moe_output_stage = routed_weighted_sum_stage + 1;
        let block_inject_stage = moe_output_stage + 1;
        let combined_stage = block_inject_stage + 1;
        let median = |index: usize| percentile_median(&stage_gpu_ns[index]);
        let routed_timing: Vec<Value> = routed_specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let gate_up_stage = routed_stage_start + index * 2;
                let down_stage = gate_up_stage + 1;
                json!({
                    "expert_index": spec.expert_index,
                    "gate_up_swiglu_gpu_ns": stage_gpu_ns[gate_up_stage],
                    "gate_up_swiglu_gpu_ns_median": median(gate_up_stage),
                    "down_projection_gpu_ns": stage_gpu_ns[down_stage],
                    "down_projection_gpu_ns_median": median(down_stage)
                })
            })
            .collect();
        let routed_stage_hashes: Vec<Value> = routed_specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let gate_up_stage = routed_stage_start + index * 2;
                let down_stage = gate_up_stage + 1;
                json!({
                    "expert_index": spec.expert_index,
                    "gate_up_swiglu": stage_hashes[gate_up_stage],
                    "down": stage_hashes[down_stage]
                })
            })
            .collect();
        Ok(json!({
            "schema": EXACT_HYPERCONNECTION_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutable",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "EXACT_QWEN38_FLASH_NEXT_HYPERCONNECTION_READ_WRITE_AROUND_LAYER0_ROUTED_SHARED_MOE_CANDIDATE",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "layer": 0,
            "source_reference": {
                "implementation": "Qwen3_8_FlashNextHyperConnection",
                "hc_norm": "grouped RMSNorm over four contiguous 2560-wide streams; output = x * rsqrt(mean(x^2)+1e-6) * (1 + BF16 weight)",
                "mix": "normalized -> SiLU(input_mix_weight_down(normalized) / hc_count) -> sigmoid(input_mix_weight_up(...)); mean over streams of gate * normalized",
                "combine": "residual + block_output broadcast over streams * (2 * sigmoid(block_inject_weight(normalized) / hc_count))",
                "hc_count": HYPER_STATE_STREAMS,
                "hidden_size": HYPER_STATE_HIDDEN,
                "lowrank_size": HYPER_LOWRANK,
                "rms_norm_eps": 1.0e-6f32
            },
            "dependencies": {
                "hc_norm_vector_receipt": component_ref(&hc_norm_receipt_path, &hc_norm_receipt, Some(&hc_norm_receipt_digest)),
                "input_mix_down_receipt": body_ref(&input_down_receipt_path, &input_down_receipt)?,
                "input_mix_up_receipt": body_ref(&input_up_receipt_path, &input_up_receipt)?,
                "block_inject_receipt": body_ref(&block_inject_receipt_path, &block_inject_receipt)?,
                "shared_gate_receipt": body_ref(&gate_receipt_path, &gate_receipt)?,
                "shared_up_receipt": body_ref(&up_receipt_path, &up_receipt)?,
                "shared_down_receipt": body_ref(&down_receipt_path, &down_receipt)?,
                "shared_scalar_gate_receipt": body_ref(&scalar_gate_receipt_path, &scalar_gate_receipt)?,
                "router_receipt": component_ref(&args.router_receipt, &router, Some(&router_sha256)),
                "campaign_receipt": component_ref(&args.campaign_receipt, &campaign, Some(&campaign_sha256)),
                "routed_expert_receipts": routed_dependency_refs
            },
            "execution": {
                "provider": "apple-metal",
                "operation": "exact HyperConnection read -> normalized top-10 routed experts + sigmoid-gated shared expert -> device-resident MoE weighted sum/add -> exact HyperConnection write",
                "dispatches_per_graph": measured_stage_count,
                "measured_graphs": args.reps,
                "total_measured_dispatches": args.reps * measured_stage_count,
                "source_hc_norm_payload_loaded": true,
                "hc_norm_loaded": true,
                "hc_norm_source_payload_exact": true,
                "native_hyperconnection_read_observed": true,
                "native_hyperconnection_write_observed": true,
                "exact_hyperconnection_semantics_observed": true,
                "native_shared_expert_gate_up_swiglu_observed": true,
                "native_shared_expert_down_projection_observed": true,
                "native_shared_expert_sigmoid_gate_observed": true,
                "native_routed_expert_gate_up_swiglu_observed": true,
                "native_routed_expert_down_projection_observed": true,
                "native_moe_weighted_sum_observed": true,
                "native_moe_shared_add_observed": true,
                "routed_expert_count": routed_specs.len(),
                "routed_expert_ids": selected_ids,
                "selected_weight_sum": selected_weight_sum,
                "complete_layer0_moe_candidate": true,
                "complete_moe_combine": true,
                "device_intermediate_no_host_roundtrip": true,
                "verification_reads_after_graph": true,
                "source_reference_used_for_execution": false,
                "body_mutated": false,
                "model_loaded": false,
                "complete_token_runtime": false
            },
            "input": {
                "residual_state": {"definition": "((index * 71) mod 509 - 254) / 509", "values": base_state.len(), "deterministic_sha256": base_state_sha256, "label": "[V]"},
                "stream_layout": "four contiguous 2560-wide streams; read and write preserve source stream order",
                "projection_dtype": "F32 activation at native Q4/G64 matrix boundaries"
            },
            "semantics": {
                "status": "EXACT_SOURCE_EQUATIONS_BOUND_TO_PINNED_TENSOR_LAYOUT",
                "hc_norm": "grouped RMSNorm per stream with exact persisted BF16 weight and eps=1e-6",
                "read_stream_indexing": "contiguous stream-major [stream, hidden] with mean across stream axis",
                "read_scaling": "input_mix_weight_down output divided by hc_count before SiLU",
                "write_scaling": "block_inject_weight output divided by hc_count before sigmoid, then multiplied by 2",
                "write_routing": "one shared block output broadcast to every stream; no guessed stream replacement",
                "moe": "normalized source-selected top-10 routed expert outputs are weighted and summed, then the sigmoid-gated shared expert output is added on-device",
                "routed_selection": "the persisted router receipt supplies ten selected expert IDs and normalized selected weights; source selection parity is reported separately",
                "weight_numeric_boundary": "hyperconnection, routed-expert, and shared-expert matrices are persisted Q4/G64 candidates; equations and device graph are exact, source BF16 activation parity remains unqualified"
            },
            "parity": {
                "candidate_space": "CPU reference uses the same persisted Q4/G64 matrix bodies and exact BF16 hc_norm vector; all native stages are compared after the graph",
                "hc_norm": parity[0].clone(),
                "input_mix_down": parity[1].clone(),
                "read_silu_scaled": parity[2].clone(),
                "input_mix_up": parity[3].clone(),
                "read_mix": parity[4].clone(),
                "shared_gate_up_swiglu": parity[5].clone(),
                "shared_down": parity[6].clone(),
                "shared_scalar_gate": parity[7].clone(),
                "shared_block_output": parity[8].clone(),
                "routed_experts": routed_specs.iter().enumerate().map(|(index, spec)| json!({
                    "expert_index": spec.expert_index,
                    "gate_up_swiglu": parity[routed_stage_start + index * 2].clone(),
                    "down": parity[routed_stage_start + index * 2 + 1].clone()
                })).collect::<Vec<_>>(),
                "routed_weighted_sum": parity[routed_weighted_sum_stage].clone(),
                "moe_output": parity[moe_output_stage].clone(),
                "block_inject": parity[block_inject_stage].clone(),
                "combined_residual": parity[combined_stage].clone()
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_graph_gpu_ns": warmup_graph_gpu_ns,
                "hc_norm_gpu_ns": stage_gpu_ns[0], "hc_norm_gpu_ns_median": median(0),
                "input_mix_down_gpu_ns": stage_gpu_ns[1], "input_mix_down_gpu_ns_median": median(1),
                "read_silu_scaled_gpu_ns": stage_gpu_ns[2], "read_silu_scaled_gpu_ns_median": median(2),
                "input_mix_up_gpu_ns": stage_gpu_ns[3], "input_mix_up_gpu_ns_median": median(3),
                "read_mix_gpu_ns": stage_gpu_ns[4], "read_mix_gpu_ns_median": median(4),
                "shared_gate_up_gpu_ns": stage_gpu_ns[5], "shared_gate_up_gpu_ns_median": median(5),
                "shared_down_gpu_ns": stage_gpu_ns[6], "shared_down_gpu_ns_median": median(6),
                "shared_scalar_gate_gpu_ns": stage_gpu_ns[7], "shared_scalar_gate_gpu_ns_median": median(7),
                "shared_sigmoid_gate_gpu_ns": stage_gpu_ns[8], "shared_sigmoid_gate_gpu_ns_median": median(8),
                "routed_experts": routed_timing,
                "routed_weighted_sum_gpu_ns": stage_gpu_ns[routed_weighted_sum_stage], "routed_weighted_sum_gpu_ns_median": median(routed_weighted_sum_stage),
                "moe_add_shared_gpu_ns": stage_gpu_ns[moe_output_stage], "moe_add_shared_gpu_ns_median": median(moe_output_stage),
                "block_inject_gpu_ns": stage_gpu_ns[block_inject_stage], "block_inject_gpu_ns_median": median(block_inject_stage),
                "combine_gpu_ns": stage_gpu_ns[combined_stage], "combine_gpu_ns_median": median(combined_stage),
                "graph_gpu_ns": graph_gpu_ns, "graph_gpu_ns_median": percentile_median(&graph_gpu_ns),
                "graph_host_wall_ns": graph_host_ns, "graph_host_wall_ns_median": percentile_median(&graph_host_ns),
                "dispatches_per_graph": measured_stage_count,
                "stage_output_hashes": {
                    "hc_norm": stage_hashes[0], "input_mix_down": stage_hashes[1], "read_silu_scaled": stage_hashes[2],
                    "input_mix_up": stage_hashes[3], "read_mix": stage_hashes[4], "shared_gate_up_swiglu": stage_hashes[5],
                    "shared_down": stage_hashes[6], "shared_scalar_gate": stage_hashes[7], "shared_block_output": stage_hashes[8],
                    "routed_experts": routed_stage_hashes,
                    "routed_weighted_sum": stage_hashes[routed_weighted_sum_stage],
                    "moe_output": stage_hashes[moe_output_stage],
                    "block_inject": stage_hashes[block_inject_stage], "combined_residual": stage_hashes[combined_stage]
                },
                "memory_limits": {"max_buffer_length": device_memory.max_buffer_length, "recommended_max_working_set_size": device_memory.recommended_max_working_set_size, "current_allocated_size": device_memory.current_allocated_size, "has_unified_memory": device_memory.has_unified_memory},
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for every native exact-boundary and selected-routed-MoE dispatch; host wall is reported separately"
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "representation": "source_bf16_exact_hc_norm_plus_independent_q4_g64_matrices",
                "operations": [
                    "load_exact_pinned_mlp_hyperconnection_hc_norm_bf16_vector",
                    "execute_native_grouped_rmsnorm_per_hyperconnection_stream",
                    "execute_native_q4_g64_input_mix_down",
                    "execute_native_silu_after_hc_count_scaling",
                    "execute_native_q4_g64_input_mix_up",
                    "execute_native_hyperconnection_read_mean_of_sigmoid_gated_normalized_streams",
                    "execute_native_shared_expert_gate_up_swiglu_on_exact_read_input",
                    "execute_native_shared_expert_down_projection",
                    "execute_native_sigmoid_shared_expert_gate",
                    "load_native_router_selection",
                    "load_full_selected_routed_expert_bodies",
                    "execute_native_routed_gate_up_swiglu_per_selected_expert",
                    "execute_native_routed_down_projection_per_selected_expert",
                    "execute_native_device_resident_weighted_sum",
                    "execute_native_moe_shared_add",
                    "execute_native_q4_g64_block_inject_on_exact_normalized_state",
                    "execute_native_hyperconnection_write_broadcast_with_two_sigmoid_gate",
                    "emit_bounded_exact_layer0_routed_shared_moe_candidate"
                ],
                "source_independent": true,
                "exact_hyperconnection_equations": true,
                "complete_moe_combine": true,
                "complete_layer0_moe_candidate": true,
                "complete_model": false,
                "complete_token": false
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "source_independent_execution": true,
            "source_hc_norm_payload_exact": true,
            "hc_norm_loaded": true,
            "native_hyperconnection_read_observed": true,
            "native_hyperconnection_write_observed": true,
            "exact_hyperconnection_semantics_observed": true,
            "native_shared_expert_gate_up_swiglu_observed": true,
            "native_shared_expert_down_projection_observed": true,
            "native_shared_expert_sigmoid_gate_observed": true,
            "native_routed_expert_gate_up_swiglu_observed": true,
            "native_routed_expert_down_projection_observed": true,
            "native_moe_weighted_sum_observed": true,
            "native_moe_shared_add_observed": true,
            "routed_expert_count": routed_specs.len(),
            "routed_expert_ids": selected_ids,
            "selected_weight_sum": selected_weight_sum,
            "complete_layer0_moe_candidate": true,
            "complete_moe_combine": true,
            "device_intermediate_no_host_roundtrip": true,
            "body_mutated": false,
            "model_loaded": false,
            "promotion_allowed": false,
            "source_selection_parity": router.get("source_selection_parity").cloned().unwrap_or(Value::Null),
            "claim_boundary": "PASSED bounded exact Qwen3.8-Flash-Next HyperConnection read/write equations around a device-resident layer-0 routed-plus-shared MoE candidate. hc_norm BF16 payload and equations are exact; all expert/hyper/shared matrices are persisted Q4/G64 candidate bodies; current native route selection source parity is mismatched (8/10 overlap), so source BF16 output parity, complete model/layer/token, TPS, EBPW, and promotion remain unqualified.",
            "next_action": "close source router/top-k parity, then qualify source BF16 activation/output parity and extend the exact layer-0 candidate through attention/state before attempting a complete-token baseline",
            "elapsed_s": started.elapsed().as_secs_f64()
        }))
    }

    fn gpu_ns(timing: MetalDispatchTiming) -> Result<u64, Box<dyn Error>> {
        match (timing.gpu_start_ns, timing.gpu_end_ns) {
            (Some(start), Some(end)) if end > start => Ok(end - start),
            _ => Err("Metal completed without a usable GPU timestamp".into()),
        }
    }

    fn output_metrics(reference: &[f32], observed: &[f32]) -> Value {
        let mut max_abs = 0.0f32;
        let mut squared = 0.0f64;
        let mut left_norm = 0.0f64;
        let mut right_norm = 0.0f64;
        let mut dot = 0.0f64;
        for (left, right) in reference.iter().zip(observed) {
            let error = *left - *right;
            max_abs = max_abs.max(error.abs());
            squared += f64::from(error) * f64::from(error);
            left_norm += f64::from(*left) * f64::from(*left);
            right_norm += f64::from(*right) * f64::from(*right);
            dot += f64::from(*left) * f64::from(*right);
        }
        let count = reference.len();
        json!({
            "count": count,
            "max_abs_error": max_abs,
            "rmse": (squared / count as f64).sqrt(),
            "cosine": if left_norm > 0.0 && right_norm > 0.0 { Some(dot / (left_norm.sqrt() * right_norm.sqrt())) } else { None },
            "finite": observed.iter().all(|value| value.is_finite()),
            "tolerance": OUTPUT_ERROR_TOLERANCE,
            "within_tolerance": max_abs <= OUTPUT_ERROR_TOLERANCE,
        })
    }

    fn percentile_median(values: &[u64]) -> u64 {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    fn component_ref(path: &Path, value: &Value, sha256: Option<&str>) -> Value {
        json!({
            "path": path,
            "sha256": sha256,
            "schema": value.get("schema"),
            "status": value.get("status"),
            "label": "[V]",
        })
    }

    fn write_atomic(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
        fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn run(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let (campaign, campaign_sha256) = validate_campaign(&args.campaign_receipt)?;
        let components = campaign
            .get("components")
            .and_then(Value::as_array)
            .ok_or("routed-expert campaign has no components")?;

        let mut selected_specs = Vec::with_capacity(selected_ids.len());
        for expert_id in &selected_ids {
            let component = components
                .iter()
                .find(|component| {
                    component
                        .get("window")
                        .and_then(|window| window.get("expert_index"))
                        .and_then(Value::as_u64)
                        == Some(*expert_id as u64)
                })
                .ok_or_else(|| {
                    format!("campaign has no persisted body for selected expert {expert_id}")
                })?;
            let body_path = component
                .get("body_receipt")
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!("campaign component for expert {expert_id} has no body receipt")
                })?;
            let kernel_path = component
                .get("kernel_receipt")
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!("campaign component for expert {expert_id} has no kernel receipt")
                })?;
            let (body, body_receipt) = validate_body(Path::new(body_path))?;
            let (kernel, kernel_sha256) = validate_kernel(Path::new(kernel_path), &body)?;
            if body.expert_index != *expert_id {
                return Err(format!(
                    "selected expert {expert_id} does not match body expert {}",
                    body.expert_index
                )
                .into());
            }
            if let Some(input) = kernel.get("input") {
                let expected_input = sha256_bytes(&f32_bytes(&deterministic_input(body.columns)));
                if input.get("deterministic_sha256").and_then(Value::as_str)
                    != Some(expected_input.as_str())
                {
                    return Err(format!(
                        "expert {expert_id} kernel input does not match deterministic route input"
                    )
                    .into());
                }
            }
            selected_specs.push((
                body,
                body_receipt,
                PathBuf::from(body_path),
                kernel,
                PathBuf::from(kernel_path),
                kernel_sha256,
            ));
        }

        let columns = selected_specs
            .first()
            .map(|(body, _, _, _, _, _)| body.columns)
            .ok_or("no selected routed experts")?;
        let rows = selected_specs
            .first()
            .map(|(body, _, _, _, _, _)| body.rows)
            .ok_or("no selected routed experts")?;
        if selected_specs
            .iter()
            .any(|(body, _, _, _, _, _)| body.columns != columns || body.rows != rows)
        {
            return Err("selected expert body windows have different shapes".into());
        }
        let input = deterministic_input(columns);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let input_buffer_bytes = f32_bytes(&input);
        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&input_buffer_bytes)?;
        let mut loaded = Vec::with_capacity(selected_specs.len());
        for (body, body_receipt, body_receipt_path, kernel, kernel_receipt_path, kernel_sha256) in
            selected_specs
        {
            let expected = cpu_matvec(&body, &input);
            loaded.push(LoadedBody {
                codes: context.new_buffer_with_bytes_checked(&body.codes)?,
                scales: context.new_buffer_with_bytes_checked(&body.scales)?,
                output: context.new_buffer_checked(body.rows * std::mem::size_of::<f32>())?,
                body,
                body_receipt,
                body_receipt_path,
                kernel_receipt: kernel,
                kernel_receipt_path,
                kernel_receipt_sha256: kernel_sha256,
                expected,
            });
        }

        let mut warmup_route_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let mut total = 0u64;
            for item in &loaded {
                total = total.saturating_add(gpu_ns(dispatch(
                    &context,
                    &item.body,
                    &item.codes,
                    &item.scales,
                    &input_buffer,
                    &item.output,
                )?)?);
            }
            warmup_route_gpu_ns.push(total);
        }

        let mut per_body_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_body_host_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_body_output_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut route_gpu_ns = Vec::with_capacity(args.reps);
        let mut route_host_ns = Vec::with_capacity(args.reps);
        let mut mixed_hashes = Vec::with_capacity(args.reps);
        let mut last_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        for _ in 0..args.reps {
            let mut total_gpu = 0u64;
            let mut total_host = 0u64;
            let mut mixed = vec![0.0f32; loaded[0].body.rows];
            for (index, item) in loaded.iter().enumerate() {
                let timing = dispatch(
                    &context,
                    &item.body,
                    &item.codes,
                    &item.scales,
                    &input_buffer,
                    &item.output,
                )?;
                let observed = read_f32(&item.output, item.body.rows);
                if !observed.iter().all(|value| value.is_finite()) {
                    return Err(format!(
                        "native routed body {} output is non-finite",
                        item.body.expert_index
                    )
                    .into());
                }
                let parity = output_metrics(&item.expected, &observed);
                if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(format!(
                        "native routed body {} parity failed: {parity}",
                        item.body.expert_index
                    )
                    .into());
                }
                let gpu = gpu_ns(timing)?;
                let host = timing.host_wall_us.saturating_mul(1000);
                total_gpu = total_gpu.saturating_add(gpu);
                total_host = total_host.saturating_add(host);
                per_body_gpu_ns[index].push(gpu);
                per_body_host_ns[index].push(host);
                per_body_output_hashes[index].push(sha256_bytes(&f32_bytes_for_hash(&observed)));
                let weight = selected_weights[index];
                for (slot, value) in observed.iter().enumerate() {
                    mixed[slot] += weight * value;
                }
                last_outputs[index] = observed;
            }
            route_gpu_ns.push(total_gpu);
            route_host_ns.push(total_host);
            mixed_hashes.push(sha256_bytes(&f32_bytes_for_hash(&mixed)));
        }

        if per_body_output_hashes
            .iter()
            .any(|hashes| hashes.windows(2).any(|pair| pair[0] != pair[1]))
            || mixed_hashes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err("native routed body outputs changed across repeated executions".into());
        }
        let mut expected_mixed = vec![0.0f32; loaded[0].body.rows];
        for (index, item) in loaded.iter().enumerate() {
            for (slot, value) in item.expected.iter().enumerate() {
                expected_mixed[slot] += selected_weights[index] * value;
            }
        }
        let observed_mixed = loaded.iter().enumerate().fold(
            vec![0.0f32; loaded[0].body.rows],
            |mut mixed, (index, _item)| {
                for (slot, value) in last_outputs[index].iter().enumerate() {
                    mixed[slot] += selected_weights[index] * value;
                }
                mixed
            },
        );
        let mixed_parity = output_metrics(&expected_mixed, &observed_mixed);
        if mixed_parity
            .get("within_tolerance")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(format!("weighted routed gather parity failed: {mixed_parity}").into());
        }

        let source_selection_parity = router
            .get("source_selection_parity")
            .cloned()
            .unwrap_or_else(|| json!({"status": "UNKNOWN"}));
        let source_mismatch = source_selection_parity
            .get("expert_ids_exact_match")
            .and_then(Value::as_bool)
            == Some(false);
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "native router receipt -> selected persisted routed-expert body windows -> Q4/G64 Metal matvecs -> host weighted gather; no complete expert/token graph",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_kernel_execution_observed": true,
            "selected_expert_count": loaded.len(),
            "dispatches_per_route": loaded.len(),
            "source_selection_parity_status": source_selection_parity.get("status"),
            "source_selection_parity_qualified": source_selection_parity.get("expert_ids_exact_match"),
            "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let mut components_json = Vec::with_capacity(loaded.len());
        for (index, item) in loaded.iter().enumerate() {
            components_json.push(json!({
                "expert_index": item.body.expert_index,
                "row_start": item.body.row_start,
                "row_count": item.body.rows,
                "body_receipt": component_ref(&item.body_receipt_path, &item.body_receipt, Some(&item.body.receipt_sha256)),
                "kernel_receipt": component_ref(
                    &item.kernel_receipt_path,
                    &item.kernel_receipt,
                    Some(&item.kernel_receipt_sha256),
                ),
                "candidate_body": {
                    "path": item.body.path,
                    "sha256": item.body.body_sha256,
                    "bytes": item.body.code_bytes + item.body.scale_bytes,
                    "source_independent": true,
                    "candidate_body_persisted": true,
                    "label": "[D]",
                },
                "native_dispatch": {
                    "kernel": KERNEL_NAME,
                    "dispatches_per_route": 1,
                    "gpu_ns": per_body_gpu_ns[index],
                    "gpu_ns_median": percentile_median(&per_body_gpu_ns[index]),
                    "host_wall_ns": per_body_host_ns[index],
                    "host_wall_ns_median": percentile_median(&per_body_host_ns[index]),
                    "output_hashes": per_body_output_hashes[index],
                    "stable_output": true,
                },
                "parity": output_metrics(&item.expected, &last_outputs[index]),
                "source_reference_used_for_execution": false,
            }));
        }

        let router_selection = router
            .get("selection")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let route_gpu_median = percentile_median(&route_gpu_ns);
        let route_host_median = percentile_median(&route_host_ns);
        Ok(json!({
            "schema": SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_BODY_DISPATCH",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "router_receipt": {
                "path": args.router_receipt.canonicalize()?,
                "sha256": router_sha256,
                "schema": ROUTER_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "campaign_receipt": {
                "path": args.campaign_receipt.canonicalize()?,
                "sha256": campaign_sha256,
                "schema": CAMPAIGN_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "selection": router_selection,
            "source_selection_parity": source_selection_parity,
            "components": components_json,
            "execution": {
                "provider": "apple-metal",
                "operation": "native_router_selection_receipt -> selected_persisted_routed_expert_body_load -> Q4_G64_matvec_per_selected_body -> host_selected_weight_gather",
                "selected_expert_ids": selected_ids,
                "selected_expert_count": loaded.len(),
                "dispatches_per_route": loaded.len(),
                "measured_routes": args.reps,
                "total_measured_dispatches": loaded.len() * args.reps,
                "native_routed_body_dispatch_observed": true,
                "source_reference_used_for_execution": false,
                "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
                "body_mutated": false,
                "model_loaded": false,
                "complete_expert_activation": false,
                "complete_token_runtime": false,
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": columns,
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_route_gpu_ns": warmup_route_gpu_ns,
                "route_gpu_ns": route_gpu_ns,
                "route_gpu_ns_median": route_gpu_median,
                "route_host_wall_ns": route_host_ns,
                "route_host_wall_ns_median": route_host_median,
                "dispatches_per_route": loaded.len(),
                "output_hashes": mixed_hashes,
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime summed across selected body dispatches; host wall reported separately",
            },
            "gather": {
                "status": "BOUNDED_HOST_WEIGHTED_GATHER",
                "weights": selected_weights,
                "output_rows": loaded[0].body.rows,
                "output_sha256": mixed_hashes.last(),
                "parity": mixed_parity,
                "fused_native_gather": false,
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "consume_native_router_selection",
                    "resolve_selected_persisted_routed_expert_body_windows",
                    "load_source_independent_q4_g64_body_windows",
                    "execute_native_q4_g64_matvec_per_selected_body",
                    "apply_selected_router_weights_on_host",
                    "emit_bounded_routed_body_outputs",
                ],
                "source_independent": true,
                "complete_expert": false,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_routed_body_dispatch_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent dispatch of the selected persisted routed-expert Q4/G64 body windows with a host-side selected-weight gather. This is not full expert activation, fused route/gather, complete-model loading, complete-token runtime, Flash TPS, or EBPW evidence; source-selection mismatch remains explicit.",
            "next_action": "extend from bounded routed body windows to independently validated gate/up activation and native expert composition; do not measure or claim complete-token Flash TPS/EBPW until the full protected graph is capability-qualified",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    fn run_gate_up_swiglu(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let mut selected_specs = Vec::with_capacity(selected_ids.len());
        for expert_id in &selected_ids {
            let body_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_BODY_E{expert_id}_R0_1280.json"
            ));
            let kernel_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_KERNEL_E{expert_id}_R0_1280_PARITY.json"
            ));
            let (body, body_receipt) = validate_body(&body_path)?;
            if body.expert_index != *expert_id
                || body.row_start != 0
                || body.rows != GATE_UP_BODY_ROWS
            {
                return Err(format!(
                    "gate/up body receipt for expert {expert_id} is not the full 1280-row body"
                )
                .into());
            }
            let (kernel, kernel_sha256) = validate_kernel(&kernel_path, &body)?;
            let expected_input = sha256_bytes(&f32_bytes(&deterministic_input(body.columns)));
            if kernel
                .get("input")
                .and_then(|input| input.get("deterministic_sha256"))
                .and_then(Value::as_str)
                != Some(expected_input.as_str())
            {
                return Err(format!(
                    "gate/up kernel input for expert {expert_id} does not match the deterministic input"
                )
                .into());
            }
            selected_specs.push((
                body,
                body_receipt,
                body_path,
                kernel,
                kernel_path,
                kernel_sha256,
            ));
        }
        let columns = selected_specs
            .first()
            .map(|(body, _, _, _, _, _)| body.columns)
            .ok_or("no selected gate/up experts")?;
        if selected_specs.iter().any(|(body, _, _, _, _, _)| {
            body.columns != columns || body.row_start != 0 || body.rows != GATE_UP_BODY_ROWS
        }) {
            return Err("selected gate/up bodies have inconsistent full-body shapes".into());
        }

        let input = deterministic_input(columns);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&input))?;
        let mut loaded = Vec::with_capacity(selected_specs.len());
        for (body, body_receipt, body_receipt_path, kernel, kernel_receipt_path, kernel_sha256) in
            selected_specs
        {
            let (gate, up) = split_gate_up(&body)?;
            let expected = cpu_gate_up_swiglu(&gate, &up, &input);
            loaded.push(LoadedGateUp {
                gate_codes: context.new_buffer_with_bytes_checked(&gate.codes)?,
                gate_scales: context.new_buffer_with_bytes_checked(&gate.scales)?,
                up_codes: context.new_buffer_with_bytes_checked(&up.codes)?,
                up_scales: context.new_buffer_with_bytes_checked(&up.scales)?,
                output: context.new_buffer_checked(gate.rows * std::mem::size_of::<f32>())?,
                body,
                gate,
                up,
                body_receipt,
                body_receipt_path,
                kernel_receipt: kernel,
                kernel_receipt_path,
                kernel_receipt_sha256: kernel_sha256,
                expected,
            });
        }

        let mut warmup_route_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let mut total = 0u64;
            for item in &loaded {
                total = total.saturating_add(gpu_ns(dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.output,
                )?)?);
            }
            warmup_route_gpu_ns.push(total);
        }

        let mut per_expert_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_host_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_output_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut route_gpu_ns = Vec::with_capacity(args.reps);
        let mut route_host_ns = Vec::with_capacity(args.reps);
        let mut mixed_hashes = Vec::with_capacity(args.reps);
        let mut last_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        for _ in 0..args.reps {
            let mut total_gpu = 0u64;
            let mut total_host = 0u64;
            let mut mixed = vec![0.0f32; GATE_UP_ROWS];
            for (index, item) in loaded.iter().enumerate() {
                let timing = dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.output,
                )?;
                let observed = read_f32(&item.output, GATE_UP_ROWS);
                if !observed.iter().all(|value| value.is_finite()) {
                    return Err(format!(
                        "native gate/up SwiGLU output for expert {} is non-finite",
                        item.body.expert_index
                    )
                    .into());
                }
                let parity = output_metrics(&item.expected, &observed);
                if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(format!(
                        "native gate/up SwiGLU parity failed for expert {}: {parity}",
                        item.body.expert_index
                    )
                    .into());
                }
                let gpu = gpu_ns(timing)?;
                let host = timing.host_wall_us.saturating_mul(1000);
                total_gpu = total_gpu.saturating_add(gpu);
                total_host = total_host.saturating_add(host);
                per_expert_gpu_ns[index].push(gpu);
                per_expert_host_ns[index].push(host);
                per_expert_output_hashes[index].push(sha256_bytes(&f32_bytes_for_hash(&observed)));
                for (slot, value) in observed.iter().enumerate() {
                    mixed[slot] += selected_weights[index] * value;
                }
                last_outputs[index] = observed;
            }
            route_gpu_ns.push(total_gpu);
            route_host_ns.push(total_host);
            mixed_hashes.push(sha256_bytes(&f32_bytes_for_hash(&mixed)));
        }
        if per_expert_output_hashes
            .iter()
            .any(|hashes| hashes.windows(2).any(|pair| pair[0] != pair[1]))
            || mixed_hashes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err("native gate/up SwiGLU outputs changed across repeated executions".into());
        }
        let mut expected_mixed = vec![0.0f32; GATE_UP_ROWS];
        let mut observed_mixed = vec![0.0f32; GATE_UP_ROWS];
        for (index, item) in loaded.iter().enumerate() {
            for (slot, value) in item.expected.iter().enumerate() {
                expected_mixed[slot] += selected_weights[index] * value;
            }
            for (slot, value) in last_outputs[index].iter().enumerate() {
                observed_mixed[slot] += selected_weights[index] * value;
            }
        }
        let mixed_parity = output_metrics(&expected_mixed, &observed_mixed);
        if mixed_parity
            .get("within_tolerance")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(format!("weighted gate/up gather parity failed: {mixed_parity}").into());
        }

        let source_selection_parity = router
            .get("source_selection_parity")
            .cloned()
            .unwrap_or_else(|| json!({"status": "UNKNOWN"}));
        let source_mismatch = source_selection_parity
            .get("expert_ids_exact_match")
            .and_then(Value::as_bool)
            == Some(false);
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "native router receipt -> selected full persisted gate_up bodies -> split gate/up Q4/G64 buffers -> native Metal gate_up_swiglu -> host weighted gather; down projection and complete token graph excluded",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_kernel": GATE_UP_KERNEL_NAME,
            "native_kernel_execution_observed": true,
            "selected_expert_count": loaded.len(),
            "dispatches_per_route": loaded.len(),
            "gate_rows": GATE_UP_ROWS,
            "full_body_rows": GATE_UP_BODY_ROWS,
            "source_selection_parity_status": source_selection_parity.get("status"),
            "source_selection_parity_qualified": source_selection_parity.get("expert_ids_exact_match"),
            "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let mut components_json = Vec::with_capacity(loaded.len());
        for (index, item) in loaded.iter().enumerate() {
            components_json.push(json!({
                "expert_index": item.body.expert_index,
                "body_rows": item.body.rows,
                "gate_rows": item.gate.rows,
                "up_rows": item.up.rows,
                "body_receipt": component_ref(&item.body_receipt_path, &item.body_receipt, Some(&item.body.receipt_sha256)),
                "kernel_receipt": component_ref(&item.kernel_receipt_path, &item.kernel_receipt, Some(&item.kernel_receipt_sha256)),
                "candidate_body": {
                    "path": item.body.path,
                    "sha256": item.body.body_sha256,
                    "bytes": item.body.code_bytes + item.body.scale_bytes,
                    "source_independent": true,
                    "candidate_body_persisted": true,
                    "layout": "fused gate_up body split into contiguous gate/up row halves",
                    "label": "[D]",
                },
                "native_dispatch": {
                    "kernel": GATE_UP_KERNEL_NAME,
                    "dispatches_per_route": 1,
                    "gpu_ns": per_expert_gpu_ns[index],
                    "gpu_ns_median": percentile_median(&per_expert_gpu_ns[index]),
                    "host_wall_ns": per_expert_host_ns[index],
                    "host_wall_ns_median": percentile_median(&per_expert_host_ns[index]),
                    "output_hashes": per_expert_output_hashes[index],
                    "stable_output": true,
                },
                "parity": output_metrics(&item.expected, &last_outputs[index]),
                "source_reference_used_for_execution": false,
            }));
        }
        let route_gpu_median = percentile_median(&route_gpu_ns);
        let route_host_median = percentile_median(&route_host_ns);
        Ok(json!({
            "schema": GATE_UP_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_GATE_UP_SWIGLU_ACTIVATION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "router_receipt": {
                "path": args.router_receipt.canonicalize()?,
                "sha256": router_sha256,
                "schema": ROUTER_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "component_receipt_policy": {
                "directory": repo.join("receipts/headless"),
                "body_pattern": "FLASH_NOETIC_GATE_UP_BODY_E{expert}_R0_1280.json",
                "kernel_pattern": "FLASH_NOETIC_GATE_UP_KERNEL_E{expert}_R0_1280_PARITY.json",
                "body_rows": GATE_UP_BODY_ROWS,
                "gate_rows": GATE_UP_ROWS,
            },
            "selection": router.get("selection").cloned().unwrap_or_else(|| json!({})),
            "source_selection_parity": source_selection_parity,
            "components": components_json,
            "execution": {
                "provider": "apple-metal",
                "operation": "native_router_selection_receipt -> full persisted fused gate_up body load -> gate/up row split -> native Q4_G64 gate_up_swiglu -> host_selected_weight_gather",
                "selected_expert_ids": selected_ids,
                "selected_expert_count": loaded.len(),
                "dispatches_per_route": loaded.len(),
                "measured_routes": args.reps,
                "total_measured_dispatches": loaded.len() * args.reps,
                "native_gate_up_swiglu_observed": true,
                "native_expert_gate_up_activation_observed": true,
                "source_reference_used_for_execution": false,
                "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
                "body_mutated": false,
                "model_loaded": false,
                "complete_expert_activation": false,
                "complete_token_runtime": false,
                "down_projection": "NOT_TESTED",
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": columns,
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_route_gpu_ns": warmup_route_gpu_ns,
                "route_gpu_ns": route_gpu_ns,
                "route_gpu_ns_median": route_gpu_median,
                "route_host_wall_ns": route_host_ns,
                "route_host_wall_ns_median": route_host_median,
                "dispatches_per_route": loaded.len(),
                "output_hashes": mixed_hashes,
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for native gate_up_swiglu dispatches summed across selected experts; host wall reported separately",
            },
            "gather": {
                "status": "BOUNDED_HOST_WEIGHTED_GATHER",
                "weights": selected_weights,
                "output_rows": GATE_UP_ROWS,
                "output_sha256": mixed_hashes.last(),
                "parity": mixed_parity,
                "fused_native_gather": false,
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "consume_native_router_selection",
                    "resolve_selected_full_persisted_fused_gate_up_bodies",
                    "split_fused_gate_up_body_into_gate_and_up_q4_g64_buffers",
                    "execute_native_q4_g64_gate_up_swiglu_per_selected_expert",
                    "apply_selected_router_weights_on_host",
                    "emit_bounded_gate_up_activation_outputs",
                ],
                "source_independent": true,
                "complete_expert": false,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_gate_up_swiglu_observed": true,
            "native_expert_gate_up_activation_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent Q4/G64 gate+up/SwiGLU activation for the ten selected persisted expert bodies, with host-side selected-weight gather. Down projection, complete expert activation, complete-model loading, complete-token runtime, Flash TPS, and EBPW remain untested; source-selection mismatch remains explicit.",
            "next_action": "add and validate source-independent down-projection bodies, then compose a bounded native expert output before attempting any protected complete-token Flash graph",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    fn run_expert_composition(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let input = deterministic_input(2560);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&input))?;
        let mut loaded = Vec::with_capacity(selected_ids.len());

        for expert_id in &selected_ids {
            let gate_up_body_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_BODY_E{expert_id}_R0_1280.json"
            ));
            let gate_up_kernel_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_KERNEL_E{expert_id}_R0_1280_PARITY.json"
            ));
            let down_body_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_DOWN_BODY_E{expert_id}_R0_2560.json"
            ));
            let down_kernel_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_DOWN_KERNEL_E{expert_id}_R0_2560_PARITY.json"
            ));

            let (gate_up_body, gate_up_body_receipt) = validate_body(&gate_up_body_receipt_path)?;
            if gate_up_body.expert_index != *expert_id
                || gate_up_body.row_start != 0
                || gate_up_body.rows != GATE_UP_BODY_ROWS
                || gate_up_body.columns != 2560
            {
                return Err(format!(
                    "gate/up body receipt for expert {expert_id} is not the full [1280,2560] body"
                )
                .into());
            }
            let (gate_up_kernel, gate_up_kernel_receipt_sha256) =
                validate_kernel(&gate_up_kernel_receipt_path, &gate_up_body)?;
            let expected_gate_input =
                sha256_bytes(&f32_bytes(&deterministic_input(gate_up_body.columns)));
            if gate_up_kernel
                .get("input")
                .and_then(|input| input.get("deterministic_sha256"))
                .and_then(Value::as_str)
                != Some(expected_gate_input.as_str())
            {
                return Err(format!(
                    "gate/up kernel input for expert {expert_id} does not match the deterministic 2560-value input"
                )
                .into());
            }

            let (down, down_body_receipt) = validate_down_body(&down_body_receipt_path)?;
            if down.expert_index != *expert_id
                || down.row_start != 0
                || down.rows != DOWN_BODY_ROWS
                || down.columns != DOWN_COLUMNS
            {
                return Err(format!(
                    "down body receipt for expert {expert_id} is not the full [2560,640] body"
                )
                .into());
            }
            let (down_kernel, down_kernel_receipt_sha256) =
                validate_down_kernel(&down_kernel_receipt_path, &down)?;
            let expected_down_input = sha256_bytes(&f32_bytes(&deterministic_input(down.columns)));
            if down_kernel
                .get("input")
                .and_then(|input| input.get("deterministic_sha256"))
                .and_then(Value::as_str)
                != Some(expected_down_input.as_str())
            {
                return Err(format!(
                    "down kernel input for expert {expert_id} does not match the deterministic 640-value parity input"
                )
                .into());
            }

            let (gate, up) = split_gate_up(&gate_up_body)?;
            let expected_gate_up = cpu_gate_up_swiglu(&gate, &up, &input);
            let expected_down = cpu_matvec(&down, &expected_gate_up);
            loaded.push(LoadedComposition {
                gate_codes: context.new_buffer_with_bytes_checked(&gate.codes)?,
                gate_scales: context.new_buffer_with_bytes_checked(&gate.scales)?,
                up_codes: context.new_buffer_with_bytes_checked(&up.codes)?,
                up_scales: context.new_buffer_with_bytes_checked(&up.scales)?,
                gate_up_output: context
                    .new_buffer_checked(GATE_UP_ROWS * std::mem::size_of::<f32>())?,
                down_codes: context.new_buffer_with_bytes_checked(&down.codes)?,
                down_scales: context.new_buffer_with_bytes_checked(&down.scales)?,
                down_output: context
                    .new_buffer_checked(DOWN_BODY_ROWS * std::mem::size_of::<f32>())?,
                gate_up_body,
                gate,
                gate_up_body_receipt,
                gate_up_body_receipt_path: gate_up_body_receipt_path.canonicalize()?,
                gate_up_kernel_receipt: gate_up_kernel,
                gate_up_kernel_receipt_path: gate_up_kernel_receipt_path.canonicalize()?,
                gate_up_kernel_receipt_sha256,
                down,
                down_body_receipt,
                down_body_receipt_path: down_body_receipt_path.canonicalize()?,
                down_kernel_receipt: down_kernel,
                down_kernel_receipt_path: down_kernel_receipt_path.canonicalize()?,
                down_kernel_receipt_sha256,
                expected_gate_up,
                expected_down,
            });
        }

        if loaded.is_empty() {
            return Err("no selected experts were available for native composition".into());
        }

        let mut warmup_route_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let mut total = 0u64;
            for item in &loaded {
                total = total.saturating_add(gpu_ns(dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.gate_up_output,
                )?)?);
                total = total.saturating_add(gpu_ns(dispatch(
                    &context,
                    &item.down,
                    &item.down_codes,
                    &item.down_scales,
                    &item.gate_up_output,
                    &item.down_output,
                )?)?);
            }
            warmup_route_gpu_ns.push(total);
        }

        let mut per_expert_gate_up_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_down_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_total_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_host_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_gate_up_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_down_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut route_gpu_ns = Vec::with_capacity(args.reps);
        let mut route_host_ns = Vec::with_capacity(args.reps);
        let mut mixed_hashes = Vec::with_capacity(args.reps);
        let mut last_gate_up_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        let mut last_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        for _ in 0..args.reps {
            let mut total_gpu = 0u64;
            let mut total_host = 0u64;
            let mut mixed = vec![0.0f32; DOWN_BODY_ROWS];
            for (index, item) in loaded.iter().enumerate() {
                let gate_up_timing = dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.gate_up_output,
                )?;
                let down_timing = dispatch(
                    &context,
                    &item.down,
                    &item.down_codes,
                    &item.down_scales,
                    &item.gate_up_output,
                    &item.down_output,
                )?;
                let observed_gate_up = read_f32(&item.gate_up_output, GATE_UP_ROWS);
                let observed_down = read_f32(&item.down_output, DOWN_BODY_ROWS);
                if !observed_gate_up.iter().all(|value| value.is_finite())
                    || !observed_down.iter().all(|value| value.is_finite())
                {
                    return Err(format!(
                        "native expert composition output for expert {} is non-finite",
                        item.gate_up_body.expert_index
                    )
                    .into());
                }
                let gate_up_parity = output_metrics(&item.expected_gate_up, &observed_gate_up);
                if gate_up_parity
                    .get("within_tolerance")
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    return Err(format!(
                        "native composition gate/up parity failed for expert {}: {gate_up_parity}",
                        item.gate_up_body.expert_index
                    )
                    .into());
                }
                let down_parity = output_metrics(&item.expected_down, &observed_down);
                if down_parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(format!(
                        "native composition down parity failed for expert {}: {down_parity}",
                        item.gate_up_body.expert_index
                    )
                    .into());
                }
                let gate_up_gpu = gpu_ns(gate_up_timing)?;
                let down_gpu = gpu_ns(down_timing)?;
                let total_expert_gpu = gate_up_gpu.saturating_add(down_gpu);
                let host = gate_up_timing
                    .host_wall_us
                    .saturating_add(down_timing.host_wall_us)
                    .saturating_mul(1000);
                total_gpu = total_gpu.saturating_add(total_expert_gpu);
                total_host = total_host.saturating_add(host);
                per_expert_gate_up_gpu_ns[index].push(gate_up_gpu);
                per_expert_down_gpu_ns[index].push(down_gpu);
                per_expert_total_gpu_ns[index].push(total_expert_gpu);
                per_expert_host_ns[index].push(host);
                per_expert_gate_up_hashes[index]
                    .push(sha256_bytes(&f32_bytes_for_hash(&observed_gate_up)));
                per_expert_down_hashes[index]
                    .push(sha256_bytes(&f32_bytes_for_hash(&observed_down)));
                for (slot, value) in observed_down.iter().enumerate() {
                    mixed[slot] += selected_weights[index] * value;
                }
                last_gate_up_outputs[index] = observed_gate_up;
                last_outputs[index] = observed_down;
            }
            route_gpu_ns.push(total_gpu);
            route_host_ns.push(total_host);
            mixed_hashes.push(sha256_bytes(&f32_bytes_for_hash(&mixed)));
        }

        if per_expert_gate_up_hashes
            .iter()
            .chain(per_expert_down_hashes.iter())
            .any(|hashes| hashes.windows(2).any(|pair| pair[0] != pair[1]))
            || mixed_hashes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err(
                "native expert composition outputs changed across repeated executions".into(),
            );
        }

        let mut expected_mixed = vec![0.0f32; DOWN_BODY_ROWS];
        let mut observed_mixed = vec![0.0f32; DOWN_BODY_ROWS];
        for (index, item) in loaded.iter().enumerate() {
            for (slot, value) in item.expected_down.iter().enumerate() {
                expected_mixed[slot] += selected_weights[index] * value;
            }
            for (slot, value) in last_outputs[index].iter().enumerate() {
                observed_mixed[slot] += selected_weights[index] * value;
            }
        }
        let mixed_parity = output_metrics(&expected_mixed, &observed_mixed);
        if mixed_parity
            .get("within_tolerance")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(
                format!("weighted expert composition parity failed: {mixed_parity}").into(),
            );
        }

        let source_selection_parity = router
            .get("source_selection_parity")
            .cloned()
            .unwrap_or_else(|| json!({"status": "UNKNOWN"}));
        let source_mismatch = source_selection_parity
            .get("expert_ids_exact_match")
            .and_then(Value::as_bool)
            == Some(false);
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "native router receipt -> selected full persisted gate_up bodies -> native gate_up_swiglu -> device-resident 640-value activation -> selected full persisted down bodies -> native Q4/G64 down projection -> host weighted gather; no complete token graph",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_gate_up_kernel": GATE_UP_KERNEL_NAME,
            "native_down_kernel": KERNEL_NAME,
            "native_kernel_execution_observed": true,
            "native_gate_up_swiglu_observed": true,
            "native_down_projection_observed": true,
            "native_expert_composition_observed": true,
            "device_intermediate_no_host_roundtrip": true,
            "selected_expert_count": loaded.len(),
            "dispatches_per_route": loaded.len() * 2,
            "gate_up_rows": GATE_UP_ROWS,
            "down_rows": DOWN_BODY_ROWS,
            "down_columns": DOWN_COLUMNS,
            "source_selection_parity_status": source_selection_parity.get("status"),
            "source_selection_parity_qualified": source_selection_parity.get("expert_ids_exact_match"),
            "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let mut components_json = Vec::with_capacity(loaded.len());
        for (index, item) in loaded.iter().enumerate() {
            components_json.push(json!({
                "expert_index": item.gate_up_body.expert_index,
                "gate_up": {
                    "body_receipt": component_ref(&item.gate_up_body_receipt_path, &item.gate_up_body_receipt, Some(&item.gate_up_body.receipt_sha256)),
                    "kernel_receipt": component_ref(&item.gate_up_kernel_receipt_path, &item.gate_up_kernel_receipt, Some(&item.gate_up_kernel_receipt_sha256)),
                    "candidate_body": {
                        "path": item.gate_up_body.path,
                        "sha256": item.gate_up_body.body_sha256,
                        "bytes": item.gate_up_body.code_bytes + item.gate_up_body.scale_bytes,
                        "source_independent": true,
                        "candidate_body_persisted": true,
                        "layout": "fused gate_up body split into contiguous gate/up row halves",
                        "label": "[D]",
                    },
                    "native_dispatch": {
                        "kernel": GATE_UP_KERNEL_NAME,
                        "gpu_ns": per_expert_gate_up_gpu_ns[index],
                        "gpu_ns_median": percentile_median(&per_expert_gate_up_gpu_ns[index]),
                        "output_hashes": per_expert_gate_up_hashes[index],
                        "stable_output": true,
                    },
                    "parity": output_metrics(&item.expected_gate_up, &last_gate_up_outputs[index]),
                },
                "down": {
                    "body_receipt": component_ref(&item.down_body_receipt_path, &item.down_body_receipt, Some(&item.down.receipt_sha256)),
                    "kernel_receipt": component_ref(&item.down_kernel_receipt_path, &item.down_kernel_receipt, Some(&item.down_kernel_receipt_sha256)),
                    "candidate_body": {
                        "path": item.down.path,
                        "sha256": item.down.body_sha256,
                        "bytes": item.down.code_bytes + item.down.scale_bytes,
                        "source_independent": true,
                        "candidate_body_persisted": true,
                        "label": "[D]",
                    },
                    "native_dispatch": {
                        "kernel": KERNEL_NAME,
                        "gpu_ns": per_expert_down_gpu_ns[index],
                        "gpu_ns_median": percentile_median(&per_expert_down_gpu_ns[index]),
                        "output_hashes": per_expert_down_hashes[index],
                        "stable_output": true,
                    },
                    "parity": output_metrics(&item.expected_down, &last_outputs[index]),
                },
                "native_dispatch": {
                    "dispatches_per_route": 2,
                    "gpu_ns": per_expert_total_gpu_ns[index],
                    "gpu_ns_median": percentile_median(&per_expert_total_gpu_ns[index]),
                    "host_wall_ns": per_expert_host_ns[index],
                    "host_wall_ns_median": percentile_median(&per_expert_host_ns[index]),
                    "device_intermediate": "gate_up_swiglu_output_buffer -> down_projection_input",
                    "host_intermediate_materialization": false,
                },
                "source_reference_used_for_execution": false,
            }));
        }

        let route_gpu_median = percentile_median(&route_gpu_ns);
        let route_host_median = percentile_median(&route_host_ns);
        Ok(json!({
            "schema": COMPOSITION_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_GATE_UP_DOWN_COMPOSITION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "router_receipt": {
                "path": args.router_receipt.canonicalize()?,
                "sha256": router_sha256,
                "schema": ROUTER_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "component_receipt_policy": {
                "directory": repo.join("receipts/headless"),
                "gate_up_body_pattern": "FLASH_NOETIC_GATE_UP_BODY_E{expert}_R0_1280.json",
                "gate_up_kernel_pattern": "FLASH_NOETIC_GATE_UP_KERNEL_E{expert}_R0_1280_PARITY.json",
                "down_body_pattern": "FLASH_NOETIC_DOWN_BODY_E{expert}_R0_2560.json",
                "down_kernel_pattern": "FLASH_NOETIC_DOWN_KERNEL_E{expert}_R0_2560_PARITY.json",
                "gate_up_body_rows": GATE_UP_BODY_ROWS,
                "gate_up_rows": GATE_UP_ROWS,
                "down_body_rows": DOWN_BODY_ROWS,
                "down_columns": DOWN_COLUMNS,
            },
            "selection": router.get("selection").cloned().unwrap_or_else(|| json!({})),
            "source_selection_parity": source_selection_parity,
            "components": components_json,
            "execution": {
                "provider": "apple-metal",
                "operation": "native_router_selection_receipt -> full persisted gate_up body load -> native Q4_G64 gate_up_swiglu -> device-resident gate/up activation buffer -> full persisted down body load -> native Q4_G64 down_projection -> host_selected_weight_gather",
                "selected_expert_ids": selected_ids,
                "selected_expert_count": loaded.len(),
                "dispatches_per_expert": 2,
                "dispatches_per_route": loaded.len() * 2,
                "measured_routes": args.reps,
                "total_measured_dispatches": loaded.len() * args.reps * 2,
                "native_gate_up_swiglu_observed": true,
                "native_down_projection_observed": true,
                "native_expert_composition_observed": true,
                "bounded_selected_expert_output_observed": true,
                "device_intermediate_no_host_roundtrip": true,
                "source_reference_used_for_execution": false,
                "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
                "body_mutated": false,
                "model_loaded": false,
                "complete_expert_activation": false,
                "complete_token_runtime": false,
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": input.len(),
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "intermediate": {
                "semantic_type": "NoeticActivationBuffer",
                "shape": [GATE_UP_ROWS],
                "dtype": "F32",
                "producer": GATE_UP_KERNEL_NAME,
                "consumer": KERNEL_NAME,
                "device_resident": true,
                "host_roundtrip": false,
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_route_gpu_ns": warmup_route_gpu_ns,
                "route_gpu_ns": route_gpu_ns,
                "route_gpu_ns_median": route_gpu_median,
                "route_host_wall_ns": route_host_ns,
                "route_host_wall_ns_median": route_host_median,
                "dispatches_per_route": loaded.len() * 2,
                "output_hashes": mixed_hashes,
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for native gate_up_swiglu plus down projection dispatches; host wall reported separately",
            },
            "gather": {
                "status": "BOUNDED_HOST_WEIGHTED_GATHER",
                "weights": selected_weights,
                "output_rows": DOWN_BODY_ROWS,
                "output_sha256": mixed_hashes.last(),
                "parity": mixed_parity,
                "fused_native_gather": false,
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "consume_native_router_selection",
                    "resolve_selected_full_persisted_fused_gate_up_bodies",
                    "execute_native_q4_g64_gate_up_swiglu_per_selected_expert",
                    "retain_gate_up_activation_in_device_buffer",
                    "resolve_selected_full_persisted_down_projection_bodies",
                    "execute_native_q4_g64_down_projection_per_selected_expert",
                    "apply_selected_router_weights_on_host",
                    "emit_bounded_selected_expert_outputs",
                ],
                "source_independent": true,
                "complete_expert": false,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_gate_up_swiglu_observed": true,
            "native_down_projection_observed": true,
            "native_expert_composition_observed": true,
            "bounded_selected_expert_output_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent composition of the ten selected persisted routed experts: Q4/G64 gate+up/SwiGLU writes a device-resident 640-value activation consumed directly by Q4/G64 down projection, followed by a host-side selected-weight gather. This is not a complete Flash model, complete-token runtime, Flash TPS, or EBPW result; source-selection mismatch remains explicit.",
            "next_action": "qualify the remaining Flash graph and protected complete-token capability; do not populate Flash TPS or EBPW until the full native token runtime is independently observed",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    fn run_shared_residual_composition(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;

        let prior_shared_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_COMPOSITION_NATIVE.json");
        let (prior_shared_receipt, prior_shared_receipt_sha256) =
            read_json(&prior_shared_receipt_path)?;
        if string_field(&prior_shared_receipt, "schema")? != SHARED_EXPERT_SCHEMA
            || string_field(&prior_shared_receipt, "status")? != "PASSED"
            || prior_shared_receipt
                .get("native_shared_expert_composition_observed")
                .and_then(Value::as_bool)
                != Some(true)
            || prior_shared_receipt
                .get("promotion_allowed")
                .and_then(Value::as_bool)
                != Some(false)
        {
            return Err(
                "prior shared-expert receipt is not a PASSED bounded non-promoted graph".into(),
            );
        }

        let gate_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_GATE_BODY_L0_R0_640.json");
        let up_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_UP_BODY_L0_R0_640.json");
        let down_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_DOWN_BODY_L0_R0_2560.json");
        let scalar_gate_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_SCALAR_GATE_BODY_L0_R0_1.json");
        let input_down_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_INPUT_DOWN_BODY_L0_R0_320.json");
        let input_up_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_INPUT_UP_BODY_L0_R0_10240.json");
        let block_inject_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_MLP_HYPER_BLOCK_INJECT_BODY_L0_R0_4.json");

        let (gate, gate_receipt) = validate_matrix_body_for(
            &gate_receipt_path,
            SHARED_GATE_TENSOR_NAME,
            [SHARED_INTERMEDIATE, SHARED_HIDDEN],
        )?;
        let (up, up_receipt) = validate_matrix_body_for(
            &up_receipt_path,
            SHARED_UP_TENSOR_NAME,
            [SHARED_INTERMEDIATE, SHARED_HIDDEN],
        )?;
        let (down, down_receipt) = validate_matrix_body_for(
            &down_receipt_path,
            SHARED_DOWN_TENSOR_NAME,
            [SHARED_HIDDEN, SHARED_INTERMEDIATE],
        )?;
        let (scalar_gate, scalar_gate_receipt) = validate_matrix_body_for(
            &scalar_gate_receipt_path,
            SHARED_SCALAR_GATE_TENSOR_NAME,
            [1, SHARED_HIDDEN],
        )?;
        let (input_down, input_down_receipt) = validate_matrix_body_for(
            &input_down_receipt_path,
            HYPER_INPUT_DOWN_TENSOR_NAME,
            [HYPER_LOWRANK, HYPER_STATE_ELEMENTS],
        )?;
        let (input_up, input_up_receipt) = validate_matrix_body_for(
            &input_up_receipt_path,
            HYPER_INPUT_UP_TENSOR_NAME,
            [HYPER_STATE_ELEMENTS, HYPER_LOWRANK],
        )?;
        let (block_inject, block_inject_receipt) = validate_matrix_body_for(
            &block_inject_receipt_path,
            HYPER_BLOCK_INJECT_TENSOR_NAME,
            [HYPER_STATE_STREAMS, HYPER_STATE_ELEMENTS],
        )?;

        if gate.row_start != 0
            || gate.rows != SHARED_INTERMEDIATE
            || up.row_start != 0
            || up.rows != SHARED_INTERMEDIATE
            || down.row_start != 0
            || down.rows != SHARED_HIDDEN
            || scalar_gate.row_start != 0
            || scalar_gate.rows != 1
            || input_down.row_start != 0
            || input_down.rows != HYPER_LOWRANK
            || input_up.row_start != 0
            || input_up.rows != HYPER_STATE_ELEMENTS
            || block_inject.row_start != 0
            || block_inject.rows != HYPER_STATE_STREAMS
        {
            return Err("shared/residual bodies are not complete bounded layer-0 windows".into());
        }

        let shared_input = deterministic_input(SHARED_HIDDEN);
        let shared_input_sha256 = sha256_bytes(&f32_bytes(&shared_input));
        let expected_gate_up = cpu_gate_up_swiglu(&gate, &up, &shared_input);
        let expected_shared_down = cpu_matvec(&down, &expected_gate_up);
        let expected_scalar_gate = cpu_matvec(&scalar_gate, &shared_input);
        if expected_scalar_gate.len() != 1 {
            return Err("shared-expert scalar gate did not produce one logit".into());
        }
        let expected_shared_output: Vec<f32> = expected_shared_down
            .iter()
            .map(|value| value * sigmoid(expected_scalar_gate[0]))
            .collect();

        let base_state = deterministic_input(HYPER_STATE_ELEMENTS);
        let base_state_sha256 = sha256_bytes(&f32_bytes(&base_state));
        let mut expected_hyper_state = base_state.clone();
        expected_hyper_state[HYPER_STATE_HIDDEN..(HYPER_STATE_HIDDEN * 2)]
            .copy_from_slice(&expected_shared_output);
        let expected_low_rank = cpu_matvec(&input_down, &expected_hyper_state);
        let expected_correction = cpu_matvec(&input_up, &expected_low_rank);
        let expected_block_logits = cpu_matvec(&block_inject, &expected_hyper_state);
        if expected_block_logits.len() != HYPER_STATE_STREAMS {
            return Err("hyperconnection block inject did not produce four logits".into());
        }
        let expected_residual_output: Vec<f32> = expected_hyper_state
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    + expected_correction[index]
                        * sigmoid(expected_block_logits[index / HYPER_STATE_HIDDEN])
            })
            .collect();

        let context = MetalContext::new_with_trace(true)?;
        let shared_input_buffer =
            context.new_buffer_with_bytes_checked(&f32_bytes(&shared_input))?;
        let base_state_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&base_state))?;
        let gate_codes = context.new_buffer_with_bytes_checked(&gate.codes)?;
        let gate_scales = context.new_buffer_with_bytes_checked(&gate.scales)?;
        let up_codes = context.new_buffer_with_bytes_checked(&up.codes)?;
        let up_scales = context.new_buffer_with_bytes_checked(&up.scales)?;
        let gate_up_output =
            context.new_buffer_checked(SHARED_INTERMEDIATE * std::mem::size_of::<f32>())?;
        let down_codes = context.new_buffer_with_bytes_checked(&down.codes)?;
        let down_scales = context.new_buffer_with_bytes_checked(&down.scales)?;
        let shared_down_output =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;
        let scalar_gate_codes = context.new_buffer_with_bytes_checked(&scalar_gate.codes)?;
        let scalar_gate_scales = context.new_buffer_with_bytes_checked(&scalar_gate.scales)?;
        let scalar_gate_output = context.new_buffer_checked(std::mem::size_of::<f32>())?;
        let shared_output =
            context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?;

        let input_down_codes = context.new_buffer_with_bytes_checked(&input_down.codes)?;
        let input_down_scales = context.new_buffer_with_bytes_checked(&input_down.scales)?;
        let input_up_codes = context.new_buffer_with_bytes_checked(&input_up.codes)?;
        let input_up_scales = context.new_buffer_with_bytes_checked(&input_up.scales)?;
        let block_inject_codes = context.new_buffer_with_bytes_checked(&block_inject.codes)?;
        let block_inject_scales = context.new_buffer_with_bytes_checked(&block_inject.scales)?;
        let hyper_state =
            context.new_buffer_checked(HYPER_STATE_ELEMENTS * std::mem::size_of::<f32>())?;
        let low_rank_output =
            context.new_buffer_checked(HYPER_LOWRANK * std::mem::size_of::<f32>())?;
        let correction_output =
            context.new_buffer_checked(HYPER_STATE_ELEMENTS * std::mem::size_of::<f32>())?;
        let block_logits_output =
            context.new_buffer_checked(HYPER_STATE_STREAMS * std::mem::size_of::<f32>())?;
        let residual_output =
            context.new_buffer_checked(HYPER_STATE_ELEMENTS * std::mem::size_of::<f32>())?;

        let dispatch_graph = || -> Result<Vec<MetalDispatchTiming>, Box<dyn Error>> {
            let mut timings = Vec::with_capacity(9);
            timings.push(dispatch_gate_up_swiglu(
                &context,
                &gate,
                &gate_codes,
                &gate_scales,
                &up_codes,
                &up_scales,
                &shared_input_buffer,
                &gate_up_output,
            )?);
            timings.push(dispatch(
                &context,
                &down,
                &down_codes,
                &down_scales,
                &gate_up_output,
                &shared_down_output,
            )?);
            timings.push(dispatch(
                &context,
                &scalar_gate,
                &scalar_gate_codes,
                &scalar_gate_scales,
                &shared_input_buffer,
                &scalar_gate_output,
            )?);
            timings.push(dispatch_shared_expert_sigmoid_gate(
                &context,
                &shared_down_output,
                &scalar_gate_output,
                &shared_output,
                SHARED_HIDDEN,
            )?);
            timings.push(dispatch_expand_shared_to_hyper_state(
                &context,
                &base_state_buffer,
                &shared_output,
                &hyper_state,
            )?);
            timings.push(dispatch(
                &context,
                &input_down,
                &input_down_codes,
                &input_down_scales,
                &hyper_state,
                &low_rank_output,
            )?);
            timings.push(dispatch(
                &context,
                &input_up,
                &input_up_codes,
                &input_up_scales,
                &low_rank_output,
                &correction_output,
            )?);
            timings.push(dispatch(
                &context,
                &block_inject,
                &block_inject_codes,
                &block_inject_scales,
                &hyper_state,
                &block_logits_output,
            )?);
            timings.push(dispatch_hyperconnection_residual_mix(
                &context,
                &hyper_state,
                &correction_output,
                &block_logits_output,
                &residual_output,
            )?);
            Ok(timings)
        };

        let sum_gpu_ns = |timings: &[MetalDispatchTiming]| -> Result<u64, Box<dyn Error>> {
            timings.iter().try_fold(0u64, |total, timing| {
                Ok::<u64, Box<dyn Error>>(total.saturating_add(gpu_ns(*timing)?))
            })
        };
        let sum_host_ns = |timings: &[MetalDispatchTiming]| -> u64 {
            timings
                .iter()
                .map(|timing| timing.host_wall_us.saturating_mul(1000))
                .fold(0u64, u64::saturating_add)
        };

        let mut warmup_graph_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            warmup_graph_gpu_ns.push(sum_gpu_ns(&dispatch_graph()?)?);
        }

        let mut shared_gate_up_gpu_ns = Vec::with_capacity(args.reps);
        let mut shared_down_gpu_ns = Vec::with_capacity(args.reps);
        let mut shared_scalar_gate_gpu_ns = Vec::with_capacity(args.reps);
        let mut shared_sigmoid_gate_gpu_ns = Vec::with_capacity(args.reps);
        let mut expand_gpu_ns = Vec::with_capacity(args.reps);
        let mut low_rank_down_gpu_ns = Vec::with_capacity(args.reps);
        let mut low_rank_up_gpu_ns = Vec::with_capacity(args.reps);
        let mut block_inject_gpu_ns = Vec::with_capacity(args.reps);
        let mut residual_mix_gpu_ns = Vec::with_capacity(args.reps);
        let mut graph_gpu_ns = Vec::with_capacity(args.reps);
        let mut graph_host_ns = Vec::with_capacity(args.reps);
        let mut shared_output_hashes = Vec::with_capacity(args.reps);
        let mut hyper_state_hashes = Vec::with_capacity(args.reps);
        let mut low_rank_hashes = Vec::with_capacity(args.reps);
        let mut correction_hashes = Vec::with_capacity(args.reps);
        let mut block_logits_hashes = Vec::with_capacity(args.reps);
        let mut residual_output_hashes = Vec::with_capacity(args.reps);
        let mut last_shared_output_parity = json!({});
        let mut last_hyper_state_parity = json!({});
        let mut last_low_rank_parity = json!({});
        let mut last_correction_parity = json!({});
        let mut last_block_logits_parity = json!({});
        let mut last_residual_output_parity = json!({});

        for _ in 0..args.reps {
            let timings = dispatch_graph()?;
            let observed_shared_output = read_f32(&shared_output, SHARED_HIDDEN);
            let observed_hyper_state = read_f32(&hyper_state, HYPER_STATE_ELEMENTS);
            let observed_low_rank = read_f32(&low_rank_output, HYPER_LOWRANK);
            let observed_correction = read_f32(&correction_output, HYPER_STATE_ELEMENTS);
            let observed_block_logits = read_f32(&block_logits_output, HYPER_STATE_STREAMS);
            let observed_residual_output = read_f32(&residual_output, HYPER_STATE_ELEMENTS);
            if !observed_shared_output.iter().all(|value| value.is_finite())
                || !observed_hyper_state.iter().all(|value| value.is_finite())
                || !observed_low_rank.iter().all(|value| value.is_finite())
                || !observed_correction.iter().all(|value| value.is_finite())
                || !observed_block_logits.iter().all(|value| value.is_finite())
                || !observed_residual_output
                    .iter()
                    .all(|value| value.is_finite())
            {
                return Err(
                    "native shared/residual composition produced a non-finite value".into(),
                );
            }
            last_shared_output_parity =
                output_metrics(&expected_shared_output, &observed_shared_output);
            last_hyper_state_parity = output_metrics(&expected_hyper_state, &observed_hyper_state);
            last_low_rank_parity = output_metrics(&expected_low_rank, &observed_low_rank);
            last_correction_parity = output_metrics(&expected_correction, &observed_correction);
            last_block_logits_parity =
                output_metrics(&expected_block_logits, &observed_block_logits);
            last_residual_output_parity =
                output_metrics(&expected_residual_output, &observed_residual_output);
            for (label, parity) in [
                ("shared output", &last_shared_output_parity),
                ("hyper state", &last_hyper_state_parity),
                ("low-rank down", &last_low_rank_parity),
                ("low-rank correction", &last_correction_parity),
                ("block inject", &last_block_logits_parity),
                ("residual output", &last_residual_output_parity),
            ] {
                if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(
                        format!("native shared/residual {label} parity failed: {parity}").into(),
                    );
                }
            }
            let gpu_values: Vec<u64> = timings
                .iter()
                .map(|timing| gpu_ns(*timing))
                .collect::<Result<_, _>>()?;
            shared_gate_up_gpu_ns.push(gpu_values[0]);
            shared_down_gpu_ns.push(gpu_values[1]);
            shared_scalar_gate_gpu_ns.push(gpu_values[2]);
            shared_sigmoid_gate_gpu_ns.push(gpu_values[3]);
            expand_gpu_ns.push(gpu_values[4]);
            low_rank_down_gpu_ns.push(gpu_values[5]);
            low_rank_up_gpu_ns.push(gpu_values[6]);
            block_inject_gpu_ns.push(gpu_values[7]);
            residual_mix_gpu_ns.push(gpu_values[8]);
            graph_gpu_ns.push(gpu_values.iter().copied().fold(0u64, u64::saturating_add));
            graph_host_ns.push(sum_host_ns(&timings));
            shared_output_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_shared_output)));
            hyper_state_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_hyper_state)));
            low_rank_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_low_rank)));
            correction_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_correction)));
            block_logits_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_block_logits)));
            residual_output_hashes
                .push(sha256_bytes(&f32_bytes_for_hash(&observed_residual_output)));
        }

        for hashes in [
            &shared_output_hashes,
            &hyper_state_hashes,
            &low_rank_hashes,
            &correction_hashes,
            &block_logits_hashes,
            &residual_output_hashes,
        ] {
            if hashes.windows(2).any(|pair| pair[0] != pair[1]) {
                return Err(
                    "native shared/residual outputs changed across repeated executions".into(),
                );
            }
        }

        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "layer-0 shared expert output injected into a four-stream MLP hyperconnection candidate: shared gate/up/SwiGLU -> down -> sigmoid gate -> device-side stream injection -> low-rank down/up -> block-inject gated residual mix",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "nodes": [
                {"id": "shared_input", "shape": [SHARED_HIDDEN], "dtype": "F32", "device_resident": true},
                {"id": "shared_gate_proj", "shape": [SHARED_INTERMEDIATE, SHARED_HIDDEN], "representation": "independent_q4_g64"},
                {"id": "shared_up_proj", "shape": [SHARED_INTERMEDIATE, SHARED_HIDDEN], "representation": "independent_q4_g64"},
                {"id": "shared_gate_up_swiglu", "shape": [SHARED_INTERMEDIATE], "kernel": GATE_UP_KERNEL_NAME, "device_resident": true},
                {"id": "shared_down_proj", "shape": [SHARED_HIDDEN, SHARED_INTERMEDIATE], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "shared_expert_gate", "shape": [1, SHARED_HIDDEN], "representation": "independent_q4_g64"},
                {"id": "shared_sigmoid_gate", "shape": [SHARED_HIDDEN], "kernel": SHARED_SIGMOID_GATE_KERNEL_NAME, "device_resident": true},
                {"id": "base_hyper_state", "shape": [HYPER_STATE_ELEMENTS], "dtype": "F32", "device_resident": true},
                {"id": "inject_shared_output", "shape": [HYPER_STATE_ELEMENTS], "kernel": HYPER_EXPAND_KERNEL_NAME, "device_resident": true},
                {"id": "hyper_input_mix_down", "shape": [HYPER_LOWRANK, HYPER_STATE_ELEMENTS], "representation": "independent_q4_g64"},
                {"id": "hyper_low_rank_state", "shape": [HYPER_LOWRANK], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "hyper_input_mix_up", "shape": [HYPER_STATE_ELEMENTS, HYPER_LOWRANK], "representation": "independent_q4_g64"},
                {"id": "hyper_correction", "shape": [HYPER_STATE_ELEMENTS], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "hyper_block_inject", "shape": [HYPER_STATE_STREAMS, HYPER_STATE_ELEMENTS], "representation": "independent_q4_g64"},
                {"id": "hyper_block_logits", "shape": [HYPER_STATE_STREAMS], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "hyper_residual_mix", "shape": [HYPER_STATE_ELEMENTS], "kernel": HYPER_RESIDUAL_MIX_KERNEL_NAME, "device_resident": true}
            ],
            "edges": [
                ["shared_input", "shared_gate_proj"],
                ["shared_input", "shared_up_proj"],
                ["shared_gate_proj", "shared_gate_up_swiglu"],
                ["shared_up_proj", "shared_gate_up_swiglu"],
                ["shared_gate_up_swiglu", "shared_down_proj"],
                ["shared_input", "shared_expert_gate"],
                ["shared_down_proj", "shared_sigmoid_gate"],
                ["shared_expert_gate", "shared_sigmoid_gate"],
                ["shared_sigmoid_gate", "inject_shared_output"],
                ["base_hyper_state", "inject_shared_output"],
                ["inject_shared_output", "hyper_input_mix_down"],
                ["hyper_input_mix_down", "hyper_low_rank_state"],
                ["hyper_low_rank_state", "hyper_input_mix_up"],
                ["hyper_input_mix_up", "hyper_correction"],
                ["inject_shared_output", "hyper_block_inject"],
                ["hyper_block_inject", "hyper_block_logits"],
                ["inject_shared_output", "hyper_residual_mix"],
                ["hyper_correction", "hyper_residual_mix"],
                ["hyper_block_logits", "hyper_residual_mix"]
            ],
            "native_kernels": [GATE_UP_KERNEL_NAME, KERNEL_NAME, SHARED_SIGMOID_GATE_KERNEL_NAME, HYPER_EXPAND_KERNEL_NAME, HYPER_RESIDUAL_MIX_KERNEL_NAME],
            "native_kernel_execution_observed": true,
            "dispatches_per_graph": 9,
            "device_intermediate_no_host_roundtrip": true,
            "verification_reads_after_graph": true,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let body_ref = |path: &Path, receipt: &Value| -> Result<Value, Box<dyn Error>> {
            let digest = sha256_bytes(&fs::read(path)?);
            Ok(component_ref(path, receipt, Some(&digest)))
        };
        let device_memory = context.device_memory_limits();
        let file_name = |path: &Path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned()
        };
        Ok(json!({
            "schema": SHARED_RESIDUAL_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_SHARED_EXPERT_RESIDUAL_HYPERCONNECTION_COMPOSITION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "layer": 0,
            "dependencies": {
                "prior_shared_expert_composition_receipt": {"path": prior_shared_receipt_path, "sha256": prior_shared_receipt_sha256, "status": "PASSED", "promotion_allowed": false},
            },
            "component_receipt_policy": {
                "directory": repo.join("receipts/headless"),
                "shared_gate_body": file_name(&gate_receipt_path),
                "shared_up_body": file_name(&up_receipt_path),
                "shared_down_body": file_name(&down_receipt_path),
                "shared_scalar_gate_body": file_name(&scalar_gate_receipt_path),
                "hyper_input_down_body": file_name(&input_down_receipt_path),
                "hyper_input_up_body": file_name(&input_up_receipt_path),
                "hyper_block_inject_body": file_name(&block_inject_receipt_path),
            },
            "components": {
                "shared_gate_proj": {"body_receipt": body_ref(&gate_receipt_path, &gate_receipt)?, "candidate_body": {"path": gate.path, "sha256": gate.body_sha256, "bytes": gate.code_bytes + gate.scale_bytes, "shape": [gate.rows, gate.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
                "shared_up_proj": {"body_receipt": body_ref(&up_receipt_path, &up_receipt)?, "candidate_body": {"path": up.path, "sha256": up.body_sha256, "bytes": up.code_bytes + up.scale_bytes, "shape": [up.rows, up.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
                "shared_down_proj": {"body_receipt": body_ref(&down_receipt_path, &down_receipt)?, "candidate_body": {"path": down.path, "sha256": down.body_sha256, "bytes": down.code_bytes + down.scale_bytes, "shape": [down.rows, down.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
                "shared_expert_gate": {"body_receipt": body_ref(&scalar_gate_receipt_path, &scalar_gate_receipt)?, "candidate_body": {"path": scalar_gate.path, "sha256": scalar_gate.body_sha256, "bytes": scalar_gate.code_bytes + scalar_gate.scale_bytes, "shape": [scalar_gate.rows, scalar_gate.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
                "hyper_input_mix_down": {"body_receipt": body_ref(&input_down_receipt_path, &input_down_receipt)?, "candidate_body": {"path": input_down.path, "sha256": input_down.body_sha256, "bytes": input_down.code_bytes + input_down.scale_bytes, "shape": [input_down.rows, input_down.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
                "hyper_input_mix_up": {"body_receipt": body_ref(&input_up_receipt_path, &input_up_receipt)?, "candidate_body": {"path": input_up.path, "sha256": input_up.body_sha256, "bytes": input_up.code_bytes + input_up.scale_bytes, "shape": [input_up.rows, input_up.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
                "hyper_block_inject": {"body_receipt": body_ref(&block_inject_receipt_path, &block_inject_receipt)?, "candidate_body": {"path": block_inject.path, "sha256": block_inject.body_sha256, "bytes": block_inject.code_bytes + block_inject.scale_bytes, "shape": [block_inject.rows, block_inject.columns], "source_independent": true}, "native_kernel": KERNEL_NAME},
            },
            "execution": {
                "provider": "apple-metal",
                "operation": "shared expert output -> device-side stream-1 injection into 4x2560 state -> Q4/G64 low-rank input mix down/up -> Q4/G64 block inject logits -> per-stream sigmoid-gated residual mix",
                "shared_expert_hidden": SHARED_HIDDEN,
                "hyperconnection_hidden": HYPER_STATE_HIDDEN,
                "hyperconnection_streams": HYPER_STATE_STREAMS,
                "hyperconnection_elements": HYPER_STATE_ELEMENTS,
                "hyperconnection_lowrank": HYPER_LOWRANK,
                "injected_stream": 1,
                "dispatches_per_graph": 9,
                "measured_graphs": args.reps,
                "total_measured_dispatches": args.reps * 9,
                "native_shared_expert_gate_up_swiglu_observed": true,
                "native_shared_expert_down_projection_observed": true,
                "native_shared_expert_sigmoid_gate_observed": true,
                "native_hyperconnection_stream_injection_observed": true,
                "native_hyperconnection_low_rank_down_observed": true,
                "native_hyperconnection_low_rank_up_observed": true,
                "native_hyperconnection_block_inject_observed": true,
                "native_hyperconnection_residual_mix_observed": true,
                "native_shared_residual_composition_observed": true,
                "device_intermediate_no_host_roundtrip": true,
                "verification_reads_after_graph": true,
                "source_reference_used_for_execution": false,
                "body_mutated": false,
                "model_loaded": false,
                "hc_norm_loaded": false,
                "complete_moe_combine": false,
                "complete_token_runtime": false,
            },
            "input": {
                "shared_input": {"definition": "((index * 71) mod 509 - 254) / 509", "values": shared_input.len(), "deterministic_sha256": shared_input_sha256, "label": "[V]"},
                "base_hyper_state": {"definition": "((index * 71) mod 509 - 254) / 509", "values": base_state.len(), "deterministic_sha256": base_state_sha256, "label": "[V]"},
                "stream_layout": "four contiguous 2560-wide streams; stream 1 is replaced on-device by the gated shared-expert output",
            },
            "intermediates": {
                "shared_output": {"semantic_type": "NoeticActivationBuffer", "shape": [SHARED_HIDDEN], "dtype": "F32", "device_resident": true, "host_roundtrip": false},
                "hyper_state": {"semantic_type": "NoeticHyperconnectionState", "shape": [HYPER_STATE_ELEMENTS], "dtype": "F32", "device_resident": true, "host_roundtrip": false, "stream_count": HYPER_STATE_STREAMS},
                "low_rank_state": {"semantic_type": "NoeticLowRankActivation", "shape": [HYPER_LOWRANK], "dtype": "F32", "device_resident": true, "host_roundtrip": false},
                "correction": {"semantic_type": "NoeticResidualCorrection", "shape": [HYPER_STATE_ELEMENTS], "dtype": "F32", "device_resident": true, "host_roundtrip": false},
                "block_logits": {"semantic_type": "NoeticBlockInjectLogits", "shape": [HYPER_STATE_STREAMS], "dtype": "F32", "device_resident": true, "host_roundtrip": false},
                "residual_output": {"semantic_type": "NoeticResidualOutput", "shape": [HYPER_STATE_ELEMENTS], "dtype": "F32", "device_resident": true, "host_roundtrip": false},
            },
            "candidate_semantics": {
                "status": "BOUNDED_CANDIDATE_ONLY",
                "formula": "state + low_rank_up(low_rank_down(state)) * sigmoid(block_inject(state)[stream])",
                "injected_stream": 1,
                "hc_norm": "NOT_LOADED",
                "source_model_activation_parity": "NOT_TESTED",
            },
            "parity": {
                "candidate_space": "CPU reference uses the same persisted Q4/G64 candidate bodies and the same explicit candidate stream-injection/gating formula; source BF16 activation parity remains unqualified",
                "shared_output": last_shared_output_parity,
                "hyper_state": last_hyper_state_parity,
                "low_rank_down": last_low_rank_parity,
                "low_rank_up_correction": last_correction_parity,
                "block_inject": last_block_logits_parity,
                "residual_output": last_residual_output_parity,
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_graph_gpu_ns": warmup_graph_gpu_ns,
                "shared_gate_up_gpu_ns": shared_gate_up_gpu_ns,
                "shared_gate_up_gpu_ns_median": percentile_median(&shared_gate_up_gpu_ns),
                "shared_down_gpu_ns": shared_down_gpu_ns,
                "shared_down_gpu_ns_median": percentile_median(&shared_down_gpu_ns),
                "shared_scalar_gate_gpu_ns": shared_scalar_gate_gpu_ns,
                "shared_scalar_gate_gpu_ns_median": percentile_median(&shared_scalar_gate_gpu_ns),
                "shared_sigmoid_gate_gpu_ns": shared_sigmoid_gate_gpu_ns,
                "shared_sigmoid_gate_gpu_ns_median": percentile_median(&shared_sigmoid_gate_gpu_ns),
                "expand_gpu_ns": expand_gpu_ns,
                "expand_gpu_ns_median": percentile_median(&expand_gpu_ns),
                "low_rank_down_gpu_ns": low_rank_down_gpu_ns,
                "low_rank_down_gpu_ns_median": percentile_median(&low_rank_down_gpu_ns),
                "low_rank_up_gpu_ns": low_rank_up_gpu_ns,
                "low_rank_up_gpu_ns_median": percentile_median(&low_rank_up_gpu_ns),
                "block_inject_gpu_ns": block_inject_gpu_ns,
                "block_inject_gpu_ns_median": percentile_median(&block_inject_gpu_ns),
                "residual_mix_gpu_ns": residual_mix_gpu_ns,
                "residual_mix_gpu_ns_median": percentile_median(&residual_mix_gpu_ns),
                "graph_gpu_ns": graph_gpu_ns,
                "graph_gpu_ns_median": percentile_median(&graph_gpu_ns),
                "graph_host_wall_ns": graph_host_ns,
                "graph_host_wall_ns_median": percentile_median(&graph_host_ns),
                "dispatches_per_graph": 9,
                "output_hashes": residual_output_hashes,
                "stage_output_hashes": {"shared_output": shared_output_hashes, "hyper_state": hyper_state_hashes, "low_rank_state": low_rank_hashes, "correction": correction_hashes, "block_logits": block_logits_hashes, "residual_output": residual_output_hashes},
                "memory_limits": {"max_buffer_length": device_memory.max_buffer_length, "recommended_max_working_set_size": device_memory.recommended_max_working_set_size, "current_allocated_size": device_memory.current_allocated_size, "has_unified_memory": device_memory.has_unified_memory},
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for all nine native shared/residual dispatches; host wall is reported separately",
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "representation": "independent_q4_g64",
                "operations": [
                    "load_source_independent_shared_expert_gate_proj_body",
                    "load_source_independent_shared_expert_up_proj_body",
                    "execute_native_q4_g64_shared_gate_up_swiglu",
                    "execute_native_q4_g64_shared_down_projection",
                    "execute_native_q4_g64_shared_scalar_gate",
                    "execute_native_sigmoid_shared_expert_gate",
                    "inject_gated_shared_output_into_hyperconnection_stream_1_on_device",
                    "load_source_independent_mlp_hyperconnection_input_mix_down_body",
                    "execute_native_q4_g64_hyperconnection_low_rank_down",
                    "load_source_independent_mlp_hyperconnection_input_mix_up_body",
                    "execute_native_q4_g64_hyperconnection_low_rank_up",
                    "load_source_independent_mlp_hyperconnection_block_inject_body",
                    "execute_native_q4_g64_hyperconnection_block_inject",
                    "execute_native_candidate_block_gated_residual_mix",
                    "emit_bounded_shared_residual_candidate_output",
                ],
                "source_independent": true,
                "complete_shared_expert": true,
                "complete_hyperconnection_candidate": true,
                "complete_model": false,
                "complete_token": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "source_independent_execution": true,
            "body_mutated": false,
            "model_loaded": false,
            "native_shared_expert_gate_up_swiglu_observed": true,
            "native_shared_expert_down_projection_observed": true,
            "native_shared_expert_sigmoid_gate_observed": true,
            "native_hyperconnection_stream_injection_observed": true,
            "native_hyperconnection_low_rank_down_observed": true,
            "native_hyperconnection_low_rank_up_observed": true,
            "native_hyperconnection_block_inject_observed": true,
            "native_hyperconnection_residual_mix_observed": true,
            "native_shared_residual_composition_observed": true,
            "device_intermediate_no_host_roundtrip": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native layer-0 shared-expert-to-hyperconnection candidate graph: the measured shared-expert output is injected into stream 1 on-device, passed through source-independent Q4/G64 low-rank input mix down/up and source-independent block-inject logits, then combined by an explicit per-stream sigmoid-gated residual candidate kernel. This does not establish the source model's hc_norm or exact hyperconnection semantics, BF16 activation parity, routed-expert/MoE combine, attention/state, complete layers, complete-token runtime, Flash TPS, EBPW, or promotion.",
            "next_action": "qualify hc_norm and exact same-model hyperconnection activation semantics, then compose the candidate with routed-expert selection and the next residual boundary; keep protected complete-token Flash TPS and EBPW unmeasured until all remaining native organs are capability-qualified",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    fn run_shared_expert_composition(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;

        let gate_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_GATE_BODY_L0_R0_640.json");
        let up_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_UP_BODY_L0_R0_640.json");
        let down_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_DOWN_BODY_L0_R0_2560.json");
        let scalar_gate_receipt_path =
            repo.join("receipts/headless/FLASH_NOETIC_SHARED_EXPERT_SCALAR_GATE_BODY_L0_R0_1.json");

        let (gate, gate_receipt) = validate_matrix_body_for(
            &gate_receipt_path,
            SHARED_GATE_TENSOR_NAME,
            [SHARED_INTERMEDIATE, SHARED_HIDDEN],
        )?;
        let (up, up_receipt) = validate_matrix_body_for(
            &up_receipt_path,
            SHARED_UP_TENSOR_NAME,
            [SHARED_INTERMEDIATE, SHARED_HIDDEN],
        )?;
        let (down, down_receipt) = validate_matrix_body_for(
            &down_receipt_path,
            SHARED_DOWN_TENSOR_NAME,
            [SHARED_HIDDEN, SHARED_INTERMEDIATE],
        )?;
        let (scalar_gate, scalar_gate_receipt) = validate_matrix_body_for(
            &scalar_gate_receipt_path,
            SHARED_SCALAR_GATE_TENSOR_NAME,
            [1, SHARED_HIDDEN],
        )?;
        if gate.row_start != 0
            || gate.rows != SHARED_INTERMEDIATE
            || up.row_start != 0
            || up.rows != SHARED_INTERMEDIATE
            || down.row_start != 0
            || down.rows != SHARED_HIDDEN
            || scalar_gate.row_start != 0
            || scalar_gate.rows != 1
        {
            return Err("shared-expert bodies are not complete layer-0 row windows".into());
        }

        let input = deterministic_input(SHARED_HIDDEN);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let expected_gate_up = cpu_gate_up_swiglu(&gate, &up, &input);
        let expected_down = cpu_matvec(&down, &expected_gate_up);
        let expected_scalar_gate = cpu_matvec(&scalar_gate, &input);
        if expected_scalar_gate.len() != 1 {
            return Err("shared-expert scalar gate did not produce one logit".into());
        }
        let scalar = expected_scalar_gate[0];
        let sigmoid = 1.0f32 / (1.0f32 + (-scalar).exp());
        let expected_gated_output: Vec<f32> =
            expected_down.iter().map(|value| value * sigmoid).collect();

        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&input))?;
        let loaded = LoadedSharedExpertComposition {
            gate_codes: context.new_buffer_with_bytes_checked(&gate.codes)?,
            gate_scales: context.new_buffer_with_bytes_checked(&gate.scales)?,
            up_codes: context.new_buffer_with_bytes_checked(&up.codes)?,
            up_scales: context.new_buffer_with_bytes_checked(&up.scales)?,
            gate_up_output: context
                .new_buffer_checked(SHARED_INTERMEDIATE * std::mem::size_of::<f32>())?,
            down_codes: context.new_buffer_with_bytes_checked(&down.codes)?,
            down_scales: context.new_buffer_with_bytes_checked(&down.scales)?,
            down_output: context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?,
            scalar_gate_codes: context.new_buffer_with_bytes_checked(&scalar_gate.codes)?,
            scalar_gate_scales: context.new_buffer_with_bytes_checked(&scalar_gate.scales)?,
            scalar_gate_output: context.new_buffer_checked(std::mem::size_of::<f32>())?,
            gated_output: context.new_buffer_checked(SHARED_HIDDEN * std::mem::size_of::<f32>())?,
            gate,
            up,
            down,
            scalar_gate,
            gate_receipt,
            gate_receipt_path: gate_receipt_path.canonicalize()?,
            gate_receipt_sha256: sha256_bytes(&fs::read(&gate_receipt_path)?),
            up_receipt,
            up_receipt_path: up_receipt_path.canonicalize()?,
            up_receipt_sha256: sha256_bytes(&fs::read(&up_receipt_path)?),
            down_receipt,
            down_receipt_path: down_receipt_path.canonicalize()?,
            down_receipt_sha256: sha256_bytes(&fs::read(&down_receipt_path)?),
            scalar_gate_receipt,
            scalar_gate_receipt_path: scalar_gate_receipt_path.canonicalize()?,
            scalar_gate_receipt_sha256: sha256_bytes(&fs::read(&scalar_gate_receipt_path)?),
            expected_gate_up,
            expected_down,
            expected_scalar_gate,
            expected_gated_output,
        };

        let dispatch_graph = |item: &LoadedSharedExpertComposition| {
            let gate_up = dispatch_gate_up_swiglu(
                &context,
                &item.gate,
                &item.gate_codes,
                &item.gate_scales,
                &item.up_codes,
                &item.up_scales,
                &input_buffer,
                &item.gate_up_output,
            )?;
            let down = dispatch(
                &context,
                &item.down,
                &item.down_codes,
                &item.down_scales,
                &item.gate_up_output,
                &item.down_output,
            )?;
            let scalar_gate = dispatch(
                &context,
                &item.scalar_gate,
                &item.scalar_gate_codes,
                &item.scalar_gate_scales,
                &input_buffer,
                &item.scalar_gate_output,
            )?;
            let sigmoid_gate = dispatch_shared_expert_sigmoid_gate(
                &context,
                &item.down_output,
                &item.scalar_gate_output,
                &item.gated_output,
                SHARED_HIDDEN,
            )?;
            Ok::<
                (
                    MetalDispatchTiming,
                    MetalDispatchTiming,
                    MetalDispatchTiming,
                    MetalDispatchTiming,
                ),
                Box<dyn Error>,
            >((gate_up, down, scalar_gate, sigmoid_gate))
        };

        let mut warmup_graph_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let timings = dispatch_graph(&loaded)?;
            let total = gpu_ns(timings.0)?
                .saturating_add(gpu_ns(timings.1)?)
                .saturating_add(gpu_ns(timings.2)?)
                .saturating_add(gpu_ns(timings.3)?);
            warmup_graph_gpu_ns.push(total);
        }

        let mut gate_up_gpu_ns = Vec::with_capacity(args.reps);
        let mut down_gpu_ns = Vec::with_capacity(args.reps);
        let mut scalar_gate_gpu_ns = Vec::with_capacity(args.reps);
        let mut sigmoid_gate_gpu_ns = Vec::with_capacity(args.reps);
        let mut graph_gpu_ns = Vec::with_capacity(args.reps);
        let mut graph_host_ns = Vec::with_capacity(args.reps);
        let mut gate_up_hashes = Vec::with_capacity(args.reps);
        let mut down_hashes = Vec::with_capacity(args.reps);
        let mut scalar_gate_hashes = Vec::with_capacity(args.reps);
        let mut gated_output_hashes = Vec::with_capacity(args.reps);
        let mut last_gate_up_parity = json!({});
        let mut last_down_parity = json!({});
        let mut last_scalar_gate_parity = json!({});
        let mut last_gated_output_parity = json!({});

        for _ in 0..args.reps {
            let timings = dispatch_graph(&loaded)?;
            // All four buffers are inspected only after the graph has
            // completed. The down projection consumes the gate/up buffer and
            // the sigmoid gate consumes the down/scalar buffers on-device.
            let observed_gate_up = read_f32(&loaded.gate_up_output, SHARED_INTERMEDIATE);
            let observed_down = read_f32(&loaded.down_output, SHARED_HIDDEN);
            let observed_scalar_gate = read_f32(&loaded.scalar_gate_output, 1);
            let observed_gated_output = read_f32(&loaded.gated_output, SHARED_HIDDEN);
            if !observed_gate_up.iter().all(|value| value.is_finite())
                || !observed_down.iter().all(|value| value.is_finite())
                || !observed_scalar_gate.iter().all(|value| value.is_finite())
                || !observed_gated_output.iter().all(|value| value.is_finite())
            {
                return Err("native shared-expert composition produced a non-finite value".into());
            }
            last_gate_up_parity = output_metrics(&loaded.expected_gate_up, &observed_gate_up);
            last_down_parity = output_metrics(&loaded.expected_down, &observed_down);
            last_scalar_gate_parity =
                output_metrics(&loaded.expected_scalar_gate, &observed_scalar_gate);
            last_gated_output_parity =
                output_metrics(&loaded.expected_gated_output, &observed_gated_output);
            for (label, parity) in [
                ("gate/up", &last_gate_up_parity),
                ("down", &last_down_parity),
                ("scalar gate", &last_scalar_gate_parity),
                ("gated output", &last_gated_output_parity),
            ] {
                if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(
                        format!("native shared-expert {label} parity failed: {parity}").into(),
                    );
                }
            }
            let gate_up_gpu = gpu_ns(timings.0)?;
            let down_gpu = gpu_ns(timings.1)?;
            let scalar_gate_gpu = gpu_ns(timings.2)?;
            let sigmoid_gate_gpu = gpu_ns(timings.3)?;
            let graph_gpu = gate_up_gpu
                .saturating_add(down_gpu)
                .saturating_add(scalar_gate_gpu)
                .saturating_add(sigmoid_gate_gpu);
            let graph_host = timings
                .0
                .host_wall_us
                .saturating_add(timings.1.host_wall_us)
                .saturating_add(timings.2.host_wall_us)
                .saturating_add(timings.3.host_wall_us)
                .saturating_mul(1000);
            gate_up_gpu_ns.push(gate_up_gpu);
            down_gpu_ns.push(down_gpu);
            scalar_gate_gpu_ns.push(scalar_gate_gpu);
            sigmoid_gate_gpu_ns.push(sigmoid_gate_gpu);
            graph_gpu_ns.push(graph_gpu);
            graph_host_ns.push(graph_host);
            gate_up_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_gate_up)));
            down_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_down)));
            scalar_gate_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_scalar_gate)));
            gated_output_hashes.push(sha256_bytes(&f32_bytes_for_hash(&observed_gated_output)));
        }

        if gate_up_hashes.windows(2).any(|pair| pair[0] != pair[1])
            || down_hashes.windows(2).any(|pair| pair[0] != pair[1])
            || scalar_gate_hashes.windows(2).any(|pair| pair[0] != pair[1])
            || gated_output_hashes
                .windows(2)
                .any(|pair| pair[0] != pair[1])
        {
            return Err(
                "native shared-expert composition outputs changed across repeated executions"
                    .into(),
            );
        }

        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "layer-0 shared expert only: independent Q4/G64 gate/up -> native SwiGLU -> device-resident down projection -> independent Q4/G64 scalar gate -> native sigmoid gate",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "nodes": [
                {"id": "input", "shape": [SHARED_HIDDEN], "dtype": "F32", "device_resident": true},
                {"id": "shared_gate_proj", "shape": [SHARED_INTERMEDIATE, SHARED_HIDDEN], "representation": "independent_q4_g64"},
                {"id": "shared_up_proj", "shape": [SHARED_INTERMEDIATE, SHARED_HIDDEN], "representation": "independent_q4_g64"},
                {"id": "shared_gate_up_swiglu", "shape": [SHARED_INTERMEDIATE], "kernel": GATE_UP_KERNEL_NAME, "device_resident": true},
                {"id": "shared_down_proj", "shape": [SHARED_HIDDEN, SHARED_INTERMEDIATE], "kernel": KERNEL_NAME, "device_resident": true},
                {"id": "shared_expert_gate", "shape": [1, SHARED_HIDDEN], "representation": "independent_q4_g64"},
                {"id": "shared_sigmoid_gate", "shape": [SHARED_HIDDEN], "kernel": SHARED_SIGMOID_GATE_KERNEL_NAME, "device_resident": true},
                {"id": "shared_output", "shape": [SHARED_HIDDEN], "dtype": "F32", "device_resident": true}
            ],
            "edges": [
                ["input", "shared_gate_proj"],
                ["input", "shared_up_proj"],
                ["shared_gate_proj", "shared_gate_up_swiglu"],
                ["shared_up_proj", "shared_gate_up_swiglu"],
                ["shared_gate_up_swiglu", "shared_down_proj"],
                ["input", "shared_expert_gate"],
                ["shared_down_proj", "shared_sigmoid_gate"],
                ["shared_expert_gate", "shared_sigmoid_gate"],
                ["shared_sigmoid_gate", "shared_output"]
            ],
            "native_kernels": [GATE_UP_KERNEL_NAME, KERNEL_NAME, SHARED_SIGMOID_GATE_KERNEL_NAME],
            "native_kernel_execution_observed": true,
            "dispatches_per_graph": 4,
            "device_intermediate_no_host_roundtrip": true,
            "verification_reads_after_graph": true,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let components = json!({
            "shared_gate_proj": {
                "body_receipt": component_ref(&loaded.gate_receipt_path, &loaded.gate_receipt, Some(&loaded.gate_receipt_sha256)),
                "candidate_body": {"path": loaded.gate.path, "sha256": loaded.gate.body_sha256, "bytes": loaded.gate.code_bytes + loaded.gate.scale_bytes, "shape": [loaded.gate.rows, loaded.gate.columns], "source_independent": true, "label": "[D]"},
                "native_kernel": KERNEL_NAME,
            },
            "shared_up_proj": {
                "body_receipt": component_ref(&loaded.up_receipt_path, &loaded.up_receipt, Some(&loaded.up_receipt_sha256)),
                "candidate_body": {"path": loaded.up.path, "sha256": loaded.up.body_sha256, "bytes": loaded.up.code_bytes + loaded.up.scale_bytes, "shape": [loaded.up.rows, loaded.up.columns], "source_independent": true, "label": "[D]"},
                "native_kernel": KERNEL_NAME,
            },
            "shared_down_proj": {
                "body_receipt": component_ref(&loaded.down_receipt_path, &loaded.down_receipt, Some(&loaded.down_receipt_sha256)),
                "candidate_body": {"path": loaded.down.path, "sha256": loaded.down.body_sha256, "bytes": loaded.down.code_bytes + loaded.down.scale_bytes, "shape": [loaded.down.rows, loaded.down.columns], "source_independent": true, "label": "[D]"},
                "native_kernel": KERNEL_NAME,
            },
            "shared_expert_gate": {
                "body_receipt": component_ref(&loaded.scalar_gate_receipt_path, &loaded.scalar_gate_receipt, Some(&loaded.scalar_gate_receipt_sha256)),
                "candidate_body": {"path": loaded.scalar_gate.path, "sha256": loaded.scalar_gate.body_sha256, "bytes": loaded.scalar_gate.code_bytes + loaded.scalar_gate.scale_bytes, "shape": [loaded.scalar_gate.rows, loaded.scalar_gate.columns], "source_independent": true, "label": "[D]"},
                "native_kernel": KERNEL_NAME,
            },
        });

        Ok(json!({
            "schema": SHARED_EXPERT_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_SHARED_EXPERT_Q4_G64_GATED_SWIGLU_COMPOSITION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "layer": 0,
            "components": components,
            "component_receipt_policy": {
                "directory": repo.join("receipts/headless"),
                "gate_body": "FLASH_NOETIC_SHARED_EXPERT_GATE_BODY_L0_R0_640.json",
                "up_body": "FLASH_NOETIC_SHARED_EXPERT_UP_BODY_L0_R0_640.json",
                "down_body": "FLASH_NOETIC_SHARED_EXPERT_DOWN_BODY_L0_R0_2560.json",
                "scalar_gate_body": "FLASH_NOETIC_SHARED_EXPERT_SCALAR_GATE_BODY_L0_R0_1.json",
            },
            "execution": {
                "provider": "apple-metal",
                "operation": "full layer-0 shared_expert gate_proj + up_proj -> native gate_up_swiglu -> device-resident down_proj -> native shared_expert_gate matvec -> native sigmoid gate",
                "shared_expert_hidden": SHARED_HIDDEN,
                "shared_expert_intermediate": SHARED_INTERMEDIATE,
                "dispatches_per_graph": 4,
                "measured_graphs": args.reps,
                "total_measured_dispatches": args.reps * 4,
                "native_shared_expert_gate_up_swiglu_observed": true,
                "native_shared_expert_down_projection_observed": true,
                "native_shared_expert_scalar_gate_observed": true,
                "native_shared_expert_sigmoid_gate_observed": true,
                "complete_shared_expert_candidate_graph": true,
                "device_intermediate_no_host_roundtrip": true,
                "verification_reads_after_graph": true,
                "source_reference_used_for_execution": false,
                "body_mutated": false,
                "model_loaded": false,
                "complete_moe_combine": false,
                "complete_token_runtime": false,
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": input.len(),
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "intermediates": {
                "gate_up": {"semantic_type": "NoeticActivationBuffer", "shape": [SHARED_INTERMEDIATE], "dtype": "F32", "producer": GATE_UP_KERNEL_NAME, "consumer": KERNEL_NAME, "device_resident": true, "host_roundtrip": false},
                "down": {"semantic_type": "NoeticActivationBuffer", "shape": [SHARED_HIDDEN], "dtype": "F32", "producer": KERNEL_NAME, "consumer": SHARED_SIGMOID_GATE_KERNEL_NAME, "device_resident": true, "host_roundtrip": false},
                "scalar_gate": {"semantic_type": "NoeticGateLogit", "shape": [1], "dtype": "F32", "producer": KERNEL_NAME, "consumer": SHARED_SIGMOID_GATE_KERNEL_NAME, "device_resident": true, "host_roundtrip": false},
            },
            "parity": {
                "gate_up_swiglu": last_gate_up_parity,
                "down_projection": last_down_parity,
                "scalar_gate": last_scalar_gate_parity,
                "sigmoid_gated_output": last_gated_output_parity,
                "expected_sigmoid_gate": sigmoid,
                "candidate_space": "CPU reference uses the same persisted Q4/G64 candidate bodies; source BF16 activation parity remains a separate unqualified boundary",
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_graph_gpu_ns": warmup_graph_gpu_ns,
                "gate_up_gpu_ns": gate_up_gpu_ns,
                "gate_up_gpu_ns_median": percentile_median(&gate_up_gpu_ns),
                "down_gpu_ns": down_gpu_ns,
                "down_gpu_ns_median": percentile_median(&down_gpu_ns),
                "scalar_gate_gpu_ns": scalar_gate_gpu_ns,
                "scalar_gate_gpu_ns_median": percentile_median(&scalar_gate_gpu_ns),
                "sigmoid_gate_gpu_ns": sigmoid_gate_gpu_ns,
                "sigmoid_gate_gpu_ns_median": percentile_median(&sigmoid_gate_gpu_ns),
                "graph_gpu_ns": graph_gpu_ns,
                "graph_gpu_ns_median": percentile_median(&graph_gpu_ns),
                "graph_host_wall_ns": graph_host_ns,
                "graph_host_wall_ns_median": percentile_median(&graph_host_ns),
                "dispatches_per_graph": 4,
                "output_hashes": gated_output_hashes,
                "stage_output_hashes": {
                    "gate_up": gate_up_hashes,
                    "down": down_hashes,
                    "scalar_gate": scalar_gate_hashes,
                    "gated_output": gated_output_hashes,
                },
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for all four native shared-expert dispatches; host wall is reported separately",
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "representation": "independent_q4_g64",
                "operations": [
                    "load_source_independent_shared_expert_gate_proj_body",
                    "load_source_independent_shared_expert_up_proj_body",
                    "execute_native_q4_g64_gate_up_swiglu",
                    "retain_640_value_activation_on_device",
                    "load_source_independent_shared_expert_down_proj_body",
                    "execute_native_q4_g64_down_projection",
                    "load_source_independent_shared_expert_scalar_gate_body",
                    "execute_native_q4_g64_scalar_gate_matvec",
                    "execute_native_sigmoid_shared_expert_gate",
                    "emit_bounded_shared_expert_output",
                ],
                "source_independent": true,
                "complete_shared_expert_candidate_graph": true,
                "complete_moe_combine": false,
                "complete_model": false,
                "complete_token": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_shared_expert_gate_up_swiglu_observed": true,
            "native_shared_expert_down_projection_observed": true,
            "native_shared_expert_scalar_gate_observed": true,
            "native_shared_expert_sigmoid_gate_observed": true,
            "native_shared_expert_composition_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native layer-0 shared-expert candidate graph: full independent Q4/G64 gate_proj and up_proj feed native gate_up_swiglu, the device-resident 640-value activation feeds full independent Q4/G64 down_proj, and a full independent Q4/G64 shared_expert_gate logit is consumed by native sigmoid gating. This does not establish BF16 source-output parity, routed-expert/MoE combine, norms, hyperconnections, attention, recurrent state, MTP, lm_head, complete-model capability, complete-token runtime, Flash TPS, or EBPW.",
            "next_action": "compose this bounded shared-expert candidate with the already observed routed-expert output and residual boundary; keep protected complete-token Flash TPS and EBPW unmeasured until the remaining native organs are capability-qualified",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let destination = args.out.clone();
        let report = match if args.exact_hyperconnection_composition {
            run_exact_hyperconnection_composition(&args)
        } else if args.shared_residual_composition {
            run_shared_residual_composition(&args)
        } else if args.shared_expert_composition {
            run_shared_expert_composition(&args)
        } else if args.expert_composition {
            run_expert_composition(&args)
        } else if args.gate_up_swiglu {
            run_gate_up_swiglu(&args)
        } else {
            run(&args)
        } {
            Ok(report) => report,
            Err(error) => json!({
                "schema": if args.exact_hyperconnection_composition {
                    EXACT_HYPERCONNECTION_SCHEMA
                } else if args.shared_residual_composition {
                    SHARED_RESIDUAL_SCHEMA
                } else if args.shared_expert_composition {
                    SHARED_EXPERT_SCHEMA
                } else if args.expert_composition {
                    COMPOSITION_SCHEMA
                } else if args.gate_up_swiglu {
                    GATE_UP_SCHEMA
                } else {
                    SCHEMA
                },
                "nomenclature_version": NOMENCLATURE_VERSION,
                "status": "FAILED",
                "repo": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "error": {"type": "RoutedExpertDispatchError", "message": error.to_string()},
                "body_mutated": false,
                "model_loaded": false,
                "native_routed_body_dispatch_observed": false,
                "native_gate_up_swiglu_observed": false,
                "native_expert_gate_up_activation_observed": false,
                "native_down_projection_observed": false,
                "native_expert_composition_observed": false,
                "native_shared_expert_composition_observed": false,
                "native_shared_residual_composition_observed": false,
                "native_hyperconnection_stream_injection_observed": false,
                "native_hyperconnection_low_rank_down_observed": false,
                "native_hyperconnection_low_rank_up_observed": false,
                "native_hyperconnection_block_inject_observed": false,
                "native_hyperconnection_residual_mix_observed": false,
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "promotion_allowed": false,
            }),
        };
        write_atomic(&destination, &report)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        if report.get("status").and_then(Value::as_str) == Some("PASSED") {
            Ok(())
        } else {
            Err("Flash Noetic native routed-expert dispatch failed; see the receipt".into())
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}
