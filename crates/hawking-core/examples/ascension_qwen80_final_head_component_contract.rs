//! CPU-only, fixture-only contract for Qwen3-Coder-Next's terminal head.
//!
//! Qwen80's existing hybrid scheduler already expresses the required tail
//! sequence as final RMSNorm -> lm_head -> reserved-tail mask -> sampler. This
//! standalone target makes that seam strict and testable without opening the
//! live artifact, a Metal device, the Qwen80 watcher, or the scheduler itself.
//!
//! The fixture retains the exact source geometry (hidden=2048, direct group
//! size=128, lm-head rows=151936, tokenizer namespace=151669, reserved
//! tail=267) but carries only four tiny packed lm-head rows. It therefore
//! proves component ordering/parity/rejection behavior, never a full token,
//! a full lm-head projection, generation, HCLI, TPS, TG, or capability.
//!
//! Run:
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_final_head_component_contract -- \
//!   --out /absolute/path/QWEN80_FINAL_HEAD_COMPONENT_CONTRACT_CPU.json
//! ```

use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_final_head_component_contract_cpu.v1";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const FINAL_NORM_NAME: &str = "model.norm.weight";
const LM_HEAD_NAME: &str = "lm_head.weight";
const HIDDEN: usize = 2_048;
const LM_HEAD_VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON: f32 = 1.0e-6;
const FIXTURE_VALID_PRIMARY: u32 = 42;
const FIXTURE_VALID_LAST: u32 = (TOKENIZER_VOCAB - 1) as u32;
const FIXTURE_RESERVED_FIRST: u32 = TOKENIZER_VOCAB as u32;
const FIXTURE_RESERVED_LAST: u32 = (LM_HEAD_VOCAB - 1) as u32;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PackedTensorBinding {
    name: String,
    shape: Vec<usize>,
    group_size: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ArtifactBinding {
    model_id: String,
    source_repository: String,
    source_revision: String,
    manifest_schema: String,
    manifest_seal_sha256: String,
    final_norm: PackedTensorBinding,
    lm_head: PackedTensorBinding,
    tokenizer_vocab_size: usize,
    reserved_lm_head_tail_rows: usize,
    rms_epsilon_bits: u32,
}

#[derive(Clone, Debug)]
struct DirectPackedVector {
    /// One FP16 scale per contiguous direct-binary group.
    scales_f16_bits: Vec<u16>,
    /// Little-endian sign bits, one bit per element; one = positive.
    signs: Vec<u8>,
    elements: usize,
    group_size: usize,
}

#[derive(Clone)]
struct SourceShapedFixture {
    binding: ArtifactBinding,
    final_norm: DirectPackedVector,
    /// Only selected rows are stored. The declared binding remains the full
    /// source shape, while the fixture's partial row surface is fail-closed.
    lm_head_rows: BTreeMap<u32, DirectPackedVector>,
}

#[derive(Default)]
struct FinalHeadComponent {
    fixture: Option<SourceShapedFixture>,
    normalized: Option<Vec<f32>>,
    logits: BTreeMap<u32, f32>,
    tail_masked: bool,
}

#[derive(Serialize)]
struct FixtureExecutionReport {
    raw_fixture_argmax_before_mask: u32,
    expected_reserved_raw_argmax: u32,
    masked_fixture_argmax: u32,
    sampled_token_id: u32,
    expected_sampled_token_id: u32,
    reserved_fixture_rows_are_negative_infinity_after_mask: bool,
    final_norm_max_abs_f32_vs_fp64_reference: f64,
    final_norm_all_finite: bool,
    deterministic_repeat_matches: bool,
}

#[derive(Serialize)]
struct RejectionReport {
    malformed_final_norm_shape_rejected: bool,
    malformed_lm_head_shape_rejected: bool,
    wrong_group_size_rejected: bool,
    bad_reserved_tail_count_rejected: bool,
    bad_mask_cutoff_rejected: bool,
    sample_before_tail_mask_rejected: bool,
    fixture_row_outside_declared_domain_rejected: bool,
}

#[derive(Serialize)]
struct IntegrationContract {
    existing_scheduler_terminal_order: Vec<&'static str>,
    artifact_binding_requirements: Vec<&'static str>,
    device_component_requirements: Vec<&'static str>,
    promotion_boundary: Vec<&'static str>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    fixture_only: bool,
    live_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    artifact_binding_contract: ArtifactBinding,
    fixture_rows: Vec<u32>,
    fixture_execution: FixtureExecutionReport,
    rejection_tests: RejectionReport,
    existing_hybrid_scheduler_integration_contract: IntegrationContract,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl ArtifactBinding {
    fn validate(&self) -> Result<(), String> {
        if self.model_id != MODEL_ID
            || self.source_repository != SOURCE_REPOSITORY
            || self.manifest_schema != MANIFEST_SCHEMA
        {
            return Err(
                "binding does not identify the exact Qwen3-Coder-Next artifact family".into(),
            );
        }
        if !is_lower_sha256(&self.manifest_seal_sha256) || !is_git_revision(&self.source_revision) {
            return Err("binding has malformed manifest seal or source revision".into());
        }
        if self.final_norm.name != FINAL_NORM_NAME
            || self.final_norm.shape != [HIDDEN]
            || self.final_norm.group_size != GROUP_SIZE
        {
            return Err(
                "final RMSNorm binding drifted from exact Qwen80 direct-packed geometry".into(),
            );
        }
        if self.lm_head.name != LM_HEAD_NAME
            || self.lm_head.shape != [LM_HEAD_VOCAB, HIDDEN]
            || self.lm_head.group_size != GROUP_SIZE
        {
            return Err("lm_head binding drifted from exact Qwen80 direct-packed geometry".into());
        }
        if self.tokenizer_vocab_size != TOKENIZER_VOCAB
            || self.reserved_lm_head_tail_rows != RESERVED_TAIL_ROWS
            || self.tokenizer_vocab_size + self.reserved_lm_head_tail_rows != LM_HEAD_VOCAB
        {
            return Err("tokenizer namespace and reserved lm_head tail do not exactly partition Qwen80 vocab".into());
        }
        if self.rms_epsilon_bits != RMS_EPSILON.to_bits() {
            return Err("RMS epsilon drifted from source Qwen80 1e-6".into());
        }
        Ok(())
    }
}

impl DirectPackedVector {
    fn validate(&self, expected_elements: usize) -> Result<(), String> {
        if self.elements != expected_elements || self.group_size != GROUP_SIZE {
            return Err(
                "direct-packed fixture vector has unexpected element/group geometry".into(),
            );
        }
        let groups = expected_elements.div_ceil(GROUP_SIZE);
        if self.scales_f16_bits.len() != groups || self.signs.len() != groups * (GROUP_SIZE / 8) {
            return Err("direct-packed fixture vector has invalid scale/sign byte geometry".into());
        }
        if self
            .scales_f16_bits
            .iter()
            .any(|bits| !f16::from_bits(*bits).is_finite())
        {
            return Err("direct-packed fixture has a non-finite FP16 scale".into());
        }
        Ok(())
    }

    fn value(&self, index: usize) -> f32 {
        debug_assert!(index < self.elements);
        let group = index / self.group_size;
        let within_group = index % self.group_size;
        let scale = f16::from_bits(self.scales_f16_bits[group]).to_f32();
        let byte = self.signs[group * (self.group_size / 8) + within_group / 8];
        if ((byte >> (within_group % 8)) & 1) != 0 {
            scale
        } else {
            -scale
        }
    }

    fn dot_f32(&self, input: &[f32]) -> Result<f32, String> {
        self.validate(input.len())?;
        let mut sum = 0.0f32;
        for (index, &value) in input.iter().enumerate() {
            sum += self.value(index) * value;
        }
        if !sum.is_finite() {
            return Err("direct-packed fixture dot product is non-finite".into());
        }
        Ok(sum)
    }
}

impl SourceShapedFixture {
    fn validate(&self) -> Result<(), String> {
        self.binding.validate()?;
        self.final_norm.validate(HIDDEN)?;
        if self.lm_head_rows.len() != 4 {
            return Err(
                "source-shaped fixture must carry exactly four declared lm_head rows".into(),
            );
        }
        for (&row, vector) in &self.lm_head_rows {
            if row as usize >= LM_HEAD_VOCAB {
                return Err("fixture lm_head row is outside declared full source vocab".into());
            }
            vector.validate(HIDDEN)?;
        }
        for required in [
            FIXTURE_VALID_PRIMARY,
            FIXTURE_VALID_LAST,
            FIXTURE_RESERVED_FIRST,
            FIXTURE_RESERVED_LAST,
        ] {
            if !self.lm_head_rows.contains_key(&required) {
                return Err(
                    "source-shaped fixture is missing a required valid/reserved boundary row"
                        .into(),
                );
            }
        }
        Ok(())
    }
}

fn packed_all_positive(scale: f32) -> DirectPackedVector {
    let scales = vec![f16::from_f32(scale).to_bits(); HIDDEN / GROUP_SIZE];
    DirectPackedVector {
        scales_f16_bits: scales,
        signs: vec![0xff; HIDDEN / 8],
        elements: HIDDEN,
        group_size: GROUP_SIZE,
    }
}

fn packed_alternating(scale: f32) -> DirectPackedVector {
    let scales = vec![f16::from_f32(scale).to_bits(); HIDDEN / GROUP_SIZE];
    DirectPackedVector {
        scales_f16_bits: scales,
        signs: (0..HIDDEN / 8)
            .map(|byte| {
                if byte % 2 == 0 {
                    0b0101_0101
                } else {
                    0b1010_1010
                }
            })
            .collect(),
        elements: HIDDEN,
        group_size: GROUP_SIZE,
    }
}

fn fixture_binding() -> ArtifactBinding {
    ArtifactBinding {
        model_id: MODEL_ID.to_owned(),
        source_repository: SOURCE_REPOSITORY.to_owned(),
        // Fixture-only identity: syntactically a pinned git revision but never
        // presented as a live source or artifact admission.
        source_revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        manifest_schema: MANIFEST_SCHEMA.to_owned(),
        manifest_seal_sha256: "a3f1c5d7e9b24680a3f1c5d7e9b24680a3f1c5d7e9b24680a3f1c5d7e9b24680"
            .to_owned(),
        final_norm: PackedTensorBinding {
            name: FINAL_NORM_NAME.to_owned(),
            shape: vec![HIDDEN],
            group_size: GROUP_SIZE,
        },
        lm_head: PackedTensorBinding {
            name: LM_HEAD_NAME.to_owned(),
            shape: vec![LM_HEAD_VOCAB, HIDDEN],
            group_size: GROUP_SIZE,
        },
        tokenizer_vocab_size: TOKENIZER_VOCAB,
        reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
        rms_epsilon_bits: RMS_EPSILON.to_bits(),
    }
}

fn source_shaped_fixture() -> SourceShapedFixture {
    let mut rows = BTreeMap::new();
    // The primary valid row wins after the source-reserved tail is masked.
    rows.insert(FIXTURE_VALID_PRIMARY, packed_all_positive(0.5));
    rows.insert(FIXTURE_VALID_LAST, packed_alternating(0.45));
    // These deliberately win before masking so the test proves that tail
    // masking is a mandatory correctness boundary, not a cosmetic filter.
    rows.insert(FIXTURE_RESERVED_FIRST, packed_all_positive(1.0));
    rows.insert(FIXTURE_RESERVED_LAST, packed_all_positive(0.9));
    SourceShapedFixture {
        binding: fixture_binding(),
        final_norm: packed_all_positive(1.0),
        lm_head_rows: rows,
    }
}

fn fixture_hidden() -> Vec<f32> {
    (0..HIDDEN)
        .map(|index| 0.25 + ((index * 17 % 29) as f32) / 29.0)
        .collect()
}

fn final_rms_norm_f32(
    binding: &ArtifactBinding,
    norm: &DirectPackedVector,
    hidden: &[f32],
) -> Result<Vec<f32>, String> {
    binding.validate()?;
    norm.validate(HIDDEN)?;
    if hidden.len() != HIDDEN || hidden.iter().any(|value| !value.is_finite()) {
        return Err(
            "final RMSNorm hidden state has invalid Qwen80 shape or non-finite values".into(),
        );
    }
    let mut sum_sq = 0.0f32;
    for &value in hidden {
        sum_sq += value * value;
    }
    let inv_rms = 1.0 / (sum_sq / HIDDEN as f32 + f32::from_bits(binding.rms_epsilon_bits)).sqrt();
    let output = hidden
        .iter()
        .enumerate()
        .map(|(index, &value)| value * inv_rms * norm.value(index))
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("final RMSNorm produced a non-finite result".into());
    }
    Ok(output)
}

fn final_rms_norm_f64_reference(
    binding: &ArtifactBinding,
    norm: &DirectPackedVector,
    hidden: &[f32],
) -> Result<Vec<f64>, String> {
    binding.validate()?;
    norm.validate(HIDDEN)?;
    if hidden.len() != HIDDEN || hidden.iter().any(|value| !value.is_finite()) {
        return Err("reference final RMSNorm hidden state is invalid".into());
    }
    let sum_sq = hidden
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let inv_rms =
        1.0 / (sum_sq / HIDDEN as f64 + f64::from(f32::from_bits(binding.rms_epsilon_bits))).sqrt();
    Ok(hidden
        .iter()
        .enumerate()
        .map(|(index, &value)| f64::from(value) * inv_rms * f64::from(norm.value(index)))
        .collect())
}

impl FinalHeadComponent {
    fn from_fixture(fixture: SourceShapedFixture) -> Result<Self, String> {
        fixture.validate()?;
        Ok(Self {
            fixture: Some(fixture),
            normalized: None,
            logits: BTreeMap::new(),
            tail_masked: false,
        })
    }

    fn fixture(&self) -> Result<&SourceShapedFixture, String> {
        self.fixture
            .as_ref()
            .ok_or_else(|| "final-head component has no artifact-bound fixture".into())
    }

    fn final_rms_norm(&mut self, hidden: &[f32]) -> Result<(), String> {
        if self.normalized.is_some() || !self.logits.is_empty() || self.tail_masked {
            return Err(
                "final RMSNorm may begin only one fresh terminal-head component sequence".into(),
            );
        }
        let fixture = self.fixture()?;
        self.normalized = Some(final_rms_norm_f32(
            &fixture.binding,
            &fixture.final_norm,
            hidden,
        )?);
        Ok(())
    }

    fn direct_packed_lm_head(&mut self) -> Result<(), String> {
        if self.tail_masked || !self.logits.is_empty() {
            return Err(
                "lm_head cannot run after logits exist or after reserved-tail masking".into(),
            );
        }
        let normalized = self
            .normalized
            .as_ref()
            .ok_or("lm_head requires completed final RMSNorm")?;
        let fixture = self.fixture()?;
        fixture.binding.validate()?;
        let mut logits = BTreeMap::new();
        for (&row, packed_row) in &fixture.lm_head_rows {
            if row as usize >= fixture.binding.lm_head.shape[0] {
                return Err("fixture row exceeded artifact-bound lm_head domain".into());
            }
            logits.insert(row, packed_row.dot_f32(normalized)?);
        }
        self.logits = logits;
        Ok(())
    }

    fn raw_fixture_argmax(&self) -> Result<u32, String> {
        if self.logits.is_empty() {
            return Err("raw fixture argmax requires direct-packed lm_head logits".into());
        }
        self.logits
            .iter()
            .filter(|(_, logit)| logit.is_finite())
            .max_by(|(left_id, left), (right_id, right)| {
                left.total_cmp(right).then_with(|| right_id.cmp(left_id))
            })
            .map(|(&id, _)| id)
            .ok_or_else(|| "fixture lm_head has no finite logit".into())
    }

    fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<(), String> {
        let fixture = self.fixture()?;
        if self.normalized.is_none() || self.logits.is_empty() || self.tail_masked {
            return Err(
                "reserved-tail mask requires exactly one completed lm_head projection".into(),
            );
        }
        if first_reserved_id as usize != fixture.binding.tokenizer_vocab_size {
            return Err(
                "reserved-tail mask cutoff differs from artifact-bound tokenizer vocabulary".into(),
            );
        }
        for (&id, logit) in &mut self.logits {
            if id >= first_reserved_id {
                *logit = f32::NEG_INFINITY;
            }
        }
        self.tail_masked = true;
        Ok(())
    }

    fn greedy_sample_token(&self) -> Result<u32, String> {
        let fixture = self.fixture()?;
        if !self.tail_masked {
            return Err(
                "sampler must refuse logits before source-reserved lm_head tail masking".into(),
            );
        }
        let token = self
            .logits
            .iter()
            .filter(|(&id, logit)| {
                (id as usize) < fixture.binding.tokenizer_vocab_size && logit.is_finite()
            })
            .max_by(|(left_id, left), (right_id, right)| {
                left.total_cmp(right).then_with(|| right_id.cmp(left_id))
            })
            .map(|(&id, _)| id)
            .ok_or_else(|| {
                "sampler has no finite tokenizer-addressable fixture logit".to_owned()
            })?;
        if token as usize >= fixture.binding.tokenizer_vocab_size {
            return Err("sampler would emit a source-reserved lm_head tail id".into());
        }
        Ok(token)
    }

    fn all_fixture_tail_logits_masked(&self) -> bool {
        self.fixture()
            .ok()
            .map(|fixture| {
                self.logits.iter().all(|(&id, &logit)| {
                    (id as usize) < fixture.binding.tokenizer_vocab_size
                        || logit == f32::NEG_INFINITY
                })
            })
            .unwrap_or(false)
    }
}

fn execute_fixture() -> Result<FixtureExecutionReport, String> {
    let fixture = source_shaped_fixture();
    let hidden = fixture_hidden();
    let normalized = final_rms_norm_f32(&fixture.binding, &fixture.final_norm, &hidden)?;
    let reference = final_rms_norm_f64_reference(&fixture.binding, &fixture.final_norm, &hidden)?;
    let max_abs = normalized
        .iter()
        .zip(&reference)
        .map(|(&left, &right)| (f64::from(left) - right).abs())
        .fold(0.0, f64::max);
    let mut component = FinalHeadComponent::from_fixture(fixture)?;
    component.final_rms_norm(&hidden)?;
    component.direct_packed_lm_head()?;
    let raw_fixture_argmax_before_mask = component.raw_fixture_argmax()?;
    component.mask_reserved_lm_head_tail(TOKENIZER_VOCAB as u32)?;
    let masked_fixture_argmax = component.raw_fixture_argmax()?;
    let sampled_token_id = component.greedy_sample_token()?;
    let mut repeat = FinalHeadComponent::from_fixture(source_shaped_fixture())?;
    repeat.final_rms_norm(&hidden)?;
    repeat.direct_packed_lm_head()?;
    repeat.mask_reserved_lm_head_tail(TOKENIZER_VOCAB as u32)?;
    let deterministic_repeat_matches = sampled_token_id == repeat.greedy_sample_token()?;
    Ok(FixtureExecutionReport {
        raw_fixture_argmax_before_mask,
        expected_reserved_raw_argmax: FIXTURE_RESERVED_FIRST,
        masked_fixture_argmax,
        sampled_token_id,
        expected_sampled_token_id: FIXTURE_VALID_PRIMARY,
        reserved_fixture_rows_are_negative_infinity_after_mask: component
            .all_fixture_tail_logits_masked(),
        final_norm_max_abs_f32_vs_fp64_reference: max_abs,
        final_norm_all_finite: normalized.iter().all(|value| value.is_finite()),
        deterministic_repeat_matches,
    })
}

fn rejection_report() -> RejectionReport {
    let mut malformed_norm = fixture_binding();
    malformed_norm.final_norm.shape = vec![HIDDEN + 1];
    let malformed_final_norm_shape_rejected = malformed_norm.validate().is_err();

    let mut malformed_head = fixture_binding();
    malformed_head.lm_head.shape = vec![LM_HEAD_VOCAB - 1, HIDDEN];
    let malformed_lm_head_shape_rejected = malformed_head.validate().is_err();

    let mut wrong_group = fixture_binding();
    wrong_group.lm_head.group_size = 64;
    let wrong_group_size_rejected = wrong_group.validate().is_err();

    let mut bad_tail = fixture_binding();
    bad_tail.reserved_lm_head_tail_rows = RESERVED_TAIL_ROWS - 1;
    let bad_reserved_tail_count_rejected = bad_tail.validate().is_err();

    let hidden = fixture_hidden();
    let mut bad_mask_component = FinalHeadComponent::from_fixture(source_shaped_fixture()).unwrap();
    bad_mask_component.final_rms_norm(&hidden).unwrap();
    bad_mask_component.direct_packed_lm_head().unwrap();
    let bad_mask_cutoff_rejected = bad_mask_component
        .mask_reserved_lm_head_tail((TOKENIZER_VOCAB - 1) as u32)
        .is_err();

    let mut early_sample = FinalHeadComponent::from_fixture(source_shaped_fixture()).unwrap();
    early_sample.final_rms_norm(&hidden).unwrap();
    early_sample.direct_packed_lm_head().unwrap();
    let sample_before_tail_mask_rejected = early_sample.greedy_sample_token().is_err();

    let mut out_of_domain_fixture = source_shaped_fixture();
    out_of_domain_fixture
        .lm_head_rows
        .insert(LM_HEAD_VOCAB as u32, packed_all_positive(1.0));
    let fixture_row_outside_declared_domain_rejected = out_of_domain_fixture.validate().is_err();

    RejectionReport {
        malformed_final_norm_shape_rejected,
        malformed_lm_head_shape_rejected,
        wrong_group_size_rejected,
        bad_reserved_tail_count_rejected,
        bad_mask_cutoff_rejected,
        sample_before_tail_mask_rejected,
        fixture_row_outside_declared_domain_rejected,
    }
}

fn integration_contract() -> IntegrationContract {
    IntegrationContract {
        existing_scheduler_terminal_order: vec![
            "After all 48 hybrid layers and both state domains have completed: final_rms_norm(model.norm.weight).",
            "Then direct-packed lm_head(lm_head.weight) over all 151936 rows, with no BF16/MPS shadow path.",
            "Then mask_reserved_lm_head_tail(first_reserved_id=151669) on the complete logit domain.",
            "Only then sample_token(tokenizer_vocab_size=151669), validate sampled_id < 151669, and feed that ID to the next native token step.",
        ],
        artifact_binding_requirements: vec![
            "The executor must reject unless its final norm and lm-head tensor bindings exactly match the admitted hybrid plan's names, shapes, group size, manifest seal, source revision, epsilon, tokenizer namespace, and 267-row tail.",
            "A later integration must retain the direct-packed payload authority. This fixture contract cannot authorize decoded BF16, CPU shadow, or MPS production weights.",
            "The full lm-head result must cover every declared row; four fixture rows are intentionally insufficient for a model token.",
        ],
        device_component_requirements: vec![
            "GPU parity must compare final RMSNorm and selected direct-packed lm-head rows against a source-bound CPU reference under an exclusive Qwen80 lease.",
            "The device implementation must prove that every logits index 151669..151935 is masked before sampling, including when an unmasked reserved row would otherwise win argmax.",
            "The device sampler must use deterministic documented tie-breaking and reject non-finite/no-valid-token states.",
        ],
        promotion_boundary: vec![
            "This fixture validates only a terminal component contract. It is not a complete Qwen80 layer/token, generation, HCLI, BASE_TRUE_TPS, TG, capability, Agent OS, or tournament receipt.",
            "Integration requires the existing hybrid scheduler's all-layer state/routing work plus fresh component, full-token, multi-token, HCLI, capability, and clean-performance evidence.",
        ],
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_report_atomic(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn parse_out() -> Result<PathBuf, Box<dyn Error>> {
    let mut values = env::args().skip(1);
    let flag = values
        .next()
        .ok_or("usage: ascension_qwen80_final_head_component_contract --out ABSOLUTE_PATH")?;
    if flag != "--out" {
        return Err("expected --out ABSOLUTE_PATH".into());
    }
    let out = PathBuf::from(values.next().ok_or("missing absolute path after --out")?);
    if values.next().is_some() || !out.is_absolute() {
        return Err("--out must be the sole option and must be absolute".into());
    }
    Ok(out)
}

fn run(out: PathBuf) -> Result<(), Box<dyn Error>> {
    let fixture = source_shaped_fixture();
    fixture.validate()?;
    let fixture_execution = execute_fixture()?;
    let rejection_tests = rejection_report();
    let all_rejections_pass = [
        rejection_tests.malformed_final_norm_shape_rejected,
        rejection_tests.malformed_lm_head_shape_rejected,
        rejection_tests.wrong_group_size_rejected,
        rejection_tests.bad_reserved_tail_count_rejected,
        rejection_tests.bad_mask_cutoff_rejected,
        rejection_tests.sample_before_tail_mask_rejected,
        rejection_tests.fixture_row_outside_declared_domain_rejected,
    ]
    .into_iter()
    .all(|passed| passed);
    if fixture_execution.raw_fixture_argmax_before_mask != FIXTURE_RESERVED_FIRST
        || fixture_execution.masked_fixture_argmax != FIXTURE_VALID_PRIMARY
        || fixture_execution.sampled_token_id != FIXTURE_VALID_PRIMARY
        || !fixture_execution.reserved_fixture_rows_are_negative_infinity_after_mask
        || !fixture_execution.final_norm_all_finite
        || !fixture_execution.deterministic_repeat_matches
        || fixture_execution.final_norm_max_abs_f32_vs_fp64_reference > 3.0e-6
        || !all_rejections_pass
    {
        return Err(
            "fixture final-head contract did not satisfy its deterministic acceptance conditions"
                .into(),
        );
    }
    let mut report = Report {
        schema: SCHEMA,
        status: "EARNED_SOURCE_SHAPED_FINAL_HEAD_COMPONENT_CONTRACT_NOT_LIVE_ARTIFACT_OR_TOKEN",
        fixture_only: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        artifact_binding_contract: fixture.binding,
        fixture_rows: vec![
            FIXTURE_VALID_PRIMARY,
            FIXTURE_VALID_LAST,
            FIXTURE_RESERVED_FIRST,
            FIXTURE_RESERVED_LAST,
        ],
        fixture_execution,
        rejection_tests,
        existing_hybrid_scheduler_integration_contract: integration_contract(),
        claim_boundary: vec![
            "The binding uses a synthetic fixture seal/revision and selected synthetic packed rows; it does not open, hash, admit, or scan the live Qwen80 artifact.",
            "No Metal context/device/command buffer was created and no Qwen80 runtime, watcher, gatekeeper, registry, or tournament file was changed.",
            "The result proves only CPU fixture behavior for final RMSNorm, selected lm-head rows, reserved-tail masking, and greedy sampling order.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = sha256_hex(&serde_json::to_vec(&report)?);
    write_report_atomic(&out, &report)
}

fn main() {
    match parse_out().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_final_head_component_contract: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_shaped_fixture_has_exact_qwen80_terminal_geometry() {
        let fixture = source_shaped_fixture();
        fixture.validate().unwrap();
        assert_eq!(fixture.binding.final_norm.shape, [HIDDEN]);
        assert_eq!(fixture.binding.lm_head.shape, [LM_HEAD_VOCAB, HIDDEN]);
        assert_eq!(fixture.binding.tokenizer_vocab_size, 151_669);
        assert_eq!(fixture.binding.reserved_lm_head_tail_rows, 267);
    }

    #[test]
    fn final_head_masks_reserved_tail_before_sampling() {
        let report = execute_fixture().unwrap();
        assert_eq!(
            report.raw_fixture_argmax_before_mask,
            FIXTURE_RESERVED_FIRST
        );
        assert_eq!(report.masked_fixture_argmax, FIXTURE_VALID_PRIMARY);
        assert_eq!(report.sampled_token_id, FIXTURE_VALID_PRIMARY);
        assert!(report.reserved_fixture_rows_are_negative_infinity_after_mask);
        assert!(report.final_norm_all_finite);
        assert!(report.deterministic_repeat_matches);
        assert!(report.final_norm_max_abs_f32_vs_fp64_reference <= 3.0e-6);
    }

    #[test]
    fn sampler_refuses_pre_mask_logits_and_wrong_cutoff() {
        let hidden = fixture_hidden();
        let mut component = FinalHeadComponent::from_fixture(source_shaped_fixture()).unwrap();
        component.final_rms_norm(&hidden).unwrap();
        component.direct_packed_lm_head().unwrap();
        assert!(component.greedy_sample_token().is_err());
        assert!(component
            .mask_reserved_lm_head_tail((TOKENIZER_VOCAB - 1) as u32)
            .is_err());
        component
            .mask_reserved_lm_head_tail(TOKENIZER_VOCAB as u32)
            .unwrap();
        assert_eq!(
            component.greedy_sample_token().unwrap(),
            FIXTURE_VALID_PRIMARY
        );
    }

    #[test]
    fn malformed_artifact_bound_geometry_is_rejected() {
        let report = rejection_report();
        assert!(report.malformed_final_norm_shape_rejected);
        assert!(report.malformed_lm_head_shape_rejected);
        assert!(report.wrong_group_size_rejected);
        assert!(report.bad_reserved_tail_count_rejected);
        assert!(report.fixture_row_outside_declared_domain_rejected);
    }

    #[test]
    fn direct_packed_vector_rejects_wrong_group_shape() {
        let mut vector = packed_all_positive(1.0);
        vector.scales_f16_bits.pop();
        assert!(vector.validate(HIDDEN).is_err());
    }
}
