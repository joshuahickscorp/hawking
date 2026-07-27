//! One trait/ABI for architecture-family adapters.
//!
//! Every registered family MUST declare every ABI field. Unknown or
//! inapplicable fields use [`AbiField::null`] / [`AbiListField::null`] with a
//! reason — silent omission is a registry error.

use crate::support_level::SupportLevel;

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

/// Kind of evidence required by a support grade. The grade names the evidence;
/// promotion without the matching kind fails the honesty test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceKind {
    /// Narrative / archaeology / module existence (DECLARED only needs this).
    Description,
    /// Official config.json / tokenizer / safetensors header parsed and mapped.
    SourceHeader,
    /// Deterministic reference match on a synthetic twin.
    SyntheticParity,
    /// At least one real tensor decoded from a real checkpoint.
    RealTensorDecode,
    /// Real small checkpoint of the family runs end to end.
    SmallCheckpointRun,
    /// Real full-size parent validated.
    FullParentValidation,
    /// Served under test with a standing production parity receipt.
    ProductionReceipt,
}

impl EvidenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EvidenceKind::Description => "description",
            EvidenceKind::SourceHeader => "source_header",
            EvidenceKind::SyntheticParity => "synthetic_parity",
            EvidenceKind::RealTensorDecode => "real_tensor_decode",
            EvidenceKind::SmallCheckpointRun => "small_checkpoint_run",
            EvidenceKind::FullParentValidation => "full_parent_validation",
            EvidenceKind::ProductionReceipt => "production_receipt",
        }
    }
}

/// A named piece of evidence that backs a support level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evidence {
    /// Repo-relative path to a test, receipt JSON, or module.
    pub path: &'static str,
    /// One-line claim this evidence supports.
    pub claim: &'static str,
    /// What kind of proof this entry is.
    pub kind: EvidenceKind,
}

// ---------------------------------------------------------------------------
// Nullable ABI fields
// ---------------------------------------------------------------------------

/// A single string ABI field: present value, or explicit null + reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiField {
    pub value: Option<&'static str>,
    pub null_reason: Option<&'static str>,
}

impl AbiField {
    pub const fn some(value: &'static str) -> Self {
        Self {
            value: Some(value),
            null_reason: None,
        }
    }

    pub const fn null(reason: &'static str) -> Self {
        Self {
            value: None,
            null_reason: Some(reason),
        }
    }

    /// Field is well-formed: either value XOR (for null) a reason.
    pub fn is_complete(&self) -> bool {
        match (self.value, self.null_reason) {
            (Some(_), None) => true,
            (None, Some(_)) => true,
            _ => false,
        }
    }
}

/// A list ABI field: present values, or explicit null + reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiListField {
    pub values: Option<&'static [&'static str]>,
    pub null_reason: Option<&'static str>,
}

impl AbiListField {
    pub const fn some(values: &'static [&'static str]) -> Self {
        Self {
            values: Some(values),
            null_reason: None,
        }
    }

    pub const fn null(reason: &'static str) -> Self {
        Self {
            values: None,
            null_reason: Some(reason),
        }
    }

    pub fn is_complete(&self) -> bool {
        match (self.values, self.null_reason) {
            (Some(_), None) => true,
            (None, Some(_)) => true,
            _ => false,
        }
    }
}

/// Context window claim + whether that claim has been validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLimits {
    /// Declared max context tokens, if known.
    pub max_context_tokens: Option<u64>,
    /// Validation status of the limit (not of the family overall).
    pub validation_status: &'static str,
    /// When max_context_tokens is None, why.
    pub null_reason: Option<&'static str>,
}

impl ContextLimits {
    pub const fn known(max: u64, validation_status: &'static str) -> Self {
        Self {
            max_context_tokens: Some(max),
            validation_status,
            null_reason: None,
        }
    }

    pub const fn unknown(reason: &'static str) -> Self {
        Self {
            max_context_tokens: None,
            validation_status: "unvalidated",
            null_reason: Some(reason),
        }
    }

    pub fn is_complete(&self) -> bool {
        match (self.max_context_tokens, self.null_reason) {
            (Some(_), _) => !self.validation_status.is_empty(),
            (None, Some(_)) => true,
            (None, None) => false,
        }
    }
}

/// CPU / Metal / CUDA provider availability for a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderAvailability {
    pub cpu: AbiField,
    pub metal: AbiField,
    pub cuda: AbiField,
}

impl ProviderAvailability {
    pub fn is_complete(&self) -> bool {
        self.cpu.is_complete() && self.metal.is_complete() && self.cuda.is_complete()
    }
}

// ---------------------------------------------------------------------------
// Full family ABI surface
// ---------------------------------------------------------------------------

/// Every field the adapter ABI requires a family to declare.
///
/// Fields are either a concrete value or explicitly `null` with a reason.
/// See `HAWKING_ADAPTER_ABI.json` (generated) for the schema documentation.
#[derive(Debug, Clone, Copy)]
pub struct FamilyAbi {
    /// Source / config class identifiers (e.g. `gguf.llama`, `hf.config.json`).
    pub source_config_classes: AbiListField,
    /// Tensor namespace rules (naming templates, prefixes, packing).
    pub tensor_namespace_rules: AbiField,
    /// Tokenizer identity / protocol.
    pub tokenizer: AbiField,
    /// Chat template identity (jinja / protocol name).
    pub chat_template: AbiField,
    /// Attention or state mechanism (GQA, MLA, SSM, …).
    pub attention_or_state: AbiField,
    /// Dense / MoE topology description.
    pub topology: AbiField,
    /// Normalization (RMSNorm, LayerNorm, …).
    pub normalization: AbiField,
    /// Positional encoding (RoPE, ALiBi, none, …).
    pub positional_encoding: AbiField,
    /// KV cache or recurrent state format.
    pub kv_or_state_format: AbiField,
    /// Context limits + validation status of those limits.
    pub context_limits: ContextLimits,
    /// Artifact codecs the family can ingest (gguf, gravity, safetensors, …).
    pub artifact_codecs: AbiListField,
    /// CPU / Metal / CUDA provider availability.
    pub providers: ProviderAvailability,
    /// Fabric partition boundaries (what may be sharded / placed).
    pub fabric_partition_boundaries: AbiField,
    /// Known capability limitations (always a list; empty is fine).
    pub capability_limitations: &'static [&'static str],
    /// Source precision classes (BF16, Q4_K, F8_E4M3, …).
    pub source_precision_classes: AbiListField,
}

impl FamilyAbi {
    /// All ABI fields complete (value or null+reason).
    pub fn validate_complete(&self, family_id: &str) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        let mut check = |name: &str, ok: bool| {
            if !ok {
                errs.push(format!(
                    "family {family_id}: ABI field `{name}` must be a value or null with a reason"
                ));
            }
        };
        check(
            "source_config_classes",
            self.source_config_classes.is_complete(),
        );
        check(
            "tensor_namespace_rules",
            self.tensor_namespace_rules.is_complete(),
        );
        check("tokenizer", self.tokenizer.is_complete());
        check("chat_template", self.chat_template.is_complete());
        check("attention_or_state", self.attention_or_state.is_complete());
        check("topology", self.topology.is_complete());
        check("normalization", self.normalization.is_complete());
        check(
            "positional_encoding",
            self.positional_encoding.is_complete(),
        );
        check("kv_or_state_format", self.kv_or_state_format.is_complete());
        check("context_limits", self.context_limits.is_complete());
        check("artifact_codecs", self.artifact_codecs.is_complete());
        check("providers", self.providers.is_complete());
        check(
            "fabric_partition_boundaries",
            self.fabric_partition_boundaries.is_complete(),
        );
        // capability_limitations is always present (possibly empty)
        check(
            "source_precision_classes",
            self.source_precision_classes.is_complete(),
        );
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

/// Ordered list of ABI field names (canonical schema keys).
pub const ABI_FIELD_NAMES: &[&str] = &[
    "family_id",
    "aliases",
    "source_config_classes",
    "tensor_namespace_rules",
    "tokenizer",
    "chat_template",
    "attention_or_state",
    "topology",
    "normalization",
    "positional_encoding",
    "kv_or_state_format",
    "context_limits",
    "artifact_codecs",
    "providers",
    "fabric_partition_boundaries",
    "capability_limitations",
    "source_precision_classes",
    "parity_evidence",
    "support_level",
];

// ---------------------------------------------------------------------------
// Family descriptor + trait
// ---------------------------------------------------------------------------

/// Static descriptor exported by each family module.
#[derive(Debug, Clone, Copy)]
pub struct FamilyDescriptor {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub display_name: &'static str,
    pub level: SupportLevel,
    /// Parity / grade evidence (paths must exist for non-DECLARED levels).
    pub evidence: &'static [Evidence],
    /// Rust module path (or pack id) that implements / declares the family.
    pub module: &'static str,
    /// True when *some* path in-tree can execute a forward/generate for this family.
    pub executes: bool,
    /// True when `load_engine` (or gravity registry) dispatches this family today.
    pub serve_registered: bool,
    pub gaps: &'static [&'static str],
    /// Full ABI surface.
    pub abi: FamilyAbi,
}

/// The one family-adapter ABI. Implementations are thin static descriptors;
/// this trait exists so the registry can be extended without a giant match.
pub trait FamilyAdapter: Send + Sync {
    fn descriptor(&self) -> FamilyDescriptor;

    fn family_id(&self) -> &'static str {
        self.descriptor().id
    }

    fn support_level(&self) -> SupportLevel {
        self.descriptor().level
    }
}

/// Evidence kind required to claim a given support grade.
pub fn required_evidence_kind(level: SupportLevel) -> Option<EvidenceKind> {
    match level {
        SupportLevel::Declared => None,
        SupportLevel::SourceHeaderValidated => Some(EvidenceKind::SourceHeader),
        SupportLevel::SyntheticParity => Some(EvidenceKind::SyntheticParity),
        SupportLevel::RealTensorDecode => Some(EvidenceKind::RealTensorDecode),
        SupportLevel::SmallRealCheckpoint => Some(EvidenceKind::SmallCheckpointRun),
        SupportLevel::FullParentValidated => Some(EvidenceKind::FullParentValidation),
        SupportLevel::Production => Some(EvidenceKind::ProductionReceipt),
    }
}
