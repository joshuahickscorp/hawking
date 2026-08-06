//! Transport-neutral JSONL machine-control contract for HCLI.
//!
//! This module deliberately does **not** execute a model, start an agent, or
//! call the existing turn path.  It is the small, stable seam a future `hcli`
//! binary (or an external Codex/Claude/script driver) can use to ask the host
//! for work over stdin/stdout without inheriting a UI protocol.  An adapter is
//! responsible for turning a validated [`HcliOperation`] into a durable host
//! call and for reporting the *effective* limits it applied.
//!
//! Each request and response is one JSON object per line.  The outer envelope
//! stays stable while each method owns a strongly typed `params` payload:
//!
//! ```json
//! {"protocol":"hcli.machine.v1","id":"req-7","method":"generate","params":{"prompt":"Explain this repository","runtime_url":"http://127.0.0.1:8787","durable":true}}
//! ```
//!
//! In particular, `generate.durable=true` means the eventual adapter must use
//! a durable external-turn path (including a user-intent event) rather than
//! treating a transient decode as a conversation turn.  This contract never
//! claims that a raw decode, a swarm budget, or a TPS field was actually run.

use crate::hcli_profile::HcliProfile;
use crate::hcli_sources::MAX_CONTEXT_SOURCES;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Protocol marker accepted by this bridge.  Bump this rather than silently
/// changing the meaning of a field consumed by a script or another assistant.
pub const HCLI_JSONL_PROTOCOL: &str = "hcli.machine.v1";
/// Schema for the small, stable model-optimization identity advertised by
/// HCLI capability surfaces. The canonical values are derived from the shared
/// `lab/semantic_taxonomy.json` contract rather than redefined here.
pub const HCLI_MODEL_OPTIMIZATION_IDENTITY_SCHEMA: &str = "hcli.model_optimization.identity.v1";
/// Schema for HCLI's explicit non-execution boundary around Gravity/Condense.
///
/// This is deliberately separate from the shared Hawking taxonomy: the
/// taxonomy names the canonical artifact/runtime identity, while this HCLI
/// contract states what this particular control surface can *not* execute.
pub const HCLI_MODEL_OPTIMIZATION_EXECUTION_BOUNDARY_SCHEMA: &str =
    "hcli.model_optimization.execution_boundary.v1";
/// Stable JSONL error-code spelling returned when a caller asks HCLI itself
/// to run Condense/Gravity model optimization.
pub const HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE: &str = "model_optimization_not_available";
const HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_MESSAGE: &str =
    "HCLI reports Gravity/Condense identity and planning boundaries, but cannot execute model optimization, Condense, packing, or artifact creation";

/// Hard transport bound, independent from the model's context window.
pub const MAX_JSONL_RECORD_BYTES: usize = 16 * 1024 * 1024;
/// Largest prompt or objective accepted by the control plane.
pub const MAX_PROMPT_BYTES: usize = 8 * 1024 * 1024;
/// Largest requested context target.  The runtime must still report whether it
/// could actually honor this request.
pub const MAX_CONTEXT_TOKENS: u32 = 2_000_000;
/// Largest requested completion cap.  It is a request, not a promise that a
/// model artifact or runtime supports that output length.
pub const MAX_OUTPUT_TOKENS: u32 = 131_072;
/// A bounded, explicit upper limit for a single local swarm request.
pub const MAX_SWARM_WORKERS: u16 = 512;

const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_MODEL_BYTES: usize = 512;
const MAX_URL_BYTES: usize = 2_048;
const MAX_METADATA_ITEMS: usize = 64;
const MAX_METADATA_VALUE_BYTES: usize = 4_096;
const MAX_LABEL_BYTES: usize = 256;
const MAX_WARNINGS: usize = 128;
const MAX_RESPONSE_TEXT_BYTES: usize = 64 * 1024 * 1024;

/// The HCLI-facing projection of Hawking's canonical Gravity/Condense
/// taxonomy. This is intentionally identity and claim-boundary metadata, not
/// a report that an optimization was run or a model artifact is loadable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliModelOptimizationIdentity {
    pub schema: String,
    pub taxonomy_schema: String,
    pub canonical_identity: String,
    pub canonical_display_name: String,
    pub public_cli: HcliGravityPublicCli,
    pub condense_engine_operation: HcliCondenseEngineOperation,
    pub claim_limits: HcliModelOptimizationClaimLimits,
    /// A typed, stable refusal contract.  This prevents consumers from
    /// inferring execution capability merely because HCLI exposes the public
    /// Gravity naming and Hawking command references.
    pub execution_boundary: HcliModelOptimizationExecutionBoundary,
    /// Compact forward-only projection of the taxonomy's legacy Condense
    /// schema/command mapping.  A taxonomy-derived default lets older HCLI
    /// identity records decode and validate without pretending their legacy
    /// fields were absent from the current contract.
    #[serde(default)]
    pub legacy_compatibility: HcliLegacyCompatibility,
}

/// The actual CLI surface for an already-produced Gravity artifact. It is kept
/// separate from the public identity so a consumer never mistakes the identity
/// label for proof that a new model-optimization command or runtime exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliGravityPublicCli {
    pub command: String,
    pub status_command: String,
    pub plan_command: String,
    pub environment_variable: String,
    /// The selector variable applies only to a compatibility path. It must not
    /// be inferred to configure the canonical `hawking gravity` command.
    pub environment_variable_scope: String,
    pub command_exists: bool,
    pub compatibility_commands: Vec<HcliGravityCompatibilityCommand>,
}

/// A retained compatibility spelling. It is descriptive only; callers should
/// prefer [`HcliGravityPublicCli::command`] for new automation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliGravityCompatibilityCommand {
    pub command: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_environment_variable: Option<String>,
}

/// Condense is the engine operation beneath the canonical Gravity identity;
/// it is not a competing public artifact identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliCondenseEngineOperation {
    pub id: String,
    pub engine_operation: bool,
    pub canonical_identity: String,
    pub output_identity: String,
}

/// Explicit negatives that prevent a capability report from becoming an
/// accidental optimization-run or DeepSeek V4 runtime claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliModelOptimizationClaimLimits {
    pub hcli_executes_optimization: bool,
    pub v4_runtime_claim: bool,
    pub artifact_identity_is_not_capability_evidence: bool,
}

/// HCLI's current execution boundary for Gravity/Condense work.
///
/// `available=false` is a product capability fact, not a transient runtime
/// health state.  JSONL callers can use [`HcliErrorCode::ModelOptimizationNotAvailable`]
/// and this fixed `refusal_code` instead of scraping explanatory prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliModelOptimizationExecutionBoundary {
    pub schema: String,
    pub available: bool,
    pub refusal_code: String,
    pub message: String,
    /// The canonical Hawking planning surface named by the shared taxonomy.
    /// This is a direction for callers, not proof that a plan will optimize a
    /// model or produce an artifact.
    pub canonical_hawking_plan_command: String,
}

/// HCLI's projection of the shared Condense-to-Gravity compatibility mapping.
///
/// The values below are copied from `lab/semantic_taxonomy.json`; this type is
/// deliberately descriptive and never rewrites a historical receipt or turns
/// a legacy command into an HCLI optimization operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliLegacyCompatibility {
    /// Identifier of the authoritative semantic field/tag mapping.
    pub field_tag_mapping_id: String,
    /// Deprecated legacy schema namespaces and their Gravity successor.
    pub schema_aliases: Vec<HcliLegacySchemaAlias>,
    /// Deprecated command aliases retained by Hawking for compatibility.
    pub command_aliases: Vec<HcliLegacyCommandAlias>,
}

/// One legacy schema namespace projected from the taxonomy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliLegacySchemaAlias {
    pub prefix: String,
    pub status: String,
    pub deprecated: bool,
    pub superseded_by: String,
    pub replacement_command: String,
    pub canonical_identity: String,
    pub operation: String,
    pub field_tag_mapping_id: String,
    pub historical_records_are_rewritten: bool,
}

/// One retained command spelling projected from the taxonomy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliLegacyCommandAlias {
    pub command: String,
    pub status: String,
    pub deprecated: bool,
    pub superseded_by: String,
    pub maps_to_engine_operation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_an_operation: Option<bool>,
}

fn hcli_semantic_taxonomy() -> Value {
    serde_json::from_str(include_str!("../../../lab/semantic_taxonomy.json"))
        .expect("checked-in Hawking semantic taxonomy must be valid JSON")
}

fn hcli_legacy_compatibility_from_taxonomy(taxonomy: &Value) -> HcliLegacyCompatibility {
    let compatibility = taxonomy
        .pointer("/compatibility")
        .and_then(Value::as_object)
        .expect("semantic taxonomy must provide a compatibility object");
    let required_string = |value: &Value, context: &str, field: &str| {
        value
            .get(field)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                panic!("semantic taxonomy must provide non-empty string at {context}/{field}")
            })
            .to_string()
    };
    let required_boolean = |value: &Value, context: &str, field: &str| {
        value
            .get(field)
            .and_then(Value::as_bool)
            .unwrap_or_else(|| {
                panic!("semantic taxonomy must provide boolean at {context}/{field}")
            })
    };
    let schema_aliases = compatibility
        .get("legacy_schema_prefixes")
        .and_then(Value::as_array)
        .expect("semantic taxonomy must provide compatibility/legacy_schema_prefixes array")
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let context = format!("/compatibility/legacy_schema_prefixes/{index}");
            HcliLegacySchemaAlias {
                prefix: required_string(entry, &context, "prefix"),
                status: required_string(entry, &context, "status"),
                deprecated: required_boolean(entry, &context, "deprecated"),
                superseded_by: required_string(entry, &context, "superseded_by"),
                replacement_command: required_string(entry, &context, "replacement_command"),
                canonical_identity: required_string(entry, &context, "canonical_identity"),
                operation: required_string(entry, &context, "operation"),
                field_tag_mapping_id: required_string(entry, &context, "field_tag_mapping_id"),
                historical_records_are_rewritten: required_boolean(
                    entry,
                    &context,
                    "historical_records_are_rewritten",
                ),
            }
        })
        .collect::<Vec<_>>();
    if schema_aliases.is_empty() {
        panic!("semantic taxonomy must provide at least one legacy schema alias");
    }
    let command_aliases = compatibility
        .get("legacy_command_aliases")
        .and_then(Value::as_array)
        .expect("semantic taxonomy must provide compatibility/legacy_command_aliases array")
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let context = format!("/compatibility/legacy_command_aliases/{index}");
            HcliLegacyCommandAlias {
                command: required_string(entry, &context, "command"),
                status: required_string(entry, &context, "status"),
                deprecated: required_boolean(entry, &context, "deprecated"),
                superseded_by: required_string(entry, &context, "superseded_by"),
                maps_to_engine_operation: required_string(
                    entry,
                    &context,
                    "maps_to_engine_operation",
                ),
                not_an_operation: entry.get("not_an_operation").map(|value| {
                    value.as_bool().unwrap_or_else(|| {
                        panic!(
                            "semantic taxonomy must provide boolean at {context}/not_an_operation"
                        )
                    })
                }),
            }
        })
        .collect::<Vec<_>>();
    let field_tag_mapping_id = taxonomy
        .pointer("/tag_contract/field_tag_mapping/id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .expect("semantic taxonomy must provide tag_contract/field_tag_mapping/id")
        .to_string();
    HcliLegacyCompatibility {
        field_tag_mapping_id,
        schema_aliases,
        command_aliases,
    }
}

impl Default for HcliLegacyCompatibility {
    fn default() -> Self {
        hcli_legacy_compatibility_from_taxonomy(&hcli_semantic_taxonomy())
    }
}

/// Build HCLI's stable model-optimization identity from the shared semantic
/// taxonomy. A malformed checked-in taxonomy is a build-contract error, so it
/// is intentionally surfaced immediately instead of silently falling back to
/// stale duplicated naming.
pub fn hcli_model_optimization_identity() -> HcliModelOptimizationIdentity {
    let taxonomy = hcli_semantic_taxonomy();
    let string = |pointer: &str| {
        taxonomy
            .pointer(pointer)
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                panic!("semantic taxonomy must provide non-empty string at {pointer}")
            })
            .to_string()
    };
    let boolean = |pointer: &str| {
        taxonomy
            .pointer(pointer)
            .and_then(Value::as_bool)
            .unwrap_or_else(|| panic!("semantic taxonomy must provide boolean at {pointer}"))
    };
    let hcli_executes_optimization = boolean("/capability_claim_limits/hcli_executes_optimization");
    if hcli_executes_optimization {
        panic!(
            "HCLI's model-optimization execution boundary must be deliberately implemented before the shared taxonomy can advertise execution"
        );
    }
    HcliModelOptimizationIdentity {
        schema: HCLI_MODEL_OPTIMIZATION_IDENTITY_SCHEMA.to_string(),
        taxonomy_schema: string("/schema"),
        canonical_identity: string("/canonical_identity/id"),
        canonical_display_name: string("/canonical_identity/display_name"),
        public_cli: HcliGravityPublicCli {
            command: string("/canonical_identity/public_cli/command"),
            status_command: string("/canonical_identity/public_cli/status_command"),
            plan_command: string("/canonical_identity/public_cli/plan_command"),
            environment_variable: string("/canonical_identity/public_cli/environment_variable"),
            environment_variable_scope: string(
                "/canonical_identity/public_cli/environment_variable_scope",
            ),
            command_exists: boolean("/canonical_identity/public_cli/command_exists"),
            compatibility_commands: taxonomy
                .pointer("/canonical_identity/public_cli/compatibility_commands")
                .and_then(Value::as_array)
                .unwrap_or_else(|| {
                    panic!(
                        "semantic taxonomy must provide compatibility command array at /canonical_identity/public_cli/compatibility_commands"
                    )
                })
                .iter()
                .enumerate()
                .map(|(index, command)| {
                    let prefix = format!(
                        "/canonical_identity/public_cli/compatibility_commands/{index}"
                    );
                    let value = |field: &str| {
                        command
                            .get(field)
                            .and_then(Value::as_str)
                            .unwrap_or_else(|| {
                                panic!(
                                    "semantic taxonomy must provide non-empty string at {prefix}/{field}"
                                )
                            })
                            .to_string()
                    };
                    HcliGravityCompatibilityCommand {
                        command: value("command"),
                        status: value("status"),
                        selector_environment_variable: command
                            .get("selector_environment_variable")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    }
                })
                .collect(),
        },
        condense_engine_operation: HcliCondenseEngineOperation {
            id: string("/operations/condense/id"),
            engine_operation: boolean("/operations/condense/engine_operation"),
            canonical_identity: string("/operations/condense/canonical_identity"),
            output_identity: string("/operations/condense/output_identity"),
        },
        claim_limits: HcliModelOptimizationClaimLimits {
            hcli_executes_optimization,
            v4_runtime_claim: boolean("/capability_claim_limits/v4_runtime_claim"),
            artifact_identity_is_not_capability_evidence: boolean(
                "/capability_claim_limits/artifact_identity_is_not_capability_evidence",
            ),
        },
        execution_boundary: HcliModelOptimizationExecutionBoundary {
            schema: HCLI_MODEL_OPTIMIZATION_EXECUTION_BOUNDARY_SCHEMA.to_string(),
            available: false,
            refusal_code: HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE.to_string(),
            message: HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_MESSAGE.to_string(),
            canonical_hawking_plan_command: string(
                "/canonical_identity/public_cli/plan_command",
            ),
        },
        legacy_compatibility: hcli_legacy_compatibility_from_taxonomy(&taxonomy),
    }
}

impl Default for HcliModelOptimizationIdentity {
    fn default() -> Self {
        hcli_model_optimization_identity()
    }
}

impl HcliModelOptimizationIdentity {
    fn validate(&self) -> HcliBridgeResult<()> {
        let expected = hcli_model_optimization_identity();
        if self != &expected {
            return Err(HcliBridgeError::validation(
                "result.model_optimization_identity",
                "must match the current Hawking Gravity/Condense semantic taxonomy",
            ));
        }
        Ok(())
    }
}

/// Errors returned while parsing or validating an HCLI JSONL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HcliBridgeError {
    /// The line was not a single valid JSON object matching the wire shape.
    Parse { message: String },
    /// A syntactically valid record violated an HCLI contract invariant.
    Validation { field: String, message: String },
    /// Serializing a valid request/response failed unexpectedly.
    Encode { message: String },
}

impl HcliBridgeError {
    fn parse(error: impl fmt::Display) -> Self {
        Self::Parse {
            message: error.to_string(),
        }
    }

    fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    fn encode(error: impl fmt::Display) -> Self {
        Self::Encode {
            message: error.to_string(),
        }
    }

    /// Stable machine-readable error family suitable for [`HcliError`].
    pub const fn code(&self) -> HcliErrorCode {
        match self {
            Self::Parse { .. } => HcliErrorCode::ParseError,
            Self::Validation { .. } => HcliErrorCode::InvalidRequest,
            Self::Encode { .. } => HcliErrorCode::Internal,
        }
    }
}

impl fmt::Display for HcliBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { message } => write!(formatter, "JSONL parse error: {message}"),
            Self::Validation { field, message } => {
                write!(formatter, "invalid {field}: {message}")
            }
            Self::Encode { message } => write!(formatter, "JSONL encode error: {message}"),
        }
    }
}

impl std::error::Error for HcliBridgeError {}

pub type HcliBridgeResult<T> = std::result::Result<T, HcliBridgeError>;

/// A JSONL request envelope.  `params` stays raw at this level so unknown
/// methods cannot accidentally deserialize into a permissive generic type.
/// Use [`HcliRequest::operation`] to obtain the validated typed operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliRequest {
    pub protocol: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<HcliClient>,
    pub method: HcliMethod,
    #[serde(default = "empty_json_object")]
    pub params: Value,
}

impl HcliRequest {
    /// Create a wire envelope from a typed operation.
    pub fn new(id: impl Into<String>, operation: HcliOperation) -> Self {
        let (method, params) = operation.into_wire();
        Self {
            protocol: HCLI_JSONL_PROTOCOL.to_string(),
            id: id.into(),
            client: None,
            method,
            params,
        }
    }

    /// Parse exactly one JSONL record (a single optional terminal line ending
    /// is accepted for direct use with `BufRead::read_line`).
    pub fn parse_jsonl_line(line: &str) -> HcliBridgeResult<Self> {
        let line = strip_terminal_line_ending(line);
        if line.is_empty() {
            return Err(HcliBridgeError::validation("record", "must not be empty"));
        }
        if line.len() > MAX_JSONL_RECORD_BYTES {
            return Err(HcliBridgeError::validation(
                "record",
                format!("exceeds {MAX_JSONL_RECORD_BYTES} byte JSONL limit"),
            ));
        }
        if line.contains('\n') || line.contains('\r') {
            return Err(HcliBridgeError::validation(
                "record",
                "must contain one JSON object, not multiple lines",
            ));
        }
        let request: Self = serde_json::from_str(line).map_err(HcliBridgeError::parse)?;
        request.validate()?;
        Ok(request)
    }

    /// Serialize a validated request without a trailing newline.  A JSONL
    /// writer should append one `\n` after each returned record.
    pub fn to_jsonl_line(&self) -> HcliBridgeResult<String> {
        self.validate()?;
        serde_json::to_string(self).map_err(HcliBridgeError::encode)
    }

    /// Validate the envelope and its typed method parameters.
    pub fn validate(&self) -> HcliBridgeResult<()> {
        if self.protocol != HCLI_JSONL_PROTOCOL {
            return Err(HcliBridgeError::validation(
                "protocol",
                format!("expected {HCLI_JSONL_PROTOCOL:?}"),
            ));
        }
        validate_identifier("id", &self.id)?;
        if let Some(client) = &self.client {
            client.validate()?;
        }
        self.operation()?.validate()
    }

    /// Decode and validate the raw `params` value into an operation-specific
    /// request.  An adapter should call this before selecting any backend path.
    pub fn operation(&self) -> HcliBridgeResult<HcliOperation> {
        let operation = HcliOperation::from_wire(self.method, self.params.clone())?;
        operation.validate()?;
        Ok(operation)
    }

    /// Produce an owned, validated request convenient for a dispatcher.
    pub fn into_validated(self) -> HcliBridgeResult<ValidatedHcliRequest> {
        self.validate()?;
        let operation = HcliOperation::from_wire(self.method, self.params)?;
        Ok(ValidatedHcliRequest {
            id: self.id,
            client: self.client,
            operation,
        })
    }
}

/// A request after protocol, envelope, and method parameters are all valid.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedHcliRequest {
    pub id: String,
    pub client: Option<HcliClient>,
    pub operation: HcliOperation,
}

/// Optional origin information for audit logs.  It is metadata only: a client
/// name never changes authority or effect permissions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliClient {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl HcliClient {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_short_text("client.name", &self.name, MAX_IDENTIFIER_BYTES)?;
        if let Some(version) = &self.version {
            validate_short_text("client.version", version, MAX_IDENTIFIER_BYTES)?;
        }
        Ok(())
    }
}

/// Names of the machine operations supported by this protocol revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HcliMethod {
    Capabilities,
    /// A deliberately non-executing request that returns HCLI's stable
    /// Gravity/Condense optimization refusal.  It lets external controllers
    /// distinguish an intentional product boundary from an unknown method.
    ModelOptimization,
    Generate,
    Agent,
    Swarm,
    Session,
    Status,
}

impl HcliMethod {
    pub const ALL: [Self; 7] = [
        Self::Capabilities,
        Self::ModelOptimization,
        Self::Generate,
        Self::Agent,
        Self::Swarm,
        Self::Session,
        Self::Status,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Capabilities => "capabilities",
            Self::ModelOptimization => "model_optimization",
            Self::Generate => "generate",
            Self::Agent => "agent",
            Self::Swarm => "swarm",
            Self::Session => "session",
            Self::Status => "status",
        }
    }
}

/// Typed request variants used by an HCLI dispatcher after it validates the
/// outer JSONL envelope.
#[derive(Debug, Clone, PartialEq)]
pub enum HcliOperation {
    Capabilities(CapabilitiesRequest),
    ModelOptimization(ModelOptimizationRequest),
    Generate(GenerateRequest),
    Agent(AgentRequest),
    Swarm(SwarmRequest),
    Session(SessionRequest),
    Status(StatusRequest),
}

impl HcliOperation {
    pub const fn method(&self) -> HcliMethod {
        match self {
            Self::Capabilities(_) => HcliMethod::Capabilities,
            Self::ModelOptimization(_) => HcliMethod::ModelOptimization,
            Self::Generate(_) => HcliMethod::Generate,
            Self::Agent(_) => HcliMethod::Agent,
            Self::Swarm(_) => HcliMethod::Swarm,
            Self::Session(_) => HcliMethod::Session,
            Self::Status(_) => HcliMethod::Status,
        }
    }

    fn into_wire(self) -> (HcliMethod, Value) {
        let method = self.method();
        let params = match self {
            Self::Capabilities(request) => json_value(request),
            Self::ModelOptimization(request) => json_value(request),
            Self::Generate(request) => json_value(request),
            Self::Agent(request) => json_value(request),
            Self::Swarm(request) => json_value(request),
            Self::Session(request) => json_value(request),
            Self::Status(request) => json_value(request),
        };
        (method, params)
    }

    fn from_wire(method: HcliMethod, params: Value) -> HcliBridgeResult<Self> {
        if !params.is_object() {
            return Err(HcliBridgeError::validation(
                "params",
                "must be a JSON object",
            ));
        }
        match method {
            HcliMethod::Capabilities => decode_params(params).map(Self::Capabilities),
            HcliMethod::ModelOptimization => decode_params(params).map(Self::ModelOptimization),
            HcliMethod::Generate => decode_params(params).map(Self::Generate),
            HcliMethod::Agent => decode_params(params).map(Self::Agent),
            HcliMethod::Swarm => decode_params(params).map(Self::Swarm),
            HcliMethod::Session => decode_params(params).map(Self::Session),
            HcliMethod::Status => decode_params(params).map(Self::Status),
        }
    }

    pub fn validate(&self) -> HcliBridgeResult<()> {
        match self {
            Self::Capabilities(request) => request.validate(),
            Self::ModelOptimization(request) => request.validate(),
            Self::Generate(request) => request.validate(),
            Self::Agent(request) => request.validate(),
            Self::Swarm(request) => request.validate(),
            Self::Session(request) => request.validate(),
            Self::Status(request) => request.validate(),
        }
    }
}

/// Capability sections an external controller may ask the host to describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityArea {
    Runtime,
    Generate,
    Agent,
    Swarm,
    Session,
    Storage,
    Tools,
}

/// `capabilities` method parameters.  An empty `include` means every section.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilitiesRequest {
    #[serde(default)]
    pub include: Vec<CapabilityArea>,
    #[serde(default)]
    pub include_experimental: bool,
}

impl CapabilitiesRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        ensure_unique("params.include", self.include.iter().copied())
    }
}

/// `model_optimization` intentionally takes no controls.  HCLI must refuse
/// before accepting weights, targets, or execution knobs so none can be
/// mistaken for a partially applied Condense/Gravity operation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOptimizationRequest {}

impl ModelOptimizationRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        Ok(())
    }
}

/// Sampling controls for a direct generation.  All fields are optional so the
/// runtime's model-specific defaults can stay explicit in its response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl Default for SamplingRequest {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            seed: None,
        }
    }
}

impl SamplingRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        if let Some(temperature) = self.temperature {
            if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
                return Err(HcliBridgeError::validation(
                    "params.sampling.temperature",
                    "must be a finite number from 0.0 through 2.0",
                ));
            }
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
                return Err(HcliBridgeError::validation(
                    "params.sampling.top_p",
                    "must be a finite number greater than 0.0 and at most 1.0",
                ));
            }
        }
        Ok(())
    }
}

/// Direct model generation parameters.
///
/// A non-empty `runtime_url` identifies a live local Hawking endpoint.  A
/// future adapter may also support a configured supervised runtime when it is
/// absent, but must report which endpoint/artifact it actually used.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default = "default_generate_output_tokens")]
    pub max_output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u32>,
    #[serde(default)]
    pub sampling: SamplingRequest,
    #[serde(default = "default_stream")]
    pub stream: bool,
    /// If true, the adapter must append a durable user-intent event before
    /// generating and return the durable event identifiers it actually wrote.
    #[serde(default = "default_durable_turn")]
    pub durable: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl GenerateRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_text("params.prompt", &self.prompt, MAX_PROMPT_BYTES)?;
        validate_optional_identifier("params.session_id", self.session_id.as_deref())?;
        validate_optional_runtime_url("params.runtime_url", self.runtime_url.as_deref())?;
        validate_optional_model("params.model", self.model.as_deref())?;
        validate_range(
            "params.max_output_tokens",
            self.max_output_tokens,
            1,
            MAX_OUTPUT_TOKENS,
        )?;
        validate_optional_context_tokens("params.max_context_tokens", self.max_context_tokens)?;
        self.sampling.validate()?;
        validate_metadata("params.metadata", &self.metadata)
    }
}

/// Effects are deliberately controlled separately from compute profile.  An
/// adapter must still pass every tool/effect through the host policy layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HcliAutonomy {
    ReadOnly,
    SuggestOnly,
    FullAuto,
}

impl Default for HcliAutonomy {
    fn default() -> Self {
        Self::SuggestOnly
    }
}

/// Optional agent-level overrides on top of a named [`HcliProfile`].  `None`
/// means the profile decides; an adapter must report its resulting effective
/// budget rather than assuming requested breadth was realized.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentBudgetOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_transitions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wallclock_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_subagents: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_breadth: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_consistency_k: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u32>,
    /// Informational compute target only.  Zero is rejected here so HCLI never
    /// represents an unbounded token request as a normal preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_budget_hint: Option<u64>,
}

impl AgentBudgetOverride {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_optional_range(
            "params.budget.max_transitions",
            self.max_transitions,
            1,
            100_000,
        )?;
        validate_optional_range(
            "params.budget.max_wallclock_ms",
            self.max_wallclock_ms,
            1,
            86_400_000,
        )?;
        validate_optional_range(
            "params.budget.max_tool_calls",
            self.max_tool_calls,
            1,
            1_000_000,
        )?;
        if let Some(max_subagents) = self.max_subagents {
            if max_subagents > u32::from(MAX_SWARM_WORKERS) {
                return Err(HcliBridgeError::validation(
                    "params.budget.max_subagents",
                    format!("must be at most {MAX_SWARM_WORKERS}"),
                ));
            }
        }
        validate_optional_range("params.budget.search_breadth", self.search_breadth, 1, 64)?;
        validate_optional_range(
            "params.budget.self_consistency_k",
            self.self_consistency_k,
            1,
            64,
        )?;
        validate_optional_context_tokens(
            "params.budget.max_context_tokens",
            self.max_context_tokens,
        )?;
        validate_optional_range(
            "params.budget.token_budget_hint",
            self.token_budget_hint,
            1,
            1_000_000_000,
        )?;
        Ok(())
    }
}

/// An agent-goal request.  Compute can be high (`power`/`maximum` profiles and
/// bounded overrides), but it does not imply permission to run effects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentRequest {
    pub goal: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default = "default_hcli_profile")]
    pub profile: HcliProfile,
    #[serde(default)]
    pub budget: AgentBudgetOverride,
    /// Existing workspace-local HCLI evidence references selected explicitly
    /// for this one agent run.  The adapter resolves them through the same
    /// bounded derivative-only path as `hcli agent --attach`; raw object bytes
    /// and URLs never enter the JSONL request or model prompt.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<String>,
    #[serde(default)]
    pub autonomy: HcliAutonomy,
    /// Explicit acknowledgement required before an adapter may even request
    /// full-auto effects.  It does not bypass host approvals/policy.
    #[serde(default)]
    pub acknowledge_effects: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl AgentRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_text("params.goal", &self.goal, MAX_PROMPT_BYTES)?;
        validate_optional_identifier("params.session_id", self.session_id.as_deref())?;
        validate_optional_runtime_url("params.runtime_url", self.runtime_url.as_deref())?;
        validate_optional_model("params.model", self.model.as_deref())?;
        self.budget.validate()?;
        validate_source_attachment_refs("params.attachments", &self.attachments)?;
        if matches!(self.autonomy, HcliAutonomy::FullAuto) && !self.acknowledge_effects {
            return Err(HcliBridgeError::validation(
                "params.acknowledge_effects",
                "must be true when autonomy is full_auto",
            ));
        }
        validate_metadata("params.metadata", &self.metadata)
    }
}

/// Scheduling pattern requested for a distributed/local swarm.  A future
/// adapter must return realized workers and explicitly mark unsupported
/// patterns rather than silently downgrading them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmPattern {
    Independent,
    MapReduce,
    Review,
    Debate,
}

impl Default for SwarmPattern {
    fn default() -> Self {
        Self::Independent
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmMergeStrategy {
    BestOf,
    Synthesize,
    Consensus,
}

impl Default for SwarmMergeStrategy {
    fn default() -> Self {
        Self::Synthesize
    }
}

/// Parameters for a finite, observable swarm run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwarmRequest {
    pub goal: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default = "default_hcli_profile")]
    pub profile: HcliProfile,
    #[serde(default = "default_swarm_workers")]
    pub workers: u16,
    #[serde(default = "default_swarm_concurrency")]
    pub max_concurrency: u16,
    #[serde(default = "default_worker_transitions")]
    pub max_transitions_per_worker: u32,
    #[serde(default = "default_total_model_calls")]
    pub max_total_model_calls: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u32>,
    #[serde(default)]
    pub pattern: SwarmPattern,
    #[serde(default)]
    pub merge: SwarmMergeStrategy,
    #[serde(default)]
    pub autonomy: HcliAutonomy,
    #[serde(default)]
    pub acknowledge_effects: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl SwarmRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_text("params.goal", &self.goal, MAX_PROMPT_BYTES)?;
        validate_optional_identifier("params.session_id", self.session_id.as_deref())?;
        validate_optional_runtime_url("params.runtime_url", self.runtime_url.as_deref())?;
        validate_optional_model("params.model", self.model.as_deref())?;
        validate_range("params.workers", self.workers, 2, MAX_SWARM_WORKERS)?;
        validate_range(
            "params.max_concurrency",
            self.max_concurrency,
            1,
            self.workers,
        )?;
        validate_range(
            "params.max_transitions_per_worker",
            self.max_transitions_per_worker,
            1,
            100_000,
        )?;
        validate_range(
            "params.max_total_model_calls",
            self.max_total_model_calls,
            1,
            10_000_000,
        )?;
        validate_optional_context_tokens("params.max_context_tokens", self.max_context_tokens)?;
        if matches!(self.autonomy, HcliAutonomy::FullAuto) && !self.acknowledge_effects {
            return Err(HcliBridgeError::validation(
                "params.acknowledge_effects",
                "must be true when autonomy is full_auto",
            ));
        }
        validate_metadata("params.metadata", &self.metadata)
    }
}

/// Session lifecycle requests.  They are separated from `status` so scripts
/// can create/resume/list sessions without pretending to initiate inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum SessionRequest {
    Create {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    Resume {
        session_id: String,
    },
    List {
        #[serde(default = "default_session_list_limit")]
        limit: u16,
    },
    Close {
        session_id: String,
    },
}

impl SessionRequest {
    pub const fn action(&self) -> SessionAction {
        match self {
            Self::Create { .. } => SessionAction::Create,
            Self::Resume { .. } => SessionAction::Resume,
            Self::List { .. } => SessionAction::List,
            Self::Close { .. } => SessionAction::Close,
        }
    }

    pub fn validate(&self) -> HcliBridgeResult<()> {
        match self {
            Self::Create { label } => {
                if let Some(label) = label {
                    validate_text("params.label", label, MAX_LABEL_BYTES)?;
                }
            }
            Self::Resume { session_id } | Self::Close { session_id } => {
                validate_identifier("params.session_id", session_id)?;
            }
            Self::List { limit } => validate_range("params.limit", *limit, 1, 1_000)?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionAction {
    Create,
    Resume,
    List,
    Close,
}

/// Individual status groups that can be queried without doing work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatusArea {
    Host,
    Runtime,
    Models,
    Session,
    Agents,
    Swarms,
    Storage,
}

/// `status` method parameters.  Empty `include` requests all status groups.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default)]
    pub include: Vec<StatusArea>,
    /// An adapter may use this to refresh live runtime health before replying.
    #[serde(default)]
    pub refresh: bool,
}

impl StatusRequest {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_optional_identifier("params.session_id", self.session_id.as_deref())?;
        ensure_unique("params.include", self.include.iter().copied())
    }
}

/// Common lifecycle state returned from an accepted control operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    Accepted,
    Running,
    Completed,
    Paused,
    Blocked,
    Failed,
    Cancelled,
}

/// HCLI response envelope.  `ok` means the control-plane request was accepted
/// and produced a typed result; it does not mean a run reached `completed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliResponse {
    pub protocol: String,
    pub id: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<HcliResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<HcliError>,
}

impl HcliResponse {
    pub fn success(id: impl Into<String>, result: HcliResult) -> HcliBridgeResult<Self> {
        let response = Self {
            protocol: HCLI_JSONL_PROTOCOL.to_string(),
            id: id.into(),
            ok: true,
            result: Some(result),
            error: None,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn failure(id: impl Into<String>, error: HcliError) -> HcliBridgeResult<Self> {
        let response = Self {
            protocol: HCLI_JSONL_PROTOCOL.to_string(),
            id: id.into(),
            ok: false,
            result: None,
            error: Some(error),
        };
        response.validate()?;
        Ok(response)
    }

    pub fn parse_jsonl_line(line: &str) -> HcliBridgeResult<Self> {
        let line = strip_terminal_line_ending(line);
        if line.is_empty() {
            return Err(HcliBridgeError::validation("record", "must not be empty"));
        }
        if line.len() > MAX_JSONL_RECORD_BYTES {
            return Err(HcliBridgeError::validation(
                "record",
                format!("exceeds {MAX_JSONL_RECORD_BYTES} byte JSONL limit"),
            ));
        }
        if line.contains('\n') || line.contains('\r') {
            return Err(HcliBridgeError::validation(
                "record",
                "must contain one JSON object, not multiple lines",
            ));
        }
        let response: Self = serde_json::from_str(line).map_err(HcliBridgeError::parse)?;
        response.validate()?;
        Ok(response)
    }

    pub fn to_jsonl_line(&self) -> HcliBridgeResult<String> {
        self.validate()?;
        serde_json::to_string(self).map_err(HcliBridgeError::encode)
    }

    pub fn validate(&self) -> HcliBridgeResult<()> {
        if self.protocol != HCLI_JSONL_PROTOCOL {
            return Err(HcliBridgeError::validation(
                "protocol",
                format!("expected {HCLI_JSONL_PROTOCOL:?}"),
            ));
        }
        validate_identifier("id", &self.id)?;
        match (self.ok, &self.result, &self.error) {
            (true, Some(result), None) => result.validate(),
            (false, None, Some(error)) => error.validate(),
            (true, None, _) => Err(HcliBridgeError::validation(
                "result",
                "successful response requires one result and no error",
            )),
            (false, _, None) => Err(HcliBridgeError::validation(
                "error",
                "failed response requires one error and no result",
            )),
            _ => Err(HcliBridgeError::validation(
                "response",
                "must contain exactly one of result or error consistent with ok",
            )),
        }
    }
}

/// Typed successful result payloads.  A dispatcher must return the matching
/// variant for the requested method, but that cross-check belongs to the
/// adapter because a response is also valid as a standalone JSONL record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum HcliResult {
    Capabilities(CapabilitiesResponse),
    Generate(GenerateResponse),
    Agent(AgentResponse),
    Swarm(SwarmResponse),
    Session(SessionResponse),
    Status(StatusResponse),
}

impl HcliResult {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        match self {
            Self::Capabilities(response) => response.validate(),
            Self::Generate(response) => response.validate(),
            Self::Agent(response) => response.validate(),
            Self::Swarm(response) => response.validate(),
            Self::Session(response) => response.validate(),
            Self::Status(response) => response.validate(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HcliErrorCode {
    ParseError,
    InvalidRequest,
    Unsupported,
    Unavailable,
    /// Stable, non-retryable refusal for a requested HCLI-side
    /// Gravity/Condense optimization.  This is intentionally more precise
    /// than `unavailable`, which can describe transient runtime health.
    ModelOptimizationNotAvailable,
    Conflict,
    Cancelled,
    Internal,
}

/// Structured negative response.  The caller can distinguish a protocol error
/// from a runtime that was merely unavailable without scraping a human string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliError {
    pub code: HcliErrorCode,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

impl HcliError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: HcliErrorCode::InvalidRequest,
            message: message.into(),
            retryable: false,
        }
    }

    /// The exact error returned by HCLI's non-executing model-optimization
    /// control path.  Keep this paired with
    /// [`HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE`].
    pub fn model_optimization_not_available() -> Self {
        Self {
            code: HcliErrorCode::ModelOptimizationNotAvailable,
            message: HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_MESSAGE.to_string(),
            retryable: false,
        }
    }

    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_short_text("error.message", &self.message, MAX_LABEL_BYTES)
    }
}

/// One advertised control operation and whether it is actually available on
/// the current host.  `available=false` is the honest representation for a
/// planned but unwired HCLI feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodCapability {
    pub method: HcliMethod,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl MethodCapability {
    fn validate(&self) -> HcliBridgeResult<()> {
        if let Some(detail) = &self.detail {
            validate_short_text("result.operations.detail", detail, MAX_LABEL_BYTES)?;
        }
        Ok(())
    }
}

/// Hard control-plane limits advertised alongside runtime-specific capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HcliLimits {
    pub max_jsonl_record_bytes: usize,
    pub max_prompt_bytes: usize,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
    pub max_swarm_workers: u16,
}

impl Default for HcliLimits {
    fn default() -> Self {
        Self {
            max_jsonl_record_bytes: MAX_JSONL_RECORD_BYTES,
            max_prompt_bytes: MAX_PROMPT_BYTES,
            max_context_tokens: MAX_CONTEXT_TOKENS,
            max_output_tokens: MAX_OUTPUT_TOKENS,
            max_swarm_workers: MAX_SWARM_WORKERS,
        }
    }
}

impl HcliLimits {
    fn validate(&self) -> HcliBridgeResult<()> {
        if self.max_jsonl_record_bytes == 0
            || self.max_prompt_bytes == 0
            || self.max_context_tokens == 0
            || self.max_output_tokens == 0
            || self.max_swarm_workers == 0
        {
            return Err(HcliBridgeError::validation(
                "result.limits",
                "all advertised limits must be positive",
            ));
        }
        Ok(())
    }
}

/// Result of a `capabilities` query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilitiesResponse {
    pub operations: Vec<MethodCapability>,
    /// Canonical model-optimization naming and explicit claim boundaries. A
    /// default keeps older bridge fixtures decodable while all newly emitted
    /// capability responses carry this stable block.
    #[serde(default)]
    pub model_optimization_identity: HcliModelOptimizationIdentity,
    #[serde(default)]
    pub limits: HcliLimits,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl CapabilitiesResponse {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        ensure_unique(
            "result.operations",
            self.operations.iter().map(|item| item.method),
        )?;
        self.model_optimization_identity.validate()?;
        self.limits.validate()?;
        validate_notes("result.notes", &self.notes)?;
        for operation in &self.operations {
            operation.validate()?;
        }
        Ok(())
    }
}

/// Runtime decode telemetry.  `decode_forwards_per_second` is only permitted
/// when its denominator is present, preventing a headless wall-clock result
/// from being mislabeled as model TPS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeTelemetry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_decode_forwards: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_forwards_per_second: Option<f64>,
}

impl DecodeTelemetry {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        match (self.decode_ms, self.completed_decode_forwards) {
            (Some(milliseconds), Some(forwards)) => {
                if !milliseconds.is_finite() || milliseconds <= 0.0 {
                    return Err(HcliBridgeError::validation(
                        "result.telemetry.decode_ms",
                        "must be finite and greater than zero",
                    ));
                }
                if forwards == 0 {
                    return Err(HcliBridgeError::validation(
                        "result.telemetry.completed_decode_forwards",
                        "must be greater than zero when decode_ms is present",
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(HcliBridgeError::validation(
                    "result.telemetry",
                    "decode_ms and completed_decode_forwards must be reported together",
                ));
            }
        }
        if let Some(tps) = self.decode_forwards_per_second {
            let (milliseconds, forwards) = self
                .decode_ms
                .zip(self.completed_decode_forwards)
                .ok_or_else(|| {
                    HcliBridgeError::validation(
                        "result.telemetry.decode_forwards_per_second",
                        "requires decode_ms and completed_decode_forwards",
                    )
                })?;
            if !tps.is_finite() || tps <= 0.0 {
                return Err(HcliBridgeError::validation(
                    "result.telemetry.decode_forwards_per_second",
                    "must be finite and greater than zero",
                ));
            }
            let expected = forwards as f64 * 1_000.0 / milliseconds;
            let tolerance = expected.mul_add(0.05, 0.05);
            if (expected - tps).abs() > tolerance {
                return Err(HcliBridgeError::validation(
                    "result.telemetry.decode_forwards_per_second",
                    "does not agree with completed_decode_forwards / decode_ms",
                ));
            }
        }
        Ok(())
    }
}

/// Durable event references reported only when a `generate` adapter actually
/// wrote them.  A direct ephemeral decode leaves this field absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableTurnReceipt {
    pub user_event_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assistant_event_id: Option<String>,
}

impl DurableTurnReceipt {
    fn validate(&self) -> HcliBridgeResult<()> {
        validate_identifier("result.durable_turn.user_event_id", &self.user_event_id)?;
        validate_optional_identifier(
            "result.durable_turn.assistant_event_id",
            self.assistant_event_id.as_deref(),
        )
    }
}

/// Result of a direct `generate` request.  `status=accepted` is valid for a
/// streaming response, while a terminal response may include output and exact
/// decode telemetry if the runtime supplied it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateResponse {
    pub status: OperationStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_turn: Option<DurableTurnReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<DecodeTelemetry>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl GenerateResponse {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_optional_identifier("result.session_id", self.session_id.as_deref())?;
        validate_optional_model("result.model", self.model.as_deref())?;
        if let Some(output) = &self.output {
            if output.len() > MAX_RESPONSE_TEXT_BYTES {
                return Err(HcliBridgeError::validation(
                    "result.output",
                    format!("exceeds {MAX_RESPONSE_TEXT_BYTES} byte response limit"),
                ));
            }
            ensure_no_nul("result.output", output)?;
        }
        validate_optional_identifier("result.stream_id", self.stream_id.as_deref())?;
        if let Some(receipt) = &self.durable_turn {
            receipt.validate()?;
        }
        if let Some(telemetry) = &self.telemetry {
            telemetry.validate()?;
        }
        validate_notes("result.warnings", &self.warnings)
    }
}

/// Work that actually occurred in an agent run.  Every field is optional so a
/// backend can distinguish an unavailable metric from a genuine zero.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentRealization {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transitions: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_calls: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagents_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_elapsed_ms: Option<u64>,
}

/// Result of an `agent` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentResponse {
    pub status: OperationStatus,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub requested_profile: HcliProfile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_profile: Option<HcliProfile>,
    #[serde(default)]
    pub realized: AgentRealization,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_path: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl AgentResponse {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_identifier("result.session_id", &self.session_id)?;
        validate_optional_identifier("result.run_id", self.run_id.as_deref())?;
        if let Some(receipt_path) = &self.receipt_path {
            validate_short_text("result.receipt_path", receipt_path, 4_096)?;
        }
        validate_notes("result.warnings", &self.warnings)
    }
}

/// Result of a `swarm` request.  Requested and effective worker counts remain
/// separate because resource admission may lawfully lower concurrency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwarmResponse {
    pub status: OperationStatus,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swarm_id: Option<String>,
    pub workers_requested: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers_effective: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers_completed: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report_path: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl SwarmResponse {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        validate_identifier("result.session_id", &self.session_id)?;
        validate_optional_identifier("result.swarm_id", self.swarm_id.as_deref())?;
        validate_range(
            "result.workers_requested",
            self.workers_requested,
            1,
            MAX_SWARM_WORKERS,
        )?;
        if let Some(effective) = self.workers_effective {
            validate_range(
                "result.workers_effective",
                effective,
                1,
                self.workers_requested,
            )?;
        }
        if let Some(completed) = self.workers_completed {
            let upper = self.workers_effective.unwrap_or(self.workers_requested);
            validate_range("result.workers_completed", completed, 0, upper)?;
        }
        if let Some(report_path) = &self.report_path {
            validate_short_text("result.report_path", report_path, 4_096)?;
        }
        validate_notes("result.warnings", &self.warnings)
    }
}

/// Compact session summary used by session and status responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionSummary {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub state: OperationStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_ms: Option<u64>,
}

impl SessionSummary {
    fn validate(&self) -> HcliBridgeResult<()> {
        validate_identifier("result.session.session_id", &self.session_id)?;
        if let Some(label) = &self.label {
            validate_short_text("result.session.label", label, MAX_LABEL_BYTES)?;
        }
        Ok(())
    }
}

/// Result of a session lifecycle operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionResponse {
    pub status: OperationStatus,
    pub action: SessionAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionSummary>,
    #[serde(default)]
    pub sessions: Vec<SessionSummary>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl SessionResponse {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        if self.sessions.len() > 1_000 {
            return Err(HcliBridgeError::validation(
                "result.sessions",
                "must contain at most 1000 entries",
            ));
        }
        if let Some(session) = &self.session {
            session.validate()?;
        }
        let mut ids = BTreeSet::new();
        for session in &self.sessions {
            session.validate()?;
            if !ids.insert(session.session_id.as_str()) {
                return Err(HcliBridgeError::validation(
                    "result.sessions",
                    "must not contain duplicate session_id values",
                ));
            }
        }
        validate_notes("result.warnings", &self.warnings)
    }
}

/// Health state of a component named in a `status` response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentState {
    Ready,
    Booting,
    Degraded,
    Down,
    Unavailable,
    Unknown,
}

/// A named, structured status datum.  Values are intentionally JSON rather
/// than a fake universal schema because runtime, storage, and fleet state are
/// genuinely different domains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentStatus {
    pub state: ComponentState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ComponentStatus {
    fn validate(&self) -> HcliBridgeResult<()> {
        if let Some(detail) = &self.detail {
            validate_short_text("result.status.detail", detail, 4_096)?;
        }
        Ok(())
    }
}

/// Result of a non-mutating `status` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusResponse {
    #[serde(default)]
    pub components: BTreeMap<String, ComponentStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionSummary>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl StatusResponse {
    pub fn validate(&self) -> HcliBridgeResult<()> {
        if self.components.len() > 128 {
            return Err(HcliBridgeError::validation(
                "result.components",
                "must contain at most 128 components",
            ));
        }
        for (name, component) in &self.components {
            validate_identifier("result.components key", name)?;
            component.validate()?;
        }
        if let Some(session) = &self.session {
            session.validate()?;
        }
        validate_notes("result.warnings", &self.warnings)
    }
}

fn empty_json_object() -> Value {
    Value::Object(Map::new())
}

fn json_value<T: Serialize>(value: T) -> Value {
    // Every type passed here is derived Serialize and this is an in-process
    // typed conversion, so inability to serialize is a programmer error rather
    // than a malformed client record.
    serde_json::to_value(value).expect("HCLI bridge request must serialize")
}

fn decode_params<T: for<'de> Deserialize<'de>>(params: Value) -> HcliBridgeResult<T> {
    serde_json::from_value(params).map_err(HcliBridgeError::parse)
}

fn strip_terminal_line_ending(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

fn default_generate_output_tokens() -> u32 {
    512
}

fn default_stream() -> bool {
    true
}

fn default_durable_turn() -> bool {
    true
}

fn default_hcli_profile() -> HcliProfile {
    // HCLI is the high-capability local surface.  The profile remains bounded
    // and effect-gated; callers can explicitly choose balanced or maximum.
    HcliProfile::Power
}

fn default_swarm_workers() -> u16 {
    8
}

fn default_swarm_concurrency() -> u16 {
    8
}

fn default_worker_transitions() -> u32 {
    480
}

fn default_total_model_calls() -> u32 {
    10_000
}

fn default_session_list_limit() -> u16 {
    100
}

fn validate_identifier(field: &str, value: &str) -> HcliBridgeResult<()> {
    if value.is_empty() || value.len() > MAX_IDENTIFIER_BYTES {
        return Err(HcliBridgeError::validation(
            field,
            format!("must be 1 through {MAX_IDENTIFIER_BYTES} bytes"),
        ));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(HcliBridgeError::validation(
            field,
            "must use only ASCII letters, digits, underscore, hyphen, or dot",
        ));
    }
    Ok(())
}

fn validate_optional_identifier(field: &str, value: Option<&str>) -> HcliBridgeResult<()> {
    if let Some(value) = value {
        validate_identifier(field, value)?;
    }
    Ok(())
}

fn validate_optional_model(field: &str, value: Option<&str>) -> HcliBridgeResult<()> {
    if let Some(value) = value {
        validate_short_text(field, value, MAX_MODEL_BYTES)?;
    }
    Ok(())
}

fn validate_optional_runtime_url(field: &str, value: Option<&str>) -> HcliBridgeResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.len() > MAX_URL_BYTES || value.trim() != value {
        return Err(HcliBridgeError::validation(
            field,
            format!("must be a trimmed URL of at most {MAX_URL_BYTES} bytes"),
        ));
    }
    if !(value.starts_with("http://") || value.starts_with("https://"))
        || value.contains(char::is_whitespace)
    {
        return Err(HcliBridgeError::validation(
            field,
            "must be an http:// or https:// URL without whitespace",
        ));
    }
    Ok(())
}

fn validate_optional_context_tokens(field: &str, value: Option<u32>) -> HcliBridgeResult<()> {
    if let Some(value) = value {
        validate_range(field, value, 1, MAX_CONTEXT_TOKENS)?;
    }
    Ok(())
}

/// Validate only the stable *wire identity* of an HCLI evidence attachment.
///
/// Existence, readiness, permission, duplicate-content detection, derivative
/// availability, and context-size accounting remain the responsibility of
/// `HcliSourceStore::select_context` at dispatch time.  Keeping that lookup
/// out of the protocol parser makes JSONL parsing side-effect-free while still
/// refusing malformed or unbounded controls before a model call can start.
fn validate_source_attachment_refs(field: &str, attachments: &[String]) -> HcliBridgeResult<()> {
    if attachments.len() > MAX_CONTEXT_SOURCES {
        return Err(HcliBridgeError::validation(
            field,
            format!("must contain at most {MAX_CONTEXT_SOURCES} explicit local source references"),
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, attachment) in attachments.iter().enumerate() {
        let item_field = format!("{field}[{index}]");
        if attachment.trim() != attachment {
            return Err(HcliBridgeError::validation(item_field, "must be trimmed"));
        }
        let valid_object_ref = attachment.starts_with("oref_")
            && attachment.len() > "oref_".len()
            && attachment.len() <= MAX_IDENTIFIER_BYTES
            && attachment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
        let hash_suffix = attachment.strip_prefix("blake3:");
        let valid_content_hash = hash_suffix.is_some_and(|suffix| {
            suffix.len() == 64 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
        });
        if !valid_object_ref && !valid_content_hash {
            return Err(HcliBridgeError::validation(
                item_field,
                "must be an oref_… local reference or blake3:<64-hex> content hash",
            ));
        }
        if !seen.insert(attachment) {
            return Err(HcliBridgeError::validation(
                field,
                "must not contain duplicate attachment references",
            ));
        }
    }
    Ok(())
}

fn validate_metadata(field: &str, metadata: &BTreeMap<String, String>) -> HcliBridgeResult<()> {
    if metadata.len() > MAX_METADATA_ITEMS {
        return Err(HcliBridgeError::validation(
            field,
            format!("must contain at most {MAX_METADATA_ITEMS} entries"),
        ));
    }
    for (key, value) in metadata {
        validate_identifier(&format!("{field}.{key}"), key)?;
        validate_short_text(&format!("{field}.{key}"), value, MAX_METADATA_VALUE_BYTES)?;
    }
    Ok(())
}

fn validate_notes(field: &str, notes: &[String]) -> HcliBridgeResult<()> {
    if notes.len() > MAX_WARNINGS {
        return Err(HcliBridgeError::validation(
            field,
            format!("must contain at most {MAX_WARNINGS} entries"),
        ));
    }
    for note in notes {
        validate_short_text(field, note, MAX_LABEL_BYTES)?;
    }
    Ok(())
}

fn validate_short_text(field: &str, value: &str, maximum: usize) -> HcliBridgeResult<()> {
    validate_text(field, value, maximum)
}

fn validate_text(field: &str, value: &str, maximum: usize) -> HcliBridgeResult<()> {
    if value.trim().is_empty() {
        return Err(HcliBridgeError::validation(field, "must not be blank"));
    }
    if value.len() > maximum {
        return Err(HcliBridgeError::validation(
            field,
            format!("exceeds {maximum} byte limit"),
        ));
    }
    ensure_no_nul(field, value)
}

fn ensure_no_nul(field: &str, value: &str) -> HcliBridgeResult<()> {
    if value.contains('\0') {
        return Err(HcliBridgeError::validation(
            field,
            "must not contain a NUL byte",
        ));
    }
    Ok(())
}

fn validate_range<T>(field: &str, value: T, minimum: T, maximum: T) -> HcliBridgeResult<()>
where
    T: PartialOrd + fmt::Display + Copy,
{
    if value < minimum || value > maximum {
        return Err(HcliBridgeError::validation(
            field,
            format!("must be from {minimum} through {maximum}"),
        ));
    }
    Ok(())
}

fn validate_optional_range<T>(
    field: &str,
    value: Option<T>,
    minimum: T,
    maximum: T,
) -> HcliBridgeResult<()>
where
    T: PartialOrd + fmt::Display + Copy,
{
    if let Some(value) = value {
        validate_range(field, value, minimum, maximum)?;
    }
    Ok(())
}

fn ensure_unique<T>(field: &str, values: impl IntoIterator<Item = T>) -> HcliBridgeResult<()>
where
    T: Ord,
{
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(HcliBridgeError::validation(
                field,
                "must not contain duplicates",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(method: &str, params: Value) -> String {
        json!({
            "protocol": HCLI_JSONL_PROTOCOL,
            "id": "req-1",
            "client": { "name": "codex", "version": "test" },
            "method": method,
            "params": params,
        })
        .to_string()
    }

    #[test]
    fn parses_a_durable_generate_request_and_round_trips_jsonl() {
        let line = request(
            "generate",
            json!({
                "prompt": "Summarize the current status.",
                "session_id": "ses_local",
                "runtime_url": "http://127.0.0.1:8787",
                "model": "deepseek-v4-flash-gravity",
                "max_output_tokens": 4096,
                "max_context_tokens": 131072,
                "sampling": { "temperature": 0.2, "top_p": 0.95 },
                "durable": true,
                "metadata": { "caller": "codex" }
            }),
        );

        let parsed = HcliRequest::parse_jsonl_line(&(line + "\n")).unwrap();
        let operation = parsed.operation().unwrap();
        let HcliOperation::Generate(generate) = operation else {
            panic!("expected generate operation");
        };
        assert!(generate.durable);
        assert_eq!(generate.max_context_tokens, Some(131072));
        assert_eq!(generate.sampling.top_p, Some(0.95));

        let encoded = parsed.to_jsonl_line().unwrap();
        assert!(!encoded.contains('\n'));
        assert_eq!(HcliRequest::parse_jsonl_line(&encoded).unwrap(), parsed);
    }

    #[test]
    fn rejects_full_auto_without_an_explicit_acknowledgement() {
        let line = request(
            "agent",
            json!({
                "goal": "Refactor the parser and run the test suite.",
                "autonomy": "full_auto",
                "profile": "maximum"
            }),
        );
        let error = HcliRequest::parse_jsonl_line(&line).unwrap_err();
        assert!(matches!(
            error,
            HcliBridgeError::Validation { ref field, .. }
                if field == "params.acknowledge_effects"
        ));
    }

    #[test]
    fn accepts_high_compute_agent_controls_without_equating_them_to_effects() {
        let line = request(
            "agent",
            json!({
                "goal": "Perform a broad repository audit.",
                "profile": "maximum",
                "budget": {
                    "max_transitions": 3200,
                    "max_tool_calls": 20000,
                    "max_subagents": 128,
                    "search_breadth": 24,
                    "self_consistency_k": 12,
                    "max_context_tokens": 262144,
                    "token_budget_hint": 50000000
                },
                "autonomy": "suggest_only"
            }),
        );
        let request = HcliRequest::parse_jsonl_line(&line).unwrap();
        let HcliOperation::Agent(agent) = request.operation().unwrap() else {
            panic!("expected agent operation");
        };
        assert_eq!(agent.profile, HcliProfile::Maximum);
        assert_eq!(agent.budget.max_subagents, Some(128));
        assert_eq!(agent.autonomy, HcliAutonomy::SuggestOnly);
    }

    #[test]
    fn agent_attachment_refs_are_bounded_and_round_trip_without_source_bytes() {
        let source_hash = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let line = request(
            "agent",
            json!({
                "goal": "Audit the supplied local evidence.",
                "attachments": ["oref_01J1K4M8T9Y6B2V3C4D5E6F7G8", source_hash],
            }),
        );
        let parsed = HcliRequest::parse_jsonl_line(&line).unwrap();
        let HcliOperation::Agent(agent) = parsed.operation().unwrap() else {
            panic!("expected agent operation");
        };
        assert_eq!(
            agent.attachments,
            vec![
                "oref_01J1K4M8T9Y6B2V3C4D5E6F7G8".to_string(),
                source_hash.to_string(),
            ]
        );
        let encoded = parsed.to_jsonl_line().unwrap();
        assert!(encoded.contains("attachments"));
        assert!(!encoded.contains("source bytes"));

        let duplicate = request(
            "agent",
            json!({
                "goal": "Audit.",
                "attachments": ["oref_01J1K4M8T9Y6B2V3C4D5E6F7G8", "oref_01J1K4M8T9Y6B2V3C4D5E6F7G8"],
            }),
        );
        let error = HcliRequest::parse_jsonl_line(&duplicate).unwrap_err();
        assert!(matches!(
            error,
            HcliBridgeError::Validation { ref field, .. } if field == "params.attachments"
        ));

        let overflow = request(
            "agent",
            json!({
                "goal": "Audit.",
                "attachments": (0..=MAX_CONTEXT_SOURCES)
                    .map(|index| format!("oref_source_{index}"))
                    .collect::<Vec<_>>(),
            }),
        );
        let error = HcliRequest::parse_jsonl_line(&overflow).unwrap_err();
        assert!(matches!(
            error,
            HcliBridgeError::Validation { ref field, .. } if field == "params.attachments"
        ));

        let malformed = request(
            "agent",
            json!({
                "goal": "Audit.",
                "attachments": ["https://example.invalid/evidence.txt"],
            }),
        );
        let error = HcliRequest::parse_jsonl_line(&malformed).unwrap_err();
        assert!(matches!(
            error,
            HcliBridgeError::Validation { ref field, .. }
                if field == "params.attachments[0]"
        ));
    }

    #[test]
    fn rejects_a_swarm_that_requests_more_concurrency_than_workers() {
        let line = request(
            "swarm",
            json!({
                "goal": "Investigate independent solution paths.",
                "workers": 8,
                "max_concurrency": 9
            }),
        );
        let error = HcliRequest::parse_jsonl_line(&line).unwrap_err();
        assert!(matches!(
            error,
            HcliBridgeError::Validation { ref field, .. }
                if field == "params.max_concurrency"
        ));
    }

    #[test]
    fn validates_capabilities_session_and_status_requests() {
        for (method, params) in [
            ("capabilities", json!({ "include": ["runtime", "agent"] })),
            (
                "session",
                json!({ "action": "create", "label": "DeepSeek work" }),
            ),
            (
                "status",
                json!({ "session_id": "ses_local", "include": ["runtime", "swarms"], "refresh": true }),
            ),
        ] {
            HcliRequest::parse_jsonl_line(&request(method, params)).unwrap();
        }

        let duplicate = request("capabilities", json!({ "include": ["runtime", "runtime"] }));
        assert!(HcliRequest::parse_jsonl_line(&duplicate).is_err());
    }

    #[test]
    fn capability_identity_uses_gravity_as_the_public_identity_and_condense_as_engine() {
        let identity = hcli_model_optimization_identity();
        let taxonomy: Value =
            serde_json::from_str(include_str!("../../../lab/semantic_taxonomy.json")).unwrap();
        assert_eq!(identity.schema, HCLI_MODEL_OPTIMIZATION_IDENTITY_SCHEMA);
        assert_eq!(identity.taxonomy_schema, "hawking.semantic_taxonomy.v1");
        assert_eq!(identity.canonical_identity, "gravity");
        assert_eq!(identity.canonical_display_name, "Gravity");
        assert_eq!(
            identity.public_cli.command,
            taxonomy["canonical_identity"]["public_cli"]["command"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            identity.public_cli.command_exists,
            taxonomy["canonical_identity"]["public_cli"]["command_exists"]
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            identity.public_cli.status_command,
            taxonomy["canonical_identity"]["public_cli"]["status_command"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            identity.public_cli.plan_command,
            taxonomy["canonical_identity"]["public_cli"]["plan_command"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            identity.public_cli.environment_variable_scope,
            taxonomy["canonical_identity"]["public_cli"]["environment_variable_scope"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            identity.public_cli.compatibility_commands.len(),
            taxonomy["canonical_identity"]["public_cli"]["compatibility_commands"]
                .as_array()
                .unwrap()
                .len()
        );
        assert_eq!(
            identity.public_cli.compatibility_commands[0].command,
            taxonomy["canonical_identity"]["public_cli"]["compatibility_commands"][0]["command"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            identity.public_cli.compatibility_commands[0]
                .selector_environment_variable
                .as_deref(),
            Some("HAWKING_GRAVITY")
        );
        assert_eq!(identity.condense_engine_operation.id, "condense");
        assert!(identity.condense_engine_operation.engine_operation);
        assert_eq!(
            identity.condense_engine_operation.canonical_identity,
            "gravity"
        );
        assert_eq!(
            identity.condense_engine_operation.output_identity,
            "gravity"
        );
        assert!(!identity.claim_limits.hcli_executes_optimization);
        assert!(!identity.claim_limits.v4_runtime_claim);
        assert!(
            identity
                .claim_limits
                .artifact_identity_is_not_capability_evidence
        );
        assert_eq!(
            identity.execution_boundary.schema,
            HCLI_MODEL_OPTIMIZATION_EXECUTION_BOUNDARY_SCHEMA
        );
        assert!(!identity.execution_boundary.available);
        assert_eq!(
            identity.execution_boundary.refusal_code,
            HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE
        );
        assert_eq!(
            identity.execution_boundary.canonical_hawking_plan_command,
            identity.public_cli.plan_command
        );
        assert_eq!(
            identity.legacy_compatibility.field_tag_mapping_id,
            taxonomy["tag_contract"]["field_tag_mapping"]["id"]
                .as_str()
                .unwrap()
        );
        let expected_schema_aliases = taxonomy["compatibility"]["legacy_schema_prefixes"]
            .as_array()
            .unwrap();
        assert_eq!(
            identity.legacy_compatibility.schema_aliases.len(),
            expected_schema_aliases.len()
        );
        for (actual, expected) in identity
            .legacy_compatibility
            .schema_aliases
            .iter()
            .zip(expected_schema_aliases)
        {
            assert_eq!(actual.prefix, expected["prefix"].as_str().unwrap());
            assert_eq!(actual.deprecated, expected["deprecated"].as_bool().unwrap());
            assert_eq!(
                actual.superseded_by,
                expected["superseded_by"].as_str().unwrap()
            );
            assert_eq!(
                actual.field_tag_mapping_id,
                expected["field_tag_mapping_id"].as_str().unwrap()
            );
            assert_eq!(actual.operation, expected["operation"].as_str().unwrap());
        }
        let expected_command_aliases = taxonomy["compatibility"]["legacy_command_aliases"]
            .as_array()
            .unwrap();
        assert_eq!(
            identity.legacy_compatibility.command_aliases.len(),
            expected_command_aliases.len()
        );
        for (actual, expected) in identity
            .legacy_compatibility
            .command_aliases
            .iter()
            .zip(expected_command_aliases)
        {
            assert_eq!(actual.command, expected["command"].as_str().unwrap());
            assert_eq!(actual.deprecated, expected["deprecated"].as_bool().unwrap());
            assert_eq!(
                actual.superseded_by,
                expected["superseded_by"].as_str().unwrap()
            );
            assert_eq!(
                actual.maps_to_engine_operation,
                expected["maps_to_engine_operation"].as_str().unwrap()
            );
            assert_eq!(
                actual.not_an_operation,
                expected.get("not_an_operation").and_then(Value::as_bool)
            );
        }

        // Older capability records omitted this new additive field.  The
        // taxonomy-derived serde default makes them decode as the current
        // semantic mapping and pass the same exact validation contract.
        let mut historic_identity_wire = serde_json::to_value(&identity).unwrap();
        historic_identity_wire
            .as_object_mut()
            .unwrap()
            .remove("legacy_compatibility");
        let historic_identity: HcliModelOptimizationIdentity =
            serde_json::from_value(historic_identity_wire).unwrap();
        assert_eq!(historic_identity, identity);
        historic_identity.validate().unwrap();

        let response = CapabilitiesResponse {
            operations: Vec::new(),
            model_optimization_identity: identity,
            limits: HcliLimits::default(),
            notes: Vec::new(),
        };
        response.validate().unwrap();
        let wire = serde_json::to_value(response).unwrap();
        assert_eq!(
            wire["model_optimization_identity"]["canonical_identity"],
            "gravity"
        );
        assert_eq!(
            wire["model_optimization_identity"]["condense_engine_operation"]["id"],
            "condense"
        );
        assert_eq!(
            wire["model_optimization_identity"]["execution_boundary"]["refusal_code"],
            HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE
        );
        assert_eq!(
            wire["model_optimization_identity"]["legacy_compatibility"]["field_tag_mapping_id"],
            taxonomy["tag_contract"]["field_tag_mapping"]["id"]
        );
    }

    #[test]
    fn model_optimization_is_a_typed_non_executing_protocol_refusal() {
        let parsed = HcliRequest::parse_jsonl_line(&request("model_optimization", json!({})))
            .expect("the explicit boundary method must be parseable");
        assert_eq!(parsed.method, HcliMethod::ModelOptimization);
        assert!(matches!(
            parsed.operation().unwrap(),
            HcliOperation::ModelOptimization(ModelOptimizationRequest {})
        ));

        let error = HcliError::model_optimization_not_available();
        assert_eq!(error.code, HcliErrorCode::ModelOptimizationNotAvailable);
        assert!(!error.retryable);
        assert_eq!(
            serde_json::to_value(&error).unwrap()["code"],
            HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE
        );

        let response = HcliResponse::failure("req-1", error).unwrap();
        let encoded = response.to_jsonl_line().unwrap();
        assert_eq!(HcliResponse::parse_jsonl_line(&encoded).unwrap(), response);
    }

    #[test]
    fn response_requires_truthful_tps_denominator_and_durable_event_reference() {
        let invalid = HcliResponse::success(
            "req-1",
            HcliResult::Generate(GenerateResponse {
                status: OperationStatus::Completed,
                session_id: Some("ses_local".to_string()),
                model: Some("deepseek-v4-flash-gravity".to_string()),
                output: Some("done".to_string()),
                stream_id: None,
                durable_turn: Some(DurableTurnReceipt {
                    user_event_id: "evt_user".to_string(),
                    assistant_event_id: Some("evt_assistant".to_string()),
                }),
                telemetry: Some(DecodeTelemetry {
                    input_tokens: Some(100),
                    output_tokens: Some(64),
                    decode_ms: None,
                    completed_decode_forwards: None,
                    decode_forwards_per_second: Some(80.0),
                }),
                warnings: vec![],
            }),
        );
        assert!(
            invalid.is_err(),
            "TPS without measured denominator is rejected"
        );

        let response = HcliResponse::success(
            "req-1",
            HcliResult::Generate(GenerateResponse {
                status: OperationStatus::Completed,
                session_id: Some("ses_local".to_string()),
                model: Some("deepseek-v4-flash-gravity".to_string()),
                output: Some("done".to_string()),
                stream_id: None,
                durable_turn: Some(DurableTurnReceipt {
                    user_event_id: "evt_user".to_string(),
                    assistant_event_id: Some("evt_assistant".to_string()),
                }),
                telemetry: Some(DecodeTelemetry {
                    input_tokens: Some(100),
                    output_tokens: Some(64),
                    decode_ms: Some(800.0),
                    completed_decode_forwards: Some(64),
                    decode_forwards_per_second: Some(80.0),
                }),
                warnings: vec!["runtime supplied complete-forward telemetry".to_string()],
            }),
        )
        .unwrap();
        let encoded = response.to_jsonl_line().unwrap();
        assert_eq!(HcliResponse::parse_jsonl_line(&encoded).unwrap(), response);
    }

    #[test]
    fn rejects_multiline_and_unknown_outer_fields() {
        let multiline = format!(
            "{}\n{}",
            request("status", json!({})),
            request("status", json!({}))
        );
        assert!(HcliRequest::parse_jsonl_line(&multiline).is_err());

        let unknown = json!({
            "protocol": HCLI_JSONL_PROTOCOL,
            "id": "req-1",
            "method": "status",
            "params": {},
            "surprise": true,
        })
        .to_string();
        assert!(HcliRequest::parse_jsonl_line(&unknown).is_err());
    }
}
