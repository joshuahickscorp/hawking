//! Tool defects classified separately from model failures (bible §16).

use super::bundle::ToolRef;
use serde::{Deserialize, Serialize};

/// What the tool gateway / runtime observed for one call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeObservation {
    /// `ToolResult.ok` from hide-core (true even when exit_code != 0 for process tools).
    pub tool_ok: bool,
    pub tool_error: Option<String>,
    pub exit_code: Option<i32>,
    /// Tool the model was supposed to call (if known from plan/oracle).
    pub model_expected_tool: Option<String>,
    /// Whether model-emitted args validated against the tool input schema.
    pub model_emitted_args_valid: bool,
    pub timed_out: bool,
    /// Tool performed an effect outside its declared set / policy.
    pub effect_violated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolDefectKind {
    SchemaViolation,
    Timeout,
    Crash,
    Unhealthy,
    EffectBoundaryBreach,
    VersionMismatch,
    CredentialFailure,
    TransportFailure,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFailureKind {
    WrongTool,
    BadArguments,
    IgnoredToolResult,
    HallucinatedTool,
    PolicyCircumventionAttempt,
    Other,
}

/// Partitioned failure taxonomy — never collapse tool defects into model scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "class")]
pub enum FailureClass {
    /// Tool layer returned ok; any exit_code is data for the model to handle.
    SuccessWithData { exit_code: Option<i32> },
    ToolDefect {
        kind: ToolDefectKind,
        detail: String,
        tool_id: String,
        tool_version: String,
    },
    ModelFailure {
        kind: ModelFailureKind,
        detail: String,
    },
    /// Both sides contributed; keep both for honest scoring.
    Mixed {
        tool: ToolDefectKind,
        model: ModelFailureKind,
        detail: String,
    },
}

/// Classify one tool-call outcome. Pure function; no I/O.
pub fn classify_outcome(tool: &ToolRef, obs: &OutcomeObservation) -> FailureClass {
    // 1) Effect breach is always a tool-side defect (runtime failed to contain).
    if obs.effect_violated {
        return FailureClass::ToolDefect {
            kind: ToolDefectKind::EffectBoundaryBreach,
            detail: obs
                .tool_error
                .clone()
                .unwrap_or_else(|| "effect boundary violated".into()),
            tool_id: tool.id.clone(),
            tool_version: tool.version.raw.clone(),
        };
    }

    // 2) Timeout → tool defect (or infra); not a model quality signal.
    if obs.timed_out {
        return FailureClass::ToolDefect {
            kind: ToolDefectKind::Timeout,
            detail: obs
                .tool_error
                .clone()
                .unwrap_or_else(|| "tool timed out".into()),
            tool_id: tool.id.clone(),
            tool_version: tool.version.raw.clone(),
        };
    }

    // 3) Model chose the wrong tool (oracle / plan mismatch).
    if let Some(expected) = &obs.model_expected_tool {
        if expected != &tool.name && expected != &tool.id {
            // If the tool also hard-failed, mark mixed.
            if !obs.tool_ok {
                return FailureClass::Mixed {
                    tool: ToolDefectKind::Other,
                    model: ModelFailureKind::WrongTool,
                    detail: format!(
                        "model expected {expected}, got {}; tool also failed: {}",
                        tool.name,
                        obs.tool_error.as_deref().unwrap_or("unknown")
                    ),
                };
            }
            return FailureClass::ModelFailure {
                kind: ModelFailureKind::WrongTool,
                detail: format!("model expected {expected}, invoked {}", tool.name),
            };
        }
    }

    // 4) Bad arguments from the model.
    if !obs.model_emitted_args_valid {
        // Schema rejection at the gateway is a tool-reported validation, but the
        // root cause is model args — attribute to model unless transport crashed.
        if obs
            .tool_error
            .as_deref()
            .is_some_and(|e| e.to_lowercase().contains("schema"))
        {
            // Ambiguous boundary: schema path is tool enforcement of model input.
            // Bible: separate tool defect accounting for schema machinery vs model
            // bad-args. We surface ToolDefect::SchemaViolation when the tool layer
            // rejected the call, and callers may *also* debit model BadArguments.
            return FailureClass::ToolDefect {
                kind: ToolDefectKind::SchemaViolation,
                detail: obs
                    .tool_error
                    .clone()
                    .unwrap_or_else(|| "input schema validation failed".into()),
                tool_id: tool.id.clone(),
                tool_version: tool.version.raw.clone(),
            };
        }
        return FailureClass::ModelFailure {
            kind: ModelFailureKind::BadArguments,
            detail: "model emitted args that failed validation".into(),
        };
    }

    // 5) Tool hard-failed (ok:false) after valid args.
    if !obs.tool_ok {
        let detail = obs
            .tool_error
            .clone()
            .unwrap_or_else(|| "tool returned ok:false".into());
        let kind = if detail.to_lowercase().contains("credential") {
            ToolDefectKind::CredentialFailure
        } else if detail.to_lowercase().contains("transport")
            || detail.to_lowercase().contains("connection")
        {
            ToolDefectKind::TransportFailure
        } else if detail.to_lowercase().contains("schema") {
            ToolDefectKind::SchemaViolation
        } else {
            ToolDefectKind::Crash
        };
        return FailureClass::ToolDefect {
            kind,
            detail,
            tool_id: tool.id.clone(),
            tool_version: tool.version.raw.clone(),
        };
    }

    // 6) ok:true — exit_code is data (hide-core EXEC_NONZERO contract).
    FailureClass::SuccessWithData {
        exit_code: obs.exit_code,
    }
}
