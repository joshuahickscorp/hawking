//! YOU personal tool registry: typed effects, permissions, rollback, receipts.
//!
//! High-risk external actions are never silently executed — they need the
//! effect system (proposal → permission gate → single-use receipt → execute).
//!
//! Two tools are implemented against fixtures: **calculator** and **task_lists**.
//! Every other family is fully declared with typed effects and is
//! **non-constructible**.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use hide_core::error::{HideError, Result};

// ---------------------------------------------------------------------------
// Effect / permission model
// ---------------------------------------------------------------------------

/// Effect class a personal tool may perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffectClass {
    /// Pure computation; no side effects.
    PureCompute,
    /// Read local state only.
    LocalRead,
    /// Mutate local user-owned state (tasks, reminders).
    LocalWrite,
    /// Produce a document / file derivative locally.
    LocalGenerate,
    /// Touch the filesystem under policy.
    Filesystem,
    /// Network / open-world.
    Network,
    /// Send or draft external communications.
    ExternalComm,
    /// Execute code or shell under policy.
    CodeExec,
    /// MCP-bridged third-party tool.
    McpBridge,
}

impl ToolEffectClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PureCompute => "pure_compute",
            Self::LocalRead => "local_read",
            Self::LocalWrite => "local_write",
            Self::LocalGenerate => "local_generate",
            Self::Filesystem => "filesystem",
            Self::Network => "network",
            Self::ExternalComm => "external_comm",
            Self::CodeExec => "code_exec",
            Self::McpBridge => "mcp_bridge",
        }
    }

    /// High-risk effects require the receipt path; pure/local-read may run
    /// without a receipt when the tool marks `requires_receipt = false`.
    pub fn is_high_risk(self) -> bool {
        matches!(
            self,
            Self::Network
                | Self::ExternalComm
                | Self::CodeExec
                | Self::McpBridge
                | Self::Filesystem
                | Self::LocalWrite
                | Self::LocalGenerate
        )
    }
}

/// Permissions a tool declares it needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPermissions {
    pub effects: Vec<ToolEffectClass>,
    /// When true, execute requires a single-use receipt from the permission gate.
    pub requires_receipt: bool,
    /// Whether rollback is supported after execute.
    pub supports_rollback: bool,
    pub read_only: bool,
    pub open_world: bool,
}

/// Implementation status for the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Implemented,
    Declared,
}

/// Static ABI for one personal tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonalToolAbi {
    pub id: String,
    pub title: String,
    pub description: String,
    pub permissions: ToolPermissions,
    pub status: ToolStatus,
    /// Input schema (JSON Schema-ish, fixture-friendly).
    pub input_schema: Value,
    /// Output schema.
    pub output_schema: Value,
}

// ---------------------------------------------------------------------------
// Effects / receipts
// ---------------------------------------------------------------------------

/// A prepared, un-executed tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolProposal {
    pub tool_id: String,
    pub effects: Vec<ToolEffectClass>,
    pub summary: String,
    pub args: Value,
}

/// Permission decision recorded before any high-risk execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    Deny,
}

/// Single-use receipt proving a proposal was authorized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolReceipt {
    pub id: String,
    pub proposal: ToolProposal,
    pub decision: PermissionDecision,
    pub issued_at_ms: u64,
    pub consumed: bool,
}

impl ToolReceipt {
    pub fn is_allow(&self) -> bool {
        self.decision == PermissionDecision::Allow && !self.consumed
    }
}

/// Result of a successful execute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecuteResult {
    pub receipt_id: Option<String>,
    pub tool_id: String,
    pub output: Value,
    /// Opaque rollback token when the tool supports rollback.
    pub rollback_token: Option<String>,
}

/// Permission gate for high-risk personal tools. Deny-by-default.
#[derive(Debug, Default)]
pub struct ToolPermissionGate {
    allow_tool_ids: Vec<String>,
    next_receipt: AtomicU64,
    clock_ms: AtomicU64,
}

impl ToolPermissionGate {
    pub fn deny_by_default() -> Self {
        Self::default()
    }

    pub fn allow_tool(mut self, tool_id: impl Into<String>) -> Self {
        self.allow_tool_ids.push(tool_id.into());
        self
    }

    pub fn decide(&self, proposal: &ToolProposal) -> PermissionDecision {
        if self.allow_tool_ids.iter().any(|id| id == &proposal.tool_id) {
            PermissionDecision::Allow
        } else {
            PermissionDecision::Deny
        }
    }

    pub fn issue_receipt(&self, proposal: ToolProposal) -> ToolReceipt {
        let decision = self.decide(&proposal);
        let id = format!("rcpt-{}", self.next_receipt.fetch_add(1, Ordering::Relaxed));
        ToolReceipt {
            id,
            proposal,
            decision,
            issued_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
            consumed: false,
        }
    }
}

/// Execute without a receipt — always fails for tools that require receipts.
/// High-risk external actions are never silently executed.
pub fn execute_without_receipt(
    tool: &dyn PersonalTool,
    proposal: &ToolProposal,
) -> Result<ToolExecuteResult> {
    if tool.abi().permissions.requires_receipt {
        return Err(HideError::PolicyDenied(format!(
            "tool {} requires a receipt; silent execute refused",
            proposal.tool_id
        )));
    }
    tool.execute(proposal, None)
}

/// Execute with a single-use allow receipt.
pub fn execute_with_receipt(
    tool: &dyn PersonalTool,
    proposal: &ToolProposal,
    receipt: &mut ToolReceipt,
) -> Result<ToolExecuteResult> {
    if receipt.proposal.tool_id != proposal.tool_id {
        return Err(HideError::InvalidState(
            "receipt tool_id does not match proposal".into(),
        ));
    }
    if receipt.consumed {
        return Err(HideError::InvalidState("receipt already consumed".into()));
    }
    if receipt.decision != PermissionDecision::Allow {
        return Err(HideError::PolicyDenied("receipt decision is deny".into()));
    }
    if tool.abi().permissions.requires_receipt {
        receipt.consumed = true;
        let mut result = tool.execute(proposal, Some(receipt))?;
        result.receipt_id = Some(receipt.id.clone());
        Ok(result)
    } else {
        // Tool does not require receipt; still allow path for uniformity.
        receipt.consumed = true;
        let mut result = tool.execute(proposal, Some(receipt))?;
        result.receipt_id = Some(receipt.id.clone());
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tool trait + registry
// ---------------------------------------------------------------------------

/// Live personal tool surface.
pub trait PersonalTool: Send + Sync {
    fn abi(&self) -> &PersonalToolAbi;
    fn prepare(&self, args: Value) -> Result<ToolProposal>;
    fn execute(
        &self,
        proposal: &ToolProposal,
        receipt: Option<&ToolReceipt>,
    ) -> Result<ToolExecuteResult>;
    fn rollback(&self, token: &str) -> Result<()>;
}

/// Registry of every personal tool ABI; only implemented tools construct.
pub struct PersonalToolRegistry {
    by_id: BTreeMap<String, PersonalToolAbi>,
}

impl PersonalToolRegistry {
    pub fn builtin() -> Self {
        let mut by_id = BTreeMap::new();
        for abi in all_tool_abis() {
            by_id.insert(abi.id.clone(), abi);
        }
        Self { by_id }
    }

    pub fn get(&self, id: &str) -> Option<&PersonalToolAbi> {
        self.by_id.get(id)
    }

    pub fn tools(&self) -> impl Iterator<Item = &PersonalToolAbi> {
        self.by_id.values()
    }

    pub fn implemented(&self) -> Vec<&PersonalToolAbi> {
        self.tools()
            .filter(|a| a.status == ToolStatus::Implemented)
            .collect()
    }

    pub fn declared(&self) -> Vec<&PersonalToolAbi> {
        self.tools()
            .filter(|a| a.status == ToolStatus::Declared)
            .collect()
    }

    /// Construct a live tool. Declared tools refuse construction.
    pub fn construct(&self, id: &str) -> Result<LivePersonalTool> {
        let abi = self
            .by_id
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("personal tool {id}")))?;
        match abi.status {
            ToolStatus::Declared => Err(HideError::RuntimeUnavailable(format!(
                "personal tool {id} is declared but non-constructible"
            ))),
            ToolStatus::Implemented => match id {
                "calculator" => Ok(LivePersonalTool::Calculator(CalculatorTool::new(
                    abi.clone(),
                ))),
                "task_lists" => Ok(LivePersonalTool::TaskLists(TaskListsTool::new(abi.clone()))),
                other => Err(HideError::RuntimeUnavailable(format!(
                    "personal tool {other} marked implemented but has no constructor"
                ))),
            },
        }
    }

    pub fn export_document(&self) -> Value {
        json!({
            "schema": "hide.you.personal_tools.v1",
            "tools": self.tools().cloned().collect::<Vec<_>>(),
            "implemented": self.implemented().iter().map(|a| a.id.clone()).collect::<Vec<_>>(),
            "declared": self.declared().iter().map(|a| a.id.clone()).collect::<Vec<_>>(),
        })
    }
}

/// Constructible tools only.
#[derive(Debug)]
pub enum LivePersonalTool {
    Calculator(CalculatorTool),
    TaskLists(TaskListsTool),
}

impl LivePersonalTool {
    pub fn as_tool(&self) -> &dyn PersonalTool {
        match self {
            Self::Calculator(t) => t,
            Self::TaskLists(t) => t,
        }
    }
}

// ---------------------------------------------------------------------------
// ABI catalogue
// ---------------------------------------------------------------------------

fn pure_compute_perms() -> ToolPermissions {
    ToolPermissions {
        effects: vec![ToolEffectClass::PureCompute],
        requires_receipt: false,
        supports_rollback: false,
        read_only: true,
        open_world: false,
    }
}

fn local_write_perms() -> ToolPermissions {
    ToolPermissions {
        effects: vec![ToolEffectClass::LocalWrite],
        requires_receipt: true,
        supports_rollback: true,
        read_only: false,
        open_world: false,
    }
}

fn declared(
    id: &str,
    title: &str,
    description: &str,
    permissions: ToolPermissions,
    input: Value,
    output: Value,
) -> PersonalToolAbi {
    PersonalToolAbi {
        id: id.into(),
        title: title.into(),
        description: description.into(),
        permissions,
        status: ToolStatus::Declared,
        input_schema: input,
        output_schema: output,
    }
}

fn all_tool_abis() -> Vec<PersonalToolAbi> {
    vec![
        // --- implemented ---
        PersonalToolAbi {
            id: "calculator".into(),
            title: "Calculator".into(),
            description: "Evaluate arithmetic expressions (fixture; pure compute).".into(),
            permissions: pure_compute_perms(),
            status: ToolStatus::Implemented,
            input_schema: json!({"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}),
            output_schema: json!({"type":"object","properties":{"result":{"type":"number"}}}),
        },
        PersonalToolAbi {
            id: "task_lists".into(),
            title: "Task lists".into(),
            description: "Create, list, complete, and remove personal tasks (fixture store)."
                .into(),
            permissions: local_write_perms(),
            status: ToolStatus::Implemented,
            input_schema: json!({
                "type":"object",
                "properties":{
                    "op":{"enum":["add","list","complete","remove"]},
                    "title":{"type":"string"},
                    "task_id":{"type":"string"}
                },
                "required":["op"]
            }),
            output_schema: json!({"type":"object"}),
        },
        // --- declared only ---
        declared(
            "file_conversion",
            "File conversion",
            "Convert between document formats under local policy.",
            ToolPermissions {
                effects: vec![ToolEffectClass::Filesystem, ToolEffectClass::LocalGenerate],
                requires_receipt: true,
                supports_rollback: false,
                read_only: false,
                open_world: false,
            },
            json!({"type":"object","properties":{"src":{"type":"string"},"to":{"type":"string"}}}),
            json!({"type":"object","properties":{"path":{"type":"string"}}}),
        ),
        declared(
            "document_generation",
            "Document generation",
            "Generate a document from a template and inputs.",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalGenerate],
                requires_receipt: true,
                supports_rollback: true,
                read_only: false,
                open_world: false,
            },
            json!({"type":"object","properties":{"template":{"type":"string"},"fields":{"type":"object"}}}),
            json!({"type":"object","properties":{"document_id":{"type":"string"}}}),
        ),
        declared(
            "spreadsheet_analysis",
            "Spreadsheet analysis",
            "Analyse spreadsheet tables; local read of attached objects.",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalRead],
                requires_receipt: false,
                supports_rollback: false,
                read_only: true,
                open_world: false,
            },
            json!({"type":"object","properties":{"object_hash":{"type":"string"},"query":{"type":"string"}}}),
            json!({"type":"object","properties":{"summary":{"type":"string"}}}),
        ),
        declared(
            "calendar_planning",
            "Calendar planning",
            "Plan events against calendar connectors (write is external_comm).",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalRead, ToolEffectClass::ExternalComm],
                requires_receipt: true,
                supports_rollback: true,
                read_only: false,
                open_world: true,
            },
            json!({"type":"object","properties":{"op":{"type":"string"},"event":{"type":"object"}}}),
            json!({"type":"object"}),
        ),
        declared(
            "email_drafting",
            "Email drafting",
            "Draft or send email via connector; send is high-risk.",
            ToolPermissions {
                effects: vec![ToolEffectClass::ExternalComm],
                requires_receipt: true,
                supports_rollback: false,
                read_only: false,
                open_world: true,
            },
            json!({"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}}}),
            json!({"type":"object","properties":{"draft_id":{"type":"string"}}}),
        ),
        declared(
            "contact_lookup",
            "Contact lookup",
            "Look up contacts via connector or local address book.",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalRead, ToolEffectClass::Network],
                requires_receipt: true,
                supports_rollback: false,
                read_only: true,
                open_world: true,
            },
            json!({"type":"object","properties":{"query":{"type":"string"}}}),
            json!({"type":"object","properties":{"contacts":{"type":"array"}}}),
        ),
        declared(
            "reminders",
            "Reminders",
            "Schedule local reminders.",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalWrite],
                requires_receipt: true,
                supports_rollback: true,
                read_only: false,
                open_world: false,
            },
            json!({"type":"object","properties":{"text":{"type":"string"},"at_ms":{"type":"integer"}}}),
            json!({"type":"object","properties":{"reminder_id":{"type":"string"}}}),
        ),
        declared(
            "web_search",
            "Web search",
            "Open-world web search. Network handle required at session boundary.",
            ToolPermissions {
                effects: vec![ToolEffectClass::Network],
                requires_receipt: true,
                supports_rollback: false,
                read_only: true,
                open_world: true,
            },
            json!({"type":"object","properties":{"query":{"type":"string"}}}),
            json!({"type":"object","properties":{"hits":{"type":"array"}}}),
        ),
        declared(
            "image_analysis",
            "Image analysis",
            "Analyse an attached image (model-deferred; declared only).",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalRead],
                requires_receipt: false,
                supports_rollback: false,
                read_only: true,
                open_world: false,
            },
            json!({"type":"object","properties":{"object_hash":{"type":"string"}}}),
            json!({"type":"object","properties":{"description":{"type":"string"}}}),
        ),
        declared(
            "transcription",
            "Transcription",
            "Transcribe audio/video attachments (model-deferred).",
            ToolPermissions {
                effects: vec![ToolEffectClass::LocalRead, ToolEffectClass::LocalGenerate],
                requires_receipt: true,
                supports_rollback: false,
                read_only: false,
                open_world: false,
            },
            json!({"type":"object","properties":{"object_hash":{"type":"string"}}}),
            json!({"type":"object","properties":{"text":{"type":"string"}}}),
        ),
        declared(
            "code_execution",
            "Code execution",
            "Sandboxed code execution under policy.",
            ToolPermissions {
                effects: vec![ToolEffectClass::CodeExec],
                requires_receipt: true,
                supports_rollback: false,
                read_only: false,
                open_world: false,
            },
            json!({"type":"object","properties":{"language":{"type":"string"},"source":{"type":"string"}}}),
            json!({"type":"object","properties":{"stdout":{"type":"string"},"exit_code":{"type":"integer"}}}),
        ),
        declared(
            "local_shell",
            "Local shell under policy",
            "Shell.run under sandbox policy; never silent.",
            ToolPermissions {
                effects: vec![ToolEffectClass::CodeExec, ToolEffectClass::Filesystem],
                requires_receipt: true,
                supports_rollback: false,
                read_only: false,
                open_world: false,
            },
            json!({"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"}}}),
            json!({"type":"object","properties":{"stdout":{"type":"string"},"exit_code":{"type":"integer"}}}),
        ),
        declared(
            "mcp_tools",
            "MCP tools",
            "Bridge to MCP server tools; each call is an effect with receipt.",
            ToolPermissions {
                effects: vec![ToolEffectClass::McpBridge, ToolEffectClass::Network],
                requires_receipt: true,
                supports_rollback: false,
                read_only: false,
                open_world: true,
            },
            json!({"type":"object","properties":{"server":{"type":"string"},"tool":{"type":"string"},"args":{"type":"object"}}}),
            json!({"type":"object"}),
        ),
    ]
}

// ---------------------------------------------------------------------------
// Calculator (implemented, pure)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CalculatorTool {
    abi: PersonalToolAbi,
}

impl CalculatorTool {
    pub fn new(abi: PersonalToolAbi) -> Self {
        Self { abi }
    }

    /// Tiny fixture evaluator: integers and + - * / with left-to-right, no precedence.
    fn eval_expr(expr: &str) -> Result<f64> {
        let s = expr.replace(' ', "");
        if s.is_empty() {
            return Err(HideError::msg("empty expression"));
        }
        // Tokenize numbers and ops.
        let mut nums: Vec<f64> = Vec::new();
        let mut ops: Vec<char> = Vec::new();
        let mut cur = String::new();
        for ch in s.chars() {
            if ch.is_ascii_digit() || ch == '.' {
                cur.push(ch);
            } else if matches!(ch, '+' | '-' | '*' | '/') {
                if cur.is_empty() {
                    // unary minus on first number
                    if ch == '-' && nums.is_empty() && ops.is_empty() {
                        cur.push('-');
                        continue;
                    }
                    return Err(HideError::msg("malformed expression"));
                }
                nums.push(
                    cur.parse::<f64>()
                        .map_err(|e| HideError::msg(format!("bad number: {e}")))?,
                );
                cur.clear();
                ops.push(ch);
            } else {
                return Err(HideError::msg(format!("unsupported char {ch}")));
            }
        }
        if cur.is_empty() {
            return Err(HideError::msg("expression ends with operator"));
        }
        nums.push(
            cur.parse::<f64>()
                .map_err(|e| HideError::msg(format!("bad number: {e}")))?,
        );
        if ops.len() + 1 != nums.len() {
            return Err(HideError::msg("mismatched operators"));
        }
        let mut acc = nums[0];
        for (i, op) in ops.iter().enumerate() {
            let rhs = nums[i + 1];
            acc = match op {
                '+' => acc + rhs,
                '-' => acc - rhs,
                '*' => acc * rhs,
                '/' => {
                    if rhs == 0.0 {
                        return Err(HideError::msg("division by zero"));
                    }
                    acc / rhs
                }
                _ => unreachable!(),
            };
        }
        Ok(acc)
    }
}

impl PersonalTool for CalculatorTool {
    fn abi(&self) -> &PersonalToolAbi {
        &self.abi
    }

    fn prepare(&self, args: Value) -> Result<ToolProposal> {
        let expr = args
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HideError::msg("calculator requires expression"))?;
        Ok(ToolProposal {
            tool_id: "calculator".into(),
            effects: self.abi.permissions.effects.clone(),
            summary: format!("calculate {expr}"),
            args,
        })
    }

    fn execute(
        &self,
        proposal: &ToolProposal,
        _receipt: Option<&ToolReceipt>,
    ) -> Result<ToolExecuteResult> {
        let expr = proposal
            .args
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HideError::msg("missing expression"))?;
        let result = Self::eval_expr(expr)?;
        Ok(ToolExecuteResult {
            receipt_id: None,
            tool_id: "calculator".into(),
            output: json!({"result": result, "expression": expr}),
            rollback_token: None,
        })
    }

    fn rollback(&self, _token: &str) -> Result<()> {
        Err(HideError::InvalidState("calculator has no rollback".into()))
    }
}

// ---------------------------------------------------------------------------
// Task lists (implemented, local write + rollback)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Task {
    id: String,
    title: String,
    done: bool,
}

#[derive(Debug)]
pub struct TaskListsTool {
    abi: PersonalToolAbi,
    tasks: Mutex<BTreeMap<String, Task>>,
    next: AtomicU64,
    /// Snapshots for rollback: token → prior task map.
    snapshots: Mutex<BTreeMap<String, BTreeMap<String, Task>>>,
}

impl TaskListsTool {
    pub fn new(abi: PersonalToolAbi) -> Self {
        Self {
            abi,
            tasks: Mutex::new(BTreeMap::new()),
            next: AtomicU64::new(1),
            snapshots: Mutex::new(BTreeMap::new()),
        }
    }
}

impl PersonalTool for TaskListsTool {
    fn abi(&self) -> &PersonalToolAbi {
        &self.abi
    }

    fn prepare(&self, args: Value) -> Result<ToolProposal> {
        let op = args
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HideError::msg("task_lists requires op"))?;
        Ok(ToolProposal {
            tool_id: "task_lists".into(),
            effects: self.abi.permissions.effects.clone(),
            summary: format!("task_lists.{op}"),
            args,
        })
    }

    fn execute(
        &self,
        proposal: &ToolProposal,
        _receipt: Option<&ToolReceipt>,
    ) -> Result<ToolExecuteResult> {
        let op = proposal
            .args
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HideError::msg("missing op"))?;

        // Snapshot for rollback on mutating ops.
        let rollback_token = if matches!(op, "add" | "complete" | "remove") {
            let token = format!("rb-{}", self.next.fetch_add(1, Ordering::Relaxed));
            let snap = self.tasks.lock().unwrap().clone();
            self.snapshots.lock().unwrap().insert(token.clone(), snap);
            Some(token)
        } else {
            None
        };

        let output = match op {
            "list" => {
                let tasks: Vec<_> = self.tasks.lock().unwrap().values().cloned().collect();
                json!({"tasks": tasks})
            }
            "add" => {
                let title = proposal
                    .args
                    .get("title")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| HideError::msg("add requires title"))?
                    .to_string();
                let id = format!("task-{}", self.next.fetch_add(1, Ordering::Relaxed));
                let task = Task {
                    id: id.clone(),
                    title,
                    done: false,
                };
                self.tasks.lock().unwrap().insert(id.clone(), task.clone());
                json!({"task": task})
            }
            "complete" => {
                let id = proposal
                    .args
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| HideError::msg("complete requires task_id"))?;
                let mut map = self.tasks.lock().unwrap();
                let task = map
                    .get_mut(id)
                    .ok_or_else(|| HideError::NotFound(format!("task {id}")))?;
                task.done = true;
                json!({"task": task.clone()})
            }
            "remove" => {
                let id = proposal
                    .args
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| HideError::msg("remove requires task_id"))?;
                let removed = self.tasks.lock().unwrap().remove(id);
                json!({"removed": removed})
            }
            other => {
                return Err(HideError::msg(format!("unknown task_lists op {other}")));
            }
        };

        Ok(ToolExecuteResult {
            receipt_id: None,
            tool_id: "task_lists".into(),
            output,
            rollback_token,
        })
    }

    fn rollback(&self, token: &str) -> Result<()> {
        let snap = self
            .snapshots
            .lock()
            .unwrap()
            .remove(token)
            .ok_or_else(|| HideError::NotFound(format!("rollback token {token}")))?;
        *self.tasks.lock().unwrap() = snap;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn property_high_risk_external_actions_never_silent() {
        let reg = PersonalToolRegistry::builtin();
        let live = reg.construct("task_lists").unwrap();
        let tool = live.as_tool();
        let proposal = tool
            .prepare(json!({"op":"add","title":"buy milk"}))
            .unwrap();
        let err = execute_without_receipt(tool, &proposal).unwrap_err();
        assert!(matches!(err, HideError::PolicyDenied(_)));
        let gate = ToolPermissionGate::deny_by_default();
        let mut denied = gate.issue_receipt(proposal.clone());
        assert_eq!(denied.decision, PermissionDecision::Deny);
        assert!(execute_with_receipt(tool, &proposal, &mut denied).is_err());
        let gate = ToolPermissionGate::deny_by_default().allow_tool("task_lists");
        let mut receipt = gate.issue_receipt(proposal.clone());
        let result = execute_with_receipt(tool, &proposal, &mut receipt).unwrap();
        assert!(result.receipt_id.is_some());
        assert_eq!(result.output["task"]["title"], "buy milk");
        assert!(execute_with_receipt(tool, &proposal, &mut receipt).is_err());
    }
    #[test]
    fn property_calculator_fixture_pure_compute() {
        let reg = PersonalToolRegistry::builtin();
        let live = reg.construct("calculator").unwrap();
        let tool = live.as_tool();
        let proposal = tool.prepare(json!({"expression":"2 + 3 * 4"})).unwrap();
        let result = execute_without_receipt(tool, &proposal).unwrap();
        assert_eq!(result.output["result"], 20.0);
    }
    #[test]
    fn property_task_lists_rollback() {
        let reg = PersonalToolRegistry::builtin();
        let live = reg.construct("task_lists").unwrap();
        let tool = live.as_tool();
        let gate = ToolPermissionGate::deny_by_default().allow_tool("task_lists");
        let p1 = tool.prepare(json!({"op":"add","title":"a"})).unwrap();
        let mut r1 = gate.issue_receipt(p1.clone());
        let res = execute_with_receipt(tool, &p1, &mut r1).unwrap();
        let token = res.rollback_token.as_deref().unwrap();
        let list = tool.prepare(json!({"op":"list"})).unwrap();
        let mut rl = gate.issue_receipt(list.clone());
        let listed = execute_with_receipt(tool, &list, &mut rl).unwrap();
        assert_eq!(listed.output["tasks"].as_array().unwrap().len(), 1);
        tool.rollback(token).unwrap();
        let list2 = tool.prepare(json!({"op":"list"})).unwrap();
        let mut rl2 = gate.issue_receipt(list2.clone());
        let listed2 = execute_with_receipt(tool, &list2, &mut rl2).unwrap();
        assert_eq!(listed2.output["tasks"].as_array().unwrap().len(), 0);
    }
    #[test]
    fn property_declared_tools_non_constructible() {
        let reg = PersonalToolRegistry::builtin();
        assert_eq!(reg.implemented().len(), 2);
        assert!(reg.declared().len() >= 13);
        for id in [
            "file_conversion",
            "document_generation",
            "spreadsheet_analysis",
            "calendar_planning",
            "email_drafting",
            "contact_lookup",
            "reminders",
            "web_search",
            "image_analysis",
            "transcription",
            "code_execution",
            "local_shell",
            "mcp_tools",
        ] {
            let abi = reg.get(id).expect(id);
            assert_eq!(abi.status, ToolStatus::Declared);
            assert!(
                !abi.permissions.effects.is_empty(),
                "{id} must declare effects"
            );
            assert!(reg.construct(id).is_err(), "{id} must refuse construct");
        }
        for id in [
            "web_search",
            "email_drafting",
            "local_shell",
            "mcp_tools",
            "code_execution",
        ] {
            assert!(
                reg.get(id).unwrap().permissions.requires_receipt,
                "{id} must require receipt"
            );
        }
    }
}
