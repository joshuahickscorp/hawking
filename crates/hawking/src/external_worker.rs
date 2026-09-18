//! Small, provider-neutral boundary for an external engineering worker.
//!
//! This module owns only staging-worktree provisioning and result observation.
//! It does not launch a model, call a vendor adapter, grant production
//! authority, or promote a change. The contract is a bounded Hawking worker
//! staging surface rather than a second agent/orchestration system.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

pub const STAGE_RECEIPT_SCHEMA: &str = "hawking.external_worker.stage.v1";
pub const RESULT_RECEIPT_SCHEMA: &str = "hawking.external_worker.result.v1";
pub const CONTEXT_SCHEMA: &str = "hawking.external_worker.context.v1";
pub const WORKER_BRIEF_PATH: &str = "docs/worker/HAWKING_WORKER.md";
pub const TASK_PACKET_DIR: &str = "packets";
pub const RECEIPT_DIR: &str = "receipts";
const CONTINUATION_PATH: &str = "workspace/campaign/odyssey/CONTINUATION.json";
const GRAVITY_REGISTRY_PATH: &str = "tools/foundry/GRAVITY_METHOD_REGISTRY.json";
const RESULT_REPORT_PATHS: &[&str] =
    &[".hawking-worker/result.json", ".hawking-worker-result.json"];

const DEFAULT_ALLOWED_ACTIONS: &[&str] = &[
    "read and search files in the staging worktree",
    "inspect git history, status, and diffs",
    "edit files inside the staging worktree",
    "compile and run CPU-only tests explicitly named by the task",
    "prepare a commit or patch for review",
    "write an evidence packet in the staging worktree",
];

const DEFAULT_FORBIDDEN_ACTIONS: &[&str] = &[
    "production deployment or release promotion",
    "push or force-update a canonical branch",
    "daemon restart, launchd changes, or service control",
    "sudo, credential acquisition, or access to production secrets",
    "Kimi/Flash model loading, GPU leases, or protected runtime work",
    "source-teacher admission, capability certification, or Gravity promotion",
    "destructive mutation of the canonical checkout",
    "invocation of Claude, Codex, a vendor bot, or another frontier agent",
];

#[derive(Subcommand, Debug)]
pub enum AgentCmd {
    /// Provision an isolated worktree for an external engineering worker.
    Stage {
        #[command(subcommand)]
        worker: WorkerCmd,
    },
    /// Observe a staged worker task and emit its review packet.
    Result(AgentResultArgs),
    /// Project canonical Hawking state into a task-scoped worker context.
    Context {
        #[command(subcommand)]
        worker: ContextWorkerCmd,
    },
}

#[derive(Subcommand, Debug)]
pub enum WorkerCmd {
    /// Provision a staging worktree for a Hawking worker. This never launches a bot.
    Hawking(StageArgs),
}

#[derive(Subcommand, Debug)]
pub enum ContextWorkerCmd {
    /// Generate a context view for a staged Hawking worker task.
    Hawking(ContextArgs),
}

#[derive(Args, Debug)]
pub struct StageArgs {
    /// Stable task identifier. Use letters, digits, '.', '_' or '-' only.
    pub task: String,
    /// Human-readable objective copied into the task packet. The task id is
    /// used as a placeholder when this option is omitted.
    #[arg(long, short = 'o')]
    pub objective: Option<String>,
    /// Explicit source revision. Defaults to the canonical checkout's HEAD.
    #[arg(long, value_name = "REVISION")]
    pub source_commit: Option<String>,
    /// Parent directory for isolated staging worktrees. Defaults to the
    /// sibling hawking-worker-staging directory.
    #[arg(long, value_name = "PATH")]
    pub staging_root: Option<PathBuf>,
    /// Relative path scope recorded in the task packet. Repeat for multiple
    /// scopes; the default is the entire staging worktree.
    #[arg(long = "scope", value_name = "PATH")]
    pub scope: Vec<PathBuf>,
    /// Explicitly reuse an existing clean, receipt-backed staging task.
    /// Without this flag an existing path is always rejected.
    #[arg(long, default_value_t = false)]
    pub reuse: bool,
    /// Emit one machine-readable receipt object.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct AgentResultArgs {
    /// Stable task identifier previously passed to agent stage.
    pub task: String,
    /// Parent directory used when the task was staged. Defaults to the
    /// sibling hawking-worker-staging directory.
    #[arg(long, value_name = "PATH")]
    pub staging_root: Option<PathBuf>,
    /// Emit one machine-readable result receipt.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ContextArgs {
    /// Stable task identifier previously passed to agent stage.
    pub task: String,
    /// Parent directory used when the task was staged. Defaults to the
    /// sibling hawking-worker-staging directory.
    #[arg(long, value_name = "PATH")]
    pub staging_root: Option<PathBuf>,
    /// Emit the structured context packet on stdout.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageReceipt {
    schema: String,
    worker_kind: String,
    task_id: String,
    #[serde(default)]
    objective: String,
    source_repository: String,
    source_commit: String,
    source_branch: String,
    canonical_worktree_dirty_at_creation: bool,
    worktree_path: String,
    branch: String,
    created_at_epoch_secs: u64,
    allowed_scope: Vec<String>,
    allowed_actions: Vec<String>,
    forbidden_actions: Vec<String>,
    worker_brief_path: String,
    task_packet_path: String,
    receipt_path: String,
    status: String,
    worktree_registered: bool,
    latest_result_path: Option<String>,
    latest_result: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResultReceipt {
    schema: String,
    worker_kind: String,
    task_id: String,
    source_repository: String,
    base_commit: String,
    current_commit: String,
    resulting_commit: Option<String>,
    worktree_path: String,
    branch: String,
    observed_at_epoch_secs: u64,
    worktree_status: String,
    changed_files: Vec<String>,
    diff_summary: String,
    worker_report_path: Option<String>,
    worker_report_advisory: Option<Value>,
    tests_or_checks: Vec<Value>,
    checks: Vec<Value>,
    pass_fail_results: Vec<Value>,
    unsupported_assumptions: Vec<String>,
    remaining_risks: Vec<String>,
    claims_supported: Vec<String>,
    suggested_review_locations: Vec<String>,
    promotion_allowed: bool,
}

pub fn run(command: AgentCmd) -> Result<()> {
    match command {
        AgentCmd::Stage { worker } => match worker {
            WorkerCmd::Hawking(args) => stage_hawking(args),
        },
        AgentCmd::Result(args) => observe_result(args),
        AgentCmd::Context { worker } => match worker {
            ContextWorkerCmd::Hawking(args) => project_context(args),
        },
    }
}

fn stage_hawking(args: StageArgs) -> Result<()> {
    validate_task_id(&args.task)?;
    let canonical_root = canonical_repository_root()?;
    let staging_root = resolve_staging_root(&canonical_root, args.staging_root.as_deref())?;
    let worktree_path = staging_root.join(&args.task);
    let receipt_path = staging_root
        .join(RECEIPT_DIR)
        .join(format!("{}.json", args.task));
    let packet_path = staging_root
        .join(TASK_PACKET_DIR)
        .join(format!("{}.md", args.task));
    let branch = format!("worker/{}", args.task);
    let requested_source = args.source_commit.as_deref().unwrap_or("HEAD");
    let source_commit = git_output(
        &canonical_root,
        [
            "rev-parse".to_owned(),
            "--verify".to_owned(),
            "--end-of-options".to_owned(),
            format!("{requested_source}^{{commit}}"),
        ],
    )?;
    let source_branch = git_output_allow_failure(
        &canonical_root,
        [
            "symbolic-ref".to_owned(),
            "--short".to_owned(),
            "-q".to_owned(),
            "HEAD".to_owned(),
        ],
    )
    .unwrap_or_else(|| "detached".to_owned());
    let canonical_status = git_output(
        &canonical_root,
        [
            "status".to_owned(),
            "--porcelain=v1".to_owned(),
            "--untracked-files=all".to_owned(),
        ],
    )?;
    let scope = normalize_scopes(&args.scope)?;
    let objective = args.objective.unwrap_or_else(|| {
        format!(
            "Complete the bounded Hawking engineering task identified by {}. Replace this placeholder with the concrete objective before handing the packet to a worker.",
            args.task
        )
    });

    let path_exists = entry_exists(&worktree_path)?;
    let branch_exists = git_ref_exists(&canonical_root, &branch)?;
    let receipt_exists = entry_exists(&receipt_path)?;
    if path_exists || branch_exists || receipt_exists {
        if !args.reuse {
            bail!(
                "refusing implicit staging reuse for task {:?}; existing path, branch, or receipt detected. Inspect it with hawking agent result {} or pass --reuse only after verifying the task",
                args.task,
                args.task
            );
        }
        let receipt = load_stage_receipt(&receipt_path)?;
        validate_reusable_task(
            &canonical_root,
            &worktree_path,
            &branch,
            &receipt,
            &source_commit,
        )?;
        emit_stage(&receipt, true, args.json)?;
        return Ok(());
    }

    fs::create_dir_all(staging_root.join(RECEIPT_DIR))
        .with_context(|| format!("create {}", staging_root.join(RECEIPT_DIR).display()))?;
    fs::create_dir_all(staging_root.join(TASK_PACKET_DIR))
        .with_context(|| format!("create {}", staging_root.join(TASK_PACKET_DIR).display()))?;

    let now = now_epoch_secs();
    let receipt = StageReceipt {
        schema: STAGE_RECEIPT_SCHEMA.to_owned(),
        worker_kind: "hawking_worker".to_owned(),
        task_id: args.task.clone(),
        objective: objective.clone(),
        source_repository: canonical_root.display().to_string(),
        source_commit: source_commit.clone(),
        source_branch,
        canonical_worktree_dirty_at_creation: !canonical_status.trim().is_empty(),
        worktree_path: worktree_path.display().to_string(),
        branch: branch.clone(),
        created_at_epoch_secs: now,
        allowed_scope: scope,
        allowed_actions: DEFAULT_ALLOWED_ACTIONS
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        forbidden_actions: DEFAULT_FORBIDDEN_ACTIONS
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        worker_brief_path: canonical_root.join(WORKER_BRIEF_PATH).display().to_string(),
        task_packet_path: packet_path.display().to_string(),
        receipt_path: receipt_path.display().to_string(),
        status: "provisioning".to_owned(),
        worktree_registered: false,
        latest_result_path: None,
        latest_result: None,
    };
    write_new_json(&receipt_path, &receipt)?;
    let packet = render_task_packet(&receipt, &objective, &[]);
    write_new_text(&packet_path, &packet)?;

    let worktree_result = git_status(
        &canonical_root,
        [
            "worktree".to_owned(),
            "add".to_owned(),
            "-b".to_owned(),
            branch,
            worktree_path.display().to_string(),
            source_commit,
        ],
    );
    if let Err(error) = worktree_result {
        let mut failed = receipt;
        failed.status = "provision_failed".to_owned();
        failed.latest_result = Some(json!({
            "error": error.to_string(),
            "promotion_allowed": false
        }));
        replace_json(&receipt_path, &failed)?;
        return Err(error);
    }

    let mut staged = receipt;
    staged.status = "staged".to_owned();
    staged.worktree_registered = true;
    replace_json(&receipt_path, &staged)?;
    emit_stage(&staged, false, args.json)
}

fn project_context(args: ContextArgs) -> Result<()> {
    validate_task_id(&args.task)?;
    let canonical_root = canonical_repository_root()?;
    let staging_root = resolve_staging_root(&canonical_root, args.staging_root.as_deref())?;
    let receipt_path = staging_root
        .join(RECEIPT_DIR)
        .join(format!("{}.json", args.task));
    let stage = load_stage_receipt(&receipt_path)?;
    if stage.task_id != args.task {
        bail!(
            "receipt task id does not match requested task {:?}",
            args.task
        );
    }
    let stage_repo = PathBuf::from(&stage.source_repository).canonicalize()?;
    if stage_repo != canonical_root {
        bail!(
            "task belongs to {}, not the current canonical repository {}",
            stage_repo.display(),
            canonical_root.display()
        );
    }
    let worktree = PathBuf::from(&stage.worktree_path);
    validate_worktree_membership(&canonical_root, &worktree, &stage.branch)?;

    let (context, markdown) = build_context_packet(&canonical_root, &staging_root, &stage)?;
    let context_json_path = staging_root
        .join(TASK_PACKET_DIR)
        .join(format!("{}.context.json", args.task));
    let context_markdown_path = staging_root
        .join(TASK_PACKET_DIR)
        .join(format!("{}.context.md", args.task));
    replace_json(&context_json_path, &context)?;
    replace_text(&context_markdown_path, &markdown)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&context)?);
    } else {
        println!("CONTEXT {}", args.task);
        println!("JSON {}", context_json_path.display());
        println!("MARKDOWN {}", context_markdown_path.display());
        println!("REVISION {}", stage.source_commit);
        println!("CONTEXT IS A DERIVED VIEW; PROMOTION ALLOWED false");
    }
    Ok(())
}

fn build_context_packet(
    canonical_root: &Path,
    staging_root: &Path,
    stage: &StageReceipt,
) -> Result<(Value, String)> {
    let (continuation, continuation_digest) = read_context_json(canonical_root, CONTINUATION_PATH)?;
    let (registry, registry_digest) = read_context_json(canonical_root, GRAVITY_REGISTRY_PATH)?;
    let stage_receipt_path = staging_root
        .join(RECEIPT_DIR)
        .join(format!("{}.json", stage.task_id));
    let stage_digest = sha256_path(&stage_receipt_path)?;
    let objective = if stage.objective.trim().is_empty() {
        format!(
            "Inspect the bounded task packet for {} and return reviewable staging evidence.",
            stage.task_id
        )
    } else {
        stage.objective.clone()
    };
    let terms = context_terms(&stage.task_id, &objective);
    let current_frontier = project_current_frontier(&continuation);
    let organ_knowledge = project_organ_knowledge(&registry, &terms)?;
    let gravity_knowledge = project_gravity_knowledge(&registry, &terms)?;
    let evidence = project_evidence(
        canonical_root,
        &continuation,
        &terms,
        continuation_digest.clone(),
        registry_digest.clone(),
        stage_receipt_path.clone(),
        stage_digest.clone(),
    )?;
    let context = json!({
        "schema": CONTEXT_SCHEMA,
        "generated_at_epoch_secs": now_epoch_secs(),
        "identity": {
            "task_id": stage.task_id,
            "worker": stage.worker_kind,
            "canonical_revision": stage.source_commit,
            "staging_base_revision": stage.source_commit,
            "source_branch": stage.source_branch,
            "worktree": stage.worktree_path,
            "worker_branch": stage.branch,
            "canonical_worktree_dirty_at_creation": stage.canonical_worktree_dirty_at_creation,
        },
        "objective": objective,
        "authority": {
            "boundary": "isolated staging worktree only",
            "allowed_scope": stage.allowed_scope,
            "allowed_actions": stage.allowed_actions,
            "forbidden_actions": stage.forbidden_actions,
            "promotion_allowed": false,
            "model_gpu_daemon_control": "forbidden",
            "canonical_checkout_mutation": "forbidden",
        },
        "current_frontier": current_frontier,
        "organ_knowledge": organ_knowledge,
        "gravity_knowledge": gravity_knowledge,
        "evidence": evidence,
        "stop_conditions": [
            "Stop when credentials, owner authorization, protected runtime access, GPU/Metal, or stronger scientific adjudication is required.",
            "Do not restart a daemon, load Kimi/Flash, acquire a GPU lease, admit a teacher, promote Gravity, or certify capability.",
            "Return the exact blocker, attempted checks, evidence paths, and smallest next action to the supervisor.",
        ],
        "provenance": {
            "continuation": {"path": CONTINUATION_PATH, "sha256": continuation_digest},
            "gravity_registry": {"path": GRAVITY_REGISTRY_PATH, "sha256": registry_digest},
            "stage_receipt": {"path": stage_receipt_path, "sha256": stage_digest},
            "projection": "derived view; not a new source of truth",
        },
    });
    let markdown = render_context_markdown(&context)?;
    Ok((context, markdown))
}

fn read_context_json(root: &Path, relative: &str) -> Result<(Value, String)> {
    let path = root.join(relative);
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("inspect canonical context source {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "canonical context source is not a direct regular file: {}",
            path.display()
        );
    }
    let bytes = fs::read(&path)?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse canonical context source {}", path.display()))?;
    Ok((value, sha256_bytes(&bytes)))
}

fn sha256_path(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect context evidence {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "context evidence is not a direct regular file: {}",
            path.display()
        );
    }
    Ok(sha256_bytes(&fs::read(path)?))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn context_terms(task: &str, objective: &str) -> Vec<String> {
    let mut terms = BTreeSet::new();
    for raw in format!("{task} {objective}")
        .to_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
    {
        if raw.len() >= 3 {
            terms.insert(raw.to_owned());
        }
    }
    terms.into_iter().collect()
}

fn value_field(value: &Value, name: &str) -> Value {
    value.get(name).cloned().unwrap_or(Value::Null)
}

fn compact_fields(value: &Value, fields: &[&str]) -> Value {
    let mut object = serde_json::Map::new();
    for field in fields {
        if let Some(entry) = value.get(*field) {
            object.insert((*field).to_owned(), entry.clone());
        }
    }
    Value::Object(object)
}

fn context_score(value: &Value, terms: &[String]) -> usize {
    let text = serde_json::to_string(value)
        .unwrap_or_default()
        .to_lowercase();
    terms.iter().map(|term| text.matches(term).count()).sum()
}

fn project_current_frontier(continuation: &Value) -> Value {
    let source = continuation
        .get("current_source_correctness_frontier")
        .unwrap_or(&Value::Null);
    let source_summary = compact_fields(
        source,
        &[
            "status",
            "receipt",
            "last_bounded_frontier",
            "captured_source_greedy_token_ids",
            "real_v2_reference_admitted",
            "machine_admin_trust_anchor_present",
            "source_boundary_capture_owner",
            "boundary_executable_closure",
            "boundary_transaction_status",
            "native_ple_frontier",
        ],
    );
    let organ = continuation
        .get("organ_passport_frontier")
        .unwrap_or(&Value::Null);
    let organ_summary = compact_fields(
        organ,
        &[
            "status",
            "receipt",
            "passport_count",
            "static_flash_projection",
            "scar_policy",
            "next",
        ],
    );
    let nr = continuation
        .get("native_nr_admission_frontier")
        .unwrap_or(&Value::Null);
    let nr_summary = compact_fields(
        nr,
        &[
            "status",
            "receipt",
            "actual_flash_nr_admitted",
            "complete_runnable_nr_present",
            "candidate_freeze_earned",
            "complete_ebpw",
            "native_worker_tps",
            "capability_qualified",
            "boundary",
        ],
    );
    let accounting = continuation
        .get("complete_nr_accounting_frontier")
        .unwrap_or(&Value::Null);
    json!({
        "objective": value_field(continuation, "objective"),
        "active_specimen": value_field(continuation, "active_specimen"),
        "active_workunit": value_field(continuation, "active_workunit"),
        "next_action": value_field(continuation, "next_action"),
        "source_correctness": source_summary,
        "organ_passports": organ_summary,
        "native_nr": nr_summary,
        "complete_nr_accounting": compact_fields(accounting, &["status", "receipt", "complete_nr_ebpw", "source_independent", "direct_execution", "current_science", "boundary"]),
    })
}

fn project_organ_knowledge(registry: &Value, terms: &[String]) -> Result<Value> {
    let entries = registry
        .pointer("/organ_passports/entries")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let broad = terms
        .iter()
        .any(|term| matches!(term.as_str(), "passport" | "organ" | "gravity" | "flash"));
    let mut selected = Vec::new();
    for entry in entries {
        let score = context_score(&entry, terms);
        if broad || score > 0 {
            selected.push(json!({
                "id": value_field(&entry, "id"),
                "semantic_role": value_field(&entry, "semantic_role"),
                "recognition": value_field(&entry, "recognition"),
                "trait_projection": value_field(&entry, "trait_projection"),
                "tensor_contract": value_field(&entry, "tensor_contract"),
                "state_contract": value_field(&entry, "state_contract"),
                "source_accounting": value_field(&entry, "source_accounting"),
            }));
        }
    }
    Ok(json!({
        "selection": if broad { "passport/organ task projection" } else { "task-term match" },
        "entries": selected,
    }))
}

fn project_gravity_knowledge(registry: &Value, terms: &[String]) -> Result<Value> {
    let execution = registry.get("execution_frontier").unwrap_or(&Value::Null);
    let methods = registry
        .get("methods")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut scored = methods
        .into_iter()
        .map(|method| (context_score(&method, terms), method))
        .filter(|(score, _)| *score > 0)
        .collect::<Vec<_>>();
    scored.sort_by(|(left_score, left), (right_score, right)| {
        right_score.cmp(left_score).then_with(|| {
            value_field(left, "id")
                .to_string()
                .cmp(&value_field(right, "id").to_string())
        })
    });
    let omitted = scored.len().saturating_sub(12);
    let selected = scored
        .into_iter()
        .take(12)
        .map(|(_, method)| {
            compact_fields(
                &method,
                &[
                    "id",
                    "title",
                    "family",
                    "status",
                    "required_controls",
                    "evidence",
                    "laws",
                    "claim_boundary",
                    "reopen_if",
                ],
            )
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "selection_terms": terms,
        "execution_frontier": compact_fields(execution, &["updated_at", "ordering_rule", "sealed_now", "active_native_lane", "withheld_without_native_bank"]),
        "methods": selected,
        "omitted_method_count": omitted,
    }))
}

fn project_evidence(
    canonical_root: &Path,
    continuation: &Value,
    terms: &[String],
    continuation_digest: String,
    registry_digest: String,
    stage_receipt_path: PathBuf,
    stage_digest: String,
) -> Result<Vec<Value>> {
    let mut paths = vec![
        CONTINUATION_PATH.to_owned(),
        GRAVITY_REGISTRY_PATH.to_owned(),
    ];
    if let Some(refs) = continuation.get("evidence_refs").and_then(Value::as_array) {
        for path in refs.iter().filter_map(Value::as_str) {
            let lower = path.to_lowercase();
            let relevant = terms.iter().any(|term| lower.contains(term))
                || [
                    "flash",
                    "gravity",
                    "passport",
                    "scar",
                    "organ",
                    "source",
                    "ple",
                    "nr",
                    "unification",
                ]
                .iter()
                .any(|term| lower.contains(term));
            if relevant && paths.len() < 26 {
                paths.push(path.to_owned());
            }
        }
    }
    let mut evidence = vec![
        json!({"path": CONTINUATION_PATH, "sha256": continuation_digest}),
        json!({"path": GRAVITY_REGISTRY_PATH, "sha256": registry_digest}),
        json!({"path": stage_receipt_path, "sha256": stage_digest}),
    ];
    for path in paths.into_iter().skip(2).take(23) {
        let full = canonical_root.join(&path);
        if full.is_file() {
            evidence.push(json!({"path": path, "sha256": sha256_path(&full)?}));
        }
    }
    Ok(evidence)
}

fn render_context_markdown(context: &Value) -> Result<String> {
    let section = |name: &str| -> Result<String> {
        Ok(serde_json::to_string_pretty(
            context.get(name).unwrap_or(&Value::Null),
        )?)
    };
    Ok(format!(
        "# Hawking worker context\n\nThis packet is a derived, task-scoped view. It is not a source of truth and does not grant promotion or runtime authority.\n\n## IDENTITY\n\n```json\n{}\n```\n\n## OBJECTIVE\n\n{}\n\n## AUTHORITY\n\n```json\n{}\n```\n\n## CURRENT FRONTIER\n\n```json\n{}\n```\n\n## ORGAN KNOWLEDGE\n\n```json\n{}\n```\n\n## GRAVITY KNOWLEDGE\n\n```json\n{}\n```\n\n## EVIDENCE\n\n```json\n{}\n```\n\n## STOP CONDITIONS\n\n{}\n\n## PROVENANCE\n\n```json\n{}\n```\n",
        section("identity")?,
        context.get("objective").and_then(Value::as_str).unwrap_or(""),
        section("authority")?,
        section("current_frontier")?,
        section("organ_knowledge")?,
        section("gravity_knowledge")?,
        section("evidence")?,
        context
            .get("stop_conditions")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).map(|item| format!("- {item}")).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default(),
        section("provenance")?,
    ))
}

fn observe_result(args: AgentResultArgs) -> Result<()> {
    validate_task_id(&args.task)?;
    let canonical_root = canonical_repository_root()?;
    let staging_root = resolve_staging_root(&canonical_root, args.staging_root.as_deref())?;
    let receipt_path = staging_root
        .join(RECEIPT_DIR)
        .join(format!("{}.json", args.task));
    let stage = load_stage_receipt(&receipt_path)?;
    if stage.task_id != args.task {
        bail!(
            "receipt task id does not match requested task {:?}",
            args.task
        );
    }
    let stage_repo = PathBuf::from(&stage.source_repository).canonicalize()?;
    if stage_repo != canonical_root {
        bail!(
            "task belongs to {}, not the current canonical repository {}",
            stage_repo.display(),
            canonical_root.display()
        );
    }
    let worktree = PathBuf::from(&stage.worktree_path);
    validate_worktree_membership(&canonical_root, &worktree, &stage.branch)?;
    let result = build_result_receipt(&stage, &receipt_path, &worktree)?;
    let mut updated_stage = stage;
    updated_stage.status = "observed".to_owned();
    updated_stage.latest_result_path = Some(receipt_path.display().to_string());
    updated_stage.latest_result = Some(serde_json::to_value(&result)?);
    replace_json(&receipt_path, &updated_stage)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_human_result(&result);
    }
    Ok(())
}

fn build_result_receipt(
    stage: &StageReceipt,
    stage_receipt_path: &Path,
    worktree: &Path,
) -> Result<ResultReceipt> {
    let current_commit = git_output(worktree, ["rev-parse".to_owned(), "HEAD".to_owned()])?;
    let current_branch =
        git_output_allow_failure(worktree, ["branch".to_owned(), "--show-current".to_owned()])
            .unwrap_or_else(|| "detached".to_owned());
    if current_branch != stage.branch {
        bail!(
            "staging worktree branch changed: receipt={}, observed={}",
            stage.branch,
            current_branch
        );
    }
    let status = git_output(
        worktree,
        [
            "status".to_owned(),
            "--porcelain=v1".to_owned(),
            "--untracked-files=all".to_owned(),
        ],
    )?;
    let committed_names = git_output(
        worktree,
        [
            "diff".to_owned(),
            "--name-only".to_owned(),
            format!("{}..{}", stage.source_commit, current_commit),
        ],
    )?;
    let working_names = git_output(worktree, ["diff".to_owned(), "--name-only".to_owned()])?;
    let cached_names = git_output(
        worktree,
        [
            "diff".to_owned(),
            "--cached".to_owned(),
            "--name-only".to_owned(),
        ],
    )?;
    let changed_files =
        changed_files_from(&status, [&committed_names, &working_names, &cached_names]);
    let diff_summary = combined_diff_summary(worktree, &stage.source_commit, &current_commit)?;
    let diff_check = git_status(
        worktree,
        [
            "diff".to_owned(),
            "--check".to_owned(),
            format!("{}..{}", stage.source_commit, current_commit),
        ],
    );
    let working_diff_check = git_status(worktree, ["diff".to_owned(), "--check".to_owned()]);
    let checks = vec![
        json!({
            "name": "staging_worktree_identity",
            "status": "PASS",
            "detail": format!("{} on {}", worktree.display(), stage.branch)
        }),
        check_value("committed_diff_whitespace", &diff_check),
        check_value("working_diff_whitespace", &working_diff_check),
    ];

    let (worker_report_path, worker_report_advisory, report_error) = read_worker_report(worktree)?;
    let mut tests_or_checks = Vec::new();
    let mut unsupported_assumptions = Vec::new();
    if let Some(report) = &worker_report_advisory {
        if let Some(tests) = report.get("tests").and_then(Value::as_array) {
            tests_or_checks.extend(tests.iter().cloned());
        } else if let Some(report_checks) = report.get("checks").and_then(Value::as_array) {
            tests_or_checks.extend(report_checks.iter().cloned());
        } else {
            unsupported_assumptions.push(
                "worker report was present but did not contain a tests or checks array".to_owned(),
            );
        }
    } else {
        unsupported_assumptions
            .push("no worker result report was supplied; test claims remain unknown".to_owned());
    }
    if let Some(error) = report_error {
        unsupported_assumptions.push(error);
    }

    let mut pass_fail_results = checks.clone();
    pass_fail_results.extend(tests_or_checks.iter().cloned());
    let mut claims_supported = vec![
        "The result command independently observed the recorded repository, worktree, branch, and Git state.".to_owned(),
        "No production promotion, deployment, daemon control, model selection, or GPU action is performed by this surface.".to_owned(),
    ];
    if diff_check.is_ok() && working_diff_check.is_ok() {
        claims_supported
            .push("Git whitespace checks passed for committed and working-tree diffs.".to_owned());
    }
    let resulting_commit = (current_commit != stage.source_commit).then(|| current_commit.clone());
    if resulting_commit.is_some() {
        claims_supported.push(format!(
            "The staging branch contains a commit different from the recorded base: {}.",
            current_commit
        ));
    }
    let mut remaining_risks = vec![
        "This command observes Git state; it does not execute or independently validate the worker's tests.".to_owned(),
        "Human or primary-agent review is required before any promotion outside the staging worktree.".to_owned(),
    ];
    if !status.trim().is_empty() {
        remaining_risks.push("The staging worktree has uncommitted changes.".to_owned());
    }
    if worker_report_advisory.is_some() {
        remaining_risks.push(
            "Worker-reported tests, claims, and evidence are advisory until independently reviewed."
                .to_owned(),
        );
    }

    Ok(ResultReceipt {
        schema: RESULT_RECEIPT_SCHEMA.to_owned(),
        worker_kind: stage.worker_kind.clone(),
        task_id: stage.task_id.clone(),
        source_repository: stage.source_repository.clone(),
        base_commit: stage.source_commit.clone(),
        current_commit: current_commit.clone(),
        resulting_commit,
        worktree_path: stage.worktree_path.clone(),
        branch: stage.branch.clone(),
        observed_at_epoch_secs: now_epoch_secs(),
        worktree_status: if status.trim().is_empty() {
            "clean".to_owned()
        } else {
            status
        },
        changed_files,
        diff_summary,
        worker_report_path,
        worker_report_advisory,
        tests_or_checks,
        checks,
        pass_fail_results,
        unsupported_assumptions,
        remaining_risks,
        claims_supported,
        suggested_review_locations: vec![
            format!(
                "git -C {} diff {}..{}",
                worktree.display(),
                stage.source_commit,
                current_commit
            ),
            format!("git -C {} diff", worktree.display()),
            stage_receipt_path.display().to_string(),
        ],
        promotion_allowed: false,
    })
}

fn validate_reusable_task(
    canonical_root: &Path,
    worktree: &Path,
    branch: &str,
    receipt: &StageReceipt,
    requested_source: &str,
) -> Result<()> {
    if receipt.schema != STAGE_RECEIPT_SCHEMA {
        bail!("unsupported staging receipt schema {:?}", receipt.schema);
    }
    if receipt.source_repository != canonical_root.display().to_string() {
        bail!("staging receipt belongs to a different repository");
    }
    if receipt.worktree_path != worktree.display().to_string() || receipt.branch != branch {
        bail!("staging receipt path or branch does not match the requested task");
    }
    if receipt.source_commit != requested_source {
        bail!("requested source revision does not match the recorded task base");
    }
    validate_worktree_membership(canonical_root, worktree, branch)?;
    let status = git_output(
        worktree,
        [
            "status".to_owned(),
            "--porcelain=v1".to_owned(),
            "--untracked-files=all".to_owned(),
        ],
    )?;
    if !status.trim().is_empty() {
        bail!(
            "refusing reuse of dirty staging worktree {}",
            worktree.display()
        );
    }
    Ok(())
}

fn validate_worktree_membership(
    canonical_root: &Path,
    worktree: &Path,
    branch: &str,
) -> Result<()> {
    if !worktree.is_dir() {
        bail!("staging worktree is missing: {}", worktree.display());
    }
    let observed = git_output(
        canonical_root,
        [
            "worktree".to_owned(),
            "list".to_owned(),
            "--porcelain".to_owned(),
        ],
    )?;
    let expected_path = worktree.canonicalize()?.display().to_string();
    let path_registered = observed
        .lines()
        .any(|line| line == format!("worktree {expected_path}"));
    let branch_registered = observed
        .lines()
        .any(|line| line == format!("branch refs/heads/{branch}"));
    if !path_registered || !branch_registered {
        bail!("staging worktree is not registered with the canonical repository");
    }
    Ok(())
}

fn read_worker_report(worktree: &Path) -> Result<(Option<String>, Option<Value>, Option<String>)> {
    for relative in RESULT_REPORT_PATHS {
        let path = worktree.join(relative);
        if !path.exists() {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Ok((
                Some(path.display().to_string()),
                None,
                Some(format!(
                    "worker report is not a direct regular file: {}",
                    path.display()
                )),
            ));
        }
        if metadata.len() > 1024 * 1024 {
            return Ok((
                Some(path.display().to_string()),
                None,
                Some(format!(
                    "worker report exceeds the 1 MiB bound: {}",
                    path.display()
                )),
            ));
        }
        let text = fs::read_to_string(&path)?;
        let value: Value = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(error) => {
                return Ok((
                    Some(path.display().to_string()),
                    None,
                    Some(format!("worker report is not valid JSON: {error}")),
                ));
            }
        };
        if !value.is_object() {
            return Ok((
                Some(path.display().to_string()),
                None,
                Some("worker report must be a JSON object".to_owned()),
            ));
        }
        return Ok((Some(path.display().to_string()), Some(value), None));
    }
    Ok((None, None, None))
}

fn check_value(name: &str, result: &Result<()>) -> Value {
    match result {
        Ok(()) => json!({"name": name, "status": "PASS"}),
        Err(error) => json!({"name": name, "status": "FAIL", "detail": error.to_string()}),
    }
}

fn combined_diff_summary(worktree: &Path, base: &str, current: &str) -> Result<String> {
    let committed = git_output(
        worktree,
        [
            "diff".to_owned(),
            "--stat".to_owned(),
            format!("{base}..{current}"),
        ],
    )?;
    let working = git_output(worktree, ["diff".to_owned(), "--stat".to_owned()])?;
    let cached = git_output(
        worktree,
        [
            "diff".to_owned(),
            "--cached".to_owned(),
            "--stat".to_owned(),
        ],
    )?;
    let mut sections = Vec::new();
    for (label, output) in [
        ("committed", committed),
        ("working", working),
        ("cached", cached),
    ] {
        if !output.trim().is_empty() {
            sections.push(format!("{label}:\n{}", output.trim_end()));
        }
    }
    Ok(if sections.is_empty() {
        "no committed, staged, or unstaged diff".to_owned()
    } else {
        sections.join("\n")
    })
}

fn changed_files_from<'a, I>(status: &str, name_sets: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a String>,
{
    let mut files = BTreeSet::new();
    for line in status.lines() {
        if line.len() >= 3 {
            files.insert(line[3..].to_owned());
        }
    }
    for names in name_sets {
        for line in names.lines().map(str::trim).filter(|line| !line.is_empty()) {
            files.insert(line.to_owned());
        }
    }
    files.into_iter().collect()
}

fn render_task_packet(stage: &StageReceipt, objective: &str, tests: &[String]) -> String {
    let allowed_actions = stage
        .allowed_actions
        .iter()
        .map(|item| format!("- {item}"))
        .collect::<Vec<_>>()
        .join("\n");
    let forbidden_actions = stage
        .forbidden_actions
        .iter()
        .map(|item| format!("- {item}"))
        .collect::<Vec<_>>()
        .join("\n");
    let tests = if tests.is_empty() {
        "- Select the smallest relevant CPU-only compile/test checks.\n- Run git diff --check.\n- Record exact commands and pass/fail results in .hawking-worker/result.json."
            .to_owned()
    } else {
        tests
            .iter()
            .map(|item| format!("- {item}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "# Hawking external-worker task packet\n\n## OBJECTIVE\n\n{objective}\n\n## SOURCE COMMIT\n\n{}\n\n## WORKTREE\n\n{}\n\n## ALLOWED ACTIONS\n\n{allowed_actions}\n\n## FORBIDDEN ACTIONS\n\n{forbidden_actions}\n\n## AUTHORITATIVE INPUTS\n\n- Repository source at the recorded source commit.\n- The repository continuation and canonical Gravity registry.\n- Existing receipts and focused tests.\n- Worker memory is navigation help only, never scientific authority.\n\n## EXPECTED OUTPUT\n\n- A reviewable commit or diff, if code changes are justified.\n- .hawking-worker/result.json containing exact tests/checks, pass/fail results, unsupported assumptions, remaining risks, supported claims, and suggested review locations.\n- If no code is needed, return an evidence packet instead.\n\n## TESTS\n\n{tests}\n\n## CLAIM BOUNDARY\n\nThis task can produce staging evidence only. It cannot certify capability, authorize a teacher, promote Gravity, restart a daemon, deploy production, or establish model/GPU results. The independent hawking agent result {} command remains the Git observation boundary.\n\n## STOP CONDITIONS\n\nStop and report the exact blocker, evidence, attempted approaches, smallest unresolved question, and suggested next action when stronger reasoning, authority, credentials, protected runtime access, or a non-CPU resource is required. Do not improvise around an authority boundary.\n\n## CONTROL PATHS\n\n- Worker brief: {}\n- Stage receipt: {}\n- Task packet: {}\n- Result command: hawking agent result {}\n\n",
        stage.source_commit,
        stage.worktree_path,
        stage.task_id,
        stage.worker_brief_path,
        stage.receipt_path,
        stage.task_packet_path,
        stage.task_id
    )
}

fn print_human_result(result: &ResultReceipt) {
    println!("TASK {}", result.task_id);
    println!("BASE {}", result.base_commit);
    println!("CURRENT {}", result.current_commit);
    println!("BRANCH {}", result.branch);
    println!("WORKTREE {}", result.worktree_path);
    println!(
        "STATUS {}",
        if result.worktree_status == "clean" {
            "clean"
        } else {
            "changes present"
        }
    );
    println!(
        "RESULTING COMMIT {}",
        result.resulting_commit.as_deref().unwrap_or("none")
    );
    println!(
        "CHANGED FILES {}",
        if result.changed_files.is_empty() {
            "none".to_owned()
        } else {
            result.changed_files.join(", ")
        }
    );
    println!("DIFF SUMMARY\n{}", result.diff_summary);
    println!("CHECKS");
    for check in &result.pass_fail_results {
        let name = check.get("name").and_then(Value::as_str).unwrap_or("check");
        let status = check
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN");
        println!("  {status} {name}");
    }
    if let Some(path) = &result.worker_report_path {
        println!("WORKER REPORT {path} (advisory)");
    } else {
        println!("WORKER REPORT none");
    }
    println!("PROMOTION ALLOWED false");
    println!("REVIEW");
    for location in &result.suggested_review_locations {
        println!("  {location}");
    }
}

fn emit_stage(receipt: &StageReceipt, reused: bool, json_output: bool) -> Result<()> {
    if json_output {
        let mut value = serde_json::to_value(receipt)?;
        value["reused"] = Value::Bool(reused);
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    println!("STAGED {}", receipt.task_id);
    println!("WORKER {}", receipt.worker_kind);
    println!("SOURCE COMMIT {}", receipt.source_commit);
    println!("WORKTREE {}", receipt.worktree_path);
    println!("BRANCH {}", receipt.branch);
    println!("TASK PACKET {}", receipt.task_packet_path);
    println!("WORKER BRIEF {}", receipt.worker_brief_path);
    println!("RECEIPT {}", receipt.receipt_path);
    if receipt.canonical_worktree_dirty_at_creation {
        println!("CANONICAL CHECKOUT dirty at creation; preserved and not copied into staging");
    } else {
        println!("CANONICAL CHECKOUT clean at creation");
    }
    if reused {
        println!("REUSED only after explicit clean-worktree validation");
    }
    println!("SAFE NEXT STEPS");
    println!("  Give the worker the task packet and use the printed worktree as its cwd");
    println!("  Leave local-execution approvals at ASK EVERY TIME");
    println!(
        "  Run hawking agent result {} after the worker stops",
        receipt.task_id
    );
    Ok(())
}

fn canonical_repository_root() -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let root = PathBuf::from(git_output(
        &cwd,
        ["rev-parse".to_owned(), "--show-toplevel".to_owned()],
    )?)
    .canonicalize()
    .context("canonicalize repository root")?;
    if !root.join("H-MANIFESTO.md").is_file() || !root.join("Cargo.toml").is_file() {
        bail!("external-worker commands must run from the canonical Hawking repository");
    }
    if !root.join(".git").is_dir() {
        bail!(
            "external-worker commands must run from the canonical checkout, not a linked worktree"
        );
    }
    Ok(root)
}

fn resolve_staging_root(canonical_root: &Path, requested: Option<&Path>) -> Result<PathBuf> {
    let raw = requested.map(PathBuf::from).unwrap_or_else(|| {
        canonical_root
            .parent()
            .unwrap_or(canonical_root)
            .join("hawking-worker-staging")
    });
    let absolute = if raw.is_absolute() {
        raw
    } else {
        std::env::current_dir()?.join(raw)
    };
    let resolved = canonicalize_with_missing_tail(&absolute)?;
    if resolved == canonical_root || resolved.starts_with(canonical_root) {
        bail!(
            "staging root must be outside the canonical checkout: {}",
            resolved.display()
        );
    }
    Ok(resolved)
}

fn canonicalize_with_missing_tail(path: &Path) -> Result<PathBuf> {
    let mut cursor = path.to_path_buf();
    let mut tail = Vec::new();
    while !cursor.exists() {
        let name = cursor
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("cannot resolve staging path {}", path.display()))?;
        tail.push(name.to_os_string());
        cursor = cursor
            .parent()
            .ok_or_else(|| anyhow::anyhow!("cannot resolve staging path {}", path.display()))?
            .to_path_buf();
    }
    let mut resolved = cursor.canonicalize()?;
    for component in tail.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn normalize_scopes(scopes: &[PathBuf]) -> Result<Vec<String>> {
    let scopes = if scopes.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        scopes.to_vec()
    };
    scopes
        .iter()
        .map(|scope| {
            if scope.is_absolute() {
                bail!(
                    "scope must be relative to the staging worktree: {}",
                    scope.display()
                );
            }
            let mut normalized = PathBuf::new();
            for component in scope.components() {
                match component {
                    Component::CurDir => {}
                    Component::Normal(value) => normalized.push(value),
                    Component::ParentDir => {
                        if !normalized.pop() {
                            bail!(
                                "scope cannot escape the staging worktree: {}",
                                scope.display()
                            );
                        }
                    }
                    Component::RootDir | Component::Prefix(_) => {
                        bail!("scope must be relative: {}", scope.display())
                    }
                }
            }
            Ok(if normalized.as_os_str().is_empty() {
                ".".to_owned()
            } else {
                normalized.display().to_string()
            })
        })
        .collect()
}

fn validate_task_id(task: &str) -> Result<()> {
    if task.is_empty() || task.len() > 96 {
        bail!("task id must contain 1..=96 characters");
    }
    if !task
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || task == "."
        || task == ".."
        || task.starts_with('.')
    {
        bail!("task id must use letters, digits, '.', '_' or '-' and must not start with '.'");
    }
    Ok(())
}

fn load_stage_receipt(path: &Path) -> Result<StageReceipt> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect staging receipt {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "staging receipt must be a direct regular file: {}",
            path.display()
        );
    }
    let text = fs::read_to_string(path)?;
    let receipt: StageReceipt = serde_json::from_str(&text)
        .with_context(|| format!("parse staging receipt {}", path.display()))?;
    Ok(receipt)
}

fn entry_exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn git_ref_exists(root: &Path, branch: &str) -> Result<bool> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["show-ref", "--verify", "--quiet"])
        .arg(format!("refs/heads/{branch}"))
        .output()?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(command_error("git show-ref", output)),
    }
}

fn git_output<I>(root: &Path, args: I) -> Result<String>
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    let output = Command::new("git").current_dir(root).args(args).output()?;
    if !output.status.success() {
        return Err(command_error("git", output));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn git_output_allow_failure<I>(root: &Path, args: I) -> Option<String>
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn git_status<I>(root: &Path, args: I) -> Result<()>
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    let output = Command::new("git").current_dir(root).args(args).output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error("git", output))
    }
}

fn command_error(command: &str, output: Output) -> anyhow::Error {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    anyhow::anyhow!(
        "{command} failed with status {}{}",
        output.status,
        if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        }
    )
}

fn write_new_text(path: &Path, text: &str) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    write_new_text(path, &format!("{text}\n"))
}

fn replace_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let text = serde_json::to_string_pretty(value)?;
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

fn replace_text(path: &Path, text: &str) -> Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{normalize_scopes, validate_task_id};
    use std::path::PathBuf;

    #[test]
    fn task_ids_are_path_safe() {
        assert!(validate_task_id("passport-check_01").is_ok());
        assert!(validate_task_id("../canonical").is_err());
        assert!(validate_task_id(".hidden").is_err());
        assert!(validate_task_id("").is_err());
    }

    #[test]
    fn scopes_cannot_escape_the_worktree() {
        assert_eq!(normalize_scopes(&[]).unwrap(), vec!["."]);
        assert_eq!(
            normalize_scopes(&[PathBuf::from("src/../docs")]).unwrap(),
            vec!["docs"]
        );
        assert!(normalize_scopes(&[PathBuf::from("../../canonical")]).is_err());
        assert!(normalize_scopes(&[PathBuf::from("/tmp")]).is_err());
    }
}
