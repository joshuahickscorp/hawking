use anyhow::{anyhow, Context, Result};
use clap::Subcommand;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const PREFLIGHT_SUMMARY: &str = "reports/condense/studio_preflight_summary.json";
const STUDIO_ENVIRONMENT: &str = "reports/condense/studio_environment.json";
const PROCUREMENT_GATE: &str = "reports/condense/frontier_launch_gate.preflight.json";
const CLAIM_GATE: &str = "reports/condense/frontier_claim_launch_gate.local.json";
const REVIEW_PLAN: &str = "reports/condense/frontier_review_plan.local.json";
const LEDGER: &str = "reports/condense/frontier_ledger.preflight.json";
const PROOF_PACK: &str = "reports/condense/frontier_proof_pack.local.json";
const AUDIT_GRADE: &str = "reports/condense/studio_audit_grade.local.json";
const RUNTIME_CONTRACT: &str = "reports/condense/studio_runtime_contract.local.json";
const COMPLETION_AUDIT: &str = "reports/condense/studio_completion_audit.local.json";
const DENSITY_RECEIPT: &str = "reports/condense/studio_density_receipt.local.json";

#[derive(Subcommand, Debug)]
pub enum StudioCmd {
    /// Run Studio preflight and write the signed launch summary.
    Preflight {
        /// Repository root containing tools/condense/preflight.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Suppress human output from preflight.py.
        #[arg(long, default_value_t = false)]
        quiet: bool,
    },
    /// Verify a signed Studio preflight summary.
    VerifySummary {
        /// Repository root containing tools/condense/preflight.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Summary path to verify.
        #[arg(long, default_value = "reports/condense/studio_preflight_summary.json")]
        path: PathBuf,
    },
    /// Capture signed hardware, network, power, and thermal Studio environment evidence.
    EnvironmentCapture {
        /// Repository root containing tools/condense/studio_environment.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the signed environment receipt here.
        #[arg(long, default_value = "reports/condense/studio_environment.json")]
        out: PathBuf,
        /// Machine class expected for the Studio run.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Expected download link budget in MB/s.
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        /// Minimum RAM required for a green Studio environment.
        #[arg(long, default_value_t = 120.0)]
        min_ram_gb: f64,
        /// Minimum free disk required for a green Studio environment.
        #[arg(long, default_value_t = 500.0)]
        min_free_disk_gb: f64,
        /// Skip the no-download Hugging Face network probe.
        #[arg(long, default_value_t = false)]
        skip_network: bool,
        /// Emit machine-readable JSON from studio_environment.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify a signed Studio environment receipt.
    EnvironmentVerify {
        /// Repository root containing tools/condense/studio_environment.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Environment receipt path to verify.
        #[arg(long, default_value = "reports/condense/studio_environment.json")]
        path: PathBuf,
        /// Emit machine-readable JSON from studio_environment.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Read local Studio proof artifacts and print a launch/claim snapshot.
    Snapshot {
        /// Repository root containing reports/condense.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Emit machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Group the dirty worktree by subsystem for review/stack splitting.
    WorktreePlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the worktree split plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Verify this signed worktree split plan instead of writing a new one.
        #[arg(long)]
        verify: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Build a signed local density, LOC, largest-file, and artifact-mass receipt.
    DensityReceiptBuild {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the signed density receipt here.
        #[arg(
            long,
            default_value = "reports/condense/studio_density_receipt.local.json"
        )]
        out: PathBuf,
        /// Optional previous density receipt for delta reporting.
        #[arg(long)]
        previous: Option<PathBuf>,
        /// Number of largest files to retain in the receipt.
        #[arg(long, default_value_t = 20)]
        max_files: u32,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify a signed local density receipt.
    DensityReceiptVerify {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Density receipt path to verify.
        #[arg(
            long,
            default_value = "reports/condense/studio_density_receipt.local.json"
        )]
        path: PathBuf,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Build a signed native runtime/TQ proof-mode contract.
    RuntimeContractBuild {
        /// Repository root containing reports/condense.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the signed runtime contract here.
        #[arg(
            long,
            default_value = "reports/condense/studio_runtime_contract.local.json"
        )]
        out: PathBuf,
        /// Emit machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify a signed native runtime/TQ proof-mode contract.
    RuntimeContractVerify {
        /// Repository root containing reports/condense.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Runtime contract path to verify.
        #[arg(
            long,
            default_value = "reports/condense/studio_runtime_contract.local.json"
        )]
        path: PathBuf,
        /// Emit machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Run the frontier lifecycle view through the existing guarded operator tool.
    Lifecycle {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long, default_value_t = 8000.0)]
        storage_budget_gb: f64,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print compact frontier status through the guarded operator tool.
    Status {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long, default_value_t = 8000.0)]
        storage_budget_gb: f64,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print storage-aware frontier waves and download ETAs.
    StoragePlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long, default_value_t = 8000.0)]
        storage_budget_gb: f64,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Run a dry launch-gate check through the existing guarded operator tool.
    Gate {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Gate phase: procure or claim.
        #[arg(long, default_value = "procure")]
        phase: String,
        /// Require a refresh ledger for this gate.
        #[arg(long)]
        require_refresh: Option<PathBuf>,
        /// Development escape hatch mirrored from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        allow_unreviewed: bool,
        #[arg(long)]
        storage_budget_gb: Option<f64>,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Build a signed Studio wave-0 launch packet from dry-run proof artifacts.
    LaunchPacketBuild {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the signed launch packet here.
        #[arg(
            long,
            default_value = "reports/condense/studio_wave0_launch_packet.json"
        )]
        out: PathBuf,
        /// Refresh ledger required by the procurement gate.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh.preflight.json"
        )]
        require_refresh: PathBuf,
        /// Signed license workbook to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_license_decisions.draft.json"
        )]
        license_decisions: PathBuf,
        /// Signed refresh-review workbook to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh_review_decisions.draft.json"
        )]
        review_decisions: PathBuf,
        /// Signed worktree split plan to summarize.
        #[arg(
            long,
            default_value = "reports/condense/worktree_split_plan.local.json"
        )]
        worktree_plan: PathBuf,
        /// Signed native runtime/TQ proof-mode contract to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_runtime_contract.local.json"
        )]
        runtime_contract: PathBuf,
        /// Local proof-pack summary to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_proof_pack.local.json"
        )]
        proof_pack: PathBuf,
        /// Optional run-next label/hf id to target.
        #[arg(long)]
        label: Option<String>,
        #[arg(long, default_value_t = 8000.0)]
        storage_budget_gb: f64,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify a signed Studio wave-0 launch packet.
    LaunchPacketVerify {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Launch packet path to verify.
        #[arg(
            long,
            default_value = "reports/condense/studio_wave0_launch_packet.json"
        )]
        path: PathBuf,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Build a signed Studio audit-grade receipt from the audit scorecard and proof artifacts.
    AuditGradeBuild {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the signed audit-grade receipt here.
        #[arg(long, default_value = "reports/condense/studio_audit_grade.local.json")]
        out: PathBuf,
        /// Harsher audit target grade.
        #[arg(long, default_value_t = 8.4)]
        target_grade: f64,
        /// External deep-audit markdown path.
        #[arg(
            long,
            default_value = "/Users/scammermike/Downloads/project_audits/hawking_deep_audit_2026_07_08.md"
        )]
        external_audit: PathBuf,
        /// Studio deep-audit markdown path.
        #[arg(long, default_value = "docs/plans/STUDIO_DEEP_AUDIT_2026_07_08.md")]
        studio_audit: PathBuf,
        /// Signed wave-0 launch packet to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_wave0_launch_packet.json"
        )]
        launch_packet: PathBuf,
        /// Local proof-pack summary to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_proof_pack.local.json"
        )]
        proof_pack: PathBuf,
        /// Signed worktree split plan to summarize.
        #[arg(
            long,
            default_value = "reports/condense/worktree_split_plan.local.json"
        )]
        worktree_plan: PathBuf,
        /// Signed native runtime/TQ proof-mode contract to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_runtime_contract.local.json"
        )]
        runtime_contract: PathBuf,
        /// Signed local density receipt to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_density_receipt.local.json"
        )]
        density_receipt: PathBuf,
        /// Claim launch gate to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_claim_launch_gate.local.json"
        )]
        claim_gate: PathBuf,
        /// Procurement launch gate to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_launch_gate.preflight.json"
        )]
        procurement_gate: PathBuf,
        /// Public scorecard JSON to summarize if present.
        #[arg(long, default_value = "reports/condense/scorecard.json")]
        scorecard: PathBuf,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify a signed Studio audit-grade receipt.
    AuditGradeVerify {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Audit-grade receipt path to verify.
        #[arg(long, default_value = "reports/condense/studio_audit_grade.local.json")]
        path: PathBuf,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Build a signed Hawking Studio 10/10 completion audit.
    CompletionAuditBuild {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write the signed completion audit here.
        #[arg(
            long,
            default_value = "reports/condense/studio_completion_audit.local.json"
        )]
        out: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        #[arg(long = "label")]
        label: Vec<String>,
        /// Signed Studio preflight summary to summarize.
        #[arg(long, default_value = "reports/condense/studio_preflight_summary.json")]
        preflight_summary: PathBuf,
        /// Signed Studio environment receipt to summarize.
        #[arg(long, default_value = "reports/condense/studio_environment.json")]
        environment: PathBuf,
        /// Signed wave-0 launch packet to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_wave0_launch_packet.json"
        )]
        launch_packet: PathBuf,
        /// Signed worktree split plan to summarize.
        #[arg(
            long,
            default_value = "reports/condense/worktree_split_plan.local.json"
        )]
        worktree_plan: PathBuf,
        /// Signed native runtime/TQ proof-mode contract to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_runtime_contract.local.json"
        )]
        runtime_contract: PathBuf,
        /// Signed local density receipt to summarize.
        #[arg(
            long,
            default_value = "reports/condense/studio_density_receipt.local.json"
        )]
        density_receipt: PathBuf,
        /// Local proof-pack summary to summarize.
        #[arg(
            long,
            default_value = "reports/condense/frontier_proof_pack.local.json"
        )]
        proof_pack: PathBuf,
        /// Signed audit-grade receipt to summarize.
        #[arg(long, default_value = "reports/condense/studio_audit_grade.local.json")]
        audit_grade: PathBuf,
        /// Refresh ledger required by the launch gates.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh.preflight.json"
        )]
        require_refresh: PathBuf,
        #[arg(long, default_value_t = 8000.0)]
        storage_budget_gb: f64,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify a signed Hawking Studio 10/10 completion audit.
    CompletionAuditVerify {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Completion audit path to verify.
        #[arg(
            long,
            default_value = "reports/condense/studio_completion_audit.local.json"
        )]
        path: PathBuf,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print the next frontier command without executing it.
    RunNext {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long)]
        label: Option<String>,
        #[arg(long, default_value_t = 8000.0)]
        storage_budget_gb: f64,
        #[arg(long, default_value_t = 300.0)]
        link_mbs: f64,
        #[arg(long, default_value_t = 0.7)]
        efficiency: f64,
        #[arg(long, default_value_t = 200.0)]
        scratch_gb: f64,
        #[arg(long, default_value_t = 128.0)]
        cache_reserve_gb: f64,
        #[arg(long, default_value_t = 6.0)]
        max_wave_hours: f64,
        /// Require a refresh ledger before any download command is even printed as runnable.
        #[arg(long)]
        require_refresh: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft all signed proof envelopes and blocked local claim bundles.
    ProofPack {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Overwrite existing draft/local proof artifacts. Final receipts are still preserved.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Dangerous: allow replacing final evidence or admissible bundles.
        #[arg(long, default_value_t = false)]
        force_final: bool,
        /// Suffix for generated local claim bundles.
        #[arg(long, default_value = ".local")]
        claim_suffix: String,
        /// Machine class recorded in draft envelopes.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Write the proof-pack summary here.
        #[arg(
            long,
            default_value = "reports/condense/frontier_proof_pack.local.json"
        )]
        out: PathBuf,
        /// Build serve-only local bundles; RAM-cliff proof packs should not use this.
        #[arg(long, default_value_t = false)]
        no_require_ramcliff: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print strict license/gating approval commands.
    LicensePlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the license plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, verify, or apply signed batch license/gating decisions.
    LicenseDecisions {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Mode: draft, sign, verify, or apply.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Signed license workbook written by draft/sign modes.
        #[arg(
            long,
            default_value = "reports/condense/frontier_license_decisions.draft.json"
        )]
        out: PathBuf,
        /// License workbook read by sign/verify/apply modes.
        #[arg(
            long,
            default_value = "reports/condense/frontier_license_decisions.draft.json"
        )]
        path: PathBuf,
        /// Default license status for draft mode.
        #[arg(long, default_value = "accepted")]
        status: String,
        /// Human/operator name for final draft decisions.
        #[arg(long, default_value = "")]
        by: String,
        /// License identifier recorded for accepted terms.
        #[arg(long, default_value = "")]
        license: String,
        /// Terms URL recorded for accepted terms.
        #[arg(long, default_value = "")]
        terms_url: String,
        /// Optional local terms snapshot path or URL.
        #[arg(long, default_value = "")]
        terms_snapshot: String,
        /// Allowed use: research, internal-research, personal-research, or evaluation.
        #[arg(long, default_value = "")]
        allowed_use: String,
        /// Redistribution policy: none, artifact-only, or per-license.
        #[arg(long, default_value = "")]
        redistribution: String,
        /// Source policy: local-only-delete-after-bake, local-only-retain-per-license, or per-license.
        #[arg(long, default_value = "")]
        source_policy: String,
        /// Operator note for final draft decisions.
        #[arg(long, default_value = "")]
        note: String,
        /// Mark draft decisions final; accepted rows still need all required fields.
        #[arg(long = "final", default_value_t = false)]
        final_decisions: bool,
        /// Required by apply mode before frontier_license_acceptance.json is written.
        #[arg(long, default_value_t = false)]
        confirm: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Record a human license/gating decision for a frontier model.
    RecordLicense {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Frontier label to record license status for.
        label: String,
        /// License state: unreviewed, reviewed, accepted, or rejected.
        #[arg(long)]
        status: String,
        /// Human/operator who made the decision.
        #[arg(long, default_value = "")]
        by: String,
        /// Decision note.
        #[arg(long, default_value = "")]
        note: String,
        /// License identifier recorded for accepted terms.
        #[arg(long, default_value = "")]
        license: String,
        /// Terms URL recorded for accepted terms.
        #[arg(long, default_value = "")]
        terms_url: String,
        /// Optional local terms snapshot path or URL.
        #[arg(long, default_value = "")]
        terms_snapshot: String,
        /// Allowed use: research, internal-research, personal-research, or evaluation.
        #[arg(long, default_value = "")]
        allowed_use: String,
        /// Redistribution policy: none, artifact-only, or per-license.
        #[arg(long, default_value = "")]
        redistribution: String,
        /// Source policy: local-only-delete-after-bake, local-only-retain-per-license, or per-license.
        #[arg(long, default_value = "")]
        source_policy: String,
    },
    /// Write the candidate-review command queue from a refresh ledger.
    ReviewPlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Refresh ledger to turn into a review queue.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh.preflight.json"
        )]
        refresh: PathBuf,
        /// Write the review plan here.
        #[arg(
            long,
            default_value = "reports/condense/frontier_review_plan.local.json"
        )]
        out: PathBuf,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Record a human accept/reject/watch decision for a refresh candidate.
    ReviewCandidate {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Hugging Face model id from the refresh candidate list.
        model_id: String,
        /// Candidate decision: accept, reject, or watch.
        #[arg(long)]
        decision: String,
        /// Human/operator who made the decision.
        #[arg(long)]
        by: String,
        /// Decision note.
        #[arg(long, default_value = "")]
        note: String,
    },
    /// Draft, verify, or apply signed batch refresh-candidate decisions.
    ReviewDecisions {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Mode: draft, verify, or apply.
        mode: String,
        /// Refresh ledger used by draft mode.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh.preflight.json"
        )]
        refresh: PathBuf,
        /// Signed decision workbook written by draft mode.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh_review_decisions.draft.json"
        )]
        out: PathBuf,
        /// Signed decision workbook read by verify/apply modes.
        #[arg(
            long,
            default_value = "reports/condense/frontier_refresh_review_decisions.draft.json"
        )]
        path: PathBuf,
        /// Default candidate decision for draft mode.
        #[arg(long, default_value = "watch")]
        decision: String,
        /// Human/operator name for final draft decisions.
        #[arg(long, default_value = "")]
        by: String,
        /// Operator note for final draft decisions.
        #[arg(long, default_value = "")]
        note: String,
        /// Mark draft decisions final; apply still requires --confirm.
        #[arg(long = "final", default_value_t = false)]
        final_decisions: bool,
        /// Required by apply mode before frontier_refresh_reviews.json is written.
        #[arg(long, default_value_t = false)]
        confirm: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print source-provenance receipt paths and requirements.
    SourceProvenancePlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the source-provenance plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, or verify signed source-provenance receipts.
    SourceProvenanceReceipt {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, sign, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Draft mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Sign mode: allow signing blocked drafts.
        #[arg(long, default_value_t = false)]
        allow_blocked_draft: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print baseline/eval coverage requirements and skeleton paths.
    CoveragePlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the coverage plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, or verify signed baseline/eval coverage receipts.
    CoverageReceipt {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, sign, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Coverage kind: baseline, eval, or both.
        #[arg(long, default_value = "both")]
        kind: String,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Draft mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Sign mode: allow signing blocked drafts.
        #[arg(long, default_value_t = false)]
        allow_blocked_draft: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, or verify signed architecture parity receipts.
    ParityReceipt {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, sign, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Draft mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Sign mode: allow signing blocked drafts.
        #[arg(long, default_value_t = false)]
        allow_blocked_draft: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print strict serve/RAM-cliff receipt requirements.
    ReceiptPlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the receipt plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, or verify signed native serve/RAM-cliff receipts.
    ReceiptRecord {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, sign, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Native receipt kind: serve, ramcliff, or both.
        #[arg(long, default_value = "both")]
        kind: String,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Draft mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Sign mode: allow signing blocked drafts.
        #[arg(long, default_value_t = false)]
        allow_blocked_draft: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print expensive-mode experiment matrix requirements.
    ExperimentPlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the experiment plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, or verify signed expensive-mode experiment matrices.
    ExperimentReceipt {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, sign, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Draft mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Sign mode: allow signing blocked drafts.
        #[arg(long, default_value_t = false)]
        allow_blocked_draft: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print Doctor recovery receipt requirements.
    DoctorRecoveryPlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the Doctor recovery plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, sign, or verify signed Doctor recovery receipts.
    DoctorRecoveryReceipt {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, sign, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Draft mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Sign mode: allow signing blocked drafts.
        #[arg(long, default_value_t = false)]
        allow_blocked_draft: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print same-run Studio evidence bundle requirements.
    EvidenceRunPlan {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write the Studio evidence-run plan here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Draft, build, or verify signed same-run Studio evidence bundles.
    EvidenceRunReceipt {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Receipt mode: draft, build, or verify.
        mode: String,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Write/read receipts from this directory instead of reports/condense.
        #[arg(long, default_value = "")]
        out_dir: String,
        /// Draft mode: overwrite existing draft receipt files.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Draft mode: sign intentionally blocked draft envelopes.
        #[arg(long, default_value_t = false)]
        sign_draft: bool,
        /// Build mode: stable run identifier for the Studio evidence run.
        #[arg(long, default_value = "")]
        run_id: String,
        /// Build mode: exact one-command Studio evidence runner invocation.
        #[arg(long, default_value = "")]
        runner_command: String,
        /// Build mode: source release decision.
        #[arg(long, default_value = "")]
        source_release_decision: String,
        /// Build mode: exact source-release or retain-source command.
        #[arg(long, default_value = "")]
        source_release_command: String,
        /// Build mode: reason for the source lifecycle decision.
        #[arg(long, default_value = "")]
        source_release_reason: String,
        /// Build mode: operator making the source lifecycle decision.
        #[arg(long, default_value = "")]
        decided_by: String,
        /// Draft/build mode: machine class recorded in the receipt envelope.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Verify signed public-claim bundle(s) without building new evidence.
    ClaimBundleVerify {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Bundle path(s); default verifies manifest bundle paths.
        path: Vec<PathBuf>,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Build signed public-claim bundle(s) from final evidence receipts.
    ClaimBundleBuild {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Optional frontier label(s); default all frontier models.
        label: Vec<String>,
        /// Output path for a single-label bundle.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Build a serve-only bundle; RAM-cliff public wins should not use this.
        #[arg(long, default_value_t = false)]
        no_require_ramcliff: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Capture a strict native `.tq` serve-bench JSON as a signed serve receipt.
    ServeCapture {
        /// Repository root containing tools/condense/frontier_ops.py.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Frontier label to write a serve receipt for.
        label: String,
        /// Existing `.tq` artifact served by the benchmark.
        #[arg(long)]
        artifact: PathBuf,
        /// JSON report emitted by Hawking's native serve bench.
        #[arg(long)]
        bench_json: PathBuf,
        /// Exact command that produced the serve bench report.
        #[arg(long)]
        command: String,
        /// Load trace receipt proving the `.tq` artifact loaded resident without f16 rehydrate.
        #[arg(long)]
        load_receipt: String,
        /// Served-forward trace receipt proving native execution reached the model path.
        #[arg(long)]
        served_forward_receipt: String,
        /// Parity trace receipt paired with the serve run.
        #[arg(long)]
        parity_receipt: String,
        /// Machine class recorded in the serve receipt.
        #[arg(long, default_value = "Studio-M1Ultra-128")]
        machine_class: String,
        /// Write the serve receipt here instead of the default reports/condense path.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Overwrite an existing serve receipt.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Emit machine-readable JSON from frontier_ops.py.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

pub fn run(cmd: StudioCmd) -> Result<()> {
    match cmd {
        StudioCmd::Preflight { root, quiet } => {
            let mut owned = Vec::new();
            if quiet {
                owned.push("--quiet".to_string());
            }
            let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
            run_preflight(&root, &refs)
        }
        StudioCmd::VerifySummary { root, path } => {
            let owned = ["--verify-summary".to_string(), path.display().to_string()];
            let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
            run_preflight(&root, &refs)
        }
        StudioCmd::EnvironmentCapture {
            root,
            out,
            machine_class,
            link_mbs,
            min_ram_gb,
            min_free_disk_gb,
            skip_network,
            json,
        } => {
            let mut owned = vec![
                "capture".to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--machine-class".to_string(),
                machine_class,
                "--link-mbs".to_string(),
                link_mbs.to_string(),
                "--min-ram-gb".to_string(),
                min_ram_gb.to_string(),
                "--min-free-disk-gb".to_string(),
                min_free_disk_gb.to_string(),
            ];
            if skip_network {
                owned.push("--skip-network".to_string());
            }
            run_studio_environment_owned(&root, owned, json)
        }
        StudioCmd::EnvironmentVerify { root, path, json } => {
            let owned = vec![
                "verify".to_string(),
                "--path".to_string(),
                path.display().to_string(),
            ];
            run_studio_environment_owned(&root, owned, json)
        }
        StudioCmd::Snapshot { root, json } => snapshot(&root, json),
        StudioCmd::WorktreePlan {
            root,
            out,
            verify,
            json,
        } => {
            let mut owned = vec!["worktree-plan".to_string()];
            push_optional_out(&mut owned, out);
            if let Some(path) = verify {
                owned.push("--verify".to_string());
                owned.push(path.display().to_string());
            }
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::DensityReceiptBuild {
            root,
            out,
            previous,
            max_files,
            json,
        } => {
            let mut owned = vec![
                "density-receipt".to_string(),
                "build".to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--max-files".to_string(),
                max_files.to_string(),
            ];
            if let Some(path) = previous {
                owned.push("--previous".to_string());
                owned.push(path.display().to_string());
            }
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::DensityReceiptVerify { root, path, json } => {
            let owned = vec![
                "density-receipt".to_string(),
                "verify".to_string(),
                "--path".to_string(),
                path.display().to_string(),
            ];
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::RuntimeContractBuild { root, out, json } => {
            runtime_contract_build(&root, &out, json)
        }
        StudioCmd::RuntimeContractVerify { root, path, json } => {
            runtime_contract_verify(&root, &path, json)
        }
        StudioCmd::Lifecycle {
            root,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            json,
        } => run_frontier_ops(
            &root,
            &[
                "lifecycle",
                "--storage-budget-gb",
                &storage_budget_gb.to_string(),
                "--link-mbs",
                &link_mbs.to_string(),
                "--efficiency",
                &efficiency.to_string(),
                "--scratch-gb",
                &scratch_gb.to_string(),
                "--cache-reserve-gb",
                &cache_reserve_gb.to_string(),
                "--max-wave-hours",
                &max_wave_hours.to_string(),
            ],
            json,
        ),
        StudioCmd::Status {
            root,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            json,
        } => run_frontier_ops(
            &root,
            &[
                "status",
                "--storage-budget-gb",
                &storage_budget_gb.to_string(),
                "--link-mbs",
                &link_mbs.to_string(),
                "--efficiency",
                &efficiency.to_string(),
                "--scratch-gb",
                &scratch_gb.to_string(),
                "--cache-reserve-gb",
                &cache_reserve_gb.to_string(),
                "--max-wave-hours",
                &max_wave_hours.to_string(),
            ],
            json,
        ),
        StudioCmd::StoragePlan {
            root,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            json,
        } => run_frontier_ops(
            &root,
            &[
                "storage-plan",
                "--storage-budget-gb",
                &storage_budget_gb.to_string(),
                "--link-mbs",
                &link_mbs.to_string(),
                "--efficiency",
                &efficiency.to_string(),
                "--scratch-gb",
                &scratch_gb.to_string(),
                "--cache-reserve-gb",
                &cache_reserve_gb.to_string(),
                "--max-wave-hours",
                &max_wave_hours.to_string(),
            ],
            json,
        ),
        StudioCmd::Gate {
            root,
            phase,
            require_refresh,
            allow_unreviewed,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            json,
        } => {
            if phase != "procure" && phase != "claim" {
                return Err(anyhow!("--phase must be procure or claim"));
            }
            let mut owned = vec![
                "launch-gate".to_string(),
                "--phase".to_string(),
                phase,
                "--link-mbs".to_string(),
                link_mbs.to_string(),
                "--efficiency".to_string(),
                efficiency.to_string(),
                "--scratch-gb".to_string(),
                scratch_gb.to_string(),
                "--cache-reserve-gb".to_string(),
                cache_reserve_gb.to_string(),
                "--max-wave-hours".to_string(),
                max_wave_hours.to_string(),
            ];
            if allow_unreviewed {
                owned.push("--allow-unreviewed".to_string());
            }
            if let Some(path) = require_refresh {
                owned.push("--require-refresh".to_string());
                owned.push(path.display().to_string());
            }
            if let Some(gb) = storage_budget_gb {
                owned.push("--storage-budget-gb".to_string());
                owned.push(gb.to_string());
            }
            let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
            run_frontier_ops(&root, &refs, json)
        }
        StudioCmd::LaunchPacketBuild {
            root,
            out,
            require_refresh,
            license_decisions,
            review_decisions,
            worktree_plan,
            runtime_contract,
            proof_pack,
            label,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            json,
        } => {
            let mut owned = vec![
                "launch-packet".to_string(),
                "build".to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--require-refresh".to_string(),
                require_refresh.display().to_string(),
                "--license-decisions".to_string(),
                license_decisions.display().to_string(),
                "--review-decisions".to_string(),
                review_decisions.display().to_string(),
                "--worktree-plan".to_string(),
                worktree_plan.display().to_string(),
                "--runtime-contract".to_string(),
                runtime_contract.display().to_string(),
                "--proof-pack".to_string(),
                proof_pack.display().to_string(),
                "--storage-budget-gb".to_string(),
                storage_budget_gb.to_string(),
                "--link-mbs".to_string(),
                link_mbs.to_string(),
                "--efficiency".to_string(),
                efficiency.to_string(),
                "--scratch-gb".to_string(),
                scratch_gb.to_string(),
                "--cache-reserve-gb".to_string(),
                cache_reserve_gb.to_string(),
                "--max-wave-hours".to_string(),
                max_wave_hours.to_string(),
            ];
            if let Some(label) = label {
                owned.push("--label".to_string());
                owned.push(label);
            }
            run_frontier_owned_allowing(&root, owned, json, &[1])
        }
        StudioCmd::LaunchPacketVerify { root, path, json } => {
            let owned = vec![
                "launch-packet".to_string(),
                "verify".to_string(),
                "--path".to_string(),
                path.display().to_string(),
            ];
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::AuditGradeBuild {
            root,
            out,
            target_grade,
            external_audit,
            studio_audit,
            launch_packet,
            proof_pack,
            worktree_plan,
            runtime_contract,
            density_receipt,
            claim_gate,
            procurement_gate,
            scorecard,
            json,
        } => {
            let owned = vec![
                "audit-grade".to_string(),
                "build".to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--target-grade".to_string(),
                target_grade.to_string(),
                "--external-audit".to_string(),
                external_audit.display().to_string(),
                "--studio-audit".to_string(),
                studio_audit.display().to_string(),
                "--launch-packet".to_string(),
                launch_packet.display().to_string(),
                "--proof-pack".to_string(),
                proof_pack.display().to_string(),
                "--worktree-plan".to_string(),
                worktree_plan.display().to_string(),
                "--runtime-contract".to_string(),
                runtime_contract.display().to_string(),
                "--density-receipt".to_string(),
                density_receipt.display().to_string(),
                "--claim-gate".to_string(),
                claim_gate.display().to_string(),
                "--procurement-gate".to_string(),
                procurement_gate.display().to_string(),
                "--scorecard".to_string(),
                scorecard.display().to_string(),
            ];
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::AuditGradeVerify { root, path, json } => {
            let owned = vec![
                "audit-grade".to_string(),
                "verify".to_string(),
                "--path".to_string(),
                path.display().to_string(),
            ];
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::CompletionAuditBuild {
            root,
            out,
            label,
            preflight_summary,
            environment,
            launch_packet,
            worktree_plan,
            runtime_contract,
            density_receipt,
            proof_pack,
            audit_grade,
            require_refresh,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            json,
        } => {
            let mut owned = vec![
                "completion-audit".to_string(),
                "build".to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--preflight-summary".to_string(),
                preflight_summary.display().to_string(),
                "--environment".to_string(),
                environment.display().to_string(),
                "--launch-packet".to_string(),
                launch_packet.display().to_string(),
                "--worktree-plan".to_string(),
                worktree_plan.display().to_string(),
                "--runtime-contract".to_string(),
                runtime_contract.display().to_string(),
                "--density-receipt".to_string(),
                density_receipt.display().to_string(),
                "--proof-pack".to_string(),
                proof_pack.display().to_string(),
                "--audit-grade".to_string(),
                audit_grade.display().to_string(),
                "--require-refresh".to_string(),
                require_refresh.display().to_string(),
                "--storage-budget-gb".to_string(),
                storage_budget_gb.to_string(),
                "--link-mbs".to_string(),
                link_mbs.to_string(),
                "--efficiency".to_string(),
                efficiency.to_string(),
                "--scratch-gb".to_string(),
                scratch_gb.to_string(),
                "--cache-reserve-gb".to_string(),
                cache_reserve_gb.to_string(),
                "--max-wave-hours".to_string(),
                max_wave_hours.to_string(),
            ];
            for label in label {
                owned.push("--label".to_string());
                owned.push(label);
            }
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::CompletionAuditVerify { root, path, json } => {
            let owned = vec![
                "completion-audit".to_string(),
                "verify".to_string(),
                "--path".to_string(),
                path.display().to_string(),
            ];
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::RunNext {
            root,
            label,
            storage_budget_gb,
            link_mbs,
            efficiency,
            scratch_gb,
            cache_reserve_gb,
            max_wave_hours,
            require_refresh,
            json,
        } => {
            let mut owned = vec![
                "run-next".to_string(),
                "--storage-budget-gb".to_string(),
                storage_budget_gb.to_string(),
                "--link-mbs".to_string(),
                link_mbs.to_string(),
                "--efficiency".to_string(),
                efficiency.to_string(),
                "--scratch-gb".to_string(),
                scratch_gb.to_string(),
                "--cache-reserve-gb".to_string(),
                cache_reserve_gb.to_string(),
                "--max-wave-hours".to_string(),
                max_wave_hours.to_string(),
            ];
            if let Some(label) = label {
                owned.push("--label".to_string());
                owned.push(label);
            }
            if let Some(path) = require_refresh {
                owned.push("--require-refresh".to_string());
                owned.push(path.display().to_string());
            }
            let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
            run_frontier_ops(&root, &refs, json)
        }
        StudioCmd::ProofPack {
            root,
            label,
            force,
            force_final,
            claim_suffix,
            machine_class,
            out,
            no_require_ramcliff,
            json,
        } => {
            let mut owned = vec![
                "proof-pack".to_string(),
                "--claim-suffix".to_string(),
                claim_suffix,
                "--machine-class".to_string(),
                machine_class,
                "--out".to_string(),
                out.display().to_string(),
            ];
            if force {
                owned.push("--force".to_string());
            }
            if force_final {
                owned.push("--force-final".to_string());
            }
            if no_require_ramcliff {
                owned.push("--no-require-ramcliff".to_string());
            }
            owned.extend(label);
            let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
            run_frontier_ops(&root, &refs, json)
        }
        StudioCmd::LicensePlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["license-plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::LicenseDecisions {
            root,
            mode,
            label,
            out,
            path,
            status,
            by,
            license,
            terms_url,
            terms_snapshot,
            allowed_use,
            redistribution,
            source_policy,
            note,
            final_decisions,
            confirm,
            json,
        } => {
            if !matches!(mode.as_str(), "draft" | "sign" | "verify" | "apply") {
                return Err(anyhow!(
                    "license-decisions mode must be draft, sign, verify, or apply"
                ));
            }
            if !matches!(
                status.as_str(),
                "unreviewed" | "reviewed" | "accepted" | "rejected"
            ) {
                return Err(anyhow!(
                    "--status must be unreviewed, reviewed, accepted, or rejected"
                ));
            }
            let mut owned = vec!["license-decisions".to_string(), mode.clone()];
            if mode == "draft" {
                owned.push("--out".to_string());
                owned.push(out.display().to_string());
                owned.push("--status".to_string());
                owned.push(status);
                push_nonempty_arg(&mut owned, "--by", by);
                push_nonempty_arg(&mut owned, "--license", license);
                push_nonempty_arg(&mut owned, "--terms-url", terms_url);
                push_nonempty_arg(&mut owned, "--terms-snapshot", terms_snapshot);
                push_nonempty_arg(&mut owned, "--allowed-use", allowed_use);
                push_nonempty_arg(&mut owned, "--redistribution", redistribution);
                push_nonempty_arg(&mut owned, "--source-policy", source_policy);
                push_nonempty_arg(&mut owned, "--note", note);
                if final_decisions {
                    owned.push("--final".to_string());
                }
                owned.extend(label);
            } else {
                owned.push("--path".to_string());
                owned.push(path.display().to_string());
                if mode == "sign" {
                    owned.push("--out".to_string());
                    owned.push(out.display().to_string());
                }
                if confirm {
                    owned.push("--confirm".to_string());
                }
            }
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::RecordLicense {
            root,
            label,
            status,
            by,
            note,
            license,
            terms_url,
            terms_snapshot,
            allowed_use,
            redistribution,
            source_policy,
        } => {
            let mut owned = vec![
                "record-license".to_string(),
                label,
                "--status".to_string(),
                status,
            ];
            push_nonempty_arg(&mut owned, "--by", by);
            push_nonempty_arg(&mut owned, "--note", note);
            push_nonempty_arg(&mut owned, "--license", license);
            push_nonempty_arg(&mut owned, "--terms-url", terms_url);
            push_nonempty_arg(&mut owned, "--terms-snapshot", terms_snapshot);
            push_nonempty_arg(&mut owned, "--allowed-use", allowed_use);
            push_nonempty_arg(&mut owned, "--redistribution", redistribution);
            push_nonempty_arg(&mut owned, "--source-policy", source_policy);
            run_frontier_owned(&root, owned, false)
        }
        StudioCmd::ReviewPlan {
            root,
            refresh,
            out,
            json,
        } => {
            let owned = vec![
                "review-plan".to_string(),
                "--refresh".to_string(),
                refresh.display().to_string(),
                "--out".to_string(),
                out.display().to_string(),
            ];
            run_frontier_owned_allowing(&root, owned, json, &[1])
        }
        StudioCmd::ReviewCandidate {
            root,
            model_id,
            decision,
            by,
            note,
        } => {
            let mut owned = vec![
                "review-candidate".to_string(),
                model_id,
                "--decision".to_string(),
                decision,
                "--by".to_string(),
                by,
            ];
            push_nonempty_arg(&mut owned, "--note", note);
            run_frontier_owned(&root, owned, false)
        }
        StudioCmd::ReviewDecisions {
            root,
            mode,
            refresh,
            out,
            path,
            decision,
            by,
            note,
            final_decisions,
            confirm,
            json,
        } => {
            if !matches!(mode.as_str(), "draft" | "verify" | "apply") {
                return Err(anyhow!(
                    "review-decisions mode must be draft, verify, or apply"
                ));
            }
            if !matches!(decision.as_str(), "accept" | "reject" | "watch") {
                return Err(anyhow!("--decision must be accept, reject, or watch"));
            }
            let mut owned = vec!["review-decisions".to_string(), mode.clone()];
            if mode == "draft" {
                owned.push("--refresh".to_string());
                owned.push(refresh.display().to_string());
                owned.push("--out".to_string());
                owned.push(out.display().to_string());
                owned.push("--decision".to_string());
                owned.push(decision);
                push_nonempty_arg(&mut owned, "--by", by);
                push_nonempty_arg(&mut owned, "--note", note);
                if final_decisions {
                    owned.push("--final".to_string());
                }
            } else {
                owned.push("--path".to_string());
                owned.push(path.display().to_string());
                if confirm {
                    owned.push("--confirm".to_string());
                }
            }
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::SourceProvenancePlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["source-provenance".to_string(), "plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned_allowing(&root, owned, json, &[1])
        }
        StudioCmd::SourceProvenanceReceipt {
            root,
            mode,
            label,
            out_dir,
            force,
            sign_draft,
            machine_class,
            allow_blocked_draft,
            json,
        } => {
            if mode != "draft" && mode != "sign" && mode != "verify" {
                return Err(anyhow!(
                    "source-provenance-receipt mode must be draft, sign, or verify"
                ));
            }
            let mut owned = vec!["source-provenance".to_string(), mode.clone()];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "sign" && allow_blocked_draft {
                owned.push("--allow-blocked-draft".to_string());
            }
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::CoveragePlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["coverage-plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::CoverageReceipt {
            root,
            mode,
            label,
            kind,
            out_dir,
            force,
            sign_draft,
            machine_class,
            allow_blocked_draft,
            json,
        } => {
            if mode != "draft" && mode != "sign" && mode != "verify" {
                return Err(anyhow!(
                    "coverage-receipt mode must be draft, sign, or verify"
                ));
            }
            if kind != "baseline" && kind != "eval" && kind != "both" {
                return Err(anyhow!("--kind must be baseline, eval, or both"));
            }
            let mut owned = vec![
                "coverage-receipt".to_string(),
                mode.clone(),
                "--kind".to_string(),
                kind,
            ];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "sign" && allow_blocked_draft {
                owned.push("--allow-blocked-draft".to_string());
            }
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ParityReceipt {
            root,
            mode,
            label,
            out_dir,
            force,
            sign_draft,
            machine_class,
            allow_blocked_draft,
            json,
        } => {
            if mode != "draft" && mode != "sign" && mode != "verify" {
                return Err(anyhow!(
                    "parity-receipt mode must be draft, sign, or verify"
                ));
            }
            let mut owned = vec!["parity-receipt".to_string(), mode.clone()];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "sign" && allow_blocked_draft {
                owned.push("--allow-blocked-draft".to_string());
            }
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ReceiptPlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["receipt-plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ReceiptRecord {
            root,
            mode,
            label,
            kind,
            out_dir,
            force,
            sign_draft,
            machine_class,
            allow_blocked_draft,
            json,
        } => {
            if mode != "draft" && mode != "sign" && mode != "verify" {
                return Err(anyhow!(
                    "receipt-record mode must be draft, sign, or verify"
                ));
            }
            if kind != "serve" && kind != "ramcliff" && kind != "both" {
                return Err(anyhow!("--kind must be serve, ramcliff, or both"));
            }
            let mut owned = vec![
                "receipt-record".to_string(),
                mode.clone(),
                "--kind".to_string(),
                kind,
            ];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "sign" && allow_blocked_draft {
                owned.push("--allow-blocked-draft".to_string());
            }
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ExperimentPlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["experiment-plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ExperimentReceipt {
            root,
            mode,
            label,
            out_dir,
            force,
            sign_draft,
            machine_class,
            allow_blocked_draft,
            json,
        } => {
            if mode != "draft" && mode != "sign" && mode != "verify" {
                return Err(anyhow!(
                    "experiment-receipt mode must be draft, sign, or verify"
                ));
            }
            let mut owned = vec!["experiment-receipt".to_string(), mode.clone()];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "sign" && allow_blocked_draft {
                owned.push("--allow-blocked-draft".to_string());
            }
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::DoctorRecoveryPlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["doctor-recovery-plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned_allowing(&root, owned, json, &[1])
        }
        StudioCmd::DoctorRecoveryReceipt {
            root,
            mode,
            label,
            out_dir,
            force,
            sign_draft,
            machine_class,
            allow_blocked_draft,
            json,
        } => {
            if mode != "draft" && mode != "sign" && mode != "verify" {
                return Err(anyhow!(
                    "doctor-recovery-receipt mode must be draft, sign, or verify"
                ));
            }
            let mut owned = vec!["doctor-recovery-receipt".to_string(), mode.clone()];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "sign" && allow_blocked_draft {
                owned.push("--allow-blocked-draft".to_string());
            }
            owned.extend(label);
            if mode == "draft" {
                run_frontier_owned_allowing(&root, owned, json, &[1])
            } else {
                run_frontier_owned(&root, owned, json)
            }
        }
        StudioCmd::EvidenceRunPlan {
            root,
            label,
            out,
            json,
        } => {
            let mut owned = vec!["evidence-run-plan".to_string()];
            push_optional_out(&mut owned, out);
            owned.extend(label);
            run_frontier_owned_allowing(&root, owned, json, &[1])
        }
        StudioCmd::EvidenceRunReceipt {
            root,
            mode,
            label,
            out_dir,
            force,
            sign_draft,
            run_id,
            runner_command,
            source_release_decision,
            source_release_command,
            source_release_reason,
            decided_by,
            machine_class,
            json,
        } => {
            if mode != "draft" && mode != "build" && mode != "verify" {
                return Err(anyhow!(
                    "evidence-run-receipt mode must be draft, build, or verify"
                ));
            }
            let mut owned = vec!["evidence-run-receipt".to_string(), mode.clone()];
            if !out_dir.is_empty() {
                owned.push("--out-dir".to_string());
                owned.push(out_dir);
            }
            if mode == "draft" {
                if force {
                    owned.push("--force".to_string());
                }
                if sign_draft {
                    owned.push("--sign-draft".to_string());
                }
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            } else if mode == "build" {
                if run_id.is_empty()
                    || runner_command.is_empty()
                    || source_release_decision.is_empty()
                    || source_release_command.is_empty()
                    || source_release_reason.is_empty()
                    || decided_by.is_empty()
                {
                    return Err(anyhow!(
                        "evidence-run-receipt build requires --run-id, --runner-command, --source-release-decision, --source-release-command, --source-release-reason, and --decided-by"
                    ));
                }
                owned.push("--run-id".to_string());
                owned.push(run_id);
                owned.push("--runner-command".to_string());
                owned.push(runner_command);
                owned.push("--source-release-decision".to_string());
                owned.push(source_release_decision);
                owned.push("--source-release-command".to_string());
                owned.push(source_release_command);
                owned.push("--source-release-reason".to_string());
                owned.push(source_release_reason);
                owned.push("--decided-by".to_string());
                owned.push(decided_by);
                owned.push("--machine-class".to_string());
                owned.push(machine_class);
            }
            owned.extend(label);
            if mode == "draft" {
                run_frontier_owned_allowing(&root, owned, json, &[1])
            } else {
                run_frontier_owned(&root, owned, json)
            }
        }
        StudioCmd::ClaimBundleBuild {
            root,
            label,
            out,
            no_require_ramcliff,
            json,
        } => {
            let mut owned = vec!["claim-bundle".to_string(), "build".to_string()];
            if let Some(out) = out {
                owned.push("--out".to_string());
                owned.push(out.display().to_string());
            }
            if no_require_ramcliff {
                owned.push("--no-require-ramcliff".to_string());
            }
            owned.extend(label);
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ClaimBundleVerify { root, path, json } => {
            let mut owned = vec!["claim-bundle".to_string(), "verify".to_string()];
            owned.extend(path.into_iter().map(|p| p.display().to_string()));
            run_frontier_owned(&root, owned, json)
        }
        StudioCmd::ServeCapture {
            root,
            label,
            artifact,
            bench_json,
            command,
            load_receipt,
            served_forward_receipt,
            parity_receipt,
            machine_class,
            out,
            force,
            json,
        } => {
            let mut owned = vec![
                "serve-capture".to_string(),
                label,
                "--artifact".to_string(),
                artifact.display().to_string(),
                "--bench-json".to_string(),
                bench_json.display().to_string(),
                "--command".to_string(),
                command,
                "--load-receipt".to_string(),
                load_receipt,
                "--served-forward-receipt".to_string(),
                served_forward_receipt,
                "--parity-receipt".to_string(),
                parity_receipt,
                "--machine-class".to_string(),
                machine_class,
            ];
            if let Some(out) = out {
                owned.push("--out".to_string());
                owned.push(out.display().to_string());
            }
            if force {
                owned.push("--force".to_string());
            }
            let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
            run_frontier_ops(&root, &refs, json)
        }
    }
}

fn push_optional_out(owned: &mut Vec<String>, out: Option<PathBuf>) {
    if let Some(out) = out {
        owned.push("--out".to_string());
        owned.push(out.display().to_string());
    }
}

fn push_nonempty_arg(owned: &mut Vec<String>, flag: &str, value: String) {
    if !value.is_empty() {
        owned.push(flag.to_string());
        owned.push(value);
    }
}

fn run_frontier_owned(root: &Path, owned: Vec<String>, emit_json: bool) -> Result<()> {
    run_frontier_owned_allowing(root, owned, emit_json, &[])
}

fn run_frontier_owned_allowing(
    root: &Path,
    owned: Vec<String>,
    emit_json: bool,
    allowed_failure_codes: &[i32],
) -> Result<()> {
    let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
    run_frontier_ops_allowing(root, &refs, emit_json, allowed_failure_codes)
}

fn runtime_contract_build(root: &Path, out: &Path, emit_json: bool) -> Result<()> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let out_path = rooted_path(&root, out);
    let mut doc = runtime_contract_doc(&root)?;
    sign_json_doc(&mut doc)?;
    write_json_atomic(&out_path, &doc)?;
    if emit_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "path": out.display().to_string(),
                "schema": doc.get("schema").cloned().unwrap_or(Value::Null),
                "signature_ok": verify_sha256_json_signature(&doc).unwrap_or(false),
                "profile_count": doc.get("profiles").and_then(Value::as_array).map(|v| v.len()).unwrap_or(0),
                "workload_count": doc.get("workloads").and_then(Value::as_array).map(|v| v.len()).unwrap_or(0),
                "proof_mode_required": doc.pointer("/native_tq_proof_mode/required_env").and_then(Value::as_array).map(|v| v.len()).unwrap_or(0),
            }))?
        );
    } else {
        eprintln!(
            "[hawking-studio] wrote native runtime contract {}",
            out_path.display()
        );
    }
    Ok(())
}

fn runtime_contract_verify(root: &Path, path: &Path, emit_json: bool) -> Result<()> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let full = rooted_path(&root, path);
    let doc = read_json(&full)?;
    let schema_ok =
        doc.get("schema").and_then(Value::as_str) == Some("hawking.studio_runtime_contract.v1");
    let signature_ok = verify_sha256_json_signature(&doc).unwrap_or(false);
    let profiles = doc
        .get("profiles")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let required_env = doc
        .pointer("/native_tq_proof_mode/required_env")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let ok = schema_ok && signature_ok && profiles >= 5 && required_env >= 4;
    let status = json!({
        "ok": ok,
        "path": path.display().to_string(),
        "schema_ok": schema_ok,
        "signature_ok": signature_ok,
        "profile_count": profiles,
        "workload_count": doc.get("workloads").and_then(Value::as_array).map(|v| v.len()).unwrap_or(0),
        "proof_mode_required": required_env,
    });
    if emit_json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        eprintln!(
            "[hawking-studio] runtime contract {}: {}",
            if ok { "valid" } else { "INVALID" },
            full.display()
        );
    }
    if ok {
        Ok(())
    } else {
        Err(anyhow!("runtime contract failed verification"))
    }
}

fn runtime_contract_doc(root: &Path) -> Result<Value> {
    let profile_names = ["default", "fast", "race", "efficient", "exact"];
    let profiles = profile_names
        .iter()
        .map(|name| {
            let profile = hawking_serve::RuntimeProfile::from_str(name)
                .ok_or_else(|| anyhow!("missing runtime profile {name}"))?;
            let plan = profile.lever_plan();
            Ok(json!({
                "profile": profile.as_str(),
                "contract": profile.contract(),
                "set_if_unset": plan.set_if_unset.iter().map(|(key, value)| {
                    json!({"key": key, "value": value})
                }).collect::<Vec<_>>(),
                "force_off": plan.force_off,
                "f16_kv": plan.f16_kv,
                "concurrent_qkv": plan.concurrent_qkv,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let workload_names = [
        "default",
        "code-completion",
        "chat-shared-prompt",
        "batch-summarization",
        "local-agent-loop",
    ];
    let workloads = workload_names
        .iter()
        .map(|name| {
            let pack = hawking_serve::WorkloadPack::from_str(name)
                .ok_or_else(|| anyhow!("missing workload pack {name}"))?;
            let (profile, energy, batch) = pack.defaults();
            Ok(json!({
                "workload": pack.as_str(),
                "profile": profile.as_str(),
                "energy_mode": energy.as_str(),
                "batch_policy": batch_policy_name(&batch),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let energy_modes = ["off", "balanced", "efficient"]
        .iter()
        .map(|name| {
            let mode = hawking_serve::EnergyMode::from_str(name)
                .ok_or_else(|| anyhow!("missing energy mode {name}"))?;
            Ok(json!({
                "mode": mode.as_str(),
                "gather_window_ms": mode.gather_window_ms(),
                "waits_single_request": mode.should_gather(1, 1),
                "waits_partial_batch": mode.should_gather(1, 4),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "schema": "hawking.studio_runtime_contract.v1",
        "generated_at_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "root": root,
        "git_commit": git_commit(root),
        "ok": true,
        "source_of_truth": {
            "runtime_profiles": "hawking_serve::RuntimeProfile::lever_plan",
            "workload_packs": "hawking_serve::WorkloadPack::defaults",
            "energy_modes": "hawking_serve::EnergyMode",
        },
        "default_unset_policy": {
            "profile": hawking_serve::RuntimeProfile::default_when_unset().as_str(),
            "force_off": hawking_serve::RuntimeProfile::default_unset_force_off(),
        },
        "profiles": profiles,
        "workloads": workloads,
        "energy_modes": energy_modes,
        "native_tq_proof_mode": {
            "build_feature": "cargo build -p hawking --bin hawking --features tq",
            "required_env": [
                {"key": "HAWKING_QWEN_TQ", "value": "1"},
                {"key": "HAWKING_QWEN_TQ_STRICT", "value": "1"},
                {"key": "HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR", "value": "1"},
                {"key": "HAWKING_QWEN_TQ_REQUIRE_GPU", "value": "1"}
            ],
            "forbidden_or_blocking": [
                "missing .tq sidecar",
                "partial all-linear projection coverage",
                "CPU-only TQ serve when GPU proof is required",
                "f16 rehydrate",
                "missing served-forward parity trace"
            ],
            "admissible_serve_receipt_required": {
                "schema": "hawking.frontier_serve.v1",
                "status": "pass",
                "native_tq": true,
                "rehydrate_f16": false,
                "tq_strict": true,
                "all_linear": true,
                "gpu_bitslice": true,
                "served_forward_pass": true,
                "parity_pass": true,
                "tok_s": ">0",
                "load_receipt": "load/proof trace",
                "memory_peak_gb": ">0",
                "memory_resident_gb": ">0",
                "unified_memory_gb": ">0",
                "resident_memory_ok": true,
                "artifact_sha256": "sha256",
                "commands": "exact non-placeholder command"
            },
            "serve_capture_template": "hawking studio serve-capture <label> --artifact <artifact.tq> --bench-json <serve_report.json> --command '<exact hawking serve bench command>' --load-receipt <load_trace.json> --served-forward-receipt <served_forward_trace.json> --parity-receipt <serve_parity_trace.json> --force"
        },
        "custom_build_pass": {
            "goal": "native .tq runtime proof before public frontier claims",
            "workstreams": [
                "native .tq loader/verifier",
                "Metal bitslice GEMV ownership",
                "all-linear Qwen proof mode",
                "served-forward parity trace",
                "same-box baseline/eval receipt",
                "RAM-cliff/J-token receipt"
            ]
        },
        "note": "This contract is laptop-safe. It proves the required runtime/proof policy, not that a model has served natively yet."
    }))
}

fn batch_policy_name(policy: &hawking_serve::BatchPolicy) -> &'static str {
    match policy {
        hawking_serve::BatchPolicy::Default => "default",
        hawking_serve::BatchPolicy::GreedyFirst => "greedy-first",
        hawking_serve::BatchPolicy::PrefixGrouped => "prefix-grouped",
    }
}

fn rooted_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn git_commit(root: &Path) -> String {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn sign_json_doc(doc: &mut Value) -> Result<()> {
    if let Some(obj) = doc.as_object_mut() {
        obj.remove("signature");
    }
    let bytes = serde_json::to_vec(doc)?;
    let digest = Sha256::digest(bytes);
    doc.as_object_mut()
        .ok_or_else(|| anyhow!("signed document must be a JSON object"))?
        .insert(
            "signature".to_string(),
            json!({"algorithm": "sha256-json-v1", "digest": hex_digest(&digest)}),
        );
    Ok(())
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("runtime-contract");
    let tmp = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        serde_json::to_writer_pretty(&mut file, value)?;
        use std::io::Write;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("replace {} with {}", path.display(), tmp.display()))?;
    Ok(())
}

fn snapshot(root: &Path, emit_json: bool) -> Result<()> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let preflight = artifact(&root, PREFLIGHT_SUMMARY);
    let environment = artifact(&root, STUDIO_ENVIRONMENT);
    let procure_gate = artifact(&root, PROCUREMENT_GATE);
    let claim_gate = artifact(&root, CLAIM_GATE);
    let review_plan = artifact(&root, REVIEW_PLAN);
    let ledger = artifact(&root, LEDGER);
    let proof_pack = artifact(&root, PROOF_PACK);
    let audit_grade = artifact(&root, AUDIT_GRADE);
    let runtime_contract = artifact(&root, RUNTIME_CONTRACT);
    let density_receipt = artifact(&root, DENSITY_RECEIPT);
    let completion_audit = artifact(&root, COMPLETION_AUDIT);

    let doc = json!({
        "schema": "hawking.studio_snapshot.v1",
        "root": root,
        "preflight": preflight_summary(&preflight),
        "environment": environment_summary(&environment),
        "procurement_gate": gate_summary(&procure_gate),
        "claim_gate": gate_summary(&claim_gate),
        "review_plan": review_summary(&review_plan),
        "ledger": ledger_summary(&ledger),
        "proof_pack": proof_pack_summary(&proof_pack),
        "audit_grade": audit_grade_summary(&audit_grade),
        "runtime_contract": runtime_contract_summary(&runtime_contract),
        "density_receipt": density_receipt_summary(&density_receipt),
        "completion_audit": completion_audit_summary(&completion_audit),
        "next_safe_commands": [
            "hawking studio preflight",
            "hawking studio verify-summary --path reports/condense/studio_preflight_summary.json",
            "hawking studio environment-capture --out reports/condense/studio_environment.json",
            "hawking studio environment-verify --path reports/condense/studio_environment.json",
            "hawking studio worktree-plan --out reports/condense/worktree_split_plan.local.json",
            "hawking studio worktree-plan --verify reports/condense/worktree_split_plan.local.json",
            "hawking studio density-receipt-build --out reports/condense/studio_density_receipt.local.json",
            "hawking studio density-receipt-verify --path reports/condense/studio_density_receipt.local.json",
            "hawking studio runtime-contract-build --out reports/condense/studio_runtime_contract.local.json",
            "hawking studio runtime-contract-verify --path reports/condense/studio_runtime_contract.local.json",
            "hawking studio storage-plan --storage-budget-gb 8000 --link-mbs 300 --efficiency 0.7",
            "hawking studio license-plan",
            "hawking studio license-decisions draft --out reports/condense/frontier_license_decisions.draft.json",
            "hawking studio license-decisions verify --path reports/condense/frontier_license_decisions.draft.json",
            "hawking studio review-plan",
            "hawking studio review-decisions draft --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_refresh_review_decisions.draft.json",
            "hawking studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.draft.json",
            "hawking studio source-provenance-receipt verify",
            "hawking studio parity-receipt verify",
            "hawking studio coverage-receipt verify --kind both",
            "hawking studio receipt-record verify --kind both",
            "hawking studio doctor-recovery-plan",
            "hawking studio doctor-recovery-receipt verify",
            "hawking studio evidence-run-plan",
            "hawking studio evidence-run-receipt verify",
            "hawking studio experiment-receipt verify",
            "hawking studio claim-bundle-build",
            "hawking studio claim-bundle-verify",
            "hawking studio lifecycle --storage-budget-gb 8000",
            "hawking studio gate --phase procure --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb 8000",
            "hawking studio proof-pack --force",
            "hawking studio launch-packet-build --out reports/condense/studio_wave0_launch_packet.json",
            "hawking studio launch-packet-verify --path reports/condense/studio_wave0_launch_packet.json",
            "hawking studio audit-grade-build --out reports/condense/studio_audit_grade.local.json",
            "hawking studio audit-grade-verify --path reports/condense/studio_audit_grade.local.json",
            "hawking studio completion-audit-build --out reports/condense/studio_completion_audit.local.json",
            "hawking studio completion-audit-verify --path reports/condense/studio_completion_audit.local.json",
            "hawking studio run-next --require-refresh reports/condense/frontier_refresh.preflight.json"
        ],
    });
    if emit_json {
        println!("{}", serde_json::to_string_pretty(&doc)?);
    } else {
        print_human_snapshot(&doc);
    }
    Ok(())
}

fn artifact(root: &Path, rel: &str) -> Value {
    let path = root.join(rel);
    match read_json(&path) {
        Ok(value) => json!({"path": path, "exists": true, "json": value}),
        Err(error) => json!({"path": path, "exists": false, "error": error.to_string()}),
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn preflight_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    json!({
        "exists": true,
        "ok": data.get("preflight_ok_before_summary").and_then(Value::as_bool).unwrap_or(false),
        "signature_ok": signature_ok,
        "generated_at": data.get("generated_at").cloned().unwrap_or(Value::Null),
        "git_commit": data.get("git_commit").cloned().unwrap_or(Value::Null),
        "hardware": data.get("hardware").cloned().unwrap_or(Value::Null),
        "network": data.get("network").cloned().unwrap_or(Value::Null),
    })
}

fn environment_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    json!({
        "exists": true,
        "ok": data.get("ok").and_then(Value::as_bool).unwrap_or(false),
        "signature_ok": signature_ok,
        "generated_at": data.get("generated_at").cloned().unwrap_or(Value::Null),
        "git_commit": data.get("git_commit").cloned().unwrap_or(Value::Null),
        "machine_class": data.get("machine_class").cloned().unwrap_or(Value::Null),
        "hardware": data.get("hardware").cloned().unwrap_or(Value::Null),
        "network": data.get("network").cloned().unwrap_or(Value::Null),
        "power_thermal": data.get("power_thermal").cloned().unwrap_or(Value::Null),
        "failure_count": data.get("failure_count").cloned().unwrap_or(Value::Null),
        "warning_count": data.get("warning_count").cloned().unwrap_or(Value::Null),
    })
}

fn gate_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false});
    };
    let failing = data
        .get("checks")
        .and_then(Value::as_array)
        .map(|checks| {
            checks
                .iter()
                .filter(|c| {
                    !c.get("ok").and_then(Value::as_bool).unwrap_or(false)
                        && c.get("severity").and_then(Value::as_str) == Some("fail")
                })
                .map(|c| {
                    json!({
                        "name": c.get("name").cloned().unwrap_or(Value::Null),
                        "detail": c.get("detail").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "exists": true,
        "phase": data.get("phase").cloned().unwrap_or(Value::Null),
        "ok": data.get("ok").and_then(Value::as_bool).unwrap_or(false),
        "failure_count": data.get("failure_count").cloned().unwrap_or(Value::Null),
        "warning_count": data.get("warning_count").cloned().unwrap_or(Value::Null),
        "summary": data.get("summary").cloned().unwrap_or(Value::Null),
        "failing": failing,
    })
}

fn review_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false});
    };
    let missing = data
        .get("missing_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "exists": true,
        "ok": missing == 0,
        "review_worthy_count": data.get("review_worthy_count").cloned().unwrap_or(Value::Null),
        "reviewed_count": data.get("reviewed_count").cloned().unwrap_or(Value::Null),
        "missing_count": missing,
        "refresh_path": data.get("refresh_path").cloned().unwrap_or(Value::Null),
    })
}

fn ledger_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false});
    };
    json!({
        "exists": true,
        "model_count": data.pointer("/manifest/model_count").cloned().unwrap_or(Value::Null),
        "total_source_gb": data.pointer("/manifest/total_source_gb").cloned().unwrap_or(Value::Null),
        "total_artifact_gb": data.pointer("/manifest/total_artifact_gb").cloned().unwrap_or(Value::Null),
        "cycle_peak_gb": data.pointer("/cycle_plan/cycle_peak_gb").cloned().unwrap_or(Value::Null),
        "download_eta_hours": data.pointer("/cycle_plan/download_eta_hours").cloned().unwrap_or(Value::Null),
        "storage_wave_count": data.pointer("/storage_wave_plan/wave_count").cloned().unwrap_or(Value::Null),
        "planned_peak_gb": data.pointer("/storage_wave_plan/planned_peak_gb").cloned().unwrap_or(Value::Null),
        "hardware": data.get("hardware").cloned().unwrap_or(Value::Null),
    })
}

fn proof_pack_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    let bundles = data
        .get("claim_bundles")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    json!({
                        "label": row.get("label").cloned().unwrap_or(Value::Null),
                        "claim_admissible": row.get("claim_admissible").cloned().unwrap_or(Value::Null),
                        "blocker_count": row.get("blocker_count").cloned().unwrap_or(Value::Null),
                        "written": row.get("written").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "exists": true,
        "ok": data.get("ok").and_then(Value::as_bool).unwrap_or(false) && signature_ok,
        "signature_ok": signature_ok,
        "git_commit": data.get("git_commit").cloned().unwrap_or(Value::Null),
        "model_count": data.get("model_count").cloned().unwrap_or(Value::Null),
        "blocked_claim_count": data.get("blocked_claim_count").cloned().unwrap_or(Value::Null),
        "evidence_count": data.get("evidence_rows").and_then(Value::as_array).map(|r| r.len()).unwrap_or(0),
        "claim_bundles": bundles,
    })
}

fn runtime_contract_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    let profile_count = data
        .get("profiles")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let workload_count = data
        .get("workloads")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let proof_mode_required = data
        .pointer("/native_tq_proof_mode/required_env")
        .and_then(Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let ok = data.get("ok").and_then(Value::as_bool).unwrap_or(false)
        && signature_ok
        && profile_count >= 5
        && proof_mode_required >= 4;
    json!({
        "exists": true,
        "ok": ok,
        "signature_ok": signature_ok,
        "generated_at_unix": data.get("generated_at_unix").cloned().unwrap_or(Value::Null),
        "git_commit": data.get("git_commit").cloned().unwrap_or(Value::Null),
        "profile_count": profile_count,
        "workload_count": workload_count,
        "proof_mode_required": proof_mode_required,
        "default_unset_policy": data.get("default_unset_policy").cloned().unwrap_or(Value::Null),
        "native_tq_proof_mode": data.get("native_tq_proof_mode").cloned().unwrap_or(Value::Null),
    })
}

fn density_receipt_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    let density = data.get("density").cloned().unwrap_or(Value::Null);
    json!({
        "exists": true,
        "ok": data.get("ok").and_then(Value::as_bool).unwrap_or(false) && signature_ok,
        "signature_ok": signature_ok,
        "git_commit": data.get("git_commit").cloned().unwrap_or(Value::Null),
        "density_risk": data.get("density_risk").cloned().unwrap_or(Value::Null),
        "repo_total_gb": density.get("repo_total_gb").cloned().unwrap_or(Value::Null),
        "local_generated_gb": density.get("local_generated_gb").cloned().unwrap_or(Value::Null),
        "tracked_loc": density.get("tracked_loc").cloned().unwrap_or(Value::Null),
        "disk_free_gb": density.get("disk_free_gb").cloned().unwrap_or(Value::Null),
        "largest_files": data.get("largest_files").cloned().unwrap_or(Value::Null),
        "cleanup_recommendations": data.get("cleanup_recommendations").cloned().unwrap_or(Value::Null),
    })
}

fn audit_grade_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    json!({
        "exists": true,
        "ok": data.get("ok").and_then(Value::as_bool).unwrap_or(false),
        "signature_ok": signature_ok,
        "target_grade": data.get("target_grade").cloned().unwrap_or(Value::Null),
        "target_reached": data.get("target_reached").cloned().unwrap_or(Value::Null),
        "frontier_claims_walled": data.get("frontier_claims_walled").cloned().unwrap_or(Value::Null),
        "facet_count": data.get("facet_count").cloned().unwrap_or(Value::Null),
        "below_target_count": data.get("below_target_count").cloned().unwrap_or(Value::Null),
        "facet_min": data.get("facet_min").cloned().unwrap_or(Value::Null),
        "external_audit": data.get("external_audit").cloned().unwrap_or(Value::Null),
        "current_state": data.get("current_state").cloned().unwrap_or(Value::Null),
        "blockers": data.get("blockers").cloned().unwrap_or(Value::Null),
    })
}

fn completion_audit_summary(artifact: &Value) -> Value {
    let Some(data) = artifact.get("json") else {
        return json!({"exists": false, "ok": false, "signature_ok": false});
    };
    let signature_ok = verify_sha256_json_signature(data).unwrap_or(false);
    json!({
        "exists": true,
        "ok": data.get("ok").and_then(Value::as_bool).unwrap_or(false),
        "completion_ok": data.get("completion_ok").and_then(Value::as_bool).unwrap_or(false),
        "signature_ok": signature_ok,
        "frontier_label_count": data.get("frontier_label_count").cloned().unwrap_or(Value::Null),
        "required_count": data.get("required_count").cloned().unwrap_or(Value::Null),
        "passed_count": data.get("passed_count").cloned().unwrap_or(Value::Null),
        "blocked_count": data.get("blocked_count").cloned().unwrap_or(Value::Null),
        "blocked_requirements": data.get("blocked_requirements").cloned().unwrap_or(Value::Null),
    })
}

fn verify_sha256_json_signature(data: &Value) -> Result<bool> {
    let Some(sig) = data.get("signature").and_then(Value::as_object) else {
        return Ok(false);
    };
    if sig.get("algorithm").and_then(Value::as_str) != Some("sha256-json-v1") {
        return Ok(false);
    }
    let Some(expected) = sig.get("digest").and_then(Value::as_str) else {
        return Ok(false);
    };
    let mut unsigned = data.clone();
    if let Some(obj) = unsigned.as_object_mut() {
        obj.remove("signature");
    }
    let bytes = serde_json::to_vec(&unsigned)?;
    let digest = Sha256::digest(bytes);
    if hex_digest(&digest) == expected {
        return Ok(true);
    }
    let serialized = serde_json::to_string(&unsigned)?;
    let python_style = python_compatible_json_numbers(&serialized);
    if python_style != serialized {
        let digest = Sha256::digest(python_style.as_bytes());
        return Ok(hex_digest(&digest) == expected);
    }
    Ok(false)
}

fn python_compatible_json_numbers(serialized: &str) -> String {
    let mut out = String::with_capacity(serialized.len());
    let chars: Vec<char> = serialized.chars().collect();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if ch == '"' {
            in_string = true;
            out.push(ch);
            i += 1;
            continue;
        }
        if ch == '-' || ch.is_ascii_digit() {
            let start = i;
            if ch == '-' {
                i += 1;
            }
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < chars.len() && chars[i] == '.' {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '-' || chars[i] == '+') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let token: String = chars[start..i].iter().collect();
            out.push_str(&python_compatible_number_token(&token));
            continue;
        }
        out.push(ch);
        i += 1;
    }
    out
}

fn python_compatible_number_token(token: &str) -> String {
    if token.contains('.') && !token.contains('e') && !token.contains('E') {
        if let Ok(value) = token.parse::<f64>() {
            let abs = value.abs();
            if abs > 0.0 && abs < 0.0001 {
                return format_python_scientific(value);
            }
        }
    }
    pad_single_digit_exponent_token(token)
}

fn format_python_scientific(value: f64) -> String {
    let raw = format!("{value:e}");
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return raw;
    };
    let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
    let exp_value = exponent.parse::<i32>().unwrap_or(0);
    let sign = if exp_value < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exp_value.abs())
}

fn pad_single_digit_exponent_token(token: &str) -> String {
    let Some(idx) = token.find(['e', 'E']) else {
        return token.to_string();
    };
    let (mantissa, exponent_with_e) = token.split_at(idx);
    let exponent = &exponent_with_e[1..];
    let Some(sign) = exponent.chars().next() else {
        return token.to_string();
    };
    if sign != '-' && sign != '+' {
        return token.to_string();
    }
    let digits = &exponent[1..];
    if digits.len() == 1 && digits.chars().all(|ch| ch.is_ascii_digit()) {
        return format!("{mantissa}e{sign}0{digits}");
    }
    if token.as_bytes()[idx] == b'E' {
        return format!("{mantissa}e{exponent}");
    }
    token.to_string()
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn print_human_snapshot(doc: &Value) {
    println!("== Hawking Studio Snapshot ==");
    println!(
        "root: {}",
        doc.get("root").and_then(Value::as_str).unwrap_or("?")
    );

    let pre = &doc["preflight"];
    println!(
        "preflight: {}  signature={}  generated={}",
        verdict(pre["ok"].as_bool().unwrap_or(false)),
        verdict(pre["signature_ok"].as_bool().unwrap_or(false)),
        pre["generated_at"].as_str().unwrap_or("?")
    );
    if let Some(hw) = pre.get("hardware") {
        println!(
            "hardware: RAM={}GB target={}GB  disk_free={}GB target={}GB  power={}  thermal={}",
            n(hw, "actual_ram_gb"),
            n(hw, "target_ram_gb"),
            n(hw, "disk_free_gb"),
            n(hw, "target_free_disk_gb"),
            s(hw, "power_source"),
            s(hw, "thermal_status")
        );
    }
    if let Some(net) = pre.get("network") {
        println!(
            "network: dns={} api_status={} api_ms={} route_if={} route_gateway={}",
            v(net, "dns_ok"),
            v(net, "hf_api_status"),
            n(net, "hf_api_elapsed_ms"),
            s(net, "route_interface"),
            s(net, "route_gateway")
        );
    }
    let env = &doc["environment"];
    println!(
        "environment: {}  signature={}  machine={}  failures={} warnings={}",
        verdict(env["ok"].as_bool().unwrap_or(false)),
        verdict(env["signature_ok"].as_bool().unwrap_or(false)),
        s(env, "machine_class"),
        env["failure_count"].as_u64().unwrap_or(0),
        env["warning_count"].as_u64().unwrap_or(0)
    );
    if let Some(power) = env.get("power_thermal") {
        println!(
            "measurement surfaces: powermetrics={} path={} sudo={} power={} thermal={}",
            v(power, "powermetrics_available"),
            s(power, "powermetrics_path"),
            v(power, "powermetrics_requires_sudo"),
            v(power, "power_source_ok"),
            v(power, "thermal_status_ok")
        );
    }

    print_gate("procure gate", &doc["procurement_gate"]);
    print_gate("claim gate", &doc["claim_gate"]);

    let review = &doc["review_plan"];
    println!(
        "candidate review: {}/{} reviewed, {} missing",
        review["reviewed_count"].as_u64().unwrap_or(0),
        review["review_worthy_count"].as_u64().unwrap_or(0),
        review["missing_count"].as_u64().unwrap_or(0)
    );

    let ledger = &doc["ledger"];
    println!(
        "frontier ledger: models={} source={}GB tq={}GB cycle_peak={}GB waves={} planned_peak={}GB eta={}h",
        ledger["model_count"].as_u64().unwrap_or(0),
        nf(&ledger["total_source_gb"]),
        nf(&ledger["total_artifact_gb"]),
        nf(&ledger["cycle_peak_gb"]),
        ledger["storage_wave_count"].as_u64().unwrap_or(0),
        nf(&ledger["planned_peak_gb"]),
        nf(&ledger["download_eta_hours"])
    );

    let proof = &doc["proof_pack"];
    println!(
        "proof pack: {}  signature={}  blocked_claims={}/{} evidence_rows={}",
        verdict(proof["ok"].as_bool().unwrap_or(false)),
        verdict(proof["signature_ok"].as_bool().unwrap_or(false)),
        proof["blocked_claim_count"].as_u64().unwrap_or(0),
        proof["model_count"].as_u64().unwrap_or(0),
        proof["evidence_count"].as_u64().unwrap_or(0)
    );

    let runtime = &doc["runtime_contract"];
    println!(
        "runtime contract: {}  signature={}  profiles={} workloads={} proof_env={}",
        verdict(runtime["ok"].as_bool().unwrap_or(false)),
        verdict(runtime["signature_ok"].as_bool().unwrap_or(false)),
        runtime["profile_count"].as_u64().unwrap_or(0),
        runtime["workload_count"].as_u64().unwrap_or(0),
        runtime["proof_mode_required"].as_u64().unwrap_or(0)
    );

    let density = &doc["density_receipt"];
    println!(
        "density receipt: {}  signature={}  risk={}  repo={}GB local={}GB loc={}",
        verdict(density["ok"].as_bool().unwrap_or(false)),
        verdict(density["signature_ok"].as_bool().unwrap_or(false)),
        density["density_risk"].as_str().unwrap_or("?"),
        nf(&density["repo_total_gb"]),
        nf(&density["local_generated_gb"]),
        density["tracked_loc"].as_u64().unwrap_or(0)
    );

    let audit = &doc["audit_grade"];
    println!(
        "audit grade: signature={}  target={} reached={}  claims_walled={}  facets_below_target={}",
        verdict(audit["signature_ok"].as_bool().unwrap_or(false)),
        nf(&audit["target_grade"]),
        v(audit, "target_reached"),
        v(audit, "frontier_claims_walled"),
        audit["below_target_count"].as_u64().unwrap_or(0)
    );

    let completion = &doc["completion_audit"];
    println!(
        "completion audit: signature={}  completion={}  blocked={}/{}",
        verdict(completion["signature_ok"].as_bool().unwrap_or(false)),
        verdict(completion["completion_ok"].as_bool().unwrap_or(false)),
        completion["blocked_count"].as_u64().unwrap_or(0),
        completion["required_count"].as_u64().unwrap_or(0)
    );

    println!("next safe commands:");
    if let Some(cmds) = doc.get("next_safe_commands").and_then(Value::as_array) {
        for cmd in cmds {
            println!("  {}", cmd.as_str().unwrap_or("?"));
        }
    }
}

fn print_gate(label: &str, gate: &Value) {
    println!(
        "{label}: {}  failures={} warnings={}",
        verdict(gate["ok"].as_bool().unwrap_or(false)),
        gate["failure_count"].as_u64().unwrap_or(0),
        gate["warning_count"].as_u64().unwrap_or(0)
    );
    if let Some(failing) = gate.get("failing").and_then(Value::as_array) {
        for row in failing.iter().take(6) {
            println!(
                "  - {}: {}",
                row["name"].as_str().unwrap_or("?"),
                row["detail"].as_str().unwrap_or("?")
            );
        }
    }
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "OK"
    } else {
        "RED"
    }
}

fn n(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_f64)
        .map(|x| format!("{x:.0}"))
        .unwrap_or_else(|| "?".to_string())
}

fn nf(v: &Value) -> String {
    v.as_f64()
        .map(|x| format!("{x:.0}"))
        .unwrap_or_else(|| "?".to_string())
}

fn s(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or("?")
        .replace('\n', "; ")
        .to_string()
}

fn v(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::String(s)) => s.replace('\n', "; "),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Null) | None => "?".to_string(),
        Some(other) => other.to_string(),
    }
}

fn run_frontier_ops(root: &Path, args: &[&str], emit_json: bool) -> Result<()> {
    run_frontier_ops_allowing(root, args, emit_json, &[])
}

fn run_studio_environment_owned(root: &Path, owned: Vec<String>, emit_json: bool) -> Result<()> {
    let refs = owned.iter().map(String::as_str).collect::<Vec<_>>();
    run_condense_tool(
        root,
        "tools/condense/studio_environment.py",
        &refs,
        emit_json,
    )
}

fn run_condense_tool(root: &Path, rel_tool: &str, args: &[&str], emit_json: bool) -> Result<()> {
    let tool = root.join(rel_tool);
    if !tool.exists() {
        return Err(anyhow!("missing {}", tool.display()));
    }
    let mut cmd = Command::new("python3.12");
    cmd.current_dir(root).arg(&tool);
    for arg in args {
        cmd.arg(arg);
    }
    if emit_json {
        cmd.arg("--json");
    }
    let status = cmd
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("run {rel_tool}"))?;
    if !status.success() {
        return Err(anyhow!("{rel_tool} exited with {status}"));
    }
    Ok(())
}

fn run_frontier_ops_allowing(
    root: &Path,
    args: &[&str],
    emit_json: bool,
    allowed_failure_codes: &[i32],
) -> Result<()> {
    let tool = root.join("tools/condense/frontier_ops.py");
    if !tool.exists() {
        return Err(anyhow!("missing {}", tool.display()));
    }
    let mut cmd = Command::new("python3.12");
    cmd.current_dir(root).arg(tool);
    for arg in args {
        cmd.arg(arg);
    }
    if emit_json {
        cmd.arg("--json");
    }
    let status = cmd
        .stdin(Stdio::null())
        .status()
        .context("run frontier_ops.py")?;
    if !status.success() {
        if let Some(code) = status.code() {
            if allowed_failure_codes.contains(&code) {
                return Ok(());
            }
        }
        return Err(anyhow!("frontier_ops.py exited with {status}"));
    }
    Ok(())
}

fn run_preflight(root: &Path, args: &[&str]) -> Result<()> {
    let tool = root.join("tools/condense/preflight.py");
    if !tool.exists() {
        return Err(anyhow!("missing {}", tool.display()));
    }
    let mut cmd = Command::new("python3.12");
    cmd.current_dir(root).arg(tool);
    for arg in args {
        cmd.arg(arg);
    }
    let status = cmd
        .stdin(Stdio::null())
        .status()
        .context("run preflight.py")?;
    if !status.success() {
        return Err(anyhow!("preflight.py exited with {status}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed(mut v: Value) -> Value {
        let bytes = serde_json::to_vec(&v).unwrap();
        let digest = hex_digest(&Sha256::digest(bytes));
        v.as_object_mut().unwrap().insert(
            "signature".into(),
            json!({"algorithm": "sha256-json-v1", "digest": digest}),
        );
        v
    }

    #[test]
    fn verifies_preflight_style_signature() {
        let v = signed(json!({
            "schema": "hawking.studio_preflight_summary.v1",
            "generated_at": "selftest",
            "preflight_ok_before_summary": false,
        }));
        assert!(verify_sha256_json_signature(&v).unwrap());
        let mut tampered = v.clone();
        tampered["preflight_ok_before_summary"] = json!(true);
        assert!(!verify_sha256_json_signature(&tampered).unwrap());
    }

    #[test]
    fn verifies_python_compatible_number_signature() {
        let mut v = json!({
            "schema": "hawking.studio_density_receipt.v1",
            "ok": true,
            "density": {
                "node_modules_gb": 0.000001,
                "tracked_gb": 0.000097
            },
            "note": "leave string 1e-6 untouched",
        });
        let serialized = serde_json::to_string(&v).unwrap();
        assert!(serialized.contains("1e-6"));
        assert!(serialized.contains("0.000097"));
        let python_style = python_compatible_json_numbers(&serialized);
        assert!(python_style.contains("1e-06"));
        assert!(python_style.contains("9.7e-05"));
        assert!(python_style.contains("leave string 1e-6 untouched"));
        let digest = hex_digest(&Sha256::digest(python_style.as_bytes()));
        v.as_object_mut().unwrap().insert(
            "signature".into(),
            json!({"algorithm": "sha256-json-v1", "digest": digest}),
        );
        assert!(verify_sha256_json_signature(&v).unwrap());
    }

    #[test]
    fn gate_summary_extracts_failing_checks() {
        let artifact = json!({
            "exists": true,
            "json": {
                "phase": "claim",
                "ok": false,
                "failure_count": 1,
                "warning_count": 0,
                "checks": [
                    {"name": "a", "ok": false, "severity": "fail", "detail": "blocked"},
                    {"name": "b", "ok": false, "severity": "warn", "detail": "warn"}
                ]
            }
        });
        let s = gate_summary(&artifact);
        assert_eq!(s["ok"], json!(false));
        assert_eq!(s["failing"].as_array().unwrap().len(), 1);
        assert_eq!(s["failing"][0]["name"], json!("a"));
    }

    #[test]
    fn proof_pack_summary_extracts_blocked_local_bundles() {
        let artifact = json!({
            "exists": true,
            "json": signed(json!({
                "schema": "hawking.frontier_proof_pack.v1",
                "ok": true,
                "model_count": 2,
                "blocked_claim_count": 2,
                "evidence_rows": [{}, {}, {}],
                "claim_bundles": [
                    {"label": "a", "claim_admissible": false, "blocker_count": 7, "written": true},
                    {"label": "b", "claim_admissible": false, "blocker_count": 8, "written": true}
                ]
            }))
        });
        let s = proof_pack_summary(&artifact);
        assert_eq!(s["ok"], json!(true));
        assert_eq!(s["signature_ok"], json!(true));
        assert_eq!(s["model_count"], json!(2));
        assert_eq!(s["blocked_claim_count"], json!(2));
        assert_eq!(s["evidence_count"], json!(3));
        assert_eq!(s["claim_bundles"].as_array().unwrap().len(), 2);
    }
}
