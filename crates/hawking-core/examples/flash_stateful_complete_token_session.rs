//! Complete Flash stateful-session attempt.
//!
//! This is the smallest end-to-end session-shaped executor built from the
//! already-qualified linear and full-attention organs.  It deliberately keeps
//! cross-species seams explicit (host vectors) while preserving recurrence in
//! every linear segment and KV state across every full-attention token list.
//! A receipt is emitted for either accepted-candidate success or the first
//! physical boundary; no promotion or performance claim is implied.

#![recursion_limit = "512"]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash stateful session requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod full;
#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;
#[cfg(target_os = "macos")]
#[path = "flash_source_bf16_terminal.rs"]
mod terminal;

#[cfg(target_os = "macos")]
mod macos {
    use super::{full, linear, terminal};
    use hawking_core::flash_boundary_compare::{
        validate_ordered_semantic_seams, DiagnosticSemanticSeam,
    };
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::error::Error;
    use std::fmt;
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::MetadataExt;
    use std::path::{Component, Path, PathBuf};
    use std::process::Command;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const DEFAULT_ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const MODEL_LAKE_MANIFEST: &str =
        "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const SOURCE_INPUT_IDENTITY_SCHEMA: &str = "hawking.flash.source_input_identity.v1";
    const SOURCE_CONFIG_FILE: &str = "config.json";
    const SOURCE_INDEX_FILE: &str = "model.safetensors.index.json";
    const SOURCE_TOKENIZER_FILE: &str = "tokenizer.json";
    const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const HC: usize = HIDDEN * STREAMS;
    const HC_LOWRANK: usize = 320;
    const ROUTED_EXPERT_TOP_K: usize = 10;
    const ROUTED_EXPERT_INTERMEDIATE: usize = 640;
    const ROUTED_EXPERT_ROW_BYTES: u64 = ((2 * ROUTED_EXPERT_INTERMEDIATE * HIDDEN
        + HIDDEN * ROUTED_EXPERT_INTERMEDIATE)
        * 2) as u64;
    const Q4_GROUP_SIZE: usize = 64;
    const Q4_CODE_BYTES_PER_GROUP: usize = Q4_GROUP_SIZE / 2;
    const Q4_SCALE_BYTES_PER_GROUP: usize = 2;
    const Q4_ROUTED_EXPERT_ROW_BYTES: u64 =
        (2 * ROUTED_EXPERT_INTERMEDIATE
            * (HIDDEN / Q4_GROUP_SIZE)
            * (Q4_CODE_BYTES_PER_GROUP + Q4_SCALE_BYTES_PER_GROUP)
            + HIDDEN
                * (ROUTED_EXPERT_INTERMEDIATE / Q4_GROUP_SIZE)
                * (Q4_CODE_BYTES_PER_GROUP + Q4_SCALE_BYTES_PER_GROUP)) as u64;
    const Q8_GROUP_SIZE: usize = 32;
    const Q8_CODE_BYTES_PER_GROUP: usize = Q8_GROUP_SIZE;
    const Q8_SCALE_BYTES_PER_GROUP: usize = 2;
    const Q8_ROUTED_EXPERT_ROW_BYTES: u64 =
        (2 * ROUTED_EXPERT_INTERMEDIATE
            * (HIDDEN / Q8_GROUP_SIZE)
            * (Q8_CODE_BYTES_PER_GROUP + Q8_SCALE_BYTES_PER_GROUP)
            + HIDDEN
                * (ROUTED_EXPERT_INTERMEDIATE / Q8_GROUP_SIZE)
                * (Q8_CODE_BYTES_PER_GROUP + Q8_SCALE_BYTES_PER_GROUP)) as u64;
    const Q8_RESIDUAL_INDEX_BYTES: u64 = std::mem::size_of::<u16>() as u64;
    const Q8_RESIDUAL_VALUE_BYTES: u64 = std::mem::size_of::<f32>() as u64;
    const Q8_RESIDUAL_ROW_POINTER_BYTES: u64 = std::mem::size_of::<u32>() as u64;
    const Q8_DEFAULT_RESIDUAL_FRACTION: f64 = 0.02;
    const HYPERCONNECTION_WEIGHT_BYTES_PER_LAYER: u64 =
        (2 * (2 * (HC + HC_LOWRANK * HC + HC * HC_LOWRANK + STREAMS * HC))) as u64;
    const FULL_LAYERS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];
    const PROMPT_IDS: [usize; 5] = [5423, 799, 4581, 3817, 13];
    const DEFAULT_CANDIDATE: usize = 17;
    const SESSION_RECEIPT_SCHEMA: &str = "hawking.flash.stateful_complete_token_session.v1";
    const REPEATED_ACCEPTED_STATUS: &str = "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE";
    const CANDIDATE_ROUTE_OBSERVATION_STATUS: &str =
        "CANDIDATE_ROUTE_EXACT_TERMINAL_ACCEPTED_STATE_HASH_WITHHELD";
    const TOKEN_MAJOR_RESIDENT_STATUS: &str =
        "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE";
    const PINNED_VOCAB_SIZE: usize = 248_320;

    fn q8_residual_fraction() -> f64 {
        std::env::var("HAWKING_FLASH_Q8_RESIDUAL_FRACTION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .unwrap_or(Q8_DEFAULT_RESIDUAL_FRACTION)
    }

    fn q8_residual_entries(columns: usize) -> u64 {
        ((columns as f64 * q8_residual_fraction()).ceil() as u64).min(columns as u64)
    }

    fn q8_routed_expert_row_bytes(residual: bool) -> u64 {
        if !residual {
            return Q8_ROUTED_EXPERT_ROW_BYTES;
        }
        let gate_up_entries = q8_residual_entries(HIDDEN);
        let down_entries = q8_residual_entries(ROUTED_EXPERT_INTERMEDIATE);
        Q8_ROUTED_EXPERT_ROW_BYTES
            + (2 * ROUTED_EXPERT_INTERMEDIATE as u64 * gate_up_entries
                + HIDDEN as u64 * down_entries)
                * (Q8_RESIDUAL_INDEX_BYTES + Q8_RESIDUAL_VALUE_BYTES)
            + (2 * ROUTED_EXPERT_INTERMEDIATE as u64 + HIDDEN as u64)
                * Q8_RESIDUAL_ROW_POINTER_BYTES
    }

    fn routed_expert_traffic_bytes(q4: bool, q8: bool, q8_residual: bool) -> u64 {
        if q4 {
            ROUTED_EXPERT_TOP_K as u64 * Q4_ROUTED_EXPERT_ROW_BYTES
        } else if q8 {
            ROUTED_EXPERT_TOP_K as u64 * q8_routed_expert_row_bytes(q8_residual)
        } else {
            ROUTED_EXPERT_TOP_K as u64 * ROUTED_EXPERT_ROW_BYTES
        }
    }

    fn routed_expert_traffic_reduction(q4: bool, q8: bool, q8_residual: bool) -> f64 {
        let row_bytes = if q4 {
            Q4_ROUTED_EXPERT_ROW_BYTES
        } else if q8 {
            q8_routed_expert_row_bytes(q8_residual)
        } else {
            ROUTED_EXPERT_ROW_BYTES
        };
        1.0 - row_bytes as f64 / ROUTED_EXPERT_ROW_BYTES as f64
    }
    const PROTECTED_NATIVE_LANE_MARKERS: [&str; 10] = [
        "flash_stateful_complete_token_session",
        "flash_repeated_accepted_decode.py",
        "flash_route_union_control.py",
        "flash_source_boundary_capture_mlx.py",
        "flash_source_boundary_mlx_worker.py",
        "flash_source_boundary_native.py",
        "flash_source_boundary_transaction.py",
        "flash_noetic_complete_layer0",
        "mlx_vlm.server",
        "hawkingd",
    ];
    const SUPERVISING_LAUNCHER_MARKERS: [&str; 4] = [
        "flash_repeated_accepted_decode.py",
        "flash_route_union_control.py",
        "flash_source_boundary_native.py",
        "flash_source_boundary_transaction.py",
    ];

    /// The native executable cannot authenticate a wrapper or turn this
    /// point-in-time check into a lease. It does require a caller to declare
    /// its admission mode, then refuses an unqualified direct control before
    /// any source, index, or Metal work can begin.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum NativeInvocationAdmission {
        WrapperAdmittedSourceControl,
        UnqualifiedDirectControl,
    }

    impl NativeInvocationAdmission {
        fn receipt_value(self) -> Value {
            match self {
                Self::WrapperAdmittedSourceControl => json!({
                    "policy_marker": "--wrapper-admitted-source-control",
                    "status": "WRAPPER_ASSERTED_SOURCE_CONTROL",
                    "claim_boundary": "The caller asserted that a wrapper performed its external source/lane checks before launch. Native code does not authenticate that assertion, grant a lease, or establish source-independent execution.",
                }),
                Self::UnqualifiedDirectControl => json!({
                    "policy_marker": "--unqualified-direct-control",
                    "status": "UNQUALIFIED_DIRECT_DEVELOPER_CONTROL",
                    "claim_boundary": "This is an explicitly unqualified direct developer control. It is not wrapper-admitted, source-independent, lease-backed, capability, TPS, EBPW, or promotion evidence.",
                }),
            }
        }
    }

    /// The two source-bound native capture scopes deliberately remain separate.
    /// A full layer-0 through layer-4 capture requires the admitted source PLE
    /// state; the layer-0 trace-only scope does not touch PLE or layers 1–4.
    /// Keeping the distinction in Rust prevents a fast causal localizer from
    /// being mistaken for a whole-boundary capture.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum NativeBoundaryCaptureKind {
        FullSourcePleBoundary,
        Layer0AttentionHcTraceOnly,
    }

    impl NativeBoundaryCaptureKind {
        fn parse(argv: &[String]) -> Result<Option<Self>, Box<dyn Error>> {
            let full = argv
                .iter()
                .filter(|argument| argument.as_str() == "--capture-source-boundary")
                .count();
            let layer0_trace = argv
                .iter()
                .filter(|argument| argument.as_str() == "--capture-layer0-hc-trace-only")
                .count();
            if full > 1 {
                return Err("--capture-source-boundary may appear at most once".into());
            }
            if layer0_trace > 1 {
                return Err("--capture-layer0-hc-trace-only may appear at most once".into());
            }
            match (full, layer0_trace) {
                (0, 0) => Ok(None),
                (1, 0) => Ok(Some(Self::FullSourcePleBoundary)),
                (0, 1) => Ok(Some(Self::Layer0AttentionHcTraceOnly)),
                (1, 1) => Err(
                    "--capture-source-boundary and --capture-layer0-hc-trace-only are mutually exclusive"
                        .into(),
                ),
                _ => unreachable!("duplicate capture flags were rejected above"),
            }
        }

        fn requires_source_ple_state(self) -> bool {
            matches!(self, Self::FullSourcePleBoundary)
        }

        fn flag(self) -> &'static str {
            match self {
                Self::FullSourcePleBoundary => "--capture-source-boundary",
                Self::Layer0AttentionHcTraceOnly => "--capture-layer0-hc-trace-only",
            }
        }
    }

    fn require_native_invocation_admission(
        argv: &[String],
    ) -> Result<NativeInvocationAdmission, Box<dyn Error>> {
        let wrapper_admitted = argv
            .iter()
            .any(|argument| argument == "--wrapper-admitted-source-control");
        let unqualified_direct = argv
            .iter()
            .any(|argument| argument == "--unqualified-direct-control");
        match (wrapper_admitted, unqualified_direct) {
            (true, false) => Ok(NativeInvocationAdmission::WrapperAdmittedSourceControl),
            (false, true) => Ok(NativeInvocationAdmission::UnqualifiedDirectControl),
            (false, false) => Err(
                "direct native execution requires exactly one policy marker: --wrapper-admitted-source-control or --unqualified-direct-control"
                    .into(),
            ),
            (true, true) => Err(
                "--wrapper-admitted-source-control and --unqualified-direct-control are mutually exclusive"
                    .into(),
            ),
        }
    }

    /// A policy marker is not itself authority to launch a native model path.
    /// The explicit unqualified marker is retained so direct attempts fail
    /// clearly and before any source/index/Metal access; only the wrapper mode
    /// is eligible to continue to the observed launcher/lane checks.
    fn require_wrapper_admitted_native_execution(
        admission: NativeInvocationAdmission,
    ) -> Result<(), Box<dyn Error>> {
        match admission {
            NativeInvocationAdmission::WrapperAdmittedSourceControl => Ok(()),
            NativeInvocationAdmission::UnqualifiedDirectControl => Err(
                "--unqualified-direct-control is refused before source/index/Metal setup; native execution requires --wrapper-admitted-source-control from an admitted wrapper"
                    .into(),
            ),
        }
    }

    fn supervising_launcher_pid(
        argv: &[String],
        admission: NativeInvocationAdmission,
    ) -> Result<Option<u32>, Box<dyn Error>> {
        let positions = argv
            .iter()
            .enumerate()
            .filter_map(|(index, argument)| {
                (argument == "--supervising-launcher-pid").then_some(index)
            })
            .collect::<Vec<_>>();
        if positions.is_empty() {
            return match admission {
                NativeInvocationAdmission::WrapperAdmittedSourceControl => Err(
                    "--wrapper-admitted-source-control requires --supervising-launcher-pid <PID>"
                        .into(),
                ),
                NativeInvocationAdmission::UnqualifiedDirectControl => Ok(None),
            };
        }
        if positions.len() != 1 {
            return Err("--supervising-launcher-pid may appear exactly once".into());
        }
        if admission != NativeInvocationAdmission::WrapperAdmittedSourceControl {
            return Err(
                "--supervising-launcher-pid is allowed only with --wrapper-admitted-source-control"
                    .into(),
            );
        }
        let raw_pid = argv
            .get(positions[0] + 1)
            .ok_or("--supervising-launcher-pid requires a numeric PID")?;
        let pid = raw_pid
            .parse::<u32>()
            .map_err(|_| "--supervising-launcher-pid requires a numeric PID")?;
        if pid == 0 {
            return Err("--supervising-launcher-pid must be a positive PID".into());
        }
        Ok(Some(pid))
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct ProtectedNativeLaneMatch {
        pid: u32,
        state: String,
        marker: &'static str,
    }

    struct ProtectedNativeLaneProcess {
        pid: u32,
        parent_pid: u32,
        state: String,
        command: String,
    }

    /// Match only a process's executable or its Python entrypoint.  An outer
    /// lease shell and a wrapper interpreter necessarily retain the native
    /// binary path in their argument vectors; treating arbitrary arguments as
    /// owners would make a correctly admitted native child refuse itself.
    fn command_owns_protected_native_marker(command: &str, marker: &str) -> bool {
        let mut fields = command.split_whitespace();
        let Some(executable) = fields.next() else {
            return false;
        };
        let Some(executable_name) = Path::new(executable)
            .file_name()
            .and_then(|name| name.to_str())
        else {
            return false;
        };
        match marker {
            "flash_stateful_complete_token_session"
            | "flash_noetic_complete_layer0"
            | "hawkingd" => executable_name == marker,
            "mlx_vlm.server" => {
                if executable_name == marker {
                    return true;
                }
                if !executable_name.to_ascii_lowercase().starts_with("python") {
                    return false;
                }
                let Some(entrypoint_flag) = fields.next() else {
                    return false;
                };
                entrypoint_flag == "-m" && fields.next() == Some(marker)
            }
            _ if marker.ends_with(".py") => {
                if !executable_name.to_ascii_lowercase().starts_with("python") {
                    return false;
                }
                while let Some(argument) = fields.next() {
                    if argument == "-m" {
                        return false;
                    }
                    if !argument.starts_with('-') {
                        return Path::new(argument)
                            .file_name()
                            .and_then(|name| name.to_str())
                            == Some(marker);
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn is_recognized_supervising_launcher(process: &ProtectedNativeLaneProcess) -> bool {
        SUPERVISING_LAUNCHER_MARKERS
            .iter()
            .any(|marker| command_owns_protected_native_marker(&process.command, marker))
    }

    fn protected_native_lane_matches(
        snapshot: &str,
        own_pid: u32,
        supervising_launcher_pid: Option<u32>,
    ) -> Result<Vec<ProtectedNativeLaneMatch>, Box<dyn Error>> {
        let mut processes = Vec::new();
        for row in snapshot.lines() {
            let row = row.trim();
            if row.is_empty() {
                continue;
            }
            let mut fields = row.split_whitespace();
            let raw_pid = fields
                .next()
                .ok_or("protected native-lane ps row omitted its PID")?;
            let pid = raw_pid.parse::<u32>().map_err(|_| {
                format!("protected native-lane ps row has a nonnumeric PID: {raw_pid}")
            })?;
            let raw_parent_pid = fields
                .next()
                .ok_or("protected native-lane ps row omitted its parent PID")?;
            let parent_pid = raw_parent_pid.parse::<u32>().map_err(|_| {
                format!(
                    "protected native-lane ps row has a nonnumeric parent PID: {raw_parent_pid}"
                )
            })?;
            let state = fields
                .next()
                .ok_or("protected native-lane ps row omitted its state")?;
            let command = fields.collect::<Vec<_>>().join(" ");
            if command.is_empty() {
                return Err("protected native-lane ps row omitted its command".into());
            }
            processes.push(ProtectedNativeLaneProcess {
                pid,
                parent_pid,
                state: state.to_owned(),
                command,
            });
        }
        let mut admitted_wrapper_pids = Vec::new();
        if let Some(supervising_pid) = supervising_launcher_pid {
            let supervisor = processes
                .iter()
                .find(|process| process.pid == supervising_pid)
                .ok_or(
                    "--supervising-launcher-pid was not visible in the protected native-lane ps snapshot",
                )?;
            if supervisor.state.starts_with('Z') {
                return Err(
                    "--supervising-launcher-pid must name a visible non-zombie wrapper process"
                        .into(),
                );
            }
            if !is_recognized_supervising_launcher(supervisor) {
                return Err(
                    "--supervising-launcher-pid did not name a recognized wrapper process".into(),
                );
            }
            let native_process = processes
                .iter()
                .find(|process| process.pid == own_pid)
                .ok_or("native process was not visible in the protected native-lane ps snapshot")?;
            if native_process.parent_pid != supervising_pid {
                return Err(
                    "--supervising-launcher-pid must name the immediate native launcher observed by ps"
                    .into(),
                );
            }
            admitted_wrapper_pids.push(supervisor.pid);

            // The public transaction is the direct ancestor of its leased
            // child. Exempt only recognized wrapper ancestors in that exact
            // lineage; a same-named process elsewhere remains an owner that
            // blocks the native lane.
            let mut visited_pids = vec![own_pid, supervisor.pid];
            let mut ancestor_pid = supervisor.parent_pid;
            while ancestor_pid != 0 && !visited_pids.contains(&ancestor_pid) {
                visited_pids.push(ancestor_pid);
                let Some(ancestor) = processes.iter().find(|process| process.pid == ancestor_pid)
                else {
                    break;
                };
                if !ancestor.state.starts_with('Z') && is_recognized_supervising_launcher(ancestor)
                {
                    admitted_wrapper_pids.push(ancestor.pid);
                }
                if ancestor.parent_pid == ancestor.pid {
                    break;
                }
                ancestor_pid = ancestor.parent_pid;
            }
        }
        let mut matches = Vec::new();
        for process in processes {
            if process.pid == own_pid || admitted_wrapper_pids.contains(&process.pid) {
                continue;
            }
            if process.state.starts_with('Z') {
                continue;
            }
            if let Some(marker) = PROTECTED_NATIVE_LANE_MARKERS
                .iter()
                .copied()
                .find(|marker| command_owns_protected_native_marker(&process.command, marker))
            {
                matches.push(ProtectedNativeLaneMatch {
                    pid: process.pid,
                    state: process.state,
                    marker,
                });
            }
        }
        Ok(matches)
    }

    /// A conservative observation only.  It deliberately refuses a visible
    /// native/MLX/Hawking owner but cannot reserve the lane against a process
    /// that starts after this snapshot.
    fn require_clean_protected_native_lane(
        supervising_launcher_pid: Option<u32>,
    ) -> Result<Value, Box<dyn Error>> {
        let completed = Command::new("ps")
            .args(["-axo", "pid=,ppid=,stat=,command="])
            .output()?;
        if !completed.status.success() {
            return Err(format!(
                "protected native-lane ps preflight failed with status {}",
                completed.status
            )
            .into());
        }
        let snapshot = std::str::from_utf8(&completed.stdout)
            .map_err(|_| "protected native-lane ps preflight emitted non-UTF-8 output")?;
        let matches =
            protected_native_lane_matches(snapshot, std::process::id(), supervising_launcher_pid)?;
        if !matches.is_empty() {
            let visible = matches
                .iter()
                .map(|entry| format!("{}:{}:{}", entry.pid, entry.state, entry.marker))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "protected native lane is occupied; refusing launch until visible non-zombie owners exit: {visible}"
            )
            .into());
        }
        Ok(json!({
            "clean": true,
            "matches": [],
            "excluded_supervising_launcher_pid": supervising_launcher_pid,
            "observation": "point-in-time ps preflight only; it is neither a lock, a lease, nor a timing/exclusivity witness",
            "checked_markers": PROTECTED_NATIVE_LANE_MARKERS,
        }))
    }

    fn sha256(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    fn arg_value(args: &[String], flag: &str) -> Option<String> {
        args.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].clone())
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn source_regular_file(root: &Path, file: &str) -> Result<PathBuf, Box<dyn Error>> {
        let relative = Path::new(file);
        if relative.is_absolute() || relative.components().count() != 1 {
            return Err(format!("source input {file} is not a direct source-root file").into());
        }
        let candidate = root.join(relative);
        let metadata = fs::symlink_metadata(&candidate)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(format!("source input {file} is not a regular source-root file").into());
        }
        let resolved = candidate.canonicalize()?;
        if resolved.parent() != Some(root) {
            return Err(format!("source input {file} escapes the selected source root").into());
        }
        Ok(resolved)
    }

    fn source_file_binding(root: &Path, file: &str) -> Result<Value, Box<dyn Error>> {
        let path = source_regular_file(root, file)?;
        let bytes = fs::read(&path)?;
        Ok(json!({
            "file": file,
            "sha256": sha256(&bytes),
            "bytes": bytes.len(),
        }))
    }

    /// Read only the small immutable source-identity inputs used to select and
    /// interpret the source tensor body.  This proves the selected root is the
    /// configured ModelLake specimen without rehashing its 360 GB shard set or
    /// starting a model/Metal process.  Exact shard evidence remains the
    /// responsibility of the separately admitted source-shard ledger.
    fn source_input_identity(root: &Path) -> Result<Value, Box<dyn Error>> {
        let expected_root = PathBuf::from(DEFAULT_ROOT).canonicalize()?;
        if root != expected_root {
            return Err(
                "selected source root is not the canonical ModelLake Flash specimen".into(),
            );
        }
        let manifest_candidate = PathBuf::from(MODEL_LAKE_MANIFEST);
        let manifest_metadata = fs::symlink_metadata(&manifest_candidate)?;
        if !manifest_metadata.file_type().is_file() || manifest_metadata.file_type().is_symlink() {
            return Err("canonical ModelLake manifest is not a regular file".into());
        }
        let manifest_path = manifest_candidate.canonicalize()?;
        let manifest_bytes = fs::read(&manifest_path)?;
        let manifest =
            parse_json_no_duplicate_keys(&manifest_bytes, "canonical ModelLake manifest")?;
        let manifest_root = manifest
            .get("path")
            .and_then(Value::as_str)
            .ok_or("canonical ModelLake manifest omitted its specimen path")?;
        if manifest.get("repo").and_then(Value::as_str) != Some(REPO_ID)
            || manifest.get("revision").and_then(Value::as_str) != Some(PINNED_REVISION)
            || manifest.get("resolved_sha").and_then(Value::as_str) != Some(PINNED_REVISION)
            || PathBuf::from(manifest_root).canonicalize()? != *root
        {
            return Err(
                "canonical ModelLake manifest does not bind the selected pinned Flash specimen"
                    .into(),
            );
        }
        Ok(json!({
            "schema": SOURCE_INPUT_IDENTITY_SCHEMA,
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "model_root": root,
            "model_lake_manifest": {
                "path": manifest_path,
                "sha256": sha256(&manifest_bytes),
                "repo": REPO_ID,
                "revision": PINNED_REVISION,
            },
            "config": source_file_binding(root, SOURCE_CONFIG_FILE)?,
            "safetensors_index": source_file_binding(root, SOURCE_INDEX_FILE)?,
            "tokenizer": source_file_binding(root, SOURCE_TOKENIZER_FILE)?,
        }))
    }

    fn path_entry_exists(path: &Path) -> Result<bool, Box<dyn Error>> {
        match fs::symlink_metadata(path) {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// Normalize a user-selected output lexically without requiring the leaf
    /// to exist.  Reservation deliberately targets fresh paths, so the leaf
    /// cannot be canonicalized yet.
    fn absolute_lexical_path(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            env::current_dir()?.join(path)
        };
        let mut normalized = PathBuf::new();
        for component in absolute.components() {
            match component {
                Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
                Component::RootDir => normalized.push(component.as_os_str()),
                Component::CurDir => {}
                Component::ParentDir => {
                    if normalized.file_name().is_some() {
                        normalized.pop();
                    }
                }
                Component::Normal(part) => normalized.push(part),
            }
        }
        if !normalized.is_absolute() {
            return Err("output path did not normalize to an absolute path".into());
        }
        Ok(normalized)
    }

    fn paths_overlap(left: &Path, right: &Path) -> bool {
        left.starts_with(right) || right.starts_with(left)
    }

    /// Resolve the longest existing ancestor, then append the non-existing
    /// suffix.  This catches a fresh PLE path hidden under a symlinked parent
    /// of `--out` or its artifact directory without pretending a fresh leaf
    /// itself can be canonicalized.
    fn resolve_existing_ancestor_for_overlap(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
        let mut current = path.to_path_buf();
        let mut missing_suffix = Vec::<PathBuf>::new();
        loop {
            match fs::symlink_metadata(&current) {
                Ok(_) => {
                    let target_metadata = fs::metadata(&current)?;
                    if !missing_suffix.is_empty() && !target_metadata.is_dir() {
                        return Err(format!(
                            "output path ancestor is not a directory: {}",
                            current.display()
                        )
                        .into());
                    }
                    let mut resolved = current.canonicalize()?;
                    for component in missing_suffix.iter().rev() {
                        resolved.push(component);
                    }
                    return Ok(resolved);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let component = current
                        .file_name()
                        .ok_or("fresh output path has no existing ancestor")?;
                    missing_suffix.push(PathBuf::from(component));
                    current = current
                        .parent()
                        .ok_or("fresh output path has no parent")?
                        .to_path_buf();
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn read_stable_regular_file(path: &Path, label: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        let before = fs::symlink_metadata(path)
            .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
        if before.file_type().is_symlink() || !before.is_file() {
            return Err(format!(
                "{label} must be a regular non-symlink file: {}",
                path.display()
            )
            .into());
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
        let after = fs::symlink_metadata(path)
            .map_err(|error| format!("cannot re-stat {label} {}: {error}", path.display()))?;
        if after.file_type().is_symlink()
            || !after.is_file()
            || before.len() != after.len()
            || u64::try_from(bytes.len()).ok() != Some(before.len())
        {
            return Err(format!("{label} changed while being read: {}", path.display()).into());
        }
        Ok(bytes)
    }

    fn session_artifact_dir(out: &Path) -> PathBuf {
        let out_stem = out
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("FLASH_STATEFUL_COMPLETE_TOKEN_SESSION");
        out.parent()
            .unwrap_or_else(|| Path::new("receipts/headless"))
            .join(format!("{out_stem}.session_artifacts"))
    }

    fn validate_ple_pre_layer_state_bank_output(
        output_dir: &Path,
        out: &Path,
        out_dir: &Path,
    ) -> Result<(), Box<dyn Error>> {
        if path_entry_exists(output_dir)? {
            return Err(format!(
                "--ple-pre-layer-state-bank-out refuses an existing output directory or path: {}",
                output_dir.display()
            )
            .into());
        }
        let resolved_output_dir = resolve_existing_ancestor_for_overlap(output_dir)?;
        let resolved_out = resolve_existing_ancestor_for_overlap(out)?;
        let resolved_out_dir = resolve_existing_ancestor_for_overlap(out_dir)?;
        if paths_overlap(output_dir, out)
            || paths_overlap(output_dir, out_dir)
            || paths_overlap(&resolved_output_dir, &resolved_out)
            || paths_overlap(&resolved_output_dir, &resolved_out_dir)
        {
            return Err(
                "--ple-pre-layer-state-bank-out must be a fresh directory disjoint from --out and its session-artifact directory"
                    .into(),
            );
        }
        Ok(())
    }

    /// Reserve the receipt's session-local artifact namespace after no-model
    /// argument/teacher validation and before constructing a model.  The
    /// directory creation is the ownership marker for the whole native
    /// attempt; the final receipt uses create_new as a second no-clobber guard.
    fn reserve_session_output(out: &Path) -> Result<PathBuf, Box<dyn Error>> {
        let out_dir = session_artifact_dir(out);
        if path_entry_exists(out)? {
            return Err(format!(
                "--out refuses to overwrite an existing receipt or path: {}",
                out.display()
            )
            .into());
        }
        if path_entry_exists(&out_dir)? {
            return Err(format!(
                "--out refuses to reuse an existing session-artifact directory: {}",
                out_dir.display()
            )
            .into());
        }
        let parent = out_dir
            .parent()
            .ok_or("derived session-artifact directory has no parent")?;
        fs::create_dir_all(parent)?;
        if path_entry_exists(out)? {
            return Err(format!(
                "--out became occupied while reserving this native attempt: {}",
                out.display()
            )
            .into());
        }
        match fs::create_dir(&out_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(format!(
                    "--out session-artifact directory became occupied while reserving this native attempt: {}",
                    out_dir.display()
                )
                .into())
            }
            Err(error) => return Err(error.into()),
        }
        if path_entry_exists(out)? {
            return Err(format!(
                "--out became occupied after session-artifact reservation: {}",
                out.display()
            )
            .into());
        }
        Ok(out_dir)
    }

    fn write_new_receipt(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        let mut receipt = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        receipt.write_all(bytes)?;
        Ok(())
    }

    fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(bytes)?;
        Ok(())
    }

    fn parse_token_ids(argv: &[String]) -> Result<Vec<usize>, Box<dyn Error>> {
        if let Some(raw) = arg_value(argv, "--token-ids") {
            let ids = raw
                .split(',')
                .map(|part| part.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?;
            if ids.is_empty() {
                return Err("--token-ids must not be empty".into());
            }
            return Ok(ids);
        }
        let mut ids = PROMPT_IDS.to_vec();
        ids.push(
            arg_value(argv, "--candidate")
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(DEFAULT_CANDIDATE),
        );
        Ok(ids)
    }

    /// Separate prompt from caller-supplied reference continuation tokens.
    ///
    /// Every continuation token is checked against the preceding terminal
    /// argmax.  The default preserves the historical one-candidate interface;
    /// a multi-token reference must name the boundary explicitly so no prompt
    /// token is silently counted as generated output.
    fn parse_prompt_len(argv: &[String], token_count: usize) -> Result<usize, Box<dyn Error>> {
        let prompt_len = arg_value(argv, "--prompt-length")
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(token_count.saturating_sub(1));
        if prompt_len == 0 || prompt_len >= token_count {
            return Err(format!(
                "--prompt-length must be in 1..{} for {token_count} token ids",
                token_count.saturating_sub(1)
            )
            .into());
        }
        Ok(prompt_len)
    }

    /// Exact route teacher for one bounded dense session.  This deliberately
    /// records every `(layer, token-slot)` top-k row as well as the per-layer
    /// union.  The compact candidate may use the union for storage, but must
    /// reproduce each individual teacher row before its terminal checks count.
    struct RouteTeacher {
        routes: BTreeMap<(usize, usize), Vec<u32>>,
        final_state_sha256: BTreeMap<(usize, usize), String>,
        unions: BTreeMap<usize, Vec<u32>>,
    }

    /// `serde_json::Value` normally retains only the last duplicate map key.
    /// A route teacher is an authority input, so preserve the native receipt
    /// writer's unambiguous-object property rather than allowing a raw JSON
    /// ambiguity to be hidden by its compact seal over the collapsed value.
    #[derive(Debug)]
    struct NoDuplicateJson(Value);

    impl<'de> Deserialize<'de> for NoDuplicateJson {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserializer.deserialize_any(NoDuplicateJsonVisitor)
        }
    }

    struct NoDuplicateJsonVisitor;

    impl<'de> Visitor<'de> for NoDuplicateJsonVisitor {
        type Value = NoDuplicateJson;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a JSON value without duplicate object keys")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::Bool(value)))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::Number(value.into())))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::Number(value.into())))
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            serde_json::Number::from_f64(value)
                .map(|number| NoDuplicateJson(Value::Number(number)))
                .ok_or_else(|| E::custom("JSON number must be finite"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::String(value.to_owned())))
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::String(value)))
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::Null))
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(NoDuplicateJson(Value::Null))
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            NoDuplicateJson::deserialize(deserializer)
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut values = Vec::new();
            while let Some(NoDuplicateJson(value)) = sequence.next_element()? {
                values.push(value);
            }
            Ok(NoDuplicateJson(Value::Array(values)))
        }

        fn visit_map<A>(self, mut map_access: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut values = serde_json::Map::new();
            while let Some(key) = map_access.next_key::<String>()? {
                if values.contains_key(&key) {
                    return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
                }
                let NoDuplicateJson(value) = map_access.next_value()?;
                values.insert(key, value);
            }
            Ok(NoDuplicateJson(Value::Object(values)))
        }
    }

    fn parse_json_no_duplicate_keys(raw: &[u8], label: &str) -> Result<Value, Box<dyn Error>> {
        let mut deserializer = serde_json::Deserializer::from_slice(raw);
        let NoDuplicateJson(value) =
            NoDuplicateJson::deserialize(&mut deserializer).map_err(|error| {
                format!("{label} is not valid JSON without duplicate keys: {error}")
            })?;
        deserializer
            .end()
            .map_err(|error| format!("{label} has trailing JSON data: {error}"))?;
        Ok(value)
    }

    /// Native session receipts are written pretty, but seal the parsed body
    /// with serde_json's compact UTF-8 form before the seal field is added.
    /// Keep this local verifier exact rather than treating a present string as
    /// provenance: route teachers drive which immutable expert rows are read.
    fn is_lowercase_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    }

    fn verify_native_compact_receipt_seal(doc: &Value) -> Result<(), Box<dyn Error>> {
        let object = doc
            .as_object()
            .ok_or("route teacher receipt root is not an object")?;
        let seal = object
            .get("seal_sha256")
            .and_then(Value::as_str)
            .filter(|seal| is_lowercase_sha256(seal))
            .ok_or("route teacher omitted a valid native compact UTF-8 seal")?;
        let mut body = doc.clone();
        body.as_object_mut()
            .ok_or("route teacher receipt root is not an object")?
            .remove("seal_sha256")
            .ok_or("route teacher omitted a native compact UTF-8 seal")?;
        let expected = sha256(&serde_json::to_vec(&body)?);
        if seal != expected {
            return Err("route teacher has an invalid native compact UTF-8 seal".into());
        }
        Ok(())
    }

    fn receipt_index(value: &Value, field: &str) -> Result<usize, Box<dyn Error>> {
        let value = value
            .as_u64()
            .ok_or_else(|| format!("route teacher {field} is not a non-negative integer"))?;
        if value > usize::MAX as u64 {
            return Err(
                format!("route teacher {field} exceeds this platform's index range").into(),
            );
        }
        Ok(value as usize)
    }

    fn receipt_token_id(value: &Value, field: &str) -> Result<usize, Box<dyn Error>> {
        let token = receipt_index(value, field)?;
        if token >= PINNED_VOCAB_SIZE {
            return Err(
                format!("route teacher {field} is outside the pinned Flash vocabulary").into(),
            );
        }
        Ok(token)
    }

    fn receipt_token_array(doc: &Value, field: &str) -> Result<Vec<usize>, Box<dyn Error>> {
        doc.get(field)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("route teacher omitted {field}"))?
            .iter()
            .enumerate()
            .map(|(index, value)| receipt_token_id(value, &format!("{field}[{index}]")))
            .collect()
    }

    fn segment_layer(segment: &serde_json::Map<String, Value>) -> Result<usize, Box<dyn Error>> {
        let direct = segment
            .get("layer")
            .map(|value| receipt_index(value, "segment layer"))
            .transpose()?;
        let ranged = match segment.get("layers") {
            None => None,
            Some(Value::Array(layers)) if layers.len() == 2 => {
                let start = receipt_index(&layers[0], "segment layers[0]")?;
                let end = receipt_index(&layers[1], "segment layers[1]")?;
                if start != end {
                    return Err("route teacher segment is not a single token-major layer".into());
                }
                Some(start)
            }
            Some(_) => return Err("route teacher segment has malformed layer bounds".into()),
        };
        let layer = match (direct, ranged) {
            (Some(direct), Some(ranged)) if direct != ranged => {
                return Err("route teacher segment layer forms disagree".into())
            }
            (Some(layer), _) | (_, Some(layer)) => layer,
            (None, None) => return Err("route teacher segment omitted its layer".into()),
        };
        if layer >= 48 {
            return Err("route teacher segment has invalid layer".into());
        }
        Ok(layer)
    }

    fn validate_accepted_terminal_chain(
        doc: &Value,
        token_ids: &[usize],
        prompt_len: usize,
    ) -> Result<(), Box<dyn Error>> {
        if prompt_len == 0 || prompt_len >= token_ids.len() || token_ids.len() - prompt_len < 2 {
            return Err(
                "route teacher requires a bounded prompt followed by at least two continuation tokens"
                    .into(),
            );
        }
        let prompt_tokens = receipt_token_array(doc, "prompt_token_ids")?;
        let reference_tokens = receipt_token_array(doc, "reference_generated_token_ids")?;
        if prompt_tokens != token_ids[..prompt_len] || reference_tokens != token_ids[prompt_len..] {
            return Err(
                "route teacher prompt/reference token boundary differs from the candidate".into(),
            );
        }
        if doc
            .get("accepted_generation_tokens")
            .and_then(Value::as_u64)
            != Some(reference_tokens.len() as u64)
        {
            return Err("route teacher did not accept every declared continuation token".into());
        }
        if receipt_token_id(
            doc.get("candidate_token_id")
                .ok_or("route teacher omitted its candidate token")?,
            "candidate_token_id",
        )? != reference_tokens[0]
        {
            return Err("route teacher candidate token disagrees with its accepted chain".into());
        }
        let terminal = doc
            .get("terminal")
            .and_then(Value::as_object)
            .ok_or("route teacher omitted terminal evidence")?;
        if terminal.get("candidate_accepted") != Some(&Value::Bool(true)) {
            return Err("route teacher terminal did not accept its continuation chain".into());
        }
        if receipt_token_id(
            terminal
                .get("predicted_candidate")
                .ok_or("route teacher terminal omitted predicted candidate")?,
            "terminal predicted_candidate",
        )? != reference_tokens[0]
        {
            return Err(
                "route teacher terminal summary disagrees with the first continuation".into(),
            );
        }
        let prompt_final = terminal
            .get("prompt_final")
            .and_then(Value::as_object)
            .ok_or("route teacher terminal omitted first reference boundary")?;
        if receipt_token_id(
            prompt_final
                .get("token_id")
                .ok_or("route teacher terminal first reference boundary omitted token")?,
            "terminal prompt_final token_id",
        )? != reference_tokens[0]
        {
            return Err(
                "route teacher terminal first reference boundary disagrees with the chain".into(),
            );
        }
        let checks = terminal
            .get("reference_checks")
            .and_then(Value::as_array)
            .ok_or("route teacher omitted terminal reference checks")?;
        if checks.len() != reference_tokens.len() {
            return Err(
                "route teacher terminal checks do not cover exactly its continuation chain".into(),
            );
        }
        for (generation_index, (expected_token, check)) in
            reference_tokens.iter().zip(checks).enumerate()
        {
            let check = check
                .as_object()
                .ok_or("route teacher terminal reference check is not an object")?;
            if receipt_index(
                check
                    .get("generation_index")
                    .ok_or("route teacher terminal check omitted generation index")?,
                "terminal generation_index",
            )? != generation_index
                || receipt_index(
                    check
                        .get("input_state_index")
                        .ok_or("route teacher terminal check omitted input-state index")?,
                    "terminal input_state_index",
                )? != prompt_len - 1 + generation_index
                || receipt_token_id(
                    check
                        .get("expected_token_id")
                        .ok_or("route teacher terminal check omitted expected token")?,
                    "terminal expected_token_id",
                )? != *expected_token
                || receipt_token_id(
                    check
                        .get("predicted_token_id")
                        .ok_or("route teacher terminal check omitted predicted token")?,
                    "terminal predicted_token_id",
                )? != *expected_token
                || check.get("accepted") != Some(&Value::Bool(true))
            {
                return Err("route teacher terminal reference chain is not exact".into());
            }
        }
        Ok(())
    }

    fn validate_token_major_residency(
        doc: &Value,
        expected_expert_bank_mode: &str,
    ) -> Result<(), Box<dyn Error>> {
        let execution = doc
            .get("execution")
            .and_then(Value::as_object)
            .ok_or("route teacher omitted execution evidence")?;
        if execution.get("process_boundary").and_then(Value::as_str) != Some("one native process")
            || execution.get("source_reset_or_reprefill") != Some(&Value::Bool(false))
        {
            return Err("route teacher lacks one-process persistent-session proof".into());
        }
        if execution.get("expert_bank_mode").and_then(Value::as_str)
            != Some(expected_expert_bank_mode)
        {
            if expected_expert_bank_mode == "dense" {
                return Err("route teacher is not a dense native control".into());
            }
            return Err(format!(
                "route observation has unexpected expert bank mode; expected {expected_expert_bank_mode}"
            )
            .into());
        }
        if execution
            .get("cross_species_activation_handoff")
            .and_then(Value::as_str)
            != Some(
                "device buffers; host snapshots are diagnostic-only and never feed a later layer",
            )
        {
            return Err("route teacher lacks device-only token-major activation evidence".into());
        }
        let token_major = execution
            .get("token_major_resident_banks")
            .and_then(Value::as_object)
            .ok_or("route teacher is segmented or lacks token-major residency evidence")?;
        if token_major.get("requested") != Some(&Value::Bool(true))
            || token_major.get("status").and_then(Value::as_str)
                != Some(TOKEN_MAJOR_RESIDENT_STATUS)
            || token_major.get("resident_layers").and_then(Value::as_u64) != Some(48)
            || token_major.get("expected_layers").and_then(Value::as_u64) != Some(48)
            || token_major
                .get("cross_layer_activation_handoff")
                .and_then(Value::as_str)
                != Some("device buffers")
            || token_major.get("source_reset_or_reprefill") != Some(&Value::Bool(false))
        {
            return Err("route teacher lacks all-48 device-resident token-major evidence".into());
        }
        Ok(())
    }

    fn validate_route_teacher_source_identity(
        doc: &Value,
        root: &Path,
        source_identity: &Value,
    ) -> Result<(), Box<dyn Error>> {
        let root = root
            .to_str()
            .ok_or("selected canonical source root is not valid UTF-8")?;
        if doc.get("root").and_then(Value::as_str) != Some(root) {
            return Err(
                "route teacher root differs from the selected canonical ModelLake source".into(),
            );
        }
        if doc.get("source_input_identity") != Some(source_identity) {
            return Err(
                "route teacher source input identity differs from the selected canonical ModelLake source"
                    .into(),
            );
        }
        Ok(())
    }

    fn collect_route_rows(
        doc: &Value,
        token_ids: &[usize],
        rows: &mut Vec<Value>,
        expected_expert_bank_mode: &str,
    ) -> Result<(), Box<dyn Error>> {
        let segments = doc
            .get("segments")
            .and_then(Value::as_array)
            .ok_or("route teacher omitted token-major segments")?;
        if segments.len() != 48 {
            return Err(
                "route teacher is segmented or does not have exactly 48 token-major layers".into(),
            );
        }
        let mut seen_layers = BTreeSet::new();
        for segment in segments {
            let segment = segment
                .as_object()
                .ok_or("route teacher token-major segment is not an object")?;
            if segment.contains_key("receipt") {
                return Err(
                    "route teacher may not delegate route rows to a segmented receipt".into(),
                );
            }
            if segment.get("expert_bank_mode").and_then(Value::as_str)
                != Some(expected_expert_bank_mode)
            {
                if expected_expert_bank_mode == "dense" {
                    return Err("route teacher token-major segment is not a dense control".into());
                }
                return Err(format!(
                    "route teacher token-major segment has unexpected expert bank mode; expected {expected_expert_bank_mode}"
                )
                .into());
            }
            let layer = segment_layer(segment)?;
            if !seen_layers.insert(layer) {
                return Err("route teacher has duplicate token-major layer coverage".into());
            }
            let segment_rows = segment
                .get("steps")
                .and_then(Value::as_array)
                .ok_or("route teacher token-major segment omitted inline steps")?;
            if segment_rows.len() != token_ids.len() {
                return Err(
                    "route teacher token-major layer omits the selected token chain".into(),
                );
            }
            for (step, (row, &token_id)) in segment_rows.iter().zip(token_ids).enumerate() {
                let row = row
                    .as_object()
                    .ok_or("route teacher token-major row is not an object")?;
                if receipt_index(
                    row.get("layer")
                        .ok_or("route teacher token-major row omitted layer")?,
                    "row layer",
                )? != layer
                    || receipt_index(
                        row.get("step")
                            .ok_or("route teacher token-major row omitted step")?,
                        "row step",
                    )? != step
                    || receipt_token_id(
                        row.get("token_id")
                            .ok_or("route teacher token-major row omitted token")?,
                        "row token_id",
                    )? != token_id
                {
                    return Err(
                        "route teacher token-major row drifts from its selected chain".into(),
                    );
                }
                rows.push(Value::Object(row.clone()));
            }
        }
        if seen_layers.len() != 48 {
            return Err("route teacher does not cover all 48 token-major layers".into());
        }
        Ok(())
    }

    fn parse_route_teacher(
        argv: &[String],
        token_ids: &[usize],
        prompt_len: usize,
        root: &Path,
        source_identity: &Value,
        allow_candidate_route_observation: bool,
    ) -> Result<Option<RouteTeacher>, Box<dyn Error>> {
        let Some(raw_path) = arg_value(argv, "--route-teacher") else {
            return Ok(None);
        };
        let path = PathBuf::from(raw_path);
        let raw = read_stable_regular_file(&path, "route teacher receipt")?;
        std::str::from_utf8(&raw).map_err(|_| "route teacher receipt is not valid UTF-8 JSON")?;
        let doc = parse_json_no_duplicate_keys(&raw, "route teacher receipt")?;
        verify_native_compact_receipt_seal(&doc)?;
        let status = doc.get("status").and_then(Value::as_str);
        let candidate_route_observation = status == Some(CANDIDATE_ROUTE_OBSERVATION_STATUS);
        let accepted_route_teacher = status == Some(REPEATED_ACCEPTED_STATUS);
        if doc.get("schema").and_then(Value::as_str) != Some(SESSION_RECEIPT_SCHEMA)
            || (!accepted_route_teacher
                && !(allow_candidate_route_observation && candidate_route_observation))
            || doc.get("model").and_then(Value::as_str) != Some(REPO_ID)
            || doc.get("pinned_revision").and_then(Value::as_str) != Some(PINNED_REVISION)
            || doc.get("vocab_size").and_then(Value::as_u64) != Some(PINNED_VOCAB_SIZE as u64)
        {
            return Err(
                "route teacher is not the pinned session schema, accepted session, or explicitly enabled candidate route observation".into(),
            );
        }
        if candidate_route_observation {
            if doc
                .pointer("/execution/source_independent")
                .and_then(Value::as_bool)
                != Some(false)
            {
                return Err(
                    "candidate route observation must remain explicitly source-bound".into(),
                );
            }
        } else {
            validate_route_teacher_source_identity(&doc, root, source_identity)?;
        }
        let teacher_tokens = receipt_token_array(&doc, "token_ids")?;
        if teacher_tokens != token_ids {
            return Err("route teacher token sequence differs from compact candidate".into());
        }
        validate_accepted_terminal_chain(&doc, token_ids, prompt_len)?;
        validate_token_major_residency(
            &doc,
            if candidate_route_observation {
                "route_union_compact_teacher_bound"
            } else {
                "dense"
            },
        )?;
        let mut raw_rows = Vec::new();
        collect_route_rows(
            &doc,
            token_ids,
            &mut raw_rows,
            if candidate_route_observation {
                "route_union_compact_teacher_bound"
            } else {
                "dense"
            },
        )?;
        let mut routes = BTreeMap::new();
        let mut final_state_sha256 = BTreeMap::new();
        let mut unions: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
        for row in raw_rows {
            let layer = receipt_index(
                row.get("layer").ok_or("route teacher row missing layer")?,
                "row layer",
            )?;
            let step = receipt_index(
                row.get("step").ok_or("route teacher row missing step")?,
                "row step",
            )?;
            let ids = row
                .get("route_ids")
                .and_then(Value::as_array)
                .ok_or("route teacher row omitted top-k IDs")?
                .iter()
                .map(|value| {
                    let id = value.as_u64().ok_or("non-integer route ID")?;
                    if id >= 512 {
                        return Err("route teacher route ID is outside the expert bank");
                    }
                    Ok(id as u32)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let state_hash = row
                .get("final_state_sha256")
                .and_then(Value::as_str)
                .filter(|value| is_lowercase_sha256(value))
                .ok_or("route teacher row omitted a valid final-state SHA-256")?
                .to_owned();
            if layer >= 48
                || step >= token_ids.len()
                || ids.len() != 10
                || ids.iter().collect::<BTreeSet<_>>().len() != 10
                || routes.insert((layer, step), ids.clone()).is_some()
                || final_state_sha256
                    .insert((layer, step), state_hash)
                    .is_some()
            {
                return Err("route teacher has invalid or duplicate top-k coverage".into());
            }
            unions.entry(layer).or_default().extend(ids);
        }
        let expected_coverage = 48usize
            .checked_mul(token_ids.len())
            .ok_or("route teacher token coverage overflow")?;
        if routes.len() != expected_coverage
            || final_state_sha256.len() != expected_coverage
            || unions.len() != 48
        {
            return Err("route teacher does not cover every Flash layer/token slot".into());
        }
        Ok(Some(RouteTeacher {
            routes,
            final_state_sha256,
            unions: unions
                .into_iter()
                .map(|(layer, ids)| (layer, ids.into_iter().collect()))
                .collect(),
        }))
    }

    /// The only route-teacher execution body that compares both route IDs and
    /// final-state hashes at every layer/token slot is the token-major owner.
    /// Reject the layer-major direct CLI path rather than emitting a receipt
    /// whose hash-contract claim would exceed its executed checks.
    fn require_token_major_route_teacher(
        route_teacher_requested: bool,
        token_major_resident_banks: bool,
        host_attention_seam_diagnostic: bool,
        host_linear_seam_diagnostic: bool,
    ) -> Result<(), Box<dyn Error>> {
        if route_teacher_requested && !token_major_resident_banks {
            return Err(
                "--route-teacher requires --token-major-resident-banks so every teacher route and final-state hash is checked"
                    .into(),
            );
        }
        if route_teacher_requested
            && (host_attention_seam_diagnostic || host_linear_seam_diagnostic)
        {
            return Err(
                "--route-teacher requires device-only token-major handoffs; host seam diagnostics are incompatible"
                    .into(),
            );
        }
        Ok(())
    }

    /// Partial token-major construction is a diagnostic-only body and cannot
    /// provide the all-48 resident execution contract. Reject it before any
    /// source index or Metal allocation rather than constructing then
    /// downgrading a native run.
    fn require_all_48_token_major_construction(limit: usize) -> Result<(), Box<dyn Error>> {
        if limit != 48 {
            return Err(
                "--token-major-construction-limit must be exactly 48; partial native construction is disabled"
                    .into(),
            );
        }
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16(bytes: &[u8], i: usize) -> f32 {
        f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16)
    }

    fn bf16_rne(value: f32) -> f32 {
        let bits = value.to_bits();
        let low_lsb = (bits >> 16) & 1;
        f32::from_bits((bits.wrapping_add(0x7fff).wrapping_add(low_lsb)) & 0xffff0000)
    }

    fn embedding_row(
        index: &SourceBf16Index,
        token_id: usize,
        vocab: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if token_id >= vocab {
            return Err(format!("token {token_id} outside vocab {vocab}").into());
        }
        let bytes = index.read_raw_range(EMBEDDING, token_id * HIDDEN * 2, HIDDEN * 2)?;
        Ok((0..HIDDEN).map(|i| bf16(&bytes, i)).collect())
    }

    fn repeated_streams(row: &[f32]) -> Vec<f32> {
        (0..STREAMS).flat_map(|_| row.iter().copied()).collect()
    }

    fn vocab(root: &Path) -> Result<usize, Box<dyn Error>> {
        let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        config
            .get("text_config")
            .and_then(|v| v.get("vocab_size"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| "text_config.vocab_size missing".into())
    }

    fn write_state(path: &Path, state: &[f32]) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_new_file(path, &f32_bytes(state))?;
        Ok(())
    }

    /// Captures the exact layer-0 output for each token in the complete
    /// source-bound native session. Flash's PLE formula is fed by this
    /// boundary in the existing source-formula control, so a later PLE
    /// function experiment needs measured stateful contexts here rather than
    /// synthetic perturbations of BOS.
    ///
    /// The exporter is deliberately diagnostic-only: it owns no model state,
    /// does not alter scheduling, and writes a separately sealed manifest only
    /// after the session receipt itself has been persisted.  A consumer must
    /// independently verify both the session contract and every F32 payload.
    struct PlePreLayerStateBankExporter {
        output_dir: PathBuf,
        records: Vec<Value>,
    }

    impl PlePreLayerStateBankExporter {
        const SCHEMA: &'static str = "hawking.flash.ple_pre_layer_state_bank.v1";

        fn new(output_dir: PathBuf) -> Result<Self, Box<dyn Error>> {
            let parent = output_dir
                .parent()
                .ok_or("--ple-pre-layer-state-bank-out has no parent directory")?;
            fs::create_dir_all(parent)?;
            match fs::create_dir(&output_dir) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(format!(
                        "--ple-pre-layer-state-bank-out became occupied while reserving this native attempt: {}",
                        output_dir.display()
                    )
                    .into())
                }
                Err(error) => return Err(error.into()),
            }
            Ok(Self {
                output_dir,
                records: Vec::new(),
            })
        }

        fn record(
            &mut self,
            step: usize,
            token_id: usize,
            observed: &[f32],
        ) -> Result<(), Box<dyn Error>> {
            if observed.len() != HC || observed.iter().any(|value| !value.is_finite()) {
                return Err("pre-PLE state-bank capture has invalid layer-0 state".into());
            }
            let path = self.output_dir.join(format!("state-step-{step:06}.f32"));
            let raw = f32_bytes(observed);
            write_new_file(&path, &raw)?;
            self.records.push(json!({
                "step": step,
                "token_id": token_id,
                "layer": 0,
                "path": path,
                "dtype": "F32_LE",
                "elements": observed.len(),
                "bytes": raw.len(),
                "sha256": sha256(&raw),
                "finite": true,
                "state_role": "source layer-0 output used as the bounded pre-PLE input boundary",
            }));
            Ok(())
        }

        fn receipt_summary(&self) -> Value {
            json!({
                "requested": true,
                "status": "CAPTURED__SEPARATE_MANIFEST_BINDS_SESSION_AFTER_RECEIPT_WRITE",
                "output_dir": self.output_dir,
                "layer": 0,
                "state_count": self.records.len(),
                "state_width": HC,
                "manifest": self.output_dir.join("manifest.json"),
                "claim_boundary": "Raw F32 snapshots are session diagnostics only. They are not a compact representation, independent source PLE parity, direct PLE execution, complete EBPW, capability, or TPS evidence.",
            })
        }

        fn finish(
            &self,
            session_path: &Path,
            session_receipt: &Value,
            token_ids: &[usize],
            prompt_len: usize,
        ) -> Result<(), Box<dyn Error>> {
            if self.records.len() != token_ids.len() {
                return Err(format!(
                    "pre-PLE state-bank captured {} states for {} token slots",
                    self.records.len(),
                    token_ids.len()
                )
                .into());
            }
            for (step, (record, &token_id)) in self.records.iter().zip(token_ids).enumerate() {
                if record.get("step").and_then(Value::as_u64) != Some(step as u64)
                    || record.get("token_id").and_then(Value::as_u64) != Some(token_id as u64)
                    || record.get("layer").and_then(Value::as_u64) != Some(0)
                {
                    return Err("pre-PLE state-bank record order drifted".into());
                }
            }
            let session_bytes = fs::read(session_path)?;
            let mut manifest = json!({
                "schema": Self::SCHEMA,
                "status": "EXPORTED_SOURCE_BOUND_NATIVE_PRE_PLE_STATE_BANK__FORMULA_ONLY",
                "model": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "session_receipt": {
                    "path": session_path,
                    "bytes": session_bytes.len(),
                    "sha256": sha256(&session_bytes),
                    "seal_sha256": session_receipt.get("seal_sha256").cloned().unwrap_or(Value::Null),
                    "schema": session_receipt.get("schema").cloned().unwrap_or(Value::Null),
                    "status": session_receipt.get("status").cloned().unwrap_or(Value::Null),
                },
                "token_ids": token_ids,
                "prompt_length": prompt_len,
                "capture": {
                    "layer": 0,
                    "state_width": HC,
                    "state_role": "source layer-0 output used as the bounded pre-PLE input boundary",
                    "capture_path": "actual complete-session native body, not a synthetic perturbation or reset per token",
                    "persistent_state_authority": "the bound session receipt owns recurrence/KV persistence and token acceptance; this manifest owns only captured F32 boundary states",
                    "upstream_PLE_inclusive_source_trajectory": false,
                    "trajectory_boundary": "The current native session does not embed the source PLE formula. These are real stateful native contexts, not upstream PLE-inclusive source-model trajectory parity.",
                },
                "states": self.records,
                "claim_boundary": "This exports real source-bound native-session pre-PLE boundary states for a later formula/generator discriminator. The native body does not yet embed PLE, so the states are not upstream PLE-inclusive source-model trajectory parity. It is not independent source PLE-output parity, a native PLE runtime, a compact representation, complete EBPW, capability, qualified TPS, or promotion evidence.",
                "promotion_allowed": false,
            });
            let seal = sha256(&serde_json::to_vec(&manifest)?);
            manifest["seal_sha256"] = Value::String(seal);
            write_new_file(
                &self.output_dir.join("manifest.json"),
                &serde_json::to_vec_pretty(&manifest)?,
            )?;
            Ok(())
        }
    }

    fn current_rss_bytes() -> Option<u64> {
        let pid = std::process::id().to_string();
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
            .map(|kib| kib.saturating_mul(1024))
    }

    fn run_linear_segment(
        index: &SourceBf16Index,
        context: &MetalContext,
        layer_start: usize,
        layer_end: usize,
        token_ids: &[usize],
        vocab: usize,
        input_states: Option<&[Vec<f32>]>,
        route_teacher: Option<&RouteTeacher>,
        device_only_compact_banks: bool,
        mut ple_pre_layer_state_bank: Option<&mut PlePreLayerStateBankExporter>,
    ) -> Result<(Vec<Vec<f32>>, Value, Vec<linear::StatefulLinearLayer>), Box<dyn Error>> {
        if let Some(states) = input_states {
            if states.len() != token_ids.len() || states.iter().any(|state| state.len() != HC) {
                return Err(
                    "linear segment input states do not cover every exact token state".into(),
                );
            }
        }
        let first_row = embedding_row(&index, token_ids[0], vocab)?;
        let first_base = repeated_streams(&first_row);
        let mut layers = Vec::new();
        let load_started = Instant::now();
        let bytes_before_layers = index.bytes_read_total();
        for layer in layer_start..=layer_end {
            let session = if let Some(teacher) = route_teacher {
                let union = teacher
                    .unions
                    .get(&layer)
                    .ok_or("route teacher omitted linear-layer union")?;
                if device_only_compact_banks {
                    linear::StatefulLinearLayer::new_compact_union_device_only(
                        index,
                        context,
                        layer,
                        union,
                        &first_base,
                    )?
                } else {
                    linear::StatefulLinearLayer::new_compact_union(
                        index,
                        context,
                        layer,
                        union,
                        &first_base,
                    )?
                }
            } else {
                if device_only_compact_banks {
                    return Err("--device-only-compact-banks requires --route-teacher".into());
                }
                linear::StatefulLinearLayer::new_dense(index, context, layer, &first_base)?
            };
            layers.push(session);
        }
        let persistent_state_bytes = layers
            .len()
            .saturating_mul(linear::StatefulLinearLayer::persistent_state_bytes());
        let (source_load_ns, device_prepare_ns, graph_prepare_ns) = layers
            .iter()
            .map(linear::StatefulLinearLayer::prepare_timing_ns)
            .fold((0u64, 0u64, 0u64), |acc, value| {
                (
                    acc.0.saturating_add(value.0),
                    acc.1.saturating_add(value.1),
                    acc.2.saturating_add(value.2),
                )
            });
        let layer_prepare_ns = load_started.elapsed().as_nanos() as u64;
        let layer_source_bytes = index.bytes_read_total().saturating_sub(bytes_before_layers);
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(token_ids.len());
        let mut rows = Vec::with_capacity(token_ids.len() * layers.len());
        for (step, &token_id) in token_ids.iter().enumerate() {
            let host_base = if let Some(states) = input_states {
                states[step].clone()
            } else {
                let row = embedding_row(&index, token_id, vocab)?;
                repeated_streams(&row)
            };
            let mut prior_device = None;
            let mut final_state = None;
            for (offset, layer) in layers.iter_mut().enumerate() {
                let (device_output, gpu_ns, wall_ns, dispatches, observed) = if offset == 0 {
                    layer.step(context, Some(&host_base), None, step == 0)?
                } else {
                    let input = prior_device
                        .as_ref()
                        .ok_or("missing device inter-layer state")?;
                    layer.step(context, None, Some(input), step == 0)?
                };
                let layer_index = layer_start + offset;
                if observed.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "non-finite linear state at layer {layer_index} step {step}"
                    )
                    .into());
                }
                if layer_index == 0 {
                    if let Some(exporter) = ple_pre_layer_state_bank.as_deref_mut() {
                        exporter.record(step, token_id, &observed)?;
                    }
                }
                // The dense control owns the complete immutable expert bank.
                // Record only the router's already-produced native top-k IDs
                // so a later compact candidate can bind an exact, bounded
                // route union rather than guess or reuse one token's route.
                let route_ids = layer.route_ids();
                if let Some(teacher) = route_teacher {
                    let expected = teacher
                        .routes
                        .get(&(layer_index, step))
                        .ok_or("route teacher omitted linear-layer token row")?;
                    if route_ids != *expected {
                        return Err(format!(
                            "route drift at linear layer {layer_index} token slot {step}: expected={expected:?} observed={route_ids:?}"
                        )
                        .into());
                    }
                }
                rows.push(json!({
                    "step": step,
                    "token_id": token_id,
                    "layer": layer_index,
                    "dispatches": dispatches,
                    "gpu_ns": gpu_ns,
                    "wall_ns": wall_ns,
                    "device_inter_layer_handoff": offset > 0,
                    "inter_species_input": input_states.is_some() && offset == 0,
                    "finite": true,
                    "route_ids": route_ids,
                    "final_state_sha256": sha256(&f32_bytes(&observed)),
                }));
                prior_device = Some(device_output);
                final_state = Some(observed);
            }
            outputs.push(final_state.ok_or("linear segment produced no output")?);
        }
        let execution_wall_ns = started.elapsed().as_nanos() as u64;
        let execution_gpu_ns = rows
            .iter()
            .map(|r| r.get("gpu_ns").and_then(Value::as_u64).unwrap_or(0))
            .sum::<u64>();
        let execution_dispatches = rows
            .iter()
            .map(|r| r.get("dispatches").and_then(Value::as_u64).unwrap_or(0))
            .sum::<u64>();
        let host_and_unattributed_ns = rows
            .iter()
            .map(|r| {
                r.get("wall_ns")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    .saturating_sub(r.get("gpu_ns").and_then(Value::as_u64).unwrap_or(0))
            })
            .sum::<u64>();
        Ok((
            outputs,
            json!({
                "layers": [layer_start, layer_end],
                "species": "linear_attention",
                "stateful_recurrence": true,
                "expert_bank_mode": if route_teacher.is_some() { "route_union_compact_teacher_bound" } else { "dense" },
                "immutable_weight_ownership": if device_only_compact_banks { "device_only_after_source_upload" } else { "host_and_device" },
                "host_weights_retained": layers.iter().all(linear::StatefulLinearLayer::host_weights_retained),
                "persistent_state": {
                    "layers": layers.len(),
                    "per_layer_bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                    "total_bytes": persistent_state_bytes,
                    "growth_bytes_per_token": 0,
                    "reset_only_before_first_session_token": true,
                },
                "device_inter_layer_handoffs": rows.iter().filter(|r| r.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                "steps": rows,
                "wall_ns": execution_wall_ns,
                "source_payload_bytes_read": layer_source_bytes,
                "source_payload_bytes_read_cumulative": index.bytes_read_total(),
                "timing": {
                    "layer_source_and_device_prepare_ns": layer_prepare_ns,
                    "source_load_ns": source_load_ns,
                    "device_prepare_ns": device_prepare_ns,
                    "graph_prepare_ns": graph_prepare_ns,
                    "layer_source_bytes_read": layer_source_bytes,
                    "execution_wall_ns": execution_wall_ns,
                    "execution_gpu_ns": execution_gpu_ns,
                    "execution_dispatches": execution_dispatches,
                    "host_and_unattributed_ns": host_and_unattributed_ns
                },
            }),
            layers,
        ))
    }

    fn run_full_layer(
        index: &SourceBf16Index,
        context: &MetalContext,
        layer: usize,
        token_ids: &[usize],
        input_states: &[Vec<f32>],
        out_dir: &Path,
        route_teacher: Option<&RouteTeacher>,
    ) -> Result<(Vec<Vec<f32>>, Value), Box<dyn Error>> {
        let receipt = out_dir.join(format!("layer-{layer}-attention.json"));
        let layer_call_started = Instant::now();
        let route_union = route_teacher
            .map(|teacher| {
                teacher
                    .unions
                    .get(&layer)
                    .ok_or("route teacher omitted full-attention union")
            })
            .transpose()?;
        let mut bank = full::StatefulFullAttentionLayer::new_device_only(
            index,
            context,
            layer,
            token_ids.len(),
            route_union.map(Vec::as_slice),
        )?;
        let mut states = Vec::with_capacity(token_ids.len());
        let mut rows = Vec::with_capacity(token_ids.len());
        for (step, (&token_id, input)) in token_ids.iter().zip(input_states).enumerate() {
            let (_device_state, state, row) =
                bank.step(context, Some(input), None, step, token_id)?;
            if let Some(teacher) = route_teacher {
                let expected = teacher
                    .routes
                    .get(&(layer, step))
                    .ok_or("route teacher omitted full-attention token row")?;
                let observed = row
                    .get("route_ids")
                    .and_then(Value::as_array)
                    .ok_or("resident full-attention row omitted top-k IDs")?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .map(|id| id as u32)
                            .ok_or("non-integer route ID")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if observed != *expected {
                    return Err(format!("route drift at resident full-attention layer {layer} token slot {step}: expected={expected:?} observed={observed:?}").into());
                }
            }
            states.push(state);
            rows.push(row);
        }
        if states.len() != token_ids.len()
            || states
                .iter()
                .any(|s| s.len() != HC || s.iter().any(|v| !v.is_finite()))
        {
            return Err(format!("resident full-attention layer {layer} returned invalid state count or non-finite values").into());
        }
        let (source_load_ns, device_prepare_ns, graph_prepare_ns) = bank.prepare_timing_ns();
        let execution_gpu_ns = rows
            .iter()
            .filter_map(|row| row.get("gpu_ns").and_then(Value::as_u64))
            .sum::<u64>();
        let execution_dispatches = rows
            .iter()
            .filter_map(|row| row.get("dispatches").and_then(Value::as_u64))
            .sum::<u64>();
        let execution_wall_ns = rows
            .iter()
            .filter_map(|row| row.get("wall_ns").and_then(Value::as_u64))
            .sum::<u64>();
        let attention_doc = json!({
            "schema": "hawking.flash.stateful_full_attention_resident_bank.v1",
            "status": "PASSED_STATEFUL_KV_ORGAN",
            "layer": layer,
            "steps": rows,
            "execution": {
                "process_boundary": "one native process",
                "context_reused": true,
                "weights_reused": true,
                "immutable_weight_ownership": "device_only_after_source_upload",
                "host_source_weights_retained_during_token_loop": bank.host_weights_retained(),
                "kv_cache_reused": true,
                "kv_cache_slots": token_ids.len(),
                "state_memory": {"persistent_kv_bytes": bank.persistent_state_bytes(), "growth_bytes_per_token": full::StatefulFullAttentionLayer::persistent_state_growth_bytes_per_token()},
                "expert_bank_mode": if route_teacher.is_some() { "route_union_compact" } else { "dense" },
                "source_payload_bytes_read": bank.source_payload_bytes(),
                "device_weight_bytes": bank.device_weight_bytes(),
                "timing": {"source_load_ns": source_load_ns, "device_prepare_ns": device_prepare_ns, "graph_prepare_ns": graph_prepare_ns, "execution_wall_ns": execution_wall_ns, "execution_gpu_ns": execution_gpu_ns, "execution_dispatches": execution_dispatches},
                "claim_boundary": "One full-attention bank owns its immutable Metal weights and KV state across the named token slots. This is an organ-level resident primitive, not complete token-major whole-model residency or TPS."
            }
        });
        write_new_file(&receipt, &serde_json::to_vec_pretty(&attention_doc)?)?;
        let attention_execution = attention_doc.get("execution").cloned().unwrap();
        Ok((
            states,
            json!({
                "layer": layer,
                "species": "full_attention",
                "stateful_kv": true,
                "expert_bank_mode": attention_execution.get("expert_bank_mode").cloned().unwrap_or(Value::Null),
                "immutable_weight_ownership": attention_execution.get("immutable_weight_ownership").cloned().unwrap_or(Value::Null),
                "host_source_weights_retained_during_token_loop": attention_execution.get("host_source_weights_retained_during_token_loop").cloned().unwrap_or(Value::Null),
                "state_memory": attention_execution.get("state_memory").cloned().unwrap_or_else(|| json!({"status": "UNAVAILABLE"})),
                "source_payload_bytes_read": attention_execution.get("source_payload_bytes_read").cloned().unwrap_or(Value::Null),
                "device_weight_bytes": attention_execution.get("device_weight_bytes").cloned().unwrap_or(Value::Null),
                "timing": {"full_layer_call_wall_ns": layer_call_started.elapsed().as_nanos() as u64, "receipt_read_ns": 0, "source_load_ns": attention_execution.pointer("/timing/source_load_ns").cloned().unwrap_or(Value::Null), "oracle_ns": Value::Null, "device_prepare_ns": attention_execution.pointer("/timing/device_prepare_ns").cloned().unwrap_or(Value::Null)},
                "receipt": receipt,
            }),
        ))
    }

    fn write_boundary_f32(
        out_dir: &Path,
        ordinal: usize,
        layer: Option<usize>,
        stage: &str,
        values: &[f32],
        shape: &[usize],
        producer: &str,
    ) -> Result<Value, Box<dyn Error>> {
        let expected_elements = shape.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .ok_or("boundary payload shape overflow")
        })?;
        if values.len() != expected_elements || values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "native boundary seam {ordinal}/{stage} has invalid extent or non-finite values"
            )
            .into());
        }
        let raw = f32_bytes(values);
        let path = out_dir.join(format!("seam-{ordinal:02}-{stage}.f32"));
        write_new_file(&path, &raw)?;
        Ok(json!({
            "ordinal": ordinal,
            "layer": layer,
            "stage": stage,
            "shape": shape,
            "producer": producer,
            "payload": {
                "path": path,
                "sha256": sha256(&raw),
                "dtype": "F32_LE",
                "elements": values.len(),
                "bytes": raw.len(),
            },
        }))
    }

    fn write_layer0_trace_f32(
        out_dir: &Path,
        ordinal: usize,
        stage: &str,
        values: &[f32],
        shape: &[usize],
        producer: &str,
    ) -> Result<Value, Box<dyn Error>> {
        let expected_elements = shape.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .ok_or("layer-0 trace payload shape overflow")
        })?;
        if values.len() != expected_elements || values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "native layer-0 trace {ordinal}/{stage} has invalid extent or non-finite values"
            )
            .into());
        }
        let raw = f32_bytes(values);
        let payload_path = out_dir.join(format!("layer0-trace-{ordinal:02}-{stage}.f32"));
        write_new_file(&payload_path, &raw)?;
        Ok(json!({
            "ordinal": ordinal,
            "layer": 0,
            "stage": stage,
            "shape": shape,
            "producer": producer,
            "payload": {
                "path": payload_path,
                "sha256": sha256(&raw),
                "dtype": "F32_LE",
                "elements": values.len(),
                "bytes": raw.len(),
            },
        }))
    }

    fn source_equivalent_hc_injection_weights(
        block_logits: &[f32],
        label: &str,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if block_logits.len() != STREAMS || block_logits.iter().any(|value| !value.is_finite()) {
            return Err(format!("native {label} HyperConnection block logits are invalid").into());
        }
        let divisor = STREAMS as f32;
        let weights = block_logits
            .iter()
            .map(|logit| {
                let value = bf16_rne(*logit / divisor);
                let exponential = bf16_rne(value.abs().exp());
                let denominator = bf16_rne(1.0 + exponential);
                let inverse = bf16_rne(1.0 / denominator);
                bf16_rne(2.0 * bf16_rne(if value < 0.0 { inverse } else { 1.0 - inverse }))
            })
            .collect::<Vec<_>>();
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(
                format!("native {label} HyperConnection injection weights are non-finite").into(),
            );
        }
        Ok(weights)
    }

    fn source_equivalent_hc_mix_weights(
        gate_logits: &[f32],
        label: &str,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if gate_logits.len() != HC || gate_logits.iter().any(|value| !value.is_finite()) {
            return Err(format!("native {label} HyperConnection gate logits are invalid").into());
        }
        let weights = gate_logits
            .iter()
            .map(|logit| {
                let value = bf16_rne(*logit);
                let exponential = bf16_rne(value.abs().exp());
                let denominator = bf16_rne(1.0 + exponential);
                let inverse = bf16_rne(1.0 / denominator);
                bf16_rne(if value < 0.0 { inverse } else { 1.0 - inverse })
            })
            .collect::<Vec<_>>();
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(
                format!("native {label} HyperConnection mix weights are non-finite").into(),
            );
        }
        Ok(weights)
    }

    fn write_boundary_cache_f32(
        out_dir: &Path,
        boundary: &str,
        name: &str,
        values: &[f32],
    ) -> Result<Value, Box<dyn Error>> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(
                format!("native layer-4 {boundary}/{name} cache is empty or non-finite").into(),
            );
        }
        let raw = f32_bytes(values);
        let path = out_dir.join(format!("cache-layer4-{boundary}-{name}.f32"));
        write_new_file(&path, &raw)?;
        Ok(json!({
            "name": name,
            "boundary": boundary,
            "original_dtype": "F32",
            "shape": [values.len()],
            "layout": "native_flattened",
            "payload": {
                "path": path,
                "sha256": sha256(&raw),
                "dtype": "F32_LE",
                "elements": values.len(),
                "bytes": raw.len(),
            },
        }))
    }

    fn read_bound_f32_state(
        path: &Path,
        expected_sha256: &str,
        elements: usize,
        label: &str,
    ) -> Result<(Vec<f32>, Value), Box<dyn Error>> {
        if !is_lowercase_sha256(expected_sha256) {
            return Err(format!("{label} expected SHA-256 is malformed").into());
        }
        let before = fs::symlink_metadata(path)?;
        if before.file_type().is_symlink() || !before.is_file() || before.nlink() != 1 {
            return Err(
                format!("{label} must be a regular non-symlink, non-hard-linked file").into(),
            );
        }
        let raw = read_stable_regular_file(path, label)?;
        if raw.len() != elements.saturating_mul(std::mem::size_of::<f32>())
            || sha256(&raw) != expected_sha256
        {
            return Err(
                format!("{label} bytes or SHA-256 differ from the admitted binding").into(),
            );
        }
        let values = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{label} contains a non-finite value").into());
        }
        let resolved = path.canonicalize()?;
        Ok((
            values,
            json!({
                "path": resolved,
                "sha256": expected_sha256,
                "dtype": "F32_LE",
                "elements": elements,
                "bytes": raw.len(),
            }),
        ))
    }

    fn validate_source_boundary_lineage(
        source_capture_sha256: &str,
        preflight_sha256: &str,
    ) -> Result<(), Box<dyn Error>> {
        for (label, digest) in [
            ("source capture", source_capture_sha256),
            ("boundary preflight", preflight_sha256),
        ] {
            if !is_lowercase_sha256(digest) {
                return Err(format!("{label} SHA-256 is malformed").into());
            }
        }
        Ok(())
    }

    /// Device result and explicit readbacks for the narrow layer-0 attention
    /// HyperConnection localizer.  The vectors are diagnostic artifacts only;
    /// no caller can feed them back into a resident graph through this type.
    struct NativeLayer0AttentionHcTrace {
        embedding_hc: Vec<f32>,
        trace_values: Vec<(Vec<f32>, &'static str)>,
        gpu_ns: u64,
        wall_ns: u64,
        dispatches: usize,
    }

    /// The full source-bound diagnostic adds the layer-0 MLP result after the
    /// reusable attention trace.  The fast causal localizer deliberately does
    /// not construct this wider result.
    struct NativeLayer0SourceBoundary {
        attention_trace: NativeLayer0AttentionHcTrace,
        layer0_state: Vec<f32>,
    }

    /// The causal trace contract lives in Hawking core semantics, rather than
    /// in a Python comparison wrapper.  Its order is the observation order;
    /// a mismatch identifies a boundary and never grants a semantic verdict.
    fn layer0_attention_hc_trace_plan() -> Vec<DiagnosticSemanticSeam> {
        vec![
            DiagnosticSemanticSeam {
                ordinal: 0,
                layer: Some(0),
                semantic_id: "attention_hyper_connection_normalized".to_owned(),
                shape: vec![1, 1, HC],
            },
            DiagnosticSemanticSeam {
                ordinal: 1,
                layer: Some(0),
                semantic_id: "attention_hyper_connection_down_projection".to_owned(),
                shape: vec![1, 1, HC_LOWRANK],
            },
            DiagnosticSemanticSeam {
                ordinal: 2,
                layer: Some(0),
                semantic_id: "attention_hyper_connection_low_rank_activation".to_owned(),
                shape: vec![1, 1, HC_LOWRANK],
            },
            DiagnosticSemanticSeam {
                ordinal: 3,
                layer: Some(0),
                semantic_id: "attention_hyper_connection_up_projection".to_owned(),
                shape: vec![1, 1, HC],
            },
            DiagnosticSemanticSeam {
                ordinal: 4,
                layer: Some(0),
                semantic_id: "attention_hyper_connection_mix_weights".to_owned(),
                shape: vec![1, 1, HC],
            },
            DiagnosticSemanticSeam {
                ordinal: 5,
                layer: Some(0),
                semantic_id: "attention_hyper_connection_mixed".to_owned(),
                shape: vec![1, 1, HIDDEN],
            },
            DiagnosticSemanticSeam {
                ordinal: 6,
                layer: Some(0),
                semantic_id: "attention_block_injection_weights".to_owned(),
                shape: vec![1, 1, STREAMS],
            },
            DiagnosticSemanticSeam {
                ordinal: 7,
                layer: Some(0),
                semantic_id: "attention_branch".to_owned(),
                shape: vec![1, 1, HIDDEN],
            },
            DiagnosticSemanticSeam {
                ordinal: 8,
                layer: Some(0),
                semantic_id: "post_attention_state".to_owned(),
                shape: vec![1, 1, HC],
            },
        ]
    }

    fn native_layer0_attention_hc_trace_from_snapshot(
        embedding_hc: Vec<f32>,
        layer0_snapshot: linear::DiagnosticLayer0AttentionTraceSnapshot,
        gpu_ns: u64,
        wall_ns: u64,
        dispatches: usize,
    ) -> Result<NativeLayer0AttentionHcTrace, Box<dyn Error>> {
        if layer0_snapshot.base != embedding_hc {
            return Err("native layer-0 trace snapshot drifted from its executed input".into());
        }
        let layer0_attention_injection = source_equivalent_hc_injection_weights(
            &layer0_snapshot.attn_block_logits,
            "attention",
        )?;
        let layer0_attention_mix =
            source_equivalent_hc_mix_weights(&layer0_snapshot.attn_gate_logits, "attention")?;
        Ok(NativeLayer0AttentionHcTrace {
            embedding_hc,
            trace_values: vec![
                (layer0_snapshot.attn_norm, "native_rust_metal"),
                (layer0_snapshot.attn_low_rank, "native_rust_metal"),
                (
                    layer0_snapshot.attn_low_rank_activation,
                    "native_rust_metal",
                ),
                (layer0_snapshot.attn_gate_logits, "native_rust_metal"),
                (
                    layer0_attention_mix,
                    "native_readback_source_equivalent_hc_formula",
                ),
                (layer0_snapshot.attn_input, "native_rust_metal"),
                (
                    layer0_attention_injection,
                    "native_readback_source_equivalent_hc_formula",
                ),
                (layer0_snapshot.attn_block_output, "native_rust_metal"),
                (layer0_snapshot.post_attn_state, "native_rust_metal"),
            ],
            gpu_ns,
            wall_ns,
            dispatches,
        })
    }

    /// Full layer-0 owner used only by the PLE-bound source boundary.  It
    /// retains the MLP state that the source boundary has explicitly declared.
    fn execute_native_layer0_source_boundary(
        index: &SourceBf16Index,
        context: &MetalContext,
        token_id: usize,
        attention_hc_norm_precision: linear::HcNormOutputPrecision,
    ) -> Result<NativeLayer0SourceBoundary, Box<dyn Error>> {
        if token_id != PROMPT_IDS[0] {
            return Err(
                "native source boundary is fixed to canonical token index 0 / token id 5423".into(),
            );
        }
        let embedding = embedding_row(index, token_id, PINNED_VOCAB_SIZE)?;
        let embedding_hc = repeated_streams(&embedding);
        let mut layer0 = linear::StatefulLinearLayer::new_dense_with_attention_hc_norm_precision(
            index,
            context,
            0,
            &embedding_hc,
            attention_hc_norm_precision,
        )?;
        let (_, gpu_ns, wall_ns, dispatches, layer0_state) =
            layer0.step(context, Some(&embedding_hc), None, true)?;
        let layer0_snapshot = layer0.diagnostic_layer0_attention_trace_snapshot();
        drop(layer0);
        Ok(NativeLayer0SourceBoundary {
            attention_trace: native_layer0_attention_hc_trace_from_snapshot(
                embedding_hc,
                layer0_snapshot,
                gpu_ns,
                wall_ns,
                dispatches,
            )?,
            layer0_state,
        })
    }

    /// Fast causal localizer: load and dispatch only the attention-HC prefix.
    /// This deliberately stops before layer-0's MLP rather than taking a
    /// complete layer result and discarding most of it afterwards.
    fn execute_native_layer0_attention_hc_trace(
        index: &SourceBf16Index,
        context: &MetalContext,
        token_id: usize,
        attention_hc_norm_precision: linear::HcNormOutputPrecision,
    ) -> Result<NativeLayer0AttentionHcTrace, Box<dyn Error>> {
        if token_id != PROMPT_IDS[0] {
            return Err(
                "native layer-0 attention-HC trace is fixed to canonical token index 0 / token id 5423"
                    .into(),
            );
        }
        let embedding = embedding_row(index, token_id, PINNED_VOCAB_SIZE)?;
        let embedding_hc = repeated_streams(&embedding);
        let mut layer0 = linear::StatefulAttentionTraceLayer::new(
            index,
            context,
            0,
            &embedding_hc,
            attention_hc_norm_precision,
        )?;
        let (gpu_ns, wall_ns, dispatches) = layer0.step(context, &embedding_hc)?;
        let layer0_snapshot = layer0.diagnostic_snapshot();
        drop(layer0);
        native_layer0_attention_hc_trace_from_snapshot(
            embedding_hc,
            layer0_snapshot,
            gpu_ns,
            wall_ns,
            dispatches,
        )
    }

    fn write_native_layer0_attention_hc_trace(
        out_dir: &Path,
        trace: &NativeLayer0AttentionHcTrace,
    ) -> Result<Vec<Value>, Box<dyn Error>> {
        let plan = layer0_attention_hc_trace_plan();
        validate_ordered_semantic_seams(&plan)
            .map_err(|error| -> Box<dyn Error> { Box::new(error) })?;
        if trace.trace_values.len() != plan.len() {
            return Err("native layer-0 trace did not emit every predeclared payload".into());
        }
        plan.iter()
            .zip(&trace.trace_values)
            .map(|(seam, (values, producer))| {
                write_layer0_trace_f32(
                    out_dir,
                    seam.ordinal,
                    &seam.semantic_id,
                    values,
                    &seam.shape,
                    producer,
                )
            })
            .collect()
    }

    /// Fast native-only causal localizer.  This consumes the sealed source
    /// capture lineage but executes only layer 0, never accepts a source PLE
    /// payload, and deliberately cannot be mistaken for a layers-0-through-4
    /// source-bound capture.
    #[allow(clippy::too_many_arguments)]
    fn run_native_layer0_attention_hc_trace_capture(
        index: &SourceBf16Index,
        context: &MetalContext,
        root: &Path,
        token_id: usize,
        source_capture_sha256: &str,
        preflight_sha256: &str,
        out_dir: &Path,
        native_invocation_admission: NativeInvocationAdmission,
        native_lane_preflight: Value,
        source_input_identity: Value,
        attention_hc_norm_precision: linear::HcNormOutputPrecision,
    ) -> Result<Value, Box<dyn Error>> {
        validate_source_boundary_lineage(source_capture_sha256, preflight_sha256)?;
        let started = Instant::now();
        let source_bytes_before = index.bytes_read_total();
        let layer0 = execute_native_layer0_attention_hc_trace(
            index,
            context,
            token_id,
            attention_hc_norm_precision,
        )?;
        let seams = vec![write_boundary_f32(
            out_dir,
            0,
            None,
            "embedding_hc_tiled",
            &layer0.embedding_hc,
            &[1, 1, HC],
            "native_embedding_row",
        )?];
        let layer0_trace = write_native_layer0_attention_hc_trace(out_dir, &layer0)?;
        Ok(json!({
            "schema": "hawking.flash.native_layer0_attention_hc_trace_capture.v1",
            "status": "CAPTURED_NATIVE_LAYER0_ATTENTION_HC_TRACE_ONLY_TOKEN_0",
            "capture_scope": "LAYER0_ATTENTION_HC_TRACE_ONLY",
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "source_input_identity": source_input_identity,
            "token_index": 0,
            "token_id": token_id,
            "layers_executed": [0],
            "layer0_trace_contract_version": 5,
            "source_capture_lineage": {
                "source_capture_sha256": source_capture_sha256,
                "preflight_sha256": preflight_sha256,
                "source_ple_state_consumed": false,
            },
            "layer0_trace": layer0_trace,
            "seams": seams,
            "execution": {
                "provider": "hawking-core Rust + Apple Metal",
                "process_boundary": "one native process",
                "invocation_admission": native_invocation_admission.receipt_value(),
                "native_lane_preflight": native_lane_preflight,
                "attention_hc_norm_precision": attention_hc_norm_precision.receipt_value(),
                "source_payload_bytes_read": index.bytes_read_total().saturating_sub(source_bytes_before),
                "layer0_gpu_ns": layer0.gpu_ns,
                "layer0_wall_ns": layer0.wall_ns,
                "layer0_dispatches": layer0.dispatches,
                "mlp_executed": false,
                "elapsed_ns": started.elapsed().as_nanos() as u64,
                "full_body_executed": false,
                "lm_head_executed": false,
            },
            "promotion_allowed": false,
            "claim_boundary": "This is a native-only, layer-0 attention-HyperConnection causal localizer over a sealed source capture lineage. It does not load or inject source PLE payload state, execute layer-0 MLP, layers 1 through 4, or the complete body, establish source parity by itself, qualify an NR, measure EBPW/TPS/capability, deploy, promote Pulsar, or retire Kimi.",
        }))
    }

    /// Bounded native counterpart to the owner-gated source payload capture.
    /// Flash's native body does not yet implement PLE, so the admitted source
    /// post-PLE state is injected once at the explicit layer-1 seam.  This is
    /// a discriminator, never an ordinary inference dependency or a claim
    /// that native PLE exists.
    #[allow(clippy::too_many_arguments)]
    fn run_native_source_boundary_capture(
        index: &SourceBf16Index,
        context: &MetalContext,
        root: &Path,
        token_id: usize,
        source_post_ple_state: &Path,
        source_post_ple_sha256: &str,
        source_capture_sha256: &str,
        preflight_sha256: &str,
        out_dir: &Path,
        native_invocation_admission: NativeInvocationAdmission,
        native_lane_preflight: Value,
        source_input_identity: Value,
        attention_hc_norm_precision: linear::HcNormOutputPrecision,
    ) -> Result<Value, Box<dyn Error>> {
        validate_source_boundary_lineage(source_capture_sha256, preflight_sha256)?;
        if token_id != PROMPT_IDS[0] {
            return Err("native source boundary capture is fixed to canonical token index 0 / token id 5423".into());
        }
        let started = Instant::now();
        let source_bytes_before = index.bytes_read_total();
        let layer0 = execute_native_layer0_source_boundary(
            index,
            context,
            token_id,
            attention_hc_norm_precision,
        )?;
        let mut seams = vec![write_boundary_f32(
            out_dir,
            0,
            None,
            "embedding_hc_tiled",
            &layer0.attention_trace.embedding_hc,
            &[1, 1, HC],
            "native_embedding_row",
        )?];
        let layer0_trace =
            write_native_layer0_attention_hc_trace(out_dir, &layer0.attention_trace)?;
        seams.push(write_boundary_f32(
            out_dir,
            1,
            Some(0),
            "mlp_injection",
            &layer0.layer0_state,
            &[1, 1, HC],
            "native_rust_metal",
        )?);
        seams.push(write_boundary_f32(
            out_dir,
            2,
            Some(1),
            "pre_ple",
            &layer0.layer0_state,
            &[1, 1, HC],
            "native_rust_metal",
        )?);

        let (post_ple_state, source_post_ple_binding) = read_bound_f32_state(
            source_post_ple_state,
            source_post_ple_sha256,
            HC,
            "owner-bound source post-PLE token-0 state",
        )?;
        seams.push(write_boundary_f32(
            out_dir,
            3,
            Some(1),
            "ple_additive_output",
            &post_ple_state,
            &[1, 1, HC],
            "owner_authorized_external_source_teacher_injection",
        )?);

        let mut current = post_ple_state;
        for (layer_id, ordinal) in [(1usize, 4usize), (2usize, 5usize)] {
            let mut layer =
                linear::StatefulLinearLayer::new_dense(index, context, layer_id, &current)?;
            let (_, _, _, _, output) = layer.step(context, Some(&current), None, true)?;
            seams.push(write_boundary_f32(
                out_dir,
                ordinal,
                Some(layer_id),
                "mlp_injection",
                &output,
                &[1, 1, HC],
                "native_rust_metal_after_source_ple_injection",
            )?);
            current = output;
        }

        let (layer3_states, layer3_receipt) =
            run_full_layer(index, context, 3, &[token_id], &[current], out_dir, None)?;
        let layer3_state = layer3_states
            .into_iter()
            .next()
            .ok_or("native boundary full-attention layer 3 returned no token-0 state")?;
        seams.push(write_boundary_f32(
            out_dir,
            6,
            Some(3),
            "mlp_injection_layer4_input",
            &layer3_state,
            &[1, 1, HC],
            "native_rust_metal_after_source_ple_injection",
        )?);

        let mut layer4 = linear::StatefulLinearLayer::new_dense(index, context, 4, &layer3_state)?;
        let pre = layer4.diagnostic_boundary_snapshot();
        let (_, gpu_ns, wall_ns, dispatches, layer4_state) =
            layer4.step(context, Some(&layer3_state), None, true)?;
        let post = layer4.diagnostic_boundary_snapshot();
        if post.final_state != layer4_state || post.base != layer3_state {
            return Err(
                "native layer-4 boundary snapshot drifted from its executed input/output".into(),
            );
        }
        seams.push(write_boundary_f32(
            out_dir,
            7,
            Some(4),
            "attention_hyper_connection_mixed",
            &post.attn_input,
            &[1, 1, HIDDEN],
            "native_rust_metal_after_source_ple_injection",
        )?);
        seams.push(write_boundary_f32(
            out_dir,
            8,
            Some(4),
            "attention_branch",
            &post.attn_block_output,
            &[1, 1, HIDDEN],
            "native_rust_metal_after_source_ple_injection",
        )?);
        seams.push(write_boundary_f32(
            out_dir,
            9,
            Some(4),
            "attention_injection",
            &post.post_attn_state,
            &[1, 1, HC],
            "native_rust_metal_after_source_ple_injection",
        )?);
        seams.push(write_boundary_f32(
            out_dir,
            10,
            Some(4),
            "mlp_hyper_connection_mixed",
            &post.mlp_input,
            &[1, 1, HIDDEN],
            "native_rust_metal_after_source_ple_injection",
        )?);
        let route_ids = post
            .route_ids
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>();
        let route_id_raw = route_ids
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let route_weight_raw = f32_bytes(&post.route_weights);
        let route_id_path = out_dir.join("seam-11-router-top10-ids.i32");
        let route_weight_path = out_dir.join("seam-11-router-top10-weights.f32");
        write_new_file(&route_id_path, &route_id_raw)?;
        write_new_file(&route_weight_path, &route_weight_raw)?;
        seams.push(json!({
            "ordinal": 11,
            "layer": 4,
            "stage": "router_top10_ids_and_weights",
            "shape": [10],
            "producer": "native_rust_metal_after_source_ple_injection",
            "expert_ids": route_ids,
            "weights": post.route_weights,
            "ids_payload": {"path": route_id_path, "sha256": sha256(&route_id_raw), "dtype": "I32_LE", "elements": 10, "bytes": route_id_raw.len()},
            "weights_payload": {"path": route_weight_path, "sha256": sha256(&route_weight_raw), "dtype": "F32_LE", "elements": 10, "bytes": route_weight_raw.len()},
        }));
        seams.push(write_boundary_f32(
            out_dir,
            12,
            Some(4),
            "mlp_route_and_experts",
            &post.moe_output,
            &[1, 1, HIDDEN],
            "native_rust_metal_after_source_ple_injection",
        )?);
        seams.push(write_boundary_f32(
            out_dir,
            13,
            Some(4),
            "mlp_injection_layer4_output",
            &post.final_state,
            &[1, 1, HC],
            "native_rust_metal_after_source_ple_injection",
        )?);
        let layer4_cache = json!({
            "pre_attention": [
                write_boundary_cache_f32(out_dir, "pre-attention", "conv-state", &pre.conv_state)?,
                write_boundary_cache_f32(out_dir, "pre-attention", "recurrent-state", &pre.recurrent_state)?,
            ],
            "post_attention": [
                write_boundary_cache_f32(out_dir, "post-attention", "conv-state", &post.conv_state)?,
                write_boundary_cache_f32(out_dir, "post-attention", "recurrent-state", &post.recurrent_state)?,
            ],
        });
        if seams.len() != 14 {
            return Err("native boundary capture did not emit every predeclared seam".into());
        }
        Ok(json!({
            "schema": "hawking.flash.native_source_boundary_capture.v1",
            "status": "CAPTURED_NATIVE_LAYERS_0_THROUGH_4_TOKEN_0_WITH_SOURCE_PLE_INJECTION",
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "source_input_identity": source_input_identity,
            "token_index": 0,
            "token_id": token_id,
            "layers_executed": [0, 1, 2, 3, 4],
            "layer0_trace_contract_version": 5,
            "ple": {
                "native_implementation_present": false,
                "injected_at_layer": 1,
                "source_post_ple_state": source_post_ple_binding,
                "source_capture_sha256": source_capture_sha256,
                "preflight_sha256": preflight_sha256,
                "purpose": "teacher-forced diagnostic seam only; never an ordinary inference dependency",
            },
            "layer0_trace": layer0_trace,
            "seams": seams,
            "layer4_cache": layer4_cache,
            "layer3_receipt": layer3_receipt,
            "execution": {
                "provider": "hawking-core Rust + Apple Metal",
                "process_boundary": "one native process",
                "invocation_admission": native_invocation_admission.receipt_value(),
                "native_lane_preflight": native_lane_preflight,
                "attention_hc_norm_precision": attention_hc_norm_precision.receipt_value(),
                "source_payload_bytes_read": index.bytes_read_total().saturating_sub(source_bytes_before),
                "layer4_gpu_ns": gpu_ns,
                "layer4_wall_ns": wall_ns,
                "layer4_dispatches": dispatches,
                "elapsed_ns": started.elapsed().as_nanos() as u64,
                "full_body_executed": false,
                "lm_head_executed": false,
            },
            "promotion_allowed": false,
            "claim_boundary": "Bounded native diagnostic with one explicit owner-bound source PLE state injection. It does not implement native PLE, establish source parity by itself, execute the complete body, qualify an NR, measure EBPW/TPS/capability, deploy, promote Pulsar, or retire Kimi.",
        }))
    }

    /// Execute one layer-4 recurrent handoff control while preserving exactly
    /// the incoming layer-3 host states.  The constructor base is intentionally
    /// separate from every per-token handoff: `diagnostic_step_with_host_base`
    /// writes and readbacks the actual input before submission, so this control
    /// can distinguish graph-construction allocation history from an incorrect
    /// inter-species activation handoff.
    fn run_layer4_handoff_control(
        bank: &mut linear::StatefulLinearLayer,
        context: &MetalContext,
        name: &str,
        constructor_base: &[f32],
        input_states: &[Vec<f32>],
        scratch_seed_bits: Option<u32>,
    ) -> Result<Value, Box<dyn Error>> {
        if constructor_base.len() != HC
            || constructor_base.iter().any(|value| !value.is_finite())
            || input_states.is_empty()
            || input_states
                .iter()
                .any(|state| state.len() != HC || state.iter().any(|value| !value.is_finite()))
        {
            return Err("layer-4 handoff control received invalid finite state".into());
        }
        let graph_prepare_ns = bank.diagnostic_reinitialize_graph(context, constructor_base)?;
        let seeded_transient_bytes = scratch_seed_bits
            .map(|bits| bank.diagnostic_seed_transient_buffers(bits, 0xD15E_A5ED))
            .unwrap_or(0);
        let started = Instant::now();
        let mut rows = Vec::with_capacity(input_states.len());
        for (step, input_state) in input_states.iter().enumerate() {
            let input_state_sha256 = sha256(&f32_bytes(input_state));
            let (_device_output, gpu_ns, wall_ns, dispatches, final_state, base_readback_sha256) =
                bank.diagnostic_step_with_host_base(context, input_state, step == 0)?;
            if base_readback_sha256 != input_state_sha256 {
                return Err(format!(
                    "layer-4 {name} control base readback drifted at step {step}: expected={input_state_sha256} observed={base_readback_sha256}"
                )
                .into());
            }
            let final_state_sha256 = sha256(&f32_bytes(&final_state));
            rows.push(json!({
                "step": step,
                "input_state_sha256": input_state_sha256,
                "base_readback_sha256": base_readback_sha256,
                "final_state_sha256": final_state_sha256,
                "gpu_ns": gpu_ns,
                "wall_ns": wall_ns,
                "dispatches": dispatches,
                "state_reset": step == 0,
                "finite": true,
            }));
        }
        let execution_wall_ns = started.elapsed().as_nanos() as u64;
        let execution_gpu_ns = rows
            .iter()
            .filter_map(|row| row.get("gpu_ns").and_then(Value::as_u64))
            .sum::<u64>();
        let execution_dispatches = rows
            .iter()
            .filter_map(|row| row.get("dispatches").and_then(Value::as_u64))
            .sum::<u64>();
        let output_hashes = rows
            .iter()
            .map(|row| {
                row.get("final_state_sha256")
                    .and_then(Value::as_str)
                    .ok_or("layer-4 control row omitted output hash")
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(json!({
            "name": name,
            "constructor_base_sha256": sha256(&f32_bytes(constructor_base)),
            "constructor_base_role": "graph construction only; every token writes the measured layer-3 host handoff before graph submission",
            "scratch_seed": scratch_seed_bits.map(|bits| json!({
                "f32_bits_hex": format!("0x{bits:08x}"),
                "f32_value": f32::from_bits(bits),
                "u32_route_seed_hex": "0xd15ea5ed",
                "seeded_transient_bytes": seeded_transient_bytes,
                "excluded_buffers": ["base", "conv_state", "recurrent_state"],
            })).unwrap_or_else(|| json!({"seeded": false})),
            "steps": rows,
            "output_state_sha256": output_hashes,
            "timing": {
                "graph_prepare_ns": graph_prepare_ns,
                "execution_wall_ns": execution_wall_ns,
                "execution_gpu_ns": execution_gpu_ns,
                "execution_dispatches": execution_dispatches,
            },
        }))
    }

    fn layer4_control_hashes(control: &Value) -> Result<Vec<String>, Box<dyn Error>> {
        control
            .get("output_state_sha256")
            .and_then(Value::as_array)
            .ok_or("layer-4 control omitted output hash vector")?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .ok_or("layer-4 control output hash was not a string")
                    .map(str::to_owned)
                    .map_err(Into::into)
            })
            .collect()
    }

    /// Cheap source-bound discriminator for the first fresh native divergence
    /// boundary.  It executes layers 0–3 exactly once to obtain real
    /// persistent layer-3 states, then reuses one dense layer-4 immutable bank
    /// across graph-construction and scratch-initialization controls.  It is
    /// deliberately not a 48-layer session, terminal check, TPS benchmark, or
    /// source-reference acceptance claim.
    fn run_layer4_handoff_initialization_controls(
        index: &SourceBf16Index,
        context: &MetalContext,
        root: &Path,
        token_ids: &[usize],
        vocab: usize,
        out_dir: &Path,
        native_invocation_admission: NativeInvocationAdmission,
        native_lane_preflight: Value,
    ) -> Result<Value, Box<dyn Error>> {
        if token_ids.len() < 2 {
            return Err("layer-4 handoff control requires at least two token slots".into());
        }
        let diagnostic_started = Instant::now();
        let source_bytes_before = index.bytes_read_total();
        let (layer2_states, linear_prefix, prefix_banks) = run_linear_segment(
            index, context, 0, 2, token_ids, vocab, None, None, false, None,
        )?;
        // The layer-3 handoff is host-owned by this bounded diagnostic; keep
        // only its measured F32 states, not unrelated lower-layer weight banks.
        drop(prefix_banks);
        let (layer3_states, layer3_receipt) =
            run_full_layer(index, context, 3, token_ids, &layer2_states, out_dir, None)?;
        if layer3_states.len() != token_ids.len() {
            return Err("layer-3 handoff control returned incomplete state coverage".into());
        }
        let first_embedding = embedding_row(index, token_ids[0], vocab)?;
        let normal_constructor_base = repeated_streams(&first_embedding);
        let alternate_constructor_base = (0..HC)
            .map(|index| if index % 2 == 0 { 0.25f32 } else { -0.5f32 })
            .collect::<Vec<_>>();
        let layer4_source_before = index.bytes_read_total();
        let mut bank =
            linear::StatefulLinearLayer::new_dense(index, context, 4, &normal_constructor_base)?;
        let layer4_source_bytes = index
            .bytes_read_total()
            .saturating_sub(layer4_source_before);
        let bank_prepare_timing = bank.prepare_timing_ns();
        let normal = run_layer4_handoff_control(
            &mut bank,
            context,
            "normal_constructor_base",
            &normal_constructor_base,
            &layer3_states,
            None,
        )?;
        let alternate = run_layer4_handoff_control(
            &mut bank,
            context,
            "alternate_finite_constructor_base",
            &alternate_constructor_base,
            &layer3_states,
            None,
        )?;
        let repeat = run_layer4_handoff_control(
            &mut bank,
            context,
            "normal_constructor_base_repeat",
            &normal_constructor_base,
            &layer3_states,
            None,
        )?;
        let poisoned = run_layer4_handoff_control(
            &mut bank,
            context,
            "normal_constructor_base_seeded_transient_scratch",
            &normal_constructor_base,
            &layer3_states,
            Some((-37.125f32).to_bits()),
        )?;
        let normal_hashes = layer4_control_hashes(&normal)?;
        let alternate_hashes = layer4_control_hashes(&alternate)?;
        let repeat_hashes = layer4_control_hashes(&repeat)?;
        let poisoned_hashes = layer4_control_hashes(&poisoned)?;
        let normal_vs_repeat = normal_hashes == repeat_hashes;
        let normal_vs_alternate = normal_hashes == alternate_hashes;
        let normal_vs_poisoned = normal_hashes == poisoned_hashes;
        let status = if !normal_vs_repeat {
            "OBSERVED_LAYER4_CONTROL_NONDETERMINISM"
        } else if !normal_vs_alternate || !normal_vs_poisoned {
            "OBSERVED_LAYER4_HANDOFF_INITIALIZATION_SENSITIVITY"
        } else {
            "PASSED_LAYER4_HANDOFF_INITIALIZATION_CONTROLS"
        };
        let first_divergent_step = |candidate: &[String]| {
            normal_hashes
                .iter()
                .zip(candidate)
                .position(|(normal, observed)| normal != observed)
        };
        let mut receipt = json!({
            "schema": "hawking.flash.layer4_handoff_initialization_control.v1",
            "status": status,
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "token_ids": token_ids,
            "execution": {
                "process_boundary": "one native Rust/Metal process",
                "invocation_admission": native_invocation_admission.receipt_value(),
                "native_lane_preflight": native_lane_preflight,
                "source_index_reused": true,
                "metal_context_reused": true,
                "source_reset_or_reprefill": false,
                "source_independent": false,
                "prefix": {
                    "layers": [0, 3],
                    "linear_layers": [0, 2],
                    "full_attention_layer": 3,
                    "handoff": "measured host F32 states from stateful native layer 3",
                    "linear_prefix": linear_prefix,
                    "layer3_receipt": layer3_receipt,
                },
                "target": {
                    "layer": 4,
                    "species": "linear_attention",
                    "immutable_weight_bank": "one dense layer-4 source-bound bank reused across all controls",
                    "persistent_state": {
                        "bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                        "reset_only_before_first_token_of_each_control": true,
                    },
                    "bank_prepare_timing_ns": {
                        "source_load_ns": bank_prepare_timing.0,
                        "device_prepare_ns": bank_prepare_timing.1,
                        "initial_graph_prepare_ns": bank_prepare_timing.2,
                    },
                    "source_payload_bytes_read_for_bank": layer4_source_bytes,
                },
                "source_payload_bytes_read_total": index
                    .bytes_read_total()
                    .saturating_sub(source_bytes_before),
                "diagnostic_wall_ns": diagnostic_started.elapsed().as_nanos() as u64,
            },
            "controls": {
                "normal": normal,
                "alternate_constructor_base": alternate,
                "normal_repeat": repeat,
                "seeded_transient_scratch": poisoned,
            },
            "comparison": {
                "normal_vs_repeat_exact": normal_vs_repeat,
                "normal_vs_alternate_constructor_exact": normal_vs_alternate,
                "normal_vs_seeded_transient_scratch_exact": normal_vs_poisoned,
                "first_divergent_step_normal_vs_repeat": first_divergent_step(&repeat_hashes),
                "first_divergent_step_normal_vs_alternate": first_divergent_step(&alternate_hashes),
                "first_divergent_step_normal_vs_seeded_transient_scratch": first_divergent_step(&poisoned_hashes),
                "interpretation": if !normal_vs_repeat {
                    "The normal construction control itself was not deterministic across exact repeated execution. Do not attribute alternate-base or seeded-scratch differences until this instability is independently reproduced and narrowed."
                } else if !normal_vs_alternate || !normal_vs_poisoned {
                    "An initialization/allocation-history sensitivity was observed at the layer-4 handoff boundary. This identifies a bounded physical candidate only; it does not identify the individual unread buffer or prove the complete-session rejection cause."
                } else {
                    "For this exact source-bound 0–3 prefix, seven-token layer-3 handoff window, and listed controls, layer-4 output was invariant to constructor base and deterministic finite scratch seeding. This rules out only those exercised initialization sensitivities; it does not establish whole-graph initialization completeness or explain the full-session rejection."
                },
            },
            "claim_boundary": "This is a bounded native layer-4 diagnostic. It does not execute the 48-layer body, run terminal acceptance, measure warmed TPS, establish EBPW, capability, residency, source independence, or promotion eligibility.",
            "promotion_allowed": false,
        });
        receipt["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&receipt)?));
        Ok(receipt)
    }

    /// The first complete Flash body with a real token-major schedule.  Every
    /// bank is constructed before token zero, then survives while all token
    /// slots advance through all 48 layers.  The route teacher remains a
    /// bounded correctness oracle; it never supplies activations or routing
    /// decisions to the native banks.
    enum ResidentTokenMajorBank {
        Linear {
            layer: usize,
            bank: linear::StatefulLinearLayer,
        },
        FullAttention {
            layer: usize,
            bank: full::StatefulFullAttentionLayer,
        },
    }

    fn run_token_major_resident_session(
        index: &SourceBf16Index,
        context: &MetalContext,
        token_ids: &[usize],
        vocab: usize,
        teacher: Option<&RouteTeacher>,
        host_attention_seam_diagnostic: bool,
        host_linear_seam_diagnostic: bool,
        resident_bank_limit: usize,
        enforce_state_hashes: bool,
        allow_candidate_route_order_drift: bool,
        mut ple_pre_layer_state_bank: Option<&mut PlePreLayerStateBankExporter>,
    ) -> Result<(Vec<Vec<f32>>, Vec<Value>, Vec<ResidentTokenMajorBank>), Box<dyn Error>> {
        require_all_48_token_major_construction(resident_bank_limit)?;
        let first_row = embedding_row(index, token_ids[0], vocab)?;
        let first_base = repeated_streams(&first_row);
        let construction_started = Instant::now();
        let mut banks = Vec::with_capacity(48);
        for layer in 0..resident_bank_limit {
            let routes = teacher
                .map(|bound| {
                    bound
                        .unions
                        .get(&layer)
                        .ok_or("route teacher omitted token-major route union")
                })
                .transpose()?;
            if FULL_LAYERS.contains(&layer) {
                banks.push(ResidentTokenMajorBank::FullAttention {
                    layer,
                    bank: full::StatefulFullAttentionLayer::new_device_only(
                        index,
                        context,
                        layer,
                        token_ids.len(),
                        routes.map(Vec::as_slice),
                    )?,
                });
            } else {
                banks.push(ResidentTokenMajorBank::Linear {
                    layer,
                    bank: if let Some(routes) = routes {
                        linear::StatefulLinearLayer::new_compact_union_device_only(
                            index,
                            context,
                            layer,
                            routes,
                            &first_base,
                        )?
                    } else {
                        linear::StatefulLinearLayer::new_dense(index, context, layer, &first_base)?
                    },
                });
            }
        }
        if banks.len() != resident_bank_limit {
            return Err(
                "token-major resident construction did not retain the requested layer count".into(),
            );
        }
        let construction_ns = construction_started.elapsed().as_nanos() as u64;
        let mut rows_by_layer = (0..48)
            .map(|_| Vec::with_capacity(token_ids.len()))
            .collect::<Vec<_>>();
        let mut final_states = Vec::with_capacity(token_ids.len());
        let execution_started = Instant::now();
        for (step, &token_id) in token_ids.iter().enumerate() {
            let embedding = embedding_row(index, token_id, vocab)?;
            let host_base = repeated_streams(&embedding);
            let mut prior_device = None;
            let mut prior_host: Option<Vec<f32>> = None;
            let mut final_state = None;
            for bank in banks.iter_mut() {
                let (layer, device_output, observed, mut row, device_inter_layer_handoff) =
                    match bank {
                        ResidentTokenMajorBank::Linear { layer, bank } => {
                            let (output, gpu_ns, wall_ns, dispatches, observed) =
                                if prior_device.is_none() {
                                    bank.step(context, Some(&host_base), None, step == 0)?
                                } else if host_linear_seam_diagnostic {
                                    let input = prior_host.as_deref().ok_or(
                                        "token-major linear layer lacked diagnostic host input",
                                    )?;
                                    bank.step(context, Some(input), None, step == 0)?
                                } else {
                                    bank.step(context, None, prior_device.as_ref(), step == 0)?
                                };
                            let route_ids = bank.route_ids();
                            (
                                *layer,
                                output,
                                observed,
                                json!({
                                    "layer": *layer,
                                    "step": step,
                                    "token_id": token_id,
                                    "dispatches": dispatches,
                                    "gpu_ns": gpu_ns,
                                    "wall_ns": wall_ns,
                                    "device_inter_layer_handoff": *layer != 0,
                                    "finite": true,
                                    "route_ids": route_ids,
                                }),
                                !host_linear_seam_diagnostic,
                            )
                        }
                        ResidentTokenMajorBank::FullAttention { layer, bank } => {
                            let (output, observed, row) = if host_attention_seam_diagnostic {
                                let input = prior_host.as_deref().ok_or(
                                    "token-major full-attention layer lacked diagnostic host input",
                                )?;
                                bank.step(context, Some(input), None, step, token_id)?
                            } else {
                                let input = prior_device.as_ref().ok_or(
                                    "token-major full-attention layer lacked device input",
                                )?;
                                bank.step(context, None, Some(input), step, token_id)?
                            };
                            (
                                *layer,
                                output,
                                observed,
                                row,
                                !host_attention_seam_diagnostic,
                            )
                        }
                    };
                if observed.len() != HC || observed.iter().any(|value| !value.is_finite()) {
                    return Err(format!("token-major resident layer {layer} produced invalid state at token slot {step}").into());
                }
                if layer == 0 {
                    if let Some(exporter) = ple_pre_layer_state_bank.as_deref_mut() {
                        exporter.record(step, token_id, &observed)?;
                    }
                }
                let route_ids = row
                    .get("route_ids")
                    .and_then(Value::as_array)
                    .ok_or("token-major resident row omitted route IDs")?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .map(|id| id as u32)
                            .ok_or("non-integer token-major route ID")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let observed_state_sha256 = sha256(&f32_bytes(&observed));
                if let Some(teacher) = teacher {
                    let expected_state_sha256 = teacher
                        .final_state_sha256
                        .get(&(layer, step))
                        .ok_or("route teacher omitted token-major state hash")?;
                    if enforce_state_hashes && observed_state_sha256 != *expected_state_sha256 {
                        return Err(format!(
                            "state drift at token-major resident layer {layer} token slot {step}: expected={expected_state_sha256} observed={observed_state_sha256}"
                        )
                        .into());
                    }
                    let expected = teacher
                        .routes
                        .get(&(layer, step))
                        .ok_or("route teacher omitted token-major row")?;
                    let same_route_set = route_ids.iter().copied().collect::<BTreeSet<_>>()
                        == expected.iter().copied().collect::<BTreeSet<_>>();
                    if route_ids != *expected
                        && !(allow_candidate_route_order_drift && same_route_set)
                    {
                        return Err(format!(
                            "route drift at token-major resident layer {layer} token slot {step}: expected={expected:?} observed={route_ids:?}"
                        )
                        .into());
                    }
                }
                let row_object = row
                    .as_object_mut()
                    .ok_or("token-major resident row was not an object")?;
                row_object.insert("layer".to_string(), Value::from(layer as u64));
                row_object.insert(
                    "device_inter_layer_handoff".to_string(),
                    Value::Bool(layer != 0 && device_inter_layer_handoff),
                );
                row_object.insert(
                    "final_state_sha256".to_string(),
                    Value::String(observed_state_sha256),
                );
                rows_by_layer[layer].push(row);
                prior_device = Some(device_output);
                prior_host = Some(observed.clone());
                final_state = Some(observed);
            }
            final_states
                .push(final_state.ok_or("token-major resident token produced no final state")?);
        }
        let execution_wall_ns = execution_started.elapsed().as_nanos() as u64;
        let mut segments = Vec::with_capacity(48);
        let segment_expert_bank_mode = if teacher.is_some() {
            "route_union_compact_teacher_bound"
        } else {
            "dense"
        };
        for bank in &banks {
            match bank {
                ResidentTokenMajorBank::Linear { layer, bank } => {
                    let rows = &rows_by_layer[*layer];
                    let (source_load_ns, device_prepare_ns, graph_prepare_ns) =
                        bank.prepare_timing_ns();
                    let device_weight_bytes = bank.resident_device_weight_bytes();
                    segments.push(json!({
                        "layers": [*layer, *layer],
                        "species": "linear_attention",
                        "stateful_recurrence": true,
                        "expert_bank_mode": segment_expert_bank_mode,
                        "physical_expert_bank_mode": bank.expert_bank_mode(),
                        "q4_compact_moe": bank.q4_compact_moe(),
                        "q8_compact_moe": bank.q8_compact_moe(),
                        "q8_residual": bank.q8_residual(),
                        "q8_residual_fraction": bank.q8_residual_fraction(),
                        "routed_expert_device_bytes": bank.routed_expert_device_bytes(),
                        "logical_routed_expert_traffic_bytes_per_token": routed_expert_traffic_bytes(bank.q4_compact_moe(), bank.q8_compact_moe(), bank.q8_residual()),
                        "logical_routed_expert_traffic_reduction_vs_compact_bf16": routed_expert_traffic_reduction(bank.q4_compact_moe(), bank.q8_compact_moe(), bank.q8_residual()),
                        "immutable_weight_ownership": "device_only_after_source_upload",
                        "host_weights_retained": bank.host_weights_retained(),
                        "state_memory": {
                            "layers": 1,
                            "per_layer_bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                            "total_bytes": linear::StatefulLinearLayer::persistent_state_bytes(),
                            "growth_bytes_per_token": 0,
                            "reset_only_before_first_session_token": true,
                        },
                        "device_inter_layer_handoffs": rows.iter().filter(|row| row.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                        "steps": rows,
                        "source_payload_bytes_read": bank.source_payload_bytes(),
                        "device_weight_bytes": device_weight_bytes,
                        "timing": {
                            "construction_total_ns": construction_ns,
                            "source_load_ns": source_load_ns,
                            "device_prepare_ns": device_prepare_ns,
                            "graph_prepare_ns": graph_prepare_ns,
                        },
                    }));
                }
                ResidentTokenMajorBank::FullAttention { layer, bank } => {
                    let rows = &rows_by_layer[*layer];
                    let (source_load_ns, device_prepare_ns, graph_prepare_ns) =
                        bank.prepare_timing_ns();
                    segments.push(json!({
                        "layer": *layer,
                        "species": "full_attention",
                        "stateful_kv": true,
                        "expert_bank_mode": segment_expert_bank_mode,
                        "physical_expert_bank_mode": bank.expert_bank_mode(),
                        "q4_compact_moe": bank.q4_compact_moe(),
                        "q8_compact_moe": bank.q8_compact_moe(),
                        "q8_residual": bank.q8_residual(),
                        "q8_residual_fraction": bank.q8_residual_fraction(),
                        "routed_expert_device_bytes": bank.routed_expert_device_bytes(),
                        "logical_routed_expert_traffic_bytes_per_token": routed_expert_traffic_bytes(bank.q4_compact_moe(), bank.q8_compact_moe(), bank.q8_residual()),
                        "logical_routed_expert_traffic_reduction_vs_compact_bf16": routed_expert_traffic_reduction(bank.q4_compact_moe(), bank.q8_compact_moe(), bank.q8_residual()),
                        "immutable_weight_ownership": "device_only_after_source_upload",
                        "host_source_weights_retained_during_token_loop": bank.host_weights_retained(),
                        "state_memory": {
                            "persistent_kv_bytes": bank.persistent_state_bytes(),
                            "growth_bytes_per_token": full::StatefulFullAttentionLayer::persistent_state_growth_bytes_per_token(),
                        },
                        "device_inter_layer_handoffs": rows.iter().filter(|row| row.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                        "steps": rows,
                        "source_payload_bytes_read": bank.source_payload_bytes(),
                        "device_weight_bytes": bank.device_weight_bytes(),
                        "timing": {
                            "construction_total_ns": construction_ns,
                            "source_load_ns": source_load_ns,
                            "device_prepare_ns": device_prepare_ns,
                            "graph_prepare_ns": graph_prepare_ns,
                        },
                    }));
                }
            }
        }
        if final_states.len() != token_ids.len() {
            return Err("token-major resident final state/token count mismatch".into());
        }
        let _ = execution_wall_ns;
        Ok((final_states, segments, banks))
    }

    /// Replays a fixed, already accepted prompt/continuation sequence through
    /// the retained body.  This deliberately runs after the exact diagnostic
    /// control: no state snapshots, route readbacks, terminal execution, or
    /// source tensor reads occur in its timed continuation window.
    fn run_token_major_clean_replay(
        banks: &mut [ResidentTokenMajorBank],
        context: &MetalContext,
        bases: &[Vec<f32>],
        token_ids: &[usize],
        prompt_len: usize,
        reps: usize,
        async_linear_submit: bool,
        linear_region_submit: bool,
        whole_token_command_buffer: bool,
    ) -> Result<Value, Box<dyn Error>> {
        if banks.len() != 48
            || bases.len() != token_ids.len()
            || prompt_len == 0
            || prompt_len >= token_ids.len()
            || reps == 0
        {
            return Err("invalid clean token-major replay configuration".into());
        }
        if whole_token_command_buffer && (async_linear_submit || linear_region_submit) {
            return Err(
                "whole-token command-buffer replay is mutually exclusive with async or region linear submission".into(),
            );
        }
        let mut token_wall_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_gpu_samples_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_dispatches = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_command_buffers = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_linear_wall_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_linear_gpu_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_linear_dispatches = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_linear_command_buffers =
            Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_full_wall_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_full_gpu_ns = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_full_dispatches = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_full_command_buffers =
            Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let mut token_time_ledger = Vec::with_capacity(reps * (token_ids.len() - prompt_len));
        let resident_device_weight_bytes = banks
            .iter()
            .map(|bank| match bank {
                ResidentTokenMajorBank::Linear { bank, .. } => bank.resident_device_weight_bytes(),
                ResidentTokenMajorBank::FullAttention { bank, .. } => bank.device_weight_bytes(),
            })
            .sum::<u64>();
        let resident_routed_expert_device_bytes = banks
            .iter()
            .map(|bank| match bank {
                ResidentTokenMajorBank::Linear { bank, .. } => bank.routed_expert_device_bytes(),
                ResidentTokenMajorBank::FullAttention { bank, .. } => {
                    bank.routed_expert_device_bytes()
                }
            })
            .sum::<u64>();
        let q4_compact_moe_banks = banks
            .iter()
            .filter(|bank| match bank {
                ResidentTokenMajorBank::Linear { bank, .. } => bank.q4_compact_moe(),
                ResidentTokenMajorBank::FullAttention { bank, .. } => bank.q4_compact_moe(),
            })
            .count();
        let q8_compact_moe_banks = banks
            .iter()
            .filter(|bank| match bank {
                ResidentTokenMajorBank::Linear { bank, .. } => bank.q8_compact_moe(),
                ResidentTokenMajorBank::FullAttention { bank, .. } => bank.q8_compact_moe(),
            })
            .count();
        let q8_residual_banks = banks
            .iter()
            .filter(|bank| match bank {
                ResidentTokenMajorBank::Linear { bank, .. } => bank.q8_residual(),
                ResidentTokenMajorBank::FullAttention { bank, .. } => bank.q8_residual(),
            })
            .count();
        let logical_routed_expert_traffic_bytes_per_token = banks
            .iter()
            .map(|bank| {
                let q4 = match bank {
                    ResidentTokenMajorBank::Linear { bank, .. } => bank.q4_compact_moe(),
                    ResidentTokenMajorBank::FullAttention { bank, .. } => bank.q4_compact_moe(),
                };
                let q8 = match bank {
                    ResidentTokenMajorBank::Linear { bank, .. } => bank.q8_compact_moe(),
                    ResidentTokenMajorBank::FullAttention { bank, .. } => bank.q8_compact_moe(),
                };
                let q8_residual = match bank {
                    ResidentTokenMajorBank::Linear { bank, .. } => bank.q8_residual(),
                    ResidentTokenMajorBank::FullAttention { bank, .. } => bank.q8_residual(),
                };
                routed_expert_traffic_bytes(q4, q8, q8_residual)
            })
            .sum::<u64>();
        let compact_bf16_routed_expert_traffic_bytes_per_token =
            banks.len() as u64 * ROUTED_EXPERT_TOP_K as u64 * ROUTED_EXPERT_ROW_BYTES;
        let logical_routed_expert_traffic_reduction_vs_compact_bf16 =
            if compact_bf16_routed_expert_traffic_bytes_per_token > 0 {
                1.0 - (logical_routed_expert_traffic_bytes_per_token as f64
                    / compact_bf16_routed_expert_traffic_bytes_per_token as f64)
            } else {
                0.0
            };
        let hyperconnection_device_weight_bytes =
            banks.len() as u64 * HYPERCONNECTION_WEIGHT_BYTES_PER_LAYER;
        let hyperconnection_share_of_resident_device_weights = if resident_device_weight_bytes > 0 {
            hyperconnection_device_weight_bytes as f64 / resident_device_weight_bytes as f64
        } else {
            0.0
        };
        let persistent_state_bytes = banks
            .iter()
            .map(|bank| match bank {
                ResidentTokenMajorBank::Linear { bank: _, .. } => {
                    linear::StatefulLinearLayer::persistent_state_bytes() as u64
                }
                ResidentTokenMajorBank::FullAttention { bank, .. } => {
                    bank.persistent_state_bytes() as u64
                }
            })
            .sum::<u64>();
        let replay_started = Instant::now();
        for repetition in 0..reps {
            for bank in banks.iter_mut() {
                if let ResidentTokenMajorBank::FullAttention { bank, .. } = bank {
                    bank.reset_state();
                }
            }
            for (step, (&token_id, host_base)) in token_ids.iter().zip(bases).enumerate() {
                let timed_decode = step >= prompt_len;
                let token_started = Instant::now();
                let mut token_gpu_ns = 0_u64;
                let mut dispatches = 0_usize;
                let mut command_buffers = 0_usize;
                let mut linear_wall_ns = 0_u64;
                let mut linear_gpu_ns = 0_u64;
                let mut linear_dispatches = 0_usize;
                let mut linear_command_buffers = 0_usize;
                let mut full_wall_ns = 0_u64;
                let mut full_gpu_ns = 0_u64;
                let mut full_dispatches = 0_usize;
                let mut full_command_buffers = 0_usize;
                let mut prior_device = None;
                let mut bank_index = 0_usize;
                if whole_token_command_buffer {
                    let mut token_buffer = TokenCommandBuffer::new(context);
                    while bank_index < banks.len() {
                        let (linear_bank, layer_dispatches, output) = match banks
                            .get_mut(bank_index)
                            .ok_or("token-major scheduler bank index drifted")?
                        {
                            ResidentTokenMajorBank::Linear { bank, .. } => {
                                let layer_dispatches = if prior_device.is_none() {
                                    bank.encode_step_into_clean_fast(
                                        context,
                                        &mut token_buffer,
                                        Some(host_base),
                                        None,
                                        step == 0,
                                    )?
                                } else {
                                    bank.encode_step_into_clean_fast(
                                        context,
                                        &mut token_buffer,
                                        None,
                                        prior_device.as_ref(),
                                        step == 0,
                                    )?
                                };
                                (true, layer_dispatches, bank.final_state_buffer())
                            }
                            ResidentTokenMajorBank::FullAttention { bank, .. } => {
                                let previous = prior_device
                                    .as_ref()
                                    .ok_or("whole-token scheduler reached full attention without a device predecessor")?;
                                let layer_dispatches = bank.encode_step_into(
                                    context,
                                    &mut token_buffer,
                                    None,
                                    Some(previous),
                                    step,
                                    token_id,
                                )?;
                                (false, layer_dispatches, bank.final_state_buffer())
                            }
                        };
                        dispatches = dispatches.saturating_add(layer_dispatches);
                        if linear_bank {
                            linear_dispatches = linear_dispatches.saturating_add(layer_dispatches);
                        } else {
                            full_dispatches = full_dispatches.saturating_add(layer_dispatches);
                        }
                        prior_device = Some(output);
                        bank_index += 1;
                    }
                    let timing = token_buffer.commit_and_wait_timed()?;
                    token_gpu_ns = timing.gpu_ns.unwrap_or(0);
                    command_buffers = 1;
                } else {
                    while bank_index < banks.len() {
                        if linear_region_submit
                            && matches!(
                                banks.get(bank_index),
                                Some(ResidentTokenMajorBank::Linear { .. })
                            )
                        {
                            let mut region = TokenCommandBuffer::new(context);
                            while let Some(ResidentTokenMajorBank::Linear { bank, .. }) =
                                banks.get_mut(bank_index)
                            {
                                let layer_dispatches = if prior_device.is_none() {
                                    bank.encode_step_into_clean_fast(
                                        context,
                                        &mut region,
                                        Some(host_base),
                                        None,
                                        step == 0,
                                    )?
                                } else {
                                    bank.encode_step_into_clean_fast(
                                        context,
                                        &mut region,
                                        None,
                                        prior_device.as_ref(),
                                        step == 0,
                                    )?
                                };
                                dispatches = dispatches.saturating_add(layer_dispatches);
                                linear_dispatches =
                                    linear_dispatches.saturating_add(layer_dispatches);
                                prior_device = Some(bank.final_state_buffer());
                                bank_index += 1;
                            }
                            region.commit_no_wait()?;
                            command_buffers = command_buffers.saturating_add(1);
                            linear_command_buffers = linear_command_buffers.saturating_add(1);
                            continue;
                        }
                        let linear_bank = !FULL_LAYERS.contains(&bank_index);
                        let bank = banks
                            .get_mut(bank_index)
                            .ok_or("token-major scheduler bank index drifted")?;
                        let (output, gpu_ns, wall_ns, layer_dispatches) = match bank {
                            ResidentTokenMajorBank::Linear { bank, .. }
                                if async_linear_submit && prior_device.is_none() =>
                            {
                                let (output, dispatches) = bank.step_submit_fast(
                                    context,
                                    Some(host_base),
                                    None,
                                    step == 0,
                                )?;
                                (output, 0, 0, dispatches)
                            }
                            ResidentTokenMajorBank::Linear { bank, .. } if async_linear_submit => {
                                let (output, dispatches) = bank.step_submit_fast(
                                    context,
                                    None,
                                    prior_device.as_ref(),
                                    step == 0,
                                )?;
                                (output, 0, 0, dispatches)
                            }
                            ResidentTokenMajorBank::Linear { bank, .. }
                                if prior_device.is_none() =>
                            {
                                bank.step_fast(context, Some(host_base), None, step == 0)?
                            }
                            ResidentTokenMajorBank::Linear { bank, .. } => {
                                bank.step_fast(context, None, prior_device.as_ref(), step == 0)?
                            }
                            ResidentTokenMajorBank::FullAttention { bank, .. } => bank.step_fast(
                                context,
                                None,
                                prior_device.as_ref(),
                                step,
                                token_id,
                            )?,
                        };
                        token_gpu_ns = token_gpu_ns.saturating_add(gpu_ns);
                        dispatches = dispatches.saturating_add(layer_dispatches);
                        if linear_bank {
                            linear_gpu_ns = linear_gpu_ns.saturating_add(gpu_ns);
                            linear_wall_ns = linear_wall_ns.saturating_add(wall_ns);
                            linear_dispatches = linear_dispatches.saturating_add(layer_dispatches);
                            linear_command_buffers = linear_command_buffers.saturating_add(1);
                        } else {
                            full_gpu_ns = full_gpu_ns.saturating_add(gpu_ns);
                            full_wall_ns = full_wall_ns.saturating_add(wall_ns);
                            full_dispatches = full_dispatches.saturating_add(layer_dispatches);
                            full_command_buffers = full_command_buffers.saturating_add(1);
                        }
                        prior_device = Some(output);
                        command_buffers = command_buffers.saturating_add(1);
                        bank_index += 1;
                    }
                }
                if timed_decode {
                    let wall_ns = token_started.elapsed().as_nanos() as u64;
                    token_wall_ns.push(wall_ns);
                    token_gpu_samples_ns.push(token_gpu_ns);
                    token_dispatches.push(dispatches);
                    token_command_buffers.push(command_buffers);
                    token_linear_wall_ns.push(linear_wall_ns);
                    token_linear_gpu_ns.push(linear_gpu_ns);
                    token_linear_dispatches.push(linear_dispatches);
                    token_linear_command_buffers.push(linear_command_buffers);
                    token_full_wall_ns.push(full_wall_ns);
                    token_full_gpu_ns.push(full_gpu_ns);
                    token_full_dispatches.push(full_dispatches);
                    token_full_command_buffers.push(full_command_buffers);
                    token_time_ledger.push(json!({
                        "repetition": repetition,
                        "token_slot": step,
                        "token_id": token_id,
                        "wall_ns": wall_ns,
                        "gpu_ns": token_gpu_ns,
                        "dispatches": dispatches,
                        "command_buffers": command_buffers,
                        "linear_attention": {
                            "wall_ns": linear_wall_ns,
                            "gpu_ns": linear_gpu_ns,
                            "dispatches": linear_dispatches,
                            "command_buffers": linear_command_buffers,
                        },
                        "full_attention": {
                            "wall_ns": full_wall_ns,
                            "gpu_ns": full_gpu_ns,
                            "dispatches": full_dispatches,
                            "command_buffers": full_command_buffers,
                        },
                        "wall_remainder_ns": wall_ns
                            .saturating_sub(linear_wall_ns.saturating_add(full_wall_ns)),
                    }));
                }
            }
        }
        let mut sorted_wall = token_wall_ns.clone();
        sorted_wall.sort_unstable();
        let median_wall_ns = sorted_wall[sorted_wall.len() / 2];
        let mean_wall_ns = token_wall_ns.iter().sum::<u64>() / token_wall_ns.len() as u64;
        let mean_gpu_ns =
            token_gpu_samples_ns.iter().sum::<u64>() / token_gpu_samples_ns.len() as u64;
        let mean_dispatches = token_dispatches.iter().sum::<usize>() / token_dispatches.len();
        let mean_command_buffers =
            token_command_buffers.iter().sum::<usize>() / token_command_buffers.len();
        let timed_tokens = token_wall_ns.len() as u64;
        let mean_linear_wall_ns = token_linear_wall_ns.iter().sum::<u64>() / timed_tokens;
        let mean_linear_gpu_ns = token_linear_gpu_ns.iter().sum::<u64>() / timed_tokens;
        let mean_linear_dispatches =
            token_linear_dispatches.iter().sum::<usize>() as u64 / timed_tokens;
        let mean_linear_command_buffers =
            token_linear_command_buffers.iter().sum::<usize>() as u64 / timed_tokens;
        let mean_full_wall_ns = token_full_wall_ns.iter().sum::<u64>() / timed_tokens;
        let mean_full_gpu_ns = token_full_gpu_ns.iter().sum::<u64>() / timed_tokens;
        let mean_full_dispatches =
            token_full_dispatches.iter().sum::<usize>() as u64 / timed_tokens;
        let mean_full_command_buffers =
            token_full_command_buffers.iter().sum::<usize>() as u64 / timed_tokens;
        let mean_wall_remainder_ns = token_time_ledger
            .iter()
            .filter_map(|row| row.get("wall_remainder_ns").and_then(Value::as_u64))
            .sum::<u64>()
            / timed_tokens;
        let species_timing = if whole_token_command_buffer {
            "aggregate: one whole-token command buffer contains all banks, so per-species wall/GPU intervals are intentionally not attributed"
        } else if async_linear_submit || linear_region_submit {
            "partial: linear no-wait submissions have no independent completed GPU/wall interval; full-attention intervals include queue drain"
        } else {
            "complete per-bank baseline accounting; bank wall intervals include encode, submission, wait, and completion"
        };
        Ok(json!({
            "status": "MEASURED_CLEAN_BOUNDED_REPLAY",
            "repetitions": reps,
            "prompt_tokens_replayed_untimed": prompt_len * reps,
            "timed_known_continuation_tokens": token_wall_ns.len(),
            "state_reset": "full-attention KV buffers explicitly zeroed before every replay; linear recurrent state reset at token zero",
            "timed_window": "retained device-only banks; no source tensor reads, host activation snapshots, route readback/comparison, or terminal execution",
            "scheduler": if whole_token_command_buffer { "all 48 resident banks are encoded in layer order into one queue-ordered command buffer per token, followed by one fence" } else if linear_region_submit { "each contiguous linear region is encoded into one queue-ordered command buffer; the following full-attention command buffer drains it" } else if async_linear_submit { "linear banks commit without per-layer CPU wait; succeeding full-attention command buffers drain the ordered Metal queue" } else { "every resident bank commits and waits before the next bank" },
            "async_linear_submit": async_linear_submit || linear_region_submit,
            "linear_region_submit": linear_region_submit,
            "whole_token_command_buffer": whole_token_command_buffer,
            "median_token_wall_ns": median_wall_ns,
            "mean_token_wall_ns": mean_wall_ns,
            "mean_token_gpu_ns": mean_gpu_ns,
            "gpu_timing_attribution": if whole_token_command_buffer { "aggregate: the command-buffer GPU interval covers all linear and full-attention banks; token wall and aggregate GPU are authoritative, per-species GPU/wall attribution is withheld" } else if async_linear_submit || linear_region_submit { "partial: only synchronously drained full-attention command buffers contribute per-layer GPU timings; token wall is the authoritative scheduler metric" } else { "sum of per-layer completed command-buffer GPU intervals" },
            "mean_dispatches_per_token": mean_dispatches,
            "mean_command_buffers_per_token": mean_command_buffers,
            "token_time_ledger": {
                "coverage": {
                    "timed_repetitions": reps,
                    "timed_token_count": token_wall_ns.len(),
                    "timed_token_slots": (prompt_len..token_ids.len()).collect::<Vec<_>>(),
                    "token_order": "fixed caller-supplied sequence",
                },
                "species_timing": species_timing,
                "linear_attention": {
                    "mean_wall_ns": mean_linear_wall_ns,
                    "mean_gpu_ns": mean_linear_gpu_ns,
                    "mean_dispatches": mean_linear_dispatches,
                    "mean_command_buffers": mean_linear_command_buffers,
                },
                "full_attention": {
                    "mean_wall_ns": mean_full_wall_ns,
                    "mean_gpu_ns": mean_full_gpu_ns,
                    "mean_dispatches": mean_full_dispatches,
                    "mean_command_buffers": mean_full_command_buffers,
                },
                "token_envelope": {
                    "mean_wall_ns": mean_wall_ns,
                    "mean_gpu_ns": mean_gpu_ns,
                    "mean_wall_remainder_ns": mean_wall_remainder_ns,
                },
                "events": token_time_ledger,
            },
            "active_bytes_accounting": {
                "resident_device_weight_bytes": resident_device_weight_bytes,
                "resident_device_weight_bytes_per_timed_token": resident_device_weight_bytes,
                "q4_compact_moe_banks": q4_compact_moe_banks,
                "q8_compact_moe_banks": q8_compact_moe_banks,
                "q8_residual_banks": q8_residual_banks,
                "q8_residual_fraction": if q8_residual_banks > 0 { q8_residual_fraction() } else { 0.0 },
                "resident_routed_expert_device_bytes": resident_routed_expert_device_bytes,
                "logical_routed_expert_traffic_bytes_per_token": logical_routed_expert_traffic_bytes_per_token,
                "logical_routed_expert_traffic_reduction_vs_compact_bf16": logical_routed_expert_traffic_reduction_vs_compact_bf16,
                "hyperconnection_device_weight_bytes": hyperconnection_device_weight_bytes,
                "hyperconnection_share_of_resident_device_weights": hyperconnection_share_of_resident_device_weights,
                "persistent_state_bytes": persistent_state_bytes,
                "source_reads_in_timed_window_bytes": 0,
                "source_uploads_in_timed_window_bytes": 0,
                "physical_expert_cache_traffic_bytes_per_token": Value::Null,
                "hardware_cache_traffic_measured": false,
                "claim_boundary": "Resident immutable weights and persistent state are measured from the owning banks. Selected-expert geometry and hardware cache/fetch traffic are separate quantities; this receipt does not observe hardware cache traffic or claim a physical bytes-fetched-per-token value.",
            },
            "bounded_replay_decode_tps": 1_000_000_000_f64 / mean_wall_ns as f64,
            "qualified_canonical_decode_tps": Value::Null,
            "claim_boundary": "This is a bounded known-sequence replay over a body that previously passed exact state/route/token checks. It is not capability-qualified sustained decode TPS, future-route coverage, EBPW, or promotion evidence.",
            "replay_wall_ns_including_untimed_prompt": replay_started.elapsed().as_nanos() as u64,
        }))
    }

    /// Collapse the opt-in production-CB timestamp stream into a small
    /// kernel-level attribution table.  The stream is drained only for a
    /// dedicated diagnostic run; ordinary replay receipts remain free of
    /// trace overhead and retain their measured wall/GPU authority.
    fn summarize_dispatch_trace(
        samples: Vec<hawking_core::metal::DispatchSample>,
        mode: Option<String>,
    ) -> Value {
        let mut by_kernel: BTreeMap<String, (u64, u64, u64, Option<u64>, Option<u64>)> =
            BTreeMap::new();
        let mut gpu_sample_count = 0_u64;
        let mut gpu_total_us = 0_u64;
        for sample in samples {
            let entry = by_kernel
                .entry(sample.kernel_name.to_string())
                .or_insert((0, 0, 0, None, None));
            entry.0 = entry.0.saturating_add(1);
            entry.1 = entry.1.saturating_add(sample.wall_us);
            if let Some(gpu_us) = sample.gpu_us {
                entry.2 = entry.2.saturating_add(gpu_us);
                entry.3 = Some(entry.3.map_or(gpu_us, |min| min.min(gpu_us)));
                entry.4 = Some(entry.4.map_or(gpu_us, |max| max.max(gpu_us)));
                gpu_sample_count = gpu_sample_count.saturating_add(1);
                gpu_total_us = gpu_total_us.saturating_add(gpu_us);
            }
        }
        let mut kernels = by_kernel
            .into_iter()
            .map(
                |(kernel, (count, wall_total_us, gpu_total_us, gpu_min_us, gpu_max_us))| {
                    json!({
                        "kernel": kernel,
                        "samples": count,
                        "wall_total_us": wall_total_us,
                        "gpu_samples": gpu_min_us.is_some().then_some(count),
                        "gpu_total_us": gpu_total_us,
                        "gpu_mean_us": gpu_min_us.map(|_| gpu_total_us as f64 / count as f64),
                        "gpu_min_us": gpu_min_us,
                        "gpu_max_us": gpu_max_us,
                    })
                },
            )
            .collect::<Vec<_>>();
        kernels.sort_by(|left, right| {
            right
                .get("gpu_total_us")
                .and_then(Value::as_u64)
                .cmp(&left.get("gpu_total_us").and_then(Value::as_u64))
                .then_with(|| {
                    left.get("kernel")
                        .and_then(Value::as_str)
                        .cmp(&right.get("kernel").and_then(Value::as_str))
                })
        });
        json!({
            "mode": mode,
            "sample_count": kernels.iter().map(|row| row.get("samples").and_then(Value::as_u64).unwrap_or(0)).sum::<u64>(),
            "gpu_sample_count": gpu_sample_count,
            "gpu_total_us": gpu_total_us,
            "kernels_sorted_by_gpu_total": kernels,
            "claim_boundary": "Opt-in production-CB timestamp attribution for this bounded run only. Counter markers add diagnostic work; these timings are not the clean replay TPS authority and do not establish a promotion claim.",
        })
    }

    fn terminal_token(
        executor: &terminal::TerminalExecutor,
        state: &[f32],
        path: &Path,
    ) -> Result<(usize, Value), Box<dyn Error>> {
        write_state(path, state)?;
        let receipt = path.with_extension("terminal.json");
        let (token, _doc) = executor
            .run_state(&path.to_path_buf(), Some(receipt.clone()))
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        Ok((
            token,
            json!({"token_id": token, "receipt": receipt, "state": path}),
        ))
    }

    fn main_impl() -> Result<(), Box<dyn Error>> {
        let argv: Vec<String> = env::args().collect();
        let native_invocation_admission = require_native_invocation_admission(&argv)?;
        require_wrapper_admitted_native_execution(native_invocation_admission)?;
        let supervising_launcher_pid =
            supervising_launcher_pid(&argv, native_invocation_admission)?;
        let boundary_capture_kind = NativeBoundaryCaptureKind::parse(&argv)?;
        let capture_native_boundary = boundary_capture_kind.is_some();
        let attention_hc_source_bf16_norm_flags = argv
            .iter()
            .filter(|argument| argument.as_str() == "--diagnostic-attention-hc-source-bf16-norm")
            .count();
        if attention_hc_source_bf16_norm_flags > 1 {
            return Err(
                "--diagnostic-attention-hc-source-bf16-norm may appear at most once".into(),
            );
        }
        let attention_hc_norm_full_pairwise_flags = argv
            .iter()
            .filter(|argument| argument.as_str() == "--diagnostic-attention-hc-norm-full-pairwise")
            .count();
        if attention_hc_norm_full_pairwise_flags > 1 {
            return Err(
                "--diagnostic-attention-hc-norm-full-pairwise may appear at most once".into(),
            );
        }
        let attention_hc_source_bf16_stage_flags = argv
            .iter()
            .filter(|argument| argument.as_str() == "--diagnostic-attention-hc-source-bf16-stages")
            .count();
        if attention_hc_source_bf16_stage_flags > 1 {
            return Err(
                "--diagnostic-attention-hc-source-bf16-stages may appear at most once".into(),
            );
        }
        let attention_hc_norm_threadgroup_flags = argv
            .iter()
            .filter(|argument| argument.as_str() == "--diagnostic-attention-hc-norm-threadgroup")
            .count();
        if attention_hc_norm_threadgroup_flags > 1 {
            return Err(
                "--diagnostic-attention-hc-norm-threadgroup may appear at most once".into(),
            );
        }
        let attention_hc_norm_threadgroup_size = if attention_hc_norm_threadgroup_flags == 1 {
            let raw = arg_value(&argv, "--diagnostic-attention-hc-norm-threadgroup")
                .ok_or("--diagnostic-attention-hc-norm-threadgroup requires a value")?;
            let parsed = raw.parse::<u32>().map_err(|_| {
                "--diagnostic-attention-hc-norm-threadgroup requires an integer value"
            })?;
            if !matches!(parsed, 128 | 256) {
                return Err(
                    "--diagnostic-attention-hc-norm-threadgroup accepts only 128 or 256".into(),
                );
            }
            Some(parsed)
        } else {
            None
        };
        if attention_hc_norm_threadgroup_size.is_some()
            && attention_hc_source_bf16_norm_flags != 1
            && attention_hc_source_bf16_stage_flags != 1
        {
            return Err(
                "--diagnostic-attention-hc-norm-threadgroup requires a source-BF16 attention diagnostic"
                    .into(),
            );
        }
        if attention_hc_norm_full_pairwise_flags == 1
            && attention_hc_source_bf16_norm_flags != 1
            && attention_hc_source_bf16_stage_flags != 1
        {
            return Err(
                "--diagnostic-attention-hc-norm-full-pairwise requires a source-BF16 attention diagnostic"
                    .into(),
            );
        }
        if attention_hc_norm_full_pairwise_flags == 1
            && attention_hc_norm_threadgroup_size.is_some()
        {
            return Err(
                "--diagnostic-attention-hc-norm-full-pairwise may not be combined with --diagnostic-attention-hc-norm-threadgroup"
                    .into(),
            );
        }
        if (attention_hc_source_bf16_norm_flags == 1
            || attention_hc_source_bf16_stage_flags == 1
            || attention_hc_norm_threadgroup_size.is_some()
            || attention_hc_norm_full_pairwise_flags == 1)
            && !capture_native_boundary
        {
            return Err(
                "attention-HC precision diagnostics are allowed only with --capture-source-boundary or --capture-layer0-hc-trace-only".into(),
            );
        }
        let attention_hc_norm_precision = if attention_hc_source_bf16_stage_flags == 1 {
            if attention_hc_norm_full_pairwise_flags == 1 {
                linear::HcNormOutputPrecision::SourceBf16RoundTripAllAttentionStagesFullPairwise
            } else {
                linear::HcNormOutputPrecision::SourceBf16RoundTripAllAttentionStages {
                    norm_threadgroup_size: attention_hc_norm_threadgroup_size.unwrap_or(256),
                }
            }
        } else if attention_hc_source_bf16_norm_flags == 1 {
            if attention_hc_norm_full_pairwise_flags == 1 {
                linear::HcNormOutputPrecision::SourceBf16RoundTripFullPairwise
            } else {
                linear::HcNormOutputPrecision::SourceBf16RoundTrip {
                    norm_threadgroup_size: attention_hc_norm_threadgroup_size.unwrap_or(256),
                }
            }
        } else {
            linear::HcNormOutputPrecision::NativeF32
        };
        let resident_bank_limit = arg_value(&argv, "--token-major-construction-limit")
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(48);
        if !capture_native_boundary {
            require_all_48_token_major_construction(resident_bank_limit)?;
        }
        let root = PathBuf::from(arg_value(&argv, "--root").unwrap_or_else(|| {
            env::var("HCLI_FLASH_NEXT_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_owned())
        }))
        .canonicalize()?;
        let out = absolute_lexical_path(&PathBuf::from(arg_value(&argv, "--out").unwrap_or_else(
            || {
                repo_root()
                    .join("receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json")
                    .display()
                    .to_string()
            },
        )))?;
        let token_ids = parse_token_ids(&argv)?;
        if let Some(capture_kind) = boundary_capture_kind {
            if arg_value(&argv, "--out").is_none() {
                return Err(format!(
                    "{} requires an explicit unique --out receipt path",
                    capture_kind.flag()
                )
                .into());
            }
            if token_ids != [PROMPT_IDS[0]] {
                return Err(
                    format!("{} requires exactly --token-ids 5423", capture_kind.flag()).into(),
                );
            }
            let source_capture_sha256 = arg_value(&argv, "--source-boundary-capture-sha256")
                .ok_or("native boundary capture requires --source-boundary-capture-sha256")?;
            let preflight_sha256 = arg_value(&argv, "--source-boundary-preflight-sha256")
                .ok_or("native boundary capture requires --source-boundary-preflight-sha256")?;
            if argv.iter().any(|argument| {
                matches!(
                    argument.as_str(),
                    "--diagnose-layer4-handoff"
                        | "--retain-linear-banks"
                        | "--device-only-compact-banks"
                        | "--token-major-resident-banks"
                        | "--token-major-host-attention-seam"
                        | "--token-major-host-linear-seam"
                        | "--token-major-allow-state-drift-candidate"
                        | "--ple-pre-layer-state-bank-out"
                        | "--route-teacher"
                )
            }) {
                return Err("native boundary capture cannot combine full-session, route-teacher, replay, PLE-export, or layer-4 initialization-control flags".into());
            }
            // Validate lineage before output reservation, source-index
            // construction, or Metal setup. The trace-only path uses the same
            // sealed capture/preflight lineage without accepting a PLE payload.
            validate_source_boundary_lineage(&source_capture_sha256, &preflight_sha256)?;
            let source_ple_binding = if capture_kind.requires_source_ple_state() {
                let source_post_ple_state = PathBuf::from(
                    arg_value(&argv, "--source-post-ple-state")
                        .ok_or("--capture-source-boundary requires --source-post-ple-state")?,
                );
                let source_post_ple_sha256 = arg_value(&argv, "--source-post-ple-state-sha256")
                    .ok_or("--capture-source-boundary requires --source-post-ple-state-sha256")?;
                // Preserve the full capture's pre-reservation validation
                // order: a malformed or changed teacher payload may not leave
                // a fresh receipt namespace behind.
                let _ = read_bound_f32_state(
                    &source_post_ple_state,
                    &source_post_ple_sha256,
                    HC,
                    "owner-bound source post-PLE token-0 state",
                )?;
                Some((source_post_ple_state, source_post_ple_sha256))
            } else {
                if argv.iter().any(|argument| {
                    matches!(
                        argument.as_str(),
                        "--source-post-ple-state" | "--source-post-ple-state-sha256"
                    )
                }) {
                    return Err("--capture-layer0-hc-trace-only does not accept a source PLE state; it is intentionally limited to native layer 0".into());
                }
                None
            };
            let source_input_identity = source_input_identity(&root)?;
            let native_lane_preflight =
                require_clean_protected_native_lane(supervising_launcher_pid)?;
            let diagnostic_index = SourceBf16Index::open(&root)?;
            let out_dir = reserve_session_output(&out)?;
            let diagnostic_context = MetalContext::new_with_trace(true)?;
            let mut receipt =
                if let Some((source_post_ple_state, source_post_ple_sha256)) = source_ple_binding {
                    run_native_source_boundary_capture(
                        &diagnostic_index,
                        &diagnostic_context,
                        &root,
                        PROMPT_IDS[0],
                        &source_post_ple_state,
                        &source_post_ple_sha256,
                        &source_capture_sha256,
                        &preflight_sha256,
                        &out_dir,
                        native_invocation_admission,
                        native_lane_preflight,
                        source_input_identity,
                        attention_hc_norm_precision,
                    )?
                } else {
                    run_native_layer0_attention_hc_trace_capture(
                        &diagnostic_index,
                        &diagnostic_context,
                        &root,
                        PROMPT_IDS[0],
                        &source_capture_sha256,
                        &preflight_sha256,
                        &out_dir,
                        native_invocation_admission,
                        native_lane_preflight,
                        source_input_identity,
                        attention_hc_norm_precision,
                    )?
                };
            receipt["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&receipt)?));
            write_new_receipt(&out, &serde_json::to_vec_pretty(&receipt)?)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
            return Ok(());
        }
        if token_ids.len() < 2 {
            return Err("complete session requires prompt plus candidate token".into());
        }
        let prompt_len = parse_prompt_len(&argv, token_ids.len())?;
        let source_input_identity = source_input_identity(&root)?;
        let token_major_allow_state_drift_candidate = argv
            .iter()
            .any(|arg| arg == "--token-major-allow-state-drift-candidate");
        let route_teacher = parse_route_teacher(
            &argv,
            &token_ids,
            prompt_len,
            &root,
            &source_input_identity,
            token_major_allow_state_drift_candidate,
        )?;
        let diagnose_layer4_handoff = argv.iter().any(|arg| arg == "--diagnose-layer4-handoff");
        let retain_linear_banks = argv.iter().any(|arg| arg == "--retain-linear-banks");
        let device_only_compact_banks = argv.iter().any(|arg| arg == "--device-only-compact-banks");
        let token_major_resident_banks =
            argv.iter().any(|arg| arg == "--token-major-resident-banks");
        let token_major_host_attention_seam = argv
            .iter()
            .any(|arg| arg == "--token-major-host-attention-seam");
        let token_major_host_linear_seam = argv
            .iter()
            .any(|arg| arg == "--token-major-host-linear-seam");
        require_token_major_route_teacher(
            route_teacher.is_some(),
            token_major_resident_banks,
            token_major_host_attention_seam,
            token_major_host_linear_seam,
        )?;
        let clean_replay_reps = arg_value(&argv, "--token-major-clean-replay-reps")
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let ple_pre_layer_state_bank_out = arg_value(&argv, "--ple-pre-layer-state-bank-out")
            .map(PathBuf::from)
            .map(|path| {
                if path.is_absolute() {
                    path
                } else {
                    repo_root().join(path)
                }
            })
            .map(|path| absolute_lexical_path(&path))
            .transpose()?;
        if device_only_compact_banks && route_teacher.is_none() {
            return Err("--device-only-compact-banks requires --route-teacher".into());
        }
        if token_major_resident_banks
            && route_teacher.is_some()
            && (!retain_linear_banks || !device_only_compact_banks)
        {
            return Err("teacher-bound --token-major-resident-banks requires --retain-linear-banks --device-only-compact-banks".into());
        }
        if token_major_host_attention_seam && !token_major_resident_banks {
            return Err(
                "--token-major-host-attention-seam requires --token-major-resident-banks".into(),
            );
        }
        if token_major_host_linear_seam && !token_major_resident_banks {
            return Err(
                "--token-major-host-linear-seam requires --token-major-resident-banks".into(),
            );
        }
        if clean_replay_reps > 0 && !token_major_resident_banks {
            return Err(
                "--token-major-clean-replay-reps requires --token-major-resident-banks".into(),
            );
        }
        if token_major_allow_state_drift_candidate
            && (!token_major_resident_banks || route_teacher.is_none())
        {
            return Err("--token-major-allow-state-drift-candidate requires --token-major-resident-banks and --route-teacher".into());
        }
        if diagnose_layer4_handoff {
            if arg_value(&argv, "--out").is_none() {
                return Err(
                    "--diagnose-layer4-handoff requires an explicit unique --out receipt path"
                        .into(),
                );
            }
            if route_teacher.is_some()
                || retain_linear_banks
                || device_only_compact_banks
                || token_major_resident_banks
                || token_major_host_attention_seam
                || token_major_host_linear_seam
                || token_major_allow_state_drift_candidate
                || clean_replay_reps > 0
                || ple_pre_layer_state_bank_out.is_some()
            {
                return Err("--diagnose-layer4-handoff is a dense native 0-4 control and cannot combine route-teacher, resident, compact, replay, or PLE flags".into());
            }
            let native_lane_preflight =
                require_clean_protected_native_lane(supervising_launcher_pid)?;
            let diagnostic_vocab = vocab(&root)?;
            let diagnostic_index = SourceBf16Index::open(&root)?;
            let out_dir = reserve_session_output(&out)?;
            let diagnostic_context = MetalContext::new_with_trace(true)?;
            let receipt = run_layer4_handoff_initialization_controls(
                &diagnostic_index,
                &diagnostic_context,
                &root,
                &token_ids,
                diagnostic_vocab,
                &out_dir,
                native_invocation_admission,
                native_lane_preflight,
            )?;
            write_new_receipt(&out, &serde_json::to_vec_pretty(&receipt)?)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
            return Ok(());
        }
        let prospective_out_dir = session_artifact_dir(&out);
        if let Some(ple_output_dir) = ple_pre_layer_state_bank_out.as_deref() {
            validate_ple_pre_layer_state_bank_output(ple_output_dir, &out, &prospective_out_dir)?;
        }
        // Observe the protected lane before constructing a source index or
        // Metal context.  Then validate all no-model source inputs before
        // reserving a fresh native output namespace, so malformed source
        // inputs cannot leave an empty receipt/artifact marker behind.
        let native_lane_preflight = require_clean_protected_native_lane(supervising_launcher_pid)?;
        let vocab = vocab(&root)?;
        let linear_index = SourceBf16Index::open(&root)?;
        let out_dir = reserve_session_output(&out)?;
        let mut ple_pre_layer_state_bank = ple_pre_layer_state_bank_out
            .map(PlePreLayerStateBankExporter::new)
            .transpose()?;
        // Keep the immutable source index and Metal context alive for the
        // entire native session. Linear segments still release their layer
        // weights at species seams, but no longer reopen the source or create
        // a fresh Metal device/queue for every segment.
        let linear_context = MetalContext::new_with_trace(true)?;
        let started = Instant::now();
        let mut retained_linear_banks: Vec<Vec<linear::StatefulLinearLayer>> = Vec::new();
        let mut token_major_banks: Option<Vec<ResidentTokenMajorBank>> = None;
        let (final_states, segments) = if token_major_resident_banks {
            let (states, receipts, banks) = run_token_major_resident_session(
                &linear_index,
                &linear_context,
                &token_ids,
                vocab,
                route_teacher.as_ref(),
                token_major_host_attention_seam,
                token_major_host_linear_seam,
                resident_bank_limit,
                !token_major_allow_state_drift_candidate,
                token_major_allow_state_drift_candidate,
                ple_pre_layer_state_bank.as_mut(),
            )?;
            token_major_banks = Some(banks);
            (states, receipts)
        } else {
            let mut current_states: Option<Vec<Vec<f32>>> = None;
            let mut segments = Vec::new();
            let mut cursor = 0usize;
            for &full_layer in FULL_LAYERS.iter() {
                if cursor < full_layer {
                    let end = full_layer - 1;
                    let (states, receipt, banks) = run_linear_segment(
                        &linear_index,
                        &linear_context,
                        cursor,
                        end,
                        &token_ids,
                        vocab,
                        current_states.as_deref(),
                        route_teacher.as_ref(),
                        device_only_compact_banks,
                        ple_pre_layer_state_bank.as_mut(),
                    )?;
                    current_states = Some(states);
                    segments.push(receipt);
                    if retain_linear_banks {
                        retained_linear_banks.push(banks);
                    }
                }
                let input = current_states.take().ok_or(format!(
                    "missing input state before full-attention layer {full_layer}"
                ))?;
                let (states, receipt) = run_full_layer(
                    &linear_index,
                    &linear_context,
                    full_layer,
                    &token_ids,
                    &input,
                    &out_dir,
                    route_teacher.as_ref(),
                )?;
                current_states = Some(states);
                segments.push(receipt);
                cursor = full_layer + 1;
            }
            if cursor < 48 {
                let (states, receipt, banks) = run_linear_segment(
                    &linear_index,
                    &linear_context,
                    cursor,
                    47,
                    &token_ids,
                    vocab,
                    current_states.as_deref(),
                    route_teacher.as_ref(),
                    device_only_compact_banks,
                    ple_pre_layer_state_bank.as_mut(),
                )?;
                current_states = Some(states);
                segments.push(receipt);
                if retain_linear_banks {
                    retained_linear_banks.push(banks);
                }
            }
            (
                current_states.ok_or("complete session produced no final states")?,
                segments,
            )
        };
        if final_states.len() != token_ids.len() {
            return Err("complete session final state/token count mismatch".into());
        }
        let terminal_dir = out_dir.join("terminal");
        match fs::create_dir(&terminal_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(format!(
                    "terminal artifact directory became occupied while this native attempt was running: {}",
                    terminal_dir.display()
                )
                .into())
            }
            Err(error) => return Err(error.into()),
        }
        let terminal_executor = terminal::TerminalExecutor::new(root.clone())?;
        let mut reference_checks = Vec::new();
        for (generation_index, &expected_token) in token_ids[prompt_len..].iter().enumerate() {
            let state_index = prompt_len - 1 + generation_index;
            let (predicted_token, terminal) = terminal_token(
                &terminal_executor,
                &final_states[state_index],
                &terminal_dir.join(format!("reference-{generation_index}-input.state.f32")),
            )?;
            reference_checks.push(json!({
                "generation_index": generation_index,
                "input_state_index": state_index,
                "expected_token_id": expected_token,
                "predicted_token_id": predicted_token,
                "accepted": predicted_token == expected_token,
                "terminal": terminal,
            }));
        }
        let accepted_generation_tokens = reference_checks
            .iter()
            .take_while(|check| check.get("accepted") == Some(&Value::Bool(true)))
            .count();
        let all_reference_tokens_accepted = accepted_generation_tokens == reference_checks.len();
        let final_state_index = final_states.len() - 1;
        let (next_after_reference, final_terminal) = terminal_token(
            &terminal_executor,
            &final_states[final_state_index],
            &terminal_dir.join("final-reference.state.f32"),
        )?;
        let candidate = token_ids[prompt_len];
        let predicted_candidate = reference_checks[0]
            .get("predicted_token_id")
            .and_then(Value::as_u64)
            .ok_or("first reference check omitted predicted token")?
            as usize;
        let first_reference_terminal = reference_checks[0]
            .get("terminal")
            .cloned()
            .ok_or("first reference check omitted terminal receipt")?;
        let multi_token_reference = reference_checks.len() >= 2;
        let async_linear_submit = matches!(
            env::var("HAWKING_FLASH_ASYNC_LINEAR_SUBMIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        let linear_region_submit = matches!(
            env::var("HAWKING_FLASH_LINEAR_REGION_SUBMIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        let whole_token_command_buffer =
            hawking_core::env_on("HAWKING_FLASH_WHOLE_TOKEN_COMMAND_BUFFER");
        let collect_dispatch_trace = hawking_core::env_on("HAWKING_FLASH_COLLECT_DISPATCH_TRACE");
        let clean_replay = if clean_replay_reps > 0 {
            if !all_reference_tokens_accepted || !multi_token_reference {
                return Err(
                    "clean token-major replay requires an accepted multi-token reference control"
                        .into(),
                );
            }
            let input_prepare_started = Instant::now();
            let source_before = linear_index.bytes_read_total();
            let replay_bases = token_ids
                .iter()
                .map(|&token_id| {
                    embedding_row(&linear_index, token_id, vocab).map(|row| repeated_streams(&row))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_prepare_ns = input_prepare_started.elapsed().as_nanos() as u64;
            let input_source_bytes = linear_index
                .bytes_read_total()
                .saturating_sub(source_before);
            let banks = token_major_banks
                .as_mut()
                .ok_or("clean token-major replay lost its resident banks")?;
            if collect_dispatch_trace {
                // The resident acceptance pass above uses the same context.
                // Clear its diagnostic stream so the attribution below covers
                // only the explicitly timed clean replay.
                let _ = linear_context.drain_trace();
            }
            let mut replay = run_token_major_clean_replay(
                banks,
                &linear_context,
                &replay_bases,
                &token_ids,
                prompt_len,
                clean_replay_reps,
                async_linear_submit,
                linear_region_submit,
                whole_token_command_buffer,
            )?;
            if collect_dispatch_trace {
                let mode = env::var("HAWKING_TCB_TRACE").ok();
                replay["dispatch_trace_attribution"] =
                    summarize_dispatch_trace(linear_context.drain_trace(), mode);
            }
            replay["input_prepare_ns_outside_timed_window"] = Value::from(input_prepare_ns);
            replay["input_embedding_source_bytes_outside_timed_window"] =
                Value::from(input_source_bytes);
            let mut observed_route_widths = BTreeSet::new();
            let mut active_route_samples = Vec::with_capacity(token_ids.len() - prompt_len);
            for step in prompt_len..token_ids.len() {
                let mut selected_route_slots = 0_usize;
                let mut unique_selected_route_rows = 0_usize;
                for segment in &segments {
                    let row = segment
                        .get("steps")
                        .and_then(Value::as_array)
                        .and_then(|rows| {
                            rows.iter().find(|row| {
                                row.get("step").and_then(Value::as_u64) == Some(step as u64)
                            })
                        });
                    if let Some(route_ids) = row
                        .and_then(|value| value.get("route_ids"))
                        .and_then(Value::as_array)
                    {
                        observed_route_widths.insert(route_ids.len());
                        selected_route_slots = selected_route_slots.saturating_add(route_ids.len());
                        unique_selected_route_rows = unique_selected_route_rows.saturating_add(
                            route_ids
                                .iter()
                                .filter_map(Value::as_u64)
                                .collect::<BTreeSet<_>>()
                                .len(),
                        );
                    }
                }
                let selected_weight_bytes =
                    (selected_route_slots as u64).saturating_mul(ROUTED_EXPERT_ROW_BYTES);
                let unique_selected_weight_bytes =
                    (unique_selected_route_rows as u64).saturating_mul(ROUTED_EXPERT_ROW_BYTES);
                active_route_samples.push(json!({
                    "token_slot": step,
                    "token_id": token_ids[step],
                    "routed_layers": segments.len(),
                    "selected_route_slots": selected_route_slots,
                    "unique_selected_route_rows": unique_selected_route_rows,
                    "selected_expert_weight_bytes": selected_weight_bytes,
                    "unique_selected_expert_weight_bytes": unique_selected_weight_bytes,
                }));
            }
            let active_route_sample_count = active_route_samples.len() as u64;
            let mean_selected_weight_bytes = active_route_samples
                .iter()
                .filter_map(|sample| {
                    sample
                        .get("selected_expert_weight_bytes")
                        .and_then(Value::as_u64)
                })
                .sum::<u64>()
                / active_route_sample_count;
            let mean_unique_selected_weight_bytes = active_route_samples
                .iter()
                .filter_map(|sample| {
                    sample
                        .get("unique_selected_expert_weight_bytes")
                        .and_then(Value::as_u64)
                })
                .sum::<u64>()
                / active_route_sample_count;
            replay["active_bytes_accounting"]["selected_expert_geometry"] = json!({
                "routed_layers": segments.len(),
                "expected_top_k": ROUTED_EXPERT_TOP_K,
                "observed_route_widths": observed_route_widths,
                "expert_dtype": "BF16",
                "expert_row_bytes": ROUTED_EXPERT_ROW_BYTES,
                "gate_up_elements_per_expert": 2 * ROUTED_EXPERT_INTERMEDIATE * HIDDEN,
                "down_elements_per_expert": HIDDEN * ROUTED_EXPERT_INTERMEDIATE,
                "mean_selected_expert_weight_bytes_per_timed_token": mean_selected_weight_bytes,
                "mean_unique_selected_expert_weight_bytes_per_timed_token": mean_unique_selected_weight_bytes,
                "timed_token_samples": active_route_samples,
                "measurement": "logical selected routed-expert row geometry; selected rows are already resident in compact device banks",
                "physical_cache_traffic_bytes_per_token": Value::Null,
                "claim_boundary": "This is a geometry bound for selected routed-expert rows, not a DRAM/cache counter or a claim that the same bytes are fetched from physical memory on every token.",
            });
            Some(replay)
        } else {
            None
        };
        let status = if token_major_allow_state_drift_candidate
            && all_reference_tokens_accepted
            && multi_token_reference
        {
            "CANDIDATE_ROUTE_EXACT_TERMINAL_ACCEPTED_STATE_HASH_WITHHELD"
        } else if all_reference_tokens_accepted && multi_token_reference {
            "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
        } else if all_reference_tokens_accepted {
            "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION"
        } else {
            "PASSED_COMPLETE_FORWARD_REFERENCE_REJECTED"
        };
        let linear_persistent_bytes = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .pointer("/persistent_state/total_bytes")
                    .or_else(|| segment.pointer("/state_memory/total_bytes"))
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let attention_persistent_bytes = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .pointer("/state_memory/persistent_kv_bytes")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let attention_growth_bytes_per_token = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .pointer("/state_memory/growth_bytes_per_token")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let source_payload_bytes_read = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .get("source_payload_bytes_read")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let total_persistent_bytes =
            linear_persistent_bytes.saturating_add(attention_persistent_bytes);
        // Keep the token-major owner alive through terminal verification.  Its
        // count is deliberately captured from the actual owning vector, not a
        // requested flag, so a receipt cannot claim a body that has dropped.
        let token_major_bank_count = token_major_banks.as_ref().map(Vec::len).unwrap_or(0);
        let retained_linear_layers = if token_major_resident_banks {
            segments
                .iter()
                .filter(|segment| {
                    segment.get("species") == Some(&Value::String("linear_attention".to_owned()))
                })
                .count()
        } else {
            retained_linear_banks.iter().map(Vec::len).sum::<usize>()
        };
        let retained_linear_source_bytes = if retain_linear_banks && !device_only_compact_banks {
            segments
                .iter()
                .filter(|segment| {
                    segment.get("species") == Some(&Value::String("linear_attention".to_owned()))
                })
                .filter_map(|segment| {
                    segment
                        .get("source_payload_bytes_read")
                        .and_then(Value::as_u64)
                })
                .sum::<u64>()
        } else {
            0
        };
        let retained_linear_device_bytes = if retain_linear_banks {
            segments
                .iter()
                .filter(|segment| {
                    segment.get("species") == Some(&Value::String("linear_attention".to_owned()))
                })
                .filter_map(|segment| {
                    segment
                        .get("device_weight_bytes")
                        .or_else(|| segment.get("source_payload_bytes_read"))
                        .and_then(Value::as_u64)
                })
                .sum::<u64>()
        } else {
            0
        };
        let resident_full_attention_layers = segments
            .iter()
            .filter(|segment| {
                segment.get("species") == Some(&Value::String("full_attention".to_owned()))
            })
            .filter(|segment| {
                segment.get("immutable_weight_ownership")
                    == Some(&Value::String("device_only_after_source_upload".to_owned()))
                    && segment.get("host_source_weights_retained_during_token_loop")
                        == Some(&Value::Bool(false))
            })
            .count();
        let resident_full_attention_device_bytes = segments
            .iter()
            .filter(|segment| {
                segment.get("species") == Some(&Value::String("full_attention".to_owned()))
            })
            .filter_map(|segment| segment.get("device_weight_bytes").and_then(Value::as_u64))
            .sum::<u64>();
        let resident_routed_expert_device_bytes = segments
            .iter()
            .filter_map(|segment| {
                segment
                    .get("routed_expert_device_bytes")
                    .and_then(Value::as_u64)
            })
            .sum::<u64>();
        let q4_compact_moe_banks = segments
            .iter()
            .filter(|segment| segment.get("q4_compact_moe") == Some(&Value::Bool(true)))
            .count();
        let q8_compact_moe_banks = segments
            .iter()
            .filter(|segment| segment.get("q8_compact_moe") == Some(&Value::Bool(true)))
            .count();
        let q8_residual_banks = segments
            .iter()
            .filter(|segment| segment.get("q8_residual") == Some(&Value::Bool(true)))
            .count();
        let resident_rss_bytes = if retain_linear_banks || token_major_resident_banks {
            current_rss_bytes()
        } else {
            None
        };
        let first_rejection = reference_checks
            .iter()
            .find(|check| check.get("accepted") != Some(&Value::Bool(true)))
            .cloned();
        let token_major_host_seam_diagnostic =
            token_major_host_attention_seam || token_major_host_linear_seam;
        let mut receipt = json!({
            "schema": "hawking.flash.stateful_complete_token_session.v1",
            "status": status,
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "source_input_identity": source_input_identity,
            "token_ids": token_ids,
            "prompt_token_ids": &token_ids[..prompt_len],
            "reference_generated_token_ids": &token_ids[prompt_len..],
            "candidate_token_id": candidate,
            "vocab_size": vocab,
            "segments": segments,
            "terminal": {
                "prompt_final": first_reference_terminal,
                "reference_checks": reference_checks,
                "predicted_candidate": predicted_candidate,
                "next_after_reference": next_after_reference,
                "final_reference": final_terminal,
                "candidate_accepted": all_reference_tokens_accepted,
            },
            "execution": {
                "process_boundary": "one native process",
                "invocation_admission": native_invocation_admission.receipt_value(),
                "native_lane_preflight": native_lane_preflight,
                "terminal_executor": {
                    "lifetime": "one session",
                    "source_index_reused": true,
                    "readout_weights_reused": true,
                    "lm_head_reused": true,
                    "source_payload_bytes_read": terminal_executor.source_bytes_read(),
                    "device": terminal_executor.device_name(),
                },
                "linear_recurrence_segments": true,
                "full_attention_kv_segments": true,
                "cross_species_activation_handoff": if token_major_resident_banks && token_major_host_attention_seam && token_major_host_linear_seam { "host diagnostic vectors at linear and full-attention seams; device buffers elsewhere" } else if token_major_resident_banks && token_major_host_attention_seam { "host diagnostic vectors at full-attention seams; device buffers elsewhere" } else if token_major_resident_banks && token_major_host_linear_seam { "host diagnostic vectors at linear seams; device buffers elsewhere" } else if token_major_resident_banks { "device buffers; host snapshots are diagnostic-only and never feed a later layer" } else { "host diagnostic vectors" },
                "reference_contract": "each caller-supplied continuation token is independently compared with the terminal argmax of its preceding exact 48-layer state",
                "per_layer_state_hash_contract": if token_major_allow_state_drift_candidate {
                    "withheld for candidate reduction-order experiment; exact routes and source-terminal token checks remain enforced"
                } else if route_teacher.is_some() {
                    "exact teacher state hash enforced at every layer/token slot"
                } else {
                    "not requested: dense execution records state hashes but has no route teacher to compare"
                },
                "source_reset_or_reprefill": false,
                "state_reset_scope": "linear species reset only before session token zero; full-attention KV buffers allocated once per layer and indexed by the advancing token position",
                "state_memory": {
                    "linear_persistent_bytes": linear_persistent_bytes,
                    "full_attention_persistent_bytes": attention_persistent_bytes,
                    "total_persistent_bytes": total_persistent_bytes,
                    "growth_bytes_per_additional_token": attention_growth_bytes_per_token,
                    "session_token_slots": token_ids.len(),
                },
                "linear_compact_bank_residency": {
                    "requested": retain_linear_banks,
                    "status": if token_major_resident_banks { "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE" } else if retain_linear_banks && device_only_compact_banks { "PARTIAL_PROCESS_LIFETIME_DEVICE_ONLY_LINEAR_BANKS_RETAINED" } else if retain_linear_banks { "PARTIAL_PROCESS_LIFETIME_LINEAR_BANKS_RETAINED" } else { "NOT_REQUESTED" },
                    "retained_linear_layers": retained_linear_layers,
                    "retained_linear_source_weight_bytes": retained_linear_source_bytes,
                    "retained_linear_device_weight_bytes": retained_linear_device_bytes,
                    "immutable_weight_ownership": if device_only_compact_banks { "device_only_after_source_upload" } else { "host_and_device" },
                    "observed_process_rss_bytes": resident_rss_bytes,
                    "claim_boundary": if token_major_resident_banks { "All exact teacher-bound compact linear banks remain device-only through the accepted token sequence as part of the 48-layer token-major owner. This is a correctness control with per-layer diagnostics, not a clean warmed TPS measurement." } else if device_only_compact_banks { "This retains only device-side exact linear compact banks through this process lifetime; the source-side linear weights are released after upload. Full-attention banks are separately measured per layer; token-major whole-model resident decode remains unimplemented, so this is not warmed whole-model decode or TPS." } else { "This retains only the exact linear compact banks through this process lifetime. Full-attention banks are separately measured per layer; token-major whole-model resident decode remains unimplemented, so this is not warmed whole-model decode or TPS." },
                },
                "full_attention_compact_bank_residency": {
                    "status": if token_major_resident_banks && token_major_host_attention_seam && resident_full_attention_layers == FULL_LAYERS.len() { "ALL_FULL_ATTENTION_BANKS_RETAINED__HOST_SEAM_DIAGNOSTIC" } else if token_major_resident_banks && resident_full_attention_layers == FULL_LAYERS.len() { "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE" } else if resident_full_attention_layers == FULL_LAYERS.len() { "PER_LAYER_DEVICE_ONLY_FULL_ATTENTION_BANKS_REUSED_ACROSS_SESSION_TOKENS" } else { "INCOMPLETE" },
                    "resident_full_attention_layers": resident_full_attention_layers,
                    "expected_full_attention_layers": FULL_LAYERS.len(),
                    "device_weight_bytes": resident_full_attention_device_bytes,
                    "host_source_weights_retained_during_token_loop": false,
                    "process_lifetime_all_banks_retained": token_major_resident_banks,
                    "claim_boundary": if token_major_resident_banks { "All exact teacher-bound compact full-attention banks retain device-only weights and persistent KV state while the token-major 48-layer body executes. This control still includes per-layer snapshots and route checks, so it is not a clean warmed TPS measurement." } else { "Each full-attention bank is reused across the session token slots and releases source tensors after upload, but is released after its layer-major pass. A token-major whole-model resident owner is still required before warm complete-model decode or TPS can be claimed." }
                },
                "token_major_resident_banks": {
                    "requested": token_major_resident_banks,
                    "status": if token_major_resident_banks && token_major_host_seam_diagnostic && token_major_bank_count == 48 { "ALL_48_BANKS_RETAINED__HOST_SEAM_DIAGNOSTIC" } else if token_major_resident_banks && token_major_bank_count == 48 { "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE" } else if token_major_resident_banks { "INCOMPLETE" } else { "NOT_REQUESTED" },
                    "resident_layers": token_major_bank_count,
                    "expected_layers": 48,
                    "cross_layer_activation_handoff": if token_major_resident_banks && token_major_host_attention_seam && token_major_host_linear_seam { "host diagnostic vectors at linear and full-attention seams" } else if token_major_resident_banks && token_major_host_attention_seam { "host diagnostic vector at full-attention seams" } else if token_major_resident_banks && token_major_host_linear_seam { "host diagnostic vectors at linear seams" } else if token_major_resident_banks { "device buffers" } else { "not_applicable" },
                    "source_reset_or_reprefill": false,
                    "claim_boundary": if token_major_resident_banks && token_major_host_seam_diagnostic { "Every immutable compact bank is owned by one token-major native body until after the terminal checks, but host diagnostic vectors feed one or more later layers. This is not device-only token-major residency, warm TPS, future-route coverage, EBPW, capability, or resident-promotion evidence." } else if token_major_resident_banks { "Every immutable compact bank is owned by one token-major native body until after the terminal checks. The retained teacher route union is bounded to the recorded sequence, and per-layer diagnostics remain enabled; no warm TPS, future-route coverage, EBPW, capability, or resident-promotion claim follows." } else { "Not requested." },
                },
                "source_payload_bytes_read": source_payload_bytes_read,
                "source_payload_bytes_per_session_token": source_payload_bytes_read
                    .checked_div(token_ids.len() as u64),
                "source_independent": false,
                "expert_bank_mode": if route_teacher.is_some() { "route_union_compact_teacher_bound" } else { "dense" },
                "physical_expert_bank_mode": if q8_residual_banks > 0 { "compact_routed_q8_group32_residual" } else if q8_compact_moe_banks > 0 { "compact_routed_q8_group32" } else if q4_compact_moe_banks > 0 { "compact_routed_q4_group64" } else if route_teacher.is_some() { "compact_routed_bf16" } else { "dense_full" },
                "q4_compact_moe_banks": q4_compact_moe_banks,
                "q8_compact_moe_banks": q8_compact_moe_banks,
                "q8_residual_banks": q8_residual_banks,
                "q8_residual_fraction": if q8_residual_banks > 0 { q8_residual_fraction() } else { 0.0 },
                "routed_expert_device_bytes": resident_routed_expert_device_bytes,
                "route_teacher": if route_teacher.is_some() { "validated exact dense session" } else { "none" },
                "fallback_count": 0,
                "elapsed_wall_ns": started.elapsed().as_nanos() as u64,
            },
            "accepted_generation_tokens": accepted_generation_tokens,
            "accepted_tps": Value::Null,
            "ple_pre_layer_state_bank": ple_pre_layer_state_bank
                .as_ref()
                .map(PlePreLayerStateBankExporter::receipt_summary)
                .unwrap_or_else(|| json!({"requested": false, "status": "NOT_REQUESTED"})),
            "clean_replay": clean_replay,
            "physical_configuration": {
                "source_bf16_vec4": hawking_core::env_on("HAWKING_FLASH_BF16_VEC4"),
                "source_bf16_hyperconnection_combine_vec4": hawking_core::env_on("HAWKING_FLASH_BF16_VEC4"),
                "source_bf16_geo": hawking_core::env_on("HAWKING_FLASH_BF16_GEO"),
                "source_bf16_geo_dual": hawking_core::env_on("HAWKING_FLASH_BF16_GEO_DUAL"),
                "hyperconnection_split": hawking_core::env_on("HAWKING_FLASH_HC_SPLIT"),
                "hyperconnection_fused_pairs": hawking_core::env_on("HAWKING_FLASH_HC_FUSE_PAIRS"),
                "hyperconnection_fused_up_silu": hawking_core::env_on("HAWKING_FLASH_HC_FUSE_UP_SILU"),
                "hyperconnection_router_fused": hawking_core::env_on("HAWKING_FLASH_HC_ROUTER_FUSED"),
                "hyperconnection_threadgroup_512": hawking_core::env_on("HAWKING_FLASH_HC_TG512"),
                "hyperconnection_split_norm_threadgroup_size": 512,
                "clean_mlp_hc_transient_output_compaction": hawking_core::env_on("HAWKING_FLASH_CLEAN_HC_COMPACT"),
                "router_topk_fused": hawking_core::env_on("HAWKING_FLASH_ROUTER_TOPK_FUSED"),
                "compact_moe_vec4": hawking_core::env_on("HAWKING_FLASH_MOE_VEC4"),
                "compact_moe_geo": hawking_core::env_on("HAWKING_FLASH_MOE_GEO") && !hawking_core::env_on("HAWKING_FLASH_MOE_VEC4"),
                "q4_compact_moe": hawking_core::env_on("HAWKING_FLASH_Q4_COMPACT_MOE"),
                "q4_linear_only_diagnostic": hawking_core::env_on("HAWKING_FLASH_Q4_LINEAR_ONLY"),
                "q4_full_attention_only_diagnostic": hawking_core::env_on("HAWKING_FLASH_Q4_FULL_ATTENTION_ONLY"),
                "q4_exclude_layers": std::env::var("HAWKING_FLASH_Q4_EXCLUDE_LAYERS").ok(),
                "q4_group_size": 64,
                "q4_bank_contract": "routed gate/up/down raw Q4 codes plus FP16 group scales; shared/control weights remain source BF16",
                "q8_compact_moe": hawking_core::env_on("HAWKING_FLASH_Q8_COMPACT_MOE"),
                "q8_residual": hawking_core::env_on("HAWKING_FLASH_Q8_RESIDUAL"),
                "q8_residual_fraction": q8_residual_fraction(),
                "q8_exclude_layers": std::env::var("HAWKING_FLASH_Q8_EXCLUDE_LAYERS").ok(),
                "q8_group_size": Q8_GROUP_SIZE,
                "q8_bank_contract": "routed gate/up/down raw offset-binary Q8/G32 codes plus FP16 group scales and optional sparse exact residuals; shared/control weights remain source BF16",
                "compact_moe_gateup_geo": hawking_core::env_on("HAWKING_FLASH_MOE_GATEUP_GEO"),
                "compact_moe_gateup_geo_exact": hawking_core::env_on("HAWKING_FLASH_MOE_GATEUP_GEO_EXACT"),
                "compact_moe_gateup_geo_layer": std::env::var("HAWKING_FLASH_MOE_GATEUP_GEO_LAYER").ok(),
                "deltanet_value_parallel_exact": hawking_core::env_on("HAWKING_FLASH_DELTANET_VI_EXACT"),
                "deltanet_value_parallel_simd_candidate": hawking_core::env_on("HAWKING_FLASH_DELTANET_VI_SIMD"),
                "queue_ordered_linear_submit": hawking_core::env_on("HAWKING_FLASH_ASYNC_LINEAR_SUBMIT"),
                "linear_region_submit": hawking_core::env_on("HAWKING_FLASH_LINEAR_REGION_SUBMIT"),
                "whole_token_command_buffer": whole_token_command_buffer,
                "claim_boundary": "This records selected opt-in physical kernels for reproducibility. A true flag is not a qualification or promotion claim; the receipt contract remains authoritative."
            },
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "first_physical_failure_boundary": if all_reference_tokens_accepted { Value::Null } else { json!({"stage": "reference_token_acceptance", "first_rejection": first_rejection, "reason": "complete forward retained state through the supplied sequence, but a caller-supplied reference continuation token did not match the preceding terminal argmax"}) },
            "claim_boundary": if token_major_allow_state_drift_candidate && all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "This is a non-bitwise physical candidate: routes and consecutive source-terminal tokens passed, but the exact per-layer state-hash contract was intentionally withheld because its reduction order differs. It cannot replace the exact resident body or support capability, EBPW, qualified TPS, or promotion claims without a separately earned tolerance/capability contract."
            } else if token_major_resident_banks && token_major_host_seam_diagnostic && all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "A token-major diagnostic retained all compact banks but deliberately restored prior host activation vectors at one or more linear/full-attention seams. It distinguishes seam parity from bank/state ordering only; it is not device-only token-major residency or TPS."
            } else if token_major_resident_banks && all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "A token-major complete 48-layer Flash body retained all exact teacher-bound compact immutable banks device-only across the recorded accepted sequence. Every inter-layer activation handoff used a device buffer; linear recurrent state and full-attention KV state survived token boundaries; every route and continuation terminal was independently checked. This is bounded to the recorded teacher sequence and contains diagnostic snapshots/route checks, so it does not prove future-route coverage, clean warmed TPS, EBPW, capability, source-independent NX, or resident promotion."
            } else if all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() {
                "A complete 48-layer stateful Flash forward used only an exact dense-teacher route union for each routed expert bank, retained linear recurrent and full-attention KV state across multiple caller-supplied reference continuation tokens in one native process, and checked every observed route plus each token against its preceding terminal argmax. This is bounded to the recorded teacher sequence; cross-species seams still use host diagnostic vectors, and it does not prove future-route coverage, source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else if all_reference_tokens_accepted && multi_token_reference {
                "A complete 48-layer stateful Flash forward retained linear recurrent and full-attention KV state across multiple caller-supplied reference continuation tokens in one native process, checking each token against its preceding terminal argmax. Cross-species seams still use host diagnostic vectors; this does not prove source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else if all_reference_tokens_accepted {
                "A complete 48-layer stateful Flash forward accepted one tokenizer-bound reference candidate token in one native process. Cross-species seams still use host diagnostic vectors; this does not prove repeated decode, source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else {
                "A complete 48-layer stateful Flash forward retained state through the supplied reference sequence, but a reference token was rejected by terminal argmax. This is a precise continuation diagnosis, not a TPS, EBPW, capability, or residency claim."
            },
            "next": if token_major_resident_banks && all_reference_tokens_accepted && multi_token_reference { "Use this exact resident body to construct a separate clean repeated timing path that removes per-layer host snapshots, route comparisons, and terminal calls from the timed loop while retaining source, state, dispatch, and bytes/token census." } else if all_reference_tokens_accepted && multi_token_reference && route_teacher.is_some() { "Keep the teacher-bound compact banks resident, separate source/index/upload/pipeline/wait/state timing, then measure warmed repeated decode without reloading the banks." } else if all_reference_tokens_accepted && multi_token_reference { "Run clean repeated timing, then census state/dispatch/bytes-per-token before selecting the highest-value physical or representation optimization." } else if all_reference_tokens_accepted { "Supply one more terminal-derived reference token with --prompt-length fixed at the original prompt boundary, then rerun the same persistent session." } else { "Replace the rejected reference token with the recorded preceding terminal argmax and rerun without changing the prompt boundary or source seal." },
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_complete_token_session", "machine": "Apple Metal", "rule": "S032 §3 -- complete-session timing is not a promotion benchmark"},
        });
        receipt["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&receipt)?));
        write_new_receipt(&out, &serde_json::to_vec_pretty(&receipt)?)?;
        if let Some(exporter) = ple_pre_layer_state_bank.as_ref() {
            exporter.finish(&out, &receipt, &token_ids, prompt_len)?;
        }
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        Ok(())
    }

    #[cfg(test)]
    mod route_teacher_tests {
        use super::*;
        use tempfile::tempdir;

        const TEST_MODEL_ROOT: &str = "/fixture/canonical-flash-root";

        fn seal(mut document: Value) -> Value {
            document
                .as_object_mut()
                .expect("test receipt object")
                .remove("seal_sha256");
            let seal = sha256(&serde_json::to_vec(&document).expect("serialize test receipt"));
            document["seal_sha256"] = Value::String(seal);
            document
        }

        fn test_source_input_identity() -> Value {
            serde_json::json!({
                "schema": SOURCE_INPUT_IDENTITY_SCHEMA,
                "model": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "model_root": TEST_MODEL_ROOT,
                "model_lake_manifest": {
                    "path": "/fixture/canonical-flash-manifest.json",
                    "sha256": "b".repeat(64),
                    "repo": REPO_ID,
                    "revision": PINNED_REVISION,
                },
                "config": {"file": SOURCE_CONFIG_FILE, "sha256": "c".repeat(64), "bytes": 1},
                "safetensors_index": {"file": SOURCE_INDEX_FILE, "sha256": "d".repeat(64), "bytes": 1},
                "tokenizer": {"file": SOURCE_TOKENIZER_FILE, "sha256": "e".repeat(64), "bytes": 1},
            })
        }

        fn accepted_dense_token_major_teacher(tokens: &[usize]) -> Value {
            let prompt_len = PROMPT_IDS.len();
            let references = &tokens[prompt_len..];
            let checks = references
                .iter()
                .enumerate()
                .map(|(generation_index, &token)| {
                    serde_json::json!({
                        "generation_index": generation_index,
                        "input_state_index": prompt_len - 1 + generation_index,
                        "expected_token_id": token,
                        "predicted_token_id": token,
                        "accepted": true,
                    })
                })
                .collect::<Vec<_>>();
            let segments = (0..48)
                .map(|layer| {
                    let steps = tokens
                        .iter()
                        .enumerate()
                        .map(|(step, &token_id)| {
                            serde_json::json!({
                                        "layer": layer,
                                        "step": step,
                                        "token_id": token_id,
                                        "route_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        "final_state_sha256": "a".repeat(64),
                            })
                        })
                        .collect::<Vec<_>>();
                    serde_json::json!({
                        "layers": [layer, layer],
                        "expert_bank_mode": "dense",
                        "steps": steps,
                    })
                })
                .collect::<Vec<_>>();
            seal(serde_json::json!({
                "schema": SESSION_RECEIPT_SCHEMA,
                "status": REPEATED_ACCEPTED_STATUS,
                "model": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "root": TEST_MODEL_ROOT,
                "source_input_identity": test_source_input_identity(),
                "vocab_size": PINNED_VOCAB_SIZE,
                "token_ids": tokens,
                "prompt_token_ids": &tokens[..prompt_len],
                "reference_generated_token_ids": references,
                "candidate_token_id": references[0],
                "accepted_generation_tokens": references.len(),
                "terminal": {
                    "candidate_accepted": true,
                    "predicted_candidate": references[0],
                    "prompt_final": {"token_id": references[0]},
                    "reference_checks": checks,
                },
                "execution": {
                    "process_boundary": "one native process",
                    "source_reset_or_reprefill": false,
                    "expert_bank_mode": "dense",
                    "cross_species_activation_handoff": "device buffers; host snapshots are diagnostic-only and never feed a later layer",
                    "token_major_resident_banks": {
                        "requested": true,
                        "status": TOKEN_MAJOR_RESIDENT_STATUS,
                        "resident_layers": 48,
                        "expected_layers": 48,
                        "cross_layer_activation_handoff": "device buffers",
                        "source_reset_or_reprefill": false,
                    },
                },
                "segments": segments,
            }))
        }

        fn parse_test_teacher_raw(raw: &[u8], tokens: &[usize]) -> Result<RouteTeacher, String> {
            let temp = tempdir().expect("create temporary receipt directory");
            let path = temp.path().join("teacher.json");
            std::fs::write(&path, raw).expect("write temporary receipt");
            let argv = vec![
                "flash_stateful_complete_token_session".to_owned(),
                "--route-teacher".to_owned(),
                path.display().to_string(),
            ];
            let source_identity = test_source_input_identity();
            parse_route_teacher(
                &argv,
                tokens,
                PROMPT_IDS.len(),
                std::path::Path::new(TEST_MODEL_ROOT),
                &source_identity,
                false,
            )
            .map(|teacher| teacher.expect("route teacher requested"))
            .map_err(|error| error.to_string())
        }

        fn parse_test_teacher(document: &Value, tokens: &[usize]) -> Result<RouteTeacher, String> {
            parse_test_teacher_raw(
                &serde_json::to_vec_pretty(document).expect("serialize temporary receipt"),
                tokens,
            )
        }

        fn rejection(document: &Value, tokens: &[usize]) -> String {
            match parse_test_teacher(document, tokens) {
                Ok(_) => panic!("invalid route teacher was accepted"),
                Err(error) => error,
            }
        }

        #[test]
        fn route_teacher_requires_a_sealed_dense_all_48_native_session() {
            let tokens = [PROMPT_IDS.as_slice(), &[271, 248045]].concat();
            let good = accepted_dense_token_major_teacher(&tokens);
            let parsed =
                parse_test_teacher(&good, &tokens).expect("qualified dense teacher accepted");
            assert_eq!(parsed.routes.len(), 48 * tokens.len());
            assert_eq!(parsed.final_state_sha256.len(), 48 * tokens.len());
            assert_eq!(parsed.unions.len(), 48);

            let mut unsealed = good.clone();
            unsealed["status"] = Value::String("PASSED_STATEFUL_COMPLETE_TOKEN_SESSION".to_owned());
            assert!(rejection(&unsealed, &tokens).contains("compact UTF-8 seal"));

            let raw = serde_json::to_string(&good).expect("serialize dense teacher");
            let duplicate_schema = raw.replacen(
                &format!("\"schema\":\"{SESSION_RECEIPT_SCHEMA}\""),
                &format!("\"schema\":\"wrong\",\"schema\":\"{SESSION_RECEIPT_SCHEMA}\""),
                1,
            );
            let duplicate_error = match parse_test_teacher_raw(duplicate_schema.as_bytes(), &tokens)
            {
                Ok(_) => panic!("duplicate JSON keys must not be accepted"),
                Err(error) => error,
            };
            assert!(duplicate_error.contains("duplicate JSON key"));

            let mut wrong_schema = good.clone();
            wrong_schema["schema"] = Value::String("hawking.flash.other.v1".to_owned());
            wrong_schema = seal(wrong_schema);
            assert!(rejection(&wrong_schema, &tokens).contains("session schema"));

            let mut wrong_status = good.clone();
            wrong_status["status"] =
                Value::String("PASSED_STATEFUL_COMPLETE_TOKEN_SESSION".to_owned());
            wrong_status = seal(wrong_status);
            assert!(rejection(&wrong_status, &tokens).contains("session schema"));

            let mut wrong_root = good.clone();
            wrong_root["root"] = Value::String("/fixture/other-root".to_owned());
            wrong_root = seal(wrong_root);
            assert!(rejection(&wrong_root, &tokens).contains("root differs"));

            let mut wrong_source_inputs = good.clone();
            wrong_source_inputs["source_input_identity"]["config"]["sha256"] =
                Value::String("f".repeat(64));
            wrong_source_inputs = seal(wrong_source_inputs);
            assert!(rejection(&wrong_source_inputs, &tokens).contains("source input identity"));

            let mut compact = good.clone();
            compact["execution"]["expert_bank_mode"] =
                Value::String("route_union_compact_teacher_bound".to_owned());
            compact = seal(compact);
            assert!(rejection(&compact, &tokens).contains("dense native control"));

            let mut segmented = good.clone();
            segmented["segments"]
                .as_array_mut()
                .expect("test segments")
                .pop();
            segmented = seal(segmented);
            assert!(rejection(&segmented, &tokens).contains("segmented"));

            let mut compact_segment = good.clone();
            compact_segment["segments"][0]["expert_bank_mode"] =
                Value::String("route_union_compact_teacher_bound".to_owned());
            compact_segment = seal(compact_segment);
            assert!(rejection(&compact_segment, &tokens).contains("segment is not a dense control"));

            let mut reset = good.clone();
            reset["execution"]["source_reset_or_reprefill"] = Value::Bool(true);
            reset = seal(reset);
            assert!(rejection(&reset, &tokens).contains("one-process persistent-session proof"));

            let mut incomplete_residency = good.clone();
            incomplete_residency["execution"]["token_major_resident_banks"]["resident_layers"] =
                Value::from(47);
            incomplete_residency = seal(incomplete_residency);
            assert!(rejection(&incomplete_residency, &tokens)
                .contains("all-48 device-resident token-major evidence"));

            let mut host_seam_diagnostic = good.clone();
            host_seam_diagnostic["execution"]["cross_species_activation_handoff"] = Value::String(
                "host diagnostic vectors at linear seams; device buffers elsewhere".to_owned(),
            );
            host_seam_diagnostic["execution"]["token_major_resident_banks"]["status"] =
                Value::String("ALL_48_BANKS_RETAINED__HOST_SEAM_DIAGNOSTIC".to_owned());
            host_seam_diagnostic = seal(host_seam_diagnostic);
            assert!(rejection(&host_seam_diagnostic, &tokens)
                .contains("device-only token-major activation evidence"));

            let mut broken_chain = good.clone();
            broken_chain["terminal"]["reference_checks"][1]["predicted_token_id"] = Value::from(0);
            broken_chain = seal(broken_chain);
            assert!(rejection(&broken_chain, &tokens).contains("reference chain is not exact"));

            let mut wrong_candidate = good.clone();
            wrong_candidate["candidate_token_id"] = Value::from(0);
            wrong_candidate = seal(wrong_candidate);
            assert!(rejection(&wrong_candidate, &tokens).contains("candidate token disagrees"));

            let mut missing_state_hash = good.clone();
            missing_state_hash["segments"][0]["steps"][0]["final_state_sha256"] =
                Value::String("not-a-sha256".to_owned());
            missing_state_hash = seal(missing_state_hash);
            assert!(rejection(&missing_state_hash, &tokens).contains("final-state SHA-256"));

            let mut duplicate_route = good;
            duplicate_route["segments"][0]["steps"][0]["route_ids"] =
                serde_json::json!([0, 0, 2, 3, 4, 5, 6, 7, 8, 9]);
            duplicate_route = seal(duplicate_route);
            assert!(rejection(&duplicate_route, &tokens).contains("top-k coverage"));
        }

        #[test]
        fn route_teacher_requires_the_exact_token_major_hash_checking_path() {
            assert!(require_token_major_route_teacher(false, false, false, false).is_ok());
            assert!(require_token_major_route_teacher(false, true, true, true).is_ok());
            assert!(require_token_major_route_teacher(true, true, false, false).is_ok());
            assert!(require_token_major_route_teacher(true, false, false, false)
                .expect_err("layer-major route teacher must fail before model setup")
                .to_string()
                .contains("requires --token-major-resident-banks"));
            assert!(require_token_major_route_teacher(true, true, true, false)
                .expect_err("host attention seam teacher must fail before model setup")
                .to_string()
                .contains("device-only token-major handoffs"));
            assert!(require_token_major_route_teacher(true, true, false, true)
                .expect_err("host linear seam teacher must fail before model setup")
                .to_string()
                .contains("device-only token-major handoffs"));
        }

        #[test]
        fn native_boundary_capture_scopes_are_exclusive_and_trace_contract_is_ordered() {
            let base = vec!["flash_stateful_complete_token_session".to_owned()];
            assert_eq!(NativeBoundaryCaptureKind::parse(&base).unwrap(), None);
            assert_eq!(
                NativeBoundaryCaptureKind::parse(&[
                    base[0].clone(),
                    "--capture-source-boundary".to_owned(),
                ])
                .unwrap(),
                Some(NativeBoundaryCaptureKind::FullSourcePleBoundary),
            );
            assert_eq!(
                NativeBoundaryCaptureKind::parse(&[
                    base[0].clone(),
                    "--capture-layer0-hc-trace-only".to_owned(),
                ])
                .unwrap(),
                Some(NativeBoundaryCaptureKind::Layer0AttentionHcTraceOnly),
            );
            assert!(NativeBoundaryCaptureKind::parse(&[
                base[0].clone(),
                "--capture-source-boundary".to_owned(),
                "--capture-layer0-hc-trace-only".to_owned(),
            ])
            .expect_err("the full and causal trace captures must not combine")
            .to_string()
            .contains("mutually exclusive"));
            assert!(NativeBoundaryCaptureKind::parse(&[
                base[0].clone(),
                "--capture-layer0-hc-trace-only".to_owned(),
                "--capture-layer0-hc-trace-only".to_owned(),
            ])
            .expect_err("duplicate causal trace flags must fail closed")
            .to_string()
            .contains("at most once"));

            let plan = layer0_attention_hc_trace_plan();
            validate_ordered_semantic_seams(&plan).expect("layer-0 trace plan is ordered");
            assert_eq!(plan.len(), 9);
            assert_eq!(plan[0].semantic_id, "attention_hyper_connection_normalized");
            assert_eq!(
                plan[1].semantic_id,
                "attention_hyper_connection_down_projection"
            );
            assert_eq!(plan[8].semantic_id, "post_attention_state");
            assert!(plan.iter().all(|seam| seam.layer == Some(0)));
        }

        #[test]
        fn native_boundary_capture_lineage_requires_two_lowercase_hashes() {
            let valid = "a".repeat(64);
            assert!(validate_source_boundary_lineage(&valid, &valid).is_ok());
            assert!(validate_source_boundary_lineage("A", &valid)
                .expect_err("uppercase or short capture hash must fail")
                .to_string()
                .contains("source capture SHA-256 is malformed"));
            assert!(validate_source_boundary_lineage(&valid, "a")
                .expect_err("short preflight hash must fail")
                .to_string()
                .contains("boundary preflight SHA-256 is malformed"));
        }

        #[test]
        fn native_launch_policy_and_construction_guards_fail_closed_before_metal() {
            let base = vec!["flash_stateful_complete_token_session".to_owned()];
            assert_eq!(
                require_native_invocation_admission(&[
                    base[0].clone(),
                    "--wrapper-admitted-source-control".to_owned(),
                ])
                .expect("wrapper marker is an explicit policy"),
                NativeInvocationAdmission::WrapperAdmittedSourceControl,
            );
            assert_eq!(
                require_native_invocation_admission(&[
                    base[0].clone(),
                    "--unqualified-direct-control".to_owned(),
                ])
                .expect("direct-control marker is an explicit policy"),
                NativeInvocationAdmission::UnqualifiedDirectControl,
            );
            assert!(require_wrapper_admitted_native_execution(
                NativeInvocationAdmission::WrapperAdmittedSourceControl
            )
            .is_ok());
            assert!(require_wrapper_admitted_native_execution(
                NativeInvocationAdmission::UnqualifiedDirectControl
            )
            .expect_err("unqualified direct launch must stop before source/index/Metal")
            .to_string()
            .contains("refused before source/index/Metal"));
            assert!(require_all_48_token_major_construction(48).is_ok());
            for limit in [0, 1, 47, 49] {
                assert!(require_all_48_token_major_construction(limit)
                    .expect_err("partial or oversized resident construction must fail before Metal")
                    .to_string()
                    .contains("must be exactly 48"));
            }
            assert!(require_native_invocation_admission(&base)
                .expect_err("unmarked native launch must fail")
                .to_string()
                .contains("requires exactly one policy marker"));
            assert!(require_native_invocation_admission(&[
                base[0].clone(),
                "--wrapper-admitted-source-control".to_owned(),
                "--unqualified-direct-control".to_owned(),
            ])
            .expect_err("conflicting launch policies must fail")
            .to_string()
            .contains("mutually exclusive"));
            assert!(supervising_launcher_pid(
                &[
                    base[0].clone(),
                    "--wrapper-admitted-source-control".to_owned(),
                ],
                NativeInvocationAdmission::WrapperAdmittedSourceControl,
            )
            .expect_err("wrapper-admitted execution must name its launcher PID")
            .to_string()
            .contains("requires --supervising-launcher-pid"));
            let wrapper_args = vec![
                base[0].clone(),
                "--wrapper-admitted-source-control".to_owned(),
                "--supervising-launcher-pid".to_owned(),
                "203".to_owned(),
            ];
            assert_eq!(
                supervising_launcher_pid(
                    &wrapper_args,
                    NativeInvocationAdmission::WrapperAdmittedSourceControl,
                )
                .expect("recognized wrapper may nominate exactly one PID"),
                Some(203),
            );
            assert!(supervising_launcher_pid(
                &[
                    base[0].clone(),
                    "--unqualified-direct-control".to_owned(),
                    "--supervising-launcher-pid".to_owned(),
                    "203".to_owned(),
                ],
                NativeInvocationAdmission::UnqualifiedDirectControl,
            )
            .expect_err("unqualified controls cannot nominate a wrapper PID")
            .to_string()
            .contains("only with --wrapper-admitted-source-control"));
            assert!(supervising_launcher_pid(
                &[
                    base[0].clone(),
                    "--wrapper-admitted-source-control".to_owned(),
                    "--supervising-launcher-pid".to_owned(),
                    "not-a-pid".to_owned(),
                ],
                NativeInvocationAdmission::WrapperAdmittedSourceControl,
            )
            .expect_err("a nonnumeric launcher PID must fail")
            .to_string()
            .contains("numeric PID"));

            let snapshot = "4242 203 S flash_stateful_complete_token_session --root fixture\n\
                 101 1 Z flash_stateful_complete_token_session --root fixture\n\
                 201 1 S flash_stateful_complete_token_session --root fixture\n\
                 202 1 R python tools/odyssey/flash_repeated_accepted_decode.py\n\
                 203 1 S python tools/odyssey/flash_route_union_control.py\n\
                 204 1 S python tools/odyssey/flash_source_boundary_capture_mlx.py\n\
                 205 1 S python tools/odyssey/flash_source_boundary_mlx_worker.py\n\
                 206 1 S python tools/odyssey/flash_source_boundary_native.py\n\
                 207 1 S python tools/odyssey/flash_source_boundary_transaction.py --leased-execute\n\
                 208 1 S flash_noetic_complete_layer0\n\
                 209 1 S mlx_vlm.server --model kimi\n\
                 210 1 S hawkingd --serve\n";
            let matches = protected_native_lane_matches(snapshot, 4242, None)
                .expect("well-formed ps snapshot");
            assert_eq!(
                matches.iter().map(|entry| entry.marker).collect::<Vec<_>>(),
                PROTECTED_NATIVE_LANE_MARKERS,
            );
            let excluding_wrapper = protected_native_lane_matches(snapshot, 4242, Some(203))
                .expect("the declared route wrapper may be excluded");
            assert_eq!(
                excluding_wrapper
                    .iter()
                    .map(|entry| entry.marker)
                    .collect::<Vec<_>>(),
                [
                    "flash_stateful_complete_token_session",
                    "flash_repeated_accepted_decode.py",
                    "flash_source_boundary_capture_mlx.py",
                    "flash_source_boundary_mlx_worker.py",
                    "flash_source_boundary_native.py",
                    "flash_source_boundary_transaction.py",
                    "flash_noetic_complete_layer0",
                    "mlx_vlm.server",
                    "hawkingd",
                ],
            );
            assert!(protected_native_lane_matches(
                "4242 207 S flash_stateful_complete_token_session --root fixture\n\
                 207 1 S python tools/odyssey/flash_source_boundary_transaction.py --leased-execute\n",
                4242,
                Some(207),
            )
            .expect("the admitted source-boundary transaction may supervise its direct native child")
            .is_empty());
            assert!(protected_native_lane_matches(
                "4242 207 S /fixture/flash_stateful_complete_token_session --root fixture\n\
                 207 208 S python3 tools/odyssey/flash_source_boundary_transaction.py --leased-execute --native-binary /fixture/flash_stateful_complete_token_session\n\
                 208 1 S python3 tools/odyssey/flash_source_boundary_transaction.py --execute --native-binary /fixture/flash_stateful_complete_token_session\n\
                 206 1 S bash tools/gpu_lane_lock.sh HAWKING_SOURCE_BOUNDARY_TRANSACTION python3 tools/odyssey/flash_source_boundary_transaction.py --leased-execute --native-binary /fixture/flash_stateful_complete_token_session\n",
                4242,
                Some(207),
            )
            .expect("only direct-lineage lease wrappers may be excluded as native support")
            .is_empty());
            assert!(protected_native_lane_matches(snapshot, 4242, Some(999))
                .expect_err("an absent supervising PID must fail closed")
                .to_string()
                .contains("was not visible"));
            assert!(protected_native_lane_matches(snapshot, 4242, Some(101))
                .expect_err("a zombie supervising PID must not be excluded")
                .to_string()
                .contains("non-zombie wrapper"));
            assert!(protected_native_lane_matches(snapshot, 4242, Some(202))
                .expect_err("a foreign recognized wrapper PID must not be excluded")
                .to_string()
                .contains("immediate native launcher"));
            assert!(protected_native_lane_matches(
                "4242 203 S flash_stateful_complete_token_session --root fixture\n\
                 203 1 S python tools/unrelated_launcher.py\n",
                4242,
                Some(203),
            )
            .expect_err("an unrelated supervising PID must not be excluded")
            .to_string()
            .contains("recognized wrapper"));
            assert!(
                protected_native_lane_matches("bad 1 S hawkingd --serve\n", 4242, None)
                    .expect_err("malformed ps output must fail closed")
                    .to_string()
                    .contains("nonnumeric PID")
            );
        }

        #[test]
        fn output_reservation_refuses_existing_receipts_and_session_artifacts() {
            let temp = tempdir().expect("create temporary output directory");

            let existing_receipt = temp.path().join("existing.json");
            std::fs::write(&existing_receipt, b"existing receipt").expect("write existing receipt");
            assert!(reserve_session_output(&existing_receipt)
                .expect_err("existing receipt must not be overwritten")
                .to_string()
                .contains("refuses to overwrite"));

            let existing_artifact_receipt = temp.path().join("artifact.json");
            std::fs::create_dir(session_artifact_dir(&existing_artifact_receipt))
                .expect("create existing artifact directory");
            assert!(reserve_session_output(&existing_artifact_receipt)
                .expect_err("existing artifact directory must not be reused")
                .to_string()
                .contains("refuses to reuse"));

            let fresh_receipt = temp.path().join("fresh.json");
            let fresh_artifact_dir =
                reserve_session_output(&fresh_receipt).expect("reserve fresh output namespace");
            assert!(fresh_artifact_dir.is_dir());
            write_new_receipt(&fresh_receipt, b"first receipt").expect("write fresh receipt");
            assert!(write_new_receipt(&fresh_receipt, b"replacement receipt").is_err());
            assert_eq!(
                std::fs::read(&fresh_receipt).expect("read fresh receipt"),
                b"first receipt"
            );
        }

        #[test]
        fn session_artifact_payload_writes_preserve_racing_files() {
            let temp = tempdir().expect("create temporary output directory");
            let receipt = temp.path().join("fresh-session.json");
            let artifacts = reserve_session_output(&receipt)
                .expect("reserve a fresh session artifact namespace");

            let state_path = artifacts.join("L03_STATE.f32");
            write_state(&state_path, &[1.0, -2.0]).expect("write fresh state payload");
            let first_state = std::fs::read(&state_path).expect("read fresh state payload");
            assert!(write_state(&state_path, &[3.0, 4.0]).is_err());
            assert_eq!(
                std::fs::read(&state_path).expect("read preserved state payload"),
                first_state
            );

            let layer_receipt = artifacts.join("L03_ATTENTION.json");
            write_new_file(&layer_receipt, b"first attention receipt")
                .expect("write fresh layer receipt");
            assert!(write_new_file(&layer_receipt, b"replacement attention receipt").is_err());
            assert_eq!(
                std::fs::read(&layer_receipt).expect("read preserved layer receipt"),
                b"first attention receipt"
            );
        }

        #[test]
        fn ple_state_bank_reservation_is_disjoint_exclusive_and_no_clobber() {
            let temp = tempdir().expect("create temporary output directory");
            let receipt = temp.path().join("session.json");
            let artifacts = session_artifact_dir(&receipt);

            for colliding in [
                receipt.clone(),
                artifacts.clone(),
                artifacts.join("nested-state-bank"),
            ] {
                assert!(
                    validate_ple_pre_layer_state_bank_output(&colliding, &receipt, &artifacts)
                        .expect_err("state-bank target must be disjoint from receipt namespace")
                        .to_string()
                        .contains("disjoint")
                );
            }
            assert!(
                validate_ple_pre_layer_state_bank_output(temp.path(), &receipt, &artifacts,)
                    .expect_err("an existing state-bank target must fail")
                    .to_string()
                    .contains("refuses an existing")
            );

            let physical_parent = temp.path().join("physical-output-parent");
            std::fs::create_dir(&physical_parent).expect("create physical output parent");
            let physical_receipt = physical_parent.join("aliased-session.json");
            let physical_artifacts = session_artifact_dir(&physical_receipt);
            let aliased_parent = temp.path().join("aliased-output-parent");
            std::os::unix::fs::symlink(&physical_parent, &aliased_parent)
                .expect("create symlinked output parent");
            let symlink_nested_bank = aliased_parent
                .join("aliased-session.session_artifacts")
                .join("nested-state-bank");
            assert!(validate_ple_pre_layer_state_bank_output(
                &symlink_nested_bank,
                &physical_receipt,
                &physical_artifacts,
            )
            .expect_err("symlink-hidden state bank nesting must fail")
            .to_string()
            .contains("disjoint"));

            let existing_bank = temp.path().join("existing-state-bank");
            std::fs::create_dir(&existing_bank)
                .expect("create an empty preexisting bank directory");
            assert!(
                validate_ple_pre_layer_state_bank_output(&existing_bank, &receipt, &artifacts)
                    .expect_err("preexisting PLE bank must not be reused")
                    .to_string()
                    .contains("refuses an existing")
            );
            let racing_reservation = match PlePreLayerStateBankExporter::new(existing_bank) {
                Ok(_) => {
                    panic!("exclusive PLE reservation must reject a racing existing directory")
                }
                Err(error) => error,
            };
            assert!(racing_reservation.to_string().contains("became occupied"));

            let state_bank_dir = temp.path().join("state-no-clobber");
            let mut state_exporter = PlePreLayerStateBankExporter::new(state_bank_dir.clone())
                .expect("reserve fresh state bank");
            let state_path = state_bank_dir.join("state-step-000000.f32");
            std::fs::write(&state_path, b"preserve existing state")
                .expect("seed a competing state payload");
            assert!(state_exporter
                .record(0, 7, &vec![0.0_f32; HC])
                .expect_err("state payload writes must be exclusive")
                .to_string()
                .contains("exists"));
            assert_eq!(
                std::fs::read(&state_path).expect("read preserved competing state"),
                b"preserve existing state"
            );

            let manifest_bank_dir = temp.path().join("manifest-no-clobber");
            let mut manifest_exporter =
                PlePreLayerStateBankExporter::new(manifest_bank_dir.clone())
                    .expect("reserve fresh manifest bank");
            manifest_exporter
                .record(0, 7, &vec![0.0_f32; HC])
                .expect("write initial state payload");
            let session = temp.path().join("sealed-session.json");
            write_new_receipt(&session, b"{}")
                .expect("write the session receipt used by the manifest");
            let manifest_path = manifest_bank_dir.join("manifest.json");
            std::fs::write(&manifest_path, b"preserve existing manifest")
                .expect("seed a competing manifest");
            assert!(manifest_exporter
                .finish(
                    &session,
                    &json!({
                        "schema": SESSION_RECEIPT_SCHEMA,
                        "status": REPEATED_ACCEPTED_STATUS,
                        "seal_sha256": "a".repeat(64),
                    }),
                    &[7],
                    1,
                )
                .expect_err("manifest writes must be exclusive")
                .to_string()
                .contains("exists"));
            assert_eq!(
                std::fs::read(&manifest_path).expect("read preserved competing manifest"),
                b"preserve existing manifest"
            );
        }

        #[test]
        fn source_boundary_teacher_state_is_exact_regular_and_finite() {
            let temp = tempdir().expect("create temporary boundary directory");
            let state = temp.path().join("post-ple.f32");
            let values = [1.25_f32, -0.5_f32];
            let raw = f32_bytes(&values);
            std::fs::write(&state, &raw).expect("write source boundary fixture");
            let digest = sha256(&raw);
            let (observed, binding) =
                read_bound_f32_state(&state, &digest, values.len(), "test source boundary")
                    .expect("accept exact source boundary fixture");
            assert_eq!(observed, values);
            assert_eq!(
                binding.get("sha256").and_then(Value::as_str),
                Some(digest.as_str())
            );

            let symlink = temp.path().join("post-ple-symlink.f32");
            std::os::unix::fs::symlink(&state, &symlink).expect("create source state symlink");
            assert!(
                read_bound_f32_state(&symlink, &digest, values.len(), "test source boundary",)
                    .expect_err("source boundary symlink must fail")
                    .to_string()
                    .contains("non-symlink")
            );

            let hardlink = temp.path().join("post-ple-hardlink.f32");
            std::fs::hard_link(&state, &hardlink).expect("create source state hard link");
            assert!(
                read_bound_f32_state(&state, &digest, values.len(), "test source boundary",)
                    .expect_err("source boundary hard link must fail")
                    .to_string()
                    .contains("non-hard-linked")
            );
        }

        #[test]
        fn source_boundary_payload_writer_is_extent_checked_and_no_clobber() {
            let temp = tempdir().expect("create temporary boundary directory");
            let values = [1.0_f32, 2.0_f32];
            assert!(
                write_boundary_f32(temp.path(), 0, None, "fixture", &values, &[1, 3], "test",)
                    .expect_err("mismatched boundary extent must fail")
                    .to_string()
                    .contains("invalid extent")
            );
            write_boundary_f32(temp.path(), 0, None, "fixture", &values, &[1, 2], "test")
                .expect("write exact boundary payload");
            assert!(
                write_boundary_f32(temp.path(), 0, None, "fixture", &values, &[1, 2], "test",)
                    .expect_err("boundary payload overwrite must fail")
                    .to_string()
                    .contains("exists")
            );
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        main_impl()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
