//! Host-wide process observation for the HCLI operator surface.
//!
//! [`process::ProcessSupervisor`] owns processes started by a HIDE session.
//! This module owns the narrower host diagnostic that answers "which Hawking
//! processes are alive, what do they carry, and which unowned resident bodies
//! can be reclaimed?" Keeping the argv classifier and the signal boundary in
//! Rust makes the Python HCLI skin an adapter instead of a second process
//! authority.

use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize)]
pub struct HostProcess {
    pub pid: u32,
    pub ppid: u32,
    pub rss_bytes: u64,
    pub rss_gib: f64,
    pub memory_source: &'static str,
    pub cpu_percent: f64,
    pub elapsed: String,
    pub role: &'static str,
    #[serde(rename = "class")]
    pub process_class: &'static str,
    pub safe_to_stop: bool,
    pub purpose: &'static str,
    pub command: String,
    pub body: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Role {
    name: &'static str,
    class: &'static str,
    safe_to_stop: bool,
    purpose: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostProcessReport {
    pub count: usize,
    pub total_rss_bytes: u64,
    pub by_class: BTreeMap<String, usize>,
    pub roles: Vec<String>,
    pub processes: Vec<HostProcess>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReapReport {
    pub found: Vec<u32>,
    pub bytes_held: u64,
    pub reaped: Vec<u32>,
    pub failed: Vec<ReapFailure>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReapFailure {
    pub pid: u32,
    pub error: String,
}

fn role_for(command: &str) -> Option<Role> {
    // Order is part of the contract: a downloader must beat a generic Python
    // process, and only a resident executable may claim resident-body.
    const ROLE_DEFS: &[(&str, &str, &str, bool, &str)] = &[
        (
            r"(?:^|/)(?:hawkingd|hcli)(?:\s+-P\s+-m\s+hcli\.hawkingd)?\s+serve\b",
            "hawkingd",
            "ESSENTIAL_PERSISTENT",
            false,
            "the canonical Hawking surface owner; it supervises the resident and its children",
        ),
        (
            r"\b(?:mlx_vlm|mlx_lm)\.server\b",
            "resident-provider",
            "ESSENTIAL_PERSISTENT",
            false,
            "the daemon-owned model provider child; it is not an independent Hawking launch",
        ),
        (
            r"\bopen-webui\s+serve\b",
            "web-ui",
            "ESSENTIAL_EPHEMERAL",
            true,
            "a daemon-owned client surface sharing the canonical Hawking endpoint",
        ),
        (
            r"hcli\.agentos\.resident\b.*--supervise",
            "resident-supervisor",
            "ESSENTIAL_PERSISTENT",
            false,
            "owns heartbeat, memory admission and restart limits; holds no model",
        ),
        (
            r"hcli_sovereign\.py.*--run",
            "sovereign-loop",
            "ESSENTIAL_PERSISTENT",
            false,
            "the SUB2 science loop; stopping it ends the running mission",
        ),
        (
            r"modellake_watch\.py",
            "modellake-watcher",
            "ESSENTIAL_PERSISTENT",
            false,
            "watches acquisitions and seals; parent of the download children",
        ),
        (
            r"\bhf\s+download\b",
            "modellake-download",
            "ESSENTIAL_EPHEMERAL",
            true,
            "one model acquisition; resumable, so stopping it loses only progress",
        ),
        (
            r"(?:^|/)\S*resident\S*\s+.*--(?:artifact-root|resident-identity)\b",
            "resident-body",
            "ESSENTIAL_PERSISTENT",
            false,
            "the loaded model itself; this is legitimate footprint, not overhead",
        ),
        (
            r"model_bearing_torture\.py",
            "torture-harness",
            "DEBUG_ONLY",
            true,
            "autonomy trial harness; not part of normal operation",
        ),
        (
            r"WU\.[A-Za-z0-9_.]+\.child\.py|mbt-run-",
            "resident-worker",
            "ESSENTIAL_EPHEMERAL",
            true,
            "one bounded WorkUnit slice; the mission requeues it",
        ),
    ];
    static ROLE_PATTERNS: OnceLock<Vec<(Regex, Role)>> = OnceLock::new();
    let patterns = ROLE_PATTERNS.get_or_init(|| {
        ROLE_DEFS
            .iter()
            .map(|(pattern, name, class, safe, purpose)| {
                (
                    Regex::new(pattern).expect("process role pattern is valid"),
                    Role {
                        name,
                        class,
                        safe_to_stop: *safe,
                        purpose,
                    },
                )
            })
            .collect()
    });
    patterns
        .iter()
        .find_map(|(pattern, role)| pattern.is_match(command).then_some(*role))
}

fn body_for(command: &str) -> Option<String> {
    let words: Vec<&str> = command.split_whitespace().collect();
    for (index, word) in words.iter().enumerate() {
        if matches!(*word, "--resident-identity" | "--artifact-root") {
            if *word == "--resident-identity" {
                return words.get(index + 1).map(|value| (*value).to_string());
            }
            // Preserve the former Python contract: artifact-root is only a
            // body identity when it is the final argv value. A probe carrying
            // the flag alone must not be labelled as a resident body.
            if index + 2 == words.len() {
                return words
                    .get(index + 1)
                    .and_then(|value| value.rsplit('/').next())
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
            }
        }
        if *word == "download" && index > 0 && words[index - 1] == "hf" {
            return words.get(index + 1).map(|value| (*value).to_string());
        }
        if *word == "--model" {
            return words
                .get(index + 1)
                .and_then(|value| value.rsplit('/').next())
                .filter(|value| !value.is_empty())
                .map(str::to_string);
        }
    }
    None
}

fn parse_footprint_bytes(text: &str) -> BTreeMap<u32, u64> {
    let mut values = BTreeMap::new();
    let mut current_pid = None;
    for line in text.lines() {
        if let (Some(start), Some(end)) = (line.find('['), line.find(']')) {
            if start < end {
                current_pid = line[start + 1..end].parse::<u32>().ok();
            }
        }
        let mut fields = line.split_whitespace();
        if fields.next() != Some("phys_footprint:") {
            continue;
        }
        let Some(pid) = current_pid else { continue };
        let Some(bytes) = fields.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        values.insert(pid, bytes);
    }
    values
}

fn footprint_bytes_many(pids: &[u32]) -> BTreeMap<u32, u64> {
    if pids.is_empty() {
        return BTreeMap::new();
    }
    let mut command = Command::new("footprint");
    for pid in pids {
        command.args(["-p", &pid.to_string()]);
    }
    let output = match command.args(["-f", "bytes", "--noCategories"]).output() {
        Ok(output) if output.status.success() => output,
        _ => return BTreeMap::new(),
    };
    parse_footprint_bytes(&String::from_utf8_lossy(&output.stdout))
}

fn ps_rows() -> Vec<(u32, u32, u64, f64, String, String)> {
    let output = match Command::new("ps")
        .args(["-eo", "pid,ppid,rss,pcpu,etime,command"])
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| {
            // `ps` pads numeric columns with variable-width whitespace.
            // `splitn` counts each padding byte as a separator and can lose
            // the command field; consume the five scalar fields first and
            // join the remaining argv words into the command text.
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse().ok()?;
            let ppid = fields.next()?.parse().ok()?;
            let rss_kb = fields.next()?.parse().ok()?;
            let cpu_percent = fields.next()?.parse().ok()?;
            let elapsed = fields.next()?.to_string();
            let command = fields.collect::<Vec<_>>().join(" ");
            if command.is_empty() || command.contains("ps -eo") {
                return None;
            }
            Some((pid, ppid, rss_kb, cpu_percent, elapsed, command))
        })
        .collect()
}

pub fn inspect(footprint: bool) -> HostProcessReport {
    let rows = ps_rows();
    let footprint_by_pid = if footprint {
        let pids = rows
            .iter()
            .filter_map(|(pid, _, _, _, _, command)| role_for(command).map(|_| *pid))
            .collect::<Vec<_>>();
        footprint_bytes_many(&pids)
    } else {
        BTreeMap::new()
    };
    let mut processes = rows
        .into_iter()
        .filter_map(|(pid, ppid, rss_kb, cpu_percent, elapsed, command)| {
            let role = role_for(&command)?;
            let measured = footprint
                .then(|| footprint_by_pid.get(&pid).copied())
                .flatten();
            let (rss_bytes, memory_source) = measured
                .map(|bytes| (bytes, "phys_footprint"))
                .unwrap_or((rss_kb.saturating_mul(1024), "rss"));
            Some(HostProcess {
                pid,
                ppid,
                rss_bytes,
                rss_gib: ((rss_bytes as f64) / 1024_f64.powi(3) * 1000.0).round() / 1000.0,
                memory_source,
                cpu_percent,
                elapsed,
                role: role.name,
                process_class: role.class,
                safe_to_stop: role.safe_to_stop,
                purpose: role.purpose,
                body: body_for(&command),
                command,
            })
        })
        .collect::<Vec<_>>();
    processes.sort_by_key(|process| (std::cmp::Reverse(process.rss_bytes), process.pid));
    let mut by_class = BTreeMap::new();
    for process in &processes {
        *by_class
            .entry(process.process_class.to_string())
            .or_insert(0) += 1;
    }
    let roles = processes
        .iter()
        .map(|process| process.role.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    HostProcessReport {
        count: processes.len(),
        total_rss_bytes: processes.iter().map(|process| process.rss_bytes).sum(),
        by_class,
        roles,
        processes,
    }
}

fn claimed_worker_pids(workspace: &Path) -> BTreeSet<u32> {
    let mut roots = vec![workspace.to_path_buf()];
    if let Ok(cwd) = std::env::current_dir() {
        if cwd != workspace {
            roots.push(cwd);
        }
    }
    roots
        .into_iter()
        .filter_map(|root| {
            let path = root.join(".hcli").join("resident").join("state.json");
            let raw = std::fs::read_to_string(path).ok()?;
            let value: Value = serde_json::from_str(&raw).ok()?;
            Some(
                ["worker_pid", "supervisor_pid", "body_pid"]
                    .into_iter()
                    .filter_map(|key| value.get(key).and_then(Value::as_u64))
                    .filter_map(|pid| u32::try_from(pid).ok())
                    .collect::<BTreeSet<_>>(),
            )
        })
        .flatten()
        .collect()
}

fn is_orphaned(process: &HostProcess, claimed: &BTreeSet<u32>) -> bool {
    matches!(process.role, "resident-body" | "resident-provider")
        && process.ppid == 1
        && !claimed.contains(&process.pid)
}

pub fn orphaned(workspace: &Path) -> Vec<HostProcess> {
    let claimed = claimed_worker_pids(workspace);
    inspect(false)
        .processes
        .into_iter()
        .filter(|process| is_orphaned(process, &claimed))
        .collect()
}

pub fn reap_orphaned(workspace: &Path, dry_run: bool) -> ReapReport {
    let found = orphaned(workspace);
    let mut report = ReapReport {
        found: found.iter().map(|process| process.pid).collect(),
        bytes_held: found.iter().map(|process| process.rss_bytes).sum(),
        reaped: Vec::new(),
        failed: Vec::new(),
        dry_run,
    };
    if dry_run {
        return report;
    }
    for process in found {
        #[cfg(unix)]
        let result = unsafe { libc::kill(process.pid as i32, libc::SIGTERM) };
        #[cfg(not(unix))]
        let result = -1;
        if result == 0 {
            report.reaped.push(process.pid);
        } else {
            report.failed.push(ReapFailure {
                pid: process.pid,
                error: std::io::Error::last_os_error().to_string(),
            });
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn classifier_preserves_specific_role_order_and_near_miss() {
        assert_eq!(
            role_for("Python -m hcli.agentos.resident --supervise")
                .unwrap()
                .name,
            "resident-supervisor"
        );
        assert_eq!(
            role_for("python tools/x.py hf download Qwen/model")
                .unwrap()
                .name,
            "modellake-download"
        );
        assert_eq!(
            role_for("/bin/sh -c probe --artifact-root /x").map(|role| role.name),
            None
        );
        assert_eq!(
            role_for("/bin/resident --artifact-root /x/model")
                .unwrap()
                .name,
            "resident-body"
        );
        assert_eq!(role_for("/usr/bin/python3 -m pip install requests"), None);
        assert_eq!(
            role_for("hawkingd serve :8011 [KIMI_P0_OPERATIONAL]")
                .unwrap()
                .name,
            "hawkingd"
        );
        assert_eq!(
            role_for(
                "/opt/hcli/current/hawkingd -P -m hcli.hawkingd serve \
                 --model KIMI_P0_OPERATIONAL --port 8011"
            )
            .unwrap()
            .name,
            "hawkingd"
        );
        assert_eq!(
            role_for("/opt/mlx-vlm/bin/python /usr/local/bin/mlx_vlm.server --model /models/kimi")
                .unwrap()
                .name,
            "resident-provider"
        );
        assert_eq!(
            role_for("/opt/mlx/bin/python /usr/local/bin/mlx_lm.server --model /models/kimi")
                .unwrap()
                .name,
            "resident-provider"
        );
        assert_eq!(
            role_for("/opt/open-webui/bin/open-webui serve --port 8080")
                .unwrap()
                .name,
            "web-ui"
        );
        assert_eq!(
            body_for("/opt/mlx-vlm/bin/python mlx_vlm.server --model /models/kimi-vl-a3b-q8"),
            Some("kimi-vl-a3b-q8".into())
        );
    }

    #[test]
    fn body_identity_is_argv_data() {
        assert_eq!(
            body_for("/bin/resident --resident-identity sealed-3.14"),
            Some("sealed-3.14".into())
        );
        assert_eq!(
            body_for("hf download Qwen/model config.json"),
            Some("Qwen/model".into())
        );
        assert_eq!(
            body_for("/bin/resident --artifact-root /x/sealed-3.14"),
            Some("sealed-3.14".into())
        );
        assert_eq!(
            body_for("/bin/resident --artifact-root /x/sealed-3.14 --tokenizer t"),
            None
        );
    }

    #[test]
    fn footprint_parser_keeps_pid_boundaries() {
        let output = "======================================================================\n\
hawkingd [101]: Footprint: 2 MB (16384 bytes per page)\n\
======================================================================\n\
Auxiliary data:\n\
    phys_footprint: 2097152 B\n\
======================================================================\n\
hawkingd [202]: Footprint: 3 MB (16384 bytes per page)\n\
======================================================================\n\
Auxiliary data:\n\
    phys_footprint: 3145728 B\n\
======================================================================\n\
Summary Footprint: 5242880 B\n";
        assert_eq!(
            parse_footprint_bytes(output),
            BTreeMap::from([(101, 2_097_152), (202, 3_145_728)])
        );
    }

    #[test]
    fn state_claims_are_workspace_scoped_and_union_with_cwd() {
        let workspace = tempdir().unwrap();
        let state_dir = workspace.path().join(".hcli/resident");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("state.json"), r#"{"body_pid": 4242}"#).unwrap();
        assert!(claimed_worker_pids(workspace.path()).contains(&4242));
    }

    #[test]
    fn orphan_filter_never_targets_a_claimed_body() {
        let process = HostProcess {
            pid: 4242,
            ppid: 1,
            rss_bytes: 11 * 1024 * 1024 * 1024,
            rss_gib: 11.0,
            memory_source: "rss",
            cpu_percent: 1.0,
            elapsed: "1:00".into(),
            role: "resident-body",
            process_class: "ESSENTIAL_PERSISTENT",
            safe_to_stop: false,
            purpose: "p",
            command: "/bin/resident --resident-identity sealed-3.14".into(),
            body: Some("sealed-3.14".into()),
        };
        assert!(!is_orphaned(&process, &BTreeSet::from([4242])));
        assert!(is_orphaned(&process, &BTreeSet::new()));
    }

    #[test]
    fn orphan_filter_reaps_a_provider_only_after_it_loses_hawkingd() {
        let mut process = HostProcess {
            pid: 4343,
            ppid: 2121,
            rss_bytes: 8 * 1024 * 1024 * 1024,
            rss_gib: 8.0,
            memory_source: "rss",
            cpu_percent: 1.0,
            elapsed: "1:00".into(),
            role: "resident-provider",
            process_class: "ESSENTIAL_PERSISTENT",
            safe_to_stop: false,
            purpose: "p",
            command: "/opt/mlx/bin/python mlx_lm.server --model /models/kimi".into(),
            body: Some("kimi".into()),
        };
        assert!(!is_orphaned(&process, &BTreeSet::new()));
        process.ppid = 1;
        assert!(is_orphaned(&process, &BTreeSet::new()));
    }
}
