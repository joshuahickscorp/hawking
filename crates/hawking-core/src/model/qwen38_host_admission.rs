//! Host-memory admission for Qwen3.8 decode children.
//!
//! Process-pool children do **not** share artifact pages (measured 2026-08-16:
//! 8.77 GB RSS per child against an 8.5 GB artifact). The binding failure
//! mode is the simultaneous load spike, not warm RSS, and swap does not fail
//! loudly. This gate refuses **before** the box swaps.
//!
//! Shared-session attach cost is workspace + a small overhead. First-process
//! cost uses the measured machine-wide child, not a projection.

use crate::{Error, Result};
use serde::Serialize;
use std::process::Command;

/// 2026-08-16, this box: four `ascension_qwen38_hybrid_greedy` children,
/// same `uniform-q4-v1` artifact, `--max-seq-len 2048`, spawned together.
pub const MEASURED_FOUR_CHILD_SUM_RSS_BYTES: u64 = 35_090_000_000;
pub const MEASURED_FOUR_CHILD_MACHINE_BYTES: u64 = 40_670_000_000;
pub const MEASURED_PROCESS_CHILD_RSS_BYTES: u64 =
    MEASURED_FOUR_CHILD_SUM_RSS_BYTES / 4;
pub const MEASURED_PROCESS_CHILD_MACHINE_BYTES: u64 =
    MEASURED_FOUR_CHILD_MACHINE_BYTES / 4;
pub const MEASURED_FREE_AT_FOUR_CHILD_LOAD_SPIKE_BYTES: u64 = 370_000_000;
pub const MEASURED_FREE_ONCE_WARM_BYTES: u64 = 5_110_000_000;
pub const MEASURED_FREE_BASELINE_BEFORE_BYTES: u64 = 45_780_000_000;
pub const MEASURED_PROCESS_CHILD_SEQ2048_RSS_BYTES: u64 = MEASURED_PROCESS_CHILD_RSS_BYTES;
pub const MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES: u64 = 15_500_000_000;

/// Refuse if free memory after the request would drop below this.
/// 4 GiB is above the 0.37 GB load-spike remainder that nearly swapped.
pub const DEFAULT_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Host-side slop on top of the workspace formula when attaching a session.
pub const SESSION_HOST_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;

pub const ADMISSION_SCHEMA: &str = "hawking.qwen38.host_admission.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionVerdict {
    Admit,
    Refuse,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HostMemory {
    pub page_size_bytes: u64,
    pub pages_free: u64,
    pub pages_purgeable: u64,
    pub pages_speculative: u64,
    pub memsize_bytes: u64,
    pub free_bytes: u64,
}

impl HostMemory {
    pub fn available_bytes(&self) -> u64 {
        self.free_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdmissionRequest {
    pub label: String,
    pub cost_bytes: u64,
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdmissionDecision {
    pub schema: String,
    pub verdict: AdmissionVerdict,
    pub reason: String,
    pub request: AdmissionRequest,
    pub free_before_bytes: u64,
    pub reserve_bytes: u64,
    pub free_after_if_admitted_bytes: Option<u64>,
    pub would_breach_reserve: bool,
}

pub fn parse_vm_stat(text: &str) -> Result<HostMemory> {
    let page_size_bytes = parse_page_size(text)?;
    let pages_free = parse_vm_pages(text, "Pages free:")?;
    let pages_purgeable = parse_vm_pages(text, "Pages purgeable:").unwrap_or(0);
    let pages_speculative = parse_vm_pages(text, "Pages speculative:").unwrap_or(0);
    let memsize_bytes = 96 * 1000 * 1000 * 1000;
    Ok(HostMemory {
        page_size_bytes,
        pages_free,
        pages_purgeable,
        pages_speculative,
        memsize_bytes,
        free_bytes: pages_free.saturating_mul(page_size_bytes),
    })
}

fn parse_page_size(text: &str) -> Result<u64> {
    const MARK: &str = "page size of ";
    let start = text
        .find(MARK)
        .ok_or_else(|| Error::Model("vm_stat missing page size".into()))?;
    let rest = &text[start + MARK.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .map_err(|_| Error::Model("vm_stat page size is not an integer".into()))
}

fn parse_vm_pages(text: &str, label: &str) -> Result<u64> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(label) {
            let token = rest
                .trim()
                .trim_end_matches('.')
                .split_whitespace()
                .next()
                .unwrap_or("");
            return token
                .replace(',', "")
                .parse()
                .map_err(|_| Error::Model(format!("vm_stat {label} is not an integer")));
        }
    }
    Err(Error::Model(format!("vm_stat missing {label}")))
}

pub fn host_memory_snapshot() -> Result<HostMemory> {
    let output = Command::new("vm_stat")
        .output()
        .map_err(|error| Error::Model(format!("vm_stat: {error}")))?;
    if !output.status.success() {
        return Err(Error::Model(format!(
            "vm_stat exited {}",
            output.status
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut snap = parse_vm_stat(&text)?;
    if let Ok(mem) = sysctl_u64(&["hw.memsize"]) {
        snap.memsize_bytes = mem;
    }
    Ok(snap)
}

fn sysctl_u64(args: &[&str]) -> Result<u64> {
    let output = Command::new("sysctl")
        .args(["-n"])
        .args(args)
        .output()
        .map_err(|error| Error::Model(format!("sysctl: {error}")))?;
    if !output.status.success() {
        return Err(Error::Model("sysctl failed".into()));
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .map_err(|_| Error::Model("sysctl value is not u64".into()))
}

/// macOS `ps -o rss=` is kilobytes.
pub fn process_rss_bytes(pid: u32) -> Result<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .map_err(|error| Error::Model(format!("ps: {error}")))?;
    if !output.status.success() {
        return Err(Error::Model(format!("ps -p {pid} failed")));
    }
    let kb: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .map_err(|_| Error::Model("ps rss is not an integer".into()))?;
    Ok(kb.saturating_mul(1024))
}

pub fn decide_admission(
    memory: &HostMemory,
    request: AdmissionRequest,
    reserve_bytes: u64,
) -> AdmissionDecision {
    let free = memory.available_bytes();
    if request.cost_bytes > free {
        return AdmissionDecision {
            schema: ADMISSION_SCHEMA.to_owned(),
            verdict: AdmissionVerdict::Refuse,
            reason: format!(
                "cost {} B exceeds free {} B; refusing before swap",
                request.cost_bytes, free
            ),
            request,
            free_before_bytes: free,
            reserve_bytes,
            free_after_if_admitted_bytes: None,
            would_breach_reserve: true,
        };
    }
    let after = free - request.cost_bytes;
    let would_breach = after < reserve_bytes;
    if would_breach {
        AdmissionDecision {
            schema: ADMISSION_SCHEMA.to_owned(),
            verdict: AdmissionVerdict::Refuse,
            reason: format!(
                "free after request would be {after} B < reserve {reserve_bytes} B; refusing before swap"
            ),
            request,
            free_before_bytes: free,
            reserve_bytes,
            free_after_if_admitted_bytes: Some(after),
            would_breach_reserve: true,
        }
    } else {
        AdmissionDecision {
            schema: ADMISSION_SCHEMA.to_owned(),
            verdict: AdmissionVerdict::Admit,
            reason: format!(
                "free {free} B - cost {} B leaves {after} B (>= reserve {reserve_bytes} B)",
                request.cost_bytes
            ),
            request,
            free_before_bytes: free,
            reserve_bytes,
            free_after_if_admitted_bytes: Some(after),
            would_breach_reserve: false,
        }
    }
}

pub fn process_pool_child_cost_bytes(max_seq_len: usize) -> u64 {
    if max_seq_len <= 2048 {
        return MEASURED_PROCESS_CHILD_MACHINE_BYTES;
    }
    if max_seq_len >= 8192 {
        return MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES;
    }
    let lo = MEASURED_PROCESS_CHILD_MACHINE_BYTES;
    let hi = MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES;
    let span = (8192 - 2048) as u64;
    let t = (max_seq_len as u64 - 2048) as u128;
    lo + ((u128::from(hi - lo) * t) / u128::from(span)) as u64
}

pub fn shared_session_attach_cost_bytes(workspace_bytes: u64) -> u64 {
    workspace_bytes.saturating_add(SESSION_HOST_OVERHEAD_BYTES)
}

pub fn first_shared_process_cost_bytes(max_seq_len: usize) -> u64 {
    process_pool_child_cost_bytes(max_seq_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VM: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  1000.
Pages active:                                2000.
Pages speculative:                             10.
Pages purgeable:                                5.
";

    #[test]
    fn parse_vm_stat_free_is_pages_times_page_size() {
        let snap = parse_vm_stat(VM).expect("parse");
        assert_eq!(snap.page_size_bytes, 16384);
        assert_eq!(snap.pages_free, 1000);
        assert_eq!(snap.free_bytes, 1000 * 16384);
        assert_eq!(snap.pages_purgeable, 5);
        assert_eq!(snap.pages_speculative, 10);
    }

    #[test]
    fn refuses_when_cost_exceeds_free() {
        let memory = parse_vm_stat(VM).unwrap();
        let decision = decide_admission(
            &memory,
            AdmissionRequest {
                label: "oversub".into(),
                cost_bytes: memory.free_bytes + 1,
                kind: "session".into(),
            },
            DEFAULT_RESERVE_BYTES,
        );
        assert_eq!(decision.verdict, AdmissionVerdict::Refuse);
        assert!(decision.would_breach_reserve);
        assert!(decision.reason.contains("refusing before swap"));
    }

    #[test]
    fn refuses_when_remainder_would_undercut_reserve() {
        let memory = parse_vm_stat(VM).unwrap();
        let reserve = memory.free_bytes / 2;
        let decision = decide_admission(
            &memory,
            AdmissionRequest {
                label: "tight".into(),
                cost_bytes: memory.free_bytes - reserve + 1,
                kind: "session".into(),
            },
            reserve,
        );
        assert_eq!(decision.verdict, AdmissionVerdict::Refuse);
        assert_eq!(decision.free_after_if_admitted_bytes, Some(reserve - 1));
    }

    #[test]
    fn admits_when_remainder_stays_above_reserve() {
        let memory = parse_vm_stat(VM).unwrap();
        let reserve = 1024;
        let decision = decide_admission(
            &memory,
            AdmissionRequest {
                label: "ok".into(),
                cost_bytes: 4096,
                kind: "session".into(),
            },
            reserve,
        );
        assert_eq!(decision.verdict, AdmissionVerdict::Admit);
        assert_eq!(
            decision.free_after_if_admitted_bytes,
            Some(memory.free_bytes - 4096)
        );
    }

    #[test]
    fn process_pool_cost_uses_measured_anchors_only() {
        assert_eq!(
            process_pool_child_cost_bytes(2048),
            MEASURED_PROCESS_CHILD_MACHINE_BYTES
        );
        assert_eq!(
            process_pool_child_cost_bytes(128),
            MEASURED_PROCESS_CHILD_MACHINE_BYTES
        );
        assert_eq!(
            process_pool_child_cost_bytes(8192),
            MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES
        );
        let mid = process_pool_child_cost_bytes(4096);
        assert!(mid > MEASURED_PROCESS_CHILD_MACHINE_BYTES);
        assert!(mid < MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES);
    }
}
