use hide_core::security::SandboxProfile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorktreeLease {
    pub job_id: String,
    pub branch: String,
    pub path: PathBuf,
    pub sandbox: SandboxProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortLease {
    pub job_id: String,
    pub ports: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct PortAllocator {
    start: u16,
    end: u16,
    leased: BTreeSet<u16>,
}

impl PortAllocator {
    pub fn new(start: u16, end: u16) -> Self {
        Self {
            start,
            end,
            leased: BTreeSet::new(),
        }
    }

    pub fn lease(&mut self, job_id: impl Into<String>, count: u16) -> Option<PortLease> {
        let mut ports = Vec::new();
        for port in self.start..=self.end {
            if !self.leased.contains(&port) {
                ports.push(port);
                if ports.len() == count as usize {
                    break;
                }
            }
        }
        if ports.len() != count as usize {
            return None;
        }
        for port in &ports {
            self.leased.insert(*port);
        }
        Some(PortLease {
            job_id: job_id.into(),
            ports,
        })
    }

    pub fn release(&mut self, lease: &PortLease) {
        for port in &lease.ports {
            self.leased.remove(port);
        }
    }
}
