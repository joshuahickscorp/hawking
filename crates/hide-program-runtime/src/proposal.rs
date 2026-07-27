//! Write separation: a program may *prepare* a mutation but never commit it.
//!
//! The sandbox has no write capability of any kind. When a program wants to edit
//! a file, run a shell command, reach the network, or mutate any external
//! system, it builds a [`WriteProposal`] describing the intended change. The
//! runtime collects these and returns them in the run output; it never executes
//! one. The proposal then travels the normal action plane, where the real
//! approval + execution machinery lives (outside this crate).
//!
//! This keeps the dangerous half of "tool use" out of the deterministic
//! evaluator entirely: nothing the interpreter does can touch the world.

use serde::{Deserialize, Serialize};

use crate::value::{Citation, Value};

/// The category of a prepared mutation. Mirrors the effect classes the action
/// plane knows how to gate and execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteKind {
    /// A file edit (create / modify / delete). `payload` carries path + diff.
    Edit,
    /// A shell command. `payload` carries the command and cwd.
    Shell,
    /// A network request. `payload` carries method + url + body.
    Network,
    /// Any other external mutation (a connector call, a service action).
    ExternalMutation,
}

impl WriteKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            WriteKind::Edit => "edit",
            WriteKind::Shell => "shell",
            WriteKind::Network => "network",
            WriteKind::ExternalMutation => "external_mutation",
        }
    }

    pub fn from_str(s: &str) -> Option<WriteKind> {
        match s {
            "edit" => Some(WriteKind::Edit),
            "shell" => Some(WriteKind::Shell),
            "network" => Some(WriteKind::Network),
            "external_mutation" => Some(WriteKind::ExternalMutation),
            _ => None,
        }
    }
}

/// A prepared, un-executed mutation. Produced by a program, returned to the
/// caller, executed by nobody inside this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteProposal {
    /// A deterministic id assigned by the runtime in creation order
    /// (`wp-0`, `wp-1`, ...). Lets the program reference the proposal in its
    /// result.
    pub id: String,
    pub kind: WriteKind,
    /// A one-line description of the intended change.
    pub summary: String,
    /// The typed detail the action plane needs to execute it (path, diff,
    /// command, url, ...). Opaque to the runtime.
    pub payload: Value,
    /// The evidence the program used to justify this change, carried forward so
    /// the reviewer sees the provenance.
    #[serde(default)]
    pub citations: Vec<Citation>,
}
