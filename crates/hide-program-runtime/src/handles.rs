//! Host handles: the entire external surface of the sandbox.
//!
//! A program cannot touch the world except by calling a handle from the closed
//! [`HandleName`] set below, and only if the host granted it. This is where "no
//! ambient authority" is realized *by construction*: the enum contains only
//! read-oriented handles. There is no filesystem-write, subprocess-spawn,
//! network-egress, environment-read, or credential handle to name, so no program
//! can express one. A write is never a handle - it is a [`WriteProposal`] handed
//! back to the caller (see `proposal`).
//!
//! The handle names mirror the read tool surface a HIDE agent already exposes:
//! `search.text`, `search.symbol`, `index.references`, `file.read`, `git.diff`,
//! `git.log`, `diagnostic.list`, `test.result.read`, `artifact.read`,
//! `mcp.readonly`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::HandleError;
use crate::value::Value;

/// The closed set of read-oriented capabilities a program may invoke. This is
/// the complete list; adding a mutating capability would require editing this
/// enum, which is the point - the surface is small, auditable, and read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HandleName {
    /// Full-text search over the workspace.
    #[serde(rename = "search.text")]
    SearchText,
    /// Symbol / definition search.
    #[serde(rename = "search.symbol")]
    SearchSymbol,
    /// Reference / call-site lookup from the index.
    #[serde(rename = "index.references")]
    IndexReferences,
    /// Read the contents of a file (read-only).
    #[serde(rename = "file.read")]
    FileRead,
    /// Read a diff between two revisions.
    #[serde(rename = "git.diff")]
    GitDiff,
    /// Read commit history.
    #[serde(rename = "git.log")]
    GitLog,
    /// List diagnostics (compiler / linter output).
    #[serde(rename = "diagnostic.list")]
    DiagnosticList,
    /// Read a recorded test result.
    #[serde(rename = "test.result.read")]
    TestResultRead,
    /// Read a stored artifact.
    #[serde(rename = "artifact.read")]
    ArtifactRead,
    /// Call a read-only MCP method (side-effect-free by contract on the host).
    #[serde(rename = "mcp.readonly")]
    McpReadonly,
}

impl HandleName {
    /// Every handle, in a stable order.
    pub const ALL: [HandleName; 10] = [
        HandleName::SearchText,
        HandleName::SearchSymbol,
        HandleName::IndexReferences,
        HandleName::FileRead,
        HandleName::GitDiff,
        HandleName::GitLog,
        HandleName::DiagnosticList,
        HandleName::TestResultRead,
        HandleName::ArtifactRead,
        HandleName::McpReadonly,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            HandleName::SearchText => "search.text",
            HandleName::SearchSymbol => "search.symbol",
            HandleName::IndexReferences => "index.references",
            HandleName::FileRead => "file.read",
            HandleName::GitDiff => "git.diff",
            HandleName::GitLog => "git.log",
            HandleName::DiagnosticList => "diagnostic.list",
            HandleName::TestResultRead => "test.result.read",
            HandleName::ArtifactRead => "artifact.read",
            HandleName::McpReadonly => "mcp.readonly",
        }
    }

    pub fn from_str(s: &str) -> Option<HandleName> {
        HandleName::ALL.into_iter().find(|h| h.as_str() == s)
    }

    /// Documents the invariant: every handle in this enum is read-oriented.
    /// There is intentionally no variant for which this returns false.
    pub const fn is_read_oriented(&self) -> bool {
        true
    }
}

/// The subset of handles a particular program is allowed to call. The runtime
/// checks membership before dispatching; a call to a non-granted handle is a
/// [`crate::error::RuntimeError::HandleNotGranted`], never a silent success.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HandleGrants(BTreeSet<HandleName>);

impl HandleGrants {
    /// Grant nothing. A program with no grants can still compute over its
    /// literals and produce write proposals, but cannot read the world.
    pub fn none() -> Self {
        Self(BTreeSet::new())
    }

    /// Grant every read handle.
    pub fn all() -> Self {
        Self(HandleName::ALL.into_iter().collect())
    }

    /// Grant exactly the listed handles.
    pub fn of<I: IntoIterator<Item = HandleName>>(handles: I) -> Self {
        Self(handles.into_iter().collect())
    }

    pub fn grant(&mut self, handle: HandleName) -> &mut Self {
        self.0.insert(handle);
        self
    }

    pub fn is_granted(&self, handle: HandleName) -> bool {
        self.0.contains(&handle)
    }

    pub fn granted(&self) -> impl Iterator<Item = HandleName> + '_ {
        self.0.iter().copied()
    }
}

/// A host that answers read handles. This is the ONLY trait the runtime calls
/// out through. Implementations must be deterministic and side-effect-free for a
/// given `(handle, args)` if the caller wants byte-identical program output;
/// the runtime does not and cannot enforce that from inside the sandbox, so it
/// is a host contract.
pub trait HostHandles {
    /// Answer one handle call. `attempt` starts at 0 and increments on each
    /// `retry_with_policy` retry, which lets a fixture model a flaky read
    /// deterministically. `args` is the argument value the program passed.
    fn call(&self, handle: HandleName, args: &Value, attempt: u32) -> Result<Value, HandleError>;
}

/// Adapt a closure into a [`HostHandles`]. Handy for tests and doc examples.
pub struct FnHost<F>(F);

impl<F> FnHost<F>
where
    F: Fn(HandleName, &Value, u32) -> Result<Value, HandleError>,
{
    pub fn new(f: F) -> Self {
        FnHost(f)
    }
}

impl<F> HostHandles for FnHost<F>
where
    F: Fn(HandleName, &Value, u32) -> Result<Value, HandleError>,
{
    fn call(&self, handle: HandleName, args: &Value, attempt: u32) -> Result<Value, HandleError> {
        (self.0)(handle, args, attempt)
    }
}

/// A host with no handles at all: every call fails. Useful when a program is
/// expected to be pure (compute over literals, emit proposals) and you want to
/// prove it never reached for the world.
pub struct DenyAllHost;

impl HostHandles for DenyAllHost {
    fn call(&self, handle: HandleName, _args: &Value, _attempt: u32) -> Result<Value, HandleError> {
        Err(HandleError::new(
            handle.as_str(),
            "no host handle is available",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_names_roundtrip_dotted_strings() {
        for h in HandleName::ALL {
            assert_eq!(HandleName::from_str(h.as_str()), Some(h));
            let json = serde_json::to_string(&h).unwrap();
            assert_eq!(json, format!("\"{}\"", h.as_str()));
        }
        assert_eq!(HandleName::from_str("fs.write"), None);
        assert_eq!(HandleName::from_str("shell.exec"), None);
    }

    #[test]
    fn grants_are_explicit() {
        let g = HandleGrants::of([HandleName::FileRead]);
        assert!(g.is_granted(HandleName::FileRead));
        assert!(!g.is_granted(HandleName::GitLog));
        assert!(HandleGrants::none().granted().next().is_none());
        assert_eq!(HandleGrants::all().granted().count(), 10);
    }

    #[test]
    fn every_handle_is_read_oriented() {
        assert!(HandleName::ALL.iter().all(|h| h.is_read_oriented()));
    }
}
