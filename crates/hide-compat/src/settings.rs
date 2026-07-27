//! settings.json readers across scopes, with two separate precedence rules.
//!
//! Scalar settings resolve with precedence Managed > CLI > Local > Project >
//! User (highest wins). Permission arrays (`allow`/`deny`/`ask`) do not override;
//! they MERGE across every scope, and within the merged set a `deny` beats an
//! `allow` beats an `ask`. Instruction layers use a *separate* precedence,
//! Managed > User > Project > Local, applied read-last-wins (so the highest
//! precedence layer is read last).

use std::collections::BTreeMap;

use globset::{Glob, GlobSet, GlobSetBuilder};
use serde_json::Value as Json;

use crate::error::{CompatError, Result};
use crate::layout::Layout;

/// A configuration scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    User,
    Project,
    Local,
    Cli,
    Managed,
}

/// The permission decision for a tool invocation string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Ask,
    /// No rule matched in any scope.
    Undecided,
}

/// Merged permission rule sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Permissions {
    pub allow: Vec<String>,
    pub deny: Vec<String>,
    pub ask: Vec<String>,
}

impl Permissions {
    fn extend_from(&mut self, other: &Permissions) {
        for a in &other.allow {
            if !self.allow.contains(a) {
                self.allow.push(a.clone());
            }
        }
        for d in &other.deny {
            if !self.deny.contains(d) {
                self.deny.push(d.clone());
            }
        }
        for k in &other.ask {
            if !self.ask.contains(k) {
                self.ask.push(k.clone());
            }
        }
    }

    /// Decide a tool string. Deny wins, then allow, then ask. Matching is exact
    /// on the rule string OR by a glob compiled from the rule (so a rule like
    /// `Bash(git *)` matches `Bash(git status)`).
    pub fn decide(&self, tool: &str) -> Decision {
        if matches_any(&self.deny, tool) {
            Decision::Deny
        } else if matches_any(&self.allow, tool) {
            Decision::Allow
        } else if matches_any(&self.ask, tool) {
            Decision::Ask
        } else {
            Decision::Undecided
        }
    }
}

fn matches_any(rules: &[String], tool: &str) -> bool {
    for r in rules {
        if r == tool {
            return true;
        }
        // Best-effort glob: only attempt when the rule looks like a pattern.
        if r.contains('*') {
            if let Ok(glob) = Glob::new(r) {
                if glob.compile_matcher().is_match(tool) {
                    return true;
                }
            }
        }
    }
    false
}

/// One scope's raw settings, parsed from a settings.json.
#[derive(Debug, Clone, Default)]
pub struct RawSettings {
    pub scope_present: bool,
    pub permissions: Permissions,
    pub claude_md_excludes: Vec<String>,
    /// The instruction string a scope contributes (from `instructions`, falling
    /// back to `additionalInstructions`).
    pub instructions: Option<String>,
    /// All top-level scalar/object settings, for scalar precedence resolution.
    pub values: BTreeMap<String, Json>,
}

/// The fully resolved settings for a layout.
#[derive(Debug, Clone, Default)]
pub struct ResolvedSettings {
    /// Merged permission rules (deny wins on decision).
    pub permissions: Permissions,
    /// Scalar settings after Managed > CLI > Local > Project > User resolution.
    pub values: BTreeMap<String, Json>,
    /// Instruction layers in application order (read-last wins). The last element
    /// is the highest precedence (Managed if present).
    pub instruction_layers: Vec<(Scope, String)>,
    /// Merged `claudeMdExcludes` globs across all scopes.
    pub claude_md_excludes: Vec<String>,
}

impl ResolvedSettings {
    pub fn decide(&self, tool: &str) -> Decision {
        self.permissions.decide(tool)
    }

    /// Compile `claudeMdExcludes` into a glob set for CLAUDE.md discovery.
    pub fn excludes_glob_set(&self) -> Result<Option<GlobSet>> {
        if self.claude_md_excludes.is_empty() {
            return Ok(None);
        }
        let mut builder = GlobSetBuilder::new();
        for g in &self.claude_md_excludes {
            let glob = Glob::new(g).map_err(|e| CompatError::Glob {
                glob: g.clone(),
                source: e,
            })?;
            builder.add(glob);
        }
        let set = builder.build().map_err(|e| CompatError::Glob {
            glob: self.claude_md_excludes.join(","),
            source: e,
        })?;
        Ok(Some(set))
    }

    /// The effective (highest-precedence) instruction string, or None.
    pub fn effective_instructions(&self) -> Option<&str> {
        self.instruction_layers.last().map(|(_, s)| s.as_str())
    }
}

/// Parse a settings.json file into `RawSettings`. Missing file is not an error;
/// it yields an absent scope.
pub fn parse_file(path: &std::path::Path) -> Result<RawSettings> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RawSettings::default());
        }
        Err(e) => {
            return Err(CompatError::Io {
                path: path.display().to_string(),
                source: e,
            })
        }
    };
    let json: Json = serde_json::from_str(&text).map_err(|e| CompatError::Json {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(parse_value(json))
}

/// Parse an already-decoded JSON object into `RawSettings` (used for CLI scope).
pub fn parse_value(json: Json) -> RawSettings {
    let mut raw = RawSettings {
        scope_present: true,
        ..Default::default()
    };
    let obj = match json.as_object() {
        Some(o) => o,
        None => return raw,
    };

    if let Some(perms) = obj.get("permissions").and_then(|v| v.as_object()) {
        raw.permissions.allow = string_array(perms.get("allow"));
        raw.permissions.deny = string_array(perms.get("deny"));
        raw.permissions.ask = string_array(perms.get("ask"));
    }

    raw.claude_md_excludes = string_array(obj.get("claudeMdExcludes"));

    raw.instructions = obj
        .get("instructions")
        .and_then(|v| v.as_str())
        .or_else(|| obj.get("additionalInstructions").and_then(|v| v.as_str()))
        .map(|s| s.to_string());

    for (k, v) in obj {
        raw.values.insert(k.clone(), v.clone());
    }

    raw
}

fn string_array(v: Option<&Json>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Load and resolve settings for a layout. `cli` optionally supplies the CLI
/// scope (from parsed command-line flags).
pub fn load(layout: &Layout, cli: Option<RawSettings>) -> Result<ResolvedSettings> {
    let user = parse_file(&layout.home.join(".claude").join("settings.json"))?;
    let project = parse_file(&layout.repo_root.join(".claude").join("settings.json"))?;
    let local = parse_file(
        &layout
            .repo_root
            .join(".claude")
            .join("settings.local.json"),
    )?;
    let managed = match &layout.managed_settings {
        Some(p) => parse_file(p)?,
        None => RawSettings::default(),
    };
    let cli = cli.unwrap_or_default();

    resolve(&user, &project, &local, &cli, &managed)
}

/// Resolve raw scopes into a `ResolvedSettings` (exposed for direct testing).
pub fn resolve(
    user: &RawSettings,
    project: &RawSettings,
    local: &RawSettings,
    cli: &RawSettings,
    managed: &RawSettings,
) -> Result<ResolvedSettings> {
    // Permissions MERGE across every present scope; order does not matter since
    // deny/allow/ask are unioned and decided by deny-wins.
    let mut permissions = Permissions::default();
    let mut claude_md_excludes: Vec<String> = Vec::new();
    for s in [user, project, local, cli, managed] {
        if !s.scope_present {
            continue;
        }
        permissions.extend_from(&s.permissions);
        for g in &s.claude_md_excludes {
            if !claude_md_excludes.contains(g) {
                claude_md_excludes.push(g.clone());
            }
        }
    }

    // Scalar values: apply low -> high so the highest precedence overwrites.
    // Precedence: Managed > CLI > Local > Project > User.
    let mut values: BTreeMap<String, Json> = BTreeMap::new();
    for s in [user, project, local, cli, managed] {
        if !s.scope_present {
            continue;
        }
        for (k, v) in &s.values {
            values.insert(k.clone(), v.clone());
        }
    }

    // Instruction layers: SEPARATE precedence Managed > User > Project > Local.
    // read-last-wins => apply lowest first: Local, Project, User, Managed.
    let mut instruction_layers: Vec<(Scope, String)> = Vec::new();
    let ordered = [
        (Scope::Local, local),
        (Scope::Project, project),
        (Scope::User, user),
        (Scope::Managed, managed),
    ];
    for (scope, s) in ordered {
        if !s.scope_present {
            continue;
        }
        if let Some(instr) = &s.instructions {
            instruction_layers.push((scope, instr.clone()));
        }
    }

    Ok(ResolvedSettings {
        permissions,
        values,
        instruction_layers,
        claude_md_excludes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn raw(json: Json) -> RawSettings {
        parse_value(json)
    }

    #[test]
    fn permissions_merge_and_deny_wins() {
        let user = raw(json!({"permissions": {"allow": ["Bash(git status)"]}}));
        let project = raw(json!({"permissions": {"deny": ["Bash(git status)"]}}));
        let empty = RawSettings::default();
        let resolved = resolve(&user, &project, &empty, &empty, &empty).unwrap();
        // Present in both allow and deny after merge; deny must win.
        assert_eq!(resolved.decide("Bash(git status)"), Decision::Deny);
    }

    #[test]
    fn scalar_precedence_managed_over_user() {
        let user = raw(json!({"model": "user-model"}));
        let managed = raw(json!({"model": "managed-model"}));
        let empty = RawSettings::default();
        let resolved = resolve(&user, &empty, &empty, &empty, &managed).unwrap();
        assert_eq!(
            resolved.values.get("model").and_then(|v| v.as_str()),
            Some("managed-model")
        );
    }

    #[test]
    fn instruction_layers_read_last_wins_managed_highest() {
        let local = raw(json!({"instructions": "local"}));
        let project = raw(json!({"instructions": "project"}));
        let user = raw(json!({"instructions": "user"}));
        let managed = raw(json!({"instructions": "managed"}));
        let resolved = resolve(&user, &project, &local, &RawSettings::default(), &managed).unwrap();
        let order: Vec<Scope> = resolved.instruction_layers.iter().map(|(s, _)| *s).collect();
        assert_eq!(
            order,
            vec![Scope::Local, Scope::Project, Scope::User, Scope::Managed]
        );
        assert_eq!(resolved.effective_instructions(), Some("managed"));
    }

    #[test]
    fn glob_permission_matches() {
        let project = raw(json!({"permissions": {"allow": ["Bash(git *)"]}}));
        let empty = RawSettings::default();
        let resolved = resolve(&empty, &project, &empty, &empty, &empty).unwrap();
        assert_eq!(resolved.decide("Bash(git push)"), Decision::Allow);
    }
}
