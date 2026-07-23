//! MCP server definitions with layered, whole-entry-wins precedence, plus
//! cross-agent config discovery (AGENTS.md and `.cursor/rules`).
//!
//! Servers are read from the project `.mcp.json`, the user `~/.claude.json`, and
//! an optional managed config. For a given server name the highest-precedence
//! scope's WHOLE entry wins (no deep field merge). Precedence is
//! Managed > Project > User.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;

use crate::error::{CompatError, Result};
use crate::layout::Layout;

/// The scope an MCP server entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpScope {
    User,
    Project,
    Managed,
}

/// One MCP server entry, kept as its raw JSON so no fields are lost.
#[derive(Debug, Clone)]
pub struct McpServer {
    pub name: String,
    pub scope: McpScope,
    pub entry: Json,
}

impl McpServer {
    /// Convenience: the `command` field if present.
    pub fn command(&self) -> Option<&str> {
        self.entry.get("command").and_then(|v| v.as_str())
    }

    /// Convenience: the transport `type`/`transport` field if present.
    pub fn transport(&self) -> Option<&str> {
        self.entry
            .get("type")
            .or_else(|| self.entry.get("transport"))
            .and_then(|v| v.as_str())
    }
}

/// The resolved MCP configuration plus adjacent cross-agent config.
#[derive(Debug, Clone, Default)]
pub struct McpConfig {
    /// Servers by name after whole-entry precedence resolution.
    pub servers: BTreeMap<String, McpServer>,
    /// Content of a top-level `AGENTS.md`, if present.
    pub agents_md: Option<String>,
    /// Cursor rules discovered under `.cursor/rules` (and `.cursorrules`).
    pub cursor_rules: Vec<CursorRule>,
}

impl McpConfig {
    pub fn server(&self, name: &str) -> Option<&McpServer> {
        self.servers.get(name)
    }
}

/// A single Cursor rule file.
#[derive(Debug, Clone)]
pub struct CursorRule {
    pub path: PathBuf,
    pub body: String,
}

/// Load and resolve the MCP configuration for a layout.
pub fn load(layout: &Layout) -> Result<McpConfig> {
    // Read low -> high precedence so higher scopes overwrite whole entries.
    let user_servers = read_servers(&layout.home.join(".claude.json"), McpScope::User)?;
    let project_servers = read_servers(&layout.repo_root.join(".mcp.json"), McpScope::Project)?;
    let managed_servers = match &layout.managed_mcp {
        Some(p) => read_servers(p, McpScope::Managed)?,
        None => Vec::new(),
    };

    let mut servers: BTreeMap<String, McpServer> = BTreeMap::new();
    for s in user_servers
        .into_iter()
        .chain(project_servers)
        .chain(managed_servers)
    {
        // Whole-entry-wins: a later (higher precedence) scope replaces the entry.
        servers.insert(s.name.clone(), s);
    }

    let agents_md = read_optional(&layout.repo_root.join("AGENTS.md"));
    let cursor_rules = discover_cursor_rules(layout);

    Ok(McpConfig {
        servers,
        agents_md,
        cursor_rules,
    })
}

fn read_servers(path: &Path, scope: McpScope) -> Result<Vec<McpServer>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
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
    Ok(extract_servers(&json, scope))
}

/// Pull `mcpServers` (and, for `~/.claude.json`, nested per-project maps) out of
/// a decoded JSON document.
fn extract_servers(json: &Json, scope: McpScope) -> Vec<McpServer> {
    let mut out: Vec<McpServer> = Vec::new();

    if let Some(map) = json.get("mcpServers").and_then(|v| v.as_object()) {
        for (name, entry) in map {
            out.push(McpServer {
                name: name.clone(),
                scope,
                entry: entry.clone(),
            });
        }
    }

    // ~/.claude.json also stores per-project blocks under `projects`; each may
    // carry its own mcpServers. We flatten them at the same (user) scope; a
    // later duplicate name simply overwrites within this scope's vector, which
    // is fine because cross-scope precedence is applied by the caller.
    if let Some(projects) = json.get("projects").and_then(|v| v.as_object()) {
        for block in projects.values() {
            if let Some(map) = block.get("mcpServers").and_then(|v| v.as_object()) {
                for (name, entry) in map {
                    out.push(McpServer {
                        name: name.clone(),
                        scope,
                        entry: entry.clone(),
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn read_optional(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn discover_cursor_rules(layout: &Layout) -> Vec<CursorRule> {
    let mut rules: Vec<CursorRule> = Vec::new();

    // Legacy single-file `.cursorrules`.
    let legacy = layout.repo_root.join(".cursorrules");
    if let Some(body) = read_optional(&legacy) {
        rules.push(CursorRule {
            path: legacy,
            body,
        });
    }

    // Modern `.cursor/rules/**` (typically `*.mdc`, but read any file).
    let dir = layout.repo_root.join(".cursor").join("rules");
    if dir.exists() {
        let mut files: Vec<PathBuf> = Vec::new();
        for entry in walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                files.push(entry.into_path());
            }
        }
        files.sort();
        for path in files {
            if let Some(body) = read_optional(&path) {
                rules.push(CursorRule { path, body });
            }
        }
    }

    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_pulls_top_level_and_project_servers() {
        let doc = json!({
            "mcpServers": {"a": {"command": "a-cmd"}},
            "projects": {"/some/repo": {"mcpServers": {"b": {"command": "b-cmd"}}}}
        });
        let servers = extract_servers(&doc, McpScope::User);
        let names: Vec<&str> = servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn whole_entry_wins_no_deep_merge() {
        let user = McpServer {
            name: "srv".into(),
            scope: McpScope::User,
            entry: json!({"command": "user-cmd", "args": ["--user"]}),
        };
        let project = McpServer {
            name: "srv".into(),
            scope: McpScope::Project,
            entry: json!({"command": "project-cmd"}),
        };
        let mut servers: BTreeMap<String, McpServer> = BTreeMap::new();
        servers.insert(user.name.clone(), user);
        servers.insert(project.name.clone(), project);
        let s = servers.get("srv").unwrap();
        assert_eq!(s.command(), Some("project-cmd"));
        // The whole entry was replaced: the user-only `args` field is gone.
        assert!(s.entry.get("args").is_none());
    }
}
