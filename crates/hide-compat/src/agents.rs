//! Subagent definitions: `.claude/agents/*.md` and `~/.claude/agents/*.md`.
//!
//! Each agent is a markdown file with YAML frontmatter. `disallowedTools` is
//! applied before `tools` when computing the effective tool set, so a tool that
//! is both allowed and disallowed is removed. `model` defaults to `inherit`.

use std::path::{Path, PathBuf};

use crate::frontmatter::{self, Frontmatter};
use crate::layout::Layout;

/// A parsed subagent definition.
#[derive(Debug, Clone)]
pub struct Agent {
    pub path: PathBuf,
    pub name: String,
    pub description: Option<String>,
    pub tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    /// Defaults to "inherit".
    pub model: String,
    pub skills: Vec<String>,
    pub mcp: Vec<String>,
    pub hooks: Vec<String>,
    pub memory: Vec<String>,
    pub permissions: Vec<String>,
    /// The agent's system-prompt body (frontmatter stripped).
    pub body: String,
}

impl Agent {
    /// Tools the agent may actually use: `tools` minus `disallowedTools`.
    /// disallowed is applied before allow.
    pub fn effective_tools(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter(|t| !self.disallowed_tools.contains(t))
            .cloned()
            .collect()
    }

    /// Whether `model` is the inherit sentinel.
    pub fn inherits_model(&self) -> bool {
        self.model.eq_ignore_ascii_case("inherit")
    }

    /// Whether this profile permits `tool`. This is the gate predicate: deny
    /// wins, and an EMPTY `tools` list means "inherit every tool" (omitting the
    /// `tools:` key declines to narrow the set, it does not strip the agent of
    /// tools).
    ///
    /// [`Self::effective_tools`] answers a different question, the explicitly
    /// listed set, and therefore returns empty for an inherit-all profile. A
    /// caller that gates on it would read that empty vec as deny-all (or, worse,
    /// as allow-all while silently dropping `disallowedTools`), so gate on this
    /// instead.
    pub fn allows_tool(&self, tool: &str) -> bool {
        if self.disallowed_tools.iter().any(|t| t == tool) {
            return false;
        }
        self.tools.is_empty() || self.tools.iter().any(|t| t == tool)
    }
}

/// Parse a single agent markdown file.
pub fn parse(path: &Path) -> Option<Agent> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = frontmatter::split(&raw);
    let fm = fm.unwrap_or_default();
    Some(from_frontmatter(path, &fm, body))
}

fn from_frontmatter(path: &Path, fm: &Frontmatter, body: String) -> Agent {
    let name = fm
        .str("name")
        .unwrap_or_else(|| stem(path).to_string());
    Agent {
        path: path.to_path_buf(),
        name,
        description: fm.str("description"),
        tools: fm.list("tools"),
        disallowed_tools: fm.list("disallowedTools"),
        model: fm.str("model").unwrap_or_else(|| "inherit".to_string()),
        skills: fm.list("skills"),
        mcp: fm.list("mcp"),
        hooks: fm.list("hooks"),
        memory: fm.list("memory"),
        permissions: fm.list("permissions"),
        body,
    }
}

fn stem(path: &Path) -> &str {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("agent")
}

/// Discover all agents. Project agents (`<root>/.claude/agents`) take precedence
/// over user agents (`~/.claude/agents`) of the same name. Returns them sorted
/// by name for determinism.
pub fn discover(layout: &Layout) -> Vec<Agent> {
    let mut by_name: std::collections::BTreeMap<String, Agent> =
        std::collections::BTreeMap::new();

    // User first, then project overrides by name.
    for dir in [
        layout.home.join(".claude").join("agents"),
        layout.repo_root.join(".claude").join("agents"),
    ] {
        for agent in parse_dir(&dir) {
            by_name.insert(agent.name.clone(), agent);
        }
    }

    by_name.into_values().collect()
}

fn parse_dir(dir: &Path) -> Vec<Agent> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file()
            && entry.path().extension().map(|e| e == "md").unwrap_or(false)
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    files.iter().filter_map(|p| parse(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disallowed_applied_before_allow() {
        let fm = frontmatter::parse_block(
            "name: reviewer\ntools: [Read, Write, Bash]\ndisallowedTools: [Bash]\n",
        );
        let agent = from_frontmatter(Path::new("/x/reviewer.md"), &fm, String::new());
        assert_eq!(agent.effective_tools(), vec!["Read", "Write"]);
    }

    #[test]
    fn empty_tool_list_inherits_all_and_deny_still_wins() {
        let fm = frontmatter::parse_block("name: reviewer\ndisallowedTools: [Bash]\n");
        let agent = from_frontmatter(Path::new("/x/reviewer.md"), &fm, String::new());
        // No `tools:` key, so the explicit list is empty ...
        assert!(agent.effective_tools().is_empty());
        // ... but the profile still permits Read and still denies Bash.
        assert!(agent.allows_tool("Read"));
        assert!(!agent.allows_tool("Bash"));
    }

    #[test]
    fn explicit_tool_list_is_a_closed_set() {
        let fm = frontmatter::parse_block("name: r\ntools: [Read]\ndisallowedTools: [Bash]\n");
        let agent = from_frontmatter(Path::new("/x/r.md"), &fm, String::new());
        assert!(agent.allows_tool("Read"));
        assert!(!agent.allows_tool("Write"));
        assert!(!agent.allows_tool("Bash"));
    }

    #[test]
    fn model_defaults_to_inherit() {
        let fm = frontmatter::parse_block("name: a\n");
        let agent = from_frontmatter(Path::new("/x/a.md"), &fm, String::new());
        assert!(agent.inherits_model());
    }
}
