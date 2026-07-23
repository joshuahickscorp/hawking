//! Skill definitions: `SKILL.md` frontmatter.
//!
//! Skills live under `<scope>/.claude/skills/<name>/SKILL.md`. The frontmatter
//! carries invocation metadata (`allowed-tools`, `disable-model-invocation`,
//! `user-invocable`, `context`, `model`, `effort`, `paths`).

use std::path::{Path, PathBuf};

use crate::frontmatter::{self, Frontmatter};
use crate::layout::Layout;

/// A parsed skill.
#[derive(Debug, Clone)]
pub struct Skill {
    pub path: PathBuf,
    pub name: String,
    pub description: Option<String>,
    pub allowed_tools: Vec<String>,
    /// When true the model may not auto-invoke the skill (user must trigger it).
    pub disable_model_invocation: bool,
    /// When true the user can invoke the skill directly (slash command).
    pub user_invocable: bool,
    pub context: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    /// Path globs the skill is scoped to (empty = always available).
    pub paths: Vec<String>,
    /// The skill body (frontmatter stripped).
    pub body: String,
}

impl Skill {
    /// Whether the model is allowed to auto-invoke this skill.
    pub fn model_invocable(&self) -> bool {
        !self.disable_model_invocation
    }
}

/// Parse a single SKILL.md file.
pub fn parse(path: &Path) -> Option<Skill> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = frontmatter::split(&raw);
    let fm = fm.unwrap_or_default();
    Some(from_frontmatter(path, &fm, body))
}

fn from_frontmatter(path: &Path, fm: &Frontmatter, body: String) -> Skill {
    let name = fm.str("name").unwrap_or_else(|| {
        // Fall back to the enclosing directory name (skills/<name>/SKILL.md).
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("skill")
            .to_string()
    });
    Skill {
        path: path.to_path_buf(),
        name,
        description: fm.str("description"),
        allowed_tools: fm.list("allowed-tools"),
        // Default false: model may invoke unless explicitly disabled.
        disable_model_invocation: fm.bool("disable-model-invocation").unwrap_or(false),
        // Default true: user-invocable unless explicitly disabled.
        user_invocable: fm.bool("user-invocable").unwrap_or(true),
        context: fm.str("context"),
        model: fm.str("model"),
        effort: fm.str("effort"),
        paths: fm.list("paths"),
        body,
    }
}

/// Discover skills under the project and user `.claude/skills` trees. Project
/// skills win by name. Sorted by name for determinism.
pub fn discover(layout: &Layout) -> Vec<Skill> {
    let mut by_name: std::collections::BTreeMap<String, Skill> =
        std::collections::BTreeMap::new();

    for dir in [
        layout.home.join(".claude").join("skills"),
        layout.repo_root.join(".claude").join("skills"),
    ] {
        for skill in parse_dir(&dir) {
            by_name.insert(skill.name.clone(), skill);
        }
    }

    by_name.into_values().collect()
}

fn parse_dir(dir: &Path) -> Vec<Skill> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() && entry.file_name() == "SKILL.md" {
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
    fn parses_skill_flags_with_defaults() {
        let fm = frontmatter::parse_block(
            "name: deploy\ndescription: ship it\nallowed-tools: [Bash, Read]\ndisable-model-invocation: true\neffort: high\n",
        );
        let skill = from_frontmatter(Path::new("/x/deploy/SKILL.md"), &fm, String::new());
        assert_eq!(skill.name, "deploy");
        assert_eq!(skill.allowed_tools, vec!["Bash", "Read"]);
        assert!(!skill.model_invocable());
        // user-invocable defaults to true.
        assert!(skill.user_invocable);
        assert_eq!(skill.effort.as_deref(), Some("high"));
    }

    #[test]
    fn name_falls_back_to_dir() {
        let fm = frontmatter::parse_block("description: no name here\n");
        let skill = from_frontmatter(Path::new("/x/myskill/SKILL.md"), &fm, String::new());
        assert_eq!(skill.name, "myskill");
    }
}
