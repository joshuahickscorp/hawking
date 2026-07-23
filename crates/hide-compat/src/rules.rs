//! `.claude/rules/**/*.md` discovery and path-glob gating.
//!
//! A rule with no `paths` frontmatter is *un-scoped*: it loads at launch with
//! the same standing as CLAUDE.md memory. A rule that declares `paths` globs is
//! *scoped*: it only applies when a file matching one of those globs is read.

use std::path::{Path, PathBuf};

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::frontmatter;
use crate::layout::Layout;

/// A discovered rule file.
#[derive(Debug, Clone)]
pub struct Rule {
    pub path: PathBuf,
    /// Raw path globs from frontmatter (empty when un-scoped).
    pub paths: Vec<String>,
    /// Whether the rule is scoped to matching-file reads.
    pub scoped: bool,
    /// The rule body (frontmatter stripped).
    pub body: String,
    /// Compiled glob set for `paths`, if any.
    glob_set: Option<GlobSet>,
}

impl Rule {
    /// Un-scoped rules load at launch alongside CLAUDE.md.
    pub fn loads_at_launch(&self) -> bool {
        !self.scoped
    }

    /// Whether this rule applies when `file` (relative to the repo root, or
    /// absolute) is read. Un-scoped rules always apply.
    pub fn applies_to(&self, file: &Path, repo_root: &Path) -> bool {
        if !self.scoped {
            return true;
        }
        let set = match &self.glob_set {
            Some(s) => s,
            None => return false,
        };
        let rel = file.strip_prefix(repo_root).unwrap_or(file);
        set.is_match(file) || set.is_match(rel)
    }
}

/// Discover rules under `<dir>/.claude/rules/**/*.md` for both the repo root and
/// the user home. Sorted by path for deterministic ordering.
pub fn discover(layout: &Layout) -> Vec<Rule> {
    let mut roots = vec![layout.repo_root.join(".claude").join("rules")];
    roots.push(layout.home.join(".claude").join("rules"));

    let mut files: Vec<PathBuf> = Vec::new();
    for root in roots {
        if !root.exists() {
            continue;
        }
        for entry in walkdir::WalkDir::new(&root)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file()
                && entry.path().extension().map(|e| e == "md").unwrap_or(false)
            {
                files.push(entry.into_path());
            }
        }
    }
    files.sort();
    files.dedup();

    files
        .into_iter()
        .filter_map(|path| parse_rule(&path))
        .collect()
}

/// Parse a single rule file.
pub fn parse_rule(path: &Path) -> Option<Rule> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = frontmatter::split(&raw);
    let paths = fm
        .as_ref()
        .map(|f| {
            let mut p = f.list("paths");
            if p.is_empty() {
                // Some ecosystems use `globs:` for the same concept.
                p = f.list("globs");
            }
            p
        })
        .unwrap_or_default();
    let scoped = !paths.is_empty();
    let glob_set = if scoped {
        build_glob_set(&paths).ok()
    } else {
        None
    };
    Some(Rule {
        path: path.to_path_buf(),
        paths,
        scoped,
        body,
        glob_set,
    })
}

fn build_glob_set(patterns: &[String]) -> Result<GlobSet, globset::Error> {
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        builder.add(Glob::new(p)?);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unscoped_rule_loads_at_launch_and_always_applies() {
        let rule = Rule {
            path: PathBuf::from("/repo/.claude/rules/general.md"),
            paths: vec![],
            scoped: false,
            body: "always".into(),
            glob_set: None,
        };
        assert!(rule.loads_at_launch());
        assert!(rule.applies_to(Path::new("/repo/anything.txt"), Path::new("/repo")));
    }

    #[test]
    fn scoped_rule_gates_on_glob() {
        let set = build_glob_set(&["**/*.rs".to_string()]).unwrap();
        let rule = Rule {
            path: PathBuf::from("/repo/.claude/rules/rust.md"),
            paths: vec!["**/*.rs".into()],
            scoped: true,
            body: "rust only".into(),
            glob_set: Some(set),
        };
        assert!(!rule.loads_at_launch());
        assert!(rule.applies_to(Path::new("/repo/src/main.rs"), Path::new("/repo")));
        assert!(!rule.applies_to(Path::new("/repo/README.md"), Path::new("/repo")));
    }
}
