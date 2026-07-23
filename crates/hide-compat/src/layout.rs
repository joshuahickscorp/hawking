//! Scope layout: the set of filesystem locations the compatibility readers scan.
//!
//! Every reader takes a `&Layout` rather than touching process globals, which
//! keeps the whole crate deterministic and lets tests point every scope at a
//! tempdir. `Layout::discover` is a convenience for real use that fills the
//! scopes from `$HOME` and the nearest enclosing git repo.

use std::path::{Path, PathBuf};

/// Filesystem locations for every configuration scope.
#[derive(Debug, Clone)]
pub struct Layout {
    /// The working directory the agent was launched from.
    pub cwd: PathBuf,
    /// The repository root (chain of CLAUDE.md files stops here going up).
    pub repo_root: PathBuf,
    /// The user home directory (`~/.claude`, `~/.claude.json` live under it).
    pub home: PathBuf,
    /// Optional managed (enterprise) settings.json path.
    pub managed_settings: Option<PathBuf>,
    /// Optional managed MCP config path.
    pub managed_mcp: Option<PathBuf>,
}

impl Layout {
    /// Construct an explicit layout. Prefer this in tests.
    pub fn new(
        repo_root: impl AsRef<Path>,
        cwd: impl AsRef<Path>,
        home: impl AsRef<Path>,
    ) -> Self {
        Layout {
            cwd: cwd.as_ref().to_path_buf(),
            repo_root: repo_root.as_ref().to_path_buf(),
            home: home.as_ref().to_path_buf(),
            managed_settings: None,
            managed_mcp: None,
        }
    }

    pub fn with_managed_settings(mut self, path: impl AsRef<Path>) -> Self {
        self.managed_settings = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn with_managed_mcp(mut self, path: impl AsRef<Path>) -> Self {
        self.managed_mcp = Some(path.as_ref().to_path_buf());
        self
    }

    /// Discover a layout from a starting directory: walk up to the nearest
    /// `.git`, read `$HOME` for the user scope. Falls back to `cwd` as the repo
    /// root when no `.git` is found.
    pub fn discover(cwd: impl AsRef<Path>) -> Self {
        let cwd = cwd.as_ref().to_path_buf();
        let repo_root = find_repo_root(&cwd).unwrap_or_else(|| cwd.clone());
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| cwd.clone());
        Layout {
            cwd,
            repo_root,
            home,
            managed_settings: None,
            managed_mcp: None,
        }
    }

    /// The directories from `repo_root` down to `cwd`, root first. When `cwd` is
    /// not inside `repo_root` the chain collapses to just `cwd`.
    pub fn dir_chain_root_first(&self) -> Vec<PathBuf> {
        let cwd = normalize(&self.cwd);
        let root = normalize(&self.repo_root);

        // Collect cwd and ancestors until we pass root.
        let mut up: Vec<PathBuf> = Vec::new();
        let mut cursor = cwd.clone();
        loop {
            up.push(cursor.clone());
            if cursor == root {
                up.reverse();
                return up;
            }
            match cursor.parent() {
                Some(p) => cursor = p.to_path_buf(),
                None => break,
            }
        }
        // root was never reached (cwd outside repo_root): just cwd.
        vec![cwd]
    }
}

fn normalize(p: &Path) -> PathBuf {
    // Lexical normalisation only (no filesystem access): collapse `.` segments.
    // We do not resolve symlinks; the readers operate on the paths as given.
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut cursor = start;
    loop {
        if cursor.join(".git").exists() {
            return Some(cursor.to_path_buf());
        }
        cursor = cursor.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_is_root_first_and_inclusive() {
        let layout = Layout::new("/repo", "/repo/a/b", "/home/u");
        let chain = layout.dir_chain_root_first();
        assert_eq!(
            chain,
            vec![
                PathBuf::from("/repo"),
                PathBuf::from("/repo/a"),
                PathBuf::from("/repo/a/b"),
            ]
        );
    }

    #[test]
    fn chain_when_cwd_equals_root() {
        let layout = Layout::new("/repo", "/repo", "/home/u");
        assert_eq!(layout.dir_chain_root_first(), vec![PathBuf::from("/repo")]);
    }

    #[test]
    fn chain_when_cwd_outside_root() {
        let layout = Layout::new("/repo", "/elsewhere/x", "/home/u");
        assert_eq!(
            layout.dir_chain_root_first(),
            vec![PathBuf::from("/elsewhere/x")]
        );
    }
}
