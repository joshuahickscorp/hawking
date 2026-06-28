//! The Living-Index daemon: incremental indexer (bible §4.9).
//!
//! Watches the workspace (notify FSEvents) with debounce, runs a merkle diff to
//! find the exact changed files (the watcher is a *hint*; merkle decides the
//! work), reparses only changed/renamed files, updates the store, and advances a
//! generation with crash recovery. A runnable loop (`run_until`) is provided.

use crate::merkle::{Blake3MerkleScanner, ChangeSet, MerkleNode, MerkleScanner};
use crate::query::SqliteCodeIndex;
use hide_core::Result;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDaemonConfig {
    pub debounce_ms: u64,
    pub idle_reindex_after_ms: u64,
    pub max_concurrent_lsp: usize,
}

impl Default for IndexDaemonConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 200,
            idle_reindex_after_ms: 15_000,
            max_concurrent_lsp: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDaemonState {
    pub generation: u64,
    pub queue_depth: usize,
    pub is_idle: bool,
    pub last_error: Option<String>,
}

/// The incremental indexer. Owns the index + the merkle snapshot, and applies a
/// changeset (add/modify/delete/rename) against the store with MVCC.
pub struct IndexDaemon {
    root: PathBuf,
    index: Arc<SqliteCodeIndex>,
    scanner: Blake3MerkleScanner,
    /// Last committed merkle snapshot (the recovery + diff anchor).
    snapshot: Option<MerkleNode>,
    config: IndexDaemonConfig,
}

impl IndexDaemon {
    pub fn new(root: impl Into<PathBuf>, index: Arc<SqliteCodeIndex>) -> Self {
        let root = root.into();
        Self {
            scanner: Blake3MerkleScanner::new(root.clone()),
            root,
            index,
            snapshot: None,
            config: IndexDaemonConfig::default(),
        }
    }

    pub fn with_config(mut self, config: IndexDaemonConfig) -> Self {
        self.config = config;
        self
    }

    pub fn index(&self) -> Arc<SqliteCodeIndex> {
        self.index.clone()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Crash recovery + cold/warm start: truncate any torn generation, then
    /// merkle-diff the current tree against the last snapshot (catching edits made
    /// while the daemon was down) and apply. Returns the committed generation.
    pub fn bootstrap(&mut self) -> Result<u64> {
        // Recover the store to its last good generation (drop torn tails).
        let _recovered = self.index.store().recover()?;

        let current = self.scanner.scan_workspace()?;
        let changeset = match &self.snapshot {
            Some(prev) => self.scanner.diff(prev, &current)?,
            None => {
                // Cold start: everything is an add.
                let mut cs = ChangeSet::default();
                collect_all_files(&current, &mut cs.added);
                cs
            }
        };
        self.apply_changeset(&changeset)?;
        self.snapshot = Some(current);
        Ok(self.index.generation())
    }

    /// Apply a changeset to the index: reparse adds/modifies/rename-dests, remove
    /// deletes. (Rename sources keep their content; the dest is reparsed — a
    /// future optimization is a pure ID remap, but reparse is always correct.)
    pub fn apply_changeset(&self, cs: &ChangeSet) -> Result<()> {
        for path in &cs.removed {
            if let Some(rel) = self.rel(path) {
                self.index.store().remove_file(&rel)?;
            }
        }
        for (old, _new) in &cs.renamed {
            // remove the old path; the new path is reparsed below via dirty_paths
            if let Some(rel) = self.rel(old) {
                self.index.store().remove_file(&rel)?;
            }
        }
        for path in cs.dirty_paths() {
            self.index_one(&path)?;
        }
        Ok(())
    }

    /// Index one file from disk (skips unreadable / non-UTF8 files).
    pub fn index_one(&self, path: &Path) -> Result<()> {
        let rel = match self.rel(path) {
            Some(r) => r,
            None => return Ok(()),
        };
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Ok(()), // file vanished between diff and read — ok
        };
        let hash = blake3::hash(&bytes).to_hex().to_string();
        // Skip if unchanged (merkle gate already filters, but this is the backstop).
        if let Ok(Some(existing)) = self.index.store().file_hash(&rel) {
            if existing == hash {
                return Ok(());
            }
        }
        let content = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return Ok(()), // binary file — lexical-only path could go here
        };
        self.index.index_text(&rel, &content, &hash)?;
        Ok(())
    }

    fn rel(&self, path: &Path) -> Option<String> {
        path.strip_prefix(&self.root)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
            .or_else(|| Some(path.to_string_lossy().to_string()))
    }

    /// Rescan the tree, diff against the snapshot, apply, advance the snapshot.
    /// This is the "on wake" path: the FS event is only a hint, merkle decides.
    pub fn tick(&mut self) -> Result<ChangeSet> {
        let current = self.scanner.scan_workspace()?;
        let cs = match &self.snapshot {
            Some(prev) => self.scanner.diff(prev, &current)?,
            None => {
                let mut cs = ChangeSet::default();
                collect_all_files(&current, &mut cs.added);
                cs
            }
        };
        if !cs.is_empty() {
            self.apply_changeset(&cs)?;
        }
        self.snapshot = Some(current);
        Ok(cs)
    }

    pub fn state(&self) -> IndexDaemonState {
        IndexDaemonState {
            generation: self.index.generation(),
            queue_depth: 0,
            is_idle: true,
            last_error: None,
        }
    }

    /// Run the watch loop until `stop()` returns true. Sets up a notify watcher on
    /// the root (recursively, on directories), debounces bursts, and on each
    /// settled burst runs `tick()`. Blocking; intended for a dedicated thread.
    ///
    /// `stop` is polled between debounce windows so the loop is cancellable.
    pub fn run_until<F: Fn() -> bool>(&mut self, stop: F) -> Result<()> {
        self.bootstrap()?;

        let (tx, rx) = channel::<notify::Result<notify::Event>>();
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = tx.send(res);
            },
            notify::Config::default(),
        )
        .map_err(|e| hide_core::HideError::Storage(format!("watcher: {e}")))?;
        watcher
            .watch(&self.root, RecursiveMode::Recursive)
            .map_err(|e| hide_core::HideError::Storage(format!("watch root: {e}")))?;

        let debounce = Duration::from_millis(self.config.debounce_ms);
        loop {
            if stop() {
                break;
            }
            // Block for the first event (with a timeout so we can poll `stop`).
            match rx.recv_timeout(Duration::from_millis(250)) {
                Ok(_evt) => {
                    drain_burst(&rx, debounce);
                    // The event(s) are only a hint; merkle decides the real work.
                    let _ = self.tick();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    }
}

/// Drain a burst of FS events for `window`, coalescing them (debounce).
fn drain_burst(rx: &Receiver<notify::Result<notify::Event>>, window: Duration) {
    let deadline = Instant::now() + window;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok(_) => continue, // keep draining within the window
            Err(_) => break,
        }
    }
}

fn collect_all_files(node: &MerkleNode, out: &mut Vec<PathBuf>) {
    match node.kind {
        crate::merkle::MerkleKind::File => out.push(node.path.clone()),
        crate::merkle::MerkleKind::Directory => {
            for c in &node.children {
                collect_all_files(c, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, rel: &str, content: &str) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, content).unwrap();
    }

    #[test]
    fn bootstrap_cold_indexes_all_files() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "src/a.rs", "pub fn alpha() {}");
        write(tmp.path(), "src/b.rs", "pub fn beta() { alpha(); }");
        let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
        let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
        let gen = daemon.bootstrap().unwrap();
        assert!(gen >= 2, "two files indexed → generation advanced");
        assert_eq!(index.store().file_count().unwrap(), 2);
    }

    #[test]
    fn tick_applies_incremental_change() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "a.rs", "pub fn alpha() {}");
        let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
        let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
        daemon.bootstrap().unwrap();
        assert_eq!(index.store().file_count().unwrap(), 1);

        // add a new file, modify the old one, then tick.
        write(tmp.path(), "b.rs", "pub fn beta() {}");
        write(tmp.path(), "a.rs", "pub fn alpha_renamed() {}");
        let cs = daemon.tick().unwrap();
        assert!(cs.added.iter().any(|p| p.ends_with("b.rs")));
        assert!(cs.modified.iter().any(|p| p.ends_with("a.rs")));
        assert_eq!(index.store().file_count().unwrap(), 2);
    }

    #[test]
    fn tick_handles_deletion_and_rename() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "keep.rs", "pub fn keep() {}");
        write(tmp.path(), "gone.rs", "pub fn gone() {}");
        write(tmp.path(), "old.rs", "pub fn stable_unique_body() {}");
        let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
        let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
        daemon.bootstrap().unwrap();

        fs::remove_file(tmp.path().join("gone.rs")).unwrap();
        fs::rename(tmp.path().join("old.rs"), tmp.path().join("new.rs")).unwrap();
        let cs = daemon.tick().unwrap();
        assert!(cs.removed.iter().any(|p| p.ends_with("gone.rs")));
        assert_eq!(cs.renamed.len(), 1, "rename detected, got {cs:?}");
        // store reflects: keep + new (gone removed, old→new)
        assert_eq!(index.store().file_count().unwrap(), 2);
    }

    #[test]
    fn bootstrap_recovers_from_torn_generation() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "a.rs", "pub fn a() {}");
        let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
        // simulate a torn (uncommitted) generation
        index.store().begin_generation(99, "torn").unwrap();
        let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
        daemon.bootstrap().unwrap();
        // recovery dropped the torn gen; committed generation is the real one
        assert!(index.store().last_committed_generation().unwrap() >= 1);
    }
}
