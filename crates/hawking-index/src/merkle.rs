//! BLAKE3 merkle-DAG over the workspace tree (bible §4.8).
//!
//! - Leaf hash  = `BLAKE3(file_bytes)`.
//! - Dir node   = `BLAKE3( sorted [ (name, type, child_hash) ] )` (canonical, order-independent).
//! - O(changed) diff: compare roots, recurse only into differing subtrees.
//! - Rename detection: a deleted leaf + an added leaf with the *same* content hash in
//!   one changeset is reported as a rename, so the symbol graph remaps instead of
//!   delete+reinsert (and unchanged content is never re-parsed/re-embedded).
//!
//! The merkle tree is the correctness backstop the watcher (§4.9) leans on: the
//! file-system event is only a hint; the merkle diff decides the actual work.

use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A node in the merkle-DAG: a file leaf or a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleNode {
    pub path: PathBuf,
    /// Hex-encoded BLAKE3 hash (32 bytes → 64 hex chars).
    pub hash: String,
    pub kind: MerkleKind,
    pub size_bytes: u64,
    /// Child nodes for a directory (sorted by `path` for canonical hashing).
    /// Empty for files.
    #[serde(default)]
    pub children: Vec<MerkleNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MerkleKind {
    File,
    Directory,
    Symlink,
    Missing,
}

/// The output of an O(changed) merkle diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChangeSet {
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    /// `(old_path, new_path)` pairs detected by identical leaf hash.
    pub renamed: Vec<(PathBuf, PathBuf)>,
}

impl ChangeSet {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.modified.is_empty()
            && self.removed.is_empty()
            && self.renamed.is_empty()
    }

    /// Every path whose *content* needs (re)parsing: adds + modifies + rename
    /// destinations. (Rename sources keep their content; the index just remaps.)
    pub fn dirty_paths(&self) -> Vec<PathBuf> {
        let mut out = self.added.clone();
        out.extend(self.modified.iter().cloned());
        out.extend(self.renamed.iter().map(|(_, new)| new.clone()));
        out
    }
}

/// Scans a workspace into a merkle-DAG and diffs two snapshots.
pub trait MerkleScanner: Send + Sync {
    fn scan_workspace(&self) -> hide_core::Result<MerkleNode>;
    fn diff(&self, old: &MerkleNode, new: &MerkleNode) -> hide_core::Result<ChangeSet>;
}

/// A real BLAKE3 merkle scanner over the file system.
///
/// Honors `.gitignore` (ripgrep's `ignore` crate) plus a built-in noise set
/// (`.git`, `target`, `node_modules`, …) so generated/vendored churn never
/// enters the index (bible §4.9, §6).
pub struct Blake3MerkleScanner {
    root: PathBuf,
    respect_gitignore: bool,
}

impl Blake3MerkleScanner {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            respect_gitignore: true,
        }
    }

    pub fn without_gitignore(mut self) -> Self {
        self.respect_gitignore = false;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Built-in ignore set: directories that pollute a code index with machine noise.
const BUILTIN_IGNORE_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "target",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".hide",
];

fn hash_file(path: &Path) -> std::io::Result<(String, u64)> {
    let bytes = std::fs::read(path)?;
    let hash = blake3::hash(&bytes);
    Ok((hash.to_hex().to_string(), bytes.len() as u64))
}

/// Directory node hash = BLAKE3 over the canonical, sorted child digest list.
fn hash_dir(children: &[MerkleNode]) -> String {
    let mut hasher = blake3::Hasher::new();
    // `children` is sorted by caller; encode (name, type, child_hash) per entry.
    for child in children {
        let name = child
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let type_tag: u8 = match child.kind {
            MerkleKind::File => 0,
            MerkleKind::Directory => 1,
            MerkleKind::Symlink => 2,
            MerkleKind::Missing => 3,
        };
        hasher.update(&(name.len() as u32).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&[type_tag]);
        hasher.update(child.hash.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

impl MerkleScanner for Blake3MerkleScanner {
    fn scan_workspace(&self) -> hide_core::Result<MerkleNode> {
        // Walk all non-ignored files, hashing leaves, then fold bottom-up into
        // directory nodes. `ignore`'s parallel walker yields files (not dirs),
        // so we collect leaves keyed by parent dir then build the tree.
        let mut leaves: BTreeMap<PathBuf, MerkleNode> = BTreeMap::new();

        let mut builder = WalkBuilder::new(&self.root);
        builder
            .hidden(false)
            .git_ignore(self.respect_gitignore)
            .git_global(self.respect_gitignore)
            .git_exclude(self.respect_gitignore)
            .parents(self.respect_gitignore)
            .filter_entry(|entry| {
                if let Some(name) = entry.file_name().to_str() {
                    if entry.file_type().is_some_and(|ft| ft.is_dir())
                        && BUILTIN_IGNORE_DIRS.contains(&name)
                    {
                        return false;
                    }
                }
                true
            });

        for result in builder.build() {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue, // skip unreadable entries; merkle is best-effort over readable tree
            };
            let ft = match entry.file_type() {
                Some(ft) => ft,
                None => continue,
            };
            if !ft.is_file() {
                continue;
            }
            let path = entry.path().to_path_buf();
            let (hash, size) = match hash_file(&path) {
                Ok(v) => v,
                Err(_) => continue,
            };
            leaves.insert(
                path.clone(),
                MerkleNode {
                    path,
                    hash,
                    kind: MerkleKind::File,
                    size_bytes: size,
                    children: Vec::new(),
                },
            );
        }

        Ok(build_tree(&self.root, &leaves))
    }

    fn diff(&self, old: &MerkleNode, new: &MerkleNode) -> hide_core::Result<ChangeSet> {
        Ok(diff_nodes(old, new))
    }
}

/// Fold a flat leaf map into a directory tree rooted at `root`.
fn build_tree(root: &Path, leaves: &BTreeMap<PathBuf, MerkleNode>) -> MerkleNode {
    // Group leaves by their parent directory chain. We materialize directory
    // nodes lazily: collect direct children per dir, recurse.
    fn children_of(dir: &Path, leaves: &BTreeMap<PathBuf, MerkleNode>) -> Vec<MerkleNode> {
        // Direct child dirs and files of `dir`.
        let mut child_dirs: BTreeMap<PathBuf, ()> = BTreeMap::new();
        let mut child_files: Vec<MerkleNode> = Vec::new();
        for (path, node) in leaves.range(dir.to_path_buf()..) {
            if !path.starts_with(dir) || path == dir {
                if path > &dir.join("\u{10ffff}") {
                    break;
                }
                continue;
            }
            // relative component after `dir`
            let rel = match path.strip_prefix(dir) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let mut comps = rel.components();
            let first = match comps.next() {
                Some(c) => c.as_os_str(),
                None => continue,
            };
            if comps.next().is_none() {
                // direct file child
                child_files.push(node.clone());
            } else {
                // belongs to a subdirectory
                child_dirs.insert(dir.join(first), ());
            }
        }

        let mut children: Vec<MerkleNode> = child_files;
        for (sub, _) in child_dirs {
            let sub_children = children_of(&sub, leaves);
            let size: u64 = sub_children.iter().map(|c| c.size_bytes).sum();
            let hash = hash_dir(&sub_children);
            children.push(MerkleNode {
                path: sub,
                hash,
                kind: MerkleKind::Directory,
                size_bytes: size,
                children: sub_children,
            });
        }
        children.sort_by(|a, b| a.path.cmp(&b.path));
        children
    }

    let children = children_of(root, leaves);
    let size: u64 = children.iter().map(|c| c.size_bytes).sum();
    let hash = hash_dir(&children);
    MerkleNode {
        path: root.to_path_buf(),
        hash,
        kind: MerkleKind::Directory,
        size_bytes: size,
        children,
    }
}

/// O(changed) recursive diff between two merkle nodes.
fn diff_nodes(old: &MerkleNode, new: &MerkleNode) -> ChangeSet {
    let mut cs = ChangeSet::default();
    diff_into(old, new, &mut cs);
    detect_renames(&mut cs, old, new);
    cs
}

fn diff_into(old: &MerkleNode, new: &MerkleNode, cs: &mut ChangeSet) {
    // Equal subtrees prune immediately — the whole point of merkle diffing.
    if old.hash == new.hash {
        return;
    }
    match (old.kind, new.kind) {
        (MerkleKind::Directory, MerkleKind::Directory) => {
            let old_children: BTreeMap<&Path, &MerkleNode> =
                old.children.iter().map(|c| (c.path.as_path(), c)).collect();
            let new_children: BTreeMap<&Path, &MerkleNode> =
                new.children.iter().map(|c| (c.path.as_path(), c)).collect();

            for (path, oc) in &old_children {
                match new_children.get(path) {
                    Some(nc) => diff_into(oc, nc, cs),
                    None => collect_removed(oc, cs),
                }
            }
            for (path, nc) in &new_children {
                if !old_children.contains_key(path) {
                    collect_added(nc, cs);
                }
            }
        }
        (MerkleKind::File, MerkleKind::File) => {
            // Same path, different hash → modified.
            cs.modified.push(new.path.clone());
        }
        // Type change at a path (file↔dir): treat as remove old + add new.
        _ => {
            collect_removed(old, cs);
            collect_added(new, cs);
        }
    }
}

fn collect_added(node: &MerkleNode, cs: &mut ChangeSet) {
    match node.kind {
        MerkleKind::File => cs.added.push(node.path.clone()),
        MerkleKind::Directory => {
            for c in &node.children {
                collect_added(c, cs);
            }
        }
        _ => {}
    }
}

fn collect_removed(node: &MerkleNode, cs: &mut ChangeSet) {
    match node.kind {
        MerkleKind::File => cs.removed.push(node.path.clone()),
        MerkleKind::Directory => {
            for c in &node.children {
                collect_removed(c, cs);
            }
        }
        _ => {}
    }
}

/// Pair `removed`+`added` files with identical leaf hash into renames.
fn detect_renames(cs: &mut ChangeSet, old: &MerkleNode, new: &MerkleNode) {
    if cs.added.is_empty() || cs.removed.is_empty() {
        return;
    }
    let old_hashes = leaf_hash_map(old);
    let new_hashes = leaf_hash_map(new);

    // hash → removed paths with that hash
    let mut removed_by_hash: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for p in &cs.removed {
        if let Some(h) = old_hashes.get(p) {
            removed_by_hash.entry(h.clone()).or_default().push(p.clone());
        }
    }

    let mut consumed_removed: Vec<PathBuf> = Vec::new();
    let mut consumed_added: Vec<PathBuf> = Vec::new();
    let mut renamed: Vec<(PathBuf, PathBuf)> = Vec::new();

    for added_path in &cs.added {
        if let Some(h) = new_hashes.get(added_path) {
            if let Some(candidates) = removed_by_hash.get_mut(h) {
                // pick a not-yet-consumed source with the same content hash
                if let Some(pos) = candidates
                    .iter()
                    .position(|p| !consumed_removed.contains(p))
                {
                    let old_path = candidates.remove(pos);
                    consumed_removed.push(old_path.clone());
                    consumed_added.push(added_path.clone());
                    renamed.push((old_path, added_path.clone()));
                }
            }
        }
    }

    cs.removed.retain(|p| !consumed_removed.contains(p));
    cs.added.retain(|p| !consumed_added.contains(p));
    cs.renamed.extend(renamed);
}

fn leaf_hash_map(node: &MerkleNode) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    fn walk(node: &MerkleNode, out: &mut BTreeMap<PathBuf, String>) {
        match node.kind {
            MerkleKind::File => {
                out.insert(node.path.clone(), node.hash.clone());
            }
            MerkleKind::Directory => {
                for c in &node.children {
                    walk(c, out);
                }
            }
            _ => {}
        }
    }
    walk(node, &mut out);
    out
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
    fn identical_trees_have_equal_root_and_empty_diff() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "src/a.rs", "fn a() {}");
        write(tmp.path(), "src/b.rs", "fn b() {}");
        let scanner = Blake3MerkleScanner::new(tmp.path());
        let t1 = scanner.scan_workspace().unwrap();
        let t2 = scanner.scan_workspace().unwrap();
        assert_eq!(t1.hash, t2.hash);
        let cs = scanner.diff(&t1, &t2).unwrap();
        assert!(cs.is_empty(), "no change → empty changeset, got {cs:?}");
    }

    #[test]
    fn detects_add_modify_delete() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "a.rs", "fn a() {}");
        write(tmp.path(), "b.rs", "fn b() {}");
        let scanner = Blake3MerkleScanner::new(tmp.path());
        let before = scanner.scan_workspace().unwrap();

        write(tmp.path(), "a.rs", "fn a() { changed(); }"); // modify
        fs::remove_file(tmp.path().join("b.rs")).unwrap(); // delete
        write(tmp.path(), "c.rs", "fn c() {}"); // add
        let after = scanner.scan_workspace().unwrap();

        let cs = scanner.diff(&before, &after).unwrap();
        assert!(cs.modified.iter().any(|p| p.ends_with("a.rs")));
        assert!(cs.removed.iter().any(|p| p.ends_with("b.rs")));
        assert!(cs.added.iter().any(|p| p.ends_with("c.rs")));
        assert!(cs.renamed.is_empty());
    }

    #[test]
    fn detects_rename_by_identical_hash() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "old_name.rs", "fn unchanged_content() {}");
        let scanner = Blake3MerkleScanner::new(tmp.path());
        let before = scanner.scan_workspace().unwrap();

        fs::rename(
            tmp.path().join("old_name.rs"),
            tmp.path().join("new_name.rs"),
        )
        .unwrap();
        let after = scanner.scan_workspace().unwrap();

        let cs = scanner.diff(&before, &after).unwrap();
        assert_eq!(cs.renamed.len(), 1, "expected one rename, got {cs:?}");
        let (old, new) = &cs.renamed[0];
        assert!(old.ends_with("old_name.rs"));
        assert!(new.ends_with("new_name.rs"));
        assert!(cs.added.is_empty() && cs.removed.is_empty());
    }

    #[test]
    fn diff_is_pruned_for_unchanged_subtrees() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "untouched/x.rs", "fn x() {}");
        write(tmp.path(), "touched/y.rs", "fn y() {}");
        let scanner = Blake3MerkleScanner::new(tmp.path());
        let before = scanner.scan_workspace().unwrap();
        write(tmp.path(), "touched/y.rs", "fn y() { more(); }");
        let after = scanner.scan_workspace().unwrap();

        // The 'untouched' subtree hash must be unchanged across snapshots.
        let find = |t: &MerkleNode, name: &str| -> String {
            t.children
                .iter()
                .find(|c| c.path.ends_with(name))
                .map(|c| c.hash.clone())
                .unwrap()
        };
        assert_eq!(find(&before, "untouched"), find(&after, "untouched"));
        assert_ne!(find(&before, "touched"), find(&after, "touched"));
    }
}
